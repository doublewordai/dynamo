// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered reasoning, text, and tool-call parsing through one `dynamo-parsers-v2`
//! unified parser.
//!
//! The split path chains two independent parsers: the reasoning parser strips
//! `<think>...</think>` into `reasoning_content` and the tool-call jail (or the v2
//! tool parser) scans whatever is left. DeepSeek V4.1 ships no reasoning parser in
//! `dynamo-parsers` and its DSML tool grammar (`<｜DSML｜ calls>` blocks with
//! leading-space tag names) is only implemented as a unified parser in
//! `dynamo-parsers-v2`, so the family is served here: ONE [`UnifiedParser`] per
//! response choice owns the whole grammar, emits events in the order the model
//! produced them, and this module maps those events onto the OpenAI streaming and
//! batch wire shapes.
//!
//! Every [`UnifiedParserEvent`] becomes its own chunk: a stream delta carries
//! `content`, `reasoning_content` and `tool_calls` side by side with no way to say
//! which came first, so packing one `push` into one delta would lose the ordering
//! this path exists to keep.

use std::collections::{HashMap, HashSet};

use async_stream::stream;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, ToolCallDelta, UnifiedEvent, UnifiedParser,
    UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, UnifiedToolOutputMode, create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption, FinishReason, FunctionCall, FunctionCallStream, FunctionType,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use uuid::Uuid;

use super::NvCreateChatCompletionStreamResponse;

/// The `dynamo-parsers-v2` unified family that serves DeepSeek V4.1. It is also the
/// `--dyn-tool-call-parser` / `--dyn-reasoning-parser` name; both must be set to it.
pub const DEEPSEEK_V41_UNIFIED_FAMILY: &str = "deepseek_v41";

/// The `dynamo-parsers-v2` unified family that serves Muse Glimmer: reasoning,
/// content and ATEM tool calls are channels of one recipient-routed grammar, so no
/// split reasoning/tool parser pair can serve it.
pub const MUSE_GLIMMER_UNIFIED_FAMILY: &str = "muse_glimmer";

pub use super::hunyuan_parser::HUNYUAN_UNIFIED_FAMILY;
pub use super::mimo_parser::MIMO_UNIFIED_FAMILY;

/// Every `--dyn-tool-call-parser` / `--dyn-reasoning-parser` name a unified family
/// answers to, paired with the family it selects. A name is here because the legacy
/// `dynamo-parsers` registries do not know it, so the worker's flag validation (which
/// appends this list) would otherwise reject it. Aliases carry the engine's own
/// spelling for the same format, so one catalogue entry can use either.
const UNIFIED_FAMILY_NAMES: &[(&str, &str)] = &[
    (DEEPSEEK_V41_UNIFIED_FAMILY, DEEPSEEK_V41_UNIFIED_FAMILY),
    (MUSE_GLIMMER_UNIFIED_FAMILY, MUSE_GLIMMER_UNIFIED_FAMILY),
    ("muse", MUSE_GLIMMER_UNIFIED_FAMILY),
    (HUNYUAN_UNIFIED_FAMILY, HUNYUAN_UNIFIED_FAMILY),
    ("hy3", HUNYUAN_UNIFIED_FAMILY),
    (MIMO_UNIFIED_FAMILY, MIMO_UNIFIED_FAMILY),
    ("mimo_v2", MIMO_UNIFIED_FAMILY),
];

/// Parser names served only by a unified parser, for the worker's flag validation.
pub static UNIFIED_PARSER_NAMES: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| UNIFIED_FAMILY_NAMES.iter().map(|(name, _)| *name).collect());

/// The unified family `name` selects, if any.
pub fn unified_family(name: &str) -> Option<&'static str> {
    UNIFIED_FAMILY_NAMES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, family)| *family)
}

/// The name every per-parser check should compare against: a unified family alias
/// resolves to its family, and any other name is returned unchanged.
pub fn canonical_parser_name(name: &str) -> &str {
    unified_family(name).unwrap_or(name)
}

/// Whether a tool-call / reasoning parser pair is servable: a unified family owns
/// both channels, so naming one on either side requires a name for the same family
/// on the other.
pub fn is_valid_parser_pair(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> bool {
    let unified = |parser: Option<&str>| parser.and_then(unified_family);
    unified(tool_call_parser) == unified(reasoning_parser)
}

/// The message for a pair [`is_valid_parser_pair`] rejects.
pub fn invalid_parser_pair_message(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> String {
    let family = tool_call_parser
        .and_then(unified_family)
        .or_else(|| reasoning_parser.and_then(unified_family))
        .unwrap_or_default();
    format!("{family} requires both tool_call_parser and reasoning_parser to name {family}")
}

/// Both parser names must agree because a unified parser owns reasoning and tool calls.
pub(crate) fn selected_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    let family = unified_family(tool_call_parser?)?;
    (reasoning_parser.and_then(unified_family) == Some(family)).then_some(family)
}

/// Build the parser for `family`: in-tree families first, then `dynamo-parsers-v2`.
fn create_parser(family: &str, tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> {
    match family {
        HUNYUAN_UNIFIED_FAMILY => super::hunyuan_parser::hunyuan_unified(tools),
        MIMO_UNIFIED_FAMILY => super::mimo_parser::mimo_unified(tools),
        _ => create_unified_parser_for_family(family, tools),
    }
}

/// Whether the rendered prompt left generation inside an open thought.
pub(crate) fn prompt_opens_reasoning(parser: &str, prompt: &str) -> Option<bool> {
    match canonical_parser_name(parser) {
        HUNYUAN_UNIFIED_FAMILY => Some(super::hunyuan_parser::prompt_opens_reasoning(prompt)),
        MIMO_UNIFIED_FAMILY => Some(super::mimo_parser::prompt_opens_reasoning(prompt)),
        _ => None,
    }
}

/// Which channel the rendered generation prompt already opened, for the streaming path.
///
/// `prompt_injected_reasoning` is the per-request fact the preprocessor already
/// computes: the rendered prompt ended with `<think>`, so generated output starts
/// inside a thought the model will close without ever emitting the opener. Otherwise
/// the V4.1 prompt ends with `</think>` (thinking off) and the model is already in
/// the visible response channel.
pub(crate) fn stream_prefill(
    family: &str,
    prompt_injected_reasoning: bool,
) -> UnifiedParserStartingState {
    if prompt_injected_reasoning {
        return UnifiedParserStartingState::Reasoning;
    }
    match family {
        DEEPSEEK_V41_UNIFIED_FAMILY => UnifiedParserStartingState::Response,
        _ => UnifiedParserStartingState::None,
    }
}

/// Resolve a `Reasoning`-prefill choice's ACTUAL starting state when guided JSON is
/// installed, from the shape of that choice's first content-bearing chunk.
///
/// Guided decoding forbids the model from emitting the reasoning closer, so a request
/// whose rendered prompt opened reasoning would, if started unconditionally at
/// `Reasoning`, have its entire guided JSON payload swallowed as `reasoning_content`.
/// A payload whose first non-whitespace byte is `[` or `{` is bare guided JSON with
/// no reasoning to split out; anything else is a real reasoning block that will close
/// normally. An empty first chunk is inconclusive and keeps the `Reasoning` default.
fn bare_guided_json_prefill(
    first_content: Option<&ChatCompletionMessageContent>,
) -> UnifiedParserStartingState {
    let Some(ChatCompletionMessageContent::Text(text)) = first_content else {
        return UnifiedParserStartingState::Reasoning;
    };
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return UnifiedParserStartingState::Reasoning;
    }
    match trimmed.as_bytes()[0] {
        b'[' | b'{' => UnifiedParserStartingState::None,
        _ => UnifiedParserStartingState::Reasoning,
    }
}

/// Which channel the prompt opened, inferred from complete output text.
///
/// The batch path has no prompt in hand, only what the model produced. For the
/// `<think>` families: an opener means the model opened reasoning itself; a closer
/// before any opener means the prompt had already opened it; neither marker means
/// reasoning never ran. Muse Glimmer names every channel in a generated header, so
/// nothing is inferred for it.
fn detect_prefill(family: &str, content: &str) -> UnifiedParserStartingState {
    match family {
        HUNYUAN_UNIFIED_FAMILY => return super::hunyuan_parser::detect_starting_state(content),
        MIMO_UNIFIED_FAMILY => return super::mimo_parser::detect_starting_state(content),
        MUSE_GLIMMER_UNIFIED_FAMILY => return UnifiedParserStartingState::None,
        _ => {}
    }
    let opener = content.find("<think>");
    let closer = content.find("</think>");
    match (opener, closer) {
        (None, Some(_)) => UnifiedParserStartingState::Reasoning,
        (Some(open_at), Some(close_at)) if close_at < open_at => {
            UnifiedParserStartingState::Reasoning
        }
        (Some(_), _) => UnifiedParserStartingState::None,
        (None, None) => UnifiedParserStartingState::Response,
    }
}

/// Map dynamo's v1 [`ToolDefinition`]s onto the v2 parser's [`Tool`] shape.
fn to_v2_tools(tools: Option<&[ToolDefinition]>) -> Vec<Tool> {
    tools
        .unwrap_or(&[])
        .iter()
        .map(|tool| Tool {
            name: tool.name.clone(),
            description: None,
            parameters: tool.parameters.clone().unwrap_or(serde_json::Value::Null),
            strict: tool.strict,
        })
        .collect()
}

/// Map the request's `tool_choice` onto the wire format the backend will produce.
///
/// A named or `required` choice is served by guided decoding, which constrains the
/// model to bare JSON instead of native DSML markup: a named choice to that one
/// tool's argument object, `required` to a call object or an array of them. Unset,
/// `auto` and `none` leave the model in native markup, and so does a structural tag,
/// which constrains generation to the native grammar no matter what `tool_choice`
/// says.
pub(crate) fn tool_output_mode(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
) -> UnifiedToolOutputMode {
    if uses_tool_call_structural_tag {
        return UnifiedToolOutputMode::Native;
    }
    match tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(named)) => UnifiedToolOutputMode::GuidedJson {
            named_tool: Some(named.function.name.clone()),
        },
        Some(ChatCompletionToolChoiceOption::Required) => {
            UnifiedToolOutputMode::GuidedJson { named_tool: None }
        }
        None
        | Some(ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None) => {
            UnifiedToolOutputMode::Native
        }
    }
}

/// Decode a guided-decoding payload: the named tool's bare argument object, or one
/// `{"name", "arguments" | "parameters"}` object or an array of them.
pub(crate) fn guided_json_calls(
    payload: &str,
    named_tool: Option<&str>,
) -> Option<Vec<ToolCallDelta>> {
    let value: serde_json::Value = serde_json::from_str(payload.trim()).ok()?;
    let delta = |tool_index: usize, name: &str, arguments: &serde_json::Value| {
        arguments.is_object().then(|| ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: arguments.to_string(),
            complete: true,
        })
    };
    if let Some(name) = named_tool {
        return Some(vec![delta(0, name, &value)?]);
    }
    let items = match value {
        serde_json::Value::Array(items) => items,
        object @ serde_json::Value::Object(_) => vec![object],
        _ => return None,
    };
    if items.is_empty() {
        return None;
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let name = item.get("name")?.as_str()?;
            let arguments = match (item.get("arguments"), item.get("parameters")) {
                (Some(arguments), None) | (None, Some(arguments)) => arguments,
                _ => return None,
            };
            delta(index, name, arguments)
        })
        .collect()
}

/// Merge adjacent same-kind text/reasoning deltas so one `push` does not become three
/// chunks that say the same thing. Tool-call deltas never merge: the name-carrying
/// first delta of a call must stay distinct from its argument-only successors.
fn coalesce(deltas: Vec<UnifiedParserEvent>) -> Vec<UnifiedParserEvent> {
    let mut out: Vec<UnifiedParserEvent> = Vec::with_capacity(deltas.len());
    for delta in deltas {
        match (out.last_mut(), delta) {
            (Some(UnifiedParserEvent::Text(prev)), UnifiedParserEvent::Text(text)) => {
                prev.push_str(&text)
            }
            (Some(UnifiedParserEvent::Reasoning(prev)), UnifiedParserEvent::Reasoning(text)) => {
                prev.push_str(&text)
            }
            (_, delta) => out.push(delta),
        }
    }
    out
}

/// An empty streaming choice for `index`, used as the base every emitted chunk is
/// filled in from. `logprobs` is dropped on purpose: once parsing rewrites a choice
/// the emitted text no longer lines up token-for-token with the backend's stream.
fn empty_choice(index: u32) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    }
}

/// Pick the one fan-out child that keeps the source chunk's token metrics: the
/// reasoning child when there is one, so the reasoning-usage estimator downstream
/// still attributes the chunk's tokens to reasoning; otherwise the last child.
fn fanout_llm_metrics_position(choices: &[ChatChoiceStream]) -> Option<usize> {
    choices
        .iter()
        .position(|choice| choice.delta.reasoning_content.is_some())
        .or_else(|| choices.len().checked_sub(1))
}

/// Per-choice streaming state: one parser instance plus the bookkeeping the OpenAI
/// streaming tool-call contract needs. One instance parses exactly one choice of one
/// request.
struct ChoiceState {
    family: String,
    parser: Box<dyn UnifiedParser>,
    /// Tool indices whose opening chunk (id + type + name) has already gone out.
    opened_calls: HashSet<usize>,
    /// Whether any tool-call chunk was emitted; flips a terminal `Stop` to `ToolCalls`.
    tool_emitted: bool,
    /// Parsing failed. The stream adapter turns this into a terminal error so a
    /// malformed DSML call is never served as a successful assistant message.
    failed: bool,
}

impl ChoiceState {
    fn new(
        family: &str,
        tools: &[Tool],
        prefill: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> anyhow::Result<Self> {
        let mut parser = create_parser(family, tools)?;
        // `prompt_token_ids` stays empty: the starting state comes from the rendered
        // prompt text, which the preprocessor has already consumed by this point.
        // Guided calls buffer to completion and surface a malformed payload as text
        // (`RecoverAsText`); nothing on this path commits a call before its payload
        // closes.
        parser.initialize_request(UnifiedParserInit {
            prompt_token_ids: Vec::new(),
            starting_state: prefill,
            tool_output_mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
        })?;
        Ok(Self {
            family: family.to_string(),
            parser,
            opened_calls: HashSet::new(),
            tool_emitted: false,
            failed: false,
        })
    }

    /// Feed one decoded text delta through the parser.
    fn push(&mut self, text: &str) -> Vec<UnifiedParserEvent> {
        if self.failed {
            return text_delta(text.to_string());
        }
        let mut output = UnifiedParserOutput::default();
        match self.parser.parse_into(text, &mut output) {
            Ok(()) => output.events,
            Err(error) => {
                tracing::warn!(error = %error, family = self.family, "unified parser push failed");
                output.events.extend(self.give_up(text));
                output.events
            }
        }
    }

    /// Flush buffered partial state at end of stream.
    fn finish(&mut self) -> Vec<UnifiedParserEvent> {
        if self.failed {
            return Vec::new();
        }
        match self.parser.finish() {
            Ok(output) => output.events,
            Err(error) => {
                tracing::warn!(error = %error, family = self.family, "unified parser finish failed");
                self.give_up("")
            }
        }
    }

    /// Stop using the parser and surface whatever it was holding as visible text so
    /// the client sees a truncated answer rather than silently losing bytes.
    fn give_up(&mut self, fallback: &str) -> Vec<UnifiedParserEvent> {
        self.failed = true;
        let recovered = self.parser.reset();
        if recovered.is_empty() {
            text_delta(fallback.to_string())
        } else {
            text_delta(recovered)
        }
    }

    /// Convert one ordered delta into a streaming choice for `index`.
    fn delta_to_choice(&mut self, index: u32, delta: UnifiedParserEvent) -> ChatChoiceStream {
        let mut choice = empty_choice(index);
        match delta {
            UnifiedParserEvent::Text(text) => {
                choice.delta.content = Some(ChatCompletionMessageContent::Text(text));
            }
            UnifiedParserEvent::Reasoning(text) => {
                choice.delta.reasoning_content = Some(text);
            }
            UnifiedParserEvent::ToolCall(call) => {
                self.tool_emitted = true;
                // OpenAI streaming contract: the FIRST chunk for a tool index carries
                // id + type + name, later chunks only argument fragments. The parser
                // mints no ids, so one is minted here per call, exactly once.
                let first = self.opened_calls.insert(call.tool_index);
                choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                    index: call.tool_index as u32,
                    id: first.then(|| format!("call-{}", Uuid::new_v4())),
                    r#type: first.then_some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: first.then_some(call.name).flatten(),
                        arguments: Some(call.arguments),
                    }),
                }]);
            }
        }
        choice
    }

    /// Convert an ordered delta run into the streaming choices it becomes.
    ///
    /// `role` / `refusal` ride on the first emitted choice and the terminating
    /// `finish_reason` on the last, so a client reassembling the stream sees the same
    /// envelope it would have without this path.
    fn choices_for(
        &mut self,
        original: &ChatChoiceStream,
        deltas: Vec<UnifiedParserEvent>,
        finish_reason: Option<FinishReason>,
    ) -> Vec<ChatChoiceStream> {
        let deltas = coalesce(deltas);
        let index = original.index;
        let count = deltas.len();
        let mut choices = Vec::with_capacity(count.max(1));

        for (position, delta) in deltas.into_iter().enumerate() {
            let mut choice = self.delta_to_choice(index, delta);
            if position == 0 {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
                choice.delta.function_call = original.delta.function_call.clone();
                if let Some(reasoning) = original.delta.reasoning_content.as_deref() {
                    choice
                        .delta
                        .reasoning_content
                        .get_or_insert_default()
                        .insert_str(0, reasoning);
                }
                if let Some(mut original_calls) = original.delta.tool_calls.clone() {
                    if !original_calls.is_empty() {
                        self.tool_emitted = true;
                    }
                    if let Some(parsed_calls) = choice.delta.tool_calls.take() {
                        original_calls.extend(parsed_calls);
                    }
                    choice.delta.tool_calls = Some(original_calls);
                }
            }
            if position + 1 == count {
                choice.finish_reason = self.normalize_finish_reason(finish_reason);
            }
            choices.push(choice);
        }

        // The parser produced nothing, but the chunk still carried envelope state that
        // has to reach the client (the opening role chunk, a refusal, or the terminal
        // finish_reason).
        if choices.is_empty()
            && (original.delta.role.is_some()
                || original.delta.refusal.is_some()
                || original.delta.reasoning_content.is_some()
                || original.delta.tool_calls.is_some()
                || original.delta.function_call.is_some()
                || finish_reason.is_some())
        {
            let mut choice = empty_choice(index);
            choice.delta.role = original.delta.role;
            choice.delta.refusal = original.delta.refusal.clone();
            choice.delta.reasoning_content = original.delta.reasoning_content.clone();
            choice.delta.tool_calls = original.delta.tool_calls.clone();
            choice.delta.function_call = original.delta.function_call.clone();
            if choice
                .delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
            {
                self.tool_emitted = true;
            }
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
            choices.push(choice);
        }

        choices
    }

    /// OpenAI streaming contract: once a choice has emitted tool calls, a `Stop`
    /// terminating reason must be reported as `ToolCalls`. `Length` /
    /// `ContentFilter` describe why generation stopped and are preserved as-is.
    fn normalize_finish_reason(&self, finish_reason: Option<FinishReason>) -> Option<FinishReason> {
        if finish_reason == Some(FinishReason::Stop) && self.tool_emitted {
            Some(FinishReason::ToolCalls)
        } else {
            finish_reason
        }
    }
}

/// One text delta, or nothing at all when the text is empty.
fn text_delta(text: String) -> Vec<UnifiedParserEvent> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![UnifiedParserEvent::Text(text)]
    }
}

/// The aggregated result of parsing one complete (non-streaming) output.
pub(crate) struct CompleteOutput {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
}

/// Batch (non-streaming) path: run the whole output through the same parser lifecycle
/// and fold the assembled events into the final message.
///
/// Reasoning spans are concatenated because the non-streaming message schema has ONE
/// `reasoning_content` string. Only native markup is decoded here: a request served
/// by guided JSON has its calls parsed on the streaming path, which is what feeds the
/// aggregator on this frontend.
pub(crate) fn parse_complete(
    family: &str,
    content: &str,
    tool_definitions: &[ToolDefinition],
) -> anyhow::Result<CompleteOutput> {
    let tools = to_v2_tools(Some(tool_definitions));
    let mut parser = create_parser(family, &tools)?;
    parser.initialize_request(UnifiedParserInit {
        starting_state: detect_prefill(family, content),
        tool_output_mode: UnifiedToolOutputMode::Native,
        ..UnifiedParserInit::default()
    })?;

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for event in parser.parse_complete(content)? {
        match event {
            UnifiedEvent::Text { text: chunk } => text.push_str(&chunk),
            UnifiedEvent::Reasoning { text: chunk } => reasoning.push_str(&chunk),
            UnifiedEvent::ToolCall { name, arguments } => {
                tool_calls.push(ChatCompletionMessageToolCall {
                    id: format!("call-{}", Uuid::new_v4()),
                    r#type: FunctionType::Function,
                    // `assemble` already parsed the argument fragments into a typed
                    // object, so this re-serializes rather than passing the model's
                    // bytes through.
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(&arguments)?,
                    },
                });
            }
        }
    }

    Ok(CompleteOutput {
        text,
        reasoning,
        tool_calls,
    })
}

/// Per-choice bookkeeping that outlives any single `ChoiceState` for that index: an
/// already-parsed chunk (structured `tool_calls` / `reasoning_content` from upstream)
/// drops the live parser, but whether the choice ever emitted a tool call and whether
/// its terminal chunk has gone out must still be known afterwards.
#[derive(Default)]
struct ChoiceRecord {
    tool_emitted: bool,
    finished: bool,
    /// An already-parsed chunk interrupted this choice. Structured output only
    /// appears after any reasoning phase concluded, so a raw run resuming after it
    /// must start at `Response`, never at the request-level prefill.
    detoured: bool,
}

/// Finish every choice that never received a terminating chunk, in index order.
///
/// A tool-emitting choice must terminate with `ToolCalls` even when the backend never
/// sent a finish_reason, or a strict client waits forever; a text-only choice gets no
/// synthetic chunk.
fn finish_unterminated_choices(
    states: &mut HashMap<u32, ChoiceState>,
    records: &mut HashMap<u32, ChoiceRecord>,
) -> Vec<ChatChoiceStream> {
    let mut indices: Vec<u32> = states
        .keys()
        .copied()
        .chain(records.keys().copied())
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let mut choices = Vec::new();
    for index in indices {
        if records.get(&index).is_some_and(|record| record.finished) {
            continue;
        }
        records.entry(index).or_default().finished = true;
        let base = empty_choice(index);
        match states.get_mut(&index) {
            Some(state) => {
                let deltas = state.finish();
                let finish_reason = state.tool_emitted.then_some(FinishReason::ToolCalls);
                choices.extend(state.choices_for(&base, deltas, finish_reason));
            }
            None => {
                if records
                    .get(&index)
                    .is_some_and(|record| record.tool_emitted)
                {
                    let mut choice = base;
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                    choices.push(choice);
                }
            }
        }
    }
    choices
}

/// Wrap one rewritten choice in a response built from `template`. Usage, nvext and
/// metrics were cleared from the template: they belong to the chunk that carried them.
fn response_with_choice(
    template: &NvCreateChatCompletionStreamResponse,
    choice: ChatChoiceStream,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut data = template.clone();
    data.inner.choices = vec![choice];
    Annotated::from_data(data)
}

/// Whether a choice arrived already parsed by something upstream and must be passed
/// through untouched rather than re-parsed.
fn already_parsed(choice: &ChatChoiceStream) -> bool {
    let has_raw_text = matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Text(_))
    );
    matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Parts(_))
    ) || (!has_raw_text
        && (choice.delta.tool_calls.is_some()
            || choice.delta.function_call.is_some()
            || choice.delta.reasoning_content.is_some()))
}

const PARSE_FAILED: &str = "model output parsing failed";

/// Streaming path: one unified parser per response choice, replacing both the
/// reasoning parser and the tool-call jail for this request.
///
/// A DSML block split across deltas is buffered by the parser and emitted once its
/// invocation closes; markers quoted inside string parameters are data. A parse
/// failure ends the stream with a typed error rather than serving malformed DSML as
/// assistant content, matching the batch path.
pub(crate) fn apply_stream<S>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    prefill: UnifiedParserStartingState,
    family: &'static str,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tools = to_v2_tools(tool_definitions.as_deref());
    let mode = tool_output_mode(tool_choice.as_ref(), uses_tool_call_structural_tag);
    let guided_json = matches!(mode, UnifiedToolOutputMode::GuidedJson { .. });
    stream! {
        let mut states: HashMap<u32, ChoiceState> = HashMap::new();
        let mut records: HashMap<u32, ChoiceRecord> = HashMap::new();
        // Last data response with its choices and per-chunk fields cleared, so an
        // end-of-stream flush has an envelope (id, model, created) to attach to.
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;

        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            if response.is_error() {
                yield response;
                return;
            }
            let Some(chat) = response.data.as_mut() else {
                // Non-data annotations (comments, metadata) pass through untouched.
                yield response;
                continue;
            };

            {
                let mut next = chat.clone();
                next.inner.choices.clear();
                next.inner.usage = None;
                next.nvext = None;
                next.llm_metrics = None;
                template = Some(next);
            }

            if chat.inner.choices.is_empty() {
                // A usage-only chunk. OpenAI stream ordering requires every choice's
                // terminal finish_reason to precede it, so flush first.
                if let Some(template) = &template {
                    let choices = finish_unterminated_choices(&mut states, &mut records);
                    if states.values().any(|state| state.failed) {
                        yield Annotated::from_error(PARSE_FAILED);
                        return;
                    }
                    for choice in choices {
                        yield response_with_choice(template, choice);
                    }
                }
                yield response;
                continue;
            }

            let originals = std::mem::take(&mut chat.inner.choices);
            let mut emitted: Vec<ChatChoiceStream> = Vec::new();
            for mut original in originals {
                if already_parsed(&original) {
                    let record = records.entry(original.index).or_default();
                    record.detoured = true;
                    if original
                        .delta
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| !calls.is_empty())
                    {
                        record.tool_emitted = true;
                    }
                    if let Some(mut state) = states.remove(&original.index) {
                        if state.tool_emitted {
                            record.tool_emitted = true;
                        }
                        if original.finish_reason == Some(FinishReason::Stop) && record.tool_emitted {
                            original.finish_reason = Some(FinishReason::ToolCalls);
                        }
                        let deltas = state.finish();
                        if state.failed {
                            yield Annotated::from_error(PARSE_FAILED);
                            return;
                        }
                        emitted.extend(state.choices_for(&empty_choice(original.index), deltas, None));
                    } else if record.tool_emitted
                        && original.finish_reason == Some(FinishReason::Stop)
                    {
                        original.finish_reason = Some(FinishReason::ToolCalls);
                    }
                    if original.finish_reason.is_some() {
                        record.finished = true;
                    }
                    emitted.push(original);
                    continue;
                }

                let state = match states.entry(original.index) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let choice_prefill = if records
                            .get(&original.index)
                            .is_some_and(|record| record.detoured)
                        {
                            UnifiedParserStartingState::Response
                        } else if prefill == UnifiedParserStartingState::Reasoning && guided_json {
                            bare_guided_json_prefill(original.delta.content.as_ref())
                        } else {
                            prefill
                        };
                        match ChoiceState::new(family, &tools, choice_prefill, mode.clone()) {
                            Ok(mut state) => {
                                if records.get(&original.index).is_some_and(|record| record.tool_emitted) {
                                    state.tool_emitted = true;
                                }
                                entry.insert(state)
                            }
                            Err(error) => {
                                tracing::warn!(
                                    error = %error,
                                    family,
                                    choice = original.index,
                                    "unified parser construction failed; passing choice through"
                                );
                                emitted.push(original);
                                continue;
                            }
                        }
                    }
                };

                let mut deltas = Vec::new();
                if let Some(ChatCompletionMessageContent::Text(text)) =
                    original.delta.content.as_ref()
                {
                    deltas.extend(state.push(text));
                }
                let terminal = original.finish_reason;
                let already_finished = records
                    .get(&original.index)
                    .is_some_and(|record| record.finished);
                if terminal.is_some() {
                    if !already_finished {
                        deltas.extend(state.finish());
                    }
                    records.entry(original.index).or_default().finished = true;
                }

                if state.failed {
                    yield Annotated::from_error(PARSE_FAILED);
                    return;
                }
                let mut parsed = state.choices_for(&original, deltas, terminal);
                if parsed.is_empty() {
                    // A marker-only chunk produced no deltas. Keep it as an empty
                    // choice so the typed llm_metrics and annotation metadata it
                    // carries still reach the client.
                    parsed.push(empty_choice(original.index));
                }
                emitted.extend(parsed);
            }

            if emitted.is_empty() {
                continue;
            }

            // One upstream chunk can fan out into several. Usage, nvext and annotation
            // fields stay on the last child; token metrics stay on one reasoning child
            // when present so the reasoning-usage estimator keeps the source chunk's
            // classification without counting it twice.
            let last = emitted.len() - 1;
            let Some(llm_metrics_position) = fanout_llm_metrics_position(&emitted) else {
                continue;
            };
            for (position, choice) in emitted.into_iter().enumerate() {
                let is_last = position == last;
                let mut data = chat.clone();
                data.inner.choices = vec![choice];
                if !is_last {
                    data.inner.usage = None;
                    data.nvext = None;
                }
                if position != llm_metrics_position {
                    data.llm_metrics = None;
                }
                yield Annotated {
                    data: Some(data),
                    id: if is_last { response.id.take() } else { None },
                    event: if is_last { response.event.take() } else { None },
                    comment: if is_last { response.comment.take() } else { None },
                    error: if is_last { response.error.take() } else { None },
                };
            }
        }

        // Backstop: the stream ended without a terminating chunk for some choice.
        if let Some(template) = &template {
            let choices = finish_unterminated_choices(&mut states, &mut records);
            if states.values().any(|state| state.failed) {
                yield Annotated::from_error(PARSE_FAILED);
                return;
            }
            for choice in choices {
                yield response_with_choice(template, choice);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{
        ChatCompletionNamedToolChoice, ChatCompletionToolType, FunctionName, Role,
    };
    use futures::stream;

    const CALL_OPEN: &str = "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">";
    const CALL_PARAM: &str =
        "<｜DSML｜ parameter name=\"location\" string=\"true\">Paris</｜DSML｜ parameter>";
    const CALL_CLOSE: &str = "</｜DSML｜ invoke></｜DSML｜ calls>";

    fn chunk(
        text: &str,
        finish: Option<FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let response = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        role: Some(Role::Assistant),
                        content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: finish,
                    logprobs: None,
                }],
                created: 0,
                model: "deepseek-v4.1".to_string(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
        };
        Annotated::from_data(response)
    }

    fn usage_chunk() -> Annotated<NvCreateChatCompletionStreamResponse> {
        let mut usage = chunk("", None);
        let data = usage.data.as_mut().unwrap();
        data.inner.choices.clear();
        data.inner.usage = Some(dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });
        usage
    }

    struct Collected {
        content: String,
        reasoning: String,
        calls: Vec<(String, String)>,
        finish: Vec<FinishReason>,
        errors: Vec<String>,
        chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    }

    async fn collect(
        input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
        prefill: UnifiedParserStartingState,
    ) -> Collected {
        let out: Vec<_> = apply_stream(
            stream::iter(input),
            None,
            tool_choice,
            false,
            prefill,
            DEEPSEEK_V41_UNIFIED_FAMILY,
        )
        .collect()
        .await;
        let mut collected = Collected {
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            finish: Vec::new(),
            errors: Vec::new(),
            chunks: Vec::new(),
        };
        for annotated in &out {
            if let Some(error) = &annotated.error {
                collected.errors.push(error.to_string());
            }
            let Some(data) = annotated.data.as_ref() else {
                continue;
            };
            for choice in &data.inner.choices {
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    collected.content.push_str(text);
                }
                if let Some(reasoning) = &choice.delta.reasoning_content {
                    collected.reasoning.push_str(reasoning);
                }
                for call in choice.delta.tool_calls.iter().flatten() {
                    let function = call.function.as_ref().unwrap();
                    if let Some(name) = &function.name {
                        collected.calls.push((name.clone(), String::new()));
                    }
                    if let Some(arguments) = &function.arguments {
                        collected.calls.last_mut().unwrap().1.push_str(arguments);
                    }
                }
                if let Some(finish) = choice.finish_reason {
                    collected.finish.push(finish);
                }
            }
        }
        collected.chunks = out;
        collected
    }

    #[tokio::test]
    async fn dsml_call_split_across_chunks_parses_once_closed() {
        // The block opener, the parameter, and the closer all arrive in different
        // deltas, with the opener itself cut inside a tag name.
        let (open_a, open_b) = CALL_OPEN.split_at(CALL_OPEN.find("invoke").unwrap());
        let input = vec![
            chunk("Sure. ", None),
            chunk(open_a, None),
            chunk(open_b, None),
            chunk(CALL_PARAM, None),
            chunk(CALL_CLOSE, Some(FinishReason::Stop)),
        ];
        let out = collect(input, None, UnifiedParserStartingState::Response).await;
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        assert_eq!(out.content, "Sure. ");
        assert_eq!(out.calls.len(), 1, "{:?}", out.calls);
        assert_eq!(out.calls[0].0, "get_weather");
        let arguments: serde_json::Value = serde_json::from_str(&out.calls[0].1).unwrap();
        assert_eq!(arguments, serde_json::json!({"location": "Paris"}));
        assert_eq!(out.finish, vec![FinishReason::ToolCalls]);
    }

    #[tokio::test]
    async fn reasoning_prefill_splits_thought_from_answer_in_order() {
        let input = vec![
            chunk("Let me think.", None),
            chunk("</think>", None),
            chunk("It is 323.", Some(FinishReason::Stop)),
        ];
        let out = collect(input, None, UnifiedParserStartingState::Reasoning).await;
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        assert_eq!(out.reasoning, "Let me think.");
        assert_eq!(out.content, "It is 323.");
        assert_eq!(out.finish, vec![FinishReason::Stop]);
        // Every emitted choice carries at most one of reasoning / content so the
        // order the model produced survives on the wire.
        for annotated in &out.chunks {
            for choice in annotated.data.iter().flat_map(|data| &data.inner.choices) {
                assert!(
                    !(choice.delta.reasoning_content.is_some() && choice.delta.content.is_some())
                );
            }
        }
    }

    #[tokio::test]
    async fn no_finish_reason_flushes_before_usage_and_terminates_tool_calls() {
        let input = vec![
            chunk(CALL_OPEN, None),
            chunk(CALL_PARAM, None),
            chunk(CALL_CLOSE, None),
            usage_chunk(),
        ];
        let out = collect(input, None, UnifiedParserStartingState::Response).await;
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.finish, vec![FinishReason::ToolCalls]);
        // The usage-only chunk is last.
        let last = out.chunks.last().unwrap().data.as_ref().unwrap();
        assert!(last.inner.choices.is_empty() && last.inner.usage.is_some());
    }

    #[tokio::test]
    async fn malformed_closed_call_is_a_terminal_error() {
        let input = vec![chunk(
            concat!(
                "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">",
                "<｜DSML｜ parameter name=\"count\" string=\"false\">not-json</｜DSML｜ parameter>",
                "</｜DSML｜ invoke></｜DSML｜ calls>"
            ),
            Some(FinishReason::Stop),
        )];
        let out = collect(input, None, UnifiedParserStartingState::Response).await;
        assert_eq!(out.errors.len(), 1, "{:?}", out.errors);
        assert!(out.calls.is_empty());
    }

    #[tokio::test]
    async fn named_tool_choice_reads_guided_json() {
        let named = ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        });
        let input = vec![
            chunk("{\"location\": ", None),
            chunk("\"Paris\"}", Some(FinishReason::Stop)),
        ];
        let out = collect(input, Some(named), UnifiedParserStartingState::Response).await;
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        assert_eq!(out.calls.len(), 1, "{:?}", out.calls);
        assert_eq!(out.calls[0].0, "get_weather");
        let arguments: serde_json::Value = serde_json::from_str(&out.calls[0].1).unwrap();
        assert_eq!(arguments, serde_json::json!({"location": "Paris"}));
        assert!(out.content.is_empty());
    }

    #[test]
    fn parse_complete_recovers_reasoning_text_and_calls() {
        let text = format!("thinking</think>Here you go.{CALL_OPEN}{CALL_PARAM}{CALL_CLOSE}");
        let out = parse_complete(DEEPSEEK_V41_UNIFIED_FAMILY, &text, &[]).unwrap();
        assert_eq!(out.reasoning, "thinking");
        assert_eq!(out.text, "Here you go.");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
        let arguments: serde_json::Value =
            serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(arguments, serde_json::json!({"location": "Paris"}));
    }

    #[test]
    fn selected_family_requires_the_pair() {
        assert_eq!(
            selected_family(Some("deepseek_v41"), Some("deepseek_v41")),
            Some(DEEPSEEK_V41_UNIFIED_FAMILY)
        );
        assert_eq!(selected_family(Some("deepseek_v41"), None), None);
        assert_eq!(selected_family(None, Some("deepseek_v41")), None);
        assert_eq!(
            selected_family(Some("deepseek_v4"), Some("deepseek_v4")),
            None
        );
    }

    #[test]
    fn prompt_opened_reasoning_resolves_aliases() {
        for (name, _) in UNIFIED_FAMILY_NAMES {
            let expected = match unified_family(name).unwrap() {
                HUNYUAN_UNIFIED_FAMILY => Some(true),
                MIMO_UNIFIED_FAMILY => Some(false),
                _ => None,
            };
            assert_eq!(
                prompt_opens_reasoning(name, "…<think:opensource>"),
                expected,
                "{name}"
            );
        }
        assert_eq!(prompt_opens_reasoning("mimo_v2", "…<think>"), Some(true));
    }

    #[test]
    fn every_family_name_selects_its_family_on_both_sides() {
        for (name, family) in UNIFIED_FAMILY_NAMES {
            assert_eq!(unified_family(name), Some(*family), "{name}");
            assert_eq!(selected_family(Some(name), Some(name)), Some(*family));
            assert!(is_valid_parser_pair(Some(name), Some(family)));
            assert!(!is_valid_parser_pair(Some(name), Some("qwen3")));
            assert!(!is_valid_parser_pair(Some(name), None));
            assert!(create_parser(family, &[]).is_ok(), "{family} has no parser");
        }
        assert!(is_valid_parser_pair(Some("qwen3_coder"), Some("qwen3")));
        assert!(is_valid_parser_pair(None, None));
        assert_eq!(unified_family("qwen3"), None);
    }

    /// One Muse Glimmer turn: a thought, a tool call, and the answer, all in the
    /// recipient-routed channel grammar the pinned parser crate implements.
    const MUSE_TURN: &str = concat!(
        "<|start|>assistant to=self<|message|>Check the weather.<|eom|>",
        "<|start|>assistant to=get_weather<|message|>",
        "<atem:function_calls><atem:invoke name=\"get_weather\">",
        "<atem:parameter name=\"city\">Paris</atem:parameter>",
        "</atem:invoke></atem:function_calls><|eom|>",
        "<|start|>assistant to=user<|message|>It is sunny.<|eot|>",
    );

    #[tokio::test]
    async fn muse_glimmer_stream_splits_thought_call_and_answer() {
        for split in [12, 60, 140, 220] {
            let input = vec![
                chunk(&MUSE_TURN[..split], None),
                chunk(&MUSE_TURN[split..], Some(FinishReason::Stop)),
            ];
            let out: Vec<_> = apply_stream(
                stream::iter(input),
                None,
                None,
                false,
                stream_prefill(MUSE_GLIMMER_UNIFIED_FAMILY, false),
                MUSE_GLIMMER_UNIFIED_FAMILY,
            )
            .collect()
            .await;
            let mut content = String::new();
            let mut reasoning = String::new();
            let mut calls = Vec::new();
            for annotated in &out {
                assert!(annotated.error.is_none(), "{:?}", annotated.error);
                for choice in annotated.data.iter().flat_map(|data| &data.inner.choices) {
                    if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                        content.push_str(text);
                    }
                    if let Some(thought) = &choice.delta.reasoning_content {
                        reasoning.push_str(thought);
                    }
                    for call in choice.delta.tool_calls.iter().flatten() {
                        let function = call.function.as_ref().unwrap();
                        if let Some(name) = &function.name {
                            calls.push((name.clone(), String::new()));
                        }
                        if let Some(arguments) = &function.arguments {
                            calls.last_mut().unwrap().1.push_str(arguments);
                        }
                    }
                }
            }
            assert_eq!(reasoning, "Check the weather.", "split {split}");
            assert_eq!(content, "It is sunny.", "split {split}");
            assert_eq!(calls.len(), 1, "split {split}");
            assert_eq!(calls[0].0, "get_weather");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&calls[0].1).unwrap(),
                serde_json::json!({"city": "Paris"})
            );
        }
    }

    #[test]
    fn muse_glimmer_batch_parses_a_whole_turn() {
        let out = parse_complete(MUSE_GLIMMER_UNIFIED_FAMILY, MUSE_TURN, &[]).unwrap();
        assert_eq!(out.reasoning, "Check the weather.");
        assert_eq!(out.text, "It is sunny.");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn muse_glimmer_plain_and_unterminated_turns_survive() {
        let plain = "<|start|>assistant to=user<|message|>Just an answer.<|eot|>";
        let out = parse_complete(MUSE_GLIMMER_UNIFIED_FAMILY, plain, &[]).unwrap();
        assert_eq!(out.text, "Just an answer.");
        assert!(out.reasoning.is_empty());
        assert!(out.tool_calls.is_empty());

        let cut = "<|start|>assistant to=self<|message|>Still thinking";
        let out = parse_complete(MUSE_GLIMMER_UNIFIED_FAMILY, cut, &[]).unwrap();
        assert_eq!(out.reasoning, "Still thinking");
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn prefill_follows_the_prompt() {
        assert_eq!(
            stream_prefill(DEEPSEEK_V41_UNIFIED_FAMILY, true),
            UnifiedParserStartingState::Reasoning
        );
        assert_eq!(
            stream_prefill(DEEPSEEK_V41_UNIFIED_FAMILY, false),
            UnifiedParserStartingState::Response
        );
        assert_eq!(
            detect_prefill(DEEPSEEK_V41_UNIFIED_FAMILY, "a</think>b"),
            UnifiedParserStartingState::Reasoning
        );
        assert_eq!(
            detect_prefill(DEEPSEEK_V41_UNIFIED_FAMILY, "<think>a</think>b"),
            UnifiedParserStartingState::None
        );
        assert_eq!(
            detect_prefill(DEEPSEEK_V41_UNIFIED_FAMILY, "plain"),
            UnifiedParserStartingState::Response
        );
    }
}
