// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified reasoning + tool-call parser for XiaomiMiMo MiMo output.
//!
//! ```text
//! reasoning:  <think>…</think>
//! tool calls: <tool_call>
//!             <function=NAME>
//!             <parameter=KEY>VALUE</parameter>
//!             </function>
//!             </tool_call>
//! ```
//!
//! The markup is Qwen3-Coder's, with one difference that decides correctness for
//! code-editing tools: MiMo's prompt states that the text between the parameter
//! tags is the value EXACTLY, including leading and trailing newlines and spaces,
//! and its reference parser does no trimming. The Qwen3-Coder parsers this crate
//! already ships trim each value, which silently rewrites a patch or a file body,
//! so this family gets its own block decoder rather than an alias.
//!
//! One `<tool_call>` block holds one `<function=…>` call, and a turn may contain
//! several blocks. A block is decoded once its `</tool_call>` has arrived, which is
//! also what the engine's parser does, so a call reaches the client complete rather
//! than argument-by-argument.

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, ToolCallDelta, UnifiedParser, UnifiedParserInit,
    UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
};

/// The unified family name, used for both `--dyn-tool-call-parser` and
/// `--dyn-reasoning-parser`.
pub const MIMO_UNIFIED_FAMILY: &str = "mimo";

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

const MARKERS: [&str; 4] = [THINK_OPEN, THINK_CLOSE, CALL_OPEN, CALL_CLOSE];

/// Read one `<TAG=NAME>BODY</TAG>` element at the first `open` in `text`.
///
/// Returns the name, the body verbatim, and the offset just past the closing tag.
/// A missing `>` or closing tag reads as no element at all, so a truncated block
/// never yields half a call.
fn read_element<'a>(text: &'a str, open: &str, close: &str) -> Option<(&'a str, &'a str, usize)> {
    let at = text.find(open)?;
    let after_open = at + open.len();
    let name_len = text[after_open..].find('>')?;
    let body_start = after_open + name_len + 1;
    let body_len = text[body_start..].find(close)?;
    Some((
        &text[after_open..after_open + name_len],
        &text[body_start..body_start + body_len],
        body_start + body_len + close.len(),
    ))
}

/// Length of the longest suffix of `text` that could still grow into a marker.
fn held_back(text: &str) -> usize {
    let longest = MARKERS.iter().map(|marker| marker.len()).max().unwrap_or(0);
    let max = text.len().min(longest - 1);
    (1..=max)
        .rev()
        .find(|len| {
            text.is_char_boundary(text.len() - len)
                && MARKERS
                    .iter()
                    .any(|marker| marker.starts_with(&text[text.len() - len..]))
        })
        .unwrap_or(0)
}

/// Whether a rendered prompt leaves generation inside an open thought. MiMo's
/// template writes `<think></think>` when thinking is off and nothing when it is on,
/// so a prompt never opens one — but an upstream template that does is honoured.
pub(crate) fn prompt_opens_reasoning(prompt: &str) -> bool {
    prompt.trim_end().ends_with(THINK_OPEN)
}

/// Which channel complete output text starts in, for the batch path: a closer with
/// no opener before it means the prompt had already opened the thought.
pub(crate) fn detect_starting_state(content: &str) -> UnifiedParserStartingState {
    match (content.find(THINK_OPEN), content.find(THINK_CLOSE)) {
        (None, Some(_)) => UnifiedParserStartingState::Reasoning,
        (Some(open_at), Some(close_at)) if close_at < open_at => {
            UnifiedParserStartingState::Reasoning
        }
        _ => UnifiedParserStartingState::None,
    }
}

/// HTML entity decoding, matching the `html.unescape` the engine's parser applies
/// to every value before typing it.
fn html_unescape(value: &str) -> String {
    if !value.contains('&') {
        return value.to_string();
    }
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", "\u{a0}")
        .replace("&amp;", "&")
}

/// The JSON Schema type declared for one parameter, defaulting to `string`.
fn schema_type(tools: &[Tool], function: &str, parameter: &str) -> String {
    tools
        .iter()
        .find(|tool| tool.name == function)
        .and_then(|tool| tool.parameters.get("properties"))
        .and_then(|properties| properties.get(parameter))
        .and_then(|schema| schema.get("type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("string")
        .to_string()
}

/// Type one raw parameter value against its schema.
///
/// A string-typed value is the model's bytes verbatim. A value that does not parse
/// as its declared type stays a string rather than failing the call, which is what
/// the engine's parser does.
fn typed_value(raw: &str, declared: &str) -> serde_json::Value {
    let value = html_unescape(raw);
    if value.eq_ignore_ascii_case("null") {
        return serde_json::Value::Null;
    }
    let as_string = || serde_json::Value::String(value.clone());
    let trimmed = value.trim();
    let declared = declared.to_ascii_lowercase();
    match declared.as_str() {
        "string" | "str" | "text" | "varchar" | "char" | "enum" => as_string(),
        "boolean" | "bool" | "binary" => match trimmed.to_ascii_lowercase().as_str() {
            "true" => serde_json::Value::Bool(true),
            "false" => serde_json::Value::Bool(false),
            _ => as_string(),
        },
        _ if declared.starts_with("int")
            || declared.starts_with("uint")
            || declared.starts_with("long")
            || declared.starts_with("short")
            || declared.starts_with("unsigned") =>
        {
            trimmed
                .parse::<i64>()
                .map(serde_json::Value::from)
                .unwrap_or_else(|_| as_string())
        }
        _ if declared.starts_with("num") || declared.starts_with("float") => trimmed
            .parse::<f64>()
            .ok()
            .map(|number| {
                if number.fract() == 0.0 && number.abs() < i64::MAX as f64 {
                    serde_json::Value::from(number as i64)
                } else {
                    serde_json::Value::from(number)
                }
            })
            .unwrap_or_else(as_string),
        _ => serde_json::from_str(trimmed).unwrap_or_else(|_| as_string()),
    }
}

/// Decode one `<tool_call>` block body into a call, or `None` when it carries no
/// complete `<function=…>` envelope.
fn decode_block(body: &str, tools: &[Tool]) -> Option<(String, serde_json::Value)> {
    let (name, function_body, _) = read_element(body, FUNCTION_OPEN, FUNCTION_CLOSE)?;
    let name = name.trim().to_string();
    let mut arguments = serde_json::Map::new();
    let mut rest = function_body;
    while let Some((key, value, next)) = read_element(rest, PARAMETER_OPEN, PARAMETER_CLOSE) {
        let key = key.trim().to_string();
        let declared = schema_type(tools, &name, &key);
        arguments.insert(key, typed_value(value, &declared));
        rest = &rest[next..];
    }
    Some((name, serde_json::Value::Object(arguments)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Reasoning,
    Response,
    /// Inside a `<tool_call>` block, buffering until its close.
    Call,
}

/// Build one MiMo parser for one response stream.
pub fn mimo_unified(tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(MimoUnifiedParser::new(tools)))
}

pub struct MimoUnifiedParser {
    tools: Vec<Tool>,
    channel: Channel,
    /// Text whose classification is not yet decided: an open call block, a guided
    /// payload, or a tail that could still grow into a marker.
    buffer: String,
    next_index: usize,
    guided: bool,
    named_tool: Option<String>,
}

impl MimoUnifiedParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            tools: tools.to_vec(),
            channel: Channel::Response,
            buffer: String::new(),
            next_index: 0,
            guided: false,
            named_tool: None,
        }
    }

    fn push_run(&mut self, text: &str, output: &mut UnifiedParserOutput) {
        if text.is_empty() {
            return;
        }
        match self.channel {
            Channel::Reasoning => output.push_reasoning(text),
            Channel::Response => output.push_text(text),
            Channel::Call => self.buffer.push_str(text),
        }
    }

    fn open_call(&mut self) {
        self.channel = Channel::Call;
        self.buffer.clear();
    }

    fn close_call(&mut self, output: &mut UnifiedParserOutput) {
        let body = std::mem::take(&mut self.buffer);
        self.channel = Channel::Response;
        let Some((name, arguments)) = decode_block(&body, &self.tools) else {
            tracing::warn!(
                why = "mimo_unparsable_block",
                "MiMo tool-call block decoded to no call"
            );
            return;
        };
        if !self.tools.is_empty() && !self.tools.iter().any(|tool| tool.name == name) {
            tracing::warn!(name, "MiMo tool call names an unknown tool");
            return;
        }
        output.push_call(ToolCallDelta {
            tool_index: self.next_index,
            name: Some(name),
            arguments: arguments.to_string(),
            complete: true,
        });
        self.next_index += 1;
    }
}

impl UnifiedParser for MimoUnifiedParser {
    fn initialize_request(&mut self, init: UnifiedParserInit) -> anyhow::Result<()> {
        self.channel = match init.starting_state {
            UnifiedParserStartingState::Reasoning => Channel::Reasoning,
            UnifiedParserStartingState::None | UnifiedParserStartingState::Response => {
                Channel::Response
            }
        };
        match init.tool_output_mode {
            UnifiedToolOutputMode::Native => {}
            UnifiedToolOutputMode::GuidedJson { named_tool } => {
                anyhow::ensure!(
                    init.invalid_guided_payload != InvalidGuidedPayloadPolicy::StreamBestEffort,
                    "mimo buffers guided tool output and cannot stream it"
                );
                self.guided = true;
                self.named_tool = named_tool;
            }
        }
        Ok(())
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        // Guided decoding replaces the tool markup with bare JSON, so only the
        // reasoning close still means anything; the payload is held for `finish`.
        if self.guided {
            let text = format!("{}{delta}", std::mem::take(&mut self.buffer));
            if self.channel == Channel::Reasoning {
                match text.find(THINK_CLOSE) {
                    Some(at) => {
                        output.push_reasoning(&text[..at]);
                        self.channel = Channel::Response;
                        self.buffer = text[at + THINK_CLOSE.len()..].to_string();
                    }
                    None => {
                        let keep = held_back(&text);
                        output.push_reasoning(&text[..text.len() - keep]);
                        self.buffer = text[text.len() - keep..].to_string();
                    }
                }
            } else {
                self.buffer = text;
            }
            return Ok(());
        }

        let mut text = std::mem::take(&mut self.buffer);
        let carry_over = self.channel == Channel::Call;
        if carry_over {
            // The open block's body is already buffered; re-scanning it costs
            // nothing and keeps one code path for a close marker split across
            // chunks.
            text.push_str(delta);
            self.buffer.clear();
        } else {
            text.push_str(delta);
        }
        if carry_over {
            self.channel = Channel::Call;
        }

        let mut at = 0;
        let mut run_start = 0;
        while let Some(found) = text[at..].find('<') {
            let start = at + found;
            let rest = &text[start..];
            let matched = MARKERS
                .iter()
                .find(|marker| rest.starts_with(**marker))
                .copied();
            let Some(marker) = matched else {
                if MARKERS.iter().any(|marker| marker.starts_with(rest)) {
                    break;
                }
                at = start + 1;
                continue;
            };
            let run = &text[run_start..start].to_string();
            match (self.channel, marker) {
                (Channel::Call, CALL_CLOSE) => {
                    self.push_run(run, output);
                    self.close_call(output);
                }
                // Inside a block every other marker is part of an argument value.
                (Channel::Call, _) => {
                    at = start + marker.len();
                    continue;
                }
                (Channel::Reasoning, THINK_CLOSE) => {
                    self.push_run(run, output);
                    self.channel = Channel::Response;
                }
                // A call opening before `</think>` closes the thought, matching the
                // engine's reasoning detector.
                (Channel::Reasoning, CALL_OPEN) | (Channel::Response, CALL_OPEN) => {
                    self.push_run(run, output);
                    self.channel = Channel::Response;
                    self.open_call();
                }
                (Channel::Response, THINK_OPEN) => {
                    self.push_run(run, output);
                    self.channel = Channel::Reasoning;
                }
                // A stray opener inside a thought, a stray closer outside one, and a
                // close with no block open are all markup the client must not see.
                _ => self.push_run(run, output),
            }
            at = start + marker.len();
            run_start = at;
        }

        let tail = &text[run_start..];
        if self.channel == Channel::Call {
            // A block holds until its close; nothing of it may go out early.
            self.buffer.push_str(tail);
        } else {
            let split = tail.len() - held_back(tail);
            let (emit, hold) = tail.split_at(split);
            let hold = hold.to_string();
            self.push_run(emit, output);
            self.buffer = hold;
        }
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        let tail = std::mem::take(&mut self.buffer);
        match self.channel {
            // An unterminated block is truncation: dropped, never leaked as text.
            Channel::Call => {
                tracing::warn!(
                    why = "mimo_incomplete_tool_call",
                    "MiMo stream ended inside a tool-call block"
                );
            }
            Channel::Reasoning => output.push_reasoning(tail),
            Channel::Response if self.guided => {
                if !tail.trim().is_empty() {
                    match super::unified_parser::guided_json_calls(
                        &tail,
                        self.named_tool.as_deref(),
                    ) {
                        Some(calls) => {
                            for call in calls {
                                output.push_call(call);
                            }
                        }
                        None => output.push_text(tail),
                    }
                }
            }
            Channel::Response => output.push_text(tail),
        }
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let unconsumed = std::mem::take(&mut self.buffer);
        *self = Self::new(&self.tools);
        unconsumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::{UnifiedEvent, assemble};
    use serde_json::json;

    fn tools() -> Vec<Tool> {
        let tool = |name: &str, properties: serde_json::Value| Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": properties}),
            strict: None,
        };
        vec![
            tool("execute_bash", json!({"command": {"type": "string"}})),
            tool(
                "edit_file",
                json!({"path": {"type": "string"}, "body": {"type": "string"}}),
            ),
            tool(
                "get_weather",
                json!({
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "metric": {"type": "boolean"},
                    "tags": {"type": "array"}
                }),
            ),
        ]
    }

    fn run(
        chunks: &[&str],
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> Vec<UnifiedEvent> {
        let mut parser = MimoUnifiedParser::new(&tools());
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                tool_output_mode,
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assemble(&output.events)
    }

    fn native(chunks: &[&str]) -> Vec<UnifiedEvent> {
        run(
            chunks,
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::Native,
        )
    }

    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }

    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    fn assert_split_invariant(input: &str, starting_state: UnifiedParserStartingState) {
        let whole = run(&[input], starting_state, UnifiedToolOutputMode::Native);
        for (at, _) in input.char_indices().skip(1) {
            assert_eq!(
                run(
                    &[&input[..at], &input[at..]],
                    starting_state,
                    UnifiedToolOutputMode::Native
                ),
                whole,
                "split at byte {at} of {input:?}"
            );
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            run(&chars, starting_state, UnifiedToolOutputMode::Native),
            whole,
            "char-at-a-time {input:?}"
        );
    }

    const BASH_CALL: &str = "<tool_call>\n<function=execute_bash>\n\
         <parameter=command>pwd && ls</parameter>\n</function>\n</tool_call>";

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(
            native(&["This is a plain response."]),
            vec![text("This is a plain response.")]
        );
        assert_eq!(native(&["a < b and x<y>"]), vec![text("a < b and x<y>")]);
    }

    #[test]
    fn reasoning_block_then_answer() {
        assert_eq!(
            native(&["<think>Let me think.</think>The answer is 42."]),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn prompt_opened_reasoning_closes_without_an_opener() {
        assert_eq!(
            run(
                &["Let me think.</think>The answer is 42."],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn single_tool_call() {
        assert_eq!(
            native(&[BASH_CALL]),
            vec![call("execute_bash", json!({"command": "pwd && ls"}))]
        );
    }

    #[test]
    fn text_reasoning_and_multiple_calls_stay_in_order() {
        let input = format!(
            "<think>Two things.</think>On it.{BASH_CALL}\n\
             <tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n\
             <parameter=days>3</parameter>\n</function>\n</tool_call>Done."
        );
        assert_eq!(
            native(&[&input]),
            vec![
                reasoning("Two things."),
                text("On it."),
                call("execute_bash", json!({"command": "pwd && ls"})),
                text("\n"),
                call("get_weather", json!({"city": "Paris", "days": 3})),
                text("Done."),
            ]
        );
    }

    #[test]
    fn parameter_values_are_preserved_byte_for_byte() {
        let body = "\nline one\n  indented\n\n";
        let input = format!(
            "<tool_call>\n<function=edit_file>\n<parameter=path>src/main.rs</parameter>\n\
             <parameter=body>{body}</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            native(&[&input]),
            vec![call(
                "edit_file",
                json!({"path": "src/main.rs", "body": body})
            )]
        );
    }

    #[test]
    fn values_are_typed_from_the_schema_and_entities_decoded() {
        let input = "<tool_call><function=get_weather>\
             <parameter=city>a &lt;b&gt; &amp; c</parameter>\
             <parameter=days>3</parameter>\
             <parameter=ratio>1.5</parameter>\
             <parameter=metric>true</parameter>\
             <parameter=tags>[\"a\", \"b\"]</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "get_weather",
                json!({
                    "city": "a <b> & c",
                    "days": 3,
                    "ratio": 1.5,
                    "metric": true,
                    "tags": ["a", "b"]
                })
            )]
        );
    }

    #[test]
    fn a_value_that_defies_its_type_stays_a_string() {
        let input = "<tool_call><function=get_weather>\
             <parameter=days>soon</parameter><parameter=metric>maybe</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "get_weather",
                json!({"days": "soon", "metric": "maybe"})
            )]
        );
    }

    #[test]
    fn markup_inside_a_value_is_value_text() {
        let input = "<tool_call><function=execute_bash>\
             <parameter=command>echo '<think>' && echo '</tool_call'</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "execute_bash",
                json!({"command": "echo '<think>' && echo '</tool_call'"})
            )]
        );
    }

    #[test]
    fn a_call_opening_inside_a_thought_ends_it() {
        assert_eq!(
            run(
                &[&format!("Need the listing.{BASH_CALL}")],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![
                reasoning("Need the listing."),
                call("execute_bash", json!({"command": "pwd && ls"}))
            ]
        );
    }

    #[test]
    fn chunk_boundaries_never_change_the_result() {
        assert_split_invariant(
            &format!("<think>Think.</think>Text. {BASH_CALL} tail"),
            UnifiedParserStartingState::None,
        );
        assert_split_invariant(
            "thought</think>a < b, <b>bold</b>",
            UnifiedParserStartingState::Reasoning,
        );
    }

    #[test]
    fn split_marker_is_held_until_decided() {
        let mut parser = MimoUnifiedParser::new(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("Hello <tool_c", &mut output).unwrap();
        assert_eq!(assemble(&output.events), vec![text("Hello ")]);
        parser.parse_into("ase is closed", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![text("Hello <tool_case is closed")]
        );
    }

    #[test]
    fn unterminated_reasoning_is_reasoning() {
        assert_eq!(
            native(&["<think>still thinking when the budget ran out"]),
            vec![reasoning("still thinking when the budget ran out")]
        );
    }

    #[test]
    fn unterminated_call_is_dropped_without_leaking_markup() {
        assert_eq!(
            native(&["On it.<tool_call>\n<function=execute_bash>\n<parameter=command>pw"]),
            vec![text("On it.")]
        );
    }

    #[test]
    fn a_block_with_no_function_envelope_emits_nothing() {
        assert_eq!(native(&["<tool_call>garbage</tool_call>"]), vec![]);
    }

    #[test]
    fn unknown_tool_is_not_a_call() {
        let input = format!("<tool_call><function=nonexistent></function></tool_call>{BASH_CALL}");
        assert_eq!(
            native(&[&input]),
            vec![call("execute_bash", json!({"command": "pwd && ls"}))]
        );
    }

    #[test]
    fn guided_named_and_required_choices() {
        assert_eq!(
            run(
                &["Pick Paris.</think>{\"city\": ", "\"Paris\"}"],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".into())
                }
            ),
            vec![
                reasoning("Pick Paris."),
                call("get_weather", json!({"city": "Paris"}))
            ]
        );
        assert_eq!(
            run(
                &[r#"[{"name":"execute_bash","arguments":{"command":"ls"}}]"#],
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None }
            ),
            vec![call("execute_bash", json!({"command": "ls"}))]
        );
        assert_eq!(
            run(
                &[r#"[{"name":"execute_bash","argu"#],
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None }
            ),
            vec![text(r#"[{"name":"execute_bash","argu"#)]
        );
    }

    #[test]
    fn prompt_and_batch_starting_state_detection() {
        assert!(prompt_opens_reasoning("<|im_start|>assistant\n<think>"));
        assert!(!prompt_opens_reasoning(
            "<|im_start|>assistant\n<think></think>"
        ));
        assert!(!prompt_opens_reasoning("<|im_start|>assistant\n"));

        assert_eq!(
            detect_starting_state("thought</think>answer"),
            UnifiedParserStartingState::Reasoning
        );
        assert_eq!(
            detect_starting_state("<think>thought</think>answer"),
            UnifiedParserStartingState::None
        );
        assert_eq!(
            detect_starting_state("answer"),
            UnifiedParserStartingState::None
        );
    }

    #[test]
    fn reset_returns_unconsumed_text() {
        let mut parser = MimoUnifiedParser::new(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("abc <tool_c", &mut output).unwrap();
        assert_eq!(parser.reset(), "<tool_c");
        parser.parse_into("plain", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(assemble(&output.events), vec![text("abc plain")]);
    }
}
