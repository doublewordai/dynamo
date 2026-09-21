// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified reasoning + tool-call parser for Tencent Hunyuan (Hy3) output.
//!
//! ```text
//! reasoning:  <think>…</think>
//! tool calls: <tool_calls>
//!             <tool_call>NAME<tool_sep>
//!             <arg_key>K</arg_key>
//!             <arg_value>V</arg_value>
//!             </tool_call>
//!             </tool_calls>
//! ```
//!
//! The shipping tokenizer appends one shared suffix to every marker
//! (`<think:opensource>`, `</tool_call:opensource>`, …); preview tokenizers use the
//! bare spelling. Both are accepted: a marker is its name plus an optional
//! `:suffix`. Newer checkpoints drop `<tool_sep>`, which is optional here too.
//!
//! Inside one `<tool_call>` block the grammar is GLM-4.7's
//! (`NAME<arg_key>…</arg_key><arg_value>…</arg_value>`), so each block is rewritten
//! to the bare GLM spelling and handed to [`Glm47ToolStreamParser`], which owns
//! block buffering and schema-aware argument typing. This type owns what Hunyuan
//! adds: marker suffixes, the `<tool_calls>` wrapper, `<tool_sep>`, and the
//! reasoning channel, including a `<tool_calls>` that arrives before `</think>`
//! and closes the thought implicitly.

use dynamo_parsers_v2::{
    Glm47ToolStreamParser, InvalidGuidedPayloadPolicy, Tool, ToolParser, UnifiedParser,
    UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
};

/// The unified family name, used for both `--dyn-tool-call-parser` and
/// `--dyn-reasoning-parser`.
pub const HUNYUAN_UNIFIED_FAMILY: &str = "hunyuan";

/// Longest `:suffix` accepted on a marker. Bounds how much text a `<name:` tail can
/// hold back while waiting for its `>`.
const MAX_SUFFIX_LEN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    Think,
    ToolCalls,
    ToolCall,
    ToolSep,
    ArgKey,
    ArgValue,
}

impl Marker {
    // `tool_calls` precedes `tool_call` so the longer name is tried first.
    const ALL: [(Marker, &'static str); 6] = [
        (Marker::Think, "think"),
        (Marker::ToolCalls, "tool_calls"),
        (Marker::ToolCall, "tool_call"),
        (Marker::ToolSep, "tool_sep"),
        (Marker::ArgKey, "arg_key"),
        (Marker::ArgValue, "arg_value"),
    ];

    fn bare(self, close: bool) -> &'static str {
        match (self, close) {
            (Marker::ToolCall, false) => "<tool_call>",
            (Marker::ToolCall, true) => "</tool_call>",
            (Marker::ArgKey, false) => "<arg_key>",
            (Marker::ArgKey, true) => "</arg_key>",
            (Marker::ArgValue, false) => "<arg_value>",
            (Marker::ArgValue, true) => "</arg_value>",
            _ => "",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Scan {
    /// A complete marker of `len` bytes.
    Full {
        marker: Marker,
        close: bool,
        len: usize,
    },
    /// The text so far is a proper prefix of some marker.
    Partial,
    /// Not a marker.
    None,
}

/// Classify `text`, which starts at a `<`.
fn scan_marker(text: &str) -> Scan {
    debug_assert!(text.starts_with('<'));
    let after_open = &text[1..];
    let (close, body) = match after_open.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, after_open),
    };
    let head_len = text.len() - body.len();
    let mut partial = false;
    for (marker, name) in Marker::ALL {
        let Some(rest) = body.strip_prefix(name) else {
            partial |= name.starts_with(body);
            continue;
        };
        if rest.is_empty() {
            partial = true;
            continue;
        }
        if rest.starts_with('>') {
            return Scan::Full {
                marker,
                close,
                len: head_len + name.len() + 1,
            };
        }
        let Some(suffix) = rest.strip_prefix(':') else {
            continue;
        };
        let suffix_len = suffix
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
            .unwrap_or(suffix.len());
        if suffix_len > MAX_SUFFIX_LEN {
            continue;
        }
        match suffix[suffix_len..].chars().next() {
            Option::None => partial = true,
            Some('>') if suffix_len > 0 => {
                return Scan::Full {
                    marker,
                    close,
                    len: head_len + name.len() + 1 + suffix_len + 1,
                };
            }
            Some(_) => {}
        }
    }
    if partial { Scan::Partial } else { Scan::None }
}

/// Whether a rendered prompt ends by opening a thought, so generated output starts
/// inside reasoning. A prompt ending `<think></think>` (thinking off) does not.
pub(crate) fn prompt_opens_reasoning(prompt: &str) -> bool {
    let prompt = prompt.trim_end();
    let Some(open_at) = prompt.rfind('<') else {
        return false;
    };
    let tail = &prompt[open_at..];
    scan_marker(tail)
        == Scan::Full {
            marker: Marker::Think,
            close: false,
            len: tail.len(),
        }
}

/// Which channel complete output text starts in, for the batch path that has no
/// prompt in hand: a closer with no opener before it means the prompt opened the
/// thought.
pub(crate) fn detect_starting_state(content: &str) -> UnifiedParserStartingState {
    let mut at = 0;
    while let Some(found) = content[at..].find('<') {
        let start = at + found;
        if let Scan::Full {
            marker: Marker::Think,
            close,
            ..
        } = scan_marker(&content[start..])
        {
            return if close {
                UnifiedParserStartingState::Reasoning
            } else {
                UnifiedParserStartingState::None
            };
        }
        at = start + 1;
    }
    UnifiedParserStartingState::None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Reasoning,
    Response,
}

/// Build one Hunyuan parser for one response stream.
pub fn hunyuan_unified(tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(HunyuanUnifiedParser::new(tools)))
}

pub struct HunyuanUnifiedParser {
    tools: Vec<Tool>,
    calls: Glm47ToolStreamParser,
    channel: Channel,
    /// Inside the `<tool_calls>` wrapper.
    in_wrapper: bool,
    /// Inside one `<tool_call>` block.
    in_call: bool,
    /// Undecided tail: a possible marker prefix split across a chunk boundary.
    pending: String,
    /// `Some` when guided decoding replaces native markup with bare JSON; holds the
    /// response-channel payload until the stream ends.
    guided: Option<String>,
    /// The guided payload's first byte has arrived; everything after it is payload.
    guided_started: bool,
    named_tool: Option<String>,
    next_index: usize,
}

impl HunyuanUnifiedParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            tools: tools.to_vec(),
            calls: Glm47ToolStreamParser::new(tools),
            channel: Channel::Response,
            in_wrapper: false,
            in_call: false,
            pending: String::new(),
            guided: None,
            guided_started: false,
            named_tool: None,
            next_index: 0,
        }
    }

    fn route_text(&mut self, text: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if self.channel == Channel::Reasoning {
            output.push_reasoning(text);
            return Ok(());
        }
        if let Some(payload) = &mut self.guided {
            payload.push_str(text);
            return Ok(());
        }
        // Layout whitespace between the wrapper and its blocks is not content.
        if self.in_wrapper && !self.in_call && text.trim().is_empty() {
            return Ok(());
        }
        self.feed_calls(text, output)
    }

    fn feed_calls(&mut self, text: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        let result = self.calls.push(text)?;
        self.emit(result, output);
        Ok(())
    }

    /// Forward one block-parser result. A call naming a tool the request did not
    /// offer is dropped, and the calls that remain are numbered contiguously.
    fn emit(
        &mut self,
        result: dynamo_parsers_v2::ToolParseResult,
        output: &mut UnifiedParserOutput,
    ) {
        if !result.normal_text.is_empty() {
            output.push_text(result.normal_text);
        }
        for mut call in result.calls {
            let offered = self.tools.is_empty()
                || call
                    .name
                    .as_deref()
                    .is_some_and(|name| self.tools.iter().any(|tool| tool.name == name));
            if !offered {
                tracing::warn!(name = ?call.name, "hunyuan tool call names an unknown tool");
                continue;
            }
            call.tool_index = self.next_index;
            self.next_index += 1;
            output.push_call(call);
        }
    }

    fn route_marker(
        &mut self,
        marker: Marker,
        close: bool,
        raw: &str,
        output: &mut UnifiedParserOutput,
    ) -> anyhow::Result<()> {
        if self.channel == Channel::Reasoning {
            match (marker, close) {
                (Marker::Think, true) => self.channel = Channel::Response,
                (Marker::Think, false) => {}
                // A call opened before `</think>` closes the thought.
                (Marker::ToolCalls | Marker::ToolCall, false) => {
                    self.channel = Channel::Response;
                    return self.route_marker(marker, close, raw, output);
                }
                _ => output.push_reasoning(raw),
            }
            return Ok(());
        }
        // A marker spelled inside an argument value is that value's text.
        if self.in_call && matches!(marker, Marker::Think | Marker::ToolCalls) {
            return self.feed_calls(raw, output);
        }
        match (marker, close) {
            (Marker::Think, false) => self.channel = Channel::Reasoning,
            (Marker::Think, true) => {}
            (Marker::ToolCalls, false) => self.in_wrapper = true,
            (Marker::ToolCalls, true) => self.in_wrapper = false,
            (Marker::ToolSep, _) => {}
            (Marker::ToolCall, false) => {
                self.in_call = true;
                self.feed_calls(marker.bare(false), output)?;
            }
            (Marker::ToolCall, true) => {
                self.in_call = false;
                self.feed_calls(marker.bare(true), output)?;
            }
            (Marker::ArgKey | Marker::ArgValue, _) => {
                self.feed_calls(marker.bare(close), output)?;
            }
        }
        Ok(())
    }

    /// Advance a guided-decoding stream. Only a thought AHEAD of the payload is
    /// framing: once the payload's first byte arrives every byte belongs to it, so a
    /// marker spelled inside a JSON string argument reaches the tool unchanged.
    fn parse_guided(&mut self, delta: &str, output: &mut UnifiedParserOutput) {
        self.pending.push_str(delta);
        let text = std::mem::take(&mut self.pending);
        let mut rest = text.as_str();
        loop {
            if self.channel == Channel::Reasoning {
                let mut at = 0;
                let mut end = rest.len();
                let mut closed = None;
                while let Some(found) = rest[at..].find('<') {
                    let start = at + found;
                    match scan_marker(&rest[start..]) {
                        Scan::Full {
                            marker: Marker::Think,
                            close: true,
                            len,
                        } => {
                            closed = Some(start + len);
                            end = start;
                            break;
                        }
                        Scan::Partial => {
                            end = start;
                            break;
                        }
                        _ => at = start + 1,
                    }
                }
                if end > 0 {
                    output.push_reasoning(&rest[..end]);
                }
                match closed {
                    Some(after) => {
                        self.channel = Channel::Response;
                        rest = &rest[after..];
                        continue;
                    }
                    None => {
                        self.pending.push_str(&rest[end..]);
                        return;
                    }
                }
            }
            if !self.guided_started {
                let body = rest.trim_start();
                let opener = match body.starts_with('<').then(|| scan_marker(body)) {
                    _ if body.is_empty() => Scan::Partial,
                    Some(scan) => scan,
                    None => Scan::None,
                };
                match opener {
                    Scan::Full {
                        marker: Marker::Think,
                        close: false,
                        len,
                    } => {
                        self.channel = Channel::Reasoning;
                        rest = &body[len..];
                        continue;
                    }
                    Scan::Partial => {
                        self.pending.push_str(rest);
                        return;
                    }
                    _ => self.guided_started = true,
                }
            }
            if let Some(payload) = &mut self.guided {
                payload.push_str(rest);
            }
            return;
        }
    }

    fn flush_guided(&mut self, output: &mut UnifiedParserOutput) {
        let Some(payload) = self.guided.as_mut().map(std::mem::take) else {
            return;
        };
        if payload.trim().is_empty() {
            return;
        }
        match super::unified_parser::guided_json_calls(&payload, self.named_tool.as_deref()) {
            Some(calls) => {
                for call in calls {
                    output.push_call(call);
                }
            }
            None => output.push_text(payload),
        }
    }
}

impl UnifiedParser for HunyuanUnifiedParser {
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
                    "hunyuan buffers guided tool output and cannot stream it"
                );
                self.guided = Some(String::new());
                self.named_tool = named_tool;
            }
        }
        Ok(())
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        if self.guided.is_some() {
            self.parse_guided(delta, output);
            return Ok(());
        }
        self.pending.push_str(delta);
        let text = std::mem::take(&mut self.pending);
        let mut run_start = 0;
        let mut at = 0;
        while let Some(found) = text[at..].find('<') {
            let start = at + found;
            match scan_marker(&text[start..]) {
                Scan::Full { marker, close, len } => {
                    self.route_text(&text[run_start..start], output)?;
                    self.route_marker(marker, close, &text[start..start + len], output)?;
                    at = start + len;
                    run_start = at;
                }
                Scan::Partial => {
                    self.route_text(&text[run_start..start], output)?;
                    self.pending.push_str(&text[start..]);
                    return Ok(());
                }
                Scan::None => at = start + 1,
            }
        }
        self.route_text(&text[run_start..], output)
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        let tail = std::mem::take(&mut self.pending);
        self.route_text(&tail, &mut output)?;
        if self.guided.is_some() {
            self.flush_guided(&mut output);
        } else {
            let tail = self.calls.finish()?;
            self.emit(tail, &mut output);
        }
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let mut unconsumed = self.guided.take().unwrap_or_default();
        unconsumed.push_str(&std::mem::take(&mut self.pending));
        *self = Self::new(&self.tools);
        unconsumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::{UnifiedEvent, assemble};
    use serde_json::json;

    const SFX: &str = ":opensource";

    fn tools() -> Vec<Tool> {
        let tool = |name: &str, properties: serde_json::Value| Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": properties}),
            strict: None,
        };
        vec![
            tool(
                "get_weather",
                json!({
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "tags": {"type": "array"}
                }),
            ),
            tool("get_current_date", json!({})),
            tool("search", json!({"query": {"type": "string"}})),
        ]
    }

    /// Spell a bare-marker fixture the way the shipping tokenizer does.
    fn suffixed(bare: &str) -> String {
        let mut out = bare.to_string();
        for (_, name) in Marker::ALL {
            out = out
                .replace(&format!("<{name}>"), &format!("<{name}{SFX}>"))
                .replace(&format!("</{name}>"), &format!("</{name}{SFX}>"));
        }
        out
    }

    fn run(
        chunks: &[&str],
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> Vec<UnifiedEvent> {
        let mut parser = HunyuanUnifiedParser::new(&tools());
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

    /// Every two-chunk split of `input`, plus one character at a time, must
    /// assemble to the same events as the whole input.
    fn assert_split_invariant(input: &str, starting_state: UnifiedParserStartingState) {
        let whole = run(&[input], starting_state, UnifiedToolOutputMode::Native);
        for (at, _) in input.char_indices().skip(1) {
            let got = run(
                &[&input[..at], &input[at..]],
                starting_state,
                UnifiedToolOutputMode::Native,
            );
            assert_eq!(got, whole, "split at byte {at} of {input:?}");
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            run(&chars, starting_state, UnifiedToolOutputMode::Native),
            whole,
            "char-at-a-time {input:?}"
        );
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
        let input = suffixed("<think>Let me think.</think>The answer is 42.");
        assert_eq!(
            native(&[&input]),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn prompt_opened_reasoning_closes_without_an_opener() {
        let input = suffixed("Let me think.</think>The answer is 42.");
        assert_eq!(
            run(
                &[&input],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn zero_argument_call_inline_and_with_newlines() {
        for bare in [
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>",
        ] {
            for input in [bare.to_string(), suffixed(bare)] {
                assert_eq!(
                    native(&[&input]),
                    vec![call("get_current_date", json!({}))],
                    "{input:?}"
                );
            }
        }
    }

    #[test]
    fn single_call_with_template_layout() {
        let input = suffixed(
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n\
             <arg_value>Beijing</arg_value>\n<arg_key>date</arg_key>\n\
             <arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![call(
                "get_weather",
                json!({"city": "Beijing", "date": "2026-03-30"})
            )]
        );
    }

    #[test]
    fn call_without_tool_separator() {
        let input = suffixed(
            "<tool_calls><tool_call>get_weather\n<arg_key>city</arg_key>\n\
             <arg_value>Beijing</arg_value>\n</tool_call></tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![call("get_weather", json!({"city": "Beijing"}))]
        );
    }

    #[test]
    fn content_before_tool_call_is_kept() {
        let input = suffixed(
            "Checking.<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![text("Checking."), call("get_current_date", json!({}))]
        );
    }

    #[test]
    fn multiple_tool_calls_keep_order() {
        let input = suffixed(
            "<tool_calls>\
             <tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>\n\
             <tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Hangzhou</arg_value></tool_call>\n\
             <tool_call>get_current_date<tool_sep></tool_call>\
             </tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![
                call("get_weather", json!({"city": "Beijing"})),
                call("get_weather", json!({"city": "Hangzhou"})),
                call("get_current_date", json!({})),
            ]
        );
    }

    #[test]
    fn arguments_are_typed_from_the_schema() {
        let input = suffixed(
            "<tool_calls><tool_call>get_weather<tool_sep>\
             <arg_key>city</arg_key><arg_value>123</arg_value>\
             <arg_key>days</arg_key><arg_value>3</arg_value>\
             <arg_key>metric</arg_key><arg_value>true</arg_value>\
             <arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>\
             </tool_call></tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![call(
                "get_weather",
                json!({"city": "123", "days": 3, "metric": true, "tags": ["a", "b"]})
            )]
        );
    }

    #[test]
    fn reasoning_then_tool_call() {
        let input = suffixed(
            "<think>Need the date.</think>\n<tool_calls>\n<tool_call>get_current_date<tool_sep>\n\
             </tool_call>\n</tool_calls>",
        );
        let events = native(&[&input]);
        assert_eq!(events[0], reasoning("Need the date."));
        assert_eq!(events.last().unwrap(), &call("get_current_date", json!({})));
    }

    #[test]
    fn tool_call_before_think_close_ends_the_thought() {
        let input = suffixed(
            "Need the date.<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
        );
        assert_eq!(
            run(
                &[&input],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![
                reasoning("Need the date."),
                call("get_current_date", json!({}))
            ]
        );
    }

    #[test]
    fn think_marker_inside_an_argument_value_is_value_text() {
        let input = suffixed(
            "<tool_calls><tool_call>search<tool_sep><arg_key>query</arg_key>\
             <arg_value>what does <think> mean</arg_value></tool_call></tool_calls>",
        );
        let expected = format!("what does <think{SFX}> mean");
        assert_eq!(
            native(&[&input]),
            vec![call("search", json!({"query": expected}))]
        );
    }

    #[test]
    fn chunk_boundaries_never_change_the_result() {
        let calls = suffixed(
            "<think>Compare two cities.</think>I'll check both.<tool_calls>\n\
             <tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n\
             <arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\n\
             <tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Hangzhou</arg_value>\n\
             </tool_call>\n</tool_calls>",
        );
        assert_split_invariant(&calls, UnifiedParserStartingState::None);
        let whole = native(&[&calls]);
        assert_eq!(
            whole,
            vec![
                reasoning("Compare two cities."),
                text("I'll check both."),
                call("get_weather", json!({"city": "Beijing", "days": 3})),
                call("get_weather", json!({"city": "Hangzhou"})),
            ]
        );

        let answer = suffixed("Short thought.</think>Use a < b, then <b>bold</b>.");
        assert_split_invariant(&answer, UnifiedParserStartingState::Reasoning);
        assert_split_invariant(
            "<think>bare</think>answer",
            UnifiedParserStartingState::None,
        );
    }

    #[test]
    fn split_marker_is_held_until_decided() {
        let mut parser = HunyuanUnifiedParser::new(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("Hello <tool_ca", &mut output).unwrap();
        assert_eq!(assemble(&output.events), vec![text("Hello ")]);
        parser.parse_into("ke is good", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![text("Hello <tool_cake is good")]
        );
    }

    #[test]
    fn unterminated_reasoning_is_reasoning() {
        let input = suffixed("<think>still thinking when the budget ran out");
        assert_eq!(
            native(&[&input]),
            vec![reasoning("still thinking when the budget ran out")]
        );
    }

    #[test]
    fn unterminated_tool_call_is_dropped_without_leaking_markup() {
        let input = suffixed(
            "On it.<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Bei",
        );
        assert_eq!(native(&[&input]), vec![text("On it.")]);
    }

    #[test]
    fn partial_marker_at_end_of_stream_is_text() {
        assert_eq!(native(&["value <tool_cal"]), vec![text("value <tool_cal")]);
    }

    #[test]
    fn unknown_tool_is_not_a_call() {
        let input = suffixed(
            "<tool_calls><tool_call>nonexistent<tool_sep></tool_call>\
             <tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
        );
        let events = native(&[&input]);
        let calls: Vec<_> = events
            .iter()
            .filter(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
            .collect();
        assert_eq!(calls, vec![&call("get_current_date", json!({}))]);
    }

    #[test]
    fn guided_named_choice_reads_bare_arguments_after_reasoning() {
        let input = suffixed("Pick Paris.</think>{\"city\": \"Paris\"}");
        assert_eq!(
            run(
                &[&input[..20], &input[20..]],
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
    }

    #[test]
    fn guided_required_choice_reads_call_array_and_recovers_malformed_as_text() {
        let mode = UnifiedToolOutputMode::GuidedJson { named_tool: None };
        assert_eq!(
            run(
                &[
                    r#"[{"name":"search","arguments":{"query":"a"}},"#,
                    r#"{"name":"get_current_date","parameters":{}}]"#
                ],
                UnifiedParserStartingState::None,
                mode.clone()
            ),
            vec![
                call("search", json!({"query": "a"})),
                call("get_current_date", json!({}))
            ]
        );
        assert_eq!(
            run(
                &[r#"[{"name":"search","arguments":"#],
                UnifiedParserStartingState::None,
                mode
            ),
            vec![text(r#"[{"name":"search","arguments":"#)]
        );
    }

    #[test]
    fn guided_payload_keeps_reasoning_markers_inside_json_strings() {
        let named = |payload: &str, start| {
            run(
                &[payload],
                start,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("search".into()),
                },
            )
        };
        for marker in [
            "</think>",
            "<think>",
            "</think:opensource>",
            "<think:opensource>",
        ] {
            let query = format!("literal {marker} text");
            let payload = json!({"query": query}).to_string();
            assert_eq!(
                named(&payload, UnifiedParserStartingState::None),
                vec![call("search", json!({"query": query}))],
                "{marker}"
            );
            let after_thought = format!("Thinking.</think:opensource>{payload}");
            assert_eq!(
                named(&after_thought, UnifiedParserStartingState::Reasoning),
                vec![
                    reasoning("Thinking."),
                    call("search", json!({"query": query}))
                ],
                "{marker} after a thought"
            );
        }
        // A generated thought ahead of the payload is still reasoning.
        let payload = json!({"query": "a </think> b"}).to_string();
        let input = format!("\n<think:opensource>Hm.</think:opensource>\n{payload}");
        for at in 1..input.len() {
            if !input.is_char_boundary(at) {
                continue;
            }
            assert_eq!(
                run(
                    &[&input[..at], &input[at..]],
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson {
                        named_tool: Some("search".into()),
                    },
                ),
                vec![
                    reasoning("Hm."),
                    call("search", json!({"query": "a </think> b"}))
                ],
                "split {at}"
            );
        }
    }

    #[test]
    fn prompt_and_batch_starting_state_detection() {
        assert!(prompt_opens_reasoning(
            "…<｜hy_Assistant:opensource｜><think:opensource>"
        ));
        assert!(prompt_opens_reasoning("…<think>\n"));
        assert!(!prompt_opens_reasoning(
            "…<think:opensource></think:opensource>"
        ));
        assert!(!prompt_opens_reasoning("…<｜hy_Assistant:opensource｜>"));

        assert_eq!(
            detect_starting_state("thought</think:opensource>answer"),
            UnifiedParserStartingState::Reasoning
        );
        assert_eq!(
            detect_starting_state("<think:opensource>thought</think:opensource>answer"),
            UnifiedParserStartingState::None
        );
        assert_eq!(
            detect_starting_state("answer"),
            UnifiedParserStartingState::None
        );
    }

    #[test]
    fn reset_returns_unconsumed_text() {
        let mut parser = HunyuanUnifiedParser::new(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("abc <tool_c", &mut output).unwrap();
        assert_eq!(parser.reset(), "<tool_c");
        parser.parse_into("plain", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(assemble(&output.events), vec![text("abc plain")]);
    }
}
