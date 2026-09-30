// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4.1: the unified `dynamo-parsers-v2` parser (reasoning, text and tool calls).
//!
//! The wire format below is taken from `dynamo-parsers-v2` 0.5.3
//! `src/unified/deepseek_v41.rs`, which is the parser Dynamo's frontend runs for
//! family `deepseek_v41`. It is a unified state machine: one stream owns the
//! `<think>...</think>` reasoning span, visible text and the tool-call markup
//! together.
//!
//! The prompt renderer (`DeepSeekV41Formatter`) ends the rendered prompt with
//! `<think>` when thinking is enabled and with `</think>` when it is disabled.
//! The frontend therefore starts the parser in
//! `UnifiedParserStartingState::Reasoning` for the thinking-on case and in
//! `Response` for thinking off (see `unified_parser::stream_prefill`). The renderer
//! follows that state: it emits the opener only when the parser starts outside, so
//! the same provider deltas round-trip from either state.
//!
//! Tool calls use the parser's native grammar (every marker wraps `DSML` in the
//! fullwidth bar `｜`, U+FF5C):
//!
//! ```text
//! <｜DSML｜ calls>
//! <｜DSML｜ invoke name="NAME">
//! <｜DSML｜ parameter name="KEY" string="true">raw string value</｜DSML｜ parameter>
//! <｜DSML｜ parameter name="OTHER" string="false">{"compact":"json"}</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! </｜DSML｜ calls>
//! ```
//!
//! `string="true"` values are taken literally, so a string is written raw; every
//! other JSON value is written as compact JSON under `string="false"`. No
//! whitespace is inserted around the markup because any byte outside a block
//! becomes visible content, which would corrupt the provider's `content`.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{OutputRenderer, ReasoningStart, RenderError, ToolCallIndex};

/// DeepSeek's reserved guard token wraps `DSML` in U+FF5C (`｜`).
/// `<｜DSML｜ calls>`.
const BLOCK_OPEN: &str = "<\u{ff5c}DSML\u{ff5c} calls>";
/// `</｜DSML｜ calls>`.
const BLOCK_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c} calls>";
/// `<｜DSML｜ invoke name="`.
const INVOKE_OPEN: &str = "<\u{ff5c}DSML\u{ff5c} invoke name=\"";
/// `</｜DSML｜ invoke>`.
const INVOKE_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c} invoke>";
/// `<｜DSML｜ parameter name="`.
const PARAM_OPEN: &str = "<\u{ff5c}DSML\u{ff5c} parameter name=\"";
/// `</｜DSML｜ parameter>`.
const PARAM_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c} parameter>";

/// One tool call assembled from OpenAI streaming fragments.
#[derive(Default)]
struct PartialCall {
    name: String,
    arguments: String,
}

pub struct DeepseekV41Renderer {
    /// Prompt-injected opener not yet consumed by reasoning text or a closer.
    injected_open: bool,
    /// A reasoning block is currently open in the rendered stream.
    reasoning_open: bool,
    /// Complete calls are removed as they are flushed; the map orders them.
    calls: BTreeMap<usize, PartialCall>,
    keyer: ToolCallIndex,
}

impl DeepseekV41Renderer {
    pub fn new(start: ReasoningStart) -> Self {
        Self {
            injected_open: start == ReasoningStart::InsideReasoning,
            reasoning_open: false,
            calls: BTreeMap::new(),
            keyer: ToolCallIndex::default(),
        }
    }

    /// Close an open thought exactly once, before content or a tool block.
    fn close_reasoning(&mut self, out: &mut String) {
        if self.reasoning_open {
            out.push_str("</think>");
            self.reasoning_open = false;
        } else if self.injected_open {
            out.push_str("</think>");
            self.injected_open = false;
        }
    }

    /// Render the buffered calls as one native block, in index order.
    fn render_call_block(
        &mut self,
        calls: Vec<(usize, PartialCall)>,
    ) -> Result<String, RenderError> {
        if calls.is_empty() {
            return Ok(String::new());
        }
        let mut out = String::new();
        self.close_reasoning(&mut out);
        out.push_str(BLOCK_OPEN);
        for (_, call) in calls {
            if call.name.is_empty() {
                return Err(RenderError::Unsupported(
                    "DeepSeek V4.1 tool call has no function name".into(),
                ));
            }
            if call.name.contains('"') || contains_dsml_marker(&call.name) {
                return Err(RenderError::Unsupported(
                    "DeepSeek V4.1 tool call name contains a reserved marker".into(),
                ));
            }
            out.push_str(INVOKE_OPEN);
            out.push_str(&call.name);
            out.push_str("\">");
            for (name, value) in parse_arguments(&call.arguments)? {
                out.push_str(&render_parameter(&name, &value)?);
            }
            out.push_str(INVOKE_CLOSE);
        }
        out.push_str(BLOCK_CLOSE);
        Ok(out)
    }

    fn absorb_tool_calls(&mut self, fragments: &[Value]) -> Result<(), RenderError> {
        for fragment in fragments {
            let index = self.keyer.resolve(fragment);
            // Fragments of different indices may interleave, so nothing is flushed here;
            // the complete set is rendered at `finish` (or not at all until then).
            let call = self.calls.entry(index).or_default();
            if let Some(name) = fragment
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                && !name.is_empty()
            {
                call.name = name.to_string();
            }
            if let Some(arguments) = fragment
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
            {
                call.arguments.push_str(arguments);
            }
        }
        Ok(())
    }

    fn push_reasoning(&mut self, out: &mut String, text: &str) {
        if !self.reasoning_open {
            // The prompt already opened the first block; later blocks need their own opener.
            if !self.injected_open {
                out.push_str("<think>");
            }
            self.injected_open = false;
            self.reasoning_open = true;
        }
        out.push_str(text);
    }
}

impl OutputRenderer for DeepseekV41Renderer {
    fn push_delta(&mut self, delta: &Value) -> Result<String, RenderError> {
        let mut out = String::new();

        let reasoning = delta
            .get("reasoning")
            .or_else(|| delta.get("reasoning_content"))
            .and_then(Value::as_str);
        if let Some(text) = reasoning.filter(|text| !text.is_empty()) {
            self.push_reasoning(&mut out, text);
        }

        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            self.close_reasoning(&mut out);
            out.push_str(text);
        }

        if let Some(fragments) = delta.get("tool_calls").and_then(Value::as_array) {
            self.absorb_tool_calls(fragments)?;
        }

        Ok(out)
    }

    fn finish(&mut self, _finish_reason: Option<&str>) -> Result<String, RenderError> {
        let mut out = String::new();
        let remaining: Vec<(usize, PartialCall)> = self
            .calls
            .iter()
            .map(|(i, call)| {
                (
                    *i,
                    PartialCall {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                )
            })
            .collect();
        self.calls.clear();
        out.push_str(&self.render_call_block(remaining)?);
        self.close_reasoning(&mut out);
        Ok(out)
    }
}

/// Parse accumulated OpenAI function arguments into ordered parameters.
///
/// OpenAI always sends a JSON object string. An empty string is the empty object,
/// which is how a no-argument call arrives.
fn parse_arguments(arguments: &str) -> Result<Vec<(String, Value)>, RenderError> {
    if arguments.trim().is_empty() {
        return Ok(Vec::new());
    }
    let value: Value = serde_json::from_str(arguments).map_err(|error| {
        RenderError::Unsupported(format!(
            "DeepSeek V4.1 tool arguments are not JSON: {error}"
        ))
    })?;
    let Value::Object(object) = value else {
        return Err(RenderError::Unsupported(
            "DeepSeek V4.1 tool arguments are not a JSON object".into(),
        ));
    };
    Ok(object.into_iter().collect())
}

/// Render one `<parameter>`; strings verbatim, everything else as compact JSON.
fn render_parameter(name: &str, value: &Value) -> Result<String, RenderError> {
    if name.contains('"') || contains_dsml_marker(name) {
        return Err(RenderError::Unsupported(
            "DeepSeek V4.1 parameter name contains a reserved marker".into(),
        ));
    }
    let (string, body) = match value {
        Value::String(text) => ("true", text.clone()),
        other => (
            "false",
            serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
        ),
    };
    // A body carrying a closing marker would terminate the parameter (or the enclosing
    // call/block) early and break the unified parser.
    if contains_dsml_marker(&body) {
        return Err(RenderError::Unsupported(
            "DeepSeek V4.1 parameter value contains a reserved marker".into(),
        ));
    }
    Ok(format!(
        "{PARAM_OPEN}{name}\" string=\"{string}\">{body}{PARAM_CLOSE}"
    ))
}

/// Any DSML marker that would end a parameter, invocation or block.
fn contains_dsml_marker(text: &str) -> bool {
    text.contains(PARAM_CLOSE) || text.contains(INVOKE_CLOSE) || text.contains(BLOCK_CLOSE)
}
