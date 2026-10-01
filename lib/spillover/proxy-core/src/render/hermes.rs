// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.x: reasoning parser `qwen3`, tool-call parser `hermes`.
//!
//! Wire formats (from `dynamo-parsers` 9.1.1):
//! - reasoning: ` thinking` / `</think>` around the text. `qwen3` maps to the shared
//!   `BasicReasoningParser` (`src/reasoning/mod.rs`), non-forced.
//! - tool calls: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` with the
//!   argument object passed through verbatim (`src/tool_calling/config.rs`,
//!   `ToolCallConfig::hermes`).
//!
//! The Qwen3 thinking template writes `assistant\n thinking\n` at `add_generation_prompt`,
//! so the frontend starts inside the first reasoning block and the completion must not
//! emit its own opening ` thinking` (only a closer when content arrives). Thinking off uses
//! a different template whose opener is closed in the prompt, so the parser starts
//! outside. The renderer takes that starting state explicitly.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{CallKey, OutputRenderer, ReasoningStart, RenderError, ToolCallIndex};

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

/// One in-flight tool call, assembled from streamed argument fragments.
#[derive(Default)]
struct PendingToolCall {
    name: Option<String>,
    arguments: String,
}

#[derive(Default)]
pub struct HermesRenderer {
    /// Prompt-injected opener not yet consumed by reasoning text or a closer.
    injected_open: bool,
    reasoning_open: bool,
    tools: BTreeMap<CallKey, PendingToolCall>,
    keyer: ToolCallIndex,
}

impl HermesRenderer {
    pub fn new(start: ReasoningStart) -> Self {
        Self {
            injected_open: start == ReasoningStart::InsideReasoning,
            reasoning_open: false,
            tools: BTreeMap::new(),
            keyer: ToolCallIndex::default(),
        }
    }

    /// Leave reasoning, closing the prompt-injected block if no reasoning was streamed.
    ///
    /// A lone opener would make the parser treat the following text as reasoning, so the
    /// prompt-injected block is closed immediately when content arrives first.
    fn enter_normal(&mut self, out: &mut String) {
        if self.reasoning_open {
            out.push_str(THINK_END);
            self.reasoning_open = false;
        } else if self.injected_open {
            out.push_str(THINK_END);
            self.injected_open = false;
        }
    }

    fn push_reasoning(&mut self, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        if !self.reasoning_open {
            if !self.injected_open {
                out.push_str(THINK_START);
            }
            self.injected_open = false;
            self.reasoning_open = true;
        }
        out.push_str(text);
    }

    fn push_content(&mut self, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        self.enter_normal(out);
        out.push_str(text);
    }

    fn push_tool_calls(&mut self, calls: &[Value]) -> Result<(), RenderError> {
        for call in calls {
            let index = self.keyer.resolve(call);
            // Calls are only flushed at a non-tool boundary or at `finish`: fragments of
            // different indices may interleave, so a new index does not prove the
            // previously buffered calls are complete.
            let entry = self.tools.entry(index).or_default();
            if let Some(function) = call.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    entry.name = Some(name.to_string());
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    entry.arguments.push_str(arguments);
                }
            }
        }
        Ok(())
    }

    fn flush_tools(&mut self, out: &mut String) -> Result<(), RenderError> {
        if self.tools.is_empty() {
            return Ok(());
        }
        self.enter_normal(out);
        for (_, call) in std::mem::take(&mut self.tools) {
            out.push_str(&render_tool_call(&call)?);
        }
        Ok(())
    }
}

impl OutputRenderer for HermesRenderer {
    fn push_delta(&mut self, delta: &Value) -> Result<String, RenderError> {
        let mut out = String::new();
        let reasoning = reasoning_text(delta);
        // Many providers emit `content: ""` on tool-call deltas; an empty string does not
        // end the in-flight call the way real content does.
        let content = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty());
        let tool_calls = delta.get("tool_calls").and_then(Value::as_array);

        // Any non-tool-call field ends the in-flight tool call.
        if reasoning.is_some() || content.is_some() {
            self.flush_tools(&mut out)?;
        }
        if let Some(text) = reasoning {
            self.push_reasoning(text, &mut out);
        }
        if let Some(text) = content {
            self.push_content(text, &mut out);
        }
        if let Some(calls) = tool_calls {
            self.push_tool_calls(calls)?;
        }
        Ok(out)
    }

    fn finish(&mut self, _finish_reason: Option<&str>) -> Result<String, RenderError> {
        let mut out = String::new();
        self.flush_tools(&mut out)?;
        // Close the prompt-injected block even when no reasoning arrived, so an
        // otherwise-empty completion does not end mid-thought.
        self.enter_normal(&mut out);
        Ok(out)
    }
}

fn reasoning_text(delta: &Value) -> Option<&str> {
    ["reasoning_content", "reasoning"].iter().find_map(|key| {
        delta
            .get(key)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
    })
}

fn render_tool_call(call: &PendingToolCall) -> Result<String, RenderError> {
    let name = call
        .name
        .as_deref()
        .ok_or_else(|| RenderError::Unsupported("tool call without a function name".into()))?;
    // The arguments are validated and re-serialized as a JSON object so invalid/truncated
    // JSON is surfaced instead of being spliced through verbatim.
    let arguments: Value = if call.arguments.trim().is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(&call.arguments).map_err(|e| {
            RenderError::Unsupported(format!("tool call arguments are not valid JSON: {e}"))
        })?
    };
    let arguments = arguments.as_object().ok_or_else(|| {
        RenderError::Unsupported("tool call arguments are not a JSON object".into())
    })?;
    let arguments = serde_json::to_string(arguments)
        .map_err(|e| RenderError::Unsupported(format!("tool call arguments are not JSON: {e}")))?;
    // The `hermes` parser terminates the block at the first literal `</tool_call>`. JSON
    // string escapes cannot carry the marker, and the parser's JSON decoder restores the
    // original bytes, so escaping `<`/`>` keeps the call lossless instead of rejecting it.
    let arguments = escape_markers(&arguments);
    let name = serde_json::to_string(name)
        .map_err(|e| RenderError::Unsupported(format!("tool call name is not JSON: {e}")))?;
    let name = escape_markers(&name);
    Ok(format!(
        "<tool_call>\n{{\"name\": {name}, \"arguments\": {arguments}}}\n</tool_call>"
    ))
}

/// Escape `<` and `>` in serialized JSON so it cannot carry a structural marker.
/// `\u003c` / `\u003e` are valid JSON escapes that decode back to the original bytes.
fn escape_markers(json: &str) -> String {
    json.replace('<', "\\u003c").replace('>', "\\u003e")
}
