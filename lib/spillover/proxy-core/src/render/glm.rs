// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-5.x: reasoning parser `glm45`, tool-call parser `glm47`.
//!
//! Wire formats (from `dynamo-parsers` 9.1.1):
//! - reasoning: `<think>` / `</think>` around the text. `glm45` maps to
//!   `ReasoningParserType::NemotronDeci`, the non-forced `BasicReasoningParser` shape
//!   (`src/reasoning/mod.rs`).
//! - tool calls: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`.
//!   String arguments are written verbatim; everything else is JSON
//!   (`src/tool_calling/xml/glm47_parser.rs`).
//!
//! The GLM-5 chat template ends the prompt with `<think>` when thinking is enabled (see the
//! model card's `chat_template.jinja`), so the frontend sees
//! `prompt_injected_reasoning_start == true` and calls `set_in_reasoning(true)`. The
//! completion then starts inside the first reasoning block and must not emit an opening
//! `<think>` of its own; only later blocks need one. With thinking disabled the template
//! leaves the channel closed and the completion opens and closes every block itself, so the
//! renderer takes the frontend's actual starting state from [`ReasoningStart`].

use std::collections::BTreeMap;

use serde_json::Value;

use super::{OutputRenderer, ReasoningStart, RenderError};

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

/// One in-flight tool call, assembled from streamed argument fragments.
#[derive(Default)]
struct PendingToolCall {
    name: Option<String>,
    arguments: String,
}

pub struct GlmRenderer {
    /// Prompt-injected opener not yet consumed by reasoning text or a closer.
    injected_open: bool,
    reasoning_open: bool,
    tools: BTreeMap<usize, PendingToolCall>,
}

impl GlmRenderer {
    pub fn new(start: ReasoningStart) -> Self {
        Self {
            injected_open: start == ReasoningStart::InsideReasoning,
            reasoning_open: false,
            tools: BTreeMap::new(),
        }
    }

    fn push_reasoning(&mut self, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        if !self.reasoning_open {
            // The prompt already opened the first block; later blocks need their own opener.
            if !self.injected_open {
                out.push_str(THINK_START);
            }
            self.injected_open = false;
            self.reasoning_open = true;
        }
        out.push_str(text);
    }

    /// Close the open reasoning block, or the prompt-injected empty one if no reasoning arrived.
    fn close_reasoning(&mut self, out: &mut String) {
        if self.reasoning_open {
            out.push_str(THINK_END);
            self.reasoning_open = false;
        } else if self.injected_open {
            out.push_str(THINK_END);
            self.injected_open = false;
        }
    }

    fn push_content(&mut self, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        self.close_reasoning(out);
        out.push_str(text);
    }

    fn push_tool_calls(&mut self, calls: &[Value]) -> Result<(), RenderError> {
        for call in calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
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
        self.close_reasoning(out);
        for (_, call) in std::mem::take(&mut self.tools) {
            out.push_str(&render_tool_call(&call)?);
        }
        Ok(())
    }
}

impl OutputRenderer for GlmRenderer {
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
        if self.reasoning_open {
            out.push_str(THINK_END);
            self.reasoning_open = false;
        }
        Ok(out)
    }
}

fn reasoning_text(delta: &Value) -> Option<&str> {
    ["reasoning_content", "reasoning"]
        .iter()
        .find_map(|key| delta.get(key).and_then(Value::as_str))
        .filter(|text| !text.is_empty())
}

/// Markup the `glm47` parser treats as structural; a value carrying one of these would
/// terminate an element early and corrupt the call.
const GLM_MARKERS: [&str; 6] = [
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
];

fn reject_markers(text: &str, what: &str) -> Result<(), RenderError> {
    if let Some(marker) = GLM_MARKERS.iter().find(|marker| text.contains(**marker)) {
        return Err(RenderError::Unsupported(format!(
            "GLM {what} contains reserved marker {marker}"
        )));
    }
    Ok(())
}

fn render_tool_call(call: &PendingToolCall) -> Result<String, RenderError> {
    let name = call
        .name
        .as_deref()
        .ok_or_else(|| RenderError::Unsupported("tool call without a function name".into()))?;
    reject_markers(name, "tool name")?;
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

    let mut out = String::from("<tool_call>");
    out.push_str(name);
    for (key, value) in arguments {
        reject_markers(key, "argument key")?;
        // The GLM template writes strings verbatim and JSON-encodes everything else.
        let body = match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        reject_markers(&body, "argument value")?;
        out.push_str("<arg_key>");
        out.push_str(key);
        out.push_str("</arg_key><arg_value>");
        out.push_str(&body);
        out.push_str("</arg_value>");
    }
    out.push_str("</tool_call>");
    Ok(out)
}
