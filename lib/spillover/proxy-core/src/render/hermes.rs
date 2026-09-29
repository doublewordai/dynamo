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
//! Unlike GLM, the Qwen3 chat template does not write an opener at `add_generation_prompt`
//! when thinking is enabled (it only emits an empty ` thinking</think>` pair when thinking is
//! disabled). The frontend therefore starts outside reasoning and the completion must emit
//! its own opening ` thinking`. The renderer takes that starting state explicitly so the
//! content-only case stays exact and a hypothetical prompt-injected turn is still handled.

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

#[derive(Default)]
pub struct HermesRenderer {
    /// Prompt-injected opener not yet consumed by reasoning text or a closer.
    injected_open: bool,
    reasoning_open: bool,
    tools: BTreeMap<usize, PendingToolCall>,
}

impl HermesRenderer {
    pub fn new(start: ReasoningStart) -> Self {
        Self {
            injected_open: start == ReasoningStart::InsideReasoning,
            reasoning_open: false,
            tools: BTreeMap::new(),
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

    fn push_tool_calls(&mut self, calls: &[Value], out: &mut String) -> Result<(), RenderError> {
        for call in calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            // A new index means every previously buffered call has finished streaming.
            if !self.tools.contains_key(&index) && !self.tools.is_empty() {
                self.flush_tools(out)?;
            }
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
        let content = delta.get("content").and_then(Value::as_str);
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
            self.push_tool_calls(calls, &mut out)?;
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

fn render_tool_call(call: &PendingToolCall) -> Result<String, RenderError> {
    let name = call
        .name
        .as_deref()
        .ok_or_else(|| RenderError::Unsupported("tool call without a function name".into()))?;
    let arguments = if call.arguments.trim().is_empty() {
        "{}"
    } else {
        call.arguments.as_str()
    };
    let name = serde_json::to_string(name)
        .map_err(|e| RenderError::Unsupported(format!("tool call name is not JSON: {e}")))?;
    Ok(format!(
        "<tool_call>\n{{\"name\": {name}, \"arguments\": {arguments}}}\n</tool_call>"
    ))
}
