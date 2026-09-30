// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 (moonshotai/kimi-k3): the XTML output grammar.
//!
//! Wire format, matching Moonshot's `encoding_k3.py` and the fork's
//! `dynamo-parsers` v1 `xtml::kimi_k3_parser` and `dynamo-parsers-v2`
//! unified `kimi_k3` parser:
//!
//! ```text
//! <|open|>think<|sep|>REASONING<|close|>think<|sep|>
//! <|open|>response<|sep|>CONTENT<|close|>response<|sep|>
//! <|open|>tools<|sep|>
//!   <|open|>call tool="NAME" index="N"<|sep|>
//!     <|open|>argument key="K" type="string"<|sep|>VALUE<|close|>argument<|sep|>
//!     <|open|>argument key="K2" type="object"<|sep|>{"nested":true}<|close|>argument<|sep|>
//!   <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! ```
//!
//! Argument values follow the native prompt renderer
//! (`dynamo-renderer`'s `renderer/src/kimi_k3.rs`, `xtml_type`/`xtml_value`):
//! `type="string"` writes the string verbatim, every other JSON value is
//! compact JSON under its own type (`number`, `boolean`, `null`, `array`,
//! `object`). Attribute values are escaped as `&amp;` / `&quot;`, which the
//! parser's `unescape_attr` reverses.
//!
//! ## Prompt-opened reasoning
//!
//! The K3 chat template's generation prompt ends with `<|open|>think<|sep|>`
//! when thinking is enabled and `<|open|>response<|sep|>` when it is disabled
//! (see the renderer's `add_generation_prompt` branch). Dynamo detects the
//! think opener (`prompt_injected_reasoning_start`) and starts the reasoning
//! parser inside the first block, so the completion must not open it again.
//! With thinking off the prompt already opened the response channel, so the
//! completion must not emit `<|open|>response<|sep|>` either — the response
//! mode would treat it as ordinary text.
//!
//! The renderer therefore tracks both channels: `injected_open` mirrors the
//! prompt's think opener, and `response_open` starts true when the prompt
//! opened the response channel. Every block is closed before the next one
//! opens, so the emitted stream is the exact model-native form.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{OutputRenderer, ReasoningStart, RenderError, ToolCallIndex};

const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const CALL_OPEN: &str = "<|open|>call tool=\"";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARG_OPEN: &str = "<|open|>argument key=\"";
const ARG_CLOSE: &str = "<|close|>argument<|sep|>";
const SEP: &str = "<|sep|>";

/// One in-flight tool call, assembled from streamed OpenAI argument fragments.
#[derive(Default)]
struct PartialCall {
    name: Option<String>,
    arguments: String,
}

pub struct KimiK3Renderer {
    /// The prompt opened a think channel that no completion text has consumed yet.
    injected_open: bool,
    /// A reasoning block is open in the rendered stream.
    reasoning_open: bool,
    /// A response channel is open in the rendered stream, or the prompt opened one.
    response_open: bool,
    /// Calls are removed as they are flushed; the map keeps index order.
    calls: BTreeMap<usize, PartialCall>,
    keyer: ToolCallIndex,
}

impl KimiK3Renderer {
    pub fn new(start: ReasoningStart) -> Self {
        let injected_open = start == ReasoningStart::InsideReasoning;
        Self {
            injected_open,
            reasoning_open: false,
            // With thinking off the prompt ends in the response channel; its
            // opener must not be re-emitted.
            response_open: !injected_open,
            calls: BTreeMap::new(),
            keyer: ToolCallIndex::default(),
        }
    }

    fn push_reasoning(&mut self, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        if !self.reasoning_open {
            // The prompt already opened the first block; later blocks need their own opener.
            if !self.injected_open {
                out.push_str(THINK_OPEN);
            }
            self.injected_open = false;
            self.reasoning_open = true;
        }
        out.push_str(text);
    }

    /// Close the open reasoning block, or the prompt-injected empty one.
    fn close_reasoning(&mut self, out: &mut String) {
        if self.reasoning_open {
            out.push_str(THINK_CLOSE);
            self.reasoning_open = false;
        } else if self.injected_open {
            out.push_str(THINK_CLOSE);
            self.injected_open = false;
        }
    }

    /// Open the response channel if it is not already open; the prompt may have.
    fn ensure_response_open(&mut self, out: &mut String) {
        if !self.response_open {
            out.push_str(RESPONSE_OPEN);
            self.response_open = true;
        }
    }

    /// Close the response channel: the one this renderer opened, or the one the
    /// prompt opened with thinking off. The parser consumes the matching closer
    /// in either starting state.
    fn close_response(&mut self, out: &mut String) {
        if self.response_open {
            out.push_str(RESPONSE_CLOSE);
            self.response_open = false;
        }
    }

    fn push_content(&mut self, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        self.close_reasoning(out);
        self.ensure_response_open(out);
        out.push_str(text);
    }

    fn push_tool_calls(&mut self, calls: &[Value]) -> Result<(), RenderError> {
        for call in calls {
            let index = self.keyer.resolve(call);
            // All calls are buffered and emitted in one native tools block at
            // the next non-tool field or at finish, so calls that stream
            // interleaved by index keep their order.
            let entry = self.calls.entry(index).or_default();
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
        if self.calls.is_empty() {
            return Ok(());
        }
        self.close_reasoning(out);
        self.close_response(out);
        out.push_str(TOOLS_OPEN);
        for (position, (_, call)) in std::mem::take(&mut self.calls).into_iter().enumerate() {
            out.push_str(&render_tool_call(&call, position + 1)?);
        }
        out.push_str(TOOLS_CLOSE);
        Ok(())
    }
}

impl OutputRenderer for KimiK3Renderer {
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
        self.close_response(&mut out);
        self.close_reasoning(&mut out);
        Ok(out)
    }
}

fn reasoning_text(delta: &Value) -> Option<&str> {
    ["reasoning_content", "reasoning"]
        .iter()
        .find_map(|key| delta.get(key).and_then(Value::as_str))
        .filter(|text| !text.is_empty())
}

/// Render one complete native call: one `argument` element per top-level key.
fn render_tool_call(call: &PartialCall, index: usize) -> Result<String, RenderError> {
    let name = call.name.as_deref().ok_or_else(|| {
        RenderError::Unsupported("Kimi K3 tool call without a function name".into())
    })?;
    let arguments = parse_arguments(&call.arguments)?;

    let mut out = String::new();
    out.push_str(CALL_OPEN);
    out.push_str(&escape_attr(name));
    out.push_str("\" index=\"");
    out.push_str(&index.to_string());
    out.push('"');
    out.push_str(SEP);
    for (key, value) in arguments {
        out.push_str(ARG_OPEN);
        out.push_str(&escape_attr(&key));
        out.push_str("\" type=\"");
        out.push_str(xtml_type(&value));
        out.push('"');
        out.push_str(SEP);
        out.push_str(&xtml_value(&value)?);
        out.push_str(ARG_CLOSE);
    }
    out.push_str(CALL_CLOSE);
    Ok(out)
}

/// Parse accumulated OpenAI function arguments into ordered key/value pairs.
///
/// OpenAI always sends a JSON object string. An empty string is the empty
/// object, which is how a no-argument call arrives.
fn parse_arguments(arguments: &str) -> Result<Vec<(String, Value)>, RenderError> {
    if arguments.trim().is_empty() {
        return Ok(Vec::new());
    }
    let value: Value = serde_json::from_str(arguments).map_err(|error| {
        RenderError::Unsupported(format!("Kimi K3 tool arguments are not JSON: {error}"))
    })?;
    let Value::Object(object) = value else {
        return Err(RenderError::Unsupported(
            "Kimi K3 tool arguments are not a JSON object".into(),
        ));
    };
    Ok(object.into_iter().collect())
}

/// The `type` attribute for one argument value, from the native prompt renderer.
fn xtml_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

/// The structural XTML tokens; an argument body carrying one would terminate its element
/// early or open a nested one.
const XTML_MARKERS: [&str; 3] = ["<|open|>", "<|close|>", "<|sep|>"];

/// The argument body: strings verbatim, everything else compact JSON.
fn xtml_value(value: &Value) -> Result<String, RenderError> {
    let body = match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).map_err(|error| {
            RenderError::Unsupported(format!("Kimi K3 argument is not JSON: {error}"))
        })?,
    };
    if let Some(marker) = XTML_MARKERS.iter().find(|marker| body.contains(**marker)) {
        return Err(RenderError::Unsupported(format!(
            "Kimi K3 argument value contains reserved marker {marker}"
        )));
    }
    Ok(body)
}

/// Escape an attribute value the way the native prompt renderer does.
fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}
