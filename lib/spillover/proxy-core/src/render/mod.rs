// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turn a provider's parsed streaming output back into the model's raw output format.
//!
//! Providers return reasoning and tool calls already parsed. Dynamo's frontend runs its own
//! reasoning and tool-call parsers over worker output, so the proxy must emit what the model
//! itself would have emitted (e.g. `<think>...</think>`, native tool-call markup). Each
//! renderer must round-trip: parsing its output with the matching `dynamo-parsers` /
//! `dynamo-parsers-v2` parser gives back the original content, reasoning and tool calls.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

pub mod deepseek_v41;
pub mod glm;
pub mod hermes;
pub mod kimi_k3;

/// Parser families used by production models. Matches the frontend's configured parsers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParserFamily {
    /// Reasoning parser `glm45`, tool-call parser `glm47` (GLM-5.x).
    Glm47,
    /// The unified `dynamo-parsers-v2` DeepSeek V4.1 parser.
    DeepseekV41,
    /// Kimi K3 (moonshotai/kimi-k3): reasoning parser `kimi_k3` and the XTML tool-call parser;
    /// the frontend routes K3 through the legacy jail, not the unified parser.
    KimiK3,
    /// Reasoning parser `qwen3`, tool-call parser `hermes` (Qwen3.x).
    Hermes,
}

impl ParserFamily {
    /// The frontend tool-call parser that reads this family's output. Primary workers of the
    /// same model, on any engine, register the same name (`--dyn-tool-call-parser`).
    pub fn tool_call_parser(self) -> &'static str {
        match self {
            ParserFamily::Glm47 => "glm47",
            ParserFamily::DeepseekV41 => "deepseek_v41",
            ParserFamily::KimiK3 => "kimi_k3",
            ParserFamily::Hermes => "hermes",
        }
    }

    /// The frontend reasoning parser that reads this family's output (`--dyn-reasoning-parser`).
    pub fn reasoning_parser(self) -> &'static str {
        match self {
            ParserFamily::Glm47 => "glm45",
            ParserFamily::DeepseekV41 => "deepseek_v41",
            ParserFamily::KimiK3 => "kimi_k3",
            ParserFamily::Hermes => "qwen3",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("delta cannot be rendered: {0}")]
    Unsupported(String),
}

pub trait OutputRenderer: Send {
    /// Render one OpenAI streaming delta (`choices[0].delta`). Reads `content`, `reasoning` or
    /// `reasoning_content`, and `tool_calls` (streamed by index, arguments in pieces).
    /// Returns raw model-format text to emit now; may be empty while buffering.
    fn push_delta(&mut self, delta: &Value) -> Result<String, RenderError>;

    /// Close open sections (reasoning, tool calls) at stream end and return the remaining text.
    fn finish(&mut self, finish_reason: Option<&str>) -> Result<String, RenderError>;
}

/// Which reasoning state the frontend's parser starts in for one request.
///
/// Dynamo's preprocessor decides this from the rendered prompt: a template that ends
/// with the reasoning opener leaves the parser inside the first block, so the
/// completion must not open it again. A template that opens nothing (or closes
/// immediately) leaves the parser outside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningStart {
    /// The prompt already opened a reasoning block; the completion starts inside it.
    InsideReasoning,
    /// The prompt opened nothing; the completion must emit its own opener to reason.
    Outside,
}

/// Derive the parser's starting state from the worker's `extra_args`.
///
/// Dynamo's preprocessor forwards two signals when a reasoning parser is configured
/// (`OpenAIPreprocessor::backend_extra_args`): `reasoning_ended` and
/// `reasoning_parser_kwargs.chat_template_kwargs`. `reasoning_ended == false` is the
/// direct "the prompt opened reasoning and it has not ended" signal, computed from the
/// actual rendered prompt. The template kwargs carry the same `thinking` /
/// `enable_thinking` toggle the prompt renderer used.
///
/// Families differ in their default, so the fallback when no signal is present is
/// per-family:
/// - GLM (`glm45`): thinking on unless the request disables it, so the prompt ends
///   with ` thinking` and the parser starts inside.
/// - DeepSeek V4.1: thinking is on by default; starts inside.
/// - Kimi K3: thinking is on by default; starts inside.
/// - Hermes (Qwen3): the thinking template writes `assistant\n thinking\n` at
///   `add_generation_prompt`, so the parser starts inside; thinking off uses a different
///   template whose opener is closed in the prompt. There is no family-level default to
///   infer from request args alone, so absent any signal the parser starts outside.
pub fn reasoning_start(family: ParserFamily, extra_args: Option<&Value>) -> ReasoningStart {
    // The direct signal wins: it was computed from the actual rendered prompt.
    if matches!(
        family,
        ParserFamily::DeepseekV41 | ParserFamily::KimiK3 | ParserFamily::Hermes
    ) && let Some(ended) = extra_args
        .and_then(|args| args.get("reasoning_ended"))
        .and_then(Value::as_bool)
    {
        return if ended {
            ReasoningStart::Outside
        } else {
            ReasoningStart::InsideReasoning
        };
    }
    let kwargs = extra_args
        .and_then(|args| args.get("reasoning_parser_kwargs"))
        .and_then(|parser| parser.get("chat_template_kwargs"));
    let thinking = match family {
        ParserFamily::Glm47 => kwargs.and_then(thinking_bool),
        ParserFamily::DeepseekV41 => kwargs.and_then(deepseek_thinking),
        ParserFamily::KimiK3 => kwargs.and_then(thinking_bool),
        ParserFamily::Hermes => kwargs.and_then(thinking_bool),
    };
    match (family, thinking) {
        // Hermes has no family-level default: the Qwen3 templates differ, so without an
        // explicit toggle or the direct signal the parser's start is unknown. Keep the
        // conservative outside default.
        (ParserFamily::Hermes, None) => ReasoningStart::Outside,
        (_, Some(false)) => ReasoningStart::Outside,
        _ => ReasoningStart::InsideReasoning,
    }
}

/// The reasoning opener the frontend's parser looks for at the end of the rendered prompt
/// (`OpenAIPreprocessor::prompt_injected_reasoning_start` in `lib/llm/src/preprocessor.rs`).
pub fn reasoning_opener(family: ParserFamily) -> &'static str {
    match family {
        ParserFamily::KimiK3 => "<|open|>think<|sep|>",
        ParserFamily::Glm47 | ParserFamily::DeepseekV41 | ParserFamily::Hermes => "<think>",
    }
}

/// Derive the parser's starting state from the end of the rendered prompt, by the frontend's own
/// rule: the parser starts inside reasoning exactly when the prompt, trimmed of trailing
/// whitespace, ends with the opener. This is authoritative for every family, including those for
/// which Dynamo forwards no signal in `extra_args` (Qwen3 / Hermes, GLM).
pub fn reasoning_start_from_prompt(family: ParserFamily, prompt_tail: &str) -> ReasoningStart {
    if prompt_tail.trim_end().ends_with(reasoning_opener(family)) {
        ReasoningStart::InsideReasoning
    } else {
        ReasoningStart::Outside
    }
}

/// Assigns a stable buffer key to streamed tool-call fragments.
///
/// OpenAI always streams `index`; a provider that omits it would otherwise collapse every
/// call into key `0`, concatenating parallel calls into one corrupted call. Fragments are
/// keyed by `index` when present, by `id` when it matches a known call, and otherwise a
/// name-bearing fragment starts a new call while an argument-only fragment continues the
/// most recent one.
#[derive(Default)]
pub(crate) struct ToolCallIndex {
    by_id: BTreeMap<String, usize>,
    next: usize,
    last: Option<usize>,
}

impl ToolCallIndex {
    pub(crate) fn resolve(&mut self, call: &Value) -> usize {
        if let Some(index) = call.get("index").and_then(Value::as_u64) {
            let index = index as usize;
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                self.by_id.insert(id.to_string(), index);
            }
            self.next = self.next.max(index + 1);
            self.last = Some(index);
            return index;
        }
        if let Some(id) = call.get("id").and_then(Value::as_str) {
            if let Some(&key) = self.by_id.get(id) {
                self.last = Some(key);
                return key;
            }
            let key = self.next;
            self.next += 1;
            self.by_id.insert(id.to_string(), key);
            self.last = Some(key);
            return key;
        }
        let starts_call = call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .is_some_and(|name| !name.is_empty());
        if starts_call {
            let key = self.next;
            self.next += 1;
            self.last = Some(key);
            key
        } else {
            // An argument-only fragment continues the most recent call; if there is none
            // (a provider that dropped the opening fragment) start a fresh one.
            match self.last {
                Some(last) => last,
                None => {
                    let key = self.next;
                    self.next += 1;
                    self.last = Some(key);
                    key
                }
            }
        }
    }
}

/// `thinking` / `enable_thinking` as a bool, matching `dynamo-renderer`'s helper.
fn thinking_bool(kwargs: &Value) -> Option<bool> {
    ["thinking", "enable_thinking"]
        .iter()
        .find_map(|key| kwargs.get(*key).and_then(Value::as_bool))
}

/// DeepSeek additionally honors `thinking_mode`: `chat` turns reasoning off,
/// `thinking` turns it on; anything else keeps the family default.
fn deepseek_thinking(kwargs: &Value) -> Option<bool> {
    if let Some(enabled) = thinking_bool(kwargs) {
        return Some(enabled);
    }
    match kwargs.get("thinking_mode").and_then(Value::as_str) {
        Some("chat") => Some(false),
        Some("thinking") => Some(true),
        _ => None,
    }
}

pub fn renderer_for(family: ParserFamily, start: ReasoningStart) -> Box<dyn OutputRenderer> {
    match family {
        ParserFamily::Glm47 => Box::new(glm::GlmRenderer::new(start)),
        ParserFamily::DeepseekV41 => Box::new(deepseek_v41::DeepseekV41Renderer::new(start)),
        ParserFamily::KimiK3 => Box::new(kimi_k3::KimiK3Renderer::new(start)),
        ParserFamily::Hermes => Box::new(hermes::HermesRenderer::new(start)),
    }
}
