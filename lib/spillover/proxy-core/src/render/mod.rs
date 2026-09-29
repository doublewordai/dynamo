//! Turn a provider's parsed streaming output back into the model's raw output format.
//!
//! Providers return reasoning and tool calls already parsed. Dynamo's frontend runs its own
//! reasoning and tool-call parsers over worker output, so the proxy must emit what the model
//! itself would have emitted (e.g. `<think>...</think>`, native tool-call markup). Each
//! renderer must round-trip: parsing its output with the matching `dynamo-parsers` /
//! `dynamo-parsers-v2` parser gives back the original content, reasoning and tool calls.

use serde::Deserialize;
use serde_json::Value;

pub mod deepseek_v41;
pub mod glm;
pub mod hermes;

/// Parser families used by production models. Matches the frontend's configured parsers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParserFamily {
    /// Reasoning parser `glm45`, tool-call parser `glm47` (GLM-5.x).
    Glm47,
    /// The unified `dynamo-parsers-v2` DeepSeek V4.1 parser.
    DeepseekV41,
    /// Reasoning parser `qwen3`, tool-call parser `hermes` (Qwen3.x).
    Hermes,
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
/// direct "the prompt opened reasoning and it has not ended" signal, forwarded for
/// DeepSeek V4.1. The template kwargs carry the same `thinking` / `enable_thinking`
/// toggle the prompt renderer used.
///
/// Families differ in their default, so the fallback when no signal is present is
/// per-family:
/// - GLM (`glm45`): thinking on unless the request disables it, so the prompt ends
///   with ` thinking` and the parser starts inside.
/// - DeepSeek V4.1: thinking is on by default; starts inside.
/// - Hermes (Qwen3): the template never leaves a bare ` thinking` at the end, so the
///   parser starts outside whether thinking is on or off.
pub fn reasoning_start(family: ParserFamily, extra_args: Option<&Value>) -> ReasoningStart {
    // The direct signal wins: it was computed from the actual rendered prompt.
    if family == ParserFamily::DeepseekV41
        && let Some(ended) = extra_args
            .and_then(|args| args.get("reasoning_ended"))
            .and_then(Value::as_bool)
    {
        return if ended {
            ReasoningStart::Outside
        } else {
            ReasoningStart::InsideReasoning
        };
    }
    if family == ParserFamily::Hermes {
        return ReasoningStart::Outside;
    }
    let kwargs = extra_args
        .and_then(|args| args.get("reasoning_parser_kwargs"))
        .and_then(|parser| parser.get("chat_template_kwargs"));
    let thinking = match family {
        ParserFamily::Glm47 => kwargs.and_then(thinking_bool),
        ParserFamily::DeepseekV41 => kwargs.and_then(deepseek_thinking),
        ParserFamily::Hermes => None,
    };
    match thinking {
        Some(false) => ReasoningStart::Outside,
        _ => ReasoningStart::InsideReasoning,
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
        ParserFamily::Hermes => Box::new(hermes::HermesRenderer::new(start)),
    }
}
