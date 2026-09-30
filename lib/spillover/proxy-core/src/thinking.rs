// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thinking controls: read once from the chat request, sent in one of a few fixed dialects.
//!
//! Clients ask for thinking in several ways (`thinking: {type}`, `reasoning_effort`,
//! `chat_template_args.enable_thinking` and friends). Dynamo's frontend normalizes them before
//! the proxy sees the request (`NvCreateChatCompletionRequest::normalize_reasoning_template_args`
//! and the chat preprocessor): a decision is written into `chat_template_args` as `thinking` and
//! `enable_thinking` booleans and usually a `thinking_mode` string, `reasoning_effort` is kept as a
//! grade, and `thinking_token_budget` stays a top-level integer. [`ThinkingIntent::from_request`]
//! reads that normalized form with the frontend's precedence.
//!
//! Providers spell thinking in a handful of ways. Each is a [`ThinkingDialect`] with a fixed
//! translation, so a proxy config names a dialect instead of writing JSON, and every translation
//! is tested in one place.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Enabled,
    Disabled,
    /// The client asked the model to decide.
    Adaptive,
}

/// What the client asked for. Every field is `None` when neither the client nor the deployment's
/// default decided anything, in which case the provider's own default applies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkingIntent {
    pub mode: Option<ThinkingMode>,
    pub effort: Option<String>,
    pub budget_tokens: Option<u64>,
}

impl ThinkingIntent {
    /// Read the intent from the chat request the frontend attached.
    pub fn from_request(request: &Value) -> Self {
        let args = request.get("chat_template_args");
        let effort = request
            .get("reasoning_effort")
            .filter(|value| !value.is_null())
            .or_else(|| args.and_then(|args| args.get("reasoning_effort")))
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase);
        let mode = args
            .and_then(mode_from_template_args)
            .or_else(|| request.get("thinking").and_then(mode_from_thinking_field))
            .or_else(|| {
                effort.as_deref().map(|effort| {
                    if effort == "none" {
                        ThinkingMode::Disabled
                    } else {
                        ThinkingMode::Enabled
                    }
                })
            });
        let budget_tokens = request.get("thinking_token_budget").and_then(Value::as_u64);
        Self {
            mode,
            effort,
            budget_tokens,
        }
    }
}

/// The frontend's precedence: the `thinking` then `enable_thinking` toggle, then `thinking_mode`.
fn mode_from_template_args(args: &Value) -> Option<ThinkingMode> {
    for key in ["thinking", "enable_thinking"] {
        if let Some(on) = args.get(key).and_then(toggle) {
            return Some(on_off(on));
        }
    }
    match args.get("thinking_mode")? {
        Value::Bool(on) => Some(on_off(*on)),
        Value::String(mode) => match mode.to_ascii_lowercase().as_str() {
            "enabled" | "thinking" | "true" => Some(ThinkingMode::Enabled),
            "disabled" | "chat" | "false" => Some(ThinkingMode::Disabled),
            "adaptive" => Some(ThinkingMode::Adaptive),
            _ => None,
        },
        _ => None,
    }
}

/// The raw `thinking` field, which the frontend normally folds into `chat_template_args`.
fn mode_from_thinking_field(thinking: &Value) -> Option<ThinkingMode> {
    if let Some(on) = thinking.as_bool() {
        return Some(on_off(on));
    }
    match thinking.get("type")?.as_str()? {
        "enabled" => Some(ThinkingMode::Enabled),
        "disabled" => Some(ThinkingMode::Disabled),
        "adaptive" => Some(ThinkingMode::Adaptive),
        _ => None,
    }
}

fn on_off(on: bool) -> ThinkingMode {
    if on {
        ThinkingMode::Enabled
    } else {
        ThinkingMode::Disabled
    }
}

fn toggle(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(on) => Some(*on),
        Value::Number(number) => number.as_f64().map(|n| n != 0.0),
        Value::String(text) => match text.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" | "enabled" => Some(true),
            "false" | "0" | "no" | "off" | "disabled" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// How a provider expects thinking to be requested.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDialect {
    /// OpenAI's `reasoning_effort` grade. Thinking off is expressed only as effort `none`.
    #[default]
    ReasoningEffort,
    /// A `reasoning` object: `{enabled, effort, max_tokens}`. A budget is sent instead of the
    /// effort when both are set, because the two are alternatives in this dialect.
    ReasoningObject,
    /// `chat_template_kwargs` for providers that run SGLang or vLLM: `enable_thinking` and
    /// `thinking` booleans. No effort or budget.
    ChatTemplateKwargs,
    /// Send nothing; the provider's default applies.
    None,
}

/// The body fields for an intent in a dialect, and the parts of the intent it could not express.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Translation {
    pub fields: Map<String, Value>,
    pub unexpressed: Vec<&'static str>,
}

impl ThinkingDialect {
    /// Translate `intent`. Adaptive always sends nothing: the model decides, which is the
    /// provider's default.
    pub fn translate(self, intent: &ThinkingIntent) -> Translation {
        let mut out = Translation::default();
        if intent.mode == Some(ThinkingMode::Adaptive) {
            // Any effort or budget would switch thinking on at the provider instead of leaving
            // the choice to the model, so they are reported, not sent.
            if intent.effort.is_some() {
                out.unexpressed.push("reasoning effort");
            }
            if intent.budget_tokens.is_some() {
                out.unexpressed.push("thinking token budget");
            }
            return out;
        }
        let explicit_off = intent.mode == Some(ThinkingMode::Disabled);
        let explicit_on = intent.mode == Some(ThinkingMode::Enabled);
        match self {
            ThinkingDialect::ReasoningEffort => {
                // An explicit toggle outranks the grade, as in the frontend.
                if explicit_off {
                    out.fields.insert("reasoning_effort".into(), json!("none"));
                } else if let Some(effort) = &intent.effort {
                    out.fields.insert("reasoning_effort".into(), json!(effort));
                } else if explicit_on {
                    out.unexpressed.push("thinking enabled");
                }
                if intent.budget_tokens.is_some() && !explicit_off {
                    out.unexpressed.push("thinking token budget");
                }
            }
            ThinkingDialect::ReasoningObject => {
                let mut reasoning = Map::new();
                if explicit_on || explicit_off {
                    reasoning.insert("enabled".into(), json!(explicit_on));
                }
                if !explicit_off {
                    if let Some(budget) = intent.budget_tokens {
                        reasoning.insert("max_tokens".into(), json!(budget));
                    } else if let Some(effort) = &intent.effort {
                        reasoning.insert("effort".into(), json!(effort));
                    }
                }
                if !reasoning.is_empty() {
                    out.fields
                        .insert("reasoning".into(), Value::Object(reasoning));
                }
            }
            ThinkingDialect::ChatTemplateKwargs => {
                if explicit_on || explicit_off {
                    out.fields.insert(
                        "chat_template_kwargs".into(),
                        json!({"enable_thinking": explicit_on, "thinking": explicit_on}),
                    );
                }
                if intent.effort.is_some() && !explicit_off {
                    out.unexpressed.push("reasoning effort");
                }
                if intent.budget_tokens.is_some() && !explicit_off {
                    out.unexpressed.push("thinking token budget");
                }
            }
            ThinkingDialect::None => {
                if explicit_on {
                    out.unexpressed.push("thinking enabled");
                }
                if explicit_off {
                    out.unexpressed.push("thinking disabled");
                }
                if intent.effort.is_some() && !explicit_off {
                    out.unexpressed.push("reasoning effort");
                }
                if intent.budget_tokens.is_some() && !explicit_off {
                    out.unexpressed.push("thinking token budget");
                }
            }
        }
        out
    }
}
