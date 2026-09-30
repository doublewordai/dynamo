// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thinking controls: read once from the chat request, mapped per provider.
//!
//! Clients can ask for thinking in several dialects (`thinking: {type}`, `reasoning_effort`,
//! `chat_template_args.enable_thinking` and friends). Dynamo's frontend normalizes them before
//! the proxy sees the request (`NvCreateChatCompletionRequest::normalize_reasoning_template_args`
//! and the chat preprocessor): a decision is written into `chat_template_args` as `thinking` and
//! `enable_thinking` booleans and usually a `thinking_mode` string, `reasoning_effort` is kept as a
//! grade, and `thinking_token_budget` stays a top-level integer. [`ThinkingIntent::from_request`]
//! reads that normalized form with the frontend's own precedence, tolerating the less complete
//! shapes some entry points produce (the Anthropic endpoint sets only `enable_thinking`).
//!
//! Providers spell thinking differently, so each proxy's config maps the intent onto body JSON
//! ([`ThinkingMapping`]). Nothing here is specific to one provider.

use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Placeholder in [`ThinkingMapping::effort`] replaced by the requested effort.
pub const EFFORT_PLACEHOLDER: &str = "{effort}";
/// Placeholder in [`ThinkingMapping::budget`] replaced by the requested budget, as a number.
pub const BUDGET_PLACEHOLDER: &str = "{budget_tokens}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Enabled,
    Disabled,
    /// The client asked the model to decide.
    Adaptive,
}

/// What the client asked for. Every field is `None` when the client (and the deployment's
/// default) decided nothing, in which case the provider's own default applies.
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
            .map(str::to_string);
        let mode = args
            .and_then(mode_from_template_args)
            .or_else(|| request.get("thinking").and_then(mode_from_thinking_field))
            .or_else(|| {
                effort.as_deref().map(|effort| {
                    if effort.eq_ignore_ascii_case("none") {
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

    pub fn is_empty(&self) -> bool {
        self.mode.is_none() && self.effort.is_none() && self.budget_tokens.is_none()
    }
}

/// The frontend's precedence: the `thinking` then `enable_thinking` toggle, then `thinking_mode`.
fn mode_from_template_args(args: &Value) -> Option<ThinkingMode> {
    for key in ["thinking", "enable_thinking"] {
        if let Some(on) = args.get(key).and_then(toggle) {
            return Some(if on {
                ThinkingMode::Enabled
            } else {
                ThinkingMode::Disabled
            });
        }
    }
    match args.get("thinking_mode")? {
        Value::Bool(on) => Some(if *on {
            ThinkingMode::Enabled
        } else {
            ThinkingMode::Disabled
        }),
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
        return Some(if on {
            ThinkingMode::Enabled
        } else {
            ThinkingMode::Disabled
        });
    }
    match thinking.get("type")?.as_str()? {
        "enabled" => Some(ThinkingMode::Enabled),
        "disabled" => Some(ThinkingMode::Disabled),
        "adaptive" => Some(ThinkingMode::Adaptive),
        _ => None,
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

/// How one provider expresses thinking. Each entry is JSON deep-merged into the request body
/// when the request asks for that thing; an absent entry sends nothing for it. `body_overrides`
/// is applied afterwards and wins.
///
/// ```yaml
/// thinking:
///   enabled: {reasoning: {enabled: true}}
///   disabled: {reasoning: {enabled: false}}
///   effort: {reasoning: {effort: "{effort}"}}
///   budget: {reasoning: {max_tokens: "{budget_tokens}"}}
///   require_mapping: true
/// ```
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ThinkingMapping {
    /// Merged when the request turns thinking on.
    pub enabled: Option<Value>,
    /// Merged when the request turns thinking off.
    pub disabled: Option<Value>,
    /// Merged when the request leaves the decision to the model.
    pub adaptive: Option<Value>,
    /// Merged when the request sets an effort; `"{effort}"` becomes the effort string. Defaults
    /// to the OpenAI field, `{"reasoning_effort": "{effort}"}`; set it to `null` to send nothing.
    pub effort: Option<Value>,
    /// Merged when the request sets a thinking budget; `"{budget_tokens}"` becomes the number.
    pub budget: Option<Value>,
    /// Refuse a request whose thinking choice this mapping cannot express, so it is retried on a
    /// hosted worker instead of being answered with the provider's default. Off by default,
    /// because a deployment default thinking mode marks every request as decided.
    pub require_mapping: bool,
}

impl Default for ThinkingMapping {
    fn default() -> Self {
        Self {
            enabled: None,
            disabled: None,
            adaptive: None,
            effort: Some(json!({ "reasoning_effort": EFFORT_PLACEHOLDER })),
            budget: None,
            require_mapping: false,
        }
    }
}

impl ThinkingMapping {
    /// The first part of `intent` this mapping cannot express, if any.
    pub fn unmapped(&self, intent: &ThinkingIntent) -> Option<&'static str> {
        let mode = match intent.mode {
            Some(ThinkingMode::Enabled) if self.enabled.is_none() => Some("thinking enabled"),
            Some(ThinkingMode::Disabled) if self.disabled.is_none() => Some("thinking disabled"),
            Some(ThinkingMode::Adaptive) if self.adaptive.is_none() => Some("adaptive thinking"),
            _ => None,
        };
        // An effort implies a mode, so an effort mapping alone expresses an enabled request.
        let mode = mode.filter(|_| !(intent.effort.is_some() && self.effort.is_some()));
        mode.or_else(|| {
            (intent.effort.is_some() && self.effort.is_none()).then_some("reasoning effort")
        })
        .or_else(|| {
            (intent.budget_tokens.is_some() && self.budget.is_none())
                .then_some("thinking token budget")
        })
    }

    /// Merge the body JSON for `intent` into `body`.
    pub fn apply(&self, intent: &ThinkingIntent, body: &mut Map<String, Value>) {
        let mode = match intent.mode {
            Some(ThinkingMode::Enabled) => self.enabled.as_ref(),
            Some(ThinkingMode::Disabled) => self.disabled.as_ref(),
            Some(ThinkingMode::Adaptive) => self.adaptive.as_ref(),
            None => None,
        };
        if let Some(template) = mode {
            merge(body, template.clone());
        }
        if let (Some(effort), Some(template)) = (&intent.effort, &self.effort) {
            merge(
                body,
                substitute(template, EFFORT_PLACEHOLDER, &Value::from(effort.as_str())),
            );
        }
        if let (Some(budget), Some(template)) = (intent.budget_tokens, &self.budget) {
            merge(
                body,
                substitute(template, BUDGET_PLACEHOLDER, &Value::from(budget)),
            );
        }
    }
}

/// Replace a string that is exactly `placeholder` with `value`; replace it textually inside
/// longer strings.
fn substitute(template: &Value, placeholder: &str, value: &Value) -> Value {
    match template {
        Value::String(text) if text == placeholder => value.clone(),
        Value::String(text) if text.contains(placeholder) => {
            let replacement = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            Value::String(text.replace(placeholder, &replacement))
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| substitute(item, placeholder, value))
                .collect(),
        ),
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, item)| (key.clone(), substitute(item, placeholder, value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Deep-merge `patch` into `body`: objects merge key by key, anything else replaces.
fn merge(body: &mut Map<String, Value>, patch: Value) {
    let Value::Object(patch) = patch else {
        return;
    };
    for (key, value) in patch {
        match (body.get_mut(&key), value) {
            (Some(Value::Object(existing)), Value::Object(incoming)) => {
                merge(existing, Value::Object(incoming));
            }
            (_, value) => {
                body.insert(key, value);
            }
        }
    }
}
