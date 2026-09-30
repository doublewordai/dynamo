// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading the thinking intent from the normalized chat request and mapping it per provider.

use dw_proxy_core::thinking::{ThinkingIntent, ThinkingMapping, ThinkingMode};
use serde_json::{Map, Value, json};

fn intent(request: Value) -> ThinkingIntent {
    ThinkingIntent::from_request(&request)
}

#[test]
fn reads_the_frontends_normalized_toggles() {
    // What `normalize_reasoning_template_args` writes for `thinking: {type: disabled}`.
    let off = intent(json!({"chat_template_args": {
        "thinking": false, "enable_thinking": false, "thinking_mode": "disabled"
    }}));
    assert_eq!(off.mode, Some(ThinkingMode::Disabled));
    // The Anthropic endpoint writes only `enable_thinking`.
    let on = intent(json!({"chat_template_args": {"enable_thinking": true}}));
    assert_eq!(on.mode, Some(ThinkingMode::Enabled));
    // `adaptive` clears the toggles and keeps the mode string.
    let adaptive = intent(json!({"chat_template_args": {"thinking_mode": "adaptive"}}));
    assert_eq!(adaptive.mode, Some(ThinkingMode::Adaptive));
    // The `thinking` toggle wins over `enable_thinking`, as in the frontend.
    let first = intent(json!({"chat_template_args": {"thinking": true, "enable_thinking": false}}));
    assert_eq!(first.mode, Some(ThinkingMode::Enabled));
    assert!(intent(json!({"messages": []})).is_empty());
}

#[test]
fn effort_and_budget_are_read_and_imply_a_mode() {
    let high = intent(json!({"reasoning_effort": "high", "thinking_token_budget": 2048}));
    assert_eq!(high.mode, Some(ThinkingMode::Enabled));
    assert_eq!(high.effort.as_deref(), Some("high"));
    assert_eq!(high.budget_tokens, Some(2048));
    let none = intent(json!({"chat_template_args": {"reasoning_effort": "none"}}));
    assert_eq!(none.mode, Some(ThinkingMode::Disabled));
    // An explicit toggle outranks the effort grade.
    let toggled = intent(json!({
        "reasoning_effort": "high",
        "chat_template_args": {"enable_thinking": false}
    }));
    assert_eq!(toggled.mode, Some(ThinkingMode::Disabled));
}

fn mapping(yaml: &str) -> ThinkingMapping {
    serde_yaml::from_str(yaml).unwrap()
}

fn applied(mapping: &ThinkingMapping, request: Value) -> Value {
    let mut body = Map::new();
    body.insert("messages".to_string(), json!([]));
    mapping.apply(&ThinkingIntent::from_request(&request), &mut body);
    Value::Object(body)
}

#[test]
fn default_mapping_forwards_only_the_effort() {
    let default = ThinkingMapping::default();
    assert_eq!(
        applied(&default, json!({"reasoning_effort": "low"})),
        json!({"messages": [], "reasoning_effort": "low"})
    );
    assert_eq!(
        applied(
            &default,
            json!({"chat_template_args": {"enable_thinking": false}})
        ),
        json!({"messages": []})
    );
}

#[test]
fn a_provider_mapping_is_merged_with_placeholders_filled() {
    let nested = mapping(
        r#"
enabled: {reasoning: {enabled: true}}
disabled: {reasoning: {enabled: false}}
effort: {reasoning: {effort: "{effort}"}}
budget: {reasoning: {max_tokens: "{budget_tokens}"}}
"#,
    );
    assert_eq!(
        applied(
            &nested,
            json!({"reasoning_effort": "medium", "thinking_token_budget": 1024})
        ),
        json!({"messages": [], "reasoning": {"enabled": true, "effort": "medium", "max_tokens": 1024}})
    );
    assert_eq!(
        applied(&nested, json!({"chat_template_args": {"thinking": false}})),
        json!({"messages": [], "reasoning": {"enabled": false}})
    );
    // A provider served by an SGLang or vLLM backend takes the template arguments.
    let template = mapping(
        r#"
enabled: {chat_template_kwargs: {enable_thinking: true}}
disabled: {chat_template_kwargs: {enable_thinking: false}}
effort: null
"#,
    );
    assert_eq!(
        applied(&template, json!({"reasoning_effort": "high"})),
        json!({"messages": [], "chat_template_kwargs": {"enable_thinking": true}})
    );
}

#[test]
fn unmapped_choices_are_reported() {
    let default = ThinkingMapping::default();
    let off = ThinkingIntent::from_request(&json!({"chat_template_args": {"thinking": false}}));
    assert_eq!(default.unmapped(&off), Some("thinking disabled"));
    // An effort mapping expresses an effort-only request, including the mode it implies.
    let effort = ThinkingIntent::from_request(&json!({"reasoning_effort": "low"}));
    assert_eq!(default.unmapped(&effort), None);
    let budget = ThinkingIntent::from_request(&json!({"thinking_token_budget": 64}));
    assert_eq!(default.unmapped(&budget), Some("thinking token budget"));
    assert_eq!(default.unmapped(&ThinkingIntent::default()), None);
}

#[test]
fn unknown_mapping_keys_are_rejected() {
    assert!(serde_yaml::from_str::<ThinkingMapping>("enable: {}").is_err());
    assert!(!ThinkingMapping::default().require_mapping);
}
