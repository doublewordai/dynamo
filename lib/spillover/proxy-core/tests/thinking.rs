// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading the thinking intent from the normalized chat request, and every dialect's
//! translation of every kind of intent.

use dw_proxy_core::thinking::{ThinkingDialect, ThinkingIntent, ThinkingMode};
use serde_json::{Value, json};

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
    // What the preprocessor writes when it normalizes a bare `enable_thinking`.
    let on = intent(json!({"chat_template_args": {"thinking": true, "enable_thinking": true}}));
    assert_eq!(on.mode, Some(ThinkingMode::Enabled));
    // `adaptive` clears the toggles and keeps the mode string.
    let adaptive = intent(json!({"chat_template_args": {"thinking_mode": "adaptive"}}));
    assert_eq!(adaptive.mode, Some(ThinkingMode::Adaptive));
    // The `thinking` toggle wins over `enable_thinking`, as in the frontend.
    let first = intent(json!({"chat_template_args": {"thinking": true, "enable_thinking": false}}));
    assert_eq!(first.mode, Some(ThinkingMode::Enabled));
    assert_eq!(intent(json!({"messages": []})), ThinkingIntent::default());
}

#[test]
fn effort_and_budget_are_read_and_imply_a_mode() {
    let high = intent(json!({"reasoning_effort": "High", "thinking_token_budget": 2048}));
    assert_eq!(high.mode, Some(ThinkingMode::Enabled));
    assert_eq!(high.effort.as_deref(), Some("high"));
    assert_eq!(high.budget_tokens, Some(2048));
    let none = intent(json!({"chat_template_args": {"reasoning_effort": "none"}}));
    assert_eq!(none.mode, Some(ThinkingMode::Disabled));
    // An explicit toggle outranks the grade.
    let toggled = intent(json!({
        "reasoning_effort": "high",
        "chat_template_args": {"enable_thinking": false}
    }));
    assert_eq!(toggled.mode, Some(ThinkingMode::Disabled));
}

#[test]
fn dialects_deserialize_by_name_and_default_to_reasoning_effort() {
    for (name, dialect) in [
        ("reasoning_effort", ThinkingDialect::ReasoningEffort),
        ("reasoning_object", ThinkingDialect::ReasoningObject),
        ("chat_template_kwargs", ThinkingDialect::ChatTemplateKwargs),
        ("none", ThinkingDialect::None),
    ] {
        assert_eq!(
            serde_yaml::from_str::<ThinkingDialect>(name).unwrap(),
            dialect
        );
    }
    assert!(serde_yaml::from_str::<ThinkingDialect>("reasoning").is_err());
    assert_eq!(ThinkingDialect::default(), ThinkingDialect::ReasoningEffort);
}

/// The intents a request can carry after the frontend's normalization.
fn cases() -> Vec<(&'static str, Value)> {
    vec![
        ("nothing", json!({})),
        (
            "on",
            json!({"chat_template_args": {"thinking": true, "enable_thinking": true}}),
        ),
        (
            "off",
            json!({"chat_template_args": {"thinking": false, "enable_thinking": false}}),
        ),
        (
            "adaptive",
            json!({"chat_template_args": {"thinking_mode": "adaptive"}}),
        ),
        ("effort", json!({"reasoning_effort": "low"})),
        ("effort none", json!({"reasoning_effort": "none"})),
        ("budget", json!({"thinking_token_budget": 512})),
        (
            "effort and budget",
            json!({"reasoning_effort": "high", "thinking_token_budget": 512}),
        ),
        (
            "off with effort",
            json!({"reasoning_effort": "high", "chat_template_args": {"enable_thinking": false}}),
        ),
    ]
}

/// (case, fields sent, unexpressed parts) for one dialect, in `cases()` order.
fn check(dialect: ThinkingDialect, expected: &[(&str, Value, &[&str])]) {
    let cases = cases();
    assert_eq!(cases.len(), expected.len(), "one expectation per case");
    for ((name, request), (expected_name, fields, unexpressed)) in cases.iter().zip(expected) {
        assert_eq!(name, expected_name);
        let translation = dialect.translate(&ThinkingIntent::from_request(request));
        assert_eq!(
            Value::Object(translation.fields),
            *fields,
            "{dialect:?} / {name}: fields"
        );
        assert_eq!(
            translation.unexpressed, *unexpressed,
            "{dialect:?} / {name}: unexpressed"
        );
    }
}

#[test]
fn reasoning_effort_dialect() {
    check(
        ThinkingDialect::ReasoningEffort,
        &[
            ("nothing", json!({}), &[]),
            ("on", json!({}), &["thinking enabled"]),
            ("off", json!({"reasoning_effort": "none"}), &[]),
            ("adaptive", json!({}), &[]),
            ("effort", json!({"reasoning_effort": "low"}), &[]),
            ("effort none", json!({"reasoning_effort": "none"}), &[]),
            ("budget", json!({}), &["thinking token budget"]),
            (
                "effort and budget",
                json!({"reasoning_effort": "high"}),
                &["thinking token budget"],
            ),
            ("off with effort", json!({"reasoning_effort": "none"}), &[]),
        ],
    );
}

#[test]
fn reasoning_object_dialect() {
    check(
        ThinkingDialect::ReasoningObject,
        &[
            ("nothing", json!({}), &[]),
            ("on", json!({"reasoning": {"enabled": true}}), &[]),
            ("off", json!({"reasoning": {"enabled": false}}), &[]),
            ("adaptive", json!({}), &[]),
            (
                "effort",
                json!({"reasoning": {"enabled": true, "effort": "low"}}),
                &[],
            ),
            ("effort none", json!({"reasoning": {"enabled": false}}), &[]),
            ("budget", json!({"reasoning": {"max_tokens": 512}}), &[]),
            (
                "effort and budget",
                json!({"reasoning": {"enabled": true, "max_tokens": 512}}),
                &[],
            ),
            (
                "off with effort",
                json!({"reasoning": {"enabled": false}}),
                &[],
            ),
        ],
    );
}

#[test]
fn chat_template_kwargs_dialect() {
    let on = json!({"chat_template_kwargs": {"enable_thinking": true, "thinking": true}});
    let off = json!({"chat_template_kwargs": {"enable_thinking": false, "thinking": false}});
    check(
        ThinkingDialect::ChatTemplateKwargs,
        &[
            ("nothing", json!({}), &[]),
            ("on", on.clone(), &[]),
            ("off", off.clone(), &[]),
            ("adaptive", json!({}), &[]),
            ("effort", on.clone(), &["reasoning effort"]),
            ("effort none", off.clone(), &[]),
            ("budget", json!({}), &["thinking token budget"]),
            (
                "effort and budget",
                on,
                &["reasoning effort", "thinking token budget"],
            ),
            ("off with effort", off, &[]),
        ],
    );
}

#[test]
fn none_dialect() {
    check(
        ThinkingDialect::None,
        &[
            ("nothing", json!({}), &[]),
            ("on", json!({}), &["thinking enabled"]),
            ("off", json!({}), &["thinking disabled"]),
            ("adaptive", json!({}), &[]),
            (
                "effort",
                json!({}),
                &["thinking enabled", "reasoning effort"],
            ),
            ("effort none", json!({}), &["thinking disabled"]),
            ("budget", json!({}), &["thinking token budget"]),
            (
                "effort and budget",
                json!({}),
                &[
                    "thinking enabled",
                    "reasoning effort",
                    "thinking token budget",
                ],
            ),
            ("off with effort", json!({}), &["thinking disabled"]),
        ],
    );
}
