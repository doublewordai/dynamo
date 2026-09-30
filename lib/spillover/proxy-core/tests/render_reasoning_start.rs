// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `reasoning_start`: the parser state derived from the worker's `extra_args`.

use dw_proxy_core::render::{ParserFamily, ReasoningStart, reasoning_start};
use serde_json::{Value, json};

fn kwargs(family: ParserFamily, value: Value) -> ReasoningStart {
    reasoning_start(
        family,
        Some(&json!({"reasoning_parser_kwargs": {"chat_template_kwargs": value}})),
    )
}

#[test]
fn missing_signals_use_each_family_default() {
    assert_eq!(
        reasoning_start(ParserFamily::Glm47, None),
        ReasoningStart::InsideReasoning
    );
    assert_eq!(
        reasoning_start(ParserFamily::DeepseekV41, None),
        ReasoningStart::InsideReasoning
    );
    assert_eq!(
        reasoning_start(ParserFamily::Hermes, None),
        ReasoningStart::Outside
    );
}

#[test]
fn unrelated_extra_args_keep_the_default() {
    let extra = json!({
        "nvext": {"extra_fields": ["engine_data"]},
        "sampling_options": {"temperature": 0.2},
        "reasoning_parser_kwargs": {},
    });
    assert_eq!(
        reasoning_start(ParserFamily::Glm47, Some(&extra)),
        ReasoningStart::InsideReasoning
    );
    assert_eq!(
        reasoning_start(ParserFamily::DeepseekV41, Some(&extra)),
        ReasoningStart::InsideReasoning
    );
    assert_eq!(
        reasoning_start(ParserFamily::Hermes, Some(&extra)),
        ReasoningStart::Outside
    );
}

#[test]
fn deepseek_reasoning_ended_is_the_direct_signal() {
    let open = json!({"reasoning_ended": false});
    let closed = json!({"reasoning_ended": true});
    assert_eq!(
        reasoning_start(ParserFamily::DeepseekV41, Some(&open)),
        ReasoningStart::InsideReasoning
    );
    assert_eq!(
        reasoning_start(ParserFamily::DeepseekV41, Some(&closed)),
        ReasoningStart::Outside
    );
}

#[test]
fn reasoning_ended_wins_over_template_kwargs() {
    // The signal was computed from the rendered prompt, so it outranks a request arg.
    let extra = json!({
        "reasoning_ended": false,
        "reasoning_parser_kwargs": {"chat_template_kwargs": {"enable_thinking": false}},
    });
    assert_eq!(
        reasoning_start(ParserFamily::DeepseekV41, Some(&extra)),
        ReasoningStart::InsideReasoning
    );
}

#[test]
fn thinking_bool_disables_glm_and_deepseek() {
    for value in [
        json!({"thinking": false}),
        json!({"enable_thinking": false}),
    ] {
        assert_eq!(
            kwargs(ParserFamily::Glm47, value.clone()),
            ReasoningStart::Outside,
            "{value}"
        );
        assert_eq!(
            kwargs(ParserFamily::DeepseekV41, value.clone()),
            ReasoningStart::Outside,
            "{value}"
        );
    }
}

#[test]
fn thinking_bool_enables_glm_and_deepseek() {
    for value in [json!({"thinking": true}), json!({"enable_thinking": true})] {
        assert_eq!(
            kwargs(ParserFamily::Glm47, value.clone()),
            ReasoningStart::InsideReasoning,
            "{value}"
        );
        assert_eq!(
            kwargs(ParserFamily::DeepseekV41, value.clone()),
            ReasoningStart::InsideReasoning,
            "{value}"
        );
    }
}

#[test]
fn deepseek_honors_thinking_mode() {
    assert_eq!(
        kwargs(ParserFamily::DeepseekV41, json!({"thinking_mode": "chat"})),
        ReasoningStart::Outside
    );
    assert_eq!(
        kwargs(
            ParserFamily::DeepseekV41,
            json!({"thinking_mode": "thinking"})
        ),
        ReasoningStart::InsideReasoning
    );
    // An unknown mode falls back to the family default.
    assert_eq!(
        kwargs(
            ParserFamily::DeepseekV41,
            json!({"thinking_mode": "adaptive"})
        ),
        ReasoningStart::InsideReasoning
    );
}

#[test]
fn glm_ignores_thinking_mode() {
    // GLM's template is toggled by `thinking`/`enable_thinking`, not `thinking_mode`.
    assert_eq!(
        kwargs(ParserFamily::Glm47, json!({"thinking_mode": "chat"})),
        ReasoningStart::InsideReasoning
    );
}

#[test]
fn hermes_is_always_outside() {
    for value in [
        json!({"thinking": true}),
        json!({"thinking": false}),
        json!({"enable_thinking": true}),
        json!({}),
    ] {
        assert_eq!(
            kwargs(ParserFamily::Hermes, value.clone()),
            ReasoningStart::Outside,
            "{value}"
        );
    }
}

#[test]
fn non_bool_signals_fall_back_to_the_default() {
    let value = json!({"thinking": "yes", "enable_thinking": 1});
    assert_eq!(
        kwargs(ParserFamily::Glm47, value.clone()),
        ReasoningStart::InsideReasoning
    );
    assert_eq!(
        kwargs(ParserFamily::DeepseekV41, value),
        ReasoningStart::InsideReasoning
    );
}
