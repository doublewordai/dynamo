// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for reading the frontend-supplied chat request.

use dw_proxy_core::chat_request::{self, ChatRequestError};
use serde_json::{Value, json};

fn request() -> Value {
    json!({
        "model": "zai-org/GLM-5.3",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "nvext": {"extra_fields": ["something"]},
        "stream": true
    })
}

#[test]
fn select_keeps_only_carried_fields() {
    let selected = chat_request::select_carried_fields(&request());
    assert_eq!(
        selected,
        json!({
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7
        })
    );
}

#[test]
fn select_does_not_forward_thinking_controls() {
    // The provider's `thinking` mapping decides what to send for these.
    let request = json!({
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_args": {"enable_thinking": false},
        "reasoning_effort": "low",
        "thinking_token_budget": 512
    });
    assert_eq!(
        chat_request::select_carried_fields(&request),
        json!({"messages": [{"role": "user", "content": "hello"}]})
    );
}

#[test]
fn select_forwards_sampling_controls() {
    let request = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "frequency_penalty": 1.5,
        "presence_penalty": 0.5,
        "logit_bias": {"123": -100},
        "user": "u-1",
        "prompt_cache_key": "conversation-1",
        "response_format": {"type": "json_object"}
    });
    assert_eq!(
        chat_request::select_carried_fields(&request),
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 1.5,
            "presence_penalty": 0.5,
            "logit_bias": {"123": -100},
            "response_format": {"type": "json_object"}
        })
    );
}

#[test]
fn from_extra_args_reads_the_chat_request() {
    let extra_args = json!({"chat_request": request(), "other": 1});
    let found = chat_request::from_extra_args(Some(&extra_args))
        .unwrap()
        .unwrap();
    assert_eq!(found, &request());
}

#[test]
fn from_extra_args_returns_none_when_absent() {
    assert!(chat_request::from_extra_args(None).unwrap().is_none());
    let extra_args = json!({"messages": [{"role": "user", "content": "hi"}]});
    assert!(
        chat_request::from_extra_args(Some(&extra_args))
            .unwrap()
            .is_none()
    );
}

#[test]
fn from_extra_args_rejects_malformed_requests() {
    let not_object = json!({"chat_request": "text"});
    assert!(matches!(
        chat_request::from_extra_args(Some(&not_object)),
        Err(ChatRequestError::NotAnObject)
    ));
    let no_messages = json!({"chat_request": {"model": "m"}});
    assert!(matches!(
        chat_request::from_extra_args(Some(&no_messages)),
        Err(ChatRequestError::NoMessages)
    ));
}

#[test]
fn from_extra_args_rejects_a_replayed_migration() {
    let extra_args = json!({
        "chat_request": request(),
        "chat_request_replayed_tokens": 17
    });
    assert!(matches!(
        chat_request::from_extra_args(Some(&extra_args)),
        Err(ChatRequestError::Replayed {
            replayed_tokens: 17
        })
    ));
    // A non-positive or malformed replay count does not reject.
    let zero = json!({"chat_request": request(), "chat_request_replayed_tokens": 0});
    assert!(
        chat_request::from_extra_args(Some(&zero))
            .unwrap()
            .is_some()
    );
    let text = json!({"chat_request": request(), "chat_request_replayed_tokens": "many"});
    assert!(
        chat_request::from_extra_args(Some(&text))
            .unwrap()
            .is_some()
    );
}

#[test]
fn from_extra_args_rejects_multiple_choices() {
    let extra_args = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "n": 3
    }});
    assert!(matches!(
        chat_request::from_extra_args(Some(&extra_args)),
        Err(ChatRequestError::MultipleChoices { n: 3 })
    ));
    // n = 1 is the provider default and is harmless.
    let one = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "n": 1
    }});
    assert!(chat_request::from_extra_args(Some(&one)).unwrap().is_some());
    // Anything that is not an integer 1 would be silently changed to one choice.
    for n in [json!(0), json!(2.0), json!(-1), json!("2")] {
        let odd = json!({"chat_request": {
            "messages": [{"role": "user", "content": "hi"}],
            "n": n
        }});
        assert!(
            chat_request::from_extra_args(Some(&odd)).is_err(),
            "n = {n} must be refused"
        );
    }
    let null = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "n": null
    }});
    assert!(
        chat_request::from_extra_args(Some(&null))
            .unwrap()
            .is_some()
    );
}

#[test]
fn from_extra_args_rejects_unsupported_generation_controls() {
    for (field, value) in [
        ("guided_json", json!({"type": "object"})),
        ("guided_regex", json!("^a+$")),
        ("guided_grammar", json!("root ::= \"a\"")),
        ("guided_choice", json!(["a", "b"])),
        ("guided_decoding_backend", json!("xgrammar")),
        ("guided_whitespace_pattern", json!("\\s")),
        ("top_k", json!(5)),
        ("min_p", json!(0.1)),
        ("repetition_penalty", json!(1.2)),
        ("min_tokens", json!(4)),
        ("top_logprobs", json!(3)),
        ("prompt_logprobs", json!(2)),
        ("functions", json!([])),
        ("function_call", json!("auto")),
        ("modalities", json!(["text"])),
        ("audio", json!({})),
        ("prediction", json!({})),
        ("web_search_options", json!({})),
    ] {
        let extra_args = json!({"chat_request": {
            "messages": [{"role": "user", "content": "hi"}],
            field: value
        }});
        match chat_request::from_extra_args(Some(&extra_args)) {
            Err(ChatRequestError::UnsupportedField { field: rejected }) => {
                assert_eq!(rejected, field)
            }
            other => panic!("{field} should be rejected, got {other:?}"),
        }
    }
}

#[test]
fn from_extra_args_rejects_meaningful_boolean_controls_only() {
    // The value that changes generation is rejected...
    for field in [
        "ignore_eos",
        "include_stop_str_in_output",
        "skip_special_tokens",
    ] {
        let extra_args = json!({"chat_request": {
            "messages": [{"role": "user", "content": "hi"}],
            field: true
        }});
        assert!(
            matches!(
                chat_request::from_extra_args(Some(&extra_args)),
                Err(ChatRequestError::UnsupportedField { .. })
            ),
            "{field}=true should be rejected"
        );
    }
    // ...but the neutral value is not.
    for field in [
        "ignore_eos",
        "include_stop_str_in_output",
        "skip_special_tokens",
    ] {
        let extra_args = json!({"chat_request": {
            "messages": [{"role": "user", "content": "hi"}],
            field: false
        }});
        assert!(
            chat_request::from_extra_args(Some(&extra_args))
                .unwrap()
                .is_some(),
            "{field}=false should be served"
        );
    }
    // `add_generation_prompt` defaults to true, so false is the meaningful value.
    let off = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": false
    }});
    assert!(matches!(
        chat_request::from_extra_args(Some(&off)),
        Err(ChatRequestError::UnsupportedField {
            field: "add_generation_prompt"
        })
    ));
    let on = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": true
    }});
    assert!(chat_request::from_extra_args(Some(&on)).unwrap().is_some());
}

#[test]
fn from_extra_args_rejects_requested_logprobs() {
    let extra_args = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": true,
        "top_logprobs": 5
    }});
    assert!(matches!(
        chat_request::from_extra_args(Some(&extra_args)),
        Err(ChatRequestError::UnsupportedField { field: "logprobs" })
    ));
    let plain = json!({"chat_request": {
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": false
    }});
    assert!(
        chat_request::from_extra_args(Some(&plain))
            .unwrap()
            .is_some()
    );
}

/// Every top-level field the frontend can serialize into `extra_args.chat_request`, so a new
/// field cannot be added to the snapshot without an explicit forward / reject / drop decision.
const SNAPSHOT_FIELDS: &[&str] = &[
    // `CreateChatCompletionRequest`.
    "messages",
    "model",
    "mm_processor_kwargs",
    "store",
    "reasoning_effort",
    "metadata",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "modalities",
    "prediction",
    "audio",
    "presence_penalty",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "user",
    "prompt_cache_key",
    "function_call",
    "functions",
    "web_search_options",
    // `CommonExt`.
    "ignore_eos",
    "min_tokens",
    "top_k",
    "min_p",
    "repetition_penalty",
    "include_stop_str_in_output",
    "guided_json",
    "guided_regex",
    "guided_grammar",
    "guided_choice",
    "guided_decoding_backend",
    "guided_whitespace_pattern",
    "skip_special_tokens",
    "prompt_logprobs",
    "add_generation_prompt",
    "continue_final_message",
    // `NvCreateChatCompletionRequest` extensions.
    "nvext",
    "chat_template_args",
    "thinking",
    "thinking_token_budget",
    "media_io_kwargs",
    "return_tokens_as_token_ids",
];

#[test]
fn every_snapshot_field_has_an_explicit_decision() {
    let mut categorized: Vec<&str> = chat_request::CARRIED_FIELDS.to_vec();
    categorized.extend_from_slice(chat_request::UNSUPPORTED_FIELDS);
    categorized.extend_from_slice(chat_request::DROPPED_FIELDS);
    categorized.extend_from_slice(chat_request::THINKING_FIELDS);
    categorized.extend_from_slice(chat_request::CACHE_KEY_FIELDS);
    // Boolean controls are rejected at their meaningful value and `n` above one.
    categorized.extend([
        "logprobs",
        "ignore_eos",
        "include_stop_str_in_output",
        "skip_special_tokens",
        "add_generation_prompt",
        "continue_final_message",
        "n",
    ]);
    for field in SNAPSHOT_FIELDS {
        assert!(
            categorized.contains(field),
            "chat request field `{field}` is neither carried, rejected nor documented as dropped"
        );
    }
    for field in &categorized {
        assert!(
            SNAPSHOT_FIELDS.contains(field),
            "`{field}` is categorized but is not a chat request snapshot field"
        );
    }
}

#[test]
fn a_client_supplied_carrier_is_ignored() {
    // The old `dw.orig` carrier came from the client; nothing reads it now.
    let extra_args = json!({"nvext": {"extra_fields": ["dw.orig.v1:e30"]}});
    assert!(
        chat_request::from_extra_args(Some(&extra_args))
            .unwrap()
            .is_none()
    );
}
