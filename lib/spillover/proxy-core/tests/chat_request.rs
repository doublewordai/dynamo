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
fn a_client_supplied_carrier_is_ignored() {
    // The old `dw.orig` carrier came from the client; nothing reads it now.
    let extra_args = json!({"nvext": {"extra_fields": ["dw.orig.v1:e30"]}});
    assert!(
        chat_request::from_extra_args(Some(&extra_args))
            .unwrap()
            .is_none()
    );
}
