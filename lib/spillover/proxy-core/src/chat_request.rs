// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The client's chat request, as the proxy receives it from Dynamo.
//!
//! The proxy worker advertises the `chat_request` runtime capability. When the KV router
//! dispatches a chat request to it, the frontend puts the request (after its own normalization)
//! in `extra_args.chat_request`; workers without the capability never see it. The frontend writes
//! the field itself, so a client cannot supply or forge it.

use serde_json::{Map, Value};

/// `extra_args` key holding the chat request. Must equal
/// `dynamo_llm::local_model::runtime_config::CHAT_REQUEST_EXTRA_ARGS_KEY`.
pub const EXTRA_ARGS_KEY: &str = "chat_request";

/// Top-level request fields forwarded to the provider. Everything else is dropped.
pub const CARRIED_FIELDS: &[&str] = &[
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "reasoning",
    "chat_template_kwargs",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "logprobs",
    "top_logprobs",
];

#[derive(Debug, thiserror::Error)]
pub enum ChatRequestError {
    #[error("extra_args.chat_request is not a JSON object")]
    NotAnObject,
    #[error("extra_args.chat_request has no messages array")]
    NoMessages,
}

/// Keep only `CARRIED_FIELDS` from a chat request object.
pub fn select_carried_fields(request: &Value) -> Value {
    let mut kept = Map::new();
    if let Value::Object(request) = request {
        for field in CARRIED_FIELDS {
            if let Some(value) = request.get(*field) {
                kept.insert((*field).to_string(), value.clone());
            }
        }
    }
    Value::Object(kept)
}

/// Find the chat request in a worker's `extra_args`. `Ok(None)` means the frontend did not
/// attach one (for example a non-chat request, or a router mode other than KV).
pub fn from_extra_args(extra_args: Option<&Value>) -> Result<Option<&Value>, ChatRequestError> {
    let Some(request) = extra_args.and_then(|extra_args| extra_args.get(EXTRA_ARGS_KEY)) else {
        return Ok(None);
    };
    let Value::Object(fields) = request else {
        return Err(ChatRequestError::NotAnObject);
    };
    if !fields.get("messages").is_some_and(Value::is_array) {
        return Err(ChatRequestError::NoMessages);
    }
    Ok(Some(request))
}
