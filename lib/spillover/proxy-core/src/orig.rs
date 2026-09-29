// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The original chat request, carried from onwards to the proxy worker.
//!
//! onwards adds one entry `"dw.orig.v1:<base64url-no-pad of JSON>"` to `nvext.extra_fields`.
//! Dynamo's preprocessor forwards `nvext.extra_fields` verbatim into the worker's
//! `extra_args.nvext.extra_fields`. For multimodal requests Dynamo also puts the original
//! `messages` in `extra_args.messages`.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::{Map, Value};

/// Prefix of the `extra_fields` entry that carries the original request.
pub const PREFIX: &str = "dw.orig.v1:";

/// Largest decoded carrier we accept, to bound memory from a client-supplied
/// (and therefore untrusted) `nvext.extra_fields` entry.
pub const MAX_CARRIER_BYTES: usize = 8 * 1024 * 1024;

/// Top-level request fields onwards copies into the carrier. Everything else is dropped.
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
pub enum OrigError {
    #[error("dw.orig entry is not valid base64url: {0}")]
    Base64(String),
    #[error("dw.orig entry is not valid JSON: {0}")]
    Json(String),
    #[error("dw.orig entry is not a JSON object")]
    NotAnObject,
    #[error("dw.orig entry exceeds the {limit}-byte limit")]
    TooLarge { limit: usize },
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

/// Encode a (field-selected) request as one `extra_fields` entry, including `PREFIX`.
pub fn encode(request: &Value) -> String {
    let json = serde_json::to_vec(request).expect("a serde_json::Value always serializes");
    format!("{PREFIX}{}", URL_SAFE_NO_PAD.encode(json))
}

/// Decode one `extra_fields` entry. Returns `Ok(None)` if it does not start with `PREFIX`.
pub fn decode(entry: &str) -> Result<Option<Value>, OrigError> {
    let Some(payload) = entry.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    // Reject before decoding so an oversized payload is never fully allocated.
    if payload.len() > MAX_CARRIER_BYTES.div_ceil(3) * 4 {
        return Err(OrigError::TooLarge {
            limit: MAX_CARRIER_BYTES,
        });
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|e| OrigError::Base64(e.to_string()))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| OrigError::Json(e.to_string()))?;
    if !value.is_object() {
        return Err(OrigError::NotAnObject);
    }
    Ok(Some(value))
}

/// Find the original request in a worker's `extra_args`.
///
/// Order: the first `dw.orig` entry in `extra_args.nvext.extra_fields`; otherwise, if
/// `extra_args.messages` exists, `{"messages": <that>}`; otherwise `Ok(None)`.
pub fn from_extra_args(extra_args: Option<&Value>) -> Result<Option<Value>, OrigError> {
    let Some(extra_args) = extra_args else {
        return Ok(None);
    };
    if let Some(fields) = extra_args
        .get("nvext")
        .and_then(|nvext| nvext.get("extra_fields"))
        .and_then(Value::as_array)
    {
        for field in fields {
            if let Some(entry) = field.as_str()
                && entry.starts_with(PREFIX)
            {
                // A single malformed carrier (client-injectable, or a partly
                // written legacy entry) must not fail the whole request: skip it
                // and keep looking, including the `messages` fallback below.
                match decode(entry) {
                    Ok(Some(request)) => return Ok(Some(request)),
                    Ok(None) | Err(_) => continue,
                }
            }
        }
    }
    if let Some(messages) = extra_args.get("messages") {
        let mut wrapper = Map::new();
        wrapper.insert("messages".to_string(), messages.clone());
        return Ok(Some(Value::Object(wrapper)));
    }
    Ok(None)
}
