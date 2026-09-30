// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The client's chat request, as the proxy receives it from Dynamo.
//!
//! The proxy worker advertises the `chat_request` runtime capability. When the KV router
//! dispatches a chat request to it, the frontend puts the request (after its own normalization)
//! in `extra_args.chat_request`; workers without the capability never see it. The frontend writes
//! the field itself, so a client cannot supply or forge it.
//!
//! Not every field of that snapshot reaches the provider. [`CARRIED_FIELDS`] are forwarded
//! verbatim, [`unsupported_field`] rejects anything the proxy cannot honour faithfully, and the
//! rest are dropped because they are Dynamo-internal or have no generation effect. Deciding each
//! field explicitly (and rejecting rather than silently changing generation) is the point: a
//! request must not produce a different answer because the router happened to pick a proxy.

use serde_json::{Map, Value};

/// `extra_args` key holding the chat request. Must equal
/// `dynamo_llm::local_model::runtime_config::CHAT_REQUEST_EXTRA_ARGS_KEY`.
pub const EXTRA_ARGS_KEY: &str = "chat_request";

/// `extra_args` key carrying the number of output tokens the KV router has already streamed to
/// the client before re-dispatching a migration retry. Set by the coordinator's migration path;
/// a proxy cannot continue a partial response, so its presence rejects the request.
pub const REPLAYED_TOKENS_KEY: &str = "chat_request_replayed_tokens";

/// Top-level request fields forwarded verbatim to the provider: OpenAI-compatible chat fields,
/// including the commonly supported penalties and `logit_bias`. Thinking controls are
/// not here; see [`THINKING_FIELDS`].
///
/// `max_tokens`/`max_completion_tokens` are deliberately absent: the provider cap comes from the
/// frontend's `stop_conditions.max_tokens` and is set by `UpstreamClient::build_body`.
pub const CARRIED_FIELDS: &[&str] = &[
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
];

/// Fields read only to derive the opaque provider cache key (`cache_key`), never forwarded: `user`
/// identifies our customer's end users to a third party.
pub const CACHE_KEY_FIELDS: &[&str] = &["prompt_cache_key", "user"];

/// Fields that carry the client's thinking choice. They are not forwarded as they are: the
/// proxy reads the choice from them (`thinking::ThinkingIntent`) and the provider's
/// `thinking_dialect` decides what to send. `chat_template_args` also carries template-only
/// variables, which mean nothing to a provider.
pub const THINKING_FIELDS: &[&str] = &[
    "chat_template_args",
    "reasoning_effort",
    "thinking",
    "thinking_token_budget",
];

/// Fields whose mere presence (with a non-null value) means the proxy cannot serve the request
/// faithfully. Rejecting is better than silently dropping a generation control: the client sees
/// a clear error instead of a different answer. `stream`, `nvext` and the metadata-only fields
/// are not here because dropping them cannot change generation.
pub const UNSUPPORTED_FIELDS: &[&str] = &[
    "guided_json",
    "guided_regex",
    "guided_grammar",
    "guided_choice",
    "guided_decoding_backend",
    "guided_whitespace_pattern",
    "top_k",
    "min_p",
    "repetition_penalty",
    "min_tokens",
    "top_logprobs",
    "prompt_logprobs",
    "functions",
    "function_call",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
];

/// Fields deliberately not forwarded and not rejected: Dynamo-internal, response-shape metadata,
/// or fields the frontend already normalized into another carrier (`max_tokens` ->
/// `stop_conditions`). Dropping any of them cannot change the generated tokens.
pub const DROPPED_FIELDS: &[&str] = &[
    "model",
    "stream",
    "stream_options",
    "nvext",
    "store",
    "metadata",
    "mm_processor_kwargs",
    "media_io_kwargs",
    "return_tokens_as_token_ids",
    "service_tier",
    "max_tokens",
    "max_completion_tokens",
];

/// Boolean fields whose *neutral* value is harmless to drop, mapped to the value that makes the
/// field meaningful (and therefore unsupported). `add_generation_prompt` defaults to `true`
/// (vLLM 0.27.1 and the Python frontend), so `false` is the value that must be rejected.
const UNSUPPORTED_BOOLEANS: &[(&str, bool)] = &[
    ("logprobs", true),
    ("ignore_eos", true),
    ("include_stop_str_in_output", true),
    ("skip_special_tokens", true),
    ("add_generation_prompt", false),
    ("continue_final_message", true),
];

#[derive(Debug, thiserror::Error)]
pub enum ChatRequestError {
    #[error("extra_args.chat_request is not a JSON object")]
    NotAnObject,
    #[error("extra_args.chat_request has no messages array")]
    NoMessages,
    #[error(
        "the proxy cannot faithfully serve chat request field `{field}`; it has no equivalent at the provider"
    )]
    UnsupportedField { field: &'static str },
    #[error("the proxy serves one choice, so n > 1 is unsupported (n = {n})")]
    MultipleChoices { n: u64 },
    #[error(
        "the router replayed {replayed_tokens} output tokens to this proxy, which cannot \
         continue a partial response"
    )]
    Replayed { replayed_tokens: u64 },
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

/// The first field in the request the proxy cannot honour faithfully, if any. Checks the
/// explicit non-boolean list, the boolean fields' meaningful values, `n > 1` and the logprob
/// fields (which are computed by the provider but the renderer cannot return).
pub fn unsupported_field(request: &Value) -> Option<&'static str> {
    let Value::Object(fields) = request else {
        return None;
    };
    for (field, meaningful) in UNSUPPORTED_BOOLEANS {
        if fields.get(*field).and_then(Value::as_bool) == Some(*meaningful) {
            return Some(field);
        }
    }
    for field in UNSUPPORTED_FIELDS {
        if fields.get(*field).is_some_and(|value| !value.is_null()) {
            return Some(field);
        }
    }
    if fields
        .get("n")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        return Some("n");
    }
    None
}

/// The number of already-streamed output tokens when the router re-dispatches a migration
/// retry, if the frontend attached it. A non-positive or non-integer value is ignored.
pub fn replayed_tokens(extra_args: Option<&Value>) -> Option<u64> {
    extra_args
        .and_then(|extra_args| extra_args.get(REPLAYED_TOKENS_KEY))
        .and_then(Value::as_u64)
        .filter(|tokens| *tokens > 0)
}

/// Find the chat request in a worker's `extra_args`. `Ok(None)` means the frontend did not
/// attach one (for example a non-chat request, or a router mode other than KV); the caller must
/// treat that as a migratable error so the request can be retried on a hosted worker.
pub fn from_extra_args(extra_args: Option<&Value>) -> Result<Option<&Value>, ChatRequestError> {
    if let Some(replayed_tokens) = replayed_tokens(extra_args) {
        return Err(ChatRequestError::Replayed { replayed_tokens });
    }
    let Some(request) = extra_args.and_then(|extra_args| extra_args.get(EXTRA_ARGS_KEY)) else {
        return Ok(None);
    };
    let Value::Object(fields) = request else {
        return Err(ChatRequestError::NotAnObject);
    };
    if !fields.get("messages").is_some_and(Value::is_array) {
        return Err(ChatRequestError::NoMessages);
    }
    if let Some(field) = unsupported_field(request) {
        if field == "n" {
            let n = fields.get("n").and_then(Value::as_u64).unwrap_or(0);
            return Err(ChatRequestError::MultipleChoices { n });
        }
        return Err(ChatRequestError::UnsupportedField { field });
    }
    Ok(Some(request))
}
