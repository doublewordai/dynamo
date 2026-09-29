// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming chat calls to an OpenAI-compatible provider (OpenRouter and others).

use std::collections::BTreeMap;
use std::pin::Pin;
use std::time::Duration;

use anyhow::Context;
use bytes::Bytes;
use futures::stream::{BoxStream, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::errors::UpstreamError;
use crate::orig;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// Name used in logs, metrics and the served-by tag.
    pub name: String,
    /// Base URL ending in `/v1`, e.g. `https://openrouter.ai/api/v1`.
    pub base_url: String,
    /// Environment variable holding the API key.
    pub api_key_env: String,
    /// Provider-side model slug, e.g. `z-ai/glm-5.3`.
    pub model: String,
    /// Merged into the body as `provider` (OpenRouter provider routing preferences).
    #[serde(default)]
    pub provider_preferences: Option<Value>,
    /// Extra JSON merged into every request body (e.g. reasoning settings), applied last.
    #[serde(default)]
    pub body_overrides: Option<Value>,
    #[serde(default)]
    pub extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
}

fn default_connect_timeout_ms() -> u64 {
    10_000
}

/// Parsed `chat.completion.chunk` objects, in order. Ends after `[DONE]`.
pub type ChunkStream = BoxStream<'static, Result<Value, UpstreamError>>;

pub struct UpstreamClient {
    config: ProviderConfig,
    api_key: String,
    client: reqwest::Client,
}

impl UpstreamClient {
    /// Reads the API key from `config.api_key_env`; fails if it is unset.
    pub fn new(config: ProviderConfig) -> anyhow::Result<Self> {
        let api_key = std::env::var(&config.api_key_env)
            .with_context(|| format!("environment variable {} is not set", config.api_key_env))?;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
            .build()
            .context("building the HTTP client")?;
        Ok(Self {
            config,
            api_key,
            client,
        })
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Provider request body from the original chat request: set `model`, `stream: true`,
    /// `stream_options.include_usage: true`, `provider` preferences and `body_overrides`;
    /// drop `nvext` and any field not in `orig::CARRIED_FIELDS`.
    pub fn build_body(&self, original: &Value) -> Value {
        let mut body = match orig::select_carried_fields(original) {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        body.insert(
            "model".to_string(),
            Value::String(self.config.model.clone()),
        );
        body.insert("stream".to_string(), Value::Bool(true));
        body.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
        if let Some(preferences) = &self.config.provider_preferences {
            body.insert("provider".to_string(), preferences.clone());
        }
        if let Some(Value::Object(overrides)) = &self.config.body_overrides {
            for (key, value) in overrides {
                body.insert(key.clone(), value.clone());
            }
        }
        Value::Object(body)
    }

    /// POST `{base_url}/chat/completions` and return the chunk stream.
    /// Errors before the first chunk (HTTP status, connect, first-line parse) are returned as
    /// `Err` so the worker can fail before emitting anything. Dropping the stream cancels the
    /// HTTP request.
    pub async fn stream_chat(&self, body: Value) -> Result<ChunkStream, UpstreamError> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let mut request = self
            .client
            .post(&url)
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key),
            )
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&body);
        for (name, value) in &self.config.extra_headers {
            request = request.header(name.as_str(), value.as_str());
        }
        let response = request
            .send()
            .await
            .map_err(|e| UpstreamError::Transport(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let text = response.text().await.unwrap_or_default();
            return Err(UpstreamError::from_status(
                status.as_u16(),
                &text,
                retry_after.as_deref(),
            ));
        }

        let mut state = SseState::new(Box::pin(response.bytes_stream()));
        match state.pump().await {
            Some(Ok(first)) => {
                let head = futures::stream::once(async move { Ok(first) });
                let tail = futures::stream::unfold(state, |mut state| async move {
                    let item = state.pump().await;
                    item.map(|item| (item, state))
                });
                Ok(Box::pin(head.chain(tail)))
            }
            Some(Err(error)) => Err(error),
            // `[DONE]` (or a finish_reason) arrived before any chunk.
            None => Ok(Box::pin(futures::stream::empty())),
        }
    }
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

enum Dispatch {
    Event(Result<Value, UpstreamError>),
    Done,
    Ignored,
}

/// Incremental SSE parser: handles CRLF, comments, multi-line data, `[DONE]` and events split
/// across network chunks. Yields `chat.completion.chunk` objects and ends after `[DONE]`.
struct SseState {
    bytes: ByteStream,
    buffer: Vec<u8>,
    data_lines: Vec<String>,
    done: bool,
    saw_finish_reason: bool,
    eof: bool,
}

impl SseState {
    fn new(bytes: ByteStream) -> Self {
        Self {
            bytes,
            buffer: Vec::new(),
            data_lines: Vec::new(),
            done: false,
            saw_finish_reason: false,
            eof: false,
        }
    }

    /// Next chunk, `None` when the provider ended cleanly. A broken stream becomes an `Err`
    /// item so the consumer still sees it.
    async fn pump(&mut self) -> Option<Result<Value, UpstreamError>> {
        loop {
            while let Some(line) = self.take_line() {
                match self.handle_line(&line) {
                    Dispatch::Event(item) => return Some(item),
                    Dispatch::Done => return None,
                    Dispatch::Ignored => {}
                }
            }
            if self.eof {
                if !self.buffer.is_empty() {
                    let line = String::from_utf8_lossy(&self.buffer).into_owned();
                    self.buffer.clear();
                    match self.handle_line(&line) {
                        Dispatch::Event(item) => return Some(item),
                        Dispatch::Done => return None,
                        Dispatch::Ignored => continue,
                    }
                }
                return match self.dispatch_event() {
                    Dispatch::Event(item) => Some(item),
                    Dispatch::Done => None,
                    Dispatch::Ignored => {
                        if self.done || self.saw_finish_reason {
                            None
                        } else {
                            self.done = true;
                            Some(Err(UpstreamError::StreamBroken(
                                "stream ended before [DONE]".to_string(),
                            )))
                        }
                    }
                };
            }
            match self.bytes.next().await {
                Some(Ok(chunk)) => self.buffer.extend_from_slice(&chunk),
                Some(Err(error)) => {
                    return Some(Err(UpstreamError::Transport(error.to_string())));
                }
                None => self.eof = true,
            }
        }
    }

    /// Pull one complete line out of the buffer, normalizing a trailing CR.
    fn take_line(&mut self) -> Option<String> {
        let end = self.buffer.iter().position(|&byte| byte == b'\n')?;
        let mut line: Vec<u8> = self.buffer.drain(..=end).collect();
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Some(String::from_utf8_lossy(&line).into_owned())
    }

    fn handle_line(&mut self, line: &str) -> Dispatch {
        if line.is_empty() {
            return self.dispatch_event();
        }
        if line.starts_with(':') {
            return Dispatch::Ignored;
        }
        if let Some(value) = line.strip_prefix("data:") {
            self.data_lines
                .push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
        Dispatch::Ignored
    }

    fn dispatch_event(&mut self) -> Dispatch {
        if self.data_lines.is_empty() {
            return Dispatch::Ignored;
        }
        let data = self.data_lines.join("\n");
        self.data_lines.clear();
        if data == "[DONE]" {
            self.done = true;
            return Dispatch::Done;
        }
        match serde_json::from_str::<Value>(&data) {
            Err(error) => Dispatch::Event(Err(UpstreamError::StreamBroken(format!(
                "invalid SSE data: {error}"
            )))),
            Ok(Value::Object(mut map)) => match map.remove("error") {
                Some(error) => Dispatch::Event(Err(UpstreamError::InStream(error_message(&error)))),
                None => {
                    let value = Value::Object(map);
                    self.note_finish_reason(&value);
                    Dispatch::Event(Ok(value))
                }
            },
            Ok(value) => {
                self.note_finish_reason(&value);
                Dispatch::Event(Ok(value))
            }
        }
    }

    fn note_finish_reason(&mut self, value: &Value) {
        if value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    choice
                        .get("finish_reason")
                        .is_some_and(|reason| !reason.is_null())
                })
            })
        {
            self.saw_finish_reason = true;
        }
    }
}

/// Provider stream errors may be a string or an object with a `message`.
fn error_message(error: &Value) -> String {
    if let Some(message) = error.get("message").and_then(Value::as_str) {
        message.to_string()
    } else if let Some(message) = error.as_str() {
        message.to_string()
    } else {
        error.to_string()
    }
}
