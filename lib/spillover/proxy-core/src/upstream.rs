// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming chat calls to an OpenAI-compatible provider (OpenRouter and others).

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use bytes::{Bytes, BytesMut};
use futures::stream::{BoxStream, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::chat_request;
use crate::errors::UpstreamError;

#[derive(Clone, Deserialize, PartialEq)]
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
    /// Idle timeout on each provider read, including the wait for the first byte. A stream may
    /// legitimately run for minutes, so this bounds the gap between bytes rather than the whole
    /// request. Raise it for a provider that thinks silently without SSE keepalives.
    #[serde(default = "default_read_timeout_ms")]
    pub read_timeout_ms: u64,
}

fn default_connect_timeout_ms() -> u64 {
    10_000
}

fn default_read_timeout_ms() -> u64 {
    120_000
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `extra_headers` and `body_overrides` can carry gateway credentials, and
        // `api_key_env` names the secret, so the derived `Debug` is replaced with one
        // that never prints their values.
        f.debug_struct("ProviderConfig")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api_key_env", &self.api_key_env)
            .field("model", &self.model)
            .field("provider_preferences", &self.provider_preferences)
            .field(
                "body_overrides",
                &self.body_overrides.as_ref().map(|_| "<redacted>"),
            )
            .field("extra_headers", &RedactedHeaders(&self.extra_headers))
            .field("connect_timeout_ms", &self.connect_timeout_ms)
            .field("read_timeout_ms", &self.read_timeout_ms)
            .finish()
    }
}

/// Prints header names only; their values may be credentials.
struct RedactedHeaders<'a>(&'a BTreeMap<String, String>);

impl std::fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for key in self.0.keys() {
            map.entry(key, &"<redacted>");
        }
        map.finish()
    }
}

/// Upper bound on waiting for response headers or a non-2xx body, even when
/// `read_timeout_ms` is large: a provider that accepts the connection but never
/// answers must not park the request.
const MAX_RESPONSE_HEADER_WAIT: Duration = Duration::from_secs(30);
/// Grace after a `finish_reason` for a trailing usage chunk before ending the stream,
/// so a provider that keeps the connection open does not delay the terminal chunk by
/// the full read timeout.
const FINISH_GRACE: Duration = Duration::from_secs(2);
/// Floor for the per-client 429 cooldown when the provider sends no usable `Retry-After`.
const DEFAULT_RATE_LIMIT_COOLDOWN_MS: u64 = 1_000;
/// Largest single (unterminated) SSE line held in memory.
const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
/// Largest SSE event (all `data:` lines) held in memory before a blank line.
const MAX_SSE_EVENT_BYTES: usize = 4 * 1024 * 1024;
/// Largest number of `data:` lines in one SSE event.
const MAX_SSE_EVENT_LINES: usize = 10_000;
/// Largest non-2xx provider body read into memory while classifying an error.
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Parsed `chat.completion.chunk` objects, in order. Ends after `[DONE]`.
pub type ChunkStream = BoxStream<'static, Result<Value, UpstreamError>>;

pub struct UpstreamClient {
    config: ProviderConfig,
    api_key: String,
    client: reqwest::Client,
    read_timeout: Duration,
    /// Epoch-millisecond deadline until which this client refuses to call the provider
    /// after a 429, so re-probes do not hammer an already rate-limited provider.
    rate_limited_until: AtomicU64,
}

impl UpstreamClient {
    /// Reads the API key from `config.api_key_env`; fails if it is unset.
    pub fn new(config: ProviderConfig) -> anyhow::Result<Self> {
        let api_key = std::env::var(&config.api_key_env)
            .with_context(|| format!("environment variable {} is not set", config.api_key_env))?;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
            // A provider's streaming endpoint should not redirect; following one
            // would re-send the carried conversation (and custom headers) to an
            // arbitrary host chosen by the provider.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("building the HTTP client")?;
        let read_timeout = Duration::from_millis(config.read_timeout_ms);
        Ok(Self {
            config,
            api_key,
            client,
            read_timeout,
            rate_limited_until: AtomicU64::new(0),
        })
    }

    /// Override the idle timeout applied to each read from the provider.
    pub fn with_read_timeout(mut self, read_timeout: Duration) -> Self {
        self.read_timeout = read_timeout;
        self
    }

    /// Extend the client's 429 cooldown so re-probes wait for the provider's `Retry-After`.
    fn note_rate_limited(&self, retry_after_ms: u64) {
        let cooldown_ms = retry_after_ms.max(DEFAULT_RATE_LIMIT_COOLDOWN_MS);
        let until = now_epoch_ms().saturating_add(cooldown_ms);
        self.rate_limited_until.fetch_max(until, Ordering::Relaxed);
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Provider request body from the original chat request: set `model`, `stream: true`,
    /// `stream_options.include_usage: true`, the provider `max_tokens`, `provider` preferences
    /// and `body_overrides`; drop `nvext` and any field not in `chat_request::CARRIED_FIELDS`.
    ///
    /// `max_tokens` is the frontend's authoritative cap
    /// (`PreprocessedRequest::stop_conditions.max_tokens`, already reduced to the remaining
    /// context and decremented on migration), not the chat request's own value: `None` preserves
    /// the frontend's omission so the provider applies its own default.
    pub fn build_body(&self, original: &Value, max_tokens: Option<u32>) -> Value {
        let mut body = match chat_request::select_carried_fields(original) {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        // Never let the client's stale cap survive; generation terminates on the frontend's
        // stop conditions, which the proxy must reproduce at the provider.
        body.remove("max_tokens");
        body.remove("max_completion_tokens");
        body.insert(
            "model".to_string(),
            Value::String(self.config.model.clone()),
        );
        body.insert("stream".to_string(), Value::Bool(true));
        body.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
        if let Some(max_tokens) = max_tokens {
            body.insert("max_tokens".to_string(), Value::from(max_tokens));
        }
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
        // While a recent 429 cooldown is active, fail fast instead of opening another
        // provider request: the proxy's own overload lease is much shorter than a
        // provider's `Retry-After`, so without this the frontend re-offers the proxy
        // into the rate limit.
        let now = now_epoch_ms();
        let until = self.rate_limited_until.load(Ordering::Relaxed);
        if until > now {
            return Err(UpstreamError::RateLimited {
                retry_after_ms: Some(until - now),
            });
        }
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
        // `connect_timeout` covers only connection establishment and reqwest has no
        // client-wide deadline, so the wait for response headers and the error body is
        // bounded explicitly here.
        let head_timeout = self.read_timeout.min(MAX_RESPONSE_HEADER_WAIT);
        let response = tokio::time::timeout(head_timeout, request.send())
            .await
            .map_err(|_| {
                UpstreamError::Transport("provider response header timed out".to_string())
            })?
            .map_err(|e| UpstreamError::Transport(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            // A stalled error body must not hang either; classify from whatever was read.
            let text = tokio::time::timeout(
                head_timeout,
                read_body_limited(response, MAX_ERROR_BODY_BYTES),
            )
            .await
            .unwrap_or_default();
            let error = UpstreamError::from_status(status.as_u16(), &text, retry_after.as_deref());
            if let UpstreamError::RateLimited { retry_after_ms } = &error {
                self.note_rate_limited(retry_after_ms.unwrap_or(DEFAULT_RATE_LIMIT_COOLDOWN_MS));
            }
            return Err(error);
        }

        let mut state = SseState::new(Box::pin(response.bytes_stream()), self.read_timeout);
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

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

/// Read at most `limit` bytes of a response body, for error classification. The
/// unused remainder is dropped so a hostile provider cannot force a huge allocation.
async fn read_body_limited(response: reqwest::Response, limit: usize) -> String {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        let remaining = limit.saturating_sub(body.len());
        if remaining == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    String::from_utf8_lossy(&body).into_owned()
}

enum Dispatch {
    Event(Result<Value, UpstreamError>),
    Done,
    Ignored,
}

/// Incremental SSE parser: handles CRLF, comments, multi-line data, `[DONE]` and events split
/// across network chunks. Yields `chat.completion.chunk` objects and ends after `[DONE]`.
struct SseState {
    bytes: ByteStream,
    buffer: BytesMut,
    data_lines: Vec<String>,
    data_bytes: usize,
    read_timeout: Duration,
    finish_grace: Duration,
    finish_deadline: Option<tokio::time::Instant>,
    done: bool,
    saw_finish_reason: bool,
    eof: bool,
}

impl SseState {
    fn new(bytes: ByteStream, read_timeout: Duration) -> Self {
        Self {
            bytes,
            buffer: BytesMut::new(),
            data_lines: Vec::new(),
            data_bytes: 0,
            read_timeout,
            finish_grace: read_timeout.min(FINISH_GRACE),
            finish_deadline: None,
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
            let wait = if let Some(deadline) = self.finish_deadline {
                tokio::time::timeout_at(deadline, self.bytes.next()).await
            } else {
                tokio::time::timeout(self.read_timeout, self.bytes.next()).await
            };
            match wait {
                Err(_) if self.finish_deadline.is_some() => {
                    // The provider reported `finish_reason` and then kept the connection
                    // open without `[DONE]`; end the stream rather than waiting out the
                    // full read timeout (or forever, if keepalive comments keep arriving)
                    // for a chunk that may never come.
                    return None;
                }
                Err(_) => {
                    return Some(Err(UpstreamError::Transport(
                        "provider read timed out".to_string(),
                    )));
                }
                Ok(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                    // Only the unterminated tail counts toward the line cap; complete
                    // lines are drained above and bounded by the event cap.
                    let tail = self
                        .buffer
                        .rsplit(|&byte| byte == b'\n')
                        .next()
                        .map_or(0, <[u8]>::len);
                    if tail > MAX_SSE_LINE_BYTES {
                        return Some(Err(UpstreamError::StreamBroken(format!(
                            "SSE line exceeds {MAX_SSE_LINE_BYTES} bytes"
                        ))));
                    }
                }
                Ok(Some(Err(error))) => {
                    return Some(Err(UpstreamError::Transport(error.to_string())));
                }
                Ok(None) => self.eof = true,
            }
        }
    }

    /// Pull one complete line out of the buffer, normalizing a trailing CR.
    fn take_line(&mut self) -> Option<String> {
        let end = self.buffer.iter().position(|&byte| byte == b'\n')?;
        // `split_to` is O(1): it advances the buffer instead of shifting the tail.
        let mut line = self.buffer.split_to(end + 1);
        line.truncate(end);
        if line.last() == Some(&b'\r') {
            line.truncate(line.len() - 1);
        }
        Some(String::from_utf8_lossy(&line).into_owned())
    }

    fn handle_line(&mut self, line: &str) -> Dispatch {
        if line.len() > MAX_SSE_LINE_BYTES {
            return Dispatch::Event(Err(UpstreamError::StreamBroken(format!(
                "SSE line exceeds {MAX_SSE_LINE_BYTES} bytes"
            ))));
        }
        if line.is_empty() {
            return self.dispatch_event();
        }
        if line.starts_with(':') {
            return Dispatch::Ignored;
        }
        if let Some(value) = line.strip_prefix("data:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            self.data_bytes = self.data_bytes.saturating_add(value.len() + 1);
            if self.data_lines.len() >= MAX_SSE_EVENT_LINES || self.data_bytes > MAX_SSE_EVENT_BYTES
            {
                self.data_lines.clear();
                self.data_bytes = 0;
                return Dispatch::Event(Err(UpstreamError::StreamBroken(
                    "SSE event exceeds the size limit".to_string(),
                )));
            }
            self.data_lines.push(value.to_string());
        }
        Dispatch::Ignored
    }

    fn dispatch_event(&mut self) -> Dispatch {
        if self.data_lines.is_empty() {
            return Dispatch::Ignored;
        }
        let data = self.data_lines.join("\n");
        self.data_lines.clear();
        self.data_bytes = 0;
        let data = data.trim();
        // An empty `data:` field is a keepalive, not a JSON event.
        if data.is_empty() {
            return Dispatch::Ignored;
        }
        if data == "[DONE]" {
            self.done = true;
            return Dispatch::Done;
        }
        match serde_json::from_str::<Value>(data) {
            Err(error) => Dispatch::Event(Err(UpstreamError::StreamBroken(format!(
                "invalid SSE data: {error}"
            )))),
            Ok(Value::Object(mut map)) => match map.remove("error") {
                Some(error) if is_stream_error(&error) => {
                    Dispatch::Event(Err(stream_error(&error)))
                }
                _ => {
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
            self.finish_deadline
                .get_or_insert_with(|| tokio::time::Instant::now() + self.finish_grace);
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

/// Classify an in-stream provider error. Providers such as OpenRouter put an HTTP-like
/// status in `code`/`status`; rate-limit and overload errors must map to the same
/// retryable variants as the HTTP status would, so failover sees them. Everything else
/// stays `InStream`.
fn stream_error(error: &Value) -> UpstreamError {
    match stream_error_status(error) {
        Some(429) => UpstreamError::RateLimited {
            retry_after_ms: None,
        },
        Some(status) if status == 408 || (500..=599).contains(&status) => {
            UpstreamError::Unavailable { status }
        }
        _ => UpstreamError::InStream(error_message(error)),
    }
}

fn stream_error_status(error: &Value) -> Option<u16> {
    ["code", "status"].iter().find_map(|key| {
        let value = error.get(*key)?;
        value
            .as_u64()
            .and_then(|number| u16::try_from(number).ok())
            .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
    })
}

/// Only a non-empty string or object under `error` is a provider error; some
/// providers serialise the optional field as `null` on every healthy chunk.
fn is_stream_error(error: &Value) -> bool {
    match error {
        Value::Null => false,
        Value::String(text) => !text.is_empty(),
        Value::Object(map) => !map.is_empty(),
        _ => false,
    }
}
