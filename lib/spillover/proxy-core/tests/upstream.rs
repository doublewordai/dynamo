// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the provider client and SSE parsing, against a local raw HTTP server.

use std::collections::BTreeMap;
use std::time::Duration;

use dw_proxy_core::errors::UpstreamError;
use dw_proxy_core::upstream::{ProviderConfig, UpstreamClient};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

const KEY_ENV: &str = "DW_PROXY_CORE_TEST_KEY";

fn config(base_url: String) -> ProviderConfig {
    ProviderConfig {
        name: "test".to_string(),
        base_url,
        api_key_env: KEY_ENV.to_string(),
        model: "test/model".to_string(),
        provider_preferences: None,
        body_overrides: None,
        extra_headers: BTreeMap::new(),
        connect_timeout_ms: 2_000,
        read_timeout_ms: 120_000,
        thinking_dialect: Default::default(),
        thinking_strict: false,
    }
}

/// One response: raw header bytes then body chunks written with a delay and a flush.
struct Mock {
    head: String,
    chunks: Vec<(Vec<u8>, u64)>,
}

async fn start_server(mock: Mock) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        socket.write_all(mock.head.as_bytes()).await.unwrap();
        for (data, delay_ms) in mock.chunks {
            if delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
            if socket.write_all(&data).await.is_err() {
                break;
            }
            let _ = socket.flush().await;
        }
        let _ = socket.shutdown().await;
        request
    });
    (format!("http://{addr}/v1"), handle)
}

fn sse(headers: &[(&str, &str)], events: &[&str]) -> Mock {
    let mut head = String::from("HTTP/1.1 200 OK\r\n");
    head.push_str("Content-Type: text/event-stream\r\n");
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("Connection: close\r\n\r\n");
    Mock {
        head,
        chunks: events
            .iter()
            .map(|event| (event.as_bytes().to_vec(), 0))
            .collect(),
    }
}

fn http_error(status: &str, headers: &[(&str, &str)], body: &str) -> Mock {
    let mut head = format!("HTTP/1.1 {status}\r\n");
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    head.push_str("Connection: close\r\n\r\n");
    Mock {
        head,
        chunks: vec![(body.as_bytes().to_vec(), 0)],
    }
}

async fn read_request(socket: &mut TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut scratch = [0u8; 1024];
    loop {
        let n = socket.read(&mut scratch).await.unwrap();
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&scratch[..n]);
        if let Some(head_end) = find(&buffer, b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buffer[..head_end]);
            let content_length = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if buffer.len() >= head_end + 4 + content_length {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buffer).into_owned()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn client(base_url: String) -> UpstreamClient {
    client_with(config(base_url))
}

/// `set_var` is process-global and unsafe: serialize it with every env read (`new`) so test
/// threads never race.
fn client_with(provider: ProviderConfig) -> UpstreamClient {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    unsafe { std::env::set_var(KEY_ENV, "test-key") };
    UpstreamClient::new(provider).unwrap()
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

async fn chat(client: &UpstreamClient, body: Value) -> Result<Vec<Value>, UpstreamError> {
    let stream = client.stream_chat(body).await?;
    stream.collect::<Vec<_>>().await.into_iter().collect()
}

/// `stream_chat`'s Ok type is a stream without `Debug`, so unwrap by hand.
async fn chat_error(client: &UpstreamClient, body: Value) -> UpstreamError {
    match client.stream_chat(body).await {
        Ok(_) => panic!("expected stream_chat to fail"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn normal_stream_with_comments_and_done() {
    let (base, server) = start_server(sse(
        &[],
        &[
            ": OPENROUTER PROCESSING\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data:{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ],
    ))
    .await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks[0],
        json!({"choices": [{"delta": {"content": "hi"}}]})
    );
    assert_eq!(
        chunks[1],
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    );
}

#[tokio::test]
async fn events_split_across_network_chunks() {
    let (base, server) = start_server(sse(
        &[],
        &[
            ": comment\n\ndata: {\"cho",
            "ices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
    ))
    .await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        chunks,
        vec![json!({"choices": [{"delta": {"content": "hi"}}]})]
    );
}

#[tokio::test]
async fn crlf_line_endings() {
    let (base, server) = start_server(sse(
        &[],
        &[
            ": OPENROUTER PROCESSING\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n",
            "data: [DONE]\r\n\r\n",
        ],
    ))
    .await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        chunks,
        vec![json!({"choices": [{"delta": {"content": "hi"}}]})]
    );
}

#[tokio::test]
async fn rate_limited_429_retry_after_seconds() {
    let (base, server) = start_server(http_error(
        "429 Too Many Requests",
        &[("Retry-After", "3")],
        "slow down",
    ))
    .await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert_eq!(
        error,
        UpstreamError::RateLimited {
            retry_after_ms: Some(3000)
        }
    );
}

#[tokio::test]
async fn unavailable_503() {
    let (base, server) = start_server(http_error(
        "503 Service Unavailable",
        &[],
        "upstream is down",
    ))
    .await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert_eq!(error, UpstreamError::Unavailable { status: 503 });
}

#[tokio::test]
async fn rejected_400_uses_error_message() {
    let (base, server) = start_server(http_error(
        "400 Bad Request",
        &[],
        "{\"error\":{\"message\":\"bad model\"}}",
    ))
    .await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert_eq!(
        error,
        UpstreamError::Rejected {
            status: 400,
            message: "bad model".to_string()
        }
    );
}

#[tokio::test]
async fn in_stream_error_after_a_chunk() {
    let (base, server) = start_server(sse(
        &[],
        &[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"error\":{\"message\":\"boom\"}}\n\n",
        ],
    ))
    .await;
    let mut stream = client(base)
        .stream_chat(json!({"messages": []}))
        .await
        .unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, json!({"choices": [{"delta": {"content": "hi"}}]}));
    let second = stream.next().await.unwrap().unwrap_err();
    server.await.unwrap();
    assert_eq!(second, UpstreamError::InStream("boom".to_string()));
}

#[tokio::test]
async fn in_stream_error_as_the_first_event_fails_the_call() {
    let (base, server) =
        start_server(sse(&[], &["data: {\"error\":{\"message\":\"boom\"}}\n\n"])).await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert_eq!(error, UpstreamError::InStream("boom".to_string()));
}

#[tokio::test]
async fn connection_closed_mid_stream() {
    let (base, server) = start_server(sse(
        &[],
        &["data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"],
    ))
    .await;
    let mut stream = client(base)
        .stream_chat(json!({"messages": []}))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    let broken = stream.next().await.unwrap().unwrap_err();
    server.await.unwrap();
    assert!(matches!(broken, UpstreamError::StreamBroken(_)));
}

#[tokio::test]
async fn body_is_built_from_carried_fields() {
    let (base, server) = start_server(sse(
        &[],
        &[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
    ))
    .await;
    let mut provider = config(base);
    provider.provider_preferences = Some(json!({"order": ["a"]}));
    provider.body_overrides = Some(json!({
        "reasoning": {"effort": "low"},
        "temperature": 0
    }));
    provider.extra_headers = BTreeMap::from([("X-Custom".to_string(), "yes".to_string())]);
    let client = client_with(provider);

    let original = json!({
        "model": "ignored/model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "nvext": {"extra_fields": ["something"]},
        "unknown_field": 1,
        "user": "u-1"
    });
    let body = client.build_body(&original, Some(2048));
    chat(&client, body.clone()).await.unwrap();

    let request = server.await.unwrap();
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    assert!(request.contains("authorization: Bearer test-key"));
    assert!(request.contains("accept: text/event-stream"));
    assert!(request.contains("x-custom: yes"));

    let sent_body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        sent_body,
        json!({
            "model": "test/model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "provider": {"order": ["a"]},
            "reasoning": {"effort": "low"},
            "temperature": 0,
            "max_tokens": 2048,
            "user": "u-1"
        })
    );
    assert_eq!(sent_body, body);
}

#[test]
fn body_uses_the_authoritative_max_tokens_not_the_chat_request_cap() {
    let client = client("http://127.0.0.1:1/v1".to_string());
    let original = json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4096,
        "max_completion_tokens": 8192
    });
    // The frontend's stop_conditions cap wins, even when the chat request carries its own.
    let body = client.build_body(&original, Some(37));
    assert_eq!(body["max_tokens"], json!(37));
    assert!(body.get("max_completion_tokens").is_none());

    // Omitting the cap preserves the frontend's omission rather than forwarding a stale one.
    let body = client.build_body(&original, None);
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("max_completion_tokens").is_none());
}

#[tokio::test]
async fn dropping_the_stream_closes_the_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        socket
            .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
            .await
            .unwrap();
        socket.flush().await.unwrap();
        let mut scratch = [0u8; 64];
        loop {
            match socket.read(&mut scratch).await {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
        }
    });

    let stream = client(format!("http://{addr}/v1"))
        .stream_chat(json!({"messages": []}))
        .await
        .unwrap();
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("connection should close when the stream is dropped")
        .unwrap();
}

#[tokio::test]
async fn new_fails_without_the_api_key_env() {
    let missing = "DW_PROXY_CORE_TEST_KEY_MISSING";
    let mut provider = config("http://127.0.0.1:1/v1".to_string());
    provider.api_key_env = missing.to_string();
    let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    assert!(UpstreamClient::new(provider).is_err());
}

#[tokio::test]
async fn empty_data_keepalive_is_ignored() {
    let (base, server) = start_server(sse(
        &[],
        &[
            "data:\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
    ))
    .await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        chunks,
        vec![json!({"choices": [{"delta": {"content": "hi"}}]})]
    );
}

#[tokio::test]
async fn done_with_trailing_whitespace_is_recognized() {
    let (base, server) = start_server(sse(
        &[],
        &[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE] \n\n",
        ],
    ))
    .await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(chunks.len(), 1);
}

#[tokio::test]
async fn null_error_field_is_not_an_error() {
    let (base, server) = start_server(sse(
        &[],
        &[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"error\":null}\n\n",
            "data: [DONE]\n\n",
        ],
    ))
    .await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        chunks,
        vec![json!({"choices": [{"delta": {"content": "hi"}}]})]
    );
}

#[tokio::test]
async fn oversized_sse_line_is_rejected() {
    let giant = format!("data: {}", "A".repeat(2 * 1024 * 1024));
    let (base, server) = start_server(sse(&[], &[&giant])).await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    drop(server);
    assert!(
        matches!(error, UpstreamError::StreamBroken(ref message) if message.contains("SSE line")),
        "unexpected {error:?}"
    );
}

#[tokio::test]
async fn too_many_sse_data_lines_are_rejected() {
    let mut event = String::new();
    for _ in 0..10_100 {
        event.push_str("data: x\n");
    }
    let (base, server) = start_server(sse(&[], &[&event])).await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    drop(server);
    assert!(
        matches!(error, UpstreamError::StreamBroken(ref message) if message.contains("SSE event")),
        "unexpected {error:?}"
    );
}

#[tokio::test]
async fn stalled_provider_read_times_out() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        socket.flush().await.unwrap();
        // Never send a body byte: the read-idle timeout must fire.
        tokio::time::sleep(Duration::from_secs(30)).await;
    });

    let provider = config(format!("http://{addr}/v1"));
    let client = client_with(provider).with_read_timeout(Duration::from_millis(100));
    let error = chat_error(&client, json!({"messages": []})).await;
    drop(server);
    assert!(
        matches!(error, UpstreamError::Transport(ref message) if message.contains("timed out")),
        "unexpected {error:?}"
    );
}

#[tokio::test]
async fn provider_redirect_is_not_followed() {
    let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target_addr = target.local_addr().unwrap();
    let target_task = tokio::spawn(async move {
        matches!(
            tokio::time::timeout(Duration::from_millis(300), target.accept()).await,
            Ok(Ok(_))
        )
    });

    let location = format!("http://{target_addr}/evil");
    let (base, server) = start_server(http_error(
        "307 Temporary Redirect",
        &[("Location", location.as_str())],
        "",
    ))
    .await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert!(
        matches!(error, UpstreamError::Rejected { status: 307, .. }),
        "unexpected {error:?}"
    );
    assert!(!target_task.await.unwrap(), "redirect target was contacted");
}

#[tokio::test]
async fn coalesced_events_in_one_chunk_all_parse() {
    let mut burst = String::new();
    for index in 0..500 {
        burst.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{index}\"}}}}]}}\n\n"
        ));
    }
    burst.push_str("data: [DONE]\n\n");
    let (base, server) = start_server(sse(&[], &[&burst])).await;
    let chunks = chat(&client(base), json!({"messages": []})).await.unwrap();
    server.await.unwrap();
    assert_eq!(chunks.len(), 500);
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "0");
    assert_eq!(chunks[499]["choices"][0]["delta"]["content"], "499");
}

#[test]
fn from_status_parses_retry_after_forms() {
    assert_eq!(
        UpstreamError::from_status(429, "", Some("12")),
        UpstreamError::RateLimited {
            retry_after_ms: Some(12_000)
        }
    );
    assert_eq!(
        UpstreamError::from_status(429, "", Some("Thu, 01 Jan 1970 00:00:00 GMT")),
        UpstreamError::RateLimited {
            retry_after_ms: Some(0)
        }
    );
    let far = UpstreamError::from_status(429, "", Some("Wed, 21 Oct 2099 07:28:00 GMT"));
    match far {
        UpstreamError::RateLimited {
            retry_after_ms: Some(ms),
        } => assert!(ms > 1_000_000_000_000),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(
        UpstreamError::from_status(429, "", Some("not a date")),
        UpstreamError::RateLimited {
            retry_after_ms: None
        }
    );
    assert_eq!(
        UpstreamError::from_status(429, "", None),
        UpstreamError::RateLimited {
            retry_after_ms: None
        }
    );
}

#[test]
fn from_status_truncates_long_bodies() {
    let body = "x".repeat(600);
    match UpstreamError::from_status(400, &body, None) {
        UpstreamError::Rejected { message, .. } => assert_eq!(message.chars().count(), 500),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn from_status_caps_json_error_message() {
    let body = json!({"error": {"message": "x".repeat(100_000)}}).to_string();
    match UpstreamError::from_status(400, &body, None) {
        UpstreamError::Rejected { message, .. } => assert_eq!(message.chars().count(), 500),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn from_status_ignores_overflowing_retry_after_date() {
    assert_eq!(
        UpstreamError::from_status(
            429,
            "",
            Some("Wed, 21 Oct 9223372036854775807 07:28:00 GMT")
        ),
        UpstreamError::RateLimited {
            retry_after_ms: None
        }
    );
}

#[test]
fn retry_elsewhere_classification() {
    assert!(
        UpstreamError::RateLimited {
            retry_after_ms: None
        }
        .retry_elsewhere()
    );
    assert!(UpstreamError::Unavailable { status: 503 }.retry_elsewhere());
    assert!(UpstreamError::Transport("x".to_string()).retry_elsewhere());
    assert!(UpstreamError::StreamBroken("x".to_string()).retry_elsewhere());
    assert!(UpstreamError::InStream("x".to_string()).retry_elsewhere());
    assert!(
        !UpstreamError::Rejected {
            status: 400,
            message: "bad".to_string()
        }
        .retry_elsewhere()
    );
}

#[test]
fn payment_required_402_is_retryable() {
    let error = UpstreamError::from_status(402, "out of credits", None);
    assert_eq!(error, UpstreamError::Unavailable { status: 402 });
    assert!(error.retry_elsewhere());
}

#[test]
fn provider_config_debug_redacts_secrets() {
    let mut provider = config("https://openrouter.ai/api/v1".to_string());
    provider.extra_headers = BTreeMap::from([
        ("X-Gateway-Token".to_string(), "super-secret".to_string()),
        ("X-Trace".to_string(), "trace-value".to_string()),
    ]);
    provider.body_overrides = Some(json!({"api_key": "also-secret"}));

    let rendered = format!("{provider:?}");
    assert!(rendered.contains("X-Gateway-Token"), "{rendered}");
    assert!(rendered.contains("<redacted>"), "{rendered}");
    assert!(!rendered.contains("super-secret"), "{rendered}");
    assert!(!rendered.contains("trace-value"), "{rendered}");
    assert!(!rendered.contains("also-secret"), "{rendered}");
}

/// Accept one connection, read the request, write `head`, then hold the socket open
/// without ever sending a body.
async fn start_stalling_server(head: &str) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let head = head.to_string();
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket.write_all(head.as_bytes()).await.unwrap();
        let _ = socket.flush().await;
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    (format!("http://{addr}/v1"), handle)
}

/// Accept one connection, read the request, and never write a response header.
async fn start_silent_server() -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    (format!("http://{addr}/v1"), handle)
}

/// Serve SSE events and then keep the connection open without `[DONE]`.
async fn start_held_open_sse_server(events: &[&str]) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let events: Vec<String> = events.iter().map(|event| (*event).to_string()).collect();
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n",
            )
            .await
            .unwrap();
        for event in &events {
            socket.write_all(event.as_bytes()).await.unwrap();
        }
        let _ = socket.flush().await;
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    (format!("http://{addr}/v1"), handle)
}

#[tokio::test]
async fn stalled_response_headers_time_out() {
    let (base, server) = start_silent_server().await;
    let client = client_with(config(base)).with_read_timeout(Duration::from_millis(200));
    let started = std::time::Instant::now();
    let error = chat_error(&client, json!({"messages": []})).await;
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "header wait was not bounded"
    );
    drop(server);
    assert!(
        matches!(error, UpstreamError::Transport(ref message) if message.contains("header")),
        "unexpected {error:?}"
    );
}

#[tokio::test]
async fn stalled_error_body_times_out_without_hanging() {
    let (base, server) = start_stalling_server(
        "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\nConnection: close\r\n\r\n",
    )
    .await;
    let client = client_with(config(base)).with_read_timeout(Duration::from_millis(200));
    let started = std::time::Instant::now();
    let error = chat_error(&client, json!({"messages": []})).await;
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "error body read was not bounded"
    );
    drop(server);
    assert_eq!(error, UpstreamError::Unavailable { status: 500 });
}

#[tokio::test]
async fn finish_reason_without_done_ends_after_a_short_grace() {
    let (base, server) = start_held_open_sse_server(&[
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    ])
    .await;
    let client = client_with(config(base)).with_read_timeout(Duration::from_millis(200));
    let chunks = tokio::time::timeout(
        Duration::from_secs(3),
        chat(&client, json!({"messages": []})),
    )
    .await
    .expect("stream should end after the finish-reason grace, not the read timeout")
    .unwrap();
    drop(server);
    assert_eq!(chunks.len(), 1);
}

#[tokio::test]
async fn rate_limit_cooldown_skips_the_provider() {
    let (base, server) = start_server(http_error(
        "429 Too Many Requests",
        &[("Retry-After", "60")],
        "slow down",
    ))
    .await;
    let client = client(base);

    let first = chat_error(&client, json!({"messages": []})).await;
    assert!(
        matches!(first, UpstreamError::RateLimited { retry_after_ms: Some(ms) } if ms >= 60_000),
        "unexpected {first:?}"
    );

    let started = std::time::Instant::now();
    let second = chat_error(&client, json!({"messages": []})).await;
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the cooldown must fail fast"
    );
    assert!(
        matches!(second, UpstreamError::RateLimited { .. }),
        "the cooldown must not call the provider: {second:?}"
    );
    server.await.unwrap();
}

#[tokio::test]
async fn in_stream_error_with_429_code_is_rate_limited() {
    let (base, server) = start_server(sse(
        &[],
        &["data: {\"error\":{\"message\":\"rate limited\",\"code\":429}}\n\n"],
    ))
    .await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert_eq!(
        error,
        UpstreamError::RateLimited {
            retry_after_ms: None
        }
    );
}

#[tokio::test]
async fn in_stream_error_with_502_code_is_unavailable() {
    let (base, server) = start_server(sse(
        &[],
        &["data: {\"error\":{\"message\":\"bad gateway\",\"status\":\"502\"}}\n\n"],
    ))
    .await;
    let error = chat_error(&client(base), json!({"messages": []})).await;
    server.await.unwrap();
    assert_eq!(error, UpstreamError::Unavailable { status: 502 });
}
