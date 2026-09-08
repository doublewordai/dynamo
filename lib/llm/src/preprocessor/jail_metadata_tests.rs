// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{ANNOTATION_PAYLOAD_USAGE, OpenAIPreprocessor};
use crate::protocols::{
    Annotated,
    common::metrics::LLMMetricAnnotation,
    openai::{
        ParsingOptions,
        chat_completions::{NvCreateChatCompletionStreamResponse, aggregator::DeltaAggregator},
    },
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionToolChoiceOption, FinishReason,
};
use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
use futures::{StreamExt, stream};
use serde_json::{Value, json};

fn chunk(
    content: &str,
    finish_reason: Option<FinishReason>,
    nvext: Option<Value>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    Annotated::from_data(
        serde_json::from_value(json!({
            "id": "chatcmpl-metadata-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "nvext": nvext,
        }))
        .unwrap(),
    )
}

async fn parse(
    parser: &str,
    chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    parse_with_choice(parser, None, chunks).await
}

async fn parse_with_choice(
    parser: &str,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let output: Vec<_> = OpenAIPreprocessor::apply_tool_calling_jail(
        Some(parser.to_string()),
        tool_choice,
        None,
        false,
        stream::iter(chunks),
    )
    .collect()
    .await;
    assert!(output.iter().all(|chunk| chunk.error.is_none()));
    output
}

fn metrics(chunk_tokens: usize, output_tokens: usize) -> LLMMetricAnnotation {
    LLMMetricAnnotation {
        input_tokens: 103,
        chunk_tokens,
        output_tokens,
        ..Default::default()
    }
}

fn metadata_only(
    finish_reason: Option<FinishReason>,
    chunk_tokens: usize,
    output_tokens: usize,
    nvext: Value,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut response = chunk("", finish_reason, Some(nvext));
    let data = response.data.as_mut().unwrap();
    data.inner.choices[0].delta.content = None;
    data.inner.choices[0].delta.role = None;
    data.llm_metrics = Some(metrics(chunk_tokens, output_tokens));
    response
}

fn usage_chunk(
    completion_tokens: usize,
    payload_only: bool,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut response = chunk("", None, None);
    let data = response.data.as_mut().unwrap();
    data.inner.choices.clear();
    data.inner.usage = Some(
        serde_json::from_value(json!({
            "prompt_tokens": 103,
            "completion_tokens": completion_tokens,
            "total_tokens": 103 + completion_tokens,
        }))
        .unwrap(),
    );
    if payload_only {
        response.event = Some(ANNOTATION_PAYLOAD_USAGE.to_string());
    }
    response
}

fn assert_metric_tokens(
    output: &[Annotated<NvCreateChatCompletionStreamResponse>],
    expected_tokens: usize,
) {
    // Match the HTTP collector's typed-field-first, annotation-fallback contract.
    let observed: Vec<_> = output
        .iter()
        .filter_map(|response| {
            response
                .data
                .as_ref()
                .and_then(|data| data.llm_metrics.clone())
                .or_else(|| {
                    LLMMetricAnnotation::from_annotation(response)
                        .ok()
                        .flatten()
                })
        })
        .collect();
    assert_eq!(
        observed
            .iter()
            .map(|metrics| metrics.chunk_tokens)
            .sum::<usize>(),
        expected_tokens,
        "parser output and EOF metadata must not count backend tokens twice"
    );
    assert_eq!(
        observed.iter().map(|metrics| metrics.output_tokens).max(),
        Some(expected_tokens)
    );
}

fn completion_ids(chunks: &[Annotated<NvCreateChatCompletionStreamResponse>]) -> Vec<u64> {
    chunks
        .iter()
        .filter_map(|chunk| chunk.data.as_ref()?.nvext.as_ref())
        .filter_map(|nvext| nvext.get("completion_token_ids")?.as_array())
        .flatten()
        .map(|id| id.as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn kimi_no_tools_preserves_response_metadata_through_aggregation() {
    let mut terminal = chunk(
        "",
        Some(FinishReason::Stop),
        Some(json!({
            "completion_token_ids": [],
            "engine_data": {
                "prompt_token_ids": [101, 102],
                "completion_token_ids": [163587, 4503, 163589, 28492, 163588, 4503, 163589, 163586],
            },
            "worker_id": {"prefill_worker_id": 7, "decode_worker_id": 8},
            "timing": {"ttft_ms": 12.5},
            "stop_reason": 163586,
        })),
    );
    let terminal_delta = &mut terminal.data.as_mut().unwrap().inner.choices[0].delta;
    terminal_delta.content = None;
    terminal_delta.role = None;
    let output = parse(
        "kimi_k3",
        vec![
            chunk(
                "<|open|>response<|sep|>Hello",
                None,
                Some(json!({
                    "token_ids": [101, 102],
                    "completion_token_ids": [163587, 4503, 163589, 28492],
                })),
            ),
            chunk(
                "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>",
                None,
                Some(json!({
                    "completion_token_ids": [163588, 4503, 163589, 163586],
                })),
            ),
            terminal,
        ],
    )
    .await;
    let expected_ids = vec![163587, 4503, 163589, 28492, 163588, 4503, 163589, 163586];
    assert_eq!(completion_ids(&output), expected_ids);

    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    assert_eq!(
        response.inner.choices[0].message.content,
        Some(ChatCompletionMessageContent::Text("Hello".to_string()))
    );
    assert!(response.inner.choices[0].message.tool_calls.is_none());
    assert_eq!(
        response.nvext,
        Some(json!({
            "token_ids": [101, 102],
            "completion_token_ids": expected_ids,
            "worker_id": {"prefill_worker_id": 7, "decode_worker_id": 8},
            "engine_data": {
                "prompt_token_ids": [101, 102],
                "completion_token_ids": expected_ids,
            },
            "timing": {"ttft_ms": 12.5},
            "stop_reason": 163586,
        }))
    );
}

#[tokio::test]
async fn buffered_tool_chunks_merge_metadata_and_preserve_metrics() {
    let mut chunks = vec![
        chunk(
            "<tool_call>",
            None,
            Some(json!({"completion_token_ids": [11]})),
        ),
        chunk(
            r#"{"name":"weather","arguments":"#,
            None,
            Some(json!({"completion_token_ids": [12, 13]})),
        ),
        chunk(
            r#"{"city":"London"}}</tool_call>"#,
            Some(FinishReason::Stop),
            Some(json!({
                "completion_token_ids": [14],
                "engine_data": {"complete": true},
                "worker_id": {"decode_worker_id": 8},
                "timing": {"ttft_ms": 12.5},
            })),
        ),
    ];
    for (chunk, (chunk_tokens, output_tokens)) in chunks.iter_mut().zip([(1, 1), (2, 3), (1, 4)]) {
        chunk.data.as_mut().unwrap().llm_metrics = Some(LLMMetricAnnotation {
            input_tokens: 103,
            chunk_tokens,
            output_tokens,
            cached_tokens: Some(17),
            decode_worker_id: Some(8),
            ..Default::default()
        });
    }
    let heartbeat =
        Annotated::from_annotation("metadata-heartbeat", &json!({"sequence": 1})).unwrap();
    let heartbeat_comment = heartbeat.comment.clone();
    chunks.insert(1, heartbeat);
    let output = parse("hermes", chunks).await;
    let heartbeat = output
        .iter()
        .find(|chunk| chunk.event.as_deref() == Some("metadata-heartbeat"))
        .expect("data-less annotations must pass through while content is buffered");
    assert!(heartbeat.data.is_none());
    assert_eq!(heartbeat.comment, heartbeat_comment);
    assert_eq!(completion_ids(&output), [11, 12, 13, 14]);
    let metadata: Vec<_> = output
        .iter()
        .filter_map(|chunk| chunk.data.as_ref()?.nvext.as_ref())
        .collect();
    assert_eq!(
        metadata.len(),
        1,
        "buffered input must produce one metadata batch"
    );
    assert_eq!(metadata[0]["completion_token_ids"], json!([11, 12, 13, 14]));
    let metrics: Vec<_> = output
        .iter()
        .filter_map(|chunk| chunk.data.as_ref()?.llm_metrics.as_ref())
        .collect();
    assert_eq!(metrics.len(), 1);
    assert_eq!(
        *metrics[0],
        LLMMetricAnnotation {
            input_tokens: 103,
            chunk_tokens: 4,
            output_tokens: 4,
            cached_tokens: Some(17),
            decode_worker_id: Some(8),
            ..Default::default()
        }
    );

    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    let calls = response.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "weather");
    assert_eq!(
        response.nvext,
        Some(json!({
            "completion_token_ids": [11, 12, 13, 14],
            "engine_data": {"complete": true},
            "worker_id": {"decode_worker_id": 8},
            "timing": {"ttft_ms": 12.5},
        }))
    );
}

#[tokio::test]
async fn split_tool_chunk_emits_completion_ids_exactly_once() {
    let nvext = json!({
        "completion_token_ids": [21, 22, 23, 24],
        "engine_data": {"complete": true},
        "worker_id": {"decode_worker_id": 8},
        "timing": {"ttft_ms": 12.5},
    });
    let output = parse(
        "hermes",
        vec![chunk(
            "Before.<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"London\"}}</tool_call>After.",
            Some(FinishReason::Stop),
            Some(nvext.clone()),
        )],
    )
    .await;
    let data: Vec<_> = output
        .iter()
        .filter_map(|chunk| chunk.data.as_ref())
        .collect();
    assert!(
        data.len() >= 3,
        "jail must split prefix, tool call, and suffix"
    );
    assert_eq!(data.iter().filter(|chunk| chunk.nvext.is_some()).count(), 1);
    assert_eq!(completion_ids(&output), [21, 22, 23, 24]);

    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    assert_eq!(response.nvext, Some(nvext));
    assert_eq!(
        response.inner.choices[0].message.content,
        Some(ChatCompletionMessageContent::Text(
            "Before.After.".to_string()
        ))
    );
    assert_eq!(
        response.inner.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn usage_only_terminal_metadata_survives_synthesized_tool_finish() {
    let final_metadata = json!({
        "completion_token_ids": [],
        "engine_data": {
            "prompt_token_ids": [101, 102],
            "completion_token_ids": [31, 32, 33, 34],
        },
        "worker_id": {"decode_worker_id": 8},
        "timing": {"ttft_ms": 12.5},
    });
    let mut usage = chunk("", None, Some(final_metadata.clone()));
    let usage_data = usage.data.as_mut().unwrap();
    usage_data.inner.choices.clear();
    usage_data.inner.usage = Some(
        serde_json::from_value(json!({
            "prompt_tokens": 103,
            "completion_tokens": 4,
            "total_tokens": 107,
        }))
        .unwrap(),
    );
    let output = parse(
        "hermes",
        vec![
            chunk(
                r#"<tool_call>{"name":"weather","arguments":{"city":"London"}}</tool_call>"#,
                None,
                Some(json!({"completion_token_ids": [31, 32, 33, 34]})),
            ),
            usage,
        ],
    )
    .await;
    let data: Vec<_> = output
        .iter()
        .filter_map(|chunk| chunk.data.as_ref())
        .collect();
    let terminals: Vec<_> = data
        .iter()
        .filter(|chunk| {
            chunk
                .inner
                .choices
                .iter()
                .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
        })
        .collect();
    assert_eq!(
        terminals.len(),
        1,
        "jail must synthesize the missing finish"
    );
    assert!(terminals[0].inner.usage.is_none());
    assert_eq!(
        data.iter()
            .filter(|chunk| chunk.inner.choices.is_empty() && chunk.inner.usage.is_some())
            .count(),
        1,
        "usage-only input must remain exactly once"
    );
    assert_eq!(completion_ids(&output), [31, 32, 33, 34]);
    assert_eq!(
        data.iter()
            .filter(|chunk| chunk
                .nvext
                .as_ref()
                .is_some_and(|nvext| nvext.get("engine_data").is_some()))
            .count(),
        1,
        "synthetic finish and usage must not duplicate final metadata"
    );

    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    let mut expected_metadata = final_metadata;
    expected_metadata["completion_token_ids"] = json!([31, 32, 33, 34]);
    assert_eq!(response.nvext, Some(expected_metadata));
    assert_eq!(
        response.inner.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    let usage = response.inner.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 103);
    assert_eq!(usage.completion_tokens, 4);
    assert_eq!(usage.total_tokens, 107);
}

#[tokio::test]
async fn absent_response_metadata_stays_absent() {
    let output = parse(
        "kimi_k3",
        vec![chunk(
            "<|open|>response<|sep|>Hello<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>",
            Some(FinishReason::Stop),
            None,
        )],
    )
    .await;
    assert!(output.iter().any(|chunk| chunk.data.is_some()));
    assert!(
        output
            .iter()
            .filter_map(|chunk| chunk.data.as_ref())
            .all(|chunk| chunk.nvext.is_none())
    );

    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    assert!(response.nvext.is_none());
    assert_eq!(
        response.inner.choices[0].message.content,
        Some(ChatCompletionMessageContent::Text("Hello".to_string()))
    );
}

#[tokio::test]
async fn eof_metadata_survives_without_parser_output_before_client_usage() {
    let final_engine_data = json!({
        "prompt_token_ids": [101, 102],
        "completion_token_ids": [30, 31],
        "completion_logprobs": [-0.1, -0.2],
    });
    // Required mode buffers these content-less choices and produces no choice
    // at EOF. The adapter must provide its own metadata-only response envelope.
    let output = parse_with_choice(
        "hermes",
        Some(ChatCompletionToolChoiceOption::Required),
        vec![
            metadata_only(
                None,
                1,
                1,
                json!({
                    "completion_token_ids": [30],
                    "engine_data": {"phase": "partial"},
                    "worker_id": {"decode_worker_id": 8},
                }),
            ),
            metadata_only(
                Some(FinishReason::Stop),
                1,
                2,
                json!({
                    "completion_token_ids": [31],
                    "engine_data": final_engine_data,
                    "timing": {"total_ms": 12.5},
                    "prompt_logprobs": [null, {"token": 101}],
                }),
            ),
            usage_chunk(2, false),
        ],
    )
    .await;

    assert_eq!(
        output.len(),
        2,
        "only EOF metadata and client usage are emitted"
    );
    let metadata = output[0].data.as_ref().unwrap();
    assert!(metadata.inner.choices.is_empty());
    assert!(metadata.inner.usage.is_none());
    assert_eq!(metadata.inner.id, "chatcmpl-metadata-test");
    assert_eq!(metadata.inner.object, "chat.completion.chunk");
    assert_eq!(metadata.inner.model, "test-model");
    assert!(output[0].event.is_none());
    assert!(output[0].comment.is_none());
    assert_eq!(
        metadata.nvext,
        Some(json!({
            "completion_token_ids": [30, 31],
            "engine_data": final_engine_data,
            "worker_id": {"decode_worker_id": 8},
            "timing": {"total_ms": 12.5},
            "prompt_logprobs": [null, {"token": 101}],
        }))
    );
    let usage = output[1].data.as_ref().unwrap();
    assert!(usage.inner.choices.is_empty());
    assert_eq!(usage.inner.usage.as_ref().unwrap().completion_tokens, 2);
    assert!(usage.nvext.is_none());
    assert_eq!(completion_ids(&output), [30, 31]);
    assert_metric_tokens(&output, 2);
}

#[tokio::test]
async fn payload_usage_nvext_is_client_visible_without_duplicate_token_metrics() {
    let final_metadata = json!({
        "completion_token_ids": [51, 52],
        "engine_data": {
            "prompt_token_ids": [101, 102],
            "completion_token_ids": [51, 52],
        },
    });
    let mut text = chunk("Hello", Some(FinishReason::Stop), None);
    text.data.as_mut().unwrap().llm_metrics = Some(metrics(1, 1));
    let mut usage = usage_chunk(2, true);
    let terminal_metrics = metrics(1, 2);
    usage.comment = terminal_metrics.to_annotation::<()>().unwrap().comment;
    usage.data.as_mut().unwrap().llm_metrics = Some(terminal_metrics);
    usage.data.as_mut().unwrap().nvext = Some(final_metadata.clone());
    let output = parse("hermes", vec![text, usage]).await;

    let metadata: Vec<_> = output
        .iter()
        .filter(|response| {
            response
                .data
                .as_ref()
                .is_some_and(|data| data.nvext.is_some())
        })
        .collect();
    assert_eq!(metadata.len(), 1);
    let metadata_frame = metadata[0];
    assert!(
        metadata_frame.event.is_none(),
        "payload_usage would hide nvext from SSE"
    );
    assert!(metadata_frame.comment.is_none());
    let metadata = metadata_frame.data.as_ref().unwrap();
    assert!(metadata.inner.choices.is_empty());
    assert!(metadata.inner.usage.is_none());
    assert_eq!(metadata.nvext, Some(final_metadata.clone()));
    assert!(
        metadata.llm_metrics.is_none(),
        "EOF metadata must not replay metrics already carried by payload usage"
    );
    let payload = output.last().unwrap();
    assert_eq!(payload.event.as_deref(), Some(ANNOTATION_PAYLOAD_USAGE));
    assert!(payload.data.as_ref().unwrap().nvext.is_none());
    assert_eq!(completion_ids(&output), [51, 52]);
    assert_metric_tokens(&output, 2);

    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    assert_eq!(response.nvext, Some(final_metadata));
    assert_eq!(response.inner.usage.unwrap().completion_tokens, 2);
}

#[tokio::test]
async fn required_tool_metadata_waits_for_choices_across_payload_usage() {
    let engine_data = json!({
        "prompt_token_ids": [101, 102],
        "completion_token_ids": [60, 61, 62, 63, 64, 65, 66],
    });
    // Required mode completes arrays inline, but keeps a bare single-tool
    // object buffered for its supported EOF JSON fallback. Keep the JSON valid:
    // this cached parser does not repair a truncated required-mode array.
    let mut first = chunk(
        r#"{"name":"weather","parameters":{"ci"#,
        None,
        Some(json!({"completion_token_ids": [60, 61, 62]})),
    );
    first.data.as_mut().unwrap().llm_metrics = Some(metrics(3, 3));
    let mut second = chunk(
        r#"ty":"London"}}"#,
        None,
        Some(json!({
            "completion_token_ids": [63, 64, 65, 66],
            "engine_data": engine_data,
        })),
    );
    second.data.as_mut().unwrap().llm_metrics = Some(metrics(4, 7));
    let output = parse_with_choice(
        "hermes",
        Some(ChatCompletionToolChoiceOption::Required),
        vec![first, second, usage_chunk(7, true)],
    )
    .await;

    let metadata: Vec<_> = output
        .iter()
        .filter_map(|response| response.data.as_ref().filter(|data| data.nvext.is_some()))
        .collect();
    assert_eq!(metadata.len(), 1);
    assert!(!metadata[0].inner.choices.is_empty());
    assert_eq!(
        metadata[0].nvext.as_ref().unwrap()["engine_data"],
        engine_data
    );
    let payload = output.last().unwrap();
    assert_eq!(payload.event.as_deref(), Some(ANNOTATION_PAYLOAD_USAGE));
    assert!(payload.data.as_ref().unwrap().nvext.is_none());
    assert_eq!(completion_ids(&output), [60, 61, 62, 63, 64, 65, 66]);
    assert!(metadata[0].llm_metrics.is_none());
    assert_eq!(
        payload
            .data
            .as_ref()
            .unwrap()
            .llm_metrics
            .as_ref()
            .unwrap()
            .chunk_tokens,
        7,
        "payload usage must consume buffered metrics before the tool choice is released"
    );
    assert_metric_tokens(&output, 7);

    assert!(
        output
            .iter()
            .filter_map(|chunk| chunk.data.as_ref())
            .any(|data| {
                data.inner.choices.iter().any(|choice| {
                    choice
                        .delta
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| !calls.is_empty())
                })
            }),
        "required-mode EOF fallback must parse the complete tool object: {output:#?}"
    );
    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    let calls = response.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "weather");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
        json!({"city": "London"})
    );
    assert_eq!(
        response.inner.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
}

fn deepseek_unclosed_tool_block() -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut response = chunk(
        "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">London</｜DSML｜parameter>\n\
</｜DSML｜invoke>",
        None,
        Some(json!({
            "completion_token_ids": [71, 72],
            "engine_data": {"completion_token_ids": [71, 72]},
        })),
    );
    response.data.as_mut().unwrap().llm_metrics = Some(metrics(2, 2));
    response
}

#[tokio::test]
async fn deepseek_clean_eof_recovers_tool_call_and_metadata() {
    let output = parse("deepseek_v4", vec![deepseek_unclosed_tool_block()]).await;
    assert_eq!(completion_ids(&output), [71, 72]);
    assert_metric_tokens(&output, 2);
    let response = DeltaAggregator::apply(stream::iter(output), ParsingOptions::default())
        .await
        .unwrap();
    let calls = response.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "weather");
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
        json!({"city": "London"})
    );
    assert_eq!(
        response.inner.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn typed_errors_suppress_jail_finalize_pending_metadata_and_usage() {
    // The clean-EOF control above proves this same input normally produces a
    // tool call. A failed stream must preserve its error rather than finalize it.
    for error_type in [
        ErrorType::Backend(BackendError::InvalidArgument),
        ErrorType::Backend(BackendError::EngineShutdown),
        ErrorType::Backend(BackendError::Disconnected),
    ] {
        let message = format!("typed {error_type} error");
        let error = Annotated {
            data: None,
            id: Some("terminal-error-id".to_string()),
            event: Some("error".to_string()),
            comment: Some(vec!["original transport annotation".to_string()]),
            error: Some(
                DynamoError::builder()
                    .error_type(error_type)
                    .message(&message)
                    .build(),
            ),
        };
        let output: Vec<_> = OpenAIPreprocessor::apply_tool_calling_jail(
            Some("deepseek_v4".to_string()),
            None,
            None,
            false,
            stream::iter(vec![
                deepseek_unclosed_tool_block(),
                usage_chunk(2, false),
                error,
                chunk(
                    "must not be read after error",
                    Some(FinishReason::Stop),
                    None,
                ),
            ]),
        )
        .collect()
        .await;

        assert_eq!(
            output.len(),
            1,
            "no successful finalize, EOF metadata, or held usage: {output:#?}"
        );
        let terminal = &output[0];
        assert!(terminal.data.is_none());
        assert_eq!(terminal.id.as_deref(), Some("terminal-error-id"));
        assert_eq!(terminal.event.as_deref(), Some("error"));
        assert_eq!(
            terminal.comment.as_ref().unwrap(),
            &["original transport annotation"]
        );
        let error = terminal
            .error
            .as_ref()
            .expect("original typed terminal error");
        assert_eq!(error.error_type(), error_type);
        assert_eq!(error.message(), message);
        assert!(completion_ids(&output).is_empty());
    }
}
