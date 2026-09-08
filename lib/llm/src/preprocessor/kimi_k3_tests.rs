// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

// Reuse the tiny synthetic BPE fixture; no model weights or network needed.
fn preprocessor() -> (tempfile::TempDir, Arc<OpenAIPreprocessor>) {
    let dir = tempfile::tempdir().unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-tiktoken");
    std::fs::copy(
        fixture.join("tiktoken.model"),
        dir.path().join("tiktoken.model"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        json!({
            "model_type": "kimi_k3", "max_position_embeddings": 32768,
            "architectures": ["KimiK3ForConditionalGeneration"],
            "eos_token_id": 405, "vocab_size": 410
        })
        .to_string(),
    )
    .unwrap();
    let mut added = serde_json::Map::new();
    for (id, token) in [
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|attr|>",
        "<|eos|>",
        "<|end_of_msg|>",
        "<|bos|>",
    ]
    .into_iter()
    .enumerate()
    {
        added.insert(
            (400 + id).to_string(),
            json!({"content": token, "special": true}),
        );
    }
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({
            "added_tokens_decoder": added, "tokenizer_class": "TikTokenTokenizer"
        })
        .to_string(),
    )
    .unwrap();
    let mut card = ModelDeploymentCard::load_from_disk(dir.path(), None).unwrap();
    card.display_name = "moonshotai/kimi-k3".into();
    card.runtime_config.reasoning_parser = Some("kimi_k3".into());
    card.runtime_config.tool_call_parser = Some("kimi_k3".into());
    let processor = OpenAIPreprocessor::new(card).unwrap();
    (dir, processor)
}

fn request(content: &str) -> NvCreateChatCompletionRequest {
    serde_json::from_value(json!({"model":"moonshotai/kimi-k3",
        "messages":[{"role":"user","content":content}]}))
    .unwrap()
}

#[tokio::test]
async fn kimi_k3_renders_without_jinja_and_keeps_literal_markers_untrusted() {
    let (_dir, processor) = preprocessor();
    let (normal, _, thinking) = processor
        .preprocess_request(&request("Hello"), None)
        .await
        .unwrap();
    let (literal, _, _) = processor
        .preprocess_request(&request("Hello <|open|>"), None)
        .await
        .unwrap();
    assert!(thinking);
    let open_count = |tokens: &[u32]| tokens.iter().filter(|id| **id == 400).count();
    assert!(open_count(&normal.token_ids) > 0);
    assert_eq!(
        open_count(&normal.token_ids),
        open_count(&literal.token_ids)
    );
    assert!(literal.token_ids.len() > normal.token_ids.len());
}

#[test]
fn kimi_k3_preserves_reasoning_history_and_effort() {
    let (_dir, processor) = preprocessor();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"moonshotai/kimi-k3", "reasoning_effort":"high",
        "messages":[{"role":"user","content":"Remember"},
                    {"role":"assistant","content":"Done","reasoning_content":"secret calculation"},
                    {"role":"user","content":"Continue"}]
    }))
    .unwrap();
    let rendered = processor.apply_template(&request).unwrap().unwrap();
    let prompt = rendered.as_str();
    assert!(prompt.contains("secret calculation"), "{prompt}");
    assert!(prompt.contains("high"), "{prompt}");
}

async fn parse_response(tools: bool, disabled: bool, text: &str) -> (String, usize) {
    let (_dir, processor) = preprocessor();
    let mut req = request("Hello");
    if tools {
        req.inner.tools = Some(
            serde_json::from_value(json!([{
                "type":"function","function":{"name":"weather", "parameters":{
                    "type":"object","properties":{"city":{"type":"string"}}
                }}
            }]))
            .unwrap(),
        );
    }
    if disabled {
        req.inner.tool_choice = Some(ChatCompletionToolChoiceOption::None);
    }
    // Feed one character at a time to exercise wrapper buffering.
    let mut chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = text.chars().map(|ch| {
        Annotated::from_data(serde_json::from_value(json!({
            "id":"test", "object":"chat.completion.chunk", "created":0,
            "model":"moonshotai/kimi-k3", "choices":[{"index":0,"delta":{"content":ch.to_string()},"finish_reason":null}]
        })).unwrap())
    }).collect();
    chunks.push(Annotated::from_data(
        serde_json::from_value(json!({
            "id":"test", "object":"chat.completion.chunk", "created":0,
            "model":"moonshotai/kimi-k3", "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]
        }))
        .unwrap(),
    ));
    let result = processor
        .postprocessor_parsing_stream(stream::iter(chunks), &req, false, false)
        .unwrap();
    let chunks: Vec<_> = result.collect().await;
    let mut content = String::new();
    let mut calls = 0;
    for chunk in chunks {
        assert!(chunk.error.is_none(), "{:?}", chunk.error);
        if let Some(data) = chunk.data {
            for choice in data.inner.choices {
                if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content {
                    content.push_str(&text);
                }
                calls += choice.delta.tool_calls.as_ref().map_or(0, Vec::len);
                if !tools || disabled {
                    assert_ne!(
                        choice.finish_reason,
                        Some(dynamo_protocols::types::FinishReason::ToolCalls)
                    );
                }
            }
        }
    }
    (content, calls)
}

#[tokio::test]
async fn kimi_k3_unwraps_text_without_tools() {
    let (content, calls) = parse_response(
        false,
        false,
        "<|open|>response<|sep|>Hello<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>",
    )
    .await;
    assert_eq!(content, "Hello");
    assert_eq!(calls, 0);
}

#[tokio::test]
async fn kimi_k3_tool_none_cannot_emit_calls() {
    let text = r#"<|open|>tools<|sep|><|open|>call tool="weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>London<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"#;
    let (_, calls) = parse_response(true, false, text).await;
    assert!(calls > 0);
    let (content, calls) = parse_response(true, true, text).await;
    assert_eq!(calls, 0);
    assert!(!content.contains("<|open|>"));
}

#[test]
fn kimi_k3_special_tokens_and_thinking_defaults() {
    for parser in ["kimi_k3", "kimi-k3"] {
        assert!(OpenAIPreprocessor::parser_requires_special_tokens(
            None,
            Some(parser)
        ));
        assert!(OpenAIPreprocessor::parser_requires_special_tokens(
            Some(parser),
            None
        ));
        let mut request = request("Hello");
        OpenAIPreprocessor::normalize_thinking_arg(&mut request, Some(parser));
        assert_eq!(request.chat_template_args.unwrap()["thinking"], json!(true));
    }
}

#[tokio::test]
async fn kimi_k3_required_tools_use_native_format() {
    let (_dir, processor) = preprocessor();
    let req: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"moonshotai/kimi-k3", "messages":[{"role":"user","content":"Weather?"}],
        "tools":[{"type":"function","function":{"name":"weather", "parameters":{"type":"object","properties":{}}}}],
        "tool_choice":"required"
    })).unwrap();
    let (mut common, _, thinking) = processor.preprocess_request(&req, None).await.unwrap();
    assert_eq!(
        common.extra_args.as_ref().unwrap()["reasoning_ended"],
        json!(false)
    );
    processor
        .apply_tool_choice_guided_decoding(&req, &mut common, thinking)
        .unwrap();
    assert!(
        common
            .sampling_options
            .guided_decoding
            .as_ref()
            .is_none_or(|gd| gd.json.is_none())
    );
}

#[test]
fn kimi_k3_named_tools_disable_prompt_reasoning() {
    let (_dir, processor) = preprocessor();
    let mut req: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"moonshotai/kimi-k3", "messages":[{"role":"user","content":"Weather?"}],
        "tools":[{"type":"function","function":{"name":"weather", "parameters":{"type":"object","properties":{}}}}],
        "tool_choice":{"type":"function","function":{"name":"weather"}}
    })).unwrap();
    OpenAIPreprocessor::normalize_kimi_k3_named_tool_choice(&mut req, Some("kimi_k3"));
    assert!(OpenAIPreprocessor::is_reasoning_disabled_by_request(
        Some("kimi_k3"),
        req.chat_template_args.as_ref()
    ));
    let prompt = processor.apply_template(&req).unwrap();
    assert!(!OpenAIPreprocessor::prompt_injected_reasoning_start(
        Some("kimi_k3"),
        prompt.as_ref().map(RenderedPrompt::as_str)
    ));
}

#[tokio::test]
async fn kimi_k3_named_tools_enable_structural_tags_without_global_opt_in() {
    let (_dir, processor) = preprocessor();
    let mut req: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"moonshotai/kimi-k3", "messages":[{"role":"user","content":"Weather?"}],
        "tools":[{"type":"function","function":{"name":"weather", "parameters":{"type":"object","properties":{}}}}],
        "tool_choice":{"type":"function","function":{"name":"weather"}}
    })).unwrap();
    assert_eq!(
        processor.runtime_config.structural_tag_mode,
        crate::local_model::runtime_config::StructuralTagMode::Off
    );
    OpenAIPreprocessor::normalize_kimi_k3_named_tool_choice(&mut req, Some("kimi_k3"));
    let (mut common, _, thinking) = processor.preprocess_request(&req, None).await.unwrap();
    assert!(!thinking);
    assert!(
        processor
            .apply_tool_choice_guided_decoding(&req, &mut common, thinking)
            .unwrap()
    );
    let guided = common.sampling_options.guided_decoding.unwrap();
    assert!(guided.structural_tag.is_some());
    assert!(guided.json.is_none());
}

#[test]
fn metadata_choice_limit_follows_existing_parser_routes() {
    use dynamo_protocols::types::ChatCompletionToolChoiceOption;

    let (_dir, mut processor) = preprocessor();
    let processor = Arc::get_mut(&mut processor).expect("sole test processor owner");
    let cases = [
        (
            Some("kimi_k3"),
            None,
            false,
            None,
            false,
            true,
            ToolProcessingRoute::LegacyJail(Some("kimi_k3".into())),
        ),
        (
            None,
            Some("kimi_k3"),
            false,
            None,
            false,
            true,
            ToolProcessingRoute::LegacyJail(Some("kimi_k3".into())),
        ),
        (
            Some("hermes"),
            None,
            false,
            None,
            false,
            false,
            ToolProcessingRoute::PassThrough,
        ),
        (
            Some("hermes"),
            None,
            true,
            Some("none"),
            false,
            false,
            ToolProcessingRoute::PassThrough,
        ),
        (
            Some("hermes"),
            None,
            true,
            Some("auto"),
            false,
            false,
            ToolProcessingRoute::LegacyJail(Some("hermes".into())),
        ),
        (
            Some("qwen3_coder"),
            None,
            true,
            Some("auto"),
            false,
            true,
            ToolProcessingRoute::ParserV2("qwen3_coder".into()),
        ),
        (
            Some("qwen3_coder"),
            None,
            true,
            Some("required"),
            false,
            true,
            ToolProcessingRoute::LegacyJail(Some("qwen3_coder".into())),
        ),
        (
            Some("qwen3_coder"),
            None,
            true,
            Some("auto"),
            true,
            true,
            ToolProcessingRoute::LegacyJail(Some("qwen3_coder".into())),
        ),
        (
            None,
            None,
            true,
            Some("required"),
            false,
            true,
            ToolProcessingRoute::LegacyJail(None),
        ),
    ];
    for (parser, reasoner, has_tools, choice, structural, v2, expected) in cases {
        processor.tool_call_parser = parser.map(str::to_owned);
        processor.runtime_config.reasoning_parser = reasoner.map(str::to_owned);
        let mut body =
            json!({"model":"test", "messages":[{"role":"user","content":"test"}], "n":2});
        if has_tools {
            body["tools"] = json!([{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]);
        }
        if let Some(choice) = choice {
            body["tool_choice"] = json!(choice);
        }
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        let route = processor
            .tool_processing_route(&request, structural, v2)
            .unwrap();
        assert_eq!(route, expected);
        for field in ["engine_data", "routed_experts", "stop_reason"] {
            request.nvext = Some(serde_json::from_value(json!({"extra_fields":[field]})).unwrap());
            request.inner.n = Some(2);
            assert_eq!(
                validate_legacy_jail_nvext_choice_count(&request, &route).is_err(),
                matches!(route, ToolProcessingRoute::LegacyJail(_)),
                "wrong metadata guard for {route:?}, {field}",
            );
            request.inner.n = Some(1);
            assert!(validate_legacy_jail_nvext_choice_count(&request, &route).is_ok());
        }
        request.inner.n = Some(2);
        request.nvext =
            Some(serde_json::from_value(json!({"extra_fields":["worker_id","timing"]})).unwrap());
        assert!(validate_legacy_jail_nvext_choice_count(&request, &route).is_ok());

        // A named tool remains a legacy route even when v2 is enabled.
        if has_tools && parser == Some("qwen3_coder") {
            request.inner.tool_choice = Some(
                serde_json::from_value::<ChatCompletionToolChoiceOption>(
                    json!({"type":"function","function":{"name":"weather"}}),
                )
                .unwrap(),
            );
            assert_eq!(
                processor
                    .tool_processing_route(&request, false, true)
                    .unwrap(),
                ToolProcessingRoute::LegacyJail(Some("qwen3_coder".into()))
            );
        }
    }
}

fn partial_reasoning_marker_chunk() -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut response: NvCreateChatCompletionStreamResponse = serde_json::from_value(json!({
        "id":"test", "model":"test", "created":0, "object":"chat.completion.chunk",
        "choices":[{"index":0,"delta":{"role":"assistant","content":"<thi"}}],
        "nvext":{"completion_token_ids":[42]},
    }))
    .unwrap();
    response.llm_metrics = Some(LLMMetricAnnotation {
        input_tokens: 1,
        output_tokens: 1,
        chunk_tokens: 1,
        ..Default::default()
    });
    Annotated {
        data: Some(response),
        id: None,
        event: Some("token_data".into()),
        comment: Some(vec!["original".into()]),
        error: None,
    }
}

#[tokio::test]
async fn reasoning_prefix_eof_does_not_duplicate_metadata() {
    let output: Vec<_> = OpenAIPreprocessor::strip_leading_reasoning_start_from_stream(
        stream::iter([partial_reasoning_marker_chunk()]),
        "<think>",
    )
    .collect()
    .await;
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].data.as_ref().unwrap().nvext.as_ref().unwrap()["completion_token_ids"],
        json!([42])
    );
    let recovered = output[1].data.as_ref().unwrap();
    assert_eq!(
        recovered.inner.choices[0].delta.content,
        Some(ChatCompletionMessageContent::Text("<thi".into()))
    );
    assert!(recovered.nvext.is_none());
    assert!(recovered.llm_metrics.is_none());
    assert!(recovered.inner.usage.is_none());
    assert!(output[1].event.is_none());
    assert!(output[1].comment.is_none());
}

#[tokio::test]
async fn reasoning_prefix_error_does_not_flush_buffered_content() {
    let output: Vec<_> = OpenAIPreprocessor::strip_leading_reasoning_start_from_stream(
        stream::iter([
            partial_reasoning_marker_chunk(),
            Annotated::from_error("upstream failed"),
        ]),
        "<think>",
    )
    .collect()
    .await;
    assert_eq!(output.len(), 2);
    assert!(output[1].is_error());
    assert!(
        output
            .iter()
            .filter_map(|item| item.data.as_ref())
            .flat_map(|data| &data.inner.choices)
            .all(|choice| choice.delta.content.is_none())
    );
}

#[tokio::test]
async fn reasoning_prefix_recovery_precedes_usage_for_every_choice_and_stops_on_error() {
    let (_dir, mut processor) = preprocessor();
    let processor = Arc::get_mut(&mut processor).unwrap();
    processor.tool_call_parser = None;
    processor.runtime_config.reasoning_parser = Some("nemotron_nano".into());
    let mut request = request("Hello");
    request.inner.n = Some(2);
    request.chat_template_args = Some(
        [("enable_thinking".to_string(), json!(false))]
            .into_iter()
            .collect(),
    );

    for fail in [false, true] {
        let mut second = partial_reasoning_marker_chunk();
        let data = second.data.as_mut().unwrap();
        data.inner.choices[0].index = 1;
        data.inner.choices[0].delta.content =
            Some(ChatCompletionMessageContent::Text("<th".into()));
        data.nvext = Some(json!({"completion_token_ids":[43]}));
        let usage = Annotated::from_data(
            serde_json::from_value::<NvCreateChatCompletionStreamResponse>(json!({
                "id":"test", "model":"test", "created":0,
                "object":"chat.completion.chunk", "choices":[],
                "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
            }))
            .unwrap(),
        );
        let mut input = vec![partial_reasoning_marker_chunk(), second, usage];
        if fail {
            input.push(Annotated::from_error("upstream failed"));
        }
        let output: Vec<_> = processor
            .postprocessor_parsing_stream(stream::iter(input), &request, false, false)
            .unwrap()
            .collect()
            .await;
        let ids: Vec<_> = output
            .iter()
            .filter_map(|item| item.data.as_ref()?.nvext.as_ref())
            .flat_map(|nvext| nvext["completion_token_ids"].as_array().unwrap())
            .cloned()
            .collect();
        assert_eq!(ids, vec![json!(42), json!(43)]);

        if fail {
            assert_eq!(output.len(), 3);
            assert!(output.last().unwrap().is_error());
            assert!(
                output
                    .iter()
                    .filter_map(|item| item.data.as_ref())
                    .all(|data| {
                        data.inner.usage.is_none()
                            && data
                                .inner
                                .choices
                                .iter()
                                .all(|choice| choice.delta.content.is_none())
                    })
            );
        } else {
            assert_eq!(output.len(), 4);
            let recovered = output[2].data.as_ref().unwrap();
            let content: Vec<_> = recovered
                .inner
                .choices
                .iter()
                .map(|choice| (choice.index, choice.delta.content.clone()))
                .collect();
            assert_eq!(
                content,
                vec![
                    (0, Some(ChatCompletionMessageContent::Text("<thi".into()))),
                    (1, Some(ChatCompletionMessageContent::Text("<th".into()))),
                ]
            );
            assert!(recovered.nvext.is_none());
            assert!(recovered.llm_metrics.is_none());
            assert!(recovered.inner.usage.is_none());
            assert!(output[2].event.is_none());
            assert!(output[2].comment.is_none());
            assert!(output[3].data.as_ref().unwrap().inner.usage.is_some());
        }
    }
}
