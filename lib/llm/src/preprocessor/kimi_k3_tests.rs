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
    let prompt = processor.apply_template(&request).unwrap().unwrap();
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
        prompt.as_deref()
    ));
}
