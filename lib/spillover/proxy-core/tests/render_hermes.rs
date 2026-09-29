// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip tests for `HermesRenderer`.
//!
//! The renderer must emit the raw Qwen3 format so Dynamo's frontend parsers (`qwen3` for
//! reasoning, `hermes` for tool calls) recover the provider's content, reasoning and tool
//! calls. Unlike GLM, the Qwen3 prompt template does not inject an opener when thinking is
//! enabled, so the renderer emits its own ` thinking` and `qwen3` starts outside reasoning.
//! Thinking off leaves it outside too, so content and tool calls carry no markers at all.

use dw_proxy_core::render::{ParserFamily, ReasoningStart, renderer_for};
use dynamo_parsers::{ReasoningParser, ReasoningParserType, detect_and_parse_tool_call};
use serde_json::{Value, json};

const THINK_START: &str = "\u{3c}think\u{3e}";
const THINK_END: &str = "\u{3c}/think\u{3e}";

fn render(deltas: &[Value], start: ReasoningStart) -> String {
    let mut renderer = renderer_for(ParserFamily::Hermes, start);
    let mut out = String::new();
    for delta in deltas {
        out.push_str(&renderer.push_delta(delta).expect("push_delta"));
    }
    out.push_str(&renderer.finish(Some("tool_calls")).expect("finish"));
    out
}

struct Parsed {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

async fn parse(text: &str, in_reasoning: bool) -> Parsed {
    let mut parser = ReasoningParserType::get_reasoning_parser_from_name("qwen3");
    parser.set_in_reasoning(in_reasoning);
    let split = parser.detect_and_parse_reasoning(text, &[]);
    let (calls, content) = detect_and_parse_tool_call(&split.normal_text, Some("hermes"), None)
        .await
        .expect("tool call parse");
    Parsed {
        reasoning: split.reasoning_text,
        content: content.unwrap_or_default(),
        calls: calls
            .into_iter()
            .map(|call| {
                let arguments = serde_json::from_str(&call.function.arguments)
                    .expect("tool call arguments are valid JSON");
                (call.function.name, arguments)
            })
            .collect(),
    }
}

fn tool_delta(index: u64, id: Option<&str>, name: Option<&str>, arguments: &str) -> Value {
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), json!(name));
    }
    function.insert("arguments".to_string(), json!(arguments));
    json!({
        "tool_calls": [{
            "index": index,
            "id": id,
            "type": "function",
            "function": Value::Object(function),
        }],
    })
}

fn get_weather_call() -> Value {
    tool_delta(
        0,
        Some("call_1"),
        Some("get_weather"),
        r#"{"location":"San Francisco, CA","unit":"celsius"}"#,
    )
}

#[tokio::test]
async fn thinking_on_reasoning_only() {
    let text = render(
        &[json!({"reasoning_content": "Plan the steps."})],
        ReasoningStart::Outside,
    );
    assert_eq!(text, format!("{THINK_START}Plan the steps.{THINK_END}"));

    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "Plan the steps.");
    assert_eq!(parsed.content, "");
    assert!(parsed.calls.is_empty());
}

#[tokio::test]
async fn thinking_on_reasoning_and_content() {
    let text = render(
        &[
            json!({"reasoning_content": "Plan the steps."}),
            json!({"content": "Here is the answer."}),
        ],
        ReasoningStart::Outside,
    );
    assert_eq!(
        text,
        format!("{THINK_START}Plan the steps.{THINK_END}Here is the answer.")
    );

    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "Plan the steps.");
    assert_eq!(parsed.content, "Here is the answer.");
    assert!(parsed.calls.is_empty());
}

#[tokio::test]
async fn thinking_on_content_only_has_no_markers() {
    let text = render(
        &[json!({"content": "Here is the answer."})],
        ReasoningStart::Outside,
    );
    assert_eq!(text, "Here is the answer.");

    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "");
    assert_eq!(parsed.content, "Here is the answer.");
}

#[tokio::test]
async fn thinking_on_tool_call_only_has_no_markers() {
    let text = render(&[get_weather_call()], ReasoningStart::Outside);
    assert!(!text.contains("think"));

    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "");
    assert_eq!(parsed.content, "");
    assert_eq!(
        parsed.calls,
        vec![(
            "get_weather".to_string(),
            json!({"location": "San Francisco, CA", "unit": "celsius"})
        )]
    );
}

#[tokio::test]
async fn thinking_on_two_tool_calls_nested_json_split_arguments() {
    let mut deltas = Vec::new();
    deltas.extend(split_tool_call(
        0,
        Some("call_1"),
        Some("get_weather"),
        r#"{"location":"Paris","days":3,"urgent":true,"options":{"units":["celsius","fahrenheit"]}}"#,
    ));
    deltas.extend(split_tool_call(
        1,
        Some("call_2"),
        Some("search"),
        r#"{"query":"rust parsers","options":{"limit":10,"tags":["a","b"]}}"#,
    ));

    let text = render(&deltas, ReasoningStart::Outside);
    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "");
    assert_eq!(parsed.content, "");
    assert_eq!(
        parsed.calls,
        vec![
            (
                "get_weather".to_string(),
                json!({
                    "location": "Paris",
                    "days": 3,
                    "urgent": true,
                    "options": {"units": ["celsius", "fahrenheit"]}
                })
            ),
            (
                "search".to_string(),
                json!({"query": "rust parsers", "options": {"limit": 10, "tags": ["a", "b"]}})
            ),
        ]
    );
}

#[tokio::test]
async fn thinking_off_content_only_has_no_markers() {
    let text = render(
        &[json!({"content": "Here is the answer."})],
        ReasoningStart::Outside,
    );
    assert_eq!(text, "Here is the answer.");

    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "");
    assert_eq!(parsed.content, "Here is the answer.");
}

#[tokio::test]
async fn thinking_off_tool_call_only_has_no_markers() {
    let text = render(&[get_weather_call()], ReasoningStart::Outside);
    assert!(!text.contains("think"));
    let parsed = parse(&text, false).await;
    assert_eq!(parsed.calls.len(), 1);
}

/// A prompt-injected state is not what Qwen3 needs, but the renderer must still be exact
/// if the frontend ever reports it: no opener, and a closer before non-reasoning output.
#[tokio::test]
async fn prompt_injected_state_closes_the_block() {
    let reasoning = render(
        &[json!({"reasoning_content": "Plan the steps."})],
        ReasoningStart::InsideReasoning,
    );
    assert_eq!(reasoning, format!("Plan the steps.{THINK_END}"));
    let text = render(
        &[json!({"content": "Here is the answer."})],
        ReasoningStart::InsideReasoning,
    );
    assert_eq!(text, format!("{THINK_END}Here is the answer."));

    let parsed = parse(&text, true).await;
    assert_eq!(parsed.reasoning, "");
    assert_eq!(parsed.content, "Here is the answer.");
}

fn split_tool_call(
    index: u64,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> Vec<Value> {
    arguments
        .as_bytes()
        .chunks(2)
        .enumerate()
        .map(|(position, chunk)| {
            let arguments = std::str::from_utf8(chunk).unwrap();
            if position == 0 {
                tool_delta(index, id, name, arguments)
            } else {
                tool_delta(index, None, None, arguments)
            }
        })
        .collect()
}
