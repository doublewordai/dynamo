// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip tests for `GlmRenderer`.
//!
//! The renderer must emit the raw GLM format so Dynamo's frontend parsers (`glm45` for
//! reasoning, `glm47` for tool calls) recover the provider's content, reasoning and tool
//! calls. `glm45` is told the frontend's actual starting state: thinking on means the
//! template already injected the opening ` thinking` (`set_in_reasoning(true)`), thinking
//! off means it did not, so the renderer opens and closes every block itself.

use dw_proxy_core::render::{ParserFamily, ReasoningStart, renderer_for};
use dynamo_parsers::{
    ReasoningParser, ReasoningParserType, ToolDefinition, detect_and_parse_tool_call,
};
use serde_json::{Value, json};

const THINK_END: &str = "\u{3c}/think\u{3e}";

fn render(deltas: &[Value], start: ReasoningStart) -> String {
    let mut renderer = renderer_for(ParserFamily::Glm47, start);
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
    let mut parser = ReasoningParserType::get_reasoning_parser_from_name("glm45");
    parser.set_in_reasoning(in_reasoning);
    let split = parser.detect_and_parse_reasoning(text, &[]);
    let (calls, content) =
        detect_and_parse_tool_call(&split.normal_text, Some("glm47"), Some(&tools()))
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

fn tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                    "urgent": {"type": "boolean"},
                    "options": {"type": "object"}
                }
            })),
            strict: None,
        },
        ToolDefinition {
            name: "search".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "options": {"type": "object"}
                }
            })),
            strict: None,
        },
    ]
}

/// One streamed tool-call delta. `name`/`id` are usually only present on the first fragment.
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
        ReasoningStart::InsideReasoning,
    );
    // The prompt opened the first block, so the renderer must not emit another opener.
    assert_eq!(text, format!("Plan the steps.{THINK_END}"));

    let parsed = parse(&text, true).await;
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
        ReasoningStart::InsideReasoning,
    );
    assert_eq!(
        text,
        format!("Plan the steps.{THINK_END}Here is the answer.")
    );

    let parsed = parse(&text, true).await;
    assert_eq!(parsed.reasoning, "Plan the steps.");
    assert_eq!(parsed.content, "Here is the answer.");
    assert!(parsed.calls.is_empty());
}

#[tokio::test]
async fn thinking_on_content_only_closes_the_injected_block() {
    // Without the closer the parser would swallow the content as reasoning.
    let text = render(
        &[json!({"content": "Here is the answer."})],
        ReasoningStart::InsideReasoning,
    );
    assert_eq!(text, format!("{THINK_END}Here is the answer."));

    let parsed = parse(&text, true).await;
    assert_eq!(parsed.reasoning, "");
    assert_eq!(parsed.content, "Here is the answer.");
}

#[tokio::test]
async fn thinking_on_tool_call_only() {
    let text = render(&[get_weather_call()], ReasoningStart::InsideReasoning);
    assert_eq!(
        text,
        format!(
            "{THINK_END}<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco, CA</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>"
        )
    );

    let parsed = parse(&text, true).await;
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
    // Stream arguments one to three characters at a time, like a real provider.
    let mut deltas = vec![json!({"content": "working"})];
    deltas.extend(split_tool_call(
        0,
        Some("call_1"),
        Some("get_weather"),
        r#"{"location":"Paris","days":3,"urgent":true,"options":{"units":["celsius","fahrenheit"],"nested":{"deep":[1,2,3]}}}"#,
    ));
    deltas.extend(split_tool_call(
        1,
        Some("call_2"),
        Some("search"),
        r#"{"query":"rust parsers","options":{"limit":10,"tags":["a","b"]}}"#,
    ));

    let text = render(&deltas, ReasoningStart::InsideReasoning);
    let parsed = parse(&text, true).await;
    assert_eq!(parsed.content, "working");
    assert_eq!(parsed.calls.len(), 2);
    assert_eq!(
        parsed.calls[0],
        (
            "get_weather".to_string(),
            json!({
                "location": "Paris",
                "days": 3,
                "urgent": true,
                "options": {
                    "units": ["celsius", "fahrenheit"],
                    "nested": {"deep": [1, 2, 3]}
                }
            })
        )
    );
    assert_eq!(
        parsed.calls[1],
        (
            "search".to_string(),
            json!({"query": "rust parsers", "options": {"limit": 10, "tags": ["a", "b"]}})
        )
    );
}

#[tokio::test]
async fn thinking_off_reasoning_and_content() {
    let text = render(
        &[
            json!({"reasoning_content": "Plan the steps."}),
            json!({"content": "Here is the answer."}),
        ],
        ReasoningStart::Outside,
    );
    assert_eq!(
        text,
        format!("\u{3c}think\u{3e}Plan the steps.{THINK_END}Here is the answer.")
    );

    let parsed = parse(&text, false).await;
    assert_eq!(parsed.reasoning, "Plan the steps.");
    assert_eq!(parsed.content, "Here is the answer.");
    assert!(parsed.calls.is_empty());
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

/// Providers commonly put `content: ""` on tool-call deltas. That is not the start of a
/// content section, so it must not flush the in-flight call.
#[tokio::test]
async fn empty_content_between_tool_fragments_does_not_flush() {
    let deltas = vec![
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":"}}]}),
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "\"Paris\""}}]}),
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "}"}}]}),
    ];
    let text = render(&deltas, ReasoningStart::InsideReasoning);
    let parsed = parse(&text, true).await;
    assert_eq!(
        parsed.calls,
        vec![("get_weather".to_string(), json!({"location": "Paris"}))]
    );
}

/// Fragments of parallel calls may interleave; a later index does not prove the earlier
/// call is complete, so nothing may be flushed until a real boundary.
#[tokio::test]
async fn interleaved_tool_call_indices_round_trip() {
    let deltas = vec![
        tool_delta(0, Some("call_1"), Some("get_weather"), "{\"location\":"),
        tool_delta(1, Some("call_2"), Some("search"), "{\"query\":"),
        tool_delta(0, None, None, "\"Paris\"}"),
        tool_delta(1, None, None, "\"rust\"}"),
    ];
    let text = render(&deltas, ReasoningStart::InsideReasoning);
    let parsed = parse(&text, true).await;
    assert_eq!(
        parsed.calls,
        vec![
            ("get_weather".to_string(), json!({"location": "Paris"})),
            ("search".to_string(), json!({"query": "rust"})),
        ]
    );
}

/// The `glm47` grammar has no escaping, so a value carrying a structural marker would be
/// truncated or split. The renderer must fail loudly instead of emitting corrupt markup.
#[test]
fn reserved_marker_in_argument_value_is_rejected() {
    let mut renderer = renderer_for(ParserFamily::Glm47, ReasoningStart::Outside);
    let delta = tool_delta(
        0,
        Some("call_1"),
        Some("write"),
        r#"{"text":"a</arg_value>b"}"#,
    );
    renderer.push_delta(&delta).expect("buffers the call");
    assert!(renderer.finish(Some("tool_calls")).is_err());
}

#[test]
fn reserved_marker_in_argument_key_is_rejected() {
    let mut renderer = renderer_for(ParserFamily::Glm47, ReasoningStart::Outside);
    let delta = tool_delta(0, Some("call_1"), Some("write"), r#"{"</arg_key>":"b"}"#);
    renderer.push_delta(&delta).expect("buffers the call");
    assert!(renderer.finish(Some("tool_calls")).is_err());
}

/// One tool call with its arguments streamed in small pieces.
fn split_tool_call(
    index: u64,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> Vec<Value> {
    arguments
        .as_bytes()
        .chunks(3)
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
