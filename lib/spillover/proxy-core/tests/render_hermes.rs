// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip tests for `HermesRenderer`.
//!
//! The renderer must emit the raw Qwen3 format so Dynamo's frontend parsers (`qwen3` for
//! reasoning, `hermes` for tool calls) recover the provider's content, reasoning and tool
//! calls. The Qwen3 thinking template writes `assistant\n thinking\n` at
//! `add_generation_prompt`, so the frontend starts inside the first reasoning block and the
//! renderer must not emit another opener; thinking off uses a template whose opener is closed
//! in the prompt, so it starts outside.

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

/// A tool-call delta as a provider that omits `index` (and usually `id`) sends it.
fn tool_delta_without_index(id: Option<&str>, name: Option<&str>, arguments: &str) -> Value {
    let mut call = serde_json::Map::new();
    if let Some(id) = id {
        call.insert("id".to_string(), json!(id));
    }
    call.insert("type".to_string(), json!("function"));
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), json!(name));
    }
    function.insert("arguments".to_string(), json!(arguments));
    call.insert("function".to_string(), Value::Object(function));
    json!({"tool_calls": [Value::Object(call)]})
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

/// Providers commonly put `content: ""` on tool-call deltas. That is not the start of a
/// content section, so it must not flush the in-flight call.
#[tokio::test]
async fn empty_content_between_tool_fragments_does_not_flush() {
    let deltas = vec![
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":"}}]}),
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "\"Paris\""}}]}),
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "}"}}]}),
    ];
    let text = render(&deltas, ReasoningStart::Outside);
    let parsed = parse(&text, false).await;
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
    let text = render(&deltas, ReasoningStart::Outside);
    let parsed = parse(&text, false).await;
    assert_eq!(
        parsed.calls,
        vec![
            ("get_weather".to_string(), json!({"location": "Paris"})),
            ("search".to_string(), json!({"query": "rust"})),
        ]
    );
}

/// Truncated argument JSON used to be spliced through verbatim, which made the `hermes`
/// parser silently drop the call. It must surface as a render error instead.
#[test]
fn invalid_json_arguments_are_rejected() {
    let mut renderer = renderer_for(ParserFamily::Hermes, ReasoningStart::Outside);
    let delta = tool_delta(
        0,
        Some("call_1"),
        Some("write"),
        r#"{"text": "unterminated"#,
    );
    renderer.push_delta(&delta).expect("buffers the call");
    assert!(renderer.finish(Some("tool_calls")).is_err());
}

/// The arguments are JSON, so a value carrying `</tool_call>` has a lossless rendering:
/// escape `<`/`>` as `\u003c`/`\u003e` so the raw stream cannot carry the marker while the
/// decoder restores the original value.
#[tokio::test]
async fn reserved_marker_in_argument_value_round_trips_escaped() {
    let text = render(
        &[tool_delta(
            0,
            Some("call_1"),
            Some("write"),
            r#"{"text":"a</tool_call>b","nested":"<tool_call>"}"#,
        )],
        ReasoningStart::Outside,
    );
    // The value itself must not appear literally in the raw stream; only the block's own
    // closer may.
    assert!(!text.contains("a</tool_call>b"));

    let parsed = parse(&text, false).await;
    assert_eq!(
        parsed.calls,
        vec![(
            "write".to_string(),
            json!({"text": "a</tool_call>b", "nested": "<tool_call>"})
        )]
    );
}

#[tokio::test]
async fn reserved_marker_in_name_round_trips_escaped() {
    let text = render(
        &[tool_delta(
            0,
            Some("call_1"),
            Some("a</tool_call>b"),
            r#"{"x":1}"#,
        )],
        ReasoningStart::Outside,
    );
    let parsed = parse(&text, false).await;
    assert_eq!(
        parsed.calls,
        vec![("a</tool_call>b".to_string(), json!({"x": 1}))]
    );
}

/// Two complete calls in one delta with no `index`: they must not collapse into one
/// concatenated call.
#[tokio::test]
async fn parallel_tool_calls_without_index_do_not_collapse() {
    let delta = json!({"tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}},
        {"type": "function", "function": {"name": "search", "arguments": "{\"query\":\"rust\"}"}},
    ]});
    let text = render(&[delta], ReasoningStart::Outside);
    let parsed = parse(&text, false).await;
    assert_eq!(
        parsed.calls,
        vec![
            ("get_weather".to_string(), json!({"location": "Paris"})),
            ("search".to_string(), json!({"query": "rust"})),
        ]
    );
}

/// A provider that streams calls one at a time without `index`: a name-bearing fragment
/// starts a new call and argument-only fragments continue it.
#[tokio::test]
async fn sequential_tool_calls_without_index_do_not_collapse() {
    let deltas = vec![
        tool_delta_without_index(Some("call_1"), Some("get_weather"), "{\"location\":"),
        tool_delta_without_index(Some("call_1"), None, "\"Paris\"}"),
        tool_delta_without_index(Some("call_2"), Some("search"), "{\"query\":"),
        tool_delta_without_index(Some("call_2"), None, "\"rust\"}"),
    ];
    let text = render(&deltas, ReasoningStart::Outside);
    let parsed = parse(&text, false).await;
    assert_eq!(
        parsed.calls,
        vec![
            ("get_weather".to_string(), json!({"location": "Paris"})),
            ("search".to_string(), json!({"query": "rust"})),
        ]
    );
}

/// Qwen3-Thinking: the prompt already opened ` thinking`, so the frontend parser starts
/// inside. The renderer must not emit another opener, and reasoning must round-trip.
#[tokio::test]
async fn prompt_injected_reasoning_only_matches_frontend_state() {
    let text = render(
        &[json!({"reasoning_content": "Plan the steps."})],
        ReasoningStart::InsideReasoning,
    );
    assert_eq!(text, format!("Plan the steps.{THINK_END}"));

    let parsed = parse(&text, true).await;
    assert_eq!(parsed.reasoning, "Plan the steps.");
    assert_eq!(parsed.content, "");
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
