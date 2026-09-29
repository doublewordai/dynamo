// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip the Kimi K3 renderer through the fork's real K3 parser.
//!
//! Each test builds the OpenAI streaming deltas a provider would send, renders
//! them to raw model text with [`KimiK3Renderer`], then runs that text through
//! the parsers Dynamo's frontend actually runs for Kimi K3:
//! `dynamo_parsers`' v1 `ReasoningParserType::KimiK3` and
//! `try_tool_call_parse_kimi_k3`. K3 is not in the v2 `V2_FAMILIES` /
//! `UNIFIED_FAMILIES` sets, so the frontend routes it through the legacy jail
//! (`lib/llm/src/preprocessor.rs`, `ToolProcessingRoute::LegacyJail`), not the
//! v2 unified parser.
//!
//! The reasoning parser starts in the state the frontend uses: `in_reasoning`
//! when the prompt injected the `<|open|>think<|sep|>` opener (thinking on) and
//! outside it when the prompt opened the response channel instead (thinking off).
//!
//! Argument fragments are streamed in 1-3 character pieces to exercise the
//! renderer's per-index buffering.

use dw_proxy_core::render::OutputRenderer;
use dw_proxy_core::render::ReasoningStart;
use dw_proxy_core::render::kimi_k3::KimiK3Renderer;
use dynamo_parsers::reasoning::{ReasoningParser, ReasoningParserType};
use dynamo_parsers::tool_calling::{KimiK3ParserConfig, try_tool_call_parse_kimi_k3};
use serde_json::{Value, json};

/// Split into 1-3 character pieces, the way a provider streams text.
fn chunk_1_3(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut target = 1usize;
    for ch in text.chars() {
        current.push(ch);
        if current.chars().count() >= target {
            pieces.push(std::mem::take(&mut current));
            target = target % 3 + 1;
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

fn render(deltas: &[Value], start: ReasoningStart) -> String {
    let mut renderer = KimiK3Renderer::new(start);
    let mut out = String::new();
    for delta in deltas {
        out.push_str(&renderer.push_delta(delta).unwrap());
    }
    out.push_str(&renderer.finish(Some("stop")).unwrap());
    out
}

#[derive(Debug, PartialEq)]
struct Recovered {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

/// Run the rendered text through the streaming reasoning splitter, then the
/// aggregate K3 tool parser, exactly as the legacy jail composes them.
fn parse(rendered: &str, thinking_on: bool) -> Recovered {
    let mut parser = ReasoningParserType::KimiK3.get_reasoning_parser();
    parser.set_in_reasoning(thinking_on);

    let mut reasoning = String::new();
    let mut normal = String::new();
    for piece in chunk_1_3(rendered) {
        let parsed = parser.parse_reasoning_streaming_incremental(&piece, &[]);
        reasoning.push_str(&parsed.reasoning_text);
        normal.push_str(&parsed.normal_text);
    }
    let finalised = parser.finish_reasoning_stream();
    reasoning.push_str(&finalised.reasoning_text);
    normal.push_str(&finalised.normal_text);

    let (calls, content) =
        try_tool_call_parse_kimi_k3(&normal, &KimiK3ParserConfig::default(), None).unwrap();
    let calls = calls
        .into_iter()
        .map(|call| {
            let arguments = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
            (call.function.name, arguments)
        })
        .collect();

    Recovered {
        reasoning,
        content: content.unwrap_or_default(),
        calls,
    }
}

fn reasoning_deltas(text: &str) -> Vec<Value> {
    chunk_1_3(text)
        .into_iter()
        .map(|piece| json!({"reasoning_content": piece}))
        .collect()
}

fn content_deltas(text: &str) -> Vec<Value> {
    chunk_1_3(text)
        .into_iter()
        .map(|piece| json!({"content": piece}))
        .collect()
}

/// Stream one tool call: the name first, then arguments in 1-3 character pieces.
fn tool_call_deltas(index: usize, name: &str, arguments: &str) -> Vec<Value> {
    let mut deltas = vec![json!({
        "tool_calls": [{"index": index, "type": "function",
            "function": {"name": name, "arguments": ""}}]
    })];
    for piece in chunk_1_3(arguments) {
        deltas.push(json!({
            "tool_calls": [{"index": index, "type": "function",
                "function": {"arguments": piece}}]
        }));
    }
    deltas
}

// ---------------------------------------------------------------------------
// Thinking on: the prompt opened reasoning, so the parser starts inside it.
// ---------------------------------------------------------------------------

#[test]
fn thinking_on_reasoning_only() {
    let deltas = reasoning_deltas("Let me think about it.");
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: "Let me think about it.".into(),
            content: String::new(),
            calls: vec![],
        }
    );
}

#[test]
fn thinking_on_reasoning_then_content() {
    let mut deltas = reasoning_deltas("Check the tools.\n");
    deltas.extend(content_deltas("The answer is 18."));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: "Check the tools.\n".into(),
            content: "The answer is 18.".into(),
            calls: vec![],
        }
    );
}

/// Content with no reasoning must close the prompt-injected block first, or the
/// parser would swallow the whole answer as reasoning.
#[test]
fn thinking_on_content_only() {
    let deltas = content_deltas("The answer is 18.");
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: String::new(),
            content: "The answer is 18.".into(),
            calls: vec![],
        }
    );
}

#[test]
fn thinking_on_tool_call_only() {
    let deltas = tool_call_deltas(0, "ping", "{}");
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: String::new(),
            content: String::new(),
            calls: vec![("ping".into(), json!({}))],
        }
    );
}

#[test]
fn thinking_on_content_then_one_tool_call() {
    let mut deltas = content_deltas("Looking that up.");
    deltas.extend(tool_call_deltas(
        0,
        "weather",
        r#"{"city":"Paris","days":3}"#,
    ));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: String::new(),
            content: "Looking that up.".into(),
            calls: vec![("weather".into(), json!({"city": "Paris", "days": 3}),)],
        }
    );
}

#[test]
fn thinking_on_reasoning_then_tool_call_without_content() {
    let mut deltas = reasoning_deltas("I should call the tool.");
    deltas.extend(tool_call_deltas(0, "ping", "{}"));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: "I should call the tool.".into(),
            content: String::new(),
            calls: vec![("ping".into(), json!({}))],
        }
    );
}

#[test]
fn thinking_on_two_tool_calls_with_nested_json() {
    let mut deltas = content_deltas("Two calls.");
    deltas.extend(tool_call_deltas(
        0,
        "search",
        r#"{"query":"rust","options":{"limit":5,"tags":["a","b"]}}"#,
    ));
    deltas.extend(tool_call_deltas(
        1,
        "weather",
        r#"{"city":"東京","when":null,"metrics":[1,2.5,true]}"#,
    ));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true),
        Recovered {
            reasoning: String::new(),
            content: "Two calls.".into(),
            calls: vec![
                (
                    "search".into(),
                    json!({
                        "query": "rust",
                        "options": {"limit": 5, "tags": ["a", "b"]},
                    }),
                ),
                (
                    "weather".into(),
                    json!({
                        "city": "東京",
                        "when": null,
                        "metrics": [1, 2.5, true],
                    }),
                ),
            ],
        }
    );
}

// ---------------------------------------------------------------------------
// Thinking off: the prompt opened the response channel, so the parser starts
// outside reasoning and the completion must not re-open it.
// ---------------------------------------------------------------------------

#[test]
fn thinking_off_content_only() {
    let deltas = content_deltas("The answer is 18.");
    let rendered = render(&deltas, ReasoningStart::Outside);
    // The prompt opened the response channel; the completion closes it.
    assert_eq!(rendered, "The answer is 18.<|close|>response<|sep|>");
    assert_eq!(
        parse(&rendered, false),
        Recovered {
            reasoning: String::new(),
            content: "The answer is 18.".into(),
            calls: vec![],
        }
    );
}

#[test]
fn thinking_off_tool_call_only() {
    let deltas = tool_call_deltas(0, "ping", "{}");
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert!(!rendered.contains("<|open|>think"));
    assert_eq!(
        parse(&rendered, false),
        Recovered {
            reasoning: String::new(),
            content: String::new(),
            calls: vec![("ping".into(), json!({}))],
        }
    );
}

#[test]
fn thinking_off_content_then_tool_call() {
    let mut deltas = content_deltas("Looking that up.");
    deltas.extend(tool_call_deltas(
        0,
        "weather",
        r#"{"city":"Paris","days":3}"#,
    ));
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert_eq!(
        parse(&rendered, false),
        Recovered {
            reasoning: String::new(),
            content: "Looking that up.".into(),
            calls: vec![("weather".into(), json!({"city": "Paris", "days": 3}),)],
        }
    );
}

// ---------------------------------------------------------------------------
// The byte-for-byte markup for a prompt-injected thinking-on turn. The prompt
// already emitted `<|open|>think<|sep|>`, so the completion starts at the
// reasoning text and closes the block itself; the concatenation matches the
// frontend's own composition test
// (`test_kimi_k3_reasoning_output_composes_with_tool_parser`).
// ---------------------------------------------------------------------------

#[test]
fn rendered_markup_is_native_xtml() {
    let mut deltas = reasoning_deltas("use the calculator");
    deltas.extend(content_deltas("I will calculate."));
    deltas.extend(tool_call_deltas(0, "calc", r#"{"x":42}"#));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        rendered,
        concat!(
            "use the calculator<|close|>think<|sep|>",
            "<|open|>response<|sep|>I will calculate.<|close|>response<|sep|>",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"calc\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"x\" type=\"number\"<|sep|>42",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|>",
        )
    );
}

/// Providers commonly put `content: ""` on tool-call deltas. That is not the start of a
/// content section, so it must not flush the in-flight call.
#[test]
fn empty_content_between_tool_fragments_does_not_flush() {
    let deltas = vec![
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":"}}]}),
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "\"Paris\"}"}}]}),
    ];
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true).calls,
        vec![("get_weather".into(), json!({"city": "Paris"}))]
    );
}

/// Fragments of parallel calls may interleave; all calls are buffered and rendered in
/// index order at the boundary.
#[test]
fn interleaved_tool_call_indices_round_trip() {
    let deltas = vec![
        json!({"tool_calls": [{"index": 0, "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":"}}]}),
        json!({"tool_calls": [{"index": 1, "type": "function", "function": {"name": "search", "arguments": "{\"query\":"}}]}),
        json!({"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "\"Paris\"}"}}]}),
        json!({"tool_calls": [{"index": 1, "type": "function", "function": {"arguments": "\"rust\"}"}}]}),
    ];
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, true).calls,
        vec![
            ("get_weather".into(), json!({"city": "Paris"})),
            ("search".into(), json!({"query": "rust"})),
        ]
    );
}

/// The XTML grammar has no escaping for argument bodies, so a value carrying a
/// structural token must be rejected rather than terminating the element early.
#[test]
fn reserved_marker_in_argument_value_is_rejected() {
    let mut renderer = KimiK3Renderer::new(ReasoningStart::InsideReasoning);
    let delta = json!({
        "tool_calls": [{"index": 0, "type": "function", "function": {
            "name": "write", "arguments": "{\"text\":\"a<|close|>argument<|sep|>b\"}"
        }}]
    });
    renderer.push_delta(&delta).expect("buffers the call");
    assert!(renderer.finish(Some("tool_calls")).is_err());
}
