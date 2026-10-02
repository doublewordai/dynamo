// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip the DeepSeek V4.1 renderer through the v2 unified parser.
//!
//! Each test builds the OpenAI streaming deltas a provider would send, renders
//! them to raw model text, then parses that text with the same
//! `dynamo-parsers-v2` unified `deepseek_v41` parser Dynamo's frontend runs, and
//! asserts the recovered reasoning, content and tool calls. The parser is started
//! in the state the frontend uses: `Reasoning` when the prompt injected the
//! ` thinking` opener (thinking on), `Response` when it did not (thinking off).

use dw_proxy_core::render::OutputRenderer;
use dw_proxy_core::render::ReasoningStart;
use dw_proxy_core::render::deepseek_v41::DeepseekV41Renderer;
use dynamo_parsers_v2::{
    UnifiedEvent, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    create_unified_parser_for_family,
};
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
    let mut renderer = DeepseekV41Renderer::new(start);
    let mut out = String::new();
    for delta in deltas {
        out.push_str(&renderer.push_delta(delta).unwrap());
    }
    out.push_str(&renderer.finish(Some("stop")).unwrap());
    out
}

fn parse(input: &str, state: UnifiedParserStartingState) -> Vec<UnifiedEvent> {
    let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..Default::default()
        })
        .unwrap();
    let mut output = UnifiedParserOutput::default();
    for piece in chunk_1_3(input) {
        parser.parse_into(&piece, &mut output).unwrap();
    }
    output.append(&mut parser.finish().unwrap());
    output.assembled()
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
// Thinking on: the prompt opened reasoning, so the parser starts at `Reasoning`.
// ---------------------------------------------------------------------------

#[test]
fn thinking_on_reasoning_only() {
    let deltas = reasoning_deltas("Let me think about it.");
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![UnifiedEvent::Reasoning {
            text: "Let me think about it.".into()
        }]
    );
}

#[test]
fn thinking_on_reasoning_then_content() {
    let mut deltas = reasoning_deltas("Check the tools.\n");
    deltas.extend(content_deltas("The answer is 18."));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Reasoning {
                text: "Check the tools.\n".into()
            },
            UnifiedEvent::Text {
                text: "The answer is 18.".into()
            },
        ]
    );
}

/// Content with no reasoning must close the prompt-injected block first, or the
/// parser would swallow the whole answer as reasoning.
#[test]
fn thinking_on_content_only() {
    let deltas = content_deltas("The answer is 18.");
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert!(rendered.starts_with("\u{3c}/think\u{3e}"));
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![UnifiedEvent::Text {
            text: "The answer is 18.".into()
        }]
    );
}

#[test]
fn thinking_on_tool_call_only() {
    let deltas = tool_call_deltas(0, "ping", "{}");
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert!(rendered.starts_with("\u{3c}/think\u{3e}"));
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![UnifiedEvent::ToolCall {
            name: "ping".into(),
            arguments: json!({}),
        }]
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
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Text {
                text: "Looking that up.".into()
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: json!({"city": "Paris", "days": 3}),
            },
        ]
    );
}

#[test]
fn thinking_on_reasoning_then_tool_call_without_content() {
    let mut deltas = reasoning_deltas("I should call the tool.");
    deltas.extend(tool_call_deltas(0, "ping", "{}"));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Reasoning {
                text: "I should call the tool.".into()
            },
            UnifiedEvent::ToolCall {
                name: "ping".into(),
                arguments: json!({}),
            },
        ]
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
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Text {
                text: "Two calls.".into()
            },
            UnifiedEvent::ToolCall {
                name: "search".into(),
                arguments: json!({
                    "query": "rust",
                    "options": {"limit": 5, "tags": ["a", "b"]},
                }),
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: json!({
                    "city": "東京",
                    "when": null,
                    "metrics": [1, 2.5, true],
                }),
            },
        ]
    );
}

// ---------------------------------------------------------------------------
// Thinking off: the prompt opened the response channel, so the parser starts at
// `Response` and reasoning markers are disabled. Content is verbatim.
// ---------------------------------------------------------------------------

#[test]
fn thinking_off_content_only() {
    let deltas = content_deltas("The answer is 18.");
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert_eq!(rendered, "The answer is 18.");
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Response),
        vec![UnifiedEvent::Text {
            text: "The answer is 18.".into()
        }]
    );
}

#[test]
fn thinking_off_tool_call_only() {
    let deltas = tool_call_deltas(0, "ping", "{}");
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert!(!rendered.contains("\u{3c}think\u{3e}"));
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Response),
        vec![UnifiedEvent::ToolCall {
            name: "ping".into(),
            arguments: json!({}),
        }]
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
        parse(&rendered, UnifiedParserStartingState::Response),
        vec![
            UnifiedEvent::Text {
                text: "Looking that up.".into()
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: json!({"city": "Paris", "days": 3}),
            },
        ]
    );
}

/// A provider that ignores thinking-off and streams `reasoning_content` anyway must have
/// the chain of thought dropped: the frontend's unified parser starts in `Response`,
/// where its reasoning markers are literal text, so rendering the reasoning would leak
/// it to the client as content. The visible answer keeps flowing.
#[test]
fn thinking_off_reasoning_is_dropped() {
    let mut deltas = reasoning_deltas("hidden chain of thought.");
    deltas.extend(content_deltas("The answer is 18."));
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert_eq!(rendered, "The answer is 18.");
    assert!(!rendered.contains("hidden"));
    assert!(!rendered.contains("think"));
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Response),
        vec![UnifiedEvent::Text {
            text: "The answer is 18.".into()
        }]
    );
}

/// Reasoning-only output with thinking off renders nothing at all, not a stray marker.
#[test]
fn thinking_off_reasoning_only_is_dropped() {
    let deltas = reasoning_deltas("hidden chain of thought.");
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert!(rendered.is_empty());
    assert!(parse(&rendered, UnifiedParserStartingState::Response).is_empty());
}

/// `reasoning: null` alongside a real `reasoning_content` must not hide the text.
#[test]
fn null_reasoning_does_not_hide_reasoning_content() {
    let deltas = vec![
        json!({"reasoning": null, "reasoning_content": "Let me think."}),
        json!({"content": "The answer is 18."}),
    ];
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Reasoning {
                text: "Let me think.".into()
            },
            UnifiedEvent::Text {
                text: "The answer is 18.".into()
            },
        ]
    );
}

// ---------------------------------------------------------------------------
// Regression cases: empty content, interleaved indices, reserved markers.
// ---------------------------------------------------------------------------

/// `reasoning: ""` alongside a real `reasoning_content` must not hide the text.
#[test]
fn empty_reasoning_does_not_hide_reasoning_content() {
    let deltas = vec![
        json!({"reasoning": "", "reasoning_content": "Let me think."}),
        json!({"content": "The answer is 18."}),
    ];
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Reasoning {
                text: "Let me think.".into()
            },
            UnifiedEvent::Text {
                text: "The answer is 18.".into()
            },
        ]
    );
}

/// A reasoning delta after content started cannot re-enter the parser's reasoning state;
/// rendering it would leak the chain of thought to the client as literal text.
#[test]
fn late_reasoning_after_content_is_dropped() {
    let mut deltas = content_deltas("The answer is 18.");
    deltas.extend(reasoning_deltas("actually, wait."));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert!(
        !rendered.contains("actually"),
        "reasoning leaked: {rendered}"
    );
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![UnifiedEvent::Text {
            text: "The answer is 18.".into()
        }]
    );
}

/// A tool call buffered before later content must be rendered first, or the client sees
/// the answer before the call that produced it.
#[test]
fn tool_call_then_content_keeps_stream_order() {
    let mut deltas = tool_call_deltas(0, "weather", r#"{"city":"Paris"}"#);
    deltas.extend(content_deltas("It is sunny."));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: json!({"city": "Paris"}),
            },
            UnifiedEvent::Text {
                text: "It is sunny.".into()
            },
        ]
    );
}

/// An unindexed call followed by an indexed one must not merge: the proxy's fallback key
/// lives in a separate space from the provider's own `index`.
#[test]
fn unindexed_then_indexed_calls_do_not_merge() {
    let deltas = vec![
        json!({"tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{\"query\":\"rust\"}"}}]}),
        json!({"tool_calls": [{"index": 0, "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
    ];
    let rendered = render(&deltas, ReasoningStart::Outside);
    let parsed = parse(&rendered, UnifiedParserStartingState::Response);
    assert_eq!(parsed.len(), 2, "calls merged: {rendered}");
    assert!(parsed.contains(&UnifiedEvent::ToolCall {
        name: "search".into(),
        arguments: json!({"query": "rust"}),
    }));
    assert!(parsed.contains(&UnifiedEvent::ToolCall {
        name: "weather".into(),
        arguments: json!({"city": "Paris"}),
    }));
}

/// Providers commonly put `content: ""` on tool-call deltas. That is not the start of a
/// content section, so it must not close reasoning or flush the in-flight call.
#[test]
fn empty_content_between_tool_fragments_does_not_flush() {
    let deltas = vec![
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"name": "weather", "arguments": "{\"city\":"}}]}),
        json!({"content": "", "tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "\"Paris\"}"}}]}),
    ];
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![UnifiedEvent::ToolCall {
            name: "weather".into(),
            arguments: json!({"city": "Paris"}),
        }]
    );
}

/// Fragments of parallel calls may interleave; a later index does not prove the earlier
/// call is complete, so all calls are rendered together at `finish`.
#[test]
fn interleaved_tool_call_indices_round_trip() {
    let mut deltas = content_deltas("Two calls.");
    deltas.push(json!({"tool_calls": [{"index": 0, "type": "function", "function": {"name": "search", "arguments": "{\"query\":"}}]}));
    deltas.push(json!({"tool_calls": [{"index": 1, "type": "function", "function": {"name": "weather", "arguments": "{\"city\":"}}]}));
    deltas.push(json!({"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "\"rust\"}"}}]}));
    deltas.push(json!({"tool_calls": [{"index": 1, "type": "function", "function": {"arguments": "\"Paris\"}"}}]}));
    let rendered = render(&deltas, ReasoningStart::InsideReasoning);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Reasoning),
        vec![
            UnifiedEvent::Text {
                text: "Two calls.".into()
            },
            UnifiedEvent::ToolCall {
                name: "search".into(),
                arguments: json!({"query": "rust"}),
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: json!({"city": "Paris"}),
            },
        ]
    );
}

/// Two complete calls in one delta with no `index`: they must not collapse into one
/// concatenated call.
#[test]
fn parallel_tool_calls_without_index_do_not_collapse() {
    let deltas = vec![json!({"tool_calls": [
        {"type": "function", "function": {"name": "search", "arguments": "{\"query\":\"rust\"}"}},
        {"type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}},
    ]})];
    let rendered = render(&deltas, ReasoningStart::Outside);
    assert_eq!(
        parse(&rendered, UnifiedParserStartingState::Response),
        vec![
            UnifiedEvent::ToolCall {
                name: "search".into(),
                arguments: json!({"query": "rust"}),
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: json!({"city": "Paris"}),
            },
        ]
    );
}

/// The DSML grammar has no escaping, so a string value carrying the parameter-close
/// marker must be rejected rather than breaking the unified parser mid-stream.
#[test]
fn reserved_marker_in_argument_value_is_rejected() {
    let mut renderer = DeepseekV41Renderer::new(ReasoningStart::Outside);
    let bar = '\u{ff5c}';
    let arguments = format!("{{\"text\":\"a</{bar}DSML{bar} parameter>b\"}}");
    let delta = json!({
        "tool_calls": [{"index": 0, "type": "function", "function": {
            "name": "write", "arguments": arguments
        }}]
    });
    renderer.push_delta(&delta).expect("buffers the call");
    assert!(renderer.finish(Some("tool_calls")).is_err());
}
