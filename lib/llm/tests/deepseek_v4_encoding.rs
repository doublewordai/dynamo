// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for DeepSeek V4 encoding against official test data
//!
//! These tests use the official test files from:
//! https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/encoding

use dynamo_renderer::deepseek::v4::{ThinkingMode, encode_messages};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::PathBuf;

fn get_test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/deepseek-v4")
}

/// Load an input fixture. V4 fixtures come in two shapes:
///   1. `{"tools": [...], "messages": [...]}` — tools injected on first (system) message
///   2. bare `[...]` — just the messages array
fn load_messages(path: &PathBuf) -> Vec<JsonValue> {
    let raw: JsonValue = serde_json::from_str(
        &fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {:?}", path)),
    )
    .unwrap_or_else(|_| panic!("Failed to parse {:?}", path));

    if let Some(messages) = raw.get("messages").and_then(|m| m.as_array()) {
        let mut messages = messages.clone();
        if let Some(tools) = raw.get("tools")
            && let Some(first) = messages.get_mut(0)
            && let Some(obj) = first.as_object_mut()
        {
            obj.insert("tools".to_string(), tools.clone());
        }
        messages
    } else if let Some(arr) = raw.as_array() {
        arr.clone()
    } else {
        panic!("Unexpected input shape in {:?}", path);
    }
}

fn run_official_test(input_file: &str, output_file: &str, thinking_mode: ThinkingMode) {
    let test_dir = get_test_data_path();
    let messages = load_messages(&test_dir.join(input_file));
    let expected = fs::read_to_string(test_dir.join(output_file))
        .unwrap_or_else(|_| panic!("Failed to read {}", output_file));

    let actual = encode_messages(&messages, thinking_mode, true)
        .unwrap_or_else(|e| panic!("encode_messages failed for {}: {:?}", input_file, e));

    let exp = expected.trim_end();
    let act = actual.trim_end();

    if exp != act {
        println!("=== Test: {} ===", input_file);
        let exp_lines: Vec<&str> = exp.lines().collect();
        let act_lines: Vec<&str> = act.lines().collect();
        for (i, (el, al)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
            if el != al {
                println!("Line {} differs:", i + 1);
                println!("  Expected: {:?}", el);
                println!("  Actual:   {:?}", al);
                break;
            }
        }
        if exp_lines.len() != act_lines.len() {
            println!(
                "\nLine count mismatch: expected {} lines, got {} lines",
                exp_lines.len(),
                act_lines.len()
            );
        }
        panic!("Output does not match expected for {}", input_file);
    }
}

/// Case 1 — thinking mode, single tool, tool result round-trip.
#[test]
fn test_official_thinking_with_tools() {
    run_official_test(
        "test_input_1.json",
        "test_output_1.txt",
        ThinkingMode::Thinking,
    );
}

/// Case 2 — thinking mode, no tools, multi-turn (drop_thinking strips earlier reasoning).
#[test]
fn test_official_thinking_no_tools_multiturn() {
    run_official_test(
        "test_input_2.json",
        "test_output_2.txt",
        ThinkingMode::Thinking,
    );
}

/// Case 3 — thinking mode, developer role with tools + latest_reminder + tool result.
#[test]
fn test_official_developer_with_tools_and_reminder() {
    run_official_test(
        "test_input_3.json",
        "test_output_3.txt",
        ThinkingMode::Thinking,
    );
}

/// Case 4 — chat mode, latest_reminder + task="action" + mask preservation.
#[test]
fn test_official_chat_mode_action_task() {
    run_official_test("test_input_4.json", "test_output_4.txt", ThinkingMode::Chat);
}

/// The reference encoder shipped with DeepSeek-V4-Flash-0731 defines three effort
/// levels; `low` is the no-prefix baseline. These must stay byte-identical to the
/// engine-side encoder, because the same model is served both frontend-rendered
/// (SGLang workers) and engine-rendered (vLLM workers).
const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EFFORT_HIGH_PREFIX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const EFFORT_MAX_PREFIX: &str = "Reasoning Effort: Beyond maximum \u{2014} exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

fn render_with_effort(effort: Option<&str>) -> String {
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use dynamo_renderer::OAIPromptFormatter;
    use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;

    let mut body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": "hi"}],
    });
    if let Some(effort) = effort {
        body["reasoning_effort"] = JsonValue::from(effort);
    }
    let mut request: NvCreateChatCompletionRequest =
        serde_json::from_value(body).expect("build request");
    request
        .normalize_reasoning_template_args()
        .expect("normalize reasoning controls");
    let prompt = DeepSeekV4Formatter::new_thinking()
        .render(&request)
        .expect("render prompt");
    // The effort prefix sits immediately after the BOS token; a missing BOS is
    // itself a regression, so fail rather than fall through to the raw prompt.
    prompt
        .strip_prefix(BOS_TOKEN)
        .unwrap_or_else(|| panic!("prompt should open with the BOS token, got: {prompt:?}"))
        .to_string()
}

#[test]
fn reasoning_effort_prefixes_match_reference_encoder() {
    for effort in [None, Some("high"), Some("xhigh"), Some("medium")] {
        let prompt = render_with_effort(effort);
        assert!(
            prompt.starts_with(EFFORT_HIGH_PREFIX),
            "effort {effort:?} should render the high prefix, got: {prompt:?}"
        );
    }

    let prompt = render_with_effort(Some("max"));
    assert!(
        prompt.starts_with(EFFORT_MAX_PREFIX),
        "max should render its own prefix, got: {prompt:?}"
    );

    for effort in [Some("low"), Some("minimal"), Some("none")] {
        let prompt = render_with_effort(effort);
        assert!(
            !prompt.contains("Reasoning Effort:"),
            "effort {effort:?} is the no-prefix baseline, got: {prompt:?}"
        );
    }
}
