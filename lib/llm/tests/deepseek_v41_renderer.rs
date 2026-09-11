// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4.1 prompt rendering through the frontend's request type: the
//! reasoning-effort budget reaches the renderer from both the top-level
//! `reasoning_effort` field and `chat_template_kwargs`, named levels map as in
//! the reference encoder, and `none` / `thinking: false` turn thinking off.

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
use serde_json::{Value, json};

fn render(fields: Value) -> anyhow::Result<String> {
    let mut payload =
        json!({"model":"deepseek-v4.1", "messages":[{"role":"user", "content":"Hello"}]});
    payload
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(payload)?;
    request.normalize_reasoning_template_args()?;
    DeepSeekV41Formatter.render(&request)
}

#[test]
fn normalized_effort_matches_the_reference_encoder_for_both_request_fields() {
    // Thinking on at the default budget when nothing is set.
    assert!(render(json!({})).unwrap().contains("Reasoning Effort: 75 "));
    // Levels the model does not define are rejected rather than clamped.
    assert!(render(json!({"reasoning_effort":"xhigh"})).is_err());
    for field in ["chat_template_args", "chat_template_kwargs"] {
        for (effort, budget) in [("low", 50), ("high", 75), ("max", 100)] {
            for fields in [
                json!({"reasoning_effort":effort}),
                json!({(field):{"reasoning_effort":effort}}),
            ] {
                let output = render(fields).unwrap();
                assert!(output.contains(&format!("Reasoning Effort: {budget} ")));
                assert!(output.ends_with("<think>"));
            }
        }
        assert!(
            render(json!({(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 37 ")
        );
        // The top-level field wins over the template argument.
        assert!(
            render(json!({"reasoning_effort":"low",(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 50 ")
        );
        for fields in [
            json!({"reasoning_effort":"none"}),
            json!({(field):{"reasoning_effort":"none"}}),
            json!({(field):{"thinking":false}}),
        ] {
            let output = render(fields).unwrap();
            assert!(!output.contains("Reasoning Effort:"));
            assert!(output.ends_with("</think>"));
        }
        for invalid in [
            json!(0),
            json!(101),
            json!(true),
            json!(1.5),
            json!("xhigh"),
        ] {
            assert!(render(json!({(field):{"reasoning_effort":invalid}})).is_err());
        }
    }
}

#[test]
fn v41_tool_calls_render_as_dsml_with_leading_space_tag_names() {
    let output = render(json!({
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }}],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "18C"}
        ]
    }))
    .unwrap();
    assert!(output.contains("<｜DSML｜ calls>"), "{output}");
    assert!(
        output.contains(
            "<｜DSML｜ parameter name=\"location\" string=\"true\">Paris</｜DSML｜ parameter>"
        ),
        "{output}"
    );
    assert!(output.contains("<tool_result>"), "{output}");
}
