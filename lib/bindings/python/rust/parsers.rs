// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::reasoning::get_available_reasoning_parsers;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use pyo3::prelude::*;

/// Parser names served only by `dynamo-parsers-v2` unified parsers, which the
/// legacy `dynamo-parsers` registries do not know. The worker validates its
/// `--dyn-tool-call-parser` / `--dyn-reasoning-parser` flags against these lists,
/// so a unified-only family must be appended here or it cannot be configured.
fn with_unified_families(mut names: Vec<&'static str>) -> Vec<&'static str> {
    for &name in
        dynamo_llm::protocols::openai::chat_completions::unified_parser::UNIFIED_PARSER_NAMES
    {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Get list of available tool parser names
#[pyfunction]
pub fn get_tool_parser_names() -> Vec<&'static str> {
    with_unified_families(get_available_tool_parsers())
}

/// Get list of available reasoning parser names
#[pyfunction]
pub fn get_reasoning_parser_names() -> Vec<&'static str> {
    with_unified_families(get_available_reasoning_parsers())
}

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    Ok(())
}
