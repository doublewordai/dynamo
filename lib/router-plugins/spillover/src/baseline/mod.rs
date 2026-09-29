// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseline: a per-candidate port of Dynamo's default scorer and picker.
//!
//! The fork keeps the default scorer and picker private
//! (`lib/kv-router/src/scheduling/selector/default.rs`), and its plugin API scores one
//! [`WorkerCandidate`](dynamo_kv_router::plugins::worker_selection::WorkerCandidate) at a time
//! rather than a batch. This module re-implements the same cost formula and the same
//! seeded softmax/tie-breaking picker using only the public plugin inputs.
//!
//! Inputs the default computes privately, and how the port reads them through the plugin API:
//!
//! - `min_active_prefill_tokens`: the default computes a batch-wide minimum over eligible
//!   workers and subtracts it before decaying overlap credit. The host now materializes that same
//!   floor on [`WorkerSelectionContext`](dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext);
//!   the port subtracts it identically.
//! - `raw_prefill_blocks`: the host computes it from the request's `isl_tokens`, the worker's
//!   effective cached tokens and its active prefill. [`WorkerLoadInput`](dynamo_kv_router::plugins::worker_selection::WorkerLoadInput)
//!   now exposes it, and the port reads it verbatim.
//! - the `effective_overlap_blocks` fallback: when the request has no per-tier overlap map the
//!   default uses effective overlap instead of the reported device overlap. The port reads
//!   [`WorkerCacheInput::effective_overlap_blocks`](dynamo_kv_router::plugins::worker_selection::WorkerCacheInput::effective_overlap_blocks)
//!   when [`WorkerSelectionContext::has_tier_overlap_blocks`](dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext::has_tier_overlap_blocks)
//!   is false.
//! - per-request `router_config_override` weights: the override is applied by the host before the
//!   context is built, and the port reads the overridden weights from the context rather than
//!   from the config captured at policy-construction time.

mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{WorkerPicker, WorkerScorer};
use dynamo_kv_router::{KvRouterConfig, WorkerType};
use parking_lot::Mutex;

/// Shared random source for the picker. `Some` gives reproducible selection (tests, sims).
pub type PickerRng = Option<Arc<Mutex<fastrand::Rng>>>;

/// Dynamo's default scorer for this role, configured from `config`.
pub fn baseline_scorer(config: &KvRouterConfig, role: WorkerType) -> Box<dyn WorkerScorer> {
    Box::new(scorer::BaselineScorer::new(
        config,
        role.default_selector_label(),
    ))
}

/// Dynamo's default picker, configured from `config`.
pub fn baseline_picker(config: &KvRouterConfig, rng: PickerRng) -> Box<dyn WorkerPicker> {
    Box::new(picker::BaselinePicker::new(config.router_temperature, rng))
}
