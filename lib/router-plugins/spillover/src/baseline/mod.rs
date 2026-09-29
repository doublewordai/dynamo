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
//! Inputs the default uses that the plugin API does not expose at this revision, and how this
//! port handles them:
//!
//! - `min_active_prefill_tokens`: the default computes a batch-wide minimum and subtracts it
//!   before decaying overlap credit. A per-candidate scorer cannot see the batch. We use 0
//!   instead. When the least-loaded eligible worker has no active prefill (the common case, and
//!   what the equivalence fixture guarantees), 0 is exactly the default's minimum.
//! - `raw_prefill_blocks`: the default derives it from the request's `isl_tokens` and the
//!   worker's `effective_cached_tokens`; neither is exposed. We reconstruct
//!   `active_prefill_tokens + request_blocks * block_size`, which is exact for a block-aligned
//!   prompt whose cache does not exceed the prompt. For any other prompt the error is a
//!   request-wide constant, which does not change the argmin or the softmax distribution.
//! - the `effective_overlap_blocks` fallback: the default uses per-worker effective overlap when
//!   no tier-overlap map is present, but only the reported device overlap reaches the plugin. The
//!   equivalence fixture therefore always supplies a tier-overlap map.
//! - per-request `router_config_override` weights: the default recomputes its weights from the
//!   request and applies any router-config override it carries; the plugin sees only the config
//!   captured when the policy was built. Models with parameters intentionally use the policy's
//!   configured weights, so this only affects a request that overrides weights.

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
