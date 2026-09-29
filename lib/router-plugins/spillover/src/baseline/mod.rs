// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copied from ai-dynamo/dynamo@494d6e24 lib/router-plugins/builtin/src/default/

//! Baseline: Dynamo's default scorer and picker.
//!
//! Upstream keeps these private (`lib/router-plugins/builtin/src/default/`), so this module
//! carries a copy taken from the pinned upstream rev. Keep it byte-for-byte equivalent in
//! behaviour; `tests/equivalence.rs` checks that models without spillover parameters choose
//! exactly what `dynamo_kv_router::DefaultWorkerSelector` chooses.

mod parameters;
mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{WorkerPicker, WorkerScorer};
use dynamo_kv_router::{KvRouterConfig, WorkerType};
use parking_lot::Mutex;

use parameters::PolicyParameters;

/// Shared random source for the picker. `Some` gives reproducible selection (tests, sims).
pub type PickerRng = Option<Arc<Mutex<fastrand::Rng>>>;

/// Dynamo's default scorer for this role, configured from `config`.
pub fn baseline_scorer(config: &KvRouterConfig, role: WorkerType) -> Box<dyn WorkerScorer> {
    let parameters = PolicyParameters::from(config);
    let is_plain_decode = role == WorkerType::Decode && !config.conditional_disagg_enabled;
    scorer::build(&parameters, role.default_selector_label(), is_plain_decode)
}

/// Dynamo's default picker, configured from `config`.
pub fn baseline_picker(config: &KvRouterConfig, rng: PickerRng) -> Box<dyn WorkerPicker> {
    let parameters = PolicyParameters::from(config);
    Box::new(picker::DefaultPicker::new(
        parameters.router_temperature,
        rng,
    ))
}
