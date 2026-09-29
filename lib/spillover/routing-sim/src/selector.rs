// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selector abstraction: the real spillover policy, a development stand-in, and upstream's
//! reference selector used to check the no-parameters case.
//!
//! The simulation never talks to a frontend; it builds a `SchedulingRequest` (through `testkit`)
//! for each decision and hands it to one of these implementations.

use std::collections::HashMap;
use std::sync::Arc;

use dw_spillover_policy::{ModelParameters, SpilloverParameters, build_policy};
use dw_spillover_testkit as testkit;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    DefaultWorkerSelector, KvRouterConfig, SchedulingRequest, WorkerSelector, WorkerType,
};
use parking_lot::Mutex;

/// Everything a selector needs for one decision.
pub struct SelectionInput<'a> {
    pub request: &'a SchedulingRequest,
    pub workers: &'a HashMap<u64, testkit::SimWorker>,
    pub block_size: u32,
    /// Hosted worker id to capacity in blocks.
    pub hosted_capacity: &'a HashMap<u64, f64>,
}

/// One routing decision.
pub trait Selector {
    fn select(&mut self, input: &SelectionInput<'_>) -> Option<WorkerWithDpRank>;
    /// Short name used in reports.
    fn label(&self) -> &'static str;
}

/// The real policy, built exactly as the router builds it.
pub struct PolicySelector {
    policy: dynamo_kv_router::WorkerSelectionPolicy,
}

impl PolicySelector {
    pub fn new(
        config: &KvRouterConfig,
        model: &str,
        params: &SpilloverParameters,
        seed: u64,
    ) -> Self {
        let rng = Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed))));
        Self {
            policy: build_policy(config, WorkerType::Aggregated, model, params, rng),
        }
    }
}

impl Selector for PolicySelector {
    fn select(&mut self, input: &SelectionInput<'_>) -> Option<WorkerWithDpRank> {
        let selection = testkit::selection_input(input.workers, input.request, input.block_size);
        self.policy
            .select_worker(selection)
            .ok()
            .map(|result| result.worker)
    }

    fn label(&self) -> &'static str {
        "policy"
    }
}

/// Upstream's reference selector, used to check that models without spillover parameters make the
/// same decisions. Only available because this crate enables `dynamo-kv-router`'s `bench` feature.
pub struct DefaultSelector {
    selector: DefaultWorkerSelector,
}

impl DefaultSelector {
    pub fn new(config: &KvRouterConfig, seed: u64) -> Self {
        Self {
            selector: DefaultWorkerSelector::new_seeded(
                Some(config.clone()),
                WorkerType::Aggregated.default_selector_label(),
                seed,
            ),
        }
    }
}

impl Selector for DefaultSelector {
    fn select(&mut self, input: &SelectionInput<'_>) -> Option<WorkerWithDpRank> {
        let selection = testkit::selection_input(input.workers, input.request, input.block_size);
        self.selector
            .select_worker(selection)
            .ok()
            .map(|result| result.worker)
    }

    fn label(&self) -> &'static str {
        "default"
    }
}

/// Development stand-in for the real policy: the documented cost formula, implemented directly.
///
/// Scenario tests use the real policy; this is kept for the `--heuristic` CLI flag and one smoke
/// test so the event loop, workload and report stay independently exercisable.
pub struct HeuristicSelector {
    model: ModelParameters,
}

impl HeuristicSelector {
    pub fn new(model: ModelParameters) -> Self {
        Self { model }
    }

    fn tier(&self, dp_rank: u32) -> Option<&dw_spillover_policy::TierParameters> {
        self.model
            .tiers
            .iter()
            .find(|tier| (tier.dp_ranks[0]..=tier.dp_ranks[1]).contains(&dp_rank))
    }

    fn cost(&self, worker: WorkerWithDpRank, input: &SelectionInput<'_>) -> f64 {
        let overlap = input
            .request
            .overlap
            .tier_overlap_blocks
            .device
            .get(&worker)
            .copied()
            .unwrap_or(0) as f64;
        let cached_tokens = input
            .request
            .overlap
            .effective_cached_tokens
            .get(&worker)
            .copied()
            .unwrap_or(0);
        let load = input
            .request
            .worker_loads
            .get(&worker)
            .copied()
            .unwrap_or_default();
        let uncached_tokens = input.request.isl_tokens.saturating_sub(cached_tokens);
        // The reference formula counts the prompt once, minus the cache-affinity credit.
        let raw_prefill_blocks = (load.active_prefill_tokens + uncached_tokens + cached_tokens)
            as f64
            / input.block_size as f64;
        let prefill_blocks = (raw_prefill_blocks - overlap).max(0.0);
        let decode_blocks = load.active_decode_blocks as f64;
        let pending = self.model.pending_weight_blocks * load.active_requests as f64;

        if let Some(tier) = self.tier(worker.dp_rank) {
            return prefill_blocks
                + decode_blocks
                + pending
                + tier.penalty_blocks
                + tier.weight_blocks;
        }

        let capacity = input
            .hosted_capacity
            .get(&worker.worker_id)
            .copied()
            .unwrap_or(self.model.hosted_capacity_blocks);
        let occupancy = if capacity > 0.0 {
            decode_blocks / capacity
        } else {
            0.0
        };
        let failover = if occupancy >= self.model.occupancy_threshold {
            self.model.failover_penalty_blocks
        } else {
            0.0
        };
        prefill_blocks + decode_blocks + pending + failover
    }
}

impl Selector for HeuristicSelector {
    fn select(&mut self, input: &SelectionInput<'_>) -> Option<WorkerWithDpRank> {
        input
            .workers
            .keys()
            .filter(|worker_id| {
                input
                    .request
                    .allowed_worker_ids
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(worker_id))
            })
            .map(|worker_id| {
                let dp_rank = input
                    .workers
                    .get(worker_id)
                    .map(|config| config.dp_start_rank)
                    .unwrap_or(0);
                WorkerWithDpRank::new(*worker_id, dp_rank)
            })
            .min_by(|a, b| {
                let cost_a = self.cost(*a, input);
                let cost_b = self.cost(*b, input);
                cost_a
                    .total_cmp(&cost_b)
                    .then_with(|| a.worker_id.cmp(&b.worker_id))
                    .then_with(|| a.dp_rank.cmp(&b.dp_rank))
            })
    }

    fn label(&self) -> &'static str {
        "heuristic"
    }
}
