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

/// Development stand-in for the real policy: the spillover cost formula, implemented directly.
///
/// Scenario tests use the real policy; this is kept for the `--heuristic` CLI flag and one smoke
/// test so the event loop, workload and report stay independently exercisable.
///
/// The cost mirrors `TierScorer` stacked on the baseline load signal: the projected decode
/// footprint (current decode blocks plus the arriving request's uncached blocks), the pending
/// term, and the tier preference or primary failover penalty. It deliberately has no separate
/// prefill term: the arriving prompt is already counted once by `additional_active_blocks`, and
/// charging it again (or charging its cached prefix) would double-count what the real selector
/// counts once.
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
        let load = input
            .request
            .worker_loads
            .get(&worker)
            .copied()
            .unwrap_or_default();
        // `additional_active_blocks` is the arriving request's uncached blocks and
        // `active_decode_blocks` is the worker's current footprint, so this is already the
        // projected post-admission decode cost. The real policy's `decode_cost_blocks` is the
        // same quantity; there is no separate prefill charge.
        let decode_blocks = (load.active_decode_blocks + load.additional_active_blocks) as f64;
        let pending = self.model.pending_weight_blocks * load.active_requests as f64;

        if let Some(tier) = self.tier(worker.dp_rank) {
            return decode_blocks + pending + tier.penalty_blocks + tier.weight_blocks;
        }

        let capacity = input
            .workers
            .get(&worker.worker_id)
            .and_then(|config| config.total_kv_blocks)
            .filter(|blocks| *blocks > 0)
            .map(|blocks| blocks as f64)
            .or(self.model.primary_capacity_blocks);
        let seq_capacity = input
            .workers
            .get(&worker.worker_id)
            .and_then(|config| config.max_num_seqs)
            .filter(|seqs| *seqs > 0)
            .map(|seqs| seqs as f64)
            .or(self.model.primary_max_requests);
        let kv_occupancy = capacity.map(|capacity| decode_blocks / capacity);
        // Match the real policy: count the arriving request, which `decode_blocks` already does.
        let concurrency_occupancy =
            seq_capacity.map(|capacity| (load.active_requests as f64 + 1.0) / capacity);
        let occupancy = match (kv_occupancy, concurrency_occupancy) {
            (Some(kv), Some(concurrency)) => Some(kv.max(concurrency)),
            (Some(kv), None) => Some(kv),
            (None, Some(concurrency)) => Some(concurrency),
            (None, None) => None,
        };
        let failover = match occupancy {
            Some(occupancy) if occupancy > self.model.occupancy_threshold => {
                self.model.failover_penalty_blocks
            }
            _ => 0.0,
        };
        decode_blocks + pending + failover
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The stand-in's cost is the projected decode footprint: `additional_active_blocks`, the
    /// arriving request's uncached blocks, must be included, because that is the quantity the
    /// real policy reads as `decode_cost_blocks`.
    #[test]
    fn decode_cost_includes_additional_active_blocks() {
        let model = ModelParameters {
            occupancy_threshold: 0.8,
            primary_capacity_blocks: Some(1000.0),
            primary_max_requests: None,
            failover_penalty_blocks: 0.0,
            pending_weight_blocks: 0.0,
            tiers: Vec::new(),
        };
        let selector = HeuristicSelector::new(model);
        let worker = WorkerWithDpRank::new(0, 0);
        let mut workers = HashMap::new();
        workers.insert(0, testkit::SimWorker::primary(1000));
        let mut request = testkit::empty_request(64);
        testkit::set_rank(
            &mut request,
            worker,
            testkit::RankSignals {
                additional_active_blocks: 5,
                ..testkit::RankSignals::default()
            },
            16,
        );
        let input = SelectionInput {
            request: &request,
            workers: &workers,
            block_size: 16,
        };
        // 5 additional decode blocks, below the failover threshold so no penalty applies.
        assert_eq!(selector.cost(worker, &input), 5.0);
    }
}
