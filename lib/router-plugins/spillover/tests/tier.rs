// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the tier scorer, driven through `WorkerSelector` with testkit inputs.

use std::collections::HashMap;

use dw_spillover_policy::TierScorer;
use dw_spillover_policy::params::{ModelParameters, TierParameters};
use dw_spillover_testkit::{RankSignals, SimWorker, empty_request, selection_input, set_rank};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    KvRouterConfig, SchedulingRequest, WorkerInputView, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector,
};

/// The tier scorer only writes cost contributions; the picker just takes the cheapest row.
struct LowestCostPicker;

impl WorkerPicker for LowestCostPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost().total_cmp(&b.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))
    }
}

fn params() -> ModelParameters {
    ModelParameters {
        occupancy_threshold: 0.9,
        hosted_capacity_blocks: 1000.0,
        failover_penalty_blocks: 500.0,
        pending_weight_blocks: 10.0,
        tiers: vec![
            TierParameters {
                name: "x".into(),
                dp_ranks: [1000, 1999],
                penalty_blocks: 150.0,
                weight_blocks: 0.0,
            },
            TierParameters {
                name: "y".into(),
                dp_ranks: [2000, 2999],
                penalty_blocks: 150.0,
                weight_blocks: 50.0,
            },
        ],
    }
}

fn signals(active_requests: usize, decode_blocks: usize) -> RankSignals {
    RankSignals {
        active_requests,
        active_decode_blocks: decode_blocks,
        ..Default::default()
    }
}

fn select(workers: &HashMap<u64, SimWorker>, request: &SchedulingRequest) -> WorkerWithDpRank {
    let policy = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(TierScorer::new(params()))],
        Box::new(LowestCostPicker),
    );
    policy
        .select_worker(selection_input(workers, request, 16))
        .unwrap()
        .worker
}

fn cluster() -> HashMap<u64, SimWorker> {
    HashMap::from([
        (0, SimWorker::hosted(1000)),
        (1, SimWorker::proxy(1000)),
        (2, SimWorker::proxy(2000)),
    ])
}

fn request_with_hosted(active_requests: usize, decode_blocks: usize) -> SchedulingRequest {
    let mut request = empty_request(16);
    set_rank(
        &mut request,
        WorkerWithDpRank::new(0, 0),
        signals(active_requests, decode_blocks),
        16,
    );
    request
}

fn set_proxy(
    request: &mut SchedulingRequest,
    worker_id: u64,
    dp_rank: u32,
    active_requests: usize,
) {
    set_rank(
        request,
        WorkerWithDpRank::new(worker_id, dp_rank),
        signals(active_requests, 0),
        16,
    );
}

#[test]
fn hosted_under_threshold_beats_both_proxies() {
    let workers = cluster();
    let mut request = request_with_hosted(0, 100);
    set_proxy(&mut request, 1, 1000, 0);
    set_proxy(&mut request, 2, 2000, 0);

    assert_eq!(select(&workers, &request).worker_id, 0);
}

#[test]
fn hosted_at_or_over_threshold_loses_to_proxy_x() {
    let workers = cluster();

    for decode_blocks in [900, 950] {
        let mut request = request_with_hosted(0, decode_blocks);
        set_proxy(&mut request, 1, 1000, 0);
        set_proxy(&mut request, 2, 2000, 0);

        assert_eq!(
            select(&workers, &request).worker_id,
            1,
            "decode={decode_blocks}"
        );
    }
}

#[test]
fn proxy_x_beats_proxy_y_at_equal_load() {
    let workers = cluster();
    let mut request = request_with_hosted(0, 950);
    set_proxy(&mut request, 1, 1000, 0);
    set_proxy(&mut request, 2, 2000, 0);

    assert_eq!(select(&workers, &request).worker_id, 1);
}

#[test]
fn proxy_y_wins_once_x_pending_cost_exceeds_the_weight_gap() {
    let workers = cluster();

    // Gap: Y.weight_blocks - X.weight_blocks = 50, pending weight is 10.
    // At 4 requests X costs 190 < Y's 200; at 6 requests X costs 210 > Y's 200.
    for (x_requests, expected) in [(4, 1), (6, 2)] {
        let mut request = request_with_hosted(0, 950);
        set_proxy(&mut request, 1, 1000, x_requests);
        set_proxy(&mut request, 2, 2000, 0);

        assert_eq!(
            select(&workers, &request).worker_id,
            expected,
            "x_requests={x_requests}"
        );
    }
}

#[test]
fn missing_load_observation_is_treated_as_idle() {
    // Two identical hosted workers; only worker 0 has a (busy) load observation.
    let workers = HashMap::from([(0, SimWorker::hosted(1000)), (1, SimWorker::hosted(1000))]);
    let mut request = request_with_hosted(5, 100);

    // Worker 1 has no entry at all in `worker_loads`.
    assert_eq!(select(&workers, &request).worker_id, 1);

    // Swapping the busy observation to worker 1 confirms the missing row, not the values,
    // decided: worker 0 is now the one with no observation and wins.
    request.worker_loads.clear();
    set_rank(
        &mut request,
        WorkerWithDpRank::new(1, 0),
        signals(5, 100),
        16,
    );
    assert_eq!(select(&workers, &request).worker_id, 0);
}
