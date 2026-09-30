// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the tier scorer, driven through `WorkerSelector` with testkit inputs.

use std::collections::{HashMap, HashSet};

use dw_spillover_policy::TierScorer;
use dw_spillover_policy::params::{ModelParameters, TierParameters};
use dw_spillover_testkit::{RankSignals, SimWorker, empty_request, selection_input, set_rank};
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::{
    KvRouterConfig, SchedulingRequest, WorkerInputView, WorkerLoadProjection, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionInput, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelector,
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
        primary_capacity_blocks: Some(1000.0),
        primary_max_requests: None,
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

/// Parameters whose failover boundary depends only on occupancy: zero pending weight keeps the
/// active-request term out of the comparison, so the proxy's 150 penalty is the whole difference.
fn boundary_params(occupancy_threshold: f64) -> ModelParameters {
    ModelParameters {
        occupancy_threshold,
        primary_capacity_blocks: None,
        primary_max_requests: None,
        failover_penalty_blocks: 500.0,
        pending_weight_blocks: 0.0,
        tiers: vec![TierParameters {
            name: "x".into(),
            dp_ranks: [1000, 1999],
            penalty_blocks: 150.0,
            weight_blocks: 0.0,
        }],
    }
}

fn signals(active_requests: usize, decode_blocks: usize) -> RankSignals {
    RankSignals {
        active_requests,
        active_decode_blocks: decode_blocks,
        ..Default::default()
    }
}

fn policy(params: ModelParameters) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(TierScorer::new(params))],
        Box::new(LowestCostPicker),
    )
}

fn select_with(
    params: ModelParameters,
    workers: &HashMap<u64, SimWorker>,
    request: &SchedulingRequest,
) -> WorkerWithDpRank {
    policy(params)
        .select_worker(selection_input(workers, request, 16))
        .unwrap()
        .worker
}

fn select(workers: &HashMap<u64, SimWorker>, request: &SchedulingRequest) -> WorkerWithDpRank {
    select_with(params(), workers, request)
}

fn cluster() -> HashMap<u64, SimWorker> {
    HashMap::from([
        (0, SimWorker::primary(1000)),
        (1, SimWorker::proxy(1000)),
        (2, SimWorker::proxy(2000)),
    ])
}

/// A primary worker that advertises neither a KV block total nor a sequence limit.
fn unknown_primary() -> SimWorker {
    SimWorker {
        total_kv_blocks: None,
        max_num_seqs: None,
        ..SimWorker::primary(0)
    }
}

fn request_with_primary(active_requests: usize, decode_blocks: usize) -> SchedulingRequest {
    let mut request = empty_request(16);
    set_rank(
        &mut request,
        WorkerWithDpRank::new(0, 0),
        signals(active_requests, decode_blocks),
        16,
    );
    request
}

fn set_primary(
    request: &mut SchedulingRequest,
    worker_id: u64,
    active_requests: usize,
    decode_blocks: usize,
) {
    set_rank(
        request,
        WorkerWithDpRank::new(worker_id, 0),
        signals(active_requests, decode_blocks),
        16,
    );
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
fn primary_under_threshold_beats_both_proxies() {
    let workers = cluster();
    let mut request = request_with_primary(0, 100);
    set_proxy(&mut request, 1, 1000, 0);
    set_proxy(&mut request, 2, 2000, 0);

    assert_eq!(select(&workers, &request).worker_id, 0);
}

#[test]
fn primary_above_threshold_loses_to_proxy_x() {
    let workers = cluster();

    // Occupancy is strictly greater than the threshold: exactly 0.9 (900 blocks) no longer
    // spills, so the first spilling value is 901.
    for decode_blocks in [901, 950] {
        let mut request = request_with_primary(0, decode_blocks);
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
fn primary_at_exactly_the_threshold_does_not_spill() {
    let workers = cluster();
    let mut request = request_with_primary(0, 900);
    set_proxy(&mut request, 1, 1000, 0);
    set_proxy(&mut request, 2, 2000, 0);

    assert_eq!(select(&workers, &request).worker_id, 0);
}

#[test]
fn proxy_x_beats_proxy_y_at_equal_load() {
    let workers = cluster();
    let mut request = request_with_primary(0, 950);
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
        let mut request = request_with_primary(0, 950);
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
    // Two identical primary workers; only worker 0 has a (busy) load observation.
    let workers = HashMap::from([(0, SimWorker::primary(1000)), (1, SimWorker::primary(1000))]);
    let mut request = request_with_primary(5, 100);

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

#[test]
fn primary_occupancy_counts_the_requests_own_blocks() {
    // The host materializes `decode_cost_blocks` as active decode blocks plus the request's own
    // uncached blocks, i.e. post-admission. Pin that: 890 + 20 crosses the 0.9 threshold even
    // though the worker only holds 890 blocks today.
    let workers = HashMap::from([(0, SimWorker::primary(1000)), (1, SimWorker::proxy(1000))]);
    let mut request = empty_request(16);
    let mut primary = WorkerLoadProjection {
        active_requests: 0,
        active_prefill_tokens: 0,
        active_decode_blocks: 890,
        additional_active_blocks: 20,
    };
    set_proxy(&mut request, 1, 1000, 0);

    request
        .worker_loads
        .insert(WorkerWithDpRank::new(0, 0), primary);
    assert_eq!(select(&workers, &request).worker_id, 1, "over threshold");

    primary.additional_active_blocks = 0;
    request
        .worker_loads
        .insert(WorkerWithDpRank::new(0, 0), primary);
    assert_eq!(select(&workers, &request).worker_id, 0, "under threshold");
}

#[test]
fn mixed_capacities_use_each_workers_advertised_blocks() {
    // The same decode footprint is a different occupancy on each worker, so only the smaller
    // primary spills.
    let workers = HashMap::from([
        (0, SimWorker::primary(1000)),
        (1, SimWorker::primary(2000)),
        (2, SimWorker::proxy(1000)),
    ]);
    let mut request = empty_request(16);
    set_primary(&mut request, 0, 0, 950);
    set_primary(&mut request, 1, 0, 950);
    set_proxy(&mut request, 2, 1000, 0);

    assert_eq!(
        select(&workers, &request).worker_id,
        1,
        "0.95 occupancy on worker 0 spills; 0.475 on worker 1 does not"
    );
}

#[test]
fn concurrency_signal_spills_when_kv_does_not() {
    // Advertised KV is huge, so the decode footprint is negligible; the sequence limit is what
    // spills once the arriving request would take the worker past its concurrency limit.
    let workers = HashMap::from([
        (0, SimWorker::primary_with_seq_capacity(1_000_000, 8)),
        (1, SimWorker::proxy(1000)),
    ]);

    let mut under = empty_request(16);
    set_primary(&mut under, 0, 0, 100);
    set_proxy(&mut under, 1, 1000, 0);
    assert_eq!(
        select(&workers, &under).worker_id,
        0,
        "1/8 occupancy is below the threshold"
    );

    let mut over = empty_request(16);
    set_primary(&mut over, 0, 8, 100);
    set_proxy(&mut over, 1, 1000, 0);
    assert_eq!(
        select(&workers, &over).worker_id,
        1,
        "(8 + this request)/8 exceeds the threshold"
    );
}

#[test]
fn kv_signal_spills_when_concurrency_does_not() {
    // A large advertised sequence limit leaves the concurrency signal tiny; the KV signal spills.
    let workers = HashMap::from([
        (0, SimWorker::primary_with_seq_capacity(1000, 1000)),
        (1, SimWorker::proxy(1000)),
    ]);

    let mut under = empty_request(16);
    set_primary(&mut under, 0, 0, 100);
    set_proxy(&mut under, 1, 1000, 0);
    assert_eq!(select(&workers, &under).worker_id, 0);

    let mut over = empty_request(16);
    set_primary(&mut over, 0, 0, 950);
    set_proxy(&mut over, 1, 1000, 0);
    assert_eq!(select(&workers, &over).worker_id, 1);
}

#[test]
fn occupancy_takes_the_worst_of_both_signals() {
    // Neither signal alone spills: KV is 0.5 and concurrency is 2/8. Together, the larger of the
    // two still does not. Push concurrency to the limit and the worker spills despite low KV.
    let workers = HashMap::from([
        (0, SimWorker::primary_with_seq_capacity(2000, 8)),
        (1, SimWorker::proxy(1000)),
    ]);

    let mut request = empty_request(16);
    set_primary(&mut request, 0, 0, 1000);
    set_proxy(&mut request, 1, 1000, 0);
    assert_eq!(select(&workers, &request).worker_id, 0, "0.5 KV, 1/8 conc");

    set_primary(&mut request, 0, 7, 1000);
    assert_eq!(
        select(&workers, &request).worker_id,
        1,
        "0.5 KV, 8/8 conc (strictly over at 0.9)"
    );
}

#[test]
fn threshold_boundaries_pick_the_active_count_that_spills() {
    // max_num_seqs 8; the arriving request is counted, so occupancy is (active + 1) / 8.
    //   threshold 0.8: spills at active >= 6 (7/8 = 0.875)
    //   threshold 1.0: spills at active >= 8 (9/8 = 1.125)
    //   threshold 1.2: spills at active >= 9 (10/8 = 1.25)
    let cases: [(f64, usize, u64); 6] = [
        (0.8, 5, 0),
        (0.8, 6, 1),
        (1.0, 7, 0),
        (1.0, 8, 1),
        (1.2, 8, 0),
        (1.2, 9, 1),
    ];
    let workers = HashMap::from([
        (0, SimWorker::primary_with_seq_capacity(1_000_000, 8)),
        (1, SimWorker::proxy(1000)),
    ]);
    for (threshold, active, expected) in cases {
        let mut request = empty_request(16);
        set_primary(&mut request, 0, active, 0);
        set_proxy(&mut request, 1, 1000, 0);
        assert_eq!(
            select_with(boundary_params(threshold), &workers, &request).worker_id,
            expected,
            "threshold={threshold} active={active}"
        );
    }
}

#[test]
fn fallback_capacity_blocks_used_when_worker_reports_none() {
    let workers = HashMap::from([(0, unknown_primary()), (1, SimWorker::proxy(1000))]);
    let mut params = params();
    params.primary_capacity_blocks = Some(1000.0);
    params.primary_max_requests = None;

    let mut request = empty_request(16);
    set_primary(&mut request, 0, 0, 950);
    set_proxy(&mut request, 1, 1000, 0);
    assert_eq!(
        select_with(params, &workers, &request).worker_id,
        1,
        "the configured fallback capacity makes 950 blocks spill"
    );
}

#[test]
fn fallback_max_requests_used_when_worker_reports_none() {
    let workers = HashMap::from([(0, unknown_primary()), (1, SimWorker::proxy(1000))]);
    let mut params = params();
    params.primary_capacity_blocks = None;
    params.primary_max_requests = Some(8.0);

    let mut request = empty_request(16);
    set_primary(&mut request, 0, 8, 0);
    set_proxy(&mut request, 1, 1000, 0);
    assert_eq!(
        select_with(params, &workers, &request).worker_id,
        1,
        "the configured fallback concurrency limit makes 9 active requests spill"
    );
}

#[test]
fn reported_zero_capacity_counts_as_unknown() {
    // A worker that reports zero blocks and no sequence limit falls back to the configured
    // capacity, exactly as if it had reported nothing.
    let workers = HashMap::from([
        (
            0,
            SimWorker {
                total_kv_blocks: Some(0),
                max_num_seqs: Some(0),
                ..SimWorker::primary(0)
            },
        ),
        (1, SimWorker::proxy(1000)),
    ]);
    let mut params = params();
    params.primary_capacity_blocks = Some(1000.0);
    params.primary_max_requests = None;

    let mut request = empty_request(16);
    set_primary(&mut request, 0, 0, 950);
    set_proxy(&mut request, 1, 1000, 0);
    assert_eq!(select_with(params, &workers, &request).worker_id, 1);
}

#[test]
fn no_signal_gives_no_failover_penalty() {
    // Nothing reported and nothing configured: the worker has no occupancy signal, so even a
    // huge decode footprint earns no failover penalty and it wins over the proxy.
    let workers = HashMap::from([(0, unknown_primary()), (1, SimWorker::proxy(1000))]);
    let mut params = params();
    params.primary_capacity_blocks = None;
    params.primary_max_requests = None;
    params.pending_weight_blocks = 0.0;

    let mut request = empty_request(16);
    set_primary(&mut request, 0, 0, 5000);
    set_proxy(&mut request, 1, 1000, 0);
    assert_eq!(select_with(params, &workers, &request).worker_id, 0);
}

#[test]
fn preferred_taint_multiplier_scales_the_failover_cost() {
    // A worker type that advertises a taint, so the host can compute a preferred-taint
    // multiplier for it.
    #[derive(Clone)]
    struct TaintedWorker {
        dp_start_rank: u32,
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for TaintedWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            self.dp_start_rank
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1000)
        }
        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    let workers = HashMap::from([
        (
            0u64,
            TaintedWorker {
                dp_start_rank: 0,
                taints: HashSet::from(["spot".to_string()]),
            },
        ),
        (
            1u64,
            TaintedWorker {
                dp_start_rank: 1000,
                taints: HashSet::new(),
            },
        ),
    ]);

    // Primary worker 0 is over threshold, so the failover penalty applies: 300. Proxy x's tier
    // cost is 200. Without the client preference the proxy wins. A preferred taint on the primary
    // worker with weight 1.0 gives multiplier exp(-tanh(1)) ~= 0.467: scaling the whole cost
    // (matching DefaultWorkerScorer::worker_cost) makes the primary cost ~140, so it wins.
    let mut local = params();
    local.failover_penalty_blocks = 300.0;
    local.tiers[0].penalty_blocks = 200.0;
    local.tiers[0].weight_blocks = 0.0;

    let mut request = empty_request(16);
    set_rank(
        &mut request,
        WorkerWithDpRank::new(0, 0),
        signals(0, 950),
        16,
    );
    set_rank(
        &mut request,
        WorkerWithDpRank::new(1, 1000),
        signals(0, 0),
        16,
    );
    request
        .routing_constraints
        .preferred_taints
        .insert("spot".into(), 1.0);

    let policy = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(TierScorer::new(local))],
        Box::new(LowestCostPicker),
    );
    let input = WorkerSelectionInput::configured(&workers, &request, request.eligibility(), 16);
    let selected = policy.select_worker(input).unwrap().worker;
    assert_eq!(
        selected.worker_id, 0,
        "preferred taint must scale the failover cost"
    );
}
