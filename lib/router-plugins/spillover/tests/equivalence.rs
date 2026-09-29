// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Equivalence between our baseline scorer/picker and Dynamo's default selector.
//!
//! The baseline module re-implements the fork's private `DefaultWorkerScorer` and
//! `DefaultWorkerPicker`. This drives the public `build_policy` factory and the reference
//! `dynamo_kv_router::DefaultWorkerSelector::new_seeded` with the same seed over a fuzzed grid
//! of cache and load shapes, asserting the same worker every time.
//!
//! The parameters used here name a model whose proxy tiers match no worker and whose failover
//! penalty is zero, so our `TierScorer` contributes exactly zero and only the baseline decides.
//!
//! Grid constraints that keep the comparison bit-exact, matching the gaps documented in
//! `baseline/mod.rs`:
//! - prompts are block-aligned and at least as large as every worker's cached-token count, so
//!   the reconstructed raw prefill equals the default's;
//! - a tier-overlap map is always present, so both sides use the reported device overlap;
//! - worker 0 has zero active prefill, so the default's batch-wide minimum is 0.

use std::collections::HashMap;
use std::sync::Arc;

use dw_spillover_policy::baseline::PickerRng;
use dw_spillover_policy::build_policy;
use dw_spillover_policy::params::{ModelParameters, SpilloverParameters, TierParameters};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use dynamo_kv_router::{
    DefaultWorkerSelector, KvRouterConfig, SharedCacheHits, WorkerLoadProjection,
    WorkerSelectionInput, WorkerSelector, WorkerType,
};
use parking_lot::Mutex;

#[derive(Clone, Copy)]
struct TestWorker;

impl WorkerConfigLike for TestWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        2
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(16384)
    }
}

/// A model whose tier range excludes every worker's DP rank, with a zero failover penalty and
/// zero pending weight, so `TierScorer` adds nothing.
fn inert_params() -> SpilloverParameters {
    let model = ModelParameters {
        occupancy_threshold: 1.0,
        hosted_capacity_blocks: 1.0e9,
        failover_penalty_blocks: 0.0,
        pending_weight_blocks: 0.0,
        tiers: vec![TierParameters {
            name: "never-matches".into(),
            dp_ranks: [1000, 1999],
            penalty_blocks: 1234.0,
            weight_blocks: 5678.0,
        }],
    };
    let mut params = SpilloverParameters::default();
    params.models.insert("model-with-params".into(), model);
    params
}

const BLOCK_SIZE: u32 = 16;

fn fixture(count: usize, prompt_tokens: usize) -> (HashMap<u64, TestWorker>, SchedulingRequest) {
    let mut request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: prompt_tokens,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    };
    let workers = (0..count as u64).map(|id| (id, TestWorker)).collect();
    for id in 0..count as u64 {
        for dp_rank in 0..2 {
            let w = WorkerWithDpRank::new(id, dp_rank);
            let overlap = (id as usize * 7 + dp_rank as usize) % 9;
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(w, overlap);
            request.overlap.tier_overlap_blocks.host_pinned.insert(w, 2);
            request.overlap.tier_overlap_blocks.disk.insert(w, 1);
            request
                .overlap
                .effective_overlap_blocks
                .insert(w, overlap as f64 + 0.5);
            // Kept <= every prompt so the default's cached-token path stays under isl.
            request
                .overlap
                .effective_cached_tokens
                .insert(w, overlap * BLOCK_SIZE as usize + 8);
            request.worker_loads.insert(
                w,
                WorkerLoadProjection {
                    active_requests: id as usize % 5,
                    // Worker 0 (and worker 7, 14) contribute a zero active-prefill minimum.
                    active_prefill_tokens: (id as usize % 7) * 19,
                    active_decode_blocks: id as usize % 11,
                    additional_active_blocks: 3,
                },
            );
        }
    }
    (workers, request)
}

fn selection_input<'a>(
    workers: &'a HashMap<u64, TestWorker>,
    request: &'a SchedulingRequest,
) -> WorkerSelectionInput<'a, TestWorker> {
    WorkerSelectionInput::configured(workers, request, request.eligibility(), BLOCK_SIZE)
}

fn config_for(mode: usize, temperature: f64) -> KvRouterConfig {
    KvRouterConfig {
        router_temperature: temperature,
        overlap_score_credit: if mode & 1 == 0 { 1.0 } else { 0.5 },
        overlap_score_credit_decay: if mode & 2 == 0 { 0.0 } else { 0.6 },
        host_cache_hit_weight: if mode & 4 == 0 { 0.0 } else { 0.25 },
        disk_cache_hit_weight: if mode & 8 == 0 { 0.0 } else { 0.1 },
        decode_active_request_weight: if mode & 16 == 0 { 0.0 } else { 0.7 },
        shared_cache_multiplier: if mode & 32 == 0 { 0.0 } else { 0.6 },
        ..Default::default()
    }
}

fn seeded_rng() -> PickerRng {
    Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(42))))
}

#[test]
fn model_with_parameters_matches_reference_across_cache_and_load_shapes() {
    let params = inert_params();
    // `WorkerType::Aggregated` maps to the "decode" pool label, like the reference.
    let role = WorkerType::Aggregated;
    let label = role.default_selector_label();

    for temperature in [0.0, 0.7] {
        for prompt in [256, 512, 1024, 2048] {
            for mode in 0..64 {
                let (workers, mut request) = fixture(16, prompt);
                let config = config_for(mode, temperature);

                // Shared cache hits exercise the shared-cache credit path.
                if mode & 32 != 0 {
                    request.shared_cache_hits =
                        Some(SharedCacheHits::from_ranges(vec![1..3, 5..12]));
                }
                // Vary a few load/cache shapes without dropping the tier-overlap map.
                match mode % 4 {
                    1 => request.overlap.tier_overlap_blocks.host_pinned.clear(),
                    2 => request.worker_loads.clear(),
                    3 => request.track_prefill_tokens = false,
                    _ => {}
                }

                let reference = DefaultWorkerSelector::new_seeded(Some(config.clone()), label, 42);
                let policy =
                    build_policy(&config, role, "model-with-params", &params, seeded_rng());
                for _ in 0..32 {
                    let input = selection_input(&workers, &request);
                    let expected = reference.select_worker(input).unwrap();
                    let actual = policy.select_worker(input).unwrap();
                    assert_eq!(
                        actual.worker, expected.worker,
                        "temperature={temperature} prompt={prompt} mode={mode}"
                    );
                    assert_eq!(actual.cached_tokens, expected.cached_tokens);
                    assert_eq!(
                        actual.potential_decode_blocks,
                        expected.potential_decode_blocks
                    );
                }
            }
        }
    }
}

#[test]
fn model_without_parameters_gets_the_default_policy() {
    let params = SpilloverParameters::default();
    let role = WorkerType::Aggregated;
    let label = role.default_selector_label();

    // `WorkerSelectionPolicy::default` is the only policy state that owns exclusive affinity
    // and asks for exactly the built-in cache+load inputs. That is what the factory must return
    // for a model with no parameters.
    let config = KvRouterConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    // Production calls build_policy without an rng; that is when the built-in default applies.
    let policy = build_policy(&config, role, "model-without-params", &params, None);
    assert!(<dynamo_kv_router::WorkerSelectionPolicy as WorkerSelector<TestWorker>>::uses_exclusive_affinity_target(&policy));
    assert_eq!(
        <dynamo_kv_router::WorkerSelectionPolicy as WorkerSelector<TestWorker>>::required_worker_inputs(&policy),
        dynamo_kv_router::WorkerInputs::CACHE | dynamo_kv_router::WorkerInputs::LOAD
    );

    // A controlled case with a unique cheapest worker: the built-in default selector and the
    // factory's policy must choose the same one is deterministic at temperature 0.0.
    let workers = HashMap::from([(0, TestWorker), (1, TestWorker)]);
    let mut request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: 256,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    };
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(WorkerWithDpRank::new(0, 0), 8);
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(WorkerWithDpRank::new(1, 0), 0);

    let reference = DefaultWorkerSelector::new_seeded(Some(config.clone()), label, 42);
    let expected = reference
        .select_worker(selection_input(&workers, &request))
        .unwrap();
    let actual = policy
        .select_worker(selection_input(&workers, &request))
        .unwrap();
    assert_eq!(actual.worker, expected.worker);
    assert_eq!(actual.worker.worker_id, 0);
}
