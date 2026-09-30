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
//! The main grid keeps the reference's inputs aligned (block-aligned prompts, a tier-overlap map
//! always present, worker 0 idle) so the comparison is exact; the extra tests deliberately break
//! each of those assumptions: a missing tier-overlap map, a busy prefill pool with a non-zero
//! batch minimum, and non-block-aligned prompts with per-request weight overrides.

use std::collections::HashMap;
use std::sync::Arc;

use dw_spillover_policy::baseline::PickerRng;
use dw_spillover_policy::baseline::baseline_picker;
use dw_spillover_policy::build_policy;
use dw_spillover_policy::params::{ModelParameters, SpilloverParameters, TierParameters};
use dynamo_kv_router::protocols::{
    RoutingConstraints, WorkerAffinityTarget, WorkerConfigLike, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use dynamo_kv_router::{
    DefaultWorkerSelector, KvRouterConfig, RouterConfigOverride, SharedCacheHits, WorkerCandidate,
    WorkerLoadProjection, WorkerScorer, WorkerSelectionContext, WorkerSelectionInput,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector, WorkerType,
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
        primary_capacity_blocks: 1.0e9,
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
                    assert_eq!(
                        actual.logit, expected.logit,
                        "logit: temperature={temperature} prompt={prompt} mode={mode}"
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
fn model_with_parameters_without_active_block_tracking_falls_back_to_default() {
    let params = inert_params();
    let role = WorkerType::Aggregated;
    let config = KvRouterConfig {
        router_temperature: 0.0,
        router_track_active_blocks: false,
        ..Default::default()
    };

    // Without tracking the tier scorer could never see primary occupancy, so the factory must
    // hand back the built-in default policy rather than a policy that silently never spills.
    let policy = build_policy(&config, role, "model-with-params", &params, seeded_rng());
    assert!(<dynamo_kv_router::WorkerSelectionPolicy as WorkerSelector<TestWorker>>::uses_exclusive_affinity_target(&policy));
    assert_eq!(
        <dynamo_kv_router::WorkerSelectionPolicy as WorkerSelector<TestWorker>>::required_worker_inputs(&policy),
        dynamo_kv_router::WorkerInputs::CACHE | dynamo_kv_router::WorkerInputs::LOAD
    );
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

/// Add `base` active prefill tokens to every worker, so the batch minimum is non-zero.
fn bump_active_prefill(request: &mut SchedulingRequest, base: usize) {
    for load in request.worker_loads.values_mut() {
        load.active_prefill_tokens += base;
    }
}

/// A request with no overlap or load, for the hand-built cases below.
fn bare_request(isl_tokens: usize) -> SchedulingRequest {
    SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens,
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
    }
}

fn assert_equivalent(
    workers: &HashMap<u64, TestWorker>,
    request: &SchedulingRequest,
    config: &KvRouterConfig,
    role: WorkerType,
    params: &SpilloverParameters,
    context: &str,
) {
    let reference =
        DefaultWorkerSelector::new_seeded(Some(config.clone()), role.default_selector_label(), 42);
    let policy = build_policy(config, role, "model-with-params", params, seeded_rng());
    for _ in 0..32 {
        let input = selection_input(workers, request);
        let expected = reference.select_worker(input).unwrap();
        let actual = policy.select_worker(input).unwrap();
        assert_eq!(actual.worker, expected.worker, "worker: {context}");
        assert_eq!(actual.logit, expected.logit, "logit: {context}");
        assert_eq!(
            actual.cached_tokens, expected.cached_tokens,
            "cached: {context}"
        );
        assert_eq!(
            actual.potential_decode_blocks, expected.potential_decode_blocks,
            "decode: {context}"
        );
    }
}

#[test]
fn matches_reference_without_a_tier_overlap_map() {
    let params = inert_params();
    let role = WorkerType::Aggregated;

    for temperature in [0.0, 0.7] {
        for prompt in [256, 512, 1024, 2048] {
            for mode in 0..64 {
                let (workers, mut request) = fixture(16, prompt);
                // No per-tier map: the default falls back to the effective overlap. Shared-cache
                // beyond-device is host-materialized from the reported depth, so leave it unset.
                request.overlap.tier_overlap_blocks.device.clear();
                request.overlap.tier_overlap_blocks.host_pinned.clear();
                request.overlap.tier_overlap_blocks.disk.clear();
                request.shared_cache_hits = None;
                if mode % 4 == 3 {
                    request.track_prefill_tokens = false;
                }
                let config = config_for(mode, temperature);
                assert_equivalent(
                    &workers,
                    &request,
                    &config,
                    role,
                    &params,
                    &format!("no-tier temperature={temperature} prompt={prompt} mode={mode}"),
                );
            }
        }
    }
}

#[test]
fn matches_reference_with_a_busy_prefill_pool() {
    let params = inert_params();
    let role = WorkerType::Aggregated;

    for base in [100usize, 500] {
        for temperature in [0.0, 0.7] {
            for prompt in [256, 512, 1024, 2048] {
                for mode in 0..64 {
                    // Only decay paths use the batch minimum.
                    if mode & 2 == 0 {
                        continue;
                    }
                    let (workers, mut request) = fixture(16, prompt);
                    bump_active_prefill(&mut request, base);
                    let config = config_for(mode, temperature);
                    assert_equivalent(
                        &workers,
                        &request,
                        &config,
                        role,
                        &params,
                        &format!(
                            "busy base={base} temperature={temperature} prompt={prompt} mode={mode}"
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn matches_reference_for_non_aligned_prompts_and_weight_overrides() {
    let params = inert_params();
    let role = WorkerType::Aggregated;

    for prompt in [257usize, 511, 1023, 2051] {
        for temperature in [0.0, 0.7] {
            for mode in 0..64 {
                let (workers, mut request) = fixture(16, prompt);
                // Overrides are applied by the host before the context is built.
                if mode & 1 != 0 {
                    request.router_config_override = Some(RouterConfigOverride {
                        overlap_score_credit: Some(0.25),
                        prefill_load_scale: Some(1.7),
                        shared_cache_multiplier: Some(0.9),
                        ..Default::default()
                    });
                }
                if mode & 32 != 0 {
                    request.shared_cache_hits =
                        Some(SharedCacheHits::from_ranges(vec![1..3, 5..12]));
                }
                if mode % 4 == 3 {
                    request.track_prefill_tokens = false;
                }
                let config = config_for(mode, temperature);
                assert_equivalent(
                    &workers,
                    &request,
                    &config,
                    role,
                    &params,
                    &format!("aligned temperature={temperature} prompt={prompt} mode={mode}"),
                );
            }
        }
    }
}

#[test]
fn baseline_picker_honours_session_affinity_target() {
    let params = inert_params();
    let role = WorkerType::Aggregated;
    let config = KvRouterConfig {
        router_temperature: 0.0,
        ..Default::default()
    };

    let workers = HashMap::from([(0, TestWorker), (1, TestWorker)]);
    let mut request = bare_request(256);
    // Worker 0 is the cheapest by cache overlap; the session is bound to worker 1.
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
    request.affinity_target = Some(WorkerAffinityTarget::new(1, Some(0)));

    let policy = build_policy(&config, role, "model-with-params", &params, seeded_rng());
    let selected = policy
        .select_worker(selection_input(&workers, &request))
        .unwrap();
    assert_eq!(selected.worker.worker_id, 1);
}

#[test]
fn baseline_picker_affinity_target_picks_the_cheapest_rank() {
    let params = inert_params();
    let role = WorkerType::Aggregated;
    let config = KvRouterConfig {
        router_temperature: 0.0,
        ..Default::default()
    };

    // One worker with two DP ranks; rank 1 holds the session prefix, rank 0 is cold. The
    // affinity target is worker-only (`dp_rank == None`), so the policy must choose between the
    // worker's ranks by cost rather than returning the first matching row.
    let workers = HashMap::from([(5u64, TestWorker)]);
    let mut request = bare_request(256);
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(WorkerWithDpRank::new(5, 0), 0);
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(WorkerWithDpRank::new(5, 1), 8);
    request.affinity_target = Some(WorkerAffinityTarget::new(5, None));

    let policy = build_policy(&config, role, "model-with-params", &params, seeded_rng());
    let selected = policy
        .select_worker(selection_input(&workers, &request))
        .unwrap();
    assert_eq!(selected.worker, WorkerWithDpRank::new(5, 1));
}

#[test]
fn unseeded_picker_matches_reference_cost_at_temperature_zero() {
    // Production constructs the picker with no rng. The seeded equivalence grid cannot exercise
    // that branch, so run the same fuzzed shapes with `None`: at temperature 0 the default's
    // tie handling only samples among candidates of equal (minimum) cost, so the returned cost
    // must still match the seeded reference exactly.
    let params = inert_params();
    let role = WorkerType::Aggregated;
    let label = role.default_selector_label();

    for prompt in [256, 512, 1024, 2048] {
        for mode in 0..64 {
            let (workers, mut request) = fixture(16, prompt);
            if mode & 32 != 0 {
                request.shared_cache_hits = Some(SharedCacheHits::from_ranges(vec![1..3, 5..12]));
            }
            match mode % 4 {
                1 => request.overlap.tier_overlap_blocks.host_pinned.clear(),
                2 => request.worker_loads.clear(),
                3 => request.track_prefill_tokens = false,
                _ => {}
            }
            let config = config_for(mode, 0.0);
            let reference = DefaultWorkerSelector::new_seeded(Some(config.clone()), label, 42);
            let policy = build_policy(&config, role, "model-with-params", &params, None);

            let expected = reference
                .select_worker(selection_input(&workers, &request))
                .unwrap();
            let actual = policy
                .select_worker(selection_input(&workers, &request))
                .unwrap();
            assert_eq!(
                actual.logit, expected.logit,
                "prompt={prompt} mode={mode}: unseeded pick must still land on the minimum cost"
            );
        }
    }
}

#[test]
fn unseeded_softmax_returns_the_sampled_candidate() {
    struct DominantWorkerScorer {
        cheap_worker_id: u64,
    }

    impl WorkerScorer for DominantWorkerScorer {
        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: &WorkerCandidate,
        ) -> Result<f64, WorkerSelectionPolicyError> {
            // One candidate dominates the tiny-temperature softmax; the rest are effectively
            // impossible, so every draw must return the cheap worker.
            Ok(if candidate.worker().worker_id == self.cheap_worker_id {
                0.0
            } else {
                1.0e12
            })
        }
    }

    let workers: HashMap<u64, TestWorker> = (0..8).map(|id| (id, TestWorker)).collect();
    let request = bare_request(256);
    let config = KvRouterConfig {
        router_temperature: 1.0e-3,
        ..Default::default()
    };
    let policy = WorkerSelectionPolicy::new(
        config.clone(),
        "test",
        vec![Box::new(DominantWorkerScorer { cheap_worker_id: 7 })],
        baseline_picker(&config, None),
    );

    // Candidate row order comes from a randomized HashMap; a sort/row index confusion only
    // survives when the cheap worker happens to sit at its sorted position, so repeat.
    for _ in 0..256 {
        let selected = policy
            .select_worker(selection_input(&workers, &request))
            .unwrap();
        assert_eq!(selected.worker.worker_id, 7);
    }
}

#[test]
fn invalid_parameters_fall_back_to_default_policy() {
    let mut params = inert_params();
    params
        .models
        .get_mut("model-with-params")
        .unwrap()
        .primary_capacity_blocks = 0.0;
    let config = KvRouterConfig {
        router_temperature: 0.0,
        ..Default::default()
    };

    // `build_policy` is a public seam that bypasses `provider`; it must not build a tier policy
    // from parameters that can only produce NaN costs.
    let policy = build_policy(
        &config,
        WorkerType::Aggregated,
        "model-with-params",
        &params,
        seeded_rng(),
    );
    assert!(
        <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::uses_exclusive_affinity_target(
            &policy
        )
    );
}
