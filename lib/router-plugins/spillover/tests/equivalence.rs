// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copied from ai-dynamo/dynamo@494d6e24 lib/router-plugins/builtin/tests/default_policy.rs (and support/mod.rs)

//! Exhaustive equivalence check for models without spillover parameters.
//!
//! The baseline module is a copy of Dynamo's private default policy. This drives the public
//! `build_policy` factory and the reference `dynamo_kv_router::DefaultWorkerSelector` with the
//! same seed over a fuzzed grid of cache and load shapes, asserting the same worker, cached
//! tokens and potential decode blocks every time.

use std::collections::HashMap;
use std::sync::Arc;

use dw_spillover_policy::{SpilloverParameters, build_policy};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use dynamo_kv_router::{
    KvRouterConfig, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector, WorkerType,
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
            request
                .overlap
                .effective_cached_tokens
                .insert(w, overlap * 16 + 8);
            request.worker_loads.insert(
                w,
                WorkerLoadProjection {
                    active_requests: id as usize % 5,
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
    block_size: u32,
) -> WorkerSelectionInput<'a, TestWorker> {
    WorkerSelectionInput::configured(workers, request, request.eligibility(), block_size)
}

#[test]
fn seeded_selection_matches_reference_across_cache_and_load_shapes() {
    // WorkerType::Aggregated maps to the "decode" pool label, matching the reference's worker_type.
    let label = WorkerType::Aggregated.default_selector_label();
    for temperature in [0.0, 0.7] {
        for prompt in [1, 17, 127, 2048] {
            for mode in 0..32 {
                let (workers, mut request) = fixture(16, prompt);
                let config = KvRouterConfig {
                    router_temperature: temperature,
                    overlap_score_credit_decay: 0.6,
                    host_cache_hit_weight: 0.25,
                    disk_cache_hit_weight: 0.1,
                    decode_active_request_weight: if mode & 8 == 0 { 0.0 } else { 0.7 },
                    shared_cache_multiplier: if mode & 16 == 0 { 0.0 } else { 0.6 },
                    ..Default::default()
                };
                if mode % 8 >= 4 {
                    request.shared_cache_hits =
                        Some(dynamo_kv_router::SharedCacheHits::from_ranges(vec![
                            1..3,
                            5..12,
                        ]));
                }
                match mode % 4 {
                    1 => request.overlap.tier_overlap_blocks = Default::default(),
                    2 => request.worker_loads.clear(),
                    3 => request.track_prefill_tokens = false,
                    _ => {}
                }
                let reference = dynamo_kv_router::DefaultWorkerSelector::new_seeded(
                    Some(config.clone()),
                    label,
                    42,
                );
                let policy = build_policy(
                    &config,
                    WorkerType::Aggregated,
                    "model-without-params",
                    &SpilloverParameters::default(),
                    Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(42)))),
                );
                for _ in 0..64 {
                    let input = selection_input(&workers, &request, 16);
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
