// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker selection benchmarks: the default logit selector against
//! reported-load selection, over the same pool and request.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench selector`

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{
    OverlapSignals, ReportedRankLoad, ScheduleMode, SchedulingRequest,
};
use dynamo_kv_router::selector::{DefaultWorkerSelector, WorkerSelector};
use dynamo_kv_router::sequences::WorkerLoadProjection;
use rustc_hash::FxHashMap;

const BLOCK_SIZE: u32 = 64;
const ISL_TOKENS: usize = 37_000;

#[derive(Debug, Clone)]
struct BenchWorkerConfig {
    data_parallel_size: u32,
}

impl WorkerConfigLike for BenchWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(2_480_000)
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(38_822)
    }
}

fn pool(workers: u64, dp_size: u32) -> (HashMap<u64, BenchWorkerConfig>, Vec<WorkerWithDpRank>) {
    let mut configs = HashMap::new();
    let mut ranks = Vec::new();
    for worker_id in 0..workers {
        configs.insert(
            worker_id,
            BenchWorkerConfig {
                data_parallel_size: dp_size,
            },
        );
        for rank in 0..dp_size {
            ranks.push(WorkerWithDpRank::new(worker_id, rank));
        }
    }
    (configs, ranks)
}

fn request(ranks: &[WorkerWithDpRank], with_reports: bool) -> SchedulingRequest {
    let mut overlap = OverlapSignals {
        tier_overlap_blocks: Default::default(),
        effective_overlap_blocks: HashMap::default(),
        effective_cached_tokens: HashMap::default(),
    };
    let mut worker_loads = FxHashMap::default();
    for (i, rank) in ranks.iter().enumerate() {
        // Every fourth rank holds part of the prompt; one holds nearly all of it.
        let overlap_blocks = match i % 4 {
            0 if i == 8 => 560,
            0 => 40,
            _ => 0,
        };
        if overlap_blocks > 0 {
            overlap
                .tier_overlap_blocks
                .device
                .insert(*rank, overlap_blocks);
            overlap
                .effective_overlap_blocks
                .insert(*rank, overlap_blocks as f64);
            overlap
                .effective_cached_tokens
                .insert(*rank, overlap_blocks * BLOCK_SIZE as usize);
        }
        let waiting = (i as u64 * 7) % 13;
        worker_loads.insert(
            *rank,
            WorkerLoadProjection {
                active_prefill_tokens: (i * 3_000) % 90_000,
                active_decode_blocks: 20_000 + (i * 331) % 9_000,
                active_requests: 40 + (i % 17),
                additional_active_blocks: 580,
                reported: with_reports.then_some(ReportedRankLoad {
                    waiting_requests: waiting,
                    kv_used_blocks: Some(24_000 + (i as u64 * 977) % 12_000),
                    kv_total_blocks: Some(38_822),
                    report_revision: Some(i as u64),
                }),
                dispatched_tokens_since_report: if with_reports {
                    (i * 1_500) % 20_000
                } else {
                    0
                },
            },
        );
    }
    SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: ISL_TOKENS,
        lora_name: None,
        expected_output_tokens: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        excluded_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_id: None,
        overlap,
        shared_cache_hits: None,
        worker_loads,
        resp_tx: None,
    }
}

fn bench_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_worker");
    for &(workers, dp_size) in &[(6u64, 8u32), (24, 8), (96, 8)] {
        let (configs, ranks) = pool(workers, dp_size);
        let plain = request(&ranks, false);
        let reported = request(&ranks, true);

        let default_selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "bench");
        group.bench_with_input(
            BenchmarkId::new("default", ranks.len()),
            &plain,
            |b, req| {
                b.iter(|| {
                    default_selector
                        .select_worker(&configs, black_box(req), req.eligibility(), BLOCK_SIZE)
                        .unwrap()
                })
            },
        );

        let reported_selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_reported_load: true,
                ..Default::default()
            }),
            "bench",
        );
        group.bench_with_input(
            BenchmarkId::new("reported_load", ranks.len()),
            &reported,
            |b, req| {
                b.iter(|| {
                    reported_selector
                        .select_worker(&configs, black_box(req), req.eligibility(), BLOCK_SIZE)
                        .unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
