// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers that build `SchedulingRequest`s and `WorkerSelectionInput`s directly, the way
//! upstream's `lib/router-plugins/builtin/tests/support` does, so policies can be driven
//! without a running frontend.

use std::collections::HashMap;

use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use dynamo_kv_router::{WorkerLoadProjection, WorkerSelectionInput};

/// One registered worker as the router sees its runtime config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimWorker {
    pub dp_start_rank: u32,
    pub dp_size: u32,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
}

impl SimWorker {
    /// A hosted worker with one rank starting at 0.
    pub fn hosted(total_kv_blocks: u64) -> Self {
        Self {
            dp_start_rank: 0,
            dp_size: 1,
            total_kv_blocks: Some(total_kv_blocks),
            max_num_batched_tokens: None,
        }
    }

    /// A proxy worker registered at a reserved DP rank.
    pub fn proxy(dp_rank: u32) -> Self {
        Self {
            dp_start_rank: dp_rank,
            dp_size: 1,
            total_kv_blocks: None,
            max_num_batched_tokens: None,
        }
    }
}

impl WorkerConfigLike for SimWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        self.dp_start_rank
    }
    fn data_parallel_size(&self) -> u32 {
        self.dp_size
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }
}

/// Per-rank signals for one selection.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RankSignals {
    /// Device-resident prefix overlap, in blocks.
    pub device_overlap_blocks: usize,
    pub active_requests: usize,
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
    /// The arriving request's own uncached blocks, the way the router's prompt registry
    /// projects it (`query_len - overlap_depth`). Counted in the selector's decode cost.
    pub additional_active_blocks: usize,
}

/// A request with no overlap and no load, ready for `set_rank`.
pub fn empty_request(prompt_tokens: usize) -> SchedulingRequest {
    SchedulingRequest {
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
    }
}

/// Record one rank's overlap and load on `request`.
pub fn set_rank(
    request: &mut SchedulingRequest,
    worker: WorkerWithDpRank,
    signals: RankSignals,
    block_size: u32,
) {
    let overlap = signals.device_overlap_blocks;
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(worker, overlap);
    request
        .overlap
        .effective_overlap_blocks
        .insert(worker, overlap as f64);
    request
        .overlap
        .effective_cached_tokens
        .insert(worker, overlap * block_size as usize);
    request.worker_loads.insert(
        worker,
        WorkerLoadProjection {
            active_requests: signals.active_requests,
            active_prefill_tokens: signals.active_prefill_tokens,
            active_decode_blocks: signals.active_decode_blocks,
            additional_active_blocks: signals.additional_active_blocks,
        },
    );
}

/// Build the selection input the router hands a selector.
pub fn selection_input<'a>(
    workers: &'a HashMap<u64, SimWorker>,
    request: &'a SchedulingRequest,
    block_size: u32,
) -> WorkerSelectionInput<'a, SimWorker> {
    WorkerSelectionInput::configured(workers, request, request.eligibility(), block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The router always projects the arriving request's own uncached blocks into
    /// `additional_active_blocks`; `set_rank` must be able to express that term or the
    /// policy's decode-cost signal silently loses it.
    #[test]
    fn set_rank_records_the_arriving_requests_uncached_blocks() {
        let worker = WorkerWithDpRank::new(0, 0);
        let mut request = empty_request(64);
        set_rank(
            &mut request,
            worker,
            RankSignals {
                device_overlap_blocks: 2,
                additional_active_blocks: 4,
                ..RankSignals::default()
            },
            16,
        );
        let load = request.worker_loads.get(&worker).expect("rank load");
        assert_eq!(load.additional_active_blocks, 4);
        assert_eq!(load.potential_decode_blocks(), 4);
    }
}
