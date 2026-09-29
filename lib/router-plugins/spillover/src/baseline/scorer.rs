// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-candidate port of `DefaultWorkerScorer`'s cost formula.
//!
//! Mirrors `lib/kv-router/src/scheduling/selector/default.rs::worker_logit` using public plugin
//! inputs only. See the module docs in `baseline/mod.rs` for the inputs that are not visible at
//! this plugin API revision.

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

pub(super) struct BaselineScorer {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    host_cache_hit_weight: f64,
    disk_cache_hit_weight: f64,
    shared_cache_multiplier: f64,
    decode_active_request_weight: f64,
    prefill_load_scale: f64,
    worker_label: &'static str,
}

impl BaselineScorer {
    pub(super) fn new(config: &KvRouterConfig, worker_label: &'static str) -> Self {
        Self {
            overlap_score_credit: config.overlap_score_credit,
            overlap_score_credit_decay: config.overlap_score_credit_decay,
            host_cache_hit_weight: config.host_cache_hit_weight,
            disk_cache_hit_weight: config.disk_cache_hit_weight,
            shared_cache_multiplier: config.shared_cache_multiplier,
            decode_active_request_weight: config.decode_active_request_weight,
            prefill_load_scale: config.prefill_load_scale,
            worker_label,
        }
    }

    fn score_candidate(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let track_prefill = context.tracks_prefill_tokens();
        let block_size = context.block_size() as f64;
        let request_blocks = context.request_blocks() as f64;

        let (device_overlap, host_overlap, disk_overlap, shared_beyond) =
            candidate.cache().map_or((0.0, 0.0, 0.0, 0.0), |cache| {
                (
                    cache.device_overlap_blocks(),
                    cache.host_overlap_blocks(),
                    cache.disk_overlap_blocks(),
                    cache.shared_beyond_device_blocks() as f64,
                )
            });

        // `min_active_prefill_tokens` is batch-wide and unavailable here; see module docs.
        let overlap_credit_decay = if track_prefill && self.overlap_score_credit_decay > 0.0 {
            let excess_active_prefill_blocks = load.active_prefill_tokens() as f64 / block_size;
            let normalized_prefill_load = excess_active_prefill_blocks / request_blocks;
            1.0 / (1.0 + self.overlap_score_credit_decay * normalized_prefill_load)
        } else {
            1.0
        };
        let effective_overlap_score_credit = self.overlap_score_credit * overlap_credit_decay;
        let overlap_credit_blocks = effective_overlap_score_credit * device_overlap
            + self.host_cache_hit_weight * host_overlap
            + self.disk_cache_hit_weight * disk_overlap
            + self.shared_cache_multiplier * shared_beyond;
        let decode_cost_blocks = load.decode_cost_blocks();
        let active_request_cost_blocks =
            self.decode_active_request_weight * load.active_requests() as f64;

        let logit =
            if self.worker_label == "decode" && !track_prefill && self.overlap_score_credit > 0.0 {
                // Decode routers normally force overlap_score_credit to zero through a per-request
                // override; when cache credit survives, prefer cache-hot decode workers.
                (decode_cost_blocks - overlap_credit_blocks).max(0.0) + active_request_cost_blocks
            } else {
                // The default derives raw prefill from isl and cached tokens; reconstruct it from the
                // request's block count. Exact for block-aligned prompts with cache <= prompt.
                let raw_prefill_blocks = if track_prefill {
                    let prompt_tokens = context.request_blocks() * context.block_size() as u64;
                    (load.active_prefill_tokens() as u64 + prompt_tokens) as f64 / block_size
                } else {
                    0.0
                };
                let adjusted_prefill_blocks = (raw_prefill_blocks - overlap_credit_blocks).max(0.0);
                let prefill_cost_blocks = self.prefill_load_scale * adjusted_prefill_blocks;
                prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks
            };

        // Matches DefaultWorkerScorer::worker_cost's preferred-taint multiplier.
        Ok(logit * candidate.preferred_taint_multiplier().unwrap_or(1.0))
    }
}

impl WorkerScorer for BaselineScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        self.score_candidate(context, candidate)
    }
}
