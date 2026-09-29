// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Our scorer: failover and tier preference. Stacked after the baseline scorer.
//!
//! Per candidate (lower is better):
//! - proxy worker (DP rank inside a tier): `tier.penalty_blocks + tier.weight_blocks
//!   + pending_weight_blocks * active_requests`
//! - hosted worker: `(decode_cost_blocks / hosted_capacity_blocks >= occupancy_threshold
//!   ? failover_penalty_blocks : 0) + pending_weight_blocks * active_requests`
//! - when the host supplied no load observation, the load inputs are all zero (idle): zero
//!   active requests, zero decode blocks.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

use crate::params::ModelParameters;

pub struct TierScorer {
    params: ModelParameters,
}

impl TierScorer {
    pub fn new(params: ModelParameters) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &ModelParameters {
        &self.params
    }
}

impl WorkerScorer for TierScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let params = &self.params;
        let worker = candidate.worker();
        // Missing load is an observation gap, not a busy worker: score it as idle.
        let (active_requests, decode_blocks) = match candidate.load() {
            Some(load) => (load.active_requests(), load.decode_cost_blocks()),
            None => (0, 0.0),
        };
        let cost = if let Some(tier) = params.tier_for_rank(worker.dp_rank) {
            tier.penalty_blocks
                + tier.weight_blocks
                + params.pending_weight_blocks * active_requests as f64
        } else {
            let occupancy = decode_blocks / params.hosted_capacity_blocks;
            let failover_penalty = if occupancy >= params.occupancy_threshold {
                params.failover_penalty_blocks
            } else {
                0.0
            };
            failover_penalty + params.pending_weight_blocks * active_requests as f64
        };
        if !decode_blocks.is_finite() || !cost.is_finite() {
            return Err(WorkerSelectionPolicyError::failed(format!(
                "spillover cost for worker {} dp rank {} is not finite",
                worker.worker_id, worker.dp_rank
            )));
        }
        Ok(cost)
    }
}
