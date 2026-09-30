// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Our scorer: failover and tier preference. Stacked after the baseline scorer.
//!
//! Per candidate (lower is better):
//! - proxy worker (DP rank inside a tier): `tier.penalty_blocks + tier.weight_blocks
//!   + pending_weight_blocks * active_requests`
//! - primary worker: `(decode_cost_blocks / primary_capacity_blocks >= occupancy_threshold
//!   ? failover_penalty_blocks : 0) + pending_weight_blocks * active_requests`
//! - when the host supplied no load observation, the load inputs are all zero (idle): zero
//!   active requests, zero decode blocks.
//!
//! `decode_cost_blocks` is the host's projected decode footprint after admitting this request
//! (current decode blocks plus the request's own uncached prompt blocks), not the worker's
//! current occupancy. A large cold prompt can therefore push an otherwise idle primary worker over
//! the threshold; that matches the default selector's use of the same projected quantity.

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
        // The client's preferred-taint multiplier scales the whole worker cost in the default
        // selector, so this scorer must read it to scale its own contribution by the same factor.
        WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let params = &self.params;
        let worker = candidate.worker();
        // `required_worker_inputs` asks for LOAD, so the host materializes a row even when the
        // worker has no observation (a zero projection). This arm is defensive: score a genuine
        // gap as idle, not busy. `tests/tier.rs::missing_load_observation_is_treated_as_idle`
        // exercises the host-materialized zero path.
        let (active_requests, decode_blocks) = match candidate.load() {
            Some(load) => (load.active_requests(), load.decode_cost_blocks()),
            None => (0, 0.0),
        };
        let (tier_name, cost) = if let Some(tier) = params.tier_for_rank(worker.dp_rank) {
            (
                Some(tier.name.as_str()),
                tier.penalty_blocks
                    + tier.weight_blocks
                    + params.pending_weight_blocks * active_requests as f64,
            )
        } else {
            let occupancy = decode_blocks / params.primary_capacity_blocks;
            let failover_penalty = if occupancy >= params.occupancy_threshold {
                params.failover_penalty_blocks
            } else {
                0.0
            };
            (
                None,
                failover_penalty + params.pending_weight_blocks * active_requests as f64,
            )
        };
        // Match `DefaultWorkerScorer::worker_cost`: the preferred-taint multiplier scales the
        // whole cost, not only the baseline term. The baseline scorer already multiplied its own
        // logit, so applying it here too keeps `multiplier * (baseline + spillover)`.
        let cost = cost * candidate.preferred_taint_multiplier().unwrap_or(1.0);
        if !decode_blocks.is_finite() || !cost.is_finite() {
            return Err(WorkerSelectionPolicyError::failed(format!(
                "spillover cost for worker {} dp rank {} is not finite",
                worker.worker_id, worker.dp_rank
            )));
        }
        tracing::debug!(
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            tier = tier_name.unwrap_or("primary"),
            active_requests,
            decode_blocks,
            cost,
            "dw-spillover candidate cost"
        );
        Ok(cost)
    }
}
