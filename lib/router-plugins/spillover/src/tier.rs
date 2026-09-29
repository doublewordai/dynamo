//! Our scorer: failover and tier preference. Stacked after the baseline scorer.
//!
//! Per candidate (lower is better):
//! - proxy worker (DP rank inside a tier): `tier.penalty_blocks + tier.weight_blocks
//!   + pending_weight_blocks * active_requests`
//! - hosted worker: `(decode_cost_blocks / hosted_capacity_blocks >= occupancy_threshold
//!   ? failover_penalty_blocks : 0) + pending_weight_blocks * active_requests`
//! - when the host supplied no load observation (`is_available() == false`), treat the worker
//!   as idle: zero active requests, zero decode blocks.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
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
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let _ = context;
        let params = &self.params;
        for (row, candidate) in candidates.iter().enumerate() {
            let worker = candidate.worker();
            // Missing load is an observation gap, not a busy worker: score it as idle.
            let (active_requests, decode_blocks) = match candidate.load() {
                Some(load) if load.is_available() => {
                    (load.active_requests(), load.decode_cost_blocks())
                }
                _ => (0, 0.0),
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
            costs[row] = cost;
        }
        Ok(())
    }
}
