// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Our scorer: failover and tier preference. Stacked after the baseline scorer.
//!
//! Per candidate (lower is better):
//! - proxy worker (DP rank inside a tier): `tier.penalty_blocks + tier.weight_blocks
//!   + pending_weight_blocks * active_requests`
//! - primary worker: `(occupancy > occupancy_threshold ? failover_penalty_blocks : 0)
//!   + pending_weight_blocks * active_requests`
//! - when the host supplied no load observation, the load inputs are all zero (idle): zero
//!   active requests, zero decode blocks.
//!
//! Occupancy for a primary worker is the larger of the signals it can support, both meaning
//! "occupancy if this request lands here":
//! - KV: `decode_cost_blocks / total_kv_blocks`, falling back to `primary_capacity_blocks`
//!   when the worker advertises none.
//! - concurrency: projected active requests over `max_num_seqs`, falling back to
//!   `primary_max_requests` when the worker advertises none. The arriving request is counted,
//!   because `active_requests` is the worker's current count while `decode_cost_blocks` is
//!   already post-admission.
//!
//! A reported capacity of zero counts as unknown, and a worker with neither a reported value nor
//! a fallback gets no failover penalty (and a one-time warning). The penalty applies strictly
//! above the threshold, so `1.0` means "the engine is exactly full", `0.8` spills before the last
//! 20%, and `1.2` accepts 20% queueing before spilling.
//!
//! `decode_cost_blocks` is the host's projected decode footprint after admitting this request
//! (current decode blocks plus the request's own uncached prompt blocks), not the worker's
//! current occupancy. A large cold prompt can therefore push an otherwise idle primary worker over
//! the threshold; that matches the default selector's use of the same projected quantity.

use std::collections::HashSet;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerCapacity, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

use crate::params::ModelParameters;

pub struct TierScorer {
    params: ModelParameters,
    /// Primary workers already warned about having no usable occupancy signal. Keeps the warning
    /// to once per worker, not once per request, without allocating on the steady-state path.
    warned_no_signal: HashSet<u64>,
}

impl TierScorer {
    pub fn new(params: ModelParameters) -> Self {
        Self {
            params,
            warned_no_signal: HashSet::new(),
        }
    }

    pub fn params(&self) -> &ModelParameters {
        &self.params
    }

    /// Occupancy of a primary worker for this request, or `None` when it has no usable signal.
    fn primary_occupancy(
        &mut self,
        capacity: WorkerCapacity,
        decode_blocks: f64,
        active_requests: usize,
        worker_id: u64,
    ) -> Option<f64> {
        let kv_capacity = match capacity.total_kv_blocks() {
            Some(blocks) if blocks > 0 => Some(blocks as f64),
            _ => self.params.primary_capacity_blocks,
        };
        let concurrency_capacity = match capacity.max_num_seqs() {
            Some(seqs) if seqs > 0 => Some(seqs as f64),
            _ => self.params.primary_max_requests,
        };
        let kv = kv_capacity.map(|capacity| decode_blocks / capacity);
        // Project the arriving request into the concurrency signal. The host materializes
        // `active_requests` from the worker's current count, while `decode_cost_blocks` already
        // includes this request; adding one makes both signals mean "occupancy if it lands here".
        let concurrency =
            concurrency_capacity.map(|capacity| (active_requests as f64 + 1.0) / capacity);
        match (kv, concurrency) {
            (Some(kv), Some(concurrency)) => Some(kv.max(concurrency)),
            (Some(kv), None) => Some(kv),
            (None, Some(concurrency)) => Some(concurrency),
            (None, None) => {
                if self.warned_no_signal.insert(worker_id) {
                    tracing::warn!(
                        worker_id,
                        "dw-spillover primary worker advertises neither total_kv_blocks nor \
                         max_num_seqs and has no fallback configured, so it never gets a \
                         failover penalty. Set primary_capacity_blocks or primary_max_requests, \
                         or make the worker advertise its capacity."
                    );
                }
                None
            }
        }
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
        let worker = candidate.worker();
        // `required_worker_inputs` asks for LOAD, so the host materializes a row even when the
        // worker has no observation (a zero projection). This arm is defensive: score a genuine
        // gap as idle, not busy. `tests/tier.rs::missing_load_observation_is_treated_as_idle`
        // exercises the host-materialized zero path.
        let (active_requests, decode_blocks) = match candidate.load() {
            Some(load) => (load.active_requests(), load.decode_cost_blocks()),
            None => (0, 0.0),
        };
        let (tier_name, cost) = if let Some(tier) = self.params.tier_for_rank(worker.dp_rank) {
            (
                Some(tier.name.as_str()),
                tier.penalty_blocks
                    + tier.weight_blocks
                    + self.params.pending_weight_blocks * active_requests as f64,
            )
        } else {
            let occupancy = self.primary_occupancy(
                candidate.capacity(),
                decode_blocks,
                active_requests,
                worker.worker_id,
            );
            let failover_penalty = match occupancy {
                Some(occupancy) if occupancy > self.params.occupancy_threshold => {
                    self.params.failover_penalty_blocks
                }
                _ => 0.0,
            };
            (
                None,
                failover_penalty + self.params.pending_weight_blocks * active_requests as f64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ModelParameters;
    use tracing_test::traced_test;

    fn scorer(
        primary_capacity_blocks: Option<f64>,
        primary_max_requests: Option<f64>,
    ) -> TierScorer {
        TierScorer::new(ModelParameters {
            occupancy_threshold: 0.9,
            primary_capacity_blocks,
            primary_max_requests,
            failover_penalty_blocks: 500.0,
            pending_weight_blocks: 0.0,
            tiers: Vec::new(),
        })
    }

    #[test]
    fn no_signal_is_none_and_marks_the_worker() {
        let mut scorer = scorer(None, None);
        assert_eq!(
            scorer.primary_occupancy(WorkerCapacity::new(None, None), 5000.0, 0, 7),
            None
        );
        assert!(scorer.warned_no_signal.contains(&7));
    }

    #[test]
    fn zero_reported_capacity_counts_as_unknown() {
        let mut scorer = scorer(Some(1000.0), Some(8.0));
        // Both reported values are zero, so both fallbacks are used.
        assert_eq!(
            scorer.primary_occupancy(WorkerCapacity::new(Some(0), Some(0)), 950.0, 0, 7),
            Some(0.95)
        );
    }

    #[test]
    fn kv_and_projected_concurrency_take_the_larger() {
        let mut scorer = scorer(None, None);

        // 950 / 1000 = 0.95 KV, (0 + this request) / 8 = 0.125 concurrency.
        assert_eq!(
            scorer.primary_occupancy(WorkerCapacity::new(Some(1000), Some(8)), 950.0, 0, 0),
            Some(0.95)
        );

        // KV is tiny; the arriving request takes the worker to 8 / 8 = 1.0.
        assert_eq!(
            scorer.primary_occupancy(WorkerCapacity::new(Some(1_000_000), Some(8)), 0.0, 7, 0),
            Some(1.0)
        );
    }

    #[test]
    fn fallback_used_only_when_the_worker_reports_nothing() {
        let mut scorer = scorer(Some(1000.0), Some(8.0));

        assert_eq!(
            scorer.primary_occupancy(WorkerCapacity::new(None, None), 500.0, 0, 0),
            Some(0.5)
        );
        // A reported value wins over the fallback.
        assert_eq!(
            scorer.primary_occupancy(WorkerCapacity::new(Some(2000), None), 500.0, 0, 0),
            Some(0.25)
        );
    }

    #[traced_test]
    #[test]
    fn no_signal_warns_once_per_worker() {
        let mut scorer = scorer(None, None);
        let capacity = WorkerCapacity::new(None, None);
        for _ in 0..2 {
            assert!(scorer.primary_occupancy(capacity, 1.0, 0, 7).is_none());
        }
        logs_assert(|lines| {
            let count = lines
                .iter()
                .filter(|line| line.contains("dw-spillover primary worker advertises neither"))
                .count();
            if count == 1 {
                Ok(())
            } else {
                Err(format!(
                    "expected exactly one no-signal warning, got {count}"
                ))
            }
        });
    }
}
