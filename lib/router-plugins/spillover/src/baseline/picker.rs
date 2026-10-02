// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-candidate port of `DefaultWorkerPicker`.
//!
//! Mirrors the seeded branch of `pick_default_worker` in
//! `lib/kv-router/src/scheduling/selector/default.rs`: candidates are ordered by
//! `(worker_id, dp_rank)`, ties at `temperature == 0.0` are broken with a reservoir draw from
//! the shared seedable RNG, and a positive temperature samples the softmax distribution over
//! the same ordered costs. The softmax maths is copied verbatim so sample boundaries match.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};
use parking_lot::Mutex;

fn softmax_sample_index<T>(
    entries: &[T],
    cost: impl Fn(&T) -> f64,
    temperature: f64,
    sample: f64,
    probabilities: &mut Vec<f64>,
) -> usize {
    assert!(!entries.is_empty(), "Empty entries for softmax sampling");
    debug_assert_ne!(temperature, 0.0);

    let (min_cost, max_cost) = entries
        .iter()
        .map(&cost)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), cost| {
            (lo.min(cost), hi.max(cost))
        });

    probabilities.clear();
    if min_cost == max_cost {
        probabilities.resize(entries.len(), 1.0 / entries.len() as f64);
    } else {
        let range = max_cost - min_cost;
        let magnitude = if range.is_finite() {
            1.0
        } else {
            min_cost.abs().max(max_cost.abs())
        };
        let min_normalized = min_cost / magnitude;
        let scale = -1.0 / ((max_cost / magnitude - min_normalized) * temperature);
        let max_scaled = min_normalized * scale;
        probabilities.extend(
            entries
                .iter()
                .map(|entry| (cost(entry) / magnitude * scale - max_scaled).exp()),
        );
    }

    let sum: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= sum;
    }
    let mut cumulative = 0.0;
    for (row, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if sample <= cumulative {
            return row;
        }
    }
    entries.len() - 1
}

pub(super) struct BaselinePicker {
    temperature: f64,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    order: Vec<usize>,
    probabilities: Vec<f64>,
}

impl BaselinePicker {
    pub(super) fn new(temperature: f64, rng: Option<Arc<Mutex<fastrand::Rng>>>) -> Self {
        Self {
            temperature,
            rng,
            order: Vec::new(),
            probabilities: Vec::new(),
        }
    }
}

impl WorkerPicker for BaselinePicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }

        // The default selector makes an eligible session-affinity target exclusive: it narrows
        // eligibility to that worker's ranks and then runs its normal cost/softmax pick over
        // that subset. The host does not narrow a custom policy's candidate set, so filter here
        // and then apply the same selection over the target's ranks. Returning the first match
        // instead would ignore rank cost and temperature.
        self.order.clear();
        if let Some(target) = context.affinity_target() {
            self.order.extend(
                candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(row, candidate)| {
                        let worker = candidate.worker();
                        (worker.worker_id == target.worker_id
                            && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank))
                        .then_some(row)
                    }),
            );
            if self.order.is_empty() {
                // The host narrows only when the affinity target is eligible; a custom policy
                // gets its un-narrowed eligibility, so an ineligible target falls back to the
                // ordinary pick instead of erroring.
                self.order.extend(0..candidates.len());
            }
        } else {
            self.order.extend(0..candidates.len());
        }

        let temperature = context
            .router_temperature_override()
            .unwrap_or(self.temperature);

        let Some(rng) = &self.rng else {
            // The borrow checker cannot lend `candidates` and `&mut self` at once through a
            // method call, so run the unseeded selection inline.
            if temperature == 0.0 {
                let mut best_row = self.order[0];
                let mut best_cost = f64::INFINITY;
                let mut ties = 0;
                for &row in &self.order {
                    let cost = candidates[row].cost();
                    if cost < best_cost {
                        best_row = row;
                        best_cost = cost;
                        ties = 1;
                    } else if cost == best_cost {
                        ties += 1;
                        if fastrand::usize(0..ties) == 0 {
                            best_row = row;
                        }
                    }
                }
                return Ok(best_row);
            }
            let selected = softmax_sample_index(
                &self.order,
                |&row| candidates[row].cost(),
                temperature,
                fastrand::f64(),
                &mut self.probabilities,
            );
            return Ok(self.order[selected]);
        };

        // Canonical order: the default's deterministic path sorts by (worker_id, dp_rank).
        self.order.sort_unstable_by_key(|&row| {
            let worker = candidates[row].worker();
            (worker.worker_id, worker.dp_rank)
        });

        let mut rng = rng.lock();
        if temperature == 0.0 {
            let mut best = 0;
            let mut best_cost = f64::INFINITY;
            let mut ties = 0;
            for (index, &row) in self.order.iter().enumerate() {
                let cost = candidates[row].cost();
                if cost < best_cost {
                    best = index;
                    best_cost = cost;
                    ties = 1;
                } else if cost == best_cost {
                    ties += 1;
                    if rng.usize(0..ties) == 0 {
                        best = index;
                    }
                }
            }
            return Ok(self.order[best]);
        }
        let selected = softmax_sample_index(
            &self.order,
            |&row| candidates[row].cost(),
            temperature,
            rng.f64(),
            &mut self.probabilities,
        );
        Ok(self.order[selected])
    }
}
