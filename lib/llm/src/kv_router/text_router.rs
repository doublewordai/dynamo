// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reported-load KV routing for backends that receive OpenAI text requests.
//!
//! Text-input backends cannot use the token path's prompt-block index because
//! tokenization happens behind the frontend. They instead report per-rank KV
//! capacity and queue depth. This module chooses an exact `(worker, DP rank)`
//! from those reports and uses the shared session-affinity coordinator to keep
//! later requests on the same target.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use dynamo_runtime::{
    component::Client,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Data, Error, ManyOut, PushRouter, SingleIn,
        async_trait,
    },
    protocols::maybe_error::MaybeError,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    discovery::{KvWorkerMonitor, RuntimeConfigWatch, WorkerLoadState},
    protocols::{
        common::extensions::NvExt,
        openai::{
            chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
        },
    },
    session_affinity::{
        AffinityAcquire, AffinityCoordinator, AffinityTarget, affinity_id, invalid_argument,
    },
};

#[derive(Clone, Copy, Debug)]
struct CandidateLoad {
    target: AffinityTarget,
    kv_used_blocks: Option<u64>,
    kv_total_blocks: Option<u64>,
    num_waiting_reqs: Option<u64>,
    load_report_revision: Option<u64>,
}

impl CandidateLoad {
    fn capacity(&self) -> Option<u64> {
        self.kv_total_blocks.filter(|total| *total > 0)
    }

    fn report_identity(&self) -> Option<LoadReportIdentity> {
        self.load_report_revision
            .map(LoadReportIdentity::Versioned)
            .or_else(|| {
                (self.kv_used_blocks.is_some() || self.num_waiting_reqs.is_some()).then_some(
                    LoadReportIdentity::Legacy {
                        kv_used_blocks: self.kv_used_blocks,
                        num_waiting_reqs: self.num_waiting_reqs,
                    },
                )
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadReportIdentity {
    Versioned(u64),
    Legacy {
        kv_used_blocks: Option<u64>,
        num_waiting_reqs: Option<u64>,
    },
}

#[derive(Default)]
struct TextKvRouterState {
    dispatches_since_report: HashMap<AffinityTarget, u64>,
    last_report_identity: HashMap<AffinityTarget, LoadReportIdentity>,
    fallback_counter: u64,
}

impl TextKvRouterState {
    fn reconcile_candidate(&mut self, candidate: &CandidateLoad) {
        let Some(identity) = candidate.report_identity() else {
            return;
        };
        let previous = self.last_report_identity.insert(candidate.target, identity);
        let should_reset = match previous {
            Some(previous) => previous != identity,
            None => matches!(identity, LoadReportIdentity::Versioned(_)),
        };
        if should_reset {
            self.dispatches_since_report.remove(&candidate.target);
            tracing::debug!(
                worker_id = candidate.target.worker_id,
                dp_rank = ?candidate.target.dp_rank,
                ?identity,
                reported_queue = candidate.num_waiting_reqs.unwrap_or_default(),
                "Reset text-router dispatch accounting from a new worker load report"
            );
        }
    }

    fn reconcile_candidates(&mut self, candidates: &[CandidateLoad]) {
        let active_targets: HashSet<_> = candidates
            .iter()
            .map(|candidate| candidate.target)
            .collect();
        self.dispatches_since_report
            .retain(|target, _| active_targets.contains(target));
        self.last_report_identity
            .retain(|target, _| active_targets.contains(target));
        for candidate in candidates {
            self.reconcile_candidate(candidate);
        }
    }

    fn record_dispatch(&mut self, target: AffinityTarget) -> u64 {
        let dispatches = self.dispatches_since_report.entry(target).or_default();
        *dispatches = dispatches.saturating_add(1);
        *dispatches
    }

    fn cancel_dispatch(&mut self, target: AffinityTarget) {
        let Some(dispatches) = self.dispatches_since_report.get_mut(&target) else {
            return;
        };
        *dispatches = dispatches.saturating_sub(1);
        if *dispatches == 0 {
            self.dispatches_since_report.remove(&target);
        }
    }
}

#[derive(Clone)]
pub(crate) struct TextKvRouter {
    client: Client,
    monitor: KvWorkerMonitor,
    runtime_configs: RuntimeConfigWatch,
    state: Arc<Mutex<TextKvRouterState>>,
}

impl TextKvRouter {
    pub(crate) fn new(
        client: Client,
        monitor: KvWorkerMonitor,
        runtime_configs: RuntimeConfigWatch,
    ) -> Self {
        Self {
            client,
            monitor,
            runtime_configs,
            state: Arc::new(Mutex::new(TextKvRouterState::default())),
        }
    }

    fn target_is_available(&self, target: AffinityTarget) -> bool {
        if !self.client.instance_ids_avail().contains(&target.worker_id) {
            return false;
        }
        if !self
            .runtime_configs
            .borrow()
            .contains_key(&target.worker_id)
        {
            return false;
        }
        candidate_targets(
            &[target.worker_id],
            &self.monitor.load_states_for(&[target.worker_id]),
        )
        .contains(&target)
    }

    fn select(
        &self,
        explicit: Option<AffinityTarget>,
        allowed_worker_ids: Option<&HashSet<u64>>,
    ) -> Result<AffinityTarget, Error> {
        let mut worker_ids = self.client.instance_ids_free();
        let mut active_worker_ids = self.client.instance_ids_avail();
        let model_workers = self.runtime_configs.borrow();
        worker_ids.retain(|worker_id| model_workers.contains_key(worker_id));
        active_worker_ids.retain(|worker_id| model_workers.contains_key(worker_id));
        drop(model_workers);
        if let Some(allowed_worker_ids) = allowed_worker_ids {
            worker_ids.retain(|worker_id| allowed_worker_ids.contains(worker_id));
        }
        worker_ids.sort_unstable();
        if worker_ids.is_empty() {
            return Err(anyhow::anyhow!(
                "no free workers available for text KV routing on endpoint {}",
                self.client.endpoint.id()
            ));
        }

        if let Some(explicit) = explicit {
            if !worker_ids.contains(&explicit.worker_id) {
                return Err(invalid_argument(format!(
                    "worker {} is not available for text KV routing",
                    explicit.worker_id
                )));
            }
            worker_ids.retain(|worker_id| *worker_id == explicit.worker_id);
        }

        let load_states = self.monitor.load_states_for(&worker_ids);
        let mut candidates = candidate_loads(&worker_ids, &load_states);
        if let Some(explicit) = explicit
            && let Some(dp_rank) = explicit.dp_rank
        {
            candidates.retain(|candidate| candidate.target.dp_rank == Some(dp_rank));
        }
        if candidates.is_empty() {
            return Err(invalid_argument(
                "requested text KV routing target is unavailable",
            ));
        }

        // Reconcile and prune against the complete live fleet. `candidates`
        // may be a one-worker migration/explicit-target subset; using that
        // subset here would discard accounting for unrelated live ranks.
        active_worker_ids.sort_unstable();
        let active_load_states = self.monitor.load_states_for(&active_worker_ids);
        let active_candidates = candidate_loads(&active_worker_ids, &active_load_states);
        let mut state = self.state.lock().unwrap();
        state.reconcile_candidates(&active_candidates);
        let target = choose_candidate(&candidates, &mut state);
        let dispatches_since_report = state.record_dispatch(target);
        if let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.target == target)
        {
            tracing::debug!(
                worker_id = target.worker_id,
                dp_rank = ?target.dp_rank,
                reported_queue = candidate.num_waiting_reqs.unwrap_or_default(),
                dispatches_since_report,
                total_kv_blocks = ?candidate.capacity(),
                load_report_revision = ?candidate.load_report_revision,
                "Recorded new text-router placement"
            );
        }
        Ok(target)
    }

    fn record_existing(&self, target: AffinityTarget) {
        let load_states = self.monitor.load_states_for(&[target.worker_id]);
        let candidate = candidate_loads(&[target.worker_id], &load_states)
            .into_iter()
            .find(|candidate| candidate.target == target);
        let mut state = self.state.lock().unwrap();
        if let Some(candidate) = candidate.as_ref() {
            state.reconcile_candidate(candidate);
        }
        let dispatches_since_report = state.record_dispatch(target);
        tracing::debug!(
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            dispatches_since_report,
            "Recorded affinity-reused text-router dispatch"
        );
    }

    fn cancel_dispatch(&self, target: AffinityTarget) {
        self.state.lock().unwrap().cancel_dispatch(target);
    }
}

pub(crate) trait TextKvRequest: Data + Serialize {
    fn nvext(&self) -> Option<&NvExt>;
    fn nvext_mut(&mut self) -> &mut Option<NvExt>;
}

impl TextKvRequest for NvCreateChatCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn nvext_mut(&mut self) -> &mut Option<NvExt> {
        &mut self.nvext
    }
}

impl TextKvRequest for NvCreateCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn nvext_mut(&mut self) -> &mut Option<NvExt> {
        &mut self.nvext
    }
}

fn explicit_target(request: &impl TextKvRequest) -> Result<Option<AffinityTarget>, Error> {
    let Some(nvext) = request.nvext() else {
        return Ok(None);
    };
    let worker_id = nvext.decode_worker_id.or(nvext.backend_instance_id);
    if worker_id.is_none() && nvext.dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for text KV routing",
        ));
    }
    Ok(worker_id.map(|worker_id| AffinityTarget {
        worker_id,
        dp_rank: nvext.dp_rank,
    }))
}

fn apply_target(request: &mut impl TextKvRequest, target: AffinityTarget) {
    match target.dp_rank {
        Some(dp_rank) => {
            request
                .nvext_mut()
                .get_or_insert_with(NvExt::default)
                .dp_rank = Some(dp_rank);
        }
        None => {
            if let Some(nvext) = request.nvext_mut().as_mut() {
                nvext.dp_rank = None;
            }
        }
    }
}

pub(crate) struct TextKvPushRouter<T, U>
where
    T: TextKvRequest,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    inner: PushRouter<T, U>,
    selector: Arc<TextKvRouter>,
    affinity: Option<AffinityCoordinator>,
}

impl<T, U> TextKvPushRouter<T, U>
where
    T: TextKvRequest,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    pub(crate) fn new(
        inner: PushRouter<T, U>,
        selector: Arc<TextKvRouter>,
        affinity: Option<AffinityCoordinator>,
    ) -> Self {
        Self {
            inner,
            selector,
            affinity,
        }
    }

    async fn acquire_routable(
        &self,
        session_id: &crate::protocols::common::extensions::SessionAffinityId,
        explicit: Option<AffinityTarget>,
        request_context: &dyn dynamo_runtime::pipeline::AsyncEngineContext,
    ) -> Result<AffinityAcquire, Error> {
        let affinity = self
            .affinity
            .as_ref()
            .expect("affinity acquisition requires an enabled coordinator");
        let operation = affinity
            .acquire_with_context(session_id, explicit, request_context)
            .await?;
        let Some(target) = operation.target() else {
            return Ok(operation);
        };
        if self.selector.target_is_available(target) {
            return Ok(operation);
        }

        operation.invalidate();
        affinity
            .acquire_with_context(session_id, explicit, request_context)
            .await
    }

    async fn route(&self, mut request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        let explicit = explicit_target(request.content())?;
        let session_id = if self.affinity.is_some() {
            affinity_id(&request)?
        } else {
            None
        };

        let mut operation = match session_id.as_ref() {
            Some(session_id) => Some(
                self.acquire_routable(session_id, explicit, request.context().as_ref())
                    .await?,
            ),
            None => None,
        };

        let target = match operation.as_ref().and_then(AffinityAcquire::target) {
            Some(target) => {
                self.selector.record_existing(target);
                target
            }
            None => self.selector.select(
                explicit,
                operation
                    .as_ref()
                    .and_then(AffinityAcquire::migration_worker_ids),
            )?,
        };
        apply_target(&mut *request, target);

        tracing::info!(
            router_mode = "kv",
            kv_routing_mechanism = "reported-load",
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            affinity_reuse = operation.as_ref().is_some_and(|operation| operation.target().is_some()),
            affinity_action = operation.as_ref().map_or("none", AffinityAcquire::action_name),
            "Selected text-input KV target"
        );
        if let Some(session_id) = session_id.as_ref() {
            tracing::debug!(
                session_id = %session_id.as_str(),
                worker_id = target.worker_id,
                dp_rank = ?target.dp_rank,
                affinity_action = operation.as_ref().map_or("none", AffinityAcquire::action_name),
                "Observed text-input KV session target"
            );
        }

        let stream = match self.inner.dispatch_exact(request, target.worker_id).await {
            Ok(stream) => stream,
            Err(error) => {
                self.selector.cancel_dispatch(target);
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        match operation {
            Some(operation) => operation.into_stream(target, stream),
            None => Ok(stream),
        }
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for TextKvPushRouter<T, U>
where
    T: TextKvRequest,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        self.route(request).await
    }
}

fn candidate_targets(
    worker_ids: &[u64],
    load_states: &HashMap<u64, WorkerLoadState>,
) -> Vec<AffinityTarget> {
    let mut targets = Vec::new();
    for worker_id in worker_ids {
        let Some(load_state) = load_states.get(worker_id) else {
            targets.push(AffinityTarget {
                worker_id: *worker_id,
                dp_rank: None,
            });
            continue;
        };

        let dp_size = load_state.data_parallel_size.max(1);
        let uses_dp_routing = load_state.data_parallel_start_rank != 0 || dp_size > 1;
        for offset in 0..dp_size {
            targets.push(AffinityTarget {
                worker_id: *worker_id,
                dp_rank: uses_dp_routing
                    .then_some(load_state.data_parallel_start_rank.saturating_add(offset)),
            });
        }
    }
    targets.sort_unstable();
    targets
}

fn candidate_loads(
    worker_ids: &[u64],
    load_states: &HashMap<u64, WorkerLoadState>,
) -> Vec<CandidateLoad> {
    candidate_targets(worker_ids, load_states)
        .into_iter()
        .map(|target| {
            let rank = target.dp_rank.unwrap_or(0);
            let state = load_states.get(&target.worker_id);
            CandidateLoad {
                target,
                kv_used_blocks: state.and_then(|state| state.kv_used_blocks.get(&rank).copied()),
                kv_total_blocks: state.and_then(|state| state.kv_total_blocks.get(&rank).copied()),
                num_waiting_reqs: state
                    .and_then(|state| state.num_waiting_reqs.get(&rank).copied()),
                load_report_revision: state
                    .and_then(|state| state.load_report_revisions.get(&rank).copied()),
            }
        })
        .collect()
}

fn choose_candidate(candidates: &[CandidateLoad], state: &mut TextKvRouterState) -> AffinityTarget {
    let has_usable_capacity = candidates
        .iter()
        .any(|candidate| candidate.capacity().is_some());
    if !has_usable_capacity {
        let target = candidates[state.fallback_counter as usize % candidates.len()].target;
        state.fallback_counter = state.fallback_counter.wrapping_add(1);
        return target;
    }

    let mut best = Vec::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.capacity().is_some())
    {
        let Some(current_best) = best.first().copied() else {
            best.push(candidate);
            continue;
        };
        match compare_candidate(candidate, current_best, state) {
            Ordering::Less => {
                best.clear();
                best.push(candidate);
            }
            Ordering::Equal => best.push(candidate),
            Ordering::Greater => {}
        }
    }

    let index = if best.len() == 1 {
        0
    } else {
        rand::rng().random_range(0..best.len())
    };
    best[index].target
}

fn compare_candidate(
    left: &CandidateLoad,
    right: &CandidateLoad,
    state: &TextKvRouterState,
) -> Ordering {
    match (left.capacity(), right.capacity()) {
        (Some(left_capacity), Some(right_capacity)) => {
            let left_projected = u128::from(left.num_waiting_reqs.unwrap_or_default())
                + u128::from(
                    state
                        .dispatches_since_report
                        .get(&left.target)
                        .copied()
                        .unwrap_or_default(),
                )
                + 1;
            let right_projected = u128::from(right.num_waiting_reqs.unwrap_or_default())
                + u128::from(
                    state
                        .dispatches_since_report
                        .get(&right.target)
                        .copied()
                        .unwrap_or_default(),
                )
                + 1;
            let lhs = left_projected * u128::from(right_capacity);
            let rhs = right_projected * u128::from(left_capacity);
            // Otherwise identical workers report capacities that differ by a
            // few blocks (profiling noise); a strict comparison would let one
            // win every dispatch. Within 1% is a tie, left to the random pick.
            if lhs.abs_diff(rhs) < lhs.max(rhs) / 100 {
                Ordering::Equal
            } else {
                lhs.cmp(&rhs)
            }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        worker_id: u64,
        dp_rank: Option<u32>,
        used: Option<u64>,
        total: Option<u64>,
        waiting: Option<u64>,
        load_report_revision: Option<u64>,
    ) -> CandidateLoad {
        CandidateLoad {
            target: AffinityTarget { worker_id, dp_rank },
            kv_used_blocks: used,
            kv_total_blocks: total,
            num_waiting_reqs: waiting,
            load_report_revision,
        }
    }

    #[test]
    fn chooses_lowest_projected_queue_per_capacity() {
        let candidates = [
            candidate(1, Some(0), Some(0), Some(500), Some(5), Some(1)),
            candidate(2, Some(1), Some(0), Some(100), Some(1), Some(1)),
        ];
        let mut state = TextKvRouterState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target,
        );
        for _ in 0..5 {
            state.record_dispatch(candidates[0].target);
        }
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target,
        );
    }

    #[test]
    fn near_equal_capacities_are_a_tie() {
        let state = TextKvRouterState::default();
        let left = candidate(1, Some(0), Some(0), Some(164_000), Some(0), Some(1));
        let right = candidate(2, Some(0), Some(0), Some(164_007), Some(0), Some(1));
        assert_eq!(compare_candidate(&left, &right, &state), Ordering::Equal);
        assert_eq!(compare_candidate(&right, &left, &state), Ordering::Equal);
    }

    #[test]
    fn two_percent_capacity_difference_still_orders() {
        let state = TextKvRouterState::default();
        let left = candidate(1, Some(0), Some(0), Some(100_000), Some(0), Some(1));
        let right = candidate(2, Some(0), Some(0), Some(102_000), Some(0), Some(1));
        assert_eq!(compare_candidate(&left, &right, &state), Ordering::Greater);
        assert_eq!(compare_candidate(&right, &left, &state), Ordering::Less);
    }

    #[test]
    fn dispatch_accounting_redirects_bursts_before_reports_update() {
        let candidates = [
            candidate(1, Some(0), Some(0), Some(1000), Some(0), Some(1)),
            candidate(1, Some(1), Some(0), Some(1000), Some(0), Some(1)),
        ];
        let mut state = TextKvRouterState::default();
        state.reconcile_candidates(&candidates);
        state.record_dispatch(candidates[0].target);
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target,
        );
    }

    #[test]
    fn zero_queue_burst_tracks_capacity_ratio() {
        let candidates = [
            candidate(1, Some(0), Some(0), Some(500), Some(0), Some(1)),
            candidate(1, Some(1), Some(0), Some(500), Some(0), Some(1)),
            candidate(1, Some(2), Some(0), Some(100), Some(0), Some(1)),
        ];
        let mut state = TextKvRouterState::default();
        state.reconcile_candidates(&candidates);
        let mut counts: HashMap<AffinityTarget, u64> = HashMap::new();
        for _ in 0..110 {
            let target = choose_candidate(&candidates, &mut state);
            state.record_dispatch(target);
            *counts.entry(target).or_default() += 1;
        }
        assert_eq!(counts.get(&candidates[0].target), Some(&50));
        assert_eq!(counts.get(&candidates[1].target), Some(&50));
        assert_eq!(counts.get(&candidates[2].target), Some(&10));
    }

    #[test]
    fn new_load_report_revision_resets_dispatches_when_values_are_unchanged() {
        let first = candidate(1, Some(0), Some(10), Some(100), Some(3), Some(7));
        let second = candidate(1, Some(0), Some(10), Some(100), Some(3), Some(8));
        let mut state = TextKvRouterState::default();
        state.reconcile_candidate(&first);
        state.record_dispatch(first.target);
        state.record_dispatch(first.target);
        state.reconcile_candidate(&second);
        assert!(!state.dispatches_since_report.contains_key(&first.target));
        assert_eq!(
            state.last_report_identity.get(&first.target),
            Some(&LoadReportIdentity::Versioned(8)),
        );
    }

    #[test]
    fn first_versioned_load_report_resets_preexisting_dispatches() {
        let report = candidate(1, Some(0), Some(10), Some(100), Some(3), Some(7));
        let mut state = TextKvRouterState::default();
        state.record_dispatch(report.target);
        state.record_dispatch(report.target);

        state.reconcile_candidate(&report);

        assert!(!state.dispatches_since_report.contains_key(&report.target));
        assert_eq!(
            state.last_report_identity.get(&report.target),
            Some(&LoadReportIdentity::Versioned(7)),
        );
    }

    #[test]
    fn heartbeat_with_same_load_report_revision_does_not_reset_dispatches() {
        let report = candidate(1, Some(0), Some(10), Some(100), Some(3), Some(7));
        let mut state = TextKvRouterState::default();
        state.reconcile_candidate(&report);
        state.record_dispatch(report.target);
        state.reconcile_candidate(&report);
        assert_eq!(state.dispatches_since_report.get(&report.target), Some(&1));
    }

    #[test]
    fn legacy_reports_reset_only_when_reported_values_change() {
        let first = candidate(1, Some(0), Some(10), Some(100), Some(3), None);
        let changed = candidate(1, Some(0), Some(11), Some(100), Some(3), None);
        let mut state = TextKvRouterState::default();
        state.reconcile_candidate(&first);
        state.record_dispatch(first.target);
        state.reconcile_candidate(&first);
        assert_eq!(state.dispatches_since_report.get(&first.target), Some(&1));
        state.reconcile_candidate(&changed);
        assert!(!state.dispatches_since_report.contains_key(&first.target));
    }

    #[test]
    fn first_legacy_identity_does_not_reset_preexisting_dispatches() {
        let report = candidate(1, Some(0), Some(0), Some(100), Some(0), None);
        let mut state = TextKvRouterState::default();
        state.record_dispatch(report.target);

        state.reconcile_candidate(&report);

        assert_eq!(state.dispatches_since_report.get(&report.target), Some(&1));
        assert_eq!(
            state.last_report_identity.get(&report.target),
            Some(&LoadReportIdentity::Legacy {
                kv_used_blocks: Some(0),
                num_waiting_reqs: Some(0),
            }),
        );
    }

    #[test]
    fn cancel_dispatch_rolls_back_only_the_failed_dispatch() {
        let target = AffinityTarget {
            worker_id: 1,
            dp_rank: Some(0),
        };
        let mut state = TextKvRouterState::default();
        state.record_dispatch(target);
        state.record_dispatch(target);
        state.cancel_dispatch(target);
        assert_eq!(state.dispatches_since_report.get(&target), Some(&1));
        state.cancel_dispatch(target);
        assert!(!state.dispatches_since_report.contains_key(&target));
    }

    #[test]
    fn candidates_with_capacity_outrank_candidates_without_it() {
        let usable = candidate(1, Some(0), None, Some(100), Some(4), Some(1));
        let missing = candidate(2, Some(1), None, None, Some(0), Some(1));
        let mut state = TextKvRouterState::default();
        assert_eq!(
            choose_candidate(&[missing, usable], &mut state),
            usable.target
        );
    }

    #[test]
    fn unknown_capacity_uses_stable_round_robin() {
        let candidates = [
            candidate(1, None, None, None, None, None),
            candidate(2, None, None, None, Some(0), None),
        ];
        let mut state = TextKvRouterState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target
        );
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target
        );
    }

    #[test]
    fn candidate_loads_include_load_report_revision() {
        let mut worker = WorkerLoadState::default();
        worker.kv_total_blocks.insert(2, 500);
        worker.num_waiting_reqs.insert(2, 4);
        worker.load_report_revisions.insert(2, 19);
        worker.data_parallel_start_rank = 2;
        let states = HashMap::from([(20, worker)]);

        let candidates = candidate_loads(&[20], &states);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].load_report_revision, Some(19));
    }

    #[test]
    fn reconciliation_prunes_removed_targets() {
        let removed = candidate(1, None, Some(0), Some(100), Some(0), Some(1));
        let retained = candidate(2, None, Some(0), Some(100), Some(0), Some(1));
        let mut state = TextKvRouterState::default();
        state.reconcile_candidates(&[removed, retained]);
        state.record_dispatch(removed.target);
        state.record_dispatch(retained.target);
        state.reconcile_candidates(&[retained]);
        assert!(!state.dispatches_since_report.contains_key(&removed.target));
        assert!(!state.last_report_identity.contains_key(&removed.target));
        assert_eq!(
            state.dispatches_since_report.get(&retained.target),
            Some(&1)
        );
    }

    #[test]
    fn expands_dp_workers_but_not_plain_workers() {
        let mut states = HashMap::new();
        let mut dp = WorkerLoadState::default();
        dp.data_parallel_start_rank = 4;
        dp.data_parallel_size = 2;
        states.insert(20, dp);
        states.insert(10, WorkerLoadState::default());

        assert_eq!(
            candidate_targets(&[10, 20], &states),
            vec![
                AffinityTarget {
                    worker_id: 10,
                    dp_rank: None,
                },
                AffinityTarget {
                    worker_id: 20,
                    dp_rank: Some(4),
                },
                AffinityTarget {
                    worker_id: 20,
                    dp_rank: Some(5),
                },
            ]
        );
    }
}
