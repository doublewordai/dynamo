// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-load-aware routing for text-input model backends.
//!
//! Unlike the token path, this router cannot hash prompt tokens or consult the
//! KV index. It instead uses each backend's reported KV occupancy and queue
//! depth to choose an exact `(worker, DP rank)` target. Session affinity then
//! keeps subsequent requests on that exact target.

use std::{
    cmp::Ordering,
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use dynamo_runtime::{
    component::Client,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Data, Error, ManyOut, PushRouter, ResponseStream,
        RouterMode, SingleIn, WorkerLoadMonitor, async_trait,
    },
    protocols::maybe_error::MaybeError,
};
use futures::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    discovery::{KvWorkerMonitor, WorkerLoadState},
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

/// Request operations needed by the text KV router.
pub(crate) trait TextKvRequest: Data {
    fn explicit_target(&self) -> Result<Option<AffinityTarget>, Error>;
    fn set_dp_rank(&mut self, dp_rank: Option<u32>);
}

macro_rules! impl_text_kv_request {
    ($request:ty) => {
        impl TextKvRequest for $request {
            fn explicit_target(&self) -> Result<Option<AffinityTarget>, Error> {
                let Some(nvext) = self.nvext.as_ref() else {
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

            fn set_dp_rank(&mut self, dp_rank: Option<u32>) {
                match dp_rank {
                    Some(dp_rank) => {
                        self.nvext.get_or_insert_with(NvExt::default).dp_rank = Some(dp_rank);
                    }
                    None => {
                        if let Some(nvext) = self.nvext.as_mut() {
                            nvext.dp_rank = None;
                        }
                    }
                }
            }
        }
    };
}

impl_text_kv_request!(NvCreateChatCompletionRequest);
impl_text_kv_request!(NvCreateCompletionRequest);

/// State shared by the chat and completions pipelines for one model WorkerSet.
pub(crate) struct TextKvRoutingState {
    monitor: KvWorkerMonitor,
    selector: ReportedKvLoadSelector,
    affinity: Option<AffinityCoordinator>,
    model_name: String,
}

impl TextKvRoutingState {
    pub(crate) fn new(
        monitor: KvWorkerMonitor,
        session_affinity_ttl_secs: Option<u64>,
        model_name: String,
    ) -> Result<Arc<Self>, Error> {
        let affinity = session_affinity_ttl_secs
            .map(|ttl| AffinityCoordinator::new(Duration::from_secs(ttl)))
            .transpose()?;
        Ok(Arc::new(Self {
            monitor,
            selector: ReportedKvLoadSelector::default(),
            affinity,
            model_name,
        }))
    }

    fn worker_belongs_to_model(&self, worker_id: u64) -> bool {
        self.monitor
            .worker_belongs_to_model(worker_id, &self.model_name)
    }

    fn target_is_available(&self, client: &Client, target: AffinityTarget) -> bool {
        if !client.instance_ids_avail().contains(&target.worker_id)
            || !self.worker_belongs_to_model(target.worker_id)
        {
            return false;
        }
        candidate_targets(
            &[target.worker_id],
            &self.monitor.load_states_for(&[target.worker_id]),
        )
        .contains(&target)
    }
}

/// Typed adapter around the shared routing state and a direct Dynamo transport.
pub(crate) struct TextKvPushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    inner: PushRouter<T, U>,
    state: Arc<TextKvRoutingState>,
}

impl<T, U> TextKvPushRouter<T, U>
where
    T: TextKvRequest + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    pub(crate) async fn new(client: Client, state: Arc<TextKvRoutingState>) -> Result<Self, Error> {
        let monitor = Arc::new(state.monitor.clone()) as Arc<dyn WorkerLoadMonitor>;
        let inner =
            PushRouter::from_client_with_monitor(client, RouterMode::Direct, Some(monitor)).await?;
        Ok(Self { inner, state })
    }

    async fn acquire_affinity(
        &self,
        request: &SingleIn<T>,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityAcquire>, Error> {
        let Some(coordinator) = self.state.affinity.as_ref() else {
            return Ok(None);
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok(None);
        };
        let request_context = request.context();
        let operation = coordinator
            .acquire_with_context(&session_id, requested_target, request_context.as_ref())
            .await?;
        let Some(target) = operation.target() else {
            return Ok(Some(operation));
        };
        if self.state.target_is_available(&self.inner.client, target) {
            return Ok(Some(operation));
        }

        // A worker or DP rank disappeared. Drop the stale binding and let this
        // request initialize a new exact target.
        operation.invalidate();
        coordinator
            .acquire_with_context(&session_id, requested_target, request_context.as_ref())
            .await
            .map(Some)
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for TextKvPushRouter<T, U>
where
    T: TextKvRequest + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, mut request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        let explicit = request.explicit_target()?;
        let operation = self.acquire_affinity(&request, explicit).await?;

        let (target, permit) = if let Some(target) = operation.as_ref().and_then(|op| op.target()) {
            (target, self.state.selector.reserve_existing(target))
        } else {
            self.state
                .selector
                .select(&self.inner.client, self.state.as_ref(), explicit)?
        };

        request.set_dp_rank(target.dp_rank);
        tracing::info!(
            router_mode = "kv",
            kv_routing_mechanism = "reported-load",
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            affinity_reuse = operation.as_ref().is_some_and(|op| op.target().is_some()),
            "Selected text-input KV target"
        );

        let dispatch = self.inner.dispatch_exact(request, target.worker_id).await;
        let stream = match dispatch {
            Ok(stream) => permit.into_tracked_stream(stream),
            Err(error) => {
                if let Some(operation) = operation {
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

#[derive(Clone, Copy, Debug)]
struct CandidateLoad {
    target: AffinityTarget,
    kv_used_blocks: Option<u64>,
    kv_total_blocks: Option<u64>,
    num_waiting_reqs: Option<u64>,
}

impl CandidateLoad {
    fn has_usable_kv_load(&self) -> bool {
        self.kv_used_blocks.is_some() && self.kv_total_blocks.is_some_and(|total| total > 0)
    }
}

#[derive(Default)]
struct SelectorState {
    frontend_inflight: HashMap<AffinityTarget, u64>,
    fallback_counter: u64,
}

#[derive(Default)]
struct ReportedKvLoadSelector {
    state: Arc<Mutex<SelectorState>>,
}

impl ReportedKvLoadSelector {
    fn select(
        &self,
        client: &Client,
        routing_state: &TextKvRoutingState,
        explicit: Option<AffinityTarget>,
    ) -> Result<(AffinityTarget, TextKvPermit), Error> {
        let mut worker_ids = client.instance_ids_free();
        worker_ids.retain(|worker_id| routing_state.worker_belongs_to_model(*worker_id));
        worker_ids.sort_unstable();
        if worker_ids.is_empty() {
            return Err(anyhow::anyhow!(
                "no free workers available for text KV routing on endpoint {}",
                client.endpoint.id()
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

        let load_states = routing_state.monitor.load_states_for(&worker_ids);
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

        let mut state = self.state.lock().unwrap();
        let target = choose_candidate(&candidates, &mut state);
        *state.frontend_inflight.entry(target).or_default() += 1;
        drop(state);
        Ok((target, TextKvPermit::new(self.state.clone(), target)))
    }

    fn reserve_existing(&self, target: AffinityTarget) -> TextKvPermit {
        let mut state = self.state.lock().unwrap();
        *state.frontend_inflight.entry(target).or_default() += 1;
        drop(state);
        TextKvPermit::new(self.state.clone(), target)
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
            let dp_rank = load_state.data_parallel_start_rank.saturating_add(offset);
            targets.push(AffinityTarget {
                worker_id: *worker_id,
                dp_rank: uses_dp_routing.then_some(dp_rank),
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
            }
        })
        .collect()
}

fn choose_candidate(candidates: &[CandidateLoad], state: &mut SelectorState) -> AffinityTarget {
    let has_usable_load = candidates.iter().any(CandidateLoad::has_usable_kv_load);
    if !has_usable_load {
        let target = candidates[state.fallback_counter as usize % candidates.len()].target;
        state.fallback_counter = state.fallback_counter.wrapping_add(1);
        return target;
    }

    let mut best = Vec::new();
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.has_usable_kv_load())
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
    state: &SelectorState,
) -> Ordering {
    let left_used = left.kv_used_blocks.expect("usable load has used blocks") as u128;
    let left_total = left.kv_total_blocks.expect("usable load has total blocks") as u128;
    let right_used = right.kv_used_blocks.expect("usable load has used blocks") as u128;
    let right_total = right.kv_total_blocks.expect("usable load has total blocks") as u128;

    (left_used * right_total)
        .cmp(&(right_used * left_total))
        .then_with(|| compare_optional_load(left.num_waiting_reqs, right.num_waiting_reqs))
        .then_with(|| {
            state
                .frontend_inflight
                .get(&left.target)
                .copied()
                .unwrap_or_default()
                .cmp(
                    &state
                        .frontend_inflight
                        .get(&right.target)
                        .copied()
                        .unwrap_or_default(),
                )
        })
}

fn compare_optional_load(left: Option<u64>, right: Option<u64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

struct TextKvPermit {
    state: Arc<Mutex<SelectorState>>,
    target: AffinityTarget,
    released: bool,
}

impl TextKvPermit {
    fn new(state: Arc<Mutex<SelectorState>>, target: AffinityTarget) -> Self {
        Self {
            state,
            target,
            released: false,
        }
    }

    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let mut state = self.state.lock().unwrap();
        let Some(inflight) = state.frontend_inflight.get_mut(&self.target) else {
            return;
        };
        *inflight = inflight.saturating_sub(1);
        if *inflight == 0 {
            state.frontend_inflight.remove(&self.target);
        }
    }

    fn into_tracked_stream<U: Data>(self, stream: ManyOut<U>) -> ManyOut<U> {
        let context = stream.context();
        ResponseStream::new(
            Box::pin(TextKvTrackedStream {
                stream,
                permit: Some(self),
            }),
            context,
        )
    }
}

impl Drop for TextKvPermit {
    fn drop(&mut self) {
        self.release();
    }
}

struct TextKvTrackedStream<U: Data> {
    stream: ManyOut<U>,
    permit: Option<TextKvPermit>,
}

impl<U: Data> Stream for TextKvTrackedStream<U> {
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.stream.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) {
            drop(self.permit.take());
        }
        poll
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
    ) -> CandidateLoad {
        CandidateLoad {
            target: AffinityTarget { worker_id, dp_rank },
            kv_used_blocks: used,
            kv_total_blocks: total,
            num_waiting_reqs: waiting,
        }
    }

    #[test]
    fn chooses_lowest_normalized_kv_occupancy() {
        let candidates = [
            candidate(1, Some(0), Some(40), Some(100), Some(0)),
            candidate(2, Some(1), Some(300), Some(1000), Some(9)),
        ];
        let mut state = SelectorState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );
    }

    #[test]
    fn queue_then_frontend_inflight_break_equal_occupancy() {
        let candidates = [
            candidate(1, Some(0), Some(30), Some(100), Some(2)),
            candidate(2, Some(1), Some(300), Some(1000), Some(1)),
        ];
        let mut state = SelectorState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );

        let candidates = [
            candidate(1, Some(0), Some(30), Some(100), Some(1)),
            candidate(2, Some(1), Some(300), Some(1000), Some(1)),
        ];
        state.frontend_inflight.insert(candidates[1].target, 1);
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[0].target
        );
    }

    #[test]
    fn unknown_load_uses_stable_round_robin() {
        let candidates = [
            candidate(1, None, None, None, None),
            candidate(2, None, None, None, Some(0)),
        ];
        let mut state = SelectorState::default();
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

    #[test]
    fn known_load_is_preferred_to_unknown_load() {
        let candidates = [
            candidate(1, None, None, None, Some(0)),
            candidate(2, None, Some(99), Some(100), Some(8)),
        ];
        let mut state = SelectorState::default();
        assert_eq!(
            choose_candidate(&candidates, &mut state),
            candidates[1].target
        );
    }

    #[test]
    fn frontend_inflight_is_released_with_the_permit() {
        let selector = ReportedKvLoadSelector::default();
        let target = AffinityTarget {
            worker_id: 10,
            dp_rank: Some(2),
        };

        let permit = selector.reserve_existing(target);
        assert_eq!(selector.state.lock().unwrap().frontend_inflight[&target], 1);

        drop(permit);
        assert!(
            !selector
                .state
                .lock()
                .unwrap()
                .frontend_inflight
                .contains_key(&target)
        );
    }
}
