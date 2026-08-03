// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::protocols::{TokensWithHashes, WorkerWithDpRank};
use dynamo_runtime::{
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait},
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tracing::Instrument;

use super::{
    OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY, OUTPUT_REPLAY_ID_ANNOTATION_KEY,
    cancellation::{cancel_on_stop, cancelled_error},
    request_guard::RequestGuard,
    selection::{RoutingRequestParts, SelectionOptions, WorkerSelection},
    strategy::{
        KvDispatchMode, KvRoutePin, KvRouteReservation, KvRoutingConstraints, KvRoutingOutcome,
        KvRoutingStrategy, KvSelectedRoute,
    },
};
use crate::{
    kv_router::{KvRouter, metrics::RouterRequestMetrics},
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RoutingData},
    },
    session_affinity::{AffinityTarget, explicit_target},
};

pub(crate) struct TokenRouteReservation {
    guard: RequestGuard,
    route_guard: Option<StageGuard>,
    context_id: String,
    phase: RequestPhase,
}

impl TokenRouteReservation {
    fn new(
        guard: RequestGuard,
        route_guard: StageGuard,
        context_id: String,
        phase: RequestPhase,
    ) -> Self {
        Self {
            guard,
            route_guard: Some(route_guard),
            context_id,
            phase,
        }
    }
}

#[async_trait]
impl KvRouteReservation<Annotated<LLMEngineOutput>> for TokenRouteReservation {
    fn start_dispatch(&mut self) {
        drop(self.route_guard.take());
        self.guard.start_dispatch(&self.phase.to_string());
        self.guard.record_prefill_start();
    }

    async fn abort(&mut self) {
        self.guard.abort().await;
    }

    fn into_stream(
        mut self,
        mut response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        self.guard.mark_dispatched();
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();
        let context_id = self.context_id;
        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut guard = self.guard;

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(item) = item else {
                            break;
                        };
                        guard.on_item(&item).await;
                        yield item;
                    }
                }
            }

            guard.finish().await;
        });
        ResponseStream::new(wrapped_stream, stream_context)
    }
}

#[async_trait]
impl KvRoutingStrategy<PreprocessedRequest> for Arc<KvRouter> {
    type Response = Annotated<LLMEngineOutput>;
    type Reservation = TokenRouteReservation;

    fn explicit_target(
        &self,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Result<Option<AffinityTarget>, Error> {
        explicit_target(request, request_phase(request))
    }

    fn is_query_only(&self, request: &SingleIn<PreprocessedRequest>) -> bool {
        request.get_annotation_value("query_instance_id").is_some()
    }

    async fn select_and_reserve(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        constraints: KvRoutingConstraints,
        affinity_active: bool,
    ) -> Result<KvRoutingOutcome<Self::Response, Self::Reservation>, Error> {
        let phase = request_phase(request);
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let is_query_only = self.is_query_only(request);
        let affinity_worker = match constraints.pin {
            Some(KvRoutePin::Affinity(target)) => affinity_worker(target),
            Some(KvRoutePin::Explicit(_)) | None => None,
        };
        let mut selection = select_request(
            self,
            request,
            phase,
            is_query_only,
            affinity_worker,
            constraints.allowed_worker_ids,
        )
        .await?;

        if is_query_only {
            return Ok(KvRoutingOutcome::LocalResponse(query_only_response(
                self, request, phase, &selection,
            )));
        }

        let guard = track_selection(self, request, &mut selection).await?;
        let dispatch_mode = if affinity_active {
            KvDispatchMode::Exact
        } else {
            KvDispatchMode::AllowFallback
        };
        Ok(KvRoutingOutcome::Dispatch(selected_route(
            self,
            request,
            selection,
            guard,
            route_guard,
            phase,
            dispatch_mode,
        )))
    }

    fn apply_target(&self, request: &mut PreprocessedRequest, target: AffinityTarget) {
        request.routing_mut().dp_rank = target.dp_rank;
    }
}

pub(super) fn selected_route(
    chooser: &Arc<KvRouter>,
    request: &SingleIn<PreprocessedRequest>,
    selection: WorkerSelection,
    guard: RequestGuard,
    route_guard: StageGuard,
    phase: RequestPhase,
    dispatch_mode: KvDispatchMode,
) -> KvSelectedRoute<Annotated<LLMEngineOutput>, TokenRouteReservation> {
    warn_if_output_replay_annotation_ignored(chooser, request, &selection);
    let target = AffinityTarget {
        worker_id: selection.instance_id,
        dp_rank: Some(selection.dp_rank),
    };
    let dispatch_span = tracing::info_span!(
        "kv_router.route_request",
        request_id = %request.context().id(),
        worker_id = selection.instance_id,
        dp_rank = selection.dp_rank,
        overlap_blocks = selection.overlap_amount,
        phase = ?phase,
    );
    let reservation = TokenRouteReservation::new(
        guard,
        route_guard,
        request.context().id().to_string(),
        phase,
    );
    KvSelectedRoute::new(target, reservation, dispatch_mode, dispatch_span)
}

fn request_phase(request: &PreprocessedRequest) -> RequestPhase {
    request
        .tracker
        .as_ref()
        .map(|tracker| tracker.phase())
        .unwrap_or(RequestPhase::Aggregated)
}

pub(super) async fn select_request(
    chooser: &Arc<KvRouter>,
    request: &SingleIn<PreprocessedRequest>,
    phase: RequestPhase,
    is_query_only: bool,
    affinity_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<std::collections::HashSet<u64>>,
) -> Result<WorkerSelection, Error> {
    let context_id = request.context().id().to_string();
    let policy_class = request.metadata().get("policy-class").cloned();
    let routing_parts = RoutingRequestParts::new(request);
    let request_context = request.context().clone();
    let mut selection_future = Box::pin(async {
        chooser
            .select_worker(
                &context_id,
                request,
                routing_parts,
                phase,
                is_query_only,
                SelectionOptions {
                    affinity_worker,
                    allowed_worker_ids,
                    policy_class,
                },
            )
            .instrument(tracing::info_span!("kv_router.select_worker"))
            .await
    });
    let selection_result = tokio::select! {
        biased;

        _ = request_context.stopped() => None,
        result = &mut selection_future => Some(result),
    };
    drop(selection_future);

    match selection_result {
        Some(result) => result,
        None => {
            if !is_query_only && let Err(error) = chooser.free(&context_id).await {
                tracing::warn!(
                    request_id = %context_id,
                    %error,
                    "Failed to free scheduler state after cancellation during worker selection"
                );
            }
            Err(cancelled_error(&context_id))
        }
    }
}

pub(super) async fn track_selection(
    chooser: &Arc<KvRouter>,
    request: &SingleIn<PreprocessedRequest>,
    selection: &mut WorkerSelection,
) -> Result<RequestGuard, Error> {
    let context_id = request.context().id().to_string();
    let request_context = request.context().clone();
    let routing_parts = RoutingRequestParts::new(request);
    let block_size = chooser.block_size() as usize;
    let mut guard = RequestGuard::new(
        chooser.clone(),
        context_id.clone(),
        request,
        selection.scheduler_tracked,
    );

    let record_result: Result<(), Error> = async {
        if chooser.indexer().records_routing_decisions() {
            let worker = WorkerWithDpRank::new(selection.instance_id, selection.dp_rank);
            let record_result = if let Some(hashes) = selection.routing_hashes.take() {
                cancel_on_stop(
                    request_context.as_ref(),
                    &context_id,
                    chooser.record_routing_decision_hashes(hashes, worker),
                )
                .await?
            } else {
                let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
                let mut tokens_with_hashes =
                    TokensWithHashes::new(routing_parts.token_ids.to_vec(), chooser.block_size())
                        .with_is_eagle(chooser.is_eagle());
                if let Some(infos) = routing_parts.block_mm_infos {
                    tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                }
                if let Some(lora_name) = lora_name {
                    tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                }
                cancel_on_stop(
                    request_context.as_ref(),
                    &context_id,
                    chooser.record_routing_decision(tokens_with_hashes, worker),
                )
                .await?
            };
            if let Err(error) = record_result {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = selection.instance_id,
                    dp_rank = selection.dp_rank,
                    error = %error,
                    "Failed to record routing decision"
                );
            }
        }

        if let Some(ref tracker) = request.tracker {
            let isl_blocks = routing_parts.token_ids.len().div_ceil(block_size);
            tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
            tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
            tracker.record_worker(
                selection.instance_id,
                Some(selection.dp_rank),
                chooser.worker_type(),
            );
            tracker.record_router_queue_depth(chooser.pending_count());
            if let Some(hit_rate) = tracker.kv_hit_rate() {
                guard.request_metrics().kv_hit_rate.observe(hit_rate);
            }
        }
        guard
            .request_metrics()
            .input_sequence_tokens
            .observe(request.token_ids.len() as f64);
        Ok(())
    }
    .await;

    if let Err(error) = record_result {
        guard.abort().await;
        return Err(error);
    }
    Ok(guard)
}

fn query_only_response(
    chooser: &Arc<KvRouter>,
    request: &SingleIn<PreprocessedRequest>,
    phase: RequestPhase,
    selection: &WorkerSelection,
) -> ManyOut<Annotated<LLMEngineOutput>> {
    let routing_parts = RoutingRequestParts::new(request);
    if let Some(ref tracker) = request.tracker {
        let isl_blocks = routing_parts
            .token_ids
            .len()
            .div_ceil(chooser.block_size() as usize);
        tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
        tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
        tracker.record_worker(
            selection.instance_id,
            Some(selection.dp_rank),
            chooser.worker_type(),
        );
        tracker.record_router_queue_depth(chooser.pending_count());
    }
    RouterRequestMetrics::from_component(chooser.client().endpoint.component())
        .input_sequence_tokens
        .observe(request.token_ids.len() as f64);
    let stream_context = request.context().clone();
    let worker_id_info = request
        .tracker
        .as_ref()
        .and_then(|tracker| tracker.get_worker_info());

    tracing::trace!(
        ?phase,
        worker_id = selection.instance_id,
        ?worker_id_info,
        "Returning worker selection (query-only mode)"
    );

    let output = LLMEngineOutput {
        routing_data: Some(RoutingData {
            worker_id: worker_id_info,
            token_ids: Some(request.token_ids.clone()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let response = Annotated::from_data(output);
    ResponseStream::new(Box::pin(stream::iter(vec![response])), stream_context)
}

fn warn_if_output_replay_annotation_ignored(
    chooser: &Arc<KvRouter>,
    request: &SingleIn<PreprocessedRequest>,
    selection: &WorkerSelection,
) {
    let Some(replay_key) = request.get_annotation_value(OUTPUT_REPLAY_ID_ANNOTATION_KEY) else {
        return;
    };
    let consumes_replay = chooser
        .workers_with_configs
        .borrow()
        .get(&selection.instance_id)
        .and_then(|config| {
            config
                .get_engine_specific::<bool>(OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY)
                .ok()
                .flatten()
        })
        .unwrap_or(false);
    if consumes_replay {
        return;
    }

    tracing::warn!(
        replay_key,
        worker_id = selection.instance_id,
        dp_rank = selection.dp_rank,
        "request has output token replay annotation but selected worker has not declared replay-token consumption"
    );
}

fn affinity_worker(target: AffinityTarget) -> Option<WorkerWithDpRank> {
    target
        .dp_rank
        .map(|rank| WorkerWithDpRank::new(target.worker_id, rank))
}
