// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kv_router::{FindBestMatchAdmission, routing_host::kv_selection::SelectionOutcome};
use crate::local_model::runtime_config::{
    CHAT_REQUEST_CAPABILITY, CHAT_REQUEST_EXTRA_ARGS_KEY,
    CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY,
};

/// Give the client's chat request to a worker that advertises `CHAT_REQUEST_CAPABILITY`, in
/// `extra_args`, in place of the media fields. Every other worker receives only the tokenized
/// request, and the snapshot stays on the request for any later dispatch.
///
/// A migration retry that replays earlier output appends those tokens to `token_ids`; the count
/// is passed as `CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY` so a worker that serves from the
/// chat request can tell it is being asked to continue, not to start.
pub(super) fn attach_chat_request(request: &mut PreprocessedRequest, wants_chat_request: bool) {
    if !wants_chat_request {
        return;
    }
    let Some(snapshot) = request.chat_request.as_ref() else {
        return;
    };
    let value = match snapshot.to_value() {
        Ok(value) => value,
        Err(error) => {
            tracing::error!(%error, "could not serialize the chat request for a chat-request worker");
            return;
        }
    };
    let replayed_tokens = request
        .token_ids
        .len()
        .saturating_sub(snapshot.prompt_tokens());
    let extra_args = request
        .extra_args
        .get_or_insert_with(|| serde_json::Value::Object(Default::default()));
    let Some(extra_args) = extra_args.as_object_mut() else {
        tracing::error!("extra_args is not a JSON object; not attaching the chat request");
        return;
    };
    extra_args.insert(CHAT_REQUEST_EXTRA_ARGS_KEY.to_string(), value);
    if replayed_tokens > 0 {
        extra_args.insert(
            CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY.to_string(),
            serde_json::json!(replayed_tokens),
        );
    } else {
        // A stale marker from an earlier dispatch would tell the worker to continue a fresh
        // request. The snapshot was just re-attached for this dispatch, so remove it when this
        // attempt has no replayed tokens.
        extra_args.remove(CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY);
    }
    // The chat request already carries any media and the worker serves from it, so the media
    // fields would only double the request-plane frame. They are cleared together: the UUID map
    // is meaningless without the data it is aligned to.
    request.multi_modal_data = None;
    request.multi_modal_uuids = None;
}

impl RoutingHost {
    #[allow(clippy::too_many_arguments)]
    async fn select_request_outcome(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        affinity_target: Option<AffinityTarget>,
        planned_worker: Option<WorkerWithDpRank>,
        admission: FindBestMatchAdmission,
        budget: &CleanupBudget,
    ) -> Result<SelectionOutcome, Error> {
        let context_id = request.context().id().to_string();
        let staged_kv = StagedKv::for_request(request.content());
        let policy_class = request.metadata().get("policy-class").cloned();
        let session_context = request
            .agent_context
            .as_ref()
            .map(to_worker_selection_session_context);
        let routing_parts = RoutingRequestParts::new(request);
        let request_context = request.context().clone();
        let selection_future = self
            .select_worker_outcome(
                &context_id,
                request,
                routing_parts,
                phase,
                is_query_only,
                SelectionOptions {
                    pinned_target: match self.session_affinity_mode {
                        SessionAffinityMode::Hard => affinity_target,
                        SessionAffinityMode::Soft => None,
                    },
                    affinity_target: match self.session_affinity_mode {
                        SessionAffinityMode::Hard => None,
                        SessionAffinityMode::Soft => affinity_target,
                    },
                    planned_worker,
                    policy_class,
                    session_context,
                    admission,
                },
            )
            .instrument(tracing::info_span!("kv_router.select_worker"));

        await_with_cleanup_policy(
            request_context.as_ref(),
            phase,
            staged_kv,
            "kv.select_worker",
            budget,
            selection_future,
        )
        .await?
    }

    async fn select_request(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        affinity_target: Option<AffinityTarget>,
        budget: &CleanupBudget,
    ) -> Result<WorkerSelection, Error> {
        self.select_request_outcome(
            request,
            phase,
            is_query_only,
            affinity_target,
            None,
            FindBestMatchAdmission::WithAdmission,
            budget,
        )
        .await?
        .into_result()
    }

    pub(super) async fn select_with_affinity(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        budget: &CleanupBudget,
    ) -> Result<(WorkerSelection, Option<Hold>), Error> {
        self.validate_explicit_worker(request.content(), phase)?;
        let select = || {
            self.select_with_session_affinity(request, phase, is_query_only, budget, |target| {
                self.select_request(request, phase, is_query_only, target, budget)
            })
        };
        if is_query_only {
            return select().await;
        }
        self.select_with_request_lifecycle(request, phase, select)
            .await
    }

    /// Claim or begin the classifier lifecycle for `request`, run the selection
    /// future built by `select` under it, and attach it to the selection.
    /// Selection failures either park the lifecycle for a migration retry or
    /// abort it with the cause.
    ///
    /// `select` is a constructor rather than a future so the selection future
    /// lives in exactly one slot: taking it by value gave the caller's future a
    /// second copy, and in debug builds that doubled footprint overflowed the
    /// test-thread stack under `block_on`.
    async fn select_with_request_lifecycle<Fut>(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        select: impl FnOnce() -> Fut,
    ) -> Result<(WorkerSelection, Option<Hold>), Error>
    where
        Fut: Future<Output = Result<(WorkerSelection, Option<Hold>), Error>>,
    {
        // Decode/aggregated routing owns the logical request's classifier.
        // A retry can run prefill again with the same MigrationState; that hop
        // must leave the parked decode lifecycle for the decode host to resume.
        match phase {
            RequestPhase::Prefill => return select().await,
            RequestPhase::Decode | RequestPhase::Aggregated => {}
        }
        let mut lifecycle = request
            .migration_state
            .as_ref()
            .and_then(|state| state.take_request_lifecycle());
        if lifecycle.is_none() {
            lifecycle = self
                .kv_router()
                .begin_request_lifecycle(request.context().id())
                .map_err(|error| classifier_failure_response(request.context().id(), &error))?
                .map(Box::new);
        }

        let (mut selection, affinity) = match select().await {
            Ok(selection) => selection,
            Err(error) => {
                if let Some(mut lifecycle) = lifecycle.take() {
                    if let Some(classifier_error) = classification_failure(&error) {
                        lifecycle.abort(Some(classifier_abort_error(classifier_error)));
                        return Err(classifier_failure_response(
                            request.context().id(),
                            classifier_error,
                        ));
                    }
                    if crate::migration::is_migratable(error.as_ref())
                        && let Some(state) = request.migration_state.as_ref()
                    {
                        lifecycle.prepare_retry();
                        state.store_request_lifecycle(lifecycle);
                    } else {
                        lifecycle.abort(Some(
                            crate::protocols::common::preprocessor::owned_abort_error(
                                error.as_ref(),
                            ),
                        ));
                    }
                }
                return Err(error);
            }
        };
        if let Some(lifecycle) = lifecycle.as_mut() {
            lifecycle.selected(selection.worker);
        }
        selection.request_lifecycle = lifecycle;
        Ok((selection, affinity))
    }

    fn route_signals(&self, selection: &WorkerSelection) -> RoutePlanSignals {
        let total_kv_blocks = match selection.selected_worker_load {
            Some(load) => load
                .total_kv_blocks
                .and_then(|blocks| blocks.try_into().ok()),
            None => self
                .kv_router()
                .workers_with_configs
                .borrow()
                .get(&selection.worker.worker_id)
                .and_then(WorkerConfigLike::total_kv_blocks),
        };
        RoutePlanSignals {
            worker: selection.worker,
            overlap_blocks: selection.overlap_amount,
            cached_tokens: selection.cached_tokens,
            potential_decode_blocks: selection.potential_decode_blocks,
            total_kv_blocks,
            logit: selection.logit,
        }
    }

    pub(crate) async fn preview_kv_route(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
    ) -> Result<RoutePreview, Error> {
        // The conditional route's first stage. The budget travels with the
        // preview into the plan and on into dispatch, so the whole route shares
        // one deadline.
        let budget = CleanupBudget::default();
        if self.kv_router_if_enabled().is_none() {
            return Err(anyhow::anyhow!("KV route previews require KV routing"));
        }
        self.validate_explicit_worker(request.content(), phase)?;

        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let (outcome, _) = self
            .select_with_session_affinity(request, phase, true, &budget, |target| {
                self.select_request_outcome(
                    request,
                    phase,
                    true,
                    target,
                    None,
                    FindBestMatchAdmission::WithoutAdmission,
                    &budget,
                )
            })
            .await?;
        let selection = outcome.into_result()?;
        let signals = self.route_signals(&selection);
        drop(route_guard);
        Ok(RoutePreview {
            request_id: request.context().id().to_string(),
            phase,
            signals,
            budget,
        })
    }

    pub(crate) async fn plan_kv_route_from_preview(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        preview: RoutePreview,
    ) -> Result<RoutePlan, Error> {
        // Inherited, not restarted: this stage continues the route the preview
        // opened.
        let budget = preview.budget;
        if self.kv_router_if_enabled().is_none() {
            return Err(anyhow::anyhow!("KV route plans require KV routing"));
        }
        if request.context().id() != preview.request_id {
            return Err(anyhow::anyhow!(
                "KV route preview belongs to request {}, not {}",
                preview.request_id,
                request.context().id(),
            ));
        }

        let phase = preview.phase;
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let planned_worker = preview.signals.worker;
        let select = || {
            self.select_with_session_affinity(request, phase, false, &budget, |target| {
                let budget = &budget;
                async move {
                    self.select_request_outcome(
                        request,
                        phase,
                        false,
                        target,
                        Some(planned_worker),
                        FindBestMatchAdmission::WithAdmission,
                        budget,
                    )
                    .await?
                    .into_result()
                }
            })
        };
        let (mut selection, affinity) = self
            .select_with_request_lifecycle(request, phase, select)
            .await?;
        let signals = self.route_signals(&selection);
        drop(route_guard);
        Ok(RoutePlan {
            signals,
            cleanup: KvRequestCleanup::new(
                Arc::clone(self.kv_router()),
                request.context().id().to_string(),
                selection.worker,
                selection.booking.take(),
            ),
            selection,
            affinity,
            budget,
        })
    }

    pub(crate) async fn dispatch_kv_plan(
        &self,
        request: SingleIn<PreprocessedRequest>,
        plan: RoutePlan,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let RoutePlan {
            mut selection,
            cleanup,
            mut affinity,
            budget,
            ..
        } = plan;
        let selected_target = route_target(selection.worker);
        let guard = match self
            .track_planned_selection(&request, &mut selection, cleanup, &budget)
            .await
        {
            Ok(guard) => guard,
            Err(error) => return Err(error),
        };
        let stream = match self
            .dispatch_selection(request, selection, guard, &budget)
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && !self.affinity_target_is_valid(selected_target)
                    && let Some(operation) = affinity.take()
                {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        self.bind_affinity(affinity, selected_target, stream)
    }

    pub(crate) async fn prefill_worker_busy(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        threshold: f64,
    ) -> Result<bool, Error> {
        // A local budget is sound here and only here: the probe is pinned to
        // `RequestPhase::Prefill`, which never selects `DispatchWhenStopped`, so
        // nothing it runs can arm or spend a budget.
        let budget = CleanupBudget::default();
        if self.kv_router_if_enabled().is_none() {
            return Err(anyhow::anyhow!("prefill load probe requires KV routing"));
        }

        let (outcome, _) = self
            .select_with_session_affinity(request, RequestPhase::Prefill, true, &budget, |target| {
                self.select_request_outcome(
                    request,
                    RequestPhase::Prefill,
                    true,
                    target,
                    None,
                    FindBestMatchAdmission::WithoutAdmission,
                    &budget,
                )
            })
            .await?;
        match outcome {
            SelectionOutcome::Routed(selection) => selection
                .selected_worker_load
                .map(|load| load.prefill_load_exceeds(threshold))
                .ok_or_else(|| anyhow::anyhow!("advisory prefill selection returned no load")),
            SelectionOutcome::QueueRejected(_) => Ok(true),
        }
    }

    pub(super) async fn track_selection(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        phase: RequestPhase,
        is_query_only: bool,
        budget: &CleanupBudget,
    ) -> Result<RequestGuard, Error> {
        self.track_selection_with_cleanup(request, selection, phase, is_query_only, None, budget)
            .await
    }

    async fn track_planned_selection(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        cleanup: KvRequestCleanup,
        budget: &CleanupBudget,
    ) -> Result<RequestGuard, Error> {
        let phase = request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated);
        self.track_selection_with_cleanup(request, selection, phase, false, Some(cleanup), budget)
            .await
    }

    async fn track_selection_with_cleanup(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        phase: RequestPhase,
        is_query_only: bool,
        cleanup: Option<KvRequestCleanup>,
        budget: &CleanupBudget,
    ) -> Result<RequestGuard, Error> {
        let context_id = request.context().id().to_string();
        let staged_kv = StagedKv::for_request(request.content());
        let request_context = request.context().clone();
        let routing_parts = RoutingRequestParts::new(request);
        let chooser = self.kv_router();
        let block_size = chooser.block_size() as usize;
        let selected_worker = selection.worker;
        let mut guard = match cleanup {
            Some(cleanup) => RequestGuard::new_kv_with_cleanup(
                self.request_metrics.clone(),
                cleanup,
                request,
                selection.request_lifecycle.take(),
            ),
            None => RequestGuard::new_kv(
                Arc::clone(chooser),
                self.request_metrics.clone(),
                context_id.clone(),
                selected_worker,
                selection.booking.take(),
                request,
                selection.request_lifecycle.take(),
            ),
        };

        let record_result: Result<(), Error> = async {
            if !is_query_only && chooser.indexer().records_routing_decisions() {
                let worker = selected_worker;
                let hashes = if let Some(hashes) = selection.routing_hashes.take() {
                    hashes
                } else {
                    let routing = request.routing.as_ref();
                    let mut tokens_with_hashes = TokensWithHashes::new(
                        routing_parts.token_ids.to_vec(),
                        chooser.block_size(),
                    )
                    .with_is_eagle(chooser.is_eagle());
                    if let Some(infos) = routing_parts.block_mm_infos {
                        tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                    }
                    if let Some(lora_name) = routing.and_then(|r| r.lora_name.clone()) {
                        tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                    }
                    if let Some(cache_namespace) = routing.and_then(|r| r.cache_namespace.clone()) {
                        tokens_with_hashes =
                            tokens_with_hashes.with_cache_namespace(cache_namespace);
                    }
                    let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
                    let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
                    dynamo_kv_router::indexer::RoutingDecisionHashes {
                        local_hashes,
                        sequence_hashes,
                    }
                };
                let record_result = if guard.has_approximate_lru() {
                    await_with_cleanup_policy(
                        request_context.as_ref(),
                        phase,
                        staged_kv,
                        "kv.record_routing_decision",
                        budget,
                        guard.acquire_approximate_lru(hashes),
                    )
                    .await?
                } else {
                    await_with_cleanup_policy(
                        request_context.as_ref(),
                        phase,
                        staged_kv,
                        "kv.record_routing_decision",
                        budget,
                        chooser.record_routing_decision_hashes(hashes, worker),
                    )
                    .await?
                };
                if let Err(error) = record_result {
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = selection.worker.worker_id,
                        dp_rank = selection.worker.dp_rank,
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
                    selection.worker.worker_id,
                    Some(selection.worker.dp_rank),
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
            guard.abort_with_error(Some(error.as_ref())).await;
            return Err(error);
        }
        Ok(guard)
    }

    pub(super) async fn dispatch_selection(
        &self,
        request: SingleIn<PreprocessedRequest>,
        selection: WorkerSelection,
        mut guard: RequestGuard,
        budget: &CleanupBudget,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let route_trace_context = get_route_trace_context(&request);
        let phase = request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated);
        let staged_kv = StagedKv::for_request(request.content());
        let phase_label = phase.to_string();
        guard.start_dispatch(&phase_label);
        self.warn_if_output_replay_annotation_ignored(&request, &selection);

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(selection.worker.dp_rank);
        backend_input.kv_hint = selection.kv_hint;
        let wants_chat_request = self
            .kv_router()
            .worker_supports_capability(selection.worker.worker_id, CHAT_REQUEST_CAPABILITY);
        attach_chat_request(&mut backend_input, wants_chat_request);
        let updated_request = context.map(|_| backend_input);
        guard.record_prefill_start(updated_request.content());

        let dispatch = self
            .inner
            .dispatch_kv_admitted(updated_request, selection.worker.worker_id);
        let route_span = tracing::info_span!(
            target: "request_span",
            "kv_router.route_request",
            otel.kind = "client",
            request_id = %context_id,
            worker_id = tracing::field::Empty,
            dp_rank = selection.worker.dp_rank,
            overlap_blocks = selection.overlap_amount,
            phase = ?phase,
            "request.attempt" = tracing::field::Empty,
            "request.outcome" = tracing::field::Empty,
            "migration.is_retry" = tracing::field::Empty,
            "migration.reason" = tracing::field::Empty,
            "migration.from_worker_id" = tracing::field::Empty,
            "migration.tokens_completed" = tracing::field::Empty,
            "cancellation.signal" = tracing::field::Empty,
            "error.type" = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
            otel.status_description = tracing::field::Empty,
        );
        record_route_span_start(
            &route_span,
            route_trace_context.as_deref(),
            selection.worker.worker_id,
        );
        let dispatch_result = await_with_cleanup_policy(
            request_context.as_ref(),
            phase,
            staged_kv,
            "kv.dispatch",
            budget,
            dispatch.instrument(route_span.clone()),
        )
        .await
        .and_then(|result| result);
        let response_stream = match dispatch_result {
            Ok(stream) => stream,
            Err(error) => {
                record_route_error(&route_span, error.as_ref());
                let typed_error = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<DynamoError>().cloned());
                guard.record_migration_failure(typed_error);
                if !crate::migration::is_migratable(error.as_ref())
                    || !guard.release_for_retry().await
                {
                    guard.abort_with_error(Some(error.as_ref())).await;
                }
                return Err(error);
            }
        };

        guard.mark_dispatched();
        Ok(wrap_route_span(
            into_monitored_response(response_stream, guard),
            route_span,
        ))
    }

    fn warn_if_output_replay_annotation_ignored(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &WorkerSelection,
    ) {
        let Some(replay_key) = request.get_annotation_value(OUTPUT_REPLAY_ID_ANNOTATION_KEY) else {
            return;
        };
        let consumes_replay = self
            .kv_router()
            .workers_with_configs
            .borrow()
            .get(&selection.worker.worker_id)
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
            worker_id = selection.worker.worker_id,
            dp_rank = selection.worker.dp_rank,
            "request has output token replay annotation but selected worker has not declared replay-token consumption"
        );
    }

    pub(super) async fn select_and_dispatch_kv_prefill<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let budget = CleanupBudget::default();
        let phase = RequestPhase::Prefill;
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let (mut selection, mut operation) = self
            .select_with_affinity(&request, phase, is_query_only, &budget)
            .await?;
        let mut guard = match self
            .track_selection(&request, &mut selection, phase, is_query_only, &budget)
            .await
        {
            Ok(guard) => guard,
            Err(error) => return Err(error),
        };
        let selected_target = route_target(selection.worker);
        let metadata = match prepare(&mut request, selected_target) {
            Ok(metadata) => metadata,
            Err(error) => {
                guard.abort_with_error(Some(error.as_ref())).await;
                return Err(error);
            }
        };
        drop(route_guard);
        let stream = match self
            .dispatch_selection(request, selection, guard, &budget)
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && !self.affinity_target_is_valid(selected_target)
                    && let Some(operation) = operation.take()
                {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        Ok((
            metadata,
            self.bind_affinity(operation, selected_target, stream)?,
        ))
    }
}

#[cfg(test)]
mod chat_request_tests {
    use super::*;
    use crate::protocols::common::preprocessor::ChatRequestSnapshot;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

    const PROMPT: [u32; 3] = [1, 2, 3];

    fn request_with_snapshot(extra_args: Option<serde_json::Value>) -> PreprocessedRequest {
        let chat: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let mut request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(PROMPT.to_vec())
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .extra_args(extra_args)
            .build()
            .unwrap();
        request.chat_request = Some(ChatRequestSnapshot::new(Arc::new(chat), PROMPT.len()));
        request.multi_modal_data = Some(Default::default());
        request.multi_modal_uuids = Some(Default::default());
        request
    }

    #[test]
    fn a_worker_that_asks_receives_the_chat_request() {
        let mut request = request_with_snapshot(Some(serde_json::json!({"keep": 1})));
        attach_chat_request(&mut request, true);
        let extra_args = request.extra_args.as_ref().unwrap();
        assert_eq!(extra_args["keep"], 1);
        assert_eq!(
            extra_args[CHAT_REQUEST_EXTRA_ARGS_KEY]["messages"][0]["content"],
            "hi"
        );
        assert!(
            extra_args
                .get(CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY)
                .is_none()
        );
        assert!(request.multi_modal_data.is_none());
        assert!(request.multi_modal_uuids.is_none());
    }

    #[test]
    fn a_worker_that_does_not_ask_receives_only_tokens() {
        let mut request = request_with_snapshot(None);
        attach_chat_request(&mut request, false);
        assert!(request.extra_args.is_none());
        assert!(request.multi_modal_data.is_some());
        assert!(request.multi_modal_uuids.is_some());
        // Kept for a later dispatch of this request to a worker that does ask.
        assert!(request.chat_request.is_some());
    }

    #[test]
    fn a_migration_retry_reports_the_replayed_output() {
        let mut request = request_with_snapshot(None);
        Arc::make_mut(&mut request.token_ids).extend([7, 8]);
        attach_chat_request(&mut request, true);
        let extra_args = request.extra_args.as_ref().unwrap();
        assert_eq!(extra_args[CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY], 2);
    }

    #[test]
    fn a_fresh_dispatch_clears_a_stale_replayed_marker() {
        // The same request object is dispatched again after a migration; the marker from the
        // earlier attempt must not survive to make the worker continue a fresh request.
        let mut request = request_with_snapshot(Some(serde_json::json!({
            CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY: 5,
        })));
        attach_chat_request(&mut request, true);
        let extra_args = request.extra_args.as_ref().unwrap();
        assert!(
            extra_args
                .get(CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY)
                .is_none(),
            "a request with no replayed tokens must not carry the marker"
        );
        assert!(extra_args.get(CHAT_REQUEST_EXTRA_ARGS_KEY).is_some());
    }

    #[test]
    fn non_object_extra_args_are_left_alone() {
        let mut request = request_with_snapshot(Some(serde_json::json!("opaque")));
        attach_chat_request(&mut request, true);
        assert_eq!(request.extra_args, Some(serde_json::json!("opaque")));
        assert!(request.multi_modal_data.is_some());
    }
}
