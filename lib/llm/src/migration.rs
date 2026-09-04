// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::error::Error as StdError;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Error, Result};
use futures::{stream, stream::StreamExt};

use crate::{
    http::service::metrics::Metrics,
    model_card::ModelDeploymentCard,
    protocols::{
        TokenIdType,
        common::{
            extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
            llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest},
        },
    },
};

use dynamo_kv_router::KvSchedulerError;
use dynamo_protocols::types::CompletionUsage;
use dynamo_runtime::engine::Data;
use dynamo_runtime::error::{self, BackendError, DynamoError, ErrorType};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, Operator, PipelineOperator,
    ResponseStream, ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;

/// Accessors the migration RetryManager needs from a response chunk.
/// `token_ids` lets it replay already-delivered tokens; `worker_trace_link`
/// lets it stamp the failed worker's span onto the next attempt's
/// `migration_link`.
pub(crate) trait HasTokenIds {
    fn token_ids(&self) -> &[TokenIdType];
    fn worker_trace_link(&self) -> Option<&crate::protocols::common::preprocessor::TraceLink>;
    /// The worker-reported usage carried by this chunk, if any. On a retried
    /// stream the worker counts the replayed tokens as prompt; the migrator
    /// corrects that before the chunk reaches the postprocessor.
    fn completion_usage_mut(&mut self) -> Option<&mut CompletionUsage>;
    /// The other worker sets of the model that carry this response type.
    fn fallback_targets(source: &dyn MigrationFallbackSource) -> Vec<MigrationTarget<Self>>
    where
        Self: Sized;
}

impl HasTokenIds for BackendOutput {
    fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }
    fn worker_trace_link(&self) -> Option<&crate::protocols::common::preprocessor::TraceLink> {
        self.worker_trace_link.as_ref()
    }
    fn completion_usage_mut(&mut self) -> Option<&mut CompletionUsage> {
        self.completion_usage.as_mut()
    }
    fn fallback_targets(source: &dyn MigrationFallbackSource) -> Vec<MigrationTarget<Self>> {
        source.backend_output_targets()
    }
}

impl HasTokenIds for LLMEngineOutput {
    fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }
    fn worker_trace_link(&self) -> Option<&crate::protocols::common::preprocessor::TraceLink> {
        self.worker_trace_link.as_ref()
    }
    fn completion_usage_mut(&mut self) -> Option<&mut CompletionUsage> {
        self.completion_usage.as_mut()
    }
    fn fallback_targets(source: &dyn MigrationFallbackSource) -> Vec<MigrationTarget<Self>> {
        source.llm_engine_output_targets()
    }
}

/// A token-level engine in another worker set of the same model. A request
/// that loses its worker continues here once its own worker set is empty.
pub struct MigrationTarget<Resp> {
    pub namespace: String,
    pub engine: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    /// The target set's own lookup, adopted by a request that moves there so
    /// a later failure is judged from the set it is then running in.
    pub fallback: Option<Arc<dyn MigrationFallbackSource>>,
}

impl<Resp> Clone for MigrationTarget<Resp> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            engine: self.engine.clone(),
            fallback: self.fallback.clone(),
        }
    }
}

/// Looks up the other worker sets a migrating request can continue on.
///
/// Every worker set owns its own pipeline, so a retry issued inside one
/// pipeline only reaches that set's workers. When the failed worker was the
/// last one in its set, which is what the tail of a rolling update looks
/// like, the retry needs the sets that are still serving. The lookup runs at
/// retry time so it sees the sets that exist then, not the ones that existed
/// when the pipeline was built.
pub trait MigrationFallbackSource: Send + Sync {
    fn backend_output_targets(&self) -> Vec<MigrationTarget<BackendOutput>>;
    fn llm_engine_output_targets(&self) -> Vec<MigrationTarget<LLMEngineOutput>>;
}

/// The request's own worker set has no worker left to dispatch to. The KV
/// scheduler reports this as `NoEndpoints`; the round-robin, random and
/// load-based routers report an empty pool as `Unavailable`.
fn has_no_worker_left(err: &Error) -> bool {
    err.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<KvSchedulerError>(),
            Some(KvSchedulerError::NoEndpoints)
        )
    }) || error::match_error_chain(err.as_ref(), &[ErrorType::Unavailable], &[])
}

/// Check if an error chain indicates the request should be migrated.
fn is_migratable(err: &(dyn StdError + 'static)) -> bool {
    const MIGRATABLE: &[ErrorType] = &[
        ErrorType::CannotConnect,
        ErrorType::Disconnected,
        ErrorType::ConnectionTimeout,
        ErrorType::Backend(BackendError::EngineShutdown),
    ];
    const NON_MIGRATABLE: &[ErrorType] = &[ErrorType::Cancelled, ErrorType::ResourceExhausted];
    error::match_error_chain(err, MIGRATABLE, NON_MIGRATABLE)
}

pub struct Migration {
    migration_limit: u32,
    max_seq_len: Option<u32>,
    model_name: Arc<String>,
    metrics: Arc<Metrics>,
    fallback: Option<Arc<dyn MigrationFallbackSource>>,
}

impl Migration {
    pub fn new(
        migration_limit: u32,
        max_seq_len: Option<u32>,
        model_name: String,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        tracing::debug!(
            "model {} migration limit {} max_seq_len {:?}",
            model_name,
            migration_limit,
            max_seq_len
        );
        Arc::new(Self {
            migration_limit,
            max_seq_len,
            model_name: Arc::new(model_name),
            metrics,
            fallback: None,
        })
    }

    pub fn from_mdc(
        mdc: &ModelDeploymentCard,
        migration_limit: u32,
        max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        Self::from_mdc_with_fallback(mdc, migration_limit, max_seq_len, metrics, None)
    }

    /// `fallback` names the other worker sets a request may continue on when
    /// this pipeline's worker set has no workers left.
    pub fn from_mdc_with_fallback(
        mdc: &ModelDeploymentCard,
        migration_limit: u32,
        max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
        fallback: Option<Arc<dyn MigrationFallbackSource>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            migration_limit,
            max_seq_len,
            model_name: Arc::new(mdc.display_name.clone()),
            metrics,
            fallback,
        })
    }

    /// Wrap as a `PipelineOperator` over the given response type to
    /// disambiguate between the `Operator` impls on `Migration` since
    /// the response type doesn't appear in the struct.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_operator_for<Resp>(
        self: &Arc<Self>,
    ) -> Arc<
        PipelineOperator<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
        >,
    >
    where
        Resp: Data + HasTokenIds,
    {
        Operator::into_operator(self)
    }
}

#[async_trait]
impl<Resp>
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
    > for Migration
where
    Resp: Data + HasTokenIds,
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    ) -> Result<ManyOut<Annotated<Resp>>> {
        let (preprocessed_request, context) = request.transfer(());
        let engine_ctx = context.context();
        let engine_ctx_ = engine_ctx.clone();
        let session_affinity = context
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(Error::msg)?
            .map(|session_id| session_id.as_ref().clone());
        let retry_manager = RetryManager::build_with_fallback(
            engine_ctx,
            context.metadata().clone(),
            preprocessed_request,
            next,
            self.migration_limit,
            self.max_seq_len,
            self.model_name.clone(),
            self.metrics.clone(),
            session_affinity,
            self.fallback.clone(),
        )
        .await?;
        let response_stream = stream::unfold(retry_manager, move |mut retry_manager| async move {
            retry_manager
                .next()
                .await
                .map(|response| (response, retry_manager))
        })
        .fuse();
        Ok(ResponseStream::new(Box::pin(response_stream), engine_ctx_))
    }
}

struct RetryManager<Resp>
where
    Resp: Data + HasTokenIds,
{
    context: Arc<dyn AsyncEngineContext>,
    metadata: BTreeMap<String, String>,
    request: PreprocessedRequest,
    session_affinity: Option<SessionAffinityId>,
    next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    next_stream: Option<ManyOut<Annotated<Resp>>>,
    /// Other worker sets of the model, consulted when `next_generate` has no
    /// eligible worker left for a retry.
    fallback: Option<Arc<dyn MigrationFallbackSource>>,
    /// False when the configured limit, or the request, rules migration out;
    /// then no worker set other than the request's own is ever tried.
    migration_enabled: bool,
    /// True while a broken stream is being replaced, as opposed to the
    /// request's first dispatch.
    recovering_stream: bool,
    retries_left: u32,
    max_seq_len: Option<u32>,
    model_name: Arc<String>,
    metrics: Arc<Metrics>,
    /// Latest worker span pointer seen on the active stream; stamped as
    /// `migration_link` on the next retry. Populated by `track_response`.
    last_worker_link: Option<crate::protocols::common::preprocessor::TraceLink>,
    /// Prompt length of the original request. `request.token_ids` grows by
    /// every generated token so a retry can replay them; the worker serving
    /// the retry reports that longer sequence as its prompt.
    original_isl: usize,
    /// Tokens replayed as prompt on the stream currently being consumed:
    /// the amount by which that worker's `prompt_tokens` overstates the
    /// client's prompt. Zero on the first attempt.
    replayed_tokens: usize,
}

impl<Resp> RetryManager<Resp>
where
    Resp: Data + HasTokenIds,
{
    /// Test convenience: a retry manager with no other worker sets to fall back to.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub async fn build(
        context: Arc<dyn AsyncEngineContext>,
        metadata: BTreeMap<String, String>,
        preprocessed_request: PreprocessedRequest,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
        retries_left: u32,
        max_seq_len: Option<u32>,
        model_name: Arc<String>,
        metrics: Arc<Metrics>,
        session_affinity: Option<SessionAffinityId>,
    ) -> Result<Self> {
        Self::build_with_fallback(
            context,
            metadata,
            preprocessed_request,
            next,
            retries_left,
            max_seq_len,
            model_name,
            metrics,
            session_affinity,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn build_with_fallback(
        context: Arc<dyn AsyncEngineContext>,
        metadata: BTreeMap<String, String>,
        preprocessed_request: PreprocessedRequest,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
        mut retries_left: u32,
        max_seq_len: Option<u32>,
        model_name: Arc<String>,
        metrics: Arc<Metrics>,
        session_affinity: Option<SessionAffinityId>,
        fallback: Option<Arc<dyn MigrationFallbackSource>>,
    ) -> Result<Self> {
        // Disable migration for structured-output (guided-decoding) requests.
        // Inference backends initialize the guided-decoding FSM (finite state machine) fresh
        // for every new request and only advance it on newly-generated tokens, not on
        // context/prompt tokens. Migrating a partial structured-output response would replay
        // already-generated tokens as context, causing the FSM to restart from the schema
        // root and producing duplicated or nested JSON. This applies to all backends
        // (vLLM, SGLang, TRT-LLM) equally. Propagate the error cleanly instead.
        if preprocessed_request
            .sampling_options
            .guided_decoding
            .is_some()
        {
            if retries_left > 0 {
                tracing::warn!(
                    "Guided-decoding request: migration disabled — FSM state is not transferable (applies to all backends)"
                );
            }
            retries_left = 0;
        }

        if preprocessed_request.sampling_options.n.unwrap_or(1) > 1 {
            if retries_left > 0 {
                tracing::warn!(
                    "n>1 request: migration disabled - per-choice generation state is not transferable"
                );
            }
            retries_left = 0;
        }
        let original_isl = preprocessed_request.token_ids.len();
        let mut slf = Self {
            context,
            metadata,
            request: preprocessed_request,
            session_affinity,
            next_generate: next,
            next_stream: None,
            fallback,
            recovering_stream: false,
            migration_enabled: retries_left > 0
                && max_seq_len.is_none_or(|max_seq_len| original_isl <= max_seq_len as usize),
            retries_left: retries_left + 1, // +1 to account for the initial attempt
            max_seq_len,
            model_name,
            metrics,
            last_worker_link: None,
            original_isl,
            replayed_tokens: 0,
        };
        slf.new_stream().await?;
        slf.exceed_max_seq_len(0); // disable migration if prompt len > max_seq_len
        Ok(slf)
    }

    /// Keep the worker that served the failed attempt out of the next
    /// selection. The router records the selected worker on the request
    /// tracker before every dispatch.
    fn exclude_last_worker(&mut self) {
        let Some(worker_id) = self
            .request
            .tracker
            .as_ref()
            .and_then(|tracker| tracker.last_selected_worker_id())
        else {
            return;
        };
        let excluded = self
            .request
            .routing_mut()
            .excluded_worker_ids
            .get_or_insert_with(HashSet::new);
        if excluded.insert(worker_id) {
            tracing::debug!(worker_id, "excluding failed worker from migration retry");
        }
    }

    pub async fn next(&mut self) -> Option<Annotated<Resp>> {
        loop {
            // None once a failed stream was released and could not be replaced:
            // its error has been delivered and the response is over.
            let response_stream = self.next_stream.as_mut()?;
            if let Some(mut response) = response_stream.next().await {
                // Check if this is a migratable error that should trigger stream recreation.
                if let Some(err) = response.error.as_ref()
                    && is_migratable(err)
                {
                    tracing::warn!(error = %err, "Stream disconnected, recreating stream");
                    self.metrics.inc_migration_ongoing_request(&self.model_name);
                    self.exclude_last_worker();
                    self.release_failed_stream().await;
                    self.recovering_stream = true;
                    let recreated = self.new_stream().await;
                    self.recovering_stream = false;
                    if let Err(err) = recreated {
                        tracing::warn!(error = ?err, "Cannot recreate stream");
                    } else {
                        continue;
                    }
                }
                self.correct_replayed_usage(&mut response);
                self.track_response(&response);
                return Some(response);
            }
            return None;
        }
    }

    /// Run the failed stream to its end before asking for a replacement. The
    /// router frees this request's booking only when its stream is polled
    /// past the failure; re-dispatching first makes the scheduler reject the
    /// retry as a request still assigned to the dead worker. A stream that
    /// does not end promptly is dropped, which frees the booking in the
    /// background instead.
    async fn release_failed_stream(&mut self) {
        let Some(mut stream) = self.next_stream.take() else {
            return;
        };
        let drained = tokio::time::timeout(Duration::from_millis(500), async {
            while stream.next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            tracing::debug!("Failed stream did not end promptly; dropping it");
        }
    }

    async fn new_stream(&mut self) -> Result<()> {
        self.replayed_tokens = self
            .request
            .token_ids
            .len()
            .saturating_sub(self.original_isl);
        let mut response_stream: Option<Result<ManyOut<Annotated<Resp>>>> = None;
        while self.retries_left > 0 {
            self.retries_left -= 1;
            // Once any chunks have arrived from a previous attempt, stamp
            // that worker's span as `migration_link` so the next worker's
            // span renders an OTel Link back to it. Guarded so the initial
            // attempt doesn't clobber a `migration_link` set upstream.
            if let Some(link) = self.last_worker_link.as_ref() {
                self.request.migration_link = Some(link.clone());
            }
            if self.context.is_stopped() || self.context.is_killed() {
                tracing::debug!("Abort creating new stream after context is stopped or killed");
                return Err(DynamoError::builder()
                    .error_type(ErrorType::Cancelled)
                    .message(format!(
                        "Context id {} is stopped or killed",
                        self.context.id()
                    ))
                    .build()
                    .into());
            }
            match self.next_generate.generate(self.build_request()).await {
                Err(err) if is_migratable(err.as_ref()) => {
                    tracing::warn!(error = %err, "Creating new stream, retrying");
                    self.metrics.inc_migration_new_request(&self.model_name);
                    self.exclude_last_worker();
                    // Reported to the caller if this was the last retry.
                    response_stream = Some(Err(err));
                    continue;
                }
                Err(err) if has_no_worker_left(&err) => {
                    response_stream = Some(match self.continue_in_other_worker_set().await {
                        Ok(Some(stream)) => Ok(stream),
                        Ok(None) => Err(err),
                        Err(target_err) => Err(target_err),
                    });
                    break;
                }
                attempt => {
                    response_stream = Some(attempt);
                    break;
                }
            }
        }
        match response_stream {
            Some(Ok(next_stream)) => {
                self.next_stream = Some(next_stream);
                Ok(())
            }
            Some(Err(err)) => Err(err), // should propagate original error if any
            None => Err(Error::msg(
                "Migration limit exhausted", // should propagate original error if any
            )),
        }
    }

    fn build_request(&self) -> Context<PreprocessedRequest> {
        let mut request = Context::with_id_and_metadata(
            self.request.clone(),
            self.context.id().to_string(),
            self.metadata.clone(),
        );
        if let Some(session_affinity) = self.session_affinity.as_ref() {
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
        }
        self.context.link_child(request.context());
        request
    }

    /// The request's own worker set has no worker left for it. Continue on
    /// another worker set of the same model, most workers first, and send any
    /// later retry there too. A worker of a candidate set that fails to take
    /// the request is excluded and its peers tried, each attempt charged to
    /// the retry budget; once the budget is spent no further set is tried.
    /// `Ok(None)` means no set could take the request; `Err` carries a
    /// target's own verdict (cancelled, overloaded, ...), which stands.
    /// Never done when migration is disabled.
    async fn continue_in_other_worker_set(&mut self) -> Result<Option<ManyOut<Annotated<Resp>>>> {
        if !self.migration_enabled {
            return Ok(None);
        }
        let Some(source) = self.fallback.clone() else {
            return Ok(None);
        };
        // A move made on the request's first dispatch is a new-request
        // migration; a mid-stream move was already counted when the stream broke.
        let first_dispatch = !self.recovering_stream;
        for target in Resp::fallback_targets(source.as_ref()) {
            // Excluded ids name workers of the set being left behind.
            if let Some(routing) = self.request.routing.as_mut() {
                routing.excluded_worker_ids = None;
            }
            loop {
                if self.context.is_stopped() || self.context.is_killed() {
                    return Ok(None);
                }
                match target.engine.generate(self.build_request()).await {
                    Ok(stream) => {
                        tracing::info!(
                            namespace = %target.namespace,
                            "Continuing request in another worker set"
                        );
                        if first_dispatch {
                            self.metrics.inc_migration_new_request(&self.model_name);
                        }
                        self.next_generate = target.engine.clone();
                        if target.fallback.is_some() {
                            self.fallback = target.fallback.clone();
                        }
                        return Ok(Some(stream));
                    }
                    Err(err) if has_no_worker_left(&err) => {
                        tracing::debug!(
                            namespace = %target.namespace,
                            "Worker set has no worker for the request either"
                        );
                        break;
                    }
                    Err(err) if is_migratable(err.as_ref()) => {
                        if self.retries_left == 0 {
                            tracing::warn!(
                                namespace = %target.namespace,
                                error = %err,
                                "Migration budget spent while moving to another worker set"
                            );
                            return Ok(None);
                        }
                        tracing::warn!(
                            namespace = %target.namespace,
                            error = %err,
                            "Worker in the other set failed to take the request, trying its peers"
                        );
                        self.retries_left -= 1;
                        self.metrics.inc_migration_new_request(&self.model_name);
                        self.exclude_last_worker();
                    }
                    Err(err) => {
                        tracing::warn!(
                            namespace = %target.namespace,
                            error = %err,
                            "Worker set refused the request"
                        );
                        return Err(err);
                    }
                }
            }
        }
        Ok(None)
    }

    /// A worker serving a retry received the original prompt plus every
    /// token generated before the failure, and reports that as its prompt
    /// count; the postprocessor takes a worker-reported `prompt_tokens` over
    /// its own. Subtract the replayed tokens so usage reflects the client's
    /// request, and keep the cached-token detail within the corrected prompt.
    fn correct_replayed_usage(&self, response: &mut Annotated<Resp>) {
        if self.replayed_tokens == 0 {
            return;
        }
        let Some(usage) = response
            .data
            .as_mut()
            .and_then(|data| data.completion_usage_mut())
        else {
            return;
        };
        let replayed = u32::try_from(self.replayed_tokens).unwrap_or(u32::MAX);
        usage.prompt_tokens = usage.prompt_tokens.saturating_sub(replayed);
        if let Some(cached) = usage
            .prompt_tokens_details
            .as_mut()
            .and_then(|details| details.cached_tokens.as_mut())
        {
            *cached = (*cached).min(usage.prompt_tokens);
        }
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
    }

    fn track_response(&mut self, response: &Annotated<Resp>) {
        if self.retries_left == 0 {
            return;
        }
        let llm_engine_output = match response.data.as_ref() {
            Some(output) => output,
            None => return,
        };
        // Capture the worker's engine.generate span pointer so a future
        // migration retry can render an OTel Link back to it. The adapter
        // stamps this on the first non-empty chunk; subsequent chunks may
        // also carry it. Keep the most-recently-seen value.
        if let Some(link) = llm_engine_output.worker_trace_link() {
            self.last_worker_link = Some(link.clone());
        }
        let token_ids = llm_engine_output.token_ids();
        if self.exceed_max_seq_len(token_ids.len() as u32) {
            return;
        }
        if let Some(max_tokens) = self.request.stop_conditions.max_tokens {
            self.request.stop_conditions.max_tokens =
                Some(max_tokens.saturating_sub(token_ids.len() as u32));
        }
        for token_id in token_ids.iter() {
            self.request.token_ids.push(*token_id);
        }
    }

    /// Returns `true` if the tracked request token length plus `new_output_len`
    /// exceeds the configured max_seq_len, in which case migration is disabled.
    fn exceed_max_seq_len(&mut self, new_output_len: u32) -> bool {
        if let Some(max_seq_len) = self.max_seq_len {
            let total_len = self.request.token_ids.len() as u32 + new_output_len;
            if total_len > max_seq_len {
                tracing::warn!(
                    "Sequence length {} exceeds migration max_seq_len {}, \
                     disabling migration",
                    total_len,
                    max_seq_len
                );
                self.metrics
                    .inc_migration_max_seq_len_exceeded(&self.model_name);
                self.retries_left = 0; // disable migration
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::service::metrics::Metrics;
    use crate::protocols::common::timing::RequestTracker;
    use crate::protocols::common::{
        GuidedDecodingOptions, OutputOptions, SamplingOptions, StopConditions,
    };
    use dynamo_runtime::error::{DynamoError, ErrorType};
    use dynamo_runtime::pipeline::AsyncEngine;
    use dynamo_runtime::pipeline::context::Controller;
    use dynamo_runtime::protocols::maybe_error::MaybeError;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::mpsc;

    const TEST_MODEL: &str = "test-model";

    // Helper to create a mock preprocessed request
    fn create_mock_request(max_tokens: u32) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens: Some(max_tokens),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    // Helper to create mock LLM engine output
    fn create_mock_output(token_id: u32) -> Annotated<BackendOutput> {
        Annotated::from_data(BackendOutput {
            token_ids: vec![token_id],
            tokens: vec![],
            text: Some(format!("token_{token_id}")),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            stop_reason: None,
            index: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            completion_usage: None,
            engine_data: None,
            routing_data: None,
        })
    }

    #[derive(Debug, Clone)]
    enum MockBehavior {
        /// Always succeeds with all responses
        Success,
        /// Fails on first call with NoResponders error, then succeeds on subsequent calls
        FailThenSuccess,
        FailThenSuccessWithAffinity,
        /// Succeeds initially, fails mid-stream with specific error, then succeeds on retry
        MidStreamFail {
            fail_after: usize,
        },
        /// Succeeds initially, fails mid-stream, then the KV scheduler has no worker left
        MidStreamFailThenNoEndpoints {
            fail_after: usize,
        },
        /// Like MidStreamFail, but a retry is refused while the failed stream
        /// has not been run to its end, as the router's booking does
        MidStreamFailBookedUntilDrained {
            fail_after: usize,
        },
        /// Succeeds initially, fails mid-stream, then the round-robin pool is empty
        MidStreamFailThenUnavailable {
            fail_after: usize,
        },
        /// Succeeds initially, fails mid-stream with specific error, then always fails on retry attempts
        MidStreamFailAlways {
            fail_after: usize,
        },
        /// Succeeds initially, fails mid-stream, then always fails with stream error on retry attempts
        MidStreamFailAlwaysStreamError {
            fail_after: usize,
        },
        /// Always fails with NoResponders error (same as FailThenSuccess first call)
        AlwaysFail,
        /// The set has no worker at all, on every call
        AlwaysNoEndpoints,
        /// Every worker of the set is overloaded, on every call
        AlwaysOverloaded,
    }

    // Unified mock server streaming engine that can simulate different scenarios
    struct MockEngine {
        behavior: MockBehavior,
        num_responses: usize,
        token_offset: u32,
        call_count: Arc<AtomicU32>,
        context_id: String,
        /// Prompt length of `create_mock_request` (token_ids [1, 2, 3]).
        prompt_len: usize,
        excluded_worker_ids_seen: Arc<std::sync::Mutex<Vec<Option<HashSet<u64>>>>>,
        /// Set while a stream this engine returned is alive and not yet run to
        /// its end, like the router's booking of the request.
        booked: Arc<std::sync::atomic::AtomicBool>,
    }

    /// A stream that clears its booking once polled to the end or dropped.
    struct BookedStream {
        inner: std::pin::Pin<Box<dyn futures::Stream<Item = Annotated<BackendOutput>> + Send>>,
        booked: Arc<std::sync::atomic::AtomicBool>,
    }

    impl futures::Stream for BookedStream {
        type Item = Annotated<BackendOutput>;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            let polled = self.inner.as_mut().poll_next(cx);
            if matches!(polled, std::task::Poll::Ready(None)) {
                self.booked.store(false, Ordering::SeqCst);
            }
            polled
        }
    }

    impl Drop for BookedStream {
        fn drop(&mut self) {
            self.booked.store(false, Ordering::SeqCst);
        }
    }

    impl MockEngine {
        fn new(
            behavior: MockBehavior,
            num_responses: usize,
            token_offset: u32,
            context_id: String,
        ) -> Self {
            Self {
                behavior,
                num_responses,
                token_offset,
                call_count: Arc::new(AtomicU32::new(0)),
                context_id,
                prompt_len: 3,
                excluded_worker_ids_seen: Arc::new(std::sync::Mutex::new(Vec::new())),
                booked: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }
        }
    }

    #[async_trait]
    impl
        AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, anyhow::Error>
        for MockEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>> {
            let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
            if matches!(self.behavior, MockBehavior::FailThenSuccessWithAffinity) {
                let actual = request
                    .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                    .expect("session affinity context missing after migration wrapper");
                assert_eq!(actual.as_str(), "session-123");
            }
            let (preprocessed_request, context) = request.transfer(());
            self.excluded_worker_ids_seen.lock().unwrap().push(
                preprocessed_request
                    .routing
                    .as_ref()
                    .and_then(|routing| routing.excluded_worker_ids.clone()),
            );

            // Assert that the context_id matches the expected one
            assert_eq!(
                context.id().to_string(),
                self.context_id,
                "Context ID mismatch"
            );

            // Calculate how many responses we've already generated based on request token_ids
            // Initial request has [1, 2, 3], so anything beyond that are generated responses
            let initial_tokens = 3; // [1, 2, 3]
            let responses_already_generated = preprocessed_request
                .token_ids
                .len()
                .saturating_sub(initial_tokens);

            // Assert that max_tokens reflects the expected remaining tokens
            let expected_max_tokens =
                self.num_responses
                    .saturating_sub(responses_already_generated) as u32;
            assert_eq!(
                preprocessed_request.stop_conditions.max_tokens,
                Some(expected_max_tokens),
                "max_tokens should be {} but got {:?}",
                expected_max_tokens,
                preprocessed_request.stop_conditions.max_tokens
            );

            match &self.behavior {
                MockBehavior::Success => {
                    // Always succeed with remaining responses
                    self.send_responses(responses_already_generated, self.num_responses)
                        .await
                }
                MockBehavior::FailThenSuccess | MockBehavior::FailThenSuccessWithAffinity => {
                    if call_num == 0 {
                        // First call - return "No responders available" error to trigger retry
                        return Err(anyhow::anyhow!(
                            DynamoError::builder()
                                .error_type(ErrorType::CannotConnect)
                                .message("no responders")
                                .build()
                        ));
                    } else {
                        // Subsequent calls - succeed with remaining responses
                        self.send_responses(responses_already_generated, self.num_responses)
                            .await
                    }
                }
                MockBehavior::MidStreamFail { fail_after }
                | MockBehavior::MidStreamFailThenNoEndpoints { fail_after }
                | MockBehavior::MidStreamFailThenUnavailable { fail_after } => {
                    if call_num > 0 {
                        match self.behavior {
                            MockBehavior::MidStreamFailThenNoEndpoints { .. } => {
                                return Err(anyhow::Error::from(KvSchedulerError::NoEndpoints));
                            }
                            MockBehavior::MidStreamFailThenUnavailable { .. } => {
                                return Err(DynamoError::builder()
                                    .error_type(ErrorType::Unavailable)
                                    .message("No workers available for endpoint")
                                    .build()
                                    .into());
                            }
                            _ => {}
                        }
                    }
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });
                    } else {
                        // Second call - send remaining responses from where we left off.
                        // Like a real worker, the finishing chunk carries usage counted
                        // against the prompt this worker received: the original prompt
                        // plus the tokens replayed from the first stream.
                        let prompt_len = self.prompt_len;
                        tokio::spawn(async move {
                            for i in responses_already_generated..num_responses {
                                let mut response = create_mock_output(token_offset + 1 + i as u32);
                                if i + 1 == num_responses
                                    && let Some(data) = response.data.as_mut()
                                {
                                    let prompt_tokens =
                                        (prompt_len + responses_already_generated) as u32;
                                    let completion_tokens =
                                        (num_responses - responses_already_generated) as u32;
                                    data.completion_usage = Some(CompletionUsage {
                                        prompt_tokens,
                                        completion_tokens,
                                        total_tokens: prompt_tokens + completion_tokens,
                                        prompt_tokens_details: Some(
                                            dynamo_protocols::types::PromptTokensDetails {
                                                cached_tokens: Some(prompt_tokens),
                                                audio_tokens: None,
                                            },
                                        ),
                                        completion_tokens_details: None,
                                    });
                                }
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }

                    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                    let ctx = Arc::new(Controller::new(self.context_id.clone()));
                    Ok(dynamo_runtime::pipeline::ResponseStream::new(
                        Box::pin(stream),
                        ctx,
                    ))
                }
                MockBehavior::MidStreamFailAlways { fail_after } => {
                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        let (tx, rx) = mpsc::channel(1);
                        let token_offset = self.token_offset;
                        let fail_after = *fail_after;
                        let num_responses = self.num_responses;

                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - always fail with NoResponders error (same as AlwaysFail)
                        Err(anyhow::anyhow!(
                            DynamoError::builder()
                                .error_type(ErrorType::CannotConnect)
                                .message("no responders")
                                .build()
                        ))
                    }
                }
                MockBehavior::MidStreamFailAlwaysStreamError { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - immediately send stream error (no successful responses)
                        tokio::spawn(async move {
                            // Send the stream error immediately
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    }
                }
                MockBehavior::MidStreamFailBookedUntilDrained { fail_after } => {
                    if self.booked.load(Ordering::SeqCst) {
                        return Err(anyhow::Error::from(KvSchedulerError::BookingFailed(
                            format!(
                                "Request {} already exists (assigned to worker 7)",
                                self.context_id
                            ),
                        )));
                    }
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;
                    if call_num == 0 {
                        tokio::spawn(async move {
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                if tx
                                    .send(create_mock_output(token_offset + 1 + i as u32))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            let _ = tx
                                .send(Annotated::from_err(
                                    DynamoError::builder()
                                        .error_type(ErrorType::Disconnected)
                                        .message("Stream ended before generation completed")
                                        .build(),
                                ))
                                .await;
                        });
                    } else {
                        tokio::spawn(async move {
                            for i in responses_already_generated..num_responses {
                                if tx
                                    .send(create_mock_output(token_offset + 1 + i as u32))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        });
                    }
                    self.booked.store(true, Ordering::SeqCst);
                    let stream = BookedStream {
                        inner: Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)),
                        booked: self.booked.clone(),
                    };
                    let ctx = Arc::new(Controller::new(self.context_id.clone()));
                    Ok(dynamo_runtime::pipeline::ResponseStream::new(
                        Box::pin(stream),
                        ctx,
                    ))
                }
                MockBehavior::AlwaysNoEndpoints => {
                    Err(anyhow::Error::from(KvSchedulerError::NoEndpoints))
                }
                MockBehavior::AlwaysOverloaded => Err(DynamoError::builder()
                    .error_type(ErrorType::ResourceExhausted)
                    .message("All workers are busy, please retry later")
                    .build()
                    .into()),
                MockBehavior::AlwaysFail => {
                    // Always fail with NoResponders error (same as FailThenSuccess first call)
                    Err(anyhow::anyhow!(
                        DynamoError::builder()
                            .error_type(ErrorType::CannotConnect)
                            .message("no responders")
                            .build()
                    ))
                }
            }
        }
    }

    impl MockEngine {
        async fn send_responses(
            &self,
            start: usize,
            end: usize,
        ) -> Result<ManyOut<Annotated<BackendOutput>>> {
            let (tx, rx) = mpsc::channel(1);
            let token_offset = self.token_offset;

            tokio::spawn(async move {
                for i in start..end {
                    let response = create_mock_output(token_offset + 1 + i as u32);
                    if tx.send(response).await.is_err() {
                        break;
                    }
                }
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let ctx = Arc::new(Controller::new(self.context_id.clone()));
            Ok(dynamo_runtime::pipeline::ResponseStream::new(
                Box::pin(stream),
                ctx,
            ))
        }
    }

    /// Test case 1: No migration needed
    /// Tests the normal case where the RetryManager successfully processes all responses
    /// from a single stream without any failures or need for retries/migration.
    /// Expected behavior: All 10 responses should be received successfully.
    #[tokio::test]
    async fn test_retry_manager_no_migration() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            0,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    #[tokio::test]
    async fn test_migration_preserves_session_affinity_across_retry() {
        let context_id = uuid::Uuid::new_v4().to_string();
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::FailThenSuccessWithAffinity,
            1,
            100,
            context_id.clone(),
        ));
        let calls = mock_engine.call_count.clone();
        let mut request =
            Context::with_id_and_metadata(create_mock_request(1), context_id, BTreeMap::new());
        request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("session-123"),
        );

        let migration = Migration::new(1, None, TEST_MODEL.to_string(), Arc::new(Metrics::new()));
        let mut stream = migration.generate(request, mock_engine).await.unwrap();
        while stream.next().await.is_some() {}

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// Test case 2: New request migration
    /// Tests the scenario where a worker becomes unreachable for new requests initially,
    /// triggering the RetryManager to retry the request. The MockEngine with FailThenSuccess
    /// fails on the first call with a "No responders available" error, then succeeds
    /// on subsequent calls, simulating a worker becoming available after initial failure.
    /// Expected behavior: All 10 responses should be received successfully after retry.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::FailThenSuccess,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// A retry must not return to the worker whose stream failed. The router
    /// records every selected worker on the request tracker; after a failure
    /// that worker is in the request's excluded set for the next attempt.
    #[tokio::test]
    async fn test_retry_excludes_failed_worker() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let mut request = create_mock_request(10);
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_worker(7, Some(0), "decode");
        request.tracker = Some(tracker);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::FailThenSuccess,
            10,
            100,
            context_id.clone(),
        ));
        let excluded_seen = mock_engine.excluded_worker_ids_seen.clone();
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");
        let mut responses = 0;
        while let Some(response) = retry_manager.next().await {
            assert!(response.err().is_none());
            responses += 1;
        }
        assert_eq!(responses, 10);
        let excluded_seen = excluded_seen.lock().unwrap();
        assert_eq!(excluded_seen.len(), 2);
        assert_eq!(excluded_seen[0], None);
        assert_eq!(excluded_seen[1], Some(HashSet::from([7])));
    }

    /// Test case 3: Ongoing request migration
    /// Tests the scenario where a worker fails mid-stream during an ongoing request.
    /// This simulates a connection being lost after partial response delivery, requiring
    /// the RetryManager to detect the failure (via "Stream ended before generation completed" error),
    /// create a new stream, and continue from where it left off.
    /// Expected behavior: 5 responses from first stream + 5 responses from retry stream = 10 total.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 5 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received all 10 responses (5 from first stream + 5 from second stream)
        assert_eq!(responses.len(), 10);

        // Check that we received responses from both streams
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);

        // The retry worker counted the 5 replayed tokens as prompt (3 + 5); the
        // client's usage must still say 3, with cached tokens clamped to it.
        let usage = responses
            .last()
            .and_then(|r| r.data.as_ref())
            .and_then(|d| d.completion_usage.as_ref())
            .expect("finishing chunk carries usage");
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 8);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens),
            Some(3)
        );
    }

    /// A worker dies abruptly: its stream fails while the router still holds
    /// the request's booking. The retry must not race that booking; the
    /// failed stream is run to its end first, so the replacement is accepted.
    #[tokio::test]
    async fn test_retry_releases_failed_stream_before_rebooking() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailBookedUntilDrained { fail_after: 5 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            engine.clone();
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");
        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }
        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(
                response.err().is_none(),
                "response {i} is an error: {:?}",
                response.err()
            );
            assert_eq!(
                response.data.as_ref().map(|d| d.token_ids.clone()),
                Some(vec![101 + i as u32])
            );
        }
        // One failed stream, one replacement; no attempt was refused.
        assert_eq!(engine.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
    }

    struct StaticFallback {
        targets: Vec<MigrationTarget<BackendOutput>>,
    }

    impl MigrationFallbackSource for StaticFallback {
        fn backend_output_targets(&self) -> Vec<MigrationTarget<BackendOutput>> {
            self.targets.clone()
        }
        fn llm_engine_output_targets(&self) -> Vec<MigrationTarget<LLMEngineOutput>> {
            Vec::new()
        }
    }

    /// The worker dies mid-stream and its worker set has nobody left, which is
    /// what the last worker of a rolling-update generation looks like. The
    /// request continues on another worker set of the model.
    #[tokio::test]
    async fn test_retry_continues_in_another_worker_set_when_own_set_is_empty() {
        continue_in_another_worker_set(MockBehavior::MidStreamFailThenNoEndpoints {
            fail_after: 5,
        })
        .await;
    }

    /// Round-robin, random and load-based routers report an empty pool as
    /// `Unavailable` rather than through the KV scheduler.
    #[tokio::test]
    async fn test_retry_continues_in_another_worker_set_when_round_robin_pool_is_empty() {
        continue_in_another_worker_set(MockBehavior::MidStreamFailThenUnavailable {
            fail_after: 5,
        })
        .await;
    }

    async fn continue_in_another_worker_set(own_set_behavior: MockBehavior) {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let mut request = create_mock_request(10);
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_worker(7, Some(0), "decode");
        request.tracker = Some(tracker);
        let own_set = Arc::new(MockEngine::new(
            own_set_behavior,
            10,
            100,
            context_id.clone(),
        ));
        let other_set = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let fallback: Arc<dyn MigrationFallbackSource> = Arc::new(StaticFallback {
            targets: vec![MigrationTarget {
                namespace: "other-set".to_string(),
                engine: other_set.clone(),
                fallback: None,
            }],
        });
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(fallback),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // 5 tokens from the dying worker, 5 from the other worker set.
        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(
                response.err().is_none(),
                "response {i} is an error: {:?}",
                response.err()
            );
            assert_eq!(
                response.data.as_ref().map(|d| d.token_ids.clone()),
                Some(vec![101 + i as u32])
            );
        }
        // The own set was asked once more and had nobody; the other set took
        // the request straight away.
        assert_eq!(own_set.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(other_set.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        // The failed worker was excluded within its own set; the exclusion
        // does not follow the request into the other set.
        let own_seen = own_set.excluded_worker_ids_seen.lock().unwrap();
        assert_eq!(own_seen[1], Some(HashSet::from([7])));
        let other_seen = other_set.excluded_worker_ids_seen.lock().unwrap();
        assert_eq!(other_seen[0], None);
    }

    /// A request that moved to another worker set judges a later failure from
    /// that set: its lookup is adopted along with its engine, so a second
    /// move reaches the sets the second set knows about.
    #[tokio::test]
    async fn test_second_move_uses_the_adopted_worker_sets_lookup() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let first = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailThenNoEndpoints { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let second = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailThenNoEndpoints { fail_after: 7 },
            10,
            100,
            context_id.clone(),
        ));
        let third = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let second_lookup: Arc<dyn MigrationFallbackSource> = Arc::new(StaticFallback {
            targets: vec![MigrationTarget {
                namespace: "third".to_string(),
                engine: third.clone(),
                fallback: None,
            }],
        });
        let first_lookup: Arc<dyn MigrationFallbackSource> = Arc::new(StaticFallback {
            targets: vec![MigrationTarget {
                namespace: "second".to_string(),
                engine: second.clone(),
                fallback: Some(second_lookup),
            }],
        });
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            first.clone();

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(first_lookup),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // 3 tokens from the first set, 4 from the second, 3 from the third.
        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(
                response.err().is_none(),
                "response {i} is an error: {:?}",
                response.err()
            );
            assert_eq!(
                response.data.as_ref().map(|d| d.token_ids.clone()),
                Some(vec![101 + i as u32])
            );
        }
        assert_eq!(first.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(second.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(third.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 2);
    }

    /// A worker of the other set fails to take the request: its peers are
    /// tried within the remaining budget instead of skipping the set.
    #[tokio::test]
    async fn test_move_retries_within_the_other_worker_set() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailThenNoEndpoints { fail_after: 5 },
            10,
            100,
            context_id.clone(),
        ));
        // First dispatch into the other set hits a worker that cannot be reached.
        let other_set = Arc::new(MockEngine::new(
            MockBehavior::FailThenSuccess,
            10,
            100,
            context_id.clone(),
        ));
        let fallback: Arc<dyn MigrationFallbackSource> = Arc::new(StaticFallback {
            targets: vec![MigrationTarget {
                namespace: "other-set".to_string(),
                engine: other_set.clone(),
                fallback: None,
            }],
        });
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(fallback),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }
        assert_eq!(responses.len(), 10);
        assert!(responses.iter().all(|r| r.err().is_none()));
        assert_eq!(own_set.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(other_set.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 1);
    }

    fn static_targets(engines: &[(&str, Arc<MockEngine>)]) -> Arc<dyn MigrationFallbackSource> {
        Arc::new(StaticFallback {
            targets: engines
                .iter()
                .map(|(namespace, engine)| MigrationTarget {
                    namespace: namespace.to_string(),
                    engine: engine.clone(),
                    fallback: None,
                })
                .collect(),
        })
    }

    /// A first dispatch that finds its set empty moves to another set and is
    /// counted as a new-request migration.
    #[tokio::test]
    async fn test_first_dispatch_moves_when_own_set_is_empty() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::AlwaysNoEndpoints,
            10,
            100,
            context_id.clone(),
        ));
        let other_set = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(static_targets(&[("other-set", other_set.clone())])),
        )
        .await
        .expect("Failed to build RetryManager");
        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }
        assert_eq!(responses.len(), 10);
        assert!(responses.iter().all(|r| r.err().is_none()));
        assert_eq!(other_set.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Once the retry budget is spent on a set whose workers cannot be
    /// reached, no further set is tried.
    #[tokio::test]
    async fn test_budget_spent_in_one_set_stops_the_move() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::AlwaysNoEndpoints,
            10,
            100,
            context_id.clone(),
        ));
        let unreachable = Arc::new(MockEngine::new(
            MockBehavior::AlwaysFail,
            10,
            100,
            context_id.clone(),
        ));
        let healthy = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let result = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            1,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(static_targets(&[
                ("unreachable", unreachable.clone()),
                ("healthy", healthy.clone()),
            ])),
        )
        .await;
        let err = result
            .err()
            .expect("the budget is spent before a set takes the request");
        assert!(is_no_endpoints_error(&err), "unexpected error: {err}");
        // The first dispatch found the own set empty and cost nothing. The
        // unreachable set gets that dispatch plus the one retry the budget
        // allows, then the move stops before the healthy set.
        assert_eq!(unreachable.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(healthy.call_count.load(Ordering::SeqCst), 0);
    }

    /// A target set's own verdict, such as overload, stands: it is reported
    /// rather than hopping to yet another set.
    #[tokio::test]
    async fn test_overloaded_other_set_is_reported_not_skipped() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::AlwaysNoEndpoints,
            10,
            100,
            context_id.clone(),
        ));
        let overloaded = Arc::new(MockEngine::new(
            MockBehavior::AlwaysOverloaded,
            10,
            100,
            context_id.clone(),
        ));
        let healthy = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let result = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(static_targets(&[
                ("overloaded", overloaded.clone()),
                ("healthy", healthy.clone()),
            ])),
        )
        .await;
        let err = result.err().expect("overload must be reported");
        assert!(
            error::match_error_chain(err.as_ref(), &[ErrorType::ResourceExhausted], &[]),
            "unexpected error: {err}"
        );
        assert_eq!(overloaded.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(healthy.call_count.load(Ordering::SeqCst), 0);
    }

    /// A prompt already past the migration sequence limit never moves.
    #[tokio::test]
    async fn test_prompt_over_max_seq_len_never_moves() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10); // prompt [1, 2, 3]
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::AlwaysNoEndpoints,
            10,
            100,
            context_id.clone(),
        ));
        let other_set = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let result = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            Some(2),
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(static_targets(&[("other-set", other_set.clone())])),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(other_set.call_count.load(Ordering::SeqCst), 0);
    }

    /// With migration disabled, a request never leaves its own worker set,
    /// even when that set has no worker for its first dispatch.
    #[tokio::test]
    async fn test_disabled_migration_never_moves_to_another_worker_set() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::AlwaysNoEndpoints,
            10,
            100,
            context_id.clone(),
        ));
        let other_set = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let fallback: Arc<dyn MigrationFallbackSource> = Arc::new(StaticFallback {
            targets: vec![MigrationTarget {
                namespace: "other-set".to_string(),
                engine: other_set.clone(),
                fallback: None,
            }],
        });
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let result = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            0,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(fallback),
        )
        .await;
        let err = result.err().expect("dispatch must fail without migration");
        assert!(is_no_endpoints_error(&err), "unexpected error: {err}");
        assert_eq!(own_set.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(other_set.call_count.load(Ordering::SeqCst), 0);
    }

    fn is_no_endpoints_error(err: &anyhow::Error) -> bool {
        err.chain().any(|cause| {
            matches!(
                cause.downcast_ref::<KvSchedulerError>(),
                Some(KvSchedulerError::NoEndpoints)
            )
        })
    }

    /// With no other worker set to continue on, the request fails with the
    /// disconnect that started the migration.
    #[tokio::test]
    async fn test_retry_fails_when_no_other_worker_set_can_take_over() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let own_set = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailThenNoEndpoints { fail_after: 5 },
            10,
            100,
            context_id.clone(),
        ));
        let fallback: Arc<dyn MigrationFallbackSource> = Arc::new(StaticFallback {
            targets: Vec::new(),
        });
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            own_set.clone();

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build_with_fallback(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
            Some(fallback),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 6);
        for (i, response) in responses[0..5].iter().enumerate() {
            assert!(response.err().is_none());
            assert_eq!(
                response.data.as_ref().map(|d| d.token_ids.clone()),
                Some(vec![101 + i as u32])
            );
        }
        let err = responses[5].err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);
        // An empty worker set is not retried against.
        assert_eq!(own_set.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
    }

    /// Test case 4: New request migration - indefinite failure
    /// Tests the scenario where a worker becomes unreachable for new requests indefinitely.
    /// The RetryManager should exhaust all retries and return the original error from the first attempt.
    /// Expected behavior: Should receive an error after all retries are exhausted, with the original error.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration_indefinite_failure() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(0);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::AlwaysFail,
            0,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        // Should fail to build due to initial stream creation failure after exhausting all 3 retries
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let retry_manager_result = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(error.to_string().contains("no responders"));
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 4); // 3 retries + 1 final failure
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 5: Ongoing request migration - indefinite failure
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests.
    /// The RetryManager should exhaust all retries and return the original stream disconnection error.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlways { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        ) // 3 retries
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be a Disconnected error after retries are exhausted
        let error_response = &responses[3];
        let err = error_response.err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 3); // 2 retries + 1 final failure
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1); // initial ongoing failure retry
    }

    /// Test case 6: Ongoing request migration - indefinite failure with stream errors
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests,
    /// and all retry attempts also fail with stream errors instead of NATS errors.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure_stream_error() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlwaysStreamError { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        ) // 3 retries
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be a Disconnected error after retries are exhausted
        let error_response = &responses[3];
        let err = error_response.err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 4); // 3 retries + 1 final failure
    }

    /// Test case 7: Request cancelled when creating new stream
    /// Tests the scenario where context.stop_generating() is called when creating a new stream.
    /// The RetryManager should detect that the context is stopped and abort creating new streams.
    /// Expected behavior: Should fail to build RetryManager with "Context is stopped or killed" error.
    #[tokio::test]
    async fn test_retry_manager_context_stopped_before_stream() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));

        // Stop the context before building RetryManager
        ctx.stop_generating();

        // Should fail to build due to stopped context
        let metrics = Arc::new(Metrics::new());
        let retry_manager_result = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(
                error
                    .to_string()
                    .contains(&format!("Context id {} is stopped or killed", context_id))
            );
            // Verify the error is a typed DynamoError with Cancelled type
            let dynamo_err = error
                .downcast_ref::<DynamoError>()
                .expect("Error should be a DynamoError");
            assert_eq!(
                dynamo_err.error_type(),
                ErrorType::Cancelled,
                "Stopped/killed context should produce a Cancelled error"
            );
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 8: No migration for guided-decoding (structured-output) requests
    ///
    /// Bug (#7634): When a worker crashes mid-stream during a structured-output
    /// (json_schema) request, migration appends already-generated token IDs back onto
    /// token_ids and replays the request to a new worker. However, backends initialize
    /// the guided-decoding FSM fresh for every new request and only advance it on newly-
    /// generated tokens — not on context/prompt tokens. This causes the FSM to restart
    /// from the schema root while treating already-generated tokens as context, producing
    /// duplicated or nested JSON in the final response.
    ///
    /// Fix: Disable migration for structured-output requests by zeroing retries_left in
    /// RetryManager::build() when guided_decoding is set, propagating the error cleanly.
    ///
    /// Expected behavior BEFORE fix: All 10 responses received (migration happened — wrong)
    /// Expected behavior AFTER fix: 3 successful + 1 error (migration blocked — correct)
    #[tokio::test]
    async fn test_retry_manager_no_migration_for_guided_decoding() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        let mut request = create_mock_request(10);
        // Set guided decoding (json_schema structured output) on the request
        request.sampling_options.guided_decoding = Some(GuidedDecodingOptions::new(
            Some(serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}})),
            None,
            None,
            None,
            None,
            None,
            None,
        ));

        // MidStreamFail after 3 tokens: without the fix, migration would succeed and
        // deliver all 10 responses; with the fix, migration is blocked and an error
        // is returned after the 3 partial responses.
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3, // migration_limit=3 — should be ignored for guided-decoding requests
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Must receive 3 successful tokens + 1 Disconnected error, NOT all 10.
        // Before the fix this assertion fails because migration proceeds and returns 10.
        assert_eq!(
            responses.len(),
            4,
            "Expected 3 successful + 1 error response (migration must be blocked for \
             guided-decoding), but got {} responses",
            responses.len()
        );

        // First 3 responses should be successful
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(
                response.err().is_none(),
                "Response {} should be successful",
                i
            );
        }

        // Last response must be the stream-disconnection error
        let last = responses.last().unwrap();
        let err = last
            .err()
            .expect("Last response should be a Disconnected error");
        assert_eq!(
            err.error_type(),
            ErrorType::Disconnected,
            "Error type should be Disconnected"
        );
    }

    /// Test case 9: max_seq_len exceeded limit + 1 disables migration
    ///
    /// Boundary test: prompt has 3 tokens, max_seq_len = 5. After 2 generated tokens the
    /// total is 5 (== max_seq_len) — still migratable. The 3rd generated token would push
    /// the total to 6 (> max_seq_len), which disables migration and stops caching.
    /// The failure is placed right at that point (fail_after: 3) so we see the error
    /// propagated instead of retried.
    #[tokio::test]
    async fn test_retry_manager_max_seq_len_exceeded() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        // Prompt = [1, 2, 3] (len 3). max_seq_len = 5.
        // Token 101 → total 4 ≤ 5: tracked.
        // Token 102 → total 5 ≤ 5: tracked.
        // Token 103 → would-be 6 > 5: NOT tracked, migration disabled.
        // Error follows immediately (fail_after: 3) → not retried.
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            Some(5), // prompt(3) + 3 generated = 6 > 5 → disables migration
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // 3 successful tokens + 1 Disconnected error (migration disabled at token 103).
        assert_eq!(
            responses.len(),
            4,
            "Expected 3 successful + 1 error (migration disabled by max_seq_len)"
        );

        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none(), "Response {} should be OK", i);
        }

        let err = responses[3]
            .err()
            .expect("Last response should be Disconnected error");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        // Migration was attempted but blocked because max_seq_len set retries_left to 0.
        // The ongoing metric is still incremented (it counts attempts, not successes).
        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
        // max_seq_len limit was triggered once (at token 103).
        assert_eq!(
            metrics.get_migration_max_seq_len_exceeded_count(TEST_MODEL),
            1
        );
    }

    /// Test case 10: Migration succeeds when sequence length is at max_seq_len
    ///
    /// Boundary test: prompt has 3 tokens, max_seq_len = 5. After 2 generated tokens
    /// the total is exactly 5 (== max_seq_len). The failure occurs at that point
    /// (fail_after: 2). Because we use strict inequality (> not >=), the request is
    /// still migratable and the retry succeeds.
    #[tokio::test]
    async fn test_retry_manager_max_seq_len_at_limit() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        // Prompt = [1, 2, 3] (len 3). max_seq_len = 5.
        // Token 101 → total 4 ≤ 5: tracked.
        // Token 102 → total 5 == 5: tracked (still migratable — strict >).
        // Error (fail_after: 2) → migration succeeds, retry delivers remaining tokens.
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 2 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            Some(5), // prompt(3) + 2 generated = 5 == max_seq_len → still migratable
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Migration succeeds — all 10 responses delivered
        assert_eq!(responses.len(), 10);
        for response in &responses {
            assert!(response.err().is_none());
        }

        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);

        // Tracked token_ids must equal exactly max_seq_len (5).
        // The 2 tokens from the first stream were tracked (prompt 3 + gen 2 = 5).
        // After migration the retry stream delivers remaining tokens, but the first
        // new token would push to 6 > 5, so tracking stops and no more are appended.
        assert_eq!(
            retry_manager.request.token_ids.len(),
            5,
            "tracked token_ids should be exactly max_seq_len"
        );

        // The limit was triggered once (first token of the retry stream exceeded 5).
        assert_eq!(
            metrics.get_migration_max_seq_len_exceeded_count(TEST_MODEL),
            1
        );
    }

    /// Test case 11: Prompt length alone exceeds max_seq_len
    ///
    /// When the prompt tokens already exceed max_seq_len, migration is disabled
    /// in RetryManager::build before any tokens are generated. A mid-stream
    /// failure should propagate the error without attempting migration.
    #[tokio::test]
    async fn test_retry_manager_max_seq_len_exceeded_by_prompt() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        // Prompt = [1, 2, 3] (len 3). max_seq_len = 2, so prompt alone exceeds the limit.
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            Some(2), // prompt(3) > max_seq_len(2) → migration disabled at build time
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // 3 successful tokens + 1 Disconnected error (migration disabled from the start).
        assert_eq!(
            responses.len(),
            4,
            "Expected 3 successful + 1 error (migration disabled by prompt exceeding max_seq_len)"
        );

        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none(), "Response {} should be OK", i);
        }

        let err = responses[3]
            .err()
            .expect("Last response should be Disconnected error");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        // max_seq_len was exceeded at build time (prompt len 3 > 2).
        assert_eq!(
            metrics.get_migration_max_seq_len_exceeded_count(TEST_MODEL),
            1
        );
        // Migration was attempted but blocked (retries_left was already 0).
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
    }

    /// Smoke test for the byo-preprocessor response shape.
    #[tokio::test]
    async fn test_retry_manager_generic_over_llm_engine_output() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();

        struct LlmEngineMock(String);

        #[async_trait]
        impl
            AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<LLMEngineOutput>>,
                anyhow::Error,
            > for LlmEngineMock
        {
            async fn generate(
                &self,
                _request: SingleIn<PreprocessedRequest>,
            ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
                let responses = stream::iter((0..3u32).map(|i| {
                    Annotated::from_data(LLMEngineOutput {
                        token_ids: vec![200 + i],
                        ..Default::default()
                    })
                }));
                let ctx = Arc::new(Controller::new(self.0.clone()));
                Ok(ResponseStream::new(Box::pin(responses), ctx))
            }
        }

        let request = create_mock_request(3);
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(LlmEngineMock(context_id.clone()));

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            1,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics,
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(r) = retry_manager.next().await {
            responses.push(r);
        }
        assert_eq!(responses.len(), 3);
        assert_eq!(
            retry_manager.request.token_ids,
            vec![1, 2, 3, 200, 201, 202]
        );
    }

    /// 2-hop migration: A → fail → B → fail → C. Each retry's
    /// `migration_link` must point at the *latest* failed worker, not the
    /// original.
    #[tokio::test]
    async fn test_retry_manager_propagates_migration_link_over_two_hops() {
        use crate::protocols::common::preprocessor::TraceLink;
        use std::sync::Mutex;

        dynamo_runtime::logging::init();

        struct LinkingMockEngine {
            captured_links: Arc<Mutex<Vec<Option<TraceLink>>>>,
            worker_links: Vec<TraceLink>,
            context_id: String,
            call_count: Arc<AtomicU32>,
        }

        #[async_trait]
        impl
            AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
                anyhow::Error,
            > for LinkingMockEngine
        {
            async fn generate(
                &self,
                request: SingleIn<PreprocessedRequest>,
            ) -> Result<ManyOut<Annotated<BackendOutput>>> {
                let call_num = self.call_count.fetch_add(1, Ordering::SeqCst) as usize;
                let (preprocessed_request, _ctx) = request.transfer(());
                self.captured_links
                    .lock()
                    .unwrap()
                    .push(preprocessed_request.migration_link.clone());

                let (tx, rx) = mpsc::channel(1);
                let context_id = self.context_id.clone();
                let fail_this_call = call_num < 2;
                let link = self.worker_links.get(call_num).cloned();
                let responses_already_generated =
                    preprocessed_request.token_ids.len().saturating_sub(3);
                let total_chunks: usize = 6;

                tokio::spawn(async move {
                    let start = responses_already_generated;
                    let end = if fail_this_call {
                        (start + 2).min(total_chunks)
                    } else {
                        total_chunks
                    };
                    for i in start..end {
                        let mut out = create_mock_output(100 + 1 + i as u32);
                        if i == start
                            && let (Some(link), Some(data)) = (&link, out.data.as_mut())
                        {
                            data.worker_trace_link = Some(link.clone());
                        }
                        if tx.send(out).await.is_err() {
                            return;
                        }
                    }
                    if fail_this_call {
                        let err = Annotated::from_err(
                            DynamoError::builder()
                                .error_type(ErrorType::Disconnected)
                                .message("Stream ended before generation completed")
                                .build(),
                        );
                        let _ = tx.send(err).await;
                    }
                });

                let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                let ctx = Arc::new(Controller::new(context_id));
                Ok(dynamo_runtime::pipeline::ResponseStream::new(
                    Box::pin(stream),
                    ctx,
                ))
            }
        }

        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(6);
        let captured = Arc::new(Mutex::new(Vec::<Option<TraceLink>>::new()));
        let link_a = TraceLink {
            trace_id: "0123456789abcdef0123456789abcdef".to_string(),
            span_id: "aaaaaaaaaaaaaaaa".to_string(),
        };
        let link_b = TraceLink {
            trace_id: "0123456789abcdef0123456789abcdef".to_string(),
            span_id: "bbbbbbbbbbbbbbbb".to_string(),
        };

        let engine = Arc::new(LinkingMockEngine {
            captured_links: captured.clone(),
            worker_links: vec![link_a.clone(), link_b.clone()],
            context_id: context_id.clone(),
            call_count: Arc::new(AtomicU32::new(0)),
        });
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            engine;

        let ctx = Arc::new(Controller::new(context_id));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            BTreeMap::new(),
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
            None,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 6, "expected all 6 chunks across 2 hops");
        for response in &responses {
            assert!(response.err().is_none(), "no chunk should be an error");
        }

        let links = captured.lock().unwrap();
        assert_eq!(
            links.len(),
            3,
            "engine.generate must be called 3 times for a 2-hop migration"
        );
        assert!(
            links[0].is_none(),
            "first attempt has no predecessor — migration_link must be None"
        );
        assert_eq!(
            links[1].as_ref(),
            Some(&link_a),
            "second attempt must link back to worker A"
        );
        assert_eq!(
            links[2].as_ref(),
            Some(&link_b),
            "third attempt must link back to worker B (latest worker, not original)"
        );

        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 2);
    }
}
