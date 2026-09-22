// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool selection: which worker set serves a request when a model has more
//! than one.
//!
//! A worker set is one namespace's workers behind one KV router. A model has
//! several during a rolling update, across regions, or with a canary. The set
//! whose pipeline a request entered is its home set. This operator asks the
//! home set's router and every compatible set's router which worker they would
//! pick and what that worker costs, then continues in the cheapest set. The
//! cost is the router's own selection logit, so a request stays where its
//! prefix is cached unless another set is clearly better placed or less
//! loaded. The home set wins ties, so a model with one set pays nothing.
//!
//! A mirror set shadows one worker of a serving set. When a set's router
//! places a request on that worker, whether the request entered that set or
//! was placed there from another, a copy of the request is sent to the mirror
//! set as well; the copy's output is discarded. A copy runs to its own end,
//! whenever the real request ends; only a kill of the real request's context
//! (a cancelled or disconnected client, or the inactivity timeout) cuts it
//! off. The mirror thus sees the same requests, in the same order and at the
//! same arrival rate, as the worker it shadows, so a configuration under
//! test compares like for like with a serving worker without touching a
//! client; a mirror slower than that worker builds a backlog of copies.
//! Mirror sets never serve.
//!
//! The operator sits below the migration operator and above the token
//! backend. A retry after a failed worker re-enters selection, and a request
//! placed in another set runs through that set's pipeline below its own
//! migration operator, the same entry that cross-set migration uses.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, Operator, PipelineOperator,
    ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

use crate::http::service::metrics::Metrics;
use crate::kv_router::{KvRouter, protocols::WorkerWithDpRank};
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest};
use crate::protocols::common::timing::RequestTracker;

/// What a set's router would do with a request, without booking it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoolPreview {
    pub worker: WorkerWithDpRank,
    /// Selection cost in block units; lower is better.
    pub cost: f64,
    pub overlap_blocks: u32,
}

/// A worker set's router, asked for a placement without booking one.
#[async_trait]
pub trait PoolPreviewer: Send + Sync {
    /// The worker this set would pick and its cost, or `None` when the set
    /// has no worker that could take the request now.
    async fn preview(&self, request: &PreprocessedRequest) -> Result<Option<PoolPreview>>;
}

#[async_trait]
impl PoolPreviewer for KvRouter {
    async fn preview(&self, request: &PreprocessedRequest) -> Result<Option<PoolPreview>> {
        let (token_ids, block_mm_infos) = request.block_mm_routing_info();
        let routing = request.routing.as_ref();
        let placement = self
            .preview_placement(
                token_ids,
                block_mm_infos,
                request.router_config_override.as_ref(),
                routing.and_then(|hints| hints.lora_name.clone()),
                routing.and_then(|hints| hints.cache_namespace.clone()),
                routing.and_then(|hints| hints.priority_jump).unwrap_or(0.0),
                routing.and_then(|hints| hints.strict_priority).unwrap_or(0),
                request
                    .agent_context
                    .as_ref()
                    .map(|context| context.session_id.clone()),
                routing.and_then(|hints| hints.expected_output_tokens),
                routing.and_then(|hints| hints.allowed_worker_ids.clone()),
                routing.and_then(|hints| hints.excluded_worker_ids.clone()),
                routing
                    .and_then(|hints| hints.routing_constraints.clone())
                    .unwrap_or_default(),
            )
            .await?;
        Ok(placement.map(|placement| PoolPreview {
            worker: placement.worker,
            cost: placement.logit,
            overlap_blocks: placement.overlap_blocks,
        }))
    }
}

/// A worker set that shadows one worker of a serving set: every request that
/// set's router places on the worker is copied to this set.
pub struct PoolMirror<Resp> {
    pub namespace: String,
    /// Instance id of the serving worker this set shadows.
    pub worker_id: u64,
    /// The set's pipeline below its migration operator.
    pub engine: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
}

impl<Resp> Clone for PoolMirror<Resp> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            worker_id: self.worker_id,
            engine: self.engine.clone(),
        }
    }
}

/// Another worker set of the model a request may be placed in.
pub struct PoolCandidate<Resp> {
    pub namespace: String,
    pub previewer: Arc<dyn PoolPreviewer>,
    /// The set's pipeline below its migration operator.
    pub engine: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    /// The mirror sets shadowing this set's workers. A request placed here
    /// enters below this set's own pool-selection stage, so its mirrors are
    /// applied by the stage that placed it.
    pub mirrors: Vec<PoolMirror<Resp>>,
}

impl<Resp> Clone for PoolCandidate<Resp> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            previewer: self.previewer.clone(),
            engine: self.engine.clone(),
            mirrors: self.mirrors.clone(),
        }
    }
}

/// Lookup of the other worker sets a home set's request may be placed in,
/// and of the mirror sets shadowing its workers, evaluated per request so
/// sets that join or leave are seen at once.
pub trait PoolSelectionSource: Send + Sync {
    /// The home set's namespace, for logs.
    fn namespace(&self) -> &str;
    fn backend_output_candidates(&self) -> Vec<PoolCandidate<BackendOutput>>;
    fn llm_engine_output_candidates(&self) -> Vec<PoolCandidate<LLMEngineOutput>>;
    fn backend_output_mirrors(&self) -> Vec<PoolMirror<BackendOutput>> {
        Vec::new()
    }
    fn llm_engine_output_mirrors(&self) -> Vec<PoolMirror<LLMEngineOutput>> {
        Vec::new()
    }
}

/// Response types that have a per-set pipeline below the migration operator.
pub trait PoolCandidates: Sized {
    fn candidates(source: &dyn PoolSelectionSource) -> Vec<PoolCandidate<Self>>;
    fn mirrors(source: &dyn PoolSelectionSource) -> Vec<PoolMirror<Self>>;
    /// Tokens carried by one response, for the mirror copy's latencies.
    fn token_count(&self) -> usize;
    /// Whether this response ends the request in failure.
    fn failed(&self) -> bool;
}

impl PoolCandidates for BackendOutput {
    fn candidates(source: &dyn PoolSelectionSource) -> Vec<PoolCandidate<Self>> {
        source.backend_output_candidates()
    }
    fn mirrors(source: &dyn PoolSelectionSource) -> Vec<PoolMirror<Self>> {
        source.backend_output_mirrors()
    }
    fn token_count(&self) -> usize {
        self.token_ids.len()
    }
    fn failed(&self) -> bool {
        matches!(
            self.finish_reason,
            Some(FinishReason::Error(_) | FinishReason::Cancelled)
        )
    }
}

impl PoolCandidates for LLMEngineOutput {
    fn candidates(source: &dyn PoolSelectionSource) -> Vec<PoolCandidate<Self>> {
        source.llm_engine_output_candidates()
    }
    fn mirrors(source: &dyn PoolSelectionSource) -> Vec<PoolMirror<Self>> {
        source.llm_engine_output_mirrors()
    }
    fn token_count(&self) -> usize {
        self.token_ids.len()
    }
    fn failed(&self) -> bool {
        matches!(
            self.finish_reason,
            Some(FinishReason::Error(_) | FinishReason::Cancelled)
        )
    }
}

/// Where a request goes: its home set, or the candidate at this index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolDecision {
    Home,
    Candidate(usize),
}

/// What became of a request copy in a mirror set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorOutcome {
    /// The copy ran to its own end.
    Completed,
    /// The copy was cut off because the real request's context was killed:
    /// the client cancelled or disconnected, or the request timed out.
    Stopped,
    /// The mirror set failed the copy.
    Failed,
}

/// The cheapest placement wins. The home set wins ties, and wins outright when
/// no candidate can take the request. A home set with no placement of its own
/// hands the request to any candidate that has one.
pub fn choose(home: Option<PoolPreview>, candidates: &[Option<PoolPreview>]) -> PoolDecision {
    let mut best = home.map(|preview| (PoolDecision::Home, preview.cost));
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(preview) = candidate else {
            continue;
        };
        let better = match best {
            Some((_, cost)) => preview.cost < cost,
            None => true,
        };
        if better {
            best = Some((PoolDecision::Candidate(index), preview.cost));
        }
    }
    best.map(|(decision, _)| decision)
        .unwrap_or(PoolDecision::Home)
}

pub struct PoolSelection {
    model_name: Arc<String>,
    previewer: Option<Arc<dyn PoolPreviewer>>,
    source: Option<Arc<dyn PoolSelectionSource>>,
    metrics: Arc<Metrics>,
}

impl PoolSelection {
    /// A stage that places requests across the model's worker sets. Without a
    /// previewer for the home set or a source of other sets it passes every
    /// request through.
    pub fn new(
        model_name: String,
        previewer: Option<Arc<dyn PoolPreviewer>>,
        source: Option<Arc<dyn PoolSelectionSource>>,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        Arc::new(Self {
            model_name: Arc::new(model_name),
            previewer,
            source,
            metrics,
        })
    }

    /// Wrap as a `PipelineOperator` over the given response type; the response
    /// type does not appear in the struct, so the caller names it.
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
        Resp: Data + PoolCandidates,
    {
        Operator::into_operator(self)
    }

    /// A request that only asks which worker it would get runs nowhere.
    fn query_only(request: &PreprocessedRequest) -> bool {
        request.get_annotation_value("query_instance_id").is_some()
    }

    /// A request that names its worker stays in the set it entered.
    fn pinned(request: &PreprocessedRequest) -> bool {
        request.routing.as_ref().is_some_and(|hints| {
            hints.backend_instance_id.is_some()
                || hints.decode_worker_id.is_some()
                || hints.prefill_worker_id.is_some()
        })
    }

    async fn preview_or_none(
        previewer: &dyn PoolPreviewer,
        request: &PreprocessedRequest,
        namespace: &str,
    ) -> Option<PoolPreview> {
        match previewer.preview(request).await {
            Ok(preview) => preview,
            Err(error) => {
                tracing::debug!(namespace, %error, "Pool preview failed; set not considered");
                None
            }
        }
    }

    /// Serve the request through `engine`, the pipeline of the set it is
    /// placed in. When that set's router places it on a worker that mirror
    /// sets shadow, a copy goes to each of them too.
    async fn serve_with<Resp>(
        &self,
        engine: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
        request: SingleIn<PreprocessedRequest>,
        mirrors: Vec<PoolMirror<Resp>>,
        preview: Option<PoolPreview>,
    ) -> Result<ManyOut<Annotated<Resp>>>
    where
        Resp: Data + PoolCandidates,
    {
        if mirrors.is_empty() {
            return engine.generate(request).await;
        }
        let copy = request.content().clone();
        let metadata = request.metadata().clone();
        let tracker = request.tracker.clone();
        let parent = request.context();
        let stream = engine.generate(request).await?;
        // The router records the worker it chose on the tracker before it
        // returns the stream. A request without a tracker uses the preview.
        let placed = tracker
            .as_ref()
            .and_then(|tracker| tracker.last_selected_worker_id())
            .or(preview.map(|preview| preview.worker.worker_id));
        let Some(worker_id) = placed else {
            return Ok(stream);
        };
        let matching: Vec<PoolMirror<Resp>> = mirrors
            .into_iter()
            .filter(|mirror| mirror.worker_id == worker_id)
            .collect();
        if matching.is_empty() {
            return Ok(stream);
        }
        Ok(self.mirror(matching, copy, metadata, parent, stream))
    }

    /// Send a copy of the request to each mirror set. A copy outlives the
    /// real request's stream; only a kill of the real request's context
    /// stops it.
    fn mirror<Resp>(
        &self,
        mirrors: Vec<PoolMirror<Resp>>,
        mut copy: PreprocessedRequest,
        metadata: std::collections::BTreeMap<String, String>,
        parent: Arc<dyn AsyncEngineContext>,
        stream: ManyOut<Annotated<Resp>>,
    ) -> ManyOut<Annotated<Resp>>
    where
        Resp: Data + PoolCandidates,
    {
        // A pin names a serving worker; the mirror set's router places the
        // copy on its own.
        if let Some(routing) = copy.routing.as_mut() {
            routing.backend_instance_id = None;
            routing.prefill_worker_id = None;
            routing.decode_worker_id = None;
            routing.dp_rank = None;
            routing.prefill_dp_rank = None;
        }
        for mirror in mirrors {
            let mut copy = copy.clone();
            // The copy records its own placement and timings; sharing the
            // real request's tracker would make the mirror's worker the one
            // its metrics and retries are attributed to.
            copy.tracker = Some(Arc::new(RequestTracker::new()));
            let shadow = Context::with_id_and_metadata(
                copy,
                format!("{}-mirror-{}", parent.id(), mirror.namespace),
                metadata.clone(),
            );
            tracing::debug!(
                model = %self.model_name,
                request_id = %parent.id(),
                mirror = %mirror.namespace,
                worker_id = mirror.worker_id,
                "Copying request to mirror set"
            );
            let job = MirrorJob {
                model: self.model_name.clone(),
                metrics: self.metrics.clone(),
                namespace: mirror.namespace,
                worker_id: mirror.worker_id,
                request_id: parent.id().to_string(),
                parent: parent.clone(),
            };
            tokio::spawn(job.run(mirror.engine, shadow));
        }

        // A copy runs to its own end whatever becomes of the real stream:
        // a mirror slower than the shadowed worker builds a backlog, and
        // that backlog is what the mirror is there to show.
        stream
    }
}

/// Runs one request copy in a mirror set, discarding its output and recording
/// its latencies and outcome.
struct MirrorJob {
    model: Arc<String>,
    metrics: Arc<Metrics>,
    namespace: String,
    worker_id: u64,
    request_id: String,
    /// The real request's context. Held for the copy's lifetime, so
    /// `killed()` resolves only on a real kill.
    parent: Arc<dyn AsyncEngineContext>,
}

impl MirrorJob {
    async fn run<Resp>(
        self,
        engine: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
        shadow: SingleIn<PreprocessedRequest>,
    ) where
        Resp: Data + PoolCandidates,
    {
        let inflight = self.metrics.mirror_inflight_gauge(&self.model);
        inflight.inc();
        let started = Instant::now();
        let mut tokens = 0usize;
        // A cancelled or disconnected client kills the real request's
        // context, and the copy with it. The real request ending on its own
        // only stops its context, which the copy does not follow.
        let cancel = tokio::spawn({
            let parent = self.parent.clone();
            let shadow = shadow.context();
            async move {
                parent.killed().await;
                shadow.kill();
            }
        });
        let outcome = match engine.generate(shadow).await {
            Err(error) => {
                tracing::debug!(
                    model = %self.model,
                    request_id = %self.request_id,
                    mirror = %self.namespace,
                    %error,
                    "Mirror set refused the request copy"
                );
                // A kill that lands while the copy is still being placed
                // surfaces as an error here.
                if self.parent.is_killed() {
                    MirrorOutcome::Stopped
                } else {
                    MirrorOutcome::Failed
                }
            }
            Ok(mut stream) => {
                let mut failed = false;
                let mut last_tokens_at: Option<Instant> = None;
                while let Some(item) = stream.next().await {
                    // Failures arrive as an error item or as a terminal
                    // finish reason on a data item.
                    if item.error.is_some()
                        || item.event.as_deref() == Some("error")
                        || item.data.as_ref().is_some_and(Resp::failed)
                    {
                        failed = true;
                    }
                    let count = item.data.as_ref().map(Resp::token_count).unwrap_or(0);
                    if count == 0 {
                        continue;
                    }
                    let now = Instant::now();
                    match last_tokens_at {
                        None => self.metrics.observe_mirror_time_to_first_token(
                            &self.model,
                            now.duration_since(started).as_secs_f64(),
                        ),
                        Some(last) => {
                            let per_token = now.duration_since(last).as_secs_f64() / count as f64;
                            for _ in 0..count {
                                self.metrics
                                    .observe_mirror_inter_token_latency(&self.model, per_token);
                            }
                        }
                    }
                    last_tokens_at = Some(now);
                    tokens += count;
                }
                // The copy's own context is also stopped by its pipeline on
                // a local stop condition, so only a kill of the real request
                // tells a cancelled copy from one that ended on its own.
                if self.parent.is_killed() {
                    MirrorOutcome::Stopped
                } else if failed {
                    MirrorOutcome::Failed
                } else {
                    MirrorOutcome::Completed
                }
            }
        };
        cancel.abort();
        inflight.dec();
        self.metrics.inc_mirror_request(&self.model, outcome);
        tracing::debug!(
            model = %self.model,
            request_id = %self.request_id,
            mirror = %self.namespace,
            worker_id = self.worker_id,
            ?outcome,
            tokens,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Mirror copy finished"
        );
    }
}

#[async_trait]
impl<Resp>
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
    > for PoolSelection
where
    Resp: Data + PoolCandidates,
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    ) -> Result<ManyOut<Annotated<Resp>>> {
        let (Some(previewer), Some(source)) = (self.previewer.as_ref(), self.source.as_ref())
        else {
            return next.generate(request).await;
        };
        if Self::query_only(&request) {
            return next.generate(request).await;
        }
        let mirrors = Resp::mirrors(source.as_ref());
        if Self::pinned(&request) {
            return self.serve_with(next, request, mirrors, None).await;
        }
        let candidates = Resp::candidates(source.as_ref());
        if candidates.is_empty() {
            return self.serve_with(next, request, mirrors, None).await;
        }

        let home_namespace = source.namespace().to_string();
        let home = Self::preview_or_none(previewer.as_ref(), &request, &home_namespace).await;
        let previews = futures::future::join_all(candidates.iter().map(|candidate| {
            Self::preview_or_none(candidate.previewer.as_ref(), &request, &candidate.namespace)
        }))
        .await;

        match choose(home, &previews) {
            PoolDecision::Home => {
                self.metrics
                    .inc_pool_selection(&self.model_name, PoolDecision::Home);
                self.serve_with(next, request, mirrors, home).await
            }
            PoolDecision::Candidate(index) => {
                let candidate = &candidates[index];
                let chosen = previews[index].expect("chosen candidate has a preview");
                tracing::info!(
                    model = %self.model_name,
                    request_id = %request.context().id(),
                    from = %home_namespace,
                    to = %candidate.namespace,
                    worker_id = chosen.worker.worker_id,
                    dp_rank = chosen.worker.dp_rank,
                    cost = chosen.cost,
                    overlap_blocks = chosen.overlap_blocks,
                    home_cost = home.map(|preview| preview.cost),
                    "Placed request in another worker set"
                );
                self.metrics
                    .inc_pool_selection(&self.model_name, PoolDecision::Candidate(index));
                self.serve_with(
                    candidate.engine.clone(),
                    request,
                    candidate.mirrors.clone(),
                    Some(chosen),
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::pipeline::{AsyncEngine, Context, Error, ResponseStream};
    use futures::stream;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn preview(cost: f64) -> Option<PoolPreview> {
        Some(PoolPreview {
            worker: WorkerWithDpRank::new(1, 0),
            cost,
            overlap_blocks: 0,
        })
    }

    #[test]
    fn cheapest_candidate_wins_and_home_wins_ties() {
        assert_eq!(choose(preview(5.0), &[preview(7.0)]), PoolDecision::Home);
        assert_eq!(
            choose(preview(5.0), &[preview(7.0), preview(2.0)]),
            PoolDecision::Candidate(1)
        );
        assert_eq!(choose(preview(5.0), &[preview(5.0)]), PoolDecision::Home);
        assert_eq!(choose(preview(5.0), &[None, None]), PoolDecision::Home);
    }

    #[test]
    fn home_without_placement_defers_to_a_candidate() {
        assert_eq!(
            choose(None, &[None, preview(9.0)]),
            PoolDecision::Candidate(1)
        );
        assert_eq!(choose(None, &[None]), PoolDecision::Home);
        assert_eq!(choose(None, &[]), PoolDecision::Home);
    }

    struct FixedPreview(Option<PoolPreview>);

    #[async_trait]
    impl PoolPreviewer for FixedPreview {
        async fn preview(&self, _request: &PreprocessedRequest) -> Result<Option<PoolPreview>> {
            Ok(self.0)
        }
    }

    struct FailingPreview;

    #[async_trait]
    impl PoolPreviewer for FailingPreview {
        async fn preview(&self, _request: &PreprocessedRequest) -> Result<Option<PoolPreview>> {
            Err(anyhow::anyhow!("indexer unavailable"))
        }
    }

    /// Counts calls and answers with one token so the caller can tell sets apart.
    struct CountingEngine {
        calls: AtomicUsize,
        token: u32,
    }

    impl CountingEngine {
        fn new(token: u32) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                token,
            })
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CountingEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let context = request.context();
            let output = Annotated::from_data(LLMEngineOutput {
                token_ids: vec![self.token],
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![output])),
                context,
            ))
        }
    }

    struct StaticSource {
        candidates: Vec<PoolCandidate<LLMEngineOutput>>,
        mirrors: Vec<PoolMirror<LLMEngineOutput>>,
    }

    impl PoolSelectionSource for StaticSource {
        fn namespace(&self) -> &str {
            "home"
        }
        fn backend_output_candidates(&self) -> Vec<PoolCandidate<BackendOutput>> {
            Vec::new()
        }
        fn llm_engine_output_candidates(&self) -> Vec<PoolCandidate<LLMEngineOutput>> {
            self.candidates.clone()
        }
        fn llm_engine_output_mirrors(&self) -> Vec<PoolMirror<LLMEngineOutput>> {
            self.mirrors.clone()
        }
    }

    fn request() -> SingleIn<PreprocessedRequest> {
        let request = PreprocessedRequest::builder()
            .model("pool".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap();
        Context::new(request)
    }

    async fn first_token(stream: ManyOut<Annotated<LLMEngineOutput>>) -> u32 {
        use futures::StreamExt;
        let mut stream = stream;
        let item = stream.next().await.expect("one response");
        item.data.expect("data").token_ids[0]
    }

    fn selection(
        home: Option<PoolPreview>,
        candidates: Vec<(Option<PoolPreview>, Arc<CountingEngine>)>,
    ) -> Arc<PoolSelection> {
        let candidates = candidates
            .into_iter()
            .enumerate()
            .map(|(index, (preview, engine))| PoolCandidate {
                namespace: format!("other-{index}"),
                previewer: Arc::new(FixedPreview(preview)) as Arc<dyn PoolPreviewer>,
                engine: engine
                    as ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
                mirrors: Vec::new(),
            })
            .collect();
        PoolSelection::new(
            "pool".to_string(),
            Some(Arc::new(FixedPreview(home)) as Arc<dyn PoolPreviewer>),
            Some(Arc::new(StaticSource {
                candidates,
                mirrors: Vec::new(),
            }) as Arc<dyn PoolSelectionSource>),
            Arc::new(Metrics::new()),
        )
    }

    #[tokio::test]
    async fn request_stays_home_when_home_is_cheapest() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let selection = selection(preview(1.0), vec![(preview(4.0), other_engine.clone())]);
        let stream = Operator::generate(
            selection.as_ref(),
            request(),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
        assert_eq!(home_engine.calls.load(Ordering::SeqCst), 1);
        assert_eq!(other_engine.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn request_moves_to_the_cheaper_set() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let selection = selection(preview(6.0), vec![(preview(2.0), other_engine.clone())]);
        let stream = Operator::generate(
            selection.as_ref(),
            request(),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 20);
        assert_eq!(home_engine.calls.load(Ordering::SeqCst), 0);
        assert_eq!(other_engine.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn request_moves_when_home_has_no_worker() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let selection = selection(None, vec![(preview(9.0), other_engine.clone())]);
        let stream = Operator::generate(
            selection.as_ref(),
            request(),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 20);
    }

    #[tokio::test]
    async fn a_failing_preview_does_not_fail_the_request() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let candidates = vec![PoolCandidate {
            namespace: "other".to_string(),
            previewer: Arc::new(FailingPreview) as Arc<dyn PoolPreviewer>,
            engine: other_engine.clone()
                as ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
            mirrors: Vec::new(),
        }];
        let selection = PoolSelection::new(
            "pool".to_string(),
            Some(Arc::new(FailingPreview) as Arc<dyn PoolPreviewer>),
            Some(Arc::new(StaticSource {
                candidates,
                mirrors: Vec::new(),
            }) as Arc<dyn PoolSelectionSource>),
            Arc::new(Metrics::new()),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            request(),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
    }

    #[tokio::test]
    async fn pinned_and_query_only_requests_stay_home() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let selection = selection(preview(6.0), vec![(preview(2.0), other_engine.clone())]);

        let mut pinned = request();
        pinned.routing_mut().backend_instance_id = Some(7);
        let stream = Operator::generate(
            selection.as_ref(),
            pinned,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);

        let mut query = request();
        query.annotations.push("query_instance_id:true".to_string());
        let stream = Operator::generate(
            selection.as_ref(),
            query,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
        assert_eq!(other_engine.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn without_a_source_every_request_passes_through() {
        let home_engine = CountingEngine::new(10);
        let selection = PoolSelection::new(
            "pool".to_string(),
            Some(Arc::new(FixedPreview(preview(1.0))) as Arc<dyn PoolPreviewer>),
            None,
            Arc::new(Metrics::new()),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            request(),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
    }

    // -- Mirror sets --

    /// A mirror set's engine: records every copy it receives; emits `tokens`
    /// tokens one per `pace`, or fails outright when `tokens` is zero. Like
    /// the real token backend it stops its own context when it reaches its
    /// end, and with `terminal_error` it ends on an error finish reason.
    struct MirrorEngine {
        calls: AtomicUsize,
        copies: Mutex<Vec<PreprocessedRequest>>,
        tokens: usize,
        pace: Duration,
        terminal_error: bool,
    }

    impl MirrorEngine {
        fn new(tokens: usize, pace: Duration) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                copies: Mutex::new(Vec::new()),
                tokens,
                pace,
                terminal_error: false,
            })
        }

        fn failing_at_end(tokens: usize) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                copies: Mutex::new(Vec::new()),
                tokens,
                pace: Duration::from_millis(1),
                terminal_error: true,
            })
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for MirrorEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.copies.lock().unwrap().push(request.content().clone());
            if self.tokens == 0 {
                anyhow::bail!("mirror set has no worker");
            }
            let context = request.context();
            let stop = context.clone();
            let pace = self.pace;
            let tokens = self.tokens;
            let terminal_error = self.terminal_error;
            let stream = stream::unfold(0usize, move |emitted| {
                let stop = stop.clone();
                async move {
                    if emitted == tokens {
                        // The token backend stops the context on its own
                        // stop condition; a copy that ends this way is
                        // complete, not cut off.
                        stop.stop_generating();
                        return terminal_error.then(|| {
                            (
                                Annotated::from_data(LLMEngineOutput {
                                    finish_reason: Some(FinishReason::Error(
                                        "engine gave up".to_string(),
                                    )),
                                    ..Default::default()
                                }),
                                emitted + 1,
                            )
                        });
                    }
                    if emitted > tokens {
                        return None;
                    }
                    tokio::select! {
                        _ = stop.stopped() => None,
                        _ = tokio::time::sleep(pace) => Some((
                            Annotated::from_data(LLMEngineOutput {
                                token_ids: vec![emitted as u32],
                                ..Default::default()
                            }),
                            emitted + 1,
                        )),
                    }
                }
            });
            Ok(ResponseStream::new(Box::pin(stream), context))
        }
    }

    fn mirrors_of(mirrors: Vec<(u64, Arc<MirrorEngine>)>) -> Vec<PoolMirror<LLMEngineOutput>> {
        mirrors
            .into_iter()
            .enumerate()
            .map(|(index, (worker_id, engine))| PoolMirror {
                namespace: format!("mirror-{index}-of-{worker_id}"),
                worker_id,
                engine: engine
                    as ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
            })
            .collect()
    }

    /// Candidates are `(preview, engine, that set's mirrors)`.
    #[allow(clippy::type_complexity)]
    fn mirrored_selection(
        home: Option<PoolPreview>,
        candidates: Vec<(
            Option<PoolPreview>,
            Arc<CountingEngine>,
            Vec<(u64, Arc<MirrorEngine>)>,
        )>,
        mirrors: Vec<(u64, Arc<MirrorEngine>)>,
        metrics: Arc<Metrics>,
    ) -> Arc<PoolSelection> {
        let candidates = candidates
            .into_iter()
            .enumerate()
            .map(|(index, (preview, engine, mirrors))| PoolCandidate {
                namespace: format!("other-{index}"),
                previewer: Arc::new(FixedPreview(preview)) as Arc<dyn PoolPreviewer>,
                engine: engine
                    as ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
                mirrors: mirrors_of(mirrors),
            })
            .collect();
        let mirrors = mirrors_of(mirrors);
        PoolSelection::new(
            "pool".to_string(),
            Some(Arc::new(FixedPreview(home)) as Arc<dyn PoolPreviewer>),
            Some(Arc::new(StaticSource {
                candidates,
                mirrors,
            }) as Arc<dyn PoolSelectionSource>),
            metrics,
        )
    }

    /// A request whose router placed it on `worker_id`.
    fn placed_request(worker_id: u64) -> SingleIn<PreprocessedRequest> {
        let mut request = request();
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_worker(worker_id, Some(0), "decode");
        request.tracker = Some(tracker);
        request
    }

    async fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn request_placed_on_the_shadowed_worker_is_copied_to_the_mirror() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(3, Duration::from_millis(1));
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let request = placed_request(7);
        let primary_tracker = request.tracker.clone().unwrap();
        let stream = Operator::generate(
            selection.as_ref(),
            request,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        // The copy runs on its own task.
        wait_until("copy to start", || {
            mirror_engine.calls.load(Ordering::SeqCst) == 1
        })
        .await;
        let inflight = metrics.mirror_inflight_gauge("pool");

        // The client sees only the home set's answer.
        assert_eq!(first_token(stream).await, 10);
        assert_eq!(home_engine.calls.load(Ordering::SeqCst), 1);

        // The copy carries its own tracker, so the mirror's placement never
        // lands on the real request's record.
        let copy_tracker = mirror_engine.copies.lock().unwrap()[0]
            .tracker
            .clone()
            .unwrap();
        assert!(!Arc::ptr_eq(&copy_tracker, &primary_tracker));
        assert_eq!(primary_tracker.last_selected_worker_id(), Some(7));

        wait_until("copy to finish", || inflight.get() == 0).await;
        let stopped = metrics.get_mirror_request_count("pool", MirrorOutcome::Stopped);
        let completed = metrics.get_mirror_request_count("pool", MirrorOutcome::Completed);
        assert_eq!(stopped + completed, 1);
        assert_eq!(
            metrics.get_mirror_request_count("pool", MirrorOutcome::Failed),
            0
        );
    }

    #[tokio::test]
    async fn request_placed_elsewhere_is_not_copied() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(3, Duration::from_millis(1));
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            Arc::new(Metrics::new()),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(8),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
        assert_eq!(mirror_engine.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn without_a_tracker_the_preview_names_the_shadowed_worker() {
        // `preview()` places on worker 1; a candidate exists but costs more,
        // so the request stays home and is copied to worker 1's mirror.
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let mirror_engine = MirrorEngine::new(1, Duration::from_millis(1));
        let selection = mirrored_selection(
            preview(1.0),
            vec![(preview(4.0), other_engine.clone(), Vec::new())],
            vec![(1, mirror_engine.clone())],
            Arc::new(Metrics::new()),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            request(),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
        wait_until("copy to start", || {
            mirror_engine.calls.load(Ordering::SeqCst) == 1
        })
        .await;
        assert_eq!(other_engine.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn request_moved_to_another_set_is_not_copied() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let mirror_engine = MirrorEngine::new(1, Duration::from_millis(1));
        let selection = mirrored_selection(
            preview(6.0),
            vec![(preview(2.0), other_engine.clone(), Vec::new())],
            vec![(1, mirror_engine.clone())],
            Arc::new(Metrics::new()),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(1),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 20);
        assert_eq!(mirror_engine.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_copy_outlives_the_real_stream() {
        let home_engine = CountingEngine::new(10);
        // The mirror is slower than the real stream, which is dropped at
        // once; the copy still runs to its own end.
        let mirror_engine = MirrorEngine::new(50, Duration::from_millis(1));
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(7),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        drop(stream);
        wait_until("copy to complete", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Completed) == 1
        })
        .await;
        assert_eq!(
            metrics.get_mirror_request_count("pool", MirrorOutcome::Stopped),
            0
        );
        assert_eq!(metrics.mirror_inflight_gauge("pool").get(), 0);
    }

    #[tokio::test]
    async fn the_real_request_stopping_on_its_own_does_not_stop_the_copy() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(50, Duration::from_millis(1));
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let request = placed_request(7);
        let client = request.context();
        let stream = Operator::generate(
            selection.as_ref(),
            request,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        // What the backend does on a local stop condition.
        client.stop_generating();
        drop(stream);
        wait_until("copy to complete", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Completed) == 1
        })
        .await;
        assert_eq!(
            metrics.get_mirror_request_count("pool", MirrorOutcome::Stopped),
            0
        );
    }

    #[tokio::test]
    async fn cancelling_the_client_cancels_the_copy() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(1_000, Duration::from_millis(1));
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let request = placed_request(7);
        let client = request.context();
        let stream = Operator::generate(
            selection.as_ref(),
            request,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        // The stream is still held; only the client's context is killed.
        client.kill();
        wait_until("copy to be stopped", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Stopped) == 1
        })
        .await;
        drop(stream);
    }

    #[tokio::test]
    async fn a_copy_refused_after_the_client_cancelled_counts_as_stopped() {
        let home_engine = CountingEngine::new(10);
        // This mirror refuses every copy, as a router does for a request
        // whose context is killed before its stream exists.
        let mirror_engine = MirrorEngine::new(0, Duration::ZERO);
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let request = placed_request(7);
        request.context().kill();
        let stream = Operator::generate(
            selection.as_ref(),
            request,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        wait_until("copy to be stopped", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Stopped) == 1
        })
        .await;
        assert_eq!(
            metrics.get_mirror_request_count("pool", MirrorOutcome::Failed),
            0
        );
        drop(stream);
    }

    #[tokio::test]
    async fn a_failing_mirror_does_not_touch_the_real_request() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(0, Duration::ZERO);
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(7),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
        wait_until("copy to fail", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Failed) == 1
        })
        .await;
    }

    #[tokio::test]
    async fn a_copy_that_finishes_first_counts_as_completed() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(2, Duration::from_millis(1));
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(7),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        wait_until("copy to complete", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Completed) == 1
        })
        .await;
        assert_eq!(first_token(stream).await, 10);
        assert_eq!(
            metrics.get_mirror_request_count("pool", MirrorOutcome::Stopped),
            0
        );
    }

    #[tokio::test]
    async fn pinned_request_on_the_shadowed_worker_is_copied_without_its_pin() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::new(1, Duration::from_millis(1));
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            Arc::new(Metrics::new()),
        );
        let mut request = placed_request(7);
        request.routing_mut().backend_instance_id = Some(7);
        request.routing_mut().dp_rank = Some(2);
        let stream = Operator::generate(
            selection.as_ref(),
            request,
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 10);
        wait_until("copy to start", || {
            mirror_engine.calls.load(Ordering::SeqCst) == 1
        })
        .await;
        let copy = mirror_engine.copies.lock().unwrap()[0].clone();
        let routing = copy.routing.expect("routing hints are kept");
        assert_eq!(routing.backend_instance_id, None);
        assert_eq!(routing.dp_rank, None);
    }

    #[tokio::test]
    async fn request_placed_in_another_set_is_copied_to_that_sets_mirror() {
        let home_engine = CountingEngine::new(10);
        let other_engine = CountingEngine::new(20);
        let home_mirror = MirrorEngine::new(1, Duration::from_millis(1));
        let other_mirror = MirrorEngine::new(1, Duration::from_millis(1));
        // The router of the other set is the one that recorded worker 5.
        let selection = mirrored_selection(
            preview(6.0),
            vec![(
                preview(2.0),
                other_engine.clone(),
                vec![(5, other_mirror.clone())],
            )],
            vec![(5, home_mirror.clone())],
            Arc::new(Metrics::new()),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(5),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        assert_eq!(first_token(stream).await, 20);
        wait_until("copy to start", || {
            other_mirror.calls.load(Ordering::SeqCst) == 1
        })
        .await;
        assert_eq!(home_mirror.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn every_mirror_of_the_worker_gets_a_copy() {
        let home_engine = CountingEngine::new(10);
        let first = MirrorEngine::new(1, Duration::from_millis(1));
        let second = MirrorEngine::new(1, Duration::from_millis(1));
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, first.clone()), (7, second.clone())],
            metrics.clone(),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(7),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        wait_until("both copies to complete", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Completed) == 2
        })
        .await;
        assert_eq!(first.calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.calls.load(Ordering::SeqCst), 1);
        assert_eq!(first_token(stream).await, 10);
    }

    #[tokio::test]
    async fn an_error_finish_reason_counts_as_a_failed_copy() {
        let home_engine = CountingEngine::new(10);
        let mirror_engine = MirrorEngine::failing_at_end(2);
        let metrics = Arc::new(Metrics::new());
        let selection = mirrored_selection(
            preview(1.0),
            Vec::new(),
            vec![(7, mirror_engine.clone())],
            metrics.clone(),
        );
        let stream = Operator::generate(
            selection.as_ref(),
            placed_request(7),
            home_engine.clone() as ServerStreamingEngine<_, _>,
        )
        .await
        .unwrap();
        wait_until("copy to fail", || {
            metrics.get_mirror_request_count("pool", MirrorOutcome::Failed) == 1
        })
        .await;
        assert_eq!(
            metrics.get_mirror_request_count("pool", MirrorOutcome::Completed),
            0
        );
        assert_eq!(first_token(stream).await, 10);
    }
}
