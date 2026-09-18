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
//! The operator sits below the migration operator and above the token
//! backend. A retry after a failed worker re-enters selection, and a request
//! placed in another set runs through that set's pipeline below its own
//! migration operator, the same entry that cross-set migration uses.

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, ManyOut, Operator, PipelineOperator, ServerStreamingEngine,
    SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::http::service::metrics::Metrics;
use crate::kv_router::{KvRouter, protocols::WorkerWithDpRank};
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest};

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

/// Another worker set of the model a request may be placed in.
pub struct PoolCandidate<Resp> {
    pub namespace: String,
    pub previewer: Arc<dyn PoolPreviewer>,
    /// The set's pipeline below its migration operator.
    pub engine: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
}

impl<Resp> Clone for PoolCandidate<Resp> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            previewer: self.previewer.clone(),
            engine: self.engine.clone(),
        }
    }
}

/// Lookup of the other worker sets a home set's request may be placed in,
/// evaluated per request so sets that join or leave are seen at once.
pub trait PoolSelectionSource: Send + Sync {
    /// The home set's namespace, for logs.
    fn namespace(&self) -> &str;
    fn backend_output_candidates(&self) -> Vec<PoolCandidate<BackendOutput>>;
    fn llm_engine_output_candidates(&self) -> Vec<PoolCandidate<LLMEngineOutput>>;
}

/// Response types that have a per-set pipeline below the migration operator.
pub trait PoolCandidates: Sized {
    fn candidates(source: &dyn PoolSelectionSource) -> Vec<PoolCandidate<Self>>;
}

impl PoolCandidates for BackendOutput {
    fn candidates(source: &dyn PoolSelectionSource) -> Vec<PoolCandidate<Self>> {
        source.backend_output_candidates()
    }
}

impl PoolCandidates for LLMEngineOutput {
    fn candidates(source: &dyn PoolSelectionSource) -> Vec<PoolCandidate<Self>> {
        source.llm_engine_output_candidates()
    }
}

/// Where a request goes: its home set, or the candidate at this index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolDecision {
    Home,
    Candidate(usize),
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

    /// A request that names its worker, or only asks which worker it would
    /// get, stays in the set it entered.
    fn stays_home(request: &PreprocessedRequest) -> bool {
        if request.get_annotation_value("query_instance_id").is_some() {
            return true;
        }
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
        let candidates = Resp::candidates(source.as_ref());
        if candidates.is_empty() || Self::stays_home(&request) {
            return next.generate(request).await;
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
                next.generate(request).await
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
                candidate.engine.generate(request).await
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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
            })
            .collect();
        PoolSelection::new(
            "pool".to_string(),
            Some(Arc::new(FixedPreview(home)) as Arc<dyn PoolPreviewer>),
            Some(Arc::new(StaticSource { candidates }) as Arc<dyn PoolSelectionSource>),
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
        }];
        let selection = PoolSelection::new(
            "pool".to_string(),
            Some(Arc::new(FailingPreview) as Arc<dyn PoolPreviewer>),
            Some(Arc::new(StaticSource { candidates }) as Arc<dyn PoolSelectionSource>),
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
}
