// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Placement across a model's worker sets.
//!
//! A model has several worker sets during a rolling update, across regions or
//! with a canary, each with its own router. A request enters one set's
//! pipeline, its home set. Once the request is tokenized, and before any
//! set's encoder, prefill or decode stage has run, this stage asks the home
//! set's router and every comparable set's router what they would charge for
//! the request, and continues in the cheapest set: the rest of the request,
//! prefill included, runs on that set's workers. The cost is the router's
//! own selection logit, so a request stays where its prefix is cached unless
//! another set is clearly better placed or less loaded. The home set wins
//! ties, and wins outright when nothing else can take the request, so a model
//! with one set pays nothing beyond the advisory selection. Requests pinned to
//! a worker and query-only probes pass straight through.

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    pipeline::{ManyOut, Operator, ServerStreamingEngine, SingleIn, async_trait},
    protocols::annotated::Annotated,
};

use crate::discovery::ModelManager;
use crate::http::service::metrics::Metrics;
use crate::kv_router::{AdvisoryPlacement, RoutingHost};
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

/// A set's pipeline below the placement stage: encoder, prefill and router.
pub(crate) type PlacementEngine =
    ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>;

/// A worker set a request may be placed in: its router previews, and the
/// request continues through the set's pipeline below this stage.
#[async_trait]
pub(crate) trait PlacementTarget: Send + Sync {
    fn namespace(&self) -> &str;
    async fn preview(
        &self,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Result<Option<AdvisoryPlacement>>;
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>>;
}

struct WorkerSetTarget {
    namespace: String,
    host: Arc<RoutingHost>,
    entry: PlacementEngine,
}

#[async_trait]
impl PlacementTarget for WorkerSetTarget {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    async fn preview(
        &self,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Result<Option<AdvisoryPlacement>> {
        self.host.preview(request).await
    }

    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        self.entry.generate(request).await
    }
}

/// The other worker sets of a model a request can be placed in, resolved when
/// the request is placed so a set that joined or left since is seen.
pub(crate) trait PlacementCandidates: Send + Sync {
    fn candidates(&self) -> Vec<Arc<dyn PlacementTarget>>;
}

/// What two worker sets must share for a request preprocessed for one to
/// run on the other and for their routers' costs to compare: the same token
/// space (tokenizer), context limit, block size and router configuration.
#[derive(PartialEq, Eq)]
struct Compatibility {
    tokenizer: Option<String>,
    context_length: Option<u32>,
    block_size: u32,
    router_config: Option<String>,
}

impl Compatibility {
    fn of(card: &ModelDeploymentCard) -> Self {
        Self {
            tokenizer: card
                .tokenizer
                .as_ref()
                .map(|tokenizer| tokenizer.checksum()),
            context_length: card.runtime_config.context_length,
            block_size: card.kv_cache_block_size,
            router_config: card
                .router_config
                .as_ref()
                .and_then(|config| serde_json::to_string(config).ok()),
        }
    }
}

/// Every other ready serving set of the model whose router scores in the home
/// set's token space with the same weights, and that is not disaggregated on
/// either side: a decode router's cost ignores the prefill placement that
/// decides its latency. Readiness is the model's own gate, so a decode set
/// whose prefill peer is missing is never a candidate.
struct WorkerSetCandidates {
    manager: Arc<ModelManager>,
    model_name: String,
    home_namespace: String,
    home: Compatibility,
}

impl PlacementCandidates for WorkerSetCandidates {
    fn candidates(&self) -> Vec<Arc<dyn PlacementTarget>> {
        let Some(model) = self.manager.get_model(&self.model_name) else {
            return Vec::new();
        };
        let sets = model.worker_sets();
        let disaggregated = |namespace: &str| {
            sets.iter()
                .any(|set| set.namespace() == namespace && set.is_prefill_set())
        };
        if disaggregated(&self.home_namespace) {
            return Vec::new();
        }
        sets.iter()
            .filter(|set| {
                set.namespace() != self.home_namespace
                    && set.has_decode_engine()
                    && model.is_workers_ready(set.namespace())
                    && !disaggregated(set.namespace())
                    && Compatibility::of(set.card()) == self.home
            })
            .filter_map(|set| {
                Some(Arc::new(WorkerSetTarget {
                    namespace: set.namespace().to_string(),
                    host: set.routing_host.clone()?,
                    entry: set.placement_entry.clone()?,
                }) as Arc<dyn PlacementTarget>)
            })
            .collect()
    }
}

/// The stage that places a request across the model's worker sets. It passes
/// through when built without a home router.
pub struct PoolSelection {
    model_name: String,
    home: Option<Arc<dyn PlacementTarget>>,
    candidates: Option<Arc<dyn PlacementCandidates>>,
    metrics: Option<Arc<Metrics>>,
}

impl PoolSelection {
    /// A stage that places nothing: every request stays in its home set.
    pub fn passthrough() -> Arc<Self> {
        Arc::new(Self {
            model_name: String::new(),
            home: None,
            candidates: None,
            metrics: None,
        })
    }

    /// The stage for one worker set of `model_name` in `namespace`, whose
    /// router is `host` and whose pipeline below this stage is `entry`.
    pub(crate) fn for_worker_set(
        manager: Arc<ModelManager>,
        model_name: String,
        namespace: String,
        card: &ModelDeploymentCard,
        host: Arc<RoutingHost>,
        entry: PlacementEngine,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        let candidates = Arc::new(WorkerSetCandidates {
            manager,
            model_name: model_name.clone(),
            home_namespace: namespace.clone(),
            home: Compatibility::of(card),
        });
        let home = Arc::new(WorkerSetTarget {
            namespace,
            host,
            entry,
        });
        Self::new(model_name, home, candidates, Some(metrics))
    }

    fn new(
        model_name: String,
        home: Arc<dyn PlacementTarget>,
        candidates: Arc<dyn PlacementCandidates>,
        metrics: Option<Arc<Metrics>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            model_name,
            home: Some(home),
            candidates: Some(candidates),
            metrics,
        })
    }

    fn query_only(request: &PreprocessedRequest) -> bool {
        request.get_annotation_value("query_instance_id").is_some()
    }

    /// The other worker sets this stage may place a request in, when it
    /// places at all.
    pub(crate) fn candidates(&self) -> Option<Arc<dyn PlacementCandidates>> {
        self.candidates.clone()
    }

    /// Whether the request names the worker it must run on.
    pub(crate) fn pinned(request: &PreprocessedRequest) -> bool {
        request.routing.as_ref().is_some_and(|hints| {
            hints.backend_instance_id.is_some()
                || hints.decode_worker_id.is_some()
                || hints.prefill_worker_id.is_some()
        })
    }

    /// A set's cost for the request, or `None` when it cannot take it now. A
    /// failed preview is a set not considered, never a failed request.
    async fn cost(
        target: &dyn PlacementTarget,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Option<f64> {
        match target.preview(request).await {
            Ok(preview) => preview.map(|placement| placement.logit),
            Err(error) => {
                tracing::debug!(
                    namespace = target.namespace(),
                    %error,
                    "Placement preview failed; set not considered"
                );
                None
            }
        }
    }

    fn record(&self, placed_elsewhere: bool) {
        if let Some(metrics) = &self.metrics {
            metrics.inc_pool_selection(&self.model_name, placed_elsewhere);
        }
    }
}

/// The cheapest set wins: the index of that candidate, or `None` for home.
/// The home set wins ties, and wins outright when no candidate can take the
/// request. A home set with no placement of its own hands the request to any
/// candidate that has one.
pub(crate) fn choose(home: Option<f64>, candidates: &[Option<f64>]) -> Option<usize> {
    let mut best = home.map(|cost| (None, cost));
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(cost) = candidate else {
            continue;
        };
        if best.is_none_or(|(_, best_cost)| *cost < best_cost) {
            best = Some((Some(index), *cost));
        }
    }
    best.and_then(|(decision, _)| decision)
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for PoolSelection
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (Some(home), Some(candidates)) = (&self.home, &self.candidates) else {
            return next.generate(request).await;
        };
        if Self::query_only(&request) || Self::pinned(&request) {
            return next.generate(request).await;
        }
        let candidates = candidates.candidates();
        if candidates.is_empty() {
            return next.generate(request).await;
        }
        let (home_cost, candidate_costs) = futures::future::join(
            Self::cost(home.as_ref(), &request),
            futures::future::join_all(
                candidates
                    .iter()
                    .map(|candidate| Self::cost(candidate.as_ref(), &request)),
            ),
        )
        .await;
        match choose(home_cost, &candidate_costs) {
            None => {
                self.record(false);
                next.generate(request).await
            }
            Some(index) => {
                let target = &candidates[index];
                tracing::debug!(
                    model = %self.model_name,
                    request_id = %request.context().id(),
                    from = home.namespace(),
                    to = target.namespace(),
                    home_cost = ?home_cost,
                    cost = ?candidate_costs[index],
                    "Placing request in another worker set"
                );
                self.record(true);
                target.generate(request).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::preprocessor::RoutingHints;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_kv_router::protocols::WorkerWithDpRank;
    use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
    use dynamo_runtime::pipeline::{Context, Error};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn cheapest_set_wins_and_home_wins_ties() {
        assert_eq!(choose(Some(2.0), &[Some(3.0), Some(2.0)]), None);
        assert_eq!(choose(Some(2.0), &[Some(3.0), Some(1.5)]), Some(1));
        assert_eq!(choose(Some(2.0), &[None, None]), None);
        assert_eq!(choose(None, &[None, Some(9.0)]), Some(1));
        assert_eq!(choose(None, &[]), None);
    }

    struct FakeTarget {
        namespace: &'static str,
        cost: Option<f64>,
        served: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl PlacementTarget for FakeTarget {
        fn namespace(&self) -> &str {
            self.namespace
        }
        async fn preview(
            &self,
            _request: &SingleIn<PreprocessedRequest>,
        ) -> Result<Option<AdvisoryPlacement>> {
            Ok(self.cost.map(|logit| AdvisoryPlacement {
                worker: WorkerWithDpRank::from_worker_id(1),
                logit,
            }))
        }
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            self.served.fetch_add(1, Ordering::SeqCst);
            Ok(ResponseStream::new(
                Box::pin(futures::stream::empty()),
                request.context(),
            ))
        }
    }

    struct Fixed(Vec<Arc<dyn PlacementTarget>>);
    impl PlacementCandidates for Fixed {
        fn candidates(&self) -> Vec<Arc<dyn PlacementTarget>> {
            self.0.clone()
        }
    }

    struct HomeEngine(Arc<AtomicUsize>);
    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for HomeEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(ResponseStream::new(
                Box::pin(futures::stream::empty()),
                request.context(),
            ))
        }
    }

    fn request(routing: Option<RoutingHints>) -> SingleIn<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(vec![1, 2, 3])
                .stop_conditions(StopConditions::default())
                .sampling_options(SamplingOptions::default())
                .output_options(OutputOptions::default())
                .routing(routing)
                .build()
                .expect("valid request"),
        )
    }

    fn target(
        namespace: &'static str,
        cost: Option<f64>,
    ) -> (Arc<dyn PlacementTarget>, Arc<AtomicUsize>) {
        let served = Arc::new(AtomicUsize::new(0));
        let target = Arc::new(FakeTarget {
            namespace,
            cost,
            served: served.clone(),
        });
        (target, served)
    }

    async fn place(
        home_cost: Option<f64>,
        candidates: Vec<Arc<dyn PlacementTarget>>,
        request: SingleIn<PreprocessedRequest>,
    ) -> usize {
        let (home, _) = target("home", home_cost);
        let stage = PoolSelection::new("m".to_string(), home, Arc::new(Fixed(candidates)), None);
        let served_at_home = Arc::new(AtomicUsize::new(0));
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(HomeEngine(served_at_home.clone()));
        stage.generate(request, next).await.expect("placed");
        served_at_home.load(Ordering::SeqCst)
    }

    #[tokio::test]
    async fn a_cheaper_set_takes_the_request() {
        let (cheaper, served) = target("other", Some(1.0));
        assert_eq!(place(Some(4.0), vec![cheaper], request(None)).await, 0);
        assert_eq!(served.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn home_keeps_the_request_on_a_tie_or_when_nothing_else_can_take_it() {
        let (tie, served_tie) = target("tie", Some(4.0));
        assert_eq!(place(Some(4.0), vec![tie], request(None)).await, 1);
        assert_eq!(served_tie.load(Ordering::SeqCst), 0);
        let (full, served_full) = target("full", None);
        assert_eq!(place(Some(4.0), vec![full], request(None)).await, 1);
        assert_eq!(served_full.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn pinned_requests_pass_through() {
        let (cheaper, served) = target("other", Some(0.0));
        let pinned = request(Some(RoutingHints {
            backend_instance_id: Some(7),
            ..Default::default()
        }));
        assert_eq!(place(Some(9.0), vec![cheaper], pinned).await, 1);
        assert_eq!(served.load(Ordering::SeqCst), 0);
    }
}
