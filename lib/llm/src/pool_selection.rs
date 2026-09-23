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
//!
//! A mirror worker shadows one worker of a serving set. It publishes a mirror
//! taint naming that worker, and the router never places a request on it
//! unless the request requires that taint. Every request the serving set's
//! router places on the shadowed worker is copied, requiring the taint, into
//! the mirror's own set above its encoder and prefill stages, whose routers
//! place the copy on the mirror; the copy's output is discarded. The mirror thus sees the same requests, in the same
//! order and at the same time, as the shadowed worker, so its engine metrics
//! compare like for like with that worker's. A copy is sent once the real
//! request's stream is open, so it trails the real request by the dispatch
//! setup time and two copies may cross when their real requests' setups do. A
//! copy runs to its own end; only the client giving up on the real request
//! stops it, so a mirror slower than the shadowed worker builds a backlog, and
//! that backlog is what the mirror is there to show. The mirror's router
//! tracks its prefix cache like any of its workers', so a mirror that drops
//! its taint serves its set's traffic with the cache it built.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::{
        Context, ManyOut, Operator, ServerStreamingEngine, SingleIn, async_trait,
        attach_first_response_guard,
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;

use crate::backend::Backend;
use crate::discovery::ModelManager;
use crate::http::service::metrics::Metrics;
use crate::kv_router::{AdvisoryPlacement, RoutingHost};
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use crate::protocols::common::preprocessor::MultimodalData;
use crate::protocols::common::timing::RequestTracker;
use crate::tokenizers::Tokenizer;

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

    /// The mirror workers shadowing workers of the set in `namespace`.
    fn mirrors_of(&self, namespace: &str) -> Vec<Mirror> {
        let _ = namespace;
        Vec::new()
    }
}

/// A mirror worker, reached through its set's router, and the serving worker
/// it shadows.
pub(crate) struct Mirror {
    /// Namespace of the mirror's set.
    pub namespace: String,
    /// The shadowed worker.
    pub worker_id: u64,
    /// The mirror's taint, which a copy requires so only the mirror takes it.
    pub taint: String,
    pub engine: PlacementEngine,
}

/// What became of a request copy in a mirror set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorOutcome {
    /// The mirror set ran the copy to its end.
    Completed,
    /// The real request was killed and the copy stopped with it.
    Stopped,
    /// The mirror set refused or failed the copy.
    Failed,
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

    /// A mirror naming a worker that is not live in the shadowed set is left
    /// out until that worker returns; one naming no worker follows the lowest
    /// live instance id. A mirror's set must be comparable with the home set
    /// like a placement candidate. Only a KV-routed set's router honours
    /// worker taints, so mirrors in any other set are left out. A copy enters
    /// the mirror's set above its encoder and prefill stages, so either set
    /// may be disaggregated: a disaggregated mirror prefills the copy on its
    /// own prefill workers, which carry the same mirror taint as its decode
    /// workers, since the copy requires it at both hops. Mirror workers of one set that name the same
    /// worker share one copy, which the set's router places on one of them.
    fn mirrors_of(&self, namespace: &str) -> Vec<Mirror> {
        let Some(model) = self.manager.get_model(&self.model_name) else {
            return Vec::new();
        };
        let mirrors = model.mirrors_of(namespace);
        if mirrors.is_empty() {
            return Vec::new();
        }
        let sets = model.worker_sets();
        let mut seen = HashSet::new();
        let workers: Vec<u64> = sets
            .iter()
            .filter(|set| set.namespace() == namespace && set.has_decode_engine())
            .flat_map(|set| set.serving_instance_ids())
            .collect();
        mirrors
            .into_iter()
            .filter(|(set, _, _)| {
                set.has_decode_engine() && Compatibility::of(set.card()) == self.home
            })
            .filter_map(|(set, mirror, target)| {
                set.routing_host
                    .as_ref()
                    .filter(|host| host.kv_router_if_enabled().is_some())?;
                let entry = set.placement_entry.clone()?;
                let worker_id = match target.worker_id {
                    Some(worker_id) if workers.contains(&worker_id) => worker_id,
                    Some(worker_id) => {
                        tracing::debug!(
                            model = %self.model_name,
                            shadowed = namespace,
                            mirror,
                            worker_id,
                            "Mirror names a worker that is not live; not mirroring"
                        );
                        return None;
                    }
                    None => workers.iter().copied().min()?,
                };
                Some(Mirror {
                    namespace: set.namespace().to_string(),
                    worker_id,
                    taint: target.taint(),
                    engine: entry,
                })
            })
            .filter(|mirror| seen.insert((mirror.namespace.clone(), mirror.taint.clone())))
            .collect()
    }
}

/// The stage that places a request across the model's worker sets. It passes
/// through when built without a home router.
pub struct PoolSelection {
    model_name: String,
    home: Option<Arc<dyn PlacementTarget>>,
    candidates: Option<Arc<dyn PlacementCandidates>>,
    /// Enforces the frontend's stop conditions on a request copy, as the
    /// pipeline's own backend stage does above this one for the real request.
    stop: Option<Arc<Backend>>,
    metrics: Option<Arc<Metrics>>,
}

impl PoolSelection {
    /// A stage that places nothing: every request stays in its home set.
    pub fn passthrough() -> Arc<Self> {
        Arc::new(Self {
            model_name: String::new(),
            home: None,
            candidates: None,
            stop: None,
            metrics: None,
        })
    }

    /// The stage for one worker set of `model_name` in `namespace`, whose
    /// router is `host` and whose pipeline below this stage is `entry`.
    /// `tokenizer` is the set's, already loaded for its pipelines; without
    /// one no copy can have its stop conditions enforced, so none is sent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_worker_set(
        manager: Arc<ModelManager>,
        model_name: String,
        namespace: String,
        card: &ModelDeploymentCard,
        host: Arc<RoutingHost>,
        entry: PlacementEngine,
        tokenizer: Option<Tokenizer>,
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
        Self::new(
            model_name,
            home,
            candidates,
            tokenizer.map(Backend::from_tokenizer),
            Some(metrics),
        )
    }

    fn new(
        model_name: String,
        home: Arc<dyn PlacementTarget>,
        candidates: Arc<dyn PlacementCandidates>,
        stop: Option<Arc<Backend>>,
        metrics: Option<Arc<Metrics>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            model_name,
            home: Some(home),
            candidates: Some(candidates),
            stop,
            metrics,
        })
    }

    fn query_only(request: &PreprocessedRequest) -> bool {
        request.get_annotation_value("query_instance_id").is_some()
    }

    fn pinned(request: &PreprocessedRequest) -> bool {
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

    /// The set the request continues in: `None` for home.
    async fn place(
        &self,
        home: &dyn PlacementTarget,
        candidates: Vec<Arc<dyn PlacementTarget>>,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Option<Arc<dyn PlacementTarget>> {
        if candidates.is_empty() {
            return None;
        }
        let (home_cost, candidate_costs) = futures::future::join(
            Self::cost(home, request),
            futures::future::join_all(
                candidates
                    .iter()
                    .map(|candidate| Self::cost(candidate.as_ref(), request)),
            ),
        )
        .await;
        let index = choose(home_cost, &candidate_costs);
        self.record(index.is_some());
        let index = index?;
        tracing::debug!(
            model = %self.model_name,
            request_id = %request.context().id(),
            from = home.namespace(),
            to = candidates[index].namespace(),
            home_cost = ?home_cost,
            cost = ?candidate_costs[index],
            "Placing request in another worker set"
        );
        Some(candidates[index].clone())
    }

    /// Send a copy of the request to each mirror shadowing the worker the
    /// router placed it on. A copy outlives the real request's stream; only a
    /// kill of the real request's context stops it.
    fn mirror(
        &self,
        mirrors: Vec<Mirror>,
        mut copy: PreprocessedRequest,
        metadata: BTreeMap<String, String>,
        parent: Arc<dyn AsyncEngineContext>,
    ) {
        let (Some(worker_id), Some(stop)) = (
            copy.tracker.as_ref().and_then(|t| t.decode_worker_id()),
            self.stop.as_ref(),
        ) else {
            return;
        };
        // A pin names a serving worker; the mirror's set's router places the
        // copy on the mirror, the one worker the copy's taint admits it to.
        let routing = copy.routing.get_or_insert_with(Default::default);
        routing.backend_instance_id = None;
        routing.prefill_worker_id = None;
        routing.decode_worker_id = None;
        routing.dp_rank = None;
        routing.prefill_dp_rank = None;
        routing.allowed_worker_ids = None;
        // Media decoded on the frontend lives in registered memory the real
        // request releases when it ends; each copy holds it until its own
        // worker has read it, as a migration retry does.
        let media: Vec<_> = copy
            .multi_modal_data
            .as_ref()
            .into_iter()
            .flat_map(|media| media.values())
            .flatten()
            .filter_map(|item| match item {
                MultimodalData::Decoded(descriptor) => descriptor.source_storage.clone(),
                _ => None,
            })
            .collect();
        let mirrors = mirrors.into_iter().filter(|m| m.worker_id == worker_id);
        for (index, mirror) in mirrors.enumerate() {
            let mut copy = copy.clone();
            // The copy records its own placement, timings and retries;
            // sharing the real request's tracker or migration state would
            // make the mirror's worker the one its metrics and exclusions
            // are attributed to.
            copy.tracker = Some(Arc::new(RequestTracker::new()));
            copy.migration_state = None;
            // Only the mirror's taint constrains the copy. A session binding
            // would name a worker of the mirror's set that the taint excludes.
            if let Some(routing) = copy.routing.as_mut() {
                routing.routing_constraints = Some(RoutingConstraints {
                    required_taints: HashSet::from([mirror.taint.clone()]),
                    ..Default::default()
                });
            }
            let mut shadow = Context::with_id_and_metadata(
                copy,
                format!("{}-mirror-{}-{index}", parent.id(), mirror.namespace),
                metadata.clone(),
            );
            if !media.is_empty() {
                attach_first_response_guard(&mut shadow, Arc::new(media.clone()));
            }
            tracing::debug!(
                model = %self.model_name,
                request_id = %parent.id(),
                mirror = %mirror.namespace,
                worker_id,
                "Copying request to mirror set"
            );
            let job = MirrorJob {
                model: self.model_name.clone(),
                metrics: self.metrics.clone(),
                stop: stop.clone(),
                namespace: mirror.namespace,
                shadowed_worker_id: worker_id,
                parent: parent.clone(),
            };
            tokio::spawn(job.run(mirror.engine, shadow));
        }
    }
}

/// Runs one request copy in a mirror set, discarding its output and
/// recording its outcome.
struct MirrorJob {
    model: String,
    metrics: Option<Arc<Metrics>>,
    stop: Arc<Backend>,
    namespace: String,
    /// The worker the copy shadows.
    shadowed_worker_id: u64,
    /// The real request's context, held for the copy's lifetime so
    /// `killed()` resolves only on a real kill.
    parent: Arc<dyn AsyncEngineContext>,
}

impl MirrorJob {
    async fn run(self, engine: PlacementEngine, shadow: SingleIn<PreprocessedRequest>) {
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
        let outcome = match self.stop.generate(shadow, engine).await {
            Err(error) => {
                tracing::debug!(
                    model = %self.model,
                    request_id = %self.parent.id(),
                    mirror = %self.namespace,
                    %error,
                    "Mirror set refused the request copy"
                );
                if self.parent.is_killed() {
                    MirrorOutcome::Stopped
                } else {
                    MirrorOutcome::Failed
                }
            }
            Ok(mut stream) => {
                let mut failed = false;
                while let Some(item) = stream.next().await {
                    // A cancellation the mirror set raised on its own, with
                    // the real request still running, is a failed copy.
                    failed |= item.is_error()
                        || item.data.as_ref().is_some_and(|out| {
                            matches!(
                                out.finish_reason,
                                Some(FinishReason::Error(_) | FinishReason::Cancelled)
                            )
                        });
                }
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
        if let Some(metrics) = &self.metrics {
            metrics.inc_mirror_request(&self.model, self.shadowed_worker_id, outcome);
        }
        tracing::debug!(
            model = %self.model,
            request_id = %self.parent.id(),
            mirror = %self.namespace,
            ?outcome,
            "Request copy in mirror set ended"
        );
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
        if Self::query_only(&request) {
            return next.generate(request).await;
        }
        let target = if Self::pinned(&request) {
            None
        } else {
            self.place(home.as_ref(), candidates.candidates(), &request)
                .await
        };
        let namespace = target
            .as_ref()
            .map_or(home.namespace(), |target| target.namespace());
        // The router records the worker it chose on the tracker, once, before
        // it returns the stream. A retry after migration finds it recorded
        // already and is not mirrored: the shadowed worker's copy is running
        // to its own end. A first dispatch that fails after selection leaves
        // the worker recorded too, so that request's retry is not mirrored.
        // A request carrying a prefill or encoder handoff is never copied:
        // the handoff has one consumer, whatever the discovery snapshot says
        // about the set's peers now.
        let unplaced = request
            .tracker
            .as_ref()
            .is_some_and(|tracker| tracker.decode_worker_id().is_none());
        let handoff = request.prefill_result.is_some()
            || request.bootstrap_info.is_some()
            || request.encoder_result.is_some()
            || request.staged_kv_cleanup;
        let copy = (unplaced && !handoff)
            .then(|| candidates.mirrors_of(namespace))
            .filter(|mirrors| !mirrors.is_empty())
            .map(|mirrors| {
                (
                    mirrors,
                    (*request).clone(),
                    request.metadata().clone(),
                    request.context(),
                )
            });
        let stream = match &target {
            None => next.generate(request).await?,
            Some(target) => target.generate(request).await?,
        };
        if let Some((mirrors, copy, metadata, parent)) = copy {
            self.mirror(mirrors, copy, metadata, parent);
        }
        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId};
    use crate::protocols::common::preprocessor::RoutingHints;
    use crate::protocols::common::timing::WORKER_TYPE_DECODE;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_kv_router::protocols::WorkerWithDpRank;
    use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
    use dynamo_runtime::pipeline::Error;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn cheapest_set_wins_and_home_wins_ties() {
        assert_eq!(choose(Some(2.0), &[Some(3.0), Some(2.0)]), None);
        assert_eq!(choose(Some(2.0), &[Some(3.0), Some(1.5)]), Some(1));
        assert_eq!(choose(Some(2.0), &[None, None]), None);
        assert_eq!(choose(None, &[None, Some(9.0)]), Some(1));
        assert_eq!(choose(None, &[]), None);
    }

    /// What a fake set saw of the last request it served.
    struct Seen {
        routing: Option<RoutingHints>,
        metadata: BTreeMap<String, String>,
        affinity: Option<String>,
        context: Arc<dyn AsyncEngineContext>,
    }

    struct FakeTarget {
        namespace: &'static str,
        cost: Option<f64>,
        served: Arc<AtomicUsize>,
        last: Mutex<Option<Seen>>,
    }

    impl FakeTarget {
        fn serve(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> ManyOut<Annotated<LLMEngineOutput>> {
            self.served.fetch_add(1, Ordering::SeqCst);
            *self.last.lock().unwrap() = Some(Seen {
                routing: request.routing.clone(),
                metadata: request.metadata().clone(),
                affinity: request
                    .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
                    .unwrap()
                    .map(|affinity| affinity.as_str().to_string()),
                context: request.context(),
            });
            // The stream ends only when the request's context is killed, so
            // a copy's lifetime follows the real request's.
            let context = request.context();
            let stream = futures::stream::once({
                let context = context.clone();
                async move {
                    context.killed().await;
                    Annotated::<LLMEngineOutput>::from_error("killed")
                }
            });
            ResponseStream::new(Box::pin(stream), context)
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for FakeTarget
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Ok(self.serve(request))
        }
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
            Ok(self.serve(request))
        }
    }

    struct Fixed(Vec<Arc<dyn PlacementTarget>>);
    impl PlacementCandidates for Fixed {
        fn candidates(&self) -> Vec<Arc<dyn PlacementTarget>> {
            self.0.clone()
        }
    }

    struct Mirrored(Vec<Mirror>);
    impl PlacementCandidates for Mirrored {
        fn candidates(&self) -> Vec<Arc<dyn PlacementTarget>> {
            Vec::new()
        }
        fn mirrors_of(&self, namespace: &str) -> Vec<Mirror> {
            assert_eq!(namespace, "home");
            self.0
                .iter()
                .map(|mirror| Mirror {
                    namespace: mirror.namespace.clone(),
                    worker_id: mirror.worker_id,
                    taint: mirror.taint.clone(),
                    engine: mirror.engine.clone(),
                })
                .collect()
        }
    }

    /// The home set's router: places every request on worker 1.
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
            if let Some(tracker) = &request.tracker {
                tracker.record_worker(1, None, WORKER_TYPE_DECODE);
            }
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
                .tracker(Some(Arc::new(RequestTracker::new())))
                .build()
                .expect("valid request"),
        )
    }

    fn fake(namespace: &'static str, cost: Option<f64>) -> Arc<FakeTarget> {
        Arc::new(FakeTarget {
            namespace,
            cost,
            served: Arc::new(AtomicUsize::new(0)),
            last: Mutex::new(None),
        })
    }

    fn target(
        namespace: &'static str,
        cost: Option<f64>,
    ) -> (Arc<dyn PlacementTarget>, Arc<AtomicUsize>) {
        let target = fake(namespace, cost);
        let served = target.served.clone();
        (target, served)
    }

    fn stop() -> Arc<Backend> {
        let card = ModelDeploymentCard::load_from_disk(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/sample-models/mock-llama-3.1-8b-instruct"
            ),
            None,
        )
        .expect("mock model card");
        Backend::from_mdc(&card)
    }

    async fn until(condition: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !condition() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("condition within 2 s");
    }

    async fn place(
        home_cost: Option<f64>,
        candidates: Vec<Arc<dyn PlacementTarget>>,
        request: SingleIn<PreprocessedRequest>,
    ) -> usize {
        let (home, _) = target("home", home_cost);
        let stage = PoolSelection::new(
            "m".to_string(),
            home,
            Arc::new(Fixed(candidates)),
            Some(stop()),
            None,
        );
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
    async fn a_request_is_copied_to_the_mirrors_of_its_worker_and_stops_with_it() {
        let (home, _) = target("home", None);
        let of_worker_1 = fake("mirror-1", None);
        let of_worker_2 = fake("mirror-2", None);
        let mirrors = Mirrored(vec![
            Mirror {
                namespace: "mirror-1".into(),
                worker_id: 1,
                taint: "dynamo.pool/mirror-of=home/1".into(),
                engine: of_worker_1.clone(),
            },
            Mirror {
                namespace: "mirror-2".into(),
                worker_id: 2,
                taint: "dynamo.pool/mirror-of=home/2".into(),
                engine: of_worker_2.clone(),
            },
        ]);
        let stage =
            PoolSelection::new("m".to_string(), home, Arc::new(mirrors), Some(stop()), None);
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(HomeEngine(Arc::new(AtomicUsize::new(0))));
        let pinned = request(Some(RoutingHints {
            backend_instance_id: Some(1),
            lora_name: Some("adapter".into()),
            routing_constraints: Some(RoutingConstraints {
                required_taints: HashSet::from(["zone-a".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        }));
        let metadata =
            BTreeMap::from([("x-dynamo-admission-priority".to_string(), "3".to_string())]);
        let pinned = pinned.into_parts();
        let mut pinned = Context::with_id_and_metadata(pinned.0, "req-1".into(), metadata.clone());
        pinned.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("session-7"),
        );
        let parent = pinned.context();
        let stream = stage.generate(pinned, next).await.expect("served");
        until(|| of_worker_1.served.load(Ordering::SeqCst) == 1).await;
        assert_eq!(of_worker_2.served.load(Ordering::SeqCst), 0);

        let seen = of_worker_1
            .last
            .lock()
            .unwrap()
            .take()
            .expect("copy served");
        let routing = seen.routing.expect("hints kept");
        assert_eq!(routing.backend_instance_id, None);
        assert_eq!(routing.lora_name.as_deref(), Some("adapter"));
        assert_eq!(
            routing
                .routing_constraints
                .expect("taint required")
                .required_taints,
            HashSet::from(["dynamo.pool/mirror-of=home/1".to_string()])
        );
        let shadow = seen.context;
        assert_eq!(shadow.id(), "req-1-mirror-mirror-1-0");
        assert_eq!(seen.metadata, metadata);
        assert_eq!(seen.affinity, None);

        // The real stream ending leaves the copy running; a kill stops it.
        drop(stream);
        parent.stop_generating();
        tokio::task::yield_now().await;
        assert!(!shadow.is_killed());
        parent.kill();
        until(|| shadow.is_killed()).await;
    }

    #[tokio::test]
    async fn a_retry_after_migration_is_not_copied_again() {
        let (home, _) = target("home", None);
        let of_worker_1 = fake("mirror-1", None);
        let mirrors = Mirrored(vec![Mirror {
            namespace: "mirror-1".into(),
            worker_id: 1,
            taint: "dynamo.pool/mirror-of=home/1".into(),
            engine: of_worker_1.clone(),
        }]);
        let stage =
            PoolSelection::new("m".to_string(), home, Arc::new(mirrors), Some(stop()), None);
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(HomeEngine(Arc::new(AtomicUsize::new(0))));
        let retry = request(None);
        retry
            .tracker
            .as_ref()
            .unwrap()
            .record_worker(1, None, WORKER_TYPE_DECODE);
        stage.generate(retry, next).await.expect("served");
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(of_worker_1.served.load(Ordering::SeqCst), 0);
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
