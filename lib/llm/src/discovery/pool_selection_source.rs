// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The other worker sets of a model that a home set's request may be placed
//! in: every serving set whose router scores requests in the same token space
//! and with the same weights, so its costs compare with the home set's. Also
//! the mirror sets that shadow one of the home set's workers.

use std::sync::Arc;

use super::migration_fallback::TokenCompatibility;
use crate::discovery::{ModelManager, WorkerSet};
use crate::model_card::ModelDeploymentCard;
use crate::pool_selection::{PoolCandidate, PoolMirror, PoolPreviewer, PoolSelectionSource};
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput};

pub struct WorkerSetPoolSelection {
    manager: Arc<ModelManager>,
    model_name: String,
    namespace: String,
    worker_set_key: String,
    compatibility: TokenCompatibility,
    block_size: u32,
    /// The set's router config override, serialised; `None` when it uses the
    /// frontend's config, which every set on this frontend shares.
    router_config: Option<String>,
}

impl WorkerSetPoolSelection {
    pub fn new(
        manager: Arc<ModelManager>,
        card: &ModelDeploymentCard,
        namespace: String,
        worker_set_key: String,
    ) -> Self {
        Self {
            manager,
            model_name: card.display_name.clone(),
            namespace,
            worker_set_key,
            compatibility: TokenCompatibility::of(card),
            block_size: card.kv_cache_block_size,
            router_config: router_config_key(card),
        }
    }

    /// A set whose router scores in this set's token space, block size and
    /// weights, and that has a router to ask. A disaggregated set is left
    /// out: its decode router's cost ignores the prefill placement that
    /// actually decides its latency.
    fn comparable(&self, worker_set: &WorkerSet) -> bool {
        let card = worker_set.card();
        worker_set.kv_router.is_some()
            && !is_disaggregated(worker_set)
            && self
                .compatibility
                .compatible_with(&TokenCompatibility::of(card))
            && card.kv_cache_block_size == self.block_size
            && router_config_key(card) == self.router_config
    }

    fn home_set(&self) -> Option<Arc<WorkerSet>> {
        self.manager
            .get_model(&self.model_name)
            .and_then(|model| model.get_worker_set(&self.worker_set_key))
    }

    fn home_is_disaggregated(&self) -> bool {
        self.home_set()
            .is_some_and(|worker_set| is_disaggregated(&worker_set))
    }

    /// The mirror sets shadowing a worker of `set`, each with the instance
    /// id of the worker it shadows. A mirror naming a worker that is not
    /// live in the set is left out until that worker returns; one naming no
    /// worker follows the lowest live instance id.
    fn mirrors_of(&self, set: &WorkerSet) -> Vec<(Arc<WorkerSet>, u64)> {
        if set.is_mirror() {
            return Vec::new();
        }
        let workers = set.instance_ids();
        self.manager
            .mirror_sets(&self.model_name, set.namespace())
            .into_iter()
            .filter_map(|mirror| {
                let target = mirror.card().mirror_target()?;
                let worker_id = match target.worker_id {
                    Some(worker_id) if workers.contains(&worker_id) => worker_id,
                    Some(worker_id) => {
                        tracing::debug!(
                            model = %self.model_name,
                            shadowed = %set.namespace(),
                            mirror = %mirror.namespace(),
                            worker_id,
                            "Mirror set names a worker that is not live; not mirroring"
                        );
                        return None;
                    }
                    None => workers.iter().copied().min()?,
                };
                Some((mirror, worker_id))
            })
            .collect()
    }

    fn backend_output_mirrors_of(&self, set: &WorkerSet) -> Vec<PoolMirror<BackendOutput>> {
        self.mirrors_of(set)
            .into_iter()
            .filter_map(|(mirror, worker_id)| {
                Some(PoolMirror {
                    namespace: mirror.namespace().to_string(),
                    worker_id,
                    engine: mirror.migration_target_backend_output()?,
                })
            })
            .collect()
    }

    fn llm_engine_output_mirrors_of(&self, set: &WorkerSet) -> Vec<PoolMirror<LLMEngineOutput>> {
        self.mirrors_of(set)
            .into_iter()
            .filter_map(|(mirror, worker_id)| {
                Some(PoolMirror {
                    namespace: mirror.namespace().to_string(),
                    worker_id,
                    engine: mirror.migration_target_llm_output()?,
                })
            })
            .collect()
    }

    fn comparable_sets(&self) -> Vec<Arc<WorkerSet>> {
        if self.home_is_disaggregated() {
            return Vec::new();
        }
        self.manager
            .migration_alternatives(&self.model_name, &self.worker_set_key)
            .into_iter()
            .filter(|worker_set| {
                let comparable = self.comparable(worker_set);
                if !comparable {
                    tracing::debug!(
                        model = %self.model_name,
                        from = %self.namespace,
                        to = %worker_set.namespace(),
                        "Worker set not comparable for pool selection: tokenizer, block size or router config differ"
                    );
                }
                comparable
            })
            .collect()
    }
}

/// A decode set with an active prefill router: its requests are placed on a
/// prefill worker first, which the decode router's cost does not see.
fn is_disaggregated(worker_set: &WorkerSet) -> bool {
    worker_set
        .prefill_router
        .as_ref()
        .is_some_and(|prefill_router| prefill_router.is_activated())
}

fn router_config_key(card: &ModelDeploymentCard) -> Option<String> {
    card.router_config
        .as_ref()
        .and_then(|config| serde_json::to_string(config).ok())
}

impl PoolSelectionSource for WorkerSetPoolSelection {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn backend_output_candidates(&self) -> Vec<PoolCandidate<BackendOutput>> {
        self.comparable_sets()
            .into_iter()
            .filter_map(|worker_set| {
                let previewer = worker_set
                    .kv_router
                    .clone()
                    .map(|router| router as Arc<dyn PoolPreviewer>)?;
                let engine = worker_set.migration_target_backend_output()?;
                Some(PoolCandidate {
                    namespace: worker_set.namespace().to_string(),
                    previewer,
                    engine,
                    mirrors: self.backend_output_mirrors_of(&worker_set),
                })
            })
            .collect()
    }

    fn llm_engine_output_candidates(&self) -> Vec<PoolCandidate<LLMEngineOutput>> {
        self.comparable_sets()
            .into_iter()
            .filter_map(|worker_set| {
                let previewer = worker_set
                    .kv_router
                    .clone()
                    .map(|router| router as Arc<dyn PoolPreviewer>)?;
                let engine = worker_set.migration_target_llm_output()?;
                Some(PoolCandidate {
                    namespace: worker_set.namespace().to_string(),
                    previewer,
                    engine,
                    mirrors: self.llm_engine_output_mirrors_of(&worker_set),
                })
            })
            .collect()
    }

    fn backend_output_mirrors(&self) -> Vec<PoolMirror<BackendOutput>> {
        self.home_set()
            .map(|home| self.backend_output_mirrors_of(&home))
            .unwrap_or_default()
    }

    fn llm_engine_output_mirrors(&self) -> Vec<PoolMirror<LLMEngineOutput>> {
        self.home_set()
            .map(|home| self.llm_engine_output_mirrors_of(&home))
            .unwrap_or_default()
    }
}
