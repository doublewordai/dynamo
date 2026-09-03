// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::discovery::{ModelManager, WorkerSet};
use crate::migration::{MigrationFallbackSource, MigrationTarget};
use crate::model_card::ModelDeploymentCard;
use crate::model_type::{ModelInput, ModelType};
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput};

/// What two worker sets must agree on for one to continue the other's
/// requests. A migrated request replays the token ids the source produced, so
/// the target must read them with the same vocabulary and expose the same
/// surfaces. Worker sets of one model may otherwise differ (that is what lets
/// a rolling update change engine settings).
#[derive(Debug, Clone)]
pub(crate) struct TokenCompatibility {
    /// Checksum of the frontend tokenizer. Without one on both sides the
    /// vocabularies cannot be shown to agree.
    tokenizer: Option<String>,
    model_input: ModelInput,
    model_type: ModelType,
}

impl TokenCompatibility {
    pub(crate) fn of(card: &ModelDeploymentCard) -> Self {
        Self {
            tokenizer: card
                .tokenizer
                .as_ref()
                .map(|tokenizer| tokenizer.checksum()),
            model_input: card.model_input,
            model_type: card.model_type,
        }
    }

    /// True when a request that started on `self` can be replayed on `other`:
    /// both carry a frontend tokenizer, with equal checksums, and expose the
    /// same input and surfaces. An unknown vocabulary is never assumed
    /// compatible.
    pub(crate) fn compatible_with(&self, other: &Self) -> bool {
        self.model_input == other.model_input
            && self.model_type == other.model_type
            && matches!((&self.tokenizer, &other.tokenizer), (Some(a), Some(b)) if a == b)
    }
}

/// Resolves, at retry time, the other worker sets of a model that a request
/// can continue on once the set it started in has no workers left.
pub struct WorkerSetMigrationFallback {
    manager: Arc<ModelManager>,
    model_name: String,
    worker_set_key: String,
    compatibility: TokenCompatibility,
}

impl WorkerSetMigrationFallback {
    /// `card` and `worker_set_key` identify the worker set whose pipeline
    /// this lookup serves.
    pub fn new(
        manager: Arc<ModelManager>,
        card: &ModelDeploymentCard,
        worker_set_key: String,
    ) -> Self {
        Self {
            manager,
            model_name: card.display_name.clone(),
            worker_set_key,
            compatibility: TokenCompatibility::of(card),
        }
    }

    fn compatible_alternatives(&self) -> Vec<Arc<WorkerSet>> {
        self.manager
            .migration_alternatives(&self.model_name, &self.worker_set_key)
            .into_iter()
            .filter(|worker_set| {
                let compatible = self
                    .compatibility
                    .compatible_with(&TokenCompatibility::of(worker_set.card()));
                if !compatible {
                    tracing::debug!(
                        model = %self.model_name,
                        from = %self.worker_set_key,
                        to = %worker_set.namespace(),
                        "Worker set cannot continue requests from this one: vocabulary or surfaces differ"
                    );
                }
                compatible
            })
            .collect()
    }
}

impl MigrationFallbackSource for WorkerSetMigrationFallback {
    fn backend_output_targets(&self) -> Vec<MigrationTarget<BackendOutput>> {
        self.compatible_alternatives()
            .into_iter()
            .filter_map(|worker_set| {
                worker_set
                    .migration_target_backend_output()
                    .map(|engine| MigrationTarget {
                        namespace: worker_set.namespace().to_string(),
                        engine,
                        fallback: worker_set.migration_fallback(),
                    })
            })
            .collect()
    }

    fn llm_engine_output_targets(&self) -> Vec<MigrationTarget<LLMEngineOutput>> {
        self.compatible_alternatives()
            .into_iter()
            .filter_map(|worker_set| {
                worker_set
                    .migration_target_llm_output()
                    .map(|engine| MigrationTarget {
                        namespace: worker_set.namespace().to_string(),
                        engine,
                        fallback: worker_set.migration_fallback(),
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenised(tokenizer: Option<&str>) -> TokenCompatibility {
        TokenCompatibility {
            tokenizer: tokenizer.map(str::to_string),
            model_input: ModelInput::Tokens,
            model_type: ModelType::Chat | ModelType::Completions,
        }
    }

    #[test]
    fn same_tokenizer_and_surfaces_are_compatible() {
        assert!(tokenised(Some("abc")).compatible_with(&tokenised(Some("abc"))));
    }

    #[test]
    fn different_tokenizer_or_surfaces_are_not_compatible() {
        let base = tokenised(Some("abc"));
        assert!(!base.compatible_with(&tokenised(Some("def"))));
        assert!(!base.compatible_with(&TokenCompatibility {
            model_input: ModelInput::Text,
            ..tokenised(Some("abc"))
        }));
        assert!(!base.compatible_with(&TokenCompatibility {
            model_type: ModelType::Chat,
            ..tokenised(Some("abc"))
        }));
    }

    #[test]
    fn unknown_vocabulary_is_never_compatible() {
        assert!(!tokenised(None).compatible_with(&tokenised(None)));
        assert!(!tokenised(None).compatible_with(&tokenised(Some("abc"))));
        assert!(!tokenised(Some("abc")).compatible_with(&tokenised(None)));
        let card = ModelDeploymentCard::with_name_only("m");
        let of_card = TokenCompatibility::of(&card);
        assert_eq!(of_card.tokenizer, None);
        assert!(!of_card.compatible_with(&of_card.clone()));
    }
}
