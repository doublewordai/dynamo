// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider, factory and registration.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerScorer, WorkerSelectionPolicy, WorkerSelectionPolicyFactory,
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
};
use dynamo_kv_router::plugins::{RouterPluginRegistry, WorkerSelectionPolicyRegistryError};
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};

use crate::POLICY_TYPE;
use crate::baseline::{self, PickerRng};
use crate::params::SpilloverParameters;
use crate::tier::TierScorer;

/// Parse and validate parameters once at startup, then return the per-partition factory.
pub fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let params: SpilloverParameters = parameters.deserialize()?;
    params
        .validate()
        .map_err(WorkerSelectionPolicyProviderError::new)?;
    let params = Arc::new(params);
    Ok(Arc::new(
        move |config: &KvRouterConfig, role: WorkerType, partition: RoutingPartitionRef<'_>| {
            build_policy(config, role, partition.model_name, &params, None)
        },
    ))
}

/// Build the policy for one routing partition.
///
/// A model with no entry in `params` gets Dynamo's default policy exactly, so it routes like the
/// built-in selector. A model with parameters stacks our tier scorer after the baseline scorer
/// and keeps the baseline picker.
pub fn build_policy(
    config: &KvRouterConfig,
    role: WorkerType,
    model_name: &str,
    params: &SpilloverParameters,
    rng: PickerRng,
) -> WorkerSelectionPolicy {
    let model = params.for_model(model_name);
    // Direct callers (routing-sim, tests) bypass `provider`, which is the only place parameters
    // were validated. Validate here too so an invalid parameter cannot silently produce NaN costs
    // or disable failover.
    if let Err(error) = params.validate() {
        tracing::error!(
            model = model_name,
            error = %error,
            "dw-spillover parameters are invalid; falling back to Dynamo's default policy"
        );
        return WorkerSelectionPolicy::default(config.clone(), role.default_selector_label());
    }
    // The tier scorer's primary-occupancy estimate is router-tracked decode blocks over the
    // worker's advertised KV capacity (or `primary_capacity_blocks`). With active-block tracking
    // off those blocks are always zero, so a parametered model would claim to spill but never
    // fail over from the KV signal. The concurrency signal still reduces this to a warning: it
    // is live whenever the engine advertises `max_num_seqs`. Refuse to build the tier
    // policy in that case, say exactly what to change, and fall back to Dynamo's default for
    // this model.
    if model.is_some() && !config.router_track_active_blocks {
        tracing::error!(
            model = model_name,
            setting = "router_track_active_blocks",
            "dw-spillover has parameters for this model but active-block tracking is off, so \
             primary occupancy would always be zero and failover would never fire; falling back \
             to Dynamo's default policy for this model. Advertise router_track_active_blocks on \
             this worker set's model card: start each primary worker with \
             --router-track-active-blocks and give the proxies the same router_config \
             (spillover-deploy emits both). A frontend-wide --router-track-active-blocks \
             (or DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true) also covers it but changes tracking for \
             every model on the frontend."
        );
        return WorkerSelectionPolicy::default(config.clone(), role.default_selector_label());
    }
    // Production passes no rng: a model without parameters gets Dynamo's own default policy.
    // Tests and simulations pass a seeded rng, so they get the ported baseline instead, which
    // tests/equivalence.rs proves chooses exactly what DefaultWorkerSelector does, but
    // reproducibly.
    if model.is_none() && rng.is_none() {
        // A model whose key is missing from a non-empty `models` map silently routes like
        // Dynamo's default. That is the expected behaviour for a deliberately parameterless
        // model, so it cannot be an error, but it is also the most common misconfiguration
        // (a typo or a served-model alias in the YAML key). Say what was configured so a typo
        // is visible in the logs instead of only as "spillover never fires".
        if !params.models.is_empty() {
            tracing::warn!(
                model = model_name,
                configured = ?params.models.keys().collect::<Vec<_>>(),
                "dw-spillover has parameters for other models but none for this routing \
                 partition, so this model routes exactly like Dynamo's default policy. Check \
                 the `models` keys in the router-policy YAML against the partition's model \
                 name (the worker set's primary served model name)."
            );
        }
        return WorkerSelectionPolicy::default(config.clone(), role.default_selector_label());
    }
    let mut scorers: Vec<Box<dyn WorkerScorer>> = vec![baseline::baseline_scorer(config, role)];
    if let Some(model) = model {
        scorers.push(Box::new(TierScorer::new(model.clone())));
        tracing::info!(
            model = model_name,
            tiers = ?model
                .tiers
                .iter()
                .map(|tier| (tier.name.as_str(), tier.dp_ranks))
                .collect::<Vec<_>>(),
            occupancy_threshold = model.occupancy_threshold,
            primary_capacity_blocks = ?model.primary_capacity_blocks,
            primary_max_requests = ?model.primary_max_requests,
            "dw-spillover tier policy installed"
        );
        for tier in &model.tiers {
            // The generator reserves tier ranks from 1000 up (`tier_rank_base` in
            // `lib/spillover/deploy/src/lib.rs`). A range that starts lower almost certainly
            // overlaps the primary workers' DP ranks, which would then be scored as this tier.
            if tier.dp_ranks[0] < 1000 {
                tracing::warn!(
                    model = model_name,
                    tier = tier.name.as_str(),
                    dp_ranks = ?tier.dp_ranks,
                    "dw-spillover tier dp_ranks start below 1000; primary data-parallel ranks \
                     inside this range would be treated as this proxy tier. spillover-deploy \
                     assigns tier ranks from 1000 up, so a range starting lower almost certainly \
                     covers primary ranks."
                );
            }
        }
    }
    WorkerSelectionPolicy::new(
        config.clone(),
        role.default_selector_label(),
        scorers,
        baseline::baseline_picker(config, rng),
    )
}

/// Register `dw-spillover` with a router plugin registry.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ModelParameters, TierParameters};
    use tracing_test::traced_test;

    fn model() -> ModelParameters {
        ModelParameters {
            occupancy_threshold: 0.9,
            primary_capacity_blocks: Some(1000.0),
            primary_max_requests: None,
            failover_penalty_blocks: 200.0,
            pending_weight_blocks: 10.0,
            tiers: vec![TierParameters {
                name: "openrouter".into(),
                dp_ranks: [1000, 1999],
                penalty_blocks: 200.0,
                weight_blocks: 8.0,
            }],
        }
    }

    fn tracking_config() -> KvRouterConfig {
        KvRouterConfig {
            router_track_active_blocks: true,
            ..Default::default()
        }
    }

    #[traced_test]
    #[test]
    fn unmatched_model_key_warns_with_model_and_configured_keys() {
        let mut params = SpilloverParameters::default();
        params.models.insert("zai-org/GLM-5.3".into(), model());

        // Production calls `build_policy` with no rng; a missing key must not be silent.
        let _policy = build_policy(
            &tracking_config(),
            WorkerType::Aggregated,
            "my-glm",
            &params,
            None,
        );

        assert!(logs_contain("my-glm"), "warning must name the partition");
        assert!(
            logs_contain("zai-org/GLM-5.3"),
            "warning must list the configured keys"
        );
        assert!(logs_contain("dw-spillover has parameters for other models"));
    }

    #[traced_test]
    #[test]
    fn empty_parameters_do_not_warn_about_a_missing_model_key() {
        let params = SpilloverParameters::default();
        let _policy = build_policy(
            &tracking_config(),
            WorkerType::Aggregated,
            "any-model",
            &params,
            None,
        );
        assert!(!logs_contain(
            "dw-spillover has parameters for other models"
        ));
    }

    #[traced_test]
    #[test]
    fn installed_tier_policy_logs_the_model_and_tiers() {
        let mut params = SpilloverParameters::default();
        params.models.insert("zai-org/GLM-5.3".into(), model());

        let _policy = build_policy(
            &tracking_config(),
            WorkerType::Aggregated,
            "zai-org/GLM-5.3",
            &params,
            None,
        );

        assert!(logs_contain("dw-spillover tier policy installed"));
        assert!(logs_contain("zai-org/GLM-5.3"));
        assert!(logs_contain("openrouter"));
        assert!(logs_contain("1000"));
    }

    #[traced_test]
    #[test]
    fn warns_when_a_tier_range_starts_below_the_generator_base() {
        let mut params = SpilloverParameters::default();
        let mut model = model();
        model.tiers[0].dp_ranks = [1, 1999];
        params.models.insert("zai-org/GLM-5.3".into(), model);

        let _policy = build_policy(
            &tracking_config(),
            WorkerType::Aggregated,
            "zai-org/GLM-5.3",
            &params,
            None,
        );

        assert!(logs_contain("dp_ranks start below 1000"));
        assert!(logs_contain("openrouter"));
        assert!(logs_contain("zai-org/GLM-5.3"));
    }
}
