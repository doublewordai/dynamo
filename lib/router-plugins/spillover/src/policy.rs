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
    // The tier scorer's hosted-occupancy estimate is router-tracked decode blocks over
    // `hosted_capacity_blocks`. With active-block tracking off those blocks are always zero, so
    // a parametered model would claim to spill but never fail over. Refuse to build the tier
    // policy in that case, say exactly what to change, and fall back to Dynamo's default for
    // this model.
    if model.is_some() && !config.router_track_active_blocks {
        tracing::error!(
            model = model_name,
            setting = "router_track_active_blocks",
            "dw-spillover has parameters for this model but active-block tracking is off, so \
             hosted occupancy would always be zero and failover would never fire; falling back \
             to Dynamo's default policy for this model. Advertise router_track_active_blocks on \
             this worker set's model card: start each hosted worker with \
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
        return WorkerSelectionPolicy::default(config.clone(), role.default_selector_label());
    }
    let mut scorers: Vec<Box<dyn WorkerScorer>> = vec![baseline::baseline_scorer(config, role)];
    if let Some(model) = model {
        scorers.push(Box::new(TierScorer::new(model.clone())));
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
