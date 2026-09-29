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

/// Build the policy for one routing partition. Models without parameters get the baseline only,
/// which makes them route exactly like Dynamo's default policy.
pub fn build_policy(
    config: &KvRouterConfig,
    role: WorkerType,
    model_name: &str,
    params: &SpilloverParameters,
    rng: PickerRng,
) -> WorkerSelectionPolicy {
    let mut scorers: Vec<Box<dyn WorkerScorer>> = vec![baseline::baseline_scorer(config, role)];
    if let Some(model) = params.for_model(model_name) {
        scorers.push(Box::new(TierScorer::new(model.clone())));
    }
    WorkerSelectionPolicy::new(
        config.clone(),
        role.default_selector_label(),
        scorers,
        baseline::baseline_picker(config, rng),
    )
    .with_exclusive_affinity(true)
}

/// Register `dw-spillover` with a router plugin registry.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}
