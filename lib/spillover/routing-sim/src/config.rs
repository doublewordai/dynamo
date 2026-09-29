// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scenario files: simulation configuration, spillover policy parameters and assertions.
//!
//! One YAML file describes both the environment (workers, proxies, workload) and the
//! pass/fail criteria, so `cargo test` and the `routing-sim` binary run exactly the same thing.

use std::collections::BTreeMap;

use dw_spillover_policy::{ModelParameters, SpilloverParameters, TierParameters};
use serde::Deserialize;

fn default_seed() -> u64 {
    42
}

fn default_block_size() -> u32 {
    16
}

/// A complete scenario.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    pub duration_seconds: f64,
    #[serde(default = "default_block_size")]
    pub block_size: u32,
    /// Session arrivals per second over time (piecewise linear).
    pub arrival_rate: Vec<RatePoint>,
    #[serde(default)]
    pub hosted_online: Vec<OnlineChange>,
    pub hosted: Vec<HostedConfig>,
    pub proxies: Vec<ProxyConfig>,
    pub workload: WorkloadConfig,
    pub policy: PolicyConfig,
    /// Optional model of the fork's engine-queue admission margin. Absent means the
    /// margin is off and every worker is eligible, matching a worker process with
    /// `DYN_ADMISSION_QUEUE_MARGIN` unset.
    #[serde(default)]
    pub admission: Option<AdmissionConfig>,
    #[serde(default)]
    pub phases: Vec<PhaseConfig>,
    #[serde(default)]
    pub assertions: Assertions,
}

/// Model of the hosted worker's engine-queue admission margin.
///
/// On the fork the margin is a single environment value read per worker process
/// (`DYN_ADMISSION_QUEUE_MARGIN`, `lib/runtime/src/admission_margin.rs`), so a
/// deployment gives every hosted process its own value and there is no frontend
/// override map. The simulation mirrors that with one value that applies to every
/// hosted worker plus optional per-worker overrides keyed by worker id.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionConfig {
    /// Engine-waiting requests at or above which a hosted worker is excluded from
    /// selection. This is the margin a real worker process is launched with.
    pub hosted_queue_margin: u64,
    /// Per-worker margins, overriding `hosted_queue_margin` for the named worker id.
    /// Models giving individual hosted processes different `DYN_ADMISSION_QUEUE_MARGIN`
    /// values (or leaving one unenforced).
    #[serde(default)]
    pub hosted_queue_margin_overrides: BTreeMap<u64, u64>,
}

impl AdmissionConfig {
    /// The margin that applies to `worker_id`.
    pub fn margin_for(&self, worker_id: u64) -> u64 {
        self.hosted_queue_margin_overrides
            .get(&worker_id)
            .copied()
            .unwrap_or(self.hosted_queue_margin)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RatePoint {
    pub time: f64,
    pub rate: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlineChange {
    pub time: f64,
    pub online: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostedConfig {
    pub id: u64,
    pub capacity_blocks: usize,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub max_concurrent_requests: usize,
    /// Fractional decode-rate loss per extra concurrent request. 0 disables batching slowdown.
    #[serde(default)]
    pub batching_slowdown: f64,
}

/// One proxy tier. `workers` ranks are created starting at `dp_rank_start`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyConfig {
    pub tier: String,
    pub dp_rank_start: u32,
    #[serde(default = "one")]
    pub workers: usize,
    pub ttft_seconds: f64,
    #[serde(default)]
    pub ttft_jitter: f64,
    pub decode_tokens_per_second: f64,
    #[serde(default)]
    pub decode_jitter: f64,
    #[serde(default)]
    pub concurrency_limit: Option<usize>,
    pub cache_ttl_seconds: f64,
}

fn one() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadConfig {
    pub system_prompt_tokens: usize,
    pub user_tokens: IntDist,
    pub output_tokens: IntDist,
    pub think_time_seconds: FloatDist,
    pub turns_per_session: IntDist,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntDist {
    pub min: usize,
    pub max: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloatDist {
    pub min: f64,
    pub max: f64,
}

/// Spillover parameters plus the model name they apply to.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    pub model: String,
    pub occupancy_threshold: f64,
    pub hosted_capacity_blocks: f64,
    #[serde(default)]
    pub failover_penalty_blocks: f64,
    #[serde(default)]
    pub pending_weight_blocks: f64,
    #[serde(default)]
    pub tiers: Vec<TierDef>,
    /// When true, no model entry is handed to the policy, so it routes exactly like Dynamo's
    /// default selector (used by the `no_parameters` scenario).
    #[serde(default)]
    pub no_parameters: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierDef {
    pub name: String,
    pub dp_ranks: [u32; 2],
    #[serde(default)]
    pub penalty_blocks: f64,
    #[serde(default)]
    pub weight_blocks: f64,
}

impl PolicyConfig {
    /// The model's parameters regardless of `no_parameters`, for the development stand-in.
    pub fn model_parameters(&self) -> ModelParameters {
        ModelParameters {
            occupancy_threshold: self.occupancy_threshold,
            hosted_capacity_blocks: self.hosted_capacity_blocks,
            failover_penalty_blocks: self.failover_penalty_blocks,
            pending_weight_blocks: self.pending_weight_blocks,
            tiers: self
                .tiers
                .iter()
                .map(|tier| TierParameters {
                    name: tier.name.clone(),
                    dp_ranks: tier.dp_ranks,
                    penalty_blocks: tier.penalty_blocks,
                    weight_blocks: tier.weight_blocks,
                })
                .collect(),
        }
    }

    /// Convert to the policy crate's parameter type. Empty when `no_parameters` is set.
    pub fn parameters(&self) -> SpilloverParameters {
        if self.no_parameters {
            return SpilloverParameters::default();
        }
        SpilloverParameters {
            models: BTreeMap::from([(self.model.clone(), self.model_parameters())]),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseConfig {
    pub name: String,
    pub start: f64,
    pub end: f64,
}

/// Pass/fail criteria. Every field is optional so a scenario states only what it checks.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assertions {
    /// Hosted share over the whole run.
    pub hosted_share_min: Option<f64>,
    /// Proxy share over the whole run.
    pub proxy_share_max: Option<f64>,
    /// Peak hosted decode occupancy over the whole run.
    pub peak_hosted_occupancy_min: Option<f64>,
    /// Proxy share must stay at or below this in every window whose *maximum* hosted occupancy
    /// stays below the policy threshold, i.e. windows that never reach the failover point.
    /// The maximum (not the mean) is deliberate: once the policy correctly spills, spill lowers
    /// occupancy back under the threshold, so a mean-based check would flag correct regulation
    /// during overload as early spillover. Catches spillover happening while hosted is idle.
    pub proxy_share_max_when_hosted_under_threshold: Option<f64>,
    /// Tier shares must be strictly decreasing in this order.
    pub tier_order: Option<Vec<String>>,
    /// Minimum overall share for named tiers.
    #[serde(default)]
    pub tier_share_min: BTreeMap<String, f64>,
    /// Minimum fraction of follow-up turns that return to the previous turn's class (hosted vs
    /// proxy) while that class is still available.
    pub class_stickiness_min: Option<f64>,
    /// Minimum gap between the policy's worker stickiness and `DefaultWorkerSelector`'s on the
    /// same scenario and seed. Negative values require the policy to stay within a tolerance.
    pub worker_stickiness_vs_default_min_delta: Option<f64>,
    pub failures_max: Option<usize>,
    /// Maximum total number of hosted workers excluded by the admission margin, summed over
    /// requests. Catches a margin that steers away from hosted when it should not.
    pub steering_exclusions_max: Option<usize>,
    /// Maximum number of requests refused as 529 because every hosted worker was saturated
    /// and no proxy was available.
    pub admission_529_max: Option<usize>,
    /// Compare every decision against upstream's reference selector. Requires a report built
    /// by `run_scenario_with_default_reference`; a scenario that sets this and is run without a
    /// recorded reference fails the assertion.
    #[serde(default)]
    pub all_decisions_match_default: bool,
    /// Per-phase checks keyed by phase name.
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseAssertion>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseAssertion {
    pub hosted_share_min: Option<f64>,
    pub proxy_share_min: Option<f64>,
    pub hosted_share_max: Option<f64>,
}

impl Scenario {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(serde_yaml::from_str(text)?)
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// All hosted capacity in blocks.
    pub fn total_hosted_capacity(&self) -> f64 {
        self.hosted.iter().map(|h| h.capacity_blocks as f64).sum()
    }
}
