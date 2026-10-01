// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generate the router-policy YAML and every proxy-worker config from one deployment file.
//!
//! `lib/spillover/deploy/config/deployments.yaml` is the single source of truth for how each
//! model is deployed.
//! From it this crate builds:
//!
//! - `router-policy.yaml`: the `worker_selection` document Dynamo's frontend is started with,
//!   whose `parameters.models` carries the spillover settings.
//! - one `ProxyConfig` YAML per (deployment, tier, replica), placed under a directory named
//!   after the Dynamo model.
//!
//! Tier DP rank ranges are assigned here and used for both outputs, so the policy's
//! `dp_ranks` and the proxy configs' `dp_rank`s can never drift apart. The generated output is
//! validated by parsing it back with the very types that consume it
//! ([`dw_spillover_policy::SpilloverParameters`] and
//! [`dw_proxy_core::config::ProxyConfig`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use dw_proxy_core::cache_key::CacheKeyField;
use dw_proxy_core::circuit_breaker::CircuitBreakerConfig;
use dw_proxy_core::config::ProxyConfig;
use dw_proxy_core::render::ParserFamily;
use dw_proxy_core::thinking::ThinkingDialect;
use dw_spillover_policy::SpilloverParameters;
use dw_spillover_policy::params::MAX_COST_BLOCKS;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Inclusive width of one tier's reserved DP rank range.
pub const RANKS_PER_TIER: u32 = 1000;

/// Reserved rank range for tier `index` (0-based): `[1000 * (index + 1), 1000 * (index + 1) + 999]`.
pub fn tier_rank_base(index: usize) -> u32 {
    RANKS_PER_TIER * (index as u32 + 1)
}

/// Reserved rank for one replica of tier `index`.
pub fn replica_rank(tier_index: usize, replica: u32) -> u32 {
    tier_rank_base(tier_index) + replica
}

/// The card `router_config` every spillover worker set advertises.
///
/// `dw-spillover` estimates primary occupancy from router-tracked decode blocks,
/// which production frontends do not track (`--no-router-track-active-blocks`).
/// Enabling it frontend-wide would change routing for every other model on the
/// frontend, so each spillover deployment advertises it per worker set instead:
/// the primary workers get [`RouterAdvertisement::primary_args`] and each proxy config
/// carries the same values, so both cards hash equal and stay one worker set.
/// TokenSpeed cannot advertise a card `router_config`, so a TokenSpeed deployment
/// emits none; see [`PrimaryEngine::advertises_card_router_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterAdvertisement {
    /// `--router-mode`; spillover requires KV routing.
    pub mode: &'static str,
    pub track_active_blocks: bool,
    pub track_output_blocks: bool,
}

impl RouterAdvertisement {
    /// The primary worker CLI flags that make `build_router_config`
    /// (`components/src/dynamo/common/configuration/groups/router_args.py`)
    /// advertise this advertisement on the card. `--router-mode` is required:
    /// without a mode the helper returns `None` and the card carries no config.
    ///
    /// The flags are shared by every engine that parses worker router config
    /// (SGLang, vLLM, TRT-LLM and the mocker through `parse_worker_router_config`).
    /// TokenSpeed has no such flags and does not advertise a card `router_config`.
    pub fn primary_args(&self) -> Vec<String> {
        vec![
            "--router-mode".to_string(),
            self.mode.to_string(),
            if self.track_active_blocks {
                "--router-track-active-blocks"
            } else {
                "--no-router-track-active-blocks"
            }
            .to_string(),
            if self.track_output_blocks {
                "--router-track-output-blocks"
            } else {
                "--no-router-track-output-blocks"
            }
            .to_string(),
            // The card checksum includes the whole KvRouterConfig. The proxy advertises
            // shared_cache_multiplier = 0.5 (the worker CLI default for every engine); pin
            // it here so a DYN_SHARED_CACHE_MULTIPLIER set on primary workers can never
            // split the set.
            "--shared-cache-multiplier".to_string(),
            "0.5".to_string(),
        ]
    }
}

/// The advertisement emitted for every deployment. One value drives both the
/// primary worker flags and the proxy YAML, so the two can never drift.
pub const ROUTER_ADVERTISEMENT: RouterAdvertisement = RouterAdvertisement {
    mode: "kv",
    track_active_blocks: true,
    track_output_blocks: false,
};

/// The inference engine running the primary workers.
///
/// The engine decides which card fields a proxy can mirror and how the
/// engine-queue admission margin behaves, because engines differ in what they
/// advertise (model aliases, a card `router_config`) and in whether they report
/// their waiting queue to the admission gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrimaryEngine {
    /// SGLang. Reports waiting queue, registers aliases, advertises card router
    /// config, supports EAGLE/MTP.
    Sglang,
    /// vLLM. Reports waiting queue, registers aliases, advertises card router config.
    Vllm,
    /// TensorRT-LLM. Registers no aliases and advertises card router config; reports
    /// waiting only when started with `--publish-metrics`.
    Trtllm,
    /// The GPU-free mock engine. Registers no aliases but advertises card router
    /// config; never reports waiting.
    Mocker,
    /// TokenSpeed. Registers no aliases and cannot advertise a card router config
    /// (`lib/bindings/python/rust/backend.rs` hard-codes `None`); never reports
    /// waiting.
    Tokenspeed,
}

impl PrimaryEngine {
    /// Every accepted value, for error messages.
    pub const ALL: [PrimaryEngine; 5] = [
        PrimaryEngine::Sglang,
        PrimaryEngine::Vllm,
        PrimaryEngine::Trtllm,
        PrimaryEngine::Mocker,
        PrimaryEngine::Tokenspeed,
    ];

    /// The `primary.engine` spelling.
    pub fn name(self) -> &'static str {
        match self {
            PrimaryEngine::Sglang => "sglang",
            PrimaryEngine::Vllm => "vllm",
            PrimaryEngine::Trtllm => "trtllm",
            PrimaryEngine::Mocker => "mocker",
            PrimaryEngine::Tokenspeed => "tokenspeed",
        }
    }

    /// Whether the engine reports its own waiting queue (`report_engine_waiting`),
    /// the signal `DYN_ADMISSION_QUEUE_MARGIN` bounds. SGLang and vLLM always do;
    /// TRT-LLM does only with `--publish-metrics` (checked separately so the
    /// generated file can carry the caveat); mocker and TokenSpeed never do.
    pub fn reports_engine_waiting(self) -> bool {
        matches!(self, PrimaryEngine::Sglang | PrimaryEngine::Vllm)
    }

    /// Whether the engine registers model aliases beyond its primary served name.
    /// TRT-LLM, mocker and TokenSpeed register none, so a proxy with more than one
    /// `served_model_names` entry would advertise aliases the primary lacks and
    /// split the worker set.
    pub fn registers_model_aliases(self) -> bool {
        matches!(self, PrimaryEngine::Sglang | PrimaryEngine::Vllm)
    }

    /// Whether the engine can advertise a `router_config` on its model card.
    /// TokenSpeed cannot, so its deployments emit no primary router args file and
    /// no proxy `router_config`.
    pub fn advertises_card_router_config(self) -> bool {
        !matches!(self, PrimaryEngine::Tokenspeed)
    }

    /// Whether EAGLE/MTP speculative decoding applies. Only SGLang supports it;
    /// `enable_eagle` on another engine's deployment is warned about.
    pub fn supports_eagle(self) -> bool {
        matches!(self, PrimaryEngine::Sglang)
    }
}

/// Top-level shape of `deployments.yaml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentsFile {
    /// Keyed by the Dynamo model name, e.g. `zai-org/GLM-5.3`.
    pub deployments: BTreeMap<String, Deployment>,
}

/// One Dynamo deployment: primary settings, the model card the proxies mirror, and proxy tiers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    pub primary: PrimarySettings,
    pub model: ModelCard,
    /// Proxy virtual-cache lifetime in seconds; defaults to `ProxyConfig`'s 300.
    #[serde(default)]
    pub vcache_ttl_secs: Option<u64>,
    /// Proxy virtual-cache block cap; defaults to `ProxyConfig`'s 1,000,000.
    #[serde(default)]
    pub vcache_max_blocks: Option<usize>,
    pub tiers: Vec<Tier>,
}

/// Settings that go into the router policy's per-model parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimarySettings {
    /// The inference engine running the primary workers. Required, because the
    /// engine decides what the proxies can mirror and how the admission margin
    /// behaves.
    pub engine: PrimaryEngine,
    /// Fallback KV capacity of one primary rank, in blocks, used only when the
    /// router cannot read a positive `total_kv_blocks` from the worker's
    /// advertised runtime config. Omitted means "rely on the advertised
    /// capacity"; set it to a value greater than 0 to force a denominator.
    #[serde(default)]
    pub primary_capacity_blocks: Option<f64>,
    /// Fraction of a primary worker's advertised capacity at which it counts as
    /// full. In (0, 4]; values above 1.0 are allowed only when the frontend
    /// admission-queue margin is large enough to hold the implied queue.
    pub occupancy_threshold: f64,
    /// Fallback maximum concurrently scheduled sequences for one primary rank,
    /// used only when the worker advertises no `max_num_seqs`. Omitted means
    /// "rely on the advertised value"; set it to a value greater than 0.
    #[serde(default)]
    pub primary_max_requests: Option<u64>,
    pub failover_penalty_blocks: f64,
    pub pending_weight_blocks: f64,
    /// Engine-queue admission margin for every primary worker process
    /// (`DYN_ADMISSION_QUEUE_MARGIN`, in engine-waiting requests). Defaults to
    /// [`DEFAULT_ADMISSION_QUEUE_MARGIN`] for engines that report their waiting
    /// queue (SGLang and vLLM, and TRT-LLM with `--publish-metrics`). Must be
    /// omitted for `mocker` and `tokenspeed`, which never report waiting: a set
    /// margin there *removes* admission control rather than bounding it, and the
    /// generator rejects it. See `docs/spillover/tuning.md` for the
    /// routing-sim derivation.
    #[serde(default)]
    pub admission_queue_margin: Option<u64>,
}

/// Margin used when a deployment does not set one. The routing-sim admission sweep shows the
/// gate's steering knee is single-digit requests for a normal primary worker, so this sits well
/// above the policy's failover point and cannot fire before the policy decides to spill.
/// It is a floor for each primary process, not a per-model frontend value: the fork has no
/// override map.
pub const DEFAULT_ADMISSION_QUEUE_MARGIN: u64 = 256;

/// Model card facts the proxy must mirror so it joins the primary worker set.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCard {
    pub model_path: String,
    pub served_model_names: Vec<String>,
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub kv_block_size: u32,
    /// Context length advertised on the model card. `Some(n)` (n > 0) mirrors a
    /// primary started with `--context-length` (SGLang), `--max-model-len`
    /// (vLLM/mocker), `--max-seq-len` (TRT-LLM) or the TokenSpeed cache length.
    /// `None` mirrors a primary that advertises nothing and lets the card fall
    /// back to the model's architectural maximum. vLLM always publishes its
    /// resolved `max_model_len`, so set it and match that number.
    #[serde(default)]
    pub context_length: Option<u32>,
    pub parser_family: ParserFamily,
    /// Optional path to the same custom Jinja chat template the primary workers
    /// were started with (`--custom-jinja-template`). It is part of the card's
    /// chat-template checksum, so a primary with a custom template must be
    /// mirrored with the same file or the two split the worker set.
    #[serde(default)]
    pub custom_jinja_template: Option<PathBuf>,
    /// Whether the primary workers emit bigram-keyed KV events for EAGLE/MTP
    /// speculative decoding. The router hashes prompts differently for EAGLE, so
    /// this **must equal the primary's**. Only SGLang supports it; setting it on
    /// another engine's deployment warns.
    #[serde(default)]
    pub enable_eagle: bool,
    /// Endpoint types every generated proxy advertises, a comma-separated subset
    /// of {chat, completions}.
    ///
    /// This feeds the card's `model_type`, which is part of `worker_set_key`, so
    /// it must equal the primary workers' advertisement or the proxies and
    /// primaries land in different WorkerSets. Defaults to `chat,completions`,
    /// the `WorkerConfig` default production primaries use.
    #[serde(default = "default_endpoint_types")]
    pub endpoint_types: String,
}

fn default_endpoint_types() -> String {
    "chat,completions".to_string()
}

/// A proxy tier: how many proxies to spawn and which provider they call.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tier {
    pub name: String,
    pub provider: ProviderInput,
    pub penalty_blocks: f64,
    pub weight_blocks: f64,
    pub replicas: u32,
}

/// Provider connection settings, a subset of `dw_proxy_core::upstream::ProviderConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInput {
    pub name: String,
    pub base_url: String,
    pub api_key_env: String,
    pub model: String,
    #[serde(default)]
    pub provider_preferences: Option<Value>,
    /// How this provider expects thinking to be requested (`dw_proxy_core::thinking`).
    #[serde(default)]
    pub thinking_dialect: Option<ThinkingDialect>,
    /// Retry a request on a primary worker when the dialect cannot express its thinking choice.
    #[serde(default)]
    pub thinking_strict: Option<bool>,
    /// Which field carries the opaque provider cache key (`dw_proxy_core::cache_key`).
    #[serde(default)]
    pub cache_key: Option<CacheKeyField>,
    /// Environment variable holding the cache key secret; required with `cache_key`.
    #[serde(default)]
    pub cache_key_secret_env: Option<String>,
    /// Per-proxy circuit breaker. Omitted, the proxy uses
    /// [`CircuitBreakerConfig::default`].
    #[serde(default)]
    pub circuit_breaker: Option<CircuitBreakerConfig>,
}

/// Path of the file that records what the last `generate` wrote, so a later run can prune
/// exactly the files it owns without touching unrelated files in `--out`.
pub const MANIFEST_FILE: &str = ".generated-files";

/// Reject a deployment file that could never produce consistent output.
pub fn validate_input(doc: &DeploymentsFile) -> anyhow::Result<()> {
    if doc.deployments.is_empty() {
        bail!("deployments must not be empty");
    }
    let mut deployment_dirs: BTreeMap<String, String> = BTreeMap::new();
    for (name, deployment) in &doc.deployments {
        if name.trim().is_empty() {
            bail!("deployment names must not be empty");
        }
        let directory = safe_component(name, "deployment name")?;
        if let Some(other) = deployment_dirs.insert(directory.clone(), name.clone()) {
            bail!(
                "deployment names {other:?} and {name:?} both map to directory {directory:?} \
                 after sanitizing"
            );
        }
        if deployment.primary.admission_queue_margin == Some(0) {
            bail!(
                "deployment {name:?}: admission_queue_margin must be at least 1; 0 makes the \
                 engine-queue gate fire on every arrival (use a value above the policy's \
                 failover point)"
            );
        }
        // Engines that never report their waiting queue cannot enforce the margin; a set
        // value *removes* the default concurrency limit instead of bounding anything.
        if deployment.primary.admission_queue_margin.is_some()
            && !deployment.primary.engine.reports_engine_waiting()
            && deployment.primary.engine != PrimaryEngine::Trtllm
        {
            bail!(
                "deployment {name:?}: primary.engine {} never reports its engine waiting \
                 queue, so DYN_ADMISSION_QUEUE_MARGIN cannot be enforced; omit \
                 admission_queue_margin (setting one removes the default concurrency limit \
                 rather than adding an admission bound)",
                deployment.primary.engine.name()
            );
        }
        // TRT-LLM, mocker and TokenSpeed register no model aliases. A proxy advertising one
        // would carry a served name the primary does not, so `worker_set_key` splits the set.
        if !deployment.primary.engine.registers_model_aliases()
            && deployment.model.served_model_names.len() > 1
        {
            bail!(
                "deployment {name:?}: primary.engine {} registers no model aliases, so \
                 served_model_names may name only the primary model; it has {} entries {:?}",
                deployment.primary.engine.name(),
                deployment.model.served_model_names.len(),
                deployment.model.served_model_names
            );
        }
        if !deployment.primary.occupancy_threshold.is_finite()
            || deployment.primary.occupancy_threshold <= 0.0
            || deployment.primary.occupancy_threshold > 4.0
        {
            bail!(
                "deployment {name:?}: occupancy_threshold must be in (0, 4]; a value above 1.0 \
                 requires the frontend admission-queue margin to be large enough to hold the \
                 queue it implies"
            );
        }
        if let Some(capacity) = deployment.primary.primary_capacity_blocks
            && (!capacity.is_finite() || capacity <= 0.0)
        {
            bail!(
                "deployment {name:?}: primary_capacity_blocks must be a positive finite number \
                 when set"
            );
        }
        if deployment.primary.primary_max_requests == Some(0) {
            bail!("deployment {name:?}: primary_max_requests must be greater than 0 when set");
        }
        if deployment.model.kv_block_size == 0 {
            bail!(
                "deployment {name:?}: model.kv_block_size must be greater than 0; it is a \
                 divisor in the policy's occupancy and hard-cap math"
            );
        }
        if deployment.model.context_length == Some(0) {
            bail!(
                "deployment {name:?}: model.context_length must be greater than 0 when set; \
                 omit it to let the card fall back to the model's architectural maximum"
            );
        }
        // The policy's own bounds: a finite value in [0, MAX_COST_BLOCKS]. Values outside
        // this range are rejected at policy load, but `build()` must not emit a tree that
        // `validate_dir` then rejects, so check them here too.
        for (field, value) in [
            (
                "primary.failover_penalty_blocks",
                deployment.primary.failover_penalty_blocks,
            ),
            (
                "primary.pending_weight_blocks",
                deployment.primary.pending_weight_blocks,
            ),
        ] {
            if !value.is_finite() || !(0.0..=MAX_COST_BLOCKS).contains(&value) {
                bail!(
                    "deployment {name:?}: {field} must be a finite number in [0, \
                     {MAX_COST_BLOCKS}] blocks"
                );
            }
        }
        // The router keys the spillover policy by the worker set's primary served name,
        // which is `served_model_names[0]`, while the generator keys it by the deployment
        // name. They must be the same string or the policy silently never matches.
        match deployment.model.served_model_names.first() {
            Some(primary) if primary == name => {}
            Some(primary) => bail!(
                "deployment {name:?}: served_model_names[0] must be the Dynamo model name \
                 {name:?}, the primary name the router keys the spillover policy by, but it is \
                 {primary:?}"
            ),
            None => bail!("deployment {name:?}: served_model_names must not be empty"),
        }
        dw_proxy_core::config::validate_endpoint_types(&deployment.model.endpoint_types)
            .map_err(|error| anyhow::anyhow!("deployment {name:?}: {error}"))?;
        if deployment.tiers.is_empty() {
            bail!("deployment {name:?}: at least one proxy tier is required");
        }
        let mut seen = BTreeSet::new();
        let mut seen_dirs = BTreeMap::new();
        for (index, tier) in deployment.tiers.iter().enumerate() {
            if tier.name.trim().is_empty() {
                bail!("deployment {name:?}: tier[{index}].name must not be empty");
            }
            let tier_dir = safe_component(&tier.name, "tier name")?;
            if !seen.insert(tier.name.as_str()) {
                bail!("deployment {name:?}: duplicate tier name {:?}", tier.name);
            }
            if let Some(other) = seen_dirs.insert(tier_dir.clone(), tier.name.clone()) {
                bail!(
                    "deployment {name:?}: tier names {other:?} and {:?} both map to file stem \
                     {tier_dir:?} after sanitizing",
                    tier.name
                );
            }
            if tier
                .provider
                .cache_key
                .is_some_and(|field| field != CacheKeyField::None)
                && tier.provider.cache_key_secret_env.is_none()
            {
                bail!(
                    "deployment {name:?}: tier {:?}: provider.cache_key requires \
                     provider.cache_key_secret_env",
                    tier.name
                );
            }
            if tier.replicas == 0 {
                bail!(
                    "deployment {name:?}: tier {:?} replicas must be at least 1",
                    tier.name
                );
            }
            for (field, value) in [
                ("penalty_blocks", tier.penalty_blocks),
                ("weight_blocks", tier.weight_blocks),
            ] {
                if !value.is_finite() || !(0.0..=MAX_COST_BLOCKS).contains(&value) {
                    bail!(
                        "deployment {name:?}: tier {:?} {field} must be a finite number in \
                         [0, {MAX_COST_BLOCKS}] blocks",
                        tier.name
                    );
                }
            }
            if tier.replicas > RANKS_PER_TIER {
                bail!(
                    "deployment {name:?}: tier {:?} replicas must fit in one reserved range of \
                     {RANKS_PER_TIER} ranks",
                    tier.name
                );
            }
        }
    }
    Ok(())
}

/// Smallest `failover_penalty_blocks` that makes `occupancy_threshold` a hard cap.
///
/// Once a primary worker is over the threshold it carries `failover_penalty_blocks`, and a
/// provider tier carries `penalty_blocks + weight_blocks` plus the full prompt it has not
/// cached. With the default overlap weights and `prefill_load_scale` 1, the most a full primary
/// can win back is a cached prefix of the whole context (`ceil(context_length / kv_block_size)`
/// blocks), so a penalty above that plus the costliest tier always loses to some tier (one block
/// more than the sum, so a tie cannot go to the full primary on the picker's tie-break). This
/// floor ignores two terms: `prefill_load_scale` multiplies the context advantage (scale the
/// context term if it is raised), and `pending_weight_blocks` charges both sides by concurrency
/// (add it times the concurrency a spill tier reaches). Below the floor the threshold is a soft
/// cap: follow-up turns with a long cached prefix stay on a full primary and queue there.
pub fn hard_cap_failover_penalty(deployment: &Deployment) -> Option<f64> {
    let context_length = deployment.model.context_length?;
    let context_blocks =
        f64::from(context_length) / f64::from(deployment.model.kv_block_size.max(1));
    let costliest_tier = deployment
        .tiers
        .iter()
        .map(|tier| tier.penalty_blocks + tier.weight_blocks)
        .fold(0.0, f64::max);
    Some(context_blocks.ceil() + costliest_tier + 1.0)
}

/// Warn when `failover_penalty_blocks` leaves the threshold a soft cap.
///
/// The floor needs `context_length`; without it there is no bounded context term,
/// so the check is skipped rather than guessed.
fn warn_on_soft_failover_penalty(doc: &DeploymentsFile) {
    for (name, deployment) in &doc.deployments {
        let Some(hard_cap) = hard_cap_failover_penalty(deployment) else {
            continue;
        };
        if deployment.primary.failover_penalty_blocks < hard_cap {
            eprintln!(
                "warning: deployment {name:?}: failover_penalty_blocks {} is below {hard_cap} \
                 (context blocks + the costliest tier's penalty and weight, assuming default \
                 overlap weights with prefill_load_scale 1 and ignoring the pending-weight term), \
                 so occupancy_threshold is a soft cap: conversations with a long cached prefix \
                 stay on a full primary worker and queue there. Raise it to at least \
                 {hard_cap} to fail over whenever primary is over the threshold; scale the \
                 context term by prefill_load_scale and add pending_weight_blocks times the \
                 concurrency a spill tier reaches for the true floor.",
                deployment.primary.failover_penalty_blocks
            );
        }
    }
}

/// Warn about engine/field combinations that silently break card matching.
///
/// - vLLM always publishes its resolved `max_model_len`, so a proxy that omits
///   `context_length` advertises a different value and splits the worker set.
/// - EAGLE/MTP (`enable_eagle`) is SGLang-only; setting it on another engine is
///   almost certainly a copy-paste error, though it is emitted either way so a
///   hypothetical engine that supports it still mirrors.
pub fn engine_field_warnings(doc: &DeploymentsFile) -> Vec<String> {
    let mut warnings = Vec::new();
    for (name, deployment) in &doc.deployments {
        let engine = deployment.primary.engine;
        if engine == PrimaryEngine::Vllm && deployment.model.context_length.is_none() {
            warnings.push(format!(
                "deployment {name:?}: primary.engine vllm always publishes its \
                 resolved max_model_len, so the proxy card must set model.context_length to \
                 the same number; omitting it lets the card fall back to the model's \
                 architectural maximum and split the worker set."
            ));
        }
        if deployment.model.enable_eagle && !engine.supports_eagle() {
            warnings.push(format!(
                "deployment {name:?}: model.enable_eagle is set but primary.engine {} does not \
                 support EAGLE/MTP (only sglang does); it is mirrored anyway, but check the \
                 primary actually emits bigram-keyed KV events.",
                engine.name()
            ));
        }
        if engine == PrimaryEngine::Trtllm {
            warnings.push(format!(
                "deployment {name:?}: primary.engine trtllm publishes its engine waiting queue \
                 only with --publish-metrics; without that flag on every primary worker \
                 DYN_ADMISSION_QUEUE_MARGIN is unenforced, and because a set margin replaces the \
                 default concurrency limit, admission control is effectively removed. See \
                 admission/<model>/primary.env."
            ));
        }
    }
    warnings
}

/// Print [`engine_field_warnings`] to stderr.
fn warn_on_engine_field_mismatches(doc: &DeploymentsFile) {
    for warning in engine_field_warnings(doc) {
        eprintln!("warning: {warning}");
    }
}

/// Warn when an `occupancy_threshold` above 1.0 is a deliberate over-subscription.
///
/// The policy counts a primary worker full once its projected occupancy (the larger of decode
/// blocks over `total_kv_blocks` and requests over `max_num_seqs`) exceeds the threshold. A
/// value above 1.0 therefore lets the engine queue before the policy spills, which is only safe
/// if the worker's admission settings can hold that queue, so print a per-deployment warning
/// naming the environment variables.
fn warn_on_high_occupancy_thresholds(doc: &DeploymentsFile) {
    for (name, deployment) in &doc.deployments {
        if deployment.primary.occupancy_threshold > 1.0 {
            eprintln!(
                "warning: deployment {name:?}: occupancy_threshold {} is above 1.0, so primary \
                 workers take requests beyond their advertised capacity before the policy \
                 spills, and the excess queues in the engine. Make sure \
                 DYN_ADMISSION_QUEUE_MARGIN (see admission/<model>/primary.env) is large enough \
                 to hold that queue; the worker admission gate also queues anything above \
                 ceil(1.5 * max_num_seqs * data_parallel_size) unless DYN_ENGINE_REQUEST_LIMIT is set.",
                deployment.primary.occupancy_threshold
            );
        }
    }
}

/// Paths of the generated files, relative to the output directory, mapped to their contents.
pub fn build(input: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    let raw = fs::read_to_string(input)
        .with_context(|| format!("reading deployments {}", input.display()))?;
    let doc: DeploymentsFile = serde_yaml::from_str(&raw)
        .with_context(|| format!("parsing deployments {}", input.display()))?;
    validate_input(&doc)?;
    warn_on_high_occupancy_thresholds(&doc);
    warn_on_soft_failover_penalty(&doc);
    warn_on_engine_field_mismatches(&doc);

    let models = doc
        .deployments
        .iter()
        .map(|(name, deployment)| (name.clone(), generated_model(deployment)))
        .collect();
    let policy = RouterPolicyDoc {
        worker_selection: WorkerSelection {
            aggregated: dw_spillover_policy::POLICY_TYPE.to_string(),
            instances: vec![Instance {
                name: dw_spillover_policy::POLICY_TYPE.to_string(),
                kind: dw_spillover_policy::POLICY_TYPE.to_string(),
                parameters: GeneratedParameters { models },
            }],
        },
    };

    let mut files = BTreeMap::new();
    insert_file(
        &mut files,
        "router-policy.yaml".to_string(),
        serde_yaml::to_string(&policy).context("serializing router-policy.yaml")?,
    )?;
    insert_file(&mut files, "frontend.env".to_string(), frontend_env(&doc))?;

    for (name, deployment) in &doc.deployments {
        let directory = sanitize(name);
        // The card `router_config` is advertised per worker set, so the primary
        // workers get the primary router flags and the proxies carry the same values in
        // their YAML. TokenSpeed cannot advertise a card `router_config`, so it emits
        // neither. See [`ROUTER_ADVERTISEMENT`] and
        // [`PrimaryEngine::advertises_card_router_config`].
        if deployment.primary.engine.advertises_card_router_config() {
            insert_file(
                &mut files,
                format!("router/{directory}/primary.args"),
                primary_router_args_file(name),
            )?;
        }
        // The margin is read per worker process, so emit it as environment files: primary
        // workers get DYN_ADMISSION_QUEUE_MARGIN, proxies get an explicit opt-out so a value
        // cannot leak in from a shared launch environment.
        let (primary_env, proxy_env) = admission_env(name, deployment);
        insert_file(
            &mut files,
            format!("admission/{directory}/primary.env"),
            primary_env,
        )?;
        insert_file(
            &mut files,
            format!("admission/{directory}/proxy.env"),
            proxy_env,
        )?;
        for (index, tier) in deployment.tiers.iter().enumerate() {
            for replica in 0..tier.replicas {
                let config = proxy_config(deployment, tier, index, replica);
                let path = format!("{directory}/{}-{replica}.yaml", sanitize(&tier.name));
                insert_file(
                    &mut files,
                    path,
                    serde_yaml::to_string(&config).context("serializing a proxy config")?,
                )?;
            }
        }
    }
    // The manifest lets a later `generate` prune exactly the files this run owns. It lists
    // the generated paths (not itself) so `validate_dir` can validate precisely this set.
    let manifest: String = files.keys().map(|path| format!("{path}\n")).collect();
    files.insert(MANIFEST_FILE.to_string(), manifest);
    // Every committed file in this repository carries the SPDX header (copyright-check).
    Ok(files
        .into_iter()
        .map(|(path, contents)| (path, format!("{SPDX_HEADER}{contents}")))
        .collect())
}

/// Insert one generated file, refusing to silently overwrite an earlier entry.
///
/// Output paths are keyed on `sanitize`d deployment and tier names, and `validate_input`
/// rejects raw names that collide, but this is the backstop that makes any future path
/// collision a hard error rather than a lost config.
fn insert_file(
    files: &mut BTreeMap<String, String>,
    path: String,
    contents: String,
) -> anyhow::Result<()> {
    if files.contains_key(&path) {
        bail!("generated path {path:?} was produced by two inputs; names collide after sanitizing");
    }
    files.insert(path, contents);
    Ok(())
}

/// SPDX header prepended to every generated file; `#` comments suit YAML, env and args files.
const SPDX_HEADER: &str = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. \
All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n\n";

/// The frontend environment note.
///
/// For engines that advertise a card `router_config`, no frontend-wide flag is
/// emitted: each spillover worker set advertises `router_track_active_blocks` on
/// its own model card, so the frontend runs with whatever it already used. A
/// frontend-wide `DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true` (equivalently
/// `--router-track-active-blocks`) also works, but it turns tracking on for every
/// other model the frontend serves. TokenSpeed cannot advertise a card config, so
/// its deployment needs the global flag; the note says so.
fn frontend_env(doc: &DeploymentsFile) -> String {
    let mut env = String::from(
        "# No frontend-wide active-block tracking flag for the per-set engines.\n\
# Each spillover worker set advertises `router_track_active_blocks` on the model\n\
# card itself: the primary workers via router/<model>/primary.args (when the\n\
# engine parses worker router flags) and the proxies via their configs'\n\
# `router_config`. The card checksum includes `router_config`, so all workers in\n\
# the set advertise identical values and stay one worker set.\n\
#\n\
# A frontend-wide DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true (equivalently\n\
# --router-track-active-blocks) also works but changes tracking for every other\n\
# model on the frontend, which is why this deployment does not rely on it.\n\
#\n\
# The models below use the dw-spillover policy and set tracking per worker set:\n",
    );
    for name in doc.deployments.keys() {
        env.push_str("#   - ");
        env.push_str(name);
        env.push('\n');
    }
    for (name, deployment) in &doc.deployments {
        if !deployment.primary.engine.advertises_card_router_config() {
            env.push_str(&format!(
                "#\n# {name} runs on tokenspeed, which cannot advertise a card router_config.\n\
# Start the frontend with --router-track-active-blocks (or\n\
# DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true) globally so the occupancy KV signal works\n\
# for its worker set; this also turns tracking on for every other model. The\n\
# per-request concurrency signal still works without it.\n"
            ));
        }
    }
    env
}

/// One shell file per deployment carrying the primary worker `--router-*` flags.
///
/// The single non-comment, non-empty line is the flags to append to the worker
/// command (for example `xargs` or a shell array). The same values are written
/// into every proxy config's `router_config`, so the cards hash equal. The flags
/// are shared by SGLang, vLLM, TRT-LLM and the mocker through
/// `parse_worker_router_config`; TokenSpeed has no such flags and does not get
/// this file.
fn primary_router_args_file(model_name: &str) -> String {
    let args = ROUTER_ADVERTISEMENT.primary_args().join(" ");
    format!(
        "# Primary worker router flags for {model_name}.\n\
# Append them to every primary worker's command line so its model card carries\n\
# the worker set's `router_config`. The proxies advertise the same values, and\n\
# the card checksum includes `router_config`, so a mismatch splits the set.\n\
# `--router-mode` is required: without it `build_router_config` advertises\n\
# nothing and the frontend-wide config applies.\n\
{args}\n"
    )
}

/// Environment files that carry the engine-queue admission margin to a deployment's workers.
///
/// The fork reads `DYN_ADMISSION_QUEUE_MARGIN` from the worker process
/// (`lib/runtime/src/admission_gate.rs`); the frontend never reads it and there is no per-model
/// override (no `DYN_ADMISSION_QUEUE_MARGIN_OVERRIDES`). Each primary worker therefore gets its
/// own value, but only when its engine can enforce it:
///
/// - SGLang and vLLM always report their waiting queue, so they get the value.
/// - TRT-LLM reports only with `--publish-metrics`; it gets the value with a comment (and a
///   stderr warning) naming that requirement. A set margin replaces the default concurrency
///   limit, so without the flag admission control is effectively removed.
/// - mocker and TokenSpeed never report waiting. They get an explicit unset: setting a margin
///   would merely remove the default concurrency limit. `validate_input` rejects an explicit
///   margin for them.
///
/// Proxy workers never report engine waiting at all, so their file always clears the variable.
fn admission_env(model_name: &str, deployment: &Deployment) -> (String, String) {
    let engine = deployment.primary.engine;
    let margin = || {
        deployment
            .primary
            .admission_queue_margin
            .unwrap_or(DEFAULT_ADMISSION_QUEUE_MARGIN)
    };
    let primary = match engine {
        PrimaryEngine::Sglang | PrimaryEngine::Vllm => format!(
            "# Primary workers for {model_name} ({}).\n\
# lib/runtime/src/admission_gate.rs reads this from each worker process; the\n\
# frontend does not read it. `export` so sourcing the file without `set -a`\n\
# still reaches the worker process. Set it on every primary worker.\n\
# {} reports its engine waiting queue, so the margin is enforceable.\n\
# The value bounds how many requests may sit in the engine's own queue before\n\
# the worker is excluded from selection. Keep it above the policy's failover\n\
# point so the policy decides to spill first.\n\
# 0 is rejected: the runtime reads a present 0 as an always-firing gate.\n\
# Source docs/spillover/tuning.md for the default's derivation.\n\
# See admission/<model>/proxy.env for the proxy opt-out.\n\
\n\
export DYN_ADMISSION_QUEUE_MARGIN={}\n",
            engine.name(),
            engine.name(),
            margin()
        ),
        PrimaryEngine::Trtllm => format!(
            "# Primary workers for {model_name} (trtllm).\n\
# lib/runtime/src/admission_gate.rs reads this from each worker process; the\n\
# frontend does not read it. TRT-LLM reports its engine waiting queue only when\n\
# started with --publish-metrics. Without that flag the margin is unenforced,\n\
# and because a set margin replaces the default concurrency limit, admission\n\
# control is effectively removed. Keep this only if --publish-metrics is on\n\
# every primary worker.\n\
\n\
export DYN_ADMISSION_QUEUE_MARGIN={}\n",
            margin()
        ),
        PrimaryEngine::Mocker | PrimaryEngine::Tokenspeed => format!(
            "# Primary workers for {model_name} ({}).\n\
# {} never reports its engine waiting queue, so DYN_ADMISSION_QUEUE_MARGIN cannot\n\
# be enforced: the estimate stays at zero and a set margin would replace the\n\
# default concurrency limit rather than bound the engine queue. Clear any value\n\
# inherited from a shared launch environment.\n\
\n\
unset DYN_ADMISSION_QUEUE_MARGIN\n",
            engine.name(),
            engine.name()
        ),
    };
    let proxy = format!(
        "# Proxy workers for {model_name}. They never report num_waiting_reqs, so the\n\
# engine-queue margin is unenforceable on them. Clear it explicitly so a value\n\
# cannot leak in from a shared launch environment.\n\
\n\
unset DYN_ADMISSION_QUEUE_MARGIN\n"
    );
    (primary, proxy)
}

/// Write every generated file, creating directories as needed.
///
/// The `.generated-files` manifest is written last: it is the sole record of what a run
/// owns, so if writing any config fails, the previous manifest still describes the previous
/// (intact) tree and a later run can still prune it correctly.
pub fn write_files(out: &Path, files: &BTreeMap<String, String>) -> anyhow::Result<()> {
    let mut ordered: Vec<(&String, &String)> = files.iter().collect();
    ordered.sort_by_key(|(path, _)| *path == MANIFEST_FILE);
    for (relative, contents) in ordered {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!("refusing to write generated path outside --out: {relative:?}");
        }
        let path = out.join(relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// Load the generated output from `dir` with the consuming types, rejecting unusable configs.
///
/// Only the files named by the `.generated-files` manifest are validated, so unrelated YAML
/// in the output directory cannot fail generation. Every proxy's `dp_rank`/`tier` is
/// cross-checked against the policy, and every policy tier must have at least one proxy, which
/// catches a dropped or mis-ranked config.
pub fn validate_dir(dir: &Path) -> anyhow::Result<()> {
    let manifest_path = dir.join(MANIFEST_FILE);
    let manifest_raw = fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let listed = manifest_paths(&manifest_raw);
    if listed.is_empty() {
        bail!("{} lists no generated files", manifest_path.display());
    }

    let policy_path = dir.join("router-policy.yaml");
    let raw = fs::read_to_string(&policy_path)
        .with_context(|| format!("reading {}", policy_path.display()))?;
    let value: Value =
        serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", policy_path.display()))?;
    let parameters = value
        .pointer("/worker_selection/instances/0/parameters")
        .cloned()
        .with_context(|| {
            format!(
                "{} has no worker_selection.instances[0].parameters",
                policy_path.display()
            )
        })?;
    let params: SpilloverParameters =
        serde_json::from_value(parameters).context("parsing generated policy parameters")?;
    params.validate().map_err(|error| anyhow::anyhow!(error))?;

    let mut ranks_by_model: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
    let mut tiers_by_model: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut checked = 0usize;
    for relative in &listed {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!("manifest lists a path outside {dir:?}: {relative:?}");
        }
        if relative == "router-policy.yaml"
            || relative_path.extension().and_then(|e| e.to_str()) != Some("yaml")
        {
            continue;
        }
        let proxy = ProxyConfig::load(&dir.join(relative_path))?;
        // Match the *primary* served name, exactly as the router keys the policy at
        // runtime; matching any alias would accept an output tree the runtime ignores.
        let primary = proxy
            .served_model_names
            .first()
            .with_context(|| format!("{relative}: served_model_names must not be empty"))?;
        let model = params.models.get_key_value(primary).with_context(|| {
            format!(
                "{relative}: primary served name {primary:?} (served_model_names[0]) matches no \
                 policy model"
            )
        })?;
        let model = model.0;
        let model_params = &params.models[model];
        let tier = model_params.tier_for_rank(proxy.dp_rank).with_context(|| {
            format!(
                "{relative}: dp_rank {} falls in no tier of {model:?}",
                proxy.dp_rank
            )
        })?;
        if tier.name != proxy.tier {
            bail!(
                "{relative}: dp_rank {} is in tier {:?} but the config says {:?}",
                proxy.dp_rank,
                tier.name,
                proxy.tier
            );
        }
        if !ranks_by_model
            .entry(model.clone())
            .or_default()
            .insert(proxy.dp_rank)
        {
            bail!(
                "{relative}: dp_rank {} is assigned to two proxy configs of {model:?}",
                proxy.dp_rank
            );
        }
        tiers_by_model
            .entry(model.clone())
            .or_default()
            .insert(tier.name.clone());
        checked += 1;
    }
    if checked == 0 {
        bail!("{} contains no proxy configs", dir.display());
    }
    for (model, model_params) in &params.models {
        for tier in &model_params.tiers {
            if !tiers_by_model
                .get(model)
                .is_some_and(|seen| seen.contains(&tier.name))
            {
                bail!("model {model:?}: tier {:?} has no proxy config", tier.name);
            }
        }
    }
    Ok(())
}

/// Parse a `.generated-files` manifest: every non-comment, non-empty line.
fn manifest_paths(raw: &str) -> Vec<String> {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// `generate`: build and validate a staging tree first, then write it into `out` and only
/// afterwards remove what the previous run wrote that this run no longer emits.
///
/// Validating before touching `out` means an input the generator itself rejects can never
/// delete or overwrite the operator's last-good generated tree. Stale pruning runs last so a
/// failure while writing the new tree leaves the previous files in place.
pub fn generate(input: &Path, out: &Path) -> anyhow::Result<()> {
    let files = build(input)?;
    let staging = tempfile::tempdir().context("creating a staging directory")?;
    write_files(staging.path(), &files)?;
    validate_dir(staging.path())?;

    // Read the previous manifest before `write_files` overwrites it; the new tree's manifest
    // lists only the new files.
    let stale = stale_files(out, &files)?;
    write_files(out, &files)?;
    remove_files(out, &stale)?;
    Ok(())
}

/// Paths the previous `.generated-files` manifest lists that this run no longer emits.
///
/// Only files a previous `generate` wrote are considered, so unrelated files in `out` are
/// never touched. A fresh `out` has no manifest and nothing is stale.
fn stale_files(out: &Path, files: &BTreeMap<String, String>) -> anyhow::Result<Vec<String>> {
    let manifest_path = out.join(MANIFEST_FILE);
    let Ok(raw) = fs::read_to_string(&manifest_path) else {
        return Ok(Vec::new());
    };
    Ok(manifest_paths(&raw)
        .into_iter()
        .filter(|relative| !files.contains_key(relative))
        .collect())
}

/// Remove stale generated files and the now-empty directories that held them.
fn remove_files(out: &Path, stale: &[String]) -> anyhow::Result<()> {
    let mut directories: BTreeSet<PathBuf> = BTreeSet::new();
    for relative in stale {
        let path = out.join(relative);
        if path.is_file() {
            fs::remove_file(&path).with_context(|| format!("removing stale {}", path.display()))?;
        }
        if let Some(parent) = path.parent()
            && parent != out
        {
            directories.insert(parent.to_path_buf());
        }
    }
    // Deepest first; `remove_dir` fails (ignored) for directories that still hold files.
    for directory in directories.into_iter().rev() {
        let _ = fs::remove_dir(&directory);
    }
    Ok(())
}

/// `check`: build and validate without touching the requested output directory.
pub fn check(input: &Path) -> anyhow::Result<()> {
    let files = build(input)?;
    let temp = tempfile::tempdir().context("creating a temporary directory")?;
    write_files(temp.path(), &files)?;
    validate_dir(temp.path())?;
    Ok(())
}

fn generated_model(deployment: &Deployment) -> GeneratedModel {
    GeneratedModel {
        occupancy_threshold: deployment.primary.occupancy_threshold,
        primary_capacity_blocks: deployment.primary.primary_capacity_blocks,
        primary_max_requests: deployment.primary.primary_max_requests,
        failover_penalty_blocks: deployment.primary.failover_penalty_blocks,
        pending_weight_blocks: deployment.primary.pending_weight_blocks,
        tiers: deployment
            .tiers
            .iter()
            .enumerate()
            .map(|(index, tier)| {
                let base = tier_rank_base(index);
                GeneratedTier {
                    name: tier.name.clone(),
                    dp_ranks: [base, base + RANKS_PER_TIER - 1],
                    penalty_blocks: tier.penalty_blocks,
                    weight_blocks: tier.weight_blocks,
                }
            })
            .collect(),
    }
}

fn proxy_config(deployment: &Deployment, tier: &Tier, index: usize, replica: u32) -> ProxyYaml {
    ProxyYaml {
        model_path: deployment.model.model_path.clone(),
        served_model_names: deployment.model.served_model_names.clone(),
        namespace: deployment.model.namespace.clone(),
        component: deployment.model.component.clone(),
        endpoint: deployment.model.endpoint.clone(),
        kv_block_size: deployment.model.kv_block_size,
        context_length: deployment.model.context_length,
        custom_jinja_template: deployment.model.custom_jinja_template.clone(),
        enable_eagle: deployment.model.enable_eagle,
        dp_rank: replica_rank(index, replica),
        tier: tier.name.clone(),
        parser_family: parser_family_name(deployment.model.parser_family).to_string(),
        endpoint_types: deployment.model.endpoint_types.clone(),
        provider: ProviderYaml {
            name: tier.provider.name.clone(),
            base_url: tier.provider.base_url.clone(),
            api_key_env: tier.provider.api_key_env.clone(),
            model: tier.provider.model.clone(),
            provider_preferences: tier.provider.provider_preferences.as_ref().map(sorted_keys),
            thinking_dialect: tier.provider.thinking_dialect,
            thinking_strict: tier.provider.thinking_strict,
            cache_key: tier.provider.cache_key,
            cache_key_secret_env: tier.provider.cache_key_secret_env.clone(),
            circuit_breaker: tier.provider.circuit_breaker,
        },
        // TokenSpeed cannot advertise a card `router_config`; every other engine
        // advertises the worker set's shared values so the card checksums match.
        router_config: deployment
            .primary
            .engine
            .advertises_card_router_config()
            .then(|| ProxyRouterYaml {
                mode: ROUTER_ADVERTISEMENT.mode.to_string(),
                track_active_blocks: ROUTER_ADVERTISEMENT.track_active_blocks,
                track_output_blocks: ROUTER_ADVERTISEMENT.track_output_blocks,
            }),
        vcache_ttl_secs: deployment.vcache_ttl_secs.unwrap_or(300),
        vcache_max_blocks: deployment.vcache_max_blocks.unwrap_or(1_000_000),
    }
}

/// Rebuild `value` with object keys inserted in sorted order. serde_json keeps insertion
/// order when another workspace crate enables its `preserve_order` feature, so without this
/// the generated files would depend on which crates are built together.
fn sorted_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|key| (key.clone(), sorted_keys(&map[key])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted_keys).collect()),
        other => other.clone(),
    }
}

fn parser_family_name(family: ParserFamily) -> &'static str {
    match family {
        ParserFamily::Glm47 => "glm47",
        ParserFamily::DeepseekV41 => "deepseek_v41",
        ParserFamily::KimiK3 => "kimi_k3",
        ParserFamily::Hermes => "hermes",
    }
}

/// Replace anything that is not filesystem-safe with `_`.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Validate a raw deployment or tier name as one filesystem path component and return its
/// sanitized form. `sanitize` maps every other character to `_`, so the only unsafe results
/// are empty, `.` and `..`, which `write_files` would resolve outside `--out`.
fn safe_component(raw: &str, what: &str) -> anyhow::Result<String> {
    let sanitized = sanitize(raw);
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        bail!("{what} {raw:?} sanitizes to unsafe path component {sanitized:?}");
    }
    Ok(sanitized)
}

#[derive(Debug, Serialize)]
struct RouterPolicyDoc {
    worker_selection: WorkerSelection,
}

#[derive(Debug, Serialize)]
struct WorkerSelection {
    aggregated: String,
    instances: Vec<Instance>,
}

#[derive(Debug, Serialize)]
struct Instance {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    parameters: GeneratedParameters,
}

#[derive(Debug, Serialize)]
struct GeneratedParameters {
    models: BTreeMap<String, GeneratedModel>,
}

#[derive(Debug, Serialize)]
struct GeneratedModel {
    occupancy_threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    primary_capacity_blocks: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    primary_max_requests: Option<u64>,
    failover_penalty_blocks: f64,
    pending_weight_blocks: f64,
    tiers: Vec<GeneratedTier>,
}

#[derive(Debug, Serialize)]
struct GeneratedTier {
    name: String,
    dp_ranks: [u32; 2],
    penalty_blocks: f64,
    weight_blocks: f64,
}

#[derive(Debug, Serialize)]
struct ProxyYaml {
    model_path: String,
    served_model_names: Vec<String>,
    namespace: String,
    component: String,
    endpoint: String,
    kv_block_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_jinja_template: Option<PathBuf>,
    #[serde(skip_serializing_if = "is_false")]
    enable_eagle: bool,
    dp_rank: u32,
    tier: String,
    parser_family: String,
    endpoint_types: String,
    provider: ProviderYaml,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_config: Option<ProxyRouterYaml>,
    vcache_ttl_secs: u64,
    vcache_max_blocks: usize,
}

/// Skip a `false` bool when serializing so a mirrored default is not written out.
fn is_false(value: &bool) -> bool {
    !*value
}

/// Mirrors `dw_proxy_core::config::ProxyRouterConfig`'s serialized shape.
#[derive(Debug, Serialize)]
struct ProxyRouterYaml {
    mode: String,
    track_active_blocks: bool,
    track_output_blocks: bool,
}

#[derive(Debug, Serialize)]
struct ProviderYaml {
    name: String,
    base_url: String,
    api_key_env: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_preferences: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_dialect: Option<ThinkingDialect>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_key: Option<CacheKeyField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_key_secret_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    circuit_breaker: Option<CircuitBreakerConfig>,
}
