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
/// the primary SGLang workers get [`RouterAdvertisement::primary_args`] and each proxy config
/// carries the same values, so both cards hash equal and stay one worker set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterAdvertisement {
    /// `--router-mode`; spillover requires KV routing.
    pub mode: &'static str,
    pub track_active_blocks: bool,
    pub track_output_blocks: bool,
}

impl RouterAdvertisement {
    /// The SGLang worker CLI flags that make `build_router_config`
    /// (`components/src/dynamo/common/configuration/groups/router_args.py`)
    /// advertise this advertisement on the card. `--router-mode` is required:
    /// without a mode the helper returns `None` and the card carries no config.
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
            // shared_cache_multiplier = 0.5 (the SGLang CLI default); pin it here so a
            // DYN_SHARED_CACHE_MULTIPLIER set on primary workers can never split the set.
            "--shared-cache-multiplier".to_string(),
            "0.5".to_string(),
        ]
    }
}

/// The advertisement emitted for every deployment. One value drives both the
/// primary SGLang flags and the proxy YAML, so the two can never drift.
pub const ROUTER_ADVERTISEMENT: RouterAdvertisement = RouterAdvertisement {
    mode: "kv",
    track_active_blocks: true,
    track_output_blocks: false,
};

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
    /// [`DEFAULT_ADMISSION_QUEUE_MARGIN`]; see `docs/spillover/tuning.md` for the
    /// routing-sim derivation.
    #[serde(default = "default_admission_queue_margin")]
    pub admission_queue_margin: u64,
}

/// Margin used when a deployment does not set one. The routing-sim admission sweep shows the
/// gate's steering knee is single-digit requests for a normal primary worker, so this sits well
/// above the policy's failover point and cannot fire before the policy decides to spill.
/// It is a floor for each primary process, not a per-model frontend value: the fork has no
/// override map.
pub const DEFAULT_ADMISSION_QUEUE_MARGIN: u64 = 256;

fn default_admission_queue_margin() -> u64 {
    DEFAULT_ADMISSION_QUEUE_MARGIN
}

/// Model card facts the proxy must mirror so it joins the SGLang worker set.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCard {
    pub model_path: String,
    pub served_model_names: Vec<String>,
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub kv_block_size: u32,
    pub context_length: u32,
    pub parser_family: ParserFamily,
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
        if deployment.primary.admission_queue_margin == 0 {
            bail!(
                "deployment {name:?}: admission_queue_margin must be at least 1; 0 makes the \
                 engine-queue gate fire on every arrival (use a value above the policy's \
                 failover point)"
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
/// cached. The most a full primary can win back is a cached prefix of the whole context
/// (`ceil(context_length / kv_block_size)` blocks), so a penalty at least that plus the
/// costliest tier always loses to some tier. Below it the threshold is a soft cap: follow-up
/// turns with a long cached prefix stay on a full primary and queue there.
pub fn hard_cap_failover_penalty(deployment: &Deployment) -> f64 {
    let context_blocks = f64::from(deployment.model.context_length)
        / f64::from(deployment.model.kv_block_size.max(1));
    let costliest_tier = deployment
        .tiers
        .iter()
        .map(|tier| tier.penalty_blocks + tier.weight_blocks)
        .fold(0.0, f64::max);
    context_blocks.ceil() + costliest_tier
}

/// Warn when `failover_penalty_blocks` leaves the threshold a soft cap.
fn warn_on_soft_failover_penalty(doc: &DeploymentsFile) {
    for (name, deployment) in &doc.deployments {
        let hard_cap = hard_cap_failover_penalty(deployment);
        if deployment.primary.failover_penalty_blocks < hard_cap {
            eprintln!(
                "warning: deployment {name:?}: failover_penalty_blocks {} is below {hard_cap} \
                 (context blocks + the costliest tier's penalty and weight), so \
                 occupancy_threshold is a soft cap: conversations with a long cached prefix \
                 stay on a full primary worker and queue there. Raise it to at least \
                 {hard_cap} to fail over whenever primary is over the threshold.",
                deployment.primary.failover_penalty_blocks
            );
        }
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
                 ceil(1.5 * max_num_seqs) unless DYN_ENGINE_REQUEST_LIMIT is set.",
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
        // workers get the SGLang flags and the proxies carry the same values in
        // their YAML. See [`ROUTER_ADVERTISEMENT`].
        insert_file(
            &mut files,
            format!("router/{directory}/primary.args"),
            primary_router_args_file(name),
        )?;
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
/// No frontend-wide flag is emitted: each spillover worker set advertises
/// `router_track_active_blocks` on its own model card, so the frontend runs with
/// whatever it already used. A frontend-wide `DYN_ROUTER_TRACK_ACTIVE_BLOCKS=true`
/// (equivalently `--router-track-active-blocks`) also works, but it turns
/// tracking on for every other model the frontend serves.
fn frontend_env(doc: &DeploymentsFile) -> String {
    let mut env = String::from(
        "# No frontend-wide active-block tracking flag.\n\
# This model's worker set advertises `router_track_active_blocks` on the model\n\
# card itself: the primary SGLang workers via router/<model>/primary.args and the\n\
# proxies via their configs' `router_config`. The card checksum includes\n\
# `router_config`, so all workers in the set advertise identical values and stay\n\
# one worker set.\n\
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
    env
}

/// One shell file per deployment carrying the primary SGLang `--router-*` flags.
///
/// The single non-comment, non-empty line is the flags to append to the worker
/// command (for example `xargs` or a shell array). The same values are written
/// into every proxy config's `router_config`, so the cards hash equal.
fn primary_router_args_file(model_name: &str) -> String {
    let args = ROUTER_ADVERTISEMENT.primary_args().join(" ");
    format!(
        "# Primary SGLang worker router flags for {model_name}.\n\
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
/// own value. Proxy workers never report engine waiting, so the margin cannot apply to them and
/// their file clears the variable.
fn admission_env(model_name: &str, deployment: &Deployment) -> (String, String) {
    let margin = deployment.primary.admission_queue_margin;
    let primary = format!(
        "# Primary workers for {model_name}.\n\
# lib/runtime/src/admission_gate.rs reads this from each worker process; the\n\
# frontend does not read it. `export` so sourcing the file without `set -a`\n\
# still reaches the worker process. Set it on every primary worker.\n\
export DYN_ADMISSION_QUEUE_MARGIN={margin}\n"
    );
    let proxy = format!(
        "# Proxy workers for {model_name}. They never report num_waiting_reqs, so the\n\
# engine-queue margin is unenforceable on them. Clear it explicitly so a value\n\
# cannot leak in from a shared launch environment.\n\
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
        dp_rank: replica_rank(index, replica),
        tier: tier.name.clone(),
        parser_family: parser_family_name(deployment.model.parser_family).to_string(),
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
        router_config: Some(ProxyRouterYaml {
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
    context_length: u32,
    dp_rank: u32,
    tier: String,
    parser_family: String,
    provider: ProviderYaml,
    router_config: Option<ProxyRouterYaml>,
    vcache_ttl_secs: u64,
    vcache_max_blocks: usize,
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
