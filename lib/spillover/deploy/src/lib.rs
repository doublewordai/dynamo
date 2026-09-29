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
use dw_proxy_core::config::ProxyConfig;
use dw_proxy_core::render::ParserFamily;
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
/// `dw-spillover` estimates hosted occupancy from router-tracked decode blocks,
/// which production frontends do not track (`--no-router-track-active-blocks`).
/// Enabling it frontend-wide would change routing for every other model on the
/// frontend, so each spillover deployment advertises it per worker set instead:
/// the hosted SGLang workers get [`RouterAdvertisement::hosted_args`] and each proxy config
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
    pub fn hosted_args(&self) -> Vec<String> {
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
            // DYN_SHARED_CACHE_MULTIPLIER set on hosted workers can never split the set.
            "--shared-cache-multiplier".to_string(),
            "0.5".to_string(),
        ]
    }
}

/// The advertisement emitted for every deployment. One value drives both the
/// hosted SGLang flags and the proxy YAML, so the two can never drift.
pub const ROUTER_ADVERTISEMENT: RouterAdvertisement = RouterAdvertisement {
    mode: "kv",
    track_active_blocks: true,
    track_output_blocks: false,
};

/// Top-level shape of `deployments.yaml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentsFile {
    /// Keyed by the Dynamo model name, e.g. `zai-org/GLM-5.3@interactive`.
    pub deployments: BTreeMap<String, Deployment>,
}

/// One Dynamo deployment: hosted settings, the model card the proxies mirror, and proxy tiers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    pub hosted: HostedSettings,
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
pub struct HostedSettings {
    pub hosted_capacity_blocks: f64,
    pub occupancy_threshold: f64,
    pub failover_penalty_blocks: f64,
    pub pending_weight_blocks: f64,
    /// Engine-queue admission margin for every hosted worker process
    /// (`DYN_ADMISSION_QUEUE_MARGIN`, in engine-waiting requests). Defaults to
    /// [`DEFAULT_ADMISSION_QUEUE_MARGIN`]; see `docs/spillover/tuning.md` for the
    /// routing-sim derivation.
    #[serde(default = "default_admission_queue_margin")]
    pub admission_queue_margin: u64,
}

/// Margin used when a deployment does not set one. The routing-sim admission sweep shows the
/// gate's steering knee is single-digit requests for a normal hosted worker, so this sits well
/// above the policy's failover point and cannot fire before the policy decides to spill.
/// It is a floor for each hosted process, not a per-model frontend value: the fork has no
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
}

/// Reject a deployment file that could never produce consistent output.
pub fn validate_input(doc: &DeploymentsFile) -> anyhow::Result<()> {
    if doc.deployments.is_empty() {
        bail!("deployments must not be empty");
    }
    for (name, deployment) in &doc.deployments {
        if name.trim().is_empty() {
            bail!("deployment names must not be empty");
        }
        if !deployment
            .model
            .served_model_names
            .iter()
            .any(|served| served == name)
        {
            bail!(
                "deployment {name:?}: served_model_names must include the Dynamo model name {name:?}"
            );
        }
        if deployment.tiers.is_empty() {
            bail!("deployment {name:?}: at least one proxy tier is required");
        }
        let mut seen = BTreeSet::new();
        for (index, tier) in deployment.tiers.iter().enumerate() {
            if tier.name.trim().is_empty() {
                bail!("deployment {name:?}: tier[{index}].name must not be empty");
            }
            if !seen.insert(tier.name.as_str()) {
                bail!("deployment {name:?}: duplicate tier name {:?}", tier.name);
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

/// Paths of the generated files, relative to the output directory, mapped to their contents.
pub fn build(input: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    let raw = fs::read_to_string(input)
        .with_context(|| format!("reading deployments {}", input.display()))?;
    let doc: DeploymentsFile = serde_yaml::from_str(&raw)
        .with_context(|| format!("parsing deployments {}", input.display()))?;
    validate_input(&doc)?;

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
    files.insert(
        "router-policy.yaml".to_string(),
        serde_yaml::to_string(&policy).context("serializing router-policy.yaml")?,
    );
    files.insert("frontend.env".to_string(), frontend_env(&doc));

    for (name, deployment) in &doc.deployments {
        let directory = sanitize(name);
        // The card `router_config` is advertised per worker set, so the hosted
        // workers get the SGLang flags and the proxies carry the same values in
        // their YAML. See [`ROUTER_ADVERTISEMENT`].
        files.insert(
            format!("router/{directory}/hosted.args"),
            hosted_router_args_file(name),
        );
        // The margin is read per worker process, so emit it as environment files: hosted
        // workers get DYN_ADMISSION_QUEUE_MARGIN, proxies get an explicit opt-out so a value
        // cannot leak in from a shared launch environment.
        let (hosted_env, proxy_env) = admission_env(name, deployment);
        files.insert(format!("admission/{directory}/hosted.env"), hosted_env);
        files.insert(format!("admission/{directory}/proxy.env"), proxy_env);
        for (index, tier) in deployment.tiers.iter().enumerate() {
            for replica in 0..tier.replicas {
                let config = proxy_config(deployment, tier, index, replica);
                let path = format!("{directory}/{}-{replica}.yaml", sanitize(&tier.name));
                files.insert(
                    path,
                    serde_yaml::to_string(&config).context("serializing a proxy config")?,
                );
            }
        }
    }
    Ok(files)
}

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
# card itself: the hosted SGLang workers via router/<model>/hosted.args and the\n\
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

/// One shell file per deployment carrying the hosted SGLang `--router-*` flags.
///
/// The single non-comment, non-empty line is the flags to append to the worker
/// command (for example `xargs` or a shell array). The same values are written
/// into every proxy config's `router_config`, so the cards hash equal.
fn hosted_router_args_file(model_name: &str) -> String {
    let args = ROUTER_ADVERTISEMENT.hosted_args().join(" ");
    format!(
        "# Hosted SGLang worker router flags for {model_name}.\n\
# Append them to every hosted worker's command line so its model card carries\n\
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
/// override (no `DYN_ADMISSION_QUEUE_MARGIN_OVERRIDES`). Each hosted worker therefore gets its
/// own value. Proxy workers never report engine waiting, so the margin cannot apply to them and
/// their file clears the variable.
fn admission_env(model_name: &str, deployment: &Deployment) -> (String, String) {
    let margin = deployment.hosted.admission_queue_margin;
    let hosted = format!(
        "# Hosted workers for {model_name}.\n\
# lib/runtime/src/admission_gate.rs reads this from each worker process; the\n\
# frontend does not read it. Set it on every hosted worker.\n\
DYN_ADMISSION_QUEUE_MARGIN={margin}\n"
    );
    let proxy = format!(
        "# Proxy workers for {model_name}. They never report num_waiting_reqs, so the\n\
# engine-queue margin is unenforceable on them. Clear it explicitly so a value\n\
# cannot leak in from a shared launch environment.\n\
unset DYN_ADMISSION_QUEUE_MARGIN\n"
    );
    (hosted, proxy)
}

/// Write every generated file, creating directories as needed.
pub fn write_files(out: &Path, files: &BTreeMap<String, String>) -> anyhow::Result<()> {
    for (relative, contents) in files {
        let path = out.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// Load the generated output from `dir` with the consuming types, rejecting unusable configs.
pub fn validate_dir(dir: &Path) -> anyhow::Result<()> {
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

    let mut checked = 0usize;
    for path in yaml_files(dir)? {
        if path.file_name().and_then(|n| n.to_str()) == Some("router-policy.yaml") {
            continue;
        }
        ProxyConfig::load(&path)?;
        checked += 1;
    }
    if checked == 0 {
        bail!("{} contains no proxy configs", dir.display());
    }
    Ok(())
}

/// `generate`: build, write and validate the output for `input` under `out`.
pub fn generate(input: &Path, out: &Path) -> anyhow::Result<()> {
    let files = build(input)?;
    write_files(out, &files)?;
    validate_dir(out)?;
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
        occupancy_threshold: deployment.hosted.occupancy_threshold,
        hosted_capacity_blocks: deployment.hosted.hosted_capacity_blocks,
        failover_penalty_blocks: deployment.hosted.failover_penalty_blocks,
        pending_weight_blocks: deployment.hosted.pending_weight_blocks,
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

/// Collect every `*.yaml` file under `dir`, recursively.
fn yaml_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in
            fs::read_dir(&current).with_context(|| format!("reading {}", current.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
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
    hosted_capacity_blocks: f64,
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
}
