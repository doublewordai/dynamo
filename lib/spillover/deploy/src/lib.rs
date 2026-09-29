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

    for (name, deployment) in &doc.deployments {
        let directory = sanitize(name);
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
    vcache_ttl_secs: u64,
    vcache_max_blocks: usize,
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
