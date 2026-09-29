//! The proxy worker's config file (YAML).

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

use crate::render::ParserFamily;
use crate::upstream::ProviderConfig;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProxyConfig {
    /// Same model path or HF repo id as the SGLang workers, so the model card matches exactly.
    pub model_path: String,
    /// Served model name(s), identical to the SGLang workers', e.g. `zai-org/GLM-5.3@interactive`.
    pub served_model_names: Vec<String>,
    /// Dynamo namespace, component and endpoint of the SGLang workers this proxy joins.
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    /// Must equal the SGLang workers' KV block size.
    pub kv_block_size: u32,
    /// Must equal the SGLang workers' context length.
    pub context_length: u32,
    /// Reserved DP rank that marks this proxy's tier to the spillover policy.
    pub dp_rank: u32,
    pub tier: String,
    pub parser_family: ParserFamily,
    pub provider: ProviderConfig,
    #[serde(default = "default_vcache_ttl_secs")]
    pub vcache_ttl_secs: u64,
    #[serde(default = "default_vcache_max_blocks")]
    pub vcache_max_blocks: usize,
}

fn default_vcache_ttl_secs() -> u64 {
    300
}

fn default_vcache_max_blocks() -> usize {
    1_000_000
}

impl ProxyConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let config: Self = serde_yaml::from_str(&raw)
            .with_context(|| format!("parsing config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject configs that could never join the worker set (empty names, block size 0,
    /// `dp_rank` 0, unknown fields are already rejected by serde).
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.model_path.trim().is_empty() {
            anyhow::bail!("model_path must not be empty");
        }
        if self.served_model_names.is_empty() {
            anyhow::bail!("served_model_names must not be empty");
        }
        if self.served_model_names.iter().any(|n| n.trim().is_empty()) {
            anyhow::bail!("served_model_names must not contain empty names");
        }
        if self.namespace.trim().is_empty() {
            anyhow::bail!("namespace must not be empty");
        }
        if self.component.trim().is_empty() {
            anyhow::bail!("component must not be empty");
        }
        if self.endpoint.trim().is_empty() {
            anyhow::bail!("endpoint must not be empty");
        }
        if self.kv_block_size == 0 {
            anyhow::bail!("kv_block_size must be greater than 0");
        }
        if self.context_length == 0 {
            anyhow::bail!("context_length must be greater than 0");
        }
        if self.dp_rank == 0 {
            anyhow::bail!("dp_rank must be greater than 0");
        }
        if self.tier.trim().is_empty() {
            anyhow::bail!("tier must not be empty");
        }
        if self.provider.name.trim().is_empty() {
            anyhow::bail!("provider.name must not be empty");
        }
        if self.provider.base_url.trim().is_empty() {
            anyhow::bail!("provider.base_url must not be empty");
        }
        if self.provider.api_key_env.trim().is_empty() {
            anyhow::bail!("provider.api_key_env must not be empty");
        }
        if self.provider.model.trim().is_empty() {
            anyhow::bail!("provider.model must not be empty");
        }
        Ok(())
    }
}
