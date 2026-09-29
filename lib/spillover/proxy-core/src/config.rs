// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    /// Optional router advertisement written to the model card's `router_config`.
    ///
    /// The SGLang workers this proxy joins set the same fields through their
    /// `--router-*` flags, so the card checksums match and the two register as
    /// one worker set. `None` keeps today's behaviour: the card advertises no
    /// router config and the worker set inherits the frontend-wide one.
    #[serde(default)]
    pub router_config: Option<ProxyRouterConfig>,
    #[serde(default = "default_vcache_ttl_secs")]
    pub vcache_ttl_secs: u64,
    #[serde(default = "default_vcache_max_blocks")]
    pub vcache_max_blocks: usize,
}

/// The subset of the worker set's router advertisement the proxy mirrors.
///
/// The SGLang side sets these through `--router-mode`,
/// `--router-track-active-blocks` and `--router-track-output-blocks`
/// (`components/src/dynamo/common/configuration/groups/router_args.py`), which
/// `build_router_config` turns into the card's `RouterConfig`. The SGLang CLI
/// defaults match the Rust defaults for every other forwarded field except
/// `shared_cache_multiplier` (CLI 0.5, Rust 0.0), which `registration.rs` sets
/// explicitly when it builds the card.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProxyRouterConfig {
    /// `--router-mode`. Defaults to `kv`, which the spillover policy requires.
    #[serde(default = "default_router_mode")]
    pub mode: ProxyRouterMode,
    /// `--router-track-active-blocks`. The spillover occupancy estimate reads
    /// router-tracked decode blocks, so this must be `true`.
    #[serde(default)]
    pub track_active_blocks: bool,
    /// `--router-track-output-blocks`. Defaults off, matching the SGLang default.
    #[serde(default)]
    pub track_output_blocks: bool,
}

/// Router mode advertised by a spillover worker set. Mirrors
/// `dynamo_runtime::pipeline::RouterMode`; proxy-core stays free of the runtime.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProxyRouterMode {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    #[default]
    Kv,
    Direct,
    LeastLoaded,
    DeviceAwareWeighted,
}

fn default_router_mode() -> ProxyRouterMode {
    ProxyRouterMode::Kv
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
        // The frontend computes the tier's range end as `start_rank + size`, so a
        // rank of `u32::MAX` would wrap to 0 and silently drop the proxy from routing.
        if self.dp_rank == u32::MAX {
            anyhow::bail!("dp_rank must be less than u32::MAX");
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
        if let Some(router) = &self.router_config {
            // `dw-spillover` reads router-tracked decode blocks; a proxy that
            // advertises tracking off would drag the whole worker set to that
            // value, because the card checksum includes `router_config`.
            if router.mode != ProxyRouterMode::Kv {
                anyhow::bail!("router_config.mode must be kv for a spillover worker set");
            }
            if !router.track_active_blocks {
                anyhow::bail!(
                    "router_config.track_active_blocks must be true for a spillover worker set"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_config_defaults_to_none() {
        let yaml = r#"
model_path: /models/m
served_model_names: [m]
namespace: dynamo
component: backend
endpoint: generate
kv_block_size: 64
context_length: 1024
dp_rank: 1000
tier: openrouter
parser_family: glm47
provider:
  name: p
  base_url: https://x/v1
  api_key_env: K
  model: m
"#;
        let config: ProxyConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.router_config.is_none());
        config.validate().unwrap();
    }

    #[test]
    fn router_config_parses_and_validates() {
        let yaml = r#"
model_path: /models/m
served_model_names: [m]
namespace: dynamo
component: backend
endpoint: generate
kv_block_size: 64
context_length: 1024
dp_rank: 1000
tier: openrouter
parser_family: glm47
provider:
  name: p
  base_url: https://x/v1
  api_key_env: K
  model: m
router_config:
  mode: kv
  track_active_blocks: true
  track_output_blocks: false
"#;
        let config: ProxyConfig = serde_yaml::from_str(yaml).unwrap();
        let router = config.router_config.clone().unwrap();
        assert_eq!(router.mode, ProxyRouterMode::Kv);
        assert!(router.track_active_blocks);
        config.validate().unwrap();
    }

    #[test]
    fn router_config_rejects_untracked_active_blocks() {
        let yaml = r#"
model_path: /models/m
served_model_names: [m]
namespace: dynamo
component: backend
endpoint: generate
kv_block_size: 64
context_length: 1024
dp_rank: 1000
tier: openrouter
parser_family: glm47
provider:
  name: p
  base_url: https://x/v1
  api_key_env: K
  model: m
router_config:
  mode: kv
  track_active_blocks: false
"#;
        let config: ProxyConfig = serde_yaml::from_str(yaml).unwrap();
        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("track_active_blocks"), "{error}");
    }
}
