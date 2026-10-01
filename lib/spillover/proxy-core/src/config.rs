// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The proxy worker's config file (YAML).

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use crate::render::ParserFamily;
use crate::upstream::ProviderConfig;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProxyConfig {
    /// Same model path or HF repo id as the primary workers, so the model card matches exactly.
    pub model_path: String,
    /// Served model name(s), identical to the primary workers', e.g. `zai-org/GLM-5.3`.
    pub served_model_names: Vec<String>,
    /// Dynamo namespace, component and endpoint of the primary workers this proxy joins.
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    /// Must equal the primary workers' KV block size.
    pub kv_block_size: u32,
    /// Context length advertised on the model card.
    ///
    /// `Some(n)` (n > 0) advertises exactly `n`, matching a primary started
    /// with `--context-length`/`--max-model-len`/`--max-seq-len`. `None`
    /// advertises nothing so the card falls back to the model's architectural
    /// maximum (`config.json`'s `max_position_embeddings`), exactly like an
    /// SGLang worker without `--context-length` or a TRT-LLM worker without
    /// `--max-seq-len`. A hub id must use the same setting as the primary side
    /// or the checksum splits the worker set.
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Optional path to a custom Jinja chat template, identical to the primary
    /// worker's `--custom-jinja-template`.
    ///
    /// The template is part of the model card's chat-template checksum, so a
    /// primary started with a custom template must be mirrored with the same
    /// file or the two land in different worker sets. `None` uses the template
    /// shipped with the model.
    #[serde(default)]
    pub custom_jinja_template: Option<PathBuf>,
    /// Whether the primary workers emit bigram-keyed KV events for EAGLE/MTP
    /// speculative decoding.
    ///
    /// The router hashes prompts differently for EAGLE, so this **must equal
    /// the primary's** setting or a proxy and its primary hash the same tokens
    /// to different blocks and cache affinity breaks. Sets both
    /// `LlmRegistration.enable_eagle` on the card and `HashOptions.is_eagle` on
    /// the proxy's virtual cache.
    #[serde(default)]
    pub enable_eagle: bool,
    /// Register the card without a `source_path`. Only for a primary registered through the
    /// `make_engine` entrypoint with a local model path (the mocker), whose card records none;
    /// every other engine records the model string, as the proxy does by default. The source
    /// path feeds the card checksum, so a mismatch splits the worker set.
    #[serde(default)]
    pub omit_source_path: bool,
    /// Reserved DP rank that marks this proxy's tier to the spillover policy.
    pub dp_rank: u32,
    pub tier: String,
    pub parser_family: ParserFamily,
    /// Endpoint types this proxy advertises on its model card, as a comma-separated
    /// list, e.g. `chat,completions`.
    ///
    /// `endpoint_types` feeds the card's `model_type`, which is part of
    /// `worker_set_key` (`lib/llm/src/discovery/watcher.rs`). The default mirrors the
    /// primary workers' default (`WorkerConfig::default()` is
    /// `chat,completions`), so the two register as one worker set and spillover can
    /// engage. A proxy that advertises only `chat` would land in a *different* set
    /// from a `chat,completions` primary, and neither side would ever route to the
    /// other. Validated to a non-empty subset of {chat, completions}.
    #[serde(default = "default_endpoint_types")]
    pub endpoint_types: String,
    pub provider: ProviderConfig,
    /// Optional router advertisement written to the model card's `router_config`.
    ///
    /// The primary workers this proxy joins set the same fields through their
    /// `--router-*` flags, so the card checksums match and the two register as
    /// one worker set. `None` keeps today's behaviour: the card advertises no
    /// router config and the worker set inherits the frontend-wide one.
    #[serde(default)]
    pub router_config: Option<ProxyRouterConfig>,
    /// Engine capacity the proxy advertises to the router on its model card.
    ///
    /// A proxy normally owns no KV cache, so it advertises no capacity (the
    /// field is omitted and the router reads `None`). When the proxy fronts an
    /// actual primary engine (for example a simulated primary), set this so the
    /// spillover policy's occupancy estimate uses the real engine limits instead
    /// of treating the primary as having unknown capacity.
    #[serde(default)]
    pub advertised_capacity: Option<AdvertisedCapacity>,
    #[serde(default = "default_vcache_ttl_secs")]
    pub vcache_ttl_secs: u64,
    #[serde(default = "default_vcache_max_blocks")]
    pub vcache_max_blocks: usize,
}

/// The subset of the worker set's router advertisement the proxy mirrors.
///
/// The primary side sets these through `--router-mode`,
/// `--router-track-active-blocks` and `--router-track-output-blocks`
/// (`components/src/dynamo/common/configuration/groups/kv_router_args.py`), which
/// `build_router_config` turns into the card's `RouterConfig`. The shared CLI
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
    /// `--router-track-output-blocks`. Defaults off, matching the shared CLI default.
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

/// Engine capacity advertised on the model card's runtime config.
///
/// Mirrors the capacity fields of `ModelRuntimeConfig`: `kv_blocks` becomes
/// `total_kv_blocks` and `max_requests` becomes `max_num_seqs`. Either may be
/// omitted; whichever is set is validated to be greater than 0.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvertisedCapacity {
    /// Total KV-cache blocks the engine reports. Advertised as
    /// `ModelRuntimeConfig::total_kv_blocks`.
    #[serde(default)]
    pub kv_blocks: Option<u64>,
    /// Maximum concurrently scheduled sequences the engine reports. Advertised
    /// as `ModelRuntimeConfig::max_num_seqs`.
    #[serde(default)]
    pub max_requests: Option<u64>,
}

fn default_endpoint_types() -> String {
    "chat,completions".to_string()
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
        if self.context_length == Some(0) {
            anyhow::bail!("context_length must be greater than 0 when set");
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
        validate_endpoint_types(&self.endpoint_types)?;
        if self.provider.name.trim().is_empty() {
            anyhow::bail!("provider.name must not be empty");
        }
        if self.provider.base_url.trim().is_empty() {
            anyhow::bail!("provider.base_url must not be empty");
        }
        validate_base_url(&self.provider.base_url, self.provider.allow_insecure_http)?;
        if self.provider.api_key_env.trim().is_empty() {
            anyhow::bail!("provider.api_key_env must not be empty");
        }
        if self.provider.model.trim().is_empty() {
            anyhow::bail!("provider.model must not be empty");
        }
        if self.provider.connect_timeout_ms == 0 {
            anyhow::bail!("provider.connect_timeout_ms must be greater than 0");
        }
        if self.provider.read_timeout_ms == 0 {
            anyhow::bail!("provider.read_timeout_ms must be greater than 0");
        }
        if let Some(breaker) = &self.provider.circuit_breaker {
            if breaker.failure_threshold == 0 {
                anyhow::bail!("provider.circuit_breaker.failure_threshold must be greater than 0");
            }
            if breaker.cooldown_ms == 0 {
                anyhow::bail!("provider.circuit_breaker.cooldown_ms must be greater than 0");
            }
            if breaker.max_cooldown_ms < breaker.cooldown_ms {
                anyhow::bail!(
                    "provider.circuit_breaker.max_cooldown_ms must be greater than or equal to \
                     cooldown_ms"
                );
            }
        }
        if let Some(capacity) = &self.advertised_capacity {
            if capacity.kv_blocks == Some(0) {
                anyhow::bail!("advertised_capacity.kv_blocks must be greater than 0");
            }
            if capacity.max_requests == Some(0) {
                anyhow::bail!("advertised_capacity.max_requests must be greater than 0");
            }
        }
        if self.vcache_ttl_secs == 0 {
            anyhow::bail!("vcache_ttl_secs must be greater than 0");
        }
        if self.vcache_max_blocks == 0 {
            anyhow::bail!("vcache_max_blocks must be greater than 0");
        }
        if self.provider.cache_key != crate::cache_key::CacheKeyField::None
            && self
                .provider
                .cache_key_secret_env
                .as_deref()
                .is_none_or(|env| env.trim().is_empty())
        {
            anyhow::bail!("provider.cache_key requires provider.cache_key_secret_env");
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

/// Reject an `endpoint_types` value the backend would not accept (or that would
/// describe a pipeline the proxy cannot serve).
///
/// The proxy serves only chat completions, but it must advertise the *same* set as
/// its primary workers or the card's `model_type` splits the worker set.
/// A `completions`-only or `chat`-only advertisement is accepted because a primary
/// set may legitimately use either; an `embedding`/`images`/... endpoint is rejected
/// because the proxy's token engine cannot serve it at all.
pub fn validate_endpoint_types(raw: &str) -> anyhow::Result<()> {
    let mut any = false;
    for part in raw.split(',') {
        let t = part.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        match t.as_str() {
            "chat" | "completions" => any = true,
            other => anyhow::bail!(
                "endpoint_types must be a non-empty subset of {{chat, completions}}, got {other:?}"
            ),
        }
    }
    if !any {
        anyhow::bail!("endpoint_types must not be empty");
    }
    Ok(())
}

/// Parse and sanity-check the provider endpoint. The URL is assembled per request with
/// `format!`, so a malformed value would otherwise only surface as a transport error under
/// load. Plain HTTP is allowed only to a loopback host (local dev / mock provider) unless
/// `allow_insecure_http` is set for an in-cluster simulator; anything else would leak the API key
/// and the conversation in cleartext.
fn validate_base_url(base_url: &str, allow_insecure_http: bool) -> anyhow::Result<()> {
    let url = reqwest::Url::parse(base_url)
        .map_err(|error| anyhow::anyhow!("provider.base_url is not a valid URL: {error}"))?;
    if url.host_str().is_none() {
        anyhow::bail!("provider.base_url must include a host");
    }
    if url.query().is_some() || url.fragment().is_some() {
        anyhow::bail!("provider.base_url must not contain a query or fragment");
    }
    match url.scheme() {
        "https" => Ok(()),
        "http" if allow_insecure_http || is_loopback(&url) => Ok(()),
        scheme => anyhow::bail!(
            "provider.base_url must use https (http is allowed only for a loopback host, or with \
             provider.allow_insecure_http), got {scheme}"
        ),
    }
}

fn is_loopback(url: &reqwest::Url) -> bool {
    match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_types_default_to_chat_and_completions() {
        // The default must mirror `WorkerConfig::default()` (chat,completions):
        // `model_type` is part of `worker_set_key`, so a proxy that advertised only
        // `chat` would never share a worker set with a default primary.
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
        assert_eq!(config.endpoint_types, "chat,completions");
        config.validate().unwrap();
    }

    #[test]
    fn endpoint_types_override_and_validate() {
        let base = r#"
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
        // `chat` alone is a legitimate primary set the proxy must be able to mirror.
        let chat: ProxyConfig =
            serde_yaml::from_str(&format!("{base}endpoint_types: chat\n")).unwrap();
        assert_eq!(chat.endpoint_types, "chat");
        chat.validate().unwrap();

        // An endpoint the proxy's token engine cannot serve is rejected.
        for bad in ["embedding", "chat,images", "  ,  "] {
            let config: ProxyConfig =
                serde_yaml::from_str(&format!("{base}endpoint_types: \"{bad}\"\n")).unwrap();
            let error = config.validate().unwrap_err().to_string();
            assert!(error.contains("endpoint_types"), "{bad}: {error}");
        }
    }

    #[test]
    fn context_length_is_optional_and_zero_when_set_is_rejected() {
        // `None` mirrors a primary with no explicit context length: the card
        // falls back to `config.json`'s `max_position_embeddings`.
        let omitted: ProxyConfig = serde_yaml::from_str(
            &yaml_with_provider_tail("").replace("context_length: 1024\n", ""),
        )
        .unwrap();
        assert_eq!(omitted.context_length, None);
        omitted.validate().unwrap();

        // A number that is present must be positive.
        let zero: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail("")).unwrap();
        let mut zero = zero;
        zero.context_length = Some(0);
        let error = zero.validate().unwrap_err().to_string();
        assert!(error.contains("context_length"), "{error}");
    }

    #[test]
    fn primary_mirroring_fields_default_off_and_parse() {
        let config: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail("")).unwrap();
        assert_eq!(config.custom_jinja_template, None);
        assert!(!config.enable_eagle);

        // The fields are serialized on the card/wire, so they must parse from YAML.
        let mirrored: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail(
            "custom_jinja_template: /templates/primary.jinja\nenable_eagle: true\n",
        ))
        .unwrap();
        assert_eq!(
            mirrored.custom_jinja_template.as_deref(),
            Some(Path::new("/templates/primary.jinja"))
        );
        assert!(mirrored.enable_eagle);
        config.validate().unwrap();
        mirrored.validate().unwrap();
    }

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

    #[test]
    fn advertised_capacity_defaults_to_none() {
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
        assert!(config.advertised_capacity.is_none());
        config.validate().unwrap();
    }

    #[test]
    fn advertised_capacity_parses_and_validates() {
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
advertised_capacity:
  kv_blocks: 4096
  max_requests: 32
"#;
        let config: ProxyConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            config.advertised_capacity,
            Some(AdvertisedCapacity {
                kv_blocks: Some(4096),
                max_requests: Some(32),
            })
        );
        config.validate().unwrap();
    }

    #[test]
    fn advertised_capacity_allows_either_field_alone() {
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
advertised_capacity:
  kv_blocks: 4096
"#;
        let config: ProxyConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.advertised_capacity.unwrap().max_requests, None);
        config.validate().unwrap();
    }

    #[test]
    fn advertised_capacity_rejects_zero() {
        for (field, value) in [
            ("kv_blocks", "kv_blocks: 0"),
            ("max_requests", "max_requests: 0"),
        ] {
            let yaml = format!(
                r#"
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
advertised_capacity:
  {value}
"#
            );
            let config: ProxyConfig = serde_yaml::from_str(&yaml).unwrap();
            let error = config.validate().unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn advertised_capacity_rejects_unknown_fields() {
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
advertised_capacity:
  kv_blocks: 4096
  surprise: 1
"#;
        let error = serde_yaml::from_str::<ProxyConfig>(yaml).unwrap_err();
        assert!(error.to_string().contains("surprise"), "{error}");
    }

    fn yaml_with_provider_tail(tail: &str) -> String {
        format!(
            r#"
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
{tail}"#
        )
    }

    #[test]
    fn circuit_breaker_defaults_to_config_defaults() {
        let config: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail("")).unwrap();
        assert!(config.provider.circuit_breaker.is_none());
        config.validate().unwrap();
        let defaults = crate::circuit_breaker::CircuitBreakerConfig::default();
        assert_eq!(defaults.failure_threshold, 5);
        assert_eq!(defaults.cooldown_ms, 30_000);
        assert_eq!(defaults.max_cooldown_ms, 300_000);
    }

    #[test]
    fn circuit_breaker_parses_and_validates() {
        let config: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail(
            "  circuit_breaker:\n    failure_threshold: 3\n    cooldown_ms: 500\n    max_cooldown_ms: 5000\n",
        ))
        .unwrap();
        let breaker = config.provider.circuit_breaker.unwrap();
        assert_eq!(breaker.failure_threshold, 3);
        assert_eq!(breaker.cooldown_ms, 500);
        assert_eq!(breaker.max_cooldown_ms, 5_000);
        config.validate().unwrap();
    }

    #[test]
    fn circuit_breaker_omitted_fields_use_defaults() {
        let config: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail(
            "  circuit_breaker:\n    cooldown_ms: 1000\n",
        ))
        .unwrap();
        let breaker = config.provider.circuit_breaker.unwrap();
        assert_eq!(breaker.failure_threshold, 5);
        assert_eq!(breaker.cooldown_ms, 1_000);
        assert_eq!(breaker.max_cooldown_ms, 300_000);
        config.validate().unwrap();
    }

    #[test]
    fn circuit_breaker_rejects_invalid_values() {
        for (tail, expected) in [
            (
                "  circuit_breaker:\n    failure_threshold: 0\n",
                "failure_threshold",
            ),
            ("  circuit_breaker:\n    cooldown_ms: 0\n", "cooldown_ms"),
            (
                "  circuit_breaker:\n    cooldown_ms: 1000\n    max_cooldown_ms: 500\n",
                "max_cooldown_ms",
            ),
        ] {
            let config: ProxyConfig = serde_yaml::from_str(&yaml_with_provider_tail(tail)).unwrap();
            let error = config.validate().unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn circuit_breaker_rejects_unknown_fields() {
        let error = serde_yaml::from_str::<ProxyConfig>(&yaml_with_provider_tail(
            "  circuit_breaker:\n    surprise: 1\n",
        ))
        .unwrap_err();
        assert!(error.to_string().contains("surprise"), "{error}");
    }
}
