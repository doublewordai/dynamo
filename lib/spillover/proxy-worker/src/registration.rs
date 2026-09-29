//! Turning [`ProxyConfig`] into the Dynamo registration metadata.
//!
//! The frontend must treat this worker exactly like the SGLang workers it
//! joins, so the model card mirrors what `components/src/dynamo/sglang/register.py`
//! registers: the same model path and served names, `ModelType::Chat` (via the
//! `chat,completions` endpoint types), the SGLang KV block size and context
//! length, and the reserved DP rank. Only the capacity numbers are deliberately
//! huge: the proxy owns no GPU KV cache, so it must never look overloaded to the
//! router's KV-aware scorer.
//!
//! `Worker` consumes this in `lib/backend-common/src/worker.rs`
//! (`resolve_served_name`, `resolve_model_type`, `build_local_model`).
//! `build_local_model` treats a non-empty `model_name` that exists on disk as a
//! local model dir and loads only its tokenizer/chat template via
//! `lib/llm/src/local_model.rs`; it never reads weight files. That is why the
//! config's `model_path` is expected to be the same local path the SGLang
//! workers mount: a bare HF repo id would make `LocalModel::fetch` download
//! weights.

use dw_proxy_core::config::{ProxyConfig, ProxyRouterConfig, ProxyRouterMode};
use dynamo_backend_common::{EngineConfig, LlmRegistration, ModelInput, WorkerConfig};
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_runtime::pipeline::RouterMode;

/// Advertised KV capacity. Large enough that KV-aware routing never avoids the
/// proxy for lack of blocks; the spillover policy decides when it is used.
pub const TOTAL_KV_BLOCKS: u64 = 1_000_000;

/// Advertised batched-token budget, matching the same intent as
/// [`TOTAL_KV_BLOCKS`].
pub const MAX_NUM_BATCHED_TOKENS: u64 = 1_000_000;

/// Endpoint types registered with the model card. `chat` is what makes the
/// router see a chat worker; `completions` mirrors SGLang's OpenAI surface.
const ENDPOINT_TYPES: &str = "chat,completions";

/// The `Worker` lifecycle config (`dynamo_backend_common::WorkerConfig`).
///
/// Runtime transports are left at their defaults so `RuntimeConfig::default()`
/// applies no overrides and the Dynamo runtime reads the discovery/request/event
/// planes from the environment, exactly as a hand-written SGLang worker does.
pub fn worker_config(config: &ProxyConfig) -> WorkerConfig {
    WorkerConfig {
        namespace: config.namespace.clone(),
        component: config.component.clone(),
        endpoint: config.endpoint.clone(),
        // Empty means name-only registration; we point at the SGLang model so
        // the card carries the same tokenizer/chat template, but no weights.
        model_name: config.model_path.clone(),
        served_model_name: served_name(config),
        model_input: ModelInput::Tokens,
        endpoint_types: ENDPOINT_TYPES.to_string(),
        // The proxy hosts no in-process KV indexer: it publishes virtual-cache
        // events to the router but keeps no local index.
        enable_local_indexer: false,
        // Mirror the SGLang workers' card `router_config` so the checksums match
        // and the proxy joins their worker set. `None` leaves the card without
        // one, inheriting the frontend-wide configuration as before.
        router_config: config.router_config.as_ref().map(card_router_config),
        ..WorkerConfig::default()
    }
}

/// Registration metadata returned from `LLMEngine::start`.
pub fn engine_config(config: &ProxyConfig) -> EngineConfig {
    let aliases = config
        .served_model_names
        .iter()
        .skip(1)
        .cloned()
        .collect::<Vec<_>>();
    EngineConfig {
        model: config.model_path.clone(),
        served_model_name: served_name(config),
        model_aliases: aliases,
        runtime_data: Default::default(),
        llm: Some(LlmRegistration {
            context_length: Some(config.context_length),
            kv_cache_block_size: Some(config.kv_block_size),
            total_kv_blocks: Some(TOTAL_KV_BLOCKS),
            max_num_batched_tokens: Some(MAX_NUM_BATCHED_TOKENS),
            data_parallel_size: Some(1),
            data_parallel_start_rank: Some(config.dp_rank),
            ..LlmRegistration::default()
        }),
    }
}

/// Canary payload registered with the runtime's `HealthCheckManager`.
///
/// A one-token prompt is enough: the engine answers `is_probe` requests locally
/// without touching the provider. `Worker` stamps the `_HEALTH_CHECK` marker on
/// top, which deserializes into `PreprocessedRequest::is_probe`.
pub fn health_check_payload(config: &ProxyConfig) -> serde_json::Value {
    serde_json::json!({
        "model": served_name(config).unwrap_or_else(|| config.model_path.clone()),
        "token_ids": [1],
    })
}

fn served_name(config: &ProxyConfig) -> Option<String> {
    config.served_model_names.first().cloned()
}

/// Build the `RouterConfig` the SGLang workers advertise from the same
/// `--router-*` flags, so the two model cards hash equal.
///
/// Only the fields that differ from `KvRouterConfig::default()` are named; every
/// other field keeps the Rust default, which matches the SGLang CLI default
/// (`components/src/dynamo/common/configuration/groups/kv_router_args.py`) for
/// every field the CLI forwards. `shared_cache_multiplier` is the exception: the
/// CLI defaults it to 0.5 while the Rust default is 0.0, and the card checksum
/// serializes it, so it is named here too. `build_router_config` turns those
/// flags and defaults into the card's `RouterConfig`.
fn card_router_config(router: &ProxyRouterConfig) -> RouterConfig {
    let mode = match router.mode {
        ProxyRouterMode::RoundRobin => RouterMode::RoundRobin,
        ProxyRouterMode::Random => RouterMode::Random,
        ProxyRouterMode::PowerOfTwoChoices => RouterMode::PowerOfTwoChoices,
        ProxyRouterMode::Kv => RouterMode::KV,
        ProxyRouterMode::Direct => RouterMode::Direct,
        ProxyRouterMode::LeastLoaded => RouterMode::LeastLoaded,
        ProxyRouterMode::DeviceAwareWeighted => RouterMode::DeviceAwareWeighted,
    };
    RouterConfig {
        router_mode: mode,
        kv_router_config: dynamo_kv_router::KvRouterConfig {
            router_track_active_blocks: router.track_active_blocks,
            router_track_output_blocks: router.track_output_blocks,
            shared_cache_multiplier: SGLANG_CLI_SHARED_CACHE_MULTIPLIER,
            ..Default::default()
        },
        ..RouterConfig::default()
    }
}

/// `--shared-cache-multiplier`'s CLI default. The SGLang workers advertise it
/// (`kv_router_kwargs` forwards every KV-router field), and `mdcsum()` serializes
/// `KvRouterConfig`, so a worker set only forms if the proxy advertises the same
/// value. The Rust `KvRouterConfig::default()` is 0.0.
const SGLANG_CLI_SHARED_CACHE_MULTIPLIER: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    use dw_proxy_core::render::ParserFamily;
    use dw_proxy_core::upstream::ProviderConfig;

    fn sample() -> ProxyConfig {
        ProxyConfig {
            model_path: "/models/glm-5.3".to_string(),
            served_model_names: vec![
                "zai-org/GLM-5.3@interactive".to_string(),
                "zai-org/GLM-5.3".to_string(),
            ],
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            kv_block_size: 64,
            context_length: 202_752,
            dp_rank: 7,
            tier: "spillover".to_string(),
            parser_family: ParserFamily::Glm47,
            provider: ProviderConfig {
                name: "openrouter".to_string(),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key_env: "OPENROUTER_API_KEY".to_string(),
                model: "z-ai/glm-5.3".to_string(),
                provider_preferences: None,
                body_overrides: None,
                extra_headers: Default::default(),
                connect_timeout_ms: 10_000,
            },
            vcache_ttl_secs: 300,
            vcache_max_blocks: 1_000_000,
            router_config: Some(ProxyRouterConfig {
                mode: ProxyRouterMode::Kv,
                track_active_blocks: true,
                track_output_blocks: false,
            }),
        }
    }

    #[test]
    fn worker_config_mirrors_sglang_surface() {
        let cfg = sample();
        let wc = worker_config(&cfg);
        assert_eq!(wc.namespace, "dynamo");
        assert_eq!(wc.component, "backend");
        assert_eq!(wc.endpoint, "generate");
        assert_eq!(wc.model_name, "/models/glm-5.3");
        assert_eq!(
            wc.served_model_name.as_deref(),
            Some("zai-org/GLM-5.3@interactive")
        );
        assert_eq!(wc.model_input, ModelInput::Tokens);
        assert_eq!(wc.endpoint_types, "chat,completions");
        assert!(!wc.enable_local_indexer);
        // No explicit transport overrides: the runtime reads them from env.
        assert!(!wc.runtime.has_overrides());
        let router = wc
            .router_config
            .expect("proxy card carries a router config");
        assert_eq!(router.router_mode, RouterMode::KV);
        assert!(router.kv_router_config.router_track_active_blocks);
        assert!(!router.kv_router_config.router_track_output_blocks);
    }

    #[test]
    fn proxy_card_matches_sglang_card_checksum() {
        // A proxy and an SGLang worker started from the same deployment must
        // advertise the same card `router_config`, because the checksum covers
        // it and a mismatch splits the worker set. The SGLang equivalent is what
        // `build_router_config` produces for
        // `--router-mode kv --router-track-active-blocks`
        // (`components/src/dynamo/common/configuration/groups/router_args.py`):
        // `RouterConfig(mode=KV, KvRouterConfig(**kv_router_kwargs()))`, whose
        // `router_track_active_blocks` default is already true. The only other
        // field the SGLang CLI leaves off the Rust default is
        // `shared_cache_multiplier`, so the hosted side names it too.
        let proxy_router = card_router_config(&ProxyRouterConfig {
            mode: ProxyRouterMode::Kv,
            track_active_blocks: true,
            track_output_blocks: false,
        });
        let hosted_router = RouterConfig {
            router_mode: RouterMode::KV,
            kv_router_config: dynamo_kv_router::KvRouterConfig {
                shared_cache_multiplier: SGLANG_CLI_SHARED_CACHE_MULTIPLIER,
                ..Default::default()
            },
            ..RouterConfig::default()
        };
        assert_eq!(
            serde_json::to_value(&proxy_router).unwrap(),
            serde_json::to_value(&hosted_router).unwrap(),
            "proxy and hosted router configs must serialize identically"
        );

        let mut proxy_card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("m");
        proxy_card.router_config = Some(proxy_router);
        let mut hosted_card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("m");
        hosted_card.router_config = Some(hosted_router);
        assert_eq!(proxy_card.mdcsum(), hosted_card.mdcsum());
    }

    #[test]
    fn engine_config_carries_kv_and_dp_metadata() {
        let cfg = sample();
        let ec = engine_config(&cfg);
        assert_eq!(ec.model, "/models/glm-5.3");
        assert_eq!(
            ec.served_model_name.as_deref(),
            Some("zai-org/GLM-5.3@interactive")
        );
        assert_eq!(ec.model_aliases, vec!["zai-org/GLM-5.3".to_string()]);

        let llm = ec.llm.expect("token engines carry LlmRegistration");
        assert_eq!(llm.context_length, Some(202_752));
        assert_eq!(llm.kv_cache_block_size, Some(64));
        assert_eq!(llm.total_kv_blocks, Some(TOTAL_KV_BLOCKS));
        assert_eq!(llm.max_num_batched_tokens, Some(MAX_NUM_BATCHED_TOKENS));
        assert_eq!(llm.data_parallel_size, Some(1));
        assert_eq!(llm.data_parallel_start_rank, Some(7));
        assert!(!llm.enable_eagle);
    }

    #[test]
    fn health_payload_is_an_object_with_tokens() {
        let payload = health_check_payload(&sample());
        assert!(payload.is_object());
        assert_eq!(
            payload["model"],
            serde_json::json!("zai-org/GLM-5.3@interactive")
        );
        assert_eq!(payload["token_ids"], serde_json::json!([1]));
    }
}
