// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turning [`ProxyConfig`] into the Dynamo registration metadata.
//!
//! The frontend must treat this worker exactly like the primary worker (any
//! engine: SGLang, vLLM, TRT-LLM, mocker) it joins, so the model card mirrors
//! the primary's registration: the same model path and served names,
//! `ModelType::Chat` (via the `chat,completions` endpoint types), the same KV
//! block size and context length, the same custom Jinja template and EAGLE/MTP
//! setting, and the reserved DP rank. The capacity numbers are only what the
//! proxy config advertises: a plain proxy owns no GPU KV cache, so it advertises
//! no `total_kv_blocks`/`max_num_seqs`, and the router reads `None` rather than a
//! placeholder it might mistake for real capacity. A proxy fronting a real
//! primary engine (a simulated primary, for example) sets `advertised_capacity`
//! and those exact limits reach the router.
//!
//! `Worker` consumes this in `lib/backend-common/src/worker.rs`
//! (`resolve_served_name`, `resolve_model_type`, `build_local_model`).
//! `build_local_model` treats a non-empty `model_name` that exists on disk as a
//! local model dir and loads only its tokenizer/chat template via
//! `lib/llm/src/local_model.rs`; it never reads weight files. A bare HF repo id
//! is resolved through `LocalModel::fetch`, and the proxy sets
//! `WorkerConfig::ignore_weights` so only the config/tokenizer files are
//! downloaded, exactly matching a primary that registers the same repo id as
//! its card's `source_path`.

use dw_proxy_core::config::{
    ProxyConfig, ProxyLoadThresholdConfig, ProxyRouterConfig, ProxyRouterMode,
    ProxySessionAffinityMode,
};
use dynamo_backend_common::{EngineConfig, LlmRegistration, ModelInput, WorkerConfig};
use dynamo_llm::discovery::LoadThresholdConfig;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::local_model::runtime_config::{
    CHAT_REQUEST_CAPABILITY, TOKEN_BUDGET_RUNTIME_KEY, TokenBudget,
};
use dynamo_llm::session_affinity::SessionAffinityMode;
use dynamo_runtime::pipeline::RouterMode;

/// Advertised batched-token budget. Large enough that the router never treats
/// the proxy as token-batch-limited; the spillover policy decides when it is used.
pub const MAX_NUM_BATCHED_TOKENS: u64 = 1_000_000;

/// Endpoint types the proxy registers on its model card.
///
/// The proxy calls a provider's *chat* API, so it can only serve chat
/// completions. It advertises `chat,completions` by default anyway, because
/// `endpoint_types` feeds the card's `model_type`, and `model_type` is part of
/// `worker_set_key` (`lib/llm/src/discovery/watcher.rs`). Primary workers use
/// the default `chat,completions`, so a `chat`-only proxy would
/// land in a different WorkerSet and spillover would never engage. Configurable
/// via [`ProxyConfig::endpoint_types`] for a primary set that uses a different
/// (still chat/completions) advertisement.
///
/// Advertising `completions` means the frontend may route a `/v1/completions`
/// request to the proxy. A completions request carries no chat request, so
/// `ProxyEngine::generate` refuses it with the migratable `NoChatRequest`
/// refusal (see `engine.rs`), and the router retries it on another worker. Each
/// proxy tier therefore costs one fast refused hop for a completions request;
/// Dynamo's frontend migration limit bounds how many hops a request can take.
pub fn endpoint_types(config: &ProxyConfig) -> String {
    config.endpoint_types.clone()
}

/// The `Worker` lifecycle config (`dynamo_backend_common::WorkerConfig`).
///
/// Runtime transports are left at their defaults so `RuntimeConfig::default()`
/// applies no overrides and the Dynamo runtime reads the discovery/request/event
/// planes from the environment, exactly as a hand-written worker does.
pub fn worker_config(config: &ProxyConfig) -> WorkerConfig {
    WorkerConfig {
        namespace: config.namespace.clone(),
        component: config.component.clone(),
        endpoint: config.endpoint.clone(),
        // Empty means name-only registration; we point at the primary model so
        // the card carries the same tokenizer/chat template, but no weights.
        model_name: config.model_path.clone(),
        served_model_name: served_name(config),
        model_input: ModelInput::Tokens,
        endpoint_types: endpoint_types(config),
        // Keep a local index ahead of the published events so the worker
        // advertises a recovery target. A live-only proxy cannot be re-synced
        // after a frontend restart: a `Stored` that extends a prefix the proxy
        // still holds carries a `parent_hash` the fresh frontend never saw, and
        // the router drops the whole chain. With the local indexer the fresh
        // frontend pulls the full held tree instead.
        enable_local_indexer: true,
        // Mirror the primary workers' card `router_config` so the checksums match
        // and the proxy joins their worker set. `None` leaves the card without
        // one, inheriting the frontend-wide configuration as before.
        router_config: config.router_config.as_ref().map(card_router_config),
        // The frontend builds a model's parsing from the card of the first worker it sees, so a
        // proxy must carry the same parsers as the primary workers: without them a proxy that
        // registers first (or a set of proxies alone) would hand clients raw model markup.
        tool_call_parser: Some(config.parser_family.tool_call_parser().to_string()),
        reasoning_parser: Some(config.parser_family.reasoning_parser().to_string()),
        // Mirror the primary's `--custom-jinja-template`, if any. The template is
        // hashed into the card, so a mismatch splits the worker set.
        custom_jinja_template: config.custom_jinja_template.clone(),
        // A proxy needs only the tokenizer/config files, never the weights. A
        // primary registering a bare HF repo id causes `LocalModel::fetch` to
        // download the full weights; the proxy must not.
        ignore_weights: true,
        omit_source_path: config.omit_source_path,
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
        // Ask the frontend for the client's chat request: the provider needs messages and
        // tools, not token ids. A runtime flag, so it does not split the worker set.
        runtime_data: [(CHAT_REQUEST_CAPABILITY.to_string(), serde_json::json!(true))]
            .into_iter()
            .chain(token_budget(config))
            .collect(),
        llm: Some(LlmRegistration {
            // `Some(n)` mirrors a primary's explicit context length; `None`
            // advertises nothing so the card falls back to the model's
            // architectural maximum, exactly like a primary with no context
            // flag. `enable_eagle` must equal the primary's or the router
            // hashes the same tokens to different blocks.
            context_length: config.context_length,
            enable_eagle: config.enable_eagle,
            kv_cache_block_size: Some(config.kv_block_size),
            // Publish the configured capacity, or `None` when the proxy owns no
            // engine. `total_kv_blocks` is not required to be `Some`: the router
            // and planner treat `None` as "not advertised", the approximate-LRU
            // indexer (off by default for a spillover worker set) falls back to
            // TTL, and the native `decode_load_exceeds` busy gate simply does not
            // fire. A placeholder would instead read as real capacity to the
            // spillover policy's per-worker occupancy estimate and to the planner.
            total_kv_blocks: config.advertised_capacity.and_then(|c| c.kv_blocks),
            // `max_num_seqs` feeds the admission gate's automatic concurrency
            // limit; leaving it `None` keeps that gate off for a cacheless proxy,
            // matching `admission/<model>/proxy.env`'s explicit margin clear.
            max_num_seqs: config.advertised_capacity.and_then(|c| c.max_requests),
            max_num_batched_tokens: Some(MAX_NUM_BATCHED_TOKENS),
            data_parallel_size: Some(1),
            data_parallel_start_rank: Some(config.dp_rank),
            ..LlmRegistration::default()
        }),
    }
}

/// The token budget a primary publishes for its context length, so the frontend refuses a
/// request that does not fit before routing it, whichever worker's card built its
/// preprocessor. SGLang workers publish exactly this (`components/src/dynamo/sglang/register.py`,
/// `_get_token_budget`, without `--allow-auto-truncate`), and the frontend answers the refusal
/// with HTTP 400 on the streaming and non-streaming paths alike. A refusal from the worker's own
/// stream instead reaches a streaming client behind HTTP 200 unless the frontend holds the
/// status for the first event. `None` without a configured context length.
fn token_budget(config: &ProxyConfig) -> Option<(String, serde_json::Value)> {
    let budget = TokenBudget {
        combined_limit: config.context_length?,
        reject_prompt_overflow: true,
        reject_total_overflow: true,
    };
    Some((
        TOKEN_BUDGET_RUNTIME_KEY.to_string(),
        serde_json::to_value(budget).expect("a token budget serializes"),
    ))
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

/// Build the `RouterConfig` the primary workers advertise from the same
/// `--router-*` flags, so the two model cards hash equal.
///
/// The base is `KvRouterConfig::default()` with the standard `DYN_ROUTER_*` /
/// `DYN_SHARED_CACHE_*` overrides applied
/// (`dynamo_kv_router::config::kv_router_config_from_dynamo_env`), which is
/// exactly what the shared worker CLI picks up through its `env_var=` arguments.
/// The three fields the deployment YAML controls then override it. Every other
/// field keeps that shared value, which matches the shared CLI default
/// (`components/src/dynamo/common/configuration/groups/kv_router_args.py`) for
/// every field the CLI forwards. `shared_cache_multiplier` is the exception:
/// the primary workers are launched with an explicit
/// `--shared-cache-multiplier 0.5` (the CLI default), so the proxy pins the same
/// value instead of honouring `DYN_SHARED_CACHE_MULTIPLIER`. The variable must
/// never decide one side's card: `KvRouterConfig` is hashed into the card, so a
/// mismatch splits the worker set.
fn card_router_config(router: &ProxyRouterConfig) -> RouterConfig {
    card_router_config_pinning_shared_cache(
        router,
        dynamo_kv_router::config::kv_router_config_from_dynamo_env(),
    )
}

/// Force the shared worker CLI's `--shared-cache-multiplier` default on top of a
/// base `KvRouterConfig`.
///
/// Split from [`card_router_config`] so the pin is testable without touching
/// process environment variables.
fn card_router_config_pinning_shared_cache(
    router: &ProxyRouterConfig,
    mut base: dynamo_kv_router::KvRouterConfig,
) -> RouterConfig {
    base.shared_cache_multiplier = WORKER_CLI_SHARED_CACHE_MULTIPLIER;
    card_router_config_with_base(router, base)
}

/// Apply the proxy YAML's advertisement to a base `KvRouterConfig`.
///
/// Split from [`card_router_config`] so the layering is testable without
/// touching process environment variables.
fn card_router_config_with_base(
    router: &ProxyRouterConfig,
    mut kv_router_config: dynamo_kv_router::KvRouterConfig,
) -> RouterConfig {
    let mode = match router.mode {
        ProxyRouterMode::RoundRobin => RouterMode::RoundRobin,
        ProxyRouterMode::Random => RouterMode::Random,
        ProxyRouterMode::PowerOfTwoChoices => RouterMode::PowerOfTwoChoices,
        ProxyRouterMode::Kv => RouterMode::KV,
        ProxyRouterMode::Direct => RouterMode::Direct,
        ProxyRouterMode::LeastLoaded => RouterMode::LeastLoaded,
        ProxyRouterMode::DeviceAwareWeighted => RouterMode::DeviceAwareWeighted,
    };
    kv_router_config.router_track_active_blocks = router.track_active_blocks;
    kv_router_config.router_track_output_blocks = router.track_output_blocks;
    RouterConfig {
        router_mode: mode,
        kv_router_config,
        // Mirror the busy-worker rejection thresholds and session affinity too:
        // the card checksum covers the whole `RouterConfig`, so a primary that
        // set any `--active-*` or `--router-session-affinity-*` flag would
        // otherwise advertise a different checksum and split the worker set.
        load_threshold_config: proxy_load_threshold_config(&router.load_threshold_config),
        session_affinity_ttl_secs: router.session_affinity_ttl_secs,
        session_affinity_mode: match router.session_affinity_mode {
            ProxySessionAffinityMode::Hard => SessionAffinityMode::Hard,
            ProxySessionAffinityMode::Soft => SessionAffinityMode::Soft,
        },
        ..RouterConfig::default()
    }
}

/// Convert the proxy YAML's mirror of `--active-*` into the card's type.
fn proxy_load_threshold_config(config: &ProxyLoadThresholdConfig) -> LoadThresholdConfig {
    LoadThresholdConfig {
        active_decode_blocks_threshold: config.active_decode_blocks_threshold,
        active_prefill_tokens_threshold: config.active_prefill_tokens_threshold,
        active_prefill_tokens_threshold_frac: config.active_prefill_tokens_threshold_frac,
    }
}

/// `--shared-cache-multiplier`'s shared worker CLI default (every engine that
/// uses the common KV-router args group; `kv_router_kwargs` forwards every
/// KV-router field). The primary side advertises it, and `mdcsum()` serializes
/// `KvRouterConfig`, so a worker set only forms if the proxy advertises the same
/// value. The Rust `KvRouterConfig::default()` is 0.0.
const WORKER_CLI_SHARED_CACHE_MULTIPLIER: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    use dw_proxy_core::render::ParserFamily;
    use dw_proxy_core::upstream::ProviderConfig;

    fn sample() -> ProxyConfig {
        ProxyConfig {
            model_path: "/models/glm-5.3".to_string(),
            served_model_names: vec!["zai-org/GLM-5.3".to_string(), "glm-5.3".to_string()],
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            kv_block_size: 64,
            context_length: Some(202_752),
            custom_jinja_template: None,
            enable_eagle: false,
            omit_source_path: false,
            dp_rank: 7,
            tier: "spillover".to_string(),
            parser_family: ParserFamily::Glm47,
            endpoint_types: "chat,completions".to_string(),
            provider: ProviderConfig {
                name: "example-provider".to_string(),
                base_url: "https://api.provider.example/v1".to_string(),
                api_key_env: "PROVIDER_API_KEY".to_string(),
                model: "z-ai/glm-5.3".to_string(),
                provider_preferences: None,
                body_overrides: None,
                extra_headers: Default::default(),
                connect_timeout_ms: 10_000,
                read_timeout_ms: 120_000,
                thinking_dialect: Default::default(),
                thinking_strict: false,
                cache_key: Default::default(),
                cache_key_secret_env: None,
                allow_insecure_http: false,
                circuit_breaker: None,
            },
            vcache_ttl_secs: 300,
            vcache_max_blocks: 1_000_000,
            router_config: Some(ProxyRouterConfig {
                mode: ProxyRouterMode::Kv,
                track_active_blocks: true,
                track_output_blocks: false,
                ..Default::default()
            }),
            advertised_capacity: None,
        }
    }

    #[test]
    fn worker_config_mirrors_primary_surface() {
        let cfg = sample();
        let wc = worker_config(&cfg);
        assert_eq!(wc.namespace, "dynamo");
        assert_eq!(wc.component, "backend");
        assert_eq!(wc.endpoint, "generate");
        assert_eq!(wc.model_name, "/models/glm-5.3");
        assert_eq!(wc.served_model_name.as_deref(), Some("zai-org/GLM-5.3"));
        assert_eq!(wc.model_input, ModelInput::Tokens);
        assert_eq!(wc.endpoint_types, "chat,completions");
        assert!(wc.enable_local_indexer);
        // A proxy needs only config/tokenizer files, never full weights.
        assert!(wc.ignore_weights);
        assert_eq!(wc.custom_jinja_template, None);
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
    fn worker_config_uses_the_configured_endpoint_types() {
        // The proxy must be able to mirror a primary set that advertises only `chat`,
        // and the configured value must reach `WorkerConfig` verbatim (the backend
        // parser trims and lowercases it).
        let mut cfg = sample();
        cfg.endpoint_types = "chat".to_string();
        assert_eq!(worker_config(&cfg).endpoint_types, "chat");

        // A `chat,completions` proxy mirrors the production primary default, so the
        // two land in one WorkerSet (`model_type` is part of `worker_set_key`).
        let wc = worker_config(&sample());
        assert_eq!(wc.endpoint_types, "chat,completions");
        assert!(
            wc.endpoint_types
                .split(',')
                .all(|e| matches!(e, "chat" | "completions")),
            "{wc:?}"
        );
    }

    #[test]
    fn worker_config_advertises_a_recovery_target() {
        // A live-only proxy cannot re-sync after a frontend restart: a child
        // `Stored` published after the restart carries a `parent_hash` the new
        // frontend never indexed. Enabling the local indexer makes the worker
        // advertise a recovery target so the fresh frontend can pull the held
        // tree.
        let wc = worker_config(&sample());
        assert!(
            wc.enable_local_indexer,
            "the proxy must expose a local KV index for recovery"
        );
    }

    #[test]
    fn proxy_card_pins_shared_cache_multiplier_over_env() {
        // The primary side is launched with an explicit
        // `--shared-cache-multiplier 0.5`. If a cluster-wide
        // `DYN_SHARED_CACHE_MULTIPLIER` leaked into the proxy, the env value
        // would win here and `KvRouterConfig` would hash differently, rejecting
        // one cohort from the worker set. The proxy pins the CLI default
        // regardless of the environment-derived base.
        let base = dynamo_kv_router::KvRouterConfig {
            shared_cache_multiplier: 0.2,
            ..Default::default()
        };
        let rc = card_router_config_pinning_shared_cache(
            &ProxyRouterConfig {
                mode: ProxyRouterMode::Kv,
                track_active_blocks: true,
                track_output_blocks: false,
                ..Default::default()
            },
            base,
        );
        assert_eq!(
            rc.kv_router_config.shared_cache_multiplier, WORKER_CLI_SHARED_CACHE_MULTIPLIER,
            "the proxy card must not inherit DYN_SHARED_CACHE_MULTIPLIER"
        );
    }

    #[test]
    fn proxy_card_matches_primary_card_checksum() {
        // A proxy and a primary worker started from the same deployment must
        // advertise the same card `router_config`, because the checksum covers
        // it and a mismatch splits the worker set. The primary equivalent is what
        // `build_router_config` produces for
        // `--router-mode kv --router-track-active-blocks`
        // (`components/src/dynamo/common/configuration/groups/kv_router_args.py`):
        // `RouterConfig(mode=KV, KvRouterConfig(**kv_router_kwargs()))`, whose
        // `router_track_active_blocks` default is already true. The only other
        // field the shared CLI leaves off the Rust default is
        // `shared_cache_multiplier`, so the primary side names it too.
        //
        // NOTE: both sides are Rust structs, so this still cannot catch a
        // Python-side default change (r03-1/r10-5). It does exercise the same
        // base the deployed proxy uses.
        let proxy_router = card_router_config_with_base(
            &ProxyRouterConfig {
                mode: ProxyRouterMode::Kv,
                track_active_blocks: true,
                track_output_blocks: false,
                ..Default::default()
            },
            dynamo_kv_router::KvRouterConfig {
                shared_cache_multiplier: WORKER_CLI_SHARED_CACHE_MULTIPLIER,
                ..Default::default()
            },
        );
        let primary_router = RouterConfig {
            router_mode: RouterMode::KV,
            kv_router_config: dynamo_kv_router::KvRouterConfig {
                shared_cache_multiplier: WORKER_CLI_SHARED_CACHE_MULTIPLIER,
                ..Default::default()
            },
            ..RouterConfig::default()
        };
        assert_eq!(
            serde_json::to_value(&proxy_router).unwrap(),
            serde_json::to_value(&primary_router).unwrap(),
            "proxy and primary router configs must serialize identically"
        );

        let mut proxy_card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("m");
        proxy_card.router_config = Some(proxy_router);
        let mut primary_card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("m");
        primary_card.router_config = Some(primary_router);
        assert_eq!(proxy_card.mdcsum(), primary_card.mdcsum());
    }

    #[test]
    fn proxy_card_matches_primary_card_with_thresholds_and_affinity() {
        // The card checksum covers the whole `RouterConfig`, not only the KV
        // flags. A primary started with `--active-decode-blocks-threshold`,
        // `--active-prefill-tokens-threshold` or a session-affinity setting
        // advertises those fields, so a proxy that rebuilt the card from
        // `RouterConfig::default()` would advertise a different checksum and
        // never join the worker set.
        let proxy_router = card_router_config_with_base(
            &ProxyRouterConfig {
                mode: ProxyRouterMode::Kv,
                track_active_blocks: true,
                track_output_blocks: false,
                load_threshold_config: ProxyLoadThresholdConfig {
                    active_decode_blocks_threshold: Some(0.8),
                    active_prefill_tokens_threshold: Some(4096),
                    active_prefill_tokens_threshold_frac: Some(0.5),
                },
                session_affinity_ttl_secs: Some(3600),
                session_affinity_mode: ProxySessionAffinityMode::Soft,
            },
            dynamo_kv_router::KvRouterConfig {
                shared_cache_multiplier: WORKER_CLI_SHARED_CACHE_MULTIPLIER,
                ..Default::default()
            },
        );
        // The primary side is what the Python bindings' `build_router_config`
        // produces for the matching flags: `RouterConfig` with the thresholds in
        // `load_threshold_config` and the affinity fields set.
        let primary_router = RouterConfig {
            router_mode: RouterMode::KV,
            kv_router_config: dynamo_kv_router::KvRouterConfig {
                shared_cache_multiplier: WORKER_CLI_SHARED_CACHE_MULTIPLIER,
                ..Default::default()
            },
            load_threshold_config: LoadThresholdConfig {
                active_decode_blocks_threshold: Some(0.8),
                active_prefill_tokens_threshold: Some(4096),
                active_prefill_tokens_threshold_frac: Some(0.5),
            },
            session_affinity_ttl_secs: Some(3600),
            session_affinity_mode: SessionAffinityMode::Soft,
            ..RouterConfig::default()
        };
        assert_eq!(
            serde_json::to_value(&proxy_router).unwrap(),
            serde_json::to_value(&primary_router).unwrap(),
            "thresholds and session affinity must mirror onto the primary card"
        );

        let mut proxy_card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("m");
        proxy_card.router_config = Some(proxy_router);
        let mut primary_card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("m");
        primary_card.router_config = Some(primary_router);
        assert_eq!(proxy_card.mdcsum(), primary_card.mdcsum());
    }

    #[test]
    fn router_config_layers_base_and_yaml_overrides() {
        // The env-derived base survives except for the three fields the proxy
        // YAML owns, so a primary `--router-temperature`/env override is not
        // silently replaced by a Rust default (r10-4).
        let base = dynamo_kv_router::KvRouterConfig {
            router_temperature: 0.7,
            use_kv_events: false,
            router_track_active_blocks: false,
            router_track_output_blocks: true,
            ..Default::default()
        };
        let rc = card_router_config_with_base(
            &ProxyRouterConfig {
                mode: ProxyRouterMode::Kv,
                track_active_blocks: true,
                track_output_blocks: false,
                ..Default::default()
            },
            base,
        );
        assert_eq!(rc.router_mode, RouterMode::KV);
        assert!(rc.kv_router_config.router_track_active_blocks);
        assert!(!rc.kv_router_config.router_track_output_blocks);
        assert_eq!(rc.kv_router_config.router_temperature, 0.7);
        assert!(!rc.kv_router_config.use_kv_events);
    }

    #[test]
    fn engine_config_carries_kv_and_dp_metadata() {
        let cfg = sample();
        let ec = engine_config(&cfg);
        assert_eq!(ec.model, "/models/glm-5.3");
        assert_eq!(ec.served_model_name.as_deref(), Some("zai-org/GLM-5.3"));
        assert_eq!(ec.model_aliases, vec!["glm-5.3".to_string()]);

        let llm = ec.llm.expect("token engines carry LlmRegistration");
        assert_eq!(llm.context_length, Some(202_752));
        assert_eq!(llm.kv_cache_block_size, Some(64));
        // No advertised capacity: a plain proxy reports `None`, never a placeholder.
        assert_eq!(llm.total_kv_blocks, None);
        assert_eq!(llm.max_num_seqs, None);
        assert_eq!(llm.max_num_batched_tokens, Some(MAX_NUM_BATCHED_TOKENS));
        assert_eq!(llm.data_parallel_size, Some(1));
        assert_eq!(llm.data_parallel_start_rank, Some(7));
        assert!(!llm.enable_eagle);
    }

    #[test]
    fn engine_config_omits_context_length_when_unset() {
        // `None` must reach the card as `None` so it falls back to the model's
        // architectural maximum, exactly like a primary with no context flag.
        let mut cfg = sample();
        cfg.context_length = None;
        let llm = engine_config(&cfg).llm.expect("token engine");
        assert_eq!(llm.context_length, None);
    }

    #[test]
    fn engine_config_mirrors_custom_template_and_eagle() {
        let mut cfg = sample();
        cfg.custom_jinja_template = Some(std::path::PathBuf::from("/templates/primary.jinja"));
        cfg.enable_eagle = true;
        let wc = worker_config(&cfg);
        assert_eq!(
            wc.custom_jinja_template.as_deref(),
            Some(std::path::Path::new("/templates/primary.jinja"))
        );
        let llm = engine_config(&cfg).llm.expect("token engine");
        assert!(llm.enable_eagle);
    }

    #[test]
    fn engine_config_advertises_configured_capacity() {
        let mut cfg = sample();
        cfg.advertised_capacity = Some(dw_proxy_core::config::AdvertisedCapacity {
            kv_blocks: Some(4096),
            max_requests: Some(32),
        });
        let llm = engine_config(&cfg).llm.expect("token engine");
        assert_eq!(llm.total_kv_blocks, Some(4096));
        assert_eq!(llm.max_num_seqs, Some(32));
    }

    #[test]
    fn engine_config_leaves_disabled_capacity_unset() {
        // Either field may be absent on its own; the other stays unadvertised.
        let mut cfg = sample();
        cfg.advertised_capacity = Some(dw_proxy_core::config::AdvertisedCapacity {
            kv_blocks: Some(4096),
            max_requests: None,
        });
        let llm = engine_config(&cfg).llm.expect("token engine");
        assert_eq!(llm.total_kv_blocks, Some(4096));
        assert_eq!(llm.max_num_seqs, None);
    }

    #[test]
    fn worker_config_registers_the_frontend_parsers_for_its_family() {
        let wc = worker_config(&sample());
        // `sample()` is a GLM proxy.
        assert_eq!(wc.tool_call_parser.as_deref(), Some("glm47"));
        assert_eq!(wc.reasoning_parser.as_deref(), Some("glm45"));
    }

    #[test]
    fn engine_config_asks_for_the_chat_request() {
        let ec = engine_config(&sample());
        assert_eq!(
            ec.runtime_data.get(CHAT_REQUEST_CAPABILITY),
            Some(&serde_json::json!(true))
        );
        // The proxy reads the key the frontend writes.
        assert_eq!(
            dw_proxy_core::chat_request::EXTRA_ARGS_KEY,
            dynamo_llm::local_model::runtime_config::CHAT_REQUEST_EXTRA_ARGS_KEY
        );
    }

    #[test]
    fn engine_config_publishes_the_primary_token_budget() {
        use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
        // Read back the way the frontend's preprocessor reads it from the card.
        let budget = |cfg: &ProxyConfig| {
            let runtime_config = ModelRuntimeConfig {
                runtime_data: engine_config(cfg).runtime_data,
                ..ModelRuntimeConfig::default()
            };
            runtime_config
                .get_engine_specific::<TokenBudget>(TOKEN_BUDGET_RUNTIME_KEY)
                .expect("the budget parses")
        };
        let mut cfg = sample();
        assert_eq!(
            budget(&cfg),
            Some(TokenBudget {
                combined_limit: 202_752,
                reject_prompt_overflow: true,
                reject_total_overflow: true,
            })
        );
        cfg.context_length = None;
        assert_eq!(budget(&cfg), None, "no context length, no budget");
    }

    #[test]
    fn health_payload_is_an_object_with_tokens() {
        let payload = health_check_payload(&sample());
        assert!(payload.is_object());
        assert_eq!(payload["model"], serde_json::json!("zai-org/GLM-5.3"));
        assert_eq!(payload["token_ids"], serde_json::json!([1]));
    }
}
