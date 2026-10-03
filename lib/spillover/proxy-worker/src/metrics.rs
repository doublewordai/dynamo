// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the proxy worker.
//!
//! The proxy registers as an ordinary Dynamo worker, so it exposes metrics on
//! the same surface as every other worker: the runtime's `/metrics` endpoint on
//! `DYN_SYSTEM_PORT`. [`ProxyMetrics`] is built from the
//! [`EngineMetrics`](dynamo_backend_common::EngineMetrics) handle the
//! framework hands to [`LLMEngine::setup_metrics`](dynamo_backend_common::LLMEngine::setup_metrics),
//! exactly like `LifecycleGauges` and `ComponentGauges` in
//! `lib/backend-common/src/metrics.rs`. `create_metric` auto-injects
//! `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint` and `worker_id`;
//! `model`/`model_name` arrive from `EngineMetrics::auto_labels` and
//! `tier`/`provider` are added here, one series set per proxy process.
//!
//! Names are therefore prefixed by the framework:
//! `dynamo_component_proxy_requests_total`, etc.

use std::sync::Arc;
use std::time::Instant;

use dynamo_backend_common::EngineMetrics;
use dynamo_runtime::metrics::{create_metric, prometheus_names::labels};
use prometheus::{Counter, Histogram, IntCounter, IntCounterVec, IntGauge};
use serde_json::Value;

/// Request outcome, one value of the `outcome` label on
/// `dynamo_component_proxy_requests_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Finished with a provider `finish_reason`.
    Ok,
    /// Provider returned HTTP 429 or a rate-limit error.
    RateLimited,
    /// Provider returned 408/5xx or a stream error object.
    Unavailable,
    /// Provider rejected the request (any 4xx other than 401), for example moderation or a
    /// provider-specific limit; retried on another worker.
    Rejected,
    /// Provider rejected the key (401): the proxy is unusable and is reported down.
    AuthError,
    /// Could not connect, TLS failure or a dropped connection.
    Transport,
    /// The stream ended or failed before a finish reason.
    StreamBroken,
    /// The proxy's own renderer, tokenizer or retokenizer failed while turning provider text
    /// into tokens. This is local to the proxy, so it does not touch `provider_healthy` or the
    /// circuit breaker: the provider may be perfectly healthy.
    RenderFailed,
    /// The caller cancelled before the stream finished.
    Cancelled,
    /// A migration retry that replayed output the proxy cannot continue.
    MigrationReplay,
    /// No chat request was attached, so the proxy cannot call the provider.
    NoChatRequest,
    /// The provider stopped the response with its content filter; retried on another worker.
    ContentFiltered,
    /// The chat request asks for something the proxy cannot serve faithfully (for example
    /// `n > 1`, logprobs or guided decoding); retried on another worker.
    Unsupported,
    /// The proxy's circuit breaker is open, so the request was refused before any provider
    /// call; retried on another worker. Does not touch `provider_healthy`, which the failure
    /// that opened the breaker already cleared.
    CircuitOpen,
    /// The prompt, or the prompt plus the requested output, does not fit the model's context
    /// length: refused before any provider call as a client error, as a primary worker does.
    ContextOverflow,
}

impl Outcome {
    /// Every outcome, for tests and for documenting the label's value set.
    #[cfg(test)]
    pub const ALL: [Outcome; 15] = [
        Outcome::Ok,
        Outcome::RateLimited,
        Outcome::Unavailable,
        Outcome::Rejected,
        Outcome::AuthError,
        Outcome::Transport,
        Outcome::StreamBroken,
        Outcome::RenderFailed,
        Outcome::Cancelled,
        Outcome::MigrationReplay,
        Outcome::NoChatRequest,
        Outcome::Unsupported,
        Outcome::ContentFiltered,
        Outcome::CircuitOpen,
        Outcome::ContextOverflow,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::RateLimited => "rate_limited",
            Outcome::Unavailable => "unavailable",
            Outcome::Rejected => "rejected",
            Outcome::AuthError => "auth_error",
            Outcome::Transport => "transport",
            Outcome::StreamBroken => "stream_broken",
            Outcome::RenderFailed => "render_failed",
            Outcome::Cancelled => "cancelled",
            Outcome::MigrationReplay => "migration_replay",
            Outcome::NoChatRequest => "no_chat_request",
            Outcome::Unsupported => "unsupported",
            Outcome::ContentFiltered => "content_filtered",
            Outcome::CircuitOpen => "circuit_open",
            Outcome::ContextOverflow => "context_overflow",
        }
    }
}

/// Label values of `proxy_thinking_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingEvent {
    /// The provider's dialect could not express the request's thinking choice.
    Unexpressed,
    /// The provider returned reasoning although the request turned thinking off.
    Ignored,
}

impl ThinkingEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingEvent::Unexpressed => "unexpressed",
            ThinkingEvent::Ignored => "ignored",
        }
    }
}

/// Latency buckets in seconds, from a fast provider to a long generation.
fn latency_buckets() -> Vec<f64> {
    vec![
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    ]
}

/// Counters and gauges for one proxy process. A process serves exactly one
/// provider at one tier, so `tier` and `provider` are constant labels alongside
/// the model labels the framework computes.
pub struct ProxyMetrics {
    requests: IntCounterVec,
    ttft_seconds: Histogram,
    duration_seconds: Histogram,
    prompt_tokens: IntCounter,
    completion_tokens: IntCounter,
    cached_prompt_tokens: IntCounter,
    provider_cost: Counter,
    inflight: IntGauge,
    vcache_blocks: IntGauge,
    provider_healthy: IntGauge,
    circuit_open: IntGauge,
    kv_events: IntCounterVec,
    kv_events_dropped: IntCounter,
    thinking: IntCounterVec,
}

impl ProxyMetrics {
    /// Register every instrument. `tier` and `provider` are the config values;
    /// the model labels come from the framework's [`EngineMetrics`].
    pub fn new(metrics: &EngineMetrics, tier: &str, provider: &str) -> anyhow::Result<Self> {
        let const_labels = const_labels(metrics, tier, provider);
        let labels: Vec<(&str, &str)> = const_labels
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let hierarchy = metrics.hierarchy().as_ref();

        let requests = create_metric::<IntCounterVec, _>(
            hierarchy,
            "proxy_requests_total",
            "Proxy requests by final outcome.",
            &labels,
            None,
            Some(&["outcome"]),
        )?;
        let ttft_seconds = create_metric::<Histogram, _>(
            hierarchy,
            "proxy_time_to_first_token_seconds",
            "Time from accepting a proxy request to its first content chunk.",
            &labels,
            Some(latency_buckets()),
            None,
        )?;
        let duration_seconds = create_metric::<Histogram, _>(
            hierarchy,
            "proxy_request_duration_seconds",
            "Total proxy request duration, whatever the outcome.",
            &labels,
            Some(latency_buckets()),
            None,
        )?;
        let prompt_tokens = create_metric::<IntCounter, _>(
            hierarchy,
            "proxy_prompt_tokens_total",
            "Prompt tokens billed by the provider, from its usage object.",
            &labels,
            None,
            None,
        )?;
        let completion_tokens = create_metric::<IntCounter, _>(
            hierarchy,
            "proxy_completion_tokens_total",
            "Completion tokens billed by the provider, from its usage object.",
            &labels,
            None,
            None,
        )?;
        let cached_prompt_tokens = create_metric::<IntCounter, _>(
            hierarchy,
            "proxy_cached_prompt_tokens_total",
            "Prompt tokens the provider reported serving from its prompt cache \
             (usage.prompt_tokens_details.cached_tokens).",
            &labels,
            None,
            None,
        )?;
        let provider_cost = create_metric::<Counter, _>(
            hierarchy,
            "proxy_provider_cost_total",
            "Cost the provider reported in its usage object (usage.cost), in the provider's \
             billing unit. Zero for providers that do not report it.",
            &labels,
            None,
            None,
        )?;
        let inflight = create_metric::<IntGauge, _>(
            hierarchy,
            "proxy_inflight_requests",
            "Proxy requests currently streaming from the provider.",
            &labels,
            None,
            None,
        )?;
        let vcache_blocks = create_metric::<IntGauge, _>(
            hierarchy,
            "proxy_virtual_cache_blocks",
            "Blocks currently held in the proxy's virtual cache.",
            &labels,
            None,
            None,
        )?;
        let provider_healthy = create_metric::<IntGauge, _>(
            hierarchy,
            "proxy_provider_healthy",
            "0 until a provider outcome is recorded, then 1 after a provider success or 0 after a \
             provider-side failure; readiness is process liveness only, so alert provider health \
             from this gauge. Start-unknown lets an alert tell 'no traffic yet' from 'provider \
             healthy'.",
            &labels,
            None,
            None,
        )?;
        // Unknown until the first provider outcome; a success flips it to 1 and a provider-side
        // failure to 0. Starting at 1 would report a provider healthy before it was ever called.
        provider_healthy.set(0);
        let circuit_open = create_metric::<IntGauge, _>(
            hierarchy,
            "proxy_circuit_open",
            "1 while the provider circuit breaker is open or half-open (refusing requests), \
             0 while it is closed.",
            &labels,
            None,
            None,
        )?;
        circuit_open.set(0);
        let kv_events = create_metric::<IntCounterVec, _>(
            hierarchy,
            "proxy_kv_events_total",
            "Virtual-cache events published to the router, by kind.",
            &labels,
            None,
            Some(&["kind"]),
        )?;
        let kv_events_dropped = create_metric::<IntCounter, _>(
            hierarchy,
            "proxy_kv_events_dropped_total",
            "Virtual-cache events that never reached the router: a failed publish, a pending \
             overflow, or a replay after the publisher closed.",
            &labels,
            None,
            None,
        )?;

        let thinking = create_metric::<IntCounterVec, _>(
            hierarchy,
            "proxy_thinking_total",
            "Requests whose thinking choice the provider's dialect could not express \
             (event=unexpressed), and responses that reasoned after thinking was turned off \
             (event=ignored).",
            &labels,
            None,
            Some(&["event"]),
        )?;
        Ok(Self {
            requests,
            ttft_seconds,
            duration_seconds,
            prompt_tokens,
            completion_tokens,
            inflight,
            vcache_blocks,
            provider_healthy,
            circuit_open,
            kv_events,
            kv_events_dropped,
            thinking,
            cached_prompt_tokens,
            provider_cost,
        })
    }

    /// Record one terminal request outcome and its total duration.
    ///
    /// A provider-side failure clears [`Self::provider_healthy`] so an alert can fire on a dead
    /// or revoked provider instead of waiting for the frontend to report the worker down.
    pub fn record_outcome(&self, outcome: Outcome, duration_seconds: f64) {
        self.requests.with_label_values(&[outcome.as_str()]).inc();
        self.duration_seconds.observe(duration_seconds);
        match outcome {
            // A provider response (even a client 4xx) proves reachability and credentials.
            Outcome::Ok | Outcome::Rejected | Outcome::ContentFiltered => {
                self.provider_healthy.set(1)
            }
            Outcome::AuthError
            | Outcome::RateLimited
            | Outcome::Unavailable
            | Outcome::Transport
            | Outcome::StreamBroken => self.provider_healthy.set(0),
            // Proxy-side refusals, cancellations and local render failures say nothing about the
            // provider.
            Outcome::Cancelled
            | Outcome::MigrationReplay
            | Outcome::NoChatRequest
            | Outcome::Unsupported
            | Outcome::RenderFailed
            | Outcome::CircuitOpen
            | Outcome::ContextOverflow => {}
        }
    }

    /// Count a thinking event: `unexpressed` or `ignored` (see `proxy_thinking_total`).
    pub fn record_thinking(&self, event: ThinkingEvent) {
        self.thinking.with_label_values(&[event.as_str()]).inc();
    }

    /// Add provider-reported cached prompt tokens.
    pub fn add_cached_prompt_tokens(&self, tokens: u64) {
        self.cached_prompt_tokens.inc_by(tokens);
    }

    /// Add provider-reported cost.
    pub fn add_provider_cost(&self, cost: f64) {
        self.provider_cost.inc_by(cost);
    }

    /// Record time to first content chunk.
    pub fn observe_ttft(&self, seconds: f64) {
        self.ttft_seconds.observe(seconds);
    }

    /// Add provider-reported prompt tokens.
    pub fn add_prompt_tokens(&self, tokens: u64) {
        self.prompt_tokens.inc_by(tokens);
    }

    /// Add provider-reported completion tokens.
    pub fn add_completion_tokens(&self, tokens: u64) {
        self.completion_tokens.inc_by(tokens);
    }

    /// Replace the virtual-cache block gauge with a fresh reading.
    pub fn set_vcache_blocks(&self, blocks: usize) {
        self.vcache_blocks.set(blocks as i64);
    }

    /// Set the provider circuit gauge: 1 while open or half-open, 0 while closed.
    pub fn set_circuit_open(&self, open: bool) {
        self.circuit_open.set(i64::from(open));
    }

    /// Count virtual-cache events published to the router, by kind.
    pub fn add_kv_events(&self, kind: &str, count: u64) {
        if count == 0 {
            return;
        }
        self.kv_events.with_label_values(&[kind]).inc_by(count);
    }

    /// Count virtual-cache events that never reached the router.
    pub fn add_kv_events_dropped(&self, count: u64) {
        if count == 0 {
            return;
        }
        self.kv_events_dropped.inc_by(count);
    }

    /// A guard that decrements the in-flight gauge when the request stream is
    /// dropped, so a client disconnect cannot leak the gauge.
    pub fn inflight_guard(self: &Arc<Self>) -> InflightGuard {
        self.inflight.inc();
        InflightGuard {
            metrics: self.clone(),
        }
    }
}

/// RAII decrement for [`ProxyMetrics::inflight_guard`].
pub struct InflightGuard {
    metrics: Arc<ProxyMetrics>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.metrics.inflight.dec();
    }
}

/// Record a terminal outcome, TTFT and token usage in one call. `metrics` is
/// `None` before `setup_metrics` has run (e.g. a health probe answered before
/// the framework wires metrics).
pub fn record_terminal(
    metrics: &Option<Arc<ProxyMetrics>>,
    started: Instant,
    outcome: Outcome,
    first_token_at: Option<Instant>,
    usage: Option<&dynamo_backend_common::CompletionUsage>,
) {
    let Some(metrics) = metrics.as_ref() else {
        return;
    };
    if let Some(first) = first_token_at {
        metrics.observe_ttft(first.duration_since(started).as_secs_f64());
    }
    metrics.record_outcome(outcome, started.elapsed().as_secs_f64());
    if let Some(usage) = usage {
        metrics.add_prompt_tokens(u64::from(usage.prompt_tokens));
        metrics.add_completion_tokens(u64::from(usage.completion_tokens));
    }
}

/// The served-by tag attached to every proxy output. Accounting reads it from
/// `nvext.engine_data` when the request opts into `nvext.extra_fields`.
pub fn served_by(config: &dw_proxy_core::config::ProxyConfig) -> Value {
    serde_json::json!({
        "served_by": config.tier,
        "tier": config.tier,
    })
}

/// Build the constant-label set: the framework's model labels plus this
/// process's tier and provider, minus the labels `create_metric` injects from
/// the hierarchy (passing them would be rejected as duplicates).
fn const_labels(metrics: &EngineMetrics, tier: &str, provider: &str) -> Vec<(String, String)> {
    let auto_injected = [
        labels::NAMESPACE,
        labels::COMPONENT,
        labels::ENDPOINT,
        labels::WORKER_ID,
    ];
    let mut out: Vec<(String, String)> = metrics
        .auto_labels()
        .iter()
        .filter(|(k, _)| !auto_injected.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    out.push(("tier".to_string(), tier.to_string()));
    out.push(("provider".to_string(), provider.to_string()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::EngineConfig;
    use dynamo_runtime::metrics::{MetricsHierarchy, MetricsRegistry};

    /// Standalone hierarchy with no parent, mirroring `TestHierarchy` in
    /// `lib/backend-common/src/metrics.rs`. The model labels come from
    /// `with_engine_config`; the tier/provider labels from `ProxyMetrics::new`.
    #[derive(Default)]
    struct TestHierarchy {
        registry: MetricsRegistry,
    }

    impl MetricsHierarchy for TestHierarchy {
        fn basename(&self) -> String {
            "generate".to_string()
        }
        fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
            Vec::new()
        }
        fn get_metrics_registry(&self) -> &MetricsRegistry {
            &self.registry
        }
    }

    /// Keep the `EngineMetrics` alive so the test can scrape the registry the
    /// instruments were registered on.
    fn setup() -> (EngineMetrics, ProxyMetrics) {
        let config = EngineConfig {
            model: "/models/glm-5.3".to_string(),
            served_model_name: Some("zai-org/GLM-5.3".to_string()),
            ..EngineConfig::default()
        };
        let engine_metrics = EngineMetrics::with_engine_config(TestHierarchy::default(), &config);
        let metrics = ProxyMetrics::new(&engine_metrics, "spillover", "example-provider")
            .expect("register metrics");
        (engine_metrics, metrics)
    }

    fn scrape(engine_metrics: &EngineMetrics) -> String {
        engine_metrics
            .hierarchy()
            .get_metrics_registry()
            .prometheus_expfmt_combined()
            .expect("expfmt")
    }

    /// A single data row (not a HELP/TYPE comment) whose name starts with
    /// `prefix`. Panics with the full exposition on absence.
    fn data_row<'a>(text: &'a str, prefix: &str) -> &'a str {
        text.lines()
            .find(|line| line.starts_with(prefix) && !line.starts_with('#'))
            .unwrap_or_else(|| panic!("no data row for {prefix} in:\n{text}"))
    }

    #[test]
    fn outcome_names_cover_the_contract() {
        let names: Vec<&str> = Outcome::ALL.iter().map(|o| o.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "ok",
                "rate_limited",
                "unavailable",
                "rejected",
                "auth_error",
                "transport",
                "stream_broken",
                "render_failed",
                "cancelled",
                "migration_replay",
                "no_chat_request",
                "unsupported",
                "content_filtered",
                "circuit_open",
                "context_overflow",
            ]
        );
    }

    #[test]
    fn records_every_outcome_path() {
        let (engine_metrics, metrics) = setup();
        for outcome in Outcome::ALL {
            metrics.record_outcome(outcome, 0.5);
        }
        let text = scrape(&engine_metrics);
        for outcome in Outcome::ALL {
            let row = text
                .lines()
                .find(|line| {
                    line.starts_with("dynamo_component_proxy_requests_total")
                        && line.contains(&format!("outcome=\"{}\"", outcome.as_str()))
                })
                .unwrap_or_else(|| panic!("missing outcome {} in:\n{text}", outcome.as_str()));
            assert!(
                row.ends_with(" 1"),
                "unexpected count for {}: {row}",
                outcome.as_str()
            );
        }
        // Constant labels are attached to every series.
        assert!(text.contains("provider=\"example-provider\""));
        assert!(text.contains("tier=\"spillover\""));
        assert!(text.contains("model_name=\"zai-org/GLM-5.3\""));
    }

    #[test]
    fn records_provider_cache_and_cost() {
        let (engine_metrics, metrics) = setup();
        metrics.add_cached_prompt_tokens(128);
        metrics.add_provider_cost(0.25);
        metrics.add_provider_cost(0.5);
        let text = scrape(&engine_metrics);
        assert!(
            data_row(&text, "dynamo_component_proxy_cached_prompt_tokens_total").ends_with(" 128")
        );
        assert!(data_row(&text, "dynamo_component_proxy_provider_cost_total").ends_with(" 0.75"));
    }

    #[test]
    fn records_ttft_duration_and_tokens() {
        let (engine_metrics, metrics) = setup();
        let started = Instant::now();
        let first = started + std::time::Duration::from_millis(20);
        let usage = dynamo_backend_common::usage(11, 7);
        record_terminal(
            &Some(Arc::new(metrics)),
            started,
            Outcome::Ok,
            Some(first),
            Some(&usage),
        );
        let text = scrape(&engine_metrics);
        assert!(data_row(&text, "dynamo_component_proxy_prompt_tokens_total").ends_with(" 11"));
        assert!(data_row(&text, "dynamo_component_proxy_completion_tokens_total").ends_with(" 7"));
        assert!(text.contains("dynamo_component_proxy_time_to_first_token_seconds_bucket"));
        assert!(text.contains("dynamo_component_proxy_request_duration_seconds_count"));
    }

    #[test]
    fn records_gauges_and_kv_event_kinds() {
        let (engine_metrics, metrics) = setup();
        metrics.set_vcache_blocks(42);
        metrics.add_kv_events("stored", 3);
        metrics.add_kv_events("removed", 2);
        metrics.add_kv_events("cleared", 1);
        metrics.add_kv_events_dropped(4);

        let metrics = Arc::new(metrics);
        let guard = metrics.inflight_guard();
        let text = scrape(&engine_metrics);
        assert!(data_row(&text, "dynamo_component_proxy_virtual_cache_blocks").ends_with(" 42"));
        assert!(data_row(&text, "dynamo_component_proxy_inflight_requests").ends_with(" 1"));
        assert!(data_row(&text, "dynamo_component_proxy_kv_events_dropped_total").ends_with(" 4"));
        for (kind, count) in [("stored", 3), ("removed", 2), ("cleared", 1)] {
            let row = text
                .lines()
                .find(|line| {
                    line.starts_with("dynamo_component_proxy_kv_events_total")
                        && line.contains(&format!("kind=\"{kind}\""))
                })
                .unwrap_or_else(|| panic!("missing kind {kind} in:\n{text}"));
            assert!(
                row.ends_with(&format!(" {count}")),
                "unexpected {kind}: {row}"
            );
        }
        drop(guard);
        let text = scrape(&engine_metrics);
        assert!(data_row(&text, "dynamo_component_proxy_inflight_requests").ends_with(" 0"));
    }

    #[test]
    fn provider_healthy_gauge_tracks_provider_outcomes() {
        let (engine_metrics, metrics) = setup();
        // Unknown before the first provider outcome: not healthy and not failed.
        let text = scrape(&engine_metrics);
        assert!(
            data_row(&text, "dynamo_component_proxy_provider_healthy").ends_with(" 0"),
            "the health gauge must start unknown, not healthy:\n{text}"
        );
        metrics.record_outcome(Outcome::Ok, 0.1);
        let text = scrape(&engine_metrics);
        assert!(
            data_row(&text, "dynamo_component_proxy_provider_healthy").ends_with(" 1"),
            "a success marks the provider healthy:\n{text}"
        );
        metrics.record_outcome(Outcome::Unavailable, 0.1);
        let text = scrape(&engine_metrics);
        assert!(
            data_row(&text, "dynamo_component_proxy_provider_healthy").ends_with(" 0"),
            "a provider-side failure clears the health gauge:\n{text}"
        );
        // A proxy-side refusal or a local render failure is not evidence about the provider, so
        // neither changes the gauge.
        metrics.record_outcome(Outcome::MigrationReplay, 0.1);
        metrics.record_outcome(Outcome::RenderFailed, 0.1);
        let text = scrape(&engine_metrics);
        assert!(data_row(&text, "dynamo_component_proxy_provider_healthy").ends_with(" 0"));
    }

    #[test]
    fn circuit_gauge_tracks_the_breaker_and_refusals_do_not_touch_provider_health() {
        let (engine_metrics, metrics) = setup();
        // The failure that opens the breaker already cleared health; refusing
        // more requests must not flip it back to healthy.
        metrics.record_outcome(Outcome::Unavailable, 0.1);
        metrics.set_circuit_open(true);
        metrics.record_outcome(Outcome::CircuitOpen, 0.01);
        let text = scrape(&engine_metrics);
        assert!(data_row(&text, "dynamo_component_proxy_circuit_open").ends_with(" 1"));
        assert!(data_row(&text, "dynamo_component_proxy_provider_healthy").ends_with(" 0"));

        metrics.set_circuit_open(false);
        let text = scrape(&engine_metrics);
        assert!(data_row(&text, "dynamo_component_proxy_circuit_open").ends_with(" 0"));
    }

    #[test]
    fn served_by_tag_carries_the_tier_and_not_the_provider() {
        // The provider name must not reach clients: `served_by()` deliberately
        // returns the tier in both fields, so an `nvext.engine_data` request
        // identifies the tier but never leaks which external provider answered
        // (see the leak notes in lib/spillover/deploy/config/README.md).
        use dw_proxy_core::config::ProxyConfig;
        use dw_proxy_core::render::ParserFamily;
        use dw_proxy_core::upstream::ProviderConfig;
        let config = ProxyConfig {
            model_path: "/models/glm-5.3".to_string(),
            served_model_names: vec!["zai-org/GLM-5.3".to_string()],
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            kv_block_size: 64,
            context_length: Some(202_752),
            custom_jinja_template: None,
            enable_eagle: false,
            omit_source_path: false,
            model_revision: None,
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
                omitted_max_tokens: 131_072,
                thinking_dialect: Default::default(),
                thinking_strict: false,
                cache_key: Default::default(),
                cache_key_secret_env: None,
                allow_insecure_http: false,
                circuit_breaker: None,
            },
            vcache_ttl_secs: 300,
            vcache_max_blocks: 1_000_000,
            router_config: None,
            advertised_capacity: None,
        };
        let tag = served_by(&config);
        assert_eq!(tag["served_by"], tag["tier"]);
        assert_eq!(tag["tier"], "spillover");
        let serialized = tag.to_string();
        assert!(
            !serialized.contains("example-provider"),
            "the provider name must not appear in the served-by tag: {serialized}"
        );
    }
}
