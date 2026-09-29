// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The proxy engine: an [`LLMEngine`] that answers every Dynamo request from a
//! third-party OpenAI-compatible provider.
//!
//! The request arrives as a `PreprocessedRequest` (token ids, sampling options)
//! with the original chat body carried in `extra_args` by onwards (see
//! `dw_proxy_core::orig`). We rebuild the provider body, stream the provider's
//! deltas, render them back into the model's raw output format, retokenize the
//! text, and yield `LLMEngineOutput` chunks. The frontend sees a normal SGLang
//! worker: token ids it can count and migrate, plus raw text its own parsers
//! consume.
//!
//! The trait lives in `lib/backend-common/src/engine.rs`; the streaming contract
//! (`Err` before the stream is a request failure, `Err` inside the stream ends it
//! with an `Annotated::from_err`) is `lib/backend-common/src/adapter.rs`.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dw_proxy_core::config::ProxyConfig;
use dw_proxy_core::errors::UpstreamError;
use dw_proxy_core::orig;
use dw_proxy_core::render::{self, RenderError};
use dw_proxy_core::retokenize::Retokenizer;
use dw_proxy_core::upstream::UpstreamClient;
use dw_proxy_core::vcache::{HashOptions, VirtualCache, VirtualCacheConfig};
use dynamo_backend_common::{
    BackendError, CompletionUsage, DynamoError, EngineConfig, ErrorType, FinishReason,
    GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput, MetricsBindings, MetricsCtx,
    PreprocessedRequest,
};
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;
use tokio::task::JoinHandle;

use crate::kv::EventSink;
use crate::metrics::{self, Outcome, ProxyMetrics, record_terminal};
use crate::registration;

/// Shared, interior-mutable engine state. `LLMEngine` is driven concurrently,
/// so the cache and the publisher slot live behind mutexes and the expire timer
/// behind a `JoinHandle`.
struct EngineState {
    vcache: Mutex<VirtualCache>,
    events: EventSink,
    expire_task: Mutex<Option<JoinHandle<()>>>,
    /// Set once by [`LLMEngine::setup_metrics`]. `None` before the framework
    /// wires metrics (health probes answered earlier still work).
    metrics: Mutex<Option<Arc<ProxyMetrics>>>,
}

/// A Dynamo engine backed by one third-party provider.
pub struct ProxyEngine {
    config: Arc<ProxyConfig>,
    client: UpstreamClient,
    /// The model's tokenizer, loaded once and shared by every stream's `Retokenizer`.
    tokenizer: Arc<tokenizers::Tokenizer>,
    state: Arc<EngineState>,
}

impl ProxyEngine {
    /// Build the engine. Reads the provider API key (`UpstreamClient::new`), so
    /// this fails fast when the environment is misconfigured. A `model_path` that
    /// is not on disk is fetched like the worker card does, so a hub id resolves
    /// from the offline cache instead of being treated as a tokenizer file.
    pub async fn new(config: ProxyConfig) -> anyhow::Result<Self> {
        let client = UpstreamClient::new(config.provider.clone())?;
        let vcache = VirtualCache::new(VirtualCacheConfig {
            block_size: config.kv_block_size,
            ttl: Duration::from_secs(config.vcache_ttl_secs),
            max_blocks: config.vcache_max_blocks,
        });
        let state = Arc::new(EngineState {
            vcache: Mutex::new(vcache),
            events: EventSink::new(config.dp_rank),
            expire_task: Mutex::new(None),
            metrics: Mutex::new(None),
        });
        let tokenizer = load_tokenizer(&config.model_path).await?;
        Ok(Self {
            config: Arc::new(config),
            client,
            tokenizer: Arc::new(tokenizer),
            state,
        })
    }

    /// Record the served prompt in the virtual cache and publish the resulting
    /// stored events so the router keeps conversations sticky to this rank.
    fn record_prompt(&self, prompt_tokens: &[u32], options: &HashOptions) {
        let (events, blocks) = {
            let mut cache = self.state.vcache.lock().unwrap_or_else(|e| e.into_inner());
            let events = cache.on_request(prompt_tokens, options, Instant::now());
            (events, cache.len_blocks())
        };
        self.record_cache(events, blocks);
    }

    /// The metrics handle, if `setup_metrics` has run.
    fn metrics(&self) -> Option<Arc<ProxyMetrics>> {
        self.state
            .metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Publish a batch of cache events and account for it: KV events by kind
    /// and the current virtual-cache block count.
    fn record_cache(&self, events: Vec<dw_proxy_core::vcache::CacheEvent>, blocks: usize) {
        let published = self.state.events.publish(events);
        if let Some(metrics) = self.metrics() {
            for (kind, count) in published.kinds() {
                metrics.add_kv_events(kind, count);
            }
            metrics.set_vcache_blocks(blocks);
        }
    }

    /// Publish the prompt's cache state without waiting for the provider. The
    /// provider may fail after this point; the router falls back to routing on
    /// tokens, and TTL expiry eventually removes the virtual blocks.
    fn spawn_expirer(&self) {
        let state = self.state.clone();
        let period = expire_period(&self.config);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(period);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                let (events, blocks) = {
                    let mut cache = state.vcache.lock().unwrap_or_else(|e| e.into_inner());
                    let events = cache.expire(Instant::now());
                    (events, cache.len_blocks())
                };
                let published = state.events.publish(events);
                if let Some(metrics) = state
                    .metrics
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    for (kind, count) in published.kinds() {
                        metrics.add_kv_events(kind, count);
                    }
                    metrics.set_vcache_blocks(blocks);
                }
            }
        });
        *self
            .state
            .expire_task
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(handle);
    }
}

#[async_trait]
impl LLMEngine for ProxyEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        self.spawn_expirer();
        Ok(registration::engine_config(&self.config))
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        // Runtime canary: answer locally so a provider outage never marks the
        // worker unhealthy. `Worker` stamped `_HEALTH_CHECK`, which deserialized
        // into `is_probe` (see `lib/runtime/src/health_check.rs` and
        // `lib/backend-common/src/adapter.rs`'s `JsonProbeAdapter`).
        if request.is_probe {
            let stream = async_stream::stream! {
                yield Ok(LLMEngineOutput::stop());
            };
            return Ok(Box::pin(stream));
        }

        // The preprocessor only carries the original chat body when onwards put
        // it in `nvext.extra_fields`; without it the proxy has nothing to send.
        let metrics = self.metrics();
        let started = Instant::now();
        let original = match orig::from_extra_args(request.extra_args.as_ref()) {
            Ok(Some(original)) => original,
            Ok(None) => {
                record_terminal(&metrics, started, Outcome::Rejected, None, None);
                return Err(client_error("request is missing the dw.orig chat payload"));
            }
            Err(err) => {
                record_terminal(&metrics, started, Outcome::Rejected, None, None);
                return Err(client_error(format!("invalid dw.orig chat payload: {err}")));
            }
        };

        let prompt_tokens = request.token_ids.as_ref().len() as u32;
        let options = hash_options(&request);
        self.record_prompt(request.token_ids.as_ref(), &options);

        // Held-back-tail retokenization is per stream, so each request gets its own
        // `Retokenizer` over the shared tokenizer.
        // The frontend's parser starts inside reasoning for templates that end the
        // prompt with the opener (GLM/DeepSeek with thinking on). Render the provider
        // deltas for that state; `reasoning_start` reads the signals Dynamo forwards in
        // `extra_args`.
        let mut renderer = render::renderer_for(
            self.config.parser_family,
            render::reasoning_start(self.config.parser_family, request.extra_args.as_ref()),
        );
        let mut retokenizer = Retokenizer::with_shared(self.tokenizer.clone());
        // Every output chunk carries the served-by tag so downstream accounting
        // can separate provider spend from hosted spend.
        let served_by = metrics::served_by(&self.config);

        // Count the provider round-trip as in flight, and release the gauge when
        // the returned stream is dropped (including on a client disconnect).
        let inflight = metrics.as_ref().map(|metrics| metrics.inflight_guard());

        let body = self.client.build_body(&original);
        let chunks = match self.client.stream_chat(body).await {
            Ok(chunks) => chunks,
            Err(err) => {
                let retry_elsewhere = err.retry_elsewhere();
                record_terminal(&metrics, started, outcome_for_upstream(&err), None, None);
                return Err(map_upstream_error(&err, retry_elsewhere, false));
            }
        };

        let stream = async_stream::stream! {
            let _inflight = inflight;
            let mut chunks = chunks;
            let mut produced = false;
            let mut first_token_at: Option<Instant> = None;
            let mut finish_reason: Option<String> = None;
            let mut provider_usage: Option<CompletionUsage> = None;
            let mut generated_tokens: u32 = 0;

            loop {
                let next = tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        let usage = dynamo_backend_common::usage(prompt_tokens, generated_tokens);
                        record_terminal(
                            &metrics, started, Outcome::Cancelled, first_token_at, Some(&usage),
                        );
                        yield Ok(stamp_served_by(LLMEngineOutput::cancelled(), &served_by));
                        break;
                    }
                    _ = ctx.killed() => {
                        let usage = dynamo_backend_common::usage(prompt_tokens, generated_tokens);
                        record_terminal(
                            &metrics, started, Outcome::Cancelled, first_token_at, Some(&usage),
                        );
                        yield Ok(stamp_served_by(LLMEngineOutput::cancelled(), &served_by));
                        break;
                    }
                    next = chunks.next() => next,
                };

                let chunk = match next {
                    None => {
                        let text = match renderer.finish(finish_reason.as_deref()) {
                            Ok(text) => text,
                            Err(err) => {
                                record_terminal(
                                    &metrics, started, Outcome::StreamBroken, first_token_at, None,
                                );
                                yield Err(render_error(err, produced));
                                break;
                            }
                        };
                        if finish_reason.is_none() {
                            // The provider stopped without saying why; treat it
                            // as an incomplete stream so the frontend retries.
                            let err = UpstreamError::StreamBroken(
                                "provider stream ended without a finish_reason".to_string(),
                            );
                            record_terminal(
                                &metrics, started, Outcome::StreamBroken, first_token_at, None,
                            );
                            yield Err(map_upstream_error(&err, true, produced));
                            break;
                        }
                        let mut ids = retokenizer.push(&text);
                        ids.extend(retokenizer.finish());
                        generated_tokens = generated_tokens.saturating_add(ids.len() as u32);
                        let usage = provider_usage
                            .unwrap_or_else(|| dynamo_backend_common::usage(prompt_tokens, generated_tokens));
                        let reason = finish_reason_from(finish_reason.as_deref());
                        record_terminal(
                            &metrics,
                            started,
                            outcome_for_finish(&reason),
                            first_token_at,
                            Some(&usage),
                        );
                        yield Ok(stamp_served_by(terminal(reason, text, ids, usage), &served_by));
                        break;
                    }
                    Some(Err(err)) => {
                        let retry_elsewhere = err.retry_elsewhere();
                        record_terminal(
                            &metrics,
                            started,
                            outcome_for_upstream(&err),
                            first_token_at,
                            None,
                        );
                        yield Err(map_upstream_error(&err, retry_elsewhere, produced));
                        break;
                    }
                    Some(Ok(chunk)) => chunk,
                };

                if let Some(usage) = chunk.get("usage").and_then(parse_usage) {
                    provider_usage = Some(usage);
                }

                let choice = chunk.get("choices").and_then(|choices| choices.get(0));
                if let Some(reason) = choice
                    .and_then(|choice| choice.get("finish_reason"))
                    .and_then(Value::as_str)
                {
                    finish_reason = Some(reason.to_string());
                }

                let Some(delta) = choice.and_then(|choice| choice.get("delta")) else {
                    continue;
                };
                let text = match renderer.push_delta(delta) {
                    Ok(text) => text,
                    Err(err) => {
                        record_terminal(
                            &metrics, started, Outcome::StreamBroken, first_token_at, None,
                        );
                        yield Err(render_error(err, produced));
                        break;
                    }
                };
                if text.is_empty() {
                    continue;
                }
                produced = true;
                if first_token_at.is_none() {
                    first_token_at = Some(Instant::now());
                }
                let ids = retokenizer.push(&text);
                generated_tokens = generated_tokens.saturating_add(ids.len() as u32);
                yield Ok(stamp_served_by(text_chunk(text, ids), &served_by));
            }
        };
        Ok(Box::pin(stream))
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let state = self.state.clone();
        Ok(vec![KvEventSource::Push {
            dp_rank: self.config.dp_rank,
            on_ready: Box::new(move |publisher| {
                state.events.set(publisher);
                Ok(())
            }),
        }])
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        Ok(Some(registration::health_check_payload(&self.config)))
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        let metrics = ProxyMetrics::new(ctx.metrics, &self.config.tier, &self.config.provider.name)
            .map_err(|err| backend_error(BackendError::Unknown, err.to_string()))?;
        // Seed the gauge so the scrape reflects a cold cache before traffic.
        let blocks = self
            .state
            .vcache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len_blocks();
        metrics.set_vcache_blocks(blocks);
        *self.state.metrics.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(metrics));
        Ok(MetricsBindings::default())
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        if let Some(handle) = self
            .state
            .expire_task
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            handle.abort();
        }
        let (events, blocks) = {
            let mut cache = self.state.vcache.lock().unwrap_or_else(|e| e.into_inner());
            let events = cache.clear();
            (events, cache.len_blocks())
        };
        self.record_cache(events, blocks);
        Ok(())
    }
}

/// Load the tokenizer for the configured model. A path already on disk (a model
/// directory or a `tokenizer.json`) is used directly; anything else is a hub id
/// and is resolved like the worker's own model card, from the offline cache.
async fn load_tokenizer(model_path: &str) -> anyhow::Result<tokenizers::Tokenizer> {
    let path = Path::new(model_path);
    if path.is_file() {
        return tokenizers::Tokenizer::from_file(path)
            .map_err(|err| anyhow::anyhow!("load tokenizer {}: {err}", path.display()));
    }
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        dynamo_llm::local_model::LocalModel::fetch(model_path, true)
            .await
            .map_err(|err| anyhow::anyhow!("resolve model '{model_path}': {err}"))?
    };
    let tokenizer_json = dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tokenizer_json)
        .map_err(|err| anyhow::anyhow!("load tokenizer {}: {err}", tokenizer_json.display()))
}

/// Expire virtual-cache blocks four times per TTL, at least once a second.
fn expire_period(config: &ProxyConfig) -> Duration {
    (Duration::from_secs(config.vcache_ttl_secs) / 4).max(Duration::from_secs(1))
}

/// Hash inputs the router uses beyond the tokens. The proxy registers with
/// `enable_eagle = false`, so only the routing hints matter.
fn hash_options(request: &PreprocessedRequest) -> HashOptions {
    let routing = request.routing.as_ref();
    HashOptions {
        lora_name: routing.and_then(|routing| routing.lora_name.clone()),
        cache_salt: routing.and_then(|routing| routing.cache_namespace.clone()),
        is_eagle: false,
    }
}

/// Map a provider failure to the request-outcome label.
pub fn outcome_for_upstream(err: &UpstreamError) -> Outcome {
    match err {
        UpstreamError::RateLimited { .. } => Outcome::RateLimited,
        UpstreamError::Unavailable { .. } => Outcome::Unavailable,
        UpstreamError::Rejected { .. } => Outcome::Rejected,
        UpstreamError::Transport(_) => Outcome::Transport,
        UpstreamError::StreamBroken(_) | UpstreamError::InStream(_) => Outcome::StreamBroken,
    }
}

/// Map the terminal finish reason to the request-outcome label. An error
/// reason reaches the client as an error, and a provider-reported cancellation
/// is a cancellation, not a success.
pub fn outcome_for_finish(reason: &FinishReason) -> Outcome {
    match reason {
        FinishReason::Error(_) => Outcome::StreamBroken,
        FinishReason::Cancelled => Outcome::Cancelled,
        _ => Outcome::Ok,
    }
}

/// Attach the served-by tag to an output chunk as `engine_data`, which the
/// frontend copies into `nvext.engine_data` for callers that request it.
pub fn stamp_served_by(mut output: LLMEngineOutput, served_by: &Value) -> LLMEngineOutput {
    output.engine_data = Some(served_by.clone());
    output
}

/// A non-terminal chunk carrying provider text and the ids it retokenized to.
pub fn text_chunk(text: String, token_ids: Vec<u32>) -> LLMEngineOutput {
    LLMEngineOutput {
        text: Some(text),
        token_ids,
        ..LLMEngineOutput::default()
    }
}

/// The single terminal chunk: finish reason, trailing text and usage.
pub fn terminal(
    reason: FinishReason,
    text: String,
    token_ids: Vec<u32>,
    usage: CompletionUsage,
) -> LLMEngineOutput {
    LLMEngineOutput {
        text: (!text.is_empty()).then_some(text),
        token_ids,
        finish_reason: Some(reason),
        completion_usage: Some(usage),
        ..LLMEngineOutput::default()
    }
}

/// Map the provider's `finish_reason` onto Dynamo's. `tool_calls` and unknown
/// strings are normal stops: Dynamo derives tool calls from the rendered text.
pub fn finish_reason_from(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("cancelled") | Some("abort") => FinishReason::Cancelled,
        Some("error") => FinishReason::Error("provider reported an error".to_string()),
        _ => FinishReason::Stop,
    }
}

/// Parse the provider's usage object. `None` when the value is not an object,
/// so a stray non-object `usage` leaves the computed fallback in place.
pub fn parse_usage(value: &Value) -> Option<CompletionUsage> {
    let object = value.as_object()?;
    let field = |name: &str| -> u32 {
        object
            .get(name)
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
    };
    Some(CompletionUsage {
        prompt_tokens: field("prompt_tokens"),
        completion_tokens: field("completion_tokens"),
        total_tokens: field("total_tokens"),
        prompt_tokens_details: None,
        completion_tokens_details: None,
    })
}

/// Map a provider failure to the framework error.
///
/// `retry_elsewhere` is `UpstreamError::retry_elsewhere()` — passed in so the
/// mapping stays testable while that classifier is still another task's stub.
/// `output_started` says whether a chunk was already emitted: a failure after
/// that must be `StreamIncomplete` so the frontend's migration retry can resume
/// from the tokens already delivered. `Rejected` is never migrated: retrying
/// the same request elsewhere cannot fix a bad request.
pub fn map_upstream_error(
    err: &UpstreamError,
    retry_elsewhere: bool,
    output_started: bool,
) -> DynamoError {
    let message = err.to_string();
    if !retry_elsewhere {
        return backend_error(BackendError::InvalidArgument, message);
    }
    if output_started {
        return backend_error(BackendError::StreamIncomplete, message);
    }
    let class = match err {
        UpstreamError::RateLimited { .. } => ErrorType::WorkerOverloaded,
        UpstreamError::Unavailable { .. } => ErrorType::Backend(BackendError::EngineShutdown),
        UpstreamError::Transport(_) => ErrorType::Backend(BackendError::CannotConnect),
        UpstreamError::StreamBroken(_) | UpstreamError::InStream(_) => {
            ErrorType::Backend(BackendError::StreamIncomplete)
        }
        // Unreachable while RetryElsewhere is false for Rejected; kept for
        // exhaustiveness so the mapping stays explicit.
        UpstreamError::Rejected { .. } => ErrorType::Backend(BackendError::InvalidArgument),
    };
    error(class, message)
}

/// A render failure means the provider sent a delta our renderer cannot turn
/// back into model-format text. Before any output that is a bad response; after
/// output it is an incomplete stream the frontend can retry.
fn render_error(err: RenderError, output_started: bool) -> DynamoError {
    let kind = if output_started {
        BackendError::StreamIncomplete
    } else {
        BackendError::InvalidArgument
    };
    backend_error(kind, err.to_string())
}

fn client_error(message: impl Into<String>) -> DynamoError {
    backend_error(BackendError::InvalidArgument, message.into())
}

fn backend_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    error(ErrorType::Backend(kind), message.into())
}

fn error(class: ErrorType, message: String) -> DynamoError {
    DynamoError::builder()
        .error_type(class)
        .message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_chunk_carries_text_and_ids() {
        let chunk = text_chunk("hi".to_string(), vec![7, 8]);
        assert_eq!(chunk.text.as_deref(), Some("hi"));
        assert_eq!(chunk.token_ids, vec![7, 8]);
        assert!(chunk.finish_reason.is_none());
        assert!(chunk.completion_usage.is_none());
    }

    #[test]
    fn terminal_carries_reason_and_usage() {
        let usage = dynamo_backend_common::usage(5, 3);
        let chunk = terminal(FinishReason::Length, "end".to_string(), vec![9], usage);
        assert_eq!(chunk.finish_reason, Some(FinishReason::Length));
        assert_eq!(chunk.text.as_deref(), Some("end"));
        let usage = chunk.completion_usage.expect("usage carried");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 8);
    }

    #[test]
    fn terminal_with_empty_text_reports_no_text() {
        let chunk = terminal(
            FinishReason::Stop,
            String::new(),
            vec![],
            dynamo_backend_common::usage(1, 0),
        );
        assert!(chunk.text.is_none());
    }

    #[test]
    fn finish_reasons_map_to_dynamo() {
        assert_eq!(finish_reason_from(Some("length")), FinishReason::Length);
        assert_eq!(
            finish_reason_from(Some("content_filter")),
            FinishReason::ContentFilter
        );
        assert_eq!(
            finish_reason_from(Some("cancelled")),
            FinishReason::Cancelled
        );
        assert_eq!(finish_reason_from(Some("stop")), FinishReason::Stop);
        assert_eq!(finish_reason_from(Some("tool_calls")), FinishReason::Stop);
        assert_eq!(finish_reason_from(None), FinishReason::Stop);
        assert!(matches!(
            finish_reason_from(Some("error")),
            FinishReason::Error(_)
        ));
    }

    #[test]
    fn usage_parses_provider_numbers() {
        let value = serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        });
        let usage = parse_usage(&value).expect("object parses");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.total_tokens, 14);
    }

    #[test]
    fn usage_ignores_non_objects() {
        assert!(parse_usage(&serde_json::json!(null)).is_none());
        assert!(parse_usage(&serde_json::json!([1, 2])).is_none());
    }

    #[test]
    fn rejected_maps_to_a_non_retryable_client_error() {
        let err = UpstreamError::Rejected {
            status: 400,
            message: "bad request".to_string(),
        };
        // Even though the caller passes a retryable flag, `Rejected` stays a
        // client error because the classifier returns false for it.
        let mapped = map_upstream_error(&err, false, false);
        assert_eq!(
            mapped.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn transport_error_before_output_maps_to_migratable_connect_error() {
        let err = UpstreamError::Transport("connection reset".to_string());
        let mapped = map_upstream_error(&err, true, false);
        assert_eq!(
            mapped.error_type(),
            ErrorType::Backend(BackendError::CannotConnect)
        );
    }

    #[test]
    fn rate_limit_maps_to_worker_overloaded() {
        let err = UpstreamError::RateLimited {
            retry_after_ms: Some(500),
        };
        let mapped = map_upstream_error(&err, true, false);
        assert_eq!(mapped.error_type(), ErrorType::WorkerOverloaded);
    }

    #[test]
    fn failure_after_output_maps_to_stream_incomplete() {
        let err = UpstreamError::StreamBroken("no DONE".to_string());
        let mapped = map_upstream_error(&err, true, true);
        assert_eq!(
            mapped.error_type(),
            ErrorType::Backend(BackendError::StreamIncomplete)
        );
    }

    #[test]
    fn unavailable_before_output_maps_to_engine_shutdown() {
        let err = UpstreamError::Unavailable { status: 529 };
        let mapped = map_upstream_error(&err, true, false);
        assert_eq!(
            mapped.error_type(),
            ErrorType::Backend(BackendError::EngineShutdown)
        );
    }

    #[test]
    fn served_by_is_attached_to_terminal_output() {
        let tag = serde_json::json!({"served_by": "openrouter", "tier": "spillover"});
        let output = stamp_served_by(
            terminal(
                FinishReason::Stop,
                "hi".to_string(),
                vec![1],
                dynamo_backend_common::usage(2, 1),
            ),
            &tag,
        );
        let data = output.engine_data.expect("engine_data attached");
        assert_eq!(data["served_by"], "openrouter");
        assert_eq!(data["tier"], "spillover");
        // Stamping does not disturb the rest of the terminal chunk.
        assert_eq!(output.finish_reason, Some(FinishReason::Stop));
        assert_eq!(output.text.as_deref(), Some("hi"));
    }

    #[test]
    fn upstream_errors_map_to_outcome_labels() {
        assert_eq!(
            outcome_for_upstream(&UpstreamError::RateLimited {
                retry_after_ms: None
            }),
            Outcome::RateLimited
        );
        assert_eq!(
            outcome_for_upstream(&UpstreamError::Unavailable { status: 503 }),
            Outcome::Unavailable
        );
        assert_eq!(
            outcome_for_upstream(&UpstreamError::Rejected {
                status: 400,
                message: "bad".to_string(),
            }),
            Outcome::Rejected
        );
        assert_eq!(
            outcome_for_upstream(&UpstreamError::Transport("reset".to_string())),
            Outcome::Transport
        );
        assert_eq!(
            outcome_for_upstream(&UpstreamError::StreamBroken("eof".to_string())),
            Outcome::StreamBroken
        );
        assert_eq!(
            outcome_for_upstream(&UpstreamError::InStream("bad frame".to_string())),
            Outcome::StreamBroken
        );
    }

    #[test]
    fn finish_reasons_map_to_outcome_labels() {
        assert_eq!(outcome_for_finish(&FinishReason::Stop), Outcome::Ok);
        assert_eq!(outcome_for_finish(&FinishReason::Length), Outcome::Ok);
        assert_eq!(
            outcome_for_finish(&FinishReason::Cancelled),
            Outcome::Cancelled
        );
        assert_eq!(
            outcome_for_finish(&FinishReason::Error("boom".to_string())),
            Outcome::StreamBroken
        );
    }
}
