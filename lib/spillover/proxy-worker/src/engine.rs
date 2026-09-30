// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The proxy engine: an [`LLMEngine`] that answers every Dynamo request from a
//! third-party OpenAI-compatible provider.
//!
//! The request arrives as a `PreprocessedRequest` (token ids, sampling options)
//! plus the client's chat request, which the frontend's KV router puts in `extra_args` because
//! this worker advertises the `chat_request` capability (see `dw_proxy_core::chat_request`).
//! We rebuild the provider body, stream the provider's
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
use dw_proxy_core::chat_request;
use dw_proxy_core::config::ProxyConfig;
use dw_proxy_core::errors::UpstreamError;
use dw_proxy_core::render::{self, RenderError};
use dw_proxy_core::retokenize::Retokenizer;
use dw_proxy_core::thinking::{ThinkingDialect, ThinkingIntent, ThinkingMode};
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
use crate::metrics::{self, Outcome, ProxyMetrics, ThinkingEvent, record_terminal};
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
    /// Where the frontend's reasoning parser starts for this request. The prompt tokens are the
    /// frontend's rendered prompt (the proxy refuses migration replays, so nothing is appended),
    /// so decoding their tail and applying the frontend's own rule gives the exact answer; the
    /// `extra_args` signals are only a fallback if the tail cannot be decoded.
    fn reasoning_start(&self, request: &PreprocessedRequest) -> render::ReasoningStart {
        const TAIL_TOKENS: usize = 32;
        let ids = request.token_ids.as_ref();
        let tail = &ids[ids.len().saturating_sub(TAIL_TOKENS)..];
        match self.tokenizer.decode(tail, false) {
            Ok(text) => render::reasoning_start_from_prompt(self.config.parser_family, &text),
            Err(error) => {
                tracing::warn!(%error, "could not decode the prompt tail; using forwarded reasoning signals");
                render::reasoning_start(self.config.parser_family, request.extra_args.as_ref())
            }
        }
    }

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
        on_request_and_publish(
            &self.state.vcache,
            prompt_tokens,
            options,
            |events, blocks| {
                self.record_cache(events, blocks);
            },
        );
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
        // `start` is contracted to run once, but guard against a second call:
        // dropping the old `JoinHandle` would leak the task and its ticker.
        let mut slot = self
            .state
            .expire_task
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return;
        }
        let state = self.state.clone();
        let period = expire_period(&self.config);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(period);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                // Publish before releasing the cache lock, for the same
                // parent-before-child ordering reason as `on_request`.
                let (published, blocks) = {
                    let mut cache = state.vcache.lock().unwrap_or_else(|e| e.into_inner());
                    let events = cache.expire(Instant::now());
                    let blocks = cache.len_blocks();
                    (state.events.publish(events), blocks)
                };
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
        *slot = Some(handle);
    }
}

/// Apply a prompt to the virtual cache and publish its events *before*
/// releasing the cache lock. The router's radix indexer discards a `Stored`
/// whose parent block has not been indexed yet, so a concurrent request must
/// not be able to publish a child ahead of the request that published its
/// parent. Publishing under the same lock makes cache mutation order and
/// event publication order identical.
fn on_request_and_publish(
    cache: &Mutex<VirtualCache>,
    prompt_tokens: &[u32],
    options: &HashOptions,
    publish: impl FnOnce(Vec<dw_proxy_core::vcache::CacheEvent>, usize),
) {
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    let events = guard.on_request(prompt_tokens, options, Instant::now());
    let blocks = guard.len_blocks();
    publish(events, blocks);
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

        // The frontend attaches the chat request only for chat requests routed by the KV router;
        // without it the proxy has nothing to send. A migration retry is refused outright: the
        // proxy has no assistant prefix to continue from, so it must fail over to a hosted worker.
        // Every refusal is migratable ([`ErrorType::WorkerOverloaded`]) so the router retries.
        let metrics = self.metrics();
        let started = Instant::now();
        // Owned so the `'static` response stream does not borrow `self` for logging.
        let provider = self.config.provider.name.clone();
        let provider_config = self.client.config();
        let original = match admit(
            request.extra_args.as_ref(),
            provider_config.thinking_dialect,
            provider_config.thinking_strict,
        ) {
            Ok(original) => original,
            Err((outcome, err)) => {
                record_terminal(&metrics, started, outcome, None, None);
                tracing::warn!(
                    provider = %provider,
                    outcome = outcome.as_str(),
                    error = %err,
                    "refusing to serve a request the proxy cannot complete"
                );
                return Err(err);
            }
        };

        // What the provider is asked about thinking, for the two thinking metrics.
        let intent = ThinkingIntent::from_request(&original);
        let translation = provider_config.thinking_dialect.translate(&intent);
        if !translation.unexpressed.is_empty() {
            if let Some(metrics) = &metrics {
                metrics.record_thinking(ThinkingEvent::Unexpressed);
            }
            tracing::debug!(
                provider = %provider,
                unexpressed = ?translation.unexpressed,
                "the provider's thinking dialect cannot express the request's choice"
            );
        }
        // Thinking off was actually sent, so reasoning in the response means the provider
        // ignored it.
        let thinking_off_sent =
            intent.mode == Some(ThinkingMode::Disabled) && !translation.fields.is_empty();
        let mut ignored_recorded = false;

        let prompt_tokens = request.token_ids.as_ref().len() as u32;
        // Multimodal requests route on an MM-expanded token sequence with
        // per-block MM hashes; `HashOptions`/`VirtualCache` cannot reproduce
        // that hash. Recording them would publish block hashes the router never
        // looks up (no stickiness) or, worse, a false overlap. Skip
        // virtual-cache recording for those requests and let the router fall
        // back to token routing.
        if let Some(tokens) = vcache_prompt(&request) {
            let options = hash_options(&request);
            self.record_prompt(tokens, &options);
        }

        // Held-back-tail retokenization is per stream, so each request gets its own
        // `Retokenizer` over the shared tokenizer.
        // The frontend's parser starts inside reasoning for templates that end the
        // prompt with the opener (GLM/DeepSeek with thinking on). Render the provider
        // deltas for that state; `reasoning_start` reads the signals Dynamo forwards in
        // `extra_args`.
        let mut renderer =
            render::renderer_for(self.config.parser_family, self.reasoning_start(&request));
        let mut retokenizer = Retokenizer::with_shared(self.tokenizer.clone());
        // Every output chunk carries the served-by tag so downstream accounting
        // can separate provider spend from hosted spend.
        let served_by = metrics::served_by(&self.config);

        // Count the provider round-trip as in flight, and release the gauge when
        // the returned stream is dropped (including on a client disconnect).
        let inflight = metrics.as_ref().map(|metrics| metrics.inflight_guard());

        // `stop_conditions.max_tokens` is the frontend's authoritative cap: it is clamped to the
        // context window and reduced on each migration, so it, not the chat request's own value,
        // decides the provider's cap.
        let body = self
            .client
            .build_body(&original, request.stop_conditions.max_tokens);
        let chunks = match self.client.stream_chat(body).await {
            Ok(chunks) => chunks,
            Err(err) => {
                let retry_elsewhere = err.retry_elsewhere();
                tracing::warn!(
                    provider = %provider,
                    error = %err,
                    retry_elsewhere,
                    "provider request failed before the stream opened"
                );
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
            let mut provider_usage: Option<ProviderUsage> = None;
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

                // `None` means the response is complete: the provider closed
                // the body cleanly, or reported a `finish_reason` and then
                // failed the connection. In the latter case the generation is
                // already done, so the trailing transport error must not turn
                // it into a retry.
                let chunk = match next {
                    None => None,
                    Some(Err(err)) if finish_reason.is_some() => {
                        tracing::debug!(?err, "ignoring provider error after finish_reason");
                        None
                    }
                    Some(Err(err)) => {
                        let retry_elsewhere = err.retry_elsewhere();
                        tracing::warn!(
                            provider = %provider,
                            error = %err,
                            retry_elsewhere,
                            output_started = produced,
                            "provider stream failed"
                        );
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
                    Some(Ok(chunk)) => Some(chunk),
                };

                let Some(chunk) = chunk else {
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
                    // Content the renderer held back until `finish` still has
                    // a first token time.
                    if first_token_at.is_none() && !text.is_empty() {
                        first_token_at = Some(Instant::now());
                    }
                    let fallback = dynamo_backend_common::usage(prompt_tokens, generated_tokens);
                    let usage = match provider_usage {
                        Some(provider) => provider.over(fallback),
                        None => fallback,
                    };
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
                if thinking_off_sent && !ignored_recorded && has_reasoning(delta) {
                    ignored_recorded = true;
                    if let Some(metrics) = &metrics {
                        metrics.record_thinking(ThinkingEvent::Ignored);
                    }
                    tracing::warn!(
                        provider = %provider,
                        "provider returned reasoning although thinking was turned off"
                    );
                }
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

/// The token sequence to record in the virtual cache, or `None` when the
/// request must not be recorded.
///
/// A request with multimodal routing info is hashed by the router over the
/// MM-expanded `routing_token_ids` plus its per-block MM hashes. `HashOptions`
/// carries neither, so the proxy cannot compute the router's hashes for it;
/// recording the execution `token_ids` would publish hashes the router never
/// looks up (or a false prefix overlap). Such requests are skipped and routed
/// on tokens instead.
fn vcache_prompt(request: &PreprocessedRequest) -> Option<&[u32]> {
    if request.block_mm_routing_info().1.is_some() {
        return None;
    }
    Some(request.token_ids.as_slice())
}

/// `extra_args` key the frontend sets on a migration retry for a chat-request worker: the number
/// of output tokens already delivered to the client. Must equal the frontend's
/// `CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY`.
const CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY: &str = "chat_request_replayed_tokens";

/// The number of output tokens the client already received, when this request is a migration
/// retry. `None` for a fresh request. A present-but-malformed value is treated as a retry so a
/// proxy never regenerates over output the client already has.
fn replayed_tokens(extra_args: Option<&Value>) -> Option<u64> {
    let value = extra_args?.get(CHAT_REQUEST_REPLAYED_TOKENS_EXTRA_ARGS_KEY)?;
    match value.as_u64() {
        Some(0) => None,
        Some(tokens) => Some(tokens),
        None => {
            tracing::warn!(
                ?value,
                "chat_request_replayed_tokens is not a positive integer; refusing to serve"
            );
            Some(1)
        }
    }
}

/// Decide whether the proxy may serve the request, returning the chat request to forward.
///
/// A migration replay already delivered output to the client, and a missing chat request means
/// the router had not yet observed this worker's capability (or is not in KV mode). Neither is a
/// client error: both return a migratable error so the router retries on a worker that can
/// serve the request.
fn admit(
    extra_args: Option<&Value>,
    thinking_dialect: ThinkingDialect,
    thinking_strict: bool,
) -> Result<Value, (Outcome, DynamoError)> {
    if let Some(replayed) = replayed_tokens(extra_args) {
        return Err((
            Outcome::MigrationReplay,
            migratable_error(format!(
                "proxy cannot continue a partially generated completion ({replayed} tokens already \
                 delivered); retry on a hosted worker"
            )),
        ));
    }
    match chat_request::from_extra_args(extra_args) {
        Ok(Some(original)) => {
            if thinking_strict
                && let Some(choice) = thinking_dialect
                    .translate(&ThinkingIntent::from_request(original))
                    .unexpressed
                    .first()
            {
                return Err((
                    Outcome::Unsupported,
                    migratable_error(format!(
                        "this provider's thinking dialect cannot express the request's {choice}"
                    )),
                ));
            }
            Ok(original.clone())
        }
        Ok(None) => Err((
            Outcome::NoChatRequest,
            migratable_error(
                "request has no chat request: the proxy serves chat completions routed by the KV \
                 router",
            ),
        )),
        Err(
            err @ (chat_request::ChatRequestError::UnsupportedField { .. }
            | chat_request::ChatRequestError::MultipleChoices { .. }),
        ) => Err((Outcome::Unsupported, migratable_error(err.to_string()))),
        Err(err) => Err((
            Outcome::NoChatRequest,
            migratable_error(format!("invalid chat request: {err}")),
        )),
    }
}

/// Whether a provider delta carries reasoning text, in either field name providers use.
fn has_reasoning(delta: &Value) -> bool {
    ["reasoning_content", "reasoning"].iter().any(|key| {
        delta
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
    })
}

/// Map a provider failure to the request-outcome label.
pub fn outcome_for_upstream(err: &UpstreamError) -> Outcome {
    match err {
        UpstreamError::RateLimited { .. } => Outcome::RateLimited,
        UpstreamError::Unavailable { .. } => Outcome::Unavailable,
        // A rejected key is the one provider failure that gets the worker reported down; it must
        // be distinguishable from an ordinary provider 4xx (a moderation 403 included).
        UpstreamError::Rejected { status: 401, .. } => Outcome::AuthError,
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

/// A provider `usage` object reduced to the fields it actually carried.
/// Streaming providers often send only part of the object; a missing field
/// must not overwrite the value the proxy computed locally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

impl ProviderUsage {
    /// Patch `fallback` with the fields the provider reported.
    pub fn over(self, fallback: CompletionUsage) -> CompletionUsage {
        CompletionUsage {
            prompt_tokens: self.prompt_tokens.unwrap_or(fallback.prompt_tokens),
            completion_tokens: self.completion_tokens.unwrap_or(fallback.completion_tokens),
            total_tokens: self.total_tokens.unwrap_or(fallback.total_tokens),
            ..fallback
        }
    }
}

/// Parse the provider's `usage` object into the fields it actually carried.
/// `None` when the value is not an object, so a stray non-object `usage` leaves
/// the computed fallback in place.
pub fn parse_usage(value: &Value) -> Option<ProviderUsage> {
    let object = value.as_object()?;
    let field = |name: &str| -> Option<u32> {
        object
            .get(name)
            .and_then(Value::as_u64)
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
    };
    Some(ProviderUsage {
        prompt_tokens: field("prompt_tokens"),
        completion_tokens: field("completion_tokens"),
        total_tokens: field("total_tokens"),
    })
}

/// Map a provider failure to the error the frontend sees.
///
/// The frontend reports a worker down (`report_instance_down`) for connection, shutdown and
/// incomplete-stream errors, which takes it out of routing. A proxy is only unusable when its
/// provider key is rejected (401), so that is the one case mapped to `EngineShutdown`. Every
/// other provider failure concerns one request or a transient provider condition, so it maps to
/// `WorkerOverloaded`: migratable, without quarantining the proxy. That includes provider 4xx
/// rejections such as a moderation 403 or a smaller provider context limit, which a hosted worker
/// may still serve; a request that is genuinely bad is then rejected there. After output has
/// started the migration layer replays the delivered tokens, and the retry cannot land back on a
/// proxy (it refuses replays), so the same mapping applies.
///
/// `retry_elsewhere` and `output_started` are accepted for the callers' logging and kept in the
/// signature so the mapping stays testable per case.
pub fn map_upstream_error(
    err: &UpstreamError,
    _retry_elsewhere: bool,
    _output_started: bool,
) -> DynamoError {
    let message = err.to_string();
    if matches!(err, UpstreamError::Rejected { status: 401, .. }) {
        return backend_error(BackendError::EngineShutdown, message);
    }
    migratable_error(message)
}

/// A render failure means the provider sent a delta our renderer cannot turn back into
/// model-format text. It is about this response, not the proxy, so it is retried elsewhere
/// without quarantining the worker.
fn render_error(err: RenderError, _output_started: bool) -> DynamoError {
    migratable_error(err.to_string())
}

/// A refusal the router is expected to retry on another worker.
fn migratable_error(message: impl Into<String>) -> DynamoError {
    error(ErrorType::WorkerOverloaded, message.into())
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
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(4));
        assert_eq!(usage.total_tokens, Some(14));
    }

    #[test]
    fn partial_provider_usage_preserves_the_computed_fallback() {
        // Only `total_tokens` is present (or the provider renamed the others).
        // A missing field must keep the locally computed value, not zero it.
        let value = serde_json::json!({ "total_tokens": 99 });
        let provider = parse_usage(&value).expect("object parses");
        assert_eq!(provider.prompt_tokens, None);
        assert_eq!(provider.completion_tokens, None);
        let fallback = dynamo_backend_common::usage(11, 7);
        let merged = provider.over(fallback);
        assert_eq!(merged.prompt_tokens, 11);
        assert_eq!(merged.completion_tokens, 7);
        assert_eq!(merged.total_tokens, 99);
    }

    #[test]
    fn usage_ignores_non_objects() {
        assert!(parse_usage(&serde_json::json!(null)).is_none());
        assert!(parse_usage(&serde_json::json!([1, 2])).is_none());
    }

    #[test]
    fn only_a_rejected_key_reports_the_proxy_down() {
        let key = UpstreamError::Rejected {
            status: 401,
            message: "bad api key".to_string(),
        };
        assert_eq!(
            map_upstream_error(&key, key.retry_elsewhere(), false).error_type(),
            ErrorType::Backend(BackendError::EngineShutdown)
        );
        // Everything else concerns one request or a transient provider condition: retried
        // elsewhere, and never an error type the frontend quarantines the instance for.
        let others = [
            UpstreamError::Rejected {
                status: 403,
                message: "flagged by moderation".to_string(),
            },
            UpstreamError::Rejected {
                status: 400,
                message: "context too long for this provider".to_string(),
            },
            UpstreamError::Transport("connection reset".to_string()),
            UpstreamError::StreamBroken("no DONE".to_string()),
            UpstreamError::RateLimited {
                retry_after_ms: Some(500),
            },
            UpstreamError::Unavailable { status: 529 },
        ];
        for err in others {
            for output_started in [false, true] {
                let mapped = map_upstream_error(&err, err.retry_elsewhere(), output_started);
                assert_eq!(
                    mapped.error_type(),
                    ErrorType::WorkerOverloaded,
                    "{err} (output_started = {output_started})"
                );
            }
        }
    }

    #[test]
    fn unavailable_before_output_maps_to_worker_overloaded() {
        // A transient provider 408/5xx is pressure on a live provider, not
        // evidence that the proxy worker died.
        let err = UpstreamError::Unavailable { status: 529 };
        let mapped = map_upstream_error(&err, true, false);
        assert_eq!(mapped.error_type(), ErrorType::WorkerOverloaded);
    }

    #[test]
    fn vcache_prompt_skips_multimodal_requests() {
        use dynamo_llm::protocols::common::preprocessor::MmRoutingInfo;

        let plain = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1u32, 2, 3])
            .sampling_options(Default::default())
            .output_options(Default::default())
            .stop_conditions(dynamo_backend_common::StopConditions::default())
            .build()
            .expect("build request");
        assert_eq!(vcache_prompt(&plain), Some([1u32, 2, 3].as_slice()));

        let mut mm = plain.clone();
        mm.mm_routing_info = Some(MmRoutingInfo {
            routing_token_ids: vec![1, 2, 3, 4, 5, 0, 0, 0],
            block_mm_infos: vec![None],
            expanded_prompt_len: 5,
        });
        assert_eq!(
            vcache_prompt(&mm),
            None,
            "multimodal routing hashes cannot be reproduced by HashOptions"
        );

        // Empty routing tokens mean the router falls back to the execution
        // sequence too, so recording is safe again.
        mm.mm_routing_info = Some(MmRoutingInfo {
            routing_token_ids: Vec::new(),
            block_mm_infos: Vec::new(),
            expanded_prompt_len: 0,
        });
        assert_eq!(vcache_prompt(&mm), Some([1u32, 2, 3].as_slice()));
    }

    #[test]
    fn cache_events_are_published_under_the_cache_lock() {
        let cache = Mutex::new(VirtualCache::new(VirtualCacheConfig {
            block_size: 4,
            ttl: Duration::from_secs(60),
            max_blocks: 16,
        }));
        let options = HashOptions::default();
        on_request_and_publish(&cache, &[1, 2, 3, 4], &options, |events, _blocks| {
            // The same (non-reentrant) mutex is held here if publication is
            // ordered with cache mutation. If the events were published after
            // the lock was released, this `try_lock` would succeed.
            assert!(
                cache.try_lock().is_err(),
                "cache events were published after the cache lock was released"
            );
            assert!(!events.is_empty(), "a full block should emit Stored");
        });
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
    fn migration_replay_is_refused_as_migratable() {
        let extra = serde_json::json!({
            "chat_request": {"messages": [{"role": "user", "content": "hi"}]},
            "chat_request_replayed_tokens": 7,
        });
        let (outcome, err) = admit(Some(&extra), ThinkingDialect::default(), false)
            .expect_err("a replay must not be served");
        assert_eq!(outcome, Outcome::MigrationReplay);
        assert_eq!(err.error_type(), ErrorType::WorkerOverloaded);
        assert!(
            err.to_string().contains('7'),
            "the message must name the replayed token count: {err}"
        );
    }

    #[test]
    fn missing_chat_request_is_migratable_not_a_client_error() {
        let extra = serde_json::json!({});
        let (outcome, err) = admit(Some(&extra), ThinkingDialect::default(), false)
            .expect_err("no chat request cannot be served");
        assert_eq!(outcome, Outcome::NoChatRequest);
        assert_eq!(err.error_type(), ErrorType::WorkerOverloaded);
    }

    #[test]
    fn fresh_chat_request_is_admitted() {
        let extra = serde_json::json!({"chat_request": {"messages": []}});
        assert!(admit(Some(&extra), ThinkingDialect::default(), false).is_ok());

        // A zero count is a fresh request, not a replay.
        let extra = serde_json::json!({
            "chat_request": {"messages": []},
            "chat_request_replayed_tokens": 0,
        });
        assert!(admit(Some(&extra), ThinkingDialect::default(), false).is_ok());
    }

    #[test]
    fn malformed_replay_marker_is_refused() {
        let extra = serde_json::json!({
            "chat_request": {"messages": []},
            "chat_request_replayed_tokens": "many",
        });
        let (outcome, err) = admit(Some(&extra), ThinkingDialect::default(), false)
            .expect_err("an unparseable marker is unsafe");
        assert_eq!(outcome, Outcome::MigrationReplay);
        assert_eq!(err.error_type(), ErrorType::WorkerOverloaded);
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
            outcome_for_upstream(&UpstreamError::Rejected {
                status: 401,
                message: "bad key".to_string(),
            }),
            Outcome::AuthError,
            "a rejected credential is a provider-side fault, not a client 4xx"
        );
        assert_eq!(
            outcome_for_upstream(&UpstreamError::Rejected {
                status: 403,
                message: "forbidden".to_string(),
            }),
            Outcome::Rejected,
            "a 403 is often request-scoped (moderation), not a key fault"
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
    #[test]
    fn an_unexpressible_thinking_choice_is_retried_elsewhere_only_when_strict() {
        let extra = serde_json::json!({"chat_request": {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_args": {"thinking": true, "enable_thinking": true}
        }});
        // `reasoning_effort` cannot say "on" without a grade: sent as-is unless strict.
        assert!(admit(Some(&extra), ThinkingDialect::ReasoningEffort, false).is_ok());
        let (outcome, err) = admit(Some(&extra), ThinkingDialect::ReasoningEffort, true)
            .expect_err("thinking on is unexpressed");
        assert_eq!(outcome, Outcome::Unsupported);
        assert_eq!(err.error_type(), ErrorType::WorkerOverloaded);
        // A dialect that can say it is served even when strict.
        assert!(admit(Some(&extra), ThinkingDialect::ReasoningObject, true).is_ok());
    }

    #[test]
    fn reasoning_is_detected_in_either_field() {
        assert!(has_reasoning(
            &serde_json::json!({"reasoning_content": "x"})
        ));
        assert!(has_reasoning(&serde_json::json!({"reasoning": "x"})));
        assert!(!has_reasoning(&serde_json::json!({"reasoning": ""})));
        assert!(!has_reasoning(&serde_json::json!({"content": "x"})));
    }
}
