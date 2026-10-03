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
//! text, and yield `LLMEngineOutput` chunks. The frontend sees a normal primary
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
use dw_proxy_core::circuit_breaker::{Admission, CircuitBreaker, CircuitHealth, Transition};
use dw_proxy_core::config::ProxyConfig;
use dw_proxy_core::errors::UpstreamError;
use dw_proxy_core::render::{self, RenderError};
use dw_proxy_core::retokenize::{ModelTokenizer, Retokenizer};
use dw_proxy_core::thinking::{ThinkingDialect, ThinkingIntent, ThinkingMode};
use dw_proxy_core::upstream::UpstreamClient;
use dw_proxy_core::vcache::{HashOptions, VirtualCache, VirtualCacheConfig};
use dynamo_backend_common::{
    BackendError, CompletionUsage, DynamoError, EngineConfig, ErrorType, FinishReason,
    GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput, MetricsBindings, MetricsCtx,
    PreprocessedRequest,
};
use dynamo_llm::model_card::TokenizerKind;
use dynamo_llm::tokenizers::TikTokenTokenizer;
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
    /// Per-proxy circuit breaker. One proxy fronts exactly one provider, so a
    /// single breaker is the whole scope.
    circuit: Mutex<CircuitBreaker>,
}

impl EngineState {
    /// Fold one terminal request outcome into the circuit breaker, update its
    /// gauge and log a state change once (open at `warn`, close at `info`).
    /// Outcomes that say nothing about the provider are ignored here; the
    /// caller still counts them in `proxy_requests_total`.
    fn record_circuit(&self, admission: Admission, outcome: Outcome) {
        // No provider verdict: nothing to record. A probe that ends this way is handed back by
        // its `ProbeGuard` when the request's stream is dropped.
        let Some(health) = circuit_health(outcome) else {
            return;
        };
        let (transition, open) = {
            let mut circuit = self.circuit.lock().unwrap_or_else(|e| e.into_inner());
            let transition = circuit.record(Instant::now(), admission, health);
            (transition, circuit.is_open())
        };
        if let Some(metrics) = self
            .metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            metrics.set_circuit_open(open);
        }
        match transition {
            Some(Transition::Opened {
                failures,
                cooldown_ms,
            }) => tracing::warn!(
                failures,
                cooldown_ms,
                "provider circuit opened after consecutive failures; refusing requests until the cooldown elapses"
            ),
            Some(Transition::Closed) => {
                tracing::info!("provider circuit closed after a successful probe")
            }
            None => {}
        }
    }
}

/// A Dynamo engine backed by one third-party provider.
pub struct ProxyEngine {
    config: Arc<ProxyConfig>,
    client: UpstreamClient,
    /// The model's tokenizer, loaded once and shared by every stream's `Retokenizer`.
    tokenizer: ModelTokenizer,
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
    /// `model_dir` is the pinned revision's snapshot, when `model_revision` is set.
    pub async fn new(config: ProxyConfig, model_dir: Option<&Path>) -> anyhow::Result<Self> {
        let client = UpstreamClient::new(config.provider.clone())?;
        let vcache = VirtualCache::new(VirtualCacheConfig {
            block_size: config.kv_block_size,
            ttl: Duration::from_secs(config.vcache_ttl_secs),
            max_blocks: config.vcache_max_blocks,
        });
        let state = Arc::new(EngineState {
            vcache: Mutex::new(vcache),
            // KV routing on: buffer until `kv_event_sources` installs the publisher. Routing off:
            // no publisher will ever come, so drop events instead of retaining them to the cap.
            events: if config.router_config.is_some() {
                EventSink::new(config.dp_rank)
            } else {
                EventSink::with_buffering(config.dp_rank, false)
            },
            expire_task: Mutex::new(None),
            metrics: Mutex::new(None),
            circuit: Mutex::new(CircuitBreaker::new(
                config.provider.circuit_breaker.unwrap_or_default(),
            )),
        });
        let model = model_dir.map_or_else(
            || config.model_path.clone(),
            |dir| dir.display().to_string(),
        );
        let tokenizer = load_tokenizer(&model).await?;
        Ok(Self {
            config: Arc::new(config),
            client,
            tokenizer,
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
            metrics.add_kv_events_dropped(published.dropped);
            metrics.set_vcache_blocks(blocks);
        }
    }

    /// Start the task that expires virtual-cache blocks past their TTL, publishing the removals.
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
                    metrics.add_kv_events_dropped(published.dropped);
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
        // proxy has no assistant prefix to continue from, so it must fail over to a primary worker.
        // These refusals are migratable ([`ErrorType::WorkerOverloaded`]) so the router retries.
        let metrics = self.metrics();
        let started = Instant::now();
        // Owned so the `'static` response stream does not borrow `self` for logging.
        let provider = self.config.provider.name.clone();

        // A request that does not fit the context window is the client's error on every worker,
        // so the proxy answers it as a primary does, before the breaker or the provider. The
        // frontend normally refuses it first, from the token budget the card publishes
        // (`registration::token_budget`); this covers a frontend whose preprocessor was built
        // from a card without one. The
        // refusal is the stream's first item, where a primary's engine reports it: an error
        // returned before the stream reaches the frontend as a pre-stream failure, which the
        // router migrates and for which it reports the worker down.
        if let Some(message) = context_overflow(
            request.token_ids.len(),
            request.stop_conditions.max_tokens,
            self.config.context_length,
        ) {
            record_terminal(&metrics, started, Outcome::ContextOverflow, None, None);
            tracing::debug!(provider = %provider, "refusing a request that exceeds the context length");
            let refusal = backend_error(BackendError::InvalidArgument, message);
            return Ok(Box::pin(futures::stream::iter([Err(refusal)])));
        }

        // Per-proxy circuit breaker: refuse before touching the provider while
        // the breaker is open, so an outage costs no provider round-trip. The
        // refusal is migratable, so the router sends the request to a worker
        // that can still serve it. One request is let through once the cooldown
        // elapses (half-open probe); concurrent requests see `Refused`.
        let (admission, probe_guard) = {
            let mut circuit = self.state.circuit.lock().unwrap_or_else(|e| e.into_inner());
            let admission = circuit.admit(Instant::now());
            if let Some(metrics) = &metrics {
                metrics.set_circuit_open(circuit.is_open());
            }
            let guard = (admission == Admission::Probe).then(|| ProbeGuard {
                state: self.state.clone(),
                generation: circuit.probe_generation(),
            });
            (admission, guard)
        };
        if admission == Admission::Refused {
            record_terminal(&metrics, started, Outcome::CircuitOpen, None, None);
            tracing::debug!(
                provider = %provider,
                "refusing request because the provider circuit is open"
            );
            return Err(migratable_error(
                "provider circuit is open after repeated failures; retry on another worker",
            ));
        }
        let state = self.state.clone();

        let provider_config = self.client.config();
        let original = match admit(
            request.extra_args.as_ref(),
            provider_config.thinking_dialect,
            provider_config.thinking_strict,
        ) {
            Ok(original) => original,
            Err((outcome, err)) => {
                record_terminal_with_circuit(
                    &state, &metrics, started, admission, outcome, None, None,
                );
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
        // Held-back-tail retokenization is per stream, so each request gets its own
        // `Retokenizer` over the shared tokenizer.
        // The frontend's parser starts inside reasoning for templates that end the
        // prompt with the opener (GLM/DeepSeek with thinking on). Render the provider
        // deltas for that state; `reasoning_start` reads the signals Dynamo forwards in
        // `extra_args`.
        let mut renderer =
            render::renderer_for(self.config.parser_family, self.reasoning_start(&request));
        let mut retokenizer = Retokenizer::with_model(self.tokenizer.clone());
        // Every output chunk carries the served-by tag so downstream accounting
        // can separate provider spend from primary spend.
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
                record_terminal_with_circuit(
                    &state,
                    &metrics,
                    started,
                    admission,
                    outcome_for_upstream(&err),
                    None,
                    None,
                );
                let mapped = map_upstream_error(&err, retry_elsewhere, false);
                // A request the provider cannot serve is answered as a primary answers an invalid
                // request: as the stream's first item, since an error returned from `generate` is
                // a pre-stream failure the router migrates.
                if mapped.error_type() == ErrorType::Backend(BackendError::InvalidArgument) {
                    return Ok(Box::pin(futures::stream::iter([Err(mapped)])));
                }
                return Err(mapped);
            }
        };

        // Advertise the prompt in the virtual cache only once the provider has accepted the
        // request (`stream_chat` returns after the first stream event): a request refused or
        // failed before that never materialized a provider cache, and advertising it would
        // attract affinity for the whole TTL. Multimodal requests route on an MM-expanded token
        // sequence with per-block MM hashes that `HashOptions`/`VirtualCache` cannot reproduce,
        // so they are not recorded; the router falls back to token routing for them.
        if let Some(tokens) = vcache_prompt(&request) {
            let options = hash_options(&request, self.config.enable_eagle);
            self.record_prompt(tokens, &options);
        }

        let stream = async_stream::stream! {
            let _inflight = inflight;
            // Held for the stream's lifetime: hands the probe back if the stream ends or is
            // dropped without a provider verdict.
            let _probe_guard = probe_guard;
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
                        // Only the provider's own usage feeds the provider-billed metrics; the
                        // locally counted fallback is not what the provider billed.
                        let billed = provider_billed(
                            dynamo_backend_common::usage(prompt_tokens, generated_tokens),
                            provider_usage,
                        );
                        record_terminal_with_circuit(
                            &state, &metrics, started, admission, Outcome::Cancelled,
                            first_token_at, billed.as_ref(),
                        );
                        yield Ok(stamp_served_by(LLMEngineOutput::cancelled(), &served_by));
                        break;
                    }
                    _ = ctx.killed() => {
                        let billed = provider_billed(
                            dynamo_backend_common::usage(prompt_tokens, generated_tokens),
                            provider_usage,
                        );
                        record_terminal_with_circuit(
                            &state, &metrics, started, admission, Outcome::Cancelled,
                            first_token_at, billed.as_ref(),
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
                        record_terminal_with_circuit(
                            &state,
                            &metrics,
                            started,
                            admission,
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
                            record_terminal_with_circuit(
                                &state, &metrics, started, admission, Outcome::RenderFailed,
                                first_token_at, None,
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
                        record_terminal_with_circuit(
                            &state, &metrics, started, admission, Outcome::StreamBroken,
                            first_token_at, None,
                        );
                        yield Err(map_upstream_error(&err, true, produced));
                        break;
                    }
                    // The rest of the text, with the ids of everything still held back.
                    let (text, ids) = match retokenizer.push_with_text(&text).and_then(|pushed| {
                        retokenizer.finish_with_text().map(|tail| {
                            let mut ids = pushed.ids;
                            ids.extend(tail.ids);
                            (pushed.text + &tail.text, ids)
                        })
                    }) {
                        Ok(emitted) => emitted,
                        Err(err) => {
                            record_terminal_with_circuit(
                                &state, &metrics, started, admission, Outcome::RenderFailed,
                                first_token_at, None,
                            );
                            yield Err(migratable_error(err.to_string()));
                            break;
                        }
                    };
                    generated_tokens = generated_tokens.saturating_add(ids.len() as u32);
                    // Content the renderer held back until `finish` still has
                    // a first token time.
                    if first_token_at.is_none() && !text.is_empty() {
                        first_token_at = Some(Instant::now());
                    }
                    // The client sees the counts a primary worker would report, from our tokenizer,
                    // and the prompt tokens the provider served from its cache (`client_usage`).
                    // The provider's own prompt and completion counts come from its tokenizer and
                    // template: they would reveal a third party and bill the prompt differently, so
                    // they only feed the proxy's billing metrics. When the provider sent no `usage`
                    // at all there is no provider-billed figure to record; the local counts are not it.
                    let local = client_usage(prompt_tokens, generated_tokens, provider_usage.as_ref());
                    let billed = provider_billed(local.clone(), provider_usage);
                    if let Some(provider) = &provider_usage {
                        provider.record(&metrics);
                    }
                    if finish_reason.as_deref() == Some("content_filter") {
                        // A primary worker never filters, so a provider's filter must not decide
                        // the answer: retry elsewhere. Before output a primary worker serves the
                        // request; after output the migration layer continues from the tokens
                        // already delivered, on a primary worker, since proxies refuse replays.
                        record_terminal_with_circuit(
                            &state,
                            &metrics,
                            started,
                            admission,
                            Outcome::ContentFiltered,
                            first_token_at,
                            billed.as_ref(),
                        );
                        tracing::warn!(
                            provider = %provider,
                            output_started = produced,
                            "provider stopped the response with its content filter"
                        );
                        yield Err(migratable_error(
                            "the provider stopped the response with its content filter",
                        ));
                        break;
                    }
                    let reason = finish_reason_from(finish_reason.as_deref());
                    record_terminal_with_circuit(
                        &state,
                        &metrics,
                        started,
                        admission,
                        outcome_for_finish(&reason),
                        first_token_at,
                        billed.as_ref(),
                    );
                    yield Ok(stamp_served_by(terminal(reason, text, ids, local), &served_by));
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
                        record_terminal_with_circuit(
                            &state, &metrics, started, admission, Outcome::RenderFailed,
                            first_token_at, None,
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
                // Text is streamed only together with its ids. The frontend counts usage and
                // migrates from the ids, so text ahead of them would be shown twice: once now,
                // and again when the held-back ids are decoded or regenerated.
                let emitted = match retokenizer.push_with_text(&text) {
                    Ok(emitted) => emitted,
                    Err(err) => {
                        record_terminal_with_circuit(
                            &state, &metrics, started, admission, Outcome::RenderFailed,
                            first_token_at, None,
                        );
                        yield Err(migratable_error(err.to_string()));
                        break;
                    }
                };
                if emitted.text.is_empty() && emitted.ids.is_empty() {
                    continue;
                }
                generated_tokens = generated_tokens.saturating_add(emitted.ids.len() as u32);
                yield Ok(stamp_served_by(text_chunk(emitted.text, emitted.ids), &served_by));
            }
        };
        Ok(Box::pin(stream))
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let state = self.state.clone();
        Ok(vec![KvEventSource::Push {
            dp_rank: self.config.dp_rank,
            on_ready: Box::new(move |publisher| {
                let replayed = state.events.set(publisher);
                if let Some(metrics) = state
                    .metrics
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    for (kind, count) in replayed.kinds() {
                        metrics.add_kv_events(kind, count);
                    }
                    metrics.add_kv_events_dropped(replayed.dropped);
                }
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
async fn load_tokenizer(model_path: &str) -> anyhow::Result<ModelTokenizer> {
    let path = Path::new(model_path);
    if path.is_file() {
        return load_hf_tokenizer(path);
    }
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        dynamo_llm::local_model::LocalModel::fetch(model_path, true)
            .await
            .map_err(|err| anyhow::anyhow!("resolve model '{model_path}': {err}"))?
    };
    let tokenizer_json = dir.join("tokenizer.json");
    if tokenizer_json.is_file() {
        return load_hf_tokenizer(&tokenizer_json);
    }
    // Kimi K3 ships only `tiktoken.model`: find it as the model card does and load it with
    // Dynamo's tiktoken loader, as the frontend does for the same card.
    let Some(TokenizerKind::TikTokenModel(file)) = TokenizerKind::from_disk(&dir)? else {
        anyhow::bail!("no tokenizer.json or tiktoken model in {}", dir.display());
    };
    let path = file
        .path()
        .ok_or_else(|| anyhow::anyhow!("tiktoken model is not a local file"))?;
    let tokenizer = TikTokenTokenizer::from_file_auto(&path.to_string_lossy())
        .map_err(|err| anyhow::anyhow!("load tiktoken tokenizer {}: {err}", path.display()))?;
    Ok(ModelTokenizer::TikToken(Arc::new(tokenizer).into()))
}

fn load_hf_tokenizer(path: &Path) -> anyhow::Result<ModelTokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map(|tokenizer| ModelTokenizer::HuggingFace(Arc::new(tokenizer)))
        .map_err(|err| anyhow::anyhow!("load tokenizer {}: {err}", path.display()))
}

/// Expire virtual-cache blocks four times per TTL, at least once a second.
fn expire_period(config: &ProxyConfig) -> Duration {
    (Duration::from_secs(config.vcache_ttl_secs) / 4).max(Duration::from_secs(1))
}

/// Hash inputs the router uses beyond the tokens.
///
/// `is_eagle` must equal the primary workers' EAGLE/MTP setting
/// ([`ProxyConfig::enable_eagle`]): the router hashes prompts differently for
/// EAGLE, so a proxy that disagreed would publish block hashes the router never
/// looks up and cache affinity would break.
fn hash_options(request: &PreprocessedRequest, is_eagle: bool) -> HashOptions {
    let routing = request.routing.as_ref();
    HashOptions {
        lora_name: routing.and_then(|routing| routing.lora_name.clone()),
        cache_salt: routing.and_then(|routing| routing.cache_namespace.clone()),
        is_eagle,
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
                 delivered); retry on a primary worker"
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

/// Why a request does not fit the model's context window, or `None` when it fits or the config
/// sets no context length.
///
/// The same two checks a primary worker makes, against the context length the proxy's card
/// advertises: a prompt that fills the window leaves no room for output, and the prompt plus the
/// requested output may not exceed it (SGLang's request validation, and the frontend's check
/// against the token budget SGLang and vLLM workers publish, whose messages these are). A
/// primary's engine also counts tokens it reserves, for example for speculative decoding, which
/// the card does not carry.
fn context_overflow(
    prompt_tokens: usize,
    max_tokens: Option<u32>,
    context_length: Option<u32>,
) -> Option<String> {
    let limit = context_length? as usize;
    if prompt_tokens >= limit {
        return Some(format!(
            "This model's maximum context length is {limit} tokens. However, your messages \
             resulted in {prompt_tokens} tokens. Please reduce the length of the messages."
        ));
    }
    let max_tokens = max_tokens?;
    let requested = prompt_tokens.saturating_add(max_tokens as usize);
    (requested > limit).then(|| {
        format!(
            "This model configuration accepts at most {limit} combined input and output \
             tokens. However, your request has {prompt_tokens} input tokens and asks for \
             {max_tokens} output tokens ({requested} tokens total). Please reduce the input \
             length or requested output length."
        )
    })
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

/// Map a terminal request outcome onto what the breaker should do with it.
/// `None` for proxy-side outcomes (a refusal, a cancellation, a migration
/// replay, an unsupported request) that carry no information about the
/// provider's health.
/// Owns a half-open probe for the life of its request. Dropping it hands the probe back unless
/// a provider verdict already settled it (then `release_probe` is a no-op), so a probe that is
/// cancelled, refused by the proxy, or whose stream is dropped cannot leave the breaker
/// half-open and refusing every request.
struct ProbeGuard {
    state: Arc<EngineState>,
    generation: u64,
}

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        self.state
            .circuit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release_probe(Instant::now(), self.generation);
    }
}

fn circuit_health(outcome: Outcome) -> Option<CircuitHealth> {
    match outcome {
        Outcome::Ok => Some(CircuitHealth::Success),
        // A provider response, even a rejection, proves reachability.
        Outcome::Rejected | Outcome::ContentFiltered => Some(CircuitHealth::Answered),
        Outcome::RateLimited
        | Outcome::Unavailable
        | Outcome::AuthError
        | Outcome::Transport
        | Outcome::StreamBroken => Some(CircuitHealth::Failure),
        Outcome::Cancelled
        | Outcome::MigrationReplay
        | Outcome::NoChatRequest
        | Outcome::Unsupported
        | Outcome::RenderFailed
        | Outcome::CircuitOpen
        | Outcome::ContextOverflow => None,
    }
}

/// [`record_terminal`] plus the circuit-breaker feed, so every call site keeps
/// the two in sync.
fn record_terminal_with_circuit(
    state: &EngineState,
    metrics: &Option<Arc<ProxyMetrics>>,
    started: Instant,
    admission: Admission,
    outcome: Outcome,
    first_token_at: Option<Instant>,
    usage: Option<&CompletionUsage>,
) {
    record_terminal(metrics, started, outcome, first_token_at, usage);
    state.record_circuit(admission, outcome);
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
///
/// `text` is always set, even when empty: with `text: None` the frontend decodes `token_ids`
/// itself, and would show the held-back tail a second time.
pub fn terminal(
    reason: FinishReason,
    text: String,
    token_ids: Vec<u32>,
    usage: CompletionUsage,
) -> LLMEngineOutput {
    LLMEngineOutput {
        text: Some(text),
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
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ProviderUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    /// `prompt_tokens_details.cached_tokens`: prompt tokens served from the provider's cache.
    pub cached_prompt_tokens: Option<u32>,
    /// `cost`, as some gateways report it, in the provider's billing unit.
    pub cost: Option<f64>,
}

impl ProviderUsage {
    /// Count the provider-only figures (cache and cost). They go to the proxy's metrics only,
    /// never to the client.
    fn record(&self, metrics: &Option<Arc<ProxyMetrics>>) {
        let Some(metrics) = metrics else {
            return;
        };
        if let Some(cached) = self.cached_prompt_tokens {
            metrics.add_cached_prompt_tokens(u64::from(cached));
        }
        if let Some(cost) = self.cost {
            metrics.add_provider_cost(cost);
        }
    }
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

/// The usage the client sees: our tokenizer's prompt and completion counts, and the prompt tokens
/// the provider served from its cache as `prompt_tokens_details.cached_tokens`, the field our
/// workers set from their prefix cache and the gateway bills at the cache-read price (on every
/// route, OpenRouter's included). Capped at the prompt the client is billed for; absent, as on a
/// worker, when nothing was cached.
fn client_usage(
    prompt_tokens: u32,
    completion_tokens: u32,
    provider: Option<&ProviderUsage>,
) -> CompletionUsage {
    let mut usage = dynamo_backend_common::usage(prompt_tokens, completion_tokens);
    if let Some(cached) = provider
        .and_then(|provider| provider.cached_prompt_tokens)
        .filter(|cached| *cached > 0)
    {
        usage
            .prompt_tokens_details
            .get_or_insert_with(Default::default)
            .cached_tokens = Some(cached.min(prompt_tokens));
    }
    usage
}

/// The usage to record in the provider-billed metrics.
///
/// `None` when the provider sent no `usage` object: the locally computed fallback is not what
/// the provider billed, so it must not be counted under a provider-billed metric. A provider
/// that sent a partial object is merged over the local counts field by field by
/// [`ProviderUsage::over`].
fn provider_billed(
    local: dynamo_backend_common::CompletionUsage,
    provider: Option<ProviderUsage>,
) -> Option<dynamo_backend_common::CompletionUsage> {
    provider.map(|provider| provider.over(local))
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
        cached_prompt_tokens: object
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
        // A negative or non-finite cost would corrupt a monotonic counter.
        cost: object
            .get("cost")
            .and_then(Value::as_f64)
            .filter(|cost| cost.is_finite() && *cost >= 0.0),
    })
}

/// Map a provider failure to the error the frontend sees.
///
/// The frontend reports a worker down (`report_instance_down`) for connection, shutdown and
/// incomplete-stream errors, which takes it out of routing. A proxy is only unusable when its
/// provider key is rejected (401), so that is the one case mapped to `EngineShutdown`. Every
/// other provider failure concerns one request or a transient provider condition, so it maps to
/// `WorkerOverloaded`: migratable, without quarantining the proxy, which a moderation 403 is too.
///
/// A rejection of the request's shape or size ([`cannot_serve`]) is the exception: every proxy of
/// the tier shares the provider and its configuration and gets the same rejection, and a proxy is
/// only chosen when the primaries are past their failover point, so a migration lands on the
/// sibling proxy and then finds no worker, and the client gets 503/529 (and, through the
/// gateway's OpenRouter fallback, the same rejection late). It maps to
/// `Backend(InvalidArgument)`, which the frontend returns as a 400 without migrating, as for a
/// primary's own invalid-request refusal. After output has started the migration layer replays
/// the delivered tokens, and the retry cannot land back on a proxy (it refuses replays), so the
/// same mapping applies.
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
    if cannot_serve(err) {
        return backend_error(BackendError::InvalidArgument, message);
    }
    migratable_error(message)
}

/// A provider rejection of this request's shape or size: a 400, 413 or 422 (OpenRouter's context
/// length refusal, and its "Provider returned error" for a provider's 400), or OpenRouter's 404
/// when no endpoint of the model has a capability the request needs ("No endpoints found that
/// support image input", "... that can handle the requested parameters") or room for it (its
/// routing funnel names "Filter by Context Length", for a prompt or `max_tokens` above every
/// endpoint's limit). OpenRouter's 404 for an account-wide data policy ("No endpoints found
/// matching your data policy") concerns every request, not this one, and keeps the migratable
/// mapping.
fn cannot_serve(err: &UpstreamError) -> bool {
    match err {
        UpstreamError::Rejected {
            status: 400 | 413 | 422,
            ..
        } => true,
        UpstreamError::Rejected {
            status: 404,
            message,
        } => {
            message.starts_with("No endpoints found that")
                || message.contains("Filter by Context Length")
        }
        _ => false,
    }
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

    /// Kimi K3 ships `tiktoken.model` and no `tokenizer.json`. The proxy loads such a directory
    /// with Dynamo's tiktoken loader and retokenizes with it, structural tokens included.
    #[tokio::test]
    async fn loads_a_tiktoken_only_model_directory() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../proxy-core/tests/fixtures/tiktoken-tiny");
        assert!(!dir.join("tokenizer.json").exists());
        let tokenizer = load_tokenizer(dir.to_str().unwrap()).await.unwrap();
        assert!(matches!(tokenizer, ModelTokenizer::TikToken(_)));

        let text = "<|open|>think<|sep|> hello";
        let mut retokenizer = Retokenizer::with_model(tokenizer.clone());
        let mut ids = retokenizer.push(text).unwrap();
        ids.extend(retokenizer.finish().unwrap());
        assert_eq!(ids, vec![275, 116, 104, 105, 110, 107, 277, 260]);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }

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
    fn terminal_with_empty_text_still_carries_text() {
        // Ids with no new text must not be left for the frontend to decode.
        let chunk = terminal(
            FinishReason::Stop,
            String::new(),
            vec![42],
            dynamo_backend_common::usage(1, 1),
        );
        assert_eq!(chunk.text.as_deref(), Some(""));
        assert_eq!(chunk.token_ids, vec![42]);
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
    fn usage_parses_provider_cache_and_cost() {
        let value = serde_json::json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 64},
            "cost": 0.0012,
        });
        let usage = parse_usage(&value).expect("object parses");
        assert_eq!(usage.cached_prompt_tokens, Some(64));
        assert_eq!(usage.cost, Some(0.0012));
        // A negative cost would corrupt a monotonic counter, so it is ignored.
        let negative = parse_usage(&serde_json::json!({"cost": -1.0})).unwrap();
        assert_eq!(negative.cost, None);
        let absent = parse_usage(&serde_json::json!({"prompt_tokens": 1})).unwrap();
        assert_eq!((absent.cached_prompt_tokens, absent.cost), (None, None));
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
    fn client_usage_carries_the_provider_cache_hits() {
        let provider = parse_usage(&serde_json::json!({
            "prompt_tokens": 1010, "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 896, "cache_write_tokens": 0},
        }))
        .unwrap();
        let usage = client_usage(1000, 7, Some(&provider));
        // Our counts, the provider's cache hits.
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (1000, 7));
        let cached = |usage: &CompletionUsage| {
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
        };
        assert_eq!(cached(&usage), Some(896));
        // Never more than the prompt the client is billed for.
        assert_eq!(cached(&client_usage(500, 7, Some(&provider))), Some(500));
        // Nothing cached, or no provider usage: no details, as a worker reports it.
        let cold = parse_usage(&serde_json::json!({"prompt_tokens_details": {"cached_tokens": 0}}));
        assert!(
            client_usage(1000, 7, cold.as_ref())
                .prompt_tokens_details
                .is_none()
        );
        assert!(client_usage(1000, 7, None).prompt_tokens_details.is_none());
    }

    #[test]
    fn provider_billed_is_none_without_provider_usage() {
        // The success path records only the provider's own usage. A provider that omitted the
        // object leaves the provider-billed counters untouched instead of counting the local
        // fallback as if the provider had billed it.
        let fallback = dynamo_backend_common::usage(11, 7);
        assert!(provider_billed(fallback.clone(), None).is_none());
        let provider = parse_usage(&serde_json::json!({"total_tokens": 99})).unwrap();
        let billed = provider_billed(fallback, Some(provider)).expect("provider usage records");
        assert_eq!(billed.total_tokens, 99);
        assert_eq!(billed.prompt_tokens, 11);
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
        // A rejection of the request's shape or size is the client's error on every proxy of the
        // tier: a 400 the frontend does not migrate.
        let cannot_serve = [
            (
                400,
                "This endpoint's maximum context length is 262144 tokens.",
            ),
            (400, "Provider returned error"),
            (413, "Request too large"),
            (404, "No endpoints found that support image input"),
            (
                404,
                "No endpoints found that can handle the requested parameters.",
            ),
            (
                404,
                "No endpoints found for z-ai/glm-5.2. Every candidate endpoint was removed during \
                 routing: Filter by Context Length removed deepinfra/fp4, z-ai/fp8.",
            ),
        ];
        for (status, message) in cannot_serve {
            let err = UpstreamError::Rejected {
                status,
                message: message.to_string(),
            };
            for output_started in [false, true] {
                let mapped = map_upstream_error(&err, err.retry_elsewhere(), output_started);
                assert_eq!(
                    mapped.error_type(),
                    ErrorType::Backend(BackendError::InvalidArgument),
                    "{err}"
                );
            }
        }
        // Everything else concerns one request or a transient provider condition: retried
        // elsewhere, and never an error type the frontend quarantines the instance for.
        let others = [
            UpstreamError::Rejected {
                status: 403,
                message: "flagged by moderation".to_string(),
            },
            UpstreamError::Rejected {
                status: 404,
                message: "No endpoints found matching your data policy".to_string(),
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
        let tag = serde_json::json!({"served_by": "spillover", "tier": "spillover"});
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
        assert_eq!(data["served_by"], "spillover");
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

    const TEST_KEY_ENV: &str = "DW_PROXY_ENGINE_TEST_KEY";

    /// A minimal but valid byte-level tokenizer, enough for the engine to
    /// decode a prompt tail and retokenize. The provider is never reached in
    /// the circuit-breaker test, so the exact vocabulary does not matter.
    fn test_tokenizer() -> Arc<tokenizers::Tokenizer> {
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hi": 1, "there": 2},
                "unk_token": "[UNK]"
            }
        }"#;
        Arc::new(
            tokenizers::Tokenizer::from_bytes(TOKENIZER_JSON).expect("test tokenizer must build"),
        )
    }

    /// A chat request the proxy admits, with its own stop conditions and a KV
    /// routing-ready prompt.
    fn chat_request() -> PreprocessedRequest {
        chat_request_with_tokens(vec![1u32, 2, 3])
    }

    fn chat_request_with_tokens(token_ids: Vec<u32>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock/model".to_string())
            .token_ids(token_ids)
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .extra_args(Some(serde_json::json!({
                "chat_request": {"messages": [{"role": "user", "content": "hi"}]}
            })))
            .build()
            .expect("request builds")
    }

    fn context() -> GenerateContext {
        use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
        GenerateContext::new(Context::<()>::new(()).context(), None)
    }

    /// Build a `ProxyEngine` directly, bypassing `ProxyEngine::new`'s tokenizer
    /// fetch so the test needs no model files.
    fn engine(base_url: String, failure_threshold: u32) -> ProxyEngine {
        engine_with_cooldown(base_url, failure_threshold, 30_000)
    }

    fn engine_with_cooldown(
        base_url: String,
        failure_threshold: u32,
        cooldown_ms: u64,
    ) -> ProxyEngine {
        use dw_proxy_core::circuit_breaker::CircuitBreakerConfig;
        use dw_proxy_core::upstream::ProviderConfig;
        let provider = ProviderConfig {
            name: "mock".to_string(),
            base_url,
            api_key_env: TEST_KEY_ENV.to_string(),
            model: "mock/model".to_string(),
            provider_preferences: None,
            body_overrides: None,
            extra_headers: Default::default(),
            connect_timeout_ms: 1_000,
            read_timeout_ms: 5_000,
            omitted_max_tokens: 131_072,
            thinking_dialect: Default::default(),
            thinking_strict: false,
            cache_key: Default::default(),
            cache_key_secret_env: None,
            allow_insecure_http: true,
            circuit_breaker: Some(CircuitBreakerConfig {
                failure_threshold,
                cooldown_ms,
                max_cooldown_ms: 300_000,
            }),
        };
        let client = UpstreamClient::new(provider.clone()).expect("provider client builds");
        let config = ProxyConfig {
            model_path: "/models/mock".to_string(),
            served_model_names: vec!["mock/model".to_string()],
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            kv_block_size: 16,
            context_length: Some(4096),
            custom_jinja_template: None,
            enable_eagle: false,
            omit_source_path: false,
            model_revision: None,
            dp_rank: 7,
            tier: "spillover".to_string(),
            parser_family: render::ParserFamily::Glm47,
            endpoint_types: "chat,completions".to_string(),
            provider,
            router_config: None,
            advertised_capacity: None,
            vcache_ttl_secs: 300,
            vcache_max_blocks: 1024,
        };
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
            circuit: Mutex::new(CircuitBreaker::new(
                config.provider.circuit_breaker.unwrap_or_default(),
            )),
        });
        ProxyEngine {
            config: Arc::new(config),
            client,
            tokenizer: ModelTokenizer::HuggingFace(test_tokenizer()),
            state,
        }
    }

    /// Read one HTTP request fully so the client sees a clean exchange.
    async fn read_http_request(socket: &mut tokio::net::TcpStream) {
        use tokio::io::AsyncReadExt;
        let mut buffer = Vec::new();
        let mut scratch = [0u8; 1024];
        while let Ok(n) = socket.read(&mut scratch).await {
            if n == 0 {
                break;
            }
            buffer.extend_from_slice(&scratch[..n]);
            if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buffer[..pos]);
                let content_length: usize = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if buffer.len() >= pos + 4 + content_length {
                    break;
                }
            }
        }
    }

    #[tokio::test]
    async fn a_failed_provider_call_advertises_no_virtual_cache() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                read_http_request(&mut socket).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnope",
                    )
                    .await;
                let _ = socket.shutdown().await;
            }
        });

        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        let engine = engine(format!("http://{addr}/v1"), 5);
        // Four full 16-token blocks: a successful call would advertise them.
        let request = chat_request_with_tokens((0..64).collect());
        assert!(engine.generate(request, context()).await.is_err());
        assert_eq!(
            engine
                .state
                .vcache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len_blocks(),
            0,
            "a request the provider never accepted must not attract cache affinity"
        );
        server.abort();
    }

    #[tokio::test]
    async fn a_probe_that_ends_without_a_verdict_is_handed_back() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let reached = Arc::new(AtomicUsize::new(0));
        let reached_server = reached.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                reached_server.fetch_add(1, Ordering::SeqCst);
                read_http_request(&mut socket).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnope",
                    )
                    .await;
                let _ = socket.flush().await;
                let _ = socket.shutdown().await;
            }
        });

        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        let engine = engine_with_cooldown(format!("http://{addr}/v1"), 1, 50);

        // One failure opens the breaker.
        assert!(engine.generate(chat_request(), context()).await.is_err());
        assert_eq!(reached.load(Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_millis(80)).await;

        // The cooldown has elapsed, so this request is the probe; the proxy refuses it before
        // calling the provider (n = 3 is unsupported), which is no verdict on the provider.
        let mut unsupported = chat_request();
        unsupported.extra_args = Some(serde_json::json!({
            "chat_request": {"messages": [{"role": "user", "content": "hi"}], "n": 3}
        }));
        assert!(engine.generate(unsupported, context()).await.is_err());
        assert_eq!(reached.load(Ordering::SeqCst), 1);

        // The probe was handed back, so the next request probes the provider instead of being
        // refused by a breaker stuck half-open.
        let err = engine
            .generate(chat_request(), context())
            .await
            .err()
            .expect("the provider is still down");
        assert!(
            !err.to_string().contains("circuit is open"),
            "the breaker must not stay half-open: {err}"
        );
        assert_eq!(reached.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[tokio::test]
    async fn circuit_opens_after_consecutive_failures_and_then_refuses_without_calling_the_provider()
     {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let reached = Arc::new(AtomicUsize::new(0));
        let reached_server = reached.clone();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                reached_server.fetch_add(1, Ordering::SeqCst);
                read_http_request(&mut socket).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnope",
                    )
                    .await;
                let _ = socket.flush().await;
                let _ = socket.shutdown().await;
            }
        });

        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        let engine = engine(format!("http://{addr}/v1"), 2);

        // Two consecutive provider failures open the breaker (threshold 2).
        for attempt in 1..=2 {
            let err = engine
                .generate(chat_request(), context())
                .await
                .err()
                .expect("the provider is down");
            assert_eq!(
                err.error_type(),
                ErrorType::WorkerOverloaded,
                "attempt {attempt} must stay migratable: {err}"
            );
        }
        assert_eq!(
            reached.load(Ordering::SeqCst),
            2,
            "both failed attempts reached the provider"
        );
        assert!(
            engine
                .state
                .circuit
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_open(),
            "two failures open the breaker"
        );

        // The next request is refused before any provider call.
        let err = engine
            .generate(chat_request(), context())
            .await
            .err()
            .expect("the open breaker refuses");
        assert_eq!(err.error_type(), ErrorType::WorkerOverloaded);
        assert!(
            err.to_string().contains("circuit is open"),
            "the refusal must name the circuit: {err}"
        );
        assert_eq!(
            reached.load(Ordering::SeqCst),
            2,
            "an open breaker must not call the provider"
        );
        server.abort();
    }

    /// A tokenizer whose pre-tokens carry their leading space (`"Hello"`, `" world"`) and whose
    /// decoder concatenates them, so decoding ids reproduces the text exactly. The retokenizer
    /// holds the last pre-token back, which is what exposed the repeated-tail bug.
    fn spaced_tokenizer() -> Arc<tokenizers::Tokenizer> {
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Split",
                "pattern": {"Regex": "\\s*\\S+"},
                "behavior": "Isolated",
                "invert": false
            },
            "post_processor": null,
            "decoder": {"type": "Fuse"},
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "Hello": 1, " world": 2, "The": 3, " answer": 4,
                          " is": 5, " 42.": 6},
                "unk_token": "[UNK]"
            }
        }"#;
        Arc::new(tokenizers::Tokenizer::from_bytes(TOKENIZER_JSON).expect("tokenizer builds"))
    }

    /// Serve streamed chat completions with these content deltas, then `stop`.
    async fn serve_completion(deltas: Vec<&'static str>) -> (String, JoinHandle<()>) {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                read_http_request(&mut socket).await;
                let mut body = String::new();
                for delta in &deltas {
                    let chunk = serde_json::json!(
                        {"choices": [{"index": 0, "delta": {"content": delta}}]}
                    );
                    body.push_str(&format!("data: {chunk}\n\n"));
                }
                let stop = serde_json::json!(
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                );
                body.push_str(&format!("data: {stop}\n\ndata: [DONE]\n\n"));
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}/v1"), server)
    }

    type Annotated = dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>;

    /// Replays a proxy's outputs into the frontend's `Backend`, as the KV router would.
    struct Replay(Vec<LLMEngineOutput>);

    #[async_trait]
    impl
        dynamo_runtime::pipeline::AsyncEngine<
            dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
            dynamo_runtime::pipeline::ManyOut<Annotated>,
            dynamo_runtime::pipeline::Error,
        > for Replay
    {
        async fn generate(
            &self,
            request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        ) -> Result<dynamo_runtime::pipeline::ManyOut<Annotated>, dynamo_runtime::pipeline::Error>
        {
            use dynamo_runtime::pipeline::AsyncEngineContextProvider;
            let outputs: Vec<Annotated> =
                self.0.iter().cloned().map(Annotated::from_data).collect();
            Ok(dynamo_runtime::pipeline::ResponseStream::new(
                Box::pin(futures::stream::iter(outputs)),
                request.context(),
            ))
        }
    }

    /// What a client sees for a proxy-served answer: the proxy's stream run through the
    /// frontend `Backend`, which decodes the ids itself whenever a chunk carries no text.
    async fn client_text(deltas: Vec<&'static str>) -> (String, Vec<LLMEngineOutput>) {
        use dynamo_runtime::pipeline::Operator;

        let (base_url, server) = serve_completion(deltas).await;
        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        let mut engine = engine(base_url, 5);
        engine.tokenizer = ModelTokenizer::HuggingFace(spaced_tokenizer());
        let outputs: Vec<LLMEngineOutput> = engine
            .generate(chat_request(), context())
            .await
            .expect("the proxy accepts the request")
            .map(|output| output.expect("the proxy streams without error"))
            .collect()
            .await;
        server.abort();

        let hf = dynamo_llm::tokenizers::HuggingFaceTokenizer::from_tokenizer(
            (*spaced_tokenizer()).clone(),
        );
        let shared: Arc<dyn dynamo_llm::tokenizers::traits::Tokenizer> = Arc::new(hf);
        let backend = dynamo_llm::backend::Backend::from_tokenizer(shared.into());
        let request = PreprocessedRequest::builder()
            .model("mock/model".to_string())
            .token_ids(vec![1u32])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .expect("request builds");
        let replay: dynamo_runtime::pipeline::ServerStreamingEngine<
            PreprocessedRequest,
            Annotated,
        > = Arc::new(Replay(outputs.clone()));
        let stream = Operator::generate(
            backend.as_ref(),
            dynamo_runtime::pipeline::SingleIn::new(request),
            replay,
        )
        .await
        .expect("the backend accepts the stream");
        let text = stream
            .filter_map(|annotated| async move { annotated.data.and_then(|data| data.text) })
            .collect::<Vec<_>>()
            .await
            .concat();
        (text, outputs)
    }

    #[tokio::test]
    async fn proxy_answers_reach_the_client_exactly_once() {
        for (deltas, expected) in [
            (vec!["Hello", " world"], "Hello world"),
            (vec!["Hello world"], "Hello world"),
            (vec!["The answer", " is 42."], "The answer is 42."),
        ] {
            let (text, outputs) = client_text(deltas.clone()).await;
            assert_eq!(text, expected, "client text for {deltas:?}");

            // Every chunk's text is exactly what its ids encode, so usage and a mid-stream
            // migration (which continues from the ids) agree with what the client has seen.
            let tokenizer = spaced_tokenizer();
            for output in &outputs {
                let decoded = tokenizer.decode(&output.token_ids, false).unwrap();
                assert_eq!(
                    output.text.as_deref(),
                    Some(decoded.as_str()),
                    "chunk {output:?}"
                );
            }
        }
    }

    #[test]
    fn context_overflow_mirrors_the_primary_checks() {
        let limit = Some(4096);
        assert_eq!(context_overflow(4095, None, limit), None);
        assert_eq!(
            context_overflow(4000, Some(96), limit),
            None,
            "exactly full fits"
        );
        // A prompt that fills the window leaves no room for output.
        let full = context_overflow(4096, None, limit).expect("a full prompt is refused");
        assert!(
            full.contains("4096 tokens") && full.contains("resulted in 4096"),
            "{full}"
        );
        assert!(context_overflow(1_169_033, Some(1), limit).is_some());
        // The prompt plus the requested output may not exceed it.
        let total = context_overflow(4000, Some(97), limit).expect("4097 tokens are refused");
        assert!(total.contains("(4097 tokens total)"), "{total}");
        // No configured context length: the card advertises none, so nothing to check.
        assert_eq!(context_overflow(1_169_033, Some(1), None), None);
    }

    #[tokio::test]
    async fn a_request_over_the_context_length_is_a_client_error_without_a_provider_call() {
        use dynamo_runtime::error::ErrorClass;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let reached = Arc::new(AtomicUsize::new(0));
        let reached_server = reached.clone();
        let server = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                reached_server.fetch_add(1, Ordering::SeqCst);
                read_http_request(&mut socket).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnope",
                    )
                    .await;
                let _ = socket.shutdown().await;
            }
        });
        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        // The test card advertises a 4096-token context.
        let engine = engine(format!("http://{addr}/v1"), 5);

        let full_prompt = chat_request_with_tokens(vec![1; 4096]);
        let mut over_total = chat_request_with_tokens(vec![1; 100]);
        over_total.stop_conditions.max_tokens = Some(3_997);
        // A completions request carries no chat request; it is refused the same way.
        let mut no_chat_request = chat_request_with_tokens(vec![1; 5_000]);
        no_chat_request.extra_args = None;
        for request in [full_prompt, over_total, no_chat_request] {
            let outputs: Vec<_> = engine
                .generate(request, context())
                .await
                .expect("the refusal is the stream's first item, not a pre-stream failure")
                .collect()
                .await;
            let [Err(err)] = outputs.as_slice() else {
                panic!("expected exactly one error item, got {outputs:?}");
            };
            // What an SGLang worker's own context check yields: the frontend answers 400 and
            // the router neither migrates it nor counts it against the worker.
            assert_eq!(
                err.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert_eq!(err.class(), ErrorClass::InvalidRequest);
            assert_eq!(err.reason().as_str(), "backend.invalid_argument");
        }
        assert_eq!(reached.load(Ordering::SeqCst), 0, "no provider call");

        // A request that fits still reaches the provider, and its failure keeps its mapping.
        let mut fits = chat_request_with_tokens(vec![1; 100]);
        fits.stop_conditions.max_tokens = Some(3_996);
        let err = engine
            .generate(fits, context())
            .await
            .err()
            .expect("the provider is down");
        assert_eq!(err.error_type(), ErrorType::WorkerOverloaded);
        assert_eq!(reached.load(Ordering::SeqCst), 1);
        server.abort();
    }

    /// Read one HTTP request and return its JSON body.
    async fn read_http_body(socket: &mut tokio::net::TcpStream) -> Value {
        use tokio::io::AsyncReadExt;
        let mut buffer = Vec::new();
        let mut scratch = [0u8; 4096];
        loop {
            let n = socket.read(&mut scratch).await.unwrap_or(0);
            if n == 0 {
                return Value::Null;
            }
            buffer.extend_from_slice(&scratch[..n]);
            let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&buffer[..pos]).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(0);
            if buffer.len() >= pos + 4 + length {
                return serde_json::from_slice(&buffer[pos + 4..pos + 4 + length])
                    .unwrap_or(Value::Null);
            }
        }
    }

    /// A provider that answers every request with these SSE chunks and reports each request body.
    async fn serve_chunks(
        chunks: Vec<Value>,
    ) -> (
        String,
        tokio::sync::mpsc::UnboundedReceiver<Value>,
        JoinHandle<()>,
    ) {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (bodies, received) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let _ = bodies.send(read_http_body(&mut socket).await);
                let mut body = String::new();
                for chunk in &chunks {
                    body.push_str(&format!("data: {chunk}\n\n"));
                }
                body.push_str("data: [DONE]\n\n");
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}/v1"), received, server)
    }

    fn completion_chunks() -> Vec<Value> {
        vec![
            serde_json::json!({"choices": [{"index": 0, "delta": {"content": "hi"}}]}),
            serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ]
    }

    /// The context guard checks the frontend's cap against the window; the cap sent to the
    /// provider for a client that set none is then clamped to `omitted_max_tokens`. A client's own
    /// cap is never clamped, and one that does not fit is refused without a provider call.
    #[tokio::test]
    async fn the_context_guard_sees_the_frontend_cap_and_the_provider_gets_the_clamped_one() {
        let (base_url, mut bodies, server) = serve_chunks(completion_chunks()).await;
        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        let mut engine = engine(base_url, 5);
        engine.client = UpstreamClient::new(dw_proxy_core::upstream::ProviderConfig {
            omitted_max_tokens: 1_000,
            ..engine.client.config().clone()
        })
        .unwrap();
        let request = |client_cap: Option<u32>, frontend_cap: u32| {
            let mut request = chat_request_with_tokens(vec![1; 100]);
            let mut chat = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
            if let Some(cap) = client_cap {
                chat["max_tokens"] = cap.into();
            }
            request.extra_args = Some(serde_json::json!({ "chat_request": chat }));
            request.stop_conditions.max_tokens = Some(frontend_cap);
            request
        };
        let sent_cap = |body: Value| body["max_tokens"].as_u64();

        // Omitted: the frontend filled 4096 - 100, which fits; the provider gets 1,000.
        let outputs: Vec<_> = engine
            .generate(request(None, 3_996), context())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(outputs.iter().all(Result::is_ok), "{outputs:?}");
        assert_eq!(sent_cap(bodies.recv().await.unwrap()), Some(1_000));

        // Explicit and fitting: sent unchanged, above the omitted default.
        let outputs: Vec<_> = engine
            .generate(request(Some(3_000), 3_000), context())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(outputs.iter().all(Result::is_ok), "{outputs:?}");
        assert_eq!(sent_cap(bodies.recv().await.unwrap()), Some(3_000));

        // Explicit and over the window: refused as the client's error, no provider call, even
        // though the clamp would have fitted it.
        let outputs: Vec<_> = engine
            .generate(request(Some(5_000), 5_000), context())
            .await
            .unwrap()
            .collect()
            .await;
        let [Err(err)] = outputs.as_slice() else {
            panic!("expected one refusal, got {outputs:?}");
        };
        assert_eq!(
            err.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(bodies.try_recv().is_err(), "no provider call");
        server.abort();
    }

    /// A provider that rejects every request with this status and JSON body.
    async fn serve_rejection(status: u16, body: &'static str) -> (String, JoinHandle<()>) {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                read_http_request(&mut socket).await;
                let response = format!(
                    "HTTP/1.1 {status} Rejected\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}/v1"), server)
    }

    /// OpenRouter's refusal of a request no endpoint can serve reaches the frontend as the stream's
    /// one item, a 400 it neither migrates (the sibling proxy would get the same refusal) nor
    /// counts against the proxy; the circuit stays closed.
    #[tokio::test]
    async fn a_provider_capability_rejection_is_a_client_error() {
        use dynamo_runtime::error::ErrorClass;
        let (base_url, server) = serve_rejection(
            404,
            r#"{"error":{"message":"No endpoints found that support image input","code":404}}"#,
        )
        .await;
        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
        let engine = engine(base_url, 1);
        for _ in 0..3 {
            let outputs: Vec<_> = engine
                .generate(chat_request_with_tokens(vec![1; 10]), context())
                .await
                .expect("the refusal is the stream's first item, not a pre-stream failure")
                .collect()
                .await;
            let [Err(err)] = outputs.as_slice() else {
                panic!("expected one error item, got {outputs:?}");
            };
            assert_eq!(
                err.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert_eq!(err.class(), ErrorClass::InvalidRequest);
            assert!(err.to_string().contains("support image input"), "{err}");
        }
        // A failure threshold of 1 would have opened the circuit on a provider failure.
        assert!(
            !engine
                .state
                .circuit
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_open()
        );
        server.abort();
    }
}
