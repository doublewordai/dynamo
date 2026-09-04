// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local admission control: every request dispatched to a worker is
//! recorded here for the lifetime of its response stream, together with its
//! scheduling priority, admission order, and response-stream context.
//!
//! The registry is always on (disable with `DYN_ADMISSION_TRACKING=0`) and is
//! pure accounting: an authoritative per-worker view of the work the frontend
//! has accepted, exported as Prometheus gauges.
//!
//! Enforcement is opt-in via `DYN_ADMISSION_QUEUE_MARGIN` and bounds each
//! worker's **engine queue length**, not its total in-flight: the engine's own
//! scheduler is the capacity oracle for the running set, so the frontend only
//! keeps the waiting work shallow (one global margin, no per-model tuning, no
//! capacity knowledge). The queue signal is the worker's reported waiting
//! count; between reports a burst can overshoot the margin by at most one
//! report-interval's arrivals, which the next report shuts off.
//!
//! Admission follows the priority rule: a request is admitted iff the worker's
//! queue estimate is below the margin, or an in-flight request of strictly
//! lower priority — anywhere in the worker's whole in-flight set, running or
//! queued — can be evicted to make room (victim = lowest priority, tie-break
//! most-recently-admitted, which naturally picks engine-queued, zero-progress
//! work). With no eviction candidate the request is rejected with a typed
//! [`AdmissionRejection`] that the HTTP layer maps to a retryable overload
//! response. Workers that have never reported a queue depth are unenforced.
//!
//! Keyed per worker instance. One state is shared per endpoint via
//! [`get_or_create_admission_state`], mirroring the routing occupancy registry.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use dashmap::DashMap;
use prometheus::{IntCounterVec, IntGaugeVec, Opts};
use tokio_util::sync::CancellationToken;

use crate::component::Endpoint;
use crate::config::environment_names::runtime as env_runtime;
use crate::engine::AsyncEngineContext;
use crate::metrics::prometheus_names::{frontend_service, labels, name_prefix};
use crate::traits::DistributedRuntimeProvider;

/// One tracked in-flight request on a worker.
pub struct AdmissionEntry {
    /// Frontend request id, as carried by the request context. Not unique
    /// across dispatch attempts (migration retries reuse it), so entries are
    /// keyed by `admit_seq` instead.
    pub request_id: String,
    /// Scheduling priority as carried by the request (higher wins). Requests
    /// with no priority hint are recorded at 0.
    pub priority: i32,
    /// Context of the dispatched request. Held so an evictor can terminate the
    /// request on the worker.
    pub context: Arc<dyn AsyncEngineContext>,
    /// Fired when this request is evicted; its tracked stream synthesizes the
    /// client-visible rejection response.
    pub evict_token: CancellationToken,
}

/// Outcome of an enforced admission attempt.
pub(crate) enum AdmissionDecision {
    /// Admitted (slot charged). The charge names the final target worker,
    /// which may differ from the preferred one when retargeting was allowed.
    Admit(AdmissionCharge),
    /// Every eligible worker's queue is at the margin with no lower-priority
    /// victim in its in-flight set.
    Reject { queued: u64, margin: u64 },
}

/// A lower-priority request removed from the registry to make room. Its
/// `evict_token` has already been fired; the victim's own tracked stream
/// reacts by aborting the request on the worker and synthesizing the
/// client-visible rejection response. Returned for logging only.
pub(crate) struct EvictedVictim {
    pub request_id: String,
    pub priority: i32,
    pub worker: u64,
}

/// Per-endpoint registry of in-flight requests, keyed by worker instance id.
/// Each worker's entries are keyed by a monotone admission sequence number,
/// shared across the endpoint's workers so it also orders entries within a
/// priority for victim tie-breaking.
pub struct AdmissionState {
    workers: DashMap<u64, StdMutex<HashMap<u64, AdmissionEntry>>>,
    /// Latest reported engine-queue depth per worker. Absent = never
    /// reported = unenforced.
    reported_waiting: DashMap<u64, u64>,
    /// Queue-length margin; negative = enforcement disabled. Pushed once from
    /// `DYN_ADMISSION_QUEUE_MARGIN` by the worker monitor.
    queue_margin: std::sync::atomic::AtomicI64,
    admit_seq: AtomicU64,
    /// Serializes enforced admission decisions so concurrent requests cannot
    /// both claim the last slot or the same victim. Charges on unenforced
    /// paths bypass it (accounting tolerates that).
    decision_lock: StdMutex<()>,
}

impl Default for AdmissionState {
    fn default() -> Self {
        Self {
            workers: DashMap::new(),
            reported_waiting: DashMap::new(),
            queue_margin: std::sync::atomic::AtomicI64::new(-1),
            admit_seq: AtomicU64::new(0),
            decision_lock: StdMutex::new(()),
        }
    }
}

impl AdmissionState {
    /// Record a dispatched request against `instance_id`. The returned permit
    /// removes the entry when dropped (or when its tracked stream ends).
    pub(crate) fn charge(
        self: &Arc<Self>,
        instance_id: u64,
        request_id: String,
        priority: i32,
        context: Arc<dyn AsyncEngineContext>,
    ) -> AdmissionCharge {
        let admit_seq = self.admit_seq.fetch_add(1, Ordering::Relaxed);
        let evict_token = CancellationToken::new();
        let inflight = {
            let worker = self
                .workers
                .entry(instance_id)
                .or_insert_with(|| StdMutex::new(HashMap::new()));
            let mut entries = worker.lock().unwrap();
            entries.insert(
                admit_seq,
                AdmissionEntry {
                    request_id,
                    priority,
                    context,
                    evict_token: evict_token.clone(),
                },
            );
            entries.len()
        };
        observe_charge(instance_id, inflight);
        AdmissionCharge {
            state: self.clone(),
            instance_id,
            admit_seq,
            evict_token,
        }
    }

    /// Move a charged entry between workers (transport fallback reselected the
    /// target after admission). Accounting only — the destination's cap is not
    /// re-checked on this rare path.
    fn move_entry(&self, from: u64, to: u64, admit_seq: u64) {
        let Some(entry) = self.workers.get(&from).and_then(|worker| {
            let mut entries = worker.lock().unwrap();
            let entry = entries.remove(&admit_seq);
            if entry.is_some() {
                observe_release(from, entries.len());
            }
            entry
        }) else {
            return;
        };
        let inflight = {
            let worker = self
                .workers
                .entry(to)
                .or_insert_with(|| StdMutex::new(HashMap::new()));
            let mut entries = worker.lock().unwrap();
            entries.insert(admit_seq, entry);
            entries.len()
        };
        observe_move(to, inflight);
    }

    /// Enforced admission at priority `priority`, preferring `preferred` and —
    /// when `retarget_candidates` is given — free to admit on any candidate
    /// with headroom instead. At cap everywhere, evicts the lowest-priority
    /// in-flight request strictly below `priority` (tie-break: most recently
    /// admitted); with no such victim, rejects.
    ///
    /// Returns the decision plus the victim, if one was evicted. The victim's
    /// registry entry is already removed and its token fired; the caller kills
    /// its context outside the lock.
    pub(crate) fn admit(
        self: &Arc<Self>,
        preferred: u64,
        retarget_candidates: Option<&[u64]>,
        request_id: String,
        priority: i32,
        context: Arc<dyn AsyncEngineContext>,
    ) -> (AdmissionDecision, Option<EvictedVictim>) {
        let _guard = self.decision_lock.lock().unwrap();

        if self.has_headroom(preferred) {
            return (
                AdmissionDecision::Admit(self.charge(preferred, request_id, priority, context)),
                None,
            );
        }

        let candidates = retarget_candidates.unwrap_or(&[]);
        for &candidate in candidates {
            if candidate != preferred && self.has_headroom(candidate) {
                return (
                    AdmissionDecision::Admit(self.charge(candidate, request_id, priority, context)),
                    None,
                );
            }
        }

        // Victim search across the eligible workers: lowest priority strictly
        // below the incoming request, most-recently-admitted within it.
        let scope: Vec<u64> = std::iter::once(preferred)
            .chain(candidates.iter().copied().filter(|&id| id != preferred))
            .collect();
        let mut best: Option<(u64, u64, i32)> = None; // (worker, admit_seq, priority)
        for &worker_id in &scope {
            let Some(worker) = self.workers.get(&worker_id) else {
                continue;
            };
            let entries = worker.lock().unwrap();
            for (&admit_seq, entry) in entries.iter() {
                if entry.priority >= priority {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((_, best_seq, best_priority)) => {
                        entry.priority < best_priority
                            || (entry.priority == best_priority && admit_seq > best_seq)
                    }
                };
                if better {
                    best = Some((worker_id, admit_seq, entry.priority));
                }
            }
        }

        if let Some((worker_id, admit_seq, _)) = best {
            let victim = {
                let worker = self.workers.get(&worker_id).unwrap();
                let mut entries = worker.lock().unwrap();
                let entry = entries.remove(&admit_seq).unwrap();
                observe_release(worker_id, entries.len());
                entry
            };
            victim.evict_token.cancel();
            observe_eviction(worker_id);
            let evicted = EvictedVictim {
                request_id: victim.request_id,
                priority: victim.priority,
                worker: worker_id,
            };
            return (
                AdmissionDecision::Admit(self.charge(worker_id, request_id, priority, context)),
                Some(evicted),
            );
        }

        let queued = self.reported_queue(preferred).unwrap_or(0);
        let margin = self.queue_margin().unwrap_or(0);
        observe_rejection(preferred);
        (AdmissionDecision::Reject { queued, margin }, None)
    }

    /// Whether `instance_id` can accept another request without eviction:
    /// unenforced (never reported a queue depth, or no margin configured), or
    /// its reported engine queue is below the margin.
    pub(crate) fn has_headroom(&self, instance_id: u64) -> bool {
        let Some(margin) = self.queue_margin() else {
            return true;
        };
        match self.reported_queue(instance_id) {
            None => true,
            Some(queued) => queued < margin,
        }
    }

    /// Workers the gate would reject for right now: their reported engine
    /// queue is at or above the margin. Empty when no margin is configured.
    pub(crate) fn saturated_instances(&self) -> Vec<u64> {
        let Some(margin) = self.queue_margin() else {
            return Vec::new();
        };
        self.reported_waiting
            .iter()
            .filter(|entry| *entry.value() >= margin)
            .map(|entry| *entry.key())
            .collect()
    }

    /// Whether enforcement can currently reject anything: a margin is
    /// configured and at least one worker has reported a queue depth. Cheap
    /// pre-check so unenforced deployments skip the decision lock.
    pub(crate) fn enforcement_active(&self) -> bool {
        self.queue_margin().is_some() && !self.reported_waiting.is_empty()
    }

    /// Configure the queue-length margin (from `DYN_ADMISSION_QUEUE_MARGIN`).
    pub fn set_queue_margin(&self, margin: Option<u64>) {
        let value = margin
            .map(|m| i64::try_from(m).unwrap_or(i64::MAX))
            .unwrap_or(-1);
        self.queue_margin.store(value, Ordering::Relaxed);
    }

    fn queue_margin(&self) -> Option<u64> {
        let value = self.queue_margin.load(Ordering::Relaxed);
        (value >= 0).then_some(value as u64)
    }

    /// Record a worker's reported engine-queue depth (summed across dp ranks).
    /// Pushed by the discovery layer from worker load reports.
    pub fn report_queue_depth(&self, instance_id: u64, waiting: u64) {
        self.reported_waiting.insert(instance_id, waiting);
    }

    /// Latest reported engine-queue depth for a worker, if it has reported.
    pub fn reported_queue(&self, instance_id: u64) -> Option<u64> {
        self.reported_waiting.get(&instance_id).map(|w| *w)
    }

    fn release(&self, instance_id: u64, admit_seq: u64) {
        let Some(worker) = self.workers.get(&instance_id) else {
            return;
        };
        let mut entries = worker.lock().unwrap();
        if entries.remove(&admit_seq).is_some() {
            observe_release(instance_id, entries.len());
        }
    }

    /// Number of tracked in-flight requests on a worker.
    pub fn inflight(&self, instance_id: u64) -> usize {
        self.workers
            .get(&instance_id)
            .map(|w| w.lock().unwrap().len())
            .unwrap_or(0)
    }

    /// Number of tracked in-flight requests on a worker with priority >= `priority`.
    pub fn inflight_at_or_above(&self, instance_id: u64, priority: i32) -> usize {
        self.workers
            .get(&instance_id)
            .map(|w| {
                w.lock()
                    .unwrap()
                    .values()
                    .filter(|e| e.priority >= priority)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Drop tracking (and caps) for workers that no longer exist in discovery.
    pub(crate) fn retain(&self, instance_ids: &[u64]) {
        self.workers.retain(|id, _| {
            if instance_ids.contains(id) {
                true
            } else {
                remove_worker_metrics(*id);
                false
            }
        });
        self.reported_waiting
            .retain(|id, _| instance_ids.contains(id));
    }
}

/// Un-armed accounting handle for one charged request. Wrapped by the router's
/// RAII permit; split out so the state module stays free of stream types.
pub(crate) struct AdmissionCharge {
    state: Arc<AdmissionState>,
    instance_id: u64,
    admit_seq: u64,
    evict_token: CancellationToken,
}

impl AdmissionCharge {
    pub(crate) fn release(&self) {
        self.state.release(self.instance_id, self.admit_seq);
    }

    /// Worker this charge was admitted on (the final target after any
    /// admission-time retarget).
    pub(crate) fn instance_id(&self) -> u64 {
        self.instance_id
    }

    /// Follow a transport-fallback reselection: move the charged entry to the
    /// worker the request actually dispatched to.
    pub(crate) fn retarget(&mut self, new_instance_id: u64) {
        if new_instance_id == self.instance_id {
            return;
        }
        self.state
            .move_entry(self.instance_id, new_instance_id, self.admit_seq);
        self.instance_id = new_instance_id;
    }

    /// Token fired if this request is later chosen as an eviction victim.
    pub(crate) fn evict_token(&self) -> CancellationToken {
        self.evict_token.clone()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Enforcement knobs, read once from the environment. Enforcement is enabled
/// iff the queue margin is configured; otherwise the registry stays
/// accounting-only.
pub struct AdmissionEnforcement {
    /// `DYN_ADMISSION_QUEUE_MARGIN`: per-worker engine-queue length (in
    /// requests) beyond which admission requires eviction or rejects. One
    /// global constant — the engine's own scheduler bounds the running set.
    pub queue_margin: Option<u64>,
    /// `DYN_ADMISSION_RETRY_AFTER_MS`: retry hint attached to rejections and
    /// evictions (default 1000).
    pub retry_after_ms: u64,
}

impl AdmissionEnforcement {
    fn from_env(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let parse = |raw: Option<String>, name: &str| -> Option<u64> {
            let raw = raw?;
            match raw.trim().parse::<u64>() {
                Ok(v) => Some(v),
                Err(err) => {
                    tracing::warn!(value = raw, %err, "invalid {name}; ignoring");
                    None
                }
            }
        };
        let queue_margin = parse(
            lookup(env_runtime::DYN_ADMISSION_QUEUE_MARGIN),
            env_runtime::DYN_ADMISSION_QUEUE_MARGIN,
        );
        let retry_after_ms = parse(
            lookup(env_runtime::DYN_ADMISSION_RETRY_AFTER_MS),
            env_runtime::DYN_ADMISSION_RETRY_AFTER_MS,
        )
        .unwrap_or(1000);
        Self {
            queue_margin,
            retry_after_ms,
        }
    }

    pub fn enabled(&self) -> bool {
        self.queue_margin.is_some()
    }
}

/// Process-wide enforcement configuration.
pub fn admission_enforcement() -> &'static AdmissionEnforcement {
    static CONFIG: LazyLock<AdmissionEnforcement> =
        LazyLock::new(|| AdmissionEnforcement::from_env(|name| std::env::var(name).ok()));
    &CONFIG
}

// ---------------------------------------------------------------------------
// Typed rejection/eviction errors
// ---------------------------------------------------------------------------

/// Typed admission rejection, findable in the error chain by the HTTP layer.
///
/// Response-body discipline: `Display` (and anything derived from it) must
/// stay free of internal numbers — priorities encode scheduling internals and
/// in-flight/cap values reveal capacity — so nothing sensitive can leak even
/// through generic error-to-details paths. The fields exist for logging and
/// tests; only `retry_after_ms` may reach a client.
#[derive(Debug, Clone)]
pub struct AdmissionRejection {
    pub priority: i32,
    pub queued: u64,
    pub margin: u64,
    pub retry_after_ms: u64,
}

impl std::fmt::Display for AdmissionRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "service over capacity, please retry later")
    }
}

impl std::error::Error for AdmissionRejection {}

/// Overload status code shared with the HTTP layer (`DYN_HTTP_OVERLOAD_STATUS_CODE`,
/// default 529). Read here so synthesized eviction frames carry the same code
/// the HTTP layer would use.
fn overload_status_code_value() -> u16 {
    static CODE: LazyLock<u16> = LazyLock::new(|| {
        std::env::var(crate::config::environment_names::llm::DYN_HTTP_OVERLOAD_STATUS_CODE)
            .ok()
            .and_then(|v| v.trim().parse::<u16>().ok())
            .unwrap_or(529)
    });
    *CODE
}

/// Build the error carried by the synthesized response of an evicted request.
/// The message is `{"message", "code", ...}` JSON: the pre-commit HTTP path
/// parses that shape into a real status code, and the mid-stream path emits it
/// as a structured in-band error frame.
///
/// Response-body discipline: the wording matches a plain rejection and carries
/// no priorities or capacity numbers — a client can distinguish eviction from
/// rejection structurally (in-band frame vs HTTP status), but the body reveals
/// nothing about scheduling internals. Victim details go to logs and metrics.
pub(crate) fn eviction_error() -> crate::error::DynamoError {
    let enforcement = admission_enforcement();
    let code = overload_status_code_value();
    let message = serde_json::json!({
        "message": "service over capacity, please retry later",
        "code": code,
        "retry_after_ms": enforcement.retry_after_ms,
    })
    .to_string();
    crate::error::DynamoError::builder()
        .error_type(crate::error::ErrorType::ResourceExhausted)
        .message(message)
        .build()
}

/// Whether frontend admission tracking is enabled. Defaults to on; set
/// `DYN_ADMISSION_TRACKING=0` (or `false`/`no`/`off`) to disable.
pub(crate) fn admission_tracking_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var(env_runtime::DYN_ADMISSION_TRACKING)
            .ok()
            .and_then(|value| crate::config::parse_bool_opt(&value))
            .unwrap_or(true)
    });
    *ENABLED
}

/// Get or create the shared admission state for an endpoint.
pub async fn get_or_create_admission_state(endpoint: &Endpoint) -> Arc<AdmissionState> {
    let drt = endpoint.drt();
    let registry = drt.admission_states();
    let mut registry = registry.lock().await;

    if let Some(weak) = registry.get(endpoint) {
        if let Some(state) = weak.upgrade() {
            return state;
        } else {
            registry.remove(endpoint);
        }
    }

    let state = Arc::new(AdmissionState::default());
    registry.insert(endpoint.clone(), Arc::downgrade(&state));
    state
}

// ---------------------------------------------------------------------------
// Prometheus export
// ---------------------------------------------------------------------------

struct AdmissionMetrics {
    inflight: IntGaugeVec,
    admitted_total: IntCounterVec,
    evictions_total: IntCounterVec,
    rejections_total: IntCounterVec,
    reselects_total: IntCounterVec,
}

static ADMISSION_METRICS: LazyLock<AdmissionMetrics> = LazyLock::new(|| {
    AdmissionMetrics {
    inflight: IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ADMISSION_INFLIGHT
            ),
            "Frontend-tracked in-flight requests per worker (admission registry)",
        ),
        &[labels::WORKER_ID],
    )
    .expect("failed to create worker_admission_inflight gauge"),
    admitted_total: IntCounterVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ADMISSION_TOTAL
            ),
            "Total requests dispatched to each worker through the admission registry",
        ),
        &[labels::WORKER_ID],
    )
    .expect("failed to create worker_admission_total counter"),
    evictions_total: IntCounterVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ADMISSION_EVICTIONS
            ),
            "In-flight requests evicted per worker to admit higher-priority work",
        ),
        &[labels::WORKER_ID],
    )
    .expect("failed to create worker_admission_evictions counter"),
    rejections_total: IntCounterVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ADMISSION_REJECTIONS
            ),
            "Requests rejected at admission per worker (at cap, no eviction victim)",
        ),
        &[labels::WORKER_ID],
    )
    .expect("failed to create worker_admission_rejections counter"),
    reselects_total: IntCounterVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_ADMISSION_RESELECTS
            ),
            "Requests routed around this worker because its engine queue was at the admission margin",
        ),
        &[labels::WORKER_ID],
    )
    .expect("failed to create worker_admission_reselects counter"),
}
});

fn observe_charge(instance_id: u64, inflight: usize) {
    let m = &*ADMISSION_METRICS;
    let id = instance_id.to_string();
    m.inflight.with_label_values(&[&id]).set(inflight as i64);
    m.admitted_total.with_label_values(&[&id]).inc();
}

fn observe_release(instance_id: u64, inflight: usize) {
    ADMISSION_METRICS
        .inflight
        .with_label_values(&[&instance_id.to_string()])
        .set(inflight as i64);
}

fn observe_move(instance_id: u64, inflight: usize) {
    // A moved entry is not a new admission; only the gauge changes.
    ADMISSION_METRICS
        .inflight
        .with_label_values(&[&instance_id.to_string()])
        .set(inflight as i64);
}

fn observe_eviction(instance_id: u64) {
    ADMISSION_METRICS
        .evictions_total
        .with_label_values(&[&instance_id.to_string()])
        .inc();
}

fn observe_rejection(instance_id: u64) {
    ADMISSION_METRICS
        .rejections_total
        .with_label_values(&[&instance_id.to_string()])
        .inc();
}

/// Record that the router abandoned a selection of `instance_id` before
/// dispatch because the worker's engine queue was at the admission margin.
pub(crate) fn observe_reselect(instance_id: u64) {
    ADMISSION_METRICS
        .reselects_total
        .with_label_values(&[&instance_id.to_string()])
        .inc();
}

fn remove_worker_metrics(instance_id: u64) {
    let m = &*ADMISSION_METRICS;
    let id = instance_id.to_string();
    let _ = m.inflight.remove_label_values(&[&id]);
    let _ = m.admitted_total.remove_label_values(&[&id]);
    let _ = m.evictions_total.remove_label_values(&[&id]);
    let _ = m.rejections_total.remove_label_values(&[&id]);
    let _ = m.reselects_total.remove_label_values(&[&id]);
}

/// Register the admission registry metrics with a Prometheus registry.
/// Called during frontend HTTP service setup.
pub fn register_admission_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*ADMISSION_METRICS;
    registry.register(Box::new(m.inflight.clone()))?;
    registry.register(Box::new(m.admitted_total.clone()))?;
    registry.register(Box::new(m.evictions_total.clone()))?;
    registry.register(Box::new(m.rejections_total.clone()))?;
    registry.register(Box::new(m.reselects_total.clone()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::Controller;

    fn ctx() -> Arc<dyn AsyncEngineContext> {
        Arc::new(Controller::default())
    }

    fn state() -> Arc<AdmissionState> {
        Arc::new(AdmissionState::default())
    }

    #[test]
    fn charge_and_release_track_per_worker_counts() {
        let state = state();
        let a = state.charge(1, "r1".into(), 0, ctx());
        let b = state.charge(1, "r2".into(), 0, ctx());
        let c = state.charge(2, "r3".into(), 0, ctx());
        assert_eq!(state.inflight(1), 2);
        assert_eq!(state.inflight(2), 1);

        a.release();
        assert_eq!(state.inflight(1), 1);
        b.release();
        c.release();
        assert_eq!(state.inflight(1), 0);
        assert_eq!(state.inflight(2), 0);
    }

    #[test]
    fn double_release_is_idempotent() {
        let state = state();
        let a = state.charge(1, "r1".into(), 0, ctx());
        let _b = state.charge(1, "r2".into(), 0, ctx());
        a.release();
        a.release();
        assert_eq!(state.inflight(1), 1);
    }

    #[test]
    fn inflight_at_or_above_counts_by_priority() {
        let state = state();
        let _a = state.charge(1, "bg".into(), i32::MIN, ctx());
        let _b = state.charge(1, "flex".into(), -3600, ctx());
        let _c = state.charge(1, "rt".into(), 0, ctx());
        assert_eq!(state.inflight_at_or_above(1, 0), 1);
        assert_eq!(state.inflight_at_or_above(1, -3600), 2);
        assert_eq!(state.inflight_at_or_above(1, i32::MIN), 3);
        assert_eq!(state.inflight_at_or_above(1, 1), 0);
        assert_eq!(state.inflight_at_or_above(2, i32::MIN), 0);
    }

    #[test]
    fn admit_seq_is_monotone_across_workers() {
        let state = state();
        let _a = state.charge(1, "r1".into(), 0, ctx());
        let _b = state.charge(2, "r2".into(), 0, ctx());
        let seq1 = *state
            .workers
            .get(&1)
            .unwrap()
            .lock()
            .unwrap()
            .keys()
            .next()
            .unwrap();
        let seq2 = *state
            .workers
            .get(&2)
            .unwrap()
            .lock()
            .unwrap()
            .keys()
            .next()
            .unwrap();
        assert!(seq2 > seq1);
    }

    #[test]
    fn redispatch_with_same_request_id_keeps_entries_distinct() {
        // Migration retries reuse the frontend request id; the registry must
        // not confuse the retry's entry with the original's release.
        let state = state();
        let first = state.charge(1, "r1".into(), 0, ctx());
        let _retry = state.charge(1, "r1".into(), 0, ctx());
        assert_eq!(state.inflight(1), 2);
        first.release();
        assert_eq!(state.inflight(1), 1);
    }

    #[test]
    fn retain_drops_departed_workers_only() {
        let state = state();
        let _a = state.charge(1, "r1".into(), 0, ctx());
        let _b = state.charge(2, "r2".into(), 0, ctx());
        state.report_queue_depth(1, 4);
        state.retain(&[2]);
        assert_eq!(state.inflight(1), 0);
        assert_eq!(state.inflight(2), 1);
        assert_eq!(state.reported_queue(1), None);
    }

    #[test]
    fn release_after_retain_is_harmless() {
        let state = state();
        let a = state.charge(1, "r1".into(), 0, ctx());
        state.retain(&[]);
        a.release();
        assert_eq!(state.inflight(1), 0);
    }

    // ---- enforcement ----

    fn admit(
        state: &Arc<AdmissionState>,
        preferred: u64,
        retarget: Option<&[u64]>,
        priority: i32,
    ) -> (AdmissionDecision, Option<EvictedVictim>) {
        state.admit(preferred, retarget, "req".into(), priority, ctx())
    }

    fn assert_admitted_on(decision: AdmissionDecision, worker: u64) -> AdmissionCharge {
        match decision {
            AdmissionDecision::Admit(charge) => {
                assert_eq!(charge.instance_id(), worker);
                charge
            }
            AdmissionDecision::Reject { queued, margin } => {
                panic!("expected admit on {worker}, got reject (queued={queued}, margin={margin})")
            }
        }
    }

    #[test]
    fn unreported_worker_admits_freely() {
        let state = state();
        state.set_queue_margin(Some(0));
        let (decision, victim) = admit(&state, 1, None, i32::MIN);
        assert_admitted_on(decision, 1);
        assert!(victim.is_none());
    }

    #[test]
    fn saturated_instances_lists_workers_at_the_margin() {
        let state = state();
        state.report_queue_depth(1, 5);
        assert!(
            state.saturated_instances().is_empty(),
            "no margin, no saturation"
        );
        state.set_queue_margin(Some(2));
        state.report_queue_depth(1, 2);
        state.report_queue_depth(2, 1);
        state.report_queue_depth(3, 5);
        let mut saturated = state.saturated_instances();
        saturated.sort_unstable();
        assert_eq!(saturated, vec![1, 3]);
    }

    #[test]
    fn no_margin_means_no_enforcement() {
        let state = state();
        state.report_queue_depth(1, 1_000_000);
        assert!(!state.enforcement_active());
        let (decision, victim) = admit(&state, 1, None, i32::MIN);
        assert_admitted_on(decision, 1);
        assert!(victim.is_none());
    }

    #[test]
    fn admits_below_margin_and_rejects_at_margin_same_priority() {
        let state = state();
        state.set_queue_margin(Some(2));
        state.report_queue_depth(1, 1);
        let _a = assert_admitted_on(admit(&state, 1, None, 0).0, 1);

        state.report_queue_depth(1, 2);
        match admit(&state, 1, None, 0).0 {
            AdmissionDecision::Reject { queued, margin } => {
                assert_eq!((queued, margin), (2, 2));
            }
            AdmissionDecision::Admit(_) => panic!("expected rejection at margin"),
        }
        assert_eq!(state.inflight(1), 1);
    }

    #[test]
    fn full_queue_with_empty_registry_rejects() {
        // The engine can report queued work the frontend never tracked (e.g.
        // after a frontend restart); with no in-flight victims the request
        // must reject rather than evict.
        let state = state();
        state.set_queue_margin(Some(1));
        state.report_queue_depth(1, 5);
        match admit(&state, 1, None, 0).0 {
            AdmissionDecision::Reject { queued, margin } => {
                assert_eq!((queued, margin), (5, 1));
            }
            AdmissionDecision::Admit(_) => panic!("expected rejection"),
        }
    }

    #[test]
    fn evicts_lowest_priority_victim_at_margin() {
        let state = state();
        state.set_queue_margin(Some(1));
        let _bg = state.charge(1, "bg".into(), -100, ctx());
        let _flex = state.charge(1, "flex".into(), -10, ctx());
        state.report_queue_depth(1, 1);

        let (decision, victim) = admit(&state, 1, None, 0);
        assert_admitted_on(decision, 1);
        let victim = victim.expect("expected an eviction");
        assert_eq!(victim.priority, -100);
        assert_eq!(victim.worker, 1);
        assert_eq!(state.inflight(1), 2, "slot transferred, not leaked");
    }

    #[test]
    fn eviction_tie_breaks_most_recently_admitted() {
        let state = state();
        state.set_queue_margin(Some(0));
        let older = state.charge(1, "older".into(), -5, ctx());
        let _newer = state.charge(1, "newer".into(), -5, ctx());
        state.report_queue_depth(1, 0);
        let older_seq = older.admit_seq;

        let (_, victim) = admit(&state, 1, None, 0);
        let victim = victim.expect("expected an eviction");
        // The most recently admitted of the two equal-priority entries dies;
        // the older one survives.
        let worker = state.workers.get(&1).unwrap();
        let entries = worker.lock().unwrap();
        assert!(entries.contains_key(&older_seq), "older entry survives");
        assert_eq!(victim.priority, -5);
    }

    #[test]
    fn equal_priority_is_not_a_victim() {
        let state = state();
        state.set_queue_margin(Some(0));
        let _a = state.charge(1, "a".into(), -7, ctx());
        state.report_queue_depth(1, 0);
        match admit(&state, 1, None, -7).0 {
            AdmissionDecision::Reject { .. } => {}
            AdmissionDecision::Admit(_) => panic!("equal priority must not evict"),
        }
    }

    #[test]
    fn retargets_to_worker_with_queue_headroom_before_evicting() {
        let state = state();
        state.set_queue_margin(Some(1));
        let _bg = state.charge(1, "bg".into(), -100, ctx());
        state.report_queue_depth(1, 1);
        state.report_queue_depth(2, 0);

        let (decision, victim) = admit(&state, 1, Some(&[1, 2]), 0);
        assert_admitted_on(decision, 2);
        assert!(victim.is_none(), "headroom elsewhere must not evict");
    }

    #[test]
    fn pinned_mode_does_not_retarget() {
        let state = state();
        state.set_queue_margin(Some(1));
        state.report_queue_depth(1, 1);
        state.report_queue_depth(2, 0);
        match admit(&state, 1, None, 0).0 {
            AdmissionDecision::Reject { .. } => {}
            AdmissionDecision::Admit(charge) => {
                panic!("pinned admission must not move to {}", charge.instance_id())
            }
        }
    }

    #[test]
    fn eviction_fires_the_victims_token_and_frees_its_release() {
        let state = state();
        state.set_queue_margin(Some(0));
        let victim_charge = state.charge(1, "victim".into(), -1, ctx());
        state.report_queue_depth(1, 0);
        let victim_token = victim_charge.evict_token();
        assert!(!victim_token.is_cancelled());

        let (_, victim) = admit(&state, 1, None, 0);
        assert!(victim.is_some());
        assert!(victim_token.is_cancelled());
        assert_eq!(state.inflight(1), 1);

        // The victim's own release (stream teardown) is a harmless no-op.
        victim_charge.release();
        assert_eq!(state.inflight(1), 1);
    }

    #[test]
    fn queue_margin_env_parsing() {
        let enf = AdmissionEnforcement::from_env(|name| match name {
            env_runtime::DYN_ADMISSION_QUEUE_MARGIN => Some("8".to_string()),
            _ => None,
        });
        assert_eq!(enf.queue_margin, Some(8));
        assert_eq!(enf.retry_after_ms, 1000);
        assert!(enf.enabled());

        let enf = AdmissionEnforcement::from_env(|name| match name {
            env_runtime::DYN_ADMISSION_RETRY_AFTER_MS => Some("250".to_string()),
            _ => None,
        });
        assert_eq!(enf.queue_margin, None);
        assert_eq!(enf.retry_after_ms, 250);
        assert!(!enf.enabled());
    }
}
