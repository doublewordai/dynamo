// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local admission registry: every request dispatched to a worker is
//! recorded here for the lifetime of its response stream, together with its
//! scheduling priority, admission order, and response-stream context.
//!
//! This module is accounting only. It gives the frontend an authoritative
//! per-worker view of the work it has accepted — the prerequisite for bounding
//! the accepted set and for choosing eviction victims by priority. The counts
//! are exported as Prometheus gauges so the accounting can be validated against
//! worker-reported load before any enforcement keys off it.
//!
//! Keyed per worker instance. One state is shared per endpoint via
//! [`get_or_create_admission_state`], mirroring the routing occupancy registry.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use dashmap::DashMap;
use prometheus::{IntCounterVec, IntGaugeVec, Opts};

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
    /// Context of the dispatched response stream. Held so an evictor can
    /// terminate the request on the worker.
    pub context: Arc<dyn AsyncEngineContext>,
}

/// Per-endpoint registry of in-flight requests, keyed by worker instance id.
/// Each worker's entries are keyed by a monotone admission sequence number,
/// shared across the endpoint's workers so it also orders entries within a
/// priority for victim tie-breaking.
#[derive(Default)]
pub struct AdmissionState {
    workers: DashMap<u64, StdMutex<HashMap<u64, AdmissionEntry>>>,
    admit_seq: AtomicU64,
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
                },
            );
            entries.len()
        };
        observe_charge(instance_id, inflight);
        AdmissionCharge {
            state: self.clone(),
            instance_id,
            admit_seq,
        }
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

    /// Drop tracking for workers that no longer exist in discovery.
    pub(crate) fn retain(&self, instance_ids: &[u64]) {
        self.workers.retain(|id, _| {
            if instance_ids.contains(id) {
                true
            } else {
                remove_worker_metrics(*id);
                false
            }
        });
    }
}

/// Un-armed accounting handle for one charged request. Wrapped by the router's
/// RAII permit; split out so the state module stays free of stream types.
pub(crate) struct AdmissionCharge {
    state: Arc<AdmissionState>,
    instance_id: u64,
    admit_seq: u64,
}

impl AdmissionCharge {
    pub(crate) fn release(&self) {
        self.state.release(self.instance_id, self.admit_seq);
    }
}

/// Whether frontend admission tracking is enabled. Defaults to on; set
/// `DYN_ADMISSION_TRACKING=0` (or `false`/`no`/`off`) to disable.
pub(crate) fn admission_tracking_enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(
            || match std::env::var(env_runtime::DYN_ADMISSION_TRACKING) {
                Ok(value) => !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                ),
                Err(_) => true,
            },
        );
    *ENABLED
}

/// Get or create the shared admission state for an endpoint.
pub(crate) async fn get_or_create_admission_state(endpoint: &Endpoint) -> Arc<AdmissionState> {
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
}

static ADMISSION_METRICS: LazyLock<AdmissionMetrics> = LazyLock::new(|| AdmissionMetrics {
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

fn remove_worker_metrics(instance_id: u64) {
    let m = &*ADMISSION_METRICS;
    let id = instance_id.to_string();
    let _ = m.inflight.remove_label_values(&[&id]);
    let _ = m.admitted_total.remove_label_values(&[&id]);
}

/// Register the admission registry metrics with a Prometheus registry.
/// Called during frontend HTTP service setup.
pub fn register_admission_metrics(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    let m = &*ADMISSION_METRICS;
    registry.register(Box::new(m.inflight.clone()))?;
    registry.register(Box::new(m.admitted_total.clone()))?;
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
        state.retain(&[2]);
        assert_eq!(state.inflight(1), 0);
        assert_eq!(state.inflight(2), 1);
    }

    #[test]
    fn release_after_retain_is_harmless() {
        let state = state();
        let a = state.charge(1, "r1".into(), 0, ctx());
        state.retain(&[]);
        a.release();
        assert_eq!(state.inflight(1), 0);
    }
}
