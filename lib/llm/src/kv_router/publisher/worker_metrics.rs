// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::config::environment_names::router as env_router;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

const PUBLISH_DEBOUNCE: Duration = Duration::from_millis(1);
const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(30);

fn heartbeat_from_env() -> Option<Duration> {
    heartbeat_from_lookup(|name| std::env::var(name).ok())
}

fn heartbeat_from_lookup(get: impl FnOnce(&str) -> Option<String>) -> Option<Duration> {
    let Some(raw) = get(env_router::DYN_WORKER_METRICS_HEARTBEAT_SECS) else {
        return Some(DEFAULT_HEARTBEAT);
    };
    match raw.trim().parse::<u64>() {
        Ok(0) => None,
        Ok(secs) => Some(Duration::from_secs(secs)),
        Err(_) => {
            tracing::warn!(
                value = %raw,
                default_secs = DEFAULT_HEARTBEAT.as_secs(),
                "Invalid {}; using default",
                env_router::DYN_WORKER_METRICS_HEARTBEAT_SECS,
            );
            Some(DEFAULT_HEARTBEAT)
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
    num_waiting_reqs: Option<u64>,
    load_report_revision: u64,
}

struct PendingMetrics {
    metrics: WorkerMetrics,
    deadline: tokio::time::Instant,
}

struct WorkerMetricsDebouncer {
    debounce: Duration,
    last_metrics: HashMap<DpRank, WorkerMetrics>,
    pending: HashMap<DpRank, PendingMetrics>,
}

impl WorkerMetricsDebouncer {
    fn new(debounce: Duration) -> Self {
        Self {
            debounce,
            last_metrics: HashMap::new(),
            pending: HashMap::new(),
        }
    }

    fn observe(
        &mut self,
        metrics_by_rank: &HashMap<DpRank, WorkerMetrics>,
        now: tokio::time::Instant,
    ) {
        for (&dp_rank, metrics) in metrics_by_rank {
            if self.last_metrics.get(&dp_rank) == Some(metrics) {
                continue;
            }

            self.last_metrics.insert(dp_rank, metrics.clone());
            self.pending.insert(
                dp_rank,
                PendingMetrics {
                    metrics: metrics.clone(),
                    deadline: now + self.debounce,
                },
            );
        }
    }

    fn next_deadline(&self) -> Option<tokio::time::Instant> {
        self.pending.values().map(|pending| pending.deadline).min()
    }

    fn take_due(&mut self, now: tokio::time::Instant) -> Vec<WorkerMetrics> {
        let due_ranks = self
            .pending
            .iter()
            .filter_map(|(&dp_rank, pending)| (pending.deadline <= now).then_some(dp_rank))
            .collect::<Vec<_>>();

        due_ranks
            .into_iter()
            .filter_map(|dp_rank| self.pending.remove(&dp_rank))
            .map(|pending| pending.metrics)
            .collect()
    }
}

#[async_trait::async_trait]
pub(super) trait WorkerMetricsSink: Send + 'static {
    async fn publish(&self, active_load: ActiveLoad) -> Result<()>;
}

#[async_trait::async_trait]
impl WorkerMetricsSink for EventPublisher {
    async fn publish(&self, active_load: ActiveLoad) -> Result<()> {
        EventPublisher::publish(self, &active_load).await
    }
}

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<HashMap<DpRank, WorkerMetrics>>,
    rx: tokio::sync::watch::Receiver<HashMap<DpRank, WorkerMetrics>>,
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        Ok(Self { tx, rx })
    }

    pub fn publish(
        &self,
        dp_rank: Option<DpRank>,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
        num_waiting_reqs: Option<u64>,
    ) -> Result<()> {
        if active_decode_blocks.is_none() && kv_used_blocks.is_none() && num_waiting_reqs.is_none()
        {
            anyhow::bail!("worker metrics publish requires at least one load metric");
        }

        let dp_rank = dp_rank.unwrap_or(0);
        let mut load_report_revision = 0;
        self.tx.send_modify(|metrics_by_rank| {
            load_report_revision = metrics_by_rank
                .get(&dp_rank)
                .map_or(1, |previous| previous.load_report_revision.wrapping_add(1));
            metrics_by_rank.insert(
                dp_rank,
                WorkerMetrics {
                    dp_rank,
                    active_decode_blocks,
                    kv_used_blocks,
                    num_waiting_reqs,
                    load_report_revision,
                },
            );
        });
        tracing::trace!(
            dp_rank,
            load_report_revision,
            active_decode_blocks = ?active_decode_blocks,
            kv_used_blocks = ?kv_used_blocks,
            num_waiting_reqs = ?num_waiting_reqs,
            "Publishing worker metrics"
        );
        Ok(())
    }

    pub async fn create_endpoint(&self, endpoint: Endpoint) -> Result<()> {
        let worker_id = endpoint.drt().connection_id();
        let event_publisher = EventPublisher::for_endpoint(&endpoint, KV_METRICS_SUBJECT).await?;
        self.start_metrics_publishing(event_publisher, worker_id);
        Ok(())
    }

    pub(super) fn start_metrics_publishing(&self, event_publisher: EventPublisher, worker_id: u64) {
        self.start_metrics_publishing_with(event_publisher, worker_id);
    }

    pub(super) fn start_metrics_publishing_with<S>(&self, sink: S, worker_id: u64)
    where
        S: WorkerMetricsSink,
    {
        let metrics_rx = self.rx.clone();
        let heartbeat = heartbeat_from_env();

        tokio::spawn(async move {
            let mut rx = metrics_rx;
            let mut debouncer = WorkerMetricsDebouncer::new(PUBLISH_DEBOUNCE);
            let publish_timer = tokio::time::sleep(tokio::time::Duration::ZERO);
            tokio::pin!(publish_timer);
            let mut heartbeat_tick = heartbeat.map(|period| {
                let mut tick =
                    tokio::time::interval_at(tokio::time::Instant::now() + period, period);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                tick
            });

            loop {
                tokio::select! {
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping event-plane background task"
                            );
                            break;
                        }

                        let now = tokio::time::Instant::now();
                        debouncer.observe(&rx.borrow_and_update(), now);
                        if let Some(deadline) = debouncer.next_deadline() {
                            publish_timer.as_mut().reset(deadline);
                        }
                    }
                    _ = async {
                        match heartbeat_tick.as_mut() {
                            Some(tick) => { tick.tick().await; }
                            None => std::future::pending().await,
                        }
                    } => {
                        let metrics = rx.borrow().clone();
                        for metrics in metrics.values() {
                            if let Err(error) = sink.publish(ActiveLoad { worker_id, dp_rank: metrics.dp_rank, active_decode_blocks: metrics.active_decode_blocks, active_prefill_tokens: None, kv_used_blocks: metrics.kv_used_blocks, num_waiting_reqs: metrics.num_waiting_reqs, load_report_revision: Some(metrics.load_report_revision) }).await {
                                tracing::warn!(%error, "Failed to publish worker metrics heartbeat");
                            }
                        }
                    }
                    _ = &mut publish_timer, if debouncer.next_deadline().is_some() => {
                        for metrics in debouncer.take_due(tokio::time::Instant::now()) {
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.dp_rank,
                                active_decode_blocks: metrics.active_decode_blocks,
                                active_prefill_tokens: None,
                                kv_used_blocks: metrics.kv_used_blocks,
                                num_waiting_reqs: metrics.num_waiting_reqs,
                                load_report_revision: Some(metrics.load_report_revision),
                            };

                            if let Err(e) = sink.publish(active_load).await {
                                tracing::warn!("Failed to publish metrics: {}", e);
                            }
                        }

                        if let Some(deadline) = debouncer.next_deadline() {
                            publish_timer.as_mut().reset(deadline);
                        }
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_env_parses_default_disable_and_invalid() {
        assert_eq!(heartbeat_from_lookup(|_| None), Some(DEFAULT_HEARTBEAT));
        assert_eq!(heartbeat_from_lookup(|_| Some("0".into())), None);
        assert_eq!(
            heartbeat_from_lookup(|_| Some("5".into())),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            heartbeat_from_lookup(|_| Some("nope".into())),
            Some(DEFAULT_HEARTBEAT)
        );
    }

    #[test]
    fn retains_the_latest_metrics_for_every_dp_rank() {
        let publisher = WorkerMetricsPublisher::new().unwrap();
        publisher.publish(Some(0), None, Some(10), Some(1)).unwrap();
        publisher.publish(Some(1), None, Some(20), Some(2)).unwrap();
        publisher.publish(Some(0), None, Some(11), Some(3)).unwrap();

        let metrics = publisher.rx.borrow();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[&0].kv_used_blocks, Some(11));
        assert_eq!(metrics[&0].num_waiting_reqs, Some(3));
        assert_eq!(metrics[&0].load_report_revision, 2);
        assert_eq!(metrics[&1].kv_used_blocks, Some(20));
        assert_eq!(metrics[&1].num_waiting_reqs, Some(2));
        assert_eq!(metrics[&1].load_report_revision, 1);
    }

    #[test]
    fn identical_observations_advance_the_load_report_revision() {
        let publisher = WorkerMetricsPublisher::new().unwrap();
        publisher.publish(Some(0), None, Some(10), Some(1)).unwrap();
        let first = publisher.rx.borrow()[&0].load_report_revision;

        publisher.publish(Some(0), None, Some(10), Some(1)).unwrap();
        let second = publisher.rx.borrow()[&0].load_report_revision;

        assert_ne!(first, second);
    }
}
