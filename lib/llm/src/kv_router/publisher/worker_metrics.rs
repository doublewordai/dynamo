// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Result;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::{Component, Namespace};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
    num_waiting_reqs: Option<u64>,
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

        let metrics = WorkerMetrics {
            dp_rank: dp_rank.unwrap_or(0),
            active_decode_blocks,
            kv_used_blocks,
            num_waiting_reqs,
        };
        tracing::trace!(
            "Publish metrics: dp_rank={}, active_decode_blocks={:?}, kv_used_blocks={:?}, num_waiting_reqs={:?}",
            metrics.dp_rank,
            metrics.active_decode_blocks,
            metrics.kv_used_blocks,
            metrics.num_waiting_reqs
        );
        self.tx.send_modify(|by_rank| {
            by_rank.insert(metrics.dp_rank, metrics);
        });
        Ok(())
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let worker_id = component.drt().connection_id();
        self.start_nats_metrics_publishing(component.namespace().clone(), worker_id);
        Ok(())
    }

    pub(super) fn start_nats_metrics_publishing(&self, namespace: Namespace, worker_id: u64) {
        let nats_rx = self.rx.clone();

        tokio::spawn(async move {
            let event_publisher =
                match EventPublisher::for_namespace(&namespace, KV_METRICS_SUBJECT).await {
                    Ok(publisher) => publisher,
                    Err(e) => {
                        tracing::error!("Failed to create metrics publisher: {}", e);
                        return;
                    }
                };

            let mut rx = nats_rx;
            let mut last_metrics: HashMap<DpRank, WorkerMetrics> = HashMap::new();
            let mut pending_publish: HashMap<DpRank, (WorkerMetrics, tokio::time::Instant)> =
                HashMap::new();
            let publish_timer = tokio::time::sleep(tokio::time::Duration::ZERO);
            tokio::pin!(publish_timer);

            loop {
                tokio::select! {
                    result = rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping NATS background task"
                            );
                            break;
                        }

                        let metrics_by_rank = rx.borrow_and_update().clone();
                        let deadline = tokio::time::Instant::now()
                            + tokio::time::Duration::from_millis(1);
                        let mut changed = false;
                        for (dp_rank, metrics) in metrics_by_rank {
                            if last_metrics.get(&dp_rank) == Some(&metrics) {
                                continue;
                            }
                            pending_publish.insert(dp_rank, (metrics.clone(), deadline));
                            last_metrics.insert(dp_rank, metrics);
                            changed = true;
                        }
                        if changed {
                            let next_deadline = pending_publish
                                .values()
                                .map(|(_, deadline)| *deadline)
                                .min()
                                .expect("changed metrics create a pending publish");
                            publish_timer.as_mut().reset(next_deadline);
                        }
                    }
                    _ = &mut publish_timer, if !pending_publish.is_empty() => {
                        let now = tokio::time::Instant::now();
                        let mut ready_ranks = pending_publish
                            .iter()
                            .filter_map(|(dp_rank, (_, deadline))| (*deadline <= now).then_some(*dp_rank))
                            .collect::<Vec<_>>();
                        ready_ranks.sort_unstable();
                        for dp_rank in ready_ranks {
                            let (metrics, _) = pending_publish
                                .remove(&dp_rank)
                                .expect("ready rank is pending");
                            let active_load = ActiveLoad {
                                worker_id,
                                dp_rank: metrics.dp_rank,
                                active_decode_blocks: metrics.active_decode_blocks,
                                active_prefill_tokens: None,
                                kv_used_blocks: metrics.kv_used_blocks,
                                num_waiting_reqs: metrics.num_waiting_reqs,
                            };

                            if let Err(e) = event_publisher.publish(&active_load).await {
                                tracing::warn!("Failed to publish metrics: {}", e);
                            }
                        }
                        if let Some(next_deadline) = pending_publish
                            .values()
                            .map(|(_, deadline)| *deadline)
                            .min()
                        {
                            publish_timer.as_mut().reset(next_deadline);
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
    fn retains_the_latest_metrics_for_every_dp_rank() {
        let publisher = WorkerMetricsPublisher::new().unwrap();
        publisher.publish(Some(0), None, Some(10), Some(1)).unwrap();
        publisher.publish(Some(1), None, Some(20), Some(2)).unwrap();
        publisher.publish(Some(0), None, Some(11), Some(3)).unwrap();

        let metrics = publisher.rx.borrow();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[&0].kv_used_blocks, Some(11));
        assert_eq!(metrics[&0].num_waiting_reqs, Some(3));
        assert_eq!(metrics[&1].kv_used_blocks, Some(20));
        assert_eq!(metrics[&1].num_waiting_reqs, Some(2));
    }
}
