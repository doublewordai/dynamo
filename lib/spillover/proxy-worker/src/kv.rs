// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishing virtual-cache events to the router.
//!
//! `dw_proxy_core::vcache::VirtualCache` produces raw `(block_hash, tokens_hash)`
//! events; the router consumes `dynamo_kv_router` `KvCacheEvent`s through a
//! `KvEventPublisher` (`lib/llm/src/kv_router/publisher/mod.rs`). This module owns
//! the translation and the publisher slot that `Worker` fills in after
//! `LLMEngine::kv_event_sources` returns (`lib/backend-common/src/publisher.rs`).

use std::sync::{Arc, Mutex};

use dw_proxy_core::vcache::CacheEvent;
use dynamo_backend_common::KvEventPublisher;
use dynamo_kv_router::protocols::KvCacheEvent;

/// The subset of [`KvEventPublisher`] the sink depends on.
///
/// A real publisher needs a Dynamo runtime, so tests substitute a fake to assert
/// ordering and delivery accounting.
trait EventPublisher: Send + Sync {
    fn next_event_id(&self) -> u64;
    /// `Err(())` marks a closed/failed publisher whose events were dropped.
    fn publish_batch(&self, events: Vec<KvCacheEvent>) -> Result<(), ()>;
}

impl EventPublisher for KvEventPublisher {
    fn next_event_id(&self) -> u64 {
        KvEventPublisher::next_event_id(self)
    }

    fn publish_batch(&self, events: Vec<KvCacheEvent>) -> Result<(), ()> {
        KvEventPublisher::publish_batch(self, events).map_err(|_| ())
    }
}

/// The publisher slot plus the events produced before it existed.
struct SinkState {
    publisher: Option<Arc<dyn EventPublisher>>,
    /// Events buffered while `publisher` is `None`. `publish` cannot undo the
    /// vcache mutation that produced them, so they are replayed by `set`
    /// instead of being dropped.
    pending: Vec<CacheEvent>,
}

/// The publisher slot plus the DP rank the events are stamped with.
pub struct EventSink {
    dp_rank: u32,
    state: Mutex<SinkState>,
}

/// Counts of events handed to the router, by kind. Returned by
/// [`EventSink::publish`] so the caller can record the
/// `dynamo_component_proxy_kv_events_total` metric without the sink depending
/// on the metrics type.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PublishedEvents {
    pub stored: u64,
    pub removed: u64,
    pub cleared: u64,
}

impl PublishedEvents {
    /// `(kind, count)` pairs for the metric's `kind` label, skipping zeroes.
    pub fn kinds(&self) -> [(&'static str, u64); 3] {
        [
            ("stored", self.stored),
            ("removed", self.removed),
            ("cleared", self.cleared),
        ]
    }
}

impl EventSink {
    pub fn new(dp_rank: u32) -> Self {
        Self {
            dp_rank,
            state: Mutex::new(SinkState {
                publisher: None,
                pending: Vec::new(),
            }),
        }
    }

    /// Store the publisher built by `Worker`. Called once, from the
    /// `KvEventSource::Push` `on_ready` callback. Events produced before this
    /// call are replayed now.
    pub fn set(&self, publisher: Arc<KvEventPublisher>) {
        self.set_publisher(publisher as Arc<dyn EventPublisher>);
    }

    fn set_publisher(&self, publisher: Arc<dyn EventPublisher>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.publisher = Some(publisher.clone());
        let pending = std::mem::take(&mut state.pending);
        if pending.is_empty() {
            return;
        }
        let (batch, _counts) = build_batch(pending, self.dp_rank, publisher.as_ref());
        if publisher.publish_batch(batch).is_err() {
            tracing::warn!("dropping buffered virtual-cache events: KV publisher closed");
        }
    }

    /// Publish a batch of cache events.
    ///
    /// While no publisher exists yet, events are buffered and replayed when
    /// `set` installs one; the returned counts are zero because nothing reached
    /// the router yet. When `publish_batch` fails the events are dropped and the
    /// returned counts are zero, so the metric only reflects delivered events.
    pub fn publish(&self, events: Vec<CacheEvent>) -> PublishedEvents {
        if events.is_empty() {
            return PublishedEvents::default();
        }
        // One critical section for id assignment and the send: `next_event_id`
        // and `publish_batch` are not atomic together, so without this guard
        // concurrent publishers could send ids out of order.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(publisher) = state.publisher.clone() else {
            state.pending.extend(events);
            return PublishedEvents::default();
        };
        let (batch, counts) = build_batch(events, self.dp_rank, publisher.as_ref());
        match publisher.publish_batch(batch) {
            Ok(()) => counts,
            Err(()) => {
                tracing::warn!("dropping virtual-cache events: KV publisher closed");
                PublishedEvents::default()
            }
        }
    }
}

/// Translate a list of virtual-cache events into one router batch, assigning the
/// next consecutive ids and counting each kind.
fn build_batch(
    events: Vec<CacheEvent>,
    dp_rank: u32,
    publisher: &dyn EventPublisher,
) -> (Vec<KvCacheEvent>, PublishedEvents) {
    let mut counts = PublishedEvents::default();
    let batch = events
        .into_iter()
        .map(|event| {
            match &event {
                CacheEvent::Stored { .. } => counts.stored += 1,
                CacheEvent::Removed { .. } => counts.removed += 1,
                CacheEvent::Cleared => counts.cleared += 1,
            }
            to_kv_event(event, dp_rank, publisher.next_event_id())
        })
        .collect();
    (batch, counts)
}

/// Translate one virtual-cache event into the router's wire event.
pub fn to_kv_event(event: CacheEvent, dp_rank: u32, event_id: u64) -> KvCacheEvent {
    event.to_router_event(event_id, dp_rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::time::Duration;

    use dw_proxy_core::vcache::StoredBlock;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoredBlockData, LocalBlockHash,
    };

    fn stored() -> CacheEvent {
        CacheEvent::Stored {
            parent_hash: None,
            blocks: vec![StoredBlock {
                tokens_hash: 1,
                block_hash: 2,
            }],
        }
    }

    /// Records delivered batches and flags overlapping `publish_batch` calls.
    #[derive(Default)]
    struct FakePublisher {
        ids: AtomicU64,
        in_flight: AtomicUsize,
        overlapped: AtomicBool,
        fail: AtomicBool,
        batches: Mutex<Vec<Vec<u64>>>,
    }

    impl FakePublisher {
        fn recorded(&self) -> Vec<Vec<u64>> {
            self.batches.lock().unwrap().clone()
        }
    }

    impl EventPublisher for FakePublisher {
        fn next_event_id(&self) -> u64 {
            self.ids.fetch_add(1, Ordering::SeqCst)
        }

        fn publish_batch(&self, events: Vec<KvCacheEvent>) -> Result<(), ()> {
            if self.in_flight.fetch_add(1, Ordering::SeqCst) != 0 {
                self.overlapped.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(2));
            self.batches
                .lock()
                .unwrap()
                .push(events.iter().map(|event| event.event_id).collect());
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                Err(())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn stored_event_maps_hashes_and_parent() {
        let event = CacheEvent::Stored {
            parent_hash: Some(11),
            blocks: vec![StoredBlock {
                tokens_hash: 100,
                block_hash: 10,
            }],
        };
        let got = to_kv_event(event, 7, 3);
        assert_eq!(got.event_id, 3);
        assert_eq!(got.dp_rank, 7);
        match got.data {
            KvCacheEventData::Stored(store) => {
                assert_eq!(store.parent_hash, Some(ExternalSequenceBlockHash(11)));
                assert!(store.start_position.is_none());
                assert_eq!(
                    store.blocks,
                    vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(10),
                        tokens_hash: LocalBlockHash(100),
                        mm_extra_info: None,
                    }]
                );
            }
            other => panic!("expected Stored, got {other:?}"),
        }
    }

    #[test]
    fn removed_event_maps_all_hashes() {
        let got = to_kv_event(
            CacheEvent::Removed {
                block_hashes: vec![1, 2],
            },
            0,
            9,
        );
        match got.data {
            KvCacheEventData::Removed(remove) => {
                assert_eq!(
                    remove.block_hashes,
                    vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)]
                );
            }
            other => panic!("expected Removed, got {other:?}"),
        }
    }

    #[test]
    fn cleared_event_maps_without_blocks() {
        let got = to_kv_event(CacheEvent::Cleared, 4, 0);
        assert_eq!(got.data, KvCacheEventData::Cleared);
    }

    #[test]
    fn publish_without_publisher_is_buffered_until_set() {
        let sink = EventSink::new(0);
        assert_eq!(sink.publish(vec![stored()]), PublishedEvents::default());

        let fake = Arc::new(FakePublisher::default());
        sink.set_publisher(fake.clone());
        assert_eq!(fake.recorded(), vec![vec![0]], "buffered event replayed");

        // Once installed, later publishes go straight out and continue the ids.
        assert_eq!(
            sink.publish(vec![stored()]),
            PublishedEvents {
                stored: 1,
                ..Default::default()
            }
        );
        assert_eq!(fake.recorded(), vec![vec![0], vec![1]]);
    }

    #[test]
    fn closed_publisher_reports_no_delivered_events() {
        let sink = EventSink::new(0);
        let fake = Arc::new(FakePublisher::default());
        fake.fail.store(true, Ordering::SeqCst);
        sink.set_publisher(fake.clone());

        assert_eq!(
            sink.publish(vec![stored()]),
            PublishedEvents::default(),
            "events dropped by a closed publisher must not be counted"
        );
        assert_eq!(fake.recorded(), vec![vec![0]]);
    }

    #[test]
    fn concurrent_publishes_are_serialized() {
        let sink = Arc::new(EventSink::new(0));
        let fake = Arc::new(FakePublisher::default());
        sink.set_publisher(fake.clone());

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let sink = sink.clone();
                std::thread::spawn(move || {
                    for _ in 0..8 {
                        sink.publish(vec![stored()]);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        assert!(
            !fake.overlapped.load(Ordering::SeqCst),
            "publish_batch ran concurrently with another publish"
        );
        let delivered: Vec<u64> = fake.recorded().into_iter().flatten().collect();
        assert_eq!(
            delivered,
            (0..64).collect::<Vec<_>>(),
            "ids must be consecutive and delivered in id order"
        );
    }
}
