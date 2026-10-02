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
    /// Whether a publisher may still arrive. `false` when KV routing is disabled,
    /// in which case `publish` drops events immediately instead of retaining them
    /// until [`MAX_PENDING_EVENTS`].
    buffering: bool,
    /// Whether a drop has already been warned about. Reset on a successful
    /// publish so an outage logs once rather than once per request.
    warned_dropped: bool,
}

/// Upper bound on events buffered before a publisher is installed.
///
/// A proxy started with `--enable-kv-routing=false` never installs one, so
/// `EventSink::with_buffering` turns buffering off for that case and the events
/// are dropped immediately. When buffering is on, reaching this cap discards the
/// whole buffer rather than the oldest events: see [`buffer_pending`].
const MAX_PENDING_EVENTS: usize = 8192;

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
    /// Events that never reached the router (a failed publish or a `pending`
    /// overflow). Distinct from the delivered counts so a dropped `Stored`
    /// batch cannot be mistaken for "the proxy had no traffic".
    pub dropped: u64,
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

    /// Events this call handed to the router.
    fn delivered(&self) -> u64 {
        self.stored + self.removed + self.cleared
    }
}

impl EventSink {
    /// A sink that buffers until a publisher is installed. Use this when KV routing is enabled
    /// and `Worker` will call `set` from the `KvEventSource` `on_ready` callback.
    pub fn new(dp_rank: u32) -> Self {
        Self::with_buffering(dp_rank, true)
    }

    /// A sink with buffering on or off. `buffering = false` is for a worker whose KV routing is
    /// disabled: no publisher will ever be installed, so events are dropped on arrival rather
    /// than retained to [`MAX_PENDING_EVENTS`] for a replay that cannot happen.
    pub fn with_buffering(dp_rank: u32, buffering: bool) -> Self {
        Self {
            dp_rank,
            state: Mutex::new(SinkState {
                publisher: None,
                pending: Vec::new(),
                buffering,
                warned_dropped: false,
            }),
        }
    }

    /// Store the publisher built by `Worker`. Called once, from the
    /// `KvEventSource::Push` `on_ready` callback. Events produced before this
    /// call are replayed now, and the returned counts describe that replay so
    /// the caller can record them in `proxy_kv_events_total` and
    /// `proxy_kv_events_dropped_total`.
    pub fn set(&self, publisher: Arc<KvEventPublisher>) -> PublishedEvents {
        self.set_publisher(publisher as Arc<dyn EventPublisher>)
    }

    fn set_publisher(&self, publisher: Arc<dyn EventPublisher>) -> PublishedEvents {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.publisher = Some(publisher.clone());
        let pending = std::mem::take(&mut state.pending);
        if pending.is_empty() {
            return PublishedEvents::default();
        }
        let (batch, counts) = build_batch(pending, self.dp_rank, publisher.as_ref());
        match publisher.publish_batch(batch) {
            Ok(()) => {
                state.warned_dropped = false;
                counts
            }
            Err(()) => {
                if !state.warned_dropped {
                    tracing::warn!("dropping buffered virtual-cache events: KV publisher closed");
                    state.warned_dropped = true;
                }
                PublishedEvents {
                    dropped: counts.delivered(),
                    ..Default::default()
                }
            }
        }
    }

    /// Publish a batch of cache events.
    ///
    /// While no publisher exists yet, events are buffered (up to
    /// [`MAX_PENDING_EVENTS`]) and replayed when `set` installs one; the
    /// returned delivered counts are zero because nothing reached the router
    /// yet. On overflow the whole buffer is discarded (see [`buffer_pending`]).
    /// A sink built with `buffering = false` (KV routing disabled) drops every
    /// event immediately instead. When `publish_batch` fails the events are
    /// dropped; every case reports the loss in [`PublishedEvents::dropped`].
    pub fn publish(&self, events: Vec<CacheEvent>) -> PublishedEvents {
        if events.is_empty() {
            return PublishedEvents::default();
        }
        // One critical section for id assignment and the send: `next_event_id`
        // and `publish_batch` are not atomic together, so without this guard
        // concurrent publishers could send ids out of order.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(publisher) = state.publisher.clone() else {
            if !state.buffering {
                // KV routing is disabled, so no publisher will come to replay these. Count them as
                // dropped rather than holding them for a replay that can never happen.
                return PublishedEvents {
                    dropped: events.len() as u64,
                    ..Default::default()
                };
            }
            let dropped = buffer_pending(&mut state.pending, events);
            if dropped > 0 && !state.warned_dropped {
                tracing::warn!(
                    dropped,
                    cap = MAX_PENDING_EVENTS,
                    "dropping buffered virtual-cache events: no KV publisher installed"
                );
                state.warned_dropped = true;
            }
            return PublishedEvents {
                dropped,
                ..Default::default()
            };
        };
        let (batch, counts) = build_batch(events, self.dp_rank, publisher.as_ref());
        match publisher.publish_batch(batch) {
            Ok(()) => {
                state.warned_dropped = false;
                counts
            }
            Err(()) => {
                if !state.warned_dropped {
                    tracing::warn!("dropping virtual-cache events: KV publisher closed");
                    state.warned_dropped = true;
                }
                PublishedEvents {
                    dropped: counts.delivered(),
                    ..Default::default()
                }
            }
        }
    }
}

/// Append `events` to `pending` unless that would exceed [`MAX_PENDING_EVENTS`].
///
/// Past the cap the whole buffer is discarded, not just the oldest entries: the router links a
/// `Stored` event to the prefix before it through `parent_hash`, so replaying a suffix without its
/// parents would leave dangling blocks. The proxy's local indexer re-emits a prompt's `Stored`
/// events on the next request, so the router rebuilds the tree from a full replay instead of a
/// broken chain. Returns the number of events dropped.
fn buffer_pending(pending: &mut Vec<CacheEvent>, events: Vec<CacheEvent>) -> u64 {
    if pending.len() + events.len() > MAX_PENDING_EVENTS {
        let dropped = pending.len() as u64 + events.len() as u64;
        pending.clear();
        return dropped;
    }
    pending.extend(events);
    0
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
        // The replay's counts are returned so the caller can record them; they
        // are otherwise lost and the metric would under-report held events.
        assert_eq!(
            sink.set_publisher(fake.clone()),
            PublishedEvents {
                stored: 1,
                ..Default::default()
            }
        );
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
    fn closed_publisher_reports_dropped_events() {
        let sink = EventSink::new(0);
        let fake = Arc::new(FakePublisher::default());
        fake.fail.store(true, Ordering::SeqCst);
        sink.set_publisher(fake.clone());

        assert_eq!(
            sink.publish(vec![stored()]),
            PublishedEvents {
                dropped: 1,
                ..Default::default()
            },
            "events dropped by a closed publisher must be reported as dropped, not delivered"
        );
        assert_eq!(fake.recorded(), vec![vec![0]]);
    }

    #[test]
    fn publish_without_buffering_drops_immediately() {
        // KV routing is disabled, so no publisher will ever be installed. Every event must be
        // dropped on arrival (and counted), not retained to the cap for a replay that cannot
        // happen.
        let sink = EventSink::with_buffering(0, false);
        for _ in 0..(MAX_PENDING_EVENTS * 2) {
            let published = sink.publish(vec![stored()]);
            assert_eq!(published.dropped, 1);
            assert_eq!(published.delivered(), 0);
        }
        // A publisher installed anyway is not replayed anything: the sink never buffered.
        let fake = Arc::new(FakePublisher::default());
        assert_eq!(sink.set_publisher(fake.clone()), PublishedEvents::default());
        assert!(fake.recorded().is_empty());
    }

    #[test]
    fn pending_overflow_drops_the_whole_buffer_to_keep_chains() {
        // No publisher is installed, so every event is buffered. Once the cap would be exceeded
        // the whole buffer is discarded, not just the oldest entries: replaying a suffix without
        // its parents would leave the router with dangling `parent_hash` links, and the proxy's
        // local indexer re-emits the full tree on the next request.
        let sink = EventSink::new(0);
        for _ in 0..MAX_PENDING_EVENTS {
            assert_eq!(sink.publish(vec![stored()]).dropped, 0);
        }
        let overflow = sink.publish(vec![stored()]);
        assert_eq!(
            overflow.dropped,
            MAX_PENDING_EVENTS as u64 + 1,
            "the whole buffer is dropped and counted, not only the oldest event"
        );
        assert_eq!(overflow.delivered(), 0);

        // A publisher installed later replays nothing; the local indexer rebuilds from scratch.
        let fake = Arc::new(FakePublisher::default());
        assert_eq!(sink.set_publisher(fake.clone()), PublishedEvents::default());
        assert!(
            fake.recorded().is_empty(),
            "the buffer was cleared, so there is nothing to replay"
        );
    }

    #[tokio::test]
    async fn published_events_are_recoverable_by_a_fresh_local_indexer() {
        use dynamo_kv_router::indexer::{KvIndexerMetrics, LocalKvIndexer, WorkerKvQueryResponse};
        use dynamo_kv_router::protocols::RouterEvent;
        use tokio_util::sync::CancellationToken;

        // Stand in for a fresh frontend: it has never seen this proxy and pulls
        // the worker's whole held tree.
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            64,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            1024,
        );

        // The proxy's event stream for a multi-turn prefix. The second Stored
        // extends a prefix it still holds, so its parent is the first block; the
        // fresh indexer only succeeds because recovery replays the parent too.
        let root = to_kv_event(
            CacheEvent::Stored {
                parent_hash: None,
                blocks: vec![StoredBlock {
                    tokens_hash: 10,
                    block_hash: 1,
                }],
            },
            0,
            0,
        );
        let child = to_kv_event(
            CacheEvent::Stored {
                parent_hash: Some(1),
                blocks: vec![StoredBlock {
                    tokens_hash: 11,
                    block_hash: 2,
                }],
            },
            0,
            1,
        );
        indexer
            .apply_event_with_buffer(RouterEvent::new(1, root))
            .await
            .unwrap();
        indexer
            .apply_event_with_buffer(RouterEvent::new(1, child))
            .await
            .unwrap();

        let response = indexer.get_events_in_id_range(None, None).await;
        let events = match response {
            WorkerKvQueryResponse::TreeDump { events, .. } => events,
            other => panic!("expected a tree dump, got {other:?}"),
        };
        let stored_blocks: Vec<u64> = events
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(data) => Some(
                    data.blocks
                        .iter()
                        .map(|b| b.block_hash.0)
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(
            stored_blocks,
            vec![1, 2],
            "recovery must return the parent before the child"
        );
    }

    #[test]
    fn closed_publisher_during_replay_reports_dropped_events() {
        // A publisher installed while the buffered events exist but already
        // closed drops the replay; `set` must report that as dropped so the
        // metric does not count it as delivered.
        let sink = EventSink::new(0);
        assert_eq!(sink.publish(vec![stored(), stored()]).dropped, 0);
        let fake = Arc::new(FakePublisher::default());
        fake.fail.store(true, Ordering::SeqCst);
        let replayed = sink.set_publisher(fake.clone());
        assert_eq!(replayed.delivered(), 0);
        assert_eq!(replayed.dropped, 2);
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
