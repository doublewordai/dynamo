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

/// The publisher slot plus the DP rank the events are stamped with.
pub struct EventSink {
    dp_rank: u32,
    publisher: Mutex<Option<Arc<KvEventPublisher>>>,
}

impl EventSink {
    pub fn new(dp_rank: u32) -> Self {
        Self {
            dp_rank,
            publisher: Mutex::new(None),
        }
    }

    /// Store the publisher built by `Worker`. Called once, from the
    /// `KvEventSource::Push` `on_ready` callback.
    pub fn set(&self, publisher: Arc<KvEventPublisher>) {
        *self.publisher.lock().unwrap_or_else(|e| e.into_inner()) = Some(publisher);
    }

    /// Publish a batch of cache events. Drops them while no publisher exists
    /// yet (the timer can tick before `Worker` builds the publisher).
    pub fn publish(&self, events: Vec<CacheEvent>) {
        if events.is_empty() {
            return;
        }
        let publisher = match self
            .publisher
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            Some(p) => p,
            None => return,
        };
        let batch: Vec<KvCacheEvent> = events
            .into_iter()
            .map(|event| to_kv_event(event, self.dp_rank, publisher.next_event_id()))
            .collect();
        if let Err(err) = publisher.publish_batch(batch) {
            tracing::warn!(?err, "dropping virtual-cache events: KV publisher closed");
        }
    }
}

/// Translate one virtual-cache event into the router's wire event.
pub fn to_kv_event(event: CacheEvent, dp_rank: u32, event_id: u64) -> KvCacheEvent {
    event.to_router_event(event_id, dp_rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dw_proxy_core::vcache::StoredBlock;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoredBlockData, LocalBlockHash,
    };

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
}
