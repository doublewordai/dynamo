//! Tests for the virtual KV cache: hashing parity with the router, prefix chaining, TTL and
//! LRU eviction, and conversion to the upstream event type.

use std::time::{Duration, Instant};

use dw_proxy_core::vcache::{
    CacheEvent, HashOptions, StoredBlock, VirtualCache, VirtualCacheConfig,
};
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash,
    compute_block_hash_for_seq, compute_seq_hash_for_block,
};

fn config(block_size: u32, ttl: Duration, max_blocks: usize) -> VirtualCacheConfig {
    VirtualCacheConfig {
        block_size,
        ttl,
        max_blocks,
    }
}

fn stored(events: &[CacheEvent]) -> Option<(Option<u64>, Vec<StoredBlock>)> {
    events.iter().find_map(|event| match event {
        CacheEvent::Stored {
            parent_hash,
            blocks,
        } => Some((*parent_hash, blocks.clone())),
        _ => None,
    })
}

fn removed(events: &[CacheEvent]) -> Option<Vec<u64>> {
    events.iter().find_map(|event| match event {
        CacheEvent::Removed { block_hashes } => Some(block_hashes.clone()),
        _ => None,
    })
}

/// Deterministic xorshift so the token sequences are random but reproducible.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// The hashes the router would compute for the complete blocks of `tokens`.
fn expected(options: &HashOptions, block_size: u32, tokens: &[u32]) -> Vec<StoredBlock> {
    let router_options = BlockHashOptions {
        block_mm_infos: None,
        lora_name: options.lora_name.as_deref(),
        cache_namespace: options.cache_salt.as_deref(),
        is_eagle: Some(options.is_eagle),
    };
    let local = compute_block_hash_for_seq(tokens, block_size, router_options);
    let seq = compute_seq_hash_for_block(&local);
    local
        .iter()
        .zip(seq.iter())
        .map(|(l, s)| StoredBlock {
            tokens_hash: l.0,
            block_hash: *s,
        })
        .collect()
}

#[test]
fn hashes_match_router_for_random_sequences() {
    let now = Instant::now();
    let options = [
        HashOptions::default(),
        HashOptions {
            lora_name: Some("adapter-a".to_string()),
            cache_salt: None,
            is_eagle: false,
        },
        HashOptions {
            lora_name: None,
            cache_salt: Some("conversation-7".to_string()),
            is_eagle: false,
        },
        HashOptions {
            lora_name: Some("adapter-b".to_string()),
            cache_salt: Some("salt".to_string()),
            is_eagle: false,
        },
    ];

    for block_size in [16u32, 64] {
        let mut state = 0x1234_5678_9abc_def0u64 ^ block_size as u64;
        for case in 0..8 {
            let len = block_size as usize * 3 + (case % 3);
            let tokens: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32).collect();
            let opt = &options[case % options.len()];

            let mut cache = VirtualCache::new(config(block_size, Duration::from_secs(60), 1024));
            let events = cache.on_request(&tokens, opt, now);
            let (parent, blocks) = stored(&events).expect("expected a stored event");

            assert_eq!(parent, None);
            assert_eq!(blocks, expected(opt, block_size, &tokens));
            assert_eq!(blocks.len(), len / block_size as usize);
        }
    }
}

#[test]
fn partial_trailing_block_is_ignored() {
    let now = Instant::now();
    let tokens: Vec<u32> = (0..7).collect();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(60), 1024));

    let events = cache.on_request(&tokens, &HashOptions::default(), now);
    let (_, blocks) = stored(&events).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(cache.len_blocks(), 1);

    // Shorter than one full block: nothing is stored.
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(60), 1024));
    let events = cache.on_request(&[1, 2, 3], &HashOptions::default(), now);
    assert!(events.is_empty());
    assert_eq!(cache.len_blocks(), 0);
}

#[test]
fn shared_prefix_stores_only_new_blocks_with_parent() {
    let t0 = Instant::now();
    let options = HashOptions::default();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(60), 1024));

    let first: Vec<u32> = (0..8).collect();
    let events = cache.on_request(&first, &options, t0);
    let (parent, blocks) = stored(&events).unwrap();
    assert_eq!(parent, None);
    assert_eq!(blocks.len(), 2);

    let second: Vec<u32> = (0..4).chain(100..104).collect();
    let events = cache.on_request(&second, &options, t0 + Duration::from_secs(1));
    let (parent, blocks) = stored(&events).unwrap();
    assert_eq!(parent, Some(blocks_of(&first, 4)[0].block_hash));
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0], expected(&options, 4, &second)[1]);

    // Only the new block was added.
    assert_eq!(cache.len_blocks(), 3);
}

/// Convenience for building the expected blocks of a request.
fn blocks_of(tokens: &[u32], block_size: u32) -> Vec<StoredBlock> {
    expected(&HashOptions::default(), block_size, tokens)
}

#[test]
fn repeat_request_refreshes_ttl() {
    let t0 = Instant::now();
    let options = HashOptions::default();
    let tokens: Vec<u32> = (0..8).collect();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(10), 1024));

    cache.on_request(&tokens, &options, t0);
    // Halfway through the TTL, send the same prefix again.
    let events = cache.on_request(&tokens, &options, t0 + Duration::from_secs(5));
    assert!(events.is_empty(), "held blocks must not be re-stored");

    // Original expiry would have been t0+10, refreshed is t0+15.
    assert!(cache.expire(t0 + Duration::from_secs(12)).is_empty());
    let hashes = blocks_of(&tokens, 4);
    let removed = removed(&cache.expire(t0 + Duration::from_secs(15)));
    assert_eq!(
        removed,
        Some(vec![hashes[1].block_hash, hashes[0].block_hash])
    );
    assert_eq!(cache.len_blocks(), 0);
}

#[test]
fn expiry_removes_children_before_parents() {
    let t0 = Instant::now();
    let options = HashOptions::default();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(10), 1024));

    let a: Vec<u32> = (0..4).collect();
    let ab: Vec<u32> = (0..8).collect();
    let abc: Vec<u32> = (0..12).collect();
    cache.on_request(&a, &options, t0);
    cache.on_request(&ab, &options, t0 + Duration::from_secs(1));
    cache.on_request(&abc, &options, t0 + Duration::from_secs(2));
    assert_eq!(cache.len_blocks(), 3);

    let events = cache.expire(t0 + Duration::from_secs(12));
    let hashes = blocks_of(&abc, 4);
    let removed = removed(&events).unwrap();
    assert_eq!(
        removed,
        vec![
            hashes[2].block_hash,
            hashes[1].block_hash,
            hashes[0].block_hash
        ]
    );
    assert_eq!(cache.len_blocks(), 0);
}

#[test]
fn lru_cap_evicts_oldest_leaf_and_reports_removed() {
    let t0 = Instant::now();
    let options = HashOptions::default();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(60), 2));

    let a: Vec<u32> = (0..4).collect();
    let b: Vec<u32> = (100..104).collect();
    cache.on_request(&a, &options, t0);
    cache.on_request(&b, &options, t0 + Duration::from_secs(1));
    assert_eq!(cache.len_blocks(), 2);

    // c arrives; the least recently used leaf (a) is evicted before c is stored.
    let c: Vec<u32> = (200..204).collect();
    let events = cache.on_request(&c, &options, t0 + Duration::from_secs(2));
    assert_eq!(removed(&events), Some(vec![blocks_of(&a, 4)[0].block_hash]));
    let (parent, blocks) = stored(&events).unwrap();
    assert_eq!(parent, None);
    assert_eq!(blocks, vec![blocks_of(&c, 4)[0]]);
    assert_eq!(cache.len_blocks(), 2);

    // Refresh b, then d arrives; now c is the least recently used leaf.
    cache.on_request(&b, &options, t0 + Duration::from_secs(3));
    let d: Vec<u32> = (300..304).collect();
    let events = cache.on_request(&d, &options, t0 + Duration::from_secs(4));
    assert_eq!(removed(&events), Some(vec![blocks_of(&c, 4)[0].block_hash]));
    assert_eq!(cache.len_blocks(), 2);
}

#[test]
fn eviction_never_removes_a_parent_before_its_children() {
    let t0 = Instant::now();
    let options = HashOptions::default();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(60), 2));

    // A root block, then a child extending it: the child is the only leaf.
    let a: Vec<u32> = (0..4).collect();
    let ab: Vec<u32> = (0..8).collect();
    cache.on_request(&a, &options, t0);
    cache.on_request(&ab, &options, t0 + Duration::from_secs(1));
    assert_eq!(cache.len_blocks(), 2);

    // A new branch forces eviction. Only the leaf child may go; its parent stays.
    let c: Vec<u32> = (100..104).collect();
    let events = cache.on_request(&c, &options, t0 + Duration::from_secs(2));
    let hashes = blocks_of(&ab, 4);
    assert_eq!(removed(&events), Some(vec![hashes[1].block_hash]));
    let (parent, blocks) = stored(&events).unwrap();
    assert_eq!(parent, None);
    assert_eq!(blocks, vec![blocks_of(&c, 4)[0]]);
    assert_eq!(cache.len_blocks(), 2);

    // Even when the parent is oldest, it is only evicted after its child is gone.
    let d: Vec<u32> = (200..204).collect();
    let events = cache.on_request(&d, &options, t0 + Duration::from_secs(3));
    assert_eq!(removed(&events), Some(vec![hashes[0].block_hash]));
}

#[test]
fn clear_drops_everything() {
    let t0 = Instant::now();
    let options = HashOptions::default();
    let tokens: Vec<u32> = (0..8).collect();
    let mut cache = VirtualCache::new(config(4, Duration::from_secs(60), 1024));

    cache.on_request(&tokens, &options, t0);
    assert_eq!(cache.len_blocks(), 2);
    assert_eq!(cache.clear(), vec![CacheEvent::Cleared]);
    assert_eq!(cache.len_blocks(), 0);
    assert!(cache.expire(t0 + Duration::from_secs(3600)).is_empty());
}

#[test]
fn to_router_event_maps_fields() {
    let tokens: Vec<u32> = (0..8).collect();
    let hashes = blocks_of(&tokens, 4);

    let stored_event = CacheEvent::Stored {
        parent_hash: Some(hashes[0].block_hash),
        blocks: vec![hashes[1]],
    };
    let router = stored_event.to_router_event(17, 3);
    assert_eq!(router.event_id, 17);
    assert_eq!(router.dp_rank, 3);
    match router.data {
        KvCacheEventData::Stored(store) => {
            assert_eq!(
                store.parent_hash,
                Some(ExternalSequenceBlockHash(hashes[0].block_hash))
            );
            assert_eq!(store.start_position, None);
            assert_eq!(store.blocks.len(), 1);
            assert_eq!(
                store.blocks[0].block_hash,
                ExternalSequenceBlockHash(hashes[1].block_hash)
            );
            assert_eq!(
                store.blocks[0].tokens_hash,
                LocalBlockHash(hashes[1].tokens_hash)
            );
            assert_eq!(store.blocks[0].mm_extra_info, None);
        }
        other => panic!("expected stored, got {other:?}"),
    }

    let removed_event = CacheEvent::Removed {
        block_hashes: vec![hashes[0].block_hash, hashes[1].block_hash],
    };
    let router = removed_event.to_router_event(18, 0);
    match router.data {
        KvCacheEventData::Removed(remove) => assert_eq!(
            remove.block_hashes,
            vec![
                ExternalSequenceBlockHash(hashes[0].block_hash),
                ExternalSequenceBlockHash(hashes[1].block_hash),
            ]
        ),
        other => panic!("expected removed, got {other:?}"),
    }

    let router = CacheEvent::Cleared.to_router_event(19, 1);
    assert_eq!(router.event_id, 19);
    assert_eq!(router.dp_rank, 1);
    assert_eq!(router.data, KvCacheEventData::Cleared);
}
