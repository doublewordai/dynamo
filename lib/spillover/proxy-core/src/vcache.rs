// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Virtual KV cache: the prompt prefixes this proxy has sent upstream, published to the router
//! as KV "stored" events so conversations stay sticky, and removed after the provider's
//! prompt-cache TTL.
//!
//! Block hashes MUST be computed exactly as Dynamo's router computes them for the same token
//! sequence (block size, LoRA name, cache salt, multimodal hashes, eagle flag), using
//! `dynamo_kv_router`'s own hashing functions, or the router never matches our blocks.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};

#[derive(Debug, Clone, PartialEq)]
pub struct VirtualCacheConfig {
    pub block_size: u32,
    /// How long a prefix counts as cached after it was last sent upstream.
    pub ttl: Duration,
    /// Upper bound on blocks held; evict least recently used beyond this.
    pub max_blocks: usize,
}

/// Inputs to block hashing beyond the tokens. Mirrors what the router hashes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HashOptions {
    pub lora_name: Option<String>,
    pub cache_salt: Option<String>,
    pub is_eagle: bool,
}

/// One full block in a stored event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredBlock {
    /// The router's per-block content hash (`LocalBlockHash`).
    pub tokens_hash: u64,
    /// The router's chained sequence hash (`ExternalSequenceBlockHash`).
    pub block_hash: u64,
}

/// Events for the proxy worker to publish, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum CacheEvent {
    /// New blocks, continuing from `parent_hash` (a `block_hash`), or from the root if `None`.
    Stored {
        parent_hash: Option<u64>,
        blocks: Vec<StoredBlock>,
    },
    /// Blocks no longer cached, by `block_hash`.
    Removed {
        block_hashes: Vec<u64>,
    },
    Cleared,
}

impl CacheEvent {
    /// Convert to the upstream router event, stamped with the publisher's event id and DP rank.
    pub fn to_router_event(&self, event_id: u64, dp_rank: u32) -> KvCacheEvent {
        let data = match self {
            CacheEvent::Stored {
                parent_hash,
                blocks,
            } => KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: blocks
                    .iter()
                    .map(|block| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block.block_hash),
                        tokens_hash: LocalBlockHash(block.tokens_hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
            CacheEvent::Removed { block_hashes } => KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: block_hashes
                    .iter()
                    .map(|hash| ExternalSequenceBlockHash(*hash))
                    .collect(),
            }),
            CacheEvent::Cleared => KvCacheEventData::Cleared,
        };
        KvCacheEvent {
            event_id,
            data,
            dp_rank,
        }
    }
}

/// One held block in the prefix tree.
struct Node {
    tokens_hash: u64,
    parent: Option<u64>,
    /// Child block hashes keyed by the child's `tokens_hash`.
    children: HashMap<u64, u64>,
    last_used: Instant,
}

pub struct VirtualCache {
    config: VirtualCacheConfig,
    /// Held blocks keyed by chained `block_hash`.
    blocks: HashMap<u64, Node>,
    /// Root blocks keyed by `tokens_hash`.
    roots: HashMap<u64, u64>,
}

impl VirtualCache {
    pub fn new(config: VirtualCacheConfig) -> Self {
        Self {
            config,
            blocks: HashMap::new(),
            roots: HashMap::new(),
        }
    }

    pub fn config(&self) -> &VirtualCacheConfig {
        &self.config
    }

    /// Record that `prompt_tokens` was sent upstream at `now`. Emits `Stored` for blocks not
    /// already held (only full blocks) and refreshes the TTL of blocks already held. May emit
    /// `Removed` first if the size cap forces eviction.
    pub fn on_request(
        &mut self,
        prompt_tokens: &[u32],
        options: &HashOptions,
        now: Instant,
    ) -> Vec<CacheEvent> {
        let (local, seq) = self.hashes(prompt_tokens, options);
        if local.is_empty() {
            return Vec::new();
        }

        // Walk the held prefix, refreshing recency. The first missing block starts the new chain.
        let mut parent: Option<u64> = None;
        let mut first_new = local.len();
        for (i, block) in local.iter().enumerate() {
            let child = match parent {
                None => self.roots.get(&block.0).copied(),
                Some(hash) => self
                    .blocks
                    .get(&hash)
                    .and_then(|n| n.children.get(&block.0))
                    .copied(),
            };
            match child {
                Some(hash) => {
                    if let Some(node) = self.blocks.get_mut(&hash) {
                        node.last_used = now;
                    }
                    parent = Some(hash);
                }
                None => {
                    first_new = i;
                    break;
                }
            }
        }

        let mut removed = Vec::new();
        if first_new < local.len() && self.config.max_blocks > 0 {
            // Insert the new chain. The first new block continues from the held prefix.
            let parent_before = parent;
            for i in first_new..local.len() {
                let block_hash = seq[i];
                let node_parent = if i == first_new {
                    parent_before
                } else {
                    Some(seq[i - 1])
                };
                self.blocks.insert(
                    block_hash,
                    Node {
                        tokens_hash: local[i].0,
                        parent: node_parent,
                        children: HashMap::new(),
                        last_used: now,
                    },
                );
                match node_parent {
                    None => {
                        self.roots.insert(local[i].0, block_hash);
                    }
                    Some(p) => {
                        if let Some(node) = self.blocks.get_mut(&p) {
                            node.children.insert(local[i].0, block_hash);
                        }
                    }
                }
            }

            // Evict least-recently-used leaves until under the cap. Never remove a parent
            // before its children, since only leaves are eligible.
            removed = self.evict_to_cap();
            // Drop new blocks evicted before they were ever published.
            removed.retain(|hash| !(first_new..local.len()).any(|i| seq[i] == *hash));
        }

        let mut events = Vec::new();
        if !removed.is_empty() {
            events.push(CacheEvent::Removed {
                block_hashes: removed,
            });
        }

        if first_new < local.len() && self.config.max_blocks > 0 {
            // Survivors of eviction form a prefix of the new chain.
            let parent_hash = if first_new == 0 {
                None
            } else {
                Some(seq[first_new - 1])
            };
            let mut blocks = Vec::new();
            for i in first_new..local.len() {
                if self.blocks.contains_key(&seq[i]) {
                    blocks.push(StoredBlock {
                        tokens_hash: local[i].0,
                        block_hash: seq[i],
                    });
                } else {
                    break;
                }
            }
            if !blocks.is_empty() {
                events.push(CacheEvent::Stored {
                    parent_hash,
                    blocks,
                });
            }
        }

        events
    }

    /// Remove blocks whose TTL expired at `now`. Children expire no later than their parents.
    pub fn expire(&mut self, now: Instant) -> Vec<CacheEvent> {
        let mut removed = Vec::new();
        loop {
            let expired = self
                .blocks
                .iter()
                .filter(|(_, node)| {
                    node.children.is_empty() && node.last_used + self.config.ttl <= now
                })
                .map(|(hash, _)| *hash)
                .collect::<Vec<_>>();
            if expired.is_empty() {
                break;
            }
            for hash in expired {
                self.remove(hash);
                removed.push(hash);
            }
        }
        if removed.is_empty() {
            Vec::new()
        } else {
            vec![CacheEvent::Removed {
                block_hashes: removed,
            }]
        }
    }

    /// Drop everything (e.g. on restart) and return `[Cleared]`.
    pub fn clear(&mut self) -> Vec<CacheEvent> {
        self.blocks.clear();
        self.roots.clear();
        vec![CacheEvent::Cleared]
    }

    pub fn len_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Router hashes for the complete blocks of `tokens`: (local, chained sequence).
    fn hashes(&self, tokens: &[u32], options: &HashOptions) -> (Vec<LocalBlockHash>, Vec<u64>) {
        let block_options = BlockHashOptions {
            block_mm_infos: None,
            lora_name: options.lora_name.as_deref(),
            cache_namespace: options.cache_salt.as_deref(),
            is_eagle: Some(options.is_eagle),
        };
        let local = compute_block_hash_for_seq(tokens, self.config.block_size, block_options);
        let seq = compute_seq_hash_for_block(&local);
        (local, seq)
    }

    /// Evict leaves until at most `max_blocks` remain, oldest first. Returns evicted hashes in
    /// removal order (children before parents).
    fn evict_to_cap(&mut self) -> Vec<u64> {
        let mut removed = Vec::new();
        while self.blocks.len() > self.config.max_blocks {
            let victim = self
                .blocks
                .iter()
                .filter(|(_, node)| node.children.is_empty())
                .min_by_key(|(hash, node)| (node.last_used, **hash))
                .map(|(hash, _)| *hash);
            match victim {
                Some(hash) => {
                    self.remove(hash);
                    removed.push(hash);
                }
                None => break,
            }
        }
        removed
    }

    fn remove(&mut self, hash: u64) {
        if let Some(node) = self.blocks.remove(&hash) {
            match node.parent {
                None => {
                    self.roots.remove(&node.tokens_hash);
                }
                Some(parent) => {
                    if let Some(parent_node) = self.blocks.get_mut(&parent) {
                        parent_node.children.remove(&node.tokens_hash);
                    }
                }
            }
        }
    }
}
