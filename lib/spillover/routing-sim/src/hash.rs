// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic synthetic tokens and block hashes.
//!
//! The simulation does not need Dynamo's exact hash values, only hashes that are stable across
//! runs and equal for equal token prefixes, so a shared system prompt hits every worker's cache.

/// FNV-1a over a byte string, used to seed synthetic token sequences.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A stable sequence of `count` token ids derived from `label`.
pub fn synth_tokens(label: &str, count: usize) -> Vec<u32> {
    let mut state = fnv1a(label.as_bytes());
    (0..count)
        .map(|_| {
            // xorshift64, deterministic and cheap.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Keep the full 32 bits: masking to 16 bits made distinct positions collide
            // often enough to create spurious prompt-prefix overlaps.
            (state & 0xffff_ffff) as u32
        })
        .collect()
}

/// Chained hashes of every full block in `tokens`.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    // A zero block size is a configuration error, not an empty result: `Scenario::validate`
    // rejects it before a run starts, so reaching here means a caller bypassed validation.
    assert!(block_size > 0, "block_size must be at least 1");
    let mut hashes = Vec::with_capacity(tokens.len() / block_size);
    let mut parent = 0u64;
    for chunk in tokens.chunks(block_size) {
        if chunk.len() < block_size {
            break;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ parent;
        for &token in chunk {
            hash ^= token as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hashes.push(hash);
        parent = hash;
    }
    hashes
}

/// Number of full blocks in `tokens`, the way `BlockTracker`/`PromptRegistry` count them:
/// the trailing partial block is not a block.
pub fn blocks_for(tokens: usize, block_size: u32) -> usize {
    assert!(block_size > 0, "block_size must be at least 1");
    tokens / block_size as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_prefixes_hash_equally() {
        let a = synth_tokens("system", 32);
        let mut b = a.clone();
        b.extend(synth_tokens("user-1", 20));
        let block_size = 8;
        let a_blocks = block_hashes(&a, block_size);
        let b_blocks = block_hashes(&b, block_size);
        assert_eq!(a_blocks, b_blocks[..a_blocks.len()].to_vec());
    }

    #[test]
    fn synth_tokens_is_stable() {
        assert_eq!(synth_tokens("x", 4), synth_tokens("x", 4));
        assert_ne!(synth_tokens("x", 4), synth_tokens("y", 4));
    }

    /// S13-1: only complete blocks count, so a partial trailing block is dropped, not rounded up.
    #[test]
    fn blocks_for_counts_only_complete_blocks() {
        assert_eq!(blocks_for(0, 16), 0);
        assert_eq!(blocks_for(15, 16), 0);
        assert_eq!(blocks_for(16, 16), 1);
        assert_eq!(blocks_for(31, 16), 1);
        assert_eq!(blocks_for(32, 16), 2);
    }
}
