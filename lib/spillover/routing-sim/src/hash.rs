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
            (state & 0xffff) as u32
        })
        .collect()
}

/// Chained hashes of every full block in `tokens`.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    if block_size == 0 {
        return Vec::new();
    }
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

/// Number of full blocks needed to hold `tokens`, rounding up.
pub fn blocks_for(tokens: usize, block_size: u32) -> usize {
    if block_size == 0 {
        0
    } else {
        tokens.div_ceil(block_size as usize)
    }
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
}
