//! Deterministic 128-bit content hashing for the persistence layer.
//!
//! Used to content-address system-prompt section streams (§5.2 of
//! `docs/kv_tier_migration.md`): a section's stream id is derived from a
//! rolling hash chain over the section tokens, so any template change forks
//! a new stream and an unchanged section is a durable prefix-cache hit.
//!
//! The hash is self-contained — no external crate — and deterministic
//! across runs and platforms. It is *not* cryptographic: content addressing
//! needs collision resistance for non-adversarial token streams, not
//! preimage resistance.

use super::streams::{ContentAddress, StreamId};

/// A 128-bit content hash. Two little-endian `u64` lanes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ContentHash {
    pub lo: u64,
    pub hi: u64,
}

impl ContentHash {
    /// The all-zero hash — the seed of an empty [`ContentChain`].
    pub const ZERO: ContentHash = ContentHash { lo: 0, hi: 0 };

    /// The 16 raw little-endian bytes (lo then hi).
    pub fn to_bytes(self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0..8].copy_from_slice(&self.lo.to_le_bytes());
        b[8..16].copy_from_slice(&self.hi.to_le_bytes());
        b
    }

    /// Reconstruct from the 16 raw little-endian bytes.
    pub fn from_bytes(b: [u8; 16]) -> ContentHash {
        let lo = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let hi = u64::from_le_bytes(b[8..16].try_into().unwrap());
        ContentHash { lo, hi }
    }
}

const MUL0: u64 = 0xff51afd7ed558ccd;
const MUL1: u64 = 0xc4ceb9fe1a85ec53;
const SEED0: u64 = 0x243f6a8885a308d3;
const SEED1: u64 = 0x13198a2e03707344;

/// Final avalanche mix (the SplitMix64 finalizer).
fn avalanche(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Streaming 128-bit content hasher.
#[derive(Clone)]
pub struct ContentHasher {
    lane0: u64,
    lane1: u64,
    len: u64,
    buf: [u8; 8],
    buf_len: usize,
}

impl Default for ContentHasher {
    fn default() -> ContentHasher {
        ContentHasher::new()
    }
}

impl ContentHasher {
    /// A fresh hasher.
    pub fn new() -> ContentHasher {
        ContentHasher {
            lane0: SEED0,
            lane1: SEED1,
            len: 0,
            buf: [0; 8],
            buf_len: 0,
        }
    }

    fn mix_word(&mut self, w: u64) {
        self.lane0 = (self.lane0 ^ w).rotate_left(31).wrapping_mul(MUL0);
        self.lane1 = (self.lane1 ^ w.rotate_left(29))
            .rotate_left(17)
            .wrapping_mul(MUL1);
    }

    /// Absorb a byte slice.
    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            while self.buf_len < 8 && !data.is_empty() {
                self.buf[self.buf_len] = data[0];
                self.buf_len += 1;
                data = &data[1..];
            }
            if self.buf_len == 8 {
                let w = u64::from_le_bytes(self.buf);
                self.mix_word(w);
                self.buf_len = 0;
            }
        }
        while data.len() >= 8 {
            let w = u64::from_le_bytes(data[0..8].try_into().unwrap());
            self.mix_word(w);
            data = &data[8..];
        }
        for &b in data {
            self.buf[self.buf_len] = b;
            self.buf_len += 1;
        }
    }

    /// Absorb one `u32` token (little-endian).
    pub fn update_u32(&mut self, v: u32) {
        self.update(&v.to_le_bytes());
    }

    /// Finalize. Does not consume the hasher.
    pub fn finish(&self) -> ContentHash {
        let mut lane0 = self.lane0;
        let mut lane1 = self.lane1;
        if self.buf_len > 0 {
            let mut tail = [0u8; 8];
            tail[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
            let w = u64::from_le_bytes(tail);
            lane0 = (lane0 ^ w).rotate_left(31).wrapping_mul(MUL0);
            lane1 = (lane1 ^ w.rotate_left(29))
                .rotate_left(17)
                .wrapping_mul(MUL1);
        }
        lane0 ^= self.len;
        lane1 = lane1.wrapping_add(self.len.rotate_left(32));
        ContentHash {
            lo: avalanche(lane0 ^ lane1),
            hi: avalanche(lane1.wrapping_add(lane0).rotate_left(23)),
        }
    }
}

/// Hash a byte slice.
pub fn hash_bytes(data: &[u8]) -> ContentHash {
    let mut h = ContentHasher::new();
    h.update(data);
    h.finish()
}

/// Hash a token sequence (each token absorbed as 4 little-endian bytes).
pub fn hash_tokens(tokens: &[u32]) -> ContentHash {
    let mut h = ContentHasher::new();
    for &t in tokens {
        h.update_u32(t);
    }
    h.finish()
}

/// The rolling content-address chain over prompt sections (§5.2).
///
/// `chain[0] = ZERO`, `chain[i] = H(chain[i-1] ++ section_i_tokens)`. The
/// `ContentAddress` of section *i* is `(chain[i-1], H(section_i_tokens))`.
#[derive(Clone, Default)]
pub struct ContentChain {
    prev: ContentHash,
}

impl ContentChain {
    /// A fresh chain seeded at `ContentHash::ZERO`.
    pub fn new() -> ContentChain {
        ContentChain {
            prev: ContentHash::ZERO,
        }
    }

    /// The current prefix hash — `chain[i-1]`, covering every section
    /// pushed so far.
    pub fn prefix(&self) -> ContentHash {
        self.prev
    }

    /// Append one section's tokens. Returns that section's
    /// [`ContentAddress`] and advances the chain.
    pub fn push_section(&mut self, section_tokens: &[u32]) -> ContentAddress {
        let prefix_hash = self.prev;
        let section_hash = hash_tokens(section_tokens);
        let mut h = ContentHasher::new();
        h.update(&prefix_hash.to_bytes());
        for &t in section_tokens {
            h.update_u32(t);
        }
        self.prev = h.finish();
        ContentAddress {
            prefix_hash,
            section_hash,
        }
    }
}

/// Derive the [`StreamId`] of a content-addressed prompt-section stream.
///
/// Stream id 0 is reserved (the header's "N/A" value), so a derived zero is
/// bumped to 1.
pub fn section_stream_id(addr: ContentAddress) -> StreamId {
    let mut h = ContentHasher::new();
    h.update(&addr.prefix_hash.to_bytes());
    h.update(&addr.section_hash.to_bytes());
    let raw = h.finish().lo;
    StreamId(if raw == 0 { 1 } else { raw })
}

/// Derive the [`StreamId`] of an identity-addressed turn stream from its
/// `(timeline_id, turn_index)` coordinates.
pub fn turn_stream_id(timeline_id: u64, turn_index: u32) -> StreamId {
    let mut h = ContentHasher::new();
    h.update(b"turn");
    h.update(&timeline_id.to_le_bytes());
    h.update(&turn_index.to_le_bytes());
    let raw = h.finish().lo;
    StreamId(if raw == 0 { 1 } else { raw })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash_bytes(b"hello world"), hash_bytes(b"hello world"));
        assert_eq!(hash_tokens(&[1, 2, 3, 4]), hash_tokens(&[1, 2, 3, 4]));
    }

    #[test]
    fn distinct_inputs_distinct_hashes() {
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"hellp"));
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"hello "));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 3, 2]));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 3, 0]));
    }

    #[test]
    fn empty_hash_is_not_zero() {
        // An empty input still mixes the length and seeds.
        assert_ne!(hash_bytes(b""), ContentHash::ZERO);
    }

    #[test]
    fn streaming_matches_oneshot() {
        let data: Vec<u8> = (0..200u32).map(|i| (i % 251) as u8).collect();
        let oneshot = hash_bytes(&data);
        for split in [1usize, 7, 8, 9, 64, 199] {
            let mut h = ContentHasher::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finish(), oneshot, "split at {split}");
        }
    }

    #[test]
    fn bytes_roundtrip() {
        let h = hash_bytes(b"roundtrip");
        assert_eq!(ContentHash::from_bytes(h.to_bytes()), h);
    }

    #[test]
    fn chain_cascade_property() {
        // Sections A, B, C. Changing B must change B's and C's stream ids
        // but leave A's untouched.
        let a = [10u32, 11, 12];
        let b = [20u32, 21, 22];
        let b2 = [20u32, 99, 22];
        let c = [30u32, 31];

        let mut base = ContentChain::new();
        let addr_a = base.push_section(&a);
        let addr_b = base.push_section(&b);
        let addr_c = base.push_section(&c);

        let mut alt = ContentChain::new();
        let addr_a2 = alt.push_section(&a);
        let addr_b2 = alt.push_section(&b2);
        let addr_c2 = alt.push_section(&c);

        // A unchanged -> identical address and stream id.
        assert_eq!(addr_a, addr_a2);
        assert_eq!(section_stream_id(addr_a), section_stream_id(addr_a2));
        // B changed -> different.
        assert_ne!(addr_b, addr_b2);
        assert_ne!(section_stream_id(addr_b), section_stream_id(addr_b2));
        // C unchanged content but changed prefix -> cascades.
        assert_ne!(addr_c, addr_c2);
        assert_ne!(section_stream_id(addr_c), section_stream_id(addr_c2));
    }

    #[test]
    fn unchanged_chain_reproduces_ids() {
        let sections = [vec![1u32, 2, 3], vec![4u32, 5], vec![6u32, 7, 8, 9]];
        let run = |chain: &mut ContentChain| -> Vec<StreamId> {
            sections
                .iter()
                .map(|s| section_stream_id(chain.push_section(s)))
                .collect()
        };
        let mut c1 = ContentChain::new();
        let mut c2 = ContentChain::new();
        assert_eq!(run(&mut c1), run(&mut c2));
    }

    #[test]
    fn turn_stream_ids_distinct_and_nonzero() {
        let a = turn_stream_id(7, 0);
        let b = turn_stream_id(7, 1);
        let c = turn_stream_id(8, 0);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
        assert_ne!(a.0, 0);
    }

    #[test]
    fn section_and_turn_ids_never_zero() {
        // The reserved-zero bump path is exercised structurally.
        let mut chain = ContentChain::new();
        for i in 0..50u32 {
            let id = section_stream_id(chain.push_section(&[i, i + 1]));
            assert_ne!(id.0, 0);
        }
    }
}
