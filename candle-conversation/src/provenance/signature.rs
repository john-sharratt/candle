//! Binary directional signature types and R16 block extraction.
//!
//! # Types
//!
//! - [`TokenSignature`] — 128-bit sign-bit vector for one token.  Agreement
//!   between two signatures is XNOR popcount in [0, 128]; random baseline = 64.
//! - [`TurnSignatures`] — per-token signatures for one sealed KV block
//!   (32 tokens).  Produced on the index side by the post-Done seal and
//!   on the query side by the scheduler's in-decode reprojection path.
//!
//! # R16 layout
//!
//! The R16 KV format stores raw F16 Q vectors alongside K values, arranged as
//! `[head][palette][token][sub_dim]` with 4 palettes of `head_dim / 4` floats
//! each.  Extraction here uses head 0 only (multi-head XOR folding was removed
//! as dead code — single-head sign bits are sufficient for retrieval).

use candle_nn::kv_cache::CHUNK_SIZE;

// ── TokenSignature ────────────────────────────────────────────────────────────

/// 128-bit binary directional signature for one token.
///
/// Each bit encodes the sign of one dimension of the Q vector captured at
/// index time.  With head_dim=128, this covers all dimensions of one KV head,
/// split evenly across 4 palettes (32 dims per palette).
///
/// Agreement = `popcount(XNOR(stored, probe))` ∈ [0, 128].
/// Baseline (random) agreement is 64; useful signal starts around 80–90.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TokenSignature {
    bits: [u8; 16], // 128 bits packed as 16 bytes, LSB-first
}

impl TokenSignature {
    /// Bytes per signature: 128 sign bits packed LSB-first.
    pub const BYTE_LEN: usize = 16;

    /// Build a signature from raw Q float values.
    ///
    /// Takes the first 128 values of `q_values`.  Each bit is `sign(q[i]) > 0`.
    /// If `q_values` is shorter than 128 the remaining bits are zero.
    pub fn from_q_flat(q_values: &[f32]) -> Self {
        let mut bits = [0u8; 16];
        for (i, &v) in q_values.iter().take(128).enumerate() {
            if v > 0.0 {
                bits[i >> 3] |= 1u8 << (i & 7);
            }
        }
        Self { bits }
    }

    /// Build by XOR-aggregating sign bits across multiple Q arrays.
    ///
    /// Available for callers that want to XOR-fold several head or layer slices;
    /// the current production path uses [`from_q_flat`] on head 0 only.
    pub fn from_q_multi(arrays: &[&[f32]]) -> Self {
        let mut bits = [0u8; 16];
        for q_values in arrays {
            for (i, &v) in q_values.iter().take(128).enumerate() {
                if v > 0.0 {
                    bits[i >> 3] ^= 1u8 << (i & 7);
                }
            }
        }
        Self { bits }
    }

    /// Hamming agreement: number of bits that agree between `self` and `other`.
    ///
    /// Equivalent to `128 - hamming_distance`.  Maximum = 128, baseline random = 64.
    /// Compiles to two POPCNT instructions on x86-64.
    #[inline]
    pub fn agreement(&self, other: &Self) -> u32 {
        (!(self.as_u128() ^ other.as_u128())).count_ones()
    }

    /// Hamming distance: number of bits that differ.
    /// Compiles to a single POPCNT instruction on x86-64.
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.as_u128() ^ other.as_u128()).count_ones()
    }

    /// Raw bits as u128 for efficient bulk operations.
    #[inline]
    pub fn as_u128(&self) -> u128 {
        u128::from_le_bytes(self.bits)
    }

    /// Construct from a raw u128.
    #[inline]
    pub fn from_u128(v: u128) -> Self {
        Self { bits: v.to_le_bytes() }
    }

    /// Raw byte slice (`BYTE_LEN` bytes, LSB-first).
    #[inline]
    pub fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.bits
    }

    /// Reconstruct from a raw `BYTE_LEN`-byte slice as written by
    /// `ProvenanceFile::append`.
    #[inline]
    pub fn from_bytes(b: &[u8; Self::BYTE_LEN]) -> Self {
        Self { bits: *b }
    }
}

// ── TurnSignatures ────────────────────────────────────────────────────────────

/// Per-token binary directional signatures for one sealed KV block.
///
/// Produced on two paths:
/// - **Index side**: the post-Done seal extracts Q sign-bits inline for each new
///   block and appends the result to `ProvenanceFile`.
/// - **Query side**: the scheduler's in-decode reprojection path
///   extracts Q sign-bits from the live decode at each cadence trigger
///   for use by `BdpScanner`.
#[derive(Clone, Debug, Default)]
pub struct TurnSignatures {
    pub sigs: Vec<TokenSignature>,
}

impl TurnSignatures {
    pub fn from_sigs(it: impl IntoIterator<Item = TokenSignature>) -> Self {
        Self { sigs: it.into_iter().collect() }
    }

    /// Build from raw Q float data for a block of tokens.
    ///
    /// `q_flat` layout: `[token][dim]` — first `n_tokens` tokens, each with
    /// `head_dim` float values.
    pub fn from_q_flat_token_major(q_flat: &[f32], n_tokens: usize, head_dim: usize) -> Self {
        let mut sigs = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let start = t * head_dim;
            let end = (start + head_dim).min(q_flat.len());
            sigs.push(TokenSignature::from_q_flat(&q_flat[start..end]));
        }
        Self { sigs }
    }
}

// ── R16 extraction ────────────────────────────────────────────────────────────

// N_PALETTE is 4 (for head_dim=128, sub_head_dim=32).
const N_PALETTE: usize = 4;

/// Convert R16 Q data for one KV block into a `TurnSignatures`.
///
/// `q_flat` layout: `[head][palette][token][sub_dim]` as returned by
/// `dump_r16_kv_for_provenance`.  Only head 0 is used — production
/// provenance should call [`r16_block_to_turn_signatures_mh`] instead.
pub fn r16_block_to_turn_signatures(
    q_flat: &[f32],
    n_kv_head: usize,
    head_dim: usize,
    n_tokens_in_block: usize,
) -> TurnSignatures {
    let sub_head_dim = (head_dim / N_PALETTE).max(1);
    let elems_per_subband = CHUNK_SIZE * sub_head_dim;
    let floats_per_head = N_PALETTE * elems_per_subband;

    if n_kv_head == 0 || q_flat.len() < floats_per_head {
        return TurnSignatures::default();
    }

    let n_tokens = n_tokens_in_block.min(CHUNK_SIZE);
    let mut sigs = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let mut q_token = Vec::with_capacity(head_dim);
        for p in 0..N_PALETTE {
            let subband_base = p * elems_per_subband;
            let token_base = subband_base + t * sub_head_dim;
            let end = (token_base + sub_head_dim).min(q_flat.len());
            if token_base < q_flat.len() {
                q_token.extend_from_slice(&q_flat[token_base..end]);
            }
        }
        sigs.push(TokenSignature::from_q_flat(&q_token));
    }
    TurnSignatures { sigs }
}

/// Multi-head version of [`r16_block_to_turn_signatures`].
///
/// XOR-folds the sign bits of **all** `n_kv_head` KV heads (each contributing
/// 128 bits) into a single 128-bit [`TokenSignature`] per token.
///
/// `q_flat` layout: `[head][palette][token][sub_dim]`.
pub fn r16_block_to_turn_signatures_mh(
    q_flat: &[f32],
    n_kv_head: usize,
    head_dim: usize,
    n_tokens_in_block: usize,
) -> TurnSignatures {
    let sub_head_dim = (head_dim / N_PALETTE).max(1);
    let elems_per_subband = CHUNK_SIZE * sub_head_dim;
    let floats_per_head = N_PALETTE * elems_per_subband;

    if n_kv_head == 0 || q_flat.len() < floats_per_head {
        return TurnSignatures::default();
    }

    let n_heads = n_kv_head.min(q_flat.len() / floats_per_head).max(1);
    let n_tokens = n_tokens_in_block.min(CHUNK_SIZE);

    // Scratch buffers — one per head, reused across tokens.
    let mut head_bufs: Vec<Vec<f32>> =
        (0..n_heads).map(|_| Vec::with_capacity(head_dim)).collect();

    let mut sigs = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        for (h, buf) in head_bufs.iter_mut().enumerate() {
            buf.clear();
            let head_start = h * floats_per_head;
            for p in 0..N_PALETTE {
                let palette_base = head_start + p * elems_per_subband;
                let token_base = palette_base + t * sub_head_dim;
                let end = (token_base + sub_head_dim).min(q_flat.len());
                if token_base < q_flat.len() {
                    buf.extend_from_slice(&q_flat[token_base..end]);
                }
            }
        }
        let refs: Vec<&[f32]> = head_bufs.iter().map(|v| v.as_slice()).collect();
        sigs.push(TokenSignature::from_q_multi(&refs));
    }
    TurnSignatures { sigs }
}

/// XOR-fold two [`TurnSignatures`] token-by-token into a single merged set.
///
/// Used to combine dual-layer signatures: if `a` encodes multi-head sign bits
/// from the band-start layer and `b` from the band-centre layer, the merged
/// result encodes both jointly in one 128-bit signature per token.
/// Length is `min(a.len(), b.len())`.
pub fn merge_turn_signatures_xor(a: &TurnSignatures, b: &TurnSignatures) -> TurnSignatures {
    let n = a.sigs.len().min(b.sigs.len());
    TurnSignatures {
        sigs: a.sigs[..n]
            .iter()
            .zip(b.sigs[..n].iter())
            .map(|(sa, sb)| TokenSignature::from_u128(sa.as_u128() ^ sb.as_u128()))
            .collect(),
    }
}

/// Extract `TurnSignatures` from the raw output of `dump_r16_kv_for_provenance`.
///
/// `blocks`: each element is `(block_idx, k_flat, v_flat, q_flat)`.
/// Returns one `TurnSignatures` per block in block order.
pub fn extract_signatures_from_r16_dump(
    blocks: &[(usize, Vec<f32>, Vec<f32>, Vec<f32>)],
    n_kv_head: usize,
    head_dim: usize,
    tokens_per_block: usize,
) -> Vec<TurnSignatures> {
    blocks
        .iter()
        .map(|(_, _k, _v, q)| r16_block_to_turn_signatures(q, n_kv_head, head_dim, tokens_per_block))
        .collect()
}

/// Multi-head version of [`extract_signatures_from_r16_dump`].
///
/// Folds all `n_kv_head` heads via XOR into each token's 128-bit signature.
pub fn extract_mh_signatures_from_r16_dump(
    blocks: &[(usize, Vec<f32>, Vec<f32>, Vec<f32>)],
    n_kv_head: usize,
    head_dim: usize,
    tokens_per_block: usize,
) -> Vec<TurnSignatures> {
    blocks
        .iter()
        .map(|(_, _k, _v, q)| {
            r16_block_to_turn_signatures_mh(q, n_kv_head, head_dim, tokens_per_block)
        })
        .collect()
}


// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_positive_dims_gives_all_ones() {
        let q = vec![1.0_f32; 128];
        assert_eq!(TokenSignature::from_q_flat(&q).as_u128(), u128::MAX);
    }

    #[test]
    fn all_negative_dims_gives_all_zeros() {
        let q = vec![-1.0_f32; 128];
        assert_eq!(TokenSignature::from_q_flat(&q).as_u128(), 0);
    }

    #[test]
    fn agreement_with_self_is_128() {
        let q: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let sig = TokenSignature::from_q_flat(&q);
        assert_eq!(sig.agreement(&sig), 128);
    }

    #[test]
    fn agreement_plus_distance_is_128() {
        let q1: Vec<f32> = (0..128).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let q2: Vec<f32> = (0..128).map(|i| if i % 5 == 0 { 1.0 } else { -1.0 }).collect();
        let s1 = TokenSignature::from_q_flat(&q1);
        let s2 = TokenSignature::from_q_flat(&q2);
        assert_eq!(s1.agreement(&s2) + s1.hamming_distance(&s2), 128);
    }

    fn make_q_flat(n_kv_head: usize, head_dim: usize, val: f32) -> Vec<f32> {
        let sub_head_dim = head_dim / N_PALETTE;
        let floats_per_head = N_PALETTE * CHUNK_SIZE * sub_head_dim;
        vec![val; n_kv_head * floats_per_head]
    }

    #[test]
    fn positive_q_gives_all_ones_signature() {
        let q = make_q_flat(4, 128, 1.0);
        let sigs = r16_block_to_turn_signatures(&q, 4, 128, CHUNK_SIZE);
        assert_eq!(sigs.sigs.len(), CHUNK_SIZE);
        for s in &sigs.sigs {
            assert_eq!(s.as_u128(), u128::MAX);
        }
    }

    #[test]
    fn negative_q_gives_all_zeros_signature() {
        let q = make_q_flat(4, 128, -1.0);
        let sigs = r16_block_to_turn_signatures(&q, 4, 128, CHUNK_SIZE);
        for s in &sigs.sigs {
            assert_eq!(s.as_u128(), 0);
        }
    }

    #[test]
    fn partial_block_produces_correct_count() {
        let q = make_q_flat(4, 128, 1.0);
        let sigs = r16_block_to_turn_signatures(&q, 4, 128, 10);
        assert_eq!(sigs.sigs.len(), 10);
    }

    #[test]
    fn extract_from_dump_produces_one_per_block() {
        let q = make_q_flat(4, 128, 1.0);
        let blocks = vec![
            (0, vec![], vec![], q.clone()),
            (1, vec![], vec![], q.clone()),
        ];
        let result = extract_signatures_from_r16_dump(&blocks, 4, 128, CHUNK_SIZE);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].sigs.len(), CHUNK_SIZE);
    }
}
