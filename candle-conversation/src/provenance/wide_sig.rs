//! Per-token Q signature — the decode→decode (`Q·Q`) consensus substrate.
//!
//! A [`WideQSig`] holds one token's `sign(Q)` as `n_heads × head_dim` bits. At seal, each
//! token's *raw* signature is captured from R16 for every head of every layer
//! (`from_band`), then [`fold_provenance`]-folded to the compact locked signature: 3 layer
//! groups × 4 KV-heads = 12 heads (1536 bits, 16× smaller than the raw 192-head form). The
//! folded per-token history is stored on the turn's substrate entry as its `WideQSig`
//! record, 1:1 aligned with the turn's `Tokens` record, and retrieval scores the groups
//! by a z-score late-fusion vote. See `docs/tool_selection_provenance_results.md` §23.

/// Sign bits packed per head: `head_dim` bits → `head_dim / 64` u64 words.
/// (`head_dim = 128` ⇒ 2 words per head.)
#[inline]
fn words_per_head(head_dim: usize) -> usize {
    head_dim.div_ceil(64)
}

/// One token's `sign(Q)` as packed sign bits.
///
/// `words` holds `n_heads × words_per_head(head_dim)` u64: bit `i` of head `h` is set iff
/// that head's `Q[i] >= 0`, heads in `(group/layer × n_kv_head + head)` order. Raw from
/// [`WideQSig::from_band`] this is `n_layers × n_kv_head` heads (192); after
/// [`fold_provenance`] it is `PROV_FOLD_SIZES.len() × n_kv_head` group-heads (12).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WideQSig {
    /// Number of heads this signature spans (raw: `n_layers × n_kv_head`; folded:
    /// `n_groups × n_kv_head`).
    pub n_heads: u16,
    /// Sign-bit words, `n_heads × words_per_head` long.
    pub words: Vec<u64>,
}

impl WideQSig {
    /// Pack `sign(Q)` from a flat band of `n_heads × head_dim` f32 (per-layer-per-head Q,
    /// `[layer][head][dim]` order). Bit set ⇔ value `>= 0`.
    pub fn from_band(band: &[f32], head_dim: usize) -> Self {
        let wph = words_per_head(head_dim);
        let n_heads = if head_dim == 0 {
            0
        } else {
            band.len() / head_dim
        };
        let mut words = vec![0u64; n_heads * wph];
        for h in 0..n_heads {
            let base = h * head_dim;
            for i in 0..head_dim {
                if band[base + i] >= 0.0 {
                    words[h * wph + i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        Self {
            n_heads: n_heads as u16,
            words,
        }
    }

    /// Total set sign bits across all heads — a cheap "is this real, non-empty" probe for
    /// the inspector (a near-half-set count means genuine sign data, all-0/all-set is dead).
    pub fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Words per head for this signature (`words.len() / n_heads`).
    pub fn words_per_head(&self) -> usize {
        if self.n_heads == 0 {
            0
        } else {
            self.words.len() / self.n_heads as usize
        }
    }
}

/// KV heads per layer (`n_kv_head`) — the head count kept separate through the fold.
pub const PROV_HEADS_PER_LAYER: usize = 4;
/// Locked provenance-signature fold: layer group sizes (bottom→top). `[46, 1, 1]` folds
/// L0–45 into one noise-absorbing group and keeps L46, L47 (where the tool-identity
/// signal lives) as their own groups. See `docs/tool_selection_provenance_results.md` §23.
pub const PROV_FOLD_SIZES: &[usize] = &[46, 1, 1];
/// Per-layer decorrelating bit-rotation: layer `p` within a group is rotated
/// `p × PROV_FOLD_SHIFT` bits before the XOR, so correlated layers are staggered out of
/// phase instead of cancelling dim-aligned.
pub const PROV_FOLD_SHIFT: usize = 32;

/// Rotate a head's `head_dim`-bit sign vector (little-endian across `wph` u64 words) left
/// by `r` bits. Only `head_dim == 128` (`wph == 2`) rotates; other widths are returned
/// unchanged (the fold shift is a 128-bit-head decorrelation trick).
fn rotate_head(src: &[u64], r: usize) -> (u64, u64) {
    if src.len() != 2 {
        return (
            src.first().copied().unwrap_or(0),
            src.get(1).copied().unwrap_or(0),
        );
    }
    let r = (r % 128) as u32;
    if r == 0 {
        return (src[0], src[1]);
    }
    let v = (src[0] as u128) | ((src[1] as u128) << 64);
    let rot = v.rotate_left(r);
    (rot as u64, (rot >> 64) as u64)
}

/// Fold a raw per-token wide `sign(Q)` (all layers × all heads) into the compact locked
/// provenance signature: [`PROV_FOLD_SIZES`] layer-groups × [`PROV_HEADS_PER_LAYER`]
/// heads. Each group's head is the XOR of its layers' per-head sign bits (heads kept
/// **separate** — they carry independent signal), with layer `p` staggered by
/// `p × PROV_FOLD_SHIFT`. For `head_dim = 128` this yields 12 heads = 1536 bits, 16×
/// smaller than the full 192-head wide-Q. Retrieval scores the groups by z-score
/// late-fusion vote. See `docs/tool_selection_provenance_results.md` §23.
pub fn fold_provenance(raw: &WideQSig) -> WideQSig {
    let wph = raw.words_per_head();
    let n_heads = raw.n_heads as usize;
    let n_layers = n_heads / PROV_HEADS_PER_LAYER;
    if wph == 0 || n_layers == 0 {
        return raw.clone();
    }
    let n_groups = PROV_FOLD_SIZES.len();
    let out_heads = n_groups * PROV_HEADS_PER_LAYER;
    let mut words = vec![0u64; out_heads * wph];
    let mut l0 = 0usize;
    for (g, &sz) in PROV_FOLD_SIZES.iter().enumerate() {
        for (p, l) in (l0..(l0 + sz).min(n_layers)).enumerate() {
            for h in 0..PROV_HEADS_PER_LAYER {
                let bi = (l * PROV_HEADS_PER_LAYER + h) * wph;
                let bo = (g * PROV_HEADS_PER_LAYER + h) * wph;
                let (w0, w1) = rotate_head(&raw.words[bi..bi + wph], p * PROV_FOLD_SHIFT);
                if wph == 2 {
                    words[bo] ^= w0;
                    words[bo + 1] ^= w1;
                } else {
                    for i in 0..wph {
                        words[bo + i] ^= raw.words[bi + i];
                    }
                }
            }
        }
        l0 += sz;
        if l0 >= n_layers {
            break;
        }
    }
    WideQSig {
        n_heads: out_heads as u16,
        words,
    }
}

/// Record magic: `WQS` + a single ASCII version digit.
///
/// The substrate is an append-only log that outlives any one daemon build, so a
/// single store holds records written by every version that ever ran against it.
/// The decoder therefore reads **every** version it has emitted; the encoder
/// always writes [`WIDE_VERSION_CURRENT`]. Mixed-version records coexist.
///
/// | version | header | `n_tokens` | notes |
/// |---------|--------|------------|-------|
/// | `WQS3`  | 10 B   | `u16` @4   | wraps mod 65536 past 65,535 tokens — a long turn decodes back as only its leading `len % 65536` signatures |
/// | `WQS4`  | 12 B   | `u32` @4   | current; the widened count fixes that truncation |
///
/// Widening `n_tokens` shifted `n_heads` / `words_per_head`, so the version digit
/// is what keeps a v3 record from being parsed at v4 offsets (which would read
/// garbage counts rather than fail).
const WIDE_MAGIC_PREFIX: &[u8; 3] = b"WQS";
const WIDE_VERSION_CURRENT: u8 = b'4';
const WIDE_HEADER: usize = 12;

/// Header fields of one encoded record: `(header_len, n_tokens, n_heads,
/// words_per_head)`. `None` for a foreign or unknown-version blob.
fn parse_wide_header(bytes: &[u8]) -> Option<(usize, usize, u16, usize)> {
    if bytes.len() < 4 || &bytes[0..3] != WIDE_MAGIC_PREFIX {
        return None;
    }
    match bytes[3] {
        b'3' => {
            if bytes.len() < 10 {
                return None;
            }
            Some((
                10,
                u16::from_le_bytes([bytes[4], bytes[5]]) as usize,
                u16::from_le_bytes([bytes[6], bytes[7]]),
                u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
            ))
        }
        b'4' => {
            if bytes.len() < 12 {
                return None;
            }
            Some((
                12,
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
                u16::from_le_bytes([bytes[8], bytes[9]]),
                u16::from_le_bytes([bytes[10], bytes[11]]) as usize,
            ))
        }
        _ => None,
    }
}

/// Encode a turn's **complete per-token** wide `sign(Q)` history — one [`WideQSig`]
/// per token, in token order — to the opaque bytes stored in the substrate's
/// `WideQSig` record. Continuous, seal-captured, per-token, all heads and layers.
/// All tokens share `n_heads`/`words_per_head`.
pub fn encode_wide_sigs(sigs: &[WideQSig]) -> Vec<u8> {
    let n_heads = sigs.first().map(|w| w.n_heads).unwrap_or(0);
    let wph = sigs.first().map(WideQSig::words_per_head).unwrap_or(0);
    let mut out = Vec::with_capacity(WIDE_HEADER + sigs.len() * n_heads as usize * wph * 8);
    out.extend_from_slice(WIDE_MAGIC_PREFIX);
    out.push(WIDE_VERSION_CURRENT);
    out.extend_from_slice(&(sigs.len() as u32).to_le_bytes());
    out.extend_from_slice(&n_heads.to_le_bytes());
    out.extend_from_slice(&(wph as u16).to_le_bytes());
    for tok in sigs {
        for &word in &tok.words {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    out
}

/// Decode a turn's complete per-token wide `sign(Q)` history, dispatching on the
/// record's version so every layout ever written stays readable. `None` on a
/// foreign / unknown-version blob or a truncated payload.
pub fn decode_wide_sigs(bytes: &[u8]) -> Option<Vec<WideQSig>> {
    let (header_len, n_tokens, n_heads, wph) = parse_wide_header(bytes)?;
    let words_per_tok = n_heads as usize * wph;
    if bytes.len() < header_len + n_tokens * words_per_tok * 8 {
        return None;
    }
    let mut sigs = Vec::with_capacity(n_tokens);
    let mut off = header_len;
    for _ in 0..n_tokens {
        let mut words = Vec::with_capacity(words_per_tok);
        for _ in 0..words_per_tok {
            words.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
            off += 8;
        }
        sigs.push(WideQSig { n_heads, words });
    }
    Some(sigs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_provenance_groups_layers_with_stagger() {
        // Raw: 48 layers × 4 heads, head_dim 128 (2 words/head) = 384 words, all zero
        // except head 0 of layers 0, 1, 46, 47.
        let mut raw = WideQSig {
            n_heads: 48 * PROV_HEADS_PER_LAYER as u16,
            words: vec![0u64; 48 * PROV_HEADS_PER_LAYER * 2],
        };
        let head = |l: usize| (l * PROV_HEADS_PER_LAYER + 0) * 2; // word index of layer l, head 0
        raw.words[head(0)] = 0xF0;
        raw.words[head(0) + 1] = 0x0F;
        raw.words[head(1)] = 0xAA;
        raw.words[head(1) + 1] = 0xBB;
        raw.words[head(46)] = 0xABCD;
        raw.words[head(46) + 1] = 0x1234;
        raw.words[head(47)] = 0xBEEF;
        raw.words[head(47) + 1] = 0x5678;

        let f = fold_provenance(&raw);
        // 3 groups × 4 heads = 12 heads, 24 words.
        assert_eq!(f.n_heads, 12);
        assert_eq!(f.words.len(), 24);

        // group 0 (layers 0..45): head0 = L0 XOR rotate(L1, 32). Rotating (0xAA, 0xBB)
        // left 32 bits inside the 128-bit head → (0xAA<<32, 0xBB<<32).
        assert_eq!(f.words[0], 0xF0 ^ (0xAAu64 << 32));
        assert_eq!(f.words[1], 0x0F ^ (0xBBu64 << 32));
        // group 1 (layer 46, position 0 → no rotation): passes through unchanged.
        assert_eq!(f.words[(4 + 0) * 2], 0xABCD);
        assert_eq!(f.words[(4 + 0) * 2 + 1], 0x1234);
        // group 2 (layer 47): passes through unchanged.
        assert_eq!(f.words[(8 + 0) * 2], 0xBEEF);
        assert_eq!(f.words[(8 + 0) * 2 + 1], 0x5678);
        // heads 1..3 of every group are zero (only head 0 was set).
        for h in [1usize, 2, 3, 5, 6, 7, 9, 10, 11] {
            assert_eq!(f.words[h * 2], 0, "head {h} word0");
            assert_eq!(f.words[h * 2 + 1], 0, "head {h} word1");
        }
    }

    #[test]
    fn from_band_packs_signs() {
        // head_dim 4, 2 heads. head0 = [+,-,+,-], head1 = [-,-,+,0] (0 counts as +).
        let band = vec![1.0, -1.0, 0.5, -0.5, -2.0, -1.0, 3.0, 0.0];
        let sig = WideQSig::from_band(&band, 4);
        assert_eq!(sig.n_heads, 2);
        assert_eq!(sig.words_per_head(), 1); // 4 bits → 1 word
        assert_eq!(sig.words[0] & 0xF, 0b0101, "head0: bits 0,2 set");
        assert_eq!(sig.words[1] & 0xF, 0b1100, "head1: bits 2,3 set");
        assert_eq!(sig.popcount(), 4);
    }

    #[test]
    fn wide_sigs_history_roundtrips() {
        let mk = |seed: usize| {
            let band: Vec<f32> = (0..4 * 128)
                .map(|i| if (i + seed) % 3 == 0 { 1.0 } else { -1.0 })
                .collect();
            WideQSig::from_band(&band, 128)
        };
        let history: Vec<WideQSig> = (0..5).map(mk).collect();
        let bytes = encode_wide_sigs(&history);
        assert_eq!(decode_wide_sigs(&bytes), Some(history));
        assert_eq!(decode_wide_sigs(b"nope"), None);
        assert_eq!(decode_wide_sigs(&encode_wide_sigs(&[])), Some(vec![]));
    }

    /// The 12-byte header, byte for byte: `WQS4` · n_tokens(u32 LE) ·
    /// n_heads(u16 LE) · words_per_head(u16 LE), then the packed words.
    #[test]
    fn wide_sigs_header_is_exact_bytes() {
        let sigs = vec![
            WideQSig {
                n_heads: 1,
                words: vec![0x0102_0304_0506_0708],
            },
            WideQSig {
                n_heads: 1,
                words: vec![0x1112_1314_1516_1718],
            },
        ];
        let bytes = encode_wide_sigs(&sigs);
        assert_eq!(
            &bytes[..WIDE_HEADER],
            &[
                b'W', b'Q', b'S', b'4', // magic
                2, 0, 0, 0, // n_tokens = 2 (u32 LE)
                1, 0, // n_heads = 1 (u16 LE)
                1, 0, // words_per_head = 1 (u16 LE)
            ],
            "wide-sig header layout"
        );
        assert_eq!(bytes.len(), WIDE_HEADER + 2 * 8);
        assert_eq!(
            &bytes[WIDE_HEADER..WIDE_HEADER + 8],
            &0x0102_0304_0506_0708u64.to_le_bytes()
        );
    }

    /// A history longer than `u16::MAX` round-trips in full. The old header
    /// encoded `n_tokens` as `u16`, so 65,537 tokens wrapped to 1 and the decode
    /// returned a single leading signature — silently discarding a large scope's
    /// entire provenance.
    #[test]
    fn wide_sigs_history_beyond_u16_roundtrips() {
        let n = u16::MAX as usize + 2; // 65_537
        let history: Vec<WideQSig> = (0..n)
            .map(|i| WideQSig {
                n_heads: 1,
                words: vec![i as u64],
            })
            .collect();
        let bytes = encode_wide_sigs(&history);
        let decoded = decode_wide_sigs(&bytes).expect("decodes");
        assert_eq!(decoded.len(), n, "every token survives the round-trip");
        assert_eq!(decoded.first().unwrap().words[0], 0);
        assert_eq!(decoded.last().unwrap().words[0], (n - 1) as u64);
    }

    /// A record written under the old 10-byte `WQS3` layout still decodes: the
    /// substrate holds records from every version that ran against it, so both
    /// coexist. Parsed at v3 offsets (`u16` n_tokens at 4, header 10), NOT v4's.
    #[test]
    fn legacy_wqs3_record_still_decodes() {
        let mut legacy = Vec::new();
        legacy.extend_from_slice(b"WQS3");
        legacy.extend_from_slice(&2u16.to_le_bytes()); // n_tokens (u16 @ offset 4)
        legacy.extend_from_slice(&1u16.to_le_bytes()); // n_heads
        legacy.extend_from_slice(&1u16.to_le_bytes()); // words_per_head
        legacy.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        legacy.extend_from_slice(&0x1112_1314_1516_1718u64.to_le_bytes());
        assert_eq!(
            decode_wide_sigs(&legacy),
            Some(vec![
                WideQSig {
                    n_heads: 1,
                    words: vec![0x0102_0304_0506_0708],
                },
                WideQSig {
                    n_heads: 1,
                    words: vec![0x1112_1314_1516_1718],
                },
            ]),
            "a v3 record decodes at v3 offsets, coexisting with v4",
        );
    }

    /// Version dispatch: a v3 blob and a v4 blob carrying the SAME signatures
    /// decode identically, even though their byte layouts differ (10- vs 12-byte
    /// header, `u16` vs `u32` count).
    #[test]
    fn v3_and_v4_records_decode_to_the_same_history() {
        let sigs = vec![
            WideQSig {
                n_heads: 1,
                words: vec![0xAAAA_BBBB_CCCC_DDDD],
            },
            WideQSig {
                n_heads: 1,
                words: vec![0x1122_3344_5566_7788],
            },
        ];
        // v4 is what the encoder writes today.
        let v4 = encode_wide_sigs(&sigs);
        assert_eq!(&v4[..4], b"WQS4");

        // Hand-build the equivalent v3 record (the retired encoder's layout).
        let mut v3 = Vec::new();
        v3.extend_from_slice(b"WQS3");
        v3.extend_from_slice(&(sigs.len() as u16).to_le_bytes());
        v3.extend_from_slice(&1u16.to_le_bytes()); // n_heads
        v3.extend_from_slice(&1u16.to_le_bytes()); // words_per_head
        for s in &sigs {
            v3.extend_from_slice(&s.words[0].to_le_bytes());
        }

        assert_eq!(decode_wide_sigs(&v3), decode_wide_sigs(&v4));
        assert_eq!(decode_wide_sigs(&v3), Some(sigs));
    }

    /// An unknown version digit (a future format this build can't read) decodes
    /// to `None` rather than mis-parsing — the turn is treated as unsigned.
    #[test]
    fn unknown_version_is_rejected() {
        let mut future = Vec::new();
        future.extend_from_slice(b"WQS9");
        future.extend_from_slice(&1u32.to_le_bytes());
        future.extend_from_slice(&1u16.to_le_bytes());
        future.extend_from_slice(&1u16.to_le_bytes());
        future.extend_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
        assert_eq!(decode_wide_sigs(&future), None);
    }
}
