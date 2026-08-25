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
        let n_heads = band.len().checked_div(head_dim).unwrap_or(0);
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

/// KV heads per layer (`n_kv_head`) on the stack the locked fold was measured against.
///
/// **Kept as the Qwen3-30B value, not as the rule.** The rule is "the model's
/// `n_kv_head`" — see [`FoldParams::derive`]. This constant survives so the
/// bit-identity gate has something to compare against.
pub const PROV_HEADS_PER_LAYER: usize = 4;
/// Locked provenance-signature fold: layer group sizes (bottom→top). `[46, 1, 1]` folds
/// L0–45 into one noise-absorbing group and keeps L46, L47 (where the tool-identity
/// signal lives) as their own groups. See `docs/tool_selection_provenance_results.md` §23.
///
/// `[46, 1, 1]` at 48 layers is `[n − 2, 1, 1]` written out longhand — the
/// existing constant already *is* the general rule.
pub const PROV_FOLD_SIZES: &[usize] = &[46, 1, 1];
/// Per-layer decorrelating bit-rotation: layer `p` within a group is rotated
/// `p × PROV_FOLD_SHIFT` bits before the XOR, so correlated layers are staggered out of
/// phase instead of cancelling dim-aligned. `32` is `head_dim / 4` at `head_dim` 128.
pub const PROV_FOLD_SHIFT: usize = 32;

/// The parameters a signature was folded with.
///
/// **A signature is only comparable to another under the same fold.** Hamming
/// distance between differently-folded bit vectors is not a degraded measure of
/// anything — it is meaningless, and the scorer returns a confident number for
/// it. So the parameters ride on the record and a mismatch is refused rather
/// than scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldParams {
    /// KV heads kept separate through the fold — the model's `n_kv_head`.
    pub heads_per_layer: usize,
    /// Layer-group sizes, bottom→top.
    pub group_sizes: [usize; 3],
    /// Per-layer decorrelating rotation, in bits.
    pub shift: usize,
    /// Width of one head's sign vector, in bits.
    pub head_dim: usize,
}

impl FoldParams {
    /// Derive the fold from a model's geometry. **Every derivation is an
    /// identity on Qwen3-30B** (4 KV heads, 48 capture layers, head_dim 128 →
    /// `[46,1,1]`, shift 32), which is what lets all four change at once
    /// without re-measuring the outgoing model.
    ///
    /// `n_capture_layers` is the number of layers that actually have a Q to
    /// capture — on a hybrid that is the attention layers, not the stack depth.
    pub fn derive(n_kv_head: usize, n_capture_layers: usize, head_dim: usize) -> Self {
        // `[n − 2, 1, 1]`: one noise-absorbing lower group, then the top two
        // layers on their own, where identity was measured to live.
        let top = n_capture_layers.saturating_sub(2).max(1);
        Self {
            heads_per_layer: n_kv_head.max(1),
            group_sizes: [top, 1, 1],
            shift: (head_dim / 4).max(1),
            head_dim,
        }
    }

    /// The Qwen3-30B fold, spelled from the locked constants — the reference
    /// the bit-identity gate compares against.
    pub fn locked() -> Self {
        Self {
            heads_per_layer: PROV_HEADS_PER_LAYER,
            group_sizes: [PROV_FOLD_SIZES[0], PROV_FOLD_SIZES[1], PROV_FOLD_SIZES[2]],
            shift: PROV_FOLD_SHIFT,
            head_dim: 128,
        }
    }

    /// Heads in a folded signature: one per group per KV head.
    pub fn folded_heads(&self) -> usize {
        self.group_sizes.len() * self.heads_per_layer
    }
}

/// The fold this process's model produces — set once at engine construction.
///
/// A genuine singleton, not configuration: one process loads one model, and the
/// fold is a function of that model's geometry. It exists so the substrate read
/// paths can refuse a record folded under a *different* geometry without every
/// caller threading the parameters down to them.
///
/// Unset means "do not check" — the shape it takes in tests, which construct
/// signatures directly rather than through a model.
static ACTIVE_FOLD: std::sync::OnceLock<FoldParams> = std::sync::OnceLock::new();

/// Publish this process's fold. Idempotent; the first call wins.
pub fn set_active_fold(params: FoldParams) {
    let _ = ACTIVE_FOLD.set(params);
}

/// This process's fold, if an engine has published one.
pub fn active_fold() -> Option<FoldParams> {
    ACTIVE_FOLD.get().copied()
}

/// Decode a stored history, refusing one folded under a different geometry than
/// this process's model produces.
///
/// The substrate read path: when no fold has been published (tests), it is a
/// plain [`decode_wide_sigs`]; when one has, a mismatched record is refused with
/// a warning rather than scored into a confident number over incomparable bits.
pub fn decode_wide_sigs_for_scoring(bytes: &[u8]) -> Option<Vec<WideQSig>> {
    match active_fold() {
        Some(expected) => decode_wide_sigs_checked(bytes, expected),
        None => decode_wide_sigs(bytes),
    }
}

/// Rotate a head's `head_dim`-bit sign vector (little-endian across `wph` u64 words)
/// left by `r` bits, in place into `dst`.
///
/// **Word-wise, over any width.** The previous form took a `(u64, u64)` and
/// returned one, so it could not express a result wider than 128 bits at all
/// and bailed to the identity for every other width — which meant that at
/// `head_dim` 256 the decorrelating stagger silently did nothing and correlated
/// layers cancelled dim-aligned under the XOR. That is invisible: the fold
/// still produces a signature, it just carries less than it should.
fn rotate_head_into(dst: &mut [u64], src: &[u64], r: usize) {
    let wph = src.len();
    debug_assert_eq!(dst.len(), wph);
    if wph == 0 {
        return;
    }
    let bits = wph * 64;
    let r = r % bits;
    if r == 0 {
        dst.copy_from_slice(src);
        return;
    }
    let word_shift = r / 64;
    let bit_shift = r % 64;
    for i in 0..wph {
        // Bit `j` of the rotated vector comes from bit `j - r` of the source.
        let lo = src[(i + wph - word_shift) % wph];
        if bit_shift == 0 {
            dst[i] = lo;
        } else {
            let lower = src[(i + wph - word_shift - 1) % wph];
            dst[i] = (lo << bit_shift) | (lower >> (64 - bit_shift));
        }
    }
}

/// Fold a raw per-token wide `sign(Q)` (all layers × all heads) into the compact locked
/// provenance signature: [`PROV_FOLD_SIZES`] layer-groups × [`PROV_HEADS_PER_LAYER`]
/// heads. Each group's head is the XOR of its layers' per-head sign bits (heads kept
/// **separate** — they carry independent signal), with layer `p` staggered by
/// `p × PROV_FOLD_SHIFT`. For `head_dim = 128` this yields 12 heads = 1536 bits, 16×
/// smaller than the full 192-head wide-Q. Retrieval scores the groups by z-score
/// late-fusion vote. See `docs/tool_selection_provenance_results.md` §23.
pub fn fold_provenance(raw: &WideQSig) -> WideQSig {
    fold_provenance_with(raw, FoldParams::locked())
}

/// Fold under explicit parameters — [`fold_provenance`] with the model's own
/// geometry instead of the locked constants.
///
/// Refuses to emit a group it cannot fill. `[46,1,1]` on a 10-layer stack does
/// exactly that today: group 0 swallows all ten and groups 1 and 2 come back
/// **all zero**, so two thirds of the signature is empty and the
/// identity-bearing top layers vanish. An all-zero group is not a weak
/// signature, it is a scorer input that agrees with everything, so the fold
/// returns `None` rather than producing one.
pub fn fold_provenance_checked(raw: &WideQSig, params: FoldParams) -> Option<WideQSig> {
    fold_fits(raw.n_heads as usize, raw.words_per_head(), params)
        .then(|| fold_provenance_with(raw, params))
}

/// Whether `params` can fill all three groups from a signature of this shape.
///
/// Split out because it is a property of the **shape**, not of the bits: every
/// token of a turn has the same `n_heads` and `words_per_head`, so a capture
/// loop answers this once at entry rather than re-deriving the same verdict per
/// token. [`fold_provenance_checked`] keeps the fused form for callers holding a
/// single signature.
pub fn fold_fits(n_heads: usize, words_per_head: usize, params: FoldParams) -> bool {
    let n_layers = n_heads / params.heads_per_layer.max(1);
    if n_layers == 0 || words_per_head == 0 {
        return false;
    }
    // Every group must receive at least one layer.
    let mut covered = 0usize;
    for &sz in &params.group_sizes {
        if covered >= n_layers {
            return false;
        }
        covered += sz;
    }
    true
}

/// Fold without re-checking that `params` fits — for a caller that has already
/// answered [`fold_fits`] for this shape and is folding many signatures of it.
///
/// Folding under parameters that do not fit produces a signature with empty
/// groups, which scores as agreeing with everything, so this is only safe
/// behind that check. Prefer [`fold_provenance_checked`] for a one-off.
pub fn fold_provenance_fitted(raw: &WideQSig, params: FoldParams) -> WideQSig {
    fold_provenance_with(raw, params)
}

fn fold_provenance_with(raw: &WideQSig, params: FoldParams) -> WideQSig {
    let wph = raw.words_per_head();
    let n_heads = raw.n_heads as usize;
    let heads_per_layer = params.heads_per_layer.max(1);
    let n_layers = n_heads / heads_per_layer;
    if wph == 0 || n_layers == 0 {
        return raw.clone();
    }
    let n_groups = params.group_sizes.len();
    let out_heads = n_groups * heads_per_layer;
    let mut words = vec![0u64; out_heads * wph];
    let mut rotated = vec![0u64; wph];
    let mut l0 = 0usize;
    for (g, &sz) in params.group_sizes.iter().enumerate() {
        for (p, l) in (l0..(l0 + sz).min(n_layers)).enumerate() {
            for h in 0..heads_per_layer {
                let bi = (l * heads_per_layer + h) * wph;
                let bo = (g * heads_per_layer + h) * wph;
                // One path, every width. The old form rotated only at
                // `wph == 2` and XOR'd the *unrotated* words otherwise, so the
                // stagger was absent at any other head width — and a test
                // written against the rotate alone could not see it, because
                // the caller threw the rotate's result away.
                rotate_head_into(&mut rotated, &raw.words[bi..bi + wph], p * params.shift);
                for i in 0..wph {
                    words[bo + i] ^= rotated[i];
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
/// | `WQS4`  | 12 B   | `u32` @4   | the widened count fixes that truncation |
/// | `WQS5`  | 20 B   | `u32` @4   | current; **carries the fold parameters** |
///
/// Widening `n_tokens` shifted `n_heads` / `words_per_head`, so the version digit
/// is what keeps a v3 record from being parsed at v4 offsets (which would read
/// garbage counts rather than fail).
///
/// **Why v5 stamps the fold.** A signature is only comparable to another under
/// the same fold: Hamming distance between differently-folded bit vectors is not
/// a degraded measure of anything, it is meaningless, and the scorer returns a
/// confident number for it. The parameters were implicit for as long as exactly
/// one model ever wrote them; the moment a second geometry can, the record has
/// to say which fold produced it. A v3/v4 record decodes as the locked
/// Qwen3-30B fold, which is a fact about those records rather than a
/// compatibility path: it is the only fold that ever wrote one.
const WIDE_MAGIC_PREFIX: &[u8; 3] = b"WQS";
const WIDE_VERSION_CURRENT: u8 = b'5';
const WIDE_HEADER: usize = 20;

/// Everything one record's header carries.
struct WideHeader {
    header_len: usize,
    n_tokens: usize,
    n_heads: u16,
    wph: usize,
    /// The fold that produced these signatures.
    params: FoldParams,
}

/// Parse one encoded record's header. `None` for a foreign or unknown-version
/// blob.
fn parse_wide_header(bytes: &[u8]) -> Option<WideHeader> {
    if bytes.len() < 4 || &bytes[0..3] != WIDE_MAGIC_PREFIX {
        return None;
    }
    match bytes[3] {
        b'3' => {
            if bytes.len() < 10 {
                return None;
            }
            Some(WideHeader {
                header_len: 10,
                n_tokens: u16::from_le_bytes([bytes[4], bytes[5]]) as usize,
                n_heads: u16::from_le_bytes([bytes[6], bytes[7]]),
                wph: u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
                params: FoldParams::locked(),
            })
        }
        b'4' => {
            if bytes.len() < 12 {
                return None;
            }
            Some(WideHeader {
                header_len: 12,
                n_tokens: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
                n_heads: u16::from_le_bytes([bytes[8], bytes[9]]),
                wph: u16::from_le_bytes([bytes[10], bytes[11]]) as usize,
                params: FoldParams::locked(),
            })
        }
        b'5' => {
            if bytes.len() < WIDE_HEADER {
                return None;
            }
            Some(WideHeader {
                header_len: WIDE_HEADER,
                n_tokens: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
                n_heads: u16::from_le_bytes([bytes[8], bytes[9]]),
                wph: u16::from_le_bytes([bytes[10], bytes[11]]) as usize,
                params: FoldParams {
                    heads_per_layer: u16::from_le_bytes([bytes[12], bytes[13]]) as usize,
                    group_sizes: [
                        u16::from_le_bytes([bytes[14], bytes[15]]) as usize,
                        u16::from_le_bytes([bytes[16], bytes[17]]) as usize,
                        1,
                    ],
                    shift: u16::from_le_bytes([bytes[18], bytes[19]]) as usize,
                    head_dim: u16::from_le_bytes([bytes[10], bytes[11]]) as usize * 64,
                },
            })
        }
        _ => None,
    }
}

/// The fold a stored record was written under, or `None` for a foreign blob.
///
/// Read this before scoring a probe against a gallery: two records folded
/// differently must not be compared, and this is what makes the difference
/// visible rather than silently absorbed into the score.
pub fn wide_sig_fold_params(bytes: &[u8]) -> Option<FoldParams> {
    parse_wide_header(bytes).map(|h| h.params)
}

/// Encode a turn's **complete per-token** wide `sign(Q)` history — one [`WideQSig`]
/// per token, in token order — to the opaque bytes stored in the substrate's
/// `WideQSig` record. Continuous, seal-captured, per-token, all heads and layers.
/// All tokens share `n_heads`/`words_per_head`.
/// Written under the fold the local build derives — see
/// [`encode_wide_sigs_with`] to stamp an explicit one.
pub fn encode_wide_sigs(sigs: &[WideQSig]) -> Vec<u8> {
    let derived = sigs
        .first()
        .and_then(fold_params_of)
        .unwrap_or_else(FoldParams::locked);
    encode_wide_sigs_with(sigs, derived)
}

/// The fold a folded signature must have been produced by, recovered from its
/// own shape.
///
/// `heads_per_layer` and `head_dim` are readable directly (three groups, so
/// `n_heads / 3`; `wph × 64` bits per head), and `shift` follows from
/// `head_dim / 4`. The group sizes are **not** recoverable — they depend on the
/// capture-layer count, which the signature does not carry — so they are the one
/// thing the record has to state.
fn fold_params_of(sig: &WideQSig) -> Option<FoldParams> {
    let wph = sig.words_per_head();
    let n_heads = sig.n_heads as usize;
    if wph == 0 || n_heads == 0 || !n_heads.is_multiple_of(3) {
        return None;
    }
    let head_dim = wph * 64;
    Some(FoldParams {
        heads_per_layer: n_heads / 3,
        // Unknown from the shape alone; the caller stamping a real fold uses
        // `encode_wide_sigs_with`. Zero reads as "unstated" on comparison.
        group_sizes: [0, 1, 1],
        shift: (head_dim / 4).max(1),
        head_dim,
    })
}

/// Encode with an explicit fold stamped into the header.
pub fn encode_wide_sigs_with(sigs: &[WideQSig], params: FoldParams) -> Vec<u8> {
    let n_heads = sigs.first().map(|w| w.n_heads).unwrap_or(0);
    let wph = sigs.first().map(WideQSig::words_per_head).unwrap_or(0);
    let mut out = Vec::with_capacity(WIDE_HEADER + sigs.len() * n_heads as usize * wph * 8);
    out.extend_from_slice(WIDE_MAGIC_PREFIX);
    out.push(WIDE_VERSION_CURRENT);
    out.extend_from_slice(&(sigs.len() as u32).to_le_bytes());
    out.extend_from_slice(&n_heads.to_le_bytes());
    out.extend_from_slice(&(wph as u16).to_le_bytes());
    // The fold: heads-per-layer, the two lower group sizes (the third is always
    // 1 — `[n−2, 1, 1]`), and the decorrelating shift. `head_dim` is recovered
    // from `wph`, so it is not stored twice.
    out.extend_from_slice(&(params.heads_per_layer as u16).to_le_bytes());
    out.extend_from_slice(&(params.group_sizes[0] as u16).to_le_bytes());
    out.extend_from_slice(&(params.group_sizes[1] as u16).to_le_bytes());
    out.extend_from_slice(&(params.shift as u16).to_le_bytes());
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
    let h = parse_wide_header(bytes)?;
    let words_per_tok = h.n_heads as usize * h.wph;
    if bytes.len() < h.header_len + h.n_tokens * words_per_tok * 8 {
        return None;
    }
    let mut sigs = Vec::with_capacity(h.n_tokens);
    let mut off = h.header_len;
    for _ in 0..h.n_tokens {
        let mut words = Vec::with_capacity(words_per_tok);
        for _ in 0..words_per_tok {
            words.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
            off += 8;
        }
        sigs.push(WideQSig {
            n_heads: h.n_heads,
            words,
        });
    }
    Some(sigs)
}

/// Decode **only if** the record's fold matches `expected` — the scoring-path
/// read.
///
/// Returns `None` and logs at WARN on a mismatch, rather than handing back
/// signatures that would score against a differently-folded gallery. Mixing
/// folds does not degrade gracefully: the agreement count between two bit
/// vectors folded under different group boundaries is a number with no meaning,
/// and every consumer downstream treats it as a similarity.
///
/// Group sizes stamped as `0` mean "unstated" (a signature whose fold was
/// inferred from its shape) and compare equal to anything — the shape-derived
/// fields still have to agree.
/// **Checks the group sizes and nothing else, deliberately.** Everything else in
/// a fold is recoverable from the signature's own shape — `heads_per_layer` is
/// `n_heads / 3`, `head_dim` is `wph × 64`, `shift` follows from `head_dim` —
/// and the scorer already derives those per signature, so a shape difference is
/// handled correctly rather than being a mismatch to refuse.
///
/// The group sizes are the one parameter the shape cannot reveal: they depend on
/// the capture-layer count, which the signature does not carry. Two galleries
/// with identical shapes and different group boundaries are exactly the case
/// that scores into a confident number over incomparable bits, and exactly the
/// case nothing else can catch.
///
/// A `0` in either side's first group means "unstated" — a record whose fold was
/// inferred from its shape rather than stamped — and compares equal to anything.
pub fn decode_wide_sigs_checked(bytes: &[u8], expected: FoldParams) -> Option<Vec<WideQSig>> {
    let h = parse_wide_header(bytes)?;
    let got = h.params;
    let both_stated = got.group_sizes[0] != 0 && expected.group_sizes[0] != 0;
    if both_stated && got.group_sizes != expected.group_sizes {
        tracing::warn!(
            "PROVENANCE FOLD MISMATCH: record folded over layer groups {:?} but \
             this build folds {:?} — refusing to score rather than return a \
             confident number over incomparable bits",
            got.group_sizes,
            expected.group_sizes,
        );
        return None;
    }
    decode_wide_sigs(bytes)
}

#[cfg(test)]
mod tests {
    // Word offsets are spelled `(group * HEADS + head) * WORDS_PER_HEAD` with the
    // head written out even when it is 0, so the fold layout stays readable as
    // an address rather than a constant.
    #![allow(clippy::identity_op)]

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

    /// **The derivations are identities on Qwen3-30B.** This is what permits
    /// changing all four parameters at once without re-measuring the outgoing
    /// model: the derived fold IS the locked fold at that geometry.
    #[test]
    fn the_derived_fold_is_the_locked_fold_on_qwen3_30b() {
        let derived = FoldParams::derive(4, 48, 128);
        assert_eq!(derived, FoldParams::locked());
        assert_eq!(derived.group_sizes, [46, 1, 1], "[n-2,1,1] at n=48");
        assert_eq!(derived.shift, 32, "head_dim/4 at 128");
        assert_eq!(derived.folded_heads(), 12, "3 groups x 4 kv heads");
    }

    /// The hybrid's geometry: 10 attention layers, 2 KV heads, head_dim 256.
    ///
    /// The bit budget does not shrink — 3 x 2 x 256 = 1536, the same as
    /// 3 x 4 x 128. Fewer heads, twice as wide.
    #[test]
    fn the_derived_fold_matches_the_hybrid_geometry() {
        let p = FoldParams::derive(2, 10, 256);
        assert_eq!(p.group_sizes, [8, 1, 1], "[n-2,1,1] at n=10");
        assert_eq!(p.shift, 64, "head_dim/4 at 256");
        assert_eq!(p.folded_heads(), 6, "3 groups x 2 kv heads");
        assert_eq!(
            p.folded_heads() * p.head_dim,
            FoldParams::locked().folded_heads() * 128,
            "the signature carries the same 1536 bits either way"
        );
    }

    /// **`fold_fits` answers from the shape alone, and agrees with the fused
    /// form.**
    ///
    /// The capture loops call it once per turn instead of
    /// `fold_provenance_checked` once per token, which is only sound while the
    /// two give the same verdict — so this pins that, and pins that the verdict
    /// really is a function of `(n_heads, words_per_head, params)` and not of
    /// the bits.
    #[test]
    fn fold_fits_agrees_with_the_checked_fold_and_ignores_the_bits() {
        let shapes = [(20usize, 4usize), (192, 2), (0, 2), (20, 0), (6, 4)];
        let folds = [
            FoldParams::locked(),
            FoldParams::derive(2, 10, 256),
            FoldParams::derive(4, 48, 128),
        ];
        for (n_heads, wph) in shapes {
            for f in folds {
                // Two signatures of the same shape, entirely different bits.
                let zeros = WideQSig {
                    n_heads: n_heads as u16,
                    words: vec![0u64; n_heads * wph],
                };
                let ones = WideQSig {
                    n_heads: n_heads as u16,
                    words: vec![u64::MAX; n_heads * wph],
                };
                let fits = fold_fits(n_heads, wph, f);
                assert_eq!(
                    fits,
                    fold_provenance_checked(&zeros, f).is_some(),
                    "shape ({n_heads}, {wph}) under {f:?}: fold_fits disagrees with \
                     the fused check"
                );
                assert_eq!(
                    fold_provenance_checked(&zeros, f).is_some(),
                    fold_provenance_checked(&ones, f).is_some(),
                    "the verdict depended on the bits, so hoisting it out of a \
                     per-token loop would be unsound"
                );
            }
        }
    }

    /// **A fold that cannot fill every group must refuse.**
    ///
    /// `[46,1,1]` over a 10-layer stack puts all ten layers in group 0 and
    /// leaves groups 1 and 2 all-zero — two thirds of the signature empty, and
    /// the identity-bearing top layers gone. An all-zero group is not a weak
    /// signal; it is a scorer input that agrees with everything.
    #[test]
    fn a_fold_that_cannot_fill_every_group_is_refused() {
        // Per-word-distinct data. All-equal words would make a whole-word
        // rotation the identity, and the layers would cancel for a reason that
        // has nothing to do with what is under test.
        let ten_layers = WideQSig {
            n_heads: (10 * 2) as u16,
            words: (0..10 * 2 * 4)
                .map(|i| 0xDEAD_BEEF_0000_0001u64.wrapping_mul(i as u64 + 1))
                .collect(),
        };
        assert!(
            fold_provenance_checked(&ten_layers, FoldParams::locked()).is_none(),
            "the locked 48-layer fold must refuse a 10-layer stack rather than \
             emit two empty groups"
        );
        let derived = FoldParams::derive(2, 10, 256);
        let folded = fold_provenance_checked(&ten_layers, derived)
            .expect("the derived fold fills every group");
        assert_eq!(folded.n_heads, 6);

        // Every group carries something.
        let wph = folded.words_per_head();
        for g in 0..3 {
            let base = g * derived.heads_per_layer * wph;
            let any = folded.words[base..base + derived.heads_per_layer * wph]
                .iter()
                .any(|&w| w != 0);
            assert!(any, "group {g} came back all zero");
        }
    }

    /// **The stagger works at `wph == 4`** — the assertion a test written
    /// against `rotate_head` alone cannot make, because the caller used to
    /// throw the rotated result away on that path.
    ///
    /// Two *identical* layers in one group cancel to zero under XOR without a
    /// stagger. With one, they must not.
    #[test]
    fn the_fold_staggers_at_head_dim_256() {
        // 2 layers x 2 kv heads, head_dim 256 (4 words/head), both layers same.
        //
        // The words must DIFFER from each other. `head_dim / 4` is 64 bits at
        // head_dim 256 — exactly one u64 — so the stagger is a whole-word
        // rotation there, and a head whose words are all equal is invariant
        // under it. That is a real property of the derived shift at this width
        // (at head_dim 128 the shift is 32, a half-word, which mixes within
        // words), and it is on the list to re-derive by measurement; it is not
        // what this test is about.
        let layer: Vec<u64> = (0..2 * 4)
            .map(|i| 0x0F0F_0F0F_0F0F_0F0Fu64.rotate_left(i as u32 * 7) ^ (i as u64))
            .collect();
        let mut words = layer.clone();
        words.extend_from_slice(&layer);
        let raw = WideQSig { n_heads: 4, words };
        // One group taking both layers, so they XOR against each other.
        let params = FoldParams {
            heads_per_layer: 2,
            group_sizes: [2, 1, 1],
            shift: 64,
            head_dim: 256,
        };
        let folded = fold_provenance_with(&raw, params);
        let head0 = &folded.words[0..4];
        assert!(
            head0.iter().any(|&w| w != 0),
            "identical layers cancelled dim-aligned — the decorrelating stagger \
             is absent at head_dim 256, so the fold carries less than it reports"
        );

        // With shift 0 they must cancel, which is what makes the above a real
        // assertion about the stagger rather than about the data.
        let unstaggered = fold_provenance_with(&raw, FoldParams { shift: 0, ..params });
        assert!(
            unstaggered.words[0..4].iter().all(|&w| w == 0),
            "without a stagger identical layers must cancel — if they do not, \
             the test above proves nothing"
        );
    }

    /// A word-wise rotate at `wph == 2` is byte-identical to the `u128` form it
    /// replaced, which is why the locked fold's output does not move.
    #[test]
    fn the_word_wise_rotate_matches_the_u128_form_at_128_bits() {
        for r in [0usize, 1, 7, 32, 63, 64, 65, 127] {
            let src = [0x0123_4567_89AB_CDEFu64, 0xFEDC_BA98_7654_3210];
            let mut got = [0u64; 2];
            rotate_head_into(&mut got, &src, r);
            let v = (src[0] as u128) | ((src[1] as u128) << 64);
            let want = v.rotate_left((r % 128) as u32);
            assert_eq!(
                got,
                [want as u64, (want >> 64) as u64],
                "word-wise rotate diverged from the u128 rotate at r={r}"
            );
        }
    }

    /// The rotate is a rotation, not a shift: no bit is lost at any width.
    #[test]
    fn the_rotate_preserves_popcount_at_every_width() {
        for wph in [1usize, 2, 4, 8] {
            let src: Vec<u64> = (0..wph).map(|i| 0x1234_5678u64 << (i % 8)).collect();
            let before: u32 = src.iter().map(|w| w.count_ones()).sum();
            for r in [1usize, 13, 64, 100, wph * 64 - 1] {
                let mut dst = vec![0u64; wph];
                rotate_head_into(&mut dst, &src, r);
                let after: u32 = dst.iter().map(|w| w.count_ones()).sum();
                assert_eq!(after, before, "wph={wph} r={r} lost bits");
            }
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
                .map(|i| {
                    if (i + seed).is_multiple_of(3) {
                        1.0
                    } else {
                        -1.0
                    }
                })
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
        let params = FoldParams {
            heads_per_layer: 4,
            group_sizes: [46, 1, 1],
            shift: 32,
            head_dim: 64,
        };
        let bytes = encode_wide_sigs_with(&sigs, params);
        assert_eq!(
            &bytes[..WIDE_HEADER],
            &[
                b'W', b'Q', b'S', b'5', // magic
                2, 0, 0, 0, // n_tokens = 2 (u32 LE)
                1, 0, // n_heads = 1 (u16 LE)
                1, 0, // words_per_head = 1 (u16 LE)
                4, 0, // fold: heads_per_layer (u16 LE)
                46, 0, // fold: group_sizes[0]
                1, 0, // fold: group_sizes[1] (the third is always 1)
                32, 0, // fold: shift
            ],
            "wide-sig v5 header layout"
        );
        assert_eq!(bytes.len(), WIDE_HEADER + 2 * 8);
        assert_eq!(
            &bytes[WIDE_HEADER..WIDE_HEADER + 8],
            &0x0102_0304_0506_0708u64.to_le_bytes()
        );
        // And it reads back as the fold that was stamped.
        assert_eq!(wide_sig_fold_params(&bytes), Some(params));
    }

    /// **A record folded differently is refused, not scored.**
    ///
    /// The agreement count between two bit vectors folded under different group
    /// boundaries is a number with no meaning, and every consumer downstream
    /// treats it as a similarity. So the decode says no.
    #[test]
    fn a_record_folded_under_another_geometry_is_refused() {
        let sigs = vec![WideQSig {
            n_heads: 12,
            words: vec![0xABCD; 24],
        }];
        let thirty_b = FoldParams::locked();
        let bytes = encode_wide_sigs_with(&sigs, thirty_b);

        // Same fold: decodes.
        assert!(decode_wide_sigs_checked(&bytes, thirty_b).is_some());

        // A different capture depth — same shapes, different group boundaries.
        let other = FoldParams {
            group_sizes: [8, 1, 1],
            ..thirty_b
        };
        assert!(
            decode_wide_sigs_checked(&bytes, other).is_none(),
            "a differently-folded record must be refused, not scored"
        );

        // A different head width is NOT refused here, and that is deliberate:
        // `heads_per_layer`, `head_dim` and `shift` are all recoverable from the
        // signature's own shape, and the scorer derives them per signature. A
        // shape difference is therefore handled correctly rather than being a
        // mismatch — checking it here would refuse comparisons that are fine and
        // would make every geometry in a test process contaminate the next.
        let wider = FoldParams {
            head_dim: 256,
            shift: 64,
            ..thirty_b
        };
        assert!(
            decode_wide_sigs_checked(&bytes, wider).is_some(),
            "shape-derivable parameters are the scorer's job, not the codec's"
        );

        // An unstated fold (a record encoded without one) compares equal to
        // anything — there is nothing to disagree with.
        let unstated = encode_wide_sigs(&sigs);
        assert!(decode_wide_sigs_checked(&unstated, other).is_some());

        // The unchecked decode still works — the refusal is a scoring-path
        // policy, not a claim that the bytes are unreadable.
        assert!(decode_wide_sigs(&bytes).is_some());
    }

    /// A v3/v4 record reads as the locked Qwen3-30B fold, because that is the
    /// only fold that ever wrote one. Not a compatibility path — a fact.
    #[test]
    fn a_legacy_record_reports_the_locked_fold() {
        let mut v4 = Vec::new();
        v4.extend_from_slice(b"WQS4");
        v4.extend_from_slice(&1u32.to_le_bytes());
        v4.extend_from_slice(&12u16.to_le_bytes());
        v4.extend_from_slice(&2u16.to_le_bytes());
        v4.extend_from_slice(&[0u8; 12 * 2 * 8]);
        assert_eq!(wide_sig_fold_params(&v4), Some(FoldParams::locked()));
        assert!(decode_wide_sigs_checked(&v4, FoldParams::locked()).is_some());
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
        // v5 is what the encoder writes today.
        let v5 = encode_wide_sigs(&sigs);
        assert_eq!(&v5[..4], b"WQS5");

        // Hand-build the equivalent v3 and v4 records (the retired layouts).
        let mut v3 = Vec::new();
        v3.extend_from_slice(b"WQS3");
        v3.extend_from_slice(&(sigs.len() as u16).to_le_bytes());
        v3.extend_from_slice(&1u16.to_le_bytes()); // n_heads
        v3.extend_from_slice(&1u16.to_le_bytes()); // words_per_head
        for s in &sigs {
            v3.extend_from_slice(&s.words[0].to_le_bytes());
        }

        let mut v4 = Vec::new();
        v4.extend_from_slice(b"WQS4");
        v4.extend_from_slice(&(sigs.len() as u32).to_le_bytes());
        v4.extend_from_slice(&1u16.to_le_bytes());
        v4.extend_from_slice(&1u16.to_le_bytes());
        for s in &sigs {
            v4.extend_from_slice(&s.words[0].to_le_bytes());
        }

        // Three layouts, one history. The substrate holds records from every
        // version that ever ran against it, and they coexist.
        assert_eq!(decode_wide_sigs(&v3), decode_wide_sigs(&v4));
        assert_eq!(decode_wide_sigs(&v4), decode_wide_sigs(&v5));
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
