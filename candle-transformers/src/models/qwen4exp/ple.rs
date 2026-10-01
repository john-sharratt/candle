//! PLE — the per-layer n-gram hash embedding block, reference implementation.
//!
//! Semantics from `qwen4exp.cpp` `build_ple` + `llm_graph_input_ple::set_input`
//! (`docs/qwen38_flash_next.md` §12.4). Three parts:
//!
//! 1. **The hash** (host-side, as in llama.cpp — ggml has no u64 xor, and this
//!    reference keeps the same split so the two implementations share it):
//!    `mixed_n = (t·m₀) ⊕ (t₋₁·m₁) ⊕ …` per n-gram order, an EOS in the window
//!    resetting everything at or before it; `row = mixed mod vocab[h] + off[h]`
//!    with per-head vocabularies.
//! 2. **The keyed injection**: the gathered `[T, hidden]` embedding is
//!    projected to a per-stream key and a value; the key is dotted against the
//!    normed wide residual per stream, and a signed-sqrt sigmoid of that dot
//!    gates the value into all streams.
//! 3. **The dilated conv**: a depthwise causal conv over time (kernel 4,
//!    dilation `ngram_size` 3) of the normed gated value, silu'd and added
//!    alongside — carrying `(kernel−1)·dilation` = 9 tokens of history per
//!    sequence, which is the third per-session recurrent state.

use candle::{Result, Tensor};

use super::config::PleConfig;
use super::hyper::hc_grouped_norm;

/// One sequence's carried PLE state: the conv history tail and the hash
/// window's preceding token ids.
#[derive(Debug, Clone)]
pub struct PleState {
    /// `[history, hc_dim]` — the last `(conv_kernel−1)·ngram_size` normed
    /// gated-value rows, zeros at sequence start (causal padding).
    pub conv_hist: Tensor,
    /// The last `ngram_size − 1` token ids seen, oldest first. Shorter at
    /// sequence start; a missing predecessor hashes as EOS.
    pub prev: Vec<u32>,
}

impl PleState {
    pub fn zeros(cfg: &PleConfig, hc_dim: usize, dev: &candle::Device) -> Result<Self> {
        Ok(Self {
            conv_hist: Tensor::zeros((cfg.conv_history(), hc_dim), candle::DType::F32, dev)?,
            prev: Vec::new(),
        })
    }

    /// An independent copy (the state is replaced, not written through, so a
    /// plain clone of the handles suffices — mirrored on [`Self::snapshot`]
    /// for symmetry with the other carried states).
    pub fn snapshot(&self) -> Self {
        Self {
            conv_hist: self.conv_hist.clone(),
            prev: self.prev.clone(),
        }
    }

    /// Encode for the turn snapshot's auxiliary blob.
    ///
    /// Self-describing in its dimensions, and versioned, because this is the
    /// only carried class with no schedule hash standing behind it: the GDN
    /// store validates its geometry against a hash the model computes, and
    /// there is no equivalent for a `[history, hc_dim]` tail. The dimensions
    /// are written so [`Self::decode`] can refuse a blob from a differently
    /// shaped checkpoint instead of reshaping bytes into a plausible lie.
    ///
    /// Kilobytes: `conv_history()` is `(conv_kernel − 1) × ngram_size` = 9 rows
    /// on this checkpoint, so the whole thing is three orders of magnitude
    /// under the GDN snapshot it rides beside.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let (rows, cols) = self.conv_hist.dims2()?;
        let vals = self.conv_hist.flatten_all()?.to_vec1::<f32>()?;
        let mut out = Vec::with_capacity(20 + vals.len() * 4 + self.prev.len() * 4);
        out.extend_from_slice(&PLE_AUX_VERSION.to_le_bytes());
        out.extend_from_slice(&(rows as u32).to_le_bytes());
        out.extend_from_slice(&(cols as u32).to_le_bytes());
        for v in &vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.prev.len() as u32).to_le_bytes());
        for t in &self.prev {
            out.extend_from_slice(&t.to_le_bytes());
        }
        Ok(out)
    }

    /// Read back what [`Self::encode`] wrote, refusing anything that does not
    /// match `(rows, cols)` — a foreign or stale layout recomputes rather than
    /// scattering bytes of the wrong shape.
    pub fn decode(blob: &[u8], rows: usize, cols: usize, dev: &candle::Device) -> Result<Self> {
        let u32_at = |o: usize| -> Result<u32> {
            let b = blob
                .get(o..o + 4)
                .ok_or_else(|| candle::Error::Msg("ple aux: blob truncated".into()))?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let version = u32_at(0)?;
        if version != PLE_AUX_VERSION {
            candle::bail!(
                "ple aux: blob version {version} unknown (this build reads {PLE_AUX_VERSION})"
            );
        }
        let (r, c) = (u32_at(4)? as usize, u32_at(8)? as usize);
        if r != rows || c != cols {
            candle::bail!(
                "ple aux: blob is [{r}, {c}] but this checkpoint's PLE state is \
                 [{rows}, {cols}] — recompute instead of reshaping"
            );
        }
        let n = r * c;
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            vals.push(f32::from_bits(u32_at(12 + i * 4)?));
        }
        let after = 12 + n * 4;
        let n_prev = u32_at(after)? as usize;
        let mut prev = Vec::with_capacity(n_prev);
        for i in 0..n_prev {
            prev.push(u32_at(after + 4 + i * 4)?);
        }
        Ok(Self {
            conv_hist: Tensor::from_vec(vals, (r, c), dev)?,
            prev,
        })
    }
}

/// Wire version of the PLE auxiliary blob; bump on layout change and keep
/// decode for every version ever written.
const PLE_AUX_VERSION: u32 = 1;

/// The PLE layer's weights.
#[derive(Debug, Clone)]
pub struct PleWeights {
    /// `[hc_dim, hidden]`.
    pub key: Tensor,
    /// `[hidden, hidden]`.
    pub value: Tensor,
    /// `[hc_dim]` each.
    pub norm_key: Tensor,
    pub norm_query: Tensor,
    pub norm_conv: Tensor,
    /// `[hc_dim, conv_kernel]` — one weight per channel per tap.
    pub conv: Tensor,
}

/// The 16 table rows a token gathers, given its hash window
/// `ctx = [t, t₋₁, t₋₂, …]` (newest first, exactly `ngram_size` long).
fn head_rows(cfg: &PleConfig, ctx: &[u64]) -> Vec<u32> {
    let n_heads = cfg.n_heads();
    let mut rows = vec![0u32; n_heads];
    for n in 2..=cfg.ngram_size {
        let mut mixed = ctx[0].wrapping_mul(cfg.multipliers[0]);
        for (j, &t) in ctx.iter().enumerate().take(n).skip(1) {
            mixed ^= t.wrapping_mul(cfg.multipliers[j]);
        }
        let base = (n - 2) * cfg.heads_per_ngram;
        for g in 0..cfg.heads_per_ngram {
            let h = base + g;
            rows[h] = (mixed % cfg.head_vocab_sizes[h] + cfg.head_offsets[h]) as u32;
        }
    }
    rows
}

/// Row ids for a token segment, advancing the hash window in `prev`.
///
/// The EOS rules are llama.cpp's exactly: a missing predecessor (before the
/// sequence start) reads as EOS, an EOS in the window resets everything at or
/// before it, and the token's *own* EOS does not cut its own context.
pub fn ple_row_ids(cfg: &PleConfig, tokens: &[u32], prev: &mut Vec<u32>) -> Vec<Vec<u32>> {
    let n_prev = cfg.ngram_size - 1;
    let eos = cfg.eos_token_id as u64;
    let mut out = Vec::with_capacity(tokens.len());
    for (i, &tok) in tokens.iter().enumerate() {
        let mut ctx = vec![0u64; cfg.ngram_size];
        ctx[0] = tok as u64;
        let mut cut = false;
        for s in 1..cfg.ngram_size {
            // Predecessor `s` positions back: this segment's own earlier
            // tokens first, then the carried window, then "missing".
            let t: Option<u32> = if i >= s {
                Some(tokens[i - s])
            } else {
                let back = s - i; // 1-based into `prev`, newest last
                prev.len().checked_sub(back).map(|ix| prev[ix])
            };
            let t = if cut { None } else { t };
            cut = cut || t.is_none() || t == Some(cfg.eos_token_id);
            ctx[s] = if cut { eos } else { t.unwrap() as u64 };
        }
        out.push(head_rows(cfg, &ctx));
    }
    // Advance the carried window with this segment's tail.
    for &tok in tokens {
        prev.push(tok);
    }
    let keep = prev.len().saturating_sub(n_prev);
    prev.drain(..keep);
    out
}

/// The PLE block over ONE sequence's rows.
///
/// `res_hc` is that sequence's `[T, hc, n_embd]` wide residual, `emb` its
/// gathered `[T, hidden]` table rows. Returns the updated wide residual;
/// `state.conv_hist` is advanced in place.
/// `capture` receives the rows this segment appended to the conv history, when
/// the caller is verifying a speculative block.
///
/// The history is a sliding window, so the state after the block's first `m`
/// rows is `(entering_hist ++ these)[m .. m + hist]` — a pure slice, no
/// arithmetic replayed. Handing the rows back here is what makes a partial
/// accept rewindable at all: they are derived from activations the wave arena
/// reclaims, so nothing outside this call could reconstruct them without a
/// second forward.
pub fn ple_apply(
    res_hc: &Tensor,
    emb: &Tensor,
    w: &PleWeights,
    cfg: &PleConfig,
    state: &mut PleState,
    eps: f64,
    capture: Option<&mut Tensor>,
) -> Result<Tensor> {
    let (t, hc, n_embd) = res_hc.dims3()?;
    let hc_dim = hc * n_embd;

    let key = emb.matmul(&w.key.t()?)?.reshape((t, hc, n_embd))?;
    let value = emb.matmul(&w.value.t()?)?; // [T, n_embd]

    let key = hc_grouped_norm(&key, &w.norm_key, eps, None)?;
    let query = hc_grouped_norm(res_hc, &w.norm_query, eps, None)?;

    // Per-stream dot, then a signed square root before the sigmoid.
    let s = (key.mul(&query)?.sum_keepdim(candle::D::Minus1)? * (1.0 / (n_embd as f64).sqrt()))?;
    let mag = s.abs()?.clamp(1e-6, 1e30)?.sqrt()?;
    // sgn(s) ∈ {−1, 0, 1}, matching ggml_sgn (an exact zero gates at 0.5).
    let sgn = s
        .gt(0f64)?
        .to_dtype(candle::DType::F32)?
        .sub(&s.lt(0f64)?.to_dtype(candle::DType::F32)?)?;
    let gate = candle_nn::ops::sigmoid(&sgn.mul(&mag)?)?; // [T, hc, 1]

    let gated = value
        .reshape((t, 1, n_embd))?
        .broadcast_mul(&gate)?
        .contiguous()?; // [T, hc, n_embd]

    // Depthwise causal conv over time, dilated by the n-gram size, over the
    // grouped-normed gated value. History rows prepend so a chunked forward
    // matches a one-shot one.
    let normalized = hc_grouped_norm(&gated, &w.norm_conv, eps, None)?.reshape((t, hc_dim))?;
    if let Some(out) = capture {
        // **`to_owned_tensor`, not `contiguous`.** These outlive the wave whose
        // arena produced them — `spec.rs`'s rewind reads them after it has
        // closed — and `contiguous()` cannot leave the arena in either branch:
        // on an already-contiguous tensor (which this is, freshly produced) it
        // returns `self.clone()`, and when it copies it allocates with
        // `self.wave_ticket()`.
        *out = normalized.to_owned_tensor()?;
    }
    let hist = cfg.conv_history();
    let padded = Tensor::cat(&[&state.conv_hist, &normalized], 0)?; // [hist+T, hc_dim]

    let kern = cfg.conv_kernel;
    let dil = cfg.ngram_size;
    let mut conv_out: Option<Tensor> = None;
    for k in 0..kern {
        // Tap k reads (kern−1−k)·dilation positions back.
        let start = hist - (kern - 1 - k) * dil;
        let shifted = padded.narrow(0, start, t)?;
        let wk = w.conv.narrow(1, k, 1)?.reshape(hc_dim)?; // one weight per channel
        let term = shifted.broadcast_mul(&wk)?;
        conv_out = Some(match conv_out {
            Some(acc) => acc.add(&term)?,
            None => term,
        });
    }
    let conv_out = conv_out.expect("conv_kernel >= 2");
    let conv_out = conv_out
        .broadcast_mul(&candle_nn::ops::sigmoid(&conv_out)?)? // silu
        .reshape((t, hc, n_embd))?;

    // Keep the last `hist` rows for the next segment.
    //
    // `to_owned_tensor`, because this is carried state and the narrow is of a
    // `[hist + t, hc_dim]` buffer. `contiguous()` would alias `padded` — the
    // narrow of a contiguous tensor is contiguous, so it returns a
    // storage-sharing clone — which pins the whole concatenation alive to keep
    // `hist` rows of it, at prefill width tens of megabytes per sequence held
    // until the next call. `Tensor::cat` also inherits `arg0`'s wave ticket, so
    // the alias is only *not* arena memory because `arg0` here happens to be
    // the previously-owned history; owning the result stops that being load-
    // bearing.
    let total = hist + t;
    state.conv_hist = padded.narrow(0, total - hist, hist)?.to_owned_tensor()?;

    res_hc.add(&gated)?.add(&conv_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn cfg() -> PleConfig {
        PleConfig {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 2,
            conv_kernel: 4,
            eos_token_id: 99,
            multipliers: vec![
                0x9E37_79B9_7F4A_7C15,
                0xC2B2_AE3D_27D4_EB4F,
                0x1656_67B1_9E37_79F9,
            ],
            head_offsets: vec![0, 100, 200, 300],
            head_vocab_sizes: vec![100, 100, 100, 100],
            head_dim: 4,
        }
    }

    #[test]
    fn bigram_heads_ignore_the_third_token_and_trigram_heads_do_not() {
        let c = cfg();
        let mut prev_a = vec![5u32, 7];
        let mut prev_b = vec![6u32, 7];
        let rows_a = ple_row_ids(&c, &[9], &mut prev_a);
        let rows_b = ple_row_ids(&c, &[9], &mut prev_b);
        // Heads 0..2 hash (t, t-1) = (9, 7) in both.
        assert_eq!(rows_a[0][0], rows_b[0][0]);
        assert_eq!(rows_a[0][1], rows_b[0][1]);
        // Heads 2..4 hash (t, t-1, t-2), which differs.
        assert_ne!(rows_a[0][2..4], rows_b[0][2..4]);
    }

    #[test]
    fn rows_land_in_their_heads_regions() {
        let c = cfg();
        let mut prev = Vec::new();
        let rows = ple_row_ids(&c, &[1, 2, 3], &mut prev);
        for token_rows in &rows {
            for (h, &r) in token_rows.iter().enumerate() {
                let off = c.head_offsets[h] as u32;
                let vocab = c.head_vocab_sizes[h] as u32;
                assert!(
                    r >= off && r < off + vocab,
                    "head {h} row {r} out of region"
                );
            }
        }
        // The carried window now holds the last two tokens.
        assert_eq!(prev, vec![2, 3]);
    }

    #[test]
    fn an_eos_in_the_window_resets_everything_at_or_before_it() {
        let c = cfg();
        // Window (t-1 = EOS): the trigram context must hash as (t, EOS, EOS) —
        // identical whatever stood before the EOS.
        let mut prev_a = vec![41u32, 99];
        let mut prev_b = vec![77u32, 99];
        let ra = ple_row_ids(&c, &[5], &mut prev_a);
        let rb = ple_row_ids(&c, &[5], &mut prev_b);
        assert_eq!(ra, rb, "tokens before an EOS leaked through the reset");
        // And a missing predecessor (sequence start) hashes exactly as EOS.
        let mut empty = Vec::new();
        let r_start = ple_row_ids(&c, &[5], &mut empty);
        let mut eos_prev = vec![99u32, 99];
        let r_eos = ple_row_ids(&c, &[5], &mut eos_prev);
        assert_eq!(r_start, r_eos, "sequence start must hash as an EOS window");
    }

    #[test]
    fn the_tokens_own_eos_does_not_cut_its_own_context() {
        let c = cfg();
        let mut prev_a = vec![3u32, 4];
        let mut prev_b = vec![3u32, 5];
        // The current token IS the EOS; its predecessors still hash.
        let ra = ple_row_ids(&c, &[99], &mut prev_a);
        let rb = ple_row_ids(&c, &[99], &mut prev_b);
        assert_ne!(ra, rb, "an EOS token erased its own preceding context");
    }

    #[test]
    fn segmented_hashing_matches_one_shot() {
        let c = cfg();
        let toks = [1u32, 99, 3, 4, 5, 6];
        let mut prev_full = Vec::new();
        let full = ple_row_ids(&c, &toks, &mut prev_full);
        let mut prev_seg = Vec::new();
        let mut seg = ple_row_ids(&c, &toks[..2], &mut prev_seg);
        seg.extend(ple_row_ids(&c, &toks[2..], &mut prev_seg));
        assert_eq!(full, seg);
        assert_eq!(prev_full, prev_seg);
    }

    #[test]
    fn segmented_conv_matches_one_shot() {
        // The carried conv history is what makes a chunked prefill match a
        // single-shot one — the llama.cpp limitation this design avoids.
        let dev = Device::Cpu;
        let c = cfg();
        let (hc, n_embd) = (2usize, c.n_heads() * c.head_dim); // hidden = 16... n_heads=4, head_dim=4 → 16
        let hc_dim = hc * n_embd;
        let lcg = |shape: &[usize], seed: u64| {
            let n: usize = shape.iter().product();
            let mut s = seed;
            let vals: Vec<f32> = (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect();
            Tensor::from_vec(vals, shape, &dev).unwrap()
        };
        let w = PleWeights {
            key: lcg(&[hc_dim, n_embd], 1).affine(0.3, 0.).unwrap(),
            value: lcg(&[n_embd, n_embd], 2).affine(0.3, 0.).unwrap(),
            norm_key: lcg(&[hc_dim], 3).affine(0.1, 1.0).unwrap(),
            norm_query: lcg(&[hc_dim], 4).affine(0.1, 1.0).unwrap(),
            norm_conv: lcg(&[hc_dim], 5).affine(0.1, 1.0).unwrap(),
            conv: lcg(&[hc_dim, c.conv_kernel], 6).affine(0.3, 0.).unwrap(),
        };
        let t = 7usize;
        let res = lcg(&[t, hc, n_embd], 7);
        let emb = lcg(&[t, n_embd], 8);

        let mut s_full = PleState::zeros(&c, hc_dim, &dev).unwrap();
        let full = ple_apply(&res, &emb, &w, &c, &mut s_full, 1e-6, None).unwrap();

        let mut s_seg = PleState::zeros(&c, hc_dim, &dev).unwrap();
        let a = 3usize;
        let p1 = ple_apply(
            &res.narrow(0, 0, a).unwrap(),
            &emb.narrow(0, 0, a).unwrap(),
            &w,
            &c,
            &mut s_seg,
            1e-6,
            None,
        )
        .unwrap();
        let p2 = ple_apply(
            &res.narrow(0, a, t - a).unwrap(),
            &emb.narrow(0, a, t - a).unwrap(),
            &w,
            &c,
            &mut s_seg,
            1e-6,
            None,
        )
        .unwrap();
        let seg = Tensor::cat(&[p1, p2], 0).unwrap();
        let d = full
            .sub(&seg)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-5, "segmented PLE conv diverged from one-shot: {d}");
    }
}
