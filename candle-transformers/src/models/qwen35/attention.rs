//! The full-attention layer: gated GQA attention at `attn_head_dim` (256 in
//! the published family), one per `full_attention_interval` layers.
//!
//! Reference semantics from `qwen35.cpp` `build_layer_attn`:
//!
//! - `wq` projects to `2 × head_dim` per head, **interleaved** `[q | gate]`:
//!   head `h`'s query occupies dims `[2h·d, 2h·d + d)` and its gate
//!   `[2h·d + d, 2(h+1)·d)`. The gate is *not* normed or roped — it is held
//!   aside and applied as `sigmoid(gate) ⊙ attn_out` after attention, before
//!   the output projection.
//! - Q and K get per-head weighted RMSNorm (`attn_q_norm`/`attn_k_norm`,
//!   `[head_dim]`) between the projection and RoPE — the Qwen3 lineage rule.
//! - RoPE is `rope_multi` over `rope.dimension_count` dims (64 of the 256 —
//!   a 0.25 partial rotary factor) with `rope.dimension_sections`; for
//!   text-only position streams every section sees the same position, which
//!   reduces exactly to classic non-interleaved (NeoX half-split) RoPE over
//!   the rotary width — what this reference implements. (The sections matter
//!   only when a multimodal caller supplies distinct per-axis positions.)
//! - Attention is causal softmax GQA, scale `1/sqrt(head_dim)`.

use candle::{Device, Result, Tensor};

/// Weights of one gated-attention layer (reference path, F32).
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// `[2·head_dim·n_head, hidden]` — interleaved `[q | gate]` per head.
    pub wq: Tensor,
    /// `[head_dim·n_kv_heads, hidden]`.
    pub wk: Tensor,
    /// `[head_dim·n_kv_heads, hidden]`.
    pub wv: Tensor,
    /// `[hidden, head_dim·n_head]`.
    pub wo: Tensor,
    /// `[head_dim]` per-head RMS weights.
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

/// Carried KV for the reference forward: plain contiguous history per layer.
/// `k`/`v` are `[n_kv_heads, P, head_dim]`.
#[derive(Debug, Clone, Default)]
pub struct AttentionState {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl AttentionState {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(1).unwrap_or(0)).unwrap_or(0)
    }
}

/// Precomputed RoPE tables over the rotary width `rope_dim` (ggml `n_rot`).
///
/// NeoX half-split *within the rotary width*: pair `(i, i + rope_dim/2)`
/// rotates at `theta^(−2i/rope_dim)`, for `i < rope_dim/2`. Head dims at or
/// above `rope_dim` are copied through — this family sets a partial rotary
/// factor of 0.25, so 64 of 256 dims rotate.
#[derive(Debug, Clone)]
pub struct RopeTables {
    /// `[max_pos, rope_dim / 2]` each.
    cos: Tensor,
    sin: Tensor,
    /// The rotary width; `apply` leaves `[rope_dim, head_dim)` untouched.
    rope_dim: usize,
}

impl RopeTables {
    pub fn new(rope_dim: usize, theta: f32, max_pos: usize, dev: &Device) -> Result<Self> {
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
            candle::bail!("rope_dim {rope_dim} must be even and nonzero");
        }
        let half = rope_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / theta.powf(2.0 * i as f32 / rope_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let pos: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (max_pos, 1), dev)?;
        let angles = pos.matmul(&inv_freq)?;
        Ok(Self {
            cos: angles.cos()?,
            sin: angles.sin()?,
            rope_dim,
        })
    }

    /// The rotary width these tables cover.
    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    /// How many positions these tables span — what a caller growing them on
    /// demand compares against.
    pub fn max_pos(&self) -> usize {
        self.cos.dim(0).unwrap_or(0)
    }

    /// Apply to `x [T, n_heads, head_dim]` for absolute positions
    /// `offset..offset + T`. Dims `[rope_dim, head_dim)` pass through.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (t, _h, d) = x.dims3()?;
        if d < self.rope_dim {
            candle::bail!(
                "rope: head_dim {d} is narrower than the rotary width {}",
                self.rope_dim
            );
        }
        let half = self.rope_dim / 2;
        let cos = self.cos.narrow(0, offset, t)?.reshape((t, 1, half))?;
        let sin = self.sin.narrow(0, offset, t)?.reshape((t, 1, half))?;
        let x1 = x.narrow(2, 0, half)?;
        let x2 = x.narrow(2, half, half)?;
        let r1 = x1.broadcast_mul(&cos)?.sub(&x2.broadcast_mul(&sin)?)?;
        let r2 = x2.broadcast_mul(&cos)?.add(&x1.broadcast_mul(&sin)?)?;
        if d == self.rope_dim {
            return Tensor::cat(&[r1, r2], 2);
        }
        let tail = x.narrow(2, self.rope_dim, d - self.rope_dim)?;
        Tensor::cat(&[r1, r2, tail], 2)
    }
}

/// Per-head weighted RMSNorm over the last dim.
fn rms_norm_head(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(candle::D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    x.broadcast_div(&denom)?.broadcast_mul(weight)
}

/// One gated-attention layer over a `[T, hidden]` segment, attending to the
/// carried history. Returns `([T, hidden], new_state)`.
#[allow(clippy::too_many_arguments)]
pub fn attention_layer_forward(
    x: &Tensor,
    w: &AttentionWeights,
    state: AttentionState,
    rope: &RopeTables,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
) -> Result<(Tensor, AttentionState)> {
    let (t, _hidden) = x.dims2()?;
    let d = head_dim;
    let (gated, state) = gated_attention_core(
        &x.matmul(&w.wq.t()?)?,
        &x.matmul(&w.wk.t()?)?,
        &x.matmul(&w.wv.t()?)?,
        &w.q_norm,
        &w.k_norm,
        state,
        rope,
        n_head,
        n_kv_head,
        head_dim,
        rms_eps,
    )?;
    let out = gated.reshape((t, n_head * d))?.matmul(&w.wo.t()?)?;
    Ok((out, state))
}

/// The gated-attention algebra, from raw projections to the gated context —
/// everything between `wq`/`wk`/`wv` and `wo`.
///
/// Split out so the **MTP draft head** ([`super::mtp`]) runs the same
/// arithmetic without a second transcription of it. The head's projections are
/// quantized `QMatMul`s where the reference's are plain matmuls, and that is the
/// only difference between them; the norms, the interleaved `[q|gate]` split,
/// the RoPE order, the GQA expansion, the causal mask and the sigmoid gate are
/// one implementation.
///
/// `qg` is `[T, 2·n_head·head_dim]` (interleaved `[q|gate]` per head), `k` and
/// `v` are `[T, n_kv_head·head_dim]`. Returns the gated context
/// `[T, n_head, head_dim]` — the caller applies its own `wo`.
#[allow(clippy::too_many_arguments)]
pub fn gated_attention_core(
    qg: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    state: AttentionState,
    rope: &RopeTables,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
) -> Result<(Tensor, AttentionState)> {
    let d = head_dim;
    let t = qg.dim(0)?;
    let past = state.seq_len();

    // Interleaved [q | gate]: head `h`'s query is dims `[2h·d, 2h·d + d)` and
    // its gate the next `d`. The gate is neither normed nor roped.
    let q_full = qg.reshape((t, n_head, 2, d))?;
    let q = q_full.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [T, H, d]
    let gate = q_full.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;

    let k = k.reshape((t, n_kv_head, d))?;
    let v = v.reshape((t, n_kv_head, d))?;

    // Norm, then RoPE — the reference order.
    let q = rope.apply(&rms_norm_head(&q, q_norm_w, rms_eps)?, past)?;
    let k = rope.apply(&rms_norm_head(&k, k_norm_w, rms_eps)?, past)?;

    // Append to history: [n_kv, P + T, d].
    let k_hist = k.transpose(0, 1)?.contiguous()?;
    let v_hist = v.transpose(0, 1)?.contiguous()?;
    let (k_all, v_all) = match (&state.k, &state.v) {
        (Some(pk), Some(pv)) => (
            Tensor::cat(&[pk, &k_hist], 1)?,
            Tensor::cat(&[pv, &v_hist], 1)?,
        ),
        _ => (k_hist, v_hist),
    };
    let total = past + t;

    // GQA by GROUPING THE QUERIES, not by broadcasting the KV.
    //
    // Head `h` reads KV head `h / group`, so heads are consecutive within a
    // group and `[H, T, d]` reshapes to `[n_kv, group·T, d]` with every row
    // already beside the KV head it wants. That turns the broadcast into a
    // batched matmul against `k_all` as it stands.
    //
    // The alternative — `expand(...).contiguous()` to `[H, total, d]` — writes
    // the whole history out again per K and per V, `group` times over. On the
    // draft head at a conversational depth that was ~38 MB of allocate-and-copy
    // per sequence per step, to feed a single query row (hot-path invariant 2:
    // teach the consumer to read the layout that exists).
    let group = n_head / n_kv_head;
    let scale = 1f64 / (d as f64).sqrt();
    let q_h = q.transpose(0, 1)?.contiguous()?; // [H, T, d]
    let q_g = q_h.reshape((n_kv_head, group * t, d))?;
    // [n_kv, group·T, total] → [H, T, total]
    let scores = (q_g.matmul(&k_all.transpose(1, 2)?)? * scale)?.reshape((n_head, t, total))?;
    // A single query row masks nothing: it sits at `past`, the history is
    // `total = past + 1` long, so every key `j < total` satisfies `j <= past`
    // and the mask is all zeros. Building it means a `total`-long host vector
    // and an upload per call — on the draft head, which runs one row per
    // sequence per step against a history that grows with the conversation,
    // that was the single largest host cost in the decode loop. Skipping it is
    // exact, not approximate: the add it replaces is `+ 0.0`.
    let probs = if t == 1 {
        candle_nn::ops::softmax_last_dim(&scores)?
    } else {
        let mask_vals: Vec<f32> = (0..t)
            .flat_map(|i| {
                (0..total).map(move |j| {
                    if j <= past + i {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask_vals, (1, t, total), qg.device())?;
        candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?
    };
    // Same grouping on the way back out: `[H, T, total]` → `[n_kv, group·T,
    // total]` reads `v_all` in place, no broadcast copy.
    let ctx = probs
        .reshape((n_kv_head, group * t, total))?
        .matmul(&v_all)?
        .reshape((n_head, t, d))?; // [H, T, d]

    // Gate — the caller applies its own output projection.
    let ctx = ctx.transpose(0, 1)?.contiguous()?; // [T, H, d]
    let gated = ctx.mul(&candle_nn::ops::sigmoid(&gate)?)?;

    Ok((
        gated,
        AttentionState {
            k: Some(k_all),
            v: Some(v_all),
        },
    ))
}

/// Zero-history state helper for model assembly.
pub fn empty_attention_state() -> AttentionState {
    AttentionState::empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
        let d = a
            .sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d <= tol, "{what}: max abs diff {d} > {tol}");
    }

    fn tiny_weights(
        hidden: usize,
        n_head: usize,
        n_kv: usize,
        d: usize,
        dev: &Device,
    ) -> AttentionWeights {
        let scale = |t: Tensor| t.affine(0.15, 0.).unwrap();
        AttentionWeights {
            wq: scale(lcg_tensor(&[2 * d * n_head, hidden], 31, dev)),
            wk: scale(lcg_tensor(&[d * n_kv, hidden], 32, dev)),
            wv: scale(lcg_tensor(&[d * n_kv, hidden], 33, dev)),
            wo: scale(lcg_tensor(&[hidden, d * n_head], 34, dev)),
            q_norm: lcg_tensor(&[d], 35, dev).affine(0.3, 1.0).unwrap(),
            k_norm: lcg_tensor(&[d], 36, dev).affine(0.3, 1.0).unwrap(),
        }
    }

    #[test]
    fn prefill_then_decode_equals_one_shot() {
        // The KV-carry contract: prefill [0..a) then decode the rest token by
        // token must reproduce the one-shot forward exactly.
        let dev = dev();
        let (hidden, n_head, n_kv, d) = (6usize, 4usize, 2usize, 8usize);
        let w = tiny_weights(hidden, n_head, n_kv, d, &dev);
        let rope = RopeTables::new(d, 1e6, 64, &dev).unwrap();
        let t = 6usize;
        let x = lcg_tensor(&[t, hidden], 41, &dev);

        let (y_full, _) = attention_layer_forward(
            &x,
            &w,
            AttentionState::empty(),
            &rope,
            n_head,
            n_kv,
            d,
            1e-6,
        )
        .unwrap();

        let a = 3usize;
        let (y1, s) = attention_layer_forward(
            &x.narrow(0, 0, a).unwrap(),
            &w,
            AttentionState::empty(),
            &rope,
            n_head,
            n_kv,
            d,
            1e-6,
        )
        .unwrap();
        let mut ys = vec![y1];
        let mut state = s;
        for i in a..t {
            let (yi, s_next) = attention_layer_forward(
                &x.narrow(0, i, 1).unwrap(),
                &w,
                state,
                &rope,
                n_head,
                n_kv,
                d,
                1e-6,
            )
            .unwrap();
            ys.push(yi);
            state = s_next;
        }
        let y_seg = Tensor::cat(&ys, 0).unwrap();
        assert_close(&y_full, &y_seg, 2e-5, "prefill+decode vs one-shot");
    }

    #[test]
    fn attention_is_causal() {
        // Changing a later token must not change earlier outputs.
        let dev = dev();
        let (hidden, n_head, n_kv, d) = (6usize, 2usize, 1usize, 4usize);
        let w = tiny_weights(hidden, n_head, n_kv, d, &dev);
        let rope = RopeTables::new(d, 1e6, 16, &dev).unwrap();
        let x = lcg_tensor(&[5, hidden], 51, &dev);
        let (y, _) = attention_layer_forward(
            &x,
            &w,
            AttentionState::empty(),
            &rope,
            n_head,
            n_kv,
            d,
            1e-6,
        )
        .unwrap();
        // Perturb the last token only.
        let bump = Tensor::from_vec(vec![10f32; hidden], (1, hidden), &dev).unwrap();
        let x2 = Tensor::cat(
            &[
                x.narrow(0, 0, 4).unwrap(),
                x.narrow(0, 4, 1).unwrap().add(&bump).unwrap(),
            ],
            0,
        )
        .unwrap();
        let (y2, _) = attention_layer_forward(
            &x2,
            &w,
            AttentionState::empty(),
            &rope,
            n_head,
            n_kv,
            d,
            1e-6,
        )
        .unwrap();
        assert_close(
            &y.narrow(0, 0, 4).unwrap(),
            &y2.narrow(0, 0, 4).unwrap(),
            1e-6,
            "causality",
        );
    }

    /// Partial rotary (the published family's 0.25 factor): the leading
    /// `rope_dim` dims rotate, everything above passes through byte-for-byte
    /// and is untouched by position.
    #[test]
    fn partial_rotary_leaves_the_tail_dims_alone() {
        let dev = dev();
        let (d, rot) = (8usize, 4usize);
        let rope = RopeTables::new(rot, 1e6, 8, &dev).unwrap();
        assert_eq!(rope.rope_dim(), rot);
        let x = lcg_tensor(&[3, 2, d], 71, &dev);
        let y = rope.apply(&x, 5).unwrap();
        // Tail dims [rot, d) are copied verbatim.
        assert_close(
            &x.narrow(2, rot, d - rot).unwrap(),
            &y.narrow(2, rot, d - rot).unwrap(),
            0.0,
            "partial rotary tail passthrough",
        );
        // The rotary head did move.
        let head_delta = x
            .narrow(2, 0, rot)
            .unwrap()
            .sub(&y.narrow(2, 0, rot).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(head_delta > 1e-3, "rotary head unchanged: {head_delta}");
        // A narrower rotary width than the head is still a valid full-width
        // rotation of its own leading block.
        let full = RopeTables::new(rot, 1e6, 8, &dev).unwrap();
        let head_only = full
            .apply(&x.narrow(2, 0, rot).unwrap().contiguous().unwrap(), 5)
            .unwrap();
        assert_close(
            &head_only,
            &y.narrow(2, 0, rot).unwrap().contiguous().unwrap(),
            1e-6,
            "rotary head equals the full-width rotation of that block",
        );
    }

    #[test]
    fn rope_rotates_positions_zero_as_identity_and_preserves_norm() {
        let dev = dev();
        let d = 8usize;
        let rope = RopeTables::new(d, 1e6, 8, &dev).unwrap();
        let x = lcg_tensor(&[1, 2, d], 61, &dev);
        // Position 0: all angles are 0 ⇒ identity.
        let y0 = rope.apply(&x, 0).unwrap();
        assert_close(&x, &y0, 1e-7, "rope at position 0");
        // Any position: rotation preserves per-pair norms.
        let y3 = rope.apply(&x, 3).unwrap();
        let n_in = x.sqr().unwrap().sum(candle::D::Minus1).unwrap();
        let n_out = y3.sqr().unwrap().sum(candle::D::Minus1).unwrap();
        assert_close(&n_in, &n_out, 1e-5, "rope preserves norm");
    }
}
