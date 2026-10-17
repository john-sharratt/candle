//! Gated DeltaNet — the reference recurrence.
//!
//! This is the sequential, numerically-transparent implementation the rest of
//! the bring-up is validated against (the CUDA chunked kernel must match it
//! bit-for-tolerance; llama.cpp's `delta-net-base.cpp` chunked graph is the
//! external oracle). Per V head `h`, with state `S ∈ R^{d_v × d_k}`:
//!
//! ```text
//!   S ← exp(g_t[h]) · S                    (log-decay gate, g ≤ 0)
//!   S ← S + β_t[h] · (v_t − S k_t) ⊗ k_t   (delta rule: correct the value
//!                                            the state predicts for k_t)
//!   o_t = S q_t                             (read AFTER the update — the
//!                                            current token sees itself, the
//!                                            inclusive-diagonal convention of
//!                                            the chunked formulation)
//! ```
//!
//! Around the recurrence, the layer is (matching `qwen35.cpp`
//! `build_layer_attn_linear` exactly):
//!
//! ```text
//!   qkv = W_qkv·x            (fused [Q|K|V] projection, conv over channels)
//!   z   = W_gate·x           (output gate, applied at the end via SiLU)
//!   β   = sigmoid(W_beta·x)                       per V head
//!   g   = a ⊙ softplus(W_alpha·x + dt_bias)       per V head (a = −exp(A_log),
//!                                                  baked into the checkpoint)
//!   qkv = SiLU(causal_conv1d(qkv))                (kernel `conv_kernel`,
//!                                                  carried tail state)
//!   q,k = l2norm per head;  q,k broadcast n_k_heads → n_v_heads (GQA)
//!   q   = q / √d_k                                (read scale — NOT a no-op
//!                                                  despite the norm below;
//!                                                  see the layer forward)
//!   o   = recurrence(q, k, v, g, β)
//!   out = W_out · ( RMSNorm_{head}(o) ⊙ SiLU(z) )
//! ```
//!
//! Everything here runs in F32 on any device; state is always F32 — the
//! recurrence is a running sum and half precision drifts (§8 of
//! `docs/qwen35_qwen38_models.md`).

#[cfg(feature = "cuda")]
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use candle::quantized::cuda::Q8a128Operand;
#[cfg(feature = "cuda")]
use candle::quantized::SumScale;
use candle::wave_provenance::WaveTicket;
#[cfg(feature = "cuda")]
use candle::LeaseAnchor;
use candle::{DType, Device, DeviceLocation, LiveTensor, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_kernels::delta_net::DELTA_NET_SHORT_SPAN_CONV;

#[cfg(feature = "cuda")]
use crate::models::wave_buffers::upload_into;

use super::timeline::{fold_by_ops, TimelineLayout, TIMELINE_ROWS};
use super::types::{DeltaNetDims, ZGate};

/// The carried per-sequence state of one DeltaNet layer.
///
/// **Deliberately not `Clone`.** The buffers are written in place, and
/// `Tensor::clone` is a shallow handle clone that shares storage — so a derived
/// `Clone` would hand back something that *looks* like a snapshot and then
/// tracks every subsequent mutation. That is not a hypothetical: it silently
/// broke snapshot-and-resume the moment the state stopped being replaced.
/// [`Self::snapshot`] is the deep copy, and it is the only way to get a second
/// one.
#[derive(Debug)]
pub struct DeltaNetState {
    /// `[n_v_heads, d_v, d_k]`, F32.
    pub s: Tensor,
    /// The last `conv_kernel − 1` conv inputs, `[conv_dim, conv_kernel − 1]`,
    /// F32. Zeros at sequence start (causal left-padding).
    pub conv_tail: Tensor,
}

impl DeltaNetState {
    /// An independent copy of both buffers — what a snapshot has to be.
    ///
    /// Allocates, so it belongs to the reference model's per-layer state
    /// bookkeeping, not to anything on the wave path (a wave preserves the
    /// entering state by not writing it — see `RecurrentStateStore`).
    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            s: self.s.copy()?,
            conv_tail: self.conv_tail.copy()?,
        })
    }

    /// Overwrite both buffers with `src`'s, in place.
    ///
    /// The same *arithmetic* as [`Self::snapshot`] into a destination that
    /// already exists, which is what a **restore** wants: `import` has bytes
    /// from a sealed snapshot and a live buffer whose address other code has
    /// already resolved, so it fills that buffer rather than replacing it.
    /// Snapshotting by allocation made the store the largest device allocator in
    /// the decode loop: 679 MB across one measured window on the 0.8B at batch
    /// 4, more than half of everything the loop allocated, in an allocate/free
    /// pair per layer per session per token.
    ///
    /// **The wave path does not use this.** It used to: every wave copied every
    /// layer's state aside so a failure could restore it, ~2 MB and two launches
    /// per layer per wave. A wave now reads one buffer and writes the other, so
    /// the entering state is preserved by not being touched and `commit_wave`
    /// installs the result with a pointer swap.
    pub fn copy_from(&mut self, src: &Self) -> Result<()> {
        self.s.slice_set(&src.s, 0, 0)?;
        self.conv_tail.slice_set(&src.conv_tail, 0, 0)
    }

    /// Overwrite both buffers from host values, in place — a restore's write.
    ///
    /// On CUDA the bytes go straight into the slot's own storage, one upload per
    /// buffer, and nothing else is touched on the device. Restoring through
    /// [`Self::copy_from`] built a device-side source first, so every resume
    /// allocated a whole layer state from the driver to copy it once and drop
    /// it: 3.2 MiB per layer on Flash-Next, outside the reservation.
    pub fn write_host(&mut self, s: &[f32], conv_tail: &[f32]) -> Result<()> {
        match self.s.device() {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                upload_into(&self.s, s)?;
                upload_into(&self.conv_tail, conv_tail)
            }
            dev => {
                let dev = dev.clone();
                self.s
                    .slice_set(&Tensor::from_vec(s.to_vec(), self.s.shape(), &dev)?, 0, 0)?;
                self.conv_tail.slice_set(
                    &Tensor::from_vec(conv_tail.to_vec(), self.conv_tail.shape(), &dev)?,
                    0,
                    0,
                )
            }
        }
    }

    pub fn zeros(dims: &DeltaNetDims, dev: &Device) -> Result<Self> {
        Ok(Self {
            s: Tensor::zeros(
                (dims.n_v_heads, dims.head_dim, dims.head_dim),
                DType::F32,
                dev,
            )?,
            conv_tail: Tensor::zeros((dims.conv_dim(), dims.conv_kernel - 1), DType::F32, dev)?,
        })
    }

    /// Bytes one state's two buffers occupy, in the order [`Self::at`] lays
    /// them out. `(s_bytes, conv_bytes)`.
    pub fn byte_sizes(dims: &DeltaNetDims) -> (usize, usize) {
        let f32_bytes = DType::F32.size_in_bytes();
        (
            dims.n_v_heads * dims.head_dim * dims.head_dim * f32_bytes,
            dims.conv_dim() * (dims.conv_kernel - 1) * f32_bytes,
        )
    }

    /// A state whose two buffers are **views** over memory owned elsewhere —
    /// `s` at `s_ptr`, the conv tail at `conv_ptr`.
    ///
    /// This is how a sequence's recurrent memory lives inside the device
    /// reservation instead of the CUDA pool. It is a per-sequence, fixed-size,
    /// uniform allocation — the same shape as a KV arena's regions or an expert
    /// slot — and allocating it from the pool instead put ~126 MiB per sequence
    /// outside the reservation, which is memory the span was never sized for
    /// and which WDDM demotes to host RAM rather than refusing. On a hybrid
    /// stack that is three quarters of the model's per-sequence memory.
    ///
    /// **Anchored to its owner.** Both buffers hold `anchor`, and so does every
    /// clone, view and re-lease of them, so the memory's owner cannot hand it to
    /// another sequence while anything still reads this state through a handle it
    /// took along the way — a `write_half`, a captured view, a re-lease. The worst a
    /// forgotten handle can then cost is ground not yet reclaimed, never a read of
    /// someone else's state.
    ///
    /// # Safety
    ///
    /// Both pointers must name at least [`Self::byte_sizes`] bytes of correctly
    /// aligned memory that stays live while `anchor` is held, and that nothing
    /// else writes through another alias.
    #[cfg(feature = "cuda")]
    pub unsafe fn at(
        dims: &DeltaNetDims,
        dev: &Device,
        s_ptr: u64,
        conv_ptr: u64,
        anchor: LeaseAnchor,
    ) -> Result<Self> {
        Ok(Self {
            s: Tensor::from_anchored_cuda_ptr(
                s_ptr,
                DType::F32,
                (dims.n_v_heads, dims.head_dim, dims.head_dim),
                dev,
                anchor.clone(),
            )?,
            conv_tail: Tensor::from_anchored_cuda_ptr(
                conv_ptr,
                DType::F32,
                (dims.conv_dim(), dims.conv_kernel - 1),
                dev,
                anchor,
            )?,
        })
    }

    /// Handles onto both buffers, as the buffers a long span WRITES — the
    /// prefill scratch the store hands it, which `commit_wave` exchanges in.
    pub fn write_half(&self) -> DeltaNetOut {
        DeltaNetOut {
            s: self.s.clone(),
            conv_tail: self.conv_tail.clone(),
        }
    }

    /// A write half for a caller carrying **one** state — the single-span entry
    /// points and the reference tests, which have no wave to roll back.
    ///
    /// `s` is the same buffer, which is what those paths have always done and is
    /// safe: the state kernels hold their `S` tile in registers for the whole
    /// sequence, so nothing reads an element after another block wrote it. The
    /// conv tail is **not** safe that way — the prefill conv's blocks for the
    /// first `kwidth − 1` outputs read the entering tail while the block owning
    /// the segment's end writes the advanced one — so it gets a scratch buffer
    /// that [`Self::absorb_solo`] folds back once the launch is behind us.
    pub fn solo_out(&self) -> Result<DeltaNetOut> {
        Ok(DeltaNetOut {
            s: self.s.clone(),
            conv_tail: Tensor::zeros(self.conv_tail.shape(), DType::F32, self.conv_tail.device())?,
        })
    }

    /// Install the advance a [`Self::solo_out`] write half received. `s` shares
    /// its buffer and is already current; the conv tail is the copy back.
    pub fn absorb_solo(&mut self, out: &DeltaNetOut) -> Result<()> {
        self.conv_tail.slice_set(&out.conv_tail, 0, 0)
    }

    /// Allocate the buffers **without zeroing** — for a write half the kernels
    /// fully overwrite before anything reads it.
    ///
    /// Hot-path invariant 6 (`CLAUDE.md`): a buffer a kernel stamps end to end
    /// must not be memset first. The zeroed twin costs a full-width device
    /// memset per layer (~2 MB × 30 layers ≈ 63 MiB per fork) on the exact
    /// bytes the next wave overwrites, and forks run ~3× per turn.
    ///
    /// Sound for the prefill scratch: [`commit_wave`] installs `s` and the conv
    /// tail together because a long span's kernels write both into it (the conv
    /// kernels take the entering and advanced tails as two pointers).
    /// [`Self::zeros`] remains the constructor for a state that is genuinely
    /// read at zero — a sequence-start value — and [`Self::solo_out`] keeps its
    /// zeroed scratch tail deliberately.
    ///
    /// [`commit_wave`]: super::state_store::RecurrentStateStore::commit_wave
    pub fn uninit(dims: &DeltaNetDims, dev: &Device) -> Result<Self> {
        Ok(Self {
            s: Tensor::empty(
                (dims.n_v_heads, dims.head_dim, dims.head_dim),
                DType::F32,
                dev,
            )?,
            conv_tail: Tensor::empty((dims.conv_dim(), dims.conv_kernel - 1), DType::F32, dev)?,
        })
    }
}

/// Where a long span WRITES one sequence's advanced state for a layer: the
/// store's prefill scratch, which `commit_wave` exchanges in.
///
/// Both buffers are handles rather than borrows, because the reference paths
/// have a single state and name the same tensors here and in the
/// [`DeltaNetState`] they read — and the kernels take addresses either way.
#[derive(Debug, Clone)]
pub struct DeltaNetOut {
    /// `[n_v_heads, d_v, d_k]`, F32 — the advanced `S`.
    pub s: Tensor,
    /// `[conv_dim, conv_kernel − 1]`, F32 — the advanced conv tail.
    pub conv_tail: Tensor,
}

/// One token of the gated delta rule across all V heads.
///
/// `state` `[H, d_v, d_k]`; `q`/`k` `[H, d_k]` (already broadcast to V heads
/// and l2-normed); `v` `[H, d_v]`; `g_log`/`beta` `[H]`. Returns
/// `(o [H, d_v], new_state)`.
pub fn delta_step(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_log: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (h, _d_v, _d_k) = state.dims3()?;
    let decay = g_log.exp()?.reshape((h, 1, 1))?;
    let decayed = state.broadcast_mul(&decay)?;
    // The value the decayed state currently predicts for k: S k → [H, d_v].
    let pred = decayed.matmul(&k.unsqueeze(2)?)?.squeeze(2)?;
    // β-scaled correction, written back along k.
    let err = v.sub(&pred)?.broadcast_mul(&beta.reshape((h, 1))?)?;
    let outer = err.unsqueeze(2)?.matmul(&k.unsqueeze(1)?)?;
    let new_state = decayed.add(&outer)?;
    // Read with the post-update state: the token attends to itself.
    let o = new_state.matmul(&q.unsqueeze(2)?)?.squeeze(2)?;
    Ok((o, new_state))
}

/// The recurrence over a whole segment.
///
/// `q`/`k` `[T, H, d_k]`, `v` `[T, H, d_v]`, `g_log`/`beta` `[T, H]`; the
/// entering state is consumed and the post-segment state returned, so calling
/// this over `[0..a)` then `[a..T)` is identical to one call over `[0..T)` —
/// the property turn sealing and resume rest on.
pub fn delta_recurrence(
    state: Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_log: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (t, h, d_v) = v.dims3()?;
    let mut s = state;
    let mut outs = Vec::with_capacity(t);
    for i in 0..t {
        let (o, s_next) = delta_step(
            &s,
            &q.get(i)?,
            &k.get(i)?,
            &v.get(i)?,
            &g_log.get(i)?,
            &beta.get(i)?,
        )?;
        outs.push(o.reshape((1, h, d_v))?);
        s = s_next;
    }
    Ok((Tensor::cat(&outs, 0)?, s))
}

/// The chunked (parallel-scan) form of [`delta_recurrence`] — the prefill
/// path. Mathematically identical to the sequential rule; the win is that a
/// chunk of `C` tokens costs a handful of `[C, d]`-shaped batched matmuls
/// plus ONE sequential state carry per chunk instead of `C` serial steps.
/// Ported from llama.cpp's `delta-net-base.cpp` chunking graph (the external
/// oracle); the sequential reference above is the in-repo oracle both are
/// tested against.
///
/// Per chunk, with entering state `S [d_v, d_k]` per head (`o = S q`
/// orientation) and within-chunk index `i`:
///
/// ```text
///   G[i]        = Σ_{j ≤ i} g[j]                       (log-decay cumsum)
///   D[i][j]     = exp(G[i] − G[j])   for j ≤ i         (decay mask)
///   A[i][j]     = (βk[i] · k[j]) D[i][j]  for j < i    (strictly lower)
///   T           = (I + A)^{-1}                          (unit lower-tri)
///   u           = T (β v)                               (pseudo-values)
///   kcd         = T (βk ⊙ e^G)
///   v_new       = u − kcd Sᵀ                            (chunk-local writes)
///   o[i]        = S (q[i] e^{G[i]})  +  Σ_{j ≤ i} (q[i]·k[j]) D[i][j] v_new[j]
///   S_next      = e^{G[C−1]} S + v_newᵀ (k ⊙ e^{G[C−1] − G})
/// ```
///
/// Shapes as in [`delta_recurrence`]; `chunk` is the chunk width (64 in the
/// production configuration, any positive value here — the tail chunk is
/// simply shorter, no padding).
pub fn delta_chunked<'w>(
    state: &mut Tensor,
    q: &LiveTensor<'w>,
    k: &LiveTensor<'w>,
    v: &LiveTensor<'w>,
    g_log: &LiveTensor<'w>,
    beta: &LiveTensor<'w>,
    chunk: usize,
) -> Result<LiveTensor<'w>> {
    let (t, h, d_k) = q.dims3()?;
    let d_v = v.dim(2)?;
    if chunk == 0 {
        candle::bail!("delta_chunked: chunk width must be positive");
    }
    let dev = q.device();

    // **The state is a buffer, not a value.** It is allocated once per sequence
    // and lives for that sequence's whole life, so every chunk writes *into* it
    // rather than producing a replacement. What that removes is not one copy
    // but four: a transpose into FLA orientation on the way in, two fresh
    // `[H, d, d]` allocations per chunk for the carry, and a transpose back on
    // the way out — ~16 MB of traffic per layer on the 35B, for memory that
    // never needed to move.
    //
    // The reads want `Sᵀ` and the store holds `S`, but a transpose is a *view*;
    // only materialising it cost anything. The carry runs in the stored
    // orientation instead — transposing `S ← e^g·S + Kᵀ V` gives
    // `S ← e^g·S + Vᵀ K`, which is two in-place ops on the buffer.
    let mut outs: Vec<LiveTensor<'w>> = Vec::with_capacity(t.div_ceil(chunk));

    // **Head-major once, not once per chunk.** The chunk loop wants `[H, c, d]`
    // and the inputs arrive `[T, H, d]`, so the transpose used to sit inside
    // the loop — five `transpose + contiguous` copies per chunk, eleven chunks
    // deep on a 649-token prefill, for a permutation that does not depend on
    // the chunk at all. Transposed once here, a chunk is a `narrow` on the
    // token axis of an already-contiguous buffer.
    let qh = q.transpose(0, 1)?.contiguous()?; // [H, T, d_k]
    let kh = k.transpose(0, 1)?.contiguous()?;
    let vh = v.transpose(0, 1)?.contiguous()?;
    let gh = g_log.transpose(0, 1)?.contiguous()?; // [H, T]
    let bh = beta.transpose(0, 1)?.contiguous()?;

    // The triangular masks depend only on the chunk width, and every chunk but
    // the last shares one. Built here and reused; the tail rebuilds its own.
    let mut mask_width = 0usize;
    let mut masks: Option<(Tensor, Tensor)> = None;

    let mut start = 0usize;
    while start < t {
        let c = chunk.min(t - start);
        // Chunk slices, head-major: [H, c, d] / [H, c].
        // Narrowing the token axis of a head-major buffer leaves a gap between
        // heads, and both `matmul` and `cumsum` (which lowers to one) require
        // contiguity — so the slice is materialised. That is the same single
        // copy the per-chunk `transpose` used to pay for, minus the transpose.
        let qc = qh.narrow(1, start, c)?.contiguous()?;
        let kc = kh.narrow(1, start, c)?.contiguous()?;
        let vc = vh.narrow(1, start, c)?.contiguous()?;
        let gc = gh.narrow(1, start, c)?.contiguous()?; // [H, c]
        let bc = bh.narrow(1, start, c)?.contiguous()?;
        if mask_width != c {
            masks = Some((
                lower_tri_mask(c, false, dev)?,
                lower_tri_mask(c, true, dev)?,
            ));
            mask_width = c;
        }
        let (mask_incl, mask_strict) = masks.as_ref().expect("built above");
        let rows = chunk_rows(state, &qc, &kc, &vc, &gc, &bc, mask_incl, mask_strict)?;
        outs.push(rows.o.transpose(0, 1)?.contiguous()?); // [c, H, d_v]

        // State carry, in place and in the stored orientation:
        //   S ← e^{G[c−1]}·S + V_newᵀ (K ⊙ e^{G[c−1]−G})
        let g_last = rows.g_cs.narrow(1, c - 1, 1)?; // [H, 1]
        let g_diff_exp = g_last.broadcast_sub(&rows.g_cs)?.exp()?; // e^{G_last − G[i]}
        let kg = kc.broadcast_mul(&g_diff_exp.unsqueeze(2)?)?; // [H, c, d_k]
        let decay_full = g_last
            .exp()?
            .unsqueeze(2)?
            .broadcast_as((h, d_v, d_k))?
            .contiguous()?;
        state.mul_mut(&decay_full)?;
        state.add_mut(&rows.v_new.transpose(1, 2)?.contiguous()?.matmul(&kg)?)?;

        start += c;
    }

    LiveTensor::cat(&outs, 0) // [T, H, d_v]
}

/// One chunk's row quantities against its entering state, which is read and
/// not advanced: the output, the inclusive log-decay scan `G` and the
/// chunk-local writes `v_new` — what [`delta_chunked`] carries the state with,
/// and what a short span keeps as its timeline instead.
struct ChunkRows<'w> {
    /// `[H, c, d_v]`.
    o: LiveTensor<'w>,
    /// `[H, c]`.
    g_cs: LiveTensor<'w>,
    /// `[H, c, d_v]`.
    v_new: LiveTensor<'w>,
}

/// [`ChunkRows`] for one head-major chunk: `qc`/`kc` `[H, c, d_k]`, `vc`
/// `[H, c, d_v]`, `gc`/`bc` `[H, c]`, against `state` `[H, d_v, d_k]`.
#[allow(clippy::too_many_arguments)]
fn chunk_rows<'w>(
    state: &Tensor,
    qc: &LiveTensor<'w>,
    kc: &LiveTensor<'w>,
    vc: &LiveTensor<'w>,
    gc: &LiveTensor<'w>,
    bc: &LiveTensor<'w>,
    mask_incl: &Tensor,
    mask_strict: &Tensor,
) -> Result<ChunkRows<'w>> {
    let d_k = qc.dim(2)?;
    let d_v = vc.dim(2)?;
    // Inclusive log-decay cumsum and its exp.
    let g_cs = gc.cumsum(1)?; // [H, c]
    let g_exp = g_cs.exp()?; // e^{G[i]}

    let bc3 = bc.unsqueeze(2)?; // [H, c, 1]
    let k_b = kc.broadcast_mul(&bc3)?;
    let v_b = vc.broadcast_mul(&bc3)?;

    // Decay mask D[i][j] = exp(G[i] − G[j]), lower incl. diagonal.
    //
    // **The clamp is load-bearing, not defensive.** `G` is a cumulative sum
    // of log-decays and every `g` is negative, so `G` decreases: for the
    // `j ≤ i` half this exponent is ≤ 0 and `exp` is a contraction. The
    // upper half is the mirror image — `G[i] − G[j]` grows *positive* with
    // the distance — and although the mask discards it, the mask is applied
    // to the result. `exp` runs first, overflows to `+inf` once the gap
    // passes ~88 in f32, and `inf × 0` is `NaN`, which then propagates
    // through `unit_lower_inverse` into every later token.
    //
    // It is depth- and content-dependent, so short probes miss it entirely:
    // random activations over 40 tokens stay finite, while real prompt
    // embeddings poisoned the row at token 23 of one prompt and 27 of
    // another. Clamping the exponent at 0 leaves the kept half untouched
    // (it is already ≤ 0) and makes the discarded half `exp(0) = 1` before
    // the mask zeroes it.
    let gi = g_cs.unsqueeze(2)?; // [H, c, 1]
    let gj = g_cs.unsqueeze(1)?; // [H, 1, c]
    let decay = gi
        .broadcast_sub(&gj)?
        .minimum(0f64)?
        .exp()?
        .broadcast_mul(&mask_incl.unsqueeze(0)?)?; // [H, c, c]

    // A[i][j] = (βk[i]·k[j]) D[i][j], strictly lower.
    let a = k_b
        .matmul(&kc.transpose(1, 2)?)?
        .mul(&decay)?
        .broadcast_mul(&mask_strict.unsqueeze(0)?)?;

    // Pseudo-values and the cumulative-decay key read: `u = T(βv)` and
    // `kcd = T(βk ⊙ e^G)` with `T = (I + A)⁻¹`.
    //
    // **Solved, not inverted.** Both share the left side `I + A`, so they
    // are one triangular solve with the two right-hand sides side by side —
    // and a solve never forms `T`. The alternative, and what the reference
    // below still does, is a recursive block inversion costing `log₂ c`
    // levels of a handful of launches each: about eighty serially dependent
    // ops per chunk, which measured as the single largest op-count item in
    // prefill. `cublasStrsmBatched` is one launch and is backward-stable.
    let kg_scaled = k_b.broadcast_mul(&g_exp.unsqueeze(2)?)?;
    let (u, kcd) = solve_pseudo_values(&a, &v_b, &kg_scaled, d_v, d_k)?;

    // Chunk-local writes: what each token adds beyond the entering state.
    // `Sᵀ` as a view of the buffer — free, because a transpose is a layout,
    // not a copy.
    let s_fla = state.transpose(1, 2)?.contiguous()?;
    let v_new = u.sub(&kcd.matmul(&s_fla)?)?; // [H, c, d_v]

    // Output: inter-chunk read of the entering state + intra-chunk reads
    // of this chunk's own writes (inclusive diagonal — the token sees its
    // own update, matching the sequential rule's post-update read).
    let q_g = qc.broadcast_mul(&g_exp.unsqueeze(2)?)?;
    let inter = q_g.matmul(&s_fla)?; // [H, c, d_v]
    let kq = qc.matmul(&kc.transpose(1, 2)?)?.mul(&decay)?; // q[i]·k[j] D[i][j], j ≤ i
    let intra = kq.matmul(&v_new)?;
    let o = inter.add(&intra)?; // [H, c, d_v]
    Ok(ChunkRows { o, g_cs, v_new })
}

/// A short span's mixer rows against `state`, which is read and not advanced:
/// its output `[T, H, d_v]`, and its `G` `[T, H]` and `v_new` `[T, H, d_v]` for
/// the timeline. One chunk — a short span is at most [`TIMELINE_ROWS`] rows.
/// Shapes as in [`delta_chunked`].
fn delta_span_rows<'w>(
    state: &Tensor,
    q: &LiveTensor<'w>,
    k: &LiveTensor<'w>,
    v: &LiveTensor<'w>,
    g_log: &LiveTensor<'w>,
    beta: &LiveTensor<'w>,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>, LiveTensor<'w>)> {
    let t = q.dim(0)?;
    if t == 0 || t > TIMELINE_ROWS {
        candle::bail!("delta_span_rows: a short span is 1..={TIMELINE_ROWS} rows, got {t}");
    }
    let dev = q.device();
    let head_major = |x: &LiveTensor<'w>| x.transpose(0, 1)?.contiguous();
    let rows = chunk_rows(
        state,
        &head_major(q)?,
        &head_major(k)?,
        &head_major(v)?,
        &head_major(g_log)?,
        &head_major(beta)?,
        &lower_tri_mask(t, false, dev)?,
        &lower_tri_mask(t, true, dev)?,
    )?;
    Ok((
        rows.o.transpose(0, 1)?.contiguous()?,
        rows.g_cs.transpose(0, 1)?.contiguous()?,
        rows.v_new.transpose(0, 1)?.contiguous()?,
    ))
}

/// `u = T·v_b` and `kcd = T·kg` for `T = (I + A)⁻¹`, `A` strictly lower.
///
/// One triangular solve over the two right-hand sides concatenated, because
/// they share the left side. On CUDA in F32 that is a single
/// `cublasStrsmBatched`; elsewhere — CPU tests, the F32 reference, any other
/// dtype — it falls back to forming `T` with [`unit_lower_inverse`] and
/// multiplying, which is the same arithmetic and is what the solve is checked
/// against (`solve_matches_the_explicit_inverse`).
// `d_v`/`d_k` split the concatenated right-hand side that only the CUDA arm
// forms; the fallback multiplies `v_b` and `kg` separately and reads their own
// shapes.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn solve_pseudo_values<'w>(
    a: &LiveTensor<'w>,
    v_b: &LiveTensor<'w>,
    kg: &LiveTensor<'w>,
    d_v: usize,
    d_k: usize,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>)> {
    #[cfg(feature = "cuda")]
    if a.device().is_cuda() && a.dtype() == DType::F32 {
        let rhs = LiveTensor::cat(&[v_b, kg], 2)?; // [H, c, d_v + d_k]
        let x = super::cuda::solve_unit_lower(a, rhs)?;
        // Both halves are strided views of one buffer — a narrow on the last
        // axis leaves a gap between rows. `kcd` is a matmul operand and must be
        // materialised; `u` is only ever the left side of a subtraction, which
        // reads strides, so it stays a view.
        return Ok((x.narrow(2, 0, d_v)?, x.narrow(2, d_v, d_k)?.contiguous()?));
    }
    let t_inv = unit_lower_inverse(a)?;
    Ok((t_inv.matmul(v_b)?, t_inv.matmul(kg)?))
}

/// Whether a fused-path guard admitted the wave, recording **which condition
/// rejected it** when it did not.
///
/// The guards are long conjunctions, and a miss on any one silently drops the
/// whole wave to the tensor-op fallback — the same arithmetic at a fraction of
/// the speed. That is a cliff you cannot see in a timing: the span table shows
/// `dn:mix` inflated ~89× per call and says nothing about why, and
/// `docs/qwen36_performance_plan.md` T1 is explicit that timings are the wrong
/// instrument here (the profile feature's own sync distorts width rows ~2×).
///
/// So this counts paths taken, not time spent, and names the first failing
/// condition. "The fused path ran 40 times and fell back 200" is a different
/// problem from "it never ran", and "it fell back on `qkv:f32`" is a different
/// problem again from "it fell back on `head_dim`".
///
/// # Cost, on a per-layer per-wave path
///
/// The conditions arrive as a slice of **thunks**, not of `bool`s, so the walk
/// short-circuits exactly as the `&&` chain it replaced did — the `state:cuda`
/// predicate is `O(n_seqs)` and must not run once the cheap dtype checks have
/// already rejected. The tally is a fixed array of atomics indexed by position,
/// so recording is one relaxed `fetch_add`: no hashing, no map, no lock. An
/// earlier version took a global `Mutex<HashMap>` and hashed a string pair on
/// every call while claiming to be free; it was neither.
#[cfg(feature = "cuda")]
const MAX_DISPATCH_CONDS: usize = 12;

/// `[path][0]` is the fused count; `[path][i + 1]` counts rejections by
/// condition `i`. One path, so one row.
#[cfg(feature = "cuda")]
static DISPATCH_TALLY: [[std::sync::atomic::AtomicU64; MAX_DISPATCH_CONDS + 1]; 1] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; MAX_DISPATCH_CONDS + 1] }; 1];

/// The paths, in `DISPATCH_TALLY` row order.
#[cfg(feature = "cuda")]
const DISPATCH_PATHS: [&str; 1] = ["dn:mix"];

#[cfg(feature = "cuda")]
fn dispatch_verdict(row: usize, conds: &[(&'static str, &dyn Fn() -> bool)]) -> bool {
    debug_assert!(conds.len() <= MAX_DISPATCH_CONDS);
    let reject = conds.iter().position(|(_, ok)| !ok());
    let slot = reject.map_or(0, |i| i + 1);
    DISPATCH_TALLY[row][slot].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    reject.is_none()
}

/// The tally [`dispatch_verdict`] has accumulated: `(path, outcome, count)`,
/// where the outcome is `FUSED` or the first condition that rejected. Sorted
/// most-frequent first. Zero rows are omitted.
#[cfg(feature = "cuda")]
pub fn dispatch_tally() -> Vec<(&'static str, &'static str, u64)> {
    let names: [&[&str]; 1] = [&[
        "device",
        "head_dim",
        "qkv:f32",
        "alpha:f32",
        "beta:f32",
        "z:f32",
        "dt_bias:f32",
        "a:f32",
        "norm:f32",
        "state:cuda",
        "short:conv",
    ]];
    let mut v = Vec::new();
    for (row, path) in DISPATCH_PATHS.iter().enumerate() {
        for (slot, cell) in DISPATCH_TALLY[row].iter().enumerate() {
            let n = cell.load(std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let outcome = if slot == 0 {
                "FUSED"
            } else {
                names[row].get(slot - 1).copied().unwrap_or("?")
            };
            v.push((*path, outcome, n));
        }
    }
    v.sort_by_key(|e| Reverse(e.2));
    v
}

/// [`lower_tri_mask`]'s cache, keyed by `(device location, chunk width, strict)`.
type MaskCache = Mutex<HashMap<(DeviceLocation, usize, bool), Tensor>>;

/// `[c, c]` mask with 1 where `j ≤ i` (or `j < i` when `strict`), else 0.
///
/// **Built once per `(device, c, strict)` and kept.** The mask is a pure
/// function of its arguments — it depends on nothing the wave carries — so
/// rebuilding it is not a temporary, it is recomputing a constant.
///
/// The call site already avoided rebuilding it per *chunk*; what it could not
/// see is that it is rebuilt per layer, per wave. Measured by the
/// forbidden-allocation detector on the 3.6-35B gate: **120 calls, 41.3 MB per
/// config, the single largest source of device allocations made outside the
/// reservation.** Those allocations come from the free space the span
/// deliberately does not cover, which is the memory a load-time repack was
/// found competing for.
///
/// Keyed by device ordinal as well as shape: two devices need two masks, and a
/// mask handed to the wrong one is an illegal address rather than a wrong
/// answer. Bounded by construction — a model has one chunk width and two
/// strictness values, so the map holds a handful of entries a few KiB each for
/// the life of the process.
fn lower_tri_mask(c: usize, strict: bool, dev: &Device) -> Result<Tensor> {
    static MASKS: OnceLock<MaskCache> = OnceLock::new();

    // Keyed by the device's own location, not by a CUDA ordinal with one
    // sentinel for "everything else" — that collapsed CPU and Metal onto a
    // single entry, so a CPU mask could be handed to a Metal caller.
    let key = (dev.location(), c, strict);
    let cache = MASKS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(map) = cache.lock() {
        if let Some(m) = map.get(&key) {
            return Ok(m.clone());
        }
    }
    // Built on the device from two index vectors rather than as a host `Vec`:
    // the host round trip costs more than the comparison it is avoiding.
    let rows = Tensor::arange(0u32, c as u32, dev)?.reshape((c, 1))?;
    let cols = Tensor::arange(0u32, c as u32, dev)?.reshape((1, c))?;
    let keep = if strict {
        cols.broadcast_lt(&rows)?
    } else {
        cols.broadcast_le(&rows)?
    };
    let mask = keep.to_dtype(DType::F32)?;
    if let Ok(mut map) = cache.lock() {
        map.insert(key, mask.clone());
    }
    Ok(mask)
}

/// Invert `I + A` for a strictly-lower-triangular `A [H, c, c]` by forward
/// substitution: row `i` of the inverse is `e_i − Σ_{j<i} A[i][j] · row j`.
/// `c` sequential steps of batched `[H, 1, c] × [H, c, c]` work — fine for a
/// reference; the CUDA kernel does this in shared memory.
/// [`unit_lower_inverse`], for the CUDA solve's parity test.
///
/// The inverse is the reference form the `trsm` path is checked against, and
/// that check lives beside the kernel in [`super::cuda`] rather than here.
pub fn unit_lower_inverse_for_test<'w>(a: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    unit_lower_inverse(a)
}

fn unit_lower_inverse<'w>(a: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    let (_h, c, _) = a.dims3()?;
    let dev = a.device();
    if c == 0 {
        return Ok(a.clone());
    }
    // Forward substitution, one row at a time: `T[i] = e_i − A[i, <i] · T[<i]`.
    //
    // **Not the Neumann series by repeated squaring.** `A` is nilpotent so
    // `Σ (−A)^k` is exact in exact arithmetic and reaches `c` terms in `log₂ c`
    // matmuls, which is tempting — it made prefill 3× faster here. But it forms
    // `B^{c/2}` explicitly, and a strictly lower triangular matrix with entries
    // near 1 grows about `C(c, c/2)` under powering: at `c = 64` the logits had
    // already lost three digits against this form (cosine 0.99 at 32 tokens,
    // 0.81 at 64) and at `c = 256` they came back non-finite. Substitution
    // forms no powers and stays exact.
    //
    // Done by halves rather than by rows. Writing `I + A` in 2×2 blocks,
    //
    //   ⎡P 0⎤⁻¹   ⎡  P⁻¹      0 ⎤
    //   ⎣C Q⎦   = ⎣−Q⁻¹ C P⁻¹  Q⁻¹⎦
    //
    // and `P`, `Q` are themselves unit lower triangular, so the same rule
    // applies to them. **The two sub-problems are independent**, so they are
    // stacked into the batch axis and solved by one recursive call: the depth
    // is `log₂ c` levels of a handful of launches each, against `c` rounds of
    // row substitution. Total element count halves every level, so the extra
    // batch costs nothing.
    //
    // Numerically this *is* substitution — every quantity is a product of
    // sub-inverses with original blocks, and no power of `A` is ever formed.
    if c == 1 {
        // A 1×1 unit lower triangular block is `[1]`, and so is its inverse.
        return Tensor::ones((_h, 1, 1), a.dtype(), dev);
    }
    let half = c / 2;
    if c % 2 != 0 {
        // Odd widths only arise from a ragged tail; peel one row the direct
        // way rather than carry a padding case through the recursion.
        let head = unit_lower_inverse(&a.narrow(1, 0, c - 1)?.narrow(2, 0, c - 1)?.contiguous()?)?;
        let a_row = a.narrow(1, c - 1, 1)?.narrow(2, 0, c - 1)?.contiguous()?;
        let last = a_row.matmul(&head)?.neg()?; // [H, 1, c-1]
        let one = Tensor::ones((_h, 1, 1), a.dtype(), dev)?;
        let zeros = Tensor::zeros((_h, c - 1, 1), a.dtype(), dev)?;
        return LiveTensor::cat(
            &[
                LiveTensor::cat(&[&head, &zeros], 2)?,
                LiveTensor::cat(&[&last, &one], 2)?,
            ],
            1,
        );
    }
    let p = a.narrow(1, 0, half)?.narrow(2, 0, half)?;
    let q = a.narrow(1, half, half)?.narrow(2, half, half)?;
    let cblk = a.narrow(1, half, half)?.narrow(2, 0, half)?.contiguous()?;
    // One call for both diagonal blocks.
    let stacked = LiveTensor::cat(&[&p.contiguous()?, &q.contiguous()?], 0)?;
    let t_stack = unit_lower_inverse(&stacked)?;
    let t_p = t_stack.narrow(0, 0, _h)?;
    let t_q = t_stack.narrow(0, _h, _h)?;
    let corner = t_q.matmul(&cblk.matmul(&t_p)?)?.neg()?;
    let zeros = Tensor::zeros((_h, half, half), a.dtype(), dev)?;
    LiveTensor::cat(
        &[
            LiveTensor::cat(&[&t_p, &zeros], 2)?,
            LiveTensor::cat(&[&corner, &t_q], 2)?,
        ],
        1,
    )
}

/// Causal 1-D conv over channels with a carried tail.
///
/// `x` `[C, T]` (channel-major), `kernel` `[C, K]` (a depthwise kernel per
/// channel), `tail` `[C, K − 1]` — the trailing inputs of the previous
/// segment, zeros at sequence start. Returns `(y [C, T], new_tail)`.
///
/// `y[c, t] = Σ_j kernel[c, j] · xpad[c, t + j]` where `xpad = [tail | x]` —
/// i.e. output `t` sees inputs `t − K + 1 ..= t`, the standard causal form.
pub fn causal_conv1d<'w>(
    x: &LiveTensor<'w>,
    kernel: &Tensor,
    tail: &Tensor,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>)> {
    let (c, t) = x.dims2()?;
    let (ck, k) = kernel.dims2()?;
    if ck != c {
        candle::bail!("causal_conv1d: {c} channels but kernel has {ck}");
    }
    // `[tail | x]`, assembled rather than concatenated. `cat` inherits its arena
    // from its **first** argument, and that one is the carried tail — a buffer
    // the sequence owns across every wave, so it names no arena and the whole
    // conv would land on the pool. Allocating beside `x` instead puts the widest
    // buffer in this layer on the wave's span; the two writes are the same two
    // copies `cat` would have made.
    let padded = x.empty_beside((c, k - 1 + t), x.dtype())?; // [C, K-1+T]
    padded.slice_set(tail, 1, 0)?;
    padded.slice_set(x, 1, k - 1)?;
    let mut acc: Option<LiveTensor<'w>> = None;
    for j in 0..k {
        let win = padded.narrow(1, j, t)?;
        let term = win.broadcast_mul(&kernel.narrow(1, j, 1)?)?;
        acc = Some(match acc {
            Some(a) => a.add(&term)?,
            None => term,
        });
    }
    let y = acc.expect("kernel width is at least 1");
    let new_tail = padded.narrow(1, t, k - 1)?.contiguous()?;
    Ok((y, new_tail))
}

/// Per-row L2 normalization: `x / max(sqrt(Σ x²), eps)` over the last dim —
/// ggml's `ggml_l2_norm` semantics, which the llama.cpp graph applies to the
/// post-conv Q and K heads.
///
/// Note the floor is on the *root*, not on the sum (`max(√Σ, ε)`, not
/// `√max(Σ, ε)`). The two differ by `ε²` in where the clamp bites, which
/// only matters for near-zero rows — but this family's epsilon terms have
/// already proven load-bearing once (see the read scale in
/// [`delta_net_layer_forward`]), so it follows the reference exactly.
pub fn l2_norm<'w>(x: &LiveTensor<'w>, eps: f64) -> Result<LiveTensor<'w>> {
    // `maximum` against the scalar, not against a materialised tensor of it:
    // the tensor form allocated and filled a buffer the size of the reduction
    // on every call, twice per DeltaNet layer, for a constant.
    let sumsq = x.sqr()?.sum_keepdim(candle::D::Minus1)?;
    let denom = sumsq.sqrt()?.maximum(eps)?;
    x.broadcast_div(&denom)
}

/// SiLU (x · sigmoid(x)) on any shape.
pub fn silu<'w>(x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    x.broadcast_mul(&candle_nn::ops::sigmoid(x)?)
}

/// Numerically-stable softplus: `ln(1 + eˣ) = max(x, 0) + ln(1 + e^{−|x|})`.
pub fn softplus<'w>(x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    // `relu`, not `maximum(0f64)`. They are the same function, but the scalar
    // form goes through the tensor-or-scalar path, which materialises the
    // constant as a device tensor — a host-to-device copy and an allocation on
    // every call, per layer, per sequence, per token.
    let max0 = x.relu()?;
    let expneg = x.abs()?.neg()?.exp()?;
    let ln1p = (expneg + 1.0)?.log()?;
    max0.add(&ln1p)
}

/// The weights of one DeltaNet layer (reference path: plain F32 tensors).
///
/// Names mirror the GGUF tensors (§7.1 of the design doc): `wqkv` =
/// `attn_qkv`, `wz` = `attn_gate`, `w_beta`/`w_alpha` = `ssm_beta`/`ssm_alpha`,
/// `dt_bias` = `ssm_dt` (a bias vector), `a` = `ssm_a` (already `−exp(A_log)`
/// in the checkpoint), `conv` = `ssm_conv1d`, `norm` = `ssm_norm` (per-V-head
/// RMS weight), `w_out` = `ssm_out`.
#[derive(Debug, Clone)]
pub struct DeltaNetWeights {
    /// `[2·key_dim + value_dim, hidden]` — fused Q|K|V projection.
    pub wqkv: Tensor,
    /// `[value_dim, hidden]` — the output gate `z`.
    pub wz: Tensor,
    /// `[n_v_heads, hidden]`.
    pub w_beta: Tensor,
    /// `[n_v_heads, hidden]`.
    pub w_alpha: Tensor,
    /// `[n_v_heads]`.
    pub dt_bias: Tensor,
    /// `[n_v_heads]`, negative (`−exp(A_log)`).
    pub a: Tensor,
    /// `[conv_dim, conv_kernel]` depthwise causal kernel.
    pub conv: Tensor,
    /// `[head_dim]` RMS weight applied per V head before the z-gate.
    pub norm: Tensor,
    /// `[hidden, value_dim]`.
    pub w_out: Tensor,
}

/// Per-V-head RMSNorm (weighted), `x [T, H, d]` → same shape.
fn rms_norm_per_head<'w>(x: &LiveTensor<'w>, weight: &Tensor, eps: f64) -> Result<LiveTensor<'w>> {
    let ms = x.sqr()?.mean_keepdim(candle::D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    x.broadcast_div(&denom)?.broadcast_mul(weight)
}

/// The projection outputs one DeltaNet layer's mixer consumes.
///
/// Splitting these out is what lets the reference (plain F32 `matmul`) and the
/// production (quantized `QMatMul`) paths share [`delta_net_mix`] verbatim.
/// Everything downstream of here is dtype-agnostic tensor algebra, and it is
/// algebra with load-bearing epsilons — it exists once, in one function, and
/// both paths call it.
/// `'w` is the wave span the projections were written into, and the mixer's
/// output carries it too: everything between them is computed from these, so
/// operand provenance puts all of it in the same arena and the borrow checker
/// holds it there. The reference path instantiates `'w` as `'static`, where the
/// bound is vacuous and the tensors really are owned.
pub struct DeltaNetProjections<'w> {
    /// `[T, conv_dim]` — the fused `[Q|K|V]` projection, un-convolved.
    pub qkv: LiveTensor<'w>,
    /// `[T, value_dim]` — the output gate, pre-SiLU.
    pub z: LiveTensor<'w>,
    /// `[T, n_v_heads]` — the raw `ssm_beta` projection, pre-sigmoid.
    pub beta_lin: LiveTensor<'w>,
    /// `[T, n_v_heads]` — the raw `ssm_alpha` projection, pre-bias/softplus.
    pub alpha_lin: LiveTensor<'w>,
}

/// One sequence's run of rows in a packed wave buffer, with the state it
/// carries.
///
/// The mixer is **row-wise except in two places**: the causal conv carries a
/// tail and the delta rule carries `S`, and both are per sequence. Everything
/// else — the projections, the gates, the norms, the GQA broadcast, the output
/// gate — computes each row from that row alone. Passing the whole wave and
/// naming the sequence boundaries is what lets the row-wise majority run once
/// instead of once per sequence.
pub struct DeltaNetSeq<'a> {
    /// First row of this sequence in the packed buffer.
    pub start: usize,
    /// How many rows it contributes.
    pub len: usize,
    /// The recurrent state this sequence enters the wave with. Read, never
    /// written: a short span's contribution lands in its timeline and a long
    /// span's advanced state in [`SpanWrite::State`].
    pub state: &'a mut DeltaNetState,
    /// Where the span's work lands.
    pub write: SpanWrite,
}

/// Where a span's work lands — chosen by its length, never by its purpose
/// (`docs/deltanet_timeline.md` §7). A clone is another handle on the same
/// buffers.
#[derive(Debug, Clone)]
pub enum SpanWrite {
    /// A span of at most [`TIMELINE_ROWS`] rows: its timeline, a
    /// [`TimelineLayout`] buffer the span writes one row per token into. The
    /// state and the conv tail are read and left as they are; the fold applies
    /// the rows the step keeps.
    Timeline(Tensor),
    /// A longer span: the buffers its advanced state and conv tail are written
    /// to — the store's prefill scratch, which `commit_wave` exchanges with
    /// the live state.
    State(DeltaNetOut),
}

impl SpanWrite {
    pub fn timeline(&self) -> Option<&Tensor> {
        match self {
            Self::Timeline(t) => Some(t),
            Self::State(_) => None,
        }
    }

    pub fn state(&self) -> Option<&DeltaNetOut> {
        match self {
            Self::Timeline(_) => None,
            Self::State(o) => Some(o),
        }
    }
}

/// Check that every span's write fits its length: a timeline for a span of at
/// most [`TIMELINE_ROWS`] rows, a state for a longer one.
fn check_span_writes(seqs: &[DeltaNetSeq<'_>]) -> Result<()> {
    for (i, s) in seqs.iter().enumerate() {
        let short = s.len <= TIMELINE_ROWS;
        if short != s.write.timeline().is_some() {
            candle::bail!(
                "delta_net_mix_spans: span {i} of {} rows was handed a {} — a span of at \
                 most {TIMELINE_ROWS} rows writes a timeline, a longer one a state",
                s.len,
                if short { "state" } else { "timeline" }
            );
        }
    }
    Ok(())
}

/// The write for a single span over **one** carried state, outside any store —
/// the single-span entry points and the reference tests, which have no wave to
/// roll back. A short span writes a timeline of its own, which
/// [`finish_solo`] folds whole; a longer one writes `s` in place and its conv
/// tail to a scratch buffer [`finish_solo`] copies back.
pub fn solo_write(state: &DeltaNetState, len: usize, dims: &DeltaNetDims) -> Result<SpanWrite> {
    if len <= TIMELINE_ROWS {
        Ok(SpanWrite::Timeline(
            TimelineLayout::of(dims).alloc(state.s.device())?,
        ))
    } else {
        Ok(SpanWrite::State(state.solo_out()?))
    }
}

/// Land a [`solo_write`] span's work in its state: fold every row of a
/// timeline, or take the advanced conv tail back from its scratch buffer.
pub fn finish_solo(
    state: &mut DeltaNetState,
    write: &SpanWrite,
    len: usize,
    dims: &DeltaNetDims,
) -> Result<()> {
    match write {
        SpanWrite::Timeline(tl) => fold_one(state, tl, len, dims),
        SpanWrite::State(out) => state.absorb_solo(out),
    }
}

/// Fold the first `m` rows of `tl` into `state` — the fold kernel on a CUDA
/// device at the kernels' head width, the tensor-op fold elsewhere.
pub fn fold_one(
    state: &mut DeltaNetState,
    tl: &Tensor,
    m: usize,
    dims: &DeltaNetDims,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    if state.s.device().is_cuda() && super::cuda::fold_runs_fused(dims) {
        return super::cuda::delta_net_fold_states(&[(&*state, tl, m)], dims);
    }
    fold_by_ops(state, tl, m, dims)
}

/// Where one sequence's rows sit in the wave's packed activation buffer.
///
/// The buffer is flat `[1, total_tokens, hidden]` with each sequence a
/// contiguous run, so a recurrent mixer only needs the run's start and
/// length to take its own slice. [`DeltaNetSeq`] is this plus the state the
/// run advances; a `SeqSpan` is the pure geometry, used where the state is
/// looked up separately (the sweep, the decode pointer table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqSpan {
    /// The sequence's id, which is what keys its recurrent state.
    pub seq: usize,
    /// First row of this sequence in the packed buffer.
    pub start: usize,
    /// How many rows it contributes to this wave.
    pub len: usize,
}

/// Build the per-sequence spans of a packed wave buffer.
///
/// `q_lens` is in the same order the buffer was packed in, which is the
/// engine's `[decode | prefill | glue]` order — one row per decode sequence,
/// `q_len` rows per prefill or glue sequence.
pub fn seq_spans(seqs: &[usize], q_lens: &[usize]) -> Result<Vec<SeqSpan>> {
    if seqs.len() != q_lens.len() {
        candle::bail!(
            "seq_spans: {} sequences against {} lengths",
            seqs.len(),
            q_lens.len()
        );
    }
    let mut spans = Vec::with_capacity(seqs.len());
    let mut start = 0usize;
    for (&seq, &len) in seqs.iter().zip(q_lens) {
        spans.push(SeqSpan { seq, start, len });
        start += len;
    }
    Ok(spans)
}

/// One layer's slice of the decode pointer table: device arrays of each
/// decode sequence's conv-tail, state and timeline base pointers
/// (`ptrs [3, n_decode]` I64) plus each sequence's row in the wave tensors
/// (`rows [n_decode]` U32).
///
/// Decode states live in per-session allocations, so batching the decode
/// conv and recurrence into one launch each needs their addresses on the
/// device. The wave driver builds the whole forward's table in ONE host
/// upload before the layer sweep ([`super::cuda::build_wave_table`]) — never
/// per layer, where the upload's stream sync would serialise the launch
/// pipeline — and each layer receives its slice. Pointers are stable within the
/// forward — the store holds the buffers and their anchors keep the slots held
/// — but not across one: compaction moves slots between forwards, so a table
/// is never kept past the forward that built it. `'w` because a slice of
/// [`super::cuda::DeltaNetWaveTable`] borrows it, and that table lives on the
/// forward-phase span — so this must not outlive the generation whose reset
/// reclaims it. A per-layer table built standalone is `'static`, which this
/// still admits.
pub struct DeltaNetLayerTable<'w> {
    /// `[3, n_decode]` I64 device tensor (or a view of the wave table): conv
    /// tail, state, timeline.
    pub ptrs: LiveTensor<'w>,
    /// `[n_decode]` U32 wave rows.
    pub rows: LiveTensor<'w>,
}

/// The same table for the wave's **multi-row** spans, in one of two forms by
/// length: a short table (`[3, n]` — conv tail, state, timeline) for spans of at
/// most [`TIMELINE_ROWS`] rows, which the short-span kernel runs, and a long
/// table (`[4, n]` — conv tail in, conv tail out, state in, state out) for the
/// prefill spans the conv + scan pair runs.
///
/// A span differs from a decode row only in occupying `len` consecutive rows
/// instead of one, so it needs the same pointers plus that extent. With
/// them on the device, every span in the wave runs in one launch per kernel
/// rather than one launch per span (hot-path invariant 5) — which is what the
/// decode rows next door have always done.
///
/// The saving is launches, not arithmetic: each span still walks its own rows
/// in order, because `S` is a running sum within a sequence. What changes is
/// that four sequences no longer serialise behind each other to do it.
/// The table is **wave-scoped**, and the lifetime says so.
///
/// Its two uploads are rebuilt every wave (the pointers and the packing extents
/// are what change), so they belong on the wave's transient tier rather than in
/// driver memory the reservation does not cover. Placing them there means they
/// die when the generation's cursor rewinds — which is precisely what `'w`
/// encodes, and why the fields cannot be plain `Tensor`: `Tensor` is
/// `LiveTensor<'static>`, and a `'static` handle onto recycled arena memory is
/// the use-after-reset the lifetimes exist to prevent.
pub struct DeltaNetSpanTable<'w> {
    /// `[3, n]` I64 for a short table, `[4, n]` for a long one.
    pub ptrs: LiveTensor<'w>,
    /// `[2, n]` U32: first row in the packed wave buffer, then row count.
    pub spans: LiveTensor<'w>,
    pub n: usize,
    /// Rows in the longest span — the launch's grid extent over tokens.
    pub max_len: usize,
}

/// The non-projection constants of a DeltaNet layer.
pub struct DeltaNetConstants<'a> {
    /// `[n_v_heads]`.
    pub dt_bias: &'a Tensor,
    /// `[n_v_heads]`, already `−exp(A_log)`.
    pub a: &'a Tensor,
    /// `[conv_dim, conv_kernel]`.
    pub conv: &'a Tensor,
    /// `[head_dim]` per-V-head RMS gain.
    pub norm: &'a Tensor,
}

/// Advance the causal conv over a channel-major segment — the tensor-op path
/// [`conv_silu_spans`] falls back to where the CUDA kernels do not apply.
/// Same numbers as the kernels — [`super::cuda`]'s tests lock them to
/// [`causal_conv1d`] column by column — so which one runs is never which
/// arithmetic.
fn conv_advance<'w>(
    x: &LiveTensor<'w>,
    kernel: &Tensor,
    tail: &Tensor,
    tail_out: &Tensor,
) -> Result<LiveTensor<'w>> {
    // The multi-token scan derives the advanced tail from the padded input
    // rather than shifting it, so it produces a *value*, which is written into
    // the destination buffer rather than replacing it: a sequence's state
    // buffers are allocated once and keep their identity, and the advanced
    // value is wave-scoped where the carried buffer outlives the wave. The copy
    // is the tensor-op path's alone — the CUDA conv kernels write the
    // destination directly.
    let (y, advanced) = causal_conv1d(x, kernel, tail)?;
    tail_out.slice_set(&advanced, 1, 0)?;
    Ok(y)
}

/// The conv + SiLU stage over the wave's spans: token-major `[T, conv_dim]`
/// in, token-major `[T, conv_dim]` out, every sequence's carried tail advanced
/// in place.
///
/// Two layouts serve one arithmetic. On CUDA in F32 the kernels read the
/// projection's own token-major rows — a narrow on the token axis of a
/// contiguous buffer, so the channel-major transpose (and its twin on the way
/// back out, two full copies of the widest buffer in the layer) never exists.
/// Everywhere else the tensor-op conv wants channel-major and the two
/// transposes are the price of admission.
fn conv_silu_spans<'w>(
    qkv: &LiveTensor<'w>,
    kernel: &Tensor,
    seqs: &mut [DeltaNetSeq<'_>],
) -> Result<LiveTensor<'w>> {
    // Channel-major tensor-op conv (CPU, non-F32 reference modes — the fused
    // CUDA path never reaches here).
    let x_cm = qkv.t()?.contiguous()?; // [conv_dim, T]
    let mut parts: Vec<LiveTensor<'w>> = Vec::with_capacity(seqs.len());
    for s in seqs.iter_mut() {
        let xs = x_cm.narrow(1, s.start, s.len)?.contiguous()?;
        parts.push(match &s.write {
            SpanWrite::State(out) => conv_advance(&xs, kernel, &s.state.conv_tail, &out.conv_tail)?,
            // A short span leaves the tail as it is: its raw rows go to the
            // timeline, and the fold shifts in the ones the step keeps.
            SpanWrite::Timeline(_) => causal_conv1d(&xs, kernel, &s.state.conv_tail)?.0,
        });
    }
    // `cat`, not a preallocated buffer written span by span. The two are the
    // same work for several sequences, and **not** the same for one: `cat` of a
    // single argument hands it straight back, where writing into a shared buffer
    // costs a full copy of it.
    let conved = LiveTensor::cat(&parts, 1)?;
    silu(&conved)?.t()?.contiguous() // back to [T, conv_dim]
}

/// The conv stage of the fused CUDA path: the token-parallel conv kernels
/// with the SiLU + Q|K-norm epilogue, writing every span's rows directly into
/// one whole-wave `[T, conv_dim]` buffer — which IS the mixer's operand
/// buffer. No split, activation, norm launch, or concatenation exists
/// downstream, and the wave's decode spans are ONE launch regardless of how
/// many sessions it carries.
#[cfg(feature = "cuda")]
fn conv_fused_spans<'w>(
    qkv: &LiveTensor<'w>,
    kernel: &Tensor,
    qk_channels: usize,
    eps: f32,
    table: Option<&DeltaNetLayerTable>,
    long: Option<&DeltaNetSpanTable>,
) -> Result<LiveTensor<'w>> {
    let (t, channels) = qkv.dims2()?;
    // Every row a kernel reads is written exactly once — by the batched decode
    // conv or by a long span's conv — so the buffer is allocated uninitialised
    // (hot-path invariant 6), in `qkv`'s arena. Short spans convolve inside
    // the short-span kernel, which reads none of these rows, so theirs stay
    // unwritten.
    let conved = qkv.empty_beside((t, channels), DType::F32)?;
    if let Some(tbl) = table {
        // All decode spans in one launch: tails via the pointer table, rows
        // scattered into the shared buffer, each row's raw input into its
        // timeline.
        super::cuda::delta_net_conv_decode(qkv, kernel, tbl, &conved, qk_channels, eps)?;
    }
    if let Some(sp) = long {
        // Every long span in ONE launch, token-parallel over the widest of
        // them, each writing its own rows of the shared buffer. The advanced
        // tails land in the prefill scratch — separate allocations from the
        // entering ones, which the kernel's readers of the first `kwidth−1`
        // outputs are still reading concurrently — so there is nothing to copy
        // back.
        super::cuda::delta_net_conv_prefill(qkv, kernel, sp, qk_channels, eps, &conved)?;
    }
    Ok(conved)
}

/// Tokens the parallel scan processes between sequential state carries.
///
/// A compute-for-launches trade: work inside a chunk grows with `c²`, while
/// the number of chunks — each one a serial state carry over `[H, c, ·]`
/// tensors small enough that dispatch dominates — falls as `1/c`. The
/// arithmetic is identical at any width (`chunked_matches_sequential_for_all_chunk_widths`).
///
/// The width is only free to be tuned because [`unit_lower_inverse`] is
/// substitution. A Neumann-series inverse by repeated squaring is far cheaper
/// asymptotically and was tried here, but it forms `B^{c/2}` explicitly, and a
/// strictly lower triangular matrix with entries near 1 grows about
/// `C(c, c/2)` under powering — three digits of the logits were gone at
/// `c = 64` and they were non-finite at `c = 256`. Widening the chunk under
/// *that* inverse would have silently traded accuracy for speed.
pub const DELTA_CHUNK: usize = 256;

/// The mixer over a packed buffer holding **one sequence**.
///
/// [`delta_net_mix_spans`] with a single span. The reference forward and the
/// per-layer parity tests are the callers; the engine passes its whole wave.
pub fn delta_net_mix<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    state: &mut DeltaNetState,
    rms_eps: f64,
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let (t, _) = p.qkv.dims2()?;
    // One carried state and no wave to roll back: a short span's timeline is
    // folded whole, a long span writes `s` in place.
    let write = solo_write(state, t, dims)?;
    let mut one = [DeltaNetSeq {
        start: 0,
        len: t,
        state,
        write,
    }];
    // The single-sequence reference path, off the sweep: no phase is open.
    let mixed = delta_net_mix_spans(p, c, dims, &mut one, rms_eps, None, zgate, None)?;
    let [seq] = one;
    finish_solo(seq.state, &seq.write, t, dims)?;
    Ok(mixed)
}

/// Everything between the input projections and the output projection, over a
/// packed buffer of **several** sequences: conv+SiLU → split → l2-norm → GQA
/// broadcast → read scale → recurrence → gated norm. Returns the
/// `[T, value_dim]` gated activations the output projection consumes; each
/// sequence's state is advanced in place.
///
/// # Why this takes the whole wave
///
/// Only two steps here are per sequence — the conv, which carries a tail, and
/// the delta rule, which carries `S`. The other forty-odd ops compute each row
/// from that row alone. Running the *whole* mixer once per sequence, which is
/// what a per-sequence caller forces, therefore re-does all of that N times and,
/// worse, re-reads every projection weight N times: a decode step is weight-
/// bandwidth-bound, so N sessions cost N× a single session and batching buys
/// nothing. Measured on the 0.8B, decode went 1 → 4 contexts for 1.37× where the
/// dense models in the same harness get 3.1×.
///
/// So the row-wise majority runs once over all `T` rows and the two carried
/// steps run per sequence, each writing its slice of a shared buffer.
// `table` is the pre-uploaded per-sequence pointer table the batched decode
// kernel reads; without `cuda` there is no kernel to hand it to.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_mix_spans<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    zgate: ZGate,
    // The open layer phase, for the per-layer tables built below.
    ticket: Option<WaveTicket>,
) -> Result<LiveTensor<'w>> {
    let (t, _) = p.qkv.dims2()?;
    // **The spans must tile the buffer exactly, in order.** The two carried
    // steps below slice their inputs by `start`/`len` and their results are
    // concatenated back in span order, so a gap, an overlap or a reordering
    // silently feeds one sequence another's rows — a wrong answer, and one that
    // looks like a model fault rather than a packing fault. Checked rather than
    // assumed: the caller derives these from the wave's own packing, and this is
    // the one place that can see both.
    check_span_tiling(seqs, t)?;
    check_span_writes(seqs)?;
    #[cfg(feature = "cuda")]
    let d = dims.head_dim;

    #[cfg(feature = "cuda")]
    if let Some(o) = fused_mix(p, c, dims, seqs, rms_eps, table, ticket)? {
        // The z-gate is a kernel template instantiation: SiLU for the Qwen3.5
        // lineage, sigmoid for qwen4exp (§12.6 — the one numerical difference
        // between the two generations' GDN).
        return super::cuda::delta_net_norm_gate(&o, &p.z, c.norm, d, rms_eps as f32, zgate);
    }
    mix_spans_by_ops(p, c, dims, seqs, rms_eps, zgate)
}

/// What [`delta_net_mix_spans_q8`] hands an int8 output projection: its operand
/// emitted by the fused epilogue, or — where there is no fused epilogue to emit
/// from — the F32 activations for the projection to quantize itself.
#[cfg(feature = "cuda")]
pub enum MixOut<'w> {
    /// The gated activations as F32.
    F32(LiveTensor<'w>),
    /// The gated activations as the projection's q8a128 operand, with no F32
    /// copy stored.
    Q8(Q8a128Operand<'w>),
}

/// [`delta_net_mix_spans`] for a layer whose output projection runs int8: on
/// the fused path the epilogue writes its output as the projection's q8a128
/// operand under `sum_scale` (`delta_net_norm_gate_q8`) and stores no F32 copy
/// nothing would read. The tensor-op path — CPU, a reference dtype, another
/// head geometry — has no fused epilogue to emit from and returns the F32
/// activations, leaving the projection to quantize them itself.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_mix_spans_q8<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    zgate: ZGate,
    ticket: Option<WaveTicket>,
    sum_scale: SumScale,
) -> Result<MixOut<'w>> {
    let (t, _) = p.qkv.dims2()?;
    check_span_tiling(seqs, t)?;
    check_span_writes(seqs)?;
    let d = dims.head_dim;
    match fused_mix(p, c, dims, seqs, rms_eps, table, ticket)? {
        Some(o) if d == 128 => Ok(MixOut::Q8(super::cuda::delta_net_norm_gate_q8(
            &o,
            &p.z,
            c.norm,
            d,
            rms_eps as f32,
            zgate,
            sum_scale,
        )?)),
        Some(o) => Ok(MixOut::F32(super::cuda::delta_net_norm_gate(
            &o,
            &p.z,
            c.norm,
            d,
            rms_eps as f32,
            zgate,
        )?)),
        None => Ok(MixOut::F32(mix_spans_by_ops(
            p, c, dims, seqs, rms_eps, zgate,
        )?)),
    }
}

/// The spans must tile the packed buffer exactly, in order — see
/// [`delta_net_mix_spans`].
fn check_span_tiling(seqs: &[DeltaNetSeq<'_>], t: usize) -> Result<()> {
    let mut cursor = 0usize;
    for (i, s) in seqs.iter().enumerate() {
        if s.start != cursor {
            candle::bail!(
                "delta_net_mix_spans: sequence {i} starts at {} but the previous \
                 one ended at {cursor} — spans must tile the packed buffer",
                s.start
            );
        }
        cursor += s.len;
    }
    if cursor != t {
        candle::bail!("delta_net_mix_spans: spans cover {cursor} rows but the buffer holds {t}");
    }
    Ok(())
}

/// The fused mixer up to its epilogue — the conv, the scan or decode step —
/// returning the whole-wave output `o` the norm-gate epilogue reads, or `None`
/// when this layer's dtypes, geometry or device rule the fused kernels out.
#[cfg(feature = "cuda")]
fn fused_mix<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    ticket: Option<WaveTicket>,
) -> Result<Option<LiveTensor<'w>>> {
    let (t, _) = p.qkv.dims2()?;
    let d = dims.head_dim;
    let (qkv, w) = (&p.qkv, c);
    let key_dim = dims.key_dim();

    // The fused path: everything from the conv to the output projection in
    // one launch over the short spans, three over the long ones, two over the
    // decode spans, plus one epilogue launch over the whole wave. The conv
    // kernels' SiLU + Q|K-norm epilogue makes their output THE operand buffer —
    // q, k and v are strided views of it, the GQA broadcast is an index
    // (`kh = h % h_k`), the read scale is applied on load, the gates are
    // computed in-kernel, and each span writes its own rows of one shared
    // output. So the split/norm launches and their full-width copy, the
    // repeat/scale materialisations, the per-span operand copies, the output
    // concatenation, and the epilogue's three full-width intermediates do not
    // exist here. The tensor-op path (`mix_spans_by_ops`) is the same
    // arithmetic (the kernels are parity-locked to it) and serves CPU,
    // reference dtypes, and any other head geometry.
    //
    // A short multi-row span has one kernel, the short-span scan, compiled for
    // one conv width; a wave carrying such a span at another width runs by
    // ops whole.
    let fused_ok = dispatch_verdict(
        0,
        &[
            ("device", &|| qkv.device().is_cuda()),
            ("head_dim", &|| d == super::cuda::DELTA_NET_PREFILL_DIM),
            ("qkv:f32", &|| qkv.dtype() == DType::F32),
            ("alpha:f32", &|| p.alpha_lin.dtype() == DType::F32),
            ("beta:f32", &|| p.beta_lin.dtype() == DType::F32),
            ("z:f32", &|| p.z.dtype() == DType::F32),
            ("dt_bias:f32", &|| c.dt_bias.dtype() == DType::F32),
            ("a:f32", &|| c.a.dtype() == DType::F32),
            ("norm:f32", &|| c.norm.dtype() == DType::F32),
            ("state:cuda", &|| {
                seqs.iter().all(|s| s.state.s.device().is_cuda())
            }),
            ("short:conv", &|| {
                dims.conv_kernel == DELTA_NET_SHORT_SPAN_CONV
                    || !seqs.iter().any(|s| s.len > 1 && s.len <= TIMELINE_ROWS)
            }),
        ],
    );
    if !fused_ok {
        return Ok(None);
    }
    // The decode spans run as ONE conv launch and ONE step launch however
    // many sessions the wave carries, through the pointer table. The hot path
    // receives the table from the wave driver (built once per forward); the
    // reference path and unit tests, which call in without one, build a
    // per-layer table here — same kernels, same arithmetic, the upload merely
    // happens closer to the launch.
    let local = match table {
        Some(_) => None,
        None if seqs.iter().any(|s| s.len == 1) => {
            Some(super::cuda::build_layer_table(seqs, ticket)?)
        }
        None => None,
    };
    let table = table.or(local.as_ref());
    // The multi-row spans get the same treatment, built here because unlike the
    // decode table their extents are a property of THIS wave's packing rather
    // than of the sequences, so there is nothing for the driver to hoist out of
    // the layer sweep. Split by length: a short span — a verify block, a short
    // prompt — runs the short-span scan into its timeline, one launch doing the
    // conv, the intra-chunk solve and the state pass's reads on chip; a long
    // span runs the conv + scan pair into the prefill scratch. A wave carrying
    // both launches both.
    let (short, long) = super::cuda::build_span_tables(seqs, qkv)?;

    let conved = conv_fused_spans(
        qkv,
        w.conv,
        2 * key_dim,
        rms_eps as f32,
        table,
        long.as_ref(),
    )?;
    let o = conved.empty_beside((t, dims.value_dim()), DType::F32)?;
    let fused = super::cuda::DeltaNetFused {
        conved: &conved,
        alpha: &p.alpha_lin,
        blin: &p.beta_lin,
        dt_bias: c.dt_bias,
        a: c.a,
        o: &o,
        // See the read-scale comment on the fallback below: this is
        // load-bearing through the epilogue's epsilon floor.
        q_scale: (1.0 / (d as f64).sqrt()) as f32,
    };
    if let Some(tbl) = table {
        super::cuda::delta_net_decode_batch(&fused, tbl)?;
    }
    if let Some(sp) = short.as_ref() {
        super::cuda::delta_net_short_span_scan(&fused, qkv, w.conv, sp, rms_eps as f32)?;
    }
    if let Some(sp) = long.as_ref() {
        super::cuda::delta_net_prefill_scan(&fused, sp)?;
    }
    Ok(Some(o))
}

/// The mixer as tensor ops — the arithmetic the fused kernels are
/// parity-locked to, for CPU, reference dtypes and any other head geometry.
fn mix_spans_by_ops<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let (t, _) = p.qkv.dims2()?;
    let (h_k, h_v, d) = (dims.n_k_heads, dims.n_v_heads, dims.head_dim);
    let (qkv, w) = (&p.qkv, c);
    let key_dim = dims.key_dim();

    // Causal conv, then SiLU.
    //
    // Per sequence, because the tail is: token `i` of a sequence convolves with
    // that sequence's own previous tokens, and its first `K−1` come from the
    // carried tail rather than from whatever sequence precedes it in the buffer.
    let conved = conv_silu_spans(qkv, w.conv, seqs)?; // [T, conv_dim]

    // The l2-normed Q | K stack.
    //
    // Q and K are adjacent in the conv output and take the same per-head l2
    // norm, so they are normed as one `[T, 2·H_k, d]` tensor and split after.
    // The norm reduces over the last axis, so treating the two projections as
    // one stack of `2·H_k` heads is the same arithmetic on each — and it halves
    // a four-launch reduction that runs twice per DeltaNet layer.
    let qk = l2_norm(
        &conved.narrow(1, 0, 2 * key_dim)?.reshape((t, 2 * h_k, d))?,
        rms_eps,
    )?;

    let beta = candle_nn::ops::sigmoid(&p.beta_lin)?; // [T, H_v]

    // g = a ⊙ softplus(α + dt_bias): per-head log-decay, ≤ 0 since a < 0.
    let g_log = softplus(&p.alpha_lin.broadcast_add(c.dt_bias)?)?.broadcast_mul(c.a)?;

    let q = qk.narrow(1, 0, h_k)?;
    let k = qk.narrow(1, h_k, h_k)?;
    let v = conved
        .narrow(1, 2 * key_dim, dims.value_dim())?
        .reshape((t, h_v, d))?;
    let group = h_v / h_k;
    // **V head `j` reads K head `j % h_k`, not `j / group`.**
    //
    // ggml broadcasts this with `ggml_repeat`, which *tiles* its source along
    // the head axis: `[k0, k1, …]` becomes `[k0, k1, …, k0, k1, …]`. The
    // blocked reading — each K head repeated `group` times in place — is the
    // other natural way to write it and is what an `expand` over an inserted
    // axis gives you, so it is easy to arrive at and impossible to tell apart
    // on any model where `h_v == h_k`.
    //
    // Which is exactly what happened: the 0.8B the reference was validated
    // against on llama.cpp has 16 K heads and 16 V heads, so `group == 1` and
    // both conventions are the identity. The 9B is 16/32 and the 32B family
    // wider still — there the two differ, and the symptom is not a crash but a
    // model that is *almost* right: it still knows that Paris is the capital
    // of France and still writes English, it just cannot hold an instruction.
    let repeat = |x: &LiveTensor<'w>| -> Result<LiveTensor<'w>> {
        // [T, H_k, d] → [T, group, H_k, d] → [T, H_v, d], so index
        // `g · h_k + i` reads K head `i` — the tiling ggml performs.
        x.unsqueeze(1)?
            .expand((t, group, h_k, d))?
            .reshape((t, h_v, d))
    };
    let (q, k) = (repeat(&q)?, repeat(&k)?);

    // Read scale, exactly as `build_delta_net_*` applies it to `q` before the
    // rule. This looks like a no-op — `o = S q` is linear in `q`, and the
    // gated RMSNorm downstream is scale-invariant — but it is not, because
    // that norm has an epsilon floor:
    //
    //   rms(o/√d) = (o/√d) / √(mean(o²)/d + ε) = o / √(mean(o²) + d·ε)
    //
    // so dropping the scale is equivalent to shrinking the norm's epsilon by
    // `d` (128×). `o` here is `β(q·k)v`, small enough that the floor is a
    // live term rather than a guard, and the difference is large enough to
    // move the argmax. Omitting this produced fluent-looking but wrong text
    // that passed every structural check.
    let q = (q * (1.0 / (d as f64).sqrt()))?;

    // The chunked parallel-scan form: identical numbers to the sequential
    // rule (locked by test at every chunk width), one state carry per chunk
    // instead of one per token. A single-token decode degenerates to one
    // 1-wide chunk, so one code path serves prefill and decode alike.
    //
    // Per sequence, because `S` is: this is the other place a row depends on
    // the rows before it *within its own sequence*. Each call reads its slice of
    // the five operands and writes its slice of one shared output.
    let lay = TimelineLayout::of(dims);
    let mut o_parts: Vec<LiveTensor<'w>> = Vec::with_capacity(seqs.len());
    for s in seqs.iter_mut() {
        // Free when the span is the whole buffer: a row range of a contiguous
        // tensor is contiguous, so `contiguous` hands the view straight back.
        let rows = |x: &LiveTensor<'w>| -> Result<LiveTensor<'w>> {
            x.narrow(0, s.start, s.len)?.contiguous()
        };
        match &s.write {
            // A short span reads the state and writes its timeline: the rows'
            // `G` and `v_new` against the entering state, the per-K-head key
            // (normed, unscaled) and the raw conv input — what the kernels
            // write, by the same algebra.
            SpanWrite::Timeline(tl) => {
                let (o, g_cs, v_new) = delta_span_rows(
                    &s.state.s,
                    &rows(&q)?,
                    &rows(&k)?,
                    &rows(&v)?,
                    &rows(&g_log)?,
                    &rows(&beta)?,
                )?;
                let k_heads = qk.narrow(1, h_k, h_k)?.narrow(0, s.start, s.len)?;
                lay.write_rows(tl, &g_cs, &k_heads, &v_new, &rows(qkv)?)?;
                o_parts.push(o);
            }
            // A long span advances into its write buffers — the prefill
            // scratch, which `commit_wave` exchanges in — and leaves the live
            // state as it entered. On the single-span entry points the two are
            // one buffer, and the advance is in place.
            SpanWrite::State(out) => {
                if !out.s.same_storage(&s.state.s) {
                    out.s.slice_set(&s.state.s, 0, 0)?;
                }
                let mut advanced = out.s.clone();
                o_parts.push(delta_chunked(
                    &mut advanced,
                    &rows(&q)?,
                    &rows(&k)?,
                    &rows(&v)?,
                    &rows(&g_log)?,
                    &rows(&beta)?,
                    DELTA_CHUNK,
                )?);
            }
        }
    }
    let o = LiveTensor::cat(&o_parts, 0)?;
    drop(o_parts);

    // Gated per-head norm, then flatten for the output projection.
    let z_heads = p.z.reshape((t, h_v, d))?;
    let z_gate = match zgate {
        ZGate::Silu => silu(&z_heads)?,
        ZGate::Sigmoid => candle_nn::ops::sigmoid(&z_heads)?,
    };
    let gated = rms_norm_per_head(&o, w.norm, rms_eps)?.mul(&z_gate)?;
    gated.reshape((t, dims.value_dim()))
}

/// One DeltaNet layer over a `[T, hidden]` segment (single sequence), from a
/// carried state. Returns `([T, hidden], new_state)`.
///
/// This is the reference forward: every op in F32, exactly the llama.cpp
/// graph order. The production path projects with quantized kernels and then
/// calls the same [`delta_net_mix`].
pub fn delta_net_layer_forward(
    x: &Tensor,
    w: &DeltaNetWeights,
    dims: &DeltaNetDims,
    state: &mut DeltaNetState,
    rms_eps: f64,
    zgate: ZGate,
) -> Result<Tensor> {
    let p = DeltaNetProjections {
        qkv: x.matmul(&w.wqkv.t()?)?,
        z: x.matmul(&w.wz.t()?)?,
        beta_lin: x.matmul(&w.w_beta.t()?)?,
        alpha_lin: x.matmul(&w.w_alpha.t()?)?,
    };
    let c = DeltaNetConstants {
        dt_bias: &w.dt_bias,
        a: &w.a,
        conv: &w.conv,
        norm: &w.norm,
    };
    let gated = delta_net_mix(&p, &c, dims, state, rms_eps, zgate)?;
    gated.matmul(&w.w_out.t()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn dev() -> Device {
        Device::Cpu
    }

    /// Deterministic pseudo-random fill (no external RNG dependency; tests
    /// must be reproducible byte-for-byte).
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
                // Map the top bits to (-0.5, 0.5).
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

    #[test]
    fn a_written_value_reads_back_along_its_key() {
        // One head, no decay (g = 0), β = 1, unit key: the delta rule stores v
        // exactly, and reading with q = k returns it exactly.
        let dev = dev();
        let d = 4usize;
        let state = Tensor::zeros((1, d, d), DType::F32, &dev).unwrap();
        let k = Tensor::from_vec(vec![1f32, 0., 0., 0.], (1, d), &dev).unwrap();
        let v = Tensor::from_vec(vec![0.25f32, -1.5, 3., 0.5], (1, d), &dev).unwrap();
        let g = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let beta = Tensor::ones((1,), DType::F32, &dev).unwrap();
        let (o, s) = delta_step(&state, &k, &k, &v, &g, &beta).unwrap();
        assert_close(&o, &v, 1e-6, "read-back");
        // Writing the SAME (k, v) again is a no-op: the state already
        // predicts v for k, so the correction is zero.
        let (_, s2) = delta_step(&s, &k, &k, &v, &g, &beta).unwrap();
        assert_close(&s, &s2, 1e-6, "idempotent rewrite");
    }

    #[test]
    fn beta_interpolates_and_decay_forgets() {
        let dev = dev();
        let d = 4usize;
        let state = Tensor::zeros((1, d, d), DType::F32, &dev).unwrap();
        let k = Tensor::from_vec(vec![0f32, 1., 0., 0.], (1, d), &dev).unwrap();
        let v = Tensor::from_vec(vec![2f32, 2., 2., 2.], (1, d), &dev).unwrap();
        let g0 = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let half = Tensor::full(0.5f32, (1,), &dev).unwrap();
        // β = 0.5 writes half the correction.
        let (o, s) = delta_step(&state, &k, &k, &v, &g0, &half).unwrap();
        let expect = v.affine(0.5, 0.).unwrap();
        assert_close(&o, &expect, 1e-6, "half write");
        // A strongly negative gate wipes the state before the next write:
        // reading k after decay-only (β = 0 write of an orthogonal key) sees ~0.
        let g_forget = Tensor::full(-40f32, (1,), &dev).unwrap();
        let beta0 = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let k_other = Tensor::from_vec(vec![1f32, 0., 0., 0.], (1, d), &dev).unwrap();
        let (_, s_forgot) = delta_step(&s, &k_other, &k_other, &v, &g_forget, &beta0).unwrap();
        let norm = s_forgot
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(norm < 1e-6, "state survived a -40 log-decay: {norm}");
    }

    #[test]
    fn segmented_recurrence_equals_one_shot() {
        // THE sealing/resume property: running [0..a) then [a..T) from the
        // carried state must equal one run over [0..T).
        let dev = dev();
        let (t, h, d) = (7usize, 3usize, 5usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 1, &dev), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 2, &dev), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 3, &dev);
        // Log-decays in a realistic range: −softplus-ish magnitudes.
        let g = lcg_tensor(&[t, h], 4, &dev).abs().unwrap().neg().unwrap();
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 5, &dev)).unwrap();
        let s0 = Tensor::zeros((h, d, d), DType::F32, &dev).unwrap();

        let (o_full, s_full) = delta_recurrence(s0.clone(), &q, &k, &v, &g, &beta).unwrap();

        let a = 3usize;
        let slice = |x: &Tensor, from: usize, len: usize| x.narrow(0, from, len).unwrap();
        let (o1, s_mid) = delta_recurrence(
            s0,
            &slice(&q, 0, a),
            &slice(&k, 0, a),
            &slice(&v, 0, a),
            &slice(&g, 0, a),
            &slice(&beta, 0, a),
        )
        .unwrap();
        let (o2, s_end) = delta_recurrence(
            s_mid,
            &slice(&q, a, t - a),
            &slice(&k, a, t - a),
            &slice(&v, a, t - a),
            &slice(&g, a, t - a),
            &slice(&beta, a, t - a),
        )
        .unwrap();
        let o_seg = Tensor::cat(&[o1, o2], 0).unwrap();
        assert_close(&o_full, &o_seg, 1e-5, "segmented outputs");
        assert_close(&s_full, &s_end, 1e-5, "segmented final state");
    }

    /// The chunked prefill form must match the sequential rule exactly —
    /// outputs and final state — across chunk widths that tile the sequence
    /// evenly, leave a short tail, exceed it, and degenerate to 1 (where the
    /// chunked algebra IS the sequential rule). Nonzero entering state, so the
    /// inter-chunk read path is exercised from the first token.
    #[test]
    fn chunked_matches_sequential_for_all_chunk_widths() {
        let dev = dev();
        let (t, h, d) = (13usize, 3usize, 5usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 61, &dev), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 62, &dev), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 63, &dev);
        let g = lcg_tensor(&[t, h], 64, &dev).abs().unwrap().neg().unwrap();
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 65, &dev)).unwrap();
        // Nonzero entering state: seed it with a couple of sequential steps.
        let s_seed = Tensor::zeros((h, d, d), DType::F32, &dev).unwrap();
        let pre_q = l2_norm(&lcg_tensor(&[2, h, d], 66, &dev), 1e-6).unwrap();
        let pre_k = l2_norm(&lcg_tensor(&[2, h, d], 67, &dev), 1e-6).unwrap();
        let pre_v = lcg_tensor(&[2, h, d], 68, &dev);
        let pre_g = lcg_tensor(&[2, h], 69, &dev).abs().unwrap().neg().unwrap();
        let pre_b = candle_nn::ops::sigmoid(&lcg_tensor(&[2, h], 70, &dev)).unwrap();
        let (_, s0) = delta_recurrence(s_seed, &pre_q, &pre_k, &pre_v, &pre_g, &pre_b).unwrap();

        let (o_ref, s_ref) = delta_recurrence(s0.clone(), &q, &k, &v, &g, &beta).unwrap();

        for chunk in [1usize, 4, 5, 13, 64] {
            let mut s_ch = s0.copy().unwrap();
            let o_ch = delta_chunked(&mut s_ch, &q, &k, &v, &g, &beta, chunk).unwrap();
            assert_close(&o_ref, &o_ch, 3e-5, &format!("chunk={chunk} outputs"));
            assert_close(&s_ref, &s_ch, 3e-5, &format!("chunk={chunk} state"));
        }
    }

    /// Strong decays must not poison the chunk.
    ///
    /// The decay matrix is `exp(G[i] − G[j])` over **all** `(i, j)`, masked to
    /// the lower triangle afterwards. The discarded half's exponent grows
    /// positive with the distance between tokens, so once a chunk accumulates
    /// enough decay it overflows to `+inf`, the mask turns that into `NaN`, and
    /// `unit_lower_inverse` spreads it over the whole chunk.
    ///
    /// It is content-dependent, which is why the width sweep above — random
    /// decays around −0.5 over 13 tokens — never saw it, while real prompt
    /// embeddings produced a non-finite logit row at token 23 of one prompt and
    /// 27 of another. Here the decays are large enough that an unclamped
    /// exponent exceeds the f32 range within one chunk, and the sequential rule
    /// (which computes one step at a time and cannot overflow this way) is the
    /// reference for what the answer should be.
    #[test]
    fn strong_decay_stays_finite_and_matches_sequential() {
        let dev = dev();
        let (t, h, d) = (48usize, 2usize, 4usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 81, &dev), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 82, &dev), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 83, &dev);
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 84, &dev)).unwrap();
        // ~−4 per token: 48 tokens accumulate ≈ −190, so the upper triangle's
        // mirror exponent passes f32's ~88 overflow point well inside one
        // 64-wide chunk. The published models reach the same place by depth
        // rather than by magnitude.
        let g = lcg_tensor(&[t, h], 85, &dev)
            .abs()
            .unwrap()
            .affine(1.0, 3.5)
            .unwrap()
            .neg()
            .unwrap();

        let s0 = Tensor::zeros((h, d, d), DType::F32, &dev).unwrap();
        let (o_ref, s_ref) = delta_recurrence(s0.clone(), &q, &k, &v, &g, &beta).unwrap();

        let finite = |x: &Tensor, what: &str| {
            let m = x
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(m.is_finite(), "{what} went non-finite under strong decay");
        };
        for chunk in [8usize, 16, 64] {
            let mut s_ch = s0.copy().unwrap();
            let o_ch = delta_chunked(&mut s_ch, &q, &k, &v, &g, &beta, chunk).unwrap();
            finite(&o_ch, &format!("chunk={chunk} output"));
            finite(&s_ch, &format!("chunk={chunk} state"));
            assert_close(&o_ref, &o_ch, 3e-5, &format!("chunk={chunk} outputs"));
            assert_close(&s_ref, &s_ch, 3e-5, &format!("chunk={chunk} state"));
        }
    }

    /// **V head `j` must read K head `j % h_k`.**
    ///
    /// ggml broadcasts the DeltaNet GQA with `ggml_repeat`, which tiles rather
    /// than blocks. The blocked reading (`j / group`) is the other natural way
    /// to write it, produces the same shapes, and is the identity whenever
    /// `h_v == h_k` — which is true of the 0.8B the reference was first
    /// validated on, so nothing caught it there. On the 9B (16 K heads, 32 V)
    /// it cost the model its ability to follow an instruction while leaving it
    /// fluent, which is the hardest possible symptom to attribute.
    ///
    /// Checked by construction rather than through a forward: give each K head
    /// a distinguishable constant and assert which head each V head received.
    #[test]
    fn delta_net_gqa_broadcast_tiles_rather_than_blocks() {
        let dev = dev();
        let (t, h_k, group, d) = (3usize, 4usize, 2usize, 5usize);
        let h_v = h_k * group;

        // K head i is filled with the value `i`.
        let vals: Vec<f32> = (0..t * h_k * d)
            .map(|idx| ((idx / d) % h_k) as f32)
            .collect();
        let x = Tensor::from_vec(vals, (t, h_k, d), &dev).unwrap();

        // The broadcast under test, written exactly as `delta_net_mix` writes it.
        let repeated = x
            .unsqueeze(1)
            .unwrap()
            .expand((t, group, h_k, d))
            .unwrap()
            .reshape((t, h_v, d))
            .unwrap();

        let got: Vec<f32> = repeated.flatten_all().unwrap().to_vec1().unwrap();
        for tt in 0..t {
            for j in 0..h_v {
                let want = (j % h_k) as f32;
                let at = (tt * h_v + j) * d;
                assert_eq!(
                    got[at],
                    want,
                    "V head {j} read K head {} (t={tt}); ggml tiles, so it must \
                     read {}",
                    got[at] as usize,
                    j % h_k
                );
            }
        }
        // And the fixture must actually distinguish the two conventions —
        // they coincide entirely when `h_v == h_k`, and even on a real
        // broadcast they agree on individual heads, so compare the whole maps.
        let tiled: Vec<usize> = (0..h_v).map(|j| j % h_k).collect();
        let blocked: Vec<usize> = (0..h_v).map(|j| j / group).collect();
        assert_ne!(
            tiled, blocked,
            "fixture does not distinguish tiled from blocked, so it would pass \
             under either"
        );
    }

    /// Geometry for the span tests: small, but with a real conv kernel width
    /// and more than one head, so a wrong slice cannot coincide with a right one.
    fn span_dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 2,
            conv_kernel: 3,
        }
    }

    /// Projections over `t` rows, deterministic.
    fn span_projections(
        t: usize,
        dims: &DeltaNetDims,
        dev: &Device,
    ) -> DeltaNetProjections<'static> {
        DeltaNetProjections {
            qkv: lcg_tensor(&[t, dims.conv_dim()], 91, dev),
            z: lcg_tensor(&[t, dims.value_dim()], 92, dev),
            beta_lin: lcg_tensor(&[t, dims.n_v_heads], 93, dev),
            alpha_lin: lcg_tensor(&[t, dims.n_v_heads], 94, dev),
        }
    }

    fn span_constants(dims: &DeltaNetDims, dev: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
        let dt_bias = lcg_tensor(&[dims.n_v_heads], 95, dev);
        // `a` is `−exp(A_log)` in the checkpoint, so strictly negative.
        let a = lcg_tensor(&[dims.n_v_heads], 96, dev)
            .abs()
            .unwrap()
            .neg()
            .unwrap();
        let conv = lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], 97, dev);
        let norm = lcg_tensor(&[dims.head_dim], 98, dev).abs().unwrap();
        (dt_bias, a, conv, norm)
    }

    /// **Batching several sequences into one call must equal running each
    /// alone.**
    ///
    /// This is the property the whole staged mixer rests on. Everything except
    /// the conv tail and the recurrence is row-wise, which is what lets the
    /// projections and the thirty-odd elementwise ops run once for a whole wave
    /// — but only if the two carried steps still see exactly their own rows and
    /// their own state. A span off by one row, a state shared between
    /// sequences, or a concatenation in the wrong order all leak one
    /// conversation into another, and the symptom is a plausible wrong answer
    /// rather than a crash.
    ///
    /// Two sequences of *different* lengths, so an off-by-one cannot alias —
    /// one long enough to write a state, one short enough to write a timeline.
    #[test]
    fn spans_equal_running_each_sequence_alone() {
        let dev = dev();
        let dims = span_dims();
        let (len_a, len_b) = (TIMELINE_ROWS + 3, 3usize);
        let total = len_a + len_b;
        let eps = 1e-6;

        let p = span_projections(total, &dims, &dev);
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };

        // Each sequence alone, from its own fresh state, over its own rows.
        let mut wants = Vec::new();
        let mut solo_states = Vec::new();
        for (start, len) in [(0usize, len_a), (len_a, len_b)] {
            let rows = |x: &Tensor| x.narrow(0, start, len).unwrap().contiguous().unwrap();
            let ps = DeltaNetProjections {
                qkv: rows(&p.qkv),
                z: rows(&p.z),
                beta_lin: rows(&p.beta_lin),
                alpha_lin: rows(&p.alpha_lin),
            };
            let mut st = DeltaNetState::zeros(&dims, &dev).unwrap();
            wants.push(delta_net_mix(&ps, &c, &dims, &mut st, eps, ZGate::Silu).unwrap());
            solo_states.push(st);
        }
        let want = Tensor::cat(&wants, 0).unwrap();

        // Both together, one call, one state each.
        let mut st_a = DeltaNetState::zeros(&dims, &dev).unwrap();
        let mut st_b = DeltaNetState::zeros(&dims, &dev).unwrap();
        let got = {
            let write_a = solo_write(&st_a, len_a, &dims).unwrap();
            let write_b = solo_write(&st_b, len_b, &dims).unwrap();
            let mut seqs = [
                DeltaNetSeq {
                    start: 0,
                    len: len_a,
                    state: &mut st_a,
                    write: write_a,
                },
                DeltaNetSeq {
                    start: len_a,
                    len: len_b,
                    state: &mut st_b,
                    write: write_b,
                },
            ];
            let mixed = delta_net_mix_spans(&p, &c, &dims, &mut seqs, eps, None, ZGate::Silu, None)
                .unwrap();
            for s in seqs.iter_mut() {
                finish_solo(s.state, &s.write, s.len, &dims).unwrap();
            }
            mixed
        };

        assert_close(&got, &want, 1e-5, "batched vs solo activations");
        // And the carried state: the activations could match while the state
        // diverged, and the next token would be wrong instead of this one.
        for (i, (batched, solo)) in [&st_a, &st_b].iter().zip(&solo_states).enumerate() {
            assert_close(&batched.s, &solo.s, 1e-5, &format!("sequence {i} state"));
            assert_close(
                &batched.conv_tail,
                &solo.conv_tail,
                1e-5,
                &format!("sequence {i} conv tail"),
            );
        }
        // The fixture must actually distinguish the sequences — two that ended
        // in the same state would pass under a shared-state bug.
        let cross = st_a
            .s
            .sub(&st_b.s)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(cross > 1e-6, "the two sequences ended in identical state");
    }

    /// **Folding the first `m` rows of a block's timeline equals having only
    /// ever run those `m` rows.**
    ///
    /// This is the property speculative decode rests on, and it is the reason
    /// the hybrid can rewind at all: `S` is a running sum with no suffix to
    /// subtract, so a partial accept is a choice of how many of the block's
    /// rows to fold into the state the block entered with — which nothing
    /// wrote, because a short span writes only its timeline.
    ///
    /// The oracle is a clean run of the same `m` rows into a fresh copy of the
    /// entering state. If the two disagree, a rejected draft has left its mark
    /// on the recurrence, and the model would answer as though it had read
    /// tokens the KV no longer holds. Run for every `m`, so the prefix
    /// stability of `G`, the solve and `v_new` is held at every row.
    #[test]
    fn folding_a_prefix_equals_running_only_that_prefix() {
        let dev = dev();
        let dims = span_dims();
        let block = 6usize;
        let p = span_projections(block, &dims, &dev);
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };

        // A non-trivial entering state, so a fold that silently started from
        // zeros could not pass.
        let entering = DeltaNetState {
            s: lcg_tensor(&[dims.n_v_heads, dims.head_dim, dims.head_dim], 77, &dev),
            conv_tail: lcg_tensor(&[dims.conv_dim(), dims.conv_kernel - 1], 78, &dev),
        };

        // The wave: the whole block, reading `entering` and writing a timeline.
        let mut read = entering.snapshot().unwrap();
        let tl = TimelineLayout::of(&dims).alloc(&dev).unwrap();
        {
            let mut seqs = [DeltaNetSeq {
                start: 0,
                len: block,
                state: &mut read,
                write: SpanWrite::Timeline(tl.clone()),
            }];
            delta_net_mix_spans(&p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None).unwrap();
        }
        assert_close(
            &read.s,
            &entering.s,
            0.0,
            "a short span left its state as it was",
        );

        for kept in 1..=block {
            let mut folded = entering.snapshot().unwrap();
            fold_by_ops(&mut folded, &tl, kept, &dims).unwrap();

            // The oracle: the same `kept` rows, nothing else, same entering state.
            let mut want = entering.snapshot().unwrap();
            let po = DeltaNetProjections {
                qkv: p.qkv.narrow(0, 0, kept).unwrap().contiguous().unwrap(),
                z: p.z.narrow(0, 0, kept).unwrap().contiguous().unwrap(),
                beta_lin: p.beta_lin.narrow(0, 0, kept).unwrap().contiguous().unwrap(),
                alpha_lin: p
                    .alpha_lin
                    .narrow(0, 0, kept)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            };
            delta_net_mix(&po, &c, &dims, &mut want, 1e-6, ZGate::Silu).unwrap();

            assert_close(&folded.s, &want.s, 1e-5, &format!("state folded at {kept}"));
            assert_close(
                &folded.conv_tail,
                &want.conv_tail,
                0.0,
                &format!("conv tail folded at {kept}"),
            );
        }

        // And the whole block really does advance further than a prefix, so
        // the assertions above are not comparing a rewind to a no-op.
        let mut whole = entering.snapshot().unwrap();
        fold_by_ops(&mut whole, &tl, block, &dims).unwrap();
        let mut one = entering.snapshot().unwrap();
        fold_by_ops(&mut one, &tl, 1, &dims).unwrap();
        let gap = whole
            .s
            .sub(&one.s)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            gap > 1e-5,
            "the {block}-row block ended in the same state as its first row — the \
             fixture cannot tell a rewind from a no-op"
        );
    }

    /// A span's write must fit its length — a short span handed a state, or a
    /// long one a timeline, is refused before anything runs.
    #[test]
    fn a_span_write_that_does_not_fit_its_length_is_refused() {
        let dev = dev();
        let dims = span_dims();
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let mut st = DeltaNetState::zeros(&dims, &dev).unwrap();
        let short = span_projections(3, &dims, &dev);
        let out = st.solo_out().unwrap();
        let mut seqs = [DeltaNetSeq {
            start: 0,
            len: 3,
            state: &mut st,
            write: SpanWrite::State(out),
        }];
        let err = delta_net_mix_spans(&short, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None)
            .unwrap_err();
        assert!(err.to_string().contains("timeline"), "{err}");

        let long = span_projections(TIMELINE_ROWS + 1, &dims, &dev);
        let tl = TimelineLayout::of(&dims).alloc(&dev).unwrap();
        let mut seqs = [DeltaNetSeq {
            start: 0,
            len: TIMELINE_ROWS + 1,
            state: &mut st,
            write: SpanWrite::Timeline(tl),
        }];
        let err = delta_net_mix_spans(&long, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None)
            .unwrap_err();
        assert!(err.to_string().contains("state"), "{err}");
    }

    /// Spans that do not tile the buffer must be refused, not silently mixed.
    ///
    /// Every one of these reads some sequence's rows into another's recurrence.
    /// The caller derives spans from the wave's packing, so a mismatch here is a
    /// packing bug — and it has to surface as one rather than as a model that
    /// answers slightly wrong.
    #[test]
    fn spans_must_tile_the_buffer() {
        let dev = dev();
        let dims = span_dims();
        let total = 8usize;
        let p = span_projections(total, &dims, &dev);
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let mut st_a = DeltaNetState::zeros(&dims, &dev).unwrap();
        let mut st_b = DeltaNetState::zeros(&dims, &dev).unwrap();

        let run = |spans: [(usize, usize); 2], sa: &mut DeltaNetState, sb: &mut DeltaNetState| {
            let write_a = solo_write(sa, spans[0].1, &dims).unwrap();
            let write_b = solo_write(sb, spans[1].1, &dims).unwrap();
            let mut seqs = [
                DeltaNetSeq {
                    start: spans[0].0,
                    len: spans[0].1,
                    state: sa,
                    write: write_a,
                },
                DeltaNetSeq {
                    start: spans[1].0,
                    len: spans[1].1,
                    state: sb,
                    write: write_b,
                },
            ];
            delta_net_mix_spans(&p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None).map(|_| ())
        };

        // A gap: rows 5..6 belong to nobody.
        let err = run([(0, 5), (6, 2)], &mut st_a, &mut st_b).unwrap_err();
        assert!(err.to_string().contains("tile"), "{err}");
        // An overlap: row 4 is fed to both sequences.
        let err = run([(0, 5), (4, 4)], &mut st_a, &mut st_b).unwrap_err();
        assert!(err.to_string().contains("tile"), "{err}");
        // Short: the last rows of the buffer are never mixed.
        let err = run([(0, 5), (5, 2)], &mut st_a, &mut st_b).unwrap_err();
        assert!(err.to_string().contains("cover"), "{err}");
        // The tiling one is accepted, so the checks above are rejecting the
        // defect rather than the shape of the call.
        assert!(run([(0, 5), (5, 3)], &mut st_a, &mut st_b).is_ok());
    }

    #[test]
    fn unit_lower_inverse_inverts() {
        let dev = dev();
        let (h, c) = (2usize, 6usize);
        let strict = lower_tri_mask(c, true, &dev).unwrap();
        let a = lcg_tensor(&[h, c, c], 71, &dev)
            .broadcast_mul(&strict.unsqueeze(0).unwrap())
            .unwrap();
        let t_inv = unit_lower_inverse(&a).unwrap();
        // (I + A) · T = I.
        let eye_vals: Vec<f32> = (0..c * c)
            .map(|i| if i % (c + 1) == 0 { 1f32 } else { 0f32 })
            .collect();
        let eye = Tensor::from_vec(eye_vals, (1, c, c), &dev)
            .unwrap()
            .expand((h, c, c))
            .unwrap();
        let prod = eye.add(&a).unwrap().matmul(&t_inv).unwrap();
        assert_close(&prod, &eye.contiguous().unwrap(), 1e-5, "inverse");
    }

    #[test]
    fn causal_conv_matches_hand_computation_and_carries_tail() {
        let dev = dev();
        // 1 channel, kernel [1, 2, 3], input [10, 20, 30, 40].
        let x = Tensor::from_vec(vec![10f32, 20., 30., 40.], (1, 4), &dev).unwrap();
        let kern = Tensor::from_vec(vec![1f32, 2., 3.], (1, 3), &dev).unwrap();
        let tail0 = Tensor::zeros((1, 2), DType::F32, &dev).unwrap();
        let (y, tail) = causal_conv1d(&x, &kern, &tail0).unwrap();
        // y[t] = 1·x[t−2] + 2·x[t−1] + 3·x[t]  (zeros left of the start)
        let expect = Tensor::from_vec(vec![30f32, 80., 140., 200.], (1, 4), &dev).unwrap();
        assert_close(&y, &expect, 1e-6, "conv values");
        let expect_tail = Tensor::from_vec(vec![30f32, 40.], (1, 2), &dev).unwrap();
        assert_close(&tail, &expect_tail, 1e-6, "conv tail");

        // Split invocation equals one-shot — same property as the recurrence.
        let (y1, tmid) = causal_conv1d(&x.narrow(1, 0, 2).unwrap(), &kern, &tail0).unwrap();
        let (y2, tend) = causal_conv1d(&x.narrow(1, 2, 2).unwrap(), &kern, &tmid).unwrap();
        let y_seg = Tensor::cat(&[y1, y2], 1).unwrap();
        assert_close(&y, &y_seg, 1e-6, "segmented conv outputs");
        assert_close(&tail, &tend, 1e-6, "segmented conv tail");
    }

    #[test]
    fn l2_norm_produces_unit_rows_and_respects_eps_floor() {
        let dev = dev();
        let x = lcg_tensor(&[4, 8], 7, &dev);
        let n = l2_norm(&x, 1e-6).unwrap();
        let sums = n.sqr().unwrap().sum(candle::D::Minus1).unwrap();
        let ones = Tensor::ones((4,), DType::F32, &dev).unwrap();
        assert_close(&sums, &ones, 1e-5, "unit rows");
        // A zero row divides by sqrt(eps), not by zero.
        let z = Tensor::zeros((1, 8), DType::F32, &dev).unwrap();
        let nz = l2_norm(&z, 1e-6).unwrap();
        let m = nz.abs().unwrap().flatten_all().unwrap().max(0).unwrap();
        assert_eq!(m.to_scalar::<f32>().unwrap(), 0.0);
    }

    fn tiny_dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        }
    }

    fn tiny_weights(dims: &DeltaNetDims, hidden: usize, dev: &Device) -> DeltaNetWeights {
        let scale = |t: Tensor| t.affine(0.2, 0.).unwrap();
        DeltaNetWeights {
            wqkv: scale(lcg_tensor(&[dims.conv_dim(), hidden], 11, dev)),
            wz: scale(lcg_tensor(&[dims.value_dim(), hidden], 12, dev)),
            w_beta: scale(lcg_tensor(&[dims.n_v_heads, hidden], 13, dev)),
            w_alpha: scale(lcg_tensor(&[dims.n_v_heads, hidden], 14, dev)),
            dt_bias: scale(lcg_tensor(&[dims.n_v_heads], 15, dev)),
            // a = −exp(A_log): strictly negative.
            a: lcg_tensor(&[dims.n_v_heads], 16, dev)
                .abs()
                .unwrap()
                .affine(-1.0, -0.05)
                .unwrap(),
            conv: scale(lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], 17, dev)),
            norm: lcg_tensor(&[dims.head_dim], 18, dev)
                .affine(0.3, 1.0)
                .unwrap(),
            w_out: scale(lcg_tensor(&[hidden, dims.value_dim()], 19, dev)),
        }
    }

    #[test]
    fn layer_forward_segments_equal_one_shot() {
        // The whole layer — projections, conv tail, recurrence state, gating —
        // must carry across a segment boundary with zero drift. This is the
        // sealing/resume contract at layer granularity.
        let dev = dev();
        let dims = tiny_dims();
        let hidden = 6usize;
        let w = tiny_weights(&dims, hidden, &dev);
        let t = 9usize;
        let x = lcg_tensor(&[t, hidden], 21, &dev);

        let mut s_full = DeltaNetState::zeros(&dims, &dev).unwrap();
        let y_full =
            delta_net_layer_forward(&x, &w, &dims, &mut s_full, 1e-6, ZGate::Silu).unwrap();

        // The same tokens in two calls, through one state buffer — which is
        // the property sealing and resume rest on, and now also the property
        // that the buffer is advanced rather than replaced.
        let a = 4usize;
        let mut s_seg = DeltaNetState::zeros(&dims, &dev).unwrap();
        let y1 = delta_net_layer_forward(
            &x.narrow(0, 0, a).unwrap(),
            &w,
            &dims,
            &mut s_seg,
            1e-6,
            ZGate::Silu,
        )
        .unwrap();
        let y2 = delta_net_layer_forward(
            &x.narrow(0, a, t - a).unwrap(),
            &w,
            &dims,
            &mut s_seg,
            1e-6,
            ZGate::Silu,
        )
        .unwrap();
        let y_seg = Tensor::cat(&[y1, y2], 0).unwrap();
        assert_close(&y_full, &y_seg, 1e-5, "layer segmented outputs");
        assert_close(&s_full.s, &s_seg.s, 1e-5, "layer segmented state");
        assert_close(
            &s_full.conv_tail,
            &s_seg.conv_tail,
            1e-6,
            "layer segmented conv tail",
        );
        // Sanity: the output is finite and non-degenerate.
        let m = y_full
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m.is_finite() && m > 0.0);
    }

    /// The `1/√d_k` read scale is load-bearing and must not be "optimised
    /// away" as cancelled by the gated RMSNorm downstream.
    ///
    /// It looks removable: `o = S q` is linear in `q` and RMSNorm is
    /// scale-invariant. It is not, because that norm has an epsilon floor —
    /// `rms(o/√d) = o/√(mean(o²) + d·ε)` — so dropping the scale silently
    /// shrinks epsilon by `d`. On the real 0.8B checkpoint that moved the
    /// argmax and turned fluent output into word salad while every
    /// structural test still passed.
    ///
    /// From a zero entering state a single token collapses the recurrence to
    /// `o = β·(k·q̂)·v` with `q̂ = q/√d`, so the whole layer has a closed form
    /// that this test spells out independently. Dropping the read scale
    /// changes `o` by `√d`, which survives the gated norm through its
    /// epsilon term and fails the comparison.
    #[test]
    fn single_token_layer_matches_the_closed_form_with_the_read_scale() {
        let dev = dev();
        let dims = tiny_dims();
        let hidden = 6usize;
        let (d, h_k, h_v) = (dims.head_dim, dims.n_k_heads, dims.n_v_heads);
        let (key_dim, val_dim) = (dims.key_dim(), dims.value_dim());
        let w = tiny_weights(&dims, hidden, &dev);
        let x = lcg_tensor(&[1, hidden], 77, &dev);
        let eps = 1e-6f64;

        let mut s = DeltaNetState::zeros(&dims, &dev).unwrap();
        let got = delta_net_layer_forward(&x, &w, &dims, &mut s, eps, ZGate::Silu).unwrap();

        // The conv tail is zero at sequence start, so only the newest kernel
        // tap contributes and the conv reduces to a per-channel scale.
        let qkv = x.matmul(&w.wqkv.t().unwrap()).unwrap();
        let tap = w
            .conv
            .narrow(1, dims.conv_kernel - 1, 1)
            .unwrap()
            .t()
            .unwrap();
        let conv = silu(&qkv.broadcast_mul(&tap).unwrap()).unwrap();

        let head = |t: &Tensor, off: usize, n: usize, heads: usize| {
            t.narrow(1, off, n).unwrap().reshape((1, heads, d)).unwrap()
        };
        let q = l2_norm(&head(&conv, 0, key_dim, h_k), eps).unwrap();
        let k = l2_norm(&head(&conv, key_dim, key_dim, h_k), eps).unwrap();
        let v = head(&conv, 2 * key_dim, val_dim, h_v);

        // The GQA broadcast, tiled as ggml does it — see
        // `delta_net_gqa_broadcast_tiles_rather_than_blocks`. Restated here to
        // keep the closed form independent of `delta_net_mix`'s *arithmetic*,
        // but it has to agree with it on this convention or the two are
        // describing different models.
        let group = h_v / h_k;
        let rep = |t: &Tensor| {
            t.unsqueeze(1)
                .unwrap()
                .expand((1, group, h_k, d))
                .unwrap()
                .reshape((1, h_v, d))
                .unwrap()
        };
        let (q, k) = (rep(&q), rep(&k));

        // o = β · (k · q/√d) · v — the g gate cannot appear, since it only
        // decays a state that is still zero.
        let q_hat = (&q * (1.0 / (d as f64).sqrt())).unwrap();
        let kq = k.mul(&q_hat).unwrap().sum(candle::D::Minus1).unwrap();
        let beta = candle_nn::ops::sigmoid(&x.matmul(&w.w_beta.t().unwrap()).unwrap()).unwrap();
        let o = v
            .broadcast_mul(&beta.mul(&kq).unwrap().unsqueeze(2).unwrap())
            .unwrap();

        let z = x
            .matmul(&w.wz.t().unwrap())
            .unwrap()
            .reshape((1, h_v, d))
            .unwrap();
        let want = rms_norm_per_head(&o, &w.norm, eps)
            .unwrap()
            .mul(&silu(&z).unwrap())
            .unwrap()
            .reshape((1, val_dim))
            .unwrap()
            .matmul(&w.w_out.t().unwrap())
            .unwrap();

        assert_close(&got, &want, 1e-6, "single-token layer closed form");

        // Guard the guard: the fixture must actually be sensitive to the
        // scale, or the assertion above proves nothing.
        let unscaled = v
            .broadcast_mul(
                &beta
                    .mul(&k.mul(&q).unwrap().sum(candle::D::Minus1).unwrap())
                    .unwrap()
                    .unsqueeze(2)
                    .unwrap(),
            )
            .unwrap();
        let drift = rms_norm_per_head(&o, &w.norm, eps)
            .unwrap()
            .sub(&rms_norm_per_head(&unscaled, &w.norm, eps).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            drift > 1e-6,
            "gated norm is insensitive to the read scale in this fixture \
             ({drift}) — it cannot detect the scale going missing"
        );
    }

    #[test]
    fn softplus_is_stable_at_extremes() {
        let dev = dev();
        let x = Tensor::from_vec(vec![-100f32, -1., 0., 1., 100.], (5,), &dev).unwrap();
        let y = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!(y[0].abs() < 1e-6, "softplus(-100) ≈ 0, got {}", y[0]);
        assert!((y[2] - std::f32::consts::LN_2).abs() < 1e-6);
        assert!(
            (y[4] - 100.0).abs() < 1e-4,
            "softplus(100) ≈ 100, got {}",
            y[4]
        );
        assert!((y[1] - 0.313_261_7).abs() < 1e-5);
        assert!((y[3] - 1.313_261_7).abs() < 1e-5);
    }
}
