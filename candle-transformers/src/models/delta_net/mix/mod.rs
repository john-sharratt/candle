//! The Gated DeltaNet mixer: the recurrence that carries state across tokens, the
//! convolution that carries a tail, and the gated norm and projections around
//! them.
//!
//! A layer computes its projections row by row, then runs the causal conv and the
//! delta rule for each sequence, carrying a convolution tail and a recurrent
//! matrix in F32. The fused CUDA path and the tensor-op path do the same
//! arithmetic, and the tensor-op path is the reference that both are locked to. A
//! packed wave buffer holds several sequences, and the spans that tile it say
//! which rows each sequence owns, so batching never mixes one conversation's
//! state into another.
//!
//! - `conv`: the causal depthwise conv with its carried tail, and its SiLU and norm steps.
//! - `fused`: the fused CUDA mixer path and its per-condition fallback tally.
//! - `mix_spans`: the mixer over a wave of several sequences.
//! - `recurrence`: the delta rule, in its sequential reference and chunked prefill forms.
//! - `spans`: sequence spans and the per-layer pointer tables.
//! - `state`: the carried per-sequence recurrent state.
//!
//! - `tests_layer`, `tests_recurrence`, `tests_spans`: tests for the layer, the
//!   recurrence and the span packing.

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

use super::span_arm::SpanHook;
use super::timeline::{fold_by_ops, TimelineLayout, TIMELINE_ROWS};
use super::types::{DeltaNetDims, ZGate};

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
        hook: SpanHook::None,
    }];
    // The single-sequence reference path, off the sweep: no phase is open.
    let mixed = delta_net_mix_spans(p, c, dims, &mut one, rms_eps, None, zgate, None)?;
    let [seq] = one;
    finish_solo(seq.state, &seq.write, t, dims)?;
    Ok(mixed)
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

mod conv;
mod fused;
mod mix_spans;
mod recurrence;
mod spans;
mod state;
#[cfg(test)]
mod tests_layer;
#[cfg(test)]
mod tests_recurrence;
#[cfg(test)]
mod tests_spans;

pub use conv::*;
pub use fused::*;
pub use mix_spans::*;
pub use recurrence::*;
pub use spans::*;
pub use state::*;
