//! The wave tensors that every fused DeltaNet kernel reads, validated once per
//! layer and resolved for each span by an offset.
//!
//! The conv output is the only operand buffer. Q, K and V are strided views of
//! it, and V head `h` reads K head `h % h_k`. Q is scaled inside the kernels, so
//! no scaled copy exists. The head geometry is derived from the wave tensors,
//! since spans that share a launch each bring their own state. Each span's state
//! is checked square per head, so a differently shaped state is not read with the
//! cohort's strides.

use super::*;

/// The wave tensors every fused DeltaNet kernel reads through strides, plus
/// the geometry derived from them — validated once per layer, not once per
/// span.
///
/// `conved` is the conv kernels' output `[T, conv_dim]` — post-SiLU, with
/// the leading `2·h_k·d` Q|K columns already l2-normed by the conv epilogue.
/// It is the *only* operand buffer: q, k and v are strided views of it (V
/// head `h` reads K head `h % h_k`; q is scaled by `q_scale` inside the
/// kernels, so no scaled copy of q ever exists). `alpha`/`blin` are the raw
/// gate projections `[T, h_v]` — the gates are computed in-kernel with the
/// reference softplus/sigmoid. `o` is the whole-wave output `[T, h_v·d]`
/// each span's kernels write their own rows of, so a multi-sequence wave
/// needs no concatenation.
pub struct DeltaNetFused<'a, 'w> {
    pub conved: &'a LiveTensor<'w>,
    pub alpha: &'a LiveTensor<'w>,
    pub blin: &'a LiveTensor<'w>,
    pub dt_bias: &'a Tensor,
    pub a: &'a Tensor,
    pub o: &'a LiveTensor<'w>,
    pub q_scale: f32,
}

/// The per-launch pointer/geometry bundle [`DeltaNetFused::resolve`] produces.
pub(super) struct FusedPtrs {
    pub(super) qk_p: u64,
    pub(super) v_p: u64,
    pub(super) alpha_p: u64,
    pub(super) blin_p: u64,
    pub(super) dt_p: u64,
    pub(super) a_p: u64,
    pub(super) o_p: u64,
    pub(super) h_v: usize,
    pub(super) h_k: usize,
    pub(super) d: usize,
    pub(super) conv_dim: usize,
}

impl<'a, 'w> DeltaNetFused<'a, 'w> {
    /// Validate the wave tensors against the state's geometry and resolve the
    /// base addresses for a span starting at `start` — an offset into every
    /// stride, never a copy.
    ///
    /// `h_k` is derived, not passed: the conv row is `[Q|K|V]` with Q and K
    /// `h_k·d` each and V `h_v·d`, so `h_k = (conv_dim/d − h_v) / 2`.
    pub(super) fn resolve_wave(&self) -> Result<FusedPtrs> {
        let (t, conv_dim) = self.conved.dims2()?;
        // Derived from the WAVE tensors rather than from a carried state: the
        // spans that share this launch each bring their own state, so no single
        // one of them can be the source of truth for the geometry they all use.
        let (ta, h_v) = self.alpha.dims2()?;
        if ta != t || h_v == 0 {
            candle::bail!(
                "delta_net cuda: alpha must be [{t}, h_v], got {:?}",
                self.alpha.dims()
            );
        }
        let (to, o_cols) = self.o.dims2()?;
        if to != t || !o_cols.is_multiple_of(h_v) {
            candle::bail!(
                "delta_net cuda: o must be [{t}, h_v·d], got {:?}",
                self.o.dims()
            );
        }
        let d = o_cols / h_v;
        if !conv_dim.is_multiple_of(d) || conv_dim <= h_v * d {
            candle::bail!(
                "delta_net cuda: conv row of {conv_dim} cannot be [Q|K|V] with \
                 h_v = {h_v}, d = {d}"
            );
        }
        let qk_heads = conv_dim / d - h_v;
        if !qk_heads.is_multiple_of(2) {
            candle::bail!(
                "delta_net cuda: conv row of {conv_dim} leaves {qk_heads} Q|K heads, \
                 which do not split into equal Q and K"
            );
        }
        let h_k = qk_heads / 2;
        if !h_v.is_multiple_of(h_k) {
            candle::bail!("delta_net cuda: h_v {h_v} must be a multiple of h_k {h_k}");
        }
        if self.blin.dims2()? != (t, h_v) {
            candle::bail!("delta_net cuda: beta_lin must be [{t}, {h_v}]");
        }
        if self.dt_bias.dims1()? != h_v || self.a.dims1()? != h_v {
            candle::bail!("delta_net cuda: dt_bias/a must be [{h_v}]");
        }
        // Bases of the packed wave buffers. q/k/v are all views of the conv
        // output — Q|K at column 0, V at the trailing block — and each kernel
        // adds its own span's row offset from the table.
        let conved_p = f32_ptr(self.conved, "conved")?;
        let v_off = conv_dim - h_v * d;
        Ok(FusedPtrs {
            qk_p: conved_p,
            v_p: conved_p + v_off as u64 * std::mem::size_of::<f32>() as u64,
            alpha_p: f32_ptr(self.alpha, "alpha")?,
            blin_p: f32_ptr(self.blin, "beta_lin")?,
            dt_p: f32_ptr(self.dt_bias, "dt_bias")?,
            a_p: f32_ptr(self.a, "a")?,
            o_p: f32_ptr(self.o, "o")?,
            h_v,
            h_k,
            d,
            conv_dim,
        })
    }
}

/// The state's shape, checked square per head: the geometry a launch shares is
/// derived from the wave tensors (`DeltaNetFused::resolve_wave`), so each span's
/// own state is checked here — a span carrying a differently shaped state would
/// otherwise be read with the cohort's strides.
pub(super) fn checked_state_dims(s: &Tensor) -> Result<(usize, usize, usize)> {
    let (h_v, d_v, d_k) = s.dims3()?;
    if d_v != d_k {
        candle::bail!("delta_net cuda: state must be square per head, got ({d_v}, {d_k})");
    }
    Ok((h_v, d_v, d_k))
}
