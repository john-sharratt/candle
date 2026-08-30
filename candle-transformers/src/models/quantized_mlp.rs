//! The gated (SwiGLU) FFN shared by the quantized models.
//!
//! `down(silu(gate(x)) * up(x))` is the same three matmuls in every model in
//! this crate, but the details that make it fast are not obvious and were
//! duplicated per model — and, in `quantized_qwen3`, twice within one model
//! (once in the plain loader and again in the int8 one, differing only by
//! the numeric mode). Those details are:
//!
//! * **gate/up fusion.** On CUDA, two same-shaped quantized weights are
//!   concatenated row-wise into one so the pair costs a single launch. Only
//!   for genuinely quantized dtypes — concatenating F32/F16/BF16 rows buys
//!   nothing and the fused path would just add a split.
//! * **where the dtype coercion goes.** The fused output is cast in place
//!   *before* it is split, because the owned contiguous buffer casts without
//!   allocating while the two aliasing narrows would each force a fallback
//!   copy.
//! * **why the intermediate is not the activation dtype.** MLP
//!   intermediates can exceed F16's ~65504 range, so silu/mul/down run in
//!   `out_dtype` (BF16 where activations are F16).

use candle::quantized::cuda::DynamicActs;
use candle::quantized::{GgmlDType, Int8Mode, QTensor};
use candle::{DType, LiveTensor, Module, Result, Tensor};
use candle_nn::Activation;

use crate::models::quantized_matmul::{QMatMul, WeightResidency};

/// The three (or two, when gate+up are fused) projections of a gated FFN.
#[derive(Debug, Clone)]
pub struct QuantizedMlp {
    /// The row-concatenated `[gate | up]` weight, when fusion applied.
    gate_up_proj: Option<QMatMul>,
    /// Separate projections, when it did not. Exactly one of these two
    /// representations is populated.
    gate_proj: Option<QMatMul>,
    up_proj: Option<QMatMul>,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl QuantizedMlp {
    /// Build from the three checkpoint weights, repacking each for `mode`.
    ///
    /// `gate` and `up` are fused when the device and dtypes allow; `down` is
    /// always its own projection (its shape does not match the other two).
    pub fn from_weights(
        gate_w: QTensor,
        up_w: QTensor,
        down_w: QTensor,
        mode: Int8Mode,
    ) -> Result<Self> {
        Self::from_weights_in(
            gate_w,
            up_w,
            down_w,
            mode,
            WeightResidency::Span,
            (None, None, None),
        )
    }

    /// [`Self::from_weights`], placing each repacked projection in `residency`.
    ///
    /// The layer-streaming pack build materialises an FFN only to read it back,
    /// so its three weights must not claim dense-block ground they will never
    /// give up. See [`WeightResidency`].
    /// `narrow` forces every projection's KO twin, for an FFN that stays resident on a card
    /// that cannot hold the model — see `QMatMul::from_qtensor_narrowed`. `None` lets the mode
    /// pick, which is every ordinary load.
    ///
    /// It has to be threaded here rather than left to the caller's `Loader`: this is the one
    /// path to a projection that does **not** go through `Loader::proj`, and the FFN is the
    /// larger part of any block — so a narrowing policy that missed it would report itself as
    /// applied while leaving roughly three quarters of the weight at full width.
    /// `narrow` is **per projection**, in `(gate, up, down)` order.
    ///
    /// One width for all three was the first shape and it silently narrowed the
    /// wrong tensors. The argument for it — that `ffn_gate`/`ffn_up` are already
    /// at or below the down-projection's target, so naming it leaves them
    /// untouched — holds only for the checkpoints where that happens to be true.
    /// Narrowing is a floor that *shrinks*, so wherever gate/up pick a wider twin
    /// than the target (a Q4_K gate at `Int8Mode::Performance` against a `Q3_KO`
    /// down) they are narrowed too — while `streaming_twin` returns `None` for
    /// those roles and `an_unnamed_role_is_untouched` asserts they are untouched.
    /// The loader and the schedule its tests describe disagreed, and the tests
    /// were the ones telling the truth.
    pub fn from_weights_in(
        gate_w: QTensor,
        up_w: QTensor,
        down_w: QTensor,
        mode: Int8Mode,
        residency: WeightResidency,
        narrow: (Option<GgmlDType>, Option<GgmlDType>, Option<GgmlDType>),
    ) -> Result<Self> {
        // One place decides, so the fused and unfused arms cannot disagree.
        let build = |qt: QTensor, narrow: Option<GgmlDType>| -> Result<QMatMul> {
            match narrow {
                Some(n)
                    if qt
                        .dtype()
                        .to_ko(mode)
                        .is_ok_and(|p| n.bits_per_weight() < p.bits_per_weight()) =>
                {
                    QMatMul::from_qtensor_narrowed(qt, mode, residency, n)
                }
                _ => QMatMul::from_qtensor_in(qt, mode, residency),
            }
        };
        let fusable = gate_w.device().is_cuda()
            && gate_w.dtype() == up_w.dtype()
            && !matches!(
                gate_w.dtype(),
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
            );

        let (gate_up_proj, gate_proj, up_proj) = if fusable {
            #[cfg(feature = "cuda")]
            {
                let (gate_n, gate_k) = gate_w.shape().dims2()?;
                let (up_n, up_k) = up_w.shape().dims2()?;
                if gate_n != up_n || gate_k != up_k {
                    candle::bail!(
                        "cannot fuse ffn_gate/ffn_up due to shape mismatch: \
                         gate=({gate_n}, {gate_k}) up=({up_n}, {up_k})"
                    );
                }
                // The fused pair is one weight, so it takes one target. Gate and
                // up are the same dtype (a fusion precondition above), so their
                // schedule entries agree; taking gate's is taking both.
                let fused = QTensor::concat_rows_cuda(&[&gate_w, &up_w])?;
                (Some(build(fused, narrow.0)?), None, None)
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle::bail!("fused gate+up requires the cuda feature");
            }
        } else {
            (
                None,
                Some(build(gate_w, narrow.0)?),
                Some(build(up_w, narrow.1)?),
            )
        };

        Ok(Self {
            gate_up_proj,
            gate_proj,
            up_proj,
            down_proj: build(down_w, narrow.2)?,
            act_fn: Activation::Silu,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    /// Build from projections that are **already** KO twins.
    ///
    /// How a layer-streaming slot's FFN is assembled: the weights are views
    /// over the slot the layer was uploaded into, built by
    /// `layer_stream::build_layer_view`, so there is nothing left to repack and
    /// nothing to fuse — the fusion happened once, before the pack was written,
    /// and the record holds the fused weight.
    ///
    /// `gate_up` carries the fused `[2·intermediate, hidden]` form and `gate`
    /// / `up` the unfused pair; exactly one of the two must be supplied, which
    /// is the same invariant [`Self::from_weights`] establishes and this
    /// checks rather than assumes.
    pub fn from_repacked(
        gate_up: Option<QMatMul>,
        gate: Option<QMatMul>,
        up: Option<QMatMul>,
        down: QMatMul,
    ) -> Result<Self> {
        let fused = gate_up.is_some();
        let split = gate.is_some() && up.is_some();
        if fused == split {
            candle::bail!(
                "QuantizedMlp::from_repacked: supply either the fused gate_up or the \
                 gate/up pair, not {}",
                if fused { "both" } else { "neither" }
            );
        }
        Ok(Self {
            gate_up_proj: gate_up,
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
            act_fn: Activation::Silu,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    /// The fused `[gate|up]` weight, when fusion applied.
    ///
    /// Borrowed: the layer-streaming pack build reads these in place, and a
    /// `QMatMul` clone would copy the weight rather than alias it.
    pub fn fused_gate_up(&self) -> Option<&QMatMul> {
        self.gate_up_proj.as_ref()
    }

    /// The unfused gate weight, when fusion did not apply.
    pub fn split_gate(&self) -> Option<&QMatMul> {
        self.gate_proj.as_ref()
    }

    /// The unfused up weight, when fusion did not apply.
    pub fn split_up(&self) -> Option<&QMatMul> {
        self.up_proj.as_ref()
    }

    /// The down projection, which every form has.
    pub fn down(&self) -> &QMatMul {
        &self.down_proj
    }

    /// The numeric mode these projections were repacked for, read off the
    /// down projection (every projection in the FFN shares one mode).
    pub fn int8mode(&self) -> Int8Mode {
        self.down_proj.int8mode()
    }

    /// `(hidden, intermediate)`, recovered from the down projection's own
    /// weight — its shape is `[hidden, intermediate]`.
    ///
    /// Reading the loaded weight rather than carrying a copy of the config
    /// means the transient plan cannot drift from the shapes the kernels
    /// actually see.
    pub fn hidden_and_intermediate(&self) -> Result<(usize, usize)> {
        let dims = self.down_proj.weight_dims();
        match dims.as_slice() {
            [hidden, intermediate] => Ok((*hidden, *intermediate)),
            other => candle::bail!("ffn_down should be 2-D, got {other:?}"),
        }
    }

    /// Width of each half of a fused `[gate | up]` output.
    fn fused_half(out_dim: usize) -> Result<usize> {
        if !out_dim.is_multiple_of(2) {
            candle::bail!("unexpected fused gate+up output dim {out_dim} (not even)");
        }
        Ok(out_dim / 2)
    }

    /// The plain FP path.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (gate, up) = if let Some(w) = &self.gate_up_proj {
            let gu = w.forward(x)?;
            let (_, _, out_dim) = gu.dims3()?;
            let half = Self::fused_half(out_dim)?;
            (gu.narrow(2, 0, half)?, gu.narrow(2, half, half)?)
        } else {
            let (gate_proj, up_proj) = self.separate()?;
            (gate_proj.forward(x)?, up_proj.forward(x)?)
        };
        let gated = (&self.act_fn.forward_live(&gate)? * &up)?;
        self.down_proj.forward(&gated)
    }

    fn separate(&self) -> Result<(&QMatMul, &QMatMul)> {
        let gate = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing gate_proj".into()))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing up_proj".into()))?;
        Ok((gate, up))
    }

    /// B3 consumer: gate/up over a producer-prepared (fused ln2) activation,
    /// shared across both projections so ln2→q8a128 is not paid twice.
    ///
    /// `work_dtype` is the width the SwiGLU intermediates are carried in — wide
    /// enough for their range, which is why an F16 activation runs this in BF16.
    /// `out_dtype` is what the residual stream wants back, and the down
    /// projection **stores** it: narrowing afterwards would be a full-tensor
    /// pass per layer per wave to undo a widening the intermediates needed and
    /// the result does not (hot-path invariant 1).
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic<'w>(
        &self,
        acts: &DynamicActs<'w>,
        work_dtype: DType,
        out_dtype: DType,
    ) -> Result<LiveTensor<'w>> {
        let (mut gate, mut up) = if let Some(w) = &self.gate_up_proj {
            let mut gu = w.forward_dynamic(acts.as_dynamic(), work_dtype)?;
            let (_, _, out_dim) = gu.dims3()?;
            let half = Self::fused_half(out_dim)?;
            // Coerce the fused output ONCE, in place, before splitting: `gu`
            // is owned + contiguous here so the cast is allocation-free,
            // whereas casting the two aliasing narrows separately forces two
            // fallback allocations.
            gu.to_dtype_mut(work_dtype)?;
            (gu.narrow(2, 0, half)?, gu.narrow(2, half, half)?)
        } else {
            let (gate_proj, up_proj) = self.separate()?;
            (
                gate_proj.forward_dynamic(acts.as_dynamic(), work_dtype)?,
                up_proj.forward_dynamic(acts.as_dynamic(), work_dtype)?,
            )
        };
        // Run silu/mul in `work_dtype`: the Float path returns the activation
        // dtype (F16), but MLP intermediates can exceed F16's ~65504 range. The
        // fused path already coerced `gu` above and the int8 path already
        // returns `work_dtype`, so these are no-ops except on the
        // separate-weight Float path.
        gate.to_dtype_mut(work_dtype)?;
        up.to_dtype_mut(work_dtype)?;
        let gated = (&self.act_fn.forward_live(&gate)? * &up)?;
        self.down_proj.forward_live_as(&gated, out_dtype)
    }
}
