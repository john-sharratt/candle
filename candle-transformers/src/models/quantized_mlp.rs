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
#[cfg(feature = "cuda")]
use candle::quantized::cuda::{silu_mul_q8a128, DynamicTensor};
use candle::quantized::{GgmlDType, Int8Mode, QTensor};
#[cfg(feature = "cuda")]
use candle::Device;
use candle::{DType, LiveTensor, Module, Result, Tensor};
use candle_nn::Activation;

use crate::models::lora::{adapt, LayerLora};
use crate::models::quantized_matmul::{QMatMul, WeightResidency};
#[cfg(feature = "cuda")]
use crate::models::qwen35::quantized_attention::lora_input;

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
        self.forward_dynamic_adapted(acts, work_dtype, out_dtype, LayerLora::default())
    }

    /// The MLP with its LoRA pairs, which is the same computation with three
    /// optional rank-`r` corrections folded into the projections that carry
    /// them.
    ///
    /// This is the *only* implementation — [`Self::forward_dynamic`] is it with
    /// an empty [`LayerLora`] — so an unadapted MLP and an adapted one cannot
    /// drift apart. An absent pair costs one null check.
    ///
    /// The adapter's input for gate and up is the post-norm activation, and for
    /// down it is the SwiGLU result. `down`'s is always float — the SwiGLU output
    /// is a real tensor whatever the layer's numeric mode — while gate/up's
    /// exists only as q8a128 on the int8 path, and is reconstructed from those
    /// blocks by [`lora_input`]. The layer's numeric mode is not changed by the
    /// presence of an adapter; see [`Qwen35AttentionLayer::int8mode`].
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic_adapted<'w>(
        &self,
        acts: &DynamicActs<'w>,
        work_dtype: DType,
        out_dtype: DType,
        lora: LayerLora<'_>,
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
        // Both adapters add to the raw projections, before SwiGLU — the point
        // PEFT trained them against. The fused-weight path above split one
        // matmul into these two halves, so a fused base and a separate-weight
        // base adapt identically.
        //
        // Resolved once for both, after the projections so `gate` can name the
        // device and width — on the int8 path this reconstructs the operand from
        // its q8a128 blocks, and doing it twice would double that.
        if lora.gate.is_some() || lora.up.is_some() {
            let x = lora_input(acts, &gate, "quantized MLP gate/up")?;
            gate = adapt(lora.gate, gate, &x)?;
            up = adapt(lora.up, up, &x)?;
        }
        // Unadapted: the back half on its own. An adapter on gate or up alone turns
        // that half into a fresh dense tensor whose rows no longer step with the
        // other half's view of the fused output, and `down`'s adapter reads the
        // float result, so an adapted MLP keeps the eager tail below.
        if lora.gate.is_none() && lora.up.is_none() && lora.down.is_none() {
            return self.forward_from_gate_up(&gate, &up, out_dtype);
        }
        let gated = (&self.act_fn.forward_live(&gate)? * &up)?;
        let out = self.down_proj.forward_live_as(&gated, out_dtype)?;
        // `down`'s adapter reads the SwiGLU result, not the layer input — it is
        // the operand of the projection it adapts, exactly as for the other two.
        match lora.down {
            Some(_) => adapt(lora.down, out, &gated),
            None => Ok(out),
        }
    }

    /// The MLP's back half — `down(act(gate) · up)` — from gate and up projections the
    /// caller already has, in `work_dtype`: the two halves of this MLP's own fused
    /// projection, or views of a launch that projection shared with other projections of
    /// the same activation. `down` stores `out_dtype`.
    ///
    /// **The SwiGLU emits the down projection's operand itself** when that projection runs
    /// int8: one launch for `silu(gate) · up` and its quantize, reading the halves where
    /// they were written, through their row stride. Its arithmetic is the eager chain
    /// below, bit for bit, at both working widths — at BF16 that is `silu(gate)` rounded,
    /// then the product rounded, exactly the two stores the chain makes — then the one
    /// q8a128 tile emitter, so the bytes are the ones `down` would quantize from `gated`.
    /// The KV calibration rows were derived on that arithmetic; a single rounding of the
    /// F32 product, though more precise, moved the 0.8B's top rungs across their edge.
    #[cfg(feature = "cuda")]
    pub fn forward_from_gate_up<'w>(
        &self,
        gate: &LiveTensor<'w>,
        up: &LiveTensor<'w>,
        out_dtype: DType,
    ) -> Result<LiveTensor<'w>> {
        if self.down_proj.int8mode().is_int8() && matches!(self.act_fn, Activation::Silu) {
            if let Device::Cuda(dev) = gate.device() {
                // The Σx convention `down` reads, as its own quantize would write it.
                let sum_scale = self.down_proj.sum_scale();
                let op = silu_mul_q8a128(gate, up, dev, gate.cuda_backing(), sum_scale)?;
                return self
                    .down_proj
                    .forward_dynamic(DynamicTensor::Int8(&op), out_dtype);
            }
        }
        let gated = (&self.act_fn.forward_live(gate)? * up)?;
        self.down_proj.forward_live_as(&gated, out_dtype)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial as gpu_guard;
    use candle::quantized::cuda::to_dynamic;
    use candle::quantized::SumScale;
    use candle_nn::ops::silu;

    fn lcg(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
            })
            .collect();
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    /// The int8 MLP's SwiGLU emitting the down projection's operand is the eager
    /// `silu(gate) · up` chain followed by the down projection quantizing its own
    /// input, bit for bit — with gate and up read as the two halves of the fused
    /// projection rather than compacted. At both intermediate widths (F32, and
    /// the BF16 production runs, where the chain's two separate roundings are
    /// what has to agree) and both `Σx` conventions (the operand carries
    /// whichever one `down` reads).
    #[test]
    fn the_fused_swiglu_operand_is_the_eager_chain_then_quantize_bit_for_bit() {
        let _gpu = gpu_guard();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let mode = Int8Mode::auto(&dev);
        assert!(
            mode.is_int8(),
            "the fused SwiGLU is the int8 path; this card must run it"
        );
        let (hidden, inter) = (512usize, 256usize);
        let ko = |t: Tensor| {
            QMatMul::from_qtensor_with_mode(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap(), mode)
                .unwrap()
        };
        let gate_up = ko(lcg(&[2 * inter, hidden], 61, 0.1, &dev));
        for sum_scale in [SumScale::Raw, SumScale::ByAmax] {
            let down = ko(lcg(&[hidden, inter], 62, 0.1, &dev)).with_sum_scale(sum_scale);
            let mlp = QuantizedMlp::from_repacked(Some(gate_up.clone()), None, None, down.clone())
                .unwrap();
            for work in [DType::F32, DType::BF16] {
                for t in [1usize, 3, 16] {
                    let x = lcg(&[1, t, hidden], 63 + t as u64, 2.0, &dev);
                    let acts = to_dynamic(&x, mode, cuda, SumScale::Raw).unwrap();
                    let got = mlp.forward_dynamic(&acts, work, DType::F32).unwrap();

                    let gu = gate_up.forward_dynamic(acts.as_dynamic(), work).unwrap();
                    let gate = gu.narrow(2, 0, inter).unwrap();
                    let up = gu.narrow(2, inter, inter).unwrap();
                    let eager = (silu(&gate).unwrap() * &up).unwrap();
                    let want = down.forward_live_as(&eager, DType::F32).unwrap();

                    let bits =
                        |a: &LiveTensor<'_>| a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert_eq!(bits(&got), bits(&want), "{sum_scale:?} {work:?} {t} tokens");
                }
            }
        }
    }
}
