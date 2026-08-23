//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::{GgmlDType, QTensor};
#[cfg(feature = "cuda")]
use candle::wave_provenance::WaveTicket;
use candle::{DType, LiveTensor, Module, Result, Tensor};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, d2)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> Result<Tensor> {
        Ok(self.inner.embeddings_native())
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn from_arc(weight: std::sync::Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias })
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    let bias = vb.get(size, "bias")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new(weight, bias, eps))
}

pub fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new_no_bias(weight, eps))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias: None })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// The quantized weight as it sits in the checkpoint.
    ///
    /// Retained so the norm weight can be **re-materialised** in a different
    /// activation dtype rather than cast from a resident F32 copy. Dequantizing
    /// straight from the source means exactly one materialised weight exists at
    /// a time; a cast keeps the F32 original alive alongside every dtype it has
    /// been asked for, which is a copy of every norm weight per dtype for the
    /// life of the process.
    src: Arc<QTensor>,
    /// `src` dequantized into the dtype activations arrive in.
    ///
    /// Behind a lock because the dtype is chosen when a session is created,
    /// which happens through `&self` — and because a model is shared across
    /// threads. Replaced wholesale by [`RmsNorm::maybe_change_dtype`], never
    /// added to.
    weight: Arc<RwLock<Tensor>>,
    eps: f64,
    span: tracing::Span,
}

/// Materialise `src` in `dtype`, taking the fused path when there is one.
///
/// `dequantize_f16` / `dequantize_bf16` are quantized *kernels*: they dispatch on
/// the source's [`GgmlDType`] and have a case only for genuinely quantized
/// formats. Norm weights are commonly stored unquantized — Qwen3-30B-A3B keeps
/// them F32 — and such a source has no `QType`, so the fused path fails outright
/// rather than falling back.
///
/// So a float-stored source dequantizes to its own dtype and converts, and only a
/// quantized one takes the fused path. The conversion on that branch is the very
/// thing the hot loop must not do, but this runs at session setup, where one
/// transient over a `[hidden]` vector costs nothing.
fn dequantize_as(src: &QTensor, dtype: DType, device: &candle::Device) -> Result<Tensor> {
    if matches!(
        src.dtype(),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
    ) {
        return src.dequantize(device)?.to_dtype(dtype);
    }
    match dtype {
        DType::F32 => src.dequantize(device),
        DType::F16 => src.dequantize_f16(device),
        DType::BF16 => src.dequantize_bf16(device),
        other => candle::bail!("RmsNorm: no dequantize path for activation dtype {other:?}"),
    }
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Self::from_arc(vb.get(size, "weight")?, eps)
    }

    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        Self::from_arc(Arc::new(weight), eps)
    }

    fn from_arc(src: Arc<QTensor>, eps: f64) -> Result<Self> {
        let weight = src.dequantize(&src.device())?;
        Ok(Self {
            src,
            weight: Arc::new(RwLock::new(weight)),
            eps,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }

    /// Re-materialise the weight in the dtype activations will arrive in, if it
    /// is not already.
    ///
    /// **Called when a session is created, never inside a wave.** The norm
    /// kernels need the weight in the activation dtype, and a quantized
    /// checkpoint dequantizes to F32 while inference runs F16 or BF16. Doing the
    /// conversion on demand inside the forward costs one device allocation and
    /// one launch per norm, per layer, per token — and it is invisible in a
    /// profile, surfacing only as a slightly slower forward. That is why
    /// [`Self::weight_for`] refuses a dtype it was not prepared for instead of
    /// quietly converting.
    ///
    /// Idempotent, and re-entrant across dtype switches: the previous weight is
    /// dropped as the new one replaces it, so switching back and forth costs a
    /// reload rather than an accumulating set of copies.
    pub fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        if self.weight.read().unwrap().dtype() == dtype {
            return Ok(());
        }
        let fresh = dequantize_as(&self.src, dtype, &self.src.device())?;
        *self.weight.write().unwrap() = fresh;
        Ok(())
    }

    /// The weight, which must already be in the activation dtype.
    ///
    /// The hot-loop guard. Cloning is an `Arc` bump, not a copy.
    fn weight_for(&self, dtype: DType) -> Result<Tensor> {
        let weight = self.weight.read().unwrap();
        if weight.dtype() != dtype {
            candle::bail!(
                "RmsNorm: weight is {:?} but activations are {dtype:?}. The weight is \
                 materialised by `maybe_change_dtype` when a session is created; converting \
                 it here would allocate and launch per call inside the wave.",
                weight.dtype(),
            )
        }
        Ok(weight.clone())
    }
}

impl RmsNorm {
    /// Normalize an activation that may live on an inference wave.
    ///
    /// The result inherits `'w` from `x`: the norm kernel writes its output into
    /// whichever arena the activation came from, so a wave-scoped input yields a
    /// wave-scoped result. The weight is a model parameter and is always owned,
    /// which is why only `x` carries the lifetime.
    ///
    /// [`Module::forward`] is this at `'static`, where the bound is vacuous.
    pub fn forward_live<'w>(&self, x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();

        let weight = self.weight_for(x_dtype)?;
        candle_nn::ops::rms_norm(x, &weight, self.eps as f32)
    }

    /// [`Self::forward_live`] as the **head** of a wave-scoped chain.
    ///
    /// A layer's first norm reads the residual stream, which lives on the pool
    /// because it crosses layers — so there is no arena for it to inherit, and
    /// this is where the layer names one. Everything computed from the result
    /// follows it there under operand provenance, with no further mention of the
    /// wave; the guard borrow on the result is what stops any of it outliving
    /// the span.
    ///
    /// `wave` of `None` is the ordinary owned allocation, which is the right
    /// answer outside a forward rather than a fallback.
    #[cfg(feature = "cuda")]
    pub fn forward_rooted<'w>(
        &self,
        x: &Tensor,
        wave: Option<&'w candle_nn::kv_cache::WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        self.forward_with_ticket(x, wave.map(|g| g.ticket()))
    }

    /// The rooted FP norm shared by [`Self::forward_rooted`] and the float arm
    /// of [`Self::forward_dynamic`]: the two entry points differ only in where
    /// the provenance ticket comes from (a wave handle vs. a producer's
    /// backing), so both resolve it and land here.
    #[cfg(feature = "cuda")]
    fn forward_with_ticket<'w>(
        &self,
        x: &Tensor,
        root: Option<WaveTicket>,
    ) -> Result<LiveTensor<'w>> {
        let _enter = self.span.enter();
        let weight = self.weight_for(x.dtype())?;
        candle_nn::ops::rms_norm_rooted(x, &weight, self.eps as f32, root)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_live(x)
    }
}

impl RmsNorm {
    /// RMSNorm as a producer epilogue: returns the matmul-ready [`DynamicActs`]. For an int8
    /// `mode` it runs the fused `rms_norm_q8a128` kernel — normalize + quantize in one launch,
    /// no FP store/re-read — and returns `Int8(q8a128)`; for [`Int8Mode::Off`] it runs the plain
    /// FP `rms_norm` and returns `Float`. The downstream matmul consumes the result via
    /// `QMatMul::forward_dynamic`, so for int8 the activation never materializes in FP. CUDA only.
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic<'w>(
        &self,
        x: &Tensor,
        mode: candle::quantized::Int8Mode,
        root: candle::cuda_backend::Backing,
    ) -> Result<candle::quantized::cuda::DynamicActs<'w>> {
        use candle::quantized::cuda::DynamicActs;
        if !mode.is_int8() {
            // `root` seeds this arm too: the FP norm writes into the arena the
            // ticket names, so the float FFN chains onto the wave span instead
            // of running off the pool while the span sits empty beside it.
            let normed = self.forward_with_ticket(x, root.inherit_ticket())?;
            return Ok(DynamicActs::Float(normed));
        }
        let _enter = self.span.enter();
        let weight = self.weight_for(x.dtype())?;
        let dev = match x.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("RmsNorm::forward_dynamic(int8) requires a CUDA tensor"),
        };
        let op = candle::quantized::cuda::rms_norm_q8a128(x, &weight, self.eps as f32, &dev, root)?;
        Ok(DynamicActs::Int8(op))
    }
}
