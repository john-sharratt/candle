//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::{Int8Mode, QTensor};
#[cfg(feature = "cuda")]
use candle::wave_provenance::WaveTicket;
use candle::{DType, LiveTensor, Module, Result, Tensor};
use std::sync::Arc;

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

    /// Declare how this layer's activation stores its per-128 `Σx`, forwarding
    /// to the inner weight — see [`candle::quantized::SumScale`].
    ///
    /// A model whose activations can overflow f16 on a block sum applies this to
    /// every projection it builds; the default is the raw convention, so a model
    /// that says nothing is unaffected.
    pub fn with_sum_scale(mut self, sum_scale: candle::quantized::SumScale) -> Self {
        self.weight = self.weight.with_sum_scale(sum_scale);
        self
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

/// [`linear_b`], with the weight repacked for `mode` and the bias held at the
/// width the model's activations run at.
///
/// At an int8 mode the projection becomes its KO twin and runs the q8a128
/// tensor-core matmul; at `Int8Mode::Off` this is `linear_b` exactly. The bias is
/// dequantised either way — it is one vector added after the matmul, not
/// something the kernel reads — but it is *also* narrowed to `dtype` here,
/// because the matmul emits the activation's width and `broadcast_add` does not
/// convert: a bias left at f32 over a bf16 model is a dtype error at the first
/// forward, and converting it per call would be a full-width pass per layer per
/// step to fix something that is decided once at load.
pub fn linear_b_mode(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    mode: Int8Mode,
    dtype: DType,
    vb: VarBuilder,
) -> Result<Linear> {
    let bias = if bias {
        Some(
            vb.get(out_dim, "bias")?
                .dequantize(vb.device())?
                .to_dtype(dtype)?,
        )
    } else {
        None
    };
    let weight = QMatMul::new_with_mode(in_dim, out_dim, mode, vb)?;
    Ok(Linear { weight, bias })
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
    /// The weight materialised once in every float width activations can arrive
    /// in — F32, F16, BF16, indexed by [`width_slot`] — when the norm is built.
    ///
    /// **At load, never later.** A session picks its widths when it is created,
    /// and materialising then meant a device allocation after the model's span
    /// had claimed the card: the weights are a few KB, but on a card the expert
    /// zone fills to the last granule the allocation had nowhere to come from,
    /// and a Flash-Next session died at creation with `CUDA_ERROR_OUT_OF_MEMORY`
    /// whenever the pool happened to hold nothing cached from the load. Three
    /// copies of a `[hidden]` vector per norm is the price of never allocating
    /// after load.
    weights: Arc<[Tensor; 3]>,
    eps: f64,
    span: tracing::Span,
    /// How the fused int8 path stores the per-128 `Σx` of the operand it emits —
    /// see [`candle::quantized::SumScale`].
    ///
    /// On the norm rather than at the call site because this IS a producer: the
    /// fused kernel writes the header, and whoever consumes the operand reads it
    /// back through `Q8a128Operand::sum_scale`. [`SumScale::Raw`] is the default
    /// and every language model's choice.
    sum_scale: candle::quantized::SumScale,
}

/// Materialise `src` in `dtype`, taking the fused path when there is one.
///
/// `dequantize_f16` / `dequantize_bf16` are quantized *kernels*: they dispatch on
/// the source's [`candle::quantized::GgmlDType`] and have a case only for genuinely quantized
/// formats. Norm weights are commonly stored unquantized — Qwen3-30B-A3B keeps
/// them F32 — and such a source has no `QType`, so the fused path fails outright
/// rather than falling back.
///
/// So a float-stored source dequantizes to its own dtype and converts, and only a
/// source whose format the device kernel reads
/// ([`candle::quantized::GgmlDType::dequantizes_to_bf16`],
/// which also covers the F16 kernel the BF16 one runs through) takes the fused
/// path. Every other source — float storage, and the formats with no narrow
/// kernel — goes through F32 and converts, which every backend can do: the norm
/// materialises all three widths at load, so a width no session asks for must
/// not be able to fail it. The conversion is the very thing the hot loop must
/// not do, but this runs at load, where one transient over a `[hidden]` vector
/// costs nothing.
fn dequantize_as(src: &QTensor, dtype: DType, device: &candle::Device) -> Result<Tensor> {
    if !src.dtype().dequantizes_to_bf16() {
        return src.dequantize(device)?.to_dtype(dtype);
    }
    match dtype {
        DType::F32 => src.dequantize(device),
        DType::F16 => src.dequantize_f16(device),
        DType::BF16 => src.dequantize_bf16(device),
        other => candle::bail!("RmsNorm: no dequantize path for activation dtype {other:?}"),
    }
}

/// Where `dtype`'s materialised weight sits in [`RmsNorm`]'s `weights`.
fn width_slot(dtype: DType) -> Option<usize> {
    match dtype {
        DType::F32 => Some(0),
        DType::F16 => Some(1),
        DType::BF16 => Some(2),
        _ => None,
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
        let device = src.device();
        let weights = [
            dequantize_as(&src, DType::F32, &device)?,
            dequantize_as(&src, DType::F16, &device)?,
            dequantize_as(&src, DType::BF16, &device)?,
        ];
        Ok(Self {
            weights: Arc::new(weights),
            eps,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
            sum_scale: candle::quantized::SumScale::default(),
        })
    }

    /// Declare the Σx convention this norm's fused int8 emit writes — see
    /// [`candle::quantized::SumScale`]. The default is the raw form.
    pub fn with_sum_scale(mut self, sum_scale: candle::quantized::SumScale) -> Self {
        self.sum_scale = sum_scale;
        self
    }

    /// Confirm the norm can serve activations of `dtype`.
    ///
    /// Called when a session is created. Every float width is materialised at
    /// load (see `weights`), so this allocates nothing and only refuses a width
    /// no norm kernel takes — at session creation rather than inside a wave.
    pub fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        self.weight_for(dtype).map(|_| ())
    }

    /// The weight in the activation dtype.
    ///
    /// The hot-loop guard: a width with no materialised weight is refused
    /// rather than converted, which would allocate and launch per call inside
    /// the wave. Cloning is an `Arc` bump, not a copy.
    fn weight_for(&self, dtype: DType) -> Result<Tensor> {
        match width_slot(dtype) {
            Some(i) => Ok(self.weights[i].clone()),
            None => candle::bail!(
                "RmsNorm: no weight for {dtype:?} activations — the norm is materialised in \
                 F32, F16 and BF16 at load"
            ),
        }
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
        x: &LiveTensor<'w>,
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
        x: &LiveTensor<'w>,
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
        let op = candle::quantized::cuda::rms_norm_q8a128(
            x,
            &weight,
            self.eps as f32,
            &dev,
            root,
            self.sum_scale,
        )?;
        Ok(DynamicActs::Int8(op))
    }
}

#[cfg(test)]
mod rms_norm_tests {
    use super::RmsNorm;
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Device, Tensor};

    fn norm() -> RmsNorm {
        let w = Tensor::new(&[0.5f32, 1.0, 1.5, 2.0], &Device::Cpu).unwrap();
        RmsNorm::from_qtensor(QTensor::quantize(&w, GgmlDType::F32).unwrap(), 1e-6).unwrap()
    }

    /// Every float width is materialised at load, so a session's width is
    /// served without converting anything — the same values in each.
    #[test]
    fn every_float_width_is_ready_at_load() {
        let n = norm();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            n.maybe_change_dtype(dtype).unwrap();
            let w = n.weight_for(dtype).unwrap();
            assert_eq!(w.dtype(), dtype);
            let v: Vec<f32> = w.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
            assert_eq!(v, vec![0.5, 1.0, 1.5, 2.0]);
        }
    }

    /// A quantized-stored norm materialises every width too. The 127 sets the
    /// `Q8_0` block's scale to exactly 1, so it holds these integers exactly and
    /// each width reads them back.
    #[test]
    fn a_quantized_norm_materialises_every_width() {
        let vals: Vec<f32> = (0..32)
            .map(|i| if i == 0 { 127.0 } else { (i % 5) as f32 })
            .collect();
        let w = Tensor::new(vals.as_slice(), &Device::Cpu).unwrap();
        let n =
            RmsNorm::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap(), 1e-6).unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let v: Vec<f32> = n
                .weight_for(dtype)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(v, vals, "{dtype:?}");
        }
    }

    /// A width no norm kernel takes is refused at session creation.
    #[test]
    fn a_non_float_width_is_refused() {
        assert!(norm().maybe_change_dtype(DType::U32).is_err());
    }
}
