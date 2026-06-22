//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::QTensor;
use candle::{Module, Result, Tensor};

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
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps, span })
    }

    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();

        let weight = if x_dtype == self.weight.dtype() {
            self.weight.clone()
        } else {
            self.weight.to_dtype(x_dtype)?
        };
        candle_nn::ops::rms_norm(x, &weight, self.eps as f32)
    }
}

impl RmsNorm {
    /// RMSNorm as a producer epilogue: returns the matmul-ready [`DynamicActs`]. For an int8
    /// `mode` it runs the fused `rms_norm_q8a128` kernel — normalize + quantize in one launch,
    /// no FP store/re-read — and returns `Int8(q8a128)`; for [`Int8Mode::Off`] it runs the plain
    /// FP `rms_norm` and returns `Float`. The downstream matmul consumes the result via
    /// `QMatMul::forward_dynamic`, so for int8 the activation never materializes in FP. CUDA only.
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic(
        &self,
        x: &Tensor,
        mode: candle::quantized::Int8Mode,
    ) -> Result<candle::quantized::cuda::DynamicActs> {
        use candle::quantized::cuda::DynamicActs;
        if !mode.is_int8() {
            return Ok(DynamicActs::Float(self.forward(x)?));
        }
        let _enter = self.span.enter();
        let weight = if x.dtype() == self.weight.dtype() {
            self.weight.clone()
        } else {
            self.weight.to_dtype(x.dtype())?
        };
        let dev = match x.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("RmsNorm::forward_dynamic(int8) requires a CUDA tensor"),
        };
        let op = candle::quantized::cuda::rms_norm_q8a128(x, &weight, self.eps as f32, &dev)?;
        Ok(DynamicActs::Int8(op))
    }
}
