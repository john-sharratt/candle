use candle::quantized::QTensor;
use candle::{DType, Module, Result, Tensor};

/// Traced quantized matmul wrapper.
///
/// Key behavior:
/// - Adds a tracing span around matmul.
/// - On CUDA: Uses K/128 format with embedded scales for efficient matmul.
/// - On CPU/Metal: Uses standard GGML kernels (no repacking).
/// - For BF16/F16 inputs, uses the dequantize+matmul fast paths and preserves input dtype.
/// - For rank-3 BF16/F16 inputs, flattens (B,S,K) -> (B*S,K) to avoid broadcasting weights.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
    /// Whether GEMX path is safe to use for this tensor (CUDA with K/128 format)
    use_gemx: bool,
}

impl QMatMul {
    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");

        // Check if this tensor supports GEMX repacking (CUDA only)
        if qtensor.supports_gemx_repacking() {
            // CUDA path: Repack to K/128 format with embedded scales
            let repacked = qtensor.repack_gemx()?;
            let inner = candle::quantized::QMatMul::from_qtensor(repacked)?;

            Ok(Self {
                inner,
                span,
                use_gemx: true,
            })
        } else {
            // CPU/Metal path: Use original tensor
            let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;

            Ok(Self {
                inner,
                span,
                use_gemx: false,
            })
        }
    }

    pub fn from_weights(ws: std::sync::Arc<QTensor>) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");

        // Check if this tensor supports GEMX repacking (CUDA only)
        if ws.supports_gemx_repacking() {
            // CUDA path: Repack to K/128 format with embedded scales
            let repacked = ws.repack_gemx()?;
            let inner = candle::quantized::QMatMul::from_qtensor(repacked)?;

            Ok(Self {
                inner,
                span,
                use_gemx: true,
            })
        } else {
            // CPU/Metal path: Use original tensor via Arc
            let inner = candle::quantized::QMatMul::from_arc(ws)?;

            Ok(Self {
                inner,
                span,
                use_gemx: false,
            })
        }
    }

    /// Create from an already-repacked QTensor.
    ///
    /// This is useful for testing when you want to control the repack separately.
    pub fn from_qtensor_repacked(repacked: QTensor) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        let inner = candle::quantized::QMatMul::from_qtensor(repacked)?;

        Ok(Self {
            inner,
            span,
            use_gemx: true,
        })
    }

    pub fn inner(&self) -> &candle::quantized::QMatMul {
        &self.inner
    }

    /// Dequantize the underlying tensor.
    /// This is primarily for testing/validation.
    pub fn dequantize(&self) -> Result<Tensor> {
        match &self.inner {
            candle::quantized::QMatMul::QTensor(qt) => qt.dequantize(&qt.device()),
            candle::quantized::QMatMul::Tensor(t) => Ok(t.clone()),
            candle::quantized::QMatMul::TensorF16(t) => t.to_dtype(DType::F32),
        }
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let in_dtype = xs.dtype();
        let (xs2, reshape_back, _m) = if xs.rank() == 3 {
            let (b, s, k) = xs.dims3()?;
            (xs.reshape((b * s, k))?, Some((b, s)), b * s)
        } else {
            let m = xs.dim(0).unwrap_or(0);
            (xs.clone(), None, m)
        };

        let use_native_gemx = self.use_gemx
            && matches!(self.inner, candle::quantized::QMatMul::QTensor(_))
            && [DType::F32, DType::BF16, DType::F16, DType::F8E4M3].contains(&in_dtype);

        let out2 = if use_native_gemx {
            // K/128 format has embedded scales
            self.inner.forward_via_gemx(&xs2)?
        } else {
            // Fall back to standard quantized matmul path for correctness.
            // Quantized CUDA kernels expect F32 inputs, so cast and restore dtype.
            let xs_f32 = xs2.to_dtype(DType::F32)?;
            let out_f32 = self.inner.forward(&xs_f32)?;
            out_f32.to_dtype(in_dtype)?
        };

        if let Some((b, s)) = reshape_back {
            let n = out2.dim(1)?;
            out2.reshape((b, s, n))
        } else {
            Ok(out2)
        }
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}
