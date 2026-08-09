#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
use candle::quantized::{Int8Mode, QTensor};
use candle::{DType, Module, Result, Tensor};

use crate::models::profile::{pipeline_record, profile_now};

/// Traced quantized matmul wrapper.
///
/// Key behavior:
/// - Adds a tracing span around matmul.
/// - On CUDA, [`Int8Mode::Off`]: Uses K/128 GEMX format with embedded scales for efficient matmul.
/// - On CUDA, an int8 mode: repacks the weight to its KO twin at load and runs the q8a128 int8
///   tensor-core matmul (the weight twin is chosen by the mode; see [`Int8Mode`]).
/// - On CPU/Metal: Uses standard GGML kernels (no repacking).
/// - For BF16/F16 inputs, uses the dequantize+matmul fast paths and preserves input dtype.
/// - For rank-3 BF16/F16 inputs, flattens (B,S,K) -> (B*S,K) to avoid broadcasting weights.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
    /// Whether GEMX path is safe to use for this tensor (CUDA with K/128 format)
    use_gemx: bool,
    /// Numeric mode of this weight: `Off` → FP GEMX/standard path; an int8 mode → the weight is
    /// the KO twin and forward runs the q8a128 int8 tensor-core matmul.
    int8mode: Int8Mode,
}

impl QMatMul {
    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_qtensor_with_mode(qtensor, Int8Mode::Off)
    }

    /// Build from an owned source QTensor, repacking for the given numeric `mode`.
    pub fn from_qtensor_with_mode(qtensor: QTensor, mode: Int8Mode) -> Result<Self> {
        // One source of truth for mode→weight resolution: `from_weights_with_mode`. The only
        // difference is owned vs `Arc` source, so delegate rather than duplicate the branch —
        // that is what let the int8 "every weight becomes a KO twin (never a float weight fed a
        // q8a128)" guarantee silently differ between the two constructors before.
        Self::from_weights_with_mode(std::sync::Arc::new(qtensor), mode)
    }

    pub fn from_weights(ws: std::sync::Arc<QTensor>) -> Result<Self> {
        Self::from_weights_with_mode(ws, Int8Mode::Off)
    }

    /// Build from a shared source QTensor, repacking for the given numeric `mode`.
    pub fn from_weights_with_mode(ws: std::sync::Arc<QTensor>, mode: Int8Mode) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        // The int8 path is CUDA-only; on other backends the mode is ignored (always Off).
        #[cfg(not(feature = "cuda"))]
        let _ = mode;

        // int8 mode (CUDA): every matmul becomes a KO weight, so the q8a128 activations the fused
        // producers emit always pair with a KO twin — a *float* weight must never receive a q8a128
        // (that pairing has no kernel). A gemx-supported quantized source repacks to its KO twin
        // directly; a float source (F32/F16/BF16 — e.g. the F32 router `ffn_gate_inp` or lm_head,
        // which have no quantized GGUF form and would otherwise dequantize to a plain float weight)
        // is first quantized to Q8_0 so it, too, gets a Q8_KO twin. Routing stays near-lossless at
        // 8 bits, and ln2's two consumers — router and experts — are then both int8.
        #[cfg(feature = "cuda")]
        if mode.is_int8() {
            let ws = if ws.supports_gemx_repacking() {
                ws
            } else {
                let f32 = ws.dequantize(&ws.device())?;
                std::sync::Arc::new(QTensor::quantize(&f32, GgmlDType::Q8_0)?)
            };
            let inner = candle::quantized::QMatMul::from_arc(ws)?.repack_for_optimization(mode)?;
            return Ok(Self {
                inner,
                span,
                use_gemx: false,
                int8mode: mode,
            });
        }

        // Off (or non-CUDA / unsupported): FP GEMX K/128 repack, or the standard path via Arc.
        if ws.supports_gemx_repacking() {
            let repacked = ws.repack_gemx()?;
            let inner = candle::quantized::QMatMul::from_qtensor(repacked)?;
            Ok(Self {
                inner,
                span,
                use_gemx: true,
                int8mode: Int8Mode::Off,
            })
        } else {
            let inner = candle::quantized::QMatMul::from_arc(ws)?;
            Ok(Self {
                inner,
                span,
                use_gemx: false,
                int8mode: Int8Mode::Off,
            })
        }
    }

    /// Create from an already-repacked QTensor. The weight may be either the FP GEMX K/128 form
    /// (→ `use_gemx`, `Off`) or a KO twin (→ the int8 tensor-core path); the KO case is how expert
    /// slots are wrapped from the pinned-pool's pre-repacked KO bytes, so they report int8 to
    /// `compute_experts_grouped`. The KO twin is already baked into the dtype.
    pub fn from_qtensor_repacked(repacked: QTensor) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        let is_ko = repacked.dtype().is_ko();
        let inner = candle::quantized::QMatMul::from_qtensor(repacked)?;

        Ok(Self {
            inner,
            span,
            use_gemx: !is_ko,
            int8mode: if is_ko {
                Int8Mode::Precision
            } else {
                Int8Mode::Off
            },
        })
    }

    /// Dimensions of the underlying weight, `[out_features, in_features]`.
    ///
    /// Reads the shape off whichever representation the weight was loaded as,
    /// so a projection's widths can be recovered from the model itself rather
    /// than carried alongside it. That matters for the transient plan: sizing
    /// from the loaded weights cannot drift from what the kernels actually see,
    /// whereas a separately-stored copy of the same number can.
    pub fn weight_dims(&self) -> Vec<usize> {
        match &self.inner {
            candle::quantized::QMatMul::QTensor(qt) => qt.shape().dims().to_vec(),
            candle::quantized::QMatMul::Tensor(t) | candle::quantized::QMatMul::TensorF16(t) => {
                t.dims().to_vec()
            }
        }
    }

    pub fn inner(&self) -> &candle::quantized::QMatMul {
        &self.inner
    }

    /// The numeric mode this weight was built for. `compute_experts_grouped` reads this off the
    /// expert slots to dispatch the grouped float vs int8 path consistently with the dense path.
    pub fn int8mode(&self) -> Int8Mode {
        self.int8mode
    }

    /// Matmul over an activation a producer already prepared as a [`DynamicTensor`]: an `Int8`
    /// operand is pre-quantized q8a128 (emitted by a fused RMSNorm/SwiGLU/attention epilogue) and
    /// goes straight to the KO tensor-core matmul — no standalone quantize launch — which stores
    /// its F32 accumulator directly at `out_dtype`; a `Float` operand runs the standard `forward`
    /// path. This is the consumer half of producer fusion: `quantize once per fan-out` (q/k/v share
    /// one ln1 operand). CUDA only.
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic<'w>(
        &self,
        input: candle::quantized::cuda::DynamicTensor<'_, 'w>,
        out_dtype: DType,
    ) -> Result<candle::LiveTensor<'w>> {
        use candle::quantized::cuda::DynamicTensor;
        // Float activation → the ordinary path (handles Off gemx and any non-int8 weight). It tags
        // its own profile bucket, so don't double-record here.
        if let DynamicTensor::Float(t) = input {
            return self.forward_live(t);
        }
        // Int8 (pre-quantized) activation × KO weight, stored at the compute dtype.
        let t_mm = profile_now();
        let out = self.inner.forward_dynamic(input, out_dtype)?;
        pipeline_record("qmatmul_q8", t_mm);
        Ok(out)
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

impl QMatMul {
    /// As [`candle::quantized::QMatMul::forward_live`]: accepts a wave-scoped
    /// activation and returns a result bounded by the same generation, because
    /// the output is allocated from whichever arena `xs` came from. The
    /// `Module` impl below is this at `'static`, where the bound is vacuous.
    pub fn forward_live<'w>(&self, xs: &candle::LiveTensor<'w>) -> Result<candle::LiveTensor<'w>> {
        let _enter = self.span.enter();
        // Tag the profile entry with the format the matmul actually ran in (`_q8` int8 tensor-core,
        // `_f16`/`_f32` FP) so a perf-vs-off run shows at a glance whether int8 engaged.
        let t_mm = profile_now();

        let in_dtype = xs.dtype();
        let (xs2, reshape_back, _m) = if xs.rank() == 3 {
            let (b, s, k) = xs.dims3()?;
            (xs.reshape((b * s, k))?, Some((b, s)), b * s)
        } else {
            let m = xs.dim(0).unwrap_or(0);
            (xs.clone(), None, m)
        };

        // int8 tensor-core path: q8a128 activations × the KO-twin weight, stored back at the input
        // dtype. The weight twin was baked in at load by `from_*_with_mode`.
        #[cfg(feature = "cuda")]
        if self.int8mode.is_int8() {
            let out2 = self.inner.forward_via_int8(&xs2, self.int8mode)?;
            pipeline_record("qmatmul_q8", t_mm);
            return if let Some((b, s)) = reshape_back {
                let n = out2.dim(1)?;
                out2.reshape((b, s, n))
            } else {
                Ok(out2)
            };
        }

        let use_native_gemx = self.use_gemx
            && matches!(self.inner, candle::quantized::QMatMul::QTensor(_))
            && [DType::F32, DType::BF16, DType::F16, DType::F8E4M3].contains(&in_dtype);

        let out2 = if use_native_gemx {
            // K/128 format has embedded scales; runs in the input dtype.
            let o = self.inner.forward_via_gemx(&xs2)?;
            pipeline_record(
                if in_dtype == DType::F32 {
                    "qmatmul_f32"
                } else {
                    "qmatmul_f16"
                },
                t_mm,
            );
            o
        } else {
            // Fall back to standard quantized matmul path for correctness.
            // Quantized CUDA kernels expect F32 inputs, so cast and restore dtype.
            let xs_f32 = xs2.to_dtype(DType::F32)?;
            let out_f32 = self.inner.forward_live(&xs_f32)?;
            pipeline_record("qmatmul_f32", t_mm);
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

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_live(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}
