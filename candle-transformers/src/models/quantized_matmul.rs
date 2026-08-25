#[cfg(feature = "cuda")]
use candle::quantized::ko_quant::ko_tileable;
#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
use candle::quantized::{Int8Mode, QTensor};
use candle::{DType, Module, Result, Tensor};

use crate::models::profile::{pipeline_record, profile_now};

/// Traced quantized matmul wrapper.
///
/// Key behavior:
/// - Adds a tracing span around matmul.
/// - On CUDA, an int8 mode: repacks the weight to its KO twin at load and runs the q8a128 int8
///   tensor-core matmul (the weight twin is chosen by the mode; see [`Int8Mode`]). A weight whose
///   shape does not fit the matmul tiling keeps the standard path and reports [`Int8Mode::Off`],
///   so [`Self::int8mode`] — not the mode the caller asked for — is what a consumer must dispatch
///   on.
/// - [`Int8Mode::Off`] and CPU/Metal: standard GGML kernels. Off is the
///   diagnostic mode (`INT8MODE=off`) — correctness over speed; the FP-GEMX
///   fast path it used to take was deleted when production went int8-only
///   (nothing but Off-mode runs exercised it, and it had rotted to NaN).
/// - For BF16/F16 inputs, casts through F32 for the standard kernels and restores the dtype.
/// - For rank-3 inputs, flattens (B,S,K) -> (B*S,K) to avoid broadcasting weights.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
    /// Numeric mode of this weight: `Off` → standard path; an int8 mode → the weight is
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

        // int8 mode (CUDA): every matmul whose shape TILES becomes a KO weight, so the q8a128
        // activations the fused producers emit pair with a KO twin — a *float* weight must never
        // receive a q8a128 (that pairing has no kernel). A gemx-supported quantized source repacks
        // to its KO twin directly; a float source (F32/F16/BF16 — e.g. the F32 router
        // `ffn_gate_inp` or lm_head, which have no quantized GGUF form and would otherwise
        // dequantize to a plain float weight) is first quantized to Q8_0 so it, too, gets a Q8_KO
        // twin. Routing stays near-lossless at 8 bits, and ln2's two consumers — router and
        // experts — are then both int8.
        //
        // **The tiling caveat is part of the invariant, not an exception to it.** A sub-tile
        // weight stays dense and reports `Off`, so it must be consumed through `forward_live_as`
        // (which dispatches on this weight's own mode) and never through a producer-fused
        // `DynamicActs::Int8` — `ensure_qmatmul_pairing` would refuse that pairing at the first
        // forward. Today's sub-tile weights are the narrow DeltaNet/mHC projections, all of which
        // take `forward_live_as`; a new fused consumer must check `int8mode()` rather than assume.
        #[cfg(feature = "cuda")]
        if mode.is_int8() {
            // **A shape that will not tile is a per-tensor fact, knowable up front — not a load
            // failure.** The q8a128 matmul tiles N in blocks of 32, and a narrow projection can
            // sit below it (Qwen3.5-0.8B's DeltaNet `w_alpha`/`w_beta` are `[16, hidden]` at 16
            // linear-V heads). Refusing the whole model over one small weight is wrong;
            // `latent_moe`'s `qlinear_int8` has always left that tensor on the dequant path, and
            // this constructor is the reason every other family could not.
            //
            // Tested BEFORE the repack rather than caught after it: reading an `Err` as "did not
            // tile" would also swallow a device OOM or a driver fault during `repack_ko`, silently
            // downgrading the weight to the FP path with no way to tell the two apart. Anything
            // that fails below is a real error and propagates.
            let dims = ws.shape().dims();
            let tileable = match dims {
                [nrows, ncols] => ko_tileable(*nrows, *ncols),
                // Only the 2-D case has a KO twin here: `repack_for_optimization` takes a single
                // matrix. Expert banks are repacked per-expert offline (`quantized::prepare`) and
                // arrive through `from_repacked`.
                _ => false,
            };
            if tileable {
                let src = if ws.supports_gemx_repacking() {
                    std::sync::Arc::clone(&ws)
                } else {
                    let f32 = ws.dequantize(&ws.device())?;
                    std::sync::Arc::new(QTensor::quantize(&f32, GgmlDType::Q8_0)?)
                };
                let inner =
                    candle::quantized::QMatMul::from_arc(src)?.repack_for_optimization(mode)?;
                return Ok(Self {
                    inner,
                    span,
                    int8mode: mode,
                });
            }
            tracing::debug!(
                shape = ?dims,
                dtype = ?ws.dtype(),
                "int8: weight does not fit the KO matmul tiling — dense fallback for this tensor"
            );
            let inner = candle::quantized::QMatMul::from_arc(ws)?;
            return Ok(Self {
                inner,
                span,
                int8mode: Int8Mode::Off,
            });
        }

        // Off (or non-CUDA): the standard GGML path, unmodified weights.
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        Ok(Self {
            inner,
            span,
            int8mode: Int8Mode::Off,
        })
    }

    /// Create from an already-repacked KO twin — how expert slots are wrapped
    /// from the pinned-pool's pre-repacked KO bytes, so they report int8 to
    /// `compute_experts_grouped`. The KO twin is baked into the dtype, and it
    /// is the only runnable repacked form: the FP GEMX K/128 kernel no longer
    /// exists, so a non-KO tensor is refused here rather than constructed into
    /// a matmul that cannot run.
    pub fn from_qtensor_repacked(repacked: QTensor) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        if !repacked.dtype().is_ko() {
            candle::bail!(
                "from_qtensor_repacked: only KO twins are runnable — the FP GEMX \
                 K/128 form has no kernel any more (deleted with the float fast path)"
            );
        }
        let inner = candle::quantized::QMatMul::from_qtensor(repacked)?;

        Ok(Self {
            inner,
            span,
            int8mode: Int8Mode::Precision,
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
        self.forward_live_as(xs, xs.dtype())
    }

    /// [`Self::forward_live`] with the output width named by the caller.
    ///
    /// The int8 kernel accumulates in F32 registers and converts on the store, so `out_dtype` costs
    /// nothing to honour — it selects which of the dense KO kernel's three output variants runs.
    /// A consumer that needs a width other than the activation's must ask here: widening the result
    /// afterwards rounds through the narrow store and back, which is both a full-tensor launch and
    /// a loss of the accumulator's own precision.
    pub fn forward_live_as<'w>(
        &self,
        xs: &candle::LiveTensor<'w>,
        out_dtype: DType,
    ) -> Result<candle::LiveTensor<'w>> {
        let _enter = self.span.enter();
        // Tag the profile entry with the format the matmul actually ran in (`_q8` int8 tensor-core,
        // `_f16`/`_f32` FP) so a perf-vs-off run shows at a glance whether int8 engaged.
        let t_mm = profile_now();

        let (xs2, reshape_back, _m) = if xs.rank() == 3 {
            let (b, s, k) = xs.dims3()?;
            (xs.reshape((b * s, k))?, Some((b, s)), b * s)
        } else {
            let m = xs.dim(0).unwrap_or(0);
            (xs.clone(), None, m)
        };

        // int8 tensor-core path: q8a128 activations × the KO-twin weight, stored at `out_dtype`.
        // The weight twin was baked in at load by `from_*_with_mode`.
        #[cfg(feature = "cuda")]
        if self.int8mode.is_int8() {
            let out2 = self
                .inner
                .forward_via_int8(&xs2, self.int8mode, out_dtype)?;
            pipeline_record("qmatmul_q8", t_mm);
            return if let Some((b, s)) = reshape_back {
                let n = out2.dim(1)?;
                out2.reshape((b, s, n))
            } else {
                Ok(out2)
            };
        }

        // Non-int8 (the Off diagnostic mode, CPU, Metal): the standard
        // quantized matmul path. The CUDA kernels expect F32 inputs, so cast
        // and restore the dtype.
        let out2 = {
            let xs_f32 = xs2.to_dtype(DType::F32)?;
            let out_f32 = self.inner.forward_live(&xs_f32)?;
            pipeline_record("qmatmul_f32", t_mm);
            if out_dtype == DType::F32 {
                out_f32
            } else {
                out_f32.to_dtype(out_dtype)?
            }
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Device, Tensor};

    /// **The row-count dispatch curve — the measurement both dispatch
    /// boundaries are pinned from.**
    ///
    /// Two decisions come from this curve. `QCudaStorage::fwd`'s `max_bm`
    /// splits the vec kernel (≤ 8 rows, its template ceiling) from MMQ; and
    /// `run_mul_mat`'s `narrow` gate splits MMQ's batch-narrow `mmq_x = 16`
    /// tile (through 32 rows) from its wide per-type tile. The narrow tile
    /// exists because MMQ's dot work scales with the tile's batch width while
    /// its weight loads do not, so a 12-row batch under a 64-wide tile spent
    /// 4/5 of its arithmetic on padding — the row counts a speculative VERIFY
    /// wave produces (`sessions × (draft + 1)`) sat exactly there.
    ///
    /// Prints µs/call and µs/row; asserts nothing, because the shape of the
    /// curve is the property and its absolute height is the machine's. Re-run
    /// this before moving either boundary.
    #[test]
    #[ignore = "GPU timing harness — prints the qmatmul row-count curve"]
    fn qmatmul_row_count_curve() {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // **Two shapes, because the boundary is not shape-independent.**
        //
        // The gate/up projection is where a DeltaNet layer spends its time and
        // is what the boundary was originally set on. The LM head is an order of
        // magnitude wider in its output dimension, and it is the matmul the MTP
        // drafter is bound by — a drafted token spends most of its time in one
        // read of it. The drafter runs one row per session, so a decode wave of
        // `n` sessions crosses this boundary at exactly `n = 9`, which is where
        // speculation was measured to fall off a cliff (1.46x at eight sessions,
        // 0.45x at ten). If the two shapes disagree about where MMQ starts
        // winning, that cliff is this dispatch and not the wave.
        for (label, out_dim, in_dim) in [
            ("gate/up", 12288usize, 4096usize),
            ("lm_head", 151936usize, 4096usize),
        ] {
            let w = Tensor::randn(0f32, 0.02, (out_dim, in_dim), &dev).unwrap();
            let q = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q6_K).unwrap()).unwrap();
            let weight_mib = (out_dim * in_dim / 256 * 210) >> 20;
            eprintln!(
                "--- {label}: Q6_K [{out_dim}, {in_dim}] = {weight_mib} MiB; \
                 vec ≤ 8, x16-MMQ 9–32, x64-MMQ above ---"
            );

            for rows in [1usize, 2, 4, 8, 9, 12, 16, 17, 24, 32, 33, 48, 49, 64, 128] {
                let x = Tensor::randn(0f32, 1.0, (rows, in_dim), &dev)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let run = || {
                    let _ = q.inner().forward(&x).unwrap();
                };
                for _ in 0..10 {
                    run();
                }
                dev.synchronize().unwrap();
                let reps = 100;
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    run();
                }
                dev.synchronize().unwrap();
                let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
                let path = if rows <= 8 {
                    "vec"
                } else if rows <= 32 {
                    "mmq-x16"
                } else {
                    "mmq-x64"
                };
                eprintln!(
                    "  {rows:>4} rows  {us:8.1} µs  {:7.2} µs/row  {path}",
                    us / rows as f64
                );
            }
        }
    }
}
