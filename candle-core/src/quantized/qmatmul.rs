//! The quantized matrix-multiply operator, and the repacking that prepares a
//! weight for the int8 or float path it will run on.
//!
//! Repacking happens once, at load. A weight becomes the lane-major KO twin for
//! the int8 path or the GEMX layout for the float path, and the inference mode
//! decides which. A narrowed repack lets a caller hold one weight at lower
//! precision for residency, and it never widens a weight. A float weight is
//! dequantized into a plain tensor, so the KO twin is built from the storage
//! while that storage still exists, whether it is device memory or host-mapped.

use super::*;

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

impl QMatMul {
    /// Get the inner `QTensor` if this is the quantized variant.
    pub fn qtensor(&self) -> Option<&QTensor> {
        match self {
            QMatMul::QTensor(qt) => Some(qt),
            _ => None,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_BF16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_BF16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else if DEQUANTIZE_ALL_BF16.with(|b| *b) {
            let tensor = qtensor.dequantize_bf16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            // A KO weight may run split-K at decode width. Every KO matmul is built here —
            // repacked at load, read prepared from disk, or viewed in a streaming slot — so
            // its stream's split-K scratch is created here too, and no forward allocates it.
            #[cfg(feature = "cuda")]
            if let QStorage::Cuda(cs) = &qtensor.storage {
                if qtensor.dtype().is_ko() {
                    cuda::ensure_split_k_scratch(cs.device())?;
                }
            }
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    /// Repack the weight for the inference numeric mode selected by `mode`: any int8 mode → the
    /// lane-major KO format read by the q8a128 int8 tensor-core matmul (`repack_ko`, twin chosen
    /// by [`GgmlDType::to_ko`]); [`Int8Mode::Off`] → the FP GEMX repack read by the dequant-weight
    /// float matmul (`repack_gemx`). The weight-side half of the single `int8mode` knob (paired
    /// with `cuda::to_dynamic` on the activation side); the matmul's KO⇔int8 pairing guard keeps
    /// the two consistent. Requires the quantized (`QTensor`) variant on CUDA.
    /// Non-CUDA build: the KO/GEMX repacked layouts are CUDA-kernel formats, so there is
    /// nothing to repack to — this always errors (the `dummy_cuda` convention).
    #[cfg(not(feature = "cuda"))]
    pub fn repack_for_optimization(&self, _mode: Int8Mode) -> Result<QMatMul> {
        crate::bail!("repack_for_optimization requires the cuda feature")
    }

    #[cfg(feature = "cuda")]
    pub fn repack_for_optimization(&self, mode: Int8Mode) -> Result<QMatMul> {
        self.repack_for_optimization_into(mode, None)
    }

    /// [`Self::repack_for_optimization`], placing the repacked weight at `dst`.
    ///
    /// The seam that puts a model's weights inside the device reservation. The
    /// repacked weight is the one that stays resident — the compact source is
    /// dropped as this returns — so this is the allocation worth placing, and
    /// the caller sizes `dst` with `cuda::ko_repacked_bytes`.
    ///
    /// `None` allocates from the pool, which is the right answer with no
    /// reservation to carve from.
    #[cfg(feature = "cuda")]
    pub fn repack_for_optimization_into(
        &self,
        mode: Int8Mode,
        dst: Option<(u64, crate::cuda_backend::wave_provenance::LeaseOrigin)>,
    ) -> Result<QMatMul> {
        self.repack_for_optimization_narrowed(mode, dst, None)
    }

    /// [`Self::repack_for_optimization_into`], with the twin's width chosen by the caller.
    ///
    /// `narrow` replaces what [`GgmlDType::to_ko`] would pick. It is **not** a second numeric
    /// mode: the mode still decides whether the int8 path runs at all and still governs every
    /// other tensor. This is a per-tensor residency decision, for the case where a weight is
    /// worth carrying at lower precision because the alternative is not carrying the model.
    ///
    /// The repack narrows in the pass it already makes — it dequantizes the source and
    /// requantizes it — so the only cost is precision. Requantizing the *source* to a narrower
    /// quant first and letting `to_ko` follow would cost a whole extra dequant/quant round trip
    /// per tensor at load, which for a 248320×5120 head is seconds, not milliseconds.
    ///
    /// A narrowing that would *widen* is refused: the caller asked for a smaller resident
    /// footprint, and silently giving it a larger one is the failure it would never think to
    /// check for.
    #[cfg(feature = "cuda")]
    pub fn repack_for_optimization_narrowed(
        &self,
        mode: Int8Mode,
        dst: Option<(u64, crate::cuda_backend::wave_provenance::LeaseOrigin)>,
        narrow: Option<GgmlDType>,
    ) -> Result<QMatMul> {
        let qt = self.qtensor().ok_or_else(|| {
            crate::Error::Msg(
                "repack_for_optimization: not a QTensor. A source that arrives already float is \
                 dequantized into a plain tensor by `QMatMul::from_qtensor`, so its quantized \
                 storage is gone by the time it reaches here — repack it with \
                 `QTensor::repack_ko` *before* building the QMatMul."
                    .into(),
            )
        })?;
        qt.repack_ko(mode, dst, narrow)
            .and_then(QMatMul::from_qtensor)
    }
}

impl QTensor {
    /// Build this weight's KO twin, straight from its storage.
    ///
    /// # Why this is on the tensor and not on the matmul
    ///
    /// [`QMatMul::repack_for_optimization_narrowed`] used to hold this code, and it could not
    /// serve a **float** source: `QMatMul::from_qtensor` dequantizes an F32/F16/BF16 weight
    /// into a plain `Tensor`, so by the time the repack ran there was no quantized storage
    /// left to repack and it failed with "not a QTensor".
    ///
    /// That mattered once `dequantize_f32_into` learned to widen a float band, because the
    /// banded repack can now read a float source perfectly well — the only thing standing in
    /// the way was the detour through a type that had already thrown the source away. So the
    /// work lives here, where the storage still exists, and the matmul entry point delegates.
    ///
    /// Both loaders reach the same kernel by the same rules: one from a device source, one
    /// from mapped bytes (`cuda::repack_ko_from_host`). A weight's twin does not depend on
    /// which of them built it.
    #[cfg(feature = "cuda")]
    pub fn repack_ko(
        &self,
        mode: Int8Mode,
        dst: Option<(u64, crate::cuda_backend::wave_provenance::LeaseOrigin)>,
        narrow: Option<GgmlDType>,
    ) -> Result<QTensor> {
        let qt = self;
        // Expects a COMPACT source weight: re-running on an already-KO-optimized weight would
        // double-process it (and `to_ko`/`repack_gemx` have no KO source). Guard explicitly.
        if qt.dtype().is_ko() {
            crate::bail!(
                "repack_for_optimization: weight is already KO-optimized ({:?}); call on the \
                 compact source weight once",
                qt.dtype()
            );
        }
        let shape = qt.shape().clone();
        let new_storage = match &qt.storage {
            QStorage::Cuda(cs) => {
                if mode.is_int8() {
                    let picked = qt.dtype().to_ko(mode)?;
                    let ko = match narrow {
                        None => picked,
                        Some(n) => {
                            if !n.is_ko() {
                                crate::bail!("repack_for_optimization: {n:?} is not a KO twin")
                            }
                            if n.bits_per_weight() > picked.bits_per_weight() {
                                crate::bail!(
                                    "repack_for_optimization: narrowing to {n:?} ({:.2} bpw) is \
                                     wider than the {picked:?} ({:.2} bpw) this mode would pick \
                                     — narrowing must not grow the resident footprint",
                                    n.bits_per_weight(),
                                    picked.bits_per_weight(),
                                )
                            }
                            n
                        }
                    };
                    QStorage::Cuda(cs.repack_ko_into(&shape, ko, dst)?)
                } else {
                    // The GEMX repack has no destination form: it is the
                    // measurement path (`repack_gemx`'s own docs), not a path a
                    // model load takes, so there is no weight here to place.
                    if dst.is_some() {
                        crate::bail!(
                            "repack_for_optimization_into: a destination was supplied for a \
                             non-int8 mode, which repacks through GEMX and allocates its own \
                             output. The reservation would be left holding an unwritten hole."
                        );
                    }
                    QStorage::Cuda(cs.repack_gemx(&shape)?)
                }
            }
            _ => crate::bail!("repack_for_optimization requires CUDA storage"),
        };
        Ok(QTensor {
            storage: new_storage,
            shape,
            lease: PhantomData,
        })
    }
}

impl QMatMul {
    /// Project an activation that may live on an inference wave.
    ///
    /// [`crate::Module`] takes `&Tensor`, so it cannot accept a wave-scoped
    /// activation — and that restriction is deliberate everywhere else, because
    /// a module may retain what it is given. This one does not retain, but it
    /// does *inherit*: the output is allocated from whichever arena `xs` came
    /// from, so the result is bounded by `xs`'s generation rather than being
    /// owned. The `Module` impl below is this method at `'static`, where the
    /// distinction collapses and the result really is owned.
    pub fn forward_live<'w>(&self, xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        match self {
            // MXFP4 has no native matmul kernel (and no native FP4 tensor-core MMA on
            // sm_120), so dequantize the weight (fast GPU kernel / CPU codec) and run a
            // standard matmul. Used by the DeepSeek-V4 routed experts.
            Self::QTensor(t) if t.dtype() == GgmlDType::MXFP4 => {
                let w = t.dequantize(xs.device())?.to_dtype(xs.dtype())?; // [out, in]
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            // The float-weight arms. Reached both by the `CANDLE_DEQUANTIZE_ALL`
            // fallback and, unavoidably, by any checkpoint that simply stores a
            // tensor unquantized — `from_arc` dequantizes F32/F16/BF16 weights
            // whatever the setting, so an FP checkpoint runs entirely here.
            //
            // `matmul` allocates its output beside its operand, so the result is
            // wave-backed when `xs` is, which is precisely what the `'w` on the
            // return type says. Copying `xs` off the wave first would make it
            // owned instead — permitted by the signature, and pure loss: a
            // full-width copy per projection whose only effect is to move the
            // output back onto the pool.
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_live(xs)
    }
}
