//! A linear layer that is either a dense (unquantized) weight or a quantized
//! [`QMatMul`], mirroring the dtype dispatch in `model.py`'s `linear()`. Dense weights
//! carry the PyTorch `[out, in]` convention (`y = x·Wᵀ`); the quantized path defers to
//! `QMatMul::forward`.

use candle::quantized::{Int8Mode, QMatMul};
use candle::{DType, Device, Module, Result, Tensor};

/// Quantize `x` to a **single shared** q8a128 activation operand and run every weight in
/// `weights` against it, replacing the redundant per-projection activation quantize that
/// [`QLinear::forward`] does independently for each weight. Bit-identical to calling
/// `w.forward(x)` on each weight: the per-call int8 path is `to_dynamic(x) + forward_dynamic`
/// ([`QMatMul::forward_via_int8`]), so sharing the operand removes only the duplicate,
/// deterministic quantize — not any arithmetic.
///
/// Returns `Ok(None)` when any weight is not an int8-KO twin (Dense/Quant test fixtures) or
/// `x` is not on CUDA; the caller then runs its ordinary per-weight `forward` path. `x` is the
/// F32 activation (its last dim, the shared input width, must be a multiple of 128 — the same
/// constraint the per-call int8 path already imposes). Every output is F32 `[lead.., N_w]`.
#[cfg(feature = "cuda")]
pub fn shared_int8_forward(x: &Tensor, weights: &[&QLinear]) -> Result<Option<Vec<Tensor>>> {
    use candle::quantized::cuda;
    if weights.is_empty() || !weights.iter().all(|w| w.is_int8()) {
        return Ok(None);
    }
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Ok(None),
    };
    // Same F32-contiguous activation each per-call `forward` would quantize — the operand
    // bytes are therefore identical, just produced once instead of once per weight.
    //
    // (A single fused `qkv_segmented` launch over all weights was measured to NOT help here:
    // its concatenated output must be split for the different `rms_norm` consumers, and the
    // per-slice contiguous copies that requires offset the one-launch saving — flat decode,
    // slightly slower prefill. `qkv_segmented` only wins when the concat is consumed as-is.)
    let xc = x.to_dtype(DType::F32)?.contiguous()?;
    let op = cuda::to_dynamic(&xc, Int8Mode::Performance, &dev)?;
    let mut out = Vec::with_capacity(weights.len());
    for w in weights {
        let q = w
            .int8_qmatmul()
            .expect("all weights checked int8 above via is_int8()");
        out.push(q.forward_dynamic(op.as_dynamic())?);
    }
    Ok(Some(out))
}

/// Non-CUDA builds have no q8a128 path; callers fall back to per-weight `forward`.
#[cfg(not(feature = "cuda"))]
pub fn shared_int8_forward(_x: &Tensor, _weights: &[&QLinear]) -> Result<Option<Vec<Tensor>>> {
    Ok(None)
}

/// The common two-projection case of [`shared_int8_forward`]: run `w0` and `w1` against a
/// single shared q8a128 activation operand (int8-KO path), or their per-weight `forward`
/// fallback (Dense/Quant fixtures, or non-CUDA), returning `(w0·x, w1·x)`.
pub fn shared_int8_pair(x: &Tensor, w0: &QLinear, w1: &QLinear) -> Result<(Tensor, Tensor)> {
    match shared_int8_forward(x, &[w0, w1])? {
        Some(v) => Ok((v[0].clone(), v[1].clone())),
        None => Ok((w0.forward(x)?, w1.forward(x)?)),
    }
}

#[derive(Debug, Clone)]
pub enum QLinear {
    /// Dense weight `[out, in]` (float, native width).
    Dense(Tensor),
    /// Quantized weight, standard matmul (weight dequantized to F32 inside `QMatMul::forward`).
    Quant(QMatMul),
    /// KO-twin weight run through the **int8 tensor-core** path: activation quantized to q8a128,
    /// int8×int8 → int32 accumulation, scaled to F32 only at the end. No F32 weight
    /// materialization — the compute-space matmul (~2× vs the dequant-and-F32-matmul path).
    Int8(QMatMul),
}

/// A dense `Tensor` weight becomes a `Dense` `QLinear` — lets constructors take
/// `impl Into<QLinear>` so float test fixtures pass a `Tensor` unchanged while the loaders pass a
/// ready `QLinear` (int8-KO on the engine path).
impl From<Tensor> for QLinear {
    fn from(w: Tensor) -> Self {
        Self::Dense(w)
    }
}

impl QLinear {
    pub fn from_weight(w: Tensor) -> Self {
        Self::Dense(w)
    }

    pub fn from_qmatmul(q: QMatMul) -> Self {
        Self::Quant(q)
    }

    /// Whether this weight runs the int8-KO tensor-core path (vs a Dense/Quant fixture).
    pub fn is_int8(&self) -> bool {
        matches!(self, Self::Int8(_))
    }

    /// The inner `QMatMul` for the int8-KO path, for shared-operand dispatch
    /// (see [`shared_int8_forward`]).
    fn int8_qmatmul(&self) -> Option<&QMatMul> {
        match self {
            Self::Int8(q) => Some(q),
            _ => None,
        }
    }

    /// The KO weight's device pointer + GGML dtype, for grouped/batched dispatch
    /// (e.g. running several KO weights through one `grouped_qmatmul` launch). `None`
    /// for non-int8 weights or non-CUDA storage.
    #[cfg(feature = "cuda")]
    pub fn ko_weight(&self) -> Option<(u64, candle::quantized::GgmlDType)> {
        match self {
            Self::Int8(QMatMul::QTensor(qt)) => Some((qt.cuda_data_ptr()?, qt.dtype())),
            _ => None,
        }
    }

    /// Wrap a KO-repacked weight for the int8 tensor-core path.
    pub fn from_int8(q: QMatMul) -> Self {
        Self::Int8(q)
    }

    /// The device the weight lives on (all variants are single-device).
    pub fn device(&self) -> Device {
        match self {
            Self::Dense(t) => t.device().clone(),
            Self::Quant(q) | Self::Int8(q) => match q {
                QMatMul::QTensor(qt) => qt.device(),
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
            },
        }
    }

    /// The weight's input dimension (`in` of the `[out, in]` matrix), for callers that reshape a
    /// row to the matmul's expected width. The KO storage keeps the logical `[out, in]` shape.
    pub fn in_dim(&self) -> usize {
        let dims = match self {
            Self::Dense(t) => t.dims().to_vec(),
            Self::Quant(q) | Self::Int8(q) => match q {
                QMatMul::QTensor(qt) => qt.shape().dims().to_vec(),
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dims().to_vec(),
            },
        };
        dims[dims.len() - 1]
    }

    /// `y = x·Wᵀ`. Dense: input dtype. Quant: `QMatMul` dequant path (F32). Int8: q8a128×KO
    /// tensor-core path (F32 accum out). All cast back to the input dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(w) => x.broadcast_matmul(&w.t()?.to_dtype(x.dtype())?),
            Self::Quant(q) => {
                let in_dtype = x.dtype();
                let y = q.forward(&x.to_dtype(DType::F32)?)?;
                y.to_dtype(in_dtype)
            }
            Self::Int8(q) => {
                let in_dtype = x.dtype();
                let y = q.forward_via_int8(
                    &x.to_dtype(DType::F32)?.contiguous()?,
                    Int8Mode::Performance,
                )?;
                y.to_dtype(in_dtype)
            }
        }
    }
}
