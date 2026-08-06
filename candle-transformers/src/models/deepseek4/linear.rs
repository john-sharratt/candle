//! A linear layer that is either a dense (unquantized) weight or a quantized
//! [`QMatMul`], mirroring the dtype dispatch in `model.py`'s `linear()`. Dense weights
//! carry the PyTorch `[out, in]` convention (`y = x·Wᵀ`); the quantized path defers to
//! `QMatMul::forward`.

use candle::quantized::{Int8Mode, QMatMul};
use candle::{DType, Device, Module, Result, Tensor};

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
