//! TensorScalar Enum and Trait
//!
use crate::{DType, LiveTensor, Result, Tensor, WithDType};
use float8::F8E4M3;
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U8(u8),
    U32(u32),
    I64(i64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    F8E4M3(F8E4M3),
}

impl<T: WithDType> From<T> for Scalar {
    fn from(value: T) -> Self {
        value.to_scalar()
    }
}

impl Scalar {
    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::U8 => Scalar::U8(0),
            DType::U32 => Scalar::U32(0),
            DType::I64 => Scalar::I64(0),
            DType::BF16 => Scalar::BF16(bf16::ZERO),
            DType::F16 => Scalar::F16(f16::ZERO),
            DType::F32 => Scalar::F32(0.0),
            DType::F64 => Scalar::F64(0.0),
            DType::F8E4M3 => Scalar::F8E4M3(F8E4M3::ZERO),
        }
    }

    pub fn one(dtype: DType) -> Self {
        match dtype {
            DType::U8 => Scalar::U8(1),
            DType::U32 => Scalar::U32(1),
            DType::I64 => Scalar::I64(1),
            DType::BF16 => Scalar::BF16(bf16::ONE),
            DType::F16 => Scalar::F16(f16::ONE),
            DType::F32 => Scalar::F32(1.0),
            DType::F64 => Scalar::F64(1.0),
            DType::F8E4M3 => Scalar::F8E4M3(F8E4M3::ONE),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Scalar::U8(_) => DType::U8,
            Scalar::U32(_) => DType::U32,
            Scalar::I64(_) => DType::I64,
            Scalar::BF16(_) => DType::BF16,
            Scalar::F16(_) => DType::F16,
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
            Scalar::F8E4M3(_) => DType::F8E4M3,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::U8(v) => *v as f64,
            Scalar::U32(v) => *v as f64,
            Scalar::I64(v) => *v as f64,
            Scalar::BF16(v) => v.to_f64(),
            Scalar::F16(v) => v.to_f64(),
            Scalar::F32(v) => *v as f64,
            Scalar::F64(v) => *v,
            Scalar::F8E4M3(v) => v.to_f64(),
        }
    }
}

/// The right-hand side of a comparison, once normalised to a tensor.
///
/// `'w` is the operand's, so comparing against a leased tensor yields a result
/// bounded by that lease rather than one claiming to live forever.
pub enum TensorScalar<'w> {
    Tensor(LiveTensor<'w>),
    Scalar(LiveTensor<'w>),
}

pub trait TensorOrScalar<'w> {
    fn to_tensor_scalar(self) -> Result<TensorScalar<'w>>;
}

impl<'w> TensorOrScalar<'w> for &LiveTensor<'w> {
    fn to_tensor_scalar(self) -> Result<TensorScalar<'w>> {
        Ok(TensorScalar::Tensor(self.clone()))
    }
}

/// A bare scalar allocates its own one-element tensor, which is owned and so
/// fits any `'w` the call site needs.
impl<'w, T: WithDType> TensorOrScalar<'w> for T {
    fn to_tensor_scalar(self) -> Result<TensorScalar<'w>> {
        let scalar = Tensor::new(self, &crate::Device::Cpu)?;
        Ok(TensorScalar::Scalar(scalar))
    }
}
