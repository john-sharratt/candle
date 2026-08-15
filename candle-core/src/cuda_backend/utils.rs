/// Helper functions to plug cuda kernels in candle.
use crate::{Layout, Result, WithDType};
pub use cudarc;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use super::{Backing, CudaDevice, CudaError, CudaStorage, WrapErr};

pub type S = super::CudaStorageSlice;

/// An op's output slice together with the [`Backing`] the storage wrapping it
/// must carry.
///
/// Returned as a pair, never separately: a wave range marked `Owned` is a double
/// free and a pool buffer marked `Lease` is a permanent leak, so the only safe
/// shape is one where the allocation decides its own tag. Every `f` below
/// produces this by calling [`super::alloc_inheriting`] rather than `dev.alloc`.
pub type Out<T> = (CudaSlice<T>, Backing);

/// The same pairing for the dtype-erased dispatchers.
pub type OutS = (S, Backing);

pub trait Map1 {
    /// `origin` is the operand's backing — the arena this op's output should be
    /// allocated from, so a result computed over wave-backed memory stays on it.
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>>;

    /// Takes the whole [`CudaStorage`], not just its slice: the slice is dtype
    /// and bytes, the storage is where those bytes came from, and the second is
    /// exactly what this dispatch exists to pass along.
    fn map(&self, s: &CudaStorage, d: &CudaDevice, l: &Layout) -> Result<OutS> {
        let o = s.backing;
        let out = match &s.slice {
            S::U8(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::U8(v), b)
            }
            S::U32(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::U32(v), b)
            }
            S::I64(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::I64(v), b)
            }
            S::BF16(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::BF16(v), b)
            }
            S::F16(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::F16(v), b)
            }
            S::F32(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::F32(v), b)
            }
            S::F64(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::F64(v), b)
            }
            S::F8E4M3(x) => {
                let (v, b) = self.f(x, d, l, o)?;
                (S::F8E4M3(v), b)
            }
            S::Moved => S::unreachable_moved(),
        };
        Ok(out)
    }
}

pub trait Map2 {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &CudaSlice<T>,
        layout1: &Layout,
        src2: &CudaSlice<T>,
        layout2: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>>;

    /// Inherits from the **first** operand. Both share one `'w` by the time they
    /// reach here — variance already unified them at the shorter — so either
    /// would serve, and naming one keeps the rule stated in a single place.
    fn map(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        d: &CudaDevice,
    ) -> Result<OutS> {
        let o = s1.backing;
        let out = match (&s1.slice, &s2.slice) {
            (S::U8(a), S::U8(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::U8(v), b)
            }
            (S::U32(a), S::U32(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::U32(v), b)
            }
            (S::I64(a), S::I64(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::I64(v), b)
            }
            (S::BF16(a), S::BF16(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::BF16(v), b)
            }
            (S::F16(a), S::F16(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::F16(v), b)
            }
            (S::F32(a), S::F32(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::F32(v), b)
            }
            (S::F64(a), S::F64(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::F64(v), b)
            }
            (S::F8E4M3(a), S::F8E4M3(c)) => {
                let (v, b) = self.f(a, l1, c, l2, d, o)?;
                (S::F8E4M3(v), b)
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in binary op".to_string(),
            ))?,
        };
        Ok(out)
    }
}

pub trait Map2InPlace {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()>;

    fn map(
        &self,
        dst: &mut S,
        dst_l: &Layout,
        src: &S,
        src_l: &Layout,
        d: &CudaDevice,
    ) -> Result<()> {
        match (dst, src) {
            (S::U8(dst), S::U8(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::U32(dst), S::U32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I64(dst), S::I64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::BF16(dst), S::BF16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F16(dst), S::F16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F32(dst), S::F32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F64(dst), S::F64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F8E4M3(dst), S::F8E4M3(src)) => self.f(dst, dst_l, src, src_l, d),
            _ => Err(CudaError::InternalError(
                "dtype mismatch in binary op".to_string(),
            ))?,
        }
    }
}

pub trait Map1Any {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        wrap: W,
        origin: Backing,
    ) -> Result<OutS>;

    fn map(&self, s: &CudaStorage, d: &CudaDevice, l: &Layout) -> Result<OutS> {
        let o = s.backing;
        let out = match &s.slice {
            S::U8(x) => self.f(x, d, l, S::U8, o)?,
            S::U32(x) => self.f(x, d, l, S::U32, o)?,
            S::I64(x) => self.f(x, d, l, S::I64, o)?,
            S::BF16(x) => self.f(x, d, l, S::BF16, o)?,
            S::F16(x) => self.f(x, d, l, S::F16, o)?,
            S::F32(x) => self.f(x, d, l, S::F32, o)?,
            S::F64(x) => self.f(x, d, l, S::F64, o)?,
            S::F8E4M3(x) => self.f(x, d, l, S::F8E4M3, o)?,
            S::Moved => S::unreachable_moved(),
        };
        Ok(out)
    }
}

pub trait Map2Any {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &CudaSlice<T>,
        layout1: &Layout,
        src2: &CudaSlice<T>,
        layout2: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<OutS>;

    fn map(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
        d: &CudaDevice,
    ) -> Result<OutS> {
        let o = s1.backing;
        let out = match (&s1.slice, &s2.slice) {
            (S::U8(a), S::U8(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::U32(a), S::U32(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::I64(a), S::I64(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::BF16(a), S::BF16(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::F16(a), S::F16(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::F32(a), S::F32(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::F64(a), S::F64(c)) => self.f(a, l1, c, l2, d, o)?,
            (S::F8E4M3(a), S::F8E4M3(c)) => self.f(a, l1, c, l2, d, o)?,
            _ => Err(CudaError::InternalError(
                "dtype mismatch in binary op".to_string(),
            ))
            .w()?,
        };
        Ok(out)
    }
}
