//! The `GgmlType` implementations for the plain Rust scalar types.
//!
//! Each of `f64`, `f32`, `f16`, `bf16` and the integer types implements the
//! trait, so an unquantized tensor can be passed where a quantized block is
//! expected. `mod.rs` notes that `f32`, `f16` and `bf16` are copied directly
//! into the matmul scratch buffer.

use super::*;

impl GgmlType for f64 {
    const DTYPE: GgmlDType = GgmlDType::F64;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f64;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs * *ys;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f64;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for bf16 {
    const DTYPE: GgmlDType = GgmlDType::BF16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = bf16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_bf16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u64 {
    const DTYPE: GgmlDType = GgmlDType::U64;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u64;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u64;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i64 {
    const DTYPE: GgmlDType = GgmlDType::I64;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i64;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i64;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u32 {
    const DTYPE: GgmlDType = GgmlDType::U32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u32;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i32 {
    const DTYPE: GgmlDType = GgmlDType::I32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i32;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u16 {
    const DTYPE: GgmlDType = GgmlDType::U16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u16;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i16 {
    const DTYPE: GgmlDType = GgmlDType::I16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i16;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u8 {
    const DTYPE: GgmlDType = GgmlDType::U8;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u8;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u8;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i8 {
    const DTYPE: GgmlDType = GgmlDType::I8;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i8;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i8;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}
