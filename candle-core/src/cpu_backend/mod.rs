//! The `BackendStorage` and `BackendDevice` implementation for the CPU, always
//! compiled in and over typed `Vec` storage.
//!
//! `CpuStorage` holds one `Vec` per dtype, `CpuStorageRef` is its borrowed form,
//! and `CpuDevice` is a zero-sized marker. Each operation is a unit struct that
//! implements the map traits in `utils`, which drive the element loops. Those loops
//! call the SIMD `vec_dot` and `vec_sum` kernels where the CPU supports them, and
//! rayon splits rows across threads. Matmul uses `gemm` by default, or MKL or
//! Accelerate when those features are enabled.
//!
//! - `backend_device`: the `BackendDevice` implementation.
//! - `storage`: in-place and sparse updates on storage.
//! - `elementwise`: unary, binary, comparison and select operations.
//! - `indexing`: gather, index-select, scatter and index-add.
//! - `matmul`: batched matrix multiplication.
//! - `reduce`: sum and index reductions.
//! - `spatial`: pooling, upsampling and convolutions.
//! - `copy`: strided copies.
//! - `dtype`: per-dtype dispatch, carried by the operation files.
//! - `utils`: the map traits the operation structs implement.

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Error, IntDType, Layout, Result, Shape, WithDType};
use float8::F8E4M3;
use half::{bf16, f16};
use rayon::prelude::*;

mod utils;
pub use utils::{
    binary_map, binary_map_vec, unary_map, unary_map_vec, Map1, Map1Any, Map2, Map2InPlace, Map2U8,
};

const USE_IM2COL_CONV1D: bool = true;

const USE_COL2IM_CONV1D_TR: bool = true;

const USE_IM2COL_CONV2D: bool = true;

// TODO: Maybe we should not implement [Clone] here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    F8E4M3(Vec<F8E4M3>),
}

#[derive(Debug, Clone)]
pub enum CpuStorageRef<'a> {
    U8(&'a [u8]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
    F8E4M3(&'a [F8E4M3]),
}

#[derive(Debug, Clone)]
pub struct CpuDevice;

#[macro_export]
macro_rules! map_dtype {
    ($name:expr, $storage:ident, $fn:expr, ($($dtypes:ident),+)) => {
        match $storage {
            $(CpuStorage::$dtypes(__e) => CpuStorage::$dtypes($fn(__e)),)*
            s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?,
        }
    };
}

mod backend_device;
mod copy;
#[cfg(test)]
mod dtype;
mod elementwise;
mod indexing;
mod matmul;
mod reduce;
mod spatial;
mod storage;

use copy::*;
use indexing::*;
use matmul::*;
use reduce::*;
use spatial::*;
