//! Shared test utilities for the quantised matmul tests: the dtype sweep, the
//! diagnostics and the tolerances.
//!
//! The GEMX tensor-core kernels are checked against a baseline that dequantises
//! the same weights and runs an ordinary matmul. The files here build those
//! inputs, compare the results, and hold the diagnostics that run when a
//! comparison fails. The sweep covers each GEMX input type, identified by its Y
//! code, at batch sizes 1, 8, 64 and 128.
//!
//! - `compare`: error statistics and tolerance checks.
//! - `diag_accumulation`: batch-dependent accumulation diagnostics.
//! - `diag_basic`: matmul mapping diagnostics.
//! - `diag_columns`: column mapping diagnostics.
//! - `diag_fp8`: FP8 activation diagnostics.
//! - `diag_krow`: K-row read diagnostics.
//! - `diag_layout`: repacked byte layout diagnostics.
//! - `diag_runners`: runners that execute the diagnostics at each batch size.
//! - `diag_tiles`: pipeline component diagnostics.
//! - `harness`: serialisation and watchdog for the GPU tests.
//! - `inputs`: deterministic weights and activations.
//! - `repack`: validation of the K/128 repacked format.
//! - `run`: the dtype sweep.
//! - `tolerance`: fixed tolerances per format.
//! - `tests_negative`: tests that the checks fire.

// Reference implementations for the GEMX kernels: the `* 1` / `+ 0` terms and
// the explicit row/col index loops spell out the addressing the kernel does, so
// a layout bug reads as an index rather than an opaque iterator chain.
#![allow(
    clippy::identity_op,
    clippy::missing_const_for_thread_local,
    clippy::needless_range_loop,
    clippy::single_match,
    clippy::unnecessary_sort_by
)]

use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
use candle::Module;
#[cfg(feature = "cuda")]
use candle_transformers::models::quantized_matmul::QMatMul as QMatMulWrapper;

/// Input data types to test - all supported GEMX Y types
/// Y type mapping: 0=F16, 1=BF16, 2=F8E4M3, 3=F32
#[cfg(feature = "cuda")]
pub const TEST_DTYPES: &[DType] = &[DType::BF16, DType::F16, DType::F32, DType::F8E4M3];

/// Test configuration for a single quant type test
#[derive(Clone)]
pub struct QuantTestConfig {
    #[allow(dead_code)] // Used only in cuda feature
    pub name: &'static str,
    pub dtype: GgmlDType,
    pub nrows: usize,
    pub ncols: usize,
}

impl QuantTestConfig {
    pub fn new(name: &'static str, dtype: GgmlDType) -> Self {
        // K-quants require dimensions divisible by 256
        // Use realistic dimensions: K=2048 (8 quant blocks), N=2048
        Self {
            name,
            dtype,
            nrows: 2048,
            ncols: 2048,
        }
    }

    /// Create config with small dimensions for initial testing
    #[cfg(feature = "cuda")]
    pub fn small(name: &'static str, dtype: GgmlDType) -> Self {
        // Minimal K-quant dimensions: 256x256 (1 quant block)
        Self {
            name,
            dtype,
            nrows: 256,
            ncols: 256,
        }
    }
}

/// Batch sizes to test for corruption detection
/// Reduced set for speed - tests key boundaries: 1 (baseline), 8 (small), 64 (tile boundary), 128 (large)
#[cfg(feature = "cuda")]
pub const TEST_BATCH_SIZES: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 16, 64, 128];

mod compare;
mod diag_accumulation;
mod diag_basic;
mod diag_columns;
mod diag_fp8;
mod diag_krow;
mod diag_layout;
mod diag_runners;
mod diag_tiles;
mod harness;
mod inputs;
mod repack;
mod run;
mod tests_negative;
mod tolerance;

pub use compare::*;
pub use diag_accumulation::*;
pub use diag_basic::*;
pub use diag_columns::*;
pub use diag_fp8::*;
pub use diag_krow::*;
pub use diag_layout::*;
pub use diag_runners::*;
pub use diag_tiles::*;
pub use harness::*;
pub use inputs::*;
pub use repack::*;
pub use run::*;
pub use tests_negative::*;
pub use tolerance::*;
