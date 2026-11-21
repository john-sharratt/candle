//! The Rust bindings to the CUDA quantization kernels, linked from the kernel archive that `build.rs` builds.
//!
//! The quant type numbers are shared by four sources that must agree: the CUDA enums, the Rust GGML types, the
//! arena formats and the selection kernel's format tags. A test locks them. The dispatch numbering is distinct
//! from the matmul dispatcher's, because the two sets of kernels look their tables up differently.
//!
//! - `ffi_dequant`: the extern declarations of the quantization and dequantization kernels.
//! - `tests_qtype`: the test that locks the quant type values.

use std::ffi::c_int;
use std::ffi::c_void;

// Quant type enum for dispatcher functions

/// Quantization type enum for the KV-quant / palette dispatcher.
///
/// Values MUST match:
///   - The C++ `QType` enum in `candle-kernels/src/quantized/block_compact.cuh`
///     (locked in via `static_assert` in that file).
///   - `GgmlDType` in `candle-core/src/quantized/mod.rs` (`#[repr(u32)]`).
///   - `ArenaFormat::*` in `candle-kernels/src/arena_table.cuh`.
///   - `SELECT_FMT_*` in `candle-kernels/src/quantize/select_kv_format.cuh`.
///
/// ⚠ Distinct from `crate::quantized::QType` (in `quantized/api.rs`),
/// which uses a smaller 0..13 numbering for the `run_quantized_matmul`
/// dispatcher. The two enums are intentionally separate because the
/// underlying CUDA dispatchers use different kernel lookup tables — see
/// the top-level QType audit comment in `api.rs` for the reasoning.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QType {
    R16 = 3,
    P2 = 4,
    QAWQ = 5,
    QAWQ_G64 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q8_K = 9,
    Q8_KS = 10,
    Q6_K = 11,
    Q5_0 = 12,
    Q5_1 = 13,
    Q5_K = 14,
    Q4_0 = 15,
    Q4_1 = 16,
    Q4_K = 17,
    Q4_KS = 18,
    Q3_0 = 19,
    Q3_1 = 20,
    Q3_K = 21,
    Q2_0 = 22,
    Q2_1 = 23,
    Q2_K = 24,
    Q2_S = 25,
    Q2_A = 26,
    Q1_S = 27,
    Q0_V = 28,
    Q1_A = 29,
    Q0_X = 30,
    Q0_M2 = 31,
    Q0_M4 = 32,
    Q0 = 33,
    F8E4M3 = 34,
    F8E5M2 = 35,
    // Kernel-only activation QType past the GgmlDType-aligned range (no GgmlDType
    // counterpart, so it is absent from the GgmlDType lock). q8a128 is the contiguous
    // q8 activation block; mirrors QTYPE_Q8A128V/X in block_compact.cuh. Two modes,
    // same block, (eventually) different layouts: V = mode-1, X = mode-2 (weight-reuse).
    Q8A128V = 36,
    Q8A128X = 37,
    // Lane-major per-128 affine ("ordered") weight twins of the K-quant blocks for the
    // q8a128 int8 path. GPU-only weight layouts; mirror QTYPE_Q*_KO in block_compact.cuh.
    // All four have standalone quantize/dequant kernels (run_quantize_ko/run_dequantize_ko).
    Q4_KO = 45,
    Q5_KO = 46,
    Q6_KO = 47,
    Q8_KO = 48,
    // Lane-major per-sub MXFP4 for the q8a128 int8 path — the KO twin the MXFP4 routed
    // experts repack to (stays 4-bit; one int32 MMA per 32-K sub, each folded with its own
    // E8M0 scale in FP). Value 50 mirrors QTYPE_MXFP4_KO. Native MXFP4 (49) has no matmul kernel.
    MXFP4_KO = 50,
    /// Lane-major per-128 affine KO twin at 2-bit — the smallest KO weight (value 0..3, the
    /// 2-bit crumb region Q6_KO carries, used as the whole value). Value 51 mirrors QTYPE_Q2_KO.
    Q2_KO = 51,
    /// Lane-major per-128 affine KO twin at 3-bit (value 0..7) — `Q3_K`'s same-width twin: the
    /// 2-bit crumb plane at bits 0-1 plus the 1-bit hi plane at bit 2, no `ql` plane. Value 52
    /// mirrors QTYPE_Q3_KO.
    Q3_KO = 52,
}

/// Output dtype enum for dequantize dispatcher
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantOutDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

mod ffi_dequant;
#[cfg(test)]
mod tests_qtype;

pub use ffi_dequant::*;
