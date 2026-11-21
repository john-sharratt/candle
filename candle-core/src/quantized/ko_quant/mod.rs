//! F32 and MXFP4 weights repacked into the lane-major KO chunks the int8 matmul
//! kernels read.
//!
//! KO weights carry a per-128 affine `(scale, min)` that K-quant storage does not
//! hold, so the affine formats are quantized from F32 in one offline pass, while
//! MXFP4_KO is repacked from its own nibbles and scales without requantizing. Each
//! chunk covers 8 rows by 128 K, laid out so each lane's sub-words are contiguous
//! for the kernel's wide loads. `ALL_KO_DTYPES` lists every format `GgmlDType::is_ko`
//! accepts, so load and dispatch sweeps cover the same set, and `ko_tileable`
//! answers from shape alone whether a weight fits the matmul tiling.
//!
//! - `affine_ko`: the per-128 affine KO codecs and their plane stack.
//! - `mxfp4_ko`: the MXFP4_KO repack and its reference int8 matmul.
//!
//! Tests:
//!
//! - `tests_affine`: raw-byte and round-trip tests for the affine codecs.
//! - `tests_mxfp4`: MXFP4_KO matmul and repack tests.

use crate::quantized::k_quants::{e8m0_to_f32_half, BlockMXFP4, GgmlType, MXFP4_KVALUES};
use crate::quantized::GgmlDType;
use half::f16;
use rayon::prelude::*;

/// Bytes in one MXFP4_KO k1024 chunk (8 rows × 128 K): 512 B of nibbles (the reordered
/// MXFP4 `qs`) + 32 B of per-sub E8M0 scales (4 per row).
pub const MXFP4_KO_CHUNK_BYTES: usize = 512 + 32;

/// Every KO weight format — exactly the set [`GgmlDType::is_ko`] answers `true` for.
///
/// **A list, because a predicate cannot be iterated**, and several places need to sweep the
/// formats rather than test one: the GGUF load path, the dispatch-table gates, the fused-QKV
/// format map. Each of those used to carry its own hand-written list, and each list stopped
/// covering the set at a different point — which is how `Q2_KO` shipped unable to load from a
/// GGUF at all. `ko_dtype_list_matches_is_ko` holds this in step with the predicate, so the
/// sweeps derive their coverage instead of restating it.
pub const ALL_KO_DTYPES: &[GgmlDType] = &[
    GgmlDType::Q2_KO,
    GgmlDType::Q3_KO,
    GgmlDType::Q4_KO,
    GgmlDType::Q5_KO,
    GgmlDType::Q6_KO,
    GgmlDType::Q8_KO,
    GgmlDType::MXFP4_KO,
];

/// Whether a weight `[nrows × ncols]` fits the KO **matmul** tiling.
///
/// A stricter bound than the KO *storage* chunk, which packs 8 rows × 128 K: the
/// q8a128 / MXFP4_KO matmul kernels tile N in blocks of 32, so a weight whose
/// `nrows` is 8, 16, or 24 mod 32 packs fine and the matmul still rejects it
/// (Qwen3.5-0.8B's DeltaNet `w_alpha`/`w_beta` are `[16, hidden]`; the mHC `fn_w`
/// is `mix_hc = 24`).
///
/// Whether a weight tiles is a **shape fact, knowable before any work happens** —
/// callers test it up front to route a sub-tile weight to the dense path, which is
/// what keeps an error out of `repack_ko` from having to double as that signal.
pub const fn ko_tileable(nrows: usize, ncols: usize) -> bool {
    nrows.is_multiple_of(32) && ncols.is_multiple_of(128)
}

/// Bytes in one MXFP4_KO **GPU** k1024 chunk (matches `block_c_mxfp4_k1024`): the 544-byte
/// CPU chunk (`ql` + per-sub `e`) plus 32 B of the per-row collapsed scale `dm` (8 × `half2`).
pub const MXFP4_KO_GPU_CHUNK_BYTES: usize = MXFP4_KO_CHUNK_BYTES + 32;

/// Where each plane starts inside a chunk, and where the value's bits live.
///
/// Derived from [`ko_params`] alone — see its table for why the shifts are a function of which
/// planes are present rather than of the dtype.
struct KoPlanes {
    maxq: i32,
    ql_bytes: usize,
    crumb_base: usize,
    crumb_bytes: usize,
    hi_base: usize,
    hi_bytes: usize,
    dm_base: usize,
    chunk_bytes: usize,
    /// Bit position of the crumb plane within the value.
    crumb_shift: u32,
    /// Bit position of the `hi` plane within the value.
    hi_shift: u32,
}

impl KoPlanes {
    fn of(dtype: GgmlDType) -> Self {
        let (maxq, ql_bytes, crumb_bytes, hi_bytes) =
            ko_params(dtype).unwrap_or_else(|| panic!("not an affine KO dtype: {dtype:?}"));
        let crumb_shift = if ql_bytes > 0 { 4 } else { 0 };
        let hi_shift = crumb_shift + if crumb_bytes > 0 { 2 } else { 0 };
        Self {
            maxq,
            ql_bytes,
            crumb_base: ql_bytes,
            crumb_bytes,
            hi_base: ql_bytes + crumb_bytes,
            hi_bytes,
            dm_base: ql_bytes + crumb_bytes + hi_bytes,
            chunk_bytes: ql_bytes + crumb_bytes + hi_bytes + 32,
            crumb_shift,
            hi_shift,
        }
    }
}

/// Bytes in one k1024 chunk (8 rows × 128 K) for a KO dtype.
pub fn ko_chunk_bytes(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q2_KO => 256 + 32,       // crumb (2-bit values) + dm
        GgmlDType::Q3_KO => 256 + 128 + 32, // crumb + hi + dm
        GgmlDType::Q4_KO => 512 + 32,       // ql + dm
        GgmlDType::Q5_KO => 512 + 128 + 32, // ql + hi + dm
        GgmlDType::Q6_KO => 512 + 256 + 32, // ql + crumb + dm
        GgmlDType::Q8_KO => 1024 + 32,      // b0 + b1 + dm
        GgmlDType::MXFP4_KO => MXFP4_KO_GPU_CHUNK_BYTES, // 512 ql + 32 e + 32 dm
        _ => 0,
    }
}

mod affine_ko;
mod mxfp4_ko;
#[cfg(test)]
mod tests_affine;
#[cfg(test)]
mod tests_mxfp4;

pub use affine_ko::*;
pub use mxfp4_ko::*;
