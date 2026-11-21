//! The CUDA storage for quantized tensors, and the kernels that run on it.
//!
//! `QCudaStorage` owns or leases quantized device bytes, and each file covers one
//! concern built on it: the q8a128 operand the int8 matmul reads, the dense and
//! grouped products that consume it, the MoE router and scatter, the repack that
//! builds KO twins at load, and the KV palette, sampling and selection kernels
//! that choose block formats. Operands and storages are carved from the arena
//! their data lives in, so a result inherits the lifetime of its input, and every
//! launch picks a kernel its caller can check.
//!
//! - `activations_q8a128`: quantizes activations and fuses producers into the int8 operand.
//! - `q8a128_operand`: the operand's backing, shape, sum convention and borrowing.
//! - `quantize`: quantizes float data, with fused transposes for KV chunks.
//! - `dequantize`: expands blocks to float, refusing formats without a kernel.
//! - `loading`: uploads weights into owned storage or weight-zone slots.
//! - `storage_io`: creates, copies and reads quantized storage.
//! - `host_memory`: pinned, device-mapped allocations and their guards.
//! - `device_info`: cached SM and L2 counts, and VRAM queries.
//! - `matmul_methods`: picks the vector or MMQ kernel, and refuses formats MMQ gets wrong.
//! - `dense_matmul`: dense int8 and float products over one activation.
//! - `grouped_grid`: grid axis order and tile height for grouped launches.
//! - `grouped_matmul`: one-launch products over several experts.
//! - `bucketize`: device tables for the grouped expert pipeline, from one launch.
//! - `moe_routing`: the router, the token gather and the deterministic scatter.
//! - `repack_host`: builds KO twins from compact sources, a band at a time.
//! - `kv_format_select`: GPU format selection for KV blocks.
//! - `kv_sampling`: quantization error sampling and per-threshold summaries.
//! - `kv_palette`: the palette-4 head record and its conversion.

// Kernel-launch wrappers mirror their CUDA signatures argument for argument —
// pointers, extents, strides, scales, stream. Grouping them into structs would
// put a shape between the call site and the kernel it is a transcription of,
// which is the opposite of what makes these auditable against the `.cu`.
#![allow(clippy::too_many_arguments)]

use super::decode_rows::DecodeRows;
use super::{GgmlDType, Int8Mode, QStorage, SumScale};
use crate::backend::{BackendDevice, BackendStorage};

use std::ffi::CStr;
use std::sync::Arc;

use crate::cuda_backend::alloc_inheriting;
use crate::cuda_backend::graph::outside_capture;
use crate::cuda_backend::last_cuda_kernel_launch;
use crate::cuda_backend::load_scratch;
use crate::cuda_backend::stream_alloc;
use crate::cuda_backend::wave_provenance::{self, wave_alloc_attributed, LeaseOrigin};
use crate::cuda_backend::Backing;
use crate::cuda_backend::INHERIT_ALIGN;
use crate::quantized::k_quants::GgmlType;
use crate::quantized::ko_quant::ko_tileable;
use crate::LiveTensor;
use crate::{CudaDevice, CudaStorage, DType, Layout, Result, Shape, Storage};
use half::{bf16, f16};

use crate::cuda_backend::WrapErr;
use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut};

// Import the FFI dispatcher functions
use candle_kernels::simple::quantized::{
    run_dequantize_block, run_dequantize_ko, run_dequantize_mul_mat_vec, run_dequantize_q8a128,
    run_mul_mat, run_mul_mat_vec_q8_1, run_quantize_block, run_quantize_ko,
    run_quantize_palette4_convert, run_quantize_q8_1, run_quantize_q8a128,
    run_quantize_transposed_batched, run_quantize_transposed_batched_typed,
    run_reduce_head_stats_format, run_sample_quant_errors_kv_paged, run_sample_quant_errors_paged,
    run_select_kv_format_palette4_paged, run_select_winners_kv_paged,
    run_summarize_winners_side_paged, DequantOutDType, QType,
};

// Fused RMSNorm → q8a128 producer epilogue (B1/B3/B5).
use candle_kernels::simple::reduce::run_rmsnorm_q8a128_op;

// Fused SwiGLU → q8a128 producer epilogue (B4).
use candle_kernels::simple::fused_silu_mul::run_silu_mul_q8a128_op;

// Import the new quantized matmul dispatcher
// K/128 blocks have embedded scales, no external scale extraction needed.
use candle_kernels::quantized::{
    dispatch_info, flush_l2_cache, qmm_last_launch_error, qmm_last_launch_site,
    run_grouped_quantized_matmul, run_qkv_segmented_matmul, run_quantized_matmul, MatmulStatus,
    OutDType, VxSegment, YType, SPLITK_MAX_SEGS,
};
/// The wait state of a grouped GEMM over a live weight table — see
/// [`grouped_qmatmul_dev_q8a128`].
pub use candle_kernels::quantized::{MoeLive, STALL_WORDS};

// Import GEMX repacking dispatcher
use candle_kernels::quantized::{get_repacked_size_bytes, is_gemx_supported, run_repack_gemx};

use super::int8_matmul_mode::{q8a128_dense_tile, q8a128_dense_use_mode2, DenseTile};
use super::int8_split_k::{q8a128_dense_plan, DensePlan};
use super::table_ring::table_ring;

#[cfg(test)]
mod graph_bench;
mod narrow;
mod palette4_block;
#[cfg(test)]
mod prefill_tile_tests;
mod silu_matmul;
mod split_k;
use narrow::q8a128_dense_matmul_narrow_segmented;
use palette4_block::select_palette4_owned;
pub use palette4_block::{
    select_palette4_in, select_palette4_inputs, select_palette4_launch,
    select_palette4_launch_staged, Palette4Inputs, Palette4Layout, Palette4Result,
};
pub(crate) use silu_matmul::q8a128_dense_matmul_silu;
pub use split_k::ensure_split_k_scratch;
use split_k::q8a128_dense_matmul_split_k_segmented;

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Debug)]
pub struct QCudaStorage {
    /// `ManuallyDrop` so [`Drop`] can move the slice out without constructing a
    /// stand-in. `leak` is by-value while `drop` has only `&mut self`, and the
    /// stand-in used to come from `upgrade_device_ptr(0, 0)` — which is not
    /// free: under cudarc's event tracking (on for every production device) it
    /// creates and destroys two `CudaEvent`s and can fail, and it `unwrap`s.
    /// Fallible CUDA work on a drop path panics *inside* `Drop`, which aborts
    /// during an unwind. `CudaStorage` avoids this with its `Empty` slice
    /// variant; this is the same trick for a struct that has no spare variant.
    ///
    /// Every read still goes through `Deref`, so `self.data.inner` is unchanged
    /// at the ~18 use sites. The obligation this adds is that **every path that
    /// destroys or replaces `data` must dispose of the old value explicitly** —
    /// [`Drop`] and [`Self::quantize`] are the only two.
    data: std::mem::ManuallyDrop<PaddedCudaSlice>,
    dtype: GgmlDType,
    device: CudaDevice,
    /// Whether dropping this storage may free its device memory.
    /// [`Backing::Owned`] for everything allocated here — which is everything
    /// except the slot views built by [`QCudaStorage::from_leased_device_ptr`].
    backing: Backing,
}

impl Clone for QCudaStorage {
    /// A clone is always **owned**, never a second lease.
    ///
    /// `CudaSlice::clone` is a device-to-device copy, so a clone has its own
    /// allocation regardless of what the source was. Carrying `Lease` across
    /// would leak that fresh allocation on every clone.
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            backing: Backing::Owned,
        }
    }
}

impl Drop for QCudaStorage {
    fn drop(&mut self) {
        // Exhaustive rather than `!= Lease`, matching `CudaStorage::drop`: a
        // future `Backing` variant must state which side it falls on here rather
        // than silently inheriting the free path.
        match self.backing {
            Backing::Owned => {
                // Dispose normally. `data` is `ManuallyDrop`, so without this the
                // allocation is never freed.
                // SAFETY: `self` is being destroyed; `data` is not read again and
                // is dropped exactly once (this arm and the lease arm below are
                // mutually exclusive and both return).
                unsafe { std::mem::ManuallyDrop::drop(&mut self.data) };
                return;
            }
            Backing::Lease(_) => {}
        }
        // Same discipline as `CudaStorage::drop`: calling `leak` rather than
        // merely suppressing the drop is load-bearing — it waits on the
        // slice's read/write events, destroys them, and decrements the
        // stream's `Arc`. Bare suppression would strand two `CudaEvent`s and a
        // stream refcount per lease, and leases are minted per band per
        // forward.
        //
        // SAFETY: `self` is being destroyed and `data` is never read again;
        // `ManuallyDrop::take` is the move-out that `leak`'s by-value signature
        // needs, and it touches no CUDA API on the way (see the field's note).
        let data = unsafe { std::mem::ManuallyDrop::take(&mut self.data) };
        let ptr = data.inner.leak();
        // An arena that frees as its storages die takes the range back; every
        // other arena ignores this — as `CudaStorage::drop`.
        if let Backing::Lease(LeaseOrigin::Wave(ticket)) = self.backing {
            wave_provenance::wave_release(ticket, ptr);
        }
    }
}

impl QCudaStorage {
    /// Wrap `elem_count` elements of quantized device memory at `ptr` as a
    /// storage that does **not** own them.
    ///
    /// This is the quantized counterpart of
    /// [`CudaStorage::from_leased_device_ptr`], and it exists for exactly one
    /// caller: a KV arena slot. Under size classes an arena is a run of untyped
    /// byte slots, so the quantize / dequantize kernels can no longer be handed
    /// "the arena's QTensor" — they are handed a view of one slot instead. See
    /// `docs/archived/arena_unification.md` principle 8.
    ///
    /// The view carries **no matrix-row padding**, unlike
    /// [`Self::zeros`]: a slot is exactly its own bytes and the next slot
    /// belongs to another chunk. That makes it unsuitable for the matmul
    /// kernels, which read into the padding — and correct for the block
    /// quantize / dequantize paths, which do not.
    ///
    /// # Safety
    /// `ptr` must point to at least `ceil(elem_count / block_size) * type_size`
    /// bytes of device memory that stays live, and un-aliased for writes, for
    /// the storage's lifetime.
    pub unsafe fn from_leased_device_ptr(
        ptr: u64,
        elem_count: usize,
        dtype: GgmlDType,
        device: &CudaDevice,
        origin: LeaseOrigin,
    ) -> Result<Self> {
        if !elem_count.is_multiple_of(dtype.block_size()) {
            crate::bail!(
                "leased quantized view of {elem_count} elements is not a whole number of \
                 {dtype:?} blocks ({})",
                dtype.block_size()
            );
        }
        let size_in_bytes = (elem_count / dtype.block_size()) * dtype.type_size();
        let inner = device
            .cuda_stream()
            .upgrade_device_ptr::<u8>(ptr, size_in_bytes);
        Ok(QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing: Backing::Lease(origin),
        })
    }
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;

pub const MMQ_X_Q4_0_AMPERE: usize = 4;

pub const MMQ_Y_Q4_0_AMPERE: usize = 32;

pub const NWARPS_Q4_0_AMPERE: usize = 4;

pub const GGML_CUDA_MMV_X: usize = 32;

pub const GGML_CUDA_MMV_Y: usize = 1;

pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;

pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;

pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// Convert GgmlDType to QType for dispatcher
pub(crate) fn dtype_to_qtype(dtype: GgmlDType) -> Result<QType> {
    Ok(match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q2_K => QType::Q2_K,
        GgmlDType::Q3_K => QType::Q3_K,
        GgmlDType::Q4_K => QType::Q4_K,
        GgmlDType::Q5_K => QType::Q5_K,
        GgmlDType::Q6_K => QType::Q6_K,
        GgmlDType::Q8_K => QType::Q8_K,
        GgmlDType::QAWQ => QType::QAWQ,
        GgmlDType::QAWQ_G64 => QType::QAWQ_G64,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        GgmlDType::Q0_V => QType::Q0_V,
        GgmlDType::Q1_A => QType::Q1_A,
        GgmlDType::Q0_X => QType::Q0_X,
        GgmlDType::Q0_M2 => QType::Q0_M2,
        GgmlDType::Q4_KO => QType::Q4_KO,
        GgmlDType::Q5_KO => QType::Q5_KO,
        GgmlDType::Q6_KO => QType::Q6_KO,
        GgmlDType::Q8_KO => QType::Q8_KO,
        GgmlDType::MXFP4_KO => QType::MXFP4_KO,
        GgmlDType::Q2_KO => QType::Q2_KO,
        GgmlDType::Q3_KO => QType::Q3_KO,
        GgmlDType::Q0_M4 => QType::Q0_M4,
        _ => crate::bail!("unsupported dtype for quantized op: {dtype:?}"),
    })
}

/// Kernel bounds of the MoE bucketize, re-exported so every gate that decides
/// "can this routing take the device-table path?" reads the SAME constants the
/// kernel and its wrapper validate against — a hand-copied bound here is the
/// one place the mirrored-constant scheme couldn't reach.
pub use candle_kernels::simple::moe_bucketize::{
    MAX_EXPERTS as MOE_MAX_EXPERTS, MAX_TOPK as MOE_MAX_TOPK,
};

#[cfg(test)]
#[path = "../cuda_tests/mod.rs"]
mod test;

#[cfg(test)]
#[path = "../padding_tail_tests.rs"]
mod padding_tail_tests;

#[cfg(test)]
#[path = "../q0_v_decode_oracle_tests.rs"]
mod q0_v_decode_oracle_tests;

#[cfg(test)]
#[path = "../q0_v_encode_oracle_tests.rs"]
mod q0_v_encode_oracle_tests;

#[cfg(test)]
#[path = "../kv_block_oracle_tests.rs"]
mod kv_block_oracle_tests;

mod activations_q8a128;
mod bucketize;
mod dense_matmul;
mod dequantize;
mod device_info;
mod grouped_grid;
mod grouped_matmul;
mod host_memory;
mod kv_format_select;
mod kv_palette;
mod kv_sampling;
mod loading;
mod matmul_methods;
mod moe_routing;
mod q8a128_operand;
mod quantize;
mod repack_host;
mod storage_io;

pub use activations_q8a128::*;
pub use bucketize::*;
pub use dense_matmul::*;
pub use dequantize::*;
pub use device_info::*;
pub use grouped_grid::*;
pub use grouped_matmul::*;
pub use host_memory::*;
pub use kv_format_select::*;
pub use kv_palette::*;
pub use kv_sampling::*;
pub use loading::*;
pub use moe_routing::*;
pub use q8a128_operand::*;
pub use quantize::*;
pub use repack_host::*;
