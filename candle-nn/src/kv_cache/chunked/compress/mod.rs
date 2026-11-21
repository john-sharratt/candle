//! Adaptive per-`(head, palette)` quantization of sealed sequences on the GPU,
//! the reference seal that the gap quantize is held to, byte for byte.
//!
//! The palette-4 selection kernel picks a format for each `(chunk, head,
//! palette)` from the policy's candidate list, under the production error
//! thresholds. The convert kernel then writes the quantized bytes into GPU
//! quant arenas, and the sequences it returns stay in VRAM with no copy to the
//! CPU. Eligible chunks have their source bands in GPU float or R16 arenas.
//! Other chunks, typically cold-loaded ones already in mixed formats, pass
//! through in order. `head_dim` must divide evenly into the palettes, since
//! each palette band is a fixed count of 32-element blocks.
//!
//! - `band_layout`: band addresses, layouts and palette maps.
//! - `dequantize`: the reverse convert back to F16 arenas.
//! - `layer_quantize`: the cross-layer selection and the deferred convert.
//! - `quantize_impl`: the per-sequence quantize and reassembly.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::{
    dtype_to_ggml_float, identity_pal_map, PalHeadDesc, FULL_CHUNK_WINDOW,
};
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::{Generation, PinnedBuf};
#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::{DType, Device, Result};

#[cfg(feature = "cuda")]
use super::arena::{ArenaKey, ArenaStorageState};
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::band_addrs::BandAddrs;
#[cfg(feature = "cuda")]
use super::compression_policy::{
    CompressionPolicy, PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
};
#[cfg(feature = "cuda")]
use super::gid_pool::ChunkGid;
#[cfg(feature = "cuda")]
use super::head_gids::{ChunkBands, HeadGids, GIDS_PER_HEAD};
#[cfg(feature = "cuda")]
use super::meta_pool::ChunkRecordSrc;
#[cfg(feature = "cuda")]
use super::sampled_selection::{PagedSelectionGpuInputs, SampleFormat};
#[cfg(all(test, feature = "cuda"))]
use super::selection_batches::{selection_batches, selection_bytes, slice_layers};
#[cfg(feature = "cuda")]
use super::types::{SealedChunk, SealedSequence, CHUNK_SIZE};
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::{ArenaFormatTag, ArenaLocation, N_PALETTE};
#[cfg(feature = "cuda")]
use crate::kv_cache::{KvFormat, QuantFormat};

/// Adaptive per-(head, palette) quantization that runs entirely on GPU
/// and returns `SealedSequence`s whose chunks still live in VRAM.
///
/// Eligible chunks land in fresh GPU Q-format arenas chosen by the
/// palette-4 selection kernel. Ineligible chunks (those whose source
/// GIDs are not in GPU `Float` / `Quantized(R16)` arenas — typically
/// cold-loaded chunks already in mixed Q-formats) flow through a
/// pass-through "preserve" bucket and get merged back unchanged in
/// the output sequence, preserving original chunk order.
///
/// **No DMA to CPU.** Callers that need warm-tier storage feed the
/// returned GPU sequences to [`ChunkedKvBacking::migrate_sealed_to_cpu_batch_async`]
/// for a separate, format-preserving DtoH copy. The split keeps the
/// in-session reconcile purely on-GPU (decode reads the convert
/// kernel's output bytes directly, no intervening `kv_migrate` scatter).
#[cfg(feature = "cuda")]
/// Single-layer entry point: quantize + launch the convert immediately. See
/// [`quantize_sealed_in_place_deferred`] for the cross-layer batched form.
pub fn quantize_sealed_in_place(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
) -> Result<Vec<SealedSequence>> {
    quantize_sealed_in_place_impl(
        backing,
        sequences,
        policy,
        device,
        copy_stream,
        pinned_scratch,
        None,
        None,
        None,
    )
}

mod band_layout;
mod dequantize;
mod layer_quantize;
mod quantize_impl;

pub(super) use band_layout::*;
pub use dequantize::*;
pub use layer_quantize::*;
use quantize_impl::*;
