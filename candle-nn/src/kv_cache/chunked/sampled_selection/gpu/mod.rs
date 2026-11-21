//! GPU selection of KV quantization formats and error sampling for paged chunks.
//!
//! The selection kernel scores every block against each candidate format and
//! chooses per head and palette slot, and the sampler summarises the errors for
//! each threshold, so the decision is made on the device and only compact results
//! come back to the host. The inputs are paged: the caller hands over the chunk
//! tables and the selection addresses the blocks in place, so no per-chunk copy is
//! made to bring them together. Every table that the kernel reads asynchronously
//! is owned or guarded, so it cannot be recycled under a running launch. A call
//! selects per chunk, so a partial chunk's valid token range is passed in and the
//! count-normalised metrics account for it.
//!
//! - `error_surface`: the fused K and V error-surface launch.
//! - `paged_inputs`: staged selection inputs and the palette-4 selection over them.
//! - `sampler`: the asynchronous threshold sampler and its pending summaries.
//! - `staging`: the staging path each device table takes.

use super::super::bump_arena::NOT_A_WAVE;
use super::params::SELECT_BLOCK;
use super::profile::sampled_profile_record_duration;
use super::CompressionSummary;
use super::{ErrorSurface, SampleFormat, SampleSide, SampledSelectionBenchmarkResult};
use crate::kv_cache::arena_table::PerHeadEntry;
use crate::kv_cache::chunked::backing::BackingInner;
use crate::kv_cache::chunked::head_gids::{ChunkBands, GIDS_PER_HEAD};
use crate::kv_cache::QuantFormat;
use crate::kv_cache::{ChunkedKvBacking, KvFormat, GID_STRIDE, N_PALETTE};
use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use candle::quantized::{
    cuda::{
        reduce_head_format_stats, sample_quant_errors_kv_paged_staged, sample_quant_errors_paged,
        select_and_summarize_kv_winners_paged_staged, select_palette4_in, Palette4Layout,
        Palette4Result,
    },
    pinned_staging::{Generation, GpuBuf, PinnedBuf},
    GgmlDType,
};
use candle::Result;
use std::sync::{Arc, Mutex};

pub enum SelectionBackend<'a> {
    Cpu,
    Gpu(&'a PagedSelectionGpuInputs),
}

#[derive(Debug, Clone)]
pub struct GpuQualityAggregation {
    pub k_head_formats: Vec<SampleFormat>,
    pub v_head_formats: Vec<SampleFormat>,
    pub k_worst_case_formats: Vec<SampleFormat>,
    pub v_worst_case_formats: Vec<SampleFormat>,
    pub k_palette4_formats: Vec<SampleFormat>,
    pub v_palette4_formats: Vec<SampleFormat>,
}

pub struct PagedSelectionGpuInputs {
    // Live per-block GIDs keep the arena chunk allocations resident; the band
    // format tags beside them are what the selection table's sub-entries are
    // built from.
    chunk_gids_keepalive: Vec<ChunkBands>,
    // Host copy of the selection table, one `PerHeadEntry` row per
    // `(chunk, head)` in `chunk_gids_keepalive` order. Kept host-side for the
    // same reason `head_gids` is: `select_chunks` narrows to a chunk sub-range
    // and must re-upload **both** buffers. Sharing the parent's table there was
    // correct only while rows were keyed by arena — under a `(chunk, head)`
    // key, chunk `start + i` of the slice would read the parent's row `i`.
    per_head_table: Vec<i64>,
    per_head_table_buf: GpuBuf,
    head_gids: Vec<i64>,
    head_gids_buf: GpuBuf,
    blocks_per_chunk: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    dev: candle::CudaDevice,
    /// Keeps the transient span's cursor from rewinding while
    /// `per_head_table_buf` / `head_gids_buf` still point into it.
    ///
    /// `None` when the buffers came from a caller-supplied pinned generation
    /// or a plain allocation — in both cases they own or borrow memory this
    /// struct does not have to pin.
    _bump_gen: Option<super::super::bump_arena::Generation>,
    /// The palette-4 selection's block — inputs, outputs and readback source —
    /// carved from `_bump_gen` on first use and reused by every selection over
    /// these inputs: each one synchronises before it returns, so the block is
    /// idle between them. `(address, bytes)`.
    palette_block: Mutex<Option<(u64, usize)>>,
    /// A block taken from the driver, for inputs staged without a span.
    palette_owned: Mutex<Option<CudaSlice<u8>>>,
}

mod error_surface;
mod paged_inputs;
mod sampler;
mod staging;

pub use error_surface::*;
pub use sampler::*;
use staging::*;
