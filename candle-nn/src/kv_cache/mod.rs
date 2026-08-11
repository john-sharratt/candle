//! Key-Value cache implementations for transformer attention.
//!
//! This module provides several types of KV cache:
//!
//! - **Contiguous caches** (`Cache`, `KvCache`): Traditional dense storage that grows as needed
//! - **Chunked (paged) caches** (`ChunkedKvBacking`): Block-based storage with:
//!   - Arc-based prefix sharing (copy-on-write)
//!   - Per-sequence slot allocation
//!   - Efficient memory reuse via free lists
//! - **Rotating caches** (`RotatingCache`, `RotatingKvCache`): Fixed-size sliding window caches
//! - **Scattered caches** (`ScatteredKvCache`): Sparse caches with batch masking
//!
//! # Example
//!
//! ```ignore
//! use candle_nn::kv_cache::{KvCache, ChunkedKvBacking};
//!
//! // Create a contiguous cache
//! let mut cache = KvCache::new(2, 1024);
//!
//! // Or create a chunked (paged) cache for prefix sharing
//! let backing = ChunkedKvBacking::new(
//!     batch_size, n_kv_head, head_dim,
//!     chunk_size, arena_chunks,
//!     dtype, &device, max_seq_len
//! )?;
//! cache.set_chunked_backing(&backing, batch_idx, None)?;
//! ```

use candle::quantized::GgmlDType;
use candle::DType;

mod arena_table;
mod cache;
mod chunked;
mod rotating;

pub use arena_table::{
    ArenaFormatTag, ArenaLocation, PaletteSubEntry, PerHeadEntry, PerHeadTable, ResolvedArenaInfo,
    N_PALETTE,
};
pub use cache::{Cache, CacheIntegrityResult, KvCache};
pub use chunked::class_promotion_count;
#[cfg(feature = "cuda")]
pub use chunked::fletcher_golden::{fletcher32_golden, fletcher32_golden_on, GoldenRecord};
#[cfg(feature = "cuda")]
pub use chunked::persistence_domain_stats;
#[cfg(feature = "cuda")]
pub use chunked::slot_state_stats;
pub use chunked::wave_plan::{
    BufferShape, Encoding, LayerPhase, ModelGeometry, WaveBuffer, WavePlan, BUMP_ALIGNMENT,
};
#[cfg(feature = "cuda")]
pub use chunked::{
    begin_wave, end_wave_transient, plan_wave_transient, wave_domain_stats, BumpRange,
    WaveGeneration, WAVE_ATTN_BYTES, WAVE_FFN_BYTES, WAVE_FORWARD_BYTES,
};
/// The span's geometry, for the model loader that installs a weight side into it.
#[cfg(feature = "cuda")]
pub use chunked::{
    initial_weight_bytes, kv_spare_regions, set_claim_reserve, set_ground_broker, set_weight_floor,
    span_end, weight_capacity_bytes, weight_floor_after,
};
#[cfg(feature = "cuda")]
pub use chunked::{region_stats, RegionStats, REGION_BYTES};
/// The weight side of the reservation. Pure arithmetic, so it is available
/// whether or not the crate was built with a GPU backend.
pub use chunked::{
    RetractPlan, WeightZone, WeightZoneStats, INITIAL_KV_RESERVE, MIN_ELASTIC_RESERVE,
};

#[cfg(feature = "cuda")]
pub use chunked::migrate::HostSealedChunk;
#[cfg(feature = "cuda")]
pub use chunked::migrate::{kv_migrate, kv_migrate_on};
pub use chunked::migrate::{MigrationPlan, MigrationRecord};
pub use chunked::sampled_selection::SampleFormat;
#[cfg(feature = "cuda")]
pub use chunked::vram_budget_available;
pub(crate) use chunked::Arena; // Internal use only
pub use chunked::MIGRATION_STAGING_CAP_BYTES;
pub use chunked::{
    all_kv_formats, class_for_format, class_for_payload, elems_per_chunk, payload_bytes,
    payload_bytes_for_tag, SizeClass, GID_STRIDE, LADDER,
};
#[cfg(feature = "cuda")]
pub use chunked::{
    convert_deferred_descs, dequantize_sealed_in_place, quantize_layers_deferred,
    quantize_sealed_in_place, quantize_sealed_in_place_deferred,
};
pub use chunked::{global_arena_gpu_bytes, global_arena_memory_report, global_print_arena_table};
pub use chunked::{is_device_oom, KV_DEVICE_OOM_MARKER};
pub use chunked::{migrate_flight, migrate_in_flight, MigrateFlight};
pub use chunked::{
    production_adaptive_candidates, BlockAllocSpec, ChunkGid, ChunkGidPool, ChunkMeta,
    ChunkedKvBacking, ClassOccupancy, CompressionPolicy, GpuArenaClassStats, HeadGids,
    KvErrorThresholdFactors, LLAMA_KV_FACTORS, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_LEVEL_TIER, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS, QWEN3_8B_KV_FACTORS, QWEN3_MOE_KV_FACTORS,
};
pub use chunked::{ArenaKey, StoragePolicy};
pub use chunked::{LiveChunkRef, MetaGid, SealedChunk, SealedSequence, WriterTail, CHUNK_SIZE};
pub use rotating::{
    IndicesAndMask, RotatingCache, RotatingKvCache, ScatteredCacheBuilder, ScatteredKvCache,
};

/// The formats **unsealed (active)** KV chunks actually occupy, given the
/// session's configured *sealed* formats and whether the backing is on GPU.
///
/// A block does not reach its configured format until its turn seals and
/// quantizes. While a sequence is live, its K sits in `R16` on GPU — raw F16
/// **plus reserved Q-capture space, 128 bytes per 32 elements, i.e. twice plain
/// F16** — and its V sits in plain F16. The configured pair (say `Q4_0` + `Q8_0`
/// at 18 + 34 B) is what the same data costs *after* it settles.
///
/// The distinction is load-bearing for admission. Pricing a candidate at the
/// sealed cost understates what it occupies for its entire working life by
/// `192 / 52 ≈ 3.7x`: measured, a KV cache admission believed was ~1.5 GiB was
/// physically 5.7 GiB (R16 3520 MiB + F16 1840 MiB = 94% of it), which is what
/// pushed a 7.5 GiB expert cache plus KV past a 13.5 GiB ceiling and refused
/// arena after arena. Admission must reason in ACTIVE formats; only the
/// steady-state footprint of a finished turn is in sealed formats.
pub fn active_kv_formats(k_format: KvFormat, on_gpu: bool) -> (KvFormat, KvFormat) {
    match k_format {
        // A float-configured backing never quantizes on append: active == sealed.
        KvFormat::Float(dtype) => (KvFormat::Float(dtype), KvFormat::Float(dtype)),
        // GPU: K accumulates in R16 (raw + Q-capture space), V in plain F16.
        KvFormat::Quantized(_) if on_gpu => (
            KvFormat::Quantized(QuantFormat::R16),
            KvFormat::Float(candle::DType::F16),
        ),
        // CPU: active K stays plain float so partial-token appends need no
        // block-aligned quantization.
        KvFormat::Quantized(_) => (
            KvFormat::Float(candle::DType::F16),
            KvFormat::Float(candle::DType::F16),
        ),
    }
}

// ==================== Quantization Format Types ====================

/// Quantized storage format for KV cache.
///
/// This type represents the quantization scheme without exposing internal
/// implementation details (GGML types).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, strum_macros::EnumIter)]
pub enum QuantFormat {
    /// 4-bit quantization with FP16 scale per 32 elements.
    /// 18 bytes per 32 elements (0.5625 bytes/element).
    Q4_0,
    /// 4-bit quantization with FP16 scale and min per 32 elements.
    /// 20 bytes per 32 elements (0.625 bytes/element).
    Q4_1,
    /// 5-bit quantization with FP16 scale per 32 elements.
    /// 22 bytes per 32 elements (2 byte scale + 4 byte qh + 16 byte ql).
    Q5_0,
    /// 5-bit quantization with FP16 scale and min per 32 elements.
    /// 24 bytes per 32 elements (2 byte scale + 2 byte min + 4 byte qh + 16 byte ql).
    Q5_1,
    /// 8-bit quantization with FP16 scale per 32 elements.
    /// 34 bytes per 32 elements (1.0625 bytes/element).
    Q8_0,
    /// 8-bit quantization with FP16 scale and sum per 32 elements.
    /// 36 bytes per 32 elements (1.125 bytes/element).
    Q8_1,
    /// Q4 with attention-sink sub-block scaling, 20 bytes per 32 elements.
    Q4_KS,
    /// Q8 with attention-sink sub-block scaling, 36 bytes per 32 elements.
    Q8_KS,
    /// 2-bit symmetric quantization, 10 bytes per 32 elements.
    Q2_0,
    /// 3-bit symmetric quantization, 14 bytes per 32 elements.
    Q3_0,
    /// R16: Raw F16 with reserved Q-capture space, 128 bytes per 32 elements.
    R16,
    /// Q0: Zero block (1 byte per 32 elements).
    Q0,
    /// Q1_S: 1-bit sign + INT8 scale (5 bytes per 32 elements).
    Q1_S,
    /// Q2_S: 2-bit symmetric + INT8 scale (9 bytes per 32 elements).
    Q2_S,
    /// Q2_A: 2-bit asymmetric + INT8 scale + INT8 bias (10 bytes per 32 elements).
    Q2_A,
    /// Q2_1: 2-bit asymmetric + F16 scale + F16 min (12 bytes per 32 elements).
    Q2_1,
    /// Q3_1: 3-bit asymmetric + F16 scale + F16 min (16 bytes per 32 elements).
    Q3_1,
    /// Q0_V: Per-Block Parametric-Curve Quantization (2 bytes per 32 elements,
    Q0_V,
    /// Q1_A: 1-bit asymmetric, separate amplitude per sign (6 bytes per 32 elements).
    Q1_A,
    /// Q0_X: INT8 bulk anchor + one outlier escape (2 bytes per 32 elements).
    Q0_X,
    /// Q0_M2: 2-centroid palette + 8-bit quartet mask (3 bytes per 32 elements).
    Q0_M2,
    /// Q0_M4: 4-centroid palette + 32-bit pair mask (8 bytes per 32 elements).
    Q0_M4,
}

impl QuantFormat {
    /// Block size for this format.
    /// All quants use 32-element blocks.
    pub const fn block_size(&self) -> usize {
        32
    }

    /// Bytes per block of 32 elements. Sourced via `size_of::<BlockX>()` on
    /// the canonical Rust block struct so this stays in sync if a struct
    /// gains/loses a field — the Rust struct, in turn, is locked to its CUDA
    /// counterpart by `static_assert(sizeof(block_x) == N)` in
    /// `candle-kernels/src/blocks.cuh`.
    pub const fn bytes_per_block(&self) -> usize {
        use candle::quantized::k_quants::{
            BlockQ0, BlockQ0M2, BlockQ0M4, BlockQ0V, BlockQ0X, BlockQ1A, BlockQ1S, BlockQ2A,
            BlockQ2S, BlockQ2_0, BlockQ2_1, BlockQ3_0, BlockQ3_1, BlockQ4_0, BlockQ4_1, BlockQ4_KS,
            BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, BlockQ8_KS, BlockR16,
        };
        use std::mem::size_of;
        match self {
            // d:f16 (2) + qs[16]: 4-bit nibbles for 32 elems.
            Self::Q4_0 => size_of::<BlockQ4_0>(),
            // d:f16 (2) + m:f16 (2) + qs[16]: 4-bit nibbles + min offset.
            Self::Q4_1 => size_of::<BlockQ4_1>(),
            // d:f16 (2) + qh[4] high-bit + qs[16] low-4-bit: 5-bit per elem.
            Self::Q5_0 => size_of::<BlockQ5_0>(),
            // d:f16 (2) + m:f16 (2) + qh[4] + qs[16]: asymmetric 5-bit.
            Self::Q5_1 => size_of::<BlockQ5_1>(),
            // d:f16 (2) + qs[32] i8: full 8-bit per elem.
            Self::Q8_0 => size_of::<BlockQ8_0>(),
            // d:f16 (2) + s:f16 sum (2) + qs[32] i8: 8-bit + cached row sum.
            Self::Q8_1 => size_of::<BlockQ8_1>(),
            // d:f16 (2) + sa:u8 + sb:u8 sub-block scales + qs[16] 4-bit:
            // attention-sink-aware K-side fine scaling on first 4 elems.
            Self::Q4_KS => size_of::<BlockQ4_KS>(),
            // d:f16 (2) + sa:u8 + sb:u8 sub-block scales + qs[32] i8:
            // attention-sink-aware K-side fine scaling, 8-bit elems.
            Self::Q8_KS => size_of::<BlockQ8_KS>(),
            // d:f16 (2) + qs[8]: 2-bit symmetric, decode d * (q − 1.5).
            Self::Q2_0 => size_of::<BlockQ2_0>(),
            // d:f16 (2) + qh[4] high-bit + qs[8] low-2-bit: 3-bit symmetric.
            Self::Q3_0 => size_of::<BlockQ3_0>(),
            // d[32] f16 (64) + q[32] u16 reserved Q-capture space (64).
            Self::R16 => size_of::<BlockR16>(),
            // Single INT8 centroid byte — every lane in the block decodes to
            // the same value: `centroid / 127 / outer_scale`. Per-block
            // constant; the cheapest legitimate quant at 1 byte / 32 lanes
            // (0.25 BPE), useful for near-flat blocks.
            Self::Q0 => size_of::<BlockQ0>(),
            // scale: 1 byte FP8(E4M3) + qs[4] sign bits (1 bit × 32 elems).
            Self::Q1_S => size_of::<BlockQ1S>(),
            // scale: 1 byte FP8(E4M3) + qs[8] 2-bit symmetric quants.
            Self::Q2_S => size_of::<BlockQ2S>(),
            // scale + bias: 2 bytes FP8(E4M3) + qs[8] 2-bit asymmetric.
            Self::Q2_A => size_of::<BlockQ2A>(),
            // dm: u32 packed (f16 scale | f16 min) + qs[8] 2-bit asymmetric.
            Self::Q2_1 => size_of::<BlockQ2_1>(),
            // dm: u32 packed (f16 scale | f16 min) + qh[4] + qs[8] 3-bit asym.
            Self::Q3_1 => size_of::<BlockQ3_1>(),
            // lo: u8 curve_idx (0..255) | hi: 5-bit scale_idx + 3-bit
            // centroid_idx — parametric-curve quantization, 0.5 BPE.
            Self::Q0_V => size_of::<BlockQ0V>(),
            // scale_pos:i8 + scale_neg:i8 amplitudes + qs[4] sign bits.
            Self::Q1_A => size_of::<BlockQ1A>(),
            // bulk_anchor:i8 + outlier_packed:u8 (5-bit lane | 3-bit signed
            // delta) — flat block + one outlier escape, 0.5 BPE.
            Self::Q0_X => size_of::<BlockQ0X>(),
            // val_fp8[2] + qmask:u8 — 2 FP8(E4M3) centroids + per-quartet
            // mask choosing which centroid each lane uses.
            Self::Q0_M2 => size_of::<BlockQ0M2>(),
            // val_fp8[4] + qmask:u32 — 4 FP8(E4M3) centroids + 2-bit-per-pair
            // selector mask choosing one centroid per pair of lanes.
            Self::Q0_M4 => size_of::<BlockQ0M4>(),
        }
    }

    /// Approximate bytes per element.
    pub fn bytes_per_elem(&self) -> f32 {
        self.bytes_per_block() as f32 / self.block_size() as f32
    }

    /// Approximate bits per element.
    pub fn bits_per_elem(&self) -> f32 {
        self.bytes_per_elem() * 8.0
    }

    /// Convert to internal GGML quantization type.
    pub fn to_ggml_dtype(&self) -> GgmlDType {
        match self {
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q4_1 => GgmlDType::Q4_1,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q5_1 => GgmlDType::Q5_1,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q8_1 => GgmlDType::Q8_1,
            Self::Q4_KS => GgmlDType::Q4_KS,
            Self::Q8_KS => GgmlDType::Q8_KS,
            Self::Q2_0 => GgmlDType::Q2_0,
            Self::Q3_0 => GgmlDType::Q3_0,
            Self::R16 => GgmlDType::R16,
            Self::Q0 => GgmlDType::Q0,
            Self::Q1_S => GgmlDType::Q1_S,
            Self::Q2_S => GgmlDType::Q2_S,
            Self::Q2_A => GgmlDType::Q2_A,
            Self::Q2_1 => GgmlDType::Q2_1,
            Self::Q3_1 => GgmlDType::Q3_1,
            Self::Q0_V => GgmlDType::Q0_V,
            Self::Q1_A => GgmlDType::Q1_A,
            Self::Q0_X => GgmlDType::Q0_X,
            Self::Q0_M2 => GgmlDType::Q0_M2,
            Self::Q0_M4 => GgmlDType::Q0_M4,
        }
    }

    /// Check if this quantization format is supported by the paged attention kernel.
    ///
    /// The kernel supports on-the-fly dequantization for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1.
    /// K-quant formats (Q2K-Q6K) are NOT supported due to their complex block
    /// structure.
    pub const fn is_kernel_supported(&self) -> bool {
        match self {
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Q4_KS
            | Self::Q8_KS
            | Self::Q2_0
            | Self::Q3_0
            | Self::R16
            | Self::Q0
            | Self::Q1_S
            | Self::Q2_S
            | Self::Q2_A
            | Self::Q2_1
            | Self::Q3_1
            | Self::Q0_V
            | Self::Q1_A
            | Self::Q0_X
            | Self::Q0_M2
            | Self::Q0_M4 => true,
        }
    }
}

/// Storage format for KV cache - either standard float or block-quantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]

pub enum KvFormat {
    /// Standard floating-point storage (F32, F16, BF16, F8E4M3).
    Float(DType),
    /// Block-quantized storage (Q4_0, Q8_0).
    Quantized(QuantFormat),
}

impl KvFormat {
    /// Returns true if this format is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// Returns the DType if this is a float format, None for quantized.
    pub fn dtype(&self) -> Option<DType> {
        match self {
            Self::Float(dt) => Some(*dt),
            Self::Quantized(_) => None,
        }
    }

    /// Approximate bytes per element.
    pub fn bytes_per_elem(&self) -> f32 {
        match self {
            Self::Float(dt) => dt.size_in_bytes() as f32,
            Self::Quantized(qf) => qf.bytes_per_elem(),
        }
    }

    /// Exact bytes one [`CHUNK_SIZE`]-element block occupies in this format.
    ///
    /// The integer counterpart of [`Self::bytes_per_elem`], for callers that
    /// must account storage in whole blocks — a quantized block's size is not
    /// generally divisible by its element count (`Q4_0` is 18 bytes for 32
    /// elements), so per-element arithmetic cannot round-trip it.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::Float(dt) => dt.size_in_bytes() * CHUNK_SIZE,
            Self::Quantized(qf) => qf.bytes_per_block(),
        }
    }

    /// Returns the QuantFormat if this is quantized, None for float.
    pub fn as_quant(&self) -> Option<QuantFormat> {
        match self {
            Self::Quantized(qf) => Some(*qf),
            Self::Float(_) => None,
        }
    }

    /// Stable `u8` tag for on-disk persistence — the [`ArenaFormatTag`]
    /// discriminant. Round-trips through [`Self::from_tag`].
    pub fn to_tag(self) -> u8 {
        use crate::kv_cache::arena_table::ArenaFormatTag;
        ArenaFormatTag::from_kv_format(self).as_u8()
    }

    /// Decode a persisted [`Self::to_tag`] byte. `None` for an unrecognised
    /// or unsupported tag.
    ///
    /// Implemented as the search-inverse of [`Self::to_tag`]: it scans the
    /// formats `ArenaFormatTag::from_kv_format` accepts and returns the one
    /// whose tag matches. There is **no** second hand-written byte→format
    /// table to drift from the forward mapping — the round trip stays exact
    /// even if `from_kv_format` / the tag discriminants change.
    pub fn from_tag(tag: u8) -> Option<KvFormat> {
        use strum::IntoEnumIterator;
        const FLOAT_DTYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F8E4M3];
        FLOAT_DTYPES
            .into_iter()
            .map(KvFormat::Float)
            .chain(QuantFormat::iter().map(KvFormat::Quantized))
            .find(|fmt| fmt.to_tag() == tag)
    }
}

impl Default for KvFormat {
    fn default() -> Self {
        Self::Float(DType::BF16)
    }
}

impl From<DType> for KvFormat {
    fn from(dt: DType) -> Self {
        Self::Float(dt)
    }
}

impl From<QuantFormat> for KvFormat {
    fn from(qf: QuantFormat) -> Self {
        Self::Quantized(qf)
    }
}

#[cfg(test)]
mod tests;
