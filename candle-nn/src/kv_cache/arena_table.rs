//! Arena table types for kernel-side KV cache access.
//!
//! This module defines the GPU-indexable structures that kernels use to access
//! KV cache arenas. These types form a stable interface between Rust and CUDA:
//!
//! - [`ArenaLocation`] - Where arena data lives (GPU or CPU)
//! - [`ArenaFormatTag`] - Storage format identifier for kernel dispatch
//! - [`ArenaEntry`] - Per-arena metadata (pointers, format, location)
//! - [`ArenaTable`] - Persistent GPU tensor of arena metadata (1 row per arena)
//! - [`PaletteSubEntry`] - Per-palette sub-entry metadata (9 fields)
//! - [`PerHeadEntry`] - Per-head metadata covering all N_PALETTE palette sub-bands (36 fields)
//! - [`PerHeadTable`] - Persistent GPU tensor of per-head metadata
//!
//! # Kernel Interface
//!
//! The [`ArenaTable`] maintains a GPU tensor of shape `(num_arenas, 3)` with dtype i64.
//! Each row contains `[k_ptr, v_ptr, metadata]` where:
//! - `k_ptr`: Device pointer to K cache data (0 if CPU-only)
//! - `v_ptr`: Device pointer to V cache data (0 if CPU-only)
//! - `metadata`: `(k_format_tag << 16) | (v_format_tag << 8) | location`
//!
//! The [`PerHeadTable`] maintains a GPU tensor of shape `(num_arenas * n_kv_head, 36)`.
//! Indexed by `pal0_arena_idx * n_kv_head + head_idx`. Each row contains N_PALETTE=4
//! sub-entries of 9 fields each (36 total), one per palette sub-band.
//! Sub-entry p at cols [p*9 .. p*9+9]:
//! `[k_ptr, v_ptr, k_byte_offset, v_byte_offset, k_chunk_byte_stride, v_chunk_byte_stride, metadata, k_outer_scale_bits, v_outer_scale_bits]`
//!
//! The kernel addresses head data as:
//! `data = (char*)ptr + byte_offset + chunk_idx * chunk_byte_stride`
//! — no derived stride arithmetic needed.

use crate::kv_cache::{KvFormat, QuantFormat};
use candle::{DType, Device, Result, Tensor};

// ==================== Arena Location ====================

/// Where an arena's data is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum_macros::EnumIter)]
#[repr(u8)]
pub enum ArenaLocation {
    /// GPU memory (CUDA, Metal, etc.)
    Gpu = 0,
    /// System RAM (CPU)
    Cpu = 1,
}

// ==================== Arena Format Tag ====================

/// Format tag for arena entries in the GPU table.
///
/// This is a u8 that encodes the storage format in a kernel-friendly way.
/// Float formats use indices 0-15.
/// Quantized formats use `QUANT_BASE` (16) + GGML type index for easy conversion.
///
/// # Examples
///
/// ```ignore
/// use candle_nn::kv_cache::ArenaFormatTag;
///
/// // From KvFormat
/// let tag = ArenaFormatTag::from_kv_format(KvFormat::Float(DType::BF16));
/// assert_eq!(tag, ArenaFormatTag::BF16);
///
/// // Check if quantized
/// assert!(!tag.is_quantized());
/// assert!(ArenaFormatTag::Q8_0.is_quantized());
/// ```
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArenaFormatTag {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
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

    /// Invalid/uninitialized
    Invalid = 255,
}

impl ArenaFormatTag {
    /// Convert from KvFormat
    pub fn from_kv_format(format: KvFormat) -> Self {
        match format {
            KvFormat::Float(DType::F32) => Self::F32,
            KvFormat::Float(DType::F16) => Self::F16,
            KvFormat::Float(DType::BF16) => Self::BF16,
            KvFormat::Float(DType::F8E4M3) => Self::F8E4M3,
            KvFormat::Float(_) => Self::Invalid, // Other dtypes not supported
            KvFormat::Quantized(QuantFormat::Q4_0) => Self::Q4_0,
            KvFormat::Quantized(QuantFormat::Q4_1) => Self::Q4_1,
            KvFormat::Quantized(QuantFormat::Q5_0) => Self::Q5_0,
            KvFormat::Quantized(QuantFormat::Q5_1) => Self::Q5_1,
            KvFormat::Quantized(QuantFormat::Q8_0) => Self::Q8_0,
            KvFormat::Quantized(QuantFormat::Q8_1) => Self::Q8_1,
            KvFormat::Quantized(QuantFormat::Q4_KS) => Self::Q4_KS,
            KvFormat::Quantized(QuantFormat::Q8_KS) => Self::Q8_KS,
            KvFormat::Quantized(QuantFormat::Q2_0) => Self::Q2_0,
            KvFormat::Quantized(QuantFormat::Q3_0) => Self::Q3_0,
            KvFormat::Quantized(QuantFormat::R16) => Self::R16,
            KvFormat::Quantized(QuantFormat::Q0) => Self::Q0,
            KvFormat::Quantized(QuantFormat::Q1_S) => Self::Q1_S,
            KvFormat::Quantized(QuantFormat::Q2_S) => Self::Q2_S,
            KvFormat::Quantized(QuantFormat::Q2_A) => Self::Q2_A,
            KvFormat::Quantized(QuantFormat::Q2_1) => Self::Q2_1,
            KvFormat::Quantized(QuantFormat::Q3_1) => Self::Q3_1,
            KvFormat::Quantized(QuantFormat::Q0_V) => Self::Q0_V,
            KvFormat::Quantized(QuantFormat::Q1_A) => Self::Q1_A,
            KvFormat::Quantized(QuantFormat::Q0_X) => Self::Q0_X,
            KvFormat::Quantized(QuantFormat::Q0_M2) => Self::Q0_M2,
            KvFormat::Quantized(QuantFormat::Q0_M4) => Self::Q0_M4,
        }
    }

    /// Convert from GGML type index (as used in GgmlDType::from_gguf_file_code)
    ///
    /// GGML indices: Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9
    ///               Q2K=10, Q3K=11, Q4K=12, Q5K=13, Q6K=14, Q8K=15
    ///               AWQ=100, AWQG64=101
    pub fn from_ggml_index(idx: u32) -> Self {
        match idx {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::R16,
            4 => Self::P2,
            5 => Self::QAWQ,
            6 => Self::QAWQ_G64,
            7 => Self::Q8_0,
            8 => Self::Q8_1,
            9 => Self::Q8_K,
            10 => Self::Q8_KS,
            11 => Self::Q6_K,
            12 => Self::Q5_0,
            13 => Self::Q5_1,
            14 => Self::Q5_K,
            15 => Self::Q4_0,
            16 => Self::Q4_1,
            17 => Self::Q4_K,
            18 => Self::Q4_KS,
            19 => Self::Q3_0,
            20 => Self::Q3_1,
            21 => Self::Q3_K,
            22 => Self::Q2_0,
            23 => Self::Q2_1,
            24 => Self::Q2_K,
            25 => Self::Q2_S,
            26 => Self::Q2_A,
            27 => Self::Q1_S,
            28 => Self::Q0_V,
            29 => Self::Q1_A,
            30 => Self::Q0_X,
            31 => Self::Q0_M2,
            32 => Self::Q0_M4,
            33 => Self::Q0,
            34 => Self::F8E4M3,
            35 => Self::F8E5M2,
            _ => Self::Invalid,
        }
    }

    /// Convert to DType (for float formats only).
    /// Returns None for quantized or invalid formats.
    pub fn to_dtype(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::BF16 => Some(DType::BF16),
            Self::F8E4M3 => Some(DType::F8E4M3),
            _ => None, // All quantized formats and Invalid return None
        }
    }

    /// Convert to u8 for storage
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Is this a quantized format?
    pub fn is_quantized(self) -> bool {
        match self {
            Self::F32 => false,
            Self::F16 => false,
            Self::BF16 => false,
            Self::F8E4M3 => false,
            Self::F8E5M2 => false,
            _ => true,
        }
    }
}

// ==================== Arena Entry ====================

/// A single entry in the arena table, representing one arena's metadata.
///
/// This is laid out for efficient GPU access:
/// - `k_ptr`: device pointer to K data (0 if CPU-only)
/// - `v_ptr`: device pointer to V data (0 if CPU-only)
/// - `k_format_tag`: [`ArenaFormatTag`] for K cache
/// - `v_format_tag`: [`ArenaFormatTag`] for V cache
/// - `location`: [`ArenaLocation`] as u8
///
/// The struct is stored as a row of i64 values in the GPU tensor for alignment:
/// `[k_ptr, v_ptr, (k_format_tag << 16) | (v_format_tag << 8) | location]`
#[derive(Debug, Clone, Copy)]
pub struct ArenaEntry {
    /// Device pointer to K cache data (0 if arena is on CPU)
    pub k_ptr: u64,
    /// Device pointer to V cache data (0 if arena is on CPU)
    pub v_ptr: u64,
    /// Storage format tag for K cache
    pub k_format_tag: ArenaFormatTag,
    /// Storage format tag for V cache
    pub v_format_tag: ArenaFormatTag,
    /// Where this arena's data lives
    pub location: ArenaLocation,
}

impl ArenaEntry {
    /// Create a new entry with null pointers (for CPU arenas or placeholder)
    pub fn new_cpu(k_format_tag: ArenaFormatTag, v_format_tag: ArenaFormatTag) -> Self {
        Self {
            k_ptr: 0,
            v_ptr: 0,
            k_format_tag,
            v_format_tag,
            location: ArenaLocation::Cpu,
        }
    }

    /// Create a new entry with GPU pointers
    pub fn new_gpu(
        k_ptr: u64,
        v_ptr: u64,
        k_format_tag: ArenaFormatTag,
        v_format_tag: ArenaFormatTag,
    ) -> Self {
        Self {
            k_ptr,
            v_ptr,
            k_format_tag,
            v_format_tag,
            location: ArenaLocation::Gpu,
        }
    }

    /// Encode K/V format tags and location into a single i64 for storage
    /// Format: [unused:40][k_format_tag:8][v_format_tag:8][location:8]
    fn encode_metadata(&self) -> i64 {
        ((self.k_format_tag.as_u8() as i64) << 16)
            | ((self.v_format_tag.as_u8() as i64) << 8)
            | (self.location as i64)
    }

    /// Convert to GPU tensor row format: [k_ptr, v_ptr, metadata]
    pub fn to_tensor_row(&self) -> [i64; 3] {
        [self.k_ptr as i64, self.v_ptr as i64, self.encode_metadata()]
    }

    /// Decode from GPU tensor row format
    pub fn from_tensor_row(row: [i64; 3]) -> Self {
        let k_ptr = row[0] as u64;
        let v_ptr = row[1] as u64;
        let metadata = row[2];
        let k_format_tag = Self::decode_format_byte(((metadata >> 16) & 0xFF) as u8);
        let v_format_tag = Self::decode_format_byte(((metadata >> 8) & 0xFF) as u8);
        let location = match (metadata & 0xFF) as u8 {
            0 => ArenaLocation::Gpu,
            _ => ArenaLocation::Cpu,
        };
        Self {
            k_ptr,
            v_ptr,
            k_format_tag,
            v_format_tag,
            location,
        }
    }

    /// Decode a format byte into an ArenaFormatTag.
    fn decode_format_byte(byte: u8) -> ArenaFormatTag {
        match byte {
            0 => ArenaFormatTag::F32,
            1 => ArenaFormatTag::F16,
            2 => ArenaFormatTag::BF16,
            3 => ArenaFormatTag::R16,
            4 => ArenaFormatTag::P2,
            5 => ArenaFormatTag::QAWQ,
            6 => ArenaFormatTag::QAWQ_G64,
            7 => ArenaFormatTag::Q8_0,
            8 => ArenaFormatTag::Q8_1,
            9 => ArenaFormatTag::Q8_K,
            10 => ArenaFormatTag::Q8_KS,
            11 => ArenaFormatTag::Q6_K,
            12 => ArenaFormatTag::Q5_0,
            13 => ArenaFormatTag::Q5_1,
            14 => ArenaFormatTag::Q5_K,
            15 => ArenaFormatTag::Q4_0,
            16 => ArenaFormatTag::Q4_1,
            17 => ArenaFormatTag::Q4_K,
            18 => ArenaFormatTag::Q4_KS,
            19 => ArenaFormatTag::Q3_0,
            20 => ArenaFormatTag::Q3_1,
            21 => ArenaFormatTag::Q3_K,
            22 => ArenaFormatTag::Q2_0,
            23 => ArenaFormatTag::Q2_1,
            24 => ArenaFormatTag::Q2_K,
            25 => ArenaFormatTag::Q2_S,
            26 => ArenaFormatTag::Q2_A,
            27 => ArenaFormatTag::Q1_S,
            28 => ArenaFormatTag::Q0_V,
            29 => ArenaFormatTag::Q1_A,
            30 => ArenaFormatTag::Q0_X,
            31 => ArenaFormatTag::Q0_M2,
            32 => ArenaFormatTag::Q0_M4,
            33 => ArenaFormatTag::Q0,
            34 => ArenaFormatTag::F8E4M3,
            35 => ArenaFormatTag::F8E5M2,
            _ => ArenaFormatTag::Invalid,
        }
    }
}

// ==================== Palette Sub-Entry ====================

/// Number of palette sub-bands per head (splits head_dim into N_PALETTE equal parts).
pub const N_PALETTE: usize = 4;

/// Single per-palette metadata sub-entry.
///
/// Describes how to access one palette sub-band (HEAD_DIM/N_PALETTE elements) of
/// one KV head within a palette sub-arena.
/// The kernel uses: `data = (char*)ptr + byte_offset + chunk_idx * chunk_byte_stride`.
#[derive(Debug, Clone, Copy)]
pub struct PaletteSubEntry {
    /// Base pointer to K data allocation for this palette sub-arena
    pub k_ptr: u64,
    /// Base pointer to V data allocation for this palette sub-arena
    pub v_ptr: u64,
    /// Byte offset from k_ptr to the start of this head's K data
    pub k_byte_offset: i64,
    /// Byte offset from v_ptr to the start of this head's V data
    pub v_byte_offset: i64,
    /// Byte stride between consecutive chunks for this head in K
    pub k_chunk_byte_stride: i64,
    /// Byte stride between consecutive chunks for this head in V
    pub v_chunk_byte_stride: i64,
    /// Format tag for this head's K data
    pub k_format_tag: ArenaFormatTag,
    /// Format tag for this head's V data
    pub v_format_tag: ArenaFormatTag,
    /// Outer scale applied to K values at encode time (decode divides by this)
    pub k_outer_scale: f32,
    /// Outer scale applied to V values at encode time (decode divides by this)
    pub v_outer_scale: f32,
}

impl PaletteSubEntry {
    /// Number of i64 columns for one palette sub-entry.
    pub const COLS: usize = 9;

    /// Encode to 9 i64 values:
    /// `[k_ptr, v_ptr, k_byte_offset, v_byte_offset, k_chunk_byte_stride, v_chunk_byte_stride, metadata, k_outer_scale_bits, v_outer_scale_bits]`
    pub fn to_cols(&self) -> [i64; Self::COLS] {
        let metadata =
            ((self.k_format_tag.as_u8() as i64) << 16) | ((self.v_format_tag.as_u8() as i64) << 8);
        [
            self.k_ptr as i64,
            self.v_ptr as i64,
            self.k_byte_offset,
            self.v_byte_offset,
            self.k_chunk_byte_stride,
            self.v_chunk_byte_stride,
            metadata,
            self.k_outer_scale.to_bits() as i64,
            self.v_outer_scale.to_bits() as i64,
        ]
    }
}

// ==================== Per-Head Entry (palette4) ====================

/// Per-head metadata for kernel consumption.
///
/// Contains N_PALETTE=4 sub-entries, one per palette sub-band.
/// Each palette sub-band covers HEAD_DIM/N_PALETTE consecutive dimensions.
///
/// GPU tensor row: 36 i64 values = 4 × 9 sub-entry fields.
/// Row index: `pal0_arena_idx * n_kv_head + head_idx`
/// Sub-entry p is at cols [p*9 .. p*9+9].
///
/// Chunk index for palette p comes from head_gids, not this table:
///   `chunk_idx = gid_pal_p % ARENA_CHUNKS`
#[derive(Debug, Clone, Copy)]
pub struct PerHeadEntry {
    /// Per-palette sub-entries, indexed by palette slot 0..N_PALETTE.
    pub palette: [PaletteSubEntry; N_PALETTE],
}

impl PerHeadEntry {
    /// Number of i64 columns per row in the GPU tensor (N_PALETTE × 9 = 36).
    pub const COLS: usize = N_PALETTE * PaletteSubEntry::COLS;

    /// Encode to GPU tensor row format (36 i64 values, 4 sub-entries of 9 each).
    pub fn to_tensor_row(&self) -> [i64; Self::COLS] {
        let mut row = [0i64; Self::COLS];
        for (p, sub) in self.palette.iter().enumerate() {
            let cols = sub.to_cols();
            let off = p * PaletteSubEntry::COLS;
            row[off..off + PaletteSubEntry::COLS].copy_from_slice(&cols);
        }
        row
    }

    /// Construct a uniform entry where all N_PALETTE slots share the same sub-arena.
    /// Used when all palette sub-bands live in the same physical arena.
    pub fn uniform(sub: PaletteSubEntry) -> Self {
        Self {
            palette: [sub; N_PALETTE],
        }
    }
}

// ==================== Per-Head Table ====================

/// Per-head lookup table for kernel consumption.
///
/// Separate from [`ArenaTable`] — this provides the kernel with per-head
/// pointers, byte offsets, byte strides, and format tags so there is no
/// derived stride arithmetic in the kernel.
///
/// GPU tensor shape: `(num_arenas * n_kv_head, 36)` i64.
/// Indexed by `pal0_arena_idx * n_kv_head + head_idx`.
/// Each row contains N_PALETTE=4 sub-entries of 9 fields each:
/// sub-entry p at cols [p*9 .. p*9+9]:
///
/// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | ... | 27 | 28 | ... | 35 |
/// |---|---|---|---|---|---|---|---|---|-----|----|----|-----|-----|
/// | pal0_k_ptr | pal0_v_ptr | ... | pal0_meta | pal0_k_scale | pal0_v_scale | ... | pal3_v_scale |
///
/// metadata: `(k_format_tag << 16) | (v_format_tag << 8) | location`
#[derive(Debug)]
pub struct PerHeadTable {
    /// Host-side entries. Flat layout: `entries[arena_idx * n_kv_head + head_idx]`.
    entries: Vec<PerHeadEntry>,
    /// GPU tensor of shape (num_arenas * n_kv_head, 36), dtype i64
    gpu_tensor: Option<Tensor>,
    /// Device for GPU tensor
    device: Device,
    /// Whether GPU tensor needs sync
    dirty: bool,
    /// Number of KV heads (fixed at construction, same for all arenas)
    n_kv_head: usize,
}

impl PerHeadTable {
    /// Create a new empty per-head table.
    pub fn new(device: &Device, n_kv_head: usize) -> Self {
        Self {
            entries: Vec::new(),
            gpu_tensor: None,
            device: device.clone(),
            dirty: false,
            n_kv_head: n_kv_head.max(1),
        }
    }

    /// Number of KV heads per arena.
    pub fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    /// Number of arenas in the table.
    pub fn len(&self) -> usize {
        self.entries.len() / self.n_kv_head
    }

    /// Is the table empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get reference to all entries.
    pub fn entries(&self) -> &[PerHeadEntry] {
        &self.entries
    }

    /// Is the dirty flag set?
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Add per-head entries for one arena. `head_entries.len()` must equal `n_kv_head`.
    pub fn push(&mut self, head_entries: &[PerHeadEntry]) {
        assert_eq!(
            head_entries.len(),
            self.n_kv_head,
            "PerHeadTable::push: expected {} entries, got {}",
            self.n_kv_head,
            head_entries.len()
        );
        self.entries.extend_from_slice(head_entries);
        self.dirty = true;
    }

    /// Truncate to a given number of arenas.
    pub fn truncate(&mut self, len: usize) {
        let flat_len = len * self.n_kv_head;
        if flat_len < self.entries.len() {
            self.entries.truncate(flat_len);
            self.dirty = true;
        }
    }

    /// Get a specific head entry for an arena.
    pub fn get(&self, arena_idx: usize, head_idx: usize) -> Option<&PerHeadEntry> {
        if head_idx >= self.n_kv_head {
            return None;
        }
        self.entries.get(arena_idx * self.n_kv_head + head_idx)
    }

    /// Update all head entries for an arena. `head_entries.len()` must equal `n_kv_head`.
    pub fn set(&mut self, arena_idx: usize, head_entries: &[PerHeadEntry]) {
        assert_eq!(head_entries.len(), self.n_kv_head);
        let base = arena_idx * self.n_kv_head;
        for (h, entry) in head_entries.iter().enumerate() {
            if let Some(e) = self.entries.get_mut(base + h) {
                *e = *entry;
            }
        }
        self.dirty = true;
    }

    /// Update a single head entry for an arena.
    ///
    /// Used after per-head quantization to patch individual heads with
    /// correct format tags and cross-arena pointers when K and V live
    /// in different arenas.
    pub fn set_head(&mut self, arena_idx: usize, head_idx: usize, entry: PerHeadEntry) {
        if head_idx >= self.n_kv_head {
            return;
        }
        let idx = arena_idx * self.n_kv_head + head_idx;
        if let Some(e) = self.entries.get_mut(idx) {
            *e = entry;
            self.dirty = true;
        }
    }

    /// Sync host entries to GPU tensor if dirty.
    pub fn sync_to_gpu(&mut self) -> Result<&Tensor> {
        if self.dirty || self.gpu_tensor.is_none() {
            let num_entries = self.entries.len();
            if num_entries == 0 {
                let tensor = Tensor::zeros((1, PerHeadEntry::COLS), DType::I64, &self.device)?;
                self.gpu_tensor = Some(tensor);
            } else {
                let mut data: Vec<i64> = Vec::with_capacity(num_entries * PerHeadEntry::COLS);
                for entry in &self.entries {
                    let row = entry.to_tensor_row();
                    data.extend_from_slice(&row);
                }
                let tensor =
                    Tensor::from_vec(data, (num_entries, PerHeadEntry::COLS), &self.device)?;
                self.gpu_tensor = Some(tensor);
            }
            self.dirty = false;
        }
        self.gpu_tensor
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("per-head table GPU tensor not initialized".into()))
    }

    /// Get the GPU tensor without syncing (may be stale or None).
    pub fn gpu_tensor(&self) -> Option<&Tensor> {
        self.gpu_tensor.as_ref()
    }

    /// Force rebuild of GPU tensor on next sync.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ResolvedArenaInfo — lightweight per-arena pointer/stride/format snapshot
// ═══════════════════════════════════════════════════════════════════════════════

/// Pre-resolved arena metadata for constructing persistent GPU slot buffers.
///
/// One entry per arena index.  Use [`ChunkedKvBacking::resolve_arena_info`]
/// to build a `Vec<ResolvedArenaInfo>` from the current arena state.
///
/// The GPU address for a specific chunk is:
///   `base_ptr + chunk_idx * chunk_byte_stride`
/// where `chunk_idx = gid.chunk_idx()`.
#[derive(Clone, Debug)]
pub struct ResolvedArenaInfo {
    /// Base device pointer for this arena's combined K+V buffer (0 for CPU arenas).
    pub base_ptr: u64,
    /// Byte stride between consecutive chunks in this arena.
    pub chunk_byte_stride: i64,
    /// Format tag for K data in this arena.
    pub k_format_tag: ArenaFormatTag,
    /// Format tag for V data in this arena.
    pub v_format_tag: ArenaFormatTag,
    /// Number of chunk slots this arena actually holds. Format-specific and
    /// generally SMALLER than `arena_gid_stride()` (the raw-GID namespace is
    /// sized for the densest format), so a gid's `chunk_idx` can be in-range
    /// for the namespace yet out of range for its arena — addressing past the
    /// arena's end. Validators compare `chunk_idx` against this, not the stride.
    pub chunk_capacity: u32,
}
