//! Arena table types for kernel-side KV cache access.
//!
//! This module defines the GPU-indexable structures that kernels use to access
//! KV cache arenas. These types form a stable interface between Rust and CUDA:
//!
//! - [`ArenaLocation`] - Where arena data lives (GPU or CPU)
//! - [`ArenaFormatTag`] - Storage format identifier for kernel dispatch
//! - [`PaletteSubEntry`] - Per-palette sub-entry metadata (9 fields)
//! - [`PerHeadEntry`] - Per-head metadata covering all N_PALETTE palette sub-bands (36 fields)
//! - [`PerHeadTable`] - Persistent GPU tensor of per-head metadata
//!
//! # Kernel Interface
//!
//! There is no per-arena table. An arena is a run of fixed-stride byte slots and
//! carries no format of its own, so the only per-arena quantity a kernel could
//! want — the base pointer — reaches it inside the per-head row that names the
//! band. See `docs/archived/arena_unification.md` principle 8.
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

    /// The inverse of [`Self::from_kv_format`].
    ///
    /// `None` for the tags that name a storage format the KV cache never
    /// allocates (`P2`, the QAWQ pair, the GGML K-quants, `F8E5M2`) and for
    /// `Invalid`. Those are legal *bytes* — a corrupt or future tag decodes to
    /// one — so a caller that needs a chunk's byte length must treat the
    /// absence as an error rather than assume a width.
    ///
    /// This is how a band's payload length is recovered now that arenas are
    /// size-class byte slabs and no longer carry a format: the chunk's tag is
    /// the only record of what its bytes are (`docs/archived/arena_unification.md`
    /// principle 8). Round-tripping every `KvFormat` through
    /// `from_kv_format` and back is pinned by
    /// [`every_kv_format_round_trips_through_its_tag`](tests).
    pub fn to_kv_format(self) -> Option<KvFormat> {
        let q = match self {
            Self::F32 => return Some(KvFormat::Float(DType::F32)),
            Self::F16 => return Some(KvFormat::Float(DType::F16)),
            Self::BF16 => return Some(KvFormat::Float(DType::BF16)),
            Self::F8E4M3 => return Some(KvFormat::Float(DType::F8E4M3)),
            Self::Q4_0 => QuantFormat::Q4_0,
            Self::Q4_1 => QuantFormat::Q4_1,
            Self::Q5_0 => QuantFormat::Q5_0,
            Self::Q5_1 => QuantFormat::Q5_1,
            Self::Q8_0 => QuantFormat::Q8_0,
            Self::Q8_1 => QuantFormat::Q8_1,
            Self::Q4_KS => QuantFormat::Q4_KS,
            Self::Q8_KS => QuantFormat::Q8_KS,
            Self::Q2_0 => QuantFormat::Q2_0,
            Self::Q3_0 => QuantFormat::Q3_0,
            Self::R16 => QuantFormat::R16,
            Self::Q0 => QuantFormat::Q0,
            Self::Q1_S => QuantFormat::Q1_S,
            Self::Q2_S => QuantFormat::Q2_S,
            Self::Q2_A => QuantFormat::Q2_A,
            Self::Q2_1 => QuantFormat::Q2_1,
            Self::Q3_1 => QuantFormat::Q3_1,
            Self::Q0_V => QuantFormat::Q0_V,
            Self::Q1_A => QuantFormat::Q1_A,
            Self::Q0_X => QuantFormat::Q0_X,
            Self::Q0_M2 => QuantFormat::Q0_M2,
            Self::Q0_M4 => QuantFormat::Q0_M4,
            Self::P2
            | Self::QAWQ
            | Self::QAWQ_G64
            | Self::Q8_K
            | Self::Q6_K
            | Self::Q5_K
            | Self::Q4_K
            | Self::Q3_K
            | Self::Q2_K
            | Self::F8E5M2
            | Self::Invalid => return None,
        };
        Some(KvFormat::Quantized(q))
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

impl ArenaFormatTag {
    /// Decode a format byte (as produced by [`Self::as_u8`]) back into a tag.
    ///
    /// This is the inverse every per-chunk format tag goes through: chunks
    /// record their bands' formats as these bytes (`SealedChunk::k_fmt`), the
    /// substrate persists the same bytes, and both are decoded here.
    /// Unrecognised bytes become [`Self::Invalid`] rather than panicking — a
    /// corrupt or future tag must fail the format checks, not the process.
    pub fn from_u8(byte: u8) -> ArenaFormatTag {
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
///
/// # What is deliberately absent
///
/// A format. An arena is a run of fixed-stride byte slots whose tenants may be
/// any formats that fit, so "the arena's format" is not a question with an
/// answer. A band's format — and therefore its payload length, its persisted
/// image, and its checksum — comes from the owning chunk's tag
/// (`SealedChunk::k_fmt`), never from here. See `docs/archived/arena_unification.md`
/// principle 8 and invariant 8.
#[derive(Clone, Debug)]
pub struct ResolvedArenaInfo {
    /// Base device pointer for this arena's byte slab (0 for CPU arenas).
    pub base_ptr: u64,
    /// Byte stride between consecutive chunk **slots** — the address step,
    /// `base_ptr + chunk_idx * chunk_byte_stride`, and the extent that must be
    /// zeroed when a slot is recycled (the next tenant may be a different
    /// format, so stale bytes past its payload would be read as data by the
    /// persist quantize pass).
    ///
    /// This is the *class* stride and is generally **larger** than the payload
    /// of the format that happens to occupy a given slot. Never use it as a
    /// copy length.
    pub chunk_byte_stride: i64,
    /// Number of chunk slots this arena actually holds — `chunks_per_region`
    /// for its class. Generally much smaller than `GID_STRIDE` (the raw-GID
    /// namespace is one fixed power of two), so a gid's `chunk_idx` can be
    /// in-range for the namespace yet past this arena's end. Validators compare
    /// `chunk_idx` against this, not against the stride.
    pub chunk_capacity: u32,
}
