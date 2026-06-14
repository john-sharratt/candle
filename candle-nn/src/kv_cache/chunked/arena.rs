//! Arena storage types and management for ChunkedKvBacking.
//!
//! This module contains:
//! - `ArenaKey` - Key identifying an arena by format and location
//! - `StoragePolicy` - How sealed chunks should be stored
//! - `ChunkStatus` - Status of a chunk (Unallocated/Sealed/Active)
//! - `Arena` - A single arena holding K and V cache tensors
//! - `ArenaStorage` - Collection of arenas with heterogeneous formats

use crate::kv_cache::{
    arena_table::{ArenaEntry, ArenaFormatTag, ArenaLocation},
    KvFormat, QuantFormat,
};
use ahash::AHashMap;
use candle::quantized::QTensor;
use candle::{DType, Result, Tensor};
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

use crate::kv_cache::chunked::types::arena_chunks_for_format;

// ==================== Arena Key ====================

/// Key identifying an arena type by its format and location.
/// Exactly two fields. K and V always share the same format within an arena.
#[derive(Debug, Clone)]
pub struct ArenaKey {
    pub format: KvFormat,
    pub location: ArenaLocation,
}

impl PartialEq for ArenaKey {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format && self.location == other.location
    }
}

impl Eq for ArenaKey {}

impl Hash for ArenaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.format.hash(state);
        self.location.hash(state);
    }
}

impl ArenaKey {
    pub fn new(format: KvFormat, location: ArenaLocation) -> Self {
        Self { format, location }
    }

    /// Create a key where K and V share the same format.
    pub fn uniform(format: KvFormat, location: ArenaLocation) -> Self {
        Self { format, location }
    }

    pub fn gpu_float(dtype: DType) -> Self {
        Self {
            format: KvFormat::Float(dtype),
            location: ArenaLocation::Gpu,
        }
    }

    /// GPU quantized arena key (unified K+V, same format).
    pub fn gpu_quant(fmt: QuantFormat) -> Self {
        Self {
            format: KvFormat::Quantized(fmt),
            location: ArenaLocation::Gpu,
        }
    }

    /// GPU quantized arena key with separate K and V formats.
    /// ⚠️ VESTIGIAL: when k_fmt != v_fmt this SILENTLY DROPS v_fmt.
    /// This constructor exists as dead code from a prior incorrect session.
    /// DO NOT add logic here to encode both formats — delete this and use
    /// two separate arenas instead. See docs/kv_cache_unification.md §7.5, §11.3.
    pub fn gpu_quant_kv(k_fmt: QuantFormat, v_fmt: QuantFormat) -> Self {
        if k_fmt == v_fmt {
            Self::gpu_quant(k_fmt)
        } else {
            Self {
                format: KvFormat::Quantized(k_fmt),
                location: ArenaLocation::Gpu,
            }
        }
    }

    pub fn cpu_float(dtype: DType) -> Self {
        Self {
            format: KvFormat::Float(dtype),
            location: ArenaLocation::Cpu,
        }
    }

    pub fn is_gpu(&self) -> bool {
        self.location == ArenaLocation::Gpu
    }

    pub fn is_quantized(&self) -> bool {
        self.format.is_quantized()
    }

    /// Check if this format is readable by the paged attention kernel without conversion.
    ///
    /// The kernel supports:
    /// - GPU float formats (F32, F16, BF16, F8E4M3) - native reads
    /// - GPU quantized formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1) - dequant on read
    ///
    /// CPU formats always require migration to GPU first.
    /// K-quant formats (Q2K-Q6K) are not supported for paged attention.
    pub fn is_kernel_readable(&self) -> bool {
        // CPU formats always need migration
        if self.location != ArenaLocation::Gpu {
            return false;
        }

        let ok = match &self.format {
            KvFormat::Float(_) => true,
            KvFormat::Quantized(qf) => qf.is_kernel_supported(),
        };
        ok
    }
}

// ==================== Storage Policy ====================

/// How a session wants its sealed KV cache chunks stored after operations complete.
/// The active (last/partial) chunk is always GPU float for writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoragePolicy {
    /// Keep on GPU in native dtype (fastest decode, most GPU memory)
    GpuFloat(DType),
    /// Keep on GPU quantized (good decode speed, less GPU memory).
    GpuQuant(QuantFormat),
    /// Offload to CPU in dtype (saves GPU memory, requires prepare step)
    CpuFloat(DType),
    /// Offload to CPU quantized (minimum memory, requires prepare + dequant for active).
    CpuQuant(QuantFormat),
}

impl StoragePolicy {
    /// Convert to ArenaKey for sealed chunks.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_arena_key(&self) -> ArenaKey {
        match self {
            Self::GpuFloat(dt) => ArenaKey::gpu_float(*dt),
            Self::GpuQuant(fmt) => ArenaKey::gpu_quant(*fmt),
            Self::CpuFloat(dt) => ArenaKey::cpu_float(*dt),
            Self::CpuQuant(fmt) => ArenaKey::new(KvFormat::Quantized(*fmt), ArenaLocation::Cpu),
        }
    }

    /// Get the dtype used for active (GPU float) chunks.
    pub fn active_dtype(&self) -> DType {
        match self {
            Self::GpuFloat(dt) | Self::CpuFloat(dt) => *dt,
            Self::GpuQuant(_) | Self::CpuQuant(_) => DType::BF16, // Default for quant policies
        }
    }

    /// Check if sealed chunks in this policy's target format can be read by the
    /// paged attention kernel without conversion.
    ///
    /// When true, we can skip prepare_for_kernel and reconcile_sealed calls
    /// because the kernel can read the target format directly. This is a
    /// significant performance optimization for kernel-supported quantized formats.
    ///
    /// Returns true for:
    /// - GpuFloat: kernel natively supports all float formats
    /// - GpuQuant: kernel supports on-the-fly dequantization (both K and V must be supported)
    ///
    /// Returns false for:
    /// - CpuFloat/CpuQuant: chunks need GPU migration before kernel access
    /// - GpuQuant with unsupported formats: would need dequantization
    pub fn is_kernel_native(&self) -> bool {
        match self {
            Self::GpuFloat(_) => true,
            Self::GpuQuant(fmt) => fmt.is_kernel_supported(),
            Self::CpuFloat(_) | Self::CpuQuant(_) => false,
        }
    }
}

impl Default for StoragePolicy {
    fn default() -> Self {
        Self::GpuFloat(DType::BF16)
    }
}

// ==================== Chunk Status ====================

/// Status of a chunk relative to the current sequence length.
#[allow(dead_code)] // Used for chunk format migration in quantization phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStatus {
    /// Chunk not yet allocated or used
    Unallocated,
    /// Chunk is full (all tokens written) - can be any format
    Sealed,
    /// Chunk is partial (being written to) - MUST be GPU float
    Active,
}

impl ChunkStatus {
    /// Determine chunk status given sequence length and chunk parameters.
    #[allow(dead_code)] // Used for determining chunk format during migration
    pub fn from_seq_len(seq_len: usize, chunk_size: usize, block_idx: usize) -> Self {
        let chunk_start = block_idx * chunk_size;
        let chunk_end = chunk_start + chunk_size;

        if seq_len <= chunk_start {
            Self::Unallocated
        } else if seq_len >= chunk_end {
            Self::Sealed
        } else {
            Self::Active
        }
    }
}

// ==================== Arena ====================

/// A single arena holding K and V cache data as a flat combined buffer.
/// Each arena tracks its own format, location, and index.
/// K and V are stored contiguously in `data`; external strides/offsets
/// handled by the attention kernel.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Arena {
    /// Standard float arena with dtype
    Float {
        data: Tensor,
        dtype: DType,
        location: ArenaLocation,
        /// Index of this arena in the storage vector
        index: usize,
    },
    /// Block-quantized arena (K format; V may use different format for tighter packing)
    Quantized {
        data: QTensor,
        format: QuantFormat,
        location: ArenaLocation,
        /// Index of this arena in the storage vector
        index: usize,
    },
}

impl Arena {
    /// Get the KvFormat for this arena's K cache (primary format).
    pub fn format(&self) -> KvFormat {
        match self {
            Self::Float { dtype, .. } => KvFormat::Float(*dtype),
            Self::Quantized { format, .. } => KvFormat::Quantized(*format),
        }
    }

    /// Check if this arena can have new chunks allocated into it.
    /// Returns true for any live float arena (GPU or CPU).
    pub(super) fn is_allocatable(&self) -> bool {
        matches!(self, Self::Float { .. })
    }

    /// Get the index of this arena in the storage vector.
    pub fn index(&self) -> usize {
        match self {
            Self::Float { index, .. } => *index,
            Self::Quantized { index, .. } => *index,
        }
    }

    /// Get the location of this arena.
    #[allow(dead_code)] // Used by kernels for heterogeneous arena handling
    pub(super) fn location(&self) -> ArenaLocation {
        match self {
            Self::Float { location, .. } => *location,
            Self::Quantized { location, .. } => *location,
        }
    }

    /// Get the float data tensor. Returns Err if not a float arena.
    pub(super) fn float_data(&self) -> Result<&Tensor> {
        match self {
            Self::Float { data, .. } => Ok(data),
            Self::Quantized { .. } => candle::bail!("expected float arena"),
        }
    }

    /// Get mutable float data tensor. Returns Err if not a float arena.
    #[allow(dead_code)]
    pub(super) fn float_data_mut(&mut self) -> Result<&mut Tensor> {
        match self {
            Self::Float { data, .. } => Ok(data),
            Self::Quantized { .. } => candle::bail!("expected float arena"),
        }
    }

    /// Get the quantized data tensor. Returns Err if not a quantized arena.
    pub(super) fn quantized_data(&self) -> Result<&QTensor> {
        match self {
            Self::Float { .. } => candle::bail!("expected quantized arena"),
            Self::Quantized { data, .. } => Ok(data),
        }
    }

    /// Get mutable quantized data tensor. Returns Err if not a quantized arena.
    pub(super) fn quantized_data_mut(&mut self) -> Result<&mut QTensor> {
        match self {
            Self::Float { .. } => candle::bail!("expected quantized arena"),
            Self::Quantized { data, .. } => Ok(data),
        }
    }

    /// Get data tensor if this is a float arena, None otherwise.
    pub(super) fn as_float_data(&self) -> Option<&Tensor> {
        match self {
            Self::Float { data, .. } => Some(data),
            Self::Quantized { .. } => None,
        }
    }

    /// Get data tensor if this is a quantized arena, None otherwise.
    pub(super) fn as_quantized_data(&self) -> Option<&QTensor> {
        match self {
            Self::Float { .. } => None,
            Self::Quantized { data, .. } => Some(data),
        }
    }

    /// Create an ArenaEntry for this arena, extracting device pointers if on GPU.
    /// For CPU arenas, pointers will be 0.
    pub(super) fn to_arena_entry(&self) -> ArenaEntry {
        // CP3 COLLAPSE POINT: Both k_tag and v_tag are set to the same format_tag.
        // This is correct for uniform (K==V format) arenas but wrong for K≠V format
        // arenas. Fix by accepting per-head k_tags/v_tags arrays and populating
        // each head's entry independently. See docs/kv_cache_unification.md §7.6.
        let format_tag = ArenaFormatTag::from_kv_format(self.format());
        let location = self.location();

        match self {
            Self::Float { data, .. } => {
                if location == ArenaLocation::Cpu {
                    ArenaEntry::new_cpu(format_tag, format_tag)
                } else {
                    let ptr = Self::extract_tensor_ptr(data).unwrap_or(0);
                    ArenaEntry::new_gpu(ptr, ptr, format_tag, format_tag)
                }
            }
            Self::Quantized { data: _data, .. } => {
                if location == ArenaLocation::Cpu {
                    ArenaEntry::new_cpu(format_tag, format_tag)
                } else {
                    #[cfg(feature = "cuda")]
                    let ptr = _data.cuda_data_ptr().unwrap_or(0);
                    #[cfg(not(feature = "cuda"))]
                    let ptr = 0u64;
                    ArenaEntry::new_gpu(ptr, ptr, format_tag, format_tag)
                }
            }
        }
    }

    /// Helper to extract the raw device pointer from a tensor.
    /// Returns None if the tensor is not on CUDA.
    #[cfg(feature = "cuda")]
    fn extract_tensor_ptr(t: &Tensor) -> Option<u64> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use half::{bf16, f16};

        let (storage, layout) = t.storage_and_layout();
        let cuda_storage = match &*storage {
            candle::Storage::Cuda(c) => c,
            _ => return None,
        };

        // Get the CudaDevice to access the stream
        let cuda_device = cuda_storage.device();
        let stream = cuda_device.cuda_stream();

        // Handle different dtypes - extract pointer with stream for proper synchronization
        let ptr = match t.dtype() {
            DType::F32 => {
                let slice = cuda_storage.as_cuda_slice::<f32>().ok()?;
                let slice = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            DType::F16 => {
                let slice = cuda_storage.as_cuda_slice::<f16>().ok()?;
                let slice = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            DType::BF16 => {
                let slice = cuda_storage.as_cuda_slice::<bf16>().ok()?;
                let slice = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            DType::F8E4M3 => {
                let slice = cuda_storage.as_cuda_slice::<float8::F8E4M3>().ok()?;
                let slice = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            DType::U8 => {
                let slice = cuda_storage.as_cuda_slice::<u8>().ok()?;
                let slice = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            _ => return None,
        };
        Some(ptr)
    }

    #[cfg(not(feature = "cuda"))]
    fn extract_tensor_ptr(_t: &Tensor) -> Option<u64> {
        None
    }

    /// Get the ArenaKey (K/V formats + location) for this arena.
    pub fn arena_key(&self) -> ArenaKey {
        ArenaKey::new(self.format(), self.location())
    }

    /// Zero the chunk at `chunk_idx`. Called on the alloc-from-free-list path
    /// so the chunk's bytes are clean before the new tenant writes only the
    /// slots it cares about — the persist quantize pass then sees zero past
    /// `token_count` instead of stale garbage from the prior tenant.
    ///
    /// Asynchronous when `stream` is supplied: the work is enqueued on that
    /// stream and the call returns once queued. Same-stream FIFO ordering
    /// then guarantees any subsequent kernel reading this chunk sees zeros
    /// without an explicit fence. When `stream` is `None` (CPU arena or a
    /// CPU device under a `cuda`-built binary) the work runs synchronously.
    pub(super) fn zero_chunk_at(
        &mut self,
        chunk_idx: usize,
        #[cfg(feature = "cuda")] stream: Option<
            &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
        >,
    ) -> Result<()> {
        match self {
            Self::Float { data, dtype, .. } => {
                let dims = data.dims();
                if dims.len() != 3 {
                    candle::bail!(
                        "zero_chunk_at: expected (arena_chunks, chunk_size, sub_head_dim), got {:?}",
                        dims
                    );
                }
                if chunk_idx >= dims[0] {
                    candle::bail!(
                        "zero_chunk_at: chunk {chunk_idx} out of range (arena holds {})",
                        dims[0]
                    );
                }
                // Tensor::zeros + slice_set are kernel-launch ops; on CUDA
                // they enqueue on the tensor's stream and return without
                // blocking. On CPU they run inline.
                let zeros = Tensor::zeros((1, dims[1], dims[2]), *dtype, data.device())?;
                data.slice_set(&zeros, 0, chunk_idx)?;
                Ok(())
            }
            Self::Quantized { data, format, .. } => {
                let (byte_offset, chunk_byte_stride) =
                    quant_chunk_byte_range(data, *format, chunk_idx)?;
                let zeros = vec![0u8; chunk_byte_stride];
                #[cfg(feature = "cuda")]
                {
                    if let Some(s) = stream {
                        data.write_bytes_at_async(s, byte_offset, &zeros)?;
                        return Ok(());
                    }
                }
                data.write_bytes_at(byte_offset, &zeros)?;
                Ok(())
            }
        }
    }

    /// Estimate the GPU memory usage of this arena in bytes.
    ///
    /// Returns the total bytes for both K and V tensors.
    /// For float arenas, this is `2 × elem_count × dtype_size`.
    /// For quantized arenas, this uses the QTensor's storage size.
    pub fn gpu_memory_bytes(&self) -> usize {
        match self {
            Self::Float {
                data,
                dtype,
                location,
                ..
            } => {
                if *location != ArenaLocation::Gpu {
                    return 0;
                }
                let elem_count = data.elem_count();
                let bytes_per_elem = dtype.size_in_bytes();
                elem_count * bytes_per_elem
            }
            Self::Quantized { data, location, .. } => {
                if *location != ArenaLocation::Gpu {
                    return 0;
                }
                data.storage_size_in_bytes()
            }
        }
    }

    /// Return a label describing this arena's format.
    pub fn format_label(&self) -> String {
        match self {
            Self::Float { dtype, .. } => format!("{:?}", dtype),
            Self::Quantized { format, .. } => format!("{:?}", format),
        }
    }
}

/// Resolve `(byte_offset, chunk_byte_stride)` for a Quantized arena chunk.
/// Used by [`Arena::zero_chunk_at`] to address one logical chunk's bytes.
fn quant_chunk_byte_range(
    data: &QTensor,
    format: QuantFormat,
    chunk_idx: usize,
) -> Result<(usize, usize)> {
    let q_ggml = format.to_ggml_dtype();
    let total_elems = data.shape().elem_count();
    if total_elems % q_ggml.block_size() != 0 {
        candle::bail!(
            "quant_chunk_byte_range: total elems {total_elems} not divisible by block_size {}",
            q_ggml.block_size()
        );
    }
    let total_bytes = (total_elems / q_ggml.block_size()) * q_ggml.type_size();
    let arena_chunks = arena_chunks_for_format(KvFormat::Quantized(format));
    if arena_chunks == 0 {
        candle::bail!(
            "quant_chunk_byte_range: arena_chunks_for_format returned 0 for {:?}",
            format
        );
    }
    let chunk_byte_stride = total_bytes / arena_chunks;
    if chunk_idx >= arena_chunks {
        candle::bail!(
            "quant_chunk_byte_range: chunk {chunk_idx} out of range (arena holds {arena_chunks})"
        );
    }
    Ok((chunk_idx * chunk_byte_stride, chunk_byte_stride))
}

#[allow(dead_code)]
struct _ArenaImplContinues; // syntactic anchor: more `impl Arena` items follow below.

impl Arena {
    /// Return the raw device pointer and byte stride for one logical chunk in
    /// this arena. Returns `None` for CPU-backed or tombstoned arenas.
    #[allow(dead_code)]
    pub(super) fn chunk_copy_span(&self, chunk_idx: usize) -> Option<(u64, u32)> {
        match self {
            Self::Float {
                data,
                dtype,
                location,
                ..
            } => {
                if *location != ArenaLocation::Gpu {
                    return None;
                }
                let base = Self::extract_tensor_ptr(data)?;
                let total_bytes = data.elem_count().checked_mul(dtype.size_in_bytes())?;
                let arena_chunks = arena_chunks_for_format(KvFormat::Float(*dtype));
                if total_bytes == 0 || total_bytes % arena_chunks != 0 {
                    return None;
                }
                let stride = total_bytes / arena_chunks;
                let offset = chunk_idx.checked_mul(stride)?;
                Some((base + offset as u64, stride as u32))
            }
            Self::Quantized {
                data,
                format,
                location,
                ..
            } => {
                if *location != ArenaLocation::Gpu {
                    return None;
                }
                #[cfg(feature = "cuda")]
                {
                    let base = data.cuda_data_ptr()?;
                    let total_bytes = data.storage_size_in_bytes();
                    let arena_chunks = arena_chunks_for_format(KvFormat::Quantized(*format));
                    if total_bytes == 0 || total_bytes % arena_chunks != 0 {
                        return None;
                    }
                    let stride = total_bytes / arena_chunks;
                    let offset = chunk_idx.checked_mul(stride)?;
                    Some((base + offset as u64, stride as u32))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (data, format, chunk_idx);
                    None
                }
            }
        }
    }
}

// ==================== Arena Storage State ====================

/// The mutable state of arena storage - NO LOCKS HERE.
/// All methods on this struct are lock-free and can safely call each other.
/// This is the key to preventing deadlocks: the lock is held by ArenaStorage,
/// and ArenaStorageState methods cannot acquire any locks.
///
/// Per-arena slot allocation state is NOT tracked here — the
/// [`ChunkGidPool`]'s lock-free `ArenaRefcounts` table is the
/// authoritative record of "which chunk slots are live". Storage only
/// keeps the physical arena tensors.
#[derive(Debug)]
pub(crate) struct ArenaStorageState {
    /// The arenas themselves, keyed by arena index.
    pub(super) arenas: AHashMap<usize, Arena>,
}

impl ArenaStorageState {
    /// Create new empty state.
    fn new() -> Self {
        // Sized to typical steady-state working set: ~30-60 live arenas across
        // all formats (production runs peak around 28). Pre-sizing avoids a
        // chain of ~6 rehash/grow cycles during warmup.
        Self {
            arenas: AHashMap::with_capacity(64),
        }
    }

    /// Get the number of arenas.
    pub(super) fn arena_count(&self) -> usize {
        self.arenas.len()
    }

    /// Get a reference to an arena by index.
    #[allow(dead_code)] // Used for direct arena access in heterogeneous operations
    pub(super) fn arena(&self, idx: usize) -> Option<&Arena> {
        self.arenas.get(&idx)
    }

    /// Get a mutable reference to an arena by index.
    #[allow(dead_code)] // Used for in-place arena operations (write, resize)
    pub(super) fn arena_mut(&mut self, idx: usize) -> Option<&mut Arena> {
        self.arenas.get_mut(&idx)
    }

    /// Get all arenas as a HashMap.
    pub(super) fn arenas(&self) -> &AHashMap<usize, Arena> {
        &self.arenas
    }

    /// Get all arenas as a mutable HashMap.
    pub(super) fn arenas_mut(&mut self) -> &mut AHashMap<usize, Arena> {
        &mut self.arenas
    }

    /// Check if an arena exists at the given index.
    pub(super) fn has_arena(&self, idx: usize) -> bool {
        self.arenas.contains_key(&idx)
    }

    /// Truncate arenas to the given count.
    pub(super) fn truncate(&mut self, count: usize) {
        self.arenas.retain(|&k, _| k < count);
    }

    /// Get the ArenaKey for a specific arena.
    pub(super) fn arena_key(&self, arena_idx: usize) -> Option<ArenaKey> {
        self.arenas.get(&arena_idx).map(|a| a.arena_key())
    }

    /// Insert a freshly created arena into storage. Slot-level allocation
    /// state lives in the [`ChunkGidPool`]; storage only owns the tensor.
    /// `_arena_chunks` is kept for call-site parity with the legacy
    /// alloc-state path but is no longer consulted here.
    pub(super) fn push_arena(&mut self, arena: Arena, arena_idx: usize, _arena_chunks: usize) {
        self.arenas.insert(arena_idx, arena);
    }

    /// Release an empty arena, removing it from storage and freeing its GPU tensors.
    pub(super) fn release_arena(&mut self, arena_idx: usize) {
        self.arenas.remove(&arena_idx);
    }

    /// Check if an arena at the given index can have chunks allocated into it.
    pub(super) fn is_allocatable(&self, arena_idx: usize) -> bool {
        if let Some(arena) = self.arenas.get(&arena_idx) {
            arena.is_allocatable()
        } else {
            false
        }
    }
}

// ==================== Arena Diagnostic Row ====================

/// Per-arena diagnostic data for the arena table report.
#[derive(Debug, Clone)]
pub(crate) struct ArenaRow {
    /// Arena slot index in the storage vector.
    pub arena_idx: usize,
    /// Human-readable type label, e.g. "Float BF16 Gpu" or "Quant Q8_0 Gpu".
    pub type_label: String,
    /// True if this slot has been tombstoned (GPU tensors freed, not allocatable).
    pub is_tombstone: bool,
    /// Maximum chunks this arena can hold (0 for tombstones).
    pub capacity: usize,
    /// High-water mark: highest chunk index ever handed out from this arena.
    pub hwm: usize,
    /// Number of chunks currently in active use (patched by backing from pool).
    pub active: usize,
    /// Number of freed chunks sitting in the free list (patched by backing from pool).
    pub free_list: usize,
    /// GPU memory in bytes currently occupied by this arena (0 for tombstones/CPU).
    pub gpu_bytes: usize,
}

impl ArenaRow {
    /// Return the number of "wasted" slots = free_list + (capacity - hwm).
    #[allow(dead_code)]
    pub fn fragmented_chunks(&self) -> usize {
        self.free_list
    }

    /// Return true if every capacity slot is actively in use.
    pub fn is_full(&self) -> bool {
        !self.is_tombstone && self.capacity > 0 && self.active >= self.capacity
    }

    /// Return true if the arena has capacity but zero active chunks.
    pub fn is_empty(&self) -> bool {
        !self.is_tombstone && self.capacity > 0 && self.active == 0
    }
}

impl ArenaStorageState {
    /// Build a per-arena diagnostic row for every slot, sorted by index.
    ///
    /// `capacity` is derived directly from the arena's format. `active`,
    /// `free_list`, and `hwm` are placeholders here — `backing.rs`
    /// patches them in from the [`ChunkGidPool`] (the authoritative
    /// per-slot owner) before exposing rows to callers.
    pub(super) fn arena_rows(&self) -> Vec<ArenaRow> {
        let mut indices: Vec<usize> = self.arenas.keys().copied().collect();
        indices.sort_unstable();
        indices
            .iter()
            .map(|&idx| {
                let arena = &self.arenas[&idx];
                let capacity = arena_chunks_for_format(arena.arena_key().format);
                let gpu_bytes = arena.gpu_memory_bytes();
                let type_label = match arena {
                    Arena::Float {
                        dtype, location, ..
                    } => {
                        format!("Float {:?} {:?}", dtype, location)
                    }
                    Arena::Quantized {
                        format, location, ..
                    } => {
                        format!("Quant {:?} {:?}", format, location)
                    }
                };
                ArenaRow {
                    arena_idx: idx,
                    type_label,
                    is_tombstone: false,
                    capacity,
                    hwm: 0,
                    active: 0,
                    free_list: 0,
                    gpu_bytes,
                }
            })
            .collect()
    }
}

// ==================== Arena Storage ====================

/// Storage for KV arenas - supports heterogeneous formats and locations.
///
/// # Deadlock Prevention
///
/// This type uses a single RwLock over ArenaStorageState. All mutable operations
/// go through `write()` which takes a closure. This design makes deadlocks impossible:
///
/// 1. There is only ONE lock - no lock ordering issues
/// 2. ArenaStorageState has NO locks - its methods can't accidentally re-acquire
/// 3. The closure pattern prevents holding the lock while calling back into ArenaStorage
/// 4. Immutable config (format, location) is outside the lock for lock-free access
#[derive(Debug)]
pub(crate) struct ArenaStorage {
    /// All mutable state behind a single lock.
    state: RwLock<ArenaStorageState>,
    /// Default K format for new arenas (immutable, lock-free access).
    default_format: KvFormat,
    /// Default V format for new arenas (immutable, lock-free access).
    default_v_format: KvFormat,
    /// Default location for new arenas (immutable, lock-free access).
    default_location: ArenaLocation,
}

impl ArenaStorage {
    /// Create new arena storage with defaults.
    pub(super) fn new(k_format: KvFormat, v_format: KvFormat, location: ArenaLocation) -> Self {
        Self {
            state: RwLock::new(ArenaStorageState::new()),
            default_format: k_format,
            default_v_format: v_format,
            default_location: location,
        }
    }

    /// Check if default format is quantized (lock-free).
    pub(super) fn is_quantized(&self) -> bool {
        self.default_format.is_quantized()
    }

    /// Get the default KvFormat for new arenas (lock-free).
    pub(super) fn k_format(&self) -> KvFormat {
        self.default_format
    }

    /// Get the default V KvFormat for new arenas (lock-free).
    pub(super) fn v_format(&self) -> KvFormat {
        self.default_v_format
    }

    /// Get the default location for new arenas (lock-free).
    pub(super) fn default_location(&self) -> ArenaLocation {
        self.default_location
    }

    /// Get the DType for default format, or None if quantized (lock-free).
    #[allow(dead_code)]
    pub(super) fn dtype(&self) -> Option<DType> {
        self.default_format.dtype()
    }

    /// Execute a read operation on the arena state.
    /// The closure receives an immutable reference to the state.
    pub(super) fn read<R>(&self, f: impl FnOnce(&ArenaStorageState) -> R) -> Result<R> {
        let guard = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("arena storage lock poisoned".into()))?;
        Ok(f(&guard))
    }

    /// Execute a write operation on the arena state.
    /// The closure receives a mutable reference to the state.
    pub(super) fn write<R>(&self, f: impl FnOnce(&mut ArenaStorageState) -> R) -> Result<R> {
        let mut guard = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("arena storage lock poisoned".into()))?;
        Ok(f(&mut guard))
    }

    /// Execute a fallible write operation on the arena state.
    /// The closure receives a mutable reference and can return a Result.
    pub(super) fn try_write<R>(
        &self,
        f: impl FnOnce(&mut ArenaStorageState) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("arena storage lock poisoned".into()))?;
        f(&mut guard)
    }

    // ==================== Convenience methods ====================
    // These acquire the lock internally for simple operations.
    // For complex operations, use read()/write() directly.

    /// Get the number of arenas currently allocated.
    pub(super) fn arena_count(&self) -> Result<usize> {
        self.read(|s| s.arena_count())
    }

    /// Truncate arenas to the given count.
    pub(super) fn truncate_arenas(&self, count: usize) -> Result<()> {
        self.write(|s| s.truncate(count))
    }

    /// Get the ArenaKey for a specific arena.
    #[allow(dead_code)] // Used for arena key lookups during heterogeneous operations
    pub(super) fn arena_key(&self, arena_idx: usize) -> Result<ArenaKey> {
        self.read(|s| {
            s.arena_key(arena_idx)
                .ok_or_else(|| candle::Error::Msg(format!("arena {} not found", arena_idx)))
        })?
    }

    /// Check if an arena at the given index can have chunks allocated into it.
    #[allow(dead_code)]
    pub(super) fn is_allocatable(&self, arena_idx: usize) -> Result<bool> {
        self.read(|s| s.is_allocatable(arena_idx))
    }

    /// Returns the actual K/V format tags from the first live arena.
    ///
    /// Falls back to the backing's configured default formats if no arenas exist yet.
    /// Use this for kernel dispatch — the backing default may be a quantized target
    /// while actual arenas are still float (e.g. pre-reconcile).
    pub(super) fn actual_kv_format_tags(
        &self,
    ) -> (
        crate::kv_cache::ArenaFormatTag,
        crate::kv_cache::ArenaFormatTag,
    ) {
        use crate::kv_cache::ArenaFormatTag;
        let from_format = |f: KvFormat| ArenaFormatTag::from_kv_format(f);
        self.read(|s| {
            if let Some(arena) = s.arenas().values().next() {
                let entry = arena.to_arena_entry();
                (entry.k_format_tag, entry.v_format_tag)
            } else {
                (
                    from_format(self.default_format),
                    from_format(self.default_v_format),
                )
            }
        })
        .unwrap_or_else(|_| {
            (
                from_format(self.default_format),
                from_format(self.default_v_format),
            )
        })
    }

    /// Release an empty arena, freeing its GPU tensors.
    /// Called by compact_arenas after the pool has tombstoned the arena.
    pub(super) fn release_arena(&self, arena_idx: usize) -> Result<()> {
        self.write(|s| s.release_arena(arena_idx))?;
        Ok(())
    }

    /// Build per-arena diagnostic rows. The `active` and `free_list` fields
    /// are set to zero/storage-local values; backing.rs patches them from pool data.
    pub(super) fn arena_rows(&self) -> Result<Vec<ArenaRow>> {
        self.read(|s| s.arena_rows())
    }

    /// Return the raw device pointer and byte stride for a specific chunk.
    #[allow(dead_code)]
    pub(super) fn chunk_copy_span(&self, arena_idx: usize, chunk_idx: usize) -> Result<(u64, u32)> {
        self.read(|s| {
            s.arena(arena_idx)
                .and_then(|arena| arena.chunk_copy_span(chunk_idx))
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "arena {} chunk {} has no GPU copy span",
                        arena_idx, chunk_idx
                    ))
                })
        })?
    }
}
