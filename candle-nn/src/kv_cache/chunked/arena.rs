//! Arena storage types and management for ChunkedKvBacking.
//!
//! This module contains:
//! - `ArenaKey` - Key identifying an arena by size class and location
//! - `StoragePolicy` - How sealed chunks should be stored
//! - `ChunkStatus` - Status of a chunk (Unallocated/Sealed/Active)
//! - `Arena` - A run of fixed-stride byte slots on one device
//! - `ArenaStorage` - Collection of arenas across classes and locations

use crate::kv_cache::{arena_table::ArenaLocation, KvFormat, QuantFormat};
use ahash::AHashMap;
#[cfg(feature = "cuda")]
use candle::quantized::LiveQTensor;
use candle::LiveTensor;
use candle::{DType, Result, Tensor};
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::region_pool::RegionHandle;
use crate::kv_cache::chunked::size_class::{class_for_format, SizeClass};

// ==================== Arena Key ====================

/// Key identifying an arena by its **size class** and location.
///
/// Not by format. An arena is a run of fixed-stride byte slots; a chunk of
/// format F occupies one slot of the smallest class that fits its bytes, and
/// the trailing pad is never read. Formats that share a class share a pool,
/// which is what makes free slots fungible across formats — the property the
/// whole initiative exists to obtain (`docs/archived/arena_unification.md` §3.4).
///
/// What a slot *holds* is recorded on the chunk (`SealedChunk::k_fmt`), never
/// here.
#[derive(Debug, Clone, Copy)]
pub struct ArenaKey {
    pub class: SizeClass,
    pub location: ArenaLocation,
}

impl PartialEq for ArenaKey {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class && self.location == other.location
    }
}

impl Eq for ArenaKey {}

impl Hash for ArenaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.class.hash(state);
        self.location.hash(state);
    }
}

impl ArenaKey {
    pub fn new(class: SizeClass, location: ArenaLocation) -> Self {
        Self { class, location }
    }

    /// The key a chunk of `format` allocates from, given the chunk geometry.
    ///
    /// Errors only if the format maps to no class, which
    /// `every_kv_format_maps_to_a_class` proves cannot happen for any
    /// `KvFormat` at the production geometry — so a failure here means the
    /// ladder and the format space have drifted apart.
    pub fn for_format(
        format: KvFormat,
        elems_per_chunk: usize,
        location: ArenaLocation,
    ) -> Result<Self> {
        let class = class_for_format(format, elems_per_chunk).ok_or_else(|| {
            candle::Error::Msg(format!(
                "no size class covers {format:?} at {elems_per_chunk} elems/chunk"
            ))
        })?;
        Ok(Self { class, location })
    }

    pub fn is_gpu(&self) -> bool {
        self.location == ArenaLocation::Gpu
    }

    /// Byte stride between consecutive chunk slots in an arena with this key.
    pub fn slot_stride(&self) -> usize {
        self.class.bytes()
    }

    /// How many chunk slots an arena with this key holds.
    pub fn chunks(&self) -> usize {
        self.class.chunks_per_region()
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
    /// The format sealed chunks are stored in under this policy.
    pub fn target_format(&self) -> KvFormat {
        match self {
            Self::GpuFloat(dt) | Self::CpuFloat(dt) => KvFormat::Float(*dt),
            Self::GpuQuant(fmt) | Self::CpuQuant(fmt) => KvFormat::Quantized(*fmt),
        }
    }

    /// Where sealed chunks live under this policy.
    pub fn location(&self) -> ArenaLocation {
        match self {
            Self::GpuFloat(_) | Self::GpuQuant(_) => ArenaLocation::Gpu,
            Self::CpuFloat(_) | Self::CpuQuant(_) => ArenaLocation::Cpu,
        }
    }

    /// Whether sealed chunks are block-quantized under this policy.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::GpuQuant(_) | Self::CpuQuant(_))
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

/// A run of fixed-stride byte slots.
///
/// One arena is `chunks × class.bytes()` raw bytes on one device. Slot `i`
/// starts at `base + i * class.bytes()`, and what lives there is whatever the
/// owning chunk says it is — the arena does not know, and does not need to.
/// That is the point: under size classes a slot's tenant can be any format
/// whose payload fits, so free slots are fungible across formats
/// (`docs/archived/arena_unification.md` §3.4, principle 8).
///
/// The trailing pad between a chunk's payload and the slot stride is never
/// read as data: kernels derive their read extent from the *format's* block
/// size, never from the stride (audit A6).
/// The slab is **one-dimensional** — `chunks * class.bytes()` bytes, not a
/// `(chunks, stride)` matrix. A band's payload is generally shorter than the
/// slot it lives in, so every read and write is a byte *range* rather than a
/// whole row; on a 1-D slab `narrow(0, off, len)` and `slice_set(src, 0, off)`
/// express exactly that, contiguously, on either device. A 2-D slab would force
/// every partial write to either pad up to the full stride (moving pad over
/// PCIe, which invariant 8 forbids) or go through raw pointer writes.
#[derive(Debug)]
pub struct Arena {
    /// The slab: `U8`, shape `(chunks * class.bytes(),)`.
    data: Tensor,
    class: SizeClass,
    location: ArenaLocation,
    /// Index of this arena in the storage map.
    index: usize,
    /// The reservation region this arena's bytes are carved from — `None` for a
    /// CPU arena, whose slab is an ordinary host allocation.
    ///
    /// Owning the handle is what keeps the region out of the free list, so
    /// every path that drops an arena returns its region: release, truncate, or
    /// the whole backing going away. There is no "free the arena" step to
    /// forget, and nothing here calls the CUDA allocator in either direction.
    #[cfg(feature = "cuda")]
    region: Option<RegionHandle>,
}

impl Arena {
    /// Wrap a byte slab that owns its own storage — a CPU arena.
    ///
    /// A GPU arena is carved instead: [`Self::in_region`].
    pub(super) fn new(
        data: Tensor,
        class: SizeClass,
        location: ArenaLocation,
        index: usize,
    ) -> Self {
        debug_assert_eq!(data.dtype(), DType::U8, "an arena slab is raw bytes");
        debug_assert_eq!(data.rank(), 1, "an arena slab is flat");
        Self {
            data,
            class,
            location,
            index,
            #[cfg(feature = "cuda")]
            region: None,
        }
    }

    /// Attach the reservation region this arena's bytes are carved from.
    ///
    /// The slab tensor is a lease over the region, so the handle is the only
    /// thing keeping those bytes claimed — and dropping the arena is what
    /// returns them.
    #[cfg(feature = "cuda")]
    pub(super) fn in_region(mut self, region: RegionHandle) -> Self {
        self.region = Some(region);
        self
    }

    /// This arena's size class.
    pub fn class(&self) -> SizeClass {
        self.class
    }

    /// Byte stride between consecutive chunk slots.
    #[inline]
    pub fn slot_stride(&self) -> usize {
        self.class.bytes()
    }

    /// Number of chunk slots.
    #[inline]
    pub fn chunks(&self) -> usize {
        self.class.chunks_per_region()
    }

    /// The raw byte slab.
    pub(super) fn byte_data(&self) -> &Tensor {
        &self.data
    }

    /// Index of this arena in the storage map.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Where this arena's bytes live.
    pub(super) fn location(&self) -> ArenaLocation {
        self.location
    }

    /// Every arena can accept new chunks: a slot is a slot.
    ///
    /// Under the old Float/Quantized duality only float arenas were
    /// allocatable, because a quantized arena's slots were written by the
    /// convert kernel rather than claimed by the allocator. Classes remove the
    /// distinction.
    pub(super) fn is_allocatable(&self) -> bool {
        true
    }

    /// Device pointer to slot 0, or `None` for a CPU arena.
    pub(super) fn base_ptr(&self) -> Option<u64> {
        if self.location != ArenaLocation::Gpu {
            return None;
        }
        Self::extract_tensor_ptr(&self.data)
    }

    /// Device pointer to slot `chunk_idx`.
    pub(super) fn slot_ptr(&self, chunk_idx: usize) -> Option<u64> {
        let base = self.base_ptr()?;
        Some(base + (chunk_idx * self.slot_stride()) as u64)
    }

    /// A typed tensor **view** over `chunk_idx`'s slot, shaped `shape`.
    ///
    /// The view is a lease: it aliases the arena's bytes and its drop never
    /// frees them (`Backing::Lease`, `docs/archived/arena_unification.md` §3.7). This is
    /// how the contiguous read/write facade still sees a slot as a typed
    /// tensor now that the arena itself is untyped.
    ///
    /// Errors if the requested view would run past the slot stride, so a wrong
    /// shape is a named host error rather than a read into the next tenant.
    #[cfg(feature = "cuda")]
    pub(super) fn slot_view<'a, S: Into<candle::Shape>>(
        &'a self,
        chunk_idx: usize,
        dtype: DType,
        shape: S,
    ) -> Result<LiveTensor<'a>> {
        let shape = shape.into();
        let want = shape.elem_count() * dtype.size_in_bytes();
        let stride = self.slot_stride();
        if want > stride {
            candle::bail!(
                "slot_view: {:?} of {dtype:?} needs {want} B but the slot stride is {stride}",
                shape
            );
        }
        if chunk_idx >= self.chunks() {
            candle::bail!(
                "slot_view: chunk {chunk_idx} out of range (arena holds {})",
                self.chunks()
            );
        }
        let ptr = self.slot_ptr(chunk_idx).ok_or_else(|| {
            candle::Error::Msg("slot_view: arena is not GPU-resident".to_string())
        })?;
        // SAFETY: the bounds check above keeps the view inside slot
        // `chunk_idx`, and zero-on-recycle (invariant 4) means the bytes are
        // always a legal bit pattern. That the slab outlives the view is no
        // longer an obligation here: `'a` ties it to this borrow of the arena,
        // exactly as for `qslot_view`.
        unsafe { LiveTensor::from_leased_cuda_ptr(ptr, dtype, shape, self.data.device()) }
    }

    /// Byte offset of slot `chunk_idx`, bounds-checked against `len`.
    ///
    /// `len` is the band's **payload**, not the stride: the pad between them is
    /// not part of the chunk and must not be copied (invariant 8). The offset,
    /// by contrast, always steps by the stride — deriving it from `len` would
    /// address slot `n` at `n * payload` and walk into a neighbour.
    fn slot_offset(&self, chunk_idx: usize, len: usize) -> Result<usize> {
        let stride = self.slot_stride();
        if len > stride {
            candle::bail!("arena slot: {len} B requested from a {stride} B slot (class {stride})");
        }
        if chunk_idx >= self.chunks() {
            candle::bail!(
                "arena slot: chunk {chunk_idx} out of range (arena holds {})",
                self.chunks()
            );
        }
        Ok(chunk_idx * stride)
    }

    /// A read-only byte view of `len` bytes at the head of slot `chunk_idx`.
    ///
    /// Works on either device — the slab is a plain `U8` tensor, so this is a
    /// contiguous `narrow` rather than a device-pointer lease.
    pub(crate) fn slot_bytes(&self, chunk_idx: usize, len: usize) -> Result<Tensor> {
        let off = self.slot_offset(chunk_idx, len)?;
        self.data.narrow(0, off, len)
    }

    /// Write `src` (a 1-D `U8` tensor on this arena's device) into the head of
    /// slot `chunk_idx`.
    pub(crate) fn write_slot_bytes(&mut self, chunk_idx: usize, src: &Tensor) -> Result<()> {
        self.write_slot_bytes_at(chunk_idx, 0, src)
    }

    /// Write `src` into slot `chunk_idx` starting `byte_offset` bytes in.
    pub(super) fn write_slot_bytes_at(
        &mut self,
        chunk_idx: usize,
        byte_offset: usize,
        src: &Tensor,
    ) -> Result<()> {
        if src.dtype() != DType::U8 || src.rank() != 1 {
            candle::bail!(
                "write_slot_bytes: expected a 1-D U8 tensor, got {:?} {:?}",
                src.dtype(),
                src.shape()
            );
        }
        let off = self.slot_offset(chunk_idx, byte_offset + src.elem_count())? + byte_offset;
        self.data.slice_set(src, 0, off)
    }

    /// A typed tensor over `elems` elements at the head of slot `chunk_idx`,
    /// shaped `shape`.
    ///
    /// On a GPU arena this is a zero-copy lease over the slab's own bytes, so a
    /// write into the returned tensor lands in the arena. On a CPU arena the
    /// bytes are decoded on the host into a fresh tensor — a *copy*, so use
    /// [`Self::write_slot_typed`] rather than mutating the result.
    pub(crate) fn read_slot_typed<'a, S: Into<candle::Shape>>(
        &'a self,
        chunk_idx: usize,
        dtype: DType,
        shape: S,
    ) -> Result<LiveTensor<'a>> {
        let shape = shape.into();
        #[cfg(feature = "cuda")]
        if self.location == ArenaLocation::Gpu {
            return self.slot_view(chunk_idx, dtype, shape);
        }
        let bytes = self
            .slot_bytes(chunk_idx, shape.elem_count() * dtype.size_in_bytes())?
            .to_vec1::<u8>()?;
        decode_bytes(&bytes, dtype, shape, self.data.device())
    }

    /// Write a typed tensor into slot `chunk_idx`, `elem_offset` elements in.
    ///
    /// The GPU path writes straight into a lease over the slab; the CPU path
    /// encodes `src` to bytes on the host and splices them into the slot. Both
    /// keep the arena itself untyped — the caller's dtype comes from the band's
    /// tag, never from the arena.
    pub(crate) fn write_slot_typed(
        &mut self,
        chunk_idx: usize,
        elem_offset: usize,
        src: &Tensor,
    ) -> Result<()> {
        let dtype = src.dtype();
        #[cfg(feature = "cuda")]
        if self.location == ArenaLocation::Gpu {
            let slot_elems = self.slot_stride() / dtype.size_in_bytes();
            let view = self.slot_view(chunk_idx, dtype, slot_elems)?;
            return view.slice_set(&src.flatten_all()?, 0, elem_offset);
        }
        let bytes = encode_bytes(&src.flatten_all()?)?;
        let host = Tensor::from_slice(&bytes, bytes.len(), self.data.device())?;
        self.write_slot_bytes_at(chunk_idx, elem_offset * dtype.size_in_bytes(), &host)
    }

    /// Helper to extract the raw device pointer from a byte slab.
    /// Returns None if the tensor is not on CUDA.
    #[cfg(feature = "cuda")]
    fn extract_tensor_ptr(t: &Tensor) -> Option<u64> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;

        let (storage, layout) = t.storage_and_layout();
        let cuda_storage = match &*storage {
            candle::Storage::Cuda(c) => c,
            _ => return None,
        };
        let stream = cuda_storage.device().cuda_stream();
        let slice = cuda_storage.as_cuda_slice::<u8>().ok()?;
        let slice = slice.slice(layout.start_offset()..);
        let (ptr, _guard) = slice.device_ptr(&stream);
        Some(ptr)
    }

    #[cfg(not(feature = "cuda"))]
    fn extract_tensor_ptr(_t: &Tensor) -> Option<u64> {
        None
    }

    /// A `QTensor` **view** over slot `chunk_idx`, covering exactly the
    /// `elems`-element band that lives there.
    ///
    /// The quantized twin of [`Self::slot_view`], and the way the block
    /// quantize / dequantize kernels reach a slot now that the arena carries no
    /// format. Like `slot_view` it is a lease: writes through it land in the
    /// arena, and dropping it frees nothing.
    ///
    /// The view spans the band's **payload**, never the class stride — a
    /// stride is not generally a whole number of blocks, and the bytes past the
    /// payload belong to no chunk. That also makes every offset into the
    /// returned tensor *slot-local*, which is what retires the arena-global
    /// `chunk_idx * elems_per_chunk + ...` arithmetic invariant 8 is about.
    ///
    /// The returned view borrows the arena, so it cannot outlive the slab it
    /// addresses: `LiveQTensor<'a>` is not `QTensor`, and the difference is
    /// what the borrow checker enforces here in place of a comment.
    #[cfg(feature = "cuda")]
    pub(super) fn qslot_view<'a>(
        &'a self,
        chunk_idx: usize,
        format: QuantFormat,
        elems: usize,
    ) -> Result<LiveQTensor<'a>> {
        let ggml = format.to_ggml_dtype();
        let payload = (elems / ggml.block_size()) * ggml.type_size();
        let off = self.slot_offset(chunk_idx, payload)?;
        let base = self.base_ptr().ok_or_else(|| {
            candle::Error::Msg("qslot_view: arena is not GPU-resident".to_string())
        })?;
        let candle::Device::Cuda(dev) = self.data.device() else {
            candle::bail!("qslot_view: arena slab is not on a CUDA device");
        };
        // SAFETY: the bounds check in `slot_offset` keeps the view inside slot
        // `chunk_idx`, and zero-on-recycle (invariant 4) means the bytes are
        // always a legal bit pattern for the format. That the slab is still
        // live is no longer an obligation here: `'a` ties the view to this
        // borrow of the arena.
        unsafe { LiveQTensor::from_leased_cuda_ptr(base + off as u64, ggml, elems, dev) }
    }

    /// Quantize `src` into slot `chunk_idx`, `elem_offset` elements in.
    ///
    /// CUDA-only, and deliberately so: the block quantize kernels are, and the
    /// only quantized *writer* chunks are GPU-resident (on CPU
    /// `active_kv_formats` keeps the writer float precisely so partial-token
    /// appends need no block-aligned quantization).
    pub(super) fn quantize_into_slot(
        &mut self,
        chunk_idx: usize,
        format: QuantFormat,
        elems: usize,
        elem_offset: usize,
        src: &Tensor,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let mut view = self.qslot_view(chunk_idx, format, elems)?;
            return view.quantize_into(src, elem_offset);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (chunk_idx, format, elems, elem_offset, src);
            candle::bail!("quantized chunk writes require the cuda feature")
        }
    }

    /// Get the [`ArenaKey`] for this arena.
    pub fn arena_key(&self) -> ArenaKey {
        ArenaKey::new(self.class, self.location)
    }

    /// Zero the chunk at `chunk_idx`, to the **full class stride**.
    ///
    /// The stride, not the payload: the next tenant may be any format whose
    /// bytes fit, so leaving the tail of a recycled slot dirty would let the
    /// persist quantize pass read a prior tenant's bytes past `token_count`
    /// (invariant 4).
    ///
    /// On CUDA this is a kernel launch enqueued on the slab's own stream and
    /// returns once queued; same-stream FIFO ordering then guarantees the next
    /// reader sees zeros without an explicit fence. On CPU it runs inline.
    pub(super) fn zero_chunk_at(&mut self, chunk_idx: usize) -> Result<()> {
        let stride = self.slot_stride();
        let zeros = Tensor::zeros(stride, DType::U8, self.data.device())?;
        self.write_slot_bytes(chunk_idx, &zeros)
    }

    /// GPU bytes this arena occupies. Zero for CPU arenas.
    pub fn gpu_memory_bytes(&self) -> usize {
        if self.location != ArenaLocation::Gpu {
            return 0;
        }
        self.data.elem_count()
    }

    /// Label describing this arena's size class and, on the GPU, which region
    /// of the reservation its bytes are.
    ///
    /// The region index is what ties an arena in a table dump to an address:
    /// arena indices are recycled and say nothing about position, while the
    /// region says exactly where in the span the bytes sit — which is the
    /// question when reading an occupancy dump, since the KV side packs
    /// lowest-first and the high end is what a reclaim reaches for.
    pub fn format_label(&self) -> String {
        #[cfg(feature = "cuda")]
        if let Some(region) = self.region.as_ref().map(RegionHandle::index) {
            return format!("class{} r{region}", self.class.bytes());
        }
        format!("class{}", self.class.bytes())
    }

    /// Raw device pointer and byte stride for one slot. `None` for CPU arenas.
    #[allow(dead_code)]
    pub(super) fn chunk_copy_span(&self, chunk_idx: usize) -> Option<(u64, u32)> {
        let ptr = self.slot_ptr(chunk_idx)?;
        Some((ptr, self.slot_stride() as u32))
    }
}

/// Host-side widening of a slot's raw bytes into a typed tensor.
///
/// The CPU counterpart of the GPU lease: with the slab untyped there is no
/// way to reinterpret its storage in place on the host, so the bytes are
/// decoded explicitly. Only the four dtypes the KV cache stores are accepted —
/// anything else is a caller error, not a runtime condition.
fn decode_bytes(
    bytes: &[u8],
    dtype: DType,
    shape: candle::Shape,
    device: &candle::Device,
) -> Result<Tensor> {
    match dtype {
        DType::F16 => Tensor::from_iter(
            bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]])),
            device,
        )?
        .reshape(shape),
        DType::BF16 => Tensor::from_iter(
            bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]])),
            device,
        )?
        .reshape(shape),
        DType::F32 => Tensor::from_iter(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            device,
        )?
        .reshape(shape),
        DType::F8E4M3 => {
            Tensor::from_iter(bytes.iter().map(|b| float8::F8E4M3::from_bits(*b)), device)?
                .reshape(shape)
        }
        other => candle::bail!("an arena slot cannot be viewed as {other:?}"),
    }
}

/// The inverse of [`decode_bytes`]: a typed tensor's little-endian image.
fn encode_bytes(src: &Tensor) -> Result<Vec<u8>> {
    fn flatten<T: Copy, const N: usize>(v: Vec<T>, le: impl Fn(T) -> [u8; N]) -> Vec<u8> {
        v.into_iter().flat_map(le).collect()
    }
    let out = match src.dtype() {
        DType::F16 => flatten(src.to_vec1::<half::f16>()?, half::f16::to_le_bytes),
        DType::BF16 => flatten(src.to_vec1::<half::bf16>()?, half::bf16::to_le_bytes),
        DType::F32 => flatten(src.to_vec1::<f32>()?, f32::to_le_bytes),
        DType::F8E4M3 => src
            .to_vec1::<float8::F8E4M3>()?
            .into_iter()
            .map(|v| v.to_bits())
            .collect(),
        other => candle::bail!("an arena slot cannot be written from {other:?}"),
    };
    Ok(out)
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
        // all classes (production runs peak around 28). Pre-sizing avoids a
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
    pub(super) fn push_arena(&mut self, arena: Arena, arena_idx: usize) {
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
    /// Human-readable type label, e.g. "class1152 Gpu".
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
    /// `capacity` is the class's slot count. `active`, `free_list`, and `hwm`
    /// are placeholders here — `backing.rs` patches them in from the
    /// [`ChunkGidPool`] (the authoritative per-slot owner) before exposing rows
    /// to callers.
    pub(super) fn arena_rows(&self) -> Vec<ArenaRow> {
        let mut indices: Vec<usize> = self.arenas.keys().copied().collect();
        indices.sort_unstable();
        indices
            .iter()
            .map(|&idx| {
                let arena = &self.arenas[&idx];
                ArenaRow {
                    arena_idx: idx,
                    type_label: format!("{} {:?}", arena.format_label(), arena.location()),
                    is_tombstone: false,
                    capacity: arena.chunks(),
                    hwm: 0,
                    active: 0,
                    free_list: 0,
                    gpu_bytes: arena.gpu_memory_bytes(),
                }
            })
            .collect()
    }
}

// ==================== Arena Storage ====================

/// Storage for KV arenas - supports heterogeneous classes and locations.
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
    /// Default K format for new sealed chunks (immutable, lock-free access).
    default_format: KvFormat,
    /// Default V format for new sealed chunks (immutable, lock-free access).
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

    /// Get the default KvFormat for new sealed chunks (lock-free).
    pub(super) fn k_format(&self) -> KvFormat {
        self.default_format
    }

    /// Get the default V KvFormat for new sealed chunks (lock-free).
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
