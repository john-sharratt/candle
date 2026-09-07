//! Host-side slot state for paged attention kernels (decode and prefill).

use candle::{Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{span_layout, SpanLayout};
use candle_nn::kv_cache::{
    HeadGids, LiveChunkRef, MetaGid, ResolvedArenaInfo, SealedChunk, CHUNK_SIZE, N_PALETTE,
};

// ---------------------------------------------------------------------------
// Device pointer extraction
// ---------------------------------------------------------------------------

/// The KV span serialized pointers are checked against.
///
/// Aliased rather than named directly so the serialization signatures are the
/// same on both builds: the span layout is a CUDA-side concept, and off CUDA
/// there is no reservation to check against, so the parameter degenerates to a
/// value no caller can supply anything but `None` for.
#[cfg(feature = "cuda")]
pub type PtrCheckSpan = SpanLayout;
/// See the CUDA definition.
#[cfg(not(feature = "cuda"))]
pub type PtrCheckSpan = ();

/// Extract the raw CUDA device pointer from a U8 tensor.
///
/// The returned `u64` is a valid GPU virtual address that can be embedded in
/// host buffers and dereferenced by CUDA kernels.
#[cfg(feature = "cuda")]
pub fn tensor_u8_device_ptr(t: &Tensor) -> Result<u64> {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::DevicePtr;

    let (storage, layout) = t.storage_and_layout();
    let cuda_storage = match &*storage {
        candle::Storage::Cuda(c) => c,
        _ => candle::bail!("tensor_u8_device_ptr: expected CUDA tensor"),
    };
    let stream = cuda_storage.device().cuda_stream();
    let slice = cuda_storage.as_cuda_slice::<u8>()?;
    let slice = slice.slice(layout.start_offset()..);
    let (ptr, _guard) = slice.device_ptr(&stream);
    Ok(ptr)
}

#[cfg(not(feature = "cuda"))]
pub fn tensor_u8_device_ptr(_t: &Tensor) -> Result<u64> {
    candle::bail!("tensor_u8_device_ptr requires the cuda feature")
}

// ---------------------------------------------------------------------------
// Identity palette map
// ---------------------------------------------------------------------------

/// Build the identity 2-bit palette map for the given head dimension.
///
/// Maps dim `d` → palette `d / (head_dim / N_PALETTE)`.  Each byte packs
/// 4 dims in little-endian order: `(d3<<6)|(d2<<4)|(d1<<2)|d0`.
/// Maximum bytes for an inline per-head palette map (`head_dim/4`). Sized for
/// head_dim up to 256 so the palette lives inline in `KvHeadHost` rather than a
/// heap `Vec` — the per-layer prefill builds ~54k of these per forward, so
/// avoiding the allocation is ~80 ms of host CPU.
const PAL_MAP_MAX_BYTES: usize = 64;

/// Fill `pal` (zeroed, `head_dim/4` bytes) with the identity 2-bit palette map.
///
/// Maps dim `d` → palette `d / (head_dim / N_PALETTE)`.  Each byte packs
/// 4 dims in little-endian order: `(d3<<6)|(d2<<4)|(d1<<2)|d0`.
fn fill_identity_pal_map(pal: &mut [u8], head_dim: usize) {
    let sub_hd = head_dim / N_PALETTE;
    for d in 0..head_dim {
        let pal_idx = (d / sub_hd).min(N_PALETTE - 1) as u8;
        let byte_idx = d / 4;
        let bit_shift = (d % 4) * 2;
        pal[byte_idx] |= pal_idx << bit_shift;
    }
}

// ---------------------------------------------------------------------------
// KvHeadHost
// ---------------------------------------------------------------------------

/// Host-side mirror of one KV head's palette/pointer state for a single chunk.
///
/// Layout matches the CUDA `KvHead` struct exactly (168 bytes for HD=128, 8-byte aligned):
/// ```text
///   k_pal[HEAD_DIM/4] — 32B (2-bit K palette indices, packed)
///   v_pal[HEAD_DIM/4] — 32B (2-bit V palette indices, packed)
///   k_ptr[4]          — 32B (pre-resolved K chunk-start pointers)
///   v_ptr[4]          — 32B (pre-resolved V chunk-start pointers)
///   k_fmt[4]          —  4B (K format tag per palette)
///   v_fmt[4]          —  4B (V format tag per palette)
///   k_scale[4]        — 16B (f32 outer scale per K palette, default 1.0)
///   v_scale[4]        — 16B (f32 outer scale per V palette, default 1.0)
/// ```
#[derive(Clone)]
pub struct KvHeadHost {
    /// K palette map: 2 bits per dimension, packed into `head_dim/4` bytes.
    /// Inline (no heap alloc) — only `[..pal_len]` is valid/serialized.
    pub k_pal: [u8; PAL_MAP_MAX_BYTES],
    /// V palette map: 2 bits per dimension, packed into `head_dim/4` bytes.
    pub v_pal: [u8; PAL_MAP_MAX_BYTES],
    /// Valid byte length of `k_pal`/`v_pal` (`head_dim/4`).
    pub pal_len: u16,
    /// Pre-resolved K pointers (one per palette sub-arena), pointing to chunk start.
    pub k_ptr: [u64; N_PALETTE],
    /// Pre-resolved V pointers (one per palette sub-arena), pointing to chunk start.
    pub v_ptr: [u64; N_PALETTE],
    /// K format tag per palette.
    pub k_fmt: [u8; N_PALETTE],
    /// V format tag per palette.
    pub v_fmt: [u8; N_PALETTE],
    /// Outer scale per K palette (f32, default 1.0). Encoder multiplies values
    /// by this before quantizing; decoder divides dequantized values by this to
    /// recover the original magnitude.
    pub k_scale: [f32; N_PALETTE],
    /// Outer scale per V palette (f32, default 1.0). Same convention as k_scale.
    pub v_scale: [f32; N_PALETTE],
}

impl KvHeadHost {
    /// Construct from real arena data for a single head within a chunk.
    ///
    /// Resolves each palette's K and V GIDs to device pointers using the arena
    /// info table.  The palette map GID (if present) is resolved to read the
    /// 2-bit palette indices into `pal`.
    ///
    /// # Arguments
    /// - `head_idx`: which KV head this is for
    /// - `head_dim`: model head dimension
    /// - `gids`: the chunk's `HeadGids` (contains all heads × palettes × K/V)
    /// - `arena_info`: pre-resolved arena base pointers and strides
    /// - `k_pal_data` / `v_pal_data`: per-head packed palette maps (`head_dim/4`
    ///   bytes each). Empty slice → identity routing.
    /// - `k_scale_data` / `v_scale_data`: per-head outer scales (`N_PALETTE`
    ///   f32s each). Empty slice → all-1.0 (no outer scaling).
    /// - `k_fmt_data` / `v_fmt_data`: per-head band format tags (`N_PALETTE`
    ///   bytes each), read from the **chunk**. The arena is consulted only for
    ///   the band's address: under size classes a region holds whatever fits
    ///   its stride and cannot say how to decode a slot
    ///   (`docs/archived/arena_unification.md` principle 8).
    #[allow(clippy::too_many_arguments)]
    pub fn from_gids(
        head_idx: usize,
        head_dim: usize,
        gids: &HeadGids,
        arena_info: &[ResolvedArenaInfo],
        k_pal_data: &[u8],
        v_pal_data: &[u8],
        k_scale_data: &[f32],
        v_scale_data: &[f32],
        k_fmt_data: &[u8],
        v_fmt_data: &[u8],
    ) -> Self {
        let mut k_ptr = [0u64; N_PALETTE];
        let mut v_ptr = [0u64; N_PALETTE];
        let mut k_fmt = [0u8; N_PALETTE];
        let mut v_fmt = [0u8; N_PALETTE];

        for p in 0..N_PALETTE {
            let k_gid = gids.k_gid_pal(head_idx, p);
            let v_gid = gids.v_gid_pal(head_idx, p);

            let k_arena = k_gid.arena_idx();
            let v_arena = v_gid.arena_idx();

            if let Some(ai) = arena_info.get(k_arena) {
                k_ptr[p] = ai.base_ptr + k_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
            }
            if let Some(ai) = arena_info.get(v_arena) {
                v_ptr[p] = ai.base_ptr + v_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
            }
            if let Some(&t) = k_fmt_data.get(p) {
                k_fmt[p] = t;
            }
            if let Some(&t) = v_fmt_data.get(p) {
                v_fmt[p] = t;
            }
        }

        // Palette maps: use provided data when non-empty, otherwise identity
        // routing. Both live INLINE (no heap alloc) — the prefill builds ~54k of
        // these per forward, and the heap churn was ~80 ms of host CPU.
        let pal_bytes = head_dim / 4;
        debug_assert!(
            pal_bytes <= PAL_MAP_MAX_BYTES,
            "head_dim {head_dim} => pal_bytes {pal_bytes} exceeds PAL_MAP_MAX_BYTES {PAL_MAP_MAX_BYTES}"
        );
        let mut k_pal = [0u8; PAL_MAP_MAX_BYTES];
        let mut v_pal = [0u8; PAL_MAP_MAX_BYTES];
        if k_pal_data.is_empty() {
            fill_identity_pal_map(&mut k_pal[..pal_bytes], head_dim);
        } else {
            k_pal[..pal_bytes].copy_from_slice(k_pal_data);
        }
        if v_pal_data.is_empty() {
            fill_identity_pal_map(&mut v_pal[..pal_bytes], head_dim);
        } else {
            v_pal[..pal_bytes].copy_from_slice(v_pal_data);
        }

        // Outer scales: copy from provided slice when long enough, otherwise
        // fall back to identity (1.0). Each side expects exactly N_PALETTE f32s.
        let mut k_scale = [1.0f32; N_PALETTE];
        let mut v_scale = [1.0f32; N_PALETTE];
        if k_scale_data.len() >= N_PALETTE {
            k_scale.copy_from_slice(&k_scale_data[..N_PALETTE]);
        }
        if v_scale_data.len() >= N_PALETTE {
            v_scale.copy_from_slice(&v_scale_data[..N_PALETTE]);
        }

        Self {
            k_pal,
            v_pal,
            pal_len: pal_bytes as u16,
            k_ptr,
            v_ptr,
            k_fmt,
            v_fmt,
            k_scale,
            v_scale,
        }
    }

    /// Serialise this head into `buf` in the exact layout the CUDA kernel expects.
    ///
    /// `layout` is the KV span to check every pointer against, fetched ONCE by
    /// the caller for the whole serialization pass — see [`Self::check_pointers`]
    /// and [`SlotStateHost::span_layout_for_checks`]. `None` disables the check,
    /// which is what a device with no reservation yields.
    pub fn serialize_into(
        &self,
        buf: &mut Vec<u8>,
        #[allow(unused_variables)] layout: Option<&PtrCheckSpan>,
    ) {
        // **Every KV pointer the attention kernels dereference passes through
        // here**, which makes this the one place worth checking them.
        //
        // `k_ptr` / `v_ptr` are per-palette addresses resolved against an arena's
        // placement, and once they are in this buffer nothing looks at them again
        // until a kernel follows them. A wrong one then reports as
        // `CUDA_ERROR_ILLEGAL_ADDRESS` on whichever thread next synchronises,
        // naming a kernel that may be several launches back and may belong to a
        // different thread entirely — three production crashes named three
        // kernels between them and two were bystanders.
        //
        // Checked against the reservation's actual layout, so this catches a
        // pointer into the transient tier (a wave's intermediates and a KV arena
        // given the same bytes), into the weight side (expert slots), or outside
        // the span altogether — the three ways a resolved address can be wrong.
        #[cfg(feature = "cuda")]
        if let Some(l) = layout {
            self.check_pointers(l);
        }
        // k_pal/v_pal share `pal_len` (= head_dim/4 bytes, bit-packed); only the
        // valid prefix is serialized so the byte layout matches the CUDA struct.
        // Sizes for k_scale / v_scale are encoded in the array types
        // (`[f32; N_PALETTE]`), so they can't drift.
        let pal_len = self.pal_len as usize;
        buf.extend_from_slice(&self.k_pal[..pal_len]);
        buf.extend_from_slice(&self.v_pal[..pal_len]);
        for &p in &self.k_ptr {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        for &p in &self.v_ptr {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        buf.extend_from_slice(&self.k_fmt);
        buf.extend_from_slice(&self.v_fmt);
        for &s in &self.k_scale {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        for &s in &self.v_scale {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    /// Every non-null per-palette address must name KV ground. See
    /// [`Self::serialize_into`].
    ///
    /// **Zero is legal and means "this palette is unused"** — a head whose
    /// format needs fewer than `N_PALETTE` sub-bands leaves the tail null, and
    /// the kernel keys on the format rather than the pointer. Only a non-null
    /// address that names memory this engine does not own is a bug.
    ///
    /// The length checked is one chunk's payload for the head's format: that is
    /// what the kernel reads from each pointer, so it is the range that has to
    /// be inside the arena rather than merely starting there.
    #[cfg(feature = "cuda")]
    fn check_pointers(&self, layout: &SpanLayout) {
        use candle_nn::kv_cache::expect_kv_range_in;

        for (side, ptrs) in [("k_ptr", &self.k_ptr), ("v_ptr", &self.v_ptr)] {
            for &addr in ptrs.iter() {
                if addr == 0 {
                    continue;
                }
                // `side` alone, not `format!("{side}[{p_idx}]")`: this runs
                // `N_PALETTE` times per side per head per slice, and formatting
                // a name for every pointer costs more than the check it labels.
                // The panic path prints the address, which is what identifies
                // the offender.
                expect_kv_range_in(layout, addr, 1, side, "KvHead::serialize_into");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TokenSliceHost
// ---------------------------------------------------------------------------

/// Host-side mirror of a TokenSlice (one sequence's view into a chunk).
///
/// Layout matches the CUDA `TokenSlice` struct:
/// ```text
///   offset: u16     — 2B
///   len:    u16     — 2B (mutable on GPU; reconciled on host writes)
///   rope:   u32     — 4B
///   head[N_KV_HEADS] — N_KV_HEADS × 168B (HD=128)
/// ```
#[derive(Clone)]
pub struct TokenSliceHost {
    /// First valid token position within the chunk.
    pub offset: u16,
    /// Number of valid tokens. Shadow-tracked on host; GPU self-increments.
    pub len: u16,
    /// Absolute RoPE position of the first token in this slice.
    pub rope: u32,
    /// Per-head palette/pointer state. **Empty when `meta` is `Some`** — a
    /// resident record already holds these bytes, so they are not rebuilt here.
    pub heads: Vec<KvHeadHost>,
    /// Resident KV-head record handle. `Some` ⇒ the kernel reads the chunk's
    /// heads from the device meta-pool slab at `device_addr(meta)`, and the host
    /// skips serializing a scratch record for it. `None` ⇒ scratch record built
    /// from `heads`.
    pub meta: Option<MetaGid>,
}

impl TokenSliceHost {
    /// Construct from a [`SealedChunk`] with resolved arena pointers.
    ///
    /// Each head's GIDs are resolved to device pointers via `arena_info`.
    ///
    /// `rope_base` is the absolute RoPE position of this chunk's
    /// first valid token *within the destination slot's layout* — the
    /// caller is responsible for computing it as the cumulative usage
    /// of all preceding chunks in the slot.  `SealedChunk` itself
    /// carries no positional state (see its doc comment); RoPE is
    /// applied at the latest responsible moment by the attention
    /// kernel using this `rope_base`.
    pub fn from_sealed_chunk(
        chunk: &SealedChunk,
        rope_base: u32,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> Self {
        Self::from_parts(
            chunk.offset,
            chunk.token_count,
            chunk.meta.as_ref(),
            &chunk.gids,
            &chunk.k_pal,
            &chunk.v_pal,
            &chunk.k_scale,
            &chunk.v_scale,
            &chunk.k_fmt,
            &chunk.v_fmt,
            rope_base,
            n_kv_head,
            head_dim,
            arena_info,
        )
    }

    /// Zero-clone entry: build from a borrowed live-chunk view (see
    /// `ChunkedKvBacking::visit_live_chunks`) — identical output to
    /// [`Self::from_sealed_chunk`] without materializing a `SealedChunk`.
    pub fn from_live_chunk(
        c: &LiveChunkRef<'_>,
        rope_base: u32,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> Self {
        Self::from_parts(
            c.offset,
            c.token_count,
            c.meta,
            c.gids,
            c.k_pal,
            c.v_pal,
            c.k_scale,
            c.v_scale,
            c.k_fmt,
            c.v_fmt,
            rope_base,
            n_kv_head,
            head_dim,
            arena_info,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        offset: u16,
        token_count: u16,
        meta: Option<&MetaGid>,
        gids: &HeadGids,
        k_pal: &[u8],
        v_pal: &[u8],
        k_scale: &[f32],
        v_scale: &[f32],
        k_fmt: &[u8],
        v_fmt: &[u8],
        rope_base: u32,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> Self {
        debug_assert!(
            head_dim >= 4,
            "head_dim must be >= 4 for 2-bit pal_map packing"
        );
        let pal_bytes = head_dim / 4;
        let pal_total = n_kv_head * pal_bytes;
        let scale_total = n_kv_head * N_PALETTE;
        debug_assert!(
            k_pal.is_empty() || k_pal.len() == pal_total,
            "k_pal length must be 0 or {pal_total}, got {}",
            k_pal.len()
        );
        debug_assert!(
            v_pal.is_empty() || v_pal.len() == pal_total,
            "v_pal length must be 0 or {pal_total}, got {}",
            v_pal.len()
        );
        debug_assert!(
            k_scale.is_empty() || k_scale.len() == scale_total,
            "k_scale length must be 0 or {scale_total}, got {}",
            k_scale.len()
        );
        debug_assert!(
            v_scale.is_empty() || v_scale.len() == scale_total,
            "v_scale length must be 0 or {scale_total}, got {}",
            v_scale.len()
        );
        // A chunk with a resident record (`meta.is_some()`) already has its
        // KvHead[n_kv_head] bytes in a device meta-pool slab — built once at
        // quantize and kept in sync across migrations. Skip rebuilding them here
        // (the dominant per-layer-per-forward cost); the slice will carry the
        // record's device address as `kvheads_ptr`. Only transient/float chunks
        // (`meta.is_none()`) build a scratch record from `heads`.
        let heads: Vec<KvHeadHost> = if meta.is_some() {
            Vec::new()
        } else {
            (0..n_kv_head)
                .map(|h| {
                    let k_pal_head = if k_pal.len() >= (h + 1) * pal_bytes {
                        &k_pal[h * pal_bytes..(h + 1) * pal_bytes]
                    } else {
                        &[]
                    };
                    let v_pal_head = if v_pal.len() >= (h + 1) * pal_bytes {
                        &v_pal[h * pal_bytes..(h + 1) * pal_bytes]
                    } else {
                        &[]
                    };
                    let k_scale_head = if k_scale.len() >= (h + 1) * N_PALETTE {
                        &k_scale[h * N_PALETTE..(h + 1) * N_PALETTE]
                    } else {
                        &[][..]
                    };
                    let v_scale_head = if v_scale.len() >= (h + 1) * N_PALETTE {
                        &v_scale[h * N_PALETTE..(h + 1) * N_PALETTE]
                    } else {
                        &[][..]
                    };
                    let k_fmt_head = if k_fmt.len() >= (h + 1) * N_PALETTE {
                        &k_fmt[h * N_PALETTE..(h + 1) * N_PALETTE]
                    } else {
                        &[][..]
                    };
                    let v_fmt_head = if v_fmt.len() >= (h + 1) * N_PALETTE {
                        &v_fmt[h * N_PALETTE..(h + 1) * N_PALETTE]
                    } else {
                        &[][..]
                    };
                    KvHeadHost::from_gids(
                        h,
                        head_dim,
                        gids,
                        arena_info,
                        k_pal_head,
                        v_pal_head,
                        k_scale_head,
                        v_scale_head,
                        k_fmt_head,
                        v_fmt_head,
                    )
                })
                .collect()
        };

        Self {
            offset,
            len: token_count,
            rope: rope_base,
            heads,
            meta: meta.cloned(),
        }
    }

    /// Fixed bytes of the 16-byte slice header (offset/len/rope + kvheads_ptr).
    pub const SLICE_HEADER_SIZE: usize = 16;

    /// Bytes of this slice's out-of-line `KvHead[n_kv_head]` record.
    pub fn record_size(n_kv_head: usize, head_dim: usize) -> usize {
        n_kv_head * Self::kv_head_size(head_dim)
    }

    fn kv_head_size(head_dim: usize) -> usize {
        // pal_map packs 4 dims/byte (2 bits each), so each side is head_dim/4 bytes.
        // The `/4` is the bit-packing density, *not* N_PALETTE — they coincidentally
        // both equal 4 today but are independent constants.
        (head_dim / 4) * 2  // k_pal + v_pal
            + 32 + 32       // k_ptr + v_ptr (4 × u64 each)
            + 4 + 4         // k_fmt + v_fmt (4 × u8 each)
            + 16 + 16 // k_scale + v_scale (4 × f32 each)
    }

    /// Serialize this slice's out-of-line `KvHead[n_kv_head]` record (the bytes
    /// `kvheads_ptr` points at). Length is [`record_size`].
    /// `layout` is fetched once per serialization pass by the caller — see
    /// [`SlotStateHost::span_layout_for_checks`].
    pub fn serialize_record(&self, buf: &mut Vec<u8>, layout: Option<&PtrCheckSpan>) {
        for head in &self.heads {
            head.serialize_into(buf, layout);
        }
    }

    /// Serialize the 16-byte slice header with the resolved device address of
    /// this slice's record.
    pub fn serialize_slice_header(&self, buf: &mut Vec<u8>, kvheads_ptr: u64) {
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.len.to_le_bytes());
        buf.extend_from_slice(&self.rope.to_le_bytes());
        buf.extend_from_slice(&kvheads_ptr.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// SlotStateHost
// ---------------------------------------------------------------------------

/// Host-side mirror of a SlotState (one active sequence).
///
/// Contains the full slice array. The GPU representation is split:
/// - Slice data lives in a contiguous slices tensor
/// - A 16B header holds (n_slices, write_slice, slices_ptr), where the
///   pointer is a resolved device pointer into the slices tensor.
///
/// There is no position lookup table. A cum_token position resolves to
/// `(slice_idx, in_blk)` by *computing* it from the slice array — see
/// [`resolve_pos_reference`] and the kernel's `resolve_pos`. Everything such a
/// table would encode is already implied by each slice's `rope`, `len` and
/// `offset`, so materialising one only created a second copy of the layout that
/// could disagree with the first: the map was built per forward on the host,
/// uploaded over PCIe, and — because it was built for one layer and reused for
/// the rest — went stale the moment a windowed prefill left a layer's chunk
/// list one entry ahead of its neighbours. Computing the answer cannot go
/// stale, because there is only ever one description of the layout.
#[derive(Clone)]
pub struct SlotStateHost {
    /// Which slice the kernel scatters into.
    pub write_slice: u32,
    /// All token slices for this sequence.
    pub slices: Vec<TokenSliceHost>,
}

/// Resolve a cum_token position to `(slice_idx, in_blk)` from slice state alone
/// — the host mirror of the kernel's `resolve_pos`
/// (`candle-kernels/src/paged-decode/slot_types.cuh`).
///
/// **This is the executable specification of that kernel function**, and the two
/// must stay identical. It exists because the kernel's version cannot be unit
/// tested directly, and the half of it that resolves *pending writes* is not
/// derivable from a slice's `len` — a first attempt at the kernel searched only
/// committed tokens and would have sent every write to slice 0.
///
/// Committed positions are found by binary search (`rope <= k < rope + len`);
/// positions past the committed total continue from `write_slice`, filling the
/// writer's chunk from `offset + len` to `chunk_size` and each later chunk from
/// its own `offset`. `None` means the position is past everything the slot can
/// address, which the kernel reports as `(0, 0)` — the case
/// [`SlotStateHost::assert_write_region_capacity`] exists to make unreachable.
pub fn resolve_pos_reference(
    slices: &[TokenSliceHost],
    write_slice: usize,
    chunk_size: usize,
    k_pos: usize,
) -> Option<(usize, usize)> {
    if slices.is_empty() {
        return None;
    }
    let (mut lo, mut hi) = (0isize, slices.len() as isize - 1);
    while lo <= hi {
        let mid = ((lo + hi) / 2) as usize;
        let s = &slices[mid];
        let start = s.rope as usize;
        let len = s.len as usize;
        if k_pos < start {
            hi = mid as isize - 1;
        } else if k_pos >= start + len {
            lo = mid as isize + 1;
        } else {
            return Some((mid, s.offset as usize + (k_pos - start)));
        }
    }
    let last = slices.last()?;
    let committed = last.rope as usize + last.len as usize;
    let mut cur = write_slice;
    if cur >= slices.len() {
        return None;
    }
    let mut cur_in_blk = slices[cur].offset as usize + slices[cur].len as usize;
    let mut j = k_pos.checked_sub(committed)?;
    loop {
        if cur_in_blk < chunk_size {
            let cap = chunk_size - cur_in_blk;
            if j < cap {
                return Some((cur, cur_in_blk + j));
            }
            j -= cap;
        }
        cur += 1;
        if cur >= slices.len() {
            return None;
        }
        cur_in_blk = slices[cur].offset as usize;
    }
}

/// [`resolve_pos_reference`], trying the slice `hint` first — the host mirror
/// of the kernel's `resolve_pos_hinted`.
///
/// Committed ranges `[rope, rope + len)` never overlap, so a position inside
/// the hint's range has exactly one owner, and it is the slice the search
/// would have returned. A hint that misses, or that is past the array (the
/// kernel's `hint < 0`), changes only the cost: the answer is always
/// `resolve_pos_reference`'s.
pub fn resolve_pos_hinted_reference(
    slices: &[TokenSliceHost],
    write_slice: usize,
    chunk_size: usize,
    hint: usize,
    k_pos: usize,
) -> Option<(usize, usize)> {
    if let Some(s) = slices.get(hint) {
        let start = s.rope as usize;
        if k_pos >= start && k_pos < start + s.len as usize {
            return Some((hint, s.offset as usize + (k_pos - start)));
        }
    }
    resolve_pos_reference(slices, write_slice, chunk_size, k_pos)
}

/// One slot's chunk list reduced to the token layout: `(offset, len)` per
/// chunk, in slice order.
///
/// Everything positional the kernel needs — which slice a `k_pos` resolves
/// to, where the writer scatters, how much the write region holds — is a
/// function of this and nothing else; the arena pointers, palettes and scales
/// that make up the rest of a slice are not. It is read per layer: layers can
/// legitimately differ here (a windowed creep prefill leaves layer 0 an empty
/// writer chunk ahead of the rest), so no layer's layout stands in for
/// another's.
pub struct SlotTokenLayout {
    /// `(offset, len)` per chunk, in slice order.
    pub chunks: Vec<(u16, u16)>,
    /// Which slice the kernel scatters into.
    pub write_slice: u32,
}

impl SlotTokenLayout {
    /// The writer is the *first chunk at or after the writer boundary that
    /// still has capacity*. The boundary is set by the host:
    /// `inject_sealed_at_tail` advances it past Arc-shared substrate chunks;
    /// `create_view_sequence` sets it to the CoW chunk (the only writer-owned
    /// partial); `push_empty_writer_chunk` leaves it alone (the pushed empty is
    /// already past it).
    ///
    /// Within the writer region, prefer the first non-full chunk — this extends
    /// partial tails (CoW, decode-extending) and starts fresh empties from
    /// `in_blk = 0`.
    pub fn new(chunks: Vec<(u16, u16)>, writer_start_idx: usize, chunk_size: usize) -> Self {
        let write_slice = if writer_start_idx >= chunks.len() {
            // The writer region has no chunks (freshly injected prefix whose
            // sealed partial tail is a gap). There is NO valid write target:
            // point write_slice at the end so an actual write attempt fails
            // loudly in `assert_write_region_capacity` instead of silently landing
            // in an Arc-shared sealed chunk. Writers (prefill with new tokens)
            // allocate writer chunks first via `ensure_for_batch_entries`,
            // which brings the boundary back inside the slice list.
            chunks.len() as u32
        } else {
            let mut wi = writer_start_idx;
            for (i, &(offset, len)) in chunks.iter().enumerate().skip(writer_start_idx) {
                wi = i;
                if (offset as usize + len as usize) < chunk_size {
                    break;
                }
            }
            wi as u32
        };
        Self {
            chunks,
            write_slice,
        }
    }

    /// The slot's logical token count — the sum of its chunks' lengths.
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(|&(_, len)| len as usize).sum()
    }

    /// Assert the slot can address `seq_len` pending writes past its committed
    /// tokens — i.e. that the caller allocated the write region.
    ///
    /// The kernel resolves a pending write by walking from `write_slice`,
    /// filling that chunk from `offset + len` to `chunk_size` and each later
    /// chunk from its own `offset`. If it walks off the end of the slice array
    /// it has nowhere to put the token and reports `(0, 0)` — which is a *valid
    /// address*, so the write lands silently in slice 0 and corrupts the
    /// sequence's first chunk. Nothing downstream can tell that apart from a
    /// real position, so the condition has to be caught here, on the host,
    /// where the allocation that should have happened is still nameable.
    ///
    /// Callers pre-allocate via `ensure_for_offsets` / `push_empty_writer_chunk`
    /// (prefill goes through `ensure_for_batch_entries`). This is one pass over
    /// the chunks rather than one per token: capacity is `chunk_size` minus the
    /// writer's cursor, plus `chunk_size - offset` for every chunk after it.
    pub fn assert_write_region_capacity(&self, seq_len: usize, chunk_size: usize) {
        if seq_len == 0 {
            return;
        }
        let writer = self.write_slice as usize;
        assert!(
            writer < self.chunks.len(),
            "assert_write_region_capacity: no writer chunk (write_slice={} of {} \
             slices, seq_len={seq_len}) — the write region was not allocated \
             before prefill (ensure_for_batch_entries)",
            self.write_slice,
            self.chunks.len(),
        );
        let (w_offset, w_len) = self.chunks[writer];
        let cursor = w_offset as usize + w_len as usize;
        let capacity: usize = chunk_size.saturating_sub(cursor)
            + self.chunks[writer + 1..]
                .iter()
                .map(|&(offset, _)| chunk_size.saturating_sub(offset as usize))
                .sum::<usize>();
        if capacity < seq_len {
            // Dump the full slot layout so the shortfall is diagnosable in
            // release, where a bare index panic would hide which chunk ran out.
            let layout: String = self
                .chunks
                .iter()
                .enumerate()
                .map(|(i, &(off, len))| format!("[{i}] off={off} len={len}"))
                .collect::<Vec<_>>()
                .join("  ");
            panic!(
                "write region too small: capacity={capacity} < seq_len={seq_len} \
                 (n_slices={}, write_slice={}, chunk_size={chunk_size}). Slot layout: {layout}",
                self.chunks.len(),
                self.write_slice,
            );
        }
    }
}

impl SlotStateHost {
    /// The KV span every serialized pointer is checked against, fetched ONCE
    /// for a whole serialization pass.
    ///
    /// [`span_layout`](candle_nn::kv_cache::span_layout) takes the region
    /// pool's global lock, so it belongs outside the loop, not inside the
    /// check: a slot's record carries `N_PALETTE` K and V addresses per head,
    /// and fetching per pointer made the metadata pack 98% lock traffic
    /// (4,608 acquisitions per attention layer per speculative step). The
    /// layout cannot change during a pass — the pool that owns it is not
    /// reachable from serialization.
    ///
    /// `None` on a device with no reservation, which disables the check
    /// exactly as it always did.
    #[cfg(feature = "cuda")]
    pub fn span_layout_for_checks() -> Option<SpanLayout> {
        // Ordinal 0: this engine runs one device, and `span_layout` answers
        // `None` for any device with no reservation, so a wrong guess here
        // disables the check rather than misfiring.
        span_layout(0)
    }

    /// Construct from a sequence of sealed chunks with resolved arena pointers.
    ///
    /// Each chunk's `rope_base` is computed as the cumulative `token_count`
    /// of all preceding chunks — the absolute RoPE position of the
    /// chunk's first valid token *within this slot's layout*.  RoPE
    /// is therefore a function of the destination slot, not of the
    /// `SealedChunk`'s origin: the same sealed bytes injected at any
    /// position yield the right kernel-visible rope value.  Sets
    /// `write_slice` to the last chunk index.
    pub fn from_sealed_chunks(
        chunks: &[SealedChunk],
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
        writer_start_idx: usize,
    ) -> Self {
        let mut cum_tokens: u32 = 0;
        let slices: Vec<TokenSliceHost> = chunks
            .iter()
            .map(|c| {
                let rope_base = cum_tokens;
                cum_tokens = cum_tokens.saturating_add(c.token_count as u32);
                TokenSliceHost::from_sealed_chunk(c, rope_base, n_kv_head, head_dim, arena_info)
            })
            .collect();
        Self::from_slices(slices, writer_start_idx)
    }

    /// Assemble a slot from already-built slices: the slice list plus writer
    /// selection. The slice list comes from either
    /// [`TokenSliceHost::from_sealed_chunk`] (owned snapshots) or
    /// [`TokenSliceHost::from_live_chunk`] (the zero-clone visitor path).
    pub fn from_slices(slices: Vec<TokenSliceHost>, writer_start_idx: usize) -> Self {
        // Writer selection is a function of the token layout alone, and
        // `SlotTokenLayout` holds the rule — the prefill header build applies
        // it to the same `(offset, len)` pairs without materialising slices.
        let write_slice = SlotTokenLayout::new(
            slices.iter().map(|s| (s.offset, s.len)).collect(),
            writer_start_idx,
            CHUNK_SIZE,
        )
        .write_slice;

        // Per-slot trace of the kernel-visible slice layout.  Enable
        // with `RUST_LOG=candle_transformers::models::slot_state=trace`.
        // Each line is one slot's full slice list — (rope_base,
        // token_count, offset) per chunk — i.e. the exact values the
        // attention kernel will read for `slice_rope(...)` /
        // `slice_len(...)` / `slice_offset(...)` on the next forward
        // pass.  Critical for diagnosing position-related bugs after
        // section injection or fork.
        if tracing::enabled!(
            target: "candle_transformers::models::slot_state",
            tracing::Level::TRACE,
        ) {
            let n = slices.len();
            let total_tokens: u32 = slices.iter().map(|s| s.len as u32).sum();
            let summary: String = slices
                .iter()
                .map(|s| format!("(rope={},len={},off={})", s.rope, s.len, s.offset))
                .collect::<Vec<_>>()
                .join(",");
            tracing::trace!(
                target: "candle_transformers::models::slot_state",
                n_slices = n,
                total_tokens,
                write_slice = write_slice,
                slices = %summary,
                "built slot state for kernel",
            );
        }

        Self {
            slices,
            write_slice,
        }
    }

    /// Assert the slot can address `seq_len` pending writes past its committed
    /// tokens — i.e. that the caller allocated the write region. The rule is
    /// [`SlotTokenLayout::assert_write_region_capacity`], applied to this slot's
    /// slice array.
    pub fn assert_write_region_capacity(&self, seq_len: usize, chunk_size: usize) {
        SlotTokenLayout {
            chunks: self.slices.iter().map(|s| (s.offset, s.len)).collect(),
            write_slice: self.write_slice,
        }
        .assert_write_region_capacity(seq_len, chunk_size);
    }
}

#[cfg(test)]
mod resolve_pos_tests {
    use super::{
        resolve_pos_hinted_reference, resolve_pos_reference, SlotStateHost, TokenSliceHost,
    };

    const CHUNK: usize = 32;

    /// The position map the host used to build and upload every forward, kept
    /// here as the **oracle** these tests check against.
    ///
    /// It states the layout the other way round from [`resolve_pos_reference`]:
    /// walking the slices forward and emitting every position in order, rather
    /// than searching for one position. Keeping it as test code is the point: it
    /// is the
    /// independent statement of what a position *should* resolve to, so
    /// `resolve_pos_reference` (and through it the kernel) is checked against a
    /// second implementation rather than against itself. Deleting it with the
    /// production copy would have left the replacement unverifiable.
    fn oracle_position_map(
        slices: &[TokenSliceHost],
        write_slice: usize,
        write_len: usize,
        chunk_size: usize,
    ) -> Vec<u32> {
        let pack = |slice_idx: usize, in_blk: usize| ((slice_idx as u32) << 16) | (in_blk as u32);
        let mut pm = Vec::new();
        // Committed: every slice contributes `len` consecutive positions.
        for (idx, s) in slices.iter().enumerate() {
            for i in 0..s.len as usize {
                pm.push(pack(idx, s.offset as usize + i));
            }
        }
        // Pending writes: continue from the writer, filling each chunk to
        // `chunk_size` and resuming at the next slice's own offset.
        if write_len > 0 {
            let mut cur = write_slice;
            let mut cur_in_blk = slices[cur].offset as usize + slices[cur].len as usize;
            for _ in 0..write_len {
                while cur_in_blk >= chunk_size {
                    cur += 1;
                    cur_in_blk = slices[cur].offset as usize;
                }
                pm.push(pack(cur, cur_in_blk));
                cur_in_blk += 1;
            }
        }
        pm
    }

    /// Slices with cumulative `rope` bases, from `(offset, len)` pairs.
    fn slices(spec: &[(u16, u16)]) -> Vec<TokenSliceHost> {
        let mut cum = 0u32;
        spec.iter()
            .map(|&(offset, len)| {
                let s = TokenSliceHost {
                    offset,
                    len,
                    rope: cum,
                    heads: Vec::new(),
                    meta: None,
                };
                cum += len as u32;
                s
            })
            .collect()
    }

    /// **The equivalence the kernel depends on.** Builds the position map the
    /// host used to upload, and asserts the computed resolution agrees at
    /// *every* position. The map is the oracle precisely because it no longer
    /// exists in production: this is what makes its removal a refactor rather
    /// than a rewrite.
    fn assert_agrees(spec: &[(u16, u16)], writer_start: usize, write_len: usize) {
        // `from_slices` still derives `write_slice`, which the kernel reads from
        // the slot header and `resolve_pos` starts its write-region walk from.
        let st = SlotStateHost::from_slices(slices(spec), writer_start);
        // The oracle indexes the write region unconditionally, so a spec that
        // cannot hold `write_len` would panic there rather than prove anything.
        st.assert_write_region_capacity(write_len, CHUNK);
        let expected = oracle_position_map(&st.slices, st.write_slice as usize, write_len, CHUNK);
        let committed: usize = st.slices.iter().map(|s| s.len as usize).sum();
        assert_eq!(
            expected.len(),
            committed + write_len,
            "the oracle should cover the committed tokens and the write region",
        );
        for (k, &entry) in expected.iter().enumerate() {
            let want = ((entry >> 16) as usize, (entry & 0xFFFF) as usize);
            let got = resolve_pos_reference(&st.slices, st.write_slice as usize, CHUNK, k);
            assert_eq!(
                got,
                Some(want),
                "position {k} of {} (committed {committed}) disagrees: spec={spec:?} \
                 writer_start={writer_start} write_len={write_len}",
                expected.len(),
            );
        }
    }

    #[test]
    fn committed_only_full_chunks() {
        assert_agrees(&[(0, 32), (0, 32), (0, 32)], 3, 0);
    }

    /// A partial tail — the ordinary shape, since a turn rarely ends on a
    /// chunk boundary.
    #[test]
    fn committed_with_a_partial_tail() {
        assert_agrees(&[(0, 32), (0, 32), (0, 7)], 2, 0);
    }

    /// Injected substrate windows do not start at 0 — `offset` is where the
    /// valid tokens begin inside the physical chunk.
    #[test]
    fn committed_windows_with_nonzero_offsets() {
        assert_agrees(&[(5, 27), (0, 32), (11, 21)], 3, 0);
    }

    /// **The write region inside the writer's own chunk** — the common decode
    /// step, extending a partial tail.
    #[test]
    fn writes_extend_the_writers_partial_chunk() {
        assert_agrees(&[(0, 32), (0, 10)], 1, 8);
    }

    /// **The write region overflowing into following empty chunks** — the case
    /// a search over `len` alone cannot resolve, because those chunks hold
    /// nothing yet. This is the half of `resolve_pos` I first got wrong.
    #[test]
    fn writes_overflow_into_empty_chunks() {
        assert_agrees(&[(0, 32), (0, 30), (0, 0), (0, 0)], 1, 40);
    }

    /// An empty writer chunk pushed ahead of any write — the trailing-empty
    /// structure that layers legitimately disagree about.
    #[test]
    fn writes_start_at_a_freshly_pushed_empty_chunk() {
        assert_agrees(&[(0, 32), (0, 32), (0, 0)], 2, 20);
    }

    /// A writer chunk that is exactly full: the walk must step past it before
    /// emitting anything.
    #[test]
    fn a_full_writer_chunk_is_stepped_over() {
        assert_agrees(&[(0, 32), (0, 0)], 0, 12);
    }

    /// Offsets on the overflow chunks too — each continues from its own
    /// `offset`, not from zero.
    #[test]
    fn overflow_chunks_resume_at_their_own_offset() {
        assert_agrees(&[(0, 20), (4, 0), (9, 0)], 0, 30);
    }

    /// A position past everything the slot can address resolves to nothing —
    /// the kernel reports `(0, 0)` there rather than reading out of bounds.
    #[test]
    fn a_position_past_the_slot_resolves_to_nothing() {
        let sl = slices(&[(0, 32), (0, 4)]);
        assert_eq!(resolve_pos_reference(&sl, 1, CHUNK, 200), None);
        assert_eq!(resolve_pos_reference(&[], 0, CHUNK, 0), None);
    }

    /// How many pending writes the slot can actually address, found by walking
    /// exactly as the kernel does — the definition
    /// `assert_write_region_capacity` has to agree with.
    fn walked_capacity(st: &SlotStateHost) -> usize {
        let mut cur = st.write_slice as usize;
        if cur >= st.slices.len() {
            return 0;
        }
        let mut in_blk = st.slices[cur].offset as usize + st.slices[cur].len as usize;
        let mut n = 0;
        loop {
            while in_blk >= CHUNK {
                cur += 1;
                if cur >= st.slices.len() {
                    return n;
                }
                in_blk = st.slices[cur].offset as usize;
            }
            n += 1;
            in_blk += 1;
        }
    }

    /// The closed-form capacity must equal the walk, for writers that are
    /// partial, exactly full, offset, and followed by offset overflow chunks.
    #[test]
    fn capacity_matches_the_walk_it_replaced() {
        for (spec, writer) in [
            (&[(0u16, 32u16), (0, 0)][..], 1usize),
            (&[(0, 32), (0, 0)][..], 0),
            (&[(0, 20), (4, 0), (9, 0)][..], 0),
            (&[(8, 10), (0, 0), (0, 0)][..], 0),
            (&[(0, 32), (0, 32), (0, 0)][..], 2),
            (&[(0, 4)][..], 0),
        ] {
            let st = SlotStateHost::from_slices(slices(spec), writer);
            let want = walked_capacity(&st);
            st.assert_write_region_capacity(want, CHUNK);
            assert!(
                std::panic::catch_unwind(|| st.assert_write_region_capacity(want + 1, CHUNK))
                    .is_err(),
                "capacity {want} should be the maximum accepted for {spec:?} writer={writer}",
            );
        }
    }

    /// A slot whose writer boundary sits past the end has no write target at
    /// all: the kernel would resolve every pending write to `(0, 0)` and
    /// scribble over slice 0, so the host refuses it first.
    #[test]
    fn a_slot_with_no_writer_chunk_refuses_any_write() {
        let st = SlotStateHost::from_slices(slices(&[(0, 32)]), 1);
        assert_eq!(st.write_slice as usize, st.slices.len());
        st.assert_write_region_capacity(0, CHUNK); // no write: nothing to check
        assert!(
            std::panic::catch_unwind(|| st.assert_write_region_capacity(1, CHUNK)).is_err(),
            "a slot with no writer chunk must refuse a write",
        );
    }

    /// **A hint changes the cost, never the answer.** Every position of each
    /// layout — committed, pending, and past the end — resolved with every
    /// slice as the hint, and with a hint past the array, agrees with the
    /// unhinted search. The kernel's QSA column loop and glue column stream
    /// pass the slice they already hold, and rely on exactly this.
    #[test]
    fn a_hint_never_changes_the_resolved_position() {
        for (spec, writer_start, write_len) in [
            (&[(0u16, 32u16), (0, 32), (0, 7)][..], 2usize, 0usize),
            (&[(5, 27), (0, 32), (11, 21)][..], 3, 0),
            (&[(0, 32), (0, 10)][..], 1, 8),
            (&[(0, 32), (0, 30), (0, 0), (0, 0)][..], 1, 40),
            (&[(0, 32), (0, 32), (0, 0)][..], 2, 20),
            (&[(0, 20), (4, 0), (9, 0)][..], 0, 30),
        ] {
            let st = SlotStateHost::from_slices(slices(spec), writer_start);
            let write_slice = st.write_slice as usize;
            let committed: usize = st.slices.iter().map(|s| s.len as usize).sum();
            for k in 0..committed + write_len + 2 {
                let want = resolve_pos_reference(&st.slices, write_slice, CHUNK, k);
                for hint in 0..=st.slices.len() {
                    assert_eq!(
                        resolve_pos_hinted_reference(&st.slices, write_slice, CHUNK, hint, k),
                        want,
                        "position {k} with hint {hint}: spec={spec:?} \
                         writer_start={writer_start}",
                    );
                }
            }
        }
    }
}
