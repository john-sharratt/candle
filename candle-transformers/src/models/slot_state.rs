//! Host-side slot state for paged attention kernels (decode and prefill).

use candle::{Result, Tensor};
use candle_nn::kv_cache::{HeadGids, ResolvedArenaInfo, SealedChunk, N_PALETTE};

// ---------------------------------------------------------------------------
// Device pointer extraction
// ---------------------------------------------------------------------------

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
fn build_identity_pal_map(head_dim: usize) -> Vec<u8> {
    let sub_hd = head_dim / N_PALETTE;
    let pal_bytes = head_dim / 4;
    let mut pal = vec![0u8; pal_bytes];
    for d in 0..head_dim {
        let pal_idx = (d / sub_hd).min(N_PALETTE - 1) as u8;
        let byte_idx = d / 4;
        let bit_shift = (d % 4) * 2;
        pal[byte_idx] |= pal_idx << bit_shift;
    }
    pal
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
    /// K palette map: 2 bits per dimension, packed into HEAD_DIM/4 bytes.
    pub k_pal: Vec<u8>,
    /// V palette map: 2 bits per dimension, packed into HEAD_DIM/4 bytes.
    pub v_pal: Vec<u8>,
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
    ) -> Self {
        let _pal_bytes = head_dim / N_PALETTE;
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
                k_fmt[p] = ai.k_format_tag.as_u8();
            }
            if let Some(ai) = arena_info.get(v_arena) {
                v_ptr[p] = ai.base_ptr + v_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                v_fmt[p] = ai.v_format_tag.as_u8();
            }
        }

        // Palette maps: use provided data when non-empty, otherwise identity routing.
        let k_pal = if k_pal_data.is_empty() {
            build_identity_pal_map(head_dim)
        } else {
            k_pal_data.to_vec()
        };
        let v_pal = if v_pal_data.is_empty() {
            build_identity_pal_map(head_dim)
        } else {
            v_pal_data.to_vec()
        };

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
            k_ptr,
            v_ptr,
            k_fmt,
            v_fmt,
            k_scale,
            v_scale,
        }
    }

    /// Serialise this head into `buf` in the exact layout the CUDA kernel expects.
    pub fn serialize_into(&self, buf: &mut Vec<u8>) {
        // Pal maps must be the same length (each is head_dim/4 bytes — bit-packed).
        // If they ever differ, the byte layout shifts for every subsequent field
        // and the kernel reads garbage. Sizes for k_scale / v_scale are encoded
        // in the array types (`[f32; N_PALETTE]`), so they can't drift.
        debug_assert_eq!(
            self.k_pal.len(),
            self.v_pal.len(),
            "k_pal and v_pal must have the same length (head_dim/4 bytes each)"
        );
        buf.extend_from_slice(&self.k_pal);
        buf.extend_from_slice(&self.v_pal);
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
    /// Per-head palette/pointer state.
    pub heads: Vec<KvHeadHost>,
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
        debug_assert!(head_dim >= 4, "head_dim must be >= 4 for 2-bit pal_map packing");
        let pal_bytes = head_dim / 4;
        let pal_total = n_kv_head * pal_bytes;
        let scale_total = n_kv_head * N_PALETTE;
        debug_assert!(
            chunk.k_pal.is_empty() || chunk.k_pal.len() == pal_total,
            "k_pal length must be 0 or {pal_total}, got {}",
            chunk.k_pal.len()
        );
        debug_assert!(
            chunk.v_pal.is_empty() || chunk.v_pal.len() == pal_total,
            "v_pal length must be 0 or {pal_total}, got {}",
            chunk.v_pal.len()
        );
        debug_assert!(
            chunk.k_scale.is_empty() || chunk.k_scale.len() == scale_total,
            "k_scale length must be 0 or {scale_total}, got {}",
            chunk.k_scale.len()
        );
        debug_assert!(
            chunk.v_scale.is_empty() || chunk.v_scale.len() == scale_total,
            "v_scale length must be 0 or {scale_total}, got {}",
            chunk.v_scale.len()
        );
        let heads: Vec<KvHeadHost> = (0..n_kv_head)
            .map(|h| {
                let k_pal_head = if chunk.k_pal.len() >= (h + 1) * pal_bytes {
                    &chunk.k_pal[h * pal_bytes..(h + 1) * pal_bytes]
                } else {
                    &[]
                };
                let v_pal_head = if chunk.v_pal.len() >= (h + 1) * pal_bytes {
                    &chunk.v_pal[h * pal_bytes..(h + 1) * pal_bytes]
                } else {
                    &[]
                };
                let k_scale_head = if chunk.k_scale.len() >= (h + 1) * N_PALETTE {
                    &chunk.k_scale[h * N_PALETTE..(h + 1) * N_PALETTE]
                } else {
                    &[][..]
                };
                let v_scale_head = if chunk.v_scale.len() >= (h + 1) * N_PALETTE {
                    &chunk.v_scale[h * N_PALETTE..(h + 1) * N_PALETTE]
                } else {
                    &[][..]
                };
                KvHeadHost::from_gids(
                    h,
                    head_dim,
                    &chunk.gids,
                    arena_info,
                    k_pal_head,
                    v_pal_head,
                    k_scale_head,
                    v_scale_head,
                )
            })
            .collect();

        Self {
            offset: chunk.offset,
            len: chunk.token_count,
            rope: rope_base,
            heads,
        }
    }

    #[allow(dead_code)]
    pub fn serialized_size(n_kv_head: usize, head_dim: usize) -> usize {
        8 + n_kv_head * Self::kv_head_size(head_dim)
    }

    fn kv_head_size(head_dim: usize) -> usize {
        // pal_map packs 4 dims/byte (2 bits each), so each side is head_dim/4 bytes.
        // The `/4` is the bit-packing density, *not* N_PALETTE — they coincidentally
        // both equal 4 today but are independent constants.
        (head_dim / 4) * 2  // k_pal + v_pal
            + 32 + 32       // k_ptr + v_ptr (4 × u64 each)
            + 4 + 4         // k_fmt + v_fmt (4 × u8 each)
            + 16 + 16       // k_scale + v_scale (4 × f32 each)
    }

    #[allow(dead_code)]
    pub fn serialize_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.len.to_le_bytes());
        buf.extend_from_slice(&self.rope.to_le_bytes());
        for head in &self.heads {
            head.serialize_into(buf);
        }
    }
}

// ---------------------------------------------------------------------------
// SlotStateHost
// ---------------------------------------------------------------------------

/// Host-side mirror of a SlotState (one active sequence).
///
/// Contains the full slice array. The GPU representation is split:
/// - Slice data lives in a contiguous slices tensor
/// - A per-slot `position_map` packs cum_token → (slice_idx, in_blk) for
///   every position in the slot's valid prefix range.  The kernel uses
///   this in place of `chunk_div`/`chunk_mod` so partial-tail slices
///   followed by additional slices read correctly.
/// - A 24B header holds (n_slices, write_slice, slices_ptr,
///   position_map_ptr) where the pointers are resolved device pointers
///   into the slices / position_map tensors.
#[derive(Clone)]
pub struct SlotStateHost {
    /// Which slice the kernel scatters into.
    pub write_slice: u32,
    /// All token slices for this sequence.
    pub slices: Vec<TokenSliceHost>,
    /// Per-cum-token-position lookup: `(slice_idx << 16) | in_blk` for
    /// every k_pos in `[0, total_tokens)`.  Built once per forward pass
    /// from the slice list — replaces the kernel's `chunk_div`/`chunk_mod`
    /// positional math.  `slice_idx` is u16, `in_blk` is u16 (only 0–31
    /// used; CHUNK_SIZE is 32).
    pub position_map: Vec<u32>,
}

/// Pack `(slice_idx, in_blk)` into a `position_map` entry.
#[inline]
pub fn pack_position_entry(slice_idx: u32, in_blk: u32) -> u32 {
    debug_assert!(slice_idx <= 0xFFFF, "slice_idx {} exceeds u16", slice_idx);
    debug_assert!(in_blk <= 0xFFFF, "in_blk {} exceeds u16", in_blk);
    (slice_idx << 16) | (in_blk & 0xFFFF)
}

impl SlotStateHost {
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

        // Under cum_token addressing the writer is the *first chunk
        // at or after the writer boundary that still has capacity*.
        // The boundary is set by the host: `inject_sealed_at_tail`
        // advances it past Arc-shared substrate chunks;
        // `create_view_sequence` sets it to the CoW chunk (the only
        // writer-owned partial); `push_empty_writer_chunk` leaves it
        // alone (the pushed empty is already past it).
        //
        // Within the writer region, prefer the first non-full chunk —
        // this extends partial tails (CoW, decode-extending) and
        // starts fresh empties from in_blk=0.
        let write_slice = if slices.is_empty() {
            0
        } else {
            let start = writer_start_idx.min(slices.len() - 1);
            let mut wi = start;
            for i in start..slices.len() {
                let s = &slices[i];
                if (s.offset as usize + s.len as usize) < 32 {
                    wi = i;
                    break;
                }
                wi = i;
            }
            wi as u32
        };

        // Build the per-cum-token-position lookup table.  For each slice
        // in order, fill positions `[cum, cum + slice.len)` with
        // `(slice_idx, slice.offset + i)`.  Empty slices contribute zero
        // entries — they're invisible to the prefix read scan because no
        // cum_token positions live in them yet.  The total length equals
        // the slot's logical token count (= sum of slice.len).
        let total_tokens: usize = slices.iter().map(|s| s.len as usize).sum();
        let mut position_map: Vec<u32> = Vec::with_capacity(total_tokens);
        for (idx, slice) in slices.iter().enumerate() {
            let slice_off = slice.offset as u32;
            for i in 0..(slice.len as u32) {
                position_map.push(pack_position_entry(idx as u32, slice_off + i));
            }
        }

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

        Self { slices, write_slice, position_map }
    }

    /// Append `seq_len` write-region entries to `position_map`, covering
    /// positions `[total_tokens, total_tokens + seq_len)`.  Each new
    /// position maps into the slot's write area starting at the
    /// write_slice's current `(offset + len)` cursor and advancing
    /// chunk-by-chunk through subsequent slices.
    ///
    /// Caller must have pre-allocated enough slices past `write_slice`
    /// (via `ensure_for_offsets` / `push_empty_writer_chunk`) to cover
    /// `seq_len` chunk overflows.  Asserts on out-of-range — the read
    /// scan in the kernel relies on the map covering every position it
    /// touches.
    pub fn extend_for_write_region(&mut self, seq_len: usize, chunk_size: usize) {
        if seq_len == 0 {
            return;
        }
        debug_assert!(
            !self.slices.is_empty(),
            "extend_for_write_region: slot has no slices — caller must \
             push at least one writer chunk before prefill",
        );
        let mut cur_slice = self.write_slice as usize;
        let mut cur_in_blk = {
            let ws = &self.slices[cur_slice];
            ws.offset as u32 + ws.len as u32
        };
        for _ in 0..seq_len {
            // Advance through chunk boundaries (in_blk overflowed past
            // CHUNK_SIZE) until we find a slice with capacity.  Each
            // subsequent slice starts at its own `offset` field — for
            // freshly-pushed empty chunks this is 0.
            while cur_in_blk as usize >= chunk_size {
                cur_slice += 1;
                debug_assert!(
                    cur_slice < self.slices.len(),
                    "extend_for_write_region: ran out of slices at cur_slice={} \
                     (n_slices={}, seq_len={}, chunk_size={}) — caller \
                     must pre-allocate enough write chunks",
                    cur_slice,
                    self.slices.len(),
                    seq_len,
                    chunk_size,
                );
                cur_in_blk = self.slices[cur_slice].offset as u32;
            }
            self.position_map
                .push(pack_position_entry(cur_slice as u32, cur_in_blk));
            cur_in_blk += 1;
        }
    }
}

