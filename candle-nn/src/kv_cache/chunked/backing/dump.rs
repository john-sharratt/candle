//! Decodes a backing's stored K/V into host floats, for offline dumps.
//!
//! A dump reads each band's own format tag, because the arena is an untyped byte
//! slab and the tag is the only record of how to read it. Quantized bands are
//! refused by the float path, which would need dequantization. Every palette
//! sub-band is dumped, since dumping one would quietly corrupt the analysis. The
//! R16 probe path gathers K, Q and V with one kernel launch on CUDA.

use super::*;

/// Decode one band's slot into `f32`, for the offline dump paths.
///
/// `None` unless the band's tag names a **float** format: these dumps
/// reconstruct raw activations, and a quantized band would need the
/// dequantize path rather than a widening cast. The tag is the only record of
/// how to read the slot — the arena is a run of untyped bytes.
///
/// The decode is done on the host from the slot's raw bytes rather than by
/// viewing the slab as a typed tensor, so it works identically for a hot GPU
/// arena and a warm CPU one, and needs no device-side reinterpretation.
pub(super) fn read_float_band(
    arena: &Arena,
    chunk_idx: usize,
    tag: ArenaFormatTag,
    elems: usize,
) -> Option<Vec<f32>> {
    let dtype = tag.to_dtype()?;
    let bytes = arena
        .slot_bytes(chunk_idx, elems * dtype.size_in_bytes())
        .ok()?
        .to_vec1::<u8>()
        .ok()?;
    let out = match dtype {
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F8E4M3 => bytes
            .iter()
            .map(|b| float8::F8E4M3::from_bits(*b).to_f32())
            .collect(),
        _ => return None,
    };
    Some(out)
}

impl ChunkedKvBacking {
    /// Dump float K and V data for all chunks in a sequence.
    ///
    /// For each chunk in the sequence at `batch_idx` that resides in a float arena,
    /// returns `(block_idx, k_flat_f32, v_flat_f32)` where each `Vec<f32>` contains
    /// `n_kv_head * chunk_size * head_dim` values in `[head, token, dim]` order.
    ///
    /// Chunks already migrated to quantized arenas are skipped.  For pure float
    /// sessions (F16 / F32 / BF16) all chunks remain float and nothing is skipped.
    pub fn dump_sequence_float_chunks(
        &self,
        batch_idx: usize,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>)>> {
        let elems = self.inner.elems_per_chunk();

        // Snapshot every band as `(arena_idx, chunk_idx, tag)` while holding the
        // state lock briefly. The tag travels with the gid because it is the
        // only thing that says how to decode the slot's bytes — a size-class
        // arena is untyped (`docs/archived/arena_unification.md` principle 8).
        //
        // IMPORTANT: all N_PALETTE sub-bands are dumped, not just palette 0.
        // Dumping one palette would emit a quarter of each head and quietly
        // invalidate the offline selection analysis this feeds.
        type Band = (usize, usize, ArenaFormatTag);
        let block_bands: Vec<Vec<(Band, Band)>> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(vec![]),
            };
            slot.chunks_slice()
                .iter()
                .map(|cw| {
                    // `bands()` yields K,V interleaved in slot order, which is
                    // head-major then palette-major — the layout this dump
                    // reconstructs.
                    let flat: Vec<Band> = cw
                        .bands()
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    flat.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|kv| (kv[0], kv[1]))
                        .collect()
                })
                .collect()
        };
        if block_bands.is_empty() {
            return Ok(vec![]);
        }

        let mut result = Vec::with_capacity(block_bands.len());
        for (block_idx, bands) in block_bands.iter().enumerate() {
            let maybe_kv: Option<(Vec<f32>, Vec<f32>)> = self.inner.storage.read(|s| {
                let arenas = s.arenas();
                let mut k_all = Vec::with_capacity(elems * bands.len());
                let mut v_all = Vec::with_capacity(elems * bands.len());
                for &((k_ai, k_ci, k_tag), (v_ai, v_ci, v_tag)) in bands.iter() {
                    k_all.extend(read_float_band(arenas.get(&k_ai)?, k_ci, k_tag, elems)?);
                    v_all.extend(read_float_band(arenas.get(&v_ai)?, v_ci, v_tag, elems)?);
                }
                Some((k_all, v_all))
            })?;

            if let Some((k_data, v_data)) = maybe_kv {
                result.push((block_idx, k_data, v_data));
            }
        }

        Ok(result)
    }

    /// Dump K (with captured Q) and V data for chunks stored as R16.
    ///
    /// Returns `(block_idx, k_flat_f32, v_flat_f32, q_flat_f32)` for each chunk
    /// whose K side resides in an R16 quantized arena.  Each `Vec<f32>` contains
    /// `n_kv_head * chunk_size * head_dim` values in palette-major layout, matching
    /// `dump_sequence_float_chunks`: `[head][palette][token][sub_dim]`.
    ///
    /// R16 block layout (per (head, palette) sub-arena chunk):
    ///
    /// ```text
    /// sub_head_dim = head_dim / N_PALETTE blocks of 128 bytes each
    /// block[d] = { F16 d[32]  // K values for tokens 0..32 at dim d
    ///            , u16 q[32]  // F16 Q values for tokens 0..32 at dim d
    ///            }
    /// ```
    ///
    /// Chunks where K is not R16 (e.g., a partial unsealed tail in float, or
    /// already-migrated quantized chunks) are skipped.
    pub fn dump_sequence_r16_kv_chunks(
        &self,
        batch_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> {
        use crate::kv_cache::arena_table::N_PALETTE;
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let sub_head_dim = (head_dim / N_PALETTE).max(1);
        // Bytes occupied by a single (head, palette) chunk's R16 storage:
        // sub_head_dim blocks × 128 bytes/block.
        let r16_bytes_per_chunk = sub_head_dim * 128;
        let elems_per_subband = CHUNK_SIZE * sub_head_dim;

        // Resolve (absolute_block_idx, head_gids) for the requested block
        // range.  `block_range = None` walks every chunk in the sequence;
        // `Some((lo, hi))` walks only chunks in `[lo, hi)`.
        type Band = (usize, usize, ArenaFormatTag);
        let block_gids: Vec<(usize, Vec<(Band, Band)>)> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(vec![]),
            };
            let chunks = slot.chunks_slice();
            let (lo, hi) = match block_range {
                Some((l, h)) => (l.min(chunks.len()), h.min(chunks.len())),
                None => (0, chunks.len()),
            };
            if hi <= lo {
                return Ok(vec![]);
            }
            chunks[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, cw)| {
                    // Each band carries its own tag: the arena it points into is
                    // an untyped byte slab and cannot say whether the slot holds
                    // R16 or a float.
                    let flat: Vec<Band> = cw
                        .bands()
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    (
                        lo + i,
                        flat.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|kv| (kv[0], kv[1]))
                            .collect(),
                    )
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok(vec![]);
        }

        let total_per_chunk = n_kv_head * N_PALETTE * elems_per_subband;
        let mut result = Vec::with_capacity(block_gids.len());
        for (block_idx, head_gids) in block_gids.iter() {
            // K + Q come from R16 raw bytes per (head, palette) chunk.
            // V comes from Float arenas (R16 mode keeps V float).
            // Layout matches dump_sequence_float_chunks: palette-major within head,
            // each (head, palette) sub-band contributes CHUNK_SIZE × sub_head_dim
            // elements in token-major order.
            let mut k_all: Vec<f32> = Vec::with_capacity(total_per_chunk);
            let mut v_all: Vec<f32> = Vec::with_capacity(total_per_chunk);
            let mut q_all: Vec<f32> = Vec::with_capacity(total_per_chunk);
            let mut ok = true;

            let _ = self.inner.storage.read(|s| {
                let arenas = s.arenas();
                for &((k_ai, k_ci, k_tag), (v_ai, v_ci, v_tag)) in head_gids.iter() {
                    // K + Q come from an R16 band; the band's own tag says so.
                    let Some(k_arena) = arenas.get(&k_ai) else {
                        ok = false;
                        return Ok::<(), candle::Error>(());
                    };
                    if k_tag != ArenaFormatTag::R16 {
                        ok = false;
                        return Ok(());
                    }
                    // Ranged read: only this slot's bytes leave VRAM.
                    let Ok(view) = k_arena.slot_bytes(k_ci, r16_bytes_per_chunk) else {
                        ok = false;
                        return Ok(());
                    };
                    let chunk_bytes = view.to_vec1::<u8>()?;
                    // Token-major append into the (head, palette) sub-band.
                    // R16 storage is dim-major: block[d][t]. Reorder to [t][d].
                    for t in 0..CHUNK_SIZE {
                        for d in 0..sub_head_dim {
                            let blk_off = d * 128;
                            let k_lo = chunk_bytes[blk_off + t * 2];
                            let k_hi = chunk_bytes[blk_off + t * 2 + 1];
                            let k_h = half::f16::from_le_bytes([k_lo, k_hi]);
                            let q_lo = chunk_bytes[blk_off + 64 + t * 2];
                            let q_hi = chunk_bytes[blk_off + 64 + t * 2 + 1];
                            let q_h = half::f16::from_le_bytes([q_lo, q_hi]);
                            k_all.push(k_h.to_f32());
                            q_all.push(q_h.to_f32());
                        }
                    }

                    // V is float in R16 mode.
                    let Some(v_arena) = arenas.get(&v_ai) else {
                        ok = false;
                        return Ok(());
                    };
                    match read_float_band(v_arena, v_ci, v_tag, elems_per_subband) {
                        Some(head_data) => v_all.extend(head_data),
                        None => {
                            ok = false;
                            return Ok(());
                        }
                    }
                }
                Ok(())
            })?;

            if ok {
                result.push((*block_idx, k_all, v_all, q_all));
            }
        }

        Ok(result)
    }

    /// Fast CUDA path: gather R16 K/Q and float-F16 V chunks into a single kernel launch.
    ///
    /// Returns the same `(block_idx, k_flat_f32, v_flat_f32, q_flat_f32)` format as
    /// `dump_sequence_r16_kv_chunks`, but replaces O(n_head × N_PALETTE × n_blocks)
    /// synchronous `memcpy_dtov` calls with:
    ///   1. One tiny HtoD upload of resolved chunk pointers (~8 bytes × n_warps)
    ///   2. One async kernel launch
    ///   3. One synchronous DtoH copy of all three output tensors
    ///
    /// Falls back to `dump_sequence_r16_kv_chunks` on non-CUDA devices.
    /// Skips blocks where K is not R16 or V is not float-F16.
    #[cfg(feature = "cuda")]
    pub fn gather_r16_kv_probe(
        &self,
        batch_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> {
        use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::kernels;

        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let sub_head_dim = (head_dim / N_PALETTE).max(1);
        let elems_per_subband = CHUNK_SIZE * sub_head_dim;
        let elems_per_block = n_kv_head * N_PALETTE * elems_per_subband;

        let Device::Cuda(cuda_dev) = &self.inner.device else {
            return self.dump_sequence_r16_kv_chunks(batch_idx, block_range);
        };

        // Step 1: collect (absolute_block_idx, [(k_ai, k_ci, v_ai, v_ci)]) for the range.
        type ProbeBand = (usize, usize, ArenaFormatTag);
        let block_gids: Vec<(usize, Vec<(ProbeBand, ProbeBand)>)> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(vec![]),
            };
            let chunks = slot.chunks_slice();
            let (lo, hi) = match block_range {
                Some((l, h)) => (l.min(chunks.len()), h.min(chunks.len())),
                None => (0, chunks.len()),
            };
            if hi <= lo {
                return Ok(vec![]);
            }
            chunks[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, cw)| {
                    // The tag rides along with the gid: whether a band is R16 or
                    // F16 is recorded on the chunk, and the arena it points into
                    // is an untyped byte slab that cannot answer.
                    let flat: Vec<(usize, usize, ArenaFormatTag)> = cw
                        .bands()
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    (
                        lo + i,
                        flat.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|kv| (kv[0], kv[1]))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: resolve arena info and build per-warp pointer arrays.
        // Only include blocks where all (h,p) K sides are R16 and V sides are F16.
        let arena_info = self.inner.resolve_arena_info()?;

        let mut k_ptrs: Vec<i64> = Vec::new();
        let mut v_ptrs: Vec<i64> = Vec::new();
        let mut r16_block_indices: Vec<usize> = Vec::new();

        for (block_idx, head_gids) in &block_gids {
            let is_r16_f16 = head_gids.iter().all(|&((_, _, k_tag), (_, _, v_tag))| {
                k_tag == ArenaFormatTag::R16 && v_tag == ArenaFormatTag::F16
            });
            if !is_r16_f16 {
                continue;
            }

            // Guard: both K (R16) and V (Float F16) arenas must be GPU-backed.
            // CPU-backed arenas have base_ptr=0; passing 0+ci*stride to the kernel
            // produces a small non-zero address that CUDA maps to unallocated memory
            // → ILLEGAL_ADDRESS fault on the GPU stream.
            let all_ptrs_nonzero = head_gids.iter().all(|&((k_ai, ..), (v_ai, ..))| {
                arena_info.get(k_ai).is_some_and(|a| a.base_ptr != 0)
                    && arena_info.get(v_ai).is_some_and(|a| a.base_ptr != 0)
            });
            if !all_ptrs_nonzero {
                tracing::error!(
                    block_idx,
                    "gather_r16_kv_probe: null arena base_ptr — arena not yet \
                     GPU-resident, skipping block"
                );
                continue;
            }

            r16_block_indices.push(*block_idx);
            for &((k_ai, k_ci, _), (v_ai, v_ci, _)) in head_gids.iter() {
                let k_arena = &arena_info[k_ai];
                let v_arena = &arena_info[v_ai];
                k_ptrs.push(k_arena.base_ptr as i64 + k_ci as i64 * k_arena.chunk_byte_stride);
                v_ptrs.push(v_arena.base_ptr as i64 + v_ci as i64 * v_arena.chunk_byte_stride);
            }
        }
        let mut result: Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> =
            Vec::with_capacity(block_gids.len());

        if !r16_block_indices.is_empty() {
            let n_r16_blocks = r16_block_indices.len();
            let n_warps = n_r16_blocks * n_kv_head * N_PALETTE;
            let total_elems = n_warps * elems_per_subband;

            // Step 3: HtoD upload of pointer arrays, alloc combined output, launch.
            let k_ptrs_gpu = cuda_dev.memcpy_stod(&k_ptrs)?;
            let v_ptrs_gpu = cuda_dev.memcpy_stod(&v_ptrs)?;

            // Combined buffer: [K section][Q section][V section], each `total_elems` halves.
            let out_kqv = unsafe { cuda_dev.alloc::<half::f16>(3 * total_elems)? };

            let stream = cuda_dev.cuda_stream();
            {
                let (kp, _kg) = k_ptrs_gpu.device_ptr(&stream);
                let (vp, _vg) = v_ptrs_gpu.device_ptr(&stream);
                let (okqv, _g) = out_kqv.device_ptr(&stream);
                candle::set_kernel_breadcrumb("run_gather_r16_kv_f16", file!(), line!());
                unsafe {
                    kernels::simple::gather_r16_kv::run_gather_r16_kv_f16(
                        kp as *const i64,
                        vp as *const i64,
                        okqv as *mut std::ffi::c_void,
                        n_warps as i32,
                        sub_head_dim as i32,
                        stream.cu_stream() as *mut _,
                    );
                }
            }

            // Step 4: single DtoH copy of combined K/Q/V output.
            let kqv_cpu: Vec<half::f16> = cuda_dev.memcpy_dtov(&out_kqv)?;
            let k_cpu = &kqv_cpu[..total_elems];
            let q_cpu = &kqv_cpu[total_elems..2 * total_elems];
            let v_cpu = &kqv_cpu[2 * total_elems..];

            // Step 5: transpose d-major kernel output → token-major and convert F16 → F32.
            //
            // The kernel writes d-major within each warp (= one head×palette sub-band):
            //   kqv[warp_off + d * CHUNK_SIZE + token]
            // The consumer (r16_block_to_turn_signatures) expects token-major:
            //   q_flat[warp_off + token * sub_head_dim + d]
            //
            // We fold the transpose into the F16→F32 conversion — no extra allocation.
            let n_subbands = n_kv_head * N_PALETTE;
            for (bi, block_idx) in r16_block_indices.iter().enumerate() {
                let mut k_f32 = Vec::with_capacity(elems_per_block);
                let mut v_f32 = Vec::with_capacity(elems_per_block);
                let mut q_f32 = Vec::with_capacity(elems_per_block);
                for warp_local in 0..n_subbands {
                    let warp_off = bi * elems_per_block + warp_local * elems_per_subband;
                    for t in 0..CHUNK_SIZE {
                        for d in 0..sub_head_dim {
                            // d-major read → token-major push order
                            let src = warp_off + d * CHUNK_SIZE + t;
                            k_f32.push(k_cpu[src].to_f32());
                            v_f32.push(v_cpu[src].to_f32());
                            q_f32.push(q_cpu[src].to_f32());
                        }
                    }
                }
                result.push((*block_idx, k_f32, v_f32, q_f32));
            }
        }

        Ok(result)
    }
}
