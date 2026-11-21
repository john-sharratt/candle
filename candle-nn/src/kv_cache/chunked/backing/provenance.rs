//! Finds where each block's provenance sign(Q) comes from, and runs the sign-pack
//! kernel.
//!
//! Q is stored beside K in the R16 chunk, so a block whose K is still R16 and
//! GPU-backed contributes its band pointers. A block the gap has already
//! converted contributes the signs it captured, and a block with neither is left
//! out. Every layer of a scope shares one launch, with one upload and one
//! download, and the launch is gated on the kernel's declared width.

use super::*;

impl ChunkedKvBacking {
    /// Where each of a sequence's blocks' provenance `sign(Q)` comes from, in
    /// block order: the R16 K-chunk base pointers the sign-pack kernel reads Q
    /// from (Q is co-located in the K chunk at +64 per dim-group),
    /// `[head][palette]` device addresses, for a block whose K is still R16 and
    /// GPU-backed — the filter [`Self::gather_r16_kv_probe`] uses, so all layers
    /// of a scope resolve alike — or the signs the gap captured for a block it
    /// has since converted ([`ProvSource::Signed`]). A block with neither has
    /// no Q to give and is left out. Empty on non-CUDA.
    pub fn provenance_sources(
        &self,
        batch_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, ProvSource)>> {
        use crate::kv_cache::arena_table::ArenaFormatTag;

        let Device::Cuda(_) = &self.inner.device else {
            return Ok(Vec::new());
        };

        // (absolute_block_idx, [(k_ai, k_ci, tag)], captured signs) — Q side
        // only (co-located with K).
        type BlockQ = (usize, Vec<(usize, usize, ArenaFormatTag)>, Option<QSign>);
        let block_gids: Vec<BlockQ> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(Vec::new()),
            };
            let chunks = slot.chunks_slice();
            let (lo, hi) = match block_range {
                Some((l, h)) => (l.min(chunks.len()), h.min(chunks.len())),
                None => (0, chunks.len()),
            };
            if hi <= lo {
                return Ok(Vec::new());
            }
            chunks[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, cw)| {
                    // K side only (Q is co-located), each band with its own tag.
                    let gids: Vec<(usize, usize, ArenaFormatTag)> = cw
                        .bands()
                        .step_by(2)
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    (lo + i, gids, cw.q_sign.clone())
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve only the arenas this scope's gids actually reference (a couple),
        // not the whole arena table — the bulk of this path's per-scope cost.
        let needed: std::collections::HashSet<usize> = block_gids
            .iter()
            .flat_map(|(_, gids, _)| gids.iter().map(|&(k_ai, ..)| k_ai))
            .collect();
        let arena_info = self.inner.resolve_arena_info_for(&needed)?;
        let mut out = Vec::with_capacity(block_gids.len());
        for (block_idx, head_gids, q_sign) in block_gids {
            // Q from the bands while every (h,p) K side is R16 and GPU-backed —
            // same filter as `gather_r16_kv_probe`, so layers stay aligned. The
            // R16-ness is the chunk answering; the residency is the arena.
            let readable = head_gids.iter().all(|&(k_ai, _, k_tag)| {
                k_tag == ArenaFormatTag::R16
                    && arena_info.get(k_ai).is_some_and(|a| a.base_ptr != 0)
            });
            if readable {
                let ptrs = head_gids
                    .iter()
                    .map(|&(k_ai, k_ci, _)| {
                        let k_arena = &arena_info[k_ai];
                        k_arena.base_ptr as i64 + k_ci as i64 * k_arena.chunk_byte_stride
                    })
                    .collect();
                out.push((block_idx, ProvSource::Q(ptrs)));
            } else if let Some(signs) = q_sign {
                // Converted after the gap captured its signs.
                out.push((block_idx, ProvSource::Signed(signs)));
            }
        }
        Ok(out)
    }

    /// Launch the provenance sign-pack kernel over R16 Q-chunk pointers spanning
    /// ALL layers of a scope, returning the packed sign bits (`n_warps ×
    /// CHUNK_SIZE` u64, warp-major: `out[warp*CHUNK_SIZE + token]` has bit `d`
    /// set iff Q dim `d` of that sub-band is `>= 0`). `all_q_ptrs` is the
    /// concatenation of every layer's [`ProvSource::Q`] pointers; all
    /// layers share this backing's device, so any backing can launch for the
    /// whole set. One HtoD (pointers) + one launch + one DtoH (packed bits) —
    /// replaces the per-layer f16 K/Q/V round-trips. CUDA only.
    #[cfg(feature = "cuda")]
    pub fn run_prov_sign_pack(&self, all_q_ptrs: &[i64], sub_head_dim: usize) -> Result<Vec<u64>> {
        use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
        use candle::cuda_backend::kernels;
        use candle::cuda_backend::staged::segment;

        let Device::Cuda(cuda_dev) = &self.inner.device else {
            return Ok(Vec::new());
        };
        // Gate on the kernel's own declared width. An out-of-range launch is a
        // silent no-op there, so the buffer below would be D2H'd uninitialised and
        // folded into signatures — this return is what makes that unreachable.
        if all_q_ptrs.is_empty()
            || sub_head_dim == 0
            || sub_head_dim > kernels::simple::prov_sign_pack::MAX_SUB_HEAD_DIM
        {
            return Ok(Vec::new());
        }
        let n_warps = all_q_ptrs.len();
        let out_len = n_warps * CHUNK_SIZE;

        // The pointers go up and the packed bits come back through the
        // device's held synced scratch: one HtoD, one launch, one DtoH, and
        // nothing allocated — a per-backing grow-only pair used to hold the
        // largest batch each layer had seen, outside the reservation.
        let stream = cuda_dev.cuda_stream();
        let mut out = vec![0u64; out_len];
        cuda_dev.with_synced_upload(
            &stream,
            &[segment(all_q_ptrs)],
            out_len * std::mem::size_of::<u64>(),
            |at, room| {
                candle::set_kernel_breadcrumb("run_prov_sign_pack", file!(), line!());
                // SAFETY: `at[0]` holds the `n_warps` uploaded pointers and
                // `room` the packed bits' space, both held until this returns;
                // the kernel writes every word of the output, and the readback
                // is ordered behind it and waited for before the scratch goes.
                unsafe {
                    kernels::simple::prov_sign_pack::run_prov_sign_pack(
                        at[0] as *const i64,
                        room as *mut std::ffi::c_void,
                        n_warps as i32,
                        sub_head_dim as i32,
                        stream.cu_stream() as *mut _,
                    );
                    memcpy_dtoh_async(&mut out, room, stream.cu_stream())
                        .map_err(candle::Error::wrap)?;
                }
                stream.synchronize().map_err(candle::Error::wrap)
            },
        )?;
        Ok(out)
    }

    /// Non-CUDA stub — no GPU kernel, so the provenance capture uses its CPU path.
    #[cfg(not(feature = "cuda"))]
    pub fn run_prov_sign_pack(
        &self,
        _all_q_ptrs: &[i64],
        _sub_head_dim: usize,
    ) -> Result<Vec<u64>> {
        Ok(Vec::new())
    }
}
