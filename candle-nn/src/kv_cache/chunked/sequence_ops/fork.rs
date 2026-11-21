//! Forks a sequence's KV cache into another slot, sharing whole chunks and
//! copying the partial tail.
//!
//! Complete chunks are shared copy-on-write by Arc, and the remainder is copied
//! into a new chunk. Where a band's source and destination chunk classes differ,
//! the band is re-encoded in the destination format rather than copied, and its
//! tag moves with it, because the fork changes the format of the tail. The fork
//! is uninterruptible: refusing part-way would leave the layers with different
//! token windows, which the decode position map cannot describe.
//! `fork_sequence_alloc` combines allocation and fork and releases its slot if
//! the fork fails.

use super::*;

impl ChunkedKvBacking {
    /// Fork (clone) a sequence's KV cache to another sequence slot.
    ///
    /// This creates a copy of the source sequence's cache in the target slot:
    /// - Complete chunks are shared via COW (copy-on-write) - no data copy
    /// - The partial/remainder tokens in the last incomplete chunk are copied
    ///   into a new chunk allocated for the target
    ///
    /// This is useful for beam search, speculative decoding, or any scenario
    /// where you need to branch from an existing sequence state.
    ///
    /// # Arguments
    /// * `source_batch` - The source sequence to fork from
    /// * `target_batch` - The target sequence slot (will be overwritten)
    /// * `seq_len` - The number of tokens in the source sequence
    ///
    /// # Returns
    /// The number of tokens in the forked sequence (same as `seq_len`)
    pub fn fork_sequence(
        &self,
        source_batch: usize,
        target_batch: usize,
        seq_len: usize,
    ) -> Result<usize> {
        let batch = self.batch_capacity();
        if source_batch >= batch {
            candle::bail!(
                "source_batch {} out of range for chunked backing (capacity {})",
                source_batch,
                batch
            )
        }
        if target_batch >= batch {
            candle::bail!(
                "target_batch {} out of range for chunked backing (capacity {})",
                target_batch,
                batch
            )
        }
        if source_batch == target_batch {
            candle::bail!("cannot fork sequence to itself")
        }
        if seq_len == 0 {
            // Just free target if any
            self.free_sequence(target_batch)?;
            return Ok(0);
        }

        let full_blocks = seq_len / CHUNK_SIZE;
        let remainder = seq_len % CHUNK_SIZE;
        let total_blocks = if remainder > 0 {
            full_blocks + 1
        } else {
            full_blocks
        };

        if total_blocks > 0 {
            self.ensure_max_blocks(total_blocks)?;
        }

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Validate source slot is allocated and has enough blocks
        let source_slot = state.sequences[source_batch].as_ref().ok_or_else(|| {
            candle::Error::Msg(format!("source slot {} not allocated", source_batch))
        })?;
        if total_blocks > source_slot.block_count() {
            candle::bail!(
                "source sequence {} only has {} blocks allocated, need {} for seq_len {}",
                source_batch,
                source_slot.block_count(),
                total_blocks,
                seq_len
            )
        }

        // Validate all needed source blocks are allocated
        for blk in 0..total_blocks {
            if source_slot.chunk_at(blk).is_none() {
                candle::bail!(
                    "source sequence {} block {} is not allocated",
                    source_batch,
                    blk
                )
            }
        }

        // Free target's existing blocks if slot exists
        // Collect chunks to free first to avoid borrow conflicts
        if let Some(ts) = state.sequences[target_batch].as_mut() {
            // Cleared GIDs are dropped via RAII, returning to pool
            ts.clear_chunks();
        }

        // Ensure target slot exists
        if state.sequences[target_batch].is_none() {
            state.sequences[target_batch] = Some(self.make_sequence_state()?);
        }

        // Share full blocks (COW) - clone increments Gid refcount
        {
            let source_slot = state.sequences[source_batch].as_ref().unwrap();
            let source_chunks: Vec<ChunkWindow> = (0..full_blocks)
                .map(|blk| {
                    let cw = source_slot.chunk_at(blk).unwrap();
                    ChunkWindow {
                        gids: cw.gids.clone(),
                        usage: CHUNK_SIZE as u32,
                        offset: 0,
                        k_pal: cw.k_pal.clone(),
                        v_pal: cw.v_pal.clone(),
                        k_scale: cw.k_scale.clone(),
                        v_scale: cw.v_scale.clone(),
                        k_fmt: cw.k_fmt.clone(),
                        v_fmt: cw.v_fmt.clone(),
                        // Fork shares the source chunk's GIDs (Arc bump), so its
                        // resident record's pointers stay valid — share it too.
                        meta: cw.meta.clone(),
                        q_sign: cw.q_sign.clone(),
                    }
                })
                .collect();
            let target_slot = state.sequences[target_batch].as_mut().unwrap();
            target_slot.replace_chunks(source_chunks);
            // The shared full blocks are the source's sealed history; the fork
            // writes only from the copied remainder chunk (or a fresh one) on.
            target_slot.set_writer_start_idx(full_blocks);
        }

        // Handle the last block with remainder -- copy per-head data from
        // potentially different source arenas into a single new float chunk.
        if remainder > 0 {
            let last_blk = full_blocks;
            let n_kv_head = self.inner.n_kv_head;

            // Clone the source block's full per-head GID vector AND its
            // pal_map / outer-scale state — the partial-block copy below
            // preserves the source's encoded byte semantics, so the new
            // chunk needs the same metadata to decode correctly.
            let (
                source_gids,
                source_k_pal,
                source_v_pal,
                source_k_scale,
                source_v_scale,
                source_k_fmt,
                source_v_fmt,
            ): (
                HeadGids,
                Arc<Vec<u8>>,
                Arc<Vec<u8>>,
                Arc<Vec<f32>>,
                Arc<Vec<f32>>,
                Arc<Vec<u8>>,
                Arc<Vec<u8>>,
            ) = {
                let cw = state.sequences[source_batch]
                    .as_ref()
                    .unwrap()
                    .chunk_at(last_blk)
                    .unwrap();
                (
                    cw.gids.clone(),
                    cw.k_pal.clone(),
                    cw.v_pal.clone(),
                    cw.k_scale.clone(),
                    cw.v_scale.clone(),
                    cw.k_fmt.clone(),
                    cw.v_fmt.clone(),
                )
            };
            // A verbatim R16 copy carries Q with it; a source whose Q is already
            // gone hands its captured bits on, which describe the same tokens.
            let source_q_sign = state.sequences[source_batch]
                .as_ref()
                .and_then(|s| s.chunk_at(last_blk))
                .and_then(|cw| cw.q_sign.clone());

            // Allocate new chunks using active_k_arena_key (R16 on GPU, Float on CPU)
            // so the active decode kernel can write directly into them.
            let k_key = self.active_k_arena_key()?;
            let v_key = self.active_v_arena_key()?;
            // Per-band tags for the forked block. They start as the source's and
            // are rewritten for any band the loop below has to re-encode, since
            // the tag is the only record of how a band's bytes decode.
            let mut k_fmt_out: Vec<u8> = source_k_fmt.as_ref().clone();
            let mut v_fmt_out: Vec<u8> = source_v_fmt.as_ref().clone();
            let head_dim = self.inner.head_dim;
            let device = self.inner.device.clone();
            let on_gpu = matches!(
                self.inner.storage.default_location(),
                crate::kv_cache::ArenaLocation::Gpu
            );
            let (active_k_fmt, active_v_fmt) =
                active_kv_formats(self.inner.storage.k_format(), on_gpu);
            // The fork's destination chunks can need an arena that does not
            // exist, and `alloc_chunk_with_arenas` creates it under the storage
            // write lock — so the window is taken here, above that lock.
            //
            // **Uninterruptible**, because a fork is per layer and the caller
            // loops over all of them with no transaction: refusing here forks
            // some layers and not others, and a sequence whose token windows
            // differ by layer is exactly what the decode position map cannot
            // describe. Proceeding past a wave-in-flight refusal claims at
            // worst one region mid-wave, which the `in-wave-arenas` tripwire
            // reports; refusing corrupts the sequence silently. A genuine
            // error still propagates — the `?` fires only for failures that
            // are not the deferral, and those mean the gate itself is broken.
            #[cfg(feature = "cuda")]
            let _window = self.inner.arena_window_uninterruptible(k_key)?;
            let target_gids = self.inner.storage.write(|arena_state| {
                let mut gid_vec: Vec<ChunkGid> = Vec::with_capacity(GIDS_PER_HEAD * n_kv_head);

                for i in 0..(GIDS_PER_HEAD * n_kv_head) {
                    let key = if i % 2 == 0 { k_key } else { v_key };
                    let gid = self.alloc_chunk_with_arenas(arena_state, key)?;
                    gid_vec.push(gid);
                }

                // Copy each source slot to its destination, byte-verbatim: the
                // band's tag travels with the chunk, so a slot's bytes mean the
                // same thing at the destination as at the source.
                //
                // **This requires source and destination to share a size
                // class**, and that is not guaranteed. Destinations come from
                // the *active* key above (R16, the 4096 B class) so decode can
                // append to the forked tail, while the source is whatever the
                // boundary block is in — and a sealed partial tail is quantized
                // like any other chunk, so it can be sitting in the 1088 B
                // `Q8_0` class. `copy_slot_bytes` refuses that, correctly: the
                // class follows the *format*, and fork deliberately changes the
                // format of the tail. Such a band is re-encoded instead — read
                // through its own tag in the kernels' token-oriented layout and
                // written in the destination's (`band_codec`), its stored
                // values carried unchanged under the inherited palette map and
                // outer scale.
                let arenas = arena_state.arenas_mut();
                let sub_head_dim = (head_dim / N_PALETTE).max(1);
                for (i, src_gid) in source_gids.iter().enumerate() {
                    let dst_gid = &gid_vec[i];
                    let is_v = i % 2 == 1;
                    let band = i / 2;
                    let src_stride = arenas
                        .get(&src_gid.arena_idx())
                        .map(|a| a.slot_stride())
                        .unwrap_or(0);
                    let dst_stride = arenas
                        .get(&dst_gid.arena_idx())
                        .map(|a| a.slot_stride())
                        .unwrap_or(0);
                    if src_stride == dst_stride {
                        ChunkedKvBacking::copy_slot_bytes(
                            arenas,
                            src_gid.arena_idx(),
                            src_gid.chunk_idx(),
                            dst_gid.arena_idx(),
                            dst_gid.chunk_idx(),
                        )?;
                        continue;
                    }
                    // Classes differ, so this band changes format and its bytes
                    // cannot travel verbatim. Read it as floats through its own
                    // tag and re-encode into the active format.
                    // `read_band_chunk` yields the canonical
                    // `(CHUNK_SIZE, sub_head_dim)` token-major tensor for a
                    // float *or* a quantized tag (a quantized band decoded from
                    // its token-oriented blocks), and `encode_band` takes the
                    // same layout back.
                    let src_tag = ArenaFormatTag::from_u8(if is_v {
                        source_v_fmt[band]
                    } else {
                        source_k_fmt[band]
                    });
                    // Taken off the source arena before the destination is
                    // borrowed mutably below: `read_band_chunk` yields a lease
                    // over `arenas`, and re-encoding writes through the same
                    // map. The copy is what lets the two coexist, and it is the
                    // reason this arm is the slow path — a same-class band
                    // takes the verbatim byte copy above instead.
                    let floats =
                        read_band_chunk(arenas, src_gid, src_tag, !is_v, sub_head_dim, &device)?
                            .to_owned_tensor()?;
                    let dst_fmt = if is_v { active_v_fmt } else { active_k_fmt };
                    let dst = arenas.get_mut(&dst_gid.arena_idx()).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "fork: dest arena {} not found",
                            dst_gid.arena_idx()
                        ))
                    })?;
                    match dst_fmt {
                        KvFormat::Float(dtype) => {
                            dst.write_slot_typed(dst_gid.chunk_idx(), 0, &floats.to_dtype(dtype)?)?
                        }
                        KvFormat::Quantized(qf) => {
                            let values = floats
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let bytes = encode_band(&values, qf, !is_v, sub_head_dim)?;
                            let len = bytes.len();
                            dst.write_slot_bytes(
                                dst_gid.chunk_idx(),
                                &Tensor::from_vec(bytes, len, &device)?,
                            )?
                        }
                    }
                    // The tag is the only record of how a band's bytes decode,
                    // so it has to move with the format.
                    let new_tag = ArenaFormatTag::from_kv_format(dst_fmt).as_u8();
                    if is_v {
                        v_fmt_out[band] = new_tag;
                    } else {
                        k_fmt_out[band] = new_tag;
                    }
                }

                Ok::<_, candle::Error>(gid_vec)
            })??;

            // Push remainder block with per-head GIDs. The arenas are freshly
            // allocated but the data was *copied* from the source, so the
            // chunk inherits the source's pal_map and outer-scale metadata
            // (otherwise the decoder would misinterpret the copied bytes).
            let gids = HeadGids::from_vec(target_gids);
            state.sequences[target_batch]
                .as_mut()
                .unwrap()
                .push_chunk(ChunkWindow {
                    gids,
                    usage: remainder as u32,
                    offset: 0,
                    k_pal: source_k_pal,
                    v_pal: source_v_pal,
                    k_scale: source_k_scale,
                    v_scale: source_v_scale,
                    // Per-band: verbatim where the class matched, re-encoded to
                    // the active format where it did not — `k_fmt_out`/`v_fmt_out`
                    // record whichever happened.
                    k_fmt: Arc::new(k_fmt_out),
                    v_fmt: Arc::new(v_fmt_out),
                    // CoW partial: freshly-allocated GIDs (copied data) → no
                    // resident record; falls back to per-forward scratch heads.
                    meta: None,
                    q_sign: source_q_sign,
                });
        }

        // block_count() = chunks.len() = total_blocks. ✓

        drop(state);

        Ok(seq_len)
    }

    /// Fork a sequence, automatically allocating a new slot for the target.
    ///
    /// This combines `alloc_sequence` and `fork_sequence` into a single operation.
    /// Complete blocks are shared (COW), partial blocks are copied.
    ///
    /// Returns the batch index of the newly allocated target sequence.
    pub fn fork_sequence_alloc(&self, source_batch: usize, seq_len: usize) -> Result<usize> {
        let target_batch = self.alloc_sequence()?;

        match self.fork_sequence(source_batch, target_batch, seq_len) {
            Ok(_) => Ok(target_batch),
            Err(e) => {
                // Clean up on error
                let _ = self.free_sequence(target_batch);
                Err(e)
            }
        }
    }
}
