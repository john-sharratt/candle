//! Writes a turn's KV chunks, tokens and commit to the redo log, in the order a reload needs them.
//!
//! Each chunk is written at the flat index its layer and chunk give, the tokens as one record, and a
//! commit marks the stream durable. Every layer seals in lockstep, so each layer has the same chunk
//! count. The tokens are written on the hot path whatever the compression, because the turn cannot be
//! rebuilt without them. The chunk records are written after quantization, so their bytes carry the
//! selected formats. The location of each write is folded into the substrate's index, since a record
//! that is not registered there is skipped by compaction and reclaimed when its segment drops.

use super::*;

/// Persist a sealed turn's KV grid to the active log: every `(layer, chunk)`
/// chunk as a `Chunk` record at its flat index, the turn's token ids as a
/// `Tokens` record, then a `Commit` marking the stream durable.
///
/// Every layer must have the same chunk count — the architecture seals
/// all layers in lockstep.
pub fn persist_turn_kv(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    layers: &TurnChunkGrid,
    token_ids: &[u32],
) -> Result<()> {
    persist_turn_chunks(p, stream_id, layers)?;
    persist_turn_tokens(p, stream_id, token_ids, layers)?;
    Ok(())
}

/// Persist only a turn's `Chunks` records — the post-quantization half of the
/// seal/persist chain (§16.12). Called from inside a bg-quantizer callback
/// once the turn's float→quant migrations have landed, so the bytes captured
/// here reflect the policy-selected K/V formats.
pub fn persist_turn_chunks(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    layers: &TurnChunkGrid,
) -> Result<()> {
    let chunks_per_layer = layers.chunks_per_layer();
    for (layer_idx, layer) in layers.iter_layers().enumerate() {
        if layer.len() != chunks_per_layer {
            return Err(PersistenceError::Corrupt(format!(
                "layer {layer_idx} has {} chunks, expected {chunks_per_layer}",
                layer.len()
            )));
        }
        for (chunk_idx, image) in layer.iter().enumerate() {
            let flat = flat_chunk_index(layer_idx, chunk_idx, chunks_per_layer);
            // The `Chunk` record's `format` header field carries a single
            // representative K format (sub-band 0) for quick inspection; the
            // authoritative per-sub-band K/V maps live in the `ChunkPayload`.
            let header_fmt = image.payload.k_formats.first().copied().unwrap_or(0);
            p.write_chunk(
                stream_id,
                flat,
                image.token_count as u64,
                header_fmt,
                Some(image.golden),
                &image.payload,
            )?;
        }
    }
    Ok(())
}

/// Persist a turn's chunks **and** return the per-layer cold references
/// the substrate's residence slab needs to wire `cold = Some(...)`.
///
/// Mirrors [`persist_turn_chunks`] (same on-disk effect) but captures the
/// `log_offset` returned by each `append_record` call so the
/// [`StoredChunk`] reference is exact. Used by the persistence thread's
/// warm→cold phase.
///
/// Also returns the `(flat_chunk_index, ChunkLoc)` pairs, each with the
/// identity its record was written with, that the caller must fold into the
/// substrate's authoritative chunk index (`apply_chunk_loc`). Without that,
/// `stream.chunks` is only repopulated by the walker on reload, so an
/// in-process cold→hot elevation (a turn evicted then re-accessed within the
/// same session) would `plan_chunked_read` an empty index and recover nothing.
pub fn persist_turn_chunks_capture(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    layers: &TurnChunkGrid,
) -> Result<(Vec<StoredSequence>, Vec<(u64, Stamped<ChunkLoc>)>)> {
    let chunks_per_layer = layers.chunks_per_layer();
    let mut out = Vec::with_capacity(layers.n_layers());
    let mut locs: Vec<(u64, Stamped<ChunkLoc>)> = Vec::new();
    for (layer_idx, layer) in layers.iter_layers().enumerate() {
        if layer.len() != chunks_per_layer {
            return Err(PersistenceError::Corrupt(format!(
                "layer {layer_idx} has {} chunks, expected {chunks_per_layer}",
                layer.len()
            )));
        }
        let mut stored = Vec::with_capacity(layer.len());
        let mut layer_token_count = 0usize;
        for (chunk_idx, image) in layer.iter().enumerate() {
            let flat = flat_chunk_index(layer_idx, chunk_idx, chunks_per_layer);
            let header_fmt = image.payload.k_formats.first().copied().unwrap_or(0);
            let encoded = image.payload.encode();
            let payload_len = encoded.len() as u64;
            let at = p.append_record(
                RecordType::Chunk,
                header_fmt,
                stream_id.0,
                flat,
                image.token_count as u64,
                image.golden,
                &encoded,
            )?;
            stored.push(StoredChunk {
                log_offset: at.offset,
                record_len: at.size,
                token_count: image.token_count,
            });
            locs.push((
                flat,
                Stamped {
                    value: ChunkLoc {
                        // The active segment this chunk's record landed in — a
                        // rotation mid-turn can land later chunks in a newer
                        // segment, so each chunk carries its own id.
                        segment: at.segment,
                        offset: at.offset,
                        payload_len,
                        record_size: at.size,
                        token_count: image.token_count as u64,
                        format: header_fmt,
                    },
                    id: at.identity,
                },
            ));
            layer_token_count += image.token_count as usize;
        }
        out.push(StoredSequence {
            chunks: stored,
            token_count: layer_token_count,
        });
    }
    Ok((out, locs))
}

/// Persist a turn's `Tokens` record and the trailing `Commit`. Always called
/// synchronously on seal — tokens are substrate-critical reconstruction data
/// regardless of compression policy. The `layers` argument is used solely to
/// compute the highest committed chunk index; pass an empty grid when no
/// chunks were persisted (e.g. `compression_policy = None`).
pub fn persist_turn_tokens(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    token_ids: &[u32],
    layers: &TurnChunkGrid,
) -> Result<()> {
    p.append_tokens(stream_id, &encode_token_ids(token_ids))?;
    let total = layers.n_layers() * layers.chunks_per_layer();
    let through = if total > 0 { total as u64 - 1 } else { 0 };
    p.commit_stream(stream_id, through)?;
    Ok(())
}

/// Write **only** a turn's `Tokens` record — no trailing `Commit`. Used
/// by the synchronous seal path when chunks (and the matching `Commit`)
/// are deferred to the persistence thread. Tokens are tiny and load-
/// bearing for reconstruction, so they stay on the hot path even when
/// chunks don't.
pub fn persist_tokens_only(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    token_ids: &[u32],
) -> Result<Stamped<RecordLoc>> {
    // Return the written record's location so the CALLER can fold it into the
    // substrate index (`apply_tokens_loc`). Without that, `entry.tokens` stays
    // `None`, and both the maintenance relocation (`gather_relocations`) and the
    // full compaction (`collect_live_records`) gate on `if let Some(loc) =
    // entry.tokens` — so they silently SKIP the Tokens record and it's reclaimed
    // when its source segment drops (the KV path already registers its chunk
    // locations; tokens must too). This is why turn-0 (prefilled/adopted) tokens
    // vanished during idle maintenance while KV survived.
    let encoded = encode_token_ids(token_ids);
    let at = p.append_tokens(stream_id, &encoded)?;
    Ok(Stamped {
        value: at.loc(encoded.len() as u64),
        id: at.identity,
    })
}
