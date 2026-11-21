//! Reads a turn back from the redo log: its KV grid, its tokens and its cold references, without touching
//! bytes it does not need.
//!
//! A cold reload reads the small tokens record and the in-memory chunk index for the metadata. Chunk payloads
//! are read only when a projection brings the turn back to hot. The layer count is not stored in the log, so
//! it is taken as the mode of the turns' chunk counts, since a partly written turn would otherwise name the
//! wrong count. A turn whose chunks never landed is still valid with an empty grid, because its tokens and its
//! signatures were kept.

use super::*;

/// Demux a turn stream's flat chunk index set back into the `L × C` grid.
///
/// `flat[chunk_index] → ChunkImage`; the result is `layers[layer][chunk]`.
/// Errors if the flat index set is not exactly `0..L*C` for the `C` derived
/// from `chunks_per_layer`.
pub fn demux_layers(
    flat: Vec<(u64, ChunkImage)>,
    n_layers: usize,
    chunks_per_layer: usize,
) -> Result<TurnChunkGrid> {
    let expected = n_layers * chunks_per_layer;
    if flat.len() != expected {
        return Err(PersistenceError::Corrupt(format!(
            "turn has {} chunk records, expected {n_layers} layers × {chunks_per_layer} = {expected}",
            flat.len()
        )));
    }
    let mut layers: Vec<Vec<Option<ChunkImage>>> = vec![vec![None; chunks_per_layer]; n_layers];
    for (flat_idx, image) in flat {
        let idx = flat_idx as usize;
        if idx >= expected {
            return Err(PersistenceError::Corrupt(format!(
                "chunk index {idx} out of range for {expected} chunks"
            )));
        }
        let layer = idx / chunks_per_layer;
        let chunk = idx % chunks_per_layer;
        if layers[layer][chunk].is_some() {
            return Err(PersistenceError::Corrupt(format!(
                "duplicate chunk at layer {layer} chunk {chunk}"
            )));
        }
        layers[layer][chunk] = Some(image);
    }
    let materialised: Result<Vec<Vec<ChunkImage>>> = layers
        .into_iter()
        .enumerate()
        .map(|(layer, row)| {
            row.into_iter()
                .enumerate()
                .map(|(chunk, slot)| {
                    slot.ok_or_else(|| {
                        PersistenceError::Corrupt(format!(
                            "missing chunk at layer {layer} chunk {chunk}"
                        ))
                    })
                })
                .collect()
        })
        .collect();
    Ok(TurnChunkGrid::new(materialised?))
}

/// Recover a turn's KV chunk grid from the log: the demuxed `L × C`
/// [`TurnChunkGrid`] with every chunk's payload bytes. This is the
/// cold→hot elevation read — the only reload path that touches chunk
/// payloads. `decl` is the turn's already-decoded `StreamDecl`;
/// `n_layers` is the running model's layer count.
pub fn recover_turn_grid(
    p: &mut SubstratePersistence,
    substrate: &Substrate,
    decl: &TurnDecl,
    n_layers: usize,
) -> Result<TurnChunkGrid> {
    let stream_id = super::super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
    let chunks_per_layer = (decl.block_end - decl.block_start) as usize;

    // Snapshot per-chunk token_count from the substrate's stream
    // index before the `&mut p` read below — we need it to assemble
    // `ChunkImage`s once the batched read returns.  The chunk index →
    // token_count map is small (one u64 per chunk), so cloning is
    // cheap.
    let token_counts: HashMap<u64, u64> = substrate
        .stream_of(stream_id)
        .map(|s| s.chunks.iter().map(|(&i, l)| (i, l.token_count)).collect())
        .unwrap_or_default();

    // Stripe-coalesced batched read — one syscall per stripe across
    // active + inherited logs. For a freshly-persisted turn the
    // records are contiguous on disk → ~1 syscall covers all of them.
    let mut buf: Vec<u8> = Vec::new();
    let chunks_with_payload = p.read_stream_chunks_batched(substrate, stream_id, &mut buf)?;

    let mut flat: Vec<(u64, ChunkImage)> = Vec::with_capacity(chunks_with_payload.len());
    for (idx, payload) in chunks_with_payload {
        let token_count = token_counts.get(&idx).copied().unwrap_or(0) as u16;
        // The batched read already verified this payload against its stored
        // golden, so a host recompute over the KV bytes reproduces it — keeping
        // the `ChunkImage` self-consistent were it ever re-persisted.
        let golden = candle::fletcher::fletcher32(&payload.kv_bytes);
        flat.push((
            idx,
            ChunkImage {
                token_count,
                golden,
                payload,
            },
        ));
    }
    // A turn whose chunks haven't landed yet — either because the bg-quantizer
    // persist callback hadn't fired before a crash, or because the run was
    // configured with `compression_policy = None` — is still valid: its
    // tokens and signatures are preserved and the layer grid is reconstructed
    // empty. The substrate's reload path handles empty sealed sequences.
    if flat.is_empty() {
        Ok(TurnChunkGrid::empty_grid(n_layers))
    } else {
        demux_layers(flat, n_layers, chunks_per_layer)
    }
}

/// Recover a turn's reload metadata: token ids and the per-layer token
/// count. Reads only the (small) `Tokens` record; the token count comes
/// straight from the in-RAM chunk index, so the turn's KV payload bytes
/// are never touched. This is what the startup restore loop calls per
/// turn — cold KV stays on disk until a projection elevates it via
/// [`recover_turn_grid`].
pub fn recover_turn_meta(
    p: &mut SubstratePersistence,
    substrate: &Substrate,
    decl: &TurnDecl,
) -> Result<RecoveredTurnMeta> {
    let stream_id = super::super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
    let chunks_per_layer = (decl.block_end - decl.block_start) as usize;

    // One layer's worth of token counts — chunk indices below
    // `chunks_per_layer` are layer 0 under the flat Option A layout,
    // and all layers seal in lockstep.
    let token_count: usize = substrate
        .stream_of(stream_id)
        .map(|s| {
            s.chunks
                .range(..chunks_per_layer as u64)
                .map(|(_, l)| l.token_count as usize)
                .sum()
        })
        .unwrap_or(0);

    let token_ids = match p.read_tokens(substrate, stream_id)? {
        Some(bytes) => decode_token_ids(&bytes)?,
        None => Vec::new(),
    };

    Ok(RecoveredTurnMeta {
        token_ids,
        token_count,
    })
}

/// Resolve a turn's per-chunk redo-log locations into per-layer
/// [`StoredSequence`] refs — the substrate's cold tier representation.
///
/// Reads the manifest only (no chunk-payload disk I/O), so this is
/// cheap to call during reload alongside [`recover_turn`]. The pair
/// of calls yields everything the substrate needs to land a restored
/// turn as cold-marker (`hot = None, warm = None, cold = Some(...)`)
/// — classifier-ready for the next `elevate_to_hot` to lift it.
///
/// Returns `None` if the stream has no `Chunk` records (the
/// bg-quantizer crashed before its callback persisted them, or
/// compression was disabled at run time). The reload path treats
/// these as `cold = None` and the substrate leaves the turn empty.
pub fn recover_turn_cold_refs(
    substrate: &Substrate,
    decl: &TurnDecl,
    n_layers: usize,
) -> Result<Option<Vec<StoredSequence>>> {
    let stream_id = super::super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
    let chunks_per_layer = (decl.block_end - decl.block_start) as usize;
    if chunks_per_layer == 0 {
        return Ok(None);
    }

    let Some(stream) = substrate.stream_of(stream_id) else {
        return Ok(None);
    };
    if stream.chunks.is_empty() {
        return Ok(None);
    }
    let expected = n_layers * chunks_per_layer;
    if stream.chunks.len() != expected {
        return Err(PersistenceError::Corrupt(format!(
            "recover_turn_cold_refs: stream {stream_id:?} has {} chunks, expected \
             {n_layers} layers × {chunks_per_layer} chunks = {expected}",
            stream.chunks.len()
        )));
    }

    // Demux flat chunk_index = layer * chunks_per_layer + chunk into
    // per-layer ordered `StoredSequence`s. Matches the layout
    // documented in `flat_chunk_index`.
    let mut per_layer: Vec<Vec<StoredChunk>> = (0..n_layers)
        .map(|_| Vec::with_capacity(chunks_per_layer))
        .collect();
    let mut tokens_per_layer: Vec<usize> = vec![0; n_layers];
    for (&flat_idx, loc) in &stream.chunks {
        let layer = (flat_idx as usize) / chunks_per_layer;
        let chunk = (flat_idx as usize) % chunks_per_layer;
        if layer >= n_layers || chunk >= chunks_per_layer {
            return Err(PersistenceError::Corrupt(format!(
                "recover_turn_cold_refs: chunk_index {flat_idx} out of range \
                 ({n_layers}×{chunks_per_layer})"
            )));
        }
        // Pad to `chunk` index with placeholder, then overwrite.
        let lane = &mut per_layer[layer];
        while lane.len() <= chunk {
            lane.push(StoredChunk {
                log_offset: 0,
                record_len: 0,
                token_count: 0,
            });
        }
        lane[chunk] = StoredChunk {
            log_offset: loc.offset,
            record_len: loc.record_size,
            token_count: loc.token_count as u16,
        };
        tokens_per_layer[layer] += loc.token_count as usize;
    }

    let stored: Vec<StoredSequence> = per_layer
        .into_iter()
        .zip(tokens_per_layer)
        .map(|(chunks, token_count)| StoredSequence {
            chunks,
            token_count,
        })
        .collect();
    Ok(Some(stored))
}

/// Section variant of [`recover_turn_cold_refs`].  Resolves the
/// per-chunk redo-log locations for a content-addressed section
/// stream into per-layer [`StoredSequence`] refs — the cold-tier
/// representation the substrate's residence slab needs.  Used by the
/// scheduler's `RestoreSection` path to install `cold = Some(...)` on
/// the freshly-restored section residence so the persistence thread
/// recognises it as durable.
pub fn recover_section_cold_refs(
    substrate: &Substrate,
    stream_id: StreamId,
    n_layers: usize,
) -> Result<Option<Vec<StoredSequence>>> {
    let Some(stream) = substrate.stream_of(stream_id) else {
        return Ok(None);
    };
    if stream.chunks.is_empty() || n_layers == 0 {
        return Ok(None);
    }
    if stream.chunks.len() % n_layers != 0 {
        return Err(PersistenceError::Corrupt(format!(
            "recover_section_cold_refs: stream {stream_id:?} has {} chunks not divisible by \
             {n_layers} layers",
            stream.chunks.len()
        )));
    }
    let chunks_per_layer = stream.chunks.len() / n_layers;
    let expected = n_layers * chunks_per_layer;
    let mut per_layer: Vec<Vec<StoredChunk>> = (0..n_layers)
        .map(|_| Vec::with_capacity(chunks_per_layer))
        .collect();
    let mut tokens_per_layer: Vec<usize> = vec![0; n_layers];
    for (&flat_idx, loc) in &stream.chunks {
        let layer = (flat_idx as usize) / chunks_per_layer;
        let chunk = (flat_idx as usize) % chunks_per_layer;
        if layer >= n_layers || chunk >= chunks_per_layer {
            return Err(PersistenceError::Corrupt(format!(
                "recover_section_cold_refs: chunk_index {flat_idx} out of range \
                 ({n_layers}×{chunks_per_layer}; expected {expected})"
            )));
        }
        let lane = &mut per_layer[layer];
        while lane.len() <= chunk {
            lane.push(StoredChunk {
                log_offset: 0,
                record_len: 0,
                token_count: 0,
            });
        }
        lane[chunk] = StoredChunk {
            log_offset: loc.offset,
            record_len: loc.record_size,
            token_count: loc.token_count as u16,
        };
        tokens_per_layer[layer] += loc.token_count as usize;
    }
    let stored: Vec<StoredSequence> = per_layer
        .into_iter()
        .zip(tokens_per_layer)
        .map(|(chunks, token_count)| StoredSequence {
            chunks,
            token_count,
        })
        .collect();
    Ok(Some(stored))
}

/// Re-read a persisted section stream's per-layer chunk **windows** —
/// `(offset, token_count)` — straight from the redo log, for
/// [`super::chunk_window_integrity::first_divergent_chunk`].
///
/// [`recover_section_cold_refs`] alone is not enough: its `StoredChunk` only
/// carries what the manifest's header-level location map holds
/// (`log_offset`, `record_len`, `token_count`), and `offset` — the "start of
/// valid data within the chunk" half of the window — lives inside the
/// `Chunk` record's decoded [`ChunkPayload`], not its header. Confirming a
/// divergence against what is actually on disk therefore costs one record
/// read per chunk (`SubstratePersistence::read_chunk`), not just a manifest
/// lookup.
///
/// This runs only when a seal has already failed the live alignment check
/// for a section — off any hot path — so reading every chunk's full payload
/// (rather than a metadata-only decode) trades a few extra bytes per chunk
/// for reusing the one `read_chunk` helper every other cold-read call site
/// already relies on.
///
/// Returns `None` when the stream holds no durable chunks (nothing to
/// confirm either way — matches [`recover_section_cold_refs`]'s `None`).
pub fn read_persisted_section_windows(
    p: &mut SubstratePersistence,
    substrate: &Substrate,
    stream_id: StreamId,
    n_layers: usize,
) -> Result<Option<Vec<Vec<(u16, u16)>>>> {
    let Some(cold_refs) = recover_section_cold_refs(substrate, stream_id, n_layers)? else {
        return Ok(None);
    };
    let chunks_per_layer = cold_refs.first().map(|s| s.chunks.len()).unwrap_or(0);
    let mut per_layer = Vec::with_capacity(cold_refs.len());
    for (layer, seq) in cold_refs.iter().enumerate() {
        let mut windows = Vec::with_capacity(seq.chunks.len());
        for (chunk, stored) in seq.chunks.iter().enumerate() {
            let flat_index = (layer * chunks_per_layer + chunk) as u64;
            let payload = p.read_chunk(substrate, stream_id, flat_index)?;
            windows.push((payload.offset, stored.token_count));
        }
        per_layer.push(windows);
    }
    Ok(Some(per_layer))
}

/// Every declared turn in the recovered index, in `(timeline_id, turn_index)`
/// order — the deterministic replay order a substrate reload walks.
///
/// Each decl's `turn_index` is the position the turn holds in its timeline's
/// stamp order (`docs/replicated_substrate.md` §6.6), which is the index
/// [`turn_stream_id`](super::content_hash::turn_stream_id) lookups resolve
/// through. For a timeline one member wrote it is the index the decl was
/// written with; for two writers' turn *N* it is where the stamps place each.
/// A position no decl holds is a gap the caller fills with a placeholder.
pub fn recovered_turn_decls(substrate: &Substrate) -> Vec<TurnDecl> {
    let mut timelines: Vec<u64> = substrate.timelines_with_turns().collect();
    timelines.sort_unstable();
    timelines
        .into_iter()
        .flat_map(|tl| {
            substrate
                .turn_streams_of(tl)
                .filter_map(|(pos, _, s)| match &s.decl {
                    Some(StreamDecl::Turn(t)) => Some(TurnDecl {
                        turn_index: pos.0,
                        ..t.clone()
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The layer count `L` this substrate's turns were sealed with, read off the
/// turns themselves.
///
/// `L` is a `ModelSpec` constant that the redo log never records: the flat grid
/// stores `chunk_index = layer * chunks_per_layer + chunk`, and only
/// `chunks_per_layer` survives in the `StreamDecl`. So a consistency check has
/// to get `L` from somewhere, and the only honest source is the corpus.
///
/// **Assuming a constant is how a checker cries wolf.** `substrate_inspect
/// validate` hard-coded 48 "for the primary model", but a turn stores one
/// sequence per *KV-bearing* layer — and on the hybrid DeltaNet/attention stack
/// only the attention layers hold KV. Measured against a real 109 GB substrate
/// that mismatch reported **350 of 5,046 turns corrupt**; the true count, once
/// `L` came from the data, was **7**. Every false positive was the same ratio
/// (11 layers read as 48), because it was one wrong constant, not 350 bad
/// turns.
///
/// Taken as the **mode** rather than the first or the maximum: a partially
/// written turn is exactly a turn whose chunk count is wrong, so letting one
/// define `L` would make the corrupt turns validate and the good ones fail.
/// Turns whose chunk count is not a whole multiple of `chunks_per_layer` cannot
/// name a layer count at all and are skipped here — they are the ragged writes
/// the caller is looking for, and they get counted there.
///
/// `None` when no turn carries enough structure to say (an empty substrate, or
/// one where every turn is chunkless).
pub fn derive_layer_count(substrate: &Substrate, decls: &[TurnDecl]) -> Option<usize> {
    let mut votes: HashMap<usize, usize> = HashMap::new();
    for decl in decls {
        let chunks_per_layer = (decl.block_end - decl.block_start) as usize;
        if chunks_per_layer == 0 {
            continue;
        }
        let stream_id =
            super::super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
        let Some(stream) = substrate.stream_of(stream_id) else {
            continue;
        };
        let n = stream.chunks.len();
        if n == 0 || !n.is_multiple_of(chunks_per_layer) {
            continue;
        }
        *votes.entry(n / chunks_per_layer).or_insert(0) += 1;
    }
    // Ties broken toward the larger count: of two layer counts equally
    // represented, the larger one is the one that cannot be produced by a
    // truncated write of the other.
    votes
        .into_iter()
        .max_by_key(|&(layers, count)| (count, layers))
        .map(|(layers, _)| layers)
}
