//! Turn-resume reconstruction — persisting and recovering a turn's KV from
//! the redo log (§5.6, §5.7 of `docs/kv_tier_migration.md`).
//!
//! A turn's KV is, in the substrate, an `L × C` grid of `SealedChunk`s — one
//! `SealedSequence` per model layer (`L`), each a `C`-long chunk list. The
//! redo log addresses a `Chunk` only by `(stream_id, chunk_index)`, with no
//! layer axis, so the grid is flattened onto the stream's 1-D chunk index
//! (the **Option A** layout):
//!
//! ```text
//!   chunk_index = layer * chunks_per_layer + chunk
//! ```
//!
//! `chunks_per_layer` (`C`) is recoverable from the turn's `StreamDecl`
//! (`block_end - block_start`); `L` is a `ModelSpec` constant. Each
//! `Chunk` record holds exactly one `SealedChunk`'s worth of metadata and
//! bytes — the [`ChunkPayload`] codec is unchanged from P5.
//!
//! This module is the **device-free** half of resume: the record layout,
//! the flat-grid demux, and the `Tokens` codec. Gathering the KV bytes off
//! the GPU at seal time and scattering them back into VRAM on resume lives
//! in `transfer.rs` — `candle-conversation` is CUDA-only so there is no
//! non-CUDA half.

use super::manifest::ChunkLoc;
use super::record::{ChunkPayload, RecordType};
use super::streams::{StreamDecl, StreamId, TurnDecl};
use super::{PersistenceError, Result, SubstratePersistence};
use crate::substrate::{StoredChunk, StoredSequence, Substrate};

/// One sealed chunk staged for the log: the `Chunk` record's `token_count`
/// header field paired with its decoded [`ChunkPayload`] (which carries the
/// per-chunk `k_format` / `v_format`, palettes, scales, and KV bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkImage {
    /// Valid token count in this chunk (≤ 32).
    pub token_count: u16,
    /// Per-chunk K/V formats, palettes, scales, and gathered KV bytes.
    pub payload: ChunkPayload,
}

/// A turn's KV chunks laid out as a per-layer grid.
///
/// `grid.layer(L)[c]` is the c-th chunk of layer L. All layers share one
/// chunk count — sealing is in lockstep across layers. The empty grid
/// (zero layers, or all layers empty) is a legal "no chunks recovered"
/// sentinel returned by [`recover_turn`] when a turn's `Tokens` are
/// durable but no `Chunks` records exist yet (e.g. the bg-quantizer
/// persist callback hadn't fired before shutdown).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TurnChunkGrid {
    layers: Vec<Vec<ChunkImage>>,
}

impl TurnChunkGrid {
    /// Wrap an existing per-layer chunk list.
    pub fn new(layers: Vec<Vec<ChunkImage>>) -> Self {
        Self { layers }
    }

    /// Empty grid, with capacity for `n_layers` layers but no chunks yet —
    /// matches the seal-callback build-up loop (`push_layer` per backing).
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            layers: Vec::with_capacity(n_layers),
        }
    }

    /// `n_layers` empty layers — the "no chunk records" sentinel a cold
    /// reload uses when `Tokens` are durable but `Chunks` are not.
    pub fn empty_grid(n_layers: usize) -> Self {
        Self {
            layers: vec![Vec::new(); n_layers],
        }
    }

    /// Number of layers in the grid.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Chunk count per layer (all layers share one count under
    /// lockstep sealing). Zero for an empty grid.
    pub fn chunks_per_layer(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len())
    }

    /// True when every layer has zero chunks (the cold-marker case).
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|l| l.is_empty())
    }

    /// Borrow the c-th layer's chunk list.
    pub fn layer(&self, idx: usize) -> &[ChunkImage] {
        &self.layers[idx]
    }

    /// Iterate `&[ChunkImage]` per layer — the form `load_to_hot` zips
    /// against the per-layer `ChunkedKvBacking`s.
    pub fn iter_layers(&self) -> impl ExactSizeIterator<Item = &[ChunkImage]> {
        self.layers.iter().map(|v| v.as_slice())
    }

    /// Append a layer's chunks to the grid. Used by the seal callback to
    /// build the grid one backing at a time.
    pub fn push_layer(&mut self, chunks: Vec<ChunkImage>) {
        self.layers.push(chunks);
    }

    /// Total KV byte footprint — sum of every chunk's `kv_bytes`. This is
    /// the warm-pool LRU budget unit and the cold-load pre-flight's
    /// `incoming` size.
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.iter())
            .map(|i| i.payload.kv_bytes.len())
            .sum()
    }
}

/// A turn fully recovered from the redo log — its declaration, token ids,
/// and the per-layer sealed-chunk grid.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveredTurn {
    pub decl: TurnDecl,
    pub token_ids: Vec<u32>,
    /// Per-layer ordered chunks for the turn. See [`TurnChunkGrid`].
    pub layers: TurnChunkGrid,
    /// Per-chunk BDP provenance signatures `(token_count, syn‖sem‖prag bytes)`,
    /// in chunk order — empty if the turn has no `Signatures` record.
    pub signatures: Vec<(u16, Vec<u8>)>,
}

/// Flat 1-D chunk index for `(layer, chunk)` under the Option A layout.
#[inline]
pub fn flat_chunk_index(layer: usize, chunk: usize, chunks_per_layer: usize) -> u64 {
    (layer * chunks_per_layer + chunk) as u64
}

/// Encode a turn's token ids as the `Tokens` record payload — little-endian
/// `u32`s, the inverse of [`decode_token_ids`].
pub fn encode_token_ids(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

/// Encode a turn's per-chunk BDP signatures as the `Signatures` record
/// payload. Each entry is `(token_count, raw bytes)` — the bytes are the
/// concatenated syn/sem/prag `TokenSignature`s read from the provenance
/// file (`token_count * 48` bytes). Inverse of [`decode_signatures`].
pub fn encode_signatures(entries: &[(u16, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (token_count, bytes) in entries {
        out.extend_from_slice(&token_count.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Decode a `Signatures` record payload into `(token_count, raw bytes)`
/// entries — one per sealed chunk, in chunk order.
pub fn decode_signatures(payload: &[u8]) -> Result<Vec<(u16, Vec<u8>)>> {
    let mut pos = 0usize;
    let take = |p: &mut usize, n: usize| -> Result<&[u8]> {
        if *p + n > payload.len() {
            return Err(PersistenceError::Truncated {
                need: n,
                have: payload.len().saturating_sub(*p),
            });
        }
        let s = &payload[*p..*p + n];
        *p += n;
        Ok(s)
    };
    let n_entries = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n_entries);
    for _ in 0..n_entries {
        let token_count = u16::from_le_bytes(take(&mut pos, 2)?.try_into().unwrap());
        let n_bytes = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        out.push((token_count, take(&mut pos, n_bytes)?.to_vec()));
    }
    if pos != payload.len() {
        return Err(PersistenceError::Corrupt(format!(
            "Signatures payload has {} trailing bytes",
            payload.len() - pos
        )));
    }
    Ok(out)
}

/// Decode a `Tokens` record payload back into token ids.
pub fn decode_token_ids(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return Err(PersistenceError::Corrupt(format!(
            "Tokens payload is {} bytes, not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

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
/// Also returns the `(flat_chunk_index, ChunkLoc)` pairs the caller must
/// fold into the substrate's authoritative chunk index (`apply_chunk_loc`).
/// Without that, `stream.chunks` is only repopulated by the walker on reload,
/// so an in-process cold→hot elevation (a turn evicted then re-accessed within
/// the same session) would `plan_chunked_read` an empty index and recover
/// nothing.
pub fn persist_turn_chunks_capture(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    layers: &TurnChunkGrid,
) -> Result<(Vec<StoredSequence>, Vec<(u64, ChunkLoc)>)> {
    let chunks_per_layer = layers.chunks_per_layer();
    let mut out = Vec::with_capacity(layers.n_layers());
    let mut locs: Vec<(u64, ChunkLoc)> = Vec::new();
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
            let (log_offset, record_len) = p.append_record(
                RecordType::Chunk,
                header_fmt,
                stream_id.0,
                flat,
                image.token_count as u64,
                &encoded,
            )?;
            stored.push(StoredChunk {
                log_offset,
                record_len,
                token_count: image.token_count,
            });
            locs.push((
                flat,
                ChunkLoc {
                    offset: log_offset,
                    payload_len,
                    record_size: record_len,
                    token_count: image.token_count as u64,
                    format: header_fmt,
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
) -> Result<()> {
    p.append_tokens(stream_id, &encode_token_ids(token_ids))?;
    Ok(())
}

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

/// Recover a single turn fully from the log: its `Tokens` and the demuxed
/// `L × C` chunk grid. `decl` is the turn's already-decoded `StreamDecl`;
/// `n_layers` is the running model's layer count.
pub fn recover_turn(
    p: &mut SubstratePersistence,
    substrate: &Substrate,
    decl: &TurnDecl,
    n_layers: usize,
) -> Result<RecoveredTurn> {
    let stream_id = super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
    let chunks_per_layer = (decl.block_end - decl.block_start) as usize;

    // Snapshot per-chunk token_count from the substrate's stream
    // index before the `&mut p` read below — we need it to assemble
    // `ChunkImage`s once the batched read returns.  The chunk index →
    // token_count map is small (one u64 per chunk), so cloning is
    // cheap.
    use std::collections::HashMap;
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
        flat.push((
            idx,
            ChunkImage {
                token_count,
                payload,
            },
        ));
    }
    // A turn whose chunks haven't landed yet — either because the bg-quantizer
    // persist callback hadn't fired before a crash, or because the run was
    // configured with `compression_policy = None` — is still valid: its
    // tokens and signatures are preserved and the layer grid is reconstructed
    // empty. The substrate's reload path handles empty sealed sequences.
    let layers = if flat.is_empty() {
        TurnChunkGrid::empty_grid(n_layers)
    } else {
        demux_layers(flat, n_layers, chunks_per_layer)?
    };

    let token_ids = match p.read_tokens(substrate, stream_id)? {
        Some(bytes) => decode_token_ids(&bytes)?,
        None => Vec::new(),
    };

    let signatures = match p.read_signatures(substrate, stream_id)? {
        Some(bytes) => decode_signatures(&bytes)?,
        None => Vec::new(),
    };

    Ok(RecoveredTurn {
        decl: decl.clone(),
        token_ids,
        layers,
        signatures,
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
    let stream_id = super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
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

/// Every turn stream in the recovered manifest, in `(timeline_id, turn_index)`
/// order — the deterministic replay order a substrate reload walks.
pub fn recovered_turn_decls(substrate: &Substrate) -> Vec<TurnDecl> {
    let mut turns: Vec<TurnDecl> = substrate
        .all_streams()
        .filter_map(|(_, s)| match &s.decl {
            Some(StreamDecl::Turn(t)) => Some(t.clone()),
            _ => None,
        })
        .collect();
    turns.sort_by_key(|t| (t.timeline_id, t.turn_index));
    turns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::ChunkPayload;
    use crate::persistence::streams::PerDepthScores;
    use crate::persistence::SUBSTRATE_DIR;
    use std::path::PathBuf;

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_resume_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn chunk_image(seed: u8, token_count: u16) -> ChunkImage {
        ChunkImage {
            token_count,
            payload: ChunkPayload {
                offset: 0,
                k_formats: vec![4, 4, 4, 4],
                v_formats: vec![5, 5, 5, 5],
                k_pal: vec![seed; 3],
                v_pal: vec![seed ^ 0xFF; 3],
                k_scale: vec![seed as f32, seed as f32 + 0.5],
                v_scale: vec![seed as f32 - 0.25],
                kv_bytes: (0..64u32).map(|i| (i as u8) ^ seed).collect(),
            },
        }
    }

    fn turn_decl(timeline: u64, turn: u32, blocks: u64) -> TurnDecl {
        TurnDecl {
            timeline_id: timeline,
            turn_index: turn,
            turn_id_day: 0,
            turn_id_seq: turn + 1,
            role: 1,
            block_start: 0,
            block_end: blocks,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            scores: PerDepthScores::default(),
            segments: Vec::new(),
        }
    }

    #[test]
    fn flat_index_round_trips_through_demux() {
        // 3 layers × 4 chunks: every (layer, chunk) maps to a distinct flat
        // index and demux puts it back.
        let (n_layers, c) = (3usize, 4usize);
        let mut flat = Vec::new();
        for layer in 0..n_layers {
            for chunk in 0..c {
                let idx = flat_chunk_index(layer, chunk, c);
                flat.push((idx, chunk_image((layer * c + chunk) as u8, 32)));
            }
        }
        let layers = demux_layers(flat, n_layers, c).unwrap();
        assert_eq!(layers.n_layers(), n_layers);
        for (layer, row) in layers.iter_layers().enumerate() {
            assert_eq!(row.len(), c);
            for (chunk, image) in row.iter().enumerate() {
                assert_eq!(image, &chunk_image((layer * c + chunk) as u8, 32));
            }
        }
    }

    #[test]
    fn demux_rejects_a_missing_chunk() {
        let mut flat = vec![(0u64, chunk_image(0, 32)), (1, chunk_image(1, 32))];
        // 2 layers × 2 = 4 expected, only 2 present.
        assert!(demux_layers(std::mem::take(&mut flat), 2, 2).is_err());
    }

    #[test]
    fn token_ids_round_trip() {
        let ids = vec![1u32, 65535, 151_935, 0, 7];
        assert_eq!(decode_token_ids(&encode_token_ids(&ids)).unwrap(), ids);
    }

    #[test]
    fn decode_token_ids_rejects_a_ragged_payload() {
        assert!(decode_token_ids(&[0u8, 1, 2]).is_err());
    }

    #[test]
    fn signatures_round_trip() {
        let entries = vec![
            (32u16, (0..32u32 * 48).map(|i| i as u8).collect::<Vec<u8>>()),
            (12u16, vec![0xABu8; 12 * 48]),
            (0u16, Vec::new()),
        ];
        let bytes = encode_signatures(&entries);
        assert_eq!(decode_signatures(&bytes).unwrap(), entries);
    }

    #[test]
    fn decode_signatures_rejects_truncation() {
        let bytes = encode_signatures(&[(4u16, vec![1u8; 4 * 48])]);
        assert!(decode_signatures(&bytes[..bytes.len() - 3]).is_err());
    }

    #[test]
    fn persist_then_recover_a_turn_round_trips() {
        let dir = tmp_dir("roundtrip");
        let decl = turn_decl(7, 0, 2); // 2 chunks per layer
        let n_layers = 3usize;
        let token_ids = vec![10u32, 20, 30, 40];

        // Build an L×C grid with unique per-(layer,chunk) seeds.
        let layers = TurnChunkGrid::new(
            (0..n_layers)
                .map(|l| {
                    (0..2)
                        .map(|c| chunk_image((l * 10 + c) as u8, if c == 1 { 12 } else { 32 }))
                        .collect()
                })
                .collect(),
        );

        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            persist_turn_kv(&mut sp, stream_id, &layers, &token_ids).unwrap();
            sp.commit().unwrap();
            sp.checkpoint().unwrap();
        }
        // Reopen — a simulated restart — and recover the turn from disk.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let decls = recovered_turn_decls(&substrate);
            assert_eq!(decls.len(), 1);
            assert_eq!(decls[0], decl);
            let recovered = recover_turn(&mut sp, &substrate, &decls[0], n_layers).unwrap();
            assert_eq!(recovered.token_ids, token_ids);
            assert_eq!(recovered.layers, layers);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn projection_events_persist_and_recover() {
        use crate::projection::{
            decode_events, encode_events, BucketKind, ProjectionBucket, ProjectionEvent,
            TimelineId, TurnIndex,
        };
        let dir = tmp_dir("proj_events");
        let decl = turn_decl(7, 0, 2);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();

        // A two-event timeline for the turn.
        let events = vec![
            ProjectionEvent {
                start_token: 0,
                seconds: 3.0,
                materialized_tokens: 1120,
                substrate_tokens: 42_000,
                buckets: vec![
                    ProjectionBucket {
                        label: "system".into(),
                        kind: BucketKind::System,
                        tokens: 320,
                    },
                    ProjectionBucket {
                        label: "code_read".into(),
                        kind: BucketKind::Section,
                        tokens: 800,
                    },
                ],
                selection: Default::default(),
                materialized: Default::default(),
            },
            ProjectionEvent {
                start_token: 120,
                seconds: 8.0,
                materialized_tokens: 540,
                substrate_tokens: 42_000,
                buckets: vec![ProjectionBucket {
                    label: "conversation".into(),
                    kind: BucketKind::Turns,
                    tokens: 540,
                }],
                selection: Default::default(),
                materialized: Default::default(),
            },
        ];
        let payload = encode_events(&events);

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_projection_events(stream_id, &payload).unwrap();
            sp.commit().unwrap();
            sp.checkpoint().unwrap();
        }
        // Reopen — a simulated daemon restart — and recover the timeline.
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(decl.timeline_id).unwrap();
            let blob = substrate
                .projection_events_blob(tl, TurnIndex(decl.turn_index))
                .expect("projection events recovered after restart");
            assert_eq!(blob, payload.as_slice());
            assert_eq!(decode_events(blob), events);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recover_turn_demux_fails_on_wrong_layer_count() {
        let dir = tmp_dir("wrong_layers");
        let decl = turn_decl(9, 0, 2);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        let layers = TurnChunkGrid::new(
            (0..3)
                .map(|l| (0..2).map(|c| chunk_image((l * 5 + c) as u8, 32)).collect())
                .collect(),
        );
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            persist_turn_kv(&mut sp, stream_id, &layers, &[1, 2]).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // The turn was written with 3 layers; recovering as 4 must fail.
            assert!(recover_turn(&mut sp, &substrate, &decl, 4).is_err());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_substrate_dir_is_used() {
        // Guards the doc-comment claim that the log lives under .substrate/.
        let dir = tmp_dir("dir_check");
        SubstratePersistence::open_in(&dir).unwrap();
        assert!(dir.join(SUBSTRATE_DIR).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// After a persist+reopen, `recover_turn_cold_refs` returns one
    /// `StoredSequence` per layer with the per-chunk `log_offset` /
    /// `record_len` / `token_count` populated from the on-disk
    /// manifest entries — the data the substrate's reload path needs
    /// to install the cold tier without reading any chunk payloads.
    #[test]
    fn recover_cold_refs_demuxes_manifest_chunks_per_layer() {
        let dir = tmp_dir("cold_refs");
        let n_layers = 3usize;
        let chunks_per_layer = 2usize;
        let decl = turn_decl(11, 0, chunks_per_layer as u64);
        let layers = TurnChunkGrid::new(
            (0..n_layers)
                .map(|l| {
                    (0..chunks_per_layer)
                        .map(|c| chunk_image((l * 10 + c) as u8, if c == 1 { 12 } else { 32 }))
                        .collect()
                })
                .collect(),
        );

        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            persist_turn_kv(&mut sp, stream_id, &layers, &[1, 2, 3]).unwrap();
            sp.commit().unwrap();
            sp.checkpoint().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let cold = recover_turn_cold_refs(&substrate, &decl, n_layers)
                .unwrap()
                .expect("Some(...) when chunks exist");
            assert_eq!(cold.len(), n_layers, "one StoredSequence per layer");
            for (layer, seq) in cold.iter().enumerate() {
                assert_eq!(seq.chunks.len(), chunks_per_layer);
                // Token counts demuxed back into the right layer slots.
                assert_eq!(seq.chunks[0].token_count, 32);
                assert_eq!(seq.chunks[1].token_count, 12);
                assert_eq!(seq.token_count, 32 + 12);
                // `log_offset` must be the actual on-disk record
                // offset — past zero (header lands at the start of
                // the file) and monotonically increasing within the
                // layer.
                let _ = layer;
                assert!(seq.chunks[0].log_offset > 0, "log_offset populated");
                assert!(seq.chunks[1].record_len > 0, "record_len populated");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A turn with no `Chunk` records (e.g. crash before bg-quantizer
    /// callback persisted them) returns `None` — the reload path uses
    /// this to leave `cold = None` on the residence so the classifier
    /// reports it `missing` instead of attempting a bogus cold→hot.
    #[test]
    fn recover_cold_refs_returns_none_when_no_chunks() {
        let dir = tmp_dir("cold_refs_empty");
        let decl = turn_decl(13, 0, 0);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            // No persist_turn_kv — stream declared but no Chunk records.
            persist_tokens_only(&mut sp, stream_id, &[7, 8]).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let cold = recover_turn_cold_refs(&substrate, &decl, 2).unwrap();
            assert!(cold.is_none(), "no chunks → None");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
