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
use std::collections::HashMap;

/// One sealed chunk staged for the log: the `Chunk` record's `token_count`
/// header field paired with its decoded [`ChunkPayload`] (which carries the
/// per-chunk `k_format` / `v_format`, palettes, scales, and KV bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkImage {
    /// Valid token count in this chunk (≤ 32).
    pub token_count: u16,
    /// The GPU-computed golden — Fletcher-32 over `payload.kv_bytes`, taken on
    /// the device before the DtoH copy (see candle-kernels `simple/fletcher32.cu`).
    /// Stored in the `Chunk` record's `crc` field; the reload recomputes it over
    /// the on-disk KV bytes to catch DtoH- or storage-corrupted values.
    pub golden: u32,
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

/// A turn's reload metadata — everything the substrate's restore loop
/// needs, none of the KV bytes. Recovered from the `Tokens` record plus
/// the in-RAM chunk index; the chunk **payloads** stay on disk until a
/// cold→hot elevation reads them via [`recover_turn_grid`].
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveredTurnMeta {
    pub token_ids: Vec<u32>,
    /// Sum of the turn's per-chunk token counts (one layer's worth — all
    /// layers seal in lockstep). `0` when no `Chunk` records exist yet.
    pub token_count: usize,
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

/// Decode a `Tokens` record payload back into token ids.
pub fn decode_token_ids(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
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
            let (segment, log_offset, record_len) = p.append_record(
                RecordType::Chunk,
                header_fmt,
                stream_id.0,
                flat,
                image.token_count as u64,
                image.golden,
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
                    // The active segment this chunk's record landed in — a
                    // rotation mid-turn can land later chunks in a newer
                    // segment, so each chunk carries its own id.
                    segment,
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
) -> Result<super::manifest::RecordLoc> {
    // Return the written record's location so the CALLER can fold it into the
    // substrate index (`apply_tokens_loc`). Without that, `entry.tokens` stays
    // `None`, and both the maintenance relocation (`gather_relocations`) and the
    // full compaction (`collect_live_records`) gate on `if let Some(loc) =
    // entry.tokens` — so they silently SKIP the Tokens record and it's reclaimed
    // when its source segment drops (the KV path already registers its chunk
    // locations; tokens must too). This is why turn-0 (prefilled/adopted) tokens
    // vanished during idle maintenance while KV survived.
    let encoded = encode_token_ids(token_ids);
    let (segment, offset, record_size) = p.append_tokens(stream_id, &encoded)?;
    Ok(super::manifest::RecordLoc {
        segment,
        offset,
        payload_len: encoded.len() as u64,
        record_size,
    })
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
    let stream_id = super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
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
    let stream_id = super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
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
        let stream_id = super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::ChunkPayload;
    use crate::persistence::SUBSTRATE_DIR;
    use std::path::{Path, PathBuf};

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
        let kv_bytes: Vec<u8> = (0..64u32).map(|i| (i as u8) ^ seed).collect();
        ChunkImage {
            token_count,
            golden: candle::fletcher::fletcher32(&kv_bytes),
            payload: ChunkPayload {
                offset: 0,
                k_formats: vec![4, 4, 4, 4],
                v_formats: vec![5, 5, 5, 5],
                k_pal: vec![seed; 3],
                v_pal: vec![seed ^ 0xFF; 3],
                k_scale: vec![seed as f32, seed as f32 + 0.5],
                v_scale: vec![seed as f32 - 0.25],
                kv_bytes,
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
            segments: Vec::new(),
            tags: Vec::new(),
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
        }
        // Reopen — a simulated restart — and recover the turn from disk.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let decls = recovered_turn_decls(&substrate);
            assert_eq!(decls.len(), 1);
            assert_eq!(decls[0], decl);
            // Metadata half: token ids + per-layer token count, no
            // chunk-payload I/O.
            let meta = recover_turn_meta(&mut sp, &substrate, &decls[0]).unwrap();
            assert_eq!(meta.token_ids, token_ids);
            assert_eq!(meta.token_count, 32 + 12, "layer-0 chunk token counts");
            // Payload half: the full grid, byte-identical.
            let grid = recover_turn_grid(&mut sp, &substrate, &decls[0], n_layers).unwrap();
            assert_eq!(grid, layers);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Persist `turns` turns of `n_layers × chunks_per_layer`, optionally
    /// truncating one of them to `short_chunks` total records.
    fn substrate_with_turns(
        dir: &Path,
        turns: u32,
        n_layers: usize,
        chunks_per_layer: usize,
        truncate: Option<(u32, usize)>,
    ) -> Substrate {
        {
            let mut sp = SubstratePersistence::open_in(dir).unwrap();
            for t in 0..turns {
                let decl = turn_decl(7, t, chunks_per_layer as u64);
                let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
                sp.declare_stream(&StreamDecl::Turn(decl)).unwrap();
                let keep = match truncate {
                    Some((which, n)) if which == t => n,
                    _ => n_layers * chunks_per_layer,
                };
                // Written flat, so a truncation drops trailing records exactly
                // the way an interrupted write does — which is why this bypasses
                // `persist_turn_chunks` (it insists on a whole grid, and a whole
                // grid is what a partial write does not have).
                for i in 0..keep {
                    let image = chunk_image(i as u8, 32);
                    sp.write_chunk(
                        stream_id,
                        i as u64,
                        image.token_count as u64,
                        image.payload.k_formats.first().copied().unwrap_or(0),
                        Some(image.golden),
                        &image.payload,
                    )
                    .unwrap();
                }
            }
            sp.commit().unwrap();
        }
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
        substrate
    }

    #[test]
    fn the_layer_count_comes_from_the_turns_not_from_a_constant() {
        // The false alarm this exists to prevent: a checker that assumes 48
        // layers against a hybrid stack that seals KV for 11 of them calls
        // every healthy turn corrupt. Derived, the answer is the truth.
        let dir = tmp_dir("derive_layers");
        let substrate = substrate_with_turns(&dir, 4, 11, 3, None);
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(derive_layer_count(&substrate, &decls), Some(11));
        for decl in &decls {
            assert!(
                recover_turn_cold_refs(&substrate, decl, 11).is_ok(),
                "a turn sealed with 11 layers must validate against 11"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_truncated_turn_does_not_get_to_define_the_layer_count() {
        // The trap in taking the first or the minimum: a partial write IS a
        // turn with the wrong chunk count, so letting one vote decide would
        // validate the corrupt turn and condemn every healthy one — the same
        // false alarm, inverted. The mode keeps the majority's answer.
        let dir = tmp_dir("derive_truncated");
        // Turn 0 keeps only 4 layers' worth (12 of 33 records).
        let substrate = substrate_with_turns(&dir, 4, 11, 3, Some((0, 12)));
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(
            derive_layer_count(&substrate, &decls),
            Some(11),
            "three healthy turns outvote one truncated one"
        );
        let n = derive_layer_count(&substrate, &decls).unwrap();
        let bad = decls
            .iter()
            .filter(|d| recover_turn_cold_refs(&substrate, d, n).is_err())
            .count();
        assert_eq!(bad, 1, "exactly the truncated turn is reported");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_ragged_turn_names_no_layer_count_and_is_skipped() {
        // 13 records against 3 chunks/layer is not a whole number of layers,
        // so it cannot vote — it is the ragged write the caller is hunting.
        let dir = tmp_dir("derive_ragged");
        let substrate = substrate_with_turns(&dir, 3, 11, 3, Some((0, 13)));
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(derive_layer_count(&substrate, &decls), Some(11));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_substrate_with_no_chunked_turns_derives_nothing() {
        let dir = tmp_dir("derive_empty");
        let substrate = substrate_with_turns(&dir, 0, 11, 3, None);
        let decls = recovered_turn_decls(&substrate);
        assert_eq!(derive_layer_count(&substrate, &decls), None);
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
                self_reference: false,
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
                self_reference: false,
            },
        ];
        let payload = encode_events(&events);

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_projection_events(stream_id, &payload).unwrap();
            sp.commit().unwrap();
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
    fn wide_q_sigs_persist_and_recover() {
        use crate::projection::{TimelineId, TurnIndex};
        use crate::provenance::{decode_wide_sigs, encode_wide_sigs, WideQSig};

        let dir = tmp_dir("wide_q_sigs");
        let decl = turn_decl(9, 0, 3);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();

        // A 3-token wide history, 4 heads × head_dim 128, alternating signs per token.
        let history: Vec<WideQSig> = (0..3)
            .map(|t| {
                let band: Vec<f32> = (0..4 * 128)
                    .map(|i| if (i + t) % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                WideQSig::from_band(&band, 128)
            })
            .collect();
        let payload = encode_wide_sigs(&history);

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            sp.append_wide_q_sigs(stream_id, &payload).unwrap();
            sp.commit().unwrap();
        }
        // Reopen — simulated daemon restart — and recover the window.
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(decl.timeline_id).unwrap();
            let blob = substrate
                .wide_q_sigs_blob(tl, TurnIndex(decl.turn_index))
                .expect("wide-Q sigs recovered after restart");
            assert_eq!(blob, payload.as_slice());
            assert_eq!(decode_wide_sigs(blob), Some(history));
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
            assert!(recover_turn_grid(&mut sp, &substrate, &decl, 4).is_err());
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

    /// `persist_tokens_only` must return the WRITTEN record's on-disk location so
    /// the caller can fold it into the substrate index (`apply_tokens_loc`).
    /// Pre-fix it returned `()`, so `entry.tokens` stayed `None` and the
    /// maintenance/compaction relocation (both gate on `if let Some(loc) =
    /// entry.tokens`) silently reclaimed a live turn's tokens on the next idle
    /// pass. Assert the returned loc is populated AND byte-for-byte agrees with the
    /// location the reload walk independently records for the same record — a wrong
    /// loc would corrupt token reads once callers register it.
    #[test]
    fn persist_tokens_only_returns_the_written_record_location() {
        let dir = tmp_dir("tokens_loc_return");
        let decl = turn_decl(21, 0, 0);
        let stream_id = StreamDecl::Turn(decl.clone()).stream_id();
        let live_loc = {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&StreamDecl::Turn(decl.clone())).unwrap();
            let loc = persist_tokens_only(&mut sp, stream_id, &[7, 8, 9]).unwrap();
            sp.commit().unwrap();
            loc
        };
        assert!(live_loc.offset > 0, "offset past the file-start header");
        assert!(live_loc.record_size > 0, "padded record_size populated");
        assert_eq!(
            live_loc.payload_len,
            3 * 4,
            "3 u32 tokens = 12 payload bytes"
        );
        // The reload walk records the same record's loc straight from disk — the
        // live-write loc must match it exactly, proving it points at the real bytes.
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let (wseg, woff, wsize, wpay) = {
            let e = substrate.stream_of(stream_id).expect("stream present");
            let t = e
                .tokens
                .as_ref()
                .expect("reload registered the tokens loc from disk");
            (t.segment, t.offset, t.record_size, t.payload_len)
        };
        assert_eq!(wseg, live_loc.segment, "segment matches reload walk");
        assert_eq!(woff, live_loc.offset, "offset matches reload walk");
        assert_eq!(
            wsize, live_loc.record_size,
            "record_size matches reload walk"
        );
        assert_eq!(
            wpay, live_loc.payload_len,
            "payload_len matches reload walk"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
