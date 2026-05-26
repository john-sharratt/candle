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
use super::record::ChunkPayload;
use super::streams::{StreamId, TurnDecl};
use super::{PersistenceError, Result, SubstratePersistence};

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
    decl: &TurnDecl,
    n_layers: usize,
) -> Result<RecoveredTurn> {
    let stream_id = super::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
    let chunks_per_layer = (decl.block_end - decl.block_start) as usize;

    // Snapshot the chunk locations from the manifest (so the immutable
    // borrow ends before the `&mut` reads below).
    let locs: Vec<(u64, ChunkLoc)> = p
        .manifest()
        .streams
        .get(&stream_id)
        .map(|s| s.chunks.iter().map(|(&i, &l)| (i, l)).collect())
        .unwrap_or_default();

    let mut flat: Vec<(u64, ChunkImage)> = Vec::with_capacity(locs.len());
    for (idx, loc) in locs {
        let payload = p.read_chunk(stream_id, idx)?;
        flat.push((
            idx,
            ChunkImage {
                token_count: loc.token_count as u16,
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

    let token_ids = match p.read_tokens(stream_id)? {
        Some(bytes) => decode_token_ids(&bytes)?,
        None => Vec::new(),
    };

    let signatures = match p.read_signatures(stream_id)? {
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

/// Every turn stream in the recovered manifest, in `(timeline_id, turn_index)`
/// order — the deterministic replay order a substrate reload walks.
pub fn recovered_turn_decls(p: &SubstratePersistence) -> Vec<TurnDecl> {
    use super::streams::StreamDecl;
    let mut turns: Vec<TurnDecl> = p
        .manifest()
        .streams
        .values()
        .filter_map(|s| match &s.decl {
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
    use crate::persistence::streams::{PerDepthScores, StreamDecl, TurnDecl};
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
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            let decls = recovered_turn_decls(&sp);
            assert_eq!(decls.len(), 1);
            assert_eq!(decls[0], decl);

            let recovered = recover_turn(&mut sp, &decls[0], n_layers).unwrap();
            assert_eq!(recovered.token_ids, token_ids);
            assert_eq!(recovered.layers, layers);
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
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            // The turn was written with 3 layers; recovering as 4 must fail.
            assert!(recover_turn(&mut sp, &decl, 4).is_err());
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
}
