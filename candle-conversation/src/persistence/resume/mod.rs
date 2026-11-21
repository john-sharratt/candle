//! Persists and recovers a turn's KV chunks from the redo log, so that a turn survives a restart.
//!
//! A turn is a grid of sealed chunks, one sequence per layer. It is flattened onto its stream's one-dimensional
//! chunk index as the layer times the chunks per layer plus the chunk. The record codec is the device-free half of
//! resume: the flat layout, its demux back into the grid, and the token codec. Gathering the bytes off the GPU and
//! scattering them back is done in a separate transfer module, since this crate is CUDA-only.
//!
//! - `persist`: writes a turn's chunk, token and commit records.
//! - `recover`: reads a turn's metadata, grid and cold references back.
//! - `tests_resume`: tests of persistence, recovery and tombstones across restarts.

use super::identity::Stamped;
use super::manifest::{ChunkLoc, RecordLoc};
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

mod persist;
mod recover;
#[cfg(test)]
mod tests_resume;

pub use persist::*;
pub use recover::*;
