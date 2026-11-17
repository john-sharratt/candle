//! One layer's part of a warm lift: its sources resolved and its destinations
//! claimed before any byte moves, and its sealed sequences rebuilt at the new
//! placement after.
//!
//! A lift carries every band of every item, ~10^5 at a working set's size, so
//! the per-band work here is an index into one table: one hash of a band's gid
//! when it is first met, the arena key from a cache by arena index, and every
//! destination claimed in two batched calls — not a claim, a hash set and a
//! vector per palette run.

use std::time::Instant;

use ahash::AHashMap;
use candle::Result;

use super::micros;
use super::plan::LiftBand;
use crate::kv_cache::arena_table::{ResolvedArenaInfo, N_PALETTE};
use crate::kv_cache::chunked::gid_pool::ChunkGid;
use crate::kv_cache::chunked::head_gids::HeadGids;
use crate::kv_cache::chunked::meta_pool::ChunkRecordSrc;
use crate::kv_cache::chunked::size_class::payload_bytes_for_tag;
use crate::kv_cache::chunked::types::{SealedChunk, SealedSequence};
use crate::kv_cache::chunked::{ArenaKey, ChunkedKvBacking, GID_STRIDE};
use crate::kv_cache::ArenaLocation;

/// A layer's lift, planned: its bands in source order, and each source gid's
/// destination — `dests[index[raw]]`, the band at the same position.
pub(crate) struct LayerLift {
    pub bands: Vec<LiftBand>,
    pub index: AHashMap<i64, usize>,
    pub dests: Vec<ChunkGid>,
    /// Microseconds spent resolving sources, claiming destinations, and
    /// resolving their addresses.
    pub resolve_us: u64,
    pub claim_us: u64,
    pub place_us: u64,
}

/// A unique source band: its gid, its warm arena's key, its payload bytes.
struct Source {
    raw: i64,
    key: ArenaKey,
    len: usize,
}

impl ChunkedKvBacking {
    /// Resolve `sequences`' warm source bands — each unique gid once, its
    /// length from its own tag — and claim a destination slot for each, in one
    /// state write: a chunk's palette bands as one contiguous run where they are
    /// distinct and share one class, singly otherwise.
    pub(crate) fn plan_warm_lift(
        &self,
        layer: usize,
        sequences: &[&SealedSequence],
    ) -> Result<LayerLift> {
        let elems = self.inner.elems_per_chunk();
        let t = Instant::now();
        // Position in `sources` of each unique gid.
        let mut index: AHashMap<i64, usize> = AHashMap::new();
        let mut sources: Vec<Source> = Vec::new();
        self.inner.storage.read(|s| -> Result<()> {
            // Warm arenas by index, each looked up once.
            let mut keys: Vec<Option<ArenaKey>> = Vec::new();
            for seq in sequences {
                for chunk in &seq.chunks {
                    for (gid, tag) in chunk.bands() {
                        let raw = gid.raw();
                        if index.contains_key(&raw) {
                            continue;
                        }
                        let arena_idx = (raw as usize) / GID_STRIDE;
                        if keys.len() <= arena_idx {
                            keys.resize(arena_idx + 1, None);
                        }
                        let key = match keys[arena_idx] {
                            Some(key) => key,
                            None => {
                                let key = s.arena_key(arena_idx).ok_or_else(|| {
                                    candle::Error::Msg(format!(
                                        "warm lift: arena {arena_idx} not found"
                                    ))
                                })?;
                                if key.location != ArenaLocation::Cpu {
                                    candle::bail!(
                                        "warm lift: source arena {arena_idx} is not CPU-resident"
                                    );
                                }
                                keys[arena_idx] = Some(key);
                                key
                            }
                        };
                        // The warm arena is a class-strided byte slab with no
                        // format to ask; the band's own tag says its length.
                        let len = payload_bytes_for_tag(tag, elems).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "warm lift: band tag {tag:?} names no storage format, so its \
                                 byte length is unknown"
                            ))
                        })?;
                        index.insert(raw, sources.len());
                        sources.push(Source { raw, key, len });
                    }
                }
            }
            Ok(())
        })??;
        let resolve_us = micros(t);

        let t = Instant::now();
        let mut dests: Vec<Option<ChunkGid>> = vec![None; sources.len()];
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            // Palette runs: a chunk side's bands, distinct, all of one class,
            // none already given a run.
            let mut taken = vec![false; sources.len()];
            let mut run_keys: Vec<(ArenaKey, usize)> = Vec::new();
            let mut run_members: Vec<[usize; N_PALETTE]> = Vec::new();
            let n_kv_head = self.inner.n_kv_head;
            for seq in sequences {
                for chunk in &seq.chunks {
                    for h in 0..n_kv_head {
                        for side_v in [false, true] {
                            let mut at = [0usize; N_PALETTE];
                            let mut whole = true;
                            for (p, slot) in at.iter_mut().enumerate() {
                                let raw = match side_v {
                                    true => chunk.gids.v_gid_pal(h, p).raw(),
                                    false => chunk.gids.k_gid_pal(h, p).raw(),
                                };
                                match index.get(&raw) {
                                    Some(&i) if raw >= 0 && !taken[i] => *slot = i,
                                    _ => {
                                        whole = false;
                                        break;
                                    }
                                }
                            }
                            if !whole {
                                continue;
                            }
                            let distinct = (1..N_PALETTE).all(|p| !at[..p].contains(&at[p]));
                            let class = sources[at[0]].key.class;
                            if !distinct || at[1..].iter().any(|&i| sources[i].key.class != class) {
                                continue;
                            }
                            for &i in &at {
                                taken[i] = true;
                            }
                            run_keys.push((sources[at[0]].key.at(ArenaLocation::Gpu), N_PALETTE));
                            run_members.push(at);
                        }
                    }
                }
            }
            let runs = self.inner.alloc_chunk_runs_for_keys(&run_keys)?;
            for (members, run) in run_members.iter().zip(runs) {
                for (&i, gid) in members.iter().zip(run) {
                    dests[i] = Some(gid);
                }
            }
            // The rest singly, one bulk claim per destination key.
            let mut singles: Vec<(ArenaKey, Vec<usize>)> = Vec::new();
            for (i, source) in sources.iter().enumerate() {
                if taken[i] {
                    continue;
                }
                let key = source.key.at(ArenaLocation::Gpu);
                match singles.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, members)) => members.push(i),
                    None => singles.push((key, vec![i])),
                }
            }
            for (key, members) in singles {
                let gids = self.alloc_chunks_for_key_bulk(key, members.len())?;
                for (i, gid) in members.into_iter().zip(gids) {
                    dests[i] = Some(gid);
                }
            }
        }
        let claim_us = micros(t);

        let t = Instant::now();
        let info = self.resolve_arena_info()?;
        let dests: Vec<ChunkGid> = sources
            .iter()
            .zip(dests)
            .map(|(source, gid)| {
                gid.ok_or_else(|| {
                    candle::Error::Msg(format!("warm lift: band {} was given no slot", source.raw))
                })
            })
            .collect::<Result<_>>()?;
        let bands: Vec<LiftBand> = sources
            .iter()
            .zip(&dests)
            .map(|(source, gid)| {
                Ok(LiftBand {
                    layer,
                    src_arena: (source.raw as usize) / GID_STRIDE,
                    src_chunk: (source.raw as usize) % GID_STRIDE,
                    bytes: source.len,
                    dst: destination(&info, gid)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(LayerLift {
            bands,
            index,
            dests,
            resolve_us,
            claim_us,
            place_us: micros(t),
        })
    }

    /// `sequences` at their lifted placement: every gid mapped to its new GPU
    /// slot, every chunk's resident KV-head record built there — one batch for
    /// the layer — and every other field, formats included, carried unchanged,
    /// since a lift copies bytes verbatim into a same-format slot.
    pub(crate) fn finish_warm_lift(
        &self,
        sequences: &[&SealedSequence],
        lift: &LayerLift,
        records_us: &mut u64,
    ) -> Result<Vec<SealedSequence>> {
        let info = self.resolve_arena_info()?;
        let chunks = sequences.iter().flat_map(|seq| seq.chunks.iter());
        let mapped: Vec<HeadGids> = chunks
            .clone()
            .map(|chunk| {
                chunk.gids.map_each(|gid| {
                    let at = lift.index.get(&gid.raw()).copied();
                    at.map(|i| lift.dests[i].clone()).ok_or_else(|| {
                        candle::Error::Msg(format!("warm lift: band {} was not planned", gid.raw()))
                    })
                })
            })
            .collect::<Result<_>>()?;
        let refs: Vec<ChunkRecordSrc<'_>> = chunks
            .clone()
            .zip(&mapped)
            .map(|(chunk, m)| ChunkRecordSrc {
                gids: m,
                k_pal: chunk.k_pal.as_slice(),
                v_pal: chunk.v_pal.as_slice(),
                k_scale: chunk.k_scale.as_slice(),
                v_scale: chunk.v_scale.as_slice(),
                k_fmt: chunk.k_fmt.as_slice(),
                v_fmt: chunk.v_fmt.as_slice(),
            })
            .collect();
        let t = Instant::now();
        let metas = self.build_meta_records(&refs, &info)?;
        *records_us += micros(t);
        drop(refs);
        let mut lifted = mapped.into_iter().zip(metas);
        sequences
            .iter()
            .map(|seq| {
                let chunks: Vec<SealedChunk> = seq
                    .chunks
                    .iter()
                    .map(|chunk| {
                        let (gids, meta) = lifted.next().ok_or_else(|| {
                            candle::Error::Msg("warm lift: fewer records than chunks".into())
                        })?;
                        Ok(SealedChunk {
                            gids,
                            meta,
                            ..chunk.clone()
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(SealedSequence {
                    chunks,
                    token_count: seq.token_count,
                    chunk_size: seq.chunk_size,
                    location: ArenaLocation::Gpu,
                })
            })
            .collect()
    }
}

/// `gid`'s slot address in its GPU arena.
fn destination(info: &[ResolvedArenaInfo], gid: &ChunkGid) -> Result<u64> {
    let arena = gid.arena_idx();
    let entry = info
        .get(arena)
        .ok_or_else(|| candle::Error::Msg(format!("warm lift: dest arena {arena} out of range")))?;
    if entry.base_ptr == 0 {
        candle::bail!("warm lift: dest arena {arena} is not GPU-resident");
    }
    Ok((entry.base_ptr as i64 + gid.chunk_idx() as i64 * entry.chunk_byte_stride) as u64)
}
