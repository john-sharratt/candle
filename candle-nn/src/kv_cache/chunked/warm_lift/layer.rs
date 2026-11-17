//! One layer's part of a warm lift: its sources resolved and its destinations
//! claimed before any byte moves, and its sealed sequences rebuilt at the new
//! placement after.

use std::collections::{HashMap, HashSet};

use candle::Result;

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
/// destination.
pub(crate) struct LayerLift {
    pub bands: Vec<LiftBand>,
    pub new_gids: HashMap<i64, ChunkGid>,
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
        let mut unique: Vec<i64> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();
        let mut src_keys: HashMap<i64, ArenaKey> = HashMap::new();
        let mut lens: HashMap<i64, usize> = HashMap::new();
        let elems = self.inner.elems_per_chunk();
        self.inner.storage.read(|s| -> Result<()> {
            for seq in sequences {
                for chunk in &seq.chunks {
                    for (gid, tag) in chunk.bands() {
                        let raw = gid.raw();
                        if !seen.insert(raw) {
                            continue;
                        }
                        let arena_idx = (raw as usize) / GID_STRIDE;
                        let key = s.arena_key(arena_idx).ok_or_else(|| {
                            candle::Error::Msg(format!("warm lift: arena {arena_idx} not found"))
                        })?;
                        if key.location != ArenaLocation::Cpu {
                            candle::bail!(
                                "warm lift: source arena {arena_idx} is not CPU-resident"
                            );
                        }
                        // The warm arena is a class-strided byte slab with no
                        // format to ask; the band's own tag says its length.
                        let len = payload_bytes_for_tag(tag, elems).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "warm lift: band tag {tag:?} names no storage format, so its \
                                 byte length is unknown"
                            ))
                        })?;
                        src_keys.insert(raw, key);
                        lens.insert(raw, len);
                        unique.push(raw);
                    }
                }
            }
            Ok(())
        })??;

        let mut new_gids: HashMap<i64, ChunkGid> = HashMap::new();
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let n_kv_head = self.inner.n_kv_head;
            for seq in sequences {
                for chunk in &seq.chunks {
                    for h in 0..n_kv_head {
                        for side_v in [false, true] {
                            let raws: Vec<i64> = (0..N_PALETTE)
                                .map(|p| match side_v {
                                    true => chunk.gids.v_gid_pal(h, p).raw(),
                                    false => chunk.gids.k_gid_pal(h, p).raw(),
                                })
                                .collect();
                            let distinct: HashSet<i64> = raws.iter().copied().collect();
                            if distinct.len() != raws.len()
                                || raws.iter().any(|r| *r < 0 || new_gids.contains_key(r))
                            {
                                continue;
                            }
                            let Some(first) = src_keys.get(&raws[0]) else {
                                continue;
                            };
                            if raws[1..]
                                .iter()
                                .any(|r| src_keys.get(r).map(|k| k.class) != Some(first.class))
                            {
                                continue;
                            }
                            let run = self.alloc_chunk_run_for_key(
                                first.at(ArenaLocation::Gpu),
                                raws.len(),
                            )?;
                            for (raw, gid) in raws.into_iter().zip(run) {
                                new_gids.insert(raw, gid);
                            }
                        }
                    }
                }
            }
            for &raw in &unique {
                if new_gids.contains_key(&raw) {
                    continue;
                }
                let gid = self.alloc_chunk_for_key(src_keys[&raw].at(ArenaLocation::Gpu))?;
                new_gids.insert(raw, gid);
            }
        }

        let info = self.resolve_arena_info()?;
        let bands = unique
            .iter()
            .map(|raw| {
                let gid = &new_gids[raw];
                let dst = destination(&info, gid)?;
                Ok(LiftBand {
                    layer,
                    src_arena: (*raw as usize) / GID_STRIDE,
                    src_chunk: (*raw as usize) % GID_STRIDE,
                    bytes: lens[raw],
                    dst,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LayerLift { bands, new_gids })
    }

    /// `sequences` at their lifted placement: every gid mapped to its new GPU
    /// slot, each chunk's resident KV-head record built there — batched per
    /// sequence — and every other field, formats included, carried unchanged,
    /// since a lift copies bytes verbatim into a same-format slot.
    pub(crate) fn finish_warm_lift(
        &self,
        sequences: &[&SealedSequence],
        lift: &LayerLift,
    ) -> Result<Vec<SealedSequence>> {
        let info = self.resolve_arena_info()?;
        sequences
            .iter()
            .map(|seq| {
                let mapped: Vec<HeadGids> = seq
                    .chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .gids
                            .map_unique(|gid| Ok(lift.new_gids[&gid.raw()].clone()))
                    })
                    .collect::<Result<_>>()?;
                let refs: Vec<ChunkRecordSrc<'_>> = seq
                    .chunks
                    .iter()
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
                let metas = self.build_meta_records(&refs, &info)?;
                let chunks: Vec<SealedChunk> = seq
                    .chunks
                    .iter()
                    .zip(mapped)
                    .zip(metas)
                    .map(|((chunk, gids), meta)| SealedChunk {
                        gids,
                        meta,
                        ..chunk.clone()
                    })
                    .collect();
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
