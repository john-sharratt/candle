//! `HeadGids` — strongly-typed per-head chunk ID collection.
//!
//! Each block in the chunked KV cache owns `N_PALETTE * 2 * n_kv_head` chunk GIDs,
//! indexed as `head * N_PALETTE * 2 + palette * 2 + is_value` (0 = K, 1 = V).
//! `HeadGids` wraps that vector so call-sites are self-documenting and cannot be
//! confused with other `Vec<ChunkGid>` meanings (e.g. one-GID-per-block lists).
//!
//! N_PALETTE = 4 palette sub-bands split HEAD_DIM into 4 equal parts.
//! Backward-compat accessors `k_gid(h)` / `v_gid(h)` return palette 0's GID.

use std::ops::{Deref, DerefMut};

use super::gid_pool::ChunkGid;
use crate::kv_cache::arena_table::N_PALETTE;
use crate::kv_cache::ResolvedArenaInfo;

/// Stride per head in the flat GID slice: N_PALETTE * 2.
pub const GIDS_PER_HEAD: usize = N_PALETTE * 2;

/// Per-head chunk GID collection for a single block.
///
/// Length is always `GIDS_PER_HEAD * n_kv_head = N_PALETTE * 2 * n_kv_head`, indexed as:
///
///   `slot = head * GIDS_PER_HEAD + palette * 2 + is_value`
///   (is_value: 0 = K, 1 = V)
///
/// Backward-compat aliases `k_gid(h)` / `v_gid(h)` return palette-0 slots.
#[derive(Debug, Clone)]
pub struct HeadGids(pub(crate) Vec<ChunkGid>);

impl HeadGids {
    /// Construct where every head × {palette} × {K, V} slot shares the same GID.
    #[inline]
    pub fn uniform(gid: ChunkGid, n_kv_head: usize) -> Self {
        Self(vec![gid; GIDS_PER_HEAD * n_kv_head])
    }

    /// Construct where all K palette slots share one GID and all V palette slots share another.
    ///
    /// Layout: for each head h, palette p: K slot = k_gid, V slot = v_gid.
    #[inline]
    pub fn from_kv(k_gid: ChunkGid, v_gid: ChunkGid, n_kv_head: usize) -> Self {
        let mut gids = Vec::with_capacity(GIDS_PER_HEAD * n_kv_head);
        for _ in 0..n_kv_head {
            for _ in 0..N_PALETTE {
                gids.push(k_gid.clone());
                gids.push(v_gid.clone());
            }
        }
        Self(gids)
    }

    /// Construct from a pre-built per-head GID vector.
    #[inline]
    pub fn from_vec(gids: Vec<ChunkGid>) -> Self {
        Self(gids)
    }

    /// Consume and return the inner vector.
    #[inline]
    pub fn into_inner(self) -> Vec<ChunkGid> {
        self.0
    }

    // Direct access to the slice for advanced use-cases. Call-sites should prefer the more specific accessors below.
    #[inline]
    pub fn as_slice(&self) -> &[ChunkGid] {
        &self.0
    }

    /// Return the K GID for head `h`, palette `p`.
    ///
    /// Indexed as `self.0[h * GIDS_PER_HEAD + p * 2]`.
    #[inline]
    pub fn k_gid_pal(&self, h: usize, p: usize) -> &ChunkGid {
        &self.0[h * GIDS_PER_HEAD + p * 2]
    }

    /// Return the V GID for head `h`, palette `p`.
    ///
    /// Indexed as `self.0[h * GIDS_PER_HEAD + p * 2 + 1]`.
    #[inline]
    pub fn v_gid_pal(&self, h: usize, p: usize) -> &ChunkGid {
        &self.0[h * GIDS_PER_HEAD + p * 2 + 1]
    }

    /// Return the K GID for head `h`, palette 0 (backward-compat alias).
    #[inline]
    pub fn k_gid(&self, h: usize) -> &ChunkGid {
        self.k_gid_pal(h, 0)
    }

    /// Return the V GID for head `h`, palette 0 (backward-compat alias).
    #[inline]
    pub fn v_gid(&self, h: usize) -> &ChunkGid {
        self.v_gid_pal(h, 0)
    }

    /// Number of KV heads (total slots / GIDS_PER_HEAD).
    #[inline]
    pub fn n_kv_head(&self) -> usize {
        self.0.len() / GIDS_PER_HEAD
    }

    /// Collect the deduplicated set of arena indices referenced by any slot.
    pub fn unique_arena_indices(&self) -> Vec<usize> {
        let mut seen = Vec::with_capacity(GIDS_PER_HEAD);
        for g in &self.0 {
            let ai = g.arena_idx();
            if !seen.contains(&ai) {
                seen.push(ai);
            }
        }
        seen
    }

    /// Collect deduplicated arena indices referenced by K slots only.
    pub fn unique_k_arena_indices(&self) -> Vec<usize> {
        let mut seen = Vec::with_capacity(GIDS_PER_HEAD);
        let n = self.n_kv_head();
        for h in 0..n {
            for p in 0..N_PALETTE {
                let ai = self.0[h * GIDS_PER_HEAD + p * 2].arena_idx();
                if !seen.contains(&ai) {
                    seen.push(ai);
                }
            }
        }
        seen
    }

    /// Collect deduplicated arena indices referenced by V slots only.
    pub fn unique_v_arena_indices(&self) -> Vec<usize> {
        let mut seen = Vec::with_capacity(GIDS_PER_HEAD);
        let n = self.n_kv_head();
        for h in 0..n {
            for p in 0..N_PALETTE {
                let ai = self.0[h * GIDS_PER_HEAD + p * 2 + 1].arena_idx();
                if !seen.contains(&ai) {
                    seen.push(ai);
                }
            }
        }
        seen
    }

    /// Total VRAM bytes for this chunk across all referenced arenas.
    ///
    /// Different heads may carry different formats (e.g. partially migrated
    /// chunks), so every GID is examined.  Deduplication uses a stack array
    /// bounded by the number of distinct `(format, location)` pairs; no heap
    /// allocation.
    pub fn arena_byte_size(&self, arena_infos: &[ResolvedArenaInfo]) -> u64 {
        // Upper bound: N_PALETTE sub-bands × K/V × number of compression
        // levels in flight (~10) × locations (2) — 32 is comfortably safe.
        const MAX_UNIQUE: usize = 32;
        let mut seen = [usize::MAX; MAX_UNIQUE];
        let mut seen_len = 0usize;
        let mut total = 0u64;
        for g in &self.0 {
            let ai = g.arena_idx();
            if !seen[..seen_len].contains(&ai) {
                if seen_len < MAX_UNIQUE {
                    seen[seen_len] = ai;
                    seen_len += 1;
                }
                total += arena_infos.get(ai).map_or(0, |i| i.chunk_byte_stride as u64);
            }
        }
        total
    }

    /// Check whether any slot references a given arena index.
    #[inline]
    pub fn references_arena(&self, arena_idx: usize) -> bool {
        self.0.iter().any(|g| g.arena_idx() == arena_idx)
    }

    /// Map each unique GID to a new one, preserving the sharing structure.
    ///
    /// `f` is called exactly once per unique raw GID value.  Slots that share a
    /// raw id (e.g. all K-palette slots in the float case from `from_kv`) receive
    /// a clone of the first mapped result — the same sharing structure is
    /// reproduced in the output `HeadGids`.  Detached / sentinel GIDs (raw < 0)
    /// are passed through without calling `f`.
    pub fn map_unique<F>(&self, mut f: F) -> candle::Result<HeadGids>
    where
        F: FnMut(&ChunkGid) -> candle::Result<ChunkGid>,
    {
        let mut cache: ahash::AHashMap<i64, ChunkGid> = ahash::AHashMap::new();
        let mut out = Vec::with_capacity(self.0.len());
        for gid in &self.0 {
            let raw = gid.raw();
            if raw < 0 {
                out.push(ChunkGid::detached(raw));
                continue;
            }
            let new_gid = match cache.get(&raw) {
                Some(cached) => cached.clone(),
                None => {
                    let new = f(gid)?;
                    cache.insert(raw, new.clone());
                    new
                }
            };
            out.push(new_gid);
        }
        Ok(HeadGids(out))
    }

    /// Whether all slots point to the same `(arena_idx, chunk_idx)`.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        match self.0.first() {
            None => true,
            Some(first) => {
                let ai = first.arena_idx();
                let ci = first.chunk_idx();
                self.0
                    .iter()
                    .all(|g| g.arena_idx() == ai && g.chunk_idx() == ci)
            }
        }
    }
}

impl Deref for HeadGids {
    type Target = [ChunkGid];
    #[inline]
    fn deref(&self) -> &[ChunkGid] {
        &self.0
    }
}

impl DerefMut for HeadGids {
    #[inline]
    fn deref_mut(&mut self) -> &mut [ChunkGid] {
        &mut self.0
    }
}

impl PartialEq for HeadGids {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for HeadGids {}

#[cfg(test)]
mod tests {
    use super::*;

    /// `map_unique` calls `f` exactly once per unique raw id and reuses the
    /// result for all slots that share that id.
    #[test]
    fn map_unique_calls_f_once_per_unique_id() {
        let gid_a = ChunkGid::detached(1);
        let gid_b = ChunkGid::detached(2);
        let gids = HeadGids::from_vec(vec![
            gid_a.clone(), gid_b.clone(),
            gid_a.clone(), gid_b.clone(),
            gid_a.clone(), gid_b.clone(),
        ]);

        let mut call_count = 0usize;
        let result = gids.map_unique(|gid| {
            call_count += 1;
            Ok(ChunkGid::detached(gid.raw() + 100))
        }).unwrap();

        assert_eq!(call_count, 2, "f must be called once per unique raw id");
        let out_raws: Vec<i64> = result.iter().map(|g| g.raw()).collect();
        assert_eq!(out_raws, vec![101, 102, 101, 102, 101, 102]);
    }

    /// Sentinel GIDs (raw < 0) pass through without calling `f`.
    #[test]
    fn map_unique_passes_through_sentinels() {
        let gids = HeadGids::from_vec(vec![
            ChunkGid::detached(-1),
            ChunkGid::detached(-1),
        ]);

        let mut call_count = 0usize;
        let result = gids.map_unique(|_| {
            call_count += 1;
            Ok(ChunkGid::detached(99))
        }).unwrap();

        assert_eq!(call_count, 0, "f must not be called for sentinel GIDs");
        assert!(result.iter().all(|g| g.raw() == -1));
    }

    /// Sharing structure is preserved: N slots with 2 unique ids → N slots with
    /// 2 unique ids in the output.
    #[test]
    fn map_unique_preserves_sharing_structure() {
        let k = ChunkGid::detached(10);
        let v = ChunkGid::detached(20);
        let gids = HeadGids::from_kv(k, v, 4); // N_PALETTE=4, 4 heads → 32 slots, 2 unique

        let result = gids.map_unique(|gid| Ok(ChunkGid::detached(gid.raw() + 1000))).unwrap();

        let unique_out: std::collections::HashSet<i64> =
            result.iter().map(|g| g.raw()).collect();
        assert_eq!(unique_out.len(), 2, "2 unique source ids → 2 unique output ids");
        assert!(unique_out.contains(&1010));
        assert!(unique_out.contains(&1020));
    }
}
