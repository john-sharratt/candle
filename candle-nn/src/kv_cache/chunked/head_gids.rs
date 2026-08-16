//! `HeadGids` — strongly-typed per-head chunk ID collection.
//!
//! Each block in the chunked KV cache owns `N_PALETTE * 2 * n_kv_head` chunk GIDs,
//! indexed as `head * N_PALETTE * 2 + palette * 2 + is_value` (0 = K, 1 = V).
//! `HeadGids` wraps that vector so call-sites are self-documenting and cannot be
//! confused with other `Vec<ChunkGid>` meanings (e.g. one-GID-per-block lists).
//!
//! N_PALETTE = 4 palette sub-bands split HEAD_DIM into 4 equal parts.
//! Backward-compat accessors `k_gid(h)` / `v_gid(h)` return palette 0's GID.

use std::ops::Deref;
use std::sync::Arc;

use super::gid_pool::ChunkGid;
use super::size_class::payload_bytes_for_tag;
use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};

/// Stride per head in the flat GID slice: N_PALETTE * 2.
pub const GIDS_PER_HEAD: usize = N_PALETTE * 2;

/// Iterate `(gid, format tag)` for every band slot of one chunk, in the
/// interleaved K,V order of [`HeadGids::as_slice`].
///
/// The gid says *where* a band's bytes are; the tag says *how to read them*.
/// Under size classes only the chunk can answer the second question
/// (`docs/archived/arena_unification.md` principle 8), so every predicate over a
/// chunk's formats walks this rather than the arenas its gids point into.
///
/// This is the one place the `[h * N_PALETTE + p]` tag indexing is written
/// down; `SealedChunk::bands` and `ChunkWindow::bands` both come through here
/// so a live chunk and the sealed chunk it becomes cannot disagree about which
/// tag belongs to which slot.
pub fn band_tags<'a>(
    gids: &'a HeadGids,
    k_fmt: &'a [u8],
    v_fmt: &'a [u8],
) -> impl Iterator<Item = (&'a ChunkGid, ArenaFormatTag)> + 'a {
    gids.as_slice().iter().enumerate().map(move |(i, gid)| {
        // Slot i is `head * (bands * 2) + band * 2 + is_value`, and tags are
        // indexed `[head * bands + band]` — so the tag index is simply `i / 2`
        // and the side the low bit, WHATEVER the per-head band count is
        // (4-band GQA and the 16-band single latent alike). Deriving it
        // through the global `GIDS_PER_HEAD` stride would mis-map every band
        // past the fourth on a single-latent window.
        let t = i / 2;
        let side = if i % 2 == 0 { k_fmt } else { v_fmt };
        (
            gid,
            side.get(t)
                .copied()
                .map_or(ArenaFormatTag::Invalid, ArenaFormatTag::from_u8),
        )
    })
}

/// Per-head chunk GID collection for a single block.
///
/// Length is always `GIDS_PER_HEAD * n_kv_head = N_PALETTE * 2 * n_kv_head`, indexed as:
///
///   `slot = head * GIDS_PER_HEAD + palette * 2 + is_value`
///   (is_value: 0 = K, 1 = V)
///
/// Backward-compat aliases `k_gid(h)` / `v_gid(h)` return palette-0 slots.
///
/// The inner gid array sits behind an `Arc` so `HeadGids::clone` is
/// **O(1)** (one Arc ref-count bump) instead of cloning every
/// `ChunkGid`. Cold-load registers each block's gids into both the
/// slot's chunk window and the per-call return Vec; the old per-gid
/// clone showed up as ~580 µs/cold-load in `register_us`. Mutating
/// HeadGids after construction is not supported (no `DerefMut`, no
/// `into_inner`) — every existing call-site treats it as immutable.
#[derive(Debug, Clone)]
pub struct HeadGids(pub(crate) Arc<Vec<ChunkGid>>);

impl HeadGids {
    /// Construct where every head × {palette} × {K, V} slot shares the same GID.
    #[inline]
    pub fn uniform(gid: ChunkGid, n_kv_head: usize) -> Self {
        Self(Arc::new(vec![gid; GIDS_PER_HEAD * n_kv_head]))
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
        Self(Arc::new(gids))
    }

    /// Construct from a pre-built per-head GID vector.
    #[inline]
    pub fn from_vec(gids: Vec<ChunkGid>) -> Self {
        Self(Arc::new(gids))
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
        for g in self.0.iter() {
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

    /// Total bytes this chunk's bands occupy across all referenced arenas.
    ///
    /// Every band contributes **its own format's payload**, read from the
    /// chunk's tags. The arenas cannot supply it: a size-class arena holds
    /// whatever fits its stride, so its slot width is an upper bound on a
    /// band's bytes, not the bytes themselves (`docs/archived/arena_unification.md`
    /// invariant 8). Bands whose tag names no storage format contribute
    /// nothing — they have no length to contribute.
    ///
    /// Deduplication is by `(arena_idx, chunk_idx)`: a single chunk's sub-band
    /// GIDs may alias, and an aliased physical slot must be counted once.
    /// (Before this was fixed, dedup was by `arena_idx` alone and `byte_size`
    /// was under-reported by a factor of `chunk_count_per_arena`, causing
    /// `seal_to_chunk_images` to silently drop 15/16 of every chunk's bytes on
    /// the persistence gather.)
    pub fn arena_byte_size(&self, k_fmt: &[u8], v_fmt: &[u8], elems_per_chunk: usize) -> u64 {
        // Upper bound: every chunk has at most GIDS_PER_HEAD × n_kv_head =
        // 8 × n_kv_head unique slot positions. The stack buffer sizes for the
        // largest production shape (Qwen3-235B GQA, n_kv_head = 8 → 64 slots)
        // plus headroom, so there is no heap allocation on this path.
        const MAX_UNIQUE: usize = 128;
        let mut seen: [(usize, usize); MAX_UNIQUE] = [(usize::MAX, usize::MAX); MAX_UNIQUE];
        let mut seen_len = 0usize;
        let mut total = 0u64;
        for (g, tag) in band_tags(self, k_fmt, v_fmt) {
            let key = (g.arena_idx(), g.chunk_idx());
            if !seen[..seen_len].contains(&key) {
                if seen_len < MAX_UNIQUE {
                    seen[seen_len] = key;
                    seen_len += 1;
                }
                total += payload_bytes_for_tag(tag, elems_per_chunk).unwrap_or(0) as u64;
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
        for gid in self.0.iter() {
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
        Ok(HeadGids(Arc::new(out)))
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

impl PartialEq for HeadGids {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for HeadGids {}

/// One chunk's band gids together with the format tags that say how to read
/// them — everything the selection path needs about a chunk, and nothing else.
///
/// The two halves are inseparable and were previously carried as parallel
/// vectors: the gid gives a band's *address*, the tag gives its *layout*. An
/// arena can answer the first and, under size classes, not the second
/// (`docs/archived/arena_unification.md` principle 8), so the selection table is built
/// from these rather than from arena state.
///
/// Cheap to clone: `HeadGids` is one `Arc` bump and the tags are `Arc`-shared
/// per chunk, so a keepalive vector costs three ref-count bumps per chunk.
#[derive(Debug, Clone)]
pub struct ChunkBands {
    /// The chunk's `(head, palette, K/V)` gid grid.
    pub gids: HeadGids,
    /// K band format tags ([`crate::kv_cache::ArenaFormatTag::as_u8`]),
    /// `n_kv_head × N_PALETTE` entries in `[h * N_PALETTE + p]` order.
    pub k_fmt: Arc<Vec<u8>>,
    /// V band format tags, same layout as `k_fmt`.
    pub v_fmt: Arc<Vec<u8>>,
}

impl ChunkBands {
    /// Borrow a sealed chunk's bands without cloning its gid grid's contents.
    pub fn from_sealed(chunk: &crate::kv_cache::chunked::types::SealedChunk) -> Self {
        Self {
            gids: chunk.gids.clone(),
            k_fmt: chunk.k_fmt.clone(),
            v_fmt: chunk.v_fmt.clone(),
        }
    }

    /// The K and V format tags for band `(h, p)`, as raw tag bytes.
    ///
    /// Falls back to [`crate::kv_cache::ArenaFormatTag::Invalid`] for a band the
    /// chunk never recorded, so an unrecorded band fails every format check
    /// rather than silently reading as `F32` (tag 0).
    #[inline]
    pub fn band_tags(&self, h: usize, p: usize) -> (u8, u8) {
        let i = h * N_PALETTE + p;
        let invalid = crate::kv_cache::ArenaFormatTag::Invalid.as_u8();
        (
            self.k_fmt.get(i).copied().unwrap_or(invalid),
            self.v_fmt.get(i).copied().unwrap_or(invalid),
        )
    }
}

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
            gid_a.clone(),
            gid_b.clone(),
            gid_a.clone(),
            gid_b.clone(),
            gid_a.clone(),
            gid_b.clone(),
        ]);

        let mut call_count = 0usize;
        let result = gids
            .map_unique(|gid| {
                call_count += 1;
                Ok(ChunkGid::detached(gid.raw() + 100))
            })
            .unwrap();

        assert_eq!(call_count, 2, "f must be called once per unique raw id");
        let out_raws: Vec<i64> = result.iter().map(|g| g.raw()).collect();
        assert_eq!(out_raws, vec![101, 102, 101, 102, 101, 102]);
    }

    /// Sentinel GIDs (raw < 0) pass through without calling `f`.
    #[test]
    fn map_unique_passes_through_sentinels() {
        let gids = HeadGids::from_vec(vec![ChunkGid::detached(-1), ChunkGid::detached(-1)]);

        let mut call_count = 0usize;
        let result = gids
            .map_unique(|_| {
                call_count += 1;
                Ok(ChunkGid::detached(99))
            })
            .unwrap();

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

        let result = gids
            .map_unique(|gid| Ok(ChunkGid::detached(gid.raw() + 1000)))
            .unwrap();

        let unique_out: std::collections::HashSet<i64> = result.iter().map(|g| g.raw()).collect();
        assert_eq!(
            unique_out.len(),
            2,
            "2 unique source ids → 2 unique output ids"
        );
        assert!(unique_out.contains(&1010));
        assert!(unique_out.contains(&1020));
    }

    // ── arena_byte_size regression tests ─────────────────────────────────
    //
    // Cover the dedup-by-(arena_idx, chunk_idx) contract that fixed the
    // cold-load Quantized data loss (commit history: persistence path was
    // under-reporting byte_size by 16x because dedup was by arena_idx only),
    // and the payload-not-stride contract that size classes introduced.

    use super::super::GID_STRIDE;
    use crate::kv_cache::arena_table::ArenaFormatTag;
    use crate::kv_cache::{KvFormat, QuantFormat};
    use candle::DType;

    /// The production palette-4 geometry: a slot is 32 tokens x 32 dims.
    const ELEMS: usize = 1024;

    /// Helper: construct a `ChunkGid` for `(arena_idx, chunk_idx)`.
    fn gid_at(arena_idx: usize, chunk_idx: usize) -> ChunkGid {
        let raw = (arena_idx * GID_STRIDE + chunk_idx) as i64;
        ChunkGid::detached(raw)
    }

    /// A uniform tag vector for `n_kv_head` heads, all bands the same format.
    fn tags(format: KvFormat, n_kv_head: usize) -> Vec<u8> {
        vec![ArenaFormatTag::from_kv_format(format).as_u8(); n_kv_head * N_PALETTE]
    }

    const BF16: KvFormat = KvFormat::Float(DType::BF16);
    const Q8_0: KvFormat = KvFormat::Quantized(QuantFormat::Q8_0);
    const Q2_0: KvFormat = KvFormat::Quantized(QuantFormat::Q2_0);

    /// **Regression**: 16 sub-band GIDs all in the same arena (uniform-format
    /// backing, different `chunk_idx` per sub-band) must sum every band's
    /// payload — not collapse to a single one. This was the bug:
    /// under-reported byte_size by 16x, so `seal_to_chunk_images` silently
    /// dropped 15/16 of every quantized chunk's payload and resumed
    /// conversations lost context after the cold-load round-trip.
    #[test]
    fn arena_byte_size_sums_distinct_chunk_idxs_in_one_arena() {
        // 16 GIDs in arena 0, chunk_idx 0..15 — the n_kv_head=2, N_PALETTE=4,
        // K+V 16-slot layout of a single SealedChunk under a Q8_0 backing.
        let gids = HeadGids::from_vec((0..16).map(|i| gid_at(0, i)).collect());
        let t = tags(Q8_0, 2);

        assert_eq!(
            gids.arena_byte_size(&t, &t, ELEMS),
            16 * 1088,
            "16 distinct chunk slots must contribute 16 payloads"
        );
    }

    /// A single repeated GID (one slot, referenced 16 times) contributes its
    /// payload **once** — dedup must still collapse exact duplicates.
    #[test]
    fn arena_byte_size_dedups_exact_duplicate_gids() {
        let single = gid_at(0, 7);
        let gids = HeadGids::from_vec(vec![single; 16]);
        let t = tags(BF16, 2);

        assert_eq!(
            gids.arena_byte_size(&t, &t, ELEMS),
            2048,
            "16 references to the SAME (arena, chunk) collapse to one payload"
        );
    }

    /// **The whole point of reading tags rather than arenas.** K and V bands of
    /// one chunk can be in different formats, and each must contribute its own
    /// payload. An arena-derived length cannot express this at all: both sides
    /// may live in the same size-class region.
    #[test]
    fn arena_byte_size_follows_each_band_own_format() {
        // n_kv_head = 1 → 8 slots: 4 K bands and 4 V bands, interleaved.
        let gids = HeadGids::from_vec((0..8).map(|i| gid_at(0, i)).collect());
        let k = tags(Q8_0, 1);
        let v = tags(Q2_0, 1);

        assert_eq!(
            gids.arena_byte_size(&k, &v, ELEMS),
            4 * 1088 + 4 * 320,
            "K bands contribute Q8_0 payloads, V bands Q2_0 payloads"
        );
    }

    /// Sub-bands routed across two arenas: each `(arena, chunk)` pair
    /// contributes exactly once, and the arena index does not change the
    /// length — only the tag does.
    #[test]
    fn arena_byte_size_sums_across_distinct_arenas() {
        let mut v: Vec<ChunkGid> = (0..8).map(|i| gid_at(0, i)).collect();
        v.extend((0..8).map(|i| gid_at(1, i)));
        let gids = HeadGids::from_vec(v);
        let t = tags(BF16, 2);

        assert_eq!(
            gids.arena_byte_size(&t, &t, ELEMS),
            16 * 2048,
            "16 distinct slots across two arenas each contribute once"
        );
    }

    /// **The A1 regression, restated for classes.** `byte_size` feeds
    /// `seal_to_chunk_images`' blob slicing and the per-chunk Fletcher goldens
    /// computed over those bytes, so it must count the format's **payload**,
    /// never the slot stride. A Q2_0 band (320 B) sits in a 320 B slot but a
    /// Q0 band (32 B) sits in the same 320 B slot; summing strides would make
    /// them equal, grow every on-disk image by the pad, cover the pad with the
    /// checksum, and push it over PCIe on every migration.
    ///
    /// This is the assertion the round-trip tests structurally cannot make —
    /// they read the same length on both sides of the trip, so a symmetric
    /// error compares equal (audit A7).
    #[test]
    fn arena_byte_size_counts_payload_not_slot_stride() {
        let gids = HeadGids::from_vec((0..8).map(|i| gid_at(0, i)).collect());
        let q0 = tags(KvFormat::Quantized(QuantFormat::Q0), 1);
        let q2 = tags(Q2_0, 1);

        // Both formats share the 320 B class, so a stride-derived sum would
        // report the same number for each.
        assert_eq!(gids.arena_byte_size(&q0, &q0, ELEMS), 8 * 32);
        assert_eq!(gids.arena_byte_size(&q2, &q2, ELEMS), 8 * 320);
    }

    /// A tag that names no storage format contributes nothing rather than
    /// guessing a width. Length is not a thing to default.
    #[test]
    fn arena_byte_size_ignores_unmapped_tags() {
        let gids = HeadGids::from_vec((0..8).map(|i| gid_at(0, i)).collect());
        let bad = vec![ArenaFormatTag::Invalid.as_u8(); N_PALETTE];

        assert_eq!(gids.arena_byte_size(&bad, &bad, ELEMS), 0);
    }

    /// Missing tags (a chunk that never recorded them) also contribute zero,
    /// and must not panic.
    #[test]
    fn arena_byte_size_tolerates_absent_tags() {
        let gids = HeadGids::from_vec(vec![gid_at(5, 0), gid_at(5, 1)]);
        assert_eq!(gids.arena_byte_size(&[], &[], ELEMS), 0);
    }

    /// Detached / sentinel GIDs (raw < 0) are still distinct slots by
    /// `(arena_idx, chunk_idx)`, and mixing them with real GIDs must not
    /// disturb the real ones' accounting.
    #[test]
    fn arena_byte_size_handles_sentinel_gids() {
        let real = HeadGids::from_vec(vec![gid_at(0, 0), gid_at(0, 1)]);
        let t = tags(BF16, 1);
        let baseline = real.arena_byte_size(&t, &t, ELEMS);
        assert_eq!(baseline, 2 * 2048);
    }
}
