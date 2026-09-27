//! Minting a fresh `KvHead` record for each chunk a compaction relocated.
//!
//! # Why a relocated chunk needs a NEW record rather than a rewritten one
//!
//! A chunk has exactly **one** record, shared by every holder of that chunk, and a
//! record holds a clone of the [`HeadGids`] its pointer words were serialized from — so
//! it cannot outlive the bands it names. Those two facts together are what make a
//! compaction safe, and they are also what make rewriting a record impossible: a pass
//! that rewrites *some* of a chunk's holders to the destination and leaves the rest on
//! the source has no correct value to put in a shared word, whichever slot it picks
//! (`compact_backings`).
//!
//! Minting sidesteps the question. The holders the sweep reached get new gids **and a
//! new record built from them**; the original record stays with the original slots, for
//! the holders that still hold it, and dies with the last of them. Each side is
//! internally consistent for its whole life, and neither had to be *found*.
//!
//! This is the rule the cache had before the compaction pass was written — "a relocation
//! mints a fresh record, records are never rewritten" — restored, and cheap now for the
//! reason records were moved into an arena in the first place: a record slot is an
//! ordinary arena claim, and the bytes are written by one batched device launch.
//!
//! # Why the source is then reclaimed
//!
//! Installing the new record drops the old one. When that was its chunk's last holder
//! the record's slot frees, and with it the clone of the *old* `HeadGids` it was
//! holding — which is the last reference to the source bands, so they free too and
//! `release_empty_arenas` can hand the region back. Without minting the source stayed
//! pinned by its record for as long as the chunk lived, so a pass moved K/V without
//! ever freeing any: measured on the Flash-Next engine probe, 1.5–1.8 M relocations for
//! 3–11 regions reclaimed, with the census re-planning the same record-pinned sources
//! every pass because no holder named them.
//!
//! # Only what moved
//!
//! One record per **replacement allocation**, and only for a chunk that had a record to
//! replace. Holders sharing a `HeadGids` allocation are the same chunk with the same
//! bands, so they share one minted record — keyed on [`HeadGids::alloc_id`], exactly as
//! the gid rewrite is memoised. A live writer window carries no record and is given
//! none; a chunk nothing relocated is never visited.

use std::collections::HashSet;
use std::sync::Arc;

use ahash::AHashMap;
use candle::Result;

use super::backing::{ChunkedKvBacking, RecordLayout};
use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
use super::meta_pool::{ChunkRecordSrc, MetaGid};

/// One chunk's record inputs, owned for the length of the pass.
///
/// The fill kernel takes [`ChunkRecordSrc`], which borrows slices — so the queue holds
/// the chunk's own `Arc`s (a refcount bump each, no copy) and the borrows are formed at
/// flush time.
struct Queued {
    /// The **new** gids, which are both what the record's words will name and what the
    /// handle holds to keep them alive.
    gids: HeadGids,
    k_pal: Arc<Vec<u8>>,
    v_pal: Arc<Vec<u8>>,
    k_scale: Arc<Vec<f32>>,
    v_scale: Arc<Vec<f32>>,
    k_fmt: Arc<Vec<u8>>,
    v_fmt: Arc<Vec<u8>>,
}

impl Queued {
    /// Whether `src` describes the same record body this entry was built from.
    ///
    /// Compared by contents, not by `Arc` identity: two holders of one chunk legitimately
    /// hold separate `Arc`s over equal data.
    ///
    /// Not gated on `debug_assertions` even though only the `debug_assert` in
    /// [`RecordMint::mint`] calls it: `debug_assert!` expands to `if cfg!(..) { .. }`, so
    /// its body is type-checked and compiled in release too, and gating the callee breaks
    /// the release build only.
    fn matches(&self, src: &RecordInputs<'_>) -> bool {
        *self.k_pal == **src.k_pal
            && *self.v_pal == **src.v_pal
            && *self.k_scale == **src.k_scale
            && *self.v_scale == **src.v_scale
            && *self.k_fmt == **src.k_fmt
            && *self.v_fmt == **src.v_fmt
    }
}

/// One record arena's slot geometry, cached per arena index.
///
/// The three scalars a record's address needs, rather than `ResolvedArenaInfo` itself,
/// which is `Clone` and not `Copy` — this is read on every mint.
#[derive(Clone, Copy)]
struct RecordExtent {
    base: u64,
    stride: i64,
    capacity: u32,
}

/// A chunk's record inputs as the holder hands them over.
///
/// Separate from [`ChunkRecordSrc`] because that one borrows and this one is cloned into
/// the queue; the fields are the `Arc`s every `SealedChunk` and `ChunkWindow` already
/// carries, so building one costs six refcount bumps.
pub(super) struct RecordInputs<'a> {
    pub k_pal: &'a Arc<Vec<u8>>,
    pub v_pal: &'a Arc<Vec<u8>>,
    pub k_scale: &'a Arc<Vec<f32>>,
    pub v_scale: &'a Arc<Vec<f32>>,
    pub k_fmt: &'a Arc<Vec<u8>>,
    pub v_fmt: &'a Arc<Vec<u8>>,
}

/// Claims record slots during a compaction's holder sweep and fills them all in one
/// launch at the end of the pass.
///
/// Holds an `Arc<BackingInner>` rather than borrowing a backing, so the sweep — which
/// runs inside a closure the caller owns, across three crates — does not have to thread
/// a lifetime through. Every layer shares one `BackingInner` (`new_layer` clones the
/// `Arc`), so the geometry this is built at is every backing's geometry by construction.
pub(super) struct RecordMint {
    layout: RecordLayout,
    /// Record arenas resolved so far, by arena index.
    ///
    /// Filled on demand and kept: a claim usually lands in an arena already here, and
    /// resolving per claim is what made a 4,096-record cold load pay 4,096 lock
    /// round-trips before `build_meta_records` was batched.
    info: AHashMap<usize, RecordExtent>,
    /// `HeadGids::alloc_id` of a replacement → the record minted for it, so holders
    /// sharing an allocation share one record.
    minted: AHashMap<usize, MetaGid>,
    /// One entry per minted record, parallel to [`Self::handles`].
    queued: Vec<Queued>,
    handles: Vec<MetaGid>,
    /// Chunks that wanted a record and could not have one, because the record pool ran
    /// out of *already-existing* slots mid-sweep.
    ///
    /// Not a failure: the chunk keeps the record it has, which still names its source and
    /// still holds it alive, so the pass is correct and simply reclaims nothing for that
    /// chunk. Counted because a persistently non-zero reading means
    /// [`ChunkedKvBacking::reserve_record_slots`] is under-provisioning and the reclaim is
    /// quietly degrading.
    declined: usize,
    /// Record arenas created up front by the reservation, for the pass's log line.
    reserved_arenas: usize,
}

impl RecordMint {
    /// A minter for this backing's geometry, or `None` when records are not
    /// device-resident and there is nothing to mint.
    pub(super) fn new(backing: &ChunkedKvBacking) -> Result<Option<Self>> {
        if !backing.records_are_resident() {
            return Ok(None);
        }
        Ok(Some(Self {
            layout: backing.record_layout()?,
            info: AHashMap::new(),
            minted: AHashMap::new(),
            queued: Vec::new(),
            handles: Vec::new(),
            declined: 0,
            reserved_arenas: 0,
        }))
    }

    /// Provision the record pool for a pass that may mint up to `want` records.
    ///
    /// **Must be called before the sweep touches a holder.** Creating a record arena is an
    /// arena *registration*, and a registration mid-sweep can re-tenant an `arena_idx` a
    /// half-rewritten holder still names — see
    /// [`ChunkedKvBacking::try_alloc_record_slot`]. Doing it here means every arena this
    /// pass needs exists before anything is published, and the sweep only pops free lists.
    ///
    /// `want` is an upper bound (the pass's move count), so this over-provisions: a
    /// relocated chunk covers `n_kv_head · n_palette · 2` moved bands, so the real mint
    /// count runs a half to a thirteenth of the moves. Over-provisioning costs a record
    /// arena or two — 16 MiB each, released again by the ordinary empty-arena sweep — and
    /// under-provisioning costs reclaim, so the bound is the right way to be wrong.
    pub(super) fn reserve(&mut self, backing: &ChunkedKvBacking, want: usize) -> Result<()> {
        self.reserved_arenas = backing.reserve_record_slots(self.layout.key, want)?;
        Ok(())
    }

    /// Records minted this pass.
    pub(super) fn len(&self) -> usize {
        self.handles.len()
    }

    /// Chunks that wanted a record and could not have one — see [`Self::declined`].
    pub(super) fn declined(&self) -> usize {
        self.declined
    }

    /// Record arenas the reservation created.
    pub(super) fn reserved_arenas(&self) -> usize {
        self.reserved_arenas
    }

    /// The record for a chunk whose gids were just rewritten to `next`.
    ///
    /// Claims a slot and computes its address now — both host-only — and queues the
    /// bytes for [`Self::flush`]. The handle is returned so the holder installs it in the
    /// same visit, which is what keeps the sweep to one traversal.
    pub(super) fn mint(
        &mut self,
        backing: &ChunkedKvBacking,
        next: &HeadGids,
        src: RecordInputs<'_>,
    ) -> Result<Option<MetaGid>> {
        let id = next.alloc_id();
        if let Some(done) = self.minted.get(&id) {
            // **The second holder's own inputs are discarded, so assert they agree.**
            //
            // Holders sharing a `HeadGids` allocation are the same chunk, so their palette
            // maps, scales and format tags must be identical — but nothing in the type
            // system says so, and the record built from the first holder's inputs is what
            // every later one gets. If they ever diverged, the second chunk's bands would be
            // decoded with the first's format tags: quantized bytes read at the wrong width,
            // which cannot fault. This is the same assumption whose earlier version — "two
            // holders sharing a record describe the same chunk and therefore hold identical
            // gids" — was wrong, and the fix there was to stop assuming silently.
            debug_assert!(
                self.queued
                    .iter()
                    .find(|q| q.gids.alloc_id() == id)
                    .is_none_or(|q| q.matches(&src)),
                "two holders of one gid allocation disagree about their record inputs, so \
                 the second would be handed a record describing the first's palettes, \
                 scales and format tags",
            );
            return Ok(Some(done.clone()));
        }
        // **Non-creating, and the caller keeps its old record if the pool is out.** See
        // `try_alloc_record_slot`: creating an arena here would re-tenant an `arena_idx`
        // that holders this sweep has not reached are still naming.
        let Some(gid) = backing.try_alloc_record_slot(self.layout.key) else {
            self.declined += 1;
            return Ok(None);
        };
        let addr = self.slot_addr(backing, &gid)?;
        // The handle holds `next` — the gids the fill is about to serialize — so the
        // record cannot outlive the destination bands any more than the original could
        // outlive the sources.
        let handle = MetaGid::from_slot(gid, next.clone(), addr);
        self.handles.push(handle.clone());
        self.queued.push(Queued {
            gids: next.clone(),
            k_pal: Arc::clone(src.k_pal),
            v_pal: Arc::clone(src.v_pal),
            k_scale: Arc::clone(src.k_scale),
            v_scale: Arc::clone(src.v_scale),
            k_fmt: Arc::clone(src.k_fmt),
            v_fmt: Arc::clone(src.v_fmt),
        });
        self.minted.insert(id, handle.clone());
        Ok(Some(handle))
    }

    /// Device address of a freshly claimed record slot, resolving its arena if this is
    /// the first slot seen there.
    fn slot_addr(&mut self, backing: &ChunkedKvBacking, gid: &ChunkGid) -> Result<u64> {
        let arena = gid.arena_idx();
        if !self.info.contains_key(&arena) {
            // A claim can create an arena, so the miss path resolves rather than
            // assuming the pass's opening snapshot covers it.
            let needed: HashSet<usize> = [arena].into_iter().collect();
            let resolved = backing.resolve_arena_info_for(&needed)?;
            let Some(r) = resolved.get(arena) else {
                candle::bail!(
                    "compaction minted a record in arena {arena}, which the arena table \
                     does not resolve. Its address would be computed from a base the pool \
                     does not own."
                )
            };
            self.info.insert(
                arena,
                RecordExtent {
                    base: r.base_ptr,
                    stride: r.chunk_byte_stride,
                    capacity: r.chunk_capacity,
                },
            );
        }
        let r = self.info[&arena];
        let slot = gid.chunk_idx();
        if r.base == 0 || r.stride <= 0 {
            candle::bail!(
                "compaction minted a record in arena {arena}, which has no device \
                 residence (base {:#x}, stride {}). A record must be readable by the \
                 paged kernels.",
                r.base,
                r.stride,
            )
        }
        if slot >= r.capacity as usize {
            candle::bail!(
                "compaction minted record slot {slot} in arena {arena}, which holds only \
                 {} slots. The address that implies lies past the arena's end, inside \
                 whatever tenant holds the next region — and every address in the \
                 reservation is mapped, so writing it would not fault.",
                r.capacity,
            )
        }
        if (r.stride as usize) < self.layout.record_bytes {
            candle::bail!(
                "compaction minted a {} B record into arena {arena}, whose slots are only \
                 {} B. The fill would run past the slot into the next one's record.",
                self.layout.record_bytes,
                r.stride,
            )
        }
        Ok(r.base + (slot * r.stride as usize) as u64)
    }

    /// Write every minted record's bytes in one launch.
    ///
    /// **Must run before the pass's closing barrier**, which is what makes the writes
    /// visible to every other stream, and before `release_empty_arenas` hands any region
    /// back. Nothing reads a minted record until then: the pass holds the arena window,
    /// so no forward is in flight.
    pub(super) fn flush(&mut self, backing: &ChunkedKvBacking) -> Result<usize> {
        if self.handles.is_empty() {
            return Ok(0);
        }
        // The destination bands, resolved once for the whole batch — these arenas were
        // claimed by this pass, so the plan's opening snapshot does not describe them.
        let needed: HashSet<usize> = self
            .queued
            .iter()
            .flat_map(|q| q.gids.as_slice().iter().map(|g| g.arena_idx()))
            .collect();
        let arena_info = backing.resolve_arena_info_for(&needed)?;
        let srcs: Vec<ChunkRecordSrc<'_>> = self
            .queued
            .iter()
            .map(|q| ChunkRecordSrc {
                gids: &q.gids,
                k_pal: q.k_pal.as_slice(),
                v_pal: q.v_pal.as_slice(),
                k_scale: q.k_scale.as_slice(),
                v_scale: q.v_scale.as_slice(),
                k_fmt: q.k_fmt.as_slice(),
                v_fmt: q.v_fmt.as_slice(),
            })
            .collect();
        backing.fill_record_batch(&self.handles, &srcs, &arena_info, self.layout)?;
        // **Did the fill write the record the serializer would have written?**
        //
        // The seal path's equivalent is held against `serialize_kv_heads` byte-for-byte by
        // unit test over six geometries; this is the same claim for the compaction's batch,
        // against the real arena table, on real relocated chunks. It reads every minted
        // record back, so it is a synchronisation per pass and belongs behind the harness —
        // but it is the only check that can tell a wrong record from a wrong *reader* of a
        // right one, which is the bisection this fault needed.
        #[cfg(feature = "tensor-assert")]
        backing.verify_minted_records(&self.handles, &srcs, &arena_info, self.layout)?;
        Ok(self.handles.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arcs(fmt: u8) -> (Arc<Vec<u8>>, Arc<Vec<f32>>, Arc<Vec<u8>>) {
        (
            Arc::new(vec![1u8, 2, 3, 4]),
            Arc::new(vec![1.0f32, 2.0]),
            Arc::new(vec![fmt; 4]),
        )
    }

    fn queued(fmt: u8) -> Queued {
        let (pal, scale, tags) = arcs(fmt);
        Queued {
            gids: HeadGids::uniform(ChunkGid::detached(7), 1),
            k_pal: Arc::clone(&pal),
            v_pal: pal,
            k_scale: Arc::clone(&scale),
            v_scale: scale,
            k_fmt: Arc::clone(&tags),
            v_fmt: tags,
        }
    }

    /// **The record-input comparison is by CONTENTS, not by `Arc` identity.**
    ///
    /// `RecordMint::mint` memoises per gid allocation and throws away every later holder's
    /// inputs, so the `debug_assert` guarding that has to hold for the ordinary case — two
    /// holders of one chunk carrying separate `Arc`s over equal data. Compared by pointer
    /// it would fire on every shared chunk and the assertion would be turned off, which is
    /// how a guard stops guarding.
    #[test]
    fn equal_record_inputs_match_across_distinct_arcs() {
        let q = queued(9);
        let (pal, scale, tags) = arcs(9);
        assert!(
            q.matches(&RecordInputs {
                k_pal: &pal,
                v_pal: &pal,
                k_scale: &scale,
                v_scale: &scale,
                k_fmt: &tags,
                v_fmt: &tags,
            }),
            "separate Arcs over equal bytes are the same record body",
        );
    }

    /// And a real disagreement is caught — the case that would hand the second holder a
    /// record describing the first's format tags, decoding quantized bytes at the wrong
    /// width with nothing to fault.
    #[test]
    fn differing_format_tags_do_not_match() {
        let q = queued(9);
        let (pal, scale, tags) = arcs(11);
        assert!(!q.matches(&RecordInputs {
            k_pal: &pal,
            v_pal: &pal,
            k_scale: &scale,
            v_scale: &scale,
            k_fmt: &tags,
            v_fmt: &tags,
        }));
    }
}
