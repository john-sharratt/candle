//! Per-chunk KV-head metadata record pool.
//!
//! A sealed chunk's KV-head metadata (palette maps, formats, outer scales, and
//! the resolved per-palette device pointers — the `KvHead[n_kv_head]` record the
//! attention kernels read) is constant for as long as the chunk is resident, and
//! is *shared* by every slot that references the chunk. Rather than rebuild and
//! re-upload it per layer per forward, the record lives once in a device-resident
//! slab and travels with the chunk.
//!
//! **A record is an arena slot** (`docs/vram_span_partition.md` §8), so this module no
//! longer owns storage or an allocator. [`MetaGid`] wraps a
//! [`ChunkGid`](super::gid_pool::ChunkGid): cloning bumps the slot's refcount so every
//! holder of a chunk resolves to the same record, and the last drop frees it. What is
//! left here is the record *layout* — [`chunk_record_bytes`], [`band_ptr_offset`] and
//! [`serialize_kv_heads`] — plus the handle.
//!
//! **A record also holds the bands it names**, and that is the property the rest of the
//! cache leans on: its pointer words are raw addresses, so without a reference to the
//! gids they were derived from the record is a dead copy of a fact somebody else owns —
//! which is what let a KV compaction rewrite the gids, free the slots, and leave records
//! reading another chunk's K/V. See [`MetaGid`] and `compact_backings`.
//!
//! It used to own growable `CudaSlice<u8>` slabs with a refcount table per slab: a
//! second allocator, over ground the span partition could not account for, whose records
//! were serialized on the host and uploaded per run. The bytes are now written on the
//! device by `backing::fill_records_on_device` from a descriptor table.
//!
//! [`serialize_kv_heads`] remains the **reference**: it still builds the inline
//! per-slice heads in `gpu_chunks`, and the fill kernel is held against it byte-for-byte
//! by test. The layout is encoded in four places already (§8) and must not gain a fifth
//! opinion.

use candle::Device;

use crate::kv_cache::arena_table::{ArenaFormatTag, ResolvedArenaInfo, N_PALETTE};
use crate::kv_cache::chunked::gid_pool::ChunkGid;
use crate::kv_cache::chunked::head_gids::HeadGids;

/// Bytes of one head's serialized `KvHead` record at `head_dim`, matching the
/// CUDA layout in `slot_types.cuh` (`kv_head_byte_size`): `head_dim/2 + 104`.
///
/// Layout: `k_pal[head_dim/4] + v_pal[head_dim/4] + k_ptr[4]·8 + v_ptr[4]·8 +
/// k_fmt[4] + v_fmt[4] + k_scale[4]·4 + v_scale[4]·4`.
///
/// `n_palette` is the per-head band count: 4 (GQA / palette4) or the
/// single-latent `LATENT_N_BANDS`. The pal_map stays `head_dim/4` bytes per side
/// (2-bit packing density — independent of the band count); only the
/// pointer/fmt/scale block scales with `n_palette` (`n_palette * 26`).
pub(crate) fn kv_head_record_bytes(head_dim: usize, n_palette: usize) -> usize {
    (head_dim / 4) * 2 + n_palette * 26
}

/// Bytes of one chunk's full `KvHead[n_kv_head]` record.
pub(crate) fn chunk_record_bytes(n_kv_head: usize, head_dim: usize, n_palette: usize) -> usize {
    n_kv_head * kv_head_record_bytes(head_dim, n_palette)
}

/// Byte offset of one band's 8-byte device pointer inside a chunk record.
///
/// The inverse of the pointer writes in [`serialize_kv_heads`]: given a band, this says
/// which word of the resident record names it. Two readers, both diagnostic — the
/// integrity check, which reads a record back and compares that word against the band's
/// own address, and the test that holds the device fill kernel against the host
/// serializer. **Nothing in production rewrites a record**; it used to be how a KV
/// compaction patched one in place, and that is exactly what corrupted K/V (see
/// `compact_backings`).
///
/// `is_value` selects the V pointer over the K pointer. `p` is the band (palette)
/// index; the gid that feeds this word is `gids[h * n_palette * 2 + p * 2 +
/// is_value]`, which is the stride the record itself indexes at (see the note on
/// the GID slice in [`serialize_kv_heads`] — it is `n_palette * 2` per head, not
/// the global `GIDS_PER_HEAD`).
///
/// Held against the serializer by
/// [`band_ptr_offset_agrees_with_the_serializer`](tests), which builds a real
/// record and reads each pointer back through this — an independent copy of the
/// layout arithmetic would be exactly the kind of second opinion that drifts.
#[cfg_attr(
    not(all(feature = "cuda", feature = "tensor-assert")),
    allow(dead_code)
)]
pub(crate) fn band_ptr_offset(
    h: usize,
    p: usize,
    is_value: bool,
    head_dim: usize,
    n_palette: usize,
) -> usize {
    let pal_bytes = head_dim / 4;
    // Per head: k_pal, v_pal, then k_ptr[n_palette], then v_ptr[n_palette].
    h * kv_head_record_bytes(head_dim, n_palette)
        + pal_bytes * 2
        + (usize::from(is_value) * n_palette + p) * 8
}

/// One chunk's contribution to a `KvHead[n_kv_head]` record, borrowed from the
/// chunk that owns it.
///
/// Everything here travels **with the chunk**. In particular `k_fmt`/`v_fmt`
/// are the band format tags: an arena under size classes holds whatever fits
/// its stride, so it can no longer say how to decode a slot and the chunk must
/// (`docs/archived/arena_unification.md` principle 8). The arena is consulted only for
/// the band's *address*.
/// Read only on CUDA: without a device there are no records, so `build_meta_records`
/// returns `None` for every chunk and nothing consumes these fields. The struct itself
/// still has to exist — it is in that function's signature, and its non-CUDA caller
/// (`alloc_sealed_blocks_bulk`) is not gated.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy)]
pub(crate) struct ChunkRecordSrc<'a> {
    /// The chunk's `(head, palette, K/V)` gid grid.
    pub gids: &'a HeadGids,
    /// Packed K palette maps, `n_kv_head·(head_dim/4)` bytes. Empty ⇒ identity routing.
    pub k_pal: &'a [u8],
    /// Packed V palette maps, same layout as `k_pal`.
    pub v_pal: &'a [u8],
    /// Outer K scales, `n_kv_head·N_PALETTE` f32s. Empty ⇒ unity.
    pub k_scale: &'a [f32],
    /// Outer V scales, same layout as `k_scale`.
    pub v_scale: &'a [f32],
    /// K band format tags ([`ArenaFormatTag::as_u8`]), `n_kv_head·N_PALETTE`
    /// entries in `[h·N_PALETTE + p]` order.
    pub k_fmt: &'a [u8],
    /// V band format tags, same layout as `k_fmt`.
    pub v_fmt: &'a [u8],
}

/// Serialize a chunk's `KvHead[n_kv_head]` record into `dst` (length must equal
/// [`chunk_record_bytes`]). This is the resident-record body — identical
/// byte-for-byte to the per-head portion of the decode/prefill inline-head
/// serialization, just lifted out of the per-slice `TokenSlice` header.
///
/// The 8 pointers per head resolve each `(head, palette, K/V)` GID against
/// `arena_info` as `base_ptr + chunk_idx·chunk_byte_stride` — the
/// location-dependent bytes a migration/defrag re-patches. The format tags
/// beside them come from `src`, not from `arena_info`.
/// CUDA-only, for the same reason [`ChunkRecordSrc`] is: its consumers are the inline
/// per-slice header builder in `gpu_chunks` and the test that holds the fill kernel
/// against it, both of which need a device.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn serialize_kv_heads(
    dst: &mut [u8],
    src: &ChunkRecordSrc<'_>,
    n_kv_head: usize,
    head_dim: usize,
    n_palette: usize,
    arena_info: &[ResolvedArenaInfo],
) {
    let ChunkRecordSrc {
        gids,
        k_pal,
        v_pal,
        k_scale,
        v_scale,
        k_fmt,
        v_fmt,
    } = *src;
    debug_assert!(
        k_fmt.len() >= n_kv_head * n_palette && v_fmt.len() >= n_kv_head * n_palette,
        "band format tags must cover every (head, palette): got k {} v {}, need {}",
        k_fmt.len(),
        v_fmt.len(),
        n_kv_head * n_palette
    );
    debug_assert!(
        head_dim >= 4,
        "head_dim must be >= 4 for 2-bit pal_map packing"
    );
    debug_assert_eq!(
        dst.len(),
        chunk_record_bytes(n_kv_head, head_dim, n_palette),
        "record dst must be exactly chunk_record_bytes"
    );
    let pal_bytes = head_dim / 4;
    // pal_map identity uses the 2-bit density (N_PALETTE), NOT the band count —
    // the map is unused on the single-latent identity-only path and cannot name
    // >4 bands anyway; only the pointer/fmt/scale block below scales to
    // `n_palette`. GID stride per head is `n_palette*2` (K,V per band).
    let sub_hd = (head_dim / N_PALETTE).max(1);
    let stride = n_palette * 2;
    let mut pos = 0usize;

    macro_rules! put {
        ($b:expr) => {{
            let b: &[u8] = $b;
            dst[pos..pos + b.len()].copy_from_slice(b);
            pos += b.len();
        }};
    }

    for h in 0..n_kv_head {
        // Palette maps: populated slice when present, else identity routing
        // (matches `KvHeadHost::from_gids` / live ChunkWindow identity bytes).
        let k_pal_head = k_pal.get(h * pal_bytes..(h + 1) * pal_bytes);
        let v_pal_head = v_pal.get(h * pal_bytes..(h + 1) * pal_bytes);
        match k_pal_head {
            Some(s) => put!(s),
            None => {
                // Identity routing ORs into dst, so the target bytes must start
                // clean — `dst` may be a reused buffer (the decode pinned buffer
                // preserves bytes across forwards).
                dst[pos..pos + pal_bytes].fill(0);
                for d in 0..head_dim {
                    let pal_idx = ((d / sub_hd).min(N_PALETTE - 1)) as u8;
                    dst[pos + d / 4] |= pal_idx << ((d % 4) * 2);
                }
                pos += pal_bytes;
            }
        }
        match v_pal_head {
            Some(s) => put!(s),
            None => {
                dst[pos..pos + pal_bytes].fill(0);
                for d in 0..head_dim {
                    let pal_idx = ((d / sub_hd).min(N_PALETTE - 1)) as u8;
                    dst[pos + d / 4] |= pal_idx << ((d % 4) * 2);
                }
                pos += pal_bytes;
            }
        }

        let mut k_ptr = vec![0u64; n_palette];
        let mut v_ptr = vec![0u64; n_palette];
        // `Invalid`, never a real format. A band whose tag was not recorded has
        // no known layout, and every other unrecorded-tag path in the cache
        // resolves to `Invalid` precisely so the kernel's format check refuses
        // it. Defaulting to a *float* tag instead would hand quantized bytes to
        // the dispatch as BF16 and decode them as floats — silently wrong
        // output rather than a failure. Producers always fill these
        // (`n_kv_head * n_palette` entries at every call site), so this is the
        // unreachable branch, which is exactly why it must fail loudly if it
        // ever becomes reachable.
        let mut k_tag = vec![ArenaFormatTag::Invalid.as_u8(); n_palette];
        let mut v_tag = vec![ArenaFormatTag::Invalid.as_u8(); n_palette];
        let tag_base = h * n_palette;
        // Index the flat GID slice at the record's own stride (n_palette*2), so
        // an 8-band single-latent head reads its 16 GIDs correctly regardless of
        // the global GIDS_PER_HEAD (which stays 4-palette for GQA).
        for p in 0..n_palette {
            let k_gid = &gids.as_slice()[h * stride + p * 2];
            let v_gid = &gids.as_slice()[h * stride + p * 2 + 1];
            // **A non-resident arena leaves the pointer null, rather than forming one
            // from a zero base.** `resolve_arena_info` reports `base_ptr: 0` with a
            // non-zero stride for a CPU/warm arena, so without this guard a band in one
            // got `chunk_idx * stride` — a small, non-null, entirely bogus address that a
            // kernel would happily dereference. The fill kernel has always guarded it
            // (`e.base != 0 && e.stride > 0`); this is the reference catching up, and the
            // two must agree because one is tested against the other.
            if let Some(ai) = arena_info.get(k_gid.arena_idx()) {
                if ai.base_ptr != 0 && ai.chunk_byte_stride > 0 {
                    k_ptr[p] = ai.base_ptr + k_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                }
            }
            if let Some(ai) = arena_info.get(v_gid.arena_idx()) {
                if ai.base_ptr != 0 && ai.chunk_byte_stride > 0 {
                    v_ptr[p] = ai.base_ptr + v_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                }
            }
            if let Some(&t) = k_fmt.get(tag_base + p) {
                k_tag[p] = t;
            }
            if let Some(&t) = v_fmt.get(tag_base + p) {
                v_tag[p] = t;
            }
        }
        for &ptr in &k_ptr {
            put!(&ptr.to_le_bytes());
        }
        for &ptr in &v_ptr {
            put!(&ptr.to_le_bytes());
        }
        put!(&k_tag);
        put!(&v_tag);
        let scale_base = h * n_palette;
        for p in 0..n_palette {
            let s = k_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
        for p in 0..n_palette {
            let s = v_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
    }
}

/// RAII handle to one per-chunk KV-head metadata record.
///
/// **A record is an arena slot, so the handle is an arena gid.** The semantics a
/// record needs are the ones [`ChunkGid`] already has: cloning bumps the slot's
/// refcount, so every slot referencing the same physical chunk resolves to the
/// *same* record, and dropping the last clone frees it. That is why `Clone` and
/// `Drop` are derived here rather than written — the refcount table this used to
/// keep of its own (`MetaSlabRefcounts` over `CudaSlice<u8>` slabs) was a second
/// implementation of the allocator's, in ground the span partition could not see.
/// See `docs/vram_span_partition.md` §8.
///
/// Stored alongside `HeadGids` on `ChunkWindow` / `SealedChunk`, so a chunk's
/// record shares the chunk's lifetime through `#[derive(Clone)]`.
///
/// # The record owns the bands it names
///
/// [`Self::bands`] is a clone of the very `HeadGids` the record's pointer words were
/// serialized from, so **a record can never outlive the slots it describes**. That is
/// a structural property, not a checked one: the allocator cannot reissue a band slot
/// while any record still points at it, because the record is one of its refcount
/// holders.
///
/// Without it the record was a *dead copy* of a fact the gid owned — the address
/// `base_ptr + chunk_idx · chunk_byte_stride`, stored with no reference to the thing
/// it was derived from — and nothing structurally stopped the copy outliving its
/// subject. That is the general rule `CLAUDE.md` states as "a captured device address
/// is invalidated by anything that moves what it names, and a reference count is not
/// a location", and it is what made a KV compaction corrupt K/V: the pass rewrote the
/// holders' gids, the source lost its last refcount, the allocator reissued that
/// ground, and records still naming it read another chunk's K/V — finite, plausibly
/// shaped and wrong, with nothing anywhere to fault.
///
/// One `Arc` bump per record, eight bytes on the handle, nothing per band: the clone
/// shares the same `ChunkGid` objects, so each refcount lands on the arena the gid
/// came from rather than on a reconstruction of it.
#[derive(Clone, Debug)]
pub struct MetaGid {
    /// The record's arena slot, which is what refcounts it. A detached record holds
    /// a detached gid rather than nothing, so cloning and dropping are counted the
    /// same way on both — an `Option` here silently made `strong_count` a constant
    /// for detached records and stopped counting their clones.
    gid: ChunkGid,
    /// The bands this record's pointer words name, held so it cannot outlive them.
    ///
    /// See the type note. `None` only for a detached record, which names nothing.
    bands: Option<HeadGids>,
    /// Cached device address of this record — `arena_base + slot · stride`,
    /// resolved at allocation.
    ///
    /// **Stable for the handle's life, and every paged kernel depends on that.**
    /// It is written raw into each slice header's `kvheads_ptr` word and
    /// dereferenced by `reinterpret_cast` with no indirection
    /// (`paged-decode/slot_types.cuh`), so nothing may relocate a held record —
    /// which is exactly why the record arenas are excluded from the compaction
    /// census. `0` ⇒ no device residence (a host-only pool), in which case the
    /// chunk has no record at all.
    device_addr: u64,
}

impl MetaGid {
    /// The slot's raw gid. Forwarded rather than stored beside it: two copies of one
    /// value can only ever disagree.
    #[inline]
    pub fn raw(&self) -> i64 {
        self.gid.raw()
    }

    /// Cached device address of this record (0 if not device-resident).
    #[inline]
    pub fn device_addr(&self) -> u64 {
        self.device_addr
    }

    /// The arena slot this record occupies.
    #[inline]
    pub fn gid(&self) -> &ChunkGid {
        &self.gid
    }

    /// Current number of holders of this record's slot. `1` = uniquely owned
    /// (safe to rewrite in place); `> 1` = shared across slots.
    #[inline]
    pub fn strong_count(&self) -> usize {
        self.gid.strong_count()
    }

    /// The bands this record's pointer words name, or `None` for a detached record.
    ///
    /// Held rather than derived: this is the clone that makes the record's addresses
    /// outlive-proof. A caller comparing it against a chunk's own gids is comparing
    /// two handles to one allocation, not two derivations of one address.
    #[inline]
    pub fn bands(&self) -> Option<&HeadGids> {
        self.bands.as_ref()
    }

    /// A detached record with no arena backing, for tests and diagnostic chunks
    /// that never resolve a real device record. Mirrors `ChunkGid::detached`.
    pub fn detached(id: i64) -> Self {
        Self {
            gid: ChunkGid::detached(id),
            bands: None,
            device_addr: 0,
        }
    }

    /// Wrap a freshly allocated record slot around the bands it will describe.
    ///
    /// `bands` must be the same `HeadGids` the record's pointer words are serialized
    /// from — that is the whole contract, and it is what
    /// `a_record_holds_the_bands_it_names` pins.
    pub(super) fn from_slot(gid: ChunkGid, bands: HeadGids, device_addr: u64) -> Self {
        Self {
            gid,
            bands: Some(bands),
            device_addr,
        }
    }
}

/// The record geometry for one backing group, and whether records are resident.
///
/// **No storage and no allocator.** Records are arena slots now
/// (`docs/vram_span_partition.md` §8), so the gid pool allocates them and the arena
/// owns their bytes; what is left here is the one number both sides have to agree on
/// — the serialized record size — and the device it is resident on. It used to own
/// growable `CudaSlice<u8>` slabs with a refcount table per slab, which was a second
/// allocator over ground the span partition could not account for.
#[derive(Debug)]
pub struct MetaPool {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    device: Device,
}

impl MetaPool {
    /// The record residence for one backing. Storage is the arena's and nothing is
    /// allocated here, so the only thing left to hold is the device.
    ///
    /// **There is no stored record size any more.** There used to be one, resized after
    /// construction by `set_single_latent` because the single latent carries twice GQA's
    /// bands. `build_meta_records` now derives the size from the backing's *live*
    /// `n_palette()` on every call and hands it to `ArenaKey::for_records`, so the
    /// resize has nothing to update and the value it would have cached cannot go stale
    /// against the geometry.
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// True when this pool backs records with device memory (a real CUDA
    /// device). A CPU/host-only pool keeps refcounts only; its records have no
    /// readable address, so a handle from it must be treated as non-resident.
    pub fn is_device_resident(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self.device, Device::Cuda(_))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::arena_table::ArenaFormatTag;
    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    /// A record's lifetime **is** its arena slot's.
    ///
    /// The allocator itself is the gid pool's and is tested there; what is this
    /// module's to prove is that wrapping a `ChunkGid` preserved the semantics the
    /// hand-rolled refcount table used to provide — a clone shares one slot so every
    /// holder of a chunk resolves to the same record, and the slot survives until the
    /// last clone goes. Asserted on `ChunkGid::detached`, which carries the same
    /// refcount machinery without needing a device.
    #[test]
    fn a_clone_shares_the_record_slot_and_the_last_drop_releases_it() {
        let bands = HeadGids::uniform(ChunkGid::detached(11), 1);
        let a = MetaGid::from_slot(ChunkGid::detached(7), bands, 0xdead_0000);
        assert_eq!(a.raw(), 7, "the handle's id is its slot's");
        assert_eq!(a.device_addr(), 0xdead_0000);
        assert_eq!(a.strong_count(), 1);

        let b = a.clone();
        let c = a.clone();
        assert_eq!((b.raw(), c.raw()), (7, 7), "a clone names the same slot");
        assert_eq!(
            (b.device_addr(), c.device_addr()),
            (0xdead_0000, 0xdead_0000),
            "and carries the same address — the kvheads_ptr every kernel dereferences",
        );
        assert_eq!(a.strong_count(), 3);

        drop(b);
        assert_eq!(a.strong_count(), 2);
        drop(c);
        assert_eq!(a.strong_count(), 1, "the slot is still held by `a`");
    }

    /// **A record holds the bands it names, so it cannot outlive them.**
    ///
    /// This is the property the whole compaction design now rests on: the pass rewrites
    /// holders' gids and leaves records alone, which is only safe because a record is
    /// itself a refcount holder of the slots its pointer words address. Drop every
    /// *chunk* reference to a band and the record's keeps the slot alive; drop the
    /// record too and it goes.
    ///
    /// Asserted on `ChunkGid::detached`, which carries the same refcount machinery
    /// without needing a device — the strong count is the observable, and it is the one
    /// the allocator consults before reissuing ground.
    #[test]
    fn a_record_holds_the_bands_it_names() {
        let band = ChunkGid::detached(41);
        // Counted as deltas, not absolutes: `HeadGids::uniform` puts a clone of the gid in
        // every `(head, palette, K/V)` slot, so the raw numbers are a property of the
        // geometry while what is under test is *who is holding*.
        let unheld = band.strong_count();
        let bands = HeadGids::uniform(band.clone(), 1);
        let held = band.strong_count();
        assert!(held > unheld, "the gid vector holds the band");

        // **The record shares the vector rather than copying the gids.** `HeadGids` is
        // `Arc<Vec<ChunkGid>>`, so this costs one `Arc` bump and nothing per band — and
        // the band's own count does not move, because the holder of that refcount is the
        // vector, which is now held twice.
        let record = MetaGid::from_slot(ChunkGid::detached(7), bands.clone(), 0xfeed_0000);
        assert_eq!(
            band.strong_count(),
            held,
            "no gid is cloned, only the vector"
        );
        let named = record.bands().expect("a record from_slot names bands");
        assert!(
            named.is_same_alloc(&bands),
            "and it is the SAME allocation the chunk holds, not a copy of it",
        );

        // The chunk lets go — which is exactly what a compaction's gid rewrite does when
        // it installs a fresh `HeadGids` over the old one.
        drop(bands);
        assert_eq!(
            band.strong_count(),
            held,
            "the record still holds the vector, so the band slot is still allocated and \
             the allocator cannot reissue this ground",
        );

        drop(record);
        assert_eq!(
            band.strong_count(),
            unheld,
            "and only when the record goes too is the band free",
        );
    }

    /// A detached record names nothing, and says so rather than pretending to bands.
    #[test]
    fn a_detached_record_names_no_bands() {
        assert!(MetaGid::detached(-1).bands().is_none());
    }

    /// **Every geometry in the model table must land on a record-stride rung that holds
    /// its record.** `ArenaKey::for_records` panics when none does, so this is the test
    /// that would catch a new checkpoint outgrowing `RECORD_STRIDES` — at build time in
    /// CI rather than at the first seal on the box that loads it.
    ///
    /// The stride properties themselves (power of two, inside the gid namespace) belong
    /// to the key and are asserted in `arena_tests`; what is this module's concern is
    /// that the *record sizes it computes* are covered.
    #[test]
    fn every_production_geometry_fits_a_record_stride_rung() {
        use crate::kv_cache::arena_table::ArenaLocation;
        use crate::kv_cache::chunked::arena::ArenaKey;

        // (n_kv_head, head_dim, n_palette): GQA at both head dims, the smallest
        // geometry the packing allows, and the single latent's 16-band width.
        for (n_kv_head, head_dim, n_palette) in [
            (2usize, 128usize, 4usize),
            (8, 128, 4),
            (8, 256, 4),
            (1, 4, 4),
            (8, 128, 16),
            (64, 512, 16),
        ] {
            let rb = chunk_record_bytes(n_kv_head, head_dim, n_palette);
            let key = ArenaKey::for_records(ArenaLocation::Gpu, rb);
            assert!(
                key.slot_stride() >= rb,
                "a {n_kv_head}x{head_dim}x{n_palette} record is {rb} B but its slot is \
                 {} B — every record would overrun its slot",
                key.slot_stride(),
            );
            assert!(
                key.chunks() > 0,
                "a {rb} B record leaves no slots in a region",
            );
        }
    }

    #[test]
    fn detached_record_has_no_pool() {
        let d = MetaGid::detached(-1);
        assert_eq!(d.raw(), -1);
        let d2 = d.clone();
        assert_eq!(d.strong_count(), 2);
        drop(d2);
        assert_eq!(d.strong_count(), 1);
    }

    /// Byte-exact golden for the record body. HD=4, n_kv_head=1, single arena
    /// (one base_ptr/stride), identity palette (empty ⇒ identity), unity scales.
    /// Asserts the exact 168-bytes-at-HD4 = `4/2 + 104 = 106`-byte layout.
    ///
    /// **The arena is deliberately stamped with the wrong formats.** Its tags
    /// say BF16 on both sides while the chunk says Q8_0/Q4_0, and the golden
    /// demands the chunk's answer. That makes this test the direct regression
    /// for format ownership: a record built from arena state fails it.
    #[test]
    fn serialize_kv_heads_golden_takes_formats_from_the_chunk() {
        let head_dim = 4usize;
        let n_kv_head = 1usize;
        let rec = chunk_record_bytes(n_kv_head, head_dim, N_PALETTE);
        assert_eq!(rec, head_dim / 2 + 104); // 106

        // One arena, base_ptr=0x1000, stride=512. All 8 sub-band GIDs point at
        // arena 0, chunk_idx 0 → every pointer == base_ptr.
        let gids = HeadGids::uniform(ChunkGid::detached(0), n_kv_head);
        let arena_info = vec![ResolvedArenaInfo {
            base_ptr: 0x1000,
            chunk_byte_stride: 512,
            chunk_capacity: u32::MAX,
        }];
        let k_fmt = vec![ArenaFormatTag::Q8_0.as_u8(); N_PALETTE];
        let v_fmt = vec![ArenaFormatTag::Q4_0.as_u8(); N_PALETTE];

        let mut dst = vec![0u8; rec];
        serialize_kv_heads(
            &mut dst,
            &ChunkRecordSrc {
                gids: &gids,
                k_pal: &[],
                v_pal: &[],
                k_scale: &[],
                v_scale: &[],
                k_fmt: &k_fmt,
                v_fmt: &v_fmt,
            },
            n_kv_head,
            head_dim,
            N_PALETTE,
            &arena_info,
        );

        let mut exp: Vec<u8> = Vec::new();
        // k_pal[1]: identity for HD4 → dims 0,1,2,3 → palettes 0,1,2,3 (sub_hd=1)
        // packed: (3<<6)|(2<<4)|(1<<2)|0 = 0xE4
        exp.push(0xE4);
        // v_pal[1]: same
        exp.push(0xE4);
        // k_ptr[4]: all base_ptr 0x1000 (chunk_idx 0)
        for _ in 0..4 {
            exp.extend_from_slice(&0x1000u64.to_le_bytes());
        }
        // v_ptr[4]: same
        for _ in 0..4 {
            exp.extend_from_slice(&0x1000u64.to_le_bytes());
        }
        // k_fmt[4] = Q8_0 tag, v_fmt[4] = Q4_0 tag
        for _ in 0..4 {
            exp.push(ArenaFormatTag::Q8_0.as_u8());
        }
        for _ in 0..4 {
            exp.push(ArenaFormatTag::Q4_0.as_u8());
        }
        // k_scale[4]=1.0, v_scale[4]=1.0
        for _ in 0..4 {
            exp.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..4 {
            exp.extend_from_slice(&1.0f32.to_le_bytes());
        }
        assert_eq!(exp.len(), rec);
        assert_eq!(dst, exp, "record body must match exact KvHead byte layout");
    }

    /// Two sub-bands in two different arenas resolve to two different pointers
    /// within one head (the multi-arena reality the design must support).
    #[test]
    fn serialize_kv_heads_multi_arena_pointers() {
        let head_dim = 4usize;
        let n_kv_head = 1usize;
        let rec = chunk_record_bytes(n_kv_head, head_dim, N_PALETTE);

        // GID layout per head: slot = palette*2 + is_value, over N_PALETTE=4.
        // Put K-palette-0 in arena 0 chunk 1, K-palette-1 in arena 1 chunk 2.
        use crate::kv_cache::chunked::head_gids::GIDS_PER_HEAD;
        let stride = crate::kv_cache::chunked::types::GID_STRIDE as i64;
        let mut raw = vec![0i64; GIDS_PER_HEAD * n_kv_head];
        // k_gid_pal(0,0) is slot 0; k_gid_pal(0,1) is slot 2 (palette*2+0).
        raw[0] = 1; // arena 0 (base 0), chunk 1
        raw[2] = stride + 2; // arena 1, chunk 2
        let gids = HeadGids::from_vec(raw.iter().map(|&r| ChunkGid::detached(r)).collect());
        let arena_info = vec![
            ResolvedArenaInfo {
                base_ptr: 0x1000,
                chunk_byte_stride: 256,
                chunk_capacity: u32::MAX,
            },
            ResolvedArenaInfo {
                base_ptr: 0x9000,
                chunk_byte_stride: 128,
                chunk_capacity: u32::MAX,
            },
        ];
        let k_fmt = vec![ArenaFormatTag::Q8_0.as_u8(); N_PALETTE];
        let v_fmt = vec![ArenaFormatTag::Q8_0.as_u8(); N_PALETTE];
        let mut dst = vec![0u8; rec];
        serialize_kv_heads(
            &mut dst,
            &ChunkRecordSrc {
                gids: &gids,
                k_pal: &[],
                v_pal: &[],
                k_scale: &[],
                v_scale: &[],
                k_fmt: &k_fmt,
                v_fmt: &v_fmt,
            },
            n_kv_head,
            head_dim,
            N_PALETTE,
            &arena_info,
        );
        // k_ptr[0] = 0x1000 + 1*256 = 0x1100; k_ptr[1] = 0x9000 + 2*128 = 0x9100.
        let kptr0 = u64::from_le_bytes(dst[2..10].try_into().unwrap());
        let kptr1 = u64::from_le_bytes(dst[10..18].try_into().unwrap());
        assert_eq!(kptr0, 0x1000 + 256);
        assert_eq!(kptr1, 0x9000 + 2 * 128);
    }

    /// **[`band_ptr_offset`] must agree with the serializer, not with a copy of
    /// its reasoning.**
    ///
    /// A KV compaction patches a resident record by storing 8 bytes at the offset
    /// this function names. If the arithmetic drifts from `serialize_kv_heads` the
    /// patch writes into a format tag or a scale — which does not fault, because
    /// the record is valid memory, and surfaces as a decode reading quantized
    /// bytes with the wrong tag. So the test builds a REAL record with a distinct
    /// address per band and reads every band back through the offset.
    ///
    /// Several heads and a head_dim above the 4 of the tests above, because
    /// `pal_bytes = head_dim / 4` is the term a per-head stride bug hides behind.
    /// Run at **both** band counts, because the band count is per backing.
    ///
    /// `n_palette()` answers `N_PALETTE` for GQA and `LATENT_N_BANDS` for the single
    /// latent, and it drives this record layout. A compaction that took the GQA
    /// constant instead of asking the backing computed every offset for a quarter of
    /// the bands on a latent backing — writing correct addresses into the wrong
    /// words, over the palette, format and scale fields of an earlier head, and
    /// leaving the upper bands naming vacated slots. Pinning only `N_PALETTE` here
    /// is what let that pass: the arithmetic was right for the count the test used.
    #[test]
    fn band_ptr_offset_agrees_with_the_serializer() {
        offsets_agree_at(N_PALETTE);
    }

    #[test]
    fn band_ptr_offset_agrees_with_the_serializer_on_the_single_latent() {
        offsets_agree_at(crate::kv_cache::arena_table::LATENT_N_BANDS);
    }

    fn offsets_agree_at(n_palette: usize) {
        let head_dim = 128usize;
        let n_kv_head = 3usize;
        let rec = chunk_record_bytes(n_kv_head, head_dim, n_palette);
        let stride = crate::kv_cache::chunked::types::GID_STRIDE as i64;

        // One distinct arena per band slot, so every pointer in the record is
        // unique and a swapped offset cannot coincidentally match.
        let per_head = n_palette * 2;
        let mut raw = vec![0i64; per_head * n_kv_head];
        let mut arena_info = Vec::new();
        for (slot, r) in raw.iter_mut().enumerate() {
            // arena `slot`, chunk 1 — so the address is base + stride.
            *r = slot as i64 * stride + 1;
            arena_info.push(ResolvedArenaInfo {
                base_ptr: 0x10_0000 + slot as u64 * 0x1000,
                chunk_byte_stride: 64,
                chunk_capacity: u32::MAX,
            });
        }
        let gids = HeadGids::from_vec(raw.iter().map(|&r| ChunkGid::detached(r)).collect());
        let k_fmt = vec![ArenaFormatTag::Q8_0.as_u8(); n_palette * n_kv_head];
        let v_fmt = vec![ArenaFormatTag::Q8_0.as_u8(); n_palette * n_kv_head];
        let mut dst = vec![0u8; rec];
        serialize_kv_heads(
            &mut dst,
            &ChunkRecordSrc {
                gids: &gids,
                k_pal: &[],
                v_pal: &[],
                k_scale: &[],
                v_scale: &[],
                k_fmt: &k_fmt,
                v_fmt: &v_fmt,
            },
            n_kv_head,
            head_dim,
            n_palette,
            &arena_info,
        );

        for h in 0..n_kv_head {
            for p in 0..n_palette {
                for is_value in [false, true] {
                    let slot = h * per_head + p * 2 + usize::from(is_value);
                    let want = arena_info[slot].base_ptr + 64;
                    let off = band_ptr_offset(h, p, is_value, head_dim, n_palette);
                    assert!(
                        off + 8 <= rec,
                        "offset for (h{h}, p{p}, v{is_value}) at {n_palette} bands \
                         runs past the record",
                    );
                    let got = u64::from_le_bytes(dst[off..off + 8].try_into().unwrap());
                    assert_eq!(
                        got, want,
                        "band (h{h}, p{p}, is_value={is_value}) at {n_palette} bands \
                         reads the wrong word: offset {off} holds {got:#x}, the \
                         serializer put {want:#x} there",
                    );
                }
            }
        }
    }

    /// Every band offset is distinct and 8-byte aligned — the property the patch
    /// kernel's sorted-scatter coalescing assumes, and which a stride bug that
    /// happened to stay in bounds would break silently.
    #[test]
    fn band_ptr_offsets_are_distinct_and_aligned() {
        let head_dim = 128usize;
        let n_kv_head = 4usize;
        let mut seen = std::collections::HashSet::new();
        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                for is_value in [false, true] {
                    let off = band_ptr_offset(h, p, is_value, head_dim, N_PALETTE);
                    assert_eq!(off % 8, 0, "band pointers must be 8-byte aligned");
                    assert!(seen.insert(off), "offset {off} is claimed by two bands");
                }
            }
        }
        assert_eq!(seen.len(), n_kv_head * N_PALETTE * 2);
    }

    /// **The fill kernel must produce byte-for-byte what the host serializer produces.**
    ///
    /// This is the test the whole device-side fill rests on. A record is a table of raw
    /// device pointers that thirteen CUDA kernels dereference with `reinterpret_cast`
    /// and no bounds check, so a single wrong byte is another chunk's K/V read as this
    /// one's — finite, plausible, and wrong, surfacing as a NaN many layers later. The
    /// record layout is already encoded in four places (`docs/vram_span_partition.md`
    /// §8); holding the kernel against [`serialize_kv_heads`] is what stops it becoming
    /// a fifth opinion that drifts.
    ///
    /// Covered deliberately: both palette sources (populated bytes and the derived
    /// identity map), both scale sources (populated and unity), several head counts and
    /// band widths, and gids spread across two arenas with *different* strides so a
    /// pointer computed from the wrong extent cannot coincide with the right answer.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_fill_kernel_matches_the_host_serializer_byte_for_byte() {
        use crate::kv_cache::chunked::size_class::GID_STRIDE;
        use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle::cuda_backend::kernels::simple::kv_record_fill as krf;

        let guard = crate::kv_cache::chunked::gpu_test_lock::gpu_serial();
        let Ok(dev @ Device::Cuda(_)) = Device::cuda_if_available(0) else {
            return;
        };
        let Device::Cuda(cuda) = &dev else { return };
        let _ = &guard;

        // Two arenas with unlike strides: a pointer resolved against the wrong one is
        // then numerically distinguishable, which a single-arena fixture cannot show.
        //
        // The third is **not resident** — `base_ptr: 0` with a non-zero stride, which is
        // exactly what `resolve_arena_info` reports for a CPU/warm arena. Both sides must
        // leave such a band's pointer null; the reference used to form
        // `chunk_idx * stride` from it, a small non-null address a kernel would
        // dereference, and the old fixture's all-non-zero bases could not catch it.
        let arena_info = vec![
            ResolvedArenaInfo {
                base_ptr: 0x1_0000,
                chunk_byte_stride: 512,
                chunk_capacity: u32::MAX,
            },
            ResolvedArenaInfo {
                base_ptr: 0x9_0000,
                chunk_byte_stride: 1088,
                chunk_capacity: u32::MAX,
            },
            ResolvedArenaInfo {
                base_ptr: 0,
                chunk_byte_stride: 2048,
                chunk_capacity: u32::MAX,
            },
        ];

        // Every geometry here is 8-byte aligned in its record layout, because the kernel
        // stores band pointers as `uint64_t` and CUDA faults on an unaligned address —
        // `head_dim % 16 == 0` keeps `head_dim / 2` aligned and `n_palette % 4 == 0`
        // keeps the per-head size aligned. `head_dim = 4` (which the host golden test
        // uses, where unaligned writes are legal) puts a pointer at offset 2 and raises
        // `CUDA_ERROR_MISALIGNED_ADDRESS`; `fill_records_on_device` now refuses such a
        // geometry outright rather than faulting mid-seal. 16 is the smallest head_dim
        // that qualifies.
        for (n_kv_head, head_dim, n_palette, with_pal, with_scale) in [
            (1usize, 16usize, N_PALETTE, false, false),
            (2, 128, N_PALETTE, false, false),
            (2, 128, N_PALETTE, true, true),
            (4, 128, N_PALETTE, true, false),
            (3, 256, N_PALETTE, false, true),
            (2, 128, 16, true, true),
        ] {
            let rb = chunk_record_bytes(n_kv_head, head_dim, n_palette);
            let pal_bytes = head_dim / 4;
            let tags = n_kv_head * n_palette;

            // Gids alternate between the two arenas and walk chunk indices, so every
            // band resolves to a distinct address.
            // Bands cycle over all three arenas, so every record exercises the resident
            // pair and the non-resident one.
            let raws: Vec<i64> = (0..n_kv_head * n_palette * 2)
                .map(|i| {
                    let arena = (i % 3) as i64;
                    let chunk = (i / 3) as i64 + 1;
                    arena * GID_STRIDE as i64 + chunk
                })
                .collect();
            let gids = HeadGids::from_vec(raws.iter().map(|&r| ChunkGid::detached(r)).collect());

            let k_pal: Vec<u8> = if with_pal {
                (0..n_kv_head * pal_bytes)
                    .map(|i| (i * 7 + 1) as u8)
                    .collect()
            } else {
                Vec::new()
            };
            let v_pal: Vec<u8> = if with_pal {
                (0..n_kv_head * pal_bytes)
                    .map(|i| (i * 13 + 5) as u8)
                    .collect()
            } else {
                Vec::new()
            };
            let k_scale: Vec<f32> = if with_scale {
                (0..tags).map(|i| 0.5 + i as f32).collect()
            } else {
                Vec::new()
            };
            let v_scale: Vec<f32> = if with_scale {
                (0..tags).map(|i| 1.5 + i as f32 * 2.0).collect()
            } else {
                Vec::new()
            };
            let k_fmt: Vec<u8> = (0..tags)
                .map(|i| ArenaFormatTag::Q8_0.as_u8() + (i % 2) as u8)
                .collect();
            let v_fmt: Vec<u8> = (0..tags)
                .map(|i| ArenaFormatTag::Q4_0.as_u8() + (i % 3) as u8)
                .collect();

            let src = ChunkRecordSrc {
                gids: &gids,
                k_pal: &k_pal,
                v_pal: &v_pal,
                k_scale: &k_scale,
                v_scale: &v_scale,
                k_fmt: &k_fmt,
                v_fmt: &v_fmt,
            };

            // The reference.
            let mut want = vec![0u8; rb];
            serialize_kv_heads(&mut want, &src, n_kv_head, head_dim, n_palette, &arena_info);

            // The kernel, into a device buffer poisoned first so an unwritten byte is a
            // failure rather than an accidental match against the reference's zeros.
            let mut d_rec = cuda.memcpy_stod(&vec![0xABu8; rb]).unwrap();
            let addr_stream = cuda.cuda_stream();
            let rec_addr = {
                let (p, _g) = d_rec.device_ptr_mut(&addr_stream);
                p
            };
            let descs: Vec<i64> = vec![
                rec_addr as i64,
                0,
                if with_pal { 0 } else { -1 },
                0,
                if with_scale { 0 } else { -1 },
            ];
            let extents: Vec<i64> = arena_info
                .iter()
                .flat_map(|r| [r.base_ptr as i64, r.chunk_byte_stride])
                .collect();
            let d_descs = cuda.memcpy_stod(&descs).unwrap();
            let d_gids = cuda.memcpy_stod(&raws).unwrap();
            let d_kpal = cuda.memcpy_stod(&pad_u8(&k_pal)).unwrap();
            let d_vpal = cuda.memcpy_stod(&pad_u8(&v_pal)).unwrap();
            let d_kfmt = cuda.memcpy_stod(&k_fmt).unwrap();
            let d_vfmt = cuda.memcpy_stod(&v_fmt).unwrap();
            let d_kscale = cuda.memcpy_stod(&pad_f32(&k_scale)).unwrap();
            let d_vscale = cuda.memcpy_stod(&pad_f32(&v_scale)).unwrap();
            let d_ext = cuda.memcpy_stod(&extents).unwrap();
            let stream = cuda.cuda_stream();
            {
                let (p_d, _a) = d_descs.device_ptr(&stream);
                let (p_g, _b) = d_gids.device_ptr(&stream);
                let (p_kp, _c) = d_kpal.device_ptr(&stream);
                let (p_vp, _e) = d_vpal.device_ptr(&stream);
                let (p_kf, _f) = d_kfmt.device_ptr(&stream);
                let (p_vf, _h) = d_vfmt.device_ptr(&stream);
                let (p_ks, _i) = d_kscale.device_ptr(&stream);
                let (p_vs, _j) = d_vscale.device_ptr(&stream);
                let (p_x, _k) = d_ext.device_ptr(&stream);
                // SAFETY: every array is device-resident and long enough for the single
                // descriptor's offsets; `rec_addr` names `rb` writable bytes.
                unsafe {
                    krf::run_kv_record_fill(
                        p_d as *const std::ffi::c_void,
                        p_g as *const i64,
                        p_kp as *const u8,
                        p_vp as *const u8,
                        p_kf as *const u8,
                        p_vf as *const u8,
                        p_ks as *const f32,
                        p_vs as *const f32,
                        p_x as *const std::ffi::c_void,
                        arena_info.len() as i32,
                        1,
                        n_kv_head as i32,
                        head_dim as i32,
                        n_palette as i32,
                        GID_STRIDE as i32,
                        ArenaFormatTag::Invalid.as_u8() as i32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            let got = cuda.memcpy_dtov(&d_rec).unwrap();

            assert_eq!(
                got.len(),
                want.len(),
                "{n_kv_head}x{head_dim}x{n_palette}: record length"
            );
            if got != want {
                let at = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .expect("lengths match and contents differ, so a byte differs");
                panic!(
                    "{n_kv_head}x{head_dim}x{n_palette} pal={with_pal} scale={with_scale}: the \
                     kernel and the host serializer disagree at byte {at} of {rb} (kernel \
                     {:#04x}, reference {:#04x}). A record is a table of raw pointers the \
                     attention kernels dereference unchecked, so this is another chunk's K/V \
                     read as this one's.",
                    got[at], want[at],
                );
            }
        }
    }

    /// **Microbench and `ncu` target for the record fill.**
    ///
    /// ```text
    /// cargo test -p candle-nn --features cuda --release --lib \
    ///   kv_cache::chunked::meta_pool::tests::bench_record_fill -- --exact --ignored --nocapture
    ///
    /// ncu --set full --kernel-name kv_record_fill_kernel -o kvrec \
    ///   target/release/deps/candle_nn-<hash>.exe \
    ///   kv_cache::chunked::meta_pool::tests::bench_record_fill --exact --ignored
    /// ```
    ///
    /// What to read, and what the kernel was shaped for:
    ///
    /// - **Occupancy** — the grid is `min(work / 256, 8 × SM)` blocks over a grid-stride
    ///   loop, so achieved occupancy should be near the theoretical limit at every batch
    ///   size. A batch of a few records used to be a few blocks (one per record) and left
    ///   the machine idle; if occupancy is low at large batch sizes now, the cap in
    ///   `kv_record_fill_blocks` is the thing to move.
    /// - **Registers per thread** — `__launch_bounds__(256)` bounds it. If ncu reports
    ///   occupancy limited by registers, the bound is being exceeded and nvcc is spilling.
    /// - **Warp execution efficiency** — the two phases are flattened into one item space
    ///   so a warp is not half-idle in the band phase, which it was when the phases were
    ///   separate loops over different item counts.
    /// - **Store efficiency** — palette bytes go out as `uchar4`. Sub-4-byte sectors here
    ///   mean the vector path is not being taken.
    ///
    /// Bench-only, so `#[ignore]`: it needs the card and says nothing about correctness —
    /// that is
    /// [`the_fill_kernel_matches_the_host_serializer_byte_for_byte`](the_fill_kernel_matches_the_host_serializer_byte_for_byte)'s
    /// job, and the numbers below mean nothing if that one is red.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "microbench / ncu target; needs the card to itself"]
    fn bench_record_fill() {
        use crate::kv_cache::chunked::size_class::GID_STRIDE;
        use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle::cuda_backend::kernels::simple::kv_record_fill as krf;

        let guard = crate::kv_cache::chunked::gpu_test_lock::gpu_serial();
        let Ok(dev @ Device::Cuda(_)) = Device::cuda_if_available(0) else {
            return;
        };
        let Device::Cuda(cuda) = &dev else { return };
        let _ = &guard;

        // Production GQA geometry, and the single latent's wider band count.
        for (n_kv_head, head_dim, n_palette) in [(8usize, 128usize, 4usize), (8, 128, 16)] {
            let rb = chunk_record_bytes(n_kv_head, head_dim, n_palette);
            let bands = n_kv_head * n_palette * 2;
            let tags = n_kv_head * n_palette;
            println!(
                "\n=== record fill: {n_kv_head} heads x HD{head_dim} x {n_palette} bands \
                 ({rb} B/record) ==="
            );
            // A batch of one is the latency case; the wide ones are what a seal pass or a
            // warm→hot elevate actually submits.
            for n_records in [1usize, 8, 64, 512, 4096] {
                let extents: Vec<i64> = vec![0x1_0000, 512, 0x9_0000, 1088];
                let raws: Vec<i64> = (0..n_records * bands)
                    .map(|i| ((i % 2) as i64) * GID_STRIDE as i64 + (i / 2) as i64 % 4096 + 1)
                    .collect();
                let fmt: Vec<u8> = vec![ArenaFormatTag::Q8_0.as_u8(); n_records * tags];
                let mut d_recs = cuda.memcpy_stod(&vec![0u8; n_records * rb]).unwrap();
                let s0 = cuda.cuda_stream();
                let base = {
                    let (p, _g) = d_recs.device_ptr_mut(&s0);
                    p
                };
                // Identity palette and unity scales: the common float-chunk case, and the
                // one the kernel derives instead of shipping.
                let descs: Vec<i64> = (0..n_records)
                    .flat_map(|r| {
                        [
                            (base + (r * rb) as u64) as i64,
                            (r * bands) as i64,
                            -1,
                            (r * tags) as i64,
                            -1,
                        ]
                    })
                    .collect();
                let d_descs = cuda.memcpy_stod(&descs).unwrap();
                let d_gids = cuda.memcpy_stod(&raws).unwrap();
                let d_pad = cuda.memcpy_stod(&vec![0u8; 1]).unwrap();
                let d_fmt = cuda.memcpy_stod(&fmt).unwrap();
                let d_fpad = cuda.memcpy_stod(&vec![0.0f32; 1]).unwrap();
                let d_ext = cuda.memcpy_stod(&extents).unwrap();
                let stream = cuda.cuda_stream();

                let launch = || {
                    let (p_d, _a) = d_descs.device_ptr(&stream);
                    let (p_g, _b) = d_gids.device_ptr(&stream);
                    let (p_p, _c) = d_pad.device_ptr(&stream);
                    let (p_f, _e) = d_fmt.device_ptr(&stream);
                    let (p_s, _h) = d_fpad.device_ptr(&stream);
                    let (p_x, _i) = d_ext.device_ptr(&stream);
                    // SAFETY: as the byte-exact test — every array outlives the launch and
                    // covers the offsets the descriptors name.
                    unsafe {
                        krf::run_kv_record_fill(
                            p_d as *const std::ffi::c_void,
                            p_g as *const i64,
                            p_p as *const u8,
                            p_p as *const u8,
                            p_f as *const u8,
                            p_f as *const u8,
                            p_s as *const f32,
                            p_s as *const f32,
                            p_x as *const std::ffi::c_void,
                            2,
                            n_records as i32,
                            n_kv_head as i32,
                            head_dim as i32,
                            n_palette as i32,
                            GID_STRIDE as i32,
                            ArenaFormatTag::Invalid.as_u8() as i32,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                };

                // Warm up, then time. `synchronize` only at the ends, so the figure is the
                // kernel's and not a fence per iteration.
                for _ in 0..10 {
                    launch();
                }
                dev.synchronize().unwrap();
                const ITERS: usize = 200;
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    launch();
                }
                dev.synchronize().unwrap();
                let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let bytes = (n_records * rb) as f64;
                println!(
                    "  {n_records:>5} records  {us:>8.2} us/call  {:>8.2} GB/s written  \
                     {:>10.0} records/s",
                    bytes / (us * 1e3),
                    n_records as f64 / (us * 1e-6),
                );
            }
        }
    }

    /// `memcpy_stod` needs something to copy; a derived-only input is never read by the
    /// kernel because its descriptor offset is -1.
    #[cfg(feature = "cuda")]
    fn pad_u8(v: &[u8]) -> Vec<u8> {
        if v.is_empty() {
            vec![0u8]
        } else {
            v.to_vec()
        }
    }

    #[cfg(feature = "cuda")]
    fn pad_f32(v: &[f32]) -> Vec<f32> {
        if v.is_empty() {
            vec![0.0f32]
        } else {
            v.to_vec()
        }
    }
}
