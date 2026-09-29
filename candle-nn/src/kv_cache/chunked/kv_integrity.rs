//! Whether a slot's K/V is still the K/V it was, and whether everything that
//! references it still legitimately does.
//!
//! # Why this exists
//!
//! A chunk's gid *is* its location, so every operation that moves a chunk has to
//! update every party that recorded where it was. There are several such parties —
//! the backings' block tables, the substrate's residences, the projection caches,
//! and the resident `KvHead` records the paged kernels dereference — and the list is
//! maintained by hand. Each defect found in this area has been one party somebody
//! forgot, and the symptom is always the same: a reader gets another sequence's K/V,
//! finite and plausibly shaped and wrong, surfacing as a NaN many layers downstream
//! rather than as a fault.
//!
//! Checking the parties one at a time is what kept missing them. These two checks
//! ask instead about the properties the parties exist to serve:
//!
//! - [`hash_slots`] — the **content**. One number per slot over the bytes its block
//!   tables name, so the same slot can be hashed either side of an operation that
//!   must not change it. Any party that starts naming different ground changes the
//!   number, whichever party it was.
//! - [`validate_references`] — the **references**, checked against the refcount
//!   tables rather than against a reconstruction of who holds what. A gid or a record
//!   pointer naming a slot the tables call free is ground the allocator will hand to
//!   somebody else while this reference still points there. That is the orphan, and
//!   it is the fault the content hash detects only *after* the reissued slot has been
//!   written.
//!
//! Neither reconstructs liveness from holders. Every attempt to do that here has
//! been incomplete — arenas pool globally across same-config layers, so another
//! layer's window can hold a slot, and sealed sequences in the substrate hold them
//! too — and an incomplete reference set reports faults that are not there. The
//! refcount table is what the allocator itself consults, so it is the authority on
//! the only question that matters: will this ground be handed to someone else.
//!
//! # Cost, and where to call it
//!
//! The hash is one kernel and one `u64` per slot off the device. The reference check
//! walks the block tables host-side and reads each chunk's resident record back — one
//! transfer per chunk, not per band. Both are cheap at between-forwards cadence and
//! neither is cheap per layer, so they belong at operation boundaries — before and after
//! a compaction, a seal, an eviction, a tier install — which is also where they are
//! useful: the first boundary at which a slot's references go bad names the operation
//! that broke them.
//!
//! Both readbacks are synchronisations, which is why neither is for the inside of a
//! wave: a fenced build stops reproducing the races this is hunting. That is also why
//! the whole module is part of the `tensor-assert` harness and compiles to nothing
//! without it.

use std::collections::{HashMap, HashSet};

use candle::Result;

use super::arena::ArenaKey;
use super::backing::ChunkedKvBacking;
use super::backing::LiveBand;
use super::meta_pool::{band_ptr_offset, kv_head_record_bytes};
use super::size_class::SizeClass;
use crate::kv_cache::ArenaLocation;

/// One slot's integrity, as of one call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotIntegrity {
    /// Batch slot index.
    pub slot: usize,
    /// Content hash over every band the slot's block tables name, across every
    /// layer. Zero when the slot holds nothing hashable.
    pub hash: u64,
    /// Bands whose gid names a slot the refcount tables call free.
    ///
    /// **A reference to ground nothing holds.** The allocator is entitled to reissue
    /// it, and when it does this band reads whatever is written there next.
    pub orphaned_gids: usize,
    /// Bands whose record pointer names a slot the refcount tables call free.
    pub orphaned_records: usize,
    /// Bands whose record pointer and gid name different addresses.
    ///
    /// Reported rather than judged: a record is shared by every holder of a chunk, so
    /// a holder that is not its owner can disagree legitimately. A *rising* count
    /// across an operation is the signal; an absolute count is not.
    ///
    /// **A compaction has one routine source of its own, so read it against it.** A
    /// writer-owned chunk carrying a record, which the sweep skips, keeps both halves on
    /// the source and is not a defect. A chunk the sweep could not mint a record for
    /// (`records_declined`) is not a source: it stays whole on its source, gids and record
    /// together (`Sweep::remint`).
    pub pointer_disagreements: usize,
}

/// What one sweep of every live slot found.
#[derive(Clone, Debug, Default)]
pub struct IntegrityReport {
    pub slots: Vec<SlotIntegrity>,
}

impl IntegrityReport {
    /// Slots whose references are provably broken — the count that must be zero.
    pub fn orphans(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.orphaned_gids + s.orphaned_records)
            .sum()
    }

    /// Slots whose content hash differs from `earlier`'s, by slot index.
    ///
    /// Only slots present in both are compared: a slot that has been freed or newly
    /// admitted between the two calls has no business being compared, and an
    /// appended token legitimately changes the hash — so this is meaningful across an
    /// operation that must not touch K/V, and meaningless across one that appends.
    pub fn content_changed_against(&self, earlier: &IntegrityReport) -> Vec<(usize, u64, u64)> {
        let was: HashMap<usize, u64> = earlier.slots.iter().map(|s| (s.slot, s.hash)).collect();
        self.slots
            .iter()
            .filter_map(|s| {
                was.get(&s.slot).and_then(|&old| {
                    (old != s.hash && old != 0 && s.hash != 0).then_some((s.slot, old, s.hash))
                })
            })
            .collect()
    }
}

/// Every occupied slot of every GPU pool, by arena index.
///
/// The authority. `occupied_slots` returns the **list of occupied slot indices**, not
/// a bitmap — reading it as packed words reported 747 healthy records as orphaned.
fn occupancy(backing: &ChunkedKvBacking) -> HashMap<usize, HashSet<u32>> {
    let mut out: HashMap<usize, HashSet<u32>> = HashMap::new();
    for class in SizeClass::all() {
        let key = ArenaKey::new(class, ArenaLocation::Gpu);
        for (arena, _capacity, slots) in backing.pool_occupancy(key) {
            out.entry(arena).or_default().extend(slots);
        }
    }
    out
}

/// Validate every reference every live slot holds, across every backing.
///
/// Two halves from two owners. The gid half is host-only — no kernel, no readback, no
/// device work at all — and walks the block tables completely rather than sampling them.
/// The record half reads each chunk's resident record back over the bus, which is a
/// synchronisation, and is why this belongs at an operation boundary rather than inside a
/// wave.
pub fn validate_references(backings: &[ChunkedKvBacking]) -> Result<IntegrityReport> {
    let Some(first) = backings.first() else {
        return Ok(IntegrityReport::default());
    };
    let occ = occupancy(first);
    let mut per_slot: HashMap<usize, SlotIntegrity> = HashMap::new();

    for backing in backings {
        let extents = backing.arena_extents()?;
        for (slot, bands) in backing.live_bands()? {
            let e = per_slot.entry(slot).or_insert_with(|| SlotIntegrity {
                slot,
                hash: 0,
                orphaned_gids: 0,
                orphaned_records: 0,
                pointer_disagreements: 0,
            });
            for band in &bands {
                let held = occ
                    .get(&band.arena)
                    .is_some_and(|s| s.contains(&(band.slot_in_arena as u32)));
                if !held {
                    e.orphaned_gids += 1;
                }
            }
            check_records(backing, &occ, &extents, &bands, e);
        }
    }
    let mut slots: Vec<SlotIntegrity> = per_slot.into_values().collect();
    slots.sort_unstable_by_key(|s| s.slot);
    Ok(IntegrityReport { slots })
}

/// What the resident records say about the bands of one slot.
///
/// **The second recording.** A band's location is written down twice — in the `ChunkGid`,
/// which holds the arena slot's refcount, and in the `KvHead` record's pointer word,
/// which is the address the paged kernels actually dereference — and the two are updated
/// by different code. Only the gid keeps a slot alive, so a record pointer naming a slot
/// no gid holds is a pointer into ground the allocator is free to hand to somebody else,
/// after which that band reads another chunk's K/V: finite, plausibly shaped, wrong.
/// Nothing on the gid side can see that, which is why this half exists.
///
/// One transfer per record, not per band: a chunk's record covers its whole
/// `(head, palette, K/V)` grid, so every band of that chunk is answered from the same
/// read. Records are shared between holders of the same chunk, and re-reading one is
/// only a wasted transfer, never a wrong answer — a slot's own bands are deduplicated
/// here, and a record shared across slots is read once per slot.
///
/// A read that fails is skipped rather than reported: this runs at a boundary the caller
/// may be unable to abandon, and a failed transfer says nothing about the record.
fn check_records(
    backing: &ChunkedKvBacking,
    occ: &HashMap<usize, HashSet<u32>>,
    extents: &[(usize, u64, i64, u32)],
    bands: &[LiveBand],
    e: &mut SlotIntegrity,
) {
    let head_dim = backing.head_dim();
    // The window's own per-head band count, never the global `N_PALETTE`: the
    // single-latent path carries `LATENT_N_BANDS` bands per head, and the flat gid
    // slice is strided by whatever that window holds. Indexing through the global
    // constant mis-maps every band past the fourth there.
    let n_palette = backing.n_palette();
    let record_bytes = backing.n_kv_head() * kv_head_record_bytes(head_dim, n_palette);
    let mut records: HashMap<u64, Option<Vec<u8>>> = HashMap::new();
    for band in bands {
        let Some(record) = band.record else { continue };
        let Some(bytes) = records
            .entry(record)
            .or_insert_with(|| backing.read_record(record, record_bytes))
            .as_ref()
        else {
            continue;
        };
        let h = band.band / (n_palette * 2);
        let rem = band.band % (n_palette * 2);
        let off = band_ptr_offset(h, rem / 2, rem % 2 == 1, head_dim, n_palette);
        let Some(word) = bytes.get(off..off + 8) else {
            continue;
        };
        let ptr = u64::from_le_bytes(word.try_into().expect("an eight-byte slice"));
        if ptr == 0 {
            continue;
        }
        let from_gid = extents
            .iter()
            .find(|(a, ..)| *a == band.arena)
            .map(|&(_, base, stride, _)| base + band.slot_in_arena as u64 * stride as u64);
        if from_gid.is_some_and(|a| a != ptr) {
            e.pointer_disagreements += 1;
        }
        if let Some((arena, slot_in_arena)) = locate(extents, ptr) {
            if !occ
                .get(&arena)
                .is_some_and(|s| s.contains(&(slot_in_arena as u32)))
            {
                e.orphaned_records += 1;
            }
        }
    }
}

/// References and content in one sweep: what a boundary compares against the next.
///
/// Both halves belong in one object because they answer the same question from opposite
/// ends. The references say "this will break"; the content says "this has broken". An
/// orphaned reference is only a fault once the allocator reissues the ground and someone
/// writes it, so the reference check leads and the hash confirms — and a content change
/// with no orphaned reference means the party that moved is one neither half names, which
/// is itself the finding.
pub fn snapshot(backings: &[ChunkedKvBacking]) -> Result<IntegrityReport> {
    let mut report = validate_references(backings)?;
    let hashes = hash_slots(backings)?;
    for s in &mut report.slots {
        s.hash = hashes.get(&s.slot).copied().unwrap_or(0);
    }
    Ok(report)
}

/// Sweep every live slot, log what is wrong, and hand back the snapshot.
///
/// The one entry point the engine calls. It exists so the checks are invoked the same
/// way everywhere — before a compaction, after one, and (as those call sites are
/// added) after a seal, an eviction or a tier install — because the value of these
/// numbers is comparative: **the first boundary at which a slot's references go bad
/// names the operation that broke them.** A check wired differently at each site
/// cannot be compared across sites.
///
/// `whence` names the boundary in the log. `moves` and `unwitnessed` are the pass's own
/// accounting, passed through so one line carries both what the pass did and what the
/// state looks like afterwards; zero for a boundary that is not a compaction. `before`
/// is the snapshot from the opening boundary of the same operation, against which content
/// is compared — `None` at an opening boundary, which has nothing to compare against.
///
/// The returned snapshot is what the closing boundary passes back in as `before`. Pair
/// the two only across an operation that must **not** change K/V: a move rewrites
/// addresses and must leave bytes alone, so an inequality there is the corruption. Across
/// an append the hash changes by design and means nothing.
///
/// Logs rather than returns findings, and never panics: this is a diagnostic at a boundary
/// the caller may be unable to abandon, and an integrity check that takes the process down
/// while reporting "your bookkeeping is slightly wrong" is not a trade worth making.
/// `snapshot` is there for a caller that wants to act on the numbers itself.
///
/// # A clean report is NOT proof that nothing is corrupt
///
/// Both halves are scoped, and a fault outside both scopes reads as perfect health.
/// Measured 2026-09-27: a `KvHead` record claim promoted into a *band* arena wrote a
/// record on top of live K/V, and this reported **zero orphans and zero content changes**
/// on a run whose engine probe answered wrongly on six sessions of eight.
///
/// - [`validate_references`] asks whether a reference names ground the refcount tables
///   call free. A record sitting in a band slot is not that: every reference involved
///   names a live slot, and the slot's own refcount is held. Nothing is orphaned.
/// - [`hash_slots`] covers only the bands that **live slots' block tables** name —
///   `live_bands` walks `state.sequences`. K/V belonging to an evicted or
///   substrate-resident chunk is outside the hash, so ground clobbered there changes no
///   number here.
///
/// The engine probe's story gate is what caught it, which is the general rule: these
/// checks narrow a fault once something else has said one exists, and they cannot be read
/// in the other direction.
pub fn report_boundary(
    backings: &[ChunkedKvBacking],
    whence: &str,
    moves: usize,
    unwitnessed: usize,
    before: Option<&IntegrityReport>,
) -> IntegrityReport {
    let report = match snapshot(backings) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "candle_nn::kv_cache::integrity",
                %whence,
                "integrity sweep skipped: {e}",
            );
            return IntegrityReport::default();
        }
    };
    if report.orphans() > 0 {
        let worst: Vec<(usize, usize, usize)> = report
            .slots
            .iter()
            .filter(|s| s.orphaned_gids + s.orphaned_records > 0)
            .map(|s| (s.slot, s.orphaned_gids, s.orphaned_records))
            .take(8)
            .collect();
        tracing::error!(
            target: "candle_nn::kv_cache::integrity",
            %whence,
            orphans = report.orphans(),
            slots = report.slots.len(),
            disagreements = report
                .slots
                .iter()
                .map(|s| s.pointer_disagreements)
                .sum::<usize>(),
            moves,
            unwitnessed,
            worst = ?worst,
            "ORPHANED REFERENCES: live slots name bands the refcount tables call free, so \
             the allocator will reissue that ground while these references still point at \
             it — after which the reissued slot is read as this sequence's K/V \
             (worst: slot, orphaned gids, orphaned records)",
        );
    }
    if let Some(before) = before {
        let changed = report.content_changed_against(before);
        if !changed.is_empty() {
            tracing::error!(
                target: "candle_nn::kv_cache::integrity",
                %whence,
                changed = changed.len(),
                slots = report.slots.len(),
                moves,
                unwitnessed,
                worst = ?changed.iter().take(8).collect::<Vec<_>>(),
                "K/V CHANGED ACROSS AN OPERATION THAT MOVES IT WITHOUT REWRITING IT: these \
                 slots read different bytes than they did at the opening boundary, so some \
                 party now names ground holding another chunk's K/V (worst: slot, was, now)",
            );
        }
    }
    report
}

/// Turn a raw device address into the `(arena, slot)` the refcount tables are keyed
/// by, or `None` when it belongs to no arena of this pool.
///
/// The inverse of `slot_addr`, and the direction an orphaned **record pointer** has to
/// be checked in: a record carries an address, while occupancy is answered per slot.
fn locate(extents: &[(usize, u64, i64, u32)], addr: u64) -> Option<(usize, usize)> {
    for &(arena, base, stride, capacity) in extents {
        if stride <= 0 {
            continue;
        }
        let stride = stride as u64;
        let end = base + capacity as u64 * stride;
        if addr >= base && addr < end {
            return Some((arena, ((addr - base) / stride) as usize));
        }
    }
    None
}

/// One content hash per live slot, over every band its block tables name.
///
/// **The content half of the check, and the reason it is a kernel.** A slot's K/V is
/// megabytes across every layer; hashing it host-side would mean pulling all of it over
/// the bus, and the point of hashing at a boundary is that it must be cheap enough to do
/// at *every* boundary. One launch and one `u64` per slot off the device is.
///
/// Compare two calls either side of an operation that must not change K/V — a
/// compaction, an eviction of other slots, a tier install — and any party that started
/// naming different ground shows up, whichever party it was. Across an operation that
/// legitimately appends, the hash changes and means nothing; see
/// [`IntegrityReport::content_changed_against`], which only compares slots present in
/// both.
///
/// The seed per band is `2·i+1` over a canonical `(slot, layer, band)` walk: odd so the
/// multiply cannot fold a band away, and positional so two bands holding identical
/// bytes are not interchangeable — which is the corruption being hunted, a band naming
/// another band's slot.
pub fn hash_slots(backings: &[ChunkedKvBacking]) -> Result<HashMap<usize, u64>> {
    use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle::cuda_backend::kernels;

    let Some(first) = backings.first() else {
        return Ok(HashMap::new());
    };
    let candle::Device::Cuda(cuda) = first.device() else {
        return Ok(HashMap::new());
    };
    let extents = first.arena_extents()?;

    // One descriptor per band, in a canonical order so the seeds are stable across
    // calls for the same logical band and two calls are therefore comparable.
    let mut slots_seen: Vec<usize> = Vec::new();
    let mut slot_index: HashMap<usize, i32> = HashMap::new();
    let (mut ptrs, mut lens, mut slot_of, mut seeds) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for backing in backings {
        for (slot, bands) in backing.live_bands()? {
            let idx = *slot_index.entry(slot).or_insert_with(|| {
                slots_seen.push(slot);
                (slots_seen.len() - 1) as i32
            });
            for band in bands {
                let Some(&(_, base, stride, capacity)) =
                    extents.iter().find(|(a, ..)| *a == band.arena)
                else {
                    continue;
                };
                if band.slot_in_arena >= capacity as usize {
                    continue;
                }
                ptrs.push((base + band.slot_in_arena as u64 * stride as u64) as i64);
                lens.push(stride);
                slot_of.push(idx);
                seeds.push((ptrs.len() as u64) * 2 + 1);
            }
        }
    }
    if ptrs.is_empty() {
        return Ok(HashMap::new());
    }

    let d_ptrs = cuda.memcpy_stod(&ptrs)?;
    let d_lens = cuda.memcpy_stod(&lens)?;
    let d_slot = cuda.memcpy_stod(&slot_of)?;
    let d_seed = cuda.memcpy_stod(&seeds)?;
    // Zeroed because the kernel accumulates into it.
    let mut d_out = cuda.memcpy_stod(&vec![0u64; slots_seen.len()])?;
    let stream = cuda.cuda_stream();
    {
        let (p, _gp) = d_ptrs.device_ptr(&stream);
        let (l, _gl) = d_lens.device_ptr(&stream);
        let (s, _gs) = d_slot.device_ptr(&stream);
        let (e, _ge) = d_seed.device_ptr(&stream);
        let (o, _go) = d_out.device_ptr_mut(&stream);
        candle::set_kernel_breadcrumb("run_kv_hash", file!(), line!());
        // SAFETY: the four input arrays are `ptrs.len()` long and device-resident;
        // `out` is `slots_seen.len()` long and zeroed; every `ptrs[i]` names `lens[i]`
        // bytes of one arena slot, bounded against that arena's capacity above.
        unsafe {
            kernels::simple::kv_hash::run_kv_hash(
                p as *const i64,
                l as *const i64,
                s as *const i32,
                e as *const u64,
                ptrs.len() as i32,
                o as *mut u64,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    let hashes = cuda.memcpy_dtov(&d_out)?;
    Ok(slots_seen.into_iter().zip(hashes).collect())
}
