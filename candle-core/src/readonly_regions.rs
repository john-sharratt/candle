//! VRAM that is written once at load and must never be written again.
//!
//! Some device memory is genuinely immutable for the life of the process: model
//! weights, repacked expert slots, rope tables — anything filled during startup
//! and only read thereafter. Nothing enforces that. The allocator does not know
//! which of its addresses are spoken for by a long-lived tensor, and a kernel
//! does not know that the buffer it was handed overlaps one, so a stale
//! pointer, a pool block handed out twice, or a store running past its tile
//! lands on a weight and surfaces much later as a wrong number — if at all.
//!
//! This is the enforcement. A declared region is checked on two paths:
//!
//! * **Allocation.** A new device allocation overlapping a read-only region
//!   means the pool handed out memory it had already given away. Caught before
//!   a byte is written.
//! * **Kernel writes.** An output buffer overlapping a read-only region is
//!   about to corrupt it. Checked at the FFI boundary, where the destination
//!   pointer and its length are both in hand.
//!
//! Both name the offending region and panic. The alternative to stopping is a
//! silently wrong model, and the silent version of this bug took days to find.
//!
//! ## Lock-free by construction
//!
//! The check sits on the allocation path and on every write-capable kernel
//! launch, so it must not take a lock — an uncontended `RwLock` read is still
//! an atomic read-modify-write, and these paths run from several threads at
//! once (the forward, the persistence thread, the expert streamer).
//!
//! So the hot path touches only atomics over a fixed array:
//!
//! 1. A **bounding box** of every region, two `u64` loads. Almost every
//!    allocation in a run is nowhere near the weights, and this rejects those
//!    with two compares and no scan at all.
//! 2. Otherwise a linear scan of at most [`MAX_REGIONS`] `(base, end)` pairs in
//!    one contiguous array — a few cache lines, no indirection, no allocation.
//!
//! Registration is append-only: claim a slot with `fetch_add`, publish the
//! bounds, then publish the count with a release store. A reader that observes
//! the count observes the bounds that preceded it. Nothing is ever removed,
//! which is what makes that ordering sufficient — a region's life is the
//! process's.
//!
//! Names are deliberately **not** on the fast path. They live behind a mutex and
//! are consulted only after an overlap has been found, which happens at most
//! once per process because the next thing that happens is a panic.
//!
//! ## Using it without being misled
//!
//! Three ways this instrument lies if you let it, all learned the hard way:
//!
//! * **A declaration goes stale.** Immutability is a property of the *owner*,
//!   not of the address. The expert weight zone is elastic — it releases ground
//!   to the KV side, and from that instant the released bytes are ordinary arena
//!   memory the wave tier will legitimately stand on and write. A region
//!   declared once at load then reports those writes as corruption. Whoever
//!   publishes the boundary must call [`release_below`]; `region_pool`'s
//!   `set_weight_floor` does.
//! * **A capacity is not a write.** Checking the *declared extent* of a buffer
//!   catches an arena whose bound overlaps a region even when nothing is ever
//!   carved that far. That is worth knowing, but it is a placement bug, not
//!   evidence of a write — do not read it as one.
//! * **A guard that is not armed reports the same as a guard that found
//!   nothing.** [`coverage`] exists so a startup line can prove regions were
//!   actually declared. "0 violations" over 0 declared bytes means nothing.
//!
//! And the standing rule for the whole harness: an instrument that allocates or
//! synchronises inside the region it observes is not an instrument, it is a
//! change to the program. The `forbid_write` stub is `#[inline(always)]` and
//! empty without the feature, but **its arguments are still evaluated** — a
//! `format!` at a call site allocates a `String` per call in a build that then
//! throws it away. Gate the whole call, not just its body.

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

/// An immutable, sorted snapshot of every declared region.
///
/// Published by pointer swap and never mutated after publication, so a reader
/// needs one atomic load and then owns a consistent view for as long as it
/// looks. That is what makes the lookup both lock-free and safe under
/// fragmentation: sorting a shared array in place would let a concurrent
/// binary search descend through a half-sorted table and miss a violation,
/// which in a guard is the one failure mode worse than being slow.
struct Table {
    /// Region starts, ascending. Regions never overlap — a guarantee of the
    /// callers, and what makes a single binary search sufficient.
    bases: Vec<u64>,
    /// `ends[i]` is the exclusive end of `bases[i]`.
    ends: Vec<u64>,
    names: Vec<String>,
    /// Bounding box over the whole set, for the two-compare fast reject.
    lo: u64,
    hi: u64,
    bytes: u64,
}

impl Table {
    /// Whether `[base, end)` overlaps any region. `O(log n)`.
    #[inline]
    fn hits(&self, base: u64, end: u64) -> bool {
        if end <= self.lo || base >= self.hi {
            return false;
        }
        self.find(base, end).is_some()
    }

    /// Index of the region `[base, end)` overlaps, if any.
    ///
    /// Regions are sorted and disjoint, so the only candidate is the last one
    /// starting strictly before `end`: anything earlier ends before it (they do
    /// not overlap each other), and anything later starts at or after it.
    #[inline]
    fn find(&self, base: u64, end: u64) -> Option<usize> {
        // `partition_point` is a binary search: the count of regions whose base
        // is < end. The candidate is the one before that boundary.
        let i = self.bases.partition_point(|&b| b < end);
        let i = i.checked_sub(1)?;
        (self.ends[i] > base).then_some(i)
    }
}

static TABLE: AtomicPtr<Table> = AtomicPtr::new(std::ptr::null_mut());

/// Serialises rebuilds. Never taken on the read path.
fn writer() -> &'static Mutex<()> {
    static W: Mutex<()> = Mutex::new(());
    &W
}

/// Load the live snapshot, if any.
#[inline]
fn table() -> Option<&'static Table> {
    // SAFETY: a published table is leaked and never mutated or freed, so the
    // pointer stays valid for the life of the process.
    unsafe { TABLE.load(Ordering::Acquire).as_ref() }
}

/// Declare `[base, base + len)` immutable for the rest of the process.
///
/// Call after the region is filled and before anything else can allocate: the
/// point is to describe memory whose contents are final, so declaring it while
/// the fill is still running would make the fill itself a violation.
///
/// Rebuilds and republishes the sorted snapshot, so this is a load-time
/// operation — use [`declare_merged`] to declare many at once rather than
/// calling this in a loop.
pub fn declare(name: impl Into<String>, base: u64, len: usize) {
    let name = name.into();
    declare_all(&[(base, len)], |_| name.clone());
}

/// Declare several spans in one rebuild.
fn declare_all(spans: &[(u64, usize)], name_of: impl Fn(usize) -> String) {
    let _w = match writer().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let cur = table();
    let mut v: Vec<(u64, u64, String)> = match cur {
        Some(t) => t
            .bases
            .iter()
            .zip(&t.ends)
            .zip(&t.names)
            .map(|((b, e), n)| (*b, *e, n.clone()))
            .collect(),
        None => Vec::new(),
    };
    for (i, &(base, len)) in spans.iter().enumerate() {
        if base == 0 || len == 0 {
            continue;
        }
        let end = base.saturating_add(len as u64);
        // Idempotent: re-declaring a span already covered is a no-op rather
        // than a second overlapping entry. Callers legitimately declare the
        // same immutable object from more than one place, and two entries for
        // one span would break the disjointness the search relies on.
        if v.iter().any(|(b, e, _)| base >= *b && end <= *e) {
            continue;
        }
        v.push((base, end, name_of(i)));
    }
    v.sort_unstable_by_key(|(b, _, _)| *b);
    let lo = v.first().map(|(b, _, _)| *b).unwrap_or(u64::MAX);
    let hi = v.iter().map(|(_, e, _)| *e).max().unwrap_or(0);
    let bytes = v.iter().map(|(b, e, _)| e - b).sum();
    let t = Box::new(Table {
        bases: v.iter().map(|(b, _, _)| *b).collect(),
        ends: v.iter().map(|(_, e, _)| *e).collect(),
        names: v.into_iter().map(|(_, _, n)| n).collect(),
        lo,
        hi,
        bytes,
    });
    // Leaked deliberately: a region's life is the process's, readers hold no
    // reference count, and there is no safe moment to free a table a
    // lock-free reader may still be inside.
    TABLE.store(Box::leak(t), Ordering::Release);
}

/// Declare many spans at once, merging adjacent and overlapping ones first.
///
/// The natural unit a caller has is one span per *object* — 31,488 expert
/// weights, say — but those objects are carved out of a handful of pool blocks
/// and sit end to end. Declaring them individually would both overflow the
/// table and turn the check into a 31,488-entry scan; merging first collapses
/// them to the few spans actually distinct in the address space, which is what
/// keeps the hot path two loads and a walk over a couple of cache lines.
///
/// `spans` is `(base, len)`. Returns the number of merged regions declared.
pub fn declare_merged(name: &str, spans: &mut [(u64, usize)]) -> usize {
    spans.sort_unstable_by_key(|(b, _)| *b);
    let mut merged: Vec<(u64, usize)> = Vec::new();
    let mut cur: Option<(u64, u64)> = None; // (base, end)
    for &(b, l) in spans.iter() {
        if b == 0 || l == 0 {
            continue;
        }
        let e = b.saturating_add(l as u64);
        match cur {
            // Touching counts as adjacent: two allocations that abut are one
            // contiguous immutable span for the purpose of this check.
            Some((cb, ce)) if b <= ce => cur = Some((cb, ce.max(e))),
            Some((cb, ce)) => {
                merged.push((cb, (ce - cb) as usize));
                cur = Some((b, e));
            }
            None => cur = Some((b, e)),
        }
    }
    if let Some((cb, ce)) = cur {
        merged.push((cb, (ce - cb) as usize));
    }
    // One rebuild for the whole set rather than one per span.
    declare_all(&merged, |i| format!("{name}[{i}]"));
    merged.len()
}

/// Give back every declared byte below `base`.
///
/// A region is only immutable while its owner still owns it. The expert weight
/// zone is elastic: it releases ground to the KV side, and from that moment the
/// released bytes are ordinary arena memory that the wave tier will legitimately
/// stand on and write. A region declared once at load does not know that, so it
/// reports those writes as violations — which is worse than not checking, since
/// a guard that cries wolf is one you stop reading.
///
/// Called by whoever publishes the boundary, with the new floor. Regions wholly
/// below it are dropped, one straddling it is clipped, and anything above is
/// untouched.
pub fn release_below(base: u64) {
    let _w = match writer().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(cur) = table() else { return };
    let mut v: Vec<(u64, u64, String)> = Vec::with_capacity(cur.bases.len());
    for ((b, e), n) in cur.bases.iter().zip(&cur.ends).zip(&cur.names) {
        let (b, e) = (*b, *e);
        if e <= base {
            continue; // wholly released
        }
        v.push((b.max(base), e, n.clone()));
    }
    let lo = v.first().map(|(b, _, _)| *b).unwrap_or(u64::MAX);
    let hi = v.iter().map(|(_, e, _)| *e).max().unwrap_or(0);
    let bytes = v.iter().map(|(b, e, _)| e - b).sum();
    let t = Box::new(Table {
        bases: v.iter().map(|(b, _, _)| *b).collect(),
        ends: v.iter().map(|(_, e, _)| *e).collect(),
        names: v.into_iter().map(|(_, _, n)| n).collect(),
        lo,
        hi,
        bytes,
    });
    TABLE.store(Box::leak(t), Ordering::Release);
}

/// How many regions are protected, and how many bytes they cover.
///
/// For a startup line that can say the guard is armed — "protecting 0 regions"
/// and "protecting 31,488" read identically in every other respect.
pub fn coverage() -> (usize, u64) {
    table().map(|t| (t.bases.len(), t.bytes)).unwrap_or((0, 0))
}

/// Whether `[base, base + len)` touches read-only memory.
///
/// Lock-free and `O(log n)`: one atomic load, a two-compare bounding-box reject
/// that catches everything outside the declared set, then a binary search over
/// the sorted disjoint regions for anything inside it.
#[inline]
pub fn hits(base: u64, len: usize) -> bool {
    if base == 0 || len == 0 {
        return false;
    }
    match table() {
        Some(t) => t.hits(base, base.wrapping_add(len as u64)),
        None => false,
    }
}

/// The name of the read-only region `[base, base + len)` overlaps.
///
/// Same search as [`hits`]; separate only because formatting a name allocates
/// and the answer is wanted exactly once, in a panic.
pub fn overlapping(base: u64, len: usize) -> Option<String> {
    if base == 0 || len == 0 {
        return None;
    }
    let t = table()?;
    let end = base.wrapping_add(len as u64);
    let i = t.find(base, end)?;
    Some(format!(
        "{} [{:#x}, {:#x})",
        t.names[i], t.bases[i], t.ends[i]
    ))
}

/// Panic if `[base, base + len)` touches read-only memory.
///
/// `what` names the writer — a kernel entry point, an allocation site — so the
/// panic identifies the culprit rather than the victim. Finding the victim was
/// never the hard part.
#[inline]
pub fn forbid_write(what: &str, base: u64, len: usize) {
    if !hits(base, len) {
        return;
    }
    let region = overlapping(base, len).unwrap_or_else(|| "<unnamed>".into());
    panic!(
        "READ-ONLY VRAM WRITTEN: {what} targets [{base:#x}, {:#x}) which overlaps {region}. \
         That memory was filled at load and is never written again by design, so this is \
         either a pool block handed out twice or a store running past its buffer. The \
         process stops here because the alternative is a silently wrong model.",
        base.wrapping_add(len as u64),
    );
}

#[cfg(test)]
mod tests {
    use super::{coverage, declare, hits, overlapping};

    // Addresses far outside anything a real allocator returns, so these cannot
    // collide with a region another test declares.
    const A: u64 = 0x7000_0000_0000;

    #[test]
    fn overlap_is_half_open_at_both_ends() {
        declare("readonly_regions::test::edges", A, 0x100);
        assert!(!hits(A - 0x100, 0x100), "ends exactly at base");
        assert!(!hits(A + 0x100, 0x100), "starts exactly at end");
        assert!(hits(A - 0x100, 0x101), "laps one byte over base");
        assert!(hits(A + 0xff, 0x100), "starts one byte before end");
        assert!(hits(A + 0x40, 0x10), "contained");
        assert!(hits(A - 0x1000, 0x9999), "containing");
        assert!(hits(A, 0x100), "identical");
    }

    #[test]
    fn a_zero_length_or_null_span_never_overlaps() {
        declare("readonly_regions::test::empty_probe", A + 0x1_0000, 0x100);
        assert!(!hits(A + 0x1_0000, 0), "an empty span writes nothing");
        assert!(!hits(0, 16), "a null base is not a write");
        assert!(overlapping(0, 16).is_none());
    }

    #[test]
    fn declaring_grows_coverage_and_the_report_names_the_region() {
        let (n0, b0) = coverage();
        declare("readonly_regions::test::alpha", A + 0x2_0000, 4096);
        let (n1, b1) = coverage();
        assert_eq!(n1, n0 + 1);
        assert_eq!(b1, b0 + 4096);
        let hit = overlapping(A + 0x2_0800, 16).expect("inside the region");
        assert!(hit.contains("alpha"), "the report names it: {hit}");
        assert!(
            !hits(A + 0x2_1000, 16),
            "one past the end is not a hit"
        );
    }

    #[test]
    fn a_declaration_with_no_extent_is_ignored() {
        let (n0, _) = coverage();
        declare("readonly_regions::test::zero_len", A + 0x3_0000, 0);
        declare("readonly_regions::test::null_base", 0, 4096);
        assert_eq!(coverage().0, n0, "neither is a region");
    }

    #[test]
    fn adjacent_and_overlapping_spans_merge_into_one_region() {
        use super::declare_merged;
        let base = A + 0x5_0000;
        // Out of order on purpose, and mixing abutting, overlapping and
        // disjoint — the caller's list is whatever order the objects came in.
        let mut spans = vec![
            (base + 0x100, 0x100), // abuts the first
            (base, 0x100),         // first
            (base + 0x180, 0x100), // overlaps the second
            (base + 0x9000, 0x50), // disjoint
        ];
        let n = declare_merged("readonly_regions::test::merged", &mut spans);
        assert_eq!(n, 2, "three touching spans collapse to one, plus the loner");
        // The merged span covers the whole run, including the gap-free joins.
        assert!(hits(base, 1));
        assert!(hits(base + 0x27f, 1), "last byte of the merged run");
        assert!(!hits(base + 0x280, 1), "one past it");
        assert!(hits(base + 0x9000, 1), "the disjoint one is still protected");
    }

    #[test]
    fn merging_ignores_empty_and_null_spans() {
        use super::declare_merged;
        let mut spans = vec![(A + 0x6_0000, 0), (0, 0x100)];
        assert_eq!(
            declare_merged("readonly_regions::test::merged_empty", &mut spans),
            0
        );
    }

    #[test]
    fn an_address_outside_the_bounding_box_is_rejected_without_scanning() {
        declare("readonly_regions::test::box", A + 0x4_0000, 0x1000);
        // Far below and far above every declared region. Correctness is what is
        // asserted; the point of the fast path is that this costs two loads.
        assert!(!hits(0x1000, 0x1000));
        assert!(!hits(0x7FFF_FFFF_0000, 0x1000));
    }

    #[test]
    fn a_fragmented_set_is_searched_correctly_at_every_gap_and_edge() {
        use super::declare_merged;
        // Many disjoint regions with gaps between them — the case a binary
        // search exists for, and the one a bounding-box reject cannot help
        // with because the probes fall inside the box.
        let base = A + 0x10_0000;
        let stride = 0x1000u64;
        let width = 0x400usize;
        let n = 64usize;
        let mut spans: Vec<(u64, usize)> = (0..n)
            .map(|i| (base + i as u64 * stride, width))
            .rev() // unsorted input
            .collect();
        assert_eq!(
            declare_merged("readonly_regions::test::frag", &mut spans),
            n,
            "disjoint spans must NOT merge"
        );
        for i in 0..n {
            let b = base + i as u64 * stride;
            assert!(hits(b, 1), "first byte of region {i}");
            assert!(hits(b + width as u64 - 1, 1), "last byte of region {i}");
            assert!(!hits(b + width as u64, 1), "first byte of the gap after {i}");
            assert!(!hits(b - 1, 1), "last byte of the gap before {i}");
            // A span covering the whole gap between two regions but touching
            // neither must miss; one that laps either edge must hit.
            if i + 1 < n {
                let gap = b + width as u64;
                let gap_len = (stride - width as u64) as usize;
                assert!(!hits(gap, gap_len), "the whole gap after {i} is free");
                assert!(hits(gap - 1, gap_len), "lapping the end of {i}");
                assert!(hits(gap, gap_len + 1), "lapping the start of {}", i + 1);
            }
        }
    }

    #[test]
    fn releasing_below_a_floor_drops_clips_and_keeps_the_right_regions() {
        use super::release_below;
        let base = A + 0x30_0000;
        // Three regions: one wholly below the floor, one straddling it, one
        // wholly above.
        let mut spans = vec![
            (base, 0x1000),
            (base + 0x2000, 0x2000),
            (base + 0x8000, 0x1000),
        ];
        super::declare_merged("readonly_regions::test::release", &mut spans);
        let floor = base + 0x3000; // inside the middle region
        assert!(hits(base, 1), "below-floor region protected before release");
        assert!(hits(base + 0x2000, 1), "straddling region's low half");

        release_below(floor);

        assert!(!hits(base, 0x1000), "the wholly-below region is released");
        assert!(
            !hits(base + 0x2000, 0x1000),
            "the straddling region's low half is released"
        );
        assert!(
            hits(floor, 1),
            "the straddling region is kept from the floor up"
        );
        assert!(hits(base + 0x8000, 1), "the wholly-above region is untouched");
    }

    #[test]
    fn re_declaring_a_covered_span_does_not_add_a_second_entry() {
        let base = A + 0x20_0000;
        declare("readonly_regions::test::once", base, 0x1000);
        let (n0, b0) = coverage();
        // Exactly the same span, and a strict subset of it. Both are already
        // protected; adding either again would break the disjointness the
        // binary search depends on.
        declare("readonly_regions::test::again", base, 0x1000);
        declare("readonly_regions::test::subset", base + 0x100, 0x100);
        assert_eq!(coverage(), (n0, b0), "neither re-declaration adds a region");
    }
}
