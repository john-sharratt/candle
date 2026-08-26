//! Bounds checks on the device pointers this engine hands its KV kernels.
//!
//! # Why this exists
//!
//! A wild pointer in a CUDA kernel reports as `CUDA_ERROR_ILLEGAL_ADDRESS` on
//! whichever thread next synchronises, with no attribution: the launch that
//! faulted may be several back in the queue, and the thread that *notices* is
//! usually not the thread that *did* it. Three production crashes cost a
//! 23-minute run each and named three different kernels, two of which were
//! bystanders. Nothing in the failure said which buffer was wrong, which chunk
//! it belonged to, or which field held the bad address.
//!
//! This module makes that class of bug report itself. **Every pointer this
//! engine gives a KV kernel is checkable before the launch**, because the
//! reservation's layout is known exactly — one VMM span, carved into a
//! persistence staging block, a region range, a moving weight boundary, and a
//! per-wave transient tier. An address either lands somewhere legal in that
//! picture or it is a bug, and the check is a handful of integer comparisons.
//!
//! # Why it panics
//!
//! Returning an error would let the caller carry on with a descriptor it has
//! already proven wrong, and the next thing to touch it is a kernel that cannot
//! check anything. A bad pointer here means the engine's bookkeeping has
//! diverged from its memory, and there is no correct way to continue from that:
//! the process is already in the state where the *next* observation is a
//! poisoned context and an unattributable fault. Failing loudly, at the site
//! that built the descriptor, with the chunk and field named, is the whole
//! point — it converts a 23-minute mystery into a stack trace.
//!
//! This is `docs/elastic_vram_partition.md` principle 7 — refuse rather than
//! corrupt — applied to the one surface that had no guard: `place_transient`
//! refuses an overlapping tier, `build_slot_headers` refuses a stale
//! position-map shape, `set_weight_floor` refuses to cut a live region. The
//! pointers those decisions produce were never checked against the result.

use super::region_pool::{span_layout, SpanLayout};
use super::REGION_BYTES;

/// Where a device address falls inside the reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanZone {
    /// The persistence thread's staging block, at the fixed left end.
    Persist,
    /// Region `i` of the KV side.
    Region(usize),
    /// Ground the KV side owns but has not carved into a region yet.
    UnclaimedKv,
    /// The wave transient tier, while one stands.
    Tier,
    /// Expert slots — the weight side of the boundary.
    Weight,
    /// Not in the reservation at all. Legal for a few things (the meta-pool
    /// slabs are their own allocations); never legal for an arena pointer.
    Outside,
}

impl SpanZone {
    /// Whether a KV arena's bytes may live here.
    ///
    /// Not the tier: that ground belongs to a running wave's intermediates, and
    /// an arena pointer into it means the two have been overlapped. Not the
    /// weight side: those are expert slots. Not outside: the reservation is the
    /// only place KV lives.
    pub fn holds_kv(self) -> bool {
        matches!(self, SpanZone::Region(_) | SpanZone::UnclaimedKv)
    }
}

/// Classify `addr` against a reservation layout.
///
/// Takes the layout rather than reading it, so the rule tests without a device.
pub(crate) fn classify_in(l: &SpanLayout, addr: u64) -> SpanZone {
    if addr < l.span_base || addr >= l.span_end {
        return SpanZone::Outside;
    }
    // The tier is checked first and deliberately: it is carved out of ground
    // that is otherwise the KV side's, so every other test would answer
    // "region" for an address inside it.
    if let Some(base) = l.transient_base {
        if addr >= base && addr < base + l.transient_bytes as u64 {
            return SpanZone::Tier;
        }
    }
    if addr >= l.weight_floor {
        return SpanZone::Weight;
    }
    if addr < l.region_base {
        return SpanZone::Persist;
    }
    let idx = ((addr - l.region_base) / REGION_BYTES as u64) as usize;
    if idx < l.total {
        SpanZone::Region(idx)
    } else {
        SpanZone::UnclaimedKv
    }
}

/// Panic unless `[addr, addr + len)` is entirely KV ground this engine owns.
///
/// `what` names the buffer and `whence` the site, because the whole value of
/// this check is that the report says which one — see the module docs.
///
/// A zero-length range is vacuously fine and a null pointer never is: a null
/// `kvheads_ptr` is the documented signature of a resident slice whose meta
/// record was never built.
pub fn expect_kv_range(ordinal: usize, addr: u64, len: usize, what: &str, whence: &str) {
    let Some(layout) = span_layout(ordinal) else {
        return;
    };
    expect_kv_range_in(&layout, addr, len, what, whence);
}

/// [`expect_kv_range`] against a layout the caller already holds.
///
/// [`span_layout`] takes the region pool's global lock and copies the layout
/// out, which is fine once and ruinous per pointer: a slot's KV record carries
/// `N_PALETTE` K and V addresses per head, so a checker that fetches the layout
/// itself turns one serialization pass into thousands of lock acquisitions. A
/// caller checking a batch of pointers fetches the layout ONCE and calls this
/// for each — the layout cannot change underneath it, because the pool that
/// owns it is not reachable from a serialization pass.
///
/// Measured before this existed: 4,608 fetches per attention layer per
/// speculative step (144 slices × 4 heads × 8 addresses), 0.79 ms of pure lock
/// traffic per layer — 98% of the slot-metadata pack.
pub fn expect_kv_range_in(layout: &SpanLayout, addr: u64, len: usize, what: &str, whence: &str) {
    if let Err(why) = check_kv_range(layout, addr, len) {
        panic!(
            "{whence}: {what} is not KV ground — {why}.\n  \
             addr={addr:#x} len={len} \n  \
             span=[{:#x},{:#x}) regions=[{:#x},{:#x}) ({} × {} B) \
             weight_floor={:#x} tier={}\n  \
             A pointer this engine built does not name memory this engine owns. \
             Continuing would hand it to a kernel, which reports it as \
             CUDA_ERROR_ILLEGAL_ADDRESS on some other thread with no attribution.",
            layout.span_base,
            layout.span_end,
            layout.region_base,
            layout.region_end(),
            layout.total,
            REGION_BYTES,
            layout.weight_floor,
            match layout.transient_base {
                Some(b) => format!("[{:#x},{:#x})", b, b + layout.transient_bytes as u64),
                None => "none".to_string(),
            },
        );
    }
}

/// The rule behind [`expect_kv_range`], as a `Result` so it tests without a
/// device and without unwinding.
pub(crate) fn check_kv_range(
    layout: &SpanLayout,
    addr: u64,
    len: usize,
) -> std::result::Result<(), String> {
    if len == 0 {
        return Ok(());
    }
    if addr == 0 {
        return Err("the pointer is null".to_string());
    }
    let end = addr
        .checked_add(len as u64)
        .ok_or_else(|| "addr + len overflows".to_string())?;

    let start_zone = classify_in(layout, addr);
    if !start_zone.holds_kv() {
        return Err(format!("it starts in {start_zone:?}"));
    }
    // The last addressed byte, not one past it: a range ending exactly on a
    // boundary is legal and `end` itself would classify as the next zone.
    let last_zone = classify_in(layout, end - 1);
    if !last_zone.holds_kv() {
        return Err(format!(
            "it starts in {start_zone:?} but ends in {last_zone:?}, so it \
             straddles a boundary"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reservation shaped like the daemon's: a staging block, 8 regions, then
    /// the weight side.
    fn layout() -> SpanLayout {
        let span_base = 0x1000_0000u64;
        let region_base = span_base + 4 * REGION_BYTES as u64;
        let total = 8;
        SpanLayout {
            span_base,
            span_end: region_base + (total * REGION_BYTES) as u64 + 0x10_0000,
            region_base,
            persist_carved: 4 * REGION_BYTES,
            total,
            weight_floor: region_base + (total * REGION_BYTES) as u64,
            transient_base: None,
            transient_bytes: 0,
        }
    }

    /// **The caller-supplied-layout form still panics on a bad pointer.**
    ///
    /// [`expect_kv_range_in`] exists so a batch of pointers is checked against
    /// one layout fetch instead of one per pointer. The risk in that change is
    /// silently weakening the guard — a caller that passes no layout, or a form
    /// that stops checking, looks exactly like a fast one. So the two properties
    /// are pinned here: a weight-side address panics, and KV ground does not.
    #[test]
    #[should_panic(expected = "not KV ground")]
    fn a_weight_side_pointer_still_panics_with_a_caller_layout() {
        let l = layout();
        expect_kv_range_in(&l, l.weight_floor, 1, "k_ptr", "test");
    }

    #[test]
    fn kv_ground_passes_the_caller_layout_form() {
        let l = layout();
        expect_kv_range_in(&l, l.region_base, 1, "k_ptr", "test");
        expect_kv_range_in(&l, l.region_end() - 1, 1, "k_ptr", "test");
    }

    /// The two forms agree — the hoisted one is the same rule, not a laxer one.
    #[test]
    fn both_forms_accept_and_reject_the_same_addresses() {
        let l = layout();
        for addr in [l.span_base, l.region_base, l.weight_floor, 0x1] {
            let direct = check_kv_range(&l, addr, 1).is_ok();
            let hoisted = std::panic::catch_unwind(|| {
                expect_kv_range_in(&l, addr, 1, "p", "test");
            })
            .is_ok();
            assert_eq!(direct, hoisted, "forms disagree at {addr:#x}");
        }
    }

    #[test]
    fn every_part_of_the_span_classifies_as_itself() {
        let l = layout();
        assert_eq!(classify_in(&l, l.span_base), SpanZone::Persist);
        assert_eq!(classify_in(&l, l.region_base), SpanZone::Region(0));
        assert_eq!(
            classify_in(&l, l.region_base + REGION_BYTES as u64),
            SpanZone::Region(1)
        );
        assert_eq!(classify_in(&l, l.weight_floor), SpanZone::Weight);
        assert_eq!(classify_in(&l, l.span_base - 1), SpanZone::Outside);
        assert_eq!(classify_in(&l, l.span_end), SpanZone::Outside);
    }

    /// **The tier is not KV ground**, even though it is carved out of the KV
    /// side's address range. An arena pointer landing in it means a wave's
    /// intermediates and a KV arena have been given the same bytes — the
    /// silent-corruption case the region ceiling exists to prevent.
    #[test]
    fn the_tier_is_not_kv_ground() {
        let mut l = layout();
        let tier = l.weight_floor - 2 * REGION_BYTES as u64;
        l.transient_base = Some(tier);
        l.transient_bytes = 2 * REGION_BYTES;

        assert_eq!(classify_in(&l, tier), SpanZone::Tier);
        assert!(!classify_in(&l, tier).holds_kv());
        assert!(check_kv_range(&l, tier, 64).is_err());
        // The region below it is still perfectly good KV.
        assert!(check_kv_range(&l, tier - REGION_BYTES as u64, 64).is_ok());
    }

    /// A range that starts legal and runs off the end is caught, which the
    /// start-only check would miss — and running off the end of the KV side is
    /// exactly what an over-long chunk stride does.
    #[test]
    fn a_range_that_straddles_a_boundary_is_refused() {
        let l = layout();
        let last = l.region_end() - 32;
        assert!(check_kv_range(&l, last, 32).is_ok(), "ending flush is fine");
        let why = check_kv_range(&l, last, 64).expect_err("running over is not");
        assert!(why.contains("straddles"), "{why}");
    }

    #[test]
    fn null_and_outside_are_refused() {
        let l = layout();
        assert!(check_kv_range(&l, 0, 64).is_err(), "null");
        assert!(check_kv_range(&l, 0xdead_0000, 64).is_err(), "outside");
        assert!(
            check_kv_range(&l, l.span_base, 64).is_err(),
            "persist block"
        );
        assert!(
            check_kv_range(&l, l.weight_floor, 64).is_err(),
            "weight side"
        );
        assert!(check_kv_range(&l, 0, 0).is_ok(), "empty is vacuous");
    }
}
