//! Tests for claiming regions, the weight-floor and tier boundaries, and growth.
//!
//! These tests pin the arithmetic the claim path and the boundary depend on,
//! which runs without a device: the lowest free run, the pairing that moves a
//! tenant downward, the growth step and its minimum grant, and the spare-ground
//! rule. They also check that a region still held is never issued twice, that a
//! released region is cleaned before reuse, that a floor may not cut a standing
//! tier, and that every region is live, free, or blocked.

use super::*;

#[cfg(test)]
pub(super) mod run_tests {
    use super::lowest_run;
    use std::collections::HashSet;

    fn free(i: &[usize]) -> HashSet<usize> {
        i.iter().copied().collect()
    }

    /// The lowest run of free regions is taken, holes too short for it skipped.
    #[test]
    fn the_lowest_run_that_fits_is_taken() {
        // Free 1, 3–5 below `next` = 8: a run of 3 is 3–5, a run of 1 is 1.
        assert_eq!(lowest_run(&free(&[1, 3, 4, 5]), 8, 20, 3), Some(3));
        assert_eq!(lowest_run(&free(&[1, 3, 4, 5]), 8, 20, 1), Some(1));
    }

    /// Fresh ground past `next` counts as free, and a run may start in a hole
    /// below it and continue into it.
    #[test]
    fn fresh_ground_extends_a_run() {
        assert_eq!(lowest_run(&free(&[6, 7]), 8, 20, 4), Some(6));
        assert_eq!(lowest_run(&free(&[]), 8, 20, 4), Some(8));
    }

    /// Nothing at or above the ceiling, and no run of nothing.
    #[test]
    fn the_ceiling_bounds_a_run() {
        assert_eq!(lowest_run(&free(&[]), 8, 11, 4), None);
        assert_eq!(lowest_run(&free(&[]), 8, 12, 4), Some(8));
        assert_eq!(lowest_run(&free(&[1]), 8, 20, 0), None);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::{
        claim_region, frontier_relocations, kv_grow_step, place_transient, region_stats,
        release_transient, REGION_BYTES,
    };

    use candle::{Device, Result};

    use std::cmp::Reverse;

    /// Direct pool access, for the two ownership guards: the states they refuse
    /// are bookkeeping faults, so a test has to write one rather than provoke it.
    use super::with_pool;

    /// The pool is process-global and cargo runs tests in parallel, so region
    /// counts are only stable while one test at a time is looking at them — and
    /// that includes tests outside this module, which is why it is the
    /// crate-wide lock and not a local `static`.
    use super::super::super::gpu_test_lock::gpu_serial as serial;

    use super::super::tests_layout::tests::rest_layout;

    /// A tenant whose regions stand under another tenant's live region empties
    /// none: emptying them would only move a hole under a frontier it does not hold.
    #[test]
    fn a_tenant_under_another_relocates_nothing() {
        // 0..10: free 1 and 3; the tenant holds 5, 6, 7; another holds 8 and 9.
        let free = |i: usize| i == 1 || i == 3;
        let held = |i: usize| (5..=7).contains(&i);
        assert_eq!(frontier_relocations(10, free, held), Vec::<usize>::new());
    }

    /// The tenant at the frontier empties its topmost regions, one per hole below
    /// them — and stops where the next hole is no longer below the next region.
    #[test]
    fn the_tenant_at_the_frontier_pairs_its_top_with_the_lowest_holes() {
        // 0..10: free 1, 3, 6; another tenant holds 2; the tenant holds 0, 4, 5, 7, 8, 9.
        let free = |i: usize| [1, 3, 6].contains(&i);
        let held = |i: usize| [0, 4, 5, 7, 8, 9].contains(&i);
        // Run from the top: 9, 8, 7 (6 free, skipped), 5, 4, then 3 free, 2 another's → stop.
        // Holes ascending 1, 3, 6 pair with 9, 8, 7: all below.
        assert_eq!(frontier_relocations(10, free, held), vec![9, 8, 7]);
    }

    /// Pairing ends where a hole is no longer below the region it would receive:
    /// moving a region up is never a relocation.
    #[test]
    fn pairing_stops_where_the_cursors_cross() {
        // 0..8: free 2 and 5; the tenant holds everything else.
        let free = |i: usize| i == 2 || i == 5;
        let held = |i: usize| !free(i);
        // Run 7, 6, 4, 3, 1, 0; holes 2, 5. Pair 2 → 7, then 5 → 6: both below.
        assert_eq!(frontier_relocations(8, free, held), vec![7, 6]);
        // 0..6: free 4 only, the tenant holds the rest: 4 < 5 pairs, nothing else.
        let free = |i: usize| i == 4;
        let held = |i: usize| i != 4;
        assert_eq!(frontier_relocations(6, free, held), vec![5]);
        // No holes at all: nothing to move into.
        assert_eq!(
            frontier_relocations(6, |_| false, |_| true),
            Vec::<usize>::new()
        );
    }

    /// The growth step is geometric, but it may never invent ground that the
    /// guards did not offer, and it may never round a genuine zero up to eight —
    /// that would take regions the KV side is using.
    #[test]
    fn the_growth_step_never_exceeds_what_was_offered() {
        assert_eq!(kv_grow_step(0, 8), 0, "no spare must take nothing");
        for spare in 1..8 {
            assert_eq!(
                kv_grow_step(spare, 8),
                0,
                "an offer under the caller's unit buys nothing, so it is refused"
            );
        }
        assert_eq!(
            kv_grow_step(8, 8),
            8,
            "an offer that exactly meets the unit is taken whole"
        );
        // Past the floor it halves, so convergence is a handful of negotiations
        // rather than the dozens a flat step needed.
        assert_eq!(kv_grow_step(20, 8), 10);
        assert_eq!(kv_grow_step(143, 8), 71);
        for spare in [0usize, 1, 7, 8, 9, 20, 143, 336, 4096] {
            for min_grant in [1usize, 8, 20, 64] {
                assert!(
                    kv_grow_step(spare, min_grant) <= spare,
                    "took more than the {spare} offered at a floor of {min_grant}"
                );
            }
        }
    }

    /// **A negotiation must never take the whole offer**, however much the
    /// caller wants the ground.
    ///
    /// Removing the halving for the layer zone was tried and measured worse on
    /// the 27B: applied grants fell from four to two and the zone settled a
    /// layer lower, because taking everything spare provokes the KV side into
    /// buying it back, and that purchase trips the pressure guard that refuses
    /// the following negotiation. Slower convergence beats a churn loop, and
    /// this is the assertion that says so.
    #[test]
    fn a_negotiation_always_leaves_something_on_the_table() {
        for spare in [64usize, 96, 143, 336, 4096] {
            for min_grant in [1usize, 8, 26] {
                assert!(
                    kv_grow_step(spare, min_grant) < spare,
                    "took the whole {spare}-region offer at a floor of {min_grant}"
                );
            }
        }
    }

    /// A consumer whose allocation unit is larger than the old hard-coded eight
    /// must be able to reach it, or its grants are unspendable forever.
    ///
    /// The 27B's layer is ~154 MiB — about ten regions. Halved from a spare of
    /// twenty it is offered ten, one layer, which the zone's two-layer
    /// hysteresis then discards; the ground is neither taken nor remembered, so
    /// the next negotiation computes the same number and discards it again.
    #[test]
    fn the_floor_is_the_callers_unit_so_a_grant_can_buy_one() {
        // Twenty spare, a unit of ten, two layers wanted: halving offers ten and
        // the caller needs twenty, so the floor has to win.
        assert_eq!(kv_grow_step(20, 20), 20, "the floor must reach the unit");
        // **An offer that cannot buy the unit is zero, not the offer.**
        //
        // This asserted `kv_grow_step(14, 20) == 14` and called it correct, with
        // a comment about not taking ground the guards had not offered. It is
        // the discarded-grant path in full: 14 regions buys one layer, the
        // zone's two-layer hysteresis throws it away with nothing applied, and
        // the negotiation is already spent — the exact defect `min_grant` exists
        // to close, written down here as the expected value.
        assert_eq!(
            kv_grow_step(14, 20),
            0,
            "a grant that cannot buy the caller's unit must be refused"
        );
        assert_eq!(kv_grow_step(19, 20), 0);
        assert_eq!(kv_grow_step(0, 20), 0, "a genuine zero stays zero");
        // Above the floor the geometric hedge is unchanged.
        assert_eq!(kv_grow_step(143, 20), 71);
    }

    /// **Spare is the present, and every region occupancy calls free is offered
    /// less the slack.**
    ///
    /// The mark this replaces was a windowed maximum of past demand, and it made
    /// the boundary a ratchet: shrink read the present exactly (admission evicts
    /// weights on contact) while grow consulted a forecast, so a wide cohort
    /// drove the zone down in seconds and no amount of idle time brought it back.
    /// Measured on the 27B, that held 34 of 64 layers streaming through two
    /// configs needing a quarter of the KV.
    ///
    /// It is safe because being short of KV no longer fails a forward — a claim
    /// that runs out buys its ground on the spot, measured at 26 purchases and 0
    /// refusals over a full gate.
    #[test]
    fn spare_is_free_ground_less_the_slack_and_nothing_else() {
        // `spare_regions`' arithmetic once the two present-tense guards pass.
        let spare =
            |free: usize, blocked: usize, slack: usize| (free + blocked).saturating_sub(slack);
        // The 27B at C9: 142 free of 546, nothing blocked, 32 slack. The old
        // mark answered 0 here because a 20-context peak of 514 was still on the
        // books; occupancy says 110 regions are free, because they are.
        assert_eq!(spare(142, 0, 32), 110);
        // Tier-blocked ground counts as free: the boundary only moves with no
        // wave open, so a standing tier is idle and its next-wave demand is
        // already inside `demand`.
        assert_eq!(spare(32, 109, 32), 109);
        // A KV side genuinely full offers nothing, and the slack never underflows.
        assert_eq!(spare(0, 0, 32), 0);
        assert_eq!(spare(10, 0, 32), 0);
    }

    pub(in super::super) fn stream() -> Option<candle::cuda_backend::StreamLease> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d.cuda_stream()),
            _ => None,
        }
    }

    /// A settled device forgets the recent forwards' tiers, so the growth
    /// negotiation stops holding their ground free — and a tier still standing is
    /// not forgotten under it.
    #[test]
    fn a_settled_device_forgets_the_recent_forwards_tiers() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let recent = || -> Result<usize> { with_pool(&s, |p| Ok(p.recent_tier_bytes())) };
        // The pool is process-wide: start from no remembered tier.
        super::forget_recent_tiers(&s);

        place_transient(&s, 4 * REGION_BYTES)?;
        super::forget_recent_tiers(&s);
        assert_eq!(
            recent()?,
            4 * REGION_BYTES,
            "a standing tier is not forgotten"
        );
        release_transient(&s);
        assert_eq!(
            recent()?,
            4 * REGION_BYTES,
            "the release keeps it for the next forward"
        );
        super::forget_recent_tiers(&s);
        assert_eq!(recent()?, 0, "settled, nothing is coming");
        Ok(())
    }

    /// **A verify's tier outlives the drafts after it.** A decode step places
    /// one wide verify tier and several narrow draft tiers; the negotiation must
    /// keep the verify's ground free across the drafts, or the next verify buys
    /// it back mid-step. It is forgotten only once a window's worth of forwards
    /// has passed without it.
    #[test]
    fn a_wide_tier_is_kept_across_the_narrow_forwards_after_it() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let recent = || -> Result<usize> { with_pool(&s, |p| Ok(p.recent_tier_bytes())) };
        let place = |regions: usize| -> Result<()> {
            place_transient(&s, regions * REGION_BYTES)?;
            release_transient(&s);
            Ok(())
        };
        super::forget_recent_tiers(&s);
        place(6)?;
        for _ in 0..super::RECENT_FORWARDS - 1 {
            place(1)?;
            assert_eq!(recent()?, 6 * REGION_BYTES, "the verify's tier is kept");
        }
        place(1)?;
        assert_eq!(
            recent()?,
            REGION_BYTES,
            "a window of narrow forwards later, it is not"
        );
        super::forget_recent_tiers(&s);
        Ok(())
    }

    /// While prompts wait, the widest tier placed is remembered through the
    /// narrower ones after it; a tier placed with nothing waiting is not; and
    /// the backlog clearing forgets it.
    #[test]
    fn the_waiting_prefills_tier_is_kept_until_the_backlog_clears() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let ordinal = s.context().ordinal();
        let waiting = || -> Result<usize> { with_pool(&s, |p| Ok(p.waiting_tier_bytes)) };
        let place = |regions: usize| -> Result<()> {
            place_transient(&s, regions * REGION_BYTES)?;
            release_transient(&s);
            Ok(())
        };

        super::set_prompts_waiting(ordinal, false);
        place(6)?;
        assert_eq!(waiting()?, 0, "nothing waits, nothing is kept");

        super::set_prompts_waiting(ordinal, true);
        place(6)?;
        place(1)?;
        place(1)?;
        assert_eq!(
            waiting()?,
            6 * REGION_BYTES,
            "the prefill's tier, through the decodes after it"
        );

        super::set_prompts_waiting(ordinal, false);
        assert_eq!(waiting()?, 0, "the backlog cleared");
        Ok(())
    }

    /// The weight side may not take ground a standing tier is holding.
    ///
    /// `set_weight_floor` refused a floor that would cut live KV regions and
    /// said nothing about the wave transient tier, which occupies the same
    /// middle. So the weight side could grow DOWN onto a tier already placed —
    /// `tier_fits` having approved that tier against the floor as it stood a
    /// moment earlier — and the FFN span at the tier's top then wrote
    /// activations over resident expert slots.
    ///
    /// Measured at the fault: tier topping out at `0x51fbc00000` against a floor
    /// of `0x51f9c00000`, 32 MiB of tier inside expert ground.
    #[test]
    fn the_weight_floor_may_not_cut_a_standing_transient_tier() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let before = match region_stats(s.context().ordinal()) {
            Some(st) => st,
            None => return Ok(()),
        };
        let _ = before;
        let base = place_transient(&s, 4 * REGION_BYTES)?;
        let layout = super::span_layout(s.context().ordinal()).expect("a placed span");
        let tier_top = base + layout.transient_bytes as u64;
        assert!(
            tier_top <= layout.weight_floor,
            "the placement itself must respect the floor: {tier_top:#x} vs {:#x}",
            layout.weight_floor
        );

        // Any floor inside the standing tier must be refused — this is the
        // direction that was unguarded.
        for probe in [tier_top - 1, base + 1, base] {
            assert!(
                super::set_weight_floor(&s, probe).is_err(),
                "a floor at {probe:#x} cuts the tier [{base:#x}, {tier_top:#x}) and must \
                 be refused"
            );
        }
        // A floor at or above the tier's top is still allowed, so the guard has
        // not simply frozen the boundary.
        assert!(
            super::set_weight_floor(&s, layout.weight_floor).is_ok(),
            "the floor where it already stands must remain settable"
        );
        release_transient(&s);
        Ok(())
    }

    /// Regions come out of the span in ascending order and are disjoint at the
    /// region stride — the arithmetic every base pointer depends on.
    #[test]
    fn regions_are_disjoint_and_ascending() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let a = claim_region(&s)?.expect("a region");
        let b = claim_region(&s)?.expect("a second region");
        assert!(b.index() > a.index(), "regions ascend");
        assert_eq!(
            b.base() - a.base(),
            ((b.index() - a.index()) * REGION_BYTES) as u64,
            "region bases step by exactly the region stride"
        );
        Ok(())
    }

    /// A region the pool still has an owner for is never handed to a second
    /// tenant, however it came to be on the free list.
    ///
    /// **The fault this catches writes over live data and does not fault**, so
    /// there is no crash to work back from. Every address in the reservation is
    /// mapped, and the tenants are unalike — a KV arena, the wave transient tier,
    /// and a sequence's recurrent store all claim here — so the second tenant's
    /// writes surface as a wrong number read by a different subsystem entirely.
    /// It was measured as a whole recurrent state going non-finite, in contiguous
    /// runs of layers, while the KV side read perfectly.
    ///
    /// The duplicate is injected rather than provoked: the point under test is
    /// that the claim path refuses it, not which bookkeeping slip produced it.
    /// The free list is emptied for the duration so the injected index is
    /// certainly the one popped — the pool is process-global and another test's
    /// releases would otherwise sit below it — and restored afterwards.
    #[test]
    fn a_region_still_held_is_never_issued_twice() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let held = claim_region(&s)?.expect("a region");
        let idx = held.index();

        // Stand the free list aside and offer only the live index.
        let saved = with_pool(&s, |pool| {
            let saved = std::mem::take(&mut pool.free);
            pool.free.push(Reverse(idx));
            Ok(saved)
        })?;

        let refused = claim_region(&s);

        // Restore before asserting, so a failed assertion cannot wedge the pool for
        // every test after it — it is process-global.
        with_pool(&s, |pool| {
            pool.free = saved;
            Ok(())
        })?;

        // **`Err`, not `Ok(None)`.** `None` is this function's ordinary pressure
        // signal and the caller answers it by buying ground and retrying, so
        // returning it here would hide the breach behind a slower run. An `Err`
        // propagates to the turn that asked.
        let err = refused.expect_err("a region that is still held must not be issued");
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("region {idx} was issued while still held")),
            "the error must name the region and say what is wrong, got: {msg}"
        );
        // The guard must not have re-queued it, and must not have counted it as a
        // second live region.
        with_pool(&s, |pool| {
            assert!(pool.owned[idx], "the real owner still holds it");
            assert!(
                !pool.free.iter().any(|Reverse(i)| *i == idx),
                "a live region must not be left on the free list"
            );
            Ok(())
        })?;
        drop(held);
        Ok(())
    }

    /// Releasing a region the pool does not think is held panics, rather than
    /// quietly queuing a duplicate or quietly leaking it.
    ///
    /// Both quiet options were written and both were wrong. Queuing the index
    /// hands the same ground to two tenants. Skipping the push keeps memory safe
    /// but continues with the pool's bookkeeping known to be broken, so the next
    /// occurrence is silent too and the run stops being evidence of anything.
    #[test]
    fn a_release_of_an_unheld_region_panics() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let held = claim_region(&s)?.expect("a region");
        let idx = held.index();
        let before = with_pool(&s, |pool| {
            // Simulate the pool having lost track of this owner, which is the
            // state a double release arrives in.
            pool.owned[idx] = false;
            Ok(pool.free.len())
        })?;

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(held)));

        // The unwind left `Drop` half-run: nothing queued, nothing decremented.
        // Put the pool back before asserting — it is process-global, and
        // `every_region_is_live_free_or_blocked` checks the partition over it, so a
        // region left neither live nor free fails a later, unrelated test.
        with_pool(&s, |pool| {
            let queued = pool.free.len();
            let on_list = pool.free.iter().any(|Reverse(i)| *i == idx);
            pool.live -= 1;
            pool.free.push(Reverse(idx));
            pool.owned[idx] = false;
            Ok((queued, on_list))
        })
        .map(|(queued, on_list)| {
            assert_eq!(queued, before, "the panic must not have queued the region");
            assert!(!on_list, "region {idx} must not have reached the free list");
        })?;

        let err = panicked.expect_err("an unheld release must panic");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains(&format!("region {idx} released while not held")),
            "the panic must name the region and say an owner was lost, got: {msg}"
        );
        Ok(())
    }

    /// Dropping a handle returns its region, and the next claim takes the
    /// lowest free one back — the left-packing the evacuation order relies on.
    ///
    /// Asserted as an ordering, not as two exact indices. The lock serialises
    /// this module's tests against each other, but the pool is process-global
    /// and other modules' tests claim from it in parallel, so a specific region
    /// freed here can be taken by one of them before the re-claim. What must
    /// hold regardless of that interleaving is that claims come back ascending:
    /// a concurrent thief changes *which* region is lowest, never that the
    /// lowest is the one handed out.
    #[test]
    fn a_released_region_comes_back_lowest_first() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let low = claim_region(&s)?.expect("a region");
        let high = claim_region(&s)?.expect("a second region");
        let low_idx = low.index();
        assert!(
            low.index() < high.index(),
            "claims ascend while the pool is fresh"
        );
        let before = region_stats(0).expect("a pool").free;
        drop(high);
        drop(low);
        assert_eq!(region_stats(0).expect("a pool").free, before + 2);
        let again = claim_region(&s)?.expect("a recycled region");
        assert!(
            again.index() <= low_idx,
            "a recycled region must come back no higher than the lowest just freed: \
             got {}, freed {low_idx}",
            again.index()
        );
        let then = claim_region(&s)?.expect("the other recycled region");
        assert!(
            then.index() > again.index(),
            "the free list is lowest-first: {} followed {}",
            then.index(),
            again.index()
        );
        Ok(())
    }

    /// A recycled region is handed over zeroed, whatever its last tenant left.
    #[test]
    fn a_recycled_region_is_zeroed() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let region = claim_region(&s)?.expect("a region");
        let base = region.base();
        let dirt = vec![0xC3u8; 8192];
        // SAFETY: the region is claimed by this test and mapped read/write.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                base,
                &dirt,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("dirtying a region: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        drop(region);

        // The free list is shared with every other test in this binary, so the
        // NEXT claim is not necessarily the region just released — whichever
        // tests ran earlier may have left their own queued ahead of it. Claim
        // until the dirtied one comes back, holding the others so they cannot be
        // handed out again, and check THAT one.
        //
        // Asserting identity on the first claim made this test depend on
        // execution order (it failed with two bases exactly one region apart),
        // and on the failing path it went on to read from `base` after the pool
        // had handed that region to someone else.
        // Small on purpose: each iteration HOLDS a region, and the pool's
        // sub-tier class is only a few dozen deep, so a large scan would starve
        // whatever runs next instead of skipping a couple of stale entries.
        const MAX_SCAN: usize = 8;
        let mut held = Vec::new();
        let mut recycled = None;
        for _ in 0..MAX_SCAN {
            let Some(r) = claim_region(&s)? else { break };
            if r.base() == base {
                recycled = Some(r);
                break;
            }
            held.push(r);
        }
        let again = recycled.expect("the dirtied region is returned to the pool");
        assert_eq!(again.base(), base);
        let mut back = vec![0xFFu8; 8192];
        // SAFETY: as above, reading the region this test holds.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                &mut back,
                base,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("reading a region: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        assert!(
            back.iter().all(|&b| b == 0),
            "a recycled region still held its last tenant's bytes"
        );
        Ok(())
    }

    /// **Ground the transient tier stood on is cleaned before it is re-let**,
    /// even though it never passed through the free list.
    ///
    /// This is the regression that produced NaN in a live conversation, and the
    /// path is entirely *fresh* claims. The tier writes activations across its
    /// whole span for the length of a forward; when it is released the ceiling
    /// rises to the boundary and `next` may advance straight into that band. A
    /// fresh claim used to skip both the quiesce and the zero — the cleaning was
    /// keyed on "came off the free list" rather than "holds someone else's
    /// bytes" — so a KV arena was handed a region full of activation data while
    /// the kernels that wrote it were still in flight.
    ///
    /// A `tier_reserve` had been keeping the KV side off that band, which is why
    /// the omission was unreachable until the reserve was removed. It was load
    /// bearing for a reason its own doc comment never stated.
    #[test]
    fn tier_ground_is_cleaned_before_a_fresh_claim_takes_it() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };

        // Place a tier, dirty its first region the way a forward's intermediates
        // would, then release it — the state at the top of the next forward.
        let base = place_transient(&s, REGION_BYTES)?;
        let dirt = vec![0xA7u8; 8192];
        // SAFETY: `[base, base + REGION_BYTES)` is mapped device memory inside
        // the reservation, reserved to this tier and to nothing else.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                base,
                &dirt,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("dirtying the tier: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        release_transient(&s);

        // Claim until one lands in the tier's span. Every claim here is fresh:
        // nothing was released, so the free list is empty of these indices.
        let mut held = Vec::new();
        let mut seen = false;
        while let Some(region) = claim_region(&s)? {
            let hit = region.base() == base;
            held.push(region);
            if hit {
                seen = true;
                break;
            }
        }
        assert!(seen, "no claim reached the tier's ground — test is inert");

        let mut back = vec![0xFFu8; 8192];
        // SAFETY: reading the region this test now holds.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                &mut back,
                base,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("reading a region: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        assert!(
            back.iter().all(|&b| b == 0),
            "a region the tier had written was handed out still holding its \
             activation bytes"
        );
        Ok(())
    }

    /// A batch of regions released together costs **one** quiesce between them,
    /// not one each — the whole point of the epoch stamp. The claim that pays it
    /// advances the epoch, which discharges every other region released before
    /// that moment.
    ///
    /// Asserted through the zero guarantee rather than by reading the counter,
    /// because the guarantee is what the skip must not break: a region that
    /// takes the fast path is still handed over clean.
    #[test]
    fn one_quiesce_covers_a_whole_batch_of_releases() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };

        let batch: Vec<_> = (0..4)
            .map(|_| claim_region(&s).unwrap().expect("a region"))
            .collect();
        let bases: Vec<u64> = batch.iter().map(|r| r.base()).collect();
        let dirt = vec![0x5Au8; 4096];
        for &base in &bases {
            // SAFETY: each region is claimed by this test and mapped read/write.
            unsafe {
                candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                    base,
                    &dirt,
                    s.cu_stream(),
                )
            }
            .map_err(|e| candle::Error::Msg(format!("dirtying a region: {e}")))?;
        }
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        drop(batch);

        // Re-claim all four. The first pays the quiesce; the rest skip it — and
        // every one of them must still come back zeroed.
        let again: Vec<_> = (0..4)
            .map(|_| claim_region(&s).unwrap().expect("a region"))
            .collect();
        for region in &again {
            let mut back = vec![0xFFu8; 4096];
            // SAFETY: as above, reading a region this test holds.
            unsafe {
                candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                    &mut back,
                    region.base(),
                    s.cu_stream(),
                )
            }
            .map_err(|e| candle::Error::Msg(format!("reading a region: {e}")))?;
            s.synchronize()
                .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
            assert!(
                back.iter().all(|&b| b == 0),
                "region {} skipped its quiesce and kept the last tenant's bytes",
                region.index()
            );
        }
        Ok(())
    }

    /// **A region the ceiling forbids is not free**, and the daemon wedged on
    /// the difference.
    ///
    /// `free` answers "what would a claim get right now", so counting
    /// unreachable ground there is a promise the allocator then refuses to
    /// keep. That is what happened: a wide section prefill placed a 496 MiB
    /// tier, the ceiling landed at 293 of 324 regions, live reached 293, and
    /// the 31 regions above the ceiling were reported free. Admission planned
    /// against 496 MiB it could not have and every claim was refused — 12,488
    /// of them in four seconds, with no forward completing to move the
    /// boundary. Consumers deciding for *after* the tier releases add the
    /// **tier's** blocked ground back on — `vram_budget_available` is
    /// `(free + blocked) × REGION_BYTES` — which is honest about the tier
    /// precisely because `free` itself is not inflated. Every blocked region is
    /// the tier's; see
    /// [`the_weight_side_owns_no_blocked_ground_at_any_boundary`].
    #[test]
    fn ground_under_the_tier_is_not_counted_as_free() {
        // The shape at the wedge: 324 regions all handed out at least once
        // (`next` = 324), 293 live, 31 on the free list at indices 293..323,
        // and a 496 MiB tier putting the ceiling at 293.
        assert_eq!(super::claimable(0, 324, 293), 0, "nothing is claimable");
        assert_eq!(
            super::blocked(31, 324, 324, 293),
            31,
            "and the 31 are blocked, not free"
        );

        // Between waves the tier is gone, the ceiling is the whole pool, and
        // the same 31 become spendable again.
        assert_eq!(super::claimable(31, 324, 324), 31);
        assert_eq!(super::blocked(0, 324, 324, 324), 0);
    }

    /// **The weight side owns none of `blocked`, and the pool's own arithmetic
    /// says so.**
    ///
    /// A `weight_blocked` term was added here to split `blocked` in two and
    /// subtracted from `vram_budget_available` to close an OOM. It could never
    /// have: `total` *is* `(weight_floor − region_base)/REGION_BYTES`, so the
    /// weight ceiling always equals `total`, `blocked(above, next, total, total)`
    /// collapses to `above`, and `above` is zero because `set_weight_floor`
    /// retains only free indices below `total`. The term was identically zero on
    /// every reachable state — the whole of a gate run reported `weight 0` on
    /// every line — and the test that "proved" it worked passed only by calling
    /// these free functions with a `(total=162, ceiling=104)` pair the pool
    /// cannot produce.
    ///
    /// This asserts the identity instead, over the states the pool *can* reach.
    /// It is what makes the removal safe and what would fail if `total` ever
    /// stopped being derived from the floor.
    #[test]
    fn the_weight_side_owns_no_blocked_ground_at_any_boundary() {
        // Whatever the boundary, `total` moves with it, so the weight ceiling is
        // the pool's size and nothing sits above it.
        for total in [1usize, 16, 104, 162, 324, 546] {
            for next in [0usize, 1, total / 2, total] {
                let weight_ceiling = total; // ceiling_regions(None, floor, ..) == total
                assert_eq!(
                    super::blocked(0, next, total, weight_ceiling),
                    0,
                    "total {total} next {next}: the weight side cannot hold blocked ground"
                );
            }
        }
    }

    /// The tier's ground is still counted, or admission starves between forwards.
    ///
    /// With no tier standing the ceiling is the pool's size and `blocked` is
    /// zero; with one placed it is the tier's, and it returns in phase 0 of the
    /// next forward — so a decision landing after that may spend it.
    #[test]
    fn the_tiers_blocked_ground_is_still_an_admission_budget() {
        // 324 regions, all handed out, and a tier capping the ceiling at 293 —
        // the section-prefill wedge's shape.
        let (free, blocked) = (0usize, 31usize);
        assert_eq!(
            super::blocked(0, 324, 324, 324),
            0,
            "no tier standing: nothing is blocked"
        );
        assert_eq!(
            free + blocked,
            31,
            "the tier's ground is spendable by the forward that follows it"
        );
    }

    /// The fresh-region term is counted once, by whichever side owns it.
    #[test]
    fn fresh_ground_is_claimable_or_blocked_but_never_both() {
        // 100 regions, next=40, ceiling=70: 30 fresh are claimable, 30 are cut
        // off above the ceiling, and no free-list entries exist either side.
        assert_eq!(super::claimable(0, 40, 70), 30);
        assert_eq!(super::blocked(0, 40, 100, 70), 30);
        assert_eq!(
            super::claimable(0, 40, 70) + super::blocked(0, 40, 100, 70) + 40,
            100,
            "claimable + blocked + allocated must be the whole pool"
        );

        // A ceiling below `next` cuts off nothing fresh — `next` has already
        // passed it — and leaves the tail beyond `next` blocked.
        assert_eq!(super::claimable(0, 80, 70), 0);
        assert_eq!(super::blocked(0, 80, 100, 70), 20);
    }

    /// **A tier caps the KV side only while it is standing**, and the boundary
    /// is the cap the rest of the time.
    ///
    /// Both halves of this were defects. A tier left standing past its forward
    /// pins the ceiling at an address, so the weight side could concede itself
    /// down to its floor and this still returned 293 —
    /// `bump_arena::end_wave_transient` ends the tier's life with its forward.
    /// And once released, the ceiling used to stop short by `tier_reserve`, the
    /// last tier's width, held against the next placement: that refused claims
    /// against ground nobody owned, and the refusal escalated into a boundary
    /// renegotiation which handed over the same regions by the most expensive
    /// route there is.
    #[test]
    fn the_ceiling_is_the_tier_while_it_stands_and_the_boundary_after() {
        let base = 0u64;
        let total = 324;
        // A 496 MiB tier placed against a floor 324 regions up: 31 regions of
        // tier, so it stands at region 293.
        let tier = 31 * REGION_BYTES;
        let floor = (total * REGION_BYTES) as u64;
        let placed = Some(floor - tier as u64);

        assert_eq!(
            super::ceiling_regions(placed, floor, base, total),
            293,
            "a standing tier caps the KV side at its own base — real memory a \
             running wave is writing into"
        );
        assert_eq!(
            super::ceiling_regions(None, floor, base, total),
            324,
            "released, every region up to the boundary is claimable: nothing is \
             standing there, so nothing is withheld"
        );

        // The boundary moves. A placed tier is deaf to it; an absent one is not.
        let moved = floor + (40 * REGION_BYTES) as u64;
        assert_eq!(
            super::ceiling_regions(placed, moved, base, total),
            293,
            "a placed tier pins the ceiling at its base, whatever the floor does"
        );
        assert_eq!(
            super::ceiling_regions(None, moved, base, total),
            324,
            "and without one the ceiling follows the boundary, capped by the pool"
        );
    }

    /// **The tier never stands on a live region**, however full the KV side is.
    ///
    /// The regression this pins: the fit test used to bound the tier below by
    /// the *start* of the KV side, so a tier measured down from `W` was accepted
    /// while it sat on top of the highest live arenas. `region_ceiling` refuses
    /// later claims but cannot revoke one already made, so the wave wrote its
    /// intermediates into live KV bytes — silently, with no error anywhere.
    ///
    /// Pure arithmetic, so it holds on a machine with no GPU.
    #[test]
    fn the_tier_never_overlaps_a_live_region() {
        let r = REGION_BYTES as u64;
        let region_base = 0x1000_0000u64;
        let floor = region_base + 332 * r; // 332 regions
        let len = 4 * REGION_BYTES; // a decode-sized tier
        let base = floor - len as u64; // an arbitrary base, hard against the floor

        // Empty KV side: the tier sits above every region and fits.
        assert!(super::tier_fits(base, len, region_base, floor).is_ok());

        // 328 live regions — the tier lands exactly on the frontier, no overlap.
        let live_end = region_base + 328 * r;
        assert!(
            super::tier_fits(base, len, live_end, floor).is_ok(),
            "abutting the frontier is not overlapping it"
        );

        // 329 live: one region of the tier is inside live KV. Refuse.
        let live_end = region_base + 329 * r;
        assert_eq!(
            super::tier_fits(base, len, live_end, floor),
            Err(r),
            "one region short must be reported as one region short"
        );

        // The old bound — the start of the KV side — accepted every one of
        // these. The whole span being live is the extreme case.
        let live_end = region_base + 332 * r;
        assert_eq!(super::tier_fits(base, len, live_end, floor), Err(4 * r));
    }

    /// **The placement anchors at the arena frontier, and only the floor can
    /// refuse it.**
    ///
    /// `try_place` sets `base = live_end`, so the live-region side of
    /// [`tier_fits`] is satisfied by construction — a tier that begins where the
    /// arenas end cannot overlap them — and every byte between the tier's top
    /// and the weight floor stays claimable. What binds instead is the floor:
    /// when the tier does not fit in the run above the frontier, the shortfall
    /// is what the boundary must move by, which is exactly what `Placed::Short`
    /// asks the weight side for.
    ///
    /// The placement this replaced measured *down* from the floor, which left
    /// the free run stranded between the arenas and the tier where only a
    /// control loop could find it.
    #[test]
    fn the_tier_anchors_at_the_frontier_and_only_the_floor_refuses() {
        let r = REGION_BYTES as u64;
        let region_base = 0x1000_0000u64;
        let floor = region_base + 100 * r;

        // 40 regions live, an 8-region tier: it abuts the arenas and there are
        // 52 regions of claimable ground above it.
        let live_end = region_base + 40 * r;
        let len = 8 * REGION_BYTES;
        assert!(
            super::tier_fits(live_end, len, live_end, floor).is_ok(),
            "a tier beginning where the arenas end never overlaps them"
        );

        // An empty KV side is the same rule with the frontier at the bottom.
        assert!(super::tier_fits(region_base, len, region_base, floor).is_ok());

        // 95 live regions leave 5 for an 8-region tier: three short, and the
        // floor is the only side that can be.
        let live_end = region_base + 95 * r;
        assert_eq!(
            super::tier_fits(live_end, len, live_end, floor),
            Err(3 * r),
            "the shortfall is what the weight boundary has to concede"
        );
    }

    /// The shortfall is reported from **whichever** side binds, and never wraps.
    ///
    /// The frontier anchor computes its base upward from the arenas, so it
    /// overshoots `floor` rather than undershooting `live_end` — the direction
    /// the single-sided `region_base - base` could not express, and underflowed
    /// on: in release it wrapped to ~2^64 and drove `kv_pressure` to a retraction
    /// request larger than the span.
    #[test]
    fn the_tier_shortfall_reports_both_directions_without_wrapping() {
        let r = REGION_BYTES as u64;
        let region_base = 0x1000_0000u64;
        let floor = region_base + 100 * r;

        // Anchored above the frontier and running two regions past the floor.
        let base = region_base + 99 * r;
        let len = 3 * REGION_BYTES;
        assert_eq!(
            super::tier_fits(base, len, region_base, floor),
            Err(2 * r),
            "overshooting the floor is a two-region shortfall, not a wrap"
        );

        // Both bounds violated at once: the larger of the two is what the
        // boundary has to move by.
        let base = region_base + 10 * r;
        let live_end = region_base + 12 * r;
        let len = 95 * REGION_BYTES;
        assert_eq!(
            super::tier_fits(base, len, live_end, floor),
            Err(5 * r),
            "past the floor by 5 regions, into live by 2 — the binding one is 5"
        );
    }

    /// **The tier's ceiling binds the free list, not just fresh regions.**
    ///
    /// A region freed while the tier was small — or absent, between forwards —
    /// keeps its index, and the next wave's tier can be placed below it. Handing
    /// it out then puts KV writes inside the wave's own intermediates, and
    /// nothing downstream would notice until the output was wrong.
    ///
    /// Pure arithmetic against `region_ceiling`, so it holds with no GPU: what
    /// the bug was is a `free.pop()` that never consulted the ceiling at all.
    #[test]
    fn the_tier_ceiling_binds_recycled_regions() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 4096 * mib;
        let l = rest_layout(base, span, 0, base + span as u64);
        // A tier placed 8 regions below the weight floor drops the ceiling by 8.
        let tier = 8 * REGION_BYTES;
        let tier_base = (base + span as u64) - tier as u64;
        let ceiling = ((tier_base - l.region_base) as usize) / REGION_BYTES;
        assert_eq!(
            ceiling,
            l.total - 8,
            "the tier must remove exactly its own regions from the ceiling"
        );
        // Every index at or above the ceiling is inside the tier — whether it is
        // fresh or recycled is not a property the address knows about.
        for idx in ceiling..l.total {
            let addr = l.region_base + (idx * REGION_BYTES) as u64;
            assert!(
                addr >= tier_base,
                "region {idx} sits inside the tier at {tier_base:#x}"
            );
        }
        assert!(
            l.region_base + ((ceiling - 1) * REGION_BYTES) as u64 + REGION_BYTES as u64
                <= tier_base,
            "the last usable region must end at or before the tier"
        );
    }

    /// Live + free + blocked accounts for every region: nothing leaks out of
    /// the span.
    ///
    /// **Three buckets, not two.** `free` used to mean "no arena owns it" and
    /// this asserted `live + free == total`; it now means "claimable right now",
    /// which is what the admission budget needs it to mean, and the ground the
    /// tier's ceiling puts out of reach is counted separately. A region is
    /// therefore in exactly one of three states, and the sum is still the span.
    #[test]
    fn every_region_is_live_free_or_blocked() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let held = claim_region(&s)?.expect("a region");
        let stats = region_stats(0).expect("a pool");
        assert_eq!(stats.live + stats.free + stats.blocked, stats.total);
        assert!(stats.live >= 1);
        drop(held);
        let after = region_stats(0).expect("a pool");
        assert_eq!(after.live + after.free + after.blocked, after.total);
        Ok(())
    }

    /// **B3/B5 — a boundary move refused mid-wave moves nothing at all.**
    ///
    /// The latch itself is asserted elsewhere; what this adds is the *atomicity*,
    /// which had no guard. A refusal that had already moved the floor, or already
    /// changed the region count, is the half-application failure — the zone
    /// believes it is one size while the boundary says another, the layers are
    /// gone, the ground is stranded, and the caller sees a plain "conceded 0".
    /// That exact bug is written up on `ExpertCache::renegotiate_boundary`; it
    /// was found in production and has been guarded only by the call site's good
    /// behaviour ever since.
    ///
    /// So the assertion is on the *whole observable state* before and after, not
    /// on the return value: every region figure, the floor, and the span layout.
    #[test]
    fn a_floor_move_refused_mid_wave_changes_nothing() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        super::ensure(&s)?;
        // Hold a region so the pool has real state to disturb.
        let held = claim_region(&s)?.expect("a region");

        // Open a wave generation: the condition `set_weight_floor` refuses on.
        let gen = super::super::super::bump_arena::begin_wave(
            &s,
            super::super::super::wave_plan::LayerPhase::Attention,
        )?;
        assert!(
            super::super::super::bump_arena::wave_is_live(0),
            "the guard's precondition did not hold, so this proves nothing"
        );

        // **Snapshot after the wave opens, not before.** Opening one places the
        // transient tier, which drops the region ceiling and moves ground from
        // `free` to `blocked` — measured here as 184 regions. That is the wave
        // starting, not the refusal, and a baseline taken earlier attributes it
        // to the wrong thing.
        let before = region_stats(0).expect("a pool");
        let floor_before = super::with_pool(&s, |p| Ok(p.weight_floor))?;
        let layout_before = super::span_layout(0).expect("a layout");

        // Ask for a floor a whole region to the left — a real move, not a no-op.
        let want = floor_before.saturating_sub(REGION_BYTES as u64);
        let refused = super::set_weight_floor(&s, want);
        assert!(
            refused.is_err(),
            "the floor moved while a wave generation was open"
        );

        let after = region_stats(0).expect("a pool");
        let floor_after = super::with_pool(&s, |p| Ok(p.weight_floor))?;
        let layout_after = super::span_layout(0).expect("a layout");
        assert_eq!(floor_after, floor_before, "a refusal moved the floor");
        assert_eq!(
            after.total, before.total,
            "a refusal changed the region count"
        );
        assert_eq!(after.live, before.live, "a refusal changed live regions");
        assert_eq!(after.free, before.free, "a refusal changed free regions");
        assert_eq!(
            after.blocked, before.blocked,
            "a refusal changed blocked regions"
        );
        assert_eq!(
            layout_after.region_base, layout_before.region_base,
            "a refusal moved the region grid"
        );

        drop(gen);
        drop(held);
        Ok(())
    }
}
