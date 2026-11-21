//! Tests for the warm tier's sizing arithmetic, with every reading of the machine
//! passed in as an argument.
//!
//! The tests pin that the launch reading beats a trough depressed by the engine's
//! own mapped checkpoint, that no ceiling may take the whole pinnable region, and
//! that the lowest ceiling binds and is named. They also pin that the process's
//! own heap is not sized over, that pinned bytes are subtracted exactly once
//! whichever reading wins, and that a machine below the headroom takes no tier
//! rather than underflowing into an enormous one.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod warm_sizing_tests {
    use super::{
        launch_less_own, warm_sizing_from, CEILING_AVAILABLE, CEILING_HOST_BUDGET, CEILING_NONE,
        CEILING_PINNABLE, PAGEABLE_RESERVE, WARM_TIER_HEADROOM,
    };

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Bytes as GiB, so a failure message reads in the units the budget is
    /// reasoned about in rather than eleven digits.
    fn gib(b: u64) -> f64 {
        b as f64 / GIB as f64
    }
    /// The 3.6-35B's slot: three projections at their aligned offsets.
    const SLOT: usize = 1_933_312;
    /// Its evictable set — 39 unpinned layers of 256.
    const EVICTABLE: usize = 9_984;

    /// A budget generous enough not to be the binder, so a case can isolate one
    /// of the other two ceilings.
    const LOOSE_BUDGET: u64 = 1024 * GIB;

    /// **The regression this exists for.** The 16 GB dev box, sized from the
    /// mid-load trough (12.18 GiB) against the launch reading (20.45 GiB).
    ///
    /// The live figure is depressed by the engine's own mapped checkpoint, and
    /// sizing from it cost 3,030 experts — every one of which then reads the
    /// pack on a miss, for the life of the process.
    #[test]
    fn the_launch_baseline_beats_the_mid_load_trough() {
        let args = |launch: u64| {
            warm_sizing_from(
                SLOT,
                EVICTABLE,
                31 * GIB + GIB / 2,
                12 * GIB + GIB / 5, // 12.18 GiB free mid-load
                launch,
                640 * 1024 * 1024, // embedding + staging already pinned
                27 * GIB,
            )
        };
        let trough = args(12 * GIB + GIB / 5);
        let launch = args(20 * GIB + GIB / 2);
        assert!(
            launch.slots > trough.slots + 3000,
            "launch baseline bought only {} slots over the trough's {}",
            launch.slots,
            trough.slots
        );
        // Availability is the ceiling that is *supposed* to decide on a machine
        // like this — the pinnable reserve is a backstop, not the everyday
        // binder. Both cases are availability-bound; the launch reading simply
        // gives it a truthful number to work from.
        assert_eq!(trough.bound_by, CEILING_AVAILABLE);
        assert_eq!(launch.bound_by, CEILING_AVAILABLE);
    }

    /// **Every ceiling leaves the headroom, including the pinnable one.**
    ///
    /// The regression: on a settled 31.5 GiB box the launch baseline (20.05 GiB)
    /// lifted the availability ceiling above the pinnable one for the first
    /// time, the pinnable ceiling bound, and — having no reserve subtracted from
    /// it — handed the tier the entire pinnable region with nothing left over.
    /// The first forward died on `CUDA_ERROR_OUT_OF_MEMORY`.
    ///
    /// The invariant: whatever binds, total pinned must stay a headroom clear of
    /// the pinnable limit. Exercised on a machine small enough that the pinnable
    /// ceiling is the one that wins.
    #[test]
    fn no_ceiling_may_take_the_whole_pinnable_region() {
        let total = 16 * GIB;
        let already = 640 * 1024 * 1024;
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            total,
            60 * GIB, // free RAM is not the constraint here
            60 * GIB,
            already,
            LOOSE_BUDGET,
        );
        assert_eq!(s.bound_by, CEILING_PINNABLE);
        assert!(
            s.taken_bytes + already + WARM_TIER_HEADROOM + PAGEABLE_RESERVE <= total,
            "tier {:.2} GiB + {:.2} GiB already pinned does not leave the {:.2} GiB headroom \
             and {:.2} GiB pageable reserve inside {:.2} GiB",
            s.taken_bytes as f64 / GIB as f64,
            already as f64 / GIB as f64,
            WARM_TIER_HEADROOM as f64 / GIB as f64,
            PAGEABLE_RESERVE as f64 / GIB as f64,
            total as f64 / GIB as f64,
        );
    }

    /// A machine with no room left over after the pageable reserve takes no
    /// warm tier at all, rather than underflowing into an enormous one.
    #[test]
    fn a_machine_smaller_than_the_pageable_reserve_takes_nothing() {
        for total_gib in [4u64, 8, 10, 12] {
            let s = warm_sizing_from(
                SLOT,
                EVICTABLE,
                total_gib * GIB,
                60 * GIB,
                60 * GIB,
                0,
                LOOSE_BUDGET,
            );
            assert_eq!(
                s.slots, 0,
                "{total_gib} GiB machine has no pinnable room past the reserve"
            );
        }
    }

    /// The same invariant, swept: no combination of machine size, free RAM or
    /// standing pinned bytes may leave the pageable reserve short. A ceiling
    /// that can bind has to leave the same room as its neighbours, so this holds
    /// whichever one wins.
    #[test]
    fn the_pageable_reserve_holds_across_machines() {
        for total_gib in [8u64, 16, 31, 64, 194] {
            for free_gib in [2u64, 8, 30, 120] {
                for pinned_gib in [0u64, 1, 6] {
                    let total = total_gib * GIB;
                    let already = pinned_gib * GIB;
                    let s = warm_sizing_from(
                        SLOT,
                        EVICTABLE,
                        total,
                        free_gib * GIB,
                        free_gib * GIB,
                        already,
                        LOOSE_BUDGET,
                    );
                    // Stated as what the tier may take, not as what total pinned
                    // must be: a process that has *already* pinned past the
                    // limit is not something this function can undo, and the
                    // only correct answer there is to take nothing — which
                    // `saturating_sub` gives.
                    assert!(
                        s.taken_bytes
                            <= total
                                .saturating_sub(PAGEABLE_RESERVE)
                                .saturating_sub(already)
                                .saturating_sub(WARM_TIER_HEADROOM),
                        "{total_gib} GiB machine, {free_gib} GiB free, {pinned_gib} GiB pinned: \
                         tier took {} bytes",
                        s.taken_bytes
                    );
                }
            }
        }
    }

    /// A machine that gained free RAM since launch is sized on the larger, newer
    /// figure — the baseline is a floor on optimism, not a cap.
    #[test]
    fn a_machine_that_freed_ram_is_not_held_to_its_launch_reading() {
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            40 * GIB, // plenty free now
            8 * GIB,  // but the process launched on a busy machine
            0,
            LOOSE_BUDGET,
        );
        assert_eq!(s.launch_ram, 40 * GIB);
        assert_eq!(s.available_less_headroom, 40 * GIB - WARM_TIER_HEADROOM);
    }

    /// **Pinned bytes are subtracted exactly once, whichever reading wins.**
    ///
    /// The live figure is taken after the process page-locked them, so it
    /// already excludes them; the launch figure predates them and does not.
    /// Subtracting from the winner double-counts on every machine where the live
    /// reading is larger — the case the `max` exists for. The two tests either
    /// side of this one are both blind to it: one passes `already_pinned = 0`,
    /// the other makes launch and live equal.
    #[test]
    fn pinned_bytes_are_not_double_subtracted_when_the_live_reading_wins() {
        let already = 4 * GIB;
        // Live is the larger *and* already nets out `already`; launch predates
        // it. Normalised, both describe the same 40 GiB of usable ground.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            40 * GIB,
            44 * GIB,
            already,
            LOOSE_BUDGET,
        );
        assert_eq!(
            s.available_less_headroom,
            40 * GIB - WARM_TIER_HEADROOM,
            "the pinned bytes were counted twice"
        );
    }

    /// **Already-pinned bytes come out of the LAUNCH reading, which predates
    /// them — never out of the live one, which already excludes them.**
    ///
    /// This test used to hold launch and live equal and demand the ceiling move
    /// one-for-one, which is the double-count: with both at 40 GiB and 4 GiB
    /// pinned, the live figure says 40 GiB is free *now, after* the pinning, and
    /// subtracting again invents a shortage. The pinnable cap is different and
    /// does subtract, because it is derived from `total_ram` — a constant, not a
    /// reading, so nothing has netted the pinned bytes out of it.
    #[test]
    fn what_is_already_pinned_is_subtracted_from_the_launch_reading() {
        // Live is stale (smaller), so the launch reading wins and must pay.
        let none = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            8 * GIB,
            40 * GIB,
            0,
            LOOSE_BUDGET,
        );
        let some = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            8 * GIB,
            40 * GIB,
            4 * GIB,
            LOOSE_BUDGET,
        );
        assert_eq!(
            none.available_less_headroom - some.available_less_headroom,
            4 * GIB,
            "pinned bytes must move a launch-derived ceiling one-for-one"
        );
        // The pinnable cap always subtracts: `total_ram` is a constant.
        assert_eq!(none.pinnable_cap - some.pinnable_cap, 4 * GIB);
    }

    /// **What is page-locked must fit the machine beside everything it owes.**
    ///
    /// The failure this pins, measured on the 16 GB box during a tool
    /// calibration: this tier page-locked 13.5 GiB at launch, the warm KV tier
    /// then grew to 7.1 GiB against a budget that believed it had 14.5, and the
    /// host reached 0.9 GiB free while paging at 1,302 pages/sec — two tiers
    /// sized against snapshots that did not contain each other.
    ///
    /// Warm KV is now a fixed pageable floor outside the partition (the OS pages
    /// it rather than refusing), so the sum that must hold is the pinned one:
    /// this tier, what else is pinned, the weights and the pageable reserve.
    #[test]
    fn the_pinned_expert_tier_fits_the_machine() {
        let total = 31 * GIB + GIB / 2;
        let weights = 2 * GIB + GIB / 2;
        let budget =
            candle::vram::host_ram_budget_from(total, 0, weights, 30, 1024 * 1024 * 1024, None);
        let already = 640 * 1024 * 1024;
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            total,
            // A machine with plenty free at launch — the case that let the old
            // sizing take everything above a fixed 4 GiB guess.
            20 * GIB,
            20 * GIB,
            already,
            budget.expert_pinned_budget_bytes,
        );
        let committed = s.taken_bytes
            + budget.weights_reserved_bytes
            + already
            + candle::vram::PAGEABLE_RESERVE;
        assert!(
            committed <= total,
            "expert tier {:.2} + weights {:.2} + pinned {:.2} + \
             pageable reserve {:.2} = {:.2} GiB on a {:.2} GiB machine",
            gib(s.taken_bytes),
            gib(budget.weights_reserved_bytes),
            gib(already),
            gib(candle::vram::PAGEABLE_RESERVE),
            gib(committed),
            gib(total),
        );
        // And the tier must still be worth having — a partition that fixes the
        // over-commit by starving the cache would pass the line above.
        assert!(
            s.taken_bytes > 8 * GIB,
            "the expert tier got only {:.2} GiB; the split is too tight to be \
             worth the pack-file misses it avoids",
            gib(s.taken_bytes),
        );
    }

    /// Each ceiling binds when it is the lowest, and says so by name — the whole
    /// point of reporting `bound_by` rather than three numbers to compare.
    #[test]
    fn the_lowest_ceiling_binds_and_is_named() {
        // Host budget lowest.
        let s = warm_sizing_from(SLOT, EVICTABLE, 64 * GIB, 60 * GIB, 60 * GIB, 0, 5 * GIB);
        assert_eq!(s.bound_by, CEILING_HOST_BUDGET);
        assert_eq!(s.slots, (5 * GIB / SLOT as u64) as usize);

        // Availability lowest.
        let s = warm_sizing_from(SLOT, EVICTABLE, 64 * GIB, 6 * GIB, 6 * GIB, 0, LOOSE_BUDGET);
        assert_eq!(s.bound_by, CEILING_AVAILABLE);

        // Pinnable region lowest — and it leaves the headroom like the others, so
        // a 16 GiB machine offers what is left after the pageable reserve, less
        // the headroom.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            16 * GIB,
            60 * GIB,
            60 * GIB,
            0,
            LOOSE_BUDGET,
        );
        assert_eq!(s.bound_by, CEILING_PINNABLE);
        assert!(s.slots > 0, "a 16 GiB machine leaves the tier some room");
        assert_eq!(
            s.slots,
            ((16 * GIB - PAGEABLE_RESERVE - WARM_TIER_HEADROOM) / SLOT as u64) as usize
        );

        // Nothing binds: the tier covers every evictable expert.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            512 * GIB,
            400 * GIB,
            400 * GIB,
            0,
            LOOSE_BUDGET,
        );
        assert_eq!(s.bound_by, CEILING_NONE);
        assert_eq!(s.slots, EVICTABLE);
    }

    /// A machine with less free than the headroom asks for takes no tier rather
    /// than underflowing into an enormous one.
    #[test]
    fn a_machine_below_the_headroom_takes_nothing() {
        let s = warm_sizing_from(SLOT, EVICTABLE, 8 * GIB, GIB, GIB, 0, LOOSE_BUDGET);
        assert_eq!(s.available_less_headroom, 0);
        assert_eq!(s.slots, 0);
        assert_eq!(s.taken_bytes, 0);

        // Same for a process already pinning more than the baseline.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            40 * GIB,
            40 * GIB,
            60 * GIB,
            LOOSE_BUDGET,
        );
        assert_eq!(s.slots, 0);
    }

    /// The reported bytes are what the slots actually cost, never the ceiling
    /// they were cut from — a report that rounded up would hide a shortfall.
    #[test]
    fn taken_bytes_follows_the_slot_count() {
        let s = warm_sizing_from(SLOT, EVICTABLE, 31 * GIB, 20 * GIB, 20 * GIB, 0, 27 * GIB);
        assert_eq!(s.taken_bytes, (s.slots * SLOT) as u64);
        assert!(s.taken_bytes <= s.available_less_headroom.min(s.pinnable_cap));
    }

    /// **The process's own heap is not droppable, and the tier is not sized
    /// over it.** WhiteWorld's 5184 trial (2026-11-17): ~50 GiB free at launch,
    /// 18.9 GiB free when the tier was sized, ~30.4 GiB of the process resident
    /// (zend's substrate; 2 GiB of it pinned), and a stated 28 GiB pin. Sized from
    /// the launch figure alone the tier took the whole pin over the free RAM and
    /// the box paged; with the process's own heap out of the launch figure, the
    /// tier fits in what was free.
    #[test]
    fn the_tier_is_not_sized_over_the_processs_own_heap() {
        let pinned = 2 * GIB;
        let launch = 50 * GIB;
        let free_now = 18 * GIB + 9 * GIB / 10;
        let own = 30 * GIB + 2 * GIB / 5;
        let pin = 28 * GIB;
        // Flash-Next's evictable experts: enough that the pin, not the count,
        // is what the old sizing stops at.
        let experts = 24_064;
        let before = warm_sizing_from(SLOT, experts, 64 * GIB, free_now, launch, pinned, pin);
        assert!(
            before.taken_bytes > free_now,
            "the old sizing took {} GiB with {} GiB free",
            before.taken_bytes / GIB,
            free_now / GIB
        );
        let corrected = launch_less_own(launch, own, pinned);
        assert_eq!(corrected, launch - (own - pinned));
        let after = warm_sizing_from(SLOT, experts, 64 * GIB, free_now, corrected, pinned, pin);
        // What the machine had, less everything the process holds, less the
        // headroom: the tier and the process's own heap fit in it together.
        assert!(
            after.taken_bytes + WARM_TIER_HEADROOM <= launch - own,
            "the tier takes {} GiB beside {} GiB of the process's own",
            after.taken_bytes / GIB,
            own / GIB
        );
        assert!(after.taken_bytes < free_now);
        assert_eq!(after.bound_by, CEILING_AVAILABLE);
    }

    /// A process whose only resident memory is what it pinned changes nothing:
    /// the launch figure stands, as the mid-load trough case needs.
    #[test]
    fn a_process_holding_only_pinned_memory_keeps_the_launch_figure() {
        assert_eq!(launch_less_own(15 * GIB, GIB, GIB), 15 * GIB);
        assert_eq!(launch_less_own(15 * GIB, GIB / 2, GIB), 15 * GIB);
        assert_eq!(launch_less_own(4 * GIB, 9 * GIB, 0), 0);
    }
}
