//! Checks that the tier admission prices is the tier the placement actually
//! buys.
//!
//! The launch scratch widens the forward span and nothing else, and every entry
//! is then aligned. `tier_bytes` includes that scratch because the placement
//! carries it: a plan one byte past a region is priced one region dearer. The
//! tier is at least the largest phase and less than the sum wherever both layer
//! phases carve, and it is priced in whole regions because placement rounds to a
//! region.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{moe, w};

    /// **The launchers' scratch widens the forward span, and only it** — then
    /// every entry is aligned. Raw figures.
    #[test]
    fn the_launch_plan_carries_the_scratch_on_the_forward_span() {
        assert_eq!(launch_plan([300, 200, 50], 1000), [512, 256, 1280]);
        assert_eq!(launch_plan([300, 200, 50], 0), [512, 256, 256]);
        assert_eq!(launch_plan([0, 0, 0], 0), [0, 0, 0]);
    }

    /// **The tier admission prices is the tier the forward places**, launch
    /// scratch included. A 48 MiB plan whose forward span the scratch widens by
    /// one byte past its region is one region dearer — the region WhiteWorld's
    /// creep window was refused by when its price left the scratch out.
    #[test]
    fn tier_bytes_prices_the_launch_scratch_the_placement_carries() {
        let p = WavePlan::new(moe());
        for rows in [1usize, 64, 4096] {
            let phases = [
                p.phase_bytes(LayerPhase::Attention, w(rows)),
                p.phase_bytes(LayerPhase::Ffn, w(rows)),
                p.phase_bytes(LayerPhase::Forward, w(rows)),
            ];
            for scratch in [0usize, 4096, 3 * TARGET_ARENA_BYTES + 1] {
                let placed = tier_footprint(launch_plan(phases, scratch));
                assert_eq!(
                    p.tier_bytes(w(rows), scratch),
                    placed.div_ceil(TARGET_ARENA_BYTES) * TARGET_ARENA_BYTES,
                    "rows {rows}, scratch {scratch}"
                );
            }
            assert!(
                p.tier_bytes(w(rows), 3 * TARGET_ARENA_BYTES + 1)
                    >= p.tier_bytes(w(rows), 0) + 3 * TARGET_ARENA_BYTES,
                "rows {rows}: the scratch must be paid for"
            );
        }
    }

    /// **The tier is the footprint of the phases — no less than the largest of
    /// them, and less than their sum wherever both layer phases carve.** The
    /// lower bound is the defect the tier bound exists to fix: sizing waves with
    /// `wave_bytes` alone let them ask for a 6.5 GiB tier against a 912 MiB
    /// guarantee and be refused 62 times in one run. The upper is what sharing
    /// the layer phases' ground gives back.
    #[test]
    fn tier_bytes_is_the_footprint_of_the_phases() {
        let p = WavePlan::new(moe());
        for rows in [1usize, 64, 512, 4096] {
            let phases = [
                p.phase_bytes(LayerPhase::Attention, w(rows)),
                p.phase_bytes(LayerPhase::Ffn, w(rows)),
                p.phase_bytes(LayerPhase::Forward, w(rows)),
            ];
            let tier = p.tier_bytes(w(rows), 0);
            assert_eq!(
                tier,
                tier_footprint(launch_plan(phases, 0)).div_ceil(TARGET_ARENA_BYTES)
                    * TARGET_ARENA_BYTES,
                "rows {rows}"
            );
            assert!(
                tier >= p.wave_bytes(w(rows)),
                "rows {rows}: tier {tier} under the largest single phase {}",
                p.wave_bytes(w(rows)),
            );
            let sum: usize = phases.iter().sum();
            assert!(
                tier_footprint(phases) < sum,
                "rows {rows}: both layer phases carve, so sharing must cost less than {sum}"
            );
        }
    }

    /// **The tier is priced in whole regions, as it is placed.** The placement
    /// rounds its length up to a region; a price that did not read a gap as
    /// enough when the placement then asked for one region more.
    #[test]
    fn tier_bytes_is_a_whole_number_of_regions() {
        let p = WavePlan::new(moe());
        for rows in [1usize, 64, 128, 512, 4096] {
            assert_eq!(
                p.tier_bytes(w(rows), 4096) % TARGET_ARENA_BYTES,
                0,
                "rows {rows}: {} is not region-aligned",
                p.tier_bytes(w(rows), 4096),
            );
        }
    }
}
