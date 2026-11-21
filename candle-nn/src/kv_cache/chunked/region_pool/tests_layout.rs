//! Tests for the span's arithmetic: the layout at rest, the dense block, and headroom.
//!
//! These tests check that the span is `usable` less the cushion and nothing
//! else, since the weights are already resident and netted out of it. They check
//! that the dense block costs regions alone, rounding up to under one region.
//! They check that the layout partitions the span with no overlap, that a dense
//! block as wide as the span leaves no regions instead of wrapping, and that load
//! headroom is never conceded below the span minimum or without a governor.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{layout_span, Layout, PERSIST_SPAN_BYTES, REGION_BYTES};

    /// The layout of a span with the fixed block every span has.
    pub(in super::super) fn rest_layout(
        base: u64,
        span: usize,
        dense: usize,
        floor: u64,
    ) -> Layout {
        layout_span(base, PERSIST_SPAN_BYTES, span, dense, floor)
    }

    /// **The span is `usable` less the pool cushion — and nothing else.**
    ///
    /// Specifically it is *not* also less the dense weights. They are resident
    /// by the time this runs, so `usable()` has already netted them out of `C`;
    /// subtracting `class_reserved(Weights)` here as well would book the same
    /// bytes twice. The load order is what makes that mistake look like
    /// prudence, so the identity is pinned rather than left to a comment.
    ///
    /// A pure-arithmetic test, so it holds on a machine with no GPU.
    #[test]
    fn the_span_is_usable_less_the_cushion_and_nothing_else() {
        let mib = 1024 * 1024;
        let cushion = 512 * mib;
        let capacity = 14_592 * mib;
        for dense_mib in [0usize, 1024, 4096] {
            let dense = dense_mib * mib;
            // What `usable()` reports once `dense` bytes are resident: the drop
            // in headroom since C was measured is exactly the weights.
            let usable = capacity - dense;
            assert_eq!(
                super::span_from(usable, cushion),
                capacity - dense - cushion,
                "the weights must be subtracted once, not twice, at {dense_mib} MiB"
            );
        }
    }

    /// The dense block takes its bytes from the **regions**, and from nothing
    /// else in the layout.
    ///
    /// This is the whole arithmetic of putting the model weights inside the
    /// reservation: the persistence block does not move, the weight floor does
    /// not move, and the region count absorbs the difference — the same
    /// property a short span already has.
    #[test]
    fn the_dense_block_costs_regions_and_moves_nothing_else() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 4096 * mib;
        let span_end = base + span as u64;
        let empty = rest_layout(base, span, 0, span_end);
        // 1,921 MiB — the measured dense footprint of Qwen3.6-35B.
        let dense = 1921 * mib;
        let loaded = rest_layout(base, span, dense, span_end);

        assert_eq!(
            loaded.dense_base, empty.dense_base,
            "the dense block starts at the persistence block's right edge and \
             never moves",
        );
        assert_eq!(
            loaded.weight_floor, empty.weight_floor,
            "the expert zone is placed from the right edge; the dense block \
             cannot push it",
        );
        assert_eq!(
            loaded.region_base - empty.region_base,
            dense.next_multiple_of(REGION_BYTES) as u64,
            "regions start past the dense block, rounded up to a whole region",
        );
        assert_eq!(
            empty.total - loaded.total,
            dense.div_ceil(REGION_BYTES),
            "and the cost lands entirely on the region count",
        );
    }

    /// Rounding the block up to a region wastes under one region, never more.
    ///
    /// The alignment is what lets every region address stay
    /// `region_base + i · REGION_BYTES` with no second rule to keep in step, and
    /// it is only worth that if the waste is bounded — so the bound is asserted
    /// rather than assumed.
    #[test]
    fn dense_rounding_wastes_less_than_one_region() {
        let base = 0x1_0000_0000u64;
        let span = 4096 * 1024 * 1024;
        let span_end = base + span as u64;
        for bytes in [1usize, 255, 256, REGION_BYTES - 1, REGION_BYTES + 1] {
            let l = rest_layout(base, span, bytes, span_end);
            let held = (l.region_base - l.dense_base) as usize;
            assert!(held >= bytes, "the block must cover what was asked for");
            assert!(
                held - bytes < REGION_BYTES,
                "{bytes} B rounded to {held} B — more than a region wasted",
            );
        }
    }

    /// The dense block and the region grid share one boundary and never overlap.
    ///
    /// `region_base` is the dense block's right edge and the first region's
    /// address at the same time, so it is the single number keeping a KV chunk
    /// out of a weight. Asserted as arithmetic because the runtime check lives
    /// inside a `with_pool` closure that needs a device.
    #[test]
    fn the_dense_block_and_the_region_grid_meet_at_region_base() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 4096 * mib;
        let span_end = base + span as u64;
        let dense = 1921 * mib;
        let l = rest_layout(base, span, dense, span_end);

        assert!(
            l.dense_base < l.region_base,
            "the block must occupy ground below the regions",
        );
        assert_eq!(
            (l.region_base - l.dense_base) as usize,
            dense.next_multiple_of(REGION_BYTES),
            "the dense block ends exactly where the first region begins",
        );
        // Every region is above the block, so no region address can land inside
        // a weight.
        for idx in [0usize, 1, l.total.saturating_sub(1)] {
            let addr = l.region_base + (idx * REGION_BYTES) as u64;
            assert!(addr >= l.region_base, "region {idx} is above the weights");
            assert!(addr < span_end, "region {idx} stays inside the span");
        }
    }

    /// Load headroom is conceded from what is spare, never from the span's
    /// floor.
    ///
    /// **This is a regression test for a real failure.** `span_target` falls
    /// back to a fixed `TEST_SPAN_BYTES` when no VRAM governor is installed —
    /// every unit test, and any CPU-side path — and the first version of the
    /// headroom subtracted the checkpoint's repack peak from that constant
    /// unconditionally. Qwen3-8B's `output.weight` peak is ~2.4 GiB, which took
    /// the span to 640 MiB against a 976 MiB minimum and refused the
    /// reservation outright: a model that loaded before would not load at all.
    ///
    /// The headroom exists to keep a *possible* pool OOM from happening during
    /// load. Trading that for a *certain* refusal at startup is never right, so
    /// the concession stops at [`super::span_minimum_bytes`] — and is skipped entirely
    /// without a governor, since the fixed test constant is not a measurement
    /// of a card and has no capacity to leave room in. Clamping alone was not
    /// enough: it moved the failure from "reservation refused" to a span of
    /// exactly the floor with `0 regions`, which fails one step later.
    #[test]
    fn headroom_is_conceded_from_spare_ground_only() {
        let floor = super::span_minimum_bytes(super::PERSIST_SPAN_BYTES);
        // Exactly the shape that failed: a target barely above the floor, and a
        // headroom several times larger than the whole span.
        let target = floor + 64 * 1024 * 1024;
        let greedy = 4 * target;
        let conceded = greedy.min(target.saturating_sub(floor));
        assert_eq!(
            target - conceded,
            floor,
            "a headroom larger than the span must leave exactly the floor, not underflow it",
        );

        // With room to spare, the whole request is honoured.
        let roomy = floor + 8 * 1024 * 1024 * 1024;
        let want = 2 * 1024 * 1024 * 1024;
        assert_eq!(
            roomy - want.min(roomy.saturating_sub(floor)),
            roomy - want,
            "when the span can afford the headroom it concedes all of it",
        );

        // And a span already at the floor concedes nothing.
        assert_eq!(floor.saturating_sub(floor), 0);
    }

    /// A dense block that would swallow the span leaves no regions, and does not
    /// wrap into a colossal count.
    ///
    /// `claim_dense` refuses this case before it can be stored, but the refusal
    /// path computes the layout it is refusing, so the arithmetic has to survive
    /// the input rather than merely be protected from it.
    #[test]
    fn a_dense_block_wider_than_the_span_yields_no_regions() {
        let base = 0x1_0000_0000u64;
        let span = 512 * 1024 * 1024;
        let span_end = base + span as u64;
        let l = rest_layout(base, span, span * 4, span_end);
        assert_eq!(l.total, 0);
        assert!(
            l.region_base <= span_end,
            "region base stays inside the span"
        );
    }

    /// A span too small for the persist block yields no regions rather than
    /// wrapping into a colossal one. (`span_from` itself is only
    /// `usable − cushion`; a span that cannot hold the fixed block is refused
    /// outright in `create`.)
    ///
    /// Note what is *not* subtracted: the wave tier. At rest it occupies
    /// nothing, so its ground belongs to the KV side until a forward places it.
    #[test]
    fn a_span_smaller_than_the_persist_block_yields_no_regions() {
        assert_eq!(super::span_from(0, 1024), 0, "a cushion larger than usable");
        let base = 0x1_0000_0000u64;
        let persist = super::PERSIST_SPAN_BYTES;
        let l = rest_layout(base, persist, 0, base + persist as u64);
        assert_eq!(l.total, 0, "the persist block leaves nothing for KV");
        assert_eq!(l.slack, 0);
        // A span below even that stays sane rather than wrapping.
        let l = rest_layout(base, persist / 2, 0, base + (persist / 2) as u64);
        assert_eq!(l.total, 0);
        assert_eq!(l.slack, 0);
        // And the tier's worth of ground *is* the KV side's while nothing runs.
        let with_tier = persist + super::MAX_WAVE_TRANSIENT_BYTES;
        let l = rest_layout(base, with_tier, 0, base + with_tier as u64);
        assert_eq!(
            l.total,
            super::MAX_WAVE_TRANSIENT_BYTES / REGION_BYTES,
            "at rest the tier's ground belongs to the KV side"
        );
    }

    /// The layout is positional from the right: the persist block is at the
    /// left, the transient tier is measured back from the weight floor, and the
    /// regions are what is left in the middle. Nothing overlaps and nothing
    /// escapes the span.
    #[test]
    fn the_layout_partitions_the_span_without_overlap() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        for span_mib in [3072usize, 8192, 14336] {
            for weight_mib in [0usize, 1024, 4096] {
                let span = span_mib * mib;
                let span_end = base + span as u64;
                let floor = span_end - (weight_mib * mib) as u64;
                let l = rest_layout(base, span, 0, floor);

                // Only the persist block is fixed; the regions start straight
                // after it and run up to the weight floor. The wave tier is not
                // in the layout at all — it exists only while a forward runs.
                assert_eq!(l.region_base, base + super::PERSIST_SPAN_BYTES as u64);
                let region_end = l.region_base + (l.total * REGION_BYTES) as u64;
                assert!(
                    region_end <= l.weight_floor,
                    "regions ran into the weight side at {span_mib}/{weight_mib}"
                );
                assert!(l.weight_floor <= span_end, "weight side past the span end");
                assert_eq!(
                    region_end + l.slack as u64,
                    l.weight_floor,
                    "slack must account for exactly the rounding remainder"
                );
            }
        }
    }

    /// A weight floor beyond either edge is clamped rather than wrapped. The
    /// floor arrives from arithmetic done a crate away, and a wrapped
    /// subtraction here would produce an address that looks plausible.
    #[test]
    fn the_layout_clamps_a_floor_outside_the_span() {
        let base = 0x1_0000_0000u64;
        let span = 4096 * 1024 * 1024;
        let past_end = rest_layout(base, span, 0, base + span as u64 + 4096);
        assert_eq!(past_end, rest_layout(base, span, 0, base + span as u64));
        let before_start = rest_layout(base, span, 0, 0);
        assert_eq!(
            before_start.total, 0,
            "a floor left of the regions leaves none"
        );
    }

    /// **The transient tier is not in the resting layout at all.**
    ///
    /// Between forwards it occupies nothing, so the regions run from the persist
    /// block straight to the weight floor. That is what makes the tier the one
    /// block in the span that can change size: at the moment its extent changes
    /// it holds nothing, so a resize leaves no hole and moves no data — and it is
    /// why it can sit *between* the arenas and the weights, absorbing both sides'
    /// movement, which nothing at a fixed address could do.
    #[test]
    fn the_resting_layout_has_no_transient_tier() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 12288 * mib;
        let span_end = base + span as u64;
        let expected = base + super::PERSIST_SPAN_BYTES as u64;
        for weight_mib in [0usize, 512, 2048, 6144, 9216] {
            let l = rest_layout(base, span, 0, span_end - (weight_mib * mib) as u64);
            assert_eq!(l.region_base, expected, "regions start after persist");
            // Every byte between the persist block and the weight floor is
            // available to regions when no forward is running.
            let floor = l.weight_floor;
            assert_eq!(
                l.total * REGION_BYTES + l.slack,
                (floor - expected) as usize,
                "resting layout left a gap at weight={weight_mib} MiB"
            );
        }
    }

    /// Moving the boundary changes exactly one thing: how many regions exist.
    /// Every megabyte the weight side gives up becomes KV regions, one for one
    /// (modulo the rounding the slack accounts for).
    #[test]
    fn giving_ground_becomes_regions_one_for_one() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 12288 * mib;
        let span_end = base + span as u64;
        let at =
            |weight_mib: usize| rest_layout(base, span, 0, span_end - (weight_mib * mib) as u64);

        let wide = at(6144);
        let narrow = at(4096);
        assert_eq!(
            narrow.total - wide.total,
            (2048 * mib) / REGION_BYTES,
            "2048 MiB of weights is exactly that many regions of KV"
        );
    }

    /// The whole span is accounted for: persist + regions + slack + weights is
    /// exactly the span, at every boundary position. (No transient term — at
    /// rest it has none.)
    #[test]
    fn every_byte_of_the_span_is_accounted_for() {
        let base = 0x2_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 9216 * mib;
        let span_end = base + span as u64;
        for weight_mib in [0usize, 512, 2048, 6144] {
            let floor = span_end - (weight_mib * mib) as u64;
            let l = rest_layout(base, span, 0, floor);
            let weights = (span_end - l.weight_floor) as usize;
            let total = super::PERSIST_SPAN_BYTES + l.total * REGION_BYTES + l.slack + weights;
            assert_eq!(total, span, "span unaccounted at weight={weight_mib} MiB");
        }
    }
}
