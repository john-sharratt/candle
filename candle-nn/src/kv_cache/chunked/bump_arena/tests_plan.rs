//! Tests for how a forward's tier plan is merged, kept, and laid out.
//!
//! These tests cover the cases where a walk or a later forward records a plan
//! while a tier stands. A covering plan takes the larger of each phase, a stale
//! plan with no tier standing is not merged into a walk, and a standing tier
//! that already covers a walk is kept as it is. A merge the tier cannot hold
//! re-places only the walk's own width. The weight-floor arithmetic and the
//! launch-scratch budget are checked here too, along with the refusals for a
//! plan that outgrows spans a live generation holds.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{cover_target, covering_plan, tier_footprint, Cover};

    use crate::kv_cache::chunked::wave_plan::WAVE_SPAN_ALIGN;

    /// A walk wider than the forward raises the phases it outgrows and keeps the
    /// ones the forward reserved for something else.
    #[test]
    fn a_covering_plan_takes_the_larger_of_each_phase() {
        let forward = [64_256, 30_720, 4_096];
        let walk = [128_512, 20_480, 8_192];
        assert_eq!(covering_plan(forward, walk), [128_512, 30_720, 8_192]);
        assert_eq!(covering_plan(walk, forward), [128_512, 30_720, 8_192]);
        assert_eq!(covering_plan(forward, forward), forward);
    }

    /// With no tier standing, a recorded plan is a past forward's and buys
    /// nothing: the walk places only what it asked for. A prefill's 4 GB plan
    /// merged into a later walk's few MiB re-bought the prefill tier into live
    /// KV on Flash-Next C5×8.
    #[test]
    fn with_no_tier_standing_a_stale_plan_is_not_merged() {
        let prefill = [4_009_754_624, 30_720, 4_096];
        let walk = [128_512, 20_480, 8_192];
        assert_eq!(cover_target(Some(prefill), None, walk), Cover::Place(walk));
        assert_eq!(cover_target(None, None, walk), Cover::Place(walk));
    }

    /// A standing tier that already covers the walk places nothing.
    #[test]
    fn a_standing_tier_that_covers_the_walk_is_kept() {
        let forward = [64_256, 30_720, 4_096];
        let narrower = [32_000, 20_480, 4_096];
        let bytes = tier_footprint(forward);
        assert_eq!(
            cover_target(Some(forward), Some(bytes), narrower),
            Cover::Keep
        );
        assert_eq!(
            cover_target(Some(forward), Some(bytes), forward),
            Cover::Keep
        );
    }

    /// A walk that outgrows one phase, but whose merged plan still fits the bytes
    /// the standing tier holds, re-lays the spans inside that tier: raised where
    /// the walk outgrows it, kept where the forward reserved more, nothing bought.
    #[test]
    fn a_merged_plan_that_fits_the_standing_tier_is_laid_out_in_place() {
        let forward = [64_256, 30_720, 4_096];
        let wider = [128_512, 20_480, 8_192];
        assert_eq!(
            cover_target(Some(forward), Some(200_000), wider),
            Cover::Relay([128_512, 30_720, 8_192])
        );
    }

    /// **A merged plan the standing tier cannot hold re-places the walk's own
    /// width, not the merge.** Re-placing releases the standing tier, so merging
    /// keeps none of its room and re-buys a previous forward's whole footprint:
    /// in zend a draft walk after an 8K-row prefill asked for 8,992,587,776 B —
    /// the prefill's tier plus 86 KB — against a gap of 8,992,501,760 B, and six
    /// files failed their ingest on it.
    #[test]
    fn a_merged_plan_that_outgrows_the_standing_tier_places_only_the_walk() {
        let prefill = [2_000_000_000, 6_000_000_000, 992_501_760];
        let bytes = tier_footprint(prefill);
        let walk = [6_000_086_016, 20_480, 8_192];
        assert_eq!(
            cover_target(Some(prefill), Some(bytes), walk),
            Cover::Place(walk)
        );
    }

    /// A plan is recorded even when no tier can be placed for it, so the
    /// recorded plan and the standing tier can disagree — which is the state the
    /// layout must not walk off the end of.
    ///
    /// This is the arithmetic of that disagreement, isolated from the domain
    /// machinery: a base chosen for a narrow purchase, plus a later, wider
    /// plan, reaches past the weight floor. The layout's job is to notice that
    /// the SUM crosses the floor — not that the plan disagrees with the bytes
    /// the tier bought, which are derived from the same array and always agree.
    #[test]
    fn a_wider_plan_over_a_narrower_tier_reaches_past_the_weight_floor() {
        const FLOOR: u64 = 0x1_0000_0000;
        // The tier was bought for this, and placed so that it just fits.
        let bought: usize = 4 * WAVE_SPAN_ALIGN;
        let base = FLOOR - bought as u64;

        // The same plan the purchase was made from fits exactly, by
        // construction — which is why comparing the two can never catch this.
        let narrow = [WAVE_SPAN_ALIGN, 2 * WAVE_SPAN_ALIGN, WAVE_SPAN_ALIGN];
        assert_eq!(narrow.iter().sum::<usize>(), bought);
        assert!(base + narrow.iter().sum::<usize>() as u64 <= FLOOR);

        // A later, wider forward records its plan while this tier still stands.
        let wide = [WAVE_SPAN_ALIGN, 6 * WAVE_SPAN_ALIGN, WAVE_SPAN_ALIGN];
        let need: usize = wide.iter().sum();
        assert!(need > bought, "the wider plan is what makes this a hazard");
        assert!(
            base + need as u64 > FLOOR,
            "laying the wider plan from the narrow tier's base crosses the floor \
             by {} B — expert-slot ground",
            base + need as u64 - FLOOR
        );

        // And the span that lands past it is the LAST one, which is the FFN's:
        // walking the widths back-to-back, phases 0 and 1 stay below.
        let mut at = base;
        let mut first_past = None;
        for (i, w) in wide.iter().enumerate() {
            at += *w as u64;
            if at > FLOOR && first_past.is_none() {
                first_past = Some(i);
            }
        }
        assert_eq!(
            first_past,
            Some(1),
            "phase 1 is `LayerPhase::Ffn` — the span the corruption was always found in"
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(super) mod wave_tests {
    use super::{
        begin_forward, begin_wave, cover_wave_transient, enter_arena_window, plan_wave_transient,
        tier_footprint, LayerPhase,
    };

    use candle::Result;

    use std::sync::Arc;

    /// The wave domain is process-global and `cargo test` runs tests in
    /// parallel, so these take turns. It is the *crate-wide* lock rather than a
    /// local one: the KV selection tests build backings on the same device, and
    /// a lock that only excludes this module would not exclude them.
    use super::super::super::gpu_test_lock::gpu_serial as serial;

    use super::super::tests_window::wave_tests::{lease, stream};

    /// **The layer phases share ground, so neither opens while the other is
    /// live.** Each refuses while the other's guard stands, and opens once it
    /// has dropped — a layer's two phases in sequence, as the forward runs them.
    #[test]
    fn a_layer_phase_refuses_while_the_other_is_live() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let attention = begin_wave(&s, LayerPhase::Attention)?;
        let refused = begin_wave(&s, LayerPhase::Ffn)
            .err()
            .expect("the FFN span opened over a live attention span");
        assert!(
            refused.to_string().contains("shares its ground"),
            "{refused}"
        );
        drop(attention);
        let ffn = begin_wave(&s, LayerPhase::Ffn)?;
        assert!(
            begin_wave(&s, LayerPhase::Attention).is_err(),
            "the attention span opened over a live FFN span"
        );
        drop(ffn);
        drop(begin_wave(&s, LayerPhase::Attention)?);
        Ok(())
    }

    /// **A forward that has placed but not yet opened keeps its plan through an
    /// arena window.** Another thread's arena between a step's placement and its
    /// forward span handed the tier back with the plan, and the span opened at
    /// the fallback's 16 MiB under a 74 MB residual (WhiteWorld, a965e3a07).
    #[test]
    fn a_placed_forward_keeps_its_plan_through_an_arena_window() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let forward_wide = [1 << 20, 1 << 20, 64 << 20];
        let forward = {
            plan_wave_transient(&s, forward_wide)?;
            drop(enter_arena_window(lease(), "a test tenant")?);
            let _open = begin_forward(&s);
            begin_wave(&s, LayerPhase::Forward)?
        };
        let (_, laid, _) = super::wave_domain_layout(&s).expect("a plan is recorded");
        assert!(
            laid[2] >= forward_wide[2],
            "the forward span was laid out at {} B for a plan of {} B",
            laid[2],
            forward_wide[2]
        );
        drop(forward);
        Ok(())
    }

    /// **And when the window left no room for it, the forward is refused** —
    /// with the retryable tier refusal, never a layout of the fallback in its
    /// place. The pending plan is what the guard places, so a partition too
    /// small for it is a refusal of that width, not a silent 16 MiB span.
    #[test]
    fn a_placed_forward_the_window_left_no_room_for_is_refused() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let ordinal = s.context().ordinal();
        // The lowest floor the pool allows: the arena frontier, or the elastic
        // reserve above the region base where that stands higher.
        drop(begin_wave(&s, LayerPhase::Attention)?);
        let layout = super::super::super::region_pool::span_layout(ordinal).expect("a span");
        let frontier = super::super::super::region_pool::region_stats(ordinal)
            .expect("a pool")
            .live_watermark;
        let frontier_end =
            layout.region_base + (frontier * super::super::super::region_pool::REGION_BYTES) as u64;
        let floor = frontier_end
            .max(layout.region_base + super::super::super::weight_zone::MIN_ELASTIC_RESERVE as u64);
        // A forward wider than the ground that floor leaves, placed while the
        // floor still stands where it was.
        let room = (floor - frontier_end) as usize;
        plan_wave_transient(&s, [1 << 20, 1 << 20, room + (64 << 20)])?;
        drop(enter_arena_window(lease(), "a test tenant")?);
        // The window's claim took the room: the floor now stands where no tier
        // of that width fits below it.
        super::super::super::region_pool::set_weight_floor(&s, floor)?;
        let opened = {
            let _open = begin_forward(&s);
            begin_wave(&s, LayerPhase::Forward).map(|g| {
                let laid = super::wave_domain_layout(&s).map(|(_, l, _)| l[2]);
                drop(g);
                laid
            })
        };
        // Restored before anything is asserted: the floor is process-global.
        super::super::super::region_pool::set_weight_floor(&s, layout.weight_floor)?;
        match opened {
            Ok(laid) => panic!("the forward opened on a {laid:?} B span with no room for its plan"),
            Err(e) => assert!(
                super::super::super::backing::is_tier_refusal(&e),
                "refused with the retryable tier refusal: {e}"
            ),
        }
        Ok(())
    }

    /// **A plan that outgrew the spans a live generation holds is refused, not
    /// handed the old spans.** A forward whose returned span is still held keeps
    /// the layout its plan made; a wider plan recorded behind it gets no layout
    /// of its own, and opening a phase on the old spans would run it on arenas
    /// sized for the earlier forward — a 320 MiB residual asked of a 16 MiB span
    /// on BlackWorld (c673b63c9). Refused with the retryable marker instead, and
    /// a plan the standing spans hold still opens.
    #[test]
    fn a_plan_outgrowing_the_spans_a_live_generation_holds_is_refused() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let small = [1 << 20, 1 << 20, 1 << 20];
        let forward = {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, small)?;
            // The earlier forward's span, still held — as a returned head span is.
            begin_wave(&s, LayerPhase::Forward)?
        };
        // A wider forward records its plan behind it: placement waits on the
        // live generation, so the spans stand as the small plan laid them out.
        let wide = small.map(|b| b + (32 << 20));
        {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, wide)?;
        }
        let err = match begin_wave(&s, LayerPhase::Attention) {
            Ok(_) => panic!("a phase opened on spans laid out for a smaller plan"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains(super::SPANS_OUTGROWN_MARKER),
            "refused with the retryable marker: {err}"
        );
        assert!(
            super::super::super::backing::is_arena_exhausted(&err),
            "and recognised as retryable"
        );
        // A plan the standing spans hold opens as it always did.
        {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, small)?;
        }
        drop(begin_wave(&s, LayerPhase::Attention)?);
        drop(forward);
        Ok(())
    }

    /// **A walk keeps its own width when an arena it creates hands its tier
    /// back**, and only while it holds its [`super::Covering`].
    ///
    /// A draft walk creates arenas mid-walk, inside its recording. With the
    /// plan dropped, its next phase re-placed the worst-case fallback there,
    /// where the weight side refuses to concede: Flash-Next's BF16 ×4 gate row
    /// failed on a 912 MiB tier 19 regions short.
    #[test]
    fn a_walk_keeps_its_width_when_an_arena_hands_its_tier_back() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let ordinal = s.context().ordinal();
        let tier = |ordinal| {
            super::super::super::region_pool::region_stats(ordinal)
                .map(|r| r.transient_bytes)
                .unwrap_or(0)
        };
        let fallback = tier_footprint(super::fallback_plan());
        let wide = super::fallback_plan().map(|b| b + (16 << 20));
        {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, wide)?;
            drop(begin_wave(&s, LayerPhase::Forward)?);
        }
        // The walk fits inside the forward's standing tier, so the cover keeps
        // the forward's plan and records the walk's width beside it.
        let walk = [2 << 20, 2 << 20, 2 << 20];
        let covering = cover_wave_transient(&s, walk)?;
        drop(enter_arena_window(lease(), "a test tenant")?);
        assert_eq!(tier(ordinal), 0, "the window handed the tier back");
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        let placed = tier(ordinal);
        assert!(
            placed > 0 && placed < fallback,
            "the walk's guard placed {placed} B: its own width, not the {fallback} B fallback"
        );
        drop(guard);
        drop(covering);

        // The walk is over: a tier handed back after it leaves no plan.
        {
            let _open = begin_forward(&s);
            plan_wave_transient(&s, wide)?;
            drop(begin_wave(&s, LayerPhase::Forward)?);
        }
        drop(enter_arena_window(lease(), "a test tenant")?);
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        let placed = tier(ordinal);
        assert!(
            placed >= fallback,
            "after the walk the guard placed {placed} B: the walk's width outlived it"
        );
        drop(guard);
        Ok(())
    }

    /// **A forward's launchers carve only the scratch its plan priced**, and the
    /// rest of the span stays the forward's own. With no budget the carve is
    /// refused outright — the first forwards after a start price none, and
    /// carving then took the MTP head's priced output and failed zend's load.
    ///
    /// On an ordinal no device has, outside any wave domain, so no forward of a
    /// test running beside this one can carve from the span.
    #[test]
    fn launch_scratch_stops_at_what_the_plan_priced() -> Result<()> {
        const NO_DEVICE: usize = 15;
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let arena = super::BumpArena::detached(&s, "test forward span");
        arena.rebase(0x1_0000_0000, 64 << 10)?;
        let generation = arena.generation(super::NOT_A_WAVE, super::FORWARD_ARENA)?;

        super::open_forward_scratch(NO_DEVICE, Arc::clone(&arena.inner), 0);
        assert!(
            super::carve_forward_scratch(NO_DEVICE, 256).is_none(),
            "a span priced no scratch carves none"
        );

        super::open_forward_scratch(NO_DEVICE, Arc::clone(&arena.inner), 4096);
        assert!(super::carve_forward_scratch(NO_DEVICE, 3000).is_some());
        // 3,000 carved; the next starts at 3,072, so 1,100 more is 4,172.
        assert!(
            super::carve_forward_scratch(NO_DEVICE, 1100).is_none(),
            "past the budget, though the span has room"
        );
        assert!(super::carve_forward_scratch(NO_DEVICE, 1000).is_some());
        super::close_forward_scratch(NO_DEVICE);
        assert!(
            super::carve_forward_scratch(NO_DEVICE, 256).is_none(),
            "a closed scope carves nothing"
        );

        // The span past the budget is still the forward's.
        let rest = generation.alloc(56 << 10, 256);
        assert!(rest.is_ok(), "the forward's own buffer still fits");
        drop(rest);
        drop(generation);
        Ok(())
    }
}
