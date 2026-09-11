//! The weight zone: how much residency the engine has, and how much it defends.
//!
//! Measurements only. Admission moved to [`super::admit`], which reads these to
//! decide what one more sequence would cost the resident experts; nothing here
//! decides anything.
//!
//! The zone shares one elastic span with the KV side, so every figure below is
//! derived from the span identity (`total × region + weight = span`) rather than
//! read off the boundary. The boundary **lags** — it moves only between
//! forwards, so a reading taken mid-wave is wherever the last wave left it, not
//! where the weight side could now stand.

use std::sync::atomic::AtomicU64;

use candle_nn::kv_cache::{MIN_ELASTIC_RESERVE, REGION_BYTES};

/// The weight zone's permitted extent, in bytes.
pub(super) fn weight_zone_bytes() -> Option<u64> {
    candle_nn::kv_cache::region_stats(0).map(|r| r.weight_bytes as u64)
}

/// The residency the weight side could hold **right now**: the span less the
/// regions live and less the floor's reserve — [`achievable_weight_bytes`]
/// evaluated on the live count instead of at idle.
///
/// **This, not the zone's extent, is what a hold compares against.** The extent
/// moves only when the boundary moves, and a store or a KV chunk claimed from
/// the free list does not move it: the zone read exactly the same after
/// twenty-five stores had been claimed as before the first, and every one of
/// them passed a hold watching a number that had not changed. The boundary
/// caught up later, when the tier had to buy ground, and the zone fell 2 GiB
/// under the mark in one step. Counting live regions charges each claim the
/// instant it is made.
pub(super) fn effective_weight_zone_bytes() -> Option<u64> {
    let r = candle_nn::kv_cache::region_stats(0)?;
    Some(achievable_weight_bytes(
        r.total,
        r.live,
        r.weight_bytes,
        REGION_BYTES,
        MIN_ELASTIC_RESERVE,
    ))
}

/// The residency the weight side could reach if nothing but what is resident
/// right now stood in its way — bytes.
static ACHIEVABLE_WEIGHT: AtomicU64 = AtomicU64::new(0);

/// What the weight zone could grow to with `live` regions standing: the span
/// less the live KV and less the ground the floor may never cross.
///
/// Computed from the span identity rather than read off the zone, because the
/// zone lags. Measured, the first fill of a run read 5,951 MiB where load held
/// 10,398 — a mark taken from that reading sat below anything the run would
/// reach.
pub(super) fn achievable_weight_bytes(
    total_regions: usize,
    live_regions: usize,
    weight_bytes: usize,
    region_bytes: usize,
    reserve_bytes: usize,
) -> u64 {
    let span = total_regions
        .saturating_mul(region_bytes)
        .saturating_add(weight_bytes);
    span.saturating_sub(live_regions.saturating_mul(region_bytes))
        .saturating_sub(reserve_bytes) as u64
}

/// Re-measure the achievable residency from what is resident **now**.
///
/// **Called when the engine is idle, and at its entry.** With nothing in flight,
/// every region live is permanent — the system prompt, the tool catalog, the
/// substrate's resident corpus — and none of it is the wave's to give back, so
/// the residency the weight side can actually reach is the span less exactly
/// that. Measuring at load instead defended a residency the tool catalog then
/// made unreachable: the mark sat 2% above a zone that could never climb.
///
/// A reading, not a controller: one identity evaluated at the moment it is
/// exact.
pub(super) fn reseed_achievable_weight() {
    use std::sync::atomic::Ordering;
    let Some(r) = candle_nn::kv_cache::region_stats(0) else {
        return;
    };
    let achievable = achievable_weight_bytes(
        r.total,
        r.live,
        r.weight_bytes,
        REGION_BYTES,
        MIN_ELASTIC_RESERVE,
    );
    let before = ACHIEVABLE_WEIGHT.swap(achievable, Ordering::Relaxed);
    // The idle branch runs once per request while the engine waits, so this
    // fires many times a second at startup; a line per *change* of a region or
    // more is what a log wants.
    if before.abs_diff(achievable) >= REGION_BYTES as u64 {
        tracing::info!(
            target: "candle_conversation::scheduler::interleave",
            achievable_mib = achievable >> 20,
            zone_mib = r.weight_bytes >> 20,
            live_kv_mib = (r.live * REGION_BYTES) >> 20,
            hold_mib = (achievable as f64 * HOLD) as u64 >> 20,
            "weight residency to defend",
        );
    }
}

/// Fraction of the achievable residency the engine refuses to give up — the
/// **floor** of the range admission works inside.
///
/// **One measured constant, not a controller.** The weight zone opens at
/// 10,398 MiB on this box and the model decodes without streaming; driven to
/// 1,417 MiB it streams every layer of every forward and throughput collapses.
/// It was set by measurement, downward, each time a run wedged: four fifths put
/// a calibration's own working set under the mark; seven tenths did the same to
/// the ingest; 0.60 still refused stores with the width happy at 8. The
/// emergency is the zone the ingest collapsed at — 4.4 GiB of 9.4 achievable,
/// three seconds a step — and the floor sits just above it.
///
/// **It is a floor, not a target** — but it is a floor both kinds stand on.
/// [`optimal_weight_bytes`] is what `promote_new_prefills` hands `WaveFill`,
/// and the fill hands it to the planner as `WaveRate::reset`'s `floor_bytes`,
/// so `judge_prefill` and `judge_decode` both refuse an admission that would
/// land under it; the decode-start rule (`admit::gate::may_start_decode`)
/// consults the same range separately.
///
/// That matters most for a prefill, whose price is dominated by one 160 MiB
/// recurrent store: with the zone near the floor a prefill is refused for want
/// of a store's worth of ground, however wide the rows behind it. Run 36 spent
/// 13.5 minutes in that state — 384 refusals, `room_mib=95` against a
/// `claimed_mib=176` — which is why the idle demote now hands stores back
/// rather than only block tables.
const HOLD: f64 = 0.50;

/// The residency the weight side could reach, as last measured at idle — the
/// `max` of the range [`optimal_weight_bytes`] takes its fraction of.
pub(super) fn achievable_weight_now() -> Option<u64> {
    use std::sync::atomic::Ordering;
    weight_zone_bytes()?;
    match ACHIEVABLE_WEIGHT.load(Ordering::Relaxed) {
        0 => None,
        v => Some(v),
    }
}

/// The floor of the weight range, or `None` when there is nothing to defend.
///
/// **"The highest residency it can achieve" is measured, not asked for.** The
/// model's own `resident_weight_bytes` reports what is resident *now*, which
/// falls as experts are evicted — using it as the target makes the target chase
/// the collapse. [`reseed_achievable_weight`] measures the reachable residency
/// when nothing is in flight, and that figure holds until the engine is next
/// idle.
///
/// `None` until a reservation has been read at all (a CPU device or a unit
/// test), which is the generality case: nothing to defend, so admission is
/// bounded only by what the allocators will give.
pub(super) fn optimal_weight_bytes() -> Option<u64> {
    let achievable = achievable_weight_now()?;
    Some((achievable as f64 * HOLD) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity: span less the live KV less the floor's reserve.
    #[test]
    fn the_achievable_zone_is_the_span_less_what_stands_in_it() {
        // 100 regions of 16 MiB + a 1 GiB zone = a 2.5 GiB span; 40 live and a
        // 256 MiB reserve leave the weight side 1,632 MiB.
        let got = achievable_weight_bytes(100, 40, 1 << 30, 16 << 20, 256 << 20);
        assert_eq!(got, (1 << 30) + (60 * (16 << 20)) - (256 << 20));
    }

    /// **The effective zone already counts every free region**, so a caller may
    /// never add the free list to it.
    ///
    /// Run BV did exactly that — free list plus `effective - floor` — and spent
    /// 1.6 GiB of ground that was the same regions counted twice. The identity
    /// below is the reason: what the weight side could reach *is* where the
    /// boundary stands plus everything not currently live.
    #[test]
    fn the_effective_zone_already_counts_every_free_region() {
        let (total, live, region, extent) = (100usize, 40usize, 16usize << 20, 1usize << 30);
        let effective = achievable_weight_bytes(total, live, extent, region, 0);
        assert_eq!(
            effective,
            (extent + (total - live) * region) as u64,
            "effective = the extent plus the regions not live",
        );
    }

    /// **Claiming a free region costs the weight side exactly that region.**
    ///
    /// This is the identity admission's budget rests on. The free list is not
    /// headroom sitting *beside* the weight zone's — it is the same ground seen
    /// from the other end, so a budget may be `effective - floor` or it may be
    /// the free list, but never their sum and never one netted against the
    /// other. Both of those shapes shipped and both drained the zone: run BV
    /// stood 95 slots open on 1.6 GiB counted twice, and run CA reached 2,314 MiB
    /// of residency against a 4,772 MiB hold by its first directory.
    #[test]
    fn claiming_a_region_lowers_the_effective_zone_by_exactly_that_region() {
        let (total, region, extent) = (100usize, 16usize << 20, 1usize << 30);
        let before = achievable_weight_bytes(total, 40, extent, region, 0);
        let after = achievable_weight_bytes(total, 41, extent, region, 0);
        assert_eq!(
            before - after,
            region as u64,
            "one region claimed, one region of residency gone",
        );
    }

    /// A span already inside its reserve is zero, not a wrap.
    #[test]
    fn a_span_inside_its_reserve_is_zero_not_a_wrap() {
        assert_eq!(achievable_weight_bytes(1, 1, 0, 16 << 20, 1 << 30), 0);
        assert_eq!(achievable_weight_bytes(0, 99, 0, 16 << 20, 0), 0);
    }

    /// Off a reservation there is nothing to defend, and the floor says so
    /// rather than inventing one.
    #[test]
    fn no_reservation_means_no_weight_point() {
        assert_eq!(optimal_weight_bytes(), None);
        assert_eq!(achievable_weight_now(), None);
    }

    /// The floor sits under the achievable residency, never at or above it —
    /// a hold at the ceiling would refuse everything the moment it was read,
    /// and one at zero would defend nothing.
    #[test]
    fn the_hold_leaves_a_range_for_admission_to_work_inside() {
        let achievable = 10_000u64;
        let floor = (achievable as f64 * HOLD) as u64;
        assert!(floor > 0, "a zero floor defends no residency at all");
        assert!(
            floor < achievable,
            "a floor at the ceiling leaves nothing to admit into",
        );
    }
}
