//! How much of a creep rides a step that also carries decode rows.
//!
//! The airtime ratio `R` (`docs/continuous_fair_waves.md` §3) sets how the GPU
//! is *shared* between the creep and the decodes beside it: a co-batched step
//! owes decode its creep share times `R` before the creep rides again. It used
//! to set the *slice* too — `ceil(N / R)` layers a step — and at `R = 1` that is
//! the whole sweep. An 8K-row ingest group at depth rode one decode step for
//! 1.7 s (`.5`, 2026-10-17, `decode:cobatched:step` 28 × 1,737 ms against
//! `decode_only:step` 1,636 × 34 ms), and every decode beside it — the ingest's
//! own summaries — stood still for each of them: ~5 tokens a second where it
//! ran ~100 alone.
//!
//! So a slice is bounded in time as well: no more layers than one decode-only
//! step's worth of creep, learned from what a layer-row of creep has cost on
//! the steps it rode. The share is unchanged — the debt still owes decode the
//! slice times `R` — but it is paid in turns of a step or two, not seconds, so
//! a co-batched step costs at most about twice a decode-only one. With nothing
//! decoding there is nothing to protect and the creep runs its `ceil(N / R)`.

use std::time::Duration;

/// The newest step's weight in [`next_layer_row_cost`].
const COST_WEIGHT: f64 = 0.25;

/// Layers a creep of `rows` rows advances beside decode rows: at most `cap`
/// (`ceil(N / R)`, and what is left of the sweep), at least one, and no more
/// than fit in `target` (a decode-only step) at the larger of `per_layer_row`
/// times the rows and `per_layer` — what a layer of creep has cost whatever
/// its rows. Unmeasured, a single layer — which errs toward the decodes and
/// measures.
///
/// **A layer costs what its experts cost, not only what its rows do.** On a
/// model that streams its experts, a layer of creep pays for staging every
/// expert its rows route, which a few hundred rows already route most of: a
/// prompt's layer on Qwen3.8-Flash-Next over the RTX 3090's PCIe 3.0 link is
/// ~60 ms at any width. Priced per row alone, the cost learned from a wide
/// slice predicted a few milliseconds for a narrower one, and slices of ~15
/// layers rode a 95 ms decode step: co-batched steps of 772 ms, 46% of the
/// wall clock, every decode beside them standing still for each (WhiteWorld,
/// 2026-10-26). The per-layer term is the floor the rows cannot go under.
pub(super) fn time_bounded_layers(
    cap: usize,
    rows: usize,
    per_layer_row: Option<Duration>,
    per_layer: Option<Duration>,
    target: Option<Duration>,
) -> usize {
    let cap = cap.max(1);
    let (Some(cost), Some(target)) = (per_layer_row, target) else {
        return 1;
    };
    let by_rows = cost.as_secs_f64() * rows.max(1) as f64;
    let per_layer = by_rows.max(per_layer.map_or(0.0, |d| d.as_secs_f64()));
    if per_layer <= 0.0 {
        return cap;
    }
    ((target.as_secs_f64() / per_layer) as usize).clamp(1, cap)
}

/// What a layer-row of creep costs after a step of length `step` that carried
/// `layer_rows` of it beside decode rows a decode-only step takes
/// `decode_only` for: the step's excess over decode, per layer-row, folded
/// into an exponential average. A step no longer than decode-only teaches
/// nothing.
pub(super) fn next_layer_row_cost(
    avg: Option<Duration>,
    step: Duration,
    decode_only: Duration,
    layer_rows: usize,
) -> Option<Duration> {
    if layer_rows == 0 {
        return avg;
    }
    let excess = step.saturating_sub(decode_only);
    if excess.is_zero() {
        return avg;
    }
    let sample = excess.div_f64(layer_rows as f64);
    Some(match avg {
        None => sample,
        Some(a) => a.mul_f64(1.0 - COST_WEIGHT) + sample.mul_f64(COST_WEIGHT),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// **The 1.7 s step, bounded.** 8,190 rows at 4.4 µs a layer-row is ~36 ms
    /// a layer; beside a 34 ms decode-only step that is one layer, not the 48
    /// a ratio of one allowed.
    #[test]
    fn a_whole_sweep_beside_decodes_becomes_a_layer() {
        let cost = Duration::from_nanos(4_400);
        assert_eq!(
            time_bounded_layers(48, 8_190, Some(cost), None, Some(ms(34))),
            1
        );
        // A short prompt's layers are cheap: many fit.
        assert_eq!(
            time_bounded_layers(48, 160, Some(cost), None, Some(ms(34))),
            48
        );
        // Never past the cap the ratio sets.
        assert_eq!(
            time_bounded_layers(3, 160, Some(cost), None, Some(ms(34))),
            3
        );
    }

    /// **A layer's own cost is the floor.** The WhiteWorld case: a per-row cost
    /// learned from a wide slice put 300 rows at 3 ms a layer, and 31 layers in
    /// a 95 ms decode step; each layer costs ~60 ms whatever its rows, so one
    /// fits.
    #[test]
    fn a_layer_that_streams_its_experts_rides_alone() {
        let row = Duration::from_nanos(10_000);
        assert_eq!(
            time_bounded_layers(48, 300, Some(row), None, Some(ms(95))),
            31
        );
        assert_eq!(
            time_bounded_layers(48, 300, Some(row), Some(ms(60)), Some(ms(95))),
            1
        );
        // Where the rows cost more than the layer's floor, the rows decide.
        assert_eq!(
            time_bounded_layers(48, 8_190, Some(row), Some(ms(1)), Some(ms(250))),
            3
        );
    }

    /// Unmeasured, the first slice is one layer.
    #[test]
    fn an_unmeasured_creep_rides_a_single_layer() {
        assert_eq!(time_bounded_layers(48, 8_190, None, None, Some(ms(34))), 1);
        assert_eq!(time_bounded_layers(48, 8_190, Some(ms(1)), None, None), 1);
    }

    /// The cost is the step's excess over decode-only, per layer-row, and it
    /// averages: the first sample stands, later ones move it by a quarter.
    #[test]
    fn the_layer_row_cost_is_the_excess_over_decode() {
        let first = next_layer_row_cost(None, ms(70), ms(34), 8_000).unwrap();
        assert_eq!(first, Duration::from_nanos(4_500));
        let second = next_layer_row_cost(Some(first), ms(34 + 72), ms(34), 8_000).unwrap();
        assert_eq!(second, Duration::from_nanos(4_500 * 3 / 4 + 9_000 / 4));
        assert_eq!(
            next_layer_row_cost(Some(first), ms(30), ms(34), 8_000),
            Some(first)
        );
        assert_eq!(next_layer_row_cost(None, ms(70), ms(34), 0), None);
    }
}
