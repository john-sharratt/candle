//! How much decode airtime a background prefill must leave before it may ride a
//! wave again, and when finished turns are cleaned up.
//!
//! A co-batched step must not take decode's share of the device, so the creep
//! that rides it owes decode its share of that step, multiplied by the airtime
//! ratio the decode protects itself with. The debt is paid in decode-only steps
//! rather than in quanta, so the creep rides again as soon as decode has had its
//! share. Until a decode-only step has been measured, the whole step counts as
//! the creep's, which errs toward decode. A finished turn is cleaned up at once
//! when it wrote a tool call, since its caller waits on the `Done`; otherwise it
//! waits for the cohort to drain, up to a bound.

use super::*;

/// Whether finished turns are cleaned up after this decode step: at once when one
/// of them is `urgent` (it wrote a tool call), once nothing is left decoding, or
/// once the earliest of them has waited [`CLEANUP_DEFER`].
pub(super) fn cleanup_due(
    first_finished: Option<Instant>,
    now: Instant,
    decoding: usize,
    urgent: bool,
) -> bool {
    match first_finished {
        None => false,
        Some(at) => urgent || decoding == 0 || now.saturating_duration_since(at) >= CLEANUP_DEFER,
    }
}

/// The decode-only airtime a step that also carried the prefill creep owes
/// before the creep may ride again: the creep's share of the step — its length
/// `step` less what a decode-only step takes now (`decode_only`) — times the
/// airtime ratio `R` the creep yielded to (the ratio of the decodes that
/// protect themselves from it — `prefill::throttle_ratio`). Decode gets `R`
/// times the creep's airtime.
///
/// **A co-batched step must not take decode's share of the device.** At `R = 1`
/// (no decode protects itself from the creep — an ingest beside its own turns)
/// a step carries every layer of an 8K-token
/// chunk: 4.2 s at depth. Measured in zend, with the creep riding every step it
/// could: 24 of 135 steps carried it and took 100 s of a 120 s window — eight
/// sessions decoding ~14% of the time where `R = 1` means equal airtime.
///
/// **And the debt is paid in decode steps, not in quanta.** Paying it by letting
/// the creep ride only a quantum's first step held a prefill to one advance per
/// [`WAVE_SLICE`] whenever anything decoded: an 8K chunk that runs in 120 ms
/// went once every 2 s, and a turn's tool result waited out the rest of another
/// conversation's slice before its prefill moved — 2 s of nothing between the
/// turns of a live dialogue, and an ingest of 30 workers finishing 0.7 folders a
/// minute. Owed by the step, the creep rides again as soon as decode has had its
/// share.
///
/// **The creep's share, not the step's length over `R`.** Owing `step / R`
/// assumed the creep's `N / R` layers were a sliver of a step; at depth they are
/// not. One layer of an eight-member ingest group at 300K positions cost ~40 ms
/// beside a ~30 ms chat decode, and at `R = 64` it owed 1 ms — so it rode every
/// step, and a `High` dialogue decoded 13–15 steps a second where it ran 33
/// alone (`.5`, 2026-10-17). The share times `R` is what `R` says: beside a
/// dialogue that ingest rides about once in 65 steps' worth of decode airtime,
/// and at full speed once the dialogue stops decoding. With no decode-only step
/// measured yet the whole step counts as the creep's, which errs toward decode.
pub(super) fn creep_owed_after(
    step: Duration,
    decode_only: Option<Duration>,
    ratio: f64,
) -> Duration {
    let share = step.saturating_sub(decode_only.unwrap_or(Duration::ZERO));
    share.mul_f64(ratio.max(0.0))
}

/// The airtime decode is owed after a step of length `step`: what a step that
/// carried the creep or a section chunk owes (`creep_owed_after`), or what was
/// owed less a step that did not.
pub(super) fn next_creep_owed(
    owed: Duration,
    step: Duration,
    carried: bool,
    decode_only: Option<Duration>,
    ratio: f64,
) -> Duration {
    if carried {
        creep_owed_after(step, decode_only, ratio)
    } else {
        owed.saturating_sub(step)
    }
}

/// What a decode-only step takes now, after one more of length `step`: an
/// exponential average weighting the newest by [`DECODE_ONLY_WEIGHT`], so it
/// follows the decode width and depth within a few steps.
pub(super) fn next_decode_only(avg: Option<Duration>, step: Duration) -> Duration {
    match avg {
        None => step,
        Some(a) => a.mul_f64(1.0 - DECODE_ONLY_WEIGHT) + step.mul_f64(DECODE_ONLY_WEIGHT),
    }
}

/// The newest decode-only step's weight in [`next_decode_only`].
pub(super) const DECODE_ONLY_WEIGHT: f64 = 0.25;
