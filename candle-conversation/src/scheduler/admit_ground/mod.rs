//! The scheduler as the admission ground: one admission pass over a settled
//! device.
//!
//! The policy of which turn runs, and whether the wave is better for carrying
//! it, lives in the admission fill. This module reads the state that policy
//! needs off the live scheduler and answers its offers. A pass is an object of
//! its own because its per-band cursor has to belong to that pass: a cursor kept
//! on the scheduler would outlive the pass and resume partway down a queue that
//! has since changed. Prefills and sections are offered, while decodes already
//! in flight are charged to the wave and never admitted again.
//!
//! - `ground`: the pass state and the residency reading each offer uses.
//! - `joins`: the rule for turns that join a spent forward.
//! - `pricing`: the tier, dislodge and rows an offer would take.
//! - `queues`: which queued work each band is offered next.
//! - `scheduler_side`: the measured headroom, floors and planner inputs.
//! - `tests_joins`: tests of joins and the deepest chunk.

use std::time::{Duration, Instant};

use super::admission::prefill_cost_bytes;
use super::admission_split::AdmitPart;
use super::admit::gate::streams_experts;
use super::admit::{Budget, Cost, FillPart, Ground, Headroom, Kind};
use super::band::rank;
use super::interleave;
use super::{Scheduler, SequenceId, WaveMember};
use super::{MAX_DECODE_WIDTH, MAX_PREFILL_WIDTH};
use crate::projection::DecodePriority;
use candle_nn::kv_cache::CHUNK_SIZE;

/// The least advance the engine hands a sequence, and so the width a forward
/// worth running is priced at. 128 rows.
pub(super) const PREFILL_MIN_ADVANCE: usize = 128;

/// The prefill cohort a pass is composing: the held creep group, whose
/// advances are fixed, then every member that will join it — admitted on an
/// earlier pass or by this one — with its remaining rows and depth.
///
/// **An offer is charged what it adds to the forward the cohort will run, not
/// what it would cost alone.** The tier is superlinear in spans and its launch
/// carve follows each member's depth, so N admissions priced as N one-sequence
/// waves understate the wave they compose — measured on the 3090 scratch
/// daemon, calibration passes admitting 32 cases against ~2 GiB of priced tier
/// whose forward then placed 4.48 GB and found no ground in 1,024 attempts.
///
/// **Composed when priced, the way the group is formed** ([`Scheduler::compose`]):
/// a member's rows are not fixed when it is admitted, because a turn that
/// joined a spent forward still rides the cohort's windows once the group
/// absorbs it. Pricing that join at no rows charged a cohort of 6,254 rows
/// that ran at 8,066.
#[derive(Clone, Default)]
pub(super) struct WaveDraft {
    held_rows: usize,
    held_advances: Vec<usize>,
    held_depths: Vec<usize>,
    /// The held members' sequences, whose live indexes their tier is priced over.
    held_seqs: Vec<usize>,
    /// `(remaining, depth, sequence)` of each member not in the held group.
    joining: Vec<(usize, usize, usize)>,
    /// The width the rate model chose for the cohort, which its forwards are
    /// composed at — `None` when no choice stands. A pass that widens clears
    /// it, one that closes sets it; the draft follows, so each offer is priced
    /// at the width its own admission leaves.
    width: Option<usize>,
}

/// A [`WaveDraft`] composed into the forward it will run.
pub(super) struct Composed {
    pub(super) rows: usize,
    seqs: usize,
    advances: Vec<usize>,
    depths: Vec<usize>,
    stage_seqs: Vec<usize>,
}

/// One admission pass: the engine, plus how far each prefill band has been
/// offered.
pub(super) struct AdmitPass<'a> {
    sched: &'a mut Scheduler,
    headroom: Headroom,
    budget: Budget,
    /// Bytes one 32-token KV block costs, in the formats a live sequence holds.
    per_block: u64,
    /// The store one more sequence costs, read at the open. A constant of the
    /// model's geometry — see [`Scheduler::recurrent_cost`].
    recurrent: u64,
    /// Residency this pass has already committed, summed over what it admitted.
    ///
    /// **Without it the floor is not defended in aggregate.** `fill` re-reads
    /// `resident_weights()` before every offer and tests
    /// `before - this_one_offer` against the floor — which is the right question
    /// only if `before` moves as the pass admits. It does not: nothing in
    /// `admit` claims a region (the K/V is claimed per chunk by the forward, not
    /// here), so the live region count the zone is derived from stands still for
    /// the whole pass. Ten turns that each individually clear a floor 4 GiB
    /// below the zone would all be admitted, committing the engine to ten times
    /// the ground it checked for once.
    ///
    /// Subtracting what has been taken so far makes each offer face the wave as
    /// it now stands, which is what the model is written to be asked.
    committed: u64,
    /// Rows a prefill chunk may carry into this wave.
    chunk_rows: usize,
    /// Rows this pass has admitted, so each offer is sized to the room that is
    /// still free rather than to the whole of [`Self::chunk_rows`]. See `peek`.
    rows_taken: usize,
    /// The ground the wave already in flight reserves — its tier and its
    /// decodes' growth over a sweep, see [`Scheduler::standing_ground_bytes`].
    /// Read once: it describes committed work, so it cannot move under the
    /// pass.
    standing_tier: usize,
    /// The rows the cohort stood at when the pass opened. Read once, like the
    /// tier: what this pass admits joins the cohort as it goes, and is counted
    /// in [`Self::rows_taken`] instead.
    standing_rows: usize,
    /// The forward this pass is composing, and its tier — what each offer's
    /// tier is priced against. See [`WaveDraft`].
    wave: WaveDraft,
    wave_tier: u64,
    /// The rows [`Self::wave`] composes to — what an offer widens it from.
    wave_rows: usize,
    /// The member the offer [`Ground::peek`] last answered would add to
    /// [`Self::wave`]: its remaining rows and depth.
    offered: Option<(usize, usize, usize)>,
    /// Whether that offer was sized to the floor, so its admission closes the
    /// forward at its width — see `offer_tier`.
    narrowed: bool,
    /// Hole ground this pass has credited to admitted claims that have not yet
    /// landed — so two offers are not both told the same hole is theirs. See
    /// `admit::Cost::hole_credit`.
    holes_spoken: u64,
    /// The tier the held cohort can give up to the offers this pass admits,
    /// and how much of it they have been credited — see `Cost::narrow_credit`.
    narrowable: u64,
    narrow_spoken: u64,
    /// How far down `prefill_queue` this pass has offered, **per priority
    /// band**.
    ///
    /// One cursor cannot serve the bands: they are offered in turn and each has
    /// to resume where *it* left off, not where the previous band did. Shared,
    /// the `Low` band would start from wherever `High` had walked to and skip
    /// every `Low` candidate ahead of it.
    prefill_cursor: [usize; 4],
    /// Per band, the next pending submission to offer ([`Scheduler::pending_submits`]),
    /// scanned once the band's queued work is spent.
    submit_cursor: [usize; 4],
    /// Whether the offer [`Ground::peek`] last answered is a pending submission,
    /// which [`Ground::admit`] then materialises before it begins.
    offering_submit: bool,
    /// Whether this pass admitted prefill rows, and whether it closed them on
    /// the rate — what [`Self::conclude`] reads to replace the cohort's width.
    widened: bool,
    closed: bool,
    /// Why an offer was never made, when one was waiting: see [`Unoffered`].
    unoffered: Unoffered,
}

/// Why a pass stopped offering work that was still waiting, without refusing
/// it. Neither is a judgement, so neither reaches the rate model's refusal
/// flags — but a pass that ends on one with turns queued reads `admitted=0`,
/// which without this looks like admission turning work away for no reason.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct Unoffered {
    /// The forward's rows were full ([`room_left`]): the cohort in flight
    /// takes every row the pass budget allows, and a turn joining it would
    /// advance by nothing.
    pub rows: bool,
    /// The [`MAX_PREFILL_WIDTH`] backstop was reached.
    pub width: bool,
}

mod ground;
mod joins;
mod pricing;
mod queues;
mod scheduler_side;
#[cfg(test)]
mod tests_joins;

use ground::*;
pub(super) use joins::*;
