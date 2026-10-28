//! The scheduler as [`admit::Ground`] — one admission pass over a settled
//! device.
//!
//! [`admit::fill`] is the policy: whose turn it is, what the offer costs, and
//! whether the wave goes faster carrying it. This file is the other half, read
//! off the live scheduler.
//!
//! # What this pass offers, and what it does not
//!
//! **Prefills and sections.** A section waits in `Scheduler::section_queue`
//! and is offered in its band, priced as a prefill is — its K/V, its recurrent
//! store, its rows in the tier — except that it never decodes. It is a first
//! admission exactly as a prefill is. Sections put into flight as their requests
//! drain take a minute to push the weight zone from 10,398 MiB to its hold, and
//! on Qwen3.8-Flash-Next on a 16 GB card fill the ground below the weight floor
//! until no wave's tier can be placed.
//!
//! A **decode** is never admitted here because nothing queues one — a decode
//! exists because a prefill finished and promoted, which is one slot changing
//! phase rather than a new claim. [`admit::fill`] already charges every active
//! decode to the wave before the first offer, so they are priced into every
//! judgement without being offered.
//!
//! # Why a pass object rather than `impl Ground for Scheduler`
//!
//! `fill` walks its bands and offers each FIFO until it stops paying, so `peek`
//! has to be able to say "nothing more from this band" without consuming
//! anything. That needs a cursor, and one that belongs to the pass: carried on
//! the scheduler it would survive between passes and the next would resume part
//! way down a queue that has since changed.
//!
//! The pass also caches the two figures `fill` reads once — the headroom and the
//! budget — so every offer in one pass is judged against the same settled
//! moment. Residency is the deliberate exception, re-read per offer for the
//! reason [`admit::Ground::resident_weights`] gives: it moves under the fill's
//! feet from places admission cannot see.

use std::collections::HashMap;
use std::rc::Rc;

use super::admission::prefill_cost_bytes;
use super::admit::{Budget, Cost, Ground, Headroom, Kind};
use super::band::rank;
use super::interleave;
use super::next_group::{Candidate, NextGroup, Offer, Prospect};
use super::walked::Walked;
use super::{Scheduler, SequenceId, WaveMember};
use crate::projection::DecodePriority;
use candle_nn::kv_cache::CHUNK_SIZE;

/// The least advance the engine hands a sequence, and so the width a forward
/// worth running is priced at. 128 rows.
pub(super) const PREFILL_MIN_ADVANCE: usize = 128;

/// Residency as admission measures it.
///
/// **Neither what is resident nor what the zone reads.** Both are wrong, in
/// opposite directions, and this engine has now been wedged by each:
///
/// * *Occupancy* (`resident_vram_bytes`, occupied slots × slot size) is what the
///   cache holds, and it is compared against figures that are capacities. It sits
///   below them on a warming cache, so every offer is refused on a healthy zone
///   and nothing an admission does moves it up. Measured: two turns queued,
///   nothing in flight, `stopped_on_weights` on every pass.
/// * *The zone's extent* fails the other way. It moves only when the boundary
///   moves, and the boundary moves only between forwards — so a store or a KV
///   chunk taken from the free list does not change it at all. The zone read the
///   same after twenty-five stores as before the first, every one of them passing
///   a hold that watched a number which had not moved; the boundary caught up
///   later and the zone fell 2 GiB under the mark in one step.
///
/// What a hold compares against is [`interleave::effective_weight_zone_bytes`]:
/// the residency the weight side could hold *right now*, derived from the span
/// identity (`total × region + weight = span`) against the **live** region count,
/// so each claim is charged the instant it is made rather than whenever the
/// boundary next catches up. `standing_tier` is the wave already committed — a
/// fact, not a choice — and the tier each new admission adds is charged
/// separately as `admit::Cost::dislodged_bytes`, so between them the wave's tier
/// is counted exactly once.
///
/// `u64::MAX` when there is no reservation to read (a CPU device, a unit test):
/// nothing to defend, so admission is bounded only by what the allocators give.
fn residency_now(standing_tier: usize) -> u64 {
    interleave::effective_weight_zone_bytes(standing_tier).unwrap_or(u64::MAX)
}

/// The first position of a `rows`-wide chunk ending at the last token of a
/// `turn`-token prefill that starts at `depth` — the deepest chunk the turn
/// runs at that width. A turn no wider than one chunk is its own deepest.
fn deepest_chunk_offset(depth: usize, turn: usize, rows: usize) -> usize {
    depth + turn.saturating_sub(rows)
}

/// The prefill cohort a pass opens on: the held creep group, whose advances
/// are fixed, then every member admitted on an earlier pass that will join it,
/// with its remaining rows and depth — the ground the standing wave reserves,
/// and the rows the rate model charges before any offer. Each offer is priced
/// in the group it joins instead ([`Scheduler::group_in`]).
///
/// **The admitted members are part of it, not only the held group.** The tier
/// is superlinear in spans and its launch carve follows each member's depth,
/// so N admissions priced as N one-sequence waves understate the wave they
/// compose — measured on the 3090 scratch daemon, calibration passes admitting
/// 32 cases against ~2 GiB of priced tier whose forward then placed 4.48 GB and
/// found no ground in 1,024 attempts.
///
/// **Composed the way the group takes them** ([`Scheduler::compose`]): a
/// member's rows are not fixed when it is admitted, because a turn that joined
/// a spent forward still rides the cohort's windows once the group absorbs it.
/// Composed at no rows, that join charged a cohort of 6,254 rows that ran at
/// 8,066.
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
    /// composed at — `None` when no choice stands.
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

impl Composed {
    /// Add `seq`'s `rows`, staged at the deepest chunk of the `remaining` it
    /// has left from `depth` — see [`deepest_chunk_offset`].
    fn stage(&mut self, seq: usize, depth: usize, remaining: usize, rows: usize) {
        self.rows += rows;
        self.advances.push(rows);
        self.depths
            .push(deepest_chunk_offset(depth, remaining, rows));
        self.stage_seqs.push(seq);
    }
}

/// A priority's slot in the per-band cursors. Four bands, in the order
/// `admit::order::BANDS` offers them.
fn band_index(prio: DecodePriority) -> usize {
    match prio {
        DecodePriority::Urgent => 0,
        DecodePriority::High => 1,
        DecodePriority::Normal => 2,
        DecodePriority::Low => 3,
    }
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
    /// The ground the wave already in flight reserves — its tier and its
    /// decodes' growth over a sweep, see [`Scheduler::standing_ground_bytes`].
    /// Read once: it describes committed work, so it cannot move under the
    /// pass.
    standing_tier: usize,
    /// The rows the cohort stood at when the pass opened. Read once, like the
    /// tier: it is what the rate model charges the wave before any offer, and
    /// what it adds [`Self::rows_taken`] to.
    standing_rows: usize,
    /// Rows this pass has admitted — with the standing rows, the cohort's
    /// width as the rate model judges it, which a refusal closes at
    /// (`close_rows`).
    rows_taken: usize,
    /// The width the next group composes at as this pass leaves it — `None`
    /// for the deployment's target. It opens on the rate model's standing
    /// choice; an admission that widens its group clears it (`take_offered`,
    /// and the cohort's at `conclude`), and closing sets it, raised to the
    /// chunk the cohort composes at no less than (`Scheduler::set_rate_width`).
    width: Option<usize>,
    /// Whether the offer [`Ground::peek`] last answered was sized to the
    /// floor, so its admission closes the forward — see `offer_tier`.
    narrowed: bool,
    /// Per band, what the formation an offer of that band joins draws from
    /// ([`Scheduler::prospect`]) — `Some(None)` when a paused creep resumes
    /// ahead of the band — and, per band, composition target and formation,
    /// that formation without any offer. Formed once, and again after each
    /// admission puts more work in flight.
    prospects: [Option<Option<Rc<Prospect>>>; 4],
    withouts: HashMap<(usize, usize, bool), Without>,
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
    /// The slots this pass walked work past on with no place in its group
    /// (`peek_prefill`).
    walked: Walked,
    /// Whether the offer [`Ground::peek`] last answered is a pending submission,
    /// which [`Ground::admit`] then materialises before it begins.
    offering_submit: bool,
    /// Whether this pass admitted prefill rows, and whether it closed them on
    /// the rate — what [`Self::conclude`] reads to replace the cohort's width.
    widened: bool,
    closed: bool,
}

/// A band's queued turn, as [`AdmitPass::peek_prefill`] finds it.
enum Queued {
    /// Priced, at the cost given.
    Offered(Cost),
    /// It has no place in the group it would join ([`AdmitPass::room`]).
    NoPlace,
    /// The band's queued work is spent; its pending submissions answer.
    Spent,
}

/// Where an offer stands ([`AdmitPass::room`]): what the formation it joins
/// draws from and the target it composes at, the group with the offer in it,
/// and that formation without it at the width the pass stands at — what the
/// next forward carries if the offer is not admitted.
struct Placed {
    prospect: Rc<Prospect>,
    target: usize,
    standing: usize,
    group: NextGroup,
    without: Without,
}

/// A formation drawn without any offer — what an offer's group is measured
/// against: the rows it carries and the tier it places.
#[derive(Clone, Copy)]
struct Without {
    rows: usize,
    tier: u64,
}

impl Without {
    /// The rows an offer widens this formation by, as `group` — the same
    /// formation with it — carries them: none when it is no member there.
    fn widened_by(&self, group: &NextGroup) -> usize {
        if group.offer_at.is_none() {
            return 0;
        }
        group.rows().saturating_sub(self.rows)
    }
}

impl<'a> AdmitPass<'a> {
    pub(super) fn new(sched: &'a mut Scheduler) -> Self {
        let per_block = sched.per_block_kv_bytes();
        let recurrent = sched.recurrent_cost();
        sched.settle_rate_width();
        // Offered against the ceiling, not the width the last pass chose: a
        // pass that finds wider rows paying widens the cohort's forwards.
        let chunk_rows = sched.prefill_pass_ceiling();
        let wave = sched.standing_wave();
        let composed = sched.compose(&wave);
        let wave_tier = sched.composed_tier(&composed);
        let narrowable = Self::held_narrowable(sched, &wave);
        // Net of what the running decodes will claim over a sweep: see
        // `Scheduler::standing_ground_bytes`.
        let growth = sched.decode_growth_bytes();
        let standing_tier = (wave_tier as usize)
            .saturating_add(growth)
            .saturating_add(sched.glue_scratch_bytes())
            .saturating_add(sched.held_claims_outstanding());
        tracing::debug!(
            target: "candle_conversation::scheduler::admission",
            standing_tier_mib = wave_tier >> 20,
            decode_growth_mib = growth >> 20,
            residency_mib = residency_now(standing_tier) >> 20,
            holes_mib = interleave::stranded_hole_bytes() >> 20,
            standing_rows = composed.rows,
            members = composed.seqs,
            "admission pass opened",
        );
        let headroom = sched.admit_headroom(standing_tier);
        let budget = sched.admit_budget_terms(&headroom, chunk_rows);
        Self {
            sched,
            headroom,
            budget,
            per_block,
            recurrent,
            committed: 0,
            standing_tier,
            standing_rows: composed.rows,
            rows_taken: 0,
            width: wave.width,
            narrowed: false,
            prospects: Default::default(),
            withouts: HashMap::new(),
            holes_spoken: 0,
            narrowable,
            narrow_spoken: 0,
            prefill_cursor: [0; 4],
            submit_cursor: [0; 4],
            walked: Walked::default(),
            offering_submit: false,
            widened: false,
            closed: false,
        }
    }

    /// End the pass: a pass that admitted rows and never stopped on the rate
    /// found every row it offered paying, so the width an earlier pass chose
    /// no longer holds and the cohort's forwards may widen to the budget. A
    /// pass that closed its rows has already recorded its own width
    /// (`close_rows`); one that admitted no rows judged nothing.
    pub(super) fn conclude(self) {
        if self.widened && !self.closed {
            self.sched.clear_rate_width();
        }
    }

    /// What an offer adds to `group`, the group it runs in
    /// ([`Scheduler::group_in`]): the tier that group places beyond what the
    /// same formation places `without` it, and the rows it advances there.
    /// Zero when the model cannot price one.
    ///
    /// **Priced in that group, not the cohort standing now.** Mid-sweep the
    /// standing cohort is the held group at its fixed rows, and a turn that
    /// pauses it — or leads the group re-formed at its boundary — runs in a
    /// forward those rows are not in. And against the group as it would form
    /// without it, its members' rows re-shared: a turn that leads a full group
    /// takes rows from the members behind it in a forward no wider, and places
    /// next to no more tier. Measured against its fellow members at the shares
    /// they keep beside it instead, it was charged the steepest slice of a
    /// superlinear tier for rows the forward was carrying anyway. A group
    /// smaller than the one it displaces — a `High` turn over a background
    /// group — adds nothing: the members left out were priced at their own
    /// admission.
    ///
    /// A join is priced at the rows the group gives it, not at none: it is
    /// only offered as a member ([`room_left`]), and it rides the group's
    /// forwards as one. Composed at no rows, a join charged a cohort of 6,254
    /// rows that ran at 8,066.
    fn priced(&self, group: &NextGroup, without: Without) -> (u64, usize) {
        let tier = self.sched.group_tier(group);
        (tier.saturating_sub(without.tier), group.offer_rows())
    }

    /// An offer whose claim (its K/V and store) is `claim` bytes, given the
    /// rows free for it where it stands, as [`Self::room`] found them: the rows
    /// it widens its group by, the tier it adds and the creep residual it
    /// owes.
    ///
    /// **The rows the rate model weighs are the group's widening, not the
    /// offer's share of it.** A turn that leads a group re-formed at its
    /// boundary takes rows from the members behind it, and the forward runs no
    /// wider: charged its share on top of the standing cohort, an 87-row `High`
    /// turn beside a creep already at the pass ceiling was refused for the
    /// rows' cap. It joins instead, as a turn does that shares the forwards to
    /// come rather than widening this one.
    ///
    /// Staged at its **deepest** chunk, not its first: the terms that follow
    /// depth — the pre-stage up to its key-chunk bound, and on a
    /// sparse-selection model the scoring over every candidate block the
    /// deepest row can see — grow as the turn advances, so a 128K turn's last
    /// chunk costs a tier its first does not. A full-width chunk ending at the
    /// turn's last token bounds every chunk the turn will run.
    ///
    /// The residual is sized by the group's rows, so each offer owes its own
    /// rows' share — a join the group absorbs as surely as a turn with rows in
    /// this forward.
    ///
    /// **Priced at the width its own admission leaves** — the one its group
    /// was placed at ([`Self::room`]).
    ///
    /// **Sized to the floor when it cannot fit whole.** A turn whose rows would
    /// take residency under the prefill floor is offered the widest rows that
    /// still clear it — never fewer than the least advance — and the pass
    /// closes at the cohort's width when it is admitted (`take_offered`). The
    /// formation does not carry the clip: a turn that leads its group takes
    /// its share of that width, and the group it forms is fitted to the
    /// ground its windows place (`fit_cohort_to_ground`). Offered whole, it is
    /// refused on every pass: measured on the 3090 scratch daemon (2026-10-18),
    /// a 4,325-row `High` tool-result prefill priced at a 4,992 MiB tier
    /// against 954 MiB of room, refused for minutes while the person waited,
    /// 22 turns cut at the client's deadline.
    fn offer_tier(
        &mut self,
        (room, placed): (usize, Placed),
        offer: Candidate,
        claim: u64,
    ) -> (usize, u64, u64) {
        self.narrowed = false;
        let rows = offer.remaining.min(room);
        let (tier, own) = self.priced(&placed.group, placed.without);
        let residual = self.residual_for(own);
        if rows == 0 {
            return (0, tier, residual);
        }
        let spendable = self
            .resident_weights()
            .saturating_sub(self.budget.prefill_floor);
        if self.dislodges(claim, residual, tier) <= spendable {
            return (placed.without.widened_by(&placed.group), tier, residual);
        }
        // Chunk-aligned rows, bisected: the tier and the residual both grow
        // with the rows, so the rows that fit are a prefix. Only the offer's
        // cap moves, so its group is formed from the same prospect each time.
        let least = PREFILL_MIN_ADVANCE.min(rows).div_ceil(CHUNK_SIZE);
        let (mut lo, mut hi) = (least, rows.div_ceil(CHUNK_SIZE) - 1);
        let mut fit = None;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let r = (mid * CHUNK_SIZE).min(rows);
            let capped = Offer {
                who: offer,
                cap: Some(r),
            };
            let group = self
                .sched
                .group_in(&placed.prospect, Some(capped), placed.target);
            let without = self.without(offer.band, &placed.prospect, &group, placed.standing);
            let (tier, own) = self.priced(&group, without);
            let residual = self.residual_for(own);
            if self.dislodges(claim, residual, tier) <= spendable {
                fit = Some((without.widened_by(&group), tier, residual));
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        match fit {
            Some(narrow) => {
                self.narrowed = true;
                narrow
            }
            // Not even the least advance fits: the whole offer is the honest
            // price, and the floor refuses it.
            None => (placed.without.widened_by(&placed.group), tier, residual),
        }
    }

    /// The formation `like` came from, drawn from `prospect` — an offer at
    /// `band`'s — without any offer and composed at `target`: formed once per
    /// band, target and formation until an admission puts more in flight.
    fn without(
        &mut self,
        band: DecodePriority,
        prospect: &Prospect,
        like: &NextGroup,
        target: usize,
    ) -> Without {
        let key = (band_index(band), target, like.alone);
        if let Some(&without) = self.withouts.get(&key) {
            return without;
        }
        let group = self.sched.without_in(prospect, like, target);
        let without = Without {
            rows: group.rows(),
            tier: self.sched.group_tier(&group),
        };
        self.withouts.insert(key, without);
        without
    }

    /// What an offer claiming `claim` with this residual and tier dislodges,
    /// its hole credit taken — the figure the floor is judged on.
    fn dislodges(&self, claim: u64, residual: u64, tier: u64) -> u64 {
        self.credited(Cost {
            kv: claim,
            residual,
            activations: tier,
            ..Default::default()
        })
        .dislodged_bytes()
    }

    /// The offer [`Self::offer_tier`] priced, widening its group by `rows`, was
    /// admitted: the rate model has charged those rows and judged them paying,
    /// so they count toward the cohort's width and toward clearing the rate
    /// model's narrower choice (`conclude`), and one sized to the floor closes
    /// the cohort's forward. When it `joined` — a turn of it in flight now,
    /// not answered elsewhere or handed back to the queue — the next offer's
    /// group carries it, and one that widened its group clears the pass's
    /// width.
    ///
    /// **A turn sized to the floor closes the cohort's forward, not one of
    /// its own.** The width is the rate model's, read by every formation until
    /// the cohort shrinks; a narrower width for the turn's group alone would
    /// bind every other group as long as that turn prefilled, a long creep's
    /// included. Within the cohort's width the formation fits the group it
    /// forms to the ground its windows place (`fit_cohort_to_ground`).
    fn take_offered(&mut self, rows: usize, joined: bool) {
        self.rows_taken = self.rows_taken.saturating_add(rows);
        self.widened |= rows > 0;
        if joined {
            self.prospects = Default::default();
            self.withouts.clear();
        }
        if std::mem::take(&mut self.narrowed) {
            self.close_at(self.standing_rows.saturating_add(self.rows_taken));
        } else if joined && rows > 0 && !self.closed {
            self.width = None;
        }
    }

    /// End the forward's rows at `width`: the rate model's choice for the
    /// cohort, which its next group composes at.
    fn close_at(&mut self, width: usize) {
        self.sched.set_rate_width(width);
        self.closed = true;
        self.width = self.sched.rate_width.map(|w| w.rows);
    }

    /// `cost` with the part of its claim that lands in holes credited: the
    /// holes stranded below the frontier now, less those this pass has
    /// already promised to claims still to land.
    fn credited(&self, cost: Cost) -> Cost {
        let free = interleave::stranded_hole_bytes().saturating_sub(self.holes_spoken);
        let holed = Cost {
            hole_credit: cost.claimed_bytes().min(free),
            narrow_credit: 0,
            ..cost
        };
        // And what the held cohort can still give up, less what this pass has
        // already handed to the offers before it.
        let narrow = self.narrowable.saturating_sub(self.narrow_spoken);
        Cost {
            narrow_credit: holed.dislodged_bytes().min(narrow),
            ..holed
        }
    }

    /// The tier the held cohort can give up at its next fit: what its members
    /// stand at now, less what they would stand at each held to the least
    /// advance. A running member is never taken out — it narrows, and keeps
    /// advancing — and `gang::gang_shares` gives the next fit's rows to the
    /// short members first, so a short newcomer finishes in the
    /// sweep it joins while the long members take what it leaves.
    ///
    /// **Only between sweeps.** The narrowing lands at the next fit, so it can
    /// pay only for claims that land there too. Mid-sweep, an admitted turn
    /// runs at once beside the sweep and claims its store and scratch now, on
    /// ground the sweep's remaining windows still need: on WhiteWorld
    /// (250ff7750, 2026-10-19) three turns credited this way mid-sweep took
    /// 432 MiB of scratch under a 6,365-row cohort, and its last window was
    /// refused seven regions short. A short prompt therefore waits for the
    /// sweep in flight to end, and joins at its re-formation.
    fn held_narrowable(sched: &Scheduler, wave: &WaveDraft) -> u64 {
        if wave.held_advances.is_empty() || sched.wave_prefill_cursor > 0 {
            return 0;
        }
        let at = |cap: usize| {
            let advances: Vec<usize> = wave.held_advances.iter().map(|&a| a.min(cap)).collect();
            Composed {
                rows: advances.iter().sum(),
                seqs: advances.len(),
                advances,
                depths: wave.held_depths.clone(),
                stage_seqs: wave.held_seqs.clone(),
            }
        };
        sched
            .composed_tier(&at(usize::MAX))
            .saturating_sub(sched.composed_tier(&at(PREFILL_MIN_ADVANCE)))
    }

    /// An admission credited `credit` of hole ground, and its carve has since
    /// filled `stranded_before` less what stands now: the rest is still owed.
    fn settle_holes(&mut self, stranded_before: u64, credit: u64) {
        let landed = stranded_before.saturating_sub(interleave::stranded_hole_bytes());
        self.holes_spoken = self
            .holes_spoken
            .saturating_add(credit)
            .saturating_sub(landed);
    }

    /// The creep residual `rows` more rows hold: the hidden state a sweep keeps
    /// between windows is sized by the cohort's rows and claimed when the
    /// cohort forms, so each offer owes its own rows' share. Linear in rows,
    /// so the shares a pass admits sum to what the cohort claims.
    fn residual_for(&self, rows: usize) -> u64 {
        self.sched
            .model
            .creep_residual_bytes(&self.sched.session, rows) as u64
    }
}

impl Ground for AdmitPass<'_> {
    fn active(&self) -> usize {
        self.sched.prefill_width() + self.sched.section_ingest_width() + self.sched.decode_width()
    }

    fn decodes_active(&self) -> usize {
        self.sched.decode_width()
    }

    fn settled(&self) -> bool {
        self.sched.settled_since_admit
    }

    fn headroom(&self) -> Headroom {
        self.headroom
    }

    fn budget(&self) -> Budget {
        self.budget
    }

    /// **Re-read per offer, in the effective zone's currency.**
    ///
    /// The cache's own resident bytes are the wrong input: they do not fall when
    /// a claim takes ground the weight side was about to fill, so admission
    /// would spend the whole free list before the model noticed anything had
    /// been dislodged, and only then discover the floor. The effective zone
    /// falls by exactly one region per region claimed, which is the identity the
    /// floor is defended in — so the rate, the dislodge and the floor are read
    /// in one currency.
    fn resident_weights(&self) -> u64 {
        residency_now(self.standing_tier).saturating_sub(self.committed)
    }

    /// Rows the next wave already carries before admission adds anything: the
    /// held creep group, which rides that wave whole, and the admitted members
    /// that will join it ([`Scheduler::standing_wave`]).
    ///
    /// **Charged to the wave before any offer is judged**, for two reasons. The
    /// offers behind it are then priced against the forward that will actually
    /// run rather than an empty one — a cohort already in flight is most of the
    /// copy those offers would otherwise look like they were amortising alone.
    /// And it denies the head waiver: the model carries its first offer
    /// unconditionally so a slot too large to ever fit cannot block the queue
    /// forever, and a wave already carrying a cohort is not that case.
    fn standing_rows(&self) -> usize {
        self.standing_rows
    }

    fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost> {
        // **The width backstop bounds the pass, not just its entry.** It is the
        // dumb ceiling beneath the rate model — there so an error in the cost
        // model costs throughput rather than the daemon — and checking it only
        // before the pass leaves it useless: one fill starting empty could take
        // as many prefills as the row budget allows, which is the case the
        // ceiling exists for.
        // Counted as work that can advance — prefills and sections alike, since
        // both are prefill rows in the same forwards — and a paused prefill
        // moves no row (`Scheduler::running_members`).
        // A prompt of a band the running members yield to is not held behind
        // them: they are why the width is spent (`Ground::outranks_active`).
        if self.sched.running_members() >= Scheduler::MAX_PREFILL_WIDTH
            && !(kind == Kind::Prefill && self.sched.lower_band_active(prio))
        {
            return None;
        }
        match kind {
            Kind::Prefill => self.peek_prefill(prio),
            Kind::Section => self.peek_section(),
            // See the module header: nothing queues a decode.
            Kind::Decode => None,
        }
    }

    /// The forward's rows end at what this pass has admitted: every later
    /// offer has no room and comes back as a join, and the forwards the cohort
    /// shares are composed at this width (`Scheduler::rate_width`).
    fn close_rows(&mut self) -> bool {
        // Closed already, the offer refused had no rows to give up: there is
        // no join to re-offer it as, and the refusal ends the band.
        if self.closed {
            return false;
        }
        // **At the cohort's width, as the rate model judged it** — the rows it
        // charged before any offer and those this pass admitted. The width is
        // the cohort's, read at every formation until the cohort shrinks
        // (`RateWidth::lapsed`); closed at the refused offer's own group
        // instead, an `Urgent` offer refused beside a 3,072-row creep would
        // compose that creep's next group at one chunk.
        self.close_at(self.standing_rows.saturating_add(self.rows_taken));
        true
    }

    fn outranks_active(&self, prio: DecodePriority) -> bool {
        self.sched.lower_band_active(prio)
    }

    fn decodes_at_or_above(&self, prio: DecodePriority) -> usize {
        self.sched.decodes_at_or_above(prio)
    }

    /// Every waiting prompt — queued work and pending submissions, of every
    /// band — counted `1 + age` with its age **in sweeps**, the time one
    /// prefill cohort takes as measured: a prompt that has waited a whole sweep
    /// counts twice, two sweeps three times. In sweeps rather than seconds so
    /// the weight follows how long this engine takes to serve a prompt, not a
    /// constant to re-derive per card; before a sweep has been measured each
    /// prompt counts once. At least one: the offer itself is waiting.
    fn prefill_weight(&self, _prio: DecodePriority) -> f64 {
        let sweep = self.sched.sweep_forecast.sweep_secs().filter(|s| *s > 0.0);
        let weight =
            |at: std::time::Instant| 1.0 + sweep.map_or(0.0, |s| at.elapsed().as_secs_f64() / s);
        let queued = self.sched.prefill_queue.iter().map(|w| w.submitted_at);
        let pending = self
            .sched
            .pending_submits
            .iter()
            .map(|p| p.work.submitted_at);
        queued.chain(pending).map(weight).sum::<f64>().max(1.0)
    }

    fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
        match kind {
            Kind::Prefill => self.admit_prefill(prio, cost),
            Kind::Section => self.admit_section(cost),
            Kind::Decode => false,
        }
    }
}

impl AdmitPass<'_> {
    /// Rows `offer` advances in the group it would run in
    /// ([`Scheduler::prospect`], [`Scheduler::group_in`]) — see `peek_prefill`
    /// — or `None` when
    /// it is no member of that group, that group has none for it beside the
    /// rest, or a paused creep resumes ahead of it. Every member of that group
    /// is counted, prompts and sections alike; work the offer sets aside, or
    /// leads at the group's re-formation, is not.
    ///
    /// Where it stands comes back beside the rows, composed at the width its
    /// admission would leave — the pass's own, cleared when it widens its
    /// group (`take_offered`) — so its pricing reads the same group.
    ///
    /// **Measured against the forward as it stands**, at the width the pass
    /// stands at, whatever width its own group is placed at: an admission that
    /// widens its group clears the cohort's width, so every member of that
    /// group widens with it, and that is the offer's to pay for.
    fn room(&mut self, offer: Candidate) -> Option<(usize, Placed)> {
        let slot = band_index(offer.band);
        if self.prospects[slot].is_none() {
            self.prospects[slot] = Some(self.sched.prospect(offer.band).map(Rc::new));
        }
        let prospect = self.prospects[slot].clone().flatten()?;
        // **Below the live band its group leads with, no formation has a place
        // for it** — the turns of a lower band queued behind a `High` group
        // are walked past without forming one each.
        if prospect.leaves_out(offer.band) {
            return None;
        }
        let asked = Offer {
            who: offer,
            cap: None,
        };
        let standing = self.sched.composition_target_at(self.width);
        let place = |this: &mut Self, target| {
            let group = this.sched.group_in(&prospect, Some(asked), target);
            let without = this.without(offer.band, &prospect, &group, standing);
            (group, without)
        };
        // **Placed at the deployment's target first**, the width an admission
        // that widens its group leaves (`take_offered`): a pass that finds
        // wider rows paying widens the cohort's forwards. A turn that would not
        // widen it there — one that joins, or has no place — is placed at the
        // width the pass stands at, which its group then composes at.
        let wide = self.sched.composition_target_at(None);
        let mut target = if self.closed { standing } else { wide };
        let (mut group, mut without) = place(self, target);
        // With no place at the wider target it has none at a narrower one: the
        // budget fills in the same order, and shrinking it only cuts the
        // members behind.
        if target != standing && group.offer_at.is_some() && without.widened_by(&group) == 0 {
            target = standing;
            (group, without) = place(self, target);
        }
        let share = group.offer_at.map(|_| group.offer_rows());
        let room = room_left(share, self.closed)?;
        Some((
            room,
            Placed {
                prospect,
                target,
                standing,
                group,
                without,
            },
        ))
    }

    /// The section band's next candidate: the head of the section queue,
    /// priced as a prefill is — its whole K/V, a recurrent store on a hybrid,
    /// and its rows in the forward's tier. It never decodes. Strictly FIFO and
    /// never paused: admission takes every section in this band, and a
    /// section's own band decides only how it rides the forwards once admitted.
    ///
    /// **A head with no place in the forward it would ride ends the band**,
    /// unlike a turn: a section can name the sections before it as its prefix
    /// (`prefix_section_ids`), and one ingested ahead of its prefix is sealed
    /// against what of it has landed — permanently.
    fn peek_section(&mut self) -> Option<Cost> {
        let s = self.sched.section_queue.front()?;
        let (turn, seq, band) = (s.tokens.len(), s.sequence_id, s.band);
        let depth = self.sched.session.sequence_offset(seq.0).unwrap_or(0);
        let offer = Candidate {
            seq: seq.0,
            band,
            remaining: turn,
            depth,
            section: true,
        };
        let room = self.room(offer)?;
        Some(self.price_section(room, offer))
    }

    /// The queued section `offer` stands for, where it stands, priced.
    fn price_section(&mut self, room: (usize, Placed), offer: Candidate) -> Cost {
        let (turn, seq, depth) = (offer.remaining, offer.seq, offer.depth);
        // A reused slot already holds a store, whose regions the residency
        // already counts: its restart zeroes that store where it stands.
        let recurrent = if self.sched.model.has_recurrent_state(seq) {
            self.sched.model.prefill_scratch_bytes(seq) as u64
        } else {
            self.recurrent
        };
        let kv = prefill_cost_bytes(turn, self.per_block)
            .saturating_add(self.sched.model.index_growth_bytes(depth, turn) as u64);
        let (rows, activations, residual) =
            self.offer_tier(room, offer, kv.saturating_add(recurrent));
        self.credited(Cost {
            kv,
            recurrent,
            residual,
            hole_credit: 0,
            narrow_credit: 0,
            activations,
            rows,
            decodes_after: false,
        })
    }

    /// Admit the section queue's head, priced at `cost`, and start it — its
    /// carried state claimed now, as a turn's is (`begin_admitted`). A claim the
    /// allocators refuse puts it back at the head and ends the band.
    fn admit_section(&mut self, cost: Cost) -> bool {
        let Some(section) = self.sched.section_queue.pop_front() else {
            return false;
        };
        let before = residency_now(self.standing_tier);
        let stranded = interleave::stranded_hole_bytes();
        if let Err(refused) = self.sched.begin_section(section) {
            let (section, why) = *refused;
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                "admitted section could not claim its state — it waits: {why}",
            );
            self.sched.section_queue.push_front(section);
            return false;
        }
        self.take_offered(cost.rows, true);
        self.settle_holes(stranded, cost.hole_credit);
        self.narrow_spoken = self.narrow_spoken.saturating_add(cost.narrow_credit);
        // See `admit_prefill`: what the section dislodges as it runs is carried
        // so the offers behind it face the wave as it now stands, less what the
        // claim has already taken.
        let claimed = before.saturating_sub(residency_now(self.standing_tier));
        self.committed = self
            .committed
            .saturating_add(cost.dislodged_bytes().saturating_sub(claimed));
        true
    }

    fn peek_prefill(&mut self, prio: DecodePriority) -> Option<Cost> {
        // **Walk past what this band is not being offered.** A band is asked for
        // its own priority's next candidate, and the queue interleaves them: on
        // the shipped configuration only the dialogue layer sets `High` and every
        // other layer takes `Low`, so a dialogue turn behind queued background
        // work is the ordinary case. Stopping at the head because it belongs to
        // another band gives that turn no precedence at all — it would be served
        // by FIFO position alone, and `admit::order`'s whole point is that at
        // `High` a person is blocked on the next token.
        //
        // The cursor is the pass's and this band's, so a band resumes where *it*
        // left off rather than where the previous band did, and nothing is
        // consumed by looking.
        let band = band_index(prio);
        loop {
            match self.peek_queued(prio, band) {
                Queued::Offered(cost) => return Some(cost),
                Queued::Spent => return self.peek_submit(prio),
                // Closed, the forward's rows end where they stand: no turn
                // behind it has rows to take either, and the band ends.
                Queued::NoPlace if self.closed => return None,
                // **A turn its group has no place for is walked past**, as
                // another band's is. Ending the band there held every turn
                // behind it: a turn of another adapter, or one a full group
                // leaves out, kept the ones that would lead the group — nearer
                // their first token, or of its adapter — from being offered.
                // Nothing that would reorder its slot goes ahead of it
                // ([`Walked`]).
                Queued::NoPlace => self.prefill_cursor[band] += 1,
            }
        }
    }

    /// The band's queued turn at its cursor, priced.
    fn peek_queued(&mut self, prio: DecodePriority, band: usize) -> Queued {
        while let Some(w) = self.sched.prefill_queue.get(self.prefill_cursor[band]) {
            if self.sched.decode_priority_or_high(w.sequence_id) == prio {
                break;
            }
            self.prefill_cursor[band] += 1;
        }
        // The band's queued work first; then the submissions behind it.
        let Some(w) = self.sched.prefill_queue.get(self.prefill_cursor[band]) else {
            return Queued::Spent;
        };
        self.offering_submit = false;
        // **Offer the turn's share of its group, not the whole cap.**
        //
        // Clipping to the pass cap made every offer the size of the entire budget,
        // so the first turn long enough to fill it took the wave alone and the next
        // one was refused `Cap` for exceeding a budget it was never shown. Measured
        // on this daemon at `--max-depth 3`: a 2,646-row head admitted, the next
        // turn offering its full 3,000 against a 4,880 cap, refused — the wave
        // running 2,646 of 4,880 with sequences queued and every wave
        // `seqs max=1`, at a fraction of the batched rate.
        //
        // Offered its share, a turn takes the rows the group will actually give
        // it and the wave fills with as many turns as it takes. A turn clipped
        // short is not turned away, it advances by that much and is offered again
        // next wave — which is how a prefill longer than one forward already
        // progresses.
        //
        // The members ahead of it count — they take their rows first — and the
        // work it sets aside or leads does not (`Scheduler::prospect`).
        //
        // **A full forward takes no more turns** ([`room_left`]): its rows are
        // filled in order (`gang::gang_shares`), so a turn past them would
        // advance by nothing while holding its store. See [`Self::room`].
        let (turn, seq) = (w.tokens.len(), w.sequence_id);
        let fork = self.sched.uncarved_case(w).is_some();
        if self.walked.holds(seq.0, fork) {
            // Held, it is walked past itself, and holds what is behind it.
            self.walked.past(seq.0, fork);
            return Queued::NoPlace;
        }
        // **A store is charged once.** A turn whose sequence already holds a
        // store has that store's regions in the residency this offer is read
        // against — charging it again prices the same ground twice. A submitted
        // turn's view is forked its parent's store at admission, and a
        // turn-group case is carved its view there, so neither holds one yet
        // and both are charged.
        let holds_store =
            self.sched.uncarved_case(w).is_none() && self.sched.model.has_recurrent_state(seq.0);
        // A store it holds is in the residency already; the prefill scratch it
        // will claim on top is not (`ManagedBatchedModel::prefill_scratch_bytes`).
        let recurrent = if holds_store {
            self.sched.model.prefill_scratch_bytes(seq.0) as u64
        } else {
            self.recurrent
        };
        let depth = self.sched.session.sequence_offset(seq.0).unwrap_or(0);
        let offer = Candidate {
            seq: seq.0,
            band: prio,
            remaining: turn,
            depth,
            section: false,
        };
        let Some(room) = self.room(offer) else {
            self.walked.past(seq.0, fork);
            return Queued::NoPlace;
        };
        let kv = prefill_cost_bytes(turn, self.per_block)
            .saturating_add(self.sched.model.index_growth_bytes(depth, turn) as u64);
        let (rows, activations, residual) =
            self.offer_tier(room, offer, kv.saturating_add(recurrent));
        Queued::Offered(self.credited(Cost {
            // The whole turn's K/V, not this chunk's: admitting the turn commits
            // the engine to feeding all of it, and a cost that priced only the
            // first chunk would admit a queue of turns whose tails cannot fit.
            kv,
            recurrent,
            residual,
            hole_credit: 0,
            narrow_credit: 0,
            activations,
            rows,
            // The turn decodes when this prefill finishes, which is what puts
            // the decode model's judgement in play beside the prefill one.
            decodes_after: true,
        }))
    }

    /// The band's next pending submission, priced as the turn it will become
    /// **plus the lift its projection puts on the device**
    /// ([`super::submit_admit`]) — the selected history brought hot and its
    /// index pages placed, ground the weight side loses exactly as the turn's
    /// own K/V. Its view is carved at admission, so it holds no store yet, and
    /// its tier is priced at the depth its projection will give it.
    fn peek_submit(&mut self, prio: DecodePriority) -> Option<Cost> {
        let band = band_index(prio);
        loop {
            while let Some(p) = self.sched.pending_submits.get(self.submit_cursor[band]) {
                if self.sched.decode_priority_or_high(p.parent()) == prio {
                    break;
                }
                self.submit_cursor[band] += 1;
            }
            let p = self.sched.pending_submits.get(self.submit_cursor[band])?;
            let (turn, depth, lift, parent) = (p.work.tokens.len(), p.depth, p.lift, p.parent());
            // Keyed by its parent until its view is carved: the parent's adapter
            // is the view's, and the index its tier is staged over is the one the
            // view forks.
            let offer = Candidate {
                seq: parent.0,
                band: prio,
                remaining: turn,
                depth,
                section: false,
            };
            // Walked past with no place in its group, as a queued turn is
            // (`peek_prefill`) — and behind a turn on its parent walked past,
            // whose history it forks.
            let placed = if self.walked.holds(parent.0, true) {
                None
            } else {
                self.room(offer)
            };
            match placed {
                Some(room) => return Some(self.price_submit(room, offer, lift)),
                None if self.closed => return None,
                None => {
                    self.walked.past(parent.0, true);
                    self.submit_cursor[band] += 1;
                }
            }
        }
    }

    /// The pending submission `offer` stands for, where it stands, priced with
    /// its projection's `lift`.
    fn price_submit(&mut self, room: (usize, Placed), offer: Candidate, lift: u64) -> Cost {
        let (turn, depth) = (offer.remaining, offer.depth);
        let kv = prefill_cost_bytes(turn, self.per_block)
            .saturating_add(lift)
            .saturating_add(self.sched.model.index_growth_bytes(depth, turn) as u64);
        let (rows, activations, residual) =
            self.offer_tier(room, offer, kv.saturating_add(self.recurrent));
        self.offering_submit = true;
        self.credited(Cost {
            kv,
            recurrent: self.recurrent,
            residual,
            hole_credit: 0,
            narrow_credit: 0,
            activations,
            rows,
            decodes_after: true,
        })
    }

    /// Admit the pending submission [`Self::peek_submit`] offered: put its
    /// projection on the device and begin it. A turn whose begin the allocators
    /// refuse stays queued as carved work, as an admitted case does.
    fn admit_submit(&mut self, prio: DecodePriority, cost: Cost) -> bool {
        let band = band_index(prio);
        let at = self.submit_cursor[band];
        let Some(pending) = self.sched.pending_submits.remove(at) else {
            return false;
        };
        for (i, c) in self.submit_cursor.iter_mut().enumerate() {
            if i != band && *c > at {
                *c -= 1;
            }
        }
        let before = residency_now(self.standing_tier);
        let stranded = interleave::stranded_hole_bytes();
        // Whether it was answered, and whether a turn of it is in flight now —
        // the one that joins the group its offer was priced in.
        let (begun, joined) = match self.sched.materialise_submit(pending) {
            Some(work) => match self.sched.begin_admitted(work) {
                Ok(()) => (true, true),
                Err(refused) => {
                    let (work, why) = *refused;
                    tracing::debug!(
                        target: "candle_conversation::scheduler::throttle",
                        "admitted submission could not begin — it waits as carved work: {why}",
                    );
                    // Ahead of every queued turn, so every band's cursor moves
                    // with what it pointed at.
                    self.sched.prefill_queue.push_front(work);
                    for c in &mut self.prefill_cursor {
                        *c += 1;
                    }
                    (false, false)
                }
            },
            // Its caller has the error, or it went to a turn group whose cases
            // are offered on their own.
            None => (true, false),
        };
        self.take_offered(cost.rows, joined);
        self.settle_holes(stranded, cost.hole_credit);
        self.narrow_spoken = self.narrow_spoken.saturating_add(cost.narrow_credit);
        let carved = before.saturating_sub(residency_now(self.standing_tier));
        self.committed = self
            .committed
            .saturating_add(cost.dislodged_bytes().saturating_sub(carved));
        begun
    }

    fn admit_prefill(&mut self, prio: DecodePriority, cost: Cost) -> bool {
        if self.offering_submit {
            return self.admit_submit(prio, cost);
        }
        let band = band_index(prio);
        let at = self.prefill_cursor[band];
        let Some(work) = self.sched.prefill_queue.remove(at) else {
            return false;
        };
        // **A turn-group case is carved here, the moment it is admitted** —
        // its writer chunks and its forked store are the ground this offer was
        // just priced for. A carve the allocators refuse puts the case back where
        // it stood and ends the band, as any refusal after the model's yes does:
        // the ground is genuinely gone, and the case waits for the next pass.
        let before = residency_now(self.standing_tier);
        let stranded = interleave::stranded_hole_bytes();
        if let Err(refused) = self.sched.begin_admitted(work) {
            let (work, why) = *refused;
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                "admitted case could not be carved its view — it waits: {why}",
            );
            self.sched.prefill_queue.insert(at, work);
            return false;
        }
        // In flight now, so the next offer's group carries it — see `room`.
        self.take_offered(cost.rows, true);
        self.settle_holes(stranded, cost.hole_credit);
        self.narrow_spoken = self.narrow_spoken.saturating_add(cost.narrow_credit);
        // **Nothing is purchased here, and that is not an omission.**
        //
        // A turn's K/V is claimed incrementally, chunk by chunk, by the admit
        // phase of each forward it rides (`admit_wave_kv`) — so the free pool
        // holds one wave's worth at a time, never a whole turn's, and that is
        // the steady state rather than a shortage. Demanding the turn's full
        // claim up front is therefore a test the engine can never pass: measured
        // on this daemon, an offer wanting 5,632 MiB against 576 MiB free, asking
        // the weight side for the difference and being conceded 0 — every pass,
        // so nothing was ever admitted and the queue only drained through the
        // keep-one-alive rule.
        //
        // The cost still does its job: it is what the rate model weighs, because
        // the residency this turn will dislodge as it runs is exactly the trade
        // being judged. What does not follow is that admission must hold that
        // ground before the first row runs.
        // Removing at the cursor shifts every later entry down by one, so the
        // *other* bands' cursors, if they sit past this index, now point one
        // place too far. Walk them back rather than leaving them to skip a
        // candidate each time another band takes one.
        for (i, c) in self.prefill_cursor.iter_mut().enumerate() {
            if i != band && *c > at {
                *c -= 1;
            }
        }
        // What the weight side loses to this admission, carried so the offers
        // behind it face the wave as it now stands — see [`Self::committed`].
        // Less what a carve has already taken: those regions are live now, so
        // the residency the next offer reads has already fallen by them.
        let carved = before.saturating_sub(residency_now(self.standing_tier));
        self.committed = self
            .committed
            .saturating_add(cost.dislodged_bytes().saturating_sub(carved));
        true
    }
}

impl Scheduler {
    /// Ground the KV side can claim without crossing into the weight zone,
    /// together with the zone's own position and range.
    ///
    /// Region-granular and **measured, not forecast**: the relief pass ahead of
    /// admission has already run, so a free region is one this process has
    /// claimed and not yet spent rather than one it hopes to recover.
    pub(super) fn admit_headroom(&self, standing_tier: usize) -> Headroom {
        let zone = residency_now(standing_tier);
        // The next forward's tier, beyond what the standing ground already holds
        // (`interleave::reserve_for` nets that out of `zone`).
        let tier_reserve = self
            .min_forward_tier_bytes()
            .saturating_sub(standing_tier as u64);
        let room = Headroom {
            // **Deliberately zero here, and netted below.** The free list is not
            // headroom *beside* the zone, it is the same ground seen twice:
            // `effective = extent + non-live regions - reserve`, so claiming a
            // free region moves it from non-live to live and drops the zone by
            // exactly its size. Any formula that adds the two lets admission
            // spend one region twice — measured as 95 slots standing open on
            // 1.6 GiB that did not exist.
            free_kv: 0,
            zone,
            zone_min: interleave::optimal_weight_bytes().unwrap_or(0),
            zone_max: interleave::achievable_weight_now().unwrap_or(u64::MAX),
            zone_min_prefill: self.expert_zone_min_bytes(),
            forward_reserve: tier_reserve,
        };
        Headroom {
            // **The spendable ground is one subtraction: how far residency
            // stands above its floor**, less a useful forward's tier.
            //
            // The tier is the span's third tenant and takes whatever K/V leaves
            // at the frontier, so admission must stop short of the weight floor
            // by that much. Reserve nothing and K/V claims all the way down: the
            // tier reaches zero, no forward can be planned at all, nothing
            // completes, and no ground comes back for it to recover with.
            //
            // Self-correcting, which is the property worth having: as slots
            // finish and their regions return, the zone rises and this room
            // reopens by the same amount.
            free_kv: zone.saturating_sub(room.floor().saturating_add(tier_reserve)),
            ..room
        }
    }

    /// The smallest weight zone the expert cache can serve a forward from —
    /// its pinned set, a worst-case working layer and one slot for the load
    /// that triggered an eviction (`minimum_resident_slots`), as the cache
    /// publishes it. Zero for a dense stack: no expert residency to defend.
    ///
    /// **The prefill floor, derived rather than set.** It was a quarter of the
    /// achievable residency plus a fixed 512 MiB margin — a fraction tuned on
    /// one ingest, the same on every card and every load, and on the 3090 with
    /// two dozen conversations' state resident a line the zone could not stand
    /// above. What a prefill must never cross is the point where the cache can
    /// no longer serve, and the cache knows that point exactly: under it every
    /// slot can be pinned and no promotion lands. Above it, spending weight for
    /// rows is the trade the rate model judges.
    pub(super) fn expert_zone_min_bytes(&self) -> u64 {
        self.model
            .expert_stats_shared()
            .and_then(|s| s.lock().ok().map(|s| s.zone_min_bytes as u64))
            .unwrap_or(0)
    }

    /// Tier bytes a forward worth running needs, held back from admission.
    ///
    /// **A useful forward's worth, not merely a placeable one.** Reserve nothing
    /// and the tier reaches zero — no forward at all, not a narrow one. Reserve
    /// one row and a forward can be placed but not filled: every wave carried
    /// one sequence while a third of the regions stood free. So it is the least
    /// advance the engine will hand a sequence, priced through the same planner
    /// that places the tier, which is what makes it follow the model's geometry
    /// instead of being a byte count to re-derive per card.
    pub(super) fn min_forward_tier_bytes(&self) -> u64 {
        // The least advance is a few chunks, far below the rows a launch
        // pre-stages for, so it stages nothing.
        self.model
            .wave_tier_bytes(
                PREFILL_MIN_ADVANCE,
                1,
                &[],
                &[],
                &[],
                &self.decode_stage(),
                self.session.activation_dtype(),
            )
            .unwrap_or(0)
    }

    /// Whether spending `bytes` more of the weight side's residency leaves it at
    /// or above the decode floor admission defends — the same line
    /// [`Self::admit_budget_terms`] gives the rate model, read now. What a
    /// mid-decode rebuild is judged by: it adds no rows, so the only question is
    /// whether the decodes it would slow can still be carried.
    pub(super) fn decode_floor_holds_after(&self, bytes: u64) -> bool {
        let standing = self.standing_ground_bytes();
        let room = self.admit_headroom(standing);
        let floor = room.floor().saturating_add(room.forward_reserve);
        residency_now(standing).saturating_sub(bytes) >= floor
    }

    /// The tier the wave already in flight reserves — the held creep group,
    /// its members' pre-stage at each one's own depth included.
    ///
    /// A fact this decision reads, not an output it feeds back into itself: the
    /// tier each *new* admission adds is charged separately, through
    /// `Cost::dislodged_bytes`, so between them the wave's tier is counted
    /// exactly once.
    pub(super) fn standing_tier_bytes(&self) -> usize {
        self.composed_tier(&self.compose(&self.standing_wave())) as usize
    }

    /// Ground the standing wave holds against the weight side: its tier, and
    /// the K/V its running decodes will claim before a sweep's windows are
    /// done ([`super::sweep_forecast`]), in whole regions as the claims take
    /// it. What every residency reading is net of.
    ///
    /// **The decodes' growth is committed ground, not headroom.** A sweep's
    /// tier is placed per window on the ground left then, and the decodes
    /// beside it keep claiming as they generate; admission that spends to the
    /// floor's edge leaves the later windows that much short. Measured on the
    /// 3090 scratch daemon (2026-10-18): floor-sized offers took residency to
    /// within 27 MiB of the prefill floor, and the sweep's last window found
    /// its 5.0 GB tier two regions short — ten decodes' claims over its 30 s.
    /// The group budget already withholds this forecast; admission now reads
    /// the same one.
    pub(super) fn standing_ground_bytes(&self) -> usize {
        self.standing_tier_bytes()
            .saturating_add(self.decode_growth_bytes())
            .saturating_add(self.glue_scratch_bytes())
            .saturating_add(self.held_claims_outstanding())
    }

    /// What the held cohort will still claim as it sweeps: all of its own
    /// claims before its first window — the residual, its stores and scratch
    /// are claimed there — and after it the K/V its remaining layers write.
    ///
    /// Its fit priced them against the ground at formation; a turn admitted
    /// mid-sweep claims from that same ground at once. Unreserved, one admitted
    /// above the floor took a 144 MiB scratch under an 8,192-row cohort fitted
    /// with 57 MiB to spare, and its last window was refused four regions short
    /// (WhiteWorld, c3ebac8bf, 2026-10-19).
    pub(super) fn held_claims_outstanding(&self) -> usize {
        let Some((own, kv)) = self.held_fit_claims else {
            return 0;
        };
        if self.wave_prefill_members.is_empty() {
            return 0;
        }
        let layers = self.session.num_layers().max(1);
        let cursor = self.wave_prefill_cursor.min(layers);
        if cursor == 0 {
            own as usize
        } else {
            (kv as usize).saturating_mul(layers - cursor) / layers
        }
    }

    /// The prefill scratch the running decodes' static runs may claim: a
    /// decode with a stencil plays its tool-call boilerplate as one prefill
    /// span (`inject_stencil_prefills`), and a span past the timeline claims
    /// its sequence's scratch at that step (`ensure_scratch`) — one slot per
    /// recurrent layer, at its share of the region. Each decode with a stencil
    /// that does not hold its scratch yet may take one, at any step, with no
    /// admission of its own: priced nowhere, a 95-token run of two decodes
    /// took 144 MiB inside a cohort fitted with 28 MiB to spare, and its next
    /// window was refused five regions short (WhiteWorld, a5e6cc3ec,
    /// 2026-10-19).
    pub(super) fn glue_scratch_bytes(&self) -> usize {
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished && s.stencil.as_ref().is_some_and(|d| !d.is_done()))
            .map(|(id, _)| self.model.prefill_scratch_bytes(id.0))
            .sum()
    }

    /// The K/V the running decodes will claim over one sweep, in whole regions
    /// — see [`Self::standing_ground_bytes`].
    pub(super) fn decode_growth_bytes(&self) -> usize {
        let growth = self
            .model
            .kv_bytes_per_row(self.session.activation_dtype())
            .map_or(0, |row| self.sweep_forecast.withheld_bytes(row));
        let region = candle_nn::kv_cache::REGION_BYTES;
        growth.div_ceil(region).saturating_mul(region)
    }

    /// The cohort the next forwards carry as a [`WaveDraft`]: the held creep
    /// group at each member's advance and depth, then **every prefill and
    /// section admitted that has not joined it yet**.
    ///
    /// Those are the next group's members as surely as the held ones — the
    /// group is formed from every prefill in flight — and an earlier pass
    /// charged their tier against the draft it composed. Starting the next pass
    /// from the held group alone forgets that charge: measured on the 3090
    /// scratch daemon (2026-10-18), five passes in two seconds each read
    /// ~9.3 GiB, admitted three turns down to ~4.6 GiB and stopped — sixteen
    /// turns whose cohort then formed at 8,192 rows on a 6.9 GB tier, and its
    /// sweep failed two to four regions short with every member lost.
    ///
    /// Each takes rows from what the ceiling has left, staged at its deepest
    /// chunk, exactly as its admission's offer was priced; past the ceiling
    /// it joins, adding a sequence and no rows.
    pub(super) fn standing_wave(&self) -> WaveDraft {
        let mut wave = WaveDraft::default();
        let mut held = Vec::with_capacity(self.wave_prefill_members.len());
        for m in &self.wave_prefill_members {
            let (WaveMember::Prefill { seq_id, advance } | WaveMember::Section { seq_id, advance }) =
                m;
            wave.held_rows += *advance;
            wave.held_advances.push(*advance);
            wave.held_depths
                .push(self.session.sequence_offset(*seq_id).unwrap_or(0));
            wave.held_seqs.push(*seq_id);
            held.push(*seq_id);
        }
        let prefills = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none())
            .map(|p| {
                (
                    p.work.sequence_id,
                    p.work.tokens.len().saturating_sub(p.offset),
                )
            });
        let sections = self
            .active_section_ingests
            .iter()
            .filter(|s| s.error.is_none())
            .map(|s| (s.sequence_id, s.tokens.len().saturating_sub(s.offset)));
        for (seq, remaining) in prefills.chain(sections) {
            if remaining > 0 && !held.contains(&seq.0) {
                let depth = self.session.sequence_offset(seq.0).unwrap_or(0);
                wave.joining.push((remaining, depth, seq.0));
            }
        }
        wave.width = self.rate_width.map(|w| w.rows);
        wave
    }

    /// `wave` composed the way the group forms: the held group at its fixed
    /// advances, then each joining member taking rows from what the ceiling
    /// has left, staged at its deepest chunk; past the ceiling it rides as a
    /// sequence with no rows.
    ///
    /// **The ceiling is the width the group will be composed at** — the group
    /// budget at the draft's width, the rule `prefill_group_budget_at` forms the
    /// group by — not the pass ceiling an offer is sized against. The group
    /// budget absorbs a total that passes the model's cap by no more than its
    /// slack, so a cohort can form wider than any ceiling an offer saw:
    /// measured on the 3090 scratch daemon (2026-10-18), a sweep at 8,066 rows
    /// whose last window's 7.26 GB tier was nine regions short. The stretch
    /// fires exactly when the model's cap binds, which is when ground is short.
    /// And a narrower width the pass closes at binds the other way: a member
    /// clipped to it is priced at the rows it will actually run.
    pub(super) fn compose(&self, wave: &WaveDraft) -> Composed {
        let total = wave
            .held_rows
            .saturating_add(wave.joining.iter().map(|&(r, _, _)| r).sum::<usize>());
        let ceiling = self.prefill_group_budget_at(total, self.composition_target_at(wave.width));
        let mut out = Composed {
            rows: wave.held_rows,
            seqs: wave.held_advances.len() + wave.joining.len(),
            advances: wave.held_advances.clone(),
            depths: wave.held_depths.clone(),
            stage_seqs: wave.held_seqs.clone(),
        };
        for &(remaining, depth, seq) in &wave.joining {
            let rows = remaining.min(ceiling.saturating_sub(out.rows));
            if rows > 0 {
                out.stage(seq, depth, remaining, rows);
            }
        }
        out
    }

    /// The tier `group` places as the forward it runs, each member staged at
    /// its deepest chunk.
    fn group_tier(&self, group: &NextGroup) -> u64 {
        let mut out = Composed {
            rows: 0,
            seqs: group.members.len(),
            advances: Vec::with_capacity(group.members.len()),
            depths: Vec::with_capacity(group.members.len()),
            stage_seqs: Vec::with_capacity(group.members.len()),
        };
        for p in &group.members {
            out.stage(p.member.seq_id(), p.depth, p.remaining, p.member.advance());
        }
        self.composed_tier(&out)
    }

    /// The tier a forward composed as `wave` places — the same planner the
    /// placement reads, with the running decodes the step carries beside it.
    pub(super) fn composed_tier(&self, wave: &Composed) -> u64 {
        self.model
            .wave_tier_bytes(
                wave.rows,
                wave.seqs.max(1),
                &wave.advances,
                &wave.depths,
                &wave.stage_seqs,
                &self.decode_stage(),
                self.session.activation_dtype(),
            )
            .unwrap_or(0)
    }

    /// The wave's opening terms, from the same settled reading as the headroom.
    pub(super) fn admit_budget_terms(&self, room: &Headroom, chunk_rows: usize) -> Budget {
        Budget {
            // One currency for the rate, the dislodge and the floor alike — see
            // [`residency_now`]. Reading any of the three in a different one is
            // what lets the model be shown the same ground twice.
            resident: room.zone,
            // The same line `headroom` nets out of the spendable ground: the
            // hold, the eviction margin that keeps the cache evictable, and a
            // useful forward's tier. The model enforces it as a hard refusal.
            prefill_floor: room.prefill_floor().saturating_add(room.forward_reserve),
            decode_floor: room.floor().saturating_add(room.forward_reserve),
            max_rows: chunk_rows,
            max_decodes: Self::MAX_DECODE_WIDTH,
        }
    }

    /// The recurrent store **one** new sequence needs, or zero on a stack that
    /// carries none.
    ///
    /// **Priced from the model's geometry, never from what is resident.** Every
    /// store the model builds has the same shape, so it can say what one costs
    /// without a store existing to measure — which matters, because admission
    /// prices a claim precisely before it is made.
    ///
    /// This was a mean over the live stores, and a mean is the one thing it
    /// could not be: `recurrent_reserved_bytes` sums every store the process
    /// holds, parked conversations included, while the only count available to
    /// divide by is what the scheduler has *in flight* — and admission runs
    /// between forwards, where that is 0 or 1. The two range over different
    /// populations, so the quotient is not a per-sequence figure but roughly
    /// the whole engine's carried state, and it climbs as conversations go
    /// idle. It therefore peaked exactly when admission should have been
    /// cheapest. Measured on a 72 GB card: a 41-row turn priced at 4,450 MiB
    /// and refused as throughput-worse on fifteen consecutive passes, seven
    /// turns queued behind it and 20 GiB standing free above the floor.
    ///
    /// Being a constant, it is also immune to the per-offer decay that made the
    /// old figure read 160, 80, 53, 40 MiB down a single pass as newly admitted
    /// prefills joined the denominator before allocating anything. It is still
    /// read once at [`AdmitPass::open`], because nothing in a pass can change
    /// it and re-reading would only take the lock again.
    pub(super) fn recurrent_cost(&self) -> u64 {
        self.model.recurrent_store_bytes() as u64
    }

    /// Fold one completed prefill forward into the planner.
    ///
    /// Narrow forwards are refused by the model itself
    /// ([`rate::WaveRate::min_learn_rows`]): a wave too narrow to route across
    /// the whole layer copies a fraction of the experts the observation divides
    /// by, and one 26-row forward read 119 GB/s on a 25 GB/s link. Passing them
    /// in anyway is correct — the refusal belongs with the model that knows the
    /// routing, not with the caller.
    pub(super) fn observe_prefill_forward(&mut self, rows: usize, pairs: f64, us: u64) {
        if rows == 0 || us == 0 {
            return;
        }
        // Counted even when the model refuses to learn from it: the loop's overhead is
        // amortised across forwards that *ran*, not across forwards wide enough to teach
        // the copy rate.
        self.wave_forwards += 1;
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_prefill_attending(rows, pairs, resident, us as f64 / 1e6);
        }
    }

    /// A co-batched decode forward, for the decode fit: its creep's rows
    /// through layers are the fit's third term.
    pub(super) fn observe_decode_beside(
        &mut self,
        decodes: usize,
        us: u64,
        creep_layer_rows: usize,
        creep_layer_pairs: f64,
        positions: f64,
        rows: usize,
    ) {
        if decodes == 0 || us == 0 {
            return;
        }
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_decode_beside(
                decodes,
                resident,
                us as f64 / 1e6,
                creep_layer_rows,
                creep_layer_pairs,
                positions,
                rows,
            );
        }
    }

    /// Fold one completed decode forward into the planner.
    ///
    /// A bus-bound step says nothing about the layer time, and the model drops
    /// it; only a compute-bound one teaches anything. `rows` is every row the
    /// step scored — a drafted decode's whole verify block — so the misses the
    /// fit learns are the ones the step routed.
    pub(super) fn observe_decode_forward(
        &mut self,
        decodes: usize,
        us: u64,
        positions: f64,
        rows: usize,
    ) {
        if decodes == 0 || us == 0 {
            return;
        }
        // Same reason as the prefill funnel: the loop's overhead is shared by every
        // forward a wave ran, of either kind.
        self.wave_forwards += 1;
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_decode_at(decodes, rows, resident, us as f64 / 1e6, positions);
        }
    }

    /// Fold the expert cache's own hit and miss counters into the planner's hit
    /// coefficient.
    ///
    /// **The one estimate that cannot be guessed.** The LRU zone and the Markov
    /// predictor together are worth far more than the resident fraction alone,
    /// and by how much depends on the workload's routing: measured at 37%
    /// resident the hit rate was 0.65, a coefficient of ~1.73 against a seed of
    /// 0.7 — a decode's copy priced two and a half times too dear, which the
    /// wave pays for by refusing decodes that would have fitted.
    ///
    /// **Read unsettled, once per decode forward.** Settling the expert
    /// pipeline (`expert_stats`) waits for the device to reach the forward's
    /// last MoE layer and for the pipeline thread to serve every layer queued
    /// behind it — on Qwen3.8-Flash-Next single-session decode (RTX PRO 5000)
    /// reading unsettled took a 4K step from 23.6 to 19.9 ms. The interval it
    /// folds needs no exact boundary: a tally that trails by the layers still
    /// queued counts them in the next interval instead.
    pub(super) fn observe_expert_hit_rate(&mut self) {
        let Some((total_hits, total_misses)) = self.model.expert_hit_counts() else {
            return;
        };
        // **The interval, not the lifetime.** Nothing in the daemon calls
        // `reset_expert_stats` — only the bench harness does — so these are
        // cumulative totals for the life of the process. Folding a lifetime
        // average into an exponential estimate converges it to that average
        // within a few dozen forwards and then freezes it, after which the
        // coefficient can no longer answer the question it exists for: how this
        // residency converts into hits. The sample count goes on rising, so the
        // telemetry reads healthy while the estimate has stopped moving.
        let (hits, misses) = (
            total_hits.saturating_sub(self.expert_hits_seen),
            total_misses.saturating_sub(self.expert_misses_seen),
        );
        self.expert_hits_seen = total_hits;
        self.expert_misses_seen = total_misses;
        let routed = hits + misses;
        if routed == 0 {
            return;
        }
        // Smoothed by half an expert each way: an interval that found every
        // expert says the hit is near certain, not that it is certain at every
        // residency, which an exponent of zero would claim.
        let hit = (hits as f64 + 0.5) / (routed as f64 + 1.0);
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_hit_rate(hit, resident);
        }
    }

    /// The layer priority a slot decodes at, defaulting to the protective
    /// `High` when its target is not resolvable yet.
    pub(super) fn decode_priority_or_high(&self, id: SequenceId) -> DecodePriority {
        self.decode_layer_priority(id)
            .unwrap_or(DecodePriority::High)
    }

    /// Whether a slot in flight — prefilling, ingesting a section, or decoding —
    /// belongs to a band below `prio`. A section ingest is at its own band.
    pub(super) fn lower_band_active(&self, prio: DecodePriority) -> bool {
        let below = |p: DecodePriority| rank(p) < rank(prio);
        let prefills = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none())
            .any(|p| below(self.decode_priority_or_high(p.work.sequence_id)));
        let sections = self
            .active_section_ingests
            .iter()
            .any(|s| s.error.is_none() && below(s.band));
        let decodes = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .any(|(&id, _)| below(self.decode_priority_or_high(id)));
        prefills || sections || decodes
    }

    /// Running decodes of `prio`'s band or above — the ones a `prio` prompt
    /// does not outrank, whose steps the residency it spends still slows.
    pub(super) fn decodes_at_or_above(&self, prio: DecodePriority) -> usize {
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .filter(|(&id, _)| rank(self.decode_priority_or_high(id)) >= rank(prio))
            .count()
    }
}

/// Rows an offer whose `share` of its group is that many is offered, or
/// `None` when the group has none for it.
///
/// **A full forward takes no more turns.** With no share the offer is no
/// member: the group it would join is full, or leaves it out — a band below
/// the group's, another adapter. Its rows are filled in order
/// (`gang::gang_shares`), so a turn joining a full forward advances by
/// nothing until the ones ahead finish — yet it claims its store and scratch
/// at admission, and holds them while it waits. Joined while every member could
/// still advance a chunk, BlackWorld's code_read ingest (2026-10-27) held 13
/// prefills' recurrent stores at once against one forward's rows. It is offered
/// again at the next pass, once rows have come free.
///
/// **A closed forward has no rows left, whatever its width**, so a member
/// comes back as a join: it rides the group's forwards at its share without
/// widening them. Closing ends the rows at what the pass admitted, and the
/// cohort composes at no less than a chunk (`Scheduler::set_rate_width`), so a
/// pass closed below a chunk is still a chunk wide. Read as room, that chunk
/// re-offered the turn the rate had just refused with the same rows, the rate
/// refused it again, and the pass never ended: on BlackWorld (d919bd410,
/// 2026-10-20) one 4-row `Normal` prompt was offered and refused every 1.4 ms
/// for four hours, and no wave ran to step the seventeen decodes the refusal
/// was protecting.
fn room_left(share: Option<usize>, closed: bool) -> Option<usize> {
    share.map(|rows| if closed { 0 } else { rows })
}

#[cfg(test)]
mod tests {
    use super::{deepest_chunk_offset, room_left, CHUNK_SIZE};

    #[test]
    fn a_closed_forward_offers_only_joins_even_below_a_chunk() {
        // Nothing standing, the rate refused the first rows offered: the pass
        // closes at width zero and composes at one chunk.
        assert_eq!(room_left(Some(CHUNK_SIZE), false), Some(CHUNK_SIZE));
        assert_eq!(
            room_left(Some(CHUNK_SIZE), true),
            Some(0),
            "a closed pass re-offers the turn as a join, not with the rows it was refused"
        );
    }

    /// **A turn its group leaves out is not offered**, full or not: filled in
    /// order, the forward would carry nothing of it while it held its store.
    #[test]
    fn a_turn_its_group_leaves_out_is_not_offered() {
        assert_eq!(room_left(None, false), None);
        assert_eq!(room_left(None, true), None);
    }

    #[test]
    fn the_deepest_chunk_ends_at_the_turns_last_token() {
        // A 128,000-token turn fed 8,192 rows at a time from a fresh sequence:
        // its deepest full-width chunk starts at 119,808 and ends at 128,000.
        assert_eq!(deepest_chunk_offset(0, 128_000, 8_192), 119_808);
        // The same turn appended behind 4,096 standing tokens.
        assert_eq!(deepest_chunk_offset(4_096, 128_000, 8_192), 123_904);
    }

    #[test]
    fn a_turn_no_wider_than_a_chunk_is_its_own_deepest() {
        assert_eq!(deepest_chunk_offset(0, 4_096, 8_192), 0);
        assert_eq!(deepest_chunk_offset(300, 8_192, 8_192), 300);
    }
}
