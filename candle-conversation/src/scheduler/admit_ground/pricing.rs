//! The cost of an admission offer: the tier it adds, the residency it dislodges
//! and the rows it earns.
//!
//! An offer is priced at its deepest chunk, because the terms that follow depth
//! grow as a turn advances, and at the width its own admission leaves, so the
//! cohort is composed the way the group will form. A turn too large to fit whole
//! is offered the widest rows that still clear the prefill floor rather than
//! refused on every pass. Ground the pass has already committed is netted off
//! each later offer, so several offers are not each measured against the same
//! unspent residency.

use super::*;

impl<'a> AdmitPass<'a> {
    pub(in super::super) fn new(sched: &'a mut Scheduler) -> Self {
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
            rows_taken: 0,
            chunk_rows,
            standing_tier,
            standing_rows: composed.rows,
            wave_rows: composed.rows,
            wave,
            wave_tier,
            offered: None,
            narrowed: false,
            holes_spoken: 0,
            narrowable,
            narrow_spoken: 0,
            prefill_cursor: [0; 4],
            submit_cursor: [0; 4],
            offering_submit: false,
            widened: false,
            closed: false,
            unoffered: Unoffered::default(),
        }
    }

    /// End the pass: a pass that admitted rows and never stopped on the rate
    /// found every row it offered paying, so the width an earlier pass chose
    /// no longer holds and the cohort's forwards may widen to the budget. A
    /// pass that closed its rows has already recorded its own width
    /// (`close_rows`); one that admitted no rows judged nothing. Answers why
    /// waiting work went unoffered, for the pass's log line.
    pub(in super::super) fn conclude(self) -> Unoffered {
        if self.widened && !self.closed {
            self.sched.clear_rate_width();
        }
        self.unoffered
    }

    /// What [`Self::wave`] gains by taking `member` (its remaining rows and
    /// depth) with its forwards composed at `width`: the tier, and the rows
    /// the composed forward widens by. Zero when the model cannot price one.
    ///
    /// A member that adds no row is not free: the span tables hold an entry
    /// per span, and pricing it at zero lets a forward widen to the sequence
    /// cap on a tier admission never charged.
    pub(super) fn added(
        &self,
        member: (usize, usize, usize),
        width: Option<usize>,
    ) -> (u64, usize) {
        let mut with = self.wave.clone();
        with.joining.push(member);
        with.width = width;
        let composed = self.sched.compose(&with);
        (
            self.sched
                .composed_tier(&composed)
                .saturating_sub(self.wave_tier),
            composed.rows.saturating_sub(self.wave_rows),
        )
    }

    /// An offer of `turn` tokens at `depth` whose claim (its K/V and store) is
    /// `claim` bytes, given `room` rows free in the forward: the rows it earns
    /// there, the tier it adds and the creep residual it owes. Records the
    /// member, for the admit that follows.
    ///
    /// Staged at its **deepest** chunk, not its first: the terms that follow
    /// depth — the pre-stage up to its key-chunk bound, and on a
    /// sparse-selection model the scoring over every candidate block the
    /// deepest row can see — grow as the turn advances, so a 128K turn's last
    /// chunk costs a tier its first does not. A full-width chunk ending at the
    /// turn's last token bounds every chunk the turn will run.
    ///
    /// The residual is sized by the cohort's composed rows, so it is owed for
    /// the rows the offer widens the composition by — a join the group
    /// absorbs owes it as surely as a turn with rows in this forward.
    ///
    /// **Priced at the width its own admission leaves.** An offer with rows
    /// widens the forward, and a pass that widens clears the rate model's
    /// width (`conclude`), so the group then composes at the deployment's
    /// target; a join leaves the width as it stands.
    ///
    /// **Sized to the floor when it cannot fit whole.** A turn whose rows would
    /// take residency under the prefill floor is offered the widest rows that
    /// still clear it — never fewer than the least advance — and the pass
    /// closes at that width when it is admitted (`take_offered`), so the group
    /// composes there and the turn advances by that much a forward. Offered
    /// whole, it is refused on every pass: measured on the 3090 scratch daemon
    /// (2026-10-18), a 4,325-row `High` tool-result prefill priced at a
    /// 4,992 MiB tier against 954 MiB of room, refused for minutes while the
    /// person waited, 22 turns cut at the client's deadline.
    pub(super) fn offer_tier(
        &mut self,
        room: usize,
        turn: usize,
        depth: usize,
        seq: usize,
        claim: u64,
    ) -> (usize, u64, u64) {
        // Charged to the window's admission split as `price`: the part of a
        // fill that composes and prices the cohort for each offer.
        let t = Instant::now();
        let priced = self.priced_tier(room, turn, depth, seq, claim);
        self.sched
            .wave_stats
            .admission_split
            .part(AdmitPart::Price, t.elapsed());
        priced
    }

    /// [`Self::offer_tier`]'s pricing.
    pub(super) fn priced_tier(
        &mut self,
        room: usize,
        turn: usize,
        depth: usize,
        seq: usize,
        claim: u64,
    ) -> (usize, u64, u64) {
        let member = (turn, depth, seq);
        self.offered = Some(member);
        self.narrowed = false;
        let rows = turn.min(room);
        if rows == 0 {
            let (tier, widened) = self.added(member, self.wave.width);
            return (0, tier, self.residual_for(widened));
        }
        let (tier, widened) = self.added(member, None);
        let residual = self.residual_for(widened);
        let spendable = self
            .resident_weights()
            .saturating_sub(self.budget.prefill_floor);
        if self.dislodges(claim, residual, tier) <= spendable {
            return (rows, tier, residual);
        }
        // Chunk-aligned rows, bisected: the tier and the residual both grow
        // with the rows, so the rows that fit are a prefix.
        let least = PREFILL_MIN_ADVANCE.min(rows).div_ceil(CHUNK_SIZE);
        let (mut lo, mut hi) = (least, rows.div_ceil(CHUNK_SIZE) - 1);
        let mut fit = None;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let r = (mid * CHUNK_SIZE).min(rows);
            let width = self.standing_rows + self.rows_taken + r;
            let (tier, widened) = self.added(member, Some(width));
            let residual = self.residual_for(widened);
            if self.dislodges(claim, residual, tier) <= spendable {
                fit = Some((r, tier, residual));
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
            None => (rows, tier, residual),
        }
    }

    /// What an offer claiming `claim` with this residual and tier dislodges,
    /// its hole credit taken — the figure the floor is judged on.
    pub(super) fn dislodges(&self, claim: u64, residual: u64, tier: u64) -> u64 {
        self.credited(Cost {
            kv: claim,
            residual,
            activations: tier,
            ..Default::default()
        })
        .dislodged_bytes()
    }

    /// The offer [`Self::offer_tier`] priced was admitted: it is a member of
    /// the cohort the next offer is priced against, at the width its
    /// admission leaves — closed where it was sized to the floor, cleared
    /// where it widened the forward.
    pub(super) fn take_offered(&mut self, rows: usize) {
        if let Some(member) = self.offered.take() {
            self.wave.joining.push(member);
        }
        if self.narrowed {
            self.narrowed = false;
            self.close_rows();
            return;
        }
        if rows > 0 && !self.closed {
            self.wave.width = None;
        }
        self.recompose();
    }

    /// Re-read [`Self::wave`]'s composed tier and rows after it changed.
    pub(super) fn recompose(&mut self) {
        let composed = self.sched.compose(&self.wave);
        self.wave_tier = self.sched.composed_tier(&composed);
        self.wave_rows = composed.rows;
    }

    /// `cost` with the part of its claim that lands in holes credited: the
    /// holes stranded below the frontier now, less those this pass has
    /// already promised to claims still to land.
    pub(super) fn credited(&self, cost: Cost) -> Cost {
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
    pub(super) fn held_narrowable(sched: &Scheduler, wave: &WaveDraft) -> u64 {
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
    pub(super) fn settle_holes(&mut self, stranded_before: u64, credit: u64) {
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
    pub(super) fn residual_for(&self, rows: usize) -> u64 {
        self.sched
            .model
            .creep_residual_bytes(&self.sched.session, rows) as u64
    }
}
