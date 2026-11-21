//! The residency reading and the pass state that admission judges each offer
//! against.
//!
//! Residency is read in the effective zone's currency, derived from the span
//! identity against the live region count, because the two obvious readings
//! each wedged the engine: the cache's occupancy sits below its capacities on a
//! warming cache, and the zone's extent does not move until the boundary does.
//! Each offer is judged against the wave with the rows it already carries
//! charged to it, and a width backstop bounds the whole pass rather than only
//! its entry. A waiting prompt counts once for each sweep it has waited, so a
//! prompt that has waited longer weighs more on the trade against decode.

use super::*;

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
pub(super) fn residency_now(standing_tier: usize) -> u64 {
    interleave::effective_weight_zone_bytes(standing_tier).unwrap_or(u64::MAX)
}

/// A priority's slot in the per-band cursors. Four bands, in the order
/// `admit::order::BANDS` offers them.
pub(super) fn band_index(prio: DecodePriority) -> usize {
    match prio {
        DecodePriority::Urgent => 0,
        DecodePriority::High => 1,
        DecodePriority::Normal => 2,
        DecodePriority::Low => 3,
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
        if self.sched.running_members() >= MAX_PREFILL_WIDTH
            && !(kind == Kind::Prefill && self.sched.lower_band_active(prio))
        {
            self.unoffered.width |= !self.sched.prefill_queue.is_empty()
                || !self.sched.pending_submits.is_empty()
                || !self.sched.section_queue.is_empty();
            return None;
        }
        match kind {
            Kind::Prefill => self.peek_prefill(prio),
            Kind::Section => self.peek_section(prio),
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
        let width = self.standing_rows.saturating_add(self.rows_taken);
        self.chunk_rows = width.max(CHUNK_SIZE);
        self.sched.set_rate_width(width);
        self.closed = true;
        self.wave.width = Some(width);
        self.recompose();
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

    fn charge(&mut self, part: FillPart, took: Duration) {
        let part = match part {
            FillPart::Peek => AdmitPart::Peek,
            FillPart::Residency => AdmitPart::Residency,
            FillPart::Judge => AdmitPart::Judge,
        };
        self.sched.wave_stats.admission_split.part(part, took);
    }

    fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
        // Charged to the window's admission split as `admit`: beginning what
        // the model said yes to — carving its view and store — and recomposing
        // the cohort the next offer is priced against.
        let t = Instant::now();
        let admitted = match kind {
            Kind::Prefill => self.admit_prefill(prio, cost),
            Kind::Section => self.admit_section(prio, cost),
            Kind::Decode => false,
        };
        self.sched
            .wave_stats
            .admission_split
            .part(AdmitPart::Admit, t.elapsed());
        admitted
    }
}
