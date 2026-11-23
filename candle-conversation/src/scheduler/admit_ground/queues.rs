//! Which queued work each priority band is offered next, and how much of the
//! wave's room that offer may use.
//!
//! A band is offered its own priority's next candidate, and the scan walks past
//! work that belongs to other bands, so a dialogue turn queued behind background
//! work is not served only by its FIFO position. Each offer takes the rows the
//! forward still has free, a store already held is charged once rather than
//! twice, and a full forward takes no more turns. Sections and pending
//! submissions are priced the same way prefills are, and an admitted item leaves
//! the queue at once, with the other bands' cursors walked back over it.

use super::super::creep_slice::short_piece;
use super::*;

impl AdmitPass<'_> {
    /// The carried state a turn of `turn` tokens on `seq` claims: a store when
    /// it holds none (`holds_store` false), and the prefill scratch a span
    /// past the short span claims on top.
    ///
    /// **A turn that rides in short pieces claims no scratch** (`short_piece`):
    /// each piece is written into the recurrent timeline. Charged it anyway, a
    /// 7-row turn on Flash-Next was priced its ~66 MiB store plus a 144 MiB
    /// scratch it never took, and a streaming card's store rule weighed its
    /// rows against three times the state it holds.
    pub(super) fn recurrent_charge(&self, seq: usize, turn: usize, holds_store: bool) -> u64 {
        let short = short_piece(self.sched.model.short_span_rows(), turn).is_some();
        match (holds_store, short) {
            (true, true) => 0,
            (true, false) => self.sched.model.prefill_scratch_bytes(seq) as u64,
            (false, true) => self
                .recurrent
                .saturating_sub(self.sched.model.sequence_scratch_bytes() as u64),
            (false, false) => self.recurrent,
        }
    }

    /// Rows of this wave's forward still free for an offer — see `peek_prefill`
    /// — or `None` when the forward is full and one more member could not
    /// advance a chunk beside the rest.
    pub(super) fn rows_left(&self) -> Option<usize> {
        room_left(
            self.chunk_rows,
            self.standing_rows.saturating_add(self.rows_taken),
            self.closed,
            self.sched.prefill_width() + self.sched.section_ingest_width(),
        )
    }

    /// [`Self::rows_left`] for a prompt of `turn` tokens on `seq` at `prio`, and
    /// whether it joins a full forward because it runs ahead of the members
    /// there (`Cost::ahead`): a live prompt the formation would give rows
    /// though the forward's rows are spent ([`Scheduler::runs_ahead`],
    /// `docs/continuous_fair_waves.md` §7). Below the live bands a full forward
    /// takes no more turns. A prompt offered as such a join was not left
    /// unoffered, so only one turned away is recorded ([`Unoffered::rows`]).
    pub(super) fn prompt_room(
        &mut self,
        prio: DecodePriority,
        turn: usize,
        seq: usize,
    ) -> Option<(usize, bool)> {
        if let Some(room) = self.rows_left() {
            return Some((room, false));
        }
        // `Normal` joins too: a full forward of `Low` ingest held a console's
        // description for minutes it could have shared. Only `Low` waits for rows.
        if prio != DecodePriority::Low && self.sched.runs_ahead(prio, turn, seq) {
            return Some((0, true));
        }
        self.unoffered.rows = true;
        None
    }

    /// [`Self::rows_left`] for a section of `turn` tokens at `prio`, and whether it
    /// joins a full forward (`Cost::ahead`). A section with a turn blocked on
    /// its seal joins as a prompt does ([`Scheduler::section_runs_ahead`],
    /// [`joins_a_full_forward`]); bulk ingest takes no more.
    pub(super) fn section_room(
        &mut self,
        prio: DecodePriority,
        turn: usize,
    ) -> Option<(usize, bool)> {
        if let Some(room) = self.rows_left() {
            return Some((room, false));
        }
        if joins_a_full_forward(prio, turn) && self.sched.section_runs_ahead(prio, turn) {
            return Some((0, true));
        }
        self.unoffered.rows = true;
        None
    }

    /// Where the queue's oldest section of band `prio` stands.
    pub(super) fn section_at(&self, prio: DecodePriority) -> Option<usize> {
        self.sched.section_queue.iter().position(|s| s.band == prio)
    }

    /// The band's next candidate: its oldest queued section, priced as a
    /// prefill is — its whole K/V, a recurrent store on a hybrid, and its rows
    /// in the forward's tier. It never decodes. Strictly FIFO within the band
    /// and never paused.
    pub(super) fn peek_section(&mut self, prio: DecodePriority) -> Option<Cost> {
        let s = &self.sched.section_queue[self.section_at(prio)?];
        let (turn, seq) = (s.tokens.len(), s.sequence_id);
        // A reused slot already holds a store, whose regions the residency
        // already counts: its restart zeroes that store where it stands.
        let recurrent =
            self.recurrent_charge(seq.0, turn, self.sched.model.has_recurrent_state(seq.0));
        let (room, ahead) = self.section_room(prio, turn)?;
        let depth = self.sched.session.sequence_offset(seq.0).unwrap_or(0);
        let kv = prefill_cost_bytes(turn, self.per_block)
            .saturating_add(self.sched.model.index_growth_bytes(depth, turn) as u64);
        let (rows, activations, residual) =
            self.offer_tier(room, turn, depth, seq.0, kv.saturating_add(recurrent));
        Some(self.credited(Cost {
            kv,
            recurrent,
            residual,
            hole_credit: 0,
            narrow_credit: 0,
            activations,
            rows,
            decodes_after: false,
            ahead,
        }))
    }

    /// Admit the band's oldest section, priced at `cost`, and start it — its
    /// carried state claimed now, as a turn's is (`begin_admitted`). A claim the
    /// allocators refuse puts it back where it stood and ends the band.
    pub(super) fn admit_section(&mut self, prio: DecodePriority, cost: Cost) -> bool {
        let Some(at) = self.section_at(prio) else {
            return false;
        };
        let Some(section) = self.sched.section_queue.remove(at) else {
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
            self.sched.section_queue.insert(at, section);
            return false;
        }
        self.rows_taken = self.rows_taken.saturating_add(cost.rows);
        self.take_offered(cost.rows);
        self.settle_holes(stranded, cost.hole_credit);
        self.narrow_spoken = self.narrow_spoken.saturating_add(cost.narrow_credit);
        self.widened |= cost.rows > 0;
        // See `admit_prefill`: what the section dislodges as it runs is carried
        // so the offers behind it face the wave as it now stands, less what the
        // claim has already taken.
        let claimed = before.saturating_sub(residency_now(self.standing_tier));
        self.committed = self
            .committed
            .saturating_add(cost.dislodged_bytes().saturating_sub(claimed));
        true
    }

    pub(super) fn peek_prefill(&mut self, prio: DecodePriority) -> Option<Cost> {
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
        while let Some(w) = self.sched.prefill_queue.get(self.prefill_cursor[band]) {
            if self.sched.decode_priority_or_high(w.sequence_id) == prio {
                break;
            }
            self.prefill_cursor[band] += 1;
        }
        // The band's queued work first; then the submissions behind it.
        let Some(w) = self.sched.prefill_queue.get(self.prefill_cursor[band]) else {
            return self.peek_submit(prio);
        };
        self.offering_submit = false;
        // **Offer what is LEFT of the wave's rows, not the whole cap.**
        //
        // Clipping to `chunk_rows` made every offer the size of the entire budget,
        // so the first turn long enough to fill it took the wave alone and the next
        // one was refused `Cap` for exceeding a budget it was never shown. Measured
        // on this daemon at `--max-depth 3`: a 2,646-row head admitted, the next
        // turn offering its full 3,000 against a 4,880 cap, refused — the wave
        // running 2,646 of 4,880 with sequences queued and every wave
        // `seqs max=1`, at a fraction of the batched rate.
        //
        // Offered the remainder, a turn takes the room that is actually free and
        // the wave fills with as many turns as it takes. A turn clipped short is
        // not turned away, it advances by that much and is offered again next wave
        // — which is how a prefill longer than one forward already progresses.
        //
        // Standing rows count: they are in the forward this fill is composing, and
        // the rate model's own cap check counts them, so a clip that ignored them
        // would offer ground that is already spoken for.
        //
        // **A full forward takes no more turns** ([`room_left`]): its rows are
        // filled in order (`gang::gang_shares`), so a turn past them would
        // advance by nothing while holding its store. Every member in flight is
        // counted, sections and paused prefills included. See [`Self::rows_left`].
        let (turn, seq) = (w.tokens.len(), w.sequence_id);
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
        let recurrent = self.recurrent_charge(seq.0, turn, holds_store);
        let (room, ahead) = self.prompt_room(prio, turn, seq.0)?;
        let depth = self.sched.session.sequence_offset(seq.0).unwrap_or(0);
        let kv = prefill_cost_bytes(turn, self.per_block)
            .saturating_add(self.sched.model.index_growth_bytes(depth, turn) as u64);
        let (rows, activations, residual) =
            self.offer_tier(room, turn, depth, seq.0, kv.saturating_add(recurrent));
        Some(self.credited(Cost {
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
            ahead,
        }))
    }

    /// The band's next pending submission, priced as the turn it will become
    /// **plus the lift its projection puts on the device**
    /// ([`super::submit_admit`]) — the selected history brought hot and its
    /// index pages placed, ground the weight side loses exactly as the turn's
    /// own K/V. Its view is carved at admission, so it holds no store yet, and
    /// its tier is priced at the depth its projection will give it.
    pub(super) fn peek_submit(&mut self, prio: DecodePriority) -> Option<Cost> {
        let band = band_index(prio);
        while let Some(p) = self.sched.pending_submits.get(self.submit_cursor[band]) {
            if self.sched.decode_priority_or_high(p.parent()) == prio {
                break;
            }
            self.submit_cursor[band] += 1;
        }
        let p = self.sched.pending_submits.get(self.submit_cursor[band])?;
        let (turn, depth, lift, parent) = (p.work.tokens.len(), p.depth, p.lift, p.parent());
        let (room, ahead) = self.prompt_room(prio, turn, parent.0)?;
        let kv = prefill_cost_bytes(turn, self.per_block)
            .saturating_add(lift)
            .saturating_add(self.sched.model.index_growth_bytes(depth, turn) as u64);
        // Its view is carved at admission, so it holds no store yet.
        let recurrent = self.recurrent_charge(parent.0, turn, false);
        let (rows, activations, residual) =
            self.offer_tier(room, turn, depth, parent.0, kv.saturating_add(recurrent));
        self.offering_submit = true;
        Some(self.credited(Cost {
            kv,
            recurrent,
            residual,
            hole_credit: 0,
            narrow_credit: 0,
            activations,
            rows,
            decodes_after: true,
            ahead,
        }))
    }

    /// Admit the pending submission [`Self::peek_submit`] offered: put its
    /// projection on the device and begin it. A turn whose begin the allocators
    /// refuse stays queued as carved work, as an admitted case does.
    pub(super) fn admit_submit(&mut self, prio: DecodePriority, cost: Cost) -> bool {
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
        let t_begin = Instant::now();
        let materialised = self.sched.materialise_submit(pending);
        let started = materialised.map(|work| self.sched.begin_admitted(work));
        self.sched
            .wave_stats
            .admission_split
            .part(AdmitPart::Begin, t_begin.elapsed());
        let begun = match started {
            Some(Ok(())) => true,
            Some(Err(refused)) => {
                let (work, why) = *refused;
                tracing::debug!(
                    target: "candle_conversation::scheduler::throttle",
                    "admitted submission could not begin — it waits as carved work: {why}",
                );
                self.sched.prefill_queue.push_front(work);
                false
            }
            // Its caller has the error, or it went to a turn group whose cases
            // are offered on their own.
            None => true,
        };
        self.rows_taken = self.rows_taken.saturating_add(cost.rows);
        self.take_offered(cost.rows);
        self.settle_holes(stranded, cost.hole_credit);
        self.narrow_spoken = self.narrow_spoken.saturating_add(cost.narrow_credit);
        self.widened |= cost.rows > 0;
        let carved = before.saturating_sub(residency_now(self.standing_tier));
        self.committed = self
            .committed
            .saturating_add(cost.dislodged_bytes().saturating_sub(carved));
        begun
    }

    pub(super) fn admit_prefill(&mut self, prio: DecodePriority, cost: Cost) -> bool {
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
        let t_begin = Instant::now();
        let begun = self.sched.begin_admitted(work);
        self.sched
            .wave_stats
            .admission_split
            .part(AdmitPart::Begin, t_begin.elapsed());
        if let Err(refused) = begun {
            let (work, why) = *refused;
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                "admitted case could not be carved its view — it waits: {why}",
            );
            self.sched.prefill_queue.insert(at, work);
            return false;
        }
        // Spend the rows this offer took, so the next offer in this pass is sized
        // to what is left — see `peek`.
        self.rows_taken = self.rows_taken.saturating_add(cost.rows);
        self.take_offered(cost.rows);
        self.settle_holes(stranded, cost.hole_credit);
        self.narrow_spoken = self.narrow_spoken.saturating_add(cost.narrow_credit);
        self.widened |= cost.rows > 0;
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
