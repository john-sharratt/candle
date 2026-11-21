//! Closes a body's missions and turns its chapters, keeping the last one closed
//! for its journal and the record of what it finished.
//!
//! A chapter closes when a mission is reported, rejected or called off, so the
//! journal can carry the work into a fresh conversation. A routine turns no
//! chapter, because resealing a journal for every quick routine would cost the
//! engine the time the real work needs. A report is filed to whoever posted the
//! work, and the operation it belongs to moves on. A report that the workflow will
//! not take still closes the mission, so the body is not left carrying a step
//! that nobody can hand in.

use super::*;

impl Missions {
    /// The last mission the body finished, if any.
    pub fn done(&self, body: &str) -> Option<&Mission> {
        self.done.get(body)
    }

    /// The chapter of its working life the body is in: how many missions it has
    /// closed, not counting the table's routines.
    pub fn chapter(&self, body: &str) -> u64 {
        self.closed.get(body).copied().unwrap_or(0)
    }

    /// The work the body closed last — reported, rejected or called off — which
    /// is what its chapter closes with; the table's routines are not counted.
    pub fn last_closed(&self, body: &str) -> Option<&Mission> {
        self.last_closed.get(body)
    }

    /// `mission` of the body's is closed: the next chapter begins — unless it
    /// was one of the table's routines.
    ///
    /// **A routine is an interlude, not a stretch of work.** A chapter closes so
    /// the work it held is carried across in the journal and the next starts in
    /// a conversation of its own, which reseals the journal from nothing. The
    /// routines a Maker draws while the table has no work for it — read a
    /// station, walk the halls — take a couple of minutes each: on WW they
    /// turned 52 chapters in an hour, each a whole journal resealed, and the
    /// reseals took the engine the table needed to write the work that would
    /// have ended the routines.
    ///
    /// A routine is not what the chapter closes with either: one handed in
    /// between a mission and the next turn would otherwise write the chapter's
    /// journal entry in the mission's place.
    pub(super) fn close(&mut self, body: &str, mission: &Mission) {
        // A standing job posted at the table is a routine still, whoever set
        // it.
        if matches!(
            mission.origin,
            Origin::Random { .. } | Origin::Ordered { job: Some(_), .. } | Origin::Own
        ) {
            return;
        }
        *self.closed.entry(body.to_string()).or_insert(0) += 1;
        self.last_closed.insert(body.to_string(), mission.clone());
    }

    /// The mission to answer an operator's question about this body with — the
    /// open one if it is on one, otherwise the last it finished.
    pub fn latest(&self, body: &str) -> Option<&Mission> {
        self.active.get(body).or_else(|| self.done.get(body))
    }

    /// Cancel the body's open mission without a report — it is called off, not
    /// finished, so it is not kept in `done`. Returns whether there was one.
    pub fn cancel(&mut self, body: &str) -> bool {
        self.cancel_for(body, CALLED_OFF)
    }

    /// [`Self::cancel`], its poster told `why` when it was posted.
    pub(super) fn cancel_for(&mut self, body: &str, why: &str) -> bool {
        match self.active.remove(body) {
            Some(m) => {
                // Called off is not settled: its target can be found again.
                if let Some(k) = target_of(&m) {
                    self.targets.remove(k);
                }
                if let Some((id, _)) = m.operation() {
                    self.operations.cancel(id, "its mission was called off");
                }
                self.called_off(body, &m, why);
                self.close(body, &m);
                self.agenda_rank(body);
                true
            }
            None => false,
        }
    }

    /// A body starts over: what it carries is called off, every mission
    /// waiting for it, whoever lodged it, is dropped — each poster told — the
    /// work it posted that nobody has taken up yet is taken off the table, and
    /// the missions it finished are forgotten with the rest of its past.
    /// Returns how many missions went. The body stays in the world, taking
    /// orders or giving them as it did.
    pub fn start_over(&mut self, body: &str) -> usize {
        let mut gone = 0;
        while gone < START_OVER_MOST && self.cancel_for(body, STARTED_OVER) {
            gone += 1;
        }
        // Called off, each went to what it finished last: started over, it
        // has finished nothing.
        self.done.remove(body);
        self.last_closed.remove(body);
        for m in self.lodged.remove(body).unwrap_or_default() {
            self.called_off(body, &m, STARTED_OVER);
            gone += 1;
        }
        let posted =
            |m: &Mission| matches!(&m.origin, Origin::Ordered { poster, .. } if poster == body);
        let waiting = self.pool.len() + self.lodged.values().map(Vec::len).sum::<usize>();
        self.pool.retain(|m| !posted(m));
        for queue in self.lodged.values_mut() {
            queue.retain(|m| !posted(m));
        }
        self.lodged.retain(|_, queue| !queue.is_empty());
        gone + waiting - self.pool.len() - self.lodged.values().map(Vec::len).sum::<usize>()
    }

    /// A body is gone from the world: what it carried is called off, what was
    /// lodged for it is dropped, and the poster of any of it is told — a
    /// mission set for somebody who is no longer there is otherwise out for
    /// good, and the table never reads empty again.
    pub fn forget(&mut self, body: &str) {
        self.cancel_for(body, GONE);
        for m in self.lodged.remove(body).unwrap_or_default() {
            self.called_off(body, &m, GONE);
        }
        self.unordered.remove(body);
    }

    /// File `m`, a posted mission ended other than by its carrier's report, to
    /// its poster as not done, for `why` — so its poster is told, and it is no
    /// longer counted as out.
    pub(super) fn called_off(&mut self, body: &str, m: &Mission, why: &str) {
        if let Origin::Ordered { poster, .. } = &m.origin {
            self.orders.filed(Filed {
                poster: poster.clone(),
                carrier: body.to_string(),
                brief: m.prompt.clone(),
                outcome: Outcome::Fail,
                notes: why.to_string(),
            });
        }
    }

    /// File the body's completion report, closing the open mission and keeping
    /// it in `done` so its outcome and answer can still be read. Returns the
    /// closed mission, or `None` if the body had no open mission to report.
    pub fn report(
        &mut self,
        body: &str,
        outcome: Outcome,
        notes: &str,
        answer: Option<String>,
    ) -> Option<Mission> {
        // **The target is settled when its operation is.** A step reported is
        // the operation moved on: where to is its workflow's.
        //
        // **A report the workflow will not take still closes the mission.**
        // Kept open, its Maker would carry a step nobody can hand in — the
        // operation called off under it, or its step moved by hand — and stand
        // at the table with it for good. The step, if it still waits, goes
        // back on offer.
        if let Some((id, step)) = self.active.get(body)?.operation() {
            let step = step.to_string();
            let said = match outcome {
                Outcome::Pass => self.forward(id),
                Outcome::Fail => Some(STUCK),
            };
            if let Err(e) =
                self.operations
                    .advance(id, &Taker::Actor(body.to_string()), said, notes)
            {
                tracing::warn!(operation = id, %step, "a report the workflow refused: {e}");
                let held = self.operations.get(id).and_then(|o| o.holder.as_deref());
                if held == Some(body) {
                    self.operations.released(id);
                }
            }
        }
        let mut mission = self.active.remove(body)?;
        mission.complete(outcome, notes, answer);
        match mission.operation() {
            Some((id, _)) => self.settle_if_over(id),
            None => self.settle(&mission, outcome, notes),
        }
        // A posted mission's report is its poster's; a standing mission's is
        // filed for whoever gives orders here, and, done, it is done there.
        match &mission.origin {
            Origin::Ordered { poster, .. } => self.orders.filed(Filed {
                poster: poster.clone(),
                carrier: body.to_string(),
                brief: mission.prompt.clone(),
                outcome,
                notes: notes.to_string(),
            }),
            Origin::Standing { id, room } => {
                let key = format!("{id}@{room}");
                match outcome {
                    Outcome::Pass => {
                        self.standing_failed.remove(&key);
                        self.standing.insert(key, None);
                    }
                    Outcome::Fail => {
                        self.standing_failed.insert(key, None);
                    }
                }
                self.orders.filed(Filed {
                    poster: EVERY_POSTER.to_string(),
                    carrier: body.to_string(),
                    brief: mission.prompt.clone(),
                    outcome,
                    notes: notes.to_string(),
                });
            }
            _ => {}
        }
        self.done.insert(body.to_string(), mission.clone());
        self.close(body, &mission);
        // The next on the agenda is worked, where it was left.
        self.agenda_rank(body);
        Some(mission)
    }

    /// The outcome a report of done is, on the step operation `id` waits on: a
    /// plain completion where the step has one result, [`PASS`] where it
    /// offers a choice.
    pub(super) fn forward(&self, id: u64) -> Option<&'static str> {
        let o = self.operations.get(id)?;
        let step = self.operations.workflow_of(o)?.step(o.step()?)?;
        match step.next {
            Next::Single(_) => None,
            Next::Outcomes(_) => Some(PASS),
        }
    }

    /// Send back the work the body's step judges: its mission closes as failed
    /// and the operation goes where the step's [`REJECT`] leads — in the
    /// workflows npcd runs, to be fixed. Returns the closed mission, or `None`
    /// when the body's step may not reject.
    pub fn reject(&mut self, body: &str, why: &str) -> Option<Mission> {
        let m = self.active.get(body)?;
        if !m.may_reject() {
            return None;
        }
        let (id, _) = m.operation()?;
        self.operations
            .advance(id, &Taker::Actor(body.to_string()), Some(REJECT), why)
            .ok()?;
        let mut mission = self.active.remove(body)?;
        mission.complete(Outcome::Fail, why, None);
        self.settle_if_over(id);
        self.done.insert(body.to_string(), mission.clone());
        self.close(body, &mission);
        Some(mission)
    }
}
