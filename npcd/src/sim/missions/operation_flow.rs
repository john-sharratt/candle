//! Moves each operation through the steps of its workflow, from the first mission
//! it offers to its settlement.
//!
//! An operation waits on one step at a time, and the table's verdict on that step
//! decides where it moves next. Its target is settled only once the whole operation
//! is over: done settles the target as done, failed as stuck, and called off leaves
//! it free to be found again. A publish that cannot reach git puts its mission down
//! without a report, and the step is offered again once the wait is over. Work that
//! no remaining maker may take is called off, since otherwise it would sit at the
//! table for ever and keep its target blocked.

use super::*;

impl Missions {
    /// Set the workflows operations run on — see [`Operations::set_workflows`].
    pub fn set_workflows(&mut self, workflows: Vec<Workflow>) {
        self.operations.set_workflows(workflows);
    }

    /// Set who may approve a critical step — empty for a world that sets none,
    /// so a critical step can then only be held.
    pub fn set_authorizers(&mut self, authorizers: Vec<Authorizer>) {
        self.authorizers = authorizers;
    }

    /// Set whether the world's order-giver may author new templates.
    pub fn set_authoring(&mut self, authoring: bool) {
        self.authoring = authoring;
    }

    /// Whether the world's order-giver may author new templates
    /// (`mission_write`).
    pub fn authoring(&self) -> bool {
        self.authoring
    }

    /// Who may approve a critical step here (`docs/mission_templates.md` §6).
    pub fn authorizers(&self) -> &[Authorizer] {
        &self.authorizers
    }

    /// Take `from`'s workflows, routines and authorizers in place of these:
    /// they are the mind's, not the world's, and a world taken in from
    /// elsewhere keeps the ones this machine loaded.
    pub fn adopt_workflows(&mut self, from: &mut Missions) {
        self.operations.adopt_workflows(&mut from.operations);
        self.routines = std::mem::take(&mut from.routines);
        self.draw = from.draw;
        self.authorizers = std::mem::take(&mut from.authorizers);
        self.authoring = from.authoring;
    }

    /// Call off every operation not yet over, for `why` — what a world that
    /// has come to run on its own missions does with the record's work opened
    /// before it did. Returns the ids called off.
    pub fn call_off_operations(&mut self, why: &str) -> Vec<u64> {
        self.call_off_operations_but(None, why)
    }

    /// [`Self::call_off_operations`], all but those on workflow `keep` — the
    /// one a world with its own missions runs (`docs/mission_templates.md`
    /// §6).
    pub fn call_off_operations_but(&mut self, keep: Option<&str>, why: &str) -> Vec<u64> {
        let open: Vec<u64> = self
            .operations
            .all()
            .filter(|o| !o.settled())
            .filter(|o| keep.is_none_or(|k| o.run.workflow != k))
            .map(|o| o.id)
            .collect();
        open.into_iter()
            .filter(|id| self.cancel_operation(*id, why))
            .collect()
    }

    /// Record what operation `id`'s document said when its step was taken up.
    pub fn found(&mut self, id: u64, text: String) {
        if let Some(o) = self.operations.get_mut(id) {
            o.found = Some(text);
        }
    }

    /// Open an operation on `workflow` for a generator's accepted proposal,
    /// for `fingerprint` of its target, waiting on the workflow's first step —
    /// set on the table by the engine's loop like every other step.
    ///
    /// `objective` says in a line what it is for, `brief` what its document is
    /// to tell, `fields` the proposal's own, which every step's prompt is filled
    /// from, and `reads` what its work reads. Returns its id.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &mut self,
        workflow: &str,
        generator: &str,
        target: &str,
        fingerprint: u64,
        objective: &str,
        document: &str,
        brief: &str,
        fields: BTreeMap<String, String>,
        reads: Vec<String>,
    ) -> Result<u64, String> {
        let id = self
            .operations
            .open(workflow, None, generator, target, objective, document)?;
        self.operations.briefed(id, brief, fields, reads);
        let stuck = self
            .targets
            .get(target)
            .filter(|e| e.fingerprint == fingerprint)
            .map_or(0, |e| e.stuck);
        self.targets.insert(
            target.to_string(),
            Ledgered {
                state: Settled::Pooled,
                fingerprint,
                generator: generator.to_string(),
                note: objective.to_string(),
                stuck,
                writes: Some(document.to_string()),
            },
        );
        Ok(id)
    }

    /// Put `mission` on the table as the step operation `id` waits on.
    pub fn offer_step(&mut self, id: u64, mut mission: Mission) {
        let step = self
            .operations
            .get(id)
            .and_then(|o| o.step())
            .unwrap_or_default()
            .to_string();
        if let Origin::Generated {
            operation,
            step: carried,
            ..
        } = &mut mission.origin
        {
            *operation = id;
            *carried = step;
        }
        self.operations.offered(id);
        self.pool.push(mission);
    }

    /// The table took the step operation `id` waits on — one of its call's
    /// verdicts, `outcome`, with what it found — and the operation moves on;
    /// its target is settled once it is over.
    pub fn table_took(&mut self, id: u64, outcome: &str, found: &str) -> Result<Where, String> {
        let at = self
            .operations
            .advance(id, &Taker::Table, Some(outcome), found)?;
        self.settle_if_over(id);
        Ok(at)
    }

    /// Settle operation `id`'s target, and stand its carriers down, once the
    /// operation is over: done is the target done; failed, stuck; called off,
    /// free to be found again.
    pub(super) fn settle_if_over(&mut self, id: u64) {
        let Some(o) = self.operations.get(id) else {
            return;
        };
        let (target, at) = (o.target.clone(), o.run.at.clone());
        match at {
            Where::NextStep(_) => {}
            Where::Done => self.settle_target(Some(target), Outcome::Pass, "it stands"),
            Where::Failed(why) => self.settle_target(Some(target), Outcome::Fail, &why),
            Where::Cancelled(_) => {
                self.targets.remove(&target);
            }
        }
    }

    /// Every operation the table has held.
    pub fn operations(&self) -> &Operations {
        &self.operations
    }

    /// Put settled operation `id` back on `step` of its workflow, in a new
    /// round — succeeded lore sent to be checked again, say.
    pub fn reopen(&mut self, id: u64, step: &str) -> Result<(), String> {
        self.operations.reopen(id, step)?;
        if let Some(o) = self.operations.get(id) {
            if let Some(entry) = self.targets.get_mut(&o.target) {
                entry.state = Settled::Pooled;
            }
        }
        Ok(())
    }

    /// Send operation `id` to `step` of its workflow: moved there when it is
    /// running ([`Self::move_to`]), reopened there when it is over
    /// ([`Self::reopen`]).
    pub fn send_to_step(&mut self, id: u64, step: &str) -> Result<(), String> {
        let settled = self
            .operations
            .get(id)
            .map(Operation::settled)
            .ok_or("no such operation")?;
        match settled {
            true => self.reopen(id, step),
            false => self.move_to(id, step),
        }
    }

    /// Move unsettled operation `id` to `step`, whatever it waited on: a step
    /// waiting at the table leaves it, and the operation goes on from `step` in
    /// a new round. `false` when it is settled or carried.
    pub fn move_to(&mut self, id: u64, step: &str) -> Result<(), String> {
        if self.carrying(id).is_some() {
            return Err("its step is being carried".into());
        }
        let target = self
            .operations
            .get(id)
            .map(|o| o.target.clone())
            .ok_or("no such operation")?;
        if !self.operations.cancel(id, "moved by hand") {
            return Err("it is over".into());
        }
        self.pool
            .retain(|m| m.operation().map(|(i, _)| i) != Some(id));
        self.operations.reopen(id, step)?;
        if let Some(entry) = self.targets.get_mut(&target) {
            entry.state = Settled::Pooled;
        }
        Ok(())
    }

    /// Record that failed operation `id`'s document has been settled.
    pub fn mark_retired(&mut self, id: u64) {
        self.operations.mark_retired(id);
    }

    /// Open an operation on `workflow` that starts at `step` with `document`,
    /// which already stands — an operator putting a document through the
    /// table's reading and a Maker's review. Returns its id.
    pub fn review_document(
        &mut self,
        workflow: &str,
        step: &str,
        document: &str,
    ) -> Result<u64, String> {
        self.operations.open(
            workflow,
            Some(step),
            "operator",
            &format!("doc:{document}"),
            &format!("Review {document}"),
            document,
        )
    }

    /// Call operation `id` off: its waiting mission leaves the table and a
    /// Maker carrying one of its stages is stood down. `false` when it was
    /// already over or there is no such operation.
    pub fn cancel_operation(&mut self, id: u64, why: &str) -> bool {
        let Some(target) = self.operations.get(id).map(|o| o.target.clone()) else {
            return false;
        };
        if !self.operations.cancel(id, why) {
            return false;
        }
        self.pool
            .retain(|m| m.operation().map(|(i, _)| i) != Some(id));
        let carriers: Vec<String> = self
            .active
            .iter()
            .filter(|(_, m)| m.operation().map(|(i, _)| i) == Some(id))
            .map(|(body, _)| body.clone())
            .collect();
        for body in carriers {
            if let Some(m) = self.active.remove(&body) {
                self.close(&body, &m);
                self.stood_down.push(body);
            }
        }
        self.targets.remove(&target);
        true
    }

    /// Edit operation `id`: its name, what it is for, and — while its next
    /// mission still waits at the table — that mission's brief.
    pub fn edit_operation(
        &mut self,
        id: u64,
        name: Option<&str>,
        objective: Option<&str>,
        brief: Option<&str>,
    ) -> Result<&Operation, String> {
        if self.operations.get(id).is_none() {
            return Err("no such operation".into());
        }
        if let Some(n) = name {
            self.operations.rename(id, n)?;
        }
        if let Some(o) = objective {
            self.operations.set_objective(id, o)?;
        }
        if let Some(b) = brief.map(str::trim) {
            if b.is_empty() {
                return Err("a brief cannot be empty".into());
            }
            let waiting = self
                .pool
                .iter_mut()
                .find(|m| m.operation().map(|(i, _)| i) == Some(id))
                .ok_or("its mission is not waiting at the table, so its brief is set")?;
            waiting.prompt = b.to_string();
        }
        Ok(self.operations.get(id).expect("checked above"))
    }

    /// The brief of operation `id`'s mission waiting at the table, if one is.
    pub fn waiting_brief(&self, id: u64) -> Option<&str> {
        self.pool
            .iter()
            .find(|m| m.operation().map(|(i, _)| i) == Some(id))
            .map(|m| m.prompt.as_str())
    }

    /// Who is carrying a stage of operation `id` right now.
    pub fn carrying(&self, id: u64) -> Option<&str> {
        self.active
            .iter()
            .find(|(_, m)| m.operation().map(|(i, _)| i) == Some(id))
            .map(|(b, _)| b.as_str())
    }

    /// Call off every operation whose next mission waits at the table for a
    /// Maker none of `makers` may be: each of them has already carried a stage
    /// of it. Returns the operations called off.
    ///
    /// **Work nobody may take is not work waiting.** Nobody checks their own
    /// work or checks it twice ([`Operations::may_take`]), so with a small
    /// cast a review reported stuck twice can run out of Makers who have not
    /// touched it. It then sat at the table for ever: never taken, its target
    /// blocked from being found again, and counted as stock — enough of them
    /// and the generator, keeping the table stocked, never opened another.
    /// Called off, its target is free to be found afresh.
    pub fn call_off_untakeable(&mut self, makers: &[String]) -> Vec<u64> {
        // A world nobody has stood in yet — at boot, before anybody is bound —
        // is not one with nobody left.
        if makers.is_empty() {
            return Vec::new();
        }
        let stranded: Vec<u64> = self
            .pool
            .iter()
            .filter_map(|m| m.operation().map(|(id, _)| id))
            .filter(|id| !makers.iter().any(|b| self.operations.may_take(*id, b)))
            .collect();
        stranded
            .into_iter()
            .filter(|id| {
                self.cancel_operation(*id, "no Maker is left who has not already worked on it")
            })
            .collect()
    }

    /// Operation `id`'s store landed on the record as `commit`.
    pub fn published(&mut self, id: u64, commit: &str) {
        self.operations.published(id, commit);
    }

    /// Offer operation `id`'s publish waiting on git at once. `false` when
    /// none is waiting.
    pub fn retry_now(&mut self, id: u64) -> bool {
        self.operations.retry_now(id)
    }

    /// The body's publish could not reach git, for `why`, at `now_ms`: its
    /// mission is put down without a report — neither an outcome nor a stuck
    /// one — and its operation's step is offered again once the wait is over
    /// (`docs/replicated_substrate.md` §10.3). Returns the operation.
    pub fn publish_waits(&mut self, body: &str, why: &str, now_ms: u64) -> Option<u64> {
        let (id, _) = self.active.get(body)?.operation()?;
        let m = self.active.remove(body)?;
        self.close(body, &m);
        self.operations.released(id);
        self.operations.publish_waits(id, why, now_ms);
        Some(id)
    }
}
