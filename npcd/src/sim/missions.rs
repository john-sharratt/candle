//! Which character is carrying which mission — the world's half of the mission
//! system.
//!
//! The pure [`Mission`](crate::engine::mission::Mission) model owns what a
//! mission *is*: the ask, its steps, the answer it builds, the report that
//! closes it. This owns *whose* it is, so it lives on [`Sim`](crate::sim::Sim),
//! where an act at the command desk can reach it through `with_sim`, where the
//! standing-task nudge can read it, and where it is serialized and persisted
//! with the rest of the world.
//!
//! A body carries at most one open mission at a time — the one it reads each
//! quiet turn and works through. Missions lodged for a body it has not collected
//! wait in a queue, oldest drawn first. A mission it has reported on is kept in
//! `done` so an operator can still read the outcome and answer after the
//! character has moved on to the next.
//!
//! The command table's generator (`engine::mission_gen`) keeps a **pool** here
//! of the missions it has written, which anybody collecting draws after its own
//! lodged ones and before the routine bank, and a **ledger** of every target it
//! has found — in hand, done, stuck, or found to need nothing — so the same work
//! is not set twice and settled work waits until what it is about changes.
//!
//! Every generated mission carries one step of an **operation** the table holds
//! ([`super::operations`]), run on a workflow from the mind's `missions.yaml`.
//! A target is settled when its operation is, not when one of its steps is
//! reported.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::operations::{Operation, Operations};
use crate::engine::mission::bank::{self, Duty, Facts};
use crate::engine::mission::{Mission, Origin, Outcome, StepOutcome};
use crate::engine::workflow::{Next, Taker, Where, Workflow, STUCK};

/// The outcome a Maker's `report_done` reports on a step that offers a choice.
pub const PASS: &str = "pass";

/// The outcome a Maker's `report_rejected` reports.
pub const REJECT: &str = "reject";

/// What the world offers a routine, owned: [`Sim::mission_material`]'s answer,
/// which a caller borrows as [`Facts`] while it draws.
///
/// [`Sim::mission_material`]: crate::sim::Sim::mission_material
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MissionMaterial {
    pub makers: Vec<String>,
    pub records: Vec<String>,
    pub duties: Vec<Duty>,
    pub table: Option<String>,
}

impl MissionMaterial {
    /// The same material, borrowed for [`Missions::collect`].
    pub fn facts(&self) -> Facts<'_> {
        Facts {
            makers: &self.makers,
            records: &self.records,
            duties: &self.duties,
            table: self.table.as_deref(),
        }
    }
}

/// What has become of a piece of work the generator found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Settled {
    /// A mission for it waits at the table.
    Pooled,
    /// Somebody has taken it up.
    Carried,
    /// Reported done.
    Done,
    /// Reported as not done.
    Stuck,
    /// The generator looked and found nothing to do.
    Nothing,
}

/// One target in the ledger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledgered {
    pub state: Settled,
    /// What the target held when it was last settled — see
    /// `engine::mission_gen::target::Target::fingerprint`.
    pub fingerprint: u64,
    /// Which generator found it.
    pub generator: String,
    /// The mission's brief, or why there was none.
    pub note: String,
    /// How many times a mission for it has been reported stuck while it held
    /// this fingerprint.
    #[serde(default)]
    pub stuck: u32,
    /// The document its mission writes, when it writes one.
    #[serde(default)]
    pub writes: Option<String>,
}

/// How many stuck reports a target takes before it is left until it changes.
const STUCK_LIMIT: u32 = 2;

impl Ledgered {
    /// Whether a target with `fingerprint` is not to be worked now.
    ///
    /// In hand is always blocked. Done, or found to need nothing, is blocked
    /// until what it is about changes — a life gains an event, a document is
    /// edited. Stuck is worked again, until it has been stuck
    /// [`STUCK_LIMIT`] times on the same text.
    pub fn blocks(&self, fingerprint: u64) -> bool {
        match self.state {
            Settled::Pooled | Settled::Carried => true,
            Settled::Done | Settled::Nothing => self.fingerprint == fingerprint,
            Settled::Stuck => self.fingerprint == fingerprint && self.stuck >= STUCK_LIMIT,
        }
    }
}

/// Every character's mission, and the ones lodged for characters yet to collect.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Missions {
    /// The open mission each body is carrying, by body id. At most one.
    active: BTreeMap<String, Mission>,
    /// Missions lodged for a body that it has not collected yet, oldest first.
    lodged: BTreeMap<String, Vec<Mission>>,
    /// The last mission each body finished, kept so an operator can read the
    /// outcome and answer after the character has reported and moved on.
    done: BTreeMap<String, Mission>,
    /// The seed for the next random routine drawn from the bank, bumped each
    /// draw so a character collecting twice does not get the same routine and a
    /// test can pin the sequence.
    seed: u64,
    /// Missions the command table's generator has written, waiting for whoever
    /// collects next, oldest first. Drawn after anything lodged for the body
    /// and before the routine bank.
    #[serde(default)]
    pool: Vec<Mission>,
    /// What has become of every target the generator found, by target key.
    #[serde(default)]
    targets: BTreeMap<String, Ledgered>,
    /// How many generations have been drawn, so the next rotates among the
    /// generators and among equally good targets.
    #[serde(default)]
    drawn: u64,
    /// Every operation the table has held.
    #[serde(default)]
    operations: Operations,
    /// How many missions each body has closed — reported, rejected or called
    /// off. Each is a chapter of its working life, and a character starts a new
    /// conversation with each (`Minds::think`), its journal carrying the work
    /// across.
    #[serde(default)]
    closed: BTreeMap<String, u64>,
    /// The mission each body closed last, however it closed — what its journal
    /// is told it finished as the chapter turns.
    #[serde(default)]
    last_closed: BTreeMap<String, Mission>,
    /// Bodies stood down because the operation they were carrying a stage of
    /// was called off, not yet told — see [`Self::take_stood_down`].
    #[serde(default)]
    stood_down: Vec<String>,
}

impl Missions {
    /// The bodies stood down since this was last asked, each to be told its
    /// mission was withdrawn.
    ///
    /// **Called off is something a Maker has to be told.** An operation called
    /// off — by an operator, or because its draft left the record — took its
    /// carrier's mission away with no word and no chapter closed: the Maker
    /// went on working from its window on a mission that no longer existed,
    /// and one at a quiet desk was not even woken to find out.
    pub fn take_stood_down(&mut self) -> Vec<String> {
        std::mem::take(&mut self.stood_down)
    }
    /// The body's open mission, if it is on one.
    pub fn active(&self, body: &str) -> Option<&Mission> {
        self.active.get(body)
    }

    /// The last mission the body finished, if any.
    pub fn done(&self, body: &str) -> Option<&Mission> {
        self.done.get(body)
    }

    /// The chapter of its working life the body is in: how many missions it has
    /// closed.
    pub fn chapter(&self, body: &str) -> u64 {
        self.closed.get(body).copied().unwrap_or(0)
    }

    /// The mission the body closed last — reported, rejected or called off.
    pub fn last_closed(&self, body: &str) -> Option<&Mission> {
        self.last_closed.get(body)
    }

    /// `mission` of the body's is closed: the next chapter begins.
    fn close(&mut self, body: &str, mission: &Mission) {
        *self.closed.entry(body.to_string()).or_insert(0) += 1;
        self.last_closed.insert(body.to_string(), mission.clone());
    }

    /// The mission to answer an operator's question about this body with — the
    /// open one if it is on one, otherwise the last it finished.
    pub fn latest(&self, body: &str) -> Option<&Mission> {
        self.active.get(body).or_else(|| self.done.get(body))
    }

    /// Whether the body is carrying an open mission right now.
    pub fn is_on_mission(&self, body: &str) -> bool {
        self.active.contains_key(body)
    }

    /// Whether a mission has been lodged for the body and not yet collected.
    pub fn has_lodged(&self, body: &str) -> bool {
        self.lodged.get(body).is_some_and(|queue| !queue.is_empty())
    }

    /// Lodge a mission for a body to collect at the desk. Oldest is drawn first.
    pub fn lodge(&mut self, body: &str, mission: Mission) {
        self.lodged
            .entry(body.to_string())
            .or_default()
            .push(mission);
    }

    /// Give the body a mission to carry now, replacing any open one it held. A
    /// generated mission replaced is called off, and its target released to be
    /// found again — left marked as carried, nothing would ever set it again.
    pub fn assign(&mut self, body: &str, mission: Mission) {
        if let Some(old) = self.active.insert(body.to_string(), mission) {
            if let Some(k) = target_of(&old) {
                if target_of(&self.active[body]) != Some(k) {
                    self.targets.remove(k);
                }
            }
            if let Some((id, _)) = old.operation() {
                if self.active[body].operation().map(|(i, _)| i) != Some(id) {
                    self.operations.cancel(id, "its mission was replaced");
                }
            }
        }
    }

    /// Hold `key` while a mission is being written for it, so a second
    /// generation running at the same time does not choose it too. `false`
    /// when it is already blocked. Released by [`Self::release`] if no mission
    /// comes of it; settled by [`Self::offer`] or [`Self::decline`] if one does.
    pub fn reserve(&mut self, key: &str, generator: &str, fingerprint: u64) -> bool {
        if self.blocks(key, fingerprint) {
            return false;
        }
        let stuck = self
            .targets
            .get(key)
            .filter(|e| e.fingerprint == fingerprint)
            .map_or(0, |e| e.stuck);
        self.targets.insert(
            key.to_string(),
            Ledgered {
                state: Settled::Pooled,
                fingerprint,
                generator: generator.to_string(),
                note: "a mission is being written for it".to_string(),
                stuck,
                writes: None,
            },
        );
        true
    }

    /// Let go of a target [`Self::reserve`] held, when no mission came of it.
    ///
    /// **A target that was stuck goes back to stuck, count and all.** Forgotten
    /// instead, a target whose generation kept failing after the reservation
    /// would never reach [`STUCK_LIMIT`] however often it was reported stuck.
    pub fn release(&mut self, key: &str) {
        if self.pool.iter().any(|m| target_of(m) == Some(key)) {
            return;
        }
        let Some(entry) = self.targets.get_mut(key) else {
            return;
        };
        if entry.state != Settled::Pooled || entry.writes.is_some() {
            return;
        }
        if entry.stuck > 0 {
            entry.state = Settled::Stuck;
            entry.note = "released: no mission came of it".to_string();
        } else {
            self.targets.remove(key);
        }
    }

    /// Collect a mission at the desk: the next one lodged for this body, or —
    /// when nothing is lodged — a fresh non-destructive routine drawn from the
    /// [`bank`] against who and what is around. Either way it becomes the body's
    /// open mission, and the collected one is returned so the desk can hand back
    /// its brief.
    pub fn collect(&mut self, body: &str, facts: &Facts) -> &Mission {
        let mission = match self.take_lodged(body).or_else(|| self.take_pooled(body)) {
            Some(mission) => mission,
            None => {
                let drawn = bank::random(facts, self.seed);
                self.seed = self.seed.wrapping_add(1);
                drawn
            }
        };
        self.active.insert(body.to_string(), mission);
        &self.active[body]
    }

    /// Whether a mission waits at the table that `body` may take — see
    /// [`Self::take_pooled`].
    pub fn has_pooled_for(&self, body: &str) -> bool {
        self.pool.iter().any(|m| match m.operation() {
            Some((id, _)) => self.operations.may_take(id, body),
            None => true,
        })
    }

    /// The oldest generated mission waiting at the table that `body` may take,
    /// now carried by it. A review is never taken by the Maker who drafted it.
    fn take_pooled(&mut self, body: &str) -> Option<Mission> {
        let at = self.pool.iter().position(|m| match m.operation() {
            Some((id, _)) => self.operations.may_take(id, body),
            None => true,
        })?;
        let mission = self.pool.remove(at);
        if let Some(entry) = target_of(&mission).and_then(|k| self.targets.get_mut(k)) {
            entry.state = Settled::Carried;
        }
        if let Some((id, _)) = mission.operation() {
            self.operations.taken(id, body, None);
        }
        Some(mission)
    }

    /// Set the workflows operations run on — see [`Operations::set_workflows`].
    pub fn set_workflows(&mut self, workflows: Vec<Workflow>) {
        self.operations.set_workflows(workflows);
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
    /// from, and `reads` what its work reads; `before` is what its document
    /// says now, when the record already holds it. Returns its id.
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
        before: Option<&str>,
    ) -> Result<u64, String> {
        let id = self
            .operations
            .open(workflow, None, generator, target, objective, document)?;
        self.operations.briefed(id, brief, fields, reads);
        if let Some(text) = before {
            self.operations.kept_before(id, text);
        }
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
    fn settle_if_over(&mut self, id: u64) {
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

    /// Record that a generator looked at a target and found nothing to do.
    pub fn decline(&mut self, key: &str, generator: &str, fingerprint: u64, why: &str) {
        self.targets.insert(
            key.to_string(),
            Ledgered {
                state: Settled::Nothing,
                fingerprint,
                generator: generator.to_string(),
                note: why.to_string(),
                stuck: 0,
                writes: None,
            },
        );
    }

    /// Whether the target `key`, holding `fingerprint`, is not to be worked now.
    pub fn blocks(&self, key: &str, fingerprint: u64) -> bool {
        self.targets.get(key).is_some_and(|e| e.blocks(fingerprint))
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

    /// The generated missions waiting at the table.
    pub fn pooled(&self) -> &[Mission] {
        &self.pool
    }

    /// Whether the table holds enough: at least `keep` missions, and something
    /// each of `makers` may take — up to `keep` more than there are Makers,
    /// past which nothing more is written however the pool falls.
    ///
    /// **Stock is per Maker, not per table.** With four missions waiting the
    /// table counted as stocked while every one of them was a stage its one
    /// free Maker had already carried: it drew routine after routine — the
    /// pressure door, read six times in twenty minutes — and no new work was
    /// ever written for it.
    pub fn stocked_for(&self, makers: &[String], keep: usize) -> bool {
        let n = self.pool.len();
        n >= keep + makers.len() || (n >= keep && makers.iter().all(|m| self.has_pooled_for(m)))
    }

    /// Strike, from every mission carried, waiting at the table or lodged, each
    /// read of a document `holds` says the record no longer has — see
    /// [`Mission::strike_gone_reads`]. Returns each body or the pool (`None`)
    /// with the paths struck from it.
    pub fn strike_gone_reads(
        &mut self,
        holds: &dyn Fn(&str) -> bool,
    ) -> Vec<(Option<String>, Vec<String>)> {
        let carried = self
            .active
            .iter_mut()
            .chain(
                self.lodged
                    .iter_mut()
                    .flat_map(|(b, q)| q.iter_mut().map(move |m| (b, m))),
            )
            .map(|(b, m)| (Some(b.clone()), m.strike_gone_reads(holds)));
        let pooled = self
            .pool
            .iter_mut()
            .map(|m| (None, m.strike_gone_reads(holds)));
        carried
            .chain(pooled)
            .filter(|(_, gone)| !gone.is_empty())
            .collect()
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

    /// Every target the generator has found, and what became of it.
    pub fn targets(&self) -> &BTreeMap<String, Ledgered> {
        &self.targets
    }

    /// Take the next generation's turn number.
    pub fn next_draw(&mut self) -> u64 {
        let turn = self.drawn;
        self.drawn += 1;
        turn
    }

    /// Forget every target that is not in somebody's hands — done, stuck and
    /// found-to-need-nothing alike — so all of it can be found again. Returns
    /// how many were forgotten.
    pub fn forget_settled(&mut self) -> usize {
        let before = self.targets.len();
        self.targets
            .retain(|_, e| matches!(e.state, Settled::Pooled | Settled::Carried));
        before - self.targets.len()
    }

    /// Throw away every generated mission still waiting, releasing its target
    /// to be found again. Returns how many there were.
    pub fn discard_pool(&mut self) -> usize {
        let gone = std::mem::take(&mut self.pool);
        for m in &gone {
            if let Some(k) = target_of(m) {
                self.targets.remove(k);
            }
            if let Some((id, _)) = m.operation() {
                self.operations
                    .cancel(id, "the table's waiting missions were discarded");
            }
        }
        gone.len()
    }

    /// Draw the next lodged mission for a body, oldest first, if one waits. The
    /// caller assigns it; taking and carrying are two steps so a draw that finds
    /// nothing can fall back to a random one without a mission going missing.
    pub fn take_lodged(&mut self, body: &str) -> Option<Mission> {
        let queue = self.lodged.get_mut(body)?;
        if queue.is_empty() {
            return None;
        }
        let mission = queue.remove(0);
        if queue.is_empty() {
            self.lodged.remove(body);
        }
        Some(mission)
    }

    /// Sign a step off on the body's open mission. `false` when it has no open
    /// mission or nothing matched — see [`Mission::check_off`].
    pub fn check_off(&mut self, body: &str, step: &str, outcome: StepOutcome) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.check_off(step, outcome))
    }

    /// Sign off the journeys on the body's open mission that end in `room` on
    /// `level`, where it now stands — see [`Mission::arrived_in`].
    pub fn arrived_in(&mut self, body: &str, room: &str, level: &str) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.arrived_in(room, level))
    }

    /// Sign off the readings of `path` on the body's open mission — see
    /// [`Mission::read_doc`].
    pub fn read_doc(&mut self, body: &str, path: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.read_doc(path))
    }

    /// Sign off the writes of `paths` on the body's open mission — see
    /// [`Mission::committed`].
    pub fn committed(&mut self, body: &str, paths: &[String]) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.committed(paths))
    }

    /// Put the writing of the body's mission document back in front of it —
    /// see [`Mission::reopen_write`].
    pub fn reopen_write(&mut self, body: &str) {
        if let Some(m) = self.active.get_mut(body) {
            if let Some(path) = m.work.as_ref().map(|w| w.writes.clone()) {
                m.reopen_write(&path);
            }
        }
    }

    /// Set the year the body's open mission is worked in — see
    /// [`Mission::travelled`]. `None` when it carries no mission; otherwise
    /// whether a step was signed off.
    pub fn travelled(&mut self, body: &str, year: u32) -> Option<bool> {
        self.active.get_mut(body).map(|m| m.travelled(year))
    }

    /// The year the body works in, set at a time machine for the mission it
    /// carries. `None` is the present.
    pub fn year_of(&self, body: &str) -> Option<u32> {
        self.active.get(body).and_then(|m| m.year)
    }

    /// Sign off the readings on the body's open mission of the machines it just
    /// scanned — see [`Mission::read_off`].
    pub fn read_off(&mut self, body: &str, seen: &[String]) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.read_off(seen))
    }

    /// Sign off the steps on the body's open mission that are about being with
    /// `who`, who is in the room with it — see [`Mission::met`].
    pub fn met(&mut self, body: &str, who: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.met(who))
    }

    /// Sign off the steps on the body's open mission that are about being with
    /// or speaking to `who`, to whom it just said something — see
    /// [`Mission::spoke_with`].
    pub fn spoke_with(&mut self, body: &str, who: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.spoke_with(who))
    }

    /// Note on the body's open mission something the engine saw — see
    /// [`Mission::observe`].
    pub fn observe(&mut self, body: &str, line: impl Into<String>) {
        if let Some(m) = self.active.get_mut(body) {
            m.observe(line);
        }
    }

    /// Count a `report_stuck` turned away on the body's open mission — see
    /// [`Mission::refuse_stuck`]. `0` with no open mission.
    pub fn refuse_stuck(&mut self, body: &str) -> u32 {
        self.active.get_mut(body).map_or(0, Mission::refuse_stuck)
    }

    /// How many missions are lodged for the body and not yet collected.
    pub fn lodged_count(&self, body: &str) -> usize {
        self.lodged.get(body).map_or(0, Vec::len)
    }

    /// Add a step to the body's open mission. `false` when it has no open
    /// mission, or the step was blank or a duplicate — see [`Mission::add_todo`].
    pub fn add_todo(&mut self, body: &str, step: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.add_todo(step))
    }

    /// Cancel the body's open mission without a report — it is called off, not
    /// finished, so it is not kept in `done`. Returns whether there was one.
    pub fn cancel(&mut self, body: &str) -> bool {
        match self.active.remove(body) {
            Some(m) => {
                // Called off is not settled: its target can be found again.
                if let Some(k) = target_of(&m) {
                    self.targets.remove(k);
                }
                if let Some((id, _)) = m.operation() {
                    self.operations.cancel(id, "its mission was called off");
                }
                self.close(body, &m);
                true
            }
            None => false,
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
        self.done.insert(body.to_string(), mission.clone());
        self.close(body, &mission);
        Some(mission)
    }

    /// The outcome a report of done is, on the step operation `id` waits on: a
    /// plain completion where the step has one result, [`PASS`] where it
    /// offers a choice.
    fn forward(&self, id: u64) -> Option<&'static str> {
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

    /// Settle the ledger entry of the target a mission that is not an
    /// operation's is about.
    fn settle(&mut self, mission: &Mission, outcome: Outcome, notes: &str) {
        let key = target_of(mission).map(str::to_string);
        self.settle_target(key, outcome, notes);
    }

    fn settle_target(&mut self, key: Option<String>, outcome: Outcome, notes: &str) {
        if let Some(entry) = key.and_then(|k| self.targets.get_mut(&k)) {
            match outcome {
                Outcome::Pass => entry.state = Settled::Done,
                Outcome::Fail => {
                    entry.state = Settled::Stuck;
                    entry.stuck += 1;
                }
            }
            entry.note = notes.to_string();
        }
    }
}

/// The target a generated mission is about.
fn target_of(m: &Mission) -> Option<&str> {
    match &m.origin {
        Origin::Generated { target, .. } => Some(target),
        _ => None,
    }
}

#[cfg(test)]
#[path = "missions_operation_tests.rs"]
mod operation_tests;

#[cfg(test)]
mod tests {
    use super::Missions;
    use crate::engine::mission::bank::Facts;
    use crate::engine::mission::{Mission, Origin, Outcome, StepOutcome, Todo};

    fn a_mission(prompt: &str) -> Mission {
        Mission::new(
            prompt,
            vec![Todo::new("step one"), Todo::new("step two")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        )
    }

    /// **Each mission closed — reported or called off — turns a chapter**, and
    /// the mission closed last is kept to close it with.
    #[test]
    fn each_mission_closed_turns_a_chapter() {
        let mut m = Missions::default();
        assert_eq!(m.chapter("bram"), 0);
        m.assign("bram", a_mission("first"));
        m.report("bram", Outcome::Pass, "done", None);
        assert_eq!(m.chapter("bram"), 1);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("first")
        );
        m.assign("bram", a_mission("second"));
        assert!(m.cancel("bram"));
        assert_eq!(m.chapter("bram"), 2);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("second"),
            "a mission called off closes a chapter too"
        );
        assert!(!m.cancel("bram"), "nothing open, nothing closed");
        assert_eq!(m.chapter("bram"), 2);
        assert_eq!(m.chapter("cindy"), 0);
    }

    #[test]
    fn a_lodged_mission_is_drawn_oldest_first_and_then_the_queue_empties() {
        let mut m = Missions::default();
        assert!(!m.has_lodged("bram"));
        m.lodge("bram", a_mission("first"));
        m.lodge("bram", a_mission("second"));
        assert!(m.has_lodged("bram"));

        assert_eq!(m.take_lodged("bram").unwrap().prompt, "first");
        assert_eq!(m.take_lodged("bram").unwrap().prompt, "second");
        assert!(m.take_lodged("bram").is_none(), "queue empties");
        assert!(!m.has_lodged("bram"));
    }

    #[test]
    fn assigning_a_mission_makes_it_the_active_one() {
        let mut m = Missions::default();
        assert!(!m.is_on_mission("wren"));
        m.assign("wren", a_mission("survey the ridge"));
        assert!(m.is_on_mission("wren"));
        assert_eq!(m.active("wren").unwrap().prompt, "survey the ridge");
        // A second assignment replaces the first — a body carries one at a time.
        m.assign("wren", a_mission("read the canon"));
        assert_eq!(m.active("wren").unwrap().prompt, "read the canon");
    }

    #[test]
    fn steps_are_checked_off_and_added_only_on_an_open_mission() {
        let mut m = Missions::default();
        // No mission: nothing to act on.
        assert!(!m.check_off("pax", "step one", StepOutcome::Achieved));
        assert!(!m.add_todo("pax", "a new step"));

        m.assign("pax", a_mission("do the thing"));
        assert!(
            m.check_off("pax", "STEP one", StepOutcome::Achieved),
            "trim + case-insensitive"
        );
        assert!(
            !m.check_off("pax", "step one", StepOutcome::Thwarted),
            "already done"
        );
        assert!(m.add_todo("pax", "a discovered step"));
        assert!(
            !m.add_todo("pax", "  a discovered step  "),
            "open duplicate"
        );
        assert_eq!(m.active("pax").unwrap().todo.len(), 3);
    }

    #[test]
    fn reporting_closes_the_open_mission_and_keeps_it_for_the_operator() {
        let mut m = Missions::default();
        assert!(
            m.report("soren", Outcome::Pass, "n/a", None).is_none(),
            "nothing to report without an open mission"
        );

        m.assign("soren", a_mission("check the record"));
        let closed = m
            .report(
                "soren",
                Outcome::Pass,
                "it holds against the storyline",
                Some("the eastern date is wrong".to_string()),
            )
            .expect("an open mission is reported");
        assert!(!closed.is_open());
        assert_eq!(closed.report.as_ref().unwrap().outcome, Outcome::Pass);

        // The body is now free of its mission, but the operator can still read it.
        assert!(!m.is_on_mission("soren"));
        assert!(m.active("soren").is_none());
        let done = m.done("soren").expect("kept for the operator");
        assert_eq!(done.answer.as_deref(), Some("the eastern date is wrong"));
        assert_eq!(m.latest("soren").unwrap().prompt, "check the record");
    }

    #[test]
    fn collecting_takes_a_lodged_mission_first_then_draws_from_the_bank() {
        let mut m = Missions::default();
        let makers = vec!["Wren".to_string(), "Pax".to_string()];
        let records = vec!["the-charge".to_string()];
        let facts = Facts {
            makers: &makers,
            records: &records,
            ..Facts::default()
        };

        // A lodged mission is taken first, verbatim, and becomes active.
        m.lodge("bram", a_mission("the lodged one"));
        assert_eq!(m.collect("bram", &facts).prompt, "the lodged one");
        assert!(m.is_on_mission("bram"));
        assert!(!m.has_lodged("bram"), "the lodged queue is drawn down");

        // With nothing lodged, a routine is drawn from the bank — a valid, open,
        // non-empty, non-destructive mission it can act on.
        let drawn = m.collect("bram", &facts).clone();
        assert!(drawn.is_open());
        assert!(!drawn.todo.is_empty(), "a routine gives steps to act on");
        assert!(matches!(drawn.origin, Origin::Random { .. }));
    }

    /// A read struck wherever a mission stands — carried, lodged or pooled —
    /// and each place it was struck named.
    #[test]
    fn a_gone_read_is_struck_from_every_mission_wherever_it_waits() {
        let gone = "layers/stories/gone.md";
        let with_read = |p: &str| {
            Mission::new(
                p,
                vec![Todo::new(format!("read {gone}")), Todo::report("report")],
                Origin::Random {
                    routine: "r".into(),
                },
            )
        };
        let mut m = Missions::default();
        m.assign("vespera", with_read("carried"));
        m.lodge("sila", with_read("lodged"));
        m.pool.push(with_read("pooled"));
        m.assign("paxon", a_mission("reads nothing"));
        let struck = m.strike_gone_reads(&|p| p != gone);
        assert_eq!(
            struck,
            [
                (Some("vespera".to_string()), vec![gone.to_string()]),
                (Some("sila".to_string()), vec![gone.to_string()]),
                (None, vec![gone.to_string()]),
            ]
        );
        for mission in [&m.active["vespera"], &m.lodged["sila"][0], &m.pool[0]] {
            assert_eq!(mission.todo.len(), 1, "{}", mission.prompt);
            assert!(mission.todo[0].reports);
        }
    }

    #[test]
    fn cancelling_removes_the_open_mission_without_recording_it() {
        let mut m = Missions::default();
        m.assign("bram", a_mission("do the thing"));
        assert!(m.cancel("bram"), "an open mission is cancelled");
        assert!(!m.is_on_mission("bram"));
        assert!(
            m.done("bram").is_none(),
            "a cancelled mission is not a finished one"
        );
        assert!(!m.cancel("bram"), "nothing to cancel twice");
    }

    #[test]
    fn missions_survive_a_round_trip_through_json() {
        let mut m = Missions::default();
        m.assign("bram", a_mission("do the thing"));
        m.add_todo("bram", "look twice");
        m.lodge("bram", a_mission("then this"));
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", Some("it holds".to_string()));
        let blob = serde_json::to_value(&m).unwrap();
        let back: Missions = serde_json::from_value(blob).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            back.active("bram").unwrap().todo.last().unwrap().text,
            "look twice"
        );
        assert_eq!(
            back.done("yen").unwrap().answer.as_deref(),
            Some("it holds")
        );
    }

    #[test]
    fn latest_is_the_open_mission_over_a_finished_one() {
        let mut m = Missions::default();
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", None);
        m.assign("yen", a_mission("second task"));
        // With both a finished and an open mission, the open one is what an
        // operator's question is about.
        assert_eq!(m.latest("yen").unwrap().prompt, "second task");
    }
}
