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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::mission::bank::{self, Duty, Facts};
use crate::engine::mission::{Mission, Origin, Outcome, StepOutcome};

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
    /// The document its mission writes, when it writes one — what a review
    /// reads once the mission is done.
    #[serde(default)]
    pub writes: Option<String>,
}

/// How many stuck reports a target takes before it is left until it changes.
const STUCK_LIMIT: u32 = 2;

/// How a review's target key begins — `review:<path>`. Kept here, where the
/// ledger is, because the ledger is what tells a review from the work it reads.
pub const REVIEW: &str = "review:";

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
    /// The documents generated missions have written, as each was reported
    /// done — see [`Missions::written`].
    #[serde(default)]
    written: Vec<String>,
}

impl Missions {
    /// The body's open mission, if it is on one.
    pub fn active(&self, body: &str) -> Option<&Mission> {
        self.active.get(body)
    }

    /// The last mission the body finished, if any.
    pub fn done(&self, body: &str) -> Option<&Mission> {
        self.done.get(body)
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
        let mission = match self.take_lodged(body).or_else(|| self.take_pooled()) {
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

    /// The oldest generated mission waiting at the table, now carried.
    fn take_pooled(&mut self) -> Option<Mission> {
        if self.pool.is_empty() {
            return None;
        }
        let mission = self.pool.remove(0);
        if let Some(entry) = target_of(&mission).and_then(|k| self.targets.get_mut(k)) {
            entry.state = Settled::Carried;
        }
        Some(mission)
    }

    /// Put a generated mission on the table for `fingerprint` of its target.
    pub fn offer(&mut self, mission: Mission, fingerprint: u64) {
        if let Origin::Generated { generator, target } = &mission.origin {
            let stuck = self
                .targets
                .get(target)
                .filter(|e| e.fingerprint == fingerprint)
                .map_or(0, |e| e.stuck);
            self.targets.insert(
                target.clone(),
                Ledgered {
                    state: Settled::Pooled,
                    fingerprint,
                    generator: generator.clone(),
                    note: mission.mission_text(),
                    stuck,
                    writes: mission.work.as_ref().map(|w| w.writes.clone()),
                },
            );
        }
        self.pool.push(mission);
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

    /// Every document a generated mission has written and been reported done
    /// for, in the order they were finished — what a review reads. A review's
    /// own revisions are not added: a revised document is not reviewed again
    /// for having been revised.
    pub fn written(&self) -> &[String] {
        &self.written
    }

    /// Put a document in line for review, as though a mission had written it.
    /// `false` when it is already in line.
    pub fn review_later(&mut self, path: &str) -> bool {
        if self.written.iter().any(|p| p == path) {
            return false;
        }
        self.written.push(path.to_string());
        true
    }

    /// The generated missions waiting at the table.
    pub fn pooled(&self) -> &[Mission] {
        &self.pool
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
        let mut mission = self.active.remove(body)?;
        mission.complete(outcome, notes, answer);
        if let (Outcome::Pass, Some(target), Some(work)) =
            (outcome, target_of(&mission), mission.work.as_ref())
        {
            match target.starts_with(REVIEW) {
                // Revised: it has had its second reading, and the revision
                // changing its text must not make it the first in line again.
                true => self.written.retain(|p| p != &work.writes),
                false if !self.written.contains(&work.writes) => {
                    self.written.push(work.writes.clone())
                }
                false => {}
            }
        }
        if let Some(entry) = target_of(&mission).and_then(|k| self.targets.get_mut(k)) {
            match outcome {
                Outcome::Pass => entry.state = Settled::Done,
                Outcome::Fail => {
                    entry.state = Settled::Stuck;
                    entry.stuck += 1;
                }
            }
            entry.note = notes.to_string();
        }
        self.done.insert(body.to_string(), mission.clone());
        Some(mission)
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
mod tests {
    use super::{Missions, Settled};
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

    fn generated(target: &str) -> Mission {
        Mission::new(
            format!("Write the next event for {target}."),
            vec![Todo::new("write it")],
            Origin::Generated {
                generator: "lives".into(),
                target: target.into(),
            },
        )
    }

    /// **A generated mission waits at the table, is carried, and settles its
    /// target.** Lodged work still comes first; the pool before the bank; and a
    /// done target is free again only once what it is about has changed.
    #[test]
    fn a_generated_mission_moves_its_target_through_the_ledger() {
        let mut m = Missions::default();
        let facts = Facts::default();
        m.offer(generated("life:keeper"), 7);
        assert!(m.blocks("life:keeper", 7), "pooled is in hand");
        assert!(m.blocks("life:keeper", 8), "whatever it holds");
        m.lodge("bram", a_mission("lodged first"));
        assert_eq!(m.collect("bram", &facts).prompt, "lodged first");
        m.cancel("bram");

        let taken = m.collect("bram", &facts).clone();
        assert_eq!(taken.prompt, "Write the next event for life:keeper.");
        assert!(m.pooled().is_empty());
        assert_eq!(m.targets()["life:keeper"].state, Settled::Carried);

        m.report("bram", Outcome::Pass, "written", None);
        assert_eq!(m.targets()["life:keeper"].state, Settled::Done);
        assert!(m.blocks("life:keeper", 7), "done, and nothing has changed");
        assert!(!m.blocks("life:keeper", 8), "the life has a new event");

        // An empty pool falls back to the bank.
        let drawn = m.collect("yen", &facts).clone();
        assert!(matches!(drawn.origin, Origin::Random { .. }));
    }

    /// **What a generated mission wrote is kept for review once it is reported
    /// done** — not when it fails — and a review reported done takes its
    /// document out of the line.
    #[test]
    fn a_document_written_and_reported_done_is_kept_for_review() {
        use crate::engine::mission::Work;
        let with_work = |target: &str, path: &str| {
            generated(target).with_work(Work {
                writes: path.into(),
                reads: vec![],
                min_words: 0,
            })
        };
        let mut m = Missions::default();
        let facts = Facts::default();
        for (target, path, outcome) in [
            ("life:keeper", "layers/life/keeper/2488 A.md", Outcome::Pass),
            ("era:x", "layers/stories/b.md", Outcome::Fail),
            (
                "review:layers/life/keeper/2488 A.md",
                "layers/life/keeper/2488 A.md",
                Outcome::Pass,
            ),
            ("life:keeper", "layers/life/keeper/2500 C.md", Outcome::Pass),
        ] {
            m.offer(with_work(target, path), 1);
            m.collect("wren", &facts);
            m.report("wren", outcome, "reported", None);
        }
        // A was written, then revised by its review: it has had its second
        // reading and leaves the line. The failed story never joined it.
        assert_eq!(m.written(), ["layers/life/keeper/2500 C.md"]);
        assert_eq!(
            m.targets()["life:keeper"].writes.as_deref(),
            Some("layers/life/keeper/2500 C.md")
        );
        // A document lined up by hand joins the end, once.
        assert!(m.review_later("layers/stories/old.md"));
        assert!(!m.review_later("layers/stories/old.md"));
        assert_eq!(m.written().last().unwrap(), "layers/stories/old.md");
    }

    /// **Stuck is retried, but not for ever**; nothing-to-do holds until the
    /// target changes; a called-off mission frees its target; and discarding the
    /// pool releases every target in it.
    #[test]
    fn stuck_nothing_cancel_and_discard_each_settle_a_target_their_own_way() {
        let mut m = Missions::default();
        let facts = Facts::default();
        for round in 1..=2 {
            m.offer(generated("pair:a|b"), 3);
            m.collect("wren", &facts);
            m.report("wren", Outcome::Fail, "no way to the desk", None);
            assert_eq!(m.targets()["pair:a|b"].stuck, round);
        }
        assert!(m.blocks("pair:a|b", 3), "stuck twice on the same text");
        assert!(!m.blocks("pair:a|b", 4));

        m.decline("pair:c|d", "boundaries", 5, "They agree.");
        assert!(m.blocks("pair:c|d", 5));
        assert!(!m.blocks("pair:c|d", 6));

        m.offer(generated("era:x"), 1);
        m.collect("pax", &facts);
        assert!(m.cancel("pax"));
        assert!(!m.blocks("era:x", 1), "called off is not settled");

        m.offer(generated("era:y"), 1);
        m.offer(generated("era:z"), 1);
        assert_eq!(m.discard_pool(), 2);
        assert!(!m.blocks("era:y", 1));

        // Forgetting the settled leaves what is in hand.
        m.offer(generated("era:w"), 1);
        assert_eq!(m.forget_settled(), 2, "the stuck pair and the declined one");
        assert!(m.blocks("era:w", 1));
        assert!(!m.blocks("pair:a|b", 3));
        assert_eq!(m.next_draw(), 0);
        assert_eq!(m.next_draw(), 1);
    }

    /// **A reserved target is held against a second generation, and let go
    /// when nothing comes of it**; an operator's mission put over a generated
    /// one releases the generated one's target.
    #[test]
    fn reserving_and_replacing_hold_and_release_targets() {
        let mut m = Missions::default();
        assert!(m.reserve("life:keeper", "lives", 1));
        assert!(!m.reserve("life:keeper", "lives", 1), "held");
        m.release("life:keeper");
        assert!(m.reserve("life:keeper", "lives", 1), "released");
        m.offer(generated("life:keeper"), 1);
        m.release("life:keeper");
        assert!(m.blocks("life:keeper", 1), "offered is not released");

        m.collect("wren", &Facts::default());
        assert_eq!(m.targets()["life:keeper"].state, Settled::Carried);
        m.assign("wren", a_mission("an operator's"));
        assert!(!m.blocks("life:keeper", 1), "replaced, so released");
    }

    /// **A stuck target released after a failed generation stays stuck**, its
    /// count kept, so it still reaches the limit.
    #[test]
    fn releasing_a_stuck_target_keeps_its_count() {
        let mut m = Missions::default();
        let facts = Facts::default();
        m.offer(generated("pair:a|b"), 3);
        m.collect("wren", &facts);
        m.report("wren", Outcome::Fail, "no way to the desk", None);
        assert!(
            m.reserve("pair:a|b", "boundaries", 3),
            "stuck once is retried"
        );
        m.release("pair:a|b");
        assert_eq!(m.targets()["pair:a|b"].state, Settled::Stuck);
        assert_eq!(m.targets()["pair:a|b"].stuck, 1);

        m.offer(generated("pair:a|b"), 3);
        m.collect("wren", &facts);
        m.report("wren", Outcome::Fail, "no way to the desk", None);
        assert!(
            m.blocks("pair:a|b", 3),
            "the second stuck reaches the limit"
        );
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
