//! Tracks which mission each character is carrying, and signs off its steps as
//! the world and its reports show them done.
//!
//! A body carries one open mission at a time. What the world can see is signed
//! off as it happens, so a journey, a reading or a write closes its step at once.
//! A step the engine cannot observe is judged only through the agenda or a
//! report. A body whose mission is called off, or which is stood down, is told so,
//! since a maker left working on a mission that no longer exists would have no way
//! to know. A body made unordered is given no work and is stood down from what it
//! carries.

use super::*;

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

    /// Mark the body unordered, or not: **the world gives an unordered
    /// character no orders.** The table never hands it work, nothing is
    /// lodged for it, and its
    /// idle wake sets it nothing — what it does with its time is its own. Made
    /// unordered, it is stood down from whatever it carries, and told so, and
    /// what was lodged for it is dropped. `docs/mission_generator.md`, "A
    /// character that takes no orders".
    pub fn set_unordered(&mut self, body: &str, unordered: bool) {
        if !unordered {
            self.unordered.remove(body);
            return;
        }
        if self.cancel(body) {
            self.stood_down.push(body.to_string());
        }
        self.lodged.remove(body);
        self.unordered.insert(body.to_string());
    }

    /// Whether the world gives this body no orders — see
    /// [`Self::set_unordered`].
    pub fn is_unordered(&self, body: &str) -> bool {
        self.unordered.contains(body)
    }

    /// Hold exactly `suspended` as the bodies whose characters are suspended.
    /// Returns whether that changed anything.
    pub fn set_suspended(&mut self, suspended: BTreeSet<String>) -> bool {
        let changed = self.suspended != suspended;
        self.suspended = suspended;
        changed
    }

    /// The bodies whose characters are suspended, as last told.
    pub fn suspended(&self) -> &BTreeSet<String> {
        &self.suspended
    }

    /// Whether `body` is a hand the table can give work to: it takes orders
    /// and its character is not suspended. One at rest is a hand — work wakes
    /// it (`docs/zen_everywhere.md` §3).
    pub fn is_hand(&self, body: &str) -> bool {
        !self.is_unordered(body) && !self.suspended.contains(body)
    }

    /// Whether `body` is a drone at rest in a world whose table is open: it
    /// takes orders, carries nothing, the world draws it no routine, and
    /// nothing on the table is its to take.
    pub fn at_rest(&self, body: &str) -> bool {
        !self.is_unordered(body)
            && !self.is_on_mission(body)
            && !self.draws()
            && !self.has_work_for(body)
    }

    /// Rewrite the mission `body` carries — an ordered body's, through a
    /// conversation with its owner or an authorizer (§7.1): a new brief, and
    /// new steps that keep every one already done and every critical one.
    /// Refused for a mission that carries an operation, whose steps are the
    /// operation's.
    pub fn edit_active(
        &mut self,
        body: &str,
        brief: Option<String>,
        steps: Option<Vec<String>>,
    ) -> Result<(), String> {
        let m = self
            .active
            .get_mut(body)
            .ok_or_else(|| "You are carrying no mission to change.".to_string())?;
        if m.operation().is_some() {
            return Err(
                "This mission's steps are its operation's: it is changed at the table, not here."
                    .into(),
            );
        }
        if let Some(brief) = brief.filter(|b| !b.trim().is_empty()) {
            m.prompt = brief;
        }
        if let Some(steps) = steps {
            m.todo = kept_steps(&m.todo, steps);
        }
        Ok(())
    }

    /// Sign off the body's open steps that post work — its `order_post`
    /// landed. Returns whether anything was signed off.
    pub fn posted_work(&mut self, body: &str) -> bool {
        self.active.get_mut(body).is_some_and(Mission::posted_work)
    }

    /// Sign off the body's open steps that read what came back — its
    /// `order_results` landed. Returns whether anything was signed off.
    pub fn read_results(&mut self, body: &str) -> bool {
        self.active.get_mut(body).is_some_and(Mission::read_results)
    }

    /// Sign off the body's open steps that read its messages — a `read` of
    /// its phone landed. Returns whether anything was signed off.
    pub fn read_messages(&mut self, body: &str) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(Mission::read_messages)
    }

    /// Sign off the steps of the body's open mission that connect to the
    /// terminal it has just worked — see [`Mission::worked_terminal`].
    pub fn worked_terminal(&mut self, body: &str, terminal: &str) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.worked_terminal(terminal))
    }

    /// The body's open mission, if it is on one.
    pub fn active(&self, body: &str) -> Option<&Mission> {
        self.active.get(body)
    }

    /// Every body carrying a mission, with it.
    pub fn carried(&self) -> impl Iterator<Item = (&str, &Mission)> {
        self.active.iter().map(|(b, m)| (b.as_str(), m))
    }

    /// The body's open mission, to change — what the critical gate signs off
    /// when an authorizer's go-ahead lands.
    pub fn active_mut(&mut self, body: &str) -> Option<&mut Mission> {
        self.active.get_mut(body)
    }

    /// Whether the body is carrying an open mission right now.
    pub fn is_on_mission(&self, body: &str) -> bool {
        self.active.contains_key(body)
    }

    /// Give the body a mission to carry now, replacing any open one it held. A
    /// generated mission replaced is called off, and its target released to be
    /// found again — left marked as carried, nothing would ever set it again.
    pub fn assign(&mut self, body: &str, mission: Mission) {
        if let Some(old) = self.active.insert(body.to_string(), mission) {
            self.called_off(body, &old, CALLED_OFF);
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

    /// Keep `task` as the one the body's open mission drives. `false` when it
    /// has no open mission.
    pub fn set_task(&mut self, body: &str, task: Task) -> bool {
        match self.active.get_mut(body) {
            Some(m) => {
                m.task = Some(task);
                true
            }
            None => false,
        }
    }

    /// Note whether the body's mission's task was found running, asking
    /// nothing — [`Task::running`].
    pub fn task_running(&mut self, body: &str, running: bool) {
        if let Some(t) = self.active.get_mut(body).and_then(|m| m.task.as_mut()) {
            t.running = running;
        }
    }

    /// Whether the body's mission waits on its task: started, and running
    /// with nothing asked of the body — so the body rests until it changes.
    pub fn waits_on_task(&self, body: &str) -> bool {
        self.active
            .get(body)
            .and_then(|m| m.task.as_ref())
            .is_some_and(|t| t.running)
    }

    /// Whether what `digest` names is news from the body's mission's task:
    /// `false` when the news last passed on said it too, or there is no task.
    /// Nothing is noted — [`Self::heard`] notes it once it is passed on.
    pub fn is_news(&self, body: &str, digest: u64) -> bool {
        self.active
            .get(body)
            .and_then(|m| m.task.as_ref())
            .is_some_and(|t| t.heard != digest)
    }

    /// Note that the body's mission's task said what `digest` names, and say
    /// whether that is news: `false` when it said it last time too, or there
    /// is no task.
    pub fn heard(&mut self, body: &str, digest: u64) -> bool {
        match self.active.get_mut(body).and_then(|m| m.task.as_mut()) {
            Some(t) if t.heard != digest => {
                t.heard = digest;
                true
            }
            _ => false,
        }
    }

    /// Sign off the steps that see the body's mission's task through — see
    /// [`Mission::task_done`].
    pub fn task_done(&mut self, body: &str) -> bool {
        self.active.get_mut(body).is_some_and(Mission::task_done)
    }

    /// Put the writing of the body's mission document back in front of it —
    /// see [`Mission::reopen_write`].
    /// What the table said when it last refused the body's mission document,
    /// or `None` once a report of it is taken — see `Mission::gate_refusal`.
    pub fn set_gate_refusal(&mut self, body: &str, refusal: Option<String>) {
        if let Some(m) = self.active.get_mut(body) {
            m.gate_refusal = refusal;
        }
    }

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

    /// Count a `report_stuck` turned away while its way is open, when it is a
    /// real one — see [`Mission::refuse_stuck_on_the_way`]. `0` with no open
    /// mission.
    pub fn refuse_stuck_on_the_way(&mut self, body: &str) -> u32 {
        self.active
            .get_mut(body)
            .map_or(0, Mission::refuse_stuck_on_the_way)
    }

    /// Count a `report_stuck` turned away over its document's faults, when the
    /// document was rewritten since the last — see
    /// [`Mission::refuse_stuck_over_faults`]. `0` with no open mission.
    pub fn refuse_stuck_over_faults(&mut self, body: &str, draft: u64) -> u32 {
        self.active
            .get_mut(body)
            .map_or(0, |m| m.refuse_stuck_over_faults(draft))
    }

    /// The world refused `body` an act; noted on its open mission, if any.
    pub fn refused_an_act(&mut self, body: &str) {
        if let Some(m) = self.active.get_mut(body) {
            m.refused_an_act();
        }
    }

    /// Add a step to the body's open mission. `false` when it has no open
    /// mission, or the step was blank or a duplicate — see [`Mission::add_todo`].
    pub fn add_todo(&mut self, body: &str, step: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.add_todo(step))
    }
}
