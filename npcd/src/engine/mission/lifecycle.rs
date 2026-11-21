//! The mission's own state: its steps, what signs each one off, and when it may be
//! reported.
//!
//! A step is signed off by a fact the engine can see, such as a journey landing in
//! the named room, a reading of a machine the body stood beside, or a document
//! committed, and never by the character's word alone. A critical step waits for an
//! authorizer. A mission is not reported done while a step that must come first is
//! still undone. Filing the report closes the mission and achieves its report step
//! whatever the verdict. A refused document puts its writing back in front of the
//! carrier, and a stuck report is believed only once the carrier has actually tried.

use super::*;

impl Mission {
    /// A fresh open mission with the given ask and steps.
    pub fn new(prompt: impl Into<String>, todo: Vec<Todo>, origin: Origin) -> Self {
        Self {
            prompt: prompt.into(),
            todo,
            answer: None,
            report: None,
            origin,
            observed: Vec::new(),
            stuck_refused: 0,
            refused_since: false,
            stuck_untried: 0,
            faulted_draft: None,
            gate_refusal: None,
            work: None,
            year: None,
            grants: Vec::new(),
            agendum: None,
            task: None,
        }
    }

    /// Sign off every open step that sees the mission's task through, because
    /// the task reported done. Returns whether anything was signed off.
    pub fn task_done(&mut self) -> bool {
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if is_task_step(&step.text) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// The same mission, on an agenda as `agendum`.
    pub fn on_agenda(mut self, agendum: Agendum) -> Self {
        self.agendum = Some(agendum);
        self
    }

    /// The same mission, about `work`.
    pub fn with_work(mut self, work: Work) -> Self {
        self.work = Some(work);
        self
    }

    /// The same mission, granting `grants` in its room.
    pub fn granting(mut self, grants: Vec<String>) -> Self {
        self.grants = grants;
        self
    }

    /// The room this mission is for — `area/node` — when it is a standing
    /// mission's.
    pub fn room(&self) -> Option<&str> {
        match &self.origin {
            Origin::Standing { room, .. } => Some(room),
            _ => None,
        }
    }

    /// The operation this mission carries a step of, and which step — `None`
    /// for a mission that is not the table's.
    pub fn operation(&self) -> Option<(u64, &str)> {
        match &self.origin {
            Origin::Generated {
                operation, step, ..
            } => Some((*operation, step.as_str())),
            _ => None,
        }
    }

    /// Whether this mission's step may send the work back with
    /// `report_rejected`.
    pub fn may_reject(&self) -> bool {
        self.work
            .as_ref()
            .is_some_and(|w| w.tools.iter().any(|t| t == "report_rejected"))
    }

    /// Whether its document's text is given whole in the brief, to be written
    /// as given — its step adds `file_write` — rather than composed: a mission
    /// template, whose bytes the order-giver wrote.
    pub fn writes_given(&self) -> bool {
        self.work
            .as_ref()
            .is_some_and(|w| w.tools.iter().any(|t| t == "file_write"))
    }

    /// Sign off every open step that reads `path`, because the body just read
    /// it. Returns whether anything was signed off.
    ///
    /// **A read is a fact, not a claim**, the same as where a body stands: the
    /// bench served the document.
    pub fn read_doc(&mut self, path: &str) -> bool {
        let path = plain_path(path);
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if reading_doc(&step.text).is_some_and(|p| p == path) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// Strike every open step that reads a document `holds` says the record no
    /// longer has, and drop it from the documents the mission reads. Returns the
    /// paths struck, as the steps spelled them.
    ///
    /// **A read of nothing is not a step.** A failed draft is moved out of the
    /// record, and missions written while it stood still listed it to read: two
    /// Makers were sent to a desk to read a page that was gone, were told there
    /// was no document there, and were sent back to read it — forty minutes of
    /// "I am done" and no story, the step the engine signs off on a read the
    /// one step nothing could ever sign off.
    pub fn strike_gone_reads(&mut self, holds: &dyn Fn(Option<u64>, &str) -> bool) -> Vec<String> {
        let op = self.operation().map(|(id, _)| id);
        let gone: Vec<String> = self
            .todo
            .iter()
            .filter(|t| !t.done && !t.reports && !t.critical)
            .filter_map(|t| raw_doc_after(&t.text, "read "))
            .filter(|p| !holds(op, p))
            .collect();
        if gone.is_empty() {
            return gone;
        }
        let plain: Vec<String> = gone.iter().map(|p| plain_path(p)).collect();
        self.todo
            .retain(|t| t.done || reading_doc(&t.text).is_none_or(|p| !plain.contains(&p)));
        if let Some(work) = &mut self.work {
            work.reads.retain(|r| !plain.contains(&plain_path(r)));
        }
        gone
    }

    /// Whether the mission's document has been committed while it was carried
    /// — its write step signed off as achieved, which nothing but a commit of
    /// that document does ([`Self::committed`]). `true` for a mission with no
    /// document to write.
    ///
    /// **The mission's own record, not the bench's.** Asking the bench who last
    /// committed the path passed a review whose Maker had written the document
    /// before it was ever reviewed, and failed every mission after a restart,
    /// because the bench's record of hands is not kept and the mission is.
    pub fn written_up(&self) -> bool {
        let Some(work) = &self.work else {
            return true;
        };
        if work.edit_optional {
            return true;
        }
        let path = plain_path(&work.writes);
        self.todo.iter().any(|t| {
            t.done
                && t.outcome == Some(StepOutcome::Achieved)
                && writing_doc(&t.text).is_some_and(|p| p == path)
        })
    }

    /// Put the writing of `path` back in front of the carrier, because the
    /// document as committed does not stand: its writing step, done, is to do
    /// again, and a mission that had none — a review whose reading found it
    /// sound, a check against the storyline — gains one before its report.
    ///
    /// **A refused document is work still to do, and the steps must say so.**
    /// With every step ticked, a Maker whose draft the gate refused was told in
    /// one breath to write it again and that its work was done and to report it
    /// now — at the command table, two levels from the only desk it could write
    /// at. It reported, was refused, and was told the same again, until four
    /// Makers sat on a bench telling each other there was nothing left to do.
    pub fn reopen_write(&mut self, path: &str) {
        let plain = plain_path(path);
        let mut found = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| writing_doc(&t.text).is_some_and(|p| p == plain))
        {
            step.done = false;
            step.outcome = None;
            found = true;
        }
        if !found {
            let at = self
                .todo
                .iter()
                .position(|t| t.reports)
                .unwrap_or(self.todo.len());
            self.todo
                .insert(at, Todo::new(format!("change {path} and commit it")));
        }
    }

    /// Work in `year` from now on, because the carrier set a time machine to
    /// it, and sign off every open step that asks for that year. Returns whether
    /// anything was signed off.
    pub fn travelled(&mut self, year: u32) -> bool {
        self.year = Some(year);
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if time_step(&step.text) == Some(year) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// Sign off every open step that writes one of `paths`, because a commit
    /// just wrote them. Returns whether anything was signed off.
    pub fn committed(&mut self, paths: &[String]) -> bool {
        let paths: Vec<String> = paths.iter().map(|p| plain_path(p)).collect();
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if writing_doc(&step.text).is_some_and(|p| paths.contains(&p)) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// Count a `report_stuck` the desk turned away because the step was plainly
    /// doable, and say how many there have been.
    pub fn refuse_stuck(&mut self) -> u32 {
        self.stuck_refused += 1;
        self.stuck_refused
    }

    /// Turn away a `report_stuck` whose way the world can see is open, and say
    /// how many such turn-aways count. **Only a real one counts**: one after
    /// which the world refused the carrier something it tried. A Maker gave two
    /// operations up as stuck at an open lift — "no mechanism exists to access
    /// other levels" — having walked between the lift and the table and never
    /// once called `lift_use`; two turn-aways and the third report was
    /// believed. Told the way and not trying it is not being blocked by it.
    ///
    /// **But a carrier that will not try it is not going to.** A Drone stood
    /// between the lift and the table for two hours, turned away every time
    /// and never trying, and held its place on the card for all of it. Every
    /// [`UNTRIED_TURN_AWAYS`] in a row with nothing tried count as one, so the
    /// report is believed in the end and the mission fails.
    pub fn refuse_stuck_on_the_way(&mut self) -> u32 {
        if std::mem::take(&mut self.refused_since) {
            self.stuck_refused += 1;
            self.stuck_untried = 0;
        } else {
            self.stuck_untried += 1;
            if self.stuck_untried == UNTRIED_TURN_AWAYS {
                self.stuck_refused += 1;
                self.stuck_untried = 0;
            }
        }
        self.stuck_refused
    }

    /// Turn away a `report_stuck` over a document whose faults were put in front
    /// of the carrier, `draft` the digest of the document as it stands, and say
    /// how many such turn-aways count. **One counts only after a rewrite**: the
    /// document must differ from what it was at the last turn-away. A Maker shown
    /// "the brief's own words are still in the text" quoted the fault back as its
    /// reason for being stuck, rewrote nothing, and the third report failed the
    /// operation. Rewritten and still refused is a real block; unrewritten is not.
    pub fn refuse_stuck_over_faults(&mut self, draft: u64) -> u32 {
        if self.faulted_draft != Some(draft) {
            self.stuck_refused += 1;
        }
        self.faulted_draft = Some(draft);
        self.stuck_refused
    }

    /// The world refused the carrier an act while this mission was open.
    pub fn refused_an_act(&mut self) {
        self.refused_since = true;
    }

    /// Note something the engine saw happen toward the mission. A line already
    /// noted is not noted twice.
    pub fn observe(&mut self, line: impl Into<String>) {
        let line = line.into();
        if !self.observed.contains(&line) {
            self.observed.push(line);
        }
    }

    /// Open until a report is filed. A character with an open mission reads it in
    /// its prompt; a closed one is retired.
    pub fn is_open(&self) -> bool {
        self.report.is_none()
    }

    /// Every step ticked off. Not the same as complete — the character still
    /// files the report — but the signal that it is ready to.
    pub fn all_todos_done(&self) -> bool {
        !self.todo.is_empty() && self.todo.iter().all(|t| t.done)
    }

    /// Sign off the first open step whose text matches `which` (trimmed,
    /// case-insensitive) with how it turned out. Returns whether a step was
    /// signed off — `false` when nothing matched or every match was already
    /// done, so the caller can tell the character its instruction landed on
    /// nothing.
    pub fn check_off(&mut self, which: &str, outcome: StepOutcome) -> bool {
        let want = which.trim().to_lowercase();
        for step in &mut self.todo {
            if !step.done && !step.critical && step.text.trim().to_lowercase() == want {
                step.done = true;
                step.outcome = Some(outcome);
                return true;
            }
        }
        false
    }

    /// The current step's text when it is a critical one — what the carrier may
    /// not do until an authorizer approves. `None` when the next step is
    /// ordinary or there is none. The critical gate reads this to know when to
    /// seek a go-ahead instead of letting the step be done.
    pub fn critical_next(&self) -> Option<&str> {
        self.todo
            .iter()
            .find(|t| !t.done)
            .filter(|t| t.critical)
            .map(|t| t.text.as_str())
    }

    /// Sign off the current step because an authorizer has approved it — the
    /// only thing that closes a `critical` step (nothing the engine sees and no
    /// `check_off` on the carrier's word does). Signs off the first open step
    /// when it is critical, and says whether it did.
    pub fn authorize(&mut self) -> bool {
        match self.todo.iter_mut().find(|t| !t.done) {
            Some(step) if step.critical => {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                true
            }
            _ => false,
        }
    }

    /// Add a step. A character adds one when it discovers work the mission did
    /// not spell out. Returns whether it was added — `false` for a blank step or
    /// an exact duplicate of an existing (still-open) one, so a restated
    /// instruction does not grow the list.
    pub fn add_todo(&mut self, text: &str) -> bool {
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        let key = text.to_lowercase();
        if self
            .todo
            .iter()
            .any(|t| !t.done && t.text.trim().to_lowercase() == key)
        {
            return false;
        }
        self.todo.push(Todo::new(text));
        true
    }

    /// Sign off every open step that is a journey to `room` on `level`, because
    /// the body is standing in it. Returns whether anything was signed off.
    ///
    /// **Where a body stands is a fact, not a claim.** A step like "go to the
    /// plant room" was left for the guardian to tick on the character's own word,
    /// over several confirmations — so a character that had walked there read
    /// `[ ] go to the plant room` on arrival, took the list to mean it had not
    /// begun, and walked back to the table to start again. The engine knows
    /// where the body is; it signs the journey off itself.
    ///
    /// **A step that names its level holds to it.** A step to write on the
    /// casting level says "go to band one on the casting level", and a room of
    /// the same name on any other level does not sign it off.
    pub fn arrived_in(&mut self, room: &str, level: &str) -> bool {
        let (room, level) = (plain(room), plain(level));
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if journey_to(&step.text).is_some_and(|to| at_place(&to, &room, &level)) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// Sign off every open step that is a reading of one of `seen` — the machines
    /// a `scan` of the room just showed the body, each with its state. Returns
    /// whether anything was signed off.
    pub fn read_off(&mut self, seen: &[String]) -> bool {
        let seen: Vec<String> = seen.iter().map(|n| plain(n)).collect();
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if reading_of(&step.text).is_some_and(|of| seen.iter().any(|n| names(&of, n))) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// Sign off every open step that asks the body to be with `who` — find
    /// them, go to them, visit them, look in on them — because it is in the
    /// same room as them. Returns whether anything was signed off.
    pub fn met(&mut self, who: &str) -> bool {
        self.tick_about(who, MEETING)
    }

    /// Sign off every open step that asks the body to be with `who` or to say
    /// something to them — ask, tell, talk to, hear them out — because a word
    /// it addressed to them landed. Returns whether anything was signed off.
    ///
    /// **Who a character spoke to is a fact, not a claim**, the same as where
    /// it stands: the world took the `ask` or the `tell`, naming them.
    pub fn spoke_with(&mut self, who: &str) -> bool {
        let met = self.tick_about(who, MEETING);
        self.tick_about(who, SPEAKING) || met
    }

    /// Sign off every open step that posts work, because the carrier's
    /// `order_post` put a mission on the table. Returns whether anything was
    /// signed off.
    pub fn posted_work(&mut self) -> bool {
        self.tick_opening(POSTING)
    }

    /// Sign off every open step that reads what came back, because the
    /// carrier's `order_results` read the reports. Returns whether anything was
    /// signed off.
    pub fn read_results(&mut self) -> bool {
        self.tick_opening(READING_RESULTS)
    }

    /// Sign off every open step that reads the carrier's messages, because a
    /// `read` took a conversation off its phone. Returns whether anything was
    /// signed off.
    pub fn read_messages(&mut self) -> bool {
        self.tick_opening(READING_MESSAGES)
    }

    /// Sign off every open step that connects to the terminal called
    /// `terminal`, because the carrier worked it: `connect to the cluster and
    /// read its status` is done once an act of the cluster's landed. Connecting
    /// alone signs nothing off — the step is the look, not the door. Returns
    /// whether anything was signed off.
    pub fn worked_terminal(&mut self, terminal: &str) -> bool {
        let name = plain(terminal);
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if connecting_to(&step.text).is_some_and(|to| names(&to, &name)) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// Sign off every open step that opens with `opening`.
    pub(super) fn tick_opening(&mut self, opening: &str) -> bool {
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            if step.text.trim().to_lowercase().starts_with(opening) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    pub(super) fn tick_about(&mut self, who: &str, verbs: &[&str]) -> bool {
        let mut ticked = false;
        for step in self
            .todo
            .iter_mut()
            .filter(|t| !t.done && !t.reports && !t.critical)
        {
            let text = step.text.trim().to_lowercase();
            if verbs.iter().any(|v| text.starts_with(v)) && mentions(&text, who) {
                step.done = true;
                step.outcome = Some(StepOutcome::Achieved);
                ticked = true;
            }
        }
        ticked
    }

    /// The first step not yet signed off, if any.
    pub fn next_step(&self) -> Option<&Todo> {
        self.todo.iter().find(|t| !t.done)
    }

    /// The step the world can see still to do: [`Self::next_step`], or — when
    /// that is one the body settles in its own mind ([`is_judgement`]), which
    /// nothing signs off — the first open step after it that is not, short of
    /// the report.
    ///
    /// **A judgement does not hide what follows it.** Checked against the next
    /// step alone, `report_done` took a mission whose "decide what they need"
    /// stood in front of a journey never made.
    pub fn step_to_do(&self) -> Option<&Todo> {
        match self.next_step() {
            Some(t) if is_judgement(&t.text) => self
                .todo
                .iter()
                .find(|t| !t.done && !t.reports && !is_judgement(&t.text)),
            next => next,
        }
    }

    /// The first open step that is a reading of one of `machines` the body has
    /// not made, if any.
    ///
    /// What `report_done` is checked against: a mission whose point is to read a
    /// machine cannot be reported done by a character that never read it. Only a
    /// step the engine can see done for itself counts — a reading of a machine
    /// the world holds, which a scan of its room always shows. A step it cannot
    /// observe ("form your own view of it", or a scan of something no scan lists)
    /// never holds a report up.
    pub fn unread_step(&self, machines: &[String]) -> Option<&str> {
        let machines: Vec<String> = machines.iter().map(|n| plain(n)).collect();
        self.todo
            .iter()
            .filter(|t| !t.done && !t.reports && !t.critical)
            .find(|t| reading_of(&t.text).is_some_and(|of| machines.iter().any(|m| names(&of, m))))
            .map(|t| t.text.as_str())
    }

    /// File the completion report, closing the mission. `answer` overwrites the
    /// built answer when the character supplies a final one; a `None` keeps
    /// whatever was already built.
    ///
    /// **Filing the report is the report step.** It was left open, so every
    /// finished mission read back with its last step unticked under a filed
    /// report — a pass that looked unfinished. A report filed is a report made,
    /// whatever its verdict: the step is achieved.
    pub fn complete(&mut self, outcome: Outcome, notes: impl Into<String>, answer: Option<String>) {
        if let Some(a) = answer {
            self.answer = Some(a);
        }
        for step in self.todo.iter_mut().filter(|t| t.reports && !t.done) {
            step.done = true;
            step.outcome = Some(StepOutcome::Achieved);
        }
        self.report = Some(Report {
            outcome,
            notes: notes.into(),
        });
    }

    /// The ask itself, as [`Self::standing_text`] states it after "What has been
    /// asked of you:".
    pub fn mission_text(&self) -> String {
        self.prompt.trim().to_string()
    }

    /// The text the `task` collection member carries — the steps, read under the
    /// schema's `task_intro` heading ("What you are working through:"). Ticked
    /// steps are struck so the character sees its own progress rather than a
    /// list that never shrinks. `None` when there are no steps, so the caller
    /// installs no `task` member and `task_intro` stays hidden.
    pub fn task_text(&self) -> Option<String> {
        if self.todo.is_empty() {
            return None;
        }
        let mut lines = Vec::with_capacity(self.todo.len());
        for step in &self.todo {
            let mark = match (step.done, step.outcome) {
                (false, _) => "[ ]",
                (true, Some(StepOutcome::Thwarted)) => "[could not be done]",
                (true, _) => "[done]",
            };
            lines.push(format!("{mark} {}", step.text.trim()));
        }
        Some(lines.join("\n"))
    }

    /// The mission as a standing instruction — what a character reads each quiet
    /// turn while it carries one: the ask, the steps that see it through, and
    /// where it ends. It ends at the command table, because that is where a
    /// mission is reported and the next taken up — so the loop closes on a report
    /// rather than trailing off. A character judges for itself when the work is
    /// done; the steps are the shape of it, not a checklist it ticks.
    pub fn standing_text(&self) -> String {
        format!("What has been asked of you: {}", self.prompt_text())
    }

    /// The mission as the section a character's system prompt carries under
    /// "What has been asked of you:" — the same words as [`Self::standing_text`]
    /// without the lead-in, which the prompt's own heading supplies.
    pub fn prompt_text(&self) -> String {
        let mut out = self.mission_text();
        if let Some(tasks) = self.task_text() {
            out.push_str("\nThe steps that see it through:\n");
            out.push_str(&tasks);
        }
        // **A mission of an agenda is closed by what is seen done**, not by a
        // report at the table (`docs/zen_moon_mind.md` §17.5).
        out.push_str(match self.agendum {
            Some(_) => {
                "\nCarry it out. It is done when every step is seen done. If it is finished in \
                 a way no step shows, close it with `mission_done`; if it has stopped being \
                 worth doing, drop it with `mission_cancel`."
            }
            None => {
                "\nCarry it out. When it is done, go back to the table where work is handed out, \
                 `scan` it, and report on your effector device — `invoke` its `report_done` with \
                 what you found, or its `report_stuck` with what stopped you if it cannot be \
                 finished — and take up the next."
            }
        });
        out
    }

    /// [`Self::prompt_text`] with the fingerprint that says whether the section
    /// already sealed for a conversation still matches it.
    pub fn prompt(&self) -> MissionPrompt {
        MissionPrompt {
            text: self.prompt_text(),
            fingerprint: self.content_fingerprint(),
        }
    }

    /// A stable content fingerprint of what the character reads — the ask plus
    /// the steps and their ticks. Two missions that render the same text hash
    /// the same, so a change that does not alter the rendered prompt does not
    /// force a re-seal of the dynamic section, and one that does gets a new id.
    pub fn content_fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.mission_text().hash(&mut h);
        for step in &self.todo {
            step.text.trim().hash(&mut h);
            step.done.hash(&mut h);
            step.outcome.map(StepOutcome::as_str).hash(&mut h);
        }
        h.finish()
    }
}

/// How many `report_stuck` in a row, turned away with the way open and nothing
/// tried, count as one real turn-away — see
/// [`Mission::refuse_stuck_on_the_way`].
pub const UNTRIED_TURN_AWAYS: u32 = 3;
