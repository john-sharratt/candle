//! A mission: what has been asked of a character, the steps it works through,
//! and the report it files when it is done.
//!
//! `docs/npcd_worlds_and_layers.md` describes the projection a character reads;
//! the schema (`D:/prog/mind/projection.yaml`) declares `mission` and `task`
//! collections with the gating that shows `mission_intro` when a mission is
//! carried and the "nothing has been asked of you" standing instruction when
//! none is. The wording is carried in the system prompt as a per-conversation
//! section the mind reconciles against [`Mission::prompt`] each turn
//! (`engine::mind`), so the character reads it as part of who it is rather than
//! as a restated instruction; [`Mission::standing_text`] is the same words as the
//! brief handed back at the desk. This module is the mission itself — deliberately
//! pure, so its shape, its steps and the text it renders can be tested without a
//! running engine.
//!
//! A mission is **collected** at the command desk (a lodged one if any is
//! waiting for this character, otherwise a random routine from [`bank`]),
//! **worked** by ticking its todo items off and adding more, and **completed**
//! by filing a [`Report`] — the character's own pass/fail judgement and answer.
//! A mission with a report is done; one without is open.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// One step of a mission. `done` flips when the character ticks it off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Todo {
    /// What the step is, in the character's own second person ("find X").
    pub text: String,
    /// Whether it has been signed off, one way or the other.
    pub done: bool,
    /// How the step turned out once signed off; `None` while it is open. A step
    /// signed off before outcomes were recorded reads as achieved.
    #[serde(default)]
    pub outcome: Option<StepOutcome>,
    /// Whether the step is the report itself. Nothing but filing the report
    /// closes it, so no observer ticks it on the character's word.
    #[serde(default)]
    pub reports: bool,
}

/// How a signed-off step turned out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepOutcome {
    /// The step was carried out.
    Achieved,
    /// The character tried and could not carry it out.
    Thwarted,
}

impl StepOutcome {
    /// The word the API and the guardian's log carry.
    pub fn as_str(self) -> &'static str {
        match self {
            StepOutcome::Achieved => "achieved",
            StepOutcome::Thwarted => "thwarted",
        }
    }
}

impl Todo {
    /// A fresh, un-ticked step.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            done: false,
            outcome: None,
            reports: false,
        }
    }

    /// A fresh, un-ticked step that is the report back.
    pub fn report(text: impl Into<String>) -> Self {
        Self {
            reports: true,
            ..Self::new(text)
        }
    }
}

/// How a completed mission turned out — the character's own verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The character judged the mission met.
    Pass,
    /// The character judged the mission not met.
    Fail,
}

impl Outcome {
    /// The word an operator or the API reads back.
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::Fail => "fail",
        }
    }
}

/// The completion report a character files at the command desk: its verdict and
/// what it wants to say about how the mission went.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    /// Pass or fail, the character's own call.
    pub outcome: Outcome,
    /// The character's account of what happened — free prose.
    pub notes: String,
}

/// Where a mission came from. A lodged mission names who asked; a random one
/// names which routine it is (from [`bank`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Origin {
    /// Lodged through the API by an operator (`by` is the account handle).
    Lodged { by: String },
    /// A routine drawn from the bank when nothing was lodged (`routine` names it).
    Random { routine: String },
    /// Written by the command table for one step of an operation's workflow
    /// (`generator` names which of the configured generators found the work,
    /// `target` the piece of the corpus it is about, `operation` the operation
    /// it belongs to, `step` the workflow step it carries) — see
    /// `engine::mission_gen`, `engine::workflow` and `sim::operations`.
    Generated {
        generator: String,
        target: String,
        #[serde(default)]
        operation: u64,
        #[serde(default)]
        step: String,
    },
}

/// The documents a mission is about: the one it is to produce or change, and the
/// ones to read first.
///
/// **What makes a mission checkable by something other than the character.** A
/// mission that writes the record is done when the record says so: `writes` was
/// committed, by this character, after it took the mission up. The report is
/// refused until then — see `engine::work`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    /// The mind path of the document to write or change.
    pub writes: String,
    /// The mind paths of the documents to read before writing, in order.
    pub reads: Vec<String>,
    /// The fewest words the written document may hold when it is committed —
    /// a floor well under what the brief asks for, below which it is a sketch
    /// of the work rather than the work. Zero for none.
    #[serde(default)]
    pub min_words: usize,
    /// Whether the document may be left as it is. A review may mend what it
    /// reads or find nothing to mend; a draft must write.
    #[serde(default)]
    pub edit_optional: bool,
    /// Whether the document is to be written anew rather than mended: a review
    /// of a draft the table failed. Sitting down to write it, the writer is not
    /// shown the failed text — see `engine::compose`.
    #[serde(default)]
    pub anew: bool,
    /// The checks the document is held to before the mission may be reported
    /// done — the names in its workflow step's `checks` (see
    /// `mission_gen::gates::CHECKS`).
    #[serde(default)]
    pub checks: Vec<String>,
    /// The acts its workflow step adds to the Maker's own — `report_rejected`
    /// for a step that may send the work back.
    #[serde(default)]
    pub tools: Vec<String>,
}

/// What a character's system prompt carries of its mission: the section text and
/// the fingerprint of what produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionPrompt {
    pub text: String,
    pub fingerprint: u64,
}

/// A mission held on a character: the ask, the steps, the answer it is building,
/// and the report that closes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mission {
    /// What has been asked of the character — the present-tense brief it reads.
    pub prompt: String,
    /// The steps it is working through, in order.
    pub todo: Vec<Todo>,
    /// What the mission asked it to produce, filled in as it goes. `None` until
    /// the character has something to submit.
    pub answer: Option<String>,
    /// The completion report. `None` while the mission is open; `Some` closes it.
    pub report: Option<Report>,
    /// Where the mission came from.
    pub origin: Origin,
    /// What the engine itself saw happen toward the mission — a machine's state
    /// when the character stood beside it, who it spoke to — in the order it
    /// happened. The character's answer is its own account; this is the record
    /// to check that account against.
    #[serde(default)]
    pub observed: Vec<String>,
    /// How many times `report_stuck` was turned away while the next step was
    /// plainly within reach — see `engine::work`.
    #[serde(default)]
    pub stuck_refused: u32,
    /// The documents the mission is about, when it is work on the record.
    #[serde(default)]
    pub work: Option<Work>,
    /// The year of the world's history its carrier works in, set at a time
    /// machine ([`Self::travelled`]). Nothing after it reaches the carrier's
    /// recall while the mission is carried; it ends with the mission.
    #[serde(default)]
    pub year: Option<u32>,
}

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
            work: None,
            year: None,
        }
    }

    /// The same mission, about `work`.
    pub fn with_work(mut self, work: Work) -> Self {
        self.work = Some(work);
        self
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

    /// Sign off every open step that reads `path`, because the body just read
    /// it. Returns whether anything was signed off.
    ///
    /// **A read is a fact, not a claim**, the same as where a body stands: the
    /// bench served the document.
    pub fn read_doc(&mut self, path: &str) -> bool {
        let path = plain_path(path);
        let mut ticked = false;
        for step in self.todo.iter_mut().filter(|t| !t.done && !t.reports) {
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
    pub fn strike_gone_reads(&mut self, holds: &dyn Fn(&str) -> bool) -> Vec<String> {
        let gone: Vec<String> = self
            .todo
            .iter()
            .filter(|t| !t.done && !t.reports)
            .filter_map(|t| raw_doc_after(&t.text, "read "))
            .filter(|p| !holds(p))
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
        for step in self.todo.iter_mut().filter(|t| !t.done && !t.reports) {
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
        for step in self.todo.iter_mut().filter(|t| !t.done && !t.reports) {
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
            if !step.done && step.text.trim().to_lowercase() == want {
                step.done = true;
                step.outcome = Some(outcome);
                return true;
            }
        }
        false
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
        for step in self.todo.iter_mut().filter(|t| !t.done && !t.reports) {
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
        for step in self.todo.iter_mut().filter(|t| !t.done && !t.reports) {
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

    fn tick_about(&mut self, who: &str, verbs: &[&str]) -> bool {
        let mut ticked = false;
        for step in self.todo.iter_mut().filter(|t| !t.done && !t.reports) {
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
            .filter(|t| !t.done && !t.reports)
            .find(|t| reading_of(&t.text).is_some_and(|of| machines.iter().any(|m| names(&of, m))))
            .map(|t| t.text.as_str())
    }

    /// File the completion report, closing the mission. `answer` overwrites the
    /// built answer when the character supplies a final one; a `None` keeps
    /// whatever was already built.
    pub fn complete(&mut self, outcome: Outcome, notes: impl Into<String>, answer: Option<String>) {
        if let Some(a) = answer {
            self.answer = Some(a);
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
        out.push_str(
            "\nCarry it out. When it is done, go back to the table where work is handed out, \
             `scan` it, and report on your effector device — `invoke` its `report_done` with \
             what you found, or its `report_stuck` with what stopped you if it cannot be \
             finished — and take up the next.",
        );
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

/// What a step asks of the world, as far as the engine can tell: a room to be
/// in, a machine to read, or somebody to be with or speak to. What
/// `report_stuck` is checked against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Aim {
    /// A journey: the [`plain`] name of the room.
    Room(String),
    /// A reading: what the step reads, to match against a machine's name.
    Machine(String),
    /// Being with or speaking to somebody, by the step's own words.
    Person(String),
}

impl Aim {
    /// What `step` asks for, or `None` for a step the engine cannot see.
    pub fn of(step: &str) -> Option<Aim> {
        if let Some(of) = reading_of(step) {
            return Some(Aim::Machine(of));
        }
        let lower = step.trim().to_lowercase();
        if MEETING.iter().chain(SPEAKING).any(|v| lower.starts_with(v)) {
            // A journey to a room is a meeting verb too ("go to"); the caller
            // tries the room first and the person after.
            if let Some(room) = journey_to(step) {
                return Some(Aim::Room(room));
            }
            return Some(Aim::Person(lower));
        }
        journey_to(step).map(Aim::Room)
    }

    /// Whether this aim is the room called `name` on the level called `level`.
    pub fn is_room(&self, name: &str, level: &str) -> bool {
        matches!(self, Aim::Room(room) if at_place(room, &plain(name), &plain(level)))
    }

    /// Whether this aim reads the machine called `name`.
    pub fn is_machine(&self, name: &str) -> bool {
        matches!(self, Aim::Machine(of) if names(of, &plain(name)))
    }

    /// Whether this aim is about the person called `who`.
    pub fn is_person(&self, who: &str) -> bool {
        match self {
            Aim::Person(step) | Aim::Room(step) => mentions(step, who),
            Aim::Machine(_) => false,
        }
    }
}

/// What a character is told when the engine signs a step off for it: `lead` (what
/// it has just done, "You are in the plant room"), that the step is done, and the
/// one thing to do next — the next step, or the report when that is all that is
/// left.
///
/// `here` is the machines standing where the body is. When the next step reads
/// one of them, the line says how in the grammar's own terms — `scan`, naming
/// no place — because "scan the breaker panel" read as naming the panel as the
/// place to look at, and a character standing beside it scanned another room.
pub fn progress_line(lead: &str, next: Option<&Todo>, here: &[String]) -> String {
    let here: Vec<String> = here.iter().map(|n| plain(n)).collect();
    match next {
        Some(step)
            if !step.reports
                && reading_of(&step.text).is_some_and(|of| here.iter().any(|m| names(&of, m))) =>
        {
            format!(
                "{lead}: that step of your mission is done. Next: {}. It is here, in this room: \
                 `scan` naming no place, and it lists it with the state it is in.",
                step.text.trim().trim_end_matches('.')
            )
        }
        Some(step) if !step.reports => format!(
            "{lead}: that step of your mission is done. Next: {}.",
            step.text.trim().trim_end_matches('.')
        ),
        _ => format!(
            "{lead}: that was the last step of your mission. Now go back to the table where \
             work is handed out, `scan` it, and `invoke` its `report_done` with exactly what \
             you found."
        ),
    }
}

/// The mission compass: the one next thing to do, and where whoever it names is
/// right now. `people` is everybody else in the world, each with the room they
/// are in, or `None` when they are in the body's own room.
///
/// **Where somebody is is the engine's to say.** A character sent to find
/// someone went to the room it had last seen them in, found it empty, and gave
/// the mission up as stuck — while the person stood in the next room, and then
/// walked into its own. It cannot track the cast; the world can.
///
/// Each person's `Some` is the way to them as a phrase — "the plant room:
/// `move_to` it", or the lift ride when they are on another level. `way` is the
/// way to the room or machine the step names, when it names one.
pub fn compass_line(next: &Todo, people: &[(String, Option<String>)], way: Option<&str>) -> String {
    let step = next.text.trim().trim_end_matches('.');
    let lower = step.to_lowercase();
    let mut line = format!("Your mission, next: {step}.");
    if let Some(way) = way {
        line.push(' ');
        line.push_str(way);
    }
    let speak = SPEAKING.iter().any(|v| lower.starts_with(v));
    for (who, room) in people.iter().filter(|(who, _)| mentions(&lower, who)) {
        match room {
            None if speak => line.push_str(&format!(
                " {who} is here with you: say it to them now — `ask` or `tell` them."
            )),
            None => line.push_str(&format!(" {who} is here with you.")),
            // Or by phone: somebody busy on their own errand keeps moving, and
            // a character chasing them room to room gave the mission up as
            // stuck. A `message` to them is a word to them like any other.
            Some(way) => line.push_str(&format!(
                " {who} is in {way} — or `message` them, which reaches them wherever they are."
            )),
        }
    }
    line
}

/// The addresses a document step is done through, at the desk the body stands
/// at — each `None` when no desk here offers it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeskVerbs {
    pub read: Option<String>,
    /// Where the piece is written whole — `compose`.
    pub compose: Option<String>,
    pub edit: Option<String>,
    pub commit: Option<String>,
}

/// The mission compass for a step that sets the year: the act at a time machine
/// standing here, else the way to one (`way`).
///
/// **The year is set at a machine, and the compass says which act.** Told only
/// "set your time to 2837 at a time machine", four Makers walked between the
/// lift and their rooms for an hour, told each other the lift was broken and
/// went looking for a level nothing had sent them to.
pub fn time_compass(year: u32, machine_here: bool, way: Option<&str>) -> String {
    match (machine_here, way) {
        (true, _) => format!(
            "Your mission, next: set your time to {year}. A time machine is here: `time_travel` \
             naming the year {year}."
        ),
        (false, Some(way)) => format!(
            "Your mission, next: set your time to {year}. The time machines are in {way}; there, \
             `time_travel` naming the year {year}."
        ),
        (false, None) => format!(
            "Your mission, next: set your time to {year}, at a time machine on the time level: \
             `move_to` the lift and `lift_use` naming the time level."
        ),
    }
}

/// The mission compass for a step about a document: the exact act that does it
/// when the body stands at a desk, else the way to one (`way`).
///
/// **The act, not the intention.** Told only "next: read layers/eras/…", a
/// Keeper who had reached the writing room reflected and dreamt for ten minutes
/// and never touched the desk: reading a document is an `invoke` of the desk's
/// `file_read` naming it, and nothing in what it read said so.
///
/// `written` is how many words of the document to write the working copy holds,
/// uncommitted, when that is already up to its floor.
///
/// **A written piece is committed, not written again.** Told "sit down and
/// write it whole" straight after it had composed five hundred words, a Maker
/// composed the same story five times over and never committed it.
pub fn doc_compass(
    doc: &DocStep,
    at: &DeskVerbs,
    way: Option<&str>,
    written: Option<usize>,
) -> String {
    match (doc, at, written) {
        (
            DocStep::Write(p),
            DeskVerbs {
                read: Some(read),
                commit: Some(commit),
                ..
            },
            Some(words),
        ) => format!(
            "Your mission, next: commit {p}. It is written — {words} words in your working copy. \
             Read it back with `invoke` {read} with `path` \"{p}\"; if it stands, `invoke` {commit} \
             with `why`, one line saying what it is. Put one passage right first if one is wrong, \
             but do not write it again."
        ),
        (doc, at, _) => doc_compass_unwritten(doc, at, way),
    }
}

fn doc_compass_unwritten(doc: &DocStep, at: &DeskVerbs, way: Option<&str>) -> String {
    match (doc, at) {
        (
            DocStep::Read(p),
            DeskVerbs {
                read: Some(url), ..
            },
        ) => format!(
            "Your mission, next: read {p}. `invoke` {url} with `path` \"{p}\" — it comes back a \
             page at a time."
        ),
        // **Sit down to it.** The piece is written whole with `compose`, in one
        // sitting with the brief and the sources open — see `engine::compose`;
        // a draft put together line by line between other acts carried the room
        // it was written in. An edit is for a passage the checks name.
        (
            DocStep::Write(p),
            DeskVerbs {
                compose: Some(compose),
                edit,
                commit: Some(commit),
                ..
            },
        ) => {
            let mend = match edit {
                Some(edit) => format!(
                    " To put one passage right afterwards, `invoke` {edit} with the words to \
                     replace as `old_str` and what replaces them as `new_str`."
                ),
                None => String::new(),
            };
            format!(
                "Your mission, next: write {p}. Sit down and write it whole: `invoke` {compose} — \
                 you write it through with your brief and what you read open in front of you, \
                 and it goes into your working copy.{mend} Then `invoke` {commit} with `why`, one \
                 line saying what it is. Until it is committed nobody else can see it, and the \
                 mission is not done."
            )
        }
        (doc, _) => {
            let (verb, p, desk) = match doc {
                DocStep::Read(p) => ("read", p, "Documents are read at a desk"),
                DocStep::Write(p) => ("write", p, "A piece is written at a writing desk"),
            };
            match way {
                Some(way) => {
                    format!("Your mission, next: {verb} {p}. {desk}: the nearest is in {way}.")
                }
                None => format!(
                    "Your mission, next: {verb} {p}. {desk}; at one, `scan` shows its address."
                ),
            }
        }
    }
}

/// The mission compass when the work is done and only the report is left.
/// `table` is the way to the table — its room, and the lift ride when it is on
/// another level; `at_table` whether the body is at it;
/// `seen` what the engine observed toward the mission, which is what the report
/// is to say.
///
/// **The report is about this mission.** Characters filed accounts of the
/// mission before last, or of the scan they made at the table; handed what it
/// saw on this one, in the line that tells it to report, it reports that.
///
/// **A review's report is its verdict** (`verdict`): told to report "exactly
/// what you found", a reviewer that had read the draft and corrected its title
/// stood at the table saying there was nothing substantive to report.
pub fn compass_report_line(
    table: Option<&str>,
    at_table: bool,
    seen: &[String],
    verdict: bool,
) -> String {
    let mut line = report_way(table, at_table, verdict);
    if !seen.is_empty() {
        line.push_str(&format!(" What you saw on it: {}.", seen.join("; ")));
    }
    line
}

fn report_way(table: Option<&str>, at_table: bool, verdict: bool) -> String {
    let what = match verdict {
        true => {
            "its `report_done` if the draft can stand, saying what you checked and what you \
                 changed — or its `report_rejected` if it cannot, saying why. That verdict is the \
                 whole report"
        }
        false => "its `report_done` with exactly what you found",
    };
    match (at_table, table) {
        (true, _) => format!(
            "Your mission's work is done and you are at the table: report it now — `invoke` \
             {what}."
        ),
        (false, Some(way)) => format!(
            "Your mission's work is done. Go back to the table, in {way}, and report it: \
             `invoke` {what}."
        ),
        (false, None) => format!(
            "Your mission's work is done. Go back to the table where work is handed out and \
             report it: `invoke` {what}."
        ),
    }
}

/// A name as it is compared: lower case, trimmed, with no leading article and no
/// closing full stop, so "The plant room." and "the plant room" are one room.
fn plain(name: &str) -> String {
    let lower = name.trim().trim_end_matches('.').trim().to_lowercase();
    lower
        .strip_prefix("the ")
        .map(str::to_string)
        .unwrap_or(lower)
}

/// Whether `said` — the rest of a step after its verb — names `thing`: the whole
/// of it, or it followed by more words ("the coolant valve and read its state").
fn names(said: &str, thing: &str) -> bool {
    !thing.is_empty()
        && (said == thing
            || said.starts_with(&format!("{thing} "))
            || said.starts_with(&format!("{thing},")))
}

/// Whether the engine signs `step` off itself, from what it sees: a journey, a
/// reading of a machine, somebody met or spoken to, a document read or
/// committed. Nothing else may sign such a step off on the character's word —
/// asked whether it had written a story, a character said it had tried and
/// could not, and the write was struck as thwarted with no document on the
/// record.
pub fn engine_sees(step: &str) -> bool {
    Aim::of(step).is_some()
        || reading_doc(step).is_some()
        || writing_doc(step).is_some()
        || time_step(step).is_some()
}

/// The step that sends a carrier to a time machine to work in `year`.
pub fn time_step_text(year: u32) -> String {
    format!("set your time to {year} at a time machine")
}

/// The year a [`time_step_text`] step asks for. `None` for any other step.
pub fn time_step(step: &str) -> Option<u32> {
    let rest = step.trim().strip_prefix("set your time to ")?;
    let (year, tail) = rest.split_once(' ')?;
    (tail.trim() == "at a time machine")
        .then(|| year.parse().ok())
        .flatten()
}

/// Whether `said` — a journey's [`plain`] destination — is `room` on `level`,
/// both [`plain`]. A destination that names no level is any room so called.
fn at_place(said: &str, room: &str, level: &str) -> bool {
    match split_level(said) {
        (r, Some(l)) => names(&r, room) && l == level,
        (r, None) => names(&r, room),
    }
}

/// A destination and the level it names, when it names one: "band one on the
/// casting level" → ("band one", Some("casting level")). Only a trailing "on …
/// level" is a level, so "the room on the left" stays one name.
fn split_level(said: &str) -> (String, Option<String>) {
    if let Some((room, level)) = said.rsplit_once(" on ") {
        let level = plain(level);
        if level.ends_with("level") {
            return (room.trim().to_string(), Some(level));
        }
    }
    (said.to_string(), None)
}

/// A mind path as compared: trimmed, forward slashes, lower case, no leading
/// slash — "Layers/eras/x.md" and "/layers/eras/x.md" are one document.
fn plain_path(path: &str) -> String {
    path.trim()
        .trim_matches('`')
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_lowercase()
}

/// The document a reading step reads: "read layers/eras/x.md" → its
/// [`plain_path`]. `None` for any other step, or one naming no document.
fn reading_doc(step: &str) -> Option<String> {
    doc_after(step, "read ")
}

/// The document a writing step writes: "write layers/life/x/2510 y.md and
/// commit it" → its [`plain_path`]. `None` for any other step.
fn writing_doc(step: &str) -> Option<String> {
    doc_after(step, "write ").or_else(|| doc_after(step, "change "))
}

/// The document path a step names after `verb`, up to its `.md`/`.yaml` end, as
/// [`plain_path`] compares it.
fn doc_after(step: &str, verb: &str) -> Option<String> {
    raw_doc_after(step, verb).map(|p| plain_path(&p))
}

/// The same path exactly as the step spells it — what a character is told to
/// name, so a new document is created with its title's own capitals.
fn raw_doc_after(step: &str, verb: &str) -> Option<String> {
    let rest = step.trim().strip_prefix(verb)?;
    // ASCII lowering keeps every byte where it was, so an offset found in
    // `lower` is a char boundary in `rest`; full Unicode lowering does not
    // (`İ` grows a byte) and a title holding one sliced mid-character.
    let lower = rest.to_ascii_lowercase();
    let end = [".md", ".yaml"]
        .iter()
        .filter_map(|ext| lower.find(ext).map(|at| at + ext.len()))
        .min()?;
    Some(rest[..end].trim().trim_matches('`').to_string())
}

/// A step about a document: one to read, or one to write and commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocStep {
    Read(String),
    Write(String),
}

impl DocStep {
    /// The document step `step` is, with its path as the step spells it.
    pub fn of(step: &str) -> Option<DocStep> {
        raw_doc_after(step, "read ").map(DocStep::Read).or_else(|| {
            raw_doc_after(step, "write ")
                .or_else(|| raw_doc_after(step, "change "))
                .map(DocStep::Write)
        })
    }
}

/// Where a journey step leads, as a [`plain`] name: "go to the plant room" →
/// "plant room". `None` for a step that is not a journey.
fn journey_to(step: &str) -> Option<String> {
    let step = step.trim().to_lowercase();
    [
        "go to ",
        "go back to ",
        "travel to ",
        "walk to ",
        "head to ",
        "make your way to ",
    ]
    .iter()
    .find_map(|verb| step.strip_prefix(verb))
    .map(plain)
}

/// How a step that is about being with somebody begins.
const MEETING: &[&str] = &[
    "find ",
    "go and find ",
    "go to ",
    "travel to ",
    "visit ",
    "seek out ",
    "look in on ",
    "meet ",
];

/// How a step that is about saying something to somebody begins.
const SPEAKING: &[&str] = &[
    "ask ",
    "tell ",
    "talk to ",
    "talk with ",
    "speak to ",
    "speak with ",
    "give ",
    "draw ",
    "hear ",
];

/// Whether a lower-cased step names `who` — by their whole name, or by their
/// first name as a word of its own ("ask Paxon what he found").
fn mentions(step: &str, who: &str) -> bool {
    let who = who.trim().to_lowercase();
    if who.is_empty() {
        return false;
    }
    if step.contains(&who) {
        return true;
    }
    // A first name, never an article: "the channel" once matched every step
    // that said "go to the …", and a message to the channel signed off journeys
    // nobody made.
    let first = who.split_whitespace().next().unwrap_or_default();
    !matches!(first, "the" | "a" | "an")
        && who.split_whitespace().count() > 1
        && first.len() >= 3
        && step
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .any(|word| word.trim_end_matches("'s") == first)
}

/// What a reading step reads, as a [`plain`] name: "scan the coolant valve and
/// read what state it is in" → "coolant valve and read what state it is in",
/// matched against a machine's name by [`names`]. `None` for any other step.
fn reading_of(step: &str) -> Option<String> {
    let step = step.trim().to_lowercase();
    step.strip_prefix("scan ").map(plain)
}

/// A bank of routine missions — the ones a character is given when nothing has
/// been lodged for it. Diverse, dynamic (parameterised by who and what is
/// around), and **non-destructive**: every routine only travels, asks, reads,
/// reviews or reports. None edits game content — that is what a lodged mission
/// with a terminal step is for.
pub mod bank {
    use super::{Mission, Origin, Todo};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// What the world offers a routine to be built around: who else is here to
    /// visit, and which archived records exist to consult. A routine picks from
    /// these so two characters rarely draw the identical mission.
    #[derive(Debug, Clone, Default)]
    pub struct Facts<'a> {
        /// Other characters' names, to contact or ask about.
        pub makers: &'a [String],
        /// Archived record ids, to read and check against the storyline.
        pub records: &'a [String],
        /// Machines standing in the world, to go and read. When there are any, a
        /// mission is built on one of them, because a reading is something a
        /// character can go and get and bring back as evidence.
        pub duties: &'a [Duty],
        /// The room the table where work is reported stands in, when the world
        /// has one. Every mission ends with a step that goes back to it.
        pub table: Option<&'a str>,
    }

    /// One machine to go and read: what it is called and the room it stands in,
    /// both as a character knows them.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Duty {
        pub room: String,
        pub device: String,
    }

    /// Every routine's name, in a fixed order. `random` picks one of these by
    /// seed; the name becomes the mission's [`Origin::Random`] routine tag.
    pub const ROUTINES: &[&str] = &[
        "read-a-station",
        "visit-and-review",
        "hear-them-out",
        "check-a-record",
        "read-the-canon",
        "walk-the-halls",
        "compare-notes",
        "take-stock",
        "look-in-on-someone",
        "trace-a-thread",
        "sit-with-a-story",
    ];

    /// Draw a routine mission for `facts`, chosen by `seed`. Routines whose
    /// material is absent (no other makers, no records) are skipped, so a lone
    /// character or a world with no archive still gets a mission it can act on.
    /// Falls back to the always-available "walk the halls" routine when nothing
    /// else fits.
    pub fn random(facts: &Facts, seed: u64) -> Mission {
        let mut order: Vec<usize> = (0..ROUTINES.len()).collect();
        // Deterministic shuffle by seed so a test pins the choice and two
        // successive calls on one character vary.
        order.sort_by_key(|&i| {
            let mut h = DefaultHasher::new();
            (seed, ROUTINES[i]).hash(&mut h);
            h.finish()
        });
        // A machine to read comes first when the world has one: a reading is
        // evidence that can be brought back, where a conversation is not. After
        // that, the first routine whose material is present, in shuffled order.
        // The rotation carries routines that need nobody and nothing
        // (`take-stock`, `walk-the-halls`), so a world with no other makers, no
        // archive and no machines still yields one.
        let mut mission = build("read-a-station", facts, seed)
            .or_else(|| order.iter().find_map(|&i| build(ROUTINES[i], facts, seed)))
            .unwrap_or_else(|| walk_the_halls(facts));
        mission.todo.push(Todo::report(match facts.table {
            Some(room) => format!("go back to the table in {room} and report it"),
            None => "go back to the table and report it".to_string(),
        }));
        mission
    }

    /// A pseudo-random pick from `items` by `seed`, or `None` when empty.
    fn pick<'a>(items: &'a [String], seed: u64, salt: &str) -> Option<&'a str> {
        if items.is_empty() {
            return None;
        }
        let mut h = DefaultHasher::new();
        (seed, salt).hash(&mut h);
        Some(items[(h.finish() % items.len() as u64) as usize].as_str())
    }

    /// Build one named routine, or `None` when its material is missing.
    fn build(routine: &str, facts: &Facts, seed: u64) -> Option<Mission> {
        let origin = |r: &str| Origin::Random {
            routine: r.to_string(),
        };
        match routine {
            "read-a-station" => {
                let mut h = DefaultHasher::new();
                (seed, "station").hash(&mut h);
                let duty = facts
                    .duties
                    .get((h.finish() % facts.duties.len().max(1) as u64) as usize)?;
                let Duty { room, device } = duty;
                Some(Mission::new(
                    format!(
                        "Go to {room} and read {device}: scan it, and find out what state it \
                         is in. Then come back and report exactly what you read."
                    ),
                    vec![
                        Todo::new(format!("go to {room}")),
                        Todo::new(format!("scan {device} and read what state it is in")),
                    ],
                    origin("read-a-station"),
                ))
            }
            "visit-and-review" => {
                let who = pick(facts.makers, seed, "visit")?;
                Some(Mission::new(
                    format!(
                        "Go and find {who}, wherever they are working, and review what they \
                         are doing. Watch a while, understand it, and tell them plainly what \
                         you think — what is good in it and what you would do differently."
                    ),
                    vec![
                        Todo::new(format!("find where {who} is")),
                        Todo::new(format!("travel to {who}")),
                        Todo::new(format!("ask {who} what they are working on")),
                        Todo::new("form your own view of it"),
                        Todo::new(format!("give {who} your feedback, to their face")),
                    ],
                    origin("visit-and-review"),
                ))
            }
            "hear-them-out" => {
                let who = pick(facts.makers, seed, "hear")?;
                Some(Mission::new(
                    format!(
                        "Seek out {who} and hear them out — properly, for a good while, about \
                         what they have been working on and what they have concluded from it. \
                         Do not settle for a line and away."
                    ),
                    vec![
                        Todo::new(format!("find {who}")),
                        Todo::new(format!("draw {who} into talking about their work")),
                        Todo::new("stay with it, and follow what they mean"),
                    ],
                    origin("hear-them-out"),
                ))
            }
            "check-a-record" => {
                let rec = pick(facts.records, seed, "check")?;
                Some(Mission::new(
                    format!(
                        "Go to the archives and read the record '{rec}'. Check it against the \
                         main storyline as you understand it, and submit your thoughts on \
                         where it holds and where it does not."
                    ),
                    vec![
                        Todo::new("go to the archives"),
                        Todo::new(format!("read the record '{rec}'")),
                        Todo::new("check it against the main storyline"),
                        Todo::new("write up your thoughts and submit them"),
                    ],
                    origin("check-a-record"),
                ))
            }
            "read-the-canon" => {
                let rec = pick(facts.records, seed, "canon")?;
                Some(Mission::new(
                    format!(
                        "Read '{rec}' in the archives — not to correct it, only to know it. \
                         Come away able to say what it is about and why it matters here."
                    ),
                    vec![
                        Todo::new("go to the archives"),
                        Todo::new(format!("read '{rec}' closely")),
                        Todo::new("settle what it is about and why it matters"),
                    ],
                    origin("read-the-canon"),
                ))
            }
            "compare-notes" => {
                let a = pick(facts.makers, seed, "cmp-a")?;
                let b = pick(facts.makers, seed.wrapping_add(1), "cmp-b")?;
                if a == b {
                    return None;
                }
                Some(Mission::new(
                    format!(
                        "Two of the makers here, {a} and {b}, are each working on something. \
                         Visit both, learn what each is doing, and come back able to say where \
                         their work meets and where it pulls apart."
                    ),
                    vec![
                        Todo::new(format!("visit {a} and learn their work")),
                        Todo::new(format!("visit {b} and learn their work")),
                        Todo::new("settle where the two meet and where they differ"),
                    ],
                    origin("compare-notes"),
                ))
            }
            "trace-a-thread" => {
                let who = pick(facts.makers, seed, "thread")?;
                Some(Mission::new(
                    format!(
                        "Something {who} said, or made, has a thread you want to follow. Go to \
                         them, ask about it, and follow it wherever it leads — another person, \
                         a record, a room. Come back knowing where it goes."
                    ),
                    vec![
                        Todo::new(format!("go to {who} and ask about it")),
                        Todo::new("follow the thread it opens"),
                        Todo::new("come back knowing where it leads"),
                    ],
                    origin("trace-a-thread"),
                ))
            }
            "sit-with-a-story" => {
                let rec = pick(facts.records, seed, "story")?;
                Some(Mission::new(
                    format!(
                        "Sit with the story '{rec}'. Read it, then find someone and tell them \
                         what you made of it — and hear what they made of it too."
                    ),
                    vec![
                        Todo::new("go to the archives"),
                        Todo::new(format!("read '{rec}'")),
                        Todo::new("find someone and talk it over with them"),
                    ],
                    origin("sit-with-a-story"),
                ))
            }
            "look-in-on-someone" => {
                let who = pick(facts.makers, seed, "lookin")?;
                Some(Mission::new(
                    format!(
                        "Look in on {who}. Not for anything in particular — see how they are, \
                         what they are about, whether they need a hand or an ear. Stay a while."
                    ),
                    vec![
                        Todo::new(format!("find {who}")),
                        Todo::new(format!("look in on {who} and stay a while")),
                    ],
                    origin("look-in-on-someone"),
                ))
            }
            "take-stock" => Some(Mission::new(
                "Take stock of what you have been working on yourself. Walk somewhere quiet, \
                 think it through, and be ready to tell the next person you meet what you have \
                 concluded — properly, not a line.",
                vec![
                    Todo::new("find somewhere to think"),
                    Todo::new("settle what you have concluded"),
                    Todo::new("tell the next person you meet, properly"),
                ],
                origin("take-stock"),
            )),
            "walk-the-halls" => Some(walk_the_halls(facts)),
            _ => None,
        }
    }

    /// The routine that always fits — no other maker and no record required.
    fn walk_the_halls(_facts: &Facts) -> Mission {
        Mission::new(
            "Walk the halls and see who is about and what is going on. When you come across \
             someone, stop and talk to them about what they are working on. Move on after a \
             while and find the next.",
            vec![
                Todo::new("set off and walk the halls"),
                Todo::new("stop and talk to the first person you find"),
                Todo::new("move on and find the next"),
            ],
            Origin::Random {
                routine: "walk-the-halls".to_string(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::bank::{random, Duty, Facts, ROUTINES};
    use super::{Mission, Origin, Outcome, StepOutcome, Todo, Work};

    fn makers() -> Vec<String> {
        vec!["Wren".to_string(), "Pax".to_string(), "Soren".to_string()]
    }
    fn records() -> Vec<String> {
        vec!["the-charge".to_string(), "the-awakening".to_string()]
    }

    fn a_mission() -> Mission {
        Mission::new(
            "Do the thing.",
            vec![Todo::new("step one"), Todo::new("step two")],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        )
    }

    fn read_the_valve() -> Mission {
        Mission::new(
            "Go to the plant room and read the coolant valve.",
            vec![
                Todo::new("go to the plant room"),
                Todo::new("scan the coolant valve and read what state it is in"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        )
    }

    /// **Arriving signs the journey off; arriving somewhere else does not.** The
    /// room is compared by name, with or without its article.
    #[test]
    fn arriving_in_the_named_room_signs_off_the_journey_to_it() {
        let mut m = read_the_valve();
        assert!(!m.arrived_in("the command room", "the command level"));
        assert!(!m.todo[0].done);
        assert!(m.arrived_in("The plant room", "the command level"));
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Achieved));
        assert!(
            !m.arrived_in("the plant room", "the command level"),
            "already signed off"
        );
        assert_eq!(
            m.next_step().map(|t| t.text.as_str()),
            Some("scan the coolant valve and read what state it is in")
        );
    }

    fn write_a_year() -> Mission {
        Mission::new(
            "Write the year the sky went out.",
            vec![
                Todo::new("go to band one on the casting level"),
                Todo::new("read layers/eras/the-awakening.md"),
                Todo::new("read layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"),
                Todo::new("write layers/life/keeper/2488 The Year After.md and commit it"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Generated {
                generator: "life-event".into(),
                target: "life:keeper".into(),
                operation: 1,
                step: "write".into(),
            },
        )
        .with_work(Work {
            writes: "layers/life/keeper/2488 The Year After.md".into(),
            reads: vec!["layers/eras/the-awakening.md".into()],
            min_words: 250,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        })
    }

    /// **A room of the same name on another level is not the one asked for**;
    /// the level decides.
    #[test]
    fn a_journey_that_names_its_level_is_signed_off_only_there() {
        let mut m = write_a_year();
        assert!(!m.arrived_in("band one", "the chronicle"));
        assert!(!m.todo[0].done);
        assert!(m.arrived_in("band one", "the casting level"));
        assert!(m.todo[0].done);
    }

    /// **Reading a named document and committing the written one sign their
    /// steps off**, whatever case or slashes the path came back in.
    #[test]
    fn reading_and_committing_the_named_documents_sign_their_steps_off() {
        let mut m = write_a_year();
        assert!(!m.read_doc("layers/eras/the-salvation.md"));
        assert!(m.read_doc("Layers/eras/the-awakening.md"));
        assert!(m.todo[1].done);
        assert!(m.read_doc("/layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"));
        assert!(!m.committed(&["layers/life/keeper/2489 Another.md".to_string()]));
        assert!(!m.todo[3].done);
        assert!(m.committed(&[
            "layers/eras/the-awakening.md".to_string(),
            "layers/life/keeper/2488 The Year After.md".to_string()
        ]));
        assert!(m.todo[3].done);
        assert!(m.written_up());
        assert!(m.arrived_in("band one", "the casting level"));
        assert!(m.next_step().unwrap().reports);
    }

    /// **Written up means committed while carried** — a write step struck on
    /// somebody's word is not the document on the record.
    #[test]
    fn only_a_commit_writes_a_mission_up() {
        let mut m = write_a_year();
        assert!(!m.written_up());
        m.check_off(
            "write layers/life/keeper/2488 The Year After.md and commit it",
            StepOutcome::Thwarted,
        );
        assert!(!m.written_up(), "thwarted is not written");
        assert!(read_the_valve().written_up(), "nothing to write");
    }

    /// **Only the engine signs off what the engine sees** — the journey, the
    /// reads and the write — and anything else may be signed off on the
    /// character's word.
    #[test]
    fn the_steps_the_engine_sees_are_named() {
        for step in &write_a_year().todo[..4] {
            assert!(super::engine_sees(&step.text), "{}", step.text);
        }
        assert!(super::engine_sees(
            "scan the coolant valve and read what state it is in"
        ));
        assert!(!super::engine_sees("form your own view of it"));
    }

    /// **The compass for the year says the act at the machine, or the way to it.**
    #[test]
    fn the_compass_for_the_year_says_the_act_or_the_way() {
        use super::time_compass;
        assert_eq!(
            time_compass(2837, true, None),
            "Your mission, next: set your time to 2837. A time machine is here: `time_travel` \
             naming the year 2837."
        );
        assert_eq!(
            time_compass(
                2837,
                false,
                Some("the first time room, on the time level: `move_to` the lift, `lift_use` naming the time level, then `move_to` the first time room")
            ),
            "Your mission, next: set your time to 2837. The time machines are in the first time \
             room, on the time level: `move_to` the lift, `lift_use` naming the time level, then \
             `move_to` the first time room; there, `time_travel` naming the year 2837."
        );
        assert!(time_compass(2837, false, None).contains("`lift_use` naming the time level"));
    }

    /// **At a desk, the compass says the exact act; away from one, the way.**
    #[test]
    fn the_compass_for_a_document_says_the_act_that_does_it() {
        use super::{doc_compass, DeskVerbs, DocStep};
        let desk = DeskVerbs {
            read: Some("http://local/desk/1/file_read".into()),
            compose: Some("http://local/desk/1/compose".into()),
            edit: None,
            commit: Some("http://local/desk/1/bench_commit".into()),
        };
        assert_eq!(
            doc_compass(&DocStep::Read("layers/eras/x.md".into()), &desk, None, None),
            "Your mission, next: read layers/eras/x.md. `invoke` http://local/desk/1/file_read \
             with `path` \"layers/eras/x.md\" — it comes back a page at a time."
        );
        let written = doc_compass(
            &DocStep::Write("layers/stories/y.md".into()),
            &desk,
            None,
            Some(537),
        );
        assert_eq!(
            written,
            "Your mission, next: commit layers/stories/y.md. It is written — 537 words in your \
             working copy. Read it back with `invoke` http://local/desk/1/file_read with `path` \
             \"layers/stories/y.md\"; if it stands, `invoke` http://local/desk/1/bench_commit with \
             `why`, one line saying what it is. Put one passage right first if one is wrong, but \
             do not write it again."
        );
        let write = doc_compass(
            &DocStep::Write("layers/stories/y.md".into()),
            &desk,
            None,
            None,
        );
        assert!(
            write.contains(
                "Sit down and write it whole: `invoke` http://local/desk/1/compose — you write it \
                 through"
            ) && write.contains("Then `invoke` http://local/desk/1/bench_commit with `why`"),
            "{write}"
        );
        assert_eq!(
            doc_compass(
                &DocStep::Read("layers/eras/x.md".into()),
                &DeskVerbs::default(),
                Some("band one, on the story level: `move_to` the lift"),
                None
            ),
            "Your mission, next: read layers/eras/x.md. Documents are read at a desk: the nearest \
             is in band one, on the story level: `move_to` the lift."
        );
        // A terminal that reads but does not compose is not where a piece is
        // written: the way goes to a writing desk.
        let reads_only = DeskVerbs {
            read: Some("http://local/chronicle/t/file_read".into()),
            commit: Some("http://local/chronicle/t/bench_commit".into()),
            ..DeskVerbs::default()
        };
        assert_eq!(
            doc_compass(
                &DocStep::Write("layers/stories/y.md".into()),
                &reads_only,
                Some("band one, on the story level: `move_to` the lift"),
                None
            ),
            "Your mission, next: write layers/stories/y.md. A piece is written at a writing \
             desk: the nearest is in band one, on the story level: `move_to` the lift."
        );
    }

    /// The paths a step names, by the step's own verb.
    #[test]
    fn a_step_names_the_document_it_reads_or_writes() {
        assert_eq!(
            super::reading_doc("read layers/eras/the-awakening.md closely"),
            Some("layers/eras/the-awakening.md".into())
        );
        assert_eq!(
            super::writing_doc("write layers/life/keeper/2488 The Year After.md and commit it"),
            Some("layers/life/keeper/2488 the year after.md".into())
        );
        assert_eq!(
            super::writing_doc("change worlds/battle-cities.yaml so both agree"),
            Some("worlds/battle-cities.yaml".into())
        );
        assert_eq!(super::reading_doc("read the record 'the-charge'"), None);
        assert_eq!(super::writing_doc("go to band one"), None);
        // Told to the character as the step spells it.
        assert_eq!(
            super::DocStep::of("write layers/life/keeper/2488 The Year After.md and commit it"),
            Some(super::DocStep::Write(
                "layers/life/keeper/2488 The Year After.md".into()
            ))
        );
        assert_eq!(
            super::DocStep::of("read layers/eras/the-awakening.md"),
            Some(super::DocStep::Read("layers/eras/the-awakening.md".into()))
        );
        assert_eq!(super::DocStep::of("go to band one"), None);
        // A title whose lowercase is longer than itself does not split a
        // character.
        assert_eq!(
            super::DocStep::of("write layers/stories/İstanbul Gate.MD and commit it"),
            Some(super::DocStep::Write(
                "layers/stories/İstanbul Gate.MD".into()
            ))
        );
    }

    /// **A scan that shows the machine signs off the reading of it**, and a
    /// mission with a reading still to make holds `report_done` up — until the
    /// reading is made, when only the report is left.
    #[test]
    fn a_scan_that_shows_the_machine_signs_off_the_reading() {
        let mut m = read_the_valve();
        let machines = ["the coolant valve".to_string()];
        assert_eq!(
            m.unread_step(&machines),
            Some("scan the coolant valve and read what state it is in")
        );
        assert_eq!(
            m.unread_step(&["the breaker panel".to_string()]),
            None,
            "a reading of nothing the world holds is not the engine's to hold up"
        );
        assert!(!m.read_off(&["the plant panel".to_string()]));
        assert!(m.read_off(&[
            "the plant panel".to_string(),
            "the coolant valve".to_string()
        ]));
        assert_eq!(m.unread_step(&machines), None);
        assert_eq!(
            m.next_step().map(|t| t.text.as_str()),
            Some("go to the plant room"),
            "the journey was never signed off, and reading does not sign it"
        );
        assert!(m.arrived_in("the plant room", "the command level"));
        assert!(m.next_step().unwrap().reports);
        assert!(!m.todo[2].done, "nothing but the report closes the report");
    }

    /// **Being with somebody signs off finding them; speaking to them signs off
    /// asking them too.** By whole name or first name; nobody else's name does.
    #[test]
    fn meeting_and_speaking_to_somebody_sign_off_the_steps_about_them() {
        let mut m = Mission::new(
            "Find out what Paxon Vael is working on.",
            vec![
                Todo::new("find Paxon Vael"),
                Todo::new("ask Paxon what he is working on"),
                Todo::new("tell Ione Valtiere what you learnt"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        );
        assert!(!m.met("Vespera Kaine"));
        assert!(m.met("Paxon Vael"));
        assert!(m.todo[0].done);
        assert!(!m.todo[1].done, "being in the room is not asking");
        assert!(m.spoke_with("Paxon Vael"));
        assert!(m.todo[1].done, "asked by first name");
        assert!(!m.todo[2].done);
        assert!(!m.spoke_with("Pax"), "a fragment of a name is nobody");
        assert!(
            !m.spoke_with("the channel"),
            "an article is not a first name"
        );
        assert!(m.spoke_with("Ione Valtiere"));
        assert!(m.next_step().unwrap().reports);
    }

    /// **The compass names the next step and where the person it names is.**
    #[test]
    fn the_compass_says_where_the_person_the_step_names_is() {
        let ask = Todo::new("ask Paxon Vael what he has been working on");
        let elsewhere = [
            (
                "Paxon Vael".to_string(),
                Some("the plant room: `move_to` it".to_string()),
            ),
            ("Vespera Kaine".to_string(), None),
        ];
        assert_eq!(
            super::compass_line(&ask, &elsewhere, None),
            "Your mission, next: ask Paxon Vael what he has been working on. Paxon Vael is in \
             the plant room: `move_to` it — or `message` them, which reaches them wherever they \
             are."
        );
        let here = [("Paxon Vael".to_string(), None)];
        assert_eq!(
            super::compass_line(&ask, &here, None),
            "Your mission, next: ask Paxon Vael what he has been working on. Paxon Vael is here \
             with you: say it to them now — `ask` or `tell` them."
        );
        assert_eq!(
            super::compass_line(
                &Todo::new("go to band three"),
                &here,
                Some("Band three is on the record level: go to the lift.")
            ),
            "Your mission, next: go to band three. Band three is on the record level: go to the \
             lift."
        );
        assert!(super::compass_report_line(
            Some("the command room: `move_to` it"),
            false,
            &[],
            false
        )
        .contains("Go back to the table, in the command room: `move_to` it, and report"));
        let review = super::compass_report_line(None, true, &[], true);
        assert!(
            review.contains("That verdict is the whole report."),
            "{review}"
        );
        assert!(review.contains("`report_rejected`"), "{review}");
        let seen = ["in the plant room, the coolant valve: tight".to_string()];
        let now = super::compass_report_line(None, true, &seen, false);
        assert!(now.contains("report it now"), "{now}");
        assert!(
            now.ends_with(" What you saw on it: in the plant room, the coolant valve: tight."),
            "{now}"
        );
    }

    /// **A document that does not stand is to be written again**: a done
    /// writing step is reopened, and a mission with none gains one before its
    /// report.
    #[test]
    fn a_refused_document_puts_its_writing_back_in_front() {
        use super::{Mission, Origin, Todo};
        let doc = "layers/life/keeper/2950 The Archive.md";
        let mut drafted = Mission::new(
            "write it",
            vec![
                Todo::new(format!("write {doc} and commit it")),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        );
        drafted.committed(&[doc.to_string()]);
        assert!(drafted.next_step().is_some_and(|t| t.reports));
        drafted.reopen_write(doc);
        assert_eq!(
            drafted.next_step().map(|t| t.text.as_str()),
            Some(format!("write {doc} and commit it").as_str())
        );

        let mut checked = Mission::new(
            "check it",
            vec![
                Todo::new(format!("read {doc}")),
                Todo::report("go back to the table and report your verdict"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        );
        checked.read_doc(doc);
        checked.reopen_write(doc);
        let steps: Vec<&str> = checked.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                format!("read {doc}").as_str(),
                format!("change {doc} and commit it").as_str(),
                "go back to the table and report your verdict"
            ]
        );
    }

    /// **A read of a document the record no longer holds is struck**, step and
    /// listed read both; one already done, and every other step, stays.
    #[test]
    fn a_read_of_a_document_gone_from_the_record_is_struck() {
        use super::{Mission, Origin, Todo, Work};
        let gone = "layers/stories/the-anchor-s-last-breath.md";
        let kept = "layers/eras/the-portal-retreat.md";
        let writes = "layers/stories/the-first-breach.md";
        let mut m = Mission::new(
            "write it",
            vec![
                Todo::new("go to the first writing room on the story level"),
                Todo::new(format!("read {kept}")),
                Todo::new(format!("read {gone}")),
                Todo::new(format!("write {writes} and commit it")),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        )
        .with_work(Work {
            writes: writes.into(),
            reads: vec![
                kept.into(),
                "Layers/Stories/The-Anchor-S-Last-Breath.md".into(),
            ],
            min_words: 0,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        });
        let holds = |p: &str| p != gone;
        assert_eq!(m.strike_gone_reads(&holds), [gone]);
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the first writing room on the story level",
                "read layers/eras/the-portal-retreat.md",
                "write layers/stories/the-first-breach.md and commit it",
                "go back to the table and report it"
            ]
        );
        assert_eq!(m.work.as_ref().unwrap().reads, [kept]);
        assert!(
            m.strike_gone_reads(&holds).is_empty(),
            "nothing left to strike"
        );

        // A read already made stays on the list as done: it happened.
        let mut read = Mission::new(
            "check it",
            vec![Todo::new(format!("read {gone}")), Todo::report("report")],
            Origin::Random {
                routine: "r".into(),
            },
        );
        read.read_doc(gone);
        assert!(read.strike_gone_reads(&holds).is_empty());
        assert_eq!(read.todo.len(), 2);
    }

    #[test]
    fn a_step_names_its_aim() {
        use super::Aim;
        assert_eq!(
            Aim::of("go to the plant room"),
            Some(Aim::Room("plant room".into()))
        );
        let read = Aim::of("scan the coolant valve and read what state it is in").unwrap();
        assert!(read.is_machine("the coolant valve"));
        assert!(!read.is_machine("the breaker panel"));
        let ask = Aim::of("ask Paxon Vael what he has been working on").unwrap();
        assert!(ask.is_person("Paxon Vael"));
        assert!(!ask.is_room("the plant room", "the command level"));
        // A destination that names its level is that room on that level only.
        let band = Aim::of("go to band one on the casting level").unwrap();
        assert!(band.is_room("band one", "the casting level"));
        assert!(!band.is_room("band one", "the chronicle"));
        assert!(Aim::of("go to the plant room")
            .unwrap()
            .is_room("the plant room", "anywhere"));
        let go_to_him = Aim::of("go to Paxon Vael").unwrap();
        assert!(go_to_him.is_person("Paxon Vael"));
        assert_eq!(Aim::of("form your own view of it"), None);
    }

    /// A step the engine cannot observe never holds a report up.
    #[test]
    fn a_step_the_engine_cannot_see_does_not_hold_the_report_up() {
        let m = Mission::new(
            "Hear Pax out.",
            vec![Todo::new("find Pax"), Todo::new("form your own view of it")],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        );
        assert_eq!(m.unread_step(&["Pax".to_string()]), None);
    }

    #[test]
    fn the_progress_line_names_the_next_step_or_the_report() {
        let m = read_the_valve();
        assert_eq!(
            super::progress_line("You are in the plant room", m.todo.get(1), &[]),
            "You are in the plant room: that step of your mission is done. Next: scan the \
             coolant valve and read what state it is in."
        );
        assert_eq!(
            super::progress_line(
                "You are in the plant room",
                m.todo.get(1),
                &[
                    "the breaker panel".to_string(),
                    "the coolant valve".to_string()
                ]
            ),
            "You are in the plant room: that step of your mission is done. Next: scan the \
             coolant valve and read what state it is in. It is here, in this room: `scan` \
             naming no place, and it lists it with the state it is in."
        );
        let last = super::progress_line("You have read it", m.todo.get(2), &[]);
        assert!(
            last.starts_with("You have read it: that was the last step of your mission."),
            "{last}"
        );
        assert!(last.contains("`report_done`"), "{last}");
        assert_eq!(super::progress_line("You have read it", None, &[]), last);
    }

    #[test]
    fn a_new_mission_is_open_with_no_report_and_no_answer() {
        let m = a_mission();
        assert!(m.is_open());
        assert!(m.report.is_none());
        assert!(m.answer.is_none());
        assert!(!m.all_todos_done());
    }

    #[test]
    fn check_off_ticks_the_named_step_case_insensitively_and_reports_the_hit() {
        let mut m = a_mission();
        assert!(
            m.check_off("  STEP one ", StepOutcome::Achieved),
            "trim + case-insensitive match"
        );
        assert!(m.todo[0].done);
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Achieved));
        assert!(!m.todo[1].done);
        assert_eq!(m.todo[1].outcome, None);
        // A second tick of the same step finds nothing left to tick.
        assert!(!m.check_off("step one", StepOutcome::Thwarted));
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Achieved));
        // A step that is not on the list ticks nothing.
        assert!(!m.check_off("step three", StepOutcome::Achieved));
    }

    #[test]
    fn a_thwarted_step_is_done_and_reads_to_the_character_as_one_it_could_not_do() {
        let mut m = a_mission();
        assert!(m.check_off("step one", StepOutcome::Thwarted));
        assert!(m.todo[0].done);
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Thwarted));
        assert_eq!(StepOutcome::Achieved.as_str(), "achieved");
        assert_eq!(StepOutcome::Thwarted.as_str(), "thwarted");
        assert_eq!(
            m.task_text().as_deref(),
            Some("[could not be done] step one\n[ ] step two")
        );
        let mut achieved = a_mission();
        achieved.check_off("step one", StepOutcome::Achieved);
        assert_ne!(m.content_fingerprint(), achieved.content_fingerprint());
    }

    #[test]
    fn a_step_signed_off_before_outcomes_were_recorded_loads_with_none() {
        let t: Todo = serde_json::from_str(r#"{"text":"old","done":true}"#).unwrap();
        assert!(t.done);
        assert_eq!(t.outcome, None);
        assert_eq!(
            serde_json::to_string(&Todo::new("x")).unwrap(),
            r#"{"text":"x","done":false,"outcome":null,"reports":false}"#
        );
    }

    #[test]
    fn all_todos_done_only_once_every_step_is_ticked() {
        let mut m = a_mission();
        assert!(!m.all_todos_done());
        assert!(m.check_off("step one", StepOutcome::Achieved));
        assert!(!m.all_todos_done());
        assert!(m.check_off("step two", StepOutcome::Thwarted));
        assert!(m.all_todos_done());
    }

    #[test]
    fn add_todo_appends_but_refuses_blank_and_open_duplicates() {
        let mut m = a_mission();
        assert!(m.add_todo("step three"));
        assert_eq!(m.todo.len(), 3);
        assert!(!m.add_todo("   "), "blank refused");
        assert!(!m.add_todo(" STEP three "), "open duplicate refused");
        assert_eq!(m.todo.len(), 3);
        // Once a step is done, the same text may be added again as new work.
        assert!(m.check_off("step three", StepOutcome::Achieved));
        assert!(m.add_todo("step three"));
        assert_eq!(m.todo.len(), 4);
    }

    #[test]
    fn complete_files_the_report_and_closes_the_mission() {
        let mut m = a_mission();
        m.complete(Outcome::Pass, "went fine", Some("the answer".to_string()));
        assert!(!m.is_open());
        assert_eq!(m.report.as_ref().unwrap().outcome, Outcome::Pass);
        assert_eq!(m.report.as_ref().unwrap().notes, "went fine");
        assert_eq!(m.answer.as_deref(), Some("the answer"));
        assert_eq!(Outcome::Pass.as_str(), "pass");
        assert_eq!(Outcome::Fail.as_str(), "fail");
    }

    #[test]
    fn mission_and_task_text_render_the_ask_and_the_ticked_steps() {
        let mut m = a_mission();
        m.check_off("step one", StepOutcome::Achieved);
        assert_eq!(m.mission_text(), "Do the thing.");
        assert_eq!(
            m.task_text().as_deref(),
            Some("[done] step one\n[ ] step two")
        );
        // A mission with no steps renders no task section (task_intro stays hidden).
        let bare = Mission::new(
            "Just be.",
            vec![],
            Origin::Random {
                routine: "x".into(),
            },
        );
        assert!(bare.task_text().is_none());
    }

    #[test]
    fn standing_text_names_the_ask_the_steps_and_where_it_ends() {
        let m = a_mission();
        let text = m.standing_text();
        assert!(text.starts_with("What has been asked of you: Do the thing."));
        assert!(text.contains("The steps that see it through:"));
        assert!(text.contains("step one") && text.contains("step two"));
        // It ends by pointing back to the table to report and take the next — the
        // loop closes there, not out in the world — by scanning it and invoking
        // its report verbs.
        assert!(
            text.contains("go back to the table where work is handed out, `scan` it")
                && text.contains("`report_done`")
                && text.contains("`report_stuck`")
        );
        // A mission carried on its ask alone still ends at the table.
        let bare = Mission::new(
            "Just be.",
            vec![],
            Origin::Random {
                routine: "x".into(),
            },
        );
        assert!(bare
            .standing_text()
            .contains("go back to the table where work is handed out"));
    }

    #[test]
    fn the_prompt_is_the_standing_text_under_its_own_heading_and_carries_the_fingerprint() {
        let m = a_mission();
        let prompt = m.prompt();
        assert!(!prompt.text.starts_with("What has been asked of you"));
        assert_eq!(
            m.standing_text(),
            format!("What has been asked of you: {}", prompt.text)
        );
        assert_eq!(prompt.fingerprint, m.content_fingerprint());
    }

    #[test]
    fn content_fingerprint_tracks_the_rendered_prompt_only() {
        let m1 = a_mission();
        let mut m2 = a_mission();
        // Filing a report or an answer does not change what the character reads,
        // so the fingerprint is unchanged and no re-seal is forced.
        m2.answer = Some("built up".to_string());
        m2.report = Some(super::Report {
            outcome: Outcome::Pass,
            notes: "done".to_string(),
        });
        assert_eq!(m1.content_fingerprint(), m2.content_fingerprint());
        // Ticking a step does change the rendered task text, so it re-fingerprints.
        let mut m3 = a_mission();
        m3.check_off("step one", StepOutcome::Achieved);
        assert_ne!(m1.content_fingerprint(), m3.content_fingerprint());
    }

    #[test]
    fn a_random_mission_is_open_non_empty_and_tagged_with_a_known_routine() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
            ..Facts::default()
        };
        for seed in 0..50u64 {
            let m = random(&facts, seed);
            assert!(m.is_open());
            assert!(!m.prompt.trim().is_empty());
            assert!(!m.todo.is_empty(), "a routine gives steps to act on");
            match &m.origin {
                Origin::Random { routine } => {
                    assert!(
                        ROUTINES.contains(&routine.as_str()),
                        "known routine: {routine}"
                    );
                }
                Origin::Lodged { .. } | Origin::Generated { .. } => {
                    panic!("bank produces Random origins")
                }
            }
        }
    }

    #[test]
    fn random_missions_vary_with_the_seed() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
            ..Facts::default()
        };
        let prompts: std::collections::HashSet<String> =
            (0..30u64).map(|s| random(&facts, s).prompt).collect();
        assert!(
            prompts.len() > 3,
            "the bank is diverse, not one repeated mission"
        );
    }

    #[test]
    fn a_lone_character_with_no_records_still_gets_an_actionable_mission() {
        // No other makers, no archive: routines needing them are skipped and the
        // always-available "walk the halls" fallback fires.
        let facts = Facts::default();
        // The only routines needing no other maker and no record.
        let no_material = ["take-stock", "walk-the-halls"];
        for seed in 0..20u64 {
            let m = random(&facts, seed);
            assert!(!m.todo.is_empty());
            match &m.origin {
                Origin::Random { routine } => assert!(
                    no_material.contains(&routine.as_str()),
                    "a lone character gets a routine needing no material, got {routine}"
                ),
                Origin::Lodged { .. } | Origin::Generated { .. } => {
                    panic!("bank produces Random origins")
                }
            }
        }
    }

    #[test]
    fn a_world_with_machines_gets_a_mission_to_go_and_read_one() {
        let mk = makers();
        let duties = vec![
            Duty {
                room: "the foundry".to_string(),
                device: "the coolant valve".to_string(),
            },
            Duty {
                room: "the armoury".to_string(),
                device: "the breaker panel".to_string(),
            },
        ];
        let facts = Facts {
            makers: &mk,
            duties: &duties,
            table: Some("the muster hall"),
            ..Facts::default()
        };
        for seed in 0..20u64 {
            let m = random(&facts, seed);
            assert_eq!(
                m.origin,
                Origin::Random {
                    routine: "read-a-station".to_string()
                }
            );
            let duty = duties
                .iter()
                .find(|d| m.prompt.contains(&d.device))
                .expect("the mission names one of the machines");
            assert!(m.prompt.contains(&duty.room), "{}", m.prompt);
            assert_eq!(m.todo[0].text, format!("go to {}", duty.room));
            assert_eq!(
                m.todo.last().unwrap().text,
                "go back to the table in the muster hall and report it",
                "it ends at the table, by the room's name"
            );
        }
    }

    #[test]
    fn every_routine_ends_with_a_step_that_reports_at_the_table() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
            ..Facts::default()
        };
        for seed in 0..30u64 {
            let m = random(&facts, seed);
            assert_eq!(
                m.todo.last().unwrap().text,
                "go back to the table and report it"
            );
            assert!(m.todo.last().unwrap().reports);
            assert_eq!(m.todo.iter().filter(|t| t.reports).count(), 1);
        }
    }
}
