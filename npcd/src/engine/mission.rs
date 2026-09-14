//! A mission: what has been asked of a character, the steps it works through,
//! and the report it files when it is done.
//!
//! `docs/npcd_worlds_and_layers.md` describes the projection a character reads;
//! the schema (`D:/prog/mind/projection.yaml`) already declares empty `mission`
//! and `task` collections with the gating that shows a mission when one exists
//! and the "nothing has been asked of you" standing instruction when none does
//! (see `engine::identity`'s module doc). This module is the *data* those
//! collections carry once a mission exists — deliberately pure, so the shape of
//! a mission, the steps under it, and the text it renders into the prompt can be
//! tested without a running engine. Storage (conversation metadata) and the
//! sealing of the rendered text into KV live with the code that owns the
//! substrate; this file owns only the mission itself.
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
    /// Whether it has been ticked off.
    pub done: bool,
}

impl Todo {
    /// A fresh, un-ticked step.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            done: false,
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

    /// Tick off the first un-ticked step whose text matches `which` (trimmed,
    /// case-insensitive). Returns whether a step was ticked — `false` when
    /// nothing matched or every match was already done, so the caller can tell
    /// the character its instruction landed on nothing.
    pub fn check_off(&mut self, which: &str) -> bool {
        let want = which.trim().to_lowercase();
        for step in &mut self.todo {
            if !step.done && step.text.trim().to_lowercase() == want {
                step.done = true;
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

    /// The text the `mission` collection member carries — the ask itself, read
    /// under the schema's `mission_intro` heading ("What has been asked of
    /// you:"). Empty missions never reach here; the schema shows `mission_none`
    /// instead.
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
            let mark = if step.done { "[done]" } else { "[ ]" };
            lines.push(format!("{mark} {}", step.text.trim()));
        }
        Some(lines.join("\n"))
    }

    /// The first step not yet ticked off — the one thing to do next.
    pub fn next_step(&self) -> Option<&str> {
        self.todo
            .iter()
            .find(|step| !step.done)
            .map(|step| step.text.trim())
    }

    /// The mission as a standing instruction — what a character reads each quiet
    /// turn while it carries one.
    ///
    /// The ask, the steps with their progress, and a pointer at the **next**
    /// unfinished step. A standing task restated every quiet turn must name the
    /// next thing once rather than describe the whole plan — the lesson recorded
    /// on [`crate::engine::runtime::NO_MISSION`], where a plan in the most-recent
    /// window position turned every turn into motion. When every step is done it
    /// points home to the command desk, so the loop closes on a report rather
    /// than trailing off.
    pub fn standing_text(&self) -> String {
        let mut out = format!("What has been asked of you: {}", self.mission_text());
        if let Some(tasks) = self.task_text() {
            out.push_str("\nYou are working through:\n");
            out.push_str(&tasks);
        }
        match self.next_step() {
            Some(step) => {
                out.push_str("\nThe next thing to do is: ");
                out.push_str(step);
                out.push('.');
            }
            // Steps existed and are all done: the work is finished, so the one
            // thing left is to say so — with `report_done` (or `report_stuck`).
            None if !self.todo.is_empty() => out.push_str(
                "\nEvery step is done. Report how it went now with `report_done` \
                 (or `report_stuck` if it could not be finished).",
            ),
            // A mission with no steps at all is carried on its ask alone.
            None => {}
        }
        out
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
        }
        h.finish()
    }
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
    #[derive(Debug, Clone)]
    pub struct Facts<'a> {
        /// Other characters' names, to contact or ask about.
        pub makers: &'a [String],
        /// Archived record ids, to read and check against the storyline.
        pub records: &'a [String],
    }

    /// Every routine's name, in a fixed order. `random` picks one of these by
    /// seed; the name becomes the mission's [`Origin::Random`] routine tag.
    pub const ROUTINES: &[&str] = &[
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
        // The first routine whose material is present, in shuffled order. The
        // rotation carries routines that need nobody and nothing (`take-stock`,
        // `walk-the-halls`), so a world with no other makers and no archive
        // still yields one; `walk_the_halls` is the default should every arm one
        // day become conditional and a barren world leave the loop empty-handed.
        order
            .iter()
            .find_map(|&i| build(ROUTINES[i], facts, seed))
            .unwrap_or_else(|| walk_the_halls(facts))
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
    use super::bank::{random, Facts, ROUTINES};
    use super::{Mission, Origin, Outcome, Todo};

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
        assert!(m.check_off("  STEP one "), "trim + case-insensitive match");
        assert!(m.todo[0].done);
        assert!(!m.todo[1].done);
        // A second tick of the same step finds nothing left to tick.
        assert!(!m.check_off("step one"));
        // A step that is not on the list ticks nothing.
        assert!(!m.check_off("step three"));
    }

    #[test]
    fn all_todos_done_only_once_every_step_is_ticked() {
        let mut m = a_mission();
        assert!(!m.all_todos_done());
        assert!(m.check_off("step one"));
        assert!(!m.all_todos_done());
        assert!(m.check_off("step two"));
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
        assert!(m.check_off("step three"));
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
        m.check_off("step one");
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
    fn standing_text_names_the_ask_the_steps_and_the_next_thing() {
        let mut m = a_mission();
        assert_eq!(m.next_step(), Some("step one"));
        assert_eq!(
            m.standing_text(),
            "What has been asked of you: Do the thing.\n\
             You are working through:\n\
             [ ] step one\n\
             [ ] step two\n\
             The next thing to do is: step one."
        );
        // Once a step is ticked, the pointer moves to the next open one.
        m.check_off("step one");
        assert_eq!(m.next_step(), Some("step two"));
        assert!(m
            .standing_text()
            .ends_with("The next thing to do is: step two."));
        // Every step done points at reporting it.
        m.check_off("step two");
        assert_eq!(m.next_step(), None);
        assert!(m
            .standing_text()
            .contains("Report how it went now with `report_done`"));
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
        m3.check_off("step one");
        assert_ne!(m1.content_fingerprint(), m3.content_fingerprint());
    }

    #[test]
    fn a_random_mission_is_open_non_empty_and_tagged_with_a_known_routine() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
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
                Origin::Lodged { .. } => panic!("bank produces Random origins"),
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
        let facts = Facts {
            makers: &[],
            records: &[],
        };
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
                Origin::Lodged { .. } => panic!("bank produces Random origins"),
            }
        }
    }
}
