//! A mission: what a character has been asked to do, the steps it works through,
//! and the report it files when it is done.
//!
//! A mission is collected at the command table, from a lodged request or from a
//! random routine drawn from the bank, worked by ticking off its steps as the engine
//! or the character sees them done, and completed by a report. The module is
//! deliberately pure, so a mission's shape, its steps and its rendered text can be
//! tested without a running engine. The text a character reads is rendered from the
//! mission itself, so a change to its steps re-fingerprints its prompt section,
//! and a filed report does not. A mission with a document is written up only when
//! this character committed that document after taking the mission up.
//!
//! - `lifecycle`: the mission's steps, their sign-off, and when it may be reported.
//! - `step_reading`: what a step asks of the world, matched by name.
//! - `compass`: the next-step line and where the people and machines it names are.
//! - `tests_compass`, `tests_steps`: tests.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::engine::mission_gen::world::Routine;
use crate::engine::order_table::TABLE_LIMIT;
use crate::engine::zen_task::{is_task_step, Task};
use crate::sim::agenda::Agendum;

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
    /// Whether the step is irreversible and may be taken only with an
    /// authorizer's go-ahead (`docs/mission_templates.md` §6). The engine never
    /// signs a critical step off from what it sees, and the carrier may not tick
    /// it itself: only [`Mission::authorize`], called once an authorizer has
    /// approved, closes it. An undone critical step keeps [`Self::all_todos_done`]
    /// false, so the mission holds at it until the go-ahead lands.
    #[serde(default)]
    pub critical: bool,
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
            critical: false,
        }
    }

    /// A fresh, un-ticked step that is irreversible — taken only with an
    /// authorizer's go-ahead. See [`Todo::critical`].
    pub fn critical(text: impl Into<String>) -> Self {
        Self {
            critical: true,
            ..Self::new(text)
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
    /// Posted on the order table by a character that takes no orders (`by` is
    /// its name, `poster` its body), whose report goes back to it — see
    /// `sim::orders` and `docs/zen_moon_mind.md` §8. `job` names the standing
    /// job it was posted from, when it was one of the world's routines.
    Ordered {
        by: String,
        poster: String,
        #[serde(default)]
        job: Option<String>,
    },
    /// Put up by the table itself, on a schedule: the standing mission `id`,
    /// for the room `room` (`area/node`) — see `engine::mission_gen::standing`
    /// and `docs/zen_moon_mind.md` §12.
    Standing { id: String, room: String },
    /// Made by its carrier for its own agenda (`mission_add`,
    /// `docs/zen_moon_mind.md` §17.2).
    Own,
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
    /// Whether the world has refused the carrier something since the desk last
    /// turned a `report_stuck` away — what makes that turn-away count toward
    /// believing the next one ([`Self::refuse_stuck_on_the_way`]).
    #[serde(default)]
    pub refused_since: bool,
    /// How many `report_stuck` in a row the desk has turned away with the way
    /// open and nothing tried — see [`Self::refuse_stuck_on_the_way`].
    #[serde(default)]
    pub stuck_untried: u32,
    /// A digest of the mission's document as it stood when the desk last
    /// turned a `report_stuck` away over its faults — what tells a rewrite from
    /// none ([`Self::refuse_stuck_over_faults`]).
    #[serde(default)]
    pub faulted_draft: Option<u64>,
    /// What the table found wrong with the mission's document the last time it
    /// was reported, until a report of it is taken — what the compass says
    /// beside the way to the desk ([`refused_compass`]).
    #[serde(default)]
    pub gate_refusal: Option<String>,
    /// The documents the mission is about, when it is work on the record.
    #[serde(default)]
    pub work: Option<Work>,
    /// The year of the world's history its carrier works in, set at a time
    /// machine ([`Self::travelled`]). Nothing after it reaches the carrier's
    /// recall while the mission is carried; it ends with the mission.
    #[serde(default)]
    pub year: Option<u32>,
    /// The world-building acts carrying it grants (`docs/zen_moon_mind.md`
    /// §14), in the room it is for — a standing mission's own.
    #[serde(default)]
    pub grants: Vec<String>,
    /// Its place on the carrier's agenda — its title and priority — when the
    /// carrier keeps one (`docs/zen_moon_mind.md` §17.1). `None` for a mission
    /// carried alone.
    #[serde(default)]
    pub agendum: Option<Agendum>,
    /// The task of Zen's this mission drives, once its carrier started one
    /// (`docs/mission_templates.md` §7).
    #[serde(default)]
    pub task: Option<Task>,
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
                        "Message {who}, wherever they are working, and review what they are \
                         doing. Ask about it, understand it, and tell them plainly what you \
                         think — what is good in it and what you would do differently."
                    ),
                    vec![
                        Todo::new(format!("ask {who} what they are working on")),
                        Todo::new("form your own view of it"),
                        Todo::new(format!("message {who} your feedback")),
                    ],
                    origin("visit-and-review"),
                ))
            }
            "hear-them-out" => {
                let who = pick(facts.makers, seed, "hear")?;
                Some(Mission::new(
                    format!(
                        "Message {who} and hear them out — properly, for a good while, about \
                         what they have been working on and what they have concluded from it. \
                         Do not settle for a line and away."
                    ),
                    vec![
                        Todo::new(format!("message {who}")),
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
                         Message both, learn what each is doing, and come back able to say \
                         where their work meets and where it pulls apart."
                    ),
                    vec![
                        Todo::new(format!("message {a} and learn their work")),
                        Todo::new(format!("message {b} and learn their work")),
                        Todo::new("settle where the two meet and where they differ"),
                    ],
                    origin("compare-notes"),
                ))
            }
            "trace-a-thread" => {
                let who = pick(facts.makers, seed, "thread")?;
                Some(Mission::new(
                    format!(
                        "Something {who} said, or made, has a thread you want to follow. \
                         Message them, ask about it, and follow it wherever it leads — another \
                         person, a record, a room. Come back knowing where it goes."
                    ),
                    vec![
                        Todo::new(format!("message {who} and ask about it")),
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

mod compass;
mod lifecycle;
mod step_reading;
#[cfg(test)]
mod tests_compass;
#[cfg(test)]
mod tests_steps;

pub use compass::*;
pub use lifecycle::*;
pub use step_reading::*;
