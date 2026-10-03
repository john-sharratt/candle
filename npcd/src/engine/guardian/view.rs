//! What a guardian module is shown of a character, and what it may conclude.

use std::collections::HashSet;
use std::time::Duration;

use crate::engine::journal::state::Waiting;
use crate::engine::mission::StepOutcome;

/// One step of the mission a character is carrying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub text: String,
    pub done: bool,
    /// The step is the report back, which only filing the report closes.
    pub reports: bool,
}

/// The open mission, as the guardian reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionView {
    /// What has been asked, without the steps.
    pub prompt: String,
    pub steps: Vec<Step>,
    /// The mission as the character's standing instruction reads: the text a
    /// refresh puts back in front of it.
    pub standing: String,
}

impl MissionView {
    /// The first step not yet ticked off.
    pub fn open_step(&self) -> Option<&str> {
        self.steps.iter().find(|s| !s.done).map(|s| s.text.as_str())
    }

    /// Whether the first step not yet ticked is the report back.
    pub fn open_step_reports(&self) -> bool {
        self.steps
            .iter()
            .find(|s| !s.done)
            .is_some_and(|s| s.reports)
    }

    pub fn done_count(&self) -> usize {
        self.steps.iter().filter(|s| s.done).count()
    }

    /// What changes when the character makes progress: the ask, how many steps
    /// there are and how many are ticked.
    pub fn progress_mark(&self) -> (String, usize, usize) {
        (self.prompt.clone(), self.done_count(), self.steps.len())
    }
}

/// A station in the room a character stands in, as its device addresses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Station {
    pub name: String,
    pub address: String,
    pub verbs: Vec<String>,
}

impl Station {
    /// The whole addresses a character would `invoke`, one per verb.
    pub fn invokable(&self) -> Vec<String> {
        self.verbs
            .iter()
            .map(|v| format!("{}/{v}", self.address))
            .collect()
    }
}

/// A character at the moment it is scanned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpcView {
    pub npc_id: u64,
    /// `None` when the character carries no open mission.
    pub mission: Option<MissionView>,
    /// How long the mission's progress mark has stood still.
    pub since_progress: Duration,
    /// What it can work where it stands.
    pub stations: Vec<Station>,
    /// The acts it took most recently, rendered as the pulse feed shows them,
    /// oldest first.
    pub recent_acts: Vec<String>,
    /// The stretch of its life its journal does not cover yet, once there is
    /// enough of it to be worth asking about.
    pub journal: Option<Waiting>,
}

/// Words shorter than this say nothing about what a text is about.
const MEANINGFUL: usize = 4;

/// The lower-cased words of `text` long enough to say what it is about.
pub fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= MEANINGFUL)
        .map(str::to_lowercase)
        .collect()
}

/// A question a module wants put to the character, with the answers it admits
/// (empty is free text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    pub text: String,
    pub choices: Vec<String>,
}

/// What a character said to a [`Question`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub answer: String,
    /// Why, in the character's words; empty when the question asked for none.
    pub reason: String,
    /// How long the answer took.
    pub ms: u64,
}

/// What is wrong with a character, in the order a ladder cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concern {
    /// What it is doing is not what it was asked to do.
    OffMission,
    /// It is going over the same ground.
    Looping,
    /// Nothing on its mission has moved for too long.
    Stalled,
}

impl Concern {
    pub fn as_str(self) -> &'static str {
        match self {
            Concern::OffMission => "off_mission",
            Concern::Looping => "looping",
            Concern::Stalled => "stalled",
        }
    }
}

/// A module's reading of one character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Healthy,
    Concern(Concern),
    /// The character has dealt with this step and it is not signed off: it
    /// carried it out, or tried and could not.
    TickStep(String, StepOutcome),
    /// The character said whether the stretch its journal does not cover is
    /// worth an entry.
    Journal {
        worth: bool,
    },
}
