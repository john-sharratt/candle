//! Turns the generator's answer into a checked mission: one call per kind of work, or a considered decline.
//!
//! Each kind of work has its own call, whose fields are the decisions the work depends on, such as an event's date, what happens and what it must agree with. The engine checks those fields against the corpus and writes the brief and the target path from them, so the model does not write a free-text brief that can skip the decisions. Quotes are checked against their documents, so a contradiction the model cannot quote is refused. Declining is always offered beside every kind and is recorded, so honest agreement between two documents is settled rather than forced into a fix.
//!
//! - `correction`: the call for a correction, naming two clashing sentences.
//! - `event`: the call for a life event and the brief built from it.
//! - `fields`: reads and bounds each field and detects loops.
//! - `grammar`: the answer grammar and the length each field is held to.
//! - `quotes`: checks that quoted spans are present in their source.
//! - `story`: the call for a story and the world and name constraints it is held to.
//! - `text`: closes text at its caps and detects looping text.
//!
//! Tests:
//!
//! - `tests_event`: life-event briefs, refusals and placement.
//! - `tests_grammar`: grammar size, length consistency and call matching.
//! - `tests_story`: correction quotes, possessives and story name reuse.
//! - `tests_text`: loop, kept-field and link behaviour.

use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::path::Path;

use candle_conversation::stencil::{Param as CallParam, ParamType, ToolSpec};
use serde_json::{Map, Value};

use super::corpus::{Corpus, Era, Event};
use super::gates::{Voice, LIFE_MIN_WORDS, STORY_MIN_WORDS};
use super::glossary;
use super::leakage;
use super::material::{belongs, cut, strip_calls, ANCHOR_WORDS, ONLY_EVENT_YEARS};
use super::research;
use super::target::{Kind, Subject, Target};
use crate::engine::life;
use journal_core::tools::arguments;

/// The call that declines.
pub const NOTHING: &str = "no_mission";

pub(super) fn choice(name: &str, arms: &[&str]) -> CallParam {
    CallParam {
        enum_values: Some(arms.iter().map(|a| a.to_string()).collect()),
        ..text(name)
    }
}

/// What a checked answer said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    /// A mission to carry out.
    Mission(Proposal),
    /// Nothing here needs doing, and why.
    Nothing(String),
}

/// A mission as the engine wrote it from a checked answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    /// What the operation is for, in a line.
    pub objective: String,
    /// What the Maker reads under "What has been asked of you".
    pub brief: String,
    /// The mind path to write or change.
    pub writes: String,
    /// The mind paths to read first, every one real.
    pub reads: Vec<String>,
    /// The fewest words the document may be committed at — see
    /// [`crate::engine::mission::Work::min_words`].
    pub min_words: usize,
    /// What the workflow's prompts are filled from: the proposal's own fields
    /// (`title`, `happens`, `turns` …) and the sections the engine wrote for
    /// it (`world-then`, `worlds-words`, `reads` …), by name.
    pub fields: BTreeMap<String, String>,
}

/// Read and check one answer. The error is worded for the model, which is shown
/// it with its refused attempt and answers again.
pub fn check(raw: &str, kind: Kind, target: &Target, corpus: &Corpus) -> Result<Answer, String> {
    let calls: Vec<String> = specs(kind).into_iter().map(|s| s.name).collect();
    let (name, args) = arguments(raw).ok_or_else(|| {
        format!(
            "That was not a call; answer with `{}`.",
            calls.join("` or `")
        )
    })?;
    let field = Fields(&args);
    match (name.as_str(), kind, &target.subject) {
        (NOTHING, _, _) => {
            let why = field.get("why");
            match why.split_whitespace().count() >= 4 {
                true => Ok(Answer::Nothing(why)),
                false => Err("Say in a sentence why there is nothing here to do.".into()),
            }
        }
        (EVENT, Kind::LifeEvent, Subject::Life { who }) => event(&field, who, corpus),
        (CORRECTION, Kind::Contradiction, Subject::Pair { a, b }) => {
            correction(&field, a, b, corpus)
        }
        (STORY, Kind::Gap, Subject::Era { path }) => story(&field, path, corpus),
        (other, _, _) => Err(format!(
            "There is no call named `{other}` here; answer with `{}`.",
            calls.join("` or `")
        )),
    }
}

/// Where a document is written: the bench that writes it and the room it stands
/// in, as `go to {room} on {level}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Desk {
    pub room: String,
    pub level: String,
}

mod correction;
mod event;
mod fields;
mod grammar;
mod quotes;
mod story;
#[cfg(test)]
mod tests_event;
#[cfg(test)]
mod tests_grammar;
#[cfg(test)]
mod tests_story;
#[cfg(test)]
mod tests_text;
mod text;

pub use correction::*;
pub use event::*;
pub(super) use fields::*;
pub use grammar::*;
pub(super) use quotes::*;
pub use story::*;
pub(super) use text::*;
