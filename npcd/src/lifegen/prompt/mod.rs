//! The prompts of the life generator: each phase's shared prefix, its per-fork
//! instruction, and reading the answers back.
//!
//! The shared prefix is the system prompt that a phase's forks share, so it holds
//! only what is identical for every fork of that phase. A fork's own year, month
//! or day, the text it expands and the consequences it must earn go in its turn.
//! Every instruction below the story tells the writer to expand what it was
//! given, because a sibling that invents people or events breaks the parallel
//! fan-out. The answers come back as short `- a | b | c` lines rather than JSON,
//! since a model produces those reliably after prose.
//!
//! - `decode`: reads a story decode back, removing reasoning and markup.
//! - `parse`: reads each phase's structured answers into typed values.
//! - `prefix`: the shared prefix each phase is primed with.
//! - `render`: consequences and tool calls rendered as writer instructions.
//! - `turns`: the per-fork instructions for years, months, days and outline runs.
//! - `voice`: the character's own idiom for every phase.
//! - `tests_decode`: tests for reading decodes back.
//! - `tests_prompts`: tests for what the prefixes carry.
//! - `tests_turns`: tests for the per-fork instructions.

use serde_json::Value;

use super::consequence::Consequence;
use super::plan::{CastMember, Plan, YearBeat};
use super::seed::Checked;
use crate::engine::authoring::by_name;

mod decode;
mod parse;
mod prefix;
mod render;
#[cfg(test)]
mod tests_decode;
#[cfg(test)]
mod tests_prompts;
#[cfg(test)]
mod tests_turns;
mod turns;
mod voice;

pub use decode::*;
pub use parse::*;
pub use prefix::*;
pub use render::*;
pub use turns::*;
pub use voice::*;
