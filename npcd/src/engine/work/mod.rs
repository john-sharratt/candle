//! The station and bench acts: what a character does through the record's
//! stations, and the working loop over them.
//!
//! Most acts are one store's state machine over the record: take a thing, write
//! into it, offer it, file it. The dispatch is short and the rules live in the
//! store, so one defect is one defect. A refusal is written for the character to
//! read and act on, and it says what would have worked.
//!
//! - `args`: reading the subject and the contents of an act.
//! - `dispatch`: routing each act to its handler.
//! - `bench_acts`: the document and working-set acts.
//! - `craft`: the library, sheet and likeness acts.
//! - `mission_acts`: taking up, progressing and reporting missions.
//! - `report`: what stands between a body and a report.
//! - Tests: `tests_authoring`, `tests_documents`, `tests_mission`, `tests_station`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use doc_units::{Doc, Edit, Op, Place};
use serde_json::{Map, Value};

use crate::engine::act::Act;
use crate::engine::archive;
use crate::engine::body::{self, Outcome};
use crate::engine::chronology;
use crate::engine::enact::order_refusal;
use crate::engine::mission::{progress_line, time_step, Aim, DocStep, Mission, Outcome as Verdict};
use crate::engine::mission_gen::config::{
    CHANGED, MERGE_PUBLISH, MERGE_STATUS, MERGE_SYNC, ROUTINE_PARSES,
};
use crate::engine::mission_gen::copied::{self, BRIEF_COPIED};
use crate::engine::mission_gen::world::Routine;
use crate::engine::mission_gen::{gates, leakage, rejection, write_template};
use crate::engine::order_table;
use crate::engine::passage;
use crate::engine::people;
use crate::engine::workflow::OnFailed;
use crate::engine::workflow::PUBLISHED;
use crate::opstore::owner::Owner;
use crate::sim::record::{slug_of, Condition, Item, Kind, State};
use crate::sim::shelf::Shelf;
use crate::sim::Sim;
use crate::world::Hosted;
use npc_map::route;
use npc_map::schema::Where;
use npc_map::text::list;

mod args;
mod bench_acts;
mod craft;
mod dispatch;
mod mission_acts;
mod report;
#[cfg(test)]
mod tests_authoring;
#[cfg(test)]
mod tests_documents;
#[cfg(test)]
mod tests_mission;
#[cfg(test)]
mod tests_station;

pub(crate) use args::*;
use bench_acts::*;
use craft::*;
pub use dispatch::*;
pub use mission_acts::*;
pub use report::*;
