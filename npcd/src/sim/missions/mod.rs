//! Which character is carrying which mission: the world's half of the mission
//! system, held on the simulation.
//!
//! The mission model owns what a mission is. This module owns whose mission it is,
//! so that commands can reach it and the world can persist it with the rest of its
//! state. A body carries one open mission at a time, lodged missions wait in a
//! queue, and a finished mission stays readable in `done`. Generated missions wait
//! in a pool, and every target they are written for is recorded in a ledger. Each
//! generated mission is a step of an operation, and its target is settled when the
//! operation is.
//!
//! - `active`: the mission each body carries and the steps signed off on it.
//! - `agendas`: each body's queue of missions, the one it works and their order.
//! - `close`: closes missions, turns chapters and files reports.
//! - `draw`: hands a body its next mission from its lodged, table or routine sources.
//! - `ledger`: records the targets the generator found and settles them.
//! - `operation_flow`: moves operations through their workflows and settles their targets.
//! - `tests_missions`: tests for chapters, draws and reports.

use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use super::agenda::{Agendum, Priority, KEEP};
use super::kept_steps::kept_steps;
use super::operations::{Operation, Operations};
use super::orders::{Filed, Notice, Orders, EVERY_POSTER};
use crate::engine::authorize::Authorizer;
use crate::engine::mission::bank::{self, Duty, Facts};
use crate::engine::mission::{Aim, Mission, MissionPrompt, Origin, Outcome, StepOutcome};
use crate::engine::mission_gen::world::{self, Draw, Routine};
use crate::engine::workflow::{Next, Taker, Where, Workflow, STUCK};
use crate::engine::zen_task::Task;

/// The outcome a Maker's `report_done` reports on a step that offers a choice.
pub const PASS: &str = "pass";

/// The outcome a Maker's `report_rejected` reports.
pub const REJECT: &str = "reject";

/// What a posted mission's poster is told of one called off before it was
/// reported.
pub const CALLED_OFF: &str = "It was called off before it was done.";

/// What a posted mission's poster is told of one whose carrier, or the one it
/// was set for, is gone from the world.
pub const GONE: &str = "Whoever it was for is gone, and it was never done.";

/// What a posted mission's poster is told of one dropped because its carrier
/// was started over ([`Missions::start_over`]).
pub const STARTED_OVER: &str = "Its carrier was started over, and it was dropped.";

/// The most missions one [`Missions::start_over`] calls off from a body's
/// hands: a bound on a loop that ends when there is nothing left to cancel.
const START_OVER_MOST: usize = 64;

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
    /// Every operation the table has held.
    #[serde(default)]
    operations: Operations,
    /// How many missions each body has closed — reported, rejected or called
    /// off. Each is a chapter of its working life, and a character starts a new
    /// conversation with each (`Minds::think`), its journal carrying the work
    /// across.
    #[serde(default)]
    closed: BTreeMap<String, u64>,
    /// The mission each body closed last, however it closed — what its journal
    /// is told it finished as the chapter turns.
    #[serde(default)]
    last_closed: BTreeMap<String, Mission>,
    /// Bodies stood down because the operation they were carrying a stage of
    /// was called off, not yet told — see [`Self::take_stood_down`].
    #[serde(default)]
    stood_down: Vec<String>,
    /// Bodies the world gives no orders — a personality with `takes_orders:
    /// false` — see [`Self::set_unordered`].
    #[serde(default)]
    unordered: BTreeSet<String>,
    /// The missions posted on the table, from the poster's side — see
    /// [`super::orders`].
    #[serde(default)]
    orders: Orders,
    /// When each standing mission was last done in each room, by `id@room` —
    /// `None` for one reported done and not yet stamped by the table's loop,
    /// which keeps the clock ([`Self::stamp_standing`]).
    #[serde(default)]
    standing: BTreeMap<String, Option<u64>>,
    /// When each standing mission last failed in each room, by `id@room`,
    /// stamped the same way — until it is done there.
    #[serde(default)]
    standing_failed: BTreeMap<String, Option<u64>>,
    /// The world's own routines, from `<mind>/missions/<world>.yaml` —
    /// drawn in place of the built-in [`bank`] when there are any. The mind's,
    /// read afresh rather than saved, like the workflows.
    #[serde(skip)]
    routines: Vec<Routine>,
    /// Whether a character with nothing set for it draws a routine — the
    /// mind's, with the routines.
    #[serde(skip)]
    draw: Draw,
    /// Who may approve a critical step (`docs/mission_templates.md` §6), by
    /// account — the mind's, read afresh like the routines.
    #[serde(skip)]
    authorizers: Vec<Authorizer>,
    /// Whether the world's order-giver may author new templates — the mind's,
    /// read afresh like the routines.
    #[serde(skip)]
    authoring: bool,
    /// The missions on each agenda set aside behind the one worked, by body
    /// (`docs/zen_moon_mind.md` §17.1): the worked one is the body's
    /// [`Self::active`].
    #[serde(default)]
    held: BTreeMap<String, Vec<Mission>>,
    /// How many missions have come onto any agenda — what orders missions of
    /// one priority, oldest first.
    #[serde(default)]
    agenda_count: u64,
    /// The bodies whose characters are suspended — hands the table cannot
    /// give work to. The cast's, told by the world's authority as it gives
    /// work out ([`Self::set_suspended`]), never saved.
    #[serde(skip)]
    suspended: BTreeSet<String>,
}

/// Whether `m` is one of the world's standing missions
/// (`docs/zen_moon_mind.md` §12).
fn is_standing(m: &Mission) -> bool {
    matches!(m.origin, Origin::Standing { .. })
}

/// The target a generated mission is about.
fn target_of(m: &Mission) -> Option<&str> {
    match &m.origin {
        Origin::Generated { target, .. } => Some(target),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../missions_operation_tests.rs"]
mod operation_tests;

mod active;
mod agendas;
mod close;
mod draw;
mod ledger;
mod operation_flow;
#[cfg(test)]
mod tests_missions;

pub use draw::*;
pub use ledger::*;
