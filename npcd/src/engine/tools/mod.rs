//! The act vocabulary a character uses to do things in the world, and the grammar,
//! prompt and tests built from it.
//!
//! Acts carry intent rather than output: `tell` takes what the character means to
//! say and the narrator renders the words, and no act writes to the belief layer,
//! which changes only through the sleep clock. Every act is offered only when the
//! character's room, companions, body and mission make it possible. The grammar,
//! the prompt and the tests all read the same world state, so the list a character
//! reads and the mask it decodes under agree. Each act carries calibration
//! examples, and a tool without one fails a test.
//!
//! - `act_tools`: speech, attention and movement acts, pitched by who hears them.
//! - `choices`: the closed sets a call's arguments are chosen from.
//! - `compile`: the grammar a turn is decoded under, and its path budget.
//! - `lists`: an act's list argument — a mission's steps — held, published and read.
//! - `offers`: the fields a chosen value brings, and the live sets the world offers.
//! - `show`: which acts a turn's prompt describes.
//! - `within_tools`: what is true where a character stands.
//! - `types`: planes, availability, modes, parameters and examples.
//! - `tests_grammar`, `tests_modes`, `tests_show`: tests of the grammar, the modes and the prompt.

use candle_conversation::projection::{
    Builder, SelectionRule, SelectionState, FORCE_TOOL_SELECTOR, FORCE_TOOL_SEPARATOR,
};
use candle_conversation::stencil::{Param as StencilParam, ParamType, Pick, ToolSpec};
use serde::Serialize;

use crate::bridge::BRIDGE_ACTS;
use crate::engine::agenda::{self, ordered, AGENDA_ACTS};
use crate::engine::build::BUILD_ACTS;
use crate::engine::invoke_body::{self, Invokable};
use crate::engine::talk::hand_off::HAND_OFF_ACTS;
use crate::engine::terminal::TERMINAL_ACTS;
use crate::engine::zen_task::TASK_ACTS;
use crate::engine::{acts, bench, mission, mission_acts, station};
use crate::house::HOUSE_ACTS;
use crate::sim::agenda::Priority;
use crate::sim::phone::Kind as ThreadKind;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Tool {
    pub name: &'static str,
    pub category: &'static str,
    pub plane: Plane,
    pub availability: Availability,
    pub description: &'static str,
    pub params: &'static [Param],
    pub examples: &'static [Example],
    /// The parts this act attaches to, by the id the map gives them.
    ///
    /// **The tool names the station, not the other way round.** The parts used
    /// to carry a `tools:` list, which meant three things went wrong at once:
    /// adding one act to six stations was six edits in six files; a typo named
    /// no real act and silently produced nothing; and `npc-map` — a crate whose
    /// whole job is describing buildings — carried this engine's vocabulary,
    /// which is a layer inversion.
    ///
    /// Declared here, an act's reach sits beside its description and examples,
    /// the act's own name is a Rust identifier the compiler checks, and the map
    /// goes back to saying only what a building *contains*.
    ///
    /// Empty for everything a body can do anywhere. Non-empty implies
    /// [`Availability::AtPart`], and [`tests::a_station_act_names_where_it_is`]
    /// holds the two together.
    pub at: &'static [&'static str],
}

/// The generic catalog: what every character can do, before any world adds to it.
///
/// # Every act here changes something
///
/// **That is the entry requirement, and it was not always met.** The catalog
/// carried twenty-one acts of which five reached the world: `say`, `tell`,
/// `move_to`, `follow`, `flee`. The other sixteen — `face`, `express`, `listen`,
/// `inspect`, `greet`, `offer`, `refuse`, `threaten`, `note_concern`,
/// `set_intent`, `broadcast_strategy`, `end_interaction` — were handled nowhere.
/// A character choosing one spent a whole turn and changed nothing: not the
/// world, not its own state, not what anybody perceived.
///
/// That is invisible from every angle the daemon has. The act is well-formed,
/// the tick succeeds, the feed shows a character doing something. It cost a
/// live cast a hundred turns: one character alternating a no-op `face` with the
/// same sentence, a second answering by walking to the room it stood in, a
/// third turning to look at furniture. None of them was malfunctioning. They
/// were choosing acts that could not have an effect, from a list that offered
/// sixteen of them.
///
/// It also made the grammar worse, not better. The stencil masks the tool name
/// to exactly these entries, so sixteen dead branches were sixteen ways for a
/// valid call to be a wasted turn — and widening what the prompt shows would
/// only have advertised them.
///
/// So the rule is the list: an act belongs here when it reaches the world,
/// changes the character's own state, or returns something the character did
/// not have. Names that were a second word for one of these went to the act
/// that does the work — `express` into [`GESTURE`], `listen` and `inspect` into
/// [`OBSERVE`], `greet`/`offer`/`refuse`/`threaten` into [`TELL`] and [`SHOUT`],
/// whose `manner` already carries the stance that distinguished them.
/// Every act a character can take, from every source.
///
/// Three groups, kept in three files because they answer to three different
/// things: the body's acts are always reachable, the station acts are reachable
/// because of what is standing in the room, and the bench acts are the working
/// loop over what a station holds. Composed here rather than declared here so
/// each file owns its own vocabulary.
pub static CATALOG: std::sync::LazyLock<Vec<Tool>> = std::sync::LazyLock::new(|| {
    BODY_ACTS
        .iter()
        .chain(acts::WORLD_ACTS)
        .chain(station::STATION_ACTS)
        .chain(bench::BENCH_ACTS)
        .chain(mission_acts::MISSION_ACTS)
        .chain(AGENDA_ACTS)
        .chain(BRIDGE_ACTS)
        .chain(TERMINAL_ACTS)
        .chain(HOUSE_ACTS)
        .chain(BUILD_ACTS)
        .chain(TASK_ACTS)
        .chain(HAND_OFF_ACTS)
        .copied()
        .collect()
});

/// What a body can do anywhere, or because of who is standing next to it.
const BODY_ACTS: &[Tool] = &[
    // Speech, attention, movement — what a body does with other bodies and
    // with rooms.
    TELL, WHISPER, SHOUT, ASK, GESTURE, MOVE_TO, FOLLOW, REFLECT, SEND_IMAGE,
    // The effector device — the fixed verb onto the world. Always offered, like
    // the body's own acts, and for the same reason: what it can reach is a fact
    // about where the body stands, not about whether the tool exists. See
    // [`INVOKE`].
    INVOKE,
];

mod act_tools;
mod choices;
mod compile;
mod lists;
mod offers;
mod show;
#[cfg(test)]
mod tests_grammar;
#[cfg(test)]
mod tests_modes;
#[cfg(test)]
mod tests_show;
mod types;
mod within_tools;

pub use act_tools::*;
pub use choices::*;
pub use compile::*;
pub use lists::*;
pub use offers::*;
pub use show::*;
pub use types::*;
pub use within_tools::*;
