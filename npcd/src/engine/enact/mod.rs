//! Performs the acts that change what the world holds, rather than only the shape of the world.
//!
//! Each act here either changes the simulation and reports what changed, or refuses in the second person and
//! says what would have made it work. Refusals still matter where the grammar prevents the mistake, because the
//! API and the harness are not grammar-constrained. A mind that takes no orders has no body to move, so none of
//! these acts is offered to it. An act that is not this module's is passed on rather than performed here.
//!
//! - `carry`: giving, releasing, equipping, using and gathering.
//! - `command`: commanding the tower and recording verdicts.
//! - `engage`: stances, and refusing an operation already in place.
//! - `messaging`: phone messages, groups, invitations and goodbyes.
//! - `movement`: attention moves, and the lift.
//! - `notices`: notices on a board, claims and promises.
//! - `people`: acts done to a person, and the names they are known by.
//! - `reading`: reading a room or a phone, and recalling a body home.
//! - `scan`: looking at a room, a floor, or the stations of a room.
//! - `tests_acts`: tests of the acts' effects on the world.
//! - `tests_phone`: tests of the acts made on the phone.
//! - `tests_scan`: tests of the scan act.
//! - `tests_tower`: tests of the tower's commands.

use npc_map::world::Where;
use serde_json::{Map, Value};

use crate::authority::ops::Voiced;
use crate::engine::act::Act;
use crate::engine::body::{destinations, refusal, rooms, Outcome};
use crate::engine::survey;
use crate::engine::tools::{CHANNEL_WAY, PRIVATE, SELF};
use crate::engine::whereabouts;
use crate::engine::whole;
use crate::sim::field::{Resource, Stance};
use crate::sim::item::{Item, Kind as ItemKind};
use crate::sim::ledger::OrderRefused;
use crate::sim::phone::{now_ms, Kind as ThreadKind, Said, Threads, Who, ALL_MESSAGES};
use crate::sim::purpose::Purpose;
use crate::sim::tower::{on_map, Coord, Posture, DRILL_COST_PER_METRE, EDGE, SIEGE_COST};
use crate::sim::Sim;
use crate::world::Hosted;

/// What a mind that takes no orders is told when it asks to move
/// (`docs/zen_everywhere.md` §4).
pub const NO_BODY: &str = "You have no body to move and are nowhere to move from. What you need \
                           to know reaches you, or you ask for it: post work on the order table, \
                           or message whoever can tell you.";

/// The acts this module performs.
pub fn is_mine(tool: &str) -> bool {
    matches!(
        tool,
        "act"
            | "sleep"
            | "give"
            | "read"
            | "post_notice"
            | "claim"
            | "release"
            | "equip"
            | "use"
            | "gather"
            | "engage"
            | "operate"
            | "recall"
            | "focus"
            | "lift_call"
            | "lift_use"
            | "scan"
            | "command_tower"
            | "produce"
            | "promise"
            | "remind"
            | "record_verdict"
            | "message"
            | "reply"
            | "invite"
            | "open_group"
            | "sign_off"
            | "reach_out"
    )
}

/// Perform one act. [`Outcome::NotOfTheBody`] for anything not [`is_mine`].
pub fn perform(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    perform_said(hosted, body, act, None)
}

/// [`perform`], where `said` is the line the lease holder voiced for a phone
/// act — what lands on the thread in place of its intent.
pub fn perform_said(hosted: &Hosted, body: &str, act: &Act, said: Option<&Voiced>) -> Outcome {
    let a = &act.args;
    match act.tool {
        // A mind that takes no orders has no body to move: none of these is
        // offered it, and each is refused when asked for anyway.
        "recall" | "focus" | "lift_call" | "lift_use"
            if hosted.sim(|s| s.missions.is_unordered(body)) =>
        {
            Outcome::Refused(NO_BODY.into())
        }
        "act" => act_on(hosted, body, a),
        "sleep" => sleep(hosted, body, a),
        "give" => give(hosted, body, a),
        "read" => read(hosted, body, a),
        "post_notice" => post_notice(hosted, body, a),
        "claim" => claim(hosted, body, a),
        "release" => release(hosted, body),
        "equip" => equip(hosted, body, a),
        "use" => use_it(hosted, body, a),
        "gather" => gather(hosted, body, a),
        "engage" => engage(hosted, body, a),
        "operate" => operate(hosted, body, a),
        "recall" => recall(hosted, body),
        "focus" => focus(hosted, body, a),
        // A body that moves by attention has no lift to call or ride: the act is
        // not offered it, and refused when it is asked for anyway.
        "lift_call" | "lift_use" if hosted.sim(|s| s.attends(body)) => Outcome::Refused(
            "You do not ride your own lift: turn your attention there with `focus`.".into(),
        ),
        "lift_call" => lift_call(hosted, body),
        "lift_use" => lift_use(hosted, body, a),
        "scan" => scan(hosted, body, a),
        "command_tower" => command_tower(hosted, body, a),
        "produce" => produce(hosted, a),
        "promise" => promise(hosted, body, a),
        "remind" => remind(hosted, body, a),
        "record_verdict" => record_verdict(hosted, body, a),
        // The same send on the thread named: what differs is the set the
        // thread is named from — of the kind `via` said for `message`, the
        // ones waiting on the character for `reply` (`tools::Choices`).
        "message" => match via(a) {
            Ok(kind) => message(hosted, body, a, Some(kind), said),
            Err(refused) => refused,
        },
        "reply" => message(hosted, body, a, None, said),
        "invite" => invite(hosted, body, a),
        "open_group" => open_group(hosted, body, a, said),
        "sign_off" => sign_off(hosted, body, a, said),
        "reach_out" => reach_out(hosted, body, a),
        _ => Outcome::NotOfTheBody,
    }
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// A whole number written as a JSON number, which is how an `integer`
/// parameter arrives. A string, even one of digits, is not one: the grammar
/// writes an integer digit by digit, so a string here came from outside it.
fn whole(args: &Map<String, Value>, key: &str) -> Option<i64> {
    args.get(key)?.as_i64()
}

/// A count written as a string, the way an optional quantity reaches this
/// catalog: left out, or empty, it is one, and so is an unparseable one, which
/// is what a character that wrote "a few" meant.
fn count(args: &Map<String, Value>, key: &str) -> u32 {
    text(args, key)
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(1)
        .max(1)
}

mod carry;
mod command;
mod engage;
mod messaging;
mod movement;
mod notices;
mod people;
mod reading;
mod scan;
#[cfg(test)]
mod tests_acts;
#[cfg(test)]
mod tests_phone;
#[cfg(test)]
mod tests_scan;
#[cfg(test)]
mod tests_tower;

use carry::*;
use command::*;
use engage::*;
use messaging::*;
use movement::*;
pub(crate) use notices::*;
use people::*;
use reading::*;
use scan::*;
