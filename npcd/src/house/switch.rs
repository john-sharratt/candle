//! A switch of the house's mode, carried out (`docs/house_modes.md` §3):
//! first the decks — rekordbox opened for a party and closed for anything
//! else, asked of `zenj` over the gateway's bus — then each device the mode
//! names, through its bridge as its own act would be, and last the mode
//! itself, kept with the world.
//!
//! **Leaving a party is refused while a deck plays**, before anything is
//! touched: the music stops at the decks, never because the stereo went to
//! standby under it. Everything else is done as far as it can be: a device
//! that does not answer, or a DJ that is not running, is said in the answer,
//! and the house is in its new mode.

use std::sync::Arc;
use std::time::Duration;

use agent::client::Agents;
use cluster::member::AppMember;
use host::cluster::bus::{ask, Unasked};
use serde_json::{json, Map, Value};

use super::mode::Mode;
use super::table::Step;
use crate::api::now_ms;
use crate::bridge::{self, parts_here, Done};
use crate::world::Hosted;

/// What the house asks rekordbox on: `zenj`'s, on BlackWorld.
pub const DECKS_TOPIC: &str = "dj/rekordbox";

/// How long the decks are waited for: inside the bus's own 30 s.
pub const DECKS_WAIT: Duration = Duration::from_secs(28);

/// What became of the decks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decks {
    /// Done, in `zenj`'s words.
    Done(String),
    /// `zenj` would not: a deck is playing, say.
    Refused(String),
    /// Nobody answered, or the ask could not be made.
    Unreached(String),
}

/// What `zenj`'s answer, or its absence, comes to.
pub fn decks_heard(answer: Result<Value, Unasked>) -> Decks {
    match answer {
        Ok(v) => {
            let line = v["line"].as_str().unwrap_or("").to_string();
            match v["ok"].as_bool() {
                Some(true) => Decks::Done(line),
                _ => Decks::Refused(line),
            }
        }
        Err(Unasked::NoSubscriber) => {
            Decks::Unreached("zenj is not running on BlackWorld, so rekordbox was not asked".into())
        }
        Err(Unasked::Timeout) => Decks::Unreached("zenj did not answer in time".into()),
        Err(Unasked::Failed(why)) => Decks::Unreached(format!("zenj could not be asked: {why}")),
    }
}

/// The act asked of the decks for `mode`.
pub fn decks_act(mode: Mode) -> &'static str {
    match mode.decks_open() {
        true => "open",
        false => "close",
    }
}

/// Ask `zenj`, through `member`'s gateways, to open or close rekordbox.
async fn ask_decks(member: Option<Arc<AppMember>>, act: &str) -> Decks {
    let Some(member) = member else {
        return Decks::Unreached("this member is in no cluster, so zenj cannot be asked".into());
    };
    let id = format!("{:x}", now_ms());
    let reply = format!("{DECKS_TOPIC}/result/{id}");
    let payload = json!({"id": id, "act": act});
    decks_heard(ask(&member, DECKS_TOPIC, payload, &reply, DECKS_WAIT).await)
}

/// The bridge act a step is, for a device of `profile`: its tool name, and
/// its arguments with the device named.
pub fn call_of(step: &Step, profile: &str) -> (String, Map<String, Value>) {
    let mut args = step.args.clone();
    args.insert("device".into(), Value::from(step.device.clone()));
    (format!("{profile}_{}", step.act), args)
}

/// One step done through the device's bridge, for `body` where it stands;
/// its line.
async fn work(hosted: &Hosted, body: &str, step: &Step, agents: Option<Agents>) -> String {
    let link = parts_here(hosted, body).into_iter().find_map(|p| {
        p.name
            .eq_ignore_ascii_case(&step.device)
            .then_some(p.link)
            .flatten()
    });
    let Some(link) = link else {
        return format!(
            "{} {}: there is no {} here that the world can work",
            step.device, step.act, step.device
        );
    };
    let (tool, args) = call_of(step, &link.profile);
    let done: Done = bridge::enact(hosted, body, &tool, &args, agents).await;
    let first = done.answer.lines().next().unwrap_or("").to_string();
    match done.landed {
        true => format!("{} {}: {first}", step.device, step.act),
        false => format!("{} {} did not happen: {first}", step.device, step.act),
    }
}

/// Set the house to `mode` for `body`, which stands in the house's room.
/// Whether it landed, and what came of each part.
pub async fn switch(
    hosted: &Hosted,
    body: &str,
    mode: Mode,
    agents: Option<Agents>,
    member: Option<Arc<AppMember>>,
) -> (bool, String) {
    let Some(table) = hosted.sim(|s| s.house.table().cloned()) else {
        return (false, "This world has no house to set.".into());
    };
    let place = hosted.standpoint_of(body);
    if place != table.room {
        return (
            false,
            format!(
                "The house's devices are in {}: set its mode from there.",
                table.room
            ),
        );
    }
    let mut lines = Vec::new();
    match ask_decks(member, decks_act(mode)).await {
        Decks::Done(line) => lines.push(format!("the decks: {line}")),
        Decks::Refused(why) if !mode.decks_open() => {
            return (
                false,
                format!("The house stays as it is: the decks would not close — {why}."),
            );
        }
        Decks::Refused(why) => lines.push(format!("the decks did not open: {why}")),
        Decks::Unreached(why) => lines.push(format!("the decks: {why}")),
    }
    for step in table.steps(mode) {
        lines.push(work(hosted, body, step, agents.clone()).await);
    }
    hosted.with_sim(|s| s.house.set(mode, body, now_ms()));
    tracing::info!(world = hosted.id(), mode = mode.name(), by = body, "house: mode set");
    let mut answer = format!("The house is in {}: {}.", mode.name(), mode.what());
    for l in lines {
        answer.push_str("\n- ");
        answer.push_str(&l);
    }
    (true, answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A party opens the decks and every other mode closes them.**
    #[test]
    fn a_party_opens_the_decks_and_the_rest_close_them() {
        assert_eq!(decks_act(Mode::Party), "open");
        for m in Mode::ALL.into_iter().filter(|m| *m != Mode::Party) {
            assert_eq!(decks_act(m), "close", "{m:?}");
        }
    }

    /// **zenj's answer reads as what became of the decks**, and a zenj that
    /// is not running is said plainly, not waited for.
    #[test]
    fn zenjs_answer_reads_as_what_became_of_the_decks() {
        assert_eq!(
            decks_heard(Ok(json!({"ok": true, "line": "rekordbox is open"}))),
            Decks::Done("rekordbox is open".into())
        );
        assert_eq!(
            decks_heard(Ok(json!({"ok": false, "line": "a deck is playing"}))),
            Decks::Refused("a deck is playing".into())
        );
        assert_eq!(
            decks_heard(Err(Unasked::NoSubscriber)),
            Decks::Unreached("zenj is not running on BlackWorld, so rekordbox was not asked".into())
        );
        assert_eq!(
            decks_heard(Err(Unasked::Timeout)),
            Decks::Unreached("zenj did not answer in time".into())
        );
    }

    /// **A step is its device's bridge act**, named by the profile the part
    /// is linked to, with the device named among its arguments.
    #[test]
    fn a_step_is_its_devices_bridge_act() {
        let mut args = Map::new();
        args.insert("source".into(), Value::from("tv"));
        let step = Step {
            device: "stereo".into(),
            act: "input".into(),
            args,
        };
        let (tool, args) = call_of(&step, "receiver");
        assert_eq!(tool, "receiver_input");
        assert_eq!(
            Value::Object(args),
            json!({"source": "tv", "device": "stereo"})
        );
    }
}
