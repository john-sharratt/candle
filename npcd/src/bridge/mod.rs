//! The world's bridges to real things (`docs/zen_moon_mind.md` §13, §15):
//! the survey of the house network from a room's house link, and one profile
//! per kind of device the world can work.
//!
//! The acts are [`acts`]'s, each bound to what stands in the room; they reach
//! the network, so the runtime awaits them ([`enact`]), off the world's lock.

pub mod acts;
pub mod discover;
pub mod finding;

use agent::client::Agents;
use agent::wire::BridgeCall;
use effector_bridge::work::profile_act;
use npc_map::part::PartLink;
use npc_map::schema::Where;
use serde_json::{Map, Value};

use crate::engine::mission_gen::standing::HOUSE_LINK;
use crate::world::Hosted;
use finding::{RECEIVER, SONOS};

pub use acts::{is_mine, BRIDGE_ACTS};

/// A part standing where a body is.
pub struct Here {
    /// What it is called.
    pub name: String,
    /// The part's id.
    pub id: String,
    /// The device it stands for, if it stands for one.
    pub link: Option<PartLink>,
}

/// The parts standing where `body` is, when it stands in the mirror of a real
/// place — none anywhere else: a bridge reaches real things only from a room
/// that mirrors where they are.
pub fn parts_here(hosted: &Hosted, body: &str) -> Vec<Here> {
    let place = hosted.standpoint_of(body);
    hosted.read(|w| {
        let Some(at) = Where::parse(&place) else {
            return Vec::new();
        };
        if !w.map().is_mirrored(&at.area) {
            return Vec::new();
        }
        let Some(node) = w.node(&at) else {
            return Vec::new();
        };
        w.map()
            .parts_at(node)
            .map(|(p, _)| Here {
                name: p.name.clone(),
                id: p.id.clone(),
                link: p.link.clone(),
            })
            .collect()
    })
}

/// The names of the house links among `parts`.
pub fn links_of(parts: &[Here]) -> Vec<String> {
    parts
        .iter()
        .filter(|p| p.id == HOUSE_LINK)
        .map(|p| p.name.clone())
        .collect()
}

/// The names of the parts among `parts` that stand for a device of `profile`.
pub fn on_profile(parts: &[Here], profile: &str) -> Vec<String> {
    parts
        .iter()
        .filter(|p| p.link.as_ref().is_some_and(|l| l.profile == profile))
        .map(|p| p.name.clone())
        .collect()
}

/// The names of the parts among `parts` that stand for a Sonos player.
pub fn sonos_of(parts: &[Here]) -> Vec<String> {
    on_profile(parts, SONOS)
}

/// The names of the parts among `parts` that stand for an AV receiver.
pub fn receiver_of(parts: &[Here]) -> Vec<String> {
    on_profile(parts, RECEIVER)
}

/// The part here named `name` that stands for a device of `profile`, and its
/// address.
fn linked_here(hosted: &Hosted, body: &str, name: &str, profile: &str) -> Option<PartLink> {
    parts_here(hosted, body).into_iter().find_map(|p| {
        let link = p.link?;
        (p.name.eq_ignore_ascii_case(name) && link.profile == profile).then_some(link)
    })
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// The act's arguments with the device's stored credentials and hardware
/// address folded in — so the agent is handed what it needs to work the device
/// (a webOS key, an Android TV certificate, the wake MAC) and the world keeps
/// the secrets, not the stateless agent. An argument the caller set wins.
fn with_link(args: &Map<String, Value>, link: &PartLink) -> Map<String, Value> {
    let mut merged = args.clone();
    for (name, value) in &link.creds {
        merged
            .entry(name.clone())
            .or_insert_with(|| Value::from(value.clone()));
    }
    if !link.hardware.is_empty() {
        merged
            .entry("mac".to_string())
            .or_insert_with(|| Value::from(link.hardware.clone()));
    }
    merged
}

/// What came of a bridge act: whether it landed, and the line its body reads.
pub struct Done {
    pub landed: bool,
    pub answer: String,
}

impl Done {
    fn did(answer: impl Into<String>) -> Done {
        Done {
            landed: true,
            answer: answer.into(),
        }
    }
    fn refused(answer: impl Into<String>) -> Done {
        Done {
            landed: false,
            answer: answer.into(),
        }
    }
}

/// Enact bridge act `tool` for `body`: survey the house network from the link
/// here [done on this member], or work a device here by handing a [`BridgeCall`]
/// to an in-network agent (`docs/zen_moon_mind.md` §15) — the device I/O runs
/// on the agent, the way the git jobs do, not on whatever member the character
/// happens to run on.
pub async fn enact(
    hosted: &Hosted,
    body: &str,
    tool: &str,
    args: &Map<String, Value>,
    agents: Option<Agents>,
) -> Done {
    match tool {
        "survey_network" => {
            let Some(at) = text(args, "at") else {
                return Done::refused("Say which house link to survey from.");
            };
            if !links_of(&parts_here(hosted, body))
                .iter()
                .any(|l| l.eq_ignore_ascii_case(&at))
            {
                return Done::refused(format!("There is no {at} here to survey from."));
            }
            // The room surveyed from, read before the survey's seconds of
            // listening: what is found is that room's, wherever the body is
            // by the time it ends.
            let place = hosted.standpoint_of(body);
            match discover::survey().await {
                Ok(found) => {
                    let lines: Vec<String> = found.iter().map(|f| f.line()).collect();
                    let n = found.len();
                    hosted.with_sim(|s| s.findings.surveyed(&place, found));
                    tracing::info!(world = hosted.id(), %place, devices = n, "house network surveyed");
                    match n {
                        0 => Done::did("The survey found nothing on the house network."),
                        _ => Done::did(format!(
                            "The survey found {n} devices on the house network:\n{}",
                            lines.join("\n")
                        )),
                    }
                }
                Err(e) => Done::refused(format!("The survey could not be made: {e}")),
            }
        }
        "wake_device" => {
            let Some(name) = text(args, "device") else {
                return Done::refused("Say which device to wake.");
            };
            let mac = parts_here(hosted, body).into_iter().find_map(|p| {
                let link = p.link?;
                (p.name.eq_ignore_ascii_case(&name) && !link.hardware.is_empty())
                    .then_some(link.hardware)
            });
            let Some(mac) = mac else {
                return Done::refused(format!(
                    "There is no device called {name} here that the world knows how to wake."
                ));
            };
            submit(
                agents,
                &name,
                BridgeCall::Wake {
                    name: name.clone(),
                    mac,
                },
            )
            .await
        }
        // The device profiles' acts — `sonos_play`, `receiver_volume`, … — each
        // bound to a device standing here. The part names the device's profile
        // and address; the act is handed to an in-network agent to carry out.
        t if profile_act(t).is_some() => {
            let (profile, action) = profile_act(tool).expect("just checked");
            let Some(name) = text(args, "device") else {
                return Done::refused("Say which device.");
            };
            let Some(link) = linked_here(hosted, body, &name, profile) else {
                return Done::refused(format!("There is no {profile} called {name} here."));
            };
            let merged = with_link(args, &link);
            submit(
                agents,
                &name,
                BridgeCall::Work {
                    profile: profile.to_string(),
                    name: name.clone(),
                    address: link.address,
                    action: action.to_string(),
                    args: merged,
                },
            )
            .await
        }
        other => Done::refused(format!("`{other}` is no bridge act.")),
    }
}

/// Hand `call` to an in-network agent and read back the device's line.
///
/// A device that refuses is a refusal, not a landed act; and with no agent to
/// reach — a member off the cluster, or none in the house — the act is refused
/// rather than silently doing nothing.
pub async fn submit(agents: Option<Agents>, name: &str, call: BridgeCall) -> Done {
    let Some(agents) = agents else {
        return Done::refused(
            "No agent in the house can work it right now. Try again in a moment.",
        );
    };
    match agents.bridge(&call).await {
        Ok(b) if b.ok => Done::did(b.line),
        Ok(b) => Done::refused(b.line),
        Err(e) => Done::refused(format!("The {name} could not be reached: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn a_links_credentials_and_mac_fold_into_the_call() {
        let mut creds = BTreeMap::new();
        creds.insert("cert".to_string(), "PEM".to_string());
        creds.insert("key".to_string(), "KEY".to_string());
        let link = PartLink {
            profile: "androidtv".into(),
            address: "192.168.0.214".into(),
            hardware: "14:ae:85:71:eb:44".into(),
            creds,
            notes: None,
        };
        let mut args = Map::new();
        args.insert("video".into(), Value::from("abc"));
        // A credential the caller already set wins over the stored one.
        args.insert("key".into(), Value::from("CALLER"));
        let merged = with_link(&args, &link);
        assert_eq!(merged.get("cert").unwrap(), "PEM");
        assert_eq!(merged.get("key").unwrap(), "CALLER");
        assert_eq!(merged.get("mac").unwrap(), "14:ae:85:71:eb:44");
        assert_eq!(merged.get("video").unwrap(), "abc");
    }
}
