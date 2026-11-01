//! The world's bridges to real things (`docs/zen_moon_mind.md` §13, §15):
//! the survey of the house network from a room's house link, and one profile
//! per kind of device the world can work.
//!
//! The acts are [`acts`]'s, each bound to what stands in the room; they reach
//! the network, so the runtime awaits them ([`enact`]), off the world's lock.

pub mod acts;
pub mod discover;
pub mod finding;
pub mod sonos;

use npc_map::part::PartLink;
use npc_map::schema::Where;
use serde_json::{Map, Value};

use crate::engine::mission_gen::standing::HOUSE_LINK;
use crate::world::Hosted;
use finding::SONOS;
use sonos::Player;

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

/// The names of the parts among `parts` that stand for a Sonos player.
pub fn sonos_of(parts: &[Here]) -> Vec<String> {
    parts
        .iter()
        .filter(|p| p.link.as_ref().is_some_and(|l| l.profile == SONOS))
        .map(|p| p.name.clone())
        .collect()
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
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

/// Enact bridge act `tool` for `body` in `hosted`, with `args`: survey from
/// the house link here, or work the Sonos player here.
pub async fn enact(hosted: &Hosted, body: &str, tool: &str, args: &Map<String, Value>) -> Done {
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
        "sonos_status" | "sonos_play" | "sonos_pause" | "sonos_volume" => {
            let Some(name) = text(args, "device") else {
                return Done::refused("Say which Sonos player.");
            };
            let address = parts_here(hosted, body).into_iter().find_map(|p| {
                let link = p.link?;
                (p.name.eq_ignore_ascii_case(&name) && link.profile == SONOS)
                    .then_some(link.address)
            });
            let Some(address) = address else {
                return Done::refused(format!("There is no Sonos player called {name} here."));
            };
            let player = match Player::at(&address, &discover::nets()) {
                Ok(p) => p,
                Err(e) => return Done::refused(format!("The {name} cannot be reached: {e}")),
            };
            let worked = match tool {
                "sonos_status" => player.status().await.map(|s| s.line(&name)),
                "sonos_play" => player
                    .play()
                    .await
                    .map(|()| format!("The {name} is playing.")),
                "sonos_pause" => player
                    .pause()
                    .await
                    .map(|()| format!("The {name} is paused.")),
                _ => {
                    let Some(v) = args.get("volume").and_then(Value::as_u64) else {
                        return Done::refused("Say the volume, 0 to 100.");
                    };
                    let v = v.min(100) as u8;
                    player
                        .set_volume(v)
                        .await
                        .map(|()| format!("The {name} is at volume {v} of 100."))
                }
            };
            match worked {
                Ok(line) => Done::did(line),
                Err(e) => Done::refused(format!("The {name} did not answer: {e}")),
            }
        }
        other => Done::refused(format!("`{other}` is no bridge act.")),
    }
}
