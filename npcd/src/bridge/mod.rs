//! The world's bridges to real things (`docs/zen_moon_mind.md` §13, §15):
//! the survey of the house network from a room's house link, and one profile
//! per kind of device the world can work.
//!
//! The acts are [`acts`]'s, each bound to what stands in the room; they reach
//! the network, so the runtime awaits them ([`enact`]), off the world's lock.

pub mod acts;
pub mod discover;
pub mod finding;

use std::collections::BTreeMap;

use agent::client::Agents;
use agent::wire::BridgeCall;
use effector_bridge::finding::pairs;
use effector_bridge::work::profile_act;
use npc_map::part::PartLink;
use npc_map::schema::Where;
use serde_json::{Map, Value};

use crate::engine::mission_gen::standing::HOUSE_LINK;
use crate::world::Hosted;
use finding::{ANDROIDTV, CAST, RECEIVER, SONOS};

/// A reserved credential key, holding the id of the agent that began a Google
/// TV's two-phase pairing, so `pair_code` finishes on that same agent (the TLS
/// session lives only there). Reserved keys begin with `_`; [`with_link`] never
/// hands one to a device.
const PAIRING_AGENT: &str = "_pairing_agent";

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

/// The names of the parts among `parts` that stand for a Cast device.
pub fn cast_of(parts: &[Here]) -> Vec<String> {
    on_profile(parts, CAST)
}

/// Whether `link` holds a credential the world actually works the device with —
/// any key that is not reserved bookkeeping. A device mid-pairing holds only a
/// reserved key (see [`PAIRING_AGENT`]) and so still counts as having none.
fn has_key(link: &PartLink) -> bool {
    link.creds.keys().any(|k| !k.starts_with('_'))
}

/// Whether a Google TV's two-phase pairing has begun: the reserved note naming
/// the agent that showed the code is recorded (see [`PAIRING_AGENT`]), set by
/// the first call and dropped when the second finishes.
fn pairing_begun(link: &PartLink) -> bool {
    link.creds.contains_key(PAIRING_AGENT)
}

/// The names of the parts among `parts` that stand for a TV the world pairs
/// with (a webOS or Google TV), has no key for yet, and has not already begun
/// pairing — what `pair_device` begins. A TV already paired is absent (there is
/// nothing to begin), and so is a Google TV mid-pairing: re-beginning it would
/// orphan the open session and show a fresh code, so once begun only
/// `pair_code` (see [`pairing_of`]) is offered until it finishes.
pub fn pairable_of(parts: &[Here]) -> Vec<String> {
    parts
        .iter()
        .filter(|p| {
            p.link
                .as_ref()
                .is_some_and(|l| pairs(&l.profile) && !has_key(l) && !pairing_begun(l))
        })
        .map(|p| p.name.clone())
        .collect()
}

/// The names of the parts among `parts` that stand for a Google TV whose
/// pairing has begun but is not yet finished — what `pair_code` completes with
/// the code the TV is showing. A webOS TV pairs in one call and never takes a
/// code, and a Google TV that has not begun has no session to finish, so
/// neither appears here: `pair_code` follows `pair_device`, never precedes it.
pub fn pairing_of(parts: &[Here]) -> Vec<String> {
    parts
        .iter()
        .filter(|p| {
            p.link
                .as_ref()
                .is_some_and(|l| l.profile == ANDROIDTV && pairing_begun(l) && !has_key(l))
        })
        .map(|p| p.name.clone())
        .collect()
}

/// The part here named `name` that stands for a device of `profile`, and its
/// address.
fn linked_here(hosted: &Hosted, body: &str, name: &str, profile: &str) -> Option<PartLink> {
    parts_here(hosted, body).into_iter().find_map(|p| {
        let link = p.link?;
        (p.name.eq_ignore_ascii_case(name) && link.profile == profile).then_some(link)
    })
}

/// The part here named `name` that stands for a device the world pairs with (a
/// webOS or Google TV), with its id and link.
fn pairable_here(hosted: &Hosted, body: &str, name: &str) -> Option<(String, PartLink)> {
    parts_here(hosted, body).into_iter().find_map(|p| {
        let link = p.link?;
        (p.name.eq_ignore_ascii_case(name) && pairs(&link.profile)).then_some((p.id, link))
    })
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// The act's arguments with the device's stored credentials and hardware
/// address folded in — so the agent is handed what it needs to work the device
/// (a webOS key, an Android TV certificate, the wake MAC) and the world keeps
/// the secrets, not the stateless agent. An argument the caller set wins. A
/// reserved key (one beginning `_`, like [`PAIRING_AGENT`]) is the world's own
/// bookkeeping and is never handed to a device.
fn with_link(args: &Map<String, Value>, link: &PartLink) -> Map<String, Value> {
    let mut merged = args.clone();
    for (name, value) in &link.creds {
        if name.starts_with('_') {
            continue;
        }
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

/// What came of a bridge act: whether it landed, the line its body reads, and
/// any credentials it earned for the world to keep on the part.
pub struct Done {
    pub landed: bool,
    pub answer: String,
    /// What a pairing earned, for the runtime to write onto the part here —
    /// `None` for every act that earns nothing. See [`Earned`].
    pub earned: Option<Earned>,
}

/// Credentials a pairing earned, to keep on the device's part: `set` is merged
/// into the part's stored credentials and `drop` names keys to remove (the
/// transient [`PAIRING_AGENT`] once a Google TV is through). The runtime writes
/// these, since it holds the map store the enact path does not.
pub struct Earned {
    pub part: String,
    pub set: BTreeMap<String, String>,
    pub drop: Vec<String>,
}

impl Done {
    fn did(answer: impl Into<String>) -> Done {
        Done {
            landed: true,
            answer: answer.into(),
            earned: None,
        }
    }
    fn refused(answer: impl Into<String>) -> Done {
        Done {
            landed: false,
            answer: answer.into(),
            earned: None,
        }
    }
    /// A pairing step that landed and left credentials for the world to keep.
    fn earned(answer: impl Into<String>, earned: Earned) -> Done {
        Done {
            landed: true,
            answer: answer.into(),
            earned: Some(earned),
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
        // Pairing a TV that has never been paired — the world earns the key.
        // `pair_device` begins it (a webOS TV in one go once the family accept
        // on screen; a Google TV shows a code); `pair_code` finishes a Google
        // TV with the code the family read back.
        "pair_device" => {
            let Some(name) = text(args, "device") else {
                return Done::refused("Say which TV to pair.");
            };
            let Some((part, link)) = pairable_here(hosted, body, &name) else {
                return Done::refused(format!(
                    "There is no TV called {name} here that the world pairs with."
                ));
            };
            pair(agents, &name, part, link, None).await
        }
        "pair_code" => {
            let Some(name) = text(args, "device") else {
                return Done::refused("Say which TV the code is for.");
            };
            let Some(code) = text(args, "code") else {
                return Done::refused("Say the six-digit code the TV is showing.");
            };
            let Some((part, link)) = pairable_here(hosted, body, &name) else {
                return Done::refused(format!(
                    "There is no TV called {name} here that the world pairs with."
                ));
            };
            // Only a Google TV takes a code; an LG TV pairs in one step with
            // `pair_device`, so a code for one is a mistake, not a fresh pairing.
            if link.profile != ANDROIDTV {
                return Done::refused(format!(
                    "The {name} pairs in one step with `pair_device` — it needs no code."
                ));
            }
            pair(agents, &name, part, link, Some(code)).await
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

/// Pair with the TV `name` here, earning the credential the world keeps on its
/// part. webOS pairs in one call (the family accept on the TV's own screen); a
/// Google TV takes two — `pair_device` (no `code`) begins it and records which
/// agent holds the session, and `pair_code` (with the `code`) finishes on that
/// same agent, since the pairing session lives only there.
async fn pair(
    agents: Option<Agents>,
    name: &str,
    part: String,
    link: PartLink,
    code: Option<String>,
) -> Done {
    let Some(agents) = agents else {
        return Done::refused(
            "No agent in the house can pair it right now. Try again in a moment.",
        );
    };
    // The finishing call pins to the agent the beginning call recorded.
    let to = code
        .as_ref()
        .and(link.creds.get(PAIRING_AGENT))
        .and_then(|a| a.parse::<u64>().ok());
    let call = BridgeCall::Pair {
        profile: link.profile.clone(),
        address: link.address.clone(),
        code: code.clone(),
    };
    let (bridged, served_by) = match agents.bridge_on(&call, to).await {
        Ok(pair) => pair,
        Err(e) => return Done::refused(format!("The {name} could not be reached: {e}")),
    };
    // The response decides, not the request: a device that could not be
    // reached or refused the pairing is `ok: false`, whichever step it was.
    if !bridged.ok {
        return Done::refused(bridged.line);
    }
    // A Google TV's beginning succeeds with no credentials yet — it has only
    // shown its code. Keep which agent holds the session (so `pair_code`
    // finishes there) and ask the family for the code.
    if bridged.creds.is_empty() {
        let set = served_by
            .map(|a| BTreeMap::from([(PAIRING_AGENT.to_string(), a.to_string())]))
            .unwrap_or_default();
        return Done::earned(
            "The TV is showing a six-digit code. Ask the family to read it back, then finish \
             with `pair_code`.",
            Earned {
                part,
                set,
                drop: Vec::new(),
            },
        );
    }
    // webOS in one call, or a Google TV's finish: the key is earned. Keep it,
    // and clear the transient note of the pairing agent.
    Done::earned(
        bridged.line,
        Earned {
            part,
            set: bridged.creds,
            drop: vec![PAIRING_AGENT.to_string()],
        },
    )
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
        // A reserved key — the world's own bookkeeping — must never reach a device.
        creds.insert(PAIRING_AGENT.to_string(), "7".to_string());
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
        assert!(
            merged.get(PAIRING_AGENT).is_none(),
            "a reserved `_`-key is the world's own, never handed to a device"
        );
    }

    /// A `Here` standing for a device of `profile` with the given credentials.
    fn device_part(name: &str, profile: &str, creds: &[(&str, &str)]) -> Here {
        Here {
            name: name.into(),
            id: format!("device-{name}"),
            link: Some(PartLink {
                profile: profile.into(),
                address: "192.168.0.1".into(),
                hardware: String::new(),
                creds: creds
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
                notes: None,
            }),
        }
    }

    /// **Each device act's set is the devices of its kind standing here, and a
    /// TV moves cleanly from `pair_device` to `pair_code` as its pairing
    /// begins.** These are the sets the grammar binds a device act's `device`
    /// to; an empty one takes the act out of the grammar, so what a room offers
    /// is exactly what stands in it — and a Google TV mid-pairing is offered
    /// only `pair_code`, never a second `pair_device` that would orphan its
    /// session.
    #[test]
    fn the_device_sets_name_the_devices_of_each_kind_here() {
        let parts = vec![
            device_part("the Sonos", SONOS, &[]),
            device_part("the stereo", RECEIVER, &[]),
            device_part("the Chromecast", CAST, &[]),
            // An unpaired LG TV: pairable (one call), never pairing (no code).
            device_part("the LG TV", finding::WEBOS, &[]),
            // A Google TV not yet begun: pairable, not yet pairing.
            device_part("the den TV", ANDROIDTV, &[]),
            // A Google TV mid-pairing holds the reserved agent note and no real
            // key: no longer pairable (re-beginning would orphan it), now
            // pairing.
            device_part("the Google TV", ANDROIDTV, &[(PAIRING_AGENT, "7")]),
            // A webOS TV already paired has a key, so no set names it.
            device_part("the bedroom TV", finding::WEBOS, &[("client-key", "abcd")]),
        ];
        assert_eq!(sonos_of(&parts), vec!["the Sonos".to_string()]);
        assert_eq!(receiver_of(&parts), vec!["the stereo".to_string()]);
        assert_eq!(cast_of(&parts), vec!["the Chromecast".to_string()]);
        assert_eq!(
            pairable_of(&parts),
            vec!["the LG TV".to_string(), "the den TV".to_string()]
        );
        assert_eq!(pairing_of(&parts), vec!["the Google TV".to_string()]);
    }
}
