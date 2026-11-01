//! The world building itself (`docs/zen_moon_mind.md` §14): the acts a
//! mission grants its carrier, in its room, in the mirror of a real place —
//! each turned into map edits ([`edit`]), checked, and kept; and what was built
//! laid over the map again whenever it is loaded ([`replay`]).
//!
//! **Granted, never standing.** An act is offered only while the body carries
//! a mission granting it, stands in that mission's room, and the room is a
//! mirror; [`grants_here`] is that test, and the runtime asks it again when
//! the act arrives.

pub mod acts;
pub mod edit;
pub mod replay;
pub mod within;

use std::collections::BTreeSet;

use npc_map::schema::Where;
use npc_map::MapEdit;
use serde_json::{Map, Value};

use crate::bridge::finding::Finding;
use crate::world::Hosted;
pub use acts::{is_mine, BUILD_ACTS};

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// The acts `body` is granted where it stands: its carried mission's grants,
/// when it stands in that mission's room and the room is a mirror — else none.
pub fn grants_here(hosted: &Hosted, body: &str) -> Vec<String> {
    let place = hosted.standpoint_of(body);
    let (grants, room) = hosted.sim(|s| {
        s.missions
            .active(body)
            .map(|m| (m.grants.clone(), m.room().map(str::to_string)))
            .unwrap_or_default()
    });
    let mirrored =
        hosted.read(|w| Where::parse(&place).is_some_and(|at| w.map().is_mirrored(&at.area)));
    match room.as_deref() == Some(place.as_str()) && mirrored {
        true => grants,
        false => Vec::new(),
    }
}

/// The findings of the last survey of room `place` that no part stands for
/// yet — what `onboard_part` and `link_part` may name.
pub fn open_findings(hosted: &Hosted, place: &str) -> Vec<Finding> {
    let found = hosted.sim(|s| s.findings.of(place).to_vec());
    if found.is_empty() {
        return found;
    }
    // The devices placed parts stand for, gathered once rather than per finding.
    let standing: BTreeSet<String> = hosted.read(|w| {
        let map = w.map();
        map.areas()
            .flat_map(|a| a.nodes.iter())
            .flat_map(|n| map.parts_at(n))
            .filter_map(|(p, _)| p.link.as_ref().map(|l| l.hardware.clone()))
            .collect()
    });
    found
        .into_iter()
        .filter(|f| !standing.contains(&f.hardware))
        .collect()
}

/// What a world-building act makes, as the edits to lay and the line its
/// carrier reads — or why it is refused.
pub fn plan(
    hosted: &Hosted,
    body: &str,
    tool: &str,
    args: &Map<String, Value>,
) -> Result<(Vec<MapEdit>, String), String> {
    if !grants_here(hosted, body).iter().any(|g| g == tool) {
        return Err(
            "Nothing you carry lets you change this room: the world is built only on a mission \
             that says so, in the room it is for."
                .into(),
        );
    }
    let place = hosted.standpoint_of(body);
    let room = Where::parse(&place).ok_or("You are not standing anywhere.")?;
    let need = |key: &str| text(args, key).ok_or(format!("Say the `{key}`."));
    // Read before the world is locked below: each takes it too.
    let open = open_findings(hosted, &place);
    let built = hosted.sim(|s| s.built.edits().to_vec());
    let finding = |label: &str| {
        open.iter()
            .find(|f| f.answers_to(label))
            .cloned()
            .ok_or(format!(
                "The survey here found nothing called \"{label}\" that is not in the world already."
            ))
    };
    hosted.read(|w| {
        let map = w.map();
        match tool {
            "onboard_part" => {
                let (from, name, description) =
                    (need("from")?, need("called")?, need("description")?);
                let f = finding(&from)?;
                let edits = edit::onboard_part(map, &room, &f, &name, &description)?;
                // A device back in the room is the part it was.
                let name = map.part(&f.part_id()).map_or(name, |p| p.name.clone());
                Ok((
                    edits,
                    format!("The {name} is in the room now, standing for {}.", f.label()),
                ))
            }
            "onboard_thing" => {
                let (name, description) = (need("called")?, need("description")?);
                let edits = edit::onboard_thing(map, &room, &name, &description)?;
                Ok((edits, format!("The {name} is in the room now.")))
            }
            "link_part" => {
                let (part, from) = (need("part")?, need("from")?);
                let f = finding(&from)?;
                let edits = edit::link_part(map, &room, &part, &f)?;
                Ok((
                    edits,
                    format!("The {part} is {} — the world can reach it now.", f.label()),
                ))
            }
            "offboard_part" => {
                let part = need("part")?;
                need("why")?;
                let edits = edit::offboard_part(map, &room, &part)?;
                Ok((edits, format!("The {part} is out of the room.")))
            }
            "discover_door" => {
                let (name, description) = (need("room")?, need("description")?);
                let edits = edit::discover_door(map, &room, &name, &description)?;
                Ok((edits, format!("A door opens off this room into {name}.")))
            }
            "undiscover_door" => {
                let name = need("room")?;
                need("why")?;
                let edits = edit::undiscover_door(map, &room, &name, &built)?;
                let at = match &edits[0] {
                    MapEdit::RemoveNode(at) => at.clone(),
                    _ => unreachable!("undiscovering is a removal"),
                };
                if !w.actors_at(&at).is_empty() {
                    return Err(format!("Somebody is in {name}: it cannot be taken away."));
                }
                Ok((edits, format!("{name} is gone from the house.")))
            }
            "describe_room" => {
                let description = need("description")?;
                let edits = edit::describe_room(map, &room, &description)?;
                Ok((edits, "The room is described as it is now.".to_string()))
            }
            other => Err(format!("`{other}` is no act that builds the world.")),
        }
    })
}

#[cfg(test)]
mod tests {
    use npc_map::part::Placement;
    use npc_map::schema::Where;
    use npc_map::world::World;
    use npc_map::MapSet;
    use serde_json::json;

    use super::*;
    use crate::engine::mission::{Mission, Origin, Todo};

    /// The vault with the green room made a mirror: a test world the builders
    /// may work in.
    fn mirrored_vault() -> Hosted {
        let map =
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps")).unwrap();
        let (mut areas, parts) = map.authored();
        areas
            .iter_mut()
            .find(|a| a.id == "vault-casting")
            .unwrap()
            .mirrored = true;
        let h = Hosted::of(
            "creators-vault",
            World::new(MapSet::assemble(areas, parts).unwrap()),
        );
        h.with(|w| {
            w.enter("drone", "Wick", Where::new("vault-casting", "green-room"))
                .unwrap();
        });
        h
    }

    fn survey(room: &str) -> Mission {
        Mission::new(
            "Survey the green room.",
            vec![Todo::new("go to the green room")],
            Origin::Standing {
                id: "survey-devices".into(),
                room: room.into(),
            },
        )
        .granting(vec!["onboard_part".into(), "describe_room".into()])
    }

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// **The world is built only as a mission grants, in its room**: with no
    /// mission, nothing; carried, the grants — in the mission's room only.
    #[test]
    fn grants_hold_only_in_the_missions_room() {
        let h = mirrored_vault();
        assert!(grants_here(&h, "drone").is_empty());
        h.with_sim(|s| {
            s.missions
                .assign("drone", survey("vault-casting/green-room"))
        });
        assert_eq!(grants_here(&h, "drone"), ["onboard_part", "describe_room"]);
        h.with_sim(|s| s.missions.assign("drone", survey("vault-casting/band-one")));
        assert!(grants_here(&h, "drone").is_empty(), "not its room");
        assert!(plan(
            &h,
            "drone",
            "describe_room",
            &args(json!({"description": "x"}))
        )
        .is_err());
    }

    /// **A device the survey found is onboarded from its label**, once; and an
    /// act the mission does not grant is refused.
    #[test]
    fn a_found_device_is_onboarded_from_its_label() {
        let h = mirrored_vault();
        h.with_sim(|s| {
            s.missions
                .assign("drone", survey("vault-casting/green-room"));
            s.findings.surveyed(
                "vault-casting/green-room",
                vec![Finding {
                    address: "192.168.0.89".into(),
                    hardware: "f0:f6:c1:a0:79:b4".into(),
                    names: vec!["SonosZP".into()],
                    profile: "sonos".into(),
                    ..Finding::default()
                }],
            );
        });
        assert_eq!(open_findings(&h, "vault-casting/green-room").len(), 1);
        let (edits, line) = plan(
            &h,
            "drone",
            "onboard_part",
            &args(json!({"from": "SonosZP at 192.168.0.89", "called": "Sonos speaker", "description": "The Sonos plays."})),
        )
        .unwrap();
        assert_eq!(
            line,
            "The Sonos speaker is in the room now, standing for SonosZP at 192.168.0.89."
        );
        h.with(|w| {
            for e in &edits {
                w.reshape(e).unwrap();
            }
        });
        let green = h.read(|w| {
            w.node(&Where::new("vault-casting", "green-room"))
                .unwrap()
                .parts
                .clone()
        });
        assert!(green.contains(&Placement::Bare("device-f0f6c1a079b4".into())));
        assert!(
            open_findings(&h, "vault-casting/green-room").is_empty(),
            "it stands in the world now"
        );
        assert!(plan(
            &h,
            "drone",
            "onboard_part",
            &args(json!({"from": "SonosZP at 192.168.0.89", "called": "x", "description": "y"}))
        )
        .is_err());
        assert!(plan(
            &h,
            "drone",
            "onboard_thing",
            &args(json!({"called": "sofa", "description": "y"}))
        )
        .unwrap_err()
        .starts_with("Nothing you carry"));
    }
}
