//! Standing missions (`docs/zen_moon_mind.md` §12): the world's own work,
//! put up by the table on a schedule in every room it applies to, and given
//! out last.
//!
//! Each is a file in the mind, `standing/<world>/<id>.yaml`. The table puts one
//! up in a room when it is due there; **never more than one in a room at a
//! time**, and with `every: never`, once in each room ever.

use serde::Deserialize;
use serde_yaml::Value as YamlValue;

use npc_map::schema::{AreaKind, Where};
use npc_map::MapSet;

use crate::engine::build::BUILD_ACTS;
use crate::engine::mission::{Mission, Origin, Todo};
use crate::engine::mission_gen::world::Step;
use crate::engine::zen_task::TASK_ACTS;
use crate::mind::files::MindFiles;
use crate::sim::missions::Missions;

/// The part a room's house link is — what `where: linked` finds rooms by.
pub const HOUSE_LINK: &str = "house-link";

/// How long a standing mission that failed in a room rests before it is put
/// up there again: an hour, so a house network that is down is not surveyed
/// on every beat of the table.
pub const RETRY_MS: u64 = 3_600_000;

/// Where a standing mission applies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rooms {
    /// Every room of the mirror with a house link.
    Linked,
    /// These rooms, by `area/node`.
    Named(Vec<Where>),
}

/// How often a standing mission is due in a room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Every {
    /// Once in each room, ever.
    Never,
    /// Again this long after it was last done there.
    Period(u64),
}

/// One standing mission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Standing {
    pub id: String,
    pub brief: String,
    pub rooms: Rooms,
    pub every: Every,
    pub grants: Vec<String>,
    /// Its steps, each plain or flagged `critical` — a step that waits on an
    /// authorizer (`docs/mission_templates.md` §6), as a template's does.
    pub steps: Vec<Step>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    id: String,
    brief: String,
    #[serde(rename = "where")]
    rooms: YamlValue,
    every: String,
    #[serde(default)]
    grants: Vec<String>,
    steps: Vec<Step>,
}

/// Whether `act` is one a standing mission may grant: a building act, or a
/// task act — a maintenance mission's carrier drives a task of Zen's
/// (`crate::engine::zen_task`).
fn grantable(act: &str) -> bool {
    BUILD_ACTS.iter().chain(TASK_ACTS).any(|t| t.name == act)
}

/// A period as written — `30m`, `6h`, `2d` — in milliseconds.
pub fn period(text: &str) -> Option<u64> {
    let text = text.trim();
    let (n, unit) = text.split_at(text.len().checked_sub(1)?);
    let n: u64 = n.parse().ok().filter(|n| *n > 0)?;
    let ms = match unit {
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => return None,
    };
    Some(n * ms)
}

impl Standing {
    /// Parse and check one standing mission's file.
    pub fn parse(file: &str, yaml: &str) -> Result<Standing, String> {
        let f: File = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
        let kebab = !f.id.is_empty()
            && f.id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !kebab {
            return Err(format!("{file}: the id `{}` is not kebab-case", f.id));
        }
        if f.brief.trim().is_empty()
            || f.steps.iter().any(|s| s.text().trim().is_empty())
            || f.steps.is_empty()
        {
            return Err(format!(
                "{file}: a standing mission needs a brief and at least one step"
            ));
        }
        let rooms = match &f.rooms {
            YamlValue::String(s) if s == "linked" => Rooms::Linked,
            YamlValue::Sequence(list) => Rooms::Named(
                list.iter()
                    .map(|v| v.as_str().and_then(Where::parse))
                    .collect::<Option<Vec<Where>>>()
                    .ok_or(format!("{file}: each room in `where` is `area/node`"))?,
            ),
            _ => {
                return Err(format!(
                    "{file}: `where` is `linked` or a list of `area/node`"
                ))
            }
        };
        let every = match f.every.trim() {
            "never" => Every::Never,
            p => Every::Period(period(p).ok_or(format!(
                "{file}: `every` is `never` or a period — `30m`, `6h`, `2d`"
            ))?),
        };
        if let Some(g) = f.grants.iter().find(|g| !grantable(g)) {
            return Err(format!("{file}: `{g}` is no act a mission can grant"));
        }
        Ok(Standing {
            id: f.id,
            brief: f.brief.trim().to_string(),
            rooms,
            every,
            grants: f.grants,
            steps: f.steps,
        })
    }

    /// The rooms this applies to in `map`.
    pub fn rooms_in(&self, map: &MapSet) -> Vec<Where> {
        match &self.rooms {
            Rooms::Named(list) => list
                .iter()
                .filter(|at| map.get(&at.area).and_then(|a| a.node(&at.node)).is_some())
                .cloned()
                .collect(),
            // A house link outside the mirror reaches nothing (§13), so
            // there is nothing for a mission there to do.
            Rooms::Linked => map
                .areas()
                .filter(|a| map.is_mirrored(&a.id))
                .flat_map(|a| {
                    a.nodes
                        .iter()
                        .filter(|n| n.parts.iter().any(|p| p.part() == HOUSE_LINK))
                        .map(move |n| Where::new(a.id.clone(), n.id.clone()))
                })
                .collect(),
        }
    }

    /// The mission for `room`, named `room_name`, reported at the table in
    /// `table`.
    pub fn mission(&self, room: &Where, room_name: &str, table: Option<&str>) -> Mission {
        let fill = |s: &str| s.replace("{room}", room_name);
        let mut todo: Vec<Todo> = self
            .steps
            .iter()
            .map(|s| match s.is_critical() {
                true => Todo::critical(fill(s.text())),
                false => Todo::new(fill(s.text())),
            })
            .collect();
        todo.push(Todo::report(match table {
            Some(t) => format!("go back to the table in {t} and report it"),
            None => "go back to the table and report it".to_string(),
        }));
        Mission::new(
            fill(&self.brief),
            todo,
            Origin::Standing {
                id: self.id.clone(),
                room: room.to_string(),
            },
        )
        .granting(self.grants.clone())
    }
}

/// Where world `world`'s standing missions stand in the mind: a folder of
/// their own, apart from the world's template library (`missions/<world>/`).
pub fn dir(world: &str) -> String {
    format!("standing/{world}")
}

/// Every standing mission of world `world`, from `standing/<world>/`.
pub fn load(mind: &MindFiles, world: &str) -> Result<Vec<Standing>, String> {
    let dir = dir(world);
    let mut out = Vec::new();
    for name in mind.list(&dir) {
        if !name.ends_with(".yaml") {
            continue;
        }
        let file = format!("{dir}/{name}");
        let Some(text) = mind.read(&file) else {
            return Err(format!("{file} is not UTF-8"));
        };
        out.push(Standing::parse(&file, &text)?);
    }
    Ok(out)
}

/// The building `area` is in — itself, when it is one — or its outermost
/// area.
fn building_of<'a>(map: &'a MapSet, area: &'a str) -> &'a str {
    let mut at = area;
    while let Some(a) = map.get(at) {
        if a.kind == AreaKind::Building {
            return &a.id;
        }
        match a.within.as_deref() {
            Some(up) if map.get(up).is_some_and(|u| u.kind != AreaKind::World) => at = up,
            _ => return &a.id,
        }
    }
    at
}

/// The name of the room the order table stands in that a mission for `room`
/// is reported at: the table in `room`'s own building, else the first there
/// is — a mirrored home has none of its own, and is reported to the moon's.
pub fn table_room(map: &MapSet, room: &Where) -> Option<String> {
    let tables: Vec<(&str, String)> = map
        .areas()
        .flat_map(|a| a.nodes.iter().map(move |n| (a.id.as_str(), n)))
        .filter(|(_, n)| n.parts.iter().any(|p| p.part() == "order-table"))
        .map(|(area, n)| (area, n.name.clone()))
        .collect();
    let home = building_of(map, &room.area);
    tables
        .iter()
        .find(|(area, _)| building_of(map, area) == home)
        .or_else(|| tables.first())
        .map(|(_, name)| name.clone())
}

/// The standing missions due in `map` at `now_ms`, given what the table holds:
/// one for each room each applies to, unless one is up or carried there, or it
/// is not due there yet.
pub fn due(standing: &[Standing], map: &MapSet, missions: &Missions, now_ms: u64) -> Vec<Mission> {
    let mut out = Vec::new();
    for s in standing {
        for room in s.rooms_in(map) {
            let table = table_room(map, &room);
            let key = room.to_string();
            if missions.standing_out(&s.id, &key) {
                continue;
            }
            // Done but not yet stamped is done just now.
            let ready = match (s.every, missions.standing_done(&s.id, &key)) {
                (_, None) => true,
                (Every::Never, Some(_)) | (_, Some(None)) => false,
                (Every::Period(p), Some(Some(at))) => now_ms >= at.saturating_add(p),
            };
            // A failure is tried again, but not before [`RETRY_MS`] has passed.
            let resting = match missions.standing_failed(&s.id, &key) {
                None => false,
                Some(None) => true,
                Some(Some(at)) => now_ms < at.saturating_add(RETRY_MS),
            };
            if !ready || resting {
                continue;
            }
            let name = map
                .get(&room.area)
                .and_then(|a| a.node(&room.node))
                .map(|n| n.name.clone())
                .unwrap_or_else(|| key.clone());
            out.push(s.mission(&room, &name, table.as_deref()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use npc_map::part::{Part, PartKind, Placement};

    use super::*;
    use crate::engine::mission::bank::Facts;
    use crate::engine::mission::Outcome;

    const SURVEY: &str = "id: survey-devices\nbrief: Survey {room}.\nwhere: linked\nevery: never\ngrants: [onboard_part, link_part]\nsteps:\n  - go to {room}\n  - survey the network at the house link\n";

    /// The vault with a house link in its green room — a mirror of a real
    /// place, or not.
    fn vault_linked(mirrored: bool) -> MapSet {
        let map =
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps")).unwrap();
        let (mut areas, mut parts) = map.authored();
        parts.push(Part {
            id: HOUSE_LINK.into(),
            kind: PartKind::Fixture,
            name: "house link".into(),
            plural: None,
            binds: None,
            short: None,
            long: "x".into(),
            modes: vec![],
            link: None,
        });
        let casting = areas.iter_mut().find(|a| a.id == "vault-casting").unwrap();
        casting.mirrored = mirrored;
        let green = casting
            .nodes
            .iter_mut()
            .find(|n| n.id == "green-room")
            .unwrap();
        green.parts.push(Placement::Bare(HOUSE_LINK.into()));
        MapSet::assemble(areas, parts).unwrap()
    }

    #[test]
    fn a_period_is_minutes_hours_or_days() {
        assert_eq!(period("30m"), Some(1_800_000));
        assert_eq!(period("6h"), Some(21_600_000));
        assert_eq!(period("2d"), Some(172_800_000));
        assert_eq!(period("0h"), None);
        assert_eq!(period("6w"), None);
        assert_eq!(period(""), None);
    }

    /// **A standing mission's file reads as its brief, rooms, period, grants
    /// and steps**; one that grants an act no mission can, names a room badly,
    /// or has no step is refused.
    #[test]
    fn a_standing_mission_reads_and_is_checked() {
        let s = Standing::parse("missions/life/survey-devices.yaml", SURVEY).unwrap();
        assert_eq!(s.rooms, Rooms::Linked);
        assert_eq!(s.every, Every::Never);
        assert_eq!(s.grants, ["onboard_part", "link_part"]);
        let named = SURVEY.replace("where: linked", "where: [sharratt-apartment/front-room]");
        assert_eq!(
            Standing::parse("f", &named).unwrap().rooms,
            Rooms::Named(vec![Where::new("sharratt-apartment", "front-room")])
        );
        for (bad, why) in [
            (
                SURVEY.replace("link_part", "reshape"),
                "`reshape` is no act",
            ),
            (
                SURVEY.replace("where: linked", "where: [nowhere]"),
                "`area/node`",
            ),
            (SURVEY.replace("every: never", "every: often"), "`every` is"),
            (
                SURVEY.replace("id: survey-devices", "id: Survey"),
                "kebab-case",
            ),
        ] {
            let e = Standing::parse("f", &bad).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
    }

    /// **A maintenance mission grants the task acts and may flag a step
    /// critical**: its mission carries the grants and a critical todo, and an
    /// act no mission can grant is still refused.
    #[test]
    fn a_standing_mission_drives_a_task_and_flags_a_critical_step() {
        let yaml = "id: journal-maintenance\nbrief: Keep the journals true.\n\
                    where: [vault-casting/green-room]\nevery: 1d\n\
                    grants: [task_start, task_reply, task_nudge]\nsteps:\n\
                    \x20 - start the journal maintenance task\n  - see the task through\n\
                    \x20 - text: land the held branch\n    critical: true\n";
        let s = Standing::parse("standing/moon/journal-maintenance.yaml", yaml).unwrap();
        assert_eq!(s.every, Every::Period(86_400_000));
        let m = s.mission(
            &Where::new("vault-casting", "green-room"),
            "the green room",
            Some("the command room"),
        );
        assert_eq!(m.grants, ["task_start", "task_reply", "task_nudge"]);
        let steps: Vec<(&str, bool)> = m
            .todo
            .iter()
            .map(|t| (t.text.as_str(), t.critical))
            .collect();
        assert_eq!(
            steps,
            [
                ("start the journal maintenance task", false),
                ("see the task through", false),
                ("land the held branch", true),
                (
                    "go back to the table in the command room and report it",
                    false
                ),
            ]
        );
        let e = Standing::parse("f", &yaml.replace("task_nudge", "task_delete")).unwrap_err();
        assert!(e.contains("`task_delete` is no act"), "{e}");
    }

    /// **One per room, never two at once, and with `every: never` once ever**:
    /// due in each linked room of the mirror until put up — never in one
    /// outside it; not again while it is out; not again once done.
    #[test]
    fn a_standing_mission_is_due_once_per_room() {
        let map = vault_linked(true);
        let standing = [Standing::parse("f", SURVEY).unwrap()];
        let mut missions = Missions::default();
        assert!(
            due(&standing, &vault_linked(false), &missions, 0).is_empty(),
            "a house link outside the mirror reaches nothing"
        );
        let due_now = due(&standing, &map, &missions, 0);
        assert_eq!(due_now.len(), 1);
        let m = &due_now[0];
        assert_eq!(m.prompt, "Survey the green room.");
        assert_eq!(m.room(), Some("vault-casting/green-room"));
        assert_eq!(m.grants, ["onboard_part", "link_part"]);
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the green room",
                "survey the network at the house link",
                "go back to the table in the command room and report it"
            ]
        );

        missions.put_up_standing(m.clone());
        assert!(
            due(&standing, &map, &missions, 0).is_empty(),
            "one is up there"
        );
        missions.collect("drone", &Facts::default()).unwrap();
        assert!(
            due(&standing, &map, &missions, 0).is_empty(),
            "one is carried there"
        );
        missions.report("drone", Outcome::Pass, "found eleven devices", None);
        assert!(
            due(&standing, &map, &missions, 0).is_empty(),
            "done, not yet stamped"
        );
        missions.stamp_standing(0);
        assert!(
            due(&standing, &map, &missions, 99_999_999).is_empty(),
            "done, ever"
        );

        let again = [Standing {
            every: Every::Period(1_000),
            ..standing[0].clone()
        }];
        assert!(due(&again, &map, &missions, 0).is_empty(), "not yet");
        assert_eq!(
            due(&again, &map, &missions, u64::MAX / 2).len(),
            1,
            "due again"
        );
    }

    /// **A standing mission that failed rests before it is put up again**:
    /// not on the next beat, but once [`RETRY_MS`] has passed; done after
    /// that, the failure is forgotten.
    #[test]
    fn a_failed_standing_mission_rests_before_it_is_tried_again() {
        let map = vault_linked(true);
        let standing = [Standing::parse("f", SURVEY).unwrap()];
        let mut missions = Missions::default();
        missions.put_up_standing(due(&standing, &map, &missions, 0).remove(0));
        missions.collect("drone", &Facts::default()).unwrap();
        missions.report("drone", Outcome::Fail, "the network did not answer", None);
        assert!(due(&standing, &map, &missions, 0).is_empty(), "just failed");
        missions.stamp_standing(1_000);
        assert!(
            due(&standing, &map, &missions, 1_000 + RETRY_MS - 1).is_empty(),
            "resting"
        );
        let again = due(&standing, &map, &missions, 1_000 + RETRY_MS);
        assert_eq!(again.len(), 1, "tried again");
        missions.put_up_standing(again[0].clone());
        missions.collect("drone", &Facts::default()).unwrap();
        missions.report("drone", Outcome::Pass, "found eleven devices", None);
        assert_eq!(
            missions.standing_failed("survey-devices", "vault-casting/green-room"),
            None
        );
    }
}
