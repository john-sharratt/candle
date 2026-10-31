//! A world's own missions: `<mind>/missions/<world>.yaml`.
//!
//! `missions.yaml` is written for the Makers — its generators read the record
//! and its workflows end with a document on it — and every world falls back to
//! it. A world whose characters are not Makers says what its own work is
//! instead, in a file named for it, and that file **replaces** `missions.yaml`
//! for that world: the record's generators do not run there and its workflows
//! are not loaded there.
//!
//! What a world's file sets is its **routines** — the missions a character
//! collects at the table when nothing has been lodged for it. Each is authored
//! whole: the brief the character is handed and the steps it works through.
//! They take the place of the built-in bank ([`crate::engine::mission::bank`]),
//! whose routines are about records and Makers. A drawn routine ends, like the
//! bank's, with going back to the table to report.
//!
//! The design is `docs/mission_generator.md` §"A world's own missions".

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

use serde::Deserialize;

use crate::engine::mission::{Mission, Origin, Todo};
use crate::mind::files::MindFiles;

/// The folder a world's own missions file stands in.
pub const FOLDER: &str = "missions";

/// One authored routine: a mission a character may be handed whole.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routine {
    /// Its name — what the mission's [`Origin::Random`] carries.
    pub id: String,
    /// What the character is asked, in a paragraph.
    pub brief: String,
    /// The steps it works through, in order.
    pub steps: Vec<String>,
}

/// A world's own missions file.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldMissions {
    pub routines: Vec<Routine>,
}

/// Where world `world`'s own missions file stands in the mind.
pub fn path(world: &str) -> String {
    format!("{FOLDER}/{world}.yaml")
}

impl WorldMissions {
    /// Parse and check a world's file. Every routine has a kebab-case id no
    /// other routine has, a brief, and at least one step.
    pub fn parse(file: &str, yaml: &str) -> Result<WorldMissions, String> {
        let m: WorldMissions = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
        if m.routines.is_empty() {
            return Err(format!("{file}: no routines"));
        }
        let mut seen = BTreeSet::new();
        for r in &m.routines {
            let kebab = !r.id.is_empty()
                && r.id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
            if !kebab {
                return Err(format!(
                    "{file}: the routine id `{}` is not kebab-case",
                    r.id
                ));
            }
            if !seen.insert(r.id.as_str()) {
                return Err(format!("{file}: two routines are called `{}`", r.id));
            }
            if r.brief.trim().is_empty() {
                return Err(format!("{file}: routine `{}` has no brief", r.id));
            }
            if r.steps.is_empty() || r.steps.iter().any(|s| s.trim().is_empty()) {
                return Err(format!(
                    "{file}: routine `{}` needs at least one step, and no empty ones",
                    r.id
                ));
            }
        }
        Ok(m)
    }

    /// Read world `world`'s own file. `Ok(None)` when it has none, and runs on
    /// `missions.yaml` like any other world.
    pub fn load(mind: &MindFiles, world: &str) -> Result<Option<WorldMissions>, String> {
        let file = path(world);
        match mind.read(&file) {
            Some(text) => WorldMissions::parse(&file, &text).map(Some),
            None if mind.is_file(&file) => Err(format!("{file} is not UTF-8")),
            None => Ok(None),
        }
    }
}

/// Draw one of `routines` by `seed`, ending with the report at the table in
/// `table` when the world has one. `None` when there are no routines.
pub fn draw(routines: &[Routine], table: Option<&str>, seed: u64) -> Option<Mission> {
    let r = routines.iter().min_by_key(|r| {
        let mut h = DefaultHasher::new();
        (seed, r.id.as_str()).hash(&mut h);
        h.finish()
    })?;
    let mut todo: Vec<Todo> = r.steps.iter().map(|s| Todo::new(s.as_str())).collect();
    todo.push(Todo::report(match table {
        Some(room) => format!("go back to the table in {room} and report it"),
        None => "go back to the table and report it".to_string(),
    }));
    Some(Mission::new(
        r.brief.as_str(),
        todo,
        Origin::Random {
            routine: r.id.clone(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIFE: &str = "\
routines:
  - id: check-on-the-sharratts
    brief: Look in on the Sharratt family.
    steps:
      - go to the Sharratt room
      - settle how the family is today
  - id: ward-shift
    brief: Take a shift on the ward.
    steps:
      - go to the ward room
";

    #[test]
    fn a_world_file_parses_into_its_routines() {
        let m = WorldMissions::parse("missions/life.yaml", LIFE).unwrap();
        let ids: Vec<&str> = m.routines.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["check-on-the-sharratts", "ward-shift"]);
        assert_eq!(
            m.routines[0].steps,
            ["go to the Sharratt room", "settle how the family is today"]
        );
    }

    /// **A file that would hand out a broken mission is refused when read**,
    /// naming the file and what is wrong.
    #[test]
    fn a_broken_routine_is_refused() {
        for (yaml, why) in [
            ("routines: []\n", "no routines"),
            (
                "routines:\n  - { id: Ward, brief: b, steps: [s] }\n",
                "`Ward` is not kebab-case",
            ),
            (
                "routines:\n  - { id: a, brief: b, steps: [s] }\n  - { id: a, brief: c, steps: [t] }\n",
                "two routines are called `a`",
            ),
            (
                "routines:\n  - { id: a, brief: ' ', steps: [s] }\n",
                "`a` has no brief",
            ),
            (
                "routines:\n  - { id: a, brief: b, steps: [] }\n",
                "`a` needs at least one step",
            ),
            (
                "routines:\n  - { id: a, brief: b, steps: [s], weight: 2 }\n",
                "unknown field `weight`",
            ),
            ("generators: []\n", "unknown field `generators`"),
        ] {
            let e = WorldMissions::parse("missions/w.yaml", yaml).unwrap_err();
            assert!(e.starts_with("missions/w.yaml: "), "{e}");
            assert!(e.contains(why), "{yaml:?}: {e}");
        }
    }

    #[test]
    fn a_world_with_no_file_has_none() {
        let dir = tempfile::tempdir().unwrap();
        let mind = MindFiles::folder(dir.path());
        assert_eq!(WorldMissions::load(&mind, "life"), Ok(None));
        std::fs::create_dir_all(dir.path().join(FOLDER)).unwrap();
        std::fs::write(dir.path().join(path("life")), LIFE).unwrap();
        let m = WorldMissions::load(&mind, "life").unwrap().unwrap();
        assert_eq!(m.routines.len(), 2);
        assert_eq!(WorldMissions::load(&mind, "battle-cities"), Ok(None));
    }

    /// **A drawn routine is the authored mission, closed by the report**, and
    /// the same seed draws the same one.
    #[test]
    fn a_draw_is_the_authored_mission_ending_at_the_table() {
        let m = WorldMissions::parse("missions/life.yaml", LIFE).unwrap();
        let a = draw(&m.routines, Some("the watch room"), 7).unwrap();
        assert_eq!(a, draw(&m.routines, Some("the watch room"), 7).unwrap());
        let r = m
            .routines
            .iter()
            .find(|r| matches!(&a.origin, Origin::Random { routine } if *routine == r.id))
            .unwrap();
        assert_eq!(a.prompt, r.brief);
        let steps: Vec<&str> = a.todo.iter().map(|t| t.text.as_str()).collect();
        let mut want: Vec<&str> = r.steps.iter().map(String::as_str).collect();
        want.push("go back to the table in the watch room and report it");
        assert_eq!(steps, want);
        let drawn: BTreeSet<String> = (0..64)
            .filter_map(|seed| match draw(&m.routines, None, seed)?.origin {
                Origin::Random { routine } => Some(routine),
                _ => None,
            })
            .collect();
        assert_eq!(drawn.len(), 2, "over many seeds every routine is drawn");
        assert_eq!(draw(&[], None, 0), None);
    }
}
