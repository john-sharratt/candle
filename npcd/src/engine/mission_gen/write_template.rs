//! The built-in `write-template` workflow (`docs/mission_templates.md` §6).
//!
//! A world with its own missions runs no authored workflows — the record's
//! generators and workflows are not loaded there. It runs this one instead, so
//! that `mission_write` has a pipeline to carry an authored mission template
//! onto the record: a Drone writes the template file at a bench (the reused
//! `map-table` kind — a world has no benches of its own otherwise, and a
//! template shapes the world as a map edit does), then publishes it at the
//! `merge-table`, and the world's loader reads it next cycle. The write step's
//! `routine-parses` check (see `work::gate_refusal`) refuses a template that
//! would not parse, so a broken one can never publish and break the loader.
//!
//! The workflow is a built-in rather than authored in the world's missions file
//! because every step, check and desk it names is already the engine's, so a
//! file would add a parser and a second source of truth for nothing.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use serde_json::{Map, Value};

use crate::engine::body::Outcome;
use crate::engine::mission_gen::world::{dir, Routine, Step};
use crate::engine::workflow::{parse_workflows, Workflow};
use crate::world::Hosted;

/// The workflow, as YAML parsed through the ordinary workflow parser. It
/// declares no `prompts:`, so every `{token}` is an operation placeholder the
/// launch fills (`id`, `document`, `yaml`).
const WRITE_TEMPLATE: &str = "\
workflows:
  write-template:
    steps:
      write:
        by: maker
        edits: new
        desk: map-table
        checks: [committed, routine-parses]
        next: publish
        prompt: |
          Write the mission template `{id}` into {document}. Write it exactly as
          given here — a complete YAML file, and nothing else:

          {yaml}
      publish:
        by: maker
        desk: merge-table
        checks: [published]
        next: done
        prompt: |
          Land {document} in the record so the world can hand the template out.
          At the merge table, invoke `merge_status` to see it merges cleanly,
          then `merge_publish` to land it.
";

/// The one workflow an own-missions world runs: authoring a mission template.
/// The world's authority sets it in place of the record's workflows
/// (`engine::mission_gen::run`), and `mission_write` opens an operation on it.
pub fn write_template_workflow() -> Workflow {
    parse_workflows(WRITE_TEMPLATE)
        .expect("the built-in write-template workflow parses")
        .remove(0)
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// A non-empty kebab-case name — a template id or a group folder.
fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// A stable digest of the template, so re-authoring the same one does not reset
/// its stuck count (`sim::missions::launch`).
fn fingerprint(yaml: &str) -> u64 {
    let mut h = DefaultHasher::new();
    yaml.hash(&mut h);
    h.finish()
}

/// `mission_write`: author a new mission template the world can hand out
/// (`docs/mission_templates.md` §6). Zen gives the template's `id`, its `group`
/// (the folder), its `brief`, its `steps` and any `params`; this renders them
/// to a `Routine` YAML and opens a `write-template` operation to write and
/// publish it. The loader reads the file next cycle, and it is then
/// `order_post`-able by name.
pub fn write(hosted: &Hosted, _body: &str, args: &Map<String, Value>) -> Outcome {
    let (Some(id), Some(group), Some(brief)) =
        (text(args, "id"), text(args, "group"), text(args, "brief"))
    else {
        return Outcome::Refused(
            "To write a template, say its `id`, the `group` it goes in, and its `brief`.".into(),
        );
    };
    if !is_name(&id) {
        return Outcome::Refused(format!(
            "The template id `{id}` must be kebab-case: lower-case letters, digits and hyphens."
        ));
    }
    if !is_name(&group) {
        return Outcome::Refused(format!(
            "The group `{group}` must be kebab-case: it is the folder the template goes in."
        ));
    }
    // **A template is written once.** Each write is an operation, and each
    // operation two missions for the Drones: written again while the first was
    // open, or after it had landed, the same template was 146 operations in a
    // night and the Drones' table held two thousand missions.
    let path = format!("{}/{group}/{id}.yaml", dir(hosted.id()));
    let target = format!("template:{path}");
    let (writing, landed) = hosted.sim(|s| {
        (
            s.missions
                .operations()
                .all()
                .any(|o| o.target == target && !o.settled()),
            s.missions.routine(&id).is_some(),
        )
    });
    if landed {
        return Outcome::Refused(format!(
            "There is already a standing job called `{id}`: `order_post` it by name."
        ));
    }
    if writing {
        return Outcome::Refused(format!(
            "The template `{id}` is already being written: you will be able to `order_post` it \
             by name once it has landed."
        ));
    }
    let steps: Vec<Step> = text(args, "steps")
        .map(|s| {
            s.lines()
                .map(|l| l.trim().trim_start_matches(['-', '*']).trim())
                .filter(|l| !l.is_empty())
                .map(Step::from)
                .collect()
        })
        .unwrap_or_default();
    if steps.is_empty() {
        return Outcome::Refused(
            "Give it at least one step, one to a line — each something the carrier can be seen \
             to do."
                .into(),
        );
    }
    let params: Vec<String> = text(args, "params")
        .map(|s| {
            s.split([',', '\n'])
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let routine = Routine {
        id: id.clone(),
        brief,
        steps,
        params,
        grants: Vec::new(),
        work: None,
    };
    let yaml = match serde_yaml::to_string(&routine) {
        Ok(y) => y,
        Err(e) => return Outcome::Refused(format!("The template could not be written out: {e}")),
    };
    // Caught here as well as by the write step's `routine-parses` check, so Zen
    // hears a malformed template at once rather than the Drone's publish holding.
    if let Err(why) = Routine::parse(&path, &yaml) {
        return Outcome::Refused(format!("That is not a valid mission template: {why}"));
    }
    let objective = format!("the mission template `{id}`");
    let brief = format!("Write the mission template `{id}` into its file, exactly as given.");
    let fields = BTreeMap::from([
        ("id".to_string(), id.clone()),
        ("yaml".to_string(), yaml.clone()),
    ]);
    let launched = hosted.with_sim(|s| {
        s.missions.launch(
            "write-template",
            "mission_write",
            &target,
            fingerprint(&yaml),
            &objective,
            &path,
            &brief,
            fields,
            Vec::new(),
        )
    });
    match launched {
        Ok(_) => Outcome::Did(format!(
            "Set to be written: a Drone will draft and publish the template `{id}` into {path}. \
             Once it has landed, `order_post` it by name."
        )),
        Err(e) => Outcome::Refused(format!("The template could not be set to write: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::world::Draw;
    use crate::engine::workflow::By;
    use crate::world::Hosted;
    use npc_map::world::Where;
    use serde_json::json;

    /// **The built-in workflow is write-then-publish**, with the write step at
    /// a `template-desk` holding the template to `committed` + `routine-parses`,
    /// and the publish step at the `merge-table` holding it to `published`.
    #[test]
    fn write_template_is_write_then_publish() {
        let wf = write_template_workflow();
        assert_eq!(wf.name, "write-template");
        let steps: Vec<&str> = wf.steps.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(steps, ["write", "publish"]);

        let write = &wf.steps[0];
        assert_eq!(write.by, By::Maker);
        assert_eq!(write.desk.as_deref(), Some("map-table"));
        assert_eq!(write.checks, ["committed", "routine-parses"]);
        assert!(!write.publishes());

        let publish = &wf.steps[1];
        assert_eq!(publish.by, By::Maker);
        assert_eq!(publish.desk.as_deref(), Some("merge-table"));
        assert_eq!(publish.checks, ["published"]);
        assert!(publish.publishes());
    }

    fn world_with_write_template() -> Hosted {
        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load");
        h.with(|w| {
            w.enter("zen", "Zen", Where::new("vault-command", "command-room"))
                .unwrap();
        });
        h.with_sim(|s| {
            s.missions.set_unordered("zen", true);
            s.missions.set_workflows(vec![write_template_workflow()]);
        });
        h
    }

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// **mission_write renders the template to YAML and opens a write-template
    /// operation for it**, keyed to its path, with the YAML in the step's
    /// fields; a bad id is refused.
    #[test]
    fn write_opens_an_operation_for_the_template() {
        let h = world_with_write_template();
        let out = write(
            &h,
            "zen",
            &args(json!({
                "id": "play-a-film",
                "group": "devices",
                "brief": "Put {title} on {device}.",
                "steps": "go to the front room\nput {title} on {device}",
                "params": "title, device",
            })),
        );
        assert!(out.happened(), "{out:?}");
        let found = h.sim(|s| {
            s.missions.operations().all().any(|o| {
                o.run.workflow == "write-template"
                    && o.document == "missions/creators-vault/devices/play-a-film.yaml"
                    && o.fields
                        .get("yaml")
                        .is_some_and(|y| y.contains("id: play-a-film"))
                    && o.fields.get("id").map(String::as_str) == Some("play-a-film")
            })
        });
        assert!(found, "an operation to write the template should be open");

        // A non-kebab id is refused, and nothing is opened for it.
        let bad = write(
            &h,
            "zen",
            &args(json!({"id": "Bad Id", "group": "devices", "brief": "b", "steps": "go"})),
        );
        assert!(!bad.happened(), "{bad:?}");
    }

    /// **A template is written once.** While a write of it is open, writing it
    /// again opens nothing, and one that has landed is posted by name instead:
    /// Zen wrote `read-dispatch-board` 146 times in a night, and every write was
    /// another two missions for the Drones.
    #[test]
    fn a_template_is_written_once() {
        let h = world_with_write_template();
        let film = || {
            args(json!({
                "id": "play-a-film",
                "group": "devices",
                "brief": "Put a film on.",
                "steps": "go to the front room",
            }))
        };
        assert!(write(&h, "zen", &film()).happened());
        assert_eq!(
            write(&h, "zen", &film()),
            Outcome::Refused(
                "The template `play-a-film` is already being written: you will be able to \
                 `order_post` it by name once it has landed."
                    .into()
            )
        );
        let open = h.sim(|s| {
            s.missions
                .operations()
                .all()
                .filter(|o| o.run.workflow == "write-template" && !o.settled())
                .count()
        });
        assert_eq!(open, 1, "the second write opened nothing");

        h.with_sim(|s| {
            s.missions.set_routines(
                vec![Routine {
                    id: "ward-shift".into(),
                    brief: "Take a shift on the ward.".into(),
                    steps: vec!["go to the ward room".into()],
                    ..Default::default()
                }],
                Draw::None,
            )
        });
        assert_eq!(
            write(
                &h,
                "zen",
                &args(json!({"id": "ward-shift", "group": "ops", "brief": "b", "steps": "go"})),
            ),
            Outcome::Refused(
                "There is already a standing job called `ward-shift`: `order_post` it by name."
                    .into()
            )
        );
    }
}
