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

use crate::engine::workflow::{parse_workflows, Workflow};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::workflow::By;

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
}
