//! The rules parsed workflows and generators must satisfy before an operation
//! may run on them.

use std::collections::BTreeSet;

use super::config::{Generator, Variants, Workflow, CANCELLED, DONE, FAILED};

/// Check `workflow` whole: at least one step, no step named for an end state,
/// no empty prompt, and every target — through `next` or `stuck` — a step that
/// exists, `done` or `failed`.
pub fn validate(workflow: &Workflow) -> Result<(), String> {
    let ctx = format!("workflow `{}`", workflow.name);
    if workflow.steps.is_empty() {
        return Err(format!("{ctx}: has no steps"));
    }
    for step in &workflow.steps {
        let ctx = format!("{ctx}, step `{}`", step.name);
        if [DONE, FAILED, CANCELLED].contains(&step.name.as_str()) {
            return Err(format!(
                "{ctx}: `{DONE}`, `{FAILED}` and `{CANCELLED}` end an operation and cannot be step names"
            ));
        }
        let empty = match &step.prompt {
            Variants::One(text) => text.trim().is_empty(),
            Variants::ByOutcome(variants) => variants.iter().any(|(_, t)| t.trim().is_empty()),
        };
        if empty {
            return Err(format!("{ctx}: has an empty prompt"));
        }
        for target in step.targets() {
            if target == CANCELLED {
                return Err(format!(
                    "{ctx}: leads to `{CANCELLED}`, which only the engine sets"
                ));
            }
            if target != DONE && target != FAILED && workflow.step(target).is_none() {
                return Err(format!(
                    "{ctx}: leads to `{target}`, which is not a step, `{DONE}` or `{FAILED}`"
                ));
            }
        }
    }
    Ok(())
}

/// Check that generator ids are distinct and each opens a workflow that
/// exists.
pub fn validate_generators(generators: &[Generator], workflows: &[Workflow]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for g in generators {
        if !seen.insert(g.id.as_str()) {
            return Err(format!("generator `{}`: the id is used twice", g.id));
        }
        if !workflows.iter().any(|w| w.name == g.workflow) {
            return Err(format!(
                "generator `{}`: opens workflow `{}`, which is not declared",
                g.id, g.workflow
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse_missions;

    fn err(yaml: &str) -> String {
        parse_missions(yaml).unwrap_err()
    }

    fn steps(steps: &str) -> String {
        format!("workflows:\n  w:\n    steps:\n{steps}")
    }

    #[test]
    fn no_steps_is_refused() {
        assert_eq!(
            err("workflows:\n  w:\n    steps: {}\n"),
            "workflow `w`: has no steps"
        );
    }

    #[test]
    fn a_step_named_for_an_end_state_is_refused() {
        for name in ["done", "failed", "cancelled"] {
            assert_eq!(
                err(&steps(&format!("      {name}:\n        by: maker\n        next: done\n        prompt: p\n"))),
                format!("workflow `w`, step `{name}`: `done`, `failed` and `cancelled` end an operation and cannot be step names")
            );
        }
    }

    #[test]
    fn an_unknown_target_is_refused_naming_it() {
        assert_eq!(
            err(&steps(
                "      a:\n        by: maker\n        next: b\n        prompt: p\n"
            )),
            "workflow `w`, step `a`: leads to `b`, which is not a step, `done` or `failed`"
        );
        assert_eq!(
            err(&steps("      a:\n        by: maker\n        next:\n          pass: done\n          fail: rewrite\n        prompt: p\n")),
            "workflow `w`, step `a`: leads to `rewrite`, which is not a step, `done` or `failed`"
        );
        assert_eq!(
            err(&steps("      a:\n        by: maker\n        next: done\n        stuck: elsewhere\n        prompt: p\n")),
            "workflow `w`, step `a`: leads to `elsewhere`, which is not a step, `done` or `failed`"
        );
    }

    #[test]
    fn cancelled_is_not_routable() {
        assert_eq!(
            err(&steps(
                "      a:\n        by: maker\n        next: cancelled\n        prompt: p\n"
            )),
            "workflow `w`, step `a`: leads to `cancelled`, which only the engine sets"
        );
    }

    #[test]
    fn an_empty_prompt_or_variant_is_refused() {
        assert_eq!(
            err(&steps(
                "      a:\n        by: maker\n        next: done\n        prompt: \"  \"\n"
            )),
            "workflow `w`, step `a`: has an empty prompt"
        );
        assert_eq!(
            err(&steps("      a:\n        by: maker\n        next: done\n        prompt:\n          start: go\n          fail: ''\n")),
            "workflow `w`, step `a`: has an empty prompt"
        );
    }

    #[test]
    fn a_later_step_and_both_end_states_are_valid_targets() {
        let m = parse_missions(&steps("      a:\n        by: maker\n        stuck: b\n        next:\n          go: b\n          win: done\n          lose: failed\n        prompt: p\n      b:\n        by: table\n        call: c\n        next: a\n        prompt: q\n")).unwrap();
        assert_eq!(m.workflows[0].steps.len(), 2);
    }

    #[test]
    fn a_generator_must_open_a_declared_workflow_under_its_own_id() {
        let wf = "workflows:\n  w:\n    steps:\n      s:\n        by: maker\n        next: done\n        prompt: p\n";
        assert_eq!(
            err(&format!(
                "generators:\n  - id: g\n    call: c\n    workflow: nowhere\n    prompt: p\n{wf}"
            )),
            "generator `g`: opens workflow `nowhere`, which is not declared"
        );
        assert_eq!(
            err(&format!("generators:\n  - id: g\n    call: c\n    workflow: w\n    prompt: p\n  - id: g\n    call: d\n    workflow: w\n    prompt: q\n{wf}")),
            "generator `g`: the id is used twice"
        );
    }
}
