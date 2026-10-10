//! Reading a mind's `missions.yaml` into [`Missions`]: the `keep`, `prompts`,
//! `generators` and `workflows` sections, every mapping in the order written,
//! validated and with prompt includes resolved. Other top-level keys belong to
//! other readers and are ignored; inside these sections an unknown key is an
//! error.

use std::collections::BTreeMap;

use serde_yaml::{Mapping, Value};

use super::config::{
    By, Edits, Generator, Missions, Next, OnFailed, Step, Variants, Workflow, FAILED, SEND_BACKS,
    STUCK, STUCK_LIMIT, WEIGHT,
};
use super::includes::resolve;
use super::validate::{validate, validate_generators};

/// Everything `missions_yaml` declares for operations.
pub fn parse_missions(missions_yaml: &str) -> Result<Missions, String> {
    let doc: Value =
        serde_yaml::from_str(missions_yaml).map_err(|e| format!("missions.yaml: {e}"))?;
    let top = match &doc {
        Value::Null => return Ok(Missions::default()),
        Value::Mapping(top) => top,
        _ => return Err("missions.yaml: the top level is not a mapping".to_string()),
    };
    let keep = match top.get("keep") {
        None | Some(Value::Null) => None,
        Some(v) => Some(count(v, "missions.yaml: `keep`")?),
    };
    let prompts = match mapping(top.get("prompts"), "prompts")? {
        Some(section) => section
            .iter()
            .map(|(k, v)| {
                let name = name_of(k, "prompts")?;
                let text = text_of(v, "prompts", &name)?;
                Ok((name, text.to_string()))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?,
        None => BTreeMap::new(),
    };
    let generators = match top.get("generators") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Sequence(items)) => items
            .iter()
            .enumerate()
            .map(|(i, item)| parse_generator(i, item))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err("missions.yaml: `generators` is not a list".to_string()),
    };
    let workflows = match mapping(top.get("workflows"), "workflows")? {
        Some(section) => section
            .iter()
            .map(|(k, v)| parse_workflow(&name_of(k, "workflows")?, v))
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    for workflow in &workflows {
        validate(workflow)?;
    }
    validate_generators(&generators, &workflows)?;
    let mut missions = Missions {
        keep,
        prompts,
        generators,
        workflows,
    };
    resolve(&mut missions)?;
    Ok(missions)
}

/// The workflows `missions_yaml` declares, in the order written.
pub fn parse_workflows(missions_yaml: &str) -> Result<Vec<Workflow>, String> {
    Ok(parse_missions(missions_yaml)?.workflows)
}

/// A top-level section that must be a mapping, or `None` when absent.
fn mapping<'a>(section: Option<&'a Value>, key: &str) -> Result<Option<&'a Mapping>, String> {
    match section {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Mapping(m)) => Ok(Some(m)),
        Some(_) => Err(format!("missions.yaml: `{key}` is not a mapping of names")),
    }
}

fn parse_generator(index: usize, item: &Value) -> Result<Generator, String> {
    let ctx = format!("generator {}", index + 1);
    let Value::Mapping(body) = item else {
        return Err(format!("{ctx}: expected a mapping"));
    };
    let mut id = None;
    let mut call = None;
    let mut workflow = None;
    let mut weight = WEIGHT;
    let mut context = Vec::new();
    let mut prompt = None;
    for (k, v) in body {
        let key = name_of(k, &ctx)?;
        match key.as_str() {
            "id" => id = Some(text_of(v, &ctx, &key)?.to_string()),
            "call" => call = Some(text_of(v, &ctx, &key)?.to_string()),
            "workflow" => workflow = Some(text_of(v, &ctx, &key)?.to_string()),
            "weight" => weight = count(v, &format!("{ctx}: `weight`"))?,
            "context" => context = names(v, &ctx, &key)?,
            "prompt" => prompt = Some(text_of(v, &ctx, &key)?.to_string()),
            other => return Err(format!("{ctx}: unknown key `{other}`")),
        }
    }
    let id = id.ok_or_else(|| format!("{ctx}: has no `id`"))?;
    let ctx = format!("generator `{id}`");
    Ok(Generator {
        call: call.ok_or_else(|| format!("{ctx}: has no `call`"))?,
        workflow: workflow.ok_or_else(|| format!("{ctx}: has no `workflow`"))?,
        weight,
        context,
        prompt: prompt.ok_or_else(|| format!("{ctx}: has no `prompt`"))?,
        id,
    })
}

fn parse_workflow(name: &str, body: &Value) -> Result<Workflow, String> {
    let ctx = format!("workflow `{name}`");
    let Value::Mapping(body) = body else {
        return Err(format!("{ctx}: expected a mapping with `steps`"));
    };
    let mut send_backs = SEND_BACKS;
    let mut desk = None;
    let mut year = None;
    let mut on_failed = OnFailed::default();
    let mut steps = None;
    for (k, v) in body {
        let key = name_of(k, &ctx)?;
        match key.as_str() {
            "send-backs" => send_backs = count(v, &format!("{ctx}: `send-backs`"))?,
            "desk" => desk = Some(text_of(v, &ctx, &key)?.to_string()),
            "year" => year = Some(text_of(v, &ctx, &key)?.to_string()),
            "on-failed" => {
                let text = text_of(v, &ctx, &key)?;
                on_failed = OnFailed::parse(text).ok_or_else(|| {
                    format!("{ctx}: `on-failed: {text}` is not one of set-aside, restore, keep")
                })?
            }
            "steps" => {
                let Value::Mapping(m) = v else {
                    return Err(format!("{ctx}: `steps` is not a mapping of step names"));
                };
                steps = Some(
                    m.iter()
                        .map(|(k, v)| parse_step(name, &name_of(k, &ctx)?, v))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            other => return Err(format!("{ctx}: unknown key `{other}`")),
        }
    }
    Ok(Workflow {
        name: name.to_string(),
        send_backs,
        desk,
        year,
        on_failed,
        steps: steps.ok_or_else(|| format!("{ctx}: has no `steps`"))?,
    })
}

fn parse_step(workflow: &str, name: &str, body: &Value) -> Result<Step, String> {
    let ctx = format!("workflow `{workflow}`, step `{name}`");
    let Value::Mapping(body) = body else {
        return Err(format!("{ctx}: expected a mapping"));
    };
    let mut by = None;
    let mut prompt = None;
    let mut next = None;
    let mut call = None;
    let mut edits = Variants::One(Edits::default());
    let mut tools = Vec::new();
    let mut checks = Vec::new();
    let mut context = Vec::new();
    let mut stuck = None;
    let mut stuck_limit = None;
    for (k, v) in body {
        let key = name_of(k, &ctx)?;
        match key.as_str() {
            "by" => {
                let text = text_of(v, &ctx, &key)?;
                by = Some(By::parse(text).ok_or_else(|| {
                    format!("{ctx}: `by: {text}` is not one of maker, another, table")
                })?)
            }
            "prompt" => {
                prompt = Some(variants(v, &ctx, &key, |text| Ok(text.to_string()))?)
            }
            "next" => next = Some(parse_next(v, &ctx)?),
            "call" => call = Some(text_of(v, &ctx, &key)?.to_string()),
            "edits" => {
                edits = variants(v, &ctx, &key, |text| {
                    Edits::parse(text).ok_or_else(|| {
                        format!("{ctx}: `edits: {text}` is not one of new, change, optional")
                    })
                })?
            }
            "tools" => tools = names(v, &ctx, &key)?,
            "checks" => checks = names(v, &ctx, &key)?,
            "context" => context = names(v, &ctx, &key)?,
            "stuck" => stuck = Some(text_of(v, &ctx, &key)?.to_string()),
            "stuck-limit" => stuck_limit = Some(count(v, &format!("{ctx}: `stuck-limit`"))?),
            other => {
                return Err(format!(
                    "{ctx}: unknown key `{other}`; a step has by, prompt, next, call, edits, tools, checks, context, stuck, stuck-limit"
                ))
            }
        }
    }
    let by = by.ok_or_else(|| format!("{ctx}: has no `by`"))?;
    let next = next.ok_or_else(|| format!("{ctx}: has no `next`"))?;
    match (by, &call) {
        (By::Table, None) => return Err(format!("{ctx}: a table step names its `call`")),
        (By::Maker | By::Another, Some(_)) => {
            return Err(format!("{ctx}: only a table step has a `call`"))
        }
        _ => {}
    }
    if by == By::Table && (stuck.is_some() || stuck_limit.is_some()) {
        return Err(format!(
            "{ctx}: a table step is never stuck; `stuck` and `stuck-limit` are for actors"
        ));
    }
    if let Next::Outcomes(outcomes) = &next {
        if outcomes.iter().any(|(o, _)| o == STUCK) {
            return Err(format!(
                "{ctx}: `{STUCK}` is routed by the step's own `stuck` key, not by `next`"
            ));
        }
    }
    let stuck_limit = stuck_limit.unwrap_or(STUCK_LIMIT);
    if stuck_limit == 0 {
        return Err(format!("{ctx}: `stuck-limit` must be at least 1"));
    }
    Ok(Step {
        name: name.to_string(),
        by,
        prompt: prompt.ok_or_else(|| format!("{ctx}: has no `prompt`"))?,
        next,
        call,
        edits,
        tools,
        checks,
        context,
        stuck: stuck.unwrap_or_else(|| FAILED.to_string()),
        stuck_limit,
    })
}

/// `next:` — a step name, or a non-empty map of outcome → target.
fn parse_next(value: &Value, ctx: &str) -> Result<Next, String> {
    match value {
        Value::String(target) => Ok(Next::Single(target.clone())),
        Value::Mapping(outcomes) if outcomes.is_empty() => {
            Err(format!("{ctx}: `next` offers no outcomes"))
        }
        Value::Mapping(outcomes) => outcomes
            .iter()
            .map(|(k, v)| {
                let outcome = name_of(k, &format!("{ctx}, `next`"))?;
                let target = v
                    .as_str()
                    .ok_or_else(|| format!("{ctx}: outcome `{outcome}` does not name a step"))?;
                Ok((outcome, target.to_string()))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(Next::Outcomes),
        _ => Err(format!(
            "{ctx}: `next` is neither a step name nor a map of outcomes to steps"
        )),
    }
}

/// A text value, or a non-empty map of incoming outcome → text, each text
/// read by `read`.
fn variants<T>(
    value: &Value,
    ctx: &str,
    key: &str,
    read: impl Fn(&str) -> Result<T, String>,
) -> Result<Variants<T>, String> {
    match value {
        Value::String(text) => Ok(Variants::One(read(text)?)),
        Value::Mapping(m) if m.is_empty() => Err(format!("{ctx}: `{key}` has no variants")),
        Value::Mapping(m) => m
            .iter()
            .map(|(k, v)| {
                let outcome = name_of(k, &format!("{ctx}, `{key}`"))?;
                let text = text_of(v, ctx, &format!("{key}.{outcome}"))?;
                Ok((outcome, read(text)?))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(Variants::ByOutcome),
        _ => Err(format!(
            "{ctx}: `{key}` is neither text nor a map of outcomes to text"
        )),
    }
}

/// A text value.
fn text_of<'a>(value: &'a Value, ctx: &str, key: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{ctx}: `{key}` is not text"))
}

/// A whole number that fits a `u32`.
fn count(value: &Value, what: &str) -> Result<u32, String> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("{what} is not a whole number"))
}

/// A YAML list of names. A name that is not text, or is blank, is refused.
fn names(value: &Value, ctx: &str, key: &str) -> Result<Vec<String>, String> {
    let Value::Sequence(items) = value else {
        return Err(format!("{ctx}: `{key}` is not a list of names"));
    };
    items
        .iter()
        .map(|item| match item.as_str().map(str::trim) {
            Some(name) if !name.is_empty() => Ok(name.to_string()),
            _ => Err(format!(
                "{ctx}: `{key}` holds {item:?}, which is not a name"
            )),
        })
        .collect()
}

/// A mapping key as a name. Keys YAML reads as numbers or booleans are refused
/// rather than coerced, so `1:` or `true:` cannot quietly become a step.
fn name_of(key: &Value, ctx: &str) -> Result<String, String> {
    match key {
        Value::String(s) if !s.trim().is_empty() => Ok(s.clone()),
        other => Err(format!("{ctx}: key {other:?} is not a name")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-workflow file around `steps`, indented as step entries.
    fn steps(steps: &str) -> String {
        format!("workflows:\n  w:\n    steps:\n{steps}")
    }

    fn one(yaml: &str) -> Workflow {
        let mut all = parse_workflows(yaml).unwrap();
        assert_eq!(all.len(), 1);
        all.remove(0)
    }

    fn err(yaml: &str) -> String {
        parse_missions(yaml).unwrap_err()
    }

    const PLAIN: &str = "      s:\n        by: maker\n        next: done\n        prompt: p\n";

    #[test]
    fn steps_keep_the_order_they_are_listed_in_and_defaults_apply() {
        let wf = one(&steps("      zeta:\n        by: maker\n        next: alpha\n        prompt: z\n      alpha:\n        by: table\n        call: reading\n        next: mid\n        prompt: a\n      mid:\n        by: another\n        next: done\n        prompt: m\n"));
        let names: Vec<&str> = wf.steps.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["zeta", "alpha", "mid"]);
        assert_eq!(wf.start().name, "zeta");
        assert_eq!(wf.send_backs, 3);
        assert_eq!(wf.desk, None);
        assert_eq!(wf.year, None);
        assert_eq!(wf.on_failed, OnFailed::Keep);
        let zeta = wf.start();
        assert_eq!(zeta.edits, Variants::One(Edits::Optional));
        assert_eq!(zeta.stuck, "failed");
        assert_eq!(zeta.stuck_limit, 1);
        assert_eq!(zeta.call, None);
        assert!(zeta.tools.is_empty() && zeta.checks.is_empty() && zeta.context.is_empty());
        assert_eq!(wf.step("alpha").unwrap().call.as_deref(), Some("reading"));
    }

    #[test]
    fn workflow_keys_are_carried() {
        let wf = one(&format!(
            "workflows:\n  w:\n    send-backs: 0\n    desk: story-desk\n    year: era-opens\n    on-failed: set-aside\n    steps:\n{PLAIN}"
        ));
        assert_eq!(wf.send_backs, 0);
        assert_eq!(wf.desk.as_deref(), Some("story-desk"));
        assert_eq!(wf.year.as_deref(), Some("era-opens"));
        assert_eq!(wf.on_failed, OnFailed::SetAside);
    }

    #[test]
    fn every_step_key_is_read() {
        let wf = one(&steps("      read:\n        by: another\n        edits:\n          sound: optional\n          fail: new\n        next:\n          pass: done\n          fail: read\n        stuck: read\n        stuck-limit: 2\n        tools:\n          - report_rejected\n        checks:\n          - ' length '\n          - leakage\n        context: [draft, era-it-tells]\n        prompt:\n          sound: |\n            Read {objective}.\n          default: Again.\n"));
        let read = wf.start();
        assert_eq!(read.by, By::Another);
        assert_eq!(
            read.prompt,
            Variants::ByOutcome(vec![
                ("sound".to_string(), "Read {objective}.\n".to_string()),
                ("default".to_string(), "Again.".to_string()),
            ])
        );
        assert_eq!(
            read.edits,
            Variants::ByOutcome(vec![
                ("sound".to_string(), Edits::Optional),
                ("fail".to_string(), Edits::New),
            ])
        );
        assert_eq!(
            read.next,
            Next::Outcomes(vec![
                ("pass".to_string(), "done".to_string()),
                ("fail".to_string(), "read".to_string()),
            ])
        );
        assert_eq!(read.stuck, "read");
        assert_eq!(read.stuck_limit, 2);
        assert_eq!(read.tools, ["report_rejected"]);
        assert_eq!(read.checks, ["length", "leakage"]);
        assert_eq!(read.context, ["draft", "era-it-tells"]);
    }

    #[test]
    fn top_level_keep_prompts_and_generators_are_read_and_others_ignored() {
        let m = parse_missions(&format!(
            "system: ignored\nkeep: 4\nprompts:\n  table: Set work.\ngenerators:\n  - id: untold\n    call: story\n    workflow: w\n    weight: 2\n    context:\n      - era\n    prompt: Choose.\n  - id: plain\n    call: story\n    workflow: w\n    prompt: Again.\nworkflows:\n  w:\n    steps:\n{PLAIN}"
        ))
        .unwrap();
        assert_eq!(m.keep, Some(4));
        assert_eq!(m.prompts["table"], "Set work.");
        assert_eq!(
            m.generators,
            [
                Generator {
                    id: "untold".to_string(),
                    call: "story".to_string(),
                    workflow: "w".to_string(),
                    weight: 2,
                    context: vec!["era".to_string()],
                    prompt: "Choose.".to_string(),
                },
                Generator {
                    id: "plain".to_string(),
                    call: "story".to_string(),
                    workflow: "w".to_string(),
                    weight: 1,
                    context: Vec::new(),
                    prompt: "Again.".to_string(),
                },
            ]
        );
        assert!(m.workflow("w").is_some());
        assert!(m.workflow("x").is_none());
    }

    #[test]
    fn workflows_keep_their_order() {
        let all = parse_workflows(&format!(
            "workflows:\n  second:\n    steps:\n{PLAIN}  first:\n    steps:\n{PLAIN}"
        ))
        .unwrap();
        let names: Vec<&str> = all.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["second", "first"]);
    }

    #[test]
    fn a_file_without_these_sections_declares_nothing() {
        assert_eq!(parse_missions("").unwrap(), Missions::default());
        assert_eq!(parse_missions("system: x\n").unwrap(), Missions::default());
        assert_eq!(parse_workflows("workflows:\n").unwrap(), Vec::new());
    }

    #[test]
    fn an_unknown_key_is_refused_at_every_level() {
        assert_eq!(
            err(&steps("      s:\n        by: maker\n        next: done\n        pass: done\n        prompt: p\n")),
            "workflow `w`, step `s`: unknown key `pass`; a step has by, prompt, next, call, edits, tools, checks, context, stuck, stuck-limit"
        );
        assert_eq!(
            err(&format!(
                "workflows:\n  w:\n    limit: 3\n    steps:\n{PLAIN}"
            )),
            "workflow `w`: unknown key `limit`"
        );
        assert_eq!(
            err("generators:\n  - id: g\n    call: c\n    workflow: w\n    prompt: p\n    rate: 2\n"),
            "generator 1: unknown key `rate`"
        );
    }

    #[test]
    fn calls_belong_to_table_steps_only() {
        assert_eq!(
            err(&steps(
                "      s:\n        by: table\n        next: done\n        prompt: p\n"
            )),
            "workflow `w`, step `s`: a table step names its `call`"
        );
        assert_eq!(
            err(&steps("      s:\n        by: maker\n        call: reading\n        next: done\n        prompt: p\n")),
            "workflow `w`, step `s`: only a table step has a `call`"
        );
    }

    #[test]
    fn stuck_belongs_to_actor_steps_only() {
        assert_eq!(
            err(&steps("      s:\n        by: table\n        call: c\n        stuck: done\n        next: done\n        prompt: p\n")),
            "workflow `w`, step `s`: a table step is never stuck; `stuck` and `stuck-limit` are for actors"
        );
        assert_eq!(
            err(&steps("      s:\n        by: maker\n        next:\n          stuck: done\n        prompt: p\n")),
            "workflow `w`, step `s`: `stuck` is routed by the step's own `stuck` key, not by `next`"
        );
        assert_eq!(
            err(&steps("      s:\n        by: maker\n        stuck-limit: 0\n        next: done\n        prompt: p\n")),
            "workflow `w`, step `s`: `stuck-limit` must be at least 1"
        );
    }

    #[test]
    fn missing_keys_are_refused() {
        assert_eq!(
            err(&steps("      s:\n        next: done\n        prompt: p\n")),
            "workflow `w`, step `s`: has no `by`"
        );
        assert_eq!(
            err(&steps("      s:\n        by: maker\n        next: done\n")),
            "workflow `w`, step `s`: has no `prompt`"
        );
        assert_eq!(
            err(&steps("      s:\n        by: maker\n        prompt: p\n")),
            "workflow `w`, step `s`: has no `next`"
        );
        assert_eq!(
            err("workflows:\n  w:\n    desk: d\n"),
            "workflow `w`: has no `steps`"
        );
        assert_eq!(
            err("generators:\n  - call: c\n"),
            "generator 1: has no `id`"
        );
        assert_eq!(
            err("generators:\n  - id: g\n    call: c\n    prompt: p\n"),
            "generator `g`: has no `workflow`"
        );
    }

    #[test]
    fn malformed_values_are_refused() {
        let cases = [
            (
                steps("      s:\n        by: anyone\n        next: done\n        prompt: p\n"),
                "workflow `w`, step `s`: `by: anyone` is not one of maker, another, table",
            ),
            (
                steps("      s:\n        by: maker\n        edits: rewrite\n        next: done\n        prompt: p\n"),
                "workflow `w`, step `s`: `edits: rewrite` is not one of new, change, optional",
            ),
            (
                format!("workflows:\n  w:\n    on-failed: drop\n    steps:\n{PLAIN}"),
                "workflow `w`: `on-failed: drop` is not one of set-aside, restore, keep",
            ),
            (
                format!("workflows:\n  w:\n    send-backs: -1\n    steps:\n{PLAIN}"),
                "workflow `w`: `send-backs` is not a whole number",
            ),
            (
                steps("      s:\n        by: maker\n        next: {}\n        prompt: p\n"),
                "workflow `w`, step `s`: `next` offers no outcomes",
            ),
            (
                steps("      s:\n        by: maker\n        next: [done]\n        prompt: p\n"),
                "workflow `w`, step `s`: `next` is neither a step name nor a map of outcomes to steps",
            ),
            (
                steps("      s:\n        by: maker\n        next:\n          pass: 3\n        prompt: p\n"),
                "workflow `w`, step `s`: outcome `pass` does not name a step",
            ),
            (
                steps("      s:\n        by: maker\n        next: done\n        prompt: {}\n"),
                "workflow `w`, step `s`: `prompt` has no variants",
            ),
            (
                steps("      s:\n        by: maker\n        next: done\n        prompt: [p]\n"),
                "workflow `w`, step `s`: `prompt` is neither text nor a map of outcomes to text",
            ),
            (
                steps("      s:\n        by: maker\n        next: done\n        prompt:\n          fail: 3\n"),
                "workflow `w`, step `s`: `prompt.fail` is not text",
            ),
            (
                steps("      s:\n        by: maker\n        next: done\n        checks: a, b\n        prompt: p\n"),
                "workflow `w`, step `s`: `checks` is not a list of names",
            ),
            (
                steps("      s:\n        by: maker\n        next: done\n        tools:\n          - ''\n        prompt: p\n"),
                "workflow `w`, step `s`: `tools` holds String(\"\"), which is not a name",
            ),
            (
                steps("      s: text\n"),
                "workflow `w`, step `s`: expected a mapping",
            ),
            (
                steps("      true:\n        by: maker\n        next: done\n        prompt: p\n"),
                "workflow `w`: key Bool(true) is not a name",
            ),
            (
                "workflows: [w]\n".to_string(),
                "missions.yaml: `workflows` is not a mapping of names",
            ),
            (
                "prompts:\n  writing: 3\n".to_string(),
                "prompts: `writing` is not text",
            ),
            (
                "generators: {}\n".to_string(),
                "missions.yaml: `generators` is not a list",
            ),
            (
                "keep: many\n".to_string(),
                "missions.yaml: `keep` is not a whole number",
            ),
            (
                "- a\n".to_string(),
                "missions.yaml: the top level is not a mapping",
            ),
        ];
        for (yaml, expected) in cases {
            assert_eq!(err(&yaml), expected, "{yaml}");
        }
    }

    #[test]
    fn a_duplicate_step_name_is_a_yaml_error() {
        let e = err(&steps(&format!("{PLAIN}{PLAIN}")));
        assert!(e.starts_with("missions.yaml: "), "{e}");
        assert!(e.contains("duplicate"), "{e}");
    }
}
