//! Prompt includes, resolved once at load.
//!
//! A shared prompt, a generator's prompt, or any step's prompt (each variant
//! of one) may include other text by reference:
//!
//! - `{name}` — the shared prompt `name` from `prompts:`, when one is
//!   declared. A `{name}` that names no shared prompt is an operation
//!   placeholder and is left for [`super::prompt::fill`]; a shared prompt
//!   therefore takes precedence over an operation placeholder of the same name.
//! - `{step.prompt}` / `{step.prompt.variant}` — a step's prompt in the same
//!   workflow, or one variant of it.
//! - `{workflow.step.prompt}` / `{workflow.step.prompt.variant}` — the same,
//!   in any workflow.
//!
//! Included text is spliced in verbatim, itself resolved first, so includes
//! nest. A load error names: a step reference that names no step, a variant
//! that does not exist, a prompt with variants included without naming one, a
//! same-workflow reference from a shared or generator prompt (which belong to
//! no workflow), and a cycle of includes. A dotted name with no `prompt` part
//! (`{a.b}`) is not an include and is left as text.

use std::collections::BTreeMap;

use super::config::{Missions, Variants};

/// A text that can include, or be included.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Source {
    Shared(String),
    /// A workflow, a step, and the variant when the prompt has variants.
    Step(String, String, Option<String>),
    Generator(String),
}

impl Source {
    /// How an error names it.
    fn label(&self) -> String {
        match self {
            Source::Shared(name) => format!("prompt `{name}`"),
            Source::Step(wf, step, None) => format!("workflow `{wf}`, step `{step}`"),
            Source::Step(wf, step, Some(v)) => {
                format!("workflow `{wf}`, step `{step}`, prompt `{v}`")
            }
            Source::Generator(id) => format!("generator `{id}`"),
        }
    }

    /// How a cycle names it: the reference that includes it.
    fn reference(&self) -> String {
        match self {
            Source::Shared(name) => format!("{{{name}}}"),
            Source::Step(wf, step, None) => format!("{{{wf}.{step}.prompt}}"),
            Source::Step(wf, step, Some(v)) => format!("{{{wf}.{step}.prompt.{v}}}"),
            Source::Generator(id) => format!("generator `{id}`"),
        }
    }
}

/// The resolution's view of everything as written.
struct Texts<'a> {
    missions: &'a Missions,
}

impl Texts<'_> {
    fn raw(&self, source: &Source) -> Option<&str> {
        match source {
            Source::Shared(name) => self.missions.prompts.get(name).map(String::as_str),
            Source::Step(wf, step, variant) => {
                let prompt = &self.missions.workflow(wf)?.step(step)?.prompt;
                match (prompt, variant) {
                    (Variants::One(text), None) => Some(text),
                    (Variants::ByOutcome(_), Some(v)) => prompt.variant(v).map(String::as_str),
                    _ => None,
                }
            }
            Source::Generator(id) => self
                .missions
                .generators
                .iter()
                .find(|g| &g.id == id)
                .map(|g| g.prompt.as_str()),
        }
    }

    /// What a `{token}` inside `from` refers to: `Ok(None)` when it is not an
    /// include.
    fn target(&self, from: &Source, token: &str) -> Result<Option<Source>, String> {
        let parts: Vec<&str> = token.split('.').collect();
        let here = || {
            match from {
            Source::Step(wf, _, _) => Ok(wf.clone()),
            Source::Shared(_) | Source::Generator(_) => Err(format!(
                "{}: includes `{{{token}}}`, but it belongs to no workflow; name the workflow, as `{{workflow.{token}}}`",
                from.label()
            )),
        }
        };
        let (wf, step, variant) = match parts.as_slice() {
            [name] => {
                return Ok(self
                    .missions
                    .prompts
                    .contains_key(*name)
                    .then(|| Source::Shared(name.to_string())))
            }
            [step, "prompt"] => (here()?, *step, None),
            [step, "prompt", v] => (here()?, *step, Some(*v)),
            [wf, step, "prompt"] => (wf.to_string(), *step, None),
            [wf, step, "prompt", v] => (wf.to_string(), *step, Some(*v)),
            _ if parts.contains(&"prompt") => {
                return Err(format!(
                    "{}: `{{{token}}}` is not a reference; a step's prompt is `{{[workflow.]step.prompt[.variant]}}`",
                    from.label()
                ))
            }
            _ => return Ok(None),
        };
        let Some(prompt) = self
            .missions
            .workflow(&wf)
            .and_then(|w| w.step(step))
            .map(|s| &s.prompt)
        else {
            return Err(format!(
                "{}: includes `{{{token}}}`, which names no step",
                from.label()
            ));
        };
        match (prompt, variant) {
            (Variants::ByOutcome(_), None) => Err(format!(
                "{}: includes `{{{token}}}`, whose prompt has variants; name one, as `{{{token}.<outcome>}}`",
                from.label()
            )),
            (Variants::One(_), Some(_)) => Err(format!(
                "{}: includes `{{{token}}}`, but that step has one prompt and no variants",
                from.label()
            )),
            (Variants::ByOutcome(_), Some(v)) if prompt.variant(v).is_none() => Err(format!(
                "{}: includes `{{{token}}}`, which names no such variant",
                from.label()
            )),
            _ => Ok(Some(Source::Step(
                wf,
                step.to_string(),
                variant.map(str::to_string),
            ))),
        }
    }

    /// `source`'s text with its includes resolved, memoised in `done`;
    /// `stack` is the chain of includes being resolved.
    fn expand(
        &self,
        source: &Source,
        done: &mut BTreeMap<Source, String>,
        stack: &mut Vec<Source>,
    ) -> Result<String, String> {
        if let Some(text) = done.get(source) {
            return Ok(text.clone());
        }
        if let Some(at) = stack.iter().position(|s| s == source) {
            let chain: Vec<String> = stack[at..]
                .iter()
                .chain([source])
                .map(Source::reference)
                .collect();
            return Err(format!(
                "{}: its prompt includes itself: {}",
                source.label(),
                chain.join(" → ")
            ));
        }
        let raw = self.raw(source).unwrap_or_default();
        stack.push(source.clone());
        let mut out = String::with_capacity(raw.len());
        let mut from = 0;
        while let Some(offset) = raw[from..].find('{') {
            let open = from + offset;
            let rest = &raw[open + 1..];
            let len = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
                .unwrap_or(rest.len());
            let token = &rest[..len];
            let included = match len > 0 && rest[len..].starts_with('}') {
                true => self.target(source, token)?,
                false => None,
            };
            match included {
                Some(target) => {
                    out.push_str(&raw[from..open]);
                    out.push_str(&self.expand(&target, done, stack)?);
                    from = open + 1 + len + 1;
                }
                None => {
                    out.push_str(&raw[from..open + 1]);
                    from = open + 1;
                }
            }
        }
        out.push_str(&raw[from..]);
        stack.pop();
        done.insert(source.clone(), out.clone());
        Ok(out)
    }
}

/// Resolve every include in `missions` in place: shared prompts, generator
/// prompts, and every variant of every step's prompt. Shared prompts are
/// resolved even when nothing includes them, so an error in one is still
/// reported.
pub fn resolve(missions: &mut Missions) -> Result<(), String> {
    let texts = Texts {
        missions: &*missions,
    };
    let mut done = BTreeMap::new();
    let mut expand = |source: Source| texts.expand(&source, &mut done, &mut Vec::new());

    let mut shared = BTreeMap::new();
    for name in texts.missions.prompts.keys() {
        shared.insert(name.clone(), expand(Source::Shared(name.clone()))?);
    }
    let mut generators = Vec::new();
    for g in &texts.missions.generators {
        generators.push(expand(Source::Generator(g.id.clone()))?);
    }
    let mut steps = Vec::new();
    for wf in &texts.missions.workflows {
        for step in &wf.steps {
            let source = |v: Option<&str>| {
                Source::Step(wf.name.clone(), step.name.clone(), v.map(str::to_string))
            };
            steps.push(match &step.prompt {
                Variants::One(_) => Variants::One(expand(source(None))?),
                Variants::ByOutcome(variants) => Variants::ByOutcome(
                    variants
                        .iter()
                        .map(|(k, _)| Ok((k.clone(), expand(source(Some(k.as_str())))?)))
                        .collect::<Result<Vec<_>, String>>()?,
                ),
            });
        }
    }

    missions.prompts = shared;
    for (g, prompt) in missions.generators.iter_mut().zip(generators) {
        g.prompt = prompt;
    }
    let all_steps = missions
        .workflows
        .iter_mut()
        .flat_map(|w| w.steps.iter_mut());
    for (step, prompt) in all_steps.zip(steps) {
        step.prompt = prompt;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::config::Variants;
    use super::super::parse::parse_missions;

    fn prompts_of(yaml: &str) -> Vec<(String, Variants<String>)> {
        parse_missions(yaml)
            .unwrap()
            .workflows
            .into_iter()
            .flat_map(|w| {
                let wf = w.name.clone();
                w.steps
                    .into_iter()
                    .map(move |s| (format!("{wf}.{}", s.name), s.prompt))
            })
            .collect()
    }

    fn one(text: &str) -> Variants<String> {
        Variants::One(text.to_string())
    }

    fn err(yaml: &str) -> String {
        parse_missions(yaml).unwrap_err()
    }

    fn step(name: &str, prompt: &str) -> String {
        format!("      {name}:\n        by: maker\n        next: done\n        prompt: {prompt}\n")
    }

    const TWO: &str = "workflows:\n  story:\n    steps:\n      read:\n        by: maker\n        next: read-again\n        prompt: \"Read {objective}. {voice}\"\n      read-again:\n        by: maker\n        next: done\n        prompt: \"Again: {read.prompt}\"\n  life-event:\n    steps:\n      read:\n        by: maker\n        next: done\n        prompt: \"{story.read-again.prompt} / {writing}\"\n";

    #[test]
    fn shared_and_step_includes_resolve_and_nest() {
        let yaml = format!("prompts:\n  voice: \"In {{tone}}.\"\n  writing: \"W\"\n{TWO}");
        assert_eq!(
            prompts_of(&yaml),
            [
                (
                    "story.read".to_string(),
                    one("Read {objective}. In {tone}.")
                ),
                (
                    "story.read-again".to_string(),
                    one("Again: Read {objective}. In {tone}.")
                ),
                (
                    "life-event.read".to_string(),
                    one("Again: Read {objective}. In {tone}. / W")
                ),
            ]
        );
    }

    #[test]
    fn included_text_is_spliced_verbatim() {
        let yaml = "prompts:\n  review: |\n    Findings: {findings}\nworkflows:\n  w:\n    steps:\n      a:\n        by: maker\n        next: done\n        prompt: |\n          {review}\n          Mend it.\n";
        assert_eq!(
            prompts_of(yaml),
            [("w.a".to_string(), one("Findings: {findings}\n\nMend it.\n"))]
        );
    }

    #[test]
    fn variants_resolve_and_can_be_included_by_name() {
        let yaml = "prompts:\n  head: \"H\"\nworkflows:\n  story:\n    steps:\n      review:\n        by: another\n        next: done\n        prompt:\n          sound: \"{head} sound\"\n          mend: \"{head} mend\"\n  life:\n    steps:\n      review:\n        by: another\n        next: done\n        prompt:\n          sound: \"{story.review.prompt.sound}\"\n          fail: \"{story.review.prompt.mend}!\"\n      again:\n        by: maker\n        next: done\n        prompt: \"{review.prompt.fail}\"\n";
        let by = |pairs: &[(&str, &str)]| {
            Variants::ByOutcome(
                pairs
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            )
        };
        assert_eq!(
            prompts_of(yaml),
            [
                (
                    "story.review".to_string(),
                    by(&[("sound", "H sound"), ("mend", "H mend")])
                ),
                (
                    "life.review".to_string(),
                    by(&[("sound", "H sound"), ("fail", "H mend!")])
                ),
                ("life.again".to_string(), one("H mend!")),
            ]
        );
    }

    #[test]
    fn generator_and_shared_prompts_resolve() {
        let m = parse_missions("prompts:\n  base: \"B\"\n  table: \"{base} table\"\ngenerators:\n  - id: g\n    call: c\n    workflow: w\n    prompt: \"{table} / {w.s.prompt}\"\nworkflows:\n  w:\n    steps:\n      s:\n        by: maker\n        next: done\n        prompt: S\n").unwrap();
        assert_eq!(m.prompts["table"], "B table");
        assert_eq!(m.generators[0].prompt, "B table / S");
    }

    #[test]
    fn a_name_that_is_no_shared_prompt_is_left_for_fill() {
        assert_eq!(
            prompts_of(TWO)[0],
            ("story.read".to_string(), one("Read {objective}. {voice}"))
        );
    }

    #[test]
    fn json_and_other_dotted_braces_are_text() {
        let yaml = format!(
            "workflows:\n  w:\n    steps:\n{}",
            step("s", "'{\"a\": 1} {a.b} {x'")
        );
        assert_eq!(
            prompts_of(&yaml),
            [("w.s".to_string(), one("{\"a\": 1} {a.b} {x"))]
        );
    }

    #[test]
    fn bad_references_are_refused() {
        let wf = |prompt: &str| format!("workflows:\n  w:\n    steps:\n{}", step("s", prompt));
        let variants = "      v:\n        by: maker\n        next: done\n        prompt:\n          sound: x\n";
        let cases = [
            (wf("\"{nope.prompt}\""), "workflow `w`, step `s`: includes `{nope.prompt}`, which names no step"),
            (wf("\"{other.s.prompt}\""), "workflow `w`, step `s`: includes `{other.s.prompt}`, which names no step"),
            (wf("\"{a.b.c.d.prompt}\""), "workflow `w`, step `s`: `{a.b.c.d.prompt}` is not a reference; a step's prompt is `{[workflow.]step.prompt[.variant]}`"),
            (wf("\"{s.prompt.sound}\""), "workflow `w`, step `s`: includes `{s.prompt.sound}`, but that step has one prompt and no variants"),
            (format!("{}{variants}", wf("\"{v.prompt}\"")), "workflow `w`, step `s`: includes `{v.prompt}`, whose prompt has variants; name one, as `{v.prompt.<outcome>}`"),
            (format!("{}{variants}", wf("\"{v.prompt.mend}\"")), "workflow `w`, step `s`: includes `{v.prompt.mend}`, which names no such variant"),
            (format!("prompts:\n  p: \"{{s.prompt}}\"\n{}", wf("x")), "prompt `p`: includes `{s.prompt}`, but it belongs to no workflow; name the workflow, as `{workflow.s.prompt}`"),
        ];
        for (yaml, expected) in cases {
            assert_eq!(err(&yaml), expected, "{yaml}");
        }
    }

    #[test]
    fn a_cycle_is_refused_naming_its_chain() {
        assert_eq!(
            err("workflows:\n  w:\n    steps:\n      a:\n        by: maker\n        next: b\n        prompt: \"{b.prompt}\"\n      b:\n        by: maker\n        next: done\n        prompt: \"{a.prompt}\"\n"),
            "workflow `w`, step `a`: its prompt includes itself: {w.a.prompt} → {w.b.prompt} → {w.a.prompt}"
        );
        assert_eq!(
            err("prompts:\n  x: \"{y}\"\n  y: \"{x}\"\n"),
            "prompt `x`: its prompt includes itself: {x} → {y} → {x}"
        );
        assert_eq!(
            err(&format!(
                "workflows:\n  w:\n    steps:\n{}",
                step("a", "\"me: {a.prompt}\"")
            )),
            "workflow `w`, step `a`: its prompt includes itself: {w.a.prompt} → {w.a.prompt}"
        );
    }
}
