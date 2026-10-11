//! A workflow step a Maker takes, as the mission set at the table for it.
//!
//! **One builder for every Maker step.** What a step asks is its workflow's:
//! the prompt for the outcome that led to it, filled from the operation —
//! what it is and what it is for, what the step before found, the proposal's
//! own fields — and the sections its `context` names. What the engine adds is
//! what it can check was done: the desk to go to, each document to read, the
//! write or change and its commit as the step's `edits` says, and the report.
//! The draft, the review that mends it, the fix sent back and the check
//! against the storyline are the same builder with different YAML.

use std::collections::BTreeMap;

use super::answer::Desk;
use super::canon::storyline;
use super::corpus::Corpus;
use super::gates::{Form, LIFE_MIN_WORDS, STORY_MIN_WORDS};
use super::reading::context;
use crate::engine::mission::{Mission, Origin, Todo, Work};
use crate::engine::work::what_happens;
use crate::engine::workflow::{fill, placeholders, Edits, Offered};
use crate::sim::operations::Operation;

/// The section whose documents a step reads.
const READS: &str = "reads";

/// The section naming the eras a step checks its document against.
const STORYLINE: &str = "storyline";

/// What `{told}` says for a document written to no brief.
const NO_BRIEF: &str = "it was written to no brief, so it owes no particular event — judge it \
                        by the event it tells.";

/// The mission for the step operation `op` waits on, as `offered` asks it,
/// done at `desk`. An error when the prompt names a placeholder the operation
/// cannot fill.
pub fn step_mission(
    op: &Operation,
    offered: &Offered<'_>,
    corpus: &Corpus,
    desk: Option<&Desk>,
) -> Result<Mission, String> {
    let step = offered.step;
    let doc = &op.document;
    let checks_storyline = step.context.iter().any(|c| c == STORYLINE);
    let eras = storyline(op, corpus);
    // What it reads: the operation's own reading list for the step that writes
    // it new; the document as it stands, and what it answers to, for any other.
    let reads: Vec<String> = match offered.edits {
        Edits::New if op.run.history.is_empty() => op.reads.clone(),
        _ => std::iter::once(doc.clone())
            .chain(match checks_storyline {
                true => eras.clone(),
                false => context(op, corpus),
            })
            .collect(),
    };
    let titles: Vec<String> = eras
        .iter()
        .filter_map(|p| corpus.eras.iter().find(|e| &e.path == p))
        .map(|e| match e.year {
            Some(y) => format!("{} ({y})", e.title),
            None => e.title.clone(),
        })
        .collect();
    // A document put through review by hand was written to no brief: there is
    // no event it owed, and its objective ("Review …") is not one.
    let told = op
        .fields
        .get("happens")
        .cloned()
        .or_else(|| what_happens(&op.brief))
        .unwrap_or_else(|| NO_BRIEF.to_string());
    let findings = op.findings().unwrap_or_default().to_string();
    let eras_said = titles.join(", ");
    let mut values: BTreeMap<&str, &str> = op
        .fields
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    values.insert("name", &op.name);
    values.insert("objective", &op.objective);
    values.insert("document", doc);
    values.insert("findings", &findings);
    values.insert("told", &told);
    values.insert("eras", &eras_said);
    let mut brief = fill(offered.prompt, &values)
        .map_err(|e| format!("step `{}` of {}: {e}", step.name, op.name))?
        .trim()
        .to_string();
    // The sections it names, after what it asks.
    let reads_said = reads_list(&reads);
    for name in &step.context {
        let section = match name.as_str() {
            READS => op
                .fields
                .get(READS)
                .filter(|s| !s.is_empty() && op.run.history.is_empty())
                .cloned()
                .unwrap_or_else(|| reads_said.clone()),
            STORYLINE => format!("The storyline around it: {eras_said}."),
            other => op.fields.get(other).cloned().unwrap_or_default(),
        };
        if !section.trim().is_empty() {
            brief.push_str("\n\n");
            brief.push_str(section.trim());
        }
    }
    let mut todo = Vec::new();
    if let Some(d) = desk {
        todo.push(Todo::new(format!("go to {} on {}", d.room, d.level)));
    }
    for r in &reads {
        todo.push(Todo::new(format!("read {r}")));
    }
    match offered.edits {
        Edits::New => todo.push(Todo::new(format!("write {doc} and commit it"))),
        Edits::Change => todo.push(Todo::new(format!(
            "change {doc} and commit it, putting right what was found"
        ))),
        Edits::Optional => {}
    }
    todo.push(Todo::report("go back to the table and report it"));
    Ok(Mission::new(
        brief,
        todo,
        Origin::Generated {
            generator: op.generator.clone(),
            target: op.target.clone(),
            operation: op.id,
            step: step.name.clone(),
        },
    )
    .with_work(Work {
        writes: doc.clone(),
        reads,
        min_words: match Form::of(doc) {
            Form::LifeEvent => LIFE_MIN_WORDS,
            Form::Story => STORY_MIN_WORDS,
            Form::Other => 0,
        },
        edit_optional: offered.edits == Edits::Optional,
        anew: offered.edits == Edits::New,
        checks: step.checks.clone(),
        tools: step.tools.clone(),
    }))
}

/// The documents a step reads, as its brief lists them.
fn reads_list(reads: &[String]) -> String {
    match reads.is_empty() {
        true => String::new(),
        false => format!(
            "Read before you decide:\n{}",
            reads
                .iter()
                .map(|r| format!("- `{r}`"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    }
}

/// The placeholders a step's prompt names that the engine fills itself, beside
/// the proposal's own fields.
pub const FILLED: &[&str] = &["name", "objective", "document", "findings", "told", "eras"];

/// The placeholders `prompt` names that neither the engine nor `fields` fill.
pub fn unfilled<'a>(prompt: &'a str, fields: &BTreeMap<String, String>) -> Vec<&'a str> {
    placeholders(prompt)
        .into_iter()
        .filter(|p| !FILLED.contains(p) && !fields.contains_key(*p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::engine::workflow::Taker;
    use crate::sim::operations::tests::workflows;
    use crate::sim::operations::Operations;

    fn keeper() -> (tempfile::TempDir, Corpus, Operations, u64) {
        let dir = mind();
        let corpus = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        let doc = "layers/life/keeper/2488 The Empty Table.md";
        let id = ops
            .open(
                "life-event",
                None,
                "life-event",
                "life:keeper",
                "Keeper's life, 2488",
                doc,
            )
            .unwrap();
        let fields: BTreeMap<String, String> = [
            ("subject", "Keeper"),
            ("grain", "year"),
            ("date", "2488"),
            ("title", "The Empty Table"),
            ("happens", "Keeper answers Alpha Centauri."),
            ("turns", "Keeper answers that it survived."),
            ("agrees", "The Fall."),
            ("leaves", "It means to wait for orders."),
            ("voice", "the first person plural — \"we\", as Keeper"),
            ("anchor", "Who Keeper is: you keep the towers."),
            ("world-then", "The world then was the Fall's."),
            (
                "reads",
                "Read first: `layers/eras/the-fall.md` — the era it falls in.",
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        ops.briefed(
            id,
            "the brief",
            fields,
            vec!["layers/eras/the-fall.md".to_string()],
        );
        (dir, corpus, ops, id)
    }

    fn mission(ops: &Operations, id: u64, corpus: &Corpus) -> Mission {
        let offer = ops.offer(id).unwrap();
        step_mission(ops.get(id).unwrap(), &offer, corpus, None).unwrap()
    }

    /// **The write step is the workflow's prompt, filled from the proposal**,
    /// with the sections it names after it, and steps the engine can see done.
    #[test]
    fn the_write_step_is_the_workflows_prompt_filled_from_the_proposal() {
        let (_dir, corpus, ops, id) = keeper();
        let m = mission(&ops, id, &corpus);
        assert!(m.prompt.starts_with(
            "Write Keeper's life where the record has nothing: the year\n2488, \"The Empty Table\"."
        ), "{}", m.prompt);
        assert!(m
            .prompt
            .contains("It turns on: Keeper answers that it survived."));
        assert!(m
            .prompt
            .contains("Write it in the first person plural — \"we\", as Keeper,"));
        assert!(
            m.prompt.contains("nothing of the vault belongs in it"),
            "the shared prompt"
        );
        assert!(m
            .prompt
            .ends_with("Read first: `layers/eras/the-fall.md` — the era it falls in."));
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "read layers/eras/the-fall.md",
                "write layers/life/keeper/2488 The Empty Table.md and commit it",
                "go back to the table and report it",
            ]
        );
        assert_eq!(m.operation(), Some((id, "write")));
        let w = m.work.unwrap();
        assert_eq!(
            (w.edit_optional, w.anew, w.min_words),
            (false, true, LIFE_MIN_WORDS)
        );
        assert!(w.checks.iter().any(|c| c == "voice"));
        assert!(w.tools.is_empty());
    }

    /// **A review answers to what the table found**: the mend variant, the
    /// reading as its findings, the document and what it answers to read, a
    /// change to commit, and `report_rejected` among its acts.
    #[test]
    fn a_review_carries_the_tables_findings_and_may_reject() {
        let (_dir, corpus, mut ops, id) = keeper();
        ops.advance(id, &Taker::Actor("wren".into()), None, "written")
            .unwrap();
        ops.advance(
            id,
            &Taker::Table,
            Some("mend"),
            "\"we slammed it\" — no body",
        )
        .unwrap();
        let m = mission(&ops, id, &corpus);
        assert!(m
            .prompt
            .starts_with("Operation Iron Lantern — Keeper's life, 2488."));
        assert!(m
            .prompt
            .contains("What happens: Keeper answers Alpha Centauri."));
        assert!(m.prompt.contains("\"we slammed it\" — no body"));
        assert!(m.prompt.contains("First, mend it."));
        assert!(m.prompt.contains("`report_rejected` with why"));
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps.first(),
            Some(&"read layers/life/keeper/2488 The Empty Table.md")
        );
        assert!(steps.contains(
            &"change layers/life/keeper/2488 The Empty Table.md and commit it, putting right what was found"
        ));
        assert!(m.may_reject());
        let w = m.work.unwrap();
        assert_eq!((w.edit_optional, w.anew), (false, false));
    }

    /// **A canon check reads the storyline around the document and names it**,
    /// changing nothing unless it finds something.
    #[test]
    fn a_canon_check_reads_and_names_the_storyline() {
        let (_dir, corpus, mut ops, id) = keeper();
        ops.advance(id, &Taker::Actor("wren".into()), None, "written")
            .unwrap();
        ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
        ops.advance(id, &Taker::Actor("pax".into()), Some("pass"), "read it")
            .unwrap();
        ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
        let m = mission(&ops, id, &corpus);
        assert!(
            m.prompt
                .contains("read the storyline around it — The Fall (2487)"),
            "{}",
            m.prompt
        );
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert!(steps.contains(&"read layers/eras/the-fall.md"), "{steps:?}");
        assert!(m.work.unwrap().edit_optional);
    }

    /// **A document put through review by hand owes no event**: its review
    /// says so rather than offering the operation's objective as one.
    #[test]
    fn a_review_of_a_document_with_no_brief_says_it_owes_no_event() {
        let dir = mind();
        let corpus = Corpus::read(dir.path(), "test");
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        let id = ops
            .open(
                "story",
                Some("read"),
                "operator",
                "doc:layers/stories/the-charge.md",
                "Review layers/stories/the-charge.md",
                "layers/stories/the-charge.md",
            )
            .unwrap();
        ops.advance(id, &Taker::Table, Some("sound"), "").unwrap();
        let m = mission(&ops, id, &corpus);
        assert!(
            m.prompt.contains(&format!("What happens: {NO_BRIEF}")),
            "{}",
            m.prompt
        );
    }

    #[test]
    fn a_placeholder_nothing_fills_is_named() {
        let fields = BTreeMap::from([("title".to_string(), "x".to_string())]);
        assert_eq!(unfilled("{title} {objective} {when}", &fields), ["when"]);
    }
}
