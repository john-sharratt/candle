//! The survey a maintenance mission starts from (`docs/zen_memory_and_guides.md`
//! Phase 7): read only, the candidates the rules can name with evidence —
//! a line past its `expires`, a line saying again what an earlier line of its
//! journal says, a summary cut short or left stale by a refused rebuild, and a
//! line whose `source` file has changed since it was written. Whether a line
//! is off its journal's cue is a judgement, and is the survey task's own.
//!
//! A candidate is not a problem: the tree is sound with every one of them in
//! it. Each is a proposal the maintenance task verifies before it becomes an
//! operation.

use std::collections::HashMap;
use std::fmt;

use crate::line::{Kind, Line, LineId};
use crate::path::JournalPath;
use crate::terms::{is_current, replaced};
use crate::time::Stamp;
use crate::tree::Tree;

/// What a candidate is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finding {
    /// Past its `expires`: hidden, and to be expired from the window.
    Expired { at: Stamp },
    /// Says what the earlier line `of` of its journal says.
    Duplicate { of: LineId },
    /// The journal's summary ends mid-sentence, or its last rebuild was
    /// refused.
    CutSummary,
    /// Its `source` names `file`, which has changed since it was written.
    SourceChanged { file: String },
}

/// One candidate: the journal, the line when it is about one, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub path: JournalPath,
    pub line: Option<LineId>,
    pub finding: Finding,
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.line {
            Some(id) => write!(f, "{}#{id}: ", self.path)?,
            None => write!(f, "{}: ", self.path)?,
        }
        match &self.finding {
            Finding::Expired { at } => write!(f, "expired at {at}"),
            Finding::Duplicate { of } => write!(f, "says again what #{of} says"),
            Finding::CutSummary => write!(f, "its summary is cut short or stale"),
            Finding::SourceChanged { file } => {
                write!(f, "its source {file} has changed since it was written")
            }
        }
    }
}

/// The file a `source` names, when it names one: the part before a line
/// number (`:12`), an anchor (`#x`) or a section (` §3`). A URL, or text
/// naming no path, names none.
pub fn source_file(source: &str) -> Option<&str> {
    let first = source.split_whitespace().next()?;
    if first.contains("://") {
        return None;
    }
    let file = first.split([':', '#']).next()?;
    (file.contains('/') || file.contains('.')).then_some(file)
}

/// A line's text as two lines are compared: lowercase, its words single
/// spaced, its closing punctuation dropped.
fn said(text: &str) -> String {
    let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    words
        .join(" ")
        .trim_end_matches(['.', '!', '?', ';', ','])
        .to_string()
}

/// Whether a summary's text ends a sentence.
fn ends_a_sentence(text: &str) -> bool {
    text.trim_end()
        .ends_with(['.', '!', '?', '…', ')', '"', '”', '\''])
}

/// Every candidate in `tree` at `now`, journal by journal, each journal's in
/// the order of its window. `changed_since(file, at)` says whether `file` has
/// changed since `at`; the survey asks it only of a line's `source` file.
pub fn survey(
    tree: &Tree,
    now: Stamp,
    changed_since: &dyn Fn(&str, Stamp) -> bool,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (path, j) in tree.journals() {
        if let Some(s) = j.summary() {
            if s.stale || (!s.is_empty() && !ends_a_sentence(&s.text())) {
                out.push(Candidate {
                    path: path.clone(),
                    line: None,
                    finding: Finding::CutSummary,
                });
            }
        }
        let replaced = replaced(&j.window);
        let mut first_said: HashMap<String, &LineId> = HashMap::new();
        for line in j.window.iter().filter(|l| !replaced.contains(&l.id)) {
            if let Some(finding) = finding(line, now, &mut first_said, changed_since) {
                out.push(Candidate {
                    path: path.clone(),
                    line: Some(line.id.clone()),
                    finding,
                });
            }
        }
    }
    out
}

/// What the survey finds of one shown line, the first finding only: an
/// expired line is not also a duplicate.
fn finding<'a>(
    line: &'a Line,
    now: Stamp,
    first_said: &mut HashMap<String, &'a LineId>,
    changed_since: &dyn Fn(&str, Stamp) -> bool,
) -> Option<Finding> {
    if !is_current(line, now) {
        return line.terms.expires.map(|at| Finding::Expired { at });
    }
    // Only a note says a fact: the same episode twice is two things done.
    if line.kind == Kind::Note {
        if let Some(of) = first_said.get(&said(&line.text)) {
            return Some(Finding::Duplicate { of: (*of).clone() });
        }
        first_said.insert(said(&line.text), &line.id);
    }
    let file = line.terms.source.as_deref().and_then(source_file)?;
    changed_since(file, line.at).then(|| Finding::SourceChanged {
        file: file.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{Meta, Sentence, Summary};
    use crate::ops::fixture::{p, zen};
    use crate::testing::{at, note};
    use crate::tree::fingerprint::fingerprint;

    #[test]
    fn a_source_names_its_file_or_none() {
        assert_eq!(
            source_file("zend/src/branch_ingest/keys.rs:12"),
            Some("zend/src/branch_ingest/keys.rs")
        );
        assert_eq!(
            source_file("docs/zen_memory_and_guides.md §3"),
            Some("docs/zen_memory_and_guides.md")
        );
        assert_eq!(source_file("README.md#usage"), Some("README.md"));
        assert_eq!(source_file("https://example.com/a.html"), None);
        assert_eq!(source_file("John said so"), None);
    }

    /// **The survey names each candidate with its evidence**: an expired
    /// reminder, a line said twice, a line whose source changed — and not a
    /// line another replaces, nor one whose source is unchanged.
    #[test]
    fn the_survey_names_its_candidates() {
        let mut t = zen();
        let topics = p("topics");
        let mut reminder = note("2026-10-29T20:03:00Z", "Service at 9am today.");
        reminder.terms.expires = Some(at("2026-10-31T00:00:00Z"));
        let first = note("2026-11-01T00:00:00Z", "The build runs on WhiteWorld.");
        let again = note("2026-11-02T00:00:00Z", "the build  runs on WhiteWorld");
        let mut sourced = note("2026-11-03T00:00:00Z", "The key version is 4.");
        sourced.terms.source = Some("zend/src/branch_ingest/keys.rs:12".into());
        let mut steady = note("2026-11-04T00:00:00Z", "The ingest has two layers.");
        steady.terms.source = Some("docs/ingest.md §2".into());
        let old = note("2026-11-05T00:00:00Z", "The gate is 30 ms.");
        let mut new = note("2026-11-06T00:00:00Z", "The gate is 30 ms.");
        new.terms.replaces = Some(old.id.clone());
        let j = t.get_mut(&topics).unwrap();
        j.window = vec![
            reminder.clone(),
            first.clone(),
            again.clone(),
            sourced.clone(),
            steady,
            old,
            new,
        ];
        let changed = |file: &str, since: Stamp| {
            file == "zend/src/branch_ingest/keys.rs" && since == at("2026-11-03T00:00:00Z")
        };
        let got: Vec<String> = survey(&t, at("2026-11-08T00:00:00Z"), &changed)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            got,
            [
                format!(
                    "@zen/topics#{}: expired at 2026-10-31T00:00:00Z",
                    reminder.id
                ),
                format!(
                    "@zen/topics#{}: says again what #{} says",
                    again.id, first.id
                ),
                format!(
                    "@zen/topics#{}: its source zend/src/branch_ingest/keys.rs has changed since \
                     it was written",
                    sourced.id
                ),
            ]
        );
    }

    /// **A summary that ends mid-sentence, or whose rebuild was refused, is
    /// cut**; one that ends a sentence is not.
    #[test]
    fn a_cut_or_stale_summary_is_a_candidate() {
        let mut t = zen();
        let topics = p("topics");
        let summary = |text: &str, stale: bool, over| Summary {
            over,
            built: at("2026-11-01T00:00:00Z"),
            lines: 1,
            children: 0,
            stale,
            digest: None,
            sentences: vec![Sentence {
                text: text.into(),
                cites: vec![],
            }],
        };
        let mut found = |text: &str, stale: bool| {
            let over = fingerprint(t.get(&topics).unwrap());
            let j = t.get_mut(&topics).unwrap();
            j.meta
                .get_or_insert_with(|| Meta::new("topics", None, at("2026-10-30T00:00:00Z")))
                .summary = Some(summary(text, stale, over));
            survey(&t, at("2026-11-08T00:00:00Z"), &|_, _| false)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            found("The work runs on two machines and the", false),
            ["@zen/topics: its summary is cut short or stale"]
        );
        assert_eq!(
            found("The work runs on two machines.", true),
            ["@zen/topics: its summary is cut short or stale"]
        );
        assert!(found("The work runs on two machines.", false).is_empty());
    }
}
