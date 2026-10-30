//! What a summary is built over (§5.3): the journal's window and each live
//! child's name, cue and summary. A journal is dirty exactly when the
//! fingerprint of those, now, is not the one its summary records — whatever
//! changed them: a line added, edited or forgotten, a child created, folded
//! or merged, a child's summary rebuilt. No history is needed to tell.

use crate::files::Fingerprint;
use crate::line::Folds;
use crate::path::JournalPath;
use crate::time::Stamp;
use crate::tree::{Journal, Tree};

/// The fingerprint of `j`'s inputs as they stand.
pub fn fingerprint(j: &Journal) -> Fingerprint {
    let mut bytes = Vec::new();
    let mut field = |s: &str| {
        bytes.extend_from_slice(s.as_bytes());
        bytes.push(0);
    };
    field("window");
    for line in &j.window {
        field(line.id.as_str());
        field(line.kind.name());
        field(&match &line.folds {
            Some(Folds::Period(p)) => format!("period {p}"),
            Some(Folds::Journal(s)) => format!("journal {s}"),
            None => String::new(),
        });
        field(if line.proposed { "proposed" } else { "" });
        field(&line.text);
    }
    field("children");
    for (name, child) in j.live() {
        field(name.as_str());
        field(&child.cue().unwrap_or_default());
        field(&child.summary().map(|s| s.text()).unwrap_or_default());
    }
    Fingerprint::of_bytes(&bytes)
}

/// It has something to summarise: lines, or a live child whose summary says
/// something. A child with none yet — or one emptied when its lines were
/// forgotten — gives its parent nothing to say, so an empty person's root and
/// fixed sections are never dirty, and a tree is summarised from its leaves
/// up.
pub fn has_inputs(j: &Journal) -> bool {
    !j.window.is_empty() || j.live().any(|(_, c)| speaks(c))
}

/// `j` has a summary that says something: one a parent can rest on.
pub fn speaks(j: &Journal) -> bool {
    j.summary().is_some_and(|s| !s.is_empty())
}

pub fn is_dirty(j: &Journal) -> bool {
    match j.summary() {
        None => has_inputs(j),
        Some(s) => s.over != fingerprint(j),
    }
}

/// A journal whose summary wants rebuilding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dirty {
    pub path: JournalPath,
    /// Window lines newer than its summary — all of them when it has none.
    pub new_lines: usize,
    /// When anything last changed in its window: the rebuild waits for it
    /// to be quiet.
    pub last_line: Option<Stamp>,
}

/// Every dirty, writable journal, deepest first — so a child is rebuilt
/// before the parent that reads its summary.
pub fn dirty(tree: &Tree) -> Vec<Dirty> {
    let mut out: Vec<Dirty> = tree
        .journals()
        .into_iter()
        .filter(|(_, j)| j.problems.is_empty() && is_dirty(j))
        .map(|(path, j)| {
            let built = j.summary().map(|s| s.built);
            Dirty {
                new_lines: j
                    .window
                    .iter()
                    .filter(|l| built.is_none_or(|b| l.at > b))
                    .count(),
                last_line: j.window.iter().map(|l| l.at).max(),
                path,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.path
            .names()
            .len()
            .cmp(&a.path.names().len())
            .then_with(|| a.path.cmp(&b.path))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{Meta, Sentence, Summary};
    use crate::ops::fixture::{p, zen};
    use crate::path::Place;
    use crate::scope::Section;
    use crate::testing::{at, note, slug};

    fn named(name: &str) -> Journal {
        Journal::new(
            Place::Named {
                section: Section::Topics,
                depth: 1,
            },
            Some(Meta::new(
                name,
                Some(format!("open for {name}")),
                at("2026-10-01T00:00:00Z"),
            )),
        )
    }

    fn summarised(mut j: Journal, text: &str) -> Journal {
        let over = fingerprint(&j);
        j.meta
            .get_or_insert_with(|| Meta::new("fixed", None, at("2026-10-01T00:00:00Z")))
            .summary = Some(Summary {
            over,
            built: at("2026-10-30T12:00:00Z"),
            lines: j.window.len(),
            children: j.live_count(),
            stale: false,
            digest: None,
            sentences: vec![Sentence {
                text: text.into(),
                cites: vec![],
            }],
        });
        j
    }

    #[test]
    fn an_empty_journal_is_never_dirty_and_a_new_line_makes_it_so() {
        let mut j = named("permits");
        assert!(!is_dirty(&j));
        j.window.push(note("2026-10-30T10:00:00Z", "first"));
        assert!(is_dirty(&j));
        let mut j = summarised(j, "First.");
        assert!(!is_dirty(&j));
        j.window.push(note("2026-10-30T13:00:00Z", "second"));
        assert!(is_dirty(&j));
    }

    #[test]
    fn an_empty_tree_is_clean_and_summaries_run_from_the_leaves_up() {
        let mut tree = zen();
        assert_eq!(dirty(&tree), vec![]);
        let topics = p("topics");
        tree.get_mut(&topics)
            .unwrap()
            .window
            .push(note("2026-10-30T10:00:00Z", "a cat"));
        let d: Vec<String> = dirty(&tree).iter().map(|d| d.path.show()).collect();
        assert_eq!(d, ["@zen/topics"], "its master waits for its summary");
        let j = tree.get_mut(&topics).unwrap();
        *j = summarised(j.clone(), "Drew a cat.");
        let d: Vec<String> = dirty(&tree).iter().map(|d| d.path.show()).collect();
        assert_eq!(d, ["@zen"]);
    }

    #[test]
    fn edits_forgets_and_acceptance_all_dirty_it() {
        let mut base = named("permits");
        base.window.push(note("2026-10-30T10:00:00Z", "first"));
        let base = summarised(base, "First.");
        let mut edited = base.clone();
        edited.window[0].text = "changed".into();
        assert!(is_dirty(&edited));
        let mut forgotten = base.clone();
        forgotten.window.clear();
        assert!(is_dirty(&forgotten));
        let mut proposed = base.clone();
        proposed.window[0].proposed = true;
        assert!(is_dirty(&proposed));
    }

    #[test]
    fn a_child_rebuilt_folded_or_added_dirties_its_parent() {
        let mut child = named("permits");
        child.window.push(note("2026-10-30T10:00:00Z", "x"));
        let child = summarised(child, "Permits are filed.");
        let mut parent = named("house-build");
        parent.children.insert(slug("permits"), child);
        let parent = summarised(parent, "House.");
        assert!(!is_dirty(&parent));
        let mut rebuilt = parent.clone();
        let c = rebuilt.children.get_mut(&slug("permits")).unwrap();
        c.meta.as_mut().unwrap().summary.as_mut().unwrap().sentences[0].text =
            "Permits approved.".into();
        assert!(is_dirty(&rebuilt));
        let mut folded = parent.clone();
        folded
            .children
            .get_mut(&slug("permits"))
            .unwrap()
            .meta
            .as_mut()
            .unwrap()
            .folded = true;
        assert!(is_dirty(&folded));
        let mut added = parent.clone();
        added.children.insert(slug("budget"), named("budget"));
        assert!(is_dirty(&added));
        // A change inside the child that leaves its summary alone does not reach
        // the parent: the parent reads the child's summary, not its lines.
        let mut inner = parent.clone();
        inner
            .children
            .get_mut(&slug("permits"))
            .unwrap()
            .window
            .push(note("2026-10-30T14:00:00Z", "y"));
        assert!(!is_dirty(&inner));
    }

    /// **An emptied child gives its parent nothing to say**: a child whose
    /// lines were all forgotten keeps a summary of no sentences, and a parent
    /// over it alone has no inputs.
    #[test]
    fn a_child_whose_summary_says_nothing_is_no_input() {
        let child = summarised(named("permits"), "x");
        let mut emptied = child.clone();
        emptied
            .meta
            .as_mut()
            .unwrap()
            .summary
            .as_mut()
            .unwrap()
            .sentences
            .clear();
        assert!(speaks(&child) && !speaks(&emptied));
        assert!(!speaks(&named("budget")), "no summary yet");
        let mut parent = named("house-build");
        parent.children.insert(slug("permits"), emptied);
        assert!(!has_inputs(&parent));
        parent.children.insert(slug("budget"), child);
        assert!(has_inputs(&parent));
    }

    #[test]
    fn the_fingerprint_separates_fields() {
        let mut a = named("a");
        a.window.push(note("2026-10-30T10:00:00Z", "ab"));
        let mut b = a.clone();
        b.window[0].text = "a".into();
        b.window[0].proposed = true;
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }
}
