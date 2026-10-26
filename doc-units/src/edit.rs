//! One anchored edit, applied to a document — `docs/npcd_anchored_edits.md` §6.
//!
//! An edit names a unit by its anchor (`at`), optionally the last unit of a
//! span (`to`), and says what to do there. **An anchor that names no unit of
//! the document as it stands is refused** ([`EditError::Stale`]): an anchor is
//! its unit's text, hashed, so a unit somebody else changed no longer has the
//! anchor it was read under, and an edit is never applied to text its author
//! did not see.
//!
//! Blank lines between units are separators, and an edit keeps them: an
//! insert brings a blank line where its neighbours are set apart by one (and
//! always, in prose, where paragraphs are), a delete takes the separator with
//! the unit so no double gap is left behind.

use std::fmt;

use crate::Doc;

/// Where an insert or a move puts its text, beside the unit it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Before,
    After,
}

impl Place {
    fn word(self) -> &'static str {
        match self {
            Place::Before => "before",
            Place::After => "after",
        }
    }
}

/// What an edit does at its anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Replace the unit (through `to`, when given) with `text` — or, with
    /// `old`, replace that substring of the unit, which must occur in it
    /// exactly once.
    Replace {
        text: String,
        old: Option<String>,
        to: Option<String>,
    },
    /// Add `text` as a new unit beside the anchored one.
    Insert { place: Place, text: String },
    /// Remove the unit (through `to`, when given).
    Delete { to: Option<String> },
    /// Take the unit (through `to`) and put it beside `dest`, its text
    /// unchanged.
    Move {
        to: Option<String>,
        dest: String,
        place: Place,
    },
}

/// An edit: the anchor it is made at and what it does there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub at: String,
    pub op: Op,
}

/// Why an edit was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    /// An anchor names no unit of the document as it stands: the unit changed
    /// since it was read, or the anchor was never one of its units.
    Stale(String),
    /// The edit is not one that can be made: an empty text, a span that runs
    /// backwards, a move into its own span.
    Invalid(String),
    /// A scoped `old` does not occur in its unit.
    Unmatched(String),
    /// A scoped `old` occurs in its unit more than once.
    Ambiguous(String),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::Stale(a) => write!(
                f,
                "{a} is not a part of it as it stands now — it has changed since you read it, \
                 or was never one of its anchors; read it again and edit by the anchors it \
                 shows"
            ),
            EditError::Invalid(why) | EditError::Unmatched(why) | EditError::Ambiguous(why) => {
                f.write_str(why)
            }
        }
    }
}

impl std::error::Error for EditError {}

/// An edit made.
#[derive(Debug, Clone)]
pub struct Applied {
    /// The document after the edit.
    pub doc: Doc,
    /// The units the edit wrote, by index into [`Applied::doc`] — empty for a
    /// delete.
    pub touched: Vec<usize>,
    /// The line (from 0) the edit happened at in [`Applied::doc`], for
    /// showing what is around a delete.
    pub at_line: usize,
    /// What was done, in a sentence: `Replaced 9fTa.`
    pub summary: String,
    /// Whether the document parsed as its language before the edit and does
    /// not after it.
    pub broke: bool,
}

/// Apply `edit` to `doc`.
pub fn apply(doc: &Doc, edit: &Edit) -> Result<Applied, EditError> {
    let at = &edit.at;
    let (lines, region, summary) = match &edit.op {
        Op::Replace {
            text,
            old: Some(old),
            to,
        } => {
            if to.is_some() {
                return Err(EditError::Invalid(
                    "`old` changes words inside one part; it takes no `to`".into(),
                ));
            }
            let i = find(doc, at)?;
            let (lines, region) = replace_within(doc, i, old, text)?;
            (lines, region, format!("Changed {at}."))
        }
        Op::Replace {
            text,
            old: None,
            to,
        } => {
            let (a, b) = span(doc, at, to.as_deref())?;
            let new = text_lines(text, eol(doc))
                .ok_or_else(|| empty("replace", "to take a part out, delete it"))?;
            let (start, end) = (doc.units()[a].start, doc.units()[b].end);
            let n = new.len();
            (
                splice(doc.lines(), start, end, new),
                (start, start + n),
                format!("Replaced {}.", span_name(at, to.as_deref())),
            )
        }
        Op::Insert { place, text } => {
            let i = find(doc, at)?;
            let new = text_lines(text, eol(doc)).ok_or_else(|| empty("insert", ""))?;
            let (lines, region) = insert(doc, i, *place, new);
            (lines, region, format!("Inserted {} {at}.", place.word()))
        }
        Op::Delete { to } => {
            let (a, b) = span(doc, at, to.as_deref())?;
            let (lines, line) = delete(doc, a, b);
            (
                lines,
                (line, line),
                format!("Deleted {}.", span_name(at, to.as_deref())),
            )
        }
        Op::Move { to, dest, place } => {
            let (a, b) = span(doc, at, to.as_deref())?;
            let d = find(doc, dest)?;
            if (a..=b).contains(&d) {
                return Err(EditError::Invalid(format!(
                    "{dest} is part of what is being moved; move it beside a part outside it"
                )));
            }
            let moved: Vec<String> = with_terminators(
                doc.lines()[doc.units()[a].start..doc.units()[b].end].to_vec(),
                eol(doc),
            );
            let (rest, _) = delete(doc, a, b);
            let rest = Doc::of(doc.language(), &rest.concat());
            // Taking the span out re-seeds only copies of its own text, so
            // the destination keeps its anchor unless it is one of those.
            let d = rest
                .find(dest)
                .ok_or_else(|| EditError::Stale(dest.clone()))?;
            let (lines, region) = insert(&rest, d, *place, moved);
            (
                lines,
                region,
                format!(
                    "Moved {} {} {dest}.",
                    span_name(at, to.as_deref()),
                    place.word()
                ),
            )
        }
    };

    let after = Doc::of(doc.language(), &lines.concat());
    let (start, end) = region;
    let touched = (0..after.units().len())
        .filter(|&i| {
            let u = &after.units()[i];
            u.start < end && u.end > start
        })
        .collect();
    let broke = doc.parses() == Some(true) && after.parses() == Some(false);
    Ok(Applied {
        doc: after,
        touched,
        at_line: start,
        summary,
        broke,
    })
}

fn find(doc: &Doc, anchor: &str) -> Result<usize, EditError> {
    doc.find(anchor)
        .ok_or_else(|| EditError::Stale(anchor.to_string()))
}

/// The units from `at` through `to` (or `at` alone), as indices.
fn span(doc: &Doc, at: &str, to: Option<&str>) -> Result<(usize, usize), EditError> {
    let a = find(doc, at)?;
    let b = match to {
        Some(to) if to != at => find(doc, to)?,
        _ => a,
    };
    if b < a {
        return Err(EditError::Invalid(format!(
            "{} comes before {at}: a span runs from `at` forward to `to`",
            to.unwrap_or_default()
        )));
    }
    Ok((a, b))
}

fn span_name(at: &str, to: Option<&str>) -> String {
    match to {
        Some(to) if to != at => format!("{at} through {to}"),
        _ => at.to_string(),
    }
}

fn empty(what: &str, instead: &str) -> EditError {
    let instead = match instead.is_empty() {
        true => String::new(),
        false => format!("; {instead}"),
    };
    EditError::Invalid(format!("there is no text to {what}{instead}"))
}

/// The document's line ending: `\r\n` when its first line ends so.
fn eol(doc: &Doc) -> &'static str {
    match doc.lines().first() {
        Some(l) if l.ends_with("\r\n") => "\r\n",
        _ => "\n",
    }
}

/// `text` as lines, each ending `eol`, with leading and trailing blank lines
/// dropped. `None` when nothing is left.
fn text_lines(text: &str, eol: &str) -> Option<Vec<String>> {
    let text = text.replace("\r\n", "\n");
    let all: Vec<&str> = text.split('\n').collect();
    let first = all.iter().position(|l| !l.trim().is_empty())?;
    let last = all.iter().rposition(|l| !l.trim().is_empty())?;
    Some(
        all[first..=last]
            .iter()
            .map(|l| format!("{l}{eol}"))
            .collect(),
    )
}

/// `lines` with every one ending `eol` — the last of a document may not.
fn with_terminators(mut lines: Vec<String>, eol: &str) -> Vec<String> {
    for l in &mut lines {
        if !l.ends_with('\n') {
            l.push_str(eol);
        }
    }
    lines
}

/// `lines[start..end]` replaced by `new`, keeping the document's own
/// trailing-newline state: a file that ended without one still does.
fn splice(lines: &[String], start: usize, end: usize, new: Vec<String>) -> Vec<String> {
    let eol = match lines.first() {
        Some(l) if l.ends_with("\r\n") => "\r\n",
        _ => "\n",
    };
    let ended_bare = lines.last().is_some_and(|l| !l.ends_with('\n'));
    let mut out: Vec<String> = Vec::with_capacity(lines.len() + new.len());
    out.extend_from_slice(&lines[..start]);
    out.extend(new);
    out.extend_from_slice(&lines[end..]);
    let mut out = with_terminators(out, eol);
    if ended_bare {
        if let Some(last) = out.last_mut() {
            let trimmed = last.trim_end_matches(['\r', '\n']).len();
            last.truncate(trimmed);
        }
    }
    out
}

/// Blank lines between unit `i` and the one after it — `None` for the last.
fn gap_after(doc: &Doc, i: usize) -> Option<usize> {
    let u = doc.units();
    u.get(i + 1).map(|next| next.start - u[i].end)
}

/// Blank lines between unit `i` and the one before it — `None` for the first.
fn gap_before(doc: &Doc, i: usize) -> Option<usize> {
    let u = doc.units();
    i.checked_sub(1).map(|p| u[i].start - u[p].end)
}

/// `new` inserted beside unit `i`, set apart by a blank line where the
/// document sets its units apart — always, in prose. Returns the lines and the
/// inserted range.
fn insert(doc: &Doc, i: usize, place: Place, new: Vec<String>) -> (Vec<String>, (usize, usize)) {
    let eol = eol(doc);
    let gap = match place {
        Place::Before => gap_before(doc, i).or(gap_after(doc, i)),
        Place::After => gap_after(doc, i).or(gap_before(doc, i)),
    };
    let apart = doc.language().is_prose() || gap.unwrap_or(0) > 0;
    let n = new.len();
    let u = &doc.units()[i];
    match place {
        Place::After => {
            let mut block = Vec::with_capacity(n + 1);
            let mut first = u.end;
            if apart {
                block.push(eol.to_string());
                first += 1;
            }
            block.extend(new);
            (splice(doc.lines(), u.end, u.end, block), (first, first + n))
        }
        Place::Before => {
            let mut block = new;
            if apart {
                block.push(eol.to_string());
            }
            (
                splice(doc.lines(), u.start, u.start, block),
                (u.start, u.start + n),
            )
        }
    }
}

/// Units `a..=b` taken out, with the separator after them (or, for the last
/// units, before them) so no double gap is left. Returns the lines and the
/// line the removal happened at.
fn delete(doc: &Doc, a: usize, b: usize) -> (Vec<String>, usize) {
    let u = doc.units();
    let (start, end) = match (u.get(b + 1), a.checked_sub(1)) {
        (Some(next), _) => (u[a].start, next.start),
        (None, Some(prev)) => (u[prev].end, u[b].end),
        (None, None) => (u[a].start, u[b].end),
    };
    (splice(doc.lines(), start, end, Vec::new()), start)
}

/// Unit `i` with the one occurrence of `old` in it replaced by `new`.
fn replace_within(
    doc: &Doc,
    i: usize,
    old: &str,
    new: &str,
) -> Result<(Vec<String>, (usize, usize)), EditError> {
    let at = &doc.units()[i].anchor;
    if old.is_empty() {
        return Err(EditError::Invalid(
            "`old` is empty — name the words to change, or leave `old` out to replace the whole \
             part"
                .into(),
        ));
    }
    if old == new {
        return Err(EditError::Invalid(
            "`old` and `text` are the same, so nothing would change".into(),
        ));
    }
    let unit = doc.unit_text(i);
    let crlf = eol(doc) == "\r\n";
    let (old, new) = match crlf {
        true => (
            old.replace("\r\n", "\n").replace('\n', "\r\n"),
            new.replace("\r\n", "\n").replace('\n', "\r\n"),
        ),
        false => (old.to_string(), new.to_string()),
    };
    match unit.matches(old.as_str()).count() {
        0 => {
            return Err(EditError::Unmatched(format!(
                "`old` does not occur in {at}, which reads:\n{}",
                unit.replace('\r', "")
            )))
        }
        1 => {}
        n => {
            return Err(EditError::Ambiguous(format!(
                "`old` occurs {n} times in {at}; give more of the words around the one to change"
            )))
        }
    }
    let changed = unit.replacen(old.as_str(), new.as_str(), 1);
    let u = &doc.units()[i];
    let last_eol = match doc.lines()[u.end - 1].ends_with('\n') {
        true => eol(doc),
        false => "",
    };
    let mut new_lines: Vec<String> = changed.split_inclusive('\n').map(str::to_string).collect();
    if let Some(last) = new_lines.last_mut() {
        if !last.ends_with('\n') {
            last.push_str(last_eol);
        }
    }
    let n = new_lines.len();
    Ok((
        splice(doc.lines(), u.start, u.end, new_lines),
        (u.start, u.start + n),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STORY: &str = "The bay was warm.\n\nThe order came down.\n\nYou said fourteen.\n";

    fn doc(text: &str) -> Doc {
        Doc::new("story.md", text)
    }

    fn a(d: &Doc, i: usize) -> String {
        d.units()[i].anchor.clone()
    }

    fn replace(at: &str, text: &str) -> Edit {
        Edit {
            at: at.into(),
            op: Op::Replace {
                text: text.into(),
                old: None,
                to: None,
            },
        }
    }

    #[test]
    fn replace_writes_the_unit_and_reports_its_new_anchor() {
        let d = doc(STORY);
        let done = apply(&d, &replace(&a(&d, 2), "You said fifteen.")).unwrap();
        assert_eq!(
            done.doc.text(),
            "The bay was warm.\n\nThe order came down.\n\nYou said fifteen.\n"
        );
        assert_eq!(done.touched, vec![2]);
        assert_eq!(done.summary, format!("Replaced {}.", a(&d, 2)));
        assert_ne!(a(&done.doc, 2), a(&d, 2));
        // The other units keep their anchors.
        assert_eq!(a(&done.doc, 0), a(&d, 0));
        assert_eq!(a(&done.doc, 1), a(&d, 1));
    }

    #[test]
    fn a_scoped_old_changes_only_those_words() {
        let d = doc(STORY);
        let edit = Edit {
            at: a(&d, 2),
            op: Op::Replace {
                text: "fifteen".into(),
                old: Some("fourteen".into()),
                to: None,
            },
        };
        let done = apply(&d, &edit).unwrap();
        assert!(done.doc.text().ends_with("You said fifteen.\n"));
        assert_eq!(done.summary, format!("Changed {}.", a(&d, 2)));
    }

    #[test]
    fn a_scoped_old_must_occur_once_in_the_unit() {
        let d = doc("a b a\n\nb\n");
        let edit = |old: &str| Edit {
            at: a(&d, 0),
            op: Op::Replace {
                text: "z".into(),
                old: Some(old.into()),
                to: None,
            },
        };
        assert!(matches!(
            apply(&d, &edit("a")),
            Err(EditError::Ambiguous(_))
        ));
        assert!(matches!(
            apply(&d, &edit("q")),
            Err(EditError::Unmatched(_))
        ));
        // `b` occurs in the document twice, in this unit once.
        assert_eq!(apply(&d, &edit("b")).unwrap().doc.text(), "a z a\n\nb\n");
    }

    #[test]
    fn insert_after_keeps_paragraphs_apart() {
        let d = doc(STORY);
        let edit = Edit {
            at: a(&d, 2),
            op: Op::Insert {
                place: Place::After,
                text: "The tank sighed.".into(),
            },
        };
        let done = apply(&d, &edit).unwrap();
        assert_eq!(done.doc.text(), format!("{STORY}\nThe tank sighed.\n"));
        assert_eq!(done.touched, vec![3]);
        let edit = Edit {
            at: a(&d, 0),
            op: Op::Insert {
                place: Place::Before,
                text: "First.".into(),
            },
        };
        assert_eq!(
            apply(&d, &edit).unwrap().doc.text(),
            format!("First.\n\n{STORY}")
        );
    }

    #[test]
    fn delete_takes_its_separator_with_it() {
        let d = doc(STORY);
        let del = |i: usize| Edit {
            at: a(&d, i),
            op: Op::Delete { to: None },
        };
        assert_eq!(
            apply(&d, &del(1)).unwrap().doc.text(),
            "The bay was warm.\n\nYou said fourteen.\n"
        );
        assert_eq!(
            apply(&d, &del(0)).unwrap().doc.text(),
            "The order came down.\n\nYou said fourteen.\n"
        );
        assert_eq!(
            apply(&d, &del(2)).unwrap().doc.text(),
            "The bay was warm.\n\nThe order came down.\n"
        );
        assert!(apply(&d, &del(1)).unwrap().touched.is_empty());
    }

    #[test]
    fn a_span_replaces_or_deletes_several_units() {
        let d = doc(STORY);
        let edit = Edit {
            at: a(&d, 0),
            op: Op::Replace {
                text: "All of it, shorter.".into(),
                old: None,
                to: Some(a(&d, 1)),
            },
        };
        assert_eq!(
            apply(&d, &edit).unwrap().doc.text(),
            "All of it, shorter.\n\nYou said fourteen.\n"
        );
        let edit = Edit {
            at: a(&d, 1),
            op: Op::Delete { to: Some(a(&d, 2)) },
        };
        assert_eq!(apply(&d, &edit).unwrap().doc.text(), "The bay was warm.\n");
        let backwards = Edit {
            at: a(&d, 2),
            op: Op::Delete { to: Some(a(&d, 0)) },
        };
        assert!(matches!(apply(&d, &backwards), Err(EditError::Invalid(_))));
    }

    #[test]
    fn move_keeps_the_text_and_its_anchor() {
        let d = doc(STORY);
        let edit = Edit {
            at: a(&d, 0),
            op: Op::Move {
                to: None,
                dest: a(&d, 2),
                place: Place::After,
            },
        };
        let done = apply(&d, &edit).unwrap();
        assert_eq!(
            done.doc.text(),
            "The order came down.\n\nYou said fourteen.\n\nThe bay was warm.\n"
        );
        assert_eq!(a(&done.doc, 2), a(&d, 0));
        assert_eq!(done.touched, vec![2]);
        let into_itself = Edit {
            at: a(&d, 0),
            op: Op::Move {
                to: Some(a(&d, 1)),
                dest: a(&d, 1),
                place: Place::After,
            },
        };
        assert!(matches!(
            apply(&d, &into_itself),
            Err(EditError::Invalid(_))
        ));
    }

    /// An anchor read before somebody changed its unit is refused, and a
    /// unit changed elsewhere does not disturb it.
    #[test]
    fn a_changed_unit_refuses_its_old_anchor() {
        let read = doc(STORY);
        let stale = a(&read, 2);
        let kept = a(&read, 0);
        let now = doc("The bay was warm.\n\nThe order came down.\n\nYou said twelve.\n");
        assert_eq!(
            apply(&now, &replace(&stale, "x")).unwrap_err(),
            EditError::Stale(stale.clone())
        );
        assert!(apply(&now, &replace(&kept, "The bay was cold.")).is_ok());
    }

    /// Deleting the first of two identical paragraphs moves the second onto
    /// the first's anchor; the second's old anchor is refused, never applied
    /// to the wrong copy.
    #[test]
    fn duplicates_never_take_an_edit_meant_for_the_other() {
        let d = doc("No.\n\nYes.\n\nNo.\n");
        let (first, second) = (a(&d, 0), a(&d, 2));
        let done = apply(
            &d,
            &Edit {
                at: first.clone(),
                op: Op::Delete { to: None },
            },
        )
        .unwrap();
        assert_eq!(done.doc.text(), "Yes.\n\nNo.\n");
        assert_eq!(a(&done.doc, 1), first);
        assert_eq!(
            apply(&done.doc, &replace(&second, "Maybe.")).unwrap_err(),
            EditError::Stale(second)
        );
    }

    #[test]
    fn code_inserts_follow_the_files_spacing() {
        let src = "use a;\nuse b;\n\nfn one() {}\n\nfn two() {}\n";
        let d = Doc::new("x.rs", src);
        let one = d
            .units()
            .iter()
            .position(|u| u.label.as_deref() == Some("fn one"))
            .unwrap();
        let after_one = Edit {
            at: a(&d, one),
            op: Op::Insert {
                place: Place::After,
                text: "fn mid() {}".into(),
            },
        };
        let done = apply(&d, &after_one).unwrap();
        assert_eq!(
            done.doc.text(),
            "use a;\nuse b;\n\nfn one() {}\n\nfn mid() {}\n\nfn two() {}\n"
        );
        assert_eq!(
            done.doc.units()[done.touched[0]].label.as_deref(),
            Some("fn mid")
        );
    }

    #[test]
    fn an_edit_that_breaks_the_parse_applies_and_says_so() {
        let d = Doc::new("x.rs", "fn one() {}\n\nfn two() {}\n");
        let done = apply(&d, &replace(&a(&d, 0), "fn one() {")).unwrap();
        assert!(done.broke);
        assert_eq!(done.doc.text(), "fn one() {\n\nfn two() {}\n");
        let fine = apply(&d, &replace(&a(&d, 0), "fn uno() {}")).unwrap();
        assert!(!fine.broke);
    }

    #[test]
    fn crlf_documents_stay_crlf() {
        let d = doc("One.\r\n\r\nTwo.\r\n");
        let done = apply(
            &d,
            &Edit {
                at: a(&d, 1),
                op: Op::Insert {
                    place: Place::After,
                    text: "Three.\nStill three.".into(),
                },
            },
        )
        .unwrap();
        assert_eq!(
            done.doc.text(),
            "One.\r\n\r\nTwo.\r\n\r\nThree.\r\nStill three.\r\n"
        );
    }

    #[test]
    fn a_file_without_a_final_newline_keeps_it_that_way() {
        let d = doc("One.\n\nTwo.");
        let done = apply(&d, &replace(&a(&d, 1), "Deux.")).unwrap();
        assert_eq!(done.doc.text(), "One.\n\nDeux.");
        let done = apply(
            &d,
            &Edit {
                at: a(&d, 1),
                op: Op::Insert {
                    place: Place::After,
                    text: "Three.".into(),
                },
            },
        )
        .unwrap();
        assert_eq!(done.doc.text(), "One.\n\nTwo.\n\nThree.");
    }

    #[test]
    fn empty_text_is_refused() {
        let d = doc(STORY);
        assert!(matches!(
            apply(&d, &replace(&a(&d, 0), "  \n\n")),
            Err(EditError::Invalid(_))
        ));
    }
}
