//! A journal read back page by page (`journal_read {journal, after}`,
//! `docs/zen_memory_and_guides.md` Phase 3): the lines before a line,
//! newest last, going on from the window into the archives, so a journal's
//! whole history is read without a search. A page names where the next one
//! begins.

use crate::line::{Line, LineId};
use crate::path::JournalPath;
use crate::terms::{is_current, replaced};
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::room::{fitting, taken, CURSOR_ROOM};
use crate::view::who::render_line;
use crate::view::NOT_INSTRUCTIONS;

/// Every line of `j` a reader is shown at `now`, oldest first: its archives,
/// period by period, then its window — an archived line as hidden as a
/// window line by its own `expires`, or by a `replaces` anywhere in either.
fn history(j: &Journal, now: Stamp) -> Vec<&Line> {
    let mut lines: Vec<&Line> = j
        .archives
        .values()
        .flat_map(|period| period.iter())
        .collect();
    lines.sort_by(|a, b| (a.at, &a.id).cmp(&(b.at, &b.id)));
    lines.extend(j.window.iter());
    let replaced = replaced(lines.iter().copied());
    lines
        .into_iter()
        .filter(|l| !replaced.contains(&l.id) && is_current(l, now))
        .collect()
}

/// The `n` lines of `j` before `after` — fewer when `n` would not fit in an
/// answer of `fit` characters — newest last, and where the page before them
/// begins; `None` when `j` holds no line `after`.
pub fn before(
    tree: &Tree,
    path: &JournalPath,
    j: &Journal,
    after: &LineId,
    n: usize,
    fit: usize,
    now: Stamp,
) -> Option<String> {
    let lines = history(j, now);
    let at = lines.iter().position(|l| &l.id == after)?;
    let mut out = vec![NOT_INSTRUCTIONS.to_string()];
    if at == 0 {
        out.push(format!("{path} holds nothing older than #{after}."));
        return Some(out.join("\n"));
    }
    // Nearest first, so what fits is the lines just before `after`.
    let nearest: Vec<String> = lines[at.saturating_sub(n)..at]
        .iter()
        .rev()
        .map(|l| format!("  {}", render_line(tree, l)))
        .collect();
    let head = format!("{path}, the {} lines before #{after}, oldest first:", n);
    let room = fit.saturating_sub(taken(&out[0]) + taken(&head) + CURSOR_ROOM);
    let count = fitting(&nearest, room);
    let from = at - count;
    out.push(format!(
        "{path}, the {count} line{} before #{after}, oldest first:",
        if count == 1 { "" } else { "s" }
    ));
    out.extend(nearest[..count].iter().rev().cloned());
    let page = &lines[from..at];
    if from > 0 {
        out.push(format!(
            "{from} older — page on with `after: {path}#{}`.",
            page[0].id
        ));
    }
    Some(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, Op};
    use crate::period::Period;
    use crate::terms::Terms;
    use crate::testing::{at, note};

    /// `topics` holding notes `n0`…`n{count-1}`, a minute apart.
    fn noted(count: usize) -> (Tree, Vec<LineId>) {
        let mut t = zen();
        let ids = (0..count)
            .map(|i| {
                apply(
                    &mut t,
                    &john(),
                    &Op::Note {
                        path: p("topics"),
                        text: format!("n{i}"),
                        speaker: None,
                        terms: Terms::default(),
                    },
                    now().plus(60 * i as i64),
                )
                .unwrap()
                .line
                .unwrap()
            })
            .collect();
        (t, ids)
    }

    /// **A page is the lines before the one named, oldest first**, and names
    /// where the page before it begins; the oldest page names none.
    #[test]
    fn a_page_is_the_lines_before_the_one_named() {
        let (t, ids) = noted(5);
        let path = p("topics");
        let j = t.get(&path).unwrap();
        let page = before(&t, &path, j, &ids[4], 2, usize::MAX, now().plus(3600)).unwrap();
        let lines: Vec<&str> = page.lines().collect();
        assert_eq!(
            lines[1],
            format!("@zen/topics, the 2 lines before #{}, oldest first:", ids[4])
        );
        assert!(
            lines[2].ends_with("n2") && lines[3].ends_with("n3"),
            "{page}"
        );
        assert_eq!(
            lines[4],
            format!("2 older — page on with `after: @zen/topics#{}`.", ids[2])
        );
        let last = before(&t, &path, j, &ids[2], 2, usize::MAX, now().plus(3600)).unwrap();
        assert!(last.ends_with("n1") && !last.contains("older"), "{last}");
        let none = before(&t, &path, j, &ids[0], 2, usize::MAX, now().plus(3600)).unwrap();
        assert!(none.ends_with(&format!("holds nothing older than #{}.", ids[0])));
        let missing = LineId::parse("2026-01-01-0000-0000").unwrap();
        assert_eq!(
            before(&t, &path, j, &missing, 2, usize::MAX, now()),
            None
        );
    }

    /// **A page that would not fit its answer holds the lines nearest the one
    /// named that do**, and still names where the next page begins.
    #[test]
    fn a_page_is_cut_to_its_answer_and_keeps_its_cursor() {
        let mut t = zen();
        let ids: Vec<LineId> = (0..10)
            .map(|i| {
                apply(
                    &mut t,
                    &john(),
                    &Op::Note {
                        path: p("topics"),
                        text: format!("{i}{}", "x".repeat(1_900)),
                        speaker: None,
                        terms: Terms::default(),
                    },
                    now().plus(60 * i as i64),
                )
                .unwrap()
                .line
                .unwrap()
            })
            .collect();
        let path = p("topics");
        let j = t.get(&path).unwrap();
        let page = before(&t, &path, j, &ids[9], 9, 8_000, now().plus(3600)).unwrap();
        assert!(page.chars().count() <= 8_000, "{}", page.chars().count());
        let lines: Vec<&str> = page.lines().collect();
        assert_eq!(
            lines[1],
            format!("@zen/topics, the 3 lines before #{}, oldest first:", ids[9])
        );
        assert!(lines[2].contains(": 6x") && lines[4].contains(": 8x"));
        assert_eq!(
            lines[5],
            format!("6 older — page on with `after: @zen/topics#{}`.", ids[6])
        );
    }

    /// **An archived line is hidden as a window line is**: one past its
    /// expiry, and one a window line replaces, are not on any page.
    #[test]
    fn a_hidden_archived_line_is_on_no_page() {
        let (mut t, ids) = noted(1);
        let path = p("topics");
        let mut expired = note("2026-09-01T10:00:00Z", "the old reminder");
        expired.terms.expires = Some(at("2026-09-02T00:00:00Z"));
        let superseded = note("2026-09-01T11:00:00Z", "the key version is 2");
        let kept = note("2026-09-01T12:00:00Z", "the archived fact");
        let j = t.get_mut(&path).unwrap();
        j.archives.insert(
            Period::parse("2026-09-01").unwrap(),
            Arc::new(vec![expired.clone(), superseded.clone(), kept.clone()]),
        );
        j.window[0].terms.replaces = Some(superseded.id.clone());
        let j = t.get(&path).unwrap();
        let page = before(&t, &path, j, &ids[0], 10, usize::MAX, now()).unwrap();
        assert!(page.ends_with("the archived fact"), "{page}");
        assert!(!page.contains("the old reminder"), "{page}");
        assert!(!page.contains("the key version is 2"), "{page}");
        assert!(page.contains("the 1 line before"), "{page}");
    }
}
