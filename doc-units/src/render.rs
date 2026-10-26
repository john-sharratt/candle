//! How a document is shown: each unit under its anchor.
//!
//! ```text
//! Qm7x| The bay was warm when you walked in, the tank fluid already cycling…
//! b4_K| Then the order came down on the shared frequency…
//! r7Lm| impl KvCache > fn seal_chunk
//!     pub fn seal_chunk(&mut self, gid: ChunkGid) -> Result<()> {
//!         …
//!     }
//! ```
//!
//! A unit named by its own first line — a paragraph, a YAML line — shows that
//! line after its anchor and the rest beneath it. A code unit shows its scope
//! path after its anchor, then its lines exactly as they are, indentation and
//! all. Blank lines between units are not shown: they are separators, and an
//! edit keeps them.

use std::ops::Range;

use crate::Doc;

/// The most lines a page shows. Pages hold whole units, so a page ends at the
/// last unit that fits; a unit is never longer than this.
pub const PAGE_LINES: usize = 200;

/// How long a unit's one-line heading may run before it is cut.
const HEAD_CHARS: usize = 72;

/// Lines unit `i` takes to show: its own, and a heading line when it has a
/// label.
fn shown_lines(doc: &Doc, i: usize) -> usize {
    let u = &doc.units()[i];
    u.end - u.start + usize::from(u.label.is_some())
}

/// The document's pages: runs of whole units, each at most `budget` lines
/// shown. A document with no units has one empty page.
pub fn pages(doc: &Doc, budget: usize) -> Vec<Range<usize>> {
    let n = doc.units().len();
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start;
        let mut used = 0;
        while end < n && (end == start || used + shown_lines(doc, end) <= budget) {
            used += shown_lines(doc, end);
            end += 1;
        }
        out.push(start..end);
        start = end;
    }
    if out.is_empty() {
        out.push(0..0);
    }
    out
}

/// A line as shown: without its terminator.
fn bare(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

/// Unit `i`, whole, ending with a newline.
pub fn unit(doc: &Doc, i: usize) -> String {
    let u = &doc.units()[i];
    let lines = &doc.lines()[u.start..u.end];
    let mut out = String::new();
    let rest = match &u.label {
        Some(label) => {
            out.push_str(&format!("{}| {label}\n", u.anchor));
            lines
        }
        None => {
            out.push_str(&format!("{}| {}\n", u.anchor, bare(&lines[0])));
            &lines[1..]
        }
    };
    for l in rest {
        out.push_str(bare(l));
        out.push('\n');
    }
    out
}

/// Units `range`, whole.
pub fn units(doc: &Doc, range: Range<usize>) -> String {
    range.map(|i| unit(doc, i)).collect()
}

/// Unit `i` in one line — its anchor and its label, or its first line cut at
/// [`HEAD_CHARS`] — for showing what is beside an edit.
pub fn head(doc: &Doc, i: usize) -> String {
    let u = &doc.units()[i];
    let text = match &u.label {
        Some(label) => label.clone(),
        None => bare(&doc.lines()[u.start]).trim().to_string(),
    };
    let more = u.end - u.start > 1 || text.chars().count() > HEAD_CHARS;
    let cut: String = text.chars().take(HEAD_CHARS).collect();
    match more {
        true => format!("{}| {}…\n", u.anchor, cut.trim_end()),
        false => format!("{}| {cut}\n", u.anchor),
    }
}

/// What an edit wrote, whole, between the units on either side of it in one
/// line each — or, for a delete, the units now either side of where it was.
pub fn around(doc: &Doc, touched: &[usize], at_line: usize) -> String {
    let n = doc.units().len();
    let (first, last) = match (touched.first(), touched.last()) {
        (Some(&f), Some(&l)) => (Some(f), Some(l)),
        _ => {
            let after = doc.units().iter().position(|u| u.start >= at_line);
            let before = match after {
                Some(a) => a.checked_sub(1),
                None => n.checked_sub(1),
            };
            let mut out = String::new();
            if let Some(b) = before {
                out.push_str(&head(doc, b));
            }
            if let Some(a) = after {
                out.push_str(&head(doc, a));
            }
            return out;
        }
    };
    let (first, last) = (first.unwrap(), last.unwrap());
    let mut out = String::new();
    if let Some(b) = first.checked_sub(1) {
        out.push_str(&head(doc, b));
    }
    out.push_str(&units(doc, first..last + 1));
    if last + 1 < n {
        out.push_str(&head(doc, last + 1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_shows_each_paragraph_under_its_anchor() {
        let doc = Doc::new("a.md", "One.\nStill one.\n\nTwo.\n");
        let [a, b] = [&doc.units()[0].anchor, &doc.units()[1].anchor];
        assert_eq!(
            units(&doc, 0..2),
            format!("{a}| One.\nStill one.\n{b}| Two.\n")
        );
    }

    #[test]
    fn code_shows_its_path_then_its_lines() {
        let doc = Doc::new("a.rs", "use x;\n\nfn go() {\n    1\n}\n");
        let [a, b] = [&doc.units()[0].anchor, &doc.units()[1].anchor];
        assert_eq!(
            units(&doc, 0..2),
            format!("{a}| use x;\n{b}| fn go\nfn go() {{\n    1\n}}\n")
        );
    }

    #[test]
    fn crlf_is_not_shown() {
        let doc = Doc::new("a.md", "One.\r\nTwo.\r\n");
        assert_eq!(
            unit(&doc, 0),
            format!("{}| One.\nTwo.\n", doc.units()[0].anchor)
        );
    }

    #[test]
    fn pages_hold_whole_units() {
        let text: String = (0..30).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        let doc = Doc::new("a.md", &text);
        let got = pages(&doc, 7);
        assert_eq!(got[0], 0..3);
        assert_eq!(got.len(), 10);
        assert_eq!(got.last().unwrap().end, 30);
        assert_eq!(pages(&Doc::new("a.md", ""), 200), vec![0..0]);
    }

    #[test]
    fn a_head_is_one_line() {
        let long = "word ".repeat(40);
        let doc = Doc::new("a.md", &format!("Short.\n\n{long}\n\nTwo\nlines\n"));
        let a = |i: usize| doc.units()[i].anchor.clone();
        assert_eq!(head(&doc, 0), format!("{}| Short.\n", a(0)));
        assert!(head(&doc, 1).ends_with("…\n"));
        assert_eq!(head(&doc, 2), format!("{}| Two…\n", a(2)));
    }

    #[test]
    fn around_shows_the_edit_between_its_neighbours() {
        let doc = Doc::new("a.md", "One.\n\nTwo.\nmore\n\nThree.\n");
        let a = |i: usize| doc.units()[i].anchor.clone();
        assert_eq!(
            around(&doc, &[1], 2),
            format!("{}| One.\n{}| Two.\nmore\n{}| Three.\n", a(0), a(1), a(2))
        );
        // A delete at line 2 shows what is either side of it now.
        assert_eq!(
            around(&doc, &[], 2),
            format!("{}| One.\n{}| Two.…\n", a(0), a(1))
        );
    }
}
