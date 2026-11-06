//! How a document is shown: each part a fenced block of its own.
//!
//! ````text
//! ```rust file=candle/src/cache.rs part=r7Lm lines=120-122 scope="impl KvCache > fn seal_chunk"
//!  +1     pub fn seal_chunk(&mut self, gid: ChunkGid) -> Result<()> {
//!  +2         self.sealed.push(gid)
//!  +3     }
//! ```
//! ````
//!
//! The fence names the file, the part's anchor, the lines of the file it holds
//! and its scope: a code item's path, or else its own first line. Every line of
//! the part follows behind its number in the part, from `+1`, so an edit names
//! lines by the part and that number and never counts. The number also keeps
//! the content off the start of the line: a line of the file that holds
//! ```` ``` ```` cannot close the block, and a line shaped like a fence's
//! attributes reads as content. Blank lines between parts are not shown: they
//! are separators, and an edit keeps them.

use std::ops::Range;

use crate::Doc;

/// The most lines a page shows. Pages hold whole parts, so a page ends at the
/// last part that fits; a part is never longer than this.
pub const PAGE_LINES: usize = 200;

/// How long a scope taken from a part's first line may run before it is cut.
const SCOPE_CHARS: usize = 72;

/// Where a document's blocks say they are from: the file as the fence names it
/// (`candle/src/cache.rs`) and the fence's language tag (`rust`; empty reads
/// `text`).
#[derive(Debug, Clone, Copy)]
pub struct Source<'a> {
    pub file: &'a str,
    pub tag: &'a str,
}

/// The lines part `i` holds.
fn part_lines(doc: &Doc, i: usize) -> usize {
    let u = &doc.units()[i];
    u.end - u.start
}

/// The document's pages: runs of whole parts, each at most `budget` lines. A
/// document with no parts has one empty page.
pub fn pages(doc: &Doc, budget: usize) -> Vec<Range<usize>> {
    let n = doc.units().len();
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start;
        let mut used = 0;
        while end < n && (end == start || used + part_lines(doc, end) <= budget) {
            used += part_lines(doc, end);
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

/// The lines of the file (from 1, inclusive) each of `pages` covers. They tile
/// it: the first starts at line 1, each runs to the line before the next one's
/// first part, and the last ends at the file's last line — so the blank lines
/// between parts belong to a page too, and a run of reads naming these ranges
/// reads every line.
pub fn page_lines(doc: &Doc, pages: &[Range<usize>]) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = pages
        .iter()
        .enumerate()
        .map(|(k, p)| match (k, doc.units().get(p.start)) {
            (0, _) | (_, None) => 1,
            (_, Some(u)) => u.start + 1,
        })
        .collect();
    let last = doc.line_count().max(1);
    (0..pages.len())
        .map(|k| {
            let end = starts.get(k + 1).map_or(last, |next| next - 1);
            (starts[k], end.max(starts[k]))
        })
        .collect()
}

/// A line as shown: without its terminator.
fn bare(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

/// Part `i`'s scope: its code path (`impl Act > fn summary`), or else its own
/// first line, cut at [`SCOPE_CHARS`] and marked `…` when more follows on it.
pub fn scope(doc: &Doc, i: usize) -> String {
    let u = &doc.units()[i];
    if let Some(label) = &u.label {
        return label.clone();
    }
    let first = bare(&doc.lines()[u.start]).trim();
    match first.chars().count() > SCOPE_CHARS {
        true => {
            let cut: String = first.chars().take(SCOPE_CHARS).collect();
            format!("{}…", cut.trim_end())
        }
        false => first.to_string(),
    }
}

/// The lines of the file part `i` holds, from 1: `46-52`, or `46` for one.
pub fn lines_of(doc: &Doc, i: usize) -> String {
    let u = &doc.units()[i];
    match u.end - u.start {
        1 => format!("{}", u.start + 1),
        _ => format!("{}-{}", u.start + 1, u.end),
    }
}

/// `text` as a fence attribute's quoted value.
fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Line `n` of a part (from 1) as the gutter shows it: `+n`, right-aligned to
/// the width of a part's longest number.
pub fn gutter(n: usize) -> String {
    format!("{:>3}", format!("+{n}"))
}

/// Part `i` as a fenced block, ending with a newline.
pub fn block(doc: &Doc, i: usize, source: Source<'_>) -> String {
    let u = &doc.units()[i];
    let tag = match source.tag {
        "" => "text",
        tag => tag,
    };
    // A file named with a space would otherwise read as two attributes.
    let file = match source.file.contains(char::is_whitespace) {
        true => quoted(source.file),
        false => source.file.to_string(),
    };
    let mut out = format!(
        "```{tag} file={file} part={} lines={} scope={}\n",
        u.anchor,
        lines_of(doc, i),
        quoted(&scope(doc, i)),
    );
    for (k, line) in doc.lines()[u.start..u.end].iter().enumerate() {
        let line = bare(line);
        out.push_str(&gutter(k + 1));
        if !line.is_empty() {
            out.push(' ');
            out.push_str(line);
        }
        out.push('\n');
    }
    out.push_str("```\n");
    out
}

/// Parts `range`, each a block.
pub fn blocks(doc: &Doc, range: Range<usize>, source: Source<'_>) -> String {
    range.map(|i| block(doc, i, source)).collect()
}

/// Part `i` in one line — `part=KehT lines=55-101 impl Act > fn summary` — as a
/// list of a file shows it, and as an edit's answer shows what is beside it.
pub fn entry(doc: &Doc, i: usize) -> String {
    format!(
        "part={} lines={} {}\n",
        doc.units()[i].anchor,
        lines_of(doc, i),
        scope(doc, i)
    )
}

/// What an edit wrote, each part a block, between the parts on either side of
/// it in one line each — or, for a delete, the parts now either side of where
/// it was.
pub fn around(doc: &Doc, touched: &[usize], at_line: usize, source: Source<'_>) -> String {
    let n = doc.units().len();
    let (first, last) = match (touched.first(), touched.last()) {
        (Some(&f), Some(&l)) => (f, l),
        _ => {
            let after = doc.units().iter().position(|u| u.start >= at_line);
            let before = match after {
                Some(a) => a.checked_sub(1),
                None => n.checked_sub(1),
            };
            return before.into_iter().chain(after).map(|i| entry(doc, i)).collect();
        }
    };
    let mut out = String::new();
    if let Some(b) = first.checked_sub(1) {
        out.push_str(&entry(doc, b));
    }
    out.push_str(&blocks(doc, first..last + 1, source));
    if last + 1 < n {
        out.push_str(&entry(doc, last + 1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MD: Source<'static> = Source {
        file: "r/a.md",
        tag: "markdown",
    };
    const RS: Source<'static> = Source {
        file: "r/a.rs",
        tag: "rust",
    };

    #[test]
    fn prose_shows_each_paragraph_as_its_own_block() {
        let doc = Doc::new("a.md", "One.\nStill one.\n\nTwo.\n");
        let [a, b] = [&doc.units()[0].anchor, &doc.units()[1].anchor];
        assert_eq!(
            blocks(&doc, 0..2, MD),
            format!(
                "```markdown file=r/a.md part={a} lines=1-2 scope=\"One.\"\n +1 One.\n +2 Still one.\n```\n\
                 ```markdown file=r/a.md part={b} lines=4 scope=\"Two.\"\n +1 Two.\n```\n"
            )
        );
    }

    #[test]
    fn code_shows_its_scope_then_its_lines_behind_their_numbers() {
        let doc = Doc::new("a.rs", "use x;\n\nfn go() {\n\n    1\n}\n");
        let [a, b] = [&doc.units()[0].anchor, &doc.units()[1].anchor];
        assert_eq!(
            blocks(&doc, 0..2, RS),
            format!(
                "```rust file=r/a.rs part={a} lines=1 scope=\"use x;\"\n +1 use x;\n```\n\
                 ```rust file=r/a.rs part={b} lines=3-6 scope=\"fn go\"\n +1 fn go() {{\n +2\n +3     1\n +4 }}\n```\n"
            )
        );
    }

    /// **Content cannot close a block or pass for a fence**: every line sits
    /// behind its number, and a quote in a scope is escaped.
    #[test]
    fn a_fence_or_a_quote_in_the_content_stays_content() {
        let doc = Doc::new("a.md", "```text\nr7Lm| x\n```\n\nSay \"hi\".\n");
        let [a, b] = [&doc.units()[0].anchor, &doc.units()[1].anchor];
        assert_eq!(
            blocks(&doc, 0..2, MD),
            format!(
                "```markdown file=r/a.md part={a} lines=1-3 scope=\"```text\"\n +1 ```text\n +2 r7Lm| x\n +3 ```\n```\n\
                 ```markdown file=r/a.md part={b} lines=5 scope=\"Say \\\"hi\\\".\"\n +1 Say \"hi\".\n```\n"
            )
        );
    }

    #[test]
    fn crlf_is_not_shown_and_an_empty_tag_reads_text() {
        let doc = Doc::new("a.txt", "One.\r\nTwo.\r\n");
        assert_eq!(
            block(&doc, 0, Source { file: "r/a.txt", tag: "" }),
            format!(
                "```text file=r/a.txt part={} lines=1-2 scope=\"One.\"\n +1 One.\n +2 Two.\n```\n",
                doc.units()[0].anchor
            )
        );
    }

    /// A file named with a space is quoted, so it stays one attribute.
    #[test]
    fn a_file_with_a_space_is_quoted() {
        let doc = Doc::new("The Body.md", "You said fourteen.\n");
        let source = Source {
            file: "layers/bram/The Body.md",
            tag: "markdown",
        };
        assert_eq!(
            block(&doc, 0, source),
            format!(
                "```markdown file=\"layers/bram/The Body.md\" part={} lines=1 scope=\"You said \
                 fourteen.\"\n +1 You said fourteen.\n```\n",
                doc.units()[0].anchor
            )
        );
    }

    #[test]
    fn the_gutter_is_right_aligned_to_two_digits() {
        assert_eq!(gutter(1), " +1");
        assert_eq!(gutter(80), "+80");
    }

    #[test]
    fn pages_hold_whole_parts() {
        let text: String = (0..30).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        let doc = Doc::new("a.md", &text);
        let got = pages(&doc, 6);
        assert_eq!(got[0], 0..3);
        assert_eq!(got.len(), 10);
        assert_eq!(got.last().unwrap().end, 30);
        assert_eq!(pages(&Doc::new("a.md", ""), 200), vec![0..0]);
    }

    /// **Pages' lines tile the file**: the blank lines between parts belong to
    /// the page before the next part, and the last page runs to the end.
    #[test]
    fn page_lines_tile_the_file() {
        let doc = Doc::new("a.md", "\none\n\ntwo\nthree\n\nfour\n\n");
        let all = pages(&doc, 2);
        assert_eq!(all, vec![0..1, 1..2, 2..3]);
        assert_eq!(page_lines(&doc, &all), vec![(1, 3), (4, 6), (7, 8)]);
        let empty = Doc::new("a.md", "");
        assert_eq!(page_lines(&empty, &pages(&empty, 200)), vec![(1, 1)]);
    }

    #[test]
    fn an_entry_is_one_line() {
        let long = "word ".repeat(40);
        let doc = Doc::new("a.md", &format!("Short.\n\n{long}\n\nTwo\nlines\n"));
        let a = |i: usize| doc.units()[i].anchor.clone();
        assert_eq!(entry(&doc, 0), format!("part={} lines=1 Short.\n", a(0)));
        assert!(entry(&doc, 1).ends_with("…\n"));
        assert_eq!(entry(&doc, 2), format!("part={} lines=5-6 Two\n", a(2)));
        let code = Doc::new("a.rs", "fn go() {\n    1\n}\n");
        assert_eq!(
            entry(&code, 0),
            format!("part={} lines=1-3 fn go\n", code.units()[0].anchor)
        );
    }

    #[test]
    fn around_shows_the_edit_between_its_neighbours() {
        let doc = Doc::new("a.md", "One.\n\nTwo.\nmore\n\nThree.\n");
        let a = |i: usize| doc.units()[i].anchor.clone();
        assert_eq!(
            around(&doc, &[1], 2, MD),
            format!(
                "part={} lines=1 One.\n\
                 ```markdown file=r/a.md part={} lines=3-4 scope=\"Two.\"\n +1 Two.\n +2 more\n```\n\
                 part={} lines=6 Three.\n",
                a(0),
                a(1),
                a(2)
            )
        );
        // A delete at line 2 shows what is either side of it now.
        assert_eq!(
            around(&doc, &[], 2, MD),
            format!("part={} lines=1 One.\npart={} lines=3-4 Two.\n", a(0), a(1))
        );
    }
}
