//! Rendering for file reads — the shared `<tool_response>` body format.
//!
//! One format, used from both ends of the system:
//!
//! * The live [`file_read`](super::read) tool returns this shape.
//! * The `code_reading` ingest drives that same tool call for every range of
//!   every file it reads ([`windows`]), so the reads prefilled into its
//!   conversations are byte-identical to the ones a live read returns.
//!
//! ````text
//! ```rust file=server/src/auth/handler.rs part=r7Lm lines=14-16 scope="impl AuthHandler > fn validate_token"
//!  +1     pub fn validate_token(&self, token: &str) -> Result<Claims> {
//!  +2         self.keys.verify(token)
//!  +3     }
//! ```
//! end of server/src/auth/handler.rs lines 1-16 of 620 — next: lines=17-208
//! ````
//!
//! Each part of the file — a function, a type, a block of imports, a
//! paragraph — is a fenced block of its own ([`doc_units::render`],
//! `docs/zen_coding_challenge_fixes.md` Cause 4): the fence names the file as a
//! workspace path (`<repo>/<path>` — a path alone is ambiguous across a
//! workspace of several repositories), the part's anchor, the lines it holds
//! and its scope, and every line follows behind its number in the part.
//! `file_edit` names a part by its anchor and lines by those numbers. A read
//! holds whole parts, at most [`PAGE_LINES`] lines of them, and ends with one
//! line naming the file again, what this read held, and the range to read
//! next — or that this was the end.

use doc_units::render::{block, blocks, entry, lines_of, page_lines, pages, Source};
use doc_units::Doc;

pub use doc_units::render::PAGE_LINES;

/// The most parts one page of a list names.
pub const LIST_ENTRIES: usize = 200;

/// Where `path` in `repo` says its blocks are from.
fn source<'a>(file: &'a str, path: &str) -> Source<'a> {
    Source {
        file,
        tag: fence_tag_for_path(path),
    }
}

/// The `lines` ranges (from 1, inclusive) a file reads in: runs of whole
/// parts, each at most [`PAGE_LINES`] lines, tiling the file from line 1 to its
/// last. Reading each in turn reads the whole file, which is what the
/// `code_reading` ingest asks for.
pub fn windows(path: &str, content: &str) -> Vec<(usize, usize)> {
    let doc = Doc::new(path, content);
    page_lines(&doc, &pages(&doc, PAGE_LINES))
}

/// `range` written as a `lines` value: `46-52`, or `46` for one line.
pub fn range_value((from, to): (usize, usize)) -> String {
    match from == to {
        true => format!("{from}"),
        false => format!("{from}-{to}"),
    }
}

/// The parts from part `first` on that a read starting there holds: whole
/// parts while they fit in [`PAGE_LINES`] lines, the first always.
fn fitting(doc: &Doc, first: usize) -> usize {
    let units = doc.units();
    let mut end = first;
    let mut used = 0;
    while end < units.len() {
        let len = units[end].end - units[end].start;
        if end > first && used + len > PAGE_LINES {
            break;
        }
        used += len;
        end += 1;
    }
    end
}

/// The range a read from part `first` covers: from its first line (line 1 for
/// the first part) to the line before the next read's first part, or the end.
fn read_range(doc: &Doc, first: usize, end: usize) -> (usize, usize) {
    let units = doc.units();
    let from = match first {
        0 => 1,
        _ => units[first].start + 1,
    };
    let to = match units.get(end) {
        Some(next) => next.start,
        None => doc.line_count(),
    };
    (from, to.max(from))
}

/// The closing line of a read of `file` that held parts `first..end`.
fn closing(file: &str, doc: &Doc, first: usize, end: usize) -> String {
    let (from, to) = read_range(doc, first, end);
    let total = doc.line_count();
    match end < doc.units().len() {
        true => {
            let next = read_range(doc, end, fitting(doc, end));
            format!(
                "end of {file} lines {} of {total} — next: lines={}\n",
                range_value((from, to)),
                range_value(next)
            )
        }
        false => format!(
            "end of {file} lines {} of {total} — the end of the file\n",
            range_value((from, to))
        ),
    }
}

/// What a read of a file with no parts says.
fn empty(file: &str, doc: &Doc) -> String {
    format!("{file} holds no text — {} line(s), all blank.\n", doc.line_count())
}

/// A read of lines `from`..=`to` (from 1) of `path` in `repo`: the whole parts
/// they touch, from the part holding `from` — or the first after it, when
/// `from` is a blank line between parts — at most [`PAGE_LINES`] lines of
/// them. A range past the end reads the last part. A one-line read says which
/// part and line of it that line is, so a compiler's line number becomes an
/// edit's without arithmetic.
pub fn lines_read(repo: &str, path: &str, doc: &Doc, from: usize, to: usize) -> String {
    let file = format!("{repo}/{path}");
    let units = doc.units();
    if units.is_empty() {
        return empty(&file, doc);
    }
    let first = units
        .iter()
        .position(|u| u.end >= from)
        .unwrap_or(units.len() - 1);
    let end = units[first..]
        .iter()
        .position(|u| u.start >= to)
        .map_or(units.len(), |k| first + k)
        .clamp(first + 1, fitting(doc, first));
    let note = match from == to {
        true => match doc.unit_at(from - 1) {
            Some(i) => format!(
                "line {from} is part={} +{}\n",
                units[i].anchor,
                from - units[i].start
            ),
            None if from <= doc.line_count() => {
                format!("line {from} is a blank line between parts\n")
            }
            None => format!("{file} has {} lines\n", doc.line_count()),
        },
        false => String::new(),
    };
    format!(
        "{note}{}{}",
        blocks(doc, first..end, source(&file, path)),
        closing(&file, doc, first, end)
    )
}

/// A read of the one part `i` of `path` in `repo`.
pub fn part_read(repo: &str, path: &str, doc: &Doc, i: usize) -> String {
    let file = format!("{repo}/{path}");
    format!(
        "{}end of part {} of {file} — lines {} of {}\n",
        block(doc, i, source(&file, path)),
        doc.units()[i].anchor,
        lines_of(doc, i),
        doc.line_count()
    )
}

/// Page `page` (from 1, clamped into range) of the list of `path`'s parts in
/// `repo`, one line each. A file that a single read holds whole comes back
/// whole instead, so reading a short file is one call.
pub fn list_read(repo: &str, path: &str, doc: &Doc, page: usize) -> String {
    let file = format!("{repo}/{path}");
    let units = doc.units();
    if units.is_empty() {
        return empty(&file, doc);
    }
    if fitting(doc, 0) == units.len() {
        return format!(
            "{}{}",
            blocks(doc, 0..units.len(), source(&file, path)),
            closing(&file, doc, 0, units.len())
        );
    }
    let total = units.len().div_ceil(LIST_ENTRIES);
    let page = page.clamp(1, total);
    let shown = (page - 1) * LIST_ENTRIES..(page * LIST_ENTRIES).min(units.len());
    let after = match page < total {
        true => format!("— next: list={}", page + 1),
        false => "— the last page".to_string(),
    };
    format!(
        "{file}: {} lines in {} parts. Read one with `part`, or a run of them with `lines`.\n\
         {}end of the parts of {file} page {page} of {total} {after}\n",
        doc.line_count(),
        units.len(),
        shown.map(|i| entry(doc, i)).collect::<String>(),
    )
}

/// Markdown fence tag for a path's extension. Mirrors `zend`'s
/// `repo_scan::Language::fence_tag`; the two agree by a test in `zend`, which is
/// the only crate that can see both.
pub fn fence_tag_for_path(path: &str) -> &'static str {
    let ext = path
        .rsplit('/')
        .next()
        .and_then(|name| name.rsplit_once('.'))
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "go" => "go",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hxx" | "hh" => "cpp",
        "java" => "java",
        "rb" | "rake" | "ru" | "gemspec" => "ruby",
        "php" | "phtml" => "php",
        "sh" | "bash" | "zsh" => "bash",
        "html" | "htm" => "html",
        "css" | "scss" | "sass" | "less" => "css",
        "md" | "markdown" | "mdx" => "markdown",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "json" | "json5" | "jsonc" => "json",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchors(doc: &Doc) -> Vec<String> {
        doc.anchors()
    }

    /// A short file read by its first range is every part, then the end.
    #[test]
    fn a_read_shows_each_part_as_a_block_then_the_end() {
        let doc = Doc::new("a.rs", "use x;\n\nfn go() {\n    1\n}\n");
        let a = anchors(&doc);
        assert_eq!(
            lines_read("r", "a.rs", &doc, 1, 5),
            format!(
                "```rust file=r/a.rs part={} lines=1 scope=\"use x;\"\n +1 use x;\n```\n\
                 ```rust file=r/a.rs part={} lines=3-5 scope=\"fn go\"\n +1 fn go() {{\n +2     1\n +3 }}\n```\n\
                 end of r/a.rs lines 1-5 of 5 — the end of the file\n",
                a[0], a[1]
            )
        );
    }

    /// **The ranges a file reads in tile it**, a read of each holds exactly its
    /// parts, and its closing line names the next.
    #[test]
    fn reads_of_each_window_hold_its_parts_and_name_the_next() {
        let text: String = (0..150).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        let doc = Doc::new("a.md", &text);
        let w = windows("a.md", &text);
        assert_eq!(w, vec![(1, 300), (301, 450)]);
        let first = lines_read("r", "a.md", &doc, 1, 300);
        assert_eq!(first.matches("```markdown").count(), 100);
        assert!(first.contains("lines=298-299 scope=\"p99 a\""), "{first}");
        assert!(first.ends_with("end of r/a.md lines 1-300 of 450 — next: lines=301-450\n"));
        let last = lines_read("r", "a.md", &doc, 301, 450);
        assert_eq!(last.matches("```markdown").count(), 50);
        assert!(last.ends_with("end of r/a.md lines 301-450 of 450 — the end of the file\n"));
    }

    /// **A one-line read names the part and its line**, a blank line names
    /// none, and a line past the end reads the last part.
    #[test]
    fn a_one_line_read_says_which_part_and_line_it_is() {
        let doc = Doc::new("a.rs", "use x;\n\nfn go() {\n    1\n}\n");
        let a = anchors(&doc);
        assert!(lines_read("r", "a.rs", &doc, 4, 4).starts_with(&format!(
            "line 4 is part={} +2\n```rust file=r/a.rs part={}",
            a[1], a[1]
        )));
        assert!(lines_read("r", "a.rs", &doc, 2, 2)
            .starts_with(&format!("line 2 is a blank line between parts\n```rust file=r/a.rs part={}", a[1])));
        assert!(lines_read("r", "a.rs", &doc, 99, 99)
            .starts_with(&format!("r/a.rs has 5 lines\n```rust file=r/a.rs part={}", a[1])));
    }

    #[test]
    fn a_part_read_is_its_block_alone() {
        let doc = Doc::new("a.rs", "use x;\n\nfn go() {\n    1\n}\n");
        let a = anchors(&doc);
        assert_eq!(
            part_read("r", "a.rs", &doc, 1),
            format!(
                "```rust file=r/a.rs part={} lines=3-5 scope=\"fn go\"\n +1 fn go() {{\n +2     1\n +3 }}\n```\n\
                 end of part {} of r/a.rs — lines 3-5 of 5\n",
                a[1], a[1]
            )
        );
    }

    /// **A list names every part in one line**, paged, and a file one read
    /// holds comes back whole.
    #[test]
    fn a_list_names_the_parts_or_reads_a_short_file_whole() {
        let short = Doc::new("a.rs", "use x;\n\nfn go() {}\n");
        assert_eq!(
            list_read("r", "a.rs", &short, 1),
            lines_read("r", "a.rs", &short, 1, 3)
        );
        let text: String = (0..150).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        let doc = Doc::new("a.md", &text);
        let a = anchors(&doc);
        let list = list_read("r", "a.md", &doc, 1);
        assert!(list.starts_with(&format!(
            "r/a.md: 450 lines in 150 parts. Read one with `part`, or a run of them with `lines`.\n\
             part={} lines=1-2 p0 a\n",
            a[0]
        )));
        assert!(list.ends_with("end of the parts of r/a.md page 1 of 1 — the last page\n"));
    }

    #[test]
    fn an_empty_file_says_so() {
        let blank = Doc::new("a.rs", "\n\n");
        assert_eq!(
            lines_read("r", "a.rs", &blank, 1, 1),
            "r/a.rs holds no text — 2 line(s), all blank.\n"
        );
        assert_eq!(
            list_read("r", "a.rs", &Doc::new("a.rs", ""), 1),
            "r/a.rs holds no text — 0 line(s), all blank.\n"
        );
    }

    #[test]
    fn an_unknown_extension_renders_a_bare_fence() {
        assert_eq!(fence_tag_for_path("x/y.bin"), "");
        assert_eq!(fence_tag_for_path("Makefile"), "");
        assert_eq!(fence_tag_for_path("a/b/c.rs"), "rust");
        assert_eq!(fence_tag_for_path("A/B/C.RS"), "rust", "case-insensitive");
    }
}
