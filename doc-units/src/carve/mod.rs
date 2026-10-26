//! Carving a file into the units an author edits — `docs/npcd_anchored_edits.md` §3.
//!
//! Every file comes out as a list of units that partition its non-blank lines:
//! in order, never overlapping, never starting or ending on a blank line, and
//! together holding every line that has anything on it. Blank lines between
//! units belong to none of them — they are the separators an edit keeps.
//!
//! - **Markdown**: the paragraph — a run of non-blank lines between blank
//!   lines — with a heading its own unit, a fenced code block one unit whatever
//!   blank lines it holds, and YAML front matter one unit
//!   ([`paragraphs::markdown`]).
//! - **YAML**: the line ([`paragraphs::lines`]).
//! - **Code**: the semantic scope. Tier 1 is the language's tree-sitter parser
//!   (Rust, Python, TypeScript/TSX, JavaScript, Go, C, C++, Java, Ruby, PHP,
//!   Bash, HTML, CSS); tier 2 splits TOML and JSON at top-level keys. What no
//!   item claims — `use` lines, module attributes, the whole of a file no
//!   parser fires on — is carved into paragraphs. A leading comment block of
//!   [`MIN_FILE_HEADER_LINES`] lines or more is its own first unit.
//!
//! **The innermost item owns its lines.** Parsers emit nested items — a method
//! inside a class, a function inside a function — and an outer item keeps only
//! the lines none of its members claims, so a class is its header and fields
//! and each method is a unit of its own.
//!
//! **No unit is larger than [`MAX_UNIT_LINES`] lines or [`MAX_UNIT_CHARS`]
//! characters.** An oversize item is split by its parser on its body's
//! statement boundaries; anything still over a cap is cut at line boundaries
//! here ([`cap`]).

mod paragraphs;
pub(crate) mod parsers;
#[cfg(test)]
pub(crate) mod test_util;
pub mod types;

use std::cmp::Reverse;

use crate::Language;
use parsers::{
    bash, c, cpp, css, go, html, java, javascript, php, python, ruby, rust, structured_config,
    typescript,
};
use types::{ChunkKind, Scope, MAX_UNIT_CHARS, MAX_UNIT_LINES, MIN_FILE_HEADER_LINES};

/// Carve `source`, written in `language`, into its units. Empty for a file
/// with nothing on it but whitespace.
pub fn carve(source: &str, language: Language) -> Vec<Scope> {
    let lines: Vec<&str> = split_lines(source);
    let units = match language {
        Language::Markdown => paragraphs::markdown(&lines),
        Language::Yaml => paragraphs::lines(&lines),
        Language::PlainText => paragraphs::runs(&lines, 1, lines.len() as u32),
        _ => partition(
            carve_raw(source, language),
            file_header_end(source, language),
            &lines,
        ),
    };
    cap(units, &lines)
}

/// The scopes a language's parser emits, before the partition: possibly
/// nested, possibly leaving lines unclaimed. Empty for a language with no
/// carver, or a source its parser could not parse at all.
pub fn carve_raw(source: &str, language: Language) -> Vec<Scope> {
    let bytes = source.as_bytes();
    let tier1 = match language {
        Language::Rust => rust::carve(bytes),
        Language::Python => python::carve(bytes),
        Language::TypeScript => typescript::carve(bytes, false),
        Language::Tsx => typescript::carve(bytes, true),
        Language::JavaScript => javascript::carve(bytes),
        Language::Go => go::carve(bytes),
        Language::C => c::carve(bytes),
        Language::Cpp => cpp::carve(bytes),
        Language::Java => java::carve(bytes),
        Language::Ruby => ruby::carve(bytes),
        Language::Php => php::carve(bytes),
        Language::Bash => bash::carve(bytes),
        Language::Html => html::carve(bytes),
        Language::Css => css::carve(bytes),
        Language::Toml | Language::Json => Some(structured_config::carve(bytes)),
        Language::Markdown | Language::Yaml | Language::PlainText => None,
    };
    tier1.unwrap_or_default()
}

/// `source` split on `\n`, without the separators — a `\r` before one stays
/// on its line. A terminating newline does not make an empty last line.
pub(crate) fn split_lines(source: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = source.split('\n').collect();
    if source.ends_with('\n') || source.is_empty() {
        lines.pop();
    }
    lines
}

/// Whether a line has nothing on it but whitespace.
pub(crate) fn blank(line: &str) -> bool {
    line.trim().is_empty()
}

/// `raw`'s scopes as a partition of `lines`, with the leading comment block
/// (`[1, header_end]`, none at 0) its own first unit. See the module's
/// "innermost item owns its lines".
fn partition(mut raw: Vec<Scope>, header_end: u32, lines: &[&str]) -> Vec<Scope> {
    let n = lines.len() as u32;
    if header_end > 0 {
        raw.push(Scope {
            path: vec!["file header".to_string()],
            kind: ChunkKind::FileHeader,
            start_line: 1,
            end_line: header_end,
        });
    }
    // Widest first, so each narrower item claims its lines from the one
    // around it; the file header last of all, over whatever item its comments
    // were attached to.
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&i| {
        (
            raw[i].kind == ChunkKind::FileHeader,
            Reverse(raw[i].line_span()),
        )
    });
    let mut owner: Vec<Option<usize>> = vec![None; n as usize + 1];
    for i in order {
        let s = &raw[i];
        for l in s.start_line.max(1)..=s.end_line.min(n) {
            owner[l as usize] = Some(i);
        }
    }

    let mut out = Vec::new();
    let mut l = 1;
    while l <= n {
        let o = owner[l as usize];
        let mut e = l;
        while e < n && owner[e as usize + 1] == o {
            e += 1;
        }
        match o {
            Some(i) => {
                if let Some((s, e)) = trimmed(lines, l, e) {
                    out.push(Scope {
                        path: raw[i].path.clone(),
                        kind: raw[i].kind,
                        start_line: s,
                        end_line: e,
                    });
                }
            }
            None => out.extend(paragraphs::runs(lines, l, e)),
        }
        l = e + 1;
    }
    out
}

/// `[start, end]` less its leading and trailing blank lines; `None` when
/// every line in it is blank.
pub(crate) fn trimmed(lines: &[&str], start: u32, end: u32) -> Option<(u32, u32)> {
    let line = |l: u32| lines[l as usize - 1];
    let mut s = start;
    while s <= end && blank(line(s)) {
        s += 1;
    }
    let mut e = end;
    while e >= s && blank(line(e)) {
        e -= 1;
    }
    (s <= e).then_some((s, e))
}

/// Cut every unit over [`MAX_UNIT_LINES`] lines or [`MAX_UNIT_CHARS`]
/// characters into parts at line boundaries, each as long as both caps allow.
/// A named unit's parts are labelled `[part N]`; a paragraph's are not, since
/// a paragraph is named by its own first line.
fn cap(units: Vec<Scope>, lines: &[&str]) -> Vec<Scope> {
    let chars = |l: u32| lines[l as usize - 1].chars().count() as u32 + 1;
    let mut out = Vec::with_capacity(units.len());
    for unit in units {
        let span_chars: u32 = (unit.start_line..=unit.end_line).map(chars).sum();
        if unit.line_span() <= MAX_UNIT_LINES && span_chars <= MAX_UNIT_CHARS {
            out.push(unit);
            continue;
        }
        let mut part = 1usize;
        let mut start = unit.start_line;
        while start <= unit.end_line {
            let mut end = start;
            let mut used = chars(start);
            while end < unit.end_line
                && end + 1 - start < MAX_UNIT_LINES
                && used + chars(end + 1) <= MAX_UNIT_CHARS
            {
                end += 1;
                used += chars(end);
            }
            // A part never starts or ends on a blank line.
            if let Some((s, e)) = trimmed(lines, start, end) {
                let mut path = unit.path.clone();
                if let Some(last) = path.last_mut() {
                    *last = format!("{last} [part {part}]");
                }
                out.push(Scope {
                    path,
                    kind: unit.kind,
                    start_line: s,
                    end_line: e,
                });
                part += 1;
            }
            start = end + 1;
        }
    }
    out
}

/// Line- and block-comment syntax for a language, used only to find a
/// leading file-header comment block. `.0` = line-comment prefixes; `.1` =
/// `(open, close)` for a block comment. Empty / `None` where the language has
/// no such form, which disables the split.
fn comment_syntax(
    language: Language,
) -> (
    &'static [&'static str],
    Option<(&'static str, &'static str)>,
) {
    match language {
        Language::Rust
        | Language::TypeScript
        | Language::Tsx
        | Language::JavaScript
        | Language::Go
        | Language::C
        | Language::Cpp
        | Language::Java => (&["//"], Some(("/*", "*/"))),
        Language::Php => (&["//", "#"], Some(("/*", "*/"))),
        Language::Python | Language::Ruby | Language::Bash | Language::Toml => (&["#"], None),
        Language::Css => (&[], Some(("/*", "*/"))),
        Language::Html => (&[], Some(("<!--", "-->"))),
        Language::Markdown | Language::Yaml | Language::Json | Language::PlainText => (&[], None),
    }
}

/// Last line (1-indexed, inclusive) of the leading comment block at the very
/// top of `source`, or `0` when there is nothing to split off. Returns `0`
/// unless the file opens with a comment section spanning at least
/// [`MIN_FILE_HEADER_LINES`] comment lines (blank lines between them belong to
/// the block but do not extend its end) **and** a real code line follows it —
/// a file that is nothing but comments has no header to split from a body.
pub(crate) fn file_header_end(source: &str, language: Language) -> u32 {
    let (line_prefixes, block) = comment_syntax(language);
    if line_prefixes.is_empty() && block.is_none() {
        return 0;
    }
    let block_close = block.map_or("", |(_, c)| c);

    let mut last_comment: u32 = 0;
    let mut comment_lines: u32 = 0;
    let mut in_block = false;
    let mut code_follows = false;

    for (i, raw) in source.split('\n').enumerate() {
        let lineno = (i + 1) as u32;
        let t = raw.trim();
        if in_block {
            last_comment = lineno;
            comment_lines += 1;
            if t.contains(block_close) {
                in_block = false;
            }
            continue;
        }
        if t.is_empty() {
            continue;
        }
        if line_prefixes.iter().any(|p| t.starts_with(p)) {
            last_comment = lineno;
            comment_lines += 1;
            continue;
        }
        if let Some((open, close)) = block {
            if t.starts_with(open) {
                in_block = true;
                last_comment = lineno;
                comment_lines += 1;
                if t.contains(close) {
                    in_block = false;
                }
                continue;
            }
        }
        code_follows = true;
        break;
    }

    if !code_follows || comment_lines < MIN_FILE_HEADER_LINES {
        return 0;
    }
    last_comment
}

#[cfg(test)]
mod tests {
    use super::test_util::assert_partition;
    use super::*;

    fn spans(units: &[Scope]) -> Vec<(u32, u32)> {
        units.iter().map(|u| (u.start_line, u.end_line)).collect()
    }

    fn names(units: &[Scope]) -> Vec<String> {
        units.iter().map(Scope::qualified_path).collect()
    }

    #[test]
    fn empty_and_blank_sources_have_no_units() {
        assert!(carve("", Language::Rust).is_empty());
        assert!(carve("\n\n  \n", Language::Markdown).is_empty());
        assert!(carve("   \n", Language::Yaml).is_empty());
    }

    /// **No merge**: two adjacent three-line functions are two units, where
    /// the ingest carver used to grow every scope to fifty lines.
    #[test]
    fn small_items_are_units_of_their_own() {
        let src = "fn a() {\n    1\n}\nfn b() {\n    2\n}\n";
        let units = carve(src, Language::Rust);
        assert_eq!(spans(&units), vec![(1, 3), (4, 6)]);
        assert_eq!(names(&units), vec!["fn a", "fn b"]);
        assert_partition(src, &units);
    }

    /// The lines no item claims are paragraphs, split at blank lines.
    #[test]
    fn unclaimed_lines_are_paragraphs() {
        let src = "use std::fmt;\nuse std::io;\n\nconst _: () = ();\n#[allow(x)]\n\nfn a() {}\n";
        let units = carve(src, Language::Rust);
        assert_partition(src, &units);
        assert_eq!(units[0].kind, ChunkKind::Paragraph);
        assert_eq!((units[0].start_line, units[0].end_line), (1, 2));
        assert!(units[0].path.is_empty());
        assert_eq!(units.last().unwrap().qualified_path(), "fn a");
    }

    /// The innermost item owns its lines: a class keeps its header and
    /// fields, each method is a unit of its own.
    #[test]
    fn a_method_is_carved_out_of_its_class() {
        let src = "class Foo:\n    x = 1\n    def bar(self):\n        return 1\n    def baz(self):\n        return 2\n";
        let units = carve(src, Language::Python);
        assert_partition(src, &units);
        assert_eq!(
            names(&units),
            vec!["class Foo", "class Foo > def bar", "class Foo > def baz"]
        );
        assert_eq!(spans(&units), vec![(1, 2), (3, 4), (5, 6)]);
    }

    #[test]
    fn a_leading_comment_block_is_the_first_unit() {
        let src = "//! Module does X.\n//! It also does Y.\n\nuse std::fmt;\n\nfn alpha() { let _ = 1; }\n";
        let units = carve(src, Language::Rust);
        assert_partition(src, &units);
        assert_eq!(units[0].kind, ChunkKind::FileHeader);
        assert_eq!((units[0].start_line, units[0].end_line), (1, 2));
        assert_eq!(units.last().unwrap().qualified_path(), "fn alpha");
    }

    /// A header the parser had attached to the function below it is peeled
    /// back off.
    #[test]
    fn a_header_attached_to_a_function_is_peeled_off() {
        let src = "//! File doc line 1.\n//! File doc line 2.\nfn alpha() { let _ = 1; }\n";
        let units = carve(src, Language::Rust);
        assert_eq!(spans(&units), vec![(1, 2), (3, 3)]);
        assert_eq!(units[1].qualified_path(), "fn alpha");
    }

    #[test]
    fn a_single_comment_line_stays_with_its_item() {
        let src = "// lone comment\nfn a() { let _ = 0; }\n";
        let units = carve(src, Language::Rust);
        assert_eq!(spans(&units), vec![(1, 2)]);
    }

    #[test]
    fn file_header_end_finds_the_block() {
        assert_eq!(file_header_end("// a\n// b\ncode\n", Language::Rust), 2);
        assert_eq!(
            file_header_end(
                "/*\n * Copyright.\n * MIT.\n */\nfn a(){}\n",
                Language::Rust
            ),
            4
        );
        assert_eq!(file_header_end("//! doc 1\n//! doc 2\n", Language::Rust), 0);
        assert_eq!(file_header_end("fn a(){}\n// mid\n", Language::Rust), 0);
        assert_eq!(file_header_end("# Title\n\nbody\n", Language::Markdown), 0);
    }

    /// A 200-line function splits on its statements, every part within the
    /// cap and no part ending mid-statement.
    #[test]
    fn an_oversize_function_splits_on_statement_boundaries() {
        let mut src = String::from("fn big() {\n");
        for i in 0..200 {
            src.push_str(&format!("    let x{i} = {{\n        {i}\n    }};\n"));
        }
        src.push_str("}\n");
        let units = carve(&src, Language::Rust);
        assert_partition(&src, &units);
        assert!(units.len() >= 8, "{} parts", units.len());
        for (n, u) in units.iter().enumerate() {
            assert!(u.line_span() <= MAX_UNIT_LINES, "{u:?}");
            assert_eq!(u.qualified_path(), format!("fn big [part {}]", n + 1));
            // Each part after the first begins on a statement: `let`, never
            // the `{i}` or `};` lines inside one.
            if n > 0 {
                let first = src.lines().nth(u.start_line as usize - 1).unwrap();
                assert!(first.trim_start().starts_with("let "), "{first:?}");
            }
        }
    }

    /// The caps hold for every tier, and splitting keeps the partition.
    #[test]
    fn no_unit_exceeds_the_caps() {
        let mut rust = String::from("fn a() {}\n");
        for i in 0..300 {
            rust.push_str(&format!("const K{i}: u32 = {i};\n"));
        }
        let mut md = String::from("# only section\n\n");
        for i in 0..300 {
            md.push_str(&format!("line {i} of one long paragraph\n"));
        }
        let mut dense = String::new();
        for _ in 0..60 {
            dense.push_str(&"x".repeat(150));
            dense.push('\n');
        }
        for (src, lang) in [
            (rust.as_str(), Language::Rust),
            (md.as_str(), Language::Markdown),
            (dense.as_str(), Language::PlainText),
        ] {
            let units = carve(src, lang);
            assert_partition(src, &units);
            let lines = split_lines(src);
            for u in &units {
                assert!(u.line_span() <= MAX_UNIT_LINES, "{lang:?}: {u:?}");
                let chars: u32 = (u.start_line..=u.end_line)
                    .map(|l| lines[l as usize - 1].chars().count() as u32 + 1)
                    .sum();
                assert!(chars <= MAX_UNIT_CHARS, "{lang:?}: {chars} chars");
            }
        }
    }

    /// Whatever a parser makes of a broken file, every non-blank line is in
    /// exactly one unit.
    #[test]
    fn broken_sources_still_partition() {
        for (src, lang) in [
            ("fn alpha() {\n    let x = 1;\n    let y = 2;\n", Language::Rust),
            ("let oops = 1;\n\nfn good() { let _ = 0; }\n", Language::Rust),
            ("$$$ not rust $$$\n@@@@\n\n!!!!\n", Language::Rust),
            ("def beta(:\n    # invalid\n\ndef gamma(): return 3\n", Language::Python),
            ("{ \"key\":\n  \"unterminated\n", Language::Json),
            ("import os\n\nX = 42\n\ndef alpha():\n    return X\n", Language::Python),
            (
                "package main\n\nimport (\n    \"fmt\"\n)\n\nfunc main() {\n    fmt.Println(1)\n}\n",
                Language::Go,
            ),
        ] {
            assert_partition(src, &carve(src, lang));
        }
    }

    #[test]
    fn crlf_lines_carve_like_lf_lines() {
        let lf = carve("fn a() {}\n\nfn b() {}\n", Language::Rust);
        let crlf = carve("fn a() {}\r\n\r\nfn b() {}\r\n", Language::Rust);
        assert_eq!(spans(&lf), spans(&crlf));
        assert_eq!(names(&lf), names(&crlf));
    }

    #[test]
    fn toml_splits_at_tables() {
        let src = "[package]\nname = \"x\"\n\n[deps]\ny = \"1\"\n";
        let units = carve(src, Language::Toml);
        assert_eq!(names(&units), vec!["package", "deps"]);
        assert_eq!(spans(&units), vec![(1, 2), (4, 5)]);
    }

    #[test]
    fn parses_reports_a_syntax_error() {
        assert_eq!(parsers::parses(b"fn a() {}\n", Language::Rust), Some(true));
        assert_eq!(parsers::parses(b"fn a() {\n", Language::Rust), Some(false));
        assert_eq!(parsers::parses(b"# x\n", Language::Markdown), None);
    }
}
