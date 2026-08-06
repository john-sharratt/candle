//! The anchor excerpt for a directory — the piece of prose that says what the
//! directory *is*.
//!
//! A listing tells you a folder's shape; it does not tell you its purpose. One
//! file per directory usually does: a `README.md`, or the module doc block at the
//! top of `lib.rs` / `main.rs` / `mod.rs`. [`pick`] finds that file and returns
//! the excerpt worth reading, so the folder's summary turn is grounded in the
//! author's own description rather than inferred from filenames.
//!
//! **The module doc outranks the leading comment block.** A crate root often
//! opens with an ordinary `//` note — a licence header, a block of lint
//! rationale — above its `#![…]` attributes, with the real `//!` description
//! below them. Taking the file's first comment run would then describe the
//! folder by whatever housekeeping happened to sit at the top:
//! `candle-conversation/src/` summarised as "a Rust library file that includes
//! several Clippy lint suppressions". [`pick`] therefore looks for the `//!`
//! block wherever it sits in the file's prelude and excerpts *that*.
//!
//! Reuses the `code_reading` carve helpers rather than re-deriving them:
//! [`split_long_lines`] bounds every line before anything is measured (a minified
//! or generated one-liner would otherwise be a single enormous "line"), and
//! [`file_header_end`] finds the leading comment block using the same
//! language-aware, line-based rule that gives every carved file its
//! `ChunkKind::FileHeader` first turn.

use std::path::Path;

use crate::code_read::carve::{file_header_end, split_long_lines};
use crate::code_read::{compute_line_offsets, slice_lines};
use crate::repo_scan::types::{FileEntry, Language};

/// Most lines an anchor excerpt carries. The excerpt exists to say what the
/// folder is, which a module doc or a README's opening states well inside this;
/// past it we are reading the file, not its description. Matches the live
/// `file_read` cap so a prefilled excerpt is a page the tool could itself return.
pub const MAX_ANCHOR_LINES: u32 = 200;

/// Filenames that describe their directory, in preference order. A README is
/// prose written for exactly this purpose, so it wins; otherwise the crate or
/// module root carries the `//!` block.
const ANCHOR_NAMES: &[&str] = &["readme.md", "lib.rs", "main.rs", "mod.rs"];

/// A directory's chosen anchor excerpt, ready to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Anchor {
    /// Workspace-relative path of the anchor file.
    pub path: String,
    /// First line of the excerpt, 1-based.
    pub start_line: u32,
    /// Last line of the excerpt, 1-based inclusive.
    pub end_line: u32,
    /// Total lines in the file, so the excerpt header can say `of N` when the
    /// excerpt is a slice — the same continuation signal `file_read` emits.
    pub total_lines: u32,
    /// The excerpt's source text.
    pub body: String,
    pub language: Language,
}

/// Choose and read the anchor excerpt for a directory, or `None` when no
/// candidate file is present (most leaf directories) or it yields nothing.
///
/// `files` are the entries **directly inside** the directory; `workspace` is the
/// root the paths are relative to.
pub fn pick(files: &[&FileEntry], workspace: &Path) -> Option<Anchor> {
    let file = ANCHOR_NAMES.iter().find_map(|want| {
        files
            .iter()
            .find(|f| basename(&f.path).eq_ignore_ascii_case(want))
    })?;
    let bytes = std::fs::read(workspace.join(&file.path)).ok()?;
    // Bound every line BEFORE measuring, so a generated one-liner can't make the
    // whole excerpt one unbounded line. The split bytes are what we slice and
    // show, so the line numbers we report are the ones the reader sees.
    let bytes = split_long_lines(&bytes);
    let offsets = compute_line_offsets(&bytes);
    let total_lines = line_count(&bytes);
    if total_lines == 0 {
        return None;
    }

    // Markdown has no comment syntax to peel — a README's opening IS the
    // description, so the excerpt is simply its head.
    let (start_line, end) = if file.language == Language::Markdown {
        (1, total_lines)
    } else if let Some(run) = module_doc_run(&bytes, file.language) {
        // The file says what it is in its own words — take exactly that, and
        // nothing of the licence header or lint rationale that may sit above it.
        run
    } else {
        let header = file_header_end(&bytes, file.language);
        let end = if header > 0 {
            header
        } else {
            // No leading block: the file opens with code (imports, attributes).
            // Its description, if any, is the doc comment on the first item.
            doc_block_after_head(&bytes, file.language).unwrap_or(total_lines)
        };
        (1, end)
    };
    let end_line = end
        .min(start_line + MAX_ANCHOR_LINES - 1)
        .min(total_lines)
        .max(start_line);
    let body = slice_lines(&bytes, &offsets, start_line, end_line);
    if body.trim().is_empty() {
        return None;
    }
    Some(Anchor {
        path: file.path.clone(),
        start_line,
        end_line,
        total_lines,
        body,
        language: file.language,
    })
}

/// Inclusive line range of the file's module-doc block — the `//!` run
/// documenting the module itself, as opposed to any item inside it.
///
/// Searched across the file's *prelude*: the region a module doc may legally
/// occupy, which is everything above the first item. Blank lines, ordinary
/// comments and inner attributes may precede it, so this walks past a licence
/// header or a block of `#![allow(…)]` and still finds the description below
/// them — [`file_header_end`] stops at the first of those and would report the
/// housekeeping instead. The first non-prelude line is the module's first item;
/// a `//!` below that documents an inner module, not this file, so the search
/// stops there.
///
/// `None` for every language but Rust, which alone in this set spells a module's
/// own documentation differently from an ordinary comment.
fn module_doc_run(source: &[u8], language: Language) -> Option<(u32, u32)> {
    let prefix = module_doc_prefix(language)?;
    let text = String::from_utf8_lossy(source);
    let mut run: Option<(u32, u32)> = None;
    // A licence header spelled `/* … */` is common in vendored and generated
    // sources, and its interior lines look like arbitrary prose — nothing in
    // them marks them as a comment. Without tracking the block, the first such
    // line reads as code, the scan stops, and the module doc below it is missed:
    // the same defect this function exists to fix, spelled with a block comment.
    let mut in_block = false;
    for (i, raw) in text.split('\n').enumerate() {
        let lineno = (i + 1) as u32;
        let t = raw.trim();
        if in_block {
            match t.split_once("*/") {
                Some((_, rest)) => {
                    in_block = false;
                    if !precedes_module_doc(rest.trim()) {
                        break;
                    }
                }
                None => continue,
            }
        } else if t.starts_with(prefix) {
            run = Some(match run {
                Some((start, _)) => (start, lineno),
                None => (lineno, lineno),
            });
        } else if run.is_some() {
            // The block ended. A later run is a second doc block, not the one
            // that opens the module.
            break;
        } else if t.starts_with("/*") {
            match t.split_once("*/") {
                // Opened and closed on one line — whatever follows still has to
                // be something a module doc may sit under.
                Some((_, rest)) => {
                    if !precedes_module_doc(rest.trim()) {
                        break;
                    }
                }
                None => in_block = true,
            }
        } else if !precedes_module_doc(t) {
            break;
        }
    }
    run
}

/// How a language spells a module's own documentation, when it distinguishes it
/// from an ordinary comment at all.
fn module_doc_prefix(language: Language) -> Option<&'static str> {
    match language {
        Language::Rust => Some("//!"),
        _ => None,
    }
}

/// Whether `line` (already trimmed) may legally sit above a module doc: a blank,
/// an ordinary line comment, or an inner attribute. Rust-shaped, and only ever
/// reached for Rust — [`module_doc_prefix`] gates the caller.
///
/// Block comments are the caller's business: they span lines, so recognising
/// them is a state machine rather than a per-line test.
fn precedes_module_doc(line: &str) -> bool {
    line.is_empty() || line.starts_with("//") || line.starts_with("#!")
}

/// End line of the first doc-comment run that follows the file's opening code.
///
/// [`file_header_end`] only reports a block at the very TOP of the file, so a
/// file that opens with `use` statements and then documents its first item gets
/// nothing from it. This picks that run up: scan for the first line-comment run
/// after the head, and return its last line. `None` when the file has no comment
/// run at all.
///
/// Deliberately the *whole* comment run, not just `///` — the carve's header rule
/// is likewise prefix-based (`//` covers `//!`, `///` and `//`), and a file whose
/// first documented item is preceded by an ordinary explanatory comment is
/// describing itself just as usefully.
fn doc_block_after_head(source: &[u8], language: Language) -> Option<u32> {
    let prefixes = line_comment_prefixes(language);
    if prefixes.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(source);
    let mut run_end: Option<u32> = None;
    for (i, raw) in text.split('\n').enumerate() {
        let lineno = (i + 1) as u32;
        if lineno > MAX_ANCHOR_LINES {
            break;
        }
        let t = raw.trim();
        if prefixes.iter().any(|p| t.starts_with(p)) {
            run_end = Some(lineno);
        } else if run_end.is_some() && !t.is_empty() {
            // The run ended at the first non-comment, non-blank line after it —
            // that is the item being documented, and the excerpt stops before it.
            break;
        }
    }
    run_end
}

/// Line-comment prefixes per language. Mirrors the carve's `comment_syntax`
/// line-prefix half; block comments are irrelevant here because a block at the
/// top of a file is already [`file_header_end`]'s job.
fn line_comment_prefixes(language: Language) -> &'static [&'static str] {
    match language {
        Language::Rust
        | Language::TypeScript
        | Language::JavaScript
        | Language::Go
        | Language::C
        | Language::Cpp
        | Language::Java => &["//"],
        Language::Php => &["//", "#"],
        Language::Python | Language::Ruby | Language::Bash | Language::Yaml | Language::Toml => {
            &["#"]
        }
        _ => &[],
    }
}

/// Lines in `bytes` — a trailing newline does not add one.
fn line_count(bytes: &[u8]) -> u32 {
    let text = String::from_utf8_lossy(bytes);
    if text.is_empty() {
        return 0;
    }
    let parts = text.split('\n').count();
    if text.ends_with('\n') {
        (parts - 1) as u32
    } else {
        parts as u32
    }
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn entry(path: &str, language: Language) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            line_count: 0,
            language,
            size_bytes: 0,
            module_hint: None,
        }
    }

    fn workspace(files: &[(&str, &str)]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        for (rel, body) in files {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    #[test]
    fn prefers_a_readme_over_a_module_root() {
        let (_d, root) = workspace(&[
            ("a/README.md", "# a\n\nWhat this folder does.\n"),
            ("a/mod.rs", "//! module doc\n//! more\npub fn x() {}\n"),
        ]);
        let files = [
            entry("a/README.md", Language::Markdown),
            entry("a/mod.rs", Language::Rust),
        ];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!(a.path, "a/README.md");
        assert!(a.body.contains("What this folder does."));
    }

    /// A module root contributes its `//!` block, not the code beneath it.
    #[test]
    fn takes_only_the_module_doc_block() {
        let (_d, root) = workspace(&[(
            "a/mod.rs",
            "//! Line one.\n//! Line two.\n\nuse std::fmt;\npub fn x() {}\n",
        )]);
        let files = [entry("a/mod.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!((a.start_line, a.end_line), (1, 2));
        assert_eq!(a.total_lines, 5);
        assert_eq!(a.body, "//! Line one.\n//! Line two.\n");
    }

    /// A crate root whose `//!` block sits BELOW a lint-rationale comment and the
    /// `#![allow(…)]` attributes it explains — the shape of
    /// `candle-conversation/src/lib.rs`. Excerpting the file's first comment run
    /// described that folder as "a Rust library file that includes several Clippy
    /// lint suppressions"; the module doc is what says what the crate is.
    #[test]
    fn the_module_doc_outranks_a_lint_preamble_above_it() {
        let (_d, root) = workspace(&[(
            "a/lib.rs",
            "// Several clippy lints are systemic in this crate.\n\
             // * `type_complexity` — nested tuples by design.\n\
             #![allow(clippy::type_complexity)]\n\
             \n\
             //! Turn-based conversation engine.\n\
             //!\n\
             //! Manages multi-turn dialogue with streaming generation.\n\
             \n\
             mod config;\n",
        )]);
        let files = [entry("a/lib.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!((a.start_line, a.end_line), (5, 7));
        assert_eq!(a.total_lines, 9);
        assert!(a.body.starts_with("//! Turn-based conversation engine."));
        assert!(!a.body.contains("clippy"), "the preamble is not the anchor");
    }

    /// Same shape, spelled as a `/* … */` licence header — the form vendored and
    /// generated sources use. Its interior lines carry no comment marker, so a
    /// per-line test reads the first of them as code and stops before ever
    /// reaching the module doc.
    #[test]
    fn the_module_doc_outranks_a_block_comment_licence_header() {
        let (_d, root) = workspace(&[(
            "a/lib.rs",
            "/*\n\
             * Copyright 2026 Example Corp.\n\
             * Licensed under the Apache License, Version 2.0.\n\
             */\n\
             \n\
             //! Turn-based conversation engine.\n\
             //!\n\
             //! Manages multi-turn dialogue with streaming generation.\n\
             \n\
             mod config;\n",
        )]);
        let files = [entry("a/lib.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!((a.start_line, a.end_line), (6, 8));
        assert!(a.body.starts_with("//! Turn-based conversation engine."));
        assert!(!a.body.contains("Copyright"), "the licence is not the anchor");
    }

    /// A block comment opened and closed on one line is still a prelude line.
    #[test]
    fn a_one_line_block_comment_does_not_end_the_prelude() {
        let (_d, root) = workspace(&[(
            "a/lib.rs",
            "/* generated by build.rs — do not edit */\n\
             \n\
             //! Generated bindings.\n\
             \n\
             pub struct X;\n",
        )]);
        let files = [entry("a/lib.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!((a.start_line, a.end_line), (3, 3));
        assert!(a.body.starts_with("//! Generated bindings."));
    }

    /// Code sharing the closing line of a block comment ends the prelude — a
    /// `//!` below THAT documents an inner module, not this file.
    #[test]
    fn code_after_a_block_comment_close_ends_the_prelude() {
        let src = b"/* header\n\
                    */ pub struct Early;\n\
                    \n\
                    //! not a module doc\n";
        assert_eq!(module_doc_run(src, Language::Rust), None);
    }

    /// The prelude is where a module doc may legally sit, and nowhere else.
    /// `//!` below the module's first item documents an INNER module, so the
    /// search must stop at the item rather than excerpt someone else's docs as
    /// this folder's description.
    #[test]
    fn the_module_doc_search_covers_the_prelude_and_stops_at_the_first_item() {
        assert_eq!(
            module_doc_run(b"//! One.\n//! Two.\n\nmod x;\n", Language::Rust),
            Some((1, 2)),
        );
        assert_eq!(
            module_doc_run(
                b"// note\n#![allow(x)]\n\n//! Real.\nmod x;\n",
                Language::Rust
            ),
            Some((4, 4)),
            "walks past a comment, an inner attribute and a blank",
        );
        assert_eq!(
            module_doc_run(
                b"use std::fmt;\n\npub mod inner {\n//! Inner's doc.\n}\n",
                Language::Rust,
            ),
            None,
            "an item was reached first",
        );
        assert_eq!(
            module_doc_run(b"//! One.\n\n//! A second block.\nmod x;\n", Language::Rust),
            Some((1, 1)),
            "only the block that opens the module",
        );
        assert_eq!(
            module_doc_run(b"# Heading\n", Language::Markdown),
            None,
            "no language but Rust separates module docs from comments",
        );
    }

    /// A licence header with no module doc anywhere still anchors on the header —
    /// the `//!` rule adds a preference, it does not remove the fallbacks.
    #[test]
    fn a_file_with_no_module_doc_keeps_its_leading_comment_block() {
        let (_d, root) = workspace(&[(
            "a/mod.rs",
            "// Copyright the authors.\n// Licensed under MIT.\npub fn x() {}\n",
        )]);
        let files = [entry("a/mod.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!((a.start_line, a.end_line), (1, 2));
        assert!(a.body.contains("Copyright the authors."));
    }

    /// The excerpt cap counts lines of excerpt, not lines of file, so a module
    /// doc that starts deep in the prelude still gets a full page.
    #[test]
    fn the_line_cap_applies_from_the_excerpts_own_start() {
        let doc: String = (1..=300).map(|i| format!("//! doc {i}\n")).collect();
        let body = format!("// preamble\n// preamble\n#![allow(unused)]\n{doc}mod x;\n");
        let (_d, root) = workspace(&[("a/lib.rs", body.as_str())]);
        let files = [entry("a/lib.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!(a.start_line, 4);
        assert_eq!(a.end_line, 4 + MAX_ANCHOR_LINES - 1);
        assert_eq!(a.body.lines().count(), MAX_ANCHOR_LINES as usize);
    }

    /// The fallback: no leading block, but the first item is documented.
    #[test]
    fn falls_back_to_the_doc_run_after_the_opening_code() {
        let (_d, root) = workspace(&[(
            "a/lib.rs",
            "use std::fmt;\n\n/// Does the thing.\n/// In detail.\npub fn x() {}\n",
        )]);
        let files = [entry("a/lib.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!(a.end_line, 4, "through the end of the /// run");
        assert!(a.body.contains("Does the thing."));
    }

    /// Nothing documented at all — the head of the file is still better than
    /// nothing, bounded by the cap.
    #[test]
    fn falls_back_to_the_file_head_when_undocumented() {
        let body: String = (1..=400).map(|i| format!("pub fn f{i}() {{}}\n")).collect();
        let (_d, root) = workspace(&[("a/mod.rs", body.as_str())]);
        let files = [entry("a/mod.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert_eq!(a.end_line, MAX_ANCHOR_LINES);
        assert_eq!(a.total_lines, 400);
    }

    #[test]
    fn a_directory_with_no_candidate_has_no_anchor() {
        let (_d, root) = workspace(&[("a/thing.rs", "pub fn x() {}\n")]);
        let files = [entry("a/thing.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        assert!(pick(&refs, &root).is_none());
    }

    #[test]
    fn an_empty_anchor_file_yields_nothing() {
        let (_d, root) = workspace(&[("a/mod.rs", "")]);
        let files = [entry("a/mod.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        assert!(pick(&refs, &root).is_none());
    }

    /// A generated one-liner is split before measuring, so the excerpt is a
    /// bounded number of bounded lines rather than one enormous one.
    #[test]
    fn an_over_long_line_is_split_before_the_excerpt_is_taken() {
        let long = format!("pub const T: &str = \"{}\";\n", "x".repeat(4000));
        let (_d, root) = workspace(&[("a/mod.rs", long.as_str())]);
        let files = [entry("a/mod.rs", Language::Rust)];
        let refs: Vec<&FileEntry> = files.iter().collect();
        let a = pick(&refs, &root).expect("anchor");
        assert!(a.total_lines > 1, "the single line was split");
        assert!(
            a.body.lines().all(|l| l.chars().count() <= 200),
            "every excerpt line is bounded",
        );
    }
}
