//! Language dispatch + tier-fallback policy.
//!
//! Order of attempt:
//!   1. Language-specific tree-sitter parser (Rust / Python / TS / JS / Go).
//!   2. Header-based fallback (Markdown headers, YAML/TOML/JSON
//!      top-level keys).
//!   3. Fixed-window fallback (100-line chunks).
//!
//! A parser is allowed to return `Some(vec![])` — that means "I parsed
//! it but it contains nothing carve-worthy"; we fall through to the
//! header fallback so we still emit something for the file.

use crate::code_read::parsers::{
    bash, c, cpp, css, fallback, go, html, java, javascript, markdown, php, python, ruby, rust,
    structured_config, typescript,
};
use crate::code_read::types::{ChunkKind, Scope};
use crate::repo_scan::Language;

/// Carve `source` for a file with the given `language` and `is_tsx`
/// (used to pick the TSX dialect of the TS grammar).  Never panics;
/// always returns at least one scope unless the source is empty.
///
/// **Content preservation invariant**: every line of `source` is
/// covered by at least one returned scope.  When the language-aware
/// tier-1 parser only partially succeeds (or skips top-level items
/// like module-level statements that aren't function/type
/// definitions), [`fill_gaps`] adds `Fallback` scopes for every
/// uncovered run of lines so no source content is ever dropped from
/// the prefill pass.  This is load-bearing — a file's content can't
/// be retrieved from the repo_map / code_reading layers if it never
/// reached either of them.
pub fn carve(source: &[u8], language: Language, is_tsx: bool) -> Vec<Scope> {
    if source.is_empty() {
        return Vec::new();
    }
    let total_lines = count_lines(source);
    let tier1 = match language {
        Language::Rust => rust::carve(source),
        Language::Python => python::carve(source),
        Language::TypeScript => typescript::carve(source, is_tsx),
        Language::JavaScript => javascript::carve(source),
        Language::Go => go::carve(source),
        Language::C => c::carve(source),
        Language::Cpp => cpp::carve(source),
        Language::Java => java::carve(source),
        Language::Ruby => ruby::carve(source),
        Language::Php => php::carve(source),
        Language::Bash => bash::carve(source),
        Language::Html => html::carve(source),
        Language::Css => css::carve(source),
        _ => None,
    };
    if let Some(scopes) = tier1 {
        if !scopes.is_empty() {
            return fill_gaps(scopes, total_lines);
        }
    }

    let tier2 = match language {
        Language::Markdown => Some(markdown::carve(source)),
        Language::Yaml | Language::Toml | Language::Json => Some(structured_config::carve(source)),
        _ => None,
    };
    if let Some(scopes) = tier2 {
        if !scopes.is_empty() {
            return fill_gaps(scopes, total_lines);
        }
    }

    // Fallback already covers the whole file in fixed-line windows;
    // no gap-fill needed.
    fallback::carve(source)
}

/// Count `source`'s logical line count — matches the semantics used
/// by the per-scope `start_line` / `end_line` values (1-indexed
/// inclusive ranges over lines split on `\n`).
fn count_lines(source: &[u8]) -> u32 {
    if source.is_empty() {
        return 0;
    }
    let nl_count = source.iter().filter(|&&b| b == b'\n').count() as u32;
    // Source without a trailing newline still has one final line for
    // its un-terminated tail.
    if source.last() == Some(&b'\n') {
        nl_count
    } else {
        nl_count + 1
    }
}

/// Insert `Fallback` scopes covering every line in `[1, total_lines]`
/// that isn't already covered by some scope in `scopes`.  Returns the
/// merged list sorted by `start_line` ascending.
///
/// "Uncovered" is computed by sweeping the input scopes' line ranges
/// in sorted order and emitting a fill scope for each gap.  Gaps are
/// merged greedily so a 50-line stretch of uncovered code produces a
/// single fill scope rather than fifty single-line scopes.  Empty
/// input is treated as "nothing covered" — the whole file becomes one
/// fill scope.
pub fn fill_gaps(mut scopes: Vec<Scope>, total_lines: u32) -> Vec<Scope> {
    if total_lines == 0 {
        return scopes;
    }
    // Sort by start_line; ties broken by end_line so smaller scopes
    // come first.  Sort is stable so input order is preserved within
    // equal ranges (useful when the parser deliberately emits two
    // scopes with the same range — e.g. a typedef-as-struct that
    // covers both an enclosing typedef declarator and the inner
    // struct body).
    scopes.sort_by_key(|s| (s.start_line, s.end_line));

    let mut filled: Vec<Scope> = Vec::with_capacity(scopes.len() + 4);
    let mut cursor: u32 = 1;
    let mut gap_idx: usize = 1;

    for scope in &scopes {
        if scope.start_line > cursor {
            // Uncovered run from `cursor` through `scope.start_line - 1`.
            filled.push(make_fill(cursor, scope.start_line - 1, gap_idx));
            gap_idx += 1;
        }
        filled.push(scope.clone());
        if scope.end_line >= cursor {
            cursor = scope.end_line + 1;
        }
    }
    if cursor <= total_lines {
        filled.push(make_fill(cursor, total_lines, gap_idx));
    }
    filled
}

fn make_fill(start: u32, end: u32, idx: usize) -> Scope {
    Scope {
        path: vec![format!("uncovered {idx}")],
        kind: ChunkKind::Fallback,
        start_line: start,
        end_line: end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_dispatches_to_tree_sitter() {
        let src = b"fn alpha() {}\n";
        let scopes = carve(src, Language::Rust, false);
        assert!(!scopes.is_empty());
        assert!(scopes.iter().any(|s| s.path.last().is_some_and(|n| n == "fn alpha")));
    }

    #[test]
    fn markdown_dispatches_to_header_carver() {
        let src = b"## hello\n\nworld\n";
        let scopes = carve(src, Language::Markdown, false);
        assert!(!scopes.is_empty());
        assert!(scopes.iter().any(|s| s.path[0].contains("hello")));
    }

    #[test]
    fn toml_dispatches_to_structured_carver() {
        let src = b"[package]\nname = \"foo\"\n";
        let scopes = carve(src, Language::Toml, false);
        assert!(!scopes.is_empty());
        assert!(scopes.iter().any(|s| s.path[0] == "package"));
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(carve(b"", Language::Rust, false).is_empty());
    }

    #[test]
    fn fallback_for_text_files() {
        let src = b"just plain text\n";
        let scopes = carve(src, Language::PlainText, false);
        assert!(!scopes.is_empty());
    }

    // ── Content-preservation invariant ───────────────────────────────────────

    /// The all-important guarantee: every line of source survives the
    /// carve pass, sitting in at least one returned scope.
    fn every_line_covered(scopes: &[Scope], total: u32) -> bool {
        let mut cursor = 1u32;
        let mut sorted = scopes.to_vec();
        sorted.sort_by_key(|s| (s.start_line, s.end_line));
        for s in &sorted {
            if s.start_line > cursor {
                return false;
            }
            if s.end_line >= cursor {
                cursor = s.end_line + 1;
            }
        }
        cursor > total
    }

    #[test]
    fn fill_gaps_covers_leading_uncovered_lines() {
        // Scope at lines 5-10 leaves 1-4 uncovered.
        let scope = Scope {
            path: vec!["fn x".into()],
            kind: ChunkKind::Function,
            start_line: 5,
            end_line: 10,
        };
        let filled = fill_gaps(vec![scope], 10);
        assert!(every_line_covered(&filled, 10));
        assert!(filled.iter().any(|s| s.start_line == 1 && s.end_line == 4));
    }

    #[test]
    fn fill_gaps_covers_trailing_uncovered_lines() {
        let scope = Scope {
            path: vec!["fn x".into()],
            kind: ChunkKind::Function,
            start_line: 1,
            end_line: 5,
        };
        let filled = fill_gaps(vec![scope], 10);
        assert!(every_line_covered(&filled, 10));
        assert!(filled.iter().any(|s| s.start_line == 6 && s.end_line == 10));
    }

    #[test]
    fn fill_gaps_covers_middle_gaps() {
        let scopes = vec![
            Scope {
                path: vec!["fn a".into()],
                kind: ChunkKind::Function,
                start_line: 1,
                end_line: 5,
            },
            Scope {
                path: vec!["fn b".into()],
                kind: ChunkKind::Function,
                start_line: 15,
                end_line: 20,
            },
        ];
        let filled = fill_gaps(scopes, 20);
        assert!(every_line_covered(&filled, 20));
        assert!(filled.iter().any(|s| s.start_line == 6 && s.end_line == 14));
    }

    #[test]
    fn fill_gaps_no_op_when_already_fully_covered() {
        let scopes = vec![Scope {
            path: vec!["fn a".into()],
            kind: ChunkKind::Function,
            start_line: 1,
            end_line: 10,
        }];
        let filled = fill_gaps(scopes.clone(), 10);
        assert_eq!(filled, scopes);
    }

    #[test]
    fn fill_gaps_handles_overlapping_scopes() {
        // Tree-sitter sometimes emits nested scopes that overlap —
        // gap-fill should treat the outer one as covering the inner
        // range without inserting a spurious fill.
        let scopes = vec![
            Scope {
                path: vec!["outer".into()],
                kind: ChunkKind::TypeDefinition,
                start_line: 1,
                end_line: 20,
            },
            Scope {
                path: vec!["inner".into()],
                kind: ChunkKind::Function,
                start_line: 5,
                end_line: 10,
            },
        ];
        let filled = fill_gaps(scopes, 20);
        assert!(every_line_covered(&filled, 20));
        // No gaps inserted between or after the overlapping pair.
        assert_eq!(filled.iter().filter(|s| s.path[0].starts_with("uncovered")).count(), 0);
    }

    #[test]
    fn carve_rust_preserves_module_level_use_statements() {
        // Without gap-fill the `use` declarations at lines 1-3 would
        // be dropped because tree-sitter-rust isn't registered for
        // them.  Gap-fill ensures every line including the imports
        // shows up in some scope.
        let src = br#"use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;

fn alpha() { let _ = (HashMap::<String, u32>::new(), Arc::new(0), Path::new(".")); }
"#;
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(
            every_line_covered(&scopes, total),
            "module-level use statements must survive carve: {:?}",
            scopes.iter().map(|s| (s.start_line, s.end_line)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn carve_rust_preserves_top_level_let_in_invalid_position() {
        // Malformed Rust (top-level `let`) — tree-sitter still parses
        // most of the file but the `let` ends up in an error node.
        // Gap-fill keeps the invalid region in the prefill so the
        // model can still see what's there.
        let src = br#"let oops = 1;

fn good() { let _ = 0; }
"#;
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(
            every_line_covered(&scopes, total),
            "invalid top-level region must survive carve: {:?}",
            scopes
        );
    }

    #[test]
    fn carve_python_preserves_module_imports_and_globals() {
        let src = br#"import os
import sys
from pathlib import Path

X = 42
Y = "hello"

def alpha():
    return X
"#;
        let scopes = carve(src, Language::Python, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_typescript_preserves_module_imports_and_exports() {
        let src = br#"import { Foo } from './foo';
import * as utils from './utils';

export const MAX = 100;

export function bar(): number { return MAX; }
"#;
        let scopes = carve(src, Language::TypeScript, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_go_preserves_package_and_import_block() {
        let src = br#"package main

import (
    "fmt"
    "os"
)

func main() {
    fmt.Println("hi")
    os.Exit(0)
}
"#;
        let scopes = carve(src, Language::Go, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_handles_completely_invalid_file_without_loss() {
        // Garbled bytes — no parser succeeds, the carver falls
        // through to the fixed-window fallback which covers the whole
        // file by construction.
        let src = b"$$$ this is not valid in any language $$$\n@@@@@@@\n!!!!\n";
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_handles_truncated_rust_fn() {
        // Function with no closing brace — tree-sitter recovers as
        // much as it can; the trailing-incomplete region is still
        // covered via gap-fill.
        let src = br#"fn alpha() {
    let x = 1;
    let y = 2;
"#;
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_handles_invalid_json_without_loss() {
        let src = b"{ \"key\":\n  \"unterminated\n";
        let scopes = carve(src, Language::Json, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }
}
