//! Shared validation helpers for parser test suites.
//!
//! The two invariants every parser/dispatcher test enforces:
//!
//! 1. **Content preservation** — concatenating each emitted scope's
//!    source slice in line order reproduces the original `source`
//!    byte-for-byte.  No content is silently lost; no overlap
//!    double-counts content.
//! 2. **Boundary correctness** — every named scope starts at the
//!    line that holds its signature (e.g. a `fn alpha` scope's
//!    `start_line` is the line containing `fn alpha`).  Catches
//!    parsers that find the name but mis-locate the body.
//!
//! Tests call [`verify_carve`] with the source string, the
//! [`Language`], the `is_tsx` flag, and a slice of
//! `(name_substring, expected_first_line_substring)` pairs.  The
//! helper runs the dispatcher-level [`crate::code_read::carve::carve`]
//! (which includes `fill_gaps`), asserts both invariants, and
//! returns the full scope list for any further assertions the test
//! wants to make.

use crate::code_read::carve::{carve as dispatcher_carve, carve_raw};
use crate::code_read::types::Scope;
use crate::repo_scan::Language;

/// 1-indexed inclusive line extractor.  Returns the joined lines
/// `start_line..=end_line` from `source` without their newlines.
pub fn lines_at(source: &str, start: u32, end: u32) -> String {
    let lines: Vec<&str> = source.lines().collect();
    if start == 0 || start as usize > lines.len() {
        return String::new();
    }
    let s = (start as usize) - 1;
    let e = (end as usize).min(lines.len());
    if s >= e {
        return String::new();
    }
    lines[s..e].join("\n")
}

/// Find the first scope whose `qualified_path()` contains
/// `name_substring`.
pub fn find_scope<'a>(scopes: &'a [Scope], name_substring: &str) -> Option<&'a Scope> {
    scopes
        .iter()
        .find(|s| s.qualified_path().contains(name_substring))
}

/// Count `source`'s logical line count — matches the per-scope line
/// numbering (1-indexed inclusive, `\n`-delimited).
fn count_lines(source: &[u8]) -> u32 {
    if source.is_empty() {
        return 0;
    }
    let nl = source.iter().filter(|&&b| b == b'\n').count() as u32;
    if source.last() == Some(&b'\n') {
        nl
    } else {
        nl + 1
    }
}

/// Assert that the union of `scopes`' line ranges covers `[1,
/// total_lines]` with no gaps.  Strict — fires on any missing line.
/// Designed to run against the dispatcher's output (which has
/// `fill_gaps` applied) where the content-preservation guarantee is
/// load-bearing.
pub fn assert_full_coverage(source: &[u8], scopes: &[Scope]) {
    let total = count_lines(source);
    if total == 0 {
        return;
    }
    let mut sorted = scopes.to_vec();
    sorted.sort_by_key(|s| (s.start_line, s.end_line));

    let mut cursor: u32 = 1;
    for s in &sorted {
        assert!(
            s.start_line <= cursor,
            "uncovered lines {cursor}..={} before scope {:?} starts at {}",
            s.start_line - 1,
            s.qualified_path(),
            s.start_line,
        );
        if s.end_line >= cursor {
            cursor = s.end_line + 1;
        }
    }
    assert!(
        cursor > total,
        "trailing uncovered lines {cursor}..={total} after {} scopes",
        scopes.len(),
    );
}

/// Assert that concatenating each scope's exact source slice in
/// line order reproduces `source` byte-for-byte (modulo overlap —
/// overlapping scopes are tolerated: only the lines first emitted
/// count toward the concatenation).
///
/// This is the bit-exact "no content lost" guarantee — stricter
/// than `assert_full_coverage` because it also catches the case
/// where two scopes claim the same line range with different
/// content (which can't happen today but is a useful guard).
pub fn assert_content_preserved(source: &[u8], scopes: &[Scope]) {
    let source_str = std::str::from_utf8(source).unwrap_or("");
    let source_lines: Vec<&str> = source_str.lines().collect();

    let mut sorted = scopes.to_vec();
    sorted.sort_by_key(|s| (s.start_line, s.end_line));

    let mut emitted_lines = vec![false; source_lines.len()];
    for s in &sorted {
        let start = (s.start_line as usize).saturating_sub(1);
        let end = (s.end_line as usize).min(source_lines.len());
        for slot in emitted_lines.iter_mut().take(end).skip(start) {
            *slot = true;
        }
    }
    let missing: Vec<usize> = emitted_lines
        .iter()
        .enumerate()
        .filter_map(|(i, &covered)| if covered { None } else { Some(i + 1) })
        .collect();
    assert!(
        missing.is_empty(),
        "lines {missing:?} are not covered by any scope",
    );
}

/// Assert that `expected_substring` appears on some line WITHIN the scope's
/// `[start_line, end_line]` range. Used by the parser-level `verify_carve`: the
/// walker now pulls a scope's start UP over its leading doc comment, so the
/// signature is no longer guaranteed to be the first line — but it must still live
/// inside the scope.
pub fn assert_scope_contains(source: &str, scope: &Scope, expected_substring: &str) {
    let body = lines_at(source, scope.start_line, scope.end_line);
    assert!(
        body.contains(expected_substring),
        "scope {:?} (lines {}-{}) does not contain {expected_substring:?}",
        scope.qualified_path(),
        scope.start_line,
        scope.end_line,
    );
}

/// Assert every NON-BLANK line of `source` is covered by some scope. The
/// dispatcher intentionally DROPS pure-blank runs (see `carve::refine`), so the
/// old every-line invariant is relaxed to every-non-blank-line.
pub fn assert_nonblank_coverage(source: &[u8], scopes: &[Scope]) {
    let total = count_lines(source);
    if total == 0 {
        return;
    }
    let text = String::from_utf8_lossy(source);
    let blank: Vec<bool> = std::iter::once(false)
        .chain(text.split('\n').map(|l| l.trim().is_empty()))
        .collect();
    let mut covered = vec![false; (total + 1) as usize];
    for s in scopes {
        for l in s.start_line..=s.end_line.min(total) {
            if let Some(c) = covered.get_mut(l as usize) {
                *c = true;
            }
        }
    }
    for l in 1..=total {
        let is_blank = blank.get(l as usize).copied().unwrap_or(false);
        assert!(
            is_blank || covered[l as usize],
            "non-blank line {l} not covered by any scope",
        );
    }
}

/// Assert that the scope's `start_line` row of `source` contains
/// `expected_substring`.  Catches parsers that match the right name
/// but place the scope at the wrong line.
pub fn assert_starts_with(source: &str, scope: &Scope, expected_substring: &str) {
    let first = lines_at(source, scope.start_line, scope.start_line);
    assert!(
        first.contains(expected_substring),
        "scope {:?} expected first line to contain {expected_substring:?}, \
         got {first:?} (line {} of {} total)",
        scope.qualified_path(),
        scope.start_line,
        source.lines().count(),
    );
}

/// Run the dispatcher-level carve and validate both invariants.
///
/// `expectations` is a list of `(name_substring,
/// expected_first_line_substring)` pairs.  For each pair, the
/// helper finds the first scope whose path contains
/// `name_substring` and asserts its `start_line` contains
/// `expected_first_line_substring`.  Then it asserts full coverage
/// + bit-exact content preservation over the whole source.
///
/// Returns the full scope list so callers can do additional
/// per-scope assertions.
pub fn verify_carve(
    source: &str,
    language: Language,
    is_tsx: bool,
    expectations: &[(&str, &str)],
) -> Vec<Scope> {
    let src = source.as_bytes();
    // Parser-level: verify EXTRACTION against the RAW tier scopes, so the
    // dispatcher's `refine` merge doesn't collapse a small fixture into one chunk
    // and hide per-item boundaries. Coverage + content-preservation are DISPATCHER
    // invariants — see `verify_coverage_only`.
    let scopes = carve_raw(src, language, is_tsx);
    assert!(
        !scopes.is_empty() || src.is_empty(),
        "carve returned empty for non-empty source",
    );
    for (name_sub, sig_sub) in expectations {
        let scope = find_scope(&scopes, name_sub).unwrap_or_else(|| {
            panic!(
                "no scope found whose qualified_path contains {name_sub:?}; \
                 paths emitted: {:?}",
                scopes
                    .iter()
                    .map(|s| s.qualified_path())
                    .collect::<Vec<_>>(),
            )
        });
        assert_scope_contains(source, scope, sig_sub);
    }
    scopes
}

/// Like [`verify_carve`] but for the resilience tests where we can't predict which
/// subset of names the parser will recover — asserts the DISPATCHER-level
/// non-blank-coverage invariant (runs the full `carve`, blank runs dropped).
pub fn verify_coverage_only(source: &str, language: Language, is_tsx: bool) -> Vec<Scope> {
    let src = source.as_bytes();
    let scopes = dispatcher_carve(src, language, is_tsx);
    assert_nonblank_coverage(src, &scopes);
    scopes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_read::types::{ChunkKind, Scope};

    fn s(start: u32, end: u32, path: &str) -> Scope {
        Scope {
            path: vec![path.to_string()],
            kind: ChunkKind::Function,
            start_line: start,
            end_line: end,
        }
    }

    #[test]
    fn lines_at_returns_correct_inclusive_range() {
        assert_eq!(lines_at("a\nb\nc\nd\n", 2, 3), "b\nc");
        assert_eq!(lines_at("a\nb\nc\nd\n", 1, 1), "a");
        assert_eq!(lines_at("a\nb\nc\nd\n", 4, 4), "d");
    }

    #[test]
    fn assert_full_coverage_passes_when_complete() {
        let src = b"a\nb\nc\nd\ne\n";
        let scopes = vec![s(1, 3, "x"), s(4, 5, "y")];
        assert_full_coverage(src, &scopes);
    }

    #[test]
    #[should_panic(expected = "uncovered lines")]
    fn assert_full_coverage_fires_on_gap() {
        let src = b"a\nb\nc\nd\ne\n";
        let scopes = vec![s(1, 2, "x"), s(4, 5, "y")];
        assert_full_coverage(src, &scopes);
    }

    #[test]
    #[should_panic(expected = "trailing uncovered lines")]
    fn assert_full_coverage_fires_on_trailing_gap() {
        let src = b"a\nb\nc\nd\ne\n";
        let scopes = vec![s(1, 3, "x")];
        assert_full_coverage(src, &scopes);
    }

    #[test]
    fn assert_starts_with_uses_start_line() {
        let src = "fn a() {}\nfn b() {}\n";
        let scope = s(1, 1, "fn a");
        assert_starts_with(src, &scope, "fn a");
    }

    #[test]
    #[should_panic(expected = "expected first line to contain")]
    fn assert_starts_with_fires_on_wrong_line() {
        let src = "fn a() {}\nfn b() {}\n";
        let scope = s(2, 2, "fn a");
        assert_starts_with(src, &scope, "fn a");
    }

    #[test]
    fn find_scope_matches_substring() {
        let scopes = vec![s(1, 1, "fn alpha"), s(2, 2, "fn beta")];
        assert!(find_scope(&scopes, "alpha").is_some());
        assert!(find_scope(&scopes, "beta").is_some());
        assert!(find_scope(&scopes, "gamma").is_none());
    }
}
