//! Shared validation helpers for parser test suites.
//!
//! The two invariants every parser/dispatcher test enforces:
//!
//! 1. **The partition** — the units of a whole carve hold every non-blank
//!    line exactly once, in order ([`assert_partition`]).
//! 2. **Boundary correctness** — every named scope holds the line with its
//!    signature (a `fn alpha` scope contains the line with `fn alpha`).
//!    Catches parsers that find the name but mis-locate the body.
//!
//! Tests call [`verify_carve`] with the source string, the
//! [`Language`], the `is_tsx` flag, and a slice of
//! `(name_substring, expected_first_line_substring)` pairs. The helper
//! checks the parser's raw scopes against them and returns the scope list for
//! any further assertions the test wants to make; [`verify_coverage_only`]
//! runs the whole carve and checks [`assert_partition`].

use super::types::Scope;
use super::{blank, carve as dispatcher_carve, carve_raw, split_lines};
use crate::Language;

/// The language a parser test names: TypeScript's TSX dialect when `is_tsx`.
fn dialect(language: Language, is_tsx: bool) -> Language {
    match (language, is_tsx) {
        (Language::TypeScript, true) => Language::Tsx,
        (l, _) => l,
    }
}

/// Assert `units` partition `source`'s non-blank lines: in order, never
/// overlapping, each starting and ending on a non-blank line, and together
/// holding every non-blank line.
pub fn assert_partition(source: &str, units: &[Scope]) {
    let lines = split_lines(source);
    let mut covered = vec![false; lines.len() + 1];
    let mut last_end = 0u32;
    for u in units {
        assert!(
            u.start_line > last_end,
            "{:?} at {}-{} overlaps or precedes the unit before it (ends {last_end})",
            u.qualified_path(),
            u.start_line,
            u.end_line
        );
        assert!(u.start_line <= u.end_line && u.end_line as usize <= lines.len());
        assert!(
            !blank(lines[u.start_line as usize - 1]) && !blank(lines[u.end_line as usize - 1]),
            "{:?} at {}-{} starts or ends on a blank line",
            u.qualified_path(),
            u.start_line,
            u.end_line
        );
        for l in u.start_line..=u.end_line {
            covered[l as usize] = true;
        }
        last_end = u.end_line;
    }
    for (i, line) in lines.iter().enumerate() {
        assert!(
            blank(line) || covered[i + 1],
            "line {} ({line:?}) is in no unit",
            i + 1
        );
    }
}

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

/// Run a language's parser and check the scopes it draws.
///
/// `expectations` is a list of `(name_substring, expected_substring)`
/// pairs. For each pair, the helper finds the first scope whose path
/// contains `name_substring` and asserts `expected_substring` is on one of
/// its lines.
///
/// Returns the raw scope list so callers can do additional per-scope
/// assertions.
pub fn verify_carve(
    source: &str,
    language: Language,
    is_tsx: bool,
    expectations: &[(&str, &str)],
) -> Vec<Scope> {
    // Parser-level: verify EXTRACTION against the RAW scopes, so per-item
    // boundaries are checked as the parser drew them. The partition is a
    // dispatcher invariant — see `verify_coverage_only`.
    let scopes = carve_raw(source, dialect(language, is_tsx));
    assert!(
        !scopes.is_empty() || source.is_empty(),
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
/// subset of names the parser will recover — runs the whole carve and asserts
/// it is a partition ([`assert_partition`]).
pub fn verify_coverage_only(source: &str, language: Language, is_tsx: bool) -> Vec<Scope> {
    let scopes = dispatcher_carve(source, dialect(language, is_tsx));
    assert_partition(source, &scopes);
    scopes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carve::types::{ChunkKind, Scope};

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
    fn assert_partition_passes_when_complete() {
        let scopes = vec![s(1, 3, "x"), s(5, 5, "y")];
        assert_partition("a\nb\nc\n\ne\n", &scopes);
    }

    #[test]
    #[should_panic(expected = "is in no unit")]
    fn assert_partition_fires_on_a_missing_line() {
        assert_partition("a\nb\nc\n", &[s(1, 2, "x")]);
    }

    #[test]
    #[should_panic(expected = "overlaps")]
    fn assert_partition_fires_on_overlap() {
        assert_partition("a\nb\nc\n", &[s(1, 2, "x"), s(2, 3, "y")]);
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
