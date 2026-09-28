//! `old` found character for character.

use super::{ambiguous, Matched, ReplaceError, Replaced};

/// Replace the exact occurrences of `old`. `None` when `old` is not in the
/// file exactly — the caller then tries the lines indentation aside, which is
/// also where an edit already applied is recognised.
///
/// **An `old` that opens with indentation matches only at a line's start.**
/// Found partway into a deeper indentation — `"  foo();"` inside
/// `"    foo();"` — it is the right line quoted at the wrong depth, and a
/// replacement spliced in there would put every line after its first at the
/// depth the model wrote rather than the file's. The line match re-indents.
pub(super) fn apply(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<Option<Replaced>, ReplaceError> {
    let at = outstanding(content, old, new);
    let found = overlapping(content, old, new);
    match at.len() {
        0 => Ok(None),
        _ if found > 1 && !replace_all => Err(ReplaceError::Ambiguous(ambiguous(found))),
        n => {
            let mut out = String::with_capacity(content.len() + n * new.len());
            let mut from = 0;
            for start in &at {
                out.push_str(&content[from..*start]);
                out.push_str(new);
                from = start + old.len();
            }
            out.push_str(&content[from..]);
            Ok(Some(Replaced {
                content: out,
                replacements: n,
                already_applied: false,
                matched: Matched::Exact,
            }))
        }
    }
}

/// Where `old` occurs, non-overlapping, less every occurrence that lies inside
/// an occurrence of `new` — when `new` contains `old`, those are the edit's
/// own result — and, for an indented `old`, every one that does not start a
/// line.
fn outstanding(content: &str, old: &str, new: &str) -> Vec<usize> {
    let done = covered_by(content, old, new);
    content
        .match_indices(old)
        .map(|(start, _)| start)
        .filter(|&start| !inside(&done, start, old.len()))
        .filter(|&start| aligned(content, old, start))
        .collect()
}

/// How many times `old` occurs outside the edit's own result, overlapping
/// occurrences counted apart: `aa` in `aaa` is two places, not one.
fn overlapping(content: &str, old: &str, new: &str) -> usize {
    let done = covered_by(content, old, new);
    let mut count = 0;
    let mut from = 0;
    while let Some(i) = content[from..].find(old) {
        let start = from + i;
        if !inside(&done, start, old.len()) && aligned(content, old, start) {
            count += 1;
        }
        // Past this occurrence's first character, onto a char boundary.
        from = start + content[start..].chars().next().map_or(1, char::len_utf8);
    }
    count
}

/// The spans `new` occupies, when it contains `old`.
fn covered_by(content: &str, old: &str, new: &str) -> Vec<(usize, usize)> {
    if !new.contains(old) {
        return Vec::new();
    }
    content
        .match_indices(new)
        .map(|(start, found)| (start, start + found.len()))
        .collect()
}

fn inside(spans: &[(usize, usize)], start: usize, len: usize) -> bool {
    spans
        .iter()
        .any(|&(from, to)| from <= start && start + len <= to)
}

/// Whether an occurrence of `old` at `start` may stand: anywhere, unless `old`
/// opens with indentation, which must then start a line.
fn aligned(content: &str, old: &str, start: usize) -> bool {
    !old.starts_with([' ', '\t']) || start == 0 || content[..start].ends_with('\n')
}
