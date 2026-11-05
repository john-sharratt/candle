//! What an edit changed: the lines it took and the lines it left.
//!
//! A whole-part replace is retyped by the caller, and a retyped part can drop
//! a character on a line nobody meant to touch — a `+` and a `-` out of a
//! regex, once. The parts around it read the same either way, so the edit
//! answer says plainly which lines moved, and the slip is seen while it is
//! still one line to mend.
//!
//! The diff is [`zend_vfs::file_delta::splices`] — the same one the session
//! layer records the delta with — so what the answer shows and what the log
//! holds are the same lines, and there is one diff in the workspace.

use zend_vfs::file_delta::{splices, Splice};

/// The most lines a diff shows, change lines and separators together.
const MAX_LINES: usize = 40;

/// The line standing for unchanged lines between two runs of change.
const GAP: &str = "…\n";

/// One line of the rendered diff.
enum Line {
    Removed(String),
    Added(String),
    Gap,
}

impl Line {
    /// Whether this line is a change rather than a gap.
    fn is_change(&self) -> bool {
        matches!(self, Self::Removed(_) | Self::Added(_))
    }

    fn text(&self) -> String {
        match self {
            Self::Removed(l) => format!("- {}\n", bare(l)),
            Self::Added(l) => format!("+ {}\n", bare(l)),
            Self::Gap => GAP.to_string(),
        }
    }
}

/// The lines `before` changed into `after`, in file order: each removed line
/// prefixed `- `, each added line `+ `, a lone `…` where unchanged lines lie
/// between two runs. Nothing when nothing changed. Past [`MAX_LINES`] the rest
/// gives way to `… and N more changed lines`.
///
/// A splice's shared lines are not changes: a pure insertion is pinned to the
/// line after it — that line removed and put back — so both sides are trimmed
/// of the lines they agree on before anything is printed.
pub(crate) fn changed(before: &str, after: &str) -> String {
    let mut lines: Vec<Line> = Vec::new();
    // Where in `before` the last run ended: a later run starting beyond it
    // means unchanged lines lie between the two.
    let mut ended: Option<usize> = None;
    for splice in splices(before, after) {
        let (at, removed, inserted) = trim(&splice);
        if ended.is_some_and(|e| at > e) {
            lines.push(Line::Gap);
        }
        ended = Some(at + removed.len());
        lines.extend(
            removed
                .split_inclusive('\n')
                .map(|l| Line::Removed(l.to_string())),
        );
        lines.extend(
            inserted
                .split_inclusive('\n')
                .map(|l| Line::Added(l.to_string())),
        );
    }
    if lines.len() <= MAX_LINES {
        return lines.iter().map(Line::text).collect();
    }
    let more = lines[MAX_LINES..].iter().filter(|l| l.is_change()).count();
    let mut out: String = lines[..MAX_LINES].iter().map(Line::text).collect();
    // The trailer says more follows; a gap standing at the cut says nothing
    // the trailer does not already say.
    if out.ends_with(GAP) {
        out.truncate(out.len() - GAP.len());
    }
    out.push_str(&format!("… and {more} more changed lines\n"));
    out
}

/// The splice's own change: where it begins in `before`, the lines it takes,
/// the lines it leaves — with the lines both sides agree on taken back out of
/// both ends, and `at` moved past the ones agreed at the front.
///
/// Two lines agree when their text agrees apart from the terminator: the last
/// line of a file with no trailing newline differs from the same line with one
/// by nothing a reader would call a change.
fn trim(splice: &Splice) -> (usize, &str, &str) {
    let (mut removed, mut inserted) = (splice.removed.as_str(), splice.inserted.as_str());
    let mut at = splice.at;
    loop {
        let (r, i) = (first_line(removed), first_line(inserted));
        if r.is_empty() || i.is_empty() || bare(r) != bare(i) {
            break;
        }
        at += r.len();
        removed = &removed[r.len()..];
        inserted = &inserted[i.len()..];
    }
    loop {
        let (r, i) = (last_line(removed), last_line(inserted));
        if r.is_empty() || i.is_empty() || bare(r) != bare(i) {
            break;
        }
        removed = &removed[..removed.len() - r.len()];
        inserted = &inserted[..inserted.len() - i.len()];
    }
    (at, removed, inserted)
}

/// The first line of `s`, its terminator included; empty when there is none.
fn first_line(s: &str) -> &str {
    s.split_inclusive('\n').next().unwrap_or("")
}

/// The last line of `s`, its terminator included; empty when there is none.
fn last_line(s: &str) -> &str {
    match s.strip_suffix('\n') {
        Some(rest) => &s[rest.rfind('\n').map_or(0, |i| i + 1)..],
        None => s,
    }
}

/// A line as shown: without its terminator.
fn bare(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **One word in one line**: the line as it was and as it is, nothing else.
    #[test]
    fn one_word_changed_shows_the_line_once_either_side() {
        assert_eq!(
            changed("let wide = a + b;\n", "let wide = a - b;\n"),
            "- let wide = a + b;\n+ let wide = a - b;\n"
        );
    }

    /// **Two lines inserted** show as two added lines: the line the insertion
    /// is pinned to for replay is not a change and is not shown.
    #[test]
    fn two_lines_inserted_show_as_two_added_lines() {
        assert_eq!(
            changed("one\nthree\n", "one\ntwo a\ntwo b\nthree\n"),
            "+ two a\n+ two b\n"
        );
    }

    /// **The case this exists for**: a whole part retyped, the line intended
    /// changed, and a `+` and a `-` dropped from a line nobody meant to touch.
    /// Both lines are named, and the unchanged line between them is a gap.
    #[test]
    fn a_retyped_part_names_the_line_it_did_not_mean_to_change() {
        let before = "fn window(t: &str) -> bool {\n    let head = t.trim_start_matches('-');\n    head.len() >= 3 && head.ends_with(\"+-\")\n}\n";
        let after = "fn window(t: &str, n: usize) -> bool {\n    let head = t.trim_start_matches('-');\n    head.len() >= 3 && head.ends_with(\"\")\n}\n";
        assert_eq!(
            changed(before, after),
            "- fn window(t: &str) -> bool {\n\
             + fn window(t: &str, n: usize) -> bool {\n\
             …\n\
             -     head.len() >= 3 && head.ends_with(\"+-\")\n\
             +     head.len() >= 3 && head.ends_with(\"\")\n"
        );
    }

    /// **Past the cap the rest is counted, not shown**: forty lines, then how
    /// many changes were left out.
    #[test]
    fn the_cap_shows_forty_lines_and_counts_the_rest() {
        let before: String = (0..60).map(|i| format!("line {i}\n")).collect();
        let after: String = (0..60).map(|i| format!("x{i}\n")).collect();
        let shown = changed(&before, &after);
        let mut want = String::new();
        for i in 0..MAX_LINES {
            want.push_str(&format!("- line {i}\n"));
        }
        want.push_str("… and 80 more changed lines\n");
        assert_eq!(shown, want);
    }

    /// Nothing changed — nothing to say.
    #[test]
    fn nothing_changed_shows_nothing() {
        assert_eq!(changed("same\ntext\n", "same\ntext\n"), "");
        assert_eq!(changed("", ""), "");
    }

    /// A file that ends without a newline keeps its last line whole, and a
    /// line appended to it reads as the one line added.
    #[test]
    fn a_last_line_without_a_newline_is_a_line() {
        assert_eq!(changed("one\ntwo", "one\ntwo\nthree\n"), "+ three\n");
    }
}
