//! `old` found as whole lines with their indentation ignored, and the line
//! endings both matches work in.

use super::indent::Depths;
use super::{ambiguous, Matched, ReplaceError, Replaced};

/// A line's terminator, kept apart from its text so matching can ignore it
/// and rebuilding can restore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eol {
    /// The last line of a file that does not end with a newline.
    None,
    Lf,
    Crlf,
}

impl Eol {
    fn as_str(self) -> &'static str {
        match self {
            Eol::None => "",
            Eol::Lf => "\n",
            Eol::Crlf => "\r\n",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Line {
    text: String,
    eol: Eol,
}

/// `old` and `new` in the file's line ending: CRLF throughout when the file
/// has any CRLF, LF otherwise.
pub(super) fn in_file_eol(content: &str, old: &str, new: &str) -> (String, String) {
    let lf = |s: &str| s.replace("\r\n", "\n");
    let (old, new) = (lf(old), lf(new));
    if content.contains("\r\n") {
        (old.replace('\n', "\r\n"), new.replace('\n', "\r\n"))
    } else {
        (old, new)
    }
}

/// Replace `old` found as whole lines, indentation aside, re-indenting `new`
/// to the file's indentation — or report the edit already applied.
pub(super) fn apply(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<Replaced, ReplaceError> {
    let mut file = split(content);
    let new_text = new;
    let old = text_lines(old);
    let new = text_lines(new);
    // As in the exact match: a run of `old` inside a run of `new` is the
    // edit's own result, not something still to change.
    let done = find(&file, &new, Step::PastEach);
    let outstanding = |starts: Vec<usize>| -> Vec<usize> {
        starts
            .into_iter()
            .filter(|&start| {
                !done
                    .iter()
                    .any(|&from| from <= start && start + old.len() <= from + new.len())
            })
            .collect()
    };
    let found = outstanding(find(&file, &old, Step::PastEach));
    if found.is_empty() {
        // Already applied only when `new` stands in the file as whole lines,
        // says something, and adds a line `old` does not have. A bare
        // substring, or lines of nothing but punctuation — a lone `}` is in
        // every file — are found by accident; and a `new` made only of `old`'s
        // own lines — a deletion quoting its context, a re-indent — stands in
        // the file whether or not the edit was made. Each would turn a
        // mis-quoted `old` into a success that wrote nothing.
        let says_something = new
            .iter()
            .any(|line| line.chars().any(char::is_alphanumeric));
        let adds_a_line = new
            .iter()
            .any(|line| !line.trim().is_empty() && !old.iter().any(|o| same(o, line)));
        if !done.is_empty() && says_something && adds_a_line {
            return Ok(Replaced {
                content: content.to_string(),
                replacements: 0,
                already_applied: true,
                matched: if content.contains(new_text) {
                    Matched::Exact
                } else {
                    Matched::Indentation
                },
            });
        }
        return Err(ReplaceError::Unmatched(unmatched(&file, &old)));
    }
    // Ambiguity counts overlapping occurrences too, as the exact match does:
    // `}` `}` found in three closing braces is two places, not one.
    if !replace_all {
        let everywhere = outstanding(find(&file, &old, Step::ByLine)).len();
        if everywhere > 1 {
            return Err(ReplaceError::Ambiguous(ambiguous(everywhere)));
        }
    }

    let eol = default_eol(&file);
    // The file's trailing newline, or its absence, belongs to the file rather
    // than to whichever line ends up last.
    let trailing = file.last().map_or(Eol::None, |line| line.eol);
    // Every window's indentation is worked out before any is replaced, so a
    // window that cannot be refuses the whole edit.
    let depths: Vec<Depths> = found
        .iter()
        .map(|&start| {
            let window: Vec<&str> = file[start..start + old.len()]
                .iter()
                .map(|line| line.text.as_str())
                .collect();
            Depths::of(&old, &window)
        })
        .collect::<Result<_, _>>()?;
    // Back to front, so each window's position still holds when it is reached.
    for (&start, depths) in found.iter().zip(&depths).rev() {
        let replacement = new.iter().map(|line| Line {
            text: depths.reindent(line),
            eol,
        });
        file.splice(start..start + old.len(), replacement.collect::<Vec<_>>());
    }
    // Every surviving line keeps its own ending — a mixed file stays mixed —
    // and only the file's end is settled: the last line takes the file's
    // trailing newline or its absence, and a line that used to be last and is
    // not any more gains one.
    let last = file.len().saturating_sub(1);
    for (i, line) in file.iter_mut().enumerate() {
        if i == last {
            line.eol = trailing;
        } else if line.eol == Eol::None {
            line.eol = eol;
        }
    }
    Ok(Replaced {
        content: file
            .iter()
            .map(|line| format!("{}{}", line.text, line.eol.as_str()))
            .collect(),
        replacements: found.len(),
        already_applied: false,
        matched: Matched::Indentation,
    })
}

/// The file's lines with their terminators.
fn split(content: &str) -> Vec<Line> {
    content
        .split_inclusive('\n')
        .map(|raw| match raw.strip_suffix("\r\n") {
            Some(text) => Line {
                text: text.to_string(),
                eol: Eol::Crlf,
            },
            None => match raw.strip_suffix('\n') {
                Some(text) => Line {
                    text: text.to_string(),
                    eol: Eol::Lf,
                },
                None => Line {
                    text: raw.to_string(),
                    eol: Eol::None,
                },
            },
        })
        .collect()
}

/// The terminator a new line takes: the file's first, or LF for a file with
/// none.
fn default_eol(file: &[Line]) -> Eol {
    file.iter()
        .map(|line| line.eol)
        .find(|eol| *eol != Eol::None)
        .unwrap_or(Eol::Lf)
}

/// An edit text's lines, without their terminators or the empty element a
/// final newline leaves behind.
fn text_lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

/// Whether two lines are the same line, their surrounding whitespace aside.
fn same(a: &str, b: &str) -> bool {
    a.trim() == b.trim()
}

/// How [`find`] moves on from a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Past the whole match: occurrences that do not overlap, to replace.
    PastEach,
    /// One line: every occurrence, overlapping ones apart, to count.
    ByLine,
}

/// Where `want` occurs as a run of whole lines, in order. Nothing for a
/// `want` with no line that is not blank: blank lines alone match everywhere
/// and locate nothing.
fn find(file: &[Line], want: &[&str], step: Step) -> Vec<usize> {
    if want.iter().all(|line| line.trim().is_empty()) {
        return Vec::new();
    }
    let mut found = Vec::new();
    let mut at = 0;
    while at + want.len() <= file.len() {
        let here = file[at..at + want.len()]
            .iter()
            .zip(want)
            .all(|(line, wanted)| same(&line.text, wanted));
        if here {
            found.push(at);
        }
        at += match (here, step) {
            (true, Step::PastEach) => want.len(),
            _ => 1,
        };
    }
    found
}

/// Why `old` was not found, and where its first line is if that much is.
fn unmatched(file: &[Line], old: &[&str]) -> String {
    let base = "`old_text` is not in the file — not exactly, and not with its indentation \
                ignored. Copy the lines to change exactly as the file has them, without the \
                line numbers `file_read` shows beside them";
    let first = old
        .iter()
        .find(|line| !line.trim().is_empty())
        .map_or("", |line| line.trim());
    let at: Vec<String> = file
        .iter()
        .enumerate()
        .filter(|(_, line)| same(&line.text, first))
        .map(|(i, _)| (i + 1).to_string())
        .collect();
    match at.as_slice() {
        [] => format!(
            "{base}. Its first line, `{first}`, is not in the file at all — read the file again"
        ),
        [n] => format!(
            "{base}. Its first line is line {n} of the file, but the lines after it differ from \
             the file's"
        ),
        many => format!(
            "{base}. Its first line is at lines {} of the file, but at each the lines after it \
             differ from the file's",
            many.join(", ")
        ),
    }
}
