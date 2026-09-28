//! Search-and-replace engine behind `file_edit`.
//!
//! # The format
//!
//! An edit is two texts: `old` — the part of the file to change, quoted as it
//! stands — and `new`, what takes its place. This is the edit format models
//! are trained to emit: nothing to count, no line numbers, no per-line
//! prefixes. Why a diff format was not kept is recorded in
//! `docs/tool-system.md` (`file_edit`).
//!
//! # Matching
//!
//! **Exactly first.** `old` is looked for as a substring of the file. One
//! occurrence is replaced; several — overlapping ones counted apart — are
//! [`ReplaceError::Ambiguous`] unless every one is asked for (`replace_all`).
//! An `old` that opens with indentation must start a line: found partway into
//! deeper indentation it is the right line at the wrong depth, and the line
//! match below re-indents it.
//!
//! **Then line by line, indentation aside** ([`lines`]). When `old` is not in
//! the file exactly, it is looked for as a run of whole lines, each compared
//! with its leading and trailing whitespace trimmed. That is the mistake a
//! model actually makes — the right lines at the wrong indentation — and the
//! replacement is re-indented depth by depth ([`indent`]): each depth `old`
//! was written at becomes the one the file has there, a line deeper keeps its
//! extra levels — in tabs where the file uses tabs — and a quoted depth that
//! stands at two depths in the file is refused. Several occurrences,
//! overlapping ones counted apart, are ambiguous here too. Nothing looser is
//! tried: a line that differs in anything but its surrounding whitespace is a
//! different line.
//!
//! # Already applied
//!
//! Sending the same edit twice is a no-op rather than a second edit, so the
//! call is safe to replay. When `old` is absent — exactly and as lines — and
//! `new` stands in the file **as whole lines**, the edit is reported already
//! applied and nothing changes. Whole lines, because a bare substring of `new`
//! is found by accident (a lone `}` is in every file): a mis-quoted `old`
//! whose `new` happened to occur would otherwise be a success that wrote
//! nothing. An edit inside a line (`localhost` → `0.0.0.0`) is therefore
//! recognised as applied only when its `new` is a whole line; re-sent
//! otherwise, it is [`ReplaceError::Unmatched`], and still writes nothing.
//!
//! When `new` contains `old` — `retries = 3` becoming `retries = 30`, or a
//! line inserted after the one quoted — an occurrence of `old` that sits
//! inside an occurrence of `new` is the edit's own result, not something still
//! to change, and is not counted: without that, a re-send would turn `30`
//! into `300`.
//!
//! An edit that deletes (`new` empty) leaves no trace to recognise, so a
//! missing `old` is [`ReplaceError::Unmatched`] for it — never a success that
//! wrote nothing. Nor does an edit whose `new` is made only of `old`'s own
//! lines — a deletion that quotes its context, a re-indent: `new` stands in
//! the file whether or not it was made.
//!
//! # Line endings
//!
//! `old` and `new` are taken in the file's line ending: in a CRLF file an LF
//! edit is matched and written as CRLF. The line-by-line match ignores the
//! terminator altogether; every line it does not replace keeps its own ending
//! — a mixed file stays mixed — the lines it writes take the file's first,
//! and the file's trailing newline, or its absence, is kept.

mod exact;
mod indent;
mod lines;
#[cfg(test)]
mod tests;

use std::error::Error;
use std::fmt;

/// How `old` was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matched {
    /// Character for character.
    Exact,
    /// As whole lines with their indentation ignored; the replacement was
    /// re-indented to the file's.
    Indentation,
}

impl Matched {
    pub fn as_str(self) -> &'static str {
        match self {
            Matched::Exact => "exact",
            Matched::Indentation => "indentation",
        }
    }
}

/// An edited file: the content to store, and what the edit did to get it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replaced {
    pub content: String,
    /// Occurrences replaced. Zero when the edit was already applied.
    pub replacements: usize,
    /// The edit's result was already in the file, so nothing changed.
    pub already_applied: bool,
    pub matched: Matched,
}

/// Why an edit did not apply. Each variant carries the message handed to the
/// model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplaceError {
    /// The edit cannot be made at all: `old` is empty, or `new` is the same.
    Invalid(String),
    /// `old` occurs more than once and only one replacement was asked for.
    Ambiguous(String),
    /// `old` is not in the file, exactly or indentation aside, and the edit
    /// is not already applied — or it is there only at indentations `new`
    /// cannot be carried over from.
    Unmatched(String),
}

impl fmt::Display for ReplaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplaceError::Invalid(why)
            | ReplaceError::Ambiguous(why)
            | ReplaceError::Unmatched(why) => f.write_str(why),
        }
    }
}

impl Error for ReplaceError {}

/// Replace `old` with `new` in `content` — the one occurrence, or every one
/// when `replace_all`.
pub fn apply(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<Replaced, ReplaceError> {
    if old.trim().is_empty() {
        return Err(ReplaceError::Invalid(
            "`old_text` is empty — quote the lines to change, exactly as the file has them"
                .to_string(),
        ));
    }
    let (old, new) = lines::in_file_eol(content, old, new);
    if old == new {
        return Err(ReplaceError::Invalid(
            "`old_text` and `new_text` are the same, so there is nothing to change".to_string(),
        ));
    }
    match exact::apply(content, &old, &new, replace_all)? {
        Some(done) => Ok(done),
        None => lines::apply(content, &old, &new, replace_all),
    }
}

/// The message for `old` found `count` times with one replacement asked for.
fn ambiguous(count: usize) -> String {
    format!(
        "`old_text` occurs {count} times in the file — include more of the lines around the one \
         to change so it matches once, or set `replace_all` to change all {count}"
    )
}
