//! A chunk of a file: whole lines, each ending its line.

use super::key::LineSpan;

/// Lines [`Chunk::lines`] of a file and their text. Every line in `text` ends
/// with its `\n`, the file's last line included even when the file itself
/// does not end with one, so a closing fence or the next chunk always starts
/// a line of its own.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Chunk {
    pub lines: LineSpan,
    pub text: String,
}
