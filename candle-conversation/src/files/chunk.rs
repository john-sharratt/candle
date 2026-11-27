//! A section of a file: the lines it holds and its text.

use super::key::LineSpan;

/// Lines [`Chunk::lines`] of a file and the section that holds them: its
/// whole sealed text, heading and closing included (`files::render`), so a
/// section placed alone reads as `file_read`'s answer for its range.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Chunk {
    pub lines: LineSpan,
    pub text: String,
}
