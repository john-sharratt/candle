//! The knowledge base's notation: Qwen-Agent's (`qwen_agent/agents/
//! assistant.py`, `KNOWLEDGE_TEMPLATE_EN` and `KNOWLEDGE_SNIPPET_EN`), and the
//! pieces the system prompt seals it as.
//!
//! The knowledge base is `# Knowledge Base`, then each file placed as one
//! snippet — its heading, its placed text in a fence — the snippets joined by
//! a blank line. [`knowledge_text`] is that text; [`assemble_pieces`] is the
//! same text cut into the pieces the prompt seals, each tokenized on its own.
//!
//! Where a piece meets a chunk is where its tokens could merge across the
//! join, so every piece is cut where Qwen's vocabulary does not merge: a line
//! keeps its own `\n` (a punctuation mark and the newlines after it are one
//! pre-token, so a `}` cannot be parted from its `\n`), and nothing that
//! follows a chunk starts with a newline (it would join the chunk's last one).
//! So the fence that closes a file opens its piece with the backticks, the gap
//! between two runs of a file opens with its dots, and a chunk's last line,
//! ending in `\n`, puts the fence on a line of its own.

use super::chunk::Chunk;

/// The knowledge base's title, as [`knowledge_text`] opens with it.
pub const KNOWLEDGE_BASE: &str = "# Knowledge Base\n\n";

/// The piece that opens the knowledge base in the system prompt: the title,
/// after the blank line that parts it from the section before, which ends its
/// own line.
pub const FILES_OPEN: &str = "\n# Knowledge Base\n\n";

/// The piece between two runs of one file's placed lines that are not
/// contiguous: the dots on a line of their own, then a blank line.
pub const GAP: &str = "...\n\n";

/// The fence that closes a file's text.
pub const FENCE_CLOSE: &str = "```";

/// The piece that closes a file another file follows: the fence, then the
/// blank line the snippets are joined by.
pub const CLOSING_JOIN: &str = "```\n\n";

/// The piece that closes the last file: the fence, ending its line.
pub const CLOSING_LAST: &str = "```\n";

/// What joins two snippets in [`knowledge_text`].
pub const SNIPPET_JOIN: &str = "\n\n";

/// The heading piece of the file at `path`: its source line, a blank line, and
/// the fence its text opens in.
pub fn heading(path: &str) -> String {
    format!("## The content from {path}:\n\n```\n")
}

/// A file as the knowledge base places it: its workspace path, and the chunks
/// of it placed, in line order and not overlapping.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RenderedFile {
    pub path: String,
    pub chunks: Vec<Chunk>,
}

/// `file`'s placed text as pieces: its chunks in order, a [`GAP`] between two
/// that are not contiguous. A contiguous run is the file's own text, since its
/// chunks are consecutive slices of it.
fn body_pieces(file: &RenderedFile) -> Vec<&str> {
    let mut out = Vec::new();
    for (i, chunk) in file.chunks.iter().enumerate() {
        if i > 0 {
            let before = file.chunks[i - 1].lines;
            debug_assert!(
                chunk.lines.first > before.last,
                "{}: chunks out of line order",
                file.path
            );
            if !chunk.lines.continues(before) {
                out.push(GAP);
            }
        }
        out.push(chunk.text.as_str());
    }
    out
}

/// The knowledge base as Qwen-Agent writes it: [`KNOWLEDGE_BASE`], then each
/// file as `## The content from {path}:`, a blank line, and its placed text
/// in a fence, the snippets joined by a blank line. No leading newline and no
/// trailing one: [`assemble_pieces`] adds both.
pub fn knowledge_text(files: &[RenderedFile]) -> String {
    let snippets: Vec<String> = files
        .iter()
        .map(|f| {
            format!(
                "{}{}{FENCE_CLOSE}",
                heading(&f.path),
                body_pieces(f).concat()
            )
        })
        .collect();
    format!("{KNOWLEDGE_BASE}{}", snippets.join(SNIPPET_JOIN))
}

/// The pieces the system prompt seals the knowledge base as: [`FILES_OPEN`],
/// then per file its [`heading`], its chunks with a [`GAP`] between runs that
/// are not contiguous, and [`CLOSING_JOIN`], or [`CLOSING_LAST`] for the last.
/// Joined, they are `"\n" + knowledge_text(files) + "\n"`.
pub fn assemble_pieces(files: &[RenderedFile]) -> Vec<String> {
    let mut out = vec![FILES_OPEN.to_string()];
    for (i, f) in files.iter().enumerate() {
        out.push(heading(&f.path));
        out.extend(body_pieces(f).into_iter().map(str::to_string));
        let closing = if i + 1 == files.len() {
            CLOSING_LAST
        } else {
            CLOSING_JOIN
        };
        out.push(closing.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::key::LineSpan;

    fn chunk(first: u32, last: u32, text: &str) -> Chunk {
        Chunk {
            lines: LineSpan { first, last },
            text: text.to_string(),
        }
    }

    fn file(path: &str, chunks: Vec<Chunk>) -> RenderedFile {
        RenderedFile {
            path: path.to_string(),
            chunks,
        }
    }

    /// **A file placed whole renders as Q1 rendered it** — `{body}` then `\n`
    /// and the fence, Q1's body being the file's text short of the `\n` that
    /// ends its last line — and its pieces join to the same text, a newline
    /// either side.
    #[test]
    fn a_whole_file_renders_as_q1_rendered_it() {
        let f = file(
            "candle/a.rs",
            vec![chunk(1, 1, "fn x() {\n"), chunk(2, 2, "}\n")],
        );
        let q1 = "# Knowledge Base\n\n## The content from candle/a.rs:\n\n```\nfn x() {\n}\n```";
        assert_eq!(knowledge_text(std::slice::from_ref(&f)), q1);
        assert_eq!(
            assemble_pieces(std::slice::from_ref(&f)),
            vec![
                "\n# Knowledge Base\n\n",
                "## The content from candle/a.rs:\n\n```\n",
                "fn x() {\n",
                "}\n",
                "```\n",
            ]
        );
        assert_eq!(assemble_pieces(&[f]).concat(), format!("\n{q1}\n"));
    }

    /// **A file with a gap keeps one heading and marks the gap** between the
    /// two runs, a contiguous run joined as the file's own text.
    #[test]
    fn a_file_with_a_gap_keeps_one_heading_and_marks_the_gap() {
        let f = file(
            "b.md",
            vec![
                chunk(1, 2, "one\ntwo\n"),
                chunk(3, 3, "three\n"),
                chunk(9, 9, "nine\n"),
            ],
        );
        assert_eq!(
            knowledge_text(std::slice::from_ref(&f)),
            "# Knowledge Base\n\n## The content from b.md:\n\n```\none\ntwo\nthree\n...\n\nnine\n```"
        );
        assert_eq!(
            assemble_pieces(&[f]),
            vec![
                "\n# Knowledge Base\n\n",
                "## The content from b.md:\n\n```\n",
                "one\ntwo\n",
                "three\n",
                "...\n\n",
                "nine\n",
                "```\n",
            ]
        );
    }

    /// **Files are joined by a blank line and the last ends on its fence**:
    /// the first file's closing piece carries the blank line, the last's ends
    /// the fence's line and nothing more.
    #[test]
    fn files_are_joined_by_a_blank_line_and_the_last_ends_on_its_fence() {
        let files = [
            file("a.rs", vec![chunk(1, 1, "one\n")]),
            file("b/c.rs", vec![chunk(1, 1, "tw\n")]),
        ];
        let text = "# Knowledge Base\n\n## The content from a.rs:\n\n```\none\n```\n\n\
                    ## The content from b/c.rs:\n\n```\ntw\n```";
        assert_eq!(knowledge_text(&files), text);
        assert_eq!(
            assemble_pieces(&files),
            vec![
                "\n# Knowledge Base\n\n",
                "## The content from a.rs:\n\n```\n",
                "one\n",
                "```\n\n",
                "## The content from b/c.rs:\n\n```\n",
                "tw\n",
                "```\n",
            ]
        );
        assert_eq!(assemble_pieces(&files).concat(), format!("\n{text}\n"));
    }

    /// With no file placed, the knowledge base is its title alone.
    #[test]
    fn an_empty_knowledge_base_is_its_title() {
        assert_eq!(knowledge_text(&[]), "# Knowledge Base\n\n");
        assert_eq!(assemble_pieces(&[]), vec!["\n# Knowledge Base\n\n"]);
    }
}
