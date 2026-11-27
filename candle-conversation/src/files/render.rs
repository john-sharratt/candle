//! The knowledge base's notation and the pieces the system prompt seals it as
//! (`docs/placed_section_layout.md` §3.5c).
//!
//! The knowledge base is `# Knowledge Base`, then each placed section of a
//! file: its heading, `## The content from {path}:`, a blank line, and its
//! text — for a file, the body a `file_read` of the section's range returns,
//! byte for byte, so a placed section carries the part anchors and relative
//! line numbers an edit names. Every section but a file's last ends with the
//! [`CUT_MARKER`] after its closing line, so a section placed without the one
//! after it reads as continuing. Each section ends on a blank line, which
//! parts it from whatever follows.
//!
//! A section is sealed on its own, so its tokens are its text's alone. Where
//! two sections meet is where their tokens could merge across the join: every
//! section therefore ends on its blank line's `\n\n` and opens on its
//! heading's `##`, which Qwen's vocabulary never merges, and the knowledge
//! base's title ends on a blank line too.

use super::chunk::Chunk;

/// The knowledge base's title, as [`knowledge_text`] opens with it.
pub const KNOWLEDGE_BASE: &str = "# Knowledge Base\n\n";

/// The piece that opens the knowledge base in the system prompt: the title,
/// after the blank line that parts it from the section before, which ends its
/// own line.
pub const FILES_OPEN: &str = "\n# Knowledge Base\n\n";

/// What ends every section of a file but its last, after its closing line.
pub const CUT_MARKER: &str = "...\n";

/// The blank line every section ends on.
pub const SECTION_END: &str = "\n";

/// The heading every section of the file at `path` opens with: its source
/// line, then a blank line.
pub fn heading(path: &str) -> String {
    format!("## The content from {path}:\n\n")
}

/// A section of the file at `path`: its [`heading`], `body` — whole lines,
/// the last ending its `\n` — the [`CUT_MARKER`] unless it is the file's
/// `last`, and the blank line it ends on.
pub fn section_text(path: &str, body: &str, last: bool) -> String {
    debug_assert!(
        body.ends_with('\n'),
        "{path}: a section's body ends its line"
    );
    let marker = if last { "" } else { CUT_MARKER };
    format!("{}{body}{marker}{SECTION_END}", heading(path))
}

/// The one section of the folder at `path`: its listing, every line ending
/// its `\n`, in a fence.
pub fn listing_section(path: &str, listing: &str) -> String {
    let listing = if listing.is_empty() || listing.ends_with('\n') {
        listing.to_string()
    } else {
        format!("{listing}\n")
    };
    section_text(path, &format!("```\n{listing}```\n"), true)
}

/// A file as the knowledge base places it: its workspace path, and the
/// sections of it placed, each a [`Chunk`] whose text is the whole section.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RenderedFile {
    pub path: String,
    pub chunks: Vec<Chunk>,
}

/// The knowledge base: [`KNOWLEDGE_BASE`], then every placed section of
/// `files`, in order. No leading newline: [`assemble_pieces`] adds it.
pub fn knowledge_text(files: &[RenderedFile]) -> String {
    let sections: String = files
        .iter()
        .flat_map(|f| f.chunks.iter().map(|c| c.text.as_str()))
        .collect();
    format!("{KNOWLEDGE_BASE}{sections}")
}

/// The pieces the system prompt seals the knowledge base as: [`FILES_OPEN`],
/// then each placed section. Joined, they are `"\n" + knowledge_text(files)`.
pub fn assemble_pieces(files: &[RenderedFile]) -> Vec<String> {
    let mut out = vec![FILES_OPEN.to_string()];
    out.extend(
        files
            .iter()
            .flat_map(|f| f.chunks.iter().map(|c| c.text.clone())),
    );
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

    /// **A section is its heading, its body, the marker unless it is the
    /// file's last, and a blank line.**
    #[test]
    fn a_section_is_its_heading_body_marker_and_blank_line() {
        assert_eq!(
            section_text("r/a.rs", "body\nend of r/a.rs lines 1-2 of 9\n", false),
            "## The content from r/a.rs:\n\nbody\nend of r/a.rs lines 1-2 of 9\n...\n\n"
        );
        assert_eq!(
            section_text("r/a.rs", "tail\n", true),
            "## The content from r/a.rs:\n\ntail\n\n"
        );
    }

    /// **A folder is one section, its listing in a fence**, its last line
    /// given its `\n`.
    #[test]
    fn a_folder_is_one_section_its_listing_fenced() {
        assert_eq!(
            listing_section("r/src/", "a.rs 12\nb/"),
            "## The content from r/src/:\n\n```\na.rs 12\nb/\n```\n\n"
        );
    }

    /// **The pieces join to the knowledge text**, a newline ahead: the title,
    /// then every section of every file in order.
    #[test]
    fn the_pieces_join_to_the_knowledge_text() {
        let files = [
            RenderedFile {
                path: "r/a.rs".into(),
                chunks: vec![
                    chunk(1, 2, &section_text("r/a.rs", "one\n", false)),
                    chunk(3, 4, &section_text("r/a.rs", "two\n", true)),
                ],
            },
            RenderedFile {
                path: "r/b/".into(),
                chunks: vec![chunk(1, 1, &listing_section("r/b/", "c.rs 3\n"))],
            },
        ];
        let text = "# Knowledge Base\n\n\
                    ## The content from r/a.rs:\n\none\n...\n\n\
                    ## The content from r/a.rs:\n\ntwo\n\n\
                    ## The content from r/b/:\n\n```\nc.rs 3\n```\n\n";
        assert_eq!(knowledge_text(&files), text);
        assert_eq!(assemble_pieces(&files).concat(), format!("\n{text}"));
        assert_eq!(assemble_pieces(&files).len(), 4);
    }

    /// With no file placed, the knowledge base is its title alone.
    #[test]
    fn an_empty_knowledge_base_is_its_title() {
        assert_eq!(knowledge_text(&[]), "# Knowledge Base\n\n");
        assert_eq!(assemble_pieces(&[]), vec!["\n# Knowledge Base\n\n"]);
    }
}
