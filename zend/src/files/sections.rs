//! A file of the workspace cut into the knowledge base's sections
//! (`docs/placed_section_layout.md` §3.5c): where `file_read` pages it — runs
//! of whole parts, at most `PAGE_LINES` lines each ([`windows`]) — each
//! section the parts and closing line a `file_read` of its range returns,
//! byte for byte ([`range_read`]), framed by its heading and, short of the
//! file's end, the cut marker ([`section_text`]).

use candle_conversation::files::render::{listing_section, section_text};
use candle_conversation::files::{Chunk, FileUnit, LineSpan};
use doc_units::Doc;
use zend_tools::tools::file::render::{range_read, windows};

/// The sections of `text`, the file or folder at the workspace path `path`
/// (its repository first). A file is a section per [`windows`] range, a file
/// with no text one section saying so; a folder is its listing, one section.
pub fn sections(path: &str, unit: FileUnit, text: &str) -> anyhow::Result<Vec<Chunk>> {
    if unit == FileUnit::Folder {
        let lines = text.lines().count().max(1) as u32;
        return Ok(vec![Chunk {
            lines: LineSpan {
                first: 1,
                last: lines,
            },
            text: listing_section(path, text),
        }]);
    }
    let (repo, rel) = path
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("{path:?} names no repository"))?;
    let doc = Doc::new(rel, text);
    let ranges = windows(rel, text);
    if ranges.is_empty() {
        let lines = doc.line_count().max(1);
        return Ok(vec![Chunk {
            lines: LineSpan {
                first: 1,
                last: lines as u32,
            },
            text: section_text(path, &range_read(repo, rel, &doc, 1, 1), true),
        }]);
    }
    let last = ranges.len() - 1;
    Ok(ranges
        .iter()
        .enumerate()
        .map(|(i, &(from, to))| Chunk {
            lines: LineSpan {
                first: from as u32,
                last: to as u32,
            },
            text: section_text(path, &range_read(repo, rel, &doc, from, to), i == last),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use zend_tools::tools::file::render::lines_read;

    use super::*;

    /// **A file is cut where `file_read` pages it, each section the read of
    /// its range** (which, a range of several lines, is `lines_read`'s own):
    /// 150 two-line paragraphs page as lines 1-300 and 301-450;
    /// the first section is the heading, the read of 1-300 and the cut
    /// marker, the last the read of 301-450 and no marker.
    #[test]
    fn a_file_is_cut_where_file_read_pages_it() {
        let text: String = (0..150).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        let doc = Doc::new("docs/a.md", &text);
        assert_eq!(
            range_read("r", "docs/a.md", &doc, 1, 300),
            lines_read("r", "docs/a.md", &doc, 1, 300)
        );
        let cut = sections("r/docs/a.md", FileUnit::File, &text).unwrap();
        assert_eq!(
            cut.iter().map(|c| c.lines).collect::<Vec<_>>(),
            vec![
                LineSpan {
                    first: 1,
                    last: 300
                },
                LineSpan {
                    first: 301,
                    last: 450
                }
            ]
        );
        assert_eq!(
            cut[0].text,
            format!(
                "## The content from r/docs/a.md:\n\n{}...\n\n",
                range_read("r", "docs/a.md", &doc, 1, 300)
            )
        );
        assert_eq!(
            cut[1].text,
            format!(
                "## The content from r/docs/a.md:\n\n{}\n",
                range_read("r", "docs/a.md", &doc, 301, 450)
            )
        );
        assert!(cut[1]
            .text
            .ends_with("end of r/docs/a.md lines 301-450 of 450 — the end of the file\n\n"));
    }

    /// **A short file is one section, exactly the read an edit works from**:
    /// its parts fenced with their anchors and relative line numbers.
    #[test]
    fn a_short_file_is_one_section_with_its_anchors() {
        let text = "use x;\n\nfn go() {\n    1\n}\n";
        let doc = Doc::new("a.rs", text);
        let a = doc.anchors();
        let cut = sections("r/a.rs", FileUnit::File, text).unwrap();
        assert_eq!(cut.len(), 1);
        assert_eq!(
            cut[0].text,
            format!(
                "## The content from r/a.rs:\n\n\
                 ```rust file=r/a.rs part={} lines=1\n +1 use x;\n```\n\
                 ```rust file=r/a.rs part={} lines=3-5 scope=\"fn go\"\n +1 fn go() {{\n +2     1\n +3 }}\n```\n\
                 end of r/a.rs lines 1-5 of 5 — the end of the file\n\n",
                a[0], a[1]
            )
        );
    }

    /// **A file with no text is one section saying so**, and a folder one
    /// section, its listing fenced.
    #[test]
    fn an_empty_file_and_a_folder_are_one_section_each() {
        let blank = sections("r/a.rs", FileUnit::File, "\n\n").unwrap();
        assert_eq!(
            blank[0].text,
            "## The content from r/a.rs:\n\nr/a.rs holds no text — 2 line(s), all blank.\n\n"
        );
        let folder = sections("r/src/", FileUnit::Folder, "a.rs 3\nb/\n").unwrap();
        assert_eq!(
            folder,
            vec![Chunk {
                lines: LineSpan { first: 1, last: 2 },
                text: "## The content from r/src/:\n\n```\na.rs 3\nb/\n```\n\n".into(),
            }]
        );
        assert!(sections("a.rs", FileUnit::File, "x\n").is_err());
    }
}
