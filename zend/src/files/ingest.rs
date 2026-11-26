//! A file of the workspace into the knowledge base
//! (`docs/provenance_system_collection.md` §7): cut into chunks of whole
//! lines whose seams tokenize as the file does, sealed as members — its
//! heading, then each chunk continuing the one before — and recorded, the
//! record naming every member by its content address. Nothing is decoded: the
//! file is prefilled, and its calibration follows.

use std::sync::OnceLock;

use candle_conversation::files::names::{
    chunk_name, heading_name, CLOSING_JOIN_NAME, CLOSING_LAST_NAME, GAP_NAME,
};
use candle_conversation::files::render::{heading, CLOSING_JOIN, CLOSING_LAST, GAP};
use candle_conversation::files::{cut_lines, Chunk, ChunkRecord, FileRecord, FileUnit, MemberRef};
use candle_conversation::{MemberSpec, SealedMember, Sequence};
use zend_tools::tools::file::render::PAGE_LINES;

use super::FILES_COLLECTION;

/// What a file is, as its record names it.
pub struct FileSource<'a> {
    /// `v{N}:{path}@{blob}` — the record's content key.
    pub content_key: &'a str,
    /// The workspace path, repository first.
    pub path: &'a str,
    pub blob: &'a str,
    pub unit: FileUnit,
    pub lang: &'a str,
    pub branches: &'a [String],
}

/// The members a file is sealed as, in order: its heading, then each chunk.
pub fn member_specs(path: &str, blob: &str, chunks: &[Chunk]) -> Vec<MemberSpec> {
    let mut out = vec![MemberSpec {
        name: heading_name(path, blob),
        content: heading(path),
    }];
    out.extend(chunks.iter().map(|c| MemberSpec {
        name: chunk_name(path, blob, c.lines),
        content: c.text.clone(),
    }));
    out
}

/// The record of `source`, its `chunks` sealed as `sealed` (the heading
/// first, as [`member_specs`] orders them).
pub fn record_of(source: &FileSource<'_>, chunks: &[Chunk], sealed: &[SealedMember]) -> FileRecord {
    let member = |m: &SealedMember| MemberRef {
        address: m.address,
        tokens: m.tokens as u32,
    };
    FileRecord {
        content_key: source.content_key.to_string(),
        path: source.path.to_string(),
        blob: source.blob.to_string(),
        unit: source.unit,
        lines: chunks.last().map_or(0, |c| c.lines.last),
        lang: source.lang.to_string(),
        branches: source.branches.to_vec(),
        heading: member(&sealed[0]),
        chunks: chunks
            .iter()
            .zip(&sealed[1..])
            .map(|(c, m)| ChunkRecord {
                lines: c.lines,
                member: member(m),
            })
            .collect(),
    }
}

/// The shared pieces every placed file is framed with, sealed once a
/// process: each its own member behind the collection's prefix.
static PIECES_SEALED: OnceLock<()> = OnceLock::new();

/// Seal the gap and the two closing pieces, when this process has not.
pub fn seal_pieces(base: &Sequence) -> anyhow::Result<()> {
    if PIECES_SEALED.get().is_some() {
        return Ok(());
    }
    for (name, content) in [
        (GAP_NAME, GAP),
        (CLOSING_JOIN_NAME, CLOSING_JOIN),
        (CLOSING_LAST_NAME, CLOSING_LAST),
    ] {
        base.seal_members(
            FILES_COLLECTION,
            &[MemberSpec {
                name: name.to_string(),
                content: content.to_string(),
            }],
        )
        .map_err(|e| anyhow::anyhow!("sealing the knowledge base's {name}: {e}"))?;
    }
    let _ = PIECES_SEALED.set(());
    Ok(())
}

/// Cut `text`, seal it as `source`'s members on `base`, and return its
/// record. A file with no text is its heading alone, placed as an empty
/// fence.
pub fn ingest_text(
    base: &Sequence,
    source: &FileSource<'_>,
    text: &str,
) -> anyhow::Result<FileRecord> {
    seal_pieces(base)?;
    let chunks = cut_lines(text, PAGE_LINES, |s| {
        base.member_token_ids(s).unwrap_or_default()
    });
    let specs = member_specs(source.path, source.blob, &chunks);
    let sealed = base
        .seal_members(FILES_COLLECTION, &specs)
        .map_err(|e| anyhow::anyhow!("sealing {}: {e}", source.path))?;
    Ok(record_of(source, &chunks, &sealed))
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::LineSpan;
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::{ContentAddress, StreamId};
    use candle_conversation::projection::SectionId;

    use super::*;

    fn chunk(first: u32, last: u32, text: &str) -> Chunk {
        Chunk {
            lines: LineSpan { first, last },
            text: text.to_string(),
        }
    }

    /// **A file is sealed as its heading, then each chunk**, each named by
    /// its file and its lines.
    #[test]
    fn a_file_is_sealed_as_its_heading_then_each_chunk() {
        let chunks = [chunk(1, 2, "a\nb\n"), chunk(3, 3, "c\n")];
        let specs = member_specs("candle/a.rs", "ab", &chunks);
        assert_eq!(
            specs,
            vec![
                MemberSpec {
                    name: "candle/a.rs@ab#heading".into(),
                    content: "## The content from candle/a.rs:\n\n```\n".into(),
                },
                MemberSpec {
                    name: "candle/a.rs@ab#1-2".into(),
                    content: "a\nb\n".into(),
                },
                MemberSpec {
                    name: "candle/a.rs@ab#3-3".into(),
                    content: "c\n".into(),
                },
            ]
        );
    }

    /// **A committed file writes one record naming every chunk** by the
    /// address it was sealed under.
    #[test]
    fn a_committed_file_writes_one_record_naming_every_chunk() {
        let chunks = [chunk(1, 200, "x\n"), chunk(201, 260, "y\n")];
        let sealed: Vec<SealedMember> = (0..3)
            .map(|i| SealedMember {
                section: SectionId::new(10 + i),
                stream: StreamId(u64::from(i)),
                address: ContentAddress {
                    prefix_hash: ContentHash {
                        lo: u64::from(i),
                        hi: 0,
                    },
                    section_hash: ContentHash { lo: 7, hi: 0 },
                },
                tokens: 100 + i as usize,
                restored: false,
            })
            .collect();
        let branches = vec!["main".to_string()];
        let source = FileSource {
            content_key: "v1:candle/a.rs@ab",
            path: "candle/a.rs",
            blob: "ab",
            unit: FileUnit::File,
            lang: "rs",
            branches: &branches,
        };
        let record = record_of(&source, &chunks, &sealed);
        assert_eq!(record.lines, 260);
        assert_eq!(record.heading.address, sealed[0].address);
        assert_eq!(record.heading.tokens, 100);
        assert_eq!(
            record.chunks,
            vec![
                ChunkRecord {
                    lines: LineSpan {
                        first: 1,
                        last: 200
                    },
                    member: MemberRef {
                        address: sealed[1].address,
                        tokens: 101
                    },
                },
                ChunkRecord {
                    lines: LineSpan {
                        first: 201,
                        last: 260
                    },
                    member: MemberRef {
                        address: sealed[2].address,
                        tokens: 102
                    },
                },
            ]
        );
        assert_eq!(record.object_key(), "files/candle/a.rs@ab");
    }
}
