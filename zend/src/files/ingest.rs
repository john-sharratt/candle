//! A file of the workspace into the knowledge base
//! (`docs/placed_section_layout.md` §3.5b, §3.5c): cut into sections where
//! `file_read` pages it ([`sections`]), sealed as members — each section
//! continuing the ones before it in its file, the chain ending at the top of
//! the collection's range — and recorded, the record naming every member by
//! its content address. Nothing is decoded: the file is prefilled, and its
//! calibration follows.

use candle_conversation::files::names::chunk_name;
use candle_conversation::files::{Chunk, ChunkRecord, FileRecord, FileUnit, MemberRef};
use candle_conversation::projection::SectionId;
use candle_conversation::{MemberSpec, SealedMember, Sequence};

use super::sections::sections;
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

/// The members a file is sealed as, in order: each section.
pub fn member_specs(path: &str, blob: &str, chunks: &[Chunk]) -> Vec<MemberSpec> {
    chunks
        .iter()
        .map(|c| MemberSpec {
            name: chunk_name(path, blob, c.lines),
            content: c.text.clone(),
        })
        .collect()
}

/// The record of `source`, its `chunks` sealed as `sealed`, in the same order.
pub fn record_of(source: &FileSource<'_>, chunks: &[Chunk], sealed: &[SealedMember]) -> FileRecord {
    FileRecord {
        content_key: source.content_key.to_string(),
        path: source.path.to_string(),
        blob: source.blob.to_string(),
        unit: source.unit,
        lines: chunks.last().map_or(0, |c| c.lines.last),
        lang: source.lang.to_string(),
        branches: source.branches.to_vec(),
        chunks: chunks
            .iter()
            .zip(sealed)
            .map(|(c, m)| ChunkRecord {
                lines: c.lines,
                member: MemberRef {
                    address: m.address,
                    tokens: m.tokens as u32,
                },
            })
            .collect(),
    }
}

/// Cut `text` into its sections, seal them as `source`'s members on `base`
/// behind `lead` ([`super::lead`]), and return its record.
pub fn ingest_text(
    base: &Sequence,
    source: &FileSource<'_>,
    text: &str,
    lead: &[SectionId],
) -> anyhow::Result<FileRecord> {
    let chunks = sections(source.path, source.unit, text)?;
    let specs = member_specs(source.path, source.blob, &chunks);
    let sealed = base
        .seal_members(FILES_COLLECTION, lead, &specs)
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

    /// **A file is sealed as its sections**, each named by its file and its
    /// lines.
    #[test]
    fn a_file_is_sealed_as_its_sections() {
        let chunks = [chunk(1, 2, "a\nb\n"), chunk(3, 3, "c\n")];
        let specs = member_specs("candle/a.rs", "ab", &chunks);
        assert_eq!(
            specs,
            vec![
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

    /// **A committed file writes one record naming every section** by the
    /// address it was sealed under.
    #[test]
    fn a_committed_file_writes_one_record_naming_every_section() {
        let chunks = [chunk(1, 200, "x\n"), chunk(201, 260, "y\n")];
        let sealed: Vec<SealedMember> = (0..2)
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
        assert_eq!(
            record.chunks,
            vec![
                ChunkRecord {
                    lines: LineSpan {
                        first: 1,
                        last: 200
                    },
                    member: MemberRef {
                        address: sealed[0].address,
                        tokens: 100
                    },
                },
                ChunkRecord {
                    lines: LineSpan {
                        first: 201,
                        last: 260
                    },
                    member: MemberRef {
                        address: sealed[1].address,
                        tokens: 101
                    },
                },
            ]
        );
        assert_eq!(record.object_key(), "files/candle/a.rs@ab");
    }
}
