//! One knowledge-base unit's sealed sections, as the substrate inspector shows
//! them when a file or folder listing is opened: each section's line span, its
//! token count and its text, decoded from the tokens the section was sealed
//! with — what the model reads, not the file as it stands on disk.

use candle_conversation::files::files_owner;
use candle_conversation::files::names::chunk_name;
use candle_conversation::projection::Conversation;
use serde::Serialize;

use super::records;

/// One sealed section of a unit.
#[derive(Debug, PartialEq, Serialize)]
pub struct SectionText {
    /// The first and last line of the unit the section holds.
    pub first: u32,
    pub last: u32,
    /// Its tokens, as the record names them.
    pub tokens: usize,
    /// The sealed tokens decoded, or `None` where this process holds no
    /// section under that name (not yet sealed or restored here).
    pub text: Option<String>,
}

/// The sections of `path` at `blob`, in line order, each decoded by `decode`;
/// `None` when no live record holds that version.
pub fn sections(
    conversation: &Conversation,
    path: &str,
    blob: &str,
    decode: &dyn Fn(&[u32]) -> String,
) -> Option<Vec<SectionText>> {
    let record = records::of_path(conversation, path)
        .into_iter()
        .find(|r| r.blob == blob)?;
    let read = conversation.read();
    Some(
        record
            .chunks
            .iter()
            .map(|c| {
                let text = read
                    .owned_section_named(files_owner(), &chunk_name(path, blob, c.lines))
                    .map(|section| decode(&read.section_tokens_of(section)));
                SectionText {
                    first: c.lines.first,
                    last: c.lines.last,
                    tokens: c.member.tokens as usize,
                    text,
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, FileRecord, FileUnit, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;

    use super::*;

    fn chunk(first: u32, last: u32, tokens: u32) -> ChunkRecord {
        ChunkRecord {
            lines: LineSpan { first, last },
            member: MemberRef {
                address: ContentAddress {
                    prefix_hash: ContentHash { lo: 1, hi: 0 },
                    section_hash: ContentHash {
                        lo: first as u64,
                        hi: 0,
                    },
                },
                tokens,
            },
        }
    }

    /// **Each section of the version asked for, in line order**: decoded
    /// where this process holds it, `None` where it does not; another version
    /// of the path, or a path with none, answers nothing.
    #[test]
    fn a_units_sections_are_decoded_where_held() {
        let conversation = Conversation::new();
        records::put(
            &conversation,
            &FileRecord {
                content_key: "v1:candle/a.rs@b1".into(),
                path: "candle/a.rs".into(),
                blob: "b1".into(),
                unit: FileUnit::File,
                lines: 20,
                lang: "rs".into(),
                branches: vec!["main".into()],
                chunks: vec![chunk(1, 10, 30), chunk(11, 20, 12)],
            },
        )
        .unwrap();
        let section = conversation.allocate_owned_section().expect("an owned id");
        conversation.register_owned_section(files_owner(), "candle/a.rs@b1#1-10", section);
        let decode = |tokens: &[u32]| format!("{} tokens", tokens.len());

        assert_eq!(
            sections(&conversation, "candle/a.rs", "b1", &decode).unwrap(),
            vec![
                SectionText {
                    first: 1,
                    last: 10,
                    tokens: 30,
                    text: Some("0 tokens".into()),
                },
                SectionText {
                    first: 11,
                    last: 20,
                    tokens: 12,
                    text: None,
                },
            ]
        );
        assert!(sections(&conversation, "candle/a.rs", "b2", &decode).is_none());
        assert!(sections(&conversation, "candle/b.rs", "b1", &decode).is_none());
    }
}
