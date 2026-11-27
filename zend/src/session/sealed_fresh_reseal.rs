//! A file's members sealed again behind the prompt that actually reads them,
//! for sealed-fresh's `reseal` arm (`session/sealed_fresh.rs`).
//!
//! A knowledge-base member is sealed behind the birth prefix up to
//! `files_open`, which leaves the tool catalog's members out; the prompt that
//! reads it places them ahead of it. Sealed again behind the sections the
//! prompt places before the file — same text, same sections, same policies —
//! under a probe version of its record, the member's K/V is computed in the
//! context it is read in, and nothing else about the prompt changes.

use anyhow::Context;
use candle_conversation::files::names::chunk_name;
use candle_conversation::files::{files_owner, ChunkRecord, FileRecord, MemberRef};
use candle_conversation::projection::{Conversation, PrefixItem, SectionId};
use candle_conversation::{MemberSpec, MembersPrefix, SealAs, Sequence};

use crate::files::records;

/// The blob a probe record of `blob` is named under.
pub fn probe_blob(blob: &str) -> String {
    format!("{blob}~probe")
}

/// `record`'s members as their text: each stored member's tokens decoded,
/// checked to encode back to the same ids, so a seal of the text is a seal of
/// exactly those tokens.
fn member_texts(
    conversation: &Conversation,
    base: &Sequence,
    decode: impl Fn(&[u32]) -> anyhow::Result<String>,
    record: &FileRecord,
) -> anyhow::Result<Vec<String>> {
    record
        .chunks
        .iter()
        .map(|c| chunk_name(&record.path, &record.blob, c.lines))
        .map(|name| {
            let id = conversation
                .read()
                .owned_section_named(files_owner(), &name)
                .with_context(|| format!("member {name} is not held here"))?;
            let ids = conversation.read().section_tokens_of(id).to_vec();
            let text = decode(&ids)?;
            let again = base
                .member_token_ids(&text)
                .map_err(|e| anyhow::anyhow!("encoding {name}: {e}"))?;
            anyhow::ensure!(
                again == ids,
                "member {name} does not encode back to its own {} ids",
                ids.len()
            );
            Ok(text)
        })
        .collect()
}

/// Seal `record`'s members again behind `prefix` under `kind`'s policy, under
/// a probe record of it, and write that record; the probe record.
pub fn reseal_behind(
    conversation: &Conversation,
    base: &Sequence,
    decode: impl Fn(&[u32]) -> anyhow::Result<String>,
    record: &FileRecord,
    prefix: &[PrefixItem],
    kind: SealAs,
) -> anyhow::Result<FileRecord> {
    let texts = member_texts(conversation, base, decode, record)?;
    let blob = match kind {
        SealAs::Member => probe_blob(&record.blob),
        SealAs::Boundary => format!("{}~q8", probe_blob(&record.blob)),
    };
    let specs: Vec<MemberSpec> = record
        .chunks
        .iter()
        .zip(&texts)
        .map(|(c, text)| MemberSpec {
            name: chunk_name(&record.path, &blob, c.lines),
            content: text.clone(),
        })
        .collect();
    // The prompt as placed already holds the hole ahead of the file, so the
    // members are sealed straight after it, as they were placed.
    let behind = MembersPrefix {
        prefix: prefix.to_vec(),
        range: None,
    };
    let sealed = base
        .seal_members_behind(&behind, &specs, kind)
        .map_err(|e| anyhow::anyhow!("sealing the probe's members: {e}"))?;
    let probe = FileRecord {
        content_key: format!("{}~{blob}", record.content_key),
        blob,
        chunks: record
            .chunks
            .iter()
            .zip(&sealed)
            .map(|(c, m)| ChunkRecord {
                lines: c.lines,
                member: MemberRef {
                    address: m.address,
                    tokens: m.tokens as u32,
                },
            })
            .collect(),
        ..record.clone()
    };
    records::put(conversation, &probe)?;
    Ok(probe)
}

/// One piece of a dumped prompt as the reseal reads it: the section it is,
/// or the hole of positions it leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DumpedPiece {
    Section(u32),
    Hole(usize),
    /// Glue or a turn: neither sealed ahead of a file nor a hole.
    Other,
}

/// What `dump`-ordered `pieces` place ahead of the first of `members`: the
/// prompt before the file, its sections and its holes.
pub fn prefix_before(pieces: &[DumpedPiece], first_member: u32) -> anyhow::Result<Vec<PrefixItem>> {
    let at = pieces
        .iter()
        .position(|p| *p == DumpedPiece::Section(first_member))
        .context("the file's first section is not in the prompt")?;
    Ok(pieces[..at]
        .iter()
        .filter_map(|p| match *p {
            DumpedPiece::Section(id) => Some(PrefixItem::Section(SectionId::new(id))),
            DumpedPiece::Hole(positions) => Some(PrefixItem::Hole(positions)),
            DumpedPiece::Other => None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The prefix is every section and hole ahead of the file's first
    /// section**, in order, glue left out.
    #[test]
    fn the_prefix_is_every_section_and_hole_ahead_of_the_first_section() {
        let pieces = [
            DumpedPiece::Section(1),
            DumpedPiece::Other,
            DumpedPiece::Hole(500),
            DumpedPiece::Section(5),
            DumpedPiece::Section(7),
            DumpedPiece::Section(8),
        ];
        assert_eq!(
            prefix_before(&pieces, 7).unwrap(),
            vec![
                PrefixItem::Section(SectionId::new(1)),
                PrefixItem::Hole(500),
                PrefixItem::Section(SectionId::new(5)),
            ]
        );
        assert!(prefix_before(&pieces, 99).is_err());
    }

    #[test]
    fn a_probe_blob_is_named_apart_from_its_file() {
        assert_eq!(probe_blob("abc"), "abc~probe");
    }
}
