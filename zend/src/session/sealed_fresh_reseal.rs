//! A file's members sealed again behind the prompt that actually reads them,
//! for sealed-fresh's `reseal` arm (`session/sealed_fresh.rs`).
//!
//! A knowledge-base member is sealed behind the birth prefix up to
//! `files_open`, which leaves the tool catalog's members out; the prompt that
//! reads it places them ahead of it. Sealed again behind the sections the
//! prompt places before the file — same text, same chunks, same policies —
//! under a probe version of its record, the member's K/V is computed in the
//! context it is read in, and nothing else about the prompt changes.

use anyhow::Context;
use candle_conversation::files::names::{chunk_name, heading_name};
use candle_conversation::files::{files_owner, ChunkRecord, FileRecord, MemberRef};
use candle_conversation::projection::{Conversation, SectionId};
use candle_conversation::{MemberSpec, Sequence};

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
    let mut names = vec![heading_name(&record.path, &record.blob)];
    names.extend(
        record
            .chunks
            .iter()
            .map(|c| chunk_name(&record.path, &record.blob, c.lines)),
    );
    names
        .iter()
        .map(|name| {
            let id = conversation
                .read()
                .owned_section_named(files_owner(), name)
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

/// Seal `record`'s members again behind `prefix`, under a probe record of
/// it, and write that record; the probe record.
pub fn reseal_behind(
    conversation: &Conversation,
    base: &Sequence,
    decode: impl Fn(&[u32]) -> anyhow::Result<String>,
    record: &FileRecord,
    prefix: &[SectionId],
) -> anyhow::Result<FileRecord> {
    let texts = member_texts(conversation, base, decode, record)?;
    let blob = probe_blob(&record.blob);
    let mut specs = vec![MemberSpec {
        name: heading_name(&record.path, &blob),
        content: texts[0].clone(),
    }];
    specs.extend(
        record
            .chunks
            .iter()
            .zip(&texts[1..])
            .map(|(c, text)| MemberSpec {
                name: chunk_name(&record.path, &blob, c.lines),
                content: text.clone(),
            }),
    );
    let sealed = base
        .seal_members_behind(prefix, &specs)
        .map_err(|e| anyhow::anyhow!("sealing the probe's members: {e}"))?;
    let member = |i: usize| MemberRef {
        address: sealed[i].address,
        tokens: sealed[i].tokens as u32,
    };
    let probe = FileRecord {
        content_key: format!("{}~probe", record.content_key),
        blob,
        heading: member(0),
        chunks: record
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| ChunkRecord {
                lines: c.lines,
                member: member(i + 1),
            })
            .collect(),
        ..record.clone()
    };
    records::put(conversation, &probe)?;
    Ok(probe)
}

/// The sections `dump`-ordered `pieces` place ahead of the first of `members`:
/// the prompt before the file.
pub fn prefix_before(
    pieces: &[(Option<u32>, usize)],
    first_member: u32,
) -> anyhow::Result<Vec<SectionId>> {
    let at = pieces
        .iter()
        .position(|(id, _)| *id == Some(first_member))
        .context("the file's heading is not in the prompt")?;
    Ok(pieces[..at]
        .iter()
        .filter_map(|(id, _)| id.map(SectionId::new))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The prefix is every section ahead of the file's heading**, in order,
    /// glue left out.
    #[test]
    fn the_prefix_is_every_section_ahead_of_the_heading() {
        let pieces = [
            (Some(1), 3),
            (None, 2),
            (Some(5), 9),
            (Some(7), 4),
            (Some(8), 1),
        ];
        assert_eq!(
            prefix_before(&pieces, 7).unwrap(),
            vec![SectionId::new(1), SectionId::new(5)]
        );
        assert!(prefix_before(&pieces, 99).is_err());
    }

    #[test]
    fn a_probe_blob_is_named_apart_from_its_file() {
        assert_eq!(probe_blob("abc"), "abc~probe");
    }
}
