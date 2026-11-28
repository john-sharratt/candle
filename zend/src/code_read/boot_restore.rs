//! Restoring the files already recorded, from their records, before the pass
//! reads any (`docs/provenance_system_collection.md` §12 D15).
//!
//! After a restart no member is sealed in this process, so every recorded file
//! would otherwise go through the pass as a new one does: read from git, cut,
//! tokenized and restored member by member, one scheduler round trip each. A
//! record names its members by address and size, which is all a restore needs,
//! so the records go to [`Sequence::restore_member_chains`] together first.
//! A file restored whole is placeable, and the pass skips it as one sealed
//! here; any other — its text changed, its prompt moved, a member partial or
//! gone — goes through the pass as before.

use std::time::Instant;

use candle_conversation::files::names::chunk_name;
use candle_conversation::{RecordedChain, RecordedMember};

use super::*;
use crate::files::FILES_COLLECTION;

/// The chain a record names, as a restore takes it.
fn chain_of_record(record: &FileRecord) -> RecordedChain {
    RecordedChain {
        members: record
            .chunks
            .iter()
            .map(|c| RecordedMember {
                name: chunk_name(&record.path, &record.blob, c.lines),
                address: c.member.address,
                tokens: c.member.tokens as usize,
            })
            .collect(),
    }
}

/// Restore every file of `keys` that is recorded and not yet placeable here,
/// from its record, on `base`. Returns how many were restored whole.
pub(super) fn restore_recorded(
    engine: &ConversationEngine,
    base: &Sequence,
    keys: &[String],
) -> usize {
    let conversation = engine.conversation();
    let wanted: HashSet<&str> = keys.iter().map(String::as_str).collect();
    let recorded: Vec<FileRecord> = records::all(&conversation)
        .into_iter()
        .filter(|r| {
            r.unit == FileUnit::File
                && wanted.contains(r.content_key.as_str())
                && !records::placeable(&conversation, r)
        })
        .collect();
    if recorded.is_empty() {
        return 0;
    }
    let chains: Vec<RecordedChain> = recorded.iter().map(chain_of_record).collect();
    let started = Instant::now();
    match base.restore_member_chains(FILES_COLLECTION, &chains) {
        Ok(whole) => {
            let restored = whole.iter().filter(|w| **w).count();
            tracing::info!(
                target: "zend::code_read::ingest",
                recorded = recorded.len(),
                restored,
                left_to_the_pass = recorded.len() - restored,
                members = chains.iter().map(|c| c.members.len()).sum::<usize>(),
                ms = started.elapsed().as_millis() as u64,
                "recorded files restored from their records",
            );
            restored
        }
        Err(e) => {
            tracing::warn!(
                target: "zend::code_read::ingest",
                recorded = recorded.len(),
                "restoring the recorded files failed; the pass seals each as it would a new one: {e}",
            );
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;

    use super::*;

    /// **A record's chain names each member as the pass registers it**, with
    /// the address and size the record holds, in line order.
    #[test]
    fn a_records_chain_names_its_members_as_the_pass_does() {
        let member = |lo: u64, tokens: u32| MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash { lo, hi: 0 },
                section_hash: ContentHash { lo: lo + 1, hi: 0 },
            },
            tokens,
        };
        let record = FileRecord {
            content_key: "v7:candle/a.rs@ab12".into(),
            path: "candle/a.rs".into(),
            blob: "ab12".into(),
            unit: FileUnit::File,
            lines: 260,
            lang: "rs".into(),
            branches: vec!["main".into()],
            chunks: vec![
                ChunkRecord {
                    lines: LineSpan {
                        first: 1,
                        last: 200,
                    },
                    member: member(3, 2900),
                },
                ChunkRecord {
                    lines: LineSpan {
                        first: 201,
                        last: 260,
                    },
                    member: member(5, 700),
                },
            ],
        };
        let chain = chain_of_record(&record);
        assert_eq!(
            chain
                .members
                .iter()
                .map(|m| (m.name.as_str(), m.address, m.tokens))
                .collect::<Vec<_>>(),
            vec![
                ("candle/a.rs@ab12#1-200", member(3, 0).address, 2900),
                ("candle/a.rs@ab12#201-260", member(5, 0).address, 700),
            ]
        );
    }
}
