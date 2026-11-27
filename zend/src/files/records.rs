//! The knowledge base's file records in the substrate
//! (`docs/provenance_system_collection.md` §12 D1, D15): written when a file's
//! members commit, retired when no branch holds the version any more, and read
//! by the planner as the files the knowledge base holds.
//!
//! A record is one keyed object, so it replicates whole: a member that has a
//! record has every member address the file was sealed as, with nothing
//! arriving after it. A record names members by address; whether this process
//! has sealed or restored them is asked of the names they register under
//! ([`placeable`]).

use candle_conversation::files::names::chunk_name;
use candle_conversation::files::{files_owner, path_prefix, FileRecord, FILES_PREFIX};
use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::projection::Conversation;

/// Every live file record.
pub fn all(conversation: &Conversation) -> Vec<FileRecord> {
    under(conversation, FILES_PREFIX)
}

/// Every live record of `path`, whatever its blob.
pub fn of_path(conversation: &Conversation, path: &str) -> Vec<FileRecord> {
    under(conversation, &path_prefix(path))
}

fn under(conversation: &Conversation, prefix: &str) -> Vec<FileRecord> {
    conversation
        .shared()
        .custom_objects_under(prefix)
        .iter()
        .filter_map(|(obj, _)| FileRecord::from_object(obj))
        .collect()
}

/// Write `record`, replacing what its key held.
pub fn put(conversation: &Conversation, record: &FileRecord) -> anyhow::Result<()> {
    conversation
        .shared()
        .put_custom_object(record.to_object())?;
    Ok(())
}

/// Retire `record`: its key ceases, every member reads it as gone.
pub fn retire(conversation: &Conversation, record: &FileRecord) -> anyhow::Result<()> {
    conversation
        .shared()
        .put_custom_object(CustomObjectPayload::ceased(record.object_key()))?;
    Ok(())
}

/// Whether this process holds `record`'s members: its last section
/// registered, which a file's seal does last of its chain, so a record whose
/// chain this process sealed or restored whole. A record that arrived from
/// another member, or survived a restart, is not until the ingest's walk
/// seals its chain here (§12 D15).
pub fn placeable(conversation: &Conversation, record: &FileRecord) -> bool {
    record.chunks.last().is_some_and(|last| {
        conversation
            .read()
            .owned_section_named(
                files_owner(),
                &chunk_name(&record.path, &record.blob, last.lines),
            )
            .is_some()
    })
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, FileUnit, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;

    use super::*;

    fn record(path: &str, blob: &str) -> FileRecord {
        let member = MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 1, hi: 0 },
                section_hash: ContentHash { lo: 2, hi: 0 },
            },
            tokens: 10,
        };
        FileRecord {
            content_key: format!("v1:{path}@{blob}"),
            path: path.to_string(),
            blob: blob.to_string(),
            unit: FileUnit::File,
            lines: 3,
            lang: "rs".into(),
            branches: vec!["main".into()],
            chunks: vec![ChunkRecord {
                lines: LineSpan { first: 1, last: 3 },
                member,
            }],
        }
    }

    /// **A record written reads back under its path, and a retired one is
    /// gone**, while another version of the same path stands.
    #[test]
    fn a_retired_record_is_gone_and_the_other_version_stands() {
        let conversation = Conversation::new();
        let old = record("candle/a.rs", "aa");
        let new = record("candle/a.rs", "bb");
        let other = record("candle/b.rs", "cc");
        for r in [&old, &new, &other] {
            put(&conversation, r).unwrap();
        }
        assert_eq!(all(&conversation).len(), 3);
        retire(&conversation, &old).unwrap();
        assert_eq!(of_path(&conversation, "candle/a.rs"), vec![new.clone()]);
        assert_eq!(all(&conversation).len(), 2);
    }

    /// **A record is placeable only once this process has named its last
    /// section**: one read from the substrate alone is not.
    #[test]
    fn a_record_is_placeable_once_its_last_section_is_named_here() {
        let conversation = Conversation::new();
        let r = record("candle/a.rs", "aa");
        put(&conversation, &r).unwrap();
        assert!(!placeable(&conversation, &r));
        let section = conversation.allocate_owned_section().expect("an owned id");
        conversation.register_owned_section(files_owner(), "candle/a.rs@aa#1-3", section);
        assert!(placeable(&conversation, &r));
    }
}
