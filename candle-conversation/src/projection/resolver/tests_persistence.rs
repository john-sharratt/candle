//! Tests that ephemeral conversations remove their directories, and that
//! conversation state and read-only mode survive a reopen and a compaction.
//!
//! An ephemeral log grows in 64 MiB extents that only a sealed segment reclaims, so
//! its directory is removed with the last clone. A workspace conversation owns no
//! directory to remove. Conversation state replays whole after a reopen, and a
//! read-only conversation changes memory but writes no bytes. A section's commit
//! must survive the compaction in the process that wrote it, since a commit the
//! index does not hold is dropped by the first rewrite.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::Conversation;

    use std::collections::BTreeMap;

    use crate::persistence::content_hash::turn_stream_id;

    use crate::persistence::manifest::ConvState;

    use crate::persistence::vfs::{VfsAppend, VfsWrite};

    use crate::persistence::{dir_fingerprint, Admit, SharedSubstrate, SubstratePersistence};

    use crate::projection::{GroupId, LayerId, TimelineId};

    use crate::substrate::Substrate;

    /// **An ephemeral conversation takes its directory with it.**
    ///
    /// Its log is a throwaway, and `LogFile` grows in 64 MiB extents — the
    /// slack is only reclaimed when a segment *seals*, which a throwaway never
    /// does. So every ephemeral conversation used to leave 64 MiB in `%TEMP%`
    /// permanently. Summarisation makes them, tests make them, and 727 of them
    /// had accumulated on this machine (~46 GB) beside the same leak from the
    /// test helpers — together half a 1 TB disk.
    ///
    /// The clone is not decoration: the guard sits behind the same `Arc` as
    /// everything else on the handle, so a clone must *not* remove the
    /// directory the original is still writing to.
    #[test]
    fn an_ephemeral_conversation_removes_its_temp_directory_on_the_last_drop() {
        let conv = Conversation::ephemeral();
        let dir = conv
            .ephemeral_dir
            .as_ref()
            .expect("an ephemeral conversation owns its directory")
            .path()
            .to_path_buf();
        assert!(dir.is_dir(), "the ephemeral log directory was never made");

        let clone = conv.clone();
        drop(conv);
        assert!(
            dir.is_dir(),
            "dropping one clone removed the directory the other is still writing to"
        );

        drop(clone);
        assert!(
            !dir.exists(),
            "the last drop left {dir:?} behind — 64 MiB per ephemeral conversation, forever"
        );
    }

    /// A conversation opened on a real workspace holds no guard, because
    /// removing *that* directory on drop would delete somebody's substrate
    /// rather than a temp file.
    #[test]
    fn a_workspace_conversation_owns_no_directory_to_delete() {
        let dir = std::env::temp_dir().join(format!(
            "candle-conv-workspace-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut substrate = crate::substrate::Substrate::new();
        let persistence =
            crate::persistence::SubstratePersistence::open_in_with_substrate(&dir, &mut substrate)
                .expect("workspace persistence");
        let conv = Conversation::from_parts(substrate, persistence);
        assert!(conv.ephemeral_dir.is_none());
        drop(conv);
        assert!(
            dir.is_dir(),
            "a workspace conversation deleted the directory it was opened on"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A conversation's state survives a reopen whole.** Branches set, a
    /// file event written, an archive after them, then one branch moved: each
    /// record carries the whole state, so the archive keeps the branches, the
    /// move keeps the archive, and the log replays to exactly the state RAM
    /// held — and the file event reads back beside it.
    #[test]
    fn conversation_state_survives_a_reopen_whole() {
        let dir = std::env::temp_dir().join(format!(
            "candle-conv-state-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let tl = TimelineId::from_raw(7).expect("timeline id");
        let map = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            pairs
                .iter()
                .map(|(r, b)| (r.to_string(), b.to_string()))
                .collect()
        };
        let expected = ConvState {
            archived: true,
            branches: map(&[("candle", "zen/work"), ("mind", "master")]),
            active: 0,
        };
        let write = VfsWrite {
            tombstones: Vec::new(),
            events: vec![VfsAppend {
                repo: "candle".into(),
                key: "src/lib.rs".into(),
                body: serde_json::json!({ "kind": "state", "size": 3, "conflict": false }),
            }],
        };
        {
            let mut substrate = Substrate::new();
            let p = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let conv = Conversation::from_parts(substrate, p);
            conv.register_timeline(tl, layer, group);
            conv.set_conversation_branches(tl, &map(&[("candle", "main"), ("mind", "master")]));
            let ids = conv.write_conversation_files(tl, &write).unwrap();
            assert_eq!(ids.iter().map(|id| id.seq).collect::<Vec<_>>(), vec![0]);
            conv.set_conversation_archived(tl, true).unwrap();
            conv.set_conversation_branches(tl, &map(&[("candle", "zen/work")]));
            assert_eq!(conv.conversation_state(tl), Some(expected.clone()));
            conv.flush_writer();
            conv.commit_persistence().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let p = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let conv = Conversation::from_parts(substrate, p);
            conv.register_timeline(tl, layer, group);
            assert_eq!(conv.conversation_state(tl), Some(expected));
            let files = conv.conversation_files(tl).unwrap();
            assert_eq!(files.len(), 1);
            assert_eq!(files[0].key, "src/lib.rs");
            assert_eq!(files[0].body, write.events[0].body);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A read-only conversation changes RAM and writes nothing.** Over a
    /// store opened with `SharedSubstrate::open_in_read_only`: a label, a
    /// debug id, a coupling, a turn tombstone and a timeline tombstone all show
    /// in the in-RAM view; the synchronous durable paths — signatures, index
    /// page, tokens, stream commit, template, commit, compaction, forced
    /// maintenance — all return `Ok`; no tokens location is registered for a
    /// record that was never written; and every file is byte-identical after
    /// the conversation drops.
    #[test]
    fn a_read_only_conversation_changes_ram_and_writes_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "candle-conv-read-only-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        {
            let mut substrate = Substrate::new();
            let mut p = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            p.write_debug_id(7, "seed", p.mint()).unwrap();
            p.commit().unwrap();
        }
        let before = dir_fingerprint(&dir);

        let conv = Conversation::from_shared(SharedSubstrate::open_in_read_only(&dir).unwrap());
        assert!(conv.is_read_only_substrate());
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let tl = TimelineId::from_raw(7).expect("timeline id");
        conv.register_timeline(tl, layer, group);

        conv.set_conversation_label(tl, "renamed").unwrap();
        assert_eq!(conv.label_of(tl).as_deref(), Some("renamed"));
        conv.set_conversation_debug_id(tl, "resume-key").unwrap();
        assert_eq!(conv.lookup_by_debug_id("resume-key"), Some(tl));
        conv.couple_turn(tl, 3).unwrap();
        assert!(conv.read().couplings_of(tl).contains(&3));
        conv.tombstone_turn(tl, 4).unwrap();
        assert!(conv.is_turn_tombstoned(tl, 4));

        let stream = turn_stream_id(tl.raw(), 0);
        conv.persist_wide_q_sigs(stream, &[1, 2, 3]).unwrap();
        conv.persist_index_page(stream, &[4, 5]).unwrap();
        conv.persist_tokens_only(stream, &[9, 10]).unwrap();
        conv.commit_stream_through(stream, 0).unwrap();
        conv.set_template(b"template").unwrap();
        conv.commit_persistence().unwrap();
        assert!(!conv.commit_persistence_if_pending().unwrap());
        conv.compact_substrate(None).unwrap();
        assert!(!conv.force_compact_persistence().unwrap());
        assert!(conv
            .read()
            .stream_of(stream)
            .and_then(|s| s.tokens)
            .is_none());

        conv.tombstone_timeline(tl).unwrap();
        assert!(conv.is_timeline_tombstoned(tl));
        conv.flush_writer();
        drop(conv);
        assert_eq!(dir_fingerprint(&dir), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A stream's commit survives the compaction that runs in the process
    /// that wrote it.** A prompt section's chunks reach the log first and its
    /// `Commit` afterwards, through `commit_stream_through`. Compaction rewrites
    /// a `Commit` from the index's `committed_through`, never from the log, so
    /// a commit only the log knew vanished with the first rewrite — the next
    /// boot found the section's chunks with no commit, refused to restore it,
    /// and prefilled the whole tool catalog again, on every boot.
    #[test]
    fn a_section_commit_survives_compaction_in_the_writing_process() {
        use crate::persistence::content_hash::ContentHash;
        use crate::persistence::record::ChunkPayload;
        use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl};

        let dir = std::env::temp_dir().join(format!(
            "candle-conv-section-commit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 0x11, hi: 0x12 },
                section_hash: ContentHash { lo: 0x13, hi: 0x14 },
            },
            debug_name: "tool_catalog_member".to_string(),
        });
        let stream = decl.stream_id();
        let payload = |seed: u8| ChunkPayload {
            offset: 0,
            k_formats: vec![4, 4, 4, 4],
            v_formats: vec![5, 5, 5, 5],
            k_pal: vec![seed; 4],
            v_pal: vec![seed; 2],
            k_scale: vec![1.0, 0.5],
            v_scale: vec![0.25],
            kv_bytes: vec![seed; 512],
        };
        // The chunks and tokens are durable; the commit is not yet written.
        {
            let mut substrate = Substrate::new();
            let mut p = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            p.declare_stream(&decl).unwrap();
            p.write_chunk(stream, 0, 32, 4, None, &payload(1)).unwrap();
            p.write_chunk(stream, 1, 32, 4, None, &payload(2)).unwrap();
            p.append_tokens(stream, &[7, 0, 0, 0, 8, 0, 0, 0]).unwrap();
            p.commit().unwrap();
        }

        {
            let conv =
                Conversation::from_shared(SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap());
            assert!(!conv.section_stream_is_persisted(stream));
            conv.commit_stream_through(stream, 1).unwrap();
            assert!(
                conv.section_stream_is_persisted(stream),
                "the writing process must see its own commit"
            );
            conv.compact_substrate(None).unwrap();
            assert!(conv.section_stream_is_persisted(stream));
            conv.flush_writer();
        }

        let reopened =
            Conversation::from_shared(SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap());
        assert_eq!(
            reopened
                .read()
                .stream_of(stream)
                .and_then(|s| s.committed_through),
            Some(1),
            "the commit must survive the compaction that rewrote its segment"
        );
        assert!(reopened.section_stream_is_persisted(stream));
        drop(reopened);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
