//! Tests of the durability rules: staged bytes stay bounded, written-through records reopen cleanly, and a read-only handle changes nothing.
//!
//! The staging buffer stays under the write-through bound plus one record, and bytes written through still owe a commit. A handle dropped mid-run loses only its staged tail, and the log opens on a record boundary. A read-only handle refuses every write, and the store is byte-identical after it drops. Active segments roll over by byte count on the append path, so each stays near the target.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::identity::RecordedModel;

    use super::super::tests_support::tests::{
        cast_on_disk, chunk_payload, model, recorded, section, tmp_dir,
    };

    /// **The staging buffer never holds more than the write-through bound plus
    /// one record**, and bytes written through still owe a commit: 70 one-MiB
    /// appends with no commit between them leave under 65 MiB staged, the rest
    /// on the segment, and `commit_if_pending` syncs once and then idles.
    #[test]
    fn a_long_append_run_writes_through_and_still_owes_a_commit() {
        let dir = tmp_dir("write_through");
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        sp.commit().unwrap();
        let start = sp.write_offset();
        let payload = vec![7u8; 1024 * 1024];
        let mut record = 0;
        for chunk in 0..70u64 {
            let at = sp
                .append_record(RecordType::Tokens, 0, 9, chunk, 0, 0, &payload)
                .unwrap();
            record = at.size as usize;
            assert!(sp.pending_bytes() < STAGE_FLUSH_BYTES + record);
        }
        assert!(sp.write_offset() > start);
        assert!(sp.pending_bytes() < STAGE_FLUSH_BYTES);
        assert!(sp.commit_if_pending().unwrap());
        assert_eq!(sp.pending_bytes(), 0);
        assert_eq!(sp.write_offset(), start + 70 * record as u64);
        assert!(!sp.commit_if_pending().unwrap());
        drop(sp);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **Records written through and never committed reopen cleanly**: a
    /// handle dropped mid-run — a crash, as far as the log can tell — leaves
    /// the written-through records on the segment and loses only the staged
    /// tail, and the log opens on a record boundary with a whole record count.
    #[test]
    fn records_written_through_but_not_committed_reopen_cleanly() {
        let dir = tmp_dir("write_through_reopen");
        let (start, record) = {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.commit().unwrap();
            let start = sp.write_offset();
            let payload = vec![7u8; 1024 * 1024];
            let mut record = 0;
            for chunk in 0..70u64 {
                let at = sp
                    .append_record(RecordType::Tokens, 0, 9, chunk, 0, 0, &payload)
                    .unwrap();
                record = at.size;
            }
            assert!(
                sp.write_offset() > start,
                "part of the run was written through"
            );
            (start, record)
        };
        let mut substrate = Substrate::new();
        let reopened = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let kept = reopened.write_offset() - start;
        assert_eq!(kept % record, 0, "the log ends on a record boundary");
        assert!(
            kept >= STAGE_FLUSH_BYTES as u64 && kept < 70 * record,
            "kept {kept} bytes"
        );
        drop(reopened);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A read-only handle refuses every write and changes no byte.** Built
    /// over a store with a sealed segment, a section, a model spec and a bound
    /// tokenizer: the appends refuse without staging, the commit paths succeed
    /// with nothing to do, the singletons report nothing written and keep the
    /// recorded values, a different tokenizer is still refused, maintenance
    /// plans nothing even when forced, compaction refuses — and every file is
    /// byte-identical after the handle drops.
    #[test]
    fn a_read_only_handle_refuses_writes_and_changes_nothing() {
        let dir = tmp_dir("read_only");
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&section("sys", 1)).unwrap();
            sp.bind_model(&model("spec-a", 1)).unwrap();
            sp.set_tokenizer(b"tokenizer-a").unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
        }
        let before = dir_fingerprint(&dir);

        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate_read_only(&dir, &mut substrate).unwrap();
        assert!(sp.is_read_only());
        let offset = sp.write_offset();
        assert!(matches!(
            sp.append_record(RecordType::Tokens, 0, 9, 0, 0, 0, b"refused"),
            Err(PersistenceError::ReadOnly)
        ));
        assert!(matches!(
            sp.write_tombstone(9, None),
            Err(PersistenceError::ReadOnly)
        ));
        assert_eq!(sp.pending_bytes(), 0);
        assert_eq!(sp.write_offset(), offset);
        sp.commit().unwrap();
        assert!(!sp.commit_if_pending().unwrap());
        sp.seal_active().unwrap();
        assert!(matches!(
            sp.bind_model(&model("spec-b", 2)),
            Err(PersistenceError::WrongModel { .. })
        ));
        // The same model under another description would be rewritten on a
        // writable handle; here nothing is.
        assert_eq!(
            sp.bind_model(&model("spec-a-renamed", 1)).unwrap(),
            ModelBinding::Redescribed
        );
        assert_eq!(recorded(&sp), Some(RecordedModel::Key(model("spec-a", 1))));
        assert!(!sp.set_template(b"template").unwrap());
        assert_eq!(sp.template(), None);
        assert!(sp.set_tokenizer(b"tokenizer-b").is_err());
        assert!(!sp.set_tokenizer(b"tokenizer-a").unwrap());
        assert!(sp.plan_maintenance(&substrate, true).unwrap().is_none());
        assert!(matches!(
            sp.compact(&mut substrate, None),
            Err(PersistenceError::ReadOnly)
        ));
        assert!(sp.has_stream(&substrate, section("sys", 1).stream_id()));
        drop(sp);
        assert_eq!(dir_fingerprint(&dir), before);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A released writer hands the directory on while it is still alive.**
    /// What it committed is on disk, it refuses every write afterwards, and the
    /// next writable open is admitted at once rather than at its drop.
    #[test]
    fn a_released_writer_admits_the_next_and_refuses_writes() {
        let dir = tmp_dir("release_writer");
        let mut substrate = Substrate::new();
        let mut first = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        first.write_npc(&test_npc(1, "Perrin")).unwrap();
        let mut second_substrate = Substrate::new();
        assert!(matches!(
            SubstratePersistence::open_in_with_substrate(&dir, &mut second_substrate),
            Err(PersistenceError::Locked { .. })
        ));

        first.release_writer().unwrap();
        first.release_writer().unwrap();
        assert!(first.is_read_only());
        assert!(matches!(
            first.append_record(RecordType::Tokens, 0, 9, 0, 0, 0, b"refused"),
            Err(PersistenceError::ReadOnly)
        ));
        assert!(matches!(
            first.compact(&mut substrate, None),
            Err(PersistenceError::ReadOnly)
        ));
        assert_eq!(first.pending_bytes(), 0);

        let mut second =
            SubstratePersistence::open_in_with_substrate(&dir, &mut second_substrate).unwrap();
        second.write_npc(&test_npc(2, "Wyneth")).unwrap();
        second.commit().unwrap();
        drop(first);
        drop(second);
        assert_eq!(
            cast_on_disk(&dir),
            vec![1, 2],
            "the record committed at release and the successor's both survive"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A read-only open never makes a store: a workspace without one, and a
    /// workspace that does not exist, are errors and stay as they were.
    #[test]
    fn a_read_only_open_of_a_workspace_without_a_store_creates_nothing() {
        let dir = tmp_dir("read_only_missing");
        assert!(SharedSubstrate::open_in_read_only(&dir).is_err());
        assert!(dir_fingerprint(&dir).is_empty());
        let missing = dir.join("absent");
        assert!(SharedSubstrate::open_in_read_only(&missing).is_err());
        assert!(!missing.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_creates_substrate_dir_and_log() {
        let dir = tmp_dir("open");
        {
            let sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.inherited_count(), 0);
        }
        // A fresh store mints the first active segment.
        assert!(dir.join(SUBSTRATE_DIR).join("seg-0000000001.log").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A long append batch (many records staged before any external commit)
    /// must roll segments **on the append path**, bounding each to ~target —
    /// not overshoot by the whole batch. Regression: with only commit-boundary
    /// rotation the active grew to ~2× the target (observed 8 GB segments
    /// against the 4 GB target), because `maybe_rotate_active` runs after a
    /// group-commit and a whole migration/ingest batch stages between commits.
    #[test]
    fn append_batch_rotates_by_bytes_not_only_at_commit() {
        let dir = tmp_dir("append_rotate");
        let sid = StreamId(9091);
        // Tiny target so a modest batch crosses it many times.
        let target = 128 * 1024u64;
        let mut sp = SubstratePersistence::open_in(&dir).unwrap();
        sp.segments.set_target_bytes_for_test(target);

        // Write a batch far larger than the target with NO intervening commit.
        let n = 2000u32;
        for i in 0..n {
            sp.write_chunk(sid, i as u64, 32, 4, None, &chunk_payload(i))
                .unwrap();
        }
        // Rotation already fired on the append path — before we ever commit.
        // (Pre-fix: the whole batch stayed in one active until the end-commit,
        // giving ~1 segment; the fix seals a fresh one every ~target bytes.)
        let mid_batch_segments = sp.segment_count();
        assert!(
            mid_batch_segments > 4,
            "append-path rotation should have sealed several segments mid-batch, got {mid_batch_segments}"
        );
        sp.commit().unwrap();

        // Every sealed segment's on-disk size is bounded to ~target (+ one
        // record of slack): the append-path check rolls before overshoot, and
        // seal truncates the over-allocated tail so file size == record bytes.
        for &id in sp.segments.sealed_ids() {
            let bytes = sp.segments.sealed_record_bytes(id).unwrap();
            assert!(
                bytes <= target + 64 * 1024,
                "sealed segment {id:?} = {bytes} B exceeds target {target} B + slack"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
