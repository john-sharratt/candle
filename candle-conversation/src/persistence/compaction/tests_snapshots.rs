//! Tests which recurrent-state snapshots survive compaction and index
//! maintenance.
//!
//! The snapshot of a distilled or tombstoned timeline is dropped, and that
//! retirement is recorded in the index, so every rewrite path sheds it. A live
//! timeline keeps exactly its newest snapshot. Branch checkpoints are capped,
//! and the newest survive. A checkpoint survives its timeline's tombstone,
//! because it belongs to the prompt and outlives every conversation that used
//! it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};

    use super::super::tests_support::tests::{
        has_synth, has_type, record, snapshot_payload, NO_EXPIRY,
    };

    /// **A distilled timeline sheds its recurrent snapshot; its sig survives.**
    ///
    /// The inverse of the test above, on the axis that was missing a gate.
    /// Distillation keeps the provenance signatures and drops the content, and
    /// the recurrent state is content — derived from the token stream, useless
    /// without the K/V shed alongside it. Ungated, a distilled timeline sheds
    /// megabytes of K/V and carries ~63 MiB of state forward on every pass, for
    /// a conversation that is never resumed.
    #[test]
    fn a_distilled_timelines_recurrent_snapshot_is_dropped() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let tl = 131u64;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: vec!["tool".to_string()],
        });
        let sid = 13131u64;
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);
        let snap = snapshot_payload(tl, 0, 0xAB);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, sid, 0, b"kv-chunk-content"));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snap,
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            !has_type(&live, RecordType::Snapshot),
            "a distilled timeline kept its recurrent snapshot — megabytes of \
             state carried forward on every compaction pass, for a conversation \
             that can never be resumed because its K/V is gone"
        );
        assert!(
            has_synth(&live, RecordType::WideQSig, &wide_payload),
            "the provenance signature must survive — that is what distillation \
             is for"
        );
    }

    /// A tombstoned timeline's snapshot is dropped too. The tombstone already
    /// removes the map entry on apply, but a timeline that is tombstoned AND
    /// distilled deliberately escapes the wholesale drop, so the snapshot gate
    /// checks the set rather than assuming the entry is gone.
    #[test]
    fn a_tombstoned_timelines_recurrent_snapshot_is_dropped() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::TombstonePayload;

        let tl = 132u64;
        let snap = snapshot_payload(tl, 3, 0xCD);
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snap,
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            !has_type(&live, RecordType::Snapshot),
            "a tombstoned timeline's recurrent snapshot survived compaction"
        );
    }

    /// **A distilled timeline's recurrent tail is retired at the INDEX**, so
    /// every rewrite path sheds it without having to consult the distilled set.
    ///
    /// The gate used to live only in `collect_live_records`. Incremental
    /// maintenance relocated the same tails forward on every pass, on the stated
    /// reasoning that "the substrate map only ever holds live conversations'
    /// tails" — true of a tombstone, which removes the entry, and false of a
    /// distill, which did not. Measured on a production store: 766 distilled
    /// timelines × ~116 MiB, about 89 GB of a 120 GB log, none of it resumable.
    ///
    /// Asserting on the index (`recurrent_snapshot_entries`) rather than on one
    /// path's output is the point — that is the property both paths read.
    #[test]
    fn a_distilled_timelines_recurrent_snapshot_is_retired_from_the_index() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{DistillMode, DistillPayload};

        let tl = 155u64;
        let snap = snapshot_payload(tl, 3, 0xAB);
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snap,
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        // The index itself — what maintenance's relocation worklist walks.
        assert_eq!(
            substrate.recurrent_snapshot_entries().count(),
            0,
            "a distilled timeline's tail is still live in the substrate index, so \
             incremental maintenance will relocate it forward forever",
        );
        // And therefore also absent from the full-compaction live set.
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            !has_type(&live, RecordType::Snapshot),
            "a distilled timeline's recurrent snapshot survived compaction",
        );
    }

    /// A LIVE timeline keeps its tail — the retirement must key on the marker,
    /// not fire for every snapshot that shares the walk.
    #[test]
    fn an_undistilled_timelines_recurrent_snapshot_is_kept() {
        use crate::persistence::content_hash::snapshot_stream_id;

        let tl = 156u64;
        let snap = snapshot_payload(tl, 3, 0xAB);
        let blob = record(RecordType::Snapshot, snapshot_stream_id(tl).0, 0, &snap);

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(substrate.recurrent_snapshot_entries().count(), 1);
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(has_type(&live, RecordType::Snapshot));
    }

    /// **Branch checkpoints are capped; conversation snapshots are not.**
    ///
    /// The one record here that is a cache: a checkpoint is a pure function of a
    /// prompt still on disk, so dropping one costs a prefill on next use, while
    /// keeping one costs ~63 MiB in every future segment. Nothing else reclaims
    /// them — they are keyed by *content*, so editing the prompt supersedes
    /// nothing and orphans the old branch, and no timeline tombstone names it.
    /// Without the cap the log grows by one orphan per prompt edit, forever.
    ///
    /// The newest survive: the drop is oldest-first by write position.
    #[test]
    fn branch_checkpoints_are_capped_and_the_newest_survive() {
        use crate::persistence::content_hash::{branch_checkpoint_stream_id, ContentHash};

        let n = MAX_BRANCH_CHECKPOINTS + 3;
        let mut blob = Vec::new();
        let mut ids = Vec::new();
        for i in 0..n {
            let sid = branch_checkpoint_stream_id(ContentHash {
                lo: 1000 + i as u64,
                hi: 0,
            });
            ids.push(sid.0);
            // Payload contents are irrelevant to the cap — it reads the index.
            blob.extend_from_slice(&record(
                RecordType::BranchCheckpoint,
                sid.0,
                0,
                &snapshot_payload(0, 0, i as u8),
            ));
        }

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        let kept: Vec<u64> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::BranchCheckpoint)
            .map(|it| it.header().stream_id)
            .collect();
        assert_eq!(
            kept.len(),
            MAX_BRANCH_CHECKPOINTS,
            "{n} checkpoints compacted to {} — the cap did not apply, and the log \
             grows without bound as the prompt is edited",
            kept.len()
        );
        // The three oldest are the ones dropped.
        for old in &ids[..3] {
            assert!(
                !kept.contains(old),
                "an older checkpoint survived while a newer one was dropped"
            );
        }
        for recent in &ids[3..] {
            assert!(
                kept.contains(recent),
                "a recent checkpoint was dropped while older ones survived — the \
                 branch a conversation is most likely to be using is the one that \
                 would have to be recomputed"
            );
        }
    }

    /// A checkpoint is not a conversation snapshot: distilling or tombstoning a
    /// timeline must not touch one, because it belongs to the *prompt* and
    /// outlives every conversation that used it.
    #[test]
    fn a_branch_checkpoint_survives_a_tombstoned_timeline() {
        use crate::persistence::content_hash::{
            branch_checkpoint_stream_id, snapshot_stream_id, ContentHash,
        };
        use crate::persistence::record::TombstonePayload;

        let tl = 141u64;
        let branch = branch_checkpoint_stream_id(ContentHash { lo: 77, hi: 0 });
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snapshot_payload(tl, 1, 0xAB),
        ));
        blob.extend_from_slice(&record(
            RecordType::BranchCheckpoint,
            branch.0,
            0,
            &snapshot_payload(0, 0, 0xCD),
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            !has_type(&live, RecordType::Snapshot),
            "the tombstoned conversation's snapshot should be gone"
        );
        assert!(
            has_type(&live, RecordType::BranchCheckpoint),
            "the prompt's branch checkpoint was dropped with the conversation — \
             it belongs to the prompt, and every other conversation on that \
             branch would have to recompute it"
        );
    }

    /// A live timeline keeps exactly one snapshot — the newest. The header key
    /// is a synthetic per-timeline stream id, so the last writer wins by append
    /// order and the accounting *is* the tombstone; this pins that the surviving
    /// bytes are the SECOND write, not merely that one survived.
    #[test]
    fn a_live_timeline_keeps_exactly_its_newest_snapshot() {
        use crate::persistence::content_hash::snapshot_stream_id;

        let tl = 133u64;
        let first = snapshot_payload(tl, 0, 0x11);
        let second = snapshot_payload(tl, 1, 0x22);
        let sid = snapshot_stream_id(tl).0;

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Snapshot, sid, 0, &first));
        blob.extend_from_slice(&record(RecordType::Snapshot, sid, 0, &second));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        let snaps: Vec<_> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::Snapshot)
            .collect();
        assert_eq!(snaps.len(), 1, "exactly one snapshot survives per timeline");
        assert_eq!(
            snaps[0].header().payload_len as usize,
            second.len(),
            "the surviving snapshot must be the newest write"
        );
    }
}
