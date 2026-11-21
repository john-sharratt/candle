//! Tests that the live set keeps only the winning record of each key, and that
//! the compacted log recovers to the same live state.
//!
//! A stale singleton and a superseded partial chunk are not carried. A stream
//! without a declaration is an orphan, so its chunks are reclaimed, while
//! singletons survive. Re-walking the compacted log restores the same live
//! streams and chunks. A guard test requires every record type that must survive
//! to appear in the live set, so a new type cannot be left out unnoticed.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::identity::RecordIdentity;

    use crate::persistence::log_file::{read_record_at, LogSource, MemLog, SUPERBLOCK_SIZE};

    use crate::persistence::manifest::ConvState;

    use crate::persistence::rebase::RebaseEntry;

    use crate::persistence::vfs::VfsEventId;

    use super::super::tests_support::tests::{
        chunk_record, has_synth, has_type, loc_at, record, NO_EXPIRY,
    };

    /// **The live list shares a turn's metadata blobs; it does not copy them.**
    /// The whole list exists before the first byte is written, so a copy is a
    /// second image of every live turn's index page and wide-Q signature held
    /// for the length of the rewrite.
    #[test]
    fn the_live_list_shares_the_substrates_blobs() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        use crate::substrate::HeldPayloadBytes;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 6,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        let mut mem =
            MemLog::with_records(&record(RecordType::StreamDecl, sid.0, 0, &decl.encode()));
        let (manifest, mut substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let none = RecordIdentity::NONE;
        substrate.apply_projection_events(sid, vec![1, 2, 3], none);
        substrate.apply_wide_q_sigs(sid, vec![4, 5, 6, 7], none);
        substrate.apply_index_page(sid, vec![8, 9], none);
        // The payloads held whole are what the memory report counts.
        assert_eq!(
            substrate.held_payload_bytes(),
            HeldPayloadBytes {
                wide_q_sigs: 4,
                projection_events: 3,
            }
        );

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
        let entry = substrate.stream_of(sid).unwrap();
        let payload_of = |rt: RecordType| {
            live.iter()
                .find_map(|it| match it {
                    CompactItem::Synth { header, payload }
                        if header.record_type == rt && header.stream_id == sid.0 =>
                    {
                        Some(payload)
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{rt:?} is missing from the live list"))
        };

        let events = payload_of(RecordType::ProjectionEvents);
        assert_eq!(&events[..], &[1, 2, 3]);
        assert!(Arc::ptr_eq(
            events,
            entry.projection_events.as_ref().unwrap()
        ));

        let sigs = payload_of(RecordType::WideQSig);
        assert_eq!(&sigs[..], &[4, 5, 6, 7]);
        assert!(Arc::ptr_eq(sigs, entry.wide_q_sigs.as_ref().unwrap()));

        let page = payload_of(RecordType::TurnIndexPage);
        assert_eq!(&page[..], &[8, 9]);
        assert!(Arc::ptr_eq(
            page,
            &entry
                .index_page
                .as_ref()
                .unwrap()
                .resident()
                .unwrap()
                .shared_bytes()
        ));
    }

    #[test]
    fn collect_keeps_only_the_live_winners() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let mut blob = Vec::new();
        // A stale ModelKey, then the live one.
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model-v1-stale"));
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model-v2-live"));
        // The stream's decl — without it the chunk is an orphan and is reclaimed.
        let decl5 = StreamDecl::Turn(TurnDecl {
            timeline_id: 5,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        blob.extend_from_slice(&record(RecordType::StreamDecl, 5, 0, &decl5.encode()));
        // A stream: a 20-token partial chunk, then the sealed winner.
        blob.extend_from_slice(&chunk_record(5, 0, b"partial-20tok-dead"));
        blob.extend_from_slice(&chunk_record(5, 0, b"sealed-final-live"));
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
        // Exactly: 1 ModelKey + 1 StreamDecl + 1 Chunk = 3 (LWW keeps one of each).
        assert_eq!(live.len(), 3);
        // Each read-back item points at the live winner's on-disk location — the
        // manifest / substrate index already resolved last-writer-wins — so the
        // stale ModelKey and the dead partial chunk are never staged. Read the
        // chosen source bytes back to prove which record was selected.
        let model = live
            .iter()
            .find(|it| it.header().record_type == RecordType::ModelKey)
            .unwrap();
        let (off, size) = raw_loc(model);
        assert_eq!(
            read_record_at(&mut mem, off, size).unwrap().payload,
            b"model-v2-live",
            "the stale ModelKey is dropped"
        );
        let chunk = live
            .iter()
            .find(|it| it.header().record_type == RecordType::Chunk)
            .unwrap();
        let (off, size) = raw_loc(chunk);
        let chunk_rec = read_record_at(&mut mem, off, size).unwrap();
        let chunk_cp =
            crate::persistence::record::ChunkPayload::decode(&chunk_rec.payload).unwrap();
        assert_eq!(
            chunk_cp.kv_bytes, b"sealed-final-live",
            "the dead partial chunk is dropped"
        );
    }

    /// An orphaned stream — KV chunks (and tokens) on disk but NO `StreamDecl` —
    /// is reclaimed by compaction, not copied forward. Regression for the
    /// silent-loss bloat bug: a decl lost in a prior generation must not leave
    /// its (huge) KV immortal on disk. The turn is already unreloadable without
    /// its decl, so dropping the chunks loses nothing recoverable and reclaims
    /// the space.
    #[test]
    fn orphan_stream_without_decl_is_reclaimed() {
        let mut blob = Vec::new();
        // Two orphan streams: one with a chunk, one with a chunk + tokens. No
        // StreamDecl for either. Plus a legit ModelKey singleton to prove only
        // the orphans are dropped.
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model"));
        blob.extend_from_slice(&chunk_record(9001, 0, b"orphan-kv-a"));
        blob.extend_from_slice(&chunk_record(9002, 0, b"orphan-kv-b"));
        blob.extend_from_slice(&record(RecordType::Tokens, 9002, 0, b"orphan-tokens"));
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
            !has_type(&live, RecordType::Chunk),
            "orphan chunks (no StreamDecl) must be reclaimed, not kept forever",
        );
        assert!(
            !has_type(&live, RecordType::Tokens),
            "an orphan stream's tokens are reclaimed with it",
        );
        // The legit singleton survives — only orphan streams are dropped.
        assert!(has_type(&live, RecordType::ModelKey));
    }

    #[test]
    fn compacted_log_recovers_to_an_identical_live_set() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"m1"));
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"m2"));
        // The stream's decl — without it the chunks are orphans and are reclaimed.
        let decl7 = StreamDecl::Turn(TurnDecl {
            timeline_id: 7,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 2,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        blob.extend_from_slice(&record(RecordType::StreamDecl, 7, 0, &decl7.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"c0-old"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"c0-new"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 1, b"c1"));
        blob.extend_from_slice(&record(RecordType::Tokens, 7, 0, b"tokens"));
        let mut mem = MemLog::with_records(&blob);
        let (before, before_sub, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(
            &before,
            &before_sub,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let path = std::env::temp_dir().join(format!(
            "kvtier_compact_{}.log",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut counted = Vec::new();
        let (mut new_log, _after_manifest) = write_compacted_log(
            &path,
            &live,
            &mut |_s, off, dest| mem.read_into(off, dest),
            None,
            &mut |n| counted.push(n),
        )
        .unwrap();
        // The counter only rises, and ends on the whole live set.
        assert!(counted.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(counted.last().copied(), Some(live.len()));

        // The compacted log re-walked into a substrate has the same
        // live streams + chunks as the source.
        let hint = new_log.superblock().last_index;
        assert_ne!(hint, (0, 0), "compacted log must carry an index chain");
        let mut after_sub = Substrate::new();
        let recovered = super::super::super::recovery::recover_with_sink(
            &mut new_log,
            super::super::super::segment::FIRST_SEGMENT,
            hint,
            |e| after_sub.apply_walker_entry(e),
        )
        .unwrap();
        assert_eq!(
            recovered.last_index,
            Some(hint),
            "recovery of a compacted log must take the chain path"
        );
        assert_eq!(before_sub.live_chunk_count(), after_sub.live_chunk_count());
        assert_eq!(after_sub.live_chunk_count(), 2);
        std::fs::remove_file(&path).ok();
    }

    /// **The guard that makes [`survival`] load-bearing.**
    ///
    /// Builds a store holding one record of every written type, compacts it, and
    /// requires that each type whose contract is `Relocated` or `Resident` comes
    /// out the other side. A type added to `RecordType` and to `survival` but not
    /// to `collect_live_records` fails here rather than deleting user data in
    /// production — which is exactly how `TurnCoupling`, `Npc`, and turn-scoped
    /// `Tombstone`s were lost: each compiled clean, passed every test, and the
    /// accounting classified it as neither live nor dead so nothing counted the
    /// loss either.
    ///
    /// [`survival`]: crate::persistence::survival::survival
    #[test]
    fn every_record_type_survives_compaction() {
        use crate::persistence::blob::BlobKey;
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{
            CustomObjectPayload, DistillMode, DistillPayload, TreeMetadataPayload,
        };
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        use crate::persistence::survival::{survival, Survival, WRITTEN_RECORD_TYPES};
        use std::collections::BTreeMap;

        let live_tl = 1001u64;
        let distilled_tl = 1002u64;
        let custom: BTreeMap<String, String> = BTreeMap::new();
        let decl = |timeline_id: u64, turn_index: u32| {
            StreamDecl::Turn(TurnDecl {
                timeline_id,
                turn_index,
                turn_id_day: 0,
                turn_id_seq: turn_index + 1,
                role: 2,
                block_start: 0,
                block_end: 2,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };
        let live_sid = 7001u64;
        let distilled_sid = 7002u64;

        let mut blob = Vec::new();
        // Singletons.
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model"));
        blob.extend_from_slice(&record(RecordType::Template, 0, 0, b"template"));
        blob.extend_from_slice(&record(RecordType::Tokenizer, 0, 0, b"tokenizer"));
        // A live turn, with every per-stream record it can carry.
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            live_sid,
            0,
            &decl(live_tl, 0).encode(),
        ));
        blob.extend_from_slice(&record(RecordType::Chunk, live_sid, 0, b"kv"));
        blob.extend_from_slice(&record(RecordType::Tokens, live_sid, 0, b"tokens"));
        blob.extend_from_slice(&record(RecordType::Commit, live_sid, 0, b""));
        blob.extend_from_slice(&record(
            RecordType::ProjectionEvents,
            live_sid,
            0,
            b"events",
        ));
        blob.extend_from_slice(&record(RecordType::TurnIndexPage, live_sid, 0, b"page"));
        blob.extend_from_slice(&record(
            RecordType::WideQSig,
            live_sid,
            0,
            &crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]),
        ));
        // Per-timeline metadata. `ConvState` is emitted only for a state that
        // differs from an untouched one, so archive this one and give it a
        // branch.
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::super::manifest::encode_label_payload(live_tl, "conv", "Live", &custom),
        ));
        blob.extend_from_slice(&record(
            RecordType::ConvState,
            live_tl,
            0,
            &encode_conv_state_payload(
                live_tl,
                &ConvState {
                    archived: true,
                    branches: [("candle".to_string(), "zen/work".to_string())].into(),
                    active: 3,
                },
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::TreeMetadata,
            0,
            0,
            &TreeMetadataPayload {
                timeline_id: live_tl,
                turn_index: 0,
                kind: 0,
                tree_height: 0,
                children: Vec::new(),
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::DebugId,
            0,
            0,
            &DebugIdPayload {
                timeline_id: live_tl,
                debug_id: "resume-key".to_string(),
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::TurnCoupling,
            0,
            0,
            &TurnCouplingPayload {
                timeline_id: live_tl,
                from_turn: 0,
            }
            .encode(),
        ));
        // A turn-scoped tombstone — on a turn of the LIVE timeline, so it is a
        // dead turn inside a living conversation rather than a dead timeline.
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: live_tl,
                turn_index: Some(1),
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        // Recurrent state: a conversation snapshot and a prompt-branch checkpoint.
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(live_tl).0,
            0,
            b"snapshot",
        ));
        blob.extend_from_slice(&record(RecordType::BranchCheckpoint, 9_001, 0, b"branch"));
        // A separate distilled timeline, so its marker is emitted without
        // shedding the live turn's content.
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            distilled_sid,
            0,
            &decl(distilled_tl, 0).encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: distilled_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        // A host-defined keyed object — the key's hash rides in the header
        // `stream_id`, exactly as `put_custom_object` writes it.
        let custom_obj = CustomObjectPayload::new(
            "command-table:vault",
            serde_json::json!({
                "open": true
            }),
        );
        blob.extend_from_slice(&record(
            RecordType::CustomObject,
            custom_obj.stream_id(),
            0,
            &custom_obj.encode(),
        ));
        // A tombstoned section stream — the section counterpart of the
        // turn-scoped `Tombstone` above.
        blob.extend_from_slice(&record(
            RecordType::SectionTombstone,
            9_501,
            0,
            &SectionTombstonePayload {
                generation: None,
                reason: None,
            }
            .encode(),
        ));
        // A dead timeline's tombstone, and the swept set naming it — a
        // compaction carries both while it is young.
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: 1003,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Deleted,
            0,
            0,
            &DeletedPayload {
                swept: vec![1003],
                ..DeletedPayload::default()
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        // Characters are tracked persistence-side, not in the substrate.
        let mut npc_locs = StampedLocs::new();
        npc_locs.offer(8812, Some(loc_at(4_096, 256)), RecordIdentity::NONE);
        // So are a conversation's file events and their tombstones.
        let mut vfs = VfsIndex::new();
        // The tombstone kills another member's event, so it is carried.
        vfs.record_event(
            live_tl,
            VfsEventId { origin: 0, seq: 0 },
            loc_at(8_192, 128),
        );
        vfs.record_tombstone(
            live_tl,
            VfsEventId { origin: 0, seq: 1 },
            loc_at(12_288, 64),
            &[VfsEventId {
                origin: 0xB,
                seq: 0,
            }],
        );

        // And blobs, by location.
        let mut blobs = BlobIndex::new();
        blobs.record(
            BlobKey { hi: 3, lo: 4 },
            loc_at(16_384, 512),
            RecordIdentity::NONE,
        );

        // What the member has applied of another origin, and the generation
        // fixed for what it wrote cut off — held persistence-side.
        let mut housekeeping = Housekeeping::default();
        housekeeping.replica.applied.insert(0xB, 1);
        housekeeping.rebases.add(RebaseEntry {
            origin: 0xA,
            from: 4,
            through: 9,
            gen: 2,
        });
        let live = collect_live_records(
            &manifest,
            &substrate,
            &npc_locs,
            &vfs,
            &blobs,
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &housekeeping,
        );

        for &rt in WRITTEN_RECORD_TYPES {
            match survival(rt) {
                Survival::Relocated | Survival::Resident => assert!(
                    has_type(&live, rt),
                    "{rt:?} declares survival::{:?} but compaction emitted no such \
                     record — it would be deleted from the store on the first pass",
                    survival(rt),
                ),
                // Derived: every copy is dropped and the writer rebuilds the
                // chain in the new file, so its absence here is correct.
                Survival::Regenerated => assert!(
                    !has_type(&live, rt),
                    "{rt:?} is regenerated by the writer; carrying a stale copy \
                     forward would corrupt the new chain",
                ),
                Survival::NeverWritten => unreachable!("{rt:?} is in WRITTEN_RECORD_TYPES"),
            }
        }

        // The conversation state comes through whole — its branches too, not
        // just the archived flag — keyed in the header by its timeline.
        let state = ConvState {
            archived: true,
            branches: [("candle".to_string(), "zen/work".to_string())].into(),
            active: 3,
        };
        assert!(has_synth(
            &live,
            RecordType::ConvState,
            &encode_conv_state_payload(live_tl, &state)
        ));
        assert!(live.iter().any(|it| {
            it.header().record_type == RecordType::ConvState && it.header().stream_id == live_tl
        }));
    }
}
