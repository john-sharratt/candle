//! Tests that characters, custom objects and tombstone markers survive
//! maintenance.
//!
//! Each of these lives in a segment that a pass rewrites or drops, and each
//! would vanish silently if maintenance did not carry it. The tests force a pass
//! over the segment holding the record, then reload the log and check that the
//! record is still there and that its tombstone still holds.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::streams::StreamId;

    use crate::persistence::{Admit, SubstratePersistence};

    use crate::projection::TimelineId;

    use crate::substrate::Substrate;

    use super::super::tests_support::tests::{chunk_payload, sealed_log, tmp_dir, turn_decl};

    /// **A character survives the maintenance of the segment holding it.**
    ///
    /// Maintenance is a second carry-forward path beside
    /// [`super::compaction::collect_live_records`], with its own two sets: the
    /// resident records it re-emits from substrate RAM, and the read-back
    /// records it relocates verbatim. A character can only be in the second —
    /// its payload belongs to the daemon above and is not in this process at all
    /// — and it was in neither, so the first pass that touched its segment
    /// deleted it.
    ///
    /// That is the bug that lost a Maker roughly every quarter of an hour: the
    /// character was written into the active segment, the segment sealed, and
    /// the next maintenance pass rewrote it out of existence. Fixing the
    /// duplicate-handle bug was not enough on its own, because this path never
    /// consulted `npc_locs` at all.
    #[test]
    fn a_character_survives_segment_maintenance() {
        let dir = tmp_dir("npc_maintenance");
        let sid = StreamId(303);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // seg 1: the character, beside a chunk that a later write supersedes
            // — the dead weight is what makes maintenance choose this segment.
            sp.write_npc(&crate::persistence::test_npc(7, "Wyneth"))
                .unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                sp.npc_locs.get(&7).map(|l| l.segment),
                Some(SegmentId(1)),
                "the character starts in the segment about to be maintained"
            );
            // Forced, so the test does not have to wait out the settle age —
            // the same phased path the daemon runs, and the same path the
            // `POST /v1/substrate/zend/maintenance` trigger takes.
            let plan = sp
                .plan_maintenance(&substrate, true)
                .unwrap()
                .expect("seg 1 carries dead weight, so a pass runs");
            let op = plan.op();
            let result = sp.execute_maintenance(&plan).unwrap();
            let stranded = result.apply_to_substrate(&plan, &mut substrate);
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            assert!(
                op.targets().contains(&SegmentId(1)),
                "the pass targets the segment holding the character, got {op:?}"
            );
            assert_eq!(
                sp.read_npc(7).unwrap().map(|n| n.name),
                Some("Wyneth".to_string()),
                "the character is readable immediately after the pass"
            );
        }
        // And across a restart, which is where the loss actually showed.
        let mut substrate = Substrate::new();
        let mut cast = Vec::new();
        SubstratePersistence::open_in_with_substrate_and_sink(
            &dir,
            &mut substrate,
            Admit::Unbound,
            |entry| {
                if entry.record.header.record_type == RecordType::Npc {
                    cast.push(entry.record.header.stream_id);
                }
            },
            &|_, _, _| {},
        )
        .unwrap();
        assert_eq!(
            cast,
            vec![7],
            "the character must still be on disk after maintenance rewrote its segment"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `CustomObject` (npcd's command-table flag, and host state generally)
    /// survives a maintenance pass that rewrites its segment — the
    /// content-hash-keyed twin of [`a_character_survives_segment_maintenance`].
    /// It is protected two ways, both exercised here: its location is tracked in
    /// `metadata_locs` (so [`SubstratePersistence::resident_to_carry`] re-emits
    /// it when its segment is a target) and [`segment_liveness`] counts it
    /// (so its segment is not understated as reclaimable and churned forward each
    /// pass). Before both, the object was neither tracked nor counted: the flag
    /// read back missing after maintenance rewrote its segment — the same silent
    /// loss the character bug had.
    #[test]
    fn a_custom_object_survives_segment_maintenance() {
        use crate::persistence::record::CustomObjectPayload;
        let dir = tmp_dir("custom_object_maintenance");
        let sid = StreamId(404);
        let key = "command-table";
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // seg 1: the object, beside a chunk a later write supersedes — the
            // dead weight is what draws maintenance to this segment.
            let obj = CustomObjectPayload::new(key, serde_json::json!({ "open": true }));
            let id = sp.mint();
            sp.write_custom_object(&obj, id).unwrap();
            substrate.apply_custom_object(obj, id);
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let stream_id = CustomObjectPayload::stream_id_for(key);
            assert_eq!(
                sp.metadata_locs
                    .get(&(RecordType::CustomObject, stream_id, 0))
                    .map(|l| l.segment),
                Some(SegmentId(1)),
                "the object's location is tracked, in the segment about to be maintained"
            );
            assert!(
                sp.segment_liveness(&substrate)
                    .get(&SegmentId(1))
                    .copied()
                    .unwrap_or(0)
                    > 0,
                "the live object pins its segment's weight, so it does not read as fully dead"
            );
            // Forced, so the test does not wait out the settle age — the same
            // phased path the daemon runs.
            let plan = sp
                .plan_maintenance(&substrate, true)
                .unwrap()
                .expect("seg 1 carries dead weight, so a pass runs");
            let op = plan.op();
            let result = sp.execute_maintenance(&plan).unwrap();
            let stranded = result.apply_to_substrate(&plan, &mut substrate);
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            assert!(
                op.targets().contains(&SegmentId(1)),
                "the pass targets the segment holding the object, got {op:?}"
            );
        }
        // Across a restart, which is where the loss actually showed.
        {
            let mut substrate = Substrate::new();
            SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                substrate.custom_object(key).map(|o| o.blob.clone()),
                Some(serde_json::json!({ "open": true })),
                "the object must still be on disk after maintenance rewrote its segment"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tombstoned timeline whose decl + chunk live only in a dropped segment
    /// must stay tombstoned after the drop — the re-emitted `Tombstone` marker
    /// keeps it dead so a reload can't resurrect it.
    #[test]
    fn tombstone_marker_survives_a_segment_drop() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let dir = tmp_dir("tomb");
        let tid = 777u64;
        let turn = StreamDecl::Turn(TurnDecl {
            timeline_id: tid,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 16,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = turn.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&turn).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1: the tombstoned-to-be turn's decl + chunk
            sp.write_tombstone(tid, None).unwrap(); // marker lands in seg 2 (active)
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(tid).unwrap();
            assert!(substrate.is_tombstoned(tl));
            // seg 1 holds only the tombstoned turn's records → fully dead → drop.
            assert!(sp.run_maintenance(&mut substrate).unwrap().is_some());
            assert!(!sealed_log(&dir, 1).exists());
        }
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(
                substrate.is_tombstoned(TimelineId::from_raw(tid).unwrap()),
                "the tombstone survived the drop that removed the timeline's records"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Characters survive the compaction of the segment holding them.
    ///
    /// Counting an NPC's bytes as live (`live_bytes_by_segment`) is not enough
    /// and reads as though it were: it keeps a segment off the `live_bytes == 0`
    /// **Drop** path only. `Compact` qualifies on a dead *ratio* — a segment a
    /// few KB of which is the entire cast still reads ~100% dead — and retires
    /// the segment just the same. So the cast has to be relocated, exactly as
    /// full compaction relocates it.
    #[test]
    fn characters_survive_a_segment_compaction() {
        use crate::persistence::record::{Modulation, NpcPayload};

        let dir = tmp_dir("npc");
        let npc = NpcPayload {
            npc_id: 8812,
            owner_id: "u_8812".to_string(),
            revision: 1,
            created_ms: 1,
            updated_ms: 1,
            state: "active".to_string(),
            name: "Commander".to_string(),
            world_id: "battle-cities".to_string(),
            personality_id: "commander".to_string(),
            hidden: false,
            heartbeat_ms: 1000,
            salience_gate: 0.5,
            tags: vec!["test".to_string()],
            persona_description: "Holds the line.".to_string(),
            persona_origin: "authored".to_string(),
            portrait_image_id: None,
            portrait_origin: None,
            // The lived and authoring planes are irrelevant to what this test
            // asserts — that a character's record is carried verbatim across a
            // maintenance pass — so they take their unauthored values.
            at: None,
            mood: None,
            suspended_from: None,
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Modulation::default(),
        };
        let decl = turn_decl(626, 0);
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            // The cast lands in seg 1 with a chunk that seg 2 supersedes, so
            // seg 1 is mostly dead and qualifies to compact.
            sp.write_npc(&npc).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                sp.npc_locs().get(&8812).map(|l| l.segment),
                Some(SegmentId(1)),
                "the character starts on the segment about to be compacted",
            );
            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();
            assert!(!sealed_log(&dir, 1).exists());
            assert_ne!(
                sp.npc_locs().get(&8812).map(|l| l.segment),
                Some(SegmentId(1)),
                "npc_locs was repointed at the relocated copy",
            );
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                sp.read_npc(8812).unwrap(),
                Some(npc),
                "the character survived the compaction of its segment, byte for byte",
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
