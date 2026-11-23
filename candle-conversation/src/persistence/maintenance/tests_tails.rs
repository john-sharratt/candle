//! Tests a conversation's single snapshot tail through reload and relocation.
//!
//! A newer snapshot supersedes the older, relocation moves only the live tail
//! and credits the exact old bytes as dead, and an op that ends before its apply
//! restores the tail map. Race tests check that a snapshot or index page written
//! between plan and execute is not overwritten by its stale copy, and that a
//! tail the index still names keeps its segment.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::identity::Stamped;

    use crate::persistence::streams::{StreamDecl, TurnDecl};

    use crate::persistence::SubstratePersistence;

    use crate::substrate::Substrate;

    use super::super::tests_support::tests::{sealed_log, snapshot_payload, tmp_dir};

    /// The single-tail contract end to end: a newer snapshot supersedes the
    /// older one across reload, its dead bytes are credited, segment
    /// maintenance relocates exactly the tail, and a timeline tombstone kills
    /// the entry.
    #[test]
    fn snapshot_single_tail_survives_reload_and_relocation() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{SnapshotLayer, SnapshotPayload};

        let dir = tmp_dir("snapshot_tail");
        let timeline: u64 = 42;
        let sid = snapshot_stream_id(timeline);
        let payload = |turn: u32, fill: u8| {
            SnapshotPayload {
                timeline_id: timeline,
                turn_index: turn,
                schedule_hash: 0xA5A5,
                layers: vec![SnapshotLayer {
                    layer_index: 0,
                    n_v_heads: 1,
                    d_v: 4,
                    d_k: 4,
                    state: vec![fill; 64],
                    conv_channels: 2,
                    conv_tail_cols: 2,
                    conv_tail: vec![fill; 16],
                }],
                aux: vec![fill; 24],
            }
            .encode()
        };

        // Write two snapshots for the same conversation; the second wins.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let dead_before = sp.accounting.dead_bytes();
            let a = sp.write_snapshot(sid, &[&payload(1, 0x11)]).unwrap();
            substrate.apply_snapshot_loc(sid, a.value, a.id);
            let b = sp.write_snapshot(sid, &[&payload(2, 0x22)]).unwrap();
            substrate.apply_snapshot_loc(sid, b.value, b.id);
            assert!(
                sp.accounting.dead_bytes() > dead_before,
                "the superseded snapshot must be credited as dead bytes"
            );
            sp.commit().unwrap();
        }

        // Reload: the walker replays both records in append order; the index
        // must hold exactly the newer one, and its payload must decode to
        // turn 2.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate
            .recurrent_snapshot_loc(sid)
            .expect("tail snapshot survives reload");
        let bytes = sp.read_record_payload(&tail).unwrap();
        let decoded = SnapshotPayload::decode(&bytes).unwrap();
        assert_eq!(decoded.turn_index, 2, "the newer snapshot is the tail");
        assert_eq!(decoded.layers[0].state[0], 0x22);
        assert_eq!(
            decoded.aux,
            vec![0x22u8; 24],
            "the model's own state rides the record, not just the layers"
        );

        // Relocation worklist contains exactly the tail (the dead copy is
        // invisible to the index).
        let (_, _, snapshots, _, _) = sp
            .live_indexes()
            .gather_relocations(&substrate, &[SegmentId(1)]);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].0, sid);

        // A timeline tombstone kills the entry.
        substrate.apply_tombstone(&crate::persistence::record::TombstonePayload {
            timeline_id: timeline,
            turn_index: None,
            turn_writer: 0,
            reason: None,
        });
        assert!(substrate.recurrent_snapshot_loc(sid).is_none());
        let (_, _, snapshots, _, _) = sp
            .live_indexes()
            .gather_relocations(&substrate, &[SegmentId(1)]);
        assert!(
            snapshots.is_empty(),
            "tombstoned snapshot must not relocate"
        );
    }

    /// Relocating a LIVE snapshot credits exactly the OLD copy's on-disk
    /// bytes as dead (the relocated copy supersedes it in the accounting
    /// map), repoints both indexes at the new location, and the relocated
    /// tail decodes correctly across a reload.
    #[test]
    fn snapshot_relocation_credits_exact_dead_bytes() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("snap_reloc_acct");
        let timeline: u64 = 91;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        // Two tails in seg 1: turn 1 is superseded (the segment's dead
        // weight), turn 2 is the live tail the compact must carry forward.
        let Stamped { value: loc_a, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 1, 0x11)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a, id);
        let Stamped { value: loc_b, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 2, 0x22)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b, id);
        sp.commit().unwrap();
        sp.seal_active().unwrap(); // seg 1 sealed; active is seg 2
        assert_eq!(
            sp.accounting.dead_bytes(),
            loc_a.record_size,
            "the superseded turn-1 copy is the only dead weight so far"
        );

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("seg 1 carries dead weight, force-compact qualifies");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));

        let result = sp.execute_maintenance(&plan).unwrap();
        assert_eq!(result.snapshot_updates.len(), 1);
        let (usid, old, new) = result.snapshot_updates[0];
        assert_eq!(usid, sid);
        assert_eq!(old, loc_b);
        assert_eq!(new.segment, SegmentId(2), "relocated into the active");
        assert_eq!(
            new.record_size, loc_b.record_size,
            "verbatim copy, same size"
        );
        assert_eq!(
            sp.accounting.dead_bytes(),
            loc_a.record_size + loc_b.record_size,
            "relocation credits exactly the OLD live copy's bytes as dead"
        );

        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(new));
        sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&new).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        drop(sp);

        // Reload: the relocated record is the tail the walk installs.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate
            .recurrent_snapshot_loc(sid)
            .expect("tail survives reload");
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&tail).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        assert_eq!(decoded.layers[0].state[0], 0x22);
        assert_eq!(
            decoded.aux,
            vec![0x22u8; 24],
            "relocation carries the model's own state too"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **The owner retires a drained segment; its file is unlinked apart.**
    /// `finish_maintenance` deletes nothing: the log stops naming the source
    /// and hands its file back. A crash before the unlink leaves the file
    /// behind, and a reload over it still installs the relocated copy, not
    /// the stale one.
    #[test]
    fn finish_retires_the_source_and_leaves_its_file_to_unlink() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("retire_apart");
        let timeline: u64 = 92;
        let sid = snapshot_stream_id(timeline);
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        for turn in [1, 2] {
            let Stamped { value: loc, id } = sp
                .write_snapshot(sid, &[&snapshot_payload(timeline, turn, turn as u8)])
                .unwrap();
            substrate.apply_snapshot_loc(sid, loc, id);
        }
        sp.commit().unwrap();
        sp.seal_active().unwrap();
        let before = sp.segment_count();
        let plan = sp.plan_maintenance(&substrate, true).unwrap().unwrap();
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
        let stranded = sp
            .execute_maintenance(&plan)
            .unwrap()
            .apply_to_substrate(&plan, &mut substrate);

        let retired = sp.finish_maintenance(&plan, &stranded).unwrap();
        assert_eq!(retired.paths(), [sealed_log(&dir, 1)]);
        assert!(sealed_log(&dir, 1).exists(), "the owner unlinked the file");
        assert_eq!(sp.segment_count(), before - 1, "the log still names seg 1");
        drop(sp);

        // A crash before the unlink: the stale source is still on disk.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate.recurrent_snapshot_loc(sid).expect("tail");
        assert_eq!(
            tail.segment,
            SegmentId(2),
            "the relocated copy, not the stale one"
        );
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&tail).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        drop(sp);

        let done = retired.unlink();
        assert_eq!(done.removed, 1, "{:?}", done.failed);
        assert!(!sealed_log(&dir, 1).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A turn's QSA index page, left in the log, survives a segment's
    /// maintenance byte for byte.** Recovered, the page is located, not held
    /// ([`IndexPage::Stored`]); a compaction of the segment holding it relocates
    /// the record by location and repoints the substrate at the copy, and the
    /// bytes read there — and after a reload — are exactly the bytes the seal
    /// wrote, the superseded page's included in none of it.
    #[test]
    fn a_located_index_page_relocates_byte_for_byte() {
        let dir = tmp_dir("index_page_reloc");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 5150,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
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
        let stale: Vec<u8> = (0..700u32).map(|i| (i % 13) as u8).collect();
        let page: Vec<u8> = (0..4_099u32)
            .map(|i| (i.wrapping_mul(37) % 251) as u8)
            .collect();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&decl).unwrap();
            // Two pages in seg 1: the first is the dead weight that qualifies
            // the segment for a compaction, the second the live page it carries.
            sp.append_turn_index_page(sid, &stale, sp.mint_for(RecordType::TurnIndexPage, sid))
                .unwrap();
            sp.append_turn_index_page(sid, &page, sp.mint_for(RecordType::TurnIndexPage, sid))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
        }
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let held = substrate.index_page_of(sid).expect("the page recovers");
        let old = held
            .stored()
            .expect("recovered, the page is located, not held");
        assert_eq!(old.segment, SegmentId(1));
        assert_eq!(
            held.read_with(&mut sp),
            page,
            "the seal's bytes, before relocation"
        );

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("seg 1 carries a superseded page, force-compact qualifies");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
        let result = sp.execute_maintenance(&plan).unwrap();
        assert_eq!(
            result.located_updates.len(),
            1,
            "the live page, and only it"
        );
        let ((_, rt), from, to) = result.located_updates[0];
        assert_eq!(rt, RecordType::TurnIndexPage);
        assert_eq!(from, old);
        assert_ne!(
            to.segment,
            SegmentId(1),
            "relocated off the compacted segment"
        );
        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");

        let moved = substrate
            .index_page_of(sid)
            .expect("the page survives the compaction");
        assert_eq!(moved.stored(), Some(to), "the substrate points at the copy");
        assert_eq!(
            moved.read_with(&mut sp),
            page,
            "the seal's bytes, after relocation"
        );
        drop(sp);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let reloaded = substrate
            .index_page_of(sid)
            .expect("the page survives a reload");
        assert_eq!(
            reloaded.read_with(&mut sp),
            page,
            "the seal's bytes, after a reload"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A turn's signature window and projection events, left in the log,
    /// survive a segment's maintenance byte for byte, and the window is not
    /// re-stamped.** Recovered, both are located, not held; a compaction of the
    /// segment holding them relocates each record by location and repoints the
    /// substrate at the copy. The bytes read there — and after a reload — are
    /// exactly the bytes written, and the window's fingerprint is the one it
    /// had before the move, so a resident gallery uploads nothing again.
    #[test]
    fn located_signatures_and_events_relocate_byte_for_byte() {
        use crate::provenance::{encode_wide_sigs, WideQSig};
        use crate::substrate::Located;

        let dir = tmp_dir("located_sig_reloc");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 5151,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
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
        let window: Vec<WideQSig> = (0..5u64)
            .map(|t| WideQSig {
                n_heads: 12,
                words: (0..24).map(|w| t * 1_000 + w).collect(),
            })
            .collect();
        let sigs = encode_wide_sigs(&window);
        let events: Vec<u8> = (0..1_300u32).map(|i| (i % 211) as u8).collect();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&decl).unwrap();
            // Superseded copies first — the dead weight that qualifies seg 1
            // for a compaction — then the live records it carries.
            sp.append_wide_q_sigs(sid, &[1, 2, 3], sp.mint_for(RecordType::WideQSig, sid))
                .unwrap();
            sp.append_projection_events(sid, b"[]", sp.mint_for(RecordType::ProjectionEvents, sid))
                .unwrap();
            sp.append_wide_q_sigs(sid, &sigs, sp.mint_for(RecordType::WideQSig, sid))
                .unwrap();
            sp.append_projection_events(
                sid,
                &events,
                sp.mint_for(RecordType::ProjectionEvents, sid),
            )
            .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
        }
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let stream = substrate.stream_of(sid).expect("the stream recovers");
        let old_sigs = stream
            .wide_q_sigs
            .as_ref()
            .and_then(Located::stored)
            .expect("recovered, the window is located, not held");
        let old_events = stream
            .projection_events
            .as_ref()
            .and_then(Located::stored)
            .expect("recovered, the events are located, not held");
        assert_eq!(old_sigs.segment, SegmentId(1));
        assert_eq!(old_events.segment, SegmentId(1));
        let before = substrate
            .wide_sig_window(sid)
            .expect("the window is sized from its header");
        assert_eq!(
            before.decode(),
            window,
            "the written window, before relocation"
        );

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("seg 1 carries superseded records, force-compact qualifies");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
        let result = sp.execute_maintenance(&plan).unwrap();
        let moved: Vec<(RecordType, RecordLoc, RecordLoc)> = result
            .located_updates
            .iter()
            .map(|((_, rt), from, to)| (*rt, *from, *to))
            .collect();
        let types: Vec<RecordType> = moved.iter().map(|(rt, _, _)| *rt).collect();
        assert_eq!(types.len(), 2, "the live window and events, and only them");
        assert!(types.contains(&RecordType::WideQSig));
        assert!(types.contains(&RecordType::ProjectionEvents));
        for (rt, from, to) in &moved {
            let old = if *rt == RecordType::WideQSig {
                old_sigs
            } else {
                old_events
            };
            assert_eq!(*from, old);
            assert_ne!(
                to.segment,
                SegmentId(1),
                "relocated off the compacted segment"
            );
        }
        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");

        let after = substrate
            .wide_sig_window(sid)
            .expect("the window survives the compaction");
        assert_eq!(
            after.fingerprint(),
            before.fingerprint(),
            "a moved window keeps its stamp: nothing re-uploads"
        );
        assert_eq!(
            after.decode(),
            window,
            "the written window, after relocation"
        );
        assert_eq!(
            &substrate.stream_projection_events(sid).unwrap().unwrap()[..],
            events.as_slice(),
            "the written events, after relocation"
        );
        drop(sp);

        let mut substrate = Substrate::new();
        let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert_eq!(
            &substrate.stream_wide_q_sigs(sid).unwrap().unwrap()[..],
            sigs.as_slice(),
            "the written window, after a reload"
        );
        assert_eq!(
            &substrate.stream_projection_events(sid).unwrap().unwrap()[..],
            events.as_slice(),
            "the written events, after a reload"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A relocated signature window and its events are carried again by the
    /// next compaction of the segment holding the copy.** The relocation's
    /// verbatim append must leave the location map naming the copy, as the
    /// substrate index does. Left naming nothing, the next op read the copy as
    /// not current, skipped it, stranded the record and kept its segment
    /// linked — and counted its live bytes as dead, so the same segment was
    /// picked and kept again: BlackWorld's log grew past 500 GB with a third of
    /// it dead (2026-11-23).
    #[test]
    fn a_relocated_window_is_carried_by_the_next_compaction_too() {
        use crate::provenance::{encode_wide_sigs, WideQSig};
        use crate::substrate::Located;

        let dir = tmp_dir("located_sig_reloc_twice");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 5152,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
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
        let window: Vec<WideQSig> = (0..5u64)
            .map(|t| WideQSig {
                n_heads: 12,
                words: (0..24).map(|w| t * 1_000 + w).collect(),
            })
            .collect();
        let sigs = encode_wide_sigs(&window);
        let events: Vec<u8> = (0..1_300u32).map(|i| (i % 211) as u8).collect();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.append_wide_q_sigs(sid, &[1, 2, 3], sp.mint_for(RecordType::WideQSig, sid))
                .unwrap();
            sp.append_wide_q_sigs(sid, &sigs, sp.mint_for(RecordType::WideQSig, sid))
                .unwrap();
            sp.append_projection_events(
                sid,
                &events,
                sp.mint_for(RecordType::ProjectionEvents, sid),
            )
            .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
        }
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();

        let compact = |sp: &mut SubstratePersistence, substrate: &mut Substrate| {
            let plan = sp
                .plan_maintenance(substrate, true)
                .unwrap()
                .expect("a segment carrying dead weight qualifies for a forced compaction");
            let MaintenanceOp::Compact(target) = plan.op() else {
                panic!("a compaction, not {:?}", plan.op());
            };
            let result = sp.execute_maintenance(&plan).unwrap();
            let carried: Vec<RecordType> = result
                .located_updates
                .iter()
                .map(|((_, rt), _, _)| *rt)
                .collect();
            let stranded = result.apply_to_substrate(&plan, substrate);
            assert!(
                stranded.is_empty(),
                "nothing the index names is left behind"
            );
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            (target, carried)
        };

        let (first, carried) = compact(&mut sp, &mut substrate);
        assert_eq!(first, SegmentId(1));
        assert_eq!(carried.len(), 2, "the live window and events");
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");
        let copy = substrate
            .stream_of(sid)
            .and_then(|s| s.wide_q_sigs.as_ref().and_then(Located::stored))
            .expect("the window is located at its copy")
            .segment;

        // Dead weight beside the copies, then seal the segment holding them.
        sp.write_debug_id(9, "first", sp.mint()).unwrap();
        sp.write_debug_id(9, "second", sp.mint()).unwrap();
        sp.commit().unwrap();
        sp.seal_active().unwrap();

        let (second, carried) = compact(&mut sp, &mut substrate);
        assert_eq!(second, copy, "the segment holding the copies");
        assert_eq!(carried.len(), 2, "the window and events carried again");
        assert!(
            !sealed_log(&dir, copy.0).exists(),
            "and that segment goes too"
        );
        assert_eq!(
            &substrate.stream_wide_q_sigs(sid).unwrap().unwrap()[..],
            sigs.as_slice()
        );
        assert_eq!(
            &substrate.stream_projection_events(sid).unwrap().unwrap()[..],
            events.as_slice()
        );
        drop(sp);

        let mut substrate = Substrate::new();
        let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert_eq!(
            &substrate.stream_wide_q_sigs(sid).unwrap().unwrap()[..],
            sigs.as_slice(),
            "the window, after two compactions and a reload"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The plan→execute supersession race: after a maintenance plan targets a
    /// stream's snapshot tail, the seal thread's writer appends a NEWER tail
    /// for the same stream. Execute must skip the planned (now stale) copy
    /// entirely — no append, no accounting movement, no index update — and a
    /// reload must still see the newer record as the tail.
    #[test]
    fn superseded_snapshot_is_not_relocated() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("snap_reloc_race");
        let timeline: u64 = 92;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let Stamped { value: loc_a, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 1, 0x11)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a, id);
        let Stamped { value: loc_b, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 2, 0x22)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b, id);
        sp.commit().unwrap();
        sp.seal_active().unwrap(); // seg 1 sealed (turn-1 dead, turn-2 the tail)

        // Phase 1 plans the relocation of the turn-2 tail out of seg 1.
        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));

        // RACE: before execute, the writer appends a newer turn-3 tail into
        // the active segment and installs it in the substrate index.
        let Stamped { value: loc_c, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 3, 0x33)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_c, id);
        sp.commit().unwrap();
        assert_eq!(
            loc_c.segment,
            SegmentId(2),
            "the newer tail lands in the active"
        );
        let dead_before = sp.accounting.dead_bytes();
        assert_eq!(
            dead_before,
            loc_a.record_size + loc_b.record_size,
            "turn-1 and turn-2 are both dead the moment turn-3 lands"
        );
        let offset_before = sp.write_offset();

        // Phase 2: the planned copy is stale — it must NOT be appended.
        let result = sp.execute_maintenance(&plan).unwrap();
        assert!(
            result.snapshot_updates.is_empty(),
            "a superseded snapshot produces no update"
        );
        assert_eq!(
            sp.write_offset(),
            offset_before,
            "no bytes were appended for the stale copy"
        );
        assert_eq!(
            sp.accounting.dead_bytes(),
            dead_before,
            "the live turn-3 record was not credited dead"
        );

        // Phase 3 + 4: the index keeps the newer tail; seg 1 still drops
        // (the stale copy is dead there).
        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(loc_c));
        sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
        assert!(
            !sealed_log(&dir, 1).exists(),
            "seg 1 dropped; the stale copy died with it"
        );
        drop(sp);

        // Reload: the walk replays only the surviving turn-3 record — the
        // tail is the newer state, not a rolled-back turn 2.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate
            .recurrent_snapshot_loc(sid)
            .expect("tail survives reload");
        assert_eq!(tail, loc_c);
        assert_eq!(
            sp.snapshot_locs.get(&sid.0),
            Some(&loc_c),
            "the persistence-side live-tail map is rebuilt by the walk"
        );
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&tail).unwrap()).unwrap();
        assert_eq!(
            decoded.turn_index, 3,
            "reload sees the NEWER snapshot as the tail"
        );
        assert_eq!(decoded.layers[0].state[0], 0x33);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A timeline tombstone landing between plan and execute empties the
    /// stream's live-tail entry, so the planned relocation is skipped and the
    /// tombstoned snapshot dies with its segment instead of being appended
    /// after the tombstone record (which a reload would resurrect as a tail).
    #[test]
    fn tombstoned_snapshot_is_not_relocated() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::TombstonePayload;

        let dir = tmp_dir("snap_reloc_tomb");
        let timeline: u64 = 93;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let Stamped { value: loc_a, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 1, 0x11)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a, id);
        let Stamped { value: loc_b, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 2, 0x22)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b, id);
        sp.commit().unwrap();
        sp.seal_active().unwrap();

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");

        // RACE: the timeline is tombstoned between plan and execute — the
        // durable record lands in the active segment, and both the substrate
        // index and the persistence-side map drop their entries.
        sp.write_tombstone(timeline, None).unwrap();
        substrate.apply_tombstone(&TombstonePayload {
            timeline_id: timeline,
            turn_index: None,
            turn_writer: 0,
            reason: None,
        });
        sp.commit().unwrap();
        let dead_before = sp.accounting.dead_bytes();
        let offset_before = sp.write_offset();

        let result = sp.execute_maintenance(&plan).unwrap();
        assert!(
            result.snapshot_updates.is_empty(),
            "a tombstoned snapshot produces no update"
        );
        assert_eq!(sp.write_offset(), offset_before);
        assert_eq!(sp.accounting.dead_bytes(), dead_before);
        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
        assert!(!sealed_log(&dir, 1).exists());
        drop(sp);

        // Reload: the tombstone record survives in the active segment and no
        // snapshot record outlived seg 1 — nothing resurrects a tail.
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert!(
            substrate.recurrent_snapshot_loc(sid).is_none(),
            "no tail resurrected after the tombstone"
        );
        assert!(!sp.snapshot_locs.contains_key(&sid.0));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A tail the index still names, which the relocation did not carry,
    /// keeps its segment.** The persistence-side live-tail map moved off the
    /// planned tail while the substrate index kept naming it: the relocation
    /// filters the tail out, the apply has nothing to repoint, and dropping
    /// the segment would leave the tail indexed but unreadable. The segment
    /// stays linked, and the tail still reads.
    #[test]
    fn a_tail_the_index_still_names_keeps_its_segment() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;
        use crate::persistence::stranded::{StrandedKind, StrandedRecord};

        let dir = tmp_dir("snap_reloc_stranded");
        let timeline: u64 = 94;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let Stamped { value: loc_a, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 1, 0x11)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a, id);
        let Stamped { value: loc_b, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 2, 0x22)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b, id);
        sp.commit().unwrap();
        sp.seal_active().unwrap();

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));

        // The two maps diverge: the persistence side no longer holds the
        // tail the substrate index names.
        sp.snapshot_locs.remove(&sid.0);
        let result = sp.execute_maintenance(&plan).unwrap();
        assert!(result.snapshot_updates.is_empty(), "the filter skipped it");

        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(loc_b));
        assert_eq!(
            stranded.records().copied().collect::<Vec<_>>(),
            vec![StrandedRecord {
                kind: StrandedKind::Snapshot,
                stream_id: sid.0,
                writer: 0,
                chunk_index: 0,
                segment: SegmentId(1),
                offset: loc_b.offset,
            }]
        );
        let retired = sp.finish_maintenance(&plan, &stranded).unwrap();
        assert!(retired.paths().is_empty(), "nothing is retired");
        retired.unlink();
        assert!(sealed_log(&dir, 1).exists(), "seg 1 stays linked");
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&loc_b).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2, "the indexed tail still reads");

        // While the index names the tail there, no op takes seg 1 again.
        let again = sp.plan_maintenance(&substrate, true).unwrap();
        assert!(
            again.is_none_or(|p| !p.op().targets().contains(&SegmentId(1))),
            "a stranded segment is left alone"
        );

        // A newer seal moves the index off it, and seg 1 is chosen again.
        let Stamped { value: loc_c, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 3, 0x33)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_c, id);
        sp.commit().unwrap();
        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("seg 1 is all dead weight now");
        assert!(
            plan.op().targets().contains(&SegmentId(1)),
            "{:?}",
            plan.op()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **An op that ends before its apply puts the live-tail map back.** Its
    /// relocation moved `snapshot_locs` onto the copy while the index stayed
    /// on the source; left so, the next op over the source's segment filtered
    /// the source out and unlinked it under the index. Unwound, the map names
    /// the source again, and the next op carries it and repoints the index.
    #[test]
    fn an_op_ended_before_its_apply_puts_the_tail_map_back() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("snap_reloc_unwind");
        let timeline: u64 = 95;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let Stamped { value: loc_a, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 1, 0x11)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a, id);
        let Stamped { value: loc_b, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 2, 0x22)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b, id);
        sp.commit().unwrap();
        sp.seal_active().unwrap();

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");
        let live_before = sp.segment_liveness(&substrate)[&SegmentId(1)];
        let dead_before = sp.accounting.dead_bytes();
        sp.watch_relocation();
        let mut run = sp.begin_maintenance(&plan).unwrap();
        while sp.relocate_batch(&plan, &mut run, u64::MAX).unwrap() {}
        while sp.relocate_tail_batch(&plan, &mut run, u64::MAX).unwrap() {}
        let copy = *sp.snapshot_locs.get(&sid.0).unwrap();
        assert_ne!(copy, loc_b, "the relocation moved the map onto its copy");
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(loc_b));

        // The op ends here, never applied.
        assert_eq!(sp.unwind_relocation(&plan, &run, &substrate, "test"), 1);
        sp.end_relocation_watch();
        assert_eq!(sp.snapshot_locs.get(&sid.0), Some(&loc_b));
        assert_eq!(
            sp.unwind_relocation(&plan, &run, &substrate, "test"),
            0,
            "unwinding twice moves nothing"
        );
        // The source reads as live as it did, and the copy's bytes are the
        // dead ones: one record's worth more than before the relocation.
        assert_eq!(sp.segment_liveness(&substrate)[&SegmentId(1)], live_before);
        assert_eq!(sp.accounting.dead_bytes(), dead_before + copy.record_size);
        assert_eq!(copy.record_size, loc_b.record_size, "a verbatim copy");

        // The next op carries the tail it names and repoints the index.
        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("seg 1 still carries dead weight");
        let result = sp.execute_maintenance(&plan).unwrap();
        let stranded = result.apply_to_substrate(&plan, &mut substrate);
        assert!(stranded.is_empty(), "the tail was carried, not stranded");
        sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
        let tail = substrate.recurrent_snapshot_loc(sid).unwrap();
        assert_ne!(tail.segment, SegmentId(1));
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&tail).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **An op whose apply ran keeps its copies on unwind.** The index already
    /// names the copy, so putting the map back on the source would split them
    /// the other way.
    #[test]
    fn an_applied_op_keeps_its_copies_on_unwind() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("snap_reloc_unwind_applied");
        let timeline: u64 = 96;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let Stamped { value: loc_a, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 1, 0x11)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a, id);
        let Stamped { value: loc_b, id } = sp
            .write_snapshot(sid, &[&snapshot_payload(timeline, 2, 0x22)])
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b, id);
        sp.commit().unwrap();
        sp.seal_active().unwrap();

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");
        sp.watch_relocation();
        let mut run = sp.begin_maintenance(&plan).unwrap();
        while sp.relocate_batch(&plan, &mut run, u64::MAX).unwrap() {}
        while sp.relocate_tail_batch(&plan, &mut run, u64::MAX).unwrap() {}
        let copy = *sp.snapshot_locs.get(&sid.0).unwrap();
        // The apply's edit, made by hand so the run keeps its updates: the
        // index moves onto the copy under the record's own identity.
        let id = substrate.snapshot_id(sid);
        substrate.apply_snapshot_loc(sid, copy, id);
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(copy));

        // Something after the apply ends the op: the unwind leaves it be.
        assert_eq!(sp.unwind_relocation(&plan, &run, &substrate, "test"), 0);
        sp.end_relocation_watch();
        assert_eq!(sp.snapshot_locs.get(&sid.0), Some(&copy));
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&copy).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
