//! Tests which resident records a sweep re-emits, and when.
//!
//! Couplings, turn and section tombstones, and the shared blobs of a plan must
//! each be carried when their segment is dropped. Retired turns must stay dead
//! through a sweep. Orphans must be neither relocated nor re-emitted, or their
//! bytes would be kept alive forever. A record is re-emitted only when its
//! current copy sits in a target segment.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::streams::{StreamDecl, StreamId};

    use crate::persistence::SubstratePersistence;

    use crate::projection::TimelineId;

    use crate::substrate::Substrate;

    use super::super::tests_support::tests::{
        chunk_payload, sealed_log, store_with_section_tombstones, tmp_dir, turn_decl,
    };

    /// A tool round-trip's coupling is in the resident set, so segment
    /// maintenance re-emits it rather than dropping it with its segment.
    ///
    /// The maintenance twin of
    /// [`super::super::tests::a_turn_coupling_survives_compaction`] — the two
    /// carry-forward paths are separate lists and a record class has to be in
    /// both.
    #[test]
    fn a_turn_coupling_is_in_the_resident_set() {
        let mut substrate = Substrate::new();
        substrate.register_timeline(
            TimelineId::from_raw(9).unwrap(),
            crate::projection::LayerId::from_raw(1).unwrap(),
            crate::projection::GroupId::from_raw(1).unwrap(),
        );
        substrate.couple_turn(TimelineId::from_raw(9).unwrap(), 4, RecordIdentity::NONE);

        let resident = gather_resident_set(&substrate, &MarkerIds::new(), 0);
        let coupling = resident
            .iter()
            .find(|r| r.rt == RecordType::TurnCoupling)
            .expect("the coupling must be re-emitted by segment maintenance");
        let decoded = TurnCouplingPayload::decode(&coupling.payload).unwrap();
        assert_eq!((decoded.timeline_id, decoded.from_turn), (9, 4));
    }

    /// **A plan shares a turn's metadata blobs; it does not copy them.**
    ///
    /// Index pages run to a megabyte a turn and wide-Q signatures to hundreds of
    /// KB, and the plan holds the whole resident set for the length of the op.
    /// Copied, that was a second image of every live turn's metadata in RAM for
    /// minutes at a time — and the copy alone took 13 s at plan time on the live
    /// daemon (`resident_ms=13096`).
    #[test]
    fn the_resident_set_shares_the_substrates_blobs() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 4242,
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
        let mut substrate = Substrate::new();
        let none = RecordIdentity::NONE;
        substrate.apply_stream_decl(sid, decl, none);
        substrate.apply_projection_events(sid, vec![1, 2, 3], none);
        substrate.apply_wide_q_sigs(sid, vec![4, 5, 6, 7], none);
        substrate.apply_index_page(sid, vec![8, 9], none);

        let resident = gather_resident_set(&substrate, &MarkerIds::new(), 0);
        let entry = substrate.stream_of(sid).unwrap();
        let payload_of = |rt: RecordType| {
            &resident
                .iter()
                .find(|r| r.rt == rt && r.stream_id == sid.0)
                .unwrap_or_else(|| panic!("{rt:?} is missing from the resident set"))
                .payload
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

    /// **A turn tombstone is re-emitted by a sweep, so retirement is durable.**
    ///
    /// The failure this closes was self-erasing and therefore invisible:
    /// `retention` marked a turn dead, the mark lived in one segment, and when
    /// maintenance relocated that segment's live records the mark was not among
    /// them. The segment was dropped, the tombstone went with it, and the next
    /// reload rebuilt `tombstoned_turns` from a log that no longer mentioned the
    /// turn — so it came back to life and its KV was pinned for good.
    ///
    /// Live evidence: 37,705 turn streams, 70 surviving tombstones, a 132 GB
    /// floor that never moved.
    #[test]
    fn a_retired_turns_tombstone_survives_a_maintenance_sweep() {
        let mut substrate = Substrate::new();
        let tl = TimelineId::from_raw(9001).unwrap();
        substrate.tombstone_turn(tl, 7);
        substrate.tombstone_turn(tl, 8);

        let residents = gather_resident_set(&substrate, &MarkerIds::new(), 0);
        let turn_marks: Vec<u32> = residents
            .iter()
            .filter(|r| r.rt == RecordType::Tombstone)
            .filter_map(|r| TombstonePayload::decode(&r.payload).ok())
            .filter(|p| p.timeline_id == 9001)
            .filter_map(|p| p.turn_index)
            .collect();
        assert!(
            turn_marks.contains(&7) && turn_marks.contains(&8),
            "a sweep dropped the turn tombstones: {turn_marks:?}"
        );
    }

    /// A wholesale tombstone subsumes the turn-scoped ones it covers, so a
    /// retired-then-retired conversation leaves nothing behind to re-emit.
    #[test]
    fn a_dead_timeline_leaves_no_turn_marks_to_carry_forward() {
        let turn_marks = |substrate: &Substrate, raw: u64| {
            gather_resident_set(substrate, &MarkerIds::new(), 0)
                .iter()
                .filter(|r| r.rt == RecordType::Tombstone)
                .filter_map(|r| TombstonePayload::decode(&r.payload).ok())
                .filter(|p| p.timeline_id == raw && p.turn_index.is_some())
                .count()
        };

        let mut substrate = Substrate::new();
        let tl = TimelineId::from_raw(5150).unwrap();
        substrate.tombstone_turn(tl, 1);
        substrate.tombstone_turn(tl, 2);
        substrate.tombstone_timeline(tl);
        assert_eq!(
            turn_marks(&substrate, 5150),
            0,
            "a dead timeline re-emitted per-turn marks"
        );

        // **And the other order lands in the same place.** A character deleted
        // mid-decode is tombstoned while the tick still running goes on to
        // retire its tail, so the turn mark arrives *after* the timeline's — and
        // a replay of the log can present the two in either order. If only one
        // direction subsumed, the in-RAM set would depend on arrival order and
        // a sweep would carry marks a reload would not.
        let mut backwards = Substrate::new();
        let tl = TimelineId::from_raw(5151).unwrap();
        backwards.tombstone_timeline(tl);
        backwards.tombstone_turn(tl, 1);
        backwards.tombstone_turn(tl, 2);
        assert_eq!(
            turn_marks(&backwards, 5151),
            0,
            "a turn retired after its timeline died left a mark behind"
        );
        assert!(
            !backwards.is_turn_tombstoned(tl, 1),
            "the in-RAM set disagrees with what a replay of the same log rebuilds"
        );
    }

    /// An orphan stream (KV chunks on disk, NO `StreamDecl`) is neither relocated
    /// nor re-emitted by the incremental maintenance path — it is left to be
    /// reclaimed when its source segment drops. Regression for the churn bug:
    /// `segment_liveness` excludes orphan chunks from live weight, so if
    /// `gather_relocations` still relocated them, a mixed segment would be
    /// compacted forever (dead-ratio never improving) and the orphan KV would
    /// migrate segment-to-segment, never reclaimed.
    #[test]
    fn orphan_stream_is_not_relocated_or_re_emitted() {
        let dir = tmp_dir("orphan_reloc");
        let sid = StreamId(4242); // no decl declared → orphan
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 holds the orphan chunk
        }
        // Re-walk so the substrate sees the chunk with no owning decl.
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert!(
            substrate
                .stream_of(sid)
                .is_some_and(|s| s.decl.is_none() && !s.chunks.is_empty()),
            "the stream is an orphan: chunk present, no decl",
        );
        // gather_relocations must NOT relocate the orphan's chunk.
        let (chunks, _tokens, _snapshots, _branch, _singletons) = sp
            .live_indexes()
            .gather_relocations(&substrate, &[SegmentId(1)]);
        assert!(
            chunks.is_empty(),
            "orphan chunks must not be relocated (would perpetuate the bloat + churn)",
        );
        // gather_resident_set must not re-emit any record for the orphan stream.
        let residents = gather_resident_set(&substrate, &MarkerIds::new(), 0);
        assert!(
            residents.iter().all(|r| r.stream_id != sid.0),
            "orphan metadata must not be re-emitted",
        );
        // And it contributes zero live weight, so its segment is reclaimable.
        let live: u64 = sp.segment_liveness(&substrate).values().sum();
        assert_eq!(live, 0, "an orphan chunk is not live weight");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **An op re-emits the resident records it is about to drop, and no
    /// others.** Re-emitting the whole set on every op was a loop on the live
    /// daemon: 2.2 GB written forward per op, the previous op's copies left dead
    /// in the segments just sealed, and those segments compacted a minute later
    /// to do it again.
    ///
    /// A record goes exactly when its current copy is in a target, and a
    /// record the map has no copy of always goes — a section tombstone by the
    /// same rule as every other resident type.
    #[test]
    fn an_op_re_emits_only_the_resident_records_it_drops() {
        use crate::persistence::streams::TurnDecl;
        let dir = tmp_dir("resident_to_carry");
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let none = RecordIdentity::NONE;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 4343,
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
        substrate.apply_stream_decl(sid, decl, none);
        substrate.apply_wide_q_sigs(sid, vec![1, 2, 3], none);
        substrate.apply_projection_events(sid, vec![4], none);
        let section = StreamId(0x5ec7);
        substrate.tombstone_section(section);

        let at = |segment: u64| RecordLoc {
            segment: SegmentId(segment),
            offset: 0,
            payload_len: 0,
            record_size: 4096,
        };
        sp.metadata_locs.clear();
        sp.resident_locs.clear();
        sp.metadata_locs
            .offer((RecordType::StreamDecl, sid.0, 0), Some(at(5)), none);
        sp.metadata_locs
            .offer((RecordType::WideQSig, sid.0, 0), Some(at(7)), none);
        // `ProjectionEvents` is left without an entry.

        let carry = |sp: &SubstratePersistence, targets: &[u64]| {
            let targets: Vec<SegmentId> = targets.iter().map(|&t| SegmentId(t)).collect();
            let mut kinds: Vec<(RecordType, u64)> = sp
                .live_indexes()
                .resident_to_carry(&substrate, &targets)
                .iter()
                .map(|r| (r.rt, r.stream_id))
                .collect();
            kinds.sort_by_key(|&(rt, id)| (rt as u8, id));
            kinds
        };
        let sorted = |mut v: Vec<(RecordType, u64)>| {
            v.sort_by_key(|&(rt, id)| (rt as u8, id));
            v
        };

        // The section tombstone has no copy on record yet: every op makes one.
        assert_eq!(
            carry(&sp, &[9]),
            sorted(vec![
                (RecordType::ProjectionEvents, sid.0),
                (RecordType::SectionTombstone, section.0),
            ])
        );

        // Its copy recorded in segment 12, keyed as an append keys it.
        let tombstone = sp
            .live_indexes()
            .resident_set(&substrate)
            .into_iter()
            .find(|r| r.rt == RecordType::SectionTombstone)
            .expect("the ended section is in the resident set");
        let key = resident_key(
            tombstone.rt,
            tombstone.stream_id,
            &tombstone.payload,
            &tombstone.id,
        )
        .expect("a section tombstone has a resident key");
        sp.resident_locs.offer(key, Some(at(12)), tombstone.id);

        // The decl's segment: the decl, plus the unlocated events.
        assert_eq!(
            carry(&sp, &[5]),
            sorted(vec![
                (RecordType::StreamDecl, sid.0),
                (RecordType::ProjectionEvents, sid.0),
            ])
        );
        // The signature's segment: the signature, plus the unlocated events.
        assert_eq!(
            carry(&sp, &[7]),
            sorted(vec![
                (RecordType::WideQSig, sid.0),
                (RecordType::ProjectionEvents, sid.0),
            ])
        );
        // A segment holding none of them: only the unlocated.
        assert_eq!(
            carry(&sp, &[9]),
            vec![(RecordType::ProjectionEvents, sid.0)]
        );
        // The tombstone's segment: the tombstone too.
        assert_eq!(
            carry(&sp, &[12]),
            sorted(vec![
                (RecordType::ProjectionEvents, sid.0),
                (RecordType::SectionTombstone, section.0),
            ])
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A section tombstone is carried only by the op that drops its copy,
    /// and counts as live where it lies.** It was the one resident type with
    /// no location: every op on a segment newer than the last full re-emission
    /// re-emitted all of them, and the segment holding them read as dead
    /// weight. On the live daemon that was 12,000 tombstones per op, holding
    /// the persistence lock for 18 s.
    #[test]
    fn a_section_tombstone_is_carried_only_off_its_own_segment() {
        let dir = store_with_section_tombstones("section_carry", 3);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.ended_unstamped_sections().count(), 3);

            let tombstones =
                |sp: &SubstratePersistence, substrate: &Substrate, targets: &[SegmentId]| {
                    sp.live_indexes()
                        .resident_to_carry(substrate, targets)
                        .iter()
                        .filter(|r| r.rt == RecordType::SectionTombstone)
                        .count()
                };
            assert_eq!(
                tombstones(&sp, &substrate, &[SegmentId(2)]),
                0,
                "not on segment 2"
            );
            assert_eq!(
                tombstones(&sp, &substrate, &[SegmentId(1)]),
                3,
                "all three on segment 1"
            );

            let liveness = sp.segment_liveness(&substrate);
            assert!(
                liveness.get(&SegmentId(1)).copied().unwrap_or(0) > 0,
                "a segment of nothing but live section tombstones is live"
            );

            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();
            assert!(
                !sealed_log(&dir, 1).exists(),
                "segment 1 was compacted away"
            );
            assert_eq!(
                tombstones(&sp, &substrate, &[SegmentId(1)]),
                0,
                "the copies now live in the active segment"
            );
        }
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let mut ended: Vec<u64> = substrate.ended_unstamped_sections().map(|s| s.0).collect();
            ended.sort_unstable();
            assert_eq!(
                ended,
                vec![0x5000, 0x5001, 0x5002],
                "every section stays ended across the compaction and a reload"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tool round-trip's couplings survive the compaction of the segment that
    /// holds them.
    ///
    /// The incremental path is where this actually bit: a busy store retires
    /// segments continuously, so every coupling written before the most recent
    /// retirement was gone. Nothing detected it — `accounting` excludes the type
    /// from supersession, and a timeline with no couplings reads as a legitimate
    /// state (one that never called a tool), so the store looked intact while
    /// every exchange had come apart into independent turns.
    #[test]
    fn couplings_survive_a_segment_compaction() {
        let dir = tmp_dir("couple");
        let tid = 424u64;
        // A three-turn repo_map chain; the first two couple.
        let decls = [turn_decl(tid, 0), turn_decl(tid, 1), turn_decl(tid, 2)];
        let sid0 = decls[0].stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            for d in &decls {
                sp.declare_stream(d).unwrap();
            }
            // Both couplings land in seg 1, alongside a chunk that seg 2 will
            // supersede — so seg 1 carries dead weight and qualifies to compact.
            for from_turn in [0u32, 1] {
                sp.write_turn_coupling(tid, from_turn, sp.mint()).unwrap();
            }
            sp.write_chunk(sid0, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid0, 0, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(tid).unwrap();
            assert_eq!(
                substrate.couplings_of(tl),
                [0u32, 1].into_iter().collect(),
                "replay must install both couplings before the pass",
            );
            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
        }
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                substrate.couplings_of(TimelineId::from_raw(tid).unwrap()),
                [0u32, 1].into_iter().collect(),
                "both couplings survived the compaction of the segment holding them",
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A turn-scoped tombstone survives, so the turn it killed stays dead.
    ///
    /// Losing one is worse than forgetting a deletion. The `is_tomb` gate in
    /// `gather_resident_set` reads `tombstoned_timelines`, so a turn-scoped
    /// tombstone leaves its stream fully live and the turn's decl, tokens and KV
    /// are all carried forward. Drop the marker alone and the turn comes back —
    /// with the corruption the `drop_turn` policy condemned it for.
    #[test]
    fn turn_scoped_tombstone_survives_a_segment_compaction() {
        use crate::substrate::TurnMark;
        let dir = tmp_dir("turntomb");
        let tid = 525u64;
        let live = turn_decl(tid, 0);
        let doomed = turn_decl(tid, 1);
        let live_sid = live.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&live).unwrap();
            sp.declare_stream(&doomed).unwrap();
            let mark = TurnMark {
                timeline: TimelineId::from_raw(tid).unwrap(),
                turn_index: 1,
                writer: sp.node(),
            };
            sp.write_turn_tombstone(mark, Some("corrupt reload (turn 1)"))
                .unwrap();
            sp.write_chunk(live_sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(live_sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(tid).unwrap();
            assert!(substrate.is_turn_tombstoned(tl, 1));
            assert!(
                !substrate.is_tombstoned(tl),
                "turn-scoped: the rest of the timeline stays live",
            );
            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();
            assert!(!sealed_log(&dir, 1).exists());
        }
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(tid).unwrap();
            assert!(
                substrate.is_turn_tombstoned(tl, 1),
                "the condemned turn must not come back",
            );
            assert!(!substrate.is_tombstoned(tl));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
