//! Tests that a batched op keeps the newest write made between its batches.
//!
//! Batches release the persistence lock between steps, so a chunk, a token set,
//! a retired section or a metadata record written in the gap must not be
//! overwritten by its planned copy. Other tests check that an op cannot be
//! completed early, that one op is in flight at a time, that a long run is
//! relocated whole, and that a batch is bounded by record count.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::record::ChunkPayload;

    use crate::persistence::streams::{StreamDecl, StreamId};

    use crate::persistence::SubstratePersistence;

    use crate::projection::TimelineId;

    use crate::substrate::Substrate;

    use super::super::tests_support::tests::{
        chunk_payload, identities_on_disk, sealed_log, store_with_section_tombstones, tmp_dir,
        turn_decl,
    };

    /// **A relocation run in batches holds the lock one batch at a time, and a
    /// chunk re-appended between batches keeps its newer copy.** Four chunks
    /// stand in a sealed segment; the op moves them one per batch, and after the
    /// first batch chunk 2 is written again — a partial tail sealed final. The
    /// planned copy of chunk 2 is then superseded: relocating it would append
    /// it after the new one, and the reload's last-writer-wins walk would bring
    /// the old bytes back. It must be skipped, and everything else moved.
    #[test]
    fn a_chunk_rewritten_between_batches_keeps_its_newer_copy() {
        let dir = tmp_dir("batched");
        let decl = turn_decl(404, 0);
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            for i in 0..4u64 {
                sp.write_chunk(sid, i, 32, 4, None, &chunk_payload(10 + i as u32))
                    .unwrap();
            }
            sp.write_chunk(sid, 4, 32, 4, None, &chunk_payload(1))
                .unwrap(); // superseded below: the segment carries dead weight
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 4, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let plan = sp.plan_maintenance(&substrate, true).unwrap().unwrap();
            assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
            sp.watch_relocation();
            let mut run = sp.begin_maintenance(&plan).unwrap();
            let mut batches = 0;
            loop {
                let more = sp.relocate_batch(&plan, &mut run, 1).unwrap();
                batches += 1;
                if batches == 1 {
                    let rewritten = sp
                        .write_chunk(sid, 2, 32, 4, None, &chunk_payload(99))
                        .unwrap();
                    sp.commit().unwrap();
                    // The writer registers the rewrite in the index, as the
                    // cold write does; a payload of the same shape has the
                    // same length.
                    let key = StreamKey::of(sid, &rewritten.identity);
                    let held = substrate
                        .stream_at(key)
                        .and_then(|s| s.chunks.get(&2).copied())
                        .unwrap();
                    let at = ChunkLoc {
                        segment: rewritten.segment,
                        offset: rewritten.offset,
                        record_size: rewritten.size,
                        ..held
                    };
                    substrate.apply_chunk_loc(sid, 2, at, rewritten.identity);
                }
                if !more {
                    break;
                }
            }
            assert_eq!(
                batches,
                plan.resident.len() + 4,
                "one resident record, then one chunk, per batch at a one-byte budget"
            );
            let result = sp.complete_maintenance(&plan, run).unwrap();
            let moved: Vec<u64> = result.chunk_updates.iter().map(|u| u.1).collect();
            assert_eq!(
                moved,
                vec![0, 1, 3],
                "chunk 2's planned copy was superseded"
            );
            let stranded = result.apply_to_substrate(&plan, &mut substrate);
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let mut read = |i: u64| sp.read_chunk(&substrate, sid, i).unwrap();
            assert_eq!(read(0), chunk_payload(10));
            assert_eq!(read(1), chunk_payload(11));
            assert_eq!(
                read(2),
                chunk_payload(99),
                "the newer copy survives the reload"
            );
            assert_eq!(read(3), chunk_payload(13));
            assert_eq!(read(4), chunk_payload(2));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **Writes during an op's batches are stepped around, for tokens and for a
    /// retired section as for chunks.** A section retired after the plan — the
    /// system-prompt rebuild route does this — must carry none of its old
    /// generation past its tombstone, or a reload revives it under the new
    /// one; a turn whose tokens were rewritten keeps the newer copy. The op is
    /// in flight from the plan until its sources are unlinked.
    #[test]
    fn a_retired_section_and_rewritten_tokens_are_not_relocated_past_them() {
        use crate::persistence::content_hash::ContentHash;
        use crate::persistence::streams::{ContentAddress, SectionDecl};

        let dir = tmp_dir("retired_mid_op");
        let section = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 0x91, hi: 0x92 },
                section_hash: ContentHash { lo: 0x93, hi: 0x94 },
            },
            debug_name: "framing".to_string(),
        });
        let ssid = section.stream_id();
        let turn = turn_decl(505, 0);
        let tsid = turn.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&section).unwrap();
            sp.declare_stream(&turn).unwrap();
            for i in 0..2u64 {
                sp.write_chunk(ssid, i, 32, 4, None, &chunk_payload(20 + i as u32))
                    .unwrap();
            }
            sp.append_tokens(ssid, b"section-tokens").unwrap();
            sp.write_chunk(tsid, 0, 32, 4, None, &chunk_payload(30))
                .unwrap();
            sp.append_tokens(tsid, b"turn-old").unwrap();
            sp.write_chunk(tsid, 1, 32, 4, None, &chunk_payload(1))
                .unwrap(); // superseded below: the segment carries dead weight
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(tsid, 1, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let plan = sp.plan_maintenance(&substrate, true).unwrap().unwrap();
            assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
            sp.watch_relocation();
            assert!(sp.relocation_in_flight());
            let mut run = sp.begin_maintenance(&plan).unwrap();
            let generation = substrate.section_generation(ssid);
            assert!(generation.is_some(), "the section is stamped");
            sp.write_section_tombstone(ssid.0, generation, None)
                .unwrap();
            let at = sp.append_tokens(tsid, b"turn-new").unwrap();
            substrate.apply_tokens_loc(tsid, at.loc(b"turn-new".len() as u64), at.identity);
            sp.commit().unwrap();
            while sp.relocate_batch(&plan, &mut run, 1).unwrap() {}
            let result = sp.complete_maintenance(&plan, run).unwrap();
            let moved: Vec<(StreamId, u64)> = result
                .chunk_updates
                .iter()
                .map(|u| (u.0.stream_id, u.1))
                .collect();
            assert_eq!(moved, vec![(tsid, 0)], "no chunk of the retired section");
            assert!(
                result.token_updates.is_empty(),
                "neither the retired section's tokens nor the turn's superseded ones"
            );
            assert!(sp.relocation_in_flight(), "in flight until the unlink");
            let stranded = result.apply_to_substrate(&plan, &mut substrate);
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            assert!(!sp.relocation_in_flight());
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                sp.read_tokens(&substrate, tsid).unwrap().as_deref(),
                Some(&b"turn-new"[..]),
                "the newer tokens survive the reload"
            );
            assert_eq!(
                sp.read_chunk(&substrate, tsid, 0).unwrap(),
                chunk_payload(30)
            );
            assert_eq!(
                sp.read_chunk(&substrate, tsid, 1).unwrap(),
                chunk_payload(2)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **The resident set is re-emitted a batch at a time, around the writes
    /// made beside it.** Re-emitting it under one hold of the lock stalled
    /// every finishing turn's token write for 44–52 s on the live daemon, so
    /// it is batched like the chunks — and the lock is then released between
    /// batches, so a metadata record can be written again before its planned
    /// copy is appended. The newer write is appended again after the planned
    /// copy: a last-writer-wins record (the debug id) keeps the newer value —
    /// which, left where it was, would sit before the timeline's re-emitted
    /// declaration and be dropped on replay — and a merged one (the label)
    /// keeps the planned fields the newer partial record does not carry, and
    /// the newer title.
    #[test]
    fn the_resident_set_is_re_emitted_in_batches_around_newer_writes() {
        let dir = tmp_dir("reemit_batched");
        let decl = turn_decl(606, 0);
        let sid = decl.stream_id();
        let label = |conv_id: &str, title: &str| {
            encode_label_payload(606, conv_id, title, &BTreeMap::new())
        };
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.append_record(
                RecordType::Label,
                0,
                0,
                0,
                0,
                0,
                &label("conv-606", "original"),
            )
            .unwrap();
            sp.write_debug_id(606, "first", sp.mint()).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(10))
                .unwrap();
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(1))
                .unwrap(); // superseded below: the segment carries dead weight
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let plan = sp.plan_maintenance(&substrate, true).unwrap().unwrap();
            assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
            assert!(
                plan.resident.len() >= 3,
                "the decl, the label and the debug id are planned"
            );
            sp.watch_relocation();
            let mut run = sp.begin_maintenance(&plan).unwrap();
            sp.write_debug_id(606, "second", sp.mint()).unwrap();
            sp.append_record(RecordType::Label, 0, 0, 0, 0, 0, &label("", "renamed"))
                .unwrap();
            sp.commit().unwrap();

            let mut resident_batches = 0;
            while run.resident_cursor < plan.resident.len() {
                assert!(sp.relocate_batch(&plan, &mut run, 1).unwrap());
                resident_batches += 1;
            }
            assert!(
                resident_batches >= 2,
                "the set spans several holds at a one-byte budget, not one"
            );
            while sp.relocate_batch(&plan, &mut run, 1).unwrap() {}
            let result = sp.complete_maintenance(&plan, run).unwrap();
            let stranded = result.apply_to_substrate(&plan, &mut substrate);
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(606).unwrap();
            assert_eq!(
                substrate.debug_id_of(tl),
                Some("second"),
                "the newer debug id survives the reload"
            );
            assert_eq!(substrate.label_of(tl), Some("renamed"));
            assert_eq!(
                substrate.conv_id_of(tl),
                Some("conv-606"),
                "the planned label's conversation id survives the merge"
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 0).unwrap(),
                chunk_payload(10)
            );
            assert_eq!(sp.read_chunk(&substrate, sid, 1).unwrap(), chunk_payload(2));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **Maintenance moves records; it never originates one.** A segment
    /// compacted with a resident set re-emitted around a newer write leaves
    /// every record carrying either the identity it had before the run or the
    /// identity the newer write was minted with — the relocated chunk, the
    /// re-emitted declaration and the re-appended newer debug id alike — and
    /// only the index records carry none.
    #[test]
    fn maintenance_keeps_every_moved_records_identity() {
        let dir = tmp_dir("reemit_identity");
        let decl = turn_decl(607, 0);
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.write_debug_id(607, "first", sp.mint()).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(10))
                .unwrap();
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        let before = identities_on_disk(&dir);
        let mut known: Vec<RecordIdentity> = before
            .iter()
            .map(|r| r.3)
            .filter(|id| !id.is_none())
            .collect();
        let chunk0 = before
            .iter()
            .find(|r| r.0 == RecordType::Chunk && r.1 == sid.0 && r.2 == 0)
            .unwrap()
            .3;
        let decl_id = before
            .iter()
            .find(|r| r.0 == RecordType::StreamDecl && r.1 == sid.0)
            .unwrap()
            .3;
        let node;
        let second_oseq;
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            node = sp.node();
            let plan = sp.plan_maintenance(&substrate, true).unwrap().unwrap();
            assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));
            sp.watch_relocation();
            let mut run = sp.begin_maintenance(&plan).unwrap();
            second_oseq = sp.origination().next_oseq();
            sp.write_debug_id(607, "second", sp.mint()).unwrap();
            sp.commit().unwrap();
            while sp.relocate_batch(&plan, &mut run, 1).unwrap() {}
            let result = sp.complete_maintenance(&plan, run).unwrap();
            let stranded = result.apply_to_substrate(&plan, &mut substrate);
            sp.finish_maintenance(&plan, &stranded).unwrap().unlink();
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
            assert_eq!(
                sp.origination().next_oseq(),
                second_oseq + 1,
                "the run minted nothing beyond the one write made beside it"
            );
        }
        let after = identities_on_disk(&dir);
        let second = after
            .iter()
            .rev()
            .find(|r| r.0 == RecordType::DebugId)
            .unwrap()
            .3;
        assert_eq!((second.origin, second.oseq), (node, second_oseq));
        known.push(second);
        for (rt, stream, chunk, id) in &after {
            if *rt == RecordType::HeaderIndex {
                assert!(id.is_none(), "an index record carries no identity");
                continue;
            }
            assert!(
                known.contains(id),
                "{rt:?} stream {stream} chunk {chunk} carries {id:?}, minted by the run"
            );
        }
        let last = |rt: RecordType, idx: u64| {
            after
                .iter()
                .rev()
                .find(|r| r.0 == rt && r.1 == sid.0 && r.2 == idx)
                .unwrap()
                .3
        };
        assert_eq!(last(RecordType::Chunk, 0), chunk0, "the relocated chunk");
        assert_eq!(
            last(RecordType::StreamDecl, 0),
            decl_id,
            "the re-emitted decl"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A batched op cannot be completed early: the commit it ends with
    /// licenses unlinking the sources.
    #[test]
    #[should_panic(expected = "a maintenance op was completed with batches still to run")]
    fn completing_an_op_with_batches_left_panics() {
        let dir = tempfile::tempdir().unwrap();
        let decl = turn_decl(707, 0);
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let plan = sp.plan_maintenance(&substrate, true).unwrap().unwrap();
        let run = sp.begin_maintenance(&plan).unwrap();
        sp.complete_maintenance(&plan, run).ok();
    }

    /// **One op at a time.** Planning a second while one is in flight panics:
    /// it would relocate out of segments the first is about to unlink.
    #[test]
    #[should_panic(expected = "a second maintenance op was planned while one is relocating")]
    fn a_second_watch_while_one_is_in_flight_panics() {
        // A `TempDir`, so the unwind removes it.
        let dir = tempfile::tempdir().unwrap();
        let mut sp = SubstratePersistence::open_in(dir.path()).unwrap();
        sp.watch_relocation();
        sp.watch_relocation();
    }

    /// **A live run longer than one read stripe is relocated whole.** Seventy
    /// one-MiB chunks stand back to back in a sealed segment — one run past
    /// [`MAX_STRIPE_BYTES`] — beside a superseded record that makes it worth
    /// compacting. Every chunk reads back byte for byte after the relocation
    /// and after a reload.
    #[test]
    fn a_live_run_past_one_stripe_is_relocated_whole() {
        let dir = tmp_dir("stripes");
        let decl = turn_decl(303, 0);
        let sid = decl.stream_id();
        let big = |index: u64| ChunkPayload {
            kv_bytes: (0..1024 * 1024u64)
                .map(|i| ((i + index * 13) % 251) as u8)
                .collect(),
            ..chunk_payload(index as u32)
        };
        const LIVE: u64 = 70;
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            for i in 0..LIVE {
                sp.write_chunk(sid, i, 32, 4, None, &big(i)).unwrap();
            }
            sp.write_chunk(sid, LIVE, 32, 4, None, &chunk_payload(1))
                .unwrap(); // superseded below
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, LIVE, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
            for i in 0..LIVE {
                assert_eq!(
                    sp.read_chunk(&substrate, sid, i).unwrap(),
                    big(i),
                    "chunk {i}"
                );
            }
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            for i in 0..LIVE {
                assert_eq!(
                    sp.read_chunk(&substrate, sid, i).unwrap(),
                    big(i),
                    "chunk {i}"
                );
            }
            assert_eq!(
                sp.read_chunk(&substrate, sid, LIVE).unwrap(),
                chunk_payload(2)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A batch re-emits at most [`RELOCATION_BATCH_RECORDS`] resident
    /// records, however small they are.** The byte budget alone let one batch
    /// take an entire set of sector-sized records under a single hold.
    #[test]
    fn a_resident_batch_is_bounded_by_record_count() {
        let n = 2 * RELOCATION_BATCH_RECORDS as u64 + 5;
        let dir = store_with_section_tombstones("section_batches", n);
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let plan = MaintenancePlan {
            op: MaintenanceOp::Compact(SegmentId(1)),
            resident: sp
                .live_indexes()
                .resident_to_carry(&substrate, &[SegmentId(1)]),
            chunk_relocs: Vec::new(),
            token_relocs: Vec::new(),
            index_page_relocs: Vec::new(),
            snapshot_relocs: Vec::new(),
            branch_checkpoint_relocs: Vec::new(),
            singleton_relocs: Vec::new(),
            npc_relocs: Vec::new(),
            vfs_relocs: Vec::new(),
            blob_relocs: Vec::new(),
        };
        assert_eq!(plan.resident.len() as u64, n);
        let mut run = sp.begin_maintenance(&plan).unwrap();
        let mut cursors = Vec::new();
        while run.resident_cursor < plan.resident.len() {
            sp.relocate_batch(&plan, &mut run, u64::MAX).unwrap();
            cursors.push(run.resident_cursor);
        }
        assert_eq!(
            cursors,
            vec![
                RELOCATION_BATCH_RECORDS,
                2 * RELOCATION_BATCH_RECORDS,
                2 * RELOCATION_BATCH_RECORDS + 5
            ],
            "three holds for 2 × the cap + 5, at an unlimited byte budget"
        );
        drop(sp);
        std::fs::remove_dir_all(&dir).ok();
    }
}
