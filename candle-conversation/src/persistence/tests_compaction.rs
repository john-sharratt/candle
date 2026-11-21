//! Tests that compaction carries every class of live record through the rewrite, and that the compacted store reopens to the same substrate.
//!
//! A turn coupling, a located index page and a live run longer than one read stripe all survive the rewrite, the page byte for byte. The dead records are dropped, and the reopened store recovers to the same live state. A freshly compacted log reads a small non-zero dead ratio, which is the metadata-liveness gap the test records and which stays well under the compaction threshold.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::identity::RecordedModel;

    use streams::TurnDecl;

    use super::super::tests_support::tests::{chunk_payload, model, recorded, tmp_dir};

    /// **A tool round-trip's coupling survives compaction.**
    ///
    /// [`RecordType::TurnCoupling`] is a resident payload-keyed record, exactly
    /// like `Label` / `DebugId` / `TreeMetadata`: the accounting excludes it from
    /// supersession and the walker replays it into
    /// [`crate::substrate::Substrate::couplings_of`], which means compaction has
    /// to re-synthesise it from RAM. Every one of its siblings is re-emitted by
    /// [`compaction::collect_live_records`]; this one was not, so each
    /// compaction silently unjoined every tool call from its response and
    /// [`crate::summary_tree::exchange`] began grouping the two halves of a
    /// round-trip as separate exchanges.
    ///
    /// For `npcd` that is a character's act and the world's answer to it coming
    /// apart in the summary tree.
    #[test]
    fn a_turn_coupling_survives_compaction() {
        let dir = tmp_dir("coupling");
        let turn = |idx: u32, seq: u32| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: 1,
                turn_index: idx,
                turn_id_day: 0,
                turn_id_seq: seq,
                role: 1,
                block_start: 0,
                block_end: 32,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };

        // A tool call at turn 0 and its response at turn 1, joined.
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for (idx, seq) in [(0u32, 1u32), (1, 2)] {
                let id = sp.declare_stream(&turn(idx, seq)).unwrap().value;
                sp.append_tokens(id, b"token-bytes").unwrap();
                sp.commit_stream(id, 0).unwrap();
            }
            sp.write_turn_coupling(1, 0, sp.mint()).unwrap();
            sp.commit().unwrap();
        }

        let timeline = TimelineId::from_raw(1).unwrap();
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(
                substrate.couplings_of(timeline).contains(&0),
                "the coupling is durable before any compaction"
            );
            drop(sp);
        }

        // Compact, then reopen: the coupling must still be there.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.compact(&mut substrate, None).unwrap();
        }
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert!(
            substrate.couplings_of(timeline).contains(&0),
            "compaction must carry the coupling forward — without it a tool call \
             and its response stop being one exchange"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A whole-store compaction carries a located index page by location,
    /// byte for byte.** Recovered pages are left in the log
    /// ([`crate::substrate::IndexPage::Stored`]); the rewrite copies the record
    /// verbatim, and the page read after it — and after a reload — is exactly
    /// the bytes the seal wrote. A page dropped by a rewrite would leave the
    /// turn borrowable and unindexable.
    #[test]
    fn a_located_index_page_survives_compaction_byte_for_byte() {
        let dir = tmp_dir("index_page_compact");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 7,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 32,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let page: Vec<u8> = (0..9_000u32)
            .map(|i| (i.wrapping_mul(131) % 253) as u8)
            .collect();
        let sid = {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            let sid = sp.declare_stream(&decl).unwrap().value;
            sp.append_tokens(sid, b"token-bytes").unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.append_turn_index_page(sid, &page, sp.mint_for(RecordType::TurnIndexPage, sid))
                .unwrap();
            sp.commit().unwrap();
            sid
        };
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let held = substrate.index_page_of(sid).expect("the page recovers");
            assert!(held.stored().is_some(), "recovered, the page is located");
            assert_eq!(held.read_with(&mut sp), page);
            sp.compact(&mut substrate, None).unwrap();
            let after = substrate
                .index_page_of(sid)
                .expect("the page survives compaction");
            assert_eq!(
                after.read_with(&mut sp),
                page,
                "the seal's bytes, after compaction"
            );
        }
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

    /// **Full compaction carries a live run longer than one read stripe
    /// whole**: seventy one-MiB chunks back to back, read back byte for byte
    /// after the rewrite and after a reload.
    #[test]
    fn compact_carries_a_live_run_past_one_stripe() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("compact_stripes");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 56,
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
        let big = |index: u64| ChunkPayload {
            kv_bytes: (0..1024 * 1024u64)
                .map(|i| ((i + index * 17) % 251) as u8)
                .collect(),
            ..chunk_payload(index as u32)
        };
        const LIVE: u64 = 70;
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.write_chunk(sid, LIVE, 32, 4, None, &chunk_payload(1))
                .unwrap(); // superseded below
            for i in 0..LIVE {
                sp.write_chunk(sid, i, 32, 4, None, &big(i)).unwrap();
            }
            sp.write_chunk(sid, LIVE, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.compact(&mut substrate, None).unwrap();
            for i in 0..LIVE {
                assert_eq!(
                    sp.read_chunk(&substrate, sid, i).unwrap(),
                    big(i),
                    "chunk {i}"
                );
            }
        }
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
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
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_drops_dead_records_and_reopens_identical() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("compact");
        // The chunk stream needs a decl, else it is an orphan whose chunks are
        // (correctly) reclaimed — here we are testing dead-record dropping, so the
        // stream must be legitimately live.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 55,
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
        let live_chunk = chunk_payload(7);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            // Dead weight: superseded model records (one model, redescribed)
            // and stale chunk-0 bodies.
            for s in ["qwen3-8b", "qwen3-14b", "qwen3-30b"] {
                sp.bind_model(&model(s, 1)).unwrap();
            }
            sp.bind_model(&model("qwen3-235b-live", 1)).unwrap();
            sp.set_template(b"the-template").unwrap();
            for seed in [97, 98, 99] {
                sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(seed))
                    .unwrap();
            }
            sp.write_chunk(sid, 0, 32, 4, None, &live_chunk).unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.commit().unwrap();
            // Re-walk so substrate.streams sees the freshly-written
            // chunks (writes don't auto-apply to substrate).
            drop(sp);
            substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();

            assert!(
                sp.dead_ratio(&substrate) >= COMPACTION_DEAD_RATIO_THRESHOLD,
                "a log half dead weight wants compaction, ratio = {}",
                sp.dead_ratio(&substrate)
            );
            assert!(
                !sp.should_compact(&substrate),
                "a small log never auto-compacts regardless of its ratio"
            );
            sp.compact(&mut substrate, None).unwrap();

            // Caches and the manifest survive the swap.
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-235b-live", 1)))
            );
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), live_chunk);
            // `dead_ratio` is liveness-derived (consistent with the maintenance
            // trigger), and `segment_liveness` intentionally counts only the
            // read-back records (chunks/tokens + manifest singletons) as live, not
            // the per-stream metadata (`StreamDecl`/`Commit`/…). So a freshly
            // compacted log reads slightly non-zero — the metadata-liveness
            // blindspot — but well under the compaction threshold (no reclaimable
            // chunk/token weight remains). In production, where KV chunks dominate,
            // this residual is negligible.
            assert!(
                sp.dead_ratio(&substrate) < COMPACTION_DEAD_RATIO_THRESHOLD,
                "a freshly compacted log has no reclaimable dead weight, ratio = {}",
                sp.dead_ratio(&substrate)
            );
            assert!(!sp.should_compact(&substrate));
        }
        // The compacted file recovers to the same live substrate on reopen.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-235b-live", 1)))
            );
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), live_chunk);
        }
        assert!(
            !dir.join(SUBSTRATE_DIR).join(".compact").exists(),
            "the compaction scratch file is renamed away, not left behind"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
