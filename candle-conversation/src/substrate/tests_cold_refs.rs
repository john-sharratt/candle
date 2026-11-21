//! Tests that cold-tier references follow their records through compaction,
//! distillation and reload, and that reload keeps the summary nodes it replayed.
//!
//! Compaction moves every record to a new offset, so a residence's cold
//! references must be re-pointed at the rebuilt index, and a stream the index no
//! longer names is left alone. Distillation drops the chunk records a reference
//! names, so the reference must go with them. A reloaded turn keeps the summary
//! kind its log replayed, and a turn coupling for a timeline not yet registered
//! waits for it.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::segment::FIRST_SEGMENT;

    use crate::persistence::streams::TurnDecl;

    use crate::projection::{GroupId, LayerId, TimelineId};

    use crate::token_buffer::TokenBuffer;

    use super::super::tests_working_set::tests::make_timeline;

    use super::super::tests_eviction::tests::install_hot_and_warm;

    /// A zero target evicts nothing — there's no incoming load to make room
    /// for, so the working set is left fully hot.
    #[test]
    fn evict_hot_to_free_zero_target_is_noop() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let report = sub.evict_hot_to_free(&[], &[], 0);
        assert_eq!(report.count, 0);
        assert_eq!(report.bytes, 0);
        assert!(sub.residence(a).has_hot());
    }

    /// On reload, `TreeMetadata` records are replayed (summary-node kinds set)
    /// during the walker open pass, BEFORE the per-turn `restore_turn` loop.
    /// `restore_turn` must not clobber a reloaded summary node's kind back to
    /// `Normal` — that collapse re-enqueued the whole history, re-summarised
    /// prior summaries into garbage, and broke the tree's AVL invariant on
    /// restart.
    #[test]
    fn restore_turn_preserves_reloaded_tree_meta() {
        let (_, _, timeline, mut sub) = make_timeline();

        // Stand in for the open-pass replay: index 0 is a SummaryOfTurns leaf.
        let leaf = TurnIndex(0);
        sub.set_tree_meta(
            timeline,
            leaf,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfTurns,
                children: vec![TurnIndex(7)],
                tree_height: 1,
            },
            RecordIdentity::NONE,
        );

        // The reconstruct loop then restores turn 0.
        let idx = sub.restore_turn(
            timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            20,
            None,
            0,
            0,
        );
        assert_eq!(idx, leaf, "first restored turn lands at index 0");
        let meta = sub.tree_meta_of(timeline, leaf).expect("tree meta present");
        assert_eq!(
            meta.kind,
            TurnKind::SummaryOfTurns,
            "restore_turn must preserve the reloaded summary-node kind",
        );
        assert_eq!(meta.children, vec![TurnIndex(7)], "children preserved");

        // A turn with no prior tree meta still defaults to Normal.
        let idx2 = sub.restore_turn(
            timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            10,
            None,
            0,
            0,
        );
        assert_eq!(
            sub.tree_meta_of(timeline, idx2).map(|m| m.kind),
            Some(TurnKind::Normal),
            "a freshly restored turn with no tree meta defaults to Normal",
        );
    }

    /// `restore_turn` with `cold = Some(...)` lands the residence as
    /// cold-marker — classifier-ready for cold→hot promotion. With
    /// `cold = None`, the residence is left empty (legitimately
    /// recoverable but missing chunks).
    #[test]
    fn restore_turn_populates_cold_tier_when_provided() {
        let (_, _, timeline, mut sub) = make_timeline();

        let cold_payload = vec![StoredSequence {
            chunks: vec![StoredChunk {
                log_offset: 4096,
                record_len: 1024,
                token_count: 32,
            }],
            token_count: 32,
        }];
        let idx = sub.restore_turn(
            timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            32,
            Some(cold_payload),
            0,
            1,
        );
        let residence = sub.turn_residence(timeline, idx).unwrap();
        assert!(sub.residence(residence).has_cold(), "cold installed");
        assert!(
            sub.residence(residence).hot().is_none(),
            "hot empty (cold-marker)"
        );
        assert!(sub.residence(residence).warm().is_none(), "warm empty");
        assert_eq!(
            sub.residence(residence).byte_size(),
            1024,
            "byte_size summed from cold record_len"
        );

        // None branch — no cold payload.
        let idx2 = sub.restore_turn(
            timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            0,
            None,
            0,
            0,
        );
        let r2 = sub.turn_residence(timeline, idx2).unwrap();
        assert!(sub.residence(r2).cold().is_none());
        assert!(sub.residence(r2).hot().is_none());
        assert!(sub.residence(r2).warm().is_none());
    }

    /// Compaction moves every record to a new offset and re-walks the
    /// stream index; `refresh_cold_refs` must re-point each residence's
    /// `StoredSequence`s at the rebuilt index (same grid shape, new
    /// offsets), and must leave residences whose stream is absent from
    /// the active index untouched (inherited-log streams).
    #[test]
    fn refresh_cold_refs_repoints_residences_at_the_stream_index() {
        let (_, _, timeline, mut sub) = make_timeline();
        let stream_id = turn_stream_id(timeline.raw(), 0);
        let n_layers = 2usize;
        let chunks_per_layer = 2usize;

        // Pre-compaction stream index + matching cold refs.
        let old_loc = |flat: u64| ChunkLoc {
            segment: FIRST_SEGMENT,
            offset: 4096 + flat * 4096,
            payload_len: 100,
            record_size: 4096,
            token_count: if flat % 2 == 1 { 12 } else { 32 },
            format: 4,
        };
        for flat in 0..(n_layers * chunks_per_layer) as u64 {
            sub.apply_chunk_loc(stream_id, flat, old_loc(flat), RecordIdentity::NONE);
        }
        let cold: Vec<StoredSequence> = (0..n_layers)
            .map(|l| StoredSequence {
                chunks: (0..chunks_per_layer)
                    .map(|c| {
                        let loc = old_loc((l * chunks_per_layer + c) as u64);
                        StoredChunk {
                            log_offset: loc.offset,
                            record_len: loc.record_size,
                            token_count: loc.token_count as u16,
                        }
                    })
                    .collect(),
                token_count: 44,
            })
            .collect();
        let idx = sub.restore_turn(
            timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            44,
            Some(cold),
            0,
            chunks_per_layer as u64,
        );

        // An unrelated residence whose stream is NOT in the active
        // index (an inherited-log borrow) — must stay untouched.
        let other_timeline = TimelineId::from_raw(4242).unwrap();
        sub.register_timeline(other_timeline, LayerId::for_test(1), GroupId::for_test(1));
        let untouched = vec![StoredSequence {
            chunks: vec![StoredChunk {
                log_offset: 777_216,
                record_len: 4096,
                token_count: 32,
            }],
            token_count: 32,
        }];
        let other_idx = sub.restore_turn(
            other_timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            32,
            Some(untouched),
            0,
            1,
        );

        // "Compaction": every chunk record lands at a new offset with a
        // new padded size.
        let new_loc = |flat: u64| ChunkLoc {
            segment: FIRST_SEGMENT,
            offset: 100_000 + flat * 8192,
            payload_len: 100,
            record_size: 8192,
            token_count: old_loc(flat).token_count,
            format: 4,
        };
        for flat in 0..(n_layers * chunks_per_layer) as u64 {
            sub.apply_chunk_loc(stream_id, flat, new_loc(flat), RecordIdentity::NONE);
        }
        sub.refresh_cold_refs();

        let residence = sub.turn_residence(timeline, idx).unwrap();
        let cold = sub.residence(residence).cold().unwrap();
        for (l, seq) in cold.iter().enumerate() {
            for (c, chunk) in seq.chunks.iter().enumerate() {
                let flat = (l * chunks_per_layer + c) as u64;
                assert_eq!(chunk.log_offset, new_loc(flat).offset);
                assert_eq!(chunk.record_len, 8192);
                assert_eq!(chunk.token_count, new_loc(flat).token_count as u16);
            }
        }
        let other_res = sub.turn_residence(other_timeline, other_idx).unwrap();
        let other_cold = sub.residence(other_res).cold().unwrap();
        assert_eq!(
            other_cold[0].chunks[0].log_offset, 777_216,
            "residence without an active-index stream stays untouched"
        );
    }

    /// **A maintenance op refreshes the residences of the streams it moved,
    /// and no others**: two turns' chunks are repointed in the index, one
    /// stream is named moved, and only its turn's cold reference follows.
    #[test]
    fn refresh_cold_refs_of_repoints_only_the_moved_streams() {
        let (_, _, first, mut sub) = make_timeline();
        let second = TimelineId::from_raw(4243).unwrap();
        sub.register_timeline(second, LayerId::for_test(1), GroupId::for_test(1));
        let loc = |offset: u64| ChunkLoc {
            segment: FIRST_SEGMENT,
            offset,
            payload_len: 100,
            record_size: 4096,
            token_count: 32,
            format: 4,
        };
        let cold_at = |offset: u64| {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: offset,
                    record_len: 4096,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };
        let mut turn = |timeline: TimelineId, offset: u64| {
            let stream = turn_stream_id(timeline.raw(), 0);
            sub.apply_chunk_loc(stream, 0, loc(offset), RecordIdentity::NONE);
            let idx = sub.restore_turn(
                timeline,
                TurnLayout::default(),
                TokenBuffer::default(),
                32,
                Some(cold_at(offset)),
                0,
                1,
            );
            (stream, idx)
        };
        let (moved_stream, moved_idx) = turn(first, 4096);
        let (kept_stream, kept_idx) = turn(second, 8192);
        sub.apply_chunk_loc(moved_stream, 0, loc(100_000), RecordIdentity::NONE);
        sub.apply_chunk_loc(kept_stream, 0, loc(200_000), RecordIdentity::NONE);

        sub.refresh_cold_refs_of(&HashSet::from([moved_stream]));

        let offset_of = |sub: &Substrate, timeline, idx| {
            let r = sub.turn_residence(timeline, idx).unwrap();
            sub.residence(r).cold().unwrap()[0].chunks[0].log_offset
        };
        assert_eq!(offset_of(&sub, first, moved_idx), 100_000);
        assert_eq!(
            offset_of(&sub, second, kept_idx),
            8192,
            "a stream the op did not move keeps its reference"
        );
    }

    /// Distillation sheds the cold REFERENCE, not just the resident copies.
    ///
    /// The compactor drops the chunk records a distilled timeline's turns point
    /// at (`keep_chunks = distill.is_none()`), and nothing removes the in-RAM
    /// reference to them: maintenance repoints relocated records and the stream
    /// index is insert-only. A ref left behind therefore names a segment that
    /// gets unlinked while still reading as a cold copy — which is how an
    /// archived conversation (`TextOnly`: tokens kept, chunks shed) kept being
    /// selected, failing `cold-load found no chunks`, and repeating every turn.
    #[test]
    fn distillation_clears_the_cold_reference_it_makes_dangling() {
        for mode in [DistillMode::TextOnly, DistillMode::ProvenanceOnly] {
            let (_, _, timeline, mut sub) = make_timeline();
            let cold = vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 4096,
                    record_len: 4096,
                    token_count: 32,
                }],
                token_count: 32,
            }];
            let idx = sub.restore_turn(
                timeline,
                TurnLayout::default(),
                TokenBuffer::default(),
                32,
                Some(cold),
                0,
                1,
            );
            assert_eq!(
                sub.turn_tier_state(timeline, idx).map(|t| t.cold),
                Some(true),
                "{mode:?}: the turn starts with a cold copy"
            );

            sub.distill_timeline(timeline, mode);

            assert_eq!(
                sub.turn_tier_state(timeline, idx).map(|t| t.cold),
                Some(false),
                "{mode:?}: a distilled turn must not advertise a cold copy the \
                 compactor is about to shed — selection chases it, the load \
                 fails, and nothing ever clears it"
            );
        }
    }

    /// A timeline in an append-only ingest layer (repo_map, code_reading)
    /// never enqueues turns for the summariser, regardless of its own
    /// `summarize` flag — enforced in `push_pending_summary` itself so no
    /// ingest path can forget to clear the flag. A dialogue timeline on a
    /// normal layer still enqueues.
    #[test]
    fn append_only_layer_turns_never_enqueue_for_summary() {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let mut sub = Substrate::new();
        let alloc = TimelineAllocator::new();

        let ingest_layer = LayerId::for_test(1);
        let ingest_tl = alloc.next();
        sub.register_timeline(ingest_tl, ingest_layer, GroupId::for_test(1));
        sub.mark_layer_append_only(ingest_layer);

        let dlg_tl = alloc.next();
        sub.register_timeline(dlg_tl, LayerId::for_test(2), GroupId::for_test(2));

        // Both gates ON, which is the whole point: the append-only refusal must
        // hold at the mechanism even for a timeline whose own flag says yes. (The
        // production default is now off for every timeline — the summariser is
        // disconnected — so this has to be set explicitly to test the refusal.)
        sub.set_timeline_summarize(ingest_tl, true);
        sub.set_timeline_summarize(dlg_tl, true);

        sub.push_pending_summary(ingest_tl, TurnIndex(0));
        sub.push_pending_summary(dlg_tl, TurnIndex(0));

        assert_eq!(
            sub.pending_summary_len(ingest_tl),
            0,
            "append-only ingest turn must not queue for the summariser"
        );
        assert_eq!(
            sub.pending_summary_len(dlg_tl),
            1,
            "dialogue turn must still queue"
        );
    }

    /// Replaying a coupling record makes the round-trip visible to the
    /// summariser and selector. Idempotent: the records carry no ordering, and a
    /// recovery walk may present the same one twice.
    #[test]
    fn turn_coupling_replays_into_the_timeline_set() {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let mut sub = Substrate::new();
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        sub.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));

        let payload = TurnCouplingPayload {
            timeline_id: timeline.raw(),
            from_turn: 4,
        };
        sub.apply_turn_coupling(&payload, RecordIdentity::NONE);
        // A duplicate — must not change the set.
        sub.apply_turn_coupling(&payload, RecordIdentity::NONE);
        assert_eq!(sub.couplings_of(timeline), [4u32].into_iter().collect());
    }

    /// A coupling naming a timeline not registered yet conjures no entry, and
    /// is not dropped either: it waits for the timeline's first turn
    /// declaration and lands when that arrives (§6.5).
    #[test]
    fn turn_coupling_for_an_unregistered_timeline_waits_for_it() {
        let mut sub = Substrate::new();
        let alloc = crate::projection::TimelineAllocator::new();
        let timeline = alloc.next();
        sub.apply_turn_coupling(
            &TurnCouplingPayload {
                timeline_id: timeline.raw(),
                from_turn: 1,
            },
            RecordIdentity::NONE,
        );
        assert!(sub.couplings_of(timeline).is_empty());
        assert!(sub.timeline_entry(timeline).is_none());
        sub.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));
        assert_eq!(sub.couplings_of(timeline), [1u32].into_iter().collect());
    }

    /// A timeline that never called a tool has no couplings, so every turn is its
    /// own exchange — the pre-coupling behaviour, unchanged.
    #[test]
    fn a_timeline_without_couplings_reports_an_empty_set() {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let mut sub = Substrate::new();
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        sub.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));
        assert!(sub.couplings_of(timeline).is_empty());
    }

    /// **Asking a turn for any of some tags reads its declared tags in place**:
    /// the answer `turn_tags` gives, for a tagged turn, an untagged one and a
    /// turn that is not there.
    #[test]
    fn a_turns_tags_are_matched_in_place() {
        let mut sub = Substrate::new();
        let decl = |tl: u64, tags: &[&str]| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
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
                tags: tags.iter().map(|t| t.to_string()).collect(),
            })
        };
        sub.apply_stream_decl(
            turn_stream_id(7, 0),
            decl(7, &["tool", "code"]),
            RecordIdentity::NONE,
        );
        sub.apply_stream_decl(turn_stream_id(8, 0), decl(8, &[]), RecordIdentity::NONE);
        let wanted = |names: &[&str]| names.iter().map(|t| t.to_string()).collect::<Vec<_>>();
        let first = TurnIndex(0);
        for (tl, tags) in [
            (7u64, wanted(&["code"])),
            (7, wanted(&["guide"])),
            (7, wanted(&[])),
            (8, wanted(&["tool"])),
            (9, wanted(&["tool"])),
        ] {
            let timeline = TimelineId::from_raw(tl).unwrap();
            let by_copy = sub
                .turn_tags(timeline, first)
                .iter()
                .any(|t| tags.contains(t));
            assert_eq!(
                sub.turn_has_any_tag(timeline, first, &tags),
                by_copy,
                "{tl} {tags:?}"
            );
        }
        assert!(sub.turn_has_any_tag(TimelineId::from_raw(7).unwrap(), first, &wanted(&["code"])));
    }

    /// `tombstoned_stream_bytes` sums the on-disk record bytes of every
    /// tombstoned timeline's turn streams — the dead weight the
    /// header-keyed accounting can't attribute.
    #[test]
    fn tombstoned_stream_bytes_sums_dead_timelines() {
        let mut sub = Substrate::new();
        let decl_for = |tl: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
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
            })
        };
        let dead_sid = turn_stream_id(7, 0);
        let live_sid = turn_stream_id(8, 0);
        for (sid, tl) in [(dead_sid, 7u64), (live_sid, 8u64)] {
            sub.apply_stream_decl(sid, decl_for(tl), RecordIdentity::NONE);
            sub.apply_chunk_loc(
                sid,
                0,
                ChunkLoc {
                    segment: FIRST_SEGMENT,
                    offset: 4096,
                    payload_len: 100,
                    record_size: 8192,
                    token_count: 32,
                    format: 4,
                },
                RecordIdentity::NONE,
            );
            sub.apply_tokens_loc(
                sid,
                RecordLoc {
                    segment: FIRST_SEGMENT,
                    offset: 20_480,
                    payload_len: 64,
                    record_size: 4096,
                },
                RecordIdentity::NONE,
            );
        }
        assert_eq!(sub.tombstoned_stream_bytes(), 0, "nothing tombstoned yet");
        sub.tombstone_timeline(TimelineId::from_raw(7).unwrap());
        assert_eq!(
            sub.tombstoned_stream_bytes(),
            8192 + 4096,
            "only the tombstoned timeline's chunk + tokens bytes count"
        );
    }

    /// **A throwaway timeline's mark ends with its tombstone's expiry**: it
    /// stays through the tombstone, so a decode winding down seals nothing to
    /// disk, and is forgotten once compaction expires the tombstone — a
    /// daemon opening one per question holds none of them for good.
    #[test]
    fn a_transient_mark_is_forgotten_when_its_tombstone_expires() {
        let mut sub = Substrate::new();
        let thrown = TimelineId::from_raw(41).unwrap();
        let kept = TimelineId::from_raw(42).unwrap();
        sub.mark_timeline_transient(thrown);
        sub.mark_timeline_transient(kept);
        sub.tombstone_timeline(thrown);
        assert!(
            sub.is_timeline_transient(thrown),
            "held through the wind-down"
        );
        sub.forget_expired_tombstones(&Expired {
            timelines: HashSet::from([thrown]),
            keys: HashSet::new(),
        });
        assert!(!sub.is_timeline_transient(thrown));
        assert!(sub.is_timeline_transient(kept), "a live one keeps its mark");
    }

    /// **A section tombstone ends a generation; the next declaration starts
    /// one.** The tombstone drops the stream's index entry — so neither the
    /// compactor nor `tombstoned_stream_bytes` sees its records (the record
    /// accounting retires them instead) — and a fresh declaration of the same
    /// content-addressed stream lifts the flag and indexes only its own
    /// records. The other stream is untouched throughout.
    #[test]
    fn a_section_tombstone_ends_a_generation_and_a_new_decl_starts_one() {
        use crate::persistence::content_hash::ContentHash;
        use crate::persistence::streams::{ContentAddress, SectionDecl};

        let mut sub = Substrate::new();
        let decl_for = |lo: u64| {
            StreamDecl::PromptSection(SectionDecl {
                address: ContentAddress {
                    prefix_hash: ContentHash { lo: 1, hi: 0 },
                    section_hash: ContentHash { lo, hi: 0 },
                },
                debug_name: "tool_catalog".to_string(),
            })
        };
        let dead_decl = decl_for(7);
        let live_decl = decl_for(8);
        let dead_sid = dead_decl.stream_id();
        let live_sid = live_decl.stream_id();
        for (sid, decl) in [(dead_sid, dead_decl), (live_sid, live_decl)] {
            sub.apply_stream_decl(sid, decl, RecordIdentity::NONE);
            sub.apply_chunk_loc(
                sid,
                0,
                ChunkLoc {
                    segment: FIRST_SEGMENT,
                    offset: 4096,
                    payload_len: 100,
                    record_size: 8192,
                    token_count: 32,
                    format: 4,
                },
                RecordIdentity::NONE,
            );
            sub.apply_tokens_loc(
                sid,
                RecordLoc {
                    segment: FIRST_SEGMENT,
                    offset: 20_480,
                    payload_len: 64,
                    record_size: 4096,
                },
                RecordIdentity::NONE,
            );
        }
        sub.tombstone_section(dead_sid);
        assert!(sub.is_section_tombstoned(dead_sid));
        assert!(
            sub.stream_of(dead_sid).is_none(),
            "the old generation is unindexed"
        );
        assert!(sub.stream_of(live_sid).is_some());
        assert_eq!(
            sub.tombstoned_stream_bytes(),
            0,
            "sections are the accounting's"
        );

        sub.apply_stream_decl(dead_sid, decl_for(7), RecordIdentity::NONE);
        sub.apply_chunk_loc(
            dead_sid,
            1,
            ChunkLoc {
                segment: FIRST_SEGMENT,
                offset: 40_960,
                payload_len: 100,
                record_size: 8192,
                token_count: 32,
                format: 4,
            },
            RecordIdentity::NONE,
        );
        assert!(
            !sub.is_section_tombstoned(dead_sid),
            "the new decl revives it"
        );
        let revived = sub.stream_of(dead_sid).expect("revived stream is indexed");
        assert_eq!(
            revived.chunks.keys().copied().collect::<Vec<_>>(),
            vec![1],
            "only the new generation's chunk"
        );
        assert!(revived.tokens.is_none(), "the old tokens record stays dead");
    }
}
