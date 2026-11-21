//! Tests of the index cache across eviction, rebuild, reuse and compaction.
//!
//! A rescan after eviction rebuilds residency and reproduces the same result. The
//! cache is keyed by fingerprint, so two segment sets used alternately both stay
//! cached. An upload of another conversation's turn leaves the index valid, while
//! a reseal of one of its own turns drops it. A compaction that moves pages must
//! leave every score identical, which exercises the kernel's own use of the page
//! addresses.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use std::cell::Cell;

    use super::super::super::scan_plan::fingerprint_segments;

    use super::super::super::tests::gallery_pool;

    use super::*;

    use crate::persistence::content_hash::turn_stream_id;

    use super::super::tests_support::tests::{rand_sig, sig};

    /// Governor relief path: a scan → evict EVERYTHING (as the cheap-rung relief
    /// closure does under VRAM pressure) → the next scan rebuilds residency from
    /// the same sigs and reproduces the result bit-identically. Skips without CUDA.
    #[test]
    fn eviction_then_rebuild_is_stable() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        let t0: Vec<WideQSig> = (0..50).map(|t| sig(0xA00 + t)).collect(); // 2 pages
        let t1: Vec<WideQSig> = (0..12).map(|t| sig(0xB00 + t)).collect();
        let t2: Vec<WideQSig> = (0..7).map(|t| sig(0xC00 + t)).collect();
        let segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(9, 0),
                        fingerprint: 1,
                        turn: (&t0).into(),
                        start: 0,
                        end: 50,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(9, 1),
                        fingerprint: 2,
                        turn: (&t1).into(),
                        start: 0,
                        end: 12,
                        case: 1,
                    },
                ],
                n_cases: 2,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            },
            PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(9, 2),
                    fingerprint: 3,
                    turn: (&t2).into(),
                    start: 0,
                    end: 7,
                    case: 0,
                }],
                n_cases: 1,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            },
        ];
        let probe = vec![sig(0xA00 + 5), sig(0xC00 + 2), sig(0x999)];
        let weights = [1.0f32, 0.5, 2.0];

        let first = arena
            .scan_weighted(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        let before = arena.resident_turns();
        assert_eq!(before, 3, "three turns resident after the first scan");
        assert!(arena.resident_bytes() > 0);

        // The scan unpinned its working set, so relief can evict everything.
        let freed = arena.evict_lru(u64::MAX);
        assert!(freed > 0, "eviction freed VRAM");
        assert_eq!(arena.resident_turns(), 0, "all turns evicted");

        // Re-scan → residency rebuilds → bit-identical result.
        let second = arena
            .scan_weighted(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        assert_eq!(arena.resident_turns(), before, "rebuilt the same residency");
        assert_eq!(first[0].len(), second[0].len());
        for (i, (a, b)) in first[0].iter().zip(&second[0]).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "case {i}: rebuild after eviction must reproduce the scan bit-identically"
            );
        }
    }

    /// The index cache is fingerprint-KEYED (not a single slot): two distinct
    /// segment sets scanned alternately both stay cached, so neither rebuilds
    /// after its first scan — the multi-belief-group reprojection pattern. Each
    /// still produces the correct result. Skips without CUDA.
    #[test]
    fn index_cache_keyed_by_fingerprint_no_thrash() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        // Two distinct "groups" — different turns/files → different fingerprints.
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xA00 + t)).collect();
        let b: Vec<WideQSig> = (0..15).map(|t| sig(0xB00 + t)).collect();
        let seg_a = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(7, 0),
                fingerprint: 1,
                turn: (&a).into(),
                start: 0,
                end: 20,
                case: 0,
            }],
            n_cases: 1,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law: VoteLaw::SINGLE,
        }];
        let seg_b = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(8, 0),
                fingerprint: 2,
                turn: (&b).into(),
                start: 0,
                end: 15,
                case: 0,
            }],
            n_cases: 1,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law: VoteLaw::SINGLE,
        }];
        let probe = vec![sig(0xA00 + 3), sig(0xB00 + 3)];

        // Prime both caches.
        let a1 = arena
            .scan_weighted(&seg_a, &[probe.as_slice()], &[])
            .unwrap();
        let b1 = arena
            .scan_weighted(&seg_b, &[probe.as_slice()], &[])
            .unwrap();
        // Interleave: with a single-slot cache these would each rebuild; keyed,
        // they both hit and reproduce their first result bit-identically. (No
        // residency mutation between scans, so the generation holds.)
        for _ in 0..3 {
            let a2 = arena
                .scan_weighted(&seg_a, &[probe.as_slice()], &[])
                .unwrap();
            let b2 = arena
                .scan_weighted(&seg_b, &[probe.as_slice()], &[])
                .unwrap();
            assert_eq!(a1, a2, "group A cached result must be stable");
            assert_eq!(b1, b2, "group B cached result must be stable");
        }
        assert_eq!(arena.resident_turns(), 2, "both turns resident, no churn");
    }

    /// **Another conversation's upload leaves this index valid; a change to
    /// one of its own turns does not.** The index cache used to be validated by
    /// a device-wide generation, so ingest sealing unrelated turns rebuilt a
    /// dialogue's whole index every reprojection. Skips without CUDA.
    #[test]
    fn an_unrelated_upload_keeps_the_index_and_a_reseal_drops_it() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xA00 + t)).collect();
        let seg_a = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(7, 0),
                fingerprint: 1,
                turn: (&a).into(),
                start: 0,
                end: 20,
                case: 0,
            }],
            n_cases: 1,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law: VoteLaw::SINGLE,
        }];
        let probe = vec![sig(0xA00 + 3)];
        let fp = fingerprint_segments(&seg_a);
        arena
            .scan_weighted(&seg_a, &[probe.as_slice()], &[])
            .unwrap();

        // Another conversation's turn uploads: the device churns, this index
        // does not.
        let other: Vec<WideQSig> = (0..10).map(|t| sig(0xC00 + t)).collect();
        let gen = arena.residency_gen();
        arena
            .ensure_resident(
                turn_stream_id(9, 0),
                &other,
                3,
                Holder::Shared,
                GroupSet::all(3),
                Keep::All,
            )
            .unwrap();
        assert!(arena.residency_gen() > gen, "the upload is counted");
        let all = [GroupSet::all(3)];
        let kept = arena
            .reuse_index(fp, &all)
            .expect("an unrelated upload keeps the index");
        drop(kept);

        // This index's own turn is re-sealed under a new fingerprint: its pages
        // are replaced, so the cached addresses are gone with them.
        arena
            .ensure_resident(
                turn_stream_id(7, 0),
                &a,
                2,
                Holder::Shared,
                GroupSet::all(3),
                Keep::All,
            )
            .unwrap();
        assert_eq!(
            arena.reuse_index(fp, &all).err(),
            Some(IndexMiss::Stale),
            "a re-sealed turn invalidates every index over it"
        );
        assert_eq!(
            arena.reuse_index(fp, &all).err(),
            Some(IndexMiss::Absent),
            "the stale index was dropped from the cache"
        );
    }

    /// **Readers weighing different groups share one index.** A turn read
    /// weighing groups 1–2 and then weighing all three is one index, widened
    /// once to all three: from then on both readers reuse it and its rows —
    /// no rebuild, no launch — and each still gets exactly its own scan, the
    /// narrow reader's unweighed group voting for nothing.
    #[test]
    fn readers_weighing_different_groups_share_one_index() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let mut rng = 0x5AA2_ED1D_u64;
        let a: Vec<WideQSig> = (0..45).map(|_| rand_sig(&mut rng)).collect();
        let probe: Vec<WideQSig> = (0..12)
            .map(|i| {
                if i % 3 == 0 {
                    a[i * 3].clone()
                } else {
                    rand_sig(&mut rng)
                }
            })
            .collect();
        let narrow = GroupSet::default().with(1).with(2);
        let full = GroupSet::all(3);
        let segment = |groups| {
            vec![PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(61, 0),
                    fingerprint: 61,
                    turn: (&a).into(),
                    start: 0,
                    end: a.len(),
                    case: 0,
                }],
                n_cases: 1,
                holder: Holder::Shared,
                groups,
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            }]
        };
        let narrow_w: &[f32] = &[0.0, 1.0, 1.0];
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
                .collect::<Vec<_>>()
        };
        // Each reader alone, on an arena of its own.
        let alone = |groups, w: &[f32]| {
            let arena = GalleryArena::new(&device, 24, 3).unwrap();
            let scored = arena
                .scan_weighted(&segment(groups), &[probe.as_slice()], w)
                .unwrap();
            let votes = arena
                .scan_votes(&segment(groups), &[probe.as_slice()])
                .unwrap();
            (bits(&scored), votes)
        };
        let (want_narrow, votes_narrow) = alone(narrow, narrow_w);
        let (want_full, votes_full) = alone(full, &[]);
        assert!(votes_narrow.case.iter().any(|&c| c >= 0), "nothing scored");

        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let fp = fingerprint_segments(&segment(narrow));
        assert_eq!(fp, fingerprint_segments(&segment(full)));
        let probes = [probe.as_slice()];
        arena
            .scan_weighted(&segment(narrow), &probes, narrow_w)
            .unwrap();
        assert!(matches!(
            arena.reuse_index(fp, &[full]),
            Err(IndexMiss::Narrower { ref covering }) if covering == &[full]
        ));
        arena.scan_weighted(&segment(full), &probes, &[]).unwrap();

        scan_split::take();
        let got_narrow = arena
            .scan_weighted(&segment(narrow), &probes, narrow_w)
            .unwrap();
        let got_full = arena.scan_weighted(&segment(full), &probes, &[]).unwrap();
        let split = scan_split::take();
        assert_eq!(bits(&got_narrow), want_narrow);
        assert_eq!(bits(&got_full), want_full);
        assert_eq!(split.launches, 0, "both read the shared index's rows");
        assert_eq!(
            arena.scan_votes(&segment(narrow), &probes).unwrap(),
            votes_narrow,
            "the unweighed group votes for nothing"
        );
        assert_eq!(
            arena.scan_votes(&segment(full), &probes).unwrap(),
            votes_full
        );
    }

    /// **An index over a turn the gallery gave up is stale.** Built, then its
    /// one turn dropped from residency: the cached index names a run that is
    /// gone, so reuse refuses it as stale rather than handing out freed pages.
    #[test]
    fn an_index_over_an_evicted_turn_is_stale() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xB00 + t)).collect();
        let seg = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(8, 0),
                fingerprint: 1,
                turn: (&a).into(),
                start: 0,
                end: 20,
                case: 0,
            }],
            n_cases: 1,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law: VoteLaw::SINGLE,
        }];
        let probe = vec![sig(0xB00 + 5)];
        let fp = fingerprint_segments(&seg);
        arena.scan_weighted(&seg, &[probe.as_slice()], &[]).unwrap();
        arena.drop_turns(&[turn_stream_id(8, 0)]);
        assert_eq!(arena.resident_turns(), 0);
        assert_eq!(
            arena.reuse_index(fp, &[GroupSet::all(3)]).err(),
            Some(IndexMiss::Stale)
        );
    }

    /// **A compaction must be invisible to the scan.** Score the same segments
    /// before and after packing the pages and require the vectors to be
    /// bit-identical.
    ///
    /// This is the test that matters for the pass, because it exercises the real
    /// consumer — the kernel dereferencing the page addresses in `PagedIndex` —
    /// rather than the bookkeeping that produced them. The two ways a move goes
    /// wrong both land here and nowhere else: a page copied to the wrong place
    /// changes the scores, and a page moved without bumping `residency_gen` leaves
    /// the cached index pointing at the vacated address, which the next scan reads
    /// as whatever the arena has since put there. Neither is a fault; both are
    /// quietly wrong retrieval.
    #[test]
    fn a_compaction_leaves_every_score_identical() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        // Enough 64-page turns to fill an arena and spill into the next, so the
        // pass has an arena it can empty (`plan_slot_moves` moves nothing it
        // cannot empty an arena with).
        let per_arena = candle_nn::kv_cache::REGION_BYTES
            / candle_nn::kv_cache::slot_stride(arena.page_bytes_for(GroupSet::all(3)) as usize);
        let turn_tokens = (64 * super::super::super::PAGE_TOKENS) as u64;
        let turns: Vec<Vec<WideQSig>> = (0..(per_arena / 64 + 3) as u64)
            .map(|t| (0..turn_tokens).map(|k| sig((t << 32) + k)).collect())
            .collect();
        let windows: Vec<PagedWindow<'_>> = turns
            .iter()
            .enumerate()
            .map(|(i, turn)| PagedWindow {
                sid: turn_stream_id(40 + i as u64, 0),
                fingerprint: 100 + i as u64,
                turn: turn.into(),
                start: 0,
                end: turn.len(),
                case: i,
            })
            .collect();
        let segments = vec![PagedSegment {
            windows,
            n_cases: turns.len(),
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law: VoteLaw::SINGLE,
        }];
        let probe = vec![sig(1 << 32), sig((3u64 << 32) + 5), sig(0xFEED)];

        // A quiet turn, uploaded first so it sits at the foot of the first
        // arena, below every hole the pass fills: no move touches it.
        let quiet_turn: Vec<WideQSig> = (0..32u64).map(|k| sig((99u64 << 32) + k)).collect();
        let quiet_sid = turn_stream_id(39, 0);
        let quiet = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: quiet_sid,
                fingerprint: 99,
                turn: (&quiet_turn).into(),
                start: 0,
                end: quiet_turn.len(),
                case: 0,
            }],
            n_cases: 1,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law: VoteLaw::SINGLE,
        }];
        arena
            .scan_weighted(&quiet, &[probe.as_slice()], &[])
            .unwrap();

        let before = arena
            .scan_weighted(&segments, &[probe.as_slice()], &[])
            .unwrap();
        // Scanned again, the quiet index is checked against the uploads since
        // it was built, and trusted from then on.
        arena
            .scan_weighted(&quiet, &[probe.as_slice()], &[])
            .unwrap();
        let stale_of = |sid| {
            arena
                .indexes
                .load()
                .iter()
                .filter(|(_, idx)| idx.members.contains_key(&sid))
                .map(|(_, idx)| idx.hold.stale())
                .collect::<Vec<bool>>()
        };
        assert_eq!(stale_of(quiet_sid), vec![false], "checked once rescanned");

        // Scatter: drop the first arena's first turns, leaving more holes under
        // live pages than the second arena holds, then pack. `scan_weighted`
        // unpinned on its way out, so every survivor is movable.
        let dropped: Vec<_> = (0..4u64).map(|i| turn_stream_id(40 + i, 0)).collect();
        arena.drop_turns(&dropped);
        let report = arena.compact(0).unwrap();
        assert!(report.moved > 0, "nothing moved, so nothing is proven");

        // **The pass named only the turns it moved.** The quiet turn's index is
        // still trusted, so its next scan does not walk its turns; the moved
        // turns' index is marked, so its next scan does.
        assert_eq!(stale_of(quiet_sid), vec![false], "an untouched index");
        assert!(
            stale_of(turn_stream_id(40 + turns.len() as u64 - 1, 0))
                .iter()
                .all(|&s| s),
            "the moved turns' index is marked"
        );

        // The dropped turns rebuild on demand; the survivors are read at their new
        // addresses. Either way every score must reproduce exactly.
        let after = arena
            .scan_weighted(&segments, &[probe.as_slice()], &[])
            .unwrap();
        assert_eq!(
            before, after,
            "compaction changed the scan's answer — a page moved to the wrong \
             place, or a cached index outlived the move",
        );
    }

    /// **A keyed scan is the scan of the segments its plan was made from, and
    /// builds them only while the arena holds no index under the key.** The
    /// first keyed scan finds the index the plain scan cached and builds
    /// nothing; one after the turn is dropped rebuilds the index, and so the
    /// segments, once.
    #[test]
    fn a_keyed_scan_builds_its_segments_only_for_an_index() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let mut rng = 0x4B_E7ED_u64;
        let turns: Vec<Vec<WideQSig>> = (0..4)
            .map(|i| (0..(20 + 17 * i)).map(|_| rand_sig(&mut rng)).collect())
            .collect();
        let segments = || -> Vec<PagedSegment> {
            vec![PagedSegment {
                windows: turns
                    .iter()
                    .enumerate()
                    .map(|(i, t)| PagedWindow {
                        sid: turn_stream_id(9_300, i as u32),
                        fingerprint: i as u64,
                        turn: t.into(),
                        start: 0,
                        end: t.len(),
                        case: i,
                    })
                    .collect(),
                n_cases: turns.len(),
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            }]
        };
        let probe: Vec<WideQSig> = (0..9).map(|_| rand_sig(&mut rng)).collect();
        let weights = [0.5f32, 1.0, 2.0];
        let per_segment: [&[f32]; 1] = [&weights];
        let plain = arena
            .scan_segment_weighted(&segments(), &[probe.as_slice()], &per_segment)
            .unwrap();
        let plan = ScanPlan::of(&segments());
        let built = Cell::new(0usize);
        let build = || {
            built.set(built.get() + 1);
            segments()
        };
        let keyed = arena
            .scan_keyed(&plan, &build, &[probe.as_slice()], &per_segment)
            .unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&keyed[0]), bits(&plain[0]), "the keyed scan's votes");
        assert_eq!(built.get(), 0, "the cached index needs no segments");
        arena.drop_turns(&[turn_stream_id(9_300, 0)]);
        let again = arena
            .scan_keyed(&plan, &build, &[probe.as_slice()], &per_segment)
            .unwrap();
        assert_eq!(built.get(), 1, "a stale index is rebuilt from the segments");
        assert_eq!(bits(&again[0]), bits(&plain[0]), "and votes the same");
    }
}
