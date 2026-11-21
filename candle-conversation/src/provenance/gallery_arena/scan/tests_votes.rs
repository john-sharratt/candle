//! Tests that the raw votes tally to the weighted scan, and that the statistics
//! and dedup paths match their references.
//!
//! Votes gated and tallied on the host give the scan's own result, bit for bit. A
//! deduped scan equals the raw scan of the tokens it keeps. A scan of only the
//! weighed groups equals the full scan on every backend, from narrower pages. A
//! launch failure's stage and error code decode exactly. The wide turn-group scan
//! is an ignored benchmark, timed per backend.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::super::super::pages_for;

    use super::super::super::scan_plan::fingerprint_segments;

    use super::super::super::tests::gallery_pool;

    use super::*;

    use crate::persistence::content_hash::turn_stream_id;

    use crate::projection::TimelineId;

    use super::super::tests_support::tests::{host_stats, rand_sig, sig};

    /// The launchers' `-(stage * 1000 + cudaError)` codes, decoded exactly —
    /// and a bare `-cudaError`, which names no stage.
    #[test]
    fn a_launch_failure_names_its_stage() {
        assert_eq!(launch_failure(-1002), ("alloc", 2));
        assert_eq!(launch_failure(-2001), ("memset", 1));
        assert_eq!(launch_failure(-3009), ("accum_launch", 9));
        assert_eq!(launch_failure(-4009), ("finalize_launch", 9));
        assert_eq!(launch_failure(-9), ("unstaged", 9));
    }

    /// **The raw votes are the scan's own.** Gated and tallied on the host,
    /// [`GalleryArena::scan_votes`]' rows give exactly what
    /// [`GalleryArena::scan_weighted`] returns, bit for bit, over several
    /// segments and probes. Skips when no CUDA device is present.
    #[test]
    fn the_raw_votes_tally_to_the_weighted_scan() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..5).map(|t| sig(0x1000 + t)).collect();
        let b: Vec<WideQSig> = (0..40).map(|t| sig(0x3000 + t)).collect();
        let c: Vec<WideQSig> = (0..7).map(|t| sig(0x5000 + t)).collect();
        fn window(sid: u64, turn: &Vec<WideQSig>, case: usize) -> PagedWindow<'_> {
            PagedWindow {
                sid: turn_stream_id(sid, 0),
                fingerprint: sid,
                turn: turn.into(),
                start: 0,
                end: turn.len(),
                case,
            }
        }
        let segs = vec![
            PagedSegment {
                windows: vec![window(1, &a, 0), window(2, &b, 1)],
                n_cases: 2,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            },
            PagedSegment {
                windows: vec![window(3, &c, 0)],
                n_cases: 1,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            },
        ];
        let p0 = vec![sig(0x1001), sig(0x3000 + 30), sig(0xBEEF)];
        let p1 = vec![sig(0x5003), sig(0x1004)];
        let probes = [p0.as_slice(), p1.as_slice()];
        let weights = [0.5f32, 1.0, 2.0];

        let raw = arena.scan_votes(&segs, &probes).unwrap();
        assert_eq!(raw.per_probe, vec![3, 2]);
        let tallied = needle_tally_segments(
            &raw.case,
            &raw.vote,
            &raw.per_probe,
            &[0, 2, 3],
            raw.n_groups,
            raw.n_segments,
            3,
            raw.k,
            &[&weights, &weights],
            &[VoteLaw::SINGLE, VoteLaw::SINGLE],
        );
        let weighted = arena.scan_weighted(&segs, &probes, &weights).unwrap();
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&tallied), bits(&weighted));
        assert!(
            weighted.iter().flatten().any(|&v| v != 0.0),
            "nothing scored"
        );
    }

    /// **A deduped scan is the raw scan of the tokens it keeps, bit for bit.**
    /// Two turns of near duplicates — three base signatures each, each token
    /// one of them, every fifth with a bit flipped — scanned keeping the first of each
    /// code, against a fresh arena scanning every token of windows cut to the
    /// kept runs. The deduped turns hold fewer pages; a reader keeping every
    /// token then widens them, and scans as the raw arena does.
    #[test]
    fn a_deduped_scan_is_the_raw_scan_of_its_kept_tokens() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let mut rng = 0xDEDD_0B5E_u64;
        let bases: Vec<WideQSig> = (0..6).map(|_| rand_sig(&mut rng)).collect();
        // Every fifth token one bit off its base; the rest repeat it exactly,
        // so whatever the codes, a turn holds tokens sharing one.
        let near = |i: usize, from: usize| {
            let mut s = bases[from + i % 3].clone();
            if i.is_multiple_of(5) {
                s.words[i % 24] ^= 1 << (i % 64);
            }
            s
        };
        let leak = |n: usize, from: usize| -> &'static [WideQSig] {
            Box::leak((0..n).map(|i| near(i, from)).collect::<Box<[_]>>())
        };
        let (a, b) = (leak(90, 0), leak(50, 3));
        let turns: [(u64, &'static [WideQSig]); 2] = [(81, a), (82, b)];
        let segment = |keep: Keep, windows: Vec<PagedWindow<'static>>| PagedSegment {
            windows,
            n_cases: 2,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep,
            law: VoteLaw::SINGLE,
        };
        let whole = |a: &'static [WideQSig], b: &'static [WideQSig]| {
            [(81u64, a, 0usize), (82, b, 1)]
                .into_iter()
                .map(|(sid, t, case)| PagedWindow {
                    sid: turn_stream_id(sid, 0),
                    fingerprint: sid,
                    turn: t.into(),
                    start: 0,
                    end: t.len(),
                    case,
                })
                .collect::<Vec<_>>()
        };
        let probe: Vec<WideQSig> = (0..12).map(|i| near(i * 7, 3 * (i % 2))).collect();
        let weights: &[f32] = &[1.0, 1.0, 1.0];
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
                .collect::<Vec<_>>()
        };

        let deduped = GalleryArena::new(&device, 24, 3).unwrap();
        let one = Keep::FirstPerCode(1);
        let got = deduped
            .scan_weighted(&[segment(one, whole(a, b))], &[probe.as_slice()], weights)
            .unwrap();
        assert!(got[0].iter().any(|&v| v != 0.0), "nothing scored");
        let book = deduped.dedup.book().expect("the scan trained a codebook");

        // The raw scan of each turn's kept runs, on an arena that holds every
        // token.
        let mut cut = Vec::new();
        let mut kept_pages = 0;
        for (case, &(sid, t)) in turns.iter().enumerate() {
            let mask = deduped
                .dedup
                .cached(turn_stream_id(sid, 0), sid, one, book.id())
                .expect("the upload computed its mask");
            assert!(
                mask.kept() < t.len(),
                "turn {sid} dropped its near duplicates"
            );
            kept_pages += pages_for(mask.kept()) as u64;
            let mut s = 0;
            while s < t.len() {
                if !mask.keeps(s) {
                    s += 1;
                    continue;
                }
                let mut e = s;
                while e < t.len() && mask.keeps(e) {
                    e += 1;
                }
                cut.push(PagedWindow {
                    sid: turn_stream_id(sid, 0),
                    fingerprint: sid,
                    turn: t.into(),
                    start: s,
                    end: e,
                    case,
                });
                s = e;
            }
        }
        let raw = GalleryArena::new(&device, 24, 3).unwrap();
        let want = raw
            .scan_weighted(&[segment(Keep::All, cut)], &[probe.as_slice()], weights)
            .unwrap();
        assert_eq!(bits(&got), bits(&want));
        let pages =
            |arena: &GalleryArena| arena.resident_runs().iter().map(|r| r.bytes).sum::<u64>();
        assert_eq!(
            pages(&deduped),
            kept_pages * 6144,
            "pages hold the kept tokens"
        );
        assert!(kept_pages < 5, "90 + 50 tokens are 5 pages raw");

        // A reader keeping every token widens the turns, and scans as raw does.
        let every = || [segment(Keep::All, whole(a, b))];
        let full = deduped
            .scan_weighted(&every(), &[probe.as_slice()], weights)
            .unwrap();
        let raw_full = raw
            .scan_weighted(&every(), &[probe.as_slice()], weights)
            .unwrap();
        assert_eq!(bits(&full), bits(&raw_full));
        assert_eq!(pages(&deduped), 5 * 6144);
        // And a deduped reader of turns held whole still scans only its own
        // kept tokens.
        let again = deduped
            .scan_weighted(&[segment(one, whole(a, b))], &[probe.as_slice()], weights)
            .unwrap();
        assert_eq!(
            bits(&again),
            bits(&got),
            "a wider holding changes no reader"
        );
    }

    /// **A scan holds what it reads for its reader.** A scan masked to one
    /// conversation holds the turn for it; the same segments read shared are a
    /// different index (the holder is in the fingerprint), and the read widens
    /// the turn to shared.
    #[test]
    fn a_scan_holds_its_turns_for_its_reader() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xD00 + t)).collect();
        let own = Holder::Conversation(TimelineId::from_raw(12).unwrap());
        let segment = |holder| {
            vec![PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(12, 0),
                    fingerprint: 1,
                    turn: (&a).into(),
                    start: 0,
                    end: 20,
                    case: 0,
                }],
                n_cases: 1,
                holder,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            }]
        };
        let holder_now = || arena.resident_runs()[0].holder;
        let probe = vec![sig(0xD00 + 4)];
        arena
            .scan_weighted(&segment(own), &[probe.as_slice()], &[])
            .unwrap();
        assert_eq!(holder_now(), own);
        assert_ne!(
            fingerprint_segments(&segment(own)),
            fingerprint_segments(&segment(Holder::Shared)),
        );
        arena
            .scan_weighted(&segment(Holder::Shared), &[probe.as_slice()], &[])
            .unwrap();
        assert_eq!(holder_now(), Holder::Shared);
    }

    /// **A scan of only the weighed groups is the full scan, bit for bit.**
    /// Two files' content weighed `[0, 1, 1]` and a calibration turn weighed
    /// `[0, 0, 1]`, one launch — on pages holding only those groups (the
    /// calibration's group 1 read from the zero page), against an arena whose
    /// pages hold all three. Every backend the card has must give identical
    /// votes, and the narrow pages are a third and two thirds smaller.
    #[test]
    fn a_scan_of_the_weighed_groups_is_the_full_scan() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let mut rng = 0x6A1E_5EED_u64;
        let a: Vec<WideQSig> = (0..70).map(|_| rand_sig(&mut rng)).collect();
        let b: Vec<WideQSig> = (0..33).map(|_| rand_sig(&mut rng)).collect();
        let c: Vec<WideQSig> = (0..9).map(|_| rand_sig(&mut rng)).collect();
        let content = GroupSet::default().with(1).with(2);
        let calibration = GroupSet::default().with(2);
        fn w(sid: u64, turn: &[WideQSig], case: usize) -> PagedWindow<'_> {
            PagedWindow {
                sid: turn_stream_id(sid, 0),
                fingerprint: sid,
                turn: turn.into(),
                start: 0,
                end: turn.len(),
                case,
            }
        }
        let segments = |content: GroupSet, calibration: GroupSet| {
            vec![
                PagedSegment {
                    windows: vec![w(91, &a, 0), w(92, &b, 1)],
                    n_cases: 2,
                    holder: Holder::Shared,
                    groups: content,
                    keep: Keep::All,
                    law: VoteLaw::SINGLE,
                },
                PagedSegment {
                    windows: vec![w(93, &c, 0)],
                    n_cases: 1,
                    holder: Holder::Shared,
                    groups: calibration,
                    keep: Keep::All,
                    law: VoteLaw::SINGLE,
                },
            ]
        };
        let weights: [&[f32]; 2] = [&[0.0, 1.0, 1.0], &[0.0, 0.0, 1.0]];
        let probe: Vec<WideQSig> = (0..20)
            .map(|i| {
                if i % 4 == 0 {
                    a[i * 3].clone()
                } else {
                    rand_sig(&mut rng)
                }
            })
            .collect();
        let narrow = GalleryArena::new(&device, 24, 3).unwrap();
        let full = GalleryArena::new(&device, 24, 3).unwrap();
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
                .collect::<Vec<_>>()
        };
        let want = full
            .scan_segment_weighted(
                &segments(GroupSet::all(3), GroupSet::all(3)),
                &[probe.as_slice()],
                &weights,
            )
            .unwrap();
        assert!(want[0].iter().any(|&v| v != 0.0), "nothing scored");
        let narrow_segments = segments(content, calibration);
        let got = narrow
            .scan_segment_weighted(&narrow_segments, &[probe.as_slice()], &weights)
            .unwrap();
        assert_eq!(bits(&got), bits(&want), "auto backend");
        let (has_bmma, has_imma) = narrow.tensor_caps();
        let forced: Vec<Option<PagedBackend>> = [
            Some(PagedBackend::Scalar),
            has_bmma.then_some(PagedBackend::Bmma),
            has_imma.then_some(PagedBackend::Imma),
        ]
        .into_iter()
        .flatten()
        .map(Some)
        .collect();
        for force in forced {
            let got = narrow
                .scan_weighted_impl(&narrow_segments, &[probe.as_slice()], &weights, force)
                .unwrap();
            assert_eq!(bits(&got), bits(&want), "{force:?}");
        }
        let bytes_of = |arena: &GalleryArena, sid: u64| {
            arena
                .resident_runs()
                .into_iter()
                .find(|r| r.sid == turn_stream_id(sid, 0))
                .map(|r| r.bytes)
        };
        // 70 tokens are 3 pages: 3 × 4 KiB holding groups 1–2, 3 × 6 KiB all.
        assert_eq!(bytes_of(&narrow, 91), Some(3 * 4096));
        assert_eq!(bytes_of(&full, 91), Some(3 * 6144));
        assert_eq!(
            bytes_of(&narrow, 93),
            Some(2048),
            "the calibration holds group 2"
        );
        // A read weighing all three widens the narrow turn, and its scan agrees.
        let wide = narrow
            .scan_segment_weighted(
                &segments(GroupSet::all(3), GroupSet::all(3)),
                &[probe.as_slice()],
                &weights,
            )
            .unwrap();
        assert_eq!(bits(&wide), bits(&want));
        assert_eq!(bytes_of(&narrow, 93), Some(6144), "widened to every group");
    }

    /// **The statistics mode writes, on every backend, what the definitions
    /// give.** Five segments — empty cases, cases supplied in descending
    /// order, an out-of-range case that is not scanned — against probes of 1,
    /// 9 and 70 tokens: every (token, group, segment) row equals the host's
    /// statistics, its cases global; the segments' `n` are the tokens each
    /// scanned; and the auto ladder, the scalar kernel and each tensor backend
    /// the card has write identical rows.
    #[test]
    fn the_statistics_mode_is_the_host_reference_on_every_backend() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let (has_bmma, has_imma) = arena.tensor_caps();
        let mut rng = 0x0005_7A75_u64;
        let lens = [1usize, 7, 32, 33, 65, 100];
        let turns: Vec<Vec<WideQSig>> = (0..18)
            .map(|i| {
                (0..lens[i % lens.len()])
                    .map(|_| rand_sig(&mut rng))
                    .collect()
            })
            .collect();
        let segs: Vec<PagedSegment> = (0..5usize)
            .map(|s| {
                let n_cases = 2 + s;
                let mut windows: Vec<PagedWindow> = (0..3usize)
                    .map(|w| {
                        let ti = (s * 3 + w) % turns.len();
                        PagedWindow {
                            sid: turn_stream_id(700 + s as u64, ti as u32),
                            fingerprint: (s * 100 + ti) as u64,
                            turn: (&turns[ti]).into(),
                            start: 0,
                            end: turns[ti].len(),
                            // Descending, with gaps: case 0 of a wide segment
                            // stays empty.
                            case: (n_cases - 1).saturating_sub(w * 2 % n_cases),
                        }
                    })
                    .collect();
                windows.push(PagedWindow {
                    sid: turn_stream_id(700 + s as u64, 99),
                    fingerprint: 99,
                    turn: (&turns[0]).into(),
                    start: 0,
                    end: 1,
                    case: n_cases + 1,
                });
                PagedSegment {
                    windows,
                    n_cases,
                    holder: Holder::Shared,
                    groups: GroupSet::all(3),
                    keep: Keep::All,
                    law: VoteLaw::SINGLE,
                }
            })
            .collect();
        let probes: Vec<Vec<WideQSig>> = [1usize, 9, 70]
            .iter()
            .map(|&n| (0..n).map(|_| rand_sig(&mut rng)).collect())
            .collect();
        let probe_refs: Vec<&[WideQSig]> = probes.iter().map(Vec::as_slice).collect();

        let auto = arena.scan_stats(&segs, &probe_refs).unwrap();
        assert_eq!(auto.per_probe, vec![1, 9, 70]);
        assert_eq!(auto.groups, vec![0, 1, 2]);
        let scanned: Vec<u64> = segs
            .iter()
            .map(|s| {
                s.windows
                    .iter()
                    .filter(|w| w.case < s.n_cases)
                    .map(|w| (w.end - w.start) as u64)
                    .sum()
            })
            .collect();
        assert_eq!(auto.seg_tokens, scanned);
        let all: Vec<&WideQSig> = probes.iter().flatten().collect();
        for (t, q) in all.iter().enumerate() {
            for g in 0..3 {
                let mut base = 0i32;
                for (s, seg) in segs.iter().enumerate() {
                    assert_eq!(
                        auto.row(t, g, s),
                        Some(&host_stats(q, g, seg, base)),
                        "token {t} group {g} segment {s}"
                    );
                    base += seg.n_cases as i32;
                }
            }
        }
        let on = |b| arena.scan_stats_on(&segs, &probe_refs, Some(b)).unwrap();
        assert_eq!(on(PagedBackend::Scalar), auto, "scalar vs auto");
        if has_bmma {
            assert_eq!(on(PagedBackend::Bmma), auto, "BMMA vs auto");
        }
        if has_imma {
            assert_eq!(on(PagedBackend::Imma), auto, "IMMA vs auto");
        }
    }

    /// **The live reprojection's widest scan, timed per backend.** The shape a
    /// turn group took in zend on `.5` (2026-10-17): 8.08 M index tokens over 9
    /// segments and 1,041 cases, scored for 66 fresh probe tokens — 16–23 ms a
    /// launch, the largest single cost in a reprojection — and for a seal's
    /// whole turn. Fresh probes each repetition, so the row cache never answers.
    /// Run with the card otherwise idle:
    ///
    /// `cargo test --release -p candle-conversation --lib bench_wide_turn_group_scan -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark: needs an idle card"]
    fn bench_wide_turn_group_scan() {
        let device = Device::new_cuda(0).expect("a CUDA card");
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let mut rng = 0xBE_4C11_0017_u64;
        // ~1 M distinct tokens: more than L2, as the resident corpus is.
        const TURN_TOKENS: usize = 512;
        let turns: Vec<Vec<WideQSig>> = (0..2_048)
            .map(|_| (0..TURN_TOKENS).map(|_| rand_sig(&mut rng)).collect())
            .collect();
        let top_8 = VoteLaw {
            top_k: 8,
            power: 0.5,
        };
        // (shape, segments, cases a segment, windows a case, law): the turn
        // group's 9 × 116, at one vote; and the code-reading group's one
        // competition of 6,678 files, a window each, at one vote and at the
        // scoring lab's top-8.
        let shapes = [
            (
                "9 x 116, one vote",
                9usize,
                116usize,
                15usize,
                VoteLaw::SINGLE,
            ),
            ("1 x 6678, one vote", 1, 6_678, 1, VoteLaw::SINGLE),
            ("1 x 6678, top-8", 1, 6_678, 1, top_8),
        ];
        let (has_bmma, has_imma) = arena.tensor_caps();
        let mut backends: Vec<&str> = Vec::new();
        if has_bmma {
            backends.push("bmma");
        }
        if has_imma {
            backends.push("imma");
        }
        backends.push("auto");
        for (shape, n_segments, n_cases, per_case, law) in shapes {
            let mut next = 0usize;
            let segs: Vec<PagedSegment> = (0..n_segments)
                .map(|_| {
                    let mut windows = Vec::new();
                    for case in 0..n_cases {
                        for _ in 0..per_case {
                            let ti = next % turns.len();
                            next += 1;
                            windows.push(PagedWindow {
                                sid: turn_stream_id(7_000, ti as u32),
                                fingerprint: ti as u64,
                                turn: (&turns[ti]).into(),
                                start: 0,
                                end: TURN_TOKENS,
                                case,
                            });
                        }
                    }
                    PagedSegment {
                        windows,
                        n_cases,
                        holder: Holder::Shared,
                        groups: GroupSet::all(3),
                        keep: Keep::All,
                        law,
                    }
                })
                .collect();
            let tokens = n_segments * n_cases * per_case * TURN_TOKENS;
            let scan = |name: &str, probe: &[WideQSig]| match name {
                "bmma" => arena.scan_weighted_bmma(&segs, &[probe], &[]),
                "imma" => arena.scan_weighted_imma(&segs, &[probe], &[]),
                _ => arena.scan_weighted(&segs, &[probe], &[]),
            };
            println!("{shape}: {tokens} index tokens");
            // A live reprojection's launch carries the few tokens decoded since
            // the last (`ProbeRows`): 1–64 is its common case.
            for probe_tokens in [1usize, 6, 16, 32, 66, 256, 1_024] {
                for name in &backends {
                    // The first scan builds the index and uploads the pages.
                    let warm: Vec<WideQSig> =
                        (0..probe_tokens).map(|_| rand_sig(&mut rng)).collect();
                    scan(name, &warm).unwrap();
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        let probe: Vec<WideQSig> =
                            (0..probe_tokens).map(|_| rand_sig(&mut rng)).collect();
                        let t = Instant::now();
                        scan(name, &probe).unwrap();
                        best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    }
                    println!("  {name:>5} probe {probe_tokens:>5}: {best:8.2} ms");
                }
            }
        }
    }
}
