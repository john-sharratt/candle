//! Tests that the paged scan matches the contiguous scan, and that every backend
//! agrees with the others bit for bit.
//!
//! Probes batched into one launch vote as they do alone, and a rescan that reuses
//! earlier rows votes as a fresh scan does. The paged scan equals the contiguous
//! scan bit for bit on the scalar oracle, and each tensor backend agrees with it
//! on a corpus that covers tile tails, descending case ids and out-of-range cases.
//! A wide gallery scanned in probe chunks gives the same result as one pass.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::super::super::tests::gallery_pool;

    use super::*;

    use crate::persistence::content_hash::turn_stream_id;

    use crate::provenance::gpu::{BatchedGpuGallery, SegmentInput};

    use crate::provenance::score_slots_law;

    use super::super::tests_support::tests::{rand_sig, sig};

    /// **Probes scanned in one launch vote exactly as they do scanned alone.**
    /// The kernel writes one result per probe token and the tally splits them
    /// back by each request's token count, so batching a turn's tail and
    /// question windows into one launch — one gallery pass, one sync — must not
    /// move a single bit of either probe's votes.
    #[test]
    fn probes_batched_into_one_launch_vote_as_they_do_alone() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..40).map(|t| sig(0x5000 + t)).collect();
        let b: Vec<WideQSig> = (0..9).map(|t| sig(0x6000 + t)).collect();
        let segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(7, 0),
                        fingerprint: 70,
                        turn: (&a).into(),
                        start: 0,
                        end: 40,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(7, 1),
                        fingerprint: 71,
                        turn: (&b).into(),
                        start: 0,
                        end: 9,
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
                    sid: turn_stream_id(8, 0),
                    fingerprint: 80,
                    turn: (&a).into(),
                    start: 5,
                    end: 30,
                    case: 0,
                }],
                n_cases: 1,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            },
        ];
        let tail = vec![
            sig(0x5000 + 3),
            sig(0xBEEF),
            sig(0x6000 + 2),
            sig(0x5000 + 31),
        ];
        let question = vec![sig(0x6000 + 7), sig(0xF00D)];
        let weights = [0.5f32, 1.0, 2.0];

        let alone_tail = arena
            .scan_weighted(&segs, &[tail.as_slice()], &weights)
            .unwrap();
        let alone_q = arena
            .scan_weighted(&segs, &[question.as_slice()], &weights)
            .unwrap();
        let both = arena
            .scan_weighted(&segs, &[tail.as_slice(), question.as_slice()], &weights)
            .unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(both.len(), 2);
        assert_eq!(
            bits(&both[0]),
            bits(&alone_tail[0]),
            "the tail's votes moved"
        );
        assert_eq!(
            bits(&both[1]),
            bits(&alone_q[0]),
            "the question's votes moved"
        );
        assert!(
            both[0].iter().any(|&v| v != 0.0),
            "the tail scored nothing — the comparison proves nothing"
        );
    }

    /// **A rescan that reuses earlier rows votes exactly as a fresh scan.**
    /// The second probe shares most of its tokens with the first — a decode's
    /// next reprojection — so only its new tokens reach the kernel, and every
    /// vote must still match an arena that never saw the first probe.
    #[test]
    fn a_rescan_reusing_rows_votes_as_a_fresh_scan() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let a: Vec<WideQSig> = (0..40).map(|t| sig(0x7000 + t)).collect();
        let b: Vec<WideQSig> = (0..12).map(|t| sig(0x8000 + t)).collect();
        let segs = || {
            vec![
                PagedSegment {
                    windows: vec![PagedWindow {
                        sid: turn_stream_id(9, 0),
                        fingerprint: 90,
                        turn: (&a).into(),
                        start: 0,
                        end: 40,
                        case: 0,
                    }],
                    n_cases: 1,
                    holder: Holder::Shared,
                    groups: GroupSet::all(3),
                    keep: Keep::All,
                    law: VoteLaw::SINGLE,
                },
                PagedSegment {
                    windows: vec![
                        PagedWindow {
                            sid: turn_stream_id(10, 0),
                            fingerprint: 100,
                            turn: (&b).into(),
                            start: 0,
                            end: 6,
                            case: 0,
                        },
                        PagedWindow {
                            sid: turn_stream_id(10, 0),
                            fingerprint: 100,
                            turn: (&b).into(),
                            start: 6,
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
            ]
        };
        let first: Vec<WideQSig> = [3, 9, 17, 0x8000 + 2, 25]
            .iter()
            .map(|&t| sig(if t < 0x8000 { 0x7000 + t } else { t }))
            .collect();
        let mut second: Vec<WideQSig> = first[1..].to_vec();
        second.extend([
            sig(0x7000 + 33),
            sig(0x8000 + 9),
            sig(0xCAFE),
            sig(0x7000 + 9),
        ]);
        let question = vec![sig(0x8000 + 2), sig(0x7000 + 3)];
        let weights = [1.0f32, 0.5, 2.0];

        let _pool = gallery_pool();
        let warm = GalleryArena::new(&device, 24, 3).unwrap();
        warm.scan_weighted(&segs(), &[first.as_slice()], &weights)
            .unwrap();
        let reused = warm
            .scan_weighted(&segs(), &[second.as_slice(), question.as_slice()], &weights)
            .unwrap();
        let fresh = GalleryArena::new(&device, 24, 3)
            .unwrap()
            .scan_weighted(&segs(), &[second.as_slice(), question.as_slice()], &weights)
            .unwrap();
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&reused), bits(&fresh));
        assert!(
            reused[0].iter().any(|&v| v != 0.0),
            "the probe scored nothing — the comparison proves nothing"
        );
    }

    /// The paged scan must be **bit-identical** to the contiguous
    /// `BatchedGpuGallery` scan over the same windows: identical bytes, identical
    /// kernel math — only the physical layout differs. Exercises multi-page turns,
    /// seam sub-windows (two windows of one turn into different cases), several
    /// files, and non-uniform group weights. Skips when no CUDA device is present.
    #[test]
    fn paged_matches_contiguous_bit_identical() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        // Turns (whole windows), owned so both paths can borrow them.
        let a0: Vec<WideQSig> = (0..3).map(|t| sig(0x1000 + t)).collect();
        let a1: Vec<WideQSig> = (0..2).map(|t| sig(0x2000 + t)).collect();
        let b0: Vec<WideQSig> = (0..40).map(|t| sig(0x3000 + t)).collect(); // 2 pages
        let c0: Vec<WideQSig> = (0..33).map(|t| sig(0x4000 + t)).collect(); // partial 2nd page

        // ── Paged segments (reference the WHOLE turn + [start,end)) ──
        let paged_segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(1, 0),
                        fingerprint: 10,
                        turn: (&a0).into(),
                        start: 0,
                        end: 3,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(1, 1),
                        fingerprint: 11,
                        turn: (&a1).into(),
                        start: 0,
                        end: 2,
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
                // Two sub-windows of ONE multi-page turn → different cases (a seam).
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(2, 0),
                        fingerprint: 20,
                        turn: (&b0).into(),
                        start: 0,
                        end: 20,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(2, 0),
                        fingerprint: 20,
                        turn: (&b0).into(),
                        start: 20,
                        end: 40,
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
                    sid: turn_stream_id(3, 0),
                    fingerprint: 30,
                    turn: (&c0).into(),
                    start: 0,
                    end: 33,
                    case: 0,
                }],
                n_cases: 1,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            },
        ];

        // ── Contiguous segments (sub-slices packed directly) ──
        let a0s = a0.as_slice();
        let a1s = a1.as_slice();
        let contig_segs = vec![
            SegmentInput {
                windows: vec![&a0s[0..3], &a1s[0..2]],
                window_case: vec![0, 1],
                n_cases: 2,
            },
            SegmentInput {
                windows: vec![&b0[0..20], &b0[20..40]],
                window_case: vec![0, 1],
                n_cases: 2,
            },
            SegmentInput {
                windows: vec![&c0[0..33]],
                window_case: vec![0],
                n_cases: 1,
            },
        ];

        let probe = vec![sig(0x1000), sig(0x3000 + 25), sig(0xBEEF), sig(0x4000 + 10)];
        let weights = [0.25f32, 1.0, 3.0];

        // Scalar-paged vs scalar-contiguous: the strict oracle (same kernel, only
        // the physical layout differs — must be bit-for-bit equal).
        let contiguous = BatchedGpuGallery::from_segments(&contig_segs)
            .unwrap()
            .scan_weighted(&device, &[probe.as_slice()], &weights)
            .unwrap();
        let paged = arena
            .scan_weighted_scalar(&paged_segs, &[probe.as_slice()], &weights)
            .unwrap();

        assert_eq!(paged.len(), contiguous.len());
        assert_eq!(paged[0].len(), contiguous[0].len(), "global case count");
        for (i, (p, c)) in paged[0].iter().zip(&contiguous[0]).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "case {i}: paged {p} vs contiguous {c} must be bit-identical"
            );
        }

        // Unweighted must also match (weights = &[] → uniform).
        let contiguous_u = BatchedGpuGallery::from_segments(&contig_segs)
            .unwrap()
            .scan(&device, &[probe.as_slice()])
            .unwrap();
        let paged_u = arena
            .scan_weighted_scalar(&paged_segs, &[probe.as_slice()], &[])
            .unwrap();
        for (i, (p, c)) in paged_u[0].iter().zip(&contiguous_u[0]).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "unweighted case {i} must be bit-identical"
            );
        }

        // The auto backend (BMMA on this hardware) must agree with the scalar
        // oracle on the same fixture — full bit equality (integer statistics are
        // identical by construction; the float finalize is the shared bdp_vote).
        let auto = arena
            .scan_weighted(&paged_segs, &[probe.as_slice()], &weights)
            .unwrap();
        for (i, (a, c)) in auto[0].iter().zip(&contiguous[0]).enumerate() {
            assert_eq!(
                a.to_bits(),
                c.to_bits(),
                "case {i}: auto backend {a} vs scalar {c} must be bit-identical"
            );
        }
    }

    /// **A segment of hundreds of cases, its agreements tied across them,
    /// votes on every backend as the scalar kernel's one ascending pass
    /// does.** The tensor finalize reads a segment's cases a warp at a time,
    /// each lane every 32nd, and merges the lanes' best cases; the 700 cases
    /// here are five turns repeated, so every agreement is shared by ~140
    /// cases spread over every lane, and only the pass's tie order — the lowest
    /// case first — picks the leader, the runner-up and the top-8. Bit for bit,
    /// at one vote and at top-8.
    #[test]
    fn a_wide_segment_with_tied_cases_votes_as_the_scalar_pass() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let (has_bmma, has_imma) = arena.tensor_caps();
        let mut rng = 0x71ED_CA5E_u64;
        let mut turns: Vec<Vec<WideQSig>> = (0..5)
            .map(|i| (0..(3 + 9 * i)).map(|_| rand_sig(&mut rng)).collect())
            .collect();
        // A probe whose tokens one case alone holds, deep in a high lane's
        // stride: its leader is case 613, not one of the first lanes'.
        let exact: Vec<WideQSig> = (0..4).map(|_| rand_sig(&mut rng)).collect();
        turns.push(exact.clone());
        const CASES: usize = 700;
        const PLANTED: usize = 613;
        let top_8 = VoteLaw {
            top_k: 8,
            power: 0.5,
        };
        let probe: Vec<WideQSig> = (0..41).map(|_| rand_sig(&mut rng)).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for law in [VoteLaw::SINGLE, top_8] {
            let seg = PagedSegment {
                windows: (0..CASES)
                    .map(|case| {
                        let ti = if case == PLANTED { 5 } else { (case * 7) % 5 };
                        PagedWindow {
                            sid: turn_stream_id(9_100, ti as u32),
                            fingerprint: ti as u64,
                            turn: (&turns[ti]).into(),
                            start: 0,
                            end: turns[ti].len(),
                            case,
                        }
                    })
                    .collect(),
                n_cases: CASES,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law,
            };
            let segs = [seg];
            let probes = [probe.as_slice(), exact.as_slice()];
            let weights = [0.5f32, 1.0, 2.0];
            let scalar = arena
                .scan_weighted_scalar(&segs, &probes, &weights)
                .unwrap();
            assert!(
                scalar[1].iter().any(|&v| v > 0.0),
                "{law:?}: the exact probe scores"
            );
            if has_bmma {
                let bmma = arena.scan_weighted_bmma(&segs, &probes, &weights).unwrap();
                for p in 0..probes.len() {
                    assert_eq!(bits(&bmma[p]), bits(&scalar[p]), "{law:?} probe {p}: BMMA");
                }
            }
            if has_imma {
                let imma = arena.scan_weighted_imma(&segs, &probes, &weights).unwrap();
                for p in 0..probes.len() {
                    assert_eq!(bits(&imma[p]), bits(&scalar[p]), "{law:?} probe {p}: IMMA");
                }
            }
            // The raw rows too: which cases each token voted for, in order.
            let raw = arena.scan_votes(&segs, &probes).unwrap();
            assert!(
                raw.case.contains(&(PLANTED as i32)),
                "{law:?}: the planted case leads its probe"
            );
        }
    }

    /// Adversarial tensor-vs-scalar parity over a large randomized corpus that
    /// hits every structural edge at once: 1-token exchanges, case-id GAPS
    /// (empty cases), windows supplied in DESCENDING case order (exercises the
    /// build_index sort the tensor backends' dense-rank relies on), out-of-range
    /// cases (dropped), seam sub-windows of one multi-page turn split across
    /// cases, segments that straddle 64-token chunk boundaries, and probe/token
    /// counts that are not multiples of the 8/32/64 tile shapes — up to probes
    /// of several query tiles. The backends' integer statistics are identical by
    /// construction and the float finalize is shared, so the votes must be
    /// bit-for-bit equal. Each tensor backend the device has is checked: b1 BMMA
    /// on Ada, INT8 IMMA on Ampere and later — Blackwell has only the latter.
    #[test]
    fn tensor_backends_match_scalar_on_adversarial_corpus() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let (has_bmma, has_imma) = arena.tensor_caps();
        assert!(
            has_bmma || has_imma,
            "a CUDA device here has at least one tensor backend"
        );

        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        // Turn lengths spanning page boundaries and tile tails.
        let lens = [1usize, 3, 7, 8, 31, 32, 33, 40, 63, 64, 65, 100, 129, 200];
        let turns: Vec<Vec<WideQSig>> = (0..40)
            .map(|i| {
                let n = lens[i % lens.len()];
                (0..n).map(|_| rand_sig(&mut rng)).collect()
            })
            .collect();

        let mut segs: Vec<PagedSegment> = Vec::new();
        let mut next_turn = 0usize;
        for si in 0..14usize {
            let n_cases = 2 + (si % 7); // 2..8 cases per segment
            let mut windows = Vec::new();
            // 2-4 turns per segment.
            for wi in 0..(2 + si % 3) {
                let ti = (next_turn + wi) % turns.len();
                let t = &turns[ti];
                // DESCENDING case ids (build_index must sort), with deliberate
                // gaps (cases that never receive a window stay empty).
                let case = (n_cases - 1).saturating_sub(wi * 2 % n_cases);
                if t.len() > 8 {
                    // Seam split: two sub-windows of one turn, different cases.
                    let mid = t.len() / 2;
                    windows.push(PagedWindow {
                        sid: turn_stream_id(500 + si as u64, ti as u32),
                        fingerprint: (si * 100 + ti) as u64,
                        turn: t.into(),
                        start: 0,
                        end: mid,
                        case,
                    });
                    windows.push(PagedWindow {
                        sid: turn_stream_id(500 + si as u64, ti as u32),
                        fingerprint: (si * 100 + ti) as u64,
                        turn: t.into(),
                        start: mid,
                        end: t.len(),
                        case: case.saturating_sub(1),
                    });
                } else {
                    windows.push(PagedWindow {
                        sid: turn_stream_id(500 + si as u64, ti as u32),
                        fingerprint: (si * 100 + ti) as u64,
                        turn: t.into(),
                        start: 0,
                        end: t.len(),
                        case,
                    });
                }
            }
            // An out-of-range case that must be dropped by both backends.
            let tdrop = &turns[next_turn % turns.len()];
            windows.push(PagedWindow {
                sid: turn_stream_id(500 + si as u64, (next_turn % turns.len()) as u32),
                fingerprint: (si * 100 + next_turn % turns.len()) as u64,
                turn: tdrop.into(),
                start: 0,
                end: tdrop.len().min(4),
                case: n_cases + 3,
            });
            next_turn += 3;
            segs.push(PagedSegment {
                windows,
                n_cases,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
                law: VoteLaw::SINGLE,
            });
        }

        // Probes exercising tile tails: 1, 9 and 100 query tokens, and 161 —
        // six query tiles, the last holding one row. A tile of 16 queries or
        // fewer runs one query block, more runs two: 16, 17 and 20 sit on
        // either side of that.
        let probes_owned: Vec<Vec<WideQSig>> = [1usize, 9, 16, 17, 20, 100, 161]
            .iter()
            .map(|&n| (0..n).map(|_| rand_sig(&mut rng)).collect())
            .collect();

        for (pi, probe) in probes_owned.iter().enumerate() {
            for weights in [&[] as &[f32], &[0.25, 1.0, 3.0]] {
                let scalar = arena
                    .scan_weighted_scalar(&segs, &[probe.as_slice()], weights)
                    .unwrap();
                let auto = arena
                    .scan_weighted(&segs, &[probe.as_slice()], weights)
                    .unwrap();
                assert_eq!(scalar[0].len(), auto[0].len(), "probe {pi}: case count");
                for (ci, (s, b)) in scalar[0].iter().zip(&auto[0]).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        b.to_bits(),
                        "probe {pi} case {ci}: scalar {s} vs auto {b} must be bit-identical"
                    );
                }
                if has_bmma {
                    let bmma = arena
                        .scan_weighted_bmma(&segs, &[probe.as_slice()], weights)
                        .unwrap();
                    for (ci, (s, m)) in scalar[0].iter().zip(&bmma[0]).enumerate() {
                        assert_eq!(
                            s.to_bits(),
                            m.to_bits(),
                            "probe {pi} case {ci}: scalar {s} vs BMMA {m} must be bit-identical"
                        );
                    }
                }
                // The INT8 (IMMA) backend — the Blackwell production path.
                if has_imma {
                    let imma = arena
                        .scan_weighted_imma(&segs, &[probe.as_slice()], weights)
                        .unwrap();
                    for (ci, (s, m)) in scalar[0].iter().zip(&imma[0]).enumerate() {
                        assert_eq!(
                            s.to_bits(),
                            m.to_bits(),
                            "probe {pi} case {ci}: scalar {s} vs IMMA {m} must be bit-identical"
                        );
                    }
                }
            }
        }
    }

    /// **Every backend scores a segment's vote law as the host scorer does.**
    /// Three segments in one launch: twelve cases at top-8, power 0.5 (the
    /// code-reading law); five at one vote (a tool catalog's law, in a launch
    /// whose rows are eight wide); two at top-3 (fewer cases than votes). Each
    /// segment's scores match `score_slots_law` over its own windows to
    /// fast-math tolerance, the backends match the scalar kernel bit for bit,
    /// and the one-vote segment scores bit for bit as it does in a launch of
    /// its own. Then the same windows at one vote everywhere: the index built
    /// for the top-k law is not reused for rows of a different width.
    #[test]
    fn every_backend_scores_a_segments_vote_law_as_the_host_does() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let (has_bmma, has_imma) = arena.tensor_caps();
        let mut rng = 0x5EED_70B8_u64;
        let laws = [
            (
                12usize,
                VoteLaw {
                    top_k: 8,
                    power: 0.5,
                },
            ),
            (5, VoteLaw::SINGLE),
            (
                2,
                VoteLaw {
                    top_k: 3,
                    power: 1.0,
                },
            ),
        ];
        // Each segment: one turn of 5..40 tokens a case.
        let turns: Vec<Vec<Vec<WideQSig>>> = laws
            .iter()
            .map(|&(n_cases, _)| {
                (0..n_cases)
                    .map(|c| (0..5 + 7 * c % 36).map(|_| rand_sig(&mut rng)).collect())
                    .collect()
            })
            .collect();
        let segment = |s: usize, law: VoteLaw| PagedSegment {
            windows: turns[s]
                .iter()
                .enumerate()
                .map(|(c, t)| PagedWindow {
                    sid: turn_stream_id(900 + s as u64, c as u32),
                    fingerprint: (s * 1000 + c) as u64,
                    turn: t.into(),
                    start: 0,
                    end: t.len(),
                    case: c,
                })
                .collect(),
            n_cases: laws[s].0,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
            law,
        };
        let segs: Vec<PagedSegment> = (0..laws.len()).map(|s| segment(s, laws[s].1)).collect();
        let probe: Vec<WideQSig> = (0..70).map(|_| rand_sig(&mut rng)).collect();
        let weights = [0.5f32, 1.0, 2.0];

        let scalar = arena
            .scan_weighted_scalar(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        let mut at = 0usize;
        for (s, &(n_cases, law)) in laws.iter().enumerate() {
            let windows: Vec<&[WideQSig]> = turns[s].iter().map(Vec::as_slice).collect();
            let slots: Vec<usize> = (0..n_cases).collect();
            let host = score_slots_law(&probe, &windows, &slots, n_cases, &weights, law);
            let card = &scalar[0][at..at + n_cases];
            assert!(host.iter().any(|&v| v > 0.0), "segment {s} scores");
            for (c, (h, g)) in host.iter().zip(card).enumerate() {
                assert!(
                    (h - g).abs() <= 1e-3 * h.abs().max(1.0),
                    "segment {s} case {c}: host {h} vs card {g}"
                );
            }
            at += n_cases;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        if has_bmma {
            let bmma = arena
                .scan_weighted_bmma(&segs, &[probe.as_slice()], &weights)
                .unwrap();
            assert_eq!(bits(&bmma[0]), bits(&scalar[0]), "BMMA vs scalar");
        }
        if has_imma {
            let imma = arena
                .scan_weighted_imma(&segs, &[probe.as_slice()], &weights)
                .unwrap();
            assert_eq!(bits(&imma[0]), bits(&scalar[0]), "IMMA vs scalar");
        }
        // The one-vote segment alone: rows one wide.
        let alone = arena
            .scan_weighted_scalar(
                &[segment(1, VoteLaw::SINGLE)],
                &[probe.as_slice()],
                &weights,
            )
            .unwrap();
        assert_eq!(bits(&alone[0]), bits(&scalar[0][12..17]));
        // One vote everywhere over the same windows: a different index.
        let single: Vec<PagedSegment> = (0..laws.len())
            .map(|s| segment(s, VoteLaw::SINGLE))
            .collect();
        let once = arena
            .scan_weighted_scalar(&single, &[probe.as_slice()], &weights)
            .unwrap();
        let windows: Vec<&[WideQSig]> = turns[0].iter().map(Vec::as_slice).collect();
        let slots: Vec<usize> = (0..12).collect();
        let host = score_slots_law(&probe, &windows, &slots, 12, &weights, VoteLaw::SINGLE);
        for (h, g) in host.iter().zip(&once[0][..12]) {
            assert!(
                (h - g).abs() <= 1e-3 * h.abs().max(1.0),
                "one vote: {h} vs {g}"
            );
        }
    }

    /// **A gallery too wide for one pass is scanned in probe chunks, bit for
    /// bit.** The tensor launchers bound their accumulators to
    /// `BDP_SCRATCH_BYTES` (`bdp_chunk.cuh`): at 18,000 cases over three groups
    /// a probe token needs 1.08 MB, so 96 MiB holds 93 of this probe's 120 and
    /// the scan runs in two passes. The single-pass scalar kernel is the
    /// reference.
    #[test]
    fn a_wide_gallery_is_scanned_in_probe_chunks_bit_identically() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let (_, has_imma) = arena.tensor_caps();
        if !has_imma {
            return;
        }
        let mut rng = 0x5E_ED0F_C4A1_u64;
        let turns: Vec<Vec<WideQSig>> = (0..64)
            .map(|_| (0..8).map(|_| rand_sig(&mut rng)).collect())
            .collect();
        const SEGMENTS: usize = 4_500;
        const CASES: usize = 4;
        let segs: Vec<PagedSegment> = (0..SEGMENTS)
            .map(|si| {
                let windows = (0..CASES)
                    .map(|case| {
                        let ti = (si * CASES + case) % turns.len();
                        let start = (si + case) % 6;
                        PagedWindow {
                            sid: turn_stream_id(9_000, ti as u32),
                            fingerprint: ti as u64,
                            turn: (&turns[ti]).into(),
                            start,
                            end: start + 2,
                            case,
                        }
                    })
                    .collect();
                PagedSegment {
                    windows,
                    n_cases: CASES,
                    holder: Holder::Shared,
                    groups: GroupSet::all(3),
                    keep: Keep::All,
                    law: VoteLaw::SINGLE,
                }
            })
            .collect();
        let probe: Vec<WideQSig> = (0..120).map(|_| rand_sig(&mut rng)).collect();
        let scalar = arena
            .scan_weighted_scalar(&segs, &[probe.as_slice()], &[])
            .unwrap();
        let imma = arena
            .scan_weighted_imma(&segs, &[probe.as_slice()], &[])
            .unwrap();
        assert_eq!(scalar[0].len(), SEGMENTS * CASES);
        assert!(scalar[0].iter().any(|v| *v != 0.0), "the scan voted");
        for (ci, (s, m)) in scalar[0].iter().zip(&imma[0]).enumerate() {
            assert_eq!(
                s.to_bits(),
                m.to_bits(),
                "case {ci}: scalar {s} vs chunked IMMA {m} must be bit-identical"
            );
        }
    }
}
