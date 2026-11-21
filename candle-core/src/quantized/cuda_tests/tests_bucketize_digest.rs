//! A seeded digest of every byte `moe_bucketize` writes across a sweep of
//! launches.
//!
//! The bucketize tests check each output against the CPU reference, but a phase
//! that moved a byte of an output nobody compares would pass them. This test
//! runs the widths, routing skews, live class mixes, promotion rings and decode
//! ranges the kernel is written for, and folds every output into one FNV-1a
//! digest. Any change to any output byte changes the pinned value. The digest
//! covers the row and tile tables, the summary, the snapshot, the promotion
//! slots and the started words.

use super::*;

/// FNV-1a over `bytes`, folded into `h`.
pub(super) fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

/// Fold every value of `v` into `h`, little-endian.
pub(super) fn fnv1a_u32s(h: &mut u64, v: &[u32]) {
    for x in v {
        fnv1a(h, &x.to_le_bytes());
    }
}

/// **Every byte bucketize writes, across a seeded sweep of launches, is the
/// pinned digest.** Widths from a one-row draft step through a verify, a
/// decode wave and prefill; dense, skewed and sentinel-strewn routing; and,
/// over a two-row live table, every class mix from all-resident to half
/// remote, with a promotion ring holding empty and victim offers (some of the
/// launch's own row and routed — skipped), marks in flight, every reserve and
/// sweep setting, and decode ranges from none through a prefix to scattered
/// ranges. The digest covers the row and tile tables, the header, the summary
/// with its sequence word, the remote list, the snapshot, the promotion slots,
/// the zeroed counters, the live table after any retarget, the started words,
/// and the ring's head, log and marks. The reference tests above pin what each
/// output means; this pins that no phase moves a byte of any of them.
#[test]
pub(super) fn cuda_moe_bucketize_output_digest() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let stream = dev.cuda_stream();
    let mut rng = StdRng::seed_from_u64(0xd16e_57ed);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    // (n_tokens, k, n_experts, tile_w)
    let shapes: &[(usize, usize, usize, usize)] = &[
        (1, 8, 128, 32),
        (1, 10, 512, 32),
        (2, 10, 512, 32),
        (3, 8, 256, 16),
        (5, 10, 512, 32),
        (5, 10, 512, 64),
        (8, 10, 512, 32),
        (16, 10, 512, 32),
        (16, 8, 128, 32),
        (31, 10, 512, 128),
        (51, 10, 512, 32),
        (64, 10, 512, 32),
        (64, 8, 128, 32),
        (4, 32, 512, 32),
        (1, 1, 4, 1),
        (7, 3, 16, 2),
        (205, 10, 512, 64),
        (333, 8, 256, 32),
        (713, 10, 512, 64),
    ];
    let rows = 2usize;
    // At the narrow kernel's widths, launches whose walk claimed a slot,
    // logged a skip, and retargeted a victim: the sweep must reach all three.
    let (mut narrow_claims, mut narrow_skips, mut narrow_retargets) = (0, 0, 0);
    for &(n_tokens, k, n_experts, tile_w) in shapes {
        for variant in 0..7u32 {
            let a_ub = n_tokens * k;
            // Variant 6 routes everything to four experts: a warp's lanes share
            // an expert far more often than not.
            let span = if variant == 6 {
                n_experts.min(4)
            } else {
                n_experts
            };
            let ids: Vec<u32> = (0..a_ub)
                .map(|_| {
                    if rng.random_ratio(1, 20) {
                        n_experts as u32 + rng.random_range(0..2)
                    } else {
                        rng.random_range(0..span as u32)
                    }
                })
                .collect();
            fnv1a_u32s(&mut h, &ids);
            let t = crate::Tensor::from_vec(ids, (n_tokens, k), &device)?;
            let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
            let mut decode = DecodeRows::none();
            match rng.random_range(0..3) {
                0 => {}
                1 => {
                    decode.push(0, rng.random_range(0..=n_tokens));
                }
                _ => {
                    for _ in 0..rng.random_range(1..4) {
                        let lo = rng.random_range(0..n_tokens);
                        decode.push(lo, rng.random_range(lo..=n_tokens));
                    }
                }
            }
            if variant == 0 {
                moe_bucketize(&t, n_experts, tile_w, &mut ws, None, &decode)?;
            } else {
                // 1 in `remote_one_in` of the live table's entries is remote.
                let remote_one_in = [0u32, 0, 16, 3, 2, 8, 4][variant as usize];
                let plane = rows * n_experts;
                let mut table = vec![0u64; 3 * plane];
                for i in 0..plane {
                    let e = (i % n_experts) as u64;
                    let vram = 0x10_0000 + i as u64 * 0x100;
                    let remote = remote_one_in != 0 && rng.random_ratio(1, remote_one_in);
                    let (g, cold_plane) = if !remote {
                        (vram, 3)
                    } else {
                        match rng.random_range(0..4) {
                            0 => (TEST_PINNED[0].0 + i as u64 * 0x100, 3),
                            1 => (TEST_PINNED[1].0 + e * 0x100, 3),
                            2 => (0, 0),
                            _ => (vram, rng.random_range(1..3)),
                        }
                    };
                    table[i] = g;
                    for p in 1..3 {
                        table[p * plane + i] = if cold_plane == p || g == 0 {
                            0
                        } else {
                            g + 0x10 * p as u64
                        };
                    }
                }
                let row = rng.random_range(0..rows);
                let table_dev = dev.memcpy_stod(&table)?;
                let snap = dev.memcpy_stod(&vec![0x5a5a_5a5a_5a5a_5a5au64; 3 * n_experts])?;
                let summary = dev.memcpy_stod(&vec![0xdead_beefu32; n_experts + 1])?;
                let remote = dev.memcpy_stod(&vec![-1i32; 4 * n_experts])?;
                let counters = dev.memcpy_stod(&[7i32, 7, 7])?;
                let remote_dst = dev.memcpy_stod(&vec![7u64; n_experts])?;
                let started = dev.memcpy_stod(&vec![0u64; rows])?;
                let ring = TestRing::new(8, rows, n_experts);
                let head_at = rng.random_range(0..20);
                ring.start_at(head_at);
                ring.set_reserve([0u32, 1, 2, u32::MAX][rng.random_range(0..4)]);
                ring.set_sweep([u32::MAX, u32::MAX, 0, 1, 2, 5][rng.random_range(0..6)]);
                for _ in 0..rng.random_range(0..4) {
                    // SAFETY: inside the ring's mapped marks.
                    unsafe {
                        std::ptr::write_volatile(ring.mark_ptr(row, rng.random_range(0..span)), 3)
                    };
                }
                let offers: Vec<(u64, Option<(u64, [u64; 3])>)> = (0..rng.random_range(0..8))
                    .map(|o| {
                        let slot = 0xa000 + o as u64 * 0x1000;
                        if rng.random_ratio(1, 2) {
                            (slot, None)
                        } else {
                            // Half the victims sit in the launch's own row,
                            // among the experts it routes most.
                            let ve = rng.random_range(0..span);
                            let vr = if rng.random_ratio(1, 2) {
                                row
                            } else {
                                rng.random_range(0..rows)
                            };
                            let r = 0x9000_0000 + (vr * n_experts + ve) as u64 * 0x100;
                            (
                                slot,
                                Some(((vr * n_experts + ve) as u64, [r, r + 0x10, r + 0x20])),
                            )
                        }
                    })
                    .collect();
                ring.offer(&offers);
                let live = BucketizeLive {
                    gate_row: table_dev.device_ptr(&stream).0 + (row * n_experts * 8) as u64,
                    table_plane: plane as i64,
                    snap: snap.device_ptr(&stream).0,
                    pinned: TEST_PINNED,
                    summary: summary.device_ptr(&stream).0,
                    summary_seq: 40 + variant,
                    remote: remote.device_ptr(&stream).0,
                    counters: counters.device_ptr(&stream).0,
                    row: row as i32,
                    promo: if variant == 1 && n_tokens > 8 {
                        None
                    } else {
                        Some(ring.ring())
                    },
                    remote_dst: remote_dst.device_ptr(&stream).0,
                    started_rows: started.device_ptr(&stream).0,
                    ticket: 1000 + variant as u64,
                    owner: OwnerCheck::default(),
                    ahead: None,
                    cpu: None,
                };
                moe_bucketize(&t, n_experts, tile_w, &mut ws, Some(&live), &decode)?;
                fnv1a_u32s(&mut h, &dev.memcpy_dtov(&summary)?);
                for v in dev.memcpy_dtov(&remote)? {
                    fnv1a(&mut h, &v.to_le_bytes());
                }
                for v in dev.memcpy_dtov(&counters)? {
                    fnv1a(&mut h, &v.to_le_bytes());
                }
                for buf in [&snap, &remote_dst, &table_dev, &started] {
                    for v in dev.memcpy_dtov(buf)? {
                        fnv1a(&mut h, &v.to_le_bytes());
                    }
                }
                fnv1a(&mut h, &ring.head().to_le_bytes());
                for i in 0..8 {
                    fnv1a(&mut h, &ring.log(i).to_le_bytes());
                }
                if a_ub <= 256 {
                    let logged: Vec<u64> = (0..8).map(|i| ring.log(i)).collect();
                    narrow_claims += usize::from(
                        logged
                            .iter()
                            .any(|&l| l != 0 && l & 0xffff != PROMO_SKIP as u64),
                    );
                    narrow_skips += usize::from(
                        logged
                            .iter()
                            .any(|&l| l != 0 && l & 0xffff == PROMO_SKIP as u64),
                    );
                    narrow_retargets += usize::from(dev.memcpy_dtov(&table_dev)? != table);
                    assert!(ring.head() >= head_at, "head never moves back");
                }
                for r in 0..rows {
                    for e in 0..n_experts {
                        fnv1a(&mut h, &ring.mark(r, e).to_le_bytes());
                    }
                }
            }
            for v in dev.memcpy_dtov(&ws.header.slice(..5))? {
                fnv1a(&mut h, &v.to_le_bytes());
            }
            for buf in [&ws.tok_ids, &ws.weight_ids, &ws.perm, &ws.rw_ids] {
                fnv1a_u32s(&mut h, &dev.memcpy_dtov(&buf.slice(..a_ub))?);
            }
            for buf in [&ws.tile_expert, &ws.tile_b_start, &ws.tile_b_cnt] {
                for v in dev.memcpy_dtov(&buf.slice(..a_ub))? {
                    fnv1a(&mut h, &v.to_le_bytes());
                }
            }
            for v in dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))? {
                fnv1a(&mut h, &v.to_le_bytes());
            }
        }
    }
    assert!(
        narrow_claims > 0 && narrow_skips > 0 && narrow_retargets > 0,
        "the narrow widths' walk coverage: {narrow_claims} claiming, {narrow_skips} skipping, \
         {narrow_retargets} retargeting launches"
    );
    assert_eq!(
        h, BUCKETIZE_OUTPUT_DIGEST,
        "bucketize output digest: {h:#018x}"
    );
    Ok(())
}

/// [`cuda_moe_bucketize_output_digest`]'s expected value: the promotion
/// slots, log and marks in the walk's cold-first order (`promotion_walk`).
pub(super) const BUCKETIZE_OUTPUT_DIGEST: u64 = 0x3f67_095c_4769_0364;
