use half::f16;

const HEAD_DIM: usize = 128;
const CHUNK_SIZE: usize = 32;
const N_PALETTE: usize = 4;
const SUB_DIM: usize = HEAD_DIM / N_PALETTE;

type PalMapBytes = [u8; HEAD_DIM / 4];

type TokenRows = [[f32; HEAD_DIM]; CHUNK_SIZE];

fn assert_close(got: f32, want: f32, msg: &str) {
    let err = (got - want).abs();
    assert!(err <= 4.0, "{msg}: got={got} want={want} err={err}");
}

fn assert_matches_f16_roundtrip(got: f32, want: f32, msg: &str) {
    let rounded = f16::from_f32(want).to_f32();
    assert_eq!(
        got, rounded,
        "{msg}: got={got} want_rounded={rounded} raw_want={want}"
    );
}

fn pal_get(pal_map: &PalMapBytes, g: usize) -> usize {
    ((pal_map[g / 4] >> (2 * (g % 4))) & 0x3) as usize
}

fn make_pal_map(pal_of_dim: &[u8; HEAD_DIM]) -> PalMapBytes {
    let mut counts = [0usize; N_PALETTE];
    for d in 0..HEAD_DIM {
        counts[pal_of_dim[d] as usize] += 1;
    }
    for p in 0..N_PALETTE {
        assert_eq!(
            counts[p], SUB_DIM,
            "palette {p} must contain {SUB_DIM} dims"
        );
    }

    let mut out = [0u8; HEAD_DIM / 4];
    for d in 0..HEAD_DIM {
        out[d / 4] |= (pal_of_dim[d] & 0x3) << (2 * (d % 4));
    }
    out
}

fn rank_in_pal(pal_map: &PalMapBytes, p: usize, global_d: usize) -> usize {
    let mut rank = 0usize;
    for g in 0..global_d {
        if pal_get(pal_map, g) == p {
            rank += 1;
        }
    }
    rank
}

fn find_nth_dim_in_pal(pal_map: &PalMapBytes, p: usize, n: usize) -> usize {
    let mut count = 0usize;
    for g in 0..HEAD_DIM {
        if pal_get(pal_map, g) == p {
            if count == n {
                return g;
            }
            count += 1;
        }
    }
    panic!("invalid pal_map: palette {p} has fewer than {} dims", n + 1);
}

fn build_xlat(src_pal_map: &PalMapBytes, dst_pal_map: &PalMapBytes) -> [u8; HEAD_DIM] {
    let mut xlat = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        let dst_pal = d / SUB_DIM;
        let dst_local = d % SUB_DIM;
        let global_d = find_nth_dim_in_pal(dst_pal_map, dst_pal, dst_local);
        let src_pal = pal_get(src_pal_map, global_d);
        let src_local = rank_in_pal(src_pal_map, src_pal, global_d);
        xlat[d] = (src_pal * SUB_DIM + src_local) as u8;
    }
    xlat
}

fn stage1_smem_columns(
    rows: &TokenRows,
    src_pal_map: &PalMapBytes,
) -> [[f16; HEAD_DIM]; CHUNK_SIZE] {
    let mut smem = [[f16::from_f32(0.0); HEAD_DIM]; CHUNK_SIZE];
    for src_col in 0..HEAD_DIM {
        let src_pal = src_col / SUB_DIM;
        let src_local = src_col % SUB_DIM;
        let global_d = find_nth_dim_in_pal(src_pal_map, src_pal, src_local);
        for t in 0..CHUNK_SIZE {
            smem[t][src_col] = f16::from_f32(rows[t][global_d]);
        }
    }
    smem
}

fn stage2_dst_palette_rows(
    smem: &[[f16; HEAD_DIM]; CHUNK_SIZE],
    xlat: &[u8; HEAD_DIM],
) -> [[[f32; SUB_DIM]; CHUNK_SIZE]; N_PALETTE] {
    let mut dst = [[[0.0f32; SUB_DIM]; CHUNK_SIZE]; N_PALETTE];
    for d in 0..HEAD_DIM {
        let dst_pal = d / SUB_DIM;
        let dst_local = d % SUB_DIM;
        let src_col = xlat[d] as usize;
        for t in 0..CHUNK_SIZE {
            dst[dst_pal][t][dst_local] = smem[t][src_col].to_f32();
        }
    }
    dst
}

fn fill_rows() -> TokenRows {
    let mut rows = [[0.0f32; HEAD_DIM]; CHUNK_SIZE];
    for t in 0..CHUNK_SIZE {
        for d in 0..HEAD_DIM {
            rows[t][d] = (t as f32) * 1000.0 + d as f32 + 0.125;
        }
    }
    rows
}

fn identity_pal_map() -> PalMapBytes {
    let mut pal_of = [0u8; HEAD_DIM];
    for (d, slot) in pal_of.iter_mut().enumerate().take(HEAD_DIM) {
        *slot = (d / SUB_DIM) as u8;
    }
    make_pal_map(&pal_of)
}

fn reversed_pal_map() -> PalMapBytes {
    let mut pal_of = [0u8; HEAD_DIM];
    for (d, slot) in pal_of.iter_mut().enumerate().take(HEAD_DIM) {
        *slot = (3 - (d / SUB_DIM)) as u8;
    }
    make_pal_map(&pal_of)
}

fn striped_pal_map(stripe: usize) -> PalMapBytes {
    let mut pal_of = [0u8; HEAD_DIM];
    for (d, slot) in pal_of.iter_mut().enumerate().take(HEAD_DIM) {
        *slot = ((d / stripe) % 4) as u8;
    }
    make_pal_map(&pal_of)
}

fn random_balanced_pal_map(seed: u64) -> PalMapBytes {
    let mut perm: Vec<usize> = (0..HEAD_DIM).collect();
    let mut rng = seed;
    for i in (1..HEAD_DIM).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }

    let mut pal_of = [0u8; HEAD_DIM];
    for (rank, &d) in perm.iter().enumerate() {
        pal_of[d] = (rank / SUB_DIM) as u8;
    }
    make_pal_map(&pal_of)
}

#[test]
fn xlat_identity_is_identity() {
    let pal = identity_pal_map();
    let xlat = build_xlat(&pal, &pal);
    for (d, &x) in xlat.iter().enumerate().take(HEAD_DIM) {
        assert_eq!(x as usize, d);
    }
}

#[test]
fn xlat_is_bijection_random_maps() {
    for seed in 0..100u64 {
        let src = random_balanced_pal_map(seed * 17 + 1);
        let dst = random_balanced_pal_map(seed * 17 + 2);
        let xlat = build_xlat(&src, &dst);

        let mut seen = [false; HEAD_DIM];
        for &x in xlat.iter().take(HEAD_DIM) {
            let xi = x as usize;
            assert!(xi < HEAD_DIM);
            assert!(!seen[xi], "duplicate src_col {xi} at seed {seed}");
            seen[xi] = true;
        }
    }
}

#[test]
fn stage1_identity_columns_match_global_dims() {
    let rows = fill_rows();
    let src = identity_pal_map();
    let smem = stage1_smem_columns(&rows, &src);

    for t in 0..CHUNK_SIZE {
        for d in 0..HEAD_DIM {
            assert_matches_f16_roundtrip(smem[t][d].to_f32(), rows[t][d], "stage1 identity");
        }
    }
}

#[test]
fn stage2_identity_to_identity_roundtrip() {
    let rows = fill_rows();
    let src = identity_pal_map();
    let dst = identity_pal_map();

    let smem = stage1_smem_columns(&rows, &src);
    let xlat = build_xlat(&src, &dst);
    let out = stage2_dst_palette_rows(&smem, &xlat);

    for (p, pal_rows) in out.iter().enumerate().take(N_PALETTE) {
        for (t, _) in rows.iter().enumerate().take(CHUNK_SIZE) {
            for ld in 0..SUB_DIM {
                let global_d = find_nth_dim_in_pal(&dst, p, ld);
                assert_matches_f16_roundtrip(
                    pal_rows[t][ld],
                    rows[t][global_d],
                    "identity->identity",
                );
            }
        }
    }
}

#[test]
fn stage2_identity_to_reversed_routes_correctly() {
    let rows = fill_rows();
    let src = identity_pal_map();
    let dst = reversed_pal_map();

    let smem = stage1_smem_columns(&rows, &src);
    let xlat = build_xlat(&src, &dst);
    let out = stage2_dst_palette_rows(&smem, &xlat);

    for (p, pal_rows) in out.iter().enumerate().take(N_PALETTE) {
        for (t, _) in rows.iter().enumerate().take(CHUNK_SIZE) {
            for ld in 0..SUB_DIM {
                let global_d = find_nth_dim_in_pal(&dst, p, ld);
                assert_matches_f16_roundtrip(
                    pal_rows[t][ld],
                    rows[t][global_d],
                    "identity->reversed",
                );
            }
        }
    }
}

#[test]
fn stage2_random_src_dst_maps_route_correctly() {
    let rows = fill_rows();
    for seed in 0..64u64 {
        let src = random_balanced_pal_map(seed * 13 + 5);
        let dst = random_balanced_pal_map(seed * 13 + 9);
        let smem = stage1_smem_columns(&rows, &src);
        let xlat = build_xlat(&src, &dst);
        let out = stage2_dst_palette_rows(&smem, &xlat);

        for (p, pal_rows) in out.iter().enumerate().take(N_PALETTE) {
            for (t, _) in rows.iter().enumerate().take(CHUNK_SIZE) {
                for ld in 0..SUB_DIM {
                    let global_d = find_nth_dim_in_pal(&dst, p, ld);
                    let msg = format!("seed={seed} p={p} t={t} ld={ld}");
                    assert_matches_f16_roundtrip(pal_rows[t][ld], rows[t][global_d], &msg);
                }
            }
        }
    }
}

#[test]
fn stage2_preserves_warp_aligned_dst_layout() {
    let rows = fill_rows();
    let src = striped_pal_map(4);
    let dst = striped_pal_map(8);
    let smem = stage1_smem_columns(&rows, &src);
    let xlat = build_xlat(&src, &dst);
    let out = stage2_dst_palette_rows(&smem, &xlat);

    for (d, &x) in xlat.iter().enumerate().take(HEAD_DIM) {
        let dst_pal = d / SUB_DIM;
        let dst_local = d % SUB_DIM;
        let src_col = x as usize;
        for (t, _) in rows.iter().enumerate().take(CHUNK_SIZE) {
            assert_close(
                out[dst_pal][t][dst_local],
                smem[t][src_col].to_f32(),
                "warp-aligned layout",
            );
        }
    }
}

#[test]
fn f16_staging_error_bound_for_large_values() {
    let mut rows = [[0.0f32; HEAD_DIM]; CHUNK_SIZE];
    for (t, row) in rows.iter_mut().enumerate().take(CHUNK_SIZE) {
        for (d, elem) in row.iter_mut().enumerate().take(HEAD_DIM) {
            *elem = ((t * HEAD_DIM + d) as f32) * 3.14159;
        }
    }

    let src = random_balanced_pal_map(1234);
    let smem = stage1_smem_columns(&rows, &src);

    // For finite values in this range, f16 round-trip should stay within a small absolute bound.
    for (src_col, _) in smem[0].iter().enumerate().take(HEAD_DIM) {
        let sp = src_col / SUB_DIM;
        let ld = src_col % SUB_DIM;
        let global_d = find_nth_dim_in_pal(&src, sp, ld);
        for (t, _) in rows.iter().enumerate().take(CHUNK_SIZE) {
            let got = smem[t][src_col].to_f32();
            let want = rows[t][global_d];
            let err = (got - want).abs();
            assert!(
                err <= 4.0,
                "t={t} src_col={src_col} err={err} got={got} want={want}"
            );
        }
    }
}

#[test]
fn fixed_patterns_cover_common_and_pathological_maps() {
    let rows = fill_rows();
    let maps = [
        identity_pal_map(),
        reversed_pal_map(),
        striped_pal_map(1),
        striped_pal_map(2),
        striped_pal_map(4),
        striped_pal_map(8),
        striped_pal_map(16),
    ];

    for (i, src) in maps.iter().enumerate() {
        for (j, dst) in maps.iter().enumerate() {
            let smem = stage1_smem_columns(&rows, src);
            let xlat = build_xlat(src, dst);
            let out = stage2_dst_palette_rows(&smem, &xlat);
            for (p, pal_rows) in out.iter().enumerate().take(N_PALETTE) {
                for (t, _) in rows.iter().enumerate().take(CHUNK_SIZE) {
                    for ld in 0..SUB_DIM {
                        let global_d = find_nth_dim_in_pal(dst, p, ld);
                        let msg = format!("src_case={i} dst_case={j}");
                        assert_matches_f16_roundtrip(pal_rows[t][ld], rows[t][global_d], &msg);
                    }
                }
            }
        }
    }
}
