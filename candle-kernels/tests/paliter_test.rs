/// Standalone unit tests for PalIter — the palette-aware dimension iterator.
///
/// This mirrors the CUDA kernel's flipped-loop ballot+popcount algorithm
/// in pure Rust, then validates it against a naive reference implementation
/// across many pal_map configurations.
///
/// INVARIANT: Every valid pal_map must assign exactly SUB (32) dims to each
/// of the 4 palettes.  The ArenaAccessor allocates [p*SUB .. (p+1)*SUB) for
/// palette p, so overflow/underflow is undefined.
///
/// DIM ORDERING: Within each palette's smem region, ArenaAccessor fills in
/// (sub-position major, lane minor) order:
///   (jj=0, lane=0), (jj=0, lane=1), ..., (jj=0, lane=31),
///   (jj=1, lane=0), ..., (jj=3, lane=31)
/// PalIter computes the reverse: for dim (lane, j), which smem offset?

const N_PALETTE: usize = 4;
const HEAD_DIM: usize = 128;
const VEC: usize = 4;
const WARP_SIZE: usize = 32;
const SUB: usize = HEAD_DIM / N_PALETTE; // 32

// ── Helpers ─────────────────────────────────────────────────────────────────

fn pal_at(pal_map: &[u8; WARP_SIZE], lane: usize, j: usize) -> usize {
    ((pal_map[lane] >> (j * 2)) & 3) as usize
}

/// Build pal_map from per-dim palette assignments.  Panics if not balanced.
fn make_pal_map(pal_of: &[u8; HEAD_DIM]) -> [u8; WARP_SIZE] {
    let mut counts = [0usize; N_PALETTE];
    for d in 0..HEAD_DIM {
        counts[pal_of[d] as usize] += 1;
    }
    for p in 0..N_PALETTE {
        assert_eq!(
            counts[p], SUB,
            "palette {p} has {} dims, need {SUB}",
            counts[p]
        );
    }
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..WARP_SIZE {
        let mut byte = 0u8;
        for j in 0..VEC {
            byte |= (pal_of[lane * VEC + j] & 3) << (j * 2);
        }
        pm[lane] = byte;
    }
    pm
}

fn is_balanced(pal_map: &[u8; WARP_SIZE]) -> bool {
    let mut counts = [0usize; N_PALETTE];
    for lane in 0..WARP_SIZE {
        for j in 0..VEC {
            counts[pal_at(pal_map, lane, j)] += 1;
        }
    }
    counts.iter().all(|&c| c == SUB)
}

// ── Reference (naive) implementation ────────────────────────────────────────
fn reference_scatter(pal_map: &[u8; WARP_SIZE]) -> [u8; HEAD_DIM] {
    let mut result = [0u8; HEAD_DIM];
    for lane in 0..WARP_SIZE {
        for j in 0..VEC {
            let p = pal_at(pal_map, lane, j);
            let mut count = 0usize;
            // Dim-sequential order: dim d = lane*VEC+j.
            // d' < d iff lane2 < lane, or (lane2 == lane && jj < j).
            for lane2 in 0..WARP_SIZE {
                for jj in 0..VEC {
                    let before = lane2 < lane || (lane2 == lane && jj < j);
                    if before && pal_at(pal_map, lane2, jj) == p {
                        count += 1;
                    }
                }
            }
            result[lane * VEC + j] = (p * SUB + count) as u8;
        }
    }
    result
}

// ── Ballot simulation ───────────────────────────────────────────────────────
fn ballot(pal_map: &[u8; WARP_SIZE], jj: usize, bit_fn: fn(u8) -> bool) -> u32 {
    let mut mask = 0u32;
    for lane in 0..WARP_SIZE {
        let p = ((pal_map[lane] >> (jj * 2)) & 3) as u8;
        if bit_fn(p) {
            mask |= 1u32 << lane;
        }
    }
    mask
}

// ── Kernel-mirroring implementation (palette-bucket accumulation) ────────────
fn kernel_scatter_for_lane(pal_map: &[u8; WARP_SIZE], lane: usize) -> [u8; VEC] {
    let my_byte = pal_map[lane];
    let lane_mask: u32 = (1u32 << lane).wrapping_sub(1);

    // Part (a): cross-lane counts per palette bucket
    let mut cross = [0i32; N_PALETTE];
    for jj in 0..VEC {
        let b0 = ballot(pal_map, jj, |p| (p & 1) != 0);
        let b1 = ballot(pal_map, jj, |p| (p >> 1) != 0);

        for p in 0..N_PALETTE {
            let pal_mask =
                (if (p & 1) != 0 { b0 } else { !b0 }) & (if (p >> 1) != 0 { b1 } else { !b1 });
            cross[p] += (pal_mask & lane_mask).count_ones() as i32;
        }
    }

    // Part (b) + final: lane-local self-contribution + cross lookup
    let mut scatter = [0u8; VEC];
    for j in 0..VEC {
        let p = ((my_byte >> (j * 2)) & 3) as usize;
        let mut local = 0i32;
        for jj in 0..j {
            let pjj = ((my_byte >> (jj * 2)) & 3) as usize;
            if pjj == p { local += 1; }
        }
        scatter[j] = (p * SUB + cross[p] as usize + local as usize) as u8;
    }
    scatter
}

fn kernel_scatter_all(pal_map: &[u8; WARP_SIZE]) -> [u8; HEAD_DIM] {
    let mut result = [0u8; HEAD_DIM];
    for lane in 0..WARP_SIZE {
        let s = kernel_scatter_for_lane(pal_map, lane);
        for j in 0..VEC {
            result[lane * VEC + j] = s[j];
        }
    }
    result
}

// ── Validation ──────────────────────────────────────────────────────────────
fn validate(label: &str, pal_map: &[u8; WARP_SIZE]) {
    assert!(is_balanced(pal_map), "{label}: pal_map not balanced");

    let ref_result = reference_scatter(pal_map);
    let kern_result = kernel_scatter_all(pal_map);

    // 1. Kernel matches reference
    for d in 0..HEAD_DIM {
        assert_eq!(
            kern_result[d],
            ref_result[d],
            "{label}: dim {d} mismatch: kernel={} ref={} (lane={}, j={})",
            kern_result[d],
            ref_result[d],
            d / VEC,
            d % VEC
        );
    }

    // 2. Bijectivity: every smem offset [0..128) used exactly once
    let mut seen = [false; HEAD_DIM];
    for d in 0..HEAD_DIM {
        let idx = ref_result[d] as usize;
        assert!(idx < HEAD_DIM, "{label}: dim {d} idx {idx} OOB");
        assert!(!seen[idx], "{label}: dim {d} idx {idx} duplicate");
        seen[idx] = true;
    }

    // 3. Per-palette range: palette p → [p*SUB .. (p+1)*SUB)
    for d in 0..HEAD_DIM {
        let p = pal_at(pal_map, d / VEC, d % VEC);
        let idx = ref_result[d] as usize;
        assert!(
            idx >= p * SUB && idx < (p + 1) * SUB,
            "{label}: dim {d} (p={p}) -> {idx}, expected [{}, {})",
            p * SUB,
            (p + 1) * SUB
        );
    }
}

// ── Deterministic RNG for balanced pal_map generation ───────────────────────
fn random_balanced_pal_map(seed: u64) -> [u8; WARP_SIZE] {
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
    for (rank, &dim) in perm.iter().enumerate() {
        pal_of[dim] = (rank / SUB) as u8;
    }
    make_pal_map(&pal_of)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_identity_routing() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (d / SUB) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("identity", &pm);
}

#[test]
fn test_reversed_palettes() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (3 - d / SUB) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("reversed", &pm);
}

#[test]
fn test_within_lane_all_different() {
    let pm = [0xE4u8; WARP_SIZE]; // j0=p0, j1=p1, j2=p2, j3=p3
    validate("within_lane_all_diff", &pm);
}

#[test]
fn test_within_lane_reversed() {
    let pm = [0x1Bu8; WARP_SIZE]; // j0=p3, j1=p2, j2=p1, j3=p0
    validate("within_lane_rev", &pm);
}

#[test]
fn test_within_lane_swapped_pairs() {
    let pm = [0xB1u8; WARP_SIZE]; // j0=p1, j1=p0, j2=p3, j3=p2
    validate("within_lane_swapped_pairs", &pm);
}

#[test]
fn test_stripe_width_1() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (d % 4) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("stripe_w1", &pm);
}

#[test]
fn test_stripe_width_2() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = ((d / 2) % 4) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("stripe_w2", &pm);
}

#[test]
fn test_stripe_width_4() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = ((d / 4) % 4) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("stripe_w4", &pm);
}

#[test]
fn test_stripe_width_8() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = ((d / 8) % 4) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("stripe_w8", &pm);
}

#[test]
fn test_stripe_width_16() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = ((d / 16) % 4) as u8;
    }
    let pm = make_pal_map(&pal_of);
    validate("stripe_w16", &pm);
}

#[test]
fn test_checkerboard() {
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..WARP_SIZE {
        let mut byte = 0u8;
        for j in 0..VEC {
            let p = ((lane & 1) * 2 + (j & 1)) as u8;
            byte |= p << (j * 2);
        }
        pm[lane] = byte;
    }
    validate("checkerboard", &pm);
}

#[test]
fn test_quarter_split_by_lane() {
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..WARP_SIZE {
        let p = (lane / 8) as u8;
        pm[lane] = p | (p << 2) | (p << 4) | (p << 6);
    }
    validate("quarter_by_lane", &pm);
}

#[test]
fn test_half_lanes_swapped() {
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..16 {
        pm[lane] = 0xE4;
    } // 0,1,2,3
    for lane in 16..32 {
        pm[lane] = 0x4E;
    } // 2,3,0,1
    validate("half_lanes_swapped", &pm);
}

#[test]
fn test_diagonal() {
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..WARP_SIZE {
        let base = lane / 8;
        let mut byte = 0u8;
        for j in 0..VEC {
            byte |= (((base + j) % 4) as u8) << (j * 2);
        }
        pm[lane] = byte;
    }
    validate("diagonal", &pm);
}

#[test]
fn test_single_lane_different() {
    let mut pm = [0xE4u8; WARP_SIZE];
    pm[0] = 0x1B;
    validate("single_lane_diff", &pm);
}

#[test]
fn test_last_lane_different() {
    let mut pm = [0xE4u8; WARP_SIZE];
    pm[31] = 0x1B;
    validate("last_lane_diff", &pm);
}

#[test]
fn test_middle_lane_different() {
    let mut pm = [0xE4u8; WARP_SIZE];
    pm[15] = 0x1B;
    validate("mid_lane_diff", &pm);
}

#[test]
fn test_two_lanes_swapped() {
    let mut pm = [0xE4u8; WARP_SIZE];
    pm[0] = 0x1B;
    pm[31] = 0x1B;
    validate("two_lanes_swapped", &pm);
}

#[test]
fn test_all_24_permutation_bytes() {
    let perms: Vec<[u8; 4]> = {
        let mut v = Vec::new();
        for a in 0..4u8 {
            for b in 0..4u8 {
                if b == a {
                    continue;
                }
                for c in 0..4u8 {
                    if c == a || c == b {
                        continue;
                    }
                    let d = 6 - a - b - c;
                    v.push([a, b, c, d]);
                }
            }
        }
        v
    };
    assert_eq!(perms.len(), 24);
    for perm in &perms {
        let byte = perm[0] | (perm[1] << 2) | (perm[2] << 4) | (perm[3] << 6);
        let pm = [byte; WARP_SIZE];
        validate(&format!("perm_{byte:#04x}"), &pm);
    }
}

#[test]
fn test_random_balanced_200() {
    for seed in 0u64..200 {
        let pm = random_balanced_pal_map(seed);
        validate(&format!("rng_{seed}"), &pm);
    }
}

#[test]
fn test_random_balanced_1000() {
    for seed in 1000u64..2000 {
        let pm = random_balanced_pal_map(seed);
        validate(&format!("rng_{seed}"), &pm);
    }
}

#[test]
fn test_inverse_mapping_roundtrip() {
    for seed in 0u64..50 {
        let pm = random_balanced_pal_map(seed + 5000);
        let scatter = reference_scatter(&pm);
        let mut smem = [0.0f32; HEAD_DIM];
        for d in 0..HEAD_DIM {
            smem[scatter[d] as usize] = d as f32;
        }
        for d in 0..HEAD_DIM {
            assert_eq!(
                smem[scatter[d] as usize] as usize, d,
                "roundtrip seed {seed}: dim {d}"
            );
        }
    }
}

#[test]
fn test_per_lane_coupling() {
    let pm_a = random_balanced_pal_map(42);
    validate("coupling_a", &pm_a);
    let mut pm_b = pm_a;
    pm_b.swap(0, 1);
    if is_balanced(&pm_b) {
        validate("coupling_b", &pm_b);
        let res_a = kernel_scatter_all(&pm_a);
        let res_b = kernel_scatter_all(&pm_b);
        assert!(
            (0..HEAD_DIM).any(|d| res_a[d] != res_b[d]),
            "swapping lanes should change scatter"
        );
    }
}

#[test]
fn test_pairwise_lane_swaps() {
    let base = [0xE4u8; WARP_SIZE];
    let mut tested = 0;
    for a in 0..WARP_SIZE {
        for b in (a + 1)..WARP_SIZE {
            let mut pm = base;
            pm.swap(a, b);
            if is_balanced(&pm) {
                validate(&format!("swap_{a}_{b}"), &pm);
                tested += 1;
            }
        }
    }
    assert!(tested > 0, "should have at least some balanced swaps");
}

#[test]
fn test_extreme_scramble() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (d / SUB) as u8;
    }
    pal_of.reverse();
    let pm = make_pal_map(&pal_of);
    validate("extreme_scramble", &pm);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hand-computed expected-value tests (driven from pal_map, not algorithm)
// ═══════════════════════════════════════════════════════════════════════════

/// Uniform byte 0xE4: every lane has j0=p0, j1=p1, j2=p2, j3=p3.
/// ArenaAccessor fills palette p's region [p*32 .. p*32+32) in
/// (sub-position major, lane minor) order.
///
/// For p0: the dims assigned to p0 are j=0 across all 32 lanes.
/// In (jj, lane) order, jj=0 is the only sub-position for p0.
/// So p0's region is filled: (jj=0, lane=0), (jj=0, lane=1), ..., (jj=0, lane=31)
/// → scatter[lane=L, j=0] = 0*32 + L = L
///
/// For p1: dims assigned are j=1 across all 32 lanes.
/// → scatter[lane=L, j=1] = 1*32 + L = 32 + L
///
/// For p2: j=2 → scatter[lane=L, j=2] = 64 + L
/// For p3: j=3 → scatter[lane=L, j=3] = 96 + L
#[test]
fn test_expected_uniform_0xe4() {
    let pm = [0xE4u8; WARP_SIZE];
    let scatter = kernel_scatter_all(&pm);
    for lane in 0..WARP_SIZE {
        assert_eq!(
            scatter[lane * 4 + 0],
            lane as u8,
            "lane {lane} j=0: expected {lane}, got {}",
            scatter[lane * 4]
        );
        assert_eq!(
            scatter[lane * 4 + 1],
            (32 + lane) as u8,
            "lane {lane} j=1: expected {}, got {}",
            32 + lane,
            scatter[lane * 4 + 1]
        );
        assert_eq!(
            scatter[lane * 4 + 2],
            (64 + lane) as u8,
            "lane {lane} j=2: expected {}, got {}",
            64 + lane,
            scatter[lane * 4 + 2]
        );
        assert_eq!(
            scatter[lane * 4 + 3],
            (96 + lane) as u8,
            "lane {lane} j=3: expected {}, got {}",
            96 + lane,
            scatter[lane * 4 + 3]
        );
    }
}

/// Uniform byte 0x1B: every lane has j0=p3, j1=p2, j2=p1, j3=p0.
/// p3 owns j=0 across all lanes → p3 region [96..128), filled lane-minor:
///   scatter[lane=L, j=0] = 96 + L
/// p2 owns j=1 → scatter[lane=L, j=1] = 64 + L
/// p1 owns j=2 → scatter[lane=L, j=2] = 32 + L
/// p0 owns j=3 → scatter[lane=L, j=3] = 0 + L
#[test]
fn test_expected_uniform_0x1b() {
    let pm = [0x1Bu8; WARP_SIZE];
    let scatter = kernel_scatter_all(&pm);
    for lane in 0..WARP_SIZE {
        assert_eq!(scatter[lane * 4 + 0], (96 + lane) as u8, "lane {lane} j=0");
        assert_eq!(scatter[lane * 4 + 1], (64 + lane) as u8, "lane {lane} j=1");
        assert_eq!(scatter[lane * 4 + 2], (32 + lane) as u8, "lane {lane} j=2");
        assert_eq!(scatter[lane * 4 + 3], lane as u8, "lane {lane} j=3");
    }
}

/// Quarter-by-lane: lanes 0..7 all-p0, lanes 8..15 all-p1, etc.
/// Each palette p owns ALL 4 sub-positions of its 8 lanes.
/// In dim-sequential order, within p's region [p*32..p*32+32),
/// dims appear in order: lane_local_0's 4 slots, lane_local_1's 4 slots, etc.
/// For lane L in group g (L = g*8 + local), palette p=g:
///   scatter[L, j] = p*32 + local*4 + j
#[test]
fn test_expected_quarter_by_lane() {
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..WARP_SIZE {
        let p = (lane / 8) as u8;
        pm[lane] = p | (p << 2) | (p << 4) | (p << 6);
    }
    let scatter = kernel_scatter_all(&pm);
    for lane in 0..WARP_SIZE {
        let p = lane / 8;
        let local = lane % 8;
        for j in 0..VEC {
            let expected = (p * 32 + local * 4 + j) as u8;
            assert_eq!(
                scatter[lane * 4 + j],
                expected,
                "lane {lane} j={j}: expected {expected}, got {}",
                scatter[lane * 4 + j]
            );
        }
    }
}

/// Identity pal_map: dims 0..31=p0, 32..63=p1, 64..95=p2, 96..127=p3.
/// Lane L has byte encoding palettes of dims [L*4 .. L*4+3].
///   Lanes 0..7:  all p0  (byte 0x00)
///   Lanes 8..15: all p1  (byte 0x55)
///   Lanes 16..23: all p2 (byte 0xAA)
///   Lanes 24..31: all p3 (byte 0xFF)
/// Same as quarter-by-lane, so same expected scatter.
/// In dim-sequential layout: scatter[L, j] = p*32 + local*4 + j
/// For identity, this means scatter[L, j] = L*4 + j (= the dim itself).
#[test]
fn test_expected_identity() {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (d / 32) as u8;
    }
    let pm = make_pal_map(&pal_of);
    let scatter = kernel_scatter_all(&pm);
    for lane in 0..WARP_SIZE {
        for j in 0..VEC {
            // Identity: dim d maps to itself!
            let expected = (lane * VEC + j) as u8;
            assert_eq!(
                scatter[lane * 4 + j],
                expected,
                "lane {lane} j={j}: expected {expected}, got {}",
                scatter[lane * 4 + j]
            );
        }
    }
}

/// Single lane differs: all lanes 0xE4 (j0=p0,j1=p1,j2=p2,j3=p3)
/// except lane 0 = 0x1B (j0=p3,j1=p2,j2=p1,j3=p0).
///
/// In dim-sequential order (iterate d=0..127, i.e. lane 0 first):
///   Lane 0:  j0=p3, j1=p2, j2=p1, j3=p0  (dims 0,1,2,3)
///   Lane 1+: j0=p0, j1=p1, j2=p2, j3=p3  (dims 4..127)
///
/// p0: lane 0 j=3 (dim 3, offset 0), then lanes 1..31 j=0 (dims 4,8,12,..).
///   scatter[0,3] = 0*32 + 0 = 0
///   scatter[L,0] = 0*32 + 1 + (L-1)*1 = L  for L>=1
///   (each lane 1..31 contributes 1 p0-dim at j=0, plus lane 0's j=3)
///
/// p3: lane 0 j=0 (dim 0, offset 0), then lanes 1..31 j=3 (dims 7,11,..).
///   scatter[0,0] = 3*32 + 0 = 96
///   scatter[L,3] = 3*32 + 1 + (L-1) = 96 + L  for L>=1
#[test]
fn test_expected_single_lane_differs() {
    let mut pm = [0xE4u8; WARP_SIZE];
    pm[0] = 0x1B; // j0=p3, j1=p2, j2=p1, j3=p0

    let scatter = kernel_scatter_all(&pm);
    let ref_scatter = reference_scatter(&pm);

    // Verify kernel matches reference (dim-sequential order)
    for d in 0..HEAD_DIM {
        assert_eq!(
            scatter[d], ref_scatter[d],
            "dim {d}: kernel={} ref={}",
            scatter[d], ref_scatter[d]
        );
    }

    // Spot-check key values:
    // p3: lane 0, j=0 → first p3 dim → offset 0 within p3
    assert_eq!(scatter[0 * 4 + 0], 96, "p3 lane 0 j=0");
    // p0: lane 0, j=3 → first p0 dim (dim 3) → offset 0 within p0
    assert_eq!(scatter[0 * 4 + 3], 0, "p0 lane 0 j=3");
    // p0: lane 1, j=0 → second p0 dim (dim 4) → offset 1 within p0
    assert_eq!(scatter[1 * 4 + 0], 1, "p0 lane 1 j=0");
    // p3: lane 1, j=3 → second p3 dim (dim 7) → offset 1 within p3
    assert_eq!(scatter[1 * 4 + 3], 97, "p3 lane 1 j=3");
}

/// Half-lanes swapped: lanes 0..15 = 0xE4, lanes 16..31 = 0x4E.
/// 0xE4 = j0=p0, j1=p1, j2=p2, j3=p3
/// 0x4E = j0=p2, j1=p3, j2=p0, j3=p1
///
/// In dim-sequential order:
///   Lanes 0..15 each contribute: 1 p0, 1 p1, 1 p2, 1 p3 dim (in j order).
///   Lanes 16..31 each contribute: 1 p2, 1 p3, 1 p0, 1 p1 dim.
///
/// p0 owners: j=0 on lanes 0..15 (64 earlier dims total before lane 16),
///   then j=2 on lanes 16..31.
///   scatter[L<16, j=0]: cross = L p0-dims from lanes <L, local = 0
///     = 0 + L*1 + 0 = L  (each earlier lane has 1 p0 dim at j=0)
///   scatter[L>=16, j=2]: cross from lanes <L: 16 p0-dims from lanes 0..15
///     + (L-16) p0-dims from lanes 16..L-1, local = 0 (j=2 is first p0 in 0x4E)
///     = 16 + (L-16) = L
///
/// Similarly p1, p2, p3 follow the same pattern.
#[test]
fn test_expected_half_swapped() {
    let mut pm = [0u8; WARP_SIZE];
    for lane in 0..16 {
        pm[lane] = 0xE4;
    }
    for lane in 16..32 {
        pm[lane] = 0x4E;
    }
    let scatter = kernel_scatter_all(&pm);
    let ref_scatter = reference_scatter(&pm);

    // Full verification against reference
    for d in 0..HEAD_DIM {
        assert_eq!(
            scatter[d], ref_scatter[d],
            "dim {d}: kernel={} ref={}",
            scatter[d], ref_scatter[d]
        );
    }

    // Spot-check p0: lanes 0..15 at j=0, lanes 16..31 at j=2
    for lane in 0..16usize {
        assert_eq!(scatter[lane * 4 + 0], lane as u8, "p0 lane {lane} j=0");
    }
    for lane in 16..32usize {
        assert_eq!(scatter[lane * 4 + 2], lane as u8, "p0 lane {lane} j=2");
    }
}

fn main() {
    println!("Run with: rustc --test --edition 2021 <this_file> -o paliter_test && ./paliter_test");
}
