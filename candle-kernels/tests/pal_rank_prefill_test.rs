/// Unit tests for the optimized prefill pal_map rank computation.
///
/// In prefill, each thread tid ∈ [0,128) owns one global dim. It needs:
///   my_p    = palette index for dim tid
///   my_rank = count of dims < tid that belong to the same palette
///
/// The optimized approach reads pal_map as 8 × uint32_t (each word = 16 dims
/// at 2 bits each), uses pal_match_mask + popc to count matches per word.
///
/// This test validates the optimized version against naive reference for many
/// pal_map configurations.

const HEAD_DIM: usize = 128;
const N_PALETTE: usize = 4;
const SUB: usize = HEAD_DIM / N_PALETTE; // 32

// ── pal_map encoding ────────────────────────────────────────────────────────
// pal_map is 32 bytes. Global dim g is stored as 2 bits at:
//   byte g/4, bits (2*(g%4))..(2*(g%4)+1)
// Equivalently as uint32_t words: word g/16, bits (2*(g%16))..(2*(g%16)+1)

fn pal_map_get(pm: &[u8; 32], g: usize) -> usize {
    ((pm[g / 4] >> (2 * (g % 4))) & 0x3) as usize
}

fn make_pal_map(pal_of: &[u8; HEAD_DIM]) -> [u8; 32] {
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
    let mut pm = [0u8; 32];
    for g in 0..HEAD_DIM {
        pm[g / 4] |= (pal_of[g] & 3) << (2 * (g % 4));
    }
    pm
}

// ── Reference (naive) ───────────────────────────────────────────────────────
fn reference_rank(pm: &[u8; 32], tid: usize) -> (usize, usize) {
    let my_p = pal_map_get(pm, tid);
    let mut rank = 0;
    for g in 0..tid {
        if pal_map_get(pm, g) == my_p {
            rank += 1;
        }
    }
    (my_p, rank)
}

// ── Optimized (popc-based) ──────────────────────────────────────────────────
// Reads pal_map as 8 × u32 words, processes 16 dims per word.

fn pal_match_mask(w: u32, p: usize) -> u32 {
    let b0 = w & 0x55555555u32;
    let b1 = (w >> 1) & 0x55555555u32;
    let m0 = if (p & 1) != 0 {
        b0
    } else {
        !b0 & 0x55555555u32
    };
    let m1 = if (p & 2) != 0 {
        b1
    } else {
        !b1 & 0x55555555u32
    };
    m0 & m1
}

fn optimized_rank(pm: &[u8; 32], tid: usize) -> (usize, usize) {
    let my_p = pal_map_get(pm, tid);

    // Reinterpret pal_map as 8 × u32
    let words: &[u32; 8] = unsafe { &*(pm.as_ptr() as *const [u32; 8]) };

    let word_idx = tid / 16;
    let _bit_pos = (tid % 16) * 2; // not directly used, but tid%16 is

    let mut rank = 0usize;

    // Full words before tid's word
    for i in 0..word_idx {
        let m = pal_match_mask(words[i], my_p);
        rank += m.count_ones() as usize;
    }

    // Partial word: mask off dims >= tid
    let w = words[word_idx];
    let mut m = pal_match_mask(w, my_p);
    // Keep only the lowest (tid%16) 1-bit positions.
    // Each dim occupies 1 bit in the match mask at position (dim_within_word * 2)
    // Wait — pal_match_mask produces bits at even positions (0x55555555 mask).
    // Dim k within the word has its match bit at position k*2.
    // We want dims 0..tid%16-1, so bits at positions 0,2,4,...,(tid%16-1)*2
    // Mask: bits below position tid%16*2
    let partial_dims = tid % 16;
    if partial_dims == 0 {
        // Don't count any bits from this word
        m = 0;
    } else {
        // Keep bits below position partial_dims*2
        m &= (1u32 << (partial_dims * 2)) - 1;
    }
    rank += m.count_ones() as usize;

    (my_p, rank)
}

// ── Test helpers ────────────────────────────────────────────────────────────

fn identity_pal_of() -> [u8; HEAD_DIM] {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (d / SUB) as u8;
    }
    pal_of
}

/// Interleaved: dims cycle through palettes 0,1,2,3,0,1,2,3,...
fn interleaved_pal_of() -> [u8; HEAD_DIM] {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (d % N_PALETTE) as u8;
    }
    pal_of
}

/// Reversed: palette 3 gets dims 0..31, palette 2 gets 32..63, etc.
fn reversed_pal_of() -> [u8; HEAD_DIM] {
    let mut pal_of = [0u8; HEAD_DIM];
    for d in 0..HEAD_DIM {
        pal_of[d] = (3 - d / SUB) as u8;
    }
    pal_of
}

/// Deterministic pseudo-random balanced assignment using a simple LCG shuffle
fn shuffled_pal_of(seed: u64) -> [u8; HEAD_DIM] {
    // Start with identity, then Fisher-Yates shuffle with LCG
    let mut pal_of = identity_pal_of();
    let mut rng = seed;
    for i in (1..HEAD_DIM).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        pal_of.swap(i, j);
    }
    pal_of
}

fn validate_all_tids(pm: &[u8; 32], label: &str) {
    for tid in 0..HEAD_DIM {
        let (ref_p, ref_rank) = reference_rank(pm, tid);
        let (opt_p, opt_rank) = optimized_rank(pm, tid);
        assert_eq!(
            (ref_p, ref_rank),
            (opt_p, opt_rank),
            "{label}: tid={tid} expected p={ref_p} rank={ref_rank}, got p={opt_p} rank={opt_rank}"
        );
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_identity_palette() {
    let pm = make_pal_map(&identity_pal_of());
    validate_all_tids(&pm, "identity");
}

#[test]
fn test_interleaved_palette() {
    let pm = make_pal_map(&interleaved_pal_of());
    validate_all_tids(&pm, "interleaved");
}

#[test]
fn test_reversed_palette() {
    let pm = make_pal_map(&reversed_pal_of());
    validate_all_tids(&pm, "reversed");
}

#[test]
fn test_many_shuffled_palettes() {
    for seed in 0..1000u64 {
        let pal_of = shuffled_pal_of(seed);
        let pm = make_pal_map(&pal_of);
        validate_all_tids(&pm, &format!("shuffle(seed={seed})"));
    }
}

#[test]
fn test_boundary_tids() {
    // Specifically check word boundaries (tid = 0, 15, 16, 31, 32, ..., 127)
    let pm = make_pal_map(&shuffled_pal_of(42));
    for &tid in &[
        0, 1, 15, 16, 17, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 111, 112, 127,
    ] {
        let (ref_p, ref_rank) = reference_rank(&pm, tid);
        let (opt_p, opt_rank) = optimized_rank(&pm, tid);
        assert_eq!((ref_p, ref_rank), (opt_p, opt_rank), "boundary: tid={tid}");
    }
}

#[test]
fn test_pal_match_mask_exhaustive() {
    // Verify pal_match_mask against manual extraction for all 4 palettes
    // on a known word
    let pm = make_pal_map(&interleaved_pal_of());
    let words: [u32; 8] = unsafe { *(pm.as_ptr() as *const [u32; 8]) };

    for word_idx in 0..8 {
        let w = words[word_idx];
        for p in 0..N_PALETTE {
            let mask = pal_match_mask(w, p);
            // Verify each bit
            for k in 0..16 {
                let global_d = word_idx * 16 + k;
                let expected = pal_map_get(&pm, global_d) == p;
                let got = (mask >> (k * 2)) & 1 == 1;
                assert_eq!(
                    expected, got,
                    "pal_match_mask: word={word_idx} p={p} k={k} (dim={global_d})"
                );
            }
        }
    }
}

#[test]
fn test_rank_gives_valid_offset() {
    // For every pal_map, p*SUB + rank must be unique across all tids
    for seed in 0..100u64 {
        let pal_of = shuffled_pal_of(seed);
        let pm = make_pal_map(&pal_of);
        let mut offsets = std::collections::HashSet::new();
        for tid in 0..HEAD_DIM {
            let (p, rank) = optimized_rank(&pm, tid);
            let offset = p * SUB + rank;
            assert!(
                offset < HEAD_DIM,
                "seed={seed} tid={tid}: offset {offset} >= HEAD_DIM"
            );
            assert!(
                offsets.insert(offset),
                "seed={seed} tid={tid}: duplicate offset {offset}"
            );
        }
        assert_eq!(offsets.len(), HEAD_DIM);
    }
}
