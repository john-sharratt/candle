//! Contiguous packed gallery + flat scan — the cache-friendly, GPU-ready form of
//! the belief scan.
//!
//! [`score_slots`](super::score_slots) flattens the gallery into a `Vec<&WideQSig>`
//! on every call, and [`score_provenance_late_fusion`](super::score_provenance_late_fusion)
//! then chases those pointers (each token's `words` is its own heap allocation)
//! **once per layer-group** — a scattered, thrice-repeated memory sweep. This
//! module packs the same gallery into one contiguous `Vec<u64>` (token-major) plus
//! a parallel case array, and scans it with the layer-group loop **interchanged**
//! so the gallery is read exactly once. The math is unchanged — the per-query
//! `(case, z × margin)` contributions and the needle gate are the same as the
//! reference — so the output is **bit-identical** (see the parity tests), just
//! produced from a linear, prefetchable, SIMD/GPU-amenable layout.

use rayon::prelude::*;

use super::scan::{group_agreement, needle_gate_tally, HEADS_PER_GROUP};
use super::WideQSig;

/// A gallery packed into one contiguous buffer, scan-ready.
///
/// Every token occupies `wpt` contiguous `u64` at `words[i*wpt .. (i+1)*wpt]`
/// (token-major, all layer-groups adjacent), labelled by `case[i]`. Only tokens
/// of the uniform folded width `wpt` are admitted (the folded `WideQSig` is always
/// that width; any short/partial token is dropped, matching the reference scan's
/// per-group length guard for real data). This is the CPU form; the same bytes
/// DMA to VRAM unchanged for the GPU scan.
#[derive(Debug, Clone, Default)]
pub struct PackedGallery {
    /// `n_tokens × wpt` sign words, token-major.
    words: Vec<u64>,
    /// `n_tokens` case (slot) ids, one per packed token.
    case: Vec<u32>,
    /// Words per token = `n_heads × words_per_head` = `n_groups × gw`.
    wpt: usize,
    /// Heads spanned by each signature (folded: `n_groups × HEADS_PER_GROUP`).
    n_heads: usize,
    /// Words per head (head_dim / 64).
    wph: usize,
    /// Number of cases (slots) the scan votes over.
    n_cases: usize,
}

impl PackedGallery {
    /// Pack `windows` (each a slice of tokens) into the contiguous form, labelling
    /// every token in window `wi` with `slots[wi]`. Windows whose slot is out of
    /// range are dropped wholesale — identical admission to
    /// [`score_slots`](super::score_slots).
    pub fn from_windows(windows: &[&[WideQSig]], slots: &[usize], n_cases: usize) -> Self {
        // Shape from the first token of the first non-empty window.
        let shape = windows.iter().flat_map(|w| w.iter()).next();
        let (n_heads, wph) = shape
            .map(|s| (s.n_heads as usize, s.words_per_head()))
            .unwrap_or((0, 0));
        let wpt = n_heads * wph;

        let total: usize = windows.iter().map(|w| w.len()).sum();
        let mut words: Vec<u64> = Vec::with_capacity(total * wpt);
        let mut case: Vec<u32> = Vec::with_capacity(total);
        if wpt == 0 {
            return Self {
                words,
                case,
                wpt,
                n_heads,
                wph,
                n_cases,
            };
        }
        for (wi, w) in windows.iter().enumerate() {
            let slot = slots.get(wi).copied().unwrap_or(usize::MAX);
            if slot >= n_cases {
                continue;
            }
            for t in w.iter() {
                // Uniform width only — the folded signature is always `wpt` wide.
                if t.words.len() != wpt {
                    continue;
                }
                words.extend_from_slice(&t.words);
                case.push(slot as u32);
            }
        }
        Self {
            words,
            case,
            wpt,
            n_heads,
            wph,
            n_cases,
        }
    }

    /// Number of packed gallery tokens.
    pub fn n_tokens(&self) -> usize {
        self.case.len()
    }

    /// Total packed size in bytes (the exact VRAM footprint) — sign words + case
    /// labels.
    pub fn byte_len(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>() + self.case.len() * std::mem::size_of::<u32>()
    }

    /// Contiguous sign-word buffer (token-major) — the block that DMAs to VRAM.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Per-token case labels, parallel to [`Self::words`] by token.
    pub fn case(&self) -> &[u32] {
        &self.case
    }

    /// Words per token (`n_heads × words_per_head` = `n_groups × gw`).
    pub fn wpt(&self) -> usize {
        self.wpt
    }

    /// Heads spanned by each signature (folded: `n_groups × HEADS_PER_GROUP`).
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Words per head (head_dim / 64).
    pub fn wph(&self) -> usize {
        self.wph
    }

    /// Number of cases (slots) voted over.
    pub fn n_cases(&self) -> usize {
        self.n_cases
    }
}

/// Scan `query` against a [`PackedGallery`], returning the per-case `z × margin`
/// vote totals — **bit-identical** to
/// [`score_slots`](super::score_slots)/[`score_provenance_late_fusion`](super::score_provenance_late_fusion)
/// on the same gallery, but reading the contiguous buffer once (layer-group loop
/// interchanged) instead of chasing scattered pointers three times.
pub fn score_packed(query: &[WideQSig], gallery: &PackedGallery) -> Vec<f32> {
    let n_cases = gallery.n_cases;
    let n_tokens = gallery.n_tokens();
    if n_tokens == 0 || gallery.n_heads < HEADS_PER_GROUP {
        return vec![0.0; n_cases];
    }
    let n_groups = gallery.n_heads / HEADS_PER_GROUP;
    let gw = HEADS_PER_GROUP * gallery.wph; // words per layer-group
    let wpt = gallery.wpt;
    let need = wpt; // n_groups × gw
    let n_gal = n_tokens as f32;
    let words = &gallery.words;
    let case = &gallery.case;

    // Each query token contributes, per group, one `z × margin` vote for the
    // leading case — same as the reference, but computed in ONE pass over the
    // gallery (all groups at once) instead of one pass per group.
    let per_query: Vec<Vec<(usize, f32)>> = query
        .par_iter()
        .filter(|q| q.words.len() >= need)
        .map(|q| {
            // `n_groups` interleaved accumulators, so the gallery is touched once.
            let mut case_max = vec![0u32; n_groups * n_cases]; // [group][case]
            let mut sum = vec![0u64; n_groups];
            let mut sumsq = vec![0u64; n_groups];
            for j in 0..n_tokens {
                let tok = &words[j * wpt..j * wpt + wpt];
                let c = case[j] as usize;
                for grp in 0..n_groups {
                    let gb = grp * gw;
                    let ag = group_agreement(&q.words[gb..gb + gw], &tok[gb..gb + gw]);
                    let cm = &mut case_max[grp * n_cases + c];
                    if ag > *cm {
                        *cm = ag;
                    }
                    sum[grp] += ag as u64;
                    sumsq[grp] += (ag as u64) * (ag as u64);
                }
            }
            let mut out = Vec::with_capacity(n_groups);
            for grp in 0..n_groups {
                let cmax = &case_max[grp * n_cases..(grp + 1) * n_cases];
                // Leader and runner-up case agreements → margin.
                let (mut top1, mut top1c, mut top2) = (0u32, usize::MAX, 0u32);
                for (c, &m) in cmax.iter().enumerate() {
                    if m > top1 {
                        top2 = top1;
                        top1 = m;
                        top1c = c;
                    } else if m > top2 {
                        top2 = m;
                    }
                }
                if top1c != usize::MAX {
                    let mean = sum[grp] as f32 / n_gal;
                    let var = (sumsq[grp] as f32 / n_gal - mean * mean).max(1e-6);
                    let z = ((top1 as f32 - mean) / var.sqrt()).max(0.0);
                    let margin = top1.saturating_sub(top2) as f32;
                    out.push((top1c, z * margin));
                }
            }
            out
        })
        .collect();

    needle_gate_tally(&per_query, n_cases)
}

#[cfg(test)]
mod tests {
    // Same fixture shape as `provenance::gpu`'s tests: a vec of per-tool token
    // windows.
    #![allow(clippy::useless_vec)]

    use super::*;
    use crate::provenance::{score_slots, WideQSig};

    /// A folded signature: 12 heads (3 groups × 4), head_dim 128 → 24 u64 words.
    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// Build a small synthetic gallery + probe and assert the packed scan is
    /// **exactly** equal (bit-for-bit) to the reference `score_slots`.
    fn assert_parity(
        windows: &[Vec<WideQSig>],
        slots: &[usize],
        probe: &[WideQSig],
        n_cases: usize,
    ) {
        let wref: Vec<&[WideQSig]> = windows.iter().map(|w| w.as_slice()).collect();
        let reference = score_slots(probe, &wref, slots, n_cases);
        let packed = PackedGallery::from_windows(&wref, slots, n_cases);
        let got = score_packed(probe, &packed);
        assert_eq!(
            got, reference,
            "packed scan must be bit-identical to the reference"
        );
    }

    #[test]
    fn parity_two_distinct_cases() {
        // Case 0 = 0xAAAA…, case 1 = its complement 0x5555…; probe matches case 0.
        let windows = vec![
            vec![sig(0xAAAA_AAAA_AAAA_AAAA)],
            vec![sig(0x5555_5555_5555_5555)],
        ];
        assert_parity(&windows, &[0, 1], &[sig(0xAAAA_AAAA_AAAA_AAAA)], 2);
    }

    #[test]
    fn parity_multi_token_windows_and_probe() {
        // Windows with several tokens each, a multi-token probe, three cases.
        let windows = vec![
            vec![sig(0xFF00_FF00_FF00_FF00), sig(0x0F0F_0F0F_0F0F_0F0F)],
            vec![sig(0x5555_5555_5555_5555)],
            vec![
                sig(0xFFFF_FFFF_0000_0000),
                sig(0x00FF_00FF_00FF_00FF),
                sig(0x1234_5678_9ABC_DEF0),
            ],
        ];
        let probe = vec![
            sig(0xFF00_FF00_FF00_FF00),
            sig(0x1234_5678_9ABC_DEF0),
            sig(0x0),
        ];
        assert_parity(&windows, &[0, 1, 2], &probe, 3);
    }

    #[test]
    fn parity_multiple_windows_same_case() {
        // Several windows mapping to the same slot (as a tool with many
        // calibration turns would) plus an out-of-range slot that must be dropped.
        let windows = vec![
            vec![sig(0xAAAA_AAAA_AAAA_AAAA)],
            vec![sig(0xABAB_ABAB_ABAB_ABAB)],
            vec![sig(0x5555_5555_5555_5555)],
            vec![sig(0xDEAD_BEEF_DEAD_BEEF)], // slot 9 → out of range, dropped
        ];
        assert_parity(&windows, &[0, 0, 1, 9], &[sig(0xAAAA_AAAA_AAAA_AAAA)], 2);
    }

    #[test]
    fn empty_gallery_and_empty_probe_match_reference() {
        assert_parity(&[], &[], &[sig(0x1)], 3);
        let windows = vec![vec![sig(0xAAAA_AAAA_AAAA_AAAA)]];
        assert_parity(&windows, &[0], &[], 2);
    }

    #[test]
    fn packed_byte_len_is_exact() {
        let windows = vec![vec![sig(0x1), sig(0x2)], vec![sig(0x3)]];
        let wref: Vec<&[WideQSig]> = windows.iter().map(|w| w.as_slice()).collect();
        let packed = PackedGallery::from_windows(&wref, &[0, 1], 2);
        // 3 tokens × 24 u64 × 8 B + 3 case × 4 B.
        assert_eq!(packed.n_tokens(), 3);
        assert_eq!(packed.byte_len(), 3 * 24 * 8 + 3 * 4);
    }
}
