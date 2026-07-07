//! Folded-provenance `z × margin` late-fusion retrieval — the belief scorer.
//!
//! Scores the compact folded provenance signature (see [`super::fold_provenance`]
//! and `docs/tool_selection_provenance_results.md` §23) — `PROV_HEADS_PER_LAYER`-head
//! layer groups (3 groups × 4 heads × 128 bits for the locked design). Each
//! **layer-group** is scored independently and combined by **late fusion**: for
//! each query token, each group votes for the **case** (tool) it best matches,
//! weighted by two factors that mute different kinds of noise:
//!
//! - **z-score confidence** `(best − mean)/std` of the group's agreement
//!   distribution — a discriminative group's match is a many-σ outlier (strong),
//!   a noise group's best is only the expected-max of random (self-muted). This
//!   mutes non-discriminative **groups**.
//! - **margin** `(best_case − second_best_case)` — a token where one tool sharply
//!   wins votes strongly; a generic token where the whole family ties votes ~0.
//!   This mutes non-discriminative **tokens** (§80/§81).
//!
//! The vote is `z × margin`; the two gates are orthogonal (group vs token noise).
//!
//! # Needle gate (§82)
//!
//! Finally, the belief is summed over only the **top [`NEEDLE_KEEP_FRAC`] of query
//! tokens by vote magnitude** — the sparse discriminative tokens (the *needle*)
//! carry the signal, the diffuse remainder (the *haystack* of generic/stale
//! tokens) is dropped. This is **position-independent** (unlike a recency decay,
//! it finds the needle wherever it sits in the window) and self-defensive: a wide
//! probe window can't pull in wrong sections through its diffuse tail, because
//! that tail never clears the gate.

use rayon::prelude::*;

use super::WideQSig;

/// KV heads per folded layer-group — equals `n_kv_head` (heads are kept separate
/// through the fold). Mirrors [`super::wide_sig::PROV_HEADS_PER_LAYER`].
const HEADS_PER_GROUP: usize = super::wide_sig::PROV_HEADS_PER_LAYER;

/// Needle-gate keep-fraction: the belief is summed over only the top
/// `NEEDLE_KEEP_FRAC` of query tokens by vote magnitude. Validated at 0.25 in
/// §82 (holds 100% Tool-3/Tool-5, sharpens Tool-1, on ~75% fewer effective
/// tokens) — and, being magnitude- not position-keyed, generalizes to content
/// windows where the relevant reference is a sparse needle among boilerplate.
const NEEDLE_KEEP_FRAC: f32 = 0.25;

/// Sign-agreement between two layer-group signatures = `popcount(XNOR)` over the
/// group's words. Fast-paths the locked 8-word (4-head × 128-bit) group width so
/// it vectorizes; falls back to a generic loop for other widths.
#[inline]
fn group_agreement(a: &[u64], b: &[u64]) -> u32 {
    if a.len() == 8 && b.len() == 8 {
        let mut s = 0u32;
        for k in 0..8 {
            s += (!(a[k] ^ b[k])).count_ones();
        }
        s
    } else {
        let n = a.len().min(b.len());
        let mut s = 0u32;
        for k in 0..n {
            s += (!(a[k] ^ b[k])).count_ones();
        }
        s
    }
}

/// Score `n_cases` corpus cases against a `query` window of folded provenance
/// signatures via `z × margin` late-fusion voting over the signature's layer-groups.
///
/// For each query token and each layer-group, the group finds the **best
/// agreement per case** over the gallery, identifies the leading case and the
/// runner-up, and casts a vote weighted by the product of the leader's **z-score
/// confidence** `(best − mean)/std` (an outlier vs the group's whole agreement
/// distribution) and its **margin** over the runner-up case (`best − second`).
/// Votes tally per case; the returned `Vec<f32>` is the per-case vote total
/// (higher = more relevant), indexed `0..n_cases`. `gallery_case[j]` is the case
/// index of gallery token `j` (both slices must be the same length).
///
/// Parallel over query tokens (each token's per-group scan is independent). This
/// is a flat O(gallery) scan — fine for the substrate-scale galleries here; large
/// corpora want an index (LSH/kNN) or the GPU Hamming path.
pub fn score_provenance_late_fusion(
    query: &[WideQSig],
    gallery: &[&WideQSig],
    gallery_case: &[u32],
    n_cases: usize,
) -> Vec<f32> {
    let shape: Option<&WideQSig> = query.first().or_else(|| gallery.first().copied());
    let Some(shape) = shape else {
        return vec![0.0; n_cases];
    };
    let wph = shape.words_per_head();
    let n_heads = shape.n_heads as usize;
    if wph == 0
        || n_heads < HEADS_PER_GROUP
        || gallery.is_empty()
        || gallery.len() != gallery_case.len()
    {
        return vec![0.0; n_cases];
    }
    let n_groups = n_heads / HEADS_PER_GROUP;
    let gw = HEADS_PER_GROUP * wph; // words per layer-group
    let need = n_groups * gw;
    let n_gal = gallery.len() as f32;

    // Each query token contributes, per group, one `z × margin` vote for the
    // leading case (the tool whose best-matching gallery token agrees most).
    let per_query: Vec<Vec<(usize, f32)>> = query
        .par_iter()
        .filter(|q| q.words.len() >= need)
        .map(|q| {
            let mut case_max = vec![0u32; n_cases];
            let mut out = Vec::with_capacity(n_groups);
            for g in 0..n_groups {
                let base = g * gw;
                let qg = &q.words[base..base + gw];
                for m in case_max.iter_mut() {
                    *m = 0;
                }
                let (mut sum, mut sumsq) = (0u64, 0u64);
                for (j, cand) in gallery.iter().enumerate() {
                    if cand.words.len() < base + gw {
                        continue;
                    }
                    let ag = group_agreement(qg, &cand.words[base..base + gw]);
                    let c = gallery_case[j] as usize;
                    if c < n_cases && ag > case_max[c] {
                        case_max[c] = ag;
                    }
                    sum += ag as u64;
                    sumsq += (ag as u64) * (ag as u64);
                }
                // Leader and runner-up case agreements → margin.
                let (mut top1, mut top1c, mut top2) = (0u32, usize::MAX, 0u32);
                for (c, &m) in case_max.iter().enumerate() {
                    if m > top1 {
                        top2 = top1;
                        top1 = m;
                        top1c = c;
                    } else if m > top2 {
                        top2 = m;
                    }
                }
                if top1c != usize::MAX {
                    let mean = sum as f32 / n_gal;
                    let var = (sumsq as f32 / n_gal - mean * mean).max(1e-6);
                    let z = ((top1 as f32 - mean) / var.sqrt()).max(0.0);
                    let margin = top1.saturating_sub(top2) as f32;
                    out.push((top1c, z * margin));
                }
            }
            out
        })
        .collect();

    // Needle gate: keep only the top `NEEDLE_KEEP_FRAC` of query tokens by total
    // vote magnitude, so a sparse discriminative signal dominates and the diffuse
    // remainder is dropped — position-independently. See the module docs (§82).
    if per_query.is_empty() {
        return vec![0.0; n_cases];
    }
    let mags: Vec<f32> = per_query
        .iter()
        .map(|contribs| contribs.iter().map(|(_, v)| *v).sum())
        .collect();
    let keep_n = ((NEEDLE_KEEP_FRAC * mags.len() as f32).ceil() as usize).clamp(1, mags.len());
    let mut sorted = mags.clone();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let thresh = sorted[keep_n - 1];

    let mut votes = vec![0f32; n_cases];
    for (contribs, &mag) in per_query.iter().zip(&mags) {
        if mag >= thresh {
            for &(case, v) in contribs {
                votes[case] += v;
            }
        }
    }
    votes
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A folded signature: 12 heads (3 groups × 4), head_dim 128 → 24 u64 words.
    fn folded_sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    #[test]
    fn late_fusion_votes_for_best_matching_case() {
        let a = folded_sig(0xAAAA_AAAA_AAAA_AAAA);
        let b = folded_sig(0x5555_5555_5555_5555); // bitwise complement of a
        let gallery = [&a, &b];
        let cases = [0u32, 1];

        // Query = A. Per group: case-0 max = A·A = 512, case-1 max = A·B = 0.
        // mean=256, std=256, z=(512-256)/256 = 1.0; margin = 512 − 0 = 512; the
        // vote is z×margin = 512 per group → 3 groups → votes[0] = 1536.
        let votes = score_provenance_late_fusion(&[a.clone()], &gallery, &cases, 2);
        assert!(
            (votes[0] - 1536.0).abs() < 1e-2,
            "case 0 exact match: {votes:?}"
        );
        assert_eq!(votes[1], 0.0, "case 1 (complement) gets no vote");

        // Query = B → case 1 wins symmetrically.
        let votes = score_provenance_late_fusion(&[b.clone()], &gallery, &cases, 2);
        assert!((votes[1] - 1536.0).abs() < 1e-2);
        assert_eq!(votes[0], 0.0);
    }

    #[test]
    fn late_fusion_empty_is_zero() {
        assert_eq!(score_provenance_late_fusion(&[], &[], &[], 3), vec![0.0; 3]);
        // Empty gallery with a query still returns zeros (nothing to vote for).
        let a = folded_sig(0xFF);
        assert_eq!(
            score_provenance_late_fusion(&[a], &[], &[], 2),
            vec![0.0; 2]
        );
    }
}
