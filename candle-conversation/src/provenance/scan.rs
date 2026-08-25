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

use super::fusion::FusionMode;
use super::WideQSig;

/// Layer groups in a folded signature — **structural, not tuned**.
///
/// The fold emits one noise-absorbing lower group plus the top two capture
/// layers on their own; [`super::wide_sig::FoldParams::group_sizes`] is a
/// `[usize; 3]`, so the count is pinned by the type.
pub const FOLD_GROUPS: usize = 3;

/// KV heads per folded layer-group, **derived from the signature** rather than
/// read off a constant.
///
/// Heads are kept separate through the fold, so a folded signature spans
/// `FOLD_GROUPS × n_kv_head` heads and this recovers the model's `n_kv_head` by
/// division. That matters because `n_kv_head` is not 4 everywhere: it is 4 on
/// the 48-layer stack the fold was measured against and **2** on the hybrid, so
/// a hardcoded 4 reads a 6-head signature as one group of four rather than
/// three of two — the group boundaries land mid-signature and every score after
/// that is a confident number computed over the wrong bits.
///
/// Returns 0 for a signature whose head count is not a whole number of groups,
/// which callers treat as "not scorable" rather than guessing.
#[inline]
pub fn heads_per_group(n_heads: usize) -> usize {
    if n_heads == 0 || !n_heads.is_multiple_of(FOLD_GROUPS) {
        return 0;
    }
    n_heads / FOLD_GROUPS
}

/// Needle-gate keep-fraction: the belief is summed over only the top
/// `NEEDLE_KEEP_FRAC` of query tokens by vote magnitude. Validated at 0.25 in
/// §82 (holds 100% Tool-3/Tool-5, sharpens Tool-1, on ~75% fewer effective
/// tokens) — and, being magnitude- not position-keyed, generalizes to content
/// windows where the relevant reference is a sparse needle among boilerplate.
pub(super) const NEEDLE_KEEP_FRAC: f32 = 0.25;

/// Sign-agreement between two layer-group signatures = `popcount(XNOR)` over the
/// group's words. Fast-paths the locked 8-word (4-head × 128-bit) group width so
/// it vectorizes; falls back to a generic loop for other widths.
#[inline]
pub(super) fn group_agreement(a: &[u64], b: &[u64]) -> u32 {
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
    score_provenance_late_fusion_weighted(query, gallery, gallery_case, n_cases, &[])
}

/// [`score_provenance_late_fusion`] with a per-layer-group weight on each group's
/// `z × margin` vote. `group_weights[g]` scales group `g`'s contribution (missing
/// / empty ⇒ 1.0, i.e. the uniform default). Used to down-weight the noisy lower
/// layer-group for repo_map retrieval, where the cluster-identity signal lives in
/// the upper groups (see `docs/tool_selection_provenance_results.md` §83).
pub fn score_provenance_late_fusion_weighted(
    query: &[WideQSig],
    gallery: &[&WideQSig],
    gallery_case: &[u32],
    n_cases: usize,
    group_weights: &[f32],
) -> Vec<f32> {
    let shape: Option<&WideQSig> = query.first().or_else(|| gallery.first().copied());
    let Some(shape) = shape else {
        return vec![0.0; n_cases];
    };
    let wph = shape.words_per_head();
    let n_heads = shape.n_heads as usize;
    let hpg = heads_per_group(n_heads);
    if wph == 0 || hpg == 0 || gallery.is_empty() || gallery.len() != gallery_case.len() {
        return vec![0.0; n_cases];
    }
    let n_groups = FOLD_GROUPS;
    let gw = hpg * wph; // words per layer-group
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
                    let w = group_weights.get(g).copied().unwrap_or(1.0);
                    out.push((top1c, z * margin * w));
                }
            }
            out
        })
        .collect();

    needle_gate_tally(&per_query, n_cases)
}

/// Per-fold-group scoring for non-additive fusion (Concept G): one pass over
/// the gallery computing each group's `z × margin` votes, then a **per-group**
/// needle gate and tally — each group finds its own needle tokens. Returns
/// `out[g][case]`. The additive scorer's cross-group gate lives in
/// [`score_provenance_late_fusion_weighted`]; this variant exists for the
/// modes that must see the groups separately before combining.
pub fn score_provenance_late_fusion_grouped(
    query: &[WideQSig],
    gallery: &[&WideQSig],
    gallery_case: &[u32],
    n_cases: usize,
    group_weights: &[f32],
) -> Vec<Vec<f32>> {
    let shape: Option<&WideQSig> = query.first().or_else(|| gallery.first().copied());
    let Some(shape) = shape else {
        return Vec::new();
    };
    let wph = shape.words_per_head();
    let n_heads = shape.n_heads as usize;
    let hpg = heads_per_group(n_heads);
    if wph == 0 || hpg == 0 || gallery.is_empty() || gallery.len() != gallery_case.len() {
        return Vec::new();
    }
    let n_groups = FOLD_GROUPS;
    let gw = hpg * wph;
    let need = n_groups * gw;
    let n_gal = gallery.len() as f32;

    // Per query token: one `(case, z × margin)` vote PER GROUP (index = group).
    let per_query: Vec<Vec<(usize, f32)>> = query
        .par_iter()
        .filter(|q| q.words.len() >= need)
        .map(|q| {
            let mut case_max = vec![0u32; n_cases];
            let mut out = vec![(usize::MAX, 0.0f32); n_groups];
            for (g, slot) in out.iter_mut().enumerate() {
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
                    let w = group_weights.get(g).copied().unwrap_or(1.0);
                    *slot = (top1c, z * margin * w);
                }
            }
            out
        })
        .collect();

    // Per-group needle gate + tally: group g's contributions across tokens.
    (0..n_groups)
        .map(|g| {
            let contribs: Vec<Vec<(usize, f32)>> = per_query
                .iter()
                .map(|token| {
                    let (case, v) = token[g];
                    if case == usize::MAX {
                        Vec::new()
                    } else {
                        vec![(case, v)]
                    }
                })
                .collect();
            needle_gate_tally(&contribs, n_cases)
        })
        .collect()
}

/// Fused scoring entry (Concept G).
///
/// - [`FusionMode::Additive`] — the shipped single-pass scorer (cross-group
///   needle gate), bit-identical to today.
/// - Every other mode — the groups score separately
///   ([`score_provenance_late_fusion_grouped`], per-group needle gates) and
///   combine per [`FusionMode::fuse`]. For [`FusionMode::ContentGated`] that
///   is the R5-measured winner: the per-group tallies sum, gated on the gate
///   group's own tally (`g_gate > 0 ? Σ_g w_g·t_g : 0`). A full-additive
///   variant gated by a one-hot scan was measured and REJECTED (the target
///   collapsed to the bottom of the pool — results doc §25). Each per-group
///   tally equals a one-hot-weighted additive scan, so any backend with the
///   additive scan (including the GPU gallery arena) serves this mode with
///   `n_groups` scans.
pub fn score_provenance_late_fusion_fused(
    query: &[WideQSig],
    gallery: &[&WideQSig],
    gallery_case: &[u32],
    n_cases: usize,
    group_weights: &[f32],
    mode: FusionMode,
) -> Vec<f32> {
    match mode {
        FusionMode::Additive => score_provenance_late_fusion_weighted(
            query,
            gallery,
            gallery_case,
            n_cases,
            group_weights,
        ),
        _ => {
            let grouped = score_provenance_late_fusion_grouped(
                query,
                gallery,
                gallery_case,
                n_cases,
                group_weights,
            );
            if grouped.is_empty() {
                return vec![0.0; n_cases];
            }
            mode.fuse(&grouped)
        }
    }
}

/// The needle gate + per-case tally shared by the pointer-gallery
/// ([`score_provenance_late_fusion`]) and packed-gallery ([`super::score_packed`])
/// scans. Keeping it in one place guarantees the two backends produce
/// **bit-identical** votes given identical per-query contributions.
///
/// `per_query[i]` is query token `i`'s list of `(case, z × margin)` group votes.
/// Only the top [`NEEDLE_KEEP_FRAC`] of query tokens by total vote magnitude are
/// summed into the per-case result — the sparse discriminative *needle*, dropping
/// the diffuse *haystack*, position-independently (§82).
pub(super) fn needle_gate_tally(per_query: &[Vec<(usize, f32)>], n_cases: usize) -> Vec<f32> {
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

    /// **The group split follows the signature, not a constant.**
    ///
    /// A folded signature always spans three layer-groups, so its head count
    /// divided by three IS the model's `n_kv_head`: 4 on the 48-layer stack the
    /// fold was measured against, 2 on the hybrid. Reading a fixed 4 turns the
    /// hybrid's 6-head signature into one group of four — the group boundaries
    /// land mid-signature, and every score computed after that is a confident
    /// number over the wrong bits.
    #[test]
    fn heads_per_group_is_read_from_the_signature() {
        assert_eq!(heads_per_group(12), 4, "Qwen3-30B: 3 groups x 4 kv heads");
        assert_eq!(heads_per_group(6), 2, "the hybrid: 3 groups x 2 kv heads");
        assert_eq!(heads_per_group(3), 1, "a single-kv-head stack still folds");
        // Not a whole number of groups — not a folded signature, so not scorable.
        assert_eq!(heads_per_group(0), 0);
        assert_eq!(heads_per_group(4), 0, "4 heads is not 3 whole groups");
        assert_eq!(heads_per_group(7), 0);
    }

    /// The hybrid's 6-head signature scores, and scores *per group*.
    ///
    /// Under the old constant this returned zeros: `n_heads < HEADS_PER_GROUP`
    /// was false (6 > 4) so it did not bail, but `n_groups` came out 1 and the
    /// group width 4 heads, so it read two groups' worth of one signature as a
    /// single group and the remaining bits were never compared.
    #[test]
    fn a_hybrid_shaped_signature_scores_across_all_three_groups() {
        // 6 heads x 4 words/head (head_dim 256) = 24 words.
        let a = WideQSig {
            n_heads: 6,
            words: vec![0xAAAA_AAAA_AAAA_AAAA; 24],
        };
        let b = WideQSig {
            n_heads: 6,
            words: vec![0x5555_5555_5555_5555; 24],
        };
        let gallery = [&a, &b];
        let cases = [0u32, 1];

        let votes = score_provenance_late_fusion(std::slice::from_ref(&a), &gallery, &cases, 2);
        // Per group: 2 heads x 4 words x 64 bits = 512 agreeing bits, so the
        // same z x margin = 512 per group, over 3 groups.
        assert!(
            (votes[0] - 1536.0).abs() < 1e-2,
            "the hybrid signature must score across all three groups: {votes:?}"
        );
        assert_eq!(votes[1], 0.0, "the complement gets no vote");
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
        let votes = score_provenance_late_fusion(std::slice::from_ref(&a), &gallery, &cases, 2);
        assert!(
            (votes[0] - 1536.0).abs() < 1e-2,
            "case 0 exact match: {votes:?}"
        );
        assert_eq!(votes[1], 0.0, "case 1 (complement) gets no vote");

        // Query = B → case 1 wins symmetrically.
        let votes = score_provenance_late_fusion(std::slice::from_ref(&b), &gallery, &cases, 2);
        assert!((votes[1] - 1536.0).abs() < 1e-2);
        assert_eq!(votes[0], 0.0);
    }

    #[test]
    fn grouped_scan_isolates_per_group_leaders() {
        // Query = A everywhere. Gallery: an "id-spike" case agreeing only in
        // groups 1–2, and a "true" case agreeing (weaker) in every group.
        let a = 0xAAAA_AAAA_AAAA_AAAAu64;
        let q = folded_sig(a);
        let mut spike = folded_sig(a);
        for w in &mut spike.words[0..8] {
            *w = !a; // group 0: zero agreement
        }
        let mut true_case = folded_sig(a);
        // One complemented word per group → agreement 448 of 512 in each.
        true_case.words[0] = !a;
        true_case.words[8] = !a;
        true_case.words[16] = !a;

        let gallery = [&spike, &true_case];
        let cases = [0u32, 1];
        let grouped = score_provenance_late_fusion_grouped(
            std::slice::from_ref(&q),
            &gallery,
            &cases,
            2,
            &[],
        );
        assert_eq!(grouped.len(), 3);
        // Group 0: spike 0, true 448 → leader true, margin 448, z = 1 → 448.
        assert!((grouped[0][1] - 448.0).abs() < 1e-2, "{grouped:?}");
        assert_eq!(grouped[0][0], 0.0);
        // Groups 1–2: spike 512 vs true 448 → leader spike, margin 64, z = 1.
        for g in 1..3 {
            assert!((grouped[g][0] - 64.0).abs() < 1e-2, "{grouped:?}");
            assert_eq!(grouped[g][1], 0.0);
        }

        // Content-gated fusion on top: the spike (gate group 0 = 0) dies, the
        // true case keeps its gate-group score.
        let fused = score_provenance_late_fusion_fused(
            std::slice::from_ref(&q),
            &gallery,
            &cases,
            2,
            &[],
            FusionMode::ContentGated { gate_group: 0 },
        );
        assert_eq!(fused[0], 0.0, "id-spike must be killed: {fused:?}");
        assert!((fused[1] - 448.0).abs() < 1e-2, "{fused:?}");

        // Additive mode stays bit-identical to the shipped scorer.
        let additive = score_provenance_late_fusion_fused(
            std::slice::from_ref(&q),
            &gallery,
            &cases,
            2,
            &[],
            FusionMode::Additive,
        );
        let shipped = score_provenance_late_fusion(&[q], &gallery, &cases, 2);
        assert_eq!(additive, shipped);
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
