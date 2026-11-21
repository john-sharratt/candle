//! The kernel's candidate cap and floor, and the typical-acceptance rule for speculative verify rows.
//!
//! `KERNEL_MAX_TOP_K` is the kernel's candidate cap, and `KERNEL_DEAD_ZONE` is its candidate floor.
//! `typical_accepts_row` applies the typical-acceptance rule to one row of logits and decides whether the row's draft is accepted.

use super::*;

/// The kernel's candidate cap (`batched_sampling::MAX_TOP_K`): with top-k off
/// it still samples the best this many.
pub(super) const KERNEL_MAX_TOP_K: usize = 256;

/// The kernel's candidate floor (`radix_select_logit_threshold`'s
/// `DEAD_ZONE`): no candidate sits more than this far below the row's best.
pub(super) const KERNEL_DEAD_ZONE: f32 = 50.0;

/// The typical-acceptance rule over one row of logits: true when `draft`'s
/// probability exceeds `min(ε, δ·e^(−H))`. The CPU twin of the kernel's
/// `typical_accepts`, measured over the distribution the kernel samples — the
/// top-k candidates (at most [`KERNEL_MAX_TOP_K`], none more than
/// [`KERNEL_DEAD_ZONE`] below the best), softmaxed at `temperature`, cut to
/// the nucleus at `top_p`, renormalised. A draft outside it has probability
/// zero.
pub(super) fn typical_accepts_row(
    logits: &[f32],
    temperature: f32,
    top_k: i32,
    top_p: f32,
    draft: u32,
    typical: TypicalAcceptance,
) -> bool {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return false;
    }
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, l)| l >= max - KERNEL_DEAD_ZONE)
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = if top_k > 0 {
        (top_k as usize).min(KERNEL_MAX_TOP_K)
    } else {
        KERNEL_MAX_TOP_K
    };
    candidates.truncate(k);

    let inv_t = 1.0 / temperature;
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&(_, l)| ((l - max) * inv_t).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut nucleus = candidates.len();
    if top_p < 1.0 {
        let mut cumsum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cumsum += w / total;
            if cumsum >= top_p {
                nucleus = i + 1;
                break;
            }
        }
    }
    let kept: f32 = weights[..nucleus].iter().sum();
    let mut entropy = 0.0f32;
    let mut p_draft = 0.0f32;
    for (&(id, _), w) in candidates[..nucleus].iter().zip(&weights) {
        let p = w / kept;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        if id == draft as usize {
            p_draft = p;
        }
    }
    p_draft > typical.epsilon.min(typical.delta * (-entropy).exp())
}
