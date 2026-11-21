//! The next-layer predictor interface and the scoring rules evaluated against it.
//!
//! The `Predictor` trait mirrors the observe, predict and reset lifecycle of the
//! production predictor, so one replay drives any candidate. `Popularity` is the
//! causal baseline: it predicts the experts most often routed so far, excluding
//! the current set. `MatrixPredictor` holds transition counts and scores them
//! under a chosen `Scoring` rule, from raw counts through row-normalized
//! conditionals and a lift over the target's marginal to a Wilson lower bound
//! that discounts single observations. Arrival training learns only the
//! transitions whose target was not already active.

use super::*;

/// A next-layer predictor under evaluation.  Mirrors the production
/// observe/predict/reset lifecycle so the same replay drives any candidate.
pub(super) trait Predictor {
    fn reset_pass(&mut self);
    fn observe(&mut self, layer: usize, experts: &[usize]);
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize>;
}

/// The production transition-matrix predictor.
impl Predictor for TransitionMatrix {
    fn reset_pass(&mut self) {
        TransitionMatrix::reset_pass(self)
    }
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        TransitionMatrix::observe(self, layer, experts)
    }
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        self.predict_topk(layer, experts, k)
    }
}

/// Causal per-layer popularity baseline: predict the most frequently routed
/// experts at the successor layer so far, excluding the current active set.
pub(super) struct Popularity {
    /// `freq[layer][expert]` running counts from observed passes.
    pub(super) freq: Vec<Vec<u32>>,
}

impl Popularity {
    pub(super) fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            freq: vec![vec![0u32; num_experts]; num_layers],
        }
    }
}

impl Predictor for Popularity {
    fn reset_pass(&mut self) {}
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        if let Some(row) = self.freq.get_mut(layer) {
            for &e in experts {
                if let Some(c) = row.get_mut(e) {
                    *c += 1;
                }
            }
        }
    }
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        let next = layer + 1;
        let Some(row) = self.freq.get(next) else {
            return vec![];
        };
        let active: HashSet<usize> = experts.iter().copied().collect();
        let mut cand: Vec<(usize, u32)> = row
            .iter()
            .enumerate()
            .filter(|(idx, &c)| c > 0 && !active.contains(idx))
            .map(|(idx, &c)| (idx, c))
            .collect();
        cand.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        cand.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
}

/// Scoring rule for [`MatrixPredictor`] — the axis we are tuning.
#[derive(Clone, Copy)]
pub(super) enum Scoring {
    /// Σ_from counts[from][to]. Identical to the production predictor; tends
    /// toward globally-popular targets because frequent sources dominate.
    Raw,
    /// Σ_from P(to|from). Row-normalized: each source contributes the sharpness
    /// of its own conditional, not its raw frequency.
    Conditional,
    /// Σ_from P(to|from) / P(to)^alpha. Conditional lift over the target's
    /// marginal — demotes globally-popular (already-cached) targets, surfacing
    /// experts *specifically* implied by the current routing.
    Pmi { alpha: f64 },
    /// Σ_from (counts + alpha·P(to)) / (row_total + alpha). Dirichlet smoothing
    /// of the conditional toward the marginal — stabilises undertrained cells.
    Dirichlet { alpha: f64 },
    /// Σ_from Wilson lower-confidence-bound of P(to|from) at z=`z`.  Suppresses
    /// single-observation noise: a cell seen once scores far below one seen 50×.
    Wilson { z: f64 },
}

/// Online transition predictor with a configurable scoring rule, used to
/// prototype scoring changes against the trace before porting the winner to
/// the production [`TransitionMatrix`].
#[derive(Clone)]
pub(super) struct MatrixPredictor {
    pub(super) e: usize,
    pub(super) num_layers: usize,
    /// Source layers per matrix (layer resolution): 1 = per-pair, large = shared.
    pub(super) group: usize,
    /// Per-pass count decay (momentum): <1.0 weights recent tokens more.
    pub(super) decay: f64,
    /// Train only on "arrivals" — transitions whose target is NOT in the source
    /// layer's set (the cold experts a prefetch must actually cover).
    pub(super) arrivals_only: bool,
    pub(super) counts: Vec<Vec<f64>>,    // [group][from*e + to]
    pub(super) row_total: Vec<Vec<f64>>, // [group][from] = Σ_to counts
    pub(super) col_total: Vec<Vec<f64>>, // [group][to]   = Σ_from counts
    pub(super) grp_total: Vec<f64>,      // [group] = Σ counts
    pub(super) total_obs: u64,
    pub(super) min_obs: u64,
    pub(super) prev: Option<(usize, Vec<usize>, Vec<f64>)>,
    pub(super) scoring: Scoring,
}

impl MatrixPredictor {
    pub(super) fn new(num_layers: usize, num_experts: usize, scoring: Scoring) -> Self {
        Self::new_cfg(num_layers, num_experts, scoring, 1, 1.0)
    }

    /// `group` = source layers sharing one matrix (1 = per-pair).  `decay` =
    /// per-pass multiplicative count decay (1.0 = none / stationary).
    pub(super) fn new_cfg(
        num_layers: usize,
        num_experts: usize,
        scoring: Scoring,
        group: usize,
        decay: f64,
    ) -> Self {
        let group = group.max(1);
        let pairs = num_layers.saturating_sub(1);
        let groups = pairs.div_ceil(group).max(1);
        Self {
            e: num_experts,
            num_layers,
            group,
            decay,
            arrivals_only: false,
            counts: vec![vec![0.0; num_experts * num_experts]; groups],
            row_total: vec![vec![0.0; num_experts]; groups],
            col_total: vec![vec![0.0; num_experts]; groups],
            grp_total: vec![0.0; groups],
            total_obs: 0,
            min_obs: 64,
            prev: None,
            scoring,
        }
    }

    /// Matrix index for a transition whose *source* is `src_layer`.
    #[inline]
    pub(super) fn gidx(&self, src_layer: usize) -> usize {
        (src_layer / self.group).min(self.counts.len() - 1)
    }
}

impl Predictor for MatrixPredictor {
    fn reset_pass(&mut self) {
        self.prev = None;
        if self.decay < 1.0 {
            let d = self.decay;
            for g in 0..self.counts.len() {
                self.counts[g].iter_mut().for_each(|c| *c *= d);
                self.row_total[g].iter_mut().for_each(|c| *c *= d);
                self.col_total[g].iter_mut().for_each(|c| *c *= d);
                self.grp_total[g] *= d;
            }
        }
    }

    fn observe(&mut self, layer: usize, experts: &[usize]) {
        let uni = vec![1.0f64; experts.len()];
        self.observe_mass(layer, experts, &uni);
    }

    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        self.score_topk(layer, experts, k, self.scoring)
    }
}

impl MatrixPredictor {
    /// Mass-weighted observation: each transition (from→to) is credited
    /// `mass[from] · mass[to]` instead of 1.  Uniform masses reduce to `observe`.
    pub(super) fn observe_mass(&mut self, layer: usize, experts: &[usize], masses: &[f64]) {
        if let Some((prev_layer, prev_experts, prev_mass)) = self.prev.take() {
            if layer == prev_layer + 1 && prev_layer + 1 < self.num_layers {
                let e = self.e;
                let g = self.gidx(prev_layer);
                for (i, &from) in prev_experts.iter().enumerate() {
                    if from >= e {
                        continue;
                    }
                    let wf = prev_mass[i];
                    let base = from * e;
                    for (j, &to) in experts.iter().enumerate() {
                        if to >= e {
                            continue;
                        }
                        // Arrival-specialised: skip targets already in the source.
                        if self.arrivals_only && prev_experts.contains(&to) {
                            continue;
                        }
                        let w = wf * masses[j];
                        self.counts[g][base + to] += w;
                        self.row_total[g][from] += w;
                        self.col_total[g][to] += w;
                        self.grp_total[g] += w;
                        self.total_obs += 1;
                    }
                }
            }
        }
        self.prev = Some((layer, experts.to_vec(), masses.to_vec()));
    }
}

/// Wilson lower confidence bound on a proportion `p` from `n` observations.
pub(super) fn wilson_lb(p: f64, n: f64, z: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let z2 = z * z;
    ((p + z2 / (2.0 * n)) - z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / (1.0 + z2 / n)
}

impl MatrixPredictor {
    /// Full score vector for predicting `layer+1` under `scoring`.  None if the
    /// matrix is not warm or there is no successor layer.
    pub(super) fn score_vec(
        &self,
        layer: usize,
        experts: &[usize],
        scoring: Scoring,
    ) -> Option<Vec<f64>> {
        if self.total_obs < self.min_obs || layer + 1 >= self.num_layers {
            return None;
        }
        let e = self.e;
        let g = self.gidx(layer);
        let matrix = &self.counts[g];
        let row_total = &self.row_total[g];
        let col_total = &self.col_total[g];
        let ptot = self.grp_total[g].max(1.0);
        // Dirichlet smooths every target (even unobserved cells), so it cannot
        // skip zero cells like the multiplicative rules.
        let dense = matches!(scoring, Scoring::Dirichlet { .. });

        let mut scores = vec![0.0f64; e];
        for &from in experts {
            if from >= e {
                continue;
            }
            let base = from * e;
            let rt = row_total[from];
            if rt <= 0.0 {
                continue;
            }
            for to in 0..e {
                let c = matrix[base + to];
                if c <= 0.0 && !dense {
                    continue;
                }
                let cond = c / rt; // P(to | from)
                scores[to] += match scoring {
                    Scoring::Raw => c,
                    Scoring::Conditional => cond,
                    Scoring::Pmi { alpha } => {
                        let p_to = (col_total[to] / ptot).max(1e-9); // P(to)
                        cond / p_to.powf(alpha)
                    }
                    Scoring::Dirichlet { alpha } => {
                        let p_to = col_total[to] / ptot;
                        (c + alpha * p_to) / (rt + alpha)
                    }
                    Scoring::Wilson { z } => wilson_lb(cond, rt, z),
                };
            }
        }
        Some(scores)
    }

    /// Score the top-`k` successors under an arbitrary `scoring` rule.
    pub(super) fn score_topk(
        &self,
        layer: usize,
        experts: &[usize],
        k: usize,
        scoring: Scoring,
    ) -> Vec<usize> {
        match self.score_vec(layer, experts, scoring) {
            Some(s) => top_k_excluding(&s, experts, k),
            None => vec![],
        }
    }
}

/// Train a predictor by observing every pass of `trace` for `epochs` passes.
pub(super) fn train_predictor<P: Predictor>(p: &mut P, trace: &Trace, epochs: usize) {
    for _ in 0..epochs {
        for pass in &trace.passes {
            p.reset_pass();
            for (l, e) in &pass.layers {
                p.observe(*l, e);
            }
        }
    }
}

/// Shared bounded top-k selection over a score vector, excluding `active`.
pub(super) fn top_k_excluding(scores: &[f64], active: &[usize], k: usize) -> Vec<usize> {
    let better = |a: (usize, f64), b: (usize, f64)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
    let mut top: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
    for (idx, &s) in scores.iter().enumerate() {
        if s <= 0.0 || active.contains(&idx) {
            continue;
        }
        if top.len() < k {
            top.push((idx, s));
        } else if better((idx, s), top[k - 1]) {
            top[k - 1] = (idx, s);
        } else {
            continue;
        }
        let mut j = top.len() - 1;
        while j > 0 && better(top[j], top[j - 1]) {
            top.swap(j, j - 1);
            j -= 1;
        }
    }
    top.into_iter().map(|(idx, _)| idx).collect()
}
