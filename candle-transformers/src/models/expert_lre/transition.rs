//! The Markov Wave promote predictor — a single online expert-transition matrix.
//!
//! For each pair of adjacent MoE layers `(L, L+1)` this learns a `[E × E]`
//! co-occurrence model purely from historical expert-routing IDs (no hidden
//! states, no extra GEMM).  Given the experts active at layer `L`, it names the
//! experts layer `L+1` will most likely need so their H2D DMA can start while
//! `L` computes — converting cold misses into overlapped loads.
//!
//! ## Model
//!
//! One matrix, learned online and **arrival-specialised**: it credits a
//! transition `from → to` only when `to` is *not* already active at the source
//! layer — i.e. the cold experts a prefetch must actually cover.  The matrix
//! accumulates over the session (in production the process *is* the session) and
//! converges to the workload's routing structure within a few thousand tokens,
//! so no cross-session prior or per-pass decay is needed.
//!
//! Successors are scored by pointwise mutual information,
//! `PMI(α) = Σ_from (c/rt) / P(to)^α` with `α = ALPHA`, which demotes
//! globally-popular (already-cached) targets in favour of experts *specifically*
//! implied by the current routing.
//!
//! ## Prefetch — confidence-gated, diversity-adaptive depth
//!
//! [`predict_prefetch`](TransitionMatrix::predict_prefetch) does not emit a fixed
//! top-K.  It keeps each candidate ranked by PMI whose confidence
//! (`max_from P(to|from)`, the strongest single active source's conditional) is
//! within [`PREFETCH_REL_CONF`] of the most confident candidate, capped at
//! [`PREFETCH_MAX_K`].  Using the relative max makes depth **scale- and
//! batch-invariant** and tie it to demand *diversity*: a homogeneous batch (one
//! prompt × N) implies a couple of sticky successors so prefetch stays shallow
//! and precise; diverse demand (many distinct prompts) implies many, so it
//! deepens toward the cap.
//!
//! ## Safety
//!
//! Mispredictions are harmless: speculatively loaded experts only ever fill free
//! VRAM slots and are reclaimed by normal eviction if unused.

/// PMI marginal-discount exponent.
const ALPHA: f32 = 0.5;

/// Minimum arrivals observed before predictions are emitted.  Prevents
/// single-observation noise in the first tokens.
const MIN_OBS: u32 = 64;

/// Upper bound on prefetch fan-out — the paper's bandwidth-knee cap.  The
/// confidence gate keeps the effective depth well below this on homogeneous
/// demand; the cap only bounds the diverse-demand case.
const PREFETCH_MAX_K: usize = 8;

/// Relative confidence floor for prefetch: an expert is prefetched only if its
/// confidence (`max_from P(to|from)`) is within this factor of the *most*
/// confident candidate's.  Relative rather than absolute so it is invariant to
/// the routing fan-out's scale — a top-8 router dilutes every per-source
/// conditional, which an absolute floor would wrongly gate to nothing.  Combined
/// with the batch-invariant max, prefetch depth then tracks demand *diversity*:
/// a sharp confidence drop (homogeneous demand) keeps only the top one or two; a
/// flat distribution (diverse demand) keeps more, up to the cap.
const PREFETCH_REL_CONF: f32 = 0.5;

/// The `[pairs × E × E]` co-occurrence matrix plus its row / column / group
/// marginals, all flat and indexed by `pair`.
struct CountTier {
    /// `counts[pair*e*e + from*e + to]`.
    counts: Vec<f32>,
    /// `row[pair*e + from] = Σ_to counts` — the conditional denominator.
    row: Vec<f32>,
    /// `col[pair*e + to] = Σ_from counts` — the marginal numerator P(to).
    col: Vec<f32>,
    /// `grp[pair] = Σ counts` — the marginal denominator.
    grp: Vec<f32>,
}

impl CountTier {
    fn new(pairs: usize, e: usize) -> Self {
        Self {
            counts: vec![0.0; pairs * e * e],
            row: vec![0.0; pairs * e],
            col: vec![0.0; pairs * e],
            grp: vec![0.0; pairs],
        }
    }

    /// Credit a single `from → to` transition in `pair`.
    #[inline]
    fn add(&mut self, pair: usize, from: usize, to: usize, e: usize) {
        self.counts[pair * e * e + from * e + to] += 1.0;
        self.row[pair * e + from] += 1.0;
        self.col[pair * e + to] += 1.0;
        self.grp[pair] += 1.0;
    }
}

/// The Markov Wave promote predictor.  See the module docs for the model.
pub(crate) struct TransitionMatrix {
    /// Experts per MoE layer (e.g. 128).
    e: usize,
    /// Total number of MoE layers (e.g. 48).
    num_moe_layers: usize,
    /// Number of adjacent layer pairs = `num_moe_layers - 1`.
    pairs: usize,
    /// The online arrival-specialised transition matrix.
    table: CountTier,
    /// Arrivals counted so far (warmup gate).
    obs: u32,
    /// Experts routed at the previous MoE layer of this forward pass, used by
    /// [`observe`](Self::observe) to build `L → L+1` transitions.
    prev_layer_experts: Option<(usize, Vec<usize>)>,
}

impl TransitionMatrix {
    /// Create a new predictor.
    ///
    /// * `num_moe_layers` — total MoE layers (e.g. 48)
    /// * `experts_per_layer` — number of experts per layer (e.g. 128)
    pub(crate) fn new(num_moe_layers: usize, experts_per_layer: usize) -> Self {
        let pairs = num_moe_layers.saturating_sub(1);
        let e = experts_per_layer;
        Self {
            e,
            num_moe_layers,
            pairs,
            table: CountTier::new(pairs, e),
            obs: 0,
            prev_layer_experts: None,
        }
    }

    /// Record the experts routed at a given MoE layer.
    ///
    /// Call this for every MoE layer in forward-pass order.  When the previous
    /// observation was the immediately preceding layer, the `L-1 → L`
    /// transitions are credited — arrival-specialised: targets already active at
    /// the source layer are skipped (they are cache hits, not the cold experts a
    /// prefetch must cover).
    pub(crate) fn observe(&mut self, moe_layer_idx: usize, expert_ids: &[usize]) {
        if self.pairs == 0 {
            return;
        }
        if let Some((prev_idx, prev_experts)) = self.prev_layer_experts.take() {
            if moe_layer_idx == prev_idx + 1 && prev_idx < self.pairs {
                let e = self.e;
                let pair = prev_idx;
                for &from in &prev_experts {
                    if from >= e {
                        continue;
                    }
                    for &to in expert_ids {
                        if to >= e || prev_experts.contains(&to) {
                            continue;
                        }
                        self.table.add(pair, from, to, e);
                        self.obs = self.obs.saturating_add(1);
                    }
                }
            }
        }
        self.prev_layer_experts = Some((moe_layer_idx, expert_ids.to_vec()));
    }

    /// Reset per-pass state at the start of each forward pass (each token):
    /// clears the previous-layer link so layer 0 of the new pass does not form a
    /// transition with the last layer of the previous pass.
    pub(crate) fn reset_pass(&mut self) {
        self.prev_layer_experts = None;
    }

    /// Score every successor expert for layer `moe_layer_idx + 1`, returning
    /// `(scores, conf)` or `None` if there is no successor or the model is not
    /// yet warm.
    ///
    /// - `scores[to]` is the PMI rank signal — what to prefer when choosing
    ///   *which* experts to prefetch.
    /// - `conf[to]` is the strongest single conditional `max_from P(to|from)`
    ///   over the active sources — "is any active expert strongly routing to
    ///   `to`?", i.e. how *confident* the model is that `to` is genuinely coming.
    ///   The max (not a sum) keeps it batch-invariant; it decides *how many*
    ///   experts are worth prefetching.
    fn score_and_conf(
        &self,
        moe_layer_idx: usize,
        expert_ids: &[usize],
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if moe_layer_idx + 1 >= self.num_moe_layers || moe_layer_idx >= self.pairs {
            return None;
        }
        if self.obs < MIN_OBS {
            return None;
        }
        let pair = moe_layer_idx;
        let e = self.e;
        let cbase = pair * e * e;
        let rbase = pair * e;
        let tot = self.table.grp[pair].max(1.0);

        let mut scores = vec![0.0f32; e];
        let mut conf = vec![0.0f32; e];

        for &from in expert_ids {
            if from >= e {
                continue;
            }
            let base = cbase + from * e;
            let rt = self.table.row[rbase + from];
            if rt <= 0.0 {
                continue;
            }
            for to in 0..e {
                let c = self.table.counts[base + to];
                if c <= 0.0 {
                    continue;
                }
                let cond = c / rt; // P(to | from)
                let p_to = (self.table.col[rbase + to] / tot).max(1e-9);
                scores[to] += cond / p_to.powf(ALPHA);
                conf[to] = conf[to].max(cond);
            }
        }

        Some((scores, conf))
    }

    /// Predict the top-`k` experts layer `moe_layer_idx + 1` will most likely
    /// need, ranked by PMI and excluding the active set.  Fixed fan-out form used
    /// only by the offline evaluation (production prefetch uses the gated form).
    #[cfg(test)]
    pub(crate) fn predict_topk(
        &self,
        moe_layer_idx: usize,
        expert_ids: &[usize],
        k: usize,
    ) -> Vec<usize> {
        if k == 0 {
            return vec![];
        }
        match self.score_and_conf(moe_layer_idx, expert_ids) {
            Some((scores, _conf)) => top_k_excluding(&scores, expert_ids, k),
            None => vec![],
        }
    }

    /// Predict the experts worth *prefetching* for layer `moe_layer_idx + 1`.
    ///
    /// The fan-out is not fixed: an expert is returned only if its confidence is
    /// within [`PREFETCH_REL_CONF`] of the most confident candidate's, capped at
    /// [`PREFETCH_MAX_K`] and ranked by PMI.  Depth therefore tracks demand
    /// diversity (see the module docs).
    pub(crate) fn predict_prefetch(
        &self,
        moe_layer_idx: usize,
        expert_ids: &[usize],
    ) -> Vec<usize> {
        match self.score_and_conf(moe_layer_idx, expert_ids) {
            Some((scores, conf)) => top_k_gated(
                &scores,
                &conf,
                expert_ids,
                PREFETCH_MAX_K,
                PREFETCH_REL_CONF,
            ),
            None => vec![],
        }
    }
}

/// Top-`k` indices of `scores` by descending value, excluding the `active` set
/// and any non-positive score.  Ties break toward the lower expert ID for
/// determinism.  An insertion sort into a `k`-sized buffer — `k` is small.
#[cfg(test)]
fn top_k_excluding(scores: &[f32], active: &[usize], k: usize) -> Vec<usize> {
    let better = |a: (usize, f32), b: (usize, f32)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
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

/// Like [`top_k_excluding`], but additionally drops any expert whose confidence
/// `conf[idx]` is below `rel_conf` times the *most* confident candidate's.  The
/// cap `max_k` bounds the result; the relative gate is what makes the effective
/// count adapt to how many experts the active set genuinely implies, without any
/// dependence on the absolute confidence scale.
fn top_k_gated(
    scores: &[f32],
    conf: &[f32],
    active: &[usize],
    max_k: usize,
    rel_conf: f32,
) -> Vec<usize> {
    if max_k == 0 {
        return vec![];
    }
    // The most-confident scoreable, non-active candidate sets the bar.
    let mut max_c = 0.0f32;
    for (idx, &s) in scores.iter().enumerate() {
        if s > 0.0 && !active.contains(&idx) && conf[idx] > max_c {
            max_c = conf[idx];
        }
    }
    if max_c <= 0.0 {
        return vec![];
    }
    let floor = rel_conf * max_c;

    let better = |a: (usize, f32), b: (usize, f32)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(max_k + 1);
    for (idx, &s) in scores.iter().enumerate() {
        if s <= 0.0 || conf[idx] < floor || active.contains(&idx) {
            continue;
        }
        if top.len() < max_k {
            top.push((idx, s));
        } else if better((idx, s), top[max_k - 1]) {
            top[max_k - 1] = (idx, s);
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

#[cfg(test)]
mod tests {
    use super::*;

    const L: usize = 6; // MoE layers
    const E: usize = 16; // experts per layer

    /// Drive `from`-at-`l` → `to`-at-`l+1` through `observe` `reps` times,
    /// resetting the pass each rep so it is a clean two-layer transition.
    fn train(m: &mut TransitionMatrix, l: usize, from: &[usize], to: &[usize], reps: usize) {
        for _ in 0..reps {
            m.reset_pass();
            m.observe(l, from);
            m.observe(l + 1, to);
        }
    }

    #[test]
    fn prefetch_uses_gated_markov_prediction() {
        // Prefetch stays on the capped, confidence-gated Markov path at all
        // densities — a cold model predicts nothing (no prefetch-all shortcut).
        let m = TransitionMatrix::new(4, E);
        assert!(m.predict_prefetch(0, &[1, 2]).is_empty());
        let dense: Vec<usize> = (0..E).collect();
        assert!(m.predict_prefetch(0, &dense).is_empty());
    }

    #[test]
    fn cold_model_predicts_nothing() {
        let m = TransitionMatrix::new(L, E);
        assert!(m.predict_topk(0, &[1], 4).is_empty());
    }

    #[test]
    fn warm_gate_blocks_until_min_obs() {
        let mut m = TransitionMatrix::new(L, E);
        // 30 arrivals (one per rep) is below MIN_OBS (64).
        train(&mut m, 0, &[1], &[7], 30);
        assert!(m.predict_topk(0, &[1], 4).is_empty());
        // Cross MIN_OBS — prediction now flows.
        train(&mut m, 0, &[1], &[7], 40);
        assert_eq!(m.predict_topk(0, &[1], 1), vec![7]);
    }

    #[test]
    fn learns_top_k_transition_in_rank_order() {
        let mut m = TransitionMatrix::new(L, E);
        // 1 → {7 dominant, 9 and 11 equal minors}.  PMI ranks 7 first; the equal
        // minors break their tie by ascending id.
        train(&mut m, 0, &[1], &[7], 80);
        train(&mut m, 0, &[1], &[9], 20);
        train(&mut m, 0, &[1], &[11], 20);
        assert_eq!(m.predict_topk(0, &[1], 3), vec![7, 9, 11]);
    }

    #[test]
    fn excludes_active_and_respects_successor_bound() {
        let mut m = TransitionMatrix::new(L, E);
        train(&mut m, 0, &[1], &[7], 80);
        // 7 is in the active set → never predicted even though it is the target.
        assert!(!m.predict_topk(0, &[1, 7], 4).contains(&7));
        // Last layer has no successor.
        assert!(m.predict_topk(L - 1, &[1], 4).is_empty());
    }

    #[test]
    fn observation_is_arrival_specialised() {
        let mut m = TransitionMatrix::new(L, E);
        // Source {1,7}; target {7,3} — 7 is already active, so it is NOT a cold
        // arrival and must not enter the counts.  3 is a true arrival.
        train(&mut m, 0, &[1, 7], &[7, 3], 80);
        let pair = 0;
        let resident = m.table.counts[(pair * E + 1) * E + 7];
        let arrival = m.table.counts[(pair * E + 1) * E + 3];
        assert_eq!(resident, 0.0, "resident target leaked into the matrix");
        assert!(arrival > 0.0, "arrival target missing from the matrix");
    }

    #[test]
    fn pmi_demotes_a_globally_popular_target() {
        let mut m = TransitionMatrix::new(L, E);
        // From 1: 5 appears as often as 9.  But 5 is *globally* popular (every
        // other source also routes to it), so PMI's marginal discount ranks the
        // specific target 9 above the popular 5.
        train(&mut m, 0, &[1], &[5], 60);
        train(&mut m, 0, &[1], &[9], 60);
        for src in 2..14usize {
            train(&mut m, 0, &[src], &[5], 60);
        }
        assert_eq!(m.predict_topk(0, &[1], 1), vec![9]);
    }

    #[test]
    fn prefetch_gates_the_low_confidence_tail() {
        let mut m = TransitionMatrix::new(L, E);
        // Source 1 routes to 7 almost always (conf ≈ 0.95) and to 9 rarely
        // (conf ≈ 0.05).  The fixed-k predictor names both; the confidence-gated
        // prefetch keeps only the genuinely-implied 7.
        train(&mut m, 0, &[1], &[7], 90);
        train(&mut m, 0, &[1], &[9], 5);
        assert_eq!(m.predict_topk(0, &[1], 8), vec![7, 9]);
        assert_eq!(m.predict_prefetch(0, &[1]), vec![7]);
    }

    #[test]
    fn prefetch_depth_grows_with_source_diversity() {
        let mut m = TransitionMatrix::new(L, E);
        // Three sources, each a strong distinct successor — the "diverse demand"
        // case.  A single active source implies one cold expert; the diverse set
        // implies three, so the prefetch deepens accordingly.
        train(&mut m, 0, &[1], &[7], 80);
        train(&mut m, 0, &[2], &[8], 80);
        train(&mut m, 0, &[3], &[9], 80);
        assert_eq!(m.predict_prefetch(0, &[1]), vec![7]);
        assert_eq!(m.predict_prefetch(0, &[1, 2, 3]), vec![7, 8, 9]);
    }

    #[test]
    fn prefetch_is_capped_at_max_k() {
        // More high-confidence successors than the cap → bounded to PREFETCH_MAX_K.
        let mut m = TransitionMatrix::new(L, 32);
        let sources: Vec<usize> = (1..=9).collect();
        for &s in &sources {
            train(&mut m, 0, &[s], &[s + 15], 80); // distinct target per source
        }
        assert_eq!(m.predict_prefetch(0, &sources).len(), PREFETCH_MAX_K);
    }

    #[test]
    fn prefetch_confidence_is_batch_invariant() {
        let mut m = TransitionMatrix::new(L, E);
        // Every source routes to 9 only ~10% of the time (weak); its real mass
        // goes to a distinct strong successor.  Summed confidence would let 9
        // through once enough sources are active (the batch bug); the max keeps
        // it gated no matter how many weakly-implying sources are active at once.
        for &s in &[1usize, 2, 3, 4, 5] {
            train(&mut m, 0, &[s], &[9], 10);
            train(&mut m, 0, &[s], &[s + 10], 90);
        }
        assert!(!m.predict_prefetch(0, &[1]).contains(&9));
        assert!(!m.predict_prefetch(0, &[1, 2, 3, 4, 5]).contains(&9));
    }
}
