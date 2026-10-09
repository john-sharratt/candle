//! The Markov Wave predictor — online expert-transition tables, one per hop.
//!
//! For each MoE row `L` and each hop `h` in `1..=HOPS` this learns an `[E × E]`
//! co-occurrence model from routing ids alone (no hidden states, no extra
//! GEMM): given the experts active at row `L`, it names the experts row
//! `L + h` will most likely need, so they can be staged into the pad and read
//! ahead into VRAM while `L .. L + h − 1` compute — converting cold misses into
//! overlapped loads.
//!
//! ## Model
//!
//! One table per hop, learned online and **arrival-specialised**: table `h`
//! credits a transition `from → to` for `from` active at row `L` and `to`
//! active at row `L + h` only when `to` is *not* already active at row `L` —
//! i.e. the cold experts a prefetch must actually cover. A direct table per
//! hop, rather than the hop-1 table chained through its own predictions,
//! because a chain compounds its precision at every hop and starts each hop
//! from a few predicted experts instead of the row's whole routing. The tables
//! accumulate over the session (in production the process *is* the session)
//! and converge to the workload's routing structure within a few thousand
//! tokens, so no cross-session prior or per-pass decay is needed. Memory is
//! `Σ_h (rows − h) · E²` counts — ~236 MB of `f32` for 48 rows of 512 experts
//! at five hops.
//!
//! Successors are scored by pointwise mutual information,
//! `PMI(α) = Σ_from (c/rt) / P(to)^α` with `α = ALPHA`, which demotes
//! globally-popular (already-cached) targets in favour of experts *specifically*
//! implied by the current routing.
//!
//! ## Prediction — candidates with their confidence
//!
//! [`candidates`](TransitionMatrix::candidates) names, for row `L + h`, every
//! successor ranked by PMI with its *per-source-relative* confidence (its
//! conditional normalized by its best source's strongest conditional — see
//! `score_and_conf`) at or above a floor. Per-source normalization keeps the
//! confidence **scale- and batch-invariant**: each active expert's strongest
//! successor reads 1.0 however flat its routing is, so one sticky pair cannot
//! raise the bar for every other source. The pipeline does not gate on it — the
//! blend (`blend`) weighs it beside the router look-ahead's votes and learns
//! what each band of it is worth.
//!
//! ## Safety
//!
//! Mispredictions are harmless: a speculatively staged expert only takes a pad
//! slot, a read-ahead claim only a promotion-ring offer, and both are reclaimed
//! by normal eviction if unused.

use std::collections::VecDeque;

/// PMI marginal-discount exponent.
const ALPHA: f32 = 0.5;

/// Minimum arrivals observed in a hop's table before it emits predictions.
/// Prevents single-observation noise in the first tokens.
const MIN_OBS: u32 = 64;

/// Hops the tables cover: a row predicts the next `HOPS` rows. Five is one
/// past read-ahead's depth (`read_ahead::READ_AHEAD_DEPTH`), the last row a
/// launch reads ahead for.
pub(crate) const HOPS: usize = 5;

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
    /// Arrivals counted so far (warmup gate).
    obs: u32,
}

impl CountTier {
    fn new(pairs: usize, e: usize) -> Self {
        Self {
            counts: vec![0.0; pairs * e * e],
            row: vec![0.0; pairs * e],
            col: vec![0.0; pairs * e],
            grp: vec![0.0; pairs],
            obs: 0,
        }
    }

    /// Credit a single `from → to` transition in `pair`.
    #[inline]
    fn add(&mut self, pair: usize, from: usize, to: usize, e: usize) {
        self.counts[pair * e * e + from * e + to] += 1.0;
        self.row[pair * e + from] += 1.0;
        self.col[pair * e + to] += 1.0;
        self.grp[pair] += 1.0;
        self.obs = self.obs.saturating_add(1);
    }
}

/// The Markov Wave predictor.  See the module docs for the model.
pub(crate) struct TransitionMatrix {
    /// Experts per MoE layer (e.g. 128).
    e: usize,
    /// Total number of MoE layers (e.g. 48).
    num_moe_layers: usize,
    /// The online arrival-specialised tables, `tables[h − 1]` for hop `h`,
    /// with `num_moe_layers − h` source rows each.
    tables: Vec<CountTier>,
    /// The routed sets of the last [`HOPS`] rows of this forward pass, oldest
    /// first, which [`observe`](Self::observe) credits each new row against.
    recent: VecDeque<(usize, Vec<usize>)>,
}

impl TransitionMatrix {
    /// Create a new predictor.
    ///
    /// * `num_moe_layers` — total MoE layers (e.g. 48)
    /// * `experts_per_layer` — number of experts per layer (e.g. 128)
    pub(crate) fn new(num_moe_layers: usize, experts_per_layer: usize) -> Self {
        let e = experts_per_layer;
        let tables = (1..=HOPS)
            .map(|h| CountTier::new(num_moe_layers.saturating_sub(h), e))
            .collect();
        Self {
            e,
            num_moe_layers,
            tables,
            recent: VecDeque::with_capacity(HOPS),
        }
    }

    /// The pairs hop `h`'s table holds: source rows `0 .. num_moe_layers − h`.
    fn pairs(&self, hop: usize) -> usize {
        self.num_moe_layers.saturating_sub(hop)
    }

    /// Record the experts routed at a given MoE layer.
    ///
    /// Call this for every MoE layer in forward-pass order.  Each of the last
    /// [`HOPS`] rows observed in this pass, `h` rows back, credits its
    /// `L − h → L` transitions into hop `h`'s table — arrival-specialised:
    /// targets already active at the source row are skipped (they are cache
    /// hits, not the cold experts a prefetch must cover). A row observed out of
    /// order (not after the previous one) starts the pass's history over.
    pub(crate) fn observe(&mut self, moe_layer_idx: usize, expert_ids: &[usize]) {
        if self.recent.back().is_some_and(|(r, _)| *r >= moe_layer_idx) {
            self.recent.clear();
        }
        let e = self.e;
        for (src_row, src) in &self.recent {
            let hop = moe_layer_idx - src_row;
            if hop == 0 || hop > HOPS || *src_row >= self.pairs(hop) {
                continue;
            }
            let table = &mut self.tables[hop - 1];
            for &from in src {
                if from >= e {
                    continue;
                }
                for &to in expert_ids {
                    if to >= e || src.contains(&to) {
                        continue;
                    }
                    table.add(*src_row, from, to, e);
                }
            }
        }
        if self.recent.len() == HOPS {
            self.recent.pop_front();
        }
        self.recent.push_back((moe_layer_idx, expert_ids.to_vec()));
    }

    /// Reset per-pass state at the start of each forward pass (each token):
    /// clears the recent rows so row 0 of the new pass forms no transition
    /// with the last rows of the previous pass.
    pub(crate) fn reset_pass(&mut self) {
        self.recent.clear();
    }

    /// Score every successor expert for row `moe_layer_idx + hop`, returning
    /// `(scores, conf)` or `None` if there is no such row or the hop's table
    /// is not yet warm.
    ///
    /// - `scores[to]` is the PMI rank signal — what to prefer when choosing
    ///   *which* experts to prefetch.
    /// - `conf[to]` is the strongest *per-source-relative* conditional over the
    ///   active sources: `max_from P(to|from) / max_to' P(to'|from)` — "does some
    ///   active expert route to `to` at close to that expert's own strongest
    ///   rate?".  Normalizing within each source keeps the gate batch-invariant
    ///   AND source-local: one sticky pair (a source with a near-certain
    ///   successor) cannot raise the bar for every other source's successors,
    ///   which matters for wide waves where dozens of sources each imply their
    ///   own cold arrivals.  The max (not a sum) across sources keeps it
    ///   batch-invariant; it decides *how many* experts are worth prefetching.
    fn score_and_conf(
        &self,
        moe_layer_idx: usize,
        hop: usize,
        expert_ids: &[usize],
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if hop == 0 || hop > HOPS || moe_layer_idx >= self.pairs(hop) {
            return None;
        }
        let table = &self.tables[hop - 1];
        if table.obs < MIN_OBS {
            return None;
        }
        let pair = moe_layer_idx;
        let e = self.e;
        let cbase = pair * e * e;
        let rbase = pair * e;
        let tot = table.grp[pair].max(1.0);

        let mut scores = vec![0.0f32; e];
        let mut conf = vec![0.0f32; e];

        for &from in expert_ids {
            if from >= e {
                continue;
            }
            let base = cbase + from * e;
            let rt = table.row[rbase + from];
            if rt <= 0.0 {
                continue;
            }
            // This source's strongest successor count — the normalizer that
            // makes its confidences source-relative (its argmax successor is
            // always confidence 1.0, however flat its routing is).
            let mut row_max = 0.0f32;
            for to in 0..e {
                row_max = row_max.max(table.counts[base + to]);
            }
            if row_max <= 0.0 {
                continue;
            }
            for to in 0..e {
                let c = table.counts[base + to];
                if c <= 0.0 {
                    continue;
                }
                let cond = c / rt; // P(to | from)
                let p_to = (table.col[rbase + to] / tot).max(1e-9);
                scores[to] += cond / p_to.powf(ALPHA);
                conf[to] = conf[to].max(c / row_max);
            }
        }

        Some((scores, conf))
    }

    /// Predict the top-`k` experts row `moe_layer_idx + 1` will most likely
    /// need, ranked by PMI and excluding the active set.  Fixed fan-out form used
    /// only by the offline evaluation (production reads [`Self::candidates`]).
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
        match self.score_and_conf(moe_layer_idx, 1, expert_ids) {
            Some((scores, _conf)) => top_k_excluding(&scores, expert_ids, k),
            None => vec![],
        }
    }

    /// The experts row `moe_layer_idx + hop` may need, each with its
    /// per-source-relative confidence: every non-active candidate at a
    /// confidence of at least `min_conf`, highest PMI first, at most `max_k`.
    /// The confidence is the evidence a blend weighs (`blend`). Empty until
    /// the hop's table is warm.
    pub(crate) fn candidates(
        &self,
        moe_layer_idx: usize,
        hop: usize,
        expert_ids: &[usize],
        min_conf: f32,
        max_k: usize,
    ) -> Vec<(usize, f32)> {
        let Some((scores, conf)) = self.score_and_conf(moe_layer_idx, hop, expert_ids) else {
            return Vec::new();
        };
        let mut out: Vec<(usize, f32, f32)> = scores
            .iter()
            .enumerate()
            .filter(|&(e, &s)| s > 0.0 && conf[e] >= min_conf && !expert_ids.contains(&e))
            .map(|(e, &s)| (e, s, conf[e]))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(max_k);
        out.into_iter().map(|(e, _, c)| (e, c)).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    const L: usize = 6; // MoE layers
    const E: usize = 16; // experts per layer

    /// The production cap for the tests below — wide enough never to bind.
    const K: usize = 8;

    /// The confidence the tests read the learned tables at: a successor at
    /// least half as strong as its source's strongest.
    const GATE: f32 = 0.5;

    /// Row `row + hop`'s candidates at [`GATE`] or above, experts only, highest
    /// PMI first, at most `k`.
    fn gated(m: &TransitionMatrix, row: usize, hop: usize, ids: &[usize], k: usize) -> Vec<usize> {
        m.candidates(row, hop, ids, GATE, k)
            .into_iter()
            .map(|(e, _)| e)
            .collect()
    }

    /// Drive `from`-at-`l` → `to`-at-`l+1` through `observe` `reps` times,
    /// resetting the pass each rep so it is a clean two-layer transition.
    fn train(m: &mut TransitionMatrix, l: usize, from: &[usize], to: &[usize], reps: usize) {
        for _ in 0..reps {
            m.reset_pass();
            m.observe(l, from);
            m.observe(l + 1, to);
        }
    }

    /// Drive a whole pass prefix, `rows[i]` routed at row `i`, `reps` times.
    fn train_pass(m: &mut TransitionMatrix, rows: &[&[usize]], reps: usize) {
        for _ in 0..reps {
            m.reset_pass();
            for (row, set) in rows.iter().enumerate() {
                m.observe(row, set);
            }
        }
    }

    #[test]
    fn prefetch_uses_gated_markov_prediction() {
        // Prefetch stays on the capped, confidence-gated Markov path at all
        // densities — a cold model predicts nothing (no prefetch-all shortcut).
        let m = TransitionMatrix::new(4, E);
        assert!(gated(&m, 0, 1, &[1, 2], K).is_empty());
        let dense: Vec<usize> = (0..E).collect();
        assert!(gated(&m, 0, 1, &dense, K).is_empty());
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

    /// The blend's view: every candidate over the confidence floor with its
    /// per-source confidence, highest PMI first — 7 is its source's strongest
    /// successor (1.0), 9 and 11 a quarter of it — the active set left out.
    #[test]
    fn candidates_carry_their_confidence_above_a_floor() {
        let mut m = TransitionMatrix::new(L, E);
        train(&mut m, 0, &[1], &[7], 80);
        train(&mut m, 0, &[1], &[9], 20);
        train(&mut m, 0, &[1], &[11], 20);
        assert_eq!(
            m.candidates(0, 1, &[1], 0.25, 8),
            vec![(7, 1.0), (9, 0.25), (11, 0.25)]
        );
        assert_eq!(m.candidates(0, 1, &[1], 0.3, 8), vec![(7, 1.0)]);
        assert_eq!(m.candidates(0, 1, &[1], 0.25, 2), vec![(7, 1.0), (9, 0.25)]);
        assert_eq!(
            m.candidates(0, 1, &[1, 7], 0.25, 8),
            vec![(9, 0.25), (11, 0.25)],
            "an active expert is not a candidate"
        );
        assert!(m.candidates(L - 1, 1, &[1], 0.25, 8).is_empty());
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
        let resident = m.tables[0].counts[(pair * E + 1) * E + 7];
        let arrival = m.tables[0].counts[(pair * E + 1) * E + 3];
        assert_eq!(resident, 0.0, "resident target leaked into the matrix");
        assert!(arrival > 0.0, "arrival target missing from the matrix");
    }

    /// Each hop has its own table, credited directly from the row that many
    /// rows back — so a two-hop prediction comes from the current row's own
    /// routing, not from chaining the hop-1 prediction — and specialised
    /// against that source row's set, not the row between.
    #[test]
    fn each_hop_learns_a_direct_transition_from_its_own_source_row() {
        let mut m = TransitionMatrix::new(L, E);
        // Row 0 routes {1}, row 1 {2, 9}, row 2 {7, 9}, row 3 {4}.
        train_pass(&mut m, &[&[1], &[2, 9], &[7, 9], &[4]], 80);
        assert_eq!(gated(&m, 0, 1, &[1], K), vec![2, 9]);
        assert_eq!(
            gated(&m, 0, 2, &[1], K),
            vec![7, 9],
            "direct: 9 is new to row 0"
        );
        assert_eq!(gated(&m, 0, 3, &[1], K), vec![4]);
        assert_eq!(
            gated(&m, 1, 1, &[2, 9], K),
            vec![7],
            "9 is resident at row 1"
        );
        assert_eq!(gated(&m, 1, 2, &[2], K), vec![4]);
        assert!(
            gated(&m, 0, 2, &[2], K).is_empty(),
            "2 was never a source at row 0"
        );
        assert!(
            gated(&m, 0, 4, &[1], K).is_empty(),
            "row 4 was never observed"
        );
        assert!(gated(&m, 0, 0, &[1], K).is_empty() && gated(&m, 0, HOPS + 1, &[1], K).is_empty());
        assert_eq!(
            m.tables.iter().map(|t| t.grp.len()).collect::<Vec<_>>(),
            vec![5, 4, 3, 2, 1],
            "hop h has rows − h source rows"
        );
    }

    /// Each hop's table warms up on its own arrivals.
    #[test]
    fn each_hops_table_warms_on_its_own_arrivals() {
        let mut m = TransitionMatrix::new(L, E);
        train_pass(&mut m, &[&[1], &[2], &[3]], 70);
        assert_eq!(gated(&m, 0, 1, &[1], K), vec![2]);
        assert_eq!(gated(&m, 0, 2, &[1], K), vec![3]);
        let mut n = TransitionMatrix::new(L, E);
        train_pass(&mut n, &[&[1], &[2], &[3]], 60);
        // Row 1 → 2 also gets 60 arrivals from the (1, 2) pair — every hop-1
        // table is one table, so it crosses 64 together with (0, 1).
        assert_eq!(gated(&n, 0, 1, &[1], K), vec![2]);
        assert!(
            gated(&n, 0, 2, &[1], K).is_empty(),
            "60 two-hop arrivals: not warm"
        );
    }

    /// A pass reset, or a row observed out of order, forgets the recent rows:
    /// the next pass's row 0 forms no transition with the last pass's tail.
    #[test]
    fn a_new_pass_forms_no_transition_with_the_last_passs_tail() {
        let mut m = TransitionMatrix::new(L, E);
        for _ in 0..80 {
            m.observe(0, &[1]);
            m.observe(1, &[2]);
            m.observe(0, &[5]);
            m.observe(1, &[6]);
        }
        assert_eq!(gated(&m, 0, 1, &[1], K), vec![2]);
        assert_eq!(gated(&m, 0, 1, &[5], K), vec![6]);
        assert!(
            gated(&m, 0, 2, &[1], K).is_empty(),
            "row 0 after row 1 restarts the pass"
        );
        assert!(
            gated(&m, 1, 1, &[2], K).is_empty(),
            "1 → 0 is no transition"
        );
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
        assert_eq!(gated(&m, 0, 1, &[1], K), vec![7]);
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
        assert_eq!(gated(&m, 0, 1, &[1], K), vec![7]);
        assert_eq!(gated(&m, 0, 1, &[1, 2, 3], K), vec![7, 8, 9]);
    }

    #[test]
    fn prefetch_is_capped_at_max_k() {
        // More high-confidence successors than the cap → bounded to `max_k`,
        // regardless of how wide the demand is: the cap is the caller's volume
        // control.
        let mut m = TransitionMatrix::new(L, 32);
        let sources: Vec<usize> = (1..=9).collect();
        for &s in &sources {
            train(&mut m, 0, &[s], &[s + 15], 80); // distinct target per source
        }
        assert_eq!(gated(&m, 0, 1, &sources, K).len(), K);
        assert_eq!(gated(&m, 0, 1, &sources, 3).len(), 3);

        // Narrow demand with many implied successors is bounded the same way.
        let mut m2 = TransitionMatrix::new(L, 32);
        for t in 16..=25 {
            train(&mut m2, 0, &[1], &[t], 20); // ten equal-confidence targets
        }
        assert_eq!(gated(&m2, 0, 1, &[1], K).len(), K);
    }

    #[test]
    fn sticky_source_does_not_gate_other_sources_successors() {
        // Source 1 is sticky (one near-certain successor). Source 2 routes
        // flatly to three successors, each under half of 1's conditional. A
        // global confidence bar set by the sticky pair would gate source 2's
        // successors out entirely; the per-source-relative gate keeps each
        // source's own genuinely-implied successors — the wide-wave case, where
        // dozens of sources each imply their own cold arrivals.
        let mut m = TransitionMatrix::new(L, E);
        train(&mut m, 0, &[1], &[7], 95);
        train(&mut m, 0, &[2], &[8], 30);
        train(&mut m, 0, &[2], &[9], 30);
        train(&mut m, 0, &[2], &[10], 30);
        let got = gated(&m, 0, 1, &[1, 2], K);
        for want in [7, 8, 9, 10] {
            assert!(got.contains(&want), "missing {want} in {got:?}");
        }
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
        assert!(!gated(&m, 0, 1, &[1], K).contains(&9));
        assert!(!gated(&m, 0, 1, &[1, 2, 3, 4, 5], K).contains(&9));
    }
}
