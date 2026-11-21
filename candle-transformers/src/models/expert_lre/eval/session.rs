//! Multi-source and session-adaptive predictors that combine recent routing with
//! the cross-prompt prior.
//!
//! `MultiSource` sums PMI contributions from its channels: the layer L to L+1
//! transition, a second hop from L-1, the previous token's same layer, and the
//! current token's pinned layers. `SessionPrior` blends a frozen cross-prompt
//! prior with a live session matrix under a weight beta. It never forgets, and the
//! session term overtakes the prior as evidence accrues. `Velocity` adds a boost
//! from the gap between fast- and slow-decayed session matrices, which picks out
//! experts whose transition probability is rising. `TwoStage` and `Recency` combine
//! a base with a session fork or a per-layer recency signal.

use super::*;

/// Input channel for the multi-source predictor (§4.3).  Each channel learns a
/// separate per-target-layer transition matrix from its own source position.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Channel {
    /// Layer L → target L+1 (the baseline first-order transition).
    Prev,
    /// Layer L-1 → target L+1 (second hop).
    Prev2,
    /// Previous token's layer L+1 → target L+1 (same-layer recurrence).
    Recur,
    /// Current token's pinned layers 0..PINNED → target L+1.
    Pinned,
}

/// Multi-source Markov predictor: sums PMI(0.5) contributions from a set of
/// input channels (§4.3).  Each channel keeps its own per-target-layer matrix.
#[derive(Clone)]
pub(super) struct MultiSource {
    pub(super) e: usize,
    pub(super) num_layers: usize,
    pub(super) alpha: f64,
    pub(super) channels: Vec<Channel>,
    // per channel: counts[target][from*e+to], row[target][from], col[target][to], tot[target]
    pub(super) counts: Vec<Vec<Vec<f64>>>,
    pub(super) row: Vec<Vec<Vec<f64>>>,
    pub(super) col: Vec<Vec<Vec<f64>>>,
    pub(super) tot: Vec<Vec<f64>>,
    pub(super) cur: Vec<Option<Vec<usize>>>,
    pub(super) prev: Vec<Option<Vec<usize>>>,
    pub(super) total_obs: u64,
    pub(super) min_obs: u64,
}

impl MultiSource {
    pub(super) fn new(num_layers: usize, num_experts: usize, channels: Vec<Channel>) -> Self {
        let nc = channels.len();
        let mk_m = || vec![vec![0.0; num_experts * num_experts]; num_layers];
        let mk_v = || vec![vec![0.0; num_experts]; num_layers];
        Self {
            e: num_experts,
            num_layers,
            alpha: 0.5,
            channels,
            counts: (0..nc).map(|_| mk_m()).collect(),
            row: (0..nc).map(|_| mk_v()).collect(),
            col: (0..nc).map(|_| mk_v()).collect(),
            tot: vec![vec![0.0; num_layers]; nc],
            cur: vec![None; num_layers],
            prev: vec![None; num_layers],
            total_obs: 0,
            min_obs: 64,
        }
    }

    /// Source expert set for a channel predicting `target`, given the current
    /// layer context.  Returns up to two source slices (pinned spans layers).
    pub(super) fn sources<'a>(
        &'a self,
        ch: Channel,
        target: usize,
        cur_layer_experts: &'a [usize],
    ) -> Vec<&'a [usize]> {
        let get = |layer: usize, store: &'a [Option<Vec<usize>>]| -> Option<&'a [usize]> {
            store.get(layer).and_then(|o| o.as_deref())
        };
        match ch {
            // target = L+1, so the current layer L's set is `cur_layer_experts`.
            Channel::Prev => vec![cur_layer_experts],
            Channel::Prev2 => target
                .checked_sub(2)
                .and_then(|l| get(l, &self.cur))
                .into_iter()
                .collect(),
            Channel::Recur => get(target, &self.prev).into_iter().collect(),
            Channel::Pinned => (0..PINNED).filter_map(|l| get(l, &self.cur)).collect(),
        }
    }
}

impl Predictor for MultiSource {
    fn reset_pass(&mut self) {
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.cur.iter_mut().for_each(|c| *c = None);
    }

    fn observe(&mut self, layer: usize, experts: &[usize]) {
        // Update each channel's matrix for target = `layer` using its sources.
        // The current layer's own set is needed for Prev when target's source
        // is layer-1; here target == layer so Prev source = cur[layer-1].
        let e = self.e;
        for (ci, &ch) in self.channels.clone().iter().enumerate() {
            let srcs: Vec<Vec<usize>> = match ch {
                Channel::Prev => layer
                    .checked_sub(1)
                    .and_then(|l| self.cur.get(l).and_then(|o| o.clone()))
                    .into_iter()
                    .collect(),
                Channel::Prev2 => layer
                    .checked_sub(2)
                    .and_then(|l| self.cur.get(l).and_then(|o| o.clone()))
                    .into_iter()
                    .collect(),
                Channel::Recur => self
                    .prev
                    .get(layer)
                    .and_then(|o| o.clone())
                    .into_iter()
                    .collect(),
                Channel::Pinned => (0..PINNED)
                    .filter_map(|l| self.cur.get(l).and_then(|o| o.clone()))
                    .collect(),
            };
            for src in &srcs {
                for &from in src {
                    if from >= e {
                        continue;
                    }
                    let base = from * e;
                    for &to in experts {
                        if to >= e {
                            continue;
                        }
                        self.counts[ci][layer][base + to] += 1.0;
                        self.row[ci][layer][from] += 1.0;
                        self.col[ci][layer][to] += 1.0;
                        self.tot[ci][layer] += 1.0;
                        self.total_obs += 1;
                    }
                }
            }
        }
        if layer < self.cur.len() {
            self.cur[layer] = Some(experts.to_vec());
        }
    }

    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        let target = layer + 1;
        if self.total_obs < self.min_obs || target >= self.num_layers {
            return vec![];
        }
        let e = self.e;
        let mut scores = vec![0.0f64; e];
        for (ci, &ch) in self.channels.iter().enumerate() {
            let m = &self.counts[ci][target];
            let row = &self.row[ci][target];
            let col = &self.col[ci][target];
            let tot = self.tot[ci][target].max(1.0);
            for src in self.sources(ch, target, experts) {
                for &from in src {
                    if from >= e {
                        continue;
                    }
                    let base = from * e;
                    let rt = row[from];
                    if rt <= 0.0 {
                        continue;
                    }
                    for to in 0..e {
                        let c = m[base + to];
                        if c <= 0.0 {
                            continue;
                        }
                        let p_to = (col[to] / tot).max(1e-9);
                        scores[to] += (c / rt) / p_to.powf(self.alpha);
                    }
                }
            }
        }
        let better = |a: (usize, f64), b: (usize, f64)| a.1 > b.1 || (a.1 == b.1 && a.0 < b.0);
        let mut top: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
        for (idx, &s) in scores.iter().enumerate() {
            if s <= 0.0 || experts.contains(&idx) {
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
}

/// Bayesian interpolation predictor: a frozen cross-prompt **prior** matrix
/// blended with a live **session** matrix that learns the current prompt online.
/// `beta` weights the prior (β→∞ pure prior, β=0 pure session).  Never forgets;
/// the session term simply overtakes the prior as evidence accrues.
#[derive(Clone)]
pub(super) struct SessionPrior {
    pub(super) prior: MatrixPredictor,   // trained on the 20, frozen
    pub(super) session: MatrixPredictor, // live, this prompt only
    pub(super) beta: f64,
    pub(super) alpha: f64,
}

impl SessionPrior {
    pub(super) fn new(prior: MatrixPredictor, beta: f64, alpha: f64) -> Self {
        let session = MatrixPredictor::new(prior.num_layers, prior.e, Scoring::Raw);
        Self {
            prior,
            session,
            beta,
            alpha,
        }
    }
}

impl Predictor for SessionPrior {
    fn reset_pass(&mut self) {
        self.session.reset_pass();
    }
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        self.session.observe(layer, experts);
    }
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        match self.scores(layer, experts) {
            Some(s) => top_k_excluding(&s, experts, k),
            None => vec![],
        }
    }
}

impl SessionPrior {
    /// Blended prior+session score vector for predicting `layer+1`.
    pub(super) fn scores(&self, layer: usize, experts: &[usize]) -> Option<Vec<f64>> {
        self.scores_with(layer, experts, self.beta, self.alpha)
    }

    /// Same, with explicit (β, α) — lets one trained fold be scored under many
    /// hyperparameters in a single held-out traversal.
    pub(super) fn scores_with(
        &self,
        layer: usize,
        experts: &[usize],
        b: f64,
        a: f64,
    ) -> Option<Vec<f64>> {
        if layer + 1 >= self.prior.num_layers {
            return None;
        }
        let e = self.prior.e;
        let g = self.prior.gidx(layer);
        let (pc, prt, pcol) = (
            &self.prior.counts[g],
            &self.prior.row_total[g],
            &self.prior.col_total[g],
        );
        let (sc, srt, scol) = (
            &self.session.counts[g],
            &self.session.row_total[g],
            &self.session.col_total[g],
        );
        let tot = (b * self.prior.grp_total[g] + self.session.grp_total[g]).max(1.0);

        let mut scores = vec![0.0f64; e];
        for &from in experts {
            if from >= e {
                continue;
            }
            let base = from * e;
            let rt = b * prt[from] + srt[from];
            if rt <= 0.0 {
                continue;
            }
            for to in 0..e {
                let c = b * pc[base + to] + sc[base + to];
                if c <= 0.0 {
                    continue;
                }
                let p_to = ((b * pcol[to] + scol[to]) / tot).max(1e-9);
                scores[to] += (c / rt) / p_to.powf(a);
            }
        }
        Some(scores)
    }
}

/// Velocity predictor: SessionPrior score plus a "rising-trend" boost from the
/// difference of a fast- and slow-decayed session matrix (catches experts whose
/// transition probability is increasing this session).
#[derive(Clone)]
pub(super) struct Velocity {
    pub(super) sp: SessionPrior,
    pub(super) fast: MatrixPredictor,
    pub(super) slow: MatrixPredictor,
    pub(super) gamma: f64,
}

impl Velocity {
    pub(super) fn new(sp: SessionPrior, gamma: f64) -> Self {
        Self::new_decays(sp, gamma, 0.7, 0.95)
    }
    pub(super) fn new_decays(sp: SessionPrior, gamma: f64, fast_d: f64, slow_d: f64) -> Self {
        let (l, e) = (sp.prior.num_layers, sp.prior.e);
        let mut fast = MatrixPredictor::new(l, e, Scoring::Raw);
        fast.decay = fast_d;
        let mut slow = MatrixPredictor::new(l, e, Scoring::Raw);
        slow.decay = slow_d;
        Self {
            sp,
            fast,
            slow,
            gamma,
        }
    }
}

impl Predictor for Velocity {
    fn reset_pass(&mut self) {
        self.sp.reset_pass();
        self.fast.reset_pass();
        self.slow.reset_pass();
    }
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        self.sp.observe(layer, experts);
        self.fast.observe(layer, experts);
        self.slow.observe(layer, experts);
    }
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        let Some(mut scores) = self.sp.scores(layer, experts) else {
            return vec![];
        };
        let e = self.sp.prior.e;
        let g = self.fast.gidx(layer);
        for &from in experts {
            if from >= e {
                continue;
            }
            let base = from * e;
            let (frt, srt) = (self.fast.row_total[g][from], self.slow.row_total[g][from]);
            for to in 0..e {
                let pf = if frt > 0.0 {
                    self.fast.counts[g][base + to] / frt
                } else {
                    0.0
                };
                let ps = if srt > 0.0 {
                    self.slow.counts[g][base + to] / srt
                } else {
                    0.0
                };
                let vel = pf - ps;
                if vel > 0.0 {
                    scores[to] += self.gamma * vel;
                }
            }
        }
        top_k_excluding(&scores, experts, k)
    }
}

/// Two-stage predictor: a frozen, slowly-accumulated **base** (cross-session
/// knowledge) + a fast per-session **fork** that may use a *different* formula.
/// Combined at the score level (each channel max-normalised, fork weighted by
/// `w_session`).  Models "base learns from history; session fork pivots to a
/// fast-learning formula".
#[derive(Clone)]
pub(super) struct TwoStage {
    pub(super) base: MatrixPredictor, // frozen, trained on prior sessions
    pub(super) session: MatrixPredictor, // live, this session only
    pub(super) sc_base: Scoring,
    pub(super) sc_session: Scoring,
    pub(super) w_session: f64,
}

impl TwoStage {
    pub(super) fn new(
        base: MatrixPredictor,
        sc_base: Scoring,
        sc_session: Scoring,
        w_session: f64,
        decay: f64,
    ) -> Self {
        let mut session = MatrixPredictor::new(base.num_layers, base.e, Scoring::Raw);
        session.decay = decay;
        session.min_obs = 8; // engage the fork as soon as a little session data exists
        Self {
            base,
            session,
            sc_base,
            sc_session,
            w_session,
        }
    }
}

impl Predictor for TwoStage {
    fn reset_pass(&mut self) {
        self.session.reset_pass();
    }
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        self.session.observe(layer, experts);
    }
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        let e = self.base.e;
        let mut combined = vec![0.0f64; e];
        let mut any = false;
        if let Some(sb) = self.base.score_vec(layer, experts, self.sc_base) {
            let m = sb.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
            for i in 0..e {
                combined[i] += sb[i] / m;
            }
            any = true;
        }
        if let Some(ss) = self.session.score_vec(layer, experts, self.sc_session) {
            let m = ss.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
            for i in 0..e {
                combined[i] += self.w_session * ss[i] / m;
            }
            any = true;
        }
        if !any {
            return vec![];
        }
        top_k_excluding(&combined, experts, k)
    }
}

/// Recency-fusion predictor: combines the SessionPrior cross-layer score with a
/// per-(target-layer, expert) recency signal (how recently an expert appeared
/// at the target layer) via weighted reciprocal-rank fusion.
#[derive(Clone)]
pub(super) struct Recency {
    pub(super) sp: SessionPrior,
    pub(super) ema: Vec<Vec<f64>>, // [layer][expert] decayed presence
    pub(super) decay: f64,
    pub(super) gamma: f64, // weight of the recency channel
}

impl Recency {
    pub(super) fn new(sp: SessionPrior, decay: f64, gamma: f64) -> Self {
        let (l, e) = (sp.prior.num_layers, sp.prior.e);
        Self {
            sp,
            ema: vec![vec![0.0; e]; l],
            decay,
            gamma,
        }
    }
}

impl Predictor for Recency {
    fn reset_pass(&mut self) {
        self.sp.reset_pass();
    }
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        self.sp.observe(layer, experts);
        let row = &mut self.ema[layer];
        row.iter_mut().for_each(|x| *x *= self.decay);
        for &x in experts {
            if x < row.len() {
                row[x] += 1.0 - self.decay;
            }
        }
    }
    fn predict(&self, layer: usize, experts: &[usize], k: usize) -> Vec<usize> {
        let target = layer + 1;
        if target >= self.sp.prior.num_layers {
            return vec![];
        }
        // Reciprocal-rank fusion of the two channels.
        const N: usize = 32;
        const C: f64 = 10.0;
        let sp_rank = self.sp.predict(layer, experts, N);
        let mut rec: Vec<(usize, f64)> = self.ema[target]
            .iter()
            .enumerate()
            .filter(|(i, &v)| v > 0.0 && !experts.contains(i))
            .map(|(i, &v)| (i, v))
            .collect();
        rec.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut fused: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for (r, &id) in sp_rank.iter().enumerate() {
            *fused.entry(id).or_insert(0.0) += 1.0 / (r as f64 + C);
        }
        for (r, &(id, _)) in rec.iter().take(N).enumerate() {
            *fused.entry(id).or_insert(0.0) += self.gamma / (r as f64 + C);
        }
        let mut v: Vec<(usize, f64)> = fused.into_iter().collect();
        v.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        v.into_iter().take(k).map(|(i, _)| i).collect()
    }
}
