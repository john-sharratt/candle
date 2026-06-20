//! Offline replay evaluation of the MoE expert predictor.
//!
//! Loads a captured routing trace (see [`crate::models::routing_capture`]) and
//! replays it through [`TransitionMatrix`] on the CPU, measuring how well the
//! predictor anticipates each layer's experts from the previous layer.  This
//! is the fast iteration loop for tuning the predictor — no model, no GPU,
//! milliseconds per run.
//!
//! Run the report with:
//! ```text
//! cargo test -p candle-transformers --lib expert_lre::eval::report -- --nocapture
//! ```
//! It is a no-op (prints a hint and returns) until the fixture has been
//! captured via `quantized_qwen3_moe::tests::capture_routing_trace`.

use std::collections::{HashMap, HashSet};
use std::io::Read;

use crate::models::routing_capture::{RoutingRecord, FIXTURE_PATH};

use super::cache::PINNED_LAYERS as PINNED;
use super::transition::TransitionMatrix;

/// A single forward pass: the per-layer active expert sets, in layer order.
struct Pass {
    /// Config index this pass belongs to (recurrence must not cross configs).
    config: u16,
    /// `(layer_idx, expert_ids)` for each MoE layer in this pass.
    layers: Vec<(usize, Vec<usize>)>,
    /// Per-layer routing mass aligned to `layers[i].1` (same order/length).
    masses: Vec<Vec<f32>>,
}

/// A replayable trace grouped into passes, plus inferred geometry.
struct Trace {
    passes: Vec<Pass>,
    num_layers: usize,
    num_experts: usize,
}

impl Trace {
    /// Group capture-ordered records into passes keyed by `(config, pass)`.
    fn from_records(records: &[RoutingRecord]) -> Self {
        let mut passes: Vec<Pass> = Vec::new();
        let mut cur_key: Option<(u16, u32)> = None;
        let mut num_layers = 0usize;
        let mut num_experts = 0usize;

        for r in records {
            let key = (r.config, r.pass);
            if cur_key != Some(key) {
                passes.push(Pass {
                    config: r.config,
                    layers: Vec::new(),
                    masses: Vec::new(),
                });
                cur_key = Some(key);
            }
            let experts: Vec<usize> = r.experts.iter().map(|&e| e as usize).collect();
            num_layers = num_layers.max(r.layer as usize + 1);
            if let Some(&m) = r.experts.iter().max() {
                num_experts = num_experts.max(m as usize + 1);
            }
            let p = passes.last_mut().unwrap();
            p.layers.push((r.layer as usize, experts));
            p.masses.push(r.mass.clone());
        }

        // Keep layer order canonical within each pass (layers + masses together).
        for p in &mut passes {
            let mut idx: Vec<usize> = (0..p.layers.len()).collect();
            idx.sort_by_key(|&i| p.layers[i].0);
            p.layers = idx.iter().map(|&i| p.layers[i].clone()).collect();
            p.masses = idx.iter().map(|&i| p.masses[i].clone()).collect();
        }

        Trace {
            passes,
            num_layers,
            num_experts,
        }
    }

    /// Average active-expert-set size across all (pass, layer) records.
    fn avg_set_size(&self) -> f64 {
        let (mut n, mut sum) = (0usize, 0usize);
        for p in &self.passes {
            for (_, e) in &p.layers {
                n += 1;
                sum += e.len();
            }
        }
        if n == 0 {
            0.0
        } else {
            sum as f64 / n as f64
        }
    }

    /// Sorted list of distinct config (prompt) indices present in the trace.
    fn configs(&self) -> Vec<u16> {
        let mut cs: Vec<u16> = self.passes.iter().map(|p| p.config).collect();
        cs.sort_unstable();
        cs.dedup();
        cs
    }

    /// A sub-trace containing only the passes of one config (prompt).
    fn for_config(&self, config: u16) -> Trace {
        self.filter(|p| p.config == config)
    }

    /// A sub-trace of every config *except* `config` (the LOOCV training set).
    fn for_configs_except(&self, config: u16) -> Trace {
        self.filter(|p| p.config != config)
    }

    fn filter(&self, keep: impl Fn(&Pass) -> bool) -> Trace {
        let passes = self
            .passes
            .iter()
            .filter(|p| keep(p))
            .map(|p| Pass {
                config: p.config,
                layers: p.layers.clone(),
                masses: p.masses.clone(),
            })
            .collect();
        Trace {
            passes,
            num_layers: self.num_layers,
            num_experts: self.num_experts,
        }
    }
}

/// Accumulated prediction-quality counters over validated transitions.
#[derive(Default, Clone)]
struct Metrics {
    /// Validated `(L → L+1)` transitions where a prediction was emitted.
    predicted_nonempty: usize,
    /// Transitions whose top-1 prediction was actually routed at `L+1`.
    top1_hits: usize,
    /// Sum of predicted-set sizes (precision denominator).
    pred_total: usize,
    /// Predicted experts actually routed at `L+1` (precision numerator).
    pred_hits: usize,
    /// Sum of next-layer "miss" experts: routed at `L+1`, not active at `L`
    /// (these are the experts a prefetch could usefully warm).
    miss_total: usize,
    /// Predicted experts that landed in the miss set (coverage numerator).
    miss_covered: usize,
}

impl Metrics {
    fn precision(&self) -> f64 {
        ratio(self.pred_hits, self.pred_total)
    }
    fn coverage(&self) -> f64 {
        ratio(self.miss_covered, self.miss_total)
    }
    fn top1(&self) -> f64 {
        ratio(self.top1_hits, self.predicted_nonempty)
    }

    /// Fold one transition: `predicted` against the actual `next` set, given
    /// the `current` active set (to compute the miss set).
    fn add(&mut self, predicted: &[usize], current: &[usize], next: &[usize]) {
        let next_set: HashSet<usize> = next.iter().copied().collect();
        let cur_set: HashSet<usize> = current.iter().copied().collect();
        let miss: HashSet<usize> = next_set.difference(&cur_set).copied().collect();

        self.miss_total += miss.len();
        if !predicted.is_empty() {
            self.predicted_nonempty += 1;
            if next_set.contains(&predicted[0]) {
                self.top1_hits += 1;
            }
            for &p in predicted {
                self.pred_total += 1;
                if next_set.contains(&p) {
                    self.pred_hits += 1;
                }
                if miss.contains(&p) {
                    self.miss_covered += 1;
                }
            }
        }
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// A next-layer predictor under evaluation.  Mirrors the production
/// observe/predict/reset lifecycle so the same replay drives any candidate.
trait Predictor {
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
struct Popularity {
    /// `freq[layer][expert]` running counts from observed passes.
    freq: Vec<Vec<u32>>,
}

impl Popularity {
    fn new(num_layers: usize, num_experts: usize) -> Self {
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
enum Scoring {
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
struct MatrixPredictor {
    e: usize,
    num_layers: usize,
    /// Source layers per matrix (layer resolution): 1 = per-pair, large = shared.
    group: usize,
    /// Per-pass count decay (momentum): <1.0 weights recent tokens more.
    decay: f64,
    /// Train only on "arrivals" — transitions whose target is NOT in the source
    /// layer's set (the cold experts a prefetch must actually cover).
    arrivals_only: bool,
    counts: Vec<Vec<f64>>,    // [group][from*e + to]
    row_total: Vec<Vec<f64>>, // [group][from] = Σ_to counts
    col_total: Vec<Vec<f64>>, // [group][to]   = Σ_from counts
    grp_total: Vec<f64>,      // [group] = Σ counts
    total_obs: u64,
    min_obs: u64,
    prev: Option<(usize, Vec<usize>, Vec<f64>)>,
    scoring: Scoring,
}

impl MatrixPredictor {
    fn new(num_layers: usize, num_experts: usize, scoring: Scoring) -> Self {
        Self::new_cfg(num_layers, num_experts, scoring, 1, 1.0)
    }

    /// `group` = source layers sharing one matrix (1 = per-pair).  `decay` =
    /// per-pass multiplicative count decay (1.0 = none / stationary).
    fn new_cfg(
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
    fn gidx(&self, src_layer: usize) -> usize {
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
    fn observe_mass(&mut self, layer: usize, experts: &[usize], masses: &[f64]) {
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
fn wilson_lb(p: f64, n: f64, z: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let z2 = z * z;
    ((p + z2 / (2.0 * n)) - z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / (1.0 + z2 / n)
}

impl MatrixPredictor {
    /// Full score vector for predicting `layer+1` under `scoring`.  None if the
    /// matrix is not warm or there is no successor layer.
    fn score_vec(&self, layer: usize, experts: &[usize], scoring: Scoring) -> Option<Vec<f64>> {
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
    fn score_topk(
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

/// Replay the trace through a predictor at fan-out `k`, returning metrics over
/// all transitions and over decode-like passes only (active set ≤ `decode_max`).
fn replay<P: Predictor>(
    trace: &Trace,
    predictor: &mut P,
    k: usize,
    decode_max: usize,
    warmup_epochs: usize,
) -> (Metrics, Metrics) {
    let mut all = Metrics::default();
    let mut decode = Metrics::default();

    // Warmup: observe the whole trace `warmup_epochs` times without scoring,
    // to approximate a matrix converged on this routing distribution (a long
    // session / cross-session-persisted matrix).  `warmup_epochs == 0` is the
    // honest cold-start: the matrix learns online from nothing.
    for _ in 0..warmup_epochs {
        for pass in &trace.passes {
            predictor.reset_pass();
            for (l, e) in &pass.layers {
                predictor.observe(*l, e);
            }
        }
    }

    for pass in &trace.passes {
        predictor.reset_pass();
        let is_decode = pass.layers.iter().all(|(_, e)| e.len() <= decode_max);

        // Predict for every layer (causally, before observing it), then
        // validate against the next layer in the same pass.
        let preds: Vec<Vec<usize>> = pass
            .layers
            .iter()
            .map(|(l, e)| {
                let p = predictor.predict(*l, e, k);
                predictor.observe(*l, e);
                p
            })
            .collect();

        for i in 0..pass.layers.len().saturating_sub(1) {
            let (l, ref cur) = pass.layers[i];
            let (ln, ref next) = pass.layers[i + 1];
            // Only score true adjacent transitions.
            if ln != l + 1 {
                continue;
            }
            all.add(&preds[i], cur, next);
            if is_decode {
                decode.add(&preds[i], cur, next);
            }
        }
    }

    (all, decode)
}

/// Per-layer token sequences: `seqs[L]` is the active expert set at layer `L`
/// for each decode pass, in token order, split so adjacency never crosses a
/// config boundary.  Prefill passes (large sets) are excluded.
fn per_layer_token_sequences(trace: &Trace, decode_max: usize) -> Vec<Vec<Vec<usize>>> {
    // For each layer, a flat list of (config, set) in pass order.
    let mut by_layer: Vec<Vec<(u16, Vec<usize>)>> = vec![Vec::new(); trace.num_layers];
    for pass in &trace.passes {
        let is_decode = pass.layers.iter().all(|(_, e)| e.len() <= decode_max);
        if !is_decode {
            continue;
        }
        for (l, experts) in &pass.layers {
            by_layer[*l].push((pass.config, experts.clone()));
        }
    }
    // Drop the config tag (kept only to detect boundaries during measurement).
    by_layer
        .into_iter()
        .map(|v| v.into_iter().map(|(_, s)| s).collect())
        .collect()
}

/// Measure same-layer token-to-token recurrence structure — the statistic that
/// governs eviction value (a `(layer, expert)` slot's next use is the same
/// layer on the next token).  Reports how predictable "needed next token" is
/// from the current/recent active sets, versus the base rate and versus LRU.
#[test]
fn recurrence() {
    let Some(records) = load_fixture() else {
        println!("\n[recurrence] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);
    const DECODE_MAX: usize = 16;
    let seqs = per_layer_token_sequences(&trace, DECODE_MAX);

    // Adjacent-token statistics, aggregated over all layers.
    // sticky: e ∈ S_next given e ∈ S_now.  arrival: e ∈ S_next given e ∉ S_now.
    let (mut sticky_hit, mut sticky_tot) = (0u64, 0u64);
    let (mut arrive_hit, mut arrive_tot) = (0u64, 0u64);
    let (mut next_sz, mut next_n) = (0u64, 0u64);
    // Of the experts needed next token, what share were already active now.
    let (mut need_from_active, mut need_tot) = (0u64, 0u64);

    let e = trace.num_experts;
    for layer_seq in &seqs {
        for w in layer_seq.windows(2) {
            let now: HashSet<usize> = w[0].iter().copied().collect();
            let next: HashSet<usize> = w[1].iter().copied().collect();
            next_sz += next.len() as u64;
            next_n += 1;
            for x in 0..e {
                let in_next = next.contains(&x);
                if now.contains(&x) {
                    sticky_tot += 1;
                    if in_next {
                        sticky_hit += 1;
                    }
                } else {
                    arrive_tot += 1;
                    if in_next {
                        arrive_hit += 1;
                    }
                }
            }
            for &x in &next {
                need_tot += 1;
                if now.contains(&x) {
                    need_from_active += 1;
                }
            }
        }
    }

    let base_rate = ratio(next_sz as usize, (next_n as usize) * e);
    let stickiness = ratio(sticky_hit as usize, sticky_tot as usize);
    let arrival = ratio(arrive_hit as usize, arrive_tot as usize);
    let recall_from_active = ratio(need_from_active as usize, need_tot as usize);

    println!("\n=== Same-layer token recurrence (decode passes) ===");
    println!(
        "layers={}  experts={}  adjacent-token steps measured={}",
        trace.num_layers, e, next_n
    );
    println!(
        "base rate  P(e routed next token at L)          = {:.1}%",
        base_rate * 100.0
    );
    println!(
        "stickiness P(e next | e active now)             = {:.1}%",
        stickiness * 100.0
    );
    println!(
        "arrival    P(e next | e NOT active now)         = {:.2}%",
        arrival * 100.0
    );
    println!(
        "lift (stickiness / base rate)                  = {:.1}×",
        if base_rate > 0.0 {
            stickiness / base_rate
        } else {
            0.0
        }
    );
    println!(
        "of experts needed next token, share already active = {:.1}%",
        recall_from_active * 100.0
    );

    // Recency window vs LRU: predict "needed next" = union of active sets over
    // the last `w` tokens.  Reports recall (cold experts covered) and the cost
    // (how many experts that keeps "hot").  This is the bar a recurrence-scored
    // eviction policy must beat.
    println!("\nrecency window → recall of next-token experts / avg kept-hot set:");
    for w in [1usize, 2, 4, 8] {
        let (mut rec_hit, mut rec_tot) = (0u64, 0u64);
        let (mut kept_sz, mut kept_n) = (0u64, 0u64);
        for layer_seq in &seqs {
            if layer_seq.len() <= w {
                continue;
            }
            for i in w..layer_seq.len() {
                let mut union: HashSet<usize> = HashSet::new();
                for prev in &layer_seq[i - w..i] {
                    union.extend(prev.iter().copied());
                }
                let next: HashSet<usize> = layer_seq[i].iter().copied().collect();
                kept_sz += union.len() as u64;
                kept_n += 1;
                for &x in &next {
                    rec_tot += 1;
                    if union.contains(&x) {
                        rec_hit += 1;
                    }
                }
            }
        }
        println!(
            "  w={w}: recall={:>5.1}%   kept-hot≈{:>4.1} experts/layer",
            ratio(rec_hit as usize, rec_tot as usize) * 100.0,
            kept_sz as f64 / kept_n.max(1) as f64,
        );
    }
    println!();
}

/// Online per-(layer, expert) recurrence model: learns how often an expert,
/// once routed at a layer, is routed there again on the next token.  This is
/// the eviction-value signal (a resident `(layer, expert)` slot's next use is
/// the same layer on the next token).
#[derive(Clone)]
struct RecurrenceModel {
    /// `active[L][e]` = times expert `e` was routed at layer `L`.
    active: Vec<Vec<u32>>,
    /// `recur[L][e]` = times `e` at `L` was followed by `e` at `L` next token.
    recur: Vec<Vec<u32>>,
    /// Previous token's active set per layer.
    prev: Vec<Option<Vec<usize>>>,
}

impl RecurrenceModel {
    fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            active: vec![vec![0; num_experts]; num_layers],
            recur: vec![vec![0; num_experts]; num_layers],
            prev: vec![None; num_layers],
        }
    }

    /// Observe layer `L`'s active set for the current token, updating the
    /// recurrence statistics against the previous token at the same layer.
    fn observe(&mut self, layer: usize, experts: &[usize]) {
        let cur: HashSet<usize> = experts.iter().copied().collect();
        if let Some(prev) = &self.prev[layer] {
            for &e in prev {
                self.active[layer][e] += 1;
                if cur.contains(&e) {
                    self.recur[layer][e] += 1;
                }
            }
        }
        self.prev[layer] = Some(experts.to_vec());
    }

    /// Clear cross-token state (call at prompt boundaries during training).
    fn reset(&mut self) {
        self.prev.iter_mut().for_each(|p| *p = None);
    }

    /// Smoothed recurrence probability for a resident `(layer, expert)` — the
    /// estimated chance it is routed at this layer again next token.  Higher =
    /// more valuable to keep.  Unseen experts fall back to the base prior.
    fn keep_value(&self, layer: usize, expert: usize, prior: f64) -> f64 {
        let a = self.active[layer][expert] as f64;
        let r = self.recur[layer][expert] as f64;
        // Laplace toward the global prior with a weight of 2 pseudo-obs.
        (r + 2.0 * prior) / (a + 2.0)
    }
}

/// Eviction victim-selection policy for the cache simulator.
#[derive(Clone, Copy, PartialEq)]
enum Evict {
    /// Least-recently-used (the production behind-layer/last_used stand-in).
    Lru,
    /// Recency primary, recurrence as a refinement worth ~`scale` clock steps:
    /// evict min(last_used + keep_value × scale).  A sticky expert effectively
    /// looks one cyclic cohort "newer" and is kept slightly longer.  `scale = 0`
    /// is exactly LRU.
    Blend { scale: f64 },
}

/// Cache simulator: replays the trace through a fixed VRAM budget, counting
/// expert misses under a given eviction policy and optional PMI prefetch.
/// Lower miss rate = fewer cold loads = higher throughput.
struct Sim {
    budget: usize,
    /// key = layer * num_experts + expert → (last_used, keep_value).
    resident: HashMap<usize, (u64, f64)>,
    clock: u64,
    misses: u64,
    accesses: u64,
}

impl Sim {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            resident: HashMap::new(),
            clock: 0,
            misses: 0,
            accesses: 0,
        }
    }

    fn touch(&mut self, key: usize, keep: f64) {
        self.clock += 1;
        let c = self.clock;
        self.resident
            .entry(key)
            .and_modify(|v| *v = (c, keep))
            .or_insert((c, keep));
    }

    /// Insert `key`, evicting the worst victim under `policy` if full.
    fn insert(&mut self, key: usize, keep: f64, policy: Evict) {
        if !self.resident.contains_key(&key) && self.resident.len() >= self.budget {
            // Pick a victim.
            let victim = self
                .resident
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let (la, ka) = **a;
                    let (lb, kb) = **b;
                    match policy {
                        Evict::Lru => la.cmp(&lb),
                        Evict::Blend { scale } => {
                            let sa = la as f64 + ka * scale;
                            let sb = lb as f64 + kb * scale;
                            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    }
                })
                .map(|(k, _)| *k);
            if let Some(v) = victim {
                self.resident.remove(&v);
            }
        }
        self.touch(key, keep);
    }
}

/// Global same-layer stickiness P(e routed next token | e active now), used as
/// the recurrence prior: a freshly-loaded expert was just active, so its prior
/// chance of recurring is the stickiness — not the much lower base rate.
fn global_stickiness(trace: &Trace, decode_max: usize) -> f64 {
    let seqs = per_layer_token_sequences(trace, decode_max);
    let (mut hit, mut tot) = (0u64, 0u64);
    for layer_seq in &seqs {
        for w in layer_seq.windows(2) {
            let next: HashSet<usize> = w[1].iter().copied().collect();
            for &x in &w[0] {
                tot += 1;
                if next.contains(&x) {
                    hit += 1;
                }
            }
        }
    }
    ratio(hit as usize, tot as usize)
}

/// Run one full cache simulation over the trace; returns the miss rate.
fn simulate(trace: &Trace, budget: usize, policy: Evict, prefetch_pmi_k: usize, prior: f64) -> f64 {
    let e = trace.num_experts;
    let mut sim = Sim::new(budget);
    let mut recur = RecurrenceModel::new(trace.num_layers, e);
    // Prefetch predictor (PMI) learned online alongside.
    let mut pmi = MatrixPredictor::new(trace.num_layers, e, Scoring::Pmi { alpha: 0.5 });

    for pass in &trace.passes {
        pmi.reset_pass();
        for (l, experts) in &pass.layers {
            // PMI prefetch into the cache (free, before the layer "runs").
            if prefetch_pmi_k > 0 {
                for pe in pmi.predict(*l, experts, prefetch_pmi_k) {
                    let key = l * e + pe;
                    let keep = recur.keep_value(*l, pe, prior);
                    // Prefetch only fills if there is room (never evict to prefetch).
                    if sim.resident.contains_key(&key) || sim.resident.len() < sim.budget {
                        sim.touch(key, keep);
                    }
                }
            }
            // Access the experts this layer actually needs.
            for &x in experts {
                sim.accesses += 1;
                let key = l * e + x;
                let keep = recur.keep_value(*l, x, prior);
                if sim.resident.contains_key(&key) {
                    sim.touch(key, keep);
                } else {
                    sim.misses += 1;
                    sim.insert(key, keep, policy);
                }
            }
            // Learn recurrence from this layer's set, and feed the PMI matrix.
            recur.observe(*l, experts);
            pmi.observe(*l, experts);
        }
    }
    ratio(sim.misses as usize, sim.accesses as usize)
}

#[test]
fn cache_sim() {
    let Some(records) = load_fixture() else {
        println!("\n[cache_sim] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);

    // Total distinct (layer, expert) the trace ever touches sets the scale.
    let distinct = {
        let mut s = HashSet::new();
        for p in &trace.passes {
            for (l, es) in &p.layers {
                for &x in es {
                    s.insert(l * trace.num_experts + x);
                }
            }
        }
        s.len()
    };

    const DECODE_MAX: usize = 16;
    let prior = global_stickiness(&trace, DECODE_MAX);

    println!("\n=== Cache simulation (miss rate, lower is better) ===");
    println!(
        "distinct (layer,expert) touched = {distinct}   recurrence prior = {:.1}%",
        prior * 100.0
    );
    // One cyclic cohort ≈ accesses per token (layers × avg active set).
    let cohort = (trace.num_layers as f64 * trace.avg_set_size()).max(1.0);
    println!(
        "\n{:>7} {:>9} {:>12} {:>13} {:>13} {:>14}",
        "budget", "LRU", "blend×1", "blend×4", "blend×4+PMI", "(Δ vs LRU)"
    );
    println!("{}", "-".repeat(72));
    for frac in [0.3f64, 0.5, 0.7, 0.85] {
        let budget = ((distinct as f64) * frac).ceil() as usize;
        let lru = simulate(&trace, budget, Evict::Lru, 0, prior);
        let b1 = simulate(&trace, budget, Evict::Blend { scale: cohort }, 0, prior);
        let b4 = simulate(
            &trace,
            budget,
            Evict::Blend {
                scale: cohort * 4.0,
            },
            0,
            prior,
        );
        let b4_pmi = simulate(
            &trace,
            budget,
            Evict::Blend {
                scale: cohort * 4.0,
            },
            4,
            prior,
        );
        println!(
            "{:>6.0}% {:>8.1}% {:>11.1}% {:>12.1}% {:>12.1}% {:>+13.1}",
            frac * 100.0,
            lru * 100.0,
            b1 * 100.0,
            b4 * 100.0,
            b4_pmi * 100.0,
            (b4_pmi - lru) * 100.0,
        );
    }
    println!("\nbudget = fraction of distinct (layer,expert) touched; Δ = best − LRU (negative = fewer misses).\n");
}

/// Per-config (per-prompt) breakdown: proves the harness distinguishes test
/// cases and shows how predictability varies by prompt.
#[test]
fn per_config() {
    let Some(records) = load_fixture() else {
        println!("\n[per_config] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);
    const DECODE_MAX: usize = 16;

    println!("\n=== Per-config (per-prompt) breakdown ===");
    println!(
        "{:>4} {:>9} {:>7} {:>9} {:>12} {:>12}",
        "cfg", "records", "passes", "avg|set|", "pmi-cov@4", "stickiness"
    );
    println!("{}", "-".repeat(58));
    for c in trace.configs() {
        let sub = trace.for_config(c);
        let records: usize = sub.passes.iter().map(|p| p.layers.len()).sum();
        let mut pmi =
            MatrixPredictor::new(sub.num_layers, sub.num_experts, Scoring::Pmi { alpha: 0.5 });
        let (_, d) = replay(&sub, &mut pmi, 4, DECODE_MAX, 0);
        let stick = global_stickiness(&sub, DECODE_MAX);
        println!(
            "{:>4} {:>9} {:>7} {:>9.1} {:>11.1}% {:>11.1}%",
            c,
            records,
            sub.passes.len(),
            sub.avg_set_size(),
            d.coverage() * 100.0,
            stick * 100.0,
        );
    }
    println!();
}

/// Train a predictor by observing every pass of `trace` for `epochs` passes.
fn train_predictor<P: Predictor>(p: &mut P, trace: &Trace, epochs: usize) {
    for _ in 0..epochs {
        for pass in &trace.passes {
            p.reset_pass();
            for (l, e) in &pass.layers {
                p.observe(*l, e);
            }
        }
    }
}

/// Unbiased promote (prefetch) evaluation via 21-fold leave-one-out CV.
///
/// For each held-out prompt the matrix is trained once on the other 20, then
/// every scoring × k is evaluated in a single held-out traversal (the counts
/// are identical across scorings, so training is not repeated).  Run with
/// `--release` for fast iteration.
#[test]
fn loocv_prefetch() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_prefetch] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    const EPOCHS: usize = 2;
    const DECODE_MAX: usize = 16;

    let scorings: [(&str, Scoring); 3] = [
        ("raw", Scoring::Raw),
        ("conditional", Scoring::Conditional),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
    ];
    let ks = [1usize, 2, 4];

    // metrics[mode][scoring][k]; mode 0 = frozen, 1 = +online.
    let mut m = vec![vec![vec![Metrics::default(); ks.len()]; scorings.len()]; 2];
    let mut mpop = vec![vec![Metrics::default(); ks.len()]; 2];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let test = trace.for_config(c);
        let mut mtx = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut mtx, &train, EPOCHS);
        let mut pop = Popularity::new(l, e);
        train_predictor(&mut pop, &train, EPOCHS);

        // Frozen first (no mutation), then +online (mutates) on the same state.
        for mode in 0..2 {
            let learn = mode == 1;
            for pass in &test.passes {
                if !pass.layers.iter().all(|(_, ex)| ex.len() <= DECODE_MAX) {
                    continue;
                }
                mtx.reset_pass();
                pop.reset_pass();
                let nl = pass.layers.len();
                let maxk = *ks.iter().max().unwrap();
                // Rank once per scoring (top-maxk); each k is a prefix.
                let mut lp: Vec<Vec<Vec<usize>>> = Vec::with_capacity(nl); // [layer][scoring]
                let mut pp: Vec<Vec<usize>> = Vec::with_capacity(nl); // [layer]
                for (li, ex) in &pass.layers {
                    lp.push(
                        scorings
                            .iter()
                            .map(|&(_, sc)| mtx.score_topk(*li, ex, maxk, sc))
                            .collect(),
                    );
                    pp.push(pop.predict(*li, ex, maxk));
                    if learn {
                        mtx.observe(*li, ex);
                        pop.observe(*li, ex);
                    }
                }
                for i in 0..nl.saturating_sub(1) {
                    let (li, ref cur) = pass.layers[i];
                    let (ln, ref next) = pass.layers[i + 1];
                    // Adjacent transitions only; exclude pinned target layers
                    // (always resident — predicting them is moot).
                    if ln != li + 1 || ln < PINNED {
                        continue;
                    }
                    for (ki, &k) in ks.iter().enumerate() {
                        for si in 0..scorings.len() {
                            let p = &lp[i][si];
                            m[mode][si][ki].add(&p[..k.min(p.len())], cur, next);
                        }
                        let pk = &pp[i];
                        mpop[mode][ki].add(&pk[..k.min(pk.len())], cur, next);
                    }
                }
            }
        }
    }

    println!(
        "\n=== Promote: 21-fold leave-one-out CV (train_epochs={EPOCHS}) ===\n\
         held-out decode coverage — every prompt tested on a model that never saw it"
    );
    for (mode, name) in [
        (0usize, "frozen (pure generalization)"),
        (1, "+online (realistic)"),
    ] {
        println!("\n── {name} ──");
        println!(
            "{:<14} {:>3} {:>8} {:>10} {:>12}",
            "predictor", "k", "top1", "precision", "decode-cov"
        );
        println!("{}", "-".repeat(52));
        for ki in 0..ks.len() {
            let pm = &mpop[mode][ki];
            println!(
                "{:<14} {:>3} {:>7.1}% {:>9.1}% {:>11.1}%",
                "popularity",
                ks[ki],
                pm.top1() * 100.0,
                pm.precision() * 100.0,
                pm.coverage() * 100.0
            );
            for si in 0..scorings.len() {
                let mm = &m[mode][si][ki];
                println!(
                    "{:<14} {:>3} {:>7.1}% {:>9.1}% {:>11.1}%",
                    scorings[si].0,
                    ks[ki],
                    mm.top1() * 100.0,
                    mm.precision() * 100.0,
                    mm.coverage() * 100.0
                );
            }
            println!();
        }
    }
}

/// Input channel for the multi-source predictor (§4.3).  Each channel learns a
/// separate per-target-layer transition matrix from its own source position.
#[derive(Clone, Copy, PartialEq)]
enum Channel {
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
struct MultiSource {
    e: usize,
    num_layers: usize,
    alpha: f64,
    channels: Vec<Channel>,
    // per channel: counts[target][from*e+to], row[target][from], col[target][to], tot[target]
    counts: Vec<Vec<Vec<f64>>>,
    row: Vec<Vec<Vec<f64>>>,
    col: Vec<Vec<Vec<f64>>>,
    tot: Vec<Vec<f64>>,
    cur: Vec<Option<Vec<usize>>>,
    prev: Vec<Option<Vec<usize>>>,
    total_obs: u64,
    min_obs: u64,
}

impl MultiSource {
    fn new(num_layers: usize, num_experts: usize, channels: Vec<Channel>) -> Self {
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
    fn sources<'a>(
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
struct SessionPrior {
    prior: MatrixPredictor,   // trained on the 20, frozen
    session: MatrixPredictor, // live, this prompt only
    beta: f64,
    alpha: f64,
}

impl SessionPrior {
    fn new(prior: MatrixPredictor, beta: f64, alpha: f64) -> Self {
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
    fn scores(&self, layer: usize, experts: &[usize]) -> Option<Vec<f64>> {
        self.scores_with(layer, experts, self.beta, self.alpha)
    }

    /// Same, with explicit (β, α) — lets one trained fold be scored under many
    /// hyperparameters in a single held-out traversal.
    fn scores_with(&self, layer: usize, experts: &[usize], b: f64, a: f64) -> Option<Vec<f64>> {
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
struct Velocity {
    sp: SessionPrior,
    fast: MatrixPredictor,
    slow: MatrixPredictor,
    gamma: f64,
}

impl Velocity {
    fn new(sp: SessionPrior, gamma: f64) -> Self {
        Self::new_decays(sp, gamma, 0.7, 0.95)
    }
    fn new_decays(sp: SessionPrior, gamma: f64, fast_d: f64, slow_d: f64) -> Self {
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

/// §7.7 — final combined model: SessionPrior (β=0.02, α=0.5) + arrival-
/// specialised training + velocity boost.  Reports the best vs the §5 winner.
#[test]
fn loocv_final() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_final] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    // (label, beta, arrivals, velocity-gamma)
    let configs: Vec<(&str, f64, bool, f64)> = vec![
        ("stationary pmi(0.5)", 1.0, false, 0.0),
        ("SessionPrior", 0.02, false, 0.0),
        ("  + arrivals", 0.02, true, 0.0),
        ("  + velocity", 0.02, false, 1.5),
        ("  + both", 0.02, true, 1.5),
    ];
    let mut out = vec![vec![Metrics::default(); ks.len()]; configs.len()];
    for &c in &trace.configs() {
        let test = trace.for_config(c);
        let train = trace.for_configs_except(c);
        for (ci, &(_, beta, arr, gamma)) in configs.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            prior.arrivals_only = arr;
            train_predictor(&mut prior, &train, 2);
            let mut sp = SessionPrior::new(prior, beta, 0.5);
            sp.session.arrivals_only = arr;
            let mut v = Velocity::new(sp, gamma);
            score_into(&mut v, &test, &ks, 4, true, &mut out[ci]);
        }
    }
    println!("\n=== §7.7 Final combined promote model (+online, held-out) ===");
    println!("{:<22} {:>9} {:>9} {:>9}", "model", "k=1", "k=2", "k=4");
    println!("{}", "-".repeat(52));
    for (ci, &(label, ..)) in configs.iter().enumerate() {
        println!(
            "{:<22} {:>8.1}% {:>8.1}% {:>8.1}%",
            label,
            out[ci][0].coverage() * 100.0,
            out[ci][1].coverage() * 100.0,
            out[ci][2].coverage() * 100.0
        );
    }
    println!();
}

/// Two-stage predictor: a frozen, slowly-accumulated **base** (cross-session
/// knowledge) + a fast per-session **fork** that may use a *different* formula.
/// Combined at the score level (each channel max-normalised, fork weighted by
/// `w_session`).  Models "base learns from history; session fork pivots to a
/// fast-learning formula".
#[derive(Clone)]
struct TwoStage {
    base: MatrixPredictor,    // frozen, trained on prior sessions
    session: MatrixPredictor, // live, this session only
    sc_base: Scoring,
    sc_session: Scoring,
    w_session: f64,
}

impl TwoStage {
    fn new(
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

/// §8 — two-stage base + session-fork architecture: sweep fork weight and fork
/// formula (base fixed at pmi(0.5)).  Compares to the §6 SessionPrior champion.
#[test]
fn loocv_twostage() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_twostage] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [4usize];
    // fork formula candidates
    let forks: Vec<(&str, Scoring)> = vec![
        ("conditional", Scoring::Conditional),
        ("pmi(0.25)", Scoring::Pmi { alpha: 0.25 }),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
    ];
    let weights = [0.5f64, 1.0, 2.0, 4.0, 8.0];
    let decays = [1.0f64, 0.9];
    // grid[decay][fork][weight] @k=4
    let mut grid = vec![vec![vec![Metrics::default(); weights.len()]; forks.len()]; decays.len()];

    for &c in &trace.configs() {
        let mut base = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut base, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (di, &decay) in decays.iter().enumerate() {
            for (fi, &(_, fsc)) in forks.iter().enumerate() {
                for (wi, &w) in weights.iter().enumerate() {
                    let mut ts =
                        TwoStage::new(base.clone(), Scoring::Pmi { alpha: 0.5 }, fsc, w, decay);
                    let mut m = vec![Metrics::default(); ks.len()];
                    score_into(&mut ts, &test, &ks, 4, true, &mut m);
                    grid[di][fi][wi].pred_total += m[0].pred_total;
                    grid[di][fi][wi].pred_hits += m[0].pred_hits;
                    grid[di][fi][wi].miss_total += m[0].miss_total;
                    grid[di][fi][wi].miss_covered += m[0].miss_covered;
                }
            }
        }
    }

    println!("\n=== §8 Two-stage base(pmi0.5) + fast fork (k=4 decode-cov, +online) ===");
    println!("reference: SessionPrior count-blend = 41.8% ; champion = 42.5%");
    let mut best = (0.0f64, String::new());
    for (di, &decay) in decays.iter().enumerate() {
        println!("\n── fork decay = {decay} ──");
        print!("{:<14}", "fork\\w_sess");
        for w in weights {
            print!("{:>8.1}", w);
        }
        println!();
        for (fi, &(flabel, _)) in forks.iter().enumerate() {
            print!("{:<14}", flabel);
            for (wi, _) in weights.iter().enumerate() {
                let cov = grid[di][fi][wi].coverage() * 100.0;
                print!("{:>7.1}%", cov);
                if cov > best.0 {
                    best = (
                        cov,
                        format!("decay={decay} fork={flabel} w={}", weights[wi]),
                    );
                }
            }
            println!();
        }
    }
    println!("\nbest two-stage: {:.1}%  ({})\n", best.0, best.1);
}

/// Build the champion predictor at given hyperparameters.
fn champion(
    prior: MatrixPredictor,
    beta: f64,
    alpha: f64,
    gamma: f64,
    fast: f64,
    slow: f64,
) -> Velocity {
    let mut sp = SessionPrior::new(prior, beta, alpha);
    sp.session.arrivals_only = true;
    Velocity::new_decays(sp, gamma, fast, slow)
}

/// §9.2 — velocity tuning at β=0.01, α=0.5 (arrivals): γ × (fast,slow) decays.
#[test]
fn tune_vel() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_vel] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let gammas = [0.0f64, 1.0, 2.0, 3.0, 4.0];
    let decays = [
        ("0.7/0.95", 0.7, 0.95),
        ("0.6/0.92", 0.6, 0.92),
        ("0.8/0.97", 0.8, 0.97),
    ];
    let mut grid = vec![vec![Metrics::default(); gammas.len()]; decays.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (di, &(_, fd, sd)) in decays.iter().enumerate() {
            for (gi, &g) in gammas.iter().enumerate() {
                let mut v = champion(prior.clone(), 0.01, 0.5, g, fd, sd);
                let mut m = vec![Metrics::default(); 1];
                score_into(&mut v, &test, &[4usize], 4, true, &mut m);
                grid[di][gi].pred_total += m[0].pred_total;
                grid[di][gi].pred_hits += m[0].pred_hits;
                grid[di][gi].miss_total += m[0].miss_total;
                grid[di][gi].miss_covered += m[0].miss_covered;
            }
        }
    }
    println!("\n=== §9.2 Velocity tuning (β=0.01 α=0.5 arrivals, k=4 cov) ===");
    print!("{:<10}", "decay\\γ");
    for g in gammas {
        print!("{:>8.1}", g);
    }
    println!();
    let mut best = (0.0f64, String::new());
    for (di, &(dl, ..)) in decays.iter().enumerate() {
        print!("{:<10}", dl);
        for (gi, &g) in gammas.iter().enumerate() {
            let cov = grid[di][gi].coverage() * 100.0;
            print!("{:>7.2}%", cov);
            if cov > best.0 {
                best = (cov, format!("decay={dl} γ={g}"));
            }
        }
        println!();
    }
    println!("best: {:.2}%  ({})\n", best.0, best.1);
}

/// §9.3 — base-prior training amount (train_epochs) at the champion config.
#[test]
fn tune_epochs() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_epochs] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let epochs = [1usize, 2, 3, 4, 6];
    let mut out = vec![Metrics::default(); epochs.len()];
    for &c in &trace.configs() {
        let test = trace.for_config(c);
        let train = trace.for_configs_except(c);
        for (ei, &ep) in epochs.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            prior.arrivals_only = true;
            train_predictor(&mut prior, &train, ep);
            let mut v = champion(prior, 0.01, 0.5, 2.0, 0.7, 0.95);
            let mut m = vec![Metrics::default(); 1];
            score_into(&mut v, &test, &[4usize], 4, true, &mut m);
            out[ei].pred_total += m[0].pred_total;
            out[ei].pred_hits += m[0].pred_hits;
            out[ei].miss_total += m[0].miss_total;
            out[ei].miss_covered += m[0].miss_covered;
        }
    }
    println!("\n=== §9.3 Base-prior train_epochs (champion, k=4 cov) ===");
    println!("{:<10} {:>10}", "epochs", "k=4 cov");
    println!("{}", "-".repeat(22));
    for (ei, &ep) in epochs.iter().enumerate() {
        println!("{:<10} {:>9.2}%", ep, out[ei].coverage() * 100.0);
    }
    println!();
}

/// §9.4 — locked champion: coverage vs fan-out k (deployment knob).
#[test]
fn champion_kcurve() {
    let Some(records) = load_fixture() else {
        println!("\n[champion_kcurve] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 3, 4, 6, 8];
    let mut out = vec![Metrics::default(); ks.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let mut v = champion(prior, 0.01, 0.5, 2.0, 0.7, 0.95);
        score_into(
            &mut v,
            &trace.for_config(c),
            &ks,
            *ks.iter().max().unwrap(),
            true,
            &mut out,
        );
    }
    println!("\n=== §9.4 Locked champion — coverage & precision vs k ===");
    println!("{:<6} {:>10} {:>12}", "k", "cov", "precision");
    println!("{}", "-".repeat(30));
    for (ki, &k) in ks.iter().enumerate() {
        println!(
            "{:<6} {:>9.1}% {:>11.1}%",
            k,
            out[ki].coverage() * 100.0,
            out[ki].precision() * 100.0
        );
    }
    println!();
}

/// §9.5 — demote LFRU frequency-bonus scale tuning vs LRU.
#[test]
fn tune_demote() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_demote] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let prior = global_stickiness(&trace, 16);
    let base_cohort = (l as f64 * trace.avg_set_size()).max(1.0);
    // LFRU bonus scale = cohort * mult; mult=0 is plain LRU.
    let mults = [0.0f64, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    let budgets = [0.5f64, 0.7];
    let mut acc = vec![vec![(0u64, 0u64); budgets.len()]; mults.len()];
    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let mut pmi0 = MatrixPredictor::new(l, e, Scoring::Pmi { alpha: 0.5 });
        train_predictor(&mut pmi0, &train, 2);
        let mut recur0 = RecurrenceModel::new(l, e);
        train_recur(&mut recur0, &train);
        let test = trace.for_config(c);
        let distinct = {
            let mut s = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            s.insert(ly * e + x);
                        }
                    }
                }
            }
            s.len()
        };
        for (bi, &frac) in budgets.iter().enumerate() {
            let budget = ((distinct as f64) * frac).ceil() as usize;
            for (mi, &mult) in mults.iter().enumerate() {
                let pol = if mult == 0.0 {
                    Demote::Lru
                } else {
                    Demote::Lfru
                };
                let mut r = recur0.clone();
                let mut p = pmi0.clone();
                let (misses, accesses, _, _) = demote_sim(
                    &test,
                    budget,
                    pol,
                    &mut r,
                    prior,
                    0.05,
                    8,
                    &mut p,
                    base_cohort * mult,
                );
                acc[mi][bi].0 += misses;
                acc[mi][bi].1 += accesses;
            }
        }
    }
    println!("\n=== §9.5 Demote LFRU freq-bonus scale (miss-rate, lower better) ===");
    println!(
        "{:<14} {:>12} {:>12}",
        "scale(×cohort)", "budget=50%", "budget=70%"
    );
    println!("{}", "-".repeat(40));
    for (mi, &mult) in mults.iter().enumerate() {
        let tag = if mult == 0.0 {
            "LRU".to_string()
        } else {
            format!("LFRU×{mult}")
        };
        println!(
            "{:<14} {:>11.1}% {:>11.1}%",
            tag,
            ratio(acc[mi][0].0 as usize, acc[mi][0].1 as usize) * 100.0,
            ratio(acc[mi][1].0 as usize, acc[mi][1].1 as usize) * 100.0
        );
    }
    println!();
}

/// Outcome of the end-to-end cache simulation (counts over decode accesses).
#[derive(Clone, Copy, Default)]
struct Outcome {
    hit: u64,      // resident, never evicted
    soft: u64,     // resident only via prefetch (overlapped load)
    hard: u64,     // demand-load on the critical path (stall)
    prefetch: u64, // total experts speculatively loaded (used + wasted)
}

impl Outcome {
    fn accesses(&self) -> u64 {
        self.hit + self.soft + self.hard
    }
    /// Stall-weighted cost per 100 accesses: hard×`w_hard` + soft×`w_soft`.
    fn stall_cost(&self, w_hard: f64, w_soft: f64) -> f64 {
        100.0 * (self.hard as f64 * w_hard + self.soft as f64 * w_soft)
            / self.accesses().max(1) as f64
    }
    /// PCIe transfers per 100 accesses (bandwidth): demand loads + all prefetch.
    fn bandwidth(&self) -> f64 {
        100.0 * (self.hard + self.prefetch) as f64 / self.accesses().max(1) as f64
    }
}

/// End-to-end cache-outcome simulation: the champion predictor drives free-slot
/// prefetch (one layer ahead), LFRU×16 drives eviction, with end-of-pass batch
/// eviction creating prefetch headroom.  Classifies every decode-layer expert
/// access into hit / soft-miss / hard-miss and totals the prefetch volume.
fn cache_outcome(
    test: &Trace,
    budget: usize,
    champ: &mut Velocity,
    cohort: f64,
    evict_frac: f64,
    k_prefetch: usize,
) -> Outcome {
    let e = test.num_experts;
    let nl = test.num_layers;
    let lfru = |last: u64, freq: u64| last as f64 + (freq as f64).ln_1p() * cohort * 16.0;
    // key -> (last_used, freq, pending_prefetch)
    let mut resident: HashMap<usize, (u64, u64, bool)> = HashMap::new();
    let mut clock = 0u64;
    let mut o = Outcome::default();

    for pass in &test.passes {
        if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
            continue;
        }
        champ.reset_pass();
        for (l, ex) in &pass.layers {
            for &x in ex {
                let key = l * e + x;
                clock += 1;
                if let Some(s) = resident.get_mut(&key) {
                    if s.2 {
                        o.soft += 1;
                        s.2 = false;
                    } else {
                        o.hit += 1;
                    }
                    s.0 = clock;
                    s.1 += 1;
                } else {
                    o.hard += 1;
                    // Demand-load on the critical path: evict an LFRU victim
                    // (never a pinned-layer expert) if the cache is full.
                    if resident.len() >= budget {
                        if let Some(v) = resident
                            .iter()
                            .filter(|(&k, _)| k / e >= PINNED)
                            .min_by(|a, b| {
                                lfru(a.1 .0, a.1 .1)
                                    .partial_cmp(&lfru(b.1 .0, b.1 .1))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(k, _)| *k)
                        {
                            resident.remove(&v);
                        }
                    }
                    resident.insert(key, (clock, 1, false));
                }
            }
            champ.observe(*l, ex);
            // Free-slot-only prefetch of the next layer's predicted experts.
            if l + 1 < nl && l + 1 >= PINNED {
                for pe in champ.predict(*l, ex, k_prefetch) {
                    let pk = (l + 1) * e + pe;
                    if resident.len() >= budget {
                        break;
                    }
                    if resident.get(&pk).is_none() {
                        resident.insert(pk, (clock, 0, true));
                        o.prefetch += 1;
                    }
                }
            }
        }
        // End-of-pass: batch-evict `evict_frac` of non-pinned slots (headroom).
        let np_count = resident.iter().filter(|(&k, _)| k / e >= PINNED).count();
        let nevict = ((np_count as f64) * evict_frac).ceil() as usize;
        if nevict > 0 {
            let mut ranked: Vec<(usize, f64)> = resident
                .iter()
                .filter(|(&k, _)| k / e >= PINNED)
                .map(|(&k, &(lu, fr, _))| (k, lfru(lu, fr)))
                .collect();
            ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(k, _) in ranked.iter().take(nevict) {
                resident.remove(&k);
            }
        }
    }
    o
}

/// §10 — modelled cache outcome at 60% VRAM budget, 5% eviction, our models.
#[test]
fn cache_model_60() {
    let Some(records) = load_fixture() else {
        println!("\n[cache_model_60] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let cohort = (l as f64 * trace.avg_set_size()).max(1.0);

    // (budget_frac, K_prefetch, evict_frac).  Train once per fold, clone per scenario.
    let scen: Vec<(f64, usize, f64)> = vec![
        // 60% budget: K and eviction-headroom sweep.
        (0.60, 4, 0.05),
        (0.60, 8, 0.05),
        (0.60, 8, 0.10),
        (0.60, 8, 0.20),
        // budget sweep at the chosen K=8 / 5% evict.
        (0.40, 8, 0.05),
        (0.50, 8, 0.05),
        (0.70, 8, 0.05),
        (0.80, 8, 0.05),
    ];
    let mut agg = vec![(0u64, 0u64, 0u64); scen.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let base = champion(prior, 0.01, 0.5, 2.0, 0.8, 0.97);
        let test = trace.for_config(c);
        let distinct = {
            let mut set = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            set.insert(ly * e + x);
                        }
                    }
                }
            }
            set.len()
        };
        for (si, &(bf, kp, ef)) in scen.iter().enumerate() {
            let budget = ((distinct as f64) * bf).ceil() as usize;
            let mut champ = base.clone();
            let o = cache_outcome(&test, budget, &mut champ, cohort, ef, kp);
            agg[si].0 += o.hit;
            agg[si].1 += o.soft;
            agg[si].2 += o.hard;
        }
    }

    let row = |bf: f64, kp: usize, ef: f64, a: (u64, u64, u64)| {
        let (h, s, hd) = a;
        let tot = (h + s + hd).max(1) as f64;
        println!(
            "{:>6.0}% {:>4} {:>6.0}% {:>9.1}% {:>10.1}% {:>10.1}% {:>11.1}%",
            bf * 100.0,
            kp,
            ef * 100.0,
            h as f64 / tot * 100.0,
            s as f64 / tot * 100.0,
            hd as f64 / tot * 100.0,
            (h + s) as f64 / tot * 100.0,
        );
    };

    println!("\n=== §10 Modelled cache outcome — champion prefetch + LFRU×16 evict ===");
    println!("21-fold held-out, all decode accesses\n");
    println!(
        "{:>7} {:>4} {:>6} {:>10} {:>11} {:>11} {:>12}",
        "budget", "K", "evict", "hit", "soft-miss", "hard-miss", "no-stall"
    );
    println!("{}", "-".repeat(64));
    println!("  -- 60% budget: K and eviction-headroom sweep --");
    for si in 0..4 {
        let (bf, kp, ef) = scen[si];
        row(bf, kp, ef, agg[si]);
    }
    println!("  -- budget sweep (K=8, 5% evict) --");
    row(0.40, 8, 0.05, agg[4]);
    row(0.50, 8, 0.05, agg[5]);
    row(0.60, 8, 0.05, agg[1]);
    row(0.70, 8, 0.05, agg[6]);
    row(0.80, 8, 0.05, agg[7]);
    println!(
        "\nhit = resident (never evicted); soft = prefetched (overlapped); hard = demand-load stall.\n"
    );
}

/// §11 — cost-driven optimisation.  Objective: minimise stall cost
/// (hard×10 + soft×1) — hard misses stall the pipeline on full PCIe latency,
/// soft misses are overlapped.  Bandwidth (demand + prefetch transfers) is the
/// second dimension.  Sweeps forced churn × K (preloads per layer) at 60% VRAM.
#[test]
fn cost_optimize() {
    let Some(records) = load_fixture() else {
        println!("\n[cost_optimize] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let cohort = (l as f64 * trace.avg_set_size()).max(1.0);
    const BUDGET_FRAC: f64 = 0.60;
    const W_HARD: f64 = 10.0;
    const W_SOFT: f64 = 1.0;
    let churns = [0.0f64, 0.005, 0.01, 0.02, 0.03];
    let ks = [4usize, 6, 8, 12, 16];
    let mut agg = vec![vec![Outcome::default(); ks.len()]; churns.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let base = champion(prior, 0.01, 0.5, 2.0, 0.8, 0.97);
        let test = trace.for_config(c);
        let distinct = {
            let mut set = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            set.insert(ly * e + x);
                        }
                    }
                }
            }
            set.len()
        };
        let budget = ((distinct as f64) * BUDGET_FRAC).ceil() as usize;
        for (ci, &ch) in churns.iter().enumerate() {
            for (ki, &k) in ks.iter().enumerate() {
                let mut champ = base.clone();
                let o = cache_outcome(&test, budget, &mut champ, cohort, ch, k);
                let a = &mut agg[ci][ki];
                a.hit += o.hit;
                a.soft += o.soft;
                a.hard += o.hard;
                a.prefetch += o.prefetch;
            }
        }
    }

    let grid = |title: &str, f: &dyn Fn(&Outcome) -> f64| {
        println!("\n{title}");
        print!("{:<10}", "churn\\K");
        for k in ks {
            print!("{:>8}", k);
        }
        println!();
        for (ci, &ch) in churns.iter().enumerate() {
            print!("{:<10}", format!("{:.0}%", ch * 100.0));
            for (ki, _) in ks.iter().enumerate() {
                print!("{:>8.2}", f(&agg[ci][ki]));
            }
            println!();
        }
    };

    println!("\n=== §11 Cost optimisation (60% VRAM; hard×10 + soft×1) ===");
    grid("-- stall cost / 100 accesses (LOWER = better) --", &|o| {
        o.stall_cost(W_HARD, W_SOFT)
    });
    grid("-- hard-miss % (the stalls) --", &|o| {
        100.0 * o.hard as f64 / o.accesses().max(1) as f64
    });
    grid("-- soft-miss % (overlapped) --", &|o| {
        100.0 * o.soft as f64 / o.accesses().max(1) as f64
    });
    grid("-- bandwidth: PCIe transfers / 100 accesses --", &|o| {
        o.bandwidth()
    });
    grid("-- prefetch waste % (loaded, never used) --", &|o| {
        100.0 * (o.prefetch.saturating_sub(o.soft)) as f64 / o.prefetch.max(1) as f64
    });

    // Bandwidth-constrained optimum: min stall cost whose PCIe transfer rate
    // fits a budget B (transfers / 100 accesses).  Maps the sweet spot to the
    // hardware's available PCIe bandwidth.
    println!("\n-- bandwidth-constrained sweet spot (min stall cost s.t. transfers/100acc ≤ B) --");
    println!(
        "{:>8} {:>8} {:>5} {:>10} {:>9} {:>9}",
        "B", "churn", "K", "stallcost", "hard%", "soft%"
    );
    for &cap in &[11.0f64, 12.0, 14.0, 17.0, 21.0, 99.0] {
        let mut best = (f64::MAX, 0.0f64, 0usize, 0.0f64, 0.0f64);
        for (ci, &ch) in churns.iter().enumerate() {
            for (ki, &k) in ks.iter().enumerate() {
                let o = &agg[ci][ki];
                if o.bandwidth() <= cap {
                    let cost = o.stall_cost(W_HARD, W_SOFT);
                    if cost < best.0 {
                        let acc = o.accesses().max(1) as f64;
                        best = (
                            cost,
                            ch,
                            k,
                            100.0 * o.hard as f64 / acc,
                            100.0 * o.soft as f64 / acc,
                        );
                    }
                }
            }
        }
        let tag = if cap >= 99.0 {
            "∞".to_string()
        } else {
            format!("{cap:.0}")
        };
        println!(
            "{:>8} {:>7.1}% {:>5} {:>10.2} {:>8.2}% {:>8.2}%",
            tag,
            best.1 * 100.0,
            best.2,
            best.0,
            best.3,
            best.4
        );
    }
    println!();
}

/// §9.1 — champion fine-tune: fine β×α grid on the full stack (arrival-
/// specialised prior+session).  Trains once per fold, scores all (β,α).
#[test]
fn tune_champion() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_champion] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let betas = [0.005f64, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04];
    let alphas = [0.40f64, 0.45, 0.50, 0.55, 0.60];
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; betas.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let mut sp = SessionPrior::new(prior, 0.02, 0.5);
        sp.session.arrivals_only = true;
        let test = trace.for_config(c);
        for pass in &test.passes {
            if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                continue;
            }
            sp.session.reset_pass();
            let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pass.layers.len());
            for (li, ex) in &pass.layers {
                let mut row = Vec::with_capacity(betas.len() * alphas.len());
                for &b in &betas {
                    for &a in &alphas {
                        let p = sp
                            .scores_with(*li, ex, b, a)
                            .map(|s| top_k_excluding(&s, ex, 4))
                            .unwrap_or_default();
                        row.push(p);
                    }
                }
                preds.push(row);
                sp.session.observe(*li, ex);
            }
            for i in 0..pass.layers.len().saturating_sub(1) {
                let (li, ref cur) = pass.layers[i];
                let (ln, ref next) = pass.layers[i + 1];
                if ln != li + 1 || ln < PINNED {
                    continue;
                }
                for bi in 0..betas.len() {
                    for ai in 0..alphas.len() {
                        grid[bi][ai].add(&preds[i][bi * alphas.len() + ai], cur, next);
                    }
                }
            }
        }
    }

    println!("\n=== §9.1 Champion β×α fine-tune (k=4 decode-cov, arrivals, +online) ===");
    print!("{:<8}", "β\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for (bi, &b) in betas.iter().enumerate() {
        print!("{:<8}", format!("{b}"));
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[bi][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, b, alphas[ai]);
            }
        }
        println!();
    }
    println!("best: cov={:.2}%  β={}  α={}\n", best.0, best.1, best.2);
}

/// §8b — literal "fork": the session model is a *clone* of the base (strong
/// warm-start) that then fast-adapts via per-token decay and may re-score with a
/// different α.  Sweeps λ (adaptation speed) × α.
#[test]
fn loocv_fork() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_fork] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let lambdas = [1.0f64, 0.99, 0.97, 0.95];
    let alphas = [0.4f64, 0.5];
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; lambdas.len()];

    for &c in &trace.configs() {
        let mut base = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut base, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (lami, &lam) in lambdas.iter().enumerate() {
            for (ai, &alpha) in alphas.iter().enumerate() {
                // Fork = clone of the base, fast-adapting within the session.
                let mut fork = base.clone();
                fork.decay = lam;
                fork.scoring = Scoring::Pmi { alpha };
                let mut m = vec![Metrics::default(); 1];
                score_into(&mut fork, &test, &[4usize], 4, true, &mut m);
                grid[lami][ai].pred_total += m[0].pred_total;
                grid[lami][ai].pred_hits += m[0].pred_hits;
                grid[lami][ai].miss_total += m[0].miss_total;
                grid[lami][ai].miss_covered += m[0].miss_covered;
            }
        }
    }

    println!("\n=== §8b Literal fork: clone(base) + fast decay (k=4 decode-cov, +online) ===");
    println!("reference: SessionPrior champion = 42.5%");
    print!("{:<8}", "λ\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for (lami, &lam) in lambdas.iter().enumerate() {
        print!("{:<8.3}", lam);
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[lami][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, lam, alphas[ai]);
            }
        }
        println!();
    }
    println!("best fork: {:.1}%  λ={} α={}\n", best.0, best.1, best.2);
}

/// §8c — SessionPrior (count-level blend) with a fast-learning session fork:
/// give the session matrix its own decay so it tracks the most recent tokens.
#[test]
fn loocv_sessiondecay() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_sessiondecay] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let decays = [1.0f64, 0.997, 0.99, 0.98, 0.96];
    let mut out = vec![Metrics::default(); decays.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (di, &sd) in decays.iter().enumerate() {
            let mut sp = SessionPrior::new(prior.clone(), 0.02, 0.5);
            sp.session.decay = sd; // fast-learning fork
            let mut m = vec![Metrics::default(); 1];
            score_into(&mut sp, &test, &[4usize], 4, true, &mut m);
            out[di].pred_total += m[0].pred_total;
            out[di].pred_hits += m[0].pred_hits;
            out[di].miss_total += m[0].miss_total;
            out[di].miss_covered += m[0].miss_covered;
        }
    }
    println!("\n=== §8c SessionPrior + fast-learning fork (session decay, k=4, +online) ===");
    println!("{:<14} {:>10}", "session decay", "k=4 cov");
    println!("{}", "-".repeat(26));
    for (di, &sd) in decays.iter().enumerate() {
        println!(
            "{:<14} {:>9.1}%",
            format!("{sd}"),
            out[di].coverage() * 100.0
        );
    }
    println!();
}

/// §7.6 — velocity (rising-trend) boost on top of SessionPrior.
#[test]
fn loocv_velocity() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_velocity] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let gammas = [0.0f64, 0.5, 1.0, 2.0];
    let mut out = vec![vec![Metrics::default(); ks.len()]; gammas.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (gi, &gamma) in gammas.iter().enumerate() {
            let sp = SessionPrior::new(prior.clone(), 0.02, 0.5);
            let mut v = Velocity::new(sp, gamma);
            score_into(&mut v, &test, &ks, 4, true, &mut out[gi]);
        }
    }
    println!("\n=== §7.6 Velocity boost (SessionPrior β=0.02 α=0.5, +online) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "γ (velocity)", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(48));
    for (gi, &gamma) in gammas.iter().enumerate() {
        let tag = if gamma == 0.0 {
            "0 (sp only)".to_string()
        } else {
            format!("{gamma}")
        };
        println!(
            "{:<14} {:>9.1}% {:>9.1}% {:>9.1}%",
            tag,
            out[gi][0].coverage() * 100.0,
            out[gi][1].coverage() * 100.0,
            out[gi][2].coverage() * 100.0
        );
    }
    println!();
}

/// Shared bounded top-k selection over a score vector, excluding `active`.
fn top_k_excluding(scores: &[f64], active: &[usize], k: usize) -> Vec<usize> {
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

/// §7.2 — Bayesian prior+session interpolation: sweep β (prior weight) at α=0.5.
#[test]
fn loocv_sessionprior() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_sessionprior] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [4usize];
    let betas = [0.0f64, 0.02, 0.04, 0.06, 0.08, 0.12];
    let alphas = [0.4f64, 0.5, 0.6];

    // grid[beta][alpha] @ k=4.  Train one prior per fold, reuse across (β,α).
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; betas.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (bi, &beta) in betas.iter().enumerate() {
            for (ai, &alpha) in alphas.iter().enumerate() {
                let mut p = SessionPrior::new(prior.clone(), beta, alpha);
                let mut m = vec![Metrics::default(); ks.len()];
                score_into(&mut p, &test, &ks, 4, true, &mut m);
                grid[bi][ai].pred_total += m[0].pred_total;
                grid[bi][ai].pred_hits += m[0].pred_hits;
                grid[bi][ai].miss_total += m[0].miss_total;
                grid[bi][ai].miss_covered += m[0].miss_covered;
            }
        }
    }

    println!("\n=== §7.2 Bayesian prior+session interpolation (k=4 decode-cov, +online) ===");
    print!("{:<10}", "β\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for (bi, &beta) in betas.iter().enumerate() {
        print!("{:<10}", format!("{beta}"));
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[bi][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, beta, alphas[ai]);
            }
        }
        println!();
    }
    println!("best: cov={:.1}%  β={}  α={}\n", best.0, best.1, best.2);
}

/// Train a matrix over all passes, optionally mass-weighting observations.
fn train_mass(p: &mut MatrixPredictor, trace: &Trace, epochs: usize, mass: bool) {
    for _ in 0..epochs {
        for pass in &trace.passes {
            p.reset_pass();
            for (i, (li, ex)) in pass.layers.iter().enumerate() {
                if mass {
                    let m: Vec<f64> = pass.masses[i].iter().map(|&x| x as f64).collect();
                    p.observe_mass(*li, ex, &m);
                } else {
                    p.observe(*li, ex);
                }
            }
        }
    }
}

/// §7.3 — mass-weighted observations (routing-weight P7).  Best base config
/// (SessionPrior β=0.02, α=0.5); compares uniform vs mass-weighted counts.
#[test]
fn loocv_mass() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_mass] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let variants = [("uniform", false), ("mass-weighted", true)];
    let mut out = vec![vec![Metrics::default(); ks.len()]; variants.len()];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let test = trace.for_config(c);
        for (vi, &(_, mass)) in variants.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            train_mass(&mut prior, &train, 2, mass);
            let mut sp = SessionPrior::new(prior, 0.02, 0.5);
            for pass in &test.passes {
                if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                    continue;
                }
                sp.session.reset_pass();
                let mut preds: Vec<Vec<usize>> = Vec::with_capacity(pass.layers.len());
                for (i, (li, ex)) in pass.layers.iter().enumerate() {
                    preds.push(sp.predict(*li, ex, 4));
                    if mass {
                        let m: Vec<f64> = pass.masses[i].iter().map(|&x| x as f64).collect();
                        sp.session.observe_mass(*li, ex, &m);
                    } else {
                        sp.session.observe(*li, ex);
                    }
                }
                for i in 0..pass.layers.len().saturating_sub(1) {
                    let (li, ref cur) = pass.layers[i];
                    let (ln, ref next) = pass.layers[i + 1];
                    if ln != li + 1 || ln < PINNED {
                        continue;
                    }
                    for (ki, &k) in ks.iter().enumerate() {
                        let pr = &preds[i];
                        out[vi][ki].add(&pr[..k.min(pr.len())], cur, next);
                    }
                }
            }
        }
    }

    println!("\n=== §7.3 Mass-weighted observations (SessionPrior β=0.02 α=0.5, +online) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "variant", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(50));
    for (vi, &(label, _)) in variants.iter().enumerate() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            out[vi][0].coverage() * 100.0,
            out[vi][1].coverage() * 100.0,
            out[vi][2].coverage() * 100.0
        );
    }
    println!();
}

/// §7.4 — arrival-specialised training (only learn cold transitions).
#[test]
fn loocv_arrivals() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_arrivals] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let variants = [("full", false), ("arrivals-only", true)];
    let mut out = vec![vec![Metrics::default(); ks.len()]; variants.len()];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let test = trace.for_config(c);
        for (vi, &(_, arr)) in variants.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            prior.arrivals_only = arr;
            train_predictor(&mut prior, &train, 2);
            let mut sp = SessionPrior::new(prior, 0.02, 0.5);
            sp.session.arrivals_only = arr;
            score_into(&mut sp, &test, &ks, 4, true, &mut out[vi]);
        }
    }

    println!("\n=== §7.4 Arrival-specialised training (SessionPrior β=0.02 α=0.5, +online) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "training", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(50));
    for (vi, &(label, _)) in variants.iter().enumerate() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            out[vi][0].coverage() * 100.0,
            out[vi][1].coverage() * 100.0,
            out[vi][2].coverage() * 100.0
        );
    }
    println!();
}

/// Recency-fusion predictor: combines the SessionPrior cross-layer score with a
/// per-(target-layer, expert) recency signal (how recently an expert appeared
/// at the target layer) via weighted reciprocal-rank fusion.
#[derive(Clone)]
struct Recency {
    sp: SessionPrior,
    ema: Vec<Vec<f64>>, // [layer][expert] decayed presence
    decay: f64,
    gamma: f64, // weight of the recency channel
}

impl Recency {
    fn new(sp: SessionPrior, decay: f64, gamma: f64) -> Self {
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

/// §7.5 — recency-fusion: add a same-layer recency channel to SessionPrior.
#[test]
fn loocv_combo() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_combo] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let gammas = [0.0f64, 0.25, 0.5, 1.0, 2.0];
    let mut out = vec![vec![Metrics::default(); ks.len()]; gammas.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (gi, &gamma) in gammas.iter().enumerate() {
            let sp = SessionPrior::new(prior.clone(), 0.02, 0.5);
            let mut r = Recency::new(sp, 0.8, gamma);
            score_into(&mut r, &test, &ks, 4, true, &mut out[gi]);
        }
    }

    println!("\n=== §7.5 Recency-fusion (SessionPrior + same-layer recency, +online) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "γ (recency)", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(48));
    for (gi, &gamma) in gammas.iter().enumerate() {
        let tag = if gamma == 0.0 {
            "0 (sp only)".to_string()
        } else {
            format!("{gamma}")
        };
        println!(
            "{:<14} {:>9.1}% {:>9.1}% {:>9.1}%",
            tag,
            out[gi][0].coverage() * 100.0,
            out[gi][1].coverage() * 100.0,
            out[gi][2].coverage() * 100.0
        );
    }
    println!();
}

/// §7.1 — joint α × λ fine-tune with *within-session* momentum (decay applied
/// only during the held-out generation, not training).  Trains one full matrix
/// per fold, clones per λ.  Prints the k=4 held-out coverage grid (λ × α).
#[test]
fn loocv_tune() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_tune] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let lambdas = [1.0f64, 0.997, 0.995, 0.99, 0.985, 0.98, 0.97];
    let alphas = [0.3f64, 0.4, 0.5, 0.6, 0.7];
    let maxk = 4usize;
    // grid[lambda][alpha] = Metrics @ k=4
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; lambdas.len()];

    for &c in &trace.configs() {
        let mut base = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut base, &trace.for_configs_except(c), 2); // full prior, no decay
        let test = trace.for_config(c);
        for (lami, &lam) in lambdas.iter().enumerate() {
            let mut p = base.clone();
            p.decay = lam; // within-session momentum only
            for pass in &test.passes {
                if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                    continue;
                }
                p.reset_pass();
                let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pass.layers.len());
                for (li, ex) in &pass.layers {
                    preds.push(
                        alphas
                            .iter()
                            .map(|&a| p.score_topk(*li, ex, maxk, Scoring::Pmi { alpha: a }))
                            .collect(),
                    );
                    p.observe(*li, ex);
                }
                for i in 0..pass.layers.len().saturating_sub(1) {
                    let (li, ref cur) = pass.layers[i];
                    let (ln, ref next) = pass.layers[i + 1];
                    if ln != li + 1 || ln < PINNED {
                        continue;
                    }
                    for ai in 0..alphas.len() {
                        grid[lami][ai].add(&preds[i][ai], cur, next);
                    }
                }
            }
        }
    }

    println!("\n=== §7.1 Promote α × λ fine-tune (k=4 decode-cov, within-session momentum) ===");
    print!("{:<8}", "λ\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 1.0f64, 0.5f64);
    for (lami, &lam) in lambdas.iter().enumerate() {
        print!("{:<8.3}", lam);
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[lami][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, lam, alphas[ai]);
            }
        }
        println!();
    }
    println!("best: cov={:.1}%  λ={}  α={}\n", best.0, best.1, best.2);
}

/// §5.3 — promote input-signal ablation (channels), pmi(0.5), +online.
#[test]
fn loocv_inputs() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_inputs] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    use Channel::*;
    let configs: Vec<(&str, Vec<Channel>)> = vec![
        ("L-1 (base)", vec![Prev]),
        ("L-1,L-2", vec![Prev, Prev2]),
        ("L-1,recur", vec![Prev, Recur]),
        ("L-1,pinned", vec![Prev, Pinned]),
        ("all", vec![Prev, Prev2, Recur, Pinned]),
    ];
    println!("\n=== Promote input ablation (pmi 0.5, +online, train_epochs=2) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "inputs", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(48));
    for (label, chans) in &configs {
        let chans = chans.clone();
        let m = loocv_metric(
            &trace,
            || MultiSource::new(l, e, chans.clone()),
            2,
            &ks,
            true,
        );
        println!(
            "{:<14} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            m[0].coverage() * 100.0,
            m[1].coverage() * 100.0,
            m[2].coverage() * 100.0
        );
    }
    println!();
}

/// Score a trained predictor on one held-out trace, folding into `m` (one
/// Metrics per k).  `learn` continues causal online learning during the test.
fn score_into<P: Predictor>(
    p: &mut P,
    test: &Trace,
    ks: &[usize],
    maxk: usize,
    learn: bool,
    m: &mut [Metrics],
) {
    for pass in &test.passes {
        if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
            continue;
        }
        p.reset_pass();
        let mut preds: Vec<Vec<usize>> = Vec::with_capacity(pass.layers.len());
        for (li, ex) in &pass.layers {
            preds.push(p.predict(*li, ex, maxk));
            if learn {
                p.observe(*li, ex);
            }
        }
        for i in 0..pass.layers.len().saturating_sub(1) {
            let (li, ref cur) = pass.layers[i];
            let (ln, ref next) = pass.layers[i + 1];
            if ln != li + 1 || ln < PINNED {
                continue;
            }
            for (ki, &k) in ks.iter().enumerate() {
                let pr = &preds[i];
                m[ki].add(&pr[..k.min(pr.len())], cur, next);
            }
        }
    }
}

/// Generic 21-fold LOOCV: build a fresh predictor per fold via `make`, train on
/// the other 20 configs, score the held-out one.  Returns one Metrics per k.
fn loocv_metric<P: Predictor>(
    trace: &Trace,
    mut make: impl FnMut() -> P,
    epochs: usize,
    ks: &[usize],
    learn: bool,
) -> Vec<Metrics> {
    let mut m = vec![Metrics::default(); ks.len()];
    let maxk = *ks.iter().max().unwrap();
    for &c in &trace.configs() {
        let mut p = make();
        train_predictor(&mut p, &trace.for_configs_except(c), epochs);
        score_into(&mut p, &trace.for_config(c), ks, maxk, learn, &mut m);
    }
    m
}

/// §5.2 — promote matrix size (layer resolution) ablation, pmi(0.5), +online.
#[test]
fn loocv_size() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_size] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let sc = Scoring::Pmi { alpha: 0.5 };
    println!("\n=== Promote size ablation (pmi 0.5, +online, train_epochs=2) ===");
    println!(
        "{:<12} {:>10} {:>10} {:>10}",
        "resolution", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(46));
    for (label, group) in [
        ("per-pair", 1usize),
        ("group-4", 4),
        ("group-8", 8),
        ("shared", l),
    ] {
        let m = loocv_metric(
            &trace,
            || MatrixPredictor::new_cfg(l, e, sc, group, 1.0),
            2,
            &ks,
            true,
        );
        println!(
            "{:<12} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            m[0].coverage() * 100.0,
            m[1].coverage() * 100.0,
            m[2].coverage() * 100.0
        );
    }
    println!();
}

/// §5.4 — promote scoring-formula sweep.  Transform rules share one trained
/// matrix per fold (counts identical); momentum re-trains with a decay.
#[test]
fn loocv_formula() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_formula] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let maxk = 4usize;

    // Transform rules evaluated on one stationary matrix per fold.
    let transforms: Vec<(&str, Scoring)> = vec![
        ("raw", Scoring::Raw),
        ("conditional", Scoring::Conditional),
        ("pmi(0.25)", Scoring::Pmi { alpha: 0.25 }),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
        ("pmi(0.75)", Scoring::Pmi { alpha: 0.75 }),
        ("pmi(1.0)", Scoring::Pmi { alpha: 1.0 }),
        ("dirichlet(2)", Scoring::Dirichlet { alpha: 2.0 }),
        ("wilson(1.0)", Scoring::Wilson { z: 1.0 }),
    ];
    let mut mt = vec![vec![Metrics::default(); ks.len()]; transforms.len()];
    for &c in &trace.configs() {
        let mut p = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut p, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for pass in &test.passes {
            if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                continue;
            }
            p.reset_pass();
            let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pass.layers.len());
            for (li, ex) in &pass.layers {
                preds.push(
                    transforms
                        .iter()
                        .map(|&(_, s)| p.score_topk(*li, ex, maxk, s))
                        .collect(),
                );
                p.observe(*li, ex); // +online
            }
            for i in 0..pass.layers.len().saturating_sub(1) {
                let (li, ref cur) = pass.layers[i];
                let (ln, ref next) = pass.layers[i + 1];
                if ln != li + 1 || ln < PINNED {
                    continue;
                }
                for ti in 0..transforms.len() {
                    for (ki, &k) in ks.iter().enumerate() {
                        let pr = &preds[i][ti];
                        mt[ti][ki].add(&pr[..k.min(pr.len())], cur, next);
                    }
                }
            }
        }
    }

    // Momentum: re-trained matrices with per-pass decay, scored as pmi(0.5).
    let sc = Scoring::Pmi { alpha: 0.5 };
    let momentum: Vec<(&str, f64)> = vec![
        ("momentum(0.99)", 0.99),
        ("momentum(0.97)", 0.97),
        ("momentum(0.95)", 0.95),
        ("momentum(0.90)", 0.90),
        ("momentum(0.80)", 0.80),
    ];
    let mom_metrics: Vec<Vec<Metrics>> = momentum
        .iter()
        .map(|&(_, d)| {
            loocv_metric(
                &trace,
                || MatrixPredictor::new_cfg(l, e, sc, 1, d),
                2,
                &ks,
                true,
            )
        })
        .collect();

    println!("\n=== Promote formula sweep (+online, train_epochs=2) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "formula", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(50));
    for ti in 0..transforms.len() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            transforms[ti].0,
            mt[ti][0].coverage() * 100.0,
            mt[ti][1].coverage() * 100.0,
            mt[ti][2].coverage() * 100.0
        );
    }
    for (mi, &(label, _)) in momentum.iter().enumerate() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            mom_metrics[mi][0].coverage() * 100.0,
            mom_metrics[mi][1].coverage() * 100.0,
            mom_metrics[mi][2].coverage() * 100.0
        );
    }
    println!();
}

/// Eviction policy for the demote evaluation.  Lower score = evict first.
#[derive(Clone, Copy, PartialEq)]
enum Demote {
    /// Least-recently-used (production stand-in; temporal locality).
    Lru,
    /// Least-frequently-used (long-tail popularity).
    Lfu,
    /// Frequency-weighted recency: frequent experts get a recency bonus.
    Lfru,
    /// Learned recurrence keep-value, recency as refinement (recency primary).
    Blend,
    /// Pure learned recurrence keep-value (no recency).
    Recurrence,
}

/// Train a recurrence model on a multi-prompt trace, resetting cross-token
/// state at each prompt boundary.
fn train_recur(r: &mut RecurrenceModel, trace: &Trace) {
    let mut cfg: Option<u16> = None;
    for pass in &trace.passes {
        if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
            continue;
        }
        if cfg != Some(pass.config) {
            r.reset();
            cfg = Some(pass.config);
        }
        for (l, ex) in &pass.layers {
            r.observe(*l, ex);
        }
    }
}

/// Production-style demote simulation on one held-out prompt: end-of-pass batch
/// eviction of the bottom `evict_frac` (group decision) + free-slot PMI
/// prefetch.  Returns (misses, accesses, evicted, re-queried-within-W).
#[allow(clippy::too_many_arguments)]
fn demote_sim(
    test: &Trace,
    budget: usize,
    policy: Demote,
    recur: &mut RecurrenceModel,
    prior: f64,
    evict_frac: f64,
    regret_w: usize,
    pmi: &mut MatrixPredictor,
    cohort: f64,
) -> (u64, u64, u64, u64) {
    let e = test.num_experts;
    let decode: Vec<&Pass> = test
        .passes
        .iter()
        .filter(|p| p.layers.iter().all(|(_, x)| x.len() <= 16))
        .collect();
    // key = layer*E + expert → token indices it is accessed at (for regret).
    let mut access_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    for (t, pass) in decode.iter().enumerate() {
        for (l, ex) in &pass.layers {
            for &x in ex {
                access_tokens.entry(l * e + x).or_default().push(t);
            }
        }
    }

    // resident: key → (last_used, freq)
    let mut resident: HashMap<usize, (u64, u64)> = HashMap::new();
    let mut clock = 0u64;
    let (mut misses, mut accesses, mut evicted, mut requeried) = (0u64, 0u64, 0u64, 0u64);

    // Eviction score (lower = evict first).
    let escore = |key: usize, last_used: u64, freq: u64, recur: &RecurrenceModel| -> f64 {
        let (l, ex) = (key / e, key % e);
        match policy {
            Demote::Lru => last_used as f64,
            Demote::Lfu => freq as f64,
            Demote::Lfru => last_used as f64 + (freq as f64).ln_1p() * cohort,
            Demote::Recurrence => recur.keep_value(l, ex, prior),
            Demote::Blend => last_used as f64 + recur.keep_value(l, ex, prior) * (cohort * 4.0),
        }
    };

    for (t, pass) in decode.iter().enumerate() {
        pmi.reset_pass();
        for (l, ex) in &pass.layers {
            // Free-slot-only PMI prefetch.
            for pe in pmi.score_topk(*l, ex, 4, Scoring::Pmi { alpha: 0.5 }) {
                let key = l * e + pe;
                if resident.contains_key(&key) || resident.len() < budget {
                    clock += 1;
                    resident.entry(key).or_insert((clock, 0)).0 = clock;
                }
            }
            for &x in ex {
                accesses += 1;
                let key = l * e + x;
                clock += 1;
                if let Some(v) = resident.get_mut(&key) {
                    v.0 = clock;
                    v.1 += 1;
                } else {
                    misses += 1;
                    if resident.len() >= budget {
                        // Evict the single worst by policy.
                        if let Some(victim) = resident
                            .iter()
                            .min_by(|a, b| {
                                escore(*a.0, a.1 .0, a.1 .1, recur)
                                    .partial_cmp(&escore(*b.0, b.1 .0, b.1 .1, recur))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(k, _)| *k)
                        {
                            resident.remove(&victim);
                        }
                    }
                    resident.insert(key, (clock, 1));
                }
            }
            recur.observe(*l, ex);
            pmi.observe(*l, ex);
        }

        // End-of-pass batch eviction of the bottom `evict_frac` group.
        let nevict = ((resident.len() as f64) * evict_frac).ceil() as usize;
        if nevict > 0 && resident.len() > nevict {
            let mut ranked: Vec<(usize, f64)> = resident
                .iter()
                .map(|(&k, &(lu, fr))| (k, escore(k, lu, fr, recur)))
                .collect();
            ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(k, _) in ranked.iter().take(nevict) {
                resident.remove(&k);
                evicted += 1;
                if let Some(ts) = access_tokens.get(&k) {
                    if ts.iter().any(|&tt| tt > t && tt <= t + regret_w) {
                        requeried += 1;
                    }
                }
            }
        }
    }
    (misses, accesses, evicted, requeried)
}

/// §5.5 — demote group-eviction evaluation via 21-fold LOOCV.
#[test]
fn loocv_demote() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_demote] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let prior = global_stickiness(&trace, 16);
    let cohort = (l as f64 * trace.avg_set_size()).max(1.0);
    const EVICT_FRAC: f64 = 0.05;
    const REGRET_W: usize = 8;

    let policies = [
        ("LRU", Demote::Lru),
        ("LFU", Demote::Lfu),
        ("LFRU", Demote::Lfru),
        ("recurrence", Demote::Recurrence),
        ("blend", Demote::Blend),
    ];
    let budgets = [0.5f64, 0.7];
    // [policy][budget] → (misses, accesses, evicted, requeried)
    let mut acc = vec![vec![(0u64, 0u64, 0u64, 0u64); budgets.len()]; policies.len()];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let mut recur0 = RecurrenceModel::new(l, e);
        train_recur(&mut recur0, &train);
        let mut pmi0 = MatrixPredictor::new(l, e, Scoring::Pmi { alpha: 0.5 });
        train_predictor(&mut pmi0, &train, 2);

        let test = trace.for_config(c);
        let distinct = {
            let mut s = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            s.insert(ly * e + x);
                        }
                    }
                }
            }
            s.len()
        };

        for (bi, &frac) in budgets.iter().enumerate() {
            let budget = ((distinct as f64) * frac).ceil() as usize;
            for (pi, &(_, pol)) in policies.iter().enumerate() {
                let mut r = recur0.clone();
                let mut p = pmi0.clone();
                let (mi, ac, ev, rq) = demote_sim(
                    &test, budget, pol, &mut r, prior, EVICT_FRAC, REGRET_W, &mut p, cohort,
                );
                let s = &mut acc[pi][bi];
                s.0 += mi;
                s.1 += ac;
                s.2 += ev;
                s.3 += rq;
            }
        }
    }

    println!(
        "\n=== Demote: 21-fold LOOCV (batch-evict {:.0}%/token, regret window W={}) ===",
        EVICT_FRAC * 100.0,
        REGRET_W
    );
    println!(
        "prior(stickiness)={:.1}%   lower miss-rate and group-regret are better",
        prior * 100.0
    );
    for (bi, &frac) in budgets.iter().enumerate() {
        println!("\n── VRAM budget = {:.0}% of working set ──", frac * 100.0);
        println!(
            "{:<12} {:>12} {:>14}",
            "policy", "miss-rate", "group-regret"
        );
        println!("{}", "-".repeat(40));
        for (pi, &(label, _)) in policies.iter().enumerate() {
            let (mi, ac, ev, rq) = acc[pi][bi];
            println!(
                "{:<12} {:>11.1}% {:>13.1}%",
                label,
                ratio(mi as usize, ac as usize) * 100.0,
                ratio(rq as usize, ev as usize) * 100.0
            );
        }
    }
    println!();
}

// ── The wave: batched decode streaming through layers over PCIe ──────────────
//
// A batch of decode sequences steps coherently through all 48 layers; at each
// layer the wave needs the UNION of experts its tokens route to.  It sweeps
// L=0→47, wraps to 0 for the next token.  Only ~60% of experts fit in VRAM, so
// the wave runs ahead of itself prefetching, and evicts from behind (L-1,
// wrapping — the longest reuse distance).  Stalls come from (a) hard misses
// (needed expert not resident → emergency load) and (b) PCIe saturation
// (prefetch can't keep up).  We model only stalls (compute is flat-out); the
// output is an estimated token rate.

const WAVE_LAYERS: usize = 48;
const WAVE_E: usize = 128;
/// Q4_K_M expert ≈ 3·768·2048 params · 4.5 bit.
const EXPERT_MB: f64 = 2.6;
/// PCIe link bandwidth (given).
const PCIE_GBPS: f64 = 8.0;
/// Expert transfer time (ms): 2.6 MB ÷ 8 GB/s.
const TX_MS: f64 = EXPERT_MB / (PCIE_GBPS * 1000.0) * 1000.0;
/// Per-layer compute window (ms) at the 30 t/s target: (1000/30)/48.
const TC_MS: f64 = (1000.0 / 30.0) / WAVE_LAYERS as f64;

#[derive(Clone, Copy, Default)]
struct WaveResult {
    rate: f64,           // estimated tokens/sec
    hit: f64,            // % resident & ready, never prefetched (always-warm)
    soft: f64,           // % prefetched & ready in time (latency hidden)
    late: f64,           // % prefetched but not finished → partial stall
    hard: f64,           // % not resident → emergency load stall
    pcie_util: f64,      // fraction of wall-time the PCIe link was busy
    stream_per_tok: f64, // experts transferred per token
}

/// Build the wave's per-(token, layer) demand for a batch of `b` sessions.
/// Each batch slot cycles one captured config at a distinct token offset, so the
/// per-layer union grows toward full demand (≈128/layer) as B increases —
/// modelling how a large, diverse production batch activates most experts.
fn build_wave_demand(trace: &Trace, b: usize, max_tokens: usize) -> Vec<Vec<Vec<usize>>> {
    let cfgs = trace.configs();
    // Per config: [token][layer] = experts (decode passes only).
    let mut seqs: Vec<Vec<[Vec<usize>; WAVE_LAYERS]>> = Vec::new();
    for &c in &cfgs {
        let sub = trace.for_config(c);
        let mut seq: Vec<[Vec<usize>; WAVE_LAYERS]> = Vec::new();
        for pass in &sub.passes {
            if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                continue;
            }
            let mut tok: [Vec<usize>; WAVE_LAYERS] = std::array::from_fn(|_| Vec::new());
            for (l, ex) in &pass.layers {
                if *l < WAVE_LAYERS {
                    tok[*l] = ex.clone();
                }
            }
            seq.push(tok);
        }
        if !seq.is_empty() {
            seqs.push(seq);
        }
    }
    let nc = seqs.len().max(1);
    // Each slot: (config index, token offset).  Distinct offsets per repeat give
    // diverse routings so the union grows with B beyond the config count.
    let slots: Vec<(usize, usize)> = (0..b.max(1)).map(|i| (i % nc, (i / nc) * 13)).collect();
    let ntok = max_tokens;
    let mut demand: Vec<Vec<Vec<usize>>> = Vec::with_capacity(ntok);
    for t in 0..ntok {
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); WAVE_LAYERS];
        for l in 0..WAVE_LAYERS {
            let mut set: HashSet<usize> = HashSet::new();
            for &(ci, off) in &slots {
                let seq = &seqs[ci];
                let tok = &seq[(t + off) % seq.len()];
                for &e in &tok[l] {
                    set.insert(e);
                }
            }
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable();
            layers[l] = v;
        }
        demand.push(layers);
    }
    demand
}

/// Discrete-event simulation of the wave.  Prefetch uses the previous token's
/// per-layer demand (recurrence) `prefetch_depth` layers ahead, capped at
/// `prefetch_cap` experts per (layer, lookahead).  Eviction drains from behind
/// the wave (L-1 first, wrapping).  Returns timing + outcome rates.
fn wave_sim(
    demand: &[Vec<Vec<usize>>],
    budget: usize,
    prefetch_depth: usize,
    prefetch_cap: usize,
    k_champ: usize,
) -> WaveResult {
    // Optional online champion predictor for arrival prefetch of the next layer
    // (recurrence covers the sticky experts; the champion covers cold arrivals).
    let mut champ = MatrixPredictor::new(WAVE_LAYERS, WAVE_E, Scoring::Pmi { alpha: 0.5 });
    // resident[layer] = { expert : (ready_at_ms, prefetched_flag) }
    let mut resident: Vec<HashMap<usize, (f64, bool)>> = vec![HashMap::new(); WAVE_LAYERS];
    let mut count = 0usize;
    let mut clock = 0.0f64;
    let mut pcie_free = 0.0f64;
    let mut pcie_busy = 0.0f64;
    let (mut hit, mut soft, mut late, mut hard) = (0u64, 0u64, 0u64, 0u64);
    let mut transfers = 0u64;

    // Evict from behind the wave: layer (cur-1, cur-2, … wrapping).
    let evict = |resident: &mut Vec<HashMap<usize, (f64, bool)>>, count: &mut usize, cur: usize| {
        for back in 1..WAVE_LAYERS {
            let el = (cur + WAVE_LAYERS - back) % WAVE_LAYERS;
            if let Some(&k) = resident[el].keys().next() {
                resident[el].remove(&k);
                *count -= 1;
                return;
            }
        }
    };

    // Exclude warmup (the one-time cost of filling the cache) — measure
    // steady-state.  Counters and the timed window reset at `warmup`.
    let warmup = (demand.len() / 4).clamp(1, 20);
    let mut clock_warm = 0.0f64;
    let mut busy_warm = 0.0f64;

    let ntok = demand.len();
    for t in 0..ntok {
        if t == warmup {
            clock_warm = clock;
            busy_warm = pcie_busy;
            hit = 0;
            soft = 0;
            late = 0;
            hard = 0;
            transfers = 0;
        }
        if k_champ > 0 {
            champ.reset_pass();
        }
        for l in 0..WAVE_LAYERS {
            // 1. Access the experts this layer needs.
            for &e in &demand[t][l] {
                match resident[l].get(&e).copied() {
                    Some((ready, pf)) if ready <= clock => {
                        if pf {
                            soft += 1;
                            resident[l].insert(e, (ready, false)); // consumed
                        } else {
                            hit += 1;
                        }
                    }
                    Some((ready, _)) => {
                        // Prefetched but not finished: wave waits (partial stall).
                        clock = ready;
                        late += 1;
                        resident[l].insert(e, (ready, false));
                    }
                    None => {
                        // Hard miss: emergency load on the critical path.
                        if count >= budget {
                            evict(&mut resident, &mut count, l);
                        }
                        let start = clock.max(pcie_free);
                        clock = start + TX_MS;
                        pcie_free = clock;
                        pcie_busy += TX_MS;
                        transfers += 1;
                        resident[l].insert(e, (clock, false));
                        count += 1;
                        hard += 1;
                    }
                }
            }
            // 2. Compute this layer (flat-out) — the overlap window.
            clock += TC_MS;
            // 3a. Champion arrival prefetch of the immediate next layer (uses
            //     spare PCIe to fetch cold arrivals recurrence can't predict).
            if k_champ > 0 {
                let tl = (l + 1) % WAVE_LAYERS;
                for pe in champ.score_topk(l, &demand[t][l], k_champ, Scoring::Pmi { alpha: 0.5 }) {
                    if !resident[tl].contains_key(&pe) {
                        // Run ahead: evict from behind (L-1, wrapping) to make room.
                        if count >= budget {
                            evict(&mut resident, &mut count, l);
                        }
                        let start = clock.max(pcie_free);
                        pcie_free = start + TX_MS;
                        pcie_busy += TX_MS;
                        transfers += 1;
                        resident[tl].insert(pe, (pcie_free, true));
                        count += 1;
                    }
                }
                champ.observe(l, &demand[t][l]);
            }
            // 3b. Recurrence prefetch ahead using the previous token's demand.
            if t > 0 {
                for d in 1..=prefetch_depth {
                    let tl = (l + d) % WAVE_LAYERS;
                    let pred = &demand[t - 1][tl];
                    let mut c = 0usize;
                    for &pe in pred {
                        if c >= prefetch_cap {
                            break;
                        }
                        if !resident[tl].contains_key(&pe) {
                            // Run ahead: evict from behind (L-1, wrapping).
                            if count >= budget {
                                evict(&mut resident, &mut count, l);
                            }
                            let start = clock.max(pcie_free);
                            pcie_free = start + TX_MS;
                            pcie_busy += TX_MS;
                            transfers += 1;
                            resident[tl].insert(pe, (pcie_free, true));
                            count += 1;
                            c += 1;
                        }
                    }
                }
            }
        }
    }

    let steady_tok = (ntok.saturating_sub(warmup)).max(1);
    let elapsed = (clock - clock_warm).max(1e-9);
    let token_time = elapsed / steady_tok as f64;
    let acc = (hit + soft + late + hard).max(1) as f64;
    WaveResult {
        rate: 1000.0 / token_time,
        hit: 100.0 * hit as f64 / acc,
        soft: 100.0 * soft as f64 / acc,
        late: 100.0 * late as f64 / acc,
        hard: 100.0 * hard as f64 / acc,
        pcie_util: (pcie_busy - busy_warm).max(0.0) / elapsed,
        stream_per_tok: transfers as f64 / steady_tok as f64,
    }
}

/// §12 — wave simulation: estimate token rate vs prefetch depth × prefetch
/// amount at 60% VRAM, 8 GB/s PCIe, 30 t/s compute budget.
#[test]
fn wave_optimize() {
    let Some(records) = load_fixture() else {
        println!("\n[wave_optimize] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let demand = build_wave_demand(&trace, trace.configs().len(), 80);

    // Diagnostics: per-layer demand and working set vs cache.
    let (mut dsum, mut dn) = (0usize, 0usize);
    let mut ws: HashSet<usize> = HashSet::new();
    for tok in &demand {
        for (l, ex) in tok.iter().enumerate() {
            dsum += ex.len();
            dn += 1;
            for &e in ex {
                ws.insert(l * WAVE_E + e);
            }
        }
    }
    let budget = (0.6 * (WAVE_LAYERS * WAVE_E) as f64).round() as usize;
    println!("\n=== §12 Wave simulation ===");
    println!(
        "tokens={} avg|demand/layer|={:.1} working-set={} VRAM budget(60%)={} slots",
        demand.len(),
        dsum as f64 / dn.max(1) as f64,
        ws.len(),
        budget
    );
    println!(
        "expert={:.1}MB  PCIe={:.0}GB/s  transfer={:.3}ms/expert  layer-compute={:.3}ms  budget/token≈{:.0} experts",
        EXPERT_MB,
        PCIE_GBPS,
        TX_MS,
        TC_MS,
        (TC_MS * WAVE_LAYERS as f64) / TX_MS
    );

    let _ = (demand, budget);
    let depths = [1usize, 2, 4, 8];
    let caps = [16usize, 64];
    let champs = [0usize, 8];
    let total = (WAVE_LAYERS * WAVE_E) as f64;
    // Optimise (depth × cap × champion-k) for a (demand, budget); return best.
    let optimise = |dem: &[Vec<Vec<usize>>], bud: usize| {
        let mut best = (0.0f64, 0usize, 0usize, 0usize, WaveResult::default());
        for &d in &depths {
            for &cap in &caps {
                for &kc in &champs {
                    let r = wave_sim(dem, bud, d, cap, kc);
                    if r.rate > best.0 {
                        best = (r.rate, d, cap, kc, r);
                    }
                }
            }
        }
        best
    };

    // ── Aggregate throughput grows with B (amortised expert loads) ──
    let bud60 = (0.60 * total).round() as usize;
    println!("\n-- per-session vs aggregate token rate at 60% VRAM (optimised) --");
    println!(
        "{:>5} {:>8} {:>8} {:>9} {:>11} {:>6} {:>6} {:>6} {:>6}",
        "B", "dem/lyr", "stream", "t/s/sess", "aggregate", "hit%", "soft%", "late%", "hard%"
    );
    for &bsz in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let dem = build_wave_demand(&trace, bsz, 60);
        let dpl = dem.iter().flatten().map(|e| e.len()).sum::<usize>() as f64
            / (dem.len() * WAVE_LAYERS) as f64;
        let (rate, _, _, _, r) = optimise(&dem, bud60);
        // hit (resident) / soft (prefetched, hidden) / late (prefetched, partial
        // stall) / hard (demand stall) — the full four-way access classification.
        println!(
            "{:>5} {:>8.1} {:>8.0} {:>9.1} {:>11.0} {:>6.0} {:>6.0} {:>6.0} {:>6.0}",
            bsz,
            dpl,
            r.stream_per_tok,
            rate,
            rate * bsz as f64,
            r.hit,
            r.soft,
            r.late,
            r.hard
        );
    }

    // ── 2D optimisation: best wave params vs (batch, VRAM residency) ──
    println!("\n-- 2D optimum params : VRAM residency × batch size --");
    println!(
        "{:>5} {:>5} {:>8} {:>9} {:>11} {:>6} {:>6} {:>12} {:>6}",
        "res%", "B", "stream", "t/s/sess", "aggregate", "hit%", "hard%", "depth/cap/kc", "pcie"
    );
    for &res in &[0.40f64, 0.50, 0.60, 0.70, 0.80, 1.00] {
        let bud = (res * total).round() as usize;
        for &bsz in &[1usize, 4, 16, 64, 256] {
            let dem = build_wave_demand(&trace, bsz, 60);
            let (rate, d, cap, kc, r) = optimise(&dem, bud);
            println!(
                "{:>5.0} {:>5} {:>8.0} {:>9.1} {:>11.0} {:>6.0} {:>6.0} {:>4}/{:<3}/{:<2} {:>5.0}%",
                res * 100.0,
                bsz,
                r.stream_per_tok,
                rate,
                rate * bsz as f64,
                r.hit,
                r.hard,
                d,
                cap,
                kc,
                r.pcie_util * 100.0
            );
        }
    }
    println!(
        "\nCompute floor = 30 t/s/session (weight-read amortised). 100% residency = control\n\
         (working set fits → no eviction → 30 t/s). depth=prefetch lookahead, cap=experts/\n\
         layer, kc=champion arrival-prefetch; eviction is always L-1 (behind, wrapping).\n"
    );

    // ── Saturation regime: drop the predictor, prefetch the full missing set ──
    // Compare the fully tuned optimum against the simple deterministic policy
    // (L+1 ahead, prefetch all missing — cap=E, no champion).  At/above the
    // streaming-saturation batch they should coincide (bandwidth-bound).
    println!(
        "-- saturation regime: tuned optimum vs prefetch-all-missing (depth=1, cap=E, kc=0) --"
    );
    println!(
        "{:>5} {:>5} {:>10} {:>14} {:>8}",
        "res%", "B", "tuned t/s", "prefetch-all", "stream"
    );
    for &res in &[0.60f64, 0.80] {
        let bud = (res * total).round() as usize;
        for &bsz in &[64usize, 256] {
            let dem = build_wave_demand(&trace, bsz, 60);
            let (tuned, ..) = optimise(&dem, bud);
            let r = wave_sim(&dem, bud, 1, WAVE_E, 0); // prefetch-all-missing
            println!(
                "{:>5.0} {:>5} {:>10.2} {:>14.2} {:>8.0}",
                res * 100.0,
                bsz,
                tuned,
                r.rate,
                r.stream_per_tok
            );
        }
    }
    println!(
        "\nAt saturation the two coincide (bandwidth-bound): the predictor adds nothing, so the\n\
         engine flips to deterministic prefetch-all-missing-L+1 + evict-L-1 — ride the wave.\n"
    );
}

/// Load and decompress the captured fixture, if present.
fn load_fixture() -> Option<Vec<RoutingRecord>> {
    let bytes = std::fs::read(FIXTURE_PATH).ok()?;
    let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut raw = Vec::new();
    decoder.read_to_end(&mut raw).ok()?;
    bincode::deserialize(&raw).ok()
}

#[test]
fn report() {
    let Some(records) = load_fixture() else {
        println!(
            "\n[expert predictor eval] no fixture at {FIXTURE_PATH}\n\
             capture one with:\n  cargo test --release --features cuda --lib \
             -p candle-transformers quantized_qwen3_moe::tests::capture_routing_trace \
             -- --ignored --nocapture\n"
        );
        return;
    };

    let trace = Trace::from_records(&records);
    println!("\n=== MoE Expert Predictor — Offline Replay ===");
    println!(
        "records={}  passes={}  layers={}  experts={}  avg|set|={:.1}",
        records.len(),
        trace.passes.len(),
        trace.num_layers,
        trace.num_experts,
        trace.avg_set_size(),
    );

    const DECODE_MAX: usize = 16;
    let variants: [(&str, Scoring); 4] = [
        ("raw", Scoring::Raw),
        ("conditional", Scoring::Conditional),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
        ("pmi(1.0)", Scoring::Pmi { alpha: 1.0 }),
    ];

    // ── Scoring comparison, at cold-start and converged ──
    // `decode-cov` (cold-expert coverage on decode passes) maps to real
    // prefetch value: of the experts the next layer needs that are not already
    // active, what fraction did we name.
    let row = |label: &str, k: usize, m: &Metrics, d: &Metrics| {
        println!(
            "{:<14} {:>3} {:>7.1}% {:>9.1}% {:>9.1}% {:>11.1}%",
            label,
            k,
            m.top1() * 100.0,
            m.precision() * 100.0,
            m.coverage() * 100.0,
            d.coverage() * 100.0,
        );
    };

    for (title, warmup) in [
        ("cold-start (online from zero)", 0usize),
        ("converged (~25 epochs)", 25),
    ] {
        println!("\n── {title} ──");
        println!(
            "{:<14} {:>3} {:>8} {:>10} {:>10} {:>12}",
            "predictor", "k", "top1", "precision", "coverage", "decode-cov"
        );
        println!("{}", "-".repeat(62));
        for k in [1usize, 2, 4] {
            let mut pop = Popularity::new(trace.num_layers, trace.num_experts);
            let (a, d) = replay(&trace, &mut pop, k, DECODE_MAX, warmup);
            row("popularity", k, &a, &d);
            for &(label, scoring) in &variants {
                let mut mp = MatrixPredictor::new(trace.num_layers, trace.num_experts, scoring);
                let (a, d) = replay(&trace, &mut mp, k, DECODE_MAX, warmup);
                row(label, k, &a, &d);
            }
            println!();
        }
    }

    // ── Learning curve: decode coverage vs amount of training ──
    // Shows the "improves the more it runs" behavior directly.
    let epochs = [0usize, 1, 2, 4, 8, 16, 32, 64];
    println!("── Learning curve: decode-cov vs training epochs (k=4) ──");
    print!("{:<14}", "epochs→");
    for e in epochs {
        print!("{e:>7}");
    }
    println!();
    for &(label, scoring) in &variants {
        print!("{label:<14}");
        for &w in &epochs {
            let mut mp = MatrixPredictor::new(trace.num_layers, trace.num_experts, scoring);
            let (_, d) = replay(&trace, &mut mp, 4, DECODE_MAX, w);
            print!("{:>6.1}%", d.coverage() * 100.0);
        }
        println!();
    }
    print!("{:<14}", "popularity");
    for &w in &epochs {
        let mut pop = Popularity::new(trace.num_layers, trace.num_experts);
        let (_, d) = replay(&trace, &mut pop, 4, DECODE_MAX, w);
        print!("{:>6.1}%", d.coverage() * 100.0);
    }
    println!();

    println!(
        "\ntop1 = top prediction routed next layer; precision = predicted∩next / predicted;\n\
         coverage = predicted∩(next\\current) / (next\\current); decode-cov = coverage on decode passes.\n\
         NOTE: one 64-token capture. Epochs simulate a longer/persisted session; the real\n\
         converged number needs a longer capture (more decode tokens / diverse prompts).\n"
    );
}

// ── Deterministic tests of the replay/metric logic (no fixture needed) ──

#[cfg(test)]
mod logic_tests {
    use super::*;

    fn rec(config: u16, pass: u32, layer: u16, experts: &[u32]) -> RoutingRecord {
        RoutingRecord {
            config,
            pass,
            layer,
            experts: experts.to_vec(),
            mass: vec![1.0; experts.len()],
        }
    }

    #[test]
    fn groups_passes_by_config_and_pass() {
        let recs = vec![
            rec(0, 0, 0, &[1]),
            rec(0, 0, 1, &[2]),
            rec(0, 1, 0, &[3]),
            rec(1, 0, 0, &[4]),
        ];
        let t = Trace::from_records(&recs);
        assert_eq!(t.passes.len(), 3);
        assert_eq!(t.passes[0].layers.len(), 2);
        assert_eq!(t.passes[1].layers.len(), 1);
        assert_eq!(t.num_layers, 2);
        assert_eq!(t.num_experts, 5);
    }

    #[test]
    fn metrics_count_precision_and_coverage() {
        let mut m = Metrics::default();
        // current = {0}, next = {1,2}; predicted = {1,3}.
        // pred_total=2, pred_hits=1 (1∈next), miss={1,2}, miss_covered=1 (1).
        m.add(&[1, 3], &[0], &[1, 2]);
        assert_eq!(m.pred_total, 2);
        assert_eq!(m.pred_hits, 1);
        assert_eq!(m.miss_total, 2);
        assert_eq!(m.miss_covered, 1);
        assert_eq!(m.top1_hits, 1); // predicted[0]=1 ∈ next
        assert!((m.precision() - 0.5).abs() < 1e-9);
        assert!((m.coverage() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn perfectly_periodic_trace_is_learned() {
        // Identical routing every pass: 0:{0} -> 1:{5} -> 2:{9}.  Once the
        // matrix is warm (64 observations = 32 passes), every transition is
        // predicted exactly.  Use enough passes that the warmup is negligible.
        let mut many = Vec::new();
        for p in 0..700u32 {
            many.push(rec(0, p, 0, &[0]));
            many.push(rec(0, p, 1, &[5]));
            many.push(rec(0, p, 2, &[9]));
        }
        let trace = Trace::from_records(&many);
        let mut mat = TransitionMatrix::new(trace.num_layers, trace.num_experts);
        let (all, _) = replay(&trace, &mut mat, 1, 16, 0);
        // Cold start is ~32/700 of transitions; the rest are perfect.
        assert!(all.coverage() > 0.9, "coverage was {}", all.coverage());
        assert!(all.top1() > 0.9, "top1 was {}", all.top1());
    }
}
