//! Replay of a routing trace through a predictor, scoring each prediction
//! against the next layer's experts.
//!
//! `Metrics` accumulates precision, coverage and top-1 counts over validated
//! adjacent transitions. Coverage counts only the experts a prefetch could
//! usefully warm: those routed at the next layer but not active at the current
//! one. `replay` predicts each layer before observing it, optionally after
//! warmup passes that approximate a converged matrix, and reports both all passes
//! and decode-like passes. `per_layer_token_sequences` keeps adjacency from
//! crossing a config boundary and leaves out prefill passes.

use super::*;

/// Accumulated prediction-quality counters over validated transitions.
#[derive(Default, Clone)]
pub(super) struct Metrics {
    /// Validated `(L → L+1)` transitions where a prediction was emitted.
    pub(super) predicted_nonempty: usize,
    /// Transitions whose top-1 prediction was actually routed at `L+1`.
    pub(super) top1_hits: usize,
    /// Sum of predicted-set sizes (precision denominator).
    pub(super) pred_total: usize,
    /// Predicted experts actually routed at `L+1` (precision numerator).
    pub(super) pred_hits: usize,
    /// Sum of next-layer "miss" experts: routed at `L+1`, not active at `L`
    /// (these are the experts a prefetch could usefully warm).
    pub(super) miss_total: usize,
    /// Predicted experts that landed in the miss set (coverage numerator).
    pub(super) miss_covered: usize,
}

impl Metrics {
    pub(super) fn precision(&self) -> f64 {
        ratio(self.pred_hits, self.pred_total)
    }
    pub(super) fn coverage(&self) -> f64 {
        ratio(self.miss_covered, self.miss_total)
    }
    pub(super) fn top1(&self) -> f64 {
        ratio(self.top1_hits, self.predicted_nonempty)
    }

    /// Fold one transition: `predicted` against the actual `next` set, given
    /// the `current` active set (to compute the miss set).
    pub(super) fn add(&mut self, predicted: &[usize], current: &[usize], next: &[usize]) {
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

pub(super) fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Replay the trace through a predictor at fan-out `k`, returning metrics over
/// all transitions and over decode-like passes only (active set ≤ `decode_max`).
pub(super) fn replay<P: Predictor>(
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
pub(super) fn per_layer_token_sequences(trace: &Trace, decode_max: usize) -> Vec<Vec<Vec<usize>>> {
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
