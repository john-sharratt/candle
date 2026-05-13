//! Online-learned transition matrix for speculative expert prefetching.
//!
//! For each pair of adjacent MoE layers `(L, L+1)`, stores a `[E × E]`
//! matrix of co-occurrence counts.  Counts are updated after every forward
//! pass by observing which experts were routed at consecutive MoE layers.
//!
//! ## Online learning
//!
//! No separate calibration pass is needed.  The matrix is built from actual
//! routing decisions during inference.  After a few dozen tokens, the
//! transition probabilities converge enough for useful predictions.
//!
//! ## Prefetch predictions
//!
//! Given the active expert set at layer L, the matrix predicts which experts
//! layer L+1 will likely need.  Their DMA can begin speculatively while
//! layer L's compute runs — converting cold misses into warm hits.
//!
//! ## Safety
//!
//! Mispredictions are harmless: speculatively loaded experts sit in cache
//! and get evicted naturally by score-based eviction if unused.  The worst case is wasted
//! DMA bandwidth on a prediction that didn't pan out.

pub(crate) struct TransitionMatrix {
    /// Transition counts: `counts[pair][from][to]` where pair = moe_layer_idx.
    /// `pair` ranges 0..num_moe_layers-1.  Each matrix is [E × E] flattened
    /// row-major.
    counts: Vec<Vec<f32>>,
    /// Number of experts per MoE layer.
    experts_per_layer: usize,
    /// Total number of MoE layers.
    num_moe_layers: usize,
    /// The expert IDs routed at the *previous* MoE layer in this forward pass.
    /// Used by `observe()` to build transitions from L→L+1.
    prev_layer_experts: Option<(usize, Vec<usize>)>,
    /// Minimum total observations before predictions are emitted.
    /// Prevents noisy predictions in the first few tokens.
    min_observations: u32,
    /// Total observations recorded (sum of all transitions).
    total_observations: u32,
}

impl TransitionMatrix {
    /// Create a new transition matrix.
    ///
    /// * `num_moe_layers` — total number of MoE layers (e.g. 48)
    /// * `experts_per_layer` — number of experts per layer (e.g. 128)
    pub(crate) fn new(num_moe_layers: usize, experts_per_layer: usize) -> Self {
        let num_pairs = if num_moe_layers > 1 {
            num_moe_layers - 1
        } else {
            0
        };
        let matrix_size = experts_per_layer * experts_per_layer;
        let counts = vec![vec![0.0f32; matrix_size]; num_pairs];
        Self {
            counts,
            experts_per_layer,
            num_moe_layers,
            prev_layer_experts: None,
            min_observations: 64,
            total_observations: 0,
        }
    }

    /// Record routing decisions at a given MoE layer.
    ///
    /// Call this for every MoE layer in forward-pass order.  When consecutive
    /// layers are observed, the transition counts are updated.
    pub(crate) fn observe(&mut self, moe_layer_idx: usize, expert_ids: &[usize]) {
        if self.counts.is_empty() {
            return;
        }

        // If we have a previous layer observation, update transitions.
        if let Some((prev_idx, ref prev_experts)) = self.prev_layer_experts {
            // Only update if this is the next consecutive MoE layer.
            if moe_layer_idx == prev_idx + 1 && prev_idx < self.counts.len() {
                let e = self.experts_per_layer;
                let pair = prev_idx;
                for &from in prev_experts {
                    if from >= e {
                        continue;
                    }
                    let row_base = from * e;
                    for &to in expert_ids {
                        if to >= e {
                            continue;
                        }
                        self.counts[pair][row_base + to] += 1.0;
                        self.total_observations += 1;
                    }
                }
            }
        }

        // Store current layer as "previous" for the next observe() call.
        self.prev_layer_experts = Some((moe_layer_idx, expert_ids.to_vec()));
    }

    /// Reset the previous-layer state between forward passes.
    ///
    /// Call this at the start of each new forward pass so that layer 0
    /// of the new pass doesn't form transitions with the last layer of
    /// the previous pass.
    pub(crate) fn reset_pass(&mut self) {
        self.prev_layer_experts = None;
    }

    /// Predict which experts layer `moe_layer_idx + 1` will likely need,
    /// given the active experts at `moe_layer_idx`.
    ///
    /// Returns a deduplicated, sorted list of predicted expert IDs.
    /// Returns empty if:
    /// - Not enough observations yet
    /// - `moe_layer_idx` is the last MoE layer (no successor)
    /// - The transition matrix has no data for this pair
    pub(crate) fn predict(&self, moe_layer_idx: usize, expert_ids: &[usize]) -> Vec<usize> {
        // Not enough data yet — return empty.
        if self.total_observations < self.min_observations {
            return vec![];
        }

        // No successor layer.
        if moe_layer_idx + 1 >= self.num_moe_layers || moe_layer_idx >= self.counts.len() {
            return vec![];
        }

        let e = self.experts_per_layer;
        let pair = moe_layer_idx;
        let matrix = &self.counts[pair];

        // Accumulate scores across all active experts at this layer.
        let mut scores = vec![0.0f32; e];
        for &from in expert_ids {
            if from >= e {
                continue;
            }
            let row_base = from * e;
            for to in 0..e {
                scores[to] += matrix[row_base + to];
            }
        }

        // Pick the single best candidate by score, excluding experts that
        // are already in the current active set (those will likely be cache
        // hits at the next layer anyway — no point prefetching them).
        let mut best_idx: Option<usize> = None;
        let mut best_score: f32 = 0.0;
        for (idx, &s) in scores.iter().enumerate() {
            if s > best_score && !expert_ids.contains(&idx) {
                best_score = s;
                best_idx = Some(idx);
            }
        }

        match best_idx {
            Some(idx) => vec![idx],
            None => vec![],
        }
    }

    /// Predict the bottom-N least-likely experts for layer `moe_layer_idx + 1`,
    /// given the active experts at `moe_layer_idx`.
    ///
    /// These are "anti-predictions" — experts unlikely to be needed next.
    /// Used to penalize their cache scores so they become eviction candidates.
    ///
    /// Returns a list of up to `count` expert IDs with the lowest transition
    /// scores (excluding the current active set, which we already handle).
    /// Returns empty if not enough observations yet.
    pub(crate) fn predict_bottom(
        &self,
        moe_layer_idx: usize,
        expert_ids: &[usize],
        count: usize,
    ) -> Vec<usize> {
        // Not enough data yet — return empty.
        if self.total_observations < self.min_observations {
            return vec![];
        }

        // No successor layer.
        if moe_layer_idx + 1 >= self.num_moe_layers || moe_layer_idx >= self.counts.len() {
            return vec![];
        }

        let e = self.experts_per_layer;
        let pair = moe_layer_idx;
        let matrix = &self.counts[pair];

        // Accumulate scores across all active experts at this layer.
        let mut scores = vec![0.0f32; e];
        for &from in expert_ids {
            if from >= e {
                continue;
            }
            let row_base = from * e;
            for to in 0..e {
                scores[to] += matrix[row_base + to];
            }
        }

        // Collect (expert_idx, score) — exclude current active set.
        let mut candidates: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| !expert_ids.contains(idx))
            .map(|(idx, &s)| (idx, s))
            .collect();

        // Sort ascending by score — lowest first.
        candidates
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return bottom N.
        candidates.iter().take(count).map(|&(idx, _)| idx).collect()
    }
}
