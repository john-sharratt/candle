//! Online tool-belief accumulator (RelLeak).
//!
//! A live decode fires reprojections on cadence through the user prompt,
//! thinking block, and response.  Each projection scans the gallery and yields
//! a per-tool confidence.  [`ToolBelief`] fuses that stream into a single
//! running belief over which tool is in scope, so the selection both pools weak
//! evidence into a confident pick and forgets a stale topic when the
//! conversation moves on.
//!
//! # Update rule — relative leak
//!
//! Per projection step, with fresh per-tool scores `s` and running belief `acc`:
//!
//! ```text
//!   m       = max(acc)
//!   acc[t]  = max(0, acc[t] − β·m) + s[t]     for all tools t
//! ```
//!
//! The leak is proportional to the *current leader*, so it self-scales — no
//! magnitude tuning across the wide score range.  Within a topic the leader is
//! the correct tool: it re-earns its `β·m` leak from fresh support while
//! followers, which lose `β·m` (more than their own small mass), are pinned near
//! zero — sharpening the selection.  On a topic switch the stale incumbent
//! receives the leak with no fresh support and bleeds out over the new topic's
//! few projections while the new tool accumulates.
//!
//! This is the frontier of a broad mechanism sweep (multiplicative EWMA,
//! leaky-max, position-ramp, simplex, two-timescale, surprise-gated reset all
//! sit below it) measured on both single-intent and synthetic topic-switch
//! trials.  See `docs/tool_selection_provenance_results.md` §24.

/// Default leak fraction.  Balances single-intent retention against topic-switch
/// recovery: single-intent Tool-1 92.5 / topic-switch Tool-1 90.3 on the 93-tool
/// corpus (§24.4).  Chosen over the 0.50 min-optimum because the synthetic switch
/// (topic A at full strength immediately before B) is harsher than a real decode
/// trajectory, so preserving single-intent accuracy is the safer default.  Raise
/// toward 0.50 if abrupt topic drift proves common.
pub const DEFAULT_LEAK_BETA: f32 = 0.40;

/// Relative-leak online belief over a fixed set of tool slots.  Slot indices are
/// a caller-owned stable tool ordering; the belief is agnostic to what a slot
/// means.  The leak fraction `β` is per-slot, so different sections can decay at
/// different rates while sharing the one global leader-relative floor.
pub struct ToolBelief {
    acc: Vec<f32>,
    beta: Vec<f32>,
}

impl ToolBelief {
    /// A belief over `n_tools` slots using [`DEFAULT_LEAK_BETA`] for every slot.
    pub fn new(n_tools: usize) -> Self {
        Self::with_beta(n_tools, DEFAULT_LEAK_BETA)
    }

    /// A belief over `n_tools` slots with one leak fraction broadcast to all.
    pub fn with_beta(n_tools: usize, beta: f32) -> Self {
        Self {
            acc: vec![0.0; n_tools],
            beta: vec![beta; n_tools],
        }
    }

    /// A belief with an explicit per-slot leak fraction (one `β` per section).
    pub fn with_betas(beta: Vec<f32>) -> Self {
        Self {
            acc: vec![0.0; beta.len()],
            beta,
        }
    }

    /// Number of tool slots.
    pub fn len(&self) -> usize {
        self.acc.len()
    }

    /// Whether the belief has no slots.
    pub fn is_empty(&self) -> bool {
        self.acc.is_empty()
    }

    /// The raw per-slot confidence accumulator.
    pub fn scores(&self) -> &[f32] {
        &self.acc
    }

    /// Reset all confidence to zero, keeping the slot count and leak fraction.
    pub fn clear(&mut self) {
        self.acc.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Overwrite the accumulator with prior per-slot confidence — used to reseed
    /// the belief from the last projection event so decay/reinforcement carries
    /// across a turn's reprojections. `scores.len()` must equal the slot count.
    pub fn set_scores(&mut self, scores: &[f32]) {
        debug_assert_eq!(scores.len(), self.acc.len(), "seed/slot count mismatch");
        self.acc.copy_from_slice(scores);
    }

    /// Apply one projection's per-tool scores.  `scores.len()` must equal the
    /// slot count.
    pub fn update(&mut self, scores: &[f32]) {
        debug_assert_eq!(scores.len(), self.acc.len(), "score/slot count mismatch");
        let m = self.acc.iter().copied().fold(0.0f32, f32::max);
        for ((acc, &beta), &score) in self.acc.iter_mut().zip(self.beta.iter()).zip(scores) {
            *acc = (*acc - beta * m).max(0.0) + score;
        }
    }

    /// The current best tool slot and its confidence, or `None` if no slot
    /// carries positive mass.
    pub fn top(&self) -> Option<(usize, f32)> {
        self.acc
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
    }

    /// The top `k` tool slots by confidence, descending.  Only slots with
    /// positive mass are returned, so the result may be shorter than `k`.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut idx: Vec<usize> = (0..self.acc.len()).filter(|&i| self.acc[i] > 0.0).collect();
        idx.sort_by(|&a, &b| {
            self.acc[b]
                .partial_cmp(&self.acc[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx.into_iter().map(|i| (i, self.acc[i])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_update_from_zero_is_identity() {
        // No prior mass → leak is zero → belief equals the fresh scores exactly.
        let mut b = ToolBelief::with_beta(3, 0.5);
        b.update(&[0.0, 10.0, 0.0]);
        assert_eq!(b.scores(), &[0.0, 10.0, 0.0]);
        assert_eq!(b.top(), Some((1, 10.0)));
    }

    #[test]
    fn relative_leak_arithmetic_is_exact() {
        // beta=0.5, leader=10 → leak=5. Followers below the leak floor to zero;
        // the leader keeps (10-5)+fresh; a fresh slot adds on top of a zeroed base.
        let mut b = ToolBelief::with_beta(3, 0.5);
        b.update(&[0.0, 10.0, 0.0]); // → [0,10,0]
        b.update(&[2.0, 1.0, 0.0]); // leak=5 → [max(-5,0)+2, max(5,0)+1, max(-5,0)+0]
        assert_eq!(b.scores(), &[2.0, 6.0, 0.0]);
        assert_eq!(b.top(), Some((1, 6.0)));
    }

    #[test]
    fn stale_topic_is_overtaken_after_switch() {
        // Slot 0 leads, then slot 1 gets fresh support: with beta=0.5 the stale
        // leader bleeds and the new tool overtakes in a single step.
        let mut b = ToolBelief::with_beta(2, 0.5);
        b.update(&[10.0, 0.0]); // → [10,0]
        b.update(&[0.0, 8.0]); // leak=5 → [max(5,0)+0, max(-5,0)+8] = [5,8]
        assert_eq!(b.scores(), &[5.0, 8.0]);
        assert_eq!(b.top(), Some((1, 8.0)));
    }

    #[test]
    fn stable_topic_leader_survives_its_own_leak() {
        // Repeated support for the same slot: the leader re-earns its leak every
        // step and stays on top while a noisy follower is pinned near zero.
        let mut b = ToolBelief::with_beta(2, 0.5);
        b.update(&[6.0, 1.0]); // → [6,1]
        b.update(&[6.0, 1.0]); // leak=3 → [3+6, max(-2,0)+1] = [9,1]
        assert_eq!(b.scores(), &[9.0, 1.0]);
        b.update(&[6.0, 1.0]); // leak=4.5 → [4.5+6, max(-3.5,0)+1] = [10.5,1]
        assert_eq!(b.scores(), &[10.5, 1.0]);
        assert_eq!(b.top(), Some((0, 10.5)));
    }

    #[test]
    fn per_slot_beta_decays_slots_at_different_rates() {
        // Slot 0 leaks fast (β=0.8), slot 1 slow (β=0.1); same global floor m.
        let mut b = ToolBelief::with_betas(vec![0.8, 0.1]);
        b.update(&[10.0, 10.0]); // → [10,10]
        b.update(&[0.0, 0.0]); // m=10 → [max(10-8,0), max(10-1,0)] = [2, 9]
        assert_eq!(b.scores(), &[2.0, 9.0]);
    }

    #[test]
    fn top_k_ranks_and_drops_zero_mass_slots() {
        let mut b = ToolBelief::with_beta(4, 0.0);
        b.update(&[3.0, 0.0, 7.0, 1.0]);
        assert_eq!(b.top_k(2), vec![(2, 7.0), (0, 3.0)]);
        // Only three slots carry mass, so top_k(10) returns three.
        assert_eq!(b.top_k(10).len(), 3);
    }

    #[test]
    fn top_is_none_before_any_mass() {
        let b = ToolBelief::new(5);
        assert_eq!(b.top(), None);
        assert!(b.top_k(3).is_empty());
    }

    #[test]
    fn clear_resets_mass_but_keeps_shape() {
        let mut b = ToolBelief::with_beta(3, 0.4);
        b.update(&[1.0, 2.0, 3.0]);
        b.clear();
        assert_eq!(b.scores(), &[0.0, 0.0, 0.0]);
        assert_eq!(b.len(), 3);
        assert_eq!(b.top(), None);
    }
}
