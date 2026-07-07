//! Online section-selection policy over a [`ToolBelief`].
//!
//! The belief accumulates a per-section confidence; this layer decides which
//! sections are actually *in projection*.  A section is selected once its
//! attenuated confidence reaches `min_score`, stays selected through the
//! hysteresis band, and is evicted once it drops below `evict_score`.  Per-group
//! budgets bound how many sections a collection or substrate layer may contribute.
//!
//! # Lifecycle
//!
//! ```text
//!   unselected ──(score ≥ min_score)──▶ selected ──(score < evict_score)──▶ unselected
//! ```
//!
//! - `evict_score ≤ min_score` gives a stable band: a selected section is not
//!   dropped the instant it dips below the selection threshold, only once it
//!   falls under the lower eviction threshold — no flapping.
//! - Group budgets are enforced after threshold transitions: `max` evicts the
//!   lowest-scoring members; `min` force-selects the highest-scoring unselected
//!   members even if they never reached `min_score`.
//!
//! Selection re-evaluates freely on every reprojection — there is no per-turn
//! pin. A committed tool is held stable during its `<tool_call>` block by the
//! scheduler suppressing reprojection there, not by pinning the selection.
//!
//! See `docs/tool_selection_provenance_results.md` §24.

use super::ToolBelief;

/// Per-section selection policy.  `group` names the collection / substrate layer
/// the section is budgeted under.
#[derive(Clone, Copy, Debug)]
pub struct SectionPolicy {
    /// Budget group (collection or substrate layer) this section belongs to.
    pub group: usize,
    /// Leak fraction for this section's belief (higher = decays faster).
    pub beta: f32,
    /// Confidence a section must reach to be selected.
    pub min_score: f32,
    /// Confidence below which a selected section is evicted. Should be
    /// `≤ min_score` for a stable hysteresis band.
    pub evict_score: f32,
}

/// Per-group selection budget.
#[derive(Clone, Copy, Debug)]
pub struct GroupBudget {
    /// Minimum sections the group must contribute (force-fills from the top).
    pub min: usize,
    /// Maximum sections the group may contribute (evicts the weakest).
    pub max: usize,
}

/// Online selector: a [`ToolBelief`] plus per-section selection state.
pub struct SectionSelector {
    belief: ToolBelief,
    policy: Vec<SectionPolicy>,
    /// Indexed by `SectionPolicy::group`.
    budgets: Vec<GroupBudget>,
    selected: Vec<bool>,
}

impl SectionSelector {
    /// Build a selector.  `policy[i]` governs slot `i`; `budgets[g]` governs the
    /// group `g` that policies point at with `SectionPolicy::group`.
    pub fn new(policy: Vec<SectionPolicy>, budgets: Vec<GroupBudget>) -> Self {
        let betas: Vec<f32> = policy.iter().map(|p| p.beta).collect();
        let n = policy.len();
        Self {
            belief: ToolBelief::with_betas(betas),
            policy,
            budgets,
            selected: vec![false; n],
        }
    }

    /// Number of section slots.
    pub fn len(&self) -> usize {
        self.policy.len()
    }

    /// Whether the selector has no slots.
    pub fn is_empty(&self) -> bool {
        self.policy.is_empty()
    }

    /// The raw per-slot belief confidence.
    pub fn scores(&self) -> &[f32] {
        self.belief.scores()
    }

    /// Reseed from a prior projection: restore the per-slot belief and the
    /// selected set (the latter drives hysteresis — a prior-selected section is
    /// held to its lower `evict_score`). This is how the online belief survives
    /// across a turn's reprojections without the scheduler holding state — the
    /// prior [`super::super::projection::ProjectionEvent`] carries the scores and
    /// selected flags.
    pub fn seed(&mut self, scores: &[f32], selected: &[bool]) {
        self.belief.set_scores(scores);
        for i in 0..self.selected.len() {
            self.selected[i] = selected.get(i).copied().unwrap_or(false);
        }
    }

    /// Apply one projection's per-section scores, then re-evaluate selection.
    pub fn update(&mut self, scores: &[f32]) {
        self.belief.update(scores);
        self.apply_selection();
    }

    /// Currently selected slots, highest confidence first.
    pub fn selected_slots(&self) -> Vec<usize> {
        let s = self.belief.scores();
        let mut sel: Vec<usize> = (0..self.selected.len())
            .filter(|&i| self.selected[i])
            .collect();
        sel.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
        sel
    }

    /// The highest-confidence selected slot, or `None` if nothing is selected.
    pub fn top_selected(&self) -> Option<usize> {
        let s = self.belief.scores();
        (0..self.selected.len())
            .filter(|&i| self.selected[i])
            .max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Whether slot `i` is currently selected.
    pub fn is_selected(&self, i: usize) -> bool {
        self.selected[i]
    }

    fn apply_selection(&mut self) {
        let s = self.belief.scores();
        for g in 0..self.budgets.len() {
            let b = self.budgets[g];
            let members: Vec<usize> = (0..s.len())
                .filter(|&i| self.policy[i].group == g)
                .collect();
            // Eligible: a selected member stays while above its eviction threshold
            // (hysteresis); an unselected member becomes eligible once it reaches
            // its selection threshold.
            let mut eligible: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&i| {
                    if self.selected[i] {
                        s[i] >= self.policy[i].evict_score
                    } else {
                        s[i] >= self.policy[i].min_score
                    }
                })
                .collect();
            eligible.sort_by(|&a, &c| s[c].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut keep: Vec<usize> = eligible.iter().copied().take(b.max).collect();
            // MIN budget: force-fill from the strongest remaining members even if they
            // never reached `min_score`.
            if keep.len() < b.min {
                let mut rest: Vec<usize> = members
                    .iter()
                    .copied()
                    .filter(|&i| !keep.contains(&i))
                    .collect();
                rest.sort_by(|&a, &c| s[c].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
                let need = b.min - keep.len();
                keep.extend(rest.into_iter().take(need));
            }
            for &i in &members {
                self.selected[i] = keep.contains(&i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize, p: SectionPolicy, budget: GroupBudget) -> SectionSelector {
        SectionSelector::new(vec![p; n], vec![budget])
    }

    const OPEN: GroupBudget = GroupBudget {
        min: 0,
        max: usize::MAX,
    };

    fn policy(beta: f32, min: f32, evict: f32) -> SectionPolicy {
        SectionPolicy {
            group: 0,
            beta,
            min_score: min,
            evict_score: evict,
        }
    }

    #[test]
    fn selects_on_reaching_min_score() {
        let mut sel = uniform(2, policy(0.0, 5.0, 2.0), OPEN);
        sel.update(&[3.0, 6.0]);
        assert!(!sel.is_selected(0)); // 3 < min 5
        assert!(sel.is_selected(1)); // 6 ≥ min 5
        assert_eq!(sel.top_selected(), Some(1));
    }

    #[test]
    fn hysteresis_holds_between_evict_and_min() {
        // Selected at 6, decays into the [evict,min) band → stays; below evict → drops.
        let mut sel = uniform(1, policy(0.5, 5.0, 2.0), OPEN);
        sel.update(&[6.0]); // [6] selected
        sel.update(&[0.0]); // m=6 leak=3 → [3]; 2 ≤ 3 < 5 → still selected
        assert!(sel.is_selected(0));
        assert_eq!(sel.scores(), &[3.0]);
        sel.update(&[0.0]); // m=3 leak=1.5 → [1.5] < evict 2 → evicted
        assert!(!sel.is_selected(0));
    }

    #[test]
    fn reselects_freely_when_belief_shifts() {
        // No pinning: a stronger newcomer takes the single budget slot the moment it
        // crosses, and the incumbent that has decayed below evict is dropped.
        let mut sel = SectionSelector::new(
            vec![policy(0.5, 1.0, 0.5); 3],
            vec![GroupBudget { min: 0, max: 1 }],
        );
        sel.update(&[2.0, 0.0, 0.0]); // slot 0 selected (fills max=1)
        assert_eq!(sel.selected_slots(), vec![0]);
        sel.update(&[0.0, 9.0, 8.0]); // slot 0 decays to 1.0; slot 1 (9) is strongest
        assert_eq!(sel.selected_slots(), vec![1]);
    }

    #[test]
    fn max_budget_evicts_weakest() {
        let mut sel = uniform(3, policy(0.0, 0.0, 0.0), GroupBudget { min: 0, max: 2 });
        sel.update(&[3.0, 5.0, 1.0]); // all cross min 0 → 3 selected > max 2
                                      // weakest (slot 2, score 1) evicted.
        assert!(sel.is_selected(0));
        assert!(sel.is_selected(1));
        assert!(!sel.is_selected(2));
    }

    #[test]
    fn min_budget_force_fills_from_top() {
        // min_score unreachable → nothing selected by threshold; min budget picks top-2.
        let mut sel = uniform(3, policy(0.0, 100.0, 50.0), GroupBudget { min: 2, max: 3 });
        sel.update(&[3.0, 5.0, 1.0]);
        assert!(sel.is_selected(0)); // 3 (2nd highest)
        assert!(sel.is_selected(1)); // 5 (highest)
        assert!(!sel.is_selected(2)); // 1 (lowest) left out
    }
}
