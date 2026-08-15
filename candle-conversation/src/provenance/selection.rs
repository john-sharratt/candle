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
    /// Whether each slot has ever been selected **on its own evidence** — i.e.
    /// its belief reached the selection threshold, rather than the min-budget
    /// force-fill putting it in scope. Sticky for the turn (carried across
    /// reprojections via [`Self::seed`]). Only a qualified slot is eligible for
    /// the early-decode carry floor: the floor exists to protect a real pick
    /// whose decode-Q is still accruing, and applying it to a force-filled slot
    /// would manufacture a lock-on out of a slot that never showed evidence.
    qualified: Vec<bool>,
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
            qualified: vec![false; n],
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
    pub fn seed(&mut self, scores: &[f32], selected: &[bool], qualified: &[bool]) {
        self.belief.set_scores(scores);
        for i in 0..self.selected.len() {
            self.selected[i] = selected.get(i).copied().unwrap_or(false);
            self.qualified[i] = qualified.get(i).copied().unwrap_or(false);
        }
    }

    /// Whether slot `i` has ever been selected on its own evidence this turn.
    pub fn is_qualified(&self, i: usize) -> bool {
        self.qualified[i]
    }

    /// Apply one projection's per-section scores, then re-evaluate selection.
    pub fn update(&mut self, scores: &[f32]) {
        self.update_with_floor(scores, None);
    }

    /// [`Self::update`] with an early-decode carry floor: after the RelLeak decay
    /// step and *before* selection is re-evaluated, every slot that was
    /// **prior-selected** (seeded via [`Self::seed`] / newly force-carried) has its
    /// belief raised to at least `floor`. A carried lock-on therefore decays
    /// *relative to fresher rivals* (a stronger newcomer still outranks it) but
    /// cannot fall below the floor — so it survives the early-decode window instead
    /// of being evicted while its own decode-Q signal is still accruing. Unselected
    /// slots are untouched (a floor there would fabricate lock-on from nothing).
    /// `None` reproduces plain [`Self::update`].
    pub fn update_with_floor(&mut self, scores: &[f32], floor: Option<f32>) {
        self.belief.update(scores);
        if let Some(floor) = floor {
            let mut s = self.belief.scores().to_vec();
            for (i, sc) in s.iter_mut().enumerate() {
                if self.selected[i] && self.qualified[i] && *sc < floor {
                    *sc = floor;
                }
            }
            self.belief.set_scores(&s);
        }
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
            // never reached `min_score` — a weak-but-real signal still beats an empty
            // scope. A slot with **no** evidence is not a weak signal, though: when
            // every belief is zero the ordering below is arbitrary (a stable sort over
            // equal keys keeps catalog order), so force-filling there selects whichever
            // member happens to sort first and presents it as a choice. Requiring a
            // positive belief leaves the scope empty instead, which is the honest
            // reading of "nothing matched".
            if keep.len() < b.min {
                let mut rest: Vec<usize> = members
                    .iter()
                    .copied()
                    .filter(|&i| !keep.contains(&i) && s[i] > 0.0)
                    .collect();
                rest.sort_by(|&a, &c| s[c].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
                let need = b.min - keep.len();
                keep.extend(rest.into_iter().take(need));
            }
            // Qualification is "reached the selection threshold on its own
            // belief", recorded stickily so the early-decode floor may protect it.
            // Deliberately NOT the `eligible` set: that admits an already-selected
            // member on the lower `evict_score`, which would let a force-filled
            // slot qualify without ever crossing the real bar.
            for &i in &members {
                if s[i] >= self.policy[i].min_score {
                    self.qualified[i] = true;
                }
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
    fn carry_floor_keeps_decaying_selection_in_scope() {
        // A slot selected at 300 then starved of fresh signal: without the floor
        // it decays past the eviction threshold and drops; with a 250 floor it is
        // held at 250 and stays selected — the early-decode grace behaviour.
        let p = policy(0.40, 250.0, 187.5);
        let mut sel = uniform(1, p, OPEN);
        sel.update(&[300.0]); // selected at 300
        assert!(sel.is_selected(0));
        sel.update_with_floor(&[0.0], Some(250.0)); // 300*0.6 = 180 → floored to 250
        assert_eq!(sel.scores(), &[250.0]);
        assert!(sel.is_selected(0));
    }

    #[test]
    fn carry_floor_still_ranks_below_fresh_newcomer() {
        // Budget 1: the floored incumbent must not out-rank a stronger fresh pick.
        let mut sel = SectionSelector::new(
            vec![policy(0.40, 250.0, 187.5); 2],
            vec![GroupBudget { min: 0, max: 1 }],
        );
        sel.update(&[300.0, 0.0]); // slot 0 selected
                                   // slot 0 decays (300*0.6=180 → floored 250); slot 1 fresh 400 wins the slot.
        sel.update_with_floor(&[0.0, 400.0], Some(250.0));
        assert_eq!(sel.scores(), &[250.0, 400.0]);
        assert!(!sel.is_selected(0));
        assert!(sel.is_selected(1));
    }

    #[test]
    fn carry_floor_does_not_lift_unselected_slots() {
        let mut sel = uniform(1, policy(0.40, 250.0, 187.5), OPEN);
        // Never selected: fresh 50, floor must not fabricate a 250 lock-on.
        sel.update_with_floor(&[50.0], Some(250.0));
        assert_eq!(sel.scores(), &[50.0]);
        assert!(!sel.is_selected(0));
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
