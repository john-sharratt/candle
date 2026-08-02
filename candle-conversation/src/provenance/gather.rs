//! The online belief gather — the runtime glue that turns a live probe into a
//! selected set of sections, carrying belief across a turn's reprojections.
//!
//! Two pieces, split so the risky part is unit-testable without synthesising
//! signature bit-layouts:
//!
//! - [`score_slots`] — scan the probe against a (tag-filtered) gallery of past
//!   turn windows, each mapped to the section slot it selected, and return the
//!   fresh per-slot `z × margin` score. A thin wrapper over the already-tested
//!   [`score_provenance_late_fusion`] (tokens tagged by slot).
//! - [`belief_step`] — seed the [`SectionSelector`] from the prior projection's
//!   scores + selected flags, apply the fresh scores under the policy, and read
//!   out the new per-slot belief + selection. Pure over plain floats.
//!
//! See `docs/tool_selection_provenance_results.md` §24.

use super::selection::{GroupBudget, SectionPolicy, SectionSelector};
use super::{score_provenance_late_fusion_weighted, WideQSig};

/// One slot's belief after a step: its confidence and whether it is selected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotBelief {
    pub score: f32,
    pub selected: bool,
    /// Whether this slot has ever reached the selection threshold on its own
    /// belief this turn (as opposed to being force-filled by the min budget).
    /// Carried across reprojections; gates the early-decode carry floor.
    pub qualified: bool,
}

/// Scan `query` (the live probe window) against a gallery of past turn windows,
/// each mapped by `gallery_slot[i]` to the section slot it selected, and return
/// the fresh per-slot score. Every gallery token is tagged with its window's
/// **slot** so [`score_provenance_late_fusion`] computes the `z × margin` vote
/// directly over slots (the margin is tool-vs-tool, not window-vs-window). The
/// caller assembles `gallery_windows` already restricted to the policy's tag
/// scope. Windows whose slot is out of range are ignored.
pub fn score_slots(
    query: &[WideQSig],
    gallery_windows: &[&[WideQSig]],
    gallery_slot: &[usize],
    n_slots: usize,
) -> Vec<f32> {
    score_slots_weighted(query, gallery_windows, gallery_slot, n_slots, &[])
}

/// [`score_slots`] with a per-layer-group weight on the vote (see
/// [`score_provenance_late_fusion_weighted`]). `group_weights` empty ⇒ uniform.
pub fn score_slots_weighted(
    query: &[WideQSig],
    gallery_windows: &[&[WideQSig]],
    gallery_slot: &[usize],
    n_slots: usize,
    group_weights: &[f32],
) -> Vec<f32> {
    let total: usize = gallery_windows.iter().map(|w| w.len()).sum();
    let mut gtoks: Vec<&WideQSig> = Vec::with_capacity(total);
    let mut gcase: Vec<u32> = Vec::with_capacity(total);
    for (wi, w) in gallery_windows.iter().enumerate() {
        let slot = gallery_slot.get(wi).copied().unwrap_or(usize::MAX);
        if slot >= n_slots {
            continue;
        }
        for t in w.iter() {
            gtoks.push(t);
            gcase.push(slot as u32);
        }
    }
    score_provenance_late_fusion_weighted(query, &gtoks, &gcase, n_slots, group_weights)
}

/// One online belief update: reseed from the prior projection (`prior_scores` +
/// `prior_selected`), apply the `fresh` per-slot scores under `policy`/`budget`,
/// and read out the new per-slot belief + selection. All slots share one policy
/// and one budget group (the collection).
///
/// `fresh.len()` is the slot count; `prior_scores` / `prior_selected` are indexed
/// the same way (shorter prior arrays read as zero / unselected).
///
/// `carry_floor` is the early-decode grace floor (see
/// [`crate::projection::PolicyConfig::windowed`]): when `Some(f)`, a carried
/// already-selected slot that has *qualified* (reached the selection threshold on
/// its own belief at some point this turn) and whose belief decays below `f` is
/// held at `f` so it survives the opening window; `None` is the steady-state (no
/// floor). A slot that only ever entered scope through the min-budget force-fill
/// is never floored — the floor protects a real pick whose decode-Q is still
/// accruing, not a slot that has shown nothing.
pub fn belief_step(
    fresh: &[f32],
    prior_scores: &[f32],
    prior_selected: &[bool],
    prior_qualified: &[bool],
    policy: SectionPolicy,
    budget: GroupBudget,
    carry_floor: Option<f32>,
) -> Vec<SlotBelief> {
    let n = fresh.len();
    let mut seed = vec![0.0f32; n];
    for (i, s) in seed.iter_mut().enumerate() {
        *s = prior_scores.get(i).copied().unwrap_or(0.0);
    }
    let mut sel = SectionSelector::new(vec![policy; n], vec![budget]);
    sel.seed(&seed, prior_selected, prior_qualified);
    sel.update_with_floor(fresh, carry_floor);
    (0..n)
        .map(|i| SlotBelief {
            score: sel.scores()[i],
            selected: sel.is_selected(i),
            qualified: sel.is_qualified(i),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(beta: f32, min: f32, evict: f32) -> SectionPolicy {
        SectionPolicy {
            group: 0,
            beta,
            min_score: min,
            evict_score: evict,
        }
    }

    const OPEN: GroupBudget = GroupBudget {
        min: 0,
        max: usize::MAX,
    };

    #[test]
    fn first_projection_seeds_from_zero_and_selects_above_min() {
        // No prior → belief equals the fresh scores; slot 1 crosses min 5.
        let out = belief_step(
            &[3.0, 6.0, 0.0],
            &[],
            &[],
            &[],
            policy(0.5, 5.0, 2.0),
            OPEN,
            None,
        );
        assert_eq!(
            out[0],
            SlotBelief {
                score: 3.0,
                selected: false,
                qualified: false
            }
        );
        assert_eq!(
            out[1],
            SlotBelief {
                score: 6.0,
                selected: true,
                qualified: true
            }
        );
        assert_eq!(
            out[2],
            SlotBelief {
                score: 0.0,
                selected: false,
                qualified: false
            }
        );
    }

    #[test]
    fn prior_belief_is_reseeded_and_decayed_across_reprojections() {
        // Prior leader slot 0 (score 10), fresh support for slot 1: RelLeak
        // β=0.5 → leak 5, so slot 0 → (10-5)+0 = 5, slot 1 → 0+8 = 8; 1 overtakes.
        let out = belief_step(
            &[0.0, 8.0],
            &[10.0, 0.0],
            &[true, false],
            &[true, false],
            policy(0.5, 5.0, 2.0),
            OPEN,
            None,
        );
        assert_eq!(out[0].score, 5.0);
        assert_eq!(out[1].score, 8.0);
        assert!(out[1].selected);
    }

    #[test]
    fn prior_leader_that_decays_below_evict_is_dropped() {
        // No pin: prior-selected slot 0 decays below evict while a fresh strong slot
        // 1 takes the single budget slot — the incumbent is not held.
        let out = belief_step(
            &[0.0, 20.0],
            &[6.0, 0.0],
            &[true, false],
            &[true, false],
            policy(0.5, 5.0, 2.0),
            GroupBudget { min: 0, max: 1 },
            None,
        );
        // slot 0 leaks 0.5*6 = 3 → below min; slot 1 is far stronger and takes max=1.
        assert!(!out[0].selected, "unpinned decayed leader is dropped");
        assert!(out[1].selected, "fresh strong slot wins the budget slot");
    }

    #[test]
    fn budget_max_caps_the_selected_set() {
        // Four slots cross min 0; max 2 keeps the two strongest.
        let out = belief_step(
            &[3.0, 9.0, 1.0, 7.0],
            &[],
            &[],
            &[],
            policy(0.0, 0.0, 0.0),
            GroupBudget { min: 0, max: 2 },
            None,
        );
        let sel: Vec<bool> = out.iter().map(|s| s.selected).collect();
        assert_eq!(sel, vec![false, true, false, true]);
    }

    #[test]
    fn carry_floor_holds_a_decaying_pick_above_the_early_min() {
        // A prior-selected leader (score 300) with no fresh support decays to
        // 300*0.6 = 180 under β=0.40; the carry floor 250 holds it at 250 so it
        // clears the (windowed) evict 187.5 and stays selected — the early-decode
        // grace the scheduler applies for the first 64 tokens of a turn.
        let out = belief_step(
            &[0.0],
            &[300.0],
            &[true],
            &[true],
            policy(0.40, 250.0, 187.5),
            OPEN,
            Some(250.0),
        );
        assert_eq!(out[0].score, 250.0);
        assert!(out[0].selected);
        // Without the floor the same pick decays out of the band.
        let out = belief_step(
            &[0.0],
            &[300.0],
            &[true],
            &[true],
            policy(0.40, 250.0, 187.5),
            OPEN,
            None,
        );
        assert_eq!(out[0].score, 180.0);
        assert!(!out[0].selected);
    }

    /// The carry floor is for a pick that showed evidence. A slot that only ever
    /// entered scope through the min-budget force-fill never qualified, so the
    /// floor must not lift it — otherwise one evidence-free opening pick is
    /// inflated to the early bar and locked in for the whole grace window, which
    /// is how an unrelated tool came to sit at exactly `early_min_score` for
    /// several reprojections while the right one never got in.
    #[test]
    fn carry_floor_ignores_a_pick_that_never_qualified() {
        let out = belief_step(
            &[0.0],
            &[10.0],
            &[true],  // carried as selected …
            &[false], // … but never crossed min on its own belief
            policy(0.40, 250.0, 187.5),
            OPEN,
            Some(250.0),
        );
        assert_eq!(out[0].score, 6.0, "decays normally, no floor applied");
        assert!(!out[0].selected, "and falls out of the band");
        assert!(!out[0].qualified);
    }

    /// Crossing `min_score` marks a slot qualified, and that survives the trip
    /// through a projection event (the caller feeds it back as `prior_qualified`).
    #[test]
    fn qualification_is_recorded_when_a_slot_crosses_min() {
        let first = belief_step(
            &[300.0],
            &[],
            &[],
            &[],
            policy(0.40, 250.0, 187.5),
            OPEN,
            None,
        );
        assert!(first[0].selected);
        assert!(first[0].qualified, "300 >= min 250");

        // Fed back, it is now eligible for the floor.
        let second = belief_step(
            &[0.0],
            &[first[0].score],
            &[first[0].selected],
            &[first[0].qualified],
            policy(0.40, 250.0, 187.5),
            OPEN,
            Some(250.0),
        );
        assert_eq!(second[0].score, 250.0, "floored, because it qualified");
    }

    /// With no evidence anywhere, a min budget must not manufacture a selection:
    /// the ordering among all-zero beliefs is arbitrary, so force-filling there
    /// presents whichever slot happens to sort first as if it were a choice.
    #[test]
    fn min_budget_does_not_force_fill_from_an_empty_field() {
        let out = belief_step(
            &[0.0, 0.0, 0.0],
            &[],
            &[],
            &[],
            policy(0.40, 800.0, 600.0),
            GroupBudget { min: 1, max: 3 },
            None,
        );
        assert!(
            out.iter().all(|b| !b.selected),
            "an all-zero field selects nothing",
        );
    }

    /// The turn-boundary challenger is seated at the selection bar but has not
    /// crossed it, so it must decay out if its fresh signal does not hold up. If
    /// the floor could reach it, it would instead sit at the seed score for the
    /// whole grace window and squat a budget slot.
    #[test]
    fn a_seated_challenger_decays_out_instead_of_being_floored() {
        // Seeded exactly at the windowed bar (250), selected, unqualified — the
        // shape `seat_turn_boundary_challenger` produces.
        let out = belief_step(
            &[0.0],
            &[250.0],
            &[true],
            &[false],
            policy(0.40, 250.0, 187.5),
            GroupBudget { min: 0, max: 3 },
            Some(250.0),
        );
        assert_eq!(out[0].score, 150.0, "decays by beta, no floor");
        assert!(!out[0].selected, "and drops out of the band");
    }

    /// A weak-but-real signal still force-fills — the guard is specifically about
    /// *no* evidence, not about being below `min_score`.
    #[test]
    fn min_budget_still_force_fills_the_strongest_nonzero_slot() {
        let out = belief_step(
            &[0.0, 12.0, 3.0],
            &[],
            &[],
            &[],
            policy(0.40, 800.0, 600.0),
            GroupBudget { min: 1, max: 3 },
            None,
        );
        let sel: Vec<bool> = out.iter().map(|b| b.selected).collect();
        assert_eq!(sel, vec![false, true, false]);
        assert!(!out[1].qualified, "force-filled, not qualified");
    }

    #[test]
    fn score_slots_aggregates_windows_into_their_slot() {
        // Empty query → the scorer returns zeros; aggregation is still well-formed
        // (this exercises the slot-mapping plumbing without bit-layout fixtures).
        let fresh = score_slots(&[], &[], &[], 3);
        assert_eq!(fresh, vec![0.0, 0.0, 0.0]);
    }
}
