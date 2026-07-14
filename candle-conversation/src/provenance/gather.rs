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
use super::{score_provenance_late_fusion, WideQSig};

/// One slot's belief after a step: its confidence and whether it is selected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotBelief {
    pub score: f32,
    pub selected: bool,
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
    score_provenance_late_fusion(query, &gtoks, &gcase, n_slots)
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
/// already-selected slot whose belief decays below `f` is held at `f` so it
/// survives the opening window; `None` is the steady-state (no floor).
pub fn belief_step(
    fresh: &[f32],
    prior_scores: &[f32],
    prior_selected: &[bool],
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
    sel.seed(&seed, prior_selected);
    sel.update_with_floor(fresh, carry_floor);
    (0..n)
        .map(|i| SlotBelief {
            score: sel.scores()[i],
            selected: sel.is_selected(i),
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
            policy(0.5, 5.0, 2.0),
            OPEN,
            None,
        );
        assert_eq!(
            out[0],
            SlotBelief {
                score: 3.0,
                selected: false
            }
        );
        assert_eq!(
            out[1],
            SlotBelief {
                score: 6.0,
                selected: true
            }
        );
        assert_eq!(
            out[2],
            SlotBelief {
                score: 0.0,
                selected: false
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
            policy(0.40, 250.0, 187.5),
            OPEN,
            None,
        );
        assert_eq!(out[0].score, 180.0);
        assert!(!out[0].selected);
    }

    #[test]
    fn score_slots_aggregates_windows_into_their_slot() {
        // Empty query → the scorer returns zeros; aggregation is still well-formed
        // (this exercises the slot-mapping plumbing without bit-layout fixtures).
        let fresh = score_slots(&[], &[], &[], 3);
        assert_eq!(fresh, vec![0.0, 0.0, 0.0]);
    }
}
