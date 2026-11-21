//! Unit tests for the carried belief: merging, decay, and the turn-boundary
//! challenger.
//!
//! The merge tests pin that an absent collection keeps its carried belief while a
//! present one is replaced wholesale. A turn that skipped projection leaves the
//! carry alone. Decay halves the scores and leaves the selected flags unchanged.
//! Turn keys are qualified by timeline, and the challenger seats only the
//! strongest unselected fresh signal, evicting the weakest incumbent only when
//! the selection is full.

use super::*;

#[cfg(test)]
pub(super) mod prior_belief_tests {
    use super::{GroupKey, PriorBelief, TimelineId, TurnIndex, TurnKey};

    /// Belief-map turn keys are timeline-qualified (`"<timeline>:<index>"`), so
    /// turn 3 of one conversation can't collide with turn 3 of another.
    pub(in super::super) fn tk(n: u32) -> TurnKey {
        TurnKey::new(TimelineId::for_test(1), TurnIndex(n))
    }

    pub(in super::super) fn tkey(n: u32) -> String {
        format!("{}:{}", TimelineId::for_test(1).raw(), n)
    }

    #[test]
    fn merge_from_overlays_present_collections_and_preserves_absent_ones() {
        // The carried belief holds a tools lock-on from earlier turns.
        let mut carried = PriorBelief::default();
        carried.set("tools", "calculator", 2500.0, true, true);
        carried.set("tools", "datetime", 400.0, false, false);

        // A tools-off turn harvests a belief WITHOUT the tools key (the
        // collection materialised nothing, so the selection omitted it) but
        // with fresh state for another collection.
        let mut newer = PriorBelief::default();
        newer.set("notes", "pinned", 40.0, true, true);

        carried.merge_from(&newer);

        // The absent collection survives untouched — the tools lock-on is
        // NOT erased by a turn that said nothing about tools...
        assert_eq!(carried.coll("tools")["calculator"], (2500.0, true, true));
        assert_eq!(carried.coll("tools")["datetime"], (400.0, false, false));
        // ...and the present collection is overlaid.
        assert_eq!(carried.coll("notes")["pinned"], (40.0, true, true));
    }

    #[test]
    fn merge_from_replaces_a_present_collection_wholesale() {
        let mut carried = PriorBelief::default();
        carried.set("tools", "calculator", 2500.0, true, true);
        carried.set("tools", "datetime", 400.0, false, false);

        // A newer turn re-projected tools with a different member set: the
        // collection entry is replaced, not unioned — sections absent from the
        // newer selection are gone (their belief genuinely ended).
        let mut newer = PriorBelief::default();
        newer.set("tools", "web_search", 900.0, true, true);

        carried.merge_from(&newer);

        let tools = carried.coll("tools");
        assert_eq!(tools.get("web_search"), Some(&(900.0, true, true)));
        assert_eq!(tools.get("calculator"), None);
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn merge_from_empty_belief_is_a_no_op() {
        // A projection-skipped turn harvests a default belief: merging it must
        // not disturb the carry.
        let mut carried = PriorBelief::default();
        carried.set("tools", "calculator", 2500.0, true, true);

        carried.merge_from(&PriorBelief::default());

        assert_eq!(carried.coll("tools")["calculator"], (2500.0, true, true));
    }

    #[test]
    fn decay_scores_scales_scores_and_leaves_selected_flags() {
        let mut carried = PriorBelief::default();
        // A carried tool lock-on (repo_task_start) plus a runner-up (datetime), as at
        // a topic-changed turn boundary.
        carried.set("tools", "repo_task_start", 1009.0, true, true);
        carried.set("tools", "datetime", 0.0, false, false);
        carried.set("notes", "pinned", 40.0, true, true);

        carried.decay_scores(0.5);

        // Scores halve so the fresh decode-Q of a new query can overtake the
        // stale leader early; the `selected` flags are the ranking input, not
        // the lever, so they carry unchanged.
        assert_eq!(
            carried.coll("tools")["repo_task_start"],
            (504.5, true, true)
        );
        assert_eq!(carried.coll("tools")["datetime"], (0.0, false, false));
        assert_eq!(carried.coll("notes")["pinned"], (20.0, true, true));
    }

    #[test]
    fn decay_scores_on_empty_belief_is_a_no_op() {
        let mut carried = PriorBelief::default();
        carried.decay_scores(0.5);
        assert!(carried.is_empty());
    }

    // ── Turn-group belief (the generalized axis) ────────────────────────────

    #[test]
    fn from_selection_carries_only_selected_memory_turns_by_group() {
        use crate::projection::event::{ProjectionSelection, SelectedTurn};
        let mk = |group: &str, index: u32, selected: bool, score: f32| SelectedTurn {
            // These fixtures stand for turns that won their slot on their own
            // belief, so qualification tracks selection.
            qualified: selected,
            layer: "repo_map".to_string(),
            group: group.to_string(),
            index,
            role: "user".to_string(),
            tokens: 10,
            kind: crate::summary_tree::TurnKind::Normal,
            reason: None,
            timeline: Some(1),
            selected,
            score,
        };
        let sel = ProjectionSelection {
            system: Vec::new(),
            turns: vec![
                mk("structure", 0, true, 500.0),       // carried
                mk("structure", 1, false, 200.0),      // unselected → not carried
                mk("structure", u32::MAX, false, 0.0), // live message → skipped
            ],
        };
        let pb = PriorBelief::from_selection(&sel);
        let group = pb.tgroup("structure");
        assert_eq!(group.get(&tkey(0)), Some(&(500.0, true, true)));
        assert_eq!(group.get(&tkey(1)), None);
        assert_eq!(group.len(), 1, "only the selected memory turn is carried");
    }

    #[test]
    fn turn_group_aligns_carried_belief_to_candidate_order() {
        let mut pb = PriorBelief::default();
        pb.set_turn("structure", tk(3), 500.0, true, true);
        pb.set_turn("structure", tk(7), 200.0, false, false);
        // Candidates in a different order than they were carried; the seed must
        // realign by turn index, and unknown turns read (0.0, false).
        let turns = vec![(tk(7), 0.0), (tk(3), 0.0), (tk(5), 0.0)];
        let (scores, selected, _qualified) = pb.turn_group("structure", &turns);
        assert_eq!(scores, vec![200.0, 500.0, 0.0]);
        assert_eq!(selected, vec![false, true, false]);
    }

    #[test]
    fn challenger_seats_a_turn_group_member_by_index_key() {
        let mut pb = PriorBelief::default();
        // One carried cluster; a spare slot at budget_max 3.
        pb.set_turn("clusters", tk(0), 900.0, true, true);
        pb.seat_turn_boundary_challenger(
            GroupKey::TurnGroup("clusters".into()),
            &[(tkey(4), 246.0), (tkey(9), 30.0)],
            3,
            100.0,
        );
        let g = pb.tgroup("clusters");
        assert_eq!(g[&tkey(0)], (900.0, true, true), "incumbent survives");
        assert_eq!(
            g[&tkey(4)],
            (100.0, true, false),
            "strongest fresh turn seated, but unqualified — it never crossed the bar"
        );
        assert!(g.get(&tkey(9)).is_none(), "weaker rival not seated");
    }

    #[test]
    fn merge_from_and_decay_span_both_axes() {
        let mut carried = PriorBelief::default();
        carried.set("tools", "calc", 400.0, true, true);
        carried.set_turn("structure", tk(2), 800.0, true, true);
        // decay scales scores on both the collection and turn-group axes.
        carried.decay_scores(0.5);
        assert_eq!(carried.coll("tools")["calc"], (200.0, true, true));
        assert_eq!(carried.tgroup("structure")[&tkey(2)], (400.0, true, true));
        // merge overlays a turn group wholesale, leaving the collection intact.
        let mut newer = PriorBelief::default();
        newer.set_turn("structure", tk(5), 900.0, true, true);
        carried.merge_from(&newer);
        assert_eq!(
            carried.tgroup("structure").get(&tkey(2)),
            None,
            "replaced wholesale"
        );
        assert_eq!(carried.tgroup("structure")[&tkey(5)], (900.0, true, true));
        assert_eq!(
            carried.coll("tools")["calc"],
            (200.0, true, true),
            "other axis intact"
        );
    }

    #[test]
    fn challenger_fills_a_spare_slot_without_evicting() {
        // Carried a single (topic-stale) tool from the previous turn; the new
        // query's strongest fresh signal is a below-threshold datetime.
        let mut b = PriorBelief::default();
        b.set("tools", "repo_task_start", 350.0, true, true);
        // budget_max 3 with one incumbent → room for the challenger, no eviction.
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 246.0), ("weather".into(), 30.0)],
            3,
            1000.0,
        );
        // The incumbent survives; datetime is seated selected at the seed score.
        assert_eq!(b.coll("tools")["repo_task_start"], (350.0, true, true));
        // Seated selected, but unqualified: it is below the bar by construction.
        assert_eq!(b.coll("tools")["datetime"], (1000.0, true, false));
        // A weaker fresh rival is NOT seated — only the single strongest.
        assert!(b.coll("tools").get("weather").is_none());
    }

    #[test]
    fn challenger_evicts_the_weakest_incumbent_when_full() {
        let mut b = PriorBelief::default();
        b.set("tools", "http_request", 1800.0, true, true);
        b.set("tools", "repo_task_start", 400.0, true, true); // weakest incumbent
        b.set("tools", "code_run", 1200.0, true, true);
        // At budget_max=3 → the challenger displaces the lowest-scored incumbent.
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 246.0)],
            3,
            1000.0,
        );
        assert_eq!(b.coll("tools")["repo_task_start"], (0.0, false, false)); // evicted
        assert_eq!(b.coll("tools")["datetime"], (1000.0, true, false)); // seated, unqualified
                                                                        // The strong incumbents survive.
        assert_eq!(b.coll("tools")["http_request"], (1800.0, true, true));
        assert_eq!(b.coll("tools")["code_run"], (1200.0, true, true));
    }

    #[test]
    fn challenger_is_a_no_op_when_top_fresh_is_already_selected() {
        // Continuation: the strongest fresh signal is the carried tool itself,
        // and no other fresh signal exists → nothing to seat, carry untouched.
        let mut b = PriorBelief::default();
        b.set("tools", "datetime", 1500.0, true, true);
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 900.0)],
            3,
            1000.0,
        );
        assert_eq!(b.coll("tools")["datetime"], (1500.0, true, true));
        assert_eq!(b.coll("tools").len(), 1);
    }

    #[test]
    fn challenger_does_not_apply_below_top_3() {
        // A top-1/top-2 collection has no spare slot to lend a challenger.
        let mut b = PriorBelief::default();
        b.set("tools", "repo_task_start", 350.0, true, true);
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 246.0)],
            2,
            1000.0,
        );
        assert_eq!(b.coll("tools")["repo_task_start"], (350.0, true, true));
        assert!(b.coll("tools").get("datetime").is_none());
    }
}
