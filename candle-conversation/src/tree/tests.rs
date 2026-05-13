use std::sync::Arc;

use super::config::ConversationTreeConfig;
use super::conversation_tree::ConversationTree;
use super::patch::TreePatch;
use super::types::{NodeId, SegmentId, TurnId, TurnType};
use crate::time_source::FixedTimeSource;

fn make_tree_fixed_day(day: i32) -> ConversationTree {
    ConversationTree::with_config(
        "You are Bramble.",
        ConversationTreeConfig {
            temporal_markers_enabled: true,
            summarize_every: 4,
            summarize_on_day_boundary: false,
            ..ConversationTreeConfig::default()
        },
    )
    .with_time_source(Arc::new(FixedTimeSource::at_day(day)))
}

#[test]
fn turn_id_marker_format() {
    let tid = TurnId { day: 3, seq: 47 };
    assert_eq!(tid.temporal_marker(), "[T-3.47]");
}

#[test]
fn ordering_seq_turn_vs_segment() {
    let tid = TurnId { day: 0, seq: 5 };
    let sid = SegmentId {
        start_turn: TurnId { day: 0, seq: 1 },
        end_turn: TurnId { day: 0, seq: 4 },
    };
    assert_eq!(NodeId::Turn(tid).ordering_seq(), 5);
    assert_eq!(NodeId::Segment(sid).ordering_seq(), 4);
}

#[test]
fn tree_appends_turns_monotonically() {
    let mut tree = make_tree_fixed_day(0);
    let t1 = tree.finish_turn("hello", "hi", TurnType::Reality, vec![], None);
    let t2 = tree.finish_turn("bye", "see ya", TurnType::Reality, vec![], None);
    assert_eq!(t1.seq, 1);
    assert_eq!(t2.seq, 2);
    assert_eq!(tree.nodes().count(), 2);
}

#[test]
fn turns_since_no_segment() {
    let mut tree = make_tree_fixed_day(0);
    tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    tree.finish_turn("c", "d", TurnType::Reality, vec![], None);
    assert_eq!(tree.turns_since_last_summarize(), 2);
}

#[test]
fn summarize_fires_at_n_turns() {
    // summarize_every = 4, day boundary off — stub fires, doesn't mutate nodes.
    let mut tree = make_tree_fixed_day(0);
    for i in 0..4u32 {
        tree.finish_turn(
            format!("user {i}"),
            format!("asst {i}"),
            TurnType::Reality,
            vec![],
            None,
        );
    }
    // Stub doesn't insert segments; nodes stays at 4.
    assert_eq!(tree.nodes().count(), 4);
}

#[test]
fn compute_marker_disabled() {
    let tree = ConversationTree::new("sys");
    assert_eq!(tree.compute_marker(), "");
}

#[test]
fn compute_marker_enabled() {
    let tree = make_tree_fixed_day(2);
    // day=2, next_seq=1 (no turns added yet)
    assert_eq!(tree.compute_marker(), "[T-2.1]");
}

#[test]
fn system_prompt_postfix_appended_when_markers_enabled() {
    let tree = make_tree_fixed_day(0);
    assert!(tree.system_prompt_text().contains("T-{days}.{seq}"));
}

#[test]
fn turn_type_default_is_reality() {
    assert_eq!(TurnType::default(), TurnType::Reality);
}

#[test]
fn fork_patch_round_trip() {
    let mut tree = ConversationTree::new("sys");
    let (mut fork, rx) = tree.fork();
    fork.inner
        .finish_turn("q", "a", TurnType::Reality, vec![], None);
    fork.finish(TreePatch {
        appended: vec![],
        metadata: None,
    });
    let patch = rx.recv().unwrap();
    tree.apply_patch(patch);
    // No nodes appended (patch.appended was empty).
    assert_eq!(tree.nodes().count(), 0);
}
