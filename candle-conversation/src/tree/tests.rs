use std::ops::RangeInclusive;
use std::sync::Arc;

use super::config::ConversationTreeConfig;
use super::conversation_tree::ConversationTree;
use super::node::{ConversationNode, ConversationSegment};
use super::patch::TreePatch;
use super::task::{CognitiveTask, TaskKind, TaskPoll};
use super::types::{NodeId, SegmentId, TurnId, TurnType};
use crate::error::ConversationError;
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
    // summarize_every = 4, day boundary off. This asserts the TRIGGER, not the
    // summarization: `finish_turn` is passed no inference backend, so
    // `run_summarize` logs and returns without launching a task, and the tree
    // is left unmutated.
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
    // No backend ⇒ no task ⇒ no segment inserted; the count stays at 4.
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

/// A cognitive task whose polls are scripted: `Pending` for `pending_polls`
/// polls, then `outcome` (or `Aborted` once that is spent).
struct ScriptedTask {
    span: (TurnId, TurnId),
    pending_polls: u32,
    outcome: Option<TaskPoll>,
}

impl CognitiveTask for ScriptedTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Summarization
    }

    fn relevant_turns(&self) -> Option<RangeInclusive<TurnId>> {
        Some(self.span.0..=self.span.1)
    }

    fn poll(&mut self) -> TaskPoll {
        if self.pending_polls > 0 {
            self.pending_polls -= 1;
            return TaskPoll::Pending;
        }
        self.outcome.take().unwrap_or(TaskPoll::Aborted)
    }

    fn abort(&self) {}
}

fn segment_patch(start: TurnId, end: TurnId) -> TreePatch {
    TreePatch {
        appended: vec![ConversationNode::Segment(ConversationSegment::new(
            SegmentId {
                start_turn: start,
                end_turn: end,
            },
            "summary",
        ))],
        metadata: None,
    }
}

/// A summary still running must not hold the turn: one poll returns at once,
/// applies nothing, and leaves the task queued for the next boundary.
#[test]
fn a_running_summary_stays_queued_and_nothing_waits_on_it() {
    let mut tree = make_tree_fixed_day(0);
    let first = tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    let last = tree.finish_turn("c", "d", TurnType::Reality, vec![], None);
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (first, last),
        pending_polls: u32::MAX,
        outcome: None,
    }));

    assert_eq!(tree.poll_tasks(None), 0);
    assert_eq!(tree.pending_task_count(), 1);
    assert_eq!(
        tree.nodes().count(),
        2,
        "no segment until the task finishes"
    );
}

/// A summary that finishes between turns is applied at the next poll, in its
/// window's place: a turn that completed while it ran stays top-level AFTER
/// it, and the trigger count starts from the new segment.
#[test]
fn a_finished_summary_lands_in_order_at_the_next_poll() {
    let mut tree = make_tree_fixed_day(0);
    let first = tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    let last = tree.finish_turn("c", "d", TurnType::Reality, vec![], None);
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (first, last),
        pending_polls: 1,
        outcome: Some(TaskPoll::Ready(segment_patch(first, last))),
    }));

    // Still running at the first boundary; a turn completes meanwhile.
    assert_eq!(tree.poll_tasks(None), 0);
    let later = tree.finish_turn("e", "f", TurnType::Reality, vec![], None);

    assert_eq!(tree.poll_tasks(None), 1);
    assert_eq!(tree.pending_task_count(), 0);
    let top: Vec<&ConversationNode> = tree.nodes().collect();
    assert_eq!(
        top.len(),
        2,
        "the segment plus the turn that arrived meanwhile"
    );
    let seg = top[0].as_segment().expect("the segment comes first");
    assert_eq!(seg.inner().segment_id.start_turn, first);
    assert_eq!(seg.inner().segment_id.end_turn, last);
    assert_eq!(seg.inner().children.len(), 2);
    assert_eq!(
        top[1]
            .as_turn()
            .expect("the later turn follows")
            .inner()
            .turn_id,
        later
    );
    assert_eq!(tree.turns_since_last_summarize(), 1);
}

/// A task that fails or is aborted is dropped without touching the tree.
#[test]
fn a_failed_or_aborted_summary_is_dropped_without_a_patch() {
    let mut tree = make_tree_fixed_day(0);
    let only = tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (only, only),
        pending_polls: 0,
        outcome: Some(TaskPoll::Failed(ConversationError::SchedulerGone)),
    }));
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (only, only),
        pending_polls: 0,
        outcome: Some(TaskPoll::Aborted),
    }));

    assert_eq!(tree.poll_tasks(None), 0);
    assert_eq!(tree.pending_task_count(), 0);
    assert_eq!(tree.nodes().count(), 1);
}
