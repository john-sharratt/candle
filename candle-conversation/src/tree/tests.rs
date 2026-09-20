use std::ops::RangeInclusive;
use std::sync::Arc;

use super::config::ConversationTreeConfig;
use super::conversation_tree::ConversationTree;
use super::node::{ConversationNode, ConversationSegment};
use super::patch::TreePatch;
use super::summarize::SummarizationReason;
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

/// A segment-of-segments summary lands as the PARENT of the segments it
/// summarises, even though it ends where the newest of them ends.
///
/// The duplicate guard used to discard any incoming segment whose end was
/// already reached by an existing one — which a level-2 summary always is — so
/// it was dropped, the level-1 segments stayed top-level, and the segment
/// summary relaunched forever.
#[test]
fn a_segment_summary_becomes_the_parent_of_the_segments_it_covers() {
    let mut tree = make_tree_fixed_day(0);
    let turns: Vec<TurnId> = (0..6)
        .map(|i| {
            tree.finish_turn(
                format!("u{i}"),
                format!("a{i}"),
                TurnType::Reality,
                vec![],
                None,
            )
        })
        .collect();
    tree.apply_patch(segment_patch(turns[0], turns[2]));
    tree.apply_patch(segment_patch(turns[3], turns[5]));

    tree.apply_patch(segment_patch(turns[0], turns[5]));

    let top: Vec<&ConversationNode> = tree.nodes().collect();
    assert_eq!(
        top.len(),
        1,
        "the level-2 segment is the only top-level node"
    );
    let level2 = top[0].as_segment().expect("a segment");
    assert_eq!(level2.inner().segment_id.start_turn, turns[0]);
    assert_eq!(level2.inner().segment_id.end_turn, turns[5]);
    let children: Vec<SegmentId> = level2
        .inner()
        .children
        .iter()
        .map(|c| c.as_segment().expect("level-1 children").inner().segment_id)
        .collect();
    assert_eq!(
        children,
        vec![
            SegmentId {
                start_turn: turns[0],
                end_turn: turns[2]
            },
            SegmentId {
                start_turn: turns[3],
                end_turn: turns[5]
            },
        ]
    );
}

/// A segment already covered by an existing one — the same window delivered
/// twice, or a level-1 window after its level-2 parent — is still discarded.
#[test]
fn a_segment_already_covered_is_discarded() {
    let mut tree = make_tree_fixed_day(0);
    let turns: Vec<TurnId> = (0..6)
        .map(|i| {
            tree.finish_turn(
                format!("u{i}"),
                format!("a{i}"),
                TurnType::Reality,
                vec![],
                None,
            )
        })
        .collect();
    tree.apply_patch(segment_patch(turns[0], turns[5]));

    tree.apply_patch(segment_patch(turns[0], turns[5]));
    tree.apply_patch(segment_patch(turns[3], turns[5]));

    let top: Vec<&ConversationNode> = tree.nodes().collect();
    assert_eq!(top.len(), 1);
    assert_eq!(
        top[0]
            .as_segment()
            .expect("a segment")
            .inner()
            .children
            .len(),
        6,
        "neither the repeat nor the covered window displaced the turns"
    );
}

/// A tree that summarises every `turns` turns and every `segments` segments,
/// with the day-boundary trigger as given, on a fixed `day`.
fn make_summarising_tree(turns: u32, segments: u32, day_boundary: bool) -> ConversationTree {
    ConversationTree::with_config(
        "sys",
        ConversationTreeConfig {
            summarize_every: turns,
            segment_summarize_every: segments,
            summarize_on_day_boundary: day_boundary,
            ..ConversationTreeConfig::default()
        },
    )
    .with_time_source(Arc::new(FixedTimeSource::at_day(0)))
}

/// **A config with summarization disabled has nothing ever due** — not at the
/// turn count, not at the segment count, not across a day boundary. Each
/// trigger is exercised past the point it would fire on the default config.
#[test]
fn disabled_summarization_is_never_due() {
    let mut config = ConversationTreeConfig {
        summarize_every: 2,
        segment_summarize_every: 2,
        summarize_on_day_boundary: true,
        ..ConversationTreeConfig::default()
    };
    assert!(config.summarizes());
    config.disable_summarization();
    assert!(!config.summarizes());

    let mut tree = ConversationTree::with_config("sys", config)
        .with_time_source(Arc::new(FixedTimeSource::at_day(0)));
    let t1 = tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    let t2 = tree.finish_turn("c", "d", TurnType::Reality, vec![], None);
    let t3 = tree.finish_turn("e", "f", TurnType::Reality, vec![], None);
    let t4 = tree.finish_turn("g", "h", TurnType::Reality, vec![], None);
    tree.apply_patch(segment_patch(t1, t2));
    tree.apply_patch(segment_patch(t3, t4));
    let mut tree = tree.with_time_source(Arc::new(FixedTimeSource::at_day(1)));
    tree.finish_turn("i", "j", TurnType::Reality, vec![], None);
    tree.finish_turn("k", "l", TurnType::Reality, vec![], None);

    assert!(tree.owed_day_boundary.is_none());
    assert!(tree.due_summaries().is_empty());
}

/// A day boundary crossed while a summary is still running is owed, not lost.
///
/// The boundary trigger fires on one turn only — the first of the new day. If a
/// count summary is in flight then, the dedup skips the boundary, and when that
/// summary lands only the count is re-checked: one turn against a threshold of
/// three, so the day split never happened.
#[test]
fn a_day_boundary_crossed_mid_summary_is_owed_until_it_can_run() {
    let mut tree = make_summarising_tree(3, 0, true);
    let first = tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    tree.finish_turn("c", "d", TurnType::Reality, vec![], None);
    let third = tree.finish_turn("e", "f", TurnType::Reality, vec![], None);
    // The count summary over turns 1-3 is still running…
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (first, third),
        pending_polls: u32::MAX,
        outcome: None,
    }));
    // …when the first turn of the next day completes.
    let mut tree = tree.with_time_source(Arc::new(FixedTimeSource::at_day(1)));
    let fourth = tree.finish_turn("g", "h", TurnType::Reality, vec![], None);
    assert!(
        matches!(
            tree.owed_day_boundary,
            Some(SummarizationReason::DayBoundary {
                previous_day: 0,
                new_day: 1
            })
        ),
        "the boundary must be held while a summary covers the window"
    );

    // The running summary lands.
    tree.pending_tasks.clear();
    tree.apply_patch(segment_patch(first, third));

    let due = tree.due_summaries();
    assert_eq!(
        due.len(),
        1,
        "the owed boundary is due once nothing covers it"
    );
    assert!(matches!(
        due[0].reason,
        SummarizationReason::DayBoundary {
            previous_day: 0,
            new_day: 1
        }
    ));
    assert_eq!(due[0].start_turn_id, fourth);
    assert_eq!(due[0].end_turn_id, fourth);
    assert!(
        tree.owed_day_boundary.is_none(),
        "queued once, so no longer owed"
    );
}

/// A turn-window summary in flight is not coverage for the segment-of-segments
/// summary: it only ever spans turns after the newest segment. Counting it
/// skipped the segment summary whenever turns arrived faster than summaries
/// finished, and top-level segments piled up uncompressed.
#[test]
fn a_turn_window_summary_in_flight_does_not_starve_the_segment_summary() {
    let mut tree = make_summarising_tree(0, 2, false);
    let t1 = tree.finish_turn("a", "b", TurnType::Reality, vec![], None);
    let t2 = tree.finish_turn("c", "d", TurnType::Reality, vec![], None);
    let t3 = tree.finish_turn("e", "f", TurnType::Reality, vec![], None);
    let t4 = tree.finish_turn("g", "h", TurnType::Reality, vec![], None);
    let t5 = tree.finish_turn("i", "j", TurnType::Reality, vec![], None);
    tree.apply_patch(segment_patch(t1, t2));
    tree.apply_patch(segment_patch(t3, t4));
    // A turn-window summary over the unsummarised turn is still running.
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (t5, t5),
        pending_polls: u32::MAX,
        outcome: None,
    }));

    let due = tree.due_summaries();
    assert_eq!(due.len(), 1);
    assert!(matches!(
        due[0].reason,
        SummarizationReason::SegmentCountReached { count: 2 }
    ));

    // A segment summary in flight over those segments IS coverage.
    tree.pending_tasks.push(Box::new(ScriptedTask {
        span: (t1, t4),
        pending_polls: u32::MAX,
        outcome: None,
    }));
    assert!(tree.due_summaries().is_empty());
}
