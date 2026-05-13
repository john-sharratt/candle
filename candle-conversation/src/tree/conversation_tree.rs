//! [`ConversationTree`] and [`ConversationTreeFork`].
//!
//! # Temporal marker placement
//!
//! Each completed turn has a `[T-{day}.{seq}]` marker injected into the
//! token stream immediately before the model's response begins. The exact
//! injection point differs by thinking mode:
//!
//! - **Thinking off:** marker is inserted after the `<|no_think|>` token,
//!   before generation starts. This is a single token boundary and can be
//!   done as a prefill constant.
//! - **Thinking on (TODO):** the marker must appear *after* the model's
//!   `</think>` close tag so it lands in the visible response, not inside the
//!   reasoning block. This requires detecting `</think>` in the live decode
//!   stream, pausing generation, injecting the marker as a prefill step, then
//!   resuming. Deferred.
//! - **Current fallback for thinking-on paths:** the marker is placed
//!   immediately after `<|im_start|>assistant\n`, before any `<think>` tag.
//!   Imprecise but self-consistent.

use std::collections::LinkedList;
use std::sync::Arc;
use std::time::Duration;

use crossbeam::channel::Sender;

use super::config::ConversationTreeConfig;
use super::node::{
    ConversationNode, ConversationSystemPrompt, ConversationTurn, ConversationTurnInner,
};
use super::patch::{TreeMetadataDelta, TreePatch};
use super::summarize::{
    SummarizationContent, SummarizationReason, SummarizationSegmentEntry, SummarizationSnapshot,
    SummarizationTask, SummarizationTurnEntry,
};
use super::task::CognitiveTask;
use super::token_text::TokenizedText;
use super::types::{NodeId, TurnId, TurnType};
use crate::prompts::TEMPORAL_MARKER_POSTFIX;
use crate::scheduler::SchedulerRequest;
use crate::time_source::{TimeSource, WallClockTimeSource};
use crate::token_buffer::TokenBuffer;

// ────────────────────────────────────────────────────────────────────────────
// ConversationTree
// ────────────────────────────────────────────────────────────────────────────

/// The canonical history of a conversation as an N-ary tree of
/// [`ConversationNode`]s.
///
/// Currently a right-growing chain (every node has at most one child).
/// TODO: multi-child N-ary tree with segment nodes from summarization.
///
/// The system prompt is stored directly as a field, not as a node —
/// it has no turn identity, no children, and is never summarized.
///
/// Two additional text fields sit outside the tree entirely:
/// - [`beliefs`](ConversationTree::beliefs): static declarative propositions
///   injected into every system prompt regardless of turn type.
/// - [`plan`](ConversationTree::plan): mutable Reasoning-produced intention
///   injected into Reality system prompts only.
///
/// All relationships are live: `children` and `decode_context` hold
/// cheaply-clonable [`ConversationNode`] values backed by `Arc`.
pub struct ConversationTree {
    /// System prompt text (with temporal marker postfix already appended if
    /// markers are enabled). Not a node; KV tracked in
    /// `KvCacheLayer::system_prompt_entry`.
    pub(crate) system_prompt: ConversationSystemPrompt,

    /// All conversation nodes: turns and segments in chronological order.
    /// No sentinel or root node. TODO: interleave segments after summarization.
    pub(crate) nodes: LinkedList<ConversationNode>,

    /// Policy configuration — markers, summarization, branching factor,
    /// KV format per tier. Stored by value so forks get a consistent copy.
    pub(crate) config: ConversationTreeConfig,

    /// Pluggable time source driving temporal coordinate computation.
    /// `Arc` so it is shared cheaply between tree and any forks.
    pub(crate) time_source: Arc<dyn TimeSource>,

    /// Monotonically increasing counter for the next turn's `seq`.
    /// Starts at 1; every call to `next_seq_and_increment()` bumps it.
    next_seq: u32,

    /// The `day` component of the most recently completed turn, or `None`
    /// before the first turn.
    last_turn_day: Option<i32>,

    /// Character beliefs — static declarative propositions injected into
    /// every system prompt regardless of turn type.
    ///
    /// Beliefs are not a process, not a tree node, and never change. They are
    /// always present, structuring every cognitive mode from beneath. Examples:
    /// ```text
    /// Trust must be demonstrated through consistency, not declared.
    /// People reveal their true nature under pressure, not in comfort.
    /// I am not someone others stay for without reason.
    /// ```
    ///
    /// `None` until [`with_beliefs`](ConversationTree::with_beliefs) is called.
    pub(crate) beliefs: Option<String>,

    /// The character's current plan — produced by the most recent
    /// [`TurnType::Reason`] turn and injected into Reality system prompts.
    ///
    /// Written in the character's own voice: what they are trying to do, what
    /// they are wary of, what they are waiting for. Replaces the previous plan
    /// each time a new Reason turn completes.
    ///
    /// Unlike `beliefs`, the plan is mutable and injected only into
    /// [`TurnType::Reality`] system prompts. `None` until the first Reason
    /// turn runs.
    pub(crate) plan: Option<String>,

    /// In-flight cognitive task handles queued by `run_summarize()` and
    /// other trigger handlers.
    ///
    /// Each entry is a launched [`SummarizationTask`] (or future task type)
    /// whose background inference is already running on the scheduler's
    /// thread. `Sequence` drains this vec after each `finish_turn()` call
    /// and polls / applies the results.
    ///
    /// `Box<dyn CognitiveTask>` is not `Clone`; the field is deliberately
    /// excluded from the manual `Clone` impl — forks and test trees start
    /// with an empty pending queue.
    pub(crate) pending_tasks: Vec<Box<dyn CognitiveTask>>,

    /// Optional observer channel for streaming cognitive task events
    /// (Token, Prefill, PrefillProgress, HealthWarning) to external
    /// callers in real time. Set via
    /// [`Sequence::set_task_observer`](crate::Sequence::set_task_observer).
    pub(crate) task_event_observer: Option<crossbeam::channel::Sender<crate::handle::TurnEvent>>,

    /// Optional maximum number of turns before the conversation tree begins
    /// culling the oldest, this is used for short conversations that don't need
    /// a full conversation
    max_turns: Option<usize>,
}

impl Clone for ConversationTree {
    /// Produce a shallow clone suitable for forking.
    ///
    /// `pending_tasks` is **not** cloned \u2014 `Box<dyn CognitiveTask>` is not
    /// `Clone`, so forks start with an empty pending queue.
    fn clone(&self) -> Self {
        Self {
            system_prompt: self.system_prompt.clone(),
            nodes: self.nodes.clone(),
            config: self.config.clone(),
            time_source: Arc::clone(&self.time_source),
            next_seq: self.next_seq,
            last_turn_day: self.last_turn_day,
            beliefs: self.beliefs.clone(),
            plan: self.plan.clone(),
            pending_tasks: Vec::new(),
            task_event_observer: self.task_event_observer.clone(),
            max_turns: self.max_turns.clone(),
        }
    }
}

impl ConversationTree {
    // ── Constructors ───────────────────────────────────────────────────

    /// Create a new tree with default config and a wall-clock time source.
    pub fn new(system_prompt_text: impl Into<String>) -> Self {
        Self::with_config(system_prompt_text, ConversationTreeConfig::default())
    }

    /// Create a tree with an explicit config.
    pub fn with_config(
        system_prompt_text: impl Into<String>,
        config: ConversationTreeConfig,
    ) -> Self {
        let mut text = system_prompt_text.into();
        if config.temporal_markers_enabled && !text.is_empty() {
            text.push('\n');
            text.push_str(TEMPORAL_MARKER_POSTFIX);
        }
        Self {
            system_prompt: ConversationSystemPrompt {
                content: TokenizedText::plaintext(text),
            },
            nodes: LinkedList::new(),
            max_turns: config.max_turns.clone(),
            config,
            time_source: Arc::new(WallClockTimeSource::new()),
            next_seq: 1,
            last_turn_day: None,
            beliefs: None,
            plan: None,
            pending_tasks: Vec::new(),
            task_event_observer: None,
        }
    }

    /// Sets the maximum number of turns
    pub fn set_max_turns(&mut self, val: usize) {
        self.max_turns = Some(val);
    }

    /// Builder: enable temporal markers (also appends system prompt postfix).
    ///
    /// Must be called before the first turn — the postfix is already baked
    /// into `system_prompt.text` at construction.
    pub fn with_temporal_markers(mut self) -> Self {
        if !self.config.temporal_markers_enabled {
            self.config.temporal_markers_enabled = true;
            if !self.system_prompt.content.is_empty() {
                let mut text = self.system_prompt.content.text().to_string();
                text.push('\n');
                text.push_str(TEMPORAL_MARKER_POSTFIX);
                self.system_prompt.content = TokenizedText::plaintext(text);
            }
        }
        self
    }

    /// Builder: replace the time source.
    pub fn with_time_source(mut self, ts: Arc<dyn TimeSource>) -> Self {
        self.time_source = ts;
        self
    }

    /// Builder: set the character's beliefs block.
    ///
    /// Beliefs are static — set once at character creation and never changed.
    /// They are injected into every system prompt regardless of turn type.
    pub fn with_beliefs(mut self, beliefs: impl Into<String>) -> Self {
        self.beliefs = Some(beliefs.into());
        self
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// The system prompt text (with any temporal marker postfix).
    pub fn system_prompt_text(&self) -> &str {
        self.system_prompt.content.text()
    }

    /// The system prompt token ids (empty until `set_system_prompt_tokens` is called).
    pub fn system_prompt_token_ids(&self) -> &[u32] {
        self.system_prompt.content.token_ids()
    }

    /// Set the token ids for the system prompt after late tokenization.
    pub fn set_system_prompt_tokens(&mut self, token_ids: TokenBuffer) {
        self.system_prompt.content.set_tokens(token_ids);
    }

    /// Clear all conversation turns, resetting to a fresh state.
    ///
    /// Preserves the system prompt (text + token ids), config, time source,
    /// and beliefs. Useful after [`Sequence::reset()`] to mirror the
    /// cleared KV cache.
    pub(crate) fn clear_turns(&mut self) {
        self.nodes.clear();
        self.next_seq = 1;
        self.last_turn_day = None;
        self.plan = None;
        self.pending_tasks.clear();
    }

    /// The character's beliefs block, or `None` if not set.
    pub fn beliefs(&self) -> Option<&str> {
        self.beliefs.as_deref()
    }

    /// The character's current plan, or `None` if no Reason turn has run yet.
    pub fn plan(&self) -> Option<&str> {
        self.plan.as_deref()
    }

    /// Replace the current plan with the output of a completed Reason turn.
    ///
    /// Pass `None` to clear the plan (e.g. when a Reason turn produces no
    /// actionable intention).
    pub fn set_plan(&mut self, plan: Option<String>) {
        self.plan = plan;
    }

    /// All nodes (turns and segments) in chronological order.
    pub fn nodes(&self) -> impl Iterator<Item = &ConversationNode> {
        self.nodes.iter()
    }

    /// The active configuration.
    pub fn config(&self) -> &ConversationTreeConfig {
        &self.config
    }

    /// Iterator over turn nodes only.
    pub fn turns(&self) -> impl Iterator<Item = &ConversationTurn> {
        self.nodes.iter().filter_map(|n| n.as_turn())
    }

    /// Iterator over segment nodes only.
    pub fn segments(&self) -> impl Iterator<Item = &super::node::ConversationSegment> {
        self.nodes.iter().filter_map(|n| n.as_segment())
    }

    // ── Derived state (computed from nodes vec) ────────────────────────

    /// The `seq` to assign to the *next* turn, and increment the internal
    /// counter.
    pub(crate) fn next_seq_and_increment(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    /// The most recent turn's `day` component, or `None` if no turns yet.
    pub fn last_turn_day(&self) -> Option<i32> {
        self.last_turn_day
    }

    /// The `end_turn.seq` of the most recent segment node, or `None` if no
    /// segments have been inserted yet.
    pub fn last_summarize_seq(&self) -> Option<u32> {
        self.nodes
            .iter()
            .rev()
            .find_map(|n| n.as_segment())
            .map(|s| s.0.segment_id.end_turn.seq)
    }

    /// Number of completed turn-pairs since the last summarization (or since
    /// the start if no summarization has happened yet).
    pub fn turns_since_last_summarize(&self) -> u32 {
        let last_seq = self.nodes.back().map(|n| n.ordering_seq()).unwrap_or(0);
        let last_summarize = self.last_summarize_seq().unwrap_or(0);
        last_seq.saturating_sub(last_summarize)
    }

    // ── Temporal marker helpers ────────────────────────────────────────

    /// Compute the temporal marker string for the next turn.
    ///
    /// Returns `""` when markers are disabled.
    pub fn compute_marker(&self) -> String {
        if !self.config.temporal_markers_enabled {
            return String::new();
        }
        let day = self.time_source.days_since_reference();
        let seq = self.next_seq;
        format!("[T-{day}.{seq}]")
    }

    // ── Turn management ────────────────────────────────────────────────

    /// Append a completed turn to the tree and run post-turn logic.
    ///
    /// Called by the engine after assistant generation finishes. Returns
    /// the `TurnId` assigned to this turn.
    ///
    /// 1. Assigns `(day, seq)` from the time source.
    /// 2. Appends the `ConversationNode::Turn` to `nodes`.
    /// 3. Updates `last_turn_day`.
    /// 4. Checks summarization triggers.
    pub(crate) fn finish_turn(
        &mut self,
        user_text: impl Into<TokenizedText>,
        assistant_text: impl Into<TokenizedText>,
        turn_type: TurnType,
        decode_context: Vec<ConversationNode>,
        inference: Option<(&Sender<SchedulerRequest>, &Arc<tokenizers::Tokenizer>)>,
    ) -> TurnId {
        let day = self.time_source.days_since_reference();
        let seq = self.next_seq_and_increment();
        let turn_id = TurnId { day, seq };

        let inner = ConversationTurnInner {
            turn_id,
            turn_type,
            user: user_text.into(),
            assistant: assistant_text.into(),
            decode_context,
            children: Vec::new(),
        };
        self.nodes
            .push_back(ConversationNode::Turn(ConversationTurn(Arc::new(inner))));

        let previous_day = self.last_turn_day;
        self.last_turn_day = Some(day);

        self.check_and_trigger_summarize(turn_id, previous_day, inference);

        turn_id
    }

    // ── Summarization ──────────────────────────────────────────────────

    fn check_and_trigger_summarize(
        &mut self,
        completed_turn: TurnId,
        previous_day: Option<i32>,
        inference: Option<(&Sender<SchedulerRequest>, &Arc<tokenizers::Tokenizer>)>,
    ) {
        let mut should_summarize = false;
        let mut reason = None;

        if self.config.summarize_every > 0 {
            let count = self.turns_since_last_summarize();
            if count >= self.config.summarize_every {
                should_summarize = true;
                reason = Some(SummarizationReason::TurnCountReached { count });
            }
        }

        if self.config.summarize_on_day_boundary {
            if let Some(prev) = previous_day {
                if completed_turn.day > prev {
                    should_summarize = true;
                    reason = Some(SummarizationReason::DayBoundary {
                        previous_day: prev,
                        new_day: completed_turn.day,
                    });
                }
            }
        }

        if should_summarize {
            // Dedup: skip if any pending task already covers turns in this
            // unsummarized window. `last_summarize_seq()` is the boundary
            // — any task whose relevant range ends beyond it is covering
            // the same window we are about to queue.
            let last_seg_seq = self.last_summarize_seq().unwrap_or(0);
            let already_pending = self.pending_tasks.iter().any(|t| {
                t.relevant_turns()
                    .map_or(false, |r| r.end().seq > last_seg_seq)
            });
            if already_pending {
                tracing::debug!("summarization task already pending for this window — skipping");
                return;
            }
            let snapshot = self.build_snapshot(reason.unwrap());
            self.run_summarize(snapshot, inference);
        }
    }

    fn build_snapshot(&self, reason: SummarizationReason) -> SummarizationSnapshot {
        let window_start = self
            .nodes
            .iter()
            .rposition(|n| n.as_segment().is_some())
            .map(|idx| idx + 1)
            .unwrap_or(0);

        let window = || self.nodes.iter().skip(window_start);

        let start_turn_id = window()
            .find_map(|n| n.as_turn())
            .map(|t| t.0.turn_id)
            .unwrap_or(TurnId { day: 0, seq: 1 });

        let end_turn_id = window()
            .filter_map(|n| n.as_turn())
            .last()
            .map(|t| t.0.turn_id)
            .unwrap_or(start_turn_id);

        let turns: Vec<SummarizationTurnEntry> = window()
            .filter_map(|n| n.as_turn())
            .map(|t| {
                let inner = &t.0;
                let marker = if self.config.temporal_markers_enabled {
                    inner.turn_id.temporal_marker()
                } else {
                    String::new()
                };
                SummarizationTurnEntry {
                    node_id: NodeId::Turn(inner.turn_id),
                    turn_type: inner.turn_type,
                    temporal_marker_text: marker,
                    user_text: inner.user.text().to_string(),
                    assistant_text: inner.assistant.text().to_string(),
                }
            })
            .collect();

        SummarizationSnapshot {
            reason,
            node_range: window_start..(window_start + window().len()),
            start_turn_id,
            end_turn_id,
            elapsed_since_last: Duration::from_secs(0),
            content: SummarizationContent::Turns(turns),
        }
    }

    /// Build a snapshot for a segment-level (recursive) summarization.
    ///
    /// Collects all top-level `Segment` nodes, extracts their summary texts,
    /// and forms a [`SummarizationContent::Segments`] payload. The
    /// `start_turn_id` / `end_turn_id` span the full range covered by the
    /// collected segments.
    fn build_segment_snapshot(&self, reason: SummarizationReason) -> SummarizationSnapshot {
        let segments: Vec<_> = self.nodes.iter().filter_map(|n| n.as_segment()).collect();

        debug_assert!(
            !segments.is_empty(),
            "build_segment_snapshot called with no segments"
        );

        let start_turn_id = segments.first().unwrap().0.segment_id.start_turn;
        let end_turn_id = segments.last().unwrap().0.segment_id.end_turn;

        let entries: Vec<SummarizationSegmentEntry> = segments
            .iter()
            .map(|s| SummarizationSegmentEntry {
                node_id: NodeId::Segment(s.0.segment_id),
                summary_text: s.0.summary_text.text().to_string(),
            })
            .collect();

        SummarizationSnapshot {
            reason,
            node_range: 0..self.nodes.len(), // diagnostic only
            start_turn_id,
            end_turn_id,
            elapsed_since_last: Duration::from_secs(0),
            content: SummarizationContent::Segments(entries),
        }
    }

    /// Launch a summarization task for the given snapshot.
    ///
    /// Performs the synchronous setup described in the design doc:
    /// 1. Tokenize the system prompt and window text.
    /// 2. Send `NewConversation` to the scheduler — one blocking round-trip
    ///    to obtain a `seq_id`.
    /// 3. Send `SubmitTurn` (fire-and-forget); inference starts immediately
    ///    on the scheduler's thread.
    /// 4. Push the resulting [`SummarizationTask`] handle onto
    ///    `pending_tasks`.
    ///
    /// If the inference backend has not been injected (`scheduler_tx` or
    /// `tokenizer` is `None` — as in unit tests and forks), logs at `debug`
    /// and returns without queuing anything.
    fn run_summarize(
        &mut self,
        snapshot: SummarizationSnapshot,
        inference: Option<(&Sender<SchedulerRequest>, &Arc<tokenizers::Tokenizer>)>,
    ) {
        tracing::debug!(
            reason = ?snapshot.reason,
            start = ?snapshot.start_turn_id,
            end = ?snapshot.end_turn_id,
            "summarization triggered"
        );
        let Some((scheduler_tx, tokenizer)) = inference else {
            tracing::debug!("no inference backend — skipping summarization");
            return;
        };
        match SummarizationTask::launch(
            &snapshot,
            &self.config,
            scheduler_tx.clone(),
            tokenizer,
            self.task_event_observer.clone(),
        ) {
            Ok(task) => {
                self.pending_tasks.push(Box::new(task));
            }
            Err(e) => {
                tracing::warn!("summarization launch failed: {}", e);
            }
        }
    }

    /// Drain all pending cognitive task handles.
    ///
    /// Called by `Sequence::finish_turn()` immediately after
    /// `tree.finish_turn()`. The caller polls each handle to completion
    /// (crude blocking) or accumulates them for async polling at the next
    /// turn boundary.
    pub(crate) fn drain_pending_tasks(&mut self) -> Vec<Box<dyn CognitiveTask>> {
        std::mem::take(&mut self.pending_tasks)
    }

    /// Check whether enough top-level `Segment` nodes have accumulated to
    /// trigger a recursive (segment-of-segments) summarization.
    ///
    /// Called by `Sequence::run_task_blocking_inner` immediately after
    /// `apply_patch` inserts a new segment. If the threshold is met a new
    /// [`SummarizationTask`] is pushed onto `pending_tasks` and the
    /// `Sequence`'s drain loop will pick it up automatically.
    pub(crate) fn check_and_trigger_segment_summarize(
        &mut self,
        inference: Option<(
            &crossbeam::channel::Sender<SchedulerRequest>,
            &std::sync::Arc<tokenizers::Tokenizer>,
        )>,
    ) {
        if self.config.segment_summarize_every == 0 {
            return;
        }
        let seg_count = self
            .nodes
            .iter()
            .filter(|n| n.as_segment().is_some())
            .count() as u32;
        if seg_count < self.config.segment_summarize_every {
            return;
        }
        // Dedup: skip if a pending task already covers the max end_turn.seq
        // of the current top-level segments.
        let max_end_seq = self
            .nodes
            .iter()
            .filter_map(|n| n.as_segment())
            .map(|s| s.0.segment_id.end_turn.seq)
            .max()
            .unwrap_or(0);
        let already_pending = self.pending_tasks.iter().any(|t| {
            t.relevant_turns()
                .map_or(false, |r| r.end().seq >= max_end_seq)
        });
        if already_pending {
            tracing::debug!("segment-level summarization task already pending — skipping");
            return;
        }
        let snapshot = self
            .build_segment_snapshot(SummarizationReason::SegmentCountReached { count: seg_count });
        self.run_summarize(snapshot, inference);
    }

    // ── Fork / Patch ───────────────────────────────────────────────────

    /// Branch the tree for background work.
    ///
    /// Returns a `Send`-able fork (shallow clone + result channel) and a
    /// receiver for the completed [`TreePatch`].
    pub fn fork(
        &self,
    ) -> (
        ConversationTreeFork,
        crossbeam::channel::Receiver<TreePatch>,
    ) {
        let (tx, rx) = crossbeam::channel::bounded(1);
        (
            ConversationTreeFork {
                inner: self.clone(),
                result_tx: tx,
            },
            rx,
        )
    }

    /// Apply a [`TreePatch`] from a completed background task.
    ///
    /// For each [`ConversationNode::Segment`] in the patch:
    ///
    /// 1. **Duplicate guard** — if any existing segment already has an
    ///    `end_turn.seq ≥` the incoming segment's, the patch node is silently
    ///    discarded.
    /// 2. **Parent promotion** — the turn nodes whose `seq` falls in
    ///    `[start_turn.seq, end_turn.seq]` are extracted from the flat `nodes`
    ///    vec and stored as the segment's `children`, making the segment the
    ///    true structural parent of the turns it summarises.
    /// 3. The segment (now with children) is appended to `nodes`, replacing
    ///    the extracted turns.
    ///
    /// Non-segment patch nodes are appended as-is (future use).
    ///
    /// Called by `Sequence`'s `run_task_blocking()` helper or the async
    /// `drain_ready_tasks()` helper.
    pub fn apply_patch(&mut self, patch: TreePatch) {
        for node in patch.appended {
            match node {
                ConversationNode::Segment(seg) => {
                    let start_seq = seg.0.segment_id.start_turn.seq;
                    let end_seq = seg.0.segment_id.end_turn.seq;

                    // Idempotency: discard if this window is already covered.
                    let already_covered = self
                        .nodes
                        .iter()
                        .rev()
                        .filter_map(|n| n.as_segment())
                        .any(|s| s.0.segment_id.end_turn.seq >= end_seq);
                    if already_covered {
                        tracing::debug!(
                            end_seq,
                            "discarding duplicate segment patch — range already covered"
                        );
                        continue;
                    }

                    // Extract nodes (Turns or Segments) that fall within the
                    // segment's turn range so the new segment becomes their parent.
                    let mut children: Vec<ConversationNode> = Vec::new();
                    let mut remaining: LinkedList<ConversationNode> = LinkedList::new();

                    let mut nodes = LinkedList::new();
                    std::mem::swap(&mut nodes, &mut self.nodes);

                    for n in nodes {
                        let contained = match &n {
                            ConversationNode::Turn(t) => {
                                let s = t.0.turn_id.seq;
                                s >= start_seq && s <= end_seq
                            }
                            ConversationNode::Segment(s) => {
                                let ss = s.0.segment_id.start_turn.seq;
                                let es = s.0.segment_id.end_turn.seq;
                                ss >= start_seq && es <= end_seq
                            }
                        };
                        if contained {
                            children.push(n);
                        } else {
                            remaining.push_back(n);
                        }
                    }
                    self.nodes = remaining;
                    // Sort children by ordering_seq defensively.
                    children.sort_by_key(|n| n.ordering_seq());

                    let seg_with_children = seg.with_children(children);
                    self.nodes
                        .push_back(ConversationNode::Segment(seg_with_children));
                }
                other => {
                    // Non-segment nodes (future use) — append as-is.
                    self.nodes.push_back(other);
                }
            }
        }
        if let Some(max_turns) = self.max_turns {
            while self.nodes.len() > max_turns {
                self.nodes.pop_front();
            }
        }
        if let Some(meta) = patch.metadata {
            // TODO: act on metadata delta (e.g. reset seq boundaries and
            // summarise counts).
            let _ = meta;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ConversationTreeFork
// ────────────────────────────────────────────────────────────────────────────

/// A cheaply-forked [`ConversationTree`] that is `Send`.
///
/// Pass to a background thread; call [`finish`](ConversationTreeFork::finish)
/// exactly once when background work is complete.
pub struct ConversationTreeFork {
    /// Inner clone of the tree at fork time.
    pub inner: ConversationTree,
    result_tx: crossbeam::channel::Sender<TreePatch>,
}

impl ConversationTreeFork {
    /// Send the completed patch back to the main engine.
    pub fn finish(self, patch: TreePatch) {
        let _ = self.result_tx.send(patch);
    }
}

// Used in metadata delta handling — suppress unused-import warnings until TODO is implemented.
#[allow(dead_code)]
const _: fn() = || {
    let _: TreeMetadataDelta;
};
