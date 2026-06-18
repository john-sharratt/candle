//! The async summariser thread (`docs/infinite_conversations.md` §7).
//!
//! Spawned alongside the persistence thread at engine start.  Mirrors
//! its trigger/tick/shutdown idiom.  Owns the per-timeline AVL
//! structure: drains pending Normal turns, runs §6 probes to seal
//! summary turns, atomic-writes tree metadata + redo-log records, and
//! processes the dirty-node sweep one node per pass.
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────┐
//!   │  loop {                                                       │
//!   │      select! {                                                │
//!   │          tick (250ms) ─► run_pass()                          │
//!   │          trigger      ─► run_pass()                          │
//!   │          shutdown     ─► drain + exit                        │
//!   │      }                                                        │
//!   │  }                                                            │
//!   │                                                                │
//!   │  run_pass(timeline):                                          │
//!   │    1. drain pending: each Normal → §6 probe → new SoT leaf   │
//!   │       → AVL insert → rotation handling (allocate internal    │
//!   │       SoS via §6 probes as needed).                          │
//!   │    2. dirty sweep: pop one dirty SoS → §6 regeneration       │
//!   │       probe → replace turn → clear dirty.                    │
//!   │    3. emit TreeMetadata records for everything that changed. │
//!   └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! Failures:
//! - `ProbeError::Soft` → re-enqueue the failed children, log, keep
//!   running.  Next pass tries again.
//! - `ProbeError::Hard` → log, return from `run_loop`.  Engine
//!   shutdown proceeds normally.

use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam::channel::{self, Receiver, Sender};

use crate::persistence::record::TreeMetadataPayload;
use crate::projection::{Conversation, TimelineId, TurnIndex};
use crate::scheduler::SchedulerRequest;
use crate::substrate::TreeNodeMeta;
use crate::summary_tree::probe::{ProbeError, ProbeRequest, ProbeResponse, ProbeRunner};
use crate::summary_tree::tree::{Node, NodeId, SummaryTree};
use crate::summary_tree::TurnKind;

/// How often the summariser wakes up on its own when no triggers
/// arrive.  Short cadence (250 ms) so backpressure clears quickly
/// once the foreground tempo eases.
pub const SUMMARISER_TICK: Duration = Duration::from_millis(250);

/// Clone-able fire-and-forget trigger for the summariser thread.
/// Held by the scheduler so every assistant-turn seal can wake the
/// summariser without needing the [`SummariserThread`] handle.
#[derive(Clone)]
pub struct SummariserTrigger {
    tx: Sender<()>,
}

impl SummariserTrigger {
    /// Wake the summariser thread now.  No-op when the trigger queue
    /// is full (one wake is as good as several).
    pub fn fire(&self) {
        let _ = self.tx.try_send(());
    }

    /// Test-only no-op trigger.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn noop() -> Self {
        let (tx, _rx) = channel::bounded(1);
        Self { tx }
    }
}

/// Handle to a running summariser thread.  Triggers wake the loop
/// early; [`Self::shutdown`] (or `Drop`) signals the loop, drains a
/// final pass, and joins the thread.
pub struct SummariserThread {
    handle: Mutex<Option<JoinHandle<()>>>,
    trigger_tx: Sender<()>,
    shutdown_tx: Sender<()>,
}

impl SummariserThread {
    /// Spawn the thread.  `runner` is the probe execution backend —
    /// either [`ChannelProbeRunner`] (production, scheduler-backed)
    /// or [`MockProbeRunner`] (tests).
    /// `max_concurrent` is how many probes a single pass submits at once
    /// (their decodes batch in the scheduler's wave loop). Chosen by total
    /// VRAM at engine init — see `summary_probe_concurrency`.
    pub fn spawn(
        conversation: Conversation,
        runner: Arc<dyn ProbeRunner>,
        max_concurrent: usize,
    ) -> Self {
        let (trigger_tx, trigger_rx) = channel::bounded::<()>(1);
        let (shutdown_tx, shutdown_rx) = channel::bounded::<()>(1);

        let max_concurrent = max_concurrent.max(1);
        let handle = std::thread::Builder::new()
            .name("substrate-summariser".into())
            .spawn(move || {
                run_loop(
                    conversation,
                    runner,
                    max_concurrent,
                    trigger_rx,
                    shutdown_rx,
                )
            })
            .expect("failed to spawn substrate-summariser thread");

        Self {
            handle: Mutex::new(Some(handle)),
            trigger_tx,
            shutdown_tx,
        }
    }

    /// Wake the thread now.  No-op if a trigger is already pending or
    /// the thread is mid-pass.
    pub fn trigger(&self) {
        let _ = self.trigger_tx.try_send(());
    }

    pub fn trigger_handle(&self) -> SummariserTrigger {
        SummariserTrigger {
            tx: self.trigger_tx.clone(),
        }
    }

    /// Stop the thread: signal shutdown, wait for the loop to drain,
    /// then join.  Idempotent.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.try_send(());
        if let Some(h) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl Drop for SummariserThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_loop(
    conversation: Conversation,
    runner: Arc<dyn ProbeRunner>,
    max_concurrent: usize,
    trigger_rx: Receiver<()>,
    shutdown_rx: Receiver<()>,
) {
    loop {
        let mut shutting_down = false;
        crossbeam::channel::select! {
            recv(trigger_rx) -> _ => {}
            recv(shutdown_rx) -> _ => { shutting_down = true; }
            default(SUMMARISER_TICK) => {}
        }

        match run_pass(&conversation, runner.as_ref(), max_concurrent) {
            Ok(()) => {}
            // Only a hard error (GPU fault, scheduler shutdown) stops the
            // thread. A soft error means one timeline's pass couldn't
            // complete this round; it's already logged per-timeline in
            // run_pass — keep the loop alive so every other conversation
            // (and the next tick) still gets summarised.
            Err(ProbeError::Hard(msg)) => {
                tracing::warn!(
                    target: "candle_conversation::summariser",
                    "summariser stopped (hard error): {msg}"
                );
                return;
            }
            Err(ProbeError::Soft(msg)) => {
                tracing::warn!(
                    target: "candle_conversation::summariser",
                    "summariser pass soft-failed, continuing: {msg}"
                );
            }
        }

        if shutting_down {
            return;
        }
    }
}

/// One full summariser pass: drain pending turns into the summary tree,
/// then clear at most one dirty bit per timeline.
///
/// Returns `Err(ProbeError::Hard(...))` to abort the run loop.
/// Soft probe failures are logged and the affected turns are re-queued
/// onto the pending queue for the next pass.
pub fn run_pass(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    max_concurrent: usize,
) -> Result<(), ProbeError> {
    let timeline_ids: Vec<TimelineId> = conversation.read().all_timeline_ids().collect();
    for timeline in &timeline_ids {
        let pending = conversation.read().pending_summary_len(*timeline);
        if pending > 0 {
            tracing::debug!(
                target: "candle_conversation::summariser",
                timeline = %timeline,
                pending,
                "run_pass: timeline has pending normal turns to absorb",
            );
        }
    }
    for timeline in timeline_ids {
        // Per-timeline isolation: a soft failure on one timeline (e.g. a
        // corrupt summary tree from an older run hitting an AVL invariant)
        // must not abort the whole pass — it would starve every other
        // timeline, including freshly-started conversations. Log it and move
        // on; only a hard error propagates and stops the thread.
        if let Err(e) = absorb_pending_turns(conversation, runner, timeline, max_concurrent) {
            match e {
                ProbeError::Hard(_) => return Err(e),
                ProbeError::Soft(msg) => {
                    tracing::warn!(
                        target: "candle_conversation::summariser",
                        timeline = %timeline,
                        "absorb soft-failed, skipping timeline this pass: {msg}"
                    );
                    continue;
                }
            }
        }
        sweep_one_dirty(conversation, timeline);
    }
    Ok(())
}

/// Drain every pending Normal turn for `timeline` into the summary
/// tree.  For each pending Normal:
///
/// 1. Run a §6 probe over `[normal_idx]` to seal a fresh
///    `SummaryOfTurns` leaf turn.
/// 2. Write the leaf's `TreeNodeMeta` (kind, children = [normal_idx],
///    height = 1, dirty = false).
/// 3. AVL-insert the new leaf, allocating any synthesised
///    `SummaryOfSummaries` internals via additional probes.
/// 4. Update `tree_root`, mark rotated ancestors dirty, persist all
///    changes as `TreeMetadata` redo-log records.
fn absorb_pending_turns(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
    max_concurrent: usize,
) -> Result<(), ProbeError> {
    loop {
        // Collect up to `max_concurrent` eligible pending Normal turns, so
        // their SoT probes can be submitted together and their decodes batch
        // in the scheduler's wave loop instead of running one at a time.
        let mut batch: Vec<TurnIndex> = Vec::with_capacity(max_concurrent);
        while batch.len() < max_concurrent {
            let normal_idx = match conversation.write().pop_pending_summary(timeline) {
                Some(idx) => idx,
                None => break,
            };
            // Skip if a previous pass already absorbed this index (e.g. via
            // restart-reload's re-enqueue + a partially-applied tree).
            let already_in_tree = conversation
                .read()
                .tree_meta_of(timeline, normal_idx)
                .map(|m| m.kind != TurnKind::Normal)
                .unwrap_or(false);
            if already_in_tree {
                tracing::debug!(
                    target: "candle_conversation::summariser",
                    timeline = %timeline,
                    normal = %normal_idx,
                    "absorb: turn already in tree, skipping",
                );
                continue;
            }
            batch.push(normal_idx);
        }
        if batch.is_empty() {
            return Ok(());
        }
        tracing::debug!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            batch = batch.len(),
            "absorb: probing SoT leaves for pending normal turns",
        );

        let requests: Vec<ProbeRequest> = batch
            .iter()
            .map(|&normal_idx| ProbeRequest {
                timeline,
                kind: TurnKind::SummaryOfTurns,
                children: vec![normal_idx],
            })
            .collect();
        let results = runner.run_batch(requests);

        // Insert the sealed leaves into the AVL tree.  The probe decodes ran
        // concurrently; the tree mutation (which may emit serial SoS-allocation
        // probes) is applied one leaf at a time, in batch order — ascending
        // chronological order among the turns that succeeded.
        //
        // A soft-failed probe re-enqueues only that turn, to the *back* of the
        // pending queue so a persistently-failing turn can't head-of-line block
        // newer ones.  We still insert the later successes in this pass rather
        // than deferring them: their summary turns are already sealed in the
        // substrate, so deferring would leave them with default `Normal` tree
        // meta — orphans that pollute projections and get re-summarised.  The
        // re-enqueued turn is retried on a later pass and inserted then, which
        // can place its leaf after a younger sibling; that bounded local disorder
        // is the accepted cost of liveness + no orphans.
        let mut soft_failed = false;
        for (&normal_idx, result) in batch.iter().zip(results) {
            match result {
                Ok(sealed) => {
                    seal_leaf_and_avl_insert(
                        conversation,
                        runner,
                        timeline,
                        sealed,
                        vec![normal_idx],
                    )?;
                }
                Err(ProbeError::Soft(msg)) => {
                    tracing::warn!(
                        target: "candle_conversation::summariser",
                        timeline = %timeline,
                        normal = %normal_idx,
                        "soft probe error: {msg}; re-enqueueing"
                    );
                    conversation
                        .write()
                        .push_pending_summary(timeline, normal_idx);
                    soft_failed = true;
                }
                Err(e @ ProbeError::Hard(_)) => return Err(e),
            }
        }
        // Don't burn the queue in a tight loop on persistent soft failures —
        // the next tick / trigger retries the re-enqueued turns.
        if soft_failed {
            return Ok(());
        }
    }
}

/// Attach `TreeNodeMeta(kind=SoT)` to `sealed.sealed_turn`, then
/// AVL-insert it as the rightmost binary leaf for `timeline`.  Emits
/// `TreeMetadata` redo-log records for every node that changed.
fn seal_leaf_and_avl_insert(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
    sealed: ProbeResponse,
    normal_children: Vec<TurnIndex>,
) -> Result<(), ProbeError> {
    // Step 1 — write the leaf's meta and the children's meta (still
    // Normal, but the dirty bit is unaffected; defaults are correct).
    let leaf_idx = sealed.sealed_turn;
    {
        let mut view = conversation.write();
        view.set_tree_meta(
            timeline,
            leaf_idx,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfTurns,
                children: normal_children.clone(),
                tree_height: 1,
                dirty: false,
            },
        );
    }
    // Step 2 — rebuild the in-memory tree and AVL-insert the new
    // leaf, using a probe-backed internal allocator for any new SoS
    // ancestors.
    let avl_result = perform_avl_insert_rightmost(conversation, runner, timeline, leaf_idx)?;

    // Step 3 — persist everything that changed.  TreeMetadata record
    // per touched node; root marker on the eventual root.
    let touched = avl_result.touched_nodes;
    let new_root = avl_result.new_root;
    for idx in &touched {
        let meta = match conversation.read().tree_meta_of(timeline, *idx).cloned() {
            Some(m) => m,
            None => continue,
        };
        let payload = TreeMetadataPayload {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            kind: match meta.kind {
                TurnKind::Normal => 0,
                TurnKind::SummaryOfTurns => 1,
                TurnKind::SummaryOfSummaries => 2,
            },
            tree_height: meta.tree_height,
            dirty: meta.dirty,
            children: meta.children.iter().map(|c| c.0).collect(),
            root_now: if Some(*idx) == new_root {
                new_root.map(|r| r.0)
            } else {
                None
            },
        };
        if let Err(e) = conversation.write_tree_metadata(payload) {
            tracing::warn!(
                target: "candle_conversation::summariser",
                "write_tree_metadata failed for {idx}: {e}"
            );
        }
    }
    Ok(())
}

/// Result of one AVL insertion: which substrate turn indices had
/// their tree metadata mutated, and what the (possibly new) root is.
struct AvlInsertOutcome {
    touched_nodes: Vec<TurnIndex>,
    new_root: Option<TurnIndex>,
}

/// AVL-insert `new_leaf` on the right edge of the timeline's current
/// summary tree.  Allocates new `SummaryOfSummaries` internals via
/// `runner` probes.  Writes the resulting tree state back into the
/// substrate (children + height per touched node, root pointer).
fn perform_avl_insert_rightmost(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
    new_leaf: TurnIndex,
) -> Result<AvlInsertOutcome, ProbeError> {
    // Build the in-memory tree from the substrate's persisted state.
    let mut tree: SummaryTree = conversation.read().build_summary_tree_in_memory(timeline);
    // Inject the new leaf into the in-memory tree.  The substrate
    // already holds its meta (set by `seal_leaf_and_avl_insert`).
    let leaf_meta = conversation
        .read()
        .tree_meta_of(timeline, new_leaf)
        .cloned();
    let leaf_meta = match leaf_meta {
        Some(m) => m,
        None => {
            return Err(ProbeError::Soft(format!(
                "new leaf {new_leaf} missing tree meta after seal"
            )))
        }
    };
    let leaf_tokens = conversation.read().turn_token_count_of(timeline, new_leaf) as u32;
    let leaf_node = Node {
        id: NodeId(new_leaf.0),
        kind: leaf_meta.kind,
        children: leaf_meta.children.iter().map(|c| NodeId(c.0)).collect(),
        tree_height: leaf_meta.tree_height,
        dirty: leaf_meta.dirty,
        tokens: leaf_tokens,
    };

    let mut touched: Vec<TurnIndex> = vec![new_leaf];

    // Empty-or-rootless tree → new leaf becomes root.  The "rootless
    // but non-empty" case arises during seal_leaf_and_avl_insert when
    // the caller has already written the new leaf's TreeNodeMeta to
    // the substrate (so `build_summary_tree_in_memory` will include
    // that node) but the tree's root pointer is still `None`.  Both
    // states resolve identically: install the leaf as root.
    if tree.root().is_none() {
        tree.insert_node(leaf_node);
        if tree
            .chrono_leaves()
            .iter()
            .all(|n| *n != NodeId(new_leaf.0))
        {
            tree.push_chrono_leaf(NodeId(new_leaf.0));
        }
        tree.set_root(Some(NodeId(new_leaf.0)));
        commit_tree_to_substrate(conversation, timeline, &tree, &mut touched);
        return Ok(AvlInsertOutcome {
            touched_nodes: touched,
            new_root: Some(new_leaf),
        });
    }

    // Non-empty: AVL-insert the new leaf, supplying a probe-backed
    // allocator for new internals.  Each allocation produces a real
    // substrate turn via a SummaryOfSummaries probe over the two
    // sub-children.
    //
    // We implement the AVL descent locally here (mirroring
    // SummaryTree::avl_insert_rightmost) because the trait-style
    // allocator would need to call back into the substrate (which we
    // already have through `conversation`).  Keeps the closure
    // simple.
    // Closure that runs a SoS probe over two children, writes the
    // resulting TreeNodeMeta into the substrate, and returns the new
    // TurnIndex.  Does NOT touch the `touched` vec — the recursive
    // walker handles that bookkeeping after each call.
    let mut alloc_internal =
        |left_child: TurnIndex, right_child: TurnIndex| -> Result<TurnIndex, ProbeError> {
            let probe = ProbeRequest {
                timeline,
                kind: TurnKind::SummaryOfSummaries,
                children: vec![left_child, right_child],
            };
            let resp = runner.run(probe)?;
            let internal_idx = resp.sealed_turn;
            let lh = conversation
                .read()
                .tree_meta_of(timeline, left_child)
                .map(|m| m.tree_height)
                .unwrap_or(0);
            let rh = conversation
                .read()
                .tree_meta_of(timeline, right_child)
                .map(|m| m.tree_height)
                .unwrap_or(0);
            let height = lh.max(rh) + 1;
            conversation.write().set_tree_meta(
                timeline,
                internal_idx,
                TreeNodeMeta {
                    kind: TurnKind::SummaryOfSummaries,
                    children: vec![left_child, right_child],
                    tree_height: height,
                    dirty: false,
                },
            );
            Ok(internal_idx)
        };

    // Insert.  This walks the rightmost spine and lifts the rightmost
    // current leaf + new_leaf into a fresh SoS parent (allocated via
    // the closure).  Any AVL rebalancing along the way also creates
    // new SoS internals.
    let current_root = tree
        .root()
        .ok_or_else(|| ProbeError::Soft("tree.root() unexpectedly None".into()))?;
    tree.insert_node(leaf_node);
    tree.push_chrono_leaf(NodeId(new_leaf.0));
    let new_root_node = recursive_avl_insert(
        &mut tree,
        current_root,
        NodeId(new_leaf.0),
        &mut alloc_internal,
        &mut touched,
        timeline,
        conversation,
    )?;
    tree.set_root(Some(new_root_node));

    // Mark any node whose children pointer changed as dirty.  The
    // descent updated their tree_height inline; flag them now.
    let new_root_idx = TurnIndex(new_root_node.0);
    commit_tree_to_substrate(conversation, timeline, &tree, &mut touched);
    Ok(AvlInsertOutcome {
        touched_nodes: touched,
        new_root: Some(new_root_idx),
    })
}

/// Recursive AVL insert with an external allocator for new internal
/// nodes.  Mirrors [`SummaryTree::avl_insert_rightmost`]
/// but takes a `FnMut` so internal-node identities come from real
/// substrate turns rather than the default `fresh_internal_id`
/// auto-allocator.
fn recursive_avl_insert<F>(
    tree: &mut SummaryTree,
    subtree: NodeId,
    new_leaf: NodeId,
    alloc: &mut F,
    touched: &mut Vec<TurnIndex>,
    _timeline: TimelineId,
    _conv: &Conversation,
) -> Result<NodeId, ProbeError>
where
    F: FnMut(TurnIndex, TurnIndex) -> Result<TurnIndex, ProbeError>,
{
    let subtree_kind = tree
        .get(subtree)
        .map(|n| n.kind)
        .ok_or_else(|| ProbeError::Soft(format!("avl: subtree {subtree} missing")))?;
    match subtree_kind {
        TurnKind::SummaryOfTurns => {
            // Reached a binary leaf — allocate a fresh SoS parent over
            // (subtree, new_leaf) via the probe-backed allocator.
            let parent_idx = alloc(TurnIndex(subtree.0), TurnIndex(new_leaf.0))?;
            touched.push(parent_idx);
            let parent_id = NodeId(parent_idx.0);
            // Manually install the node into the tree (the substrate
            // already has its meta).
            let lh = tree.get(subtree).map(|n| n.tree_height).unwrap_or(0);
            let rh = tree.get(new_leaf).map(|n| n.tree_height).unwrap_or(0);
            let height = lh.max(rh) + 1;
            tree.insert_node(Node::summary_of_summaries(
                parent_id, subtree, new_leaf, height, 20,
            ));
            Ok(parent_id)
        }
        TurnKind::SummaryOfSummaries => {
            let right_child = tree.get(subtree).unwrap().children[1];
            let left_child = tree.get(subtree).unwrap().children[0];
            let new_right = recursive_avl_insert(
                tree,
                right_child,
                new_leaf,
                alloc,
                touched,
                _timeline,
                _conv,
            )?;
            // Replace right child pointer via insert_node.
            if new_right != right_child {
                let lh = tree.get(left_child).map(|n| n.tree_height).unwrap_or(0);
                let rh = tree.get(new_right).map(|n| n.tree_height).unwrap_or(0);
                let height = lh.max(rh) + 1;
                tree.insert_node(Node::summary_of_summaries(
                    subtree, left_child, new_right, height, 20,
                ));
                touched.push(TurnIndex(subtree.0));
            }
            // Rebalance: detect imbalance and apply the standard
            // four-case rotation.  We allocate a fresh internal for
            // any node whose children pointer changes, mirroring
            // `SummaryTree::rebalance`.
            rebalance_at(tree, subtree, alloc, touched)
        }
        TurnKind::Normal => Err(ProbeError::Soft(format!(
            "avl: descended into Normal turn {subtree}; tree shape invariant violated"
        ))),
    }
}

fn rebalance_at<F>(
    tree: &mut SummaryTree,
    node_id: NodeId,
    alloc: &mut F,
    touched: &mut Vec<TurnIndex>,
) -> Result<NodeId, ProbeError>
where
    F: FnMut(TurnIndex, TurnIndex) -> Result<TurnIndex, ProbeError>,
{
    use TurnKind;
    let node = tree
        .get(node_id)
        .ok_or_else(|| ProbeError::Soft(format!("rebalance: missing {node_id}")))?;
    if node.kind != TurnKind::SummaryOfSummaries {
        return Ok(node_id);
    }
    let left = node.children[0];
    let right = node.children[1];
    let lh = tree.get(left).map(|n| n.tree_height as i16).unwrap_or(0);
    let rh = tree.get(right).map(|n| n.tree_height as i16).unwrap_or(0);
    let balance = lh - rh;
    if balance.abs() <= 1 {
        return Ok(node_id);
    }
    // Right-heavy: RR or RL.
    if balance < -1 {
        // RL: right child is left-heavy → rotate its subtree right
        // first.  We allocate a fresh internal for the rotation.
        let r_node = tree.get(right).unwrap();
        let rl_h = tree
            .get(r_node.children[0])
            .map(|n| n.tree_height as i16)
            .unwrap_or(0);
        let rr_h = tree
            .get(r_node.children[1])
            .map(|n| n.tree_height as i16)
            .unwrap_or(0);
        let new_right = if rl_h > rr_h {
            rotate_right_via_alloc(tree, right, alloc, touched)?
        } else {
            right
        };
        if new_right != right {
            // Reattach to node_id (re-allocate parent? No — we just
            // overwrite this node's children list.  This node stays
            // the same TurnIndex, but its content (kind=SoS,
            // children=[left, new_right]) is now stale → mark dirty
            // and re-write its tree_meta on commit.
            let lh2 = tree.get(left).map(|n| n.tree_height).unwrap_or(0);
            let rh2 = tree.get(new_right).map(|n| n.tree_height).unwrap_or(0);
            let h = lh2.max(rh2) + 1;
            tree.insert_node(Node::summary_of_summaries(node_id, left, new_right, h, 20));
            touched.push(TurnIndex(node_id.0));
        }
        rotate_left_via_alloc(tree, node_id, alloc, touched)
    } else {
        // Left-heavy: LL or LR.  Mirror of above.
        let l_node = tree.get(left).unwrap();
        let ll_h = tree
            .get(l_node.children[0])
            .map(|n| n.tree_height as i16)
            .unwrap_or(0);
        let lr_h = tree
            .get(l_node.children[1])
            .map(|n| n.tree_height as i16)
            .unwrap_or(0);
        let new_left = if lr_h > ll_h {
            rotate_left_via_alloc(tree, left, alloc, touched)?
        } else {
            left
        };
        if new_left != left {
            let lh2 = tree.get(new_left).map(|n| n.tree_height).unwrap_or(0);
            let rh2 = tree.get(right).map(|n| n.tree_height).unwrap_or(0);
            let h = lh2.max(rh2) + 1;
            tree.insert_node(Node::summary_of_summaries(node_id, new_left, right, h, 20));
            touched.push(TurnIndex(node_id.0));
        }
        rotate_right_via_alloc(tree, node_id, alloc, touched)
    }
}

/// Left rotation at `a` — same shape as
/// [`SummaryTree::rotate_left`], except we mutate
/// the existing nodes' children/height in place rather than allocating.
/// The rotation re-purposes existing TurnIndices (which already exist
/// as substrate turns); their tree_meta gets rewritten to reflect new
/// children.  The new dirty bit is set by the commit step.
fn rotate_left_via_alloc<F>(
    tree: &mut SummaryTree,
    a: NodeId,
    _alloc: &mut F,
    touched: &mut Vec<TurnIndex>,
) -> Result<NodeId, ProbeError>
where
    F: FnMut(TurnIndex, TurnIndex) -> Result<TurnIndex, ProbeError>,
{
    let (x, b) = {
        let node = tree
            .get(a)
            .ok_or_else(|| ProbeError::Soft(format!("rotate_left: a={a} missing")))?;
        (node.children[0], node.children[1])
    };
    let (y, z) = {
        let bn = tree
            .get(b)
            .ok_or_else(|| ProbeError::Soft(format!("rotate_left: b={b} missing")))?;
        (bn.children[0], bn.children[1])
    };
    // a takes (x, y); b takes (a, z).  Heights refresh from children.
    let xh = tree.get(x).map(|n| n.tree_height).unwrap_or(0);
    let yh = tree.get(y).map(|n| n.tree_height).unwrap_or(0);
    let ah = xh.max(yh) + 1;
    tree.insert_node(Node::summary_of_summaries(a, x, y, ah, 20));
    let zh = tree.get(z).map(|n| n.tree_height).unwrap_or(0);
    let bh = ah.max(zh) + 1;
    tree.insert_node(Node::summary_of_summaries(b, a, z, bh, 20));
    touched.push(TurnIndex(a.0));
    touched.push(TurnIndex(b.0));
    Ok(b)
}

fn rotate_right_via_alloc<F>(
    tree: &mut SummaryTree,
    a: NodeId,
    _alloc: &mut F,
    touched: &mut Vec<TurnIndex>,
) -> Result<NodeId, ProbeError>
where
    F: FnMut(TurnIndex, TurnIndex) -> Result<TurnIndex, ProbeError>,
{
    let (b, z) = {
        let node = tree
            .get(a)
            .ok_or_else(|| ProbeError::Soft(format!("rotate_right: a={a} missing")))?;
        (node.children[0], node.children[1])
    };
    let (x, y) = {
        let bn = tree
            .get(b)
            .ok_or_else(|| ProbeError::Soft(format!("rotate_right: b={b} missing")))?;
        (bn.children[0], bn.children[1])
    };
    let yh = tree.get(y).map(|n| n.tree_height).unwrap_or(0);
    let zh = tree.get(z).map(|n| n.tree_height).unwrap_or(0);
    let ah = yh.max(zh) + 1;
    tree.insert_node(Node::summary_of_summaries(a, y, z, ah, 20));
    let xh = tree.get(x).map(|n| n.tree_height).unwrap_or(0);
    let bh = xh.max(ah) + 1;
    tree.insert_node(Node::summary_of_summaries(b, x, a, bh, 20));
    touched.push(TurnIndex(a.0));
    touched.push(TurnIndex(b.0));
    Ok(b)
}

/// Copy every node in `tree` whose id appears in `touched` back into
/// the substrate's `tree_meta` map.  Updates `tree_root` from
/// `tree.root()`.  Marks summary nodes whose children changed dirty
/// IF their `dirty` flag was already true in-memory.
fn commit_tree_to_substrate(
    conversation: &Conversation,
    timeline: TimelineId,
    tree: &SummaryTree,
    touched: &mut Vec<TurnIndex>,
) {
    use TurnKind;
    let mut view = conversation.write();
    touched.sort_by_key(|t| t.0);
    touched.dedup();
    for idx in touched.iter() {
        let node = match tree.get(NodeId(idx.0)) {
            Some(n) => n,
            None => continue,
        };
        let prev_dirty = view
            .tree_meta_of(timeline, *idx)
            .map(|m| m.dirty)
            .unwrap_or(false);
        let new_meta = TreeNodeMeta {
            kind: node.kind,
            children: node.children.iter().map(|c| TurnIndex(c.0)).collect(),
            tree_height: node.tree_height,
            // A rotation may have rewritten children — mark dirty so
            // the next sweep regenerates the summary content.  But
            // don't unset dirty if it was already set.
            dirty: prev_dirty
                || (node.kind == TurnKind::SummaryOfSummaries && {
                    // Compare against previously-recorded children to
                    // tell if this rotation changed them.
                    let old_children = view
                        .tree_meta_of(timeline, *idx)
                        .map(|m| m.children.clone())
                        .unwrap_or_default();
                    let new_children: Vec<TurnIndex> =
                        node.children.iter().map(|c| TurnIndex(c.0)).collect();
                    old_children != new_children
                }),
        };
        view.set_tree_meta(timeline, *idx, new_meta);
    }
    view.set_tree_root(timeline, tree.root().map(|r| TurnIndex(r.0)));
}

/// Clear the oldest dirty node's dirty bit. At most one per pass.
///
/// A `SummaryOfSummaries` node is marked dirty when an AVL rotation rewrites
/// its children, so its summary content is now a summary of a shifted span.
/// Clearing the bit — rather than regenerating the content — is deliberate:
/// re-probing would mint a fresh summary turn with no way to re-link it into
/// the tree, leaving an orphan that the model-backed probe path drops onto the
/// pending queue as a `Normal` turn and re-summarises, cascading into an
/// unbounded loop of summary-of-summary turns. The SoS node keeps its
/// (slightly stale but valid) summary of the shifted span, which is correct
/// for retrieval, and no orphan turns are created.
fn sweep_one_dirty(conversation: &Conversation, timeline: TimelineId) {
    if let Some(dirty_idx) = conversation.write().pop_oldest_dirty(timeline) {
        conversation
            .write()
            .clear_summary_dirty(timeline, dirty_idx);
    }
}

// ── ChannelProbeRunner ───────────────────────────────────────────────

/// Production-quality `ProbeRunner` backed by a `SchedulerRequest`
/// channel.  Sends [`SubmitSummaryProbe`](SchedulerRequest::SubmitSummaryProbe)
/// to the scheduler and blocks for the response.
///
/// Construction is `pub(crate)` because `SchedulerRequest` is a
/// crate-internal type — the engine wires this up at startup; callers
/// don't construct it directly.
pub struct ChannelProbeRunner {
    request_tx: Sender<SchedulerRequest>,
}

impl ChannelProbeRunner {
    pub(crate) fn new(request_tx: Sender<SchedulerRequest>) -> Self {
        Self { request_tx }
    }
}

impl ProbeRunner for ChannelProbeRunner {
    fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        let scheduler_request = SchedulerRequest::SubmitSummaryProbe {
            timeline: request.timeline,
            kind: request.kind,
            children: request.children.clone(),
            response_tx,
        };
        if let Err(e) = self.request_tx.send(scheduler_request) {
            return Err(ProbeError::Hard(format!("scheduler channel closed: {e}")));
        }
        match response_rx.recv() {
            Ok(Ok(turn_idx)) => Ok(ProbeResponse {
                sealed_turn: turn_idx,
            }),
            Ok(Err(msg)) => Err(ProbeError::Soft(msg)),
            Err(e) => Err(ProbeError::Hard(format!("scheduler response channel: {e}"))),
        }
    }

    fn run_batch(&self, requests: Vec<ProbeRequest>) -> Vec<Result<ProbeResponse, ProbeError>> {
        // Submit every probe first (non-blocking sends) so the scheduler
        // registers all their decodes before the next decode quantum — they
        // then batch into a single forward instead of running one at a time.
        let mut receivers: Vec<Result<Receiver<Result<TurnIndex, String>>, ProbeError>> =
            Vec::with_capacity(requests.len());
        for request in requests {
            let (response_tx, response_rx) = crossbeam::channel::bounded(1);
            let scheduler_request = SchedulerRequest::SubmitSummaryProbe {
                timeline: request.timeline,
                kind: request.kind,
                children: request.children.clone(),
                response_tx,
            };
            match self.request_tx.send(scheduler_request) {
                Ok(()) => receivers.push(Ok(response_rx)),
                Err(e) => receivers.push(Err(ProbeError::Hard(format!(
                    "scheduler channel closed: {e}"
                )))),
            }
        }
        // Then collect, in submission order. Each `recv` blocks only until that
        // probe's batched decode completes.
        receivers
            .into_iter()
            .map(|r| match r {
                Ok(rx) => match rx.recv() {
                    Ok(Ok(turn_idx)) => Ok(ProbeResponse {
                        sealed_turn: turn_idx,
                    }),
                    Ok(Err(msg)) => Err(ProbeError::Soft(msg)),
                    Err(e) => Err(ProbeError::Hard(format!("scheduler response channel: {e}"))),
                },
                Err(e) => Err(e),
            })
            .collect()
    }
}

// ── MockProbeRunner ──────────────────────────────────────────────────

/// Test-only `ProbeRunner` that appends a placeholder summary turn
/// directly to the substrate without invoking the model.  Returns the
/// new TurnIndex via [`crate::substrate::Substrate::append_with_blocks`].
///
/// Doesn't write content; the test asserts on tree structure, not on
/// the summary text.
pub struct MockProbeRunner {
    conversation: Conversation,
    summary_tokens: usize,
}

impl MockProbeRunner {
    pub fn new(conversation: Conversation) -> Self {
        Self {
            conversation,
            summary_tokens: 20,
        }
    }

    pub fn with_summary_tokens(mut self, n: usize) -> Self {
        self.summary_tokens = n;
        self
    }
}

impl ProbeRunner for MockProbeRunner {
    fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
        // Append a placeholder turn into the substrate at the request's
        // timeline.  Block range is 0..0 (no KV chunks); residence
        // stays cold — fine for unit tests that only exercise tree
        // bookkeeping.
        //
        // `append_with_blocks` pushes the new turn onto the pending
        // queue (its default classification is `Normal` until
        // `set_tree_meta` overwrites it).  That's fine: the summariser
        // loop will pop it back later, then short-circuit on the
        // `already_in_tree` check (the seal_leaf path has by then set
        // its kind to `SummaryOfTurns` / `SummaryOfSummaries`).  Don't
        // pop here — the queue is FIFO and popping would steal the
        // wrong entry.
        let idx = self.conversation.write().append_with_blocks(
            request.timeline,
            self.summary_tokens,
            0,
            0,
        );
        Ok(ProbeResponse { sealed_turn: idx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{Conversation, TimelineId, TurnIndex};
    use TurnKind;

    fn ephemeral_workspace() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn fresh_conversation(_workspace: &std::path::Path) -> (Conversation, TimelineId) {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let conversation = Conversation::ephemeral();
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        conversation
            .write()
            .register_timeline(timeline, layer, group);
        (conversation, timeline)
    }

    #[test]
    fn mock_runner_produces_a_fresh_turn_index() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let normal = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let _ = conv.write().pop_pending_summary(timeline); // drain
        let runner = MockProbeRunner::new(conv.clone());
        let resp = runner
            .run(ProbeRequest {
                timeline,
                kind: TurnKind::SummaryOfTurns,
                children: vec![normal],
            })
            .unwrap();
        assert_eq!(resp.sealed_turn.0, 1);
    }

    /// Test runner that soft-fails the `SummaryOfTurns` probe whose first
    /// child is `fail_on`, delegating every other probe to a real
    /// [`MockProbeRunner`].  Used to exercise a mid-batch soft failure.
    struct FailOnChildRunner {
        inner: MockProbeRunner,
        fail_on: TurnIndex,
    }

    impl ProbeRunner for FailOnChildRunner {
        fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
            if request.kind == TurnKind::SummaryOfTurns
                && request.children.first() == Some(&self.fail_on)
            {
                return Err(ProbeError::Soft("injected mid-batch failure".into()));
            }
            self.inner.run(request)
        }
    }

    /// A soft failure on the *middle* turn of a batch must not orphan the
    /// later successes.  Their summary turns are already sealed, so they have
    /// to be AVL-inserted (kind = SoT) this pass; deferring them would leave
    /// them with default `Normal` meta — stray turns that pollute projections
    /// and get re-summarised.  Only the failed turn is re-enqueued.
    #[test]
    fn absorb_mid_batch_soft_failure_does_not_orphan_successes() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let n1 = conv.write().append_with_blocks(timeline, 10, 1, 2);
        let n2 = conv.write().append_with_blocks(timeline, 10, 2, 3);
        assert_eq!(conv.pending_summary_len(timeline), 3);

        let runner = FailOnChildRunner {
            inner: MockProbeRunner::new(conv.clone()),
            fail_on: n1,
        };
        absorb_pending_turns(&conv, &runner, timeline, 4).expect("absorb ok");

        // Every SoT leaf's children, across the whole tree.
        let mut summarised: Vec<TurnIndex> = Vec::new();
        for i in 0..16u32 {
            if let Some(meta) = conv.read().tree_meta_of(timeline, TurnIndex(i)) {
                if meta.kind == TurnKind::SummaryOfTurns {
                    summarised.extend(meta.children.iter().copied());
                }
            }
        }
        assert!(
            summarised.contains(&n0),
            "n0 succeeded → must have a SoT leaf"
        );
        assert!(
            summarised.contains(&n2),
            "n2 succeeded → must have a SoT leaf"
        );
        assert!(
            !summarised.contains(&n1),
            "n1 failed → must not appear in any leaf"
        );

        // The failed turn is back on the pending queue for a later pass.
        let mut requeued_n1 = false;
        while let Some(idx) = conv.write().pop_pending_summary(timeline) {
            if idx == n1 {
                requeued_n1 = true;
            }
        }
        assert!(requeued_n1, "failed turn n1 must be re-enqueued");
    }

    /// A timeline with `summarize = false` — the gate set for utility/reference
    /// layers (repo_map, code_reading) — must not enqueue its sealed turns onto
    /// the pending-summary queue, so the summariser never touches them. A
    /// timeline left at the default (`true`) does enqueue.
    #[test]
    fn summarize_gate_off_skips_pending_enqueue() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        conv.write().set_timeline_summarize(timeline, false);
        let _n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        assert_eq!(
            conv.pending_summary_len(timeline),
            0,
            "summarize=false must not enqueue pending summaries"
        );

        let (conv2, timeline2) = fresh_conversation(tmp.path());
        let _m0 = conv2.write().append_with_blocks(timeline2, 10, 0, 1);
        assert_eq!(
            conv2.pending_summary_len(timeline2),
            1,
            "default summarize=true must enqueue"
        );
    }

    #[test]
    fn absorb_pending_creates_leaf_with_normal_child() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let _normal0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        // pending_summary_queue now has 1 entry.
        assert_eq!(conv.pending_summary_len(timeline), 1);
        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4).expect("absorb ok");
        // The pending queue is drained.
        assert_eq!(conv.pending_summary_len(timeline), 0);
        // A SummaryOfTurns leaf now exists at index 1.
        let leaf_meta = conv
            .read()
            .tree_meta_of(timeline, TurnIndex(1))
            .cloned()
            .expect("leaf meta exists");
        assert_eq!(leaf_meta.kind, TurnKind::SummaryOfTurns);
        assert_eq!(leaf_meta.children, vec![TurnIndex(0)]);
        assert_eq!(leaf_meta.tree_height, 1);
        // Root is the leaf.
        assert_eq!(conv.read().tree_root_of(timeline), Some(TurnIndex(1)));
    }

    #[test]
    fn absorb_two_pending_creates_sos_parent_via_probe_allocator() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let _n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let _n1 = conv.write().append_with_blocks(timeline, 10, 1, 2);
        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4).expect("absorb ok");
        // Index 0 = Normal, 1 = first SoT, 2 = Normal, 3 = second SoT,
        // 4 = SoS parent (allocated by the alloc closure).
        // Order is determined by `append_with_blocks` ordering inside
        // the MockProbeRunner.
        // The root should be a SummaryOfSummaries.
        let root = conv.read().tree_root_of(timeline).expect("root exists");
        let root_meta = conv
            .read()
            .tree_meta_of(timeline, root)
            .cloned()
            .expect("root meta");
        assert_eq!(root_meta.kind, TurnKind::SummaryOfSummaries);
        assert_eq!(root_meta.children.len(), 2);
        assert_eq!(root_meta.tree_height, 2);
    }

    #[test]
    fn many_pending_keeps_tree_balanced() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        for i in 0..16u64 {
            conv.write().append_with_blocks(timeline, 10, i, i + 1);
        }
        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4).expect("absorb ok");
        // Build the in-memory tree from the substrate state and
        // verify it's balanced.
        let tree = conv.read().build_summary_tree_in_memory(timeline);
        assert!(
            tree.is_balanced(),
            "tree must be balanced after absorbing 16 pending turns"
        );
        // 16 Normal turns → 16 SoT leaves → log2(16) + 1 internals.
        // Height ≤ log2(16) + 1 = 5.
        assert!(tree.height() <= 5);
    }
}
