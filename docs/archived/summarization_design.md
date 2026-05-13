# Summarization Pipeline Design

## What exists

The trigger logic is already complete inside `ConversationTree`:

- `finish_turn()` calls `check_and_trigger_summarize()` after each turn
- That builds a `SummarizationSnapshot` — the full window of turns since the last segment, with text and `TurnId` bounds
- `run_summarize()` is called with the snapshot; it currently logs at `debug` and returns

The fork/patch machinery is also already in place:
- `ConversationTree::fork()` → `ConversationTreeFork` + `Receiver<TreePatch>`
- `ConversationTree::apply_patch()` accepts a completed `TreePatch` with new nodes

Nothing about the actual inference path is wired yet. That is what this document covers.

The scheduler already supports the exact request types we need:
- `SchedulerRequest::NewConversation` — allocates a fresh KV slot
- `SchedulerRequest::SubmitTurn` — prefills + decodes text in that slot, streams events back
- `SchedulerRequest::FreeSequence` — releases the slot

Summarization is a stateless one-shot inference: text in, summary text out. We don't need a new scheduler request type — we just use a temporary sequence slot, run one turn in it, then free it. The summarization context is completely independent of the main conversation's KV.

---

## The key structural change

`ConversationTree` is a pure data structure. It does not have access to the scheduler channel. The summarization trigger fires inside the tree, but inference must happen in `Conversation`, which owns the scheduler channel.

`finish_turn()` stays `-> TurnId`. Nothing leaks out. When triggers fire, the tree pushes boxed `CognitiveTask`s onto a queue on itself:

```rust
// On ConversationTree:
pending_tasks: Vec<Box<dyn CognitiveTask>>,
```

Multiple tasks can accumulate in a single `finish_turn()` call — a summarization window closes and a daydream fires on the same turn, for example. `Conversation::finish_turn()` calls `tree.drain_pending_tasks()` and holds the handles. Each handle exposes only what the consumer needs: task type, completion poll, abort signal, and patch retrieval. `Conversation` never touches the engine or fork machinery directly — that was already done when the task was created.

```
// Conversation::finish_turn(response):
//   calls tree.finish_turn(...)         → TurnId, no change to signature
//   calls tree.drain_pending_tasks()    → Vec<Box<dyn CognitiveTask>>
//   appends to self.pending_tasks
//   [crude: spin-poll until each is Ready before returning]
//   [async: just accumulate; drain_pending_tasks() checks on next turn]
```

Sleep, Daydream, and Reason tasks follow the same pattern: different background inference paths, same handle trait, same poll loop in `Conversation`.

---

## `CognitiveTask` trait

Defined in a new `task.rs` module inside `candle-conversation/src/`. This is a **consumer-side handle** — by the time it reaches `Conversation`, the background work has already been launched (fork allocated, scheduler slot acquired, inference running). The trait describes only what the consumer can do with a running task:

```rust
/// Identifies the kind of cognitive work a task represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaskKind {
    Summarization,
    Daydream,
    Sleep,
    Reason,
}

/// Result of polling a running task.
pub(crate) enum TaskPoll {
    /// Still running, no result yet.
    Pending,
    /// Finished. The patch is ready to apply.
    Ready(TreePatch),
    /// Task was aborted before it produced a result.
    Aborted,
    /// Task failed. The error is logged; no patch is applied.
    Failed(crate::Error),
}

pub(crate) trait CognitiveTask: Send {
    /// What kind of background work this task represents.
    fn kind(&self) -> TaskKind;

    /// The turn range this task operates over, if applicable.
    /// Used to detect duplicate tasks before launching a new one.
    /// Tasks with no specific turn scope (e.g. Sleep) return None.
    fn relevant_turns(&self) -> Option<RangeInclusive<TurnId>>;

    /// Non-blocking poll. Returns Ready/Aborted/Failed once the
    /// background work finishes; Pending otherwise.
    fn poll(&mut self) -> TaskPoll;

    /// Signal the running task to stop. Idempotent.
    /// The next poll() call will return Aborted once the task has
    /// noticed the signal and exited.
    fn abort(&self);
}
```

`Send` is required because handles can be moved across threads. No `execute()` method — the task is already running when the handle is constructed. The tree creates the handle, pushes it onto `pending_tasks`, and forgets about it; `Conversation` owns the drain loop.

The tree stores `Vec<Box<dyn CognitiveTask>>`. It pushes handles for whatever tasks fire during a turn; `Conversation` drains and polls them.

---

## `SummarizationTask`

Implements `CognitiveTask` as a threadless handle. The scheduler already runs on its own thread and streams `TurnEvent`s back via a channel. There is no need for a relay thread — the task just holds the receive end of that channel and drains it non-blockingly inside `poll()`.

`run_summarize()` does the synchronous setup before returning the handle:

1. Tokenizes the system prompt and window text
2. Sends `NewConversation` and waits for `seq_id` (one fast round-trip to the scheduler)
3. Sends `SubmitTurn` with an `event_tx` channel (fire-and-forget; inference starts immediately on the scheduler's thread)
4. Returns a `SummarizationTask` holding `event_rx`, `seq_id`, and accumulation state

```rust
pub(crate) struct SummarizationTask {
    seq_id: usize,
    cancelled: Arc<AtomicBool>,
    event_rx: crossbeam::channel::Receiver<TurnEvent>,
    scheduler_tx: crossbeam::channel::Sender<SchedulerRequest>,
    accumulated: String,
    span: (TurnId, TurnId),
}

impl CognitiveTask for SummarizationTask {
    fn kind(&self) -> TaskKind { TaskKind::Summarization }

    fn relevant_turns(&self) -> Option<RangeInclusive<TurnId>> {
        Some(self.span.0..=self.span.1)
    }

    fn poll(&mut self) -> TaskPoll {
        if self.cancelled.load(Ordering::Relaxed) {
            self.scheduler_tx.send(SchedulerRequest::FreeSequence {
                sequence_id: self.seq_id
            }).ok();
            return TaskPoll::Aborted;
        }
        // Drain all currently available events without blocking.
        loop {
            match self.event_rx.try_recv() {
                Ok(TurnEvent::Token(id)) => {
                    // decode and append
                }
                Ok(TurnEvent::Done(_)) => {
                    self.scheduler_tx.send(SchedulerRequest::FreeSequence {
                        sequence_id: self.seq_id
                    }).ok();
                    let patch = build_patch(&self.accumulated, self.span);
                    return TaskPoll::Ready(patch);
                }
                Err(TryRecvError::Empty)        => return TaskPoll::Pending,
                Err(TryRecvError::Disconnected) => return TaskPoll::Aborted,
            }
        }
    }

    fn abort(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        // FreeSequence is sent on the next poll() call to avoid
        // sending it twice if poll() is already mid-drain.
    }
}
```

No background thread anywhere. The scheduler's thread sends events; `poll()` collects them. The task holds `scheduler_tx` purely to send `FreeSequence` when done or aborted — it never drives inference.

---

## The crude version: blocking in place

The simplest complete loop. After `tree.finish_turn()` returns, drain any new task handles and spin-poll each until it resolves before returning to the caller.

```
Conversation::finish_turn(response):
    self.tree.finish_turn(user, assistant, ...)
    
    for task in self.tree.drain_pending_tasks():
        self.run_task_blocking(task)
```

```
Conversation::run_task_blocking(mut task: Box<dyn CognitiveTask>):
    loop:
        match task.poll():
            TaskPoll::Ready(patch) => { self.tree.apply_patch(patch); return }
            TaskPoll::Aborted      => return
            TaskPoll::Failed(e)    => { log error; return }
            TaskPoll::Pending      => std::thread::yield_now()
```

The synchronous setup in `run_summarize()` (the `NewConversation` round-trip) is unavoidable but fast — it's a single channel message to the scheduler with no inference work. After that, `Conversation` just spins on `poll()` which drains whatever tokens the scheduler has produced so far on each call.

The conversation thread blocks during step 4. The next user turn cannot be submitted until summarization finishes. For turn counts of 8–16 turns (short windows), this is probably a 200–500ms pause — noticeable but acceptable for a first pass.

No new types. No new scheduler requests. Uses existing machinery end-to-end.

---

## The summarization prompt

Two parts: a system prompt and the window text.

**Summarization system prompt** (injected as the `system_tokens` in `NewConversation`):

```
You are a memory assistant. You will be shown a set of conversation turns between
a user and a character. Write a concise summary of the key events, decisions, and
emotional beats in those turns. Write in past tense, from the character's perspective.
Be specific about what was said and decided. Do not editorialize or evaluate.
Maximum 200 words.
```

This can be stored in `ConversationTreeConfig::summarization_system_prompt: String` so it can be overridden per-character.

**Window text** (the `SubmitTurn` prefill):

Each `SummarizationTurnEntry` becomes:

```
[T-3.47]
User: {user_text}
{character_name}: {assistant_text}
```

Concatenated in order. The temporal markers are included if enabled — they give the model a timeline sense that helps it phrase the summary correctly ("early in that conversation", "later that day").

`character_name` can default to "Character" or be a field on `ConversationTree` (already part of the design — the system prompt names the character, so we could extract it, but hardcoding a default is fine for the crude version).

---

## The async upgrade

Instead of spin-polling, accumulate handles and check them at turn boundaries.

```
Conversation struct gains:
    pending_tasks: Vec<Box<dyn CognitiveTask>>
```

```
Conversation::finish_turn(response):
    self.tree.finish_turn(user, assistant, ...)
    self.pending_tasks.extend(self.tree.drain_pending_tasks())
    // return immediately — tasks run in background
```

```
Conversation::drain_ready_tasks():
    self.pending_tasks.retain_mut(|task| {
        match task.poll() {
            TaskPoll::Ready(patch)  => { self.tree.apply_patch(patch); false }
            TaskPoll::Aborted       => false
            TaskPoll::Failed(e)     => { log error; false }
            TaskPoll::Pending       => true
        }
    })
```

```
Conversation::submit_turn(user_text):
    self.drain_ready_tasks()    ← add this at the top
    // ... existing logic
```

All in-flight tasks make progress via the scheduler's own thread, which streams `TurnEvent`s for all active sequences concurrently. `Conversation` just polls handles at turn boundaries — no threads of its own, no cloning of channels into tasks at dispatch time.

Note: `SummarizationTask` holds a clone of `scheduler_tx` (obtained at creation time, inside `run_summarize()`) solely to send `FreeSequence` when done or aborted. It does not drive inference.

---

## What changes in code

### New: `src/task.rs`

- Define `pub(crate) enum TaskKind` — `Summarization`, `Daydream`, `Sleep`, `Reason`
- Define `pub(crate) enum TaskPoll` — `Pending`, `Ready(TreePatch)`, `Aborted`, `Failed(crate::Error)`
- Define `pub(crate) trait CognitiveTask: Send` with:
  - `fn kind(&self) -> TaskKind`
  - `fn relevant_turns(&self) -> Option<RangeInclusive<TurnId>>`
  - `fn poll(&mut self) -> TaskPoll`
  - `fn abort(&self)`

### `summarize.rs`

- Add `pub(crate) struct SummarizationTask { seq_id, cancelled, event_rx, scheduler_tx, accumulated, span }`
- `impl CognitiveTask for SummarizationTask` — `poll()` does `try_recv()` loop; `abort()` sets flag; `FreeSequence` sent in `poll()` on done/abort
- `run_summarize()`: tokenize prompt, `NewConversation` (blocking round-trip for `seq_id`), `SubmitTurn` (fire-and-forget with `event_tx`), construct handle, push onto `pending_tasks`
- Tree needs `scheduler_tx: Sender<SchedulerRequest>` and `tokenizer: Arc<Tokenizer>` available at trigger time — passed through `check_and_trigger_summarize()` or stored on the tree at construction

### `ConversationTree`

- Add `pending_tasks: Vec<Box<dyn CognitiveTask>>` field
- `run_summarize()` — synchronous setup (tokenize + `NewConversation` + `SubmitTurn`), construct `SummarizationTask` handle, push onto `pending_tasks`. No thread spawned.
- Add `drain_pending_tasks() -> Vec<Box<dyn CognitiveTask>>` method
- `finish_turn()` signature **unchanged** — stays `-> TurnId`
- `check_and_trigger_summarize()` — before calling `run_summarize()`, scan `pending_tasks` for any task whose `relevant_turns()` overlaps the candidate window; skip if found. Also needs `scheduler_tx` and `tokenizer` threaded in — either stored on the tree at construction or passed as parameters

### `patch.rs`

- `apply_patch()` — before appending a new `ConversationSegment`, scan existing nodes backwards for any segment whose `end_turn.seq >= incoming.end_turn.seq`; discard the patch if found. Makes patching unconditionally idempotent — late-landing async patches and genuine duplicates both handled by the same rule.

### `config.rs` (ConversationTreeConfig)

- Add `summarization_system_prompt: String`
- Add `summarization_max_tokens: u32` (default 256)

### `Conversation`

- `finish_turn()` — after `tree.finish_turn()`, extend `self.pending_tasks` with `tree.drain_pending_tasks()`
- Add `run_task_blocking(task: Box<dyn CognitiveTask>)` — spin-polls until `Ready`/`Aborted`/`Failed`
- Add `drain_ready_tasks()` — `retain_mut`, calls `poll()`, applies ready patches, drops finished entries
- Add `pending_tasks: Vec<Box<dyn CognitiveTask>>` field

### `scheduler/mod.rs`

- No changes needed

---

## Ordering consideration

When `apply_patch()` appends the segment node, it goes at the *end* of the `nodes` vec — after the turns it summarizes. That is correct: the tree is chronological, and the segment is created after those turns complete. The segment's `SegmentId` carries the `start_turn` / `end_turn` fields that identify the range it covers, so the ordering in the vec doesn't need to be inverted.

`turns_since_last_summarize()` scans backwards for the most recent segment's `end_turn.seq`. Once the segment is appended, the next trigger check will correctly count from zero again.

---

## Design notes

`SummarizationSnapshot.node_range` records the vec-index range at snapshot time and goes stale as turns are appended. This is intentional — it is a diagnostic field only. Correctness is anchored entirely on `TurnId`s (`SegmentId.start_turn` / `end_turn`), which never move.
