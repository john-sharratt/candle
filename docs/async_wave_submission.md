# Asynchronous Wave Submission — flume, awaitable turns, and a concurrent npcd

*How work reaches the wave engine without a thread blocked on every in-flight turn,
and how npcd is rebuilt on top of that so one slow mind never stalls the cast.*

Companion to [`unified_wave_inference_engine.md`](unified_wave_inference_engine.md),
which specifies the scheduler this document feeds, and to
[`wave_blocking_optimizations.md`](wave_blocking_optimizations.md), which covers
blocking *inside* the scheduler thread. This document is about blocking *in front of*
it: the callers.

---

## 1. The problem

The wave engine is built to batch. One scheduler thread steps every admitted sequence
through the model together, and a wave is only as wide as the work that is waiting when
it forms. The engine can serve a hundred concurrent minds; it is fed by callers that
submit one at a time.

### 1.1 Where the engine stands

| Piece | Where | Shape today |
|---|---|---|
| Request queue into the scheduler | `candle-conversation/src/scheduler/mod.rs` (`SchedulerRequest`) | `crossbeam::channel` — `Sender<SchedulerRequest>` held by every caller, drained by the scheduler thread, mid-wave included (`scheduler/run.rs`, `mid_wave_admission`) |
| Per-turn event stream | `candle-conversation/src/handle.rs` (`TurnHandle`) | wraps a `crossbeam::channel::Receiver<TurnEvent>`; `wait`, `wait_cancellable`, `stream` and `try_recv` are all blocking or polling |
| Submit vs. send | `candle-conversation/src/conversation.rs` | `submit_turn_with_options` returns a `TurnHandle` without blocking; `send_turn_with_options` is submit-then-`wait` |
| One-shot replies | `conversation.rs`, `engine.rs`, `tree/conversation_tree.rs` | dozens of `crossbeam::channel::bounded(1)` round trips, one `crossbeam::channel::Select` (`conversation.rs`) |
| Guest job replies | `candle-conversation/src/guest/queue.rs` (`GuestReceipt`) | `bounded(1)` crossbeam reply; `wait()` blocks the caller for the whole drain |
| Engine handle in npcd | `npcd/src/engine/*` | `Arc<Mutex<ConversationEngine>>`, locked per call |

The engine crate has no async runtime and must not acquire one: an engine that pulled
`tokio` in would decide the runtime for every embedder. The same constraint shaped
guest progress (`guest/progress.rs`), which reports through a caller-supplied callback
precisely so the engine never has to pick a channel — or a runtime — on the caller's
behalf.

### 1.2 Where npcd stands

| Piece | Where | Shape today |
|---|---|---|
| Tick driver | `npcd/src/engine/runtime.rs` — `drive` | **one OS thread**, walking due characters in turn on a 100 ms sleep loop |
| Character scheduling | `npcd/src/engine/tick.rs` | a due-heap plus per-character `due_at`, pace-floored heartbeats, `hold`/`release`, preempt flags — a hand-rolled scheduler quantised by the driver's sleep |
| A character's thought | `npcd/src/engine/mind.rs` — `Minds::think` | `send_turn_with_options` under the character's conversation lock: the driver thread is parked until that decode finishes |
| Reflection | `npcd/src/engine/runtime.rs` — `begin_reflection` | one OS thread per event (`reflect-<id>`), the character held until the first answer; four `send_turn_with_options` calls in sequence inside `reflect.rs` |
| Dreams | `npcd/src/engine/dreams.rs` | one OS thread per dream (`dream-<id>`), `send_turn_with_options` |
| World moments | `npcd/src/engine/driver.rs` — `Metronome` | **one OS thread per hosted world**, sweeping journeys/visitors every 500 ms |
| Life generation | `npcd/src/lifegen/{job,generate}.rs` | one OS thread per run, fanning out via `std::thread::scope` in `MAX_IN_FLIGHT` waves — a pool of blocking `send_turn` calls, by its own module doc *because* `send_turn` blocks |
| Prose (names, descriptions) | `npcd/src/engine/prose.rs` | `submit_turn_with_options`, then a blocking drain |
| HTTP handlers | `npcd/src/{api,ops,prose,guest_routes}.rs`, `engine/{mod,simulate,watcher}.rs`, `telemetry/mod.rs` | engine work bridged through `tokio::task::spawn_blocking` (ten call sites) |

### 1.3 What that costs

The driver thinks for one character at a time, so the scheduler sees **at most one
character turn in flight** from the tick loop. Everything the GPU could batch across the
cast is serialised on the host instead.

Measured on the live cast: the ticks ran fast at startup, then a reflection ran and every
character slowed together. A reflection is several decodes long, and while the driver
thread waited on it no other character was submitted. The model was never the
bottleneck; the thread in front of it was.

And the daemon is a museum of hand-rolled schedulers: the tick heap, the metronomes,
the reflection threads, the lifegen pool — four different answers to "run this later
without blocking that", each with its own wake bug fixed at its own time (the `due_at`
wedge, the eager-guard deadlock in `claim_dream`). tokio is one answer to all of them,
and npcd already runs on it for HTTP.

---

## 2. Goals and non-goals

**Goals.**

1. Any caller can submit work to the wave engine and **await** its result without holding
   an OS thread for the duration.
2. Existing synchronous callers keep working unchanged — the same channel serves both.
3. **npcd's tick thread is deleted**, not supervised into a new role. Every character is
   a long-lived async task on the daemon's runtime; its heartbeat, stalls and day
   roll-over are the task's own timers, and the due-heap goes with the thread.
4. **Character tasks are supervised.** A task that exits with an error is restarted with
   exponential backoff; deleting the character cancels its task; a daemon shutdown
   cancels them all before the engine goes down.
5. **Every engine call is cancellable.** Dropping the future stops the decode — so
   cancelling a task mid-thought releases the scheduler slot with no extra mechanism.
6. Every npcd tool call — acts on the world, engine round trips, guest jobs — is async
   end to end. `spawn_blocking` around **engine work** disappears from npcd; it remains
   the correct idiom for genuinely blocking non-engine work (NVML/sysinfo telemetry
   reads, bulk file hashing) and those sites keep it.
7. Reflection and dreams are tasks: a reflection is handed an async channel for its
   answer, delivers it, and carries on to conditionally launch the dream task behind it.
8. A long job — a reflection, a dream, a prose decode, a life episode — never delays
   another character's turn.
9. **zend converts to the same async model**, so the libraries the two daemons share can
   expose async APIs without a bridging thread in either.

**Non-goals.**

- The scheduler stays single-threaded. It is correct and it is fast; nothing here touches
  how a wave is formed or run.
- The engine crate does not depend on an async runtime. It exposes futures; the embedder
  chooses the executor.
- No change to what a turn *is* — prompts, grammars, sealing and the substrate are
  untouched.
- No change to the perception model. The inbox, the window, salience bands and
  superseding stay exactly as they are — what moves is *when the character runs*, not
  what it perceives.

---

## 3. Design

### 3.1 flume in place of crossbeam

Every channel between a caller and the scheduler becomes a `flume` channel:
the `SchedulerRequest` queue, the per-turn `TurnEvent` channel inside `TurnHandle`,
the one-shot reply channels, and the guest job's `GuestReceipt` reply.

flume is chosen for one property: **a single channel serves blocking and async receivers
alike, with no runtime dependency.** `recv` blocks a thread; `recv_async` returns a
future any executor can drive. The engine crate stays runtime-agnostic, synchronous
callers are untouched, and async callers need no bridging thread.

The scheduler side keeps its semantics exactly: bounded queues stay bounded, `try_recv`
stays non-blocking, and mid-wave admission drains the queue as it does now. The one
`crossbeam::channel::Select` becomes a `flume::Selector` over the same arms. Channels
*internal* to the engine — the persistence thread's trigger `select!`, the summariser's,
the writer pipeline — are not caller-facing and stay crossbeam.

This step is a pure substitution. Nothing gains `async` yet, and the full test suite must
pass unchanged before anything is built on it.

### 3.2 Awaitable turns

`TurnHandle` gains async counterparts to its blocking reads:

| Blocking | Async |
|---|---|
| `wait` | `wait_async` — `recv_async` until `Done` or `Error` |
| `wait_cancellable` | not needed: dropping the turn (see below) is the cancellation |
| `stream` | `stream_async` — the channel's `Stream` |
| `try_recv` | unchanged |

`Sequence` gains `send_turn_async` (and `_with_options`): submit, then `wait_async`.
The one-shot round trips an async caller needs — opening a conversation, the engine calls
npcd makes per turn — gain async variants by the same rule: send the request, await the
reply. Each of those variants lands **beside its first async caller** (step 3 and after),
not speculatively in step 2: an async variant nothing awaits is dead code the suite
cannot exercise, and the flume one-shot makes each conversion mechanical when its call
site arrives. `GuestReceipt` gains `wait_async` in step 2 itself — its blocking caller
(`guest_routes.rs`'s `spawn_blocking`) is already async-shaped and converts first — so a
guest job's caller awaits the drain instead of parking a blocking thread on it.

**Cancellation comes for free, and must be kept — and the unit of cancellation is the
`TurnHandle`, not a bare future.** The handle already states the contract: dropped
before `Done`, the scheduler sees the closed channel and stops decode at its next step.
`send_turn_async`'s future **owns** its handle, so aborting a task that is awaiting it
drops both — that is what makes goal 5 free for a character task. A `wait_async(&self)`
future only borrows the handle; dropping it alone abandons nothing, exactly as
returning early from a blocking `wait` would not — the owner drops the handle to
cancel. The drop must also clear the caller-side `turn_in_flight` guard (the blocking
path could never be abandoned mid-wait, so this state was unreachable before): a
cancelled turn leaves a usable sequence, its decoded-so-far winding down and sealing
scheduler-side. This is tested, not assumed (§5).

Two boundaries are stated rather than hidden. **The ends of an async send stay
synchronous** — submit composes and enqueues (fast; a first turn's branch checkpoint
does scheduler round trips), and the settle runs `finish_turn`, or `abort_turn` on a
failed turn, whose slot reset waits on the scheduler's next drain; callers treat a
turn's ends as brief host work, not await points, until step 3 converts the round trips
those ends make. And **the ingest-cancel latch grows an awaitable
face for step 7's async ingest waits**: a runtime-agnostic future cannot tick a timer,
so the latch itself wakes its waiters — a waker registry the latch drains on
`request_ingest_cancel` — and `wait_cancellable_async` races the turn's events against
it, woken by whichever speaks first. The blocking `wait_cancellable` keeps its poll:
it serves threads, and a thread has a timer.

### 3.3 Lock discipline

The rule is absolute: **no `std::sync::Mutex` guard is held across an `.await`.**

That forces the split the synchronous code blurs today. A turn is two phases:

1. **Submit** — under whatever lock the caller needs, briefly: compose the prompt, build
   the options, push the request onto the queue, get the handle.
2. **Wait** — with no lock held, awaiting the handle.

`Minds::think` holds the character's conversation lock across the whole decode today.
That lock exists so one character never has two turns in flight — a guarantee the task
model makes structural (§3.4): the character's loop is sequential, so it *cannot* have
two thoughts in flight, and the lock shrinks to what the HTTP routes genuinely share.

The engine mutex in npcd follows the same split. Calls that only enqueue take it for the
enqueue; nothing waits on the scheduler while holding it.

The rule is enforced by the types, not by review: character tasks are spawned, spawned
futures must be `Send`, and a held `std::sync::MutexGuard` is `!Send` — holding one
across an await is a compile error.

### 3.4 npcd: the character is a task

The tick thread is deleted. There is no driver, no 100 ms quantum, no due-heap, and no
`due_at` sentinel to wedge on. **Each character is one long-lived async task**, spawned
when the cast loads or the character is created, and its loop owns everything the tick
scheduler used to arbitrate:

- **Waking.** The task awaits its inbox. An event delivered to the character is a send
  on that channel; the task wakes when one arrives, drains whatever has queued behind
  it, and thinks over the batch — the same fat-batch semantics `tick` has today. A
  preempt is nothing special any more: it is a message arriving, and arrival is the wake.
- **No heartbeat.** The wait is bounded by the character's own deadline (`due_at`)
  alone: a quiet character parks indefinitely, and only a delivery, a pause expiring,
  or a `think_again` moves the deadline and rings the waker. Waking is not a reason to
  think — the pace table stays as the reported alertness ceiling the Pulse view reads,
  never as a timer, because an empty-inbox think is exactly the window-collapse the
  event-only model removes.
- **Stalls.** An act that takes time (`pause_for`) is the task sleeping. The *other*
  cooldown — how soon each act may be taken again — is deliberately **not** a sleep: it
  feeds the grammar (`Within::cooling` must answer "how much longer"), so it stays the
  timestamp table it is.
- **The day boundary.** Checked at the top of the loop, as `drive` checks it now. A
  roll-over retires the conversation and opens the new day's — inside the task, not by
  restarting it.
- **The standing task, perception, acts.** Unchanged. The loop body is `tick`'s
  three-phase shape — drain, think, land — with the think phase awaited instead of
  blocked on, and the inbox map locked only for the drain and the landing exactly as it
  is today.

With the cast thinking concurrently, the scheduler receives their turns together and
batches them into one wave. The width of that wave is bounded by the scheduler's own
admission — the VRAM gate that already decides what fits — not by the driver.

**One thought in flight per character** stops being a lock's job and becomes the shape
of the code: a sequential loop cannot overlap its own iterations. The `Live` map stays
for the routes that read a character's conversation from outside (`layer_counts`,
`layer_page`, `deliver_outcomes`), behind a lock that is never held across a decode.

**Supervision.** The runtime keeps one entry per character in a map keyed by npc id:
the supervisor wrapper's `JoinHandle`, and the abort handle of whichever attempt is
currently running the loop. Both, because the wrapper runs each attempt as its own
spawned task (so a panic is contained), and aborting only the wrapper would *detach*
the attempt — a dropped `JoinHandle` lets its task run on.

- **Deletion cancels.** `stop_character` aborts the supervisor and its current
  attempt; the attempt's next await returns, the in-flight turn (if any) is dropped
  and §3.2 stops its decode. A stop that races the wrapper between spawning an attempt
  and registering its handle tombstones the slot, and the wrapper puts that attempt
  down itself. `delete_npc` stops the task before retiring the mind, so re-creation
  never runs beside a zombie loop.
- **A panic restarts, with exponential backoff.** The loop returns nothing, so the
  failure a supervisor sees is a panic; the attempt is restarted first after a second,
  doubling to a cap of a minute, reset after a healthy stretch. The backoff exists for
  the systemic case — an engine that is down takes the whole cast's tasks down, and a
  hundred characters retrying hot would be a stampede on a scheduler that is trying to
  come back.
- **Shutdown cancels everything, in order.** Latch `stopping` (the metronome callbacks
  stop moving worlds and a supervisor whose attempt dies mid-shutdown stops
  restarting), then `stop_characters` cancels every task — parked or mid-await — and
  awaits the supervisors out, so every dropped turn future has released its slot
  before anything behind them is torn down. The flag survives as that latch; the
  driver whose polls it used to serve is gone.

**The metronomes go the same way.** A hosted world's moment sweep — journeys, visitor
following, place checkpoints — becomes an interval task per world instead of an OS
thread per world. Pause/resume becomes the task's own state. Same 500 ms cadence, same
sweep, one scheduler family instead of two.

**What the tick scheduler's API becomes**, so the rewrite has a checklist rather than a
discovery process. `tick.rs` is two things wearing one type: a scheduler and a data
model. The data model survives untouched — `deliver`, `broadcast`, `push`/`drain`, the
window, `window_of`, `amend_act`, `recent`, `census`, `population`, `inbox_depth`,
`quiet_for`, `nudge_due` — these are reads and writes of per-character state the routes
and the Pulse feed depend on, and delivery doubles as the wake (the inbox's
waker). The scheduling half is deleted or absorbed: the due-heap and `tick`'s
driver-facing shape go, and `roll_day` moves into the loop; `due_now` survives only as
a non-mutating read — every inbox due at a given instant — for the tests that drive
ticks by hand; `set_pace`/`pace_of` stay as reported alertness, never a timer (see the
heartbeat bullet); `pause_for` becomes the task sleeping; `think_again` becomes the
task looping without waiting; `hold`/`release`/`is_held` become the reflection await
of §3.5.

**What deliberately stays a thread.** The startup loader — bind the port first, load
the model and replay the redo log on a plain `std::thread` behind it — keeps its
rationale unchanged in both daemons; nothing in it is async work, and it runs once.
zend's GPU-poison watchdog stays a std thread for a stronger reason: it must fire when
the runtime itself is wedged behind a dead CUDA context. npcd's mind watcher is
already the target shape — a tokio debounce task in front of a `spawn_blocking` file
reconcile — and is the precedent the remaining conversions follow: async where the
engine is awaited, `spawn_blocking` where the work is honestly blocking.

### 3.5 Reflection and dreams

A `reflect` act the world accepted spawns the reflection as a task, and the character's
own task **awaits the answer channel** — a oneshot the reflection is handed at spawn.
That await *is* the hold: the character does not reach its next think until the first
question is answered, expressed as code order instead of `hold`/`release` flags. If the
channel drops without an answer — the reflection failed before it got there — the
character receives the world's own fallback line, exactly as `begin_reflection`'s
failure path delivers it today.

The reflection task then carries on behind the character, with nobody waiting: the
loosed turn, the brief, and — **conditionally** — the dream. The dream-slot guard keeps
its exact semantics (at most one dream in flight per character, the slot handed back on
every exit path); what changes is that claiming it launches the dream as its own task
rather than continuing on a thread. The character has long since had its answer and is
back in its loop.

Prose jobs submit and drain their fragments through `stream_async`. Life generation
keeps its wave shape — submit a wave together so the guest drain batches it — but the
wave is a set of concurrently awaited turns, not a `std::thread::scope` of blocking
calls; the module doc's stated reason for the thread pool ("`send_turn` blocks") is
exactly what this design removes.

### 3.6 HTTP handlers

Each `spawn_blocking` around engine work becomes a direct `.await` on the async API. A
request that is cancelled — the browser navigated away — drops its future, and §3.2's
cancellation stops the decode it started. Routes that read a character's live state go
through the same locks the task uses, held for the read and never across an await.

Of npcd's ten `spawn_blocking` call sites, seven wrap engine work and convert
(`api.rs` reflect, `prose.rs`, `guest_routes.rs`, `engine/mod.rs` ×2, `simulate.rs`,
and the engine half of `watcher.rs`'s reload). Three wrap genuinely blocking non-engine
work and stay: the telemetry sampler's NVML/sysinfo tick, the ops memory dump, and the
watcher's file-hash reconcile — `spawn_blocking` is the idiom for those, not a bridge
to be removed.

### 3.7 zend

zend converts on the same pattern, for the same reasons and one more: **the libraries
the two daemons share can only be async if both ends are.** A shared helper that
returns a future is unusable from a blocking daemon without a bridge thread, and a
blocking helper poisons every async caller with `spawn_blocking` — so as long as one
daemon is synchronous, the common code is written twice or bridged everywhere.

The engine-facing shape is identical: submit under short locks, await handles, drop to
cancel. What that lands on, concretely — zend's blocking machinery as audited:

| Piece | Where | Shape today → after |
|---|---|---|
| The submit path | `session.rs` — `run_inference_stream` | one `spawn_blocking` per request, streaming through a `tokio::sync::mpsc` bridge → an async task streaming `stream_async` directly |
| Titler | `session.rs` — dedicated worker thread fed by a `std::sync::mpsc::sync_channel(256)` | an async task fed by a bounded channel; same drop-on-backlog, same drain-on-shutdown |
| Scope ingest fan-out | `turn_sink.rs` — `std::thread::scope`, one thread per fork per chunk | concurrent awaits over the same forks; the co-batching the scheduler does is unchanged |
| code_read / repo_scan fan-outs | `code_read/mod.rs`, `repo_scan/mod.rs` — `thread::scope` pools | same conversion where the work is engine turns; repo scanning itself is file I/O and keeps its blocking pool |
| Calibration retire loop | `session.rs` — 2 ms sleep-poll over in-flight handles | await whichever handle completes next (completion order, as the section-ingest `Selector` already does) |
| Upload ingest progress | `api/files.rs` — worker thread + 150 ms progress poll | the ingest on the blocking pool, awaited by a task that streams progress on a 150 ms async tick — the tick doubles as the SSE cadence throttle, so no thread is parked on it |
| Tool execution | `tools.rs` — `run_tool_calls`, synchronous file I/O between turns | stays blocking work, run off the async pool; the turns around it are awaited |
| Startup loader + reload polls | `session.rs` — loader thread, 50 ms polls of `SubstrateReloadStatus` | stays a thread (§3.4's exception); the polls are fine where they are |
| GPU-poison watchdog | `main.rs` | stays a std thread, deliberately — it must fire when the runtime is wedged |

zend's own long jobs (ingest loops, code reading) become tasks the way npcd's
reflections do. The conversion lands after npcd's (§4), so the async API has been
proven against the harder consumer first.

---

## 4. Order of work

Each step lands on its own, passes the full suite, and leaves the system working —
except steps 3–6, which land as ONE change: a blocking `Minds::think` has no caller
once the tick thread is gone, and an async one has no driver while it remains, so
splitting them would require a `block_on` shim between the halves, which is exactly
the dual-path bridge this design forbids. (And that is how they landed: steps 1–2
first, each green on its own; then 3–6 together as the npcd rewrite; then 7.)

1. **flume substitution** in candle-conversation: the request queue, `TurnHandle`, the
   one-shot replies, the guest receipt, the `Select`. No behaviour change; every
   existing test passes unchanged.
2. **Async turn API**: `TurnHandle::wait_async` / `stream_async`,
   `Sequence::send_turn_async`, `GuestReceipt::wait_async`.
3. **`Minds::think` async**, with the submit/wait split of §3.3 — and, beside each
   converted call site, the async variant of whichever engine round trip it awaits
   (§3.2's rule: the variant lands with its first caller).
4. **Character tasks and the supervisor** (§3.4): the tick thread deleted, the due-heap
   deleted, per-character tasks with heartbeat timers, cancellation on delete, restart
   with backoff, ordered shutdown. World metronomes become interval tasks in the same
   step — it is the same conversion, and leaving them would keep the second scheduler
   family the step exists to remove.
5. **Reflection, dreams, prose, life** as tasks (§3.5) — the reflection answer channel,
   the dream slot on the task path, the lifegen wave as concurrent awaits.
6. **HTTP handlers** onto the async API; `spawn_blocking` around engine work removed
   from npcd (the three honestly-blocking sites of §3.6 stay).
7. **zend** onto the same model (§3.7); shared libraries expose async APIs.

---

## 5. Tests and measurement

**Tests, written with each step.**

- A turn awaited with `wait_async` on a current-thread executor completes with the same
  response a blocking `wait` gives.
- Dropping an in-flight turn's handle — which is what dropping a `send_turn_async`
  future does, since it owns one — stops the decode: the scheduler sees the closed
  event channel and finishes the sequence at its next step, and the sequence's
  in-flight guard is cleared so the conversation takes its next turn normally.
- A blocking and an async receiver on the same kind of channel both see every event, in
  order.
- Two thoughts are never in flight for one character, however events arrive — including
  a preempt landing mid-decode.
- Cancelling a character's task mid-thought releases its scheduler slot and removes it
  from the supervisor's map; the character can be re-created immediately after.
- A character task that panics is restarted, with the backoff observed to grow and then
  reset after a healthy run; a deleted character is **not** restarted.
- A `reflect` act's answer reaches the character before its next think, and the world's
  fallback line arrives if the reflection dies before answering.
- No lock guard crosses an await — enforced by the types (`Send` bounds on spawned
  futures reject a held `std` guard), not by review.

**Measurement, on the live cast.**

- Ticks per minute per character, with and without a reflection running. Today a
  reflection collapses the cast's rate; the target is that it does not move it beyond
  noise.
- Scheduler wave width during normal ticking. Today it is one character turn from the
  driver; the target is the number of due characters, up to what admission allows.
- Wake latency: time from an event's delivery to the character's think beginning. Today
  it is quantised by the driver's 100 ms sleep; a task woken by its own channel should
  cut that to the runtime's wake cost.

---

## 6. Risks

- **The world moves under concurrent thoughts.** A character's grammar and situation are
  read before its decode starts (`rt.within`), and the world can change before its acts
  land. That is already true of one slow decode; concurrency makes it common. Acts are
  validated against the world when they land, as now, and a rejected act is told to the
  character, as now.
- **Admission, not the driver, now bounds width.** A cast that thinks all at once asks the
  scheduler for more KV at once. The VRAM gate already refuses what does not fit; the
  risk is starvation patterns in how refused work is retried, which the measurement above
  watches.
- **Lock ordering.** Splitting submit from wait touches every lock npcd holds around the
  engine. Each split is made and tested one call site at a time.
- **Blocking inside a task.** The engine mutex taken briefly to enqueue is fine; a
  redo-log write, a world lock held through a sweep, or any other slow synchronous call
  inside a character task stalls a runtime worker and, with enough characters, the
  runtime. The discipline is the same as §3.3's: slow work happens before the submit or
  in a task of its own, never inline between awaits.
- **Cancellation points are await points.** Aborting a task takes effect at its next
  await, so a task deep in synchronous work is not cancelled until it surfaces. Guards
  drop correctly on abort — the dream slot's drop-guard pattern already assumes exactly
  this — but any synchronous section long enough to matter is the previous bullet's
  problem first.
- **The restart storm.** A systemic failure — the engine down, the substrate unwritable —
  fails every character task at once, and a supervisor that restarts eagerly turns one
  outage into a retry stampede. The exponential backoff and its cap exist for this case;
  the log line on every restart is what makes a character stuck in backoff visible
  rather than merely quiet.
- **The engine crate must stay runtime-agnostic.** Any `tokio` type appearing in
  candle-conversation's public API is a design violation, whatever it simplifies.
