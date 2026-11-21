# Turn handover: a member that leaves hands its turns on

**Status:** design, 2026-11-18. Replaces "Moving off a card needs nothing new"
(`docs/claude_zend_gateway.md` §6.3), which this measured wrong.

## 1. What failed

A member that stops for an install, a `restart` or a drain
(`docs/replicated_substrate.md` §12.4) drains its claims, waits for the HTTP
requests in flight and stops its tenants. A zend turn is not a request: a
worker's turn, an MCP chat's answer, and a dialogue turn whose client has gone
all run on after their request ends. The stop cut them mid-round:

- 2026-11-18 12:12:04, WhiteWorld installing `e74ade355`: two chats' turns were
  running; the cluster sockets closed, the scheduler failed every wave naming
  their sequences, and nothing was handed on. Another member would have revived
  the chats only once their beat had been silent for `RESUME_AFTER_MS`, ten
  minutes (`mcp/resuming.rs`); the client waited on them as on a hung call.
- 11:51 the same day, the same on the install of `1f5ad2746`.

A dialogue turn cut off in its first round could never resume at all: its
message lives only in the process until the round seals, and the boot scan
(`resume_unfinished_turns`) finishes only a tool round whose call turn sealed.

## 2. The rule

**Every round of a request is resumable until the request's final response
has sealed.** A request whose final response sealed is finished and is never
run again. A round cut off loses only what it had not sealed: its partial
reply, and its tool calls' effects where those did not run.

**A member that leaves on purpose hands every live turn on before it lets its
claims go**, and the member that takes the conversation resumes it at once.

## 3. Every round's input is durable

| Round | Its input | Where it is durable |
|---|---|---|
| A tool round's response | the tool results of call turn *k* | the resume entry `resume_call_turn = k` (`zend/src/resume.rs`), written in the call turn's commit and cleared only when the request ends; the round is unanswered while no `Normal` turn follows *k* |
| A request's first round, dialogue | the user's message | the **pending message** `resume_message`, written before the round is submitted, cleared in the commit of the round's seal |
| A request's first round, worker | the worker's prompt | the worker's record (`Worker::prompt`), as today |

A resumed tool round runs its calls again under `Dispatch::Resumed`, which
answers a call that cannot run twice instead of running it. A resumed first
round submits the pending message as the request.

## 4. Leaving

When the process is asked to stop, before any claim is handed back
(`host/src/cluster/leave.rs`), the host calls each tenant's `leave`. zend's:

1. **Takes no new turn.** A turn asked for while it leaves is refused as the
   member is (`503`), and the gateway places it elsewhere once the drain lands.
2. **Stops every live turn where it stands.** The round's decode ends at its
   next event; nothing of it is sealed. The turn sends `StreamItem::Left` and
   leaves its entries in place — the resume entry, the pending message. A turn
   between rounds stops before its next submit.
3. **Marks each for the member that takes it**:
   - a dialogue turn: the conversation's `resume_handed` entry, the time it
     was let go;
   - a worker: its record names no runner and carries `handed_ms`.
4. **Waits for those turns to end**, at most `LEAVE_TURNS_WAIT` (30 s), so their
   records are written before the drain flushes and sends them. A turn still
   running its tools past that is not marked: it is cut at the shutdown and
   resumed by this member when it starts again, so no two members ever run it.

The drain then hands the claims and leases back as before, and the records
reach the group before the sockets close (`LEAVE_WAIT`).

## 5. Taking over

Every member sweeps every `RESUME_SWEEP` (10 s):

- **Workers** (`mcp/resuming.rs`): one whose record carries `handed_ms` and
  names no runner is due at once — its member said it stopped it — and is
  resumed under its `worker/<conversation>` lease as a lost worker is
  (`workers::resume`): its unfinished round first, else its prompt. A handover
  does not count toward `MAX_RESUMES`, which bounds unexplained losses.
- **Dialogues** (`zend/src/resume.rs`): one with `resume_handed` set and a
  resume entry or a pending message, not live here, is taken with its claim
  (`turn_claims::take`). The claim is the guard against running a turn twice: a
  live member holding it is still running the turn. Once taken and held whole
  (`wait_whole`), the entries are read again, the round is resumed — the tool
  round, or the pending message as a fresh request — and `resume_handed` is
  cleared.

A member's boot scan resumes its own unfinished turns as before, and now a
pending message too.

## 6. What this does not cover

**A member that dies without leaving** — a crash, a kill, the power — hands
nothing on. Its turns stay resumable: their entries are durable, and the member
resumes them when it starts again. Another member takes them only through what
already exists: a worker after `RESUME_AFTER_MS`, a dialogue not until its own
member returns. Telling a dead member from a slow one needs a liveness signal
stronger than the beat, which is outside this change. (keep stops a member
cleanly before it would kill it, `docs/replicated_substrate.md` §3.5, so a
deploy or a restart is a leave, not a death.)

**A sealed turn's K/V is written within a persistence pass or two of its seal,
pinned or not** (`Substrate::cold_pending`). A working-set pin governs
eviction only. Deferred while pinned, an active conversation's own turns
reached the redo log only once it went quiet, and a member killed mid-answer
came back with their tokens and no K/V (BlackWorld, 2026-11-21).

**A hole no catch-up fills is asked over.** Resume waits up to 120 s only for
a *gap* — another member's turn still arriving. A turn that is empty and no gap
— this member's own, sealed in the second or two before a crash with its K/V
still queued, or retired in place — no member holds. A round cannot carry on
over a history missing turns, and left to "one that does" it stayed unfinished
forever. So the request is **asked again**: its opening message — the last one
the user wrote, never a tool round's results (`resume::opening_message`) — runs
as a first round over the turns that are whole, its tools again under
`Dispatch::Resumed` (`resume::reask`). A request with no message to ask ends at
its last sealed round (`resume::abandon`).

**A request cut off with no entry left** — its entries retracted, as the build
before re-asking did — is resumed on asking: `POST /v1/conversations/:id/resume`
(`api::resume`, `ZendSession::resume_request`) resumes by the entries when they
stand, and otherwise reads the history: a last message that made tool calls,
or a tool round's results, with no answer after it, is cut off and asked again
from its opening message (`resume::cut_off_request`). A request whose last
answer called nothing ended, and is refused (`409`).

## 7. Tests

- `resume.rs`: the pending message marks, reads back and clears; a handed mark
  reads back and clears; a request whose final response sealed carries neither.
- `mcp/resuming.rs`: a handed worker is due at once and its resume does not
  count toward `MAX_RESUMES`; one not handed waits out `RESUME_AFTER_MS`.
- `host/src/cluster/leave.rs`: `hand_over` runs every tenant's `leave`, each to
  its end, in rank order; `run.rs`'s `leave` calls it before `drain_and_send`.
- `leaving.rs`: a leave with nothing running ends at once; one waits for its
  turns to stop; a turn that does not stop in time is counted.
- Live, on BlackWorld and WhiteWorld: a chat answering on one member when that
  member is restarted is answered by the other within `RESUME_SWEEP` of the
  drain.
