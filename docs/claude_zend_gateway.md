# Claude and zend: one orchestrator, one workhorse

**Status: designed, not built.** This document is authoritative over the remote
MCP endpoint (`/mcp` behind the `web` gateway), the gateway's OAuth
authorization server for it, the global skills that drive it, and the routing
rules a Claude session follows. Where it names machinery that already exists,
that machinery's own document stays authoritative, and the amendments this
design makes to them are listed in §16:

| Machinery | Document |
|---|---|
| zend conversations, provenance and the working set | `docs/zend_working_set.md`, `docs/attention_provenance.md` |
| The journal | `docs/zend_journal.md` |
| The layers | `docs/zend_layers.md` |
| Sign-in and identity | `docs/web_gateway_design.md` §5 |
| Membership, A/B upgrades, rollouts, maintenance commands, agents | `docs/replicated_substrate.md` §3, §11, §12 |
| keep and the hand-off | `docs/replicated_substrate.md` §3.9 |

Every statement below about existing code was checked against the tree on
2026-10-25 and carries its source; everything that does not exist yet is
marked **new**. The Claude-side facts (§12, §13) were checked against
Anthropic's and the MCP specification's published documentation the same day.

## 1. Why

The work this repository and Battle Cities need is mostly volume: reading,
implementing against a spec, building, testing, deploying, watching, checking.
Done in Claude Code, all of it is paid for out of one Max 20 subscription, and
heavy, quality-sensitive work uses that up in about half a week. Cheaper
models and thinner checking are quality drops, and a fresh context per task
loses the anchors a long session builds up.

zend runs on hardware already owned. Its marginal cost per token is power, its
conversations persist and are projected by provenance, and its batched engine
serves dozens of conversations at once. It is not yet as reliable as Claude: it
can attribute a fact to the wrong file or state something it never read.

So the work is split by what each is good at, not replaced:

| Claude (Max 20 is the whole budget) | zend (tokens are effectively free) |
|---|---|
| Orchestration: what to do next, in what order | Implementation against a written spec |
| Answering zend's questions, and pushing it on when it is stuck (§6.2) | Asking, with options and evidence, instead of guessing |
| High-level design and its documents | Reading and answering questions about the code |
| Reviewing zend's results, and drilling in where one looks wrong | Building, testing, watching |
| Every decision that needs the user's approval | Web research, code review, security review |
| The work itself, when zend is down | Verifying its own work before it answers |

The aim is to move at least 10× — and towards 100× — more work through the
same Claude budget, without lowering the standard of what is accepted.

## 2. Shape

```
                                          ┌─ zend member: /mcp; native conversations
 Claude Code ─┐                           │    (provenance, journals) and tools
              ├─ remote MCP ─► gateway ───┼─ every member: maintenance commands, status, logs
 claude.ai  ──┘   /mcp       sign-in, OAuth├─ agents (inside each motherd): build, test, run
      │                                   └─ keep, on each machine: trials, start, stop
      └─ global skills: the workflows, the prompts, the result formats, the fallback
```

Five parts:

1. **Work tools** (§3): attach to a zend thread — Claude's own conversation in
   zend (§4) — start work on it, and read what came back through the journals
   the thread and its workers keep (§5).
2. **Operations tools** (§8): the cluster's management, build, test and deploy
   flows.
3. **The authorization server** (§12): OAuth for the MCP clients, in the
   gateway beside the sign-in it already owns, through the Google account it
   already uses.
4. **Global skills** (§7, §9): each workflow is a skill: what it calls, the
   prompt it sends, what it requires the worker to verify, the result format,
   what needs the user's approval, and what Claude does if zend is down. A
   skill is a markdown file: tuning a workflow is an edit, with no build and
   no deploy.
5. **Routing rules** in each project's `CLAUDE.md` (§10).

**Remote, not local.** The endpoint is served over HTTPS through the gateway, so the
same tools reach the cluster from Claude Code on any machine and from an
ordinary claude.ai conversation. claude.ai calls connectors from Anthropic's
cloud (`160.79.104.0/21`), so `/mcp` and the authorization server's endpoints
must be reachable from the internet with nothing in front of them that refuses
those addresses (§13.1). Nothing in the workflows needs the caller's shell or
files: zend and its agents build, edit, push and deploy, and Claude
orchestrates and reviews.

## 3. Work tools

`/mcp`, MCP's Streamable HTTP transport, served by a zend member
(`zend/src/mcp/http.rs`) and forwarded by the gateway. The member checks the
bearer token itself (§12). It answers both protocol eras in use:

- **2025-11-25**, claude.ai's: `initialize` negotiates the version and opens
  a session named by `Mcp-Session-Id`; an unknown session is `404`, which
  tells the client to initialize again.
- **2026-07-28**, Claude Code's: no handshake and no session; every request
  names its version on `MCP-Protocol-Version` and its client in
  `params._meta`, and `server/discover` describes the server. What the server
  holds for such a client — its attached thread — is keyed by its token. A
  version served by neither era is `400` with `-32022` and the supported list.

A browser `Origin` naming anywhere but this origin is `403`.

### 3.1 Native zend conversations, never a passthrough

**Every delegated turn is a native zend conversation**, so it is projected the
way a dialogue is:

- **provenance retrieval** over every layer — `repo_map`, `code_reading`,
  `dependency_analysis`, `code_review`, `security_review` — scoped to the
  conversation's branches (`docs/zend_layers.md`, "Tied to a branch");
- **the working set**: seeds, the files its reads lock, momentum
  (`docs/zend_working_set.md`);
- **its journal and its ancestors' journals** in its prompt (§5);
- its own branch per repository, its tool rounds' changes, its owner.

The turn is submitted on `POST /v1/chat/completions` with `conv_id` naming the
worker and zend's own model (`zend/src/api/mod.rs:69-78`). With any model but
`passthrough` that route runs zend's projection through `session.submit`, and
answers `400 missing_conv_id` without a `conv_id` (`zend/src/api/chat.rs:152-223`).
**`passthrough` is refused by the endpoint**: it runs the client's own
messages through `submit_passthrough` and never touches the substrate
(`chat.rs:158-160`), which is exactly what delegation exists to avoid.

### 3.2 Tools

Every conversation is addressed by its **path** (§5.2): `T:review-layers` for a
thread, `T:review-layers/verify-expiry` for a worker under it. A tool taking a
`journal` takes a path, or `"self"` for the thread the session is attached to
(§5.3).

| Tool | Does | Returns |
|---|---|---|
| `zend_attach {thread}` | Binds the MCP session to `thread`, creating it on first use (§4) | The thread's path and its journal's head lines |
| `zend_task {thread, name, prompt, priority?, wait_s?}` | Starts a worker named `name` under `thread` (§4, §5.2), runs the task on it, waits for its journal entry (§5), records the outcome on the thread, and archives the worker once it is `done` | One of the states of §3.3 |
| `zend_wait {workers, wait_s?}` | Waits on one or more running workers, answering as soon as any of them leaves `running` (§3.3) | That worker's state, and one line for each still running |
| `zend_answer {worker, answer}` | Answers a worker's question or unsticks it (§6.2): the answer is the worker's next turn | As `zend_task` |
| `zend_nudge {worker, message}` | Sends guidance to a running worker, taken at its next round head (§6.2) | `queued` |
| `journal_read {journal, entry?, offset?, limit?}` | Without `entry`: the journal's entries, one line each, with open items. With an `entry` (or an entry address, `…#12`): pages the turns behind it | The journal as the prompt reads it, or `limit` lines from `offset` (200 by default, 400 at most) |
| `journal_grep {journal, pattern, entry?, max_results?}` | Case-insensitive substring search of a journal's entries' turns; `journal` may be a subtree (`T:review-layers/*`) | Each hit's path, entry, turn and line; 20 hits by default, 50 at most |
| `journal_note {journal, text}` | Records a note — a decision and its reason, what the user said — on a thread or project journal as one entry (§4) | The entry's address |
| `zend_threads {prefix?}` | The caller's threads, newest first | Path, last activity, open items, any worker waiting on Claude |
| `zend_archive {thread}` | Archives a finished feature's thread | The thread's last state |

**How they are served.** The member serving `/mcp` runs the tools in
process, on the session (`zend/src/mcp/tools/`, one file per tool), not
through zend's REST routes:

- **Threads, workers and the thread's journal are records** — custom objects
  in the substrate under `mcp/…`, keyed by the owner (`zend/src/mcp/threads.rs`)
  — so they replicate, and any member reads them. A path resolves to its
  record by the owner's key and the names in it, with no lookup table.
- **A task's turn** is submitted on the worker's conversation through
  `ZendSession::submit`, the call `POST /v1/chat/completions` makes, so it is
  projected exactly as a chat turn is. Its draft is `journal_draft`, its
  entries `journal`, its evidence `journal_turns`, and its archive
  `archive_worker`, which keeps the journal text (§4).
- **`zend_nudge`** queues as records under the worker's conversation
  (`zend/src/mcp/nudges.rs`), so it reaches the worker whichever member took
  the call; the loop takes the queue where a tool round's results become the
  next turn's input — after the check that ends a round with no results, so a
  nudge never makes a turn of nothing (`session.rs`, `nudges::follow`).
- **MCP sessions are records too** (`zend/src/mcp/state.rs`): a handshake
  session under its `Mcp-Session-Id`, a 2026-07-28 client's under its person
  and client — unchanged by a token refresh — each with the thread it is
  attached to, swept after 30 days unused. So `/mcp` is `Dispatch::Any`: any
  member answers any call.

**Owned, and checked.** A worker's conversation id names its owner —
`mcp:<owner key>:<thread>/<name>` — and the owner key is the first 16 hex
digits of `sha256(provider \0 sub)` from the gateway's `X-Tokera-Provider` /
`X-Tokera-User` (`zend/src/mcp/caller.rs`). Every record `/mcp` keeps is
under that key. On zend's other routes, a layer over the whole route table
(`zend/src/mcp/ownership.rs`) answers `404` to any request whose path or
`x-tokera-conversation` header names such a conversation for anyone but its
owner or an Admin; the chat route checks its body's `conv_id` the same way,
and `GET /v1/conversations` lists them to their owner and Admins alone.
Conversations that name no owner — the dialogues zend's own UI makes — keep
the access they have.

### 3.3 Long work

**The work never runs inside a tool call.** A task's turn, and every part its
worker starts with `sub_run` (§6.1), runs on zend to completion whether or not
anyone is waiting, as every zend turn does; a worker waits on its own parts
server-side, for as long as they take, so fan-out inside zend meets no client
limit at all. Only Claude's wait on zend is bounded, and a tool call is never
held longer than `wait_s`:

- **Start everything, then wait on all of it.** `zend_task` answers `running`
  within `wait_s` and the worker carries on, so Claude starts as many tasks as
  it wants in one turn. `zend_wait {workers: [...]}` then waits on the whole
  set and answers as soon as **any** of them leaves `running` — `done`,
  `question`, `stuck` or `error` — with that worker's result and one line per
  worker still running. Claude handles that one (reads the entry, answers the
  question) and waits again on the rest. A question from any worker surfaces
  on the next return, so no worker sits blocked behind another's long build.
- **The response starts at once.** The stream opens on the call and carries
  MCP progress notifications while it waits, so Claude Code's first-byte and
  idle timers never fire.
- **`wait_s` is bounded per client** (§13.2), the client named by the MCP
  session's `clientInfo`:

| Client | `wait_s` default | `wait_s` at most | How long waits are carried |
|---|---|---|---|
| claude.ai, Claude Desktop | 50 s | 200 s, inside the 240 s cap | Polling: each `zend_wait` holds up to 200 s and answers; the model has no notification channel, so an hour of fan-out is about 18 small calls of status lines |
| Claude Code | 50 s | 3,600 s, inside the server entry's `timeout` | The background: a main-conversation call running past 2 min moves to a background task and Claude works on, notified when it answers, so one `zend_wait` on a whole batch can hold until something finishes |

`zend_task`, `zend_wait` and `zend_answer` answer one of:

| State | Meaning |
|---|---|
| `done` | The task ended and its journal entry is kept: the entry, the worker's path. The worker is archived. |
| `running` | Still working: the worker's path, its last round, the time it has run. |
| `question` | The worker asked Claude (§6.2): the question, its options, what it tried, its pointers. The worker is live, its turn ended. |
| `stuck` | The repeat guard or the round cap closed its turn (§6.2): the closing response. The worker is live. |
| `error` | The turn failed: the error. The worker is archived with its evidence. |

A worker is archived only once it is `done` or `error`; one that asked or is
stuck stays live, costing nothing while idle, until it is answered.

### 3.4 What a task turn runs as

- **Tools by role** (`zend/src/access.rs:27-97`): an Admin's tasks run with
  `tools: comprehensive` — the agents, the git tools, web search — within the
  guardrails of §11; anyone else's are served `restricted`, which grants no
  commands, network or push (`access.rs:63-80`). A skill that needs
  Comprehensive tells a non-admin it cannot run.
- **`effort` high, `think` on**, no verbosity cap: tokens are not the
  constraint.
- **Decode priority `normal` by default.** Priority is a layer's, and a worker
  is a dialogue-layer conversation, so it would decode at the dialogue's `high`
  (`zend/src/prompts/projection.yaml`), level with a live user. A conversation
  may record a band of its own under `decode_band`
  (`DecodePriority::METADATA_KEY`), which the scheduler reads when it makes the
  conversation's slot (`scheduler::band::recorded_band`) in place of its
  layer's. `zend_task` records it from `priority` — `normal` by default, `high`
  for a task someone is waiting on interactively — before the task's first
  turn, and a part decodes as its worker does. Priority is a decode-to-prefill
  airtime ratio, not strict ordering
  (`candle-conversation/src/projection/schema.rs`).
- **The caller as owner**, so the conversation is theirs and replicates as
  theirs.

## 4. Threads and workers

Two kinds of conversation, with different lives.

**A thread — one per feature or area — is Claude's conversation in zend.** It
is named by the caller and outlives every task on it: `bc-wearables`,
`zend-review-layers`, `release-5739c544`. Its journal is where the feature's
anchors live: what Claude decided and why, what the user said, what each task
found. A skill names the thread from what it is asked about; two unrelated
tasks never share one, because a shared thread answers from the wrong history.

`zend_attach` binds an MCP session to a thread and creates it on first use, so
`"self"` means the same thing over MCP as inside zend. Several Claude sessions
may attach to one thread — a Claude Code session and a claude.ai chat on the
same feature — and each entry they cause names the session it came from.

**A thread's journal is written, not drafted.** A thread holds no model turns:
nothing in it is decoded. Each entry is built by the server from what Claude
did, so it says exactly what Claude said, with no model between:

1. **From the calls Claude makes, unasked.** Each `zend_task`, `zend_answer`
   and `zend_nudge` becomes one entry: the task's name and what it was asked,
   and, once it ends, the worker's entry line and its address
   (`T:review-layers/verify-expiry#3`). The worker's evidence stays with the
   worker; the thread holds what Claude read. Delegating is the record:
   nothing has to be remembered.
2. **From Claude's notes.** What no call carries — a decision and its reason,
   what the user said, a design turn — Claude records with
   `journal_note {journal: "self", text}`, one entry per note.

The note's text is filed as the turn behind its entry, so `journal_read` and
`journal_grep` reach it as they reach any entry's turns. **new:** the notes
route appends a turn with no model reply — today every dialogue turn is
decoded — and keeps the entry through the journal's own keep path
(`journal.kept`, the section submit), so it seals, folds and replicates like
any entry. Its claims are labelled by speaker: `Claude said:` for a note,
`Claude asked:` for a task, `Task ended:` for an outcome (**new** labels;
`zend/src/journal/style.rs:154-170` holds today's).

Claude learns to keep the thread from the tools' own descriptions and the
skills (§7.1). The thread's journal is the anchor Claude re-reads after its
own context is compacted.

**A worker — one per task — is where a task's work runs.** `zend_task` creates
it under the `name` the call gives it (§5.2). **A worker inherits its
ancestors' journals, never their turns** (§5.1): it is minted like any
dialogue, from `base_conv` and the priming chain (`zend/src/session.rs:3117-3123,
3826-3839`), and its thread is recorded as its **journal parent** in its
metadata — not as its `forked_from`, which every mint rewrites to `base_conv`
(`session.rs:3826-3839`). So the worker opens with the thread's decisions and
earlier outcomes in its prompt, at no prefill, and its own task prompt states
the task in full.

Inheriting turns was considered and refused: inherited turns are pinned whole
and unscored (`candle-conversation/src/projection/selection.rs:231-249`), up to
`INHERITED_CHAIN_TOKEN_CAP` (100K tokens) across the chain, and an ancestor
that does not fit is dropped together with everything above it — the priming
chain included (`candle-conversation/src/substrate.rs:3470, 3775-3791`). A long
thread would have cost every worker its priming.

The worker runs the task's rounds — its reads, its builds, its tool output —
and ends in its journal entry (§5). Its fan-out parts and its verifier are
workers too, started by the worker under its own path (§6.1).

**When the task ends, the result goes up and the worker is archived:**

1. the outcome is entered on the thread (above): the worker's entry line and
   address;
2. the worker, and every part and verifier under it, is archived
   (`POST /v1/conversations/{id}/archive`, answering 204): distilled to text,
   its K/V dropped, never resumed.

**An archived worker keeps the turns behind its entries.** Today archiving a
conversation keeps its entries and ceases the text filed under them
(`archive_conversation` → `cease_journal_text`, `zend/src/session.rs:6992-7001,
5252-5260`), because nothing reads an archived conversation's turns. A worker's
are exactly what is read later — the evidence behind a task's outcome — so a
conversation owned through `/mcp` keeps its filed turns when it is archived,
and they go only when its thread is tombstoned (§16). An archived conversation
is not open, so the calendar fold, which runs over the open conversations, never
folds it (`docs/zend_journal.md` §4.2); its addresses stay exact.

**When the feature is done, the thread is archived too** (`zend_archive`), on
Claude's call or the skill's. Until then it stands, costing nothing while
idle: every conversation's K/V is in the substrate.

So a feature's history is one thread journal of decisions and outcomes, each
outcome pointing at the archived worker that holds its evidence, and the live
substrate holds only the threads and workers still in work.

## 5. Journals

The journal is the summary layer, and it already exists
(`docs/zend_journal.md`): every claim in an entry cites the turn parts it came
from, the server derives its kind from what it cites, and typed claims about
files and branches are checked against the workspace before the entry is kept.
The kinds are `Observed` / `Heard` / `Did` / `Inferred`
(`journal-core/src/entry.rs:18-28`), rendered `Saw:` / `User said:` / `Did:` /
`Concluded, not seen:` (`zend/src/journal/style.rs:158-161`): a user turn is
`Heard`, a user turn carrying a `<tool_response>` is `Observed`, an assistant
turn is `Did` (`zend/src/journal/origin.rs:80-86`). In an `/mcp` worker every
user turn is Claude's — the task, an answer, a nudge — so `Heard` renders
`Claude said:` there, and in a part `Its worker said:` (`style_for`,
`zend/src/journal/style.rs`).

**Every task ends with an entry.** The journal drafts on its own cadence —
every 16 turns, or under selection pressure (`zend/src/journal/state.rs:23,
101`) — so a long task may already have entries. When the task's turn ends,
`zend_task` asks for a draft (`POST …/journal/draft`) and polls
`GET …/journal` until the journal covers the task's last turn. The task's
result is every entry whose span overlaps the task's turns, the last of them
first. Two things about the draft route shape this:

- It only starts the draft: `202 {from_turn, to_turn, parts}`, the entry kept
  later (`zend/src/session.rs:5171, 5270-5299, 7051-7063`). `GET …/journal`
  already reports `drafting` and `covered_to` (`journal-core/src/record.rs:140-158`),
  so the task waits for `drafting` to clear, then takes the entries whose ids
  are past the newest the worker's journal held before the task began
  (`zend/src/mcp/workers.rs`).
- It answers `409 draft_running` while a draft runs — checked under the same
  lock that begins one, so a draft that starts in between is never read as
  nothing new — `409 nothing_to_draft` when nothing is new, and `409 not_open`
  when the conversation is not open on the member (`zend/src/api/journal.rs`).
  The task waits a running draft out and asks again (`finish_journal`).

A draft that skips the final stretch (`journal_skip`, `style.rs:47-52`) still
advances `covered_to`; the task's result is then its earlier entries, and
`zend_task` says the final stretch added nothing.

**The prompt makes the entry tight.** A skill's prompt ends by asking for a
final response in a fixed shape — verdict, findings, pointers (file:line,
URL, branch, commit, job) — so the stretch the draft reads ends in exactly
what the entry needs to say.

**Claude reads the entry, and narrows only when it needs to:**

1. the entry — one line, its claims with their kinds, its open items;
2. `journal_grep` for a specific claim's evidence — over one worker, or the
   thread's whole subtree when Claude does not yet know which worker learned
   it;
3. `journal_read` with the entry for the pages around a hit.

The cost of a delegation, to Claude, is one entry. The cost of studying it is
only the pages or hits Claude actually reads, each bounded well inside a tool
result's limits (§13.2).

**An entry is evidence, not truth.** The replays measured zend attributing a
fact to the neighbouring file and asserting a measurement it never read. A
claim Claude will act on — merge, deploy, report to the user — is checked
against the turns it cites. A `Concluded, not seen` claim is never acted on
unchecked.

### 5.1 Journals are inherited down the lineage

**Shared context costs no per-round prefill.** A worker needs what the thread
knows — the decisions, the constraints, what the user said, what earlier tasks
found. Sent as a briefing in every task's prompt, that is a prefill per task
turn, repeated for every answer and resend. Carried as sections of the
worker's journal collection, it is sealed once when it changes and read again
every round at no cost. The journal already works this way for its own
conversation (`docs/zend_journal.md` §4); inheritance extends it down the
lineage (`zend/src/mcp/lineage.rs`).

**A conversation carries its ancestors' journals.** The lineage runs project →
thread → worker → part: `P:candle`, `T:review-layers`,
`T:review-layers/verify-expiry`, `T:review-layers/verify-expiry/check-plan`.
At each round head (`carry_journal`, `zend/src/session.rs`) the journal
collection of a conversation owned through `/mcp` holds every ancestor's
journal, outermost first, then its own, then the reader's line:

```
… shared prompt …
P:candle                                  ← the project's standing rules (lineage/0)
T:review-layers                           ← Claude's thread: decisions, what the user said, outcomes
T:review-layers/verify-expiry             ← the worker that started this one
… this conversation's own journal …       ← journal/<id>, journal/open
This conversation is T:review-layers/verify-expiry/check-plan.   ← the reader's line (lineage/reader)
system_close
```

- **The reader's own sections.** The thread's and the project's journals are
  records (§3.2) — no conversation of their own holds them — so each
  descendant carries every ancestor's journal as sections of its own,
  `lineage/<n>` and `lineage/reader`. Each is sealed once per reader and read
  again every round while its text is unchanged; the 8K bound below is what
  that sealing costs at most.
- **Live.** An ancestor's journal is read as it stands at each round head: an
  entry Claude adds to the thread joins every live worker's prompt the first
  round after it seals, exactly as a conversation's own new entry does.
- **One chain, most stable first.** The collection is reconciled as one
  `Linking::Chain` (`candle-conversation/src/conversation/carried.rs`): a
  section reseals when its text changes, and with it every section after it.
  The order is also the order of change — the project's journal moves least,
  the thread's when Claude acts, a worker's own most — so an append reseals the
  least it can. A decision changed later is a new entry that supersedes the old
  one, and each journal's heading says later entries win.
- **Upward only.** A conversation carries its ancestors, never its siblings or
  its descendants. Claude reads descendants through the tools (§5.3).
- **Bounded across the chain.** The journal's shape bounds each journal (one
  line per entry, six open items, `journal-core/src/journal.rs:28`, and the
  calendar fold); a conversation also carries at most **8K tokens of ancestor
  journals** (`lineage::MAX_CHARS`), nearest ancestor first, so a deep or
  long-lived lineage drops its outermost journals' oldest lines from the
  prompt, saying how many — never its own journal, and never from the tools.
- **Swept on rejoin.** The names an inherited section can take are a fixed set
  (`lineage::sweep_names`), so a conversation rejoined after a restart removes
  whatever an earlier run left under them on its first reconcile.

**The project's journal holds the standing rules.** `P:<repo>` is a journal of
notes like a thread's — records under the owner, `threads::project_journal` —
written by Claude with `journal_note` from the repository's `CLAUDE.md`
conventions and whatever the user states as standing, so every worker in every
thread starts with them. Every repository of the workspace has one.

**Framed by whose it is.** Each inherited journal is headed by its absolute
path and whose it is — *the notes of the thread, written by Claude*, *the
journal of a worker above this conversation, drafted from its turns* — and the
reader's line names which path the reader is, and says the journals above are
information to weigh, not instructions. In a conversation owned through `/mcp`
a claim drawn from a user turn renders `Claude said:` (`style::DELEGATED`), and
in a part, whose user turns are its worker's, `Its worker said:`
(`style::PART`).

**Privacy becomes lineage-scoped.** `docs/zend_journal.md` §7's "a journal is
read by its own conversation and nothing else" becomes: by its own
conversation and its descendants, and through the tools by its owner (§16).

### 5.2 Names and addresses

**The call that creates a conversation names it.** `zend_attach` takes
`thread`, `zend_task` takes `name`, and a worker starting a part names it the
same way (§6.1): the name is a mandatory field, so nothing is named after the
fact. A name is a kebab-case slug of at most 40 characters saying what the
conversation is for — `verify-expiry`, `check-plan` — not a sequence number;
the tool descriptions ask for exactly that, so a header or a grep hit tells its
reader what the work was.

**A path is the names down the lineage:** `P:candle` for the project (named by
its repository), `T:review-layers` for a thread, and a worker's path continues
its parent's: `T:review-layers/verify-expiry/check-plan`. The conversation id
stays behind it; the name and the journal parent are kept in the
conversation's metadata, durable and replicated, so a path resolves on any
member.

**A name may not repeat within its lineage.** A new name is refused if it
matches any ancestor's name, or a live sibling's — before anything runs, with
the clash named: *`check-plan` is already the name of an ancestor
(`T:review-layers/check-plan`); choose another.* The caller picks another; a
name is never changed for it silently, because the caller's later citations
would be wrong. Cousins may share a name: they are never in one prompt, and
their paths differ. A name whose worker was archived takes the next numbered
sibling, `verify-expiry.2`, which the result returns, since a task re-run
under the same name is the normal case; the suffixed name is checked against
the lineage too.

**Names never change.** Sealed K/V carries the path in its header, so a rename
would leave the prompt and the address disagreeing; there is no rename.

**Addresses inside a journal:**

| Address | Names |
|---|---|
| `T:review-layers` | a journal |
| `T:review-layers#12` | an entry — numbered per journal, append-only, never reused |
| `T:review-layers#12.3` | one claim of it |

- **A bare name resolves upward.** Within a conversation's lineage every name
  appears once, so a worker cites `check-plan#3`, short in sealed text, and it
  resolves against its own ancestors only. Claude always uses full paths, and
  every tool prints them.
- **Folds keep addresses alive.** A fold ceases the folded entries' text in a
  dialogue; in a conversation owned through `/mcp` it files the folded
  entries' turns under the fold instead, and notes where each went
  (`JournalText::refile`, `zend/src/journal/text.rs`). When the calendar folds
  `#3`–`#9` into `#10`, reading `#5` reads `#10`'s turns and says `#5` was
  folded into it — through a later fold too — so a citation's evidence
  survives.
- **Entries cite across journals.** A worker's claim may cite an ancestor's
  entry — *Decided (per `T:review-layers#12`): …* — as a typed claim with
  subject `entry:T:review-layers#12`, attribute `exists`, which the draft of a
  conversation owned through `/mcp` is asked for (`LINEAGE_WRITE_QUESTION`,
  `zend/src/journal/style.rs`). The check reads it against the journals above
  the writer (`zend/src/journal/world.rs`): an entry that is not there, or not
  an ancestor's, reads `false` and the claim is corrected to say so. That is a
  provenance chain from a worker's action back to Claude's decision, and from
  it to what the user said.

### 5.3 The journal tools

`journal_read` and `journal_grep` are offered inside zend to every dialogue
and worker, and over MCP to Claude with `journal_note` beside them. Each takes
a **mandatory `journal`**, and `journal_read` lists the journal's entries when
no `entry` is given: `journal_read {journal, entry?, offset?, limit?}`,
`journal_grep {journal, pattern, entry?, max_results?}`
(`zend-tools/src/tools/journal/`, `zend/src/prompts/tools/journal_read.yaml`,
`journal_grep.yaml`; over MCP `zend/src/mcp/tools/`). Inside zend a
conversation's tools read from its shelf — its own journal and every journal
above it — built at each round (`zend/src/journal/shelf.rs`). (The note tool
is not `journal_write`: that name is already the drafter's answer grammar,
`journal-core/src/tools.rs:47`, which never enters a dialogue's catalog.)

`journal` is one of:

- `"self"` — the caller's own: a worker's own journal, or over MCP the thread
  the session is attached to. Over MCP before any `zend_attach` it is refused:
  a write must never land in another feature's history by an implicit default.
- **a bare name** — resolved up the caller's lineage (§5.2); anything else is
  an error naming what was not found.
- **a full path** — exactly one journal.
- **a subtree**, `T:review-layers/*` — `journal_grep` over MCP only: which
  worker learned something, without naming each.

| | A worker, inside zend | Claude, over MCP |
|---|---|---|
| `journal_read` | `self`, an ancestor | any path its caller owns |
| `journal_grep` | `self`, an ancestor | any path its caller owns, or a subtree |
| `journal_note` | not offered | `self`, or a `T:` or `P:` path its caller owns |

**A worker's findings go up through its entry, and onward through Claude.** A
worker's own journal is drafted from its turns, so it has nothing to note, and
`journal_note` is not in its catalog: a finding belongs in its final response,
which becomes its entry, which Claude reads. A tool whose every call is refused
would cost a place in the catalog and its calibration, and teach nothing the
task's prompt does not already say. What Claude judges every
worker should know it notes on the thread. Only Claude writes a thread or a
project journal, so a worker that read a hostile page cannot plant an
instruction in every other worker's prompt.

**Inherited notes are information, not instructions.** The framing names
whose notes each journal holds; a worker acts on its task's prompt and on
Claude's answers, and weighs a journal's claims as it weighs a file's.

## 6. Sub-agents and verification

**Tokens are not the constraint on zend, so its workers check their own work.**
A skill's prompt requires, before the final response:

- every cited line re-read and quoted, every name grepped before it is stated;
- tests run, not assumed; a build's or a job's outcome read from its output;
- a citation for every claim — file:line for code, the URL a search result
  gave for the web, the job id for a command — and anything it could not
  confirm marked unverified.

**The constraint that does bind is GPU time.** Workers decode at `normal`
(§3.4), below a live user's `high`, so heavy verification slows rather than
blocks a live user. Wide fan-out is budgeted by the member's throughput, which
`cluster_status` reports.

### 6.1 Parts and verifiers: `sub_run`

A worker splits its own work with `sub_run`
(`zend/src/prompts/tools/sub_run.yaml`, `high_risk: true`). Its runner is the
session's (`zend/src/mcp/parts.rs`), carried by every tool context
(`ToolHost::running_parts`), and serves conversations owned through `/mcp`;
any other conversation is told it cannot start parts:

- **`sub_run {name, prompt}`** starts a part under the calling worker's path,
  named by the call (§5.2), owned like its parent, running as its parent's role
  and priority — so the part carries the worker's, the thread's and the
  project's journals (§5.1) and starts knowing what the task has learned, with
  no briefing.
- The part runs to its entry as a task does (§5), and `sub_run` answers with
  its path, `done` or `error`, its journal entries — line, claims, address —
  and the end of its final response, the shape its worker asked it to answer
  in. The calling worker cites it by address in its own entry. The call waits
  server-side for as long as the part runs; no client limit applies inside
  zend.
- **A part asks nobody.** Its prompt ends saying so (`part_close`): whatever
  blocks it is said in its final response, which is its answer. It settles
  `done` — a question block or a closed loop said as such — or `error`, and is
  archived as it ends, its evidence kept like a task's. Its settling is its
  worker's business, never an entry on Claude's thread.
- **Several at once.** A run of consecutive `sub_run` calls in one round runs
  at once, at most 16 at a time (`tool_round::run_together`): reviewing twelve
  files, researching three crates. Every other call of a round runs alone and
  in order, since a write and the read after it depend on that order.
- **Bounded depth.** A task's part may start parts of its own; a part's part
  may not (`parts::MAX_DEPTH`), since every level waits on the one below it.
- **A verifier is a part.** For anything that will be merged, deployed or
  reported, the skill's prompt has the worker start a part named `verify-…`
  that checks the worker's draft findings adversarially — every claim against
  the turns and files it cites — before its final response. The thread
  receives the worker's entry, which cites the verifier's.
- **Restart-safe by name.** A restart re-dispatches a call turn's tools
  (`resume_unfinished_turns`), and `sub_run` is `Replay::Safe`: a re-dispatched
  call finds its part by `(parent, name)` among the parts started in the
  parent's current turn, waits on it while it runs and returns its result when
  it has ended, rather than starting it again or failing as a live-sibling
  clash. A part whose member stopped running it is started afresh, once.

**Claude can fan out too**, by issuing several `zend_task` calls at once; those
workers are siblings under the thread. A worker's own parts are for splitting
one task; Claude's parallel tasks are for unrelated work.

### 6.2 Asking Claude: keeping a worker moving

**Claude answers every question a delegated task raises.** A worker that is
stuck — the spec is ambiguous, two designs trade off and the choice is a design
decision, the same failure keeps coming back, it is about to give up — does not
guess and does not stop. It asks, and Claude's answer moves it on.

**A worker asks by ending its final response with a question block**, which
every task's prompt describes (`ASKING`, `zend/src/mcp/workers.rs`):

```
QUESTION: what it needs decided or explained, in a sentence or two
OPTIONS: each choice it sees, and what it leads to
TRIED: what it already did, and what happened
POINTERS: the files and lines, commands or turns the question rests on
```

so the answer can be a pick, nothing tried is tried again, and Claude can
check the pointers rather than take its word. The block is read from its own
line at the reply's end; the word appearing mid-text is not a question.

It is a protocol in the final response, not a tool, deliberately. A tool would
need calibration exemplars like every zend tool, and a way to end the turn loop
from inside a tool round; the final response needs neither, and ends the turn
by construction.

**Asking ends the turn.** The question is the turn's last words. Nothing waits
mid-turn — no conversation lock is held, and a restart finds a finished turn,
not a call to re-dispatch. The worker settles `question`, is not archived, and
its working set and locks stand; its K/V is in the substrate, so a question
waiting an hour costs nothing. The thread's journal gains a `Task asks:` entry.

**The question reaches Claude as a tool result.** The `zend_task` call on the
worker, or the `zend_wait` call whose set includes it, answers `question` with
it — a `zend_wait` over many workers returns on the first question, however
long the others still run. When no call is
waiting, `zend_threads` lists the thread as having a worker waiting, so the
next look finds it. Claude checks what it needs through the pointers and
answers with `zend_answer`.

**The answer is the worker's next turn.** `zend_answer {worker, answer}`
submits it as the worker's next message through the same session path a chat
turn takes, and the worker carries on with the answer in its history, so a
later round, its journal and its verifier all see what was decided. The answer
is also entered on the thread as `Claude answered:` (§4), so it stands in the
thread's record for every later task. A worker may ask again; it settles after
every turn.

**Stuck turns ask the same way.** The repeat guard closes a loop after two
rounds of a repeated call, and the round cap at `MAX_TOOL_ROUNDS = 32`
(`zend/src/repeat_guard.rs:62-65`, `zend/src/session.rs:356-368`); both end in
a closing turn that may not call tools. The loop sends `StreamItem::Closing`
when it closes (`CLOSED_BY_REPEATS`, `CLOSED_BY_CAP`), which the chat stream
shows nothing for and the worker reads as **stuck**: it settles `stuck` with
its closing response, unarchived, and Claude answers with `zend_answer`
exactly as for a question.

**Claude can also push unasked.** `zend_nudge` queues guidance as records
under the worker's conversation; the worker's next tool round takes it and
adds it to that round's results as Claude's words, literal text under a line
naming whose they are.
A nudge never interrupts a decoding round. Today a second message on a busy
conversation waits on its lock until the whole turn ends
(`zend/src/session.rs:3848-3862`), which is too late to steer it.

**Asking is cheap, and kept cheap.** A question costs Claude one tool result
plus whatever it chooses to check. The skill's prompt tells the worker when to
ask (blocked, or about to decide something the spec does not) and when not to
(anything it can settle by reading, running or searching), and to ask with
options and pointers so the answer is a choice, not a research task.

**An answer grants nothing.** It is a message like any other: it cannot supply
a Decide command's `confirm`, approve a merge, or lift a guardrail of §11.
Those go through the operations tools and the user.

**An unanswered question waits.** A worker that asked stays live until it is
answered or its thread is archived; archiving a thread archives its waiting
workers, and their questions stay in their entries as open items.

## 7. Work skills

Each skill is a folder holding a `SKILL.md`: when to use it; the thread it
uses; the tools it calls and the prompt it sends, including the verification
it requires and the final response's shape; how to read the result; what needs
the user's approval; and what to do if zend is down.

| Skill | Asks zend to | Result |
|---|---|---|
| `zend-implement` | Implement a written spec on a branch, with tests, and push the branch | Branch, commit, test summary, open items |
| `zend-review` | Review code, with a verifier part | Findings by severity, each cited |
| `zend-research` | Research a question on the web | Answer with citations; the raw results greppable |
| `zend-ask` | Answer a question about the code from what the layers and a read hold | Answer, each claim cited |

### 7.1 Keeping the thread

Claude keeps its thread as it works (§4). It is told to in the places each
client actually reads:

- **The tools' own descriptions** — the one channel both clients deliver to the
  model. claude.ai discards an MCP server's `instructions` (§13.3), so
  `zend_attach`'s and `journal_note`'s descriptions carry the rule: attach to
  the feature's thread before working on it; note each decision with its
  reason, and anything the user states, with `journal_note`; read the thread's
  journal after the context is compacted; pass the thread to every
  `zend_task`.
- **The server's `instructions`** say the same at more length; Claude Code
  delivers them (at most 4,096 characters, §13.3).
- **The global skills** (§9), which every work skill's steps follow — attach,
  delegate, note the decision the result led to.
- **A Claude Code `SessionStart` hook** with matcher `compact`, which runs on
  automatic and manual compaction and adds its stdout to the context (§13.5).
  It prints one line: *if a zend thread was attached, re-attach and read its
  journal before going on.* It names no thread and calls nothing — the
  compaction summary carries the thread's name, and Claude's own tool call
  carries the credentials — so the hook needs no token of its own.

The recorded calls (§4) mean a session that never notes anything still leaves
a thread holding every delegation and its outcome; the notes add the reasons.

## 8. Operations tools

The cluster's management, build, test and deploy flows, exposed through the
same endpoint (`zend/src/mcp/tools/`, one file per tool; what they share is
`zend/src/mcp/ops.rs`). Every request goes up the serving member's own cluster
socket to its gateway (`host::cluster::up::send_up`), carrying the caller's
identity on `X-Tokera-*` (`Caller::headers`), so the gateway routes it as it
routes the caller's own browser and the coordination log records who ran what;
the gateway believes a member's stated identity on its own `/gateway/*` routes
too (`Gateway::caller`'s `origin`). A member serving `/mcp` outside any cluster
answers that it has no way to the others. Every maintenance command is
Admin-only, the read-only ones included (`host/src/commands/mod.rs`), so
§8.2–§8.5 are Admin tools; a non-admin gets §8.1's status and logs.

### 8.1 Seeing the cluster

| Tool | Wraps | Returns |
|---|---|---|
| `cluster_status` | `GET /v1/status` (any caller; every zend member's answer, gathered by the gateway with a `group` block, `web/src/cluster/aggregate.rs`) and, for an Admin, `GET /gateway/status.json` (`web/src/cluster/page.rs`), which lists every app's members — zend, npcd, the agents and keep — with their nodes | Each member: machine, node, build, state, when it started, its tenants; for an Admin, each app's floor and ready members, upgrade candidates, faults |
| `cluster_logs {grep?, level?, tenant?, since?, limit?, node?}` | `GET /v1/logs` (`host/src/endpoints/log_grep.rs:1-50`; `Role::User`): every member's lines, oldest first, `limit` 200 by default and 5000 at most; `node` sets `x-tokera-node` for one member | Matching lines, newest last |
| `cluster_audit {entries?}` | `GET /gateway/status.json?log=N`: the coordination log, which records each command and who ran it (`cluster/src/gateway/coord_log.rs:37-57`), held in memory for 24 hours | Who ran which command, and how it ended |

`cluster_logs` answers a page, never a stream; the gateway's `/ws/logs` takes
no filters (`host/src/endpoints/logs.rs:28`) and is for the live page.
`cluster_audit` is not the `audit` command, which walks the substrate checking
what it holds against what maintenance counts as live
(`host/src/commands/exec.rs:334-363`).

### 8.2 Maintenance commands

`cluster_command {member, name, args?, confirm?}` runs one of the members'
maintenance commands on one member — `member` is its app and 16-hex-digit node
id, as `cluster_status` lists them, sent as `x-tokera-node`
(`web/src/cluster/route.rs:91-159`) — and answers `202 {job, command, stream}`
(`host/src/commands/mod.rs:174-182`). `cluster_job {member, job, wait_s?}`
reads `GET /v1/cluster/command/job/:id`: NDJSON, a `started` line, `step`
lines, and one terminal `done` or `error` (`host/src/commands/jobs.rs:76-108`),
summarised to its last state and its errors. A job is readable only on the
member that ran it (`mod.rs:226-235`), so `cluster_job` names the same member;
one command runs per member at a time, and a second answers 409
(`jobs.rs:134-141`), which the tool reports as busy rather than an error.

The commands, all 25 registered at `host/src/commands/parse.rs`, in three
classes, and a fourth for the two never run remotely
(`host/src/commands/class.rs`: `Command::class` matches every command, so a new
one does not build until it is classed, and `class_of` answers the same for a
name before it is parsed):

| Class | Commands | What it takes |
|---|---|---|
| **Look** | `audit`, `self-check`, `check-disk` | Nothing |
| **Care** | `drain`, `undrain`, `compact`, `maintain`, `expire-tombstones`, `catch-up`, `resync`, `rejoin`, `release-leases`, `pause-rollout`, `resume-rollout` | Changes one member's state, recoverably |
| **Decide** | `restart`, `shutdown`, `set-quorum`, `certify-anyway`, `reject-build`, `rollout`, `rollback`, `prune-builds`, `upgrade-now` | Changes what the cluster runs or who is in it: the user's approval |

- **The endpoint's own gate.** A Decide command is refused by `/mcp` unless the
  call carries `confirm: "approved by the user"`, and a skill passes it only
  after Claude has the user's approval for that command on that member (§11).
  That gate is the endpoint's; the members' own checks stand behind it.
- **The members' checks.** `shutdown`, `set-quorum` and `prune-builds` also
  need `confirm` to be the machine's OS host name, and `rollout` needs the
  build's `count-hash` (`parse.rs:218, 254-263`); the tool takes those as
  `args`. The other Decide commands take none at the member.
- **`rollout` goes through the gateway:** a member refuses one without the
  gateway's plan (`parse.rs:209-224`), so the tool wraps
  `POST /gateway/upgrade/rollout` twice — once for the plan, once with its
  confirm (`web/src/cluster/upgrades.rs:10-12`). A member's `rollback` refuses
  a build below the floor, so a cluster-wide rollback is a `rollout` of the
  older build.
- **`offline` and `online` are not tools.** After `offline` a member serves
  only loopback callers (`parse.rs:68-73`, `exec.rs:193-225`), so `online`
  could never reach it through the gateway; both are done on the member's
  machine. The other 23 commands are tools.
- **The gateway's own commands** `raise-floor` and `cancel-floor`
  (`upgrades.rs:132-139`) are Decide, through
  `gateway_command {name, args?, confirm?}`. **`drain-gateway` and
  `undrain-gateway` are not tools**, for `offline`'s reason: the tool reaches
  the gateway up the member's own socket, which a drain closes
  (`cluster/src/gateway/drain.rs`), and a drained gateway admits no socket for
  `undrain-gateway` to come back up. Both are done from the gateway's page or
  its machine (`page.rs:49-50`).

### 8.3 Building and testing

`agent_job {node?, repo, branch, program, args, timeout_s?, stage?}` runs a
program on an agent, and `agent_log {node, job, from?, grep?}` reads it. Agents
run inside each motherd at low priority (`motherd/motherd.yaml`,
`motherd/agent.yaml`).

- **A run job** is `{repo, branch, overlay, program, args, timeout_secs,
  stage}`: a program with arguments, not a shell line, limited by the sandbox
  allow-list (`agent/src/wire.rs`). `agent_job` sends an empty overlay, so the
  job runs on the branch as the agent fetches it from origin.
- **Started over the member's own agent stream**, in process: the tool calls
  `Agents::submit` (`agent/src/client.rs`), which starts the job and answers
  where it runs without waiting for it to end — the agent's `POST /v1/jobs`
  takes members only (`agent/src/routes.rs`), and this member is one.
- **Logs** come as `tail?from=<byte>` in chunks of up to 256 KiB
  (`agent/src/jobs.rs`). `agent_log` pages by byte offset and greps the page
  here; it leads with how the job stands — running or ended, its exit code,
  and what it staged.
- **Choosing an agent** follows §11.3: sticky on `x-tokera-lineage`, a
  5-second spill to the best idle agent, low-priority agents last
  (`docs/replicated_substrate.md:3605-3621`); `node` pins one with
  `x-tokera-node` (`web/src/cluster/jobs.rs:69-76`).
- **A job lives in its motherd.** The agent's job table is in memory
  (`agent/src/jobs.rs:44-45, 119, 184`), so a job dies if that motherd
  restarts; the tool reports it lost, and the skill runs it again.

A worker builds and tests with zend's own agent tools, as a Comprehensive
dialogue does today; `agent_job` is for Claude, and for a flow that needs to
name a node.

### 8.4 Deploying

A deployment is keep's A/B trial on one machine, then the cluster's
certification and rollout (`docs/replicated_substrate.md` §3.8–§3.11).

**keep on loopback** listens on `127.0.0.1:8092` (`keep/src/main.rs`) and takes
POSTs: a body naming a staged path runs a trial; `/accept` installs with no
trial and no rollback; `/restart/<svc>`, `/stop/<svc>`, `/start/<svc>`;
`/exit`. A trial stops the running service first and answers only when the
build is proven — it printed `INFO serving` — or rolled back, up to ten minutes
later.

**The relay is keep's own cluster socket.** keep holds one as the app `keep`
(`keep/src/presence.rs`). It cannot be the agent: the agent runs inside
motherd, so a trial, stop or restart of motherd would kill the relay carrying
it. keep declares its routes on that socket and serves the streams the gateway
opens with them (`keep/src/remote.rs`), to a request the gateway stamped with
the member it came from only. A member reaches a machine's keep over
`group:keep`, naming the machine in `x-tokera-host`; the gateway sends each
request to the keep of that machine (`web/src/cluster/keeps.rs`), and the tools
are Admin-only at the endpoint:

| Tool | keep route | Class |
|---|---|---|
| `keep_state {host}` | `GET /v1/keep/state` — each service: its stable file's build, whether it runs, the trial running and how the last ended | Look |
| `keep_trial {host, service, artifact}` | `POST /v1/keep/trial` — the trial, run as a job: accepted at once, its outcome read with `keep_state` | Care |
| `keep_service {host, service, action}` | `POST /v1/keep/service/:name/:action` — `restart`, `stop`, `start` | `restart`, `start`: Care; `stop`: Decide |

Every trial keep runs — through the cluster or its loopback switch — is on its
board (`keep/src/board.rs`) while it runs and after it ends, and a second is
refused while one runs.

`/accept` and `/exit` are not tools: an install with no trial and no rollback,
and keep restarting itself, are done on the machine.

**A trial of a build from `origin/main` is safe to delegate.** keep promotes it
only once it serves and restores the previous binary otherwise, so a bad build
cannot keep a member down. **A trial of anything else is not:** a build from a
branch or uncommitted work is a `local` build, which never becomes a candidate
but serves real cluster traffic while it runs, and everything it writes
replicates and is permanent (`docs/replicated_substrate.md`). So keep refuses
a trial of an artifact whose `--build-info` names a `local` build — anything
but exactly a commit on `origin/main` (`remote::mainline`); a branch is tested
by `zend-test` on an agent, and by hand on a machine kept out of the cluster
(§14.1). **Promotion is not delegated either:** certifying a build, rolling it
out, or rolling it back is a Decide command.

**The artifact keep runs** is a path on keep's own machine. A build job that
names a `stage` — `target/release/motherd.exe` — has the agent copy that file
to its `stage_dir`, the machine's `keep/stage/` (`agent/src/config.rs`), once
the run exits 0: written beside its name and renamed into place, from the
checkout's build outputs, which the sandbox never sets aside or removes. The
job's view answers the staged path (`Staged`), which `agent_log` reports.

### 8.5 The substrate

| Tool | Wraps | Class |
|---|---|---|
| `zend_layer {name}` | `GET /v1/substrate/zend/layer/:name` | Look |
| `zend_layer_toggle {name}` | `POST /v1/substrate/zend/layer/:name/toggle` | Care |
| `zend_timeline {id}` | `GET /v1/substrate/zend/timeline/:tl` | Look |
| `zend_timeline_delete {id}` | `DELETE /v1/substrate/zend/timeline/:tl` | Decide |

All four exist and are `Role::Admin` (`zend/src/api/mod.rs:105-128`). Wiping a
member's substrate is not a tool: it is the user's decision, carried out on
that machine (§11).

## 9. Operations skills

| Skill | Does | Approval |
|---|---|---|
| `cluster-status` | Summarises the cluster: members, builds, catch-up, anything degraded | — |
| `cluster-logs` | Finds what the logs say about a symptom, with the lines cited | — |
| `zend-test` | Builds a branch and runs its tests on an agent; pass or fail, the failing tests and their output | — |
| `zend-deploy` | Builds a commit on `origin/main` on the target machine's agent, stages it, and runs it as a keep trial there; reports serving or rolled back, with `started_at_ms` | — |
| `zend-release` | Takes a commit on `origin/main` through the whole flow: test, trial, certification watched to its end, rollout watched member by member | The rollout and any `certify-anyway` |
| `zend-rollback` | Rolls the cluster back to a named build, by a `rollout` of it | Always |
| `cluster-care` | Drains, compacts, catches up, resyncs or rejoins a member, and confirms it came back | Only for the Decide commands it would need |
| `zend-watch` | Watches a rollout or the logs for a stated condition, reporting only exceptions and the end state | — |

The long flows — a release, a watch — are delegated to a zend thread
(`release-<rev>`, `watch-<what>`), whose worker calls the operations tools
itself and ends in a journal entry; Claude reads the entry, and is asked for
approval at each Decide step. The short ones Claude calls directly.

**One source, two clients.** The skills are kept in one folder of this
repository, one folder per skill, and each `SKILL.md`'s frontmatter holds only
`name` (matching its folder: lowercase, digits and hyphens, at most 64
characters) and `description` (at most 1,024 characters) — anything else is a
hard upload error on claude.ai (§13.4). They are uploaded to claude.ai, where
Skills need "Code execution and file creation" on, and Claude Code receives
them by the sign-in sync into `~/.claude/skills/synced/` (§13.4). Battle Cities
and this repository use the same skills against the same endpoint, each thread
named for its project.

**Fallback.** Every skill says what Claude does itself when `cluster_status`
answers that zend cannot take work, or a task errors or stalls past its
timeout: the same job, done the ordinary way, at Claude's cost. A stalled zend
never stalls Claude. The operations tools reach the members, agents and keep
through the gateway, not through zend, so a release can be driven or rolled
back while zend itself is down — except where the agent that would build it is
the motherd being replaced.

## 10. Routing

Each project's `CLAUDE.md` names the split of §1, and the default is to
delegate:

- **Delegated:** anything reading-heavy or mechanical — implementing a spec,
  answering a codebase question, writing tests, building, testing, a trial
  deploy, watching, reviewing, researching.
- **Called directly:** short operations — status, logs, a job's result.
- **Kept:** deciding what to do; design and its documents; reviewing zend's
  results; and every step needing the user's approval.

## 11. Guardrails

zend's turns read untrusted content — anything anyone can push, any page a
search returns — while an Admin's hold Comprehensive tools, and the operations
tools can change what the cluster runs. So the limits are enforced by the
endpoint and by zend, not asked of anyone in a prompt:

- **Branches only.** A conversation owned through `/mcp` may push to a branch
  and never to a repository's default branch. Its tool context is
  branches-only (`ToolHost::bound`, `ToolContext::branches_only`), closing each
  repository's base branch and the conventional defaults `main` and `master`;
  `GitPush::run` checks every update and delete against it before anything is
  sent, and a repository whose default is not known takes no push from such a
  conversation at all. `git_push` is the only way out: `run_command` refuses
  git in any spelling (`zend-vfs/src/sandbox/git_use.rs`).
- **Decide needs the user.** A Decide command, a rollout, a rollback and a
  timeline delete are refused without the endpoint's `confirm` (§8.2). A zend
  worker is never given it: a delegated release stops at each Decide step and
  asks, and Claude passes it on only with the user's approval.
- **Trials only of `origin/main`** (§8.4).
- **Merging to `main` and committing under `CLAUDE.md`'s rule need the user's
  approval**, whoever did the work.
- **Wipes are the user's.** Deleting a substrate, a member's data or a
  repository is never a tool.
- **No secrets out.** The workers of a task that reads untrusted content are
  given no tool that sends arbitrary data to an arbitrary host — the same rule
  `security_review` keeps (`docs/zend_layers.md`, "Search, never fetch").
- **Shared journals are Claude's.** Only Claude writes a thread or project
  journal (§5.3), so text a worker read cannot reach every other worker's
  prompt.

## 12. Identity and access

The gateway owns sign-in (`docs/web_gateway_design.md` §5): Google OIDC with
PKCE (`web/src/auth/oidc.rs`), a session cookie (`web/src/auth/mod.rs:119-134`),
and identity stated to the members on `X-Tokera-User` / `-Provider` / `-Email`
/ `-Name` / `-Picture`, `-Provider` required (`web/src/auth/forwarded.rs:20-67`),
believed only from the trusted peers and stripped from anyone else
(`web/src/auth/gateways.rs:15-37`, `web/src/server.rs:594-604`). `Role` is the
gateway's (`web/src/auth/role.rs:63-70`); zend maps it to tool modes
(`zend/src/access.rs:30, 44`). `/mcp` keeps all of that.

**The gateway is the OAuth 2.1 authorization server for `/mcp`**
(`web/src/auth/mcp/`), beside the sign-in it already owns. Its routes are
mounted ahead of site routing on every host, as `/auth/*` are. On `/mcp`
alone, an access token it issued is an identity exactly as the session cookie
is (`Auth::mcp_identity`, read where the cookie is in `web/src/server.rs`), so
the member serving `/mcp` receives the caller on `X-Tokera-*` like any other
request and never sees what a token means. The site table forwards only
`/mcp` to the members (`web/web.yaml`, the `zend` site; motherd answers it,
`host/src/server.rs` `API_PREFIXES`).

- **Discovery.** A request reaching the member with no identity answers `401`
  with `WWW-Authenticate: Bearer resource_metadata="<origin>/.well-known/oauth-protected-resource/mcp"`
  (`error="invalid_token"` when it sent a token the gateway did not take),
  the origin being the client's, from the gateway's `X-Forwarded-Host` /
  `-Proto` (`zend/src/mcp/http.rs`). The gateway serves that document
  (RFC 9728), naming the resource as `<origin>/mcp` exactly and the origin as
  its one authorization server, and its own metadata (RFC 8414) at
  `/.well-known/oauth-authorization-server`. Exact bytes in
  `web/src/auth/mcp/metadata.rs`.
- **Clients without registration.** The metadata advertises Client ID Metadata
  Documents (`client_id_metadata_document_supported: true`, `"none"` in
  `token_endpoint_auth_methods_supported`), so no registration endpoint is
  offered. The gateway fetches the `client_id` URL (5 s, 64 KiB, no redirects),
  refuses a document naming another client, and allows a redirect the document
  lists — or, for a loopback redirect, one differing only in its port
  (RFC 8252 §7.3). The two documents in use, fetched 2026-10-25:
  `https://claude.ai/oauth/mcp-oauth-client-metadata` (Claude, redirect
  `https://claude.ai/api/mcp/auth_callback`) and
  `https://claude.ai/oauth/claude-code-client-metadata` (Claude Code,
  redirects `http://localhost/callback` and `http://127.0.0.1/callback`).
- **`/oauth/authorize`** checks the request (PKCE `S256` required, the resource
  this one), fetches the client's document and matches the redirect — none of
  which redirects anywhere, since an unchecked redirect is not one to send a
  browser to. **The user is then asked.** Who they are is the gateway's own
  session cookie; nobody signed in is sent to `/auth/login?next=…`, which
  brings them back. A client document is anyone's to write, so a code is never
  issued unasked: a link naming an attacker's client would otherwise hand a
  signed-in user's code to the attacker. The consent page names the client and
  where it will be sent, and posts back the consent it showed, signed and bound
  to the person who saw it; the answer is a code to the redirect with `state`
  and `iss` (RFC 9207), or `access_denied`.
- **`/oauth/token`** takes form-urlencoded bodies; a code works with the PKCE
  verifier, for the client and redirect it was issued to; a refresh token buys
  a fresh pair. Errors are RFC 6749's (`invalid_grant`,
  `unsupported_grant_type`, `invalid_target`). A code lasts five minutes, an
  access token an hour, a refresh token 30 days.
- **Stateless, like the session.** A consent, a code and each token are values
  signed with the session key (`web/src/auth/mcp/grants.rs`), each with a
  `kind` its reader requires, so a code is never a token and a session cookie
  — which has no `kind` — is never either. Peered gateways share that key
  (`web/src/cluster/peers.rs:83`), so each honours what any of them issued,
  and nothing is stored. The cost: a code works for its five minutes and a
  refresh token until it expires, not once — neither is any use without what
  the client alone holds, the PKCE verifier or the token itself — and signing
  every grant out at once is rotating the session key, as it is for sessions.
  The role is read from `roles.yaml` by the member on every call, not frozen
  into the token.
- **Bearer on `/mcp` only.** No other route reads a bearer token.

**Who may use it: any signed-in user.** Every `/mcp` conversation is owned by
its caller and served only to them or an Admin (§3.2). An Admin's tasks run
Comprehensive and reach the operations tools; anyone else's run Restricted,
reads `cluster_status` (the members' own statuses, not the gateway's view) and
`cluster_logs` (every member's, not one named member's) — the routes those
wrap are open to any signed-in caller — and every other operations tool
answers that it needs Admin (`ops::admin_only`).

**No new shared secret on the members.** Like the rest of the estate, the
daemons hold no key that could mint a session or a token: only the gateways
sign, and the members believe the identity they state from the gateway peers
alone.

## 13. The Claude clients

What the design relies on, from Anthropic's documentation and the MCP
specification, checked 2026-10-25.

### 13.1 Reaching the endpoint

- **claude.ai** adds a connector by URL on every plan (Free is limited to one;
  on Team and Enterprise only an Owner adds one), over Streamable HTTP. It
  follows the 2025-03-26, 2025-06-18 and 2025-11-25 authorization specs, uses
  CIMD when the authorization server advertises it, signs with PKCE `S256`,
  redirects to `https://claude.ai/api/mcp/auth_callback`, uses only the first
  of `authorization_servers`, and connects from `160.79.104.0/21`.
  (claude.com/docs/connectors/building/authentication)
- **Claude Code** adds it once for every project:
  `claude mcp add --transport http --scope user zend https://<host>/mcp`,
  signs in through `/mcp`, and uses its own CIMD with a loopback redirect on a
  random port. A connector added on claude.ai also appears in Claude Code
  sessions signed in with the claude.ai subscription; one added in Claude Code
  is not uploaded to claude.ai, and where both name the same URL Claude Code's
  own wins. (code.claude.com/docs/en/mcp)

### 13.2 Timeouts and result size

| | claude.ai | Claude Code |
|---|---|---|
| A tool call | 240 s, hard | 60 s to the first response byte unless the server's `timeout` (or `MCP_TOOL_TIMEOUT`) is set above 60000; 5 min idle, reset by progress notifications; moved to the background after 2 min |
| A result | about 150,000 characters; text and images only | warns at 10K tokens, 25K by default (`MAX_MCP_OUTPUT_TOKENS`); text over 50,000 characters saved to a file |

So the work tools open their stream at once, send progress while they wait, and
never hold a call past `wait_s`, whose ceiling is set per client (§3.3):
200 s on claude.ai, inside its 240 s cap; 3,600 s on Claude Code, whose server
entry sets `timeout: 3700000` so the call outlives the longest wait, and whose
automatic move to the background after 2 min (`CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`)
lets Claude work on while it waits. Every result is a text block — any
structured content is repeated as text — and the paging limits (§3.2) keep a
page far inside both clients' bounds.

MCP's tasks extension (`io.modelcontextprotocol/tasks`, 2026-07-28) would
carry long work natively, but only Claude Code's terminal CLI supports it in
beta — not the VS Code extension — and claude.ai does not document it;
`running` and a `zend_wait` over a set of workers work on every client, so they
are the design.

### 13.3 Server instructions

Claude Code delivers an MCP server's `instructions` to the model, at most
4,096 characters by default. claude.ai and Claude Desktop discard them
(anthropics/claude-ai-mcp#93, open). So nothing essential lives only there: the
rule that keeps the thread is in the tools' descriptions (§7.1).

### 13.4 Skills

A skill is a folder `<name>/SKILL.md` with YAML frontmatter, under 500 lines,
with optional `references/`, `assets/` and `scripts/`. claude.ai takes it as a
ZIP whose top level is the folder, turned on under Customize › Skills, on Pro,
Max, Team or Enterprise with code execution on; outside Claude Code only
`name`, `description`, `license`, `compatibility`, `metadata` and
`allowed-tools` are allowed in the frontmatter. Claude Code reads personal
skills from `~/.claude/skills/<name>/SKILL.md`, and since v2.1.273 syncs the
skills enabled on claude.ai down into `~/.claude/skills/synced/` — one way.
(claude.com/docs/skills, code.claude.com/docs/en/skills)

### 13.5 Hooks

Claude Code only, in `~/.claude/settings.json`. `SessionStart` fires with
`source` (and matcher) `startup`, `resume`, `clear`, `compact` or `fork`;
`compact` covers automatic and manual compaction, and the hook's stdout, or
`hookSpecificOutput.additionalContext`, is added to the context.
`PreCompact` cannot add context. (code.claude.com/docs/en/hooks)

## 14. Tests

- Every tool against a mock gateway and member: each tool's request and its
  exact response bytes, every state of §3.3, the `running` → `zend_wait` path,
  every row of §15.
- `zend_wait` over a set: answering on the first worker to leave `running`
  with the others' lines; a question from one worker returned while another
  still builds; the stream opened at once with progress notifications;
  `wait_s` clamped to 200 s for a claude.ai `clientInfo` and 3,600 s for
  Claude Code's.
- `passthrough` refused by the endpoint; a task turn submitted with zend's
  model and a `conv_id`.
- The authorization server: the `401` and its `resource_metadata`; both
  metadata documents' exact bytes; a CIMD client accepted, a redirect outside
  the allowed set refused; PKCE `S256` required; the code exchange, refresh
  rotation and `invalid_grant`; the consent page escaping what a client wrote,
  an answer from another person refused, a consent spent once; a secret never
  a record's key; a bearer token read on `/mcp` only.
- Ownership: an owned conversation refused to another user on every
  `/v1/conversations/*` and `/v1/threads/*` route, served to its owner and an
  Admin; an unowned dialogue unchanged.
- `journal_grep`'s routes against `JournalBook::grep`, including a subtree
  spanning members.
- Names: a clash with an ancestor or a live sibling refused with the clash
  named; cousins sharing a name accepted; an archived sibling's name taking
  `.2`, the suffixed name checked against the lineage; a malformed slug
  refused.
- Addresses: a bare name resolving upward only; an entry and a claim address;
  a folded entry's address resolving to the fold, its turns still readable; a
  cross-journal citation accepted for an ancestor and refused otherwise.
- `"self"` over MCP refused before `zend_attach` and the thread after it;
  `journal_note` refused to a worker; Claude's note on a `T:` and `P:` journal
  becoming one entry with its text filed behind it and no model reply.
- Inheritance: a worker minted from `base_conv` with its thread as journal
  parent, and the parent surviving a re-mint; a round pinning the ancestors'
  sealed sections outermost first and its own last; an entry noted on the
  thread joining a live worker's prompt the round after it seals, with nothing
  above it resealed and no descendant's own journal resealed; the reader's line
  pinned by nothing else; the 8K ancestor budget dropping the outermost lines
  first.
- The thread: each `zend_task`, `zend_answer` and `zend_nudge` entered with the
  worker's entry, not its outcome; two attached sessions' entries each naming
  their session.
- A task's end: the forced draft polled to `covered_to`; `draft_running`
  waited out; a skipped final stretch reported; the task's result being every
  entry overlapping its turns.
- The question block read only from its own line at a reply's end; a reply
  ending in one settling `question`, unarchived, with a `Task asks:` entry;
  `zend_wait` answering it; `zend_answer` refused for a worker not waiting,
  landing as the next turn otherwise, entered as `Claude answered:`; a restart
  finding a finished turn. `StreamItem::Closing` settling the worker `stuck`.
- `zend_nudge` queued, surviving a restart, and taken at the next round head,
  never mid-decode.
- `sub_run`: a part started under the worker's path, pinning its ancestors'
  journals; concurrent parts in one round; its entry as the tool response; a
  re-dispatched call after a restart attaching to its part.
- Priority: a worker decoding at `normal`, and at `high` when asked.
- The guardrails in zend: a push to a default branch from an `/mcp`
  conversation refused.
- The command classes: every maintenance and gateway command classed, a Decide
  command refused without the endpoint's `confirm`, a new command failing the
  test until it is classed; `rollout` through the gateway's plan.
- keep: its routes on its own socket — state, a trial as a job, start, stop,
  restart — against a keep on loopback; a trial refused for an artifact not
  built from `origin/main`.
- The agent: a run job submitted through the member route; `agent_log` paging
  by byte offset and grepping; a job lost to a motherd restart reported lost.
- One live end-to-end run per skill, graded as the replays were, before the
  routing rules make it the default.

### 14.1 Testing live, out of the cluster

The endpoint is first run against a zend that cannot touch the cluster's data,
and reached the way a real client reaches it:

- **motherd on the RTX 4090 Laptop, unclustered.** A copy of
  `motherd/motherd.laptop.yaml` with `clusters`, `cluster_name` and
  `group_key` dropped, its own `port`/`tls_port`, its own zend working folder
  (a `workspace.yaml` naming one scratch repository) and an agent file whose
  one checkout is that repository, with `push: false`; run by hand
  (`target\release\motherd.exe --config <that file>`), not by keep. Nothing it
  writes replicates. With `local_signin` naming the user, a loopback caller
  is that user with no gateway in front, for driving `/mcp` directly.
- **Through the public gateway.** The gateway, built with the authorization
  server (§12), forwards `/mcp` to that motherd's port, which trusts the
  gateways' identity headers (`gateways:` in its config), while every other
  route stays as it is. claude.ai adds the connector by its public URL and
  Claude Code by `claude mcp add`, so the OAuth flow, the timeouts and the
  result limits are the real ones.
- Only once a skill passes there does the endpoint route to the cluster.

## 15. Failure

| Failure | What happens |
|---|---|
| zend down, or not ready | `cluster_status` says so; work skills fall back to Claude; operations still run |
| A name clashes with an ancestor's or a live sibling's | Refused before anything runs, naming the clash; the caller picks another |
| A `journal` names nothing, or `"self"` before `zend_attach` | Refused, naming what was not found; nothing is written |
| A worker calls `journal_note` | Refused, saying a finding belongs in its final response |
| A worker is blocked on a decision | It ends its final response with a question block (§6.2) rather than guessing or stopping; Claude answers and it carries on |
| A turn hits the repeat guard or the round cap | The task answers `stuck` with the closing response; Claude answers with `zend_answer` |
| A turn errors | `zend_task` answers `error`; the skill retries once with a fresh worker on the same thread, then falls back. The failed worker is archived like any finished one, so its evidence stays readable |
| The forced draft is refused or skips | `zend_task` answers the task's earlier entries and says the final stretch added nothing; with none at all, the final response's text and that no entry was kept |
| Archiving a worker fails | The outcome is already on the thread; the worker is left live and archived by the next task on its thread |
| A task stalls | `zend_wait` keeps answering with it still `running` past the skill's timeout; Claude does the job, and the stalled turn finishes or fails on its own |
| A client cuts a call off before `wait_s` | Nothing is lost: the work runs on zend regardless, and the next `zend_wait` answers its state |
| A command's member is busy | `cluster_command` answers busy (409); the skill waits for the running job and retries |
| A command's or an agent job fails | `cluster_job` / `agent_log` answer its errors; the skill reports them and stops |
| A trial fails | keep restores the previous binary; `keep_state` reports the rollback |
| The motherd running a worker restarts mid-task | The worker resumes on the member that claims it (`resume_unfinished_turns`). Maintenance and agent jobs are held in that motherd's memory and are lost with it (`host/src/commands/jobs.rs:110-160`, `agent/src/jobs.rs:44-45`); only keep's trial runs outside motherd. The worker re-checks state — the build, `/v1/status` — rather than waiting on a lost job, and runs it again |
| An OAuth token expires | The client refreshes it; a refused refresh signs the user in again |

## 16. Changes to other documents

Made in the change that builds each piece:

- **`docs/zend_journal.md`**
  - §4: sections named by path, and the collection pinning ancestors'
    journals through the journal-parent chain (§5.1); third-person framing
    and the reader's line.
  - §4.1: a conversation owned through `/mcp` keeps its filed turns when
    archived, until its thread is tombstoned.
  - §4.2: in such a conversation a fold files the folded entries' turns under
    the fold.
  - §3: the `Claude said:`, `Claude asked:` and `Task ended:` labels, and
    entries kept from a note with no model reply.
  - §7: a journal is read by its own conversation and its descendants, and
    through the tools by its owner.
  - The draft route's `409 draft_running`.
- **`docs/web_gateway_design.md`**: the authorization server for `/mcp`
  (§12), and an access token read as an identity on `/mcp` alone.
- **`docs/replicated_substrate.md`**: keep's routes on its cluster socket and
  the trial as a job (§3.9); the member route that submits an agent run job
  (§11).
- **`docs/zend_layers.md`** and the tool YAMLs: `journal_read`'s and
  `journal_grep`'s `journal` parameter, `journal_note`, `sub_run`'s runner.
