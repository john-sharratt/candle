# Zen's reach: closing the gap between Zen's tools and Claude's

**Status: designed, not built.** This document is authoritative over which
tools Zen's own conversations, task workers and chats reach, and how a call
that needs a person's approval is asked and granted. It amends
`docs/claude_zend_gateway.md` (§3.4, §8, §11), `docs/jobs.md` §8 and
`docs/zen_chat.md`; the amendments are listed in §9. Where it names machinery
that already exists, that machinery's document stays authoritative.

Every statement about existing code below was checked against the tree on
2026-11-04 and carries its source. Everything that does not exist yet is marked
**new**.

## 1. Why

On 2026-11-03 Claude diagnosed two problems Zen could not have diagnosed
alone:

- **"Is WhiteWorld cutting over?"** WhiteWorld's motherd was mid-trial and not
  answering, so `cluster_status` listed only BlackWorld. Telling a restart from
  a split took a local look at the processes and at keep's trial.
- **"Why are they detached?"** The answer was each member's own `/v1/status`:
  phase `joined`, link `joined`, generation 64. That disproved a log line that
  compaction was raising falsely (`06d65db8a`).

Zen had the logs and the code, but neither that state nor the tools to act on
what it found. The work splits by cost (`docs/claude_zend_gateway.md` §1): Zen
does the volume, Claude orchestrates and holds the approvals. Every gap below
moves work back to Claude's budget, or to the person.

The aim is that **Zen can see everything Claude can see, and do everything
Claude can do without the person's approval**. What needs approval is asked
through one mechanism (§5), never worked around and never forged.

## 2. Who "Zen" is here

Zen's turns run in three kinds of conversation, and they reach different
tools today:

| Conversation | Made by | Tools mode | Operations tools (`OpsRunner`) |
|---|---|---|---|
| **Dialogue**: a person talking to Zen in zend's own UI, or any conversation not made by `/mcp` | the zend GUI and API | the person's role: Comprehensive for an Admin (`zend/src/access.rs:43-49`) | **none.** `OpsTools::run` refuses any conversation whose id is not `mcp:<owner>:<thread>/<name>`: "is for a task Claude delegated through this MCP server, and this conversation is not one" (`zend/src/mcp/ops_tools.rs:213-218`) |
| **Task worker**: `repo_task_start` | Claude through `/mcp` | the caller's role (`docs/claude_zend_gateway.md` §3.4) | the eleven of `WORKER_TOOLS` (`ops_tools.rs:140-152`); the two that change something only for a `coder` |
| **Chat**: `zen_chat_send` | Claude through `/mcp` | the caller's role (`docs/zen_chat.md`) | as a worker of kind `Chat`, so read only: `Needs::Changes` refuses every kind but `Coder` (`ops_tools.rs:166-168`) |

The zend-tools catalogue itself is the same for all three. In Comprehensive
mode it holds the file, git, run, web, network, crypto, credential, notes,
journal-read and operations tools (`zend-tools/src/tools/`). The differences
are in what the operations runner will serve, and in which tools only
`/mcp` serves.

## 3. The gaps

**Seeing**

| # | Gap | Today |
|---|---|---|
| S1 | A dialogue cannot read the cluster at all | `ops_tools.rs:213` |
| S2 | No member's replication state is readable: phase, link, generation, catch-ups per substrate | `/v1/status` carries them, but `cluster_status` renders members and tenants and none of them (`zend/src/mcp/tools/cluster_status.rs`) |
| S3 | A machine whose motherd is restarting simply vanishes from `cluster_status`. keep knows it is mid-trial; the member list does not say so | `cluster_status` reads `/v1/status` and `/gateway/status.json`; `service_status` reads keep, separately |
| S4 | `layer_read` is not offered to Zen | `/mcp` only (`docs/claude_zend_gateway.md` §8.5) |
| S5 | No bot reads: missions, messages, channels, Discord, threads | npcd's `bot_*` tools, on `/mcp` only |

**Doing**

| # | Gap | Today |
|---|---|---|
| D1 | `repo_job_start` is not a Zen tool. The runner dispatches it (`ops_tools.rs:189`), but it is not in `WORKER_TOOLS` and zend-tools defines no tool for it | A worker builds only its overlay through `run_command`, on whichever agent the gateway picks |
| D2 | No maintenance command: neither the Look and Care classes nor Decide | `cluster_command` is `/mcp` only (`claude_zend_gateway.md` §8.2) |
| D3 | No `service_control` (restart, start, stop through keep) | `/mcp` only (§8.4) |
| D4 | No `gateway_command`, `layer_toggle` or `timeline_delete` | `/mcp` only (§8.2, §8.5) |
| D5 | A chat may not deploy or cancel | `Needs::Changes` (`ops_tools.rs:166`) |
| D6 | Zen cannot write memory: no `journal_note`, `_create`, `_link`, `_focus` | Zen holds `journal_read`, `_grep`, `_check`, `_move` (`zend-tools/src/tools/journal/`); memory reaches its journals only by distillation |
| D7 | A dialogue or a chat cannot start long work: no `repo_task_start` or task tools. Tasks stay unable to start tasks (§6.3) | A worker splits its own work with `sub_run` (§6.1), but nothing starts a task from a dialogue |
| D8 | No bot actions: mission give/step/cancel, message send, channel post | npcd's tools, on `/mcp` only |
| D9 | No images: `image_generate`, `image_edit`, `image_upload` | `/mcp` only |

**Deliberately not closed**

| # | Stays where it is | Why |
|---|---|---|
| K1 | A Decide command runs only with a person's approval | `claude_zend_gateway.md` §11. §5 is how Zen asks |
| K2 | Wipes, `offline`/`online`, `drain-gateway`/`undrain-gateway`, keep's `/accept` and `/exit` | Done on the machine: none can be undone through the gateway, or reached back once done (§8.2, §8.4) |
| K3 | Pushing to `main` | Branches only (§11) |
| K4 | `run_command` runs only `cargo` | The agents' allow-list (`agent/src/member.rs`, `CommandPolicy::allowing(["cargo"])`). A wider list is a separate decision, not part of this |
| K5 | The person's own accounts as Claude's connectors hold them: mail, calendar, drive, documents, artifacts | They are bound to the person's claude.ai account, not to the estate. Zen reaching them would be a separate grant of the person's, designed on its own |
| K6 | Editing, forgetting or accepting another's memory lines, and changing groups | The person's and Claude's (§6.2) |
| K7 | A shell on a machine | Zen sees through tools that state what they read (§4.2). An arbitrary shell is what K4 already refuses |

## 4. Seeing everything

### 4.1 Every conversation reads the cluster (S1)

`OpsTools::run` resolves the call's **caller** from the conversation itself
rather than from a task record:

- **A task or a chat**, `mcp:<owner>:<thread>/<name>`: as today, its record's
  person, role and kind (`threads::worker`).
- **A dialogue** (**new**): the person whose request started the turn, and
  their role, as the gateway stated them (`host::access`). This is the identity
  `access::effective_mode` already reads for the turn's tools mode. The turn
  carries it into its `ToolContext` (**new**: `ToolContext::with_turn_caller`),
  and the runner states it on every request it sends up (`OpsCall::headers`),
  so each member and the gateway check the role themselves, as for a task.
  A dialogue counts as kind `Dialogue` (**new**, `roles::WorkerKind`).

A conversation with no person behind it (an unauthenticated turn, or a
scheduled one that names nobody) reads nothing that `Needs::ReadsAsPerson`
guards, and says why.

### 4.2 What a member holds of itself (S2, S3)

`cluster_status` gains, **new**, per member and per substrate (zend, npcd),
the replicator's own block from `/v1/status`. That block already carries
`phase`, `link`, `generation` and `catchups`. It gains, **new**, the first
`oseq` written detached when there is one (`Origination::detached_range`) and
`own_through`, the replicator's own line of what it has written. A member whose phase is not `joined`, or whose link is cut off, is
listed first, as a problem. Claude and Zen read the same answer, because the
tool has one implementation (`docs/jobs.md` §8).

`cluster_status` also joins keep's state (`GET /v1/keep/state`, which
`service_status` reads) with the member list (**new**). A machine whose keep
runs motherd while no member of it answers is listed as such, never just
omitted:

```
- WhiteWorld: not answering — keep is trying motherd 4419-6ecd6158, 51 s into its 10-minute window
```

That one line would have answered "is WhiteWorld cutting over?" without a look
at the machine.

`member_status {node}` (**new**, Look) answers one member's own `/v1/status`,
whole, through `x-tokera-node`, for what the summary leaves out. It is the
read Claude made over the LAN on 2026-11-03.

### 4.3 Layers and bots (S4, S5)

- `layer_read` joins `WORKER_TOOLS` as `Needs::ReadsAsPerson`. It is one
  `GET /v1/substrate/zend/layer/:name` (`zend/src/api/mod.rs`).
- The bots' read tools: `bot_list`, `bot_message_read`, `bot_channel_read`,
  `bot_discord_read`, `bot_mission_read`, `bot_thread_read`. These are npcd's,
  served by its MCP server at `/v1/mcp` behind the aggregator
  (`docs/mcp_app.md`). zend reaches them by **new** forwarding: `OpsCall::mcp`
  sends one `tools/call` to npcd's `/v1/mcp` through the gateway, as the
  caller, the way the aggregator calls a backend. So npcd stays the only
  implementation, and its role checks stand. Each becomes a zend-tools tool
  whose request is the npcd tool's own schema, fetched from npcd's
  `tools/list` when the daemon binds its runner. A schema npcd changes reaches
  Zen at the next bind, with no zend change.

## 5. Asking a person: the approval

Some calls stay a person's to make: a Decide command, `service_control stop`,
`gateway_command`, `timeline_delete`. Zen asks for them through one mechanism,
whichever conversation it is in. Zen's own words are never what grants an
approval.

**An approval is a record** (**new**, `mcp/approval/<owner>/<id>` in the
substrate, so it replicates and any member serves it). It holds:

- the exact call: the tool and its arguments;
- the conversation that asked, and why it asked, in Zen's words;
- what Zen expects the call to do, and how it will know it worked;
- its state: `asked`, `granted`, `refused` or `expired` (24 hours).

A call that needs approval does not run. It answers `needs_approval {id}`, and
the turn carries on or ends with a question, as the work allows. **Who grants
it depends on who asked:**

- **A task or a chat** (Claude's): the `question` state of
  `docs/claude_zend_gateway.md` §6.2 carries the approval's id and call.
  Claude asks the person. With their yes, Claude calls `approval_grant {id}`
  (**new**, `/mcp`, `confirm: "approved by the user"`), which runs the recorded
  call as the person, unchanged, and answers its result to the worker's next
  turn. The `confirm` gate of §8.2 moves onto `approval_grant`. The call
  itself is unchanged from today's `cluster_command`.
- **A dialogue** (the person is already there): zend's UI shows the approval
  under Zen's message, with the call laid out, an **Approve** button and a
  **Refuse** button (**new**). The button posts to
  `POST /v1/approvals/:id/grant` from the person's own signed-in session. The
  route checks that the caller is the conversation's person and an Admin, then
  runs the call. Typing "yes" grants nothing: an approval is a request the
  person's browser makes, which no text in a turn can produce.

**What an approval cannot do:**

- It runs the call exactly as recorded. Changing an argument is a new
  approval.
- It runs once.
- It runs as the person who granted it, so an Admin's grant is needed for an
  Admin's command. The members still check `confirm` themselves (the host
  name, the `count-hash`; `docs/claude_zend_gateway.md` §8.2): the record
  carries those as the call's arguments, so the person sees them in the
  approval.
- K2 stays off the list of what can be approved: those are not tools at all.

## 6. Doing what needs no approval

### 6.1 Operations (D1–D5)

`WORKER_TOOLS` becomes the one table for every conversation (§4.1), and gains
a column for each kind. A dialogue run by an Admin counts as a coder:

| Tool | Class | Coder | Researcher, Reviewer | Chat | Dialogue (Admin) |
|---|---|---|---|---|---|
| the job readers, `cluster_*`, `member_status`, `timeline_read`, `layer_read`, `service_status` | Look | ✓ | ✓ | ✓ | ✓ |
| `repo_job_start` (**new** tool), without `stage` | Look: it changes no service | ✓ | ✓ | ✓ | ✓ |
| `repo_job_start` with `stage` | Care | ✓ | — | — | ✓ |
| `service_deploy`, `job_cancel` | Care | ✓ | — | ✓ (D5) | ✓ |
| `cluster_command`, Look class | Look | ✓ | ✓ | ✓ | ✓ |
| `cluster_command`, Care class | Care | ✓ | — | ✓ | ✓ |
| `service_control` `restart`, `start` | Care | ✓ | — | ✓ | ✓ |
| `layer_toggle` | Care | ✓ | — | ✓ | ✓ |
| `cluster_command` Decide, `service_control stop`, `gateway_command`, `timeline_delete` | Decide | approval (§5) | approval | approval | approval |
| the task tools: `repo_task_start`, `_reply`, `_nudge`, a task's `job_*` | Start work | — never (§6.3) | — never | ✓ | ✓ |

- **A chat may change things (D5).** A chat is Claude asking Zen to act on
  Claude's behalf, with Claude's role. Holding it to reads made Claude start a
  task for a one-line restart. Researchers and reviewers stay read-only: what
  they read is untrusted, and they have no business changing what runs.
- **The deploy guard covers every Care step that stops a service.**
  `service_deploy` refuses a trial of motherd on the machine running the
  worker, and where an app has no other ready member
  (`zend/src/mcp/tools/service/guard.rs`). The same guard covers
  `service_control restart`, a `cluster_command` `drain`, and a `resync` or
  `rejoin` of the worker's own member (**new**). A refusal by the guard becomes
  an approval (§5), not a dead end.
- **`repo_job_start` takes `node`.** So Zen can pin a machine, as Claude does,
  with the rule kept in the tool: no job on a machine whose agent is
  `standby`, unless named (`docs/jobs.md`).

### 6.2 Memory (D6)

Zen gains `journal_note`, `journal_create`, `journal_link` and `journal_focus`
on the memory journals, `~/…` and `@group/…` (`docs/memory_backbone.md` §6).
These are the same implementations `/mcp` serves, run in process. Two limits
keep untrusted text out of everyone's prompts, the reason thread and project
journals stay Claude's (`claude_zend_gateway.md` §11):

- **A turn that read untrusted content cannot write memory.** A turn that has
  called `web_fetch`, `web_search`, `http_request`, any `*_session_recv`, or
  `file_read` on a path outside the conversation's own repositories is marked
  **read-out** for its remaining rounds (**new**, on the `ToolContext`). A
  memory write from a read-out turn answers `needs_approval` (§5): the person,
  or Claude, sees exactly the line to be written and grants it or not.
- **Notes are stamped.** Each line Zen writes names its conversation and turn,
  as the memory already does for an MCP caller (`journal_focus`'s speaker), so
  `journal_forget` can find what Zen wrote.

Thread and project journals stay Claude's alone. `journal_edit`,
`journal_forget`, `journal_accept` and the `group_*` tools stay the person's
and Claude's (K6).

### 6.3 Starting work (D7): conversations start tasks, tasks never do

**The rule: the tools that start and steer research and coding tasks belong to
conversations, never to tasks.** Those tools are `repo_task_start`,
`repo_task_reply`, `repo_task_nudge` and the task forms of `job_status`,
`job_read` and `job_cancel`. A dialogue run by an Admin, and a chat, may start
tasks. A task, and any part of one, may not.

So work starts at exactly one level. The person, or Claude, talks to a
conversation. The conversation starts tasks. Each task does all of its work
itself, splitting it with `sub_run` when it must, and answers with what it
did. Nothing a task produces can start more tasks, so a task's output can
never come back round as the instruction for the next task. The hazard is real: a task whose answer starts another task
could restate its own findings to itself indefinitely, each round reading the
last one's output as its brief, with no person or Claude in the loop to see
that it has stopped making progress.

**Where the tools run.** They are `/mcp`'s implementations, run in process
with the conversation's person as caller (§4.1). A dialogue's tasks go on a
thread named for it (`D:<conversation>`, **new**,
`docs/claude_zend_gateway.md` §5.2). A chat's tasks go on the thread Claude
attached the chat's session to. A task's `question` comes back into the
conversation as the tool's result, so the person, or Claude, sees it there.

**Enforced three times, so no single slip opens a loop:**

1. **Never offered.** The tool list a turn is offered is built from the
   conversation's kind (§6.1's table). The task tools are absent from the
   tool list of a task worker (`Coder`, `Researcher`, `Reviewer`) and of every
   part. A model cannot call a tool it was never shown, whatever text it has
   read.
2. **Refused at the call.** `repo_task_start` resolves the calling
   conversation's record before anything starts (**new** check), and refuses
   when the caller is a task or a part. That holds whatever the call's
   arguments say, and covers a call that reaches the runner by some other path
   than the tool list: `task_from_task: a task does not start tasks; do this
   work here, with your own tools, or split it off with sub_run`.
3. **Recorded on the task.** Every task records the conversation that started
   it (`started_by`, **new**, on the worker's record), and that conversation
   must not itself be a task. A record with a task as its `started_by` is
   refused when it is written, so the substrate can never hold a chain of
   tasks, even one written by a bug in the first two checks.

**What a task does instead: the work.** A task is already the level where
work is done. When it finds more work that its brief needs, it does that work
itself, in its own rounds, with its own tools: it reads, edits, builds, tests
and researches inline, and splits off a part with `sub_run` when a piece of
it runs better on its own. It never hands the work back as a proposal, and it
never answers with a list of tasks someone else should start. What it answers
is what it did.

**`sub_run` is not an exception.** Parts are how one task splits its own
work, every part waits under its task, and depth is capped: a task's part may
start parts, a part's part may not (`parts::MAX_DEPTH`,
`docs/claude_zend_gateway.md` §6.1). A part ends when its task does, and it
reports only to its task, never to a thread. The cap and the waiting together
keep parts bounded in a way tasks would not be.

**A conversation's own loop is bounded by its turn.** A dialogue or chat that
starts a task, reads the answer and starts another is one turn's tool rounds.
Its round cap and repeat guard close it as they close any turn
(`docs/claude_zend_gateway.md` §6.2). The next round of work then waits for
the person's or Claude's next message.

### 6.4 Bots and images (D8, D9)

The bots' changing tools, `bot_mission_give`, `_step` and `_cancel`,
`bot_message_send` and `bot_channel_post`, are forwarded as in §4.3, Care
class: a coder's or an Admin dialogue's. `bot_message_send` and
`bot_channel_post` send text into rooms other people read. From a read-out
turn (§6.2) each needs an approval, for the same reason a memory write does.

`image_generate` and `image_edit` are forwarded the same way, Look-class for
any Admin conversation: they make a picture and change nothing.
`image_upload` takes a file from the caller's machine, which a Zen turn does
not have, so it is not offered.

## 7. Release

In this order. Each step stands on its own, is tested, and ships as its own
trial:

1. **Seeing** (§4): the caller of a dialogue, `cluster_status`'s replication
   block and keep join, `member_status`, `layer_read`, the bots' read tools.
   It changes nothing that runs, and closes S1–S5.
2. **The approval record and its two surfaces** (§5), with `timeline_delete`
   and `gateway_command` as its first callers. Nothing else depends on it yet,
   so it is proven alone.
3. **Operations** (§6.1): the per-kind table, `repo_job_start` as a tool, Look
   and Care `cluster_command`, `service_control`, `layer_toggle`, the extended
   guard, and Decide through approvals. This step moves the `confirm` gate of
   `docs/claude_zend_gateway.md` §8.2 onto `approval_grant`.
4. **Memory** (§6.2): the read-out mark and the four memory writers.
5. **Starting work** (§6.3), with all three of its checks in the same step:
   no conversation gains the task tools before a task is unable to start one.
   Then **bots' actions and images** (§6.4).

Every step is certified and rolled out like any build. The rollout stays the
person's decision.

## 8. Tests

- `ops_tools`: for each kind and each tool of §6.1's table, the call runs,
  is refused, or answers `needs_approval`, exactly as the table says. The test
  is table-driven from the same table, so a new tool cannot be added without
  a row.
- No task starts a task (§6.3), each check alone:
  - a task's and a part's offered tool list holds none of the task tools;
  - `repo_task_start` called on the runner from a task or a part is refused
    `task_from_task`, with the tool list bypassed;
  - a worker record whose `started_by` is a task is refused when written.

  A dialogue and a chat each start a task, and that task's answer comes back
  as the conversation's tool result.
- A dialogue's caller: a turn started by an Admin reads the cluster as that
  Admin; one started by a User reads nothing an Admin's identity guards; an
  unauthenticated turn reads nothing that names a person.
- Approvals: a recorded call runs once, exactly as recorded, as its granter.
  A grant from anyone but the conversation's person (dialogue) or without
  `confirm` (`/mcp`) is refused. An expired approval does not run.
- Read-out: each tool that reads untrusted content marks the turn, and a
  memory write or bot post after it answers `needs_approval`. A turn that read
  only its own repositories writes directly.
- `cluster_status`: from recorded `/v1/status` and keep-state bodies, a
  joined member, a detached one, and a machine mid-trial each render their
  line. Asserted as exact text.
- The latency gate (`zend/src/mcp/latency_tests.rs`) gains `member_status`
  and the extended `cluster_status`. Neither runs a model, so each answers
  inside 30 ms at the 95th percentile (`CLAUDE.md`, "MCP Request-Path
  Invariants").

## 9. Changes to other documents

- `docs/claude_zend_gateway.md`:
  - §3.4: tools by kind point here.
  - §8.2: the `confirm` gate moves onto `approval_grant`.
  - §8.4: the extended guard.
  - §11: "Decide needs the user" names §5 as its mechanism, and "Shared
    journals are Claude's" names thread and project journals only, with memory
    under §6.2.
- `docs/jobs.md` §8: `repo_job_start` is a worker's tool too.
- `docs/zen_chat.md`: a chat may change what a coder may (§6.1).
