# Zen's reach: closing the gap between Zen's tools and Claude's

**Status: built.** This document is authoritative over which
tools Zen's own conversations, task workers and chats reach, and how a call
that needs a person's approval is asked and granted. It amends
`docs/claude_zend_gateway.md` (§3.4, §8, §11), `docs/jobs.md` §8 and
`docs/zen_chat.md`; the amendments are listed in §9. Where it names machinery
that already exists, that machinery's document stays authoritative.

Decided by the user on 2026-11-04: a chat acts as a coder does (§6.1), and the
bots' and images' tools are forwarded to npcd's own MCP server, never
reimplemented in zend (§4.3, §6.4).

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
| D6 | Zen cannot write memory: no `journal_note`, `_create`, `_link` | Zen holds `journal_read`, `_grep`, `_check`, `_move` (`zend-tools/src/tools/journal/`); memory reaches its journals only by distillation |
| D7 | A dialogue or a chat cannot start long work: no `repo_task_start` or task tools | Nothing starts a task from a dialogue; §6.3 gives dialogues, chats and tasks the task tools, nesting one level |
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
- **A dialogue**: the person whose request started the turn, as the gateway
  stated them (`host::access`), the identity `access::effective_mode` already
  reads for the turn's tools mode. The chat route notes them on the runner by
  conversation id before it submits the turn (`OpsTools::note_dialogue_person`;
  the turn and its tool rounds run in this process), and the runner reads their
  role from the estate's table at each call. It states them on every request it
  sends up (`OpsCall::headers`), so each member and the gateway check the role
  themselves, as for a task. A dialogue acts as `Acting::Dialogue`
  (`zend/src/mcp/ops_tools.rs`).

A conversation with no person behind it (an unauthenticated turn, or a
scheduled one that names nobody) reads nothing that `Needs::ReadsAsPerson`
guards, and says why.

### 4.2 What a member holds of itself (S2, S3)

`cluster_status` renders, per member and per app (zend, npcd), the
replicator's own block from `/v1/status` (`cluster.apps.<app>`): its `phase`,
`link`, `generation`, unsent events and catch-ups in flight. A member whose
phase or link is not `joined` is cut off or rejoining, and is listed first, as
a problem. Claude and Zen read the same answer, because the
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
  (`docs/mcp_app.md`). zend reaches them by forwarding
  (`zend/src/mcp/npcd_tools.rs`): one `tools/call` to npcd's `/v1/npc/mcp`, up
  this member's cluster socket as the conversation's person, in the 2026-07-28
  protocol, the way the aggregator calls a backend. So npcd stays the only
  implementation, and its role checks stand. Each is a zend-tools tool whose
  arguments are written in zend's catalog as a subset of the npcd tool's own
  schema (the image tools leave out a crop box and an admin's authorization),
  and `motherd/tests/bot_schemas.rs` — which links both — holds them inside
  npcd's: every argument Zen offers is one npcd takes, of the same type, and
  the two require the same. Reading a bot needs a member (User), not an
  Admin.

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

- **A task or a chat** (Claude's): the worker's answer, or its question
  (`docs/claude_zend_gateway.md` §6.2), carries the approval's id and call.
  Claude asks the person. With their yes, Claude calls
  `approval_grant {id, confirm}` (`/mcp`, `confirm: "approved by the user"`,
  `zend/src/mcp/tools/approval_grant.rs`), which runs the recorded call as
  Claude's caller, unchanged, and answers what it answered; Claude passes that
  to the worker with `repo_task_reply`. The `confirm` gate of §8.2 stands on
  `approval_grant` for a held call, and on the Decide tools Claude calls
  itself.
- **A dialogue** (the person is already there): zend's UI draws the approval
  under the call that was held, with the call laid out, an **Approve** button
  and a **Refuse** button (`approvalBox`, `zend/web/index.html`). A button
  posts to `POST /v1/approvals/:id/grant` or `…/refuse` from the person's own
  signed-in session (`zend/src/api/approvals.rs`). The route settles only an
  approval kept under the caller's own key, so another person's id names
  nothing, and runs the call as them. What it answered is then sent to the
  dialogue as the person's next message, so the model hears the outcome.
  Typing "yes" grants nothing: an approval is a request the person's browser
  makes, which no text in a turn can produce.

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
| `cluster_command` `drain`, `resync`, `rejoin` | takes a member out of service | approval (§5) | — | approval | approval |
| `service_control` `restart`, `start` | Care | ✓ | — | ✓ | ✓ |
| `layer_toggle` | Care | ✓ | — | ✓ | ✓ |
| `cluster_command` Decide, `service_control stop`, `gateway_command`, `timeline_delete` | Decide | approval (§5) | approval | approval | approval |
| the task tools: `repo_task_start`, `_reply`, `_nudge`, a task's `job_*` | Start work | — never (§6.3) | — never | ✓ | ✓ |

- **A chat may change things (D5).** A chat is Claude asking Zen to act on
  Claude's behalf, with Claude's role. Holding it to reads made Claude start a
  task for a one-line restart. Researchers and reviewers stay read-only: what
  they read is untrusted, and they have no business changing what runs.
- **The deploy guard covers every step that stops a service.**
  `service_deploy` refuses a trial of motherd on the machine running the
  conversation, and where an app has no other ready member
  (`zend/src/mcp/tools/service/guard.rs`); the same guard covers a
  `service_control restart`. A refusal by the guard becomes an approval (§5),
  not a dead end. The Care commands that take a member out of service —
  `drain`, `resync`, `rejoin` — are held for approval whichever member they
  name, since the member could be the one the conversation runs on
  (`OUT_OF_SERVICE`, `zend/src/mcp/ops_tools.rs`).
- **`repo_job_start` takes `node`.** So Zen can pin a machine, as Claude does,
  with the rule kept in the tool: no job on a machine whose agent is
  `standby`, unless named (`docs/jobs.md`).

### 6.2 Memory (D6)

Zen gains `journal_note`, `journal_create` and `journal_link` on the memory
journals, `@group/…` and paths from the person's group
(`docs/memory_backbone.md` §6). They are the same implementations `/mcp`
serves (`note_in`, `create_in`, `link_in`), run through the operations runner
as the conversation's person, who must write in their group; the runner runs
them on the blocking pool. `journal_note` is the one tool of that name: left
without a `journal`, or with `self`, it keeps a note in its turn
(`docs/zend_journal.md` §5.2), and naming a memory journal it writes a line
there. `journal_focus` is not Zen's: it sets the speaker of an MCP
conversation's own memory journal, `here`, which a conversation of Zen's does
not keep. Two limits keep untrusted text out of everyone's prompts, the reason
thread and project journals stay Claude's (`claude_zend_gateway.md` §11):

- **A turn that read untrusted content cannot write memory alone.** A call to
  any of `tool_round::READS_UNTRUSTED` — the web, a network session's reply, a
  remote shell's output, a remote file, a database query — marks its
  conversation **read-out** for the rest of the turn, before the call runs
  (`OpsRunner::reading_untrusted`; a turn starts clean). A memory write from a
  read-out turn answers `needs_approval` (§5): the person, or Claude, sees
  exactly the line to be written and grants it or not. The workspace's own
  repositories are the person's, and reading them marks nothing.
- **Lines are stamped.** Each line Zen writes is written through the `zen`
  channel (`journals::scope::Channel::Zen`), so it reads "John via Zen", and
  `journal_forget` can find what Zen wrote.

Thread and project journals stay Claude's alone. `journal_edit`,
`journal_forget`, `journal_accept` and the `group_*` tools stay the person's
and Claude's (K6).

### 6.3 Starting work (D7): conversations start tasks, and tasks nest one level

**The rule: tasks nest one level.** The tools that start and steer research
and coding tasks are `repo_task_start`, `repo_task_reply`, `repo_task_nudge`
and the task forms of `job_status`, `job_read` and `job_cancel`. A dialogue run
by an Admin, and a chat, may start tasks. So may a task that Claude or a person
started — a coder handing a research question to a researcher of its own, say.
A task a task started (`Worker::by_task`) may not: its grammar holds the task
tools out, and `repo_task_start` from it is refused
(`zend/src/mcp/zen_tasks.rs`, `Level`).

**A task a conversation starts begins on its starter's work**: on the
starter's branches, with the starter's uncommitted changes as its last finished
tool round saved them, copied into the task's own file events when its
conversation is first built (`zend/src/conv_inherit.rs`,
`docs/zen_coding_challenge_fixes.md` Cause 5). A reviewer reads the change
without a commit first; what the task changes stays its own until it commits.

So work nests at most one level below where a person or Claude started it.
The person, or Claude, talks to a conversation or starts a task; that task may
hand pieces to tasks of its own; those do all of their work themselves and
answer with what they did.
Nothing a second-level task produces can start more tasks, so no chain of
tasks can feed its own output back as the next one's brief. The hazard is
real: a task whose answer starts another task could restate its own findings
to itself indefinitely, each round reading the last one's output as its brief,
with no person or Claude in the loop to see that it has stopped making
progress. One level keeps the task that started the others in the loop: it
reads their answers and decides.

**Where the tools run.** They are `/mcp`'s implementations (`workers::start`,
`answer`, `nudge`), run through the operations runner with the conversation's
person as caller (§4.1, `zend/src/mcp/zen_tasks.rs`), and each answers at once
with the task's handle; the conversation waits on it with `job_status`, which
gives its answer, its question and its journal entries. A dialogue's tasks go
on a thread named for it, `dialogue-<ten hex digits of its id's sha256>`. A
chat's tasks go on one named for it too, `chat-<its id>`: the chat's own thread
(`~<id>`) is Zen's side of it, disposed of with the chat, and no slug a task's
handle can carry. A task's tasks go on its own thread.

**The tools are hidden from a task a level down.** Each round of a task a
task started withholds the task tools (`zen_tasks::level_of` and
`withhold_by_level`, `zend/src/session.rs`). The worker record says whether a
conversation is a task or a chat, and whether a task started it
(`Worker::by_task`); a dialogue has none. A withheld tool is out of the
round's grammar and out of its prompt alike (`SelectionState::withhold_members`),
so the turn is never shown a signature it cannot call. Its prompt adds no "do
not start tasks" rule, which would only name what it cannot use.

**Hiding is the whole mechanism.** The stencil a round decodes under holds
only the tools it was offered, so a call to a withheld tool cannot be decoded
at all. Nothing a task writes can name a task tool, so no check at the call or
on the record is needed behind it. (The catalog summary at the head of the
tools section is sealed once per tools mode and shared by every conversation
of that mode, so it still lists the task tools' names; what a task can call is
the grammar's to say, and it holds none of them.)

**What a task does instead: the work.** A task is already the level where
work is done. When it finds more work that its brief needs, it does that work
itself, in its own rounds, with its own tools: it reads, edits, builds, tests
and researches inline. It never hands the work back as a proposal, and it
never answers with a list of tasks someone else should start. What it answers
is what it did.

**There is one way to split work: a task.** A task at the top that wants a
piece done apart — a verifier re-reading its findings, a researcher answering
a question its code depends on — starts a task of its own with
`repo_task_start`, with the role that piece needs, and waits on it with
`job_status`. It goes on the same thread, is listed beside its starter, and
keeps its own journal. No second mechanism splits a task inside its own turn.

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
any member's conversation: they make a picture and change nothing.
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
5. **Starting work** (§6.3).
   Then **bots' actions and images** (§6.4).

Every step is certified and rolled out like any build. The rollout stays the
person's decision.

## 8. Tests

- `ops_tools`: for each kind and each tool of §6.1's table, the call runs,
  is refused, or answers `needs_approval`, exactly as the table says. The test
  is table-driven from the same table, so a new tool cannot be added without
  a row.
- Tasks nest one level (§6.3): for a task a task started, the offered tool
  list, and the stencil built from it, hold none of the task tools, and the
  prompt names none of them (`zen_tasks::a_nested_task_holds_out_the_task_tools`,
  `a_level_is_read_from_the_record`).

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
