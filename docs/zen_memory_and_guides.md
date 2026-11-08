# Zen's memory, guides and task driving: the build

**Design and build plan, in progress.** The work that follows from three audits
of 2026-11-08 — the MCP guides against the tool set, how a task is driven from
each client, and the journals (how they are structured, written, searched and
read, and how they reach conversations through provenance and projection) —
carried out as **one build by AlienWorld and BlackWorld together**: one branch,
the phases below, each closed by its gate, one review, one merge, one joint
release.

The design documents each phase names (`memory_backbone.md`, `guidance_layer.md`,
`mcp_user_context.md`, `mission_templates.md`, `mcp_file_tools.md`) are revised
in the phase that changes what they govern, and are authoritative over this plan
where they are more specific. This plan is authoritative over the order, the
gates, the ownership and the decisions in §2.

---

## 1. What the audits found

### 1.1 Guides

- **Every method assumes someone stays.** Each guide's coordinator waits, answers
  questions, verifies and brings approvals to the user. A Grok task cannot: the
  car drops the conversation seconds after the driver stops talking, so a task's
  question reaches no one, its result is checked by no one, and nobody reads the
  thread it ends on. The DJ design task (2026-11-07) wrote its document and
  stopped at "shall I commit it?".
- **No guide sets `role`.** Templates say "read only" in CONSTRAINTS and run as
  `coder`; the DJ task ran as `researcher` and was asked to write a document.
- **`repo_task_start.yaml:20` says a task "starts no tasks of its own"**, while
  `implement`, `review` and `research` have the worker start a `verify-*` task,
  and tasks nest one level (`zend/src/mcp/zen_tasks.rs:80, 157`).
- **Tools with no guide:** `approval_grant`; `group_create`, `group_describe`,
  `group_add`, `group_remove`, `group_default`; `member_status`; the `file_*`
  tools outside Yen's repository. `steer` says a reply cannot approve and to ask
  the user, but never names `approval_grant`, the one way a held call runs.
- **`memory` lists `journal_accept` and never explains proposals.** `grok.md`
  repeats the memory rules instead of pointing at `memory`, and the copies have
  drifted. `journal` and `knowledge` cover the same ground as `memory`.
- **`docs-write` covers documents from code only**; no guide lands a design
  proposal, the document the DJ task wrote.
- **A client never learns that the tools changed.** Two days after a release
  replaced `cluster_logs` with the `file_*` and `log_*` tools, claude.ai
  sessions still held `cluster_logs` and none of the new ones. Nothing
  between zend and claude.ai dropped them: every server declared
  `listChanged: false` (`mcp-server/src/protocol.rs`), and claude.ai keeps a
  list across chats. A reconnect of the connector brought them in at once.

### 1.2 Journals

"Journals" are five stores: the memory tree (git, the `journals` repository);
thread and project notes `T:`/`P:` (substrate records, `zend/src/mcp/threads.rs`);
a worker's journal (`journal-core` entries of its conversation); a dialogue's own
journal (`journal.kept`); and `notes_*` (SQLite). No read or search crosses them.

The live tree (read 2026-11-08):

- **Episodes swamp it.** Every MCP call that is not read-only is written as an
  episode line (`mcp_user_context.md` §2.3), one entry file each: one
  conversation's `here` held eight raw `log_grep` regexes and no decision;
  `conversations` holds 193 journals of 1–4 lines whose folds read "John used
  Claude Code to read bot missions". No outcome is recorded.
- **One Claude Code session becomes many conversations.** A session's journal is
  remembered only in the process that opened it
  (`zend/src/mcp/memory/conversations.rs:8-13`), so a call routed to the other
  member, or after a restart, opens another: 00:08, 00:19, 00:29, 00:37 … for one
  session.
- **Lines are misfiled**: `people/john` (cue: "architectural beliefs, coding
  standards, publication style") holds LAN addresses, peer-session roles,
  connector naming and car reminders.
- **Facts are stale and unmarked**: `topics/candle/zen/ingest` gives the key
  version as 2 (it is 4); `workers-threads` describes `repo_task_wait`,
  `repo_task_stop`, `sub_run` parts and "Claude started:" labels, all gone.
- **Docs are copied**, not referred to (VarBuilder, the CUDA-kernel steps, the
  platform notes), and drift from their source.
- **Expired reminders and a stored instruction** ("Always call
  get_user_context …") stand in the tree.
- **Summaries are cut mid-sentence** and 4–8 days old.
- **`threads` and `bots` are empty**: §7.3's export of worker results was never
  built, so no task outcome reaches memory.
- **Speakers are wrong**: bulk imports and Claude's own conclusions are stamped
  "John via Claude Code".

Reading and search:

- Memory `journal_read` returns up to 50 × 2,000 characters but answers are cut
  at 32,000 (`zend/src/mcp/tools/fit.rs`); the cut says to use `offset`/`limit`,
  which the memory branch of `journal_read.rs` ignores.
- Archived lines are reached only by grep or an exact address. Grep stops at 50
  hits with no paging, filters by nothing, follows no reference, and never
  reaches `T:`/`P:`. `T:<thread>/*` concatenates worker by worker, then
  truncates.
- Thread notes are numbered by position in key order, keyed by `at_ms` and a
  per-process `NOTE_SEQ` (`threads.rs:439-440, 479`): a note replicated with an
  earlier stamp renumbers every later `#N`.

How journals reach conversations:

1. **Lineage sections** — `T:`/`P:` notes enter every live worker's prompt the
   round after they seal, no prefill, capped at `lineage::MAX_CHARS`
   (`zend/src/mcp/lineage.rs`). Real time, one lineage.
2. **MCP frames** — `get_user_context` reads the in-memory tree: at once on the
   member that wrote, within the 5 s sync and a fetch elsewhere.
3. **Provenance** — the `guidance` group (`projection.yaml`, `guidance_layer.md`)
   draws the `journals/` entry files as `code_reading` units (hub files —
   `README.md`, `group.yaml`, `journal.yaml`, `person.yaml` — excepted, and left
   to `scopes`), with the recency term built (`projection.yaml:1048`). It is the
   only path by which one conversation's note reaches another zend
   conversation, at the cost of a push, a branch ingest, a reading conversation
   per file, calibration and the base taking the new version. Each episode is an
   entry file, so ingest work scales with tool calls; under the lease order that
   puts `journals/` first, episodes are read ahead of code.
   - **No route filters by group.** `scope_of` takes no person and `ScopeKey` is
     `(u64, Vec<(String, Oid)>)` (`retrieval_scope.rs:61, 273`): a conversation
     draws every group's units, and hub files, folder units and derived layers
     reach it through `scopes` as well.
   - **Use-exemplars cross people.** A read seals the reading conversation's
     request and reasoning as an exemplar of the unit (`zend/src/guidance/`):
     never placed as text, but one person's reasoning shapes scoring in
     another's conversation.

---

## 2. Decisions

Settled; a phase that meets one does not reopen it.

| # | Decision |
|---|---|
| D1 | **One branch, `zen/memory-and-guides`, from `origin/main`.** Both sessions commit and push to it; `origin/main` is merged in at each phase gate (never a rebase: the branch is pushed). It merges to `main` once, after Phase 8's review, and is released once. |
| D2 | **A conversation sees only its person's groups' journal content, on every route into a prompt** — the `guidance` draw, hub files, folder units, derived layers, `persons/`. Its person is the `/mcp` owner's, or, for a dialogue from the Zen app or VS Code, the person signed in when it was opened, recorded on the conversation. A conversation with no person draws no journal content. **Only the ingest readers and calibration conversations, named in the code, see every group.** |
| D3 | **Episodes leave the memory tree.** A conversation's tool activity is kept in its journal's activity record, so "where the last conversation left off" still works, and is never an entry file. Conversation journals leave every provenance route. |
| D4 | **Maintenance is a mission.** This build includes the drone ↔ zend-task bridge (`mission_templates.md` §7) and the `journal-maintenance` and `guide-maintenance` templates on it. |
| D5 | **A task is driven `attended` or `alone`**, defaulting from the client: `alone` for Grok, `attended` otherwise. A drone driving a task is an attended coordinator. |
| D6 | **No backward compatibility.** Existing episode entry files, stale lines and old guide text are replaced, not kept beside the new. |
| D7 | **Simplicity cuts:** no verify task chained by zend after an alone task (the templates carry one); no automatic restart of a lost alone task (it is reported); no resolver over `T:`/`P:` (D9); the live path reuses lineage sections; maintenance checks live in `journals check`; outcomes extend the settle notes zend already writes. |
| D8 | **A drone's zend task runs as Zen's own person** (`people/zen`), calling as a service-key identity, subject `service:zen` (`mcp_file_tools.md` §4, `web/src/cluster/service_key.rs`), from its own member over the cluster socket: its tasks, threads and outcomes are its own, it grants nothing, and its held calls go to the authorizers (the Creator, Yen). |
| D9 | **Thread results are exported into the memory tree** (`memory_backbone.md` §7.3): a thread's notes and its tasks' outcomes are written to `threads/<thread>` in its owner's group and referenced from the topics it is about. `T:`/`P:` stay the work records. |
| D10 | **The memory guide splits in two.** A short client guide rides the first call of every Claude conversation (`mcp_user_context.md` §2.4–2.5's budget): the frame, the write rules in brief, pointers, the `yen-apps` line. A full `memory` guide, absorbing `journal` and `knowledge`, is read on demand. `grok.md` stays the Grok client guide. |
| D11 | **BlackWorld's overlapping pieces join this build**: dropping `conversations/` from provenance (Phase 2), the transfer and preview sections of the `files` guide and the `yen-apps` text (Phase 5). |

---

## 3. How the build runs

### 3.1 Who does what

| Owner | Phases |
|---|---|
| **AlienWorld** (coordinator) | 0, 1, 2, 3, 4, the npcd half of 6, 7, 8's review and merge request |
| **BlackWorld** | 5 (the guides, D10, D11's files and yen-apps text, and first the changed-list notice); the `service:zen` identity and the npcd → zend route of 6; 8's trial and joint release |

Each session writes its own phases' specs, delegates the implementation to Zen
workers (`repo_task_start`, `role: coder`, `effort: deep`; reviews
`exhaustive`) on the thread `memory-and-guides`, verifies each result against
the turns it cites and reads every diff. Builds and tests run on a build agent
(`repo_job_start`), one cargo process at a time, the narrowest scope that covers
the change. When zend cannot take the work, the session does it itself.

The two sessions touch different files except at three seams, which the earlier
phase defines and the later one consumes: the tool shapes of Phases 2–4 (Phase 5
documents them), the served tool list (Phase 5 finds it; Phase 5's test reads
it), and the `service:zen` identity (BlackWorld's, consumed by Phase 6's npcd
half). Each seam is announced to the other session when it is pushed.

### 3.2 Gates and rules

- **Each phase closes on its gate:** its tests pass on an agent; `cargo fmt
  --all -- --check` and `cargo clippy --workspace --tests --examples -- -D
  warnings` are clean; its owner has read the diff; its design-doc revision is in
  the same commits. A phase that fails its gate is fixed before it is called
  done — never carried forward.
- **Standing rules bind every phase:** CLAUDE.md (file tools only, no shims, no
  flags, no TODOs, imports not paths, one concern per file, tests asserting raw
  expected values) and the MCP request-path invariants (30 ms at p95 for a tool
  that runs no model; a read never writes; no full scans; no network under a lock
  a request takes). Nothing is deployed from the branch before Phase 8.
- **The user's approvals:** the merge, the release's Decide commands, and any
  `critical` step the cleanup pass reaches.

---

## 4. Phases

### Phase 0 — The backlog count

One read-only task: how many `code_reading` units in the current ingest backlog
are `journals/` entry files, and how many of those sit under `conversations/`.
Source: `zend::code_read::ingest` / `ingest_backlog` (`item_done` by path) and
the branch pass's "planned the layer against every branch" line. **Gate:** the
two counts, verified against the cited lines; they size Phase 2's relief and are
quoted in Phase 8's report.

### Phase 1 — Journal content obeys the group boundary (D2)

- **A conversation's person.** zend records `person: <slug>` on the conversation
  at creation: an `/mcp` conversation from its caller (`mcp::caller::Person`,
  stored on the `Worker` record for a task), a dialogue from the identity the
  forwarded headers name at `POST /v1/conversations`
  (`web::auth::forwarded::identify`). It is replicated with the conversation.
- **The reader.** `scope_of` takes `Reader::Person(slug)` or `Reader::Corpus`.
  `Corpus` is passed by exactly two callers, named in the code: the ingest
  readers (`code_read`, `repo_scan`) and calibration (`calibrate_layers`,
  `calibrate_guidance_reads`). Every dialogue passes its person; a conversation
  with no person passes `Person` of none, which reads no group.
- **The filter**, applied in `scope_of` to every route — `in_files`,
  `in_folders`, the derived layers and the drawn groups: a `journals/` key is in
  scope only when its path is `journals/groups/<g>/…` for a group the person
  reads, or `journals/persons/<slug>/…` for that person. The groups a person
  reads come from the memory tree's group membership (`journals` crate, the same
  check the MCP memory tools make), looked up from the in-memory tree.
- **The cache.** `ScopeKey` gains the reader, so two people never share a scope;
  both caches keep their 16 entries.
- **Exemplars.** A use-exemplar is sealed under its unit as now; Phase 1 asserts
  it is never placed as text, and §5 names the scoring influence it carries.
- **Hidden lines.** A unit whose file is a line hidden by `expires` or `replaces`
  (Phase 2) leaves the scope.
- **Tests:** an owned conversation never draws a unit, hub file or folder unit of
  a group its person is not in; it draws a guide and a journal unit of its own
  group under one owner; `persons/` is its person's only; `Corpus` draws all; a
  person-less conversation draws no journal content; no exemplar text is placed;
  a hidden line's unit is out of scope; the draw's `except` and share pool are
  unchanged.
- **Docs:** `memory_backbone.md` §3.1, §7.2; `guidance_layer.md` §1.

### Phase 2 — Writing memory well (D3, D6, D11)

1. **Provenance drops conversations** (D11): `conversations/` joins the
   `guidance` draw's `except` and is excluded from `scopes` for `journals/`,
   with its test. First, because it relieves the backlog at once.
2. **The activity record.** A conversation journal's `journal.yaml` gains
   ```yaml
   activity:            # newest last, at most 200 entries
     - at: 2026-11-08T03:27:00Z
       tool: log_grep
       subject: "^2026-11-08T0(1:5…"   # the episode's subject, at most 120 chars
   activity_folded: { log_grep: 41, repo_task_start: 3 }   # counts past the cap
   ```
   `user_episode_note` stops writing an entry file. It appends to the
   conversation's activity in memory; `mcp::tending` writes it into
   `journal.yaml` at most once a minute per conversation, off the request path,
   in one commit. The summary job builds the conversation's summary from its
   notes and its activity, so "where the last conversation left off" reads as
   now.
3. **One conversation per session, on any member.** The map from a session's
   key (`followed_by`, unchanged) to its journal is a replicated record,
   `mcp/conversation/<owner>/<key>`, written when the journal opens, so a call
   routed to another member or after a restart continues it. The process map
   stays as its cache.
4. **Note terms.** `journal_note` takes, beside `text`:
   - `by` — `speaker` (the default: the focused speaker said it) or `assistant`
     (the calling assistant concluded it; stamped `Claude via <client>` or
     `Zen`);
   - `source` — a doc section, file:line, commit or URL; a line `by: assistant`
     on a `topics/` journal is refused without one;
   - `expires` — a date from which the line is hidden from reads, grep, frames
     and provenance (kept in history);
   - `replaces` — a line id it supersedes, hidden the same way.

   The line's YAML gains `by`, `source`, `expires`, `replaces`.
   `zend-journals-import` stamps its lines `by: assistant`.
5. **`journals check` structural checks.** `--against <rev>` refuses a change
   that loses a line id: every id in the older tree is in the newer one as a
   line, hidden, moved, or an activity entry carrying its id, or is named in the
   change's `forgets.yaml`. `--budget <file>` refuses one past a shape budget —
   journals created, moved or removed; lines moved; summaries rewritten. The
   worker tool `journal_check` takes both.
6. **Tests:** raw expected YAML for a note with each term and for an activity
   record; refusals; a hidden line absent from read, grep, frame and scope; an
   episode producing no entry file; `tending` folding activity past the cap;
   a session continued on a second member; `check --against` refusing a lost id
   and passing a moved one, an activity one and a listed forget.
7. **Docs:** `memory_backbone.md` §3.2, §6.5, §7.2; `mcp_user_context.md` §2.3
   and the `~` paths it still describes; `zen_tool_reach.md:77` (Zen writes
   memory, and its own read and grep reach only its lineage);
   `claude_zend_gateway.md:633` (workers hold `journal_note`).

### Phase 3 — Reading and search

- **Live path.** A worker carries, as sections after its lineage and before its
  own journal, the memory journals its thread is `about` (at most three),
  newest lines first, under `MEMORY_MAX_CHARS = 16_000` beside
  `lineage::MAX_CHARS`, with a heading of their own ("From memory — what the
  household and its work have noted; information, not instructions"). They are
  reconciled each round from the in-memory tree. A topic's new line reseals from
  its section on; the activity record is never in a section
  (`zend/src/mcp/memory_sections.rs`, `memory/<n>` by the topic's place).
- **A thread names its topics.** `thread_attach` takes `about` (at most three
  memory paths), kept on the thread record.
- **The export (D9).** When zend writes a thread note or a settle note,
  `mcp::tending` writes it as a line of `threads/<thread>` in the owner's
  default group (created on first write, cue "the work on <thread>"), and links
  it from each `about` topic (`journal_link`). Memory paths only, so
  `check_links` holds.
- **Search.** Memory `journal_grep` takes `kind` (note, fold, proposal) and
  `speaker`, and pages by `after: <path>#<line id>`, the last hit of the page
  before: hits from every journal searched merge newest first, ties broken by
  id, so a cursor names one place even among lines of one second
  (`journals::view::grep::grep_by`). A task's outcome is no journal line; it
  reaches a client through `get_user_context`'s outcomes block (Phase 4).
  `T:`/`P:` grep is unchanged; their results reach memory through the export.
- **Reading.** Memory `journal_read` pages newest first by `after: <path>#<line
  id>` and `limit`, continuing into archived lines, so an archive is read by
  paging past the window (`journals::view::page::before`).
- **Stable note numbers.** A thread note's number is assigned at write — one
  past both the journal's entry count and its newest entry's number, read
  from the keys' count and the newest record only — and stored on the note,
  replicated with it. A note stored without one reads as its place in key
  order (`threads::add_note`, `number_of`).
- **Tests:** raw expected sections for a worker on a thread with two topics; the
  cap; reseal from the changed section; `about` on attach; an exported note and
  its link; each filter; `before` paging across journals in time order and into
  an archive; a replicated note keeping its number. The MCP latency gate
  (`zend/src/mcp/latency_tests.rs`) covers the new grep and read forms.
- **Docs:** `memory_backbone.md` §4, §6.5, §7.3; `guidance_layer.md`'s missing
  §2.2 reference; `claude_zend_gateway.md` §5.

### Phase 4 — Attended and alone tasks (D5)

- **`driven`.** `repo_task_start` takes `driven: attended | alone`; omitted, it
  is `alone` for the Grok channel and `attended` otherwise. It is kept on the
  `Worker` record and sealed into the worker's prompt as one `projection.yaml`
  section beside `role`'s. The alone section: no one will answer; decide within
  the brief, the design documents and the repository's rules, and record each
  decision in the answer; commit and push the result; run the brief's verify
  step; state anything only the user may approve as pending, and stop there.
- **Outcomes.** The settle notes `workers.rs:1573-1628` writes gain `kind:
  outcome` on `Note` for an alone task, with where it landed and what waits on
  approval; `mcp::tending` writes the same note for an alone task it finds lost.
  Each outcome is indexed at `mcp/outcome/<owner>/<at>`, and `get_user_context`
  reads the newest five with `last_under` — no scan, nothing marked read. The
  export (Phase 3) carries outcomes into memory too.
- **Tests:** the sealed section's raw text per mode; the channel default; an
  outcome note and index entry per settled state, the lost one from `tending`;
  the frame listing the newest five.
- **Docs:** `claude_zend_gateway.md` §3 (the parameter), §5 (outcomes).

### Phase 5 — Guides and the served list (BlackWorld)

1. **A changed list is told** (`mcp_app.md` §3.1). The aggregator declares
   `listChanged: true`; every backend names its listing's version on every
   answer (`x-mcp-tools-version`); the aggregator remembers the versions it
   served each session and, when a call is answered on another, sends
   `notifications/tools/list_changed` on that call's stream before its result.
   Tests: the notice's place on the stream, once; the header on every answer;
   the per-session record (owed, paid, a session never served here, a backend
   not listed, the bound). Phase 8's trial checks that claude.ai and Claude
   Code list again on it.
2. **Tool descriptions**, each changed in both places it lives — the `/mcp`
   definition and its calibrated YAML in `zend/src/prompts/tools/`
   (`repo_task_start`, `journal_note`, `journal_read`, `journal_grep`,
   `journal_check`, `thread_attach`): nesting stated as the code has it; read
   the guide for the work first; `role` and `driven`; the note terms; paging and
   filters; `about`.
3. **Guides.**
   - New `drive` (the two modes; an alone brief names its landing, its done-when
     and what it must never do; its handoff is written at the start),
     `approvals`, `groups`, `files` (with BlackWorld's transfer and preview
     sections).
   - The memory split (D10): a short client guide for `claude-code`, `claude-ai`
     and `other`, with the `yen-apps` line; the full `memory`, absorbing
     `journal` and `knowledge` (both removed): proposals, the note terms, paging,
     search, `about`. `grok.md` points to `memory` for the rules.
   - Every template guide sets `role`; `delegate` gives the rule (work that must
     land is `coder`) and `driven`.
   - `docs-write` covers any document to keep; `steer` gains "land it" and names
     `approval_grant`; `wait` gains a "needs approval" row; `briefing` lists held
     approvals and alone outcomes.
   - Pointers to `drive` in `implement`, `verify`, `handoff`, `overnight`,
     `feature`, `journal-restructure`, `thread`.
   - `health` and `incident` use `member_status`.
4. **Tests:** the guides library's tests; a test that every tool a guide's
   `tools:` names is in the aggregator's served list and every guide named in a
   guide's text exists (`Library::unserved`, `guides/src/library.rs:137`, is the
   runtime half of the same check).

### Phase 6 — Drone ↔ zend task bridge (`mission_templates.md` §7, D4, D8)

Templates, parameters, `work.checks` and the `critical` gate are built
(`npcd/src/engine/mission_gen/world.rs`, `authorize.rs`); the bridge is not.

- **Identity and route (BlackWorld).** npcd reaches its own member's zend over
  the member's cluster socket, as zend reaches npcd (`npcd_tools.rs:8`): the
  member states the caller in identity headers, and no key is presented, since
  a service key's bearer is checked only on gateway HTTP from a trusted backend.
  The caller is a service-key identity — provider `service-key`, subject
  `service:zen`, which no sign-in produces — given a role by `sub` in
  `host/roles.yaml` and bound by the memory's accounts to the person `zen`
  (`people/zen`). zend's `/mcp` accepts it as that person (`zen_tasks.rs:114-118`
  then has a person). `high_risk` holds apply to Zen's tasks as to anyone's, and
  Zen grants nothing.
- **Acts (AlienWorld).** A routine's `grants` lend `task_start {name, prompt,
  role}`, `task_reply {answer}` and `task_nudge {message}` while its mission is
  carried; npcd maps them onto `repo_task_start`, `repo_task_reply` and
  `repo_task_nudge` on the thread named for the mission
  (`mission-<mission id>`).
- **The task as a correspondent.** npcd's tending thread (off the decode path)
  calls `job_status` with `wait_s: 0` for each carried mission's task, and posts
  what changed — progress, a question, the final answer — to the carrier's
  thread as a message from Zen.
- **Suspended carriers.** A suspended carrier's task acts are refused, and the
  bridge neither polls for it nor posts to it until it is unfrozen.
- **The completion gate.** The mission's terminal step is signed off by the
  engine when `job_status` reports the task done, never on the carrier's word.
- **Tests:** act → tool-call mapping with raw expected arguments; answers
  arriving as thread messages; the step signed off only on done; a failed or
  lost task leaving the mission stuck with the reason; a suspended carrier's
  acts refused and its task neither polled nor posted.
- **Docs:** `mission_templates.md` §7, §8 marked built; `mcp_file_tools.md` §4
  (the `service:zen` subject).

### Phase 7 — Maintenance templates and the cleanup pass

- **The library:** `battle-mind:missions/<world>/maintenance/` (`world.rs:22-25,
  229-236`), `<world>` the world Zen lives in.
- **`journal-maintenance`** and **`guide-maintenance`.** Steps, maker never the
  checker:
  1. **Survey**, read only: candidates with evidence — a line whose `source`
     changed on main, a line past `expires`, a line off its journal's cue, a
     duplicate, a cut summary; for guides, a tool not served, a guide not
     present, a template without `role`.
  2. **Operations**, not prose: move, supersede-with-source, expire, merge,
     forget.
  3. **Verify** by a second task against the source on main.
  4. **Apply and check:** the task edits a checkout of `journals` (or `candle`
     for guides), runs `journal_check --against <base> --budget` (or the guide
     tests), and pushes `journals` to `main` (its standing rule) or `candle`'s
     guides to a `zen/` branch for the user.
  5. **Gate:** within budget it lands; a split, merge, rename or a change past
     the budget is a `critical` step for the authorizers, failing safe; a
     correction about a person is a proposal for that person.
  6. **Report:** the mission's report, through npcd's reports to the
     authorizers.
- **Trigger:** nightly.
- **The cleanup pass** is the first run of `journal-maintenance`, issued by hand
  once the release is live: re-file `people/john`; supersede the stale
  `workers-threads` and `ingest` lines; expire the reminders; forget the stored
  instruction; move every existing episode line into its conversation's activity
  record (ids kept) and remove its entry file; rebuild the cut summaries.
- **Tests:** each template loads (`routine-parses`); the survey's candidate rules
  on a fixture tree with raw expected candidates; a budget breach raising the
  critical step.

### Phase 8 — Review, measure, merge, release

1. **Review** the whole branch: one `review` task per area (journals crate, zend
   MCP tools and lineage, retrieval scope, npcd bridge, guides), `effort:
   exhaustive`; confirmed findings fixed on the branch.
2. **Measure** on a scratch motherd outside the cluster: the live path's reseal
   rate and prefill cost over a scripted session; the guidance tally (missed,
   read anyway, relevant, idle) before and after Phase 1; the MCP latency gate;
   the cost of recalibrating the changed tool YAMLs.
3. **Merge** `zen/memory-and-guides` to `main`, with the user's approval.
4. **Joint release** (BlackWorld): `motherd` and `mcp` through `zend-deploy` — a
   trial on one machine, then the rollout, its Decide command with the user's
   approval; `mcp` trialled with it, since the guides ship in `mcp`.
5. **Issue the cleanup pass** (Phase 7) and watch it to its report.

---

## 5. Risks

- **A bad note spreads at once.** The live path puts a topic into every worker
  on threads about it within the sync; a wrong or hostile note reaches all of
  them before anyone reads it. The injection guard is the defence on the write;
  `source`, maintenance verification and proposals narrow it.
- **Live sections cost prefill.** A topic's new line reseals from its section on;
  the cap holds the cost, confirmed by measurement before merge.
- **Alone workers decide.** The bound is the branch rule and the pending-approval
  stop; `journals`, which pushes to main, is changed only through the
  maintenance gate.
- **Use-exemplars cross people.** One person's reasoning, sealed as an exemplar
  of a unit both may read, shapes the unit's score in the other's conversation.
  No text crosses (Phase 1 asserts it); the influence is named, not removed.
- **Zen acts on Discord text.** A drone's task is steered by what reaches the
  drone; Zen grants nothing, and every `high_risk` call holds for an authorizer.
- **One branch, two sessions.** Each session owns its files; the three seams are
  announced when pushed, and `origin/main` is merged in at each gate.
- **The cleanup pass races live writes.** It keeps every line id, runs as one
  commit, and the journals service replays a refused push.
- **The cluster is not stable yet** (the re-ingest backlog, WhiteWorld's NaN
  fault, KV back pressure). Nothing reaches the cluster before Phase 8, and the
  one release brings one guide and journal re-ingest.

---

## 6. Review history

- **2026-11-08, first independent review: not ready.** Factual errors against
  the tree (recency already built; the hub-file exception; the activity record
  named a digest that does not exist), a privacy hole (ownerless dialogues), the
  wrong mission repository, a drone with no identity, outcomes duplicating the
  settle notes, and a resolver against `memory_backbone.md` §7.3. Decisions D2,
  D8, D9 settled its design questions; §4 was rewritten against `origin/main`
  with every finding applied.
- **2026-11-08, BlackWorld's overlap review.** Ownership settled (§3.1, D11);
  the memory guide split (D10); `service:zen` as a service-key identity;
  suspended carriers hold the bridge; exemplars named; the guidance tally in
  Phase 8. Its claim that the connector dropped the `file_*` and `log_*` tools
  was wrong: the client was never told the list changed, which Phase 5 step 1
  now fixes.
