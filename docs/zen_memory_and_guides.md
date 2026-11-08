# Zen's memory, guides and task driving: the build

**Design and build plan, not built.** The work that follows from three audits of
2026-11-08 — the MCP guides against the tool set, how a task is driven from each
client, and the journals (how they are structured, written, searched and read,
and how they reach conversations through provenance and projection) — written to
be carried out as **one agentic flow**: one branch, the phases below in order,
each closed by its gate, one review, one merge, one rollout.

The design documents each phase names (`memory_backbone.md`, `guidance_layer.md`,
`mcp_user_context.md`, `mission_templates.md`) are revised in the phase that
changes what they govern, and are authoritative over this plan where they are
more specific. This plan is authoritative over the order, the gates and the
decisions in §2.

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
- **`repo_task_start` says a task "starts no tasks of its own"**, while
  `implement`, `review` and `research` have the worker start a `verify-*` task,
  and the code nests tasks one level (`zend/src/mcp/zen_tasks.rs`,
  `a_nested_task_holds_out_the_task_tools`).
- **Tools with no guide:** `approval_grant`; `group_create`, `group_describe`,
  `group_add`, `group_remove`, `group_default`; `member_status`; the `file_*`
  tools outside Yen's repository. `steer` says a reply cannot approve and to ask
  the user, but never names `approval_grant`, the one way a held call runs.
- **`memory` lists `journal_accept` and never explains proposals.** `grok.md`
  repeats the memory rules instead of pointing at `memory`, and the copies have
  drifted. `journal` and `knowledge` cover the same ground as `memory`.
- **`docs-write` covers documents from code only**; no guide lands a design
  proposal, the document the DJ task wrote.

### 1.2 Journals

"Journals" are five stores: the memory tree (git, the `journals` repository);
thread and project notes `T:`/`P:` (substrate records, `zend/src/mcp/threads.rs`);
a worker's journal (`journal-core` entries of its conversation); a dialogue's own
journal (`journal.kept`); and `notes_*` (SQLite). No read or search crosses them,
though `memory_backbone.md` §6.5 says the memory paths replace `T:` and `P:`.

The live tree (read 2026-11-08):

- **Episodes swamp it.** Every MCP call that is not read-only is written as an
  episode entry (`mcp_user_context.md` §2.3): one conversation's `here` held
  eight raw `log_grep` regexes and no decision; `conversations` holds 193
  journals of 1–4 lines whose folds read "John used Claude Code to read bot
  missions". No outcome is recorded.
- **One Claude Code session becomes many conversations**: journals opened at
  00:08, 00:19, 00:29, 00:37 … for one session.
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
  hits with no offset, filters by nothing (speaker, kind, time), follows no
  reference, and never reaches `T:`/`P:`. `T:<thread>/*` concatenates worker by
  worker, then truncates. `offset` means "first entry" on `T:`/`P:` and is
  ignored on memory.
- Thread notes are numbered by their place in key order, and keyed by time and a
  per-process sequence (`threads.rs`), so a note replicated with an earlier stamp
  may renumber every later `#N` — Phase 0 proves or drops this.

How journals reach conversations:

1. **Lineage sections** — `T:`/`P:` notes enter every live worker's prompt the
   round after they seal, no prefill, capped at 8K tokens
   (`zend/src/mcp/lineage.rs`). Real time, one lineage.
2. **MCP frames** — `get_user_context` reads the in-memory tree: at once on the
   member that wrote, within the 5 s sync and a fetch elsewhere.
3. **Provenance** — the `guidance` group (`zend/src/prompts/projection.yaml`,
   `guidance_layer.md`) draws every `journals/` entry file as a `code_reading`
   unit: the only path by which one conversation's note reaches another zend
   conversation. It costs a push, a branch ingest, a reading conversation per
   file, calibration, and the base taking the new version — minutes, hours in a
   backlog. The recency term (`guidance_layer.md` §4, build step 6) is not built,
   so a stale note and its correction both match. Each episode is an entry file,
   so ingest work scales with tool calls, not with what was learned (Phase 0
   measures how much). Nothing seen filters journal units by group (Phase 0
   proves or drops this).

---

## 2. Decisions

Settled before the build; a phase that meets one of these does not reopen it.

| # | Decision |
|---|---|
| D1 | **One branch, `zen/memory-and-guides`, from `origin/main`.** Every phase commits and pushes to it. It merges to `main` once, after Phase 8's review, and is rolled out once. |
| D2 | **A conversation sees only its person's groups' journal content, on every route into a prompt** (the `guidance` draw, hub files, folder units, derived layers, `persons/`). Its person is the `/mcp` owner, or, for a dialogue from the Zen app or VS Code, the person signed in when it was opened, recorded on the conversation. **Only the ingest and calibration readers, named in the code, see every group.** |
| D3 | **Episodes leave the memory tree.** A conversation's tool activity is kept in its own journal's folds — so "where the last conversation left off" still works — and is never an entry file, so provenance never ingests it. |
| D4 | **Maintenance is a mission.** This build includes the drone ↔ zend-task bridge (`mission_templates.md` §7) and the `journal-maintenance` and `guide-maintenance` templates on it. There is no second maintenance mechanism. |
| D5 | **A task is driven `attended` or `alone`**, defaulting from the client: `alone` for Grok, `attended` otherwise. A drone driving a task is an attended coordinator. |
| D6 | **No backward compatibility.** Existing episode entry files, stale lines and the old guide text are replaced, not kept beside the new. |
| D7 | **Simplicity cuts** (from the review of the first draft): no verify task chained by zend after an alone task — the templates carry one; no automatic restart of a lost alone task — it is reported; the live path reuses lineage sections; maintenance checks live in `journals check`. |
| D8 | **A drone's zend task runs as Zen's own person**, a service person added for Zen: its tasks, threads and outcomes are its own, it grants nothing, and its held calls go to the authorizers (the Creator, Yen). |
| D10 | **The memory guide splits in two.** A short client guide rides the first call of every Claude conversation (`mcp_user_context.md` §2.4–2.5's budget): the frame, the write rules in brief, pointers, the `yen-apps` line. A full `memory` guide, absorbing `journal` and `knowledge`, is read on demand. `grok.md` stays the Grok client guide. |
| D11 | **One sitting, AlienWorld and BlackWorld together, on the user's command.** BlackWorld's overlapping pieces join this build rather than landing ahead of it: dropping `conversations/` from the `guidance` draw (the first step of Phase 2), and the transfer and preview sections of the `files` guide and the `yen-apps` text (`mcp_file_tools.md`, `yenbot_preview.md`) in Phase 5. Nothing starts before the command. |
| D9 | **Thread results are exported into the memory tree** (`memory_backbone.md` §7.3): a thread's notes and its tasks' outcomes are written to memory journals under `threads/` and referenced from the topics they are about. `T:`/`P:` stay the work records; there is no resolver over them. |

---

## 3. How the flow runs

- **Coordinator: Claude** (the session running the flow). It writes each phase's
  spec from this document, notes decisions on the thread `memory-and-guides`,
  verifies each worker's result against the turns it cites, reads every diff,
  and holds every approval.
- **Workers: Zen** (`repo_task_start`, `role: coder`, `effort: deep`;
  `exhaustive` for reviews), one task per phase part, all on
  `zen/memory-and-guides`, briefed from the phase's spec with the `implement`
  template. Parts that touch disjoint crates run together.
- **Builds and tests: a build agent** (`repo_job_start`), one cargo process at a
  time, the narrowest scope that covers the phase.
- **Each phase closes on its gate:** the phase's tests pass on an agent; `cargo
  fmt --all -- --check` and `cargo clippy --workspace --tests --examples -- -D
  warnings` are clean; the coordinator has read the diff; the phase's design-doc
  revision is in the same commits. A phase that fails its gate is fixed before
  the next starts — never carried forward.
- **Standing rules bind every phase:** CLAUDE.md (file tools only, no shims, no
  flags, no TODOs, imports not paths, one concern per file, tests asserting raw
  expected values), the MCP request-path invariants (every tool that runs no
  model answers within 30 ms at p95; a read never writes; no full scans; no
  network under a lock a request takes), and nothing deployed from the branch.
- **Approvals the coordinator asks for:** the merge (Phase 8), the rollout's
  Decide command, and any `critical` step the cleanup pass reaches.

---

## 4. Phases

### Phase 0 — Prove the open findings

One `ask` task, read only (`role: researcher`), answering with file:line:

1. Is an MCP-started task `Level::Task`, able to start one nested task?
2. Does anything between `retrieval_scope` and the `guidance` draw filter
   `journals/` units by group? Name the place a group filter belongs.
3. Can a thread note replicated from another member renumber the notes after it?
4. How many `code_reading` units in the current ingest backlog are `journals/`
   entry files, and how many of those are episodes?

**Gate:** each answer verified against its cited lines. A "no" to 2 or 3 removes
that part of Phase 1 or Phase 3 from the build, noted on the thread.

### Phase 1 — Journal units obey the group boundary (D2)

- `retrieval_scope` takes the conversation's owner. For an owned conversation, a
  `journals/` unit is a candidate only when its path's group (`groups/<g>/…`) is
  one the owner reads, decided from the membership the MCP memory tools already
  use. An ownerless conversation keeps every group's units.
- The scope cache is keyed by owner as well as base, so two owners never share a
  scope.
- **Tests:** an owned conversation never draws a unit of a group its owner is not
  in; it draws one of a group it is in; an ownerless conversation draws both; a
  guide unit (`candle/mcp/src/guides/`) is unaffected.
- **Docs:** `memory_backbone.md` §3.1, §7.2 (the boundary in retrieval);
  `guidance_layer.md` §1.

### Phase 2 — Writing memory well (D3, D6)

`journals` crate, the aggregator's episode path, and the memory tools.

- **Episodes into folds.** `user_episode_note` stops writing entry files. A
  conversation's activity is appended to its conversation journal's activity
  record (the `journal.yaml` digest the fold sweep already maintains), from which
  its folds and "where you left off" are built. `get_user_context`'s last-left-off
  line reads it.
- **One conversation per client session.** The aggregator keys a conversation
  journal by the client session, not the transport connection, so reconnects
  continue it.
- **Note terms.** `journal_note` takes, beside `text`:
  - `source` — a doc section, file:line, commit or URL. A line the speaker said
    needs none; its stamp is its source. A line stamped as Claude's or Zen's own
    conclusion on a topic journal is refused without one.
  - `expires` — a date; from it the line is hidden from reads, grep and frames,
    kept in history.
  - `replaces` — a line id it supersedes, hidden the same way from then on.
- **Honest speakers.** A line an assistant derived is stamped as that assistant's
  (`Claude via Claude Code`, `Zen`), not as the person's. The import tool
  stamps its lines the same way.
- **`journals check` structural checks** (used by Phase 7): `--against <tree>`
  refuses a change that loses a line id (every id in the older tree exists in the
  newer one, live or hidden by `expires`/`replaces`/forget), and `--budget`
  refuses one past a shape budget (journals created, moved or removed; lines
  moved; summaries rewritten).
- **Tests:** raw expected YAML for a note with each term; a hidden line absent
  from read, grep and frame output; the refusals; an episode producing no entry
  file and appearing in the fold digest; reconnect continuing one conversation;
  `check --against` refusing a lost id and passing a moved one.
- **Docs:** `memory_backbone.md` §3.2, §6.5, §7.2; `mcp_user_context.md` §2.3 and
  the `~` paths it still describes; `zen_tool_reach.md:77`;
  `claude_zend_gateway.md:633`.

### Phase 3 — Reading and search

- **Live path.** A conversation with a memory `here` (an MCP conversation, a
  thread, a worker) carries, as lineage sections, the journals it is focused on:
  its `here`, its focus's `about` topics, the speaker's `people/` journal; a
  worker carries its thread's. Reconciled each round from the in-memory tree,
  `here` last so the most-changed section reseals least, under a cap of 4,000
  tokens newest lines first. The cap is confirmed in Phase 8's measurement.
- **Recency.** `guidance_layer.md` §4's term, built as specified
  (`recency: { half_life_days: 30, dated_by: path }` on the `guidance`
  selection).
- **One resolver.** `threads/<thread>[/<worker>]` and `projects/<repo>` resolve
  as memory paths for `journal_read` and `journal_grep`, onto the `T:`/`P:`
  stores where they live. `T:`/`P:` addresses are removed from the tools (D6).
- **Search.** `journal_grep` takes `kind` (note, fold, proposal, outcome),
  `speaker`, `since`, `until` and `offset`; hits from every scope merge newest
  first and page instead of truncating. Each scope is read by key range — no full
  scans.
- **Reading.** Memory `journal_read` pages its lines (`offset`, `limit`) and
  reads an archive by its period (`<path>@2026-W44`).
- **A thread names its topics.** `thread_attach` takes `about`; a thread's notes
  and its workers' outcomes appear under those topics' `threads` references —
  `memory_backbone.md` §7.3's export.
- **Stable note numbers** (if Phase 0.3 found them unstable): a note's number is
  assigned once, at write, and replicated with it.
- **Tests:** raw expected sections for a focused conversation; reseal order; the
  cap; recency scaling on a dated unit and none on a guide; each filter; paging
  across scopes in time order; an archive read; a thread's outcome under its
  topic. The MCP latency gate (`zend/src/mcp/latency_tests.rs`) covers the new
  grep and read forms.
- **Docs:** `memory_backbone.md` §4, §6.5, §7.3; `guidance_layer.md` §4, §5 and
  its missing §2.2 reference; `claude_zend_gateway.md` §5.

### Phase 4 — Attended and alone tasks (D5)

- `repo_task_start` takes `driven: attended | alone`, defaulting from the
  client. It is sealed into the worker's system prompt as one
  `projection.yaml` section, as `role` is. The alone section tells the worker
  that no one will answer: it decides within its brief, the design documents and
  the repository's rules and records each decision in its answer; commits and
  pushes its own result; runs its brief's verify step; states anything only the
  user may approve as pending and stops there.
- **Outcome delivery.** When an alone task settles — done, failed, stuck or
  lost — zend writes its outcome as an `outcome` line on the thread's memory
  journal (Phase 3), stamped `Zen`, with what was done, where it landed and what
  waits on approval. `get_user_context` lists the owner's outcomes from threads
  not yet archived, newest first. It marks nothing read: a read never writes.
- **Tests:** the sealed section's raw text per mode; the client default; an
  outcome line written for each settled state; the frame listing it and dropping
  it once its thread is archived.
- **Docs:** `claude_zend_gateway.md` §3 (the parameter), §5 (outcomes).

### Phase 5 — Guides (one change)

Every guide edit of the audit, against the tools as Phases 2–4 leave them:

- `repo_task_start`'s description: nesting stated as Phase 0.1 found it; read the
  guide for the work first; `role`'s meaning; `driven`. Its param descriptions
  likewise.
- **New** `drive` (the two modes; an alone brief names its landing, its
  done-when and what it must never do; its handoff is written at the start),
  `approvals`, `groups`, `files`.
- **Every template guide sets `role`**; `delegate` gives the rule (work that must
  land is `coder`) and `driven`.
- `docs-write` covers any document to keep, a design proposal included;
  `steer` gains "land it" and names `approval_grant`; `wait` gains a "needs
  approval" row; `briefing` lists held approvals and alone outcomes.
- `memory` absorbs `journal` and `knowledge` (both removed), covers every
  client, proposals, the note terms, paging and search; `grok.md` points to it.
- Pointers to `drive` in `implement`, `verify`, `handoff`, `overnight`,
  `feature`, `journal-restructure`, `thread` (`about` on attach).
- `health` and `incident` use `member_status`.
- **Tests:** the guides library's existing tests; a new test that every tool a
  guide's `tools:` names is in the served set and every guide named in a guide's
  text exists — the same checks Phase 7's `guide-maintenance` runs.

### Phase 6 — Drone ↔ zend task bridge (`mission_templates.md` §7, D4)

Templates, parameters, `work.checks` and the `critical` gate are built
(`npcd/src/engine/mission_gen/world.rs`, `authorize.rs`); the bridge is not.

- **The drone is an MCP client.** npcd calls zend's `/mcp` through the gateway
  under its own client identity (`npcd`), so the bridge reuses every task tool,
  owner check and audit as they stand, and a drone's task is `attended` (D5).
- **Acts.** A coding or maintenance mission's carrier gets `task_start {name,
  prompt, role}`, `task_reply {answer}` and `task_nudge {message}` while it
  carries the mission; npcd maps them onto `repo_task_start`,
  `repo_task_reply` and `repo_task_nudge` on the thread named for the mission.
- **The task as a correspondent.** npcd polls `job_status` for the mission's
  task each tick, off the decode path, and posts what it answers — progress, a
  question, the final answer — to the carrier's thread as a message from Zen.
- **The completion gate.** The mission's terminal step is signed off by the
  engine when `job_status` reports the task done, never on the carrier's word,
  as the doc-commit gate is.
- **Tests:** act → tool-call mapping with raw expected arguments; a task's
  answers arriving as thread messages; the step signed off only on done; a
  failed or lost task leaving the mission stuck with the reason.
- **Docs:** `mission_templates.md` §7, §8 marked built.

### Phase 7 — Maintenance templates and the cleanup pass

- **`journal-maintenance`** and **`guide-maintenance`** in the journals
  repository's mission library (`missions/maintenance/`). Steps, maker never the
  checker:
  1. **Survey**, read only: candidates with their evidence — a line whose
     `source` changed on main, a line past `expires`, a line off its journal's
     cue, a duplicate, a cut summary; for guides, a tool not served, a guide not
     present, a template without `role`.
  2. **Operations**, not prose: move, supersede-with-source, expire, merge.
  3. **Verify** by a second task against the source on main.
  4. **Checks** by code: `journals check --against --budget` (Phase 2), or the
     guide checks (Phase 5).
  5. **Gate:** within budget it lands; a split, merge, rename or a change past
     the budget is a `critical` step for the configured authorizers, failing
     safe; a correction about a person is a proposal for that person.
  6. **One commit**, reported as an outcome (Phase 4).
- **Triggers:** a merge to main that changes a file a line's `source` names; a
  change to a tool a guide lists; nightly for expiry, misfiling and summaries.
- **The cleanup pass** is the first run of `journal-maintenance`, issued by hand
  once the build is live: re-file `people/john`; supersede the stale
  `workers-threads` and `ingest` lines; expire the reminders; remove the stored
  instruction; move every existing episode entry file into its conversation's
  fold digest and delete the file (D3, D6); rebuild the cut summaries.
- **Tests:** each template loads (`routine-parses`); the survey's candidate rules
  on a fixture tree with raw expected candidates; a budget breach raising the
  critical step.

### Phase 8 — Review, measure, merge, roll out

1. **Review** the whole branch: one `review` task per area (journals crate,
   zend MCP tools and lineage, retrieval scope, npcd bridge, guides), `effort:
   exhaustive`; confirmed findings fixed on the branch.
2. **Measure** on a scratch motherd outside the cluster (CLAUDE.md, "Test
   locally, not clustered"): the live path's reseal rate and prefill cost over a
   scripted session, confirming or changing the 4,000-token cap; the MCP latency
   gate.
3. **Merge** `zen/memory-and-guides` to `main` — with the user's approval.
4. **Roll out** through the `zend-deploy` skill: a trial on one machine, then the
   rollout — its Decide command with the user's approval. `mcp` is rebuilt and
   trialled with it, since the guides ship in `mcp`.
5. **Issue the cleanup pass** (Phase 7) and watch it to its report.

---

## 5. Risks

- **A bad note spreads at once.** The live path puts a focused journal into
  every conversation within the sync; a wrong or hostile note reaches all of
  them before anyone reads it. The injection guard is the defence on the write;
  `source`, maintenance verification and the proposal path narrow it.
- **Live sections cost prefill.** Each note reseals from the first changed
  section; `here` goes last and the cap holds the cost, confirmed by measurement
  before merge.
- **Alone workers decide.** The bound is the branch rule and the pending-approval
  stop; the journals repository, which pushes to main, is changed only through
  the maintenance gate.
- **Ownerless conversations see every group (D2).** A calibration or ingest
  conversation's output is not shown to a person, but it seeds exemplars that
  owned conversations' scoring reads. A unit's exemplars are its own and the
  filter applies to the unit, so no other group's text is placed in an owned
  prompt — Phase 1's tests assert exactly that.
- **One branch is a large merge (D1).** Every phase is reviewed as it closes and
  `origin/main` is merged into the branch at each phase gate (never a rebase: the
  branch is pushed), so the final merge carries no surprise.
- **The cleanup pass races live writes.** It keeps every line id, runs as one
  commit, and the journals service replays a refused push.
- **The cluster is not stable yet** (the re-ingest backlog, WhiteWorld's NaN
  fault, KV back pressure). Nothing reaches the cluster before Phase 8, and the
  one rollout brings one guide and journal re-ingest, not one per phase.

---

## 6. Readiness: what the phases must absorb before the build starts

An independent review (2026-11-08) judged §4 **not ready**: the approach holds,
but several phases were written against an older tree and would leave a worker
guessing. Decisions D2, D8 and D9 settle its design questions. This list is what
the rewrite of §4 must clear, re-checked against `origin/main`; the build starts
when a second review finds none of it open.

- **Phase 0** — questions 0.1–0.3 are answered by the code (tasks nest one level,
  `zen_tasks.rs:80, 157`; no group filter, `retrieval_scope.rs:61, 273`; note
  numbers shift, `threads.rs:439-440, 479`). Only 0.4, the backlog count, stays,
  with its data source named.
- **Phase 1** — the filter applies in `scope_of` to every route, not the
  `guidance` draw alone; `persons/<slug>/person.yaml` is its person's only; a
  line hidden by `expires`/`replaces` leaves the scope too; the dialogue's
  person is recorded at open (D2).
- **Phase 2** — the activity record is new: its YAML shape and cap on the
  conversation journal's `journal.yaml`, its writer, and the summary job reading
  it (only the master and `people/` carry a digest today, `check.rs:32-43`);
  conversation journals leave every provenance group; the conversation key moves
  with the session across restarts and members
  (`mcp/memory/conversations.rs:10-13, 102-108`); note terms gain `by`; the
  import tool is named; `journal_check` gains `--against`/`--budget`.
- **Phase 3** — the recency term is built (`projection.yaml:1048`,
  `guidance_layer.md:187`): its bullet goes. The resolver goes (D9); the export
  is specified instead. The live path's cap is stated against
  `lineage::MAX_CHARS`; the activity record stays out of `here`'s section; which
  conversations carry a memory `here` is named; memory sections get their own
  heading. Search pages by an `until` cursor, not `offset`, and maps `kind` and
  `speaker` onto `T:`/`P:` notes, which carry neither today.
- **Phase 4** — outcomes extend the settle notes zend already writes
  (`workers.rs:1573-1628`, a `kind` on `Note`); `mcp::tending` writes the lost
  state; `get_user_context` reads an `mcp/outcome/<owner>/` index with
  `last_under` and a count cap, never a scan.
- **Phase 5** — the served-set test is placed where all three tool lists are
  visible, or replaced by `Library::unserved` (`guides/src/library.rs:137`); the
  calibrated zend tool YAMLs (`zend/src/prompts/tools/`) change with their `/mcp`
  definitions, and their recalibration is counted in Phase 8.
- **Phase 6** — the Zen service person (D8); the in-process path and its auth
  from npcd to zend inside motherd; the task acts granted through the routine's
  `grants`; drone-driven tasks keep `high_risk` holds.
- **Phase 7** — the library is `battle-mind:missions/<world>/maintenance/`
  (`world.rs:22-25, 229-236`); how the carrier's task edits the journals
  checkout is stated; `check --against` allows the episode moves the cleanup
  pass makes; the report is a drone-task outcome, not an alone one.

BlackWorld's review of the overlap (2026-11-08) adds:

- **§1.2** — the `guidance` draw already excludes the hub files
  (`except: [README.md, group.yaml, journal.yaml, person.yaml]`, 37f13b0ca) and
  the recency term is built; item 3 is corrected to say so.
- **Phase 1 and §5** — use-exemplars (a conversation's request and reasoning
  sealed as calibration of a guide or journal unit, `zend/src/guidance/`) are
  provenance only, never placed, yet one person's reasoning shapes scoring in
  another's conversation; the risk is named and Phase 1 asserts no exemplar is
  placed as text. The scope work keeps the draw's `except` and its share pool,
  and its tests include a drawn guide and a drawn journal under one owner.
- **Phase 2** — opens with `conversations/` added to the `guidance` draw's
  `except`, with its test (D11).
- **Phase 5** — the served-set test reads the aggregator's served list: the live
  connector serves none of `file_*`, `log_*` or `cluster_logs` although zend
  defines them (`zend/src/mcp/tools/mod.rs`), so where they are dropped —
  aggregator, `mcp/served.yaml` or the connector's cached listing — is found and
  fixed first (`yenbot_preview.md` §8.3). The `memory` split per D10.
- **Phase 6** — Zen's person is `service:zen` on the service-key path
  (`mcp_file_tools.md` §4, `web/src/cluster/service_key.rs`). A suspended
  carrier holds the bridge too: its task acts are refused and the bridge's
  polling and posting never wake it, with a test.
- **Phase 0.4** — the count comes from `ingest_backlog` (`item_done` by path)
  and the branch pass's plan line; episodes are the entries under
  `conversations/`.
- **Phase 8** — placement is measured with the guidance tally (missed, read
  anyway, relevant, idle) beside the reseal cost.
