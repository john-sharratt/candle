# Zen's memory, guides and task driving: the plan

**Design and work plan, not built.** The work that follows from three audits of
2026-11-08: the MCP guides against the tool set, how a task is driven from each
client, and the journals: how they are structured, written, searched and read,
and how they reach conversations through provenance and projection. Each
workstream below is specified enough to brief as a task; the design documents it
names (`memory_backbone.md`, `guidance_layer.md`, `mission_templates.md`) are
revised in the same change as the code they govern, and win over this plan where
they are more specific.

Status marks: **[verify]** is a finding inferred from the code, not proven; its
workstream opens by proving or dropping it.

---

## 1. What the audits found

### 1.1 Guides

- **Every method assumes someone stays.** Each guide's coordinator waits, answers
  questions, verifies and brings approvals to the user. A Grok task cannot: the
  car drops the conversation seconds after the driver stops talking, so a task's
  question reaches no one, its result is checked by no one, and nobody reads the
  thread it ends on. The DJ design task (2026-11-07) reached a written document
  and stopped at "shall I commit it?".
- **`role` is set by no guide.** Templates say "read only" in CONSTRAINTS and run
  as `coder`; the DJ task ran as `researcher` and was asked to write a document.
- **`repo_task_start` says a task "starts no tasks of its own"**, while
  `implement`, `review` and `research` have the worker start a `verify-*` task,
  and the code nests tasks one level (`zend/src/mcp/zen_tasks.rs`,
  `a_nested_task_holds_out_the_task_tools`). **[verify]** that a task started
  over MCP is `Level::Task`.
- **Tools with no guide:** `approval_grant`; `group_create`, `group_describe`,
  `group_add`, `group_remove`, `group_default`; `member_status`; the `file_*`
  tools outside Yen's repository. `steer` says a reply cannot approve and to ask
  the user, but never names `approval_grant`, the one way a held call runs.
- **`memory` lists `journal_accept` and never explains proposals.** `grok.md`
  repeats the memory rules instead of pointing at `memory`, and the copies have
  drifted.
- **`docs-write` covers documents from code only**: no guide lands a design
  proposal, which is the document the DJ task wrote.

### 1.2 Journals

"Journals" are five stores: the memory tree (git, the `journals` repository);
thread and project notes `T:`/`P:` (substrate records); a worker's journal
(`journal-core` entries of its conversation); a dialogue's own journal
(`journal.kept`); and `notes_*` (SQLite). No read or search crosses them;
`memory_backbone.md` §6.5 says the memory paths replace `T:` and `P:`, and they
do not.

The live tree (read 2026-11-08):

- **Episodes swamp it.** Every MCP call that is not read-only is written as an
  episode line (`docs/mcp_user_context.md` §2.3): this conversation's `here`
  held eight raw `log_grep` regexes and no decision; `conversations` holds 193
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
  at 32,000; the cut says to use `offset`/`limit`, which the memory branch
  ignores (`journal_read.rs`, `fit.rs`).
- Archived lines are reached only by grep or an exact address. Grep stops at 50
  hits with no offset, filters by nothing (speaker, kind, channel, time), follows
  no reference, and never reaches `T:`/`P:`. `T:<thread>/*` concatenates worker
  by worker, then truncates. `offset` means "first entry" on `T:`/`P:` and is
  ignored on memory.
- **[verify]** Thread notes are numbered by key order, keyed by time and a
  per-process sequence (`threads.rs`); a note replicated with an earlier stamp
  would renumber every later `#N`.

How journals reach conversations:

1. **Lineage sections** — `T:`/`P:` notes enter every live worker's prompt the
   round after they seal, no prefill, capped at 8K tokens. Real time, one
   lineage.
2. **MCP frames** — `get_user_context` reads the in-memory tree: at once on the
   member that wrote, within the 5 s sync and a fetch elsewhere.
3. **Provenance** — the `guidance` group (`projection.yaml`, `guidance_layer.md`)
   draws every `journals/` entry file as a `code_reading` unit. This is the only
   path by which one conversation's note reaches another zend conversation. It
   costs a push, a branch ingest, a reading conversation per file, calibration,
   and the base taking the new version: minutes, hours during a backlog. The
   recency term (`guidance_layer.md` §4, build step 6) is not built, so a stale
   note and its correction both match.
   - **[verify]** each episode is an entry file and so a unit to ingest: ingest
     work scales with tool calls, not with what was learned.
   - **[verify]** `retrieval_scope` filters journal units by no group or owner,
     so a conversation can retrieve another group's entries — against the group
     boundary of `memory_backbone.md` §3.1, §7.2.

---

## 2. The work, by workstream

Seven workstreams. Each lands as one change: its design text first, then code,
tests and guides together.

### W0 — Prove the [verify] findings (one read-only task)

One `ask` task, read only, answering four questions with file:line:

1. Is an MCP-started task `Level::Task`, able to start one nested task?
2. Does `retrieval_scope` (or anything downstream of it) filter `journals/`
   units by the groups the conversation's owner reads?
3. Can a replicated thread note renumber the notes after it?
4. Is each episode an entry file in `journals/`, and how many of the units in
   the current ingest backlog are journal entries?

Each answer either opens its fix below or drops it from the plan.

### W1 — Privacy: journal units obey the group boundary

**Only if W0.2 finds the gap.** `retrieval_scope` admits a `journals/` unit only
when its path's group is one the conversation's owner reads, decided from the
same membership the MCP tools use. Tests: a unit of a group the owner is not in
is never a candidate; one of a group they are in is. This lands before anything
else in this plan.

### W2 — Guide corrections (one guide change, `zen/guides-correct`)

What is wrong today for every client, apart from how tasks are driven:

- `repo_task_start`: the nesting sentence made true (per W0.1); "read the guide
  for the kind of work first"; `role`'s meaning — `coder` changes and commits,
  `researcher` and `reviewer` read and report.
- Every template guide sets the `role` its outcome needs: `ask`, `implement`,
  `review`, `research`, `debug` (both variants), `docs-write` (both),
  `test-write`, `refactor`. `delegate` states the rule: work that must land is
  `coder`.
- `docs-write` covers any document to keep, a design proposal included, on
  `zen/docs-<subject>`, done when pushed. `steer` gains "land it": continue a
  finished worker to commit and push its result.
- New `approvals` (a held call, `needs_approval <id>`, reading what it would do,
  asking, `approval_grant`; it runs once as recorded). `steer`'s line corrected;
  a `needs approval` row in `wait`; `briefing` lists held approvals.
- New `groups` (the five tools; owner-only; `group_default` is the user's choice
  alone). New `files` (direct reads and `file_edit_and_commit`, and when a task
  is the better tool).
- `memory`: proposals; scoped to every client. `grok.md` points to it instead of
  repeating it. `journal` and `knowledge` fold into `memory`.
- `health` and `incident`: `member_status` for one member.

### W3 — Driving a task: attended and alone

**The rule.** A task is *attended* when its coordinator stays to wait, answer and
verify; *alone* when nobody will. The worker must know which, not only the
coordinator.

zend:

- `repo_task_start` takes `driven: attended | alone`. Omitted, it is `alone`
  for the Grok client and `attended` otherwise. It is sealed into the worker's
  system prompt as one section, as `role` is. An alone worker:
  - never ends on a question: it decides within its brief, the design documents
    and the repository's rules, and records each decision in its answer;
  - lands its own result — commits and pushes its branch — without asking;
  - runs its brief's verify step itself (the templates already carry one);
  - meets anything only the user may approve by stating it as pending, and
    stops there.
- When an alone task settles — done, failed, stuck, or lost to a restart — zend
  writes its outcome onto the owner's memory as an open item: what was done,
  where it landed, what waits on approval. `get_user_context` raises it in the
  next conversation, from any client.

Guides: new `drive` (the two modes; an alone brief names its landing, its
done-when, what it must never do, and has its handoff written at the start);
client pointers through `clients:`; one line each in `implement`, `wait`,
`steer`, `verify`, `handoff`, `overnight`, `feature`, `memory`,
`journal-restructure`, `briefing`, `thread`.

### W4 — Writing memory well

- **Episodes leave the tree.** The aggregator keeps a conversation's tool
  activity in that conversation's own journal folds, not as one entry file per
  call; memory holds notes — decided, found, landed, left open — and the
  outcomes W3 delivers. `mcp_user_context.md` §2.3 is revised with it.
- **One conversation per session.** A client session's calls open one
  conversation journal, however often its MCP transport reconnects.
- **A note carries its terms.** `journal_note` takes, beside `text`:
  - `source` — a doc section, file:line, commit or URL. A note the speaker said
    needs none: its stamp is its source. A fact *derived* from the code or a doc
    without one is refused on a topic journal.
  - `expires` — a date after which the line is hidden from reads and frames
    (kept in history).
  - `replaces` — the line it supersedes, which is then hidden the same way.
- **Speakers are honest**: a line Claude or Zen derived is stamped as theirs, not
  as the person's.
- `journals check` gains the structural checks W6 relies on (no line id lost
  between two trees, a shape budget), so any writer — a person, a task, a
  mission — is gated by the same code.

### W5 — Reading and search

- **The live path.** The journals a conversation is focused on — `here`, its
  `about` topics, the speaker's `people/` journal — join its prompt as sealed
  sections, reconciled each round from the in-memory tree, under one cap, by the
  mechanism lineage sections already use. A note is in every focused
  conversation within the sync, with no ingest.
- **The broad path** stays provenance, with the recency term built
  (`guidance_layer.md` §4) and fed notes, not episodes.
- **One resolver.** `threads/<thread>[/<worker>]` and `projects/<repo>` resolve
  as memory paths for read and grep; their stores stay where they are.
- **Grep** takes `kind`, `speaker`, `since`, `until` and `offset`, merges hits
  across scopes newest first, and pages instead of truncating. **Memory
  `journal_read`** pages its lines and reads an archive by its period.
- **A thread names the topics it is about**, as a conversation does; its notes
  and its workers' outcomes then appear under those topics (§7.3's export).
- Thread-note numbering fixed if W0.3 finds it unstable.

### W6 — Maintenance as a mission

A `journal-maintenance` and a `guide-maintenance` template in the journals
repository's mission library (`mission_templates.md` §5), run on events — a merge
that changes a doc or file a note cites; a change to a tool a guide lists — and
nightly for expiry, misfiling and summaries.

Stages, maker never the checker:

1. **Survey**, read only: candidates with their evidence.
2. **Operations**, not prose: move, supersede-with-source, expire, merge. No
   free rewrite of a journal.
3. **Verify** by a second worker against the source on main.
4. **Checks** by code: `journals check` (W4's structural checks included); for
   guides, every listed tool served and every named guide present.
5. **Gate**: within budget it lands; a split, merge, rename or a change past the
   shape budget is a `critical` step for the Creator or Yen, failing safe; a
   correction about a person is a proposal for that person.
6. **One commit**, reported as a W3 outcome.

It runs once by hand first, as a coder task on the journals checkout gated by the
same checks: re-file `people/john`, supersede the stale `workers-threads` and
`ingest` notes, expire the reminders, remove the stored instruction, rebuild the
cut summaries. The template follows when `mission_templates.md` §7 (a drone
driving a zend task) is built.

### Documents

Revised with the workstream they govern: `memory_backbone.md` (master-rooted
status; §6.5 against W5's resolver; §7.3 against W5's export),
`guidance_layer.md` (§4 recency; the missing §2.2 reference),
`mcp_user_context.md` (`~` paths; §2.3 against W4), `zen_tool_reach.md:77`
(Zen writes memory and cannot read it through its own tools),
`claude_zend_gateway.md:633` (workers hold `journal_note`).

---

## 3. Order

| Step | What | Depends on |
|---|---|---|
| 1 | W0 | — |
| 2 | W1, if W0 finds the gap | W0 |
| 3 | W2 | W0.1 |
| 4 | W3 | W2 |
| 5 | W4, then the one-time cleanup pass | — |
| 6 | W5 | W4 (notes, not episodes, feed it) |
| 7 | W6's template | W4's checks; `mission_templates.md` §7 |

W2, W4 and the ingest work in progress touch different code and run side by side.
W3's guide text and W2 are the same files and land one after the other.

---

## 4. Review

Read back for simplicity, mistakes and risks. What changed in this document
because of it is folded in above; what remains is listed.

### 4.1 Simplicity — what was cut

- **No verify task chained by zend after an alone task.** The templates already
  carry a verify step; a second mechanism doing the same job is a dual path.
- **No automatic restart of a lost alone task.** A restart re-runs pushes and
  decisions it cannot see were made; a lost task is reported like any other
  outcome, and the user restarts it.
- **No store migration for `T:`/`P:`.** One resolver gives one address space;
  moving the records buys nothing a reader sees.
- **The live path reuses lineage sections** rather than adding a section kind.
- **Maintenance checks live in `journals check`,** not in the mission: the same
  gate binds a task, a mission and a hand edit, and the cleanup pass needs no
  mission machinery to be safe.
- **Search filters cut to five** (`kind`, `speaker`, `since`, `until`,
  `offset`); `channel` and following references wait until a real question
  needs them.
- **42 items became seven workstreams**, each one change.

### 4.2 Mistakes found and corrected

- **Alone restructures were to push without a diff review.** That removed the one
  check the journals repository's direct-to-main pushes had. An alone restructure
  goes through W6's gate instead: code checks, a shape budget, `critical` past it.
- **"Refuse an uncited note" would have refused the car.** A person's own word
  needs no citation; only a fact derived from code or docs does.
- **Removing episodes breaks what reads them.** `get_user_context`'s "where the
  last conversation left off" and the conversation folds are built from episodes;
  W4 keeps the activity in the conversation's folds rather than deleting it.
- **The live connector lag** (today's connector serves `cluster_logs`, not
  `log_*`) was listed as work; it is a rollout, not a change, and is left out.
- **Three claims were stated as fact that are inferred** (the privacy gap, note
  renumbering, episodes in the ingest backlog). They are now W0's questions.

### 4.3 Risks that remain

- **A bad note now spreads at once.** W5's live path puts a focused journal into
  every conversation within seconds; a wrong or hostile note reaches all of them
  before anyone reads it. The injection guard is the only line of defence on the
  write; W4's `source` and W6's verification narrow it but do not close it.
- **Live sections cost prefill.** Each note reseals the prompt from the first
  changed section. `here` changes most and goes last; the cap holds the cost,
  and the reseal rate must be measured before the cap is set.
- **Alone workers decide.** An alone worker pushes branches it chose to push and
  records decisions nobody checked in the moment. The bound is the branch rule
  (never the default branch) and the pending-approval stop; the journals
  repository, which pushes to main by the user's rule, is gated by W6's checks
  for exactly this reason.
- **W1 may hide memory that Zen's own work relies on.** A filter keyed to the
  owner's groups is right for a person's conversation; ingest and calibration
  conversations have no owner and must be decided explicitly, not by default.
- **The cleanup pass races live writes.** The journals service syncs every 5 s
  and replays a refused push; the pass keeps every line id (`git mv`) and is run
  as one commit so a conflict replays rather than loses.
- **W6 waits on unbuilt plumbing** (`mission_templates.md` §7). Until it lands,
  maintenance is the hand-run pass; the plan does not depend on §7 for anything
  else.
- **The cluster is not stable yet.** The re-ingest backlog, WhiteWorld's NaN
  fault and the KV back-pressure work are open; every guide change is a guide
  re-read on each member. W2 and W3 are batched to one ingest each for that
  reason.
