# MCP guide catalogue: the skills to write

**Status: written** — every guide below is in `mcp/src/guides/<name>.md`; the
library that serves them (`docs/mcp_guides.md` §2) is not built yet.

The guides to build for the connector's guide library (`docs/mcp_guides.md`).
Each entry is one guide: its name, when to use it (its `description`), the
tools it lists (its `tools`, which get a "Guides" line), and the method in
outline. Written 2026-10-26.

Tools are named as the connector serves them. Two are planned, not built:
**`repo_task_reply` on a finished worker** (`docs/mcp_guides.md` §3) and the
**citation audit** with its correction round (§4); guides that depend on them
say so.

Conventions every guide follows:

- Start from `thread` (A1): attach, note decisions, re-read after compaction.
- Delegate the reading and the volume; keep design, decisions and approvals.
- A worker's result is evidence, not truth: check claims against the turns
  they cite before acting.
- Anything the user must approve — a merge, a deploy, a Decide command, a
  stop — is asked first, and `confirm: "approved by the user"` is passed only
  with that approval.

---

## A. Foundations — how to work with Zen at all

### A1. `thread`
**Use when:** starting any work through the connector, and after context is
compacted.
**Tools:** `thread_attach`, `thread_list`, `thread_archive`, `journal_note`,
`journal_read`
- Call `guide_list` once; attach to the feature's thread (a kebab-case slug for
  what the work is about); read its journal.
- Note every decision and every rule the user states, as it is made.
- After compaction: re-attach, `journal_read {journal: "self"}`, carry on from
  the notes rather than from memory.
- Archive the thread when the feature is done.

### A2. `delegate`
**Use when:** writing any `repo_task_start` prompt.
**Tools:** `repo_task_start`, `repo_task_wait`
- State the task in full: what to do, what to verify before answering, the
  shape of the final response (verdict, findings, file:line pointers).
- Name the worker for what it does; never a number.
- Choose `effort`: `exhaustive` when results must be weighed against each other
  (research, review), `balanced` for mechanical work, `deep` otherwise.
  `priority: high` only when someone waits on it interactively.
- Split broad work into independent tasks, start them all, then wait on all.

### A3. `wait`
**Use when:** tasks are running.
**Tools:** `repo_task_wait`, `thread_list`, `repo_task_nudge`, `repo_task_stop`
- Wait on every running worker in one call; handle whichever settles first,
  then wait again on the rest.
- Read the progress lines and the new journal entries each answer brings; stop
  waiting when the entries already answer the question.
- Nudge a worker heading the wrong way; stop one repeating itself.
- `thread_list` to find what waits on you when no call was open.

### A4. `verify`
**Use when:** a worker has finished and its answer will be relied on.
**Tools:** `repo_task_wait`, `repo_task_reply`, `journal_read`, `journal_grep`,
`timeline_read`
- Read the citation audit (planned): anything cited but not opened is suspect.
- Follow up with the same worker (planned `repo_task_reply` on a done worker): open
  what you skipped, correct the answer, mark what stays unverified.
- Spot-check the claims you will act on: `journal_grep` for the file or URL,
  `timeline_read {turn}` for the call and its result.
- Report what was checked and what was not.

### A5. `steer`
**Use when:** a running worker needs redirecting.
**Tools:** `repo_task_nudge`, `repo_task_stop`, `repo_task_reply`
- Nudge for course corrections that should not end its work.
- Stop for a worker looping or on a path you no longer want; then `repo_task_reply`
  with the new direction.
- Reply to a question with the decision itself, never "it depends".

### A6. `journal`
**Use when:** recording or finding what a thread or project knows.
**Tools:** `journal_note`, `journal_read`, `journal_grep`
- Thread notes: decisions and why, user statements, constraints.
- Project notes (`P:<repository>`): standing rules every task should start
  with — conventions from CLAUDE.md, things the user says always hold.
- Find evidence with `journal_grep` across a thread's workers (`T:<thread>/*`).

### A7. `handoff`
**Use when:** resuming work another session started, or after a long break.
**Tools:** `thread_list`, `thread_attach`, `journal_read`, `repo_task_wait`,
`cluster_status`
- List threads; attach to the one in question; read its journal and any
  waiting questions; check running workers; summarise where things stand
  before acting.

---

## B. Code

### B1. `ask`
**Use when:** a question about a codebase whose answer needs reading many
files.
**Tools:** `repo_task_start`, `repo_task_wait`, `journal_grep`
- One worker per independent question; answer cites file:line for every claim.
- Spot-check the pointers before relying on them.

### B2. `implement`
**Use when:** building something against a written spec.
**Tools:** `repo_task_start`, `repo_task_wait`, `repo_task_reply`, `journal_read`,
`repo_job_start`, `repo_job_log`
- The spec is written first (by Claude, approved by the user) and passed whole.
- The worker implements on a branch, tests as it goes, reports what changed
  and what it could not do.
- Answer its questions with the spec in hand; review the diff yourself; run the
  tests on an agent (B6).

### B3. `review`
**Use when:** reviewing a change, a branch or a PR.
**Tools:** `repo_task_start`, `repo_task_wait`, `journal_grep`, `timeline_read`
- One worker per area for a large change, `effort: exhaustive`.
- Findings with file:line, a concrete failure scenario each, ranked.
- Verify each finding against the code before reporting it.

### B4. `security-review`
**Use when:** a change touches auth, input handling, secrets, network or
sandboxing.
**Tools:** `repo_task_start`, `repo_task_wait`, `journal_grep`
- Threat-model first (what an attacker controls), then trace each input to its
  sinks; findings need an exploit path, not a pattern match.

### B5. `debug`
**Use when:** a bug, a failing test, or a wrong answer needs its root cause.
**Tools:** `repo_task_start`, `repo_task_wait`, `repo_task_reply`, `cluster_logs`,
`repo_job_start`, `repo_job_log`
- Reproduce first; then narrow — test, narrow, test — never patch a symptom.
- Each hypothesis is checked against evidence before the next.
- The fix comes with a test that failed before it.

### B6. `test`
**Use when:** a branch needs building and testing.
**Tools:** `repo_job_start`, `repo_job_log`
- Build and test on a build agent, never on a member serving traffic.
- Read the last page for the summary; `grep: "FAILED"` for failures.
- Report pass or fail, the failing tests and their output.

### B7. `test-write`
**Use when:** code needs tests, or a bug needs a regression test.
**Tools:** `repo_task_start`, `repo_task_wait`, `repo_job_start`, `repo_job_log`
- Tests assert exact expected values; one behaviour each; named for what they
  prove. Run them before calling them done.

### B8. `refactor`
**Use when:** restructuring code without changing behaviour.
**Tools:** `repo_task_start`, `repo_task_wait`, `repo_job_start`, `repo_job_log`
- Tests green before and after; the plan names every call site; no behaviour
  change hidden in the move.

### B9. `upgrade`
**Use when:** bumping a dependency or toolchain.
**Tools:** `repo_task_start` (with research), `repo_job_start`, `repo_job_log`
- Research the changelog first (C1); list breaking changes and where they bite;
  upgrade on a branch; build and test on an agent.

### B10. `perf`
**Use when:** something is slower than it should be.
**Tools:** `repo_task_start`, `repo_job_start`, `repo_job_log`, `cluster_status`
- Measure before changing; compare like with like; one change per measurement;
  report numbers with the machine they came from.

### B11. `docs-write`
**Use when:** a design document or a README needs writing from the code.
**Tools:** `repo_task_start`, `repo_task_wait`, `journal_grep`
- Every statement about the code cites where it is; the document describes
  what is, marking anything planned as planned.

### B12. `branch`
**Use when:** a worker's change needs to land as a branch for review.
**Tools:** `repo_task_start`, `repo_task_wait`, `repo_job_start`
- Workers push branches only, never a default branch; Claude reviews; the user
  approves the merge.

---

## C. Research

### C1. `research`
**Use when:** a question answered from the web, every claim checked.
**Tools:** `repo_task_start`, `repo_task_wait`, `repo_task_reply`, `journal_grep`,
`timeline_read`
- Gather → verify (fetch primary sources; independent evidence both ways;
  conflicts settled by source quality; aggregator-only figures marked; images
  are unread, not absent) → check itself → answer in ANSWER / DETAIL /
  CONFLICTS / UNVERIFIED / SOURCES.
- `effort: exhaustive`; read the audit; follow up on anything cited but not
  opened.

### C2. `compare`
**Use when:** choosing between libraries, services or approaches.
**Tools:** as C1
- Fixed criteria applied to every option; a table; a recommendation with the
  reason; what would change it.

### C3. `release-watch`
**Use when:** a periodic survey — model releases, a dependency's changes, a
vendor's announcements — over a date window.
**Tools:** as C1
- Every date checked against the primary source; vendor claims kept apart from
  independent results; what moved since the last survey on the thread.

### C4. `fact-check`
**Use when:** a set of claims — someone's, or a worker's — needs checking.
**Tools:** as C1
- Each claim: confirmed, contradicted, or unverified, with the source that
  decides it.

### C5. `literature`
**Use when:** surveying papers on a technique.
**Tools:** as C1
- Read the papers, not the abstracts; methods and numbers side by side; what
  was measured how, and where results conflict.

---

## D. Operations — the cluster, builds, releases

### D1. `health`
**Use when:** checking the cluster, before and after any operation, or when
something stalls.
**Tools:** `cluster_status`, `service_status`, `cluster_logs`
- Every member's build and state; each app's floor; anything in fault;
  each machine's services and the last trial.

### D2. `logs`
**Use when:** a symptom needs explaining from the logs.
**Tools:** `cluster_logs`, `cluster_status`
- Narrow by `since`, `level`, `tenant`; quote the lines relied on, with their
  member.

### D2b. `watch`
**Use when:** keeping an eye on a rollout, a member or the logs for a while.
**Tools:** `repo_task_start`, `repo_task_wait`, `cluster_status`, `cluster_logs`
- Delegated to a worker that polls and reports only exceptions and the end
  state, so the watch costs one answer, not a poll loop.

### D3. `incident`
**Use when:** something is broken now.
**Tools:** `cluster_status`, `cluster_logs`, `cluster_audit`, `service_status`,
`service_control`, `cluster_command`
- Contain first (what is affected, since when, what changed —
  `cluster_audit`); then diagnose; a restart is a step, not a fix; every
  Decide action approved; write up what happened (D11).

### D4. `cluster-care`
**Use when:** a member needs maintenance.
**Tools:** `cluster_command`, `cluster_job_status`, `cluster_status`
- Look and Care commands as asked; Decide commands only with approval; follow
  each job to its end; confirm the member came back healthy.

### D5. `deploy`
**Use when:** putting a commit on origin/main onto one machine.
**Tools:** `repo_job_start`, `repo_job_log`, `service_trial`, `service_status`,
`cluster_status`
- Build and stage on that machine's agent; trial it through keep; watch to
  proven or rolled back; confirm the member serves the new build.

### D6. `release`
**Use when:** taking a commit through tests, a trial, certification and
rollout.
**Tools:** `repo_job_start`, `repo_job_log`, `service_trial`, `service_status`,
`cluster_command`, `cluster_job_status`, `gateway_command`, `cluster_status`
- Tests on an agent → a trial on one machine → certification watched → the
  rollout's plan shown, approved, started, watched member by member.
- Raise or cancel an app's floor (`gateway_command`) only with approval.

### D7. `rollback`
**Use when:** a released build misbehaves.
**Tools:** `cluster_command`, `cluster_job_status`, `cluster_status`
- Name the build to return to; show the plan; approval; watch every member
  onto it.

### D8. `ab-trial`
**Use when:** comparing a new build against the running one on one machine.
**Tools:** `service_trial`, `service_status`, `cluster_status`, `cluster_logs`
- Trial on one machine while the others serve the old build; compare what
  matters (errors, latency, answers) before rolling out.

### D9. `gateway`
**Use when:** an upgrade's grace period or floor needs changing.
**Tools:** `gateway_command`, `cluster_status`
- Explain the floor and grace period; raise or cancel only with approval.

### D10. `substrate`
**Use when:** a wrong answer may come from stored analysis — a bad layer or a
corrupt conversation.
**Tools:** `layer_read`, `layer_toggle`, `timeline_read`, `timeline_delete`
- Toggle the suspect layer off and ask again; read the timelines that layer
  holds; delete a corrupt one only with approval — the next analysis pass
  rebuilds it.

### D11. `postmortem`
**Use when:** after an incident, a rollback or a bad run.
**Tools:** `cluster_audit`, `cluster_logs`, `journal_read`, `journal_note`
- Timeline from the audit and logs; cause; what would have caught it; the
  lasting fix noted on the project journal.

---

## E. Bots — npcd's characters

### E1. `bots`
**Use when:** working with bots at all.
**Tools:** `bot_list`, `bot_mission_read`, `bot_message_read`
- What a bot is (an autonomous agent living in a world); listing them;
  reading what each is doing before asking anything of it.

### E2. `bot-mission`
**Use when:** giving a bot a mission and seeing it through.
**Tools:** `bot_mission_give`, `bot_mission_read`, `bot_mission_step`,
`bot_mission_cancel`
- A brief that says what is asked and what done looks like; steps in order,
  addressed to the bot; follow it with `bot_mission_read`; sign off or add
  steps as it works; cancel only what is no longer wanted.

### E3. `bot-documents`
**Use when:** having bots write or edit documents — research write-ups,
reports, world material.
**Tools:** `bot_mission_give`, `bot_mission_read`, `bot_message_send`,
`bot_message_read`, plus `repo_task_start` to review the result
- The brief names the document, its purpose, its sources and its shape; the
  bot works the document in parts; its report says what it wrote; the result
  is reviewed (by Claude or a zend worker) before it is relied on.

### E4. `bot-campaign`
**Use when:** several bots work one goal together.
**Tools:** `bot_list`, `bot_mission_give`, `bot_mission_read`,
`bot_channel_post`, `bot_channel_read`
- Split the goal; one mission per bot; a channel post to frame it (Creator
  only); track every mission; collect the reports.

### E5. `bot-message`
**Use when:** talking to a bot directly.
**Tools:** `bot_message_send`, `bot_message_read`
- It answers in its own time; read the thread again rather than expecting an
  immediate reply.

### E6. `bot-watch`
**Use when:** following what bots are doing over time.
**Tools:** `bot_list`, `bot_mission_read`, `bot_channel_read`,
`bot_message_read`
- States and moods across the roster; missions open and finished; what the
  world's channel carried.

### E7. `bot-debrief`
**Use when:** a mission ended and its outcome matters.
**Tools:** `bot_mission_read`, `bot_message_send`, `bot_message_read`
- Read the report (pass or fail, and its account) and the answer; ask the bot
  what is unclear; record what came of it on the thread.

### E8. `image`
**Use when:** drawing a picture with `image_generate`.
**Tools:** `image_generate`
- The model is Z-Image-Turbo, fine-tuned on prompt-enhancer captions, so
  Claude writes the enhancer's prompt itself (Tongyi's published rules: keep
  every asked element, make an idea concrete, add composition, light,
  material, colour and one style, put on-image text verbatim in double quotes,
  objective language, no quality tags). No negative prompt or weighting; the
  seed barely varies the picture, so variety comes from the prompt. People
  are stated as adults for the prompt check. 1024 on the long side for
  detail. Look at the result and refine the prompt.

---

## F. Combined workflows

### F1. `feature`
**Use when:** a feature from idea to release.
**Tools:** as A1–A4, B2, B3, B6, D5–D6
- Spec (approved) → implement → test on an agent → review → merge (approved)
  → deploy → release (approved), on one thread throughout.

### F2. `overnight`
**Use when:** the user leaves work running unattended.
**Tools:** as A1–A4, B2, B3, B6, D1
- Decisions made against the design documents and noted with reasons; nothing
  committed or deployed without prior approval; a report for the morning.

### F3. `briefing`
**Use when:** "where do things stand?"
**Tools:** `thread_list`, `journal_read`, `cluster_status`, `bot_list`,
`bot_mission_read`
- Open threads and what waits on the user; running and finished work; cluster
  health; bots' missions — in one summary.

### F4. `knowledge`
**Use when:** a lesson should outlive the conversation.
**Tools:** `journal_note`, `journal_read`
- Project rules on `P:<repository>`; decisions on the thread; what was learned
  from a bad run, so the next task starts with it.

### F5. `research-to-bots`
**Use when:** research should become world material bots work with.
**Tools:** as C1, E3
- Research verified first (C1), then a bot mission to turn it into the
  document, then a review of what the bot wrote.

---

## Order to write them

1. A1 `thread`, A2 `delegate`, A3 `wait`, A4 `verify` — every other guide
   builds on them.
2. C1 `research`, B1 `ask`, B2 `implement`, B3 `review`, B6 `test` — the ones
   in daily use, moved from `skills/`.
3. D1, D4–D7 — the operations skills that exist today, moved.
4. E1–E3 — bots, new.
5. The rest, as the work calls for them.
