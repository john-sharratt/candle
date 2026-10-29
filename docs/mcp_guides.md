# MCP guides: a skill library the connector serves

**Status: built** (2026-10-26): the guide library in the aggregator
(`mcp/src/guides/`), continuing a finished worker and the 24-hour archive
(`zend/src/mcp/archiving.rs`), and the citation audit with its correction round
(`zend/src/mcp/audit.rs`). Written from the review of two Zen research runs and
a design discussion that followed. It extends
`docs/mcp_app.md` (the aggregator) and `docs/claude_zend_gateway.md` (zend's
tools). Where this document and those disagree about guides, this one is
authoritative; fix the others in the same change that builds it.

## 1. Why

A method — how to research, how to verify a worker's output, how to deploy —
is what turns zend's tools into good work. Today that method lives in the
repository's `skills/`, which reaches only a Claude Code session opened in
this repository. Every other place Claude meets the connector goes without:

| Channel | claude.ai | Claude Code | Reach |
|---|---|---|---|
| Repo `skills/` | no | only in this repo | one checkout |
| Skills uploaded to claude.ai | yes, after a manual ZIP upload | yes, synced to `~/.claude/skills/synced/` (v2.1.273+) | every machine signed in, but hand-maintained and drifting from the tools |
| Server `instructions` | **dropped** | delivered (truncated ~2 KB) | unreliable |
| MCP prompts | requested (`prompts/list`) but **not shown** in the UI (anthropics/claude-ai-mcp#333, open) | slash commands `/mcp__<server>__<prompt>` | Claude Code only |
| **Tool descriptions and tool results** | **delivered** | **delivered** | **every client** |

The one channel every client delivers is the tools. So the method is served
as tools.

Evidence that the method must reach the worker, not just Claude: a research run
delegated from claude.ai searched ~40 times and fetched 5 pages, cited
"primary sources" it never opened, and took its standout figures from one
aggregator page. The method in the repo skill would have prevented it; it never
reached the conversation.

## 2. The guide library

### 2.1 What a guide is

One Markdown file per guide, with YAML frontmatter, in the aggregator:
`mcp/src/guides/<name>.md`. The format is SKILL.md's, plus `tools`, so a guide
can also be zipped as a claude.ai skill if ever wanted.

```markdown
---
name: research
description: A question answered from the web, every claim checked against the page it came from.
tools: [repo_task_start, repo_task_wait, repo_task_reply, journal_grep, timeline_read]
---
# Research
...the method...
```

| Field | Meaning |
|---|---|
| `name` | What `guide_read` takes; equals the file's stem |
| `description` | One line, "use when…": shown in the catalogue and on each listed tool |
| `tools` | The tools this guide applies to — any backend's |
| `clients` | Optional: the clients (`mcp_server::Client::name`) this is the guide for, served first in their user context (`docs/mcp_user_context.md` §2.4); one guide per client |
| body | The method itself |

### 2.2 Compiled in

The folder is embedded with `include_dir!` (as zend embeds its prompts) and
parsed once at startup into `Guide { name, description, tools, body }`
(`mcp/src/guides/mod.rs`). A new `.md` dropped in the folder is picked up by the
next build with no list to edit.

- The guides ship in the same binary as the aggregator, so a guide and the
  version it describes are released together.
- Editing a guide is a commit, a rebuild of `mcp` and a restart — about a
  minute on AlienWorld, touching neither zend nor npcd.
- **Tests** parse every embedded guide: frontmatter present, `name` equals the
  file stem, names unique, a non-empty `description` and `tools`, a body.
  A broken guide fails `cargo test`, never a session.
- Whether a guide's `tools` exist is known only at runtime (the backends' tools
  arrive by `tools/list`): a guide naming a tool no backend lists is logged as a
  warning when the list is merged.

### 2.3 The tools

The aggregator's own tools — the only ones it serves itself, named by the rule
of `docs/mcp_app.md` (what it acts on, then what it does):

- **`guide_list`** — the catalogue: every guide's `name` and `description`.
  Its description says to call it once at the start of any conversation that
  uses this connector, before starting work, and carries the catalogue itself
  (generated), so Claude sees what is on offer even without the call.
- **`guide_read {name}`** — one guide's body. An unknown name answers the
  catalogue with the error.

### 2.4 The "Guides" line

When the aggregator merges the backends' `tools/list`, it appends one generated
line to the description of every tool some guide lists:

> **Guides:** `research` — a question answered from the web, every claim
> checked; `verify` — checking a finished worker's output. Read one with
> `guide_read` before you start.

Only names and one-liners; the method arrives on demand. A tool more than four
guides list (`NAMES_ONLY_PAST`) gets the names alone, so the line never
outgrows the tool's own description. The backends know
nothing of guides. A guide may span backends — a bot guide can list `bot_*`
and `task_*` tools alike.

### 2.5 What moves

`skills/` becomes the first guides: `zend-thread` (as the base guide every
other refers to), `zend-research`, `zend-implement`, `zend-review`,
`zend-test`, `zend-deploy`, `zend-release`, `zend-rollback`, `zend-watch`,
`cluster-care`, plus a new `bots` guide for npcd's tools. Each repo skill
shrinks to a pointer — "call `guide_read {name: research}`" — kept only for
Claude Code's automatic loading, or is removed.

### 2.6 Later: `guide_edit`

A tool for Claude to refine a guide when a run shows the method was wrong
(the research review would have been a lasting edit, not a one-off prompt).
With compiled guides this is a commit, not a tool; if guides ever move to
records, `guide_edit` writes one, journalled with who and why, applied at once
or held for the user's approval — the user's call.

## 3. Continuing a finished worker

`repo_task_reply` answers a worker that asked, is stuck, or was stopped — and one
that finished. Archiving a worker distils its conversation to text
(`ZendSession::archive_worker`), which is one way: an archived conversation
cannot be continued with its context. So a finished task is not archived when
it ends:

- A task that ends `done` is kept whole and marked `resumable`. `repo_task_reply`
  accepts it: the reply is its next turn with its whole context, and it answers
  as `repo_task_start` does. Its new journal entries are the reply's. What a `done`
  worker answers says it can be continued.
- A part, and a task that failed, are archived as they end, as before.
- A worker is archived — distilled, no longer resumable — when its thread is
  archived (`thread_archive`), or when it has sat **24 hours** untouched. A
  thread whose newest activity is 24 hours old, with nothing running on it, is
  archived the same way. The sweep runs hourly on every member, and
  only once the model is loaded (`zend/src/mcp/archiving.rs`). A worker is
  claimed under the lock `repo_task_reply` takes (which refuses a claimed worker),
  distilled outside it, and only then written as archived — never when the
  distil failed, so no record says archived of a conversation that was not.
  `repo_task_reply` on a worker whose thread was archived reopens the thread. Two members sweeping in the moment before
  one's write replicates can both archive a thread: distilling twice is
  harmless, and the journal gets the line twice. An attach counts as activity
  (`Thread::attached_ms`, marked at most hourly), so a thread Claude reopens is
  not swept back under it. Only these workers and threads are swept: a
  person's own zend conversations are never archived automatically.
- A worker waiting on Claude (a question, stuck) that sits 24 hours is archived
  as an error saying so, its question kept, as `thread_archive` does.
- This is the verification path: the same worker that did the research is asked
  to open what it skipped and correct itself, rather than a new worker starting
  cold.
- It is an MCP tool, so claude.ai and Claude Code get it alike. (Claude Code
  can also continue its own subagents; claude.ai has none, so zend workers are
  its only ones.)

## 4. Citation audit

Every finished task's answer carries an audit of the URLs its final response
cites, against what the worker actually did in its turns:

| Mark | Meaning |
|---|---|
| opened | a `web_fetch` of it returned the page (the URL asked for, or its `final_url`) |
| seen | appeared only in a tool's result — a search hit, a snippet — never fetched |
| never seen | cited, but nothing in the run touched it — inferred or invented |

A turn's tool results are its user half's `<tool_response>` blocks
(`fast_path::coverage::response_bodies`), so the audit is deterministic and
cheap. It reads every normal turn of the worker's conversation and of each
part it started (`ZendSession::conversation_detail`), and goes at the end of
what `repo_task_wait` / `repo_task_start` answer for a `done` worker. An answer citing no
URL carries no audit.

**One automatic correction round.** When anything cited is `seen` or
`never seen`, zend sends the worker back once by itself, as the next turn of the
same run: "your answer cites X, Y, Z without having opened them — fetch each and
correct your answer, remove what does not exist, mark UNVERIFIED what you still
cannot confirm, with a CHANGES section." The thread's journal says it was sent
back. The task answers after that round, with the audit of the corrected
answer. Every task gets it; there is no opt-out. A check round that fails,
is stopped, has its tools closed or asks a question leaves the task `done` with
the answer it gave before, its audit saying what became of the round — a
finished task never turns into a failed one for being checked.

Only what tools returned counts: the `<tool_response>` blocks of the turns.
A page is opened when a `web_fetch` returned it (a result with `final_url`; the
URL asked for and the one reached both count) or an `http_request` answered it
with a success status — a fetch that failed opened nothing. A URL named in a
prompt, Claude's or the correction's, is not evidence, and neither is a part's
`sub_run` answer: the part's own turns are audited with its worker's.

## 5. The research guide

The first guide written against §3 and §4:

1. **Gather** — search widely for the candidates.
2. **Verify** each item reported: fetch its primary source; search for
   independent evidence in both directions; settle conflicting figures by
   source quality (primary, then independent measurement, then reporting) and
   say why; a figure seen only on an aggregator is marked as such; text not
   found in a fetched page is not proof it was not published — a page reporting
   `images` may have shown it as a picture.
3. **Check itself** — the draft against itself before answering.
4. **Answer** in `ANSWER / DETAIL / CONFLICTS / UNVERIFIED / SOURCES`.
5. **Claude's side**: start with `effort: exhaustive`; read the audit; if it is
   not clean after the automatic round, `repo_task_reply` the worker to fix what
   remains; spot-check with `timeline_read {turn}` — the queries, fetched text
   and the dials the worker ran at.

## 6. Build order

1. Guides: the loader, `guide_list`, `guide_read`, the "Guides" line, the
   skills converted, tests, `docs/mcp_app.md` updated.
2. Continue a finished worker (§3).
3. Citation audit and the correction round (§4).
4. The research and verify guides rewritten against 2 and 3 (§5).

Releases: 1 is the aggregator alone (AlienWorld); 2 and 3 are zend (BlackWorld,
rolling to the cluster).

## 7. Decisions

- **Tools, not prompts or instructions**, carry the method: they are the one
  channel every client delivers (§1).
- **The aggregator owns the guides**: one library over every backend's tools,
  with the backends unaware of it.
- **Compiled in, not read at runtime**: one version with the code that serves
  it, checked by tests; a guide edit costs a rebuild of the aggregator only.
- **`guide_list` / `guide_read`**, two tools rather than one with an optional
  argument, by the naming rule.
- **Verification by the same worker** (§3), not a fresh one.
- **The audit is zend's, deterministic**; the correction round is one round,
  never a loop, and applies to every task — no opt-out, which would be a flag
  with nothing to choose it.
- **A finished worker is kept, not un-archived** (§3): archiving is a lossy
  distil, so continuing one means not archiving it until it is done with —
  its thread archived, or a day idle.
- **Only `/mcp` workers and threads archive themselves.** A person's own
  conversations are theirs to archive.
- **Tool names route; descriptions instruct.** claude.ai lists a connector's
  tools by name alone until one is loaded, and drops the server's
  instructions, so a name is the only thing Claude sees when it decides
  whether a request is Zen's. Work in the user's repositories is named
  `repo_`: `repo_task_start` / `_wait` / `_reply` / `_nudge` / `_stop` (a
  worker with a shell in the repository) and `repo_job_start` / `repo_job_log`
  (a program run on a branch on one machine). Claude has no other way to reach
  those, so the name is enough to send it here. Web research is Claude's own,
  with its own tools; Zen's server instructions no longer ask for it.
- **What Claude reads never says "zend".** The server is Zen; tool
  descriptions, errors, instructions and guides speak of workers, the model and
  hardware the user owns. The app's name appears only where it is a value Claude
  passes or reads back — `app` and `tenant` in the operations tools. The
  aggregator's backends are named `coding` and `bot`, the names its failures
  give.
