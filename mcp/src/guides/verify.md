---
name: verify
description: How to check a finished worker's answer before relying on it — reading its citation audit, continuing the same worker to fix what it skipped, and spot-checking against its turns.
tools: [repo_task_wait, repo_task_reply, journal_read, journal_grep, timeline_read]
---
# Verifying a worker's answer

A worker's answer is evidence, not truth. A research run once cited "primary
sources" it never opened and took its key figures from one aggregator page;
its list of what happened was right, its details were not. Check before you
rely on it, and check in proportion to what rides on it.

## 1. Read the citation audit

A finished task's answer that cites any URL ends with an audit of each one,
against what the tools returned to the worker and to the parts it started:

- **opened** — a `web_fetch` of it returned the page. A fetch that failed
  opened nothing.
- **seen** — it appeared only inside a tool's result: a search hit, a link on
  a page, an error message. The worker had the snippet, not the page.
- **never seen** — no tool result held it. Inferred, or invented.

The audit you read already comes after one automatic round: a worker whose
answer cited anything not opened was sent back, once, to fetch those pages
and correct itself, and the answer you have is its corrected one. So what is
still `seen` or `never seen` is what it could not or would not fix — treat
those claims as unverified. When the audit says the check round failed or did
not finish, the answer is the one from before the check.

The audit proves a page was opened, not that the page says what the answer
claims. That is step 3.

No audit at all means the answer cites no URL. For research, that is itself
a finding: ask for its sources.

## 2. Continue the same worker

A finished worker can be continued with `repo_task_reply {worker, answer}` until
its thread is archived or it sits a day untouched — what `repo_task_wait` answers
for it says so. It keeps its whole context, so it fixes its own work faster
than a new worker can redo it:

```
These claims rest on pages you did not open: <the audit's seen and never-seen
URLs, with the claim each supports>.
Fetch each one and check the claim. Correct your answer where the page says
otherwise; remove a citation whose page does not exist; mark UNVERIFIED
anything you still cannot confirm.
Also: <specific doubts — a figure that looks off, two claims that disagree,
a source you distrust>.
Answer in the same shape as before, with a CHANGES section saying what you
corrected and why.
```

Its new answer carries a new audit. Past the day, or once its thread is
archived, the worker cannot be continued: start a new task that cites its
entries (`T:<thread>/<name>#<n>`) and says what to check.

## 3. Spot-check what you will act on

- `journal_grep {journal: "T:<thread>/<name>", pattern}` for a URL, a file, a
  figure — each hit names the turn it is in. `T:<thread>/*` searches every
  worker on the thread, parts included.
- `journal_read {journal: "T:<thread>/<name>"}` lists the worker's entries —
  its claims. With an entry's address (`…#<n>`) it pages the raw turns behind
  that entry, which run to hundreds of lines: find the turn with
  `journal_grep` first, then page to it with `offset`.
- `timeline_read {id: "T:<thread>/<name>", turn}` (Admins only) reads one of
  the worker's turns whole: the tool results in its user half — the fetched
  page as the worker got it — and its thinking and tool calls in the assistant
  half, a page of characters at a time. Without `turn` it lists the turns, and
  its head shows the effort the worker ran at. Its turn numbers count whole
  turns; `journal_read`'s count each half, so the two differ.
- A journal page is counted in lines, and one line of a tool result can run
  to thousands of characters: page an entry's turns with a small `limit`
  (20–50), or read the one turn with `timeline_read`.

## A code answer

An answer about a repository has no audit — its evidence is file:line, not
URLs. Check it this way:

- Every claim you will rely on cites a file:line with the line quoted; one
  that does not is unverified. `ask`'s template demands this.
- `journal_grep` the worker's journal for each cited file or symbol: the hit
  shows the turn where it read it, and whether the quoted line is what the
  file said.
- Continue the worker with what is missing — "quote the line for each of
  these", "you cited X without reading it; read it" — as in step 2.
- If you have the repository yourself (Claude Code in it), open the cited
  lines: the fastest check there is.

## 4. Say what was checked

Report to the user what you verified, what you corrected, and what remains
unverified. Keep "the worker reports" apart from what you confirmed. Never
pass a worker's claim on as fact when you did not check it.
