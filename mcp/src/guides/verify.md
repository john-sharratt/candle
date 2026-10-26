---
name: verify
description: How to check a finished worker's answer before relying on it — the citation audit, a follow-up to the same worker, and spot-checks against its turns.
tools: [task_wait, task_reply, journal_read, journal_grep, timeline_read]
---
# Verifying a worker's answer

A worker's answer is evidence, not truth. A research run once cited "primary
sources" it never opened and took its key figures from one aggregator page;
its list of what happened was right, its details were not. Check before you
rely on it, and check in proportion to what rides on it.

## 1. Read the citation audit

A finished task's answer ends with an audit of the URLs it cites:
**opened** (fetched and read), **seen** (only in a search result — the
snippet, not the page) or **never seen** (cited, but nothing in the run
touched it). Anything not opened is suspect; anything never seen may be
invented.

## 2. Follow up with the same worker

`task_reply {worker, answer}` to the finished worker — it keeps its whole
context, so it can fix its own work faster than a new worker can redo it:

```
Your answer cites these without having opened them: <list>.
Fetch each one and check the claim it supports. Correct your answer where the
page says otherwise; mark UNVERIFIED anything you still cannot confirm.
Also: <specific doubts — a figure that looks off, two claims that disagree>.
Answer in the same shape as before, with a CHANGES section saying what you
corrected and why.
```

## 3. Spot-check what you will act on

- `journal_grep {journal: "T:<thread>/<name>", pattern}` for a URL, a file, a
  figure — each hit names the turn it is in.
- `timeline_read {id, turn}` reads that turn whole: the search query, the
  fetched text, the tool's result, and the worker's reasoning. The timeline's
  metadata shows the effort the worker ran at.
- `journal_read` with an entry's address shows its claims and the turns they
  cite.

## 4. Say what was checked

Report to the user what you verified, what you corrected, and what remains
unverified. Never pass a worker's claim on as fact when you did not check it.
