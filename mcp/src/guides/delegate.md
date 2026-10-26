---
name: delegate
description: How to write a task for a worker so it does the work properly — the prompt template every other guide builds on, and how to choose effort and priority.
tools: [task_start, task_wait]
---
# Delegating a task

A **worker** is a model conversation running on hardware the user owns. It reads the repository's analysis, runs its own tools, and writes a
journal. It does exactly what its prompt makes clear and no more: a vague
prompt gets a plausible answer, not a checked one. Write every task in this
shape.

## The prompt

```
GOAL: <what to find out or do, in full, and what the answer is for>

CONTEXT: <what is already known or decided; what earlier tasks on this thread
found; what NOT to redo>

CONSTRAINTS:
- <scope: which repository, paths, date window, sources>
- <rules: e.g. "push to a branch, never main"; "no network"; "read only">

METHOD:
1. <the phases, in order — gather, verify, check yourself>

VERIFY BEFORE ANSWERING:
- <what must be checked, and how: run the tests, open each cited URL,
  re-read each file:line cited>

DONE WHEN: <the condition that ends the task>

FINAL RESPONSE, in this shape:
<SECTION>: ...
<SECTION>: ...
UNVERIFIED: what you could not confirm, and what you could not read
COULD NOT DO: anything asked that you did not do, and why
```

## Rules every prompt carries

These come from runs that went wrong. Include the ones that apply.

- **Cite what you read.** A file, symbol or setting named in the answer cites
  the file:line it was read at; a web claim cites the URL it came from. What
  the worker "knows" from its context is not a reading — it is UNVERIFIED.
- **Open what you cite.** A URL is cited only if it was fetched and read. A
  search snippet is not the page; a figure seen only in a snippet is
  UNVERIFIED.
- **Mark the second-hand.** A figure seen only on an aggregator or a roundup
  is marked as such, not reported as fact. A vendor's claim is "the vendor
  reports", kept apart from independent results.
- **Unread is not absent.** Not finding something in a page's text is not
  evidence it does not exist — the page may have shown it as an image (the
  fetch reports `images`). Say what could not be read.
- **Check yourself.** Before answering, read the draft against itself: two
  claims that contradict, a figure stated two ways, a "none" beside a number
  taken from that same thing.
- **Report the gaps.** What was not done, and why, is part of the answer —
  never silently dropped.

## Effort and priority

- `effort: exhaustive` — work that weighs many results against each other:
  research, review, root-causing. Thinking is where a contradiction gets
  noticed.
- `effort: deep` (the default) — most work.
- `effort: balanced` — mechanical work: a rename, a format fix, a known
  procedure.
- `priority: high` only when someone is waiting on it interactively; it only
  changes how soon the worker's turns decode, not how hard it thinks.

## Splitting

Split broad work into independent tasks, each with its own prompt, name and
final response. Start them all, then `task_wait` on all of them together.
Name each worker for what it does (`verify-expiry`), never a number.
