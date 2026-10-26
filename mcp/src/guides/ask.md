---
name: ask
description: How to get a codebase question answered by a worker — every claim pinned to the file and line it was read at.
tools: [task_start, task_wait, journal_grep, timeline_read]
---
# Asking about a codebase

**Use when** the answer takes reading more than a few files: how something
works, where a behaviour comes from, what calls what, whether a change is
safe. Split a broad question into independent ones and start them together.

## Prompt

`task_start {thread, name: "ask-<topic>", effort: "deep", prompt}`:

```
GOAL: <the question, in full, and what the answer is for>

CONTEXT: <what you already know; files already read; what not to re-explain>

CONSTRAINTS:
- Repository <name>; read only — change nothing.
- Scope: <paths or modules, if known>.

METHOD:
1. Find where it lives: search, then read the files themselves.
2. Follow the code paths the answer depends on to their ends — callers,
   callees, configuration.
3. Check yourself: does every claim in your draft have a file:line you read?

VERIFY BEFORE ANSWERING:
- Re-read each file:line you cite and quote the lines.
- Anything you did not read in this task is UNVERIFIED — what you know from
  your context is not a reading.

FINAL RESPONSE, in this shape:
ANSWER: the answer, in a few sentences
DETAIL: each supporting point, with file:line and the quoted lines
UNVERIFIED: what you could not confirm
COULD NOT DO: anything asked that you did not do, and why
```

## Read the result

Check the pointers you will rely on: `journal_grep` the worker's journal for
the file, `timeline_read {turn}` for the read. A claim without a file:line is
not an answer.
