---
name: refactor
description: How to have a worker restructure code without changing what it does — tests green before and after, every call site moved, no behaviour hidden in the move.
tools: [task_start, task_wait, agent_job, agent_job_log]
---
# Refactoring

**Use when** code needs restructuring — split a file, rename, move, replace
an old path — and behaviour must not change.

## Prompt

`task_start {thread, name: "refactor-<what>", effort: "deep", prompt}`:

```
GOAL: <the restructuring, in full> in <repository>, on branch
`zen/refactor-<what>`. Behaviour must not change.

CONTEXT: <why; the target shape; constraints of the repository>

METHOD:
1. Run the tests that cover the area first; record the result.
2. List every call site and reference to what moves (search, then read).
3. Make the change; leave no compatibility shim or old path behind unless
   asked.
4. Run the same tests; the result must match.

CONSTRAINTS:
- No behaviour change hidden in the move: a fix found on the way is reported,
  not slipped in.
- Push the branch; never the default branch.

FINAL RESPONSE:
BEFORE: the tests run first, result quoted
AFTER: the same tests, result quoted
CHANGES: each file and what moved
CALL SITES: each one updated, file:line
NOTICED: bugs or oddities found on the way, not fixed
COULD NOT DO: anything not done, and why
```
