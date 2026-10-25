---
name: zend-implement
description: Delegate implementing a written specification to zend — on a branch of its own, with tests written alongside, built and run on an agent, and the branch pushed — then review the result before anything is merged. Use when a change is specified clearly enough to hand over, and the repository's work can be done by zend's tools. Follows zend-thread.
---

# Implementing a spec on zend

## When

- The change is written down: a design section, an issue, or a spec you
  write now. zend implements against a spec; it does not decide the design.
- The work is volume — edits across files, tests, a build — not a judgement
  call you still need to make.

Not for: deciding what to build, anything you cannot specify in a page, or a
change the user wants made by hand.

## Thread

The feature's thread (zend-thread). Note the spec's decisions there first —
every worker starts with them.

## Delegate

`zend_task {thread, name: "implement-<what>", prompt}` with this prompt,
filled in:

```
Implement this specification on a new branch `zen/<what>` of <repository>:

<the spec, in full — what to change, where, the behaviour, the edge cases>

Work rules:
- Follow the repository's conventions (its CLAUDE.md; the project journal holds the standing rules).
- Write the tests alongside the code, at the level the behaviour lives.
- Build and run the tests with your agent tools; read the outcome from the output, never assume it.
- Commit on the branch with clear messages, and push the branch. Never push to the default branch.
- Before you answer, start a part named verify-implementation with sub_run that checks, against the
  files themselves, that every part of the spec is implemented and every test you cite passed.
- If the spec is ambiguous or two designs trade off, ask (QUESTION block) rather than guess.

Final response, in this shape:
VERDICT: implemented | partly implemented | blocked
BRANCH: <name> at <commit>
CHANGES: each file and what changed
TESTS: what ran, and the result line from the output
OPEN: anything not done, with why
POINTERS: file:line for each piece of the spec
```

Use `priority: high` only if the user is waiting on it now.

## Wait and answer

`zend_wait` until it settles. Answer questions with the spec in hand; if a
question exposes a gap in the spec, decide it, note the decision on the thread,
and answer.

## Review

Before reporting success:

1. Read the entry; check `TESTS` against the turns it cites (`journal_read`).
2. Read the diff of the branch yourself, or delegate a review with zend-review.
3. Check the verifier part's entry: it is cited in the worker's.

## Approval

Merging the branch to main is the user's decision — ask. Do not merge or
commit on main yourself unless the user says so.

## If zend is down

`cluster_status` says zend cannot take work, or the task stalls past an hour
with no progress: implement it yourself, the ordinary way.
