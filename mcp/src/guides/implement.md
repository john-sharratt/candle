---
name: implement
description: How to have a worker implement a written spec — on its own branch, tests alongside, built and run, pushed, verified by a part — then review it before anything merges.
tools: [repo_task_start, repo_task_wait, repo_task_reply, journal_read, repo_job_start, repo_job_log]
---
# Implementing a spec

**Use when** the change is written down — a design section, an issue, a spec
you write now — and the work is volume: edits across files, tests, a build.
A worker implements against a spec; it does not decide the design. If you
cannot specify it in a page, decide it first, with the user.

Note the spec's decisions on the thread first: every worker starts with them.

## Prompt

`repo_task_start {thread, name: "implement-<what>", effort: "deep", prompt}`:

```
GOAL: implement the specification below in <repository>, on the branch
`<the name the user gave, or zen/<what>>`.

SPEC:
<in full — what to change, where, the behaviour, the edge cases, what must not
change>

CONSTRAINTS:
- Follow the repository's conventions (its CLAUDE.md; the project journal
  holds the standing rules).
- Push the branch; never push to the default branch, and never force-push. If
  the branch already exists on origin, build on it.
- No stubs, no TODOs, no placeholder code: implement it fully or say what is
  missing.

METHOD:
1. Read the code the spec touches before changing anything.
2. Write the tests alongside the code, at the level the behaviour lives,
   asserting exact expected values.
3. Build and run the tests; read the outcome from the output, never assume it.
4. Commit and push as each piece works, not once at the end: a restart can
   end your turn, and what is pushed survives it.
5. If the spec is ambiguous or two designs trade off, ask (QUESTION block)
   rather than guess.

VERIFY BEFORE ANSWERING:
- Start a part named verify-implementation with sub_run: it checks against the
  files themselves that every part of the spec is implemented, and that every
  test you cite passed in the output.

FINAL RESPONSE, in this shape:
VERDICT: implemented | partly implemented | blocked
BRANCH: <name> at <commit>
CHANGES: each file and what changed
TESTS: what ran, and the result line quoted from the output
POINTERS: file:line for each part of the spec
COULD NOT DO: anything not done, and why
```

## While it runs

Answer its questions with the spec in hand. If a question exposes a gap in
the spec, decide it, note the decision on the thread, then reply.

## Review before reporting success

1. Check TESTS against the turns it cites (`journal_grep` the worker's
   journal for the result line).
2. Read the branch's diff yourself (with the repository at hand), or delegate
   a review (`review`) — a build agent runs builds and tests, not `git`.
3. Run the tests on an agent yourself if they matter (`test`).

If the worker is lost to a restart, what it pushed is on the branch and what
it found is in its journal entries; start the task again on the same branch,
pointing it at both.

Merging is the user's decision — ask.
