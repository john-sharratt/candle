---
name: test-write
description: How to have a worker write tests — one behaviour each, exact expected values, run before they are called done.
tools: [repo_task_start, repo_task_wait, repo_job_start, repo_job_log]
---
# Writing tests

**Use when** code lacks tests, or a bug needs a regression test.

## Prompt

`repo_task_start {thread, name: "tests-<what>", effort: "deep", prompt}`:

```
GOAL: write tests for <module | behaviour> in <repository>, on branch
`zen/tests-<what>`.

CONTEXT: <what the code does; the cases that matter; known bugs to pin down>

METHOD:
1. Read the code under test; list the behaviours it promises, edge cases
   included.
2. One test per behaviour, named for what it proves.
3. Assert exact expected values — the bytes, the string, the number — never
   "close enough" for code that is exact.
4. Test at the level the behaviour lives: a unit where a unit decides it, a
   route where the route does.
5. Run them; a test you did not see pass is not done. For a regression test,
   show it fails without the fix.

CONSTRAINTS:
- Follow the repository's test conventions (its CLAUDE.md, the project
  journal).
- Push the branch; never the default branch.

FINAL RESPONSE:
TESTS: each test, the behaviour it proves, file:line
RESULTS: the result line quoted from the run
GAPS: behaviours left untested, and why
COULD NOT DO: anything not done, and why
```
