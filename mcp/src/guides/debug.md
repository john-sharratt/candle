---
name: debug
description: How to find a bug's root cause with a worker — reproduce, narrow by evidence, fix the cause with a test that failed before it.
tools: [repo_task_start, repo_task_wait, repo_task_reply, cluster_logs, repo_job_start, repo_job_log]
---
# Debugging to the root cause

**Use when** a bug, a failing test, or a wrong answer needs its cause found.
A workaround is not a fix; a restart is not a diagnosis.

## Prompt

`repo_task_start {thread, name: "debug-<symptom>", effort: "exhaustive", prompt}`:

```
GOAL: find the root cause of <symptom> in <repository>, and fix it.

CONTEXT: <how it shows: the error, the log lines, the test, when it started,
what changed around then; what has been ruled out>

METHOD:
1. Reproduce it first. If it does not reproduce, say so and say what you
   tried — do not fix what you cannot see.
2. Narrow: form one hypothesis, find the evidence that confirms or kills it,
   then the next. Test, narrow, test.
3. When the cause is found, write a test that fails because of it.
4. Fix the cause, not the symptom; run the test and the suite around it.

CONSTRAINTS:
- Scope: <the paths or components to search first>.
- Work on a branch `zen/fix-<what>`; push it; never the default branch.
- No workarounds that hide the symptom.

VERIFY BEFORE ANSWERING:
- The new test failed before the fix and passes after — quote both results.
- Every file:line cited quotes the line as read.

DONE WHEN: the cause is proven and fixed, or it cannot be reproduced.

FINAL RESPONSE:
CAUSE: the root cause, with file:line
EVIDENCE: what proved it — log lines, values, the reproduction
HYPOTHESES RULED OUT: each, and what killed it
FIX: what changed, branch and commit
TEST: the test, its result before and after
UNVERIFIED: what you could not confirm, and what you could not read
COULD NOT DO: anything not done, and why
```

**To find the cause without fixing it** — a question about behaviour, or a
fix the user will decide on — make it read only: GOAL "find the root cause",
CONSTRAINTS "read only — change nothing, push nothing", drop METHOD steps 3–4,
and in place of FIX and TEST ask for PROPOSED FIX (what to change, with
file:line). `effort: deep` is enough for that.

## While it works

Narrowing reads many files, and a worker that reaches its round limit settles
as **stuck** with what it has. Its answer so far is usually most of the way:
`repo_task_reply` with what to do next — "quote the lines for these claims",
"check whether X also covers Y" — and it continues with a fresh set of rounds
and its context intact.

## Read the result

Check the CAUSE against its EVIDENCE: a cause that does not explain every
symptom is not the cause. For a production symptom, read `cluster_logs`
yourself first and put the lines in CONTEXT.
