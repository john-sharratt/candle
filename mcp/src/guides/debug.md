---
name: debug
description: How to find a bug's root cause with a worker — reproduce, narrow by evidence, fix the cause with a test that failed before it.
tools: [task_start, task_wait, task_reply, cluster_logs, agent_job, agent_job_log]
---
# Debugging to the root cause

**Use when** a bug, a failing test, or a wrong answer needs its cause found.
A workaround is not a fix; a restart is not a diagnosis.

## Prompt

`task_start {thread, name: "debug-<symptom>", effort: "exhaustive", prompt}`:

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
- Work on a branch `zen/fix-<what>`; push it; never the default branch.
- No workarounds that hide the symptom.

VERIFY BEFORE ANSWERING:
- The new test failed before the fix and passes after — quote both results.

FINAL RESPONSE:
CAUSE: the root cause, with file:line
EVIDENCE: what proved it — log lines, values, the reproduction
HYPOTHESES RULED OUT: each, and what killed it
FIX: what changed, branch and commit
TEST: the test, its result before and after
COULD NOT DO: anything not done, and why
```

## Read the result

Check the CAUSE against its EVIDENCE: a cause that does not explain every
symptom is not the cause. For a production symptom, read `cluster_logs`
yourself first and put the lines in CONTEXT.
