---
name: security-review
description: How to have a worker review a change for security — threat-model first, every finding an exploit path from attacker-controlled input to a sink.
tools: [repo_task_start, job_status, journal_grep, timeline_read]
---
# Security review

**Use when** a change, or an area of code as it stands, touches
authentication, identity, input parsing, secrets, the network, file paths,
sandboxing or anything that runs commands.

## Prompt

`repo_task_start {thread, name: "security-<area>", role: "reviewer", effort: "exhaustive", prompt}`:

```
GOAL: review <the change: branch or commits — or the area: the feature and
its entry points> of <repository> for security defects.

CONTEXT: <what it does; who calls it; what it trusts and why>

CONSTRAINTS:
- Scope: <the paths and entry points to trace from>.
- Read only — change nothing, push nothing.

METHOD:
1. Threat model: who is the attacker, what do they control (headers, bodies,
   paths, tokens, peers), what would they want.
2. Trace each attacker-controlled input to every sink it reaches: a command,
   a file path, a query, an identity decision, a log, a response.
3. For each path, decide: is there a check, does it hold for every input,
   can it be bypassed (encoding, case, a second header, a race)?
4. Check yourself: a pattern match is not a finding; only an exploit path is.

VERIFY BEFORE ANSWERING:
- Start a reviewer task named verify-findings with repo_task_start, giving it
  your findings, and wait on it with job_status: it re-reads each path and
  keeps only findings with a concrete exploit. Put its verdict in your answer.

FINAL RESPONSE:
THREAT MODEL: attacker, what they control, what they want
FINDINGS (most severe first):
- [severity] file:line — the defect
  EXPLOIT: the input, the path it takes, what the attacker gains
  EVIDENCE: the quoted lines
SOUND: trust boundaries checked and found to hold, each with file:line
VERIFIED: what verify-findings kept and dropped, and why
UNVERIFIED: what you could not confirm, and what you could not read
COULD NOT DO: what was not covered, and why
```

## While it works

The verify-findings task is the worker's own: it goes on the same thread and
`thread_list` lists it beside the worker. While it runs the worker's progress
line reads `job_status` with counters that do not move, sometimes for half an
hour — that is the worker waiting on its verifier, not a hang. The verdict
reaches you in the worker's VERIFIED. Tasks nest one level, so the verifier
starts no task of its own.

## Read the result

Confirm each exploit against the code before acting. Report findings to the
user privately when they are live.
