---
name: security-review
description: How to have a worker review a change for security — threat-model first, every finding an exploit path from attacker-controlled input to a sink.
tools: [repo_task_start, repo_task_wait, journal_grep, timeline_read]
---
# Security review

**Use when** a change touches authentication, identity, input parsing,
secrets, the network, file paths, sandboxing or anything that runs commands.

## Prompt

`repo_task_start {thread, name: "security-<area>", effort: "exhaustive", prompt}`:

```
GOAL: review <change> of <repository> for security defects.

CONTEXT: <what it does; who calls it; what it trusts and why>

METHOD:
1. Threat model: who is the attacker, what do they control (headers, bodies,
   paths, tokens, peers), what would they want.
2. Trace each attacker-controlled input to every sink it reaches: a command,
   a file path, a query, an identity decision, a log, a response.
3. For each path, decide: is there a check, does it hold for every input,
   can it be bypassed (encoding, case, a second header, a race)?
4. Check yourself: a pattern match is not a finding; only an exploit path is.

VERIFY BEFORE ANSWERING:
- Start a part named verify-findings with sub_run: it re-reads each path and
  keeps only findings with a concrete exploit.

FINAL RESPONSE:
THREAT MODEL: attacker, what they control, what they want
FINDINGS (most severe first):
- [severity] file:line — the defect
  EXPLOIT: the input, the path it takes, what the attacker gains
  EVIDENCE: the quoted lines
SOUND: trust boundaries checked and found to hold
COULD NOT DO: what was not covered, and why
```

## Read the result

Confirm each exploit against the code before acting. Report findings to the
user privately when they are live.
