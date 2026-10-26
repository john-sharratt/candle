---
name: review
description: How to have workers review a change — correctness, regressions, missing tests, convention breaks — with every finding verified against the code before it is reported.
tools: [task_start, task_wait, journal_grep, timeline_read]
---
# Reviewing code

**Use when** a review reads more than a handful of files: a branch before
merge, a module audit. For security, use `security-review`.

Split a large change by area; one task each, started together.

## Prompt

`task_start {thread, name: "review-<area>", effort: "exhaustive", prompt}`:

```
GOAL: review <branch | these files | this diff> of <repository>.

CONTEXT: <what the change is for; the design it implements; areas of concern>

LOOK FOR:
- correctness bugs: what input or state gives a wrong result or a crash;
- regressions against the code before the change;
- missing tests for behaviour the change adds;
- breaks of the repository's conventions (its CLAUDE.md; the project journal).

METHOD:
1. Read every file you will cite.
2. For each candidate finding, construct the failing input or state; a
   finding you cannot make fail is not a finding.
3. Check yourself: drop duplicates; rank by severity.

VERIFY BEFORE ANSWERING:
- Start a part named verify-findings with sub_run, giving it your findings: it
  re-reads each cited line and keeps only what it can confirm, marking each
  CONFIRMED or REFUTED with its reason.

FINAL RESPONSE, confirmed findings only, most severe first:
FINDINGS:
- [severity] file:line — the defect, in one sentence
  FAILURE: the input or state that triggers it, and what goes wrong
  EVIDENCE: the quoted lines
REFUTED: each finding the verifier dropped, and why
CLEAN: what was reviewed and found sound
COULD NOT DO: what was not reviewed, and why
```

## Read the result

Check each finding's EVIDENCE against the code yourself before acting on it
or passing it on — the verifier is a filter, not proof. A finding without
quoted lines is not a finding. Fixing findings on someone else's branch, or
merging, is the user's call.
