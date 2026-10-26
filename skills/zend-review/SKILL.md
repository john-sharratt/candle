---
name: zend-review
description: Delegate a code review to zend — a branch, a diff, or a set of files reviewed for correctness bugs, regressions, missing tests and convention breaks, with an adversarial verifier part checking every finding against the code before it is reported. Use for any review that reads more than a few files. Follows zend-thread.
---

# Reviewing code on zend

## When

A review that reads more than a handful of files: a branch before merge, a
module audit, a security pass (say so in the prompt — zend has a
`security_review` layer to draw on).

## Thread

The feature's thread, or `review-<branch>` for a standalone review.

## Delegate

For a large change, split it: one `task_start` per area, started together,
then one `task_wait` over all of them. Each with this prompt:

```
Review <branch | these files | this diff> of <repository> for:
- correctness bugs: what input or state gives a wrong result or a crash;
- regressions against the code before the change;
- missing tests for behaviour the change adds;
- breaks of the repository's conventions (its CLAUDE.md; the project journal holds them).

Read every file you cite. For each finding, re-read the lines and quote them.
Before you answer, start a part named verify-findings with sub_run, giving it your findings: it
re-reads each cited line and keeps only the findings it can confirm, marking each CONFIRMED or
REFUTED with its reason.

Final response, in this shape, confirmed findings only, most severe first:
FINDINGS:
- [severity] file:line — the defect, in one sentence
  FAILURE: the input or state that triggers it, and what goes wrong
  EVIDENCE: the quoted lines
REFUTED: each finding the verifier dropped, and why
CLEAN: what was reviewed and found sound
```

## Read the result

- Check each finding's `EVIDENCE` against the code yourself before you act on
  it or pass it on. The verifier is a filter, not proof.
- A finding without quoted lines is not a finding.

## Approval

Fixing findings on someone else's branch, or merging, is the user's call.

## If zend is down

Review it yourself.
