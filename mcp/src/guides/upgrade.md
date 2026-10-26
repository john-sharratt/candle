---
name: upgrade
description: How to upgrade a dependency or toolchain — research what changed first, then upgrade on a branch, build and test on an agent.
tools: [repo_task_start, repo_task_wait, repo_job_start, repo_job_log]
---
# Upgrading a dependency

## 1. Research the change first

A `research` task (read that guide) on the new version: its changelog, its
breaking changes, its migration notes — fetched from the project itself, not
summaries of it.

## 2. Find where it bites

An `ask` task: every use in the repository of what changed, with file:line.

## 3. Upgrade

`repo_task_start {thread, name: "upgrade-<dep>", effort: "deep", prompt}`:

```
GOAL: upgrade <dependency> from <old> to <new> in <repository>, on branch
`zen/upgrade-<dep>`.

CONTEXT: <the breaking changes found; the call sites found>

METHOD:
1. Bump the version; build; fix each break at its call site.
2. Run the tests of every crate that uses it.
3. Read the new version's behaviour where tests do not cover it.

FINAL RESPONSE:
VERDICT: upgraded | blocked
CHANGES: each break and how it was fixed, file:line
TESTS: result lines quoted
RISKS: behaviour that changed and is not covered by a test
COULD NOT DO: anything not done, and why
```

## 4. Test on an agent

`test` guide. A toolchain upgrade is the whole repository's decision — ask
the user before landing it.
