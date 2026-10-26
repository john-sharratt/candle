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

CONSTRAINTS:
- Push the branch; never push to the default branch, never force-push.
- Build to check it compiles; do not run the test suites yourself — they run
  on a build agent afterwards.

METHOD:
1. Bump the version; build; fix each break at its call site.
2. Commit and push as each crate builds.
3. Read the new version's behaviour where tests do not cover it.

FINAL RESPONSE:
VERDICT: upgraded | blocked
BRANCH: <name> at <commit>
CHANGES: each break and how it was fixed, file:line
RISKS: behaviour that changed and is not covered by a test
UNVERIFIED: what you could not confirm
COULD NOT DO: anything not done, and why
```

## 4. Test on an agent

`test` guide, on the branch, for every crate that uses the dependency.
Landing any upgrade is the user's decision — a toolchain upgrade most of all,
since it touches the whole repository.
