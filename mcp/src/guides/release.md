---
name: release
description: How to take a commit on origin/main through a full release — tests, a trial on one machine, certification, then a rollout member by member — stopping at each decision for the user's approval.
tools: [repo_job_start, repo_job_log, service_trial, service_status, cluster_command, cluster_job_status, gateway_command, cluster_status, cluster_logs, journal_note]
---
# Releasing a build

Run it on a thread `release-<short commit>` so the record survives your
context. Every decision — the rollout, any `certify-anyway`, a floor change —
is the user's, asked at the step that needs it. Admins only.

## 1. Test

Follow the `test` guide on `main` at the commit, for the crates the release
changes (the crates its changed paths are in). A failure stops the release.

## 2. Trial on one machine

Follow the `deploy` guide on one machine — the candidate. It proves the build serves. A
rollback stops the release. If the release changes what the gateway reads,
the gateway (`web`) is trialled first.

When the app has only one member (`cluster_status`), the trial takes it out
of service while it runs: say so to the user before starting it.

## 3. Certification

The candidate joins at the higher build count; the cluster certifies it by
mirroring traffic.

- `cluster_status` every few minutes: the upgrade line shows the candidate and
  its state.
- If certification fails or passes with gaps, report what `cluster_status`
  and `cluster_logs` say. Accepting it anyway is `cluster_command {name:
  "certify-anyway", args: {app, build}}` — a decision: ask, and pass
  `confirm: "approved by the user"` only with approval.

## 4. Rollout

1. Ask for the plan, which changes nothing and needs no approval:
   `cluster_command {name: "rollout", args: {app, build}}` — no `confirm`
   anywhere.
2. Show the user the plan — build, source, targets, how many at once — and
   ask.
3. With approval: `cluster_command {name: "rollout", args: {app, build,
   confirm: "<build>"}, confirm: "approved by the user"}`.
4. Watch member by member with `cluster_status` until every member runs the
   build and is ready; `cluster_job_status` reads the rollout's job.
5. The grace period: older builds keep serving until the app's floor rises.
   Ending it early is `gateway_command {name: "raise-floor", args: {app}}` —
   a decision; ask.

## Report

Every member, its build, `started` time and state; note the release on the
thread.

## If anything goes wrong

Stop, report, and offer `rollback`. Never press on past a failing step.
