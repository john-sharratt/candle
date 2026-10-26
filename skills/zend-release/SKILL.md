---
name: zend-release
description: Take a commit on origin/main through the whole release — tests on an agent, a keep trial on one machine, the cluster's certification of the candidate watched to its end, and the rollout watched member by member — stopping at each decision for the user's approval. Uses the zend MCP's operations tools. Use when the user asks to release or roll out a build. Admins only.
---

# Releasing a build

The long steps run on a zend thread `release-<short commit>` (zend-thread), so
the record of the release survives your context. Every decision — the rollout,
any `certify-anyway` — is the user's, asked for at the step that needs it.

## 1. Test

zend-test on `main` at the commit, for the crates the release changes. A
failure stops the release.

## 2. Trial on one machine

zend-deploy on one machine (the candidate member). It proves the build serves.
A rollback stops the release.

## 3. Certification

The candidate joins as a candidate at the higher build count; the cluster
certifies it by mirroring traffic.

- `cluster_status` every few minutes: the upgrade line shows the candidate and
  its state.
- If certification **fails or passes with gaps**, report what `cluster_status`
  and cluster-logs say. Accepting it anyway is `cluster_command {name:
  "certify-anyway", args: {app: "zend", build}}` — a decision: ask the user,
  and only with their approval pass `confirm: "approved by the user"`.

## 4. Rollout

1. Ask the gateway for the plan, which changes nothing:
   `cluster_command {name: "rollout", args: {app: "zend", build}}`.
2. Show the user the plan — the build, its source, the targets, how many at
   once — and ask for approval.
3. With approval: `cluster_command {name: "rollout", args: {app: "zend",
   build, confirm: "<build>"}, confirm: "approved by the user"}`.
4. Watch member by member with `cluster_status` until every member runs the
   build and is ready. `cluster_job_status` reads the rollout's job.
5. Report: every member, its build, `started` time and state.

## If anything goes wrong

Stop, report, and offer zend-rollback. Never press on past a failing step.
