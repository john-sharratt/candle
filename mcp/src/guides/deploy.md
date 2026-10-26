---
name: deploy
description: How to put a commit on origin/main onto one machine — build and stage it on that machine's agent, trial it through keep, watch it to proven or rolled back, confirm it serves.
tools: [repo_job_start, repo_job_log, service_trial, service_status, cluster_status, cluster_logs]
---
# Deploying to one machine

**keep** is the supervisor on each machine that runs its services — motherd
(the daemon that runs the models) and web (the gateway). A **trial** starts a new build in place
of the installed one and keeps it only if it reports serving within ten
minutes; otherwise it puts the previous binary back. Admins only.

## Before

- The commit is on origin/main. keep refuses anything else: a local or branch
  build would serve real cluster traffic, and everything it writes replicates
  permanently. A branch is tested (`test`), never deployed.
- `cluster_status`: note the machine's member, build and `started` time, and
  its agent's node id. No agent listed for the machine: say so and stop.
- **If the machine runs an app's only member**, the trial takes that app out
  of service while it runs, and every worker running on it is lost. Say so to
  the user and wait for their go before starting.
- **If the change alters something the gateway must understand** (the
  cluster's wire format, its route table), the gateway goes first: trial `web`
  on the gateway's machine before any `motherd`. An older gateway that cannot
  read a member's new route table drops that member's routes.

## Do

1. **Build and stage** on that machine's agent:
   `repo_job_start {node: <that machine's agent>, repo, branch: "main",
   program: "cargo", args: ["build", "--release", "-p", "motherd"],
   stage: "target/release/motherd.exe", timeout_s: 3600}`
   (`web` the same way: `-p web`, `target/release/web.exe`).
2. **Wait for it**: `repo_job_log {node, job}` until it has ended; its
   answer says where it was staged — use that path, never a guessed one: it
   differs from machine to machine. A failure: report the build error
   (`grep: "error"`) and stop.
3. **Trial**: `service_trial {host, service: "motherd", artifact: <staged
   path>}`.
4. **Watch**: `service_status {host}` every few minutes until the trial has
   ended: proven or rolled back.
5. **Confirm**: `cluster_status` — the member started after the trial, ready,
   on the new build. A gateway (`web`) has no member line: confirm it with
   `service_status`, whose trial line says proven.

## Report

Proven or rolled back, `service_status`'s line, the member's build and
`started` time. A rollback leaves the previous binary running — nothing is
down; read why with `logs` from the trial's start.

Where other members serve the app meanwhile, a trial of a commit on main is
safe to run. Rolling the build out to other members is `release`, and the
user's decision.
