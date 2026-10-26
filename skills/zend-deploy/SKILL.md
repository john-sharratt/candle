---
name: zend-deploy
description: Deploy a commit on origin/main to one machine — build it on that machine's agent, stage the binary, and run it as a keep trial there, which promotes it only if it serves and restores the previous binary otherwise — then confirm it is serving. Uses the zend MCP's agent_job, agent_job_log, service_trial and service_status tools. Use to put a merged build on one member. Admins only.
---

# Deploying a build to one machine

## Before

- The commit is on `origin/main`. keep refuses anything else: a local build
  serves real cluster traffic while it runs, and what it writes replicates
  and is permanent. A branch is tested with zend-test, never deployed.
- `cluster_status`: note the machine's member, its build and `started` time.

## Do

1. **Build and stage** on that machine's agent:
   `agent_job {node: <that machine's agent>, repo: "candle", branch: "main",
   program: "cargo", args: ["build", "--release", "-p", "motherd"],
   stage: "target/release/motherd.exe", timeout_s: 3600}`. (`web` the same
   way, `-p web`, `target/release/web.exe`.)
2. **Wait for it**: `agent_job_log {node, job}` until it has ended. Its first line
   says `staged at <path>` — the artifact on that machine. If it failed or
   staged nothing, report the build error (`grep: "error"`) and stop.
3. **Trial**: `service_trial {host, service: "motherd", artifact: <staged path>}`.
   keep stops the service, runs the new binary from the stable path, and
   promotes it once it prints its serving line within ten minutes — or puts
   the previous binary back.
4. **Watch**: `service_status {host}` every few minutes until the trial has ended:
   `last trial … proven` or `rolled back`.
5. **Confirm**: `cluster_status` — the member's `started` time is after the
   trial, its state `ready`, and its build the new one.
6. Report: proven or rolled back, with `service_status`'s line, the member's build
   and `started` time.

## If it rolled back

The previous binary is running again: nothing is down. Read why with
cluster-logs (`since` the trial's start, `level: ERROR`) and report it.

## Approval

A trial of a commit on main is safe to run — keep rolls a bad one back.
Rolling the build out to the other members is zend-release, and the user's
decision.
