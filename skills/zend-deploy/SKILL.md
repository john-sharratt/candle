---
name: zend-deploy
description: Deploy a commit on origin/main to one machine with one call — service_deploy builds it on that machine's agent and has keep try it, promoting it only if it serves and restoring the previous binary otherwise — then wait it out with job_status, read why when it fails, and confirm it is serving. Uses the zend MCP's service_deploy, service_status, job_status, job_read, job_grep and job_cancel tools. Use to put a merged build on one member. Admins only.
---

# Deploying a build to one machine

The `deploy` guide (`guide_read`) carries the whole procedure; this is its
short form.

## Before

- The commit is on `origin/main`, pushed. keep refuses anything else: a local
  build serves real cluster traffic while it runs, and what it writes
  replicates and is permanent. A branch is tested with zend-test, never
  deployed.
- `service_status {}`: what every machine runs. `job_list {kind: "deploy"}`:
  that no deploy is already under way on the one you want.
- `cluster_status`: the machine's member, its build and `started` time.
- Say — to the user and the other sessions — that a deploy is starting, on
  which machine and commit.

## Do

1. **Start it**: `service_deploy {host: "BlackWorld", service: "motherd"}`
   (`web`, `mcp` the same way). It answers at once with the deploy's handle,
   `deploy:<node>:<id>`. Post the machine and the handle.
2. **Wait for it**: `job_status {jobs: ["deploy:76c3a0f4d39bd9db:XeV_I_QPzpg"],
   wait_s: 240}`, again while it answers *still going*. It says each stage — building, built,
   keep trying it (how far into its ten-minute window), proven, rolled back.
   Post each stage.
3. **Confirm**: `cluster_status` — the member's `started` time is after the
   trial, its state `ready`, its build the new one.
4. Report: the machine, the handle, how it ended, the member's build and
   `started` time.

## If it failed

- The build, or the trial rolled back: the ended deploy's `job_status` answer
  quotes the first errors, the panic or exit, with line numbers; `job_read
  {job: <handle>, page}` and `job_grep {job: <handle>, pattern}` read the rest
  (the build log, a marker line, then the trial log). The previous binary runs
  again: nothing is down.

## Stop

`job_cancel {job: <handle>}` stops the build, or rolls the trial back.

## Approval

A trial of a commit on main is safe to run — keep rolls a bad one back.
Rolling the build out to the other members is zend-release, and the user's
decision.
