---
name: deploy
description: How to put origin/main onto one machine — one service_deploy call builds it on that machine's agent and has keep try it — then watch it to proven or rolled back, read why when it fails, stop it, and confirm it serves.
tools: [service_deploy, service_status, job_status, job_read, job_grep, job_list, job_cancel, cluster_status, log_grep]
---
# Deploying to one machine

**keep** is the supervisor on each machine that runs its services — `motherd`
(the daemon: the models, Zen, npcd), `web` (the gateway) and `mcp` (this
connector's own server). A **trial** starts a new build in place of the
installed one and keeps it once it prints that it is serving — within ten
minutes, the window restarting while it catches up on replication — and
otherwise puts the previous binary back. A **deploy** is the build and the
trial together, named by one **handle**, `deploy:<node>:<id>`, built from its build job's id.
Admins only.

## Before

- **The commit is on origin/main, pushed.** keep refuses anything else: a
  branch or local build would serve real cluster traffic, and what it writes
  replicates permanently. A branch is tested (`test`), never deployed.
- `service_status {}` — every machine's services: what each keep runs.
  `job_list {kind: "deploy"}` — the deploys under way. **One deploy per
  service per machine**: a second trial is refused while one runs.
- `cluster_status` — the machine's member, its build and `started` time.
- **A machine that runs an app's only member**: a `motherd` trial takes that
  app out of service while it runs, and every worker on it is lost. Say so to
  the user and wait for their go.
- **Tell whoever is watching** — the user, the other sessions — that a deploy
  is starting, where, and on which commit, so nothing is started twice.
- **A change the gateway must understand** (the wire format, the route table)
  goes to `web` first, on the gateway's machine, before any `motherd`: an
  older gateway that cannot read a member's new routes drops them.

## Do

1. **Start it**: `service_deploy {host: "BlackWorld", service: "motherd"}`.
   It answers at once with the deploy's handle and the build's phase. Post the
   machine and the handle.
2. **Wait for it**: `job_status {jobs: ["deploy:76c3a0f4d39bd9db:XeV_I_QPzpg"],
   wait_s: 240}` — one call
   that holds until the deploy ends or the wait runs out; call it again the
   same way while it answers *still going*. Its stages:

   | Stage | |
   |---|---|
   | building | `cargo build --release -p <service>` on that machine's agent: phase, time, lines |
   | the build failed | nothing was tried; the machine runs what it ran |
   | built and staged | keep's trial is being asked for |
   | keep is trying it | how far into the ten-minute window, and the trial log's last line |
   | proven | installed: the machine runs it from now on |
   | rolled back / cancelled | the previous binary runs again, with keep's reason |

   Post each stage as it changes.
3. **Confirm**: `cluster_status` — the member started after the trial, ready,
   on the new build. `web` and `mcp` have no member line: their `job_status`
   says proven.

## When it fails

- **The build failed**: the ended deploy's `job_status` answer gives the
  compiler errors with their line numbers; `job_read {job: <handle>, page}`
  reads around one. Fix it on a branch, test, merge, deploy again.
- **Rolled back**: the deploy's output is the build log, a marker line, then
  the trial's own stderr — the panic or exit, with line numbers, in the
  `job_status` answer; `job_grep {job: <handle>, pattern: "error"}` finds the
  rest. A rollback leaves the previous binary running: nothing is down.
- **Built, but no trial started** past a minute: the member that was to ask
  keep for it restarted. Deploy again; the build is warm.

## Stop

`job_cancel {job: <handle>}` — while it builds, the build is killed and
nothing is tried; while keep tries it, the trial is stopped and the previous
binary put back.

## Report

The machine, the handle, the commit, and how it ended — `job_status`'s
line — with the member's build and `started` time.

A proven `motherd` joins the cluster as a candidate at the higher build count;
the other members take it once it is certified, which is `release`, and the
user's decision. `web` and `mcp` are deployed machine by machine, each with
its own `service_deploy`.
