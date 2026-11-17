# Builds that outlive a restart

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This document is the design that
keeps every run (a build, a test) going through a restart of the agent, of
motherd or of the gateway. It finishes what each run was for and tells its
caller, after a delay if the caller was the process that restarted. It runs
one build at a time on each machine, with trials and deploys running
alongside. It writes no state file: the state is held by processes that do not
restart together, and each one rebuilds from the others.

`docs/jobs.md` §2 and §6 and `docs/battlecities_app.md` §4.2–§4.4 describe
the path as it is built today. Each one changes as the step that touches it
lands.

## 1. What went wrong

On 2026-11-17 a push to battle-cities `main` started a deploy on BlackWorld.
The build died partway through, because a `motherd` trial on the same machine
restarted the process the build agent runs in. Nothing built that commit
again until someone started it by hand. The causes, as the code stands:

| # | Cause | Where |
|---|---|---|
| 1 | **A run is a child of the agent, and the agent runs inside motherd.** A motherd trial, restart or rollout ends the process. Its commands are spawned `kill_on_drop` inside a Windows job object, so they end with it. | `host/src/cluster/mod.rs` `start_agent`; `zend-vfs/src/sandbox/process.rs`, `kill_tree.rs` |
| 2 | **Agents follow the floor.** Once a motherd build is certified, the gateway closes every agent below it, and each agent installs the new build. An agent waits `JOBS_WAIT` (60 s) for its jobs, then kills them. A standalone agent loses its builds the same way. | `cluster/src/gateway/hub.rs` `set_floor`; `agent/src/install.rs` `jobs_ended` |
| 3 | **The handoff to keep lives in one zend member's memory.** A task in the member that took `service_deploy` waits for the build, then asks keep for the trial. If that member restarts first, the build ends and no trial is ever asked for. | `zend/src/mcp/tools/service/deploy.rs` `drive` |
| 4 | **The agent's job table is in memory.** After a restart nothing is queued or re-run, and a handle from before the restart is not found. | `agent/src/jobs.rs` `Jobs::table` |
| 5 | **A started deploy counts as tried.** The gateway records the commit as soon as the member answers that it started a deploy. The once-a-minute watch then sends that commit as `attempted`, and the member does nothing. A killed build is never retried. The record is also in the gateway's memory, so a gateway restart retries even a commit that really failed. | `web/src/hooks/pushes.rs` `Told`, `tell`; `zend/src/mcp/tools/service/on_push.rs` `due` |
| 6 | **Nothing limits a machine to one build at a time.** The only lock is per checkout, so a candle build and a battle-cities build run side by side. The agent file's `slots: 4` is reported to the gateway but not enforced as a build limit. | `zend-vfs/src/sandbox/mod.rs`; `agent/src/jobs.rs` `slots` |

## 2. Who holds what

| What | Held by | Survives a restart of | Rebuilt from |
|---|---|---|---|
| A run that has started: its request, its checkout, its command, its outcome until collected | **its runner** (§3), a process of its own | the agent, motherd, the gateway | — |
| A run accepted but not started yet | **the gateway** | the agent, motherd | the agent's copy (§5) |
| The cluster's view of every run: where it is, how it ended | **the gateway** | the agent, motherd | the agents' reports on reconnect (§5) |
| A trial, and how it ended | **keep** (its board) | the agent, motherd, the gateway | — |

A machine that reboots, or a gateway and an agent restarting together while a
run was still queued, loses what was in flight. The next push watch (§7)
builds the commit again.

## 3. The runner

Every run gets a **runner**: the agent's own binary started as `tokera-agent
run`, detached from the agent. On Windows it is outside the agent's job object
(`CREATE_BREAKAWAY_FROM_JOB`) and in no console of its own, the way keep's
services outlive keep. The runner is the sandbox job, steps 1–9 of
`zend_vfs::sandbox`. It moves out of the agent's process whole:

1. It reads its `RunJob` from stdin, overlay chains included, and keeps it in
   memory. Its id is the run's id.
2. It takes the checkout's lock, then the machine's build slot (§4). Both are
   named OS mutexes, which the OS releases if the runner dies. This replaces
   the sandbox's in-process lock, because two agent processes may now look
   at one checkout.
3. It sets aside, materializes, runs, captures and puts back, as the sandbox
   does today. The command runs in a job object the runner holds, so
   cancelling a run is one kill of the runner's job, and a runner that
   crashes takes its command with it. A crash part way is recovered by the
   set-aside journal that already exists (`zend_vfs::checkout::preserve`).
4. It writes its output to the run's log, as the sandbox does now, and stages
   what it built (`Jobs::stage`).
5. It keeps its outcome (exit code, changed files as deltas, what it staged)
   and serves it on a named pipe, `\\.\pipe\tokera-run-<id>`, until an agent
   collects it and says it has been delivered (§6). It exits then, or after a
   day uncollected.

The pipe answers: what the run is (its `RunJob` without the chains, its phase
and times), the log's path, the outcome once there, `cancel`, and `delivered`.

**An agent adopts its runners when it starts.** It lists the pipes named
`tokera-run-*`, asks each one what it is, and puts it in its job table under
the run's own id, so every handle from before the restart still names it. No
file records which runners exist: the runner itself is the record.

`JOBS_WAIT` goes: an installing agent leaves its runners running and adopts
them on the new build. Causes 1, 2 and 4 are gone for every run that has
started.

## 4. One build at a time on each machine

The build slot is a named mutex for the whole machine,
`Global\tokera-build-slot`. A runner takes it after its checkout lock and
before it sets the checkout aside, and lets go once it has staged its output.
Pushes, fetches and listings never take it. Tests compile as much as builds
do, and the timed test gate needs a quiet machine (`cargo test-server` on
BlackWorld), so every run takes it.

An agent starts a runner only when the slot is free. A run accepted while the
slot is taken waits in the agent's queue as `Phase::Waiting {behind}`, naming
the run that holds the slot. Deploys (§6) go first, then other runs in the
order they came. A running build is never preempted.

- The agent reports `slots_free` as 1 when nothing holds the slot or waits for
  it, and 0 otherwise. The gateway already sends a run that names no machine
  to an agent with a free slot (`cluster/src/gateway/agents.rs`), so a
  worker's test goes to another machine while BlackWorld builds. A deploy
  names its machine and waits its turn.
- `slots:` leaves the agent file.

**Trials run alongside.** A trial is keep's process. Once a deploy's runner
has staged its build and the agent has asked keep for the trial (§6), the slot
is free for the next build. A Battle Cities trial, which spends minutes
loading its world, runs while the next candle build compiles.

## 5. The gateway's table, rebuilt both ways

The gateway already queues a run that finds no free agent
(`cluster/src/gateway/agents.rs`). It now also keeps a table of every run it
handed out, by `(node, id)`: its request without the chains, its phase, and
how it ended once it has. It serves the cluster's reads of runs
(`job_status`, `job_list`, `pending::decide`, §7) without asking the agent,
and the agent's own answers stay what a log read goes to.

- **On every connect, an agent reports its runs**: the ones queued in it,
  the runners it adopted, and the ones that ended since its last report. The
  gateway's entries for that node are replaced by the report.
- **A restarted gateway** has an empty table and fills it as the agents
  reconnect and report.
- **A restarted agent** asks the gateway, once connected, for the runs it was
  handed and has not started. They go back into its queue. Its runners it has
  found itself (§3).

Each side is the copy the other rebuilds from. Only a restart of both
together, while a run was queued and not started, loses that run.

## 6. Finishing the run, and telling its caller

When a runner's command ends, its run is finished by whichever agent holds it
then: the one that started it, or the one that adopted it after a restart.
That agent:

1. collects the outcome from the runner;
2. for a deploy, asks its own machine's keep for the trial (below);
3. reports the end to the gateway;
4. tells the runner `delivered` once the caller has it. A caller that has
   gone tells it nothing, and the runner holds its outcome until the day is
   up.

What "the caller has it" means depends on who asked:

| Caller | How it is told |
|---|---|
| a deploy | the trial keep starts is the next step. The agent asks for it, so no zend member has to be alive (cause 3). `delivered` follows the trial being accepted |
| `repo_job_start`, `job_status` | the handle is unchanged across every restart. `job_status` answers from the gateway's table and the agent; a caller waiting when the agent restarted is answered once the adopting agent reports the end |
| `run_command` inside a turn of Zen's | the run's id is derived from the call, `hash(conversation, turn, call)`. A turn resumed after a motherd restart (`zend::tools::Dispatch::Resumed`) asks the agent for the run by that id, instead of refusing the call as not replayable. A run still going is waited on. A run that ended while motherd was down has its outcome handed back and its deltas recorded in the conversation's store, and the turn carries on as if nothing had happened. `run_command`'s replay rule changes to that |

**The deploy handoff.** A `RunJob` that is a deploy says so: `deploy:
Option<String>` names the service. keep's local port (`127.0.0.1:8092`) gets
`POST /trial {service, artifact, deploy}`, which is what the cluster route
`/v1/keep/trial` takes (`keep/src/remote.rs`). The trial goes on keep's board
under the deploy's handle `deploy:<node>:<id>`, which the agent writes from
its node and the run's id, exactly as `service_deploy` answered. keep's check
that the artifact was built from a commit on `origin/main` still applies. The
run records what came of the handoff (`JobView::tried`: asked, or refused with
keep's reason), so `job_status` on a refused trial says why. `deploy.rs`
`drive` and its `tokio::spawn` are removed. For a motherd deploy,
`connector_job` runs ahead of the daemon's build on the same agent, and the
slot's order keeps the connector staged before the daemon's trial starts.

A build that finishes while motherd is restarting waits in its runner until
the new motherd's agent adopts it. It then reaches keep, minutes late and
otherwise as if nothing had happened.

## 7. Tried means it came to something

The gateway's `Told` and the `attempted` field go. Whether a commit is due is
read from the gateway's table (§5) and keep's board:

| Origin's `main` is | due? |
|---|---|
| the commit the installed binary was built from (keep's state) | no |
| a commit a deploy is under way for: a run in the table, or a trial on keep's board | no: that deploy is watched (`pending::decide`) |
| a commit whose build failed, or whose trial was rolled back | no |
| anything else, including a commit whose run was lost (§2) | yes |

The webhook and the once-a-minute watch send the same body, `{repo}`. A
gateway restart forgets failed commits until the agents report again, so it
cannot retry a failed build blindly. Cause 5 is gone.

## 8. Related: the game runs from the agent's build checkout

keep runs `battlecities` with its working directory set to the agent's
checkout (`keep/keep.yaml`, `cwd: D:/agent/battle-cities`). The game serves its
client files and its site folder from there: the site folder's files take
priority over the compiled-in site (`crates/sim/src/server/web/site.rs`). A
run of another branch on that checkout lays that branch's files over it for
the length of the run. While a worker tests a branch, the live game serves
that branch's site.

The game gets its own tree, `D:/battle-cities/live`, set to the deployed
commit when keep proves its trial. The build checkout is then only built in.
This is also where the site-only deploy belongs: a push that changes only
`web/`, `crates/sim/website/` and `docs/` moves the live tree to the new
commit and builds nothing, and the game serves the new files on the next
request.

## 9. To settle while building it

- **The runner's executable.** An agent's install replaces its own binary
  (`cluster::upgrade`). A runner must run from a file the install does not
  touch, a copy under the build it came from, and the pipe's messages must stay
  readable by the next build's agent, as saved records do.
- **Deltas held in a runner.** A run's captured deltas stay in its runner's
  memory until delivered. A run that rewrites a large tree holds that much for
  up to a day. The bound is the store's own limit on what a run may change.
- **Other platforms.** The agent also builds for Linux
  (`cluster/src/upgrade/runtime/fetch.rs`). There the pipe is a Unix socket in
  the runtime directory and the mutexes are `flock`s, released the same way
  when the process dies.

## 10. Landing it

Everything is in candle. It takes effect once deployed: keep on each machine
(`/exit` onto the new `keep.exe`), a motherd deploy (the agent and the
runners), and the gateway (the table, `Told` removed). The first motherd
deploy of it still kills the builds the old agent is running. Every one after
it does not.

| Step | Crates | Tests beside the code |
|---|---|---|
| §3 runner | `agent`, `zend-vfs` | a run started by one `Jobs` and adopted by a second one after the first is dropped ends with the same outcome; cancelling an adopted run kills its command and puts the checkout back; a runner exits once delivered |
| §4 build slot | `agent` | two runs on two checkouts never hold the slot at once; a deploy waiting goes ahead of an earlier test; `slots_free` follows the slot |
| §5 table | `cluster`, `agent` | an agent's report replaces its node's entries; a restarted gateway's table is refilled by reports; a restarted agent gets back its unstarted runs |
| §6 finishing | `agent`, `keep`, `zend` | keep's local `/trial` lands on the board under the handle; an adopted deploy hands off once; a resumed `run_command` with a derived id receives an outcome that ended while it was down |
| §7 due | `zend`, `web` | a failed build, a rolled-back trial and an installed commit are not due; a lost one is |
| §8 live tree | `zend`, `keep`, configs | a proven trial moves the live tree; a site-only push builds nothing |
