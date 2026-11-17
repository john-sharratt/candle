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

Every run gets a **runner**: `tokera-runner`, a small binary of the `agent`
crate, started detached (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
CREATE_NO_WINDOW`, as keep starts its services, `keep/src/trial.rs`). A build
dies today because the agent holds the job object its command runs in, and
closing the agent closes the job (`zend-vfs/src/kill_tree.rs`, `Tree::drop`).
The runner holds that job instead.

A runner never runs from a file a switch replaces. The agent in motherd has
no binary of its own: its executable is `keep/stable/motherd.exe`, which keep
overwrites on every switch, and a running image cannot be overwritten on
Windows. So each agent, as it starts, copies the `tokera-runner` of its own
build into `jobs_dir/runners/<count>-<hash>/` and starts runners from there.
A copy no runner is using is removed once its build is two behind.

The runner is the sandbox job, steps 1–9 of `zend_vfs::sandbox`. It moves out
of the agent's process whole:

1. It reads its `RunJob` from stdin, overlay chains included, and keeps it in
   memory. Its id is the run's id.
2. It takes the checkout's lock, then the machine's build slot (§4). Both are
   named OS mutexes, which the OS releases if the runner dies. This replaces
   the sandbox's in-process lock (a `tokio::sync::Mutex`,
   `zend-vfs/src/sandbox/mod.rs`), because a runner and a restarted agent
   are now different processes looking at one checkout.
3. It sets aside, materializes, runs, captures and puts back, as the sandbox
   does today. The command runs in a job object the runner holds, so
   cancelling a run is one kill of the runner's job, and a runner that
   crashes takes its command with it. A crash part way is recovered by the
   set-aside journal that already exists (`zend_vfs::checkout::preserve`).
4. It writes its output to the run's log, as the sandbox does now, and stages
   what it built (`Jobs::stage`).
5. It keeps its outcome (exit code, changed files as deltas, what it staged)
   and serves it on a named pipe, `\\.\pipe\tokera-run-<id>`, until an agent
   tells it the outcome has been delivered (§6). It exits then, or after a
   day undelivered.

The pipe answers: what the run is (its `RunJob` without the chains, its phase
and times), the log's path, the outcome once there, `cancel`, and `delivered`.
The pipe's DACL admits the agent's own account alone, and an agent believes a
pipe only when its server process is a `tokera-runner` from its runners folder
(`GetNamedPipeServerProcessId`).

**An agent adopts its runners when it starts, before anything else touches a
checkout.** In this order:

1. It lists the pipes named `tokera-run-*`, asks each one what it is, and
   puts it in its job table under the run's own id, so every handle from
   before the restart still names it. No file records which runners exist:
   the runner itself is the record.
2. It joins the gateway and reports (§5), then takes back the runs it was
   handed and had not started. A run whose id it adopted in step 1 is not
   queued again, so a run that started just before a restart never runs twice.
3. Only then does it run its start-up work on the checkouts no adopted runner
   holds: each checkout's declared setup (`Jobs::set_up`), the stale git lock
   sweep (`stale_locks`, which already spares a lock whose git still runs) and
   the disk sweep (`sweep`, which counts adopted runs as holding their
   checkout's `target`).

Every read of a run in `Jobs` goes through the checkout's in-process job
server today (`view`, `tail`, `log`, `holding` call `server.query_job`). They
read the runner instead, through its pipe and its log file, for a run the
agent started and for one it adopted alike: one path, not two.

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

A run that finds no free slot queues on an agent, never in the gateway
(`cluster/src/gateway/agents.rs`): the agent holds its queue. The gateway now
keeps a table of every run it handed out, by `(node, id)`: its request without
the chains, its machine, its base commit, its phase, and how it ended once it
has. It serves the cluster's reads of runs (`job_status`, `job_list`,
`pending::decide`, §7) without asking the agent. A log read still goes to the
agent.

- **On every connect, and on every change after, an agent reports its runs**:
  the ones queued in it, the runners it holds, and the ones that ended.
- **A report merges into the table; it never replaces it.** A run the report
  names takes the report's state. A run in the table that the report leaves
  out is left as it is: an unstarted one is a run the agent lost, and it is
  handed back (below). An ended one stays as the record of how it ended (§7),
  however many agent restarts later. Ended runs are kept by age (30 days) and
  by count (the newest 1,024 a machine), not by the agent's `KEPT`.
- **A restarted gateway** has an empty table and fills it as the agents
  reconnect and report. What it loses is the ended runs the agents no longer
  hold.
- **A restarted agent** takes back, once its report is merged, the runs the
  table holds for its node as queued and that it neither queued nor adopted
  (§3, step 2).

Each side is the copy the other rebuilds from. Only a restart of both
together, while a run was queued and not started, loses that run.

## 6. Finishing the run, and telling its caller

When a runner's command ends, its run is finished by whichever agent holds it
then: the one that started it, or the one that adopted it after a restart.
That agent:

1. collects the outcome from the runner;
2. for a deploy, asks its own machine's keep for the trial (below);
3. reports the end to the gateway;
4. tells the runner `delivered` once the outcome is kept where the caller
   reads it. Until then the runner holds it, so an agent that restarts
   between any two of these steps does them again from the runner. Each step
   is safe to do twice. An outcome no caller ever takes is held until the
   day is up.

Where the outcome is kept, and so when `delivered` is said, depends on who
asked:

| Caller | Kept where | `delivered` once |
|---|---|---|
| a deploy | keep's board, under the deploy's handle. The agent asks for the trial, so no zend member has to be alive (cause 3) | keep's board names the handle, as running or ended |
| `repo_job_start`, `job_status` | the gateway's table. The handle is unchanged across every restart; a caller waiting when the agent restarted is answered once the adopting agent reports the end | the gateway has merged the end |
| `run_command` inside a turn of Zen's | the turn's own record of the call's result in the substrate | zend has written the result. A turn resumed after a motherd restart (`zend::tools::Dispatch::Resumed`) asks the agent for the run by its id (below) instead of refusing the call: a run still going is waited on; a run that ended while motherd was down has its outcome handed over and its deltas recorded in the conversation's store, and the turn carries on. An id no agent holds was never started, and the call runs |

**A `run_command`'s run id is derived from the call**, so a resumed turn can
name it without any record: `hash(conversation, turn, call index)`, written in
the shape every job id has (11 characters, the last one of 16,
`tool_constraints.yaml`). `zend_tools::Replay` gains `Reattach` beside `Safe`
and `Unsafe` for a tool whose call, resumed, is joined to what it already
started.

**keep answers a trial asked twice with the trial it has.** `/trial` with a
deploy handle that its board already names, running or ended, answers that
trial and starts nothing. A second ask after an agent restart is then
harmless, where today it would try the artifact again.

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

The webhook and the once-a-minute watch send the same body, `{repo}`. An
agent's restart no longer forgets a failed build, because the table keeps it
(§5). A gateway restart does forget it, so a commit whose build failed before
the gateway restarted is built once more. A rolled-back trial is still on
keep's board. Cause 5 is gone.

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

- **The pipe across builds.** A runner started by one build is adopted by the
  next build's agent, so the pipe's messages must stay readable by old and new
  binaries, as saved records do. The `tokera-runner` copy (§3) must be part of
  every build that carries an agent, `motherd` included
  (`docs/replicated_substrate.md` §3.9).
- **Global names.** A `Global\` mutex needs `SeCreateGlobalPrivilege` outside
  session 0. keep runs its services from a scheduled task. Confirm the session
  they run in before choosing `Global\` or `Local\`.
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
| §3 runner | `agent`, `zend-vfs` | a run started by one `Jobs` and adopted by a second one after the first is dropped ends with the same outcome; cancelling an adopted run kills its command and puts the checkout back; a runner exits once delivered; a run both adopted and handed back by the gateway runs once; start-up setup and sweeps leave an adopted run's checkout alone |
| §4 build slot | `agent` | two runs on two checkouts never hold the slot at once; a deploy waiting goes ahead of an earlier test; `slots_free` follows the slot |
| §5 table | `cluster`, `agent` | a report merges, and an ended run it leaves out stays; a restarted gateway's table is refilled by reports; a restarted agent gets back its unstarted runs |
| §6 finishing | `agent`, `keep`, `zend`, `zend-tools` | keep's local `/trial` lands on the board under the handle, and asked again with the handle starts nothing; an adopted deploy hands off once; a resumed `run_command` with a derived id receives an outcome that ended while it was down, and one never started runs |
| §7 due | `zend`, `web` | a failed build, a rolled-back trial and an installed commit are not due; a lost one is |
| §8 live tree | `zend`, `keep`, configs | a proven trial moves the live tree; a site-only push builds nothing |
