# Builds that outlive a restart

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This document is the design that
keeps a build running through those restarts, runs builds one at a time on
each machine, and lets trials and deploys run while the next build goes on.
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
| 1 | **The build agent runs inside motherd.** A motherd trial, restart or rollout ends the process. Its builds are spawned `kill_on_drop` inside a Windows job object, so they end with it. | `host/src/cluster/mod.rs` `start_agent`; `motherd/motherd.yaml` `agent:`; `zend-vfs/src/sandbox/process.rs`, `kill_tree.rs` |
| 2 | **Agents follow the floor.** Once a motherd build is certified, the gateway closes every agent below it, and each agent installs the new build. An agent waits `JOBS_WAIT` (60 s) for its jobs, then kills them. A standalone agent still loses its builds after every motherd rollout. | `cluster/src/gateway/hub.rs` `set_floor`; `agent/src/install.rs` `jobs_ended` |
| 3 | **The handoff to keep lives in one zend member's memory.** A task in the member that took `service_deploy` waits for the build, then asks keep for the trial. If that member restarts first, the build ends and no trial is ever asked for. | `zend/src/mcp/tools/service/deploy.rs` `drive` |
| 4 | **The agent's job table is in memory.** After a restart nothing is queued or re-run, and a handle from before the restart is not found. | `agent/src/jobs.rs` `Jobs::table` |
| 5 | **A started deploy counts as tried.** The gateway records the commit as soon as the member answers that it started a deploy. The once-a-minute watch then sends that commit as `attempted`, and the member does nothing. A killed build is never retried. The record is also in the gateway's memory, so a gateway restart retries even a commit that really failed. | `web/src/hooks/pushes.rs` `Told`, `tell`; `zend/src/mcp/tools/service/on_push.rs` `due` |
| 6 | **Nothing limits a machine to one build at a time.** The only lock is per checkout, so a candle build and a battle-cities build run side by side. The agent file's `slots: 4` is reported to the gateway but not enforced as a build limit. | `zend-vfs/src/sandbox/mod.rs`; `agent/src/jobs.rs` `slots` |

## 2. Two agents on a machine

The in-process agent keeps what motherd's tenants read and write through it
(`docs/substrate_git.md` §9): `journals`, `journals-npcd` and `battle-mind`.
zend serves the journals from its checkout, and npcd fetches its mind through
it, so these checkouts stay in the process. Their jobs are git fetches and
pushes that take seconds.

The checkouts that are built from (`candle`, `battle-cities`, `yenbot`) move to
a standalone `tokera-agent`. keep supervises it as its own service, as it
already does on AlienWorld (`docs/deployment.md`, "AlienWorld's standby
agent"):

- `motherd/agent.yaml` keeps only the tenants' checkouts. A new
  `motherd/agent.builds.yaml` names the build checkouts, in the same
  directories (`D:/agent/candle`, …), so nothing is cloned again. It has its
  own `jobs_dir`, because an agent's node is kept there (`agent::member::node_in`).
  It keeps the same `stage_dir`, `keep/stage`.
- keep gets an `agent` service in `keep/keep.yaml`: `keep/stable/tokera-agent.exe
  --config motherd/agent.builds.yaml --cluster … --group-key … --host
  BlackWorld`. On WhiteWorld it goes in `C:\prog\keep-local.yaml`, with
  `agent-local.yaml` split the same way.
- The agent file gains `priority: below | normal`. This is the machine's
  setting, not a feature flag: a standalone agent next to a model member
  builds below normal priority, as the in-process agent does today.
  `agent/src/main.rs` currently hard-codes `Priority::Normal`.
- The gateway already prefers a dedicated agent to one running in a member's
  process (`docs/replicated_substrate.md` §11.5). With the build checkouts only
  on the standalone agent, every build and test run goes there.

A motherd trial, restart or rollout then leaves builds alone. Cause 1 is gone.

**An agent drains before it installs** (cause 2). When the floor moves past
it, the agent stops taking runs, refusing them with a reason the gateway
reads as "busy, send it elsewhere". It waits for its running builds to end,
each within its own timeout, and only then installs. `JOBS_WAIT` goes. Runs
still queued are recorded (§5) and start on the new build.

## 3. One build at a time on each machine

`Jobs` holds one **build slot** for the machine. Every run takes it after its
fetch and store are ready and before its sandbox job starts. It gives the slot
back when the command has ended and its output is staged. Pushes, fetches and
listings never take it.

- A run waiting for the slot is `Phase::Waiting {behind}`, naming the run that
  holds it, which is the phase a run waiting for its checkout shows today.
- Deploy builds (§4) go ahead of other waiting runs. Other runs go in the
  order they arrived. A running build is never preempted.
- The agent reports `slots_free` as 0 while the slot is held or a run is
  waiting for it, and 1 otherwise. The gateway already sends a run that does
  not name its machine to an agent with a free slot
  (`cluster/src/gateway/agents.rs`), so a worker's test spills to another
  machine while BlackWorld builds. A deploy names its machine and waits its turn.
- `slots:` leaves the agent file. A machine runs one build at a time.

Tests compile as much as builds do, and the timed test gate needs a quiet
machine (`cargo test-server` on BlackWorld), so a test run takes the slot too.

**Trials run alongside.** A trial is keep's process, not the agent's. Once the
agent has handed a build to keep (§4), the slot is free for the next build. A
Battle Cities trial, which spends minutes loading its world, runs while the
next candle build compiles.

## 4. The agent hands the build to keep

A `RunJob` that is a deploy says so: `deploy: Option<String>` names the
service. When such a run exits 0 and its binary is staged, the agent asks its
own machine's keep for the trial. keep's local port (`127.0.0.1:8092`) gets
`POST /trial {service, artifact, deploy}`, which is what the cluster route
`/v1/keep/trial` takes (`keep/src/remote.rs`): the trial is accepted at once,
runs as keep's job, and goes on keep's board under the deploy's handle
`deploy:<node>:<id>`. The agent knows its node and the job's id, so it writes
the same handle `service_deploy` answered with. keep's check that the artifact
was built from a commit on `origin/main` still applies.

- The agent file names keep's address (`keep: 127.0.0.1:8092`). An agent with
  no keep refuses a deploy run before it starts.
- The run records what came of the handoff (`JobView::tried`: asked, or
  refused with keep's reason). `job_status` on a deploy whose trial keep
  refused says why, where it now says nothing.
- `deploy.rs` `drive` and its `tokio::spawn` are removed. `service_deploy`
  submits the build with `deploy` set and answers as it does now. A zend
  restart no longer touches a deploy. Cause 3 is gone.
- For a motherd deploy, `connector_job` is queued ahead of the daemon's build
  on the same agent. The slot's order keeps the connector staged before the
  daemon's trial starts.

## 5. The agent writes its runs to disk

Each run is written to `jobs_dir/runs/<id>.json` when it is accepted: the
`RunJob`, its id, its line (branch, base, command, times) and its phase. It is
rewritten when the run ends, with its status, what it staged and what came of
the handoff. On start the agent reads the folder back, keeping the newest
`KEPT`:

| A recorded run that | On start |
|---|---|
| ended | is listed and read like any ended job, so a handle from before the restart is still answered |
| had not ended, and is a deploy | runs again under its own id: it fetches origin, builds what `main` holds then, and hands off. The deploy's handle stays the same |
| had not ended, and is anything else | ends `Interrupted`: "the agent restarted". A worker's run is waited on inside its call, and that call has already ended |

A run with an overlay is recorded without the overlay's chains, and is never a
deploy, so it is never run again. Cause 4 is gone for deploys. Every other job
ends with an answer instead of being lost.

## 6. Tried means it came to something

The gateway's `Told` and the `attempted` field go. The member decides whether
a commit is due from records that outlive every process involved:

| Origin's `main` is | due? |
|---|---|
| the commit the installed binary was built from (keep's state) | no |
| a commit a deploy is under way for: a build recorded on the agent, or a trial on keep's board | no: that deploy is watched (`pending::decide`) |
| a commit whose build failed, or whose trial was rolled back (the agent's record, keep's board) | no |
| anything else, including a commit whose build was interrupted and not re-run | yes |

The webhook and the once-a-minute watch send the same body, `{repo}`. A deploy
killed partway is either re-run by §5 or retried by the next watch. A gateway
restart no longer retries a commit that really failed. Cause 5 is gone.

## 7. Related: the game runs from the agent's build checkout

keep runs `battlecities` with its working directory set to the agent's
checkout (`keep/keep.yaml`, `cwd: D:/agent/battle-cities`). The game serves its
client files and its site folder from there: the site folder's files take
priority over the compiled-in site (`crates/sim/src/server/web/site.rs`). A
run of another branch on that checkout lays that branch's files over it for
the length of the run (`zend_vfs::sandbox`: preserve → materialize → run →
restore). While a worker tests a branch, the live game serves that branch's
site.

The game gets its own tree, `D:/battle-cities/live`, set to the deployed
commit when keep proves its trial. The build checkout is then only built in.
This is also where the site-only deploy belongs: a push that changes only
`web/`, `crates/sim/website/` and `docs/` moves the live tree to the new
commit and builds nothing, and the game serves the new files on the next
request.

## 8. Landing it

Everything is in candle. It takes effect only once it is deployed: keep on
each machine (`/exit` onto the new `keep.exe`), the new `agent` service, a
motherd deploy (the in-process agent gives up the build checkouts) and the
gateway (`Told` removed).

| Step | Crates | Tests beside the code |
|---|---|---|
| §2 split and drain | `agent`, `host`, configs | an agent file's `priority` is read; an agent asked to install refuses new runs and installs only once its runs end |
| §3 build slot | `agent` | two runs on two checkouts never run at once; a deploy waiting goes ahead of an earlier test; `slots_free` follows the slot |
| §4 handoff | `agent`, `keep`, `zend` | keep's local `/trial` lands on the board under the handle; a deploy run staged hands off once; a refusal is recorded on the run |
| §5 records | `agent` | a run written then read back in a new `Jobs` is listed; an unfinished deploy runs again under its id; an unfinished test ends `Interrupted` |
| §6 due | `zend`, `web` | a failed build, a rolled-back trial and an installed commit are not due; an interrupted one is |
| §7 live tree | `zend`, `keep`, configs | a proven trial moves the live tree; a site-only push builds nothing |

One case is left: deploying the agent itself. The agent's own install drains
(§2) and its queued runs are recorded (§5), so a build waits through it rather
than dying.
