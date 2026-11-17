# Builds that a planned restart never cuts short

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This design makes the planned
path work: a deploy, a certified install, a restart through keep. A planned
restart of a process that runs builds waits for the build that is running,
holds back the runs queued behind it, and hands them to the process that
comes up next. Builds run one at a time on each machine, and trials of other
services run alongside them.

**Out of scope: crashes.** A crash, a reboot, or a trial that is rolled back
fails the build that was running, and loses the runs queued behind it. Its
caller is told the run ended. Nothing here tries to survive that.

`docs/jobs.md` §2 and §6 describe the path as it is built today. Each
changes as the step that touches it lands.

## 1. What went wrong

On 2026-11-17 a push to battle-cities `main` started a deploy on BlackWorld.
The build died partway through, because a `motherd` trial on the same machine
restarted the process the build agent runs in.

| # | Cause | Where |
|---|---|---|
| 1 | **A run dies with the agent, and the agent runs inside motherd.** The agent holds the job object each command runs in, and closing the agent closes the job. Every motherd trial, restart and rollout closes the agent. | `host/src/cluster/mod.rs` `start_agent`; `zend-vfs/src/kill_tree.rs` `Tree::drop` |
| 2 | **No planned restart waits for running builds.** keep switches, restarts and stops a service without asking whether it is building. A certified install takes its upgrade leases, drains, releases its substrate and closes its sockets, then asks keep to switch (`cluster/src/upgrade/runtime/install.rs` `install_tick`, `cluster/src/upgrade/handoff.rs`). A standalone agent's install waits `JOBS_WAIT` (60 s), then kills its jobs. | `keep/src/main.rs`, `keep/src/remote.rs`; `agent/src/install.rs` `jobs_ended` |
| 3 | **The handoff to keep is a separate task that can lose the race.** A task in the zend member that took `service_deploy` polls the build every 25 s, then asks keep for the trial. A restart that waits only for the build to end can stop the process between the build's end and that ask. | `zend/src/mcp/tools/service/deploy.rs` `drive` |
| 4 | **Queued runs are tasks in memory.** A run waiting for its checkout is a task blocked on the checkout's mutex, gone with the process. | `agent/src/jobs.rs`; `zend-vfs/src/sandbox/mod.rs` |
| 5 | **Nothing limits a machine to one build at a time.** The only lock is per checkout, so a candle build and a battle-cities build run side by side. `slots: 4` is reported to the gateway but not enforced. | `agent/src/jobs.rs` `slots` |

## 2. One build at a time

Each machine's builds run in one agent: the one in motherd on BlackWorld and
WhiteWorld, `tokera-agent` on AlienWorld. `Jobs` gets one **build slot** and a
queue in front of it. A run takes the slot after its fetch and its store are
ready, before its checkout is set aside, and gives it back once its command has
ended and its output is staged. Runs go through in the order they came. A
running build is never preempted. Pushes, fetches and listings never take the
slot.

- Tests take the slot too. They compile as much as builds do, and the timed
  test gate needs a quiet machine (`cargo test-server` on BlackWorld).
- A run in the queue is `Phase::Waiting {behind}`, naming the run that holds
  the slot.
- The agent reports `slots_free` as 1 when the slot is free and the queue is
  empty, and 0 otherwise. The gateway already sends a run that names no
  machine to an agent with a free slot (`cluster/src/gateway/agents.rs`), so a
  worker's test goes to another machine while BlackWorld builds. A deploy names
  its machine and waits its turn.
- `slots:` leaves the agent file.

## 3. The hold

`Jobs::hold()` is what every planned restart of the agent's process asks for
first:

1. The queue stops: no run starts from it, and runs that arrive join it.
2. The hold waits for the running run, if there is one, to end and stage its
   output: at most `build_hold` (the agent file; 60 minutes on BlackWorld).
   A run still going then is killed with the process.
3. It parks the queue on the gateway (§4).
4. It answers: the process may now be stopped.

A hold the process asked for itself, for an install, stands until the process
exits, or until the install fails and lets it go. A hold keep asked for through
the route lifts by itself if keep has not stopped the process within five
minutes of its answer. Either way, a hold that lifts takes its parked runs back
and starts the queue again. A second hold
asked while one stands joins it, so a restart that the process asked for itself
and keep's restart of it do not wait for each other.

Who asks:

| Restart | Asked by | When |
|---|---|---|
| a certified install | the installing process, `install_tick` | **before it takes its upgrade leases**: they last `LEASE_TTL` (`READY_WITHIN` + 5 min) and are never renewed, so a wait after taking them could let a second member install while the first is still to go down. The rollout's step reads `waiting for builds` |
| a standalone agent's install | the agent | before it closes its sockets, in place of `JOBS_WAIT` |
| keep's `/restart`, `/stop`, `/accept`, or a trial, by hand or from the cluster | keep, through `POST /v1/agent/hold` on the service's status address (`ServiceCfg::status`) | after its guards pass and before it stops the service |

- **keep asks every service with a status address.** One with no agent answers
  404, and keep goes on at once. A Battle Cities, gateway or connector trial
  never waits for a build, so it runs while a candle build compiles.
- **A forced request asks for no hold.** A person forcing a restart wants it
  now. The cluster's install always sends `FORCE_HEADER`, and has already held
  in `install_tick`.
- keep's hold request is a long wait: it answers once the hold does, within
  `build_hold`. keep's cluster trial route already answers before it switches
  (`keep/src/remote.rs` `trial`), so the hold happens on the trial's own
  thread, and the board shows the trial as running from the request.

## 4. Parking the queue

A process about to stop gives its queue to the gateway, so the runs in it
start on the process that comes up next. When the hold has drained, the agent
sends each gateway it is joined to one message: its queued runs, each with its
id and request, and its recent ended runs (the last `KEPT`), so a handle from
before the restart is still answered after it.

- The gateway holds them for that agent's node until a new process of that
  agent connects. It then hands them over, and the agent queues the runs
  again in their order, under their own ids, and lists the ended ones as
  ended. In between, `job_status` on a parked run answers "parked: its agent
  is restarting".
- A run whose overlay carries chains is parked without them. The new process
  then lacks them and refuses it as `JobError::Missing`, which is what a run
  naming an unknown chain is refused with today. Its caller, a worker's call
  that ended with the old process, has gone, so such a run ends there.
- A hold that lifts without a restart (§3) takes its parked runs back from the
  gateway, the same way.
- A gateway that restarts while it holds a park loses it. That is the
  everything-restarts-at-once case, out of scope.

## 5. The agent hands the build to keep

A `RunJob` that is a deploy says so: `deploy: Option<String>` names the
service. When such a run has exited 0 and staged its binary, the agent itself
asks its machine's keep for the trial before it gives back the slot. keep's
local port (`127.0.0.1:8092`) gets `POST /trial {service, artifact, deploy}`,
which is what the cluster route `/v1/keep/trial` takes (`keep/src/remote.rs`).
The trial goes on keep's board under the deploy's id, `<node>:<id>` (`board.rs`),
which the agent writes from its node and the run's id, as `service_deploy` does
today. The run records what came of it (`JobView::tried`: asked, or refused
with keep's reason), so `job_status` on a refused trial says why.

- The build is handed off before the slot is free, so a hold that waits for the
  run waits for the handoff too. Cause 3 is gone. keep's `/trial` answers as
  soon as the trial is on its board, before it switches, so a trial that then
  asks this same process for a hold does not wait on the run that asked for it.
- `deploy.rs` `drive` and its `tokio::spawn` are removed. `service_deploy`
  submits the build with `deploy` set and answers as it does now.
- **A motherd deploy.** The trial keep then starts asks motherd for a hold. If
  a queued run took the slot in the moment between the handoff and keep's hold,
  the trial waits for that run: a delay, never a kill. From then on the deploy
  is followed on keep's board (`service::resolve`, `service::trial_in`), as it
  is today.
- The daemon's deploy keeps `connector_job` ahead of it on the same agent. With
  the queue in the order runs came, the connector is staged before the daemon's
  trial starts, as today.

## 6. Unchanged

- **When a pushed commit counts as tried** (`web/src/hooks/pushes.rs` `Told`,
  `on_push::due`) is left as it is. It was wrong only for a build that a restart
  killed, and a planned restart no longer kills one.
- **The game's working tree.** keep runs `battlecities` from the agent's build
  checkout (`keep/keep.yaml`, `cwd: D:/agent/battle-cities`), so a worker's run
  of another branch there briefly serves that branch's site. That is its own
  change, with the site-only deploy rule, not this one.

## 7. Landing it

Everything is in candle. It takes effect once deployed: keep on each machine
(`/exit` onto the new `keep.exe`), a motherd deploy, and the gateway. The motherd
deploy that brings it is the last planned restart that can kill a build: the old
process has no hold to ask for.

| Step | Crates | Tests beside the code |
|---|---|---|
| §2 slot | `agent` | two runs on two checkouts never run at once; runs start in the order they came; `slots_free` follows the slot and the queue |
| §3 hold | `agent`, `cluster`, `keep` | a hold stops the queue and answers once the running run has staged; a run still going at `build_hold` does not stop the answer; a hold lifts by itself and the queue starts again; a second hold joins the first; `install_tick` holds before it takes a lease; keep asks for a hold before it stops a service, goes on at a 404, and asks none when forced |
| §4 park | `agent`, `cluster` | a hold parks the queue and the recent ended runs; a new process of the agent gets them back in order under their ids; a lifted hold takes them back; a parked run's status says so |
| §5 handoff | `agent`, `keep`, `zend` | keep's local `/trial` lands on the board under the deploy's id; a deploy run asks keep before it frees the slot; a refusal is recorded on the run |
