# Builds that a planned restart never cuts short

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This design makes the planned
path work: a deploy, a certified install, a restart through keep. A planned
restart of a process that runs builds waits for the deploy build that is
running, sets other runs aside to run again afterwards, and hands the queue
to the process that comes up next. Builds run one at a time on each machine,
and trials of other services run alongside them.

**Out of scope: crashes.** A crash, a reboot, a forced restart, or keep
dying while it holds a queue (§5) fails the build that was running and loses
the runs queued behind it. Nothing here tries to survive that.

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
ended, its output is staged and, for a deploy, keep has been asked for the
trial (§6). Runs go through in the order they came. Pushes, fetches and
listings never take the slot.

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

1. **The queue stops.** No run starts from it, and runs that arrive join it.
2. **The running run is dealt with by what it is.**
   - A deploy build is waited for until it has staged and asked keep for its
     trial, at most `build_hold` (the agent file; 60 minutes). A deploy still
     going then is killed with the process.
   - Any other run, a test or a build that is no deploy, is stopped at once and
     put back at the head of the queue, under its own id. It runs again from
     its start on the next process. Its log says where it was cut and where it
     started again. So a restart waits at most for one deploy build, never for a
     long test.
3. **The queue is parked on keep** (§5).
4. **It answers**: the process may now be stopped.

A hold the process asked for itself, for an install, stands until the process
exits, or until the install fails and lets it go. A hold keep asked for lifts
by itself if keep has not stopped the process within five minutes of its
answer. A hold that lifts takes its parked runs back from keep and starts the
queue again. A second hold asked while one stands joins it.

Who asks:

| Restart | Asked by | When |
|---|---|---|
| a certified install | the installing process, `install_tick` | **before it takes its upgrade leases**: they last `LEASE_TTL` (`READY_WITHIN` + 5 min) and are never renewed, so a wait after taking them could let a second member install while the first is still to go down. The rollout's step reads `waiting for builds` |
| a standalone agent's install | the agent | before it closes its sockets, in place of `JOBS_WAIT` |
| keep's `/restart`, `/stop`, `/accept`, or a trial, by hand or from the cluster | keep, through `POST /v1/agent/hold` on the service's `agent:` address (§4) | after its guards pass and before it stops the service |

- **A forced request asks for no hold.** A person forcing a restart wants it
  now. The cluster's install always sends `FORCE_HEADER`, and has already held
  in `install_tick`.
- keep's hold request is a long wait: it answers once the hold does, within
  `build_hold`. keep's cluster trial route already answers before it switches
  (`keep/src/remote.rs` `trial`), so the hold happens on the trial's own
  thread, and the board shows the trial as running from the request.
- A Battle Cities, gateway or connector trial asks no hold (§4), so it runs
  while a candle build compiles.

## 4. keep's `agent:` address

keep asks for a hold only of a service whose entry in its file names the
address of the agent in it: `agent: "192.168.1.2:8081"` on BlackWorld's
`motherd` (its plain entrance, `motherd/motherd.yaml` `host:`/`port:`), and the
standalone agent's on AlienWorld. The route answers only a caller on the
machine itself, as keep's own port takes only loopback. It is a field of its
own. `status:`, which arms keep's last-member guard
(`keep/src/guard.rs`), is left as it is on every service, so nothing that guard
refuses or allows changes.

## 5. The queue waits on keep

The process about to stop gives its queue to its machine's keep, which
restarts it and outlives it. When the hold has drained, the agent posts keep's
local port `POST /park {node, runs}`: its queued runs in order, each with its
id and request, and its recent ended runs (the last `KEPT`), so a handle from
before the restart is answered after it.

- keep holds them in memory by node. The agent's next process, once its
  queue starts (§6), asks `POST /unpark {node}`. keep answers them and lets
  them go. The agent queues the runs again in their order, under their ids,
  and lists the ended ones as ended. A hold that lifts without a restart takes
  them back the same way.
- keep refuses `/exit` while it holds parked runs, unless forced.
- While the agent is down, `job_status` on one of its runs answers that its
  agent is restarting and to ask again, where it now says the run is not
  found.
- A run whose overlay carries chains is parked without them. The new process
  lacks them and refuses it as `JobError::Missing`, as a run naming an unknown
  chain is refused today. Its caller, a worker's call, ended with the old
  process, so such a run ends there.

## 6. A new process starts its queue once it is serving

A process's agent starts no run until the process has printed the line keep
proves a trial on, `INFO serving` (`cluster::upgrade::switch::PROOF`). keep
proves the trial on that line, so a build never starts in a process that keep
could still roll back. Until then the agent takes runs into its queue, and
takes its parked runs back from keep when the line is printed.

## 7. The agent hands the build to keep

A `RunJob` that is a deploy says so: `deploy: Option<String>` names the
service. When such a run has exited 0 and staged its binary, the agent itself
asks its machine's keep for the trial before it gives back the slot. keep's
local port gets `POST /trial {service, artifact, deploy}`, which is what the
cluster route `/v1/keep/trial` takes (`keep/src/remote.rs`). The trial goes on
keep's board under the deploy's id, `<node>:<id>` (`board.rs`), which the agent
writes from its node and the run's id, as `service_deploy` does today. The run
records what came of it (`JobView::tried`: asked, or refused with keep's
reason), so `job_status` on a refused trial says why.

- The build is handed off before the slot is free, so a hold that waits for a
  deploy waits for its handoff too. Cause 3 is gone. keep's `/trial` answers
  once the trial is on its board, before it switches, so a trial that then asks
  this same process for a hold does not wait on the run that asked for it.
- `deploy.rs` `drive` and its `tokio::spawn` are removed. `service_deploy`
  submits the build with `deploy` set and answers as it does now.
- **A motherd deploy.** The trial keep starts asks motherd for a hold. A queued
  run that took the slot in the moment between the handoff and that hold is not
  a deploy, so it is set aside (§3) and the trial goes on. From then on the
  deploy is followed on keep's board (`service::resolve`, `service::trial_in`),
  as it is today.
- The daemon's deploy keeps `connector_job` ahead of it on the same agent. With
  the queue in the order runs came, the connector is staged before the daemon's
  trial starts, as today.

## 8. Unchanged

- **When a pushed commit counts as tried** (`web/src/hooks/pushes.rs` `Told`,
  `on_push::due`) is left as it is. It was wrong only for a build that a restart
  killed, and a planned restart no longer kills one.
- **The game's working tree.** keep runs `battlecities` from the agent's build
  checkout (`keep/keep.yaml`, `cwd: D:/agent/battle-cities`), so a worker's run
  of another branch there briefly serves that branch's site. That is its own
  change, with the site-only deploy rule, not this one.

## 9. Landing it

Everything is in candle. The tests sit beside the code:

| Step | Crates | Tests |
|---|---|---|
| §2 slot | `agent` | two runs on two checkouts never run at once; runs start in the order they came; `slots_free` follows the slot and the queue |
| §3 hold | `agent`, `cluster`, `keep` | a hold stops the queue and answers once a running deploy has handed off; a running test is stopped and put back at the head of the queue under its id; a deploy still going at `build_hold` does not stop the answer; a hold lifts by itself and the queue starts again; a second hold joins the first; `install_tick` holds before it takes a lease; keep asks for a hold before it stops a service with an `agent:` address, asks none of one without, and none when forced |
| §5 park | `agent`, `keep` | a hold parks the queue and the recent ended runs on keep; the next process takes them back in order under their ids; a lifted hold takes them back; keep refuses `/exit` while it holds a park |
| §6 serving | `agent`, `host` | no run starts before the process prints `PROOF`; parked runs are taken back once it has |
| §7 handoff | `agent`, `keep`, `zend` | keep's local `/trial` lands on the board under the deploy's id; a deploy run asks keep before it frees the slot; a refusal is recorded on the run |

**Rolling it out**, in this order:

1. **keep on each machine first**: `/exit` onto the new `keep.exe`. keep's
   services keep running and the new keep adopts them. Its file gains `agent:`
   on the services that run builds:
   - BlackWorld: `keep/keep.yaml`, `motherd`, in the repository;
   - WhiteWorld: `C:\prog\keep-local.yaml`, `motherd`, edited on the machine;
   - AlienWorld: `C:\Users\johna\prog\keep-local.yaml`, the agent, edited there
     by the session on that machine.
2. **The agent files** gain `build_hold` and lose `slots`: `motherd/agent.yaml`
   (BlackWorld), `C:\prog\agent-local.yaml` (WhiteWorld),
   `motherd/agent.laptop.yaml` (AlienWorld).
3. **motherd on BlackWorld, while its agent is idle.** The process that brings
   this is the last one with no hold, so it kills whatever is running when it
   goes. Check `job_list` on BlackWorld's agent first, and deploy when it shows
   nothing running. The other members take it through certification, and their
   installs hold from then on.
4. **The standalone agent on AlienWorld** takes the new build when the floor
   rises, as it does today, and holds from then on.
