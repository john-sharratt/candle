# Builds that a planned restart never cuts short

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This design makes the planned
path work: a deploy, a certified install, a restart through keep or through
the cluster's own commands. A planned restart of a process that runs builds
waits for the deploy build that is running, sets other runs aside to run
again afterwards, and hands the queue to the process that comes up next.
Builds run one at a time on each machine, and trials of other services run
alongside them.

**Out of scope: crashes.** A crash, a reboot, a forced restart, or keep
dying while it holds a queue (§6) fails the build that was running and loses
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
| 2 | **No planned restart waits for running builds.** keep switches, restarts and stops a service without asking whether it is building. A certified install takes its upgrade leases, drains, releases its substrate and closes its sockets, then asks keep to switch (`cluster/src/upgrade/runtime/install.rs` `install_tick`, `cluster/src/upgrade/handoff.rs`). The cluster's `restart` and `shutdown` commands and npcd's relaunch go through the host's stopping path (`host/src/commands/exec.rs`). A standalone agent's install waits `JOBS_WAIT` (60 s), then kills its jobs. | `keep/src/main.rs`, `keep/src/remote.rs`; `agent/src/install.rs` `jobs_ended` |
| 3 | **The handoff to keep is a separate task.** A task in the zend member that took `service_deploy` long-polls the build, then asks keep for the trial. A restart that waits only for the build to end can stop the process between the build's end and that ask. | `zend/src/mcp/tools/service/deploy.rs` `drive` |
| 4 | **Queued runs are tasks in memory.** A run waiting for its checkout is a task blocked on the checkout's mutex, gone with the process. | `agent/src/jobs.rs`; `zend-vfs/src/sandbox/mod.rs` |
| 5 | **Nothing limits a machine to one build at a time.** The only lock is per checkout, so a candle build and a battle-cities build run side by side. `slots: 4` is reported to the gateway but not enforced. | `agent/src/jobs.rs` `slots` |

## 2. One build at a time

Each machine's builds run in one agent: the one in motherd on BlackWorld and
WhiteWorld, `tokera-agent` on AlienWorld. `Jobs` gets one **build slot** and a
queue in front of it. A run takes the slot after its fetch and its store are
ready, before its checkout is set aside, and gives it back once its command has
ended, its output is staged and, for a deploy, keep has been asked for the
trial (§7). Runs go through in the order they came. Pushes, fetches and
listings never take the slot.

- Tests take the slot too. They compile as much as builds do, and the timed
  test gate needs a quiet machine (`cargo test-server` on BlackWorld).
- A run in the queue is `Phase::Waiting {behind}`, naming the run that holds
  the slot.
- The agent reports `slots_free` as 1 when the slot is free, the queue is
  empty and no hold stands, and 0 otherwise. The gateway already sends a run
  that names no machine to an agent with a free slot
  (`cluster/src/gateway/agents.rs`), so a worker's test goes to another machine
  while BlackWorld builds or holds. A deploy names its machine and waits its
  turn.
- `slots:` leaves the agent file.

## 3. The hold

`Jobs::hold()` is what every planned restart of the agent's process asks for
first. It answers at once with a handle, and the handle says when the hold has
**drained**:

1. **The queue stops.** No run starts from it.
2. **The running run is dealt with by what it is.**
   - A deploy build is waited for until it has staged and asked keep for its
     trial, at most `build_hold_mins` (the agent file; 20 minutes, to be set
     from a measured motherd release build on BlackWorld, §10). A deploy still
     going then is killed with the process.
   - Any other run, a test or a build that is no deploy, is stopped at once and
     put back at the head of the queue, under its own id. It runs again from
     its start on the next process, with a new log. One whose command has
     already ended and is only staging is waited out. A restart waits at most
     for one deploy build, never for a long test.
3. **The queue is parked on keep** (§6).
4. **Drained**: the process may now be stopped.

**A process that is unwell is not waited for.** When the hold is asked, and
again each time it checks the running deploy, it reads the process's own
readiness, what its `/v1/status` reports: every tenant `ready`, and no device
failing (`host::cluster` `member_state`). A process that is not ready drains at
once: the deploy is killed with it, like any other run. A restart of a sick
motherd, the restart that is usually urgent, goes ahead without waiting; only a
healthy process makes a restart wait, and while it waits it keeps serving.

**A wait says what it is for and how long it should take.** While a hold waits
on a deploy, it reports the run, its service and its estimated time left: the
time the same command took the last time it succeeded on this agent, less what
it has run so far (the ended runs the agent keeps, which a park carries across
a restart, §6). Every place that waits shows that line: `install_tick`'s step
(`waiting for builds: battlecities deploy, about 1 min left`), keep's state for
the request it holds, and the stopping path's job step. A command with no
earlier success says `time left unknown`.

Holds join: a second hold asked while one stands shares it, so the paths below
that meet in one restart never wait for each other. A hold is let go by the
one that asked for it, if the restart it was for does not happen; it then takes
its parked runs back from keep and starts the queue again.

## 4. Who asks

Every planned restart asks in one place, immediately before the step that
stops the process serving:

| Restart | Where it holds |
|---|---|
| a certified install | `install_tick` (`cluster::upgrade::runtime::builds`), once it has found its target and that it is not stranded, and before it takes its leases. The tick **does not wait**: it asks for the hold, says `waiting for builds: …` as its step, and returns, so the upgrader's other ticks (announcing, certifying, fetching) carry on. A later tick that finds the hold drained goes on to the leases, and the hold stands through any wait for them, so the queue is held and a run set aside once, not on every tick. A tick that finds the target gone or stranded, or an install that does not start, lets the hold go. The leases (`LEASE_TTL`, never renewed) are therefore only ever taken with nothing left to wait for |
| a standalone agent's install | the same `install_tick`: the agent runs the cluster's upgrader. `JOBS_WAIT` goes |
| the cluster's `restart` and `shutdown`, and npcd's relaunch | the host's stopping path (`host/src/run.rs` `leave`), which holds and waits for it to drain before it drains the tenants, showing `Waiting for builds` as its step. An install that already holds is joined, not waited on twice. A Ctrl-C at the console waits for nothing |
| keep's `/restart`, `/stop`, `/accept`, a trial by hand, a trial from the cluster | keep's supervisor thread, in `restart`, `hold`, `accept_build` and `apply` (`keep/src/main.rs`, `keep/src/hold.rs`), immediately before each calls `stop`, through `POST /v1/agent/hold` on the service's `agent:` address (§5) |

- **keep holds on the supervisor thread, at the moment it stops the process.**
  A request waiting behind another on that thread, a `/restart` behind a
  10-minute trial, asks only once its turn comes, so no hold stands idle while
  the thread is busy.
- **A forced request asks for no hold.** The force is carried on the request
  to the supervisor. A person forcing a restart wants it now. The cluster's
  install always sends `FORCE_HEADER`, and has already held in `install_tick`.
- **keep's cluster answers do not wait out a hold.** Its trial route answers
  once the trial is on the board, as it does today. Its service route answers
  with the request's end when that comes within 25 s, and otherwise `202`,
  accepted, the request going on and its end read from keep's state
  (`keep/src/remote.rs`). The gateway's request timeout is never reached.
- A Battle Cities, gateway or connector restart asks no hold (§5), so it runs
  while a candle build compiles.

## 5. keep's `agent:` address

keep asks for a hold only of a service whose entry in its file names the
address of the agent in it: `agent: 192.168.1.2:8081` on BlackWorld's
`motherd` (its plain entrance, `motherd/motherd.yaml` `host:`/`port:`), and
`192.168.1.3:8081` on WhiteWorld's. It is a field of its own. `status:`, which
arms keep's last-member guard and a service's own refusal (`keep/src/guard.rs`),
is left as it is on every service, so nothing those refuse or allow changes.
AlienWorld's standalone agent serves no HTTP port, so keep asks it for nothing:
its builds are held by its own installs, through `install_tick`, and a restart
of it by hand stops its build.

`/v1/agent/hold` is a route on the process's own port (`host/src/agent_hold.rs`),
not on the cluster socket the agent's job routes are served over
(`agent/src/member.rs`). It answers only a caller on the machine itself —
loopback, or the address the process listens on — as keep's own port takes
only loopback. It answers once the hold has drained, and the hold it opens lifts
by itself five minutes later if the process is still running.

## 6. The queue waits on keep

The process about to stop gives its queue to its machine's keep, which
restarts it and outlives it. When the hold drains, the agent posts keep's local
port `POST /park {node, runs}`: its queued runs in order, each with its id and
request, and its recent ended runs (the last `KEPT`), so a handle from before
the restart is answered after it.

- **Runs that arrive after the park go to the park.** The process stays joined
  until it stops: through an install's drain, or until keep's switch stops it.
  A run that arrives then is appended to the park on keep, not to a queue that
  dies with the process. The agent keeps a copy of what it parked, and lists
  every parked run as `Waiting` while it lives, so `service_deploy`'s check for
  a deploy already under way (`pending::decide`, which reads the agent's jobs)
  sees it, and a second push does not start a duplicate.
- keep holds the park in memory by node. The agent's next process, once its
  queue starts (§8), asks `POST /unpark {node}`. keep answers and lets the park
  go. The agent queues the runs again in their order under their ids, ignoring
  an id it already holds, and lists the ended ones as ended. A hold let go
  without a restart takes its park back the same way.
- keep refuses `/exit` while it holds a park, unless forced.
- While the agent is down, `job_status` on one of its runs answers that its
  agent is restarting and to ask again, where it now says the run is not found.
- A run whose overlay carries chains is parked without them. The new process
  lacks them and refuses it as `JobError::Missing`, as a run naming an unknown
  chain is refused today. Its caller, a worker's call, ended with the old
  process, so such a run ends there.

## 7. The agent hands the build to keep

A `RunJob` that is a deploy says so: `deploy: Option<String>` names the
service. When such a run has exited 0 and staged its binary, the agent itself
asks its machine's keep for the trial before it gives back the slot: `POST
/trial {service, artifact, deploy}` on keep's local port, what the cluster
route `/v1/keep/trial` takes (`keep/src/remote.rs`). Like that route, and unlike
keep's `POST /` switch (`dispatch`, which waits out the switch), it puts the
trial on the board and answers at once, and the switch runs on the supervisor.
A trial that then asks this same process for a hold does not wait on the run
that asked for it.

- The trial goes on keep's board under the deploy's id, `<node>:<id>`
  (`board.rs`), which the agent writes from its node and the run's id, as
  `service_deploy` does today. The run records what came of it
  (`JobView::tried`: asked, or refused with keep's reason), so `job_status` on a
  refused trial says why.
- The build is handed off before the slot is free, so a hold that waits for a
  deploy waits for its handoff too. Cause 3 is gone.
- `deploy.rs` `drive` and its `tokio::spawn` are removed. `service_deploy`
  submits the build with `deploy` set and answers as it does now.
- **A motherd deploy.** The trial keep starts asks motherd for a hold. A queued
  run that took the slot in the moment between the handoff and that hold is not
  a deploy, so the hold stops it before it can stage anything (§3), and the
  trial goes on. That includes the `connector_job` of a later deploy: stopped
  minutes before it could overwrite `keep/stage/mcp.exe`. From then on the
  deploy is followed on keep's board (`service::resolve`, `service::trial_in`),
  as it is today.
- The daemon's deploy keeps `connector_job` ahead of it on the same agent. With
  the queue in the order runs came, the connector is staged before the daemon's
  trial starts, as today.

## 8. A new process starts its queue once it is serving

A process's agent starts no run until the process has printed the line keep
proves a trial on, `INFO serving` (`cluster::upgrade::switch::PROOF`). keep
proves the trial on that line, so a build never starts in a process that keep
could still roll back. Until then the agent takes runs into its queue, and it
takes its park back from keep when the line is printed.

## 9. Unchanged

- **When a pushed commit counts as tried** (`web/src/hooks/pushes.rs` `Told`,
  `on_push::due`) is left as it is. It was wrong only for a build that a restart
  killed, and a planned restart no longer kills one.
- **The game's working tree.** keep runs `battlecities` from the agent's build
  checkout (`keep/keep.yaml`, `cwd: D:/agent/battle-cities`), so a worker's run
  of another branch there briefly serves that branch's site. That is its own
  change, with the site-only deploy rule, not this one.

## 10. Landing it

Everything is in candle. The tests sit beside the code:

| Step | Crates | Tests |
|---|---|---|
| §2 slot | `agent` | two runs on two checkouts never run at once; runs start in the order they came; `slots_free` follows the slot, the queue and a hold |
| §3 hold | `agent`, `host` | a hold stops the queue and drains once a running deploy has handed off; a running test is stopped and put back at the head of the queue under its id; a deploy still going at `build_hold` does not stop the drain; a process that is not ready drains at once, and one that falls ill while a hold waits drains then; the wait reports the run and its time left from the command's last success, or unknown; a second hold shares the first; a hold let go takes its park back and starts the queue |
| §4 who asks | `cluster`, `host`, `keep` | `install_tick` asks and returns without waiting, takes the leases only on a tick that finds the hold drained, and lets it go when it cannot take them or loses its target; the stopping path holds before it drains the tenants and joins an install's hold; keep's supervisor asks immediately before `stop` in each of its four paths, asks none of a service without an `agent:` address and none when forced; keep's remote routes answer before the hold drains |
| §6 park | `agent`, `keep` | a drained hold parks the queue and the recent ended runs on keep; a run arriving after the park is appended to it and listed as waiting; the next process takes the park back in order under its ids and ignores an id it holds; keep refuses `/exit` while it holds a park |
| §7 handoff | `agent`, `keep`, `zend` | keep's local `/trial` puts the trial on the board and answers before it switches; a deploy run asks keep before it frees the slot; a refusal is recorded on the run |
| §8 serving | `agent`, `host` | no run starts before the process prints `PROOF`; the park is taken back once it has |

**Rolling it out**, in this order:

1. **keep on each machine first**: `/exit` onto the new `keep.exe`. keep's
   services keep running and the new keep adopts them. Its file gains `agent:`
   on the services that run builds:
   - BlackWorld: `keep/keep.yaml`, `motherd`, in the repository;
   - WhiteWorld: `C:\prog\keep-local.yaml`, `motherd` (`192.168.1.3:8081`),
     edited on the machine.
2. **The agent files** lose `slots` **before** the new build starts on each
   machine: an agent file is read with unknown keys refused, so a `slots:` left
   in one stops the new build's start, and one that names `build_hold_mins`
   stops the old build's. Each leaves `build_hold_mins` out, taking the default,
   until the old build is gone everywhere. `motherd/agent.yaml` (BlackWorld) and
   `motherd/agent.laptop.yaml` (AlienWorld) are in the repository;
   `C:\prog\agent-local.yaml` (WhiteWorld) is edited on the machine, and
   AlienWorld's checkout pulls its file. The default, `BUILD_HOLD_MINS`, is set
   from a measured
   `cargo build --release -p motherd -p mcp` on BlackWorld's agent, from a warm
   cache: the longest normal deploy build plus a margin, 20 minutes unless the
   measurement says otherwise.
3. **motherd on BlackWorld, while its agent is idle.** The process that brings
   this is the last one with no hold, so it kills whatever is running when it
   goes. Check `job_list` on BlackWorld's agent first, and deploy when it shows
   nothing running. The other members take it through certification, and their
   installs hold from then on.
4. **The standalone agent on AlienWorld** takes the new build when the floor
   rises, as it does today, and holds from then on.
