# Builds that are never cut short by a planned restart

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This document is the design
that stops a planned restart from killing a build. A planned restart waits
for the build that is running, holds back the ones queued behind it, and lets
them start once the new process is proven. It also runs one build at a time
on each machine, hands a finished build to keep without depending on any zend
member, and counts a commit as tried only once it came to something.

A crash or a reboot still kills the build that is running. Nothing resumes it.
Its run ends `Interrupted`, and its caller is told so. A push-deployed
commit (`battlecities`, `on_push::ON_PUSH`) is built again by the next push
watch (§8). Every other deploy is a person's, and they deploy it again.

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
| 1 | **A run dies with the agent, and the agent runs inside motherd.** The agent holds the job object each command runs in, and closing the agent closes the job, which kills the command. Every motherd trial, restart and rollout closes the agent. | `host/src/cluster/mod.rs` `start_agent`; `zend-vfs/src/kill_tree.rs` `Tree::drop` |
| 2 | **No planned restart looks for running builds.** keep switches, restarts and stops a service without asking whether it is building. A certified install takes its upgrade leases, drains the process, releases its substrate and closes its sockets, then asks keep to switch (`cluster/src/upgrade/runtime/install.rs` `install_tick`, `cluster/src/upgrade/handoff.rs`), with no step that waits for builds. A standalone agent's install waits `JOBS_WAIT` (60 s), then kills them. | `keep/src/main.rs`, `keep/src/remote.rs`; `agent/src/install.rs` `jobs_ended` |
| 3 | **The handoff to keep lives in one zend member's memory.** A task in the member that took `service_deploy` waits for the build, then asks keep for the trial. If that member restarts first, the build ends and no trial is ever asked for. | `zend/src/mcp/tools/service/deploy.rs` `drive` |
| 4 | **The agent's jobs are in memory.** A run waiting for its checkout is a task blocked on the checkout's mutex. After a restart it is gone, and a handle from before the restart is not found. | `agent/src/jobs.rs` `Jobs::table` |
| 5 | **A started deploy counts as tried.** The gateway records the commit as soon as the member answers that it started a deploy. The once-a-minute watch then sends that commit as `attempted`, and the member does nothing. A killed build is never retried. The record is also in the gateway's memory, so a gateway restart retries even a commit that really failed. | `web/src/hooks/pushes.rs` `Told`, `tell`; `zend/src/mcp/tools/service/on_push.rs` `due` |
| 6 | **Nothing limits a machine to one build at a time.** The only lock is per checkout (a `tokio::sync::Mutex`, `zend-vfs/src/sandbox/mod.rs`), so a candle build and a battle-cities build run side by side. The agent file's `slots: 4` is reported to the gateway but not enforced as a build limit. | `agent/src/jobs.rs` `slots` |

## 2. The build gate

Each machine has one **build gate**: two named mutexes, a turnstile and a
slot, `Global\tokera-build-turnstile` and `Global\tokera-build-slot`. keep
runs from a boot task as an S4U logon (`keep/src/owner_access.rs`), so it and
every service it starts are in session 0, where `Global\` names are shared and
need no extra privilege. The OS releases a mutex its holder dies holding (it
is abandoned, and the next waiter gets it), so no crash leaves the gate shut.
No file records it.

- **A run goes through the gate to start.** It takes the turnstile, tries the
  slot without waiting, and lets the turnstile go either way. Holding the slot,
  it starts. Not holding it, it waits for the agent to say the slot came free,
  and tries again. A run never waits for the slot while it holds the turnstile,
  so a restart behind it waits for one build, not two.
- **A restart shuts the gate.** It takes the turnstile and keeps it. No run can
  then get through, so every run accepted from that moment waits. The restart
  waits for the slot, which it gets once the run holding it ends, and holds
  both until it is done.
- **Each holder is a thread of its own.** A Windows mutex belongs to the thread
  that took it, and the thread that holds one can take it again. So the agent's
  runs share one gate thread, and every restart that shuts the gate, in keep or
  in an installing process, takes it on a thread of its own. A restart can never
  pass a gate that a run's thread holds, or the other way round (`agent::gate`).

**How long a restart waits.** Whoever shuts the gate waits for the slot at most
`build_hold` (keep's file and the cluster's upgrade settings; 60 minutes on
BlackWorld). After that the restart goes ahead, and the build is killed with
its process. Its run ends `Interrupted` (§7). A person who wants the restart
now sends `x-tokera-skip-gate: 1`. `FORCE_HEADER`, which the cluster's install
always sends and which skips keep's refusals (`keep/src/guard.rs`), does not
skip the gate.

## 3. One build at a time on each machine

Every run takes the slot (§2) after its fetch and its store are ready, and
before its checkout is set aside. Pushes, fetches and listings never take it.
Tests compile as much as builds do, and the timed test gate needs a quiet
machine (`cargo test-server` on BlackWorld), so every run takes the slot.

- **The agent gets a queue.** Today a waiting run is only a task blocked on its
  checkout's mutex. Now the runs waiting for the gate are in an ordered queue:
  deploys (§6) first, then other runs in the order they came. A running build
  is never preempted.
- A run waiting for the slot is `Phase::Waiting {behind}`, naming the run that
  holds it. A run held back by a shut gate is `Phase::Held {by}`, naming the
  restart.
- The agent reports `slots_free` as 1 when the slot is free and nothing waits,
  and 0 otherwise. The gateway already sends a run that names no machine to
  an agent with a free slot (`cluster/src/gateway/agents.rs`), so a worker's
  test goes to another machine while BlackWorld builds. A deploy names its
  machine and waits its turn.
- `slots:` leaves the agent file.

## 4. Who shuts the gate

Every planned restart of a process that runs builds shuts the gate first,
before it does anything that stops the process serving or that times out, so
a member is never drained while it waits for a build, and no lease runs down
while it waits.

| Restart | Who shuts the gate, and when | When it opens |
|---|---|---|
| keep's trial of a service that runs builds, by hand or from the cluster | keep, before it stops the old process | the trial is proven or rolled back |
| keep's `/accept`, `/restart` or `/stop` of such a service | keep, before it stops it | the new process has started; for `/stop`, when the service is started again |
| a certified install (`cluster::upgrade`) | the installing process, in `install_tick` **before it takes its upgrade leases**, as a step of its own (`waiting for builds`) | it hands the gate to keep (below) |
| a standalone agent's install | the agent, before it closes its sockets | it hands the gate to keep (below) |
| keep starting | keep, before it does anything else, its rollback of an interrupted trial (`roll_back_interrupted_trial`) included | that rollback, or the start of each service, is done |

- keep's file marks the services that run builds: `builds: true` on the
  service in `keep/keep.yaml` and its siblings (`motherd` on BlackWorld and
  WhiteWorld, `agent` on AlienWorld). A trial or restart of any other service
  never waits. A Battle Cities trial, which spends minutes loading its world,
  runs while a candle build compiles.
- **An upgrade lease is taken only once the gate is held.** `install_tick` takes
  `upgrade/<app>` leases of `LEASE_TTL` (`READY_WITHIN` + 5 min, never renewed)
  and then installs. If it waited for a build after taking them, the leases
  could run out mid-wait and let a second member install while the first is
  still to go down. So the gate comes first, and the rollout's step says
  `waiting for builds` while it waits.
- **An installing process hands its gate to keep.** Its handoff is a trial
  (`POST /`, `handoff.rs` `ask_keep`) sent with `x-tokera-gate-held: 1`,
  saying it holds the gate and lets go by exiting. keep, which would otherwise
  wait for a gate that the process it is about to stop holds, shuts the gate on
  its own thread as that process exits (its mutexes abandoned, keep is the
  waiter that gets them). It holds the gate until the trial is proven or rolled
  back, as for any trial of a `builds: true` service.
- **keep does not exit while it holds the gate or runs a trial.** `/exit` is
  refused then, as a trial already is, unless forced. A keep that dies anyway
  frees the gate, so builds can start in a process under trial, and the next
  keep's rollback of that trial kills them. That is a crash, and it is the
  crash's cost.
- **keep waits for a build outside the trial.** A trial of a `builds: true`
  service is `waiting for builds` on the board until keep holds the gate, and
  `running` from then. keep re-checks the service's own refusal, the last-member
  guard and the machine's reservation once it holds the gate, not only before
  the wait.

**While a trial of motherd runs, no build starts on that machine.** The new
process's agent is part of what is being tried. A build started in it would
die if the trial were rolled back, which is the restart this design exists to
keep builds out of. Trials of every other service run alongside builds.

## 5. The daemon and its connector are one build

A motherd deploy today is two runs: `connector_job`, a plain build that stages
`mcp.exe`, queued ahead of the daemon's build that stages `motherd.exe`
(`deploy.rs`). With deploys going first through the gate, the daemon's build
could overtake the connector's, and the trial would start with no connector
staged. A run's `stage` becomes a list, and the daemon's deploy is **one run**
of `cargo build --release -p motherd -p mcp` that stages both, the connector
before the daemon. `connector_job` goes.

## 6. The agent hands the build to keep

A `RunJob` that is a deploy says so: `deploy: Option<String>` names the
service. When such a run exits 0 and has staged its binaries, the agent reports
the end to the gateway (§7) and asks its own machine's keep for the trial at
keep's local port (`127.0.0.1:8092`), with `POST /trial {service, artifact,
deploy}`: what the cluster route `/v1/keep/trial` takes (`keep/src/remote.rs`).
The trial goes on keep's board under the deploy's id, `<node>:<id>`, which the
agent writes from its node and the run's id, as `service_deploy` does today
(`board.rs`). keep's check that the artifact was built from a commit on
`origin/main` still applies. The run records what came of the handoff
(`JobView::tried`: asked, or refused with keep's reason), so `job_status` on a
refused trial says why.

**When the trial is of a `builds: true` service, the build hands keep the
gate.** For motherd, the agent that built it runs inside the process the trial
replaces. The run does not give the slot back. The agent shuts the gate on its
own restart thread, then gives the slot back, so the run's thread lets go of
the slot and the gate stays shut. It posts the trial with `x-tokera-gate-held:
1`, exactly as an install does (§4). No run starts between the build and its
trial. Nothing else can stage over `keep/stage/motherd.exe` meanwhile, and the
trial does not wait for a build that started after it.

**When it is not**, the run gives the slot back as soon as its binaries are
staged, and the next build starts while keep tries this one.

- `deploy.rs` `drive` and its `tokio::spawn` are removed. `service_deploy`
  submits the build with `deploy` set and answers as it does now. A zend
  restart no longer touches a deploy. Cause 3 is gone.
- A motherd trial restarts the agent that built it. The run's end is already in
  the gateway's table, and from then on the deploy is followed on keep's board
  (`service::resolve`, `service::trial_in`), as it is today.
- The agent asks keep once. An agent killed between its report and the ask
  leaves a run that ended with no trial asked: the gateway's table shows it
  staged and not tried, and `job_status` says so.

## 7. The gateway's table

The gateway keeps a table of every run it handed out, by `(node, id)`: its
request, its machine, its base commit, its phase, and how it ended once it has.
It serves the cluster's reads of runs (`job_status`, `job_list`,
`pending::decide`, §8) without asking the agent. A log read still goes to the
agent.

- **An agent process has an instance**: a random id drawn when the process
  starts, sent in every report. A node's persistent id (`agent::member::node_in`)
  says which agent; the instance says which process of it. A reconnect of the
  same process, after a network drop or a gateway's close, is the same
  instance.
- **An agent reports its runs on every connect and on every change**: the ones
  queued in it, the one running, and the ones that ended.
- **A report merges into the table; it never replaces it.** A run the report
  names takes the report's state. A run in the table that a report leaves out is
  left as it is, unless the report is the first from a new instance:
  - a run the table holds as queued, with no overlay, goes back to the agent
    and is queued again, under its own id;
  - a run the table holds as queued with an overlay ends `Interrupted`. Its
    chains were the old process's, and the caller that sent them is a worker's
    call that has already ended;
  - a run the table holds as running ends `Interrupted`, "the agent
    restarted". A planned restart never leaves one (§4). A crash does.
- Every gateway a node is joined to keeps its own table of the runs it handed
  out (`hub.rs` peers). A new instance takes back its queued runs from each
  gateway, by id, so a run comes back from the gateway that handed it out and
  from no other.
- Ended runs stay as the record of how they ended (§8), however many agent
  restarts later: kept by age (30 days) and by count (the newest 1,024 a
  machine), not by the agent's `KEPT`.
- A restarted gateway has an empty table. It fills it as the agents report,
  and loses only the ended runs the agents no longer hold.

A run queued when its agent restarts starts on the new process, under the same
handle, and a caller waiting on it with `job_status` is answered as if nothing
happened. Only a restart of both the agent and the gateway, while a run was
queued, loses that run.

## 8. Tried means it came to something

The gateway's `Told` and the `attempted` field go. Whether a pushed commit is
due is read from the gateway's table (§7) and keep's board:

| Origin's `main` is | due? |
|---|---|
| the commit the installed binary was built from (keep's state) | no |
| a commit a deploy is under way for: a run in the table, or a trial on keep's board | no: that deploy is watched (`pending::decide`) |
| a commit whose build failed (the table), or whose trial was rolled back (keep's board) | no |
| anything else, including a commit whose run ended `Interrupted` or was lost | yes |

keep's board holds only the running trial and the last one, in memory
(`board.rs` `Trials`). It keeps the last 16 ended trials of each service
instead, so a rolled-back commit is still known after later trials. For
`battlecities`, every trial first stops the live game, so a commit that is
not known to have failed is tried again. A keep restart forgets what it held,
and the game is then tried once more on a commit that failed before the
restart. That is the one retry a keep restart costs.

The webhook and the once-a-minute watch send the same body, `{repo}`. A build
killed by a crash is built again by the next watch. A gateway restart forgets
a failed build, so a commit whose build failed before the gateway restarted is
built once more. Cause 5 is gone.

## 9. Related: the game runs from the agent's build checkout

keep runs `battlecities` with its working directory set to the agent's
checkout (`keep/keep.yaml`, `cwd: D:/agent/battle-cities`). The game serves its
client files and its site folder from there: the site folder's files take
priority over the compiled-in site (battle-cities
`crates/sim/src/server/web/site.rs`). A run of another branch on that checkout
lays that branch's files over it for the length of the run. While a worker
tests a branch, the live game serves that branch's site.

The game gets its own tree, `D:/battle-cities/live`, set to the deployed
commit when keep proves its trial. The build checkout is then only built in.
This is also where the site-only deploy belongs: a push that changes only
`web/`, `crates/sim/website/` and `docs/` moves the live tree to the new
commit and builds nothing, and the game serves the new files on the next
request.

## 10. To settle while building it

- **Other platforms.** The agent also builds for Linux
  (`cluster/src/upgrade/runtime/fetch.rs`). There the gate is two `flock`s,
  which the kernel releases when the holder dies.
- **WhiteWorld and AlienWorld.** Their keep and motherd files live outside the
  repository (`C:\prog\keep-local.yaml`, `motherd-local.yaml`,
  `C:\Users\johna\prog\keep-local.yaml`). Each gains `builds: true` and
  `build_hold` as BlackWorld's does.
- **An agent below the floor.** When the floor rises the gateway closes older
  agents (`hub.rs` `set_floor`). A standalone agent's install then waits for its
  build (§4). Whether the gateway still takes its reports and log reads while it
  waits must be checked, and made so if not.

## 11. Landing it

Everything is in candle. It takes effect once deployed: keep on each machine
(`/exit` onto the new `keep.exe`), a motherd deploy, and the gateway. The
motherd deploy that brings it is the last planned restart that can kill a
build: the old process has no gate to shut.

| Step | Crates | Tests beside the code |
|---|---|---|
| §2 gate | `agent` | a run cannot pass a shut gate; a run that finds the slot taken lets the turnstile go; a gate whose holder thread is gone is free again; a restart shutting the gate gets the slot ahead of runs queued after it; a restart and a run on different threads never pass each other's hold |
| §3 queue | `agent` | two runs on two checkouts never hold the slot at once; a deploy waiting goes ahead of an earlier test; `slots_free` follows the slot |
| §4 restarts | `keep`, `cluster`, `agent` | keep's trial, `/accept`, `/restart` and `/stop` of a `builds: true` service wait for a running run and start none until done; another service does not wait; `install_tick` holds the gate before it takes a lease; `x-tokera-gate-held` makes keep take the gate as the old process exits; `x-tokera-skip-gate` does not wait; `FORCE_HEADER` alone still waits; `/exit` is refused while the gate is held; a starting keep shuts the gate before its rollback; the guards are read again once the gate is held |
| §5 one build | `zend`, `agent` | the daemon's deploy is one run that stages the connector, then the daemon |
| §6 handoff | `agent`, `keep`, `zend` | keep's local `/trial` lands on the board under the deploy's id; a motherd deploy keeps the gate shut from its build to its trial; another service's deploy gives back the slot once staged; a refusal is recorded on the run |
| §7 table | `cluster`, `agent` | a report merges, and an ended run it leaves out stays; a reconnect of the same instance changes nothing; a new instance's first report gets back its plain queued runs and ends its overlaid and running ones `Interrupted` |
| §8 due | `zend`, `web`, `keep` | a failed build, a rolled-back trial among the last 16 and an installed commit are not due; an interrupted or lost one is |
| §9 live tree | `zend`, `keep`, configs | a proven trial moves the live tree; a site-only push builds nothing |
