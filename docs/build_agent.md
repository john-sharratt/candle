# Builds that are never cut short by a planned restart

A build is the slow part of every deploy, and today almost anything that
restarts a process on its machine kills it. This document is the design
that stops a planned restart from ever killing a build. A planned restart
waits for the build that is running, holds back the ones queued behind it,
and lets them start once the new process is up. It also runs one build at a
time on each machine, hands a finished build to keep without depending on any
zend member, and counts a commit as tried only once it came to something.

A crash or a reboot still kills the build that is running. Nothing resumes it.
Its run ends `Interrupted`, and the next push watch builds the commit again
(§7).

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
| 2 | **No planned restart looks for running builds.** keep switches, restarts and stops a service without asking whether it is building. A certified install drains the process, releases its substrate and closes its sockets, then asks keep to switch (`cluster/src/upgrade/handoff.rs`), with no step that waits for builds. A standalone agent's install waits `JOBS_WAIT` (60 s), then kills them. | `keep/src/main.rs`, `keep/src/remote.rs`; `agent/src/install.rs` `jobs_ended` |
| 3 | **The handoff to keep lives in one zend member's memory.** A task in the member that took `service_deploy` waits for the build, then asks keep for the trial. If that member restarts first, the build ends and no trial is ever asked for. | `zend/src/mcp/tools/service/deploy.rs` `drive` |
| 4 | **The agent's queue and job table are in memory.** After a restart a queued run is gone, and a handle from before the restart is not found. | `agent/src/jobs.rs` `Jobs::table` |
| 5 | **A started deploy counts as tried.** The gateway records the commit as soon as the member answers that it started a deploy. The once-a-minute watch then sends that commit as `attempted`, and the member does nothing. A killed build is never retried. The record is also in the gateway's memory, so a gateway restart retries even a commit that really failed. | `web/src/hooks/pushes.rs` `Told`, `tell`; `zend/src/mcp/tools/service/on_push.rs` `due` |
| 6 | **Nothing limits a machine to one build at a time.** The only lock is per checkout (a `tokio::sync::Mutex`, `zend-vfs/src/sandbox/mod.rs`), so a candle build and a battle-cities build run side by side. The agent file's `slots: 4` is reported to the gateway but not enforced as a build limit. | `agent/src/jobs.rs` `slots` |

## 2. The build gate

Each machine has one **build gate**: two named mutexes, a turnstile and a
slot. The OS releases a mutex its holder dies holding (it is abandoned, and
the next waiter gets it), so no crash leaves the gate shut. No file records
it. On Windows they are `tokera-build-turnstile` and `tokera-build-slot` in
the namespace keep's services share (§9).

- **A run passes the gate to start.** It takes the turnstile, then the slot,
  then lets the turnstile go at once. It holds the slot until its command has
  ended and its output is staged.
- **A restart shuts the gate.** It takes the turnstile and keeps it. A run
  can then no longer get through, so every run accepted from that moment
  waits in the queue. The restart then waits for the slot, which it gets once
  the run holding it ends. It holds both until it is done.

The turnstile is what lets a restart get in ahead of the queue: without it, a
restart waiting for the slot would compete with every queued run and could
wait for ever.

A Windows mutex belongs to the thread that took it, and the agent runs its
jobs on tokio's pool. So a gate is held by a thread of its own, which takes
and gives back the mutexes when asked (`agent::gate`).

## 3. One build at a time on each machine

Every run takes the slot (§2) after its fetch and its store are ready, and
before its checkout is set aside. Pushes, fetches and listings never take it.
Tests compile as much as builds do, and the timed test gate needs a quiet
machine (`cargo test-server` on BlackWorld), so every run takes the slot.

- A run waiting for the slot is `Phase::Waiting {behind}`, naming the run that
  holds it. A run held back by a shut gate is `Phase::Held {by}`, naming the
  restart.
- The agent decides the order the waiting runs go through the gate: deploys
  (§5) first, then other runs in the order they came. A running build is
  never preempted.
- The agent reports `slots_free` as 1 when the slot is free and nothing waits,
  and 0 otherwise. The gateway already sends a run that names no machine to
  an agent with a free slot (`cluster/src/gateway/agents.rs`), so a worker's
  test goes to another machine while BlackWorld builds. A deploy names its
  machine and waits its turn.
- `slots:` leaves the agent file.

## 4. Who shuts the gate

Every planned restart of a process that runs builds shuts the gate first. It
shuts it before it does anything that stops the process serving, so a member
is never drained while it waits for a build.

| Restart | Who shuts the gate | When it opens |
|---|---|---|
| keep's switch or trial of a service that runs builds (`motherd` on BlackWorld and WhiteWorld, `agent` on AlienWorld) | keep, before it stops the old process | the trial is proven or rolled back, or the switch has started the new process |
| keep's `/restart`, `/stop` of such a service | keep, before it stops it | the service is started again, or for `/stop` when it is started |
| a certified install (`cluster::upgrade`) | the installing process, before it drains | the process exits; the OS releases the gate it held |
| a standalone agent's install | the agent, before it closes its sockets | the process exits |

- keep's file marks the services that run builds: `builds: true` on the
  service in `keep/keep.yaml` and its siblings. A trial or restart of any
  other service never waits. A Battle Cities trial, which spends minutes
  loading its world, runs while a candle build compiles.
- **An install hands its held gate to keep.** It posts the switch with
  `x-tokera-gate-held: 1`, saying it holds the gate and will let go by
  exiting. keep, which would otherwise wait for a gate that the process it
  is about to stop holds, takes the gate as the old process exits, and holds
  it until the new process has started, so no run starts in the gap.
- `JOBS_WAIT` goes. A standalone agent's install is the table's last row.
- **A gate is shut for as long as a build has left to run, and no longer.**
  Whoever shuts it waits for the slot at most the running run's remaining
  timeout. A run that has not ended by then is killed by its own timeout
  anyway. A person who wants the restart now sends `x-tokera-skip-gate: 1`,
  which kills the running build. The cluster's install sends `FORCE_HEADER`,
  which skips keep's refusals (`keep/src/guard.rs`) but not the gate.

**A build cannot hold up the restart that would deploy it.** A deploy's run
gives the slot back once its output is staged, before it asks keep for the
trial (§5). The trial then shuts the gate behind it, and the runs queued
after it wait for the trial.

**While a trial of motherd runs, no build starts on that machine.** The new
process's agent is part of what is being tried. A build started in it would
die if the trial were rolled back, which is exactly the restart this design
exists to keep builds out of. Trials of every other service run alongside
builds.

## 5. The agent hands the build to keep

A `RunJob` that is a deploy says so: `deploy: Option<String>` names the
service. When such a run exits 0, the agent stages the binary, gives back the
slot, reports the end to the gateway (§6), and asks its own machine's keep for
the trial.

- keep's local port (`127.0.0.1:8092`) gets `POST /trial {service, artifact,
  deploy}`, which is what the cluster route `/v1/keep/trial` takes
  (`keep/src/remote.rs`). The trial goes on keep's board under the deploy's
  handle `deploy:<node>:<id>`, which the agent writes from its node and the
  run's id, exactly as `service_deploy` answered. keep's check that the
  artifact was built from a commit on `origin/main` still applies.
- **A trial asked twice is answered with the trial keep has.** `/trial` with a
  deploy handle its board already names, running or ended, starts nothing.
- The run records what came of the handoff (`JobView::tried`: asked, or
  refused with keep's reason), so `job_status` on a refused trial says why.
- `deploy.rs` `drive` and its `tokio::spawn` are removed. `service_deploy`
  submits the build with `deploy` set and answers as it does now. A zend
  restart no longer touches a deploy. Cause 3 is gone.
- For a motherd deploy, `connector_job` runs ahead of the daemon's build on
  the same agent, and the gate's order keeps the connector staged before the
  daemon's trial starts.

A motherd trial restarts the agent that built it. The run's end is already
in the gateway's table, and from then on the deploy is followed on keep's
board (`service::resolve`, `service::trial_in`), as it is today.

## 6. The gateway's table

A run that finds no free slot queues on its agent, never in the gateway
(`cluster/src/gateway/agents.rs`): the agent holds the queue. The gateway
keeps a table of every run it handed out, by `(node, id)`: its request
without the overlay's chains, its machine, its base commit, its phase, and
how it ended once it has. It serves the cluster's reads of runs
(`job_status`, `job_list`, `pending::decide`, §7) without asking the agent. A
log read still goes to the agent.

- **An agent reports its runs on every connect and on every change**: the
  ones queued in it, the one running, and the ones that ended.
- **A report merges into the table; it never replaces it.** A run the report
  names takes the report's state. A run in the table that a report leaves out
  is left as it is, unless the report is the first from a new process of that
  agent:
  - a run the table holds as queued is handed back to the agent and queued
    again, under its own id. The agent's queue was lost with the old process;
  - a run the table holds as running ends `Interrupted`, "the agent
    restarted". A planned restart never leaves one (§4). A crash does.
- Ended runs stay as the record of how they ended (§7), however many agent
  restarts later: kept by age (30 days) and by count (the newest 1,024 a
  machine), not by the agent's `KEPT`.
- A restarted gateway has an empty table. It fills it as the agents
  reconnect and report, and loses only the ended runs the agents no longer
  hold.

A run queued when its agent restarts starts on the new process, under the
same handle, and a caller waiting on it with `job_status` is answered as if
nothing happened. Only a restart of both the agent and the gateway, while a
run was queued, loses that run. A run handed back but overlaid with chains
the new process lacks is refused as `JobError::Missing` is today, and its
caller sends them again.

## 7. Tried means it came to something

The gateway's `Told` and the `attempted` field go. Whether a commit is due is
read from the gateway's table (§6) and keep's board:

| Origin's `main` is | due? |
|---|---|
| the commit the installed binary was built from (keep's state) | no |
| a commit a deploy is under way for: a run in the table, or a trial on keep's board | no: that deploy is watched (`pending::decide`) |
| a commit whose build failed, or whose trial was rolled back | no |
| anything else, including a commit whose run ended `Interrupted` or was lost | yes |

The webhook and the once-a-minute watch send the same body, `{repo}`. A build
killed by a crash is built again by the next watch. A gateway restart forgets
a failed build, so a commit whose build failed before the gateway restarted is
built once more. A rolled-back trial is still on keep's board. Cause 5 is
gone.

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

- **The mutexes' namespace.** A `Global\` name needs `SeCreateGlobalPrivilege`
  outside session 0, and a `Local\` name is per session. keep and the services
  it starts must name the same objects. Confirm the session keep's scheduled
  task runs them in, and choose the prefix from that.
- **Other platforms.** The agent also builds for Linux
  (`cluster/src/upgrade/runtime/fetch.rs`). There the gate is two `flock`s,
  which the kernel releases when the holder dies.

## 10. Landing it

Everything is in candle. It takes effect once deployed: keep on each machine
(`/exit` onto the new `keep.exe`), a motherd deploy, and the gateway. The
motherd deploy that brings it is the last planned restart that can kill a
build: the old process has no gate to shut.

| Step | Crates | Tests beside the code |
|---|---|---|
| §2 gate | `agent` | a run cannot pass a shut gate; a gate whose holder thread is gone is free again; a restart shutting the gate gets the slot ahead of runs queued after it |
| §3 one build | `agent` | two runs on two checkouts never hold the slot at once; a deploy waiting goes ahead of an earlier test; `slots_free` follows the slot |
| §4 restarts | `keep`, `cluster`, `agent` | keep's restart of a `builds: true` service waits for a running run and starts none until the service is back; one of another service does not wait; an install shuts the gate before it drains; `x-tokera-gate-held` makes keep take the gate as the old process exits; `x-tokera-skip-gate` does not wait; `FORCE_HEADER` alone still waits |
| §5 handoff | `agent`, `keep`, `zend` | keep's local `/trial` lands on the board under the handle, and asked again with the handle starts nothing; a deploy run gives back the slot before it asks keep; a refusal is recorded on the run |
| §6 table | `cluster`, `agent` | a report merges, and an ended run it leaves out stays; a new process's first report gets back its queued runs and ends its running one `Interrupted`; a restarted gateway's table is refilled |
| §7 due | `zend`, `web` | a failed build, a rolled-back trial and an installed commit are not due; an interrupted or lost one is |
| §8 live tree | `zend`, `keep`, configs | a proven trial moves the live tree; a site-only push builds nothing |
