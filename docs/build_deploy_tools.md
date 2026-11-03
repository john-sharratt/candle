# Build and deploy tools

The tools an Admin — Claude through the MCP connector, or one of Zen's own
workers — builds, tests and deploys with. Two families, named for what they
act on (`zend/src/mcp/tools/mod.rs`):

| Tool | Does | Answers within |
|---|---|---|
| `repo_job_start` | run a program (`cargo build`, `cargo test`, `cargo clippy`) on a build agent, against a branch on origin | 500 ms p95 (target) |
| `repo_job_status` | where a job is: machine, phase, elapsed, lines, last line, exit code, staged build; waits up to `wait_s` for it to end | its wait |
| `repo_job_log` | read a job's log: `summary`, `tail`, `lines`, `grep` | 1 s |
| `repo_job_list` | every job on every agent, or one machine's, newest first | 1 s |
| `repo_job_cancel` | kill a job, at whatever phase | 1 s |
| `service_deploy` | build a service from origin/main on one machine and try it there through keep | 500 ms p95 (target) |
| `service_status` | every machine's services and trials; or one deploy's phase, waiting up to `wait_s` for its end | its wait |
| `service_log` | read a service's trial log on a machine, in the same four modes | 1 s |
| `service_cancel` | stop a deploy: its build, or its trial (rolled back) | 1 s |
| `service_control` | restart, stop or start a service | — |

They replace `repo_job_start`'s old blocking start, `repo_job_log`'s byte pages,
`service_trial` (a trial of an artifact staged by hand: `service_deploy` builds
and stages it) and the old `service_status`.

## 1. Handles

**A job is named by one handle: `<node>:<id>`** — the agent's 16-hex-digit
node and the job's id, as `76c3a0f4d39bd9db:XeV_I_QPzpg`. Every answer that
names a job names it so, and every tool that reads one takes it as `job`. One
opaque string a model copies is a string it cannot pair wrongly; a node and an
id given apart were.

**A deploy is named by its build job's handle.** The build is a job like any
other, so `repo_job_log` reads a deploy's build log by the same handle. keep
records the handle on the trial it starts for the deploy
(`keep/src/board.rs`), so any member — whichever one a later call lands on —
answers a deploy's state from the agent and from keep, and from nothing held in
one process's memory.

## 2. Starting at once

A start's answer never waits on work that can take seconds.

- **The agent accepts a run before it fetches** (`agent/src/jobs.rs`). The
  request is checked — the repository is held here, the branch parses, a stage
  path stays in the checkout — the job is registered under its own id, and the
  answer goes back. Fetching origin, building the store and waiting for the
  checkout are phases of the job's own task.
- **The gateway never holds a job back** (`cluster/src/gateway/agents.rs`).
  A busy lineage holder spills to the best free agent at once, and with none
  free the job queues on its holder or the best agent — the agent queues it on
  its checkout, one run at a time.
- **`service_deploy` answers once its build job is accepted.** A driver task in
  the member that took the call waits for the build and asks keep for the
  trial; the trial is keep's from there.

A job's **phase** (`agent::wire::Phase`) is finer than its status:

| Phase | Status | |
|---|---|---|
| `fetching` | queued | fetching origin; no log yet |
| `waiting {behind}` | queued | the checkout is held by the runs `behind` |
| `running` | running | its command runs |
| `staging` | exited 0 | copying what it built to the stage folder |
| `ended` | any end | the status says how |

A fetch that fails is a failed job, not a refused start: it was accepted.

## 3. Reading a log without filling a context

A log is read **where it is written** (`cluster/src/log_read.rs`) — the agent
reads a job's, keep a trial's — and only the answer crosses the cluster. Every
line is numbered from 1 across the whole log, so a line a search found is the
line a range read returns.

| Mode | Takes | Answers |
|---|---|---|
| `summary` (the default) | — | cargo's `test result:` lines, failures, errors, counts, the last 15 lines |
| `tail` | `lines` (1–200, 40 by default) | the last lines |
| `lines` | `from`, `count` (1–200) | a range, and the line to read on from |
| `grep` | `pattern`, `context` (0–5), `from` | the lines holding it, ignoring case, with context; past 200 hits, how many more and where to go on |

Each answer holds at most 200 lines of at most 400 characters. A summary is
what a caller reads first: it says whether the run passed and where the
failure is, in a few hundred bytes, and `lines` around a numbered failure reads
the rest.

## 4. Deploys

`service_deploy {host, service}`:

1. Refuse what keep would refuse, before anything runs: the service is one keep
   runs (`motherd`, `web`, `mcp`), and the machine has an agent.
2. For a worker of Zen's: refuse a `motherd` trial on the machine it runs on (the
   trial stops the process the worker lives in) and one that takes an app's
   only ready member down (that is the user's decision) —
   `zend/src/mcp/tools/service/guard.rs`.
3. Start `cargo build --release -p <service>` on that machine's agent, on
   `main`, staging `target/release/<service>.exe`. Answer the deploy's handle.
4. The driver waits for the build. Exited 0 and staged: it posts the trial to
   that machine's keep with the deploy's handle. Anything else ends the deploy
   there; its status says why.
5. keep runs the trial: the build proves itself by printing `INFO serving`
   within ten minutes, the window restarting at each sync line, or it is
   rolled back.

A deploy's state, as `service_status {deploy}` reads it:

| State | Read from |
|---|---|
| building — the job's phase, lines, last line | the agent |
| build failed | the agent: the job did not exit 0 |
| built, trial not started | the agent says staged; keep's board names no trial of it. Past a minute, the driver is gone — its member restarted — and the deploy is run again |
| trial running — elapsed of the window, the trial log's last line | keep's board: `running.deploy` is the handle |
| proven / rolled back / cancelled — keep's account | keep's board: `last.deploy` is the handle |

`service_cancel {deploy}` cancels the build while it runs, or asks keep to stop
the trial, which rolls it back and records the end as cancelled. Between the two
— a build just ended, its trial not yet posted — there is nothing to stop yet,
and the answer says so.

## 5. Stencils: constrained, not cornered

Zen's model calls these tools under a stencil compiled from each tool's schema
(`docs/stencil_tree.md` §8.3, §12.1), and Claude reads the same schema. Each
schema is written to make the wrong call hard to write, and to leave every right
one open:

- **A job is one handle string**, never a node and an id given apart.
- **A mode is an enum, first, and each mode brings its own fields**
  (`repo_job_log`, `service_log`): the grammar writes `pattern` after `grep`
  and `from`/`count` after `lines`, so no call carries a field its mode ignores
  or lacks one its mode needs. The field a rule depends on comes before what
  depends on it.
- **Closed sets are enums**: a service is `motherd | web | mcp`.
- **Every string and array is bounded** (`maxLength`, `maxItems`) and every
  integer has its range, so no span runs on.
- **No shell**: a program and its arguments, one entry each. `&&`, `|`, `;`
  and redirects are refused by name before anything runs, with the reason —
  a model told only that the command failed tries the same line again.
- **No polling loops**: `repo_job_status` and `service_status` take `wait_s`
  and hold the answer until the job or deploy ends or the wait runs out, so a
  caller waits in one call, not twenty.
- **Every answer names the next call**, with its arguments filled in: the
  handle to watch, the mode that shows the failure. A model that has just been
  told what to call next does not invent a call.
- **What an executor accepts, the schema does not refuse**
  (`docs/stencil_tree.md` §12.1): a host is a string the executor checks
  against the agents connected, not an enum that would refuse a machine added
  tomorrow.

## 6. Zen's surface

Zen's workers get `service_deploy`, `service_status`, `service_log`,
`service_cancel`, and `repo_job_status`, `repo_job_log`, `repo_job_list`,
`repo_job_cancel` for watching a deploy's build — each defined in
`zend/src/prompts/tools/<name>.yaml` with its calibration questions and worked
examples, and executed by `zend-tools` through the daemon's `OpsRunner`, which
runs **the same implementation the MCP tools run** (`zend/src/mcp/ops_tools.rs`),
so the two surfaces cannot drift apart. A worker's own builds and tests stay
`run_command`, which runs its uncommitted changes; `repo_job_start` runs a
branch as origin holds it and is Claude's.

Every one is an Admin's; `service_deploy` and `service_cancel` are a coder's.

## 7. Guides

`mcp/src/guides/test.md` (running, watching, reading and stopping jobs) and
`mcp/src/guides/deploy.md` (a deploy, end to end, and what to do on each way it
can end) carry the procedure; `release.md`, `ab-trial.md` and
`guide-change.md` point at them, and the `zend-test` and `zend-deploy` skills
are their short forms.
