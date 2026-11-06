# Jobs

Everything that runs longer than a call — a coding or research task, a build or
test run on an agent, a deploy, a cluster maintenance job — is a **job**, and
every job is watched, read, searched, listed and stopped with the same five
tools, in the MCP connector Claude uses and in Zen's own surface alike:

| Tool | Does |
|---|---|
| `job_list {kind?, running?, limit?}` | every job the caller can see, newest first, one line each |
| `job_status {jobs, wait_s?}` | where each job is, waiting up to `wait_s` for any of them to end; ends with each one's summary and `pages: N` |
| `job_read {job, page, wait_s?}` | one page of a job's output; a page not yet full waits until it fills, the job ends, or `wait_s` passes |
| `job_grep {job, pattern, context?}` | the whole output searched, each hit with its line number and its page |
| `job_cancel {job}` | stop it, whatever kind it is |

What starts a job stays per kind, because what is asked differs:
`repo_task_start` (a task), `repo_job_start` (a build or test run),
`service_deploy` (a deploy), `cluster_command` (a maintenance job). Each
answers the new job's handle, and every later step is a `job_*` call. A task's
own turns are still steered with `repo_task_reply` and `repo_task_nudge`.

They replace `repo_task_wait`, `repo_task_report`, `repo_task_stop`,
`repo_job_status`, `repo_job_log`, `repo_job_list`, `repo_job_cancel`,
`service_status`'s deploy mode, `service_log`, `service_cancel` and
`cluster_job_status`. `service_status` keeps what is not a job: what each
machine's keep runs. A task's own turns are steered by its handle:
`repo_task_reply {task, answer}` and `repo_task_nudge {task, message}`.

## 1. Handles

A job is named by one handle, `<kind>:<id>`, the way every answer writes it and
every tool takes it back:

| Kind | Handle | The id |
|---|---|---|
| a task | `task:<thread>/<name>` | the worker's path below `T:` |
| a build | `build:<node>:<id>` | the agent's 16-hex-digit node and the job's id there |
| a deploy | `deploy:<node>:<id>` | its build job's |
| a maintenance job | `cluster:<node>:<n>` | the member's node and the job's number there |

One opaque string a model copies is a string it cannot pair wrongly. The kind
is in the handle, so a tool needs no `kind` argument to know what it reads.

**A deploy is named by its build job.** keep records the deploy's handle on the
trial it starts for it (`keep/src/board.rs`), so any member — whichever one a
call lands on — answers a deploy from the agent and from keep, and from nothing
held in one process's memory.

## 2. Starting at once

A start never waits on work that can take seconds.

- **The agent accepts a run before it fetches** (`agent/src/jobs.rs`). The
  request is checked, the job registered under its own id, and the answer goes
  back; fetching origin, building the store and waiting for the checkout are
  phases of the job's own task.
- **The gateway never holds a job back** (`cluster/src/gateway/agents.rs`). A
  busy lineage holder spills to the best free agent at once; with none free the
  job queues on its holder or the best agent.
- **`service_deploy` answers once its build is accepted.** A driver in the
  member that took the call waits for the build and asks keep for the trial.

A build's **phase** (`agent::wire::Phase`): `fetching`, `waiting {behind}` (the
checkout is held by the runs behind), `running`, `staging`, `ended`.

## 3. A job's output

Every job has an output: lines, numbered from 1 across the whole of it, which
`job_read` pages and `job_grep` searches. Line numbers never move, so a line a
search found is the line a page read returns. A page is 200 lines; page `P` is
lines `200(P-1)+1 … 200P`; each line is cut at 400 characters.

| Kind | Its output |
|---|---|
| a build | its log, read on the agent (`POST /v1/jobs/:id/read`) |
| a deploy | its build's log, then a marker line, then the trial's log on keep's machine (`POST /v1/keep/trial/log`) |
| a task | its prompt, then each journal entry as it is drafted, then — once it settles — its question or its answer, then its turn's report (rounds, time, calls by tool, what failed) |
| a maintenance job | its event stream on the member, a line per event (`started`, each `step`, `done` or `error`) |

The output is read where it is written; only the answer crosses the cluster.

**A page that is not full yet waits.** `job_read {job, page, wait_s}` on a page
the job has not finished writing holds until the page fills, the job ends or
`wait_s` passes, then answers what is there: `tail -f` in bounded chunks, so a
caller follows a running job without polling and never receives more than a
page. A page past the end answers the last page and says so.

**Every answer says where it is**: `page P of N`, whether the job is still
writing, and the call that reads on.

## 4. `job_status`: one call that waits

`job_status {jobs: [...], wait_s}` takes one to sixteen handles and answers as
soon as **any** of them ends — or when `wait_s` runs out — so a caller starts
every job first and waits on them together. For each job:

- where it is: a build's phase, machine, elapsed and lines; a deploy's stage
  (building, built, being tried, proven, rolled back, cancelled); a task's
  state (running and what it is doing, asking a question, stuck, done, failed);
  a maintenance job's last step;
- once ended, its **summary**: a build's or a trial's test results and first
  failures and errors by line (`cluster/src/log_read.rs`); a task's answer and
  its new journal entries, each given once (`reported`); a maintenance job's
  result;
- `pages: N`, and the call that reads what the summary pointed at.

`wait_s` defaults to 60 and is capped by the caller: 240 s for Claude Code,
200 s for claude.ai and Grok, 300 s for a worker. A call cut short says so and
names the call that keeps waiting. While it waits, each step is said on the
call's progress (MCP progress notifications), which a client shows and the
model does not read.

## 5. `job_list`

`job_list {kind?, running?, limit?}`: every job the caller can see, newest
first, one line each — handle, kind, where it runs (machine or thread), state,
age, what it is, `pages: N`. `running` (true by default) leaves out what ended;
`kind` narrows to `task`, `build`, `deploy` or `cluster`. A deploy is listed
once, as a deploy, never again as its build. A member lists its recent
maintenance jobs at `GET /v1/cluster/command/jobs`.

## 6. Deploys

`service_deploy {host, service}`:

1. Refuse what keep would refuse: the service is one keep runs (`motherd`,
   `web`, `mcp`), and the machine has an agent.
2. For a worker of Zen's: refuse a `motherd` trial on its own machine and one
   that takes an app's only ready member down
   (`zend/src/mcp/tools/service/guard.rs`).
3. Start `cargo build --release -p <service>` on that machine's agent, on
   `main`, staging `target/release/<service>.exe`; answer the deploy's handle.
   A `motherd` is built with the connector it releases (`-p motherd -p mcp`),
   and a build queued ahead of it on the same agent stages that connector, so
   the trial's daemon follows it and offers it to the members that install
   the release (`docs/replicated_substrate.md` §3.9).
4. The driver waits for the build and, once it is staged, posts the trial to
   that machine's keep with the deploy's handle.
5. keep runs the trial: proven on `INFO serving` within ten minutes, the window
   restarting at each sync line, or rolled back.

`job_cancel` on a deploy cancels the build while it runs, or asks keep to stop
the trial, which rolls it back and records the end as cancelled.

## 7. Stencils: constrained, not cornered

Zen's model calls these tools under a stencil compiled from each tool's schema
(`docs/stencil_tree.md` §8.3, §12.1); Claude reads the same schema. Nothing can
be taken back once written, so each schema makes the wrong call impossible to
begin and leaves every right one open:

| Rule | Where |
|---|---|
| **Required first, in order; optionals after, in order**, so the field a choice depends on is written before what depends on it. Calibration examples write their fields in that order. | every definition |
| **No modes.** Reading, searching, status and stopping are separate tools, so the tool's name settles the call's shape. | the five tools |
| **A handle has its kind's exact shape.** `job` is held to `task:` + a thread and the one worker below it, each a kebab slug, the worker possibly numbered (`verify.2`), `build:`/`deploy:` + 16 hex + `:` + an 11-character job id whose last character is one of 16, `cluster:` + 16 hex + `:` + digits — the prefix a choice of four. | `tool_constraints.yaml` |
| **A handle is one the round can name.** Each round, `job`'s values are narrowed to the handles of the jobs that exist for the caller ([`stencil::live_values`]); a tool with none to name is taken out of the round. | the stencil's live values |
| **Closed sets are enums, read from the source of truth**: a machine is one `host/machines.yaml` names, a service one keep runs, a guide one the library holds. | the definitions, at startup |
| **No `null`.** An optional field is left out. | `#[schemars(with = …)]` |
| **A wait is never optional for Zen.** `job_status` and `job_read` require `wait_s` in Zen's grammar: once the required fields are written, closing the call is the cheapest token, so an optional wait is the one a model leaves out, and the call waits a default nobody chose — measured on workers told in their prompt to wait 240 s. Claude, which writes the field it means, keeps it optional. | `zend-tools/src/tools/ops/job.rs`, the definitions |
| **Every string and array is bounded, every integer ranged**: `wait_s` 0–300 in Zen's grammar, its real ceiling; `page` from 1; `pattern` 1–200 characters of plain text (a value under a rule holds no backslash, so no regular expression can be written into it). | the definitions |
| **A tool that cannot succeed is not offered**, and the model is told why in a sealed line: `job_cancel` with nothing running; `service_deploy`, `service_cancel` and `job_cancel` for a worker that is not an Admin's coder; the cluster readers for a task whose record names no person. | `zend/src/tool_availability.rs` |
| **What the executor accepts, the grammar does not refuse** (§12.1): limits are clamped, a page past the end answers the last page, an ended job is read like a running one. | the executors |
| **No shell.** A program and its arguments, one entry each; shell words are refused by name. | `repo_job_start` |
| **No polling.** `wait_s` holds the answer; every answer that is not final names the call that keeps waiting. | `job_status`, `job_read` |

## 8. Zen's surface

A conversation of Zen's — a task, a chat, a person's dialogue — gets the five
`job_*` tools, `repo_job_start` (a branch with the task's uncommitted changes
laid onto it — the overlay its last tool round saved, carried in the job as
`run_command`'s is — on a machine it names; the answer says how many files it
carries, `docs/zen_coding_challenge_fixes.md` Cause 5), `service_deploy`, `service_status`, `cluster_status`, `cluster_logs`,
`cluster_audit`, `member_status`, `timeline_read` (another run's turns, calls,
projections and prompt, for diagnosing it), `layer_read`, the acting tools of
`docs/zen_tool_reach.md` §6.1, `guide_list` and `guide_read`, each defined in
`zend/src/prompts/tools/<name>.yaml` with
calibration questions and worked examples, and executed through the daemon's
`OpsRunner`, which runs **the implementation the MCP tools run**
(`zend/src/mcp/ops_tools.rs`). A worker acts as the person it works for: its
record carries them (`threads::Worker::person`), and every request it sends
states them, so each hop checks their role.

A worker's own builds and tests stay `run_command`, which is not a job: it runs
on the conversation's branch with its uncommitted changes, waits for the run
within the call, and records what the run changed in the conversation's store.
What it printed past the first page is read with `run_output`, from the log the
call kept — in a daemon with agents and in one running its own sandboxes alike,
and for a worker of any role. The job tools are for what outlives a call.

## 9. Guides

`mcp/src/guides/test.md` (building and testing on the agents) and
`mcp/src/guides/deploy.md` (a deploy end to end) carry the procedures;
`release.md`, `ab-trial.md` and `guide-change.md` point at them, and the
`zend-test` and `zend-deploy` skills are their short forms. The guides are read
from one library (`guides/`) by the connector and by Zen alike.
