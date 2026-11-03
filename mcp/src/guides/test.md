---
name: test
description: How to build a branch, run its tests or clippy on a build agent — never on a member serving traffic — wait for them, read their logs without pulling them whole, and report pass or fail with the evidence.
tools: [repo_job_start, job_status, job_read, job_grep, job_list, job_cancel]
---
# Building and testing on the agents

A **build agent** is a cluster machine that checks out a repository and runs
programs on it, at low priority beside a model. Builds, tests and clippy run
there, never on a member serving traffic. A **job** is one program run on one
agent, named by one **handle**, `build:<node>:<id>` —
`build:76c3a0f4d39bd9db:XeV_I_QPzpg` — which every `job_*` tool takes as `job`.
Admins only.

## Before

- **Push what you changed.** A job runs on the branch as origin holds it;
  nothing unpushed is in it. Testing `main` as it stands needs no push.
- `job_list {kind: "build"}` says what each agent is already running. Jobs
  on one checkout run one at a time, so a job started behind a long one waits
  for it — its phase says `waiting for its checkout behind <handle>`.

## Start

`repo_job_start {repo: "candle", branch, program: "cargo", args: [...]}`
answers at once — before the agent has fetched — with the handle, the machine
and the phase. Start independent jobs together; each answers in well under a
second.

- **The program alone in `program`, each word after it one entry of `args`.**
  There is no shell: `&&`, `|`, `;` and redirects are refused by name. One
  program per job.
- **Run the narrowest thing that answers the question.** One crate's tests
  (`["test", "-p", "web", "--lib"]`), one test by name, `check` or
  `["clippy", "-p", "web", "--tests", "--", "-D", "warnings"]` for a compile.
  The whole workspace's tests take many minutes and fail on things you did not
  touch. Run the full set once, before a release, not every round.
- **On a machine short of memory**, cap cargo's parallelism:
  `["test", "-j", "4", "-p", "web", "--", "--test-threads=4"]`. Linking several
  large binaries at once is what runs a machine out.
- `--no-fail-fast` runs every test binary past the first failure.
- `timeout_s` defaults to 30 minutes; a first build on a cold agent compiles
  the workspace — give it an hour.
- `node: "BlackWorld"` pins a machine. Left out, the gateway places the job:
  on the agent that last ran your jobs while it is free (its build is warm),
  else on the best free agent.

## Wait

`job_status {jobs: [<handle>, ...], wait_s: 240}` holds until any of the jobs
ends or the wait runs out — one call, not a loop. While it waits it says each
step; when it answers still running, call it once more the same way. A build's
phases:

| Phase | |
|---|---|
| fetching origin | getting the branch; no log yet |
| waiting for its checkout behind … | another job holds the checkout |
| running | the program runs |
| staging what it built | copying the `stage` file to the stage folder |

An ended job's answer says how it ended — `exited 0`, `exited 101`, `timed
out`, `cancelled`, `failed (fetching …)` before it ran — and quotes its
summary: cargo's `test result:` lines and the first errors and failures **with
their line numbers**.

## Read

`job_read` and `job_grep` read the log on the agent; only the answer comes
back, every line numbered as it stands in the log, 200 lines to a page.

1. The ended job's `job_status` answer first: results, failures and errors by
   line number, and `pages: N`.
2. `job_read {job, page}` on the page holding a line it named (page
   `(line - 1) / 200 + 1`) reads the failure in full — the panic, the
   assertion's values, the compiler's note.
3. `job_grep {job, pattern: "panicked at", context: 2}` finds every panic;
   plain text, each hit with its line and its page.
4. `job_read {job, page: <the last>, wait_s: 60}` follows a running job: a page
   not yet full waits until it fills.

Never read a log from the top a page at a time: a test log is megabytes, and
the summary and a search find what matters in a few hundred bytes.

## Stop

`job_cancel {job}` stops it at any phase: one still fetching or waiting
never runs, a running one is killed with what it started, and nothing is
staged.

## Report

- **PASS** or **FAIL** per job, quoting the `test result:` lines.
- Each failure: the test's name, the panic line, the assertion's values, with
  their line numbers.
- Each job's handle and machine, so the user can read them.

## Notes

- A job lives in its agent's process: one lost to the agent's restart is
  reported lost — start it again.
- A failure in code your change did not touch is usually not yours: check it
  on `main` before chasing it.
- If no agent is connected, say so and ask the user how to run the tests.
