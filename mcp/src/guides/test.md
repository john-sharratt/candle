---
name: test
description: How to build a branch and run its tests on a build agent — never on a member serving traffic — and report pass or fail with the evidence.
tools: [repo_job_start, repo_job_log]
---
# Testing a branch

A **build agent** is a cluster machine that checks out a repository and runs
programs on it at low priority. Tests run there, never on a member serving
traffic. Admins only.

## Do

1. **A branch you changed: push it first** — a job runs on the branch as
   origin holds it. Testing `main` as it stands needs no push.
2. **Start one job per crate the change touches**, together:
   `repo_job_start {repo: "candle", branch, program: "cargo",
   args: ["test", "-p", "<crate>", "--no-fail-fast"], timeout_s: 3600}`.
   `repo` is the repository's name — `candle`, `battle-cities`.
   `--no-fail-fast` runs every test past the first failure. Name the crates:
   the whole workspace takes long, and even one crate builds what it depends
   on. An agent may build from cold, so give an hour per crate.
3. **Read each**: `repo_job_log {node, job}` answers how it stands. While it
   runs, read again every half minute or so, with `from` set to where the
   last page ended. Once it has ended, the page's header gives the exit code
   (and `timed out` past `timeout_s`).
4. **On failure**: `repo_job_log {node, job, grep: "FAILED"}`, then read around
   each — `panicked at`, the assertion values.

## Report

- **PASS** or **FAIL** per crate, quoting the `test result:` line.
- Each failing test: its name, the panic line, the assertion values.
- The job ids and the machine that ran them, so the user can read them —
  `cluster_status` names the machine of an agent's node id.

## Notes

- A job lost to a restart is reported lost — run it again.
- Tests that need a GPU or a model run on an agent of a machine that has one;
  say which.
- If the agents are unreachable, say so and ask the user whether to run the
  tests some other way.
