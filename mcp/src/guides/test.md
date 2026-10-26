---
name: test
description: How to build a branch and run its tests on a build agent — never on a member serving traffic — and report pass or fail with the evidence.
tools: [agent_job, agent_job_log]
---
# Testing a branch

A **build agent** is a cluster machine that checks out a repository and runs
programs on it at low priority. Tests run there, never on a member serving
traffic. Admins only.

## Do

1. **Push first** — a job runs on the branch as origin holds it.
2. **Start one job per crate the change touches**, together:
   `agent_job {repo, branch, program: "cargo",
   args: ["test", "-p", "<crate>", "--no-fail-fast"], timeout_s}`.
   The whole workspace takes long; name the crates.
3. **Read each**: `agent_job_log {node, job}` answers how it stands; while it
   runs, read again with `from` set to where the last page ended. Once ended,
   the first line has the exit code (and `timed out` past `timeout_s`).
4. **On failure**: `agent_job_log {node, job, grep: "FAILED"}`, then read around
   each — `panicked at`, the assertion values.

## Report

- **PASS** or **FAIL** per crate, quoting the `test result:` line.
- Each failing test: its name, the panic line, the assertion values.
- The job ids and the machine that ran them, so the user can read them.

## Notes

- A job lost to a restart is reported lost — run it again.
- Tests that need a GPU or a model run on an agent of a machine that has one;
  say which.
- If the agents are unreachable, say so and ask the user whether to run the
  tests some other way.
