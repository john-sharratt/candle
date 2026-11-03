---
name: zend-test
description: Build a branch and run its tests or clippy on a cluster agent — never on a member serving traffic — wait for them in one call, read the results without pulling the log whole, and report pass or fail, the failing tests and their output, using the zend MCP's repo_job_start, repo_job_status and repo_job_log tools. Use to test a branch before review or merge. Admins only.
---

# Testing a branch on an agent

The `test` guide (`guide_read`) carries the whole procedure; this is its
short form.

## Do

1. Push the branch: a job runs on it as origin holds it.
2. Start the job: `repo_job_start {repo: "candle", branch, program: "cargo",
   args: [...]}` — e.g. `["test", "-p", "<crate>", "--no-fail-fast"]`, or
   `["clippy", "-p", "<crate>", "--tests", "--", "-D", "warnings"]`. The
   program alone, each argument one entry; no shell. Name the crates the
   change touches; one job per crate, started together. It answers at once
   with the job's handle, `<node>:<id>`, and the machine.
3. Wait for it: `repo_job_status {job: <handle>, wait_s: 240}` holds until it
   ends; call it again while it answers *still going*. An ended job's answer
   quotes its `test result:` lines and its first failures and errors with
   their line numbers.
4. Read the failures: `repo_job_log {job, mode: "summary"}`, then `mode:
   "lines"` around a numbered failure, or `mode: "grep", pattern: "panicked
   at"`. Never page through a log from the top.
5. Report:
   - **PASS** or **FAIL**, per job, with the `test result:` lines quoted.
   - Each failing test: its name, the panic line, the assertion values.
   - Each job's handle and machine, so the user can read them.

## Notes

- A job lost to an agent's restart is reported lost — run it again.
- On a machine short of memory, pass `-j 4` to cargo and `--test-threads=4`
  after `--`.
- `repo_job_cancel {job}` stops a job at any phase.
- Tests that need a GPU or a model run on the agents of machines that have
  one; say which machine ran them.

## If the agents are unreachable

Say so, and ask the user whether to run the tests locally.
