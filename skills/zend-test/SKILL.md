---
name: zend-test
description: Build a branch and run its tests on a cluster agent — never on a member serving traffic — and report pass or fail, the failing tests and their output, using the zend MCP's repo_job_start and repo_job_log tools. Use to test a branch before review or merge. Admins only.
---

# Testing a branch on an agent

## Do

1. Start the job: `repo_job_start {repo, branch, program: "cargo", args: [...],
   timeout_s}` — e.g. `["test", "-p", "<crate>", "--no-fail-fast"]`. Name
   the crates the change touches; the whole workspace takes long. One job per
   crate, started together, when several are touched.
2. Read it: `repo_job_log {node, job}` answers how it stands; when it is still
   running, wait a little and read again with `from` set to where the last
   page ended. The job's outcome is in its first line once it has ended: exit
   code, and `timed out` if it ran past `timeout_s`.
3. On failure, find the failures: `repo_job_log {node, job, grep: "FAILED"}`, then
   read around each — `panicked at`, the assertion values.
4. Report:
   - **PASS** or **FAIL**, per crate, with the `test result:` line quoted.
   - Each failing test: its name, the panic line, the assertion values.
   - The job ids, so the user can read them.

## Notes

- A job runs on the branch as origin holds it: push first.
- A job lost to a motherd restart is reported lost — run it again.
- `repo_job_start` is an Admin's.
- Tests that need a GPU or a model run on the agents of machines that have
  one; say which machine ran them.

## If the agents are unreachable

Say so, and ask the user whether to run the tests locally.
