---
name: feature
description: How to take a feature from idea to release on one thread — spec, implement, test, review, merge, deploy, release — with the user's approval at each decision.
tools: [thread_attach, journal_note, repo_task_start, job_status, repo_task_reply, repo_job_start, job_read, job_grep, service_deploy, cluster_command, cluster_status]
---
# A feature, end to end

One thread for the whole feature (`thread`); note each decision as it is
made.

1. **Understand** — `ask` tasks for how the code works today; `research` for
   anything outside it.
2. **Spec** — write it yourself: what changes, where, the behaviour, edge
   cases, tests. The user approves it. Note it on the thread.
3. **Implement** — `implement`, on a branch, tests alongside.
4. **Test** — `test` on an agent, every crate the change touches.
5. **Review** — `review`, and `security-review` if it touches trust.
6. **Fix** — findings back to the implementing worker (`repo_task_reply`) or a new
   task; test again.
7. **Merge** — the user's decision; ask, with what was built, tested and
   reviewed.
8. **Deploy and release** — `deploy` on one machine, then `release`; each
   decision the user's.
9. **Close** — note what shipped; `thread_archive`.

At any step that fails, stop and report; never press past a failure to the
next step.
