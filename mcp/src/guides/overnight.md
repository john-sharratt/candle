---
name: overnight
description: How to run work unattended while the user is away — decide against the design documents, note every decision, never commit or deploy without prior approval, and leave a report for the morning.
tools: [thread_attach, journal_note, journal_read, task_start, task_wait, task_reply, agent_job, agent_job_log, cluster_status]
---
# Overnight work

The user is away: nobody will answer a question until they are back.

## Rules

- **Decide, do not ask.** When a decision is needed, make the one most
  consistent with the design documents and the project's rules, and note it
  on the thread with the reason and how to reverse it.
- **Never commit to the default branch, merge, deploy, or run a Decide
  command** unless the user approved that specific action before leaving.
  Work on branches.
- **Finish every task.** A task half done is reported as such, with exactly
  what blocks it.
- **Parked items stay parked**: anything the user said to leave is left, and
  listed.

## Run

1. Attach to the thread; read its journal and the user's instructions.
2. List the tasks; delegate them (`delegate`), waiting on all together
   (`wait`); verify each result (`verify`) — nobody else will before morning.
3. Test on agents (`test`); review what was built (`review`).

## The morning report

- Headline: what is green, what is not.
- Done: each task, what changed, how it was verified.
- Decisions made for the user: each, why, how to reverse it.
- Still open: what could not be finished, and what blocks it.
- Proposed next steps: commits, merges, deploys for the user to approve.
