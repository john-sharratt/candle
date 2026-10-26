---
name: wait
description: How to wait on running workers — all at once, handling whichever settles first, reading progress, and knowing when to stop waiting.
tools: [repo_task_wait, thread_list, repo_task_nudge, repo_task_stop]
---
# Waiting on workers

## Wait on all of them

`repo_task_wait {workers: [<every running path>]}` answers as soon as any one stops
running — done, failed, asking you a question, or stuck — with a line for each
still running. Handle the one that settled, then call `repo_task_wait` again on the
rest. Never wait on one worker while others run unwatched.

## Read what each answer brings

- **Progress lines**: what each worker is doing — rounds, tool calls,
  thinking.
- **New journal entries**: what a worker has found so far. If they already
  answer your question, stop waiting and use them; the work goes on whether or
  not anyone waits.
- **A done worker** answers with its final response in full: read it, then
  check it (`verify`).

## Settled states

| State | Do |
|---|---|
| done | Read the final response; verify what you will rely on |
| asking | Reply with the decision itself (`repo_task_reply`) |
| stuck | Its loop closed: reply with what to do next (`repo_task_reply`) |
| stopped | You stopped it: redirect it (`repo_task_reply`) or archive the thread |
| failed | Read why; fix the cause or start the task again |

## When a worker goes wrong

- Heading the wrong way, but still useful: `repo_task_nudge` (see `steer`).
- Repeating a step, the same fetch failing: `repo_task_stop`, then `repo_task_reply`
  with a new direction.

## Waits are cut off

claude.ai cuts a call off at about 200 s; Claude Code can wait an hour. A wait
cut short says so: call `repo_task_wait` again. When no call is open — a new
conversation, after compaction — `thread_list` shows what is running and what
waits on you.
