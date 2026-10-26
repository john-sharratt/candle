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
| lost — "the task stopped being run: the member running it restarted or lost it" | Its turn is gone; its conversation keeps what it did. Start the task again — the same name gets the next free one (`.2`) — and point the new prompt at the lost worker's journal entries, if it wrote any |

## When a worker goes wrong

- Heading the wrong way, but still useful: `repo_task_nudge` (see `steer`).
- Repeating a step, the same fetch failing: `repo_task_stop`, then `repo_task_reply`
  with a new direction.

## When Zen restarts

A call that answers "the model is still loading" or "is unavailable" means
Zen is restarting: every call answers that way until it is back, usually
within a few minutes. Wait a minute or two and call again; do not rewrite
what you asked. Nothing on a thread is lost — only a turn that was running,
which then reports as lost (above).

A `repo_task_start` whose answer was cut off may still have started its
worker. Before starting it again, `thread_list` (or `thread_attach` the
thread): a worker it shows under the name is yours — wait on that path.

## Waits are cut off

A call holds at most about four minutes — less on claude.ai — whatever
`wait_s` asks; a wait cut short says so: call `repo_task_wait` again. Keep
`wait_s` under 200 when starting or replying, so the answer arrives before
any client gives up on the call. When no call is open — a new
conversation, after compaction — `thread_list` shows what is running and what
waits on you.
