---
name: steer
description: How to redirect a running worker — a nudge for a course correction, a stop when it is looping, a reply that gives it the decision.
tools: [repo_task_nudge, job_cancel, repo_task_reply, job_status]
---
# Steering a worker

## Nudge — correct course without stopping it

`repo_task_nudge {task, message}` when its journal or progress shows it heading
the wrong way: searching the wrong place, missing an obvious source, about to
over-reach the scope. The message joins its next tool round's results — it is
read at the next point the worker looks up, never mid-thought. A worker that
ends its turn without another tool round never reads it.

Say what to change and why, in a sentence or two:
"Stop searching aggregators; fetch the vendor's announcement page for each
model instead."

## Stop — when it is looping or wasting work

`job_cancel {job: "task:review/verify"}` when it repeats a step, retries the
same failing fetch, or is on a path you no longer want. It stops at its next
tool round and answers with what it has found and what remains. It is then
stopped but live.

- The next tool round can be minutes away while the worker thinks or writes;
  until then the stop answers that it is queued. Wait on it as on any worker.
- The stop carries no reason: the direction you want ("scope is now only the
  Admin tools") goes in the `repo_task_reply` after it.

## Reply — give it the decision

After a stop, a question or a stuck loop, `repo_task_reply {task, answer}` is its
next message:

- Make it **the decision itself** — pick one of its options, or say exactly
  what to do instead. "It depends" sends it round again.
- Check its pointers first when the question rests on them.
- A reply grants nothing: it cannot approve a merge, a deploy, or anything the
  user must approve. Ask the user, then reply with what they decided.
