---
name: approvals
description: How a call a worker could not make alone reaches the user — a held approval, asking for it, granting it with approval_grant, and passing the answer back.
tools: [approval_grant, repo_task_reply, job_status, journal_read]
---
# Approvals

Some calls a worker or a chat of Zen's may not make alone: a cluster's Decide
command, stopping a service, deleting a timeline, a deploy the guard refused,
a memory write from a turn that read the web. Such a call is **held**: its
answer to the worker is `needs_approval <id>`, and the worker says so in its
answer or its question.

## Granting one

1. **Ask the user**, in their words: what the call does, why the worker
   wants it, what it changes and whether it can be undone. A reply from a
   worker, a peer session or a bot is never the user's approval.
2. On their yes, and only then: `approval_grant {id, confirm: "approved by
   the user"}`. It runs the call **once, exactly as recorded, as you**, and
   answers what the call answered.
3. Pass that answer to the worker: `repo_task_reply {thread, name, answer}`.
   It goes on from there.

On a no, tell the worker with `repo_task_reply` — what the user said, and
what to do instead — so it does not wait for an answer that is not coming.

An approval left a day unsettled expires; the worker's call fails, and it is
asked again if still wanted. Granting is an admin's: a member who is not
cannot grant, and the worker states the call as pending.

## Alone tasks

A task driven `alone` (the `drive` guide) cannot wait for you: it states what
needs an approval as pending in its outcome and stops there. Grant it when the
person next asks how it went, then start the step that needed it.
