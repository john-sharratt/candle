---
name: drive
description: How to run a task you will not be there to steer — attended or alone, what an alone brief must name, and where its outcome is found later.
tools: [repo_task_start, job_status, repo_task_reply, journal_read, get_user_context]
---
# Driving a task: attended or alone

Every task is **attended** or **alone** (`repo_task_start`'s `driven`).

| | Attended | Alone |
|---|---|---|
| Who answers its questions | you, with `repo_task_reply` | no one: it decides within its brief |
| Who checks its result | you, against the turns it cites | its own verify step |
| Who lands it | it, once you agree; or you | it, as its brief says |
| Approvals | you carry them to the user | it states them as pending and stops |
| Default | Claude Code, claude.ai, a drone | Grok, whose conversation ends with the car |

Choose **alone** whenever nobody will be at this conversation when the task
asks something: a voice conversation, a "do it overnight", a request from a
car. Otherwise attended.

## An alone brief names three things

The worker cannot ask, so the brief must already answer:

1. **Where it lands** — the repository, the branch (or `main` for a
   repository whose rule is to push there, as `journals` is), and whether it
   pushes. An alone task with no landing writes a document and stops at
   "shall I commit it?".
2. **Done when** — the condition that ends it, checkable from outside: a test
   passing, a file at a path, a commit on a branch.
3. **Never** — what it must not do whatever it finds: push to `main` of a
   code repository, force-push, deploy, delete, change a design decision.

Add the guide's own template (`implement`, `docs-write`, `review` …) and its
verify step: an alone task checks itself because nobody else will.

## Write the handoff at the start

Before you start it, note on the thread (`journal_note {journal: "self"}`)
what it is doing and what "done" is, in a sentence: the person, or the next
conversation, finds it there, not in this conversation, which may be gone.

## Its outcome

When an alone task ends — done, failed, stuck or lost — Zen writes its
**outcome** on the thread: what was done, where it landed, what waits on an
approval. The person's next conversation opens with the newest outcomes in
its context (`get_user_context`), so "how did that go?" is answered from
there, and `journal_read {journal: "T:<thread>"}` reads the rest.

An attended task you leave becomes, in practice, an alone one with no brief
for it: if you may not be back, start it alone.
