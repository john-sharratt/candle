---
name: thread
description: How to keep a thread for work that runs tasks or makes decisions — attach before working, note every decision, re-read after compaction. Read it before starting such work; a one-off read or a single job needs no thread.
tools: [thread_attach, thread_list, thread_archive, journal_note, journal_read]
---
# Keeping a thread

A **thread** is the place for one feature or area of work. Every task you
delegate runs under it, and its journal keeps every task started, how each
ended, and every decision you note. It is what you re-read when your context
is compacted, and what every worker on it starts with in its prompt.

## Start

A thread is for work that runs tasks or makes decisions. A one-off read — how
the cluster stands, what the logs say — or a single test job needs none:
answer it and stop.

1. `guide_list` once per conversation, to see what methods exist.
2. `thread_list` if the work may already have a thread.
3. `thread_attach {thread}` — a kebab-case slug for what the work is about
   (`review-layers`, `model-releases-research`). Two unrelated pieces of work
   never share a thread; one piece of work never spans two.
4. Read what it answers: the journal so far. If it holds earlier tasks, start
   from what they found, not from scratch.

## While working

- **Note each decision as you make it** — `journal_note {journal:
  "T:<thread>", text}` — with its reason, in a sentence or two. One note per
  decision. Name the thread rather than writing `"self"`: `"self"` is the
  thread last attached by *any* of the user's conversations open with Zen at
  once, so with two open it can land on the other one's thread.
- **Note what the user says** that later work must keep: a constraint, a
  preference, an approval and what it covered.
- **A standing rule of a repository** (a convention, something the user says
  always holds) goes on the project journal, `P:<repository>`, which every task
  in every thread starts with.
- Pass the thread to every `repo_task_start`.

## After your context is compacted, or Zen restarted

Re-attach (`thread_attach` with the same name), then
`journal_read {journal: "T:<thread>"}`, and carry on from the notes — not from
what you think you remember. Running workers are still running:
`repo_task_wait` on them. Do the same after a call answered that the model is
still loading or the service was unavailable (see `wait`).

## When the work is done

`thread_archive {thread}` once nothing on it is running. Its journal stays
readable, and attaching again reopens it — but archiving ends every finished
worker on it: none can be continued with `repo_task_reply` afterwards. Archive
when there is nothing left to follow up; a thread left alone archives itself
after a day with nothing new.
