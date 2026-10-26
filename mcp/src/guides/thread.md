---
name: thread
description: How to keep a thread — attach before working, note every decision, re-read after compaction. Read first, before any other guide.
tools: [thread_attach, thread_list, thread_archive, journal_note, journal_read]
---
# Keeping a thread

A **thread** is the place for one feature or area of work. Every task you
delegate runs under it, and its journal keeps every task started, how each
ended, and every decision you note. It is what you re-read when your context
is compacted, and what every worker on it starts with in its prompt.

## Start

1. `guide_list` once per conversation, to see what methods exist.
2. `thread_list` if the work may already have a thread.
3. `thread_attach {thread}` — a kebab-case slug for what the work is about
   (`review-layers`, `model-releases-research`). Two unrelated pieces of work
   never share a thread; one piece of work never spans two.
4. Read what it answers: the journal so far. If it holds earlier tasks, start
   from what they found, not from scratch.

## While working

- **Note each decision as you make it** — `journal_note {journal: "self",
  text}` — with its reason, in a sentence or two. One note per decision.
- **Note what the user says** that later work must keep: a constraint, a
  preference, an approval and what it covered.
- **A standing rule of a repository** (a convention, something the user says
  always holds) goes on the project journal, `P:<repository>`, which every task
  in every thread starts with.
- Pass the thread to every `repo_task_start`.

## After your context is compacted

Re-attach (`thread_attach` with the same name), then
`journal_read {journal: "self"}`, and carry on from the notes — not from what
you think you remember. Running workers are still running: `repo_task_wait` on
them.

## When the work is done

`thread_archive {thread}` once nothing on it is running. Its journal stays
readable; attaching again reopens it.
