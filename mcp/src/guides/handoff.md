---
name: handoff
description: How to hand work on and pick it up — what to write on the thread before your context ends, and how to find a thread, read where it stands and act after a long break or in another session.
tools: [thread_list, thread_attach, journal_note, journal_read, repo_task_wait, cluster_status]
---
# Handing off and picking up work

## Before your context ends

When the conversation is about to end or be compacted with work open, write
one note on the thread — `journal_note {journal: "T:<thread>", text}` —
starting `HANDOFF:`, with:

- **Where it stands**: what is running, what waits on the user, what is done
  (each with its worker's path and entry, `T:<thread>/<name>#<n>`).
- **What was decided** that is not already a note.
- **What is not done**, and why.
- **What to do next**, and what needs the user's approval first.

Workers keep running without you: the note is how the next session knows
which to wait on.

## Picking up

1. `thread_list` — every thread, newest first, with workers running and
   anything waiting on you. Skip it when you were given the thread's name.
2. `thread_attach` to the one in question: its answer is the journal — what
   was decided, what was started, how each task ended, and any `HANDOFF:`
   note, which is where to start.
3. For each worker still running: `repo_task_wait` on all of them.
4. For each worker waiting on you (a question, a stuck loop): read its
   question before deciding anything.
5. `cluster_status` if the work touches the cluster or depends on the
   workers being up.
6. **Summarise where things stand** to the user before acting: done, running,
   waiting on them, and what you propose next. Do not redo work the journal
   says is done; do not assume work the journal does not record.
