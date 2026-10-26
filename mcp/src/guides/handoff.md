---
name: handoff
description: How to pick up work another session started, or resume after a long break — find the thread, read where it stands, then act.
tools: [thread_list, thread_attach, journal_read, task_wait, cluster_status]
---
# Picking up work

1. `thread_list` — every thread, newest first, with workers running and
   anything waiting on you.
2. `thread_attach` to the one in question and read its journal: what was
   decided, what was started, how each task ended.
3. For each worker still running: `task_wait` on all of them.
4. For each worker waiting on you (a question, a stuck loop): read its
   question before deciding anything.
5. `cluster_status` if the work touches the cluster or depends on zend being
   up.
6. **Summarise where things stand** to the user before acting: done, running,
   waiting on them, and what you propose next. Do not redo work the journal
   says is done; do not assume work the journal does not record.
