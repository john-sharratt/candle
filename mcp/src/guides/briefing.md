---
name: briefing
description: How to answer "where do things stand?" — open threads and what waits on the user, running and finished work, cluster health, and bots' missions, in one summary.
tools: [thread_list, journal_read, cluster_status, bot_list, bot_mission_read]
---
# A briefing

1. `thread_list` — open threads; workers running; anything waiting on the
   user (questions, stuck workers).
2. For each active thread, its latest journal entries — what finished, what
   was decided.
3. `cluster_status` — anything wrong first (`health`).
4. `bot_list` and `bot_mission_read` for bots on missions — open, finished,
   stuck.

Summarise, most urgent first:

- **Waiting on you**: questions and approvals, each with where to answer.
- **Wrong**: anything failed, stuck or unhealthy.
- **Done since last time**.
- **Running**: what, and how far along.
