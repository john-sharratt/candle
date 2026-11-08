---
name: briefing
description: How to answer "where do things stand?" — open threads and what waits on the user, running and finished work, cluster health, and bots' missions, in one summary.
tools: [get_user_context, thread_list, journal_read, cluster_status, member_status, approval_grant, bot_list, bot_mission_read]
---
# A briefing

1. `get_user_context` — the newest **outcomes** of tasks that ran alone,
   each with what it did, where it landed and what waits on an approval.
2. `thread_list` — open threads; workers running; anything waiting on the
   user (questions, stuck workers, **held approvals** — a `needs_approval
   <id>` a worker is waiting on).
3. For each active thread, its latest journal entries — what finished, what
   was decided.
4. `cluster_status` and `member_status` — anything wrong first (`health`).
5. `bot_list` and `bot_mission_read` for bots on missions — open, finished,
   stuck.

Summarise, most urgent first:

- **Waiting on you**: questions, held approvals (each with what it would do,
  granted with `approval_grant` on your yes) and anything an alone task left
  pending, each with where to answer.
- **Wrong**: anything failed, stuck or unhealthy.
- **Done since last time**.
- **Running**: what, and how far along.
