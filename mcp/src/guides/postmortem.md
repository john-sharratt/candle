---
name: postmortem
description: How to write up an incident, a rollback or a bad run — the timeline from the audit and logs, the cause, what would have caught it, and the lasting fix noted.
tools: [cluster_audit, cluster_logs, journal_read, journal_note, task_start, task_wait]
---
# Postmortem

## Gather

- `cluster_audit`: every command, restart and membership change in the
  window, and who did each.
- `cluster_logs`: the lines around each event (`logs`).
- `journal_read` the incident's or release's thread: what was decided and
  done.
- For a bad worker run: the worker's journal and its turns (`verify`).

## Write

```
WHAT HAPPENED: in a few sentences
TIMELINE: each event, UTC, with its evidence (audit entry, quoted log line)
CAUSE: the root cause, with evidence — not the trigger alone
IMPACT: what was affected, for how long
DETECTION: how it was noticed, and how long that took
WHAT WOULD HAVE CAUGHT IT: a test, a check, a rule
FIXES: what was done; what still needs doing
```

## Keep the lesson

The rule that would have prevented it goes on the project journal
(`journal_note {journal: "P:<repository>"}`), so every future task starts
with it. A gap in a guide goes to the user as a proposed edit.
