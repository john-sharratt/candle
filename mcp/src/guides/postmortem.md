---
name: postmortem
description: How to write up an incident, a rollback or a bad run — the timeline from the audit and logs, the cause, what would have caught it, and the lasting fix noted.
tools: [cluster_audit, cluster_logs, journal_read, journal_note, timeline_read, repo_task_start, repo_task_wait]
---
# Postmortem

## Gather

- `cluster_audit`: the commands, admissions, refusals, closed sockets and
  floors in the window, each with its UTC time — a restart shows as a socket
  closed then an admission. It names who sent a command when the gateway
  recorded one; a certification or a crash has no sender.
- `cluster_logs`: the lines around each event (`logs`).
- `journal_read` the incident's or release's thread: what was decided and
  done.
- For a bad worker run: the worker's journal and its turns (`verify`) —
  `timeline_read {id: "T:<thread>/<name>"}` reaches a worker's turns even
  when it was lost before writing an entry.

## Write

```
WHAT HAPPENED: in a few sentences
TIMELINE: each event, UTC, with its evidence (audit entry, quoted log line)
CAUSE: the root cause, with evidence — not the trigger alone
IMPACT: what was affected, for how long
DETECTION: how it was noticed, and how long that took
WHAT WOULD HAVE CAUGHT IT: a test, a check, a rule
FIXES: what was done; what still needs doing
RULE: the rule that would have prevented it, for the project journal
```

## Keep the lesson

The rule that would have prevented it goes on the project journal
(`journal_note {journal: "P:<repository>"}`), so every future task starts
with it. A gap in a guide goes to the user as a proposed edit.
