---
name: watch
description: How to keep an eye on a rollout, a member or the logs for a while — delegated to a worker that reports only exceptions and the end state.
tools: [task_start, task_wait, cluster_status, cluster_logs]
---
# Watching

A watch is long and repetitive: delegate it, so it costs you one answer, not
a poll loop.

## Delegate

On a thread `watch-<what>`, `task_start {thread, name: "watch-<what>",
effort: "balanced", prompt}`:

```
GOAL: watch <what> until <the end condition>, or for at most <duration>.

METHOD: every <interval>, check with cluster_status, and with cluster_logs for
<pattern> at WARN or above since your last check.
- Report nothing while all is well.
- On an exception — a member not ready, a tenant in fault, an ERROR line
  matching <pattern>, a rollout stalled for more than <minutes> — stop and
  answer with it at once.

FINAL RESPONSE:
END: the end condition met | exception | time up
STATE: the last cluster_status, one line per member
EXCEPTIONS: each, with its time and the evidence (status line or quoted log
line)
```

Then `task_wait` on it: it answers when the watch ends, either way. Report
the end state; for an exception, what it was, the evidence, and the next step
(`logs`, `cluster-care`, `rollback`).

The worker needs an Admin's tools for the gateway's view; a non-Admin's watch
sees only the members' own statuses and logs.
