---
name: zend-watch
description: Watch a rollout, a member, or the logs for a stated condition — reporting only exceptions and the end state, not every check — by delegating the watching to a zend worker that polls cluster_status and cluster_logs itself. Use when asked to keep an eye on something for a while. Follows zend-thread.
---

# Watching

A watch is long and repetitive: delegate it, so it costs you one entry, not a
poll loop.

## Do

1. Attach to `watch-<what>` (zend-thread).
2. `zend_task {thread, name: "watch-<what>", prompt}`:

```
Watch <what> until <the end condition>, or for at most <duration>.

Every <interval>: check with cluster_status, and with cluster_logs for <pattern> at WARN or above
since your last check.
- Report nothing while all is well.
- On an exception — a member not ready, a tenant in fault, an ERROR line matching <pattern>, a
  rollout stalled for more than <minutes> — stop and answer with it at once.

Final response, in this shape:
END: the end condition met | exception | time up
STATE: the last cluster_status, one line per member
EXCEPTIONS: each, with its time and the evidence (status line or quoted log line)
```

3. `zend_wait` on it — it answers when the watch ends, either way.
4. Report the end state; for an exception, what it was and the evidence, and
   suggest the next step (cluster-logs, cluster-care, zend-rollback).

## Notes

The worker needs an Admin's tools for the gateway's view; a non-Admin's watch
sees the members' own statuses and logs.
