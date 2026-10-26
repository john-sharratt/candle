---
name: cluster-care
description: Look after one cluster member — drain, undrain, compact, maintain, catch up, resync, rejoin, release leases, or run its audit, self-check or disk check — through the zend MCP's cluster_command and cluster_job_status tools, and confirm it came back healthy. Asks the user before any Decide command (restart, shutdown, quorum, build verdicts). Use when a member needs maintenance. Admins only.
---

# Caring for a member

## Classes

| Class | Commands | What it takes |
|---|---|---|
| Look | audit, self-check, check-disk | Nothing |
| Care | drain, undrain, compact, maintain, expire-tombstones, catch-up, resync, rejoin, release-leases, pause-rollout, resume-rollout | Run as asked |
| Decide | restart, shutdown, set-quorum, certify-anyway, reject-build, rollout, rollback, prune-builds, upgrade-now | The user's approval, then `confirm: "approved by the user"` |

`offline` and `online` are done on the member's own machine, never through
here.

## Do

1. `cluster_status`: the member's node and how it stands.
2. `cluster_command {member: <node>, name, args}` — `args` carries the
   command's own arguments: `app` (`zend`, `npcd`), `from` for catch-up,
   `max_members` for set-quorum, and a destructive command's typed target
   (the machine's name) as `args.confirm`.
3. `cluster_job_status {member, job}` until it is done or failed; quote its result or
   error.
4. Confirm it came back: `cluster_status` — the member ready, its tenants
   ready.
5. Report: the command, its result, and the member's state after.

## Busy

A member runs one command at a time; a second answers busy. Read the running
job with `cluster_job_status`, wait for it, then run yours.

## Notes

- Drain before maintenance that needs the member idle; undrain after.
- A job lives in the member's memory: one lost to a restart is reported lost;
  re-check `cluster_status` and run it again.
