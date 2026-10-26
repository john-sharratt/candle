---
name: cluster-care
description: How to look after one cluster member — drain, compact, catch up, resync, rejoin, audit — through its maintenance commands, asking before any Decide command, and confirming it came back healthy.
tools: [cluster_command, cluster_job_status, cluster_status]
---
# Caring for a member

## Command classes

| Class | Commands | What it takes |
|---|---|---|
| Look | audit, self-check, check-disk | Nothing |
| Care | drain, undrain, compact, maintain, expire-tombstones, catch-up, resync, rejoin, release-leases, pause-rollout, resume-rollout | Run as asked |
| Decide | restart, shutdown, set-quorum, certify-anyway, reject-build, rollout, rollback, prune-builds, upgrade-now | The user's approval, then `confirm: "approved by the user"` |

`offline` and `online` are done on the member's own machine, never through
here.

## Do

1. `cluster_status`: the member's node id and how it stands.
2. `cluster_command {member: <node>, name, args}` — `args` carries the
   command's own arguments: `app` (`zend`, `npcd`), `from` for catch-up,
   `max_members` for set-quorum, and a destructive command's typed target (the
   machine's name) as `args.confirm`.
3. `cluster_job_status {member, job}` until it is done or failed; quote its
   result or error.
4. Confirm it came back: `cluster_status` — the member ready, its tenants
   ready.
5. Report the command, its result, and the member's state after.

## Notes

- One command per member at a time; a second answers busy. Read the running
  job, wait for it, then run yours.
- Drain before maintenance that needs the member idle; undrain after.
- A job lost to a restart is reported lost; re-check and run it again.
