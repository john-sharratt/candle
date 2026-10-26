---
name: rollback
description: How to roll the cluster back to an earlier build — a rollout of that build, planned by the gateway, started only with the user's approval, watched until every member runs it.
tools: [cluster_command, cluster_job_status, cluster_status, journal_note]
---
# Rolling back

A member's own `rollback` refuses a build below the floor, so a cluster-wide
rollback is a **rollout of the older build**. Always the user's decision.
Admins only.

## Do

1. `cluster_status`: the builds each member runs. The user names the build to
   go back to, or confirm with them which.
2. Plan, which changes nothing: `cluster_command {name: "rollout", args: {app,
   build: "<older build>"}}`.
3. Show the plan and ask.
4. With approval: `cluster_command {name: "rollout", args: {app, build:
   "<older build>", confirm: "<older build>"}, confirm: "approved by the
   user"}`.
5. Watch with `cluster_status` until every member runs it and is ready.
6. Report every member's build and state; note the reason on the release
   thread.

If the gateway refuses the plan, report its reason exactly. Do not work
around it.
