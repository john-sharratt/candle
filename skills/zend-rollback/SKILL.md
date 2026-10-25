---
name: zend-rollback
description: Roll the cluster back to a named earlier build — a rollout of that build, planned by the gateway and started only with the user's approval — and confirm every member runs it. Uses the zend MCP's cluster_status and cluster_command tools. Use when a released build misbehaves. Always needs the user's approval. An Admin's.
---

# Rolling back

A member's own `rollback` refuses a build below the floor, so a cluster-wide
rollback is a **rollout of the older build**.

## Do

1. `cluster_status`: the builds each member runs; the user names the build to
   go back to, or confirm with them which one.
2. Plan: `cluster_command {name: "rollout", args: {app: "zend", build:
   "<older build>"}}` — answers the gateway's plan and changes nothing.
3. Show the plan to the user and ask: rolling back is always their decision.
4. With approval:
   `cluster_command {name: "rollout", args: {app: "zend", build: "<older build>", confirm: "<older build>"}, confirm: "approved by the user"}`.
5. Watch with `cluster_status` until every member runs it and is ready.
6. Report every member's build and state, and note the reason on the release
   thread with `journal_note`.

## If the gateway refuses the plan

Report its reason exactly. Do not work around it.
