---
name: cluster-status
description: Summarise how the Tokera cluster stands — each member's machine, build, state and tenants, the gateway's apps, floors and upgrades in progress, and anything degraded or in fault — from the zend MCP's cluster_status tool. Use when asked how the cluster is, before and after any operation, before delegating heavy work, and whenever something seems slow or stuck.
---

# Cluster status

## Do

1. `cluster_status`. It answers every zend member's own status — machine,
   node, build, state, when it started, its tenants (zend, npcd) — and, for an
   Admin, the gateway's view: each app's mode, floor and ready members, upgrade
   candidates, members in or near fault.
2. Summarise for the user, in this order:
   - **Anything wrong first**: a member not ready, a tenant loading or in
     fault, a member that gave no answer, a member in fault, a local build
     serving cluster traffic.
   - **Upgrades in progress**: candidate build, its state.
   - **The rest**, one line per member: machine, build, state, started how long
     ago.
3. If a member is not ready, say what its status line says and suggest
   `cluster-logs` for why.

## Notes

- A member answering `no answer (504 deadline)` was alive to the gateway but
  did not answer within 3 s — busy, not necessarily down.
- `started N min ago` on one member only means it restarted then: check
  `cluster_audit` for who restarted it.
- Non-Admins see only the members' own statuses.

## If zend's MCP endpoint is down

Say so; the user can open `/gateway` in a browser.
