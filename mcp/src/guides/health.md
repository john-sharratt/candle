---
name: health
description: How to read how the cluster stands — every member's build and state, each app's floor, upgrades in progress, anything in fault — and summarise it with problems first.
tools: [cluster_status, service_status, cluster_logs, cluster_audit]
---
# Cluster health

A **member** is one machine running the motherd daemon, which hosts the apps:
`zend`, the coding service these tools run on, and `npcd`, the bots'. The
**gateway** routes requests to members; each app's **floor** is the
oldest build it still routes to.

## Do

1. `cluster_status`: every member — machine, node id, build, state, when it
   started, how each app on it stands — and, for an Admin, the gateway's
   view: each app's floor and ready members, upgrade candidates, members in or
   near fault.
2. For a machine's services and its last trial: `service_status {host}`,
   with the host as `cluster_status`'s keep rows name it — those are the
   machines.
3. Summarise, in this order:
   - **Anything wrong first**: a member not ready, a tenant loading or in
     fault, a member that gave no answer, a local build serving cluster
     traffic.
   - **Upgrades in progress**: the candidate build and its state.
   - **The rest**, one line per member: machine, build, state, started how
     long ago.
4. A member not ready: quote its status line and look at why (`logs`).

## Reading it

- `no answer (504 deadline)` — the member was alive to the gateway but did
  not answer in 3 s: busy, not necessarily down.
- `started N min ago` on one member means it restarted then: `cluster_audit`
  says who restarted it.
- Report a build as its count and hash, `3825-bc8c1817` — the form the floor
  and upgrades use — rather than the longer hex a member line may show.
- Times are UTC; the user's clock may not be.
- Non-Admins see only the members' own statuses.
- If the connector itself is down, say so; the user can open `/gateway` in a
  browser.
