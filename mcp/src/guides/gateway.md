---
name: gateway
description: How to change an upgrade's grace period — raise an app's floor now, or cancel a pending rise — and what that does to members still on older builds.
tools: [gateway_command, cluster_status]
---
# The gateway's floor

During an upgrade, the gateway keeps routing to members on the older build for
a 30-minute grace period; then the app's **floor** — the oldest build it
still routes to — rises, and members below it take no traffic until they hold
the floor's build.

## Do

1. `cluster_status`: each app's floor, pending floor, and which members are
   on which build.
2. Explain to the user what the change does: raising the floor now drops
   every member still below it from routing at once; cancelling keeps the
   current floor.
3. With approval:
   - `gateway_command {name: "raise-floor", args: {app}, confirm: "approved by
     the user"}` — end the grace period now;
   - `gateway_command {name: "cancel-floor", args: {app}, confirm: "approved
     by the user"}` — drop a pending rise.
4. `cluster_status` again: the floor, and which members route.

Draining a gateway is not a tool — it would cut this connection — and is done
from the gateway's page.
