# Battle Cities on the cluster

The Battle Cities game server runs on the cluster's machines as the app `battlecities`, behind the gateway that fronts `battlecities.net`, and is built and deployed by the cluster's own engine from the game's repository. This document is the cluster's side of it. The game's side is `docs/project/web-api.md` §16 in the `battle-cities` repository, and is built and tested there.

## 1. Decisions

John's, 17 November 2026.

| Decision | Choice |
|---|---|
| Worlds | One world, on one machine, for now. A second machine's server is a second member with a world of its own until the game shares one. |
| Membership | The game links the `cluster` and `cluster-dial` crates and joins as its own app. It holds no substrate and is no part of a quorum. |
| The world's data | Starts from the copy in git (the game's `data` submodule), copied once to a folder outside the checkout. |
| Players | Reach the game through the gateway, on the LAN and from outside. The gateway spreads their datagrams over the members by the address and port they come from. |
| Deploys | Every push to the game's `main` is built and deployed at once. A newer push replaces a deploy that has not reached its trial. Asking for a deploy of a commit already being deployed, by a push or by hand, answers with that deploy. |
| The push signal | GitHub's webhook, with a watch of origin as the fallback. |
| Who builds this | It is written on the laptop and built, tested and merged by the BlackWorld session. Nothing in §8's first list has been compiled. |

## 2. The app

`battlecities` is one of the cluster's services (`cluster::gateway::SERVICES`), as `keep`, `mcp` and `zenj` are: its members are admitted on whatever build they run, carry no model lock, and are never certified or rolled out. Its build count and hash are the game repository's, stamped by the game's own `build.rs`, and are compared with nothing in this repository.

A member declares every path (`GET`, `HEAD`, `POST`, `PUT`, `DELETE` of `/` and `/*path`) to any member, and serves each stream the gateway opens with the same application its own port 4080 serves. With more than one member a request goes to the least loaded, which is right only while there is one world; routing a player to the member that holds their world is the game's to add when it shares one.

## 3. The service on a machine

keep runs it (`keep/keep.yaml`):

| | |
|---|---|
| Service | `battlecities` |
| Binary | `keep/stable/battle-cities-server.exe`, so the staged file is `battle-cities-server.exe` |
| Working folder | The agent's checkout of the game, `D:/agent/battle-cities`: the server reads `server_assets/`, which its own build writes there |
| The world | `--data D:/battle-cities/data`, seeded once from the checkout's `data` submodule by `--seed-from data` |
| Membership | `--cluster-url`, `--cluster-name`, `--group-key`, `--host`, and `--state D:/battle-cities/state` for its node id |
| Ports | TCP 4080 (the site, the API, the client's updates), UDP 4300 (the game), UDP 4123 (the time) |
| Proof | `INFO serving` on stderr, once the world is loaded and the ports are bound |
| Build | `--build-info` answers `{"kind":"battlecities",...}`; a build with anything local in it is refused as any is |

The game's `.agent/setup.yaml` checks the `data` submodule out and pulls its LFS files when the agent starts, so the seed and the pictures the server builds in are there before the first build.

The gateway's site `battlecities` (`web/web.yaml`) sends every path to the app, and to `http://127.0.0.1:4080` while no member is joined. Sign-in stays the gateway's; the game believes who a caller is from the `X-Tokera-*` headers, on a gateway's stream and from loopback.

## 4. Deploys

`service_deploy {service: "battlecities"}` builds `cargo build --release -p battle-cities-server` on the agent's checkout of `battle-cities` at `main`, stages `target/release/battle-cities-server.exe`, and asks keep for the trial, as for every service. `zend/src/mcp/tools/service/mod.rs` names the repository (`repo_of`) and the package (`package_of`) of each service; a deploy's build job is read back as its service by the package it builds.

### 4.1 The webhook

GitHub is given one webhook on the game's repository: `POST https://tokera.com/gateway/hooks/github`, content type `application/json`, the `push` event only, with a secret. The gateway keeps the secret in `web/secrets/github-hook.key` and believes a delivery only when `X-Hub-Signature-256` is the body's HMAC-SHA256 under it (`web/src/hooks/github.rs`, `signed`). A believed push of a branch (`pushed`) is published on the bus as the pushes Zen makes are:

```json
{"repo": "battle-cities", "branch": "main", "commit": "a744abc8…", "paths": []}
```

on `repo/battle-cities`. A delivery that is not signed is answered `401` and logged; one that is signed and is no push of a branch (a ping, a tag, a deleted branch) is answered `204`.

### 4.2 The watch

A webhook can be lost. Every 60 seconds the push driver (§4.4) lists origin through an agent (`Agents::list`, which fetches nothing) and treats a `main` that is not the commit it last deployed as a push of it.

### 4.3 One deploy of a commit

`zend/src/mcp/tools/service/pending.rs`. A request for a deploy first reads what origin's `main` holds, the agent's newest jobs and keep's board, and is answered from them, so any member answers the same:

| What is under way on that machine | The answer |
|---|---|
| A build of the service, of the commit `main` holds | That deploy's handle: nothing is started |
| A build of the service that has not fetched origin yet | That deploy's handle: it will build what origin holds |
| A build of the service, of an older commit | It is cancelled, and a new build starts |
| A trial of the service, of the commit `main` holds | That deploy's handle |
| A trial of the service, of an older commit | The trial is left to end; a new build starts and its trial follows |
| Nothing | A new build starts |

The agent records, on each run, the commit the run's store is on once it has fetched origin and laid the overlay down (`JobLine::base`, from `VfsStore::base` in `Jobs::start`). That is what a build is "of". It is the store's own base, not the branch resolved apart from it: a fetch moves origin's copy of a branch and never the local one, a store reads origin's copy first, and `rev-parse` of the branch names the local one — recorded that way, a run on `main` after a push named the commit before it (`agent/tests/agents.rs`, `a_run_after_a_push_is_of_the_pushed_commit`). An agent that records none names no commit, and a deploy beside such a build starts as it always did.

This holds for every service and every way of asking: a second `service_deploy` of a commit being deployed answers with the first, whoever started it.

### 4.4 The push driver

One driver per repository that deploys on a push, in the member that holds the lease for it (a second member starting the same commit at the same instant would be the one case §4.3 does not catch). It hears `repo/<repo>` and runs the watch, and for a commit on `main` it has not deployed it calls `service_deploy` for each service built from that repository (`battlecities` from `battle-cities`), to the machine the cluster builds on first. It acts as the operator named in its configuration, since a deploy's requests to keep state who they are for. §4.3 makes its call safe to repeat.

Not on a push: `motherd`, `web` and `mcp`. Their deploys stop models or the gateway and stay a person's decision.

## 5. The game's datagrams

The game's client speaks UDP to port 4300 and reads the time from port 4123. The gateway forwards both.

- **A flow** is one client address and port on one of the two ports. The gateway opens one socket of its own for it, connected to the member's port, and copies datagrams both ways. A flow that carries nothing for 60 seconds is closed.
- **The member** a new flow is given to is chosen by a hash of the client's address and port over the app's ready members, so players spread over them and one player's datagrams always reach the server that holds their session. A flow stays with its member while it lives, whatever joins or leaves.
- **Who the client is.** The server sees the gateway's socket, not the client. Before it forwards a flow's first datagram, and every 30 seconds after, the gateway tells the member: `POST /api/v1/flows` on the member's stream, `{"via": "<the flow socket's own address and port>", "client": "<the client's address>"}`, with the header `X-Tokera-Forwarder: 1`. The server then counts its per-address limits and writes its security log by the client. An announcement stands 120 seconds.
- **Trust.** `X-Tokera-Forwarder`, `X-Tokera-Scope` and `X-Tokera-Keep` are among the headers the gateway removes from every request it passes on (`proxy::IDENTITY_HEADERS`), so only the gateway itself can say them.
- **Limits.** At most 8,192 flows, and 64 from one client address; past either a new flow's datagrams are dropped and counted.

The member's address for a flow is the address its cluster socket came from. The client's server list names the gateway (`battlecities.net`, and the gateway's LAN name).

## 6. Signing in and tokens

- `battlecities.net` is a sign-in domain of its own (`auth.domains`: its cookie domain and its `redirect_uri`, which is registered at Google as `https://battlecities.net/auth/callback`).
- The OAuth server gains the game's API as a second resource, `https://battlecities.net/api`, beside `/mcp`. A token for it is granted any of six scopes: `read`, `chat`, `play`, `manage`, `account`, `admin`. The gateway forwards a token's scopes as `X-Tokera-Scope`, space-separated; a browser's session carries no such header and has every right of its account.

## 7. Stopping a service that saves

keep stops a service by ending its process. The game saves what changed every few seconds and when it is asked to stop, so a deploy on every push must ask first: a service may name `stop: {url, wait_s}` in `keep.yaml`, and keep then sends `POST <url>` with `X-Tokera-Keep: 1`, waits up to `wait_s` for the process to exit, and only then ends it. The game answers it at `http://127.0.0.1:4080/api/v1/stop`, where it finishes its tick, saves the world and exits as on Ctrl+C; its entry in `keep.yaml` names that URL once keep reads `stop`.

## 8. Where the work stands

**Written here, not compiled or run** (the laptop builds nothing of this repository; the BlackWorld session builds, tests and merges):

- The app's name among the services and its build kind (`cluster/src/gateway/hub.rs`, `admission.rs`, `version/info.rs`).
- The deploy engine's service, repository and package (`zend/src/mcp/tools/service/mod.rs`, `deploy.rs`, `service_control.rs`), with their tests.
- One deploy of a commit (§4.3): `pending.rs` and its tests, its use in `service_deploy`, and the commit an agent records on a run (`agent/src/jobs.rs`, `wire.rs`).
- The webhook's signature and body (§4.1): `web/src/hooks/github.rs` and its tests. The signature test is GitHub's own published example.
- The supervisor's entry and the site's route (`keep/keep.yaml`, `web/web.yaml`), and the three headers added to the list the gateway removes (`web/src/proxy.rs`).

**Checked when it was first built** (BlackWorld, 17 November 2026; the agent, cluster, zend and web suites pass, and clippy and fmt are clean):

- The commit a run records was stale: `Jobs::start` resolved the branch with `Repo::resolve(&Rev::Branch(..))`, which names the local branch, while the fetch moves only origin's copy and the store reads that first. It now records the store's own base (§4.3), and a run on `main` after a push is the test that holds it.
- The agent builds in the checkout itself (`motherd/agent.yaml` names `D:/agent/battle-cities`), and `server_assets/` and `target/` are ignored there, so the service's working folder is that checkout.
- `service_deploy`'s and `service_control`'s descriptions name one more service. The tool calibration reads them, so the next calibration pass after the deploy recalibrates those two tools.
- `web/src/roles.rs`'s assertions of the `battlecities` site hold with its route.

**Designed above, not written:** the route that receives the webhook and publishes it (§4.1), the watch and the push driver (§4.2, §4.4), the forwarder (§5), the sign-in domain and the second OAuth resource (§6), and keep's stop request (§7). Each reaches into machinery that should be written where it can be compiled and run: the driver needs an operator's identity for its requests to keep, the forwarder the registry's view of each member's address, the OAuth resource the token store. The game's half of each is built and tested: the flow table and `POST /api/v1/flows`, `POST /api/v1/stop`, the scopes read from `X-Tokera-Scope`, the proof line, the build answer.

Until keep asks before it ends the process (§7), a deploy ends the game where it stands, and what changed since its last save, a few seconds at most, is lost. That is the reason to write §7 before the push driver is switched on.

**What John does:** adds the webhook in GitHub (§4.1) with the secret the gateway is given, and registers `https://battlecities.net/auth/callback` at Google (§6).

## Related documents

- `docs/replicated_substrate.md` §3.5–§3.11: keep, trials, the proof line
- `docs/jobs.md` §6: `service_deploy` and the jobs it is made of
- `docs/mcp_file_tools.md` §5: the bus, `repo/<repo>` and the gates
- `docs/zenj.md` §3.1: another service that is a cluster member of its own
- `docs/yenbot_app_host.md` §5.5: another service built from its own repository
