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
| The supervisor's stop | None: keep ends the game's process as it stands when it switches builds. |
| Who builds this | It is written on the laptop and built, tested, merged and deployed by the BlackWorld session (§7). |

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
| Ports | TCP 4080 (the site, the API, the client's updates); UDP for the game and the time on the ports `--game-port` and `--time-port` name (14300 and 14123 beside the gateway, which listens on the game's own 4300 and 4123 and forwards, §5) |
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

on `repo/battle-cities`, and a push of `main` of a repository in `cluster.deploy_on_push` is told to a member (§4.4). A delivery that is not signed, or any delivery to a gateway with no secret, is answered `401` and logged; one that is signed and is no push of a branch (a ping, a tag, a deleted branch) is answered `204`. The route is `web/src/hooks/http.rs`; it is reached with no session, on every host, and the signature is its only credential.

### 4.2 The watch

A webhook can be lost. Every 60 seconds, from its first member socket on, the gateway tells a member of each repository in `cluster.deploy_on_push` with no commit named (`Gateway::watch_pushes`). The member reads origin itself (`Agents::list`, which fetches nothing), so the watch alone deploys a push within a minute, and a gateway with no webhook secret still deploys.

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

The gateway is the one that knows of a push, and it tells one ready `zend` member: `POST /v1/deploys/pushed` over that member's socket, `{"repo", "commit"}` from a webhook or `{"repo", "attempted"}` from the watch (`web/src/hooks/pushes.rs`). The request carries `X-Tokera-Forwarder`, which the gateway removes from every request it passes on and a member's own port removes from any peer that is no gateway, so the member believes it is the gateway's own (`zend/src/api/deploys.rs`). One request reaches one member, so one member acts on each word.

The member reads origin's `main` and does this (`zend/src/mcp/tools/service/on_push.rs`):

| Origin's `main` is | What is done |
|---|---|
| The commit the gateway says it last told of (`attempted`) | Nothing: it was tried, whatever came of it |
| The commit the machine's installed binary was built from, by keep's state | Nothing |
| Anything else | `service_deploy` of each service built from the repository (`battlecities` from `battle-cities`), to the machine the cluster builds on first |

It answers the commit it read, and the gateway remembers it as told (`Told`), which is what keeps a commit whose build fails from being built again every minute. `service_deploy` then answers with the deploy already under way for that commit, or replaces a build of an older one (§4.3), so the word is safe to hear twice, from two gateways or after a gateway restarts.

The deploy acts as nobody. keep answers the member a request came from and the gateway drops a caller's name on the way to it, and what is built is `origin/main` or keep refuses it.

Not on a push: `motherd`, `web` and `mcp`. Their deploys stop models or the gateway and stay a person's decision.

keep ends the old server's process when it switches to the new build, with no request to stop first (John, 17 November 2026): what the game changed since its last save, a few seconds at most, is not kept across a deploy.

## 5. The game's datagrams

The game's client speaks UDP to port 4300 and reads the time from port 4123. The gateway forwards both (`web/src/udp_forward/`, `cluster.udp_forward` in `web.yaml`).

- **A flow** is one client address and port on one of the two ports. The gateway opens one socket of its own for it, connected to the member's port, and copies datagrams both ways. A flow that carries nothing for 60 seconds is closed.
- **The member** a new flow is given to is chosen by a hash of the client's address and port (FNV-1a) over the app's members that serve, in node order, so players spread over them and one player's datagrams always reach the server that holds their session. A flow stays with its member while the member stands; when the member has left, the flow ends at its next announcement and the client's next datagram starts one to a member that serves.
- **Who the client is.** The server sees the gateway's socket, not the client. Before it forwards a flow's first datagram, and every 30 seconds after, the gateway tells the member: `POST /api/v1/flows` on the member's stream, `{"via": "<the flow socket's own address and port>", "client": "<the client's address>"}`, with the header `X-Tokera-Forwarder: 1`. The server then counts its per-address limits and writes its security log by the client. An announcement stands 120 seconds.
- **Trust.** `X-Tokera-Forwarder` and `X-Tokera-Scope` are among the headers the gateway removes from every request it passes on (`proxy::IDENTITY_HEADERS`), so only the gateway itself can say them.
- **Limits.** At most 8,192 flows a port, and 64 from one client address; past either a new flow's datagrams are dropped and counted. A datagram is at most 2,048 bytes, and at most 256 wait for a flow's task.
- **Ports.** Each forwarded port names where the gateway listens and where each member does. On BlackWorld the gateway and the game share a machine, which binds one port once: the gateway listens on 4300 and 4123, and the game is run with `--game-port 14300 --time-port 14123`.

The member's address for a flow is the address its cluster socket came from, which the hub keeps for each socket (`Hub::serving`). A member whose address is not known is not given a flow. The client's server list names the gateway (`battlecities.net`, and the gateway's LAN name).

## 6. Signing in and tokens

**The sign-in domain.** `battlecities.net` signs in on a cookie domain of its own. It is one line in the gateway's `secrets/auth.yaml`, which is not in the repository, beside Yen Bots':

```yaml
domains:
  - {cookie_domain: .battlecities.net, redirect_uri: https://battlecities.net/auth/callback}
```

and that callback registered on the same Google client. No code: `/auth/*` is served on every host, and a site's routes never require sign-in, so every page and public read of the game is reached with none, and who a signed-in visitor is goes to the game on `X-Tokera-*`.

**Tokens for the game's API.** A site may name the scopes a bearer token for its `/api` may be granted (`sites[].api_scopes` in `web.yaml`; the game names `read`, `chat`, `play`, `manage`, `account`, `admin`). For such a site the gateway's authorization server has a second resource beside `/mcp`, `https://<host>/api` (`web/src/auth/mcp/`):

- `GET /.well-known/oauth-protected-resource/api` names the resource, this origin as its authorization server, and the scopes. A site that names none answers `404`.
- `/oauth/authorize` with `resource=https://<host>/api` and `scope=<some of them>` shows the consent for that site and those scopes. A scope the site does not name, or none, is refused. Anyone who signs in may allow it: the role table says who may use Zen, and the game's own accounts say what a person may do in the game.
- The token carries the scopes allowed (`Grant::scopes`), and `/oauth/token` answers them as `scope`. A token for `/mcp` is the bytes it always was.
- On a request under that site's `/api` with no session, the gateway reads the token (`Auth::api_grant`): the person it was granted by is stated upstream as a session's is, and its scopes on `X-Tokera-Scope`, space-separated, less any the site no longer names. The header is always sent for such a token, since the game reads a caller with no scope header as a browser's session, which may do everything. The token itself is not passed on.
- The token is nobody anywhere else: on another path, on another site, as a `/mcp` token, or presented by a backend. A `/mcp` token is nobody on an API.

## 7. Where the work stands

**Written here, not compiled or run** (the laptop builds nothing of this repository; the BlackWorld session builds, tests and merges):

- The app's name among the services and its build kind (`cluster/src/gateway/hub/constants.rs`, `cluster/src/gateway/admission.rs`, `cluster/src/version/info.rs`).
- The deploy engine's service, repository and package (`zend/src/mcp/tools/service/mod.rs`, `deploy.rs`, `zend/src/mcp/tools/service_control.rs`), with their tests.
- One deploy of a commit (§4.3): `pending.rs` and its tests, its use in `service_deploy`, and the commit an agent records on a run (`agent/src/jobs.rs`, `wire.rs`).
- The webhook's signature and body (§4.1): `web/src/hooks/github.rs` and its tests. The signature test is GitHub's own published example.
- The supervisor's entry and the site's route (`keep/keep.yaml`, `web/web.yaml`), and the two headers added to the list the gateway removes (`web/src/proxy.rs`).

**Checked when it was first built** (BlackWorld, 17 November 2026; the agent, cluster, zend and web suites pass, and clippy and fmt are clean):

- The commit a run records was stale: `Jobs::start` resolved the branch with `Repo::resolve(&Rev::Branch(..))`, which names the local branch, while the fetch moves only origin's copy and the store reads that first. It now records the store's own base (§4.3), and a run on `main` after a push is the test that holds it.
- The agent builds in the checkout itself (`motherd/agent.yaml` names `D:/agent/battle-cities`), and `server_assets/` and `target/` are ignored there, so the service's working folder is that checkout.
- `service_deploy`'s and `service_control`'s descriptions name one more service. The tool calibration reads them, so the next calibration pass after the deploy recalibrates those two tools.
- `web/src/roles.rs`'s assertions of the `battlecities` site hold with its route.

**Written on the laptop after that, not compiled or run** (branch `aw/battlecities-gateway`; BlackWorld builds, tests, merges and deploys):

- The webhook's route and the telling of a member, with the watch (§4.1, §4.2, §4.4): `web/src/hooks/http.rs`, `pushes.rs`, `Gateway::watch_pushes`, `cluster.deploy_on_push`; `zend/src/api/deploys.rs` and `zend/src/mcp/tools/service/on_push.rs`.
- The forwarder (§5): `web/src/udp_forward/` (`pick`, `table`, `flow`), `cluster.udp_forward`, the address the hub keeps for each member's socket and `Hub::serving`. `flow.rs`'s tests carry real datagrams over loopback.
- Tokens for a site's API (§6): `sites[].api_scopes`, `Grant::scopes`, `authorize::Api`, `Auth::api` and `Auth::api_grant`, the scope stated upstream in `proxy::forward`.
- `web/web.yaml` names the pushes to deploy, the ports to forward and the game's scopes; `keep/keep.yaml` runs the game on ports 14300 and 14123.

**To check when it is first built:**

- A service member that joins while another already serves starts as syncing; a service replicates nothing, so it should stop being so at the next receipt tick. `Hub::serving` lists only members that are routable, so a second game member that stayed syncing would be given no flows.
- `transport::send` sends no `Host` header. The game's routes need none; if hyper's client refuses a request without one, the two requests the gateway makes of its own (`/v1/deploys/pushed`, `/api/v1/flows`) need it set.
- The gateway binds UDP 4300 and 4123 on `server.bind`'s address at start, and stops the start if either is taken.
- The first watch tick after the gateway and a member with these changes are both up deploys the game to the machine the cluster builds on first, since nothing is installed there yet.

**What is left to a person:**

- On the gateway's machine: the `domains` line in `secrets/auth.yaml` (§6), and `secrets/github-hook.key` holding the webhook's secret (§4.1).
- John: the webhook in GitHub (§4.1) with that secret, and `https://battlecities.net/auth/callback` registered at Google (§6).

## Related documents

- `docs/replicated_substrate.md` §3.5–§3.11: keep, trials, the proof line
- `docs/jobs.md` §6: `service_deploy` and the jobs it is made of
- `docs/mcp_file_tools.md` §5: the bus, `repo/<repo>` and the gates
- `docs/zenj.md` §3.1: another service that is a cluster member of its own
- `docs/yenbot_app_host.md` §5.5: another service built from its own repository
