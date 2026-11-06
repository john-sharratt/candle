# File, log and site tools in the Zen MCP, and the gateway's pub/sub

**Status: design, not built.** This document is authoritative for the file,
log and site tools the Zen connector offers, the gateway's publish/subscribe
bus and the gates it carries, and the service keys they rest on. Its first consumer is Yen Bots
(`docs/yenbot_app_host.md`), which goes live a second after a commit lands.

## 1. What it is for

From claude.ai, Zen can start tasks, read the cluster and run deploys, but it
cannot read or change a file: `file_read` and `file_edit` are tools of zend's
own workers, reached only by delegating a whole task. For small, fast work — a
word on a page, a colour, a new page in one of Yen's apps — a task is the
wrong size. This adds the files themselves to the connector:

- **Reads**: `file_read`, `file_list`, `file_search`, the same tools zend's
  workers use.
- **One write**: `file_edit_and_commit`. Every call is one commit, pushed to
  origin. There is no uncommitted state between calls, so nothing is ever half
  edited and every change can be reverted.

And so that whatever serves a repository can react the moment a commit lands,
the gateway gains a **publish/subscribe bus**, and zend publishes on it after
every push it makes.

## 2. The reads

`file_read {repo, path, page?, line?}`, `file_list {repo, path, page?}` and
`file_search {repo, pattern, path?}` answer exactly as zend's worker tools of
the same names do (`zend-tools/src/tools/file/`), against the repository's
default branch as origin holds it — the branches zend already watches and
fetches within seconds of a move (`docs/zend_branch_ingest.md` §4), and that
`file_edit_and_commit` advances itself. They take the connector's repository
enum, run no model, and answer within the MCP request-path budget (30 ms at
p95, `CLAUDE.md`).

## 3. `file_edit_and_commit`

### 3.1 The call

```text
file_edit_and_commit {
  repo,                             # a workspace repository
  branch?,                          # default: the repository's default branch
  message,                          # the commit message
  changes: [
    {path, edits: [{old, new}]}     # exact-string edits, applied in order
    {path, content}                 # create the file, or replace it whole
    {path, delete: true}
    {path, move_to}
  ]
}
```

Text files only. A change that would write bytes that are not UTF-8, or a path
whose existing content is not, is refused.

### 3.2 What it does

1. **Base.** Fetch the branch from origin; its head is the base.
2. **Apply.** Every change against the base's tree, in order. An edit's `old`
   must occur exactly once in the file as it then stands; `content` replaces
   or creates; `delete` and `move_to` must name a file that exists, and
   `move_to` one that does not. **All or nothing**: if any change does not
   apply, nothing is committed, and the answer names the change, why, and the
   file's current text around where the edit was looked for.
3. **Commit**, without a checkout: the changed blobs and trees written into
   the repository's object store and a commit on the base, by the git layer
   (`docs/zend_git.md` — it writes commits and moves refs without touching any
   working tree). The author is the repository's (§3.3).
4. **Gate**, for a repository that names one (§3.3): push the commit to a
   candidate ref, `refs/zen/candidate/<id>`, publish `gate/<repo>` on the bus
   (§5.4) and wait for the gate's answer. Refused — or no answer within ten
   seconds — deletes the candidate and fails the call with the gate's reason;
   the branch never moved. A repository with no gate skips this step.
5. **Push**, compare-and-swap: origin's branch moves from the base to the
   commit (`docs/zend_git.md` §1, every ref move names the value it replaces).
6. **Race.** If the push is refused because the branch moved, fetch the new
   head and go back to step 2 with the same changes. Edits are matched by
   their text, so re-applying them onto someone else's newer commit is safe;
   if one no longer applies — its `old` text was changed by that commit — the
   call fails as in step 2, naming it. At most three attempts.
7. **Publish** `repo/<repo>` on the gateway's bus (§5.3).

The call is all or nothing: the change is on the branch, checked by the
repository's gate, or it is nowhere. The answer: the commit id, the branch,
the paths it changed, the attempt it landed on — or the step that refused it
and why.

### 3.3 Authors

Each repository names the author its commits through the connector carry, in
the workspace manifest (`D:\prog\workspace.yaml`, read by
`zend-vfs::workspace`):

```yaml
author: {name: John, email: johnathan.sharratt@gmail.com}   # every repository's, unless it names its own
repos:
  - name: candle
  - name: battle-cities
  - name: journals
  - name: yenbot
    author: {name: Yen, email: yensharratt@gmail.com}
    gate: yenbot                # the service that checks a change before the branch moves
```

The committer is the author. A commit carries no attribution to a model or an
assistant (`CLAUDE.md`, "No AI attribution anywhere"). A repository's `gate`
names the service that answers its `gate/<repo>` requests (§5.4); every push
Zen makes to it — this tool's, a worker task's, an agent job's — goes through
the gate.

### 3.4 Access

Coarse, at the connector: the Zen connectors that exist are the household's,
and every workspace repository is offered to them, every branch included.
Narrowing it — per account, per repository — waits for a connector that is
someone else's.

### 3.5 Latency

A push to origin cannot answer within 30 ms. `file_edit_and_commit` is the
request-path budget's documented exception, as the deploy tools are: it
answers when the push has landed, about one to two seconds, and holds no lock
another call takes while it waits on the network (`CLAUDE.md`, MCP
request-path invariant 1). `CLAUDE.md`'s MCP section lists it.

## 4. Service keys

A service on the gateway's machine — yenbot first — needs to act as itself:
to call Zen's model and image routes, and to use the bus. The daemons know
callers only by the gateway's identity headers, and the only key the gateway
honours is the operator's `ops_key`, admin on every route (`web/web.yaml`).

The gateway's config gains:

```yaml
service_keys:
  - name: yenbot
    key_file: secrets/yenbot.key
    identity: "service:yenbot"
    routes: [/v1/chat/completions, /v1/models, /v1/image/generate, /v1/image/edit,
             /gateway/google/token]
    topics:
      subscribe: ["repo/yenbot", "gate/yenbot"]
      publish: ["gate/yenbot/result/*", "service/yenbot/*"]
```

A request bearing a service key as `Authorization: Bearer`, from a trusted
backend (`trusted_backends`), is forwarded with that identity on the routes
the key names and no others; `host/roles.yaml` grants the identity what the
daemons should let it do. The key is a file only the service and the gateway
read, never logged, never on a command line.

### 4.1 Upstream keys: the other direction

A service key proves the service to the gateway; a site behind the gateway
needs the converse, proof that a request with identity headers came from the
gateway. Binding loopback does not give it: a Zen worker's `tcp_session_open`
can open a raw socket to `127.0.0.1` and write any header it likes. So a site
upstream carries a key:

```yaml
sites:
  - name: yenbot
    api: [{prefix: /, upstream: "http://127.0.0.1:8096", upstream_key_file: secrets/yenbot-upstream.key}]
```

The gateway strips `X-Tokera-Upstream-Key` from every incoming request and
sets it on each one it forwards to that upstream; the site believes
`X-Tokera-*` only alongside it, compared in constant time.

The hole is closed at its source as well: `tcp_session_open` refuses
loopback, link-local and the machine's own addresses, as a worker has no
business reaching a daemon on the machine it runs on by a path the gateway
does not see. keep's HTTP API on `127.0.0.1:8092` is the daemon this matters
for today.

## 5. The gateway's pub/sub

### 5.1 What it is

An in-memory bus in the gateway: a publisher sends a message on a topic,
every subscriber to that topic gets it. Nothing is kept — a subscriber that
was not connected missed it, and every subscriber is written to resynchronise
from the source of truth when it (re)connects. Topics are strings with `/`
segments; a subscription may end in `*`.

### 5.2 The socket

`GET /gateway/events` upgrades to a WebSocket. Frames are JSON:

```json
{"op": "subscribe",   "topics": ["repo/yenbot"]}
{"op": "publish",     "topic": "service/yenbot/published", "payload": {…}}
{"op": "message",     "topic": "repo/yenbot", "payload": {…}, "at_ms": 1793931066831}
{"op": "refused",     "topic": "repo/candle", "why": "not subscribable by service:yenbot"}
```

- **Services** connect over loopback with their service key and may subscribe
  and publish only on the topics their key names (§4).
- **Cluster members** publish over the cluster socket they already hold to
  the gateway (`docs/replicated_substrate.md`), on any topic, as members are
  already trusted there; they may subscribe the same way.

A slow subscriber is dropped rather than buffered without bound, and
reconnects; a message is small (a few hundred bytes), so a subscriber that
keeps up never notices.

### 5.3 `repo/<name>`

zend publishes `repo/<name>` after **every** push it makes to a workspace
repository — `file_edit_and_commit`, a worker task's commit, an agent job's
push — so a subscriber learns of each change by one path, whoever made it:

```json
{"repo": "yenbot", "branch": "main", "commit": "4f1c…", "paths": ["apps/recipes/pages/home.js"]}
```

A push made outside Zen (a developer pushing from a laptop) publishes nothing;
a subscriber that must see it also polls the branch, slowly.

### 5.4 `gate/<repo>`: a check before the branch moves

The bus also carries a request and its answer. Before a push to a gated
repository (§3.3), zend publishes:

```json
{"op": "publish", "topic": "gate/yenbot",
 "payload": {"id": "01JD…", "ref": "refs/zen/candidate/01JD…", "commit": "4f1c…",
             "base": "a3f9…", "paths": ["apps/recipes/pages/home.js"]}}
```

The gate fetches the candidate, checks it, and answers on
`gate/<repo>/result/<id>`:

```json
{"ok": false, "reason": "apps/recipes/pages/home.js:14 uses a literal colour; use a token"}
```

zend waits for that answer — ten seconds at most — and moves the branch only
on `ok`. A gate that is not connected, or does not answer in time, refuses
the change rather than letting it through unchecked.

## 6. Logs and sites in the connector

### 6.1 Log sources

The connector reads logs today with `cluster_logs`: the gateway gathers
`GET /v1/logs` from the cluster's members (`host/src/logs.rs`,
`zend/src/mcp/tools/cluster_logs.rs`), each line a `LogLine` — time, level,
target, message, tenant — searched by `grep`, `level`, `tenant`, `since` and
`limit`. That contract is general, so it becomes the contract of a **log
source**: anything that answers `GET /v1/logs` with those parameters (plus
`source`) and `GET /v1/logs/sources`, listing what it holds.

The sources are:

- **The cluster** — every member's tenants, gathered by the gateway as now.
- **Sites the gateway lists as log sources** — `logs: true` on a site in
  `web/web.yaml` — each with the sub-sources its `/v1/logs/sources` names:
  Yen Bots is `yenbot` and one `yenbot/<app>` per app
  (`docs/yenbot_app_host.md` §18).

### 6.2 The tools

| Tool | What it does |
|---|---|
| `log_list` | Every source: its name, what it is (a member's tenant, a site, an app), how many lines it holds and its newest line's time. |
| `log_read {source, since?, level?, limit?}` | A source's newest lines, oldest first. |
| `log_grep {grep, source?, level?, since?, limit?, node?}` | A regular expression over one source, or every source when none is named; each line under its source. `node` names one member, as `cluster_logs` did, for Admins. |

`log_grep` with `source: "cluster"` is what `cluster_logs` answers today, so
`cluster_logs` is removed and the guides that name it name `log_grep`.

### 6.3 What is not a tool

Logs are tools because they are read first and most, often while a task is
running, and span the cluster and the sites alike. Anything else a site holds
— its records, jobs, accounts, usage — is that site's API, which Claude calls
from a Zen task with the workers' HTTP tools (`http_session_open`,
`http_request`) and a credential saved for the site
(`docs/yenbot_app_host.md` §6.3). There is no connector tool per site or per
kind of thing a site holds. How a page looks is not a tool either: the person
pastes a screenshot into the chat.

### 6.4 Who they act as

Each tool calls through the gateway as the person whose connector it is, with
their identity headers; the source or site decides what that person may read
— yenbot answers its admins only. The tools run no model and answer within the
request-path budget.

## 7. Tests

- **Edits, against raw expected bytes**: one edit, several in order, an `old`
  found twice or not at all, create, replace, delete, move, a change set where
  the last change fails and nothing is committed — each against a real
  repository in a temporary directory, its resulting tree compared byte for
  byte.
- **The race**: a commit pushed between the base and the push; the call
  re-applies onto it and lands on attempt two; a commit that changed an
  edit's `old` text makes the call fail and name it.
- **Authors**: the manifest's default, a repository's own.
- **The bus**: subscribe, publish, a topic outside a key's scope refused, a
  wildcard, a slow subscriber dropped, a member's publish over the cluster
  socket delivered to a service.
- **The gate**: an ok moves the branch; a refusal leaves it where it was and
  deletes the candidate; a gate that is not connected, or answers after ten
  seconds, refuses; a worker task's push to a gated repository is gated
  too.
- **`repo/<name>`**: a push through each of the three paths publishes once,
  with the commit and the paths.
- **Log sources**: `log_list` over the cluster and a site with sub-sources;
  `log_grep` across all of them, and with `source: "cluster"` answering what
  `cluster_logs` did; a site refusing a person it does not admit.

## 8. What changes

| Where | Change |
|---|---|
| `zend/src/mcp/tools/` | `file_read`, `file_list`, `file_search` (the workers' implementations, offered to the connector) and `file_edit_and_commit`. |
| `zend-vfs/src/workspace.rs` | `author:` in the manifest, the default and per repository. |
| zend's push paths | Publish `repo/<name>` after each push. |
| `web/src/` and `web/web.yaml` | `service_keys:`; the bus and `/gateway/events`. |
| `cluster/` | A publish (and subscribe) message on the member's socket. |
| `zend/src/mcp/tools/` | `log_list`, `log_read`, `log_grep` in place of `cluster_logs`. |
| `web/web.yaml`, `web/src/` | `logs` on a site; the gateway gathering a site's `/v1/logs` beside the members'; `upstream_key_file` on a site upstream (§4.1). |
| `zend-tools/src/tools/tcp_session/` | `tcp_session_open` refuses loopback, link-local and the machine's own addresses (§4.1). |
| `zend-vfs/src/workspace.rs`, zend's push paths | `gate:` in the manifest; every push to a gated repository through `gate/<repo>` first. |
| `CLAUDE.md` | `file_edit_and_commit` as the request-path budget's exception. |
| `mcp/src/guides/` | The connector's guides name the file and log tools, `log_grep` where they named `cluster_logs`, and when a task is the better tool. |

## 9. Order of work

1. `author:` in the manifest; `file_edit_and_commit` on the git layer, with
   its tests against real repositories.
2. The read tools on the connector.
3. Service keys and upstream keys at the gateway; `tcp_session_open` off
   loopback.
4. The bus, its WebSocket and the members' publish.
5. zend publishing `repo/<name>` on every push, and gating the pushes of a
   gated repository.
6. Log sources and the log tools, replacing `cluster_logs`.
