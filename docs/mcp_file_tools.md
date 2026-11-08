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
- **Two transfers that commit** (§3.6): `file_download_and_commit` lands what a
  URL serves, and `file_upload_and_commit` lands bytes the caller holds. Each
  is one commit through the same path as the write. Their worker twins,
  `file_download` and `file_upload`, move a file between a URL and a worker's
  checkout. None of them passes the file's bytes through a model: a page Claude
  wrote is not typed out again to reach the repository.

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

`file_read` also takes `paths` in place of `path`: up to 64 files in one call,
each answered in full, in order, under its path, with a total of 4 MiB. A file
that is not UTF-8 is answered as `{path, base64, bytes}`, never refused, so a
reader can take a picture as well as a page. This is what an app preview loads
itself from (`docs/yenbot_preview.md` §3): the framework and an app's files in
one call rather than one call each.

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
author: {name: John, email: <John's address>}   # every repository's, unless it names its own
repos:
  - name: candle
  - name: battle-cities
  - name: journals
  - name: yenbot
    author: {name: Yen, email: <Yen's address>}
    gate: yenbot                # the service that checks a change before the branch moves
```

The addresses live in the manifest on the machine, never in this public
repository.

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
request-path invariant 1). `CLAUDE.md`'s MCP section lists it, and the two
`_and_commit` transfers below with it.

### 3.6 The transfers

A file Claude did not type, or should not type again, reaches a repository
without passing through a model:

| Tool | Where | Moves | Answers |
|---|---|---|---|
| `file_download_and_commit {repo, branch?, message, files: [{url, path}]}` | connector | what each URL serves → one commit | live, with the commit, or refused |
| `file_upload_and_commit {repo, branch?, message, files: [{path, base64}]}` | connector | bytes the caller sends → one commit | live, with the commit, or refused |
| `file_download {url, path}` | worker | what the URL serves → a file in the worker's checkout | path, bytes, content type, SHA-256 |
| `file_upload {path, url, method?, content_type?}` | worker | a file in the worker's checkout → the URL, by `PUT` (default) or `POST` | the status and the response's first 4 KiB |

- **The `_and_commit` pair is `file_edit_and_commit`'s path from step 3 on**
  (§3.2): one commit on the branch's tip, by the repository's author, gated,
  pushed compare-and-swap, raced at most three times, announced on
  `repo/<repo>`. All or nothing: one URL that fails or one file the gate
  refuses, and nothing lands.
- **Download** is for what is already at a URL Zen can reach: a picture
  `image_generate` or `image_edit` answered with, a font or a library from a
  CDN, a raw file from a public repository. It is not a way to reach a page on
  claude.ai, which sits behind its sign-in: such a page sends its own bytes
  with `file_upload_and_commit` (`docs/yenbot_preview.md` §4).
- **Upload** on the connector is for bytes the caller holds and Zen cannot
  fetch: a photo dropped on an app preview, a page a preview built. At most
  8 MiB a file and 16 MiB a call, as base64.
- **The worker pair** keeps a file out of a worker's context. A worker that
  fetched a file with `http_request` read the whole body into its context and
  wrote it back out; `file_download` writes it to the checkout and answers
  only its size and digest, and the file goes into the repository with the
  worker's other changes, through the same gate. `file_upload` is the same in
  the other direction: a built file sent to a service, never read into the
  context.
- **No credentials.** A transfer fetches and sends as nobody. A source that
  needs a sign-in is reached with the workers' HTTP tools and a credential
  saved for the site (§6.3).

### 3.7 The address guard

A transfer runs on a member's machine, inside its network, where keep's API on
`127.0.0.1:8092` and every daemon behind the gateway believe identity headers
from loopback (§4.1). A URL that reached them would be a request forged from
inside. So every transfer's address is checked, and the check is the
connection's, not the name's:

- **Resolved, then pinned.** The host is resolved once and the connection is
  made to the address that was checked, so a name that resolves elsewhere a
  moment later (rebinding) reaches nothing new.
- **Public addresses only.** Refused: loopback, private (RFC 1918, IPv6 ULA),
  link-local, carrier-grade NAT (100.64/10), multicast, unspecified,
  documentation and benchmarking ranges, and an IPv4 address mapped or
  translated into IPv6. Only `https` and `http`, only the default ports and
  those named in the URL.
- **Every redirect again.** At most five, each checked as the first was.
- **Bounded.** 30 seconds in all, and the size caps above enforced as the
  bytes arrive: a body past its cap is abandoned, not read and then refused.

This is stricter than `tcp_session_open`, which keeps loopback open for local
test services; the guard is its own module and does not loosen for them.

### 3.8 Bytes in a commit

`file_edit_and_commit` stays text-only. The transfers commit any bytes, and
the repository decides what may hold them: a gated repository's gate sees every
path and its size (§5.4), and yenbot's refuses binary outside the paths its
rules name for it (`library/`, an app's `assets/`), a file past its size, and a
file whose content does not match its extension.

## 4. Service keys

A service on the gateway's machine — yenbot first — needs to act as itself:
to call Zen's model and image routes, and to use the bus. The daemons know
callers only by the gateway's identity headers, and the only key the gateway
honours is the operator's `ops_key`, admin on every route (`web/web.yaml`).

The gateway's `cluster:` block gains (`web/src/cluster/service_key.rs`):

```yaml
cluster:
  service_keys:
    - name: yenbot                       # its identity is the subject service:yenbot
      key_file: secrets/yenbot.key       # 64 hex digits, as `web --new-ops-key` writes
      routes: [/v1/chat/completions, /v1/models, /v1/image/generate, /v1/image/edit,
               /gateway/google/token]
      topics:
        subscribe: ["repo/yenbot", "gate/yenbot"]
        publish: ["gate/yenbot/result/*", "service/yenbot/*"]
```

A request bearing a service key as `Authorization: Bearer`, from a trusted
backend (`server.trusted_backends`), is the key's identity on the routes the
key names (each with everything under it) and nobody elsewhere. The identity is
provider `service-key`, subject `service:<name>`, no email — a subject no
sign-in can produce — and `host/roles.yaml` grants it a role by that subject
(`sub: "service:yenbot"`). The key never leaves the gateway: the request is
forwarded without its `Authorization` header, like the ops key's. It is a file
only the service and the gateway read, never logged, never on a command line.

**A service on a member, not the gateway's machine,** states the same
identity without a key. npcd's drones call zend as Zen
(`docs/zen_memory_and_guides.md` D8) over their member's cluster socket,
where a member is trusted to name its caller as zend names one when it calls
npcd: `x-tokera-user: service:zen`, `x-tokera-provider: service-key`, no
email (`service_key::subject`). `host/roles.yaml` gives `service:zen` the
role `user`, and the memory binds it to the person `zen` by its key alone
(`docs/memory_backbone.md` §3.4).

### 4.1 A known hole: identity from loopback

The daemons behind the gateway — yenbot among them, and keep's HTTP API on
`127.0.0.1:8092` — believe `X-Tokera-*` identity headers on any request from
loopback, because the gateway is what reaches them there. A Zen worker can
also reach them there: `tcp_session_open` refuses private addresses but keeps
loopback open (`zend-tools/src/tools/tcp_session/open.rs`, for local test
services), so a task can open a raw socket to `127.0.0.1` and write any
identity it likes.

This is accepted for now (2026-11-06): only John and Yen start tasks. When it
is closed, the fix is a key per site upstream that the gateway sets and strips
(`X-Tokera-Upstream-Key`), believed by the site in constant time, rather than
trust in the address.

### 4.2 Tool sessions close with their conversation

A worker's sessions — `tcp_session_*` first, and every other kind the session
registry holds (`zend-tools/src/state/sessions.rs`: HTTP, TLS, UDP, SSH,
telnet, SQL, remote file systems, code) — belong to the conversation that
opened them. When that conversation is archived (`zend/src/mcp/archiving.rs`:
by `thread_archive`, or by the sweep a day after the worker last moved), every
session it opened is closed and removed: the socket dropped, the buffers
freed, an HTTP session's credential released. An archived worker is never
continued, so nothing could use them again, and a socket that outlived its
conversation is a connection nobody is watching. A finished worker that is not
yet archived keeps its sessions, as `repo_task_reply` may continue it.

The registry records each session's conversation (its context's lineage) so
the archive can find them; `close_conversation(lineage)` closes them all at
once and answers how many it closed, which the archive's log line carries.

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

- **Services** connect from a trusted backend with their service key and may
  subscribe and publish only on the topics their key names (§4).
- **Cluster members** use it over the cluster socket they already hold to the
  gateway (`docs/replicated_substrate.md` §5.8), on any topic, as members are
  already trusted there — on a stream to the gateway's `group:bus`, never the
  stream towards their own app. That one also carries every request a
  member's listener forwards up for a client, so being on it vouches for the
  client, not the member; the bus routes on the gateway's own router admit
  service keys alone.

Beside the socket, for a caller that publishes rather than listens
(`web/src/bus/http.rs`), the same callers under the same rules:

| Route | Body | Answer |
|---|---|---|
| `POST /gateway/events/publish` | `{topic, payload}` | `{reached}`: how many subscribers it reached |
| `POST /gateway/events/request` | `{topic, payload, reply, timeout_ms}` | the first payload published on `reply` (200); `no_subscriber` (404) when nothing listens on `topic`; `timeout` (504). At most 30 s. |

A request subscribes to `reply` before it publishes, so an answer cannot
arrive unseen. Each gateway has its own bus: a member publishes to every
gateway it holds a socket to, and asks a request of each until one answers
other than `no_subscriber` — the gate is subscribed on one of them.

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
repository (§3.3) moves a branch, zend pushes the commit to a candidate
branch, `zen/candidate/<id>`, and asks on `gate/<repo>`
(`zend/src/repo_events/announced.rs`):

```json
{"id": "3f2a…", "ref": "refs/heads/zen/candidate/3f2a…", "commit": "4f1c…",
 "base": "a3f9…", "branch": "main", "paths": ["apps/recipes/pages/home.js"]}
```

`base` is what the branch held (the push's lease), `null` for a new branch.
A branch's deletion is asked too, with nothing to fetch: `"ref": null,
"commit": null, "delete": true` (a move carries `"delete": false`). Only a
subscriber naming `gate/<repo>` itself answers: a gateway whose only listeners
are wildcard watchers says `no_subscriber` at once, so the asker goes on to
the gateway that has the gate.
The candidate branch is deleted whatever the answer. Every push Zen makes goes
through one pusher per repository (`ToolContext::pusher`), so this is where
each is gated and, once it lands, announced (§5.3).

The gate fetches the candidate, checks it, and answers on
`gate/<repo>/result/<id>`:

```json
{"ok": false, "reason": "apps/recipes/pages/home.js:14 uses a literal colour; use a token"}
```

zend waits for that answer — ten seconds at most — and moves the branch only
on `ok`. A gate that is not connected, or does not answer in time, refuses
the change rather than letting it through unchecked — and so does a daemon in
no cluster, which has no bus to ask on. A refusal comes back to
whoever pushed as every spec of the push rejected by the remote with the
reason (`Rejection::Remote`), as a server hook's would, so `git_push`,
`file_edit_and_commit` and an agent job each report it the way they report
any refusal.

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

The gateway reaches a site's for a member (`web/src/cluster/site_logs.rs`):
`GET /gateway/logs` lists the sites, and `GET /gateway/logs/<site>/v1/logs?…`
and `…/v1/logs/sources` go on to the site's upstream with the caller's
identity headers. `/v1/logs/sources` answers
`{"sources": [{"name", "lines", "newest"}]}`; `/v1/logs` answers as a member
does, `{"lines", "matched", "complete"}`.

### 6.2 The tools

| Tool | What it does |
|---|---|
| `log_list` | Every source: its name, what it is (a member's tenant, a site, an app), how many lines it holds and its newest line's time. |
| `log_read {source, since?, level?, limit?}` | A source's newest lines, oldest first. |
| `log_grep {grep, source?, level?, since?, limit?, node?}` | A regular expression over one source, or every source when none is named; each line under its source. `node` names one member, as `cluster_logs` did, for Admins. |

`log_grep` with `source: "cluster"` is what `cluster_logs` answers, so the
connector offers `cluster_logs` no more and the connector's guides name
`log_grep` (`zend/src/mcp/tools/logs/`). A worker keeps its own
`cluster_logs` (`zend-tools/src/tools/ops/cluster.rs`): its catalog,
calibration and grants are a worker's, and a guide's prompt for a worker
names it.

### 6.3 What is not a tool

Logs are tools because they are read first and most, often while a task is
running, and span the cluster and the sites alike. Anything else a site holds
— its records, jobs, accounts, usage — is that site's API, which Claude calls
from a Zen task with the workers' HTTP tools (`http_session_open`,
`http_request`) and a credential saved for the site
(`docs/yenbot_app_host.md` §6.3). There is no connector tool per site or per
kind of thing a site holds. How a page of a Yen Bots app looks is seen in its
preview on claude.ai, which loads the app through the file tools and publishes
through them (`docs/yenbot_preview.md`); there is no tool that renders a page.

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
- **The transfers**, against a local HTTP server the guard is told to admit in
  the test alone: a download and an upload each land byte for byte; two files
  where the second URL fails commit nothing; a body past its cap is abandoned
  at the cap; `file_download` answers the size and SHA-256 and the file is in
  the checkout; `file_upload` sends the file's exact bytes.
- **The guard**, as a pure function over addresses and as a connection:
  every refused range above, an IPv4-mapped loopback, a name that resolves
  public then private (the pinned address is the one connected), a redirect
  to a private address refused, a sixth redirect refused.
- **`file_read` with `paths`**: several files in order, a binary one as
  base64, the total cap.
- **The bus**: subscribe, publish, a topic outside a key's scope refused, a
  wildcard, a slow subscriber dropped, a member's publish over the cluster
  socket delivered to a service.
- **The gate**: an ok moves the branch; a refusal leaves it where it was and
  deletes the candidate; a gate that is not connected, or answers after ten
  seconds, refuses; a worker task's push to a gated repository is gated
  too.
- **`repo/<name>`**: a push through each of the three paths publishes once,
  with the commit and the paths.
- **Sessions and archiving**: a worker that opened a TCP and an HTTP session
  and finished keeps both; archiving it closes both and leaves another
  conversation's sessions open; `close_conversation` answers the count.
- **Log sources**: `log_list` over the cluster and a site with sub-sources;
  `log_grep` across all of them, and with `source: "cluster"` answering what
  `cluster_logs` did; a site refusing a person it does not admit.

## 8. What changes

| Where | Change |
|---|---|
| `zend/src/mcp/tools/` | `file_read`, `file_list`, `file_search` (the workers' implementations, offered to the connector) and `file_edit_and_commit`. |
| `zend/src/mcp/tools/` | `file_download_and_commit`, `file_upload_and_commit`; `file_read` with `paths`. |
| `zend-tools/src/tools/file/` | `file_download`, `file_upload` (§3.6). |
| `zend-tools/src/net/` | The address guard (§3.7), its own module, used by every transfer. |
| `zend/src/file_commit/` | A change carrying bytes, for the transfers (§3.8). |
| `zend-vfs/src/workspace.rs` | `author:` in the manifest, the default and per repository. |
| zend's push paths | Publish `repo/<name>` after each push. |
| `web/src/` and `web/web.yaml` | `service_keys:`; the bus and `/gateway/events`. |
| `cluster/` | A publish (and subscribe) message on the member's socket. |
| `zend/src/mcp/tools/` | `log_list`, `log_read`, `log_grep` in place of `cluster_logs`. |
| `web/web.yaml`, `web/src/` | `logs` on a site; the gateway gathering a site's `/v1/logs` beside the members'. |
| `zend-tools/src/state/sessions.rs`, `zend/src/mcp/archiving.rs` | Each session records its conversation; archiving a worker closes every session it opened (§4.2). |
| `zend-vfs/src/workspace.rs`, zend's push paths | `gate:` in the manifest; every push to a gated repository through `gate/<repo>` first. |
| `CLAUDE.md` | `file_edit_and_commit` as the request-path budget's exception. |
| `mcp/src/guides/` | The connector's guides name the file and log tools, `log_grep` where they named `cluster_logs`, and when a task is the better tool. |

## 9. Order of work

1. `author:` in the manifest; `file_edit_and_commit` on the git layer, with
   its tests against real repositories.
2. The read tools on the connector.
3. Service keys at the gateway; tool sessions closed when their conversation
   is archived.
4. The bus, its WebSocket and the members' publish.
5. zend publishing `repo/<name>` on every push, and gating the pushes of a
   gated repository.
6. Log sources and the log tools, replacing `cluster_logs`.
7. The address guard, then the transfers on it: the worker pair, then the
   `_and_commit` pair; `file_read` with `paths`.
