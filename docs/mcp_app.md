# MCP: an aggregator over the apps' own servers

The remote MCP endpoint Claude reaches at `https://code.tokera.com/mcp` is an
**aggregator**. It holds no tools of its own. zend and npcd each serve their
own tools as an ordinary MCP server at `/v1/mcp`. The aggregator lists all of
those tools as one server and passes each call to the backend that lists its
tool, as the signed-in caller.

- **zend** serves the engineering tools: `zend_*`, `journal_*`, and the
  operations tools. They are zend's own. A task's turn streams in zend's
  process, `sub_run` calls back into the worker lifecycle, and the operations
  tools ride zend's cluster socket (§3).
- **npcd** serves Battle Cities' tools (§4). Each one lists characters, gives
  or manages a mission, messages a character, or reads or posts on a world's
  channel. Each tool is the route for the same act, called in-process.

There is no interface between the aggregator and the backends except MCP
itself. The protocol is written once, in the **`mcp-server`** crate: JSON-RPC,
both protocol eras, sessions, the sign-in challenge, and progress on an event
stream. zend, npcd and the aggregator each implement its `Server` trait, which
asks four things of an app: who it is, who is calling, where its sessions are
kept, and what its tools are.

The OAuth authorization server stays in the gateway
(`docs/claude_zend_gateway.md` §12). The aggregator is the protected resource
it names.

## 1. The request path

```
claude.ai / Claude Code
   │  POST /mcp  (Authorization: Bearer <gateway token>)
   ▼
web gateway (BlackWorld)  ── token → identity (auth::mcp) ── X-Tokera-* ──┐
   ▼                                                                      │
aggregator (AlienWorld, :8095)   believes X-Tokera-* only from the gateway │
   │                                                                      │
   │  POST http://<gateway>/v1/mcp   Host: code.tokera.com | bot.tokera.com
   │       Authorization: Bearer <the same token>                         │
   ▼                                                                      │
web gateway  ── the token names the caller, from a trusted backend (§2) ──┘
   ▼
zend / npcd members (cluster dispatch, as for any request)
```

The aggregator never reaches a member directly. Every request it makes goes
through the gateway's routing (stickiness, ownership, member choice) under
the caller's identity, exactly as a browser's request would. So each backend
checks the caller's role and ownership itself, and the aggregator decides
nothing about access.

## 2. Trust

- **gateway → aggregator.** The aggregator's config names the gateway's
  addresses in `trusted_gateways`: BlackWorld's `192.168.0.5` and `.4`,
  because the gateway's outbound LAN connection can leave from either. The
  aggregator believes the identity headers only from those peers, and only
  when the client's bearer token came with them. From anywhere else nobody is
  calling, and the client is sent to sign in.
- **aggregator → gateway.** The gateway's `server.trusted_backends` lists the
  aggregator's address (AlienWorld, `192.168.0.237`). From a listed backend,
  a request that carries a person's MCP access token is read as that person,
  on any path (`Auth::mcp_delegated`). A backend's own `X-Tokera-*` headers
  are still stripped on ingress. The address alone names nobody, so a process
  on that machine can act only for a person whose token it was handed.

The loop-back is plain HTTP on the LAN to the gateway's `:80` entrance, which
is the one Cloudflare's tunnel reaches, with the site named in `Host`.

## 3. The aggregator

**Its card** comes from configuration: the name, title, description,
instructions and icon a client shows. The backends' own cards are never shown.

**`tools/list`** asks every backend for its tools, in parallel and as the
caller, and lists them in backend order. It also keeps a note of which backend
lists which tool. A name an earlier backend already listed is left out. If a
backend cannot be listed, the answer is an error rather than a shorter list. A
client caches the list it gets, and a short one would leave it without that
backend's tools for the whole conversation.

**`tools/call`** goes to the backend that listed the tool. If no backend has
listed it yet, as after a restart, the tools are listed again first. The call
asks the backend for progress. Each `notifications/progress` the backend
streams becomes a step of the aggregator's own call, which the client sees as
a progress notification. The backend's result is the answer.

**Sessions.** Every backend is spoken to in the 2026-07-28 protocol, which
keeps no handshake session to lose. Each request carries two things in its
`_meta`: the caller's client, in `io.modelcontextprotocol/clientInfo`, and the
aggregator's own session id, in `com.tokera/session`. The backend keys what it
holds by person, client and that session (`mcp_server::stateless_key`). So two
claude.ai conversations of one person, each a session of the aggregator's, are
two sessions at zend, each with its own attached thread. The aggregator keeps
its own sessions in a file (`mcp/state/sessions.json`), and they survive a
restart.

**What a delivered answer changes is the backend's.** zend marks the journal
entries an answer handed over once that answer has left on its stream. When
the client goes before its answer, the aggregator drops its call to the
backend (`Call::abandoned`). zend then sees its own client gone, and leaves
the entries for the next call.

**A backend silent for a minute is gone.** A server on `mcp_server` says
something on a call's stream at least every 15 s. After 60 s with nothing, the
aggregator gives the call up and answers an error, rather than holding the
client on keep-alives of its own.

**One backend down costs only its own tools.** `tools/list` is still an
error, as above. A call to a tool whose backend is known goes straight there,
and a relisting keeps the routes of any backend that did not answer.

## 4. The backends

### zend (`/v1/mcp` on code.tokera.com)

zend's tools are the ones it always served (`zend/src/mcp/tools/`). Its
sessions are records in its substrate (`zend/src/mcp/state.rs`,
`RecordStore`), so any member answers. The thread `zend_attach` binds is held
on the session under `attached`, which is what `"self"` names. The route is
`Dispatch::Any`.

### npcd (`/v1/mcp` on bot.tokera.com)

Each tool calls the handler of the route for the same act in-process, with the
caller's headers. It reads the handler's answer back as that route's body type
(`npcd/src/engine/bodies.rs`). So ownership and the wording of a refusal are
the route's: a character the caller does not own is one npcd does not show
them. A route's role is checked by its guard, not its handler. The one tool
whose route asks more than `User`, `npc_channel_post`, checks the Creator role
itself.

| Tool | Route | Does |
|---|---|---|
| `npc_list {state?, world?, q?}` | `GET /v1/npc` | The caller's characters: id, name, world, state, mood |
| `npc_mission {npc}` | `GET /v1/npc/:nid/mission` | What the character carries or last finished: brief, steps and how each went, its answer and report, how many more are queued |
| `npc_mission_give {npc, brief, steps?, start?}` | `POST /v1/npc/:nid/mission` | Lodge a mission. `start` (true by default) carries it at once and rouses the character; false queues it for the command table |
| `npc_mission_step {npc, done?, outcome?, add?}` | `POST /v1/npc/:nid/mission/step` | Sign off the first open step (achieved or thwarted), or add one |
| `npc_mission_cancel {npc}` | `POST /v1/npc/:nid/mission/cancel` | Call off the open mission; the character is told |
| `npc_message {npc, text}` | `POST /v1/npc/:nid/message` | Message the character on its handset, as the caller |
| `npc_messages {npc}` | `GET /v1/npc/:nid/message` | The caller's thread with the character |
| `npc_channel {world}` | `GET /v1/world/:wid/channel` | The world's channel, as the caller has heard it |
| `npc_channel_post {world, text}` | `POST /v1/world/:wid/channel` | Post on the world's channel (the Creator only) |

**A call reaches the member holding what it acts on.** Character and world
routes are dispatched to the member holding that character's or world's lease
(`Dispatch::Owner`). The MCP route goes the same way: its key is derived from
the body (`Derived::McpSubject`). That key is a `tools/call`'s `npc`
argument, or failing that its `world` argument. A call naming neither, and
any other message, goes wherever the gateway's fallback sends it. Every npcd
tool that acts on a character or a world names it under one of those two
arguments. A test holds them to it.

npcd's tools hold nothing on a session, so its sessions are carried whole in
their ids (`mcp_server::SealedStore`). A handshake one member answered is
read the same way by the member a later call reaches.

**Missions as npcd keeps them.** A character holds one open mission, a queue,
and its last finished one. Missions have no ids, and only the open one can be
cancelled (`npcd/src/sim/missions.rs`). So a tool names a mission by its
character. Giving one with `start` replaces whatever the character was
carrying.

## 5. The aggregator app

`mcp/` builds a binary, `mcp`, configured by `mcp/mcp.yaml`:

| Key | Meaning |
|---|---|
| `bind` | Where it listens (`0.0.0.0:8095`) |
| `trusted_gateways` | The peers whose `X-Tokera-*` it believes |
| `gateway` | The gateway's LAN entrance every backend is reached through (`http://192.168.0.5`) |
| `state` | The folder its sessions are kept in |
| `server` | Its card: `name`, `title`, `description`, `instructions`, `icon` |
| `backends` | The servers whose tools it serves, in list order: `{name, host}` each |

| Module | What |
|---|---|
| `app` | The aggregated server: the card, the merged list, each call routed |
| `backend` | One backend, called through the gateway as the caller; its progress read off the stream |
| `sse` | Reading an event stream across chunks |
| `caller` | Who is calling, believed from the gateway, and the token that names them |
| `config` | `mcp/mcp.yaml` |

## 6. Deployment

The aggregator runs on AlienWorld while it is iterated. It is built there,
run from the repository root, and its port is open in the host firewall. The
gateway's `/mcp` route on `code.tokera.com` points at it by address
(`upstream: http://192.168.0.237:8095`, no `cluster:`), and
`server.trusted_backends` lists that address.

Deploy in this order:

1. **The gateway first.** The cluster's route table names `mcp_subject` by
   name, and a gateway older than it cannot read npcd's table.
2. **The members.** npcd is released on WhiteWorld, and zend on BlackWorld.

## 7. Decisions

- **Who may manage a character's missions** is npcd's existing rule: the
  character's owner. Through these tools, an admin reaches only their own
  characters, exactly as through npcd's routes.
- **A mission is npcd's own**: a brief and steps, lodged as a person lodges
  one (`Origin::Lodged`), not a new kind.
- **Watching a character work** means calling `npc_mission` and `npc_messages`
  again. Waits with progress stay zend's pattern.
- **One card for every backend.** The client sees the aggregator's card. Each
  backend's card describes it as a server of its own, which is what it is when
  called directly.
