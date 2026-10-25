# Replicated substrate: zend, npcd and agents as members of the gateway

**Status:** being built in §15's order. Chunks 1–6 are built: substrate identity and
order-independent apply, the `cluster` crate's wire protocol, consensus, members behind
the gateway, live replication with catch-up and rejoin, and operating the cluster —
fault mode, the maintenance commands, Ctrl-C draining first, `COORD` and the status
page. Chunks 7–10 are not.
**Scope:** a zend or npcd daemon started with
`--cluster https://<host or ip>` — and a new kind of member, the *agent*, a CPU machine
that runs command-line work inside repositories — dials that web gateway over one
WebSocket per app, proves it holds the estate's group key, and while connected *is* that
app's upstream there. `--cluster` may be given several times: each gateway is a
*cluster*, the clusters run active-active, and a member joined to all of them keeps
serving when any one fails. API traffic reaches a member over the socket it arrived on.
Every substrate record any member writes is relayed to every other member of the same
app. Events may arrive in any order: once a member has received them all, it holds the
same substrate whatever the order was. A sender re-sends what its gateway never
confirmed. A member that was away asks for what it missed. A restarted daemon also
offers what it holds. File-backed state moves into the substrate as VFS records and
reaches git through agents.

**Replication is substrate-only.** It lives in the persistence layer, the substrate's
indexes, the host's HTTP server and the gateway. It does not touch the hot path: the
scheduler, the projection assembler, the batched model, VRAM residency, and the code
that derives a turn's stream id. A remote record enters the substrate cold, and the
existing `elevate_to_hot` brings it to VRAM when a projection first reads it, exactly as
it does for a turn loaded from the local log.

§1 is the code as it stands, with references; it is what every later section answers
to. §2 lists the decisions. §3–§13 are the design. §14 is the test plan, §15 the build
order, §16 what is still open, §17 the documents this changes.

---

## 1. Where things stand

### 1.1 The gateway routes each site to one fixed URL

- One crate, three modes; the DMZ binary is the *gateway* mode
  (`docs/web_gateway_design.md` §1). It is axum 0.7 over hyper 1, with an HTTPS
  entrance on TCP and QUIC from one router (`web/src/tls/mod.rs:107-123`).
- A route names exactly one upstream: `Upstream::Local | Url(String)`
  (`web/src/config.rs:362-380`). There is no list of upstreams, no balancing, no
  stickiness (`web/src/proxy.rs`, all of it). The deployed table sends
  `bot.tokera.com` to `http://192.168.0.6:8081` and `code.tokera.com` to
  `https://192.168.0.5:8444` with `self_signed` (`web/web.yaml:237-261`).
- Health is passive: a backoff window per upstream URL string, 250 ms → 10 s, one probe
  let through when it expires (`web/src/health.rs:20-35,64-68,102-133`). A transport
  failure is `503 upstream_unavailable` with a self-retrying page
  (`web/src/errors.rs:57-76,208-258`).
- The upstream client is HTTP/1.1 only (`web/src/upstream_tls.rs:76-80`), with no
  response timeout and no idle timeout on tunnels.
- WebSockets are *tunnelled*, never terminated: on a `101` the proxy splices the two
  upgraded connections with `copy_bidirectional` (`web/src/proxy.rs:234-256`). Only
  HTTP/1.1 carries them: QUIC strips `Upgrade` (`web/src/tls/quic.rs:33-39`) and h2
  extended CONNECT is not enabled.
- `tokio-tungstenite` is a dev-dependency only (`web/Cargo.toml:81`); no production
  crate in the workspace has a WebSocket *client*.
- The gateway has **no telemetry and no status page**: no request or byte counters, no
  access log; `Health::report()` exists and nothing calls it
  (`web/src/health.rs:176-193`). `Config.roles` is parsed and never read
  (`web/src/config.rs:53-61`).

### 1.2 Daemons only listen, and trust the gateway by source IP

- A daemon's server is the same `web::Builder`, with every route `upstream: local`
  (`host/src/server.rs:87-142`). The merged API router is one `axum::Router`
  (`merge_routes`, `host/src/run.rs:280-289`), a `tower::Service` that tests already
  drive with `oneshot` (`host/src/endpoints/mod.rs:214-243`); npcd even calls its own
  router in-process (`npcd/src/engine/runtime.rs:2541-2587`).
- Identity reaches a daemon as **unsigned** `x-tokera-*` headers
  (`web/src/auth/forwarded.rs:21-26,39-87`), believed only when the TCP peer is in the
  `Gateways` allowlist — loopback, the bind address, each `--gateway` IP
  (`web/src/auth/gateways.rs:22-37`, `web/src/server.rs:458-466`). There is
  deliberately no shared signing key (`docs/web_gateway_design.md` §5).
- What depends on a real connection: `ConnectInfo<SocketAddr>` (the identity strip,
  `LocalSignin`, npcd's `--log-identity`), the `SecureEntrance` extension, and hyper's
  `OnUpgrade` for the two WebSocket routes.
- Routes are registered through `Api::route(path, min: Role, MethodRouter)`
  (`host/src/guard.rs:165-183`), recorded as `Guarded { path, min, nested }`
  (`guard.rs:48-54`); `declared(roles)` returns the table with no state
  (`host/src/endpoints/mod.rs:73-77`, zend `zend/src/api/mod.rs`, npcd
  `npcd/src/tenant.rs`).
- Streaming responses: zend chat SSE (`zend/src/api/chat.rs:330-397`), its reattach
  `GET /v1/conversations/:id/live` (`chat.rs:314-328`), upload progress SSE
  (`zend/src/api/files.rs:55-306`), npcd interaction SSE
  (`npcd/src/engine/mod.rs:1200-1312`), npcd NDJSON generation
  (`npcd/src/ndjson.rs:40-62`), and two WebSockets — `/ws/logs`
  (`host/src/endpoints/logs.rs:24-98`) and npcd's silent `/ws/events`
  (`npcd/src/engine/mod.rs:413-424`).
- The daemons know nothing of the gateway's address: `--gateway <IP>` is an *inbound*
  trust list (`host/src/cli.rs:35-39`).

### 1.3 The redo log has no global record identity

- A record is a one-line JSON header (`type, payload_len, crc, format, stream_id,
  chunk_index, token_count`) and a payload, padded to 4 KiB
  (`candle-conversation/src/persistence/record.rs:351-372,392-396`). Twenty-five types
  (`record.rs:88-300`). `encode_record` / `decode_record` are the one codec pair
  (`record.rs:451,516`).
- A record's only identity is its header key plus its local `(SegmentId, offset)`
  (`persistence/segment.rs:8-35`), and every maintenance pass moves the offset.
- **Ids that two machines could mint identically:** `TimelineId` is
  `max(now_micros, last+1)` from a process-wide counter
  (`projection/ids.rs:373-407`), including for zend's passthrough conversations, which
  are found by a content key, not by an id (`zend/src/passthrough.rs:174-183,357-358`); `TurnIndex` is the timeline's `turns.len()`
  (`substrate.rs:1568-1572`); the VFS `seq` is per timeline, continuing from the highest
  seen (`persistence/vfs/writes.rs:64,95`); npcd interaction ids are
  `now_ms` + a counter (`npcd/src/engine/interaction.rs:163-167`).
- **A turn's stream id is recomputed, not stored.** `turn_stream_id(timeline,
  turn_index)` (`persistence/content_hash.rs:217`) is derived from the decl at 34
  production call sites in 9 files: cold load, restore, scheduler and projection. No
  record holds it (`persistence/streams.rs:149-153`, `persistence/resume.rs:409,467,512`,
  `substrate.rs:5471`).
- **Checksums are weak and partial.** A `Chunk` record's `crc` is a Fletcher-32 over
  `kv_bytes` only, and the metadata (formats, palettes, scales) is covered by nothing
  (`record.rs:420-431,566-572`). Every other record carries a CRC-32. No stronger
  per-record digest exists.
- **K/V is not bit-identical across machines.** The int8 path, the expert format and the
  KV threshold row are chosen per card (`candle-core/src/quantized/mod.rs:679-715`;
  `candle-nn/src/kv_cache/chunked/sampled_selection/params.rs:702-707,1214-1224`). Batch
  composition changes summation order. So two members prefilling the same section write
  different bytes, each valid.
- **`adopt_turn` needs the source turn hot, and duplicates its K/V on disk.** It reads
  the source's resident K/V or refuses (`projection/resolver.rs:2560-2569`). The new
  turn's residence is cold-written later under the new stream as fresh `Chunk` records
  (`substrate.rs:5402,5436`; `persistence/thread.rs:1500`). Only in VRAM are the chunks
  shared.
- **Ids that already travel:** a zend dialogue timeline is
  `hash_bytes(conv_id)` — the same on every machine (`zend/src/session.rs:6659-6662`),
  and `conv_id` is a browser-minted UUID (`zend/web/index.html:1340`); section stream
  ids are content-addressed (`persistence/content_hash.rs:162-213`); `CustomObject`
  ids hash their key (`record.rs:1027-1040`); `npc_id` is SHA-256 over
  `(sub, name, now_ms, salt)` (`npcd/src/npcs.rs:983-999`).
- **Turns are not created by records.** `apply_walker_entry` replays a record into the
  substrate's *indexes* (`substrate.rs:5070-5226`) and already runs against a live
  substrate after `compact()` (`persistence/mod.rs:2305-2341`), but a turn entry and its
  residence are created only by `restore_turn` during whole-log reconstruction
  (`projection/resolver.rs:2729-3081`) or by the local seal path (`append_migrated`).
- **Runtime appends have two choke points** under the persistence mutex:
  `append_record` and `append_recurrent_parts` (`persistence/mod.rs:948-1046`), both
  reaching `account_appended` (`:1051-1088`). Maintenance relocates verbatim through
  `append_raw_record` (`:1099-1131`) and re-emits resident metadata through
  `append_record` (`persistence/maintenance.rs:1018-1057`); full `compact()` writes a
  scratch file of its own (`persistence/compaction.rs:777-884`).
- One writer per substrate directory, enforced by an OS lock on `substrate/LOCK`
  (`persistence/writer_lock.rs:1-52`).
- Sizes: a `Chunk` record is ~165 KB (one layer × one 32-token block,
  `persistence/mod.rs:637-640`); a recurrent `Snapshot` is ~63–65 MB per turn seal on
  the prescribed hybrid model (`persistence/writer.rs:54-60`); a long conversation's K/V
  is on the order of 1 GB (`npcd/src/engine/mind.rs:1352-1355`).
- **Applying a record is idempotent, but not order-independent.** Replaying the same
  record twice changes nothing, but two records for one key end in whichever was applied
  last. The log has always been replayed in its own append order, and the code relies
  on that:
  - Nearly every apply overwrites, so the last one applied wins. That covers
    `StreamDecl`, `Chunk` and `Tokens` locations, `Snapshot` and its empty-payload
    retraction, `BranchCheckpoint`, `ConvState`, `TreeMetadata`, `DebugId`,
    `CustomObject`, `Distilled`, `ProjectionEvents`, `WideQSig`, `TurnIndexPage`
    (`substrate.rs:4115-4320,5084-5206`). `Commit` overwrites rather than taking the
    max (`:4218-4220`). `Label` merges per field, last wins per field (`:4229-4258`).
  - `TreeMetadata`, `DebugId` and `TurnCoupling` are **dropped silently** when their
    timeline is not registered yet, and nothing re-applies them later (`:3505-3517`,
    `:3536-3547`, `:3575-3579`). Only `Label` and `ConvState` are stashed
    (`:4243-4271`).
  - The walk at open is a single pass in append order (`persistence/mod.rs:663-668`),
    so a record logged before its parent is lost for good.
  - **Section tombstones carry no generation.** A `PromptSection` decl lifts the
    tombstone, and old-generation records applied after it come back
    (`:4116-4118,4588-4622`; `persistence/record.rs:1417-1425`;
    `persistence/relocation_watch.rs:22-32`).
  - **A timeline tombstone is never undone, but it is not enforced at apply.** Later
    records still register and index the timeline, and a later `Snapshot` is re-inserted
    (`:3026-3035,4175`). Compaction does not re-emit timeline tombstones
    (`persistence/compaction.rs:541-546`), so after a compaction nothing would stop a
    late record bringing the timeline back.
  - **VFS events are keyed `(timeline, seq)`** with a locally allocated `seq`, so two
    writers collide (`persistence/vfs/index.rs:16-26,58-69,110-117`).
  - **A turn is rebuilt only when every chunk is present.** With some chunks it is
    `Corrupt` (`persistence/resume.rs:521-531`), and the corrupt-turn policy can
    tombstone the timeline (`projection/resolver.rs:2828-2909`). `Commit` is never
    checked against the chunks.

  §6.5 replaces these with rules under which the order does not matter.

### 1.4 Some state lives in files, not the log

| State | Where | Written by |
|---|---|---|
| zend chat uploads | `<workspace>/substrate/conv-files/` (`zend/src/conv_file_store.rs:61-71`) | `POST /v1/conversations/:id/files` |
| npcd worlds, personalities | `<mind>/worlds/`, `<mind>/personalities/` (`npcd/src/tenant.rs:288-289`) | admin `PUT`, `Registry::put` (`npcd/src/registry/mod.rs:268-305`) |
| npcd mind corpus | `<mind>/layers/…` (`npcd/src/mind/address.rs:172-223`) | admin `PUT /v1/mind/entry`; bench commits (`npcd/src/sim/bench.rs:654-704`) |
| npcd maps | `<mind>/map/<world>/` (`npcd/src/engine/runtime.rs:545-556`) | `/v1/local/reshape/:world` (`npcd/src/world/mapstore.rs:41`) |
| npcd accounts, images, device tokens | `<data>/accounts/`, `<data>/images/`, `<data>/tokens/` (`tenant.rs:300,383`, `npcd/src/images.rs:113`) | account upsert, portrait upload |

zend's *repository* edits already live in the substrate: a conversation's file store is
a chain of deltas per path over a pinned base commit, mirrored as `VfsEvent` /
`VfsTombstone` records scoped to the timeline (`docs/zend_vfs_events.md`, built). It
reaches git only through the `git_commit` / `git_merge` tools, which build the commit as
objects and push it to `origin` under a lease (`zend-vfs/src/work/commit.rs:59-218`,
`zend-vfs/src/remote/push.rs`). npcd has a git-*like* bench over the mind folder whose
commit writes straight to disk (`npcd/src/sim/bench.rs:654-704`); it runs no git
(`docs/the_effector.md`).

### 1.5 Execution is local; agents are designed, not built

`run_command` runs one allow-listed program in a borrowed checkout on the zend host
(`zend-vfs/src/sandbox/mod.rs:1-45`, `zend/src/sandbox_programs.rs:29-33`). Leased
build agents and a cluster with capability advertising and sticky routing by lineage
are written down and marked not built (`docs/zend_workspace_execution.md` §7–§8).
Nothing measures CPU for scheduling.

### 1.6 Build identity has no order

`/v1/status.build` is a 16-hex SHA-256 over the version string and every compiled-in UI
file (`host::ui::build_id`). It changes with the UI a binary carries and cannot say which
of two builds is newer. No crate embeds git.

### 1.7 The model a substrate is bound to is known but not surfaced

`ModelKey { arch, repo, file, weights, tokenizer, rope }` with `same_model()` comparing
arch, weights, tokenizer and rope (`candle-conversation/src/models/identity/key.rs:21-47`);
the substrate refuses a different model (`persistence/model_binding.rs:47-105`). No
daemon code reports it outside logs.

---

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **One WebSocket per app** — `zend`, `npcd`, `agent`, and `keep`. Each of the first three is its own replication group with its own build floor and model lock. `motherd` opens two. `keep` joins only so the status page lists the failsafe; it holds no substrate, takes no candidate, and is never rolled out. | A newer npcd raises only npcd's floor: a combined binary loses its npcd socket and keeps serving zend until rebuilt. Substrates never mix across apps. |
| D2 | **Members dial out, only when given `--cluster`; the gateways are the only listeners a group needs.** The daemons' own listeners stay up. While a member has a ready socket for an app, its listener forwards that app's requests **up** to a gateway. With none, it serves them itself. Without `--cluster` a daemon runs standalone, as today (§5.8). | Machines anywhere — LAN or internet — can join without inbound ports. A member cut off from every cluster keeps working on its own, then rejoins and catches up. |
| D2a | **Several clusters, active-active.** A member joins every `--cluster` it is given. It spreads its own events round robin across them, except a request's events, which go out on the socket the request arrived on (§6.4). | Load is shared across gateways, and losing one costs the requests in flight on it and nothing else. |
| D2b | **The gateway keeps its configured upstreams and falls back to them.** An app runs in *dumb mode* — today's fixed-URL routing — until a member of that app is ready on the gateway. It then switches to *clustered mode* and ignores the configured URL, and it reverts when the app's last member leaves (§5.7). | Clustering is opt-in by connecting. A deployment that never runs `--cluster` behaves as today, and a cluster that collapses degrades to the old routing instead of to an error page. |
| D2c | **Affinity is held by members, not gateways.** The member serving a sticky key claims it and announces the claim to every cluster it has joined (§5.4). NPC world and character leases are announced the same way for routing (§9.6). | Two gateways with private affinity tables could route one conversation to two writers. A claim the member announces is the same on every gateway. |
| D3 | **Mutual HMAC challenge over a group key that never travels** (§3.3). | The secret proves membership in both directions; replaying a transcript gains nothing. |
| D4 | **The socket carries virtual connections; both ends run their existing HTTP stack over them** (§5.1). | The redirect sits under hyper, so every route, guard, SSE, NDJSON, multipart upload and WebSocket upgrade works unchanged — the requested "low level, generalised" cut. |
| D5 | **Every runtime record replicates, K/V and recurrent snapshots included**, except derived and binding records (§6.1). | Every member holds the K/V the writer wrote, so a conversation that moves resumes without recomputing. |
| D5a | **Events may arrive in any order without changing the result.** Once every event has arrived, the substrate is the same whatever order they came in, tombstones removing what they remove. Each record carries a version stamp `(gen, epoch, hlc, origin)`. Every keyed value keeps the record with the greatest stamp, and a timeline's turns are ordered by it. Tombstones dominate, and records whose parent has not arrived wait for it (§6.5). | Idempotence alone is not enough: today's apply overwrites, so the same events in a different order leave a different substrate (§1.3). Events reach each member in whatever order the sockets deliver them. |
| D5b | **Writing while cut off is allowed. On rejoin, every event the group wrote before the rejoin is placed before every event the member wrote while cut off.** Detached events carry a provisional generation that sorts last. Once the member's catch-up completes, it takes a new generation decided by Paxos, `gen/<k>`, above every generation in use, so a rejoin needs a quorum (§6.7). Tombstones are never forgotten: an expired tombstone leaves an entry in a permanent deleted set (§6.8). | The invariant holds exactly, group-wide, including for events the member had not caught up on. The generation inside each event makes every member order it the same way. Nothing a cut-off user writes is lost, nothing becomes a branch, and no absence is too long. |
| D6 | **Records gain `(origin, oseq)`. Recovery has three parts:**<br>• the sender re-sends whatever no `RECEIPT` covered (§6.3);<br>• a member that was away asks one member, picked round robin, for the headers of everything after where its persisted catch-up stands, then fetches only what it lacks, with every reply addressed to it alone (§7.2);<br>• a restarted process also catches up, then offers what it holds (§7.5).<br>Events are applied as they arrive. | The log has no global identity today (§1.3). Headers make a catch-up cost what is missing, not what exists. `reply_to` keeps it off every other member's link. Idempotent apply makes every duplicate harmless. |
| D7 | **Each gateway is a stateless relay.** It stores no records. Its routing state (the claims it has been told, floors, locks) is soft and in memory. | A gateway restart costs a reconnect, a retransmit of what it had not receipted, and a catch-up, never data. The members re-announce their claims on reconnect. |
| D8 | **Each route declares how it is dispatched** — `Best`, `Sticky`, `Owner`, `Aggregate`, `Any` (§5.2). | Routing is a property of the route, decided where the route is written. |
| D9 | **World state is persisted as state records; transient events are broadcast and never persisted** (§9.1). | A world survives restarts and moves between machines, and the log holds what the world *is*, not every step it took. |
| D9a | **Each world has one authority, decided by a Paxos lease among the npcd members** (§5.9, §9.2). Only the authority persists world state. | Paxos's safety means two members never hold one world, so world state has one writer even across a partition, and a minority side holds no world at all. |
| D9b | **The authority lends NPC leases, spread across the npcd members by weighted rendezvous from the start; a character runs where its lease is, until the lease expires *and* its holder reconnects** (§9.3). | A character's memory has one writer, and a partitioned machine's characters are never run a second time elsewhere. Losing one machine removes only its share of the cast. |
| D9c | **A cut-off machine runs its characters in a local, unpersisted world, scrapped on reconnect** (§9.4). **It may still publish an operation's verified work to git**, through its in-process agent. An unreachable `origin` or agent makes the publish wait 30 minutes and try again. If both sides of a partition publish one operation, git merges them, and a conflict is solved by the publishing NPC at the merge table (§10.3). | Characters keep living through a partition, and each side's world never corrupts the authoritative one. git sits outside the cluster, so a partition from the gateways is no reason to hold finished work back from the record. |
| D9d | **Consensus is a general service: Paxos among each app's members. The gateways carry the vote, and the lowest-id gateway is the tie-break acceptor for a group of two** (§5.9). It offers leases and one-time decisions. The quorum is a majority of `max_members`, the largest group the app has had, persisted in the substrate and raised automatically; it can be lowered by flag or admin call. | No configuration file sizes the group. A restarted or cut-off member knows the quorum from its own substrate. One member of two can still decide while the voting gateway is up, and a larger group is unchanged. |
| D10 | **Every zend ingests its own workspace and replicates it**; a unit already ingested by any member is not ingested again, and a unit two members find still unread is read by the one that holds its lease (§10.4). | Machines with different checkouts contribute their repositories to the shared memory, and a startup where several of them plan the same tree does the reading once. |
| D11 | **File-backed state becomes VFS records**: zend uploads into the conversation's store, npcd's authored content into an operation-scoped store over the mind repository, which every step of the operation works in. Publishing is an explicit git merge run by an agent, and for npcd it is the workflow's last step, taken at the archive floor's merge table after verification has passed; only a publish step may lead to `done` (§10). | One replication mechanism; git stays the record of published work, and nothing reaches it unverified. |
| D12 | **Agents are members of app `agent`**; command-line work and git writes run on the best idle agent (§11). | Realises `docs/zend_workspace_execution.md` §7–§8 on the same socket. |
| D13 | **Build version = the count of `origin/main` commits HEAD contains**, plus a `local` digest of anything beyond it (§3.5); per-app floor on the count, held in gateway memory. A newer build raises the floor after a 30-minute grace period, during which older builds keep serving (§3.5). | A total order across checkouts, and rolling upgrades one machine at a time. |
| D14 | **A member whose disk fills stays connected in fault mode** (§7.7). No requests are routed to it. It hands back every role that needs a log write, **keeps voting** from a preallocated vote journal, keeps sending what it already wrote, and catches up when space returns. | The group keeps going without it, and keeps its quorum, and the operator sees why on the gateway page, instead of a member that crashed or vanished. |
| D15 | **Maintenance runs as admin commands on members, from the gateway page**: drain, restart, shutdown, compact, maintain, audit, catch-up, resync and others (§12.4). **Ctrl-C runs `shutdown`**: drain, wait for the outbox, flush, close; a second Ctrl-C aborts. They are routes on the member, so they work through the cluster and on a member's own listener alike. | Maintenance without SSH or launch-line juggling; every command is logged with who ran it. |
| D16 | **Gateways peer with each other** over group-key-authenticated sockets. They share ACME challenges, issued certificates and keys, and the order ledger. Rendezvous over the connected gateways names who orders each site, with no consensus; a gateway partition can cost one duplicate certificate. They refuse to peer when their session keys or OAuth clients differ (§13.4). Peer links carry nothing else: no requests, members or claims (§4.3). | Active-active gateways answer the same names: any of them must answer a challenge and serve the certificate, and a cookie from one must work on the other. A duplicate order is harmless within the CA's limits, so it is not worth a Paxos group among gateways. |
| D17 | **Every member holds a full copy, with no encryption at rest beyond what the disk provides; this design includes no backup and no key rotation** (§16). | Kept simple by decision. Replication is not backup: a mistake replicates. |
| D18 | **A/B upgrades.** A higher build loads untouched. Once ready, it is certified against the older members, on a share of live traffic and on mirrored calls to read-only routes, compared statistically. Its binaries are distributed to the older members, resumably and peer to peer. At the grace deadline, if certified, the members upgrade themselves one at a time: drain, stage the new binary, and ask `keep` to start it. `keep`, a separate process, keeps that file only after it is serving; otherwise it starts the last known-good file again. A failed certification faults the candidate and aborts the upgrade (§3.8, §3.9). | Upgrading the estate is building one machine. A bad build is caught on a sliver of traffic before it replaces anything, and a bad install puts itself back. |
| D19 | **A local build** — commits not on `origin/main`, or uncommitted edits, measured exactly by the build's `local` digest — **serves its own listener locally and never takes part in upgrades**: never a candidate, never auto-installed, never closed for being behind. Otherwise it is a full member: its writes replicate, the gateway routes to it, it holds leases and votes (§3.10). | Work-in-progress can be tested on a cluster machine through its own port, while the gateway and the upgrade machinery carry on untouched. |
| D20 | **Every member keeps its last *N* builds** (`--keep-builds`, default 5), **and an admin can force any build onto an app from any member that holds it** (`rollout`, §3.11). A forced rollout skips grace and certification, moves the floor in either direction, and is a Paxos decision, so every gateway applies it. A downgrade rejects the builds it leaves. A build is installed only where it can open the member's substrate: same log format, every record type known, same model (§3.9). | Rolling back is a swap from each member's own disk, with no transfer, so it is as fast as a restart. Deciding by Paxos keeps active-active gateways and a restarted gateway on the same rollout. The checks keep a downgrade from hiding, and then compacting away, data an older build cannot read. |

---

## 3. Membership

### 3.1 The endpoint

`GET /v1/cluster/:app` with an HTTP/1.1 `Upgrade: websocket`, where `:app` is `zend`,
`npcd` or `agent`. Merged ahead of site routing on every host, like `/auth/*`
(`web/src/server.rs:194-222`), on **both** entrances: the internet path arrives through
Cloudflare on the plain entrance (`docs/deployment.md` topology), a LAN member may dial
the HTTPS entrance on `:9443`. HTTP/2 and HTTP/3 cannot carry it (§1.1) and refuse it
with `426`.

### 3.2 Dialling

New host flags (`host/src/cli.rs`), also accepted in `motherd.yaml`'s `host:` section:

| Flag | Meaning |
|---|---|
| `--cluster <urls>` | One gateway, as one or more URLs separated by spaces, in priority order — the ways this machine reaches that gateway (*paths*, below): `https://gateway.tokera.local:9443 https://gateway.tokera.com:9443` dials the LAN name first and the internet name when it fails. Each URL is a host name or an IP. **Repeatable** — one per gateway, for high availability. Never two `--cluster`s for one gateway: each would replace the other's socket (4409, §3.4). Absent: the daemon runs standalone, as today, and never opens a socket. |
| `--cluster-name <host>` | The name a gateway's certificate is verified against, whichever path dials it; default: each URL's host. |
| `--group-key <file>` | The group key file (§13.1). Required with `--cluster`. |
| `--tombstone-retention <days>` | How long a tombstone record stays in this member's log before it shrinks to an entry in the deleted set (§6.8); default 30. Disk only: it changes no result, so members need not agree on it. |
| `--cluster-quorum-max <n>` | Write `cluster/max_members = n` as a *set* at startup, lowering the quorum's base (§5.9). For retiring machines for good, or knowingly running with less safety. |
| `--keep-builds <n>` | How many previous builds the member keeps for rollback (§3.9); default 5. |

The member upgrades an `https` request to the WebSocket (`wss` on the wire). The TLS
client verifies the gateway's Let's Encrypt certificate against webpki roots — never
`self_signed`. An `http://` cluster is refused unless its host is loopback.

**One socket per app per cluster.** A `motherd` given two clusters holds four sockets:
zend and npcd to each. Every socket runs its own handshake, keep-alive and reconnect
loop. A member is *joined* to an app while at least one of that app's sockets is
admitted. It is *ready* (§7.2) once caught up through any of them.

**Reconnect.** After any close that is not terminal (§3.4), that socket waits and dials
its cluster again. The member's other sockets are unaffected. Delay `d` starts at 250 ms and doubles per consecutive failure, **capped at
8 s**; each wait is drawn uniformly from `[d/2, d]` so a gateway restart does not
reconnect every member in the same millisecond. An admission resets `d`. Keep-alive: a
`PING` every 15 s each way; 45 s without hearing from the peer closes the socket as dead,
with 1001 (Cloudflare's idle timeout is far longer than 15 s). Hearing is a frame, or the
bytes of one still arriving: a `PING` waits behind a mebibyte fragment, which over a link
of a few kilobytes a second takes minutes to arrive whole, from a peer alive throughout.
Counting only whole frames closed such a socket every 45 s, and the catch-up it carried
started over on each redial and never finished. A transport that cannot report its reads
counts whole frames. The handshake
holds to the same 45 s: a peer silent that long mid-handshake is closed 4400.

**Paths.** A cluster's URLs, in order, are its *endpoints*. Each endpoint's host is
resolved afresh at every dial, and every address it resolves to is one *path*: a name
with two A records (`gateway.tokera.local` → `.4` and `.5`) is two paths of equal rank,
and a later endpoint ranks below every path of an earlier one. A socket runs on one path
at a time. The gateway sees a member by its node alone, so a socket that comes back on
another path is the same member, its old socket replaced (4409).

- **Dialling.** One dial tries the paths in rank order, skipping those cooling down, and
  takes the first that completes TCP, TLS and the WebSocket handshake, each within 10 s.
  The cooling paths are tried last, the soonest out of its cooldown first: by the time the
  paths ahead have timed out, it is the likeliest to answer. So a member never refuses to
  dial. Only when every path fails does the dial fail, and the reconnect delay above
  applies. Every dial is logged with its path; a path that was well failing to dial is
  logged as a warning, a failing one again only at debug.
- **What cools a path.** A path that fails to connect, drops mid-handshake, or carries a
  session that ends dead (45 s without a frame) takes a strike. A path *connecting and
  then going silent* is the fault this catches: a link that passes the handshake and
  stalls under bulk traffic. Its cooldown is 30 s, doubling per consecutive strike, capped
  at 30 min. A session that lasts 10 min clears its path's strikes. A dropped transport or
  a close the gateway sent is no strike: a gateway restart is not the path's fault.
- **Returning.** While a socket runs on a path below the best endpoint, every 60 s it
  probes the best-ranked path of a better endpoint not cooling (TCP and TLS). When the
  probe succeeds, the socket closes itself 1001, "returning to a better path", and the
  reconnect dials that path first. A failed probe is a strike. Paths of one endpoint are
  equals: a socket never leaves one A record for another that is still well.
- **Each socket keeps its own.** The zend and npcd sockets judge their paths apart: a
  bulk stall on zend's says nothing of npcd's.
- **Sources.** A machine with two links to one network — BlackWorld's wired `.4` and
  wireless `.5` on one /24 — sends a connection the system routes out of its preferred
  link whether or not that link works, so a dead preferred link black-holes every
  connection the machine starts. A path is therefore also *from* an address: each
  destination is dialled from each of this machine's addresses (up, not loopback, not
  link-local) of its family — only those on the destination's network when there are
  any — bound to it, the address the system would choose first. A destination on this
  machine is dialled unbound, as is any destination with one candidate source: a machine
  with one link dials as it always has. A failure to *reach* — the bind or the TCP connect
  failing — also makes its path's source **suspect**: its paths to every destination are
  dialled after every path from a source that is not suspect, until a session opens from
  that same source — a success from another source clears nothing. A fault after the
  connect — TLS, a handshake, a session gone silent — cools only its path: the source
  reached, and the stall is the far end's.

**Seeing a member held back.** A member admitted below the floor takes no traffic and
replicates nothing until it holds the floor's build (§3.5), so its `/v1/status` says so
per app under `upgrade`: `below_floor` (the floor's `count-hash`), `step`, `stranded`, and
`fetching` (`bytes` of `size`, and how often it resumed). A fetch stuck at 0 bytes is a
member that cannot reach the build, not one that is slow to rejoin.

### 3.3 Handshake

Nothing but these frames crosses the socket before `ADMIT`: no API stream, no substrate
event, no catch-up.

```
member                                               gateway
  HELLO { proto, app, node, host, build, model?, power, repos, now_ms, nonce_m } ──▶
  ◀── CHALLENGE { nonce_g, proof_g }
        proof_g = HMAC-SHA256(K, "tokera-cluster/1 gateway" ‖ app ‖ node ‖ nonce_m ‖ nonce_g)
  member verifies proof_g; a mismatch closes the socket: this is not our gateway
  PROVE { proof_m } ──▶
        proof_m = HMAC-SHA256(K, "tokera-cluster/1 member" ‖ app ‖ node ‖ nonce_g ‖ nonce_m)
  gateway: proof → build floor → model lock → clock skew → node session
  ◀── ADMIT { session, members, claims, floor, routes_wanted, own_applied_through }
                                                             or a close with a code (§3.4)
  ROUTES { declared dispatch table } ──▶                     (§5.2)
  CATCHUP { since } ──▶                                      (§7.2)
  HAVE { applied set } ──▶          only on the first admission after a process start (§7.5)
```

- `K` is the group key (§13.1). Nonces are 32 bytes from the OS RNG; proofs are compared
  in constant time (the `hmac`/`sha2` pair `web` already uses for sessions,
  `web/src/auth/session.rs:19,106,115`). The labels bind direction and protocol version,
  so a member's proof can never be played back as the gateway's.
- `node` is the member's persistent 64-bit id (§3.7); `host` is a display name.
- `build` is `{ count, hash, local }` (§3.5). `model` is the `ModelKey` the substrate is
  bound to — the fields `same_model` compares, plus `describe()` for display; agents
  send none. `power` and `repos` feed routing (§5.3, §10.1, §11.2).
- **One session per node.** A `HELLO` from a node that already has a session on this
  gateway for the app replaces it, once its proof passes: the old session is closed
  4409 and its streams fail as §5.6 says. A socket that drops and is re-dialled at once
  is therefore admitted at once, not after the old session's 45 s keep-alive runs out.
  Two live processes cannot share a node, because a copied substrate re-mints its node
  (§3.7), so the newer session is always the same member.
- Failed proofs are counted per source IP; more than 10 in a minute holds that IP's
  handshakes off for a minute.

### 3.4 Close codes

| Code | Meaning | Member then |
|---|---|---|
| 4400 | Malformed frame or protocol violation | reconnects (backoff) |
| 4401 | Proof failed (either direction) | reconnects (backoff); the loading page names it |
| 4409 | Replaced: a newer session of the same node was admitted (§3.3) | nothing: this socket is the stale one, and the member is already connected through the new one |
| **4410** | **Build superseded, and no way to upgrade:** `build.count` below the app's floor, and no build at the floor fits it (no artifact for its kind and platform, or the checks of §3.9 fail); or a member left behind when a rollout ends: stranded, or its install failed (§3.5, §3.9). A member below the floor that *can* take the floor's build is admitted to upgrade instead (§3.5) | **closes that app's sockets to every cluster and stops reconnecting** until the process is replaced by a rebuilt binary — a build any cluster has superseded serves on none |
| 4412 | Model mismatch: the app's group is locked to another model. A new model is moved to by hand, not by an upgrade (§3.6) | reconnects at the 8 s cap; admitted once the group empties and the lock clears |
| 4417 | Clock skew: `HELLO.now_ms` is more than 60 s from the gateway's clock. A clock far ahead would drag every member's `hlc` with it (§6.2), and Paxos lease terms and tombstone ages read the clock (§5.9, §6.8) | reconnects at the 8 s cap; the loading page names the skew |
| **4426** | **Protocol version unsupported** | **stops reconnecting** |
| 1001 | Gateway shutting down | reconnects |

Every close carries a reason string naming the values involved — e.g.
`build 4812 (a1b2c3d) is below the npcd floor 4815 (9f8e7d6), set by node 3f… on .6;
rebuild` — which the member shows on its loading page and in `/v1/status`.

### 3.5 The build floor

- **The version** is embedded at compile time by a `build.rs` in the new `cluster` crate
  (§5.1), which every member links. It has three parts:

  | Part | Value | Moves when |
  |---|---|---|
  | `count` | commits on `origin/main` that HEAD contains: `git rev-list --count $(git merge-base HEAD origin/main)` | a pushed commit is fetched and built. A local commit does not move it. |
  | `hash` | `git rev-parse --short` of that merge base | with `count` |
  | `local` | empty when the build's **inputs** — the workspace `Cargo.toml` and `Cargo.lock` and every workspace crate's directory — are exactly the merge base's. Otherwise a digest of how they differ: the working tree's diff against the merge base (`git diff <base> -- <inputs>`) and the untracked input files. A file outside the inputs (a doc, a tool's lock file) cannot change the binary and does not make a build local; and the same sources on the same base give the same `local` however they were reached — committed locally, left uncommitted, or built again | an edit, local commit or new file among the inputs |

  The build script reruns on `.git/HEAD`, the refs `HEAD` and `origin/main` name,
  `.git/packed-refs`, the workspace `Cargo.toml` and `Cargo.lock`, and every workspace
  crate's `src/`, `Cargo.toml` and the UI directories, but not `target/`. So `local` is
  accurate to the build: a binary built from edited sources always says so. `local` is
  the first 16 hex digits of a SHA-256 over the input diff and the untracked input
  files; a tree with no git or no `origin/main` is count 0, hash `none`, and a local
  build named by HEAD's history as well (`cluster/src/version/compute.rs`).

  The version therefore changes with any edit in the workspace, and every crate that
  links `cluster` rebuilds with it. Nothing below the daemons may link it for that
  reason: the applied set both the substrate and the protocol name lives in its own
  dependency-free crate, `applied-set`.
- **A build with `local` set is a *local build*** (§3.10). The floor, the grace period
  and the upgrades compare `count` only.
- **The floor is per app.** Other apps are untouched (D1).
- **A member below the floor upgrades on admission, if it can.** That is a member that
  was off or cut off while a rollout ran, or one started from an old binary.
  - The gateway admits it in the **upgrading** state (`ADMIT` then carries the floor)
    when some member connected to that gateway holds an artifact of the floor's build:
    running it, keeping it, or holding a verified download, as its `BUILD_OFFER`s and
    `BUILD_HAVE`s told the gateway (`gateway/upgrades/held.rs`). Otherwise it is closed
    4410. The gateway does not know the member's kind or platform, so the member makes
    §3.9's checks itself once admitted. One with no matching artifact, or whose checks
    fail, reports itself **stranded** in `COORD`, and the gateway closes it 4410 at once,
    with that reason.
  - An upgrading member reports `upgrading` in `CAPACITY`, takes no traffic, applies and
    sends no events, holds no claim or lease, makes no catch-up, sends no Paxos frame and
    does not vote. Its unsent events stay in its log, where the new build reads them
    after the swap.
  - It is sent `BUILD_OFFER` for the floor's build at once. It fetches it, or takes it
    from its own `builds/`, and installs it by the swap of §3.9 straight away. It takes
    no `upgrade/<app>` lease, since it serves nothing the app would lose.
  - The new process is admitted as an ordinary member and catches up (§7.1). An install
    that fails and is restored by the watchdog comes back refusing that build
    (`--upgrade-failed`); it installs nothing more, and the gateway closes it 4410 at
    once, with the reason.
  - A local build is never below the floor in this sense (§3.10).
- **A newer build starts a 30-minute grace period, then the floor rises.**
  - When a member above the floor is admitted, the gateway sets the app's **pending
    floor** to its `count`. The 30-minute deadline starts only when that member is
    **ready**, after its load and catch-up (§3.8), not at admission.
  - Until the deadline, the older members keep serving beside it, and the app is flagged
    as running mixed builds.
  - The new member is certified against the old throughout (§3.8), and its build is
    distributed to them (§3.9).
  - An even newer member admitted inside the window raises the pending floor, but not
    the deadline.
  - At the deadline each gateway sends its verdict, and the candidate has the group
    decide (§3.8). The floor rises when a member's `COORD` reports the group's `Pass` on
    the pending build, and the rollout of §3.9 then upgrades the older members one at a
    time. Those that cannot upgrade are closed 4410 as it settles. If certification
    failed, nothing rises and nothing is closed.
  - A member admitted during the window must meet the *current* floor, not the pending
    one.
  - The status page shows the countdown. An admin can end the grace period at once with
    the `raise-floor` command (§12.4): the gateway sends its verdict now, and the floor
    rises as soon as the group certifies the build. `cancel-floor` discards a pending
    raise if the newer build was a mistake.
  - `floor_grace_minutes` in the gateway's `cluster:` block (`web.yaml`) sets the period;
    the default is 30. It is configuration, not a flag, like the rest of that block.
- **Held in memory only, by each gateway.** A gateway restart forgets every floor and
  every pending raise. The first member of each app to reconnect sets the floor again,
  unless it is a local build or a build the group rejected; neither ever sets a floor.
  The group's verdicts and its latest forced rollout come back with the members'
  `COORD`, and a rollout's decision sets the floor whichever member reconnects first
  (§3.11). Active-active gateways keep their floors independently. Members join every
  cluster they are given, so the floors agree in practice. Where they do not, a member
  superseded on one cluster leaves all of them, because the 4410 rule above holds across
  clusters.
- **Agents follow the daemons' floor.** An agent is never a candidate and never takes
  part in Paxos (§11). When zend's or npcd's floor rises to a build whose artifacts
  include a `tokera-agent`, the gateway raises the `agent` floor with it and closes each
  agent below it 1001 (*going away*). The agent reconnects, is admitted upgrading, and
  installs the build as above.
- **Equal counts** come from one commit on `origin/main`, so they share a `hash`. Only a
  rewritten `origin/main` could give one count two hashes. The gateway then flags the
  app as running mixed builds and admits both. Different code at one count is otherwise
  a local build (§3.10), and says so.

### 3.6 The model lock

Per app, not for `agent`. The first member admitted to an empty group records its
`ModelKey`; every later `HELLO` must be `same_model` with it or is closed 4412 before
anything else crosses the socket. The lock clears when the group's last member leaves.
The prescribed model is one value for every daemon (`host/src/model.rs:17`), so in a
group that is all built from the same tree the lock never fires — it exists for the
member that is not.

**Changing the model is done by hand, not by an upgrade.** A substrate refuses to open
under another model (`persistence/model_binding.rs:47-105`), and its K/V is meaningless
under other weights. So a build whose `host::MODEL` differs from the app's lock:

- is never a candidate. It is closed 4412 at the handshake, with both models named in
  the reason, and is never certified, distributed or installed;
- strands any member a forced rollout would give it to (§3.9, §3.11).

Moving an app to a new model is an operator procedure:

1. Stop every member of the app.
2. Move each substrate directory aside.
3. Start the members on the new build with empty substrates.

The first member to join sets the new lock, and the app's history starts over. §15.1's
rollback applies unchanged: restoring the moved directories and the old build puts the
app back.

### 3.7 Node identity

A member's `node` is 64 random bits created at first open and stored beside its data:
`substrate/NODE` for zend and npcd (so it is the substrate's identity and survives
restarts, and the two substrates of a `motherd` are two nodes), `<agent dir>/NODE` for
an agent. It is the `origin` of every record that substrate writes (§6.2).

**A copied substrate must not keep its node.** Two directories with one `NODE` would
both issue `oseq` 1, 2, 3… under the same origin. Every member would then drop one
copy's records as duplicates of the other's, which is silent data loss. So `NODE` stores
the id beside a fingerprint of where it was minted: the machine's OS install id and the
substrate directory's canonical path.

On open, a mismatched fingerprint means the directory was copied or moved. The member
then mints a fresh node: it writes new records under the new origin, and the copy's
existing records keep the origin they carry. Nothing writes under the old origin
again, and its records replicate like anyone else's. The status page lists
re-mintings.

### 3.8 A/B upgrades: certifying a new build

A member that joins with a build above the app's floor is a **candidate**. Before the
group moves to its build, the candidate has to prove itself against the members it would
replace: the **baseline**.

**Loading gates everything.**

- A member reports `state: loading` in `CAPACITY` and `COORD` from admission until its
  load and its admission catch-up are both done (§7.2, §7.6). The gateways route nothing
  to it meanwhile.
- **The grace period of §3.5 does not start until the candidate is ready.** A model load
  or a long catch-up does not use up the window. Certification starts at the same moment.

**Certification runs through the grace period** and compares the candidate with the
baseline in two ways.

**Live share.**

- The candidate takes a share of the live traffic of the routes it can serve: 10% of
  `Best` and `Any` requests, and of new sticky keys (`canary_share` in the gateway's
  `cluster:` block). The rest goes to the baseline as before, and the baseline's picks
  never land on the candidate (`Registry::pick_canary`).
- For each route, the gateway records per member:
  - status classes (2xx, 4xx, 5xx);
  - transport failures and timeouts;
  - latency;
  - response size.

**Mirrored calls.** For routes declared safe to mirror, the gateway sends a copy of a
baseline-served request to the candidate as well. It answers the client from the
baseline, and compares the two answers.

- Every route declares its mirror mode beside its dispatch (§5.2):

  | Mirror mode | Compared | Routes |
  |---|---|---|
  | `Exact` | the JSON bodies, after dropping the fields the route names as volatile | `/v1/conversations`, with `updated_ms` and `turn_count` volatile: the one read whose every other value is replicated state, the same on every member |
  | `Shape` | status, and the response's JSON structure and types, not its values | reads whose values legitimately differ between members or builds: `/v1/tools` and `/v1/schema/layers` (the catalog and layer table are the build's own, so a new build changes them by design); `/v1/repo_map` (each machine's checkout); npcd's `/v1/world`, `/v1/world/:wid`, `/v1/personality` and `/v1/personality/:aid` (minds are per-machine checkouts, and a world's live state moves between the two calls); `/v1/status`, `/v1/telemetry` |
  | `None` | never mirrored | everything that writes or has side effects: chat turns, uploads, acts, commands, any `POST`/`PUT`/`DELETE` not declared otherwise |

- Mirroring is capped at 20% of eligible requests, so the candidate is not loaded twice.
- The client is always answered from the baseline. The copy to the candidate runs in
  the background, and only responses that declare a `content-length` of at most 8 MiB
  are buffered for comparison (`web/src/cluster/mirror.rs`).

**The verdict.** Certification passes when, by the end of the grace period, every route
with enough samples (at least 200 live or 100 mirrored) meets all of these, and no route
fails any of them:

- **errors:** the candidate's 5xx and transport-failure rate is not worse than the
  baseline's, by a one-sided two-proportion test at 99% confidence, with a tolerance of
  0.5 percentage points;
- **latency:** the candidate's p50 is within 1.5× of the baseline's, and its p95 within
  2×;
- **mirrors:** `Exact` routes mismatch on at most 0.1% of compared calls; `Shape` routes
  mismatch on none.

A route without enough samples neither passes nor fails. The gateway then reports
certification as **passed with gaps**, listing them.

**It is evaluated continuously**, once a second (`gateway/upgrades/verdict.rs`). A
significant failure ends certification at once; it does not wait for the deadline. A
pass, a pass with gaps, or *no opinion* is sent at the deadline, or when `raise-floor`
ends the grace period by hand.

**A failure aborts the upgrade:**

- the candidate is put into **fault mode** (§7.7), with the reason *certification
  failed* and the failing routes and figures. The gateways route nothing to it, it still
  votes, and it keeps sending what it wrote;
- the pending floor is cancelled. The candidate's build count stays higher, but the
  automatic upgrade is aborted for that build: §3.9 neither distributes it further nor
  installs it anywhere;
- the failure goes in the coordination log with the figures, and the build is marked
  *rejected* on the status page;
- an admin can lift it with `certify-anyway` (accept the build and resume) or
  `reject-build` (keep it rejected and drain the candidate) (§12.4).

**With more than one gateway**, each certifies on the traffic it carries, and the verdict
needs every gateway that carried enough traffic to agree. A pass on one gateway and a
failure on another is a failure.

**The group records the verdict.** Gateways hold no Paxos state (§5.9), so:

1. Each gateway sends its verdict, with its figures, to the candidate in a `VERDICT`
   frame, and sends it again every 10 s until a member's `COORD` reports the group's
   verdict on the build. A candidate whose socket dropped, or whose process restarted,
   still hears it.
2. The candidate proposes the decision `certify/<app>/<count>-<hash>/<i>` (§5.9) once
   every socket it is admitted on has sent a verdict. A gateway that carried too little
   traffic sends *no opinion*. A failure from any gateway is proposed at once. The value
   is `{ result: pass | fail, by, at_ms, reason, figures }`, with each gateway's figures
   summed up in a line (`<overall>: <n> pass, <n> gap, <n> fail`; the per-route figures
   stay on the gateway's own page) and, for a failure, the failing routes as the reason.
   A decision is at most 799 bytes (a vote journal's 800-byte value, less its tag), so
   the reason is cut short, marked `…`, and then trailing summaries are dropped until it
   fits; a value that did not fit would be refused by every journal and never be held.
   `i` is one above the latest
   verdict the candidate holds on the build; a proposal that loses its number is made
   again under the next, every 5 s, until the group holds a verdict. The latest `i` is
   the group's verdict, which is how `certify-anyway`, `reject-build` and a forced
   rollout overrule one (§3.11, §12.4).
3. Every member and gateway then reads the same verdict, from the decision record and
   from `COORD`. The members read their decisions once a second
   (`upgrade/runtime/facts.rs`): the candidate enters fault mode on a failure, a member
   on a rejected build drains itself, and an accepted one takes work again.

A candidate that stops before proposing leaves its build uncertified, and the next
candidate at that build starts again.

**What is certified is a build, per app.** A `motherd` candidate is certified for zend
and for npcd separately, on each app's traffic, and an app whose certification fails is
not upgraded even if the other passes.

### 3.9 A/B upgrades: distributing and installing the build

**The candidate offers its build** as soon as it is ready, to every member of its app
below its build. It sends `BUILD_OFFER { build, artifacts }` on every socket. The
artifacts are:

- **Every binary built beside it at the same build.** A `cargo build --release` writes
  `zend`, `npcd`, `motherd` and `tokera-agent` into one directory, all stamped with one
  `count` and `hash` by the `cluster` crate's `build.rs` (§3.5). The candidate offers
  each binary in its executable's directory that reports the same build. Every binary
  answers `--build-info` with `{ kind, count, hash, local, platform, log_format,
  record_tags, model }`, which is how the candidate checks. So a `motherd` candidate can
  also upgrade standalone zend and npcd members, if those binaries were built with it.
  The last three say what the build can open:
  - `log_format` is the build's `FILE_FORMAT_VERSION` (`log_file.rs:41`, 3 today);
  - `record_tags` is the set of `RecordType` tags it decodes;
  - `model` is the preset name of `host::MODEL` (`override_key`), the one model it runs
    (§3.6). A `ModelKey` hashes the weights, which `--build-info` does not load.

  An agent opens no substrate and runs no model, so its three are `null`, `[]` and
  `null`. Every main answers `--build-info` before it parses its arguments
  (`host::build_info::first`, `cluster::version::info::answer_if_asked`), so the probe
  never opens a model or a substrate.
- For each artifact: its kind, platform (OS and CPU architecture), size and SHA-256.

There is no UI artifact. Each binary carries its tenants' UI compiled in
(`docs/shared_model_host.md` §3.4), so the executable is the whole build.

**Which members fetch which artifact.**

- A member fetches only the artifact of **its own kind and platform**. A zend member
  takes `zend`, a `motherd` member `motherd`, an agent `tokera-agent`.
- **GPU architecture is not a concern.** Kernel archives are built for every
  architecture into every binary.
- A member for which no artifact matches is **stranded**: shown on the status page with
  the reason, and left as it is for an operator to rebuild.
- **A build must be able to open the member's substrate.** A member is also stranded,
  with the reason, when the artifact fails any of three checks against what the member
  reports in `COORD` (§12.1):
  - **log format:** the artifact's `log_format` differs from the member's substrate.
    `open` refuses any other version outright (`log_file.rs:129`), so the new process
    could not start, and the watchdog would only restore the old one;
  - **record types:** the member's substrate holds a record type outside the artifact's
    `record_tags`, by the census taken at open (`RecordCensus`). The walker skips a tag
    it does not know (`walker.rs:140`), so the data would be invisible to the build. A
    compaction then carries forward only what substrate state holds
    (`compaction::collect_live_records`, `persistence/mod.rs:2195`), so it would delete
    the data. An upgrade meets this only when the newer build dropped a record type. A
    downgrade across a newly added type always does (§3.11);
  - **model:** the artifact's `model` is not the model the member runs, which the app's
    model lock admitted (§3.6). The new process would be refused 4412.

  The checks are the member's own (`upgrade/compat.rs`), made against its running
  build's `--build-info` and the census of every record type its substrates hold
  (`Store::census`: the types found at open, and every type appended since).

**Fetching is resumable and spreads.**

- **Chunks.** `BUILD_FETCH { artifact, offset }` asks for the next chunk, of up to
  1 MiB, sent as `BUILD_CHUNK` on the catch-up lane. It takes spare capacity only
  (§7.4), so distributing a build never slows live traffic.
- **On disk.** `<data>` is the member's state directory, or an agent's jobs directory.
  Chunks are written to `<data>/upgrade/<hash>/<sha256>.part`. The `.part` file *is*
  the resume point: a dropped socket, a restarted member or a vanished source picks up
  at the file's length (`upgrade/parts.rs`).
- **Verifying.** A finished artifact is checked against its SHA-256, and renamed to
  `<data>/upgrade/<hash>/<sha256>` (with the platform's executable suffix for a
  binary). A binary is then asked for its `--build-info`, which must name the build
  offered. On a mismatch the part file is discarded and fetched again from the start.
- **Peers become sources.** Every member announces with `BUILD_HAVE`, every 30 s and on
  admission, the build it runs, each one it keeps (§3.9) and each it verified. A member
  fetches with `from_node` 0, and the gateway sends each request to the next member
  holding that artifact, round robin, so fetches spread across the holders. A holder
  that lacks the artifact answers with an empty chunk, and a fetch that hears nothing for
  30 s asks again. A slow or lost source costs one member one resume, not the upgrade.
- **Ahead of certification.** A member fetches the build it is to install, or else the
  newest build offered above its own that is neither rejected nor one it refused. So the
  artifact is there by the deadline. Once the build is rejected, its downloads and part
  files are removed; a build a member runs or keeps stays, the way back to it.

**Installing: one member at a time, after certification.** At the grace deadline, if the
build is certified (§3.8):

1. **The floor rises for admission.** A new `HELLO` below it is refused. Connected older
   members are *not* closed while they are waiting to upgrade.
2. **Members upgrade one at a time per app.** A member may upgrade only while it holds
   the Paxos lease `upgrade/<app>` (§5.9). The app therefore never loses more than one
   member at once, and its capacity and quorum hold. For a `motherd`, which is a member
   of both apps, upgrading takes both leases, and it installs a build only once both
   apps certified it.
   - **The lease outlives the process.** It is taken for 20 minutes and is never
     renewed, so it is held through the restart and no other member installs while
     this one is down. A drain hands back every other lease, but not `upgrade/` ones.
   - **The process that starts takes it back.** Once ready, the new build, or the
     stable file `keep` started again, acquires each lease for 5 s as the same node,
     which for Paxos is a renewal, and releases it for the next member. The leases
     it held are in `<data>/upgrade/watch.json`.
3. **Each member that holds the lease and a verified artifact:**
   - **drains** (§12.4): hands back claims, worlds and characters, and waits for its
     outbox;
   - **flushes and releases** its substrate writer lock, and closes its sockets;
   - **stages the binary.** It copies the verified artifact to `keep/stage/<exe name>`
     under its working directory. That path is not the stable file, and it is not the
     executable the process is running from. The UI comes with the executable,
     compiled in. The running file is not moved;
   - **asks `keep` to start it.** One POST to `127.0.0.1:8092`, the body the staged
     path, then this process exits. `keep` is already running. It stops the process
     it had been supervising and starts the staged file with the same arguments in
     the same working directory. The plan at `<data>/upgrade/watch.json` is the lease
     list the next process reads. `self_heal` relaunches the executable it is running
     from, which is the stable file: `keep` starts every trial from the stable path.
4. **`keep` is the failsafe, and it is already running.** It is its own process. It
   does not certify, roll out, carry bytes, or choose a canary share. For `web` and
   `motherd` the last known-good executable is `keep/stable/<name>.exe`, started with
   the arguments in `keep`'s config:
   - **A process that is gone** is started again from that stable file.
   - **A trial** runs from the stable path: the stable file is set aside, the
     staged file copied onto its path and started there — the path the firewall
     allows, and the one a self-relaunch starts. `keep` reads its stderr, written
     to a file under `keep/state/`, for the line `INFO serving`, which the process
     prints only after it is serving, not while it is booting. The wait is at most
     10 minutes **from the newest sync data received**: a member catching up
     (§7.6) prints `INFO syncing …` each time its catch-up count rises, and every
     such line restarts the 10 minutes. A catch-up is as long as the backlog, not
     the build, so one that keeps receiving is waited out; one that stalls prints
     nothing and is rolled back 10 minutes after its last line.
   - **No line in that time, or the process dying,** kills the trial, puts the
     set-aside file back and starts it.
   - **The line** makes the trial the service: the set-aside file is deleted and
     the trial process goes on running. A promotion does not restart it.
   - **Success for the member** is the new process. Once it is ready it writes
     `<data>/upgrade/ok`, prunes kept builds beyond *N*, and hands the leases back.

   A stage that cannot be written, or a `keep` that does not accept the path, is
   not an exit: the old process relaunches itself (`upgrade/handoff.rs`). The
   executable it was running stays in place the whole time.

   `--after-pid` and `--upgrade-failed` are one-shot flags: `LaunchSpec::capture`
   drops them, so no later relaunch repeats them.

   **keep on its cluster socket.** Beside its loopback switch, keep declares
   routes on the socket it holds as the app `keep` (`keep/src/remote.rs`):
   `GET /v1/keep/state`, `POST /v1/keep/trial` — a trial run as a job, answered
   at once, its end read from the state — and `POST
   /v1/keep/service/:name/:action`. It serves only a request the gateway stamped
   with the member it came from. A member reaches a machine's keep over
   `group:keep`, naming the machine in `x-tokera-host`
   (`web/src/cluster/keeps.rs`); this is how an Admin's operations tools try a
   build on one machine (`docs/claude_zend_gateway.md` §8.4). A trial asked for
   this way is refused for an artifact whose `--build-info` names a `local`
   build.
5. **Members that cannot upgrade** are closed 4410 once the rollout settles, 120 s after
   the floor rose: stranded ones, failed ones, and ones that hold no verified artifact
   of the build and fetch none. A member still waiting for the lease, fetching or
   installing is left to finish. This is the §3.5 floor finally applying to them, with
   the reason each one is stuck. The rollout ends when no member is left below the
   floor, and the coordination log says so.

**Old builds are kept, the last *N* of them.** Each member keeps the builds it has run
in `<exe dir>/builds/`, one directory per build, named `<count>-<hash>`. A directory holds:

- the executable, its UI compiled in;
- `build.json`: the `--build-info` answer, the executable's SHA-256, and when the build
  was installed and when it was retired.

The rules for keeping them:

- **How many.** `--keep-builds <n>` on the member sets *N*; the default is 5. Once
  the new build is ready, the oldest directories beyond *N* are deleted. A trial
  that `keep` does not keep deletes nothing.
- **What is never pruned.** The build running now is not in `builds/`, so it is never
  pruned. A kept build is never pruned while it is the target of a forced rollout in
  progress (§3.11).
- **Where they go after a rollback.** Rolling back moves the running build into
  `builds/` in turn, so the build rolled away from stays available to roll forward
  to.
- **Kept builds are artifacts.** A member announces each kept build with `BUILD_HAVE`,
  like a verified download, so a peer that lacks one can fetch it from there.
- **Disk.** The status page shows the kept builds' bytes per member. A member in fault
  mode for disk space reports them as reclaimable, and `prune-builds` (§12.4) frees them.
  Fault mode never deletes them by itself, because they are the way back.

A kept build is what makes rolling back fast. Every member that ran a build still holds
it, so a forced rollout to it (§3.11) stages that executable and asks `keep` to start it, with no fetch,
no certification and no grace period.

**Manual upgrades still work.** A member rebuilt and restarted by hand is a candidate,
or an ordinary member if its build is already the floor. The automatic path is how the
rest of the group follows.

### 3.10 Local builds: testing uncommitted work against the cluster

A **local build** is one whose `local` digest is set: built with commits not yet on
`origin/main`, or with uncommitted edits (§3.5). It is how a developer runs their
work-in-progress on a machine that is part of the cluster. The gateway's behaviour is
unchanged.

**What is different about it.** Exactly two things:

1. **Its own listener serves locally, joined or not.** Requests reaching the member's
   own `--port` or `--tls-port` are served by its own code, never forwarded up (§5.8).
   That is the point of it: whoever hits the machine directly is testing the local
   build. `/v1/status` says so in its `cluster` block.
2. **It never takes part in an upgrade:**
   - It is **never a candidate**. A local build above the floor's `count` sets no
     pending floor, is not certified, and never offers its binaries, which no one could
     reproduce from `origin` (§3.8, §3.9).
   - It is **never upgraded automatically**. A rollout skips it, so it never overwrites
     the binary under test.
   - It is **never closed 4410 for being behind**. When the floor passes its `count`,
     it stays joined, and the status page shows it as *local, behind floor N*.

**Everything else is as for any member**, by decision:

- **Its writes replicate.** Conversations and acts made on it reach the group like any
  member's.
- **The gateway routes cluster traffic to it**, by the same dispatch, and it may hold
  claims, world leases and NPC leases.
- **It votes.**

Two consequences follow, both the developer's to manage:

- **Work-in-progress code serves real traffic.** A gateway sends requests to the local
  build like any member. A developer who wants only their own traffic on it drains it
  (§12.4): it stays joined, takes no gateway traffic, and its own listener still serves
  locally.
- **Its writes are permanent.** Anything it writes is in every member's substrate, test
  data included.

**When the work is pushed and rebuilt**, `local` is empty again and the member is an
ordinary build at its `count`. If that `count` is above the floor, it becomes a candidate
like any other (§3.8).

### 3.11 Forced rollouts and fast rollback: a build chosen by hand

The automatic path of §3.8–§3.9 picks the build itself, the newest certified one, and
takes it from the candidate. A forced rollout lets an admin choose both **which build**
the app runs and **which member** it comes from. It covers three cases:

- **Push a build out now,** without waiting out the grace period or certification.
- **Roll the whole app back** to a build it ran before, when a bad build got through.
- **Re-install the floor's build** on members left on another build, such as stranded
  members that were fixed by hand.

**The command** is `rollout { app, build, from, targets, parallel }` (§12.4). Every
argument but `app` and `build` is optional.

| Argument | Meaning |
|---|---|
| `build` | A `count-hash`: the build `from` is running, or one it keeps in `builds/` (§3.9). |
| `from` | The source member. On a member's row, *Roll out this build…* fills it in, and offers the member's running build and each build it keeps. Left out, the gateway picks among the members that hold the build. The Upgrades block lists every build held anywhere in the app with its holders, so *Roll back to…* there needs no source. |
| `targets` | Defaults to every member of the app not already running `build`, except local builds (§3.10). A local build is installed over only when it is named. |
| `parallel` | How many members install at once. The default is 1. |

**It is refused outright when:**

- `build` is a local build (`local` set). Nobody can rebuild it from `origin`, so it is
  pushed first;
- the source does not hold `build`, or is not connected;
- no quorum is reachable, because the rollout is decided by Paxos (below);
- `parallel` is more than the ready members can lose and still make the quorum, so
  Paxos stays available while members restart (§5.9). The seats are counted as Paxos
  counts them: a gateway's tie-break vote fills the empty seat of a group of two, so
  such a group rolls out one member at a time — without it, a two-member group could
  never be rolled out by hand at all, not even back to an older build in an emergency.

**The gateway answers with a plan before anything moves:**

- the direction: an upgrade, a downgrade, or a re-install at the same count;
- the floor before and after;
- each target's kind, platform and current build, and where its artifact comes from. Either
  it is the target's own kept copy, matched on count, hash and SHA-256, so nothing is
  transferred, or it is fetched, with the bytes to fetch;
- the members skipped, and why: already on the build, or a local build;
- the members stranded, and why: no artifact for their kind and platform, or a failed
  log-format, record-type or model check (§3.9);
- for a downgrade, the builds it will mark rejected.

The page shows the plan. The admin confirms it by typing the build's `count-hash`, since
a rollout restarts every target and so counts as destructive (§12.4).

**A downgrade across a new record type is stranded everywhere.** Records replicate, so
once a build has written a record type the older build does not know, every member's
substrate holds it. The record-type check (§3.9) then strands every target, and the plan
says so before anything moves. The way out is to roll forward to a fixed build.

**Deciding it.** The page sends the confirmed plan to the source member, or to the
gateway's pick of one. That member proposes the Paxos decision `rollout/<app>/<n>`
(§5.9). The plan is the value: build, SHA-256s, source, targets, `parallel`, who ran it,
and when. `n` is one above the last rollout decision the member holds. A member that is
behind may find `rollout/<app>/<n>` already decided. Paxos then shows it the decided
value, and it reads that decision and proposes again at `n + 1`. Once it is decided:

- **It is in every member's substrate** as a replicated record, and each member reports
  the latest one in `COORD`. So every gateway applies it.
- **A gateway restarted mid-rollout learns it** from the first member back, and carries
  on. The floor it sets comes from the decision, not from whichever member reconnects
  first (§3.5).
- **A second rollout decided while one runs supersedes it.** The member installing now
  finishes, because a swap is never interrupted. The next member gets the newer
  decision's build.

**What it overrides:**

- **The grace period and certification.** The floor becomes `build`'s count at once,
  in either direction. A pending floor is cancelled. A candidate's certification and
  distribution stop, and its part files are removed, unless the candidate's build is the
  one rolled out.
- **A rejection.** Rolling out a rejected build accepts it, as `certify-anyway` does. A
  candidate in fault mode for *certification failed* leaves fault mode.
- **A paused rollout.** A forced rollout starts unpaused, and can be paused like any
  other.
- **A failed install.** A member that once failed `build` (`--upgrade-failed`, §3.9) is
  retried only when it is named in `targets`.

**A downgrade rejects the builds it rolls back from.** When `build` is below the floor,
each build above it that the app has run or was certifying is marked rejected, as
`reject-build` does. Each rejection is written as a rejecting verdict under the next key
of `certify/<app>/<count>-<hash>/<i>` (§3.8, §5.9). It follows that:

- **A member that comes back on a rejected build is admitted drained.** That covers one
  restarted by hand, and one whose downgrade failed and was restored. It replicates and
  votes, takes no traffic, and is never a candidate.
- **A rejected build never sets the floor** when a gateway restarts. A member reads the
  verdicts on its own build from its substrate before it dials, and keeps them current
  once a second, and its `HELLO` carries `rejected` (§4.1). A gateway that has not yet
  heard the verdict therefore neither takes that member's build for the floor nor makes
  it a candidate.
- **A member the rollout names does not drain itself** for being on a rejected build:
  it is about to install the rollout's build, under a lease a drained member could not
  take.
- **`certify-anyway` lifts a rejection.** A build above the floor that the app never
  ran, such as the fix once it is pushed, is not rejected, and becomes a candidate as
  usual.

**Installing** runs as steps 2–4 of §3.9, except for the following:

- **Sources.** The decided source sends `BUILD_OFFER` to the targets. Members that verify
  the build become sources, and members that keep it serve it from `builds/`, so the
  source machine is not the only one carrying the transfer. A target that keeps the build
  itself fetches nothing.
- **Leases.** Members take one of `parallel` leases, `upgrade/<app>/<i>`, instead of the
  single `upgrade/<app>`, asking for a slot at random so members waiting together spread
  across them. A `motherd` also takes `upgrade/<other app>`. With the default of 1, this
  is the same as an automatic rollout.
- **Never pruned.** The rollout's build is protected in the watchdog's plan, so the
  prune after a successful swap never removes it (§3.9).
- **The swap from a kept build** moves that directory's executable back into place,
  and the running build into `builds/` (§3.9). The watchdog and the restore are
  unchanged.

**When it ends:**

- **Members below the new floor** that did not take the build are closed 4410 with the
  reason, as in step 5 of §3.9.
- **Members left above it after a downgrade** are on a rejected build, and stay drained.

**How fast a rollback is.** There is no transfer, no certification and no grace period.
Each member drains, moves two directories on its own disk, and restarts. Each member's
time is its model load and catch-up (§7.6), and the rollout takes that per member
divided by `parallel`. A higher `parallel` trades serving capacity for speed, and the
quorum bound keeps Paxos available throughout.

---

## 4. The socket

### 4.1 Frames

Each WebSocket binary message is one frame:

```
u8 kind | u8 flags | u32 stream (LE) | body…        (WebSocket supplies the length)
```

| Kind | Body | Lane |
|---|---|---|
| `HELLO` `CHALLENGE` `PROVE` `ADMIT` | §3.3 | control |
| `MEMBERS` | `{ epoch, members: [{node, host, app, build, weight, state, high_water}] }` — sent on every change and every 5 s; `high_water` is each member's own highest `oseq`, so any member can see that an origin has written something it does not hold (§7.1) | control |
| `CAPACITY` | `{ state, power, idle, slots_free, decodes, prefill_backlog, vram_free_mib, cores, load, …, prefill_tok_s, decode_step_next_us, decode_step_wide_us, prefill_tok_s_wide }` every 1 s or on change | control |
| `PING` / `PONG` | 8-byte nonce | control |
| `ROUTES` | the member's dispatch table (§5.2) | control |
| `CLAIM` / `UNCLAIM` | `{ key, epoch }` / `{ key, epoch, through }` — this member serves `key` (a sticky key or a lease), or has let it go with its own `oseq` at `through` (§5.4, §9) | control |
| `PREPARE` / `PROMISE` / `NACK` | `{ key, ballot }` / `{ key, ballot, accepted, from, reply_to }` / `{ key, promised, from, reply_to }` — Paxos phase 1 between members: a proposer asking, each member answering as an acceptor (§5.9). A request's sender is its ballot's node; a reply names its acceptor in `from`, so a quorum counts distinct acceptors however many clusters a reply came through | control |
| `PROPOSE` / `ACCEPTED` | `{ key, ballot, value }` / `{ key, ballot, from, reply_to }` — Paxos phase 2 | control |
| `RELEASE` | `{ key, ballot, through }` — a lease holder handing the lease back (§5.9, §9.6) | control |
| `BUILD_OFFER` / `BUILD_HAVE` | `{ build, artifacts: [{ kind, platform, size, sha256 }] }` — a candidate offering its build, and a member announcing what it runs, keeps or has verified (§3.9); the gateway also sends a member admitted below the floor the floor's `BUILD_OFFER` (§3.5) | control |
| `BUILD_FETCH` / `BUILD_CHUNK` | `{ artifact, offset, from_node, reply_to }` / `{ artifact, offset, reply_to, bytes ≤ 1 MiB }` — resumable artifact transfer from the member `from_node` names, or with `from_node` 0 from a holder the gateway picks; an empty `bytes` says the holder lacks it | control / catch-up |
| `VERDICT` | `{ build, result: pass \| fail \| no_opinion, figures }` — gateway → candidate: this gateway's certification verdict, sent again every 10 s until the group decides (§3.8) | control |
| `PULSE` | `{ world, kind, body }` — a transient world event: never persisted, no `oseq`, no retransmit (§9.1) | ordered, dropped first under pressure |
| `COORD` | the member's coordination state, for the status page: mode, generation, grants, leases, claims, catch-up, outbox, tombstones, clock (§12.1) | control |
| `COMPLETE` | `{ keys }` — member → gateway: keys another member let go that this member now holds complete, sent as each turns complete and whole again on every admission until each is claimed (`cooperative_admission.md` §2.2) | control |
| `OPEN` | `{ stream, target }` — open a virtual connection (§5.1) | ordered |
| `DATA` | ≤ 64 KiB of one stream's bytes | ordered |
| `WINDOW` | `{ stream, credit }` | control |
| `FIN` / `RESET` | end or abort one stream | ordered |
| `EVENT` | one substrate record (§6.3), fragmented above 1 MiB | ordered, or catch-up when `flags & CATCHUP` |
| `RECEIPT` | `{ through }` — gateway → member: every frame of this session up to sequence `through` has been taken in (§6.3) | control |
| `ACK` | `{ origin → applied_through }` — what this member has applied, its own origin at its highest `oseq` it may send among them; sent on admission and at most once a second as either changes | control |
| `CATCHUP` | `{ request, reply_to, to?, since: origin → applied_through }` — a member asking what exists after what it holds; `to` names the member to ask, or is absent for the gateway to pick (§7.2, §7.5) | control |
| `HEADERS` | `{ request, reply_to, answerer, origin, from, to, gone, records: [(oseq, type, stream_id, chunk_index, len, digest)] }` — what the answerer holds of one origin's range `from..=to`, a range it covers completely; `gone` lists the `oseq`s in it the answerer knows no longer exist, each one that was an expired tombstone carrying its deleted-set entry (§6.8, §7.2) | catch-up |
| `FETCH` | `{ request, reply_to, from_node, origin, oseqs }` — ask one member for the records themselves | control |
| `EVENT` with `flags & ADDRESSED` | a fetched record, carrying `reply_to` | catch-up |
| `DONE` | `{ request, reply_to }` — the last `HEADERS` of an answer, or the last record of a `FETCH` | catch-up |
| `LOST` | `{ request, reply_to }` — gateway → requester: catch-up or fetch `request` will get no answer, because no member can answer it yet or its answerer left (§7.2) | control |
| `HAVE` | `{ request, reply_to, applied set }` — a restarted member's offer (§7.5) | catch-up |
| `PEER_HELLO`, `ACME_CHALLENGE`, `ACME_ACK`, `ACME_CLEAR`, `ACME_CERT`, `ACME_LEDGER`, `ACME_INVENTORY`, `ACME_ORDERING`, `ACME_DONE`, `ACME_ACCOUNT` | between peer gateways only, never on a member's socket (§13.4) | control |

Bodies are little-endian binary with `u32`-length blobs, in the style of the persistence
codec (`record.rs:777-914`); a record inside `EVENT` is carried as the record's own
encoded header line and payload, unpadded. The codec lives in the new `cluster` crate (no
CUDA): the gateway never decodes a record — `web` cannot link `candle-conversation`
without becoming a CUDA build — so the record bytes stay opaque to it.

**Frame bodies, exactly** (`cluster/src/frame/`). Primitives: integers little-endian;
`blob` = `u32` length + bytes; `str` = a blob of UTF-8; `list<T>` = `u32` count + items;
`opt<T>` = `u8` 0, or 1 and the value; `bool` = `u8` 0 or 1. Anything else — a short body,
a byte past the end, a `bool` of 2, an unknown kind, a stream id on a kind that takes
none (or none on one that does), a flag on any kind but `EVENT` — is refused, never
misread, and on an admitted socket closes it 4400.

| Kind | Tag | Body |
|---|---|---|
| `HELLO` | 1 | `u16 proto` · `str app` · `u64 node` · `str host` · `build` · `bool rejected` (the group rejected this build, §3.11) · `opt<model>` · `u32 power` · `list<str> repos` · `u64 now_ms` · `[32] nonce` |
| `CHALLENGE` | 2 | `[32] nonce` · `[32] proof` |
| `PROVE` | 3 | `[32] proof` |
| `ADMIT` | 4 | `u64 session` · `list<member>` · `list<claim>` · `u32 floor_count` · `str floor_hash` · `bool routes_wanted` · `u64 own_applied_through` |
| `MEMBERS` | 10 | `u64 epoch` · `list<member>` |
| `CAPACITY` | 11 | `u8 state` · `u32 power` · `bool idle` · `u32 slots_free` · `u32 decodes` · `u32 prefill_backlog` · `u32 vram_free_mib` · `u16 cores` · `u32 load_milli` · `str os` · `u32 mem_free_mib` · `u32 jobs` · `list<str> toolchains` · `bool low_priority` · `u32 prefill_tok_s` · `u32 decode_step_next_us` (protocol 2) · `u32 decode_step_wide_us` (protocol 3) · `u32 prefill_tok_s_wide` (protocol 4) · `u32 prompt_tokens` · `u32 reply_tokens` · `u32 tokens_per_step_milli` (protocol 5) |
| `PING` / `PONG` | 12 / 13 | `[8] nonce` |
| `ROUTES` | 14 | `blob table` (§5.2's encoding) |
| `CLAIM` | 15 | `blob key` · `epoch` |
| `UNCLAIM` | 16 | `blob key` · `epoch` · `u64 through` (the key's own last state record's `oseq`, protocol 4) |
| `COORD` | 17 | `blob state` (§12.1's encoding) |
| `COMPLETE` | 18 | `list<blob> keys` — keys let go that the member holds complete (protocol 4; `cooperative_admission.md` §2.2) |
| `PREPARE` | 20 | `blob key` · `ballot` |
| `PROMISE` | 21 | `blob key` · `ballot` · `opt<ballot · blob value>` accepted · `u64 from` · `u64 reply_to` |
| `NACK` | 22 | `blob key` · `ballot promised` · `u64 from` · `u64 reply_to` |
| `PROPOSE` | 23 | `blob key` · `ballot` · `blob value` |
| `ACCEPTED` | 24 | `blob key` · `ballot` · `u64 from` · `u64 reply_to` |
| `RELEASE` | 25 | `blob key` · `ballot` · `u64 through` |
| `BUILD_OFFER` / `BUILD_HAVE` | 30 / 31 | `build` · `list<str kind · str platform · u64 size · [32] sha256>` |
| `BUILD_FETCH` | 32 | `[32] artifact` · `u64 offset` · `u64 from_node` · `u64 reply_to` |
| `BUILD_CHUNK` | 33 | `[32] artifact` · `u64 offset` · `u64 reply_to` · `blob bytes` (≤ 1 MiB) |
| `VERDICT` | 34 | `build` · `u8 result` (0 pass, 1 fail, 2 no opinion) · `blob figures` |
| `PULSE` | 40 | `str world` · `str kind` · `blob body` |
| `OPEN` | 50 | `str target` |
| `DATA` | 51 | the bytes, unprefixed, 1 to 64 KiB |
| `WINDOW` | 52 | `u32 credit` (nonzero) |
| `FIN` / `RESET` | 53 / 54 | — |
| `EVENT` | 60 | `u64 origin` · `u64 oseq` · `u32 total` · `u32 offset` · [`u64 reply_to` when `ADDRESSED`] · the fragment's bytes, 1 B to 1 MiB, never past `total` |
| `RECEIPT` | 61 | `u64 through` |
| `ACK` | 62 | `list<u64 origin · u64 applied_through>` |
| `CATCHUP` | 63 | `u64 request` · `u64 reply_to` · `opt<u64> to` · `applied` since |
| `HEADERS` | 64 | `u64 request` · `u64 reply_to` · `u64 answerer` · `u64 origin` · `u64 from` · `u64 to` · `list<u64 oseq · opt<blob> deleted_entry>` gone · `list<record header>`; every `oseq` within `from..=to`. `answerer` is the member that holds the records, whom the requester's `FETCH` names |
| `FETCH` | 65 | `u64 request` · `u64 reply_to` · `u64 from_node` · `u64 origin` · `list<u64> oseqs` |
| `DONE` | 66 | `u64 request` · `u64 reply_to` |
| `HAVE` | 67 | `u64 request` · `u64 reply_to` · `applied` |
| `LOST` | 68 | `u64 request` · `u64 reply_to` |
| `PEER_HELLO` | 70 | `u64 node` · `build` · `[32] session_key` · `[32] oauth_client` |
| `ACME_CHALLENGE` | 71 | `str site` · `str token` · `str answer` · `u64 expires_ms` |
| `ACME_ACK` / `ACME_CLEAR` | 72 / 73 | `str token` |
| `ACME_CERT` | 74 | `str site` · `str chain_pem` · `str key_pem` · `u64 not_after` |
| `ACME_LEDGER` | 75 | `blob ledger` (JSON) |
| `ACME_INVENTORY` | 76 | `list<str site · u64 not_after · [32] digest>` |
| `ACME_ORDERING` / `ACME_DONE` | 77 / 78 | `str site` |
| `ACME_ACCOUNT` | 79 | `blob account` (JSON) |

where `build` = `u32 count · str hash · str local`; `model` = `str arch · [32] weights ·
[32] tokenizer · str rope · str describe`; `member` = `u64 node · str host · str app ·
build · u32 weight · u8 state · u64 high_water`, the state 0 loading, 1 ready,
2 upgrading, 3 fault, 4 draining; `claim` = `blob key · epoch`; `epoch` = `u64 ms ·
u64 node`; `ballot` = `u64 round · u64 node`; a record header = `u64 oseq · u8 type ·
u64 stream_id · u64 chunk_index · u32 len · [16] digest` (45 bytes); and `applied` is
§7.5's packing. `EVENT`'s flags are `CATCHUP` = `0x01` and `ADDRESSED` = `0x02`. The
stream field is chosen by whoever opens the stream: odd ids for the gateway, even for a
member, so the two never pick the same one. A stream's reader returns credit in a
`WINDOW` once 64 KiB of it is consumed.

### 4.2 Interleaving

Every frame is bounded (`DATA` ≤ 64 KiB, an `EVENT` fragment ≤ 1 MiB), so no single
transfer holds the socket: a 63 MB snapshot is 63 fragments, between which every other
stream's frames and every control frame get through. The writer serves three lanes:

1. **control** — always first.
2. **ordered** — one FIFO holding stream frames and live substrate events, in the order
   they were produced. The order matters (§6.4).
3. **catch-up** — only when the ordered lane is empty *and* the socket's unsent bytes
   are below the low-water mark (§7.4).

Per-stream flow control: each virtual connection starts with 1 MiB of credit each way and
is topped up by `WINDOW` as the reader consumes, so a slow browser stalls only its own
stream.

### 4.3 What the gateway relays

| From a member | To |
|---|---|
| `EVENT` (live) | every *other* admitted member of the same app — never back to the sender, never to another app, and not to a member that is upgrading (§3.5), lagging (§4.4) or in fault mode for a full disk (§7.7) |
| anything carrying `reply_to` that answers a request — `HEADERS`, an addressed `EVENT`, `DONE`, `LOST` | **only** the member `reply_to` names, so catch-up traffic reaches the one member that asked for it |
| `CATCHUP` | the member `to` names; without `to`, one other ready member of the app, picked round robin (§7.2) |
| `FETCH` | the member `from_node` names |
| `HAVE` | every other admitted member of the same app (§7.5) |
| `BUILD_OFFER`, `BUILD_HAVE` | every other admitted member of every app, agents included, once per member: a `motherd` build carries the `tokera-agent` an agent takes (§3.9). The gateway keeps who holds what, for admitting members below the floor and for the Builds held list |
| `BUILD_FETCH` | the member `from_node` names; with `from_node` 0, the next member holding the artifact, round robin, never the requester. Its `BUILD_CHUNK`s go only to `reply_to`, on the catch-up lane |
| `PULSE` | every other admitted member of the same app; dropped, never queued, when a member's ordered lane is past half its cap (§4.4) |
| `PREPARE`, `PROPOSE`, `RELEASE` | every other admitted member of the same app, members in fault mode included, since they still vote (§7.7), and upgrading members excluded, since they do not (§3.5); the gateway only carries them and holds no Paxos state (§5.9) |
| `PROMISE`, `NACK`, `ACCEPTED` | only the member `reply_to` names |
| `OPEN`/`DATA`/`FIN`/`RESET` | the other leg of that virtual connection (§5.1) |
| `CLAIM`, `UNCLAIM` | kept by the gateway for routing, and relayed to every other admitted member of the app (§5.4) |
| `CAPACITY`, `ROUTES`, `ACK`, `COORD` | kept by the gateway for routing and the status page; a candidate's `CAPACITY` turning ready starts its grace (§3.8), and the upgrade view in `COORD` carries the group's verdicts and rollouts the gateway follows (§3.5, §3.11) |

A WebSocket delivers in order, so frames need no sequence field: both ends count the
frames a member sends on a session, 1, 2, 3 and on. The gateway sends `RECEIPT` for the
count it has *relayed* every 100 ms, or every 1 MiB, whichever comes first. A frame
counts as relayed once it sits in every recipient's outbound queue, not merely when it
has been read off the socket. A gateway that dies between reading a frame and queuing
it has therefore not receipted it, and the sender sends it again.

A gateway that dies after queuing a frame but before writing it loses that frame for
its recipients. Each of them recovers it through its own catch-up, on the reconnect the
gateway's death forces (§7.1).

**An `EVENT` larger than one fragment is reassembled by the receiver**, keyed
`(origin, oseq, reply_to)`, from fragments relayed as they arrive. The `reply_to` is
part of the key because a socket carries a live lane and a catch-up lane: one record
can be in flight on both at once — live to every member, and as the answer to a member
catching up — with the two copies' fragments interleaved. Keyed by the record alone,
the answer's first fragment read as a restart of the live copy, and the gateway closed
the sender's socket. Fragments of an event whose relaying socket drops before the last
one are discarded. The sender re-sends the whole event (§6.3).

A copy of the same key that begins again — its first fragment, the same length — while
another is partly held replaces it, and the restart is logged with how much was
dropped: a sender only ever sends a record whole again, never from where it stopped,
so the earlier copy will not be continued. Closing the socket instead cut every request
riding it (WhiteWorld, 97 MiB held of a 113 MiB snapshot). The earlier copy carrying on
after all is an interleave, and still closes it.

A gateway relays only among the members connected to *it*. An event leaves its origin
through one cluster (§6.3), so a member joined to every cluster receives it once, from
that cluster. A member whose socket to that cluster was down misses it and gets it from
its catch-up (§7). Gateways never relay to each other: the members, joined to several
clusters, are what carries a record across them. That rests on **every member joining
every cluster** (`--cluster` for every gateway, §15.1). A member configured with fewer
never sees the others' members in `MEMBERS`, so no `high_water` tells it what they
wrote; their events reach it only through the catch-ups it runs for other reasons.

### 4.4 Gateway queues

Each member has an outbound queue per lane.

- **The ordered lane drops nothing it has queued, and stops taking live events from a
  member that cannot keep up.** When the lane passes 256 MiB unsent, the member becomes
  **lagging**:
  - the gateway queues no further live `EVENT`s for it, and counts what it skips;
  - it is never a catch-up's answerer and takes no request routed round robin: those
    need a member current on every origin;
  - **a conversation may still be placed on it — current enough is a conversation
    dimension, not a member one.** A new conversation has no history to be behind on;
    a released one goes only to members holding its last holder's records through the
    handover (`cooperative_admission.md` §2.2), which a lagging member fails exactly
    while it lacks them. Excluding the member as a whole kept WhiteWorld out of every
    burst that landed after a busy `.5` heavy-decode: its KV and snapshot writes pass
    the cap over a 1 Gbps link in seconds (run 26, 2026-10-20: all 48 of heavy-read's
    placements listed `.5` alone);
  - stream frames already queued are still delivered, so the requests it is serving
    finish rather than fail. Each stream's own flow control bounds them (§4.2).

  The member sees the gap through `high_water` and catches up (§7.1), at its link's
  pace. **Once it has drained the lane it receives live events again and is
  syncing** — the state of a member admitted beside one already serving: sent live
  events throughout, placed conversations as above, routed no other request until its
  `ACK`s cover what the members serving
  have applied of every origin, within 64 records (`SYNC_SLACK`), as the gateway judges
  from those `ACK`s. Waiting instead for its
  catch-up to reach every origin's `high_water` before sending it live events waits
  for ever while any origin keeps writing — the catch-up trails the writes it is not
  sent — and so it did: WhiteWorld's zend, lagging behind `.5`'s ingest, was routed
  nothing for as long as `.5` ingested (2026-10-18). Closing the socket would instead
  fail every exchange on it (§5.6), for a member that is only slow.
- **The catch-up lane is capped at 128 MiB per member.** The requester sets the pace: it
  keeps no more than 64 MiB of fetched records outstanding (§7.2), so an answer stays
  under the cap. A catch-up moves one window per round trip — the answerer's reads, the
  gateway hop and the spare-capacity gate, about 170 ms on `.5` — so the window is the
  throughput: at 8 MiB it moved ~46 MB/s, and two members whose npcd each wrote faster
  than that kept 34 and 40 GB backlogs of each other's writes (2026-10-17). A frame that would pass the cap anyway is dropped and counted. The
  requester sees the missing records when its `FETCH` ends with `DONE` short of them,
  and fetches them again. **A record larger than the cap counts as the cap**: it is
  fetched alone (§7.2) and crosses an empty lane. The cap bounds the backlog, not
  what crosses it — weighed by its own length, a record over 16 MiB was dropped on
  every try and never arrived (npcd on `.5`, 2026-10-17: 6 of 192 records, 24 of
  21,290 MiB, in ten minutes).

---

## 5. Routing API traffic

### 5.1 Virtual connections under hyper

The socket multiplexes byte streams. Each stream is an `AsyncRead + AsyncWrite` that both
ends hand to the HTTP machinery they already run:

- **Member side.** `cluster::Member` yields an incoming `VirtualConn` for each `OPEN`,
  from whichever of its sockets the stream was opened on.
  `host::server` serves it exactly as the TLS entrance serves a TCP connection —
  `hyper_util::server::conn::auto::Builder::serve_connection_with_upgrades` over the
  same router (`web/src/tls/tcp.rs:82-107`) — inserting `ConnectInfo` with a synthetic
  peer and a `ClusterPeer { session }` extension. `Gateways` gains a variant that trusts
  `ClusterPeer` (`web/src/auth/gateways.rs`), so `x-tokera-*` headers are believed on
  connections carried by an admitted socket and nowhere new. The outer `web::Builder::router()` is what
  is served, so Host-based site dispatch, UI files, `/auth/*` and compression behave as
  on the TCP entrance.
- **Gateway side.** A `Route` keeps its configured `upstream` URL and gains an optional
  `cluster: <app>` (`web/src/config.rs:336-380`). It is in clustered mode while that app
  has a ready member (§5.7). Then `proxy::forward` (`web/src/proxy.rs`) keeps its header
  rewriting, streaming and upgrade splice and swaps only its transport. Instead of the
  pooled TCP client (`web/src/upstream_tls.rs`), it:
  1. picks a member (§5.2);
  2. opens a stream to it;
  3. runs `hyper::client::conn::http1::handshake(...).with_upgrades()` over that stream.

  One request per stream. The stream lives on that gateway's socket to that member, so
  the response returns through the gateway that took the request.
- **Member-originated streams.** A member may `OPEN` towards the gateway (target: an
  app). The gateway serves that stream with its own router, which proxies it to
  `group:<target>` — the same path as a browser request. This is how zend reaches an
  agent (§11).

No handler, guard or extractor changes. SSE and NDJSON are response bodies; uploads are
request bodies; `/ws/logs` and `/ws/events` are upgrades on a stream — all carried by
the same frames.

The proxy stops forwarding `x-tokera-assertion` and the `Cookie` header to members: no
daemon reads the assertion (`npcd/src/identity.rs:17-22`), and it is a replayable
30-day session token (`docs/web_gateway_design.md` §5).

### 5.2 Every route declares its dispatch

`Api::route(path, min, method)` becomes `Api::route(path, min, dispatch, methods)`
(`host/src/guard.rs`), and `Guarded` gains `dispatch` and the methods it answers.
The handlers are built with `host::guard`'s `get`, `post`, `put`, `delete` and `patch`,
which record each method beside the handler — a `GET` answers `HEAD` too, as axum does —
so a read and a write of one path are two registrations, each with its own dispatch:
`GET /v1/world/:wid` goes `Any` and `PUT` `Best`. A mount (`nest_self_authenticating`)
names one dispatch for its path and everything under it.

A route also declares its **mirror mode** for certifying a new build: `Exact` with its
volatile fields, `Shape`, or `None` (§3.8). It defaults to `None`, so a route that writes
is never mirrored by omission.

The table a member sends in `ROUTES` is `declared(roles)` with both columns. The gateway
matches request paths against it with the same matcher axum uses (`matchit`).

| Dispatch | Picks |
|---|---|
| `Best` | the member with the most power per request in flight (§5.3) |
| `Sticky(key)` | the member holding `key`'s affinity; a key with none goes `Best` and takes it (§5.4) |
| `Owner(key)` | the member holding the lease for `key`: the world's authority for a world, the NPC-lease holder for a character (§9.6) |
| `Aggregate(merge)` | every admitted, ready member; responses combined (§5.5) |
| `Any` | round robin |
| `Ui` | the member that served the page this request belongs to (below) |

`key` is one of the following:

- `Path(param)`;
- `Header(name)`;
- `Body(field)`, a top-level JSON field read from a buffered request body of at most
  8 MiB;
- `Derived(name)`, a named key function in the `cluster` crate, computed from the
  buffered body.

**UI files come from one member per page load** — for a site that forwards its pages to
its members. The consoles' sites do not: the gateway serves their pages from its own
disk and forwards only `/v1` and `/ws` (`web/web.yaml`), so this rule never fires for
them. It governs a site whose `/` route is clustered. Members at the same build count
serve the same compiled-in pages, but two members of different builds do not, and a
page whose HTML came from one member and whose script came from another could break.
So:

- The member that serves a page's HTML is chosen round robin among ready members. The
  gateway answers with a cookie, `tokera_ui_<app> = <node>`, scoped to the site and to
  the life of the browser session.
- Every UI request carrying that cookie goes to that member while it is ready.
- If the member has left or is in fault mode, the request goes to a new member and the
  cookie is replaced. The page's existing build check (`checkBuild`,
  `zend/web/index.html:550-552`; `web/content/common/lib/build.js:33-35`) sees a
  different build id and reloads the page whole, from the new member.
- API requests ignore the cookie and route by their own dispatch.

A request may also carry `x-tokera-node: <node>` (admin only): it goes to that member
whatever the route declares. This is how the status page and the per-machine
instruments address one machine.

Routes by group (the full table is in each crate's `declared`):

| Routes | Dispatch |
|---|---|
| UI files (`/`, `/common/*`, `/brand/*`, every page and asset) | `Ui`: one member per page load (below) |
| `GET /v1/models`, `/v1/tools`, `/v1/schema/layers`, `/v1/me` | `Any` |
| zend `POST /v1/chat/completions` | `Sticky(Body("conv_id"))`, with `Header("x-tokera-conversation")` taking precedence — the zend GUI sends it so the gateway never buffers a chat body. A passthrough request carries no `conv_id`: its key is `Derived(passthrough)`, the conversation's opening — every system message, the client's tools, and each message before the first reply, length-prefixed under SHA-256 (`cluster::dispatch::conversation_key`). It is implemented once in the `cluster` crate, which the gateway links to route by it and zend links to name the conversation by it (`Transcript::key`), so the two cannot disagree |
| zend `/v1/conversations/:id*` | `Sticky(Path("id"))` |
| zend `/v1/conversations` (list), `/v1/repo_map`, `/v1/substrate/zend/*` reads | `Any` — replicated state |
| zend `/v1/substrate/zend/*` writes (retiring the system prompt, tombstoning a timeline, a projection probe) | `Best` |
| zend `/v1/substrate/zend/timeline/:tl/selection` | `Aggregate(FirstFound)` — in memory on whichever member last projected it |
| zend `/v1/substrate/zend/layer/:name/toggle` | `Aggregate(Envelope)` — each member's own kill switch. A maintenance pass is the `maintain` command (§12.4) |
| npcd `/v1/npc/:nid/*` | `Owner(Path("nid"))` |
| npcd `/v1/interaction/:ix*` | `Owner(Path("ix"))` — interaction ids are minted as `<nid>-<n>`, so the owner is in the id |
| npcd `/v1/world/:wid/*` live state (channel, posting, the internal `act` route), `/v1/pulse/operations*`, `/v1/pulse/missions*` | `Owner(Path("wid"))`, the world's authority (§9.2); a route among them that names no world goes `Best` |
| npcd `/v1/life/:who/*` | `Sticky(Path("who"))` — a generation and the job reporting on it stay on one member |
| npcd `/v1/simulate`, `/v1/generate/*`, `/v1/image/*`, `/v1/guest/image`, the effector mount `/v1/local/*` | `Best` |
| npcd authored reads (`/v1/world`, `/v1/personality`, `/v1/mind/*`, `/v1/npc` list, `/v1/me/profile*`, `/v1/life/catalog`) | `Any` |
| npcd authored writes | `Best` — they become VFS events (§10.3) |
| `/v1/status`, `/v1/telemetry`, `/v1/memory`, `/v1/phases`, `/v1/projections`, `/v1/layers`, `/v1/profile`, `/v1/substrate`, `/v1/tools/calibrate`, npcd `/v1/pulse`, `/v1/pulse/census`, `/v1/pulse/world`, `/v1/pulse/broadcast`, `/v1/pulse/announce`, `/v1/pulse/command-table`, `/v1/pulse/guardian`, `/v1/guest` | `Aggregate(Envelope)` — what each member holds or does for the characters it runs |
| `/ws/logs` | `Aggregate(Lines)` — one browser socket, every member's lines interleaved, each tagged with its node and host |
| `POST /v1/cluster/command/:name`, `GET /v1/cluster/command/job/:id`, `PUT /v1/cluster/quorum` | admin only; sent with `x-tokera-node` to one member, or once per member for "all" (§12.4) |
| npcd `/ws/events`, `/v1/commands` | `Any` |

### 5.3 Power and idle capacity

`CAPACITY` carries:

- **`power`**: the member's sustained aggregate decode rate — the peak of the last
  hour's `decode_tps` series the telemetry ring already keeps
  (`host/src/telemetry/ring.rs:59-82`). Until the member has decoded, its power is its
  VRAM total in MiB divided by 100: a ranking of last resort that a single measurement
  replaces.
- **`idle`**: the scheduler can admit now — a free slot and a prefill backlog under its
  budget (`slots`, `budget`, `used` in the wave rows, `host/src/endpoints/waves.rs:43-70`).

The gateway adds what no `CAPACITY` can say in time: **`in_flight`**, the requests it
has handed each member and the member is still answering. A request routed `Best`,
sticky or owner is counted at the pick — in the same hold of the registry, so two picks
at once each see the other — and until the last byte of its answer: a chat streams for
as long as it decodes. An upgraded connection is a session, not a request, and is not
counted.

`Best` picks the ready member with the most power to spare per request,
`power / (1 + in_flight)`, among those with room — idle, or with more free slots than
requests in flight — and among them all when none has room. Of equal shares an idle
member wins, then the lowest node. A member's own background work — ingest, a
calibration pass — is not a request and does not count: the dialogue a request serves
decodes at `High` priority, 64 shares to background work's one (`DecodePriority`). So the strongest member takes new conversations until its share per
request falls to another's, and a burst spreads over the members as it arrives rather
than all reading the same second-old `CAPACITY`. Agents rank by idle cores (§11.3).

**Once every candidate has learned what a turn costs it, `Best` ranks by when the turn
would finish instead** (`docs/cooperative_admission.md` §2.1, protocol 4). `CAPACITY`
then also carries the member's **`prefill_tok_s`**, its prefill rate at the width its
scheduler runs, **`decode_step_next_us`**, its decode step with one row more than
it decodes, and **`decode_step_wide_us`** and **`prefill_tok_s_wide`**, its decode
step and prefill rate with `PLACEMENT_WIDE` (16) rows more. All four come from the
rates its admission learns from every forward. A turn's finish on a member is:

- its `prefill_backlog`, plus a nominal prompt of 2,048 tokens for each request
  handed to it since its last `CAPACITY` and still in flight, and for this one, at
  the time per token those requests and this one leave it: on the line through
  `1 / prefill_tok_s` (one row) and `1 / prefill_tok_s_wide` (16), extended past
  it, and flat when the wide rate is no slower;
- then a nominal 512-token reply at the step with those requests and this one added:
  on the line through `decode_step_next_us` (one row) and `decode_step_wide_us` (16),
  extended past it, and flat when the wide step is no slower.

The nominal sizes are the same for every member: the gateway sees a request's new
message, not the prompt a member's projection makes of the conversation. The soonest
finish wins, among those with room. A member that has not yet learned both rates says
zero; it takes a turn while it is answering none — one at a time, to learn from — and
the rest go by finish among those that have. With none learned, the pick is by power per
request.

### 5.4 Affinity is a claim the member holds

Active-active gateways cannot each keep a private affinity table, because two of them
would send one conversation to two writers. So affinity is held by members, as
**claims**, and every gateway routes by what it is told.

**Taking a claim.**

1. A gateway that knows no claim for a sticky key routes the request `Best` (§5.3).
2. The member checks its own view of claims. If no live claim exists for the key, it
   takes one before serving: `CLAIM { key, epoch }`.
3. It announces the claim on every cluster it has joined.
4. Each gateway keeps the claims it hears and relays them to the app's other members,
   like events. A member admitted later is sent the gateway's claim table in `ADMIT`,
   so every member holds the full table.

**The epoch is `(ms, node)`.**

- `ms` is `max(now_ms, the highest ms this member has seen in any epoch for the key + 1)`.
  So a successor's epoch is above its predecessor's whatever the two machines' clocks
  say. A clock that runs slow cannot make a legitimate takeover sort before the holder
  it replaced.
- `node` breaks the last tie.

What a cut-off member wrote is ordered by its generation (§6.7), which comes before the
epoch in the stamp, so the epoch does not need to say whether a claim was taken cut off.

**Routing by a claim.**

- A gateway routes a key to its claim holder whenever that member is connected to it.
- When the holder is not connected to this gateway, the request goes `Best`. The member
  it reaches knows the claim from its relayed copy and answers `409` naming the holder
  in `x-tokera-holder`.
  The gateway answers the browser `503` with `Retry-After`, because the holder is
  reachable only through another cluster and the next attempt may land on one.

**When a claim ends.** A claim lives while its holder is a member of at least one
cluster, as seen in `MEMBERS`, and either:

- has served the key within the idle timeout (10 minutes, `--cluster-affinity-idle`) —
  npcd and the agent, whose work outlives the request that started it; or
- **is still in use**, for an app whose claims follow use (zend,
  `docs/cooperative_admission.md` §2.2): a request uses its claim until its response
  ends, a turn until its last record is written, and the claim goes with its last use,
  so a conversation's next turn is placed afresh.

Either way the holder releases it with `UNCLAIM { key, through }`, where `through` is
its own `oseq` at release. A gateway sends the key's next request only to a member that
already holds those records (§2.2 there), and the member that releases forgets what it
kept of the key in memory.

A holder that left every cluster loses it without an `UNCLAIM`:

- "Every cluster" is as the member taking over sees it, through the gateways it is
  joined to.
- The gateways are independent, so two members joined to different gateways can
  disagree. The former holder then keeps writing while its successor writes too, and
  the cost when they meet is a reload on the members whose turn order moves (§6.6).

**The epoch is written into the turns.** The member stamps its claim's epoch on the
turns it writes for that timeline (§6.6). A later claim has a later epoch, so the turns
of a former holder that was partitioned and kept writing are ordered before its
successor's.

**Taking over waits for the new member to catch up.**

- A member that takes over a released key holds the request until it has applied the
  previous holder's records through `through`.
- When the holder vanished without releasing, the member first runs a catch-up (§7) and
  holds the request until its `DONE` arrives. What the holder's sender-side retransmit
  had not yet delivered arrives when the holder returns (§6.3), and takes its place by
  stamp (§6.6).
- The hold is bounded at 30 s. Past that, the request is answered `503` with
  `Retry-After` rather than served from a substrate that is behind.

**Two first claims made at the same moment through two gateways** can only happen if one
client sends a conversation's first message to both. Both claims are then live, and the
higher epoch keeps the key. The other member sees the winning `CLAIM` relayed to it and
releases its own. Both first turns stay in the conversation, ordered by stamp (§6.6).

### 5.5 Aggregates

`Aggregate(Envelope)` fans the request out to every ready member with a 3 s deadline
each and answers:

```json
{ "members": [ { "node": "3f…", "host": ".5", "build": 4815, "status": 200, "body": { … } },
               { "node": "a0…", "host": ".6", "build": 4815, "status": 504, "error": "deadline" } ] }
```

with `x-tokera-aggregate: 1`. The consoles read it through
`web/content/common/lib/aggregate.js`, which takes a process's own answer as it is:

- **the loading screens and build checks** (`memberStatus`) — the status of a member
  ready for the tenant, since the gateway sends the console's requests to ready members;
  until one is, the first member that answered; with none, "connecting";
- **zend's telemetry page** (`pick`) — one member at a time, chosen from a member picker
  shown when more than one answered;
- **the substrate overview** (`memberWith`) — the first member that holds the tenant's
  line;
- **the npcd console's instruments** — the one member's answer, as npcd's group has one
  member (§16, item 19).

`/v1/status` aggregated also carries a `group` block — floor, model lock, member count —
that the gateway adds. `FirstFound` returns the first `2xx` body unwrapped.

An aggregate covers the members connected to the gateway that took the request. With
every member joined to every cluster, that is all of them. The envelope lists the
members it reached, so a partial view reads as partial.

### 5.6 When a member's socket drops

A gateway that loses a member's socket fails every HTTP exchange riding on it at once.
It never leaves one hanging, never completes one as if it had succeeded, and never
replays one on another member.

- **No response head yet:** the browser gets the gateway's `503 upstream_unavailable`
  with `Retry-After` (`web/src/errors.rs:57-76`), the same retryable page or JSON as a
  dead upstream today.
- **Response in progress** (SSE, NDJSON, a download): the gateway aborts the response
  body with an error rather than ending it. Over HTTP/1.1 the connection closes without
  the terminating chunk; over HTTP/2 and HTTP/3 the stream is reset. The client sees a
  truncated response, never a short one that looks complete. zend's GUI already treats
  that as a cue to reattach to `/live` (`zend/web/index.html:1518-1521`). The reattach is
  routed by the claim, which outlives the socket because the member is still joined to
  its other clusters.
- **A tunnelled WebSocket** (`/ws/logs`, `/ws/events`): closed with `1011`, and the
  browser's own reconnect loop takes over (`web/content/common/lib/ws.js`).
- **Member-originated streams** (an agent job, §11) fail the same way in the other
  direction. The calling tool reports the job as failed with the reason, and does not
  retry it.

When the dropped socket was the app's last ready member on this gateway, the app falls
back to its configured upstream (§5.7). Requests that arrive after that go there.
Requests that were in flight on the socket still fail as above; they are never moved
across.

On the member, the dropped socket resets every virtual connection it carried. Handlers
see a disconnected client, which they already handle: a zend turn runs on to completion
(`zend/src/session.rs:207-228`). The member reconnects that socket alone (§3.2). Its
other clusters carry on.

### 5.7 Dumb mode and clustered mode

The gateway keeps its configured upstreams. Each app is in one of three modes, on each
gateway independently:

| Mode | When | Routes to |
|---|---|---|
| **Dumb** | the app has no member admitted on this gateway | the route's configured `upstream` URL, exactly as today (§1.1): one fixed daemon, passive backoff, IP-trusted identity headers |
| **Loading** | members are admitted and none is ready | the member that joined first, as below |
| **Clustered** | the app has at least one ready member here (§7.2) | members only, by each route's dispatch (§5.2); the configured URL is ignored |

```yaml
# web.yaml — the zend site's route keeps its upstream and names its app
api:
  - { prefix: /, upstream: "https://192.168.0.5:8444", self_signed: true, cluster: zend }
```

**Switching to clustered** happens when the first member of the app becomes ready.
Requests already in flight to the configured upstream finish there. New requests go to
members. A route with no `cluster:` key never switches.

**Reverting to dumb** happens when the app's last ready member on this gateway leaves:
its sockets close, it is closed 4410, or it falls behind and stops being ready. New
requests go to the configured upstream again. In-flight requests on the lost sockets
fail as §5.6 says.

**While every member is loading, the first of them to join answers.** With members
admitted and none ready, every request for the app goes to the member that joined
this gateway first — the lowest admission session among those not ready
(`Registry::standing_by`) — and that member serves it itself. Its loading page shows
where the app's page would, and its own endpoints answer what the app's pages read
while it is down: `/v1/telemetry` and the other performance readings, `/v1/agents`.
The gateway's reconnecting page (`503` with `Retry-After`, `route::restarting`) is
the answer only when no admitted member is left to take a request. One member is
chosen, not each in turn, so a page and the calls it makes all see the same startup:
a reader watches one member's progress rather than cards from several. The status
page names that state **loading** rather than dumb (`status::mode_name`). The
configured upstream serves only while no member is admitted at all.

**The configured upstream gets nothing special in clustered mode.** It is often the
same daemon as one of the members, for example zend on `.5`. While the app is
clustered, the gateway sends it traffic only through its socket, if it has one. Its
direct listener stays up and behaves as §5.8 says.

**A gateway with no members at all** is today's gateway, with the same table, the same
backoff and the same error pages. Adding clustering to a deployment changes nothing
until a member joins. Removing every member puts it back.

Mode is per app and per gateway. One gateway can run zend clustered and npcd dumb.
Two active-active gateways agree on an app's mode whenever every member is joined to
both.

### 5.8 The member's own listener: forward up when joined, serve when cut off

A member keeps its own HTTP server: the `--host`/`--port` listener and the
`--tls-port` entrance (`host/src/server.rs:87-142`). What it does with a request
depends, per app, on whether the member is joined:

| Member state for the app | A request arriving on the member's own listener |
|---|---|
| **Joined**: at least one of the app's sockets is admitted and ready (§7.2) | is **forwarded up** to a gateway and answered from wherever the cluster routes it |
| **Rejoining** or **reloading**: admitted, catching up and taking its generation (§6.7), or rebuilding its substrate (§6.6) | is answered **`503` with `Retry-After`**; the member serves nothing while its turn order may still move. A rejoin that cannot reach a quorum stays cut off instead (§6.7) |
| **Upgrading** (§3.5) | is answered `503` with `Retry-After`, naming the build it is installing |
| **A local build**, joined or not (§3.10) | is **served locally**, so whoever reaches the machine directly tests its work-in-progress |
| **Cut off**: no admitted socket for the app | is **served locally**, exactly as a standalone daemon serves it today |
| **Any state, for a request the gateway sent here in dumb mode** | is **served locally**, never forwarded (below) |

**A request a gateway routed here in dumb mode is never forwarded back up.** In dumb
mode a gateway proxies to the route's configured URL (§5.7), which is often this very
listener. Forwarding that request up would send it back to the same gateway, which would
send it here again, without end.

- So a gateway marks every request it proxies to a configured upstream with
  `x-tokera-hop: dumb`.
- A listener serves a marked request itself, whatever its state.
- The listener believes the mark only from an allowlisted gateway IP, as it believes
  every `x-tokera-*` header (§1.2). A client cannot set it to bypass the cluster.

**A daemon started without a reachable cluster serves locally.** At startup it dials
every `--cluster` (§7.6). It does not wait on the loading page for one to answer:

- once every cluster has failed one attempt, or has been dialling for 10 s, it is cut
  off and serves locally;
- it keeps dialling in the background, and its first ready socket makes it rejoin
  (§6.7).

A laptop started off the network therefore works at once. A refusal that stops
reconnecting (4410, 4426) still leaves it on its loading
page with the reason, because those say the build itself is wrong.

**Forwarding up.**

- The member opens a stream to a gateway, target = its own app (§5.1). It deals these
  round robin across its admitted sockets, as it deals events (§6.3).
- It writes the request onto the stream with the identity it resolved on its own
  listener, from IP-trusted `x-tokera-*` headers or `--local-signin`.
- The gateway believes identity headers on a member-originated stream, because the
  socket is authenticated (§13.2). It routes the request by dispatch and claims (§5.2,
  §5.4), exactly like a browser request.
- The response streams back down the same stream and out of the member's listener.
  SSE, NDJSON, uploads and WebSocket upgrades work as on any stream.
- The cluster may route the request straight back to this member, for example because
  it holds the conversation's claim. It then arrives on the socket and is served there.
  A request that arrived on a socket is always served, never forwarded, so nothing loops.

**One route never forwards: `GET /v1/status`.** It always answers for this machine,
`cluster` block included (§7.6), so a health check keeps meaning "is this daemon up".
The gateway's aggregate is where the group's status lives.

**The switch is per request, at arrival.**

- A request already being served locally when the member joins finishes locally.
- A request already forwarded when the member's last socket drops fails, as §5.6 says
  for the gateway side, and the client retries locally.
- The app's state flips the moment its first socket becomes ready, and again the moment
  its last one drops.

**What this buys.** A member cut off from every cluster carries on working, for whoever
can reach it directly. That includes a laptop off the LAN, or a LAN machine whose
internet link is down. It writes to its own substrate as it always has, and its own
events wait as *detached* and unsent (§6.3, §6.7). When a socket comes back, §6.7's
rejoin runs:

1. Its catch-up runs first (§7.2), so it gets what the group did, all of it placed
   before what it did itself.
2. Its detached events take their fixed generation and go up, so the group gets what it
   did while away, placed after the group's own.
3. Once ready, its listener forwards up again, and the bigger cluster carries its load
   with it.

**Claims and leases while cut off.**

- **zend.** A cut-off member serves every request locally, including conversations
  whose claim another member held when it lost contact. Before writing, it takes a
  local claim on each key it serves (§5.4), and announces those claims when it rejoins.
  If the group wrote to the same conversation meanwhile, both sides' turns stay: the
  group's first, the cut-off member's after (§6.6–§6.7).
- **npcd.** A cut-off member switches its worlds to disconnected mode (§9.4). Its own
  characters, those whose NPC lease it holds, carry on in a local world that is never
  persisted and is scrapped on reconnect. Their memories persist as detached events.
  Characters leased elsewhere are not run here. Their routes answer `503` while cut
  off.

### 5.9 Consensus: Paxos among the members

Some decisions must be made **once, by the group**: which member is a world's authority
(§9.2), and group-wide values every member must agree on. They are decided by Paxos
among the members of the app.

**Who does what.**

- **The members are the acceptors and the proposers.** Each keeps its acceptor state in
  its own substrate, so these are durable acceptors in the classic sense.
- **The gateways carry the vote, and one of them breaks a tie.** They relay Paxos
  frames between the members of the app, as they relay events. The gateway with the
  lowest node id is also an acceptor: a `PREPARE` or `PROPOSE` that arrives on a
  member's socket is answered on that socket, with the tie-break flag. Its journal is
  `gateway-votes` beside the group key, the same layout as a member's `substrate/VOTES`,
  and a restart resumes it. A promise so flagged counts as one vote, and only when the
  group size is two — one member plus the gateway is a quorum of two, so zend and npcd
  keep deciding with one of two members down. It does not count in a larger group: a
  quorum of two is also three members' majority, and one member plus a gateway must not
  decide while the other two can. A gateway with peers does not vote until it has seen
  every configured peer's node id, and it remembers a peer after the link drops, so a
  higher id never votes while a lower one exists. A gateway with no peers is the voter
  at once. The gateway writes no decision record; the members do.

**The quorum is a majority of the largest group the app has had: `max_members`.**

- **Persisted in the substrate, not configured.** `max_members` is a keyed record,
  `cluster/max_members`, that replicates like any record. A member that sees more
  distinct ready members at once than the stored value, counted across every cluster it
  has joined, writes the higher number. No configuration file names the group's size,
  and every member, even one that has just restarted alone, knows the quorum from its
  own substrate.
- **The quorum is `⌊max_members / 2⌋ + 1`.** A member cut off from its peers cannot
  reach it on its own. That is what stops a partition's minority from deciding anything.
- **A proposer counts every ready member it sees as well.** Its quorum is a majority of
  `max_members` or of the ready members visible now, whichever is greater. The stored
  value rises only on the next once-a-second read, and a member arriving in a group
  still counted as one would otherwise make a majority of one with itself, in that
  second, and take a lease another member holds.
- **Raises and lowers resolve in any order.**
  - The value is the latest *set* by version stamp (§6.2), raised to the greatest raise
    stamped after it.
  - A raise is a max, so raises commute. A set is a stamped override, so the last set
    wins.
  - The result is the same in any arrival order (§6.5).
- **It can be lowered** by writing a smaller *set*:
  - `--cluster-quorum-max <n>` on the command line, written at startup;
  - `PUT /v1/cluster/quorum { max_members }`, admin only, from the status page or any
    member.

  Lowering is for retiring machines for good, and for a deployment that knowingly
  accepts less safety: for example two members that should keep electing when either
  is down.
- **It is lowered by itself when the group stays short**
  (`cluster/src/replicate/quorum_governor.rs`). A member left with fewer ready members
  than `max_members` — a laptop gone off the network, a machine retired without a
  *set* — would otherwise leave the rest unable to decide anything: no generation for a
  member that wrote cut off, no verdict, no rollout, no lease. Each member watches the
  ready members its gateways report:
  - after **5 minutes** short it **proposes** the lower value — `max(ready, 2)` — in its
    log and in `/v1/status` (`cluster.apps.<app>.replication.quorum.downgrade`, with how
    long the group has been short and how long until the write);
  - it **executes** — writes the *set* — as soon as no lease can be granted twice:
    - **down to two, through the voting gateway** (one cluster, and a reply flagged as
      the tie-break seen on its socket in the last 10 minutes): once the leases its own
      acceptor holds for members not ready have lapsed — at once with none, within a
      world lease's 30 s with one. The gateway answered every round, so it accepted
      every lease and decision the old group was asked for, and every quorum of the
      lowered group — two members, or one and the gateway — holds an acceptor that
      reports such a lease for its whole term;
    - **otherwise after 20 minutes** short, the longest term any lease is taken for
      (the upgrade install's, `READY_WITHIN` + 5 min). An absent member cannot renew
      without a quorum, so by then every lease it held has run out on every acceptor;
  - **never below two**, where the voting gateway's tie-break completes a majority,
    so one member and the gateway decide but a lone member never does;
  - **only from a member that sees the whole group**: joined, in no fault, not drained,
    and admitted and ready on every cluster it is configured for. A member cut off from
    a gateway sees part of the group, and lowering from a partial view would let both
    sides of a partition decide;
  - a member's return withdraws a proposal, and once it is seen ready again the usual
    *raise* puts the value back.

  A value the absent members accepted and never relayed to the rest can be lost by the
  lowering. With one gateway that is the tie-break acceptor for every round, every
  proposal reaches it, and a later round adopts what it accepted.

**Two primitives, one implementation** in the `cluster` crate, on the substrate:

| Primitive | Decides | Used for |
|---|---|---|
| **Lease** `(key, ttl)` | who holds `key` for a term, renewable and releasable | world authority (§9.2); installing a build, `upgrade/<app>` or a forced rollout's `upgrade/<app>/<i>` (§3.9, §3.11); any role exactly one member must play at a time |
| **Decide** `(key, value)` | one value for `key`, once; a later decision needs a new key, such as `key/2` | values the group must agree on: the generations minted at rejoins, `gen/<k>` (§6.7); certification verdicts, `certify/<app>/<count>-<hash>/<i>`, the latest `i` holding (§3.8); forced rollouts, `rollout/<app>/<n>` (§3.11); an app's rollout paused or resumed, `pause/<app>/<n>` (§12.4) |

Claims (§5.4) are *not* decided by Paxos. A sticky key is served by whoever routing sends
it to, and its races already resolve by stamp (§6.6), so it never pays a consensus
round.

**One round**, for one key, by a proposer:

1. **Prepare.** It sends `PREPARE { key, ballot }` to every member, through every cluster
   it has joined. A ballot is `(counter, node)`.
   - An acceptor promises if the ballot is above every ballot it has promised for that
     key, and replies `PROMISE { key, ballot, accepted }`.
   - Otherwise it replies `NACK { key, promised }`.
   - A promise is durable before it is sent: the acceptor writes it and syncs it.
     Acceptor state is this member's own vote, so it is never replicated.
   - **It lives in a vote journal, not the redo log.** `substrate/VOTES` is preallocated
     at first open, 64 MiB of fixed-size slots, one per Paxos key, each holding the
     promised ballot and the accepted ballot and value. Each write is in place, and the
     file never grows, so a member can always vote, **even with its volume full**
     (§7.7).
     - A slot carries a checksum and a sequence number, and is written to alternate
       halves. A write torn by a crash leaves the previous vote readable, never a
       half-written one.
     - A key that has not been touched for one `ttl` past its last lease frees its slot.
     - A key whose decision has been written as a replicated record frees its slot
       too. The acceptor then answers for that key from the decision: a `PROMISE`
       carrying it as accepted at a ballot above every other, so every proposer
       adopts it, and a `NACK` for a proposal of anything else.
     - The layout (`cluster/src/consensus/votes/file.rs`): a slot is two 1 KiB halves,
       so 64 MiB holds 32,768 keys at once. A half is `magic "VOTE" · u64 seq · u16
       key_len · [128] key · promised ballot · u8 accepted · accepted ballot · u16
       value_len · [800] value`, zero-padded, ending in a CRC-32 of the rest. A key is
       at most 128 bytes and a value at most 800.
     - A value is `u8` tag then, for a decision, its bytes, or for a lease `u64
       holder · u64 ttl_ms`.
2. **Propose.** With promises from a quorum, it proposes the value of the highest-ballot
   accepted value among them, or its own if there is none, as `PROPOSE { key, ballot,
   value }`. Acceptors accept, durably, unless they have promised higher, and reply
   `ACCEPTED`.
3. **Decided.** With `ACCEPTED` from a quorum, the value is decided. The proposer writes it
   as a replicated record, so every member reads the decision from its substrate. The
   record is a `CustomObject` keyed `cluster/<key>` (`record.rs:1027-1040`), so it needs
   no new record type. Paxos guarantees one value per key, so two members that both
   write the decision write the same value; the key holds the greater stamp, which every
   member agrees on (§6.5).

**Leases add a term.** A lease value is `(holder, ttl)`.

- An acceptor treats an accepted lease as present only while its `ttl` runs, counted on
  its own monotonic clock from when it accepted.
- The holder counts from when it sent `PREPARE`, and stops 2 s before the end, so it
  always stops before any acceptor would let another proposer in.
- A `PROMISE` carrying another member's unexpired lease stops a proposer, which routes
  to that holder instead. One carrying the proposer's *own* lease does not stop it, and
  that is renewal.
- `RELEASE { key, ballot, through }` makes acceptors drop the lease at once. Every
  member hears it, with the releasing member and how far its writes reach, so the next
  holder can wait for them (§9.6).
- **A lease is asked for again when the group grows.** One granted under a smaller
  quorum was accepted by a majority of the smaller group, and the members that joined
  since never saw it: a majority of the larger group made of them could grant it a
  second time. So a holder whose quorum rises above the one a held lease was last
  granted under renews that lease at once, and a majority of the group as it stands
  knows it (`replicate/lease.rs`, `reaffirm_leases`).
- Because the acceptors are durable, a restarting member resumes its promises from its
  substrate, and needs no sit-out period for decisions. A lease it had accepted before
  the restart counts as present for one full `ttl` from the restart. Its monotonic clock
  began again, and that errs on the side of the old holder.

**Routing the frames.** Paxos frames are sent to every member of the app on every
cluster the proposer has joined. Replies carry `reply_to`, and the gateway delivers them
to the proposer alone (§4.3). Each member drops a frame it has already handled, keyed
`(kind, key, ballot, from)`, so a frame that reaches it through two clusters is handled
once.

**Who proposes.** For a lease, the member that weighted rendezvous names (§9.5).
Rendezvous only keeps members from duelling over ballots; Paxos decides. For a decision,
whichever member needs it first.

**A proposer never asks for a lease its holder still holds.** While the member holding
it is in the group — known from its claim on the lease's key, which every member
receives with its admission before it may propose, or from a round that came back
`LeaseBusy` — the proposer waits for it to be handed over. Asking again would only raise
the ballot under the holder's renewals, back them off, and run a live holder's lease
out.

**Duelling proposers back off.** Two members whose views of the membership differ can
both think rendezvous names them, and out-ballot each other in turn. So a proposer that
receives a `NACK` waits a random time, drawn uniformly from 0 to the key's `ttl` capped
at a decision's 2 s, before preparing again with a ballot above the one that beat it.
One of them then completes a round while the other waits, and the other learns the
result from its next `PROMISE`. The cap matters for long leases: drawn over an
`upgrade/<app>` lease's term — minutes — two members asking for it together could both
sit out for minutes, and a duel is over in a round or two however long the lease.

- Only a promise *above* the round's ballot beats it. A `NACK` naming the round's own
  ballot answers one of the proposer's earlier rounds, delayed, from an acceptor that
  has since promised this one, and is ignored.
- A phase that gathers no quorum within 2 s backs off the same way: frames can be
  lost, and a round must not wait on them forever.
- A member that rendezvous stops naming for a key abandons its round for it, so it
  does not keep duelling the member that is named now. A lease it holds is unaffected;
  it runs out or is released.
- **A lease asked for again while its round runs starts nothing.** The round retries
  itself until it is held or a holder's promise stops it (`LeaseBusy`), so a second ask
  could only replace it with a higher ballot. The import lease's load asked once a
  second: each ask cancelled the member's own round before it could finish, and pushed
  the ballot up under the holder's renewals, so nobody held `git/<repo>` long enough to
  pull it and a joining member waited for good.

---

## 6. Replicating the substrate

### 6.1 What replicates

Every record a **runtime writer** appends through `append_record` or
`append_recurrent_parts` (`persistence/mod.rs:948-1046`) is an event, K/V `Chunk`
records and recurrent `Snapshot` records included (D5). Not events:

| Record | Why it stays local |
|---|---|
| `HeaderIndex` | derived from the local log's own offsets |
| `ModelKey`, `Tokenizer` | the binding; equal on every member by the model lock (§3.6) |
| `Template` | the projection schema compiled into the build; equal under one build floor |
| `BranchCheckpoint` | a content-keyed cache; any member can rebuild it |
| maintenance relocations (`append_raw_record`), maintenance re-emissions, full `compact()` | housekeeping of records that already have an identity |

Provenance is the entry point, not a flag. `append_record` and `append_recurrent_parts`
**originate**: they mint the record's identity (§6.2). `append_carried` writes a record
under the identity it is handed — a maintenance re-emission, a compaction, and a remote
apply (§6.5). `append_raw_record` stages a record's bytes verbatim, header and identity
included. Only originating appends are tapped. `HeaderIndex` is the one record written
with no identity at all (every identity field zero).

### 6.2 Identity

- **The header gains six fields** (`record.rs:351-372`). Housekeeping preserves all
  six: a relocation copies the header verbatim, and a re-emission or compaction
  re-encodes the record with the values it carried. That needs the resident in-RAM
  state to keep each value's stamp beside it.

  | Field | Meaning |
  |---|---|
  | `origin: u64` | the writing substrate's node (§3.7) |
  | `oseq: u64` | that origin's event count, from 1 |
  | `gen: u64` | the generation (§6.7): the writer's current generation, or ⊤ for a record written while cut off, until a local `Rebase` fixes it |
  | `hlc: u64` | a hybrid logical clock reading, `(wall_ms << 16) \| counter`: each record takes `now_ms << 16` when that passes every reading this member has taken or seen, and the highest such reading + 1 otherwise. Causally later records always carry a later `hlc`, whatever the wall clocks say, and the high 48 bits still read as milliseconds for tombstone ages (§6.8) |
  | `epoch` | the claim or lease epoch (§5.4) the record was written under; zero for records no claim governs |
  | `digest: u128` | the existing `ContentHash` (`persistence/content_hash.rs:1-24`) over the whole payload. It covers what the `crc` does not: a `Chunk`'s metadata, and every other payload |

  `(gen, epoch, hlc, origin)` is the record's **version stamp**: a total order every
  member computes the same way. §6.5 uses it wherever two records compete for one key,
  and §6.6 orders a timeline's turns by it.
- **A value and its record carry one identity from the start.** A runtime write that
  changes the substrate in memory and hands the record to the persistence writer thread
  mints the identity *when it changes the value*, through the minter the persistence
  layer shares (`persistence/identity/minter.rs`), and the writer appends the record
  under it (`append_carried`). The in-memory value is applied by the same stamped rule
  as any other record, so two local writes that reach the log in the opposite order to
  their minting still leave the substrate holding the greater one, as a replay will.
  Records whose value lands in the substrate only after the append — `Tokens`, the
  recurrent records, K/V chunks — are minted by the append itself.
- **The applied set is also where the catch-up is up to.** It is a per-origin version
  vector: `applied_through` plus the ranges above it already held. `oseq` only ever
  grows, so one number per origin says how far this member has caught up on that
  origin.
  - It is persisted as a new resident record, `Replica` (tag 28), written at each group
    commit in which it changed. Compaction keeps the latest. On open the set is the last
    `Replica` **plus** the `(origin, oseq)` of every record the walk meets, so records
    appended after the last `Replica` are never lost from it.
  - A member's **own** origin is not in it: its own `oseq`s so far are all applied, and
    the walk recovers the next one. So a member that never applies another origin's
    record writes no `Replica` at all. `Replica` is local housekeeping, never
    replicated, and carries no identity (`persistence/replica/`).
  - A catch-up cut short by a disconnect or a restart resumes from the last `Replica`,
    never from the beginning.
  - `oseq`s a catch-up learned exist nowhere any more (§7.2) are recorded in it too.
    Without it, a record superseded and dropped by compaction would look missing and be
    asked for again on every catch-up.
- **An `(origin, oseq)` → location index** sits beside the applied set and is rebuilt
  by the same walk. Maintenance relocates records, so the sender reads an event it
  owes, and an answerer reads a fetched record, through this index. A location captured
  earlier would be stale. An `oseq` whose record maintenance dropped is gone, and its
  own origin knows it. An origin's `oseq`s are dense, so the index is a vector per
  origin of 48 bytes per record: one word for the segment and the record's offset in
  4 KiB blocks, and beside it what a catch-up's headers list of the record (§7.2), so a
  listing never reads the disk. The record's size is read from its header when it is
  read.
- **A substrate written before this design** has no origins. Its first writable open
  runs the full `compact()` once, stamping as it writes (`persistence/adoption.rs`):
  every record an origin writes takes this node's origin, a fresh `oseq`, the current
  generation and a clock reading. There is no mixed mode — a record with no identity
  cannot be named, fetched or ordered against another member's, and the file events'
  payloads changed form with their naming (§6.2, VFS) — so adoption does not wait for
  `--cluster`: it runs before anything is served, and the store opens again.
  - The rewrite writes each timeline's turns in turn order and labels in creation
    order, so the stamps keep every ranking the store had: each turn stays at its
    index, and each conversation at its place in the listing.
  - A section's records take its declaration's new `oseq` as their generation; a turn
    tombstone names this node as the turn's writer; file events and their tombstones
    gain this node as their writer.
  - Local housekeeping keeps no identity: the header index, the model binding, the
    branch-checkpoint cache, and the member's `Replica`, `Deleted` and `Rebase`
    records.
  - The trigger is a record an origin writes that carries no identity. A tombstone
    without one is not a trigger: this member writes its repair tombstones that way
    (§6.5). **Two substrates that were once copies of each other must not both be adopted
  this way.** Each would claim the shared history as its own, and every member would
  hold that history twice under two origins. A machine joins from a fresh substrate and
  fetches the group's history by catch-up (§7.3). Only the first member of a group
  adopts a pre-existing substrate.
- **`TimelineId`** becomes `(unix_ms << 20) | r`, where `r` is 20 random bits, bumped
  per process to stay monotonic (`projection/ids.rs:373-407`). It keeps time order at
  millisecond grain and makes a collision between members need the same millisecond and
  the same 20 bits. The new ids are above 2^60, above every microsecond id already
  written, so old and new never meet. zend dialogue ids stay `hash(conv_id)`.
- **zend passthrough timelines become `hash(conversation key, n)`** — the key §5.2
  routes it by — like dialogue ids,
  in place of an allocated id (`passthrough::conversation_timeline`). Two members that
  serve the same passthrough conversation then name one timeline, not two. `n` counts
  the conversations already opened under the key: a client that edits its history
  opens a new conversation under the same key, and takes the first `n` whose timeline
  names nothing yet — neither registered nor deleted (`ConversationEngine::
  is_timeline_free`). Its routes are `Sticky` on the same key (§5.2).
- **npcd interaction ids** become `<nid>-<counter>` (§5.2).
- **Turn streams are indexed per writer** (§6.6): the stream id is unchanged, and the
  index keys each stream `(stream_id, origin)`, so two members' turns at one index never
  share a record key. **VFS events** are keyed `(timeline, origin, seq)`
  with `seq` per origin, and a `VfsTombstone` kills `(origin, seq)` pairs, so two
  writers' events never collide either (`persistence/vfs/index.rs`). A conversation's
  events read back by key and, within a key, in stamp order. A `VfsTombstone` that
  kills another member's event is kept by compaction, because that event can still
  arrive after it and must still die; one that killed only this member's own events
  has nothing left to kill once compaction drops them.

### 6.3 Emitting

The tap hands each `Origin` record's encoded header line and payload to the member's
cluster client: a bounded, non-blocking queue per socket, feeding that socket's ordered
lane.

**Which socket an event takes.**

- **An event a request caused** (§6.4) goes out on the socket that request arrived on.
- **A catch-up answer** (headers, fetched records, `DONE`) goes out on the socket its
  `CATCHUP` or `FETCH` arrived on. That gateway is the one that reaches `reply_to`.
- **Every other event** is dealt round robin across the app's admitted sockets, one
  whole event at a time, so its fragments never split across sockets. This covers the
  persistence thread's K/V, a background ingest and an npcd character's own turn. The
  load spreads evenly over the clusters.

**The log is the source of truth, not the queue.** Persistence never waits on the
network. The cluster client keeps no event bytes in memory, only `oseq`s. It reads an
event from the local log when it sends it, through the `(origin, oseq)` → location index
(§6.2), because maintenance may have moved it since it was written. An unsent event
that maintenance has since dropped as superseded is not sent: what superseded it is
sent instead, and a catch-up records the dropped one as gone. Each of the member's own
events is in one of three states:

| State | Means |
|---|---|
| unsent | written to the log, not yet put on any socket — every socket was down, or its queue was full |
| in flight on socket *s* | sent on *s*, not yet covered by a `RECEIPT` from *s*'s gateway |
| received | covered by a `RECEIPT`; the gateway has taken it in and relayed it |

**Retransmission is the sender's job.**

- When a socket drops, every event in flight on it goes back to *unsent*. Those are
  exactly the events it sent since the last `RECEIPT` it got, the ones with no round
  trip behind them.
- Unsent events are re-sent in `oseq` order on whichever admitted socket takes them
  next: a surviving cluster at once, or the same cluster when it reconnects.
- A reconnect therefore needs no resync of the member's own events. Everything the
  gateway might not have is still owed and goes again, and anything a receiver already
  holds is a duplicate, skipped by `(origin, oseq)` (§6.5).

A request-bound event whose socket dropped is re-sent like any other. Its request has
already failed (§5.6).

**Events are applied as they arrive.** One origin's events may travel through different
clusters and reach a receiver out of order. They are applied in the order they arrive.
§6.5 says why that is safe.

### 6.4 Events before the response

A request's response ends only after the events it caused are on the socket ahead of
it (`host/src/cluster/flush.rs`). A `GET` or a `HEAD` writes nothing, so its response
ends with its body: waiting out the writer's queue for one held every console status
poll behind the cast's seals until the gateway's 3 s deadline, and the page read that
deadline as the daemon going away. When the handler's body is done, a request that
can append takes a
`WriteTicket` — the substrate writer's enqueue counter (`persistence/writer.rs` queues
jobs FIFO) — and waits until every job at or below it is in the log, which taps its
records. Every own event still unsent then goes on the request's socket, and one that
already left round robin on another socket is waited for until that gateway has
receipted it. Only then does the response end: a body of unknown length sends its
terminating chunk, a body of known length its last frame, held until now, and a response
with no body its head. Stream frames and events share that socket's one ordered lane
(§4.2), and the gateway relays a member's events in arrival order. So any request *that
gateway* routes to another member afterwards lands behind those events on that member's
ordered lane, and that member serves a stream a gateway opens only once every record that
arrived before it is applied (`AppMember::run`). The request sees what they wrote.

Events no request takes wait 100 ms (`GRACE`) before the round robin sends them, so a
short request's own events leave on its socket rather than on whichever socket the round
robin would have picked.

A later request taken by a *different* gateway has no such ordering. For a sticky key
the claim covers it: the request goes to the claim holder, which wrote the events, and a
takeover waits to catch up (§5.4). An `Any` read of replicated state through another
cluster is eventually consistent. It sees the write once that member has applied it,
normally within one relay hop.

This covers the records a turn writes while it runs: `StreamDecl`, `Tokens`, `Label`,
`ConvState`, `VfsEvent`, `TurnIndexPage`, `WideQSig`, `ProjectionEvents`. A turn's K/V
`Chunk` records and its `Commit` are written later, when the persistence thread moves
the turn warm → cold (`persistence/thread.rs:1244-1302`, `writer.rs:527-554`). They
follow on the ordered lane one persistence pass after the response, and the affinity
catch-up barrier (§5.4) is what keeps a moved conversation from outrunning them.

### 6.5 Applying a remote record

The member's cluster client applies `EVENT`s itself, in batches, under the substrate
write lock. That is how the persistence thread installs tiers today
(`persistence/thread.rs`), so the scheduler is not involved. A projection that reads
the substrate sees each batch whole or not at all.

**As they arrive, with no reorder buffer.** The applied set is a version vector *with
gaps* (§6.2). An event that overtook a lower one is recorded exactly, and the lower one
is still taken when it arrives.

**The requirement: arrival order must not matter.** Once a member has received a set of
events, its substrate must be the same whatever order they arrived in. Tombstones are
the exception by design: they remove what they name.

Replaying a record twice is harmless today, but applying two records in the opposite
order is not, because nearly every apply overwrites (§1.3). Members receive one origin's
events in different orders, and several origins' events in different interleavings, so
every apply must be **order-independent**.

The open walk needs the same thing. A member's log is in *arrival* order now, so
replaying it at the next start meets records in an order no writer chose.

**The rules.** Each applies at runtime and in the walk alike, through one code path.

| Records | Rule |
|---|---|
| Keyed values: `StreamDecl`, `Chunk` and `Tokens` locations, `Snapshot` (an empty payload is a value meaning *retracted*), `BranchCheckpoint`, `ConvState`, `TreeMetadata`, `DebugId`, `CustomObject`, `Npc`, `Distilled`, `ProjectionEvents`, `WideQSig`, `TurnIndexPage` | **Greatest version stamp wins** (§6.2). The substrate keeps each value's stamp beside it, and a record replaces the value only if its stamp is greater. The result is a max, so it is the same in any order. |
| `Label` | Per field: each of `conv_id`, `label` and every `custom` key keeps the value with the greatest stamp. |
| `Commit` | `committed_through = max(…)`, never overwrite. |
| conversation order (`conv_order`, `substrate.rs:6015-6031`) | Taken from the `hlc` of the record that first named the `conv_id`, not from a counter that counts arrivals. |
| `TurnCoupling`, VFS kills, applied-set gaps | Set inserts, which commute already. |
| **Turn streams** (`Chunk`, `Tokens`, `WideQSig`, `TurnIndexPage` under a turn's stream id) | **Indexed per writer.** The index keys a turn stream `(stream_id, origin)`, so two writers' turn *N*, which share `turn_stream_id(timeline, N)`, never overwrite each other's entries. A lookup by stream id, which is all any caller does, resolves to the turn that holds that index in the timeline's stamp order (§6.6). That is decided by the stamps, not by which arrived last. |
| **Records that need a parent**: `TreeMetadata`, `DebugId`, `TurnCoupling`, `Label`, `ConvState`, and every timeline-scoped value | **Wait for the parent.** A record whose timeline is not registered yet is held, with its identity, in a pending map keyed by timeline (`substrate/pending.rs`), and applied by the decl that registers it. Nothing is dropped for arriving early. A stream's own records (`Chunk`, `Tokens`, `WideQSig`, `TurnIndexPage`, `ProjectionEvents`, `Commit`) need no holding: they are indexed under their stream as they arrive, and a stream another origin wrote is not an orphan for want of a decl (§6.8). |
| **Timeline `Tombstone`** | **Dominates.** It is a fact about the timeline, not a value. Every record for that timeline, earlier or later, is dead at read: no projection, listing or gather sees it, and every rewrite drops it. A remote record it dominates is not written at all (below). `register_timeline` registers it **dead** rather than refusing it: a refused registration would turn the next append of a writer that has not heard of the deletion into a fault, while a dead one is filtered at every read. Its `Snapshot` stream is **retired**: the tail goes, and a snapshot arriving after the tombstone, however late and whatever its stamp, is not installed (`Substrate::retire_snapshot_stream`; a `Distilled` marker retires it the same way). Compaction **keeps** timeline tombstones as resident records, because a record replicated late, say by a member that was cut off, must still find the tombstone and die. A tombstone **expires** once a complete compaction has run since it and it is more than *N* days old (§6.8). |
| Turn `Tombstone` | Dominates its turn the same way. It names the turn as its writer wrote it — the index the writer gave it and the writer's origin (`turn_writer`) — because a turn's index is its position in the timeline's stamp order, which another writer's turn can move (§6.6). It commutes with the timeline tombstone, which subsumes it. |
| **Section generations** | A section's **generation** is named by the writer of the `StreamDecl::PromptSection` that started it and that decl's `oseq`. Every record of the section carries the generation's `oseq` in its header's `epoch` (`Minter::mint_for`; the minter knows the generation of each section it is writing, and recovers it at open from its own records). A `SectionTombstone` names the generation it ends, `{writer, oseq}`, and that generation is ended for good: its decl is refused and every record that carries it is dropped, before or after the tombstone (`substrate/ended_sections.rs`). A decl that is installed drops whatever its writer's stream still held of an earlier generation. The live generation is the decl with the greatest stamp that no tombstone names. A tombstone written before generations names none, and keeps the append-order rule every unstamped record has: it ends the unstamped stream as the log stands, and an unstamped decl after it starts it again. |

**Deciding whether to write.** In order:

0. **Unknown type:** the record's type tag is not one this build decodes. This happens
   while builds are mixed (§3.5, §3.11). Writing it would gain nothing: the walker skips
   an unknown tag (`walker.rs:140`), and compaction keeps only what substrate state
   holds, so the record would be appended unread and later deleted. Marking it applied
   would be worse, because no catch-up would ever fetch it again, even after this member
   is upgraded.
   - So the record is not written and not marked applied. Its `(origin, oseq)` and its
     type, as its header names it, go in a **deferred** set, kept in `Replica` beside
     the applied set (§6.2). The name, not a numeric tag: a header names its type, and
     a build that does not know the type has no tag for it either.
   - A catch-up does not ask for a deferred `oseq`, so the record is not fetched over
     and over on a build that cannot read it.
   - When a build that knows the tag opens the substrate, it moves those entries back to
     wanted, and its first catch-up fetches them from a member that holds them. The
     origin always does.
1. **Duplicate:** `(origin, oseq)` is already in the applied set. Skip without writing.
2. **Dominated:** a tombstone covers the record's timeline or turn, or its section
   generation has ended, here (`Substrate::dominates`). Skip it, and mark it applied.
   - **A turn's declaration is the exception: it is written, dead at read.** It is the
     one record of a turn that names the timeline; a turn stream's own records name
     only their stream. Written, it makes every record of the stream arriving after it
     dominated too. Skipped, those records would arrive as records of no timeline and
     be kept for a declaration that never comes (§6.8).
   - A stream's own record arriving *ahead of* its declaration cannot be judged, and is
     written. The declaration's arrival makes it dead at read like the rest.
3. **Superseded:** the value the record sets is held here under a greater stamp
   (`Substrate::outranks`). It would lose on arrival and be dropped by the next
   rewrite, so skip it and mark it applied. The test is the stamp, never the `digest`:
   two copies of the same bytes under different stamps are two values, and every member
   must hold the greater. A third record ranked between them wins against one and loses
   against the other, so members that kept different stamps for one value would
   diverge on it. A `Label` merges field by field and a `Commit` takes the greatest
   mark, so neither is ever superseded whole; a set insert never is.
4. **Otherwise** verify the `digest`, and the golden or CRC (`record.rs:550`), then
   append it locally as `Remote`, with padding added back and every header field kept.
   That gives it a local location. A record that fails verification is refused and
   fetched again.

A record is also refused, never written and never applied, when it carries no identity,
when it is a type that never leaves its writer (§6.1: `HeaderIndex`, `ModelKey`,
`Tokenizer`, `Template`, `BranchCheckpoint`, `Replica`, `Deleted`), or when it is this
member's own record past every `oseq` it issued — another member holding what this one
lost. The member's clock observes every record it applies or skips, so what it writes
next ranks above it (`persistence/remote/`).

**A type that never leaves its writer is never fetched either.** A catch-up's answerer
lists such a record gone (`persistence/listing.rs`), and a requester counts any it is
listed gone too (`Store::travels`, whatever the answerer runs): the requester marks the
`oseq` applied, and its applied set has no hole there. Listed as a record, it was
fetched, refused and fetched again on every try of every catch-up — a member joining
`.5` on 2026-10-17 refused 164 of them and caught up on nothing.

**Taking the substrate lock.** Apply takes the substrate write lock once per batch and
never nests a read inside a held guard. Windows SRW locks deadlock on that
(`docs/shared_model_host.md`, `projection/resolver.rs` `recover_turn_chunks`).

**Applying it.**

- **Index state.** The record goes through `apply_walker_entry` (`substrate.rs:5070-5226`),
  rewritten to the rules above. It already works on a live substrate.
- **Turns.** The batch adopts each turn it completed (`persistence/remote/adopt.rs`)
  once the turn is **complete**: its `StreamDecl`, its `Tokens`, and every `Chunk` the
  decl's block range calls for (`n_layers × (block_end − block_start)`, the count
  `persistence/resume.rs:524-531` checks), in any order of arrival. A `Commit` is not
  enough on its own. A turn whose declaration claims no K/V — a summary node — holds
  nothing more than its declaration, and a distilled timeline's turns were shed on
  purpose, so either is complete as declared (`persistence/remote/completeness.rs`).
  - The turn goes in the slot its stamp gives it in the timeline's order (§6.6). A
    position below it that holds nothing is a **gap**: an empty placeholder the
    substrate records as one (`substrate/turn_placement.rs`). Reconstruction fills a
    lost declaration's position with a gap too.
  - **A turn that completes later takes its gap in place.** Catch-up delivers a
    timeline's turns in any order, so turn *N + 1* is often adopted before turn *N*;
    filling *N*'s gap moves no index, and needs no reload. Only a turn ranked below a
    turn this member *holds* — one that moves it — does (§6.6): the declaration's
    placement reports the positions it moved turns off, and a held turn among them
    flags the batch for a reload, which then adopts nothing.
  - Its residence is created **cold**: the cold refs point at the local copies, and K/V
    loads on first use through the existing `elevate_to_hot`. Tree metadata, the gallery
    epoch and the summariser queue are touched as `restore_turn` touches them.
- **An incomplete remote turn is pending, never corrupt.** Today a turn with some of its
  chunks is `Corrupt`, and the layer's corrupt-turn policy can tombstone the timeline
  (`persistence/resume.rs:521-531`, `projection/resolver.rs:2828-2909`). For a remote
  turn whose pieces are still arriving that would be fatal, and the tombstone would
  replicate.
  - So a remote turn short of pieces, at runtime or at open, is *pending*: it is left
    out of the projection, its position holds a gap, and its missing `oseq`s are wanted
    by the next catch-up. A remote turn whose records are all here but unreadable is
    the same: this member's bytes are at fault, not the conversation, and a tombstone
    here would refuse the copy that repairs it.
  - Integrity repair tombstones stay what they are, a local judgement about this
    member's own turns. They are written `Housekeeping` — with no identity
    (`write_repair_tombstone`, `write_repair_turn_tombstone`), so no `oseq` is spent on
    them and no member is ever sent one — and a member repairs a corrupt remote turn by
    fetching it again.
- **Sections.** A remote section becomes a cold section residence the same way, under
  its live generation.

**Portability.** K/V bytes from another machine are valid here, though not identical to
what this machine would have written (§1.3). The model lock makes the weights the same
(§3.6), and each block's format is self-describing in its `ChunkPayload`
(`record.rs:586-737`). Per-machine calibration chooses which format a block gets, not
how it is read.

### 6.6 Two writers on one timeline

Affinity and leases make one member the writer of a timeline. Two cases can still
produce a turn at an index another member also wrote:

- a member partitioned past a move, which did not see it and kept writing;
- a member cut off from every cluster that served the conversation locally (§5.8).

**Two writers' turn *N* share a stream id, and the index keeps them apart.**
`turn_stream_id(timeline, turn_index)` (`persistence/content_hash.rs:217-239`) is
unchanged. It is derived at 34 call sites across the scheduler, the projection assembler
and cold load (§1.3), and the hot path stays as it is.

Instead, the substrate's stream index keys a turn stream `(stream_id, origin)`: each
record carries its origin in its header (§6.2), so the index always knows whose it is.
Two writers' `Chunk`, `Tokens` and index-page records never overwrite each other's
entries; without that their K/V would interleave under one key, which would be
corruption, not a conflict.

**A timeline's turns are ordered by stamp.** Nothing is chosen and nothing is lost:

- Every committed turn of a timeline is in it, ordered by its decl's version stamp
  `(gen, epoch, hlc, origin)` (§6.2).
- The turn at index *N* is the *N*-th in that order, counting the gaps below. The order
  is the same whatever order the decls arrived in.
- A lookup by `turn_stream_id(timeline, N)` resolves, through the index, to the
  `(stream_id, origin)` entries of that *N*-th turn. The index keeps that map beside
  itself, so callers stay as they are (`substrate/stream_index.rs`).
- **A turn never sits below the index its writer gave it.** A declaration lost in a crash
  leaves a gap, and the turns above it stay where they were written. So each turn's
  position is `max(previous + 1, its own index)` down the stamp order: for one writer
  exactly the indices it wrote, for two the interleaving the stamps give. Reload fills a
  gap with an empty placeholder (`projection/resolver.rs` `fill_turn_gap`). A writer's
  next turn therefore always takes an index no stream of its own holds.
- A declared turn is reached only through its position. An id whose position holds no
  turn names nothing, even when another writer's turn further up was written under it.
- **Sections are indexed per writer too.** Two members' prefills of one section share its
  content-addressed id and differ in their bytes. Both are kept, and the id resolves to
  the declaration with the greatest stamp.
- Everything keyed by a stream's own records is keyed by the writer as well: the
  dead-byte accounting, the metadata locations, maintenance's carry and its relocation
  watch. Neither writer's record supersedes the other's.

For a single writer this is the order it wrote in: its stamps rise with every turn. Two
writers' turns interleave by stamp:

- **A writer that kept going past a handover** sorts before its successor, because the
  successor's claim epoch is greater (§5.4). That holds within one generation. A
  generation changes only at a rejoin (§6.7), and a turn written after a later rejoin
  sorts after, whatever its epoch.
- **A member cut off from every cluster** sorts after everything the group wrote
  meanwhile, because its generation is fixed above all of it when it rejoins (§6.7).

The turns keep the K/V they were written with. A turn written while cut off was
generated against the conversation as that member last saw it, and is placed after what
the group wrote meanwhile. The projection reads it from there as it reads any turn,
placing it by provenance, not by its original position.

**A turn can arrive ranked below a turn already held.** Most arrivals append, but two
do not:

- **At a rejoin.** The cut-off member's catch-up brings the group's turns in, ranked
  below its own detached ones (§6.7).
- **After a takeover across gateways.** The gateways are independent (§4.3). A member
  connected only to gateway Y does not see a holder connected only to gateway X, so it
  may take over a conversation that holder is still writing (§5.4). When the two sides
  meet again, the former holder's turns rank below its successor's by epoch, and land
  below turns the successor's side already holds.

Either way, the member that receives the turn **reloads**. A turn's index is what the
hot path's `turn_stream_id` derives from (§1.3), so the indices of a live timeline must
not move under it.

1. The member stops taking work, as `drain` does (§12.4). It hands back its claims and
   the gateways route nothing to it. A rejoining member is not serving yet anyway.
2. It runs the reload a compaction already uses (`engine.reload_substrate()`,
   `engine.rs:769-779`; the `ReconstructSubstrate` request on the scheduler thread). No
   new hot-path code is involved.
3. It becomes ready again.

**The reload rebuilds the whole substrate**, not one timeline. It costs what loading the
substrate costs at startup, so a rejoin or a cross-gateway takeover costs that member
one load. It runs on a live substrate, so every turn is put at the position the log
gives it, not appended (`Substrate::reload_turn`): a turn the log puts where it already
is stays there with its resident K/V; one the log does not hold yet stays as it is; one a
rerank displaced takes a fresh residence, and the displaced K/V is released once no
working set holds it. Appending instead duplicated every timeline's turns at indices no
chunk is keyed by (`.5`, 2026-10-20). The second case needs the gateways to disagree about which members are
connected, and it is rare. The status page counts reloads and their cause (§12.2).

VFS events are keyed by origin (§6.2) and follow the same order.

### 6.7 Writing while cut off: generations

A member cut off from every cluster for an app (§5.8) keeps writing. When it rejoins,
**every event the group wrote before its rejoin is placed before every event it wrote
while cut off**, on every member, in any arrival order. "The group" is every member that
was joined when it wrote. Within the cut-off period, the group's events and the member's
own never interleave. Members cut off at the same time are placed in the order they
rejoin. The invariant rests on one number, and on one Paxos decision per rejoin.

**The generation.** Every event carries `gen`, the first field of its version stamp
(§6.2).

- **Generations are minted only by the group.** Generation *k* exists once the Paxos
  decision `gen/<k>` (§5.9) is made, naming the member that rejoined with it. The
  decided generations are 1, 2, 3 and on, with no gaps: a member only ever proposes
  one above the greatest it knows to be decided, and that one was decided only after
  every lower one.
- A member's current generation is the greatest decided generation it knows, from a
  decision record or from the `gen` of an event it applied. It is persisted in its
  `Replica`. A new group starts at 0.
- A joined member stamps its own events with its current generation.

**While cut off, events are *detached*.**

- A member with no admitted socket for the app writes its events with the provisional
  generation ⊤, which sorts above every number.
- In its own substrate they therefore come after everything it holds. That is simply
  true, because nothing external arrives while it is cut off.
- Detached events stay *unsent* (§6.3).

**Rejoining, in this order:**

1. **Catch up first** (§7.2). Every external event that arrives ranks below ⊤, and so
   before the member's detached events, in its own substrate.
2. **Take a generation.** Once the catch-up is complete, the member proposes
   `gen/<k>` with itself as the value, for *k* one above the greatest decided
   generation it knows. If `gen/<k>` is already decided for another member, it learns
   that decision, which raises its current generation, and proposes `gen/<k + 1>`. When
   its own proposal is decided, its detached events take `gen = k`.
   - The assignment is persisted as a `Rebase { origin, oseq range, gen }` entry in the
     `Rebase` record (tag 30) of the member's own log, which holds every entry and is
     carried whole by every rewrite (`persistence/rebase.rs`). It is housekeeping, never
     replicated: the member applies it when reading its own records back, at open and
     when it sends or answers a fetch (`read_event`). The records themselves keep the
     header they were written with.
   - At open the walk meets the `Rebase` record after the records it fixes, so it holds
     this member's own detached records back until it has read the whole log, then
     applies them under their fixed generation: nothing reaches the substrate, or any
     index, unfixed. A detached record no entry covers means the member was cut off
     when it stopped, and it reopens cut off still.
   - A member that wrote nothing while cut off has nothing to place, and takes no
     generation.
   - The decision comes through the `MintGeneration` trait
     (`persistence/identity/generations.rs`); chunk 5 implements it with chunk 3's
     consensus, and an in-process `LocalGenerations` serves a single process and the
     tests.
   - A record carrying ⊤ is never sent, so one that arrives is refused.
   - **Why this is exact.** An event a joined member wrote before `gen/<k>` was decided
     carries a generation that had been decided by then, which is below *k*. So it
     sorts before the detached events on every member, whether or not this member had
     caught up on it. Within one generation, the rest of the stamp, `(epoch, hlc,
     origin)`, orders events the same way on every member.
3. **Send the detached events**, each carrying its fixed generation in its header. Every
   other member then places them after everything the group wrote during the cut-off.
   Because the generation is inside each event, that happens in any arrival order.
4. **Resume.** The member's current generation is now *k*, and its listener forwards up
   again (§5.8). Every member raises its current generation to *k* when it learns the
   decision or applies a detached event, so the group's later events carry *k* or more.

**A rejoin needs a quorum.** A member that rejoins into a partition without a majority
of `max_members` (§5.9) cannot take a generation. In a group of two the voting
gateway's promise is that other vote, so one member still rejoins while the other is
down. Without a quorum it stays as it was while cut off:
- it keeps serving its own listener locally, and writes more detached events;
- it keeps applying what its catch-up brings, all of which ranks below its ⊤ events;
- the gateways route nothing to it, and it sends nothing it wrote while cut off.

It takes its generation as soon as a quorum is reachable.

**Two members cut off at the same time** rejoin one after the other. The first to have
its generation decided gets *k*, the second *k + 1*, so the second's offline writes
sort after the first's on every member. That is the rejoin order. Neither member's
offline writes count as the group's for the other, because neither was joined when it
wrote them.

**What this means for each kind of record:**

- **Turns** written while cut off are appended after the group's turns of the same
  period (§6.6). In zend, what a user typed while cut off continues the conversation
  after what others added, and nothing is lost.
- **Keyed values** (§6.5) written while cut off win over the group's writes to the same
  key in that period, because they are later. A label renamed offline keeps the new
  name.
- **Tombstones** still dominate. A conversation the group deleted while a member was
  cut off stays deleted, and that member's writes to it die (§6.5). That holds however
  long the member was away: within 30 days the tombstone is still held, and past them
  the member drops what the cluster no longer holds before it offers anything (§7.8).
- **npcd** keeps its characters living while cut off, in a local world (§9.4).
  - **World state is not a detached event at all.** In disconnected mode it is never
    persisted, and on rejoin it is scrapped for the authoritative state.
  - **The characters' turns, beliefs and journals are detached events** like any other.
    They are written under NPC leases this member holds and nobody re-lends while it is
    away (§9.3), so they are appended after the group's records with no other writer to
    meet.

### 6.8 Compaction

Each member compacts its own log (`persistence/maintenance.rs`). Once appended, a
replicated record is an ordinary record.

- **It is dropped by §6.5's rules.** Those rules are deterministic, so what one member
  drops as superseded every member drops. A record loses to a greater stamp, a
  tombstone, or a generation's end.
- **Its identity survives.** `(origin, oseq)` and the version stamp survive every
  rewrite. A record compaction drops has its `oseq` marked *gone* in the applied set, so
  the member can say so when a catch-up asks (§7.2).
- **What was seen is remembered.** The `Replica` vector says what has been seen even
  after the record that carried it is gone.
- **Exceptions** are kept that today's compaction would drop:
  - timeline tombstones, kept resident until they expire (below);
  - every committed turn, whatever its place in the order: two writers' turns at one
    index are both the conversation's turns (§6.6), never one superseding the other;
  - a stream **another origin** wrote whose declaration has not arrived yet. A
    declaration is written before its stream's records, so a decl-less stream this
    member wrote — or one whose records carry no identity — lost its decl and is an
    orphan, dropped as before; but another origin's records can arrive ahead of their
    decl, and dropping them would lose them for good, since they are already applied
    (`StreamRuntime::is_orphan`);
  - records held for a timeline not registered yet (§6.5), for the same reason;
  - a snapshot **retraction** that is a stream's current record. It is a value: dropping
    it would let an older snapshot arriving later become the tail again.
- **A label is carried as the records that hold it.** Each field keeps the greatest
  record that set it (§6.5), so a rewrite writes one `Label` per winning identity,
  holding the fields it won, plus the least record that named the conversation's
  `conv_id`, which is its place in creation order.

**When a tombstone expires.** A tombstone record, 4 KiB with its padding, is compacted
away only when **both** of these hold:

1. **A complete compaction has run since it.** That compaction is the one that removed
   every record of the timeline the member held. The member records which tombstones a
   full `compact()` has finished end to end after (*swept*), in its `Deleted` record
   (`persistence/deleted.rs`, tag 31): one resident record of local housekeeping,
   rewritten whole by every compaction, never replicated, with no identity. It is the
   first of the two conditions, which a compaction can only know from the compaction
   before it.
2. **The tombstone is older than 30 days**, by its `hlc` (§6.2) against this member's
   clock (`--tombstone-retention`, `persistence/tombstone_expiry.rs`).

**An expired tombstone is forgotten.** Its record leaves the log and nothing takes its
place, so a deletion is carried for 30 days and no longer. What makes that safe is the
rule for a member away longer than that window allows (§7.8): no member applies a record
it did not write itself after more than 20 days away without first dropping everything
it did not author and syncing afresh, and past 30 days it also drops what it authored
before it left and the cluster no longer holds. Every member that could still carry a
record a deletion named has, by then, either received the tombstone or dropped the
record.

Members need not agree on the retention for any result: a record a tombstone named is
dead on every member that has it, and a member that rejoins after the tombstone expired
follows §7.8.

### 6.9 Why arrival order does not matter

**The property:** once a member has received a set of events, its substrate is the same
whatever order they arrived in, tombstones removing what they name. That holds because:

1. **Every apply is idempotent and order-independent** (§6.5):
   - keyed values are a max over a total order;
   - sets are unions;
   - tombstones dominate;
   - an early child waits for its parent;
   - a timeline's turns are ordered by stamp, which is the same in any order (§6.6);
   - what a member wrote while cut off carries a generation the group decided at its
     rejoin, above every generation in use before, inside each event, so every member
     places it the same way (§6.7).
2. **Compaction removes nothing the order could have kept.** It drops a record only
   when §6.5 makes it dead in any order. A catch-up records an `oseq` as gone only when
   the answerer's coverage says so (§7.2).

   Tombstone expiry removes a tombstone after 30 days (§6.8). An event arriving later
   could only come from a member away longer than that, and such a member drops what
   it did not author and what the cluster no longer holds before it applies or offers
   anything (§7.8).
3. **Ids cannot collide.**
   - Records are keyed by `(origin, oseq)`.
   - Turn streams and VFS events are indexed by origin.
   - Timeline ids are either content-addressed or carry 20 random bits under a
     millisecond.
   - A copied substrate re-mints its node (§3.7).

**Getting every event there is the rest of the design, and is separate.** The sender
re-sends what no `RECEIPT` covered (§6.3). A member that was away fetches what it lacks
(§7.2). A gap, even one at an origin's tail, is seen through `high_water` and fetched from
the origin (§7.1). A restarted member offers what only it holds (§7.5). None of that has
to deliver in order, because of points 1–3.

The unit tests assert point 1 directly: every permutation of a set of events applies to
the same substrate (§14.1).

---

## 7. Catching up

Two failures lose events, and each has its own repair:

- **Events a member sent that never arrived** — they were in flight on a socket that
  dropped. The sender repairs this by sending again everything not covered by a
  `RECEIPT` (§6.3). There is no resync on reconnect for these.
- **Events a member missed while it was away** — relayed by a gateway while the member's
  socket to it was down, or written while it was not running at all. The member repairs
  this itself, by asking for them. That request is the **catch-up**.

A **restart** loses the sender's side of the first repair: which of its own events were
unsent or in flight lived in memory. So a restarted member runs the **full resync**
(§7.5) once, which is a catch-up plus an offer of everything it holds. A socket that
merely reconnects inside a running process never needs it.

### 7.1 When a member catches up

- **On admission** of its first socket for the app, after a start or after being away
  from every cluster.
- **On every reconnect** of a socket. Events relayed through that gateway while the
  socket was down are lost to this member, whatever its other sockets saw.
- **During the full resync** after a process start, as its first step (§7.5).
- **When an origin's `high_water` is above what it holds** (§4.1, `MEMBERS`), and stays
  above for 5 s. This is how a lost event is found even when nothing after it ever
  comes: an origin that wrote one last event that went missing still announces it. The
  catch-up is addressed `to:` that origin, which holds all of its own records, with the
  member's whole `since`; whatever else it lists is lacking too.
- **Every 30 s while any of its sockets is down.** The catch-up runs on the sockets that
  are still up. A gateway it cannot reach may still be relaying events to others that
  this member will not otherwise see.

### 7.2 The catch-up exchange

A catch-up moves **headers first**: the requester learns what exists, then fetches only
what it lacks. Every reply carries `reply_to`, so it reaches the requester alone (§4.3).

1. **Ask.** The member sends `CATCHUP { request, reply_to: self, since }` on one admitted
   socket. `since` holds, for each origin, its `applied_through` from the persisted
   `Replica` (§6.2): a few bytes per origin, whatever the size of the substrate.
2. **Pick.** That gateway hands the `CATCHUP` to **one other ready member of the app,
   picked round robin**. When the requester is the only member there, nothing is owed
   through this gateway, and the gateway answers `DONE` itself.
3. **Headers.** The picked member answers with `HEADERS`, addressed to the requester.
   - **It only speaks for what it covers.** For each origin, it sends the ranges above
     `since[origin]` (or from 1, for an origin `since` does not name) that are
     *complete* in its own applied set, every `oseq` in them held or known gone. A gap in
     its own set ends a range. The answerer never claims a record it has not seen.
   - Each range lists every record it holds, as
     `(oseq, type, stream_id, chunk_index, len, digest)`, about 40 bytes packed, never
     the payload. It also lists the `oseq`s in the range it knows are gone.
   - `len` is the record as it travels, its header line and payload unpadded, with the
     line's `crc` counted at its widest. A header-index digest carries every field of
     the line but the `crc`, and the listing is read from memory (below), so `len` is
     never under what travels and at most nine bytes over it.
   - The answer ends with `DONE`.
   - **An answer is bounded** to 16 `HEADERS`, about 131,000 `oseq`s and 6 MiB
     (`replicate/answer.rs`). The requester fetches what they list, and an admission
     catch-up whose answer moved it on (a record it lacked is now held, or an `oseq`
     learned gone) asks again, from where it then stands, before its socket is ready.
     One that brings nothing has had everything its answerer covers. So a member far
     behind catches up an answer at a time, fetching from the first.
   - **An admission nobody answers does not hold the member.** One that goes quiet is
     asked again; one that goes quiet twice (`tick::ADMISSION_QUIET_LIMIT`) is taken as
     finished, and the member serves on what it holds — anything it lacks is a gap the
     gap watch fetches once an origin announces it. On 2026-10-17 WhiteWorld's npcd,
     ready at the gateway but with its sockets dropping every 45 s, was asked `.5`'s
     admission again and again; `.5`'s npcd never finished loading and `keep` rolled
     the build back after its ten minutes.
     Unbounded, an answer to a member lacking 8 million records was a thousand frames,
     about 360 MB, queued at once. That is far past what a gateway holds for one member
     (128 MiB, after which it drops catch-up frames). A requester waiting on frames that
     never come goes quiet and asks again, and each new answer queues behind old ones it
     ignores. Measured on `.5`: 156 catch-ups answered in 20 minutes, none of them
     recorded by whiteworld, which re-asked every 30 s and fetched nothing.
   - **A gateway never drops a `DONE`.** A dropped `HEADERS` or fetched record is
     repaired: the next answer lists from where the requester stands, and a fetch that
     ends short fetches the record again. A dropped `DONE` would leave its catch-up
     waiting until it went quiet.
   - **Loss is told, not inferred.** A catch-up goes quiet when nothing of it has arrived
     for 30 s *and* no answer is arriving on its socket: no catch-up frame (`HEADERS`,
     `DONE`, a fetched record's fragment) and no message part-way across. Over a slow
     link a reply is still arriving while its frame is not yet whole (§3.2, keep-alive),
     and a fetched record of many 1 MiB fragments is whole only at its last. A
     WebSocket's messages do not interleave, so bytes newer than the last whole message
     are one in progress, while an idle socket's `PING`s arrive whole, no catch-up's,
     and hold nothing. Counting a catch-up's own whole records alone, AlienWorld's gap
     catch-ups were dropped every 32 s while BlackWorld's ~3 KB/s kept arriving, and its
     planned admission two minutes into a 120 MB record at 137 KB/s; counting any byte
     at all, the pings held a catch-up that would never be answered for half an hour.
     The two losses a dropped frame cannot repair the gateway reports as `LOST`:
     - **No member can answer.** The `CATCHUP` the gateway leaves open (step 2) is
       answered `LOST`.
     - **The answerer left.** The gateway keeps each `CATCHUP` and `FETCH` it hands an
       answerer until that request's `DONE` passes; when the answerer's socket ends, every
       request it held open is answered `LOST`, and a `FETCH` naming a member not here is
       answered `LOST` at once. A newer `CATCHUP` from the same requester to the same
       answerer replaces the one before, which that answerer cancels.

     A fetch told `LOST` ends as one ending short does: its records are wanted again. A
     catch-up told `LOST` is timed by its own silence alone, so it is asked again at the
     quiet's pace, not at once and over and over while no member can answer it.

     `LOST` is a frame kind older members refuse, so members take a build that reads it
     before any gateway sends it: each gateway's `web` is restarted onto it only after
     every member runs one.
   - **One answer per requester.** A member far behind lacks everything, so an answer
     lists the whole log, and its fetches run for minutes. A requester
     asks again when its catch-up goes quiet (§7.1), and its gap watch asks again for as
     long as the answerer keeps writing. So a newer `CATCHUP` from the same `reply_to`
     cancels the answer before it, and an answer whose requester no gateway lists any
     more is cancelled too. A cancelled answer stops before its next range and sends no
     `DONE`. Without this, answers stacked by the dozen on `.5`, each re-listing the
     same history, which was then read off disk: 70,000 4 KiB header reads a second, a
     persistence lock nothing else could take, an engine that ran two forwards in
     fourteen minutes, and a requester that never finished catching up.
   - **A range is listed from memory.** The location index keeps, beside each record's
     place, what the headers list of it (§6.2), so a listing is a walk over memory
     under one short hold of the persistence lock. The hold, like a fetch's record
     reads, is taken behind the lock's waiters: answering a peer is background work,
     and the scheduler's and the leases' writes go first. Read off disk, 512 records
     to a hold, the listing still kept the scheduler out 75–93 s at a time on `.5`
     while whiteworld caught up: a lock released and re-taken at once is re-taken
     before a waiter wakes.
4. **Compare.** It decides what to fetch:
   - Every listed record it does not hold, and that none of its other catch-ups is
     already fetching, is wanted. It never compares by `digest`: a record keeps its
     identity through every rewrite (§6.2), so one it lacks is never held under another
     `oseq`, and two records with equal bytes under different stamps are two values
     (§6.5).
   - It records as gone, in its applied set, exactly the `oseq`s the header lists as
     gone: records superseded or tombstoned and compacted away on the answerer by rules
     every member applies alike (§6.8). An `oseq` outside the answerer's
     complete ranges stays wanted. A later catch-up, often addressed to its origin
     (§7.1), finds it.
5. **Fetch.** The requester sends `FETCH { request, reply_to: self, from_node, origin,
   oseqs }` to the member that sent the headers. That member reads each record from its
   log and sends it as an addressed `EVENT`, then `DONE` once the last of them is
   written; the catch-up lane keeps their order (§4.2). Fetches are windowed: at most
   64 MiB of records outstanding at once, so the requester sets the pace. A record a
   `FETCH` ends short of is fetched again; one its answerer cannot deliver after three
   fetches is left wanted for a later catch-up rather than holding this one open.
   Request numbers start from a random base in each process, so answers to a previous
   process's requests, still in flight to the node, never match this one's.
6. **Apply and record.** Fetched records apply like any other (§6.5). As each one
   applies, `applied_through` advances, and the `Replica` record persists it at the next
   group commit. A catch-up cut off by a disconnect or a restart resumes from there.

**Nothing is broadcast.** Headers and fetched records reach the requester only. Other
members see no catch-up traffic, so a member's return costs the group one answerer's
disk reads and one socket's bandwidth.

**The answerer may itself be behind**, because round robin does not pick the
best-informed member. Its complete ranges then stop short for some origin. The requester
leaves the rest wanted, and its applied set shows that origin still behind its announced
`high_water`. That starts a catch-up addressed to the origin itself (§7.1). A member that
sent the missing records also still holds them as unsent or in flight, and delivers
them as §6.3 says. Every route to a missing record ends at the one member certain to
hold it, its origin, for as long as that origin exists.

**A member is ready** once its admission catch-up's last `FETCH` has ended in `DONE`
with nothing missing. It tells every cluster in `CAPACITY.state`. A gateway routes only
to ready members, and the first ready member of an app is what switches that app to
clustered mode there (§5.7). A member that is alone is ready on admission. A member in
fault mode is not ready (§7.7), and the last one going into fault mode reverts the app to
dumb mode on that gateway, as its leaving would.

### 7.3 Size

The headers for a million records are about 40 MB, which is cheap to send and to compare.
The records themselves are another matter. A member joining with an empty substrate
fetches the whole log, which can be many gigabytes: that is the cost of D5. Headers
first means it fetches each record once, and nothing it already holds.
The headers' `len` gives the loading page a total to show progress against, at most
nine bytes a record over what travels (§7.2).

### 7.4 Backpressure

A catch-up uses only spare capacity:

- The requester's fetch window (64 MiB outstanding) keeps the gateway's 128 MiB catch-up
  cap per member (§4.4) from being reached; a record larger than the window is fetched
  alone, and crosses the gateway on an empty lane. A `FETCH` whose `DONE` arrives
  short of records is fetched again.
- The answerer sends catch-up frames only when its ordered lane is empty and its
  socket's unsent bytes are under 1 MiB (§4.2). Live traffic that keeps the socket
  busy therefore holds catch-up back entirely, and a live event queued behind catch-up
  waits for at most 1 MiB of it to drain. That bound is in bytes, and small records
  make it many frames for the gateway to relay first, so it is kept to what keeps a
  link busy and no more: under catch-up of 100 k small records, H4 measures the live
  p99 against it (§14.2).
- The writer waits for the socket to drain under that mark on the pipe's own count —
  woken by the fall itself (`pipe::drain`), never on a timer. Polled every 1 ms, the
  wait took a timer tick — ~15.6 ms on Windows — so about one 1 MiB mark went out per
  tick on each hop, ~64 MB/s at best, and a member joining `.5` caught up 35 GB at
  30–43 MB/s (2026-10-18).

A joining member on a slow link therefore catches up behind the live traffic, not in
front of it.

### 7.5 Full resync, once per process start

A member that has just started does not know which of its own events the group
received before it stopped. It may also hold records written by members that have
since gone, records no connected member has. It therefore offers what it holds, in both
directions, and does so **only on the first admission after the process starts**:

1. **Catch up.** It runs the catch-up of §7.2 first, from the position its `Replica`
   record persisted before it stopped. It pulls what the group wrote while it was down.
2. **Offer.** It sends `HAVE { request, reply_to: self, applied set }`, its applied set
   packed per origin, on one admitted socket. The gateway relays it to every other
   member of the app.

   ```
   u16 n_origins
   repeat: u64 origin | varint top | varint n_gaps | n_gaps × (varint start_delta, varint len)
   ```

   `top` is the origin's highest covered `oseq` and the gaps are the uncovered ranges
   below it, ascending; `start_delta` is a gap's start less the end of the gap before
   it (zero before the first), and both it and `len` are at least one. Varints are
   unsigned LEB128, minimal. Origins ascend and none repeats, so one set has one
   spelling and any other is refused. A member that is caught up sends about 10 bytes
   per origin. `CATCHUP.since` is packed the same way. The type both ends hold — a
   `through` and the covered ranges above it — is the `applied-set` crate, shared by
   the substrate and the `cluster` crate.
3. **Each other member catches up from it.** A member that receives the `HAVE` compares
   it with its own applied set. For any origin where the restarted member holds more,
   it runs §7.2's exchange against the restarted member directly: a `CATCHUP` with
   `reply_to` itself, then headers, compare, fetch. Each of those members gets back
   only what it lacked, addressed to it.
4. **Its own events are not sent again.** Every own event the previous process wrote
   counts as received (`Outbox::restart`): step 3 already delivers each one a member
   lacks, pulled by that member on the catch-up lane, and a member that joins later
   catches up at its admission (§7.2). Pushing them as well put a member alone in its
   group — no `ACK` to say any event was held — through its whole history on the
   ordered lane at every start, ahead of every request: 2.5M records on `.5`, and every
   request the gateway forwarded to it waited the ~8 minutes it took. Only events
   written since the start go out as §6.3 says.

From then on, until the process stops, the member relies on §6.3's retransmit and
§7.2's catch-up alone.

### 7.6 Loading phases

zend's `LoadStep` (`zend/src/loading.rs:27-125`) and npcd's
(`npcd/src/engine/loading.rs:31-101`) each gain two steps after `substrate`:

| id | label | unit | detail shows |
|---|---|---|---|
| `connecting` | "Connecting to the cluster" | clusters | for each `--cluster`: the URL, the attempt, the next retry in seconds, and the last close's reason (a 4410 says *rebuild*, naming both builds) |
| `catchup` | "Catching up" | records | records and bytes fetched against the total the headers listed, records skipped as already held, and which member is answering |

A daemon started with `--cluster` that reaches a cluster is not ready until it is
admitted on at least one and caught up, because serving from a substrate that is behind
is the thing to avoid. It does not wait for every cluster: the others join in the
background. One that reaches none goes on cut off and serves locally, as §5.8 says.

A later disconnect does not send it back to loading. The gateways stop routing to it on
the sockets it lost. Its `/v1/status` gains a `cluster` block with, per cluster: state,
session, last close, its own events in flight and unsent, and any catch-up in progress.

A terminal close (4410, 4426) leaves the daemon on its loading page with the reason
shown. It is not a crash, because the operator needs to read why. A member admitted to
upgrade (§3.5) shows an `upgrading` step instead: the build it is fetching, the bytes,
and its source.

The loading screens are `buildLoadingScreen` (`zend/web/index.html:3585-3614`) and the
`#boot` list (`web/content/npcd/index.html:96-113`). They render the two new steps from
`completed_labels` and `current` as they render the existing ones.

### 7.7 Fault mode: the disk is full

A member whose disk fills **stays connected and switches to fault mode**. It does not
drop its sockets, crash or go quiet, because the operator has to see what happened, and
the group has to know the member can no longer write.

**When it enters.** Either of these puts the app's member into fault mode:

- **An append or a commit fails for lack of space.** That is `ERROR_DISK_FULL` or
  `ERROR_HANDLE_DISK_FULL` on Windows, `ENOSPC` elsewhere. The redo log's staging,
  segment rotation and group fsync (`persistence/mod.rs:634-641,1917-1952`) are where it
  surfaces; the log latches it (`persistence/space.rs`) until a test write succeeds.
- **The substrate's volume falls below a floor** of 2 GiB free, or 1% of the volume if
  that is larger. The member measures the volume once a second (`replicate/fault.rs`),
  so it normally enters fault mode *before* a write fails, with room left to write
  its own fault notice.

**Fault mode has one other cause: a candidate build that failed certification** (§3.8).
It behaves the same way, with no requests routed to it, roles handed back, and still
voting, with one difference: its disk is fine, so it **keeps applying remote events**
and receiving live ones. The table's rows that exist because of a full disk do not apply
to it. It leaves only by an admin's `certify-anyway` or `reject-build`, or by a forced
rollout of its build (§3.11), never by itself.

**A device that cannot take a turn reports `fault` too.** While the GPU context is
poisoned, or every device operation has failed out of memory for 2 s
(`candle::gpu_poison::device_failing`), `CAPACITY` says `fault` and `/v1/status` says
`"state": "fault"`, whatever the tenants report, so the gateways route the member no new
work. It gives up nothing else: its disk is fine, and the self-heal watchdog relaunches the
process once the streak has run 60 s (`host/src/self_heal.rs`), which hands its claims back
as any restart does. Until then a conversation it already holds still routes to it. Before
this, a poisoned member reported ready for that minute and answered every chat with an
empty stream.

**What it gives up**, because each of these needs a write it can no longer make:

| Role | What it does |
|---|---|
| Sticky claims (§5.4) | releases every claim with `UNCLAIM`, which needs no disk; the conversations move to other members, which catch up as any takeover does |
| World authority (§9.2) | sends `RELEASE` for every world lease, after writing what it still can; the proposer rendezvous names takes each world |
| NPC leases (§9.3) | stops its characters at their next turn boundary; their world's authority lends the leases elsewhere, as it would for any joined member that hands them back |
| Paxos acceptor (§5.9) | **keeps voting.** Its votes go to the preallocated vote journal, which never needs new space, so its promises stay durable with the volume full. A faulted member still counts toward the quorum, and a small group does not lose its majority because one machine's disk filled. |
| Paxos proposer and lease holder | proposes nothing and holds no lease. Holding a world means writing its state, which it cannot do. |
| Applying remote events (§6.5) | stops for a full disk, since it cannot append them, and it sends no further `ACK`s. A member faulted by certification keeps applying. |
| Serving API requests | the gateways route nothing to it, apart from `/v1/status`, `x-tokera-node` admin requests and aggregates, which it answers with its fault |

**What it keeps doing:**

- **its sockets, keep-alives, `CAPACITY` and `COORD`**, so it stays visible;
- **sending its own unsent and in-flight events** (§6.3). Reading them back from its log
  needs no free space, so what it wrote before the disk filled still reaches the
  group;
- **answering catch-up `FETCH`es** for records it holds, which are reads;
- **trying to free space.** Once a second it runs a maintenance `Drop`, which unlinks a
  segment holding only dead records and needs no scratch space
  (`persistence/maintenance.rs`). It does not start a `Compact` or `Combine`, which would
  need room it does not have — nor expire tombstones, since a tombstone shrinks only
  inside a full compaction (§6.8).

**The gateway stops relaying live events to it** and does not queue them. Queuing them
would only fill the gateway's ordered lane and make the member lagging (§4.4). What the
member misses is what its catch-up fetches later.

**When it leaves.** Free space has to come back above twice the floor, and a test append
plus commit must succeed. The member then:

1. catches up (§7.2) on everything relayed while it was faulted;
2. becomes ready, and may propose and hold leases again; it never stopped voting;
3. becomes routable again, and takes leases and claims as any member does.

The hysteresis stops it flapping in and out of fault mode at the edge of the floor.

**A zend turn caught by the fault** fails its seal and returns the error to the client.
The conversation's claim then moves, so the client's retry is served by another member.

**Visible in the gateway.** Fault mode is a member state of its own on the status page
(§12.2):

- free space against the floor, and since when;
- what the member gave up: claims, worlds, characters. It keeps its vote, and the page
  shows it still voting;
- how much is still unsent from it;
- whether it is freeing space, and how much it has freed.

Entering and leaving are entries in the coordination log (§12.3). `/v1/status` on the
member itself carries the same fault in its `cluster` block, so its own health check
shows it too.

### 7.8 A member away too long

An expired tombstone is forgotten everywhere (§6.8). So a member away longer than the
retention cannot learn of a deletion it missed, and its copy of the deleted records would
stand for ever. No other member takes them back: each has their `oseq`s in its applied
set, gone or held. But the member itself would go on serving them. So a member that
comes back after too long drops what it can no longer trust, before it applies or offers
anything, and syncs afresh.

**When it was last joined.** `substrate/JOINED` holds one wall-clock reading in
milliseconds (`persistence/joined.rs`). The replicator rewrites it once a minute while
the member is joined and caught up on at least one socket (`replicate/tick.rs`,
`Store::mark_joined`). A store with no file has never been judged: its first open writes
one and drops nothing.

**At open.** A writable open reads the file before it walks the log
(`SubstratePersistence::open_adopting`):

| Away | Dropped |
|---|---|
| up to 20 days | nothing |
| over 20 days | every record another origin wrote |
| over 30 days | those, and its own records whose `hlc` is before the reading |

A dropped record is walked past as if absent. It never reaches the substrate, an index
or the open's sink, but it still advances the clock and the `oseq` counter, so the node
never mints an identity twice. The replica then forgets every other origin, gone
`oseq`s included. The open compacts the store, so the dropped records leave the disk, and
restarts the clock. What the member wrote after it was last joined is kept: no other
member has it.

**Taking back what the cluster still holds.**

- **Other origins.** Nothing of theirs is applied any more, so the admission catch-up
  (§7.5) lists and fetches everything the answerer holds. What a deletion took is not
  listed, because it is gone everywhere.
- **Its own records.** Each one dropped is a gap in its own written set, below that
  set's top. The applied set it offers and asks from carries the gaps, so an answerer
  that holds the records lists them, and they are fetched like any other. Applying one of
  its own records is allowed only into such a gap (`remote/apply.rs`). An `oseq` it never
  minted is still refused as `OwnOrigin`, and the refetched record is not tapped out
  again. An own `oseq` listed as gone closes its gap.
- **Across restarts.** The `Replica` record (§6.2) carries the node's own written set
  beside the other origins' applied sets, once the store holds any `Replica` record. The
  walk takes the own entry from the **latest** record alone, not as a union, so a closed
  gap stays closed. Below the entry's top its gaps stand; above it, every own record
  written since is held.

`own_through` is the top of the own written set: everything up to it is written, and below
it a gap is a record other members hold for it.

---

## 8. GUI streams: interest is an open stream

A live GUI stream is a virtual connection, so interest is the stream's existence:

- The member produces events for a stream only while the stream is open.
- A browser that goes away resets its stream, and the member stops producing for it.
- Stream frames go only to the gateway leg that opened them, never to other members,
  and they are not substrate events.

This covers zend's chat SSE and its `/live` reattach, which are sticky to the
conversation's member. A turn keeps running when its stream closes, as it does today
(`zend/src/session.rs:207-228`), and a reattach finds it on the same member. It also
covers upload progress and npcd's interaction SSE (`Owner`) and NDJSON generation.

`/ws/logs` is the one stream fanned in: the gateway terminates the browser's socket,
opens `/ws/logs` on every member, and interleaves their lines, each tagged with its node.

---

## 9. NPC worlds: state, leases and partitions

### 9.1 World state is persisted; transient events are broadcast

**Before this design**, a hosted world's live state was RAM on one machine (`Sim`,
`npcd/src/world/mod.rs`, `npcd/src/sim/mod.rs`): bodies and positions, the lift,
benches, missions and operations. A 500 ms metronome checkpointed three whole-world
CustomObjects — `missions/<world>`, `benches/<world>` and `command-table` — and a
character's place was kept on its `Npc` record (`NpcPayload.at`). Everything else was
lost on restart. A world kept that way is taken in once, from those objects and those
places, and its old objects are tombstoned once it has written its own records
(`npcd/src/world_state/legacy.rs`, §15.1).

**In this design the world is split into two kinds of thing**, and only one of them is
persisted:

| Kind | Examples | Persisted | Travels as |
|---|---|---|---|
| **State**: what the world *is* | each body's place (area and node) and the station it holds; the lift car's floor, doors and stops; everything the world's `Sim` holds — each bench's working set, each mission and operation, each body's pack, each device, each phone thread, each posting, the command table; who holds each NPC lease (§9.3) | **yes**, one keyed record per entity: `world/<wid>/<entity>` | `EVENT`, replicated, caught up and compacted like any record (§6) |
| **Transient**: what the world is *doing* | the steps of a walk between nodes; the lift moving between floors; a line spoken in a room; a passer-by perceived; the metronome's beat; what the world tells a character that runs elsewhere | **never** | `PULSE`, broadcast to the app's other members with no `oseq`, no retransmit and no catch-up |

**The records** (`npcd/src/world_state/`), named under `world/<wid>/`:

- `body/<body>` for each body standing still: its place and the station it holds. A body
  walking has none; its record is written where it stops.
- `lift`: the car's floor, doors and stops.
- `sim/<map>/<key>` for each entry of the `Sim`'s entity maps (`SIM_MAPS`: packs, the
  benches' working sets, devices, the field, the ledger's per-body cursors, missions,
  operations, the record's items, phone threads, postings), and `sim/meta` for the rest
  of it.
- `npclease/<nid>` for each character's lease (§9.3).

What a machine keeps of its own is not in them — where its benches' documents live, and
the workflows its mind's operations run on — and a world taking its records in keeps
its own (`Sim::take_in`). A world with no records is as its map seeds it, its command
table open.

**Two pulses** (`npcd/src/authority/{pulse,told}.rs`):

- `events`: the world's events since the last pulse. A member following the world takes
  them into its own log, renumbered onto its clock, and its characters perceive them
  as they would its own (`World::witness`).
- `told`: what the authority hands a character that runs elsewhere — a message off a
  handset, a mission withdrawn, a standing task the table sets. The member running the
  character puts each into its inbox. Like any pulse it is lost if missed; the threads
  and missions it tells of are state, and the next turn reads them.

**The rules that keep the two apart:**

- **Only the world's authority writes state** (§9.2). Each beat it compares the world
  with what it last wrote, and writes the entities that changed, in one commit, at most
  once per entity per beat. The whole world is never re-checkpointed.
- **A transient never changes persisted state.** It is what a state change looks like
  while it happens. A member that misses a `PULSE` misses only that animation, and the
  next state record settles the world.
- **Staleness is a tombstone.** An entity that ceases gets a tombstone on its key: a body
  that leaves the world, a mission closed, a bench put away. Compaction can then drop
  its records, and the tombstone itself expires by §6.8's rule. A world that is removed
  is stopped, and every record under its `world/<wid>/` prefix is tombstoned in one
  commit, stamped above every epoch the world was written under, so no write from
  before it brings an entity back (`Runtime::remove_world`).
- **State records are ordinary keyed values** (§6.5). The greatest stamp wins in any
  arrival order. Each carries the world lease's epoch (§9.2), so a former authority can
  never overwrite its successor.

### 9.2 The world lease

Each world has one **authority** at a time: the member holding its world lease. The
authority runs the world engine. That means the metronome and the world's share of the
mission generator (`mission_gen/run.rs:565`). It applies every act that changes the
world, writes the state records, and broadcasts the transients.

**The world lease is a Paxos lease** (§5.9), with key `world/<wid>`:

- the npcd members are the acceptors and proposers;
- the quorum is a majority of npcd's `max_members`;
- the gateways carry the frames.

**Term.** The `ttl` is 30 s, renewed by the holder every 10 s. The holder treats its
lease as over 2 s before its own count ends. Its count runs from `PREPARE`, so it stops
before any acceptor lets another proposer in.

**What follows from a quorum of members:**

- **A partition's minority holds no world.** A member cut off from the rest cannot
  renew, so its world leases run out and its worlds go to disconnected mode (§9.4). It
  needs nothing but its own clock to know that.
- **The majority side keeps electing.** The proposer rendezvous names (§9.5) takes each
  world once the old term has run out at the acceptors.
- **Losing a gateway decides nothing.** Members joined to the other clusters still reach
  each other, and acceptor state is in their substrates, not in any gateway.

**The lease epoch** `(ms, node)` (§5.4) is set when a member first gains a world: its
clock, or one above every epoch the world's records carry if its clock is behind them.
It is stamped on every state record the authority writes. Paxos decides *who* holds a
world, and the epoch orders what successive holders wrote (§6.5). A daemon with no
cluster takes each world it hosts the same way.

**Gaining a world** (`npcd/src/authority/driver.rs`): the member reloads it from its
records — scrapping whatever its own copy became — sets its epoch, claims the world's
id so the world's routes come to it (§9.6), and from its next beat writes and pulses
the world and lends its cast. **Losing it** — its term run out unrenewed — it stops
writing at once, and runs the world locally if characters of it run there (§9.4).

### 9.3 NPC leases: lent by the world's authority, held where they are lent

**Each character runs on one machine, the holder of its NPC lease.** The holder runs the
character's autonomy loop (`npcd/src/engine/tick.rs`, `character_loop` in
`runtime.rs`) and writes its turns, beliefs and journal. One writer per character's
memory means its records are never contested.

**Every member embodies every character.** Each member hosts every world, and its copy
of each has a body for every character in it; only the minds of the characters it holds
think there (`npcd/src/authority/cast.rs`). The cast replicates as `Npc` records, and
every member reads it again every 2 s for characters made, changed or deleted on another
member, so each world's authority knows every character it lends.

**The world's authority lends NPC leases**, as state records:
`world/<wid>/npclease/<nid> = { holder, expires_ms, next, after }` —
`next` the member rendezvous names now, while the holder still has the character, and
`after` the writes of the member that handed it back, `{ origin, through }`, which its
new holder applies before it wakes the character there.

- It lends them across the npcd members by weighted rendezvous over `npc_id` (§9.5),
  from the start, not only when a cast outgrows one machine. Losing any one member,
  the authority included, removes only that member's share of the cast.
- Characters lent away from the authority send their world-changing acts to it as
  requests (below). That is the price of the spread, and it is one request per act.
- It renews a lease while the holder is joined, once less than half its 60 s term is
  left. When membership changes and rendezvous names a new holder for a character, the
  lease names it as `next`, and moves only at that character's next turn boundary,
  after its current holder hands it back (`hand_back`, below). A character waiting for
  an engine is between turns.

**A lease is held until it expires *and* its holder reconnects.**

- If the holder drops out, its NPC leases run out, but they are **not lent again** until
  that holder rejoins. A character never runs in two places, so its memory never gains
  two pasts.
- The characters of a partitioned machine are therefore absent from the cluster's world
  while it is away. Their bodies' state stays as last persisted.
- When the holder rejoins, the authority renews its leases or lends them elsewhere.
- For a holder that will never return, an admin can free one with
  `POST /v1/pulse/npcleases/:nid/release`, which writes the release with a greater
  stamp.

**A character away from its world's authority** keeps a *replica* of the world: the
state records and the `PULSE`s, enough to perceive it. **Everything that writes the
world goes to the authority** as a request on the cluster, `POST /v1/world/:wid/op`
(`Owner(world)`), answered only over a cluster socket and only from a member — the
gateway stamps `x-tokera-from-member` on a request a member sends up, and strips it from
everyone else's — and only by the world's authority (`npcd/src/authority/{ops,perform,routes}.rs`):

| Op | What it writes |
|---|---|
| `begin` | a turn's opening: what the character's arrival signed off, the mission compass, what the table it stands at hands it |
| `act` | one act of a turn, as the call line it was made as, with the decision it was made in — where it stood and the company it was offered — so a line called after somebody who walked out is called after them there too |
| `message`, `enter_beside`, `leave`, `place` | a person messaging a character; standing beside it, leaving, and following it as it moves |
| `lodge`, `step`, `cancel` | a mission lodged or carried at once, a step ticked off or added, a mission called off |
| `channel`, `posting` | a person speaking on the world's open channel, or leaving words on something in it |
| `hand_back`, `release` | a character handed back at its turn boundary, its holder's writes reaching `through`; an admin freeing a lease |

The authority makes each with the same code a member running the world makes it with in
place, and answers with what came of it — an act's line for the feed and for the
character, and whether it moved the character's mission. What changes only the
character's own mind — a reflection, a mood, a journal entry — stays where the character
is. A world that does not answer is told to the character as a refusal, and it tries
again. Leases change only at the authority: a member running the world cut off answers
neither a hand-back nor a release.

### 9.4 Disconnected mode

A member that loses every socket for the app, or whose world lease lapses, switches each
world it has characters in to **disconnected mode** (a world it has none in it only
follows, and stops writing):

- **It runs a local world engine** for that world, seeded from the last state records
  and pulses it holds.
- **Its characters carry on**: the ones whose NPC lease it holds. They walk, talk and
  work with each other in that local world.
- **Nothing about the world is persisted.** State changes stay in RAM, and no state
  record is written. Pulses go nowhere.
- **The characters' own records are persisted**, as detached events (§6.7): their turns,
  beliefs and journals. They are the characters' memories, written under leases this
  member holds, so nobody else writes them.
- **Work on the world is world state, and goes with it.** That means missions,
  operations, and the edits a step makes to its operation's store (§10.3). In
  disconnected mode those edits are kept as an overlay in RAM: no `VfsEvent` is written.
  Drafts written in disconnected mode are scrapped with the local world on reconnect,
  and the operation resumes from its last persisted step.
- **Publishing still works.** git is outside the cluster, so a partition from the
  gateways says nothing about whether `origin` is reachable. An operation finished in
  the local world takes its `publish` step at the merge table as it would anywhere, and
  its verified work reaches the record (§10.3). The push needs an agent the member can
  reach without a gateway, which is its own in-process one (`motherd --agent`, §11.5).
  With none, or with `origin` down, the step waits and is tried again 30 minutes later.
- **What was published stays published.** On reconnect the local world is scrapped as
  below, but git is not. The authoritative world's copy of the operation carries on. If
  it publishes too, git merges the two: identical text is a no-op, and different text is
  a conflict its publisher solves at the bench (§10.3).

**On reconnect:**

1. The member rejoins: catch-up first, then its detached events (§6.7).
2. It **scraps the local world state**.
3. It reloads the world from the authoritative state records.
4. It resumes as a replica, or as the authority if it wins the world's lease (§9.2). Its characters'
   bodies return to where the world remembers them.

It is back for a world once it is joined and caught up, and another member claims the
world (§9.6) — or once it takes the world itself, rendezvous naming it. Until then it
keeps the world it lived in.

**A network partition partitions the characters.** Each side carries on: the cluster in
the authoritative world, and each cut-off machine in a local world holding only its own
characters. Nothing gets corrupted:

- **World state** is persisted only by the world's Paxos-decided authority, and Paxos
  never lets two members hold one world.
- **A character's memory** is written only by its NPC-lease holder, and that lease is
  never lent again while its holder is away.

What a cut-off character lived through stays in its memory as something it experienced,
even though the world forgets the place it happened.

### 9.5 Rendezvous: who asks, and where characters are lent

Rendezvous names who *proposes* for a world's lease (§9.2), and where an authority
lends a character (§9.3). It never takes a lease from its holder. For key `k` (a world id
or an `npc_id`), and the npcd members a member sees across all its clusters (the union of
their `MEMBERS` lists), each with weight `w`:

```
owner(k) = argmax over members m of   -w_m / ln(u(k, m))
           u(k, m) = (SHA-256(k ‖ m.node) as u64 + 1) / 2^64     ∈ (0, 1]
```

**Properties.**

- Every npcd computes the owner itself, with no gateway state. Members joined to the
  same clusters see the same members and agree.
- A member joining or leaving moves only the keys whose winner changed.
- Each member holds a share of the keys proportional to its power.
- `w` is the member's `weight` in `MEMBERS`: its `power` as it stood at admission,
  fixed for the session. Neither the second-by-second idle flag nor the drifting
  measured power (§5.3) feeds it, so neither load churn nor a better measurement moves
  worlds while the membership stands.

### 9.6 Moving a world and routing to it

**Handing a world over**, when rendezvous over the current membership names a new
proposer:

1. The authority finishes the beat, writes every pending state record, and sends
   `RELEASE { key, ballot, through }` to every member (§5.9). `through` is its `oseq`
   after those writes, and each member, as acceptor, discards the accepted lease. It
   lets its claim on the world go, and follows the world from then on.
2. The new proposer — which has not asked while the holder was in the group (§5.9) —
   hears the release (`LeaseEvent::Released`) and waits until it has applied the old
   authority's records through `through`. It then runs a round (§9.2), gains a
   majority's acceptance, reloads the world from the state records, and starts lending
   NPC leases.

An authority that is partitioned away cannot renew, so its lease simply runs out
(§9.2). It switches to disconnected mode itself (§9.4). The proposer rendezvous names
among the remaining members then takes the world in the first round after the
acceptors' timers expire.

**Routing.** Lease holders announce their leases as `CLAIM`s on every cluster, and
gateways route by them as they route `Sticky` (§5.4):

- a world's live routes go to its authority;
- a character's routes and its interactions go to its NPC-lease holder.

A character whose holder is partitioned away is answered `503`, naming the holder.

---

## 10. File state as VFS, published through git

### 10.1 zend repositories

zend already holds a conversation's edits as `VfsEvent` and `VfsTombstone` records
(§1.4), so they replicate with no further change. A conversation that moves carries its
uncommitted work to the new member.

Reads fall through to a lower layer: the branch read through git at the pinned base
(`zend-vfs/src/vfs/mod.rs:8-34`). The member serving the conversation therefore needs
that base commit. A commit is always pushed to `origin` when it lands
(`zend-vfs/src/work/commit.rs:173-218`), and the origin watch fetches every branch on a
change (`zend/src/origin_watch/mod.rs:38-50`). A lower-layer read of a commit the member
lacks fetches it first.

A member whose manifest lacks one of the conversation's repositories cannot serve it.
`HELLO.repos` lists the manifest's repositories and their origin URLs, and `Sticky` and
`Best` consider only members that hold every repository the conversation names.

### 10.2 zend uploads

Uploads stop being files in `substrate/conv-files/` (`zend/src/conv_file_store.rs`).
Each one becomes a `ReplaceBinary` delta (`zend-vfs/src/file_delta.rs:32-46`) in the
conversation's `uploads` store, the folder repository that already exists
(`docs/zend_workspace_execution.md` §3.2), and so it replicates as a `VfsEvent`. The
store's 10 MiB cap (`zend-vfs/src/vfs/mod.rs:108-113`) applies, and an upload past it is
refused with the cap named.

### 10.3 npcd: an operation-scoped store over the mind repository

The mind folder is a git repository (`docs/the_effector.md`). npcd gains the `zend-vfs`
store over it.

**With `--cluster`, the mind is a repository the agents hold** (`npcd/src/mind_repo.rs`),
named the way zend names a repository in its workspace: `--mind-repo <name>`
(`mind_repo:` in `motherd.yaml`), its record the branch `--mind-branch` (default
`main`). So the machine that runs inference needs neither the repository nor a way to
its origin; an agent anywhere in the cluster holds the checkout and the credentials
(§11). `--mind <dir>` remains for a daemon without `--cluster`, and the two are refused
together, and `--mind` is refused beside `--cluster`.

- **A mirror.** The member keeps its own working tree of the repository under `--data`,
  at `mind/<name>`, whose `origin` is `agent:<name>` — a name, never a URL, so nothing
  fetches or pushes it but through an agent. Every read the npcd code makes of the mind
  folder — the schema, the libraries, the registries, the layers, the corpus, the
  watcher — reads the mirror, and the stores' lower layer is its branch.
- **Through agents.** Its fetches go through an agent holding `<name>`
  (`agent::fetcher::AgentFetch`: the fetch job of §11.4), its pushes through one with
  push rights (`AgentPush`), both set when the member joins. Following the record — every
  60 s, and after a publish — fetches through the agent and checks the record out in the
  mirror, so the watcher and the registries take the change in. While no agent answers,
  the follow says so once and tries again; a publish waits as §10.3 already describes.
- **First start.** A mirror is ready once it holds the record's `projection.yaml`. A
  member whose mirror is not ready starts with no mind, saying so. Once it has joined,
  it makes the mirror — `git init` on the record's branch, `origin` the agents' — brings
  it up to the record through an agent, retrying every 30 s while no agent holding it
  answers or origin has no such branch, and then restarts the process onto it
  (`TenantContext::stopping`): the schema, the libraries and the grammar built from them
  are read once, at start. A record holding no `projection.yaml` is not a mind, and
  fails the load. A start with the mirror in place uses it at once and follows the
  record through an agent once joined.

**The work belongs to the operation, not to the mission.** An operation is one piece of
work carried through a workflow (`docs/npcd_workflows.md`): a story written, read,
reviewed, fixed and checked. The table offers its steps one at a time, and each step on
offer is a mission of its own (`Operation::offered`, `npcd/src/sim/operations.rs:45-47`;
`offer_step`, `npcd/src/engine/mission_gen/run.rs:577-604`). A store per mission would
lose the draft between the writer's mission and the reviewer's. So the store is the
operation's, and every mission that carries one of its steps works in it.

**The store.**

- Each **operation** owns one store. Its `TimelineId` is
  `hash("operation" ‖ world id ‖ operation id)`, so every member names it the same way.
  Operation ids are allocated by the world's authority alone (`Operations::open`,
  `operations.rs:159-199`), so they are unique within a world.
- The store's lower layer is the mind repository at the commit pinned when the operation
  **opened**. Its upper layer holds every edit any step has made, in order.
- The store lives as long as the operation. Its `VfsEvent`s replicate like any record,
  so a step taken on another machine, by a character leased there, sees the edits of the
  steps before it.

**What a character reads.**

- A character carrying a mission reads and writes the mind **through the store of the
  operation that mission is a step of** (`Mission::operation`,
  `npcd/src/engine/mission.rs:241-247`). The bench's file acts
  (`npcd/src/engine/bench.rs:104-387`) and every mind read under `Within` are routed
  through it.
- The table's own steps read the same store. The `reading` decode, and the `committed`
  and `changed` checks, judge the operation's document as the store holds it, and
  `Operation::found` is taken from the store.
- A character with no mission reads the published mind at the current `origin` tip.
- **Changing mission changes what a character sees** only when the new mission is a step
  of a different operation. A writer's draft is in front of the reviewer, the fixer and
  the checker of the same operation, and of no one else.

**What changes in the bench.** `Bench::commit` stops writing to disk
(`npcd/src/sim/bench.rs:654-704`). It records the change into the operation's store, so
it replicates as a `VfsEvent`. Its collision check (`bench.rs:672-688`) compares against
the store, where the operation's earlier steps are what moved the document.

**Nothing unverified reaches the record.** Today every step commits straight into the
mind folder, so three mechanisms keep half-finished work out of the way. All three go:

- **`held_drafts`** (`npcd/src/engine/runtime.rs:663-685`) keeps unsettled operations'
  documents out of memory ingest. The mind folder now holds only published work, so the
  ingest has nothing to hold back.
- **`on-failed: set-aside`** moves a failed draft out of the record, and **`restore`**
  writes back what a corrected document said before (`Sim::set_aside_failed`,
  `npcd/src/sim/mod.rs:267-281`). A failed operation never reached the record, so both
  become *discard the store*. The workflow format's `on-failed` therefore takes
  `discard` or `keep`:
  - `discard`, the default, drops the store when the run fails;
  - `keep` leaves it, unpublished, for an operator to reopen from the operations tab.

  `reopen` resumes on a kept store. On a discarded one, it starts a fresh store at the
  current tip.
- **`Operation::before`** exists only to restore, and is removed.

**Publishing is the workflow's last step, after verification.**

The archive floor is where a verified piece enters the record. Publishing is an
operation step like any other, so it runs only once the workflow's verification has
passed:

- **The step.** Each workflow that changes the record ends in a `publish` step taken at
  the merge table:

  ```yaml
      canon:
        by: another
        next:
          pass: publish          # was: done
          reject: fix
      publish:
        by: maker
        desk: merge-table
        edits: optional          # only to resolve a conflict
        tools: [merge_status, merge_sync, merge_publish]
        checks: [published]      # the step stands once the store has landed on main
        next: done
  ```

  In today's pipeline (`docs/npcd_workflows_current.yaml`), `story` and `life-event`
  publish after `canon: pass`, and `correction`, which has no canon check, publishes
  after its last check passes.
- **Only a publish step leads to `done`.** That is a load error otherwise
  (`npcd/src/engine/workflow/validate.rs`). It is what makes "after verification" a rule
  of the format rather than a habit of whoever writes the YAML.
- **A step may name its own `desk`.** The workflow format gains `desk` as a step key,
  overriding the workflow's desk for that step. `offer_step` builds each step's mission
  at the workflow's desk today (`mission_gen/run.rs:580-588`), so a step at the merge
  table needs the key. The story is written at the story desk; its publish step sends a
  Maker down to the archive floor.
- **The Maker solves a conflict at the publish bench.** A conflict means the record
  moved under the operation at lines it changed: another operation published there
  first, or the same operation was published from the other side of a partition (§9.4).
  The merge table is a bench like the story desk:
  1. `merge_publish` is refused, naming the conflicting paths.
  2. `merge_sync` brings the record's newer commit into the store, with the overlapping
     hunks marked in the document.
  3. The Maker reads them, edits the document at the bench until no marker is left,
     and commits it into the store.
  4. It publishes again.

  The step's `published` check holds it open until the store lands. A Maker that cannot
  resolve it reports `stuck`, and the step goes to another Maker, as every actor step
  does. Nothing goes back through verification: the operation's own verified text is
  what is being merged, and a record that moved only at other lines merges with no
  conflict at all.

**The archive floor and its acts.**

- A new level, `vault-archive` (ordinal 8), joins the building through its `contains:`
  list and a lift portal (`npc-map/maps/creators-vault.yaml`).
- Its archive room holds a station part, `merge-table`.
- Its acts, with the namespace `archive` (`npcd/src/effector/namespace.rs`), are:
  - `merge_status`: what the operation's store changes against its base, and whether it
    merges cleanly with the record as it stands;
  - `merge_publish`: publish the operation's store, or refuse with the conflicting
    paths;
  - `merge_sync`: bring the record's newer commit into the store, with each overlapping
    hunk marked in the document for the Maker to resolve at the bench. The bench's own
    file acts edit and commit the result.

  There is no abandon act. A failed or cancelled operation's store is discarded by the
  engine, as `on-failed` says.
- Their handlers are in `npcd/src/engine/work.rs`. Adding them takes the same pieces
  `vault-time` and its `time-machine` took: a level YAML and a part YAML under the
  world's map directory, `Tool` consts with `at: &["merge-table"]` chained into `CATALOG`
  (`npcd/src/engine/tools.rs:956-965`), and a match arm in `work::perform`
  (`work.rs:708`).
- `merge_publish` lands the operation's store on the record as **one commit on the
  store's base**, pushed under a lease on what the record held when the commit began
  (`npcd/src/opstore/merge/publish.rs`). A record that moved since the base is merged into
  the store first when it merges cleanly, and the commit lands on top of it; one that
  moved at the lines the store changes is refused with the paths. There is no operation
  branch: the record is the mind's branch on `origin`, as it is for zend's `land`. The
  member builds the commit in its own clone of the mind; only the push needs credentials,
  and it goes through an agent holding the mind with push rights (§11.4, `AgentPush`), or
  through the mind's own checkout on a member run without `--cluster`. The commit message
  names the operation, its document and every step's taker, from its run history.
- **On success** the store's base moves to the merged commit and its published paths are
  tombstoned, which is zend's `land` behaviour (`zend-vfs/src/work/commit.rs:200-218`).
  The run reaches `done`, and the store is dropped with the operation's settlement.
- **Every member's mind folder follows the record** (`npcd/src/engine/record_follow.rs`).
  The publishing member brings its folder up at once; every member fetches the record
  every 60 s and checks it out. A folder with changes of its own on disk, or a branch
  holding commits the record does not, is left as it is and said so in the log — never
  overwritten. Each time the folder moves, the watcher ingests what changed
  (`npcd/src/engine/watcher.rs`) and the worlds and personalities are read again.

**When git cannot be reached, the step waits and is tried again.** A publish needs
`origin` and an agent to push with (§11.4). Either can be unreachable: `origin` is down,
no agent is connected, or the world runs in disconnected mode on a member with no
in-process agent (§9.4).

- **The attempt fails without settling anything.** `merge_publish` answers with the
  reason, and the run stays on `publish`. It is neither an outcome nor a stuck report,
  so it routes nowhere and counts toward neither `send-backs` nor `stuck-limit`.
- **The step is offered again 30 minutes later.** The operation records `retry_at`, and
  the table does not offer its step before then. The operations tab shows the wait and
  the last reason, and an operator can offer the step at once.
- **The store keeps the work meanwhile.** Nothing about the operation changes but the
  time, so a publish that fails for an hour is published intact once git is back.
- **The wait is fixed at 30 minutes**, the same on every world and member, so a
  disconnected world needs nothing from a gateway to keep it.

**Duplicates are git's to handle.** In a partition, the same operation can be carried
to `publish` on both sides: by the authoritative world, and by a disconnected one with
its own copy (§9.4). Nothing tracks that.
- If both wrote the same text, the second publish merges as a no-op.
- If they differ, the second publisher meets a conflict and solves it at the bench like
  any other.

**Other npcd content.**

- **Admin writes are drafts** (`npcd/src/drafts/`). Everything an admin writes to the
  mind through the console — a world or personality (`PUT /v1/world/:wid`,
  `/v1/personality/:aid`, the world's clock), a corpus entry or its fields
  (`/v1/mind/entry`, `/v1/mind/fields`), a life (every `/v1/life/:who` write and the
  generation ladder) and a reshaped map (`/reshape/:world`) — becomes `VfsEvent`s on the
  admin's own **authoring store**:
  - One store per account, owned by the account's public id (`u_…`, what its characters
    are owned by). Its `TimelineId` is `hash("author" ‖ id)`. A reshape asked for
    through a character's token is a write of that character's owner.
  - The admin reads the published mind as it stands, with their own drafts laid over it,
    in every listing and document read; everyone else reads the published mind. The
    drafts are made on the commit the store held when the first of them was written, and
    an empty store takes the record as it stands again.
  - A world or personality is rendered against the text the author sees, so its comments
    survive (`npcd/src/registry/authored.rs`). A life's plan and every document it
    writes are one commit.
  - The console's **Publish** (`POST /v1/mind/publish`) runs the same merge as
    `merge_publish`, with the commit message naming the account and every drafted path.
    A conflict is refused with its paths (`409`); **Sync** (`POST /v1/mind/sync`) brings
    the record's newer text in with each overlap marked, and the author settles the marks
    in the editor and publishes again. **Discard** (`DELETE /v1/mind/drafts`) drops them;
    `GET /v1/mind/drafts` lists them.
  - Over a mind folder that is not a git repository there is no record to land on, and
    an admin's write is the file itself, read again at once.
- **Accounts** become `CustomObject account/<key>` records, `<key>` the first 128 bits of
  `SHA-256(issuer ‖ 0 ‖ subject)`. A daemon's old `accounts/` folder is taken in once at
  startup and removed.
- **Images** become a new content-addressed `Blob` record (tag 29): its payload the
  SHA-256 and then the bytes, its header key the hash's first sixteen bytes. An image's id
  is `<sha256-hex>.<ext>`, and `portrait_image_id` names it. A daemon's old `images/`
  folder is taken in once at startup, and every character's portrait pointing at an old
  id is pointed at the blob it became.
- **Device tokens** stay local. They authenticate a device to the one machine it was
  issued on.

### 10.4 Ingest happens once

Each zend ingests its own workspace (D10). Branch ingest keys a unit by content —
repository, path and blob id (`zend/src/branch_ingest/mod.rs:1-8`). It plans each layer
against what that layer already holds (`branch_ingest::plan`), and what a layer holds
is its ingested units, which are substrate records and replicate. So a member skips any
unit another member has already ingested, with no ledger of its own: the replicated
units are the ledger.

A unit nobody has finished is leased before it is read (`zend/src/ingest_lease.rs`),
under the key `ingest/<content key>`. The member the vote names reads it. Every other
member that found the same unit leaves it and takes the next unread one, and the
conversation arrives with replication. Each lease is only that one file or folder, so
the members that are up read different units at the same time: a startup splits the
unread backlog across them, and each unit is still read once.
The holder renews the lease until the reading finishes or is abandoned, then hands
it back with how far its own writes reach. The next member to take the unit waits
for those writes and reads it only when the content key is still absent — a holder
that died mid-reading leaves the unit for someone else, and a holder that finished
leaves nothing to read. A member that is not joined cannot propose, so it reads the
unit itself.

Two members can still both finish a reading: one was not joined, or both passed the
grant. Both units' conversations arrive everywhere. **The conversation whose first
turn has the lower version stamp (§6.2) holds the unit**, the timeline breaking a
tie, and the planner tombstones the other (`zend/src/branch_ingest/plan.rs`) — the
same choice on every member in any arrival order, since the stamps and timelines
are the records' own. Compaction then drops the tombstoned conversation's records,
as it drops any record §6.5 makes dead. Unlike two turns at one index (§6.6), two
ingests of one unit are the same content twice, so keeping one loses nothing.

npcd's mind ingest is done only by the authority of the world that reads the mind, by
the same rule.

---

## 11. Agents

### 11.1 What an agent is

A new binary, `tokera-agent` (crate `agent/`). It is CPU-only and links `cluster` and
`zend-vfs`, with no `candle`.

- **Joining.** It is started with one or more `--cluster` URLs and dials
  `/v1/cluster/agent` on each, with the same handshake, the same group key and its own
  build floor. It has no model and no substrate.
- **Checkouts.** It holds a checkout of every repository it advertises, and the
  credentials to push to their origins, laid out as
  `docs/zend_workspace_execution.md` §7.1 describes.
- **Serving.** It serves its job API over virtual connections like any member.

### 11.2 What it advertises

`HELLO.repos` lists each repository's name, origin URL and whether it can push.
`CAPACITY` adds the following:

- OS
- logical cores
- the one-minute load average
- free memory
- job slots in use
- toolchains: which of the programs on the allow-list (`zend/src/sandbox_programs.rs`)
  are found on its `PATH`
- its state: `ready`, or `disabled` while an admin has switched it off (§11.7)

### 11.3 Selection

Job routes are dispatched `Sticky(Header("x-tokera-lineage"))` with a spill rule. A
conversation's jobs go to the agent that last ran them, which keeps incremental builds
warm (`docs/zend_workspace_execution.md` §8.2).

If that agent is busy, the job waits up to 5 s and then spills to the **best idle
agent**: the one with the most idle cores, `cores × (1 − load/cores)`, among agents that
hold the job's repositories and toolchain.

**An agent at low priority** — one sharing its host with a model member (§11.5), which
says so in its `CAPACITY` — is chosen only when no dedicated agent fit for the job has a
free slot (`cluster/src/gateway/agents.rs`). It is never the best idle agent beside a free
dedicated one, a lineage it holds moves to a free dedicated agent, and a job that must
queue queues on a dedicated agent.

A fork's lineage header is its parent's.

### 11.4 Jobs

The agent's API (`agent/src/routes.rs`), served only over a stream a gateway opened for a
member:

| Route | |
|---|---|
| `POST /v1/jobs` | A run, `{ "job": "run", repo, branch, overlay, program, args, timeout_secs }`, a push, `{ "job": "push", repo, bundle, specs: [{ target, to, expect }] }`, or a fetch, `{ "job": "fetch", repo, branch, have }` (`agent/src/wire.rs`). Answers the job's view: its id, its status, its log's size, and once it has ended a run's outcome, a push's refs, or a fetch's `fetched: { tips, bundle }`. |
| `GET /v1/jobs` | Whether the agent takes jobs, its slots, and its newest jobs (`?limit=<n>`, 40 by default), each with its kind, its command line, its branch, its status, its log's size, when it started and ended, and the files a run changed (§11.7). |
| `GET /v1/jobs/:id` | Status: queued, running, exited, timed out, cancelled, refused or failed. `?wait=<s>` holds the answer until the job ends, up to 30 s. |
| `GET /v1/jobs/:id/log` | The log from its first byte. |
| `GET /v1/jobs/:id/tail?from=<byte>` | Up to 256 KiB of the log from byte `from`, base64 — a chunk can end inside a character — with the log's length, and whether the job had ended before the read (§11.7). |
| `DELETE /v1/jobs/:id` | Cancel, killing the process tree. |
| `PUT /v1/agent` | `{ enabled }`: switch the agent on or off (§11.7). |

`overlay` is the conversation's or operation's store `Snapshot` (`zend-vfs/src/vfs/mod.rs:380`),
its base included, which the 10 MiB store cap keeps small. That answers the open question
of sync granularity (`docs/zend_workspace_execution.md` §9.4) with a per-job overlay over a
fetched base.

**A run** goes through the existing sandbox cycle against the agent's own checkout:
fetch the base, then preserve → materialize → run → capture → restore
(`zend-vfs/src/sandbox/mod.rs:1-45`). It returns the exit code, the log, and the captured
deltas. The calling member applies the deltas to the store. They are then ordinary VFS
events and replicate.

**A push** carries what the member built. Commits and merges are made in the member's
own clone with the existing `zend-vfs` machinery; only the write to `origin` needs the
credentials an agent holds. So the member bundles the objects `origin` lacks — a git
bundle, base64 on the wire, absent when it lacks none — and names each ref the push sets
with the lease it holds on what `origin` has, and the agent pushes them in one atomic push
(`agent/src/pusher.rs`). A lease `origin` no longer matches is refused exactly as it would
be from the member's own clone.

**A fetch** brings what `origin` holds to a member that cannot reach it — npcd's mirror of
the mind (§10.3). The member names the commits its tracking refs hold as `have`, and the
branch it wants, or none for every branch. The agent fetches `origin` and answers with
each branch's commit and a git bundle of their objects that `have` does not reach —
absent when the member lacks none (`Repo::bundle_for_fetch`). The member takes the
bundle's objects and sets its tracking refs to the commits, removing those of branches
`origin` no longer has (`Repo::take_fetch`): the same refs, and the same report, as a
fetch from `origin` itself. The answer goes back with the submit only; the job keeps its
status. A fetch that brings the member nothing — no objects, every branch at a commit it
already holds — is answered and not kept: a member following origin asks every minute,
and kept, those polls filled the agent's job table and pushed the runs that did something
out of it (§11.7). A repository opened `fetching_through` an agent makes every fetch this way, as
one opened `pushing_through` one makes every push (`zend_vfs::Fetch`, `zend_vfs::Push`).

### 11.5 How zend calls it

With `--cluster`, `run_command`, `run_output` and the git writers dispatch to agents
through a stream the member opens to `group:agent` (§5.1). An Admin's operations tool
starts a run job the same way, without waiting for it (`Agents::submit`), and a run job
naming a `stage` has the agent copy what it built to the agent's `stage_dir` once it
exits 0, for that machine's keep (`docs/claude_zend_gateway.md` §8.3–§8.4). The in-process `Sandboxes`
(`zend/src/tools.rs:771`) remain the executor of a daemon without `--cluster`. That is a
deployment configuration, not a fallback: with `--cluster`, an unreachable agent is the
tool error `no_agent`, never a quiet local run.

A machine with a GPU can also be an agent. `motherd --agent <agent file>` — the file
naming its checkouts, job slots and jobs folder (`agent/src/config.rs`) — opens a third
socket that serves the job API in-process at low priority, which is §8.1's "GPU nodes may
register as low-priority build agents". Low priority is two things: its commands, and
everything they start, run below normal OS priority (`zend_vfs::Priority::Below` — the
below-normal class held by the command's job object on Windows, a nice of ten on Unix),
so a build never takes the processor the decode needs; and the gateways choose it last
(§11.3).

**A member cut off from every cluster can still reach its own in-process agent.** Agents
dial out and have no listener, so a member cut off from the gateways reaches no other
agent. When it is cut off, a `motherd --agent` sends jobs straight to the agent in its
own process. That is how a disconnected npcd world publishes (§9.4, §10.3). A member with
no in-process agent fails the job with `no_agent`. A zend tool reports that to the
model; a publish waits 30 minutes and tries again; the mind's follow tries again in a
minute.

**npcd** reaches its mind repository the same way: its fetches and pushes go to an agent
holding it (§10.3), chosen by the repository's name as any job is.

### 11.6 Safety

The rules of `docs/zend_workspace_execution.md` §7.3 apply unchanged:

- Jobs run as a low-privilege user whose writes reach only the checkouts and caches.
- Credentials are never readable by a job.
- Every job runs under a timeout that kills its whole process tree.
- Output is rewritten to repository-relative paths.

The push credentials live on the agent, never on a GPU member.

### 11.7 Watching and switching agents

An admin sees every agent a daemon's commands run on, what each is running, and its
output as it is written, and can take an agent out of service — zend's **Agents** page
(`zend/web/agents.html`, `docs/zend_ui_redesign.md` §2.8). Everything goes through the
member's own streams to `group:agent`, as its jobs do; nothing new listens anywhere.

- **The listing is the gateway's.** `GET /v1/agents` on a stream towards `group:agent`
  is answered by the gateway itself from its registry
  (`cluster/src/gateway/agent_row.rs`): each admitted agent's node, host, build, state,
  last `CAPACITY` and repositories, and when its socket connected. A member cut off from
  every cluster lists the agent in its own process instead (`Agents::listing`).
- **Each agent is then asked, by its node.** `GET /v1/jobs` gives its newest jobs, and
  `GET /v1/jobs/:id/tail?from=<byte>` its log from a byte, addressed with
  `x-tokera-node` like any read-back (§11.4). The command line a job shows is a run's
  program and arguments, or the git command a push or a fetch makes.
- **Output is read only while someone watches.** zend follows a log with its tail —
  straight on while a read comes back full, every 300 ms once caught up — and only while
  a browser holds the stream open (`zend/src/api/agents/tail.rs`). The page opens it when
  an agent is expanded and closes it when it is collapsed, so an agent nobody is looking
  at is asked for its jobs every few seconds and never for its output.
- **The switch.** `PUT /v1/agent { enabled }` switches an agent off or on, kept as a
  `DISABLED` file beside its job logs so it survives a restart (`agent/src/switch.rs`).
  Off, its `CAPACITY` reports `MemberState::Disabled`: it is not ready, so the gateway
  routes it no job (§11.3) — a job no other agent can take is `no_agent` — and the new
  `CAPACITY` goes out at once rather than at the next second. A job sent to it anyway, by
  a member cut off and reaching its own agent, is refused as `no_agent` too. The jobs it
  is running carry on to their end, and every job it kept can still be read and
  cancelled.

zend serves this at `GET /v1/agents`, `PUT /v1/agents/:node`,
`DELETE /v1/agents/:node/jobs/:id` and the SSE `GET /v1/agents/:node/jobs/:id/stream`
(`zend/src/api/agents/`) — an admin's only, since they show every conversation's commands
and switch the machines that run them. A daemon without `--cluster` lists no agents and
says why.

---

## 12. The gateway status page

`GET /gateway` serves the page and `GET /gateway/status.json` its data. Both are on
every host, merged ahead of site routing, and admin only. The data also answers a
caller presenting the **status key** as `Authorization: Bearer <key>`: a server-side
secret, written by `web --new-status-key <path>` and named by the `cluster:` block's
`status_key_file`, so a script or an operator at a shell reads how the cluster stands
without a browser session (`web/src/cluster/bearer_key.rs`). It reads the status and
nothing else, and it is its own key, never the group key, so it admits no member.

The **ops key** is the same kind of secret with an admin's reach: written by
`web --new-ops-key <path>` and named by the `cluster:` block's
`ops_key: {file, admin}`, a caller presenting it is the admin `admin` names on every
route of the gateway's own — the page, `drain-gateway`/`undrain-gateway`, every
upgrade command (§12.4), and the member commands the page sends — and that admin is
the identity the gateway names to the member a command goes to, so the role table
must list it (`web/src/cluster/ops_key.rs`). It exists for the emergency that cannot
wait for a sign-in: whoever can read the key file on the gateway's box — the same
person who can replace the binaries keep runs there — rolls a bad build back from a
shell:

```text
# the plan: which members install what, and the builds it rejects
curl -X POST -H "Authorization: Bearer $(cat ops.key)" \
     -d '{"app":"zend","build":"<previous count-hash>"}' \
     https://<gateway>/gateway/upgrade/rollout
# confirmed: the plan's source proposes rollout/<app>/<n>, which moves the floor
curl -X POST -H "Authorization: Bearer $(cat ops.key)" \
     -d '{"app":"zend","build":"<previous count-hash>","confirm":"<previous count-hash>"}' \
     https://<gateway>/gateway/upgrade/rollout
```

A member's own `rollback` refuses a build below the floor — the member would be
admitted only to be upgraded straight back — so a downgrade is always this `rollout`,
which rejects the bad build and lowers the floor with it. tokera.com's top bar links to
it as **Cluster** beside the signed-in name, for an admin or a creator only:
`/auth/me` carries the caller's `role`, and `common/lib/auth.js` offers the link where
the page would not be refused. The gateway resolves roles from `host/roles.yaml` itself
— the table every daemon compiles in — named by `web.yaml`'s `roles_file:` and read into
`Config.roles`, so the gateway and the daemons cannot disagree about who is an admin. The
page polls once a second.

**Every coordination decision in this design is visible here**: membership, floors and
locks, modes, claims, world and NPC leases, generations, catch-up, retransmit and
tombstone expiry. That includes the ones only a member can see.

### 12.1 What members report: `COORD`

A gateway sees its own queues, the claims it is told and the Paxos frames it carries. It
cannot see a member's generation, its acceptor state, the promises and acceptances its
proposals gathered, or how far it has caught up. So each member sends a `COORD` frame on
every socket: every 2 s, at once when its mode changes or a maintenance command starts
or ends, and within a second of any other change (`replicate/coord.rs`). Its body is the
JSON of `cluster::coord::CoordState`, fields in declaration order, so one state has one
encoding; the gateway diffs each against the member's previous report — kept across its
sockets, so a member that reconnects is logged for what changed while it was away — and
writes the differences to the coordination log (`gateway/coord_log.rs`).

| Field | |
|---|---|
| `mode` | joined, cut off for this app, rejoining (catching up, taking its generation or waiting for a quorum to), reloading after a turn ranked below one it held (§6.6), draining (§12.4), upgrading below the floor (§3.5), in disconnected mode for which npcd worlds (§9.4), or **in fault mode** (§7.7). Fault outranks draining, and draining the rest. Whether it holds writes from while cut off is `detached` beside it. A member *lagging* (§4.4) is the gateway's own judgement, shown from its queues |
| `upgrade` | `cluster::coord::UpgradeView` (`upgrade/runtime/view.rs`): its kind and platform; what its build opens — log format, the record census of its substrates, model — against which a build is checked (§3.9); every artifact it holds (`holds`: build, kind, platform, size, SHA-256, and whether it runs, keeps, verified or offers it); its kept builds with their bytes; the fetch in progress (build, kind, bytes against size, resumes); as a candidate, its build, the verdicts each gateway sent and what it proposed; its `step` in an install (waiting, upgrading, installing, stranded); why it is stranded; the failure its last install was restored from, with the new process's last log lines; the builds it refuses; the group's latest verdict on each build (`verdicts`: build, number, verdict); the latest `rollout/<app>` decision it holds (`rollout`: number and decision); and whether the app's rollout is paused |
| `fault` | in fault mode: the reason (disk full, or certification failed), the volume's free space against the floor, since when, what it gave up (claims, worlds, characters), that it is still voting and the vote journal's slots in use, and space freed since |
| `gen` | its current generation, the generation it took at its last rejoin, and the greatest `gen/<k>` decision it knows (§6.7) |
| `deferred` | records it holds back because its build does not decode their type, by type and origin (§6.5) |
| `clusters` | every `--cluster` it was given, with each one's socket state, so every gateway can show the others' view |
| `paxos` | for each key it proposes for, holds, or has voted on: the round in progress, its ballot and phase, the members that promised or accepted it, and the higher ballot a `NACK` beat it with; a lease's remaining term on its own clock; and as acceptor, its promised and accepted ballots. With them its `max_members` and quorum (§5.9) |
| `worlds` | each world it hosts: its role there — authority, replica or disconnected — with its epoch as authority and since when; the handover it waits on, `<origin> through <oseq>`; and, as the world's authority, every lease it lends (§9.2–§9.4) |
| `npc_leases` | the NPC leases it holds, by world, with expiry (§9.3) |
| `claims` | the sticky keys it holds, with epochs (§5.4) |
| `applied` | per origin: `applied_through`, gaps, and the origin's announced `high_water`, so a lag shows as a number (§7.1) |
| `outbox` | its own events unsent and in flight (§6.3) |
| `catchups` | each catch-up in progress: its socket and why it runs, its answerer, records fetched against wanted and skipped as held, and bytes fetched against wanted (§7.2) |
| `tombstones` | timeline tombstones held, how many a full compaction has run past and so wait only for their age, and the deleted set's entries (§6.8) |
| `disk` | the substrate's volume: free and total bytes, and the fault floor (§7.7) |
| `commands` | the last 8 maintenance commands run on it: job, command, arguments, who ran it, when, and how it ended (§12.4) |

Two of these are not the member's to report: the gateway measures a member's **clock
offset** itself, from the `now_ms` of its `HELLO` against its own clock at admission
(§3.4), and shows it on the member's row. An agent, with no substrate and so no
replication, still sends `COORD`, every 2 s, so its `upgrade` view reaches the page.

### 12.2 The page

The page is laid out for operating, top to bottom: a one-line summary and a **Needs
attention** list naming each problem in a sentence (faults, lagging or stranded members,
a failed candidate, an app still loading, recent refusals), the headline figures, one
card per **app** with its members and their actions, **Builds and upgrades**, the
**Activity** of the commands run from this page, and the blocks below under
**Diagnostics**, each a section closed until opened. It wears tokera.com's own bar and
stylesheets.

| Block | Shows |
|---|---|
| **Apps** | For zend, npcd, agent and keep: the **mode** (dumb or clustered) and when it last switched (§5.7); the build floor (count, hash, the node that set it) (§3.5); the model lock (§3.6); the member count and how many are ready; **`max_members` and the quorum**, the greatest any member reports while they converge (§5.9). Whether it was raised or set lower, and by whom, is in the log. The pending floor and its grace countdown, or *loading* while the candidate loads (§3.5). keep is the failsafe, listed as itself: no quorum, no rollout, no certification |
| **Upgrades** (§3.8, §3.9, §3.11) | Per app, the candidate build with its node and state (loading, certifying, passed, passed with gaps, no opinion, failed), the grace countdown from when it became ready, how long since the floor last rose while that rollout lasts, and the builds the group certified and rejected. **Certification, per route:** samples live, baseline and mirrored; error rate for candidate and baseline; p50 and p95 ratios; mirror mismatch rate; the route's outcome and why; and the last mismatching pairs shown side by side. **Distribution:** each member's build (a local build the floor has passed marked *behind floor N*, §3.10), its install step, the fetch in progress with bytes and resumes, its kept builds and their bytes, why it is stranded or which install failed (with the new process's last log lines), and the builds it refuses; and *Roll out this build…* on each. Which members hold `upgrade/<app>` or a rollout's `parallel` leases is in the Consensus block. **Builds held:** every build any member runs, keeps, verified or offers, newest first, with each artifact's kind, platform, size and holders, rejected builds marked, and a *Roll back to…* action on each. The gateway commands for this block sit here (§12.4), with the plan a `rollout` answers shown for confirmation. |
| **Faults** (§7.7) | Members in fault mode, at the top of the page and in red: reason, free space against the floor, since when, what each gave up, its unsent events, space freed; and members within 2× of the floor, as a warning before they fault |
| **Members** | For each one: node, host, build (`count`, `hash`, and for a local build its `local` digest, flagged *local*), power and weight, idle, free slots, state (loading — connecting or catching up — / ready / upgrading / draining / **fault**, and *lagging* beside it), its `mode` and `gen`, free disk on its substrate's volume, session, time connected, clock offset, the last close; and its `clusters` column, the socket state to every gateway it was given, which is what explains a member this gateway sees and another does not |
| **Generations** (§6.7) | Per member: its mode, the generation it writes under, the one its last rejoin took, the greatest decided, whether it holds writes from while cut off, and the records it defers for a type its build does not decode (§6.5). Each rejoin, the generation it took, and each reload are in the log |
| **Claims** (§5.4) | Live claims, with key, holder and epoch, as this gateway has been told them. Each claim taken and released is in the log; `409`-then-`503` answers are counted under Traffic |
| **Consensus** (§5.9) | Every Paxos key each member knows — leases and decisions alike: the round it has in progress, its ballot and phase, the members that promised or accepted it, the ballot a `NACK` beat it with, a lease's time left, and **its acceptor's promised and accepted ballots**, as reported in `COORD`, so a split vote shows member by member |
| **World leases** (§9.2; chunk 7) | For each npcd world: its authority and term left; the member rendezvous names as proposer, flagged where it is not the authority; worlds with no holder, and why (no quorum reachable, a term still running out); handovers in progress, with the `through` the new proposer is waiting on |
| **NPC leases** (§9.3; chunk 7) | Per world: characters by holder and expiry; characters held by members now away, which are not lent again until those members return; forced admin releases |
| **Disconnected worlds** (§9.4; chunk 7) | Members running worlds in disconnected mode: which worlds, which characters, since when |
| **Replication** | For each member and origin: `applied_through` against `high_water`, how far behind that is, and the `oseq`s missing below the top (§7.1). Per member: its outbox, unsent and in flight (§6.3); this gateway's queues to it — live and catch-up bytes, live events skipped while it lagged, catch-up frames dropped at the cap (§4.4) |
| **Recovery** | Per member: records applied and refused, catch-ups asked, catch-ups answered and records fetched for others, events retransmitted, reloads; and each catch-up in progress — its socket, why it runs, its answerer, records fetched against wanted and skipped as held, bytes (§7.2). Each catch-up starting and finishing, and each full resync's offer (§7.5), is in the log |
| **Tombstones** (§6.8) | Per member: timeline tombstones held, those waiting only for their age, and the deleted set's entries. Each expiry is in the log |
| **Traffic** | Open streams, requests per second, and the share of each dispatch class (§5.2); aggregates answered and members past their deadline (§5.5) |
| **Failures** | HTTP exchanges failed by a dropped socket in the last hour, split into failed before the response head and aborted mid-body (§5.6) |
| **Agents** (§11; chunk 8) | Cores, load, jobs queued and running, toolchains and repositories |
| **Peers** (§13.4) | This gateway's node; each peer linked, with its build, which end dialled, and whether the link is over verified TLS; each configured peer's dial state and last close; peerings refused and why (a disagreeing session key or OAuth client among them); certificates per site and gateway (`not_after`, digest, any gateway behind); each site's orderer and the gateway ordering it now; challenges outstanding with peer acknowledgements; the shared ledger |
| **Refusals** | The last 100 handshakes refused, with code, node, host and reason. That covers 4401 proof, 4410 build, 4412 model, 4417 clock, and the others (§3.4); members admitted to upgrade below the floor are listed in Upgrades, not here |

### 12.3 The coordination log

Under the blocks, a **log of every coordination decision** this gateway made or was told
of, newest first, kept for 24 hours in memory and filterable by app, member, world or
key:

- a member admitted or refused (code, reason), and a socket closed (code);
- a mode switch, dumb or clustered (§5.7);
- a newer build starting a grace period; a build floor raised at its deadline or by
  `raise-floor`, and who it closed; a pending raise cancelled (§3.5);
- a candidate loading, ready, certifying, passing or failing, with the figures that
  decided it; a build offered, an artifact fetched, resumed, verified or failing its
  hash; each rollout step per member; an install failed and restored; `rollback`,
  `certify-anyway` and `reject-build` (§3.8, §3.9); a forced rollout's plan, its
  decision, the builds a downgrade rejected, and a rollout superseded (§3.11); kept
  builds pruned (§3.9); the model lock set and cleared (§3.6);
- a claim taken, moved or released (§5.4);
- a Paxos step, from the frames this gateway carries: a round prepared, promised,
  refused with a `NACK`, proposed, accepted; and, as members report it, a lease gained,
  renewed, released or run out, and a value decided (§5.9, §9.2);
- `max_members` raised, or set lower and by whom (§5.9);
- a member entering fault mode (reason, free space, what it gave up) and leaving it
  (§7.7);
- an NPC lease lent, its move to another member asked and made, or force-released
  (§9.3); renewals are not logged, and a lease held while its holder is away shows in
  the NPC leases block;
- a world's authority gained, handed over or lost, and a handover waited on (§9.2,
  §9.6);
- a member cut off, rejoining (each step), waiting for a quorum, and the generation it
  took (§6.7); a reload and its cause (§6.6);
- a member admitted to upgrade below the floor, and how its install ended (§3.5); a
  member lagging and caught up again (§4.4);
- a world entering or leaving disconnected mode (§9.4);
- a catch-up started and finished, and a full resync after a restart (§7);
- tombstones expired (§6.8);
- a peer gateway linked, refused (digest mismatch) or lost; an order started; each
  order step; a certificate pushed or pulled (§13.4).

Each entry names the gateway it came from. Two active-active gateways' logs, read side
by side, show the same decisions from both ends.

The counters, the `COORD` table and the log are all new. The gateway has none today
(§1.1). They live in the gateway's cluster module beside the queues and tables they
describe.

Each gateway's page shows that gateway's own view, plus what every member reports about
all its clusters. A disagreement between gateways (members, modes, floors, accepted lease values)
therefore shows on both pages, together with the member reports that explain it.

### 12.4 Maintenance commands

The page runs maintenance from each member's **Actions** sheet: the commands below in
groups (traffic, process, data, storage, checks, builds), each with what it does in a
sentence and only the inputs it takes, run on that member or on every member of its app.
A destructive one is armed only once the member's host name is typed in the sheet, and
runs on one member at a time. An app card opens its `set-quorum`; a rollout is chosen,
shown as a plan in sentences, and confirmed by typing the build.

**Every command is a route on the member.** It is `POST /v1/cluster/command/:name`, in
the host crate (`host/src/commands/`), admin only, its body an optional
`{ app, from, max_members, confirm, build, builds, keep, plan }`. The page sends it through
`/gateway/member/<app>/<node>/v1/cluster/command/…` (`web/src/cluster/page.rs`), which
forwards only `/v1/cluster/…` down that member's socket — the page is served on every
host, so the member is named by app rather than by the host it was reached on. An
admin's `x-tokera-node` (§5.2) reaches the same route through a site. "Every member"
sends it once per member. The same route works on a member's own listener, so a command
can be run without a gateway: with `curl`, or on a member that is cut off. A daemon
without `--cluster` serves it too; the commands that act on a cluster then refuse with
the reason.

| Command | What it does | Built from |
|---|---|---|
| `drain` | Stop taking work but stay connected. Releases every claim, world lease and NPC lease, the way fault mode does (§7.7). The gateways route nothing new to it. It keeps sending its own events and answering fetches. | fault mode's hand-back |
| `undrain` | Take work again, after a catch-up. | |
| `restart` | `drain`, wait for in-flight requests and the outbox to empty (bounded at 60 s), flush and commit the log, close its sockets with `1001`, then relaunch the same binary with the same arguments. | `LaunchSpec::capture` / `relaunch` (`cluster/src/launch.rs`), which `self_heal` and the upgrade's swap use too, one-shot flags dropped |
| `shutdown` | `drain`, flush, close with `1001`, exit. Nothing restarts it. | |
| `compact` | A full `compact()` of its substrate (`persistence/mod.rs:2169-2344`), run drained: a member not already drained drains for it and undrains after. Refused in fault mode, and refused when free space is below the substrate's live size, because the scratch copy needs it. | zend's `--compact-substrate` (`zend/src/loading.rs:30`) as a command |
| `maintain` | One incremental maintenance pass: Drop, Compact or Combine, as the planner picks (`persistence/maintenance.rs`). Allowed in fault mode only as Drop. | zend's `POST /v1/substrate/zend/maintenance`, which it replaces, in the host so npcd has it too |
| `expire-tombstones` | The tombstone expiry check (§6.8), which runs inside a full compaction: when any tombstone has had a complete compaction run past it, a `compact` as above, reporting how many expired; otherwise it says none can yet. | |
| `audit` | The read-only liveness walk: what maintenance counts as live against what it carries (`persistence/liveness_audit.rs`). Reports, changes nothing. | |
| `self-check` | zend only: ask every stored conversation whether it is intact, reporting and tombstoning nothing. | zend's `--self-check` (`zend/src/self_check/`) |
| `catch-up` | Run a catch-up now (§7.2), from the member the gateway picks or from one named. | |
| `resync` | Run the full resync now (§7.5): catch up, then offer what it holds. | |
| `rejoin` | Close and re-dial every socket for the app. A test of the reconnect path. | |
| `check-disk` | Re-measure free space now, and leave fault mode if it allows (§7.7). | |
| `release-leases` | Hand back its world and NPC leases without draining otherwise. | |
| `set-quorum` | Write `cluster/max_members` lower, to the given number (§5.9). | `PUT /v1/cluster/quorum` |

**How a command runs.**

- **It answers at once** — `202` with its job id and stream — and its progress streams
  on `GET /v1/cluster/command/job/:id`, in the NDJSON form npcd already uses
  (`npcd/src/ndjson.rs:40-62`): a `started` line with its arguments, a `step` line per
  step, and exactly one terminal line, `done` with the result or `error` with the
  reason. The page shows each job as it streams. A member keeps its last 32 jobs.
- **It refuses with the reason** when the member is in the wrong state: `compact` in
  fault mode, `undrain` while catching up, anything while another command is running —
  one command runs per member at a time (`409`). A request the member cannot parse is
  `400`, a body with an unknown field `422`.
- **A destructive command needs its target named twice.** For `shutdown`, `set-quorum`
  and `prune-builds`, the page asks for the member's host name
  typed out, and sends it as `confirm`. For `rollout`, it asks for the build's
  `count-hash`. A request without the matching name is refused.
- **Every command is in the coordination log** (§12.3): the member reports its last 8
  commands in `COORD` — who ran it (the caller's email, or `local`), with what
  arguments, and how it ended — and the gateway logs each as it starts and ends.

**Ctrl-C drains first.** The first Ctrl-C, or `SIGTERM` on Unix, runs the same sequence
as `shutdown`. Today `shutdown_signal` (`host/src/process.rs:131-162`) resolves on the
first Ctrl-C and lets the HTTP server drain its in-flight requests. With `--cluster` it
now does, in order:

1. **`drain`.** It hands back every claim, world lease and NPC lease, and stops its
   characters at their next turn boundary. The gateways stop routing new work to it the
   moment its `CAPACITY` and `COORD` say so.
2. **Flushes and commits** the log: everything its tenants handed the writer is on disk.
3. **Waits**, bounded at 60 s, for its outbox of unsent and in-flight events to empty,
   so what it wrote reaches the group before it goes (`host/src/cluster/leave.rs`).
4. **Finishes the requests in flight** — its own listener's, as the HTTP server's
   graceful stop does, and those the gateways sent over its sockets. A request that can
   append (anything but a `GET` or `HEAD`) counts until its response body ends, and the
   member waits, bounded at 300 s, for none to be left (`host/src/cluster/in_flight.rs`).
   A turn streaming its reply rides one of those sockets; closing it first cut every
   such turn mid-stream. A `GET` — a log tail, a follow of a live reply — never ends on
   its own and is not waited for.
5. **Closes its sockets with `1001`.**
6. **Runs the tenants' existing shutdown** (`docs/shared_model_host.md`).

`shutdown` and `restart` run the same sequence (`host/src/stopping.rs`); `restart` then
starts the same binary with the same arguments, one-shot flags dropped.

The handover is complete before it exits: conversations it held move to other members,
which catch up to its last write, and its worlds go to the next proposer at once. They
do not wait out a term.

**A second Ctrl-C still aborts at once** (`exit(130)`), the escape hatch the code already
has for a drain that wedges. Whatever was not handed back then falls to the normal
rules: claims lapse with the member, world leases run out after their term (§9.2), and
unsent events go out when the member next starts (§7.5). Ctrl-C on a daemon without
`--cluster` behaves exactly as today.

The loading page shows the drain as it runs, with each step and what is left: while the
process stops, every tenant's `/v1/status` line is loading again, its load the stop's
step. So a Ctrl-C at the console is visible to anyone watching the member's own page or
the gateway's.

**Upgrade commands on the gateway**, in the Upgrades block: `POST
/gateway/upgrade/<name>`, admin only, beside the page (`web/src/cluster/upgrades.rs`),
its body `{ app, build, from, targets, parallel, confirm }`:

- **`raise-floor` / `cancel-floor`** are this gateway's own. `raise-floor` ends an app's
  grace period now: the gateway sends its verdict at once, and the floor rises when the
  group certifies the build, or when `certify-anyway` accepts it (§3.5). It is refused
  while the candidate still loads, or once its verdict is sent. `cancel-floor` discards
  the pending floor; the candidate stays on its build.
- **`certify-anyway` / `reject-build`** settle a build by hand (§3.8). Gateways hold no
  Paxos state, so the gateway sends each to a ready member of the app as that member's
  own command, and the member proposes a verdict under the next
  `certify/<app>/<count>-<hash>/<i>`, which overrules the certification's. The first
  accepts the build: it resumes distribution and rollout, and a candidate in fault mode
  for it leaves fault mode. The second keeps the build rejected: the candidate is put in
  fault mode, any other member on it drains, and its downloads and part files are
  removed everywhere.
- **`pause-rollout` / `resume-rollout`** hold or release the automatic one-at-a-time
  install (§3.9), sent to a member the same way, which proposes `pause/<app>/<n>`.
- **`rollout`** forces a chosen build onto the app, from a chosen member: `{ build,
  from, targets, parallel }` (§3.11). It is the command behind both *Roll out this
  build…* on a member's row and *Roll back to…* in the Builds held list. It answers with
  the plan first (`gateway/upgrades/plan.rs`), and runs only when the build's
  `count-hash` comes back as `confirm`: the gateway then sends the plan's decision to the
  plan's source member as its `rollout` command, which proposes `rollout/<app>/<n>`. The
  plan names every member it would restart, skip or strand.

A command sent to a member answers with that member's job and node, and the page
follows its stream like any other member command.

**Member commands for upgrades**, in the member menu, each one also the route the gateway
sends the commands above to:

- **`certify-anyway` / `reject-build`** `{ app, build }`, **`pause-rollout` /
  `resume-rollout`** `{ app }`: propose the decision, and finish once the group holds it
  (bounded at 10 s, refused without a quorum).
- **`rollout`** `{ app, plan, confirm }`: decide a confirmed plan as `rollout/<app>/<n>`,
  then, for a downgrade, a rejecting verdict on each build it rolls back from, and an
  accepting one on the rolled-out build if the group had rejected it.
- **`rollback`** `{ build? }` puts this one member on another build by the same in-place
  swap. It takes the previous build by default, or any build it keeps in `builds/`,
  chosen from the menu (§3.9). It takes `upgrade/<app>` like a rollout step. To roll
  back every member, use `rollout`, which also moves the floor. A single member rolled
  below the floor would be admitted to upgrade and put straight back on the floor's build
  (§3.5), so `rollback` refuses a build below the floor and says to use `rollout`.
- **`prune-builds`** `{ builds | keep, confirm }` deletes kept builds from `builds/`:
  those named, or all but the newest *k*. It never deletes a build that a rollout
  targets. It is destructive, because it removes the way back, so the host name is typed
  to confirm it.
- **`upgrade-now`** installs the newest verified build above its own at once, taking
  `upgrade/<app>` like the rollout does.

**Gateway commands**, for maintaining a gateway itself:

- **`drain-gateway`** stops admitting sockets and tells its members, by closing them with
  `1001`, to carry on through their other clusters. It then waits for its in-flight
  requests (bounded at 60 s) so it can be restarted without failing a request.
- **`undrain-gateway`** reverses it.

These two are the gateway's own, not a member's: `POST /gateway/command/drain-gateway`
and `/undrain-gateway`, admin only, beside the page (`web/src/cluster/page.rs`), with a
button for each in the page's header.

---

## 13. Security

### 13.1 The group key

The group key is 32 random bytes, hex-encoded in a file.

- **Gateway:** `web/secrets/group.key`, named by `cluster.group_key_file` in a new
  `cluster:` block of `web.yaml`, resolved like the other secret paths
  (`web/src/config.rs:552-561`).
- **Members:** `--group-key`.
- On Unix, a file readable by group or other is refused, as zend refuses its secrets
  file (`docs/zend_workspace_execution.md` §3.5).

The key only ever keys an HMAC (§3.3). It is never sent and never logged.

### 13.2 Trust

- A request arriving on a cluster stream carries identity headers that the gateway set,
  over a session both ends have proven. The IP allowlist remains for direct listeners.
- A request a member forwards up (§5.8) carries the identity *that member* resolved on
  its own listener. The gateway believes it because the stream rides an authenticated
  socket. A member is therefore trusted to vouch for whoever reached its listener, by
  the same IP allowlist or `--local-signin` it already applies. That is no wider than
  today, where the member would have served the request itself.
- The gateway decides which members get traffic. A member decides nothing about other
  members except through the `MEMBERS` frame it is sent.
- A member with the key but a hostile build could write records that every other member
  applies. The group key is therefore an estate-wide trust grant. It belongs only on
  machines the estate controls, and its file is guarded like `session.key`.

### 13.3 Transport

- Over the internet, the link is `wss` through Cloudflare, and the member verifies the
  certificate.
- On the LAN, the member dials `https://<ip>:9443`, or a LAN name, and verifies the
  certificate by `--cluster-name`.
- One `--cluster` may list both, LAN first (§3.2, paths): the member stays on the LAN
  while it works and falls back to the internet when it does not.
- The hop from Cloudflare to the gateway is the existing LAN HTTP hop.

### 13.4 Gateways peer with each other: certificates and shared secrets

Active-active gateways answer the same public names, so they need some state in common,
and members cannot carry it:

- **ACME challenges.** Let's Encrypt validates HTTP-01 against whichever gateway the name
  reaches. A challenge one gateway started must therefore be answerable by every
  gateway.
- **Certificates.** A certificate one gateway was issued must be served by all of them.
- **Ordering.** Only one gateway should order a certificate for a site at a time,
  against one shared account and one rate-limit ledger.
- **Sessions.** A session cookie one gateway signed must verify on the other.

So the gateways **peer**: every gateway holds one WebSocket to every other gateway.

**Setting it up.**

- `cluster.peers` in `web.yaml` lists the other gateways, each `{ url, name }`: `url` is
  the peer's TLS entrance, `https://<host>[:port]`, and `name` the name its certificate
  is verified against when `url` names an address. It is the one place the gateways
  are named to each other. A gateway without a TLS entrance has no certificates to
  share, and its config is refused with `cluster.peers` set.
- Each gateway dials every peer at `GET /v1/cluster/gateway` through `cluster-dial`, as
  a member dials a gateway: `wss`, the peer's certificate verified against the Mozilla
  roots. When two dials cross, both ends keep the link the lower node dialled and close
  the other 4409, which ends that dial loop: the lower node's dial is the one kept, and
  it redials when its link drops (`cluster/src/peer/crossing.rs`).
- The handshake is §3.3's, with the group key and `app = gateway`; `ADMIT` carries
  nothing but the session. Reconnect backs off as §3.2 does.
- Each gateway's node id lives in its ACME store directory, `<store>/node`, beside
  `account.json`, minted at its first start.
- **A link is secure** when it runs over a TLS connection whose certificate was
  verified: the dialler's `url` is `https`, and the dial arrived on the peer's TLS
  entrance. Certificate keys and the account cross secure links only; challenges, the
  ledger, the inventory and the order announcements cross any link.
- The transport is the `cluster` crate's (`cluster/src/peer/`): frames, the link, the
  dial loop, the orderer. What arrives goes to the gateway's ACME
  (`web/src/acme/peers/`).

**What crosses a peer link.**

| Frame | Carries | Applied |
|---|---|---|
| `PEER_HELLO` | the gateway's node, build, and SHA-256 digests of its session key file and its Google OAuth client id (of nothing, without sign-in) | **Mismatched digests refuse the peering**, closed 4401 with the reason recorded on both status pages, because a cookie signed by one would not verify on the other. The secrets themselves never cross. A node other than the handshake's, or the gateway's own, is refused 4400. |
| `ACME_CHALLENGE { site, token, answer, expires_ms }` / `ACME_CLEAR { token }` | the HTTP-01 answer a gateway is about to ask the CA to check | added to or removed from each peer's `Challenges` table (`web/src/acme/challenges.rs`), which `/.well-known/acme-challenge/:token` serves on every host. A peer drops one its orderer never cleared once `expires_ms` (10 minutes on) has passed. |
| `ACME_ACK { token }` | a peer holds the challenge and answers it | the orderer validates once every peer it sent the challenge to has acknowledged, or has dropped its link, or 30 s have passed |
| `ACME_CERT { site, chain_pem, key_pem, not_after }` | a certificate the gateway was issued, with its key; secure links only | installed in the peer's resolver and written to its ACME store, `certs/<site>.{crt,key}.pem`, when it is later than the one held. The later `not_after` wins, so the outcome is the same in any order. One arriving on a link that is not secure is refused and logged. |
| `ACME_LEDGER` | the gateway's order ledger, as its JSON (`web/src/acme/ledger.rs`), on connect and after every change | merged: every attempt and every account either holds, once each, in time order, under one lock with the gateway's own saves. Every gateway counts every order against the CA's rate limits. |
| `ACME_INVENTORY` | on connect, each site's `not_after` and the digest of its chain | a peer holding an older or missing certificate is sent the newer one, over a secure link |
| `ACME_ORDERING { site }` / `ACME_DONE { site }` | the orderer starting and finishing an order, and on connect an order running | a peer that sees `ACME_ORDERING` does not order that site until `ACME_DONE`, or until the link drops |
| `ACME_ACCOUNT` | the shared account's credentials, as their JSON, on connect and when created; secure links only | taken by a peer that holds none |

**Who orders: rendezvous over the connected gateways.**

- A site's **orderer** is the gateway that rendezvous (§9.5, with equal weights) names
  for the site among itself and the peers it is connected to. Every connected gateway
  computes the same answer, so exactly one of them orders. No consensus is run.
- The orderer runs the order in `acme/manager.rs`'s renewal loop: at start, then every
  12 h, renewing 30 days before expiry. It announces the order with `ACME_ORDERING`
  (`web/src/acme/order.rs`).
- A due site whose orderer is a peer, or that a peer announced, is skipped. A peer
  leaving, or finishing an order, wakes the loop, after at least 5 minutes from its
  last pass, since this gateway may now be a due site's orderer.
- Gateways share one ACME account. `account.json` is created by the first gateway and
  sent to peers that hold none, over a secure link, so the ledger counts against one
  account.

**One order, start to finish:**

1. The orderer sends `ACME_ORDERING`, creates the order, and gets its HTTP-01 tokens.
2. It sends `ACME_CHALLENGE` for each token to every peer, and waits until every
   connected peer has answered `ACME_ACK`, bounded at 30 s.
3. Only then does it tell the CA to validate. Wherever the CA's request lands, that
   gateway has the answer.
4. It sends `ACME_CLEAR` for the tokens. On issue it installs the certificate and sends
   `ACME_CERT` to every peer, and then `ACME_DONE`. The renewal loop then records the
   outcome in the ledger and sends that.

Whoever signs, the steps around it are the same: the CA is a trait (`acme/ca.rs`), and
the answers are published through another (`acme/publish.rs`) — on this gateway
alone, or on it and every peer.

A peer that was disconnected through all of this gets the certificate from the inventory
exchange when it reconnects.

**Partitions cost at most a duplicate certificate.** Two gateways that cannot reach each
other each name themselves orderer, so each may order a site the other also orders.

- Each then validates against itself alone. A CA request that lands on the other
  gateway fails, and is retried at the next 12-hour check.
- At worst the site gets two certificates. Both are valid, the later `not_after` wins
  when the peers meet again, and the shared ledger counts both against the CA's limits.
- Renewal starts 30 days early, so none of this is close to an expiry.
- With either gateway down, the other orders alone, with no wait.

**Private keys cross only an authenticated, verified link.** Certificate keys travel
only over peer links that passed the group-key handshake on a TLS connection whose
certificate was verified (§13.3). They are written to the peer's ACME store with the
same permissions as today's.

**On the status page** (§12.2), a **Peers** block shows:

- this gateway's node, and every peer linked, with its build, which end dialled it,
  whether the link is over verified TLS, and since when;
- each configured peer as this gateway dials it: its state, attempts and last close;
- peerings refused, with why: a disagreeing session key or OAuth client, a failed
  handshake;
- the certificate inventory per site and gateway: `not_after`, digest, and which gateway
  is serving an older one;
- each site's orderer, and the gateway ordering it now;
- challenges outstanding and the peers that have acknowledged them;
- the shared ledger.

Every link and unlink, refusal, certificate installed from a peer, account taken and key
withheld from an unverified link goes in the coordination log (§12.3), under app
`gateway`.

---

## 14. Testing

Every test below is CPU-only, runs in `/fast-test`'s CPU pass, needs no model and no
GPU, and uses loopback sockets or in-memory duplex pipes.

`cluster` joins the per-package `opt-level = 2` block in the root `Cargo.toml`. Without it
the burst harnesses would measure an unoptimised codec (the block today covers
`candle-*`, not `web`, `host`, `zend` or `npcd`).

### 14.1 Unit tests (raw expected bytes, per `CLAUDE.md`)

| Module | Asserts |
|---|---|
| frame codec | each frame kind encodes to exact bytes and decodes back; a truncated frame or an unknown kind is refused, never misread |
| handshake | the proofs for a fixed key and fixed nonces equal recorded HMAC bytes; a swapped direction label fails; nothing but handshake frames is accepted before `ADMIT`; constant-time compare |
| close codes | each refusal yields its code and its reason text; 4410 and 4426 stop the dial loop, the rest back off; a second authenticated `HELLO` for a node replaces its session at once and closes the old one 4409, and an unauthenticated one replaces nothing |
| backoff | under `start_paused`, delays run 250, 500, 1000, 2000, 4000, 8000, 8000 ms, each drawn within `[d/2, d]`, and reset on admission |
| build floor (`gateway/admission.rs`, `gateway/upgrades/tests.rs`, `upgrade/runtime/decide.rs`, `tests/builds.rs`) | the first member sets the floor, equal and higher builds are admitted, and a lower one is closed 4410 naming both builds and the setter, unless a member holds the floor's build, when it is admitted *upgrading*; neither a local build nor one whose `HELLO` says it is rejected sets the floor; a floor set by a rollout moves either way; another app is untouched. A newer build is the candidate and its grace starts only once it is ready; an even newer one replaces it and keeps the deadline. The group's `Pass` raises the floor and its `Fail` rejects the build and drops the pending floor; `raise-floor` sends the verdict at once and `cancel-floor` discards the pending floor. An upgrading member installs the floor's build with no lease, installs nothing once that build failed it, is never ready for work, and is closed 4410 at once when stranded or restored; members below a risen floor are closed 4410 with their reason once it settles, and the rollout ends when none is left below. In process: a member below the floor is admitted upgrading, offered the build, and installs it; a gateway restarted after a rejection takes the next member's build for the floor, not the rejected one |
| model lock | the first member sets it, a mismatch gets 4412, and the lock clears when the group empties |
| applied set | `CATCHUP.since` and `HAVE` pack to exact bytes; an event that overtook a lower one is recorded exactly and the lower one is still taken; comparing a `HAVE` with an applied set gives exactly the origins to catch up on |
| build identity | `count` is the merge base's count with `origin/main`, so a local commit leaves it unchanged; `local` is empty only at the merge base, and changes with a local commit, an edit or an untracked file; the build script reruns on a source edit and not on a `target/` change |
| local builds | a local build's own listener serves locally whether joined or not (`member/listener.rs`); it never sets the floor, is never a candidate, never installs or fetches automatically, and is skipped by a rollout's default targets and never closed 4410 for being behind (`decide.rs`, `upgrades/tests.rs`, `plan/tests.rs`); otherwise it is routed to, takes leases, replicates its writes and votes like any member |
| certification (`gateway/upgrades/{tests,stats,mirror,verdict}.rs`, `upgrade/runtime/candidate.rs`, `web/src/cluster/mirror.rs`, `host/src/guard.rs`, `tests/builds.rs`) | a loading candidate takes no share and its grace has not started; the window opens when it is ready; the live share (every tenth pick) and the mirror cap (a fifth) hold to their percentages; samples and mismatches are kept per route; `Exact` ignores the declared volatile fields only, `Shape` catches a changed structure and not a changed value, and a different status differs in every mode; a route declares its mirror mode beside its dispatch and a write's is `None`; only an answer whose length is declared and within the cap is buffered, and a 5xx or the gateway's own failure counts as an error; on generated samples the error test passes equal rates and fails a worse candidate, the latency ratios hold at their bounds, a route under its minimum is a gap, and the build fails on any failing route and otherwise passes with gaps; quantiles are nearest-rank; the verdict goes at the deadline, at once on a failure, and again every 10 s until the group decides; the candidate proposes once every socket has answered, at once on a failure, naming the failing routes. In process: a certified build's verdict is decided as `certify/zend/12-h12` with `Pass` by `certification`, and a failing one puts the candidate in fault mode with the failing route as the reason and raises nothing |
| distribution (`upgrade/{artifacts,parts,ui_archive}.rs`, `upgrade/runtime/{fetch,served}.rs`, `gateway/upgrades/held.rs`, `tests/builds.rs`) | only sibling binaries reporting the same build are offered, with the UI as one archive for any platform; the UI tree packs to exact bytes, unpacks to the tree it came from, and refuses a path out of its directory; a member fetches only its own kind and platform; a chunk is at most 1 MiB from its offset; a download resumes from its `.part` and verifies, a mismatch discards and restarts, and a rejected build's downloads are removed; the running build, a kept build and a verified download are each served; the gateway tracks holders per artifact and forgets a member that leaves. In process: the older members fetch the candidate's binary while it certifies, and the verified file is the candidate's |
| install (`upgrade/{swap,watchdog,handoff,os}.rs`, `cluster/src/launch.rs`, `upgrade/runtime/{install,decide,refused}.rs`, `host/src/cluster/upgrade.rs`, `agent/src/install.rs`, `tests/builds.rs`) | a fetched build takes the executable's and UI's paths and the old one is kept, a restore puts the old build back, a rollback swaps with a kept build, and a source missing the served UI moves nothing; the watchdog is the old executable run with its plan, and the plan round-trips through its file; under a fake process table, a ready build stands, launched only once the old process is gone; a build that exits is restored, its failure written with its last 40 log lines, and the old build relaunched with `--upgrade-failed <count>-<hash>`; one not ready in time is stopped and restored; and an old daemon that hangs is stopped first; `LaunchSpec` captures into the same command, drops the one-shot flags, and takes extra arguments and another executable; a refused build is remembered; an install names its leases by app. In process: older members install one at a time under `upgrade/zend`, the second waiting while the lease is held through the first one's restart |
| kept builds (`upgrade/kept.rs`) | kept builds list newest retired first; pruning keeps the newest *N* and every protected build; removing named builds refuses one not kept and one a rollout targets, naming it; the builds directory is beside the executable |
| build compatibility (`upgrade/compat.rs`, `version/info.rs`, `host/src/build_info.rs`, `persistence/held_types.rs`) | a build that opens everything fits, and each mismatch — log format, a record type outside its tags, another model — says why in its own words; the `--build-info` answer is exact JSON, an agent's opens nothing, output that is not an answer is none, and only the flag alone asks; the host's answer carries its log format, every record tag it writes and the preset; the substrate's census holds every type found at open and every type appended since |
| forced rollout (`gateway/upgrades/plan/tests.rs`, `gateway/upgrades/tests.rs`, `upgrade/runtime/{decide,commands}.rs`, `tests/builds.rs`) | a downgrade is planned member by member, each target's source its own kept copy or a fetch with its bytes, with skipped and stranded members and the builds it rejects; a downgrade across a new record type is stranded everywhere; a local build, a source lacking the build, no reachable quorum and `parallel` beyond ready members minus the quorum are each refused; a decided rollout moves the floor once, either way, cancels another candidate and marks what it rejects; a rollout naming a member installs its build down as well as up, over a local build and a refused one, under one of its leases, the latest rollout winning. In process: a downgrade's plan fetches nothing, the floor falls at once, a member installs from its own `builds/` under `upgrade/zend/0` with the build protected, the others wait, and the gateway learns the build above rejected |
| UI pinning | a page's HTML sets `tokera_ui_<app>`; every UI request with it reaches the same member; a pinned member that leaves or faults is replaced and the cookie rewritten; API requests ignore it |
| dispatch | the route table and requests yield the expected member for each class; the `x-tokera-node` override is admin-only; the `Body` key reads ≤ 8 MiB |
| modes | a route with `cluster:` and no ready member goes to its configured URL; the first ready member switches it; requests in flight to the URL finish there; the last member leaving reverts it; a route without `cluster:` never switches |
| claims | a new key is claimed by the member that serves it; the claim is announced on every cluster; a gateway lacking the holder answers `409` then `503`; the idle release carries `through`; the catch-up hold releases at `through`; of two simultaneous first claims the higher epoch keeps the key |
| lagging | an ordered lane past its cap makes the member lagging: no further live events are queued for it and none of its queued stream frames are dropped; it answers no catch-up and takes no round-robin request, but a new conversation, or a released one whose last holder's records it holds, is still placed on it (`registry/tests.rs` `a_lagging_member_is_placed_by_conversation`); once its lane drains it takes live events again and syncs, routed again once in sync — an origin still writing included; its socket is never closed for being slow |
| socket drop | an exchange with no response head yet gets a `503`; a body in progress is aborted rather than ended; a tunnelled WebSocket closes `1011`; nothing is replayed on another member |
| emission | a request's events leave on the request's socket; other events go round robin across sockets, one whole event at a time; with every socket down events stay *unsent*, never queued in memory |
| retransmit | events in flight on a socket that drops, and only those past its last `RECEIPT`, go back to unsent and leave again in `oseq` order on the next socket admitted; nothing is re-sent that a `RECEIPT` covered |
| arrival order | a member's log in arrival order reopens to the same state it held before the restart; an older keyed record arriving late, from the same origin or another, never replaces a value with a greater stamp |
| catch-up | the gateway picks the answerer round robin among the other ready members, or the member `to` names; `HEADERS` pack to exact bytes; an answerer with a hole in its own applied set ends its range at the hole and never lists past it; only `oseq`s a header names as gone are recorded as gone, and an `oseq` outside every complete range stays wanted; an origin's `high_water` above what a member holds for 5 s starts a catch-up addressed to that origin, which finds a lost last event; two catch-ups of one member never fetch one record; fetches stay within an 8 MiB window; a `FETCH` ending short is fetched again; replies reach only `reply_to` and no other member sees any catch-up frame; `applied_through` persists in `Replica` and a catch-up cut off part-way resumes from it; a socket reconnect, admission, and every 30 s with a socket down each start one |
| full resync | only the first admission after a process start runs it; it catches up first, then sends `HAVE`; every member holding less runs an addressed catch-up against it; afterwards its own events are classed received or unsent correctly, against the *highest* `applied_through` any member acknowledged, so one lagging member causes no re-broadcast; a reconnect in a running process never sends `HAVE` |
| clusters | four sockets for two apps × two clusters each handshake, keep alive and reconnect independently; a 4410 on one cluster closes that app on all of them; ready on the first cluster caught up |
| own listener | joined: a request on the listener is forwarded up, carrying the identity the listener resolved, and answered from wherever the cluster routes it; a request arriving on a socket is never forwarded; `/v1/status` always answers locally; cut off: served locally; rejoining, reloading or upgrading: `503` with `Retry-After`; a request marked `x-tokera-hop: dumb` from an allowlisted gateway is served locally in every state, and the mark from any other peer is ignored; a daemon started with no reachable cluster serves locally once every cluster has failed one attempt or after 10 s, and rejoins on its first ready socket; the switch flips per app on the first ready socket and the last drop; a request in flight across the switch finishes where it started or fails as §5.6 says |
| turn order | two writers' turn *N* never overwrite each other in the `(stream_id, origin)` index; a timeline's turns are its committed decls in stamp order, in any arrival order; `turn_stream_id(timeline, N)` resolves to the *N*-th of them; a single writer's turns keep the order it wrote them in; a partitioned former holder's turns sort before its successor's |
| generations | a cut-off member's events carry ⊤ and sort after everything it holds; on rejoin it catches up before sending anything detached; it takes `gen/<k>` by Paxos for *k* one above the greatest decided it knows, and moves to *k + 1* when *k* is already decided; decided generations have no gaps; a joined member's event written before the decision, including one the rejoining member never caught up on, sorts before its detached events on every member; the generation is persisted as a local `Rebase` that is never replicated, and applied on reading its own records back; without a quorum the member stays cut off, serving locally and sending nothing detached, and takes its generation once a quorum is reachable; two members cut off together sort in the order their generations were decided; offline keyed writes win over the group's from the same period; a tombstone still kills offline writes to its timeline; a turn arriving ranked below one held makes the member drain, reload and come back |
| consensus (`cluster` crate, Paxos among members) | a proposer `NACK`ed waits a random time within the key's `ttl` before trying a higher ballot, and two duelling proposers end with one decision; an acceptor promises only above its highest promised ballot and `NACK`s the rest with that ballot; a promise and an acceptance are durable in the substrate before they are sent, and survive a restart; a proposer proposes the highest-ballot accepted value it was shown; under every interleaving of two and three proposers against three and five member-acceptors, with dropped and duplicated frames and frames arriving through two clusters, a decision never changes and two lease holders never overlap; a decided value is written as a replicated record |
| leases | `PROMISE` carries an accepted lease only while its `ttl` runs; another member's unexpired lease stops a proposer, its own does not (renewal); under `start_paused`, the holder's term, counted from `PREPARE`, ends before every acceptor's; a lease accepted before an acceptor restarts counts for a full `ttl` after; `RELEASE` drops it and the next proposer waits for `through`; a partitioned authority stops persisting before its lease runs out; a member alone in a group of three can never hold one |
| gateway peering (`cluster/src/peer/`, `cluster/src/frame/tests.rs`, `web/src/acme/{ledger,certs}.rs`, `web/src/acme/peers/inventory.rs`, `web/src/cluster/peers.rs`, `web/src/config.rs`, `web/tests/peering.rs`) | every peer frame encodes to exact bytes; two gateways link and carry frames; three in a mesh name the same orderer for every site, and a gateway alone orders everything; crossed dials keep the lower node's link on both ends, and it stays; a differing session key or OAuth client refuses the peering on both ends, each recording why, and a node other than the handshake's or the gateway's own is refused; a closed link is redialled; ledgers merge to the union in either order, and a save keeps what a peer merged meanwhile; a peer is sent the certificates it lacks or holds older, and the later `not_after` wins; the node id is minted once and kept; peers need the TLS entrance. Against a local stand-in for the CA, at the `Ca` trait — it publishes through the gateways' own path and asks every gateway it could reach for each answer: three linked gateways name one orderer and only it may order, its challenges are answered on all three before the CA validates, and its certificate is installed on all three with every challenge withdrawn; a peer's `ACME_ORDERING` holds the site until `ACME_DONE`; two gateways partitioned each order, and once linked both serve the later certificate; over a link that is not verified TLS the challenges cross but no certificate key and no account do; the account is given to a peer that has none, and the ledgers merge on connect and on change |
| commands | each command of §12.4 runs on an in-process member and its effect is asserted; `drain` hands back every claim and lease and the gateway routes nothing new; `restart` waits for in-flight work and the outbox, then builds a relaunch with the original arguments, one-shot flags dropped (the process spawn is asserted on, not run); `compact` is refused in fault mode and when free space is below the live size; a destructive command without the matching `confirm` is refused; a second command while one runs is refused; every command lands in the coordination log; `drain-gateway` closes members with `1001` and they carry on through their other cluster; a delivered Ctrl-C on a joined member drains, waits for the outbox, flushes, closes `1001` and then runs the tenant shutdown, with another member taking its world at once rather than after a term; a second Ctrl-C mid-drain exits at once; without `--cluster`, Ctrl-C is unchanged |
| fault mode | a simulated `ENOSPC` on append, and free space under the floor, each enter fault mode; the member stays connected and keeps sending `COORD`; it releases every claim and world lease, stops its characters, and proposes nothing; **with the volume completely full it still promises and accepts, durably, through the vote journal, and counts toward the quorum**; it keeps sending its unsent events and answering `FETCH`es; the gateway routes no request to it and relays no live events to it, but relays Paxos frames to it; only `Drop` maintenance runs; leaving needs twice the floor and a test append, then catches up before taking work; free space hovering at the floor does not flap the mode; the status page and the log show both transitions |
| quorum | `cluster/max_members` rises when more members are seen together, never falls by itself; a *set* lowers it, and raises after the set raise it again, in any arrival order; `--cluster-quorum-max` and the admin call write a set; the quorum is `⌊max_members / 2⌋ + 1`; a freshly restarted member alone reads its quorum from its substrate; in a group of two, one member plus the voting gateway's tie-break promise and acceptance decides, and the same vote does not complete a larger quorum |
| NPC lease | the authority lends and renews `world/<wid>/npclease/<nid>`, spread by weighted rendezvous across members from the first lending; a membership change moves a character only at its turn boundary, after its holder hands it back; a holder that drops out keeps its characters, which are never lent again until it rejoins; the admin release stamps above the holder |
| world state | only the authority writes `world/<wid>/…` records, at most once per entity per beat; a ceased entity is tombstoned; `PULSE` is never persisted, never retransmitted, and dropped under pressure; a replica that loses every pulse still ends on the same state records |
| disconnected mode | a cut-off member's characters keep acting on each other in a local world; no world-state record is written while cut off; the characters' turns are written as detached events; on reconnect the local world is scrapped and reloaded from state records; an operation finished in the local world publishes through the member's in-process agent, and without one waits 30 minutes and tries again; a publish landed while disconnected is kept on reconnect, and the authoritative world's copy of the operation carries on |
| rendezvous | ownership is deterministic, minimal movement on joins and leaves, and the weighted share tends to the weights |
| **order independence** (`candle-conversation`, CPU substrate) | **the central test.** Generated event sets cover every record type, two and three origins, competing writes to one key, tombstones before and after their targets, children before parents, section generations ended and restarted, two writers' turns at one index, and events with fixed generations from a cut-off member. **Every permutation** of each set (exhaustive up to 7 events, 5,000 seeded random shuffles beyond) is applied to a fresh substrate, and the resulting state — every index, value, stamp, tombstone, lease and turn order — is byte-identical across all of them. The same holds after a compaction and a reopen part-way through. |
| apply (`candle-conversation`, CPU substrate) | a record of an unknown type is neither written nor marked applied, goes in the deferred set, and is not asked for by a catch-up; a build that knows the type moves it back to wanted on open, and its first catch-up fetches it; duplicates are not written; dominated records are not written, but for a turn's declaration, which is written dead and dominates its stream's later records; a record ranked below the value held is skipped, whatever its `digest`; a record failing its `digest` is refused; a turn is adopted only when complete, in every order of arrival, and an incomplete remote turn is pending, never `Corrupt`, at runtime and at open; repair tombstones carry no identity and are never replicated; placeholders fill gaps, and a turn completing later takes its gap without a reload; a turn ranked below a held turn flags a reload; a child record waits for its parent and is applied when it comes; `Commit` takes the max; a timeline tombstone survives compaction and kills a record arriving after it |
| tombstone expiry | a tombstone record is kept by every compaction until a complete `compact()` has finished after it **and** its `hlc` is more than *N* days old, and is then replaced by its entry in the `Deleted` record; neither condition alone shrinks it; with paused time, a record for a deleted timeline arriving a year after expiry dies in every arrival order, and a register of that timeline leaves it dead; a keyed entry kills lower-stamped writes and yields to a greater stamp; a catch-up's gone header for an expired tombstone carries its entry, and a member returning after any absence drops the timelines it names and its unsent writes to them die everywhere; `Deleted` survives compaction and reopen byte-exact |
| identity | the `TimelineId` layout and monotonicity; `hlc` is monotonic and passes every `hlc` seen; `(origin, oseq)`, stamp and `digest` preserved through relocation, re-emission and `compact()`; the applied set and the location index rebuild from the last `Replica` plus the walk; a substrate copied to another directory or machine re-mints its node; passthrough timelines are `hash(conversation_key, n)`, `n` the first opening whose timeline is free; VFS events are named `(origin, seq)`, two writers' of one number are two events, and a tombstone of another member's event outlives compaction |
| VFS scoping | the writer's, reviewer's and checker's missions of one operation see one store, and a draft written in one step is read in the next, on another in-process member too; two operations over one base see different files; a character with no mission reads the tip; the table's reading and checks read the store; nothing an operation writes reaches the mind folder before its publish step; a failed run's store is discarded under `discard` and kept under `keep`, and `reopen` resumes a kept one; a workflow with a route to `done` from anything but a publish step is a load error; a clean publish lands the store and settles the run `done`; a conflict refuses the publish with the paths named, `merge_sync` marks the overlapping hunks in the store, and once the Maker has resolved and committed them at the bench the publish lands and the run settles `done` without going back through verification; the `published` check holds the step open until then; a Maker that reports `stuck` hands the step to another; publishing the same text twice merges as a no-op; in disconnected mode a step's edits write no `VfsEvent`; with `origin` or every agent unreachable, a publish settles nothing, counts toward neither send-backs nor stuck, and under paused time is offered again exactly 30 minutes later with the store intact |
| agent selection | stickiness, the spill after the wait, and the idle-core ranking with repository and toolchain filters |
| fetch through an agent (`zend-vfs/src/remote/fetch_bundle.rs`, `agent/src/wire.rs`, `agent/tests/agents.rs`, `npcd/src/mind_repo.rs`, `npcd/src/tenant.rs`) | a member that cannot reach origin gets a branch through an agent's checkout — the commit and its objects under its own tracking ref, reported as a fetch from origin reports it — then only what it lacks, and nothing when it lacks nothing; every branch is carried and a deleted one pruned; a branch asked for alone that origin lacks is an unknown revision; a `have` the agent does not hold is passed over; a fetch job and its answer are their exact JSON; through an in-process gateway a mirror whose `origin` is `agent:r` pulls origin's branch through the agent, and cut off a member's fetch reaches only its own agent, else `no_agent`; npcd's mirror is made under `--data` named `agent:<name>`, brought up to the record and checked out — ready — and follows later commits; a branch origin lacks leaves it not ready; `--mind-repo` needs `--data`, is refused beside `--mind`, and reads the same from `motherd.yaml`; an unready mirror starts on no mind |
| status page | `COORD` encodes to exact bytes; every coordination decision of §12.3 — admission, refusal, mode switch, floor, lock, claim, grant, majority, NPC lease, cut-off, rejoin step, generation fixed, disconnected mode, catch-up, resync, tombstone expiry — driven through an in-process gateway lands in the log with its gateway and subject; `status.json` carries every block of §12.2, and a member reporting a gateway this one cannot see shows in both the `clusters` column and the grant counts |

### 14.2 Burst micro harnesses

Each harness is a plain `#[test]` with a wall-clock budget. Like
`candle-transformers/tests/prefill_replay.rs:103,260-294`, it loops until its sample
count or its time budget is reached. Each finishes within 1–4 s, all of them together in
under 20 s. Each prints its measured rate and asserts both correctness and a floor set
with headroom from its first measurement on this machine.

| # | Harness | Load | Asserts | Budget |
|---|---|---|---|---|
| H1 | codec | encode and decode 2 M mixed frames | byte-exact round trip; ≥ the frames/s floor | 1 s |
| H2 | multiplexer | 1,000 concurrent virtual streams × 64 KiB over one in-memory socket pair, with a 63 MB event interleaved | every stream intact; no stream's frame waits behind more than one 1 MiB fragment; control-frame latency bound | 3 s |
| H3 | relay fan-out | in-process gateway, 4 members, 50 k events | every member gets every other member's events exactly once, none echoed, per-origin order kept; ≥ the events/s floor | 3 s |
| H4 | recovery under load | live traffic running throughout: a member's socket dropped mid-burst with 5 k events in flight, then reconnected; a member offline while 100 k records across 3 origins go by, some since compacted, then back; a restarted member holding 10 k records nobody else has; one slow receiver | every member ends holding the same applied set and the same substrate; the dropped socket's in-flight events arrive again and nothing receipted is re-sent; the returning member fetches exactly the records it lacked, by header compare, and no other member receives a catch-up frame; killing the returning member mid-catch-up and restarting it resumes from its persisted `Replica`; live-event p99 latency stays under its bound while catch-up runs; the catch-up queue never passes its cap | 4 s |
| H5 | handshake storm | 500 handshakes: valid, wrong key, low build, wrong model, duplicate node | exact codes; zero non-handshake frames before `ADMIT`; floor and lock state correct after | 2 s |
| H6 | routing | 100 k dispatch decisions with capacity churn, plus an 8-member aggregate with one member past its deadline | decisions match a reference model; the aggregate answers in deadline + ε with the slow member marked | 1 s |
| H7 | consensus | 10 k worlds and 1 k decisions across five in-process members and two gateways carrying the frames; 50 membership changes; one member restarted mid-run and one partitioned into a minority; 100 k Paxos rounds with competing proposers and dropped, duplicated and doubly delivered frames | rendezvous moves only keys whose winner changed, and shares stay within tolerance of the weights; no world ever has two holders at any instant; no decision ever changes; the minority holds nothing; every key gets a holder or a decision once its proposers stop competing | 2 s |
| H8 | active-active and cut off | two in-process gateways, three members joined to both, 20 k events and a stream of sticky requests through both gateways and through members' own listeners; one gateway killed mid-run, then restarted; one member cut off from both, serving its listener locally and writing to a conversation another member also writes, then reconnected | no event lost; duplicates skipped; every member ends with the same applied set and the same substrate, through retransmit and catch-up alone with no full resync; requests in flight on the killed gateway fail as §5.6 says, and every later request succeeds through the other gateway; claims survive; the restarted gateway re-learns members and claims and shares load again; the cut-off member serves every listener request locally, then on reconnect catches up first, sends what it wrote with its fixed generation, and forwards up again; on every member the shared conversation holds the group's turns from the cut-off period followed by the cut-off member's, in the same order | 3 s |

The total is 19 s.

---

## 15. Build order

**This section is the anchor for building the design.** The work is ten chunks, in
dependency order:

- **Foundation first.** Chunks 1–3 depend on nothing else in this design, touch nothing
  that serves traffic, and can be built in parallel. Each is proven by CPU-only tests on
  its own before anything is built on it.
- **Every later chunk builds only on chunks already finished**, and ends at an exit bar
  that is a test result, not a judgement.
- **Nothing is half-built.** A chunk lands complete, or it does not land (`CLAUDE.md`).

| Chunk | Builds on | Deliverable |
|---|---|---|
| 1. Substrate identity and order-independent apply | — | every log replays the same in any order; two substrates converge in-process |
| 2. The wire protocol | — | the `cluster` crate's frames, multiplexer and handshake |
| 3. Consensus | — | Paxos leases and decisions over an abstract transport |
| 4. Members behind the gateway, without replication | 2 | **the first deployable milestone** |
| 5. Live replication | 1, 3, 4 | two members sharing conversations |
| 6. Operating the cluster | 5 | fault mode, maintenance, drain, the full status page |
| 7. npcd worlds | 3, 5 | worlds and characters across members |
| 8. Agents and file state | 1 (operation stores), 4 (agents over the cluster) | agents, VFS uploads, npcd operation stores and publishing |
| 9. Builds | 3, 4 | certification, distribution, install, rollouts |
| 10. The second gateway and the migration | everything | active-active gateways; today's deployment moved over |

### Chunk 1: substrate identity and order-independent apply

**In `candle-conversation` only.** No socket, no gateway, no host change. This chunk is
first because everything that replicates rests on it, and because it is the riskiest:
it changes how *every* log is replayed, today's single-machine ones included.

- **Identity (§6.2, §3.7).**
  - The header's `origin`, `oseq`, `gen`, `hlc`, `epoch` and `digest`, preserved
    through relocation, re-emission and `compact()`.
  - `Provenance` on `append_record`.
  - `substrate/NODE`, with re-minting on a copied directory.
  - `Replica` (tag 28), with the applied set and the `(origin, oseq)` location index.
  - The `TimelineId` format, passthrough timelines as `hash(conversation_key)`, and VFS
    keys carrying the origin.
- **Apply (§6.5).** Every apply path rewritten to the rules:
  - stamped maxes, with `Label` merged per field and `Commit` as a max;
  - the pending-parent map;
  - dominating tombstones;
  - section generations;
  - the stream index keyed `(stream_id, origin)`, with a timeline's turns in stamp order
    and `turn_stream_id` lookups resolved to the turn at that index. The 34 call sites
    are untouched;
  - the deferred set for record types the build does not decode.
- **Tombstone expiry (§6.8).** The `Deleted` record (tag 31) with its timeline entries,
  and expiry after a complete compaction and *N* days (`--tombstone-retention`, zend and
  npcd). The keyed-value entry lands in chunk 7 with the keyed tombstones it shrinks.
- **Remote records (§6.5).** Applying a record from another origin under the substrate
  write lock:
  - `adopt_remote_turn` for complete turns only, with incomplete remote turns pending,
    never `Corrupt`;
  - repair tombstones kept local;
  - the duplicate, dominated and superseded skips.
- **Generations, locally (§6.7).** Detached events at ⊤ and the local `Rebase` (tag 30).
  The generation number itself comes through a `MintGeneration` trait. Chunk 5 plugs
  chunk 3's decision into it; here the tests use an in-process minter.
- **The one-time origin compaction** that adopts a substrate written before this design
  (§6.2).

**Tests.**
- The order-independence permutation test, and the apply, identity and
  tombstone-expiry tables (§14.1).
- **A two-substrate convergence harness.** Two substrates in one process write
  conversations, sections, tombstones and VFS events. Every record of each is shipped to
  the other in a random order, with duplicates and a reopen part-way through. The test
  asserts the two end byte-identical in every index, value, stamp and turn order.

**Exit.**
- The permutation test and the convergence harness pass.
- **Every existing `candle-conversation`, zend and npcd test is still green**, and the
  model gates and engine probes pass under `/sweep` — the apply rewrite runs on every
  log at open.
- A copy of the live zend substrate opens through the one-time origin compaction and
  reopens to the same turns, labels and conversations.

### Chunk 2: the wire protocol

**The new `cluster` crate**, with no CUDA and nothing wired into the gateway or a host.

- The frame codec, every kind of §4.1.
- The three-lane writer, with per-stream credit and `WINDOW` (§4.2).
- Virtual connections as `AsyncRead + AsyncWrite`, with `OPEN`, `DATA`, `FIN` and
  `RESET`.
- The handshake, with HMAC proofs and the direction labels (§3.3).
- Reconnect backoff (§3.2).
- The frame counter and `RECEIPT` tracking, and the unsent / in-flight / received states
  of §6.3, as data structures.
- Applied-set packing for `CATCHUP` and `HAVE` (§7.5).
- `EVENT` fragmentation and reassembly.
- The build version: `count`, `hash` and `local`, from its `build.rs` (§3.5).

**Tests.**
- Unit tests asserting exact bytes for every frame, the proofs, the backoff schedule and
  the packing.
- H1 (codec) and H2 (multiplexer), and the handshake half of H5, over in-memory pipes.

**Exit.** Every frame round-trips byte-exact. A truncated or unknown frame is refused.
1,000 streams and a 63 MB event share one pipe within H2's bounds.

### Chunk 3: consensus

**Paxos in the `cluster` crate (§5.9)**, over an abstract transport, with no gateway.

- Leases and one-time decisions, with durable acceptors in the preallocated vote journal
  (`VOTES`, written in place to alternating halves).
- Lease terms counted on the monotonic clock, and the restart rule.
- `max_members` with automatic raises and stamped sets, and the quorum.
- The randomised backoff after a `NACK`.
- Rendezvous (§9.5).

**Tests.**
- The consensus, lease, quorum and rendezvous tables (§14.1).
- H7, against a simulated network that drops, duplicates, reorders and partitions
  frames, with members restarting mid-round.

**Exit.** Across every interleaving the tests drive, no decision ever changes, no two
lease holders overlap, and a minority decides nothing. A vote survives a torn write.

### Chunk 4: members behind the gateway, without replication

**Builds on chunk 2.** The gateway and the hosts learn to carry API traffic over the
socket. Nothing replicates yet.

- **Gateway (§3.1–§3.6, §5).**
  - The `/v1/cluster/:app` endpoint and admission: proofs, the build floor on `count`
    alone (§3.5), the model lock, the clock-skew check, and one session per node.
  - The `cluster:` key on routes, dumb and clustered modes, and the `x-tokera-hop: dumb`
    mark.
  - The transport swap in `proxy::forward`, dispatch on every route, claims, aggregates
    and UI pinning.
  - Failing in-flight exchanges when a socket drops (§5.6).
- **Members (§3.2, §5.1, §5.8).**
  - `--cluster`, `--cluster-name` and `--group-key`, with one socket per app per
    cluster.
  - `ClusterListener` into `host::server`, and `ClusterPeer` trust.
  - `Dispatch` on every route, and `CAPACITY`.
  - The member's own listener: forwarding up when joined, serving locally when cut off or
    when the dumb-mode mark is set, and starting cut off when no cluster answers.
  - The `connecting` loading step and the `cluster` status block.
- **A first status page (§12).** The Apps and Members blocks, Refusals and Traffic,
  and the coordination log.

**Tests.** The close-code, backoff, model-lock, UI-pinning, dispatch, mode, claim,
socket-drop and own-listener tables (§14.1); H5 and H6.

**Exit, and the first deployment.** One member per app behind the gateway behaves
exactly as today's fixed upstream, and a second member of an app takes routed traffic.
That configuration can go live and be watched before anything replicates.

### Chunk 5: live replication

**Builds on chunks 1, 3 and 4.**

- **Emission (§6.3, §6.4).** The tap, round-robin and request-bound emission,
  `WriteTicket`, and reading owed events back through the location index.
- **The relay (§4.3).** `RECEIPT`s, retransmit on a dropped socket, and the lagging state
  (§4.4).
- **Apply on arrival**, through chunk 1.
- **Catch-up (§7).** Headers first with coverage-honest ranges, the compare against
  what is held and what another catch-up is fetching, windowed fetches, `reply_to`
  addressing, `high_water` gap detection, deleted-set entries in gone headers, and the
  full resync after a restart.
- **The `catchup` loading step**, and readiness gating clustered mode (§7.2, §7.6).
- **Cut-off and rejoin (§6.7).** The generation comes from chunk 3's `gen/<k>` decision.
  The rejoin waits for a quorum. A turn arriving ranked below one held makes the member
  drain and reload (§6.6).

**Tests.** The emission, retransmit, catch-up, full-resync, cluster, generation and
lagging tables (§14.1); H3, H4 and H8.

**Exit.** Two zend machines behind one gateway share conversations: either serves any of
them, a conversation moves between them with its K/V intact, and a machine cut off and
reconnected ends byte-identical to the other.

### Chunk 6: operating the cluster

**Builds on chunk 5.**

- Fault mode for a full disk, with voting from the vote journal (§7.7).
- The maintenance commands and their job stream (§12.4), with zend's maintenance route
  moved into the host.
- Ctrl-C draining first.
- `COORD` and every status-page block whose state exists by now (§12.1–§12.3); the
  Upgrades, World leases, NPC leases, Disconnected worlds, Agents and Peers blocks
  arrive with the chunks that build what they show.
- The quorum admin call, and `--cluster-quorum-max`.

**Tests.** The fault-mode, command, quorum and status-page tables (§14.1):
`cluster/tests/operation.rs` (fault mode and its hysteresis, a faulted member's vote, a
member's drain, `COORD` on the page, the quorum call, a gateway's drain),
`host/tests/cluster_member.rs` (commands through a real member and the log, leaving
before a stop), `host/tests/replication_store.rs` (a real store's disk and state), and
the unit tests of `coord_log`, `paxos_log`, `commands::{parse, jobs}` and `stopping`.

**Exit.** A member whose disk fills stays connected and voting and recovers without
help. Every command runs through the gateway and on a member's own listener alike.

### Chunk 7: npcd worlds

**Builds on chunks 3 and 5.**

- World state as per-entity state records in place of the three whole-world
  CustomObjects, and `PULSE` for transients (§9.1).
- The keyed-value tombstone of a ceased entity, and its `(key hash, stamp, origin, oseq)`
  entry in the `Deleted` record when it expires (§6.8).
- The world lease, and the authority's term handling (§9.2).
- NPC leases spread by rendezvous, and acts forwarded to the authority (§9.3).
- Disconnected mode (§9.4).
- Handover and routing (§9.6).

- The cast followed across members, every member embodying every character (§9.3).
- The World leases, NPC leases and Disconnected worlds blocks, from `COORD`'s `worlds`
  and `npc_leases` (§12.1, §12.2).
- The one-time conversion of a world kept before world records (§15.1, step 3).

**Tests.** The world-state, NPC-lease and disconnected-mode tables (§14.1), with npcd
running against two and three in-process members behind the web gateway's own router
(`npcd/tests/worlds.rs`): one authority and its cast spread, an act made on a replica
by the authority and heard back as a pulse, a message told to a character on another
member; a world and its characters moving to the member rendezvous names, handed back
and waited for; a cut-off member running its world locally, writing nothing of it, its
characters not lent again, and scrapping that world on return; and a world keeping one
authority across the loss of its authority, a watcher asserting no two members are ever
its authority at once. The lease tests in `cluster/tests/leases.rs` (one holder, the
release and its `through`, a member arriving before the quorum rises, a lease taken
alone made known to the group it grows into), and the unit tests of
`npcd::world_state::*`, `npcd::authority::*`, `npc_map::world` (witnessing, restored
places and decisions) and `cluster::gateway::{world_log, worlds_view}`.

**Exit.** A world keeps one authority across a member's loss and a partition, its
characters run on the machines rendezvous lends them to, and a disconnected world
rejoins without corrupting the authoritative one.

### Chunk 8: agents and file state

**Agents build on chunk 4.** **The npcd operation stores need only chunk 1**, because
they work on one machine with no cluster at all, so they can be brought forward when the
npcd work wants them sooner.

- **Agents (§11).**
  - `tokera-agent` and the job API.
  - `motherd --agent`, and the in-process agent a cut-off member reaches.
  - zend's executor and git writers going through agents under `--cluster`.
- **zend (§10.1, §10.2).** Uploads as VFS events.
- **npcd (§10.3).**
  - Operation stores read and written by every step's mission and by the table's checks,
    and the bench committing into them.
  - The removal of `held_drafts`, `Operation::before` and the on-disk set-aside and
    restore, with `on-failed` taking `discard` or `keep`.
  - The workflow format's step `desk`, the `publish` step with its `published` check,
    and the load rule that only a publish step leads to `done`.
  - The `vault-archive` floor and merge table, with `merge_status`, `merge_publish` and
    `merge_sync`, and conflicts solved at the bench.
  - `retry_at` and the 30-minute retry of an unreachable publish, including from a
    disconnected world.
  - Authoring stores and Publish, accounts as records, and images as `Blob` records.
- **§10.4.** Ingest once, deduplicated by unit.

**Tests.** The VFS-scoping and agent-selection tables (§14.1), with a local git
repository standing in for `origin`.

**Exit.** An npcd story written, read, reviewed and checked reaches the mind repository
only through its publish step, and a conflict is solved at the bench. A zend tool runs on
an agent.

### Chunk 9: builds

**Builds on chunks 3 and 4.** Chunk 2 already stamps every binary with its version, and
chunk 4 already enforces the floor. This chunk is the machinery on top:

- `--build-info` with its compatibility fields (§3.9): `cluster::version::info`,
  answered first by every main (`host::build_info::first`), and the substrate's census
  of the record types it holds (`persistence/held_types.rs`, `Store::census`).
- The loading gate on the grace period.
- Certification at the gateway: the live share, the mirror modes on every route, the
  statistics, and `VERDICT` with the candidate's
  `certify/<app>/<count>-<hash>/<i>` decision (§3.8) — `cluster::gateway::upgrades`,
  `web/src/cluster/mirror.rs` and `upgrade/runtime/candidate.rs`; the gateway's
  `floor_grace_minutes` and `canary_share`.
- Distribution with resumable part files (`upgrade/parts.rs`, `upgrade/runtime/fetch.rs`)
  and the gateway's table of held builds.
- The one-at-a-time install under `upgrade/<app>`, the in-place swap, and the
  restore (`upgrade/{swap,watchdog,handoff,os}.rs`; the installers in
  `host/src/cluster/upgrade.rs` and `agent/src/install.rs`). The member starts the
  new executable and exits. `keep` restores the last ready executable when that
  process does not become ready.
  `LaunchSpec` moves from `self_heal` to `cluster::launch`, which both use.
- Kept builds in `builds/` (§3.9, `upgrade/kept.rs`).
- The upgrading state for members admitted below the floor (§3.5), and agents following
  the daemons' floor.
- Forced rollouts and rollback as `rollout/<app>/<n>` decisions (§3.11), the plan
  (`gateway/upgrades/plan.rs`), and `rejected` in `HELLO`.
- Local builds (§3.10).
- The Upgrades block, `POST /gateway/upgrade/<name>`, and the member commands of §12.4.

**Tests.** The build-floor, build-identity, local-build, certification, distribution,
install, kept-build, compatibility and forced-rollout tables (§14.1), and
`cluster/tests/builds.rs`: members of one process, each a machine of its own in a
temporary directory whose executable is a file holding its `--build-info` answer. The
process spawn is asserted on, never run.

**Exit.** A pushed build rebuilt on one machine certifies, reaches the others, and
installs itself one member at a time. A bad install restores itself, and a rollback swaps
from each member's own disk.

### Chunk 10: gateway peering, the mind through agents, and the migration

**Builds on everything.**

- Gateway peering (§13.4): the `gateway` app's links (`cluster::peer`), challenge
  sharing, certificate push over secure links only, the merged ledger, the inventory
  exchange and the rendezvous orderer with `ACME_ORDERING` (`web/src/acme/peers/`).
  `web --new-group-key <path>` writes a group key.
- npcd's mind as a repository the agents hold (§10.3): `--mind-repo`, its mirror under
  `--data`, and the fetch job (`Submit::Fetch`, `agent::fetcher::AgentFetch`,
  `zend_vfs::Fetch`) a member with no way to origin fetches through.
- The consoles reading the gateway's aggregates (§5.5, `common/lib/aggregate.js`).
- `docs/deployment.md`: the group key, the `cluster:` block and route keys in
  `web.yaml`, `motherd` with `clusters`, `group_key` and an in-process agent.
- The migration of today's deployment (§15.1).

**Tests.** The gateway-peering table (§14.1), against a local test double of the ACME
CA (`web/tests/peering.rs`); the mirror brought up and followed through an agent
(`npcd/src/mind_repo.rs`, `agent/tests/agents.rs`).

**Exit.** Two gateways can serve the same names with one certificate store between
them, and today's zend runs as a member of the gateway's cluster with its history
intact. A second gateway behind the public names waits on a second gateway machine.

### 15.1 Migrating today's deployment

**Where things stood** (`docs/deployment.md`, before 2026-10-12):

- the gateway, `web`, on `.5`;
- zend on `.5`, launched `D:\prog --host 192.168.0.5 --port 8081 --tls-port 8444`, with
  its substrate at `D:\prog\substrate`;
- npcd on `.6`, launched `--host 0.0.0.0 --port 8081 --gateway 192.168.0.5`, with its
  substrate under its data directory;
- `motherd` not deployed.

**As run, 2026-10-12.** `.5` moved first; `.6` stays as it is until it is rebuilt on
this framework.

- **The gateway** took the `cluster:` block (`group_key_file: secrets/group.key`,
  written by `web --new-group-key`) and `cluster:` keys on the zend and npcd routes, each
  keeping its upstream.
- **`motherd` on `.5` replaced zend.** It joins `https://192.168.0.5:9443` as
  `tokera.com` with an agent in process, whose checkouts — `candle`, `battle-cities`,
  `battle-mind` — are under `D:\agent` (`motherd/agent.yaml`). Its zend tenant adopted
  `D:\prog\substrate` as the group's first member. No copy was taken aside: the estate
  accepts rebuilding a substrate over keeping a way back.
- **npcd started fresh under `motherd`'s model key** — the second way on in step 3
  below. Its tenant has `mind_repo: battle-mind` and `data: ../npcd-data`: the first
  start found no mirror, ran with no mind, fetched the record through the in-process
  agent once joined, and relaunched onto it.
- **`bot.tokera.com` falls back to `.6`.** While `motherd`'s npcd member is joined and
  ready the gateway routes the name to it; with no npcd member it is back in dumb mode,
  on the `.6` upstream.

The steps below are the general plan, for the machines still to move.

**0. Prepare.**

- Build every machine at one commit, so the build floor sees one count (§3.5).
- Generate the group key (`web --new-group-key <path>`) and copy it to every machine
  (§13.1).
- **Copy aside each substrate directory whose history is to be kept.** The adoption in
  steps 2 and 3 rewrites it, and this design has no backup (D17), so this copy is the
  only way back.

**1. Gateway, in dumb mode.**

- Deploy the gateway with the `cluster:` block and `cluster:` keys on the zend and npcd
  routes, keeping their upstream URLs.
- Nothing changes for users: with no members, every app is in dumb mode (§5.7).
- Check the status page shows both apps dumb, with the configured upstreams healthy.

**2. zend: `.5` adopts, the others join empty.**

- Restart zend on `.5` with `--cluster` and `--group-key`. As the group's first member,
  it adopts `D:\prog\substrate`: the one-time origin compaction (§6.2) rewrites every
  record with `.5`'s origin.
- During that adoption it also does two one-time conversions:
  - its upload files in `substrate/conv-files/` become VFS events (§10.2);
  - nothing else in zend needs converting.
- It is on its loading page while this runs. When it is ready, the gateway switches
  zend to clustered mode.
- **Before this step**, start at least one agent, or give `.5` `--agent`: with
  `--cluster`, zend runs `run_command` and the git writers only on agents (§11.5).
- Every other zend machine then starts with an **empty** substrate directory and
  `--cluster`. It catches up the whole log from `.5` (§7.3). It must not start from a
  copy of `.5`'s substrate (§6.2): a copy would hold the history twice, under two
  origins.

**3. npcd: `.6` adopts, the others join empty.**

- Restart npcd on `.6` with `--cluster`. It adopts its substrate the same way.
- Its adoption converts the world state once (§9.1): the three whole-world
  CustomObjects (`missions/<world>`, `benches/<world>`, `command-table`) and each
  `NpcPayload.at` become per-entity `world/<wid>/…` state records, and the old objects
  are tombstoned.
- `.6` then wins each world's lease (§9.2) as the only member, and lends itself every NPC
  lease. As further npcd members join, rendezvous spreads the characters across them
  (§9.3).
- **`motherd` cannot join this group as it stands.** The `.6` substrate is bound to a
  model key whose RoPE minimum differs from the one `motherd`'s engine runs (the refusal
  `docs/shared_model_host.md` §4.2 records). The group's model lock is set by `.6`, so a
  `motherd` npcd tenant would be closed 4412 (§3.6). There are two ways on:
  - **keep npcd members as standalone npcd processes**;
  - **start the npcd group fresh** under `motherd`'s model key, giving up `.6`'s history
    — the way the estate took: `.6` is rebuilt onto this framework and joins the group
    `motherd` started, empty.
- Further npcd machines join with empty substrates and catch up.
- Set `--cluster-quorum-max` if the npcd group should keep electing with only two
  machines (§16, item 10).

**4. Verify, on the status page and with traffic.**

- Both apps are clustered.
- Every member is ready, with no lag against any origin's `high_water` (§12.2).
- Each npcd world has an authority and its NPC leases are lent.
- A zend conversation started through the gateway is claimed by a member, and survives
  that member's Ctrl-C by moving to another (§12.4).
- A member with its network pulled serves its own listener locally, and on reconnect
  its turns appear after the group's (§5.8, §6.7).

**5. A second gateway**, only after gateway peering is in (§13.4):

- the same `session.key` and OAuth client on both;
- `cluster.peers` naming each other;
- both behind the public names;
- every member restarted with both `--cluster` URLs.

**Rolling back**, at any step:

1. Stop the members.
2. Restore the substrate copies from step 0, where one was taken. On `.5` none was: the
   standalone zend of the same build opens the adopted substrate as it stands.
3. Restart each daemon with its old launch line, without `--cluster`.

The gateway needs no change: with no members it is back in dumb mode, routing to the
same configured upstreams as before.

---

## 16. Open

1. **Bandwidth over the internet.** Full K/V replication means a recurrent `Snapshot` of
   ~63 MB per turn seal on the prescribed hybrid model, plus ~165 KB per layer per
   32-token block. A member on a home uplink catches up behind live traffic (§7.4), but
   it may be persistently behind. That is visible on the status page, and nothing here
   bounds it.
2. **Local builds serve real traffic and write real data** (§3.10), by decision. A
   developer's work-in-progress is routed to like any member, and its writes replicate.
   Draining it keeps gateway traffic off it. Nothing keeps its test data out of the
   shared substrate.
3. **The affinity idle timeout** (10 min) and the **catch-up hold** (30 s) are first
   values, to be measured against how long a moved conversation's K/V takes to land.
4. **A kept store has no expiry.** An operation's store is dropped when it publishes, or
   when it fails under `on-failed: discard` (§10.3). Under `keep`, the store waits for an
   operator to reopen the operation, and nothing ever drops it unattended. Whether a
   kept store should be dropped after some time, with the operations tab warning first,
   is open.
5. **Uploads larger than the 10 MiB store cap** are refused under §10.2. Whether
   uploads deserve a larger cap of their own is open.
6. **A member joined to only some clusters.** The design assumes every member is given
   every gateway, and then all gateways see the same members, modes, claims and owners.
   A member left off one cluster is still correct: what it misses comes back through the
   periodic catch-up it runs while a socket is down (§7.1), concurrent writes are
   ordered by stamp, and a key whose holder is unreachable gets a retryable `503`. It
   is less efficient, and a takeover across gateways that disagree costs a reload
   (§6.6). Whether a gateway should refuse a member that reports fewer clusters than its
   peers is open.
7. **Dumb mode with members on other gateways.** One gateway can still be dumb for an
   app, routing to its configured URL, while another is clustered. The configured daemon
   serves those requests locally, never forwarding them, because they carry the dumb-mode
   hop mark (§5.8). If it is a member, what it writes replicates like any member's
   writes, and a conversation it serves this way may also be claimed elsewhere, with
   the two writers ordered by stamp (§6.6). Only a configured daemon that is not a
   member at all writes outside the scheme, and its substrate splits from the group's.
   Deployment keeps the configured upstream a member (§17, `docs/deployment.md`).
8. **Offline turns read as appended, not as interleaved history.** A conversation that
   both the group and a cut-off member continued holds the group's turns first and the
   cut-off member's after (§6.6). Those were written without seeing the group's, so a
   reader meets two runs that did not know about each other. Nothing is lost. Whether
   zend should mark the seam in the conversation, for instance with a note naming the
   host and the time away, is open.
9. **Characters whose holder never returns stay absent** until an admin releases their
   NPC leases (§9.3). Nothing frees them automatically, by design. Whether the status
   page should flag them with their holder's last-seen time is open.
10. **The tie-break voter can itself be absent** (§5.9). A group of two decides with
   one member plus the voting gateway, so one member down no longer stops zend or npcd.
   What still stops a round is the voting gateway being down, or not yet eligible
   because a configured peer has never sent its node id since this gateway started.
   Until then the quorum is two members again. A larger group never counts the gateway.
   The remaining choices, if the estate wants a quorum that does not depend on a
   gateway, are unchanged:
   - run three members of the app; the third can be a small machine that only votes;
   - lower the quorum knowingly with `--cluster-quorum-max 1`, and accept that a
     partition of two can then elect a world on both sides, and two members can mint
     generations on both sides. Two equal generations are still ordered the same way
     everywhere, by the rest of the stamp, so nothing diverges. The invariant of §6.7
     then holds only within each side.

   Which the estate runs beyond the tie-break is a deployment decision.
11. **A cut-off character remembers a world the cluster forgot.** Its turns persist, but
   the world it lived them in is scrapped (§9.4). A character can recall walking to the
   lift and find itself back in the command room. Whether the character should be told,
   say with a perception on rejoin, is open.
12. **The apply rewrite is the largest piece of this design.**
   - Every apply path in `substrate.rs:4115-4320,5070-5226` becomes a stamped max, a
     union or a dominating fact.
   - Records that need a parent gain a pending map.
   - Section tombstones gain generations.
   - Compaction keeps timeline tombstones until they expire, then keeps their entries
     in the deleted set (§6.8).
   - Records of a type the build does not decode are deferred, not applied (§6.5).
   - The stream index is keyed `(stream_id, origin)`, and a timeline's turns are its
     decls in stamp order. The 34 `turn_stream_id` call sites are not touched.

   None of it is optional. Each item is a way the same events applied in a different
   order end in a different substrate (§1.3). The permutation test (§14.1) is what says
   it is done.
13. **A record only its origin held dies with that origin.** If a machine is retired with
    records no other member received, nobody can fetch them. Order independence is over
    the events a member receives. Retiring a member should wait for its `high_water` to
    be applied everywhere: the status page shows it, and nothing enforces it.
14. **No backup, by decision (D17).** A mistaken delete or a bad build's writes replicate
    to every member, and compaction then removes the data. Periodic snapshots, or a
    member that applies events on a delay, would give a way back. Until one is designed,
    the only way back is a copy taken by hand, as the migration's step 0 does (§15.1).
15. **No encryption at rest, by decision (D17).** Every member, a laptop included, holds
    every conversation. Protection is whatever the machine's disk encryption gives.
16. **No group-key rotation, by decision.** Replacing a leaked key means replacing it on
    every gateway and member and restarting them. Members are refused 4401 until they
    have the new key.
17. **No alerts.** Fault mode, a member stuck rejoining without a quorum, a lost quorum and a member far behind show
    on the status page and in its log. Nothing notifies anyone who is not looking.
18. **No test on real machines.** Every test in §14 runs in one process. Before
    production, the design needs a staged run across two or three real machines, one of
    them over the internet through Cloudflare, exercising a cut-off, a gateway restart
    and a full disk.
19. **The npcd console shows one member's cast.** Its pulse, census, world, guardian and
    guest instruments are aggregated per member (§5.5), and the console reads the first
    member's answer (`web/content/npcd/lib/api.live.js`). With one npcd member that is
    the whole cast. With a second — `.6` rebuilt and joined — each payload needs its own
    union (characters by `nid`, census counts summed, a world from its authority), and
    the telemetry tab a member picker like zend's.

---

## 17. Documents this changes

| Doc | Change |
|---|---|
| `docs/web_gateway_design.md` | A new section for gateway peering, with ACME challenge and certificate sharing. §3: the `cluster:` route key, dumb and clustered modes, dispatch, claims and aggregates, and failing in-flight exchanges on a socket drop. §5: cluster trust next to the IP allowlist; the assertion and cookie are no longer forwarded. A new section for the status page. |
| `docs/shared_model_host.md` | §3.4: the cluster listener in the host server, and the own listener forwarding up while joined. §3.5: dispatch on every route and the `cluster` status block. |
| `docs/zend_workspace_execution.md` | §7 and §8 are built by §11 here. §9.4 is answered. |
| `docs/zend_git.md` | §9: publishing happens through agents. |
| `docs/zend_vfs_events.md` | Events are keyed `(timeline, origin, seq)` and replicate. Uploads become VFS. |
| `docs/archived/segmented_substrate_log.md`, `docs/archived/kv_tier_migration.md` §5 | The record header's new fields (`origin`, `oseq`, `hlc`, `epoch`, `digest`); `Replica`; the order-independent apply rules; timeline tombstones kept by compaction; incomplete remote turns pending, not `Corrupt`. |
| `docs/the_effector.md`, `docs/vault_world.md` | The `vault-archive` floor, the merge table, and operation-scoped mind reads. |
| `docs/npcd_workflows.md`, `docs/npcd_workflows_current.yaml` | Step key `desk`; `on-failed: discard \| keep` in place of `set-aside`, `restore` and `keep`; every workflow ends in a `publish` step after its verification, where the Maker resolves any merge conflict at the merge table; the check `published`; only a publish step leads to `done`; each step's mission works in the operation's store (§10.3). |
| `docs/deployment.md` | The group key, `--cluster` (every gateway on every member), agents, `group:` upstreams, and more than one gateway behind the public names. |
