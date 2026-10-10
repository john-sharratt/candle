# One process, two daemons: a shared model host for zend and npcd

**Status:** design, not built. **Scope:** a single binary that serves zend and npcd
together — one loaded model, one conversation engine, a separate substrate each. The
model is prescribed: one fine-tune, named in one place, for every tenant. A substrate
is bound to the model that wrote it: a daemon whose substrate was written by a
different model refuses to start.

The standing constraint is that zend and npcd keep being developed as they are, so
the design moves process-level concerns into a host and leaves each daemon's own
code almost untouched.

---

## 1. Where things stand

Two binaries, two machines, two models, two engines:

| | zend | npcd |
|---|---|---|
| Machine | `.5` (RTX PRO 5000, 72 GB) | `.6` |
| Model | `model_choice::resolve` — VRAM ladder: Qwen3.8-Flash-Next Q4_KO above 60 GiB, else Qwen3.6-35B-A3B Q4 (`zend/src/model_choice.rs:54-105`) | hard-coded `Qwen36_35B_A3B_AntiLoop_StyleTune` (`npcd/src/model.rs:54-56`); no flag, no ladder |
| Engine | one `ConversationEngine` in `InferenceState.engine: Mutex<_>` (`zend/src/session.rs:475`, built `:1292`) | one `ConversationEngine` as an `Arc<Mutex<_>>` (built `npcd/src/engine/runtime.rs:3866`), held by `Minds` (`engine/mind.rs:203-204`) |
| Substrate | engine opens `<workspace>/substrate` itself (`engine.rs:387-412`) | npcd opens `<data>/substrate` (`npcs.rs:218-252`) and hands it over with `.substrate(shared)` (`runtime.rs:3791-3795`) |
| HTTP | its own `axum::serve` + `web::tls` entrance (`zend/src/main.rs:680-721`) | `web::Builder … .behind_gateway().local_api("npcd", …)` (`npcd/src/main.rs:861-870`) |
| UI | compiled in (`include_dir!`, `zend/src/api/mod.rs:43`) | compiled in, or from disk with `--content` — which production passes (`docs/deployment.md:88`) |
| Projection | `src/prompts/projection.yaml`, 10 layers, dialogue `High` | `<mind>/projection.yaml` (the bundled mind: `npcd/projection.yaml`), 9 layers, perception/action/interaction `High` |

Neither crate depends on the other; both link `candle-conversation` (`hub`), `web` and
`candle` (`cuda`) with no conflicting features.

### 1.1 What the engine already supports, and what it does not

**The scheduler is already multi-substrate.** `NewSequence`, `NewEphemeralSequence`
and `ResumeSequence` each carry a `Conversation` (`scheduler/mod.rs:164-226`), the
scheduler keeps one per slot (`slot_conversations`, `:3250`), and compaction, relief
and cold-ingest demotion walk `distinct_substrates()` (`:6744-6759`; called at
`prefill.rs:489`, `:1494`, `:1758`). A test already runs slots from two substrates on
one scheduler (`slots_sharing_a_substrate_are_walked_once`, `:15487`).

**Everything above the scheduler is bound to one.**

- Every engine factory sends `self.conversation.clone()` (`engine.rs:2037-2052`,
  `:2194-2201`, `:2310-2317`); there is no public way to create a sequence on another
  substrate.
- One `PersistenceThread` per engine, bound to that one `Conversation`
  (`engine.rs:491`, `persistence/thread.rs:207-317`). A second substrate would get no
  hot→warm drain, and `evict_hot_to_free` only evicts what has a warm copy
  (`substrate.rs:2802`), so its hot KV could never be relieved.
- The startup reload (`reconstruct_substrate`) runs only on the engine's own
  conversation (`engine.rs:532`, `:569`, and the `ReconstructSubstrate` request at
  `:662`).

**Keys that carry no substrate.** `TurnKey`, `TimelineId`, `SectionId` and `StreamId`
key scheduler-wide maps (`turn_positional` `:3170`, `ingest_timelines` `:3456`,
`timeline_projections` `:3530`) and the gallery arena's residency map
(`gallery_arena/mod.rs:113`). Each `Conversation` has its own `TimelineAllocator`,
minting `max(now_µs, last+1)` (`projection/ids.rs:372-421`), so two substrates can
mint the same id, and two substrates opened on different machines may already hold
colliding ids. A `SectionId` is a hash of the section's tokens, so two substrates
holding the same text name it alike.

### 1.2 What a substrate records about its model — and the bug in it

| Record | Model identity | Checked on open |
|---|---|---|
| `Tokenizer` | SHA-256 of `tokenizer.json` | **yes** — a different tokenizer is refused (`engine.rs:454-467`, `persistence/mod.rs:1992-2015`); skipped, with a warning, when `tokenizer.json` cannot be read (`builder.rs:1382-1388`) |
| `ModelSpec` | text: arch, chat format, repo, filename, tokenizer repo, max_seq_len (`builder.rs:913-934`) — no revision, tensor overrides, adapters or gate donor | **no** — written last-writer-wins, read only by tests and `substrate_inspect` |
| KV chunks | `KvFormat` tag, stream, index, token count; per-head formats (`persistence/record.rs:345-365`, `:580-598`) | no model field; layer count not recorded — the reload assumes the running model's (`scheduler/mod.rs:9409`); only `substrate_inspect` infers it, by majority vote (`resume.rs:727-751`) |
| Recurrent snapshot | `schedule_hash` = layer schedule + DeltaNet dims (`candle-transformers/src/models/delta_net/state_store.rs:135-160`) | geometry only; a mismatch is logged and the timeline runs with no recurrent memory (`scheduler/mod.rs:7068-7103`) |
| WideQSig | fold stamp (`wide_sig.rs:335-344`) | only the group size; a mismatch skips scoring the record, not the open (`:537-552`) |

So a model change today does one of three things, none of them right:

- **Same tokenizer, same KV geometry, different weights** (a quant change, a
  fine-tune, Qwen3.6 → its AntiLoop hybrid): the old model's KV is injected silently.
  Nothing checks it; the only payload check, the Fletcher golden, is non-fatal
  (`persistence/pipeline.rs:515-531`). Content-addressed sections are adopted with no
  model check: their address hashes tokens only, and the restore refuses only a chunk
  count the running layer count does not divide (`resume.rs:584-601`).
- **Different layer count:** every turn fails `recover_turn_cold_refs`
  (`resume.rs:524-531`) and its layer's corrupt-turn policy fires. Under
  `DropConversation` — the default (`projection/schema.rs:951-961`), and every zend
  layer's but dialogue's — every timeline is durably tombstoned; zend's dialogue is
  `drop_turn`, so every turn of every chat is dropped instead.
- **Different tokenizer:** the engine refuses to start — the one case handled right.

**A second defect, found beside it:** npcd never passes its schema's corrupt-turn
policies to the engine (no `.corrupt_turn_policies` at `npcd/src/engine/runtime.rs:3791-3795`),
so its perception, action and interaction layers, which ask for `drop_turn`
(`npcd/projection.yaml:171`, `:215`, `:259`), run `DropConversation`: one corrupt turn
tombstones the character's whole timeline.

This is a defect independent of the host. §4 fixes it for every daemon by extending
the tokenizer's rule to the whole model.

### 1.3 Process-wide state

Shared by design, and fine with one model and one scheduler: the device span
reservation, region pool, ground broker, wave domains and VRAM governor (all keyed by
device ordinal, `region_pool.rs:547-602`, `bump_arena.rs:1078-1087`); `ACTIVE_FOLD`
(`wide_sig.rs:160`, "one process loads one model"); the scheduler's telemetry and
admission atomics (`PREFILL_OK_TOKENS`, phase rings, `memory_report::LATEST`).

Wrong once two daemons share a process:

| Global | Why it breaks |
|---|---|
| `cancel.rs:28-35` `INGEST_CANCEL` | one latch for the process; zend's shutdown cancel would stop npcd's ingest too |
| `layer_toggle.rs:24-27` `DISABLED` | keyed by layer *name* across every conversation; a toggle in one daemon reaches the other's layer of the same name |
| the global tracing subscriber | both `main`s call `.init()`; the second panics |
| zend's panic hook (`main.rs:343`), its Ctrl-C flag reset (`ctrl_c.rs`), `std::process::exit` | process-wide; zend exits the process on load failure, at shutdown and to self-heal (`session.rs:6171`, `:6183`, `:6360`; `main.rs:740`, `:776`; `self_heal.rs:217`, `:240`) |
| `#[tokio::main]` | each owns a runtime |

Daemon-private statics stay daemon-private and are fine with one copy of each daemon:
zend's `tool_def::DEFS`, `SCAN_*`, `ingest_backlog`, `ingest_report`; npcd's
`CATALOG`, `SERVED`, `READ`.

---

## 2. Decisions

1. **One host, daemons as tenants.** A new crate, `host/`, owns the process: the
   runtime, logging, panic and signal handling, model choice, the engine core, and
   shutdown. zend and npcd become libraries exposing a `Tenant`.
2. **Three binaries, one startup path.** `zend` and `npcd` keep their binaries and
   flags, each `main` becoming `host::run([tenant])`; `tokerad` is the same call with
   both tenants. All three build from the workspace. There is no solo path beside a
   combined one.
3. **What is common moves to a common home; what is specific stays.** Code both
   daemons duplicate, or that generalizes past either, moves to the host crate, to
   `candle-conversation` or to the shared front-end — each to where its kind already
   lives (§3.7). Code only one daemon needs stays in that daemon, where it is today.
4. **Split the engine into a shared core and per-substrate attachments.**
   `ConversationEngine` keeps its name and its methods, and becomes *an attachment*:
   the shared `EngineCore` (scheduler, model, tokenizer, device, guests) plus one
   substrate with its own persistence thread, writer, reload and cancel. Tenant code
   that calls `engine.new_conversation_with_projection(…)` keeps doing so unchanged.
5. **Scheduler keys carry the substrate.** A `SubstrateId`, assigned at attach, is
   part of every scheduler-wide key that today is a bare `TimelineId`/`TurnKey`/
   `StreamId`.
6. **A substrate is bound to one model.** It records the identity of the model that
   wrote it, and opening it under any other model is refused — before the weights
   load, naming both models. Nothing is regenerated, injected or tombstoned. Moving a
   daemon to a new model means a new substrate (or the old model).
7. **One prescribed model for the process.** The host names a single model —
   `Qwen36_35B_A3B_AntiLoop_StyleTune`, the fine-tune npcd already runs — and every
   tenant runs it. No candidate lists, no VRAM ladder, no per-daemon choice: zend's
   `model_choice` ladder and `--model` flag go. A tenant only declares what it needs
   *of* a model, which is checked against the prescribed one at start.
8. **One HTTP server; where the daemons overlap, one implementation.** A path both
   daemons serve is functionality both need, so the host serves it once, as the
   superset of the two (§3.5) — the shared engine reported once, each tenant's own
   data under its name. What only one daemon serves stays that daemon's. The API is
   then one namespace with no collisions; only the two web UIs, which are different
   apps at the same `/`, are told apart by host name. **The UIs are served from disk**,
   so editing one needs no rebuild and no restart.
9. **zend outranks npcd.** Priority is ordered by tenant first, band second: while
   zend has work, npcd waits; within each tenant the bands work as today.

---

## 3. The host and the tenants

### 3.1 The tenant contract

```rust
// host/src/tenant.rs
pub trait Tenant: Send + 'static {
    /// A short name: log target, status key, metric label. "zend", "npcd".
    fn name(&self) -> &'static str;

    /// What this tenant needs of the model — checked against the prescribed one (§5).
    fn requires(&self) -> Capabilities;

    /// Where this tenant's substrate lives, and how it is opened.
    fn substrate(&self) -> SubstrateSpec;

    /// What this tenant asks of the shared engine core (merged across tenants, §3.3).
    fn core_requests(&self) -> CoreRequests;

    /// Start: build state on the tenant's engine attachment, start background work,
    /// return the site to serve. Called once the model is loaded and the substrate
    /// is attached.
    fn start(self: Box<Self>, ctx: TenantContext) -> BoxFuture<'static, Result<Running>>;
}

pub struct TenantContext {
    pub engine: ConversationEngine, // this tenant's attachment (§3.2)
    pub log: LogBus,                // a scope of the host's bus, target = name()
    pub shutdown: ShutdownSignal,   // resolves when the host is stopping
    pub progress: LoadProgress,     // this tenant's row on its status page
}

pub struct Running {
    pub api: Router,                 // this tenant's own routes — none the host serves (§3.5)
    pub ui: TenantUi,                // its web UI, chosen by host name (§3.4)
    pub reports: Arc<dyn Reports>,   // what the shared endpoints report about it (§3.5)
    pub stop: Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>, // drain
}

pub struct TenantUi {
    pub hosts: Vec<String>,   // "code.tokera.com", "zend.localhost"
    pub content: Roots,       // its UI directory on disk, read per request (§3.4)
    pub default: bool,        // serves a request whose Host names no tenant
}

/// The tenant's share of the host's shared endpoints. Each method answers for this
/// tenant only; the host assembles the superset.
pub trait Reports: Send + Sync {
    fn status(&self) -> TenantStatus;        // its load steps, readiness, own extras
    fn tools(&self) -> Vec<ToolView>;        // its catalog
    fn layers(&self) -> Vec<LayerView>;      // its projection schema, with live counts
    fn series(&self) -> TenantSeries;        // its telemetry columns
    fn me(&self, caller: &Caller) -> Value;  // its per-caller extras (zend: tool modes)
}
```

What moves out of each `main` into the host: `#[tokio::main]`, the panic hook, the
tracing subscriber, the Ctrl-C handler, `set_gpu_span_period`, zend's
`std::process::exit` calls (the host exits once, after every tenant has stopped), and
model resolution. What stays in the tenant: everything else — zend's loader thread,
calibration, titler, ingest worker, OriginWatch; npcd's runtime, character loops,
metronomes, mission generator.

**The diff to each daemon is the move, not a rewrite.** zend's `InferenceState::load`
today calls `builder.engine_with_progress(&device, …)` (`session.rs:1292`); it receives
the attachment instead. npcd's `runtime::load` does the same at `runtime.rs:3819-3827`.
Their CLIs become the tenant's config struct; their `main.rs` shrinks to parsing it.

Logging: the host owns the one subscriber; each tenant's `LogBus` is a scope whose
events carry `tenant = name()`, so `/ws/logs` on zend shows zend's lines and npcd's
shows npcd's, and the host log file has both.

### 3.2 Engine core and attachments

```
EngineCore (Arc, one per process)              candle-conversation/src/engine/core.rs
├─ scheduler thread + Sender<SchedulerRequest>
├─ model (moved to the scheduler), tokenizer, device, guests
├─ model identity: ModelKey (§4.1)
└─ attachments: Vec<Weak<Attachment>>

ConversationEngine (one per tenant) = Arc<EngineCore> + Attachment
Attachment
├─ substrate_id: SubstrateId
├─ conversation: Conversation       (this substrate)
├─ persist_thread: PersistenceThread
├─ reload status
└─ cancel: CancelToken              (replaces the process-wide INGEST_CANCEL)
```

- **Factories** keep their signatures and send `self.attachment.conversation` — the
  attachment's, not a single engine-wide one.
- **Persistence:** one `PersistenceThread` per attachment. Each drives its own
  substrate's hot→warm→cold; the scheduler's relief already walks distinct substrates.
- **Reload:** `reconstruct_substrate` runs per attachment, on attach.
- **Cancel:** `ingest_cancelled()` becomes `attachment.cancel.is_cancelled()`; the
  pools read it through their `RefreshContext` (zend) and ingest context (npcd).
- **Layer toggles:** `layer_toggle` is keyed `(SubstrateId, layer)`.
- **Shutdown:** `ConversationEngine::shutdown` detaches — flush and stop its
  persistence thread, release its slots. The core stops when the last attachment
  detaches.
- **One writable handle per substrate directory** becomes enforced. Today it is a
  documented rule only (`persistence/mod.rs:194-213`): `LogFile::open` takes no lock,
  and a second writer silently loses records at the next compaction. Attaching takes
  an exclusive lock on the substrate's log, so a second process — a `zend` started
  beside a `tokerad` that already serves that workspace — is refused, naming the
  holder; and the host refuses two tenants naming the same directory before either
  opens it.
- **Per-substrate builder settings** move to the attachment: the system prompt, the
  workspace path, read-only, and the corrupt-turn policies, which already live on the
  substrate (`engine.rs:481`).

**`SubstrateId` in the scheduler.** Every scheduler-wide map keyed by `TimelineId`,
`TurnKey`, `SectionId` or `StreamId` is keyed by `(SubstrateId, …)` instead:
`turn_positional`, `ingest_timelines`, `timeline_projections`, the gallery arena's
residency map, the working-set pin republication, the relief and demotion keep-sets
(`prefill.rs:1484-1489`, `:1750-1751`). The id is in-memory only — assigned at attach,
never persisted — so it costs the substrate format nothing. The compiler finds every
site: the key types change.

### 3.3 Core settings two tenants both set

Some builder settings are per-model or per-scheduler, and the two daemons set them
differently today:

| Setting | zend | npcd | Rule |
|---|---|---|---|
| `prefill_pass_tokens` | sized from the card (2048–8192) | 8192 (`runtime.rs:133`) | max |
| `min_rope_factor` | 2 (schema `rope.min_yarn_factor`, `session.rs:1253`) | unset | max — RoPE schedule is the model's |
| `qsa_selection_budget`, `qsa_strata` | CLI (`session.rs:1249-1250`) | unset | zend's; npcd sets none |
| `max_response_tokens` | 65,536 (`session.rs:349`) | 16,384 (the builder default) | max |
| guests (`<data>/guests.yaml`) | none | npcd's, when the file exists (`runtime.rs:3757`) | union, namespaced by tenant |

`CoreRequests` carries these; the host merges by the rule column and logs the merged
values at start.

Settings that already agree need no rule: `compression_level` is 5 for both (zend sets
it, `session.rs:1258`; npcd gets the builder default, `builder.rs:205`), and a
conversation's own level overrides it either way (`SequenceConfig::kv_compression_level`);
the expert pack sits beside the GGUF for both (zend sets it, `session.rs:1274-1276`;
npcd gets the builder's default for a `Qwen35Hybrid`, `builder.rs:1190-1193`).
`thinking` needs none either: the core enables it, as zend does (`session.rs:1262`),
and npcd already sets the mode on every turn (`engine/mind.rs:909`, `:916`) — that
the per-turn mode wins over the core flag is checked in phase 3. Per-sequence
settings — sampling, dialect templates, stencils, projection schemas, think steering
— are already per conversation and need nothing.

### 3.4 One HTTP server

One `web::Builder` — the server npcd already runs on (`npcd/src/main.rs:861-870`) —
serves everything, in three layers resolved in this order:

1. **The host's shared endpoints** (§3.5) — one router, the same on every host name.
2. **Each tenant's own API** — merged into the same router. Once the overlaps are
   the host's, the remaining paths are disjoint: zend's `/v1/chat/completions`,
   `/v1/conversations/*`, `/v1/repo_map`, `/v1/models`; npcd's `/v1/npc/*`,
   `/v1/world/*`, `/v1/life/*`, `/v1/pulse/*`, `/v1/generate/*`, `/v1/image/*`,
   `/v1/interaction/*`, `/v1/local/*`, `/ws/events`, … A path two tenants register
   is refused at start: it is an overlap that belongs in §3.5.
3. **The web UIs, by host name.** zend's UI and npcd's console are different apps at
   the same `/` (and both ship a `/favicon.svg` and `/robots.txt` with different
   content), so the static layer picks by `Host`:

| UI | Hosts | Content (on disk) | Default |
|---|---|---|---|
| zend | `code.tokera.com`, `zend.localhost` | `zend/web/` | **yes** |
| npcd | `bot.tokera.com`, `npcd.localhost` | `web/content/npcd/` | no |

The shared front-end library — `zend/web/common/` and `web/content/common/`, byte for
byte the same `base.css` and `lib/*` (zend keeps its copy of `brand/` at
`zend/web/brand/`) — is served once, at `/common/`, for both; npcd's console, which
serves those files at its root today (`/base.css`, `/lib/*.js`, through its second
root), loads them from `/common/` like zend's does. The two tests that keep the copies
in step (`npcd/src/console.rs:379`, and `zend_carries_the_shared_framework_unchanged`
in `web/src/site/tokera/page.rs`) go with zend's copy.

**Served from disk, never embedded.** Every UI file — HTML, JavaScript, CSS, images —
is read from its directory when it is requested, the way the gateway already serves
tokera.com (`Roots::disk`). Saving a file is the deployment: the next request serves
it, with no rebuild and no restart.

- zend's `include_dir!` copy (`zend/src/api/mod.rs:43`) and npcd's embedded roots
  (`npcd/src/main.rs:48-49`) are deleted, and with them npcd's `--content` switch
  (`:252-266`), which production already passes (`docs/deployment.md:88`) — there is
  one way to serve a UI, not an embedded way and a disk way.
- Each content directory is a path in the tenant's config, resolved against the
  working directory (the repository root, which every daemon already runs from,
  `docs/deployment.md:67-69`). A directory that does not exist is a startup error
  naming it.
- Files go through `web::asset::respond`, as both daemons' do today: a strong ETag
  over the content, `If-None-Match` answered `304`, gzip, and `max-age` from
  `cache.short_secs` (60 s) for HTML, JS and CSS. `Roots::read` reads the file on
  every request (`web/src/content.rs:119-154`) and caches nothing, so an edit is
  served on the next request and a browser refresh revalidates.
- **Reload on change.** Both UIs already reload themselves when `/v1/status`'s `build`
  changes (zend `index.html:549-554`; npcd `common/lib/build.js`, polled every 5 s by
  `web/content/npcd/app.js:575-579`). zend's `build` hashes its embedded files today
  (`api/mod.rs:49-83`); npcd's is the constant `npcd-<crate version>` (`ops.rs:163`),
  so its console reloads only on a version bump. `build` becomes a hash over the
  content directories as they are on disk, plus the binary's version (§3.5),
  recomputed when a file under them changes — so an open page picks up an edit on its
  next status poll, not only a rebuilt binary.

**Ports.** The host serves the one router on `192.168.0.5:8081` (plain) and
`192.168.0.5:8444` (TLS, a certificate made at startup — `web::tls::self_signed`), the
two ports zend has today. npcd moves from `.6` (`0.0.0.0:8081`) onto them.

- **A request by IP** gets zend's UI — what `http://192.168.0.5:8081/` shows on the
  LAN today — and the whole API. npcd's console is reached by its names:
  `bot.tokera.com` through the gateway, `npcd.localhost` or a LAN name for `.5`
  directly.
- **The gateway** points both `code.tokera.com` and `bot.tokera.com` at
  `https://192.168.0.5:8444` with `self_signed: true`. It already forwards the
  client's `Host` unchanged (`web/src/proxy.rs:184-191`; no route sets
  `rewrite_host`), which is what the static layer picks the UI by.
- **Compression.** The builder gzips what it serves (`web/src/compress.rs`), zend's
  responses now included. It streams and flushes per chunk, so zend's SSE replies are
  not held back.

What each daemon drops: zend's `axum::serve` and TLS setup in `main.rs`
(`:680-721`) and its embedded asset fallback (`api/mod.rs:43`, `:151-187`); npcd's
`Builder` call (`main.rs:861-870`), its embedded roots and `--content`, its `ops`
router and its overlapping routes.

### 3.5 The shared endpoints

Every path both daemons serve, and every near-duplicate under a different path, is
served once by the host. Each reports the shared engine once and each tenant's own
data under `tenants.<name>`, assembled from the tenants' `Reports` (§3.1). Where the
two daemons disagree on a field, one meaning is chosen and both UIs follow it.

#### Access and identity — one rule

The two daemons trust the gateway differently today: zend believes `x-tokera-*`
headers only from a trusted peer (`zend/src/access.rs:73-88`), npcd believes them
from anyone (`behind_gateway`, `web/src/server.rs:394-404`; `npcd/src/identity.rs:62`).
zend has no route guards; npcd grades its routes by role (`Api::route(path, min_role, …)`,
`npcd/src/guard.rs:91`), all but `/v1/local/*`, the effector API, which authenticates
by device token and keeps doing so.

- **Identity:** zend's rule, for both — headers are believed from the gateway's
  address and loopback, never from anyone else. It is the stricter, and npcd's
  "behind the gateway" was an assertion about the network that nothing checked.
- **Roles:** one estate table, merging `zend/zend.roles.yaml` and the `roles:` block
  of `npcd/npcd.web.yaml`.
- **Guards:** each shared endpoint carries the stricter of the two daemons' guards.
  Consequence: zend's `/perf`, `/logs` and `/substrate` pages, which today work for
  anyone on the LAN, need a signed-in role — through the gateway, or
  `--local-signin` on loopback.
- **Accounts** (`npcd/src/accounts.rs`) become the host's, so zend gets account
  records — unique name, profile — with no code of its own.

#### The endpoints

| Shared endpoint | Replaces | Superset |
|---|---|---|
| `GET /v1/status` | zend `/v1/status`; npcd `/v1/status` | `state`, `started_at_ms`, `build`, `model{…}`, `engine_connected` (model loaded), `loading` (the model's load), `tenants.<name>{state, loading, maintenance}` plus zend's `ingest_backlog`. `loading` is npcd's richer shape for every load (`npcd/src/engine/loading.rs:95`): `current`/`completed` are **step ids**, `label` the text, `detail` the item in hand, `elapsed_ms`, `prefill_tokens`, `prefill_tps` — where zend's `current` is a label and `completed` a list of labels. `build` is **one hash over the UI content on disk and the binary's version**, recomputed when a content file changes (§3.4) — zend's meaning, so an open page of either UI reloads on a UI edit or a rebuild, where npcd's console today reloads only on a version bump. npcd gains `maintenance` for its own substrate |
| `GET /v1/me` (and `/v1/me/profile*`, `/v1/me/unique-name`) | zend `/v1/me`; npcd `/v1/me` and its account routes | `role`, the account fields (`user_id`, `unique_name`, `profile`, `email`, `display`, `avatar_url`, …), and `tenants.zend{tool_modes, default_tools}`. A signed-out caller gets **200** with `role: "unauthenticated"` and no account — zend's behaviour; npcd's console reads the role instead of a 401 |
| `GET /v1/memory` | zend `/v1/memory`; npcd `/v1/memory` | the shape both already share; `report` is the shared engine's — populated for both, where npcd's is hard-coded null today (`npcd/src/telemetry/memory.rs:72-73`) |
| `GET /v1/telemetry` | zend `/v1/telemetry`; npcd `/v1/telemetry` | zend's per-wave rows (`waves`, `fmt`, `migrate`, `has_pressure`) and npcd's columns (`gpu`, `model`, `series`, `sample_period_s`, `uptime_s`), side by side — they are different data, not the same data twice. The engine columns (`decode_tps`, `prefill_tps`, `kv_mib`, `weights_mib`, …) are fed by the shared scheduler — in npcd today `Telemetry::record` has no caller, so only `prefill_tps` is filled, from the load progress during startup, and its telemetry `engine_connected` is always false. One `host{total_mib, free_mib, rss_mib}` (npcd's units). npcd's own columns (`npcs_active`, `ticks_per_sec`, inbox depths, image queue) under `tenants.npcd` |
| `GET /v1/phases`, `/v1/promotes`, `/v1/profile` | zend's | process-wide already — the scheduler's phase rings, and the pipeline profiler (`candle-transformers/src/models/profile/pipeline.rs`), which `/v1/profile` resets on read; now visible to npcd's console too |
| `WS /ws/logs` | zend `/ws/logs`; npcd `/ws/logs` | the frame both already use, `{ts, level, target, msg}`, plus `tenant` and a `seq`. The host's one bus, with zend's broader filter (npcd's drops most `candle_conversation` output). Subscribe first, then replay the ring, skipping by `seq` what the subscription already delivered — zend can lose a line today and npcd can repeat one |
| `GET /v1/substrate`, `/v1/substrate/:tenant/…` | zend `/v1/substrate/*`, `/v1/debug/maintenance`; npcd `/v1/substrate/storage`, `/v1/npc/:nid/substrate/*` | the overview lists every attached substrate: storage (npcd's `open`, `path`, `segment_count`, with `live_chunks` and `dead_ratio`, which npcd carries as always null today and zend fills — now real for both), counts, layers. Layer, timeline, selection, system-prompt, project and maintenance views move under `/v1/substrate/:tenant/`. One field meaning each: `timeline` is a **string** (a u64 does not survive a JavaScript number — npcd sends it as one today), `turns` is an array with `turn_count` beside it, `conversations` an array with `conversation_count`. npcd's per-character view is the same handler filtered to one NPC, with npcd's ownership check |
| `GET /v1/tools`, `POST /v1/tools/calibrate` | zend `/v1/substrate/tools`; npcd `/v1/tools`, `/v1/tools/calibrate` | every catalog: `tools[]` carries `tenant` and the union of fields (both: `name`, `description`, `parameters`; zend's `high_risk`; npcd's `category`, `plane`, `modes`, `needs`, `at_named`, `source`, `calibrated`, `examples` as a count), with npcd's top-level `uncalibrated` and `acts_per_turn` under its tenant. Calibration runs the named tenant's — zend's exists today; npcd's answers 503 `no_engine` today and has the shared engine to run on |
| `GET /v1/schema/layers` | zend's layer list inside `/v1/substrate`; npcd `/v1/schema/layers` | every tenant's layers, each with `tenant`, the raw schema entry (npcd's shape, passed through from the mind's yaml: `budget`, `groups`, `summary`, `gather_scope`, `decode_priority`, …) and zend's live fields (`conv_count`, `tokens`, `disabled`). Formatting (`priority` from `budget`, group strings) moves to the UI |

What stays a tenant's own: zend's `/v1/models` (the OpenAI listing Continue reads,
naming zend's model ids), `/v1/chat/completions`, `/v1/conversations/*`,
`/v1/repo_map`; everything npcd serves about its world and characters.

**The UIs change where the meaning was chosen:** zend's status overlay reads
`loading.label` for its text and `tenants.zend.*` for its backlog and maintenance;
its substrate page uses `/v1/substrate/zend/…` and `/v1/tools`; npcd's console reads
the 200 sign-out, `/common/`, the string `timeline` and `tenants.npcd.*`.

### 3.6 zend outranks npcd

One scheduler now carries both daemons' work, so the priority bands need an order
across tenants. The order is **tenant first, band second**:

```
zend High > zend Normal > zend Low > npcd High > npcd Normal > npcd Low
```

- **Priority pause** (`scheduler/priority_pause.rs`) pauses every sequence ranked
  below one that has work, and keeps it paused for `COOLDOWN` (15 s) after. So while
  zend has anything to do — a chat turn, a tool round, the titler, an ingest pass —
  npcd's characters wait, and they resume within `PAUSED_POLL` (250 ms) of zend
  falling idle for the cooldown. Within npcd, its bands work exactly as today. Section
  ingests and deferred glue stay outside the pause in both tenants, as they are today
  (`priority_pause.rs:138-143`).
- **The prefill throttle** (`prefill.rs` `throttle_ratio`) uses the same order: work
  yields to the decodes that outrank it. A zend decode holds npcd's prompts back; an
  npcd decode never holds back a zend prompt; two conversations in the same tenant
  and band share the wave as they do today.
- **What it costs npcd:** a zend ingest pass — at startup, or when the watcher sees
  source change — pauses the characters for its duration. zend's chat is short
  bursts; its ingest is the long one. If the characters' stalls during ingest matter,
  the lever is zend's ingest pacing, not the order.

The change in code is one type: the priority a slot resolves to becomes
`(TenantRank, DecodePriority)` instead of `DecodePriority` (`decode_priority_or_high`,
`admit_ground.rs:667`), `rank` (`priority_pause.rs:41`) ranks the pair, and
`PriorityPause::last_active` grows from one slot per band to one per tenant and band
(`:52`). The host
assigns `TenantRank` from the order tenants are given to `host::run` — zend first.

### 3.7 Where code lives, and the three binaries

**The rule.** Code that both daemons carry, or that is about the process, the engine
or the estate rather than about coding or characters, moves to a common home. Code
only one daemon needs stays in that daemon, in the file it is in today. Each kind of
common code goes where that kind already lives:

| Common home | Holds |
|---|---|
| `host/` (new crate) | the process and everything served about it: tenant contract, startup and shutdown, logging, model loading, the shared endpoints, identity, accounts |
| `candle-conversation` | the engine: core and attachments, `SubstrateId`, model identity and the binding check |
| `web/` | the HTTP server, as now; and the shared front-end, `web/content/common/` |

**What moves** — each a pair of copies today, or one daemon's generalization the
other needs:

| From zend | From npcd | To |
|---|---|---|
| `log_broadcast.rs`, `log_line.rs`, `api/ws_logs.rs` | `logs.rs`, the `/ws/logs` handler in `ops.rs` | `host::logs` — one bus, one frame, one socket handler |
| `loading.rs` | `engine/loading.rs` | `host::loading` — npcd's snapshot shape (§3.5) |
| `api/status.rs` | the status handler in `ops.rs` | `host::api::status` |
| `api/telemetry.rs`, `api/memory.rs`, `api/profile.rs` | `telemetry/` (`mod.rs` sampler, `ring.rs`, `device.rs` NVML, `sample.rs`, `memory.rs`) | `host::telemetry` |
| `api/me.rs` (the shared part) | the `/v1/me` routes in `api.rs` (`:429-450`), `accounts.rs` | `host::api::me`, `host::accounts` |
| `access.rs` (peer trust, `--local-signin`, role lookup) | `identity.rs`, `guard.rs` (route grading) | `host::access` |
| `api/substrate.rs` and the substrate views in `session.rs` | `substrate.rs`, `engine/layers.rs` | `host::api::substrate` — over any attachment |
| the tools-catalog view | the `/v1/tools` view in `engine/pulse.rs` | `host::api::tools` (each catalog stays in its daemon) |
| `main.rs` process setup: panic hook, tracing init, `ctrl_c.rs` (the Ctrl-C flag reset), `self_heal.rs` | `main.rs` process setup: tracing init, Ctrl-C | `host::process` |
| `download.rs`, `model_choice.rs` (deleted, §5) | `model.rs` | `host::model` — `MODEL`, the download and prepared-artifact lookup |
| `web/common/` (a copy of the shared front-end) | — (serves `web/content/common/` at its root) | `web/content/common/`, once — zend's copy and the two sync tests deleted, both UIs load it from `/common/` |
| the `include_dir!` UI embed and its build hash (`api/mod.rs:43-83`) | the embedded roots, `--content` (`main.rs:48-49`, `:252-266`), the constant `build` (`ops.rs:163`) | the host's disk-served UI layer and its on-disk `build` hash (§3.4) |

**What stays** — the daemons' own work, untouched in place:

- **zend:** the chat and tool loop (`session.rs`, `tools.rs`, `tool_round.rs`,
  `tool_def.rs`), ingest (`repo_scan/`, `code_read/`, `branch_ingest`, `origin_watch`,
  `ingest_worker.rs`, `ingest_backlog.rs`), passthrough, titler, calibration, its
  projection (`src/prompts/projection.yaml`), its secrets, its tool-mode policy
  (`access::allowed_modes`, `default_mode`), its UI files (`zend/web/` minus
  `common/` — on disk, where they are), and
  its own routes (`/v1/chat/completions`, `/v1/conversations/*`, `/v1/repo_map`,
  `/v1/models`).
- **npcd:** the world, characters and their runtime (`engine/` — with
  `engine/mind.rs`, `engine/mission_gen/`, `engine/guardian/`, `engine/watcher.rs` —
  `effector/`, `sim/`, `world/`, `mind/`, `npcs.rs`, `lifegen/`), its act catalog, its
  guests, its console (`web/content/npcd/`), and its own routes.

Each daemon's `Reports` implementation (§3.1) is the thin adapter between its own
state and the host's shared endpoints, and lives in the daemon.

**The crates and the three binaries:**

```
candle-conversation   web
        ▲              ▲
        └──── host ────┘
             ▲    ▲
          zend    npcd          (lib + bin each)
             ▲    ▲
             tokerad            (bin)
```

| Binary | Crate | `main` |
|---|---|---|
| `zend` | `zend/` (`src/main.rs` over `src/lib.rs`, as now) | parses zend's flags; `host::run([zend::tenant(cfg)])` |
| `npcd` | `npcd/` (`src/main.rs` over `src/lib.rs`, as now) | parses npcd's flags; `host::run([npcd::tenant(cfg)])` |
| `tokerad` | `tokerad/` (new, bin only) | reads `tokerad.yaml` — zend's config (workspace, …) and npcd's (mind, data, …) under one key each; `host::run([zend::tenant(z), npcd::tenant(n)])` |

`host` depends on neither daemon, so `zend` and `npcd` each build without the other;
only `tokerad` links both. All three are workspace members, so
`cargo build --release` produces all three, and `cargo build -p zend` /
`-p npcd` / `-p tokerad` each build one. The flags each daemon has today stay its own,
parsed into its config struct; `tokerad.yaml` holds the same structs, so a setting
means the same thing whichever binary reads it.

---

## 4. A substrate is bound to its model

### 4.1 Model identity

```rust
/// What a substrate's model-derived state belongs to.
pub struct ModelKey {
    pub arch: String,          // "qwen4exp", "qwen35hybrid", …
    pub weights: [u8; 32],     // SHA-256 of the GGUF header: metadata + tensor infos
    pub tokenizer: [u8; 32],   // SHA-256 of tokenizer.json (already recorded)
    pub kv_layers: u16,
    pub kv_heads: u16,
    pub head_dim: u16,
    pub rope: RopeKey,         // base, scaling kind and factor, min YaRN factor
}
```

**Weights identity.** The GGUF header — the metadata and the tensor-info table —
changes with any quant, merge, fine-tune or recipe, and reading it costs nothing next
to loading the weights. Its SHA-256 plus the file length identifies the weights
without hashing 125 GB. Prepared artifacts already carry a recipe hash in their name
(`Qwen3.8-Flash-Next-Q4_KOEXP-189dd518216f.gguf`); the header hash covers everything
else.

Everything model-derived in a substrate — KV, recurrent snapshots, auxiliary state,
branch checkpoints, provenance signatures, index pages, content-addressed sections,
token ids — is valid only under this key. Binding the whole substrate to one key is
what makes per-record identity unnecessary.

### 4.2 The record and the check

A `ModelKey` record replaces `ModelSpec`. It carries the key and today's spec text
(for the error message and `substrate_inspect`).

**On open, before any weights load:**

| Substrate holds | Running model | Result |
|---|---|---|
| no model record (a new substrate) | any | write the key; open |
| the same key | — | open |
| a different key | — | **refuse** |

The refusal is an error, not a panic, and it says which tenant, which substrate, the
model the substrate belongs to, the model that was about to run, and what to do:

```
zend: substrate D:\prog\substrate belongs to Qwen3.8-Flash-Next Q4_KOEXP
(arch qwen4exp, weights 189d…), not Qwen3.6-35B-A3B AntiLoop+StyleTune Q4_K_M
(arch qwen35hybrid, weights 3f9a…). Start zend on a new substrate (--working-dir),
or move it aside.
```

This replaces the tokenizer check, which it subsumes: a different tokenizer is a
different key. The identity is computed from the GGUF header and `tokenizer.json`
before the weights are read, so a refused start costs seconds, not a model load.
Unlike today's tokenizer check, which is skipped with a warning when `tokenizer.json`
cannot be read (`builder.rs:1382-1388`), the key is never optional: a model whose key
cannot be computed does not start.

**Existing substrates** carry a `ModelSpec` text and a `Tokenizer` hash but no key.
On first open the host compares what can be compared — arch, repo, filename and the
tokenizer hash against the running model. If they match, it writes the full key and
opens; if not, it refuses as above. From then on the key decides. This is the check
running once against the older record, not a second format reader. It can only see
what `ModelSpec` records: a change confined to what it leaves out — revision, tensor
overrides, adapters, gate donor — passes this one adoption.

### 4.3 What it costs to change model

A daemon moved to a different model needs a substrate written by that model: a new
`--working-dir` for zend (whose ingest layers rebuild from source on first start, as
on any new workspace), a new `--data` for npcd. The conversations, memories and NPC
state in the old substrate stay where they are, readable by the old model.

Carrying a substrate across a model change — regenerating its KV from its stored
text — is a later design; nothing here precludes it. The key added in §4.1 is the
identity it would need.

### 4.4 Where it lives

| Piece | Location |
|---|---|
| `ModelKey`, header hash | `candle-conversation/src/models/identity.rs` |
| `ModelKey` record (replaces `ModelSpec`) | `candle-conversation/src/persistence/record.rs` |
| The check on open | `persistence/mod.rs`, where `set_tokenizer` refuses today (`:1992-2015`) |
| The early check, before weights | the host (§5) reads each tenant's substrate key with `substrate_inspect`'s reader, before it loads anything |

---

## 5. The prescribed model

### 5.1 One model, named once

```rust
// host/src/model.rs
/// The model every tenant runs. Changing it is a code change, and every substrate
/// written under the previous one refuses to open (§4) until it is moved aside.
pub const MODEL: Model = Model::Qwen36_35B_A3B_AntiLoop_StyleTune;
```

This is npcd's `model()` today (`npcd/src/model.rs:54-56`), moved to the host and
made the only model. It is the AntiLoop + StyleTune fine-tune of Qwen3.6-35B-A3B
(`models/qwen36_moe.rs:130-169`): the AntiLoop trunk at Q4_K_M with StyleTune's
`output.weight` and an MTP gate donor, arch `Qwen35Hybrid`, 21.7 GB tiered. It runs on
every machine in the fleet — the expert cache streams what VRAM cannot hold — and
needs no ladder.

What goes:

- zend's `model_choice` ladder (`zend/src/model_choice.rs`) and its `--model` flag.
  zend runs the fine-tune on every machine, including the 72 GB card where it ran
  Flash-Next.
- npcd's own `model()`; it reads the host's.
- `Model::PRESETS` gains `Qwen36_35B_A3B_AntiLoop_StyleTune`, which it lacks today
  (`models/mod.rs:547-564`), so the model is addressable by name everywhere a preset
  is — `substrate_inspect`, error messages, `models.override.yaml`.

`models.override.yaml` stays what it is: a repoint of the one model's repo, file,
revision, size or adapters (`candle-transformers/src/model_overrides.rs`), baked in at
build time by `candle-transformers/build.rs`. An override changes the weights, so it
changes the model key (§4.1), and a substrate written before it refuses to open.

### 5.2 What a tenant still declares

```rust
pub struct Capabilities {
    pub tokens: &'static [&'static str], // must each be a single token
    pub dialect: DialectType,
    pub min_context: usize,
}
```

- **zend:** single-token `<tool_call>`, `<think>`, `</think>`; the Qwen3.5 dialect.
  Today only the think tokens are fatal (`session.rs:1323-1334`); a `<tool_call>`
  that is not one token compiles an empty tool registry with a warning
  (`engine.rs:1660-1665`), so zend would run with no tools.
- **npcd:** the same three tokens, checked together (`engine/mind.rs:353-362`) — but
  its callers log the failure and fall back to free decoding (`:527-565`); the
  Qwen3.5 dialect, which its stencils and prompts assume (`engine/act.rs:338`,
  `engine/mind.rs:855`).

The host checks both against `MODEL` at start and refuses with the tenant's name and
the missing capability. Today both pass; the check exists so a future change of
`MODEL` fails at start, where today it degrades both daemons quietly.

### 5.3 What changing to the fine-tune costs zend

zend's live substrate (`D:\prog\substrate`) was written by Flash-Next, so under the
binding rule it refuses to open with the fine-tune. zend starts on a new substrate:

- the ingest layers (`repo_map`, `code_reading`, the analyses) rebuild from source on
  first start, as on any new workspace;
- the chat history in the old substrate stays where it is, readable by Flash-Next.

On the 72 GB card zend also trades Flash-Next's quality for the fine-tune's. Its
throughput moves both ways (forward gates on this card, 2026-10-10,
`docs/results/sweep_rtx_pro_5000_72gb_2026-10-10_post_merge.md`):

| | Flash-Next | the fine-tune (Precision) |
|---|---:|---:|
| ×1 prefill / decode, t/s | 4,701.4 / 294.0 (warm, speculative) | 10,255.0 / 250.9 |
| widest prefill / decode, t/s (aggregate) | 6,821.8 / 1,134.9 at ×16 | 17,282.0 / 2,709.0 at ×64 C10 |

Prefill is 2.2–2.5× faster, and the widest aggregate decode 2.4× (at a wider ladder
top: ×64 against ×16). At one session the fine-tune decodes about 15% under
Flash-Next's figure — but that figure is speculative on the gate's story rewrite, an
easy task for the drafter; on free-text requests Flash-Next reads 156.6 t/s
(`docs/performance.md` §3.7, note ²), and a single chat's rate depends on which of
the two its turns resemble.

---

## 6. Plan

Each phase lands green on its own and is useful on its own.

| # | Phase | Delivers | Proved by |
|---|---|---|---|
| 1 | **Model binding** (§4) | the §1.2 defects fixed in every daemon today: a model change refuses to start instead of injecting foreign KV or tombstoning every conversation; npcd passes its schema's corrupt-turn policies to the engine | unit: the `ModelKey` record round-trips as raw bytes; the header hash of a fixture GGUF is a fixed value; open with the same key opens, a different key refuses with the message, a first open of an older substrate adopts the key when spec and tokenizer match and refuses when they do not; an unreadable `tokenizer.json` refuses to start. Integration: a substrate written by Llama-3.2-3B refuses a second checkpoint with the same tokenizer, and no chunk is read. npcd: a corrupt turn on a `drop_turn` layer drops that turn and keeps the timeline |
| 2 | **Engine core / attachments + `SubstrateId`** (§3.2) | two substrates in one engine, each with persistence, reload, cancel, toggles and corrupt-turn policies; the one-writer rule enforced | unit: two attachments minting the same `TimelineId`, and holding the same section, coexist in every scheduler map; cancel on one does not reach the other; a second writable open of a held substrate — in-process or from another process — is refused. Probe: `kv_fragmentation` with two attachments — both stories pass, both persistence threads drain |
| 3 | **Host + tenants, one server** (§3.1, §3.3, §3.4, §3.7) | the `host` crate; the common code of §3.7 moved into it; `zend`, `npcd` and `tokerad` built from the workspace, the first two running through `host::run([tenant])` on the host's `web::Builder` — tenant APIs merged, UIs by host name, `/common/` once | every existing zend and npcd test; the GUI harness; `cargo build -p zend`, `-p npcd`, `-p tokerad` each green alone; unit: a path two tenants register refuses to start; `/` serves zend's UI for `code.tokera.com` and a bare IP and npcd's for `bot.tokera.com`; a file rewritten in a content directory is served changed on the next request and changes `build`, with the process untouched; a missing content directory refuses to start; an npcd turn's thinking mode is the one npcd chose, under the core's `thinking(true)`; a live restart of each |
| 4 | **Shared endpoints** (§3.5) | the overlaps served once as their superset; one identity rule, one role table, host accounts; both UIs on the chosen field meanings | unit per endpoint with raw expected bodies — both tenants' sections present, shared parts once, `timeline` a string, signed-out `/v1/me` a 200; `/ws/logs` neither drops nor repeats a line across replay; a header from a non-gateway peer is not believed. GUI: zend's status overlay, substrate page and perf page; npcd's console sign-in, substrate, tools and system pages |
| 5 | **The prescribed model** (§5) | `host::MODEL` for every tenant; zend's ladder and `--model` removed; the fine-tune in `PRESETS` | unit: `MODEL` passes both tenants' capabilities; a capability `MODEL` lacks is refused with the tenant's name. Live: zend on a new substrate under the fine-tune — ingest rebuilds, chat works |
| 6 | **zend outranks npcd** (§3.6) | priority ordered tenant first, band second | unit: a zend `Low` sequence with work pauses an npcd `High` one, and the pause lifts at `COOLDOWN`; an npcd `High` decode does not pause or throttle a zend `Low` prompt; a zend decode throttles an npcd prompt by its band's ratio |
| 7 | **`tokerad` on `.5`** | `tokerad` deployed: both daemons in one process on `:8081` / `:8444`; npcd's mind and data on `.5` — its substrate already written by the fine-tune, so it opens as is; the gateway points `bot.tokera.com` at `:8444` beside `code.tokera.com` | live: both GUIs through the gateway, both on the LAN, a refused start when a substrate's model is swapped |

Phase 1 is worth doing first and on its own: it closes a live corruption path in both
daemons regardless of whether the host is built.
