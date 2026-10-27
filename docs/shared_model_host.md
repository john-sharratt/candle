# One process, two daemons: a shared model host for zend and npcd

**Status:** phases 1–6 built (§6); phase 7, `motherd` deployed on `.5`, not yet
done. **Scope:** a single binary that serves zend and npcd together — one loaded
model, one conversation engine, a separate substrate each. The model is prescribed:
one fine-tune, named in one place, for every tenant. A substrate is bound to the model
that wrote it: a daemon whose substrate was written by a different model refuses to
start.

§1 describes the code as it stood before the host was built, and is kept because the
decisions in §2 answer it; §3 onward describes the code as built.

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
colliding ids. A `SectionId` is its schema's own — allocated in order as the schema is
built — so two schemas each have a section `7` that are different sections; it is a
section's **stream id** that is the hash of its content.

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
   flags, each `main` becoming `host::run([tenant])`; `motherd` is the same call with
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
5. **Scheduler keys stay unique across substrates.** Every substrate mints its
   timelines from one process-wide allocator, and attaching a substrate whose stored
   timeline ids overlap an attached one's is refused — so a bare `TimelineId`,
   `TurnKey` or `StreamId` names one thing in the process, and every scheduler-wide
   map stays keyed as it is (§3.2). A `SectionId` is content-addressed and shared on
   purpose: under one model, one section's KV is one section's KV whoever holds it.
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
   apps at the same `/`, are told apart by host name. **Each binary carries its UIs
   compiled in**, so one downloaded file serves its own pages; **the gateway serves the
   same directories from disk**, so editing a page needs no rebuild and no restart.
9. **zend and npcd are equal.** Priority is the band alone, whichever tenant the work
   belongs to: either daemon's live conversation outranks either daemon's ingest, and
   the two daemons' ingests run side by side.

---

## 3. The host and the tenants

### 3.1 The tenant contract

```rust
// host/src/tenant.rs
pub trait Tenant: Send + 'static {
    /// A short name: log attribution, status key, route owner. "zend", "npcd".
    fn name(&self) -> &'static str;

    /// What this tenant needs of the model — checked against the prescribed one (§5).
    fn requires(&self) -> Capabilities;

    /// The directory its substrate lives in. Two tenants naming one directory refuse
    /// the start before either opens it.
    fn substrate_dir(&self) -> PathBuf;

    /// Its web UI, chosen by host name (§3.4).
    fn ui(&self) -> TenantUi;

    /// Start: build state, start background work — the engine load among it — and
    /// return what the host serves. The engine arrives through `ctx.engine`.
    fn start(self: Box<Self>, ctx: TenantContext) -> BoxFuture<'static, anyhow::Result<Running>>;
}

pub struct TenantContext {
    pub engine: EngineSource,         // where its engine comes from (§3.2)
    pub log: Arc<LogBus>,             // the host's one bus
    pub roles: Roles,                 // the estate's role table (§3.5)
    pub telemetry: Arc<Telemetry>,    // for a tenant's own columns (`add_source`)
}

pub struct Running {
    pub routes: Routes,               // its own routes and their table (`host::guard`)
    pub reports: Arc<dyn Reports>,    // what the shared endpoints report about it (§3.5)
    pub stop: Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>, // drain
}

pub struct TenantUi {
    pub hosts: Vec<String>,           // "code.tokera.com", "zend.localhost"
    pub content: PathBuf,             // its UI directory on disk, read per request
    pub default: bool,                // serves a request whose Host names no tenant
}

/// The tenant's share of the host's shared endpoints. Each method answers for this
/// tenant only; the host assembles the superset.
pub trait Reports: Send + Sync {
    fn status(&self) -> BoxFuture<'_, TenantStatus>;              // its load, readiness, extras
    fn tools(&self) -> BoxFuture<'_, (Vec<Value>, Value)>;        // its catalog, and facts about it
    fn calibrate(&self) -> BoxFuture<'_, Result<Value, (u16, Value)>>;
    fn layers(&self) -> BoxFuture<'_, Vec<Value>>;                // its schema's layers
    fn substrate(&self) -> BoxFuture<'_, Value>;                  // its line in the overview
    fn me<'a>(&'a self, id: Option<&'a Identity>, role: Role) -> BoxFuture<'a, MeShare>;
}
```

The tenant does not declare its `CoreRequests`: they are what its own builder holds,
and they travel through the engine slot when it asks for its engine (§3.2). A tenant's
telemetry columns are a `Telemetry::add_source` it installs at start rather than a
`Reports` method, so every column shares the sampler's one time axis.

What moved out of each `main` into the host: `#[tokio::main]`, the panic hook, the
tracing subscriber, the Ctrl-C handler, the self-heal watchdog,
`set_gpu_span_period`, the model download (`host::download`), and the
`std::process::exit` calls — a tenant that cannot load calls `host::process::fatal`,
and otherwise the host exits once, after every tenant has stopped. What stays in the
tenant: everything else — zend's loader thread, calibration, titler, ingest worker,
OriginWatch; npcd's runtime, character loops, metronomes, mission generator.

**The diff to each daemon is the move, not a rewrite.** zend's `InferenceState::load`
calls `engine_source.ticket().engine_with_progress(&mut builder, &device, …)` where it
called `builder.engine_with_progress(&device, …)`; npcd's `runtime::load` does the
same. Their flags became a config struct each — `zend::cli::ZendArgs`,
`npcd::tenant::NpcdArgs` — whose `resolve()` checks them before anything loads, and
their `main.rs` shrank to parsing it. The process's flags are `host::HostArgs`.

Logging: the host owns the one subscriber, and every line on its bus names the tenant
its target's crate belongs to (`tenant`, empty for the engine and the host), with a
`seq`. Each UI filters the one stream to its own lines and the engine's, so zend's log
pane shows zend's and npcd's shows npcd's; the host's log file has them all. It is
`<binary>.log` inside the highest-ranked tenant's substrate directory, beside the
self-heal marker and attempt count — the process's state, in the one directory every
run of that binary shares.

### 3.2 Engine core and attachments

```
EngineCore (Arc, one per process)              candle-conversation/src/engine.rs
├─ scheduler thread + Sender<SchedulerRequest>
├─ model (moved to the scheduler), tokenizer, device, guests
├─ model identity: ModelKey (§4.1)
└─ attachments: the live attachment ids

ConversationEngine (one per tenant) = Arc<EngineCore> + one attachment
├─ conversation: Conversation       (this substrate; carries its CancelToken,
│                                    its TenantRank and its layer toggles)
├─ persist_thread: PersistenceThread
└─ reload status
```

- **Factories** keep their signatures and send the attachment's conversation, not a
  single engine-wide one.
- **Persistence:** one `PersistenceThread` per attachment. Each drives its own
  substrate's hot→warm→cold; the scheduler's relief already walks distinct substrates.
- **Reload:** `reconstruct_substrate` runs per attachment, on attach
  (`SchedulerRequest::AttachSubstrate`).
- **Cancel:** each substrate's `Conversation` carries its own `CancelToken`; the
  process-wide `INGEST_CANCEL` is gone. zend's pools read zend's token
  (`zend::ingest_cancel`), and cancelling one tenant's ingest never reaches another's.
- **Layer toggles** live on the substrate they toggle, so a toggle in one tenant never
  reaches another's layer of the same name.
- **Shutdown:** `ConversationEngine::shutdown` detaches — release its slots
  (`SchedulerRequest::DetachSubstrate`), then flush and stop its persistence thread.
  The core stops when the last attachment detaches. Every change to the attachment
  list — attach, detach, the stop — happens under one lock together with the
  scheduler request that announces it (`engine_attachments::Attachments`), so a
  detach is always queued ahead of the last attachment's `Shutdown`, two attaches
  cannot both pass the checks, and nothing attaches to a stopped core.
- **A detached tenant's leftover handles cannot reach another's slots.** A
  `Sequence` of the detached substrate may outlive the detach on another thread, and
  slot ids recycle. So each freed slot id stays reserved in the session until its own
  handle frees it, and every request naming a reserved slot or the detached substrate
  is dropped unanswered (`scheduler::detached`) — its caller sees the channel close,
  as on a stopped scheduler. Detach also drops the substrate's queued normalization
  warm-ups, its pending section quantizes and its hold on shared section pages.
- **One writable handle per substrate directory** is enforced: a writable open takes
  an exclusive lock on `substrate/LOCK` (`persistence::writer_lock`), so a second
  writer — another process, or another handle in this one — is refused, naming the
  directory. (An OS file lock does not report who holds it, so the refusal names the
  substrate, not the holder.) The lock is held until the handle drops or the engine
  shuts down: `shutdown` ends by committing and releasing it
  (`SubstratePersistence::release_writer`), after which the old handle refuses every
  write. So a successor can open the workspace while a shut-down engine is still
  held. The host also refuses two tenants naming one directory before either opens
  it.
- **Per-substrate builder settings** stay on the tenant's own builder: the system
  prompt, the substrate, read-only, and the corrupt-turn policies.

**Keys stay unique, not wider.** Every `Conversation` mints its timelines from one
process-wide allocator (`projection::ids::process_timeline_allocator`), and attaching
a substrate whose stored timeline ids overlap an attached one's is refused. So a bare
`TimelineId` — and the `TurnKey`s and `StreamId`s built on it — names one thing in
the process, and `turn_positional`, `ingest_timelines`, `timeline_projections`, the
gallery arena's residency map and the relief keep-sets stay keyed as they were.

**Section tables key by content, not by `SectionId`.** A `SectionId` is its schema's
own, so the scheduler's per-section tables never key by it alone: the index pages are
held by the section's content-addressed stream id, each page recording which
substrates hold it and going with the last (`scheduler::section_pages`); the name
cache is by stream id too; and a pending section quantize carries the substrate that
sealed it, so a drain takes only that substrate's. Two substrates under one model
holding the same section text share its page; two schemas' equal ids never meet.

This replaces the `(substrate, …)` re-keying the first draft of this design proposed
— the same guarantee, keyed by what is actually unique.

**Where a tenant gets its engine: the engine slot** (`candle_conversation::EngineSlot`).
Builders never cross threads — each tenant's builder holds its own substrate, system
prompt and policies, and stays with the tenant's loader. So each tenant's loader
reaches the slot where it would otherwise build an engine: it deposits its
`CoreRequests`; the highest-ranked tenant waits until every tenant has arrived (or
withdrawn — dropping an unused `EngineTicket` does that), merges the requests by §3.3's
rules, and builds the engine on its own thread with its own progress reporting; every
other tenant then binds its own substrate and attaches it through an `EngineAttacher`.
An attaching tenant checks its model against the engine's before it touches its
substrate, and binds the substrate under the **engine's** RoPE schedule — the merged
one (§3.3) that will rotate every key it writes — not its own builder's
(`ModelBuilder::attach_to`). A slot of one builds at once, so a daemon run alone
behaves as it did before. Each tenant's `Capabilities` are checked against the
engine's model when it gets its engine.

The slot knows its tenants by name, so a loader waiting there is told whom it waits
for (`EngineTicket::engine_with_progress`'s `waiting`), and told nobody once it can go
on. Each tenant's load reports that as `waiting_for` in its `/v1/status` line, and its
loading page ends its step list on "Waiting for npcd" while npcd is still on its way.

### 3.3 Core settings two tenants both set

Some builder settings are per-model or per-scheduler, and the two daemons set them
differently today:

| Setting | zend | npcd | Rule |
|---|---|---|---|
| `prefill_pass_tokens` | sized from the card (`prefill_width`: VRAM × link) | 8192 (`runtime.rs:163`) | max, then clamped to the card's width |
| `min_rope_factor` | 2 (schema `rope.min_yarn_factor`, `session.rs:1253`) | unset | max — RoPE schedule is the model's |
| `qsa_selection_budget`, `qsa_strata` | CLI (`session.rs:1249-1250`) | unset | zend's; npcd sets none |
| `max_response_tokens` | 65,536 (`session.rs:349`) | 16,384 (the builder default) | max |
| guests (`<data>/guests.yaml`) | none | npcd's, when the file exists (`runtime.rs:3757`) | union |
| `max_concurrent` | the builder's | the builder's | sum — each tenant's sessions count against one scheduler |
| `thinking` | `true` | unset | `true` if any tenant asks |

`CoreRequests` carries these (`candle_conversation::engine_slot::CoreRequests`); the
tenant that builds the engine merges every tenant's by the rule column before it
builds (§3.2), and the merged values are what the engine logs at start.

The prefill width is a request against a ceiling: the merge picks what the tenants
want, and the builder clamps it to what the card carries (`prefill_width`, by its
VRAM and measured host→device link). Past that width each window's transient tier is
bought from the expert zone and decode pays for it after the window — on the RTX PRO
5000 npcd's 8,192 over the card's 3,072 conceded ~8 GB of experts every window.

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

| UI | Hosts | Content (compiled in) | Default |
|---|---|---|---|
| zend | `code.tokera.com`, `zend.localhost` | `zend/web/` (`zend::tenant::WEB`) | **yes** |
| npcd | `bot.tokera.com`, `npcd.localhost` | `web/content/npcd/` (`npcd::tenant::CONSOLE`) | no |

Run alone, a daemon's UI is the default whatever its row says. The host answers `/v1`,
`/ws`, `/common` and `/brand` from its one router on every name; everything else is
the UI the name picks (`host::server::site_table`).

**Served while the tenants start** (`host::boot`). That router is the tenants' routes
merged with the host's, so it exists only once every tenant's `start` has returned —
and a start can be long: npcd opens its substrate in it, and a store written before
identities is rewritten whole there, once. So the host binds first, on the same ports,
with the UIs, the shared front-end and a `/v1/status` built from each tenant's
`StartLine` (`host::starting`, handed over in `TenantContext::starting`) until its
start returns and from its own reports after. npcd reports its substrate open there as
the `opening` step, counting records through a stamping rewrite
(`SubstratePersistence::open_in_with_substrate_and_sink`'s `OpenProgress`). Once every
tenant has started, the boot server finishes what it is answering and lets the ports
go, and the real server binds them.

The shared front-end library, `web/content/common/`, is served once, at `/common/`,
for both, and its estate marks at `/brand/` — where the estate switcher addresses them
on every host (`common/lib/estate.js`). zend's copies (`zend/web/common/`,
`zend/web/brand/`) are gone; npcd's console, which served the library at its root
(`/base.css`, `/lib/*.js`, through a second root), loads it from `/common/` like zend's
does — every shared module by one URL, so a module is never loaded twice under two
spellings. The two tests that kept the copies in step went with zend's copy;
`npcd/src/console.rs` resolves the console's module graph by URL, as a browser does,
and `web/src/site/tokera/page.rs` checks that every asset zend's pages name exists in
`zend/web/` or the shared library.

**Compiled into the binary; served from disk by the gateway.** Two places serve the
same files, for two different jobs.

- **The daemon embeds its UIs** (`include_dir!`, as `web::Roots::embedded`): zend's
  `zend/web`, npcd's `web/content/npcd`, and the host's shared library
  `web/content/common` (`host::ui::COMMON`). A shipped binary therefore serves its own
  pages with no checkout beside it — download `zend`, run it, open it. Each crate's
  build script reruns when a file under its embedded directory changes
  (`embed-watch`), since `include_dir!` tells cargo nothing about the files it reads.
  There is no switch to point a daemon at a directory instead: one way to serve a UI
  from a binary.
- **The gateway serves the same directories from disk** (`web/web.yaml`'s `npcd` and
  `zend` sites, `Roots::disk`), read per request, and forwards only `/v1` and `/ws`.
  Saving a page is live on the next refresh with nothing rebuilt; the daemons catch up
  when they are next built.
- Files go through `web::asset::respond` either way: a strong ETag over the content,
  `If-None-Match` answered `304`, gzip, and `max-age` from `cache.short_secs` (60 s)
  for HTML, JS and CSS.
- **Reload on change.** Both UIs reload themselves when `/v1/status`'s `build`
  changes (zend `index.html`; npcd `common/lib/build.js`, polled every 5 s). `build`
  is a hash over every compiled-in UI file — its path and its bytes — and the
  binary's version, taken once at startup (`host::ui::build_id`): the content cannot
  change while the process runs, so a new id is a new binary with a different UI.
- **Cluster upgrades carry no UI.** A build is its executables
  (`cluster::upgrade::artifacts`); the UI each serves is inside it, so a member
  fetches, keeps and swaps one file (`docs/replicated_substrate.md` §3.9).

**Ports.** The host serves the one router on `192.168.0.5:8081` (plain) and
`192.168.0.5:8444` (TLS, a certificate made at startup — `web::tls::self_signed`), the
two ports zend has today. npcd moves from `.6` (`0.0.0.0:8081`) onto them.

- **A request by IP** gets zend's UI — what `http://192.168.0.5:8081/` shows on the
  LAN today — and the whole API. npcd's console is reached by its names:
  `bot.tokera.com` through the gateway, `npcd.localhost` on the box itself. The site
  table names exactly the hosts in the table above (`host::server::site_table`); a LAN
  machine that wants npcd's console directly sends one of them as its `Host` (a hosts
  entry mapping `bot.tokera.com` to `.5`).
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

Before the host, the two daemons trusted the gateway differently: zend believed
`x-tokera-*` headers only from a trusted peer, npcd from anyone its port reached
(`web::Builder::behind_gateway`). zend had no route guards; npcd graded its routes by
role, all but `/v1/local/*`, the effector API, which authenticates by device token and
keeps doing so.

- **Identity:** zend's rule, for both — headers are believed from loopback, the bound
  address and each `--gateway`, never from anyone else. It is the stricter, and npcd's
  "behind the gateway" was an assertion about the network that nothing checked. It is
  enforced once, in the server: `web::Builder::trust(Gateways)` clears `x-tokera-*`
  from every request whose **socket peer** is not trusted, before any handler sees it.
  `--local-signin` is the one stand-in: for a genuine loopback peer that sent no
  identity, `host::access::LocalSignin` writes it as the gateway would have, so every
  tenant reads it the same way. It is refused beside `--gateway` — a daemon with a
  gateway in front has one to sign people in — and, like loopback trust itself, it
  extends to anything on the box that relays remote traffic onto loopback.
- **Roles:** one estate table, `host/roles.yaml`, merging `zend/zend.roles.yaml` and
  the `roles:` block of `npcd/npcd.web.yaml` (both deleted).
- **Guards:** every route — the host's and each tenant's — is registered through
  `host::guard::Api` (npcd's guard, moved), which cannot add a route without a role,
  and each shared endpoint carries the stricter of the two daemons' guards — with two
  deliberate exceptions, both open to anyone: `/v1/me`, whose job is to tell a
  signed-out caller that it is signed out (a guard would answer that with a 401 the UI
  then has to interpret), and `/v1/tools`, the catalog zend's chat client reads to
  build its tools dial before anyone signs in. `/v1/profile`, which resets the
  profiler on read, is admin-only. Consequence: zend's `/perf`, `/logs` and `/substrate` pages, which worked for anyone
  on the LAN, need a signed-in role — through the gateway, or `--local-signin` on
  loopback. zend's chat and conversation routes stay open: an OpenAI client sends no
  identity, and the tools dial resolves the caller's role itself.
- **Accounts stay npcd's.** They are built on npcd's authored-document `Registry`, and
  zend has no use for an account record today — so by §3.7's rule they are not common
  code. `/v1/me` is still served once: the host states the role, and each tenant
  contributes its share (`Reports::me`) — npcd its account record, created on first
  sight, at the top level; zend its tools modes under `tenants.zend`. npcd's account
  routes (`/v1/me/profile*`, `/v1/me/unique-name`) stay its own.

#### The endpoints

| Shared endpoint | Replaces | Superset |
|---|---|---|
| `GET /v1/status` | zend `/v1/status`; npcd `/v1/status` | `state` (`ready` once every tenant is), `started_at_ms`, `build`, `model{…}`, `engine_connected` (the model is loaded: some tenant is past its `model` step, which is every tenant's first), `loading` (the highest-ranked load still running — the one loading the model while there is one), `detail` (that load as one line, `label — detail`), `tenants.<name>{state, loading, maintenance}` plus zend's `ingest_backlog`. `loading` is npcd's richer shape for every load (`npcd/src/engine/loading.rs:95`): `current`/`completed` are **step ids**, `label` the current step's text and `completed_labels` the completed steps' one for one, `detail` the item in hand, `elapsed_ms`, `prefill_tokens`, `prefill_tps` — where zend's `current` is a label and `completed` a list of labels. `build` is **one hash over the UI content on disk and the binary's version**, recomputed when a content file changes (§3.4) — zend's meaning, so an open page of either UI reloads on a UI edit or a rebuild, where npcd's console today reloads only on a version bump. npcd gains `maintenance` for its own substrate |
| `GET /v1/me` | zend `/v1/me`; npcd `/v1/me` | `role`, `email`, `display`, `avatar_url`, npcd's account fields (`user_id`, `unique_name`, `profile`, …) at the top level, and `tenants.zend{tool_modes, default_tools}`. A signed-out caller gets **200** with `role: "unauthenticated"` and no account — zend's behaviour; npcd's console reads the role instead of a 401 |
| `GET /v1/memory` | zend `/v1/memory`; npcd `/v1/memory` | the shape both already share; `report` is the shared engine's — populated for both, where npcd's is hard-coded null today (`npcd/src/telemetry/memory.rs:72-73`) |
| `GET /v1/telemetry` | zend `/v1/telemetry`; npcd `/v1/telemetry` | zend's per-wave rows (`waves`, `fmt`, `migrate`, `has_pressure`) and npcd's sampled hour (`gpu`, `model`, `series`, `sample_period_s`, `uptime_s`, and `engine_measured` — whether a wave has run, so a page shows *not measured* rather than zeroes; whether the model is loaded is `/v1/status`'s `engine_connected`), side by side — they are different data, not the same data twice. The engine columns (`decode_tps`, `prefill_tps`, `kv_mib`, `weights_mib`) are read off the shared scheduler's phase rings (`host::telemetry::engine_reading`) — before, npcd's `Telemetry::record` had no caller, so only `prefill_tps` was filled, from the load progress. One `host{total_mib, free_mib, rss_mib}` (npcd's units). A tenant's own columns (npcd's `npcs_active`, `ticks_per_sec`, inbox depths, image queue) stay in `series`, on the one time axis, filled by a source the tenant installs (`Telemetry::add_source`); npcd has no producer for them yet, so they read absent |
| `GET /v1/phases`, `/v1/projections`, `/v1/layers`, `/v1/profile` | zend's | process-wide already — the scheduler's phase rings, the projection tracker, the expert pipeline's layer trace, and the pipeline profiler (`candle-transformers/src/models/profile/pipeline.rs`), which `/v1/profile` resets on read; now visible to npcd's console too |
| `WS /ws/logs` | zend `/ws/logs`; npcd `/ws/logs` | the frame both already use, `{ts, level, target, msg}`, plus `tenant` and a `seq` (zero on a replayed line). The host's one bus, with zend's broader filter (npcd's drops most `candle_conversation` output). Any signed-in caller (`user`): the logs are how anyone running a member sees what broke the cluster. The bus is the log file's tap and keeps no lines of its own: the replay is the file's last 256 KiB, read back from disk, with the subscription taken under the file's lock in the same moment — each line is in the replay or the stream, never both, never neither |
| `GET /v1/logs` | — | the log files searched, every archive then the active file, a line at a time: `grep` (a regex over the whole line), `level` (a floor), `tenant`, `since` (an RFC 3339 prefix; an unstamped continuation line is as old as the line before it), `limit` (the latest matches kept, 200 by default, at most 5000). Answers `{lines, matched, scanned, files}` with the lines as written. `user`, and aggregated (`Merge::Envelope`), so one request searches every member |
| `GET /v1/substrate`, `/v1/substrate/zend/…` | zend `/v1/substrate/*`, `/v1/debug/maintenance`; npcd `/v1/substrate/storage` | the overview lists every attached substrate under `tenants.<name>`: zend's storage, counts and layers; npcd's storage (`open`, `path`, `segment_count`, with `live_chunks` and `dead_ratio`, which npcd carried as always null and now reads from its engine — real for both). The views over one substrate are its tenant's, under its own path: zend's layer, timeline, selection, system-prompt, project and maintenance views under `/v1/substrate/zend/`, npcd's per-character views where they were (`/v1/npc/:nid/substrate/*`, with npcd's ownership check) — they read state only their own daemon holds (zend's projection schema and session, npcd's cast), so a host handler "over any attachment" would be each daemon's code again behind a second interface. One field meaning each: `timeline` is a **string** (a u64 does not survive a JavaScript number — npcd now sends it as one), `turns` is an array with `turn_count` beside it, and a count is always `*_count` (`conversation_count`, `turn_count`, zend's `timeline_count`, `section_count`, `token_count`); a size in tokens beside a thing (a layer's, a conversation's, a section's) is `tokens` |
| `GET /v1/tools`, `POST /v1/tools/calibrate?tenant=<name>` | zend `/v1/substrate/tools`; npcd `/v1/tools`, `/v1/tools/calibrate` | every catalog: `tools[]` carries `tenant` and the union of fields (both: `name`, `description`, `parameters`; zend's `high_risk`; npcd's `category`, `plane`, `modes`, `needs`, `at_named`, `source`, `calibrated`, `examples` as a count), with npcd's top-level `uncalibrated` and `acts_per_turn` under `tenants.npcd`. Calibration names its tenant, and each answers what calibration is for it: zend's runs as it loads, regenerating each case whose definition changed (`409 calibrated_at_load`); an npcd act is calibrated by the examples authored for it (`409 calibrated_by_examples`) |
| `GET /v1/schema/layers` | zend's layer list inside `/v1/substrate`; npcd `/v1/schema/layers` | every tenant's layers, each with `tenant`, the raw schema entry (npcd's shape, passed through from the mind's yaml: `budget`, `groups`, `summary`, `gather_scope`, `decode_priority`, …) and zend's live fields (`conversation_count`, `tokens`, `disabled`). Formatting (`priority` from `budget`, group strings) moves to the UI. zend's own substrate page keeps its formatted layer view inside `/v1/substrate` (`zend/src/schema_layers.rs` builds the raw one) |

What stays a tenant's own: zend's `/v1/models` (the OpenAI listing Continue reads,
naming zend's model ids), `/v1/chat/completions`, `/v1/conversations/*`,
`/v1/repo_map`; everything npcd serves about its world and characters.

**The UIs change where the meaning was chosen:** zend's status overlay reads
`loading.label` for its text and `tenants.zend.*` for its backlog and maintenance;
its substrate page uses `/v1/substrate/zend/…` and `/v1/tools`; npcd's console reads
the 200 sign-out, `/common/`, the string `timeline` and `tenants.npcd.*`.

### 3.6 zend and npcd are equal

One scheduler carries both daemons' work. The tenants are **equal**: a sequence's
priority is its layer's band alone, whichever daemon it belongs to:

```
High (either daemon's live conversation) > Normal > Low (either daemon's ingest)
```

- **Bands share the device** (`scheduler/band.rs`); none is held out of a forward. While
  either daemon has a conversation running — a chat turn, a tool round, a character's
  turn — both daemons' ingest decodes ride the same steps and their prefills creep beside
  it at the conversation's ratio (`docs/continuous_fair_waves.md` §3.1). A pause that held
  ingest out for 15 s after every conversation's activity stopped zend's ingest outright
  beside npcd's cast, which never falls idle for that long.
- **The prefill throttle** (`prefill.rs` `throttle_ratio`) uses the same order: work
  yields to the decodes in a higher band, at that band's ratio, whichever daemon
  either belongs to. Two daemons' work in one band shares the wave as one daemon's
  always has.
- **What it means:** zend's and npcd's ingests run side by side, so a long ingest of
  one — zend's at startup, npcd's mind on a fresh substrate — no longer holds the
  other's back for its whole length. Neither daemon's conversations wait on the
  other's ingest.

This was first built tenant first (`zend High > … > npcd Low`), so that while zend had
anything to do — its ingest included — npcd's characters waited. On `.5` that held
npcd's first mind ingest behind zend's hours-long background ingest, and the rule was
dropped: background work of one app is no more urgent than another's, and live
conversations already outrank ingest by band.

`TenantRank` still orders the tenants for one thing that is not priority: which of
them builds the shared engine at start (`engine_slot/rendezvous.rs`) — the host
assigns it from the order tenants are given to `host::run`, zend first.

### 3.7 Where code lives, and the three binaries

**The rule.** Code that both daemons carry, or that is about the process, the engine
or the estate rather than about coding or characters, moves to a common home. Code
only one daemon needs stays in that daemon, in the file it is in today. Each kind of
common code goes where that kind already lives:

| Common home | Holds |
|---|---|
| `host/` (new crate) | the process and everything served about it: tenant contract, startup and shutdown, logging, model loading, the shared endpoints, identity, accounts |
| `candle-conversation` | the engine: core and attachments, the process-wide timeline allocator, model identity and the binding check |
| `web/` | the HTTP server, as now; and the shared front-end, `web/content/common/` |

**What moves** — each a pair of copies today, or one daemon's generalization the
other needs:

| From zend | From npcd | To |
|---|---|---|
| `log_broadcast.rs`, `log_line.rs`, `api/ws_logs.rs` | `logs.rs`, the `/ws/logs` handler in `ops.rs` | `host::logs` (one bus, one frame), `host::endpoints::logs` (one socket handler) |
| `loading.rs` | `engine/loading.rs` | stay each tenant's — the steps are each daemon's own — reported in one shape, `host::loading::LoadingView` (§3.5) |
| `api/status.rs` | the status handler in `ops.rs` | `host::endpoints::status`; zend's maintenance trigger is the host's `maintain` command, for every tenant (`docs/replicated_substrate.md` §12.4) |
| `api/telemetry.rs`, `api/memory.rs`, `api/profile.rs` | `telemetry/` (`mod.rs` sampler, `ring.rs`, `device.rs` NVML, `sample.rs`, `memory.rs`) | `host::telemetry`, `host::endpoints::{telemetry, waves, profile}` |
| `api/me.rs` | the `/v1/me` route in `api.rs` | `host::endpoints::me`; npcd's accounts and profile routes stay npcd's (§3.5) |
| `access.rs` (peer trust, `--local-signin`, role lookup) | `identity.rs` (the refusal), `guard.rs` (route grading) | `web::auth::Gateways` and `web::Builder::trust` (peer trust, in the server); `host::access` (the role table, `--local-signin`); `host::guard` (route grading, the refusal, the path-overlap check). zend keeps its tools-mode policy; npcd keeps `Caller` and `--log-identity` |
| `api/substrate.rs` and the substrate views in `session.rs` | `substrate.rs`, `engine/layers.rs` | the overview is `host::endpoints::catalog`; each daemon's views stay its own (§3.5) |
| the tools-catalog view | the `/v1/tools` view in `engine/pulse.rs` | `host::endpoints::catalog` (each catalog stays in its daemon) |
| `main.rs` process setup: panic hook, tracing init, `ctrl_c.rs` (the Ctrl-C flag reset), `self_heal.rs`, `log_file.rs` | `main.rs` process setup: tracing init, Ctrl-C | `host::process`, `host::ctrl_c`, `host::self_heal`, `host::log_file`, `host::run` |
| `download.rs` (less the DeepSeek fetch, which ran a model no tenant runs), `model_choice.rs` (deleted, §5) | `model.rs`'s `model()` | `host::model` (`MODEL`), `host::download` |
| `web/common/`, `web/brand/` (copies of the shared front-end) | — (served `web/content/common/` at its root) | `web/content/common/`, once — zend's copies and the two sync tests deleted, both UIs load it from `/common/` and `/brand/` |
| the `include_dir!` UI embed and its build hash (`api/mod.rs:43-83`) | the embedded roots, `--content` (`main.rs:48-49`, `:252-266`), the constant `build` (`ops.rs:163`) | the host's disk-served UI layer and its on-disk `build` hash (`host::ui`, `host::server`, §3.4) |

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
             motherd            (bin)
```

| Binary | Crate | `main` |
|---|---|---|
| `zend` | `zend/` (`src/main.rs` over `src/lib.rs`, as now) | parses `HostArgs` and `ZendArgs`; `host::run(HostConfig { …, one_shot: ZendArgs::ONE_SHOT }, vec![zend tenant])` |
| `npcd` | `npcd/` (`src/main.rs` over `src/lib.rs`, as now) | parses `HostArgs` and `NpcdArgs`; `host::run(HostConfig { …, one_shot: NpcdArgs::ONE_SHOT }, vec![npcd tenant])` |
| `motherd` | `motherd/` (new, bin only) | reads `--config` (default `motherd.yaml` in the working directory; the shipped example is `motherd/motherd.yaml`) — the process's settings under `host:`, zend's under `zend:` and npcd's under `npcd:`; `host::run(…, vec![zend tenant, npcd tenant])` |

`host` depends on neither daemon, so `zend` and `npcd` each build without the other;
only `motherd` links both. All three are workspace members, so
`cargo build --release` produces all three, and `cargo build -p zend` /
`-p npcd` / `-p motherd` each build one. The process's flags are the host's in every
binary (`HostArgs`: `--host`, `--port`, `--tls-port`, `--gateway`, `--local-signin`,
`-v`); each daemon's own flags stay its own, parsed into its args struct, and
`motherd.yaml` holds the same structs, so a setting means the same thing whichever
binary reads it — except a **one-shot** flag (a wipe, a forget, a self-check), which
a config file read at every start cannot hold and a self-heal relaunch drops.

---

## 4. A substrate is bound to its model

### 4.1 Model identity

```rust
/// What a substrate's model-derived state belongs to.   models/identity/key.rs
pub struct ModelKey {
    pub arch: String,        // "Qwen35Hybrid", "Qwen4Exp", … — ModelArch's name
    pub repo: String,        // where it was resolved from — a description, not identity
    pub file: String,        // the checkpoint's general.name + ".gguf" — a description too
    pub weights: [u8; 32],   // WeightFiles::digest over every file the model loads from
    pub tokenizer: [u8; 32], // SHA-256 of tokenizer.json
    pub rope: String,        // rope_key: the schedule and the minimum YaRN factor
}
```

Two keys are the same model when `arch`, `weights`, `tokenizer` and `rope` agree
(`ModelKey::same_model`); `repo` and `file` say where it came from, so the same file
from a mirror is the same model.

**Weights identity.** Each weight file's fingerprint (`gguf_fingerprint`) is the
SHA-256 of its GGUF header — the metadata and the tensor-info table, which change with
any quant, merge, fine-tune or recipe — its length, and sixteen 64 KiB windows sampled
across its tensor data, so a file rewritten under an unchanged header is caught too.
`WeightFiles::digest` combines every file the model loads from — the primary GGUF,
tensor overrides, the gate donor, adapters. None of it reads the whole file: 125 GB is
identified in seconds.

**RoPE identity.** Keys are stored rotated, so the schedule is part of the model: a
different preset, rung table or minimum YaRN factor makes every stored key wrong while
every shape still matches. `rope_key` writes the schedule's floats as bit patterns, so
two schedules are the same key exactly when they rotate identically. A shared engine's
minimum factor is every tenant's merged (§3.3), and an attaching tenant binds under
the engine's (§3.2).

Everything model-derived in a substrate — KV, recurrent snapshots, auxiliary state,
branch checkpoints, provenance signatures, index pages, content-addressed sections,
token ids — is valid only under this key. Binding the whole substrate to one key is
what makes per-record identity unnecessary.

### 4.2 The record and the check

A `ModelKey` record replaces `ModelSpec`. It carries the key and today's spec text
(for the error message and `substrate_inspect`).

**On open, before any weights load:**

| Substrate holds | Running model | Result (`persistence::model_binding`) |
|---|---|---|
| no model record (a new substrate) | any | write the key; open (`New`) |
| the same key | — | open (`Same`) |
| the same model under another description — a moved repo, a renamed file | — | rewrite the record; open (`Redescribed`) |
| a different model | — | **refuse** |

The refusal is an error, not a panic. The persistence layer names the substrate and
both models — each by file, architecture, weights, tokenizer and RoPE schedule, so two
keys that differ are never printed alike — and the daemon's `host::process::fatal`
names the tenant:

```
error: zend: … the substrate at D:\prog\substrate belongs to
Qwen3.8-Flash-Next-Q4_KOEXP-189dd518216f.gguf (arch Qwen4Exp, weights 189dd518…,
tokenizer 4e1f0c2a…, rope …), not Qwen3.6-35B-A3B-AntiLoop-StyleTune-Q4_K_M.gguf
(arch Qwen35Hybrid, weights 3f9a61d0…, tokenizer 4e1f0c2a…, rope …). Everything
model-derived in it — K/V, recurrent state, token ids — was computed by that model
and is wrong under any other. Start on a new substrate, or move this one aside.
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
overrides, adapters, gate donor — passes this one adoption. The filename on both sides
is the checkpoint's own name — its GGUF `general.name` with `.gguf`, which the builder
puts in the spec once it reads the pack, and which builds before keys recorded too —
not the name of the file on disk.

**The check comes before the store is rewritten.** A substrate written before record
identities is adopted by its first writable open (`docs/replicated_substrate.md` §6.2),
and that open first checks the model it is opening the store for
(`persistence::model_binding::Admit`), so a refused store is left as it was found. A
host that opens its store before its pack is resolved — npcd, whose cast is read at
start while the model may still be downloading — passes what its spec knows: the
architecture, and against a record written before keys the repository. The key decides
the rest when the store is bound.

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
| `ModelKey`, `ModelIdentity`, the GGUF fingerprint, `WeightFiles`, `rope_key` | `candle-conversation/src/models/identity/` (`key.rs`, `fingerprint.rs`, `weight_files.rs`, `rope_key.rs`) |
| The binding decision and `BoundSubstrate` | `candle-conversation/src/persistence/model_binding.rs` |
| The check before weights | each tenant's own load: `ModelBuilder::bind_substrate`, before `engine_from_prepared` loads a weight; for an attaching tenant, `ModelBuilder::attach_to` checks the model against the engine's before it binds. A refused start costs the identity read — seconds — and no model load. (npcd replays its character store from the substrate at its start, before its load binds it: a cheap read, and a refused bind still stops the start.) |

---

## 5. The prescribed model

### 5.1 One model, named once

```rust
// host/src/model.rs
/// The model every tenant runs. Changing it is a code change, and every substrate
/// written under the previous one refuses to open (§4) until it is moved aside.
pub const MODEL: Model = Model::Qwen38_FlashNext;
```

It is Qwen3.8-Flash-Next (`models/qwen38_flash_next.rs`), arch `Qwen4Exp`: a `Q8_0`
trunk prepared on each machine, with the routed experts at the card's rung — `Q3_KO`
under 64 GiB, `Q4_KO` above. It runs on every machine in the fleet — the expert cache
streams what VRAM cannot hold.

**A rung is not another model.** Every rung is built from the same release, so the
preset names the release and loads whichever rung's pack the card calls for, and the
key's weights digest is the release's — its trunk and draft-head sources
(`quantized_qwen38_moe::engine_model_digest`) — rather than the pack's. A substrate
written on the 16 GB laptop opens on the 72 GB card. A new release, a new pin, is a
new model.

It replaced the AntiLoop + StyleTune fine-tune of Qwen3.6-35B-A3B, which a substrate
written before the change still records, and refuses to open under (§4).

What goes:

- zend's `model_choice` ladder (`zend/src/model_choice.rs`) and its `--model` flag.
  The only choice a card makes is its Flash-Next rung, and that is not a model choice.
- npcd's own `model()`; it reads the host's.
- One Flash-Next preset in `Model::PRESETS` in place of one per rung, so the model is
  addressable by one name everywhere a preset is — `substrate_inspect`, error messages,
  `models.override.yaml`.

`models.override.yaml` stays what it is: a repoint of the one model's repo, file,
revision, size or adapters (`candle-transformers/src/model_overrides.rs`), baked in at
build time by `candle-transformers/build.rs`. An override changes the weights, so it
changes the model key (§4.1), and a substrate written before it refuses to open.

### 5.2 What a tenant still declares

```rust
// candle_conversation::engine_slot::Capabilities
pub struct Capabilities {
    pub tokens: &'static [&'static str], // must each be a single token
    pub dialect: DialectType,
}
```

(A minimum context was in the first draft. Neither tenant has one to declare — the
model's context is far past what either asks — so it is not a field until one does.)

- **zend:** single-token `<tool_call>`, `<think>`, `</think>`; the Qwen3.5 dialect.
  Today only the think tokens are fatal (`session.rs:1323-1334`); a `<tool_call>`
  that is not one token compiles an empty tool registry with a warning
  (`engine.rs:1660-1665`), so zend would run with no tools.
- **npcd:** the same three tokens, checked together (`engine/mind.rs:353-362`) — but
  its callers log the failure and fall back to free decoding (`:527-565`); the
  Qwen3.5 dialect, which its stencils and prompts assume (`engine/act.rs:338`,
  `engine/mind.rs:855`).

Each tenant's are checked against the engine's model when it takes its engine from
the slot (§3.2), and a lack refuses the start with the tenant's name and every missing
capability. Today both pass; the check exists so a future change of `MODEL` fails at
start, where before it degraded both daemons quietly.

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
| 1 | **Model binding** (§4) | the §1.2 defects fixed in every daemon today: a model change refuses to start instead of injecting foreign KV or tombstoning every conversation; npcd passes its schema's corrupt-turn policies to the engine | unit: the `ModelKey` record round-trips as raw bytes; the header hash of a fixture GGUF is a fixed value; open with the same key opens, a different key refuses with the message, a first open of an older substrate adopts the key when spec and tokenizer match and refuses when they do not; an unreadable `tokenizer.json` refuses to start. Through the builder: a substrate bound by one checkpoint refuses a second with the same file name and tokenizer, before any weight is loaded, and still opens for the first. npcd: the engine is handed the mind's own corrupt-turn policies, and a corrupt turn on a `drop_turn` layer drops that turn and keeps the timeline |
| 2 | **Engine core / attachments** (§3.2) | two substrates in one engine, each with persistence, reload, cancel, toggles and corrupt-turn policies; the one-writer rule enforced | unit: attaching a substrate whose timelines overlap an attached one's is refused, as are the same substrate twice and anything after the stop; two substrates' equal section ids never share scheduler state; a detached tenant's leftover requests are dropped and its slot ids reserved; cancel on one does not reach the other; a second writable open of a held substrate — in-process or from another process — is refused, and admitted once the holder shuts down. Probe: `kv_fragmentation` with two attachments — both stories pass, each store holds its own conversations and none of the other's, and the first serves a turn after the second detaches |
| 3 | **Host + tenants, one server** (§3.1, §3.3, §3.4, §3.7) | the `host` crate; the common code of §3.7 moved into it; `zend`, `npcd` and `motherd` built from the workspace, the first two running through `host::run([tenant])` on the host's `web::Builder` — tenant APIs merged, UIs by host name, `/common/` once | every existing zend and npcd test; the GUI harness; `cargo build -p zend`, `-p npcd`, `-p motherd` each green alone; unit: a path two tenants register refuses to start; `/` serves zend's UI for `code.tokera.com` and a bare IP and npcd's for `bot.tokera.com`; a file rewritten in a content directory is served changed on the next request and changes `build`, with the process untouched; a missing content directory refuses to start; an npcd turn's thinking mode is the one npcd chose, under the core's `thinking(true)`; a live restart of each |
| 4 | **Shared endpoints** (§3.5) | the overlaps served once as their superset; one identity rule, one role table, host accounts; both UIs on the chosen field meanings | unit per endpoint with raw expected bodies — both tenants' sections present, shared parts once, `timeline` a string, signed-out `/v1/me` a 200; `/ws/logs` neither drops nor repeats a line across replay; a header from a non-gateway peer is not believed. GUI: zend's status overlay, substrate page and perf page; npcd's console sign-in, substrate, tools and system pages |
| 5 | **The prescribed model** (§5) | `host::MODEL` for every tenant; zend's ladder and `--model` removed; the fine-tune in `PRESETS` | unit: `MODEL` passes both tenants' capabilities; a capability `MODEL` lacks is refused with the tenant's name. Live: zend on a new substrate under the prescribed model — ingest rebuilds, chat works |
| 6 | **zend and npcd equal** (§3.6) | priority by band alone, whichever tenant | unit: two tenants' `Low` sequences do not pause each other; either tenant's `High` pauses both tenants' `Low`, and the pause lifts at `COOLDOWN`; an npcd decode throttles a zend ingest prompt by its band's ratio, and a zend section chunk yields to an npcd dialogue |
| 7 | **`motherd` on `.5`** | `motherd` deployed: both daemons in one process on `:8081` / `:8444`; npcd's mind and data on `.5`; the gateway points `bot.tokera.com` at `:8444` beside `code.tokera.com`. npcd's substrate was written by the AntiLoop + StyleTune fine-tune, which the host no longer prescribes (§5.1), so it refuses at open, before anything in it is rewritten (§4.2), and npcd starts `motherd` on a new `--data`. The new one must also be written under the engine's minimum YaRN factor — zend's `rope: min_yarn_factor: 2` — or it refuses the first time npcd runs beside zend (§4.1) | live: both GUIs through the gateway, both on the LAN, a refused start when a substrate's model is swapped |

Phase 1 is worth doing first and on its own: it closes a live corruption path in both
daemons regardless of whether the host is built.

### 6.1 Where it stands

| # | Built | Where it is proved |
|---|---|---|
| 1 | yes — `models::identity` (`ModelKey`, the GGUF fingerprint, `WeightFiles`, `rope_key`), `persistence::model_binding` (`BoundSubstrate`), the `ModelKey` record | `candle-conversation` unit tests: the fingerprint pinned to a fixed value, the record as raw bytes, bind/refuse/redescribe/adopt on reopen (`persistence::model_binding`, `persistence`); through the builder, an unreadable tokenizer refused and a same-tokenizer checkpoint refused (`models::builder::identity_tests`); the engine's `DropTurn` policy (`tests/integrity_repair.rs`) and npcd handing over its mind's policies (`npcd::engine::schema`) |
| 2 | yes — `EngineCore` + attachments (`engine_attachments`), `EngineAttacher`, `persistence::writer_lock` released at shutdown, the process-wide timeline allocator, per-substrate cancel and toggles, `scheduler::{detached, section_pages}` | unit: every attach refusal (`engine_attachments`); a second writable open refused and a released one admitted (`persistence`); a cancel that does not cross substrates; a detached substrate's requests dropped and slots reserved (`scheduler::detached`); section pages shared by content and released by their last holder (`scheduler::section_pages`); probe: `kv_fragmentation::qwen36_35b_a3b_q4_two_attachments` (`fragmentation_probe::attached`) |
| 3 | yes — `host/`, `motherd/`, `zend::tenant`, `npcd::tenant` | `host` unit tests (site table, edit-on-disk, missing directory, path overlap including catch-alls and nests, build id, one-shot flags dropped on relaunch); the core's `thinking(true)` changes nothing for a tenant that set none (`models::builder::thinking_tests`, `engine_slot::core_requests`); `zend/tests/gui_api_harness.rs` over the assembled server; `cargo build -p zend`, `-p npcd`, `-p motherd`; zend restarted live through `host::run` |
| 4 | yes — `host::endpoints`, `web::Builder::trust`, `host::access`, both UIs on the chosen fields | unit per endpoint with exact bodies; `/ws/logs` neither drops nor repeats across replay (`endpoints::logs`); a header from a non-gateway peer cleared, in the server and in the assembled host (`web/tests/roles.rs`, `host::server`); `--local-signin` refused beside `--gateway` |
| 5 | yes — `host::MODEL` (Flash-Next `Q4_KO` since 2026-10-11, §5.1); zend's ladder and `--model` gone | both tenants' `requires()` against `MODEL`; `Capabilities::check` naming every lack. Live: zend under Flash-Next, 2026-10-11, on the substrate it had run on — written before keys, admitted by §4.2's check, then adopted (`docs/replicated_substrate.md` §6.2) |
| 6 | yes — the band (`DecodePriority`) alone; first built tenant first, changed 2026-10-12 | `scheduler::prefill` unit tests: two tenants' ingests together, a second tenant's dialogue throttling a first tenant's ingest while both ingests keep advancing, a section chunk yielding to another tenant's dialogue |
| 7 | yes — 2026-10-12, as a member of the gateway's cluster (`docs/replicated_substrate.md` §15.1) | deployment, done with the operator |
