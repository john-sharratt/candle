# The NPC Engine — `npcd`

## A daemon for persistent NPCs, built on the substrate zend already proves

This document specifies the **engine**: the daemon, its API, its substrate schema, its
tool system, and the refactor that lets it share zend's machinery. It is the
implementation counterpart to [`npc_mind_design.md`](npc_mind_design.md), which specifies
the **mind** — the gather, the immutable core, the six conversation-layers, the tick loop.
Where that document says *what an NPC is*, this one says *what we build*.

The governing constraint is that almost nothing here is new machinery. The substrate
already addresses many conversations against shared groups by a numeric id; the projection
schema already declares layers with windows, budgets and selection rules; the templating
engine already loads authored sections from folders and scopes them per identity; the tool
system already binds strongly-typed Rust requests to model-visible JSON Schema. The NPC
engine is mostly a **new schema, a new API surface, and one genuinely new subsystem** (the
tick loop) over machinery that exists and is in production in zend.

---

## Part 0 — What already exists

Four findings from the current tree shape everything below. They are stated here because
each one removes a design decision that would otherwise be open.

**The substrate is already multi-tenant, keyed by a number.** `Substrate` holds
`timelines: HashMap<TimelineId, TimelineEntry>` and, crucially, the inverse index
`timelines_by_group: HashMap<GroupId, Vec<TimelineId>>` — *many timelines registered
against one group*. `TimelineId` is a `NonZeroU64`. Per-timeline state already includes
compression overrides, token totals, distill modes, tombstones, archival, and transient
marking. `active_timelines_for_group` is the live filter. **The "unique number that filters
a shared substrate" is not something to build — it is the substrate's native addressing
model.**

**Both isolation modes already exist.** A projection whose target sits on a normal dialogue
layer runs a *multi-timeline* belief scan — selection enumerates every active timeline in
the group. A projection whose target sits on an `append_only_layers` layer is scored
**self-local**: every belief group is masked to the target's own timeline. That distinction
was built so an ingest scope-summary stays grounded in its own file instead of drifting
across the corpus. It is exactly the distinction between an NPC's *private* layers and the
*shared* ones, and it is already implemented and load-bearing.

**Identity scoping over shared sections already works.** `IdentityBuilders`
(`zend/src/session.rs`) resolves a per-conversation identity and retains only that
identity's members of the `identity_anchor` and `identity` collections, with members
namespaced `<name>::<stem>`. Identities load from an `identities/<name>/*.yaml` tree —
`anchor.yaml` the always-on compressed self, every other file a detail facet.

It also carries a scar worth inheriting. The resolver has **three** states, not two:

```rust
enum IdentityScope { Named(String), Empty, Unscoped }
```

and the comment above it is blunt: collapsing these onto `Option` "is what leaked every
anchor" — an unresolved identity, when identities *are* installed, must scope to **empty**,
never fall through to the unscoped builder. At section level that leaked personas into each
other. **At turn level, over a shared NPC substrate, the same bug is NPC A remembering NPC
B's life.** Every scope resolution in this engine is three-state from the start.

**The interaction primitive already exists.** `Sequence::fork_resuming(timeline)` registers
a timeline against the parent's `(layer, group)` and forks onto it, inheriting the parent's
turns by reference. A fork shares the parent's sealed KV prefix and diverges only at its
suffix. This is precisely "a new interaction is a new conversation on an existing
substrate", and it is the mechanism behind the mind document's result that *the popular NPC
is the cheapest*.

Two smaller ones: `TimeSource` is a trait whose own docs cite "narrative-time applications"
and "fictional clocks"; and `SelectionRule::Single`'s docs already reference "a single goal
pressure, a single active threat". The schema was written with this use in mind.

---

## Part I — Shape of the daemon

`npcd` is a second daemon beside `zend`, sharing a common core.

```
   ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐
   │   GUI    │  │   API    │  │ TEST HARNESS │  │  direct crate    │
   │ (static  │─▶│  /v1/*   │  │ tests/       │  │  embedding       │
   │  assets) │  │  + /ws   │  │  harness.rs  │  │  (no HTTP)       │
   └──────────┘  └────┬─────┘  └──────┬───────┘  └────────┬─────────┘
    second-class      │ first-class   │                    │
                      ▼               ▼                    ▼
              ┌────────────────────────────────────────────────┐
              │  npc-engine  (the crate)                      │
              │  NpcRegistry · Tick scheduler                 │
              │  Interaction reaper · ToolReg                 │
              │  Environment simulator · MiscWork             │
              └───────────────────────┬────────────────────────┘
                                      ▼
              ┌────────────────────────────────────────────────┐
              │  candle-conversation                          │
              │  substrate · projection ·                     │
              │  templates · runtime · tools                  │
              └────────────────────────────────────────────────┘
```

The harness is a peer, not tooling on the side. Drawing it there enforces that **it may only use
what a real consumer can use** — a test needing a back door is evidence of a missing capability,
and the fix belongs in the core rather than in a test-only seam (Part XIV).

**One executable, two launch modes.** `battlecities.exe` is the game client and loads no engine;
`battlecities.exe --server --headless` is this daemon and loads no renderer. **Version 1 runs AI
in headless server mode only** — the two are never co-resident, so the engine owns the card
whenever it is up, and the `gpu_poison` exit-for-supervisor-restart path stays correct because
there is no game process to take down. Cognition during play is a later version.

**API is first-class, web is second-class.** zend already has the right posture and it is
kept verbatim: one axum `Router` of `/v1/*` routes plus websockets, with the web UI as
`.fallback(embedded_asset)` — structurally the last resort after every API route misses.
The GUI is a client of the API and has no privileged path into the engine.

**Four entry points, one core.** The engine's capability surface is a Rust trait object
(`NpcEngine`) that the HTTP layer wraps. Embedding the crate directly gives the same
operations without a socket; the GUI gives them with a face; the test harness (Part XIV)
uses the same surface with no test-only seam. No operation exists at only one of the four.

---

## Part II — Identity and numbering

### Three scopes, not one

The mind document has three levels of sharing, and a single number cannot address all
three. The engine names them explicitly:

| Scope | Cardinality | Mutability | Mechanism |
|---|---|---|---|
| **World** | 1 | fixed | CoW prefix, shared by every NPC |
| **Personality** | 1 per type (~300 NPCs) | doctrine only | CoW prefix, shared per type |
| **NPC** | 1 per NPC | free | per-NPC timelines over shared groups |

```rust
pub struct WorldId(String);          // the slug that is `worlds/<id>.yaml`
pub struct PersonalityId(String);    // the slug that is `personalities/<id>.yaml`
pub struct NpcId(NonZeroU64);        // the "unique number"
pub struct InteractionId(NonZeroU64);
```

The first two are slugs, not numbers, because the documents they address are **files** in the
mind and a file's identity is its name. A number beside the name would be a second identity
for the same thing, needing a table to reconcile them and free to disagree; validity is
`registry::id::check`, the same rule that decides what may become a file name, so a reference
always resolves to a document or is refused at the boundary that introduced it.

`NpcId` is the number the requirement names. It addresses the third scope only; world and
personality sharing is handled by prefix construction, not by the filter. An NPC's full scope
is the **chain** `(WorldId, PersonalityId, NpcId)` — resolved once at spawn into a projection
`Builder` cached per chain, exactly as `IdentityBuilders` caches per `(name, ToolMode)`
today.

### Timelines are derived, not assigned

An NPC is not one conversation. It has a timeline in **each** substrate layer, plus one per
live interaction. Timelines are derived deterministically so that a restart reconstructs
the same addressing without a mapping table:

```rust
fn timeline_of(npc: NpcId, layer: NpcLayer) -> TimelineId    // domain-separated hash
fn timeline_of_interaction(npc: NpcId, ix: InteractionId) -> TimelineId
```

This mirrors zend's existing `timeline_for(conv_id)`, which hashes the client-supplied
conversation string into a `TimelineId`. Domain separation by layer is mandatory — without
it two layers of the same NPC collide.

### The scope resolver is three-state

Every place a scope is resolved uses the `Named / Empty / Unscoped` discipline inherited
from `IdentityScope`. An NPC id that fails to resolve, when NPCs *are* installed, scopes to
**empty** — the NPC surfaces nothing — never to unscoped. A test asserts this directly:
*a projection for an unknown `NpcId` must contain zero turns from any registered NPC.*

---

## Part III — The substrate schema

The six conversation-layers of the mind document become six layers in the projection YAML,
plus the layers the engine itself needs. Each is an ordinary layer — window, budget,
selection rule, summariser policy — and the differences between them are entirely
declarative.

| Layer | Selection | Masking | Clock | Notes |
|---|---|---|---|---|
| `perception` | `Sequence{recent, historical_top_k}` | self-local | fast | API-fed; see Part V |
| `action` | `Sequence{…}` | self-local | tick | the act stream; ground truth |
| `agency` | `TopK` / `Single` | self-local | slow | missions, strategies, sub-goals |
| `relationships` | `TopK` | self-local | drift | per-entity calibration |
| `beliefs` | `TopK` | self-local | threshold | **write-protected**, Part VI |
| `memory` | `Sequence{…}` | self-local | unbounded | consolidation target |
| `interaction` | `Sequence{recent:N}` | self-local | conversational | one timeline per interaction |
| `environment` | `Sequence{recent:N}` | self-local | event | own system prompt, Part VIII |
| `world` | `TopK` | **cross-timeline** | slow | shared; the only unmasked layer |

Everything an NPC privately owns is marked append-only/self-local so its belief scan is
masked to its own timeline. The `world` layer is deliberately the exception: it is the one
place cross-timeline enumeration is correct, because shared world facts *should* be
retrievable across NPCs.

Layer geometry — `window`, `budget.priority`, `budget.min_percent`, `score_threshold`,
`decode_priority` — is the calibration surface the mind document's Part XII says can only
be set by watching real runs. It is YAML precisely so it can be moved without a rebuild.

### The system prompt is the immutable core, viewed

Part II of the mind document maps directly onto the existing collection machinery, which
already implements the three selection disciplines:

| Discipline | Mechanism today | Collection |
|---|---|---|
| Structurally always-present | `AlwaysVisible` retained to the NPC's chain | `identity_anchor` |
| Spiking (mood) | `Named{selector}` set per turn via `TurnOptions::selection` | `mood` |
| Locked top-k (template) | `Named{selector}`, chosen once at interaction start, frozen | `response` |

`SelectionRule::Named` is explicitly score-independent — it selects exactly the member the
caller names, ignoring provenance relevance and the score threshold. That is the correct
primitive for both a mood spike and a locked template, and it already exists.

---

## Part IV — Interactions

### An interaction is a conversation on the NPC's substrate

```
POST /v1/npc/{npc_id}/interaction
  { "mode": "physical" | "video_call" | "voice_call" | "instant_message",
    "interlocutor": { "kind": "player"|"npc"|"operator", "id": "…" },
    "idle_timeout_secs": 900 }
  → { "interaction_id": … }
```

Creating one calls `fork_resuming(timeline_of_interaction(npc, ix))` against the
`interaction` layer. The fork inherits the NPC's sealed prefix by reference and diverges
only at its suffix — so the tenth concurrent interaction with a popular NPC costs a suffix,
not a mind.

### Mode is a projection input, not a branch

The four modes are **not** four code paths. Mode selects a template from the `response`
collection and sets the observability envelope:

| Mode | Observable | Not observable |
|---|---|---|
| `physical` | speech, movement, gesture, expression, ambient world acts | internal broadcasts |
| `video_call` | speech, expression, framed gesture | movement outside frame, ambient |
| `voice_call` | speech, audible action | all visual acts |
| `instant_message` | speech only, text-shaped | everything else |

This is the mind document's *scoping by observability, not relevance* made concrete: the
mode is the vantage. A voice call is not a "chat mode" — it is the same act stream narrated
through a vantage that cannot see. An NPC that breaks off mid-sentence to look east is
narrated as a pause on a voice call and as a turn of the head in a physical encounter,
because it is one act read through two envelopes.

Mode is fixed for the life of the interaction. Changing modes ends one interaction and
starts another — which is also what happens in the fiction.

### Ending on idle

An interaction ends when no event has touched it for `idle_timeout_secs`. A reaper task
sweeps live interactions on a coarse tick.

Ending **archives**, it does not delete. The substrate already distinguishes archived
(`TimelineEntry::archived`, filtered out of `active_timelines_for_group`) from tombstoned
(permanently dead). An ended interaction is archived: it stops being enumerated, its turns
stop being gathered, and its content survives for consolidation to fold into `memory` on
the sleep clock. This is the mind document's *soft fade by non-selection* — reversible,
cue-resurfaceable — rather than its *hard forget*, which belongs only to the sleep fold.

The idle timeout is per-interaction and set at creation, defaulting per mode: a physical
encounter lapses faster than an instant-message thread, because standing silently in front
of someone means something different from not replying to a message.

---

## Part V — Perception ingest

Perception is **pushed by the caller**, never polled by the engine. The world simulation is
authoritative; the NPC engine is a consumer.

```
POST /v1/npc/{npc_id}/perceive
  { "events": [
      { "kind": "description", "text": "…", "salience": 0.8 },
      { "kind": "map", "zoom": "tactical", "ascii": "…", "legend": {…} },
      { "kind": "entity", "entity_id": "…", "observation": "…" }
    ] }
```

A batch is one call. This matters: the mind document's central asymmetry is that
**perception is prefill and action is decode**, and a batched POST is exactly the shape
that lets fifty world events absorb in one batched prefill across every NPC in the fight,
while decode is spent only when an NPC acts. A per-event endpoint would silently destroy
that property, so the batch endpoint is the primary and there is no single-event variant.

### ASCII maps at zoom levels

A map event carries a `zoom` band and renders as a fenced block with a legend. Zoom bands
are declared per world rather than hardcoded — a reasonable default set being `strategic`,
`regional`, `tactical`, `local`.

Maps are **replacing, not appending**. A new map at a given zoom supersedes the previous one
for that band, because twelve stale tactical maps in the gather is twelve chances to act on
a position that no longer exists. Descriptions accumulate; maps replace. This is the one
place the engine departs from pure append-only, and it does so by writing a superseding turn
and marking the prior one distilled — not by mutating history.

Perception events land whatever their salience. They are never dropped at emission — the
mind document is explicit that a filter which drops contradicting evidence before it lands
makes a delusion permanent. Salience biases the gather; it does not gate the write.

---

## Part VI — State mutation, and the authoring/action distinction

The requirement that relationships and beliefs be modifiable by API collides, on its face,
with the mind document's insistence that beliefs are write-protected. The collision is
apparent, not real, and resolving it explicitly is important enough to state as an
invariant:

> **The write-protection is against the model, not against the operator.**
> There are two planes. The **action plane** is what the NPC can do — and on it, the arrow
> from action to belief is structurally absent. The **authoring plane** is what an author,
> designer, or world simulator can do — and on it, beliefs are writable, because someone
> has to be able to say what this character believes when the world is built.

An API belief write is an authoring act. It is recorded as such — with a provenance marker
distinguishing it from an evidence-threshold rewrite — and it is available to the GUI
because building a cast of characters requires it. What remains impossible is for the NPC's
own decode to emit a belief mutation. No tool in the generic catalog writes to the belief
layer, and the tool registry rejects an extension tool that declares the belief layer as a
write target.

```
   AUTHORING PLANE (operator / GUI / API / world sim)
        │  may write: every layer, including beliefs
        ▼
   ┌─────────────────────────────────────────────┐
   │  SUBSTRATE                                  │
   └─────────────────────────────────────────────┘
        ▲  may write: every layer EXCEPT beliefs
        │  may read:  every layer
   ACTION PLANE (the NPC's own decode → tool calls)
        │
        └── beliefs change here only via the
            evidence-threshold process on the slow clock
```

Relationships have no such restriction on either plane — a relationship is a calibration
trajectory that is *supposed* to move easily. The asymmetry between the two layers is the
point of having both.

---

## Part VII — Tools

### The generic catalog

Every NPC gets a base vocabulary. These are the acts the mind document's fan-out produces,
and `speak` is among them rather than beside them — dialogue is a tool call, which is what
welds the conversational surface to ground truth.

| Category | Tools | Availability |
|---|---|---|
| Speech | `speak` | always |
| Movement | `move_to`, `face`, `follow`, `flee` | always |
| Gesture | `gesture`, `express` | always |
| Attention | `observe`, `listen`, `inspect` | always |
| Social | `greet`, `offer`, `refuse`, `threaten` | always |
| Internal | `note_concern`, `set_intent`, `broadcast_strategy` | always |
| Messaging | `send_image` | **messaging modes only** |
| Meta | `wait`, `end_interaction` | always |

`speak` is append-only and conflict-free; world-mutating acts commit through an arbiter
carrying the world-version they reasoned over, per the mind document's two commit paths.

**Tools carry intent, not output.** `speak` does not take a sentence and `send_image` does not
take an image prompt — each takes what the character *means* to convey, and the narrator renders
it into prose or into a scene prompt. This is the mind document's "narrate acts, never
fabricate" applied one level deeper: the mind decides substance, the surface decides wording,
and neither can produce what the other did not license. The wire format is
`candle-conversation::narrator`'s existing `NarratorInput` enum. See
`npc_api_gui_design.md` §18.

`send_image` is absent from the catalog in `physical` mode rather than present-and-refused — a
character standing in front of you does not text you a photo, and the model should never be
invited to try. It takes the interlocutor's **unique name** as its target, validated against the
interaction.

### Extension with strongly-typed parameters

zend's tool system has exactly the right ergonomics and the wrong lifetime. `Tool` is a
trait with associated `Request: DeserializeOwned + JsonSchema + Validate`, `Response:
Serialize`, and `Error: ToolError`; `RegisteredTool::new::<T>()` is a `const` capturing
three function pointers and erasing the concrete types. The dispatch pipeline —
`from_value` → `validate()` → `run` → `to_value` — is exactly what an extension tool wants.

But `RegisteredTool` is a `Copy` struct of `fn` pointers in a static table. A framework
extension needs to capture state (a game handle, a channel, a database). So the engine adds
a **dynamic registry alongside the static one**, preserving the type discipline:

```rust
pub trait NpcTool: 'static {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    type Request: DeserializeOwned + JsonSchema + Validate;
    type Response: Serialize;
    type Error: ToolError;
}

impl ToolRegistry {
    /// Register an extension tool. The JSON Schema the model sees is derived
    /// from `T::Request` via schemars — never hand-written, so the prompt and
    /// the parser cannot disagree.
    pub fn register<T: NpcTool>(
        &mut self,
        handler: impl Fn(&NpcToolCtx, T::Request) -> Result<T::Response, T::Error>
            + Send + Sync + 'static,
    ) -> Result<(), RegisterError>;
}
```

The closure is boxed into an `Arc<dyn Fn(&NpcToolCtx, &Value) -> Value + Send + Sync>` by a
generic shim running the parse/validate/run/serialize sequence — the same shim as
`tool_run::<T>`, differing only in that it closes over the handler instead of calling
`T::run`. The caller writes strongly-typed Rust; the erasure happens once, inside.

### The calibration constraint

One consequence must be stated because it is easy to miss and expensive to discover late.
zend's tool *selection* quality comes from calibration: each tool definition carries
`examples` — ChatML trajectories with `<|projection|>` markers — prefilled at startup into a
reserved calibration layer to seed the wide-Q reference substrate. A tool with no examples
is uncalibrated and selects worse.

So `register` accepts optional examples, and a tool registered **before** the calibration
phase is calibrated with the rest. A tool registered **after** the engine is live is usable
but uncalibrated until the next calibration pass. The API surfaces this honestly rather than
hiding it: registration returns whether the tool was calibrated, and there is an explicit
endpoint to trigger a calibration pass over newly-registered tools.

---

## Part VIII — The environment simulator

An NPC acts on a world. During authoring, and during any run where no external world
simulation is attached, there needs to be something that answers *what happened next*. The
environment simulator is that something.

It is **its own conversation with its own system prompt** — not a role the NPC plays, and
not a second mind with a substrate. It is a bounded, sliding-window conversation:

- Its own layer (`environment`) with `Sequence { recent: N }` — a sliding window, no
  historical top-k, because the environment's job is continuity of the immediate scene, not
  recall of everything that ever happened. Long-run world memory belongs to the `world`
  layer, which the simulator writes into.
- Its own system prompt, authored per world, describing the setting's physics, tone, and the
  rules for what may and may not change.
- Fed by the NPC's committed acts and by injected events; emits perception events back into
  the NPC's `perceive` path.

```
     NPC act stream ──▶ ENVIRONMENT SIM ──▶ perception events ──▶ NPC inbox
                        (own sysprompt,
                         sliding window)
                              ▲
                              └── injected world events (API / GUI)
```

It is **toggleable per NPC** and **defaults to on when an NPC is created through the GUI** —
because a character created in the GUI has no world attached and would otherwise perceive
nothing — and defaults to **off** when an NPC is created through the API, on the assumption
that an API caller has its own world simulation and does not want a second one inventing
events underneath it. This asymmetry is deliberate and is the one place the GUI and API
defaults intentionally differ.

The simulator runs at the same tick as its NPC and never blocks it: it consumes the act
stream after commit, so a slow environment step delays the *next* perception batch rather
than the current decode.

---

## Part IX — Templating

The templating engine is kept as-is in mechanism and generalised in sourcing.

Today: any collection declared empty in the schema's `system_prompt` is auto-filled from a
workspace folder named by pluralising the collection (`response` → `responses/`, `mood` →
`moods/`, `identity` → `identities/`). Each file is one `ResponseSection` — `id`, `category`,
`description`, `template`, `examples` — validated, `{CHAR_NAME}`/`{USER_NAME}` substituted,
and installed as the section's content with the examples driving calibration.

Three changes:

1. **Sources are explicit.** A collection may be sourced from a path, from an embedded
   default set, or from templates passed directly over the API — so a caller embedding the
   crate can supply a whole template set without touching the filesystem.

   ```rust
   pub enum TemplateSource {
       Path(PathBuf),
       Embedded(&'static Dir<'static>),
       Inline(Vec<ResponseSection>),
   }
   ```

2. **The substitution vocabulary grows** beyond `{CHAR_NAME}`/`{USER_NAME}` to the NPC's
   resolved scope chain — personality, world, and the interlocutor of the current interaction.
   Substitution stays a flat string replace over a closed, validated key set; it does not
   become an expression language.

3. **Templates are per-world overridable.** A world may ship its own `responses/` and
   `moods/` that shadow the defaults, because tone is world-specific and a merchant
   negotiation reads differently in two settings.

---

## Part X — The crate refactor

The requirement is that everything reusable in zend moves into `candle-conversation` and is
shared. This is the concrete inventory.

### Moves to `candle-conversation`

| From | To | What it is |
|---|---|---|
| `zend/src/response_section.rs` | `::templates` | the templating engine — sections, examples, substitution, install |
| `zend/src/tool_def.rs` | `::tools::def` | YAML tool definitions, `json_line`, calibration markers |
| `zend/src/tools.rs` | `::tools::catalog` | catalog install into a collection |
| `zend/src/chatml.rs` | `::runtime::chatml` | ChatML rendering |
| `zend/src/loading.rs` | `::runtime::loading` | load-state machine |
| `zend/src/download.rs` | `::runtime::download` | model acquisition |
| `zend/src/model_choice.rs` | `::runtime::model_choice` | VRAM-adaptive quant selection |
| `zend/src/turn_sink.rs` | `::runtime::turn_sink` | turn sink |
| `zend/src/projection_event.rs` | `::runtime::events` | projection events |
| `zend/src/refresh_ctx.rs` | `::runtime::refresh` | refresh context |
| `zend/src/log_{broadcast,file,line}.rs` | `::runtime::log` | the log bus |
| `zend/src/conv_file{,_store}.rs` | `::runtime::files` | attachment store |
| `InferenceState` core (`session.rs`) | `::runtime::inference` | model + sequence + streaming |
| `IdentityBuilders` (`session.rs`) | `::projection::scope` | **generalised to `ScopeBuilders`** |

`IdentityBuilders` is the most valuable of these. Generalised from a single identity name to
a scope chain, and from two collections to an arbitrary declared set, it becomes the engine's
scope resolver — carrying its three-state discipline and its per-key builder cache with it.

### Stays in zend

`code_read/` (7.3k), `repo_scan/` (4.0k), `raw_read/`, `watcher.rs`, `ingest*.rs`, and the
coding-assistant halves of `session.rs` and `api/`. These are workspace-and-code specific and
have no NPC counterpart.

### The test that keeps it honest

The refactor is only real if zend still works. `zend` is converted to a consumer of the moved
modules in the same change, and its existing behaviour is the regression test. A module that
cannot be moved without a zend-specific shim is a module that was not actually generic — the
shim is the signal to stop and reconsider, not to proceed.

`tree_gen` and the `characters/` fixtures are slated for removal once the NPC engine can
generate a life timeline natively; until then they stay, since `bramble_tree.yaml` is the only
worked example of a character's history materialised as substrate turns.

---

## Part XI — API surface

Versioned under `/v1`, mirroring zend's conventions.

```
  NPC lifecycle
    POST   /v1/npc                          create (personality, world, name, seed state)
    GET    /v1/npc                          list
    GET    /v1/npc/{id}                     full state
    PATCH  /v1/npc/{id}                     update core fields
    DELETE /v1/npc/{id}                     tombstone

  Perception
    POST   /v1/npc/{id}/perceive            batched events (descriptions, maps, entities)

  State (authoring plane)
    GET    /v1/npc/{id}/relationships       list
    PUT    /v1/npc/{id}/relationships/{e}   set/adjust
    GET    /v1/npc/{id}/beliefs             list
    PUT    /v1/npc/{id}/beliefs/{b}         authoring write (marked as such)
    GET    /v1/npc/{id}/agency              missions, strategies, sub-goals
    GET    /v1/npc/{id}/memory              consolidated memory

  Interactions
    POST   /v1/npc/{id}/interaction         open (mode, interlocutor, idle timeout)
    POST   /v1/interaction/{ix}/inject      operator/player event into the inbox
    GET    /v1/interaction/{ix}/stream      SSE: live acts, then tick-bounded narration
    DELETE /v1/interaction/{ix}             end explicitly

  Environment
    GET    /v1/npc/{id}/environment         simulator state + enabled flag
    PUT    /v1/npc/{id}/environment         toggle, set system prompt
    POST   /v1/npc/{id}/environment/inject  world event into the simulator

  Tools
    GET    /v1/tools                        catalog (generic + registered extensions)
    POST   /v1/tools/calibrate              calibration pass over uncalibrated tools

  Introspection  (mirrors zend's /v1/substrate/*)
    GET    /v1/npc/{id}/substrate           layer occupancy for this NPC
    GET    /v1/npc/{id}/substrate/layer/{n} turns in one layer
    GET    /v1/npc/{id}/projection          what the last tick actually gathered
    GET    /v1/npc/{id}/monitor             metacognition health signal
    GET    /v1/status, /v1/telemetry        daemon-wide
```

### The interaction stream is two streams

`GET /v1/interaction/{ix}/stream` is an SSE endpoint carrying two event kinds at two
latencies, per the mind document's Part VII:

```
  event: act          ← live, as each act commits
  event: narration    ← at tick close, the woven summary of the elapsed window
```

The client renders acts immediately and narration when it arrives. Collapsing these into one
stream would destroy the property that makes the interaction read as live.

### Introspection is a first-class product surface

zend already exposes `/v1/substrate`, `/v1/substrate/layer/:name`,
`/v1/substrate/timeline/:tl` and a `project` endpoint, and the GUI renders them. For NPCs
this matters more, not less: the mind document says every open question is a calibration
question answerable only by watching real runs, and `/v1/npc/{id}/projection` — *what did
this NPC actually gather on that tick* — is the instrument that makes that possible.

---

## Part XII — GUI

The GUI manages many NPCs and is a pure API client. Its screens follow the substrate rather
than inventing an organisation:

- **Roster** — every NPC, personality, world, tick rate, monitor health at a glance.
- **NPC detail** — the six layers as browsable streams, with the authoring plane exposed:
  edit relationships, author beliefs, set intent, adjust modulation parameters.
- **Interaction console** — open an interaction in a chosen mode, inject, watch the two
  streams side by side.
- **Environment panel** — toggle the simulator, edit its system prompt, inject world events.
- **Projection inspector** — for a given tick, what was gathered, what was dropped, what won
  the budget. The debugging surface for every calibration question.
- **Monitor** — the narration/substrate overlap metric over time, with the expressive band
  marked, so an NPC sliding from characterful fixation toward incoherence is visible before
  it arrives.

An NPC created here gets the environment simulator on by default (Part VIII).

---

## Part XIII — What is genuinely new

Almost everything above is assembly. Three things are not, and they are where the risk
concentrates.

**The tick loop.** zend has no counterpart: it is driven from outside by HTTP, whereas an NPC
drives itself. The loop — block on inbox, drain, gather, decode one step, fan out — is new
code, and with it the salience-gated tick, preemption on high-salience events, the timer
heartbeat whose interval is the NPC's idle metabolism, and the scheduler that batches across
all NPCs with pending events. The existing scheduler
(`candle-conversation::scheduler`, 17.6k LOC, with admission control) is the right substrate
for it, but the loop above it is new.

**The evidence-threshold belief process.** Slow-clock accumulation of disconfirming events
against a per-frame threshold — the only writer on the action plane's forbidden layer.

**The metacognition monitor.** N-gram / mutual-information overlap between the narration
stream and the substrate streams, computed outside the gather. Cheap, but new, and the only
instrument for the failure mode the architecture cannot prevent structurally.

Everything else — substrate, projection, templating, tools, streaming, persistence, model
loading, the API posture — is machinery that exists and runs today.

---

## Part XIV — The test harness

A system whose behaviour is "what the salience function selected" cannot be verified by reading
it. It needs a suite broad enough to cover every API and every interaction path, fast enough to
run constantly, and honest enough to execute real decodes rather than asserting against a fake.

The target shapes everything: **the whole suite in about a minute.**

### One binary, not fourteen

`candle-conversation/tests/` currently holds 14 separate `*.rs` files. Cargo compiles each into
its **own binary and runs it as its own process**, so a shared model would be loaded fourteen
times. Any design that shares one load must therefore start here:

> The NPC suite is **one integration-test binary** — `tests/harness.rs` — with the suite's
> structure expressed as `mod` declarations inside it, not as sibling files in `tests/`.

This is a Cargo constraint rather than a preference, and it is the single most important
structural decision in the harness. Files under `tests/` are entry points; everything else is a
module beneath one of them.

### The shared engine, and the failure rule

```rust
static ENGINE: OnceLock<Result<Arc<TestEngine>, String>> = OnceLock::new();

/// Every GPU-backed test's first line. The first caller loads; the rest wait
/// on the same `OnceLock` and share it.
pub fn engine() -> &'static TestEngine {
    match ENGINE.get_or_init(TestEngine::load) {
        Ok(e) => e,
        Err(why) => panic!("engine load failed — every engine test fails: {why}"),
    }
}
```

`OnceLock` gives the required semantics exactly: cargo's harness runs tests as threads within
one process, the first to call `engine()` performs the load while the others block, and a load
failure is stored and re-panicked for every subsequent caller. **If loading fails, all engine
tests fail** — with the root cause in each message rather than a cascade of unrelated errors.

The load itself is deliberately the suite's dominant cost and is paid once.

### Three tiers

| Tier | Needs a GPU | Count | Budget |
|---|---|---|---|
| **Pure** | no | most | milliseconds, total |
| **Engine** | yes | many | the bulk of the minute |
| **Deep** | yes, opt-in | few | excluded by default |

**Pure tests need no engine at all** and must not touch it: scope resolution and the three-state
`Named/Empty/Unscoped` discipline, tag-filter semantics, the slash-command parser against its
shared corpus, JSON Schema derivation, id derivation and domain separation, observability
envelopes per mode, error mapping. These are the majority of the suite and they run on a machine
with no CUDA card — which matters, because CI probably has none.

**Deep tests** — a full consolidation fold, a real image generation, a long-context projection —
are gated behind `--features deep-tests` and excluded from the default run. They exist, they are
run before a release, and they are not allowed to cost the minute.

### Making decodes cheap without making them fake

The suite executes real prefills and decodes. Four levers keep that affordable:

- **A tiny model.** `Model::Qwen2_0_5B` is already in the registry, documented as "tiny, great
  for CI and testing (~0.4 GB)". It loads in a second or two and decodes fast enough that
  thousands of short generations fit the budget.
- **Token caps.** Engine tests assert on structure, so they decode a handful of tokens, not a
  reply. `max_tokens: 8` is a normal ceiling.
- **Greedy sampling.** Temperature zero everywhere, so a test that *does* compare output has a
  fighting chance of stability.
- **Constrained decode where shape is the assertion.** For tool-selection and command paths, the
  stencil machinery already forces well-formed output, so the test verifies routing rather than
  prose quality.

### Mocking the substrate by forking a real one

The naive approach — a fake substrate returning canned turns — is both slow to build and
guaranteed to drift from the real one. The engine already has a better primitive:

> Build **one** base substrate for the suite, prefilled and sealed. Every test calls
> `fork_resuming` onto a fresh timeline and inherits it by reference.

This is the production path (§IV), so a test's substrate cannot diverge from a real one. It is
also nearly free: a fork shares the parent's sealed KV prefix and diverges only at its suffix,
which is the same property that makes the popular NPC cheap. The suite's fixture cost is one
prefill of a few hundred tokens, amortised across every test that needs history.

Three supporting mechanisms, all already present:

- **Transient timelines.** The substrate's `transient_timelines` set marks a timeline that is
  never reloaded and carries no redo-log marker. Tests use them, so the suite does no persistence
  I/O and leaves no state between runs.
- **Fixture depth by construction.** A test needing a thousand-turn memory does not generate a
  thousand turns; it appends sealed turns with known token counts through the test-helpers
  surface, so gather and budget behaviour can be exercised at depth without paying for depth.
- **`test-helpers`.** `candle-conversation` already ships the feature (`for_test` id
  constructors and friends). The harness extends it rather than inventing a parallel path.

The rule that keeps this honest: **fixtures may fabricate *content*, never *mechanism*.** A test
may hand the substrate a turn that no model produced. It may not stub the gather, the projection,
the scope filter, or the tick — those are the things under test.

### Parallelism is the product's own workload

Cargo runs tests concurrently by default, and on a single GPU that would normally be a problem.
Here it is the opposite:

> Concurrent tests are concurrent conversations, which is exactly the workload the scheduler is
> built for. The suite exercises wave batching, admission control and fair scheduling **by
> construction**, every time it runs.

A suite of two hundred concurrent short generations is a small load test that happens to also
assert correctness. It is also the cheapest continuous check that the batching claim — that
concurrency is nearly free — still holds.

Two constraints follow. Each test must use a **small KV budget**, or concurrency exhausts VRAM
rather than exercising it. And tests must be **independent**: each on its own NPC and its own
forked timeline, with nothing shared but the engine and the read-only base substrate. A test
that mutates global engine state serialises the whole suite and will be the reason the minute
becomes five.

### Determinism: assert invariants, not prose

Batch composition varies between runs, so the same prompt may not produce byte-identical output.
Tests therefore assert on structure:

- an act was emitted, with the expected tool and a non-empty intent
- the projection selected from the expected layers, within budget
- a belief's disconfirmation rose, and its confidence did not
- an unknown `NpcId` gathered exactly zero turns
- a hidden NPC is absent without a tag filter and present with one

Where output text genuinely matters — the narrator rendering a `Say` — the assertion is a
property (non-empty, in-character name present, no leaked control tokens) rather than an exact
string.

### The image module is stubbed by default

Loading a diffusion model would blow the budget on its own. The harness registers a **stub image
backend** returning a fixed small PNG immediately, so every path around generation — queueing,
the misc-work drain, `send_image` argument validation, regeneration on description change,
portrait state transitions in the API — is fully covered without a diffusion model present. One
`deep-tests` case does a real generation to keep the stub honest.

### The budget

```
  engine load (Qwen2-0.5B)      ~2 s
  base substrate fixture        ~1 s      one prefill, shared
  ~40 pure test modules         <1 s      no GPU
  ~200 engine tests             ~30 s     concurrent, ≤8 tokens each
  teardown                      ~1 s
  ────────────────────────────────────
  total                         ~35 s     against a 60 s target
```

The headroom is deliberate. It is what pays for the suite doubling in size before anyone has to
think about this again.

### What must never be mocked

Three things are the reason the harness exists, and stubbing any of them would leave the suite
asserting that the mocks work:

1. **Scope isolation.** NPC A must never gather NPC B's turns, and an unresolved id must scope to
   empty. This is tested against the real projection, at depth, concurrently — because the leak
   it guards against is a real one that already happened once at section level.
2. **The two planes.** The action plane must have no path to the belief layer. The test attempts
   the write through a tool context and asserts `422`, against the real registry.
3. **The tick loop.** Salience gating, preemption and the quantum boundary are behaviour, not
   plumbing.

---

## Part XV — Phasing

1. **Refactor.** Move the shared modules to `candle-conversation`, convert zend to a consumer,
   keep zend green. No NPC code yet. This is the change that de-risks everything after it.
2. **Skeleton.** `npcd` boots, loads a model, serves `/v1/status`, spawns one NPC from an
   personality, answers `/v1/npc/{id}/substrate`. No tick loop — turns submitted manually.
3. **The harness.** `tests/harness.rs` as the single binary, the shared-engine `OnceLock`, the
   base-substrate fixture, and the pure tier. Built here rather than later, because from step 4
   onward every phase's acceptance criterion is a test in it — and a harness retrofitted after
   twenty features is a harness nobody trusts (Part XIV).
4. **Scope.** `ScopeBuilders` generalised; the NPC schema authored; the isolation test
   (unknown `NpcId` gathers nothing; NPC A never gathers NPC B's turns) passing before any
   behaviour is built on top.
5. **Interactions.** Fork-per-interaction, the four modes, the two-stream SSE, the idle reaper.
6. **Perception + authoring.** Batched perceive, ascii maps with supersession, the state
   endpoints, the two-plane write discipline.
7. **Tools.** Generic catalog, dynamic registry, calibration pass.
8. **The tick loop.** Salience-gated ticks, preemption, heartbeat, multi-NPC batching.
9. **Environment simulator.**
10. **Slow clocks.** Evidence-threshold beliefs, daydream/sleep consolidation, the monitor.
11. **GUI.**

Steps 1–4 are the ones worth being slow about. A scope leak found at step 10 is a rewrite;
found at step 3 it is a test.

---

## Part XVI — Open questions

**RESOLVED — the personality is prefix only.** `PersonalityId` addresses a shared CoW prefix and
never filters turns; **`NpcId` is the only number in the turn filter.** Doctrine therefore lives
in the surfaced prompt rather than as substrate content, and `ScopeBuilders` resolves a single
key rather than a pair. This keeps the scope resolver as narrow as possible, which matters
because it is the one place a leak would put NPC A's life in NPC B's head. If personality-level
*turns* are ever genuinely needed, that is a schema change with its own isolation test, not a
widening of the filter.

**How much of a tick is one decode?** The mind document says "decode one cognitive step".
Whether a step is one tool call, one sentence, or one bounded multi-call fan-out determines
the tick's cost and the narration's granularity, and is not settled by the design.

**Where do modulation parameters live?** Affect, threat and curiosity are weights on selection
rather than streams. The projection has per-layer budgets and score thresholds; whether
modulation is expressed as runtime overrides on those, or as a separate bias term in the
salience function, is an implementation choice with different calibration surfaces.

**Does the environment simulator share the NPC's model instance?** Sharing is cheaper and
keeps one KV economy; separating gives it its own tuning. The sliding-window design assumes
sharing, but this is untested.

**What is the world simulator's authority over perception?** When an external world sim and
the environment simulator are both attached, one of them must lose. The current answer is
that the environment simulator defaults off under API creation for exactly this reason, but
the conflict is deferred, not resolved.
