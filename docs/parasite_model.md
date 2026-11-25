# Parasite models: one card, a primary model, and the models that borrow it

**Status:** design, not built. It replaces `docs/shared_model_host.md`'s
single prescribed model (decision 7, §5), and it turns the guest drain
(`candle-conversation/src/guest/`) into one case of a parasite mount.

## 0. Summary

**The model.**
- Every card has one **primary** model, chosen by the card's size and resident
  from load to exit.
- Any other model a request needs is a **parasite**. It is mounted into ground
  the primary gives up, holds that ground while it is used, and unmounts when
  the primary wants it back.
- A parasite is one of three things: an LLM serving another tenant, the image
  generator or matte model, or a vision tower. All three go through one
  mechanism.
- One driver thread per card decides which model forwards.
- One expert cache per card is shared by every model on it.

**The first build adds multi-model support and changes nothing about how the
fleet operates.**
- zend and npcd both stay on Flash-Next.
- BlackWorld runs exactly as today.
- WhiteWorld's primary is Qwen3.6-35B, which no tenant uses yet. Flash-Next
  runs there as a parasite that never unmounts, which is the live test of the
  parasite path.
- Moving npcd to Qwen3.6 is a separate change made afterwards (§16, step 7).

## 1. Where things stand

**One model per process.**
- `host::MODEL` (`host/src/model.rs:20`) names `Qwen38_FlashNext`.
- zend takes it through `DaemonConfig.model` (`zend/src/config.rs:167`). npcd
  reads it directly (`npcd/src/engine/runtime/loader.rs:125,132`,
  `npcd/src/npcs/methods.rs:123`, `npcd/src/model.rs:86`).
- A tenant declares only the capabilities it needs (`Capabilities`,
  `zend/src/tenant.rs:42`, `npcd/src/tenant/mod.rs:87`).
- `host/src/run.rs:484` builds one `EngineSlot`. Its `Rendezvous` builds one
  engine core from the highest-ranked tenant's builder, and every other tenant
  attaches to it (`engine_slot/rendezvous.rs:83`, `build_engine.rs:257-274`).

**Everything below the scheduler belongs to the card.** Each of these is keyed
by device ordinal:
- one CUDA stream (`cuda_backend/device/backend_device.rs:56-94`);
- one span reservation (`region_pool/reservations.rs:24`);
- one wave domain: `live_generations`, `planned`, the arena window
  (`bump_arena/wave_domain.rs:188`);
- one weight floor (`region_pool/weight_side.rs:178`);
- one ground broker, newest wins (`weight_side.rs:56-63`).

**Dense weights fill a one-shot block.**
- The span's dense block is a bump allocator. It grows only before the first
  region exists, is frozen at the first model's `close_load`, and has no free
  path (`region_pool/dense_block.rs:56-218`).
- A second model's load today finds the reservation taken and puts its weights
  on the CUDA pool, outside the span (`dense_span.rs:62-99`).
- Measured dense sizes: Flash-Next 5.2–5.3 GiB (`qwen4exp/engine.rs:551`,
  `unbounded_agents.md` figure 9.6-4), Qwen3.6-35B 1,827 MiB
  (`dense_span.rs:20-22`).

**The trunk pack stores raw checkpoint bytes.**
- Flash-Next's pack is `PackFamily::Routed`: dense tensors verbatim, experts as
  KO records (`quantized_qwen38_moe/engine_artifact.rs:106`,
  `self_contained_model_packs.md` §3.1).
- The loader repacks almost every dense weight on the card into KO twins: the
  head is banded from the host, the HC modules are built from F32 (2.6 GB across
  the stack), and the PLE fusion and MTP split are built too
  (`qwen4exp/engine.rs:188-461`, `hyper/ko.rs:55-103`, `ple_fused.rs:75-101`,
  `mtp.rs:186-232`). That needs pool transients well beyond the 512 MiB cushion.
- Only the vision tower's pack is `PackFamily::Twinned`, which stores every
  matrix as its KO twin, so placing it is a straight copy
  (`model_pack/family.rs:137`, `qwen_vision/source.rs:78`).

**Guests borrow the card between waves.**
- The image generator, the matte model and the vision tower are `GuestModel`s.
- `drain_guests` runs at the top of a scheduler pass (`scheduler_loop.rs:156-197`).
  It hands back the tier, sheds, claims span regions (`GuestGround`), loads the
  guest, runs its whole backlog, unloads it, and drops the ground. Every forward
  on the engine is blocked meanwhile (`guest/drain.rs`, `scheduler/guest_room.rs`).
- A guest is built fresh for each drain and dropped at its end
  (`guest/model.rs:150-156`).
- Measured costs:
  - an image draw is 6.3–8.2 s, load about 3 s (`guest_draw_orchestration.md`);
  - the vision tower's 700 MiB copy takes 150–190 ms, its drain about
    720–760 ms (`vision_handover.md`).
- No guest job carries a band (`guest/queue.rs:31-41`).
- One vision tower per engine (`GuestRegistry::vision`, `guest/model.rs:168-174`).
- One guest arena and one guest heap per device
  (`bump_arena/guest_arena.rs:18-76`, `guest_heap.rs:150-200`).

**The cluster already locks a model per app.** The gateway keeps one `ModelId`
per app name (`cluster/src/gateway/admission.rs:35-43,146-190`). Each tenant
sends its own engine's key (`zend/src/session/load/startup.rs:517`,
`npcd/src/engine/runtime/loader.rs:245`).

**Calibration is already per model,** because a substrate is bound to the model
that wrote it (`shared_model_host.md` §4) and calibration lives in the substrate
(`host/src/calibration/phase.rs:422`).

## 2. Decisions

1. **The card picks its primary by size.** Below 32 GB (decimal) the primary is
   `Qwen36_35B_A3B_AntiLoop_StyleTune`; from 32 GB it is `Qwen38_FlashNext`
   (§3).
2. **A tenant names the model it runs.** `host::MODEL` goes. The host builds one
   engine core per distinct model named, plus the card's primary (§4).
3. **The primary is always mounted.** It loads at start whether or not a tenant
   names it, into the card's dense block, and it never unmounts.
4. **Everything else is a parasite.** Its dense weights load into claimed span
   regions on demand and leave when it unmounts (§6).
5. **Trunk packs store their weights already repacked.** A load, primary or
   parasite, copies them. The packs live in the model pack cache folder and are
   resolved and rebuilt by its resolver, like every other pack (§5).
6. **One driver thread per card.** Every core's scheduler is a pass the driver
   runs. Every core's housekeeping runs every pass, and only the model holding
   the card forwards (§7).
7. **An LLM parasite is a switch mode.** It has a slice, an idle grace, and a
   starvation bound (§8).
8. **Guests are mounts with a cooldown.** A run of draws or picture turns pays
   its load once (§9).
9. **A vision tower is a companion of its text model.** It never stops that
   model's forwards, and it mounts only while that model is mounted (§9.3).
10. **One expert cache per card,** keyed by `(model, row, expert)`, under one
    eviction rule on the card's clock. An unmounted model's experts survive
    until something with more use needs their slots (§11).
11. **A model's pinned rows are protected only while it is mounted** (§11.4).
12. **Timers that measure a model's progress run on that model's held clock,**
    the time it has held the card (§8.3).

## 3. The primary

```rust
// host/src/primary.rs
/// The card's primary: the model resident from load to exit.
pub fn primary_for(card_total_bytes: u64) -> Model {
    if card_total_bytes < PRIMARY_FLASH_NEXT_BYTES {
        Model::Qwen36_35B_A3B_AntiLoop_StyleTune
    } else {
        Model::Qwen38_FlashNext
    }
}

/// 32 GB, decimal: the smallest card Flash-Next is primary on.
pub const PRIMARY_FLASH_NEXT_BYTES: u64 = 32_000_000_000;
```

The boundary is in decimal bytes so that a 32 GB card lands on Flash-Next. An
RTX 5090 reports 32,607 MiB, which is 34.2e9 B but only 31.8 GiB.

| Card | Reported | Primary |
|---|---:|---|
| RTX 4090 Mobile | 16 GB | Qwen3.6-35B |
| RTX 3090 (WhiteWorld) | 24,576 MiB = 25.8e9 B | Qwen3.6-35B |
| RTX 5090 | 32,607 MiB = 34.2e9 B | Flash-Next |
| RTX PRO 5000 (BlackWorld) | 72 GB | Flash-Next |

**Only the primary has a ladder.** No tenant chooses a model by card size.
Flash-Next keeps its rung ladder wherever it runs, primary or parasite (`Q3_KO`
below 64 GiB, `Q4_KO` to 86 GiB, `Q8_0` above;
`candle_transformers::models::quant_ladder`). A rung is not another model
(`shared_model_host.md` §5.1).

**Which packs a machine holds.** A machine fetches and builds packs for its
primary and for every model its tenants name. On WhiteWorld that is both
Flash-Next and Qwen3.6. On BlackWorld it is Flash-Next only, until a tenant
names Qwen3.6.

## 4. Tenants and engine cores

**Each tenant names its model:**

```rust
// zend::tenant / npcd::tenant
fn model(&self) -> Model;          // Qwen38_FlashNext, both, in the first build
fn requires(&self) -> Capabilities; // as today
```

- Every site that reads `host::MODEL` today reads the tenant's model instead:
  - zend's config and its standing asks (`cli.rs:303`);
  - npcd's loader, early substrate open, console page and mission replay;
  - `ModelView` for the prompt's `{model}` placeholder (`zend/src/session/builders.rs:370`).
- `host/src/build_info.rs:20`'s single `Opens.model` becomes a map from app to
  model override key, so the upgrade compatibility check
  (`cluster/src/upgrade/compat.rs:41`) compares each app's model.

**One slot per model.**
- `host::run` groups tenants by model and builds one `EngineSlot` per group,
  with one rendezvous each.
- The primary's slot is built first, so its load fills the dense block, even
  when no tenant names it.
- A primary core with no tenant has a scheduler, an attached nothing, and its
  weights mounted. It never forwards.
- Standing tenants' asks (`Seat::Stands`) carry their model, so they merge into
  their own model's slot.
- Capabilities are checked per tenant against its own model, as today
  (`engine_slot/mod.rs:186-195`).

**A parasite core starts unmounted.** It opens its model's permanent parts and
attaches its tenants' substrates. Its dense weights wait for the first mount.

## 5. Trunk packs

**The trunk pack becomes the routed family with twinned dense weights:** the
expert section as today, and the GGUF part holding every int8-tileable dense
matrix as its KO twin. This is the vision tower's mechanism
(`model_pack/twins.rs`), widened to the trunk.

**What the pack stores:**
- **Matrices the loader twins today, stored as their twins:** attention, DeltaNet
  and shared-expert projections, the stacked projections, the router, the
  shared-expert gate, the LM head and the MTP draft head.
- **Tensors built at load today, stored in their built form:**
  - the HC modules, concatenated, padded to tile and twinned;
  - the PLE key|value fusion;
  - the shared gate padded to 32 rows;
  - the MTP `eh_proj` halves.

  Each model's pack build produces these, as `qwen35/pack_build.rs` already does
  for its family, so the loader reads them as stored.
- **Verbatim, as today:** the embedding (Q8_0, 644 MiB), norms, rotary inputs,
  the QSA indexer sources and the PLE table.

**Fingerprints.** Every `(source → twin)` pair is fingerprinted and re-checked at
open, as the expert section's are (`self_contained_model_packs.md` §4.2). A pack
of the old family fails `validate` (`model_pack/family.rs:179`), is judged stale,
and is rebuilt.

**The rebuild re-fetches its sources.**
- The resolver rebuilds the way it always has (`model_pack/resolve.rs`): fetch
  the sources, build, publish, release the fetched sources.
- For Flash-Next that means re-preparing its artifact on each machine. That is
  once per machine, and the price of keeping one source kind.
- The pack keeps only what a load reads. The original bytes of a twinned tensor
  are not stored beside its twin.

**What the load does:**
- copies every dense tensor into its destination (§6);
- zero-pads the tail of each tensor the dequantising matmul reads as stored,
  `MATRIX_ROW_PADDING` elements (`guest/groundgguf.rs:286-313` is the precedent),
  because claimed regions are reused and carry no zero tail of their own;
- gives KO twins their exact size.

**The primary loads the same way.** There is one load path, and the destination
is the only thing that differs. The primary's startup gets faster because
nothing is repacked on the card.

**What stays on the pool.** The norms, rotary tables and QSA indexer weights
stay on the CUDA pool for the core's life, mounted or not:
- they are small: the indexer weights are about 81 MiB across 13 layers, and the
  norms and tables less;
- they are read every forward;
- the QSA indexer weights read wrong from dense ground (`qwen4exp/engine.rs:404-408`).

They are allocated when the core opens and are part of the model's permanent
footprint.

## 6. Mounting a model

**The model's lifecycle splits three ways:**
- **`open`** builds what the core keeps for life: the tokenizer, the pool
  residents (§5), the model's expert registration with the card cache (§11),
  and the batched model's host-side state.
- **`mount(placer)`** copies the dense tensors from the pack through a placer and
  builds the `Qwen4ExpGpu`, `Qwen4ExpBatched` or `HybridBatched` views over
  them. There are two placers:
  - `DenseBlock`, for the primary: `claim_dense`, then `freeze_dense`, as
    `dense_span` does today;
  - `Ground`, for a parasite: placements in a `GuestGround`, never crossing a
    run (`guest/ground.rs`).
- **`unmount`** drops the views, then the ground. The views are rebuilt on the
  next mount. Nothing caches a raw device pointer to a dense weight: every
  descriptor table is rebuilt per forward or per placement, and graphs re-record
  each forward with that forward's addresses (`decode_graphs.md` §2).

**A parasite mount, in order:**
1. **Claim the ground** for the model's dense bytes. They are the sum of its
   placements, padded, and are read from the pack's own tensor table. Claim
   through `GuestGround::claim` after the ladder (§10).
2. **Copy** through the pinned staging pair (`guest/staging.rs`), in pack order.
   The rate is about 3.3 GB/s (`groundgguf.rs:85-86`: 7.2 GB in 2.19 s). So
   Flash-Next's 5.3 GiB is about 1.7 s, and Qwen3.6's 1,827 MiB about 0.6 s.
3. **Declare** the ground read-only to `readonly_regions` once the copy lands.
4. **Build** the model's views over the placements.

The largest single placement is a 644 MiB embedding or a ~625 MiB head twin, so
the claim must hold a run of at least that size. `GuestGround::claim` asks for
runs of up to 64 regions (1 GiB), halving on refusal. A mount whose ground
cannot hold its largest tensor in one run is refused with `GroundError::TooLarge`
and waits for room, like a guest.

**An unmount, in order:**
1. **Release** the ground's read-only declaration (`release_below` for its
   runs).
2. **Drop** the views, then the ground.
3. **Let the weight side reclaim** (`reclaim_spare_ground`) and restore its floor.

The parasite's KV, recurrent state and gallery are **not** touched. They are
ordinary tenants of the card, relieved by the ladder when room is wanted (§10),
so a parasite that comes back finds them where they were. Its experts stay in
the card's zone (§11).

## 7. The card driver

**One thread per card runs every core's scheduler.** `Scheduler::run`'s loop
body becomes two calls on the struct, which already holds all the loop's state
(`scheduler/mod.rs:334-1033`):
- `housekeep(&mut self)`: submissions, prefill promotion, the deferred arenas
  its persistence thread is waiting on (`create_deferred_arenas`), parking,
  demotion, gap work, compaction, `drain_reclaimed`, telemetry;
- `quanta(&mut self) -> PassOutcome`: the wave quanta and their retry state.

The one-time preamble (`own_device_frees`, the free target) becomes the driver's,
run once per card.

**Each pass of the driver:**
1. `housekeep` for **every** core, the holder last. A non-holder's persistence
   thread finishes its warm copies only when its own scheduler creates the
   arenas it deferred (`persistence/thread/pass.rs:134`, `drive.rs:259-270`).
   Skipping it would cost about 8 s per pass and leave the turns hot.
2. The guest poll (§9).
3. The switch rules (§8). A switch runs the steps below.
4. `quanta` for the holder only.
5. Park when every core reports nothing to do. The three blocking waits
   (`scheduler_loop.rs:306-351,382,803`) become a returned wait, and the driver
   waits on every core's request channel and the guest queue's wake together
   (`flume::Selector`).

**A switch, between forwards only:**
1. `end_wave_transient`. Then clear the outgoing model's recorded plan with a new
   `clear_wave_plan(ordinal)`, so no `begin_wave` of the incoming model lays out
   the outgoing model's plan (`bump_arena/wave.rs:78`).
2. Require that `live_generations` is zero. A head span still held by the
   outgoing model refuses the incoming model's claims (`wave.rs:252-282`), so the
   switch waits for it to be returned.
3. Settle the card: one context sync, then `drain_reclaimed`. After it, every slot
   the outgoing model's invocations could have snapshotted is reclaimable (§11.3).
4. Mount or unmount as the rules require (§6).

**What moves from the scheduler to the driver:**

| Today | Becomes |
|---|---|
| `set_kv_free_target(gpu_id, …)` per scheduler (`kv_negotiation.rs:106`) | once per card |
| `set_prompts_waiting(gpu_id, …)`, last writer wins | the sum over cores |
| `reseed_achievable_weight`, `ACHIEVABLE_WEIGHT` with device 0 hard-coded (`interleave.rs:41,134`) | per card |
| the ground broker, newest cache wins | the card's expert cache (§11) |
| `take_starvation` | read by the driver, relieved through every core (§10) |

## 8. The LLM switch mode

### 8.1 The rules

This section uses three terms:
- `demand(m)` is the highest band of work waiting or running for model `m`, or
  none.
- `holder` is the model holding the card.
- `taken_at` is when the holder last took it.

The driver evaluates the rules between passes.

1. **The primary holds the card by default.** A parasite with demand takes the
   card at the first pass where the primary has held it for at least `SLICE`, or
   has no demand at the parasite's band or above.
2. **A parasite holds the card for at least `SLICE`** from `taken_at`, while it
   has demand.
3. **After `SLICE`, a parasite keeps the card** until primary work at or above the
   parasite's band arrives. A parasite with no demand stays mounted, and a
   forward for it carries straight on.
4. **When the primary takes the card back, the parasite unmounts** (§6).
5. **Idle grace.** A parasite with no demand for `IDLE_GRACE` gives the card to
   waiting primary work, even inside its slice. The slice protects work, not
   idleness. `IDLE_GRACE` is longer than the gap a tool call leaves between two
   steps of one turn.
6. **Starvation bound.** Work that has waited `STARVE_BOUND` at a lower band than
   the holder's takes the card. This is the airtime guarantee
   `continuous_fair_waves.md` gives `Low` beside `High`, applied between models.
7. **Two parasites.** A parasite with demand while another holds the card waits by
   the same rules. When it takes the card, the outgoing parasite stays mounted
   unless the incoming mount needs its ground (rung h, §10).

### 8.2 The constants

They are set by measurement on WhiteWorld before LLM parasites switch live (§16,
step 7). The starting values:

| Constant | Bound | Starting value |
|---|---|---|
| `SLICE` | several times the remount cost: about 1.7 s for Flash-Next, 0.6 s for Qwen3.6 (§6) | 10 s |
| `IDLE_GRACE` | above a tool round trip | 2 s |
| `STARVE_BOUND` | the longest a waiting band stands still | 4 × `SLICE` |

### 8.3 Progress timers run on the held clock

A model that does not hold the card makes no progress, and that is not a stall.

**The held clock.** Each core keeps one: the time it has held the card, which
advances only while it is the holder.

**Timers that measure a model's progress read that clock:**

| Timer | Where | Today |
|---|---|---|
| `STALL` 60 s | `candle-conversation/src/handle.rs:21` | would fail a background wait already queued ~20 s before a 40 s pause |
| `PROMOTE_STALL_GRACE` 15 s | `scheduler/prefill/admitting.rs:21` | halves the admit budget once on resume |
| `STALLED_PROMPT` 30 s | `prefill/decode_priority.rs:75` | inflates the backlog reported to the gateway |
| `HighDrain` `TAU_SECS` 60 | `high_drain.rs:24` | reads a pause as zero throughput |
| `SweepForecast` | `sweep_forecast.rs:26` | counts the pause in a sweep's duration |
| npcd `SEAL_WAIT` 20 s, `DURABLE_WAIT` 30 s | `npcd/src/engine/mind/think.rs:18`, `talk.rs:72` | drop a mission's context, or let a talk go before its state is durable |

The engine exposes the held clock to its tenants, and npcd's two waits are
measured against it.

**Capacity is reported per app.** `QUIET` (`host/src/cluster/capacity.rs:32`)
reads a paused core as `idle`. The capacity frame therefore carries each app's
own waves, rates and idleness from its own core, rather than whichever model
forwarded last (`phase_ring.rs:717,727`).

**A switch happens only at a forward boundary,** so the routing summary's
`SUMMARY_DEADLINE` (`expert_lre/dispatch/ring.rs:156`) never measures a pause.

**The interim never switches LLMs.** BlackWorld has no parasite, and WhiteWorld's
Flash-Next holds the card throughout. None of these timers sees a pause until two
LLMs alternate.

## 9. Guests and companions

### 9.1 What changes

- **A guest is built once and kept.** `GuestRegistry::build` stops building a
  fresh instance per drain. The registry keeps one instance per configured guest,
  loaded or not, and `unload` is what returns it to unloaded.
- **Each mount has its own arena or heap.** The guest arena and guest heap are
  keyed by mount, not by device, so a companion tower and an image drain do not
  share or close each other's.
- **A guest job carries its requester's band:**
  - a draw asked for by a character's turn carries that turn's band;
  - a picture turn's vision job carries the turn's band;
  - an MCP draw is `Normal`.

  The band is what §8's rules and the ladder compare.
- **One vision tower per model.** `GuestRegistry::vision` becomes a map from model
  to tower. `Guest::Vision` jobs name the model whose rows they produce, and
  `VisionRequest` carries it.

### 9.2 The image generator and the matte model

A guest's backlog still runs to completion on the pass that finds it, with
nothing else forwarding, as the drain does today. What changes is the end:

- **If the mount fit with the ordinary rungs** (a–d, §10), it stays mounted for
  its cooldown. The holder forwards beside it with less room. A job inside the
  cooldown runs at once.
- **If it needed a deep rung** (e–g), it unmounts at the end of its backlog, as
  today. Those rungs are legal only while nothing forwards.

**The image generator's two stages stay mounted together.**
- Its text encoder and its transformer share one ground today, taken in turn: the
  claim is the larger stage, 8.4 of 11.8 GB on a 3090 (`guest/image.rs:7-9`).
- A cooldown that kept only the transformer could not condition a new prompt
  (`image.rs:881-888`).
- So a mount that will stay claims both stages side by side. Where that fits the
  ordinary rungs, as on BlackWorld, a run of draws pays one load.
- Where only the larger stage fits, as on WhiteWorld, the drain stages them in
  turn and unmounts, as today.

**The matte model** keeps its placed weights. Its graph is already cached for the
process (`guest/checkpoint.rs:92`).

### 9.3 The vision tower, a companion

- It mounts only while its text model is mounted, which is always true when that
  model is the primary.
- It serves its jobs between that model's waves, with the model forwarding beside
  it. It takes only the ordinary rungs.
- Its model's forwards neither wait for it nor end it.
- It stays mounted for its cooldown after its last job.
- Unmounting its text model unmounts it first.

A picture turn then pays the 150–190 ms copy once per run of turns, not a
720–760 ms drain that blocks every forward on the card. Rows already in the
substrate skip the tower entirely, as today (`vision/store.rs:173-200`).

### 9.4 Cooldowns

| Mount | Starting cooldown |
|---|---|
| Image generator | 60 s |
| Matte model | 60 s |
| Vision tower | 120 s |

A cooldown ends early when rung c takes the mount's ground (§10).

## 10. The shedding ladder

This is the ladder a mount walks to its footprint, cheapest first. It is the
guest drain's ladder in its current order, with two rungs added.

| Rung | What | Restored by | New |
|---|---|---|---|
| a | hand back the transient tier | the next wave's tier | |
| b | KV relief: compress, evict the cold tail, release empty arenas. Walks every core's substrates, non-holders first | elevation on demand | across cores |
| c | drop idle guest and companion mounts past their last job, least recently used first | their next load | yes |
| d | the weight side concedes down to the serving floor | experts page back in | |
| e | the expert zone concedes below the serving floor | `restore_floor` before the next forward | |
| f | empty every core's gallery, non-holders first | the next scan | |
| g | stash the pinned experts in host memory | `restore_floor` | |
| h | unmount an LLM parasite that does not hold the card | its next mount | yes |

**Rung e for an LLM mount.** For a mount of an LLM parasite, rung e cuts nothing
below the parasite's own serving floor, because the parasite forwards through the
same zone.

**Rung g works for every model,** because it is the card cache's (§11). Qwen3.6's
batched model does not implement it today (`qwen35/forward/managed_model.rs`).

**Unchanged on WhiteWorld in the interim.** Rung h comes last, so a draw there
while Flash-Next is mounted sheds exactly what it sheds today. It unmounts
Flash-Next only where today's draw would have been refused for room.

## 11. The card's expert cache

### 11.1 What moves to the card

A `CardExperts` per ordinal owns what one `ExpertCache` owns today
(`expert_lre/handle/mod.rs:140-199`, `cache/mod.rs:56-106`):
- the weight zone, weight floor and ground broker;
- the warm tier, pinned and pageable, and the pad;
- one pipeline thread, one stager with its readers, and the elastic controller;
- the CPU expert pool;
- the copy stream.

The loaders stop calling `set_weight_floor` and `set_ground_broker`
(`qwen35/expert_loader.rs:154,183-189`). They register the model instead.

**What a registration carries:** model id, rows, experts per row, per-row
geometry, top-k and the expert section. In return the model gets its `Dispatch`:
its own live tables, routing summary ring, bucketize workspace and pass state,
over the card's slots.

Each model keeps its own live table. They cannot be merged, because every row of
one table shares row 0's shapes (`live_table.rs:157-178`).

### 11.2 Keys, slots and strides

**Every map and score is keyed `(model, row, expert)`:** `key_to_slot`,
`slot_to_key`, `expert_scores`, `decode_reuse`, `hit_last`, the host-backed
bitmap, paged images, resident bands, the regret ledger, the transition matrix
and the tensor-assert owner tags.

**One slot class.**
- A slot is sized to the largest expert of the models that can route on the
  card: every model a tenant names, plus the primary when a tenant names it.
- A primary no tenant names is never routed and does not widen the slots. So
  WhiteWorld's slot in the interim is Flash-Next's 1,996,800 B at `Q3_KO`, as
  today.
- Qwen3.6's experts are 1,933,312 B at its Performance KO rung and 2,195,456 B at
  Precision. Once npcd names it on WhiteWorld, the slot is 2,195,456 B under
  Precision, and Flash-Next's experts pad by 9%.

**Pack stride and slot stride separate.** Today the pad and warm stride must
equal the pack stride (`handle/construction.rs:248-253`, `stager.rs:460-505`).
Each model's section keeps its own record stride. The card's zone, pad and warm
tier use the slot stride, and a read lands a record at offset 0 of a slot.

### 11.3 Clocks and reclaim

**One card pass, one decay.**
- Today the scores decay once per pass of the model that owns them, at its last
  row (`pipeline/routing.rs:393-399`).
- Here they decay once per card pass, every model's at once. An unmounted model's
  experts age, so a model that ran once does not keep its slots forever.
- Every routed use of a resident expert is one hit, whichever model routed it.

**Distance across models.** `forward_distance` is defined only within the holder's
layer cycle (`cache/bookkeeping.rs:400-406`). A row of another model is treated
as maximally behind, so it gets no position-based protection, and only its hits
keep it.

**Reclaim across models.**
- Within a model, reclaim stays as it is: tickets per dispatch, started words per
  row (`expert_lre/reclaim.rs`).
- A slot moving from one model's expert to another's is reused only after a
  switch's settle (§7). After the settle, `settled()` makes every slot of the
  outgoing model reclaimable (`reclaim.rs:153-155`).
- Only the holder forwards, so the settle at each switch is the whole cross-model
  rule.

**The promotion ring's victims carry their model.** The device-side retarget
reads a device table of the models' live-table pointers and writes the victim's
own model's table. That is a descriptor table, per invariant 2b, replacing the
`row·E+expert` encoding against one table's width (`promo.rs:72-80`).

### 11.4 Pinned rows and the startup fill

**Pinned rows are protected only while mounted.** A model's first
`PINNED_LAYERS` rows are protected while it is mounted (`policy.rs:39`). The
primary is always mounted. An unmounted parasite's pinned rows are ordinary
experts under least-hit.

**The startup fill.**
- The VRAM fill and the warm tier's pinned part are dealt at load across the
  models a tenant names, primary first when named. Each model has its own seed
  (`VRAM_DRAW_SEED`, `WARM_DRAW_SEED`).
- The pinned part is sized once at load and never moved (`elastic_warm_tier.md`).
- In the interim on WhiteWorld only Flash-Next is named, so it is dealt the whole
  fill and the whole 28 GiB pinned part (`host/machines.yaml`), as today.

**The serving floor** is the holder's `minimum_resident_slots(E)`. That is 1,537
slots for Flash-Next and 769 for Qwen3.6 (`policy.rs:147-149`). `restore_floor`
restores the floor of whichever model is about to forward.

### 11.5 Process-wide state the cache keeps

| Today | Becomes |
|---|---|
| `WARM_SIZING` (`handle/sizing.rs:157`) | the card cache's |
| `candle::vram::set_weights_mmap`, overwritten per loader (`vram/mod.rs:591`) | the sum over registered models |
| `GROW_TALLY`, `layer_trace::LATEST`, `slot_integrity::WATCH` | the card cache's, traces keyed by model |
| `cpu_pool::params::shared` and the service statics | one pool per card, shared by its models |
| `PipelineStats` | per model, inside the card's |

## 12. Process-wide state outside the cache

| Today | Why it breaks with two cores | Becomes |
|---|---|---|
| `conversation/prefix_state.rs:136` `HELD`, keyed by token hash | two models with one tokenizer would be handed each other's recurrent state | keyed `(model, hash)` |
| `provenance/wide_sig.rs:160` `ACTIVE_FOLD`, first wins | the second model's signature records are refused | the core's own fold, from `Scheduler::prov_fold` |
| `scheduler/memory_report/mod.rs:56` `LATEST`, last wins | zend's admission widths read the other core's report (`zend/repo_scan/kv_estimate.rs:83`, `width.rs:114`) | per core, read through the tenant's engine |
| `scheduler/phase_ring.rs` rings, `LEARNED`, `LEARNED_STORE` | rates mixed across models, one `learned_rates` persisted | per core, persisted per model |
| `scheduler/counters.rs`, `IN_DRAIN`, `PREFILL_OK_TOKENS` | summed across cores | per core |
| `guest/reach.rs:30` `STATED` | last core wins | per card |
| `substrate/residence.rs:120` `RECLAIM` queue, rewritten by raw gid | gid namespaces are per backing, so two models' ids can collide (`backing/mod.rs:63`) | each entry carries its backing |
| `index_page_cache/install.rs:18`, sized at the first install | the second core finds it sized for the first | per card, sized over its models |
| `models/builder/pack.rs:65`, `engine_model_digest()` for any prepared spec | a second prepared model gets Flash-Next's digest | the model's own recipe digest |

## 13. The cluster

**The model lock is per app,** as it already is (§1). Which model is a member's
primary is not part of any lock. A zend member on WhiteWorld runs Flash-Next as a
parasite and one on BlackWorld runs it as its primary, and both satisfy zend's
lock.

**What the capacity frame reports.** It carries each app's figures from its own
core (§8.3), and it names the card's primary and its mounts.

**`/v1/status` and `/v1/telemetry`** report the primary and every mounted
parasite, each with its `ModelView`. Each tenant's entry names its model.

**Placement is unchanged.** `host/machines.yaml` and `static_placement.md` already
send most conversations to BlackWorld and most pictures to WhiteWorld. Moving the
cast to WhiteWorld is part of step 7 (§16), not before.

## 14. What the first build does on each machine

| Machine | Primary | Tenants | What runs | Against today |
|---|---|---|---|---|
| BlackWorld | Flash-Next | zend, npcd on Flash-Next | the primary only; guests as mounts | the same, plus draws (both stages fit) and picture turns that stay loaded |
| WhiteWorld | Qwen3.6-35B AntiLoop | zend, npcd on Flash-Next | Flash-Next mounted on the first forward and never unmounted; Qwen3.6 loaded and never forwarded | the same fill, slots and pinned tier, minus Qwen3.6's 1,827 MiB of dense ground |

**The one cost.** WhiteWorld's unused primary takes 1,827 MiB out of the span,
about 960 of Flash-Next's 1.9 MB slots. That is the price of running the parasite
path live before npcd moves. The expert hit rate and decode rate on WhiteWorld
are recorded before and after step 6, so the price is measured, not assumed.

## 15. Tests

Codec and placement code is tested against raw expected bytes, per the repository
rule.

- **Packs:**
  - a twinned trunk pack's GGUF part round-trips as fixed bytes for a fixture
    model;
  - the built HC, PLE, gate and MTP tensors equal, byte for byte, what today's
    loader builds from the same fixture;
  - a pack of the old family is judged stale;
  - a load from the new pack produces the same logits as today's load, bit for
    bit, on the forward gates.
- **Placement:**
  - a `Ground` placer's placements never cross a run;
  - a verbatim tensor's tail is zero;
  - a mount refused for a run too short reports `TooLarge`;
  - a mount and unmount leaves the region pool as it found it.
- **The switch rules** run against a fake clock and fake cores, with no card.
  Every rule in §8.1 is tested, and so are the held clock and the starvation
  bound.
- **The driver:**
  - `housekeep` runs for every core every pass;
  - a switch ends the tier, clears the plan and settles;
  - a switch waits for `live_generations == 0`.
- **The card cache:**
  - two registered models' experts share slots under one score;
  - an idle model's scores decay on the card's clock;
  - a slot moving between models is reused only after a settle;
  - a victim carries its model through the promotion ring;
  - the slot class follows the routable models;
  - every existing expert-cache test, with one model registered.
- **Guests:**
  - a second draw inside the cooldown loads nothing;
  - a mount that took rung e or g unmounts at its backlog's end;
  - a companion's model forwarding does not end it;
  - unmounting a model unmounts its companion first;
  - the ladder's order.
- **Probe, two models on one card:** a Qwen3.6 conversation and a Flash-Next
  conversation alternating across ten switches, each answering as it does alone.
- **Live:**
  - BlackWorld unchanged against its current figures (`performance.md`);
  - WhiteWorld with Flash-Next mounted from its first forward, hit and decode rates
    recorded against today's.

## 16. Plan

Each step lands green, deploys on its own, and leaves the fleet operating as
before. Steps 1–6 are the first build. Step 7 is a separate change.

| # | Step | Delivers |
|---|---|---|
| 1 | **Tenants name their model** | `host::MODEL` removed; `Tenant::model`; one slot per model; `Opens.model` per app; the §12 fixes that need no second core (`HELD` keyed by model, `RECLAIM` entries carrying their backing, the prepared-spec digest) |
| 2 | **Twinned trunk packs** | the trunk family with twinned and built dense tensors; the loader as a copy through `DenseBlock`; packs rebuilt from re-fetched sources on each machine |
| 3 | **The card's expert cache** | `CardExperts` with one model registered: keys by model, slot and pack strides separated, the card clock, victims carrying their model, rung g for every model |
| 4 | **The card driver** | `housekeep` and `quanta`; one driver over one core; the per-card calls moved; the wait across channels; the held clock and per-app capacity |
| 5 | **Guests as mounts** | kept instances, per-mount arenas and heaps, cooldowns, rung c, companions, one tower per model, bands on jobs, the image generator's two stages together where they fit |
| 6 | **LLM parasites** | `open` / `mount` / `unmount`; the `Ground` placer; the switch mode; rung h; the primary by card size; several cores per driver. Live: BlackWorld unchanged, WhiteWorld with Flash-Next as its parasite |
| 7 | **npcd on Qwen3.6** (separate, by decision) | `npcd::tenant::model` is Qwen3.6 AntiLoop; npcd on a new substrate; the cast moved to WhiteWorld in `host/machines.yaml`; §8.2's constants measured with two LLMs alternating |
