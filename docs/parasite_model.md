# Parasite models: one card, a primary model, and the models that borrow it

**Status:** design, not built. It replaces `docs/shared_model_host.md`'s
single prescribed model (decision 7, §5) and turns the guest drain
(`candle-conversation/src/guest/`) into one of the cases of a parasite mount.

## 0. Summary

Every card has one **primary** model. It is chosen by the card's size and is
resident from load to exit. Any other model a request needs is a **parasite**:
it is mounted into ground the primary gives up, it holds that ground while it is
used, and it unmounts when the primary wants the ground back. An LLM for another
tenant, the image generator, the cut-out (matte) model and a vision tower are
all parasites, served by one mechanism.

The aim of the first build is **multi-model support with no change in how
anything operates**:

- zend and npcd both stay on Flash-Next.
- BlackWorld runs exactly as today.
- WhiteWorld runs Flash-Next as a parasite that never unmounts, because nothing
  there uses its primary. That is the live test of the parasite path.

Moving npcd to Qwen3.6-35B is a separate change made afterwards (§9, step 5).

## 1. Where things stand

- **One model per process.** `host::MODEL` (`host/src/model.rs`) names
  `Qwen38_FlashNext`, and every tenant runs it (`shared_model_host.md` §5.1). A
  tenant declares only *capabilities* it needs of the model
  (`Capabilities`, §5.2).
- **One engine core.** It holds one scheduler, one model, one expert cache and
  one tokenizer. Its scheduler already serves several substrates
  (`shared_model_host.md` §1.1, phase 2).
- **The cluster has one model lock per group** (`cluster/src/gateway/admission.rs`,
  `cluster_status`'s `model_lock`).
- **Guests already borrow the card.** The image generator, the matte model and
  the vision tower are `GuestModel`s. The scheduler checks the guest queue once
  per pass (one atomic load). When it has work, `drain_one` does the following:
  1. hands back the transient tier;
  2. sheds KV and has the expert zone concede ground, down past its serving
     floor and the pinned experts if it must;
  3. claims 16 MiB span regions (`GuestGround`);
  4. loads the guest, runs its whole backlog, unloads it;
  5. drops the ground.

  Normal inference is blocked for the whole drain. There is no restore step: KV
  elevates on demand, experts page back in, and the zone takes its ground back
  (`guest/drain.rs`, `scheduler/guest_room.rs`).
- **Measured guest costs:**
  - an image draw is 6.3–8.2 s, of which the load is about 3 s
    (`guest_draw_orchestration.md`);
  - the vision tower's 700 MiB copies in 150–190 ms from its pack, about
    720–760 ms for the whole drain (`vision_handover.md`).

  Every one of those loads is paid again on the next request. A guest is built
  fresh for each drain and dropped at its end.

## 2. Decisions

1. **The card picks its primary by size.** Below 32 GB of device memory the
   primary is `Qwen36_35B_A3B_Q4`; from 32 GB it is `Qwen38_FlashNext`. The rule
   is one function of the card's total memory, so one binary serves every
   machine (§3).
2. **A tenant names the model it runs.** `host::MODEL` goes. zend and npcd each
   declare their model beside their capabilities. The host builds one engine
   core per distinct model its tenants and guests name, plus the card's primary.
3. **The primary is always mounted.** It is loaded at start whether or not a
   tenant names it, and it never unmounts.
4. **Everything else is a parasite, mounted on demand.** A mount claims span
   regions exactly as a guest's ground does today, loads the model's own
   weights into them, and holds them until it unmounts.
5. **Mounting holds ground; forwarding takes time. They are separate.** At most
   one model forwards at a time, the one that holds the card (§4.2). Several
   models may be mounted at once when the ground allows it.
6. **An LLM parasite is a switch mode, with a slice.** Once mounted, it holds
   the card for at least the slice. After that it keeps the card while nothing
   else asks for it, and its own forwards just carry on. Primary work at or
   above the parasite's band ends the mode and unmounts the parasite (§4.3).
7. **One expert cache per card, shared by every model on it.** A slot is keyed
   by `(model, row, expert)`. Every model's experts compete under one eviction
   rule, so an unmounted model's hot experts survive until something with more
   use needs their slots (§5).
8. **A parasite that serves its host's turn is a companion.** A vision tower
   lets its own text model forward while it is mounted, and those forwards never
   end its mount. It mounts only while its host is mounted. Unmounting the host
   unmounts it (§4.4).
9. **The cluster locks a model per app, not per group.** Every member serving
   zend runs zend's model, whether as its primary or as a parasite (§7).

## 3. The primary

```rust
// host/src/primary.rs
/// The card's primary: the model resident from load to exit.
pub fn primary_for(card_total_bytes: u64) -> Model {
    if card_total_bytes < PRIMARY_FLASH_NEXT_BYTES {
        Model::Qwen36_35B_A3B_Q4
    } else {
        Model::Qwen38_FlashNext
    }
}

/// 32 GB, decimal: the smallest card Flash-Next is primary on.
pub const PRIMARY_FLASH_NEXT_BYTES: u64 = 32_000_000_000;
```

**Why decimal bytes.** The boundary is meant to put a 32 GB card on Flash-Next.
An RTX 5090 reports 32,607 MiB, which is 34.2e9 bytes but only 31.8 GiB, so a
threshold of 32 GiB would put it on the wrong side. The fleet lands as follows:

| Card | Reported | Primary |
|---|---:|---|
| RTX 4090 Mobile | 16 GB | Qwen3.6-35B |
| RTX 3090 (WhiteWorld) | 24,576 MiB = 25.8e9 B | Qwen3.6-35B |
| RTX 5090 (production workstation) | 32,607 MiB = 34.2e9 B | Flash-Next |
| RTX PRO 5000 (BlackWorld) | 72 GB | Flash-Next |

**Only the primary has a ladder.** No tenant chooses a model by card size.
Flash-Next keeps its rung ladder (`Q3_KO` below 64 GiB, `Q4_KO` to 86 GiB,
`Q8_0` above, `candle_transformers::models::quant_ladder`) wherever it runs,
primary or parasite. A rung is still not another model, so a substrate written
on one card opens on another (`shared_model_host.md` §5.1).

**The status reports both.** `/v1/status` and `/v1/telemetry` report the
primary and every mounted parasite, each with its `ModelView`. A tenant's entry
names the model it runs.

## 4. Mounts

### 4.1 What a mount is

A mount is a model's weights standing in span regions it claimed, plus a
**cooldown**, the time it may stay mounted after its last use.

| Parasite | Weights in its ground | Its experts | Lets forward while mounted |
|---|---|---|---|
| LLM (another tenant's model) | its dense weights | in the card's shared zone (§5) | nothing else |
| Image generator, matte model | its checkpoint and activation arena | — | nothing else while its backlog runs |
| Vision tower | the tower | — | its own text model |

**What mounting does.** It is the guest drain's front half:
1. hand back the transient tier;
2. walk the shedding ladder (§4.5) to the mount's footprint;
3. claim the regions;
4. load the model.

**What unmounting does.** It is the back half: unload, drop the ground, and
have the weight side reclaim it (`reclaim_spare_ground`) and restore its floor.

**Where the weights load from.** A mount loads from the model's pack through
the pinned staging pair (`guest/staging.rs`). An LLM's dense weights load in
pack order, layer 0 first.

**Each mount is read-only while it stands.** Its regions are declared to
`readonly_regions` once the load ends, and released before its ground is
dropped. A stale declaration would blame the next tenant of that ground
(AGENTS.md, the `tensor-assert` dangers).

### 4.2 Who holds the card

The card has one **driver**, the thread that runs passes. It replaces each
engine core's own scheduler loop: every core's scheduler exposes its pass (the
submission drain, the guest poll, the wave quanta), and the driver runs the pass
of the model that holds the card.

**Why one thread.** Span claims, boundary moves and the transient tier are legal
only between forwards (`guest/ground.rs`, invariant 7). With one driver, "between
forwards" is a point in one loop rather than a handshake between two schedulers.

**A switch happens between passes.** The driver ends the transient tier
(`end_wave_transient`) and clears the outgoing model's recorded tier plan, so no
forward of the incoming model walks the other model's plan (invariant 7, third
rule).

### 4.3 The LLM switch mode

This section uses the following terms:

- `demand(m)` is the highest band of work waiting or running for model `m`, or
  none.
- `holder` is the model holding the card.
- `taken_at` is when the holder last took it.

The rules, evaluated by the driver between passes:

1. **The primary holds the card by default.** A parasite with demand takes the
   card on the next pass at which the primary has held it for at least `SLICE`,
   or has no demand at the parasite's band or above.
2. **A parasite holds the card for at least `SLICE`** from `taken_at`, while it
   has demand.
3. **After `SLICE`, a parasite keeps the card** until primary work at or above
   the parasite's band arrives. A parasite with no demand sits mounted; a
   forward for it carries straight on, with nothing reloaded.
4. **When the primary takes the card back, the parasite unmounts.** Its dense
   ground goes back, its KV is relieved to warm (§6), and its experts stay in
   the zone (§5).
5. **Lower-band work is not starved.** Work waiting at a lower band than the
   holder's takes the card once it has waited `STARVE_BOUND`. That is the
   airtime guarantee `continuous_fair_waves.md` gives `Low` beside `High`,
   applied between models.
6. **Idle grace.** A parasite with no demand for `IDLE_GRACE` gives the card to
   waiting primary work even inside its slice. The slice protects work, not
   idleness. `IDLE_GRACE` is longer than the gap a tool call leaves between two
   steps of one turn, so a turn in progress keeps its mount.

The constants are set by measurement on WhiteWorld before step 4 of §9 lands:

| Constant | Bound | Starting value |
|---|---|---|
| `SLICE` | several times the remount cost | 10 s |
| `IDLE_GRACE` | above a tool round trip | 2 s |
| `STARVE_BOUND` | the longest a waiting band stands still | 4 × `SLICE` |

**The remount cost that `SLICE` must cover:**
- Flash-Next's dense weights are 5.3 GiB on the device (BlackWorld's span,
  `unbounded_agents.md` figure 9.6-4). At the vision tower's measured copy rate
  of about 4 GB/s from the pack, that is about 1.3–1.5 s.
- Qwen3.6-35B's dense weights are estimated at 2–2.5 GiB, about 0.5–0.6 s. That
  estimate is measured at step 4.

**Two parasites.** A parasite with demand while another holds the card waits by
the same rules, primary and parasite alike. When it takes the card, the outgoing
parasite stays mounted unless the incoming one's mount needs its ground (§4.5).

### 4.4 Guests and companions

**The image generator and the matte model.** A guest's backlog still runs to
completion on the pass that finds it, with nothing else forwarding, as the drain
does today. What changes is the end:

- **If the mount fit with the ordinary rungs** (§4.5, a–d), it stays mounted for
  its cooldown. The holder forwards beside it, with less room. A draw that
  arrives within the cooldown runs at once with no load, so a run of draws pays
  the ~3 s load once.
- **If it needed the deep rungs** (e–g: the expert zone below its serving floor,
  the gallery, the pinned experts), it unmounts at the end of the backlog, as
  today. Those rungs are legal only while nothing forwards, and restoring them
  is what lets the holder forward again.

**The vision tower is a companion of its text model.**
- It mounts only while that model is mounted (always true when the model is the
  primary).
- It serves its jobs between that model's waves.
- That model's forwards neither wait for it nor end it.
- It stays mounted for its cooldown after its last job.
- Unmounting its text model unmounts it first.
- Today's tower drain blocks inference for 720–760 ms per image turn. As a
  companion, a run of picture turns pays the 150–190 ms copy once.

**Each tower belongs to its model.** The guest registry today holds one vision
source (`GuestRegistry::vision`). It becomes one per model, so Flash-Next's
tower and Qwen3.6's are two companions with two hosts.

**A guest job carries its requester's band.** A draw asked for by a character's
live turn is `High`, and a draw from an MCP call is the band that call's turn
runs at. The band is what rules 1, 3 and 5 of §4.3 compare.

### 4.5 The shedding ladder

This is the ladder a mount walks to its footprint, cheapest first.

| Rung | What | Restored by | New |
|---|---|---|---|
| a | hand back the transient tier | the next wave's tier | |
| b | KV relief: compress, evict the cold tail, release empty arenas, on **every** core's substrates, the cores not holding the card first | elevation on demand | across cores |
| c | drop idle guest and companion mounts past their use, least recently used first | their next load | yes |
| d | the weight side concedes ground down to its serving floor | experts page back in | |
| e | the expert zone concedes below its serving floor | `restore_floor` before the next forward | |
| f | empty the gallery | the next scan | |
| g | stash the pinned experts in host memory | `restore_floor` | |
| h | unmount an LLM parasite that does not hold the card | its next mount | yes |

**Whose floor rung e cuts.** Rung e is legal only while nothing forwards. For a
mount *of* an LLM parasite it cuts nothing below the parasite's own serving
floor, because the parasite forwards through the same zone.

**Unchanged on the WhiteWorld interim.** Rungs a and b–g are today's ladder in
today's order. Rung h comes last, so a draw on WhiteWorld while Flash-Next is
mounted there sheds exactly what it sheds today. It unmounts Flash-Next only
where today's draw would have been refused for room.

## 5. The shared expert cache

**One zone, one warm tier, one eviction rule per card.** The expert cache moves
from the model (`ManagedBatchedModel`'s expert handle) to the card. Each model
registers its expert geometry: rows, experts per row and bytes per expert. Each
model keeps its own per-row live tables, which `bucketize` reads, pointing into
the shared slots (`moe_live_dispatch_design.md`).

**One slot class.** A slot is sized to the largest expert of any model the card
serves, so every model's experts fit every slot.
- Flash-Next's slot is 1.9 MB at `Q3_KO`.
- Qwen3.6-35B's experts are of the same order: the ~22 GB file spread over
  10,240 experts (40 rows × 256) is about 2 MB each before the trunk is
  subtracted.

The padding a smaller expert leaves is reported at load, per machine, from the
two packs.

**Eviction stays least-hit with the pack-only tilt** (`expert_cache_design.md`).
What changes is the clock:
- **Hits decay on the card's clock**, one recency pass per card pass, not per
  pass of the model that owns the expert. An unmounted model's experts must age,
  or a model that ran once would hold its slots forever.
- **Hits compare across models as uses of a slot.** Every routed use of a
  resident expert is one hit, whichever model routed it, so one score compares a
  slot's value across models.

**The reclaim rule holds across models.** A slot is re-tenanted only once every
invocation that could have snapshotted its old address has finished
(`expert_lre/reclaim.rs`). A slot moving from one model's expert to another's
waits out the first model's invocations, exactly as within one model.

**The startup fill.** The VRAM fill and the warm tier's pinned part are dealt at
load across the models the tenants name, primary first. The pinned part is sized
once at load and never moved (`elastic_warm_tier.md`). On the WhiteWorld
interim, Flash-Next is the only model a tenant names, so it is dealt the whole
fill and the whole 28 GiB pinned part (`host/machines.yaml`), as it is today.
The primary named by no tenant is dealt none.

**The routing summary and residency scoring carry the model in their key.**

## 6. Engine cores, substrates and state

- **One engine core per model.** Each core has its own scheduler, tokenizer,
  gallery arena and recurrent store. A tenant attaches its substrates to its
  model's core, exactly as it attaches to the one core today (`EngineAttacher`).
- **A substrate stays bound to its model** (`shared_model_host.md` §4). zend's
  and npcd's substrates are both written by Flash-Next and open under
  Flash-Next's core on every machine, whatever the card's primary.
- **The KV side is the card's.** Regions are a device-wide pool already. Rung b
  walks every core's substrates, and an unmount relieves the parasite's hot KV
  to warm before its ground is dropped, so the primary does not inherit KV it
  cannot relieve.
- **The timeline allocator and the writer lock are already process-wide**
  (`shared_model_host.md` decision 5), so two cores mint no colliding ids.
- **Capabilities are checked per tenant against the model it names**, at start,
  as today (`Capabilities::check`).
- **Calibration and the battery run per model.** Every calibration record names
  the model it ran under, and a tenant reads only its own model's.

## 7. The cluster

- **The model lock is per app.** The gateway records one model key per app from
  its first member, closes a member that disagrees (4412, as today), and clears
  the app's lock when the app's last member leaves. `cluster_status` reports a
  lock per app. Which model is a member's primary is not part of any lock: a
  zend member on WhiteWorld runs Flash-Next as a parasite, one on BlackWorld as
  its primary, and both satisfy zend's lock.
- **Placement is unchanged.** `host/machines.yaml` and
  `docs/static_placement.md` already send most conversations to BlackWorld and
  most pictures to WhiteWorld. Moving the cast to WhiteWorld once npcd runs
  Qwen3.6 is a change of that table, made in step 5 of §9 and not before.

## 8. What the first build does on each machine

| Machine | Primary | Tenants | What runs | Against today |
|---|---|---|---|---|
| BlackWorld | Flash-Next | zend, npcd on Flash-Next | the primary only; guests as mounts | the same, plus draws and picture turns that stay loaded |
| WhiteWorld | Qwen3.6-35B | zend, npcd on Flash-Next | Flash-Next mounted on the first forward and never unmounted (nothing asks for the primary) | the same expert fill and pinned tier, less the primary's dense ground |

**The one cost of the interim.** WhiteWorld's primary is loaded but unused, so
its dense weights take ground out of Flash-Next's expert zone: an estimated
2–2.5 GiB, about 1,100–1,300 slots of 1.9 MB. That is the price of testing the
parasite path live before npcd moves. The expert hit rate and decode rate on
WhiteWorld are recorded before and after step 4 so the price is measured, not
assumed.

## 9. Plan

Each step lands green on its own, deploys on its own, and leaves the fleet
operating as before.

| # | Step | Delivers | Proved by |
|---|---|---|---|
| 1 | **Tenants name their model** | `host::MODEL` removed; `zend::tenant` and `npcd::tenant` each declare `Model::Qwen38_FlashNext`; the host builds one core per named model (one, today); the model lock per app | unit: each tenant's model passes its capabilities; two tenants naming one model share one core; the gateway's lock per app (a zend member and an npcd member on different models both admitted; two zend members on different models, the second closed 4412; an app's lock cleared when its last member leaves) |
| 2 | **The card's expert cache** | the zone, warm tier and eviction lifted to the card, keyed by `(model, row, expert)`; one model registered | every existing expert-cache test, unchanged; unit: two registered models' experts share slots under one score; an idle model's hits decay on the card's clock; a slot moving between models waits out the first model's invocations; the slot class is the larger expert |
| 3 | **Guests as mounts** | the cooldown; rungs c and h of the ladder; companions; one tower per model; a guest job's band | unit, without a card (an `EngineRoom` fake): a second draw inside the cooldown loads nothing; a mount that took rung e or g unmounts at its backlog's end; a companion's host forwarding does not end it; unmounting a host unmounts its companion first; the ladder's order. Live, BlackWorld: three draws in a row, one load |
| 4 | **The card driver and LLM parasites** | one driver thread over every core; the switch mode of §4.3; the primary by card size; status reporting the primary and mounts | unit: `primary_for` at the boundary (25.8e9 → Qwen3.6, 32e9 → Flash-Next, 34.2e9 → Flash-Next); every rule of §4.3 against a fake clock; a switch ends the tier and clears the plan. Probe: two models on one card, a Qwen3.6 and a Flash-Next conversation alternating, both answering correctly across ten switches. Live: BlackWorld unchanged; WhiteWorld with Flash-Next mounted from its first forward, hit and decode rates recorded against step 3's |
| 5 | **npcd on Qwen3.6-35B** (a separate change, by decision) | `npcd::tenant` names Qwen3.6; npcd on a new substrate; the cast moved to WhiteWorld in `host/machines.yaml` | npcd's tests under Qwen3.6; the battery under each tenant's model; live: the cast on WhiteWorld's primary, zend on BlackWorld's |

## 10. Open questions

1. **Which Qwen3.6 checkpoint is the small cards' primary:** the stock
   `Qwen36_35B_A3B_Q4` named in §3, or AntiLoop's
   `Qwen36_35B_A3B_AntiLoop_StyleTune` that npcd ran before Flash-Next. It
   matters only from step 5, when npcd's substrate is written under it.
2. **Whether the interim cost of §8 is acceptable.** If WhiteWorld cannot spare
   the primary's dense ground before step 5, the alternative is a rule that
   loads a primary only when a tenant names it. That rule makes WhiteWorld
   identical to today, but it leaves no machine where the LLM parasite path runs
   live before npcd moves.
