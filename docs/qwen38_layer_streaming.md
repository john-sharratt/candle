# Dense Layer Streaming — VRAM ↔ RAM ↔ NVMe, by layer

> **Status — Built, and running the 27B on a 16 GB card.** Extends the
> slot/residency machinery the expert cache already owns
> (`candle-transformers/src/models/expert_lre/`,
> `candle-nn/src/kv_cache/chunked/weight_zone.rs`) to a model's **dense layers**,
> so a dense checkpoint larger than the card runs instead of failing to load.
> §2 is the defect; §3 is the shape; §4 the invariant; §9 the numbers that
> bound what this can achieve; §13 what the build settled and what it corrected
> in the sections above it.
>
> Every number is derived for Qwen3.8-27B at Q4_K_M on the RTX 4090 Mobile
> 16 GB dev machine unless it says otherwise. Figures marked *derived* come
> from the geometry in `docs/qwen35_qwen38_models.md` §3 and the checkpoint
> size; figures marked *measured* are from a run and say which.
>
> **The checkpoint is 18.97 GB, not the 16.5 GB the derivations below assume**
> (`ggml-org/Qwen3.8-27B-GGUF` @ `Qwen3.8-27B-Q4_K_M.gguf`, revision
> `0669b986`), and its layer pack is 16.18 GB. The per-layer figures in §9.1
> are correspondingly light; the shape of the argument is unchanged.
>
> **The 16 GB card no longer runs Q4_K_M.** §9.5.7 shows the slot size is a
> super-linear term in the streamed cost, so the checkpoint is now chosen from
> the card's VRAM — Q3_K_M here, Q4_K_M at 24 GB, Q6_K at 32, Q8_0 at 64
> (`QWEN38_27B_LADDER`, all rungs pinned at one bartowski revision). The Q4_K_M
> figures throughout stay as written: they are the derivation's basis and the
> rung the 3090 runs, and §9.5.7 gives the scaling for the others. Note the
> ladder's Q4_K_M is bartowski's 17.77 GB rather than the 18.97 GB ggml-org file
> the sections below were measured on — same recipe name, different conversion,
> 6% apart, which is the size of effect a per-rung figure here carries.

---

## 1. Abstract

A dense model's weights do not fit in VRAM. Qwen3.8-27B is 64 layers at
~240 MB repacked — **~15.4 GiB of layers** before a single token of KV, on a
card whose whole span after load headroom is ~12.5 GiB. Today that model cannot
load at all: `claim_dense` refuses once the block would pass the weight floor,
the weight falls back to the CUDA pool, and the pool draws on VRAM the span
already reserved. The load ends in a CUDA OOM.

The expert cache solved the same problem for routed models. Its answer was not
"make the experts smaller" but "hold a working subset resident, stream the
rest, and make the cold tier authoritative so eviction is a bookkeeping change
rather than a copy." Every part of that answer applies to dense layers
unchanged. What differs is only *what* is streamed and *when* it is needed.

The difference that matters: **an expert cache has a hit rate and a layer cache
does not.** Routing touches a subset of experts per layer, so residency
converts into avoided traffic. Every dense layer is needed on every forward, so
there is no subset to be lucky about. The consequence is stated once, in §9,
and it bounds everything: streamed bytes per forward = `total − resident`, and
no policy moves that floor. What this design does is *reach* the floor
adaptively, and let the KV side take ground from a dense model for the first
time.

---

## 2. What is wrong today

### 2.1 Dense weights are immovable, so the elastic partition is dead for them

`dense_span::open_for_load` claims the reservation, every weight is carved out
of it by `claim_dense`, and `freeze_dense` locks the block:

```rust
// region_pool.rs — claim_dense
if pool.dense_frozen {
    candle::bail!("dense weights: the block was frozen at {} MiB …")
}
```

After the freeze the dense block is a fixed prefix of the span. The elastic
boundary — the whole point of `docs/elastic_vram_partition.md` — moves the
weight/KV line by evicting and relocating **expert slots**. A dense model has
no slots, so the boundary has nothing to trade and the partition is inert. KV
pressure on a dense model can only be answered by refusing work.

### 2.2 A model larger than the span fails, and fails indirectly

The refusal path is graceful in the small: `dense_destination` catches the
error, logs at debug, returns `None`, and the weight repacks into the CUDA
pool. That is correct for a *test* or a second model in the process. It is
wrong for a first model that simply does not fit, because the pool draws on the
same physical VRAM the span reserved. The failure surfaces as a CUDA OOM
several tensors later, in `repack_to_host → dequantize`, naming nothing about
the actual cause.

### 2.3 The load headroom is a permanent concession for a transient

`peak_repack_scratch` concedes the largest 2-D tensor × 4 bytes from the span,
because `repack_ko` dequantizes a whole tensor to F32 before quantizing to its
KO twin. On the 27B that tensor is `output.weight` at `[248320, 5120]` — the
lineage's vocabulary is 248,320, not the 151,936 of Qwen3 — so it is
**4,850 MiB of span given up for the life of the process**, to serve a peak that
exists for one tensor at load. Measured against the reservation:
`usable 14,098 MiB = span 8,736 + scratch 4,850 + cushion 512`, exactly.

At the 27B's 229.4 MB slot that concession is **21 layer slots**, which would
take `H` from 20.7 to ~42 and halve the bytes per forward. Nothing else in this
document is worth half as much. `dense_span.rs` already names the fix — chunk
the repack and the peak becomes one chunk — and it is a loop inside `repack_ko`,
not a change to any policy here.

> The same tensor costs a second time. `output.weight` is **Q6_K** where the rest
> of the checkpoint is Q3_K/Q4_K, so it is ~1.04 GB resident on top — 4.5 slots.
> One tensor holds 25 slots' worth of span against a zone of 23.7.

**And the concession exists only to build that tensor's KO twin.** `ko_tileable`
passes on `[248320, 5120]`, so `QMatMul::build` repacks the LM head like any
other projection. Price the twin: at decode the head is a GEMV reading ~1.0 GB
from *VRAM* at ~800 GB/s — **1.4 ms against an 827 ms forward**, 0.17% of the
step — and at prefill logits are wanted only at the last position (or the few
speculation verifies), so it is the same shape again. The twin is not even
smaller: Q6_K is 6.5625 bits against int8's 8.

Leaving the head on the dequant path drops `peak_repack_scratch` to the largest
tensor that *is* repacked — an FFN at `17408 × 5120 × 4` = **340 MiB** — and
returns **19.6 slots**, taking `H` to ~40 and transfers to ~24. That is a cheaper
route to most of what chunking `repack_ko` would give, and the two are
complementary rather than alternatives.

It is not a one-line change: `build`'s own comment records that a weight
reporting `Off` must be consumed through `forward_live_as` and never a
producer-fused `DynamicActs::Int8`, and the head is fed by `final_norm`, which is
exactly such a producer — `ensure_qmatmul_pairing` refuses that pairing. The
per-tensor decision itself is well-trodden (the sub-tile DeltaNet projections
already take the dequant path); it is the norm→head pairing that needs doing.

Independently: the filter above maxes over **every** 2-D tensor in the file,
including `token_embd.weight`, which is host-mapped and never dequantised on the
device at all. That is wrong on its own terms and should be narrowed to tensors
that actually repack — today it is masked only because the head happens to be the
same shape.

---

## 3. The shape

Dense layers become tenants of the **weight zone**, exactly as experts are:
equal-sized slots, filled from the right, retracted from the left.

```
        span_base                          weight_floor            span_end
            │                                    │                     │
            ├────────────── KV regions ──────────┤─── weight zone ─────┤
                                                 │                     │
                        slot 63 … slot 2   slot 1  slot 0
                        (layer 63)        (layer 1) (layer 0)
                        leftmost                    rightmost
                        evicted FIRST               NEVER evicted
```

Four rules, each of which the zone already enforces or the expert cache already
implements.

### 3.1 Layer *i* lives in slot *i* — **superseded by §14**

> **This section is wrong and §14 replaces it.** The identity mapping is kept
> here because the reasoning below is what the first build was made from, and
> because the specific way it fails is the finding: it optimises the eviction
> *order* and is blind to the eviction *pattern*, which is what actually decides
> whether a transfer can be hidden. Read §14.1.

The zone's geometry is already the one this design wants:

> *Addresses descend as the index rises: slot 0 is the rightmost slot in the
> span and the highest occupied index is the frontier.* — `weight_zone.rs`

So the identity mapping puts layer 0 at the far right and layer 63 at the far
left. Retraction eats the **highest indices**, which are the **highest layer
numbers**, which are the layers the wave reaches **last**.

**Eviction order and maximum prefetch lead time are therefore the same
ordering, for free.** Nothing has to score a layer, predict a reuse distance,
or maintain a transition matrix. The layer thrown out is always the one there
is the most time to fetch back.

What that argument misses: it is about *one* layer in isolation. Under it the
missing layers are the contiguous tail `[H, N)`, so the run of resident layers
between two consecutive missing ones is **zero**, and a fetch has nothing to
hide behind however early it was ordered. §14.1 measures the consequence.

### 3.2 The pinned head is two slots, not a separate block

Layers `0..PINNED_LAYERS` are never evicted, for the reason the expert cache
already states: they run first every pass with no compute ahead of them to
overlap a DMA against, so evicting one guarantees a cold miss at maximum stall.

`PINNED_LAYERS` is already `2`, already a constant rather than a function of
capacity, and `pack` already writes no record for the pinned prefix while the
warm draw already skips it — so the pinned head has no host or disk copy at
all. That is the point rather than a side effect: nothing can ever need to
reload it.

They need **no separate dense block**. Slots 0 and 1 are the rightmost
addresses in the zone, which retraction structurally cannot reach, so pinning
is entirely expressed by the eviction scan skipping `layer < PINNED_LAYERS` —
which `demand_eviction` already does. One mechanism, not two.

### 3.3 The whole model is mapped to warm and cold tiers

Mirror of §5 and §6 of `docs/expert_cache_design.md`, with a layer record where
an expert record stands:

| Tier | Holds | Mutability |
|------|-------|------------|
| cold | a repacked record for **every** layer ≥ `PINNED_LAYERS` | authoritative |
| warm | pinned host RAM, a stratified subset | static, immutable |
| hot | the weight zone's slots | dynamic; eviction is a pure drop |

Records are the **repacked KO image**, sector-aligned to the pack's stride, so
a cold read lands directly in a pinned slot with no bounce buffer and a warm
promotion is one H2D of contiguous bytes. No repack, no dequantise, and no
per-miss requantisation on any path.

### 3.4 The wave defines what is evictable

At wave position *L*:

* slots holding layers `< L` — **free**. The forward will not read them again.
* slots holding layers `≥ L` — **protected**. Evicting one guarantees a stall.
* layers `L+1`, `L+2` — **committed prefetch**, always in flight, joined at
  need through a per-layer fence.
* layers `> L+2` — **opportunistic**, loaded into slots freed behind the wave,
  nearest-first, and never at the cost of evicting a layer due sooner.

The last rule is `PREFETCH_EVICT_WINDOW` generalised, and it is *simpler* for
layers than for experts: the reuse distance of a layer is exact arithmetic on a
cycle, not a prediction. There is nothing to be wrong about.

---

## 4. The invariant

> **The cold tier holds a valid copy of every layer outside the pinned head,
> always.**

Everything else follows, exactly as it does for experts. Eviction is
`vram = None` with no copy and no destination to find. The warm tier needs no
eviction policy. "Where do I load this from" is a total function. A layer is
read-only after load — nothing in the engine writes a resident layer, there is
no dirty bit and no writeback path — so residency is free to be a pure cache.

---

## 5. What a slot image holds

Only the **KO-tileable projections** — the weights that are large, that repack
to an int8 twin, and that therefore dominate the bytes:

| Kind | Streamed |
|---|---|
| DeltaNet | `wqkv`, `wz`, `w_out` |
| Attention | `wq`, `wk`, `wv`, `wo` |
| Both | `ffn_gate_up`, `ffn_down` |

**The FFN is two projections, not three.** `QuantizedMlp::from_weights`
row-concatenates gate and up into a single `[2·intermediate, hidden]` weight
whenever the device is CUDA and the dtype is quantized — which is every
checkpoint this engine runs. An image built for `{gate, up, down}` would
describe a layer that does not exist. The unfused form is still reachable (a
float weight, a CPU load, a shape mismatch), so the form is carried per layer as
`FfnForm` rather than assumed: the same checkpoint fuses on CUDA and does not on
CPU, and the pack must describe the layer that was actually loaded.

Everything else in a layer **stays resident for the life of the process**:

| Stays resident | Why |
|---|---|
| `attn_norm`, `post_attn_norm` | `RmsNorm`, a fused producer rather than a raw buffer |
| DeltaNet `dt_bias`, `a`, `conv`, `norm` | F32 by design — the recurrence accumulates and must not drift |
| `w_beta`, `w_alpha` | `[n_v_heads, hidden]` = `[48, 5120]`: 48 rows packs into the storage chunk (multiple of 8) and is still rejected by the matmul tile (multiple of 32), so there is no KO twin to place |
| `q_norm`, `k_norm` | per-head `RmsNorm`, folded into the projection |

That residue is **~0.1% of a layer** — roughly 250 KB against 240 MB, or ~16 MB
across all 64. Streaming it would buy nothing measurable and would drag the
`RmsNorm` and F32-constant rebuild into a path whose entire value is that a slot
view is a pointer and a length. So the image is uniformly "things
`view_repacked` can wrap", and nothing else.

This also settles the streamed fraction quoted in §9.1: 115.4M of the DeltaNet
mixer's 115.8M parameters, and all 267.4M of the FFN, so **99.6%** of a layer's
bytes are streamable and the residue does not perturb the arithmetic.

### 5.1 Slots are uniform; layer images are not — **superseded by §14.2**

> **The arithmetic below is right and the conclusion is wrong.** The waste was
> estimated at ~2% from a *parameter count*; measured from the checkpoint's
> actual quants it is **18%**, because llama.cpp's `use_more_bits` gives a
> handful of layers wider tensors than the rest. §14.2 has the numbers and the
> two-tier layout that replaces this.

This is the one place the expert analogy strains, and it must be settled before
any code moves.

Expert slots are uniform because every expert in a model shares a geometry. A
**layer image is not uniform**: a DeltaNet layer and an attention layer carry
different tensor sets entirely, and the zone's design rests on equal size —

> *relocating a slot is a memcpy between two addresses of identical length
> rather than a compaction.*

**Resolution: size the slot to the maximum layer image over the model.** On the
27B that is the DeltaNet layer at ~383M parameters against the attention
layer's ~372M — **~2% waste**, and the zone is untouched. The alternative, a
size-classed weight zone, would buy 2% and cost the property every relocation
and every retraction depends on. It is not worth it.

The 3:1 DeltaNet:attention interleave does not disturb the mapping: layer *i*
still occupies slot *i* whatever kind it is.

### 5.2 The layer descriptor — **built**

`candle-transformers/src/models/layer_stream/descriptor.rs`. The layer analogue
of `LayerGeometry` + `slot_offsets`: a per-kind table giving each projection's
offset and byte length inside the slot image. It is the single source of truth
shared by the pack writer, the warm fill and the slot view, for the reason
`repack_ko_into` already states about its own sizing — two copies of that
arithmetic in different files is how one gets corrected and the other does not,
and the failure is a silent write past the end of one weight into the next.

Three properties the build settled:

* **Placement order is the kind's, not the caller's.** Two enumerations of the
  same layer place identically, which is what lets the pack write a record for
  one layer and a slot view read it back for another.
* **`repacked_bytes` is an input, not a computation.** It comes from
  `ko_repacked_bytes`, which needs the CUDA dtype tables; taking it as an
  argument is what lets every placement rule be exercised with no GPU. It must
  be the same number the repack writes — the two agreeing is the whole safety
  argument, exactly as for an expert slot.
* **A sub-tile projection is a named error, not a silent reservation.**
  `ImageError::NotStreamable` reports the role and shape rather than reserving
  bytes for a weight that will never arrive.

Eleven tests, none needing a device, asserting exact byte offsets rather than
tolerances.

---

## 6. Two eviction mechanisms, and they are not interchangeable

Conflating these is the most likely way to build this wrong.

| | Intra-zone recycling | Boundary move |
|---|---|---|
| Serves | "layers behind the wave are evictable" (§3.4) | "evict layers to make room for KV" |
| Call | `demand_eviction(layer, deficit, protect)` | `set_weight_floor` / `renegotiate_boundary` |
| When | **per layer, inside the wave** | **between forwards only** |
| Cost | bookkeeping — eviction is a pure drop | **device-wide synchronize** |

`set_weight_floor` hard-refuses while a wave generation is open, and
`quiesce_before_handover` gives the reason: *"between forwards bounds only what
is being issued — the last pass's GEMMs may still be executing."* Publishing a
lower floor while they run lets the KV side write bytes those GEMMs are
reading, surfacing as `CUDA_ERROR_ILLEGAL_ADDRESS` in whatever unrelated kernel
happens to be resident when the fault lands.

**Consequence for this design:** KV pressure cannot be answered mid-forward.
The zone shrinks at the pass boundary and the forward runs inside whatever
capacity it started with. This is a constraint to sit inside, not a defect to
fix — boundary moves are rare, and the drain is paid only when the boundary
actually moves.

---

## 7. No fits/does-not-fit branch

The obvious design is to detect at load whether the model fits and take a
streaming path only when it does not. **Do not.** That is two code paths for
one job, it violates the repo's standing rule against dual paths, and the
streaming half would be exercised only by the largest model anyone happens to
run — which is to say it would rot unnoticed.

**Dense layers go into slots always.** When the model fits, every layer is
resident, nothing is ever evicted, and no byte ever moves after load: "it fits"
is the degenerate case of one mechanism, not a separate mode. Every dense model
then exercises the same path, and the streaming behaviour is covered by the
gates that already run.

This subsumes the load-time detection entirely. Capacity is not a question the
loader has to answer, because residency is an outcome rather than a plan —
exactly as `dense_span` already argues about the dense block's size: *"Nothing
predicts the model's size."*

---

## 8. What this deletes

* `dense_span::open_for_load` / `close_load` / `freeze_dense` for models whose
  layers become slot tenants — the dense block shrinks to what is genuinely
  not a layer (embeddings are already host-mapped; norms and the head remain).
* The `claim_dense`-refused → CUDA-pool fallback as a *load* path. It stays for
  what it was written for — a second model in the process, a test — and stops
  being the road a too-big model takes to an OOM three tensors later.

---

## 9. What it costs, and the floor it cannot beat

### 9.1 Per-layer bytes — *derived*

From `docs/qwen35_qwen38_models.md` §3: 64 layers = 16 × (3 DN + 1 attention),
hidden 5120, dense FFN 17408, attention 24 Q / 4 KV @ head_dim 256, DeltaNet
48 V / 16 QK @ 128.

| Component | Params |
|---|---|
| Dense FFN (every layer) — 3 × 5120 × 17408 | **267.4M** |
| DeltaNet mixer — wqkv + wz + w_out + gates | ~115.8M |
| Attention mixer — wq + wk + wv + wo | ~104.9M |
| **DeltaNet layer** | **~383M** |
| **Attention layer** | **~372M** |

At Q4_K_M's ~5.1 bits/param that is **~240 MB per layer**, **~15.4 GiB** for
all 64. The embedding costs no VRAM — `host_embedding.rs` already serves it
from `cuMemHostAlloc(DEVICEMAP)`.

### 9.2 The floor — *measured, and it is not what this section first said*

**Streamed bytes per forward = `total − held`,** where *held* is the layers a
forward is allowed to *finish* still holding. That is not the same as capacity,
and the difference is the whole of this section.

The original claim was `total − resident`, on the reasoning that residency
self-organises into pinned head + streamed middle + resident tail, "the tail
survives because it is always needed last and so never comes under pressure
before use". **The tail does not survive.** Measured on the 27B at capacity 21 of
64 layers: `hits 22` over 11 forwards — exactly the two pinned layers, each
forward, and nothing else. Every other layer crossed PCIe on every forward, and
capacity did not enter into it.

That is Bélády behaving correctly on a *cyclic* reference string, not a defect.
At wave *L* the furthest-future resident is `L−1`, the layer just executed, so
each step evicts the one behind and loads the one ahead: residency is a window
that slides one step per layer and therefore `N` steps per forward, which is
exactly enough to have discarded everything it held. The tail is evicted while
the wave is still walking toward it — by the prefetch for the *next* forward's
head, which is nearer.

So capacity has to be spent deliberately. `LayerResidency::keep` exempts a
leading prefix from eviction, and the window slides over the rest:

| | Streamed per forward |
|---|---|
| Bélády alone | `N − pinned` = 62 |
| Held prefix, capacity *C* | `N − (C − COMMITTED_DEPTH − 1)` = 45 at *C* = 21 |

Prefetch depth still does not change the number — it decides whether the floor
is *reached* rather than stalled above, and `L+2` is sufficient. What changed is
that residency now does, and the two are in tension: every slot given to the
prefix is a slot the window cannot use for lookahead. `COMMITTED_DEPTH + 1` is
the window's minimum (the layer under the wave plus the loads in flight), which
is why the prefix stops exactly there — and why the same expression is the floor
a retraction may not take the zone below (`LayerResidency::min_capacity`).

### 9.3 The numbers — *derived*

At ~20 GB/s sustained pinned H2D on PCIe 4.0 ×16:

| | Streamed | Per forward | Decode, 1 session |
|---|---|---|---|
| Today (span ~12.5 GiB) | ~23–30 layers | 275–360 ms | ~3 t/s |
| **After chunked repack** (§2.3) | ~15 layers | **~180 ms** | ~5.5 t/s |
| **+ MTP speculative decode** | — | — | **~16 t/s** |

**Chunking the repack is worth more than any streaming policy.** It returns
~3.1 GiB of span — about 12 layers of residency — and it is a contained change
to `repack_ko_into` that the codebase has already identified.

Prefill is the opposite story. Compute per layer at *W* tokens is
`2 × 383M × W`; transfer per layer is 240 MB / 20 GB/s ≈ 12 ms. Break-even is
**~1,250 tokens per wave**, and prefill waves run 8,192. At that width compute
is ~78 ms against 12 ms of transfer — **fully hidden**, and the sweep lands
around **~1,600 t/s**, clearing the ≥1000 t/s target.

> The softest figure here is int8 throughput (~80 TFLOP/s assumed). The
> break-even width scales directly with it, so §10 measures it first.

### 9.4 What this means for the targets

Prefill clears. **Decode does not and cannot** reach ≥50 t/s for a dense 27B on
a 16 GB card at Q4_K_M — the floor is bandwidth, and 15 GiB of layers over a
25 GB/s link is what it is. That is a property of the model/card pair, not a
defect in this design. The levers, in order: chunk the repack (§2.3),
speculative decode on the dense NextN head, quantise the *streamed tail* below
the resident set, widen the batch for aggregate throughput.

---

## 9.5 The wave schedule — *derived, and deterministic*

Everything above prices the streaming. This section decides the schedule, and
the point of it is that **there is nothing to adapt to**: a dense sweep's access
sequence is fixed, its bytes are fixed, and the two bandwidths are properties of
the machine. The schedule is therefore *computed* from five numbers known before
the first forward — `N` layers, slot size `S`, zone capacity `C`, warm slots `W`,
and the ratio `t_x / t_c` — not learned from counters. The counters exist to
verify the schedule is being met, never to choose it.

### 9.5.1 The cost, and why there is only one objective

```
forward_time  =  H·t_c  +  (N−H)·max(t_x, t_c)
```

for `H` layers **held** across the forward boundary. Held layers cost compute
alone; streamed ones cost whichever of transfer and compute is slower.

Measured on the 4090 Mobile: `t_x ≈ 21 ms` (a 248 MiB slot at ~12 GB/s pinned
H2D) against `t_c ≈ 12 ms` at 659 tokens and ~0.1 ms at width 1. So `t_x > t_c`
at every width below ~1,150 tokens, which is the same break-even §9.3 derives
from the other direction.

Two consequences, and they are the whole design:

* **Below break-even, `(N−H)` is the only term that moves.** Prefetch depth,
  eviction order and plan cleverness change nothing about it.
* **Above break-even the transfers hide, so `H` stops mattering — but never
  hurts.** There is therefore no prefill/decode mode switch: *maximise `H`,
  always*. One path, which is what §7 asks of every choice here.

### 9.5.2 Window width: `H = C − (COMMITTED_DEPTH + 1)`, and the 2 is derivable

The zone splits into a held prefix and a sliding window. The window needs one
slot for the layer under the wave and `k` for the transfers in flight, so
`H = C − (k + 1)` and every slot spent on the window is a transfer paid on
*every* forward. `k` must therefore be the smallest value that does not starve
the copy stream.

* `k = 1` nearly suffices: PCIe is serial, so one queued transfer keeps it busy.
  It goes idle only across the host-side gap between joining `L+1` and issuing
  `L+2`.
* `k = 2` covers that gap with one spare in the queue.
* `k ≥ 3` buys nothing — the copies serialise regardless, so a third in flight
  is a slot held out of the prefix for no throughput.

`COMMITTED_DEPTH = 2`. It was asserted before and is now derived.

### 9.5.3 The held set is a prefix, and that is not arbitrary — **wrong; see §14.1**

> **A prefix is the worst choice, not the optimal one.** The argument below
> compares a prefix against *one* alternative — a suffix — and picks the winner
> of a field of two. Both are terrible for the same reason, and the reason is
> invisible from inside the comparison.

Every layer costs the same `S` and is read once per sweep, so *which* `H` layers
are held looks like a free choice. The forward boundary breaks the symmetry: a
sweep ends at `N−1` and the next begins at `0`.

* Holding `[0, H)` means the next forward opens on `H` layers that need no
  transfer, giving the copy stream `H·t_c` of **running start** before the first
  streamed layer is due.
* Holding a suffix `[N−H, N)` gives zero — layer 0 must stream immediately, so
  the pipeline stalls at the top of every forward.

A prefix is optimal, not merely convenient.

**What this misses.** The running start is one gap, at the sweep boundary. It is
the `N−H−1` gaps *between* the streamed layers that carry the other transfers,
and a contiguous missing set makes every one of them **1**. So the prefix buys
one hidden transfer and exposes all the rest; the correct question is not where
the run of held layers sits but how the *missing* layers are distributed, and the
answer is "as evenly as possible" (§14.1).

### 9.5.4 The tiers are a partition of `[0, N)`, computed from `H` and `W`

```
held  = [0, H)                     cost 0
warm  = [max(H, N−W), N)           cost t_x
cold  = the remainder              cost t_x + t_read
```

The subtlety is that **`H` is not a constant**. The zone opens small —
`INITIAL_KV_RESERVE` hands most of the span to the KV side at load — and grows
into spare KV ground over the first forwards. On the 27B `H` opens at 5 and
settles at 16.

So the warm run is drawn from the **top** of the model, not upward from `H`. The
streamed set only ever shrinks from the bottom as `H` rises, so `[N−W, N)` is
streamed under *every* `H` the tier is large enough to reach, and one membership
is correct before, during and after growth. That matters because the tier is
gigabytes of pinned host memory filled by a sequential pass over the pack — not
something to redraw in the gap between two forwards.

> **Measured, and this is what surfaced it.** Drawn upward from the load-time
> `H = 5` the tier covered layers 5–55; once `H` settled at 16 the streamed set
> was 16–63, so **layers 55–63 fell to the cold tier on every forward for the
> life of the process** — 9 synchronous NVMe reads a forward, 19% of all
> transfers — while the tier held 50 slots for 48 streamed layers. It was large
> enough and aimed at an `H` that no longer existed.

The rule this replaces reasoned that the layers the wave reaches earliest have
the least compute ahead of them to hide a fetch behind. True, but it does not
pin a *set*: the first streamed layer is `H`, and `H` moves. Being streamed under
every `H` does.

### 9.5.5 The schedule, and what it predicts

| | `H` | Transfers / forward | Cold / forward | Decode 1 / 4 ctx |
|---|---:|---:|---:|---:|
| Before this section | 16 | 48 | 9 | 2.6 / 10.1 |
| **Warm drawn from the top** | 16 | 48 | **0** | **2.9 / 11.5** |
| **The Q3_K_M rung (§9.5.7)** | **20.7** | **43.3** | **0.09** | **4.8 / 19.2** |
| Growth counted in whole slots (§9.5.6) | 19 | 45 | 0 | — |
| `peak_repack_scratch` reclaimed (§2.3) | **~42** | **22** | 0 | — |

The first three rows are measured; the last two are the model's prediction. The
decode column is the budget-4 speculative row of the sweep, held constant across
all three measured rows.

The rung row is also the cleanest confirmation the cost model has had. Over one
256-token config the counters read `hits 5332`, `joins 11116`, `cold 24`; those
sum to `16448 = 257 × 64`, so `H = 5332/257 = 20.7` and transfers
`= 11116/257 = 43.3 = 64 − 20.7` to three figures. A smaller slot bought 4.7 more
held layers *and* 4.7 fewer transfers from the same zone, which is exactly the
two-terms-at-once claim of §9.5.7 and not something a schedule change can do.

The cost model is exact rather than approximate, which is what licenses the
table. Measured steady state on the 27B before the change: `240 transfers` and
`45 cold reads` per config over 4.92 trunk forwards — `48.0` and `9.0` per
forward against the model's `N − H = 64 − 16 = 48` and "the 9 layers outside the
warm window". After it: `240 transfers`, `0 cold`. No unexplained slack in
either.

### 9.5.5a A refusal must not cost half the tier

The warm size the schedule assumes is not the size the policy asks for.
`handle::warm_sizing_from` computes a budget from three host ceilings, and then
`cuMemAllocHost` gets a vote — a fourth limit the policy cannot see. That vote
used to **halve** the request, which on the 27B took a 50-slot budget to 25 and
put 23 layers a forward back on the cold tier, discarding 6.5 GiB of pinning the
ceilings had already found affordable.

Stepping down by an eighth converges on the allocator's real ceiling from above
and can never exceed the budget, so the over-pinning guard still holds. It costs
attempts rather than tier: a refused `cuMemAllocHost` measured ~0.7 s, and the
step count is bounded by `log(want)/log(8/7)` — worth stating because a machine
whose true ceiling is far below its budget pays that at load.

### 9.5.6 What is deliberately *not* scheduled

* **Growth in whole slots.** `kv_grow_step` halves the spare and denominates in
  16 MiB regions while a slot is 248 MiB, so near convergence every step rounds
  to a fraction of a slot and `MIN_GROWTH_SLOTS` rejects it. The zone settles at
  19 slots against a structural ceiling of ~22 (`MIN_ELASTIC_RESERVE` floors the
  KV side at 2 GiB). Worth ~3 slots, against a real risk of starving a KV side
  that has already been measured failing at 20 contexts — so it is stated here
  and not taken.
* **Cold reads off the forward thread.** A cold read is a blocking `pread` on
  handle 0 of a `DirectFile` pool that owns 16 and exposes `read_at_with_handle`
  for exactly this. With the tier drawn from the top the steady state has no cold
  reads at all, so this only matters on a host too small to warm the streamed
  set — which is where it should be taken up.

### 9.5.7 The slot size is the third term, and the checkpoint sets it

§9.5.1 has three quantities in it, not two. `H` and `N` are counts; `t_x` is
`slot_bytes / bandwidth`, and the schedule can do nothing about either factor.
The checkpoint can:

```
forward_time  =  H·t_c  +  (N−H)·max(t_x, t_c)          t_x = S / B
```

Below break-even `t_x` dominates, so the streamed cost is `(N−H)·S/B` and **`S`
is a linear multiplier on the whole of it**. Worse — better — `S` also sets `H`,
because the zone is a fixed byte budget carved into slots: `C = zone/S`, so

```
streamed_cost(S)  =  (N − zone/S + k + 1) · S / B
```

which falls faster than linearly in `S`. Shrinking the checkpoint therefore moves
*both* terms the same direction at once, and it is the only lever in this
document that does. Every scheduling choice in §9.5 redistributes a fixed number
of bytes; the rung changes how many bytes there are.

Hence the quant ladder in `models/quantized_qwen38.rs`
(`QWEN38_27B_LADDER`): the checkpoint is a function of the card's total VRAM —
Q3_K_M at 16 GB, Q4_K_M at 24, Q6_K at 32, Q8_0 at 64. Two things about its
shape are worth stating, because both are easy to get backwards:

* **The rungs are not a fit test.** A 16 GB card runs Q6_K perfectly well under
  §7 — it simply spends a third more bandwidth per forward doing it. The ladder
  picks the largest quant whose *residency* the card can make good use of, which
  is a strictly lower bar than fitting and a strictly higher one than loading.
* **Coarser is not a concession on a streaming card; it is the point.** On a card
  that holds the model, a smaller quant buys nothing and costs quality, so the
  ladder climbs. On a card that streams, the relation inverts: Q3_K_M is 14.61 GB
  against Q4_K_M's 17.77, which is ~18% off every transfer *and* a slot small
  enough that the same zone holds more of them. Both terms of `(N−H)·S` improve.

**Measured** on the 4090 Mobile, the 27B's plain-decode row (no drafter, 256
tokens, StoryRewrite) moving from ggml-org's Q4_K_M to bartowski's Q3_K_M:

| | width | Q4_K_M | Q3_K_M | |
|---|---|---|---|---|
| plain decode | 1 context | 0.9 t/s | **1.2 t/s** | +33% |
| plain decode | 4 contexts | 3.7 t/s | **4.6 t/s** | +24% |
| draft budget 4 | 1 context | 2.9 t/s | **4.8 t/s** | +66% |
| draft budget 4 | 4 contexts | 11.5 t/s | **19.2 t/s** | +67% |

The speculative rows gain more than the plain ones, and that is the schedule
compounding rather than a second effect: a drafter turns one trunk forward into
several accepted tokens, so every layer this rung stops transferring is saved
against a *larger* numerator. The whole budget sweep at the rung reads
1.2 → 2.1 → 3.1 → 4.0 → 4.8 t/s at width 1, still climbing at
[`MTP_MAX_DRAFT`], which is where the head itself clamps.

Both rows keep `cold ≈ 0` (24 reads over a 256-token run, all of them the first
sweep's cold start), so the tier partition of §9.5.4 still holds at the smaller
slot and the gain is the transfer term alone. The width-1 gain tracks the byte
ratio closely; width 4 gains less because at four contexts `t_c` has grown toward
`t_x` and part of the transfer is already hidden — which is §9.5.1's break-even
appearing from the other side.

The drafter does not move with the rung. `QWEN38_27B_MTP` pins one Q4_0 sidecar
for every card, because speculation is lossless: the trunk verifies every
proposal, so the head's quant — and its publisher — can only move the acceptance
rate, never the tokens. That is also why it is the one file permitted to come
from a different repository than the trunk (only ggml-org converts the 27B's
`mtp-` sidecar; the ladder's repository publishes trunk quants and nothing else).

---

## 10. What must be measured before building

1. **int8 GEMM throughput at layer shapes** on this card — sets the prefill
   break-even width in §9.3, and is the softest number in the document.
2. **Sustained pinned H2D with the copy stream contended** by KV migration —
   §9.3 assumes 20 GB/s in isolation.
3. **Repack peak after chunking** — confirms §2.3 returns the span it claims.
4. **Max layer image** over the 27B, against §5's 2% waste estimate.

---

## 11. Risks

1. **Slot size regresses smaller models.** A uniform slot sized to the max layer
   image is wasteful if a model's layer kinds diverge more than the 27B's do.
   Measure per model; a size-classed zone is the escape hatch and §5 argues
   against taking it early.
2. **The boundary latch (§6) starves KV under a long forward.** A wide prefill
   holding the zone for hundreds of milliseconds delays the KV side's ground.
   The existing admission budget already prices KV growth; this adds a term it
   has not had to model before.
3. **Cross-forward prefetch is not throughput.** Pre-staging the next forward's
   middle during the resident tail helps a cold or isolated forward. Back to
   back, the wire is saturated either way. It must not be counted twice.
4. **Layer descriptor drift (§5.2).** The pack writer, the warm fill and the
   slot view must share one offset table. Two copies is the failure
   `repack_ko_into` already documents.

---

## 12. Build order

1. **Layer descriptor + slot image** (§5.2) — geometry, offsets, max-image
   sizing. Testable with no GPU. **Built.**
2. **Layer pack** (§3.3) — mirror `expert_lre/pack/`: repacked records,
   sector-aligned stride, identity checksum against the GGUF. **Built.**
3. **Warm tier reuse** — `WarmPool` is already generic over a record stride.
   **Built** (`layer_stream/warm.rs`, and `expert_lre::pinned` is now
   `pub(crate)` so the pool itself is shared rather than copied).
4. **Residency + the wave rules** (§3.4) — free-behind, protect-ahead,
   committed `L+2`, opportunistic beyond. **Built** — and simpler than this
   document assumed; see §13.1.
5. **Wire into the layer loop** — `qwen35/forward.rs`, the
   `for li in layer_start..layer_end` sweep: join the fence before the layer,
   release the slot after. **Not built.**
6. **Measure** against §10 and report prefill and decode separately.

---

## 13. What the build settled

### 13.1 The policy is Bélády's rule, not an approximation of it

The design stated the wave rule as two clauses — behind is free, ahead is
protected — and treated the eviction choice as a separate question. It is one
quantity:

```text
distance(layer) = (layer − wave) mod N
```

A forward walks `0..N` and the next walks it again, so the distance to a layer's
next use is a **subtraction, not a prediction**. Bélády's optimal replacement is
normally a bound no online policy can reach because it needs the future; here
the future is arithmetic, so `argmax distance` is the implementation. There is
no access frequency, no decay, no eviction key and no transition matrix in
`residency.rs`.

The two clauses fall out of the modulus rather than needing to be written. At
`wave = 10` of 64, the just-executed layer 9 sits at distance 63 — automatically
the preferred victim — while layer 11 sits at 1 and is the most protected.

### 13.2 The pinned head needs its views built explicitly

Caught by the end-to-end test on the first run: the pinned layers are the one
set that never passes through the load path, so nothing built their slot views
and the first `ensure(0)` found a resident layer with no matmuls over it. The
residency places them; the cache has to finish the job
(`LayerCache::build_pinned_views`). The corollary is a contract worth stating —
**the loader must upload the pinned images into `slot_base[0..pinned]` before
the first `ensure`**, because those layers come straight from the checkpoint and
never round-trip through a tier.

### 13.3 A slot image holds KO twins, and that is now enforced

`QMatMul::from_qtensor_repacked` refuses a non-KO dtype, but only once a view is
built over a slot — which needs a device. `layer_image` now rejects a source
dtype up front (`ImageError::NotRepacked`), turning a CUDA-only failure into a
named load-time one.

### 13.4 Direct I/O needs a sector-aligned *destination pointer*

Not merely a sector-sized length. In production this is free — a cold read lands
in a pinned warm slot, and the pool's slots are cut to the pack's stride, which
is a whole number of sectors — but it is a real contract and it is now on
`LayerPack::read_into` rather than implicit in how the caller happened to
allocate.

### 13.5 Measured: both regimes, one code path

64 synthetic layers at the lineage's 3:1 interleave, two full forwards each,
byte identity asserted per projection
(`layer_stream::cache::tests::a_wave_streams_every_layer_through_a_small_ring`):

| ring | hits | warm | cold | joins | evictions |
|---|---|---|---|---|---|
| 10 / 64 | 4 | 47 | 84 | 124 | 123 |
| 64 / 64 | 66 | 20 | 42 | 62 | **0** |

The roomy row is §7's claim holding in practice: `warm + cold = 62 = 64 − 2`
pinned, so **every layer loads exactly once, ever**, and the second forward is
pure hits moving no bytes. "It fits" really is the degenerate case of one
mechanism rather than a second path.

The tight row's 124 joins are not a defect and not a measurement of time lost: a
ring of 10 over 64 layers must stream 54 layers per forward, and §9.2 says that
number is `total − resident` and no policy reduces it. `fence_joins` is named
for what it counts.

### 13.6 The FFN is fused, and §5 said otherwise

See §5. The design's projection table was wrong for every checkpoint this engine
actually runs. Corrected in the descriptor and here together.

### 13.7 Cloning a `QMatMul` copies the weight

`QCudaStorage`'s `Clone` is documented as always-owned because *"`CudaSlice::clone`
is a device-to-device copy"*. So the forward must take streamed projections **by
reference**: rebuilding a `QuantDeltaNetWeights` per layer per wave, which was
the obvious way to splice slot views into the existing structs, would copy
~115M parameters per layer per forward.

The wiring that avoids it entirely — and avoids changing a single downstream
signature — is for the cache to hand back a whole `QuantLayer` per slot, built
once per tenancy from the slot's views plus the layer's resident parts. `RmsNorm`
holds `Arc<QTensor>` and `Tensor` is a handle, so those clones are refcount
bumps; only the `QMatMul`s are slot views, and those are built, not cloned.

### 13.8 The pack build is a streaming pass over the existing loader

Writing the pack needs every layer's repacked KO bytes on the host, and the only
thing that produces them is the load itself. `load_quantized_model` builds *all*
layers into the span before anything else runs, which for a model larger than
the card is the failure this design exists to remove — so the pack build cannot
simply reuse it as written.

It does not need a second loader either. Two changes make the existing one serve
both:

* **`load_layer`** — the trunk loop's body, extracted. The pack build calls it
  once per layer, writes the record, and drops the layer, so its peak is one
  layer rather than the model. Extracting rather than transcribing is what stops
  the pack from describing a layer the trunk loop does not actually build.
* **`WeightResidency`** — a parameter on the repack saying whether the bytes go
  to the dense block or the CUDA pool, threaded from the `Loader` that knows.
  The block is a bump allocator with no free, so a weight claimed from it holds
  its ground for the process; the pool frees on drop. `repack_ko_into` already
  took `dst: Option<_>` and `dense_destination` already returned `None` for the
  cases that belong in the pool, so this is a destination the repack path had
  all along rather than a new one. It is a *parameter* rather than a second
  constructor because everything else about the repack — the twin, the tiling
  rule, the numerics — is identical either way.

The peak is then one layer plus the repack scratch — which
`dense_span::peak_repack_scratch` already reserves headroom for — and the pass
runs only when no valid pack exists, so it is a one-time cost per checkpoint.

> An earlier revision of this section called the pack build a *structural*
> blocker on the grounds that the span was the only destination a repack could
> take. That was wrong: the pool fallback is on the same path and is exactly the
> escape hatch this needs.

### 13.9 A slot holds whichever quantized form the load produced

§5 said a slot image holds "the KO-tileable projections", and both the image
builder and the slot view enforced it: `layer_image` refused a source dtype, and
`QMatMul::from_qtensor_repacked` refused anything but a KO twin. That is right
for an *expert* slot and wrong for a layer slot, because a layer's projections
take three forms and only one of them is a twin:

| Case | Slot holds | Kernel |
|---|---|---|
| an int8 mode, tileable shape | the KO twin | int8 q8a128 |
| an int8 mode, shape the matmul cannot tile | the source quant | GGML |
| `Int8Mode::Off` | the source quant | GGML |

The third case is not a corner: `Int8Mode::Off` is the numeric reference, and the
lineage's Off-mode gates (`story_rewrite_greedy_9b`, `speculative_decode_9b`)
are how the int8 path is validated. Streaming that refused it would have taken
the reference mode away from every dense checkpoint.

`build::slot_form` now makes the same choice `QMatMul::build` makes, from the
header — mode, then `ko_tileable` — and the placement's dtype records it.

**The GGML kernels need a padding tail, and a slot must own it.** They address
`pad(ncols, MATRIX_ROW_PADDING)` columns unconditionally, which is why an owned
`QCudaStorage` allocates `padded_storage_bytes` rather than its payload. A
leased view is exactly its payload, so `QCudaStorage::fwd` refused every lease
outright — `Backing::Lease` standing in for "has no padding", which is true of
an expert slot and need not be true of a layer slot. So:

* a `Placement` carries **two** numbers: `bytes`, the weight the pack holds and
  the H2D copies, and `extent`, what the slot reserves and the next projection
  is placed past;
* `view_repacked` takes the extent as well as the payload, mapping the former
  while reporting the latter as the weight;
* `fwd` checks the reserved extent instead of the backing, which admits a slot
  that left room without admitting one that did not.

The tail costs ~280 B per projection against a ~240 MB layer.

### 13.10 One span-resident model per process, and the reload is tight

The dense block is a **bump cursor with no free path**, and that is deliberate:
`claim_dense`'s own header says the block is handed out exactly once, which is
what makes the `MATRIX_ROW_PADDING` tail past every weight a *defined zero* for
the q-matmul over-read. `freeze_dense` then closes the load for the life of the
process. So a **second** `load_hybrid_gguf` in the same process is refused by
`claim_dense` and every one of its dense weights falls back to the CUDA pool —
"as every model did before the span could hold them", in `open_for_load`'s words.

That fallback is tight on this model, and the numbers say exactly how tight.
From `KV_ARENA_STATS=1` on the 4090 Mobile: usable **14,098 MiB**, span
**8,736 MiB**, pool cushion **512 MiB**, so the pool holds ~**4,850 MiB**. The
span was sized as `span_target − peak_repack_scratch`, which concedes to the pool
*precisely* the largest tensor's F32 repack transient — **4,850 MiB** for
`output.weight` at 248320×5120 — and not a byte more:

```
usable 14,098 MiB  =  span 8,736  +  scratch 4,850  +  cushion 512
```

A reload must fit that transient **and** the whole dense residue, which the
region count gives independently as 542 → 341 regions = **3,216 MiB**
(`output.weight` 1.27 GB + the MTP sidecar 1.70 + two pinned layers 0.46). Those
do not both fit in a 5,362 MiB pool under any ordering, so the failure ought to
be deterministic — and it is not: the sweep OOMed once and completed twice. The
mechanism above is therefore established but **incomplete**; something is
returning ground the naive accounting does not see, and it must be found before
anyone builds on this section.

Measured behaviour matches a margin that thin: the five-load speculative sweep
OOMed on its second load once and then completed twice. It is not a regression in
whatever changed most recently, and chasing it there is wasted effort.

**The fix, when it is worth taking.** A span tenancy refcount: `open_for_load`
takes one, a guard held as `QuantModel`'s **last** field releases it (fields drop
in declaration order, and the guard must outlive the caches that hold slots), and
at zero the pool zeroes the dense block — restoring the padding guarantee the
no-free discipline currently provides for free — then rewinds `dense_bytes`,
clears `dense_frozen`, returns `weight_floor` to `span_end`, and re-stamps every
`dirty_epoch` as dirty, because region indices shift when `region_base` moves
back left. A reload then loads exactly as the first did. Not taken here: it is a
change to the KV allocator, on the strength of an intermittent failure in a test
harness rather than anything production does.

---

## 14. The two-tier zone and the spread residency

This section supersedes §3.1, §5.1 and §9.5.3. Those three describe one design —
equal cells, layer *i* in slot *i*, a held prefix — and each of them is wrong in
the same way: they reason about a layer in isolation and never about the
*pattern* the set of layers makes.

### 14.1 A prefix hides nothing; the missing set must be spread

A forward walks `0 … N−1`. A layer is resident (free) or missing (fetched into
the floating cell). The fetch for the next missing layer begins when the current
one stops occupying the cell, so the time available to hide it is the run of
resident layers in between:

```
window(gap) = (gap − 1) · t_c        gap = m[i+1] − m[i], cyclically
stall(gap)  = max(0, t_f − window(gap))
```

On the 4090 Mobile `t_c ≈ 1.4 ms` and a ~200 MiB layer is `t_f ≈ 8 ms`, so a gap
of 7 hides a transfer completely and a gap of 1 hides none of it.

`stall` is convex in `gap` and `Σ gap = N` is fixed by the number of missing
layers, so by Jensen the total is minimised when the gaps are equal. **The
missing set should be spread as evenly around the cycle as it can be.** A held
prefix does the exact opposite: it makes the missing set contiguous, so every
interior gap is 1 and not one byte of any transfer is hidden. Measured at
capacity 38 of 64, that is 29 consecutive fetches per forward, none overlapped.

Because the number of missing layers moves at runtime as the boundary trades
ground with KV, what is needed is not a set but an **order whose every prefix is
well spread**. `layer_stream/order.rs` builds it by repeatedly bisecting the
widest gap. On a power of two this reproduces bit-reversal — at every `k = 2^j`
the missing layers are exactly equally spaced — and unlike bit-reversal it holds
for any `N`.

Two rules that look equivalent are not. Plain farthest-point insertion — take the
candidate whose nearest missing neighbour is furthest — is blind to the gap it
leaves behind: on `N = 36` it opens `0, 18, 9`, whose third gap is 18 against an
ideal of 12. Bisecting the widest gap is the rule that keeps the bound.

**The bound, stated honestly.** A nested sequence cannot be exactly even at every
size: `{0, 32}` on 64 layers cannot extend to the perfect three-point set. The
guarantee is `max gap ≤ 2·⌈N/k⌉`, exact at every power of two. Concretely: **up
to 8 missing layers of 64 stall zero**; at 9 the ninth pick halves one 8-gap and
~7.6 ms reappears against ~90 ms of compute. That is worth taking, because the
alternative — recomputing an exactly-even set whenever capacity moves —
re-fetches nearly the whole missing set for a one-layer change and thrashes under
an oscillating boundary.

**Nesting is also what makes a boundary move cheap.** The resident set is a
prefix of the protection order at every budget, so conceding ground appends to
the missing set and taking it back pops from the end. A layer that survives the
move keeps its address: nothing is relocated, no surviving view is rebuilt, and
no layer is re-fetched for having moved. Growth therefore restores the spread
that a concession cost, rather than filling new ground with whatever faults next.

### 14.2 Uniform cells charge every layer the largest layer's size

§5.1 estimated the waste at ~2% from parameter counts. Measured from the
checkpoint's actual quants it is **18%**, because llama.cpp's positional
`use_more_bits` rule gives a few layers wider tensors than the rest — on the
27B's Q3_K_M, `ffn_down` is Q5_K on blocks 0–3 and Q4_K elsewhere, and `ssm_out`
is Q8_0 on 24 layers and Q3_K on the other 24.

```
blk 0,1,2   202.8 MiB     ← the max, and what every cell cost
mean        165.6 MiB
```

The consequence is sharper than the average suggests, because `slot_bytes` is a
`max`: a quantization schedule that shrinks 60 of 64 layers and leaves the
largest three alone shrinks the zone by **nothing at all**. That is exactly what
happened — a schedule saving a measured 954 MiB of layer bytes moved the carve
line not one slot, because all of it landed in dead space inside cells.

So the zone is split by what a layer actually does:

```
span_end ─┐
          │  resident tier   layers at their own size, most protected first
          │                  no cell, no eviction, no dead space
          │  floating cell   one, sized for the largest streamable layer
floor  ───┘
```

The streaming machinery genuinely needs one cell — the layer under the wave
occupies it while the next fetch waits for it to be freed. Everything above that
one cell was never streaming; it was residency wearing a cache's clothes and
paying a cache's overhead. Against the same ground: 34 held under equal cells,
**44 held dense**; and `Σ` over all 64 layers is 10,597 MiB against `64 × max` of
12,980 MiB, which is the difference between a model that fits the zone and one
that does not.

**The cell costs nothing when nothing streams.** It is planned only when the
budget cannot hold the trunk, so a zone that fits every layer allocates no cell,
issues no transfer and arms no fence — the §7 degenerate case reached honestly
rather than by a branch. When the boundary later concedes ground, the first layer
given up frees its own space and the cell is carved from it.

`ZonePlan::used_bytes` also claims only what it uses, so the remainder too small
for another layer goes back to KV instead of being lost inside a last cell.

### 14.3 What this deletes

The replacement policy. §13.1 recorded that the rule was Bélády's and optimal;
it was, and it was answering a question that no longer exists. With fixed homes
there is no victim to choose — `victim`, `keep`, `slot_score` and the
distance-ranked plan are gone. What remains is "is the next missing layer in
flight yet", and every decision is made once, by the zone, at carve time.

`COMMITTED_DEPTH` falls to 1 and cannot usefully be more: there is one cell, and
the layer occupying it is the layer the wave is standing on.

### 14.4 The warm tier follows the same nesting

Its membership rule was "the top `W` layers of the model", correct exactly while
the resident set was a prefix of the *layer* order. Under a spread residency the
streamed layers are spread, so the tier now holds the first `W` entries of the
eviction order — streaming under every capacity, for the same nesting reason, and
still needing no redraw when the boundary moves. It is a set rather than a run,
so the fill scatters over the pack instead of walking it contiguously; that is a
one-time startup cost paid to hold the right layers.
