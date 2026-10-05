# Decode CUDA graphs — one capture layer for every model

Scope: the decode forward of every batched model in this repository, one token per row,
with CUDA graphs at up to **four sessions**. A model-agnostic capture layer sits in the one
place all models are driven (`drive_wave`). Each model writes its decode layers as
regions; the capture layer opens one **wave chain** (§12.12) for every decode-only forward,
a scope whose links are regions and host gaps (Rust logic that cannot break the graph
invariants). At four rows or fewer each region is a graph link (captured lazily, keyed on
the row count, and replayed on the compute stream); above four the same links launch
immediately, with no capture. Regions with no host protocol between them fuse into one
graph; the MoE host protocol is where a chain cuts. Waves that are not decode-only
(prefill, glue, verify windows) run the model's existing mixed forward, built from the
same typed ops (§12.10).

The mechanism is taken from Strata (`strata`, v0.1.26), whose sources were read for this
design (§1). The model survey (§2) and the blockers (§3) come from reading each model's
wave path, the shared batched layer, the MoE dispatch, the arena and the CUDA backend.
Launch counts marked *inferred* were estimated from structure, not counted from a trace;
§8 replaces them with a measurement before anything is built.

---

## 1. What Strata does with CUDA graphs

| Path | Granularity | Where |
|---|---|---|
| Verify window (speculative, `T ≤ 8` rows) | One graph per `T` covering every layer in `[lb_, le_)` and the head. Separate commit graph (accept prefix, rewind recurrent state). | `verify.cpp`: `capture`, `record_window`, `capture_commit`; `exec_[9]` |
| Single-token decode, current | One graph for all layers. Each layer ends with a `doorbell_wait` spin on a mapped flag, then a kernel (not a memcpy node) copies the CPU pool's answer in. | `session.cpp:829-911`, `session_capture_token` |
| Single-token decode, earlier | Two graphs per layer (`pre`, `post`) split at the router, with the host running the CPU pool between them. Replaced by the single graph because the split existed only for that host step. | `session.cpp:217`, `gr.execs[l]`, `gr.posts[l]` |
| Batched verify (≤ 8 slots) | One token per row, one graph per ordered slot combination: `exec_bm_` keyed by `batch_key(rows, S, hbase)`. Attention is a per-row loop inside the capture over that slot's K/V pointers. | `verify.cpp:1780-1810`, `verify.hpp:224-230` |

Common to all of them:

- Raw runtime API only: `cudaStreamBeginCapture(…ThreadLocal)`, `cudaGraphInstantiate`,
  explicit `cudaGraphUpload`, `cudaGraphLaunch`. No `cudaGraphExecUpdate`, no node
  parameter edit. Topology and every kernel argument are fixed at capture.
- All per-run variation lives in device tables and mapped staging.
- A copy that must be a graph node is a kernel, because a copy-engine node splits the
  WDDM submission (Strata measured 67 flushes per token). WDDM batches submissions and
  drains them at sync points; all three dev machines run WDDM (`docs/performance.md`).
- Correctness bar: the argmax equals plain greedy single-token decode.
- Strata publishes no graph-on versus graph-off decode figure, and its own issue #610
  records the window as DRAM-bound. Nothing here claims a measured benefit.

What does not transfer: the batched graph is keyed on the slot combination because slot
pointers are baked into kernel arguments. Ours reads slot addressing from device tables
(§4.4), so the key carries no slot identity: layer (or span), region kind, row count,
placement epoch and, where a data-dependent grid needs it, a small variant (§12.13).

---

## 2. Model survey

Every model is driven through `drive_wave` (`wave_driver.rs:339-798`), the single choke
point: prefill slicing, folding of one-token prefills into the decode group
(`:461-483`), `[decode | prefill | glue]` assembly (`:509-513`), `sweep` (`:620`),
`rollback_wave` (`:645`), the logits `into_vec()` (`:719`) and `advance_decode_rows`
(`:728`). Models implement `WaveSweep` (`:120-277`); the uniform dense models through
`BatchedInference<M>` (`batched_inference.rs:5867`), the hybrids, Flash-Next and
DeepSeek through their own implementors.

| Model | Layers | Layer kinds | Layer function | MoE | Launches per layer *(inferred)* |
|---|---|---|---|---|---|
| `quantized_llama` (Llama-2 7B, Llama-3.2 3B) | checkpoint-defined | attention | shared `forward_layer_batched_mixed` (`batched_layer.rs:495`) | no | 12 to 16 |
| `quantized_qwen2` (0.5B) | checkpoint-defined | attention | shared | no | 12 to 16 |
| `quantized_qwen3` (8B) | checkpoint-defined | attention, QK-norm | shared | no | 14 to 17 |
| `quantized_qwen35` (0.8B, 9B) | 9B: 32 | 24 DeltaNet + 8 attention | own loop `sweep_layers` (`qwen35/forward.rs:1175`); attention layers call the shared function | no | DeltaNet 17 to 19, attention 14 to 17 |
| `quantized_qwen38` (27B) | 64 | 48 DeltaNet + 16 attention | same machinery as `qwen35` | no | as `qwen35` |
| `quantized_qwen3_moe` (30B-A3B) | 48 | attention | shared | every layer, 128 experts top-8 | 35 to 45 |
| `quantized_qwen35_moe`, `quantized_qwen36_moe` | 40 + MTP | 30 DeltaNet + 10 attention | `qwen35` loop | every layer, 256 experts top-8 + gated shared expert | DeltaNet 30 to 40, attention 35 to 45 |
| `quantized_qwen38_moe` = `qwen4exp` (Flash-Next) | 48 + MTP | 36 DeltaNet + 12 QSA attention | own loop (`qwen4exp/wave.rs:3033-3505`) | every layer, 512 experts top-10 + shared | DeltaNet+FFN about 30, attention about 43 |
| `deepseek4` (V4-Flash) via `latent_moe` | by checkpoint | latent attention + gallery recall | own loop (`latent_moe/wave.rs:1337`) | every layer, MXFP4 experts | not counted |

What the models share, and so what one capture layer can serve:

- **The wave driver and its host syncs.** Embedding gather and id readback, the logits
  `into_vec()` after the sweep, and `advance_decode_rows` are per forward and live
  outside any layer region in every model.
- **The dense layer function.** One function serves the whole attention-only lineage
  and the attention layers of the hybrids. The paged decode kernel reads slot headers
  from a device buffer built once per forward (`build_decode_metadata`,
  `batched_inference.rs:1071`, `qwen35/forward.rs:891`) at
  `buf.dev_ptr() + layer_idx * stride`, so it is already table-driven.
- **The MoE dispatch.** All four MoE models, and DeepSeek, reach
  `ExpertCache::forward_routed` (`expert_lre/handle.rs:953`) and
  `Dispatch::forward` (`expert_lre/dispatch.rs:507-787`). None has a routing readback;
  the router is not a boundary. The boundary is the host protocol around each
  invocation (§4.5). One change to `dispatch.rs` serves every MoE model. The CLAUDE.md
  invariant 3a is stale on this.
- **DeltaNet.** `qwen35`, `qwen38` and `qwen4exp` share one mixer: a per-forward
  `build_wave_table` (pool-allocated, unstable address) and a table-driven decode
  conv, step and norm-gate.
- **The stream.** Every model launches on the null compute stream
  (`device.rs:1029`), and the MoE hand-off asserts it (`live_table.rs:103-108`).

What is model-specific, and stays a model's own region or stays eager:

- DeltaNet and attention launch sequences; QK-norm; the shared expert.
- Flash-Next only: HC mix and combine, PLE at layer 1, QSA selection, the MTP block.
- DeepSeek: a per-sequence-host-loop attention half, window-ring eviction on the host,
  provenance gallery recall (GPU-side, no sync, capturable *inferred*), and a corpus
  snapshot bracket around the sweep.
- Qwen3.8 27B: trunk layers stream through the weight zone (`q.layers.ensure(li)`,
  `forward.rs:1197`), so a layer's weight addresses can move between waves.

**Evidence on whether decode is launch-bound.** There is none for the Flash-Next, hybrid
or dense decode paths. The only measurement is the DeepSeek note
`deepseek_decode_launch_overhead.md` (now stale): decode about 58 to 60% GPU-bound at 16
sessions, attention only 2.6% of GPU time. A figure quoted elsewhere as "67 µs per launch"
is the cost of one 819 KB expert upload (`moe_live_dispatch_design.md:633`), not a launch
cost, and nothing in this design depends on it. §8 measures launches and host
submission per model before any graph is built.

---

## 3. Generic design

A model's forward is split into **regions**. A region is a contiguous run of kernel
launches (a layer, a layer half, the head, or adjacent host-free regions fused) that is
*stateless with respect to the host*: it reads device tables and device buffers, allocates
only from the wave tier through its ops, and does no host interaction. A model runs each
of its regions as one link of the chain:

```rust
chain.run(&mut RegionLink::new(layer, DenseLayer::new(w), &mut NoHost), &staged, &mut x)?
```

A region is a type implementing `Region` (§12.4), whose single `record` method holds the
model's existing kernel launches and is generic over the launch mode. `drive_wave` opens one
**wave chain** (§12.12) for the whole forward: a scope that holds the wave's memory and the
lazily filled graph slots, and in which links and host gaps alternate.

**One decode path.** Every decode-only wave runs the regions, whatever its row count, so
the region bodies are *the* decode composition of each model, not a second one beside the
eager forward: they are exercised by every decode wave and by every sweep gate at every
rung of its ladder. The row count decides only how a link is issued. At `R ≤ 4` the first
sight of `(layer or span, region kind, R, placement epoch, variant)` (the link's `LinkKey`
plus the chain's `R` and epoch) records the region under capture and fills its slot, and
every later sight launches the stored executable. Above four there is no slot: the link's
`record` runs on a `Chained` recorder (§12.3), the same mode a refused slot uses, inside
the same host protocol window. Waves that are not decode-only have no chain and run the
model's mixed forward, which calls the same ops in `Eager` mode, so each kernel still has
one wrapper. No flag chooses among these: the verdict does. The model supplies nothing else:
no second implementation, and no trait method that exists only to opt out. The traits,
types and module layout that make misuse a **compile error** are in §12.

### 3.1 Eligibility

`drive_wave` decides once per wave, right after assembling `[decode | prefill | glue]`
(`wave_driver.rs:513`): every row has `q_len == 1` after the one-token-prefill fold, there
is no prefill, glue or verify member, the sweep covers the full layer range
(`kv_layer_range`, `:138`), and the model's own preconditions hold (§3.2). Such a wave
gets a chain. The row count then decides only whether its links capture: `R ≤ 4` carries
a `GraphRows`, and above four it does not (§12.5). The context carries the answer; a model
never recomputes it.

A model that fails a precondition for structural reasons (weights that stream on this
card, an activation dtype other than BF16/F16, the unfused DeltaNet route) runs its mixed
forward for decode too, and the stats line says why. That is decided by the model and the
card at load, not per wave, so it is a property of a configuration, never a path that
alternates with the regions.

### 3.2 Per-model preconditions

Checked by the model when it builds a region, reported in the stats line when they fail:

- the attention kernel takes the paged decode route with matching dtypes (no
  `to_dtype`, no non-paged host rope tables);
- the fused DeltaNet path is taken (head_dim 128, F32 operands, `mix.rs:1549-1565`); its
  tensor-op fallback loops and concatenates per span and is not capturable;
- trunk-layer weights are resident, not streamed (`qwen38` 27B on a small card), because
  weight pointers are baked into a layer's graph;
- QSA selection and host-looped attention (Flash-Next, DeepSeek) are not inside a region.
  Until their kernels are ops they end the chain (a counted chain break, §12.12); once
  they are ops they are gaps;
- a kernel whose grid depends on per-wave data either takes the kernel change below or is
  keyed by a host-known variant (§12.13).

### 3.3 Region kinds

| Kind | Contents | Used by |
|---|---|---|
| **Layer** | a whole dense layer: attention half and FFN half | llama, qwen2, qwen3, and the attention layers of `qwen35` / `qwen38` |
| **Mixer+FFN** | a DeltaNet layer and its FFN | `qwen35`, `qwen38`, `qwen35_moe`, `qwen36_moe`, Flash-Next |
| **FFN** | the FFN half alone, after an eager mixer | Flash-Next attention layers, DeepSeek |
| **MoE** (not a kind of its own) | a Layer, Mixer+FFN or FFN region whose FFN is an MoE invocation, wrapped in a `MoeLink`; the host runs the dispatch protocol around it (§4.5) | all MoE models |
| **Head** | `final_norm`, `lm_head` (and `hc_mix(out_hc)` for Flash-Next); output F32 logits at a fixed address | every model |

Kinds compose: a dense Layer region has no host protocol, so adjacent ones fuse into one
graph link; an MoE layer is one `MoeLink` (§12.4) whose host protocol runs in `before` /
`after` inside `WaveChain::run`, so it cuts the chain there. The router output and the
normed hidden state stay inside that one graph, so nothing but the residual crosses a link.

---

## 4. Capture layer (model-independent)

### 4.1 Key and cap

`(layer or span, RegionKind, R, placement epoch, variant)`, captured lazily on first use
and kept. Decode rows are one token each, so `R` equals the session count and the cap is
**R ≤ 4**: at most `links_per_forward × 4 × variants` executables per model (Flash-Next:
49 per-layer links × 4 = 196; a dense model's fused trunk: 4). Row padding is not used (padded rows
would still route to experts and fill tiles). `R` fixes every launch scalar a region
passes (dispatch `launch_tiles`, `tile_w`, `n_sub`, `workers`, `a_ub`; the paged kernel's
`num_splits`, a function of active slots × KV heads and the SM count; DeltaNet
`table.n`), so they are captured. Above the cap the same links run `Chained`, computing
those scalars from the wave's `Rows` on every launch exactly as the eager kernels do
today; no slot, key or executable exists for them.

### 4.2 Stream

The compute stream is the legacy null stream and it cannot be captured. The capture
layer **captures on a created, non-blocking stream and launches on the null stream**
(`CudaDevice::new_with_stream`, `device.rs:744`, already builds a device handle with its
own stream, cuBLAS and curand). A graph launched into the null stream orders against
everything else exactly as the eager launches do, so no other fence in the tree changes.
The capture stream is non-blocking because other threads submit null-stream work while
the forward thread captures (the persistence thread's hot-to-warm copies run on the
primary stream); against a blocking-flag capture stream, null-stream work from any thread
would become an implicit dependency on the capture and invalidate it. This rests on CUDA
semantics and is proven by the first prototype (§7), with a persistence copy in flight
during a capture, before anything depends on it.

- Launch helpers take the stream from the capture context; `live_table.rs`'s null-stream
  assertion is relaxed for the capture stream only.
- The cuBLAS handle is re-bound to the capture stream for the capture (`cublasSetStream`)
  and restored.
- The capture mode argument is `ThreadLocal`. It is a flag on the driver call and costs
  nothing in the Rust design: no thread-bound type, no `!Send`, no lock, because the
  capture begins and ends inside one synchronous `WaveChain::run` call and its session is
  a local that is never stored or sent. Its effect is that only the capturing thread is
  forbidden from making capture-illegal calls (`cudaMalloc`, `cudaFree`, a synchronise);
  other threads' launches and allocations neither break the capture nor are captured.
  That is what the design needs, because the persistence thread allocates and copies while
  the forward thread captures. `Global` would fail the capture on any of those calls, and
  `Relaxed` would let our own thread's illegal calls through silently, so `ThreadLocal`
  is the lightest mode that keeps the driver's backstop where it matters. Stream-level
  illegality (a query or synchronise on the capturing stream, an event wait on an unjoined
  stream) holds in every mode, so the stager's and pipeline's own-stream `cuStreamQuery`
  never touch it. Event fences inside a region use capture fork/join or stay outside.
- A side stream used for fork/join lanes (§12.13) is non-blocking for the same reason:
  a blocking-flag side stream would synchronise implicitly with the null stream and
  serialise the lanes. The prototype proves this before `fork` is built.

### 4.3 Allocation: the wave tier, bound by lifetime and a placement epoch

A replayed graph uses the addresses it was captured with. The wave transient tier is a
bump allocator over three per-phase spans (`LayerPhase::{Attention, Ffn, Forward}`,
`wave_plan.rs:378`). The wave chain opens the phase generations once
(`begin_wave`, `bump_arena.rs:1545`; `batched_layer.rs:530,606`, §12.12) and each region or
gap starts each phase it uses from the span start through a region mark, so a region's allocations land at
**span base plus a deterministic offset**. That makes tier memory graph-safe on two
conditions, and each is enforced differently:

1. **Valid inside the scope: a lifetime.** The generation guard already lends every tier
   tensor its `'w` (`wave_empty`, `Generation::alloc`), and `begin_wave` refuses an
   overlapping generation. A graph's captured tensors are therefore the guard's
   borrows and cannot be named after it drops.
2. **Same address on the next replay: an epoch.** The tier base moves only when the tier
   is re-placed or its plan is raised (`place_transient`, `plan_wave_transient`,
   `cover_wave_transient`, the rebase in `begin_wave`). A monotonic **placement epoch**
   is bumped at each of those and is part of the graph key; a replay in a different epoch
   recaptures. It is a counter, not a geometry comparison, for the reason hot-path
   invariant 7 gives (a boundary can move and come back reading the same).

Each op allocates in its own phase's span (attention-half ops in `Attention`, FFN ops in
`Ffn`, the head in `Forward`), and the mark rewinds every span a link used, so the offsets
are a function of the link alone. The `Forward` span today also holds per-forward setup
that every layer reads (decode metadata, position ids, RoPE tables); inside a chain that
would cross links, so it is staged as `Table`s instead (§4.4). The residual stream and the
final logits cross layers and so cannot be tier memory: they live in persistent fixed buffers, and the caller reads the logits before the
next forward starts. This keeps the existing wave-tensor kernel wrappers (which already
take `LiveTensor<'w>`) usable unchanged, and needs no second allocation per `R`.

Capture requires the tier **without the pool fallback**: the capture-mode constructor
returns a leased tier tensor or an error, never the `Owned` pool tensor `wave_empty`
falls back to when there is no wave.

During capture the following are **capture failures**, never silent: a ticket falling back
to the driver pool (`wave_buffers.rs:334-407`), default `Tensor::empty` / `zeros` /
`from_vec` (stream-ordered `cuMemAllocAsync` would become a graph memory node), pageable
host-to-device copies, `stream.synchronize()` and any `cudaMalloc`. The context counts them
and asserts; `forbidden_alloc` is the detector to build on. The bump arena is never dropped
during a graph chain: the wave chain (§12.12) opens all three phase generations before its first
link and holds them until its last launch has been enqueued, so the fence a generation drop
performs (`bump_arena.rs:660-672`) cannot land inside a capture. A drop observed during
capture is a capture failure like the others. Each region and gap reuses its phase's span
from the span start through a region mark that rewinds the cursor without a fence (§12.12),
so the addresses stay "span base plus a deterministic offset" and the tier needs one
region's peak per phase, not the sum over layers.

Grown before the chain opens, for the wave's `R` and never below `R = 4` (growth allocates
and can synchronise, so it never happens inside a scope, captured or `Chained`; it is
monotonic, so it happens once per new high-water mark, and `invalidate_where` recaptures
the graphs that bake the grown buffer): the MoE bucketize workspace
(`ensure()` reallocates on growth, `cuda.rs:7518`), split-K scratch, and the paged decode
kernel's global split-KV partial pool (`cudaMalloc`, `cudaStreamSynchronize`, `cudaFree`
on growth, `int8_decode_kernel.cuh:2157-2209`). Every kernel in a region is loaded by a
warm-up wave first: lazy module loading is an implicit synchronisation, and a spinning
expert worker can block it (`moe_live_dispatch_design.md` §0.1).

### 4.4 Staging: what varies per wave is data

Everything a region reads that changes per wave is a device table at a **fixed address**,
written before the chain by one async copy from a pinned block (outside the graphs, so
no pageable or host-pointer copy is captured). The block is a small ring covering the
run-ahead, so one is not rewritten before its copy has executed. Contents:

- slot headers and metadata (`build_decode_metadata`), already one build per forward;
- any other per-forward buffer that every layer reads and that today lives in the
  `Forward` phase (gathered position ids, RoPE tables): a region mark would rewind it
  under a later link, so it is a `Table`;
- hybrids: the DeltaNet `build_wave_table`, moved from a pool allocation to its fixed
  address and rewritten each wave (state-half swaps at `commit_wave` change which
  buffers it points to);
- MoE models: the dispatch sequence base (§4.5);
- the head-row selection is not needed for decode-only waves (every row is scored).

The same staging step yields the `Known<T>` values the chain composes on
(`StagedWave::known_*`: row count, per-layer kind, host-side counts already built for the
tables), so no composition decision reads device data (§12.13).

Because kernels already consume these as descriptor tables (hot-path invariant 2b), varied
paged KV and varied per-session windows replay without recapture.

### 4.5 MoE dispatch (shared by every MoE model)

Per invocation the host does `begin_invocation` (sequence, `PassState.reserved`, pass),
`hold_for_ring`, `clock.enqueue(row, ticket)`, the kernel launches, `cuStreamQuery` (so
WDDM submits the bucketize), then `stager.send` and `tx.send` of a `Routed` message
(`dispatch.rs:456-636`). The pipeline thread blocks in `ring.wait(slot, word)` and the
stager polls `ring.ready`; neither assumes the word is already published.

**The host keeps doing all of this, per layer, between graph launches:**
`begin_invocation`, `hold_for_ring`, `enqueue`, send both messages, launch the layer's
graph, `cuStreamQuery`. This is the `RegionHost` pattern of §12.4: `Dispatch::forward`
(`dispatch.rs:507-636`) splits into `before` (reserve, hold, enqueue, send), `record` (the
launches only) and `after` (the query). `WaveChain::run` captures and instantiates an
`Empty` slot **first**, with no host protocol (a capture executes no kernel, sends no
message and reserves no ticket), then runs `before`, the launch, and `after`. The order
in eager mode is the same: the messages are sent before the layer's kernels are launched
(safe, since receipt assumes nothing) and the query follows them, so there is one path.
Each query flushes a layer's launch before the next layer's `hold_for_ring`. Pre-sending a whole forward's messages is **not** done: it needs every
invocation's ring slot and channel entry free before the first launch
(`SUMMARY_RING = 64`; Flash-Next needs 49), blocks the 65th send, and strands
`PassState.reserved` if a capture or launch fails after the sends, so the pipeline waits
30 s and aborts.

#### Audit of the expert-offload path against the chain

The no-sync offload design (`moe_live_dispatch_design.md` §0) is already close to
chainable, because everything the host changes while kernels run lives in mapped memory
the kernels read, never in a kernel argument. What survives capture unchanged:

- **Expert weights are never baked.** The GEMMs read a per-invocation device snapshot
  (`snap`, written by `moe_bucketize` phase 1b), not an address. A concession, an
  eviction or a boundary move therefore cannot invalidate a captured MoE graph (invariant
  7). Only startup-lifetime objects are baked: `live_row` and `abort` (mapped), the
  summary and promotion rings, and the two pinned ranges.
- **`MoeLive` is passed by value** (`dispatcher.cu:1491`), so the host stack struct in
  `Dispatch::forward` is read once at capture and its address is not retained.
- **Cold-expert spinning is not captured.** A capture executes no kernel, so no worker
  spins during it. The workers wait on the live gate entry and the stager publishes it by
  host store, so a replay behaves exactly like the eager launch.
- **`ReclaimClock` is host atomics** (`reclaim.rs:54-65`): `enqueue` runs in `before`
  every launch, so Rules R and R′ hold unchanged.
- **The pipeline and stager threads** use their own non-blocking copy stream and
  `cuStreamQuery`, and a capture is stream-scoped (§4.2), so it leaves them alone.
- **Boundary moves** (`request_kv_ground`, `set_weight_floor`, `reclaim_spare_ground`)
  refuse while a wave generation is open and the chain holds them all for the scope, so they
  run outside it and the placement epoch is stable inside.

What does not work in a chain as written, and the fix for each:

| Operation | Where | Why it breaks | Fix |
|---|---|---|---|
| `summary_seq` and the summary ring slot as launch scalars | `dispatch.rs:574-575`, `moe_bucketize.cu:297-340` | Baked at capture: every replay stores the same word into the same slot, the pollers never see a new sequence and the pipeline aborts after 30 s. | The kernel takes a pointer to a per-wave staged sequence base plus the layer's fixed ordinal `i` in the wave's MoE sweep and computes `seq = base + i`, the slot and the `summary` pointer on the device. The ticket is not a kernel argument. The dispatch `seq` is a global monotonic counter (`begin_invocation` takes `f.seq` and increments it), not a function of `row`; `base` is `f.seq` read at staging, and `i` equals `row` only when the wave invokes every MoE row once from 0. `before` asserts that the host's `seq` equals `base + i`, so a model that invokes a MoE layer twice or out of order in a wave fails loudly instead of desynchronising the ring; acceptance 10 (§9) runs that check for every MoE model, including drafters. |
| `begin_invocation`, `hold_for_ring`, `clock.enqueue`, both `send`s and `cuStreamQuery` inside `forward` | `dispatch.rs:547-612` | Host protocol in the middle of the launches; a stream query on a capturing stream invalidates the capture, and a send during capture reserves a ticket for a launch that has not happened. | The `before`/`record`/`after` split above. `record` contains launches only. |
| One shared `snap`, `remote`, `counters`, `remote_dst` and `scratch` per `Dispatch`, plus the bucketize workspace | `dispatch.rs:391-398` | Their addresses are baked into every layer's graph. They are valid while the `Dispatch` lives and the workspace does not grow; and because every layer shares them, two MoE invocations must never overlap. | Typed `Fixed` buffers owned by the `Dispatch`, with the graph cache dropped or invalidated with it. The workspace is pre-grown for `R = 4` (§4.3) and `invalidate_where` recaptures on growth. `MoeProtocol` is not `Concurrent` (§12.4), so no fork lane (§12.13) can hold a MoE link or run beside one. |
| Void launchers that skip silently | `run_moe_bucketize` (`moe_bucketize.cu:550-556`), `run_grouped_quantized_matmul` (`dispatcher.cu:1438-1470`) | A failed guard (`n_sub`, `qtype`, `num_tiles`, workspace bound) returns without a launch, and `cudaLaunchKernel`'s status is dropped. Eager leaves the previous layer's tables; a captured graph bakes the missing node into every replay. | The launchers return a status and the typed op returns `Err`. The recorder compares the captured node count with the sum of the ops' `Op::LAUNCHES` before instantiating (`GraphError::MissingLaunch`, §12.7). |
| `weights.flatten_all().contiguous()` | `dispatch.rs:545` | A no-op for the router's dense output, but an allocate-and-copy if a producer ever hands a strided tensor (invariant 2), and an allocation inside a region is refused. | `expect_dense` (invariant 1b); the op takes the router's `Act` and rejects anything else at construction. |
| Outputs through `resolve_typed_out` and `resolve_u8_out` | `cuda.rs` grouped, gather and scatter wrappers | Each may fall back to the owned pool, whose address is not arena-bound. | The op owns its output typed as a wave-tier `Act` (§12.10); there is no fallback-capable form in a region. |
| Test-only observers: `capture_routing` (`to_vec2`), tensor-assert checkpoints, `gpu_span` | `quantized_qwen3_moe.rs`, `dispatch.rs:563` | A readback or a fence cannot be captured, and a profile span records inside the region. | Banned in a region (§12.11). `gpu_span` is compiled only under `all(feature = "profile", feature = "cuda")` and is a no-op otherwise, so a production build has none. With the features on it records an event, queries pending events and can block once when the pool is far behind, so it wraps the graph launch, outside the capture; the observers stay off in a build that uses graphs. |

Three findings from the code that bound the rows above:

- **One call site.** Every MoE model reaches `ExpertCache::forward_routed` through
  `quantized_qwen3_moe::SparseMoeBlock::forward_dynamic` (`qwen3_moe`, `qwen35`, `qwen4exp`
  reuse it, as do their heads) or through the `latent_moe` engine (deepseek4). `moe_layer_idx` is the compacted
  dense MoE ordinal, so one `MoeLink` implementation serves them all.
- **Drafters.** Both MTP-style heads are routed MoE blocks with their own `moe_layer_idx`
  past the trunk's (`qwen35/quantized_weights.rs`: "straight after the last trunk MoE
  layer"; `qwen4exp` merges its head's experts into the same grid). In the trunk's own
  forward the `qwen35` head pass is attention-only (`draft.rs` `head_wave_pass`: it fills
  the head's KV and discards the FFN), so the trunk's MoE sweep is ordinals `0..N-1`, once
  each, in order. The head's MoE runs only in the draft walk (`qwen35/draft.rs`
  `draft_cohort` via `mtp.rs`, `forward_layer_batched_mixed`; `qwen4exp/draft.rs`
  `head.block.moe.forward_parts`), one decode row per sequence per step, between forwards.
  Each draft step is its own one-link sweep: the same row repeats across steps, so
  `begin_invocation` starts a new pass each time (`row <= last_row`) and `seq` keeps
  counting. A walk step therefore needs its own `StagedWave` with `seq_base` read at that
  step; it is never folded into the trunk's ordinal count. Until a drafter has that, it
  stays eager.
- **Verify path.** `forward_wave` (`qwen35/forward.rs`) is `drive_wave` over
  `layer_start..layer_end`; the head pass runs only when `layer_end == num_layers`. A
  partial layer range is a sub-sweep whose `seq_base` is read at its start.
- **Per-wave host values.** `n_sub`, `tile_w` and `workers_for(num_tokens)` are computed from
  the wave's token count and baked at capture. They are constant for a fixed `R`; where they
  vary with `num_tokens` they are part of the slot's variant key or frozen per `R`.

Two limits that are not bugs:

- **Instantiate or a first kernel load while a gate is spinning** can block until that gate
  ends, because a lazy load or instantiate synchronises implicitly (§0.1). It cannot
  deadlock, since the gate waits only on the stager's host stores and the stager makes no
  CUDA calls. It is a stall, charged to the capture cost with queued work (§10); kernels
  are loaded before the first capture (§12.11).
- **The display watchdog.** Each worker stops spinning after 1.5 s, below the WDDM limit, per
  kernel. A fused multi-layer submission could sum several stalls past it, so a MoE region
  is never `Fuse`d with another MoE region. The all-resident `NoHost` case has no cold
  expert and so no spin.

If an all-resident cache later drops the dispatch host protocol (a new code path, not an
existing configuration: `handle.rs:557` detects residency but still runs the ring, sends
and query), the MoE regions become `NoHost` and `Fuse` (§12.12) coalesces them into
fewer, larger links with no change to the model code. Dropping the protocol is not part
of this design; the host protocol is the only thing that cuts an MoE chain.

### 4.6 Failure

A capture, instantiate or launch error marks that slot `Refused(error, epoch)`. The
region's `record` then runs on a `Chained` recorder (the same ops, launched live) so the
forward completes, and the failure is reported in the stats line with model, layer,
region and `R`. A capture or instantiate failure happens before the region's `before`, so
nothing is reserved. A launch failure happens after it, and the `Chained` run inside the
same `before`/`after` window launches the same bucketize, so the summary word the stager
and pipeline wait for still arrives and a failure strands nothing. There is no retry loop: a refused slot is attempted again only when the
placement epoch moves, and it is never a silent permanent fallback, because every
refusal is counted and shown.

### 4.7 Host work outside the regions (every model)

`begin_forward`, `plan_wave_transient`, `admit_wave_kv`, `reclaim_spare_ground`,
`end_wave_transient`, the embedding gather and id readback (invariant 3b), the per-layer
Vec building and `reset_caches_at_zero` checks of the dense loop, `commit_wave` /
`rollback_wave`, the logits readback and sampling. None of it moves. Sampling's
`memcpy_dtov` (`batched_sampler.rs:1971`) happens after the final graph, so a sampler
that needs host paths (penalties, allow-lists) does not affect graph eligibility.

Inside the chain, host work is a **gap** (§12.12): arbitrary Rust plus `Chained` ops, with
no handle to the chain. The embedding gather and id readback run before the scope opens.
The scope exit is the wave's single stream fence (the generation drop); the logits are
read after it, outside the chain.

---

## 5. Per-model plan

Dense models can fuse the whole trunk into one link (`Fuse`, §12.12); MoE models cut at
each layer's host protocol. "Eager" below means work that is a gap or a counted chain
break, not a graph link.

| Model | Regions | Eager | Difficulty | Gain *(inferred; launch-bound at R ≤ 4 unmeasured)* |
|---|---|---|---|---|
| llama, qwen2, qwen3 | Layer × N, Head | embedding, head-row Vec on mixed waves | **lowest**: one shared function, no state, no MoE, no streaming | moderate: about 12 to 17 launches × layers, hundreds per token |
| `qwen35` 0.8B / 9B | Mixer+FFN (DeltaNet), Layer (attention), Head | recurrent-state bookkeeping, MTP head | moderate: DeltaNet table at a fixed address, state swap | highest launch-to-FLOP ratio of the dense group |
| `qwen3_moe` 30B-A3B | MoE Layer × 48, Head | dispatch protocol (host loop) | moderate: the dispatch change, homogeneous layers, the attention half is the shared paged path | high: 35 to 45 launches × 48 |
| `qwen35_moe`, `qwen36_moe` | MoE Mixer+FFN, MoE Layer, Head | DeltaNet table bookkeeping, MTP | moderate | high |
| Flash-Next (`qwen4exp`) | MoE Mixer+FFN × 36, FFN × 12, Head | QSA attention half (12), embedding, PLE layer, MTP, verify windows | high: the eager attention half, PLE, HC | largest absolute launch count, about 1,300 capturable |
| `qwen38` 27B | as `qwen35` | streaming | **hardest and least certain**: weight pointers move; likely GPU/PCIe-bound at this size | uncertain; keep off until streaming is pinned |
| `deepseek4` | MoE-half FFN, Head | attention half (host loops), window-ring eviction, corpus bracket | high: first target is the MoE half only | measured 58 to 60% GPU-bound (stale): bounded |

Secondary consumer: the `qwen35` MTP draft cohort (`draft.rs:306`) is fixed-shape per
`(cohort size, depth)` and sync-free until one readback, so it is the best draft graph
candidate after the decode regions. Verify blocks, DSpark and prefill are never captured.

---

## 6. Common requirements before any model

1. The capture module in `candle-core/src/cuda_backend/graph/` (begin/end capture,
   instantiate, upload, launch through `sys::`; cudarc 0.17.3 is a crates.io pin, and raw
   driver calls are already routine there).
2. `CaptureCtx::scope` in `drive_wave` and the `WaveChain` it lends (§12.12): `Link` and
   `RegionLink` (§12.4), the keyed cache with lazy slots, the per-op `LAUNCHES` node-count
   check, the eligibility predicate (§3.1), `Chained` issue for refused slots and for
   waves above four rows, `Known` / `Variant` (§12.13) and the anti-trait bans (§12.11).
3. The tier placement epoch, the region mark (a fence-free cursor rewind to span start),
   generation holding for the chain's lifetime, the pool-free capture constructor and the
   no-foreign-allocation assertion (§4.3).
4. The staging ring (§4.4).

---

## 7. Order of work

1. Counters and spans (§8): launches per wave by region and model, host submission time.
2. **Prototype on one dense model, Head region first, then one Layer region**
   (`quantized_qwen3` 8B). It has no MoE protocol, no DeltaNet table, no streaming and
   one shared layer function, so it proves the null-stream arrangement, the cuBLAS
   re-bind, the tier epoch, the region mark, the chain scope with lazy slots, the
   allocation assertion and token equality in isolation. It also proves that a
   non-blocking side stream can capture a fork/join lane. If §4.2 does not hold, it is
   found here.
3. The rest of the dense group (llama, qwen2) with no new mechanism.
4. **The MoE dispatch change and the per-layer host loop**, on `quantized_qwen3_moe`
   (homogeneous layers): `MoeProtocol` with the `before`/`record`/`after` split, the
   device-derived `seq_base + ordinal`, launchers that return a status, and `MoeLink`.
5. **DeltaNet Mixer+FFN** on `qwen35` 9B (fixed-address table), then `qwen35_moe` /
   `qwen36_moe`.
6. **Flash-Next**: Mixer+FFN and FFN regions with the attention half eager.
7. DeepSeek MoE half, `qwen38` 27B once streaming is pinned, the `qwen35` draft cohort,
   then, only if measured to matter, verify windows (per-wave span tables, `capture_spans`
   in table form) and a device-built QSA selection.
8. The composition layer of §12.13, each only when a model needs it and measurement shows
   it pays: `Fuse` and variants first (no new driver mechanism), then `fork` / `join`
   (after the non-blocking side stream is proven), then `notify`, and `scope_pending`
   last (it needs a two-deep tier and its overlap is limited by the sampled-token
   dependency).

---

## 8. Measure before building

On each model's own reference machine, 1 to 4 sessions, warm, record per wave: kernel
launches by region; host time submitting launches, separate from candle op dispatch and
arena bookkeeping; GPU busy time versus wall time and the idle gap between consecutive
kernels; and the share of waves that are decode-only versus verify windows (for models
that speculate, a low share means the decode-only scope covers little and verify-window
capture moves up). If the GPU is not idle waiting on submission at a given session count
for a given model, graphs cannot raise decode there and the cap for that model sits below
it. The cap of four is the working value; the measurement sets the real one per model.

---

## 9. Acceptance

1. **Token equality.** For every region and every `R` from 1 to 4, the graph wave's
   accepted tokens equal the same wave run uncaptured (every link `Chained`, built by the
   test through the graph module's test constructor), and both equal plain greedy
   single-token decode, in the model's `test_parallel_batched_forwarding` gate across its
   modes (BF16, Q8_0, C0 to C10 where present). Above four, where every decode wave runs
   `Chained`, the gate's existing ladder is the check, unchanged. The run crosses the cap
   in both directions.
2. **No host interaction in a replay.** A trace of one forward shows the staging copy, then
   per link only the model's own host protocol, one `cudaGraphLaunch` and, for MoE
   layers, one `cuStreamQuery`; the only stream fence is the single one at scope exit
   (plus one per counted chain break), and the logits are read after it.
3. **No allocation.** A capture asserts zero foreign allocations; a replay allocates
   nothing.
4. **Rewind.** After a rejected verify window (run by the mixed forward), the next decode
   wave reads recurrent state, the PLE window and the QSA index cache equal to the
   uncaptured run's, bit for bit.
5. **Boundary moves.** A forced concession or tier move between waves leaves every graph
   valid and the next wave's tokens equal the uncaptured run's. A forced workspace growth
   (a wave above the previous high-water `R`) recaptures the affected graphs.
6. **A failed capture strands nothing.** An injected capture failure at layer *k* runs that
   region on a `Chained` recorder, completes the forward, leaves `PassState.reserved`
   equal to `observed`, and does not retry until the epoch moves.
7. **Every model in §5 that gets regions passes its own sweep gate** unchanged.
8. **Lazy slots.** After a warm-up wave, a second wave of the same `R` performs zero
   captures and zero instantiations; the stats line shows the chain-break count per
   model, and it is zero for every model whose regions are all ops.
9. **The bans hold.** Every `ban!` entry (§12.11) has a UI test that fails to build with
   the prescriptive note, and the launch-count test shows the same launches at `R = 1`
   and `R = 4` for fully batched regions.
10. **The MoE protocol holds in a chain.** For every MoE model, including its drafters, a
    wave invokes each MoE layer once and in order: `seq == seq_base + ordinal` for every link,
    with no abort and no 30 s ring stall. A forced skipped launch (an op whose guard fails)
    is refused at capture with `MissingLaunch`, never replayed. Two MoE links never overlap
    on the shared `snap`, `counters` and `scratch`.

---

## 10. Risks

- **The null-stream arrangement (§4.2).** If capturing on a created non-blocking stream
  and launching on the null stream does not behave as assumed (in particular, with the
  persistence thread's null-stream copies in flight during a capture), the compute stream has to move
  off the null stream, which re-derives every cross-stream fence (`wave_buffers.rs:257-261`,
  the persistence copy stream, the MoE hand-off). The largest cost in this design, so the
  prototype is first.
- **Silent foreign addresses** from a ticket falling back to the driver pool.
- **Pointer-baked weights.** Any model whose weight addresses move (streamed trunk
  layers) would replay stale pointers; those models stay eager.
- **Graph count and memory.** Executables scale with links per forward × R ≤ 4 ×
  variants; lazy capture and least-recently-used eviction of an `R` bound it, and
  `Variant::COUNT ≤ 8` bounds the multiplier.
- **Region-mark soundness.** The arena's own comment forbids nested release on the wave
  path because tensors cross phases. The region mark is sound only because nothing but
  `Resid` / `Table` crosses a link (the `'c` brand), and it is new arena code that needs
  its own tests.
- **Capture and instantiate cost with queued work.** Lazily capturing mid-wave while the
  GPU still has earlier links queued is unmeasured; a slow first wave per `(layer, R)` is
  the price, and §12.12's open question is per-layer split versus fused spans.
- **Fork needs a non-blocking side stream**, and its overlap benefit depends on the
  prototype proof (§4.2).
- **Launch-bound or not**, per model (§8).
- **The regions carry decode at every row count.** Above four rows the regions replace
  each model's decode composition, not just add graphs below it, so a region that is
  slower than today's decode path (a missing fusion, a launch the old path skipped)
  regresses the high-batch rungs that set the published aggregate numbers. Each model's
  sweep gate is compared rung for rung against the previous run before its regions land,
  and the launch-count test pins the op sequence at `R = 64`.
- **Run-ahead of the staging ring** against the dispatch ring.

---

## 11. Non-goals

- Row padding to bucket sizes.
- `cudaGraphExecUpdate` or node-parameter edits.
- Graphs for prefill, verify windows, glue, QSA selection or host-looped attention in the
  first change.
- A graph per combination of slots (Strata's `exec_bm_`).
- Pre-sending a forward's dispatch messages, or moving the MoE host protocol into a graph.
- Device-data control flow in a graph (conditional nodes, device-side loops); composition
  is host-known only (§12.13).
- `fork` / `join`, `notify` and `scope_pending` in the first change: they are specified
  in §12.13 and built later, after the prototype proofs they depend on.
- A per-model opt-out flag, an environment toggle, or a second code path per region.
- Moving the compute stream off the null stream, unless §10's first risk forces it.

---

## 12. Rust API

Goal: a model author implements one small trait per region and writes it as a short
sequence of typed operations, and **the mistakes that corrupt a replay do not compile**.
The framework owns capture, replay, allocation and fallback; the author never allocates.
Where the compiler cannot reach (§12.7) a runtime guard catches the mistake at the first
capture, and a test runs every region once so it is caught in CI, never in production.

Patterns used: **typed buffer roles** (activation, residual, table and weight are
different types, and an op's signature says which it takes), **a closed set of typed
ops** that own their output allocation (§12.10), **typestate** (eager versus capturing is
a type parameter, and an operand bound that is loose when eager and strict when
capturing), **branded lifetimes** (activations cannot outlive a capture), **newtypes with
private constructors** (`GraphRows`, `Resid`, `DecodeWave`), **a type parameter for the
activation dtype**, **associated types** (a region's host protocol is part of its type),
**exhaustive matches** (a new layer kind breaks the build until it is planned), **RAII**
(capture and reservations release on drop), **an exclusive stream borrow** (a capture owns
its stream for as long as it is open, whichever thread runs it), **a scope-lent chain** (`WaveChain<'w, A>` exists only inside
`CaptureCtx::scope`, so a graph, a gap or a wave tensor cannot outlive the wave),
**a mode lattice** (`Eager`, `Chained`, `Capturing`), **host-known values** (`Known<T>`
cannot be built from device data, so composition never needs a sync), and **an
anti-trait** (`NotInGraph` tombstones carry the invariant and the prescriptive fix as a
deprecation note, §12.11).

### 12.1 Module layout (one concern per file)

The split follows the crates the pieces depend on. The wave tier, its epoch and the
region mark live with the bump arena in `candle-nn`. The wave buffers, the MoE dispatch
and `drive_wave` live in `candle-transformers`, so the framework that composes them lives
there too. Only the raw driver calls go in `candle-core`.

```
candle-core/src/cuda_backend/graph/        driver layer; ALL the unsafe lives here
  mod.rs        re-exports
  stream.rs     ComputeStream (the null stream), CaptureStream (created non-blocking stream)
  session.rs    CaptureSession (RAII, holds &mut CaptureStream), begin_capture on CaptureStream only
  exec.rs       GraphExec<'m> (owns the instantiated graph; launches into ComputeStream only)
  beacon.rs     Beacon (mapped host word) and its one-thread store kernel (§12.13)
  error.rs      GraphError
candle-nn/src/kv_cache/chunked/            beside bump_arena.rs and wave_plan.rs
  tier_epoch.rs TierEpoch (monotonic placement counter, §4.3)
  region_mark.rs RegionMark (fence-free cursor rewind to span start, §12.12)
candle-transformers/src/models/graph/      #![forbid(unsafe_code)]
  mod.rs        re-exports; constructors and raw pointers are pub(in crate::models::graph)
  rows.rs       Rows (any count ≥ 1), GraphRows (1 to 4, the capture cap)
  role.rs       Act<'c, A>, Lease<'c, T>, Resid<A>, Table<T>, Weight<'m, W>, GraphAddr, ActDtype (sealed)
  recorder.rs   Recorder<'c, A, M>, Mode, Eager, Chained, Capturing, Live, Operand<M>
  op.rs         Op<A>, Fixed<T>, Scratch, LaunchDims; one file per op family under op/
  guard.rs      AllocGuard (runtime backstop, §12.7)
  banned.rs     NotInGraph anti-trait and the ban! table (§12.11)
  region.rs     Region, RegionKind, RegionKey, RegionLink, NoHost
  host.rs       RegionHost, Reserved
  link.rs       Link<A>, LinkKey, LinkCtx, Concurrent (§12.4)
  moe.rs        MoeProtocol, MoeBuffers, PreMoe, MoeLink (§12.4)
  plan.rs       GraphPlan, LayerPlan, EagerReason
  verdict.rs    WaveVerdict, DecodeWave, Ineligible, decide()
  staging.rs    StagingRing, StagedWave (the Table<T>s a region may read)
  cache.rs      GraphCache<E>, GraphSlot
  ctx.rs        CaptureCtx (scope(), stats())
  chain.rs      WaveChain<'w, A> (run, graph, gap), Gap (§12.12)
  fuse.rs       Fuse<(R1, R2, ..)>, FuseAll<R>, NoHost-only region composition (§12.12)
  known.rs      Known<T>, Variant (§12.13)
  fork.rs       fork/join lanes and event tokens (§12.13)
  pending.rs    PendingWave, event-deferred scope exit (§12.13)
```

Because the framework and the models share a crate, `pub(crate)` would let model code reach
the constructors and `dptr()`. They are `pub(in crate::models::graph)` instead, so a model
module (a sibling of `graph`) sees only the public surface.

Two lifetimes recur and are kept apart: **`'m`** is the borrow of the model's pinned
weights (`Weight<'m, W>`, `GraphExec<'m>`, the long-lived cache), and **`'w`** is one wave's
scope (`WaveChain<'w, A>`, the tier generations). A captured executable outlives the wave
that captured it, so it carries `'m`, never `'w`.

### 12.2 What the compiler enforces

| Mistake | Why it does not compile |
|---|---|
| Capturing the null stream | `CaptureSession::begin` takes `&mut CaptureStream`. `ComputeStream` has no capture method, and `CaptureStream` is constructible only as a created non-null stream. |
| Launching a graph on the wrong stream | `GraphExec::launch` takes `&ComputeStream`. |
| Capturing more than four rows | a capture needs a `GraphRows`, whose constructor is the cap (1 to 4) and whose field is private; only `decide` builds one, inside a `DecodeWave`. `Rows` itself is any count of at least one. |
| Forging or mismatching a graph key | `RegionKey` is built inside `WaveChain::run` from the link's `LinkKey`, the wave's own `GraphRows` and the chain's placement epoch; a model never constructs one. |
| Sync, readback, pageable copy or pool allocation inside a region | `Region::record` is generic over `M: Mode`. Those operations are inherent methods of `Recorder<'c, A, Eager>` only, so inside the generic body the name resolves to the `NotInGraph` tombstone and fails with its note (§12.11). |
| A region allocating at all | a region has no allocator. Each op allocates its own output, from the wave tier, through a constructor private to `models::graph` that returns `Act<'c, A>`, a leased tier tensor with no pool-backed variant. |
| An activation escaping its capture (stale address next replay) | `Act<'c, A>` is invariant in `'c`, and `'c` is higher-ranked in the call that opens the capture, so it cannot be returned or stored. The only things that outlive a region are the `Resid<A>` buffers written through an op. |
| Reading per-wave data from a pool or tier address | per-wave data is a `Table<T>`, which only `StagingRing` can construct and which a region receives through `StagedWave`. |
| Allocating a tensor outside and bringing it into a graph (pool tensor, `Tensor::zeros`, a tensor stored in the region struct or captured by a closure) | an op's operands are typed roles, and in a capturing recorder `Operand<Capturing>` is implemented only by those roles (§12.3, §12.3a). `Tensor` implements `Operand<Eager>` only, and there is no conversion from `Tensor`, so in the generic region body a `Tensor` operand does not type-check. |
| Passing a weight where an activation belongs, or the reverse | roles are distinct types and each op's `In` names the role of every operand. |
| Activation dtype disagreeing between producer and consumer (the BF16/F32 capture-buffer class of bug) | `Act<'c, A>` and `Resid<A>` carry the activation dtype as a type parameter; an op takes and returns the same `A`. A mismatch is a type error, and a model whose dtype is neither BF16 nor F16 is `Ineligible::DtypeMismatch` before any region runs. |
| Baking a weight address that later moves (streaming, reload, eviction) | `Weight<'m, W>` borrows the weight store for `'m` and `GraphExec<'m>` carries the same lifetime, so anything that moves weights needs `&mut` and cannot compile while a graph exists. Streaming models cannot produce a `Weight` at all. |
| A per-wave host value baked as a kernel scalar or grid | op `In` admits scalars only as `Fixed<T>`, built from constants, config or `Rows`; `grid` takes only `Rows`. |
| An MoE region run without its host protocol, or a stateless region run with one | `Link::Host` is an associated type; a link owns its `&mut Host`, which is `NoHost` (zero-sized) for stateless regions, and `WaveChain::run` is the only caller of `before`. |
| Reserving a dispatch slot and never launching or completing it | `RegionHost::before` returns `#[must_use] Reserved<H>`; the only consumer is `WaveChain::run`, which passes it to `after`. |
| A MoE link in a fork lane or beside another MoE link | `fork` accepts only links whose `Host: Concurrent`, and `MoeProtocol` does not implement it (§12.4). |
| Launching on the capture stream, or beginning a second capture on it, while a capture is open | `CaptureSession` holds `&mut CaptureStream` for its whole life and every `Recorder<Capturing>` is derived from it, so no other code can name the stream, on any thread. |
| Adding a layer kind and forgetting to say whether it graphs | `GraphPlan::plan` matches the model's own layer-kind enum exhaustively (the same device the `Arch` trait uses for tensor names). |
| Calling an operation that breaks a hot-path invariant (`to_dtype`, `contiguous`, `cat`, `slice_set`, `zeros`, a readback, a raw address) | the call resolves to a tombstone method of the `NotInGraph` anti-trait, whose `#[deprecated]` note states the invariant, why, and the prescribed alternative; the model-facing modules `#![forbid(deprecated)]`, so it is an error with that text (§12.11). |
| A model touching raw driver handles or addresses | the `unsafe` driver calls exist only in `candle-core::cuda_backend::graph`, `models::graph` is `#![forbid(unsafe_code)]`, and `dptr()` is `pub(in crate::models::graph)`, so a model module cannot name it. |

Each row is a `compile_fail` doctest (§12.8), so the guarantee is itself tested.

### 12.3 Modes and the recorder

```rust
pub trait Mode: sealed::Sealed {}
pub struct Eager;       // launches immediately on the compute stream; anything is allowed
pub struct Chained;     // launches immediately, inside a wave chain; graph-safe operations only (§12.12)
pub struct Capturing;   // records into a graph on the capture stream
impl Mode for Eager {}  impl Mode for Chained {}  impl Mode for Capturing {}
pub trait Live: Mode {} // modes that launch immediately and so may order host effects (an upload)
impl Live for Eager {}  impl Live for Chained {}

/// The only handle a region body receives. `A` is the activation dtype (BF16 or F16).
pub struct Recorder<'c, A: ActDtype, M: Mode> { /* stream, tier generation, rows, AllocGuard */ }

// The one way to launch work, in both modes (§12.10); ops allocate their own outputs:
impl<'c, A: ActDtype, M: Mode> Recorder<'c, A, M> {
    pub fn op<O: Op<A>>(&mut self, op: &O, inp: O::In<'_, 'c>) -> Result<O::Out<'c>>;
    pub fn rows(&self) -> Rows;
}
// Everything that is illegal or meaningless under capture exists only in Eager:
impl<'c, A: ActDtype> Recorder<'c, A, Eager> {
    pub fn synchronize(&self) -> Result<()>;
    pub fn read_back<T>(&self, src: &Table<T>) -> Result<Vec<T>>;
}

/// Which buffers an op may be handed in mode `M`.
pub trait Operand<M: Mode>: sealed::Sealed {}
impl<M: Mode, T: GraphAddr> Operand<M> for T {}   // stable roles: valid in both modes
impl Operand<Eager> for Tensor {}                  // any tensor: eager only
```

`Mode` is sealed, so no third state can be added elsewhere. The operand bound is the
point: an op is written once, generic over `M`, and what it will accept depends on the
mode. Eager, an operand can be any tensor, so the existing hot path and the legacy callers
are untouched. Capturing, an operand must be a stable role. Because a region body is
generic over `M`, it can only use what is valid in every mode, so a `Tensor` operand in a
region fails to compile while the same op called from ordinary eager code accepts it. The
op compiles twice (monomorphised) with no runtime branch.

### 12.3a Address provenance: nothing outside may be baked

A captured graph bakes every device address its kernels touch. So the question is not
"can a region allocate" but "which addresses can a region name". Exactly four roles are
stable across replays (tier memory, persistent buffers, staged tables, pinned weights),
each its own type, and they are the only operands a region can hand to an op. Tier memory
has two spellings: `Act` for the activation dtype and `Lease` for any other element type,
with the same `'c` brand and the same rules:

```rust
pub trait ActDtype: sealed::Sealed {}          // Bf16, F16; the activation dtype of a graph
pub trait GraphAddr: sealed::Sealed { fn dptr(&self) -> DevicePtr; }   // dptr is pub(in crate::models::graph)

pub struct Act<'c, A: ActDtype>   { /* leased wave-tier memory of this region's generation (§4.3) */ }
pub struct Lease<'c, T>           { /* the same tier lease for a non-activation element type (router weights, ids) */ }
pub struct Resid<A: ActDtype>     { /* persistent buffer: the residual stream, the logits */ }
pub struct Table<T>               { /* staged per-wave device table, fixed address, from StagingRing */ }
pub struct Weight<'m, W>          { /* weights proven non-moving for 'm */ }
// each implements GraphAddr; Tensor, CudaStorage and any pool-backed (Owned) type do not
```

- **The rule in one line:** bump memory leased from the wave tier is allowed; anything the
  CUDA allocator handed out (`cudaMalloc`, `cuMemAllocAsync`, the pool, an `Owned`
  tensor) is not. `Act<'c, A>` is the tier's leased tensor with the lifetime of its
  generation, and it has no `Owned` variant, so the pool fallback cannot be expressed.
  The lifetime proves the buffer is valid inside the region; the placement epoch in the
  graph key (§4.3) proves it is at the same address next replay.
- **Buffers that can fall back are never used in a graph.** `wave_empty_ticketed` and
  `wave_from_vec_ticketed` fall back to the pool when a ticket's generation has closed
  (`wave_buffers.rs:334-407`). That is a correct answer outside a capture and a hidden
  cost or a stale baked address inside one. A region has no allocator; each op carves its
  output through a constructor private to `models::graph` that returns a leased tensor or an error, with
  no fallback, and per-wave tables are staged outside the graphs. A kernel that still
  allocates through a ticketed variant is not an op yet, so a region cannot call it.
- **Outside tensors cannot enter.** Op operands are roles, `Tensor` implements
  `Operand<Eager>` only, and there is no `From<Tensor>`. A pool tensor, a `Tensor::zeros`,
  or a tensor held in the region struct or captured by a closure cannot be handed to an
  op from the generic region body.
- **`Resid` and `Table` have no adopt-from-tensor constructor.** `Resid` is created only by
  `Resid::alloc` (a dedicated allocation made once, outside the pool and outside
  capture) and `Table` only by `StagingRing`. Data reaches them by an async copy outside
  the graph, never by aliasing an existing tensor.
- **Weights are the legitimate outside allocation, so they get a proof, not a loophole.**
  `Weight::pin(&'m WeightStore)` returns `Option`, `None` for any store that streams or can
  relocate (Qwen3.8 27B trunk, expert tiers). A model whose dense weights cannot be pinned is
  `Ineligible::Streaming` before any region runs. Routed experts never need a `Weight`:
  the MoE GEMMs read the per-invocation snapshot the bucketize writes (§4.5).
- **Lifetimes make "stable" a borrow, not a convention.** `GraphExec<'m>` and every
  `Weight<'m, _>` carry the same `'m`. Reloading, evicting, growing or moving weights
  requires `&mut WeightStore`, which the borrow checker refuses while any graph exists.
  The cache must be dropped first, which is exactly the invalidation we want.
- **`Act<'c, A>` cannot outlive its capture** (invariant, higher-ranked `'c`), so a tier
  address cannot be smuggled to a later launch or another R's graph.
- **The raw pointer is unreachable.** `dptr()` and every role constructor are
  `pub(in crate::models::graph)`, so a model module cannot extract an address from a
  `Tensor` and rebuild one of the four roles around it.

What this does not stop: a region body that calls a legacy `Tensor` function directly
instead of an op. That function can read an outside tensor, and a read allocates nothing,
so `AllocGuard` does not see it. Two backstops cover it until the legacy kernels are ops
(§12.7): the per-region test replays the graph after the eager run has scribbled and
freed every pool buffer, and compares bit for bit, so a baked outside address reads
garbage and fails; and the capture test fails any region that still calls an unmigrated
function. The end state is that region code cannot name a `Tensor` at all.

### 12.4 Regions and the host protocol

```rust
pub trait Region<A: ActDtype> {
    const KIND: RegionKind;                 // Layer | MixerFfn | Ffn | Head; MoE is a MoeLink around one of them
    type Host: RegionHost;                  // NoHost for stateless regions
    type Io<'a>;                            // the Resid<A> buffers it reads and writes

    fn record<'c, M: Mode>(&self, rec: &mut Recorder<'c, A, M>, staged: &StagedWave<'_>,
                           io: &mut Self::Io<'_>) -> Result<()>;
}

pub trait RegionHost {
    fn before(&mut self, ctx: &LinkCtx) -> Result<Reserved<'_, Self>>;  // begin_invocation, hold_for_ring, enqueue, send both messages; ctx = MoE row, staged seq_base
    fn after(&mut self, r: Reserved<'_, Self>) -> Result<()>;   // cuStreamQuery
}
pub struct NoHost;   // before/after are no-ops; zero-sized
```

- `WaveChain::run` first captures and instantiates the slot if it is empty (`record` on a
  `Capturing` recorder, no host protocol), then runs `before`, then either launches the
  slot's executable or, for a refused slot, runs `record` on a `Chained` recorder, then
  `after`. A host protocol has exactly one implementation, and graph and chained-eager
  differ only in how the launches are issued.
- A capture or instantiate failure happens before `before`, so nothing is reserved and the
  slot is `Refused`; a launch failure after `before` falls back to a `Chained` run inside
  the same `before`/`after` window. `before` is never re-run, so nothing is double-sent or
  stranded (§4.6), and the type makes that the only possible order: `Reserved` is consumed
  by `after` exactly once.
- The MoE dispatch's `Dispatch::forward` is split into `before` (host reservation and
  sends), the kernel launches (`record`), and `after` (the query). The kernels and the
  device-derived sequence of §4.5 are unchanged. The split lives in one component,
  `MoeProtocol`, below.

**The link.** The chain's unit is a `Link`: a region body bound to its place in the model
and to its host protocol. A link is a value, so a model's layer is a list of links and a
cut in the graph is one more entry, not new control flow:

```rust
pub trait Link<A: ActDtype> {
    type Host: RegionHost;
    type Io<'a>;
    fn key(&self) -> LinkKey;                     // layer or span, RegionKind, variant; the chain adds R and the epoch
    fn host(&mut self) -> &mut Self::Host;
    fn record<'c, M: Mode>(&self, rec: &mut Recorder<'c, A, M>, staged: &StagedWave<'_>,
                           io: &mut Self::Io<'_>) -> Result<()>;
}
/// A `Region` bound to a layer index and a host: the adapter every dense link uses.
pub struct RegionLink<'h, R: Region<A>, A: ActDtype> { layer: usize, region: R, host: &'h mut R::Host, _a: PhantomData<A> }
```

`chain.run(link, staged, io)` does everything `chain.graph` did (slot lookup, capture
first, `before`, launch or `Chained` run, `after`, the node-count check); `chain.graph` is
`run(RegionLink::new(..))`. The launch count the check compares against is not declared
on the link: the recorder sums each op's `Op::LAUNCHES` while `record` runs, so the
expectation cannot drift from the body. A model's plan returns its layers as links: a dense
layer is a `RegionLink`, and an MoE layer is one `MoeLink` (Host = `MoeProtocol`), which
every MoE model shares.

**The MoE protocol is one component.** `MoeProtocol` owns what the dispatch owns today:
the summary ring, the abort word, the `ReclaimClock`, the pass state, the two senders, and
the shared device buffers (`snap`, `remote`, `counters`, `remote_dst`, `scratch`, the
bucketize workspace) as one `MoeBuffers` value of `Fixed` buffers whose lifetime bounds
every slot that bakes them. It implements `RegionHost`:

- `before(ctx)`: `begin_invocation`, `hold_for_ring`, `clock.enqueue`, both sends. `ctx`
  carries the link's MoE row and its ordinal in the wave's sweep. `seq` is the dispatch's
  global monotonic counter, so `seq_base` is `f.seq` read at staging and the ordinal `i`
  baked into the graph (§4.5) is the layer's position in the sweep; it equals the row
  only when the wave runs every MoE row once from 0. `before` checks
  `seq == staged.seq_base + i`: a
  model that invokes a MoE layer twice or out of order in a wave fails the wave loudly
  instead of desynchronising the ring.
- `after`: the `cuStreamQuery` flush.
- It is **not** `Concurrent`. `fork` (§12.13) accepts only links whose `Host: Concurrent`
  (implemented by `NoHost` and by any host with no shared device state), so a MoE link
  cannot sit in a fork lane or beside another MoE link, which would share `snap`,
  `counters` and `scratch`. The rule is a trait bound, not a runtime check.

**The MoE link is one graph per layer.** The model supplies the part of the layer before
the routed experts, and `MoeLink` appends the routed launches to the same recording:

```rust
pub trait PreMoe<A: ActDtype> {
    /// Mixer or attention half, FFN norm, router, and the shared expert if the model has
    /// one; returns what the routed experts read. Adds into the residual it is handed.
    fn record<'c, M: Mode>(&self, rec: &mut Recorder<'c, A, M>, staged: &StagedWave<'_>,
                           x: &mut Resid<A>) -> Result<RouterOut<'c, A>>;
}
pub struct RouterOut<'c, A: ActDtype> { pub h: Act<'c, A>, pub weights: Lease<'c, f32>, pub ids: Lease<'c, u32> }
pub struct MoeLink<'h, P: PreMoe<A>, A: ActDtype> { layer: usize, pre: P, moe: &'h mut MoeProtocol, _a: PhantomData<A> }
```

`MoeLink::record` runs `pre.record`, then bucketize, gather, gate, up, silu, down and the
scatter into the residual. The router output and the normed hidden state are `Act`s of the
one link, so nothing but the residual crosses into the next link and the region mark
cannot rewind them early. The protocol needs the CPU *between* layers, not inside one: the
messages are sent before the graph launches (receipt assumes nothing, §4.5) and the query
follows it, so splitting the attention half into its own graph would add a launch per
layer and buy nothing, since it has no `NoHost` neighbour to fuse with. `StagedWave` gains
`seq_base`, staged outside the chain with the other per-wave values. A model that needs a
host step inside a MoE layer cuts it into more links; the `Routed` send stays in
`MoeProtocol::before` of the link that contains the bucketize, and router outputs that
cross that cut go through `MoeBuffers`' `Fixed` buffers, never an `Act`.

**The plan.** The per-model plan is a trait with an exhaustive match, so a new layer kind
cannot be added without deciding. It names the region kinds of a layer; the model's link
constructors turn them into links (a MoE layer's kind becomes one `MoeLink` around the
model's `PreMoe`):

```rust
pub enum LayerPlan { Regions(&'static [RegionKind]), Eager(EagerReason) }
pub trait GraphPlan { type LayerKind: Copy; fn plan(kind: Self::LayerKind) -> LayerPlan; }
// qwen35:  match kind { LayerKind::DeltaNet => Regions(&[MixerFfn]), LayerKind::Attention => Regions(&[Layer]) }
// qwen4exp: match kind { DeltaNet => Regions(&[MixerFfn]), Attention => Eager(EagerReason::QsaSelection) }
```

### 12.5 The context and the cache

```rust
impl CaptureCtx {
    // opens the wave chain; the scope, its links and `graph` / `gap` are specified in §12.12
    pub fn scope<A: ActDtype, T>(&mut self, wave: DecodeWave,
        f: impl for<'w> FnOnce(&mut WaveChain<'w, A>) -> Result<T>) -> Result<T>;
    pub fn stats(&self) -> GraphStats;     // captured, replayed, chained above the cap, eager by reason, failures, chain breaks
}
pub struct GraphCache<E> { /* HashMap<RegionKey, E> + recency, bounded */ }
impl<E> GraphCache<E> {
    pub fn get(&mut self, k: &RegionKey) -> Option<&E>;
    pub fn insert(&mut self, k: RegionKey, e: E) -> Option<E>;     // evicts the least recent R
    pub fn invalidate_where(&mut self, pred: impl Fn(&RegionKey) -> bool);
}
pub enum WaveVerdict { Eager(Ineligible), Decode(DecodeWave) }      // Eager: the mixed forward
pub struct DecodeWave { rows: Rows, capture: Option<GraphRows> }    // private fields; capture is Some iff rows ≤ 4
pub fn decide(shape: &WaveShape) -> WaveVerdict;                    // pure
```

`CaptureCtx` owns a long-lived `GraphCache<GraphSlot>` whose slots (`Empty`, `Ready`,
`Refused`) the chain fills lazily; the unit tests use `GraphCache<u32>`. A `DecodeWave`
whose `capture` is `None` never touches the cache: its chain runs every link `Chained`. Eviction is per
`R`, and `invalidate_where` recaptures the MoE graphs when the workspace grows. The
`RegionKey` also carries the placement epoch and the variant (§4.1).

### 12.6 Using it

A dense model: one region type, run per layer.

```rust
struct DenseLayer<'m, A> { w: &'m LayerWeights<'m>, _a: PhantomData<A> }   // w: Weight<'m, _> fields
impl<A: ActDtype> Region<A> for DenseLayer<'_, A> {
    const KIND: RegionKind = RegionKind::Layer;
    type Host = NoHost;
    type Io<'a> = &'a mut Resid<A>;                       // the residual stream
    fn record<'c, M: Mode>(&self, rec: &mut Recorder<'c, A, M>, staged: &StagedWave<'_>, x: &mut Self::Io<'_>) -> Result<()> {
        let h   = rec.op(&RmsNorm,    (&**x, &self.w.attn_norm))?;           // Act<'c, A>, allocated by the op
        let qkv = rec.op(&QMatmul,    (&h, &self.w.qkv))?;
        let a   = rec.op(&PagedDecode, (&qkv, &staged.slot_headers))?;
        rec.op(&OProjResidual,        (&a, &self.w.o, &mut **x))?;           // adds into the residual
        let h   = rec.op(&RmsNorm,    (&**x, &self.w.ffn_norm))?;
        rec.op(&FfnResidual,          (&h, &self.w.ffn, &mut **x))
    }
}
ctx.scope::<Bf16, _>(wave, |chain| {
    for (layer, w) in self.layers.iter().enumerate() {
        chain.run(&mut RegionLink::new(layer, DenseLayer::new(w), &mut NoHost), &staged, &mut x)?;
    }
    Ok(())
})?;
```

The body reads as the layer and contains no allocation, no dtype, no stream and no
address. Swapping two operands, passing a weight where an activation belongs, or using an
F16 residual with a BF16 op is a type error.

An MoE layer is one link owning the shared `MoeProtocol` (§12.4). The model writes only its
`PreMoe` (attention or DeltaNet mixer, norm, router, shared expert); every MoE model writes
the same line:

```rust
chain.run(&mut MoeLink::new(layer, DenseAttnPre::new(w), &mut self.moe), &staged, &mut x)?;   // Host = MoeProtocol
```

A model's section that is not a region (Flash-Next's QSA attention half) is a `gap` in the
chain (§12.12) once its kernels are ops, and a chain break until then.

### 12.7 What the compiler cannot reach, and the backstop

Legacy candle functions that a region body might still call (a `Tensor::zeros`, a
`to_vec`, a direct `device.cuda_stream().synchronize()`) are not ops, so their illegality
is invisible to the type system. The ones that violate a hot-path invariant are named and
explained by the anti-trait (§12.11) wherever they are reachable through a role or the
recorder; a free call such as `Tensor::zeros(..)` is stopped by `Operand<Capturing>`'s
diagnostic when its result is used. Three layers cover the rest:

1. **`AllocGuard` at capture.** `CaptureSession::finish` fails with
   `GraphError::ForeignAllocation { what, site }` if any driver-pool fallback, default
   allocation, pageable copy or synchronisation occurred during the capture, and the
   driver itself rejects illegal calls on the capturing thread
   (`cudaErrorStreamCaptureUnsupported`, §4.2).
   The slot becomes `Refused`, the region runs on a `Chained` recorder and the failure is
   reported (§4.6). It also fails with `GraphError::MissingLaunch { op }` when the
   captured node count differs from the sum of the ops' `Op::LAUNCHES`: a launcher
   that skipped its launch on a failed guard leaves no node, and a replay would otherwise
   run without it, every wave, silently.
2. **A CI test that captures every region of every model once** (§12.8), so a violation is
   a failed test, not a production fallback.
3. **Migration closes the gap.** Each kernel implemented as an `Op` removes a class of
   runtime-only mistakes. The end state is that no region body calls anything that is not
   an op; until then, items 1 and 2 are the net, and the count of unmigrated functions a
   region still calls is reported by its capture test.

Not type-checkable at all: whether a model's weight pointers move (`Streaming`), and
whether a kernel's grid really is a function of `Rows` alone when its implementation
reads host state internally (an op's `grid` is typed, but a wrapper that ignores it is
not). Both are covered by `decide`'s `Ineligible` checks and by the per-op conformance
test (§12.10).

### 12.8 Tests (written with the code)

Compile-time: one `compile_fail` doctest per row of §12.2: capturing a `ComputeStream`;
`GraphRows` from 5; calling `synchronize` / `read_back` inside a generic `record`; returning an
`Act` from a capture; building a `Resid` or `Table` from a `Tensor`; passing a `Tensor`
operand to an op from a generic region body (and the same call compiling from eager code);
passing a weight where an activation belongs; an F16 `Resid` into a BF16 op; a per-wave
host value as a `Fixed` scalar; running an MoE link with `NoHost`; a `MoeLink` inside a `fork` lane (`MoeProtocol` is not
`Concurrent`); using the `CaptureStream` while a `CaptureSession` borrows it; a `GraphPlan` match missing a variant. Pure, raw
expected values, no GPU: `GraphRows::new` accepts 1 to 4 and rejects 0 and 5, and `Rows::new`
rejects only 0; `decide` for each `Ineligible` variant against a hand-built `WaveShape`, and
for decode-only shapes of 4 and 5 rows (`capture` is `Some` and `None`); a chain over a
`DecodeWave` without `GraphRows` runs every link `Chained` and touches no slot; `GraphCache` insert, hit,
least-recently-used eviction by `R`, and `invalidate_where`; the tier epoch bumps on every
re-place and plan raise, so a graph keyed on a stale epoch is not replayed; each op's
`grid` and `scratch` for `Rows` 1 to 4, 16 and 64; the cursor starts at span start in each region and
rejects an over-allocation. GPU, `--features cuda`: a captured region's output equals the
eager output bit for bit; an injected foreign allocation fails `finish`; an op whose launcher skips its launch fails
capture with `MissingLaunch { op }` and never reaches a replay; the recorder's `LAUNCHES`
sum equals the captured node count for every op; `MoeProtocol::before` rejects a `seq`
that differs from `seq_base + ordinal` (a layer invoked twice, or out of order); the
bucketize kernel derives the same summary slot and sequence from `seq_base + ordinal` as the
host-computed ticket; a `Link`'s capture runs before its `before` (event order against a
fake backend, and a capture failure leaves `PassState.reserved` untouched); a dropped session
leaves the stream usable; a captured graph launched on the null stream orders correctly
against surrounding null-stream work (the §4.2 assumption, stated as a test); and, per
model, every region captures once with zero foreign allocations. The anti-trait (§12.11)
has one UI test per banned name: the real call shape inside a generic region body must fail
with the entry's own note (pinned stderr), and a pure test checks every table entry has an
invariant, a reason and an `INSTEAD:` clause. A launch-count test records each region at
`R = 1`, `R = 4` (captured) and `R = 64` (`Chained`, counted by the recorder) and requires
the same op sequence, which is how invariant 5 (a per-row loop) is caught at both ends of
the range.

### 12.9 Vendor-neutral naming and the backend seam

Only the backend module names a vendor. Everything model-facing speaks in roles: a
**decode graph** (a recorded, replayable chain of launches), a **region**, a **recorder**,
a **capture stream**, a **compute stream**, a **graph executable**.

```rust
// candle-core, sealed; CUDA is the only implementation until a second backend exists.
pub trait GraphBackend: sealed::Sealed {
    type Compute;                        // the stream work normally runs on
    type Capture;                        // the stream a graph is recorded on
    type Exec;                           // the instantiated, replayable graph
    fn begin(c: &mut Self::Capture) -> Result<()>;
    fn end(c: &mut Self::Capture) -> Result<Self::Exec>;
    fn launch(e: &Self::Exec, on: &Self::Compute) -> Result<()>;
}
```

`CaptureCtx`, `GraphCache`, `WaveChain` and `Recorder` are generic over it, so a ROCm backend (graphs
are the same shape under HIP) is a new implementation and not a rewrite. Metal indirect
command buffers and Vulkan or D3D12 command buffers have no stream capture; their
`Capture` is an explicit command recorder, which `Recorder<'c, A, Capturing>` already
models, so they fit the same seam without changing a region. Nothing about those
backends is built or measured here, and the trait is not added until a second backend is
real: until then the names above are the whole abstraction, and the module that holds the
`unsafe` driver calls stays `cuda_backend::graph`.
### 12.10 Operations: typed, closed, and owning their outputs

Each kernel a region can launch is one typed op, launched through `rec.op(&op, inputs)`.
A region's `record` is a composition of ops and nothing else, so the set of things that
can be captured is closed and explicit, and the author never allocates, names a dtype, a
stream or an address.

```rust
pub trait Op<A: ActDtype> {
    type In<'a, 'c>;                  // a tuple of typed roles: &Act<'c,A>, &Weight<'m,W>, &Table<T>, &mut Resid<A>, Fixed<T>
    type Out<'c>;                     // Act<'c, A>, or () for an op that writes a Resid in place
    const NAME: &'static str;         // counters and the capture report
    const LAUNCHES: usize;            // kernel nodes one launch adds; the recorder sums them per link

    fn scratch(&self, rows: Rows) -> Scratch;       // workspace pre-grown before the first capture
    fn grid(&self, rows: Rows) -> LaunchDims;       // pure function of rows and constants
    fn launch<'c, M: Mode>(&self, rec: &mut Recorder<'c, A, M>,
                           inp: Self::In<'_, 'c>) -> Result<Self::Out<'c>>;
}
```

An op's signature is its documentation and its contract. For example `RmsNorm` takes an
activation or residual and a weight and returns an activation of the same `A`;
`PagedDecode` takes an activation, a slot-header `Table` and returns an activation;
`OProjResidual` takes an activation and a weight and adds into a `&mut Resid<A>`. The op
allocates its output through the recorder's tier constructor (private to `models::graph`), so there is no
allocation in a region and the fallback arms of §12.3a are unreachable from it.

The call surface also carries these, which the recorder alone did not:

| Invariant | How the op trait carries it |
|---|---|
| Frozen scalars and grid (§3.2) | `grid` takes only `Rows`, and `In` admits scalars only as `Fixed<T>` (built from constants, config or `Rows`). A per-wave host value has no way in. |
| 1: no `to_dtype` | there is no conversion op in the set, and an op takes and returns the same `A`, so a cast cannot be written inside a region and a producer cannot hand a consumer the wrong dtype. |
| 1b: validate, do not convert | activation dtype is a type; weight dtype and layout are checked once at capture with `expect_dtype`. Capture happens once per key and replay costs nothing, so this check is free on the hot path. |
| 2: no allocate-and-copy for layout | `contiguous`, `cat`, `slice_set` are not ops and are banned by name (§12.11). A consumer that needs a layout becomes an op that reads it. |
| 6: no zeroing | `Out` declares `Uninit` or `Zeroed`; `Zeroed` exists only on accumulator ops, so a blanket `zeros` is not expressible. |
| Pre-grown workspaces (§4.3) | `scratch` is collected by the plan step before the chain opens, so MoE bucketize, split-K and the split-KV partial pool are sized for the wave's `R` (never below 4) in one place, not discovered as capture failures or as an allocation inside a scope. |
| Pre-warmed kernels | the op owns its kernel handle, so loading it before capture is part of construction. |

**Where types stop.** Shapes and strides are not typed. Rows are a dynamic value held as
`Rows`, of at most four when a capture holds them as `GraphRows`; everything else is a
model constant, checked at capture. The
type-level budget goes to what is structural: role, mode, frozen scalars and activation
dtype.

**One op, both modes.** An op body is written once and is generic over `M`. Eager, its
operands may be any tensor (`Operand<Eager>`), so ordinary eager callers use the same op
the region does; capturing, only stable roles are accepted (§12.3). There is no second
implementation to keep in step.

Also gained: one generic conformance test run for every op (captured output equals eager
output bit for bit, with pool memory scribbled after the eager run), per-op names in the
capture report and the launch counters (the launch count per region falls out of the op
list, which is the measurement §8 asks for), and a visible migration list: a kernel not yet
an op is one a region cannot call.

Costs, stated: one small type per kernel wrapper (dozens), so keep the trait to the four
items above; regions are generic over `A`, so each compiles twice (BF16 and F16), which is
acceptable because graphs are decode-only and the production activation dtype is chosen
once per model; dispatch is by generics, not `dyn`, so there is no runtime cost; cuBLAS
and other library calls need an op that owns the handle and workspace; and ops with
data-dependent grids still need the kernel change of §3.2 before they can implement
`grid`. This replaces the earlier plan to migrate wrappers onto the recorder: the
migration unit is "implement `Op`".

### 12.11 Banned operations: the anti-trait

Because ops are a closed set (§12.10), an operation that violates a hot-path invariant is
already unavailable in a region. Unavailable is not the same as explained: a bare
"no method `contiguous` found" tells the author nothing about why, or what to write. The
anti-trait makes each ban a named, documented thing whose error message is the prescription.

Rust has no stable negative impls, so the anti-trait is a trait whose methods are the
banned operations, each marked `#[deprecated]` with a fixed-shape note, and the
model-facing modules are `#![forbid(deprecated)]` (`forbid`, not `deny`, so a region file
cannot add an `#[allow(deprecated)]`). Calling one is a compile error that prints the note.

```rust
// models::graph::banned, re-exported from models::graph::prelude
pub trait NotInGraph {
    #[deprecated(note = "BANNED IN GRAPHS (invariant 2: no allocate-and-copy): contiguous() \
        allocates and copies the whole tensor on every replay. INSTEAD: make the consumer read \
        the layout that exists (offset + stride, or a descriptor table, invariant 2b), or have \
        the producing op emit that layout.")]
    fn contiguous(&self) -> Banned { Banned(()) }
    // ... one method per row of the table below; signatures mirror the real Tensor method
}
impl<'c, A: ActDtype, M: Mode> NotInGraph for Recorder<'c, A, M> {}
impl<'c, A: ActDtype>          NotInGraph for Act<'c, A>        {}
impl<'c, T>                    NotInGraph for Lease<'c, T>      {}
impl<A: ActDtype>              NotInGraph for Resid<A>          {}
impl<'m, W>                    NotInGraph for Weight<'m, W>     {}
impl<T>                        NotInGraph for Table<T>          {}

#[must_use] pub struct Banned(());   // implements no Operand and no role: the result goes nowhere
```

How it composes with the rest:

- **Inherent beats trait.** Eager code calls the real methods, which are inherent on
  `Recorder<'c, A, Eager>` and on `Tensor`. A generic region body (`M: Mode`) sees only the
  trait, so the same name resolves to the tombstone there. Nothing real is shadowed.
- **The tombstone is not a stub.** Its only contract is to carry the message, and its
  result is a `Banned` value that no op accepts, so even with the lint overridden nothing
  built from it can be launched. Signatures copy the real method's so the call resolves and
  the lint fires, instead of an arity error that hides the note.
- **Foreign tensors.** A free call such as `Tensor::zeros(..)` is not a method on a role, so
  it cannot be a tombstone; its result is stopped at `rec.op` by a diagnostic on
  `Operand`:
  ```rust
  #[diagnostic::on_unimplemented(
      message = "`{Self}` was allocated outside the graph and cannot be an operand in a region",
      label   = "not a graph role (Act, Resid, Table or Weight)",
      note    = "BANNED IN GRAPHS (§4.3, invariant 7): its address is baked into the graph and the \
                 pool may move or reuse it. INSTEAD: let the op allocate its own output, or stage \
                 per-wave data outside the region as a Table through StagingRing.")]
  pub trait Operand<M: Mode> { /* §12.3 */ }
  ```
- **One source.** Each ban is one `ban!(name, invariant, "note")` entry. The macro emits the
  tombstone, an entry in `BANS: &[Ban]`, and a row for the capture report; the table
  below is that list.

**What is banned and what to write instead.** Every note has the shape
`BANNED IN GRAPHS (invariant N: <rule>): <why>. INSTEAD: <fix>`.

| Banned (the call a region author might write) | Invariant | Why it hurts | Instead |
|---|---|---|---|
| `to_dtype`, `to_dtype_into`, any cast | 1, 1b | A full-tensor memory pass on every replay, or dead code that hides a producer handing over the wrong type. | Pick the op whose output is the consumer's dtype. `A` is part of every role's type, so there is nothing to convert. A producer that emits the wrong dtype is fixed in the producer (a kernel template parameter). Validate weights once at capture with `expect_dtype`. |
| `contiguous`, `force_contiguous` | 2 | Allocate-and-copy to materialise a layout the consumer should read as it is. | Teach the consuming op to read offset and stride, or a descriptor table (2b); or have the producing op write the layout directly. |
| `Tensor::cat`, `stack`, `slice_set` | 2, 2b | One copy launch per argument, plus an allocation, to pack rows a kernel could read in place. | Producers write into their slice of one `Act` or `Resid` (the op takes an output offset as a `Fixed`), or the consumer takes a descriptor table of `{ptr, offset, stride, len}` per row. |
| `to_owned_tensor`, `copy`, `clone` of a tier tensor | 2 | Allocates and copies, and the copy is a pool tensor that is not a graph role. | Do not own anything inside a region. Take ownership after the replay, outside the region, from the `Resid` the graph wrote. |
| `to_vec*`, `to_scalar`, `to_device(Cpu)`, `read_back` | 3 | A GPU-to-CPU transfer and an implicit wait. The only readback the decode path has is the embedding ids, read before the chain opens; MoE routing is not read back at all (§2, §4.5). | Keep it on the device as an op. Host work that must sit between launches is a `RegionHost` protocol at the link boundary or a gap, and neither reads device data. To read logits, do it after the scope exits. |
| `synchronize`, `stream.synchronize()`, `device.synchronize()`, event waits | 3, 7 | Illegal in capture, and a host wait drains the pipeline the graph exists to keep full. | Ordering is stream order. Anything that needs the host's result is a region boundary (the MoE protocol), not a wait inside a region. |
| host sort, dedup, union, remap, or building a descriptor table inside a region | 4 | Host compute over per-token data, baked at capture so it is stale on replay. | Build the table at staging (§4.4) into a `Table<T>` through `StagingRing`, or write a kernel op that builds it on the device. |
| `Tensor::from_vec`, `from_slice`, `Tensor::new`, `wave_from_vec*`, pageable `cuMemcpyHtoD` | 4, 7 | A pageable host-to-device copy and a baked source address. | Per-wave host data is staged outside the region as a `Table<T>` and read through `StagedWave`. Constants are `Fixed<T>`. |
| `zeros`, `zeros_like`, `ones`, `full`, a `memset` on a buffer a kernel overwrites | 6 | A second full-width memset over bytes the kernel is about to write. | Ops allocate `Uninit` outputs. If the buffer is genuinely read before written (atomic accumulator, scatter base, ragged padding), that op declares `Zeroed` output. |
| `Tensor::empty`, `alloc_uninit` on a device, `wave_empty*`, `wave_empty_ticketed`, `cudaMalloc`, `cuMemAllocAsync`, any allocator that can fall back to the pool | 7, §4.3 | A graph memory node, or a pool address that may move or be reused: the allocation fallback hides bottlenecks and breaks replay. | A region has no allocator. The op allocates its own output through the recorder's tier constructor, which returns a leased tier tensor or an error. |
| `as_ptr`, `device_ptr`, `cu_device_ptr`, a cached per-expert slot address | 7 | A raw address survives a boundary move and silently names another tenant's data. | Roles expose no pointer. Pinned weights are `Weight<'m, W>`; routed experts reach kernels only through the per-invocation snapshot the bucketize writes on the device (§4.5). |
| lazy kernel or module load, library handle creation, workspace growth | 7, §4.3 | An implicit synchronisation and a driver allocation inside a scope. | Construct the op before the first chain: it owns its kernel handle and reports `scratch(rows)`, which is grown for the wave's `R` before the chain opens. |
| `Tensor::assert`, `check_now`, `SlotIntegrity` checks (`tensor-assert` builds) | harness | The fencing checks suppress the races they hunt, and a baked stats slot runs on every replay. | Assert on the `Resid` outputs eagerly after the replay, outside the region. |

**What a type cannot ban.**

- **Per-row loops (invariant 5).** A `for row in 0..rec.rows()` around legal ops is legal
  Rust. The launch-count test records every region at `R = 1`, `4` and `64` and requires the
  same op sequence, so a loop that scales the launch count fails in CI.
- **A free `Tensor::zeros(..)` that is never used as an operand.** Dead allocation inside a
  region is caught by `AllocGuard` at capture (§12.7), not by the type system. A blanket
  `clippy::disallowed_methods` was considered and rejected: it is crate-wide, and the same
  calls are correct everywhere outside the regions.
- **A tombstone is only reached through a role, the recorder, or an `Operand`.** A region
  that never touches a role cannot do anything the graph records, so this is the whole
  surface.

Cost: one tombstone per banned name (about a dozen), written once in `banned.rs`; no
runtime cost, since tombstones are never called in a correct build. Requires
`#[diagnostic::on_unimplemented]`, stable since Rust 1.78; confirm the workspace's
`rust-version` before relying on it.

### 12.12 The wave chain: scope, links, lazy graphs

A forward is one **chain**: an ordered run of **links**, each either a **graph link** (a
region replayed from a captured executable) or a **gap** (host logic, with optional
immediately launched ops). `drive_wave` opens the chain once, after `decide()` returned a
`DecodeWave`, and everything the chain needs lives and dies with that scope. When the wave
has no `GraphRows` (more than four rows) a graph link is issued `Chained`, exactly as a
refused slot is, so the chain, its host protocol, its memory and its bans are the same at
every row count.

```rust
impl CaptureCtx {
    /// `'w` is higher-ranked, like std::thread::scope: nothing leased inside can leave.
    pub fn scope<A: ActDtype, T>(&mut self, wave: DecodeWave,
        f: impl for<'w> FnOnce(&mut WaveChain<'w, A>) -> Result<T>) -> Result<T>;
}

impl<'w, A: ActDtype> WaveChain<'w, A> {
    /// Graph link (§12.4): capture first if the slot is empty, host protocol `before`,
    /// replay, `after`; a refused slot runs `record` on a `Chained` recorder inside the same window.
    pub fn run<L: Link<A>>(&mut self, link: &mut L, staged: &StagedWave<'_>,
                           io: &mut L::Io<'_>) -> Result<()>;
    /// `run` of a `RegionLink`.
    pub fn graph<R: Region<A>>(&mut self, layer: usize, region: &R, host: &mut R::Host,
                               staged: &StagedWave<'_>, io: &mut R::Io<'_>) -> Result<()>;
    /// Gap: arbitrary host Rust plus graph-safe ops launched immediately, in stream order.
    pub fn gap<T>(&mut self, name: &'static str,
                  f: impl for<'g> FnOnce(&mut Gap<'g, 'w, A>) -> Result<T>) -> Result<T>;
}
pub struct Gap<'g, 'w, A: ActDtype> { /* rec: Recorder<'g, A, Chained>, rows, host-only helpers */ }
```

**What the scope owns.** The three phase generations (`Attention`, `Ffn` and `Forward`,
which `begin_wave` allows to coexist) opened before the first link and dropped once, at
scope exit; a borrow of the long-lived `GraphCache<GraphSlot<'m>>` whose slots it fills
lazily and which outlives the scope; the placement epoch it read at
open; the compute stream and the capture stream; and the link log for the report. Scope exit
is the only fence of the wave: the outermost generation drop synchronises the stream
(`bump_arena.rs:660-672`) before the cursor resets, and the caller reads the logits
`Resid` after it. No link contains a wait, so the CPU runs ahead of the GPU the whole way
through, which is what lets the host protocol and the graph launches overlap.

**Lazy graphs.** Each `(layer or span, kind, R, epoch, variant)` key has a slot with three states:

```
Empty --first use--> capture on the capture stream, instantiate --> Ready(GraphExec<'m>)
Empty --capture or instantiate error--> Refused(error, epoch)   // reported, run Chained until the epoch moves
Ready --launch error--> Refused(error, epoch)                   // this wave's link runs Chained inside the same window
```

A `Ready` slot launches on the compute stream. A `Refused` slot runs the region's `record`
on a `Chained` recorder, so a failed capture keeps the chain unbroken and keeps every ban.
Nothing is built before the first decode wave of at most four rows, and the first wave per `(R, epoch)` pays
the captures; later waves only replay. The key includes the epoch, so a re-placed tier
retires the old slots instead of replaying them (§4.3). Whether instantiating a graph while
earlier graphs are still queued stalls the host, or allocates in a way that serialises, is
not known and is the first thing the prototype measures.

**Memory inside the chain.** Each region and each gap takes a **region mark** on its
phase's generation and rewinds to it when it ends: no fence, no free, and the next link's
allocations start at span start again, so the addresses are the deterministic ones the
slots were captured with. The rewind is sound for the reason the existing guard comment
gives for guest arenas (the next writer of a released range is ordered behind its last
reader on the one stream) and unsound on the wave path in general because tensors cross
phases there (`bump_arena.rs:608-616`). In a chain nothing crosses a link except `Resid` and
`Table`, and the `'c` brand on `Act` (§12.2) makes that a type fact, so the rewind is
permitted only through the chain. This is a new arena primitive (the mark), not an existing
call. Peak tier use is one link's worth per phase, so `plan_wave_transient` sizes for the
widest link, not the sum of layers.

**Modes.** A gap is concrete, not generic, so it can run any Rust; what it may *launch* is
set by its mode:

| Mode | Launches | May launch | Used by |
|---|---|---|---|
| `Capturing` | recorded | graph-safe ops, stable-role operands only | a graph slot's first sight |
| `Chained` | immediately | the same ops and operands; plus ops that order a host effect (`Live`: a pinned upload into a `Table`) | gaps, a refused slot, and every link of a wave above four rows |
| `Eager` | immediately | anything, including legacy `Tensor` calls, syncs and readbacks | everything outside a chain |

`Chained` has no `synchronize`, no `read_back` and no `Tensor` operand, so the
anti-trait of §12.11 applies inside gaps exactly as in regions. A gap cannot call
`chain.run` or `chain.graph` (it holds the `Gap`, not the chain) and cannot nest another scope, so links
cannot interleave. Anything that is not yet an op cannot be launched in a gap either: a
section of a model that needs legacy calls (Flash-Next QSA selection until it is migrated)
ends the chain with a fence and a new chain starts after it. That is a **chain break**,
counted in `GraphStats` with its reason, so the cost of each unmigrated section is a number
in the report.

**Coalescing: highly optimised chains from the plan.** The plan decides the links, not the
model's loop. Adjacent regions with `Host = NoHost` fuse into one graph:

```rust
pub struct Fuse<T>(T);
impl<A: ActDtype, R1: Region<A, Host = NoHost>, R2: Region<A, Host = NoHost>> Region<A> for Fuse<(R1, R2)> { .. }
// small tuples for a heterogeneous span; a region with a host protocol does not satisfy the bound
pub struct FuseAll<'s, R>(&'s [R]);
impl<A: ActDtype, R: Region<A, Host = NoHost>> Region<A> for FuseAll<'_, R> { .. }
// a homogeneous or enum-typed run of layers: a dense trunk, or a hybrid's layers through a
// per-model enum region whose `record` matches the layer kind
```

`Fuse` takes a region mark between members, so a fused span needs one member's peak per
phase, not the sum; nothing but the residual crosses a member boundary, for the same reason
it cannot cross a link. A dense model's whole trunk and head is one link (the Strata
whole-window graph, per `R`), keyed by its span. An MoE model has one graph link per
layer, with `MoeProtocol` around it, because the dispatch protocol needs the CPU between
layers (§4.5), so its cut points are exactly the host protocols and nowhere else. A `MoeLink` is never fused with another MoE
link: a worker's 1.5 s spin limit is per kernel on WDDM, and the MoE buffers are shared. A hot spot that needs more than the layer split (for
example, an attention half followed by the next layer's norm) is a new fused region type,
written once and keyed like any other. `GraphPlan` returns the link list from the layer
kinds with an exhaustive match, as in §12.4, and no flag chooses the granularity.

**What the chain adds to §12.2.**

| Mistake | Why it does not compile |
|---|---|
| Using the chain, a `Gap` or an `Act` after the scope | `scope`'s `'w` is higher-ranked; none of them can be returned or stored. (A `GraphExec<'m>` deliberately outlives the scope in the cache; it holds no tier address that the epoch does not key.) |
| Nesting a `graph` or a scope inside a gap | `Gap` holds no handle to the chain. |
| Fusing a region that needs the host between layers | `Fuse` requires `Host = NoHost` on every member. |
| A MoE link in a fork lane, or beside another MoE link | `MoeProtocol` is not `Concurrent`, and `fork` accepts only `Host: Concurrent` links. |
| A sync, readback or legacy `Tensor` in a gap | `Recorder<Chained>` has no such methods; the tombstones name them (§12.11), and `Operand<Chained>` admits only roles. |
| Capturing a pinned upload | the upload op needs `M: Live`, and `Capturing` is not `Live`. |
| An activation carried from one link to the next | `'c` is per link; only `Resid` and `Table` cross. |

Tests (with the code): the slot state machine with a fake backend (Empty, Ready, Refused,
epoch retirement); the event order of a chain of graph, gap, graph with host `before` and
`after` (raw expected sequence, with the capture of an empty slot ahead of `before`); a region mark restores the cursor and every link of a
phase allocates at the same offset; `Fuse` of two `NoHost` regions equals the two run
separately, bit for bit; a failed capture mid-chain completes the chain through `Chained`;
exactly one stream synchronisation per wave (a counter); a chain break is reported with its
reason. GPU, `--features cuda`: chain output equals the eager forward bit for bit at
`R = 1..4`, and a second wave replays with zero captures.

Open items the prototype decides: capture-and-instantiate cost with queued work, whether
the per-layer split already hides the launch overhead or the fused spans are needed (§8
measures first), and how many chain breaks Flash-Next carries until QSA selection is ops.

### 12.13 Composition a graph cannot express

A captured graph is a fixed dependency DAG: no branch, no loop, no choice of kernel by
value. The chain is Rust, so it can be all of those, provided the deciding value is known
to the host without asking the device. Nothing in this section adds a host wait.

**The one enabling type.**

```rust
pub struct Known<T>(T);   // a host-known value; the field is private
// constructors (all pub(in crate::models::graph) except Known::constant):
//   Rows::known(), the layer kind, model config, WaveChain::epoch(),
//   StagedWave::known_*  (metadata the host recorded when it wrote the staged tables:
//   which rows are active, a length bucket, a window class)
// there is no constructor from a device buffer, a Resid, a Table's contents or a Tensor
impl<T: Copy> Known<T> { pub fn get(self) -> T; pub fn map<U>(self, f: impl Fn(T) -> U) -> Known<U>; }
```

A value the device computed can only reach the host through a readback, which `Chained` and
`Capturing` do not have (§12.11). So `Known` cannot be forged from device data, and every
method below is sync-free by construction. Host-known is wide: the host built the paged-KV
tables and the descriptor tables, so it knows every length, window and active row.

**Variants and choosing by value.**

```rust
pub trait Variant: Copy + Eq { const COUNT: usize; fn index(self) -> usize; }   // model enums; COUNT <= 8, a const assert

impl<'w, A: ActDtype> WaveChain<'w, A> {
    pub fn when(&mut self, k: Known<bool>, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()>;
    pub fn repeat(&mut self, n: Known<usize>, f: impl FnMut(&mut Self, usize) -> Result<()>) -> Result<()>;
    pub fn variant<V: Variant, R: Region<A>>(&mut self, layer: usize, v: Known<V>,
        make: impl FnOnce(V) -> R, host: &mut R::Host, staged: &StagedWave<'_>, io: &mut R::Io<'_>) -> Result<()>;
}
pub struct Either<R1, R2>(Known<bool>, R1, R2);   // Region when R1, R2 share Io and Host; sugar for variant::<bool>
```

- `when` includes or skips a link: layers no active row needs, the MTP head when not
  drafting, a mixer on a wave with no recurrent state to advance. A skipped link adds no
  launch and is logged as skipped with its reason.
- `repeat` is a host loop with a host-known count (speculative depth, a layer range). The
  closure takes `&mut Self`, so it can only call links one after another; links cannot
  interleave.
- `variant` selects one of at most eight captured graphs by a small enum. This is how a
  data-dependent grid is handled without changing the kernel: the host knows the length
  bucket, so the bucket is the variant, the variant is part of the slot key, and the
  variant's value may become a frozen scalar through `Fixed::from_variant(v)`. `make` is
  called only when the slot is empty or refused, so a `Ready` replay builds nothing.
  Cost is one capture per variant per `(layer, kind, R, epoch)`, bounded by `COUNT` and
  the cache's least-recently-used eviction, so a bucket must be coarse (a split count, a
  window class), never a raw length.
- The launch-count test of §12.8 runs per variant and per branch: within one choice the op
  sequence is identical across `R`.

**Fork and join (concurrency across links).**

```rust
pub fn fork<T, U>(&mut self,
    a: impl FnOnce(&mut Branch<'_, 'w, A, LaneA>) -> Result<T>,
    b: impl FnOnce(&mut Branch<'_, 'w, A, LaneB>) -> Result<U>) -> Result<(T, U)>;
```

`fork` records an event on the compute stream, has a side stream wait on it (a device-side
wait, not a host wait), runs branch `a` on the compute stream and `b` on the side stream,
and joins by making the compute stream wait on an event recorded after `b`. Each `Branch`
is a gap-like scope with a graph-safe recorder, and a branch may run a link whose `Host`
is `Concurrent` (§12.4); a MoE link is not.
The types carry the hard parts: each branch gets its own **lane**, a disjoint sub-span of the
phase that `plan_wave_transient` sizes for both peaks (so the two do not rewind onto the
same addresses), and the lane is part of the `'c` brand, so an activation cannot cross
lanes. The branches capture `&mut Resid` borrows, so two branches cannot write the same
residual and a branch cannot read what the other is writing; the borrow checker proves the
join is race-free. Uses: the shared expert beside the routed experts, an expert upload
overlapping the dense half. The side stream must be non-blocking: the compute stream is the
legacy null stream, which implicitly synchronises with blocking streams (§4.2), so this
feature depends on the same unproven assumption as the whole design and is built only after
the prototype proves it.

**Device-to-host notification.**

```rust
pub struct Beacon { /* a mapped host word, owned by whoever polls it */ }
impl Beacon { pub fn reached(&self) -> u64; }                    // host-side load, never blocks
pub fn notify(&mut self, word: &Beacon, value: Known<u64>);     // one tiny kernel on the compute stream
```

`notify` launches a one-thread kernel that stores `value` into the beacon's mapped word
when the GPU reaches that point, so the CPU does not wait and nothing runs on a driver
thread. It is the mechanism the dispatch already uses (`live_row`, the abort word, the
summary ring are mapped words that kernels write and host threads poll). The host acts on
it by polling `reached()` where it would otherwise have been called back: release a staging
slot once the word passes the sequence of the uploads that read it, publish a ring word, read
a per-layer timestamp. There is no closure, so there is no `Send` bound, no driver-thread
re-entrancy rule and no `cuLaunchHostFunc` hold on the stream, and WDDM has no host-function
node to measure. `Beacon` is a plain value that owns its word; the kernel is an op
(`LAUNCHES = 1`, `Live`, so it is never captured) and a notification is a link of its own,
never inside a region, because a baked value would repeat on every replay.

**Pipelining across waves.**

```rust
pub fn scope_pending<A, T>(&mut self, wave: DecodeWave, f: …) -> Result<PendingWave<T>>;
#[must_use] pub struct PendingWave<T> { /* end event, result, deferred generation release */ }
impl<T> PendingWave<T> { pub fn resolve(self) -> Result<T>; }   // the one host wait; Drop does it too
```

`scope_pending` records an end event instead of fencing at exit and defers the generation
release until the event is complete, so the host can start the next wave's admission,
KV planning and table staging while this wave's tail runs. `resolve` is the single host
wait, taken when the logits are actually needed, and a dropped `PendingWave` resolves
itself, so a forgotten one cannot leak a generation. The limits are real: `begin_wave`
refuses overlapping generations today, so this needs a two-deep tier with the lanes of
`fork`, alternating by wave parity; and the next wave's input tokens come from sampling this
wave's logits, so only the work that does not depend on them can overlap (everything except
the sampled ids, unless sampling moves to the device). It is built last, and only if the
timeline shows the host idle behind the fence.

**Not expressible here, by design.** Control flow on device data (early exit on a sampled
EOS, skipping an empty expert) needs a readback or a device-side conditional. CUDA graphs
have conditional nodes (device-driven if and while) in recent toolkits; whether they work
under WDDM on these cards is unchecked, and they would live inside a graph, so they are a
separate extension of the op set and not a chain method.

**What this adds to §12.2.**

| Mistake | Why it does not compile |
|---|---|
| Branching or looping on a device value | `when`, `repeat` and `variant` take `Known`, which has no constructor from device data. |
| An unbounded variant count | `Variant::COUNT` is a const asserted at most eight. |
| Two forked branches writing the same buffer, or one reading what the other writes | each branch captures `&mut` borrows of its buffers; the borrow checker refuses the overlap. |
| An activation crossing fork lanes | the lane is part of the `'c` brand. |
| A notification touching a stream, tensor or chain | there is no closure: `notify` takes a `Beacon` and a `Known<u64>`, and the host side is a load of a mapped word. |
| Forgetting to resolve a pipelined wave | `PendingWave` is `#[must_use]` and its `Drop` resolves it. |

Tests: `when` and `repeat` log exactly the links run, skipped or repeated, in order;
`variant` keys give distinct slots per variant and reuse a slot for a repeated variant, and
`make` is not called on a `Ready` slot; `Known` has `compile_fail` doctests for forging
from a `Resid` and a `Tensor`; the fork join orders as expected against a fake backend
(wait before the side launch, wait before the post-join compute launch); a notification's word
is stored after the preceding launch and before the following one (a fake backend orders
the kernel in the stream) and `reached()` never blocks; a dropped `PendingWave` releases
its generation. GPU, `--features cuda`: a forked chain equals the sequential one bit for
bit, and a variant chain at each bucket equals the eager forward.
