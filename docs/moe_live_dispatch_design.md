# Live MoE Dispatch — the expert forward without a host round trip

> **Status — Proposed.** Replaces the host-orchestrated expert path
> (`route_indices` → `submit_moe_work` → pipeline-thread compute) with one
> device-side path for every MoE model, paged or all-resident. The pointer table
> the device path already uses becomes *live*: entries for non-resident experts
> are zero, the expert pipeline thread fills them as copies land, and the gate
> GEMM waits on the entry instead of the host waiting on the GPU.
>
> Everything here is grounded in the tree at `8d061e4fc` + `33dac8cf2`. Line
> references are to that tree. `docs/archived/gpu_native_moe_dispatch.md` §Phase B
> sketched this direction ("residency table with NULL sentinels, doorbell
> miss-service"); this document is the design that was missing from it.

---

## 1. Why

On every machine whose expert cache streams — the 16 GB and 24 GB cards for
every MoE model, and DeepSeek-V4 everywhere — each MoE layer of each forward
does this today (`quantized_qwen3_moe.rs:629-856`, `expert_lre/handle.rs:1266-1316`):

1. the forward thread enqueues attention, the shared expert, the router and
   `moe_route`, then **`e2.synchronize()`** (`quantized_qwen3_moe.rs:751`) —
   which, because the routing stream waits on the compute stream, drains the
   whole null stream;
2. it sorts assignments on the host, sends a `Work` request to the pipeline
   thread and **blocks on `recv`**;
3. the pipeline thread classifies, evicts, issues miss DMA, then enqueues the
   gather and grouped GEMMs itself, and answers.

The GPU is idle from the end of step 1 until the pipeline thread's first launch,
and can never be more than one layer ahead of the host. On Qwen3.8-Flash-Next the
profile puts `fwd_routing_wait` at 30.9% and `submit_roundtrip` at 14.8% of span
time (`docs/archived/qwen38_flash_next.md:1641`). DeepSeek does the same through
a synchronous `indices.to_vec2()` (`latent_moe/engine.rs:556`). Where every
expert is resident the device path already exists and removing this round trip
took decode from 26–28 ms to 18–19 ms per step
(`docs/archived/gpu_native_moe_dispatch.md:3-11`); that path is refused the
moment one expert is not resident (`handle.rs:713`, `:1036-1044`) and, for the
rest of the process, the first time the weight zone concedes ground
(`handle.rs:1665-1683`).

## 2. The shape

```text
forward thread (null stream)                      pipeline thread            copy stream
──────────────────────────────                    ───────────────            ───────────
router → route (top-k, device)
bucketize  ── reads live gate row, orders
             resident tiles first, writes
             routing summary[n]
record e_route(n); routing stream: wait,
  DtoH summary[n] → pinned[n], record C(n)
send Routed(n) ─────────────────────────────────▶ wait C(n)
gather                                            classify vs bookkeeping
gate GEMM  ── resident tiles run at once;         clear victims' gate entries ──▶ 8 B zero
              a miss tile spins on its entry      load misses ────────────────▶ record H2D
up GEMM, SwiGLU, down GEMM                        fill up, down, then gate ───▶ 3 × 8 B
scatter (unchanged, deterministic)                stats, transition, prefetch
next layer …                                      next message …
```

- **The forward thread never blocks on the MoE.** Every launch has a
  data-independent bound (as the device path already does,
  `quantized_qwen3_moe.rs:359-363`), so it enqueues the whole layer and moves on.
- **The GPU waits only on a real miss**, and only the tiles of that miss: hit
  tiles compute while the copy is in flight — the overlap the determinism fix
  gave up (`pipeline.rs:2556-2572`) comes back without touching determinism,
  because each tile writes its own rows and the scatter sums them in canonical
  expert order regardless of which tile ran first (§4.3).
- **The host is told, not asked.** The routing summary goes to the pipeline
  thread over the routing stream it already owns (Option A); the forward thread
  only records events and sends a message.
- **One path.** An all-resident cache is the case where every entry is non-zero
  and no message ever produces a load. A conceded zone is a few cleared entries.
  The host path, inline mode and the reader path are deleted (§10).

## 3. The live table

### 3.1 Layout

`GpuDispatchTables` keeps its three `[rows × n_experts]` u64 arrays
(`gate_ptrs`, `up_ptrs`, `down_ptrs`, `gpu_dispatch.rs:40-46`) and its per-row
dtypes. What changes:

- **Built for every cache**, at construction, over **every** MoE row the cache
  serves (`num_moe_layers`, which already includes the qwen4exp/qwen35 MTP head
  row, `qwen4exp/engine.rs:444-456`, `qwen35/quantized_weights.rs:340-351`).
  `min_layer` is 0. Per-row dtypes and shapes come from `layer_geometries`, not
  from resident slots, so a sparse grid is expected rather than refused.
- **A zero gate entry means "not resident".** The gate entry is the only one
  that is ever cleared and the only one anything waits on. Up and down entries
  are written before the gate entry on every fill (§3.3) and are never read for
  an expert whose gate entry was not observed non-zero first, so stale up/down
  values behind a zero gate entry are unreachable.
- **The build is a load-time contract, not a probe.** Every condition
  `GpuDispatchTables::build` checks today becomes an error from `ExpertCache::new`
  naming the model and the condition: KO weights in every row, uniform shapes
  across rows, one dtype within a row, `n_experts ≤ MOE_MAX_EXPERTS` (512),
  `k ≤ 16` (`moe_route`) / `≤ 32` (bucketize), N % 32 and K % 128 for both
  projections, compute stream is the legacy null stream
  (`gpu_dispatch.rs:358-369`). There is no host path to decline to.
- **The `hidden % 1024` condition is removed** (`gpu_dispatch.rs:241`). It cites
  "the q8a1024 byte-row gather", which is now tile-granular and serves any
  multiple of 128 (`candle-core/src/quantized/cuda.rs:7278-7281`); it is what
  refuses Flash-Next (hidden 2560) today.
- **Deleted with it:** `built_capacity`, `built_floor`, `built_concede_epoch`,
  `zone_moved` and the permanent refusal (`gpu_dispatch.rs:60-85, 567-586`;
  `handle.rs:1632-1697`). Their job — never dispatch through an address the zone
  no longer owns — is now done by clearing entries (§6), which is exact where the
  epoch was conservative.

### 3.2 Pointer constants — fills need no staging

A fill writes three u64 that are a pure function of `(slot, row geometry)`:
`slot_base(slot) + slot_offsets(geom)` (`pipeline.rs:749-762, 850-888`). They
are precomputed once into a **pinned, immutable** array
`PointerConstants[geom_class][slot] = {gate, up, down}` for every slot up to
`zone.limit()`, plus one pinned zero word. Every table write is then a
`cuMemcpyHtoDAsync` of 8 bytes **from an address that never changes**, so there
is no ring to manage, no event to wait before reuse, and no host write racing an
in-flight copy. Geometry classes are the distinct offset triples across rows —
two on the dynamically quantized 3.5/3.6 checkpoints (Q5_KO/Q6_KO down,
`gpu_dispatch.rs:93-107`), one elsewhere.

### 3.3 Writes

All table writes are copy-engine copies on a non-blocking stream — never a
kernel and never a memset (`cuMemsetAsync` can be implemented as a kernel, which
cannot be scheduled while spinning blocks hold the SMs).

| Operation | Stream | Sequence |
|---|---|---|
| **Fill** expert `e` of row `r` into slot `s` | copy stream (demand, prefetch, relocation) or streamer stream (whole-layer stream) | record H2D into `s` → `up[r,e]` → `down[r,e]` → `gate[r,e]` |
| **Clear** the tenant `(r', e')` of slot `s` | copy stream | `gate[r',e'] ← 0`, **before** any byte of the new tenant is written to `s` |

Copies on one stream execute in order, so the gate entry cannot become non-zero
before the expert's bytes and its up/down entries have landed.

`cudarc` 0.17.8's `CudaContext::new_stream` creates `CU_STREAM_NON_BLOCKING`
streams (`core.rs:454-469`), so the copy, routing and streamer streams do not
implicitly serialize with the null stream: a fill proceeds while a null-stream
kernel spins.

## 4. Kernel changes (existing kernels only)

### 4.1 `moe_bucketize` (`candle-kernels/src/simple/moe_bucketize.cu`)

Three new arguments: `const uint64_t* gate_row` (the live gate table at
`row × n_experts`), `int decode_tokens`, `uint32_t* summary`.

1. **Phase 1** (`:110-120`): alongside the histogram, a grid-stride
   `ld.volatile` of `gate_row[e]` into a new `__shared__ uint8_t sh_res[512]`
   (+512 B, ~40.5 KB of the 48 KB static cap), and an `atomicOr` of a decode bit
   into `sh_dec[e]` when assignment `i` belongs to a token `< decode_tokens`.
2. **Phase 2** (`:123-147`, thread 0): `sh_offsets` — the **row** layout — is
   computed exactly as now, ascending expert id. `sh_tile_pref` — the **tile**
   order — takes two passes: resident experts with tokens first, then the rest,
   each ascending. `num_tiles` is unchanged.
3. **Summary**: `summary[e] = count | (decode << 31)` for every expert, written
   grid-stride after phase 1. 512 words — 2 KB at the widest model.
4. Phases 3–5 (`tok_ids`, `weight_ids`, `perm`, `rw_ids`, `token_starts`) are
   untouched: the row layout is identical, so the gather, the GEMM rows and the
   scatter are bit-identical to today's.

The kernel stays one block of 256 threads on the compute-stream handle
(`:317-346`). It is not timing-sensitive: a stale read only reorders tiles (a
zero read for an expert that has just been filled makes it a spin that ends at
once; a non-zero read cannot be stale — §6.2).

### 4.2 `grouped_qmatmul_dev_q8a128` — the wait

In `quantized_matmul_grouped_entry` (`candle-kernels/src/quantized/kernel.cuh:2468-2532`),
between `expert = tile_expert[tile]` (`:2493`) and the pointer dereference
(`:2494-2495`) — before the first `cp.async` of the weights (`:2513`), so
nothing upstream depends on the pointer:

```text
if (tid == 0) {
    p = ld.acquire.sys(weight_ptrs[expert])          // not the __restrict__ const path,
    while (p == 0 && wait != nullptr) {              // which may compile to ld.global.nc
        if (ld.volatile(wait->abort) || now() - t0 > SPIN_LIMIT_NS) __trap();
        __nanosleep(ns); ns = min(2 * ns, 4096);
        p = ld.acquire.sys(weight_ptrs[expert]);
    }
    if (p == 0) __trap();                             // a zero pointer is never computable
    s_w = p;
}
__syncthreads(); weights = s_w;
```

- **New arguments**: `const MoeWait* wait` (`{abort}` in device memory; null for
  the up and down launches, which never wait) and `int tile_offset` (§4.4).
  Plumbed through `INSTANTIATE_KERNEL_GROUPED_INT8` and its m4/m8 forms
  (`kernel_instantiate.cuh:467-521`), `run_grouped_quantized_matmul`
  (`dispatcher.cu:1576-1657`), the FFI (`quantized/api.rs:282`) and the wrapper
  (`cuda.rs:6686-6781`). The FP grouped instantiation (`:352-364`) gains the same
  parameters so the ABI stays uniform; the host-table int8 path
  (`grouped_matmul_gemx_q8a128`, still used by `latent_moe/attention.rs:482`)
  passes null/0.
- **Cost on the resident path**: one acquire load and one barrier per block.
- **Weights are read through `cp.async.cg`**, which bypasses L1
  (`kernel.cuh:1970`), and the first read of the slot follows the acquire of
  its pointer, so a block cannot hold stale cached bytes of a previous tenant.
- **`__trap()` is the error path.** It turns an abort or an overrun into a
  sticky launch error, which the existing machinery already treats as
  context-fatal (`cuda_backend/error.rs:107-119`, `gpu_poison.rs`) and which
  surfaces at the next synchronizing call — the sampler's
  (`batched_sampler.rs:1777`) in every driver — before any token computed from
  the layer is returned. §8.

### 4.3 Why tile order changes nothing numerically

The grouped kernel's blocks are independent — no atomics, counters or
inter-block synchronization (`kernel.cuh:2014-2258`) — and every output row has
exactly one writer with a fixed K order. `silu_mul_q8a128` and
`fused_deterministic_scatter` index rows and `perm`, never tiles
(`moe_scatter.cu:182-246`). So reordering tiles is invisible in the output bits.
The only things that assumed ascending tile order are a test reference
(`cuda_tests.rs:9417-9431`, compared exactly at `:9507-9509, 9547-9561`) and a
doc comment (`cuda.rs:6683-6684`); both change with the kernel.

### 4.4 Gate launch shape

- **`row_fast = 1` for the gate launch.** The tile index then rides
  `blockIdx.y`, so all row tiles of the resident tiles are dispatched before
  any miss tile (`dispatcher.cu:1633`, `kernel.cuh:2485`). `row_fast` does not
  affect numerics. Up and down keep `grouped_grid_row_fast` (`cuda.rs:95-97`).
- **Sliced against the watchdog.** A spinning launch is a long launch, and the
  Windows 2 s TDR resets the device under one (`cuda.rs:2571-2582` banded a
  dequant for exactly this). The gate GEMM is issued as
  `⌈launch_tiles / gate_slice_tiles⌉` launches over `tile_offset` ranges.
  `gate_slice_tiles` is fixed at load from the slot size:
  `max(1, GATE_SLICE_BYTES / slot_bytes)` with `GATE_SLICE_BYTES` sized so that
  even a slice of all-cold experts is served well inside the watchdog at the
  pack's measured read rate. On Flash-Next (≈1.4 MB slots) decode and most
  prefill are one launch; DeepSeek's ≈12.6 MB MXFP4 experts at prefill width are
  the case it exists for. The count depends only on `launch_tiles`, so it stays
  data-independent.
- `SPIN_LIMIT_NS` is the backstop inside a slice, below the TDR, and is never
  meant to fire.

### 4.5 Routers

Unchanged. `moe_route` (`moe_scatter.cu:285-422`) serves Qwen3, 3.5, 3.6 and
Flash-Next. DeepSeek's `Gate::route` (`latent_moe/moe.rs:127-264`, sqrt-softplus
scores, selection bias, `route_scale`, device `tid2eid` for hash layers) already
produces `(weights, indices)` on the device and feeds bucketize directly — the
all-resident DSpark drafter already chains exactly this
(`latent_moe/dspark_experts.rs:45-324`).

## 5. The routing summary channel (Option A)

### 5.1 Per layer, on the forward thread

After bucketize:

1. record `e_route(n)` on the null stream;
2. the routing stream waits `e_route(n)`, copies `summary_dev[n % RING]` →
   `summary_host[n % RING]` (pinned), records `C(n)`;
3. `cuEventQuery(e_route(n))` — non-blocking; it makes WDDM submit the queued
   batch, so the bucketize the pipeline thread is about to wait on is not left
   sitting in an unflushed command buffer (§11);
4. **send `Routed { seq: n, row, pass, slot: n % RING, done: C(n) }`** — before
   the gate GEMM is enqueued, so that every enqueued gate launch has its message
   already in the FIFO (§7);
5. enqueue gather, gate (sliced), up, SwiGLU, down, scatter.

**This is not the experiment that was reverted.** DeepSeek once moved its
routing readback onto a side stream into a pinned buffer and lost 8% on the cfg8
gate (`docs/deepseek/deepseek_decode_launch_overhead.md:161-163`,
`latent_moe/engine.rs:541-545`). That change kept the forward thread *waiting*
for the copy, so it paid the extra event and stream calls and removed nothing:
the wait still drained the stream. Here the forward thread never waits for the
summary — the GPU keeps executing the layer while it travels — so the same
calls buy the removal of the drain. The per-layer host cost is still real
(§6.5); §15 measures it.

`summary_dev` is a dedicated device ring `[RING × n_experts] u32`, separate from
the bucketize workspace, so the next layer's bucketize cannot overwrite it while
the DtoH reads it. `RING = 64`: 128 KB per side at 512 experts.

### 5.2 Back-pressure is the channel bound

The pipeline channel becomes `sync_channel(RING − 2)`. Message `n + 1` can only
be dequeued after message `n` is fully processed, so when `send(n + RING − 1)`
returns, message `n` has been processed — `C(n)` waited and the host slot read.
Bucketize `n + RING` and DtoH `n + RING` are enqueued after that `send`. So
neither ring slot is overwritten before it is consumed, with no other bookkeeping.
A full channel blocks the forward thread, never the GPU, and the pipeline thread
can always drain it (it waits on nothing the forward thread holds, §7).

The single `PinnedRoutingBuffer` (`handle.rs:489-534`), whose correctness relied
on the forward thread blocking, is replaced by this ring.

### 5.3 What the summary carries

Count and decode bit per expert. That is the whole of what the host consumed
from the readback: the routed set (classify, `transition_matrix.observe`,
speculative validation), the per-expert decode attribution
(`record_hit` vs `record_prefill_hit` / `record_prefill_elevate`,
`pipeline.rs:2458-2467, 1541-1549, 1683-1685`), and hit/miss counts. It does not
carry the table bit: the host decides misses from its own bookkeeping, which §6.2
shows is always at least as current as anything bucketize read.

## 6. The pipeline thread

### 6.1 What it does per `Routed(n)`

1. **Pass boundary** if `pass` advanced: `reset_pass`, `drain_streams`,
   `adapt_prefetch_depth`, end-of-pass decay — as `process_request` does today
   (`pipeline.rs:2386-2401`, `:2699-2701`), keyed on the forward thread's pass id
   instead of "layer index went backward".
2. Wait `C(n)` (host). Read the summary slot.
3. Speculative validation and `transition_matrix.observe` — unchanged.
4. `join_stream_for(row)` — unchanged, before classify, so failed streamed jobs
   classify as honest misses (`pipeline.rs:2443-2447`).
5. **Classify** — `classify_and_load` (`pipeline.rs:1496-1754`) with the routed
   set and decode set from the summary: hit = bookkeeping says resident (in VRAM
   *or* an in-flight speculative/streamed install whose fill is queued); miss =
   otherwise. Demand eviction and `allocate_slot` as today.
6. On the copy stream: wait `e_route(n)` (edge 1, §6.2), **clear** each
   victim's gate entry (§3.3), **load** the misses (`load_experts_batched`, cold
   reads then warm H2Ds, unchanged) and **fill** each one.
   `cuStreamQuery(copy_stream)` to flush.
7. Stats, then speculative prefetch for the next rows and, at prefill width,
   `stream_next_layer` — under the rules of §6.3.

It computes nothing. `process_request`'s compute half, `post_compute`'s response
plumbing, `compute_experts_grouped`, the `Work` response channel, `Hint`, and
every compute-stream wait it issued are gone (§10).

### 6.2 Ordering — the async side keeps its shape

The expert copies have produced corruption through races before (the fence-ring
key collision, `pipeline.rs:3602-3612`; the retraction that let KV memset bytes a
GEMM was reading, `docs/expert_cache_design.md` §12.4). So this design does not
re-architect the copy side. **No stream is added, no stream is removed, and no
new kind of ordering primitive is introduced.** The copy stream, the streamer
stream and its thread, the cold staging rings, `load_experts_batched`,
`load_expert`, the warm/cold sourcing and the per-plan done fences all stay as
they are. What changes is listed exhaustively here, edge by edge:

| # | Ordering edge today | Here | Why |
|---|---|---|---|
| 1 | copy stream waits a **fresh** compute event before every load batch (`order_copies_after_compute`, `pipeline.rs:1804-1812`, at 1983) | **kept, re-pointed**: it waits `e_route(n)`, the event the forward thread recorded after bucketize(`n`) and carried in the message | a fresh record now lands *behind* the spinning gate GEMM, and the copy that would release the spin would wait on it. `e_route(n)` covers exactly the readers the fresh event covered for every invocation `< n`; readers `≥ n` are §6.2's Rule D |
| 2 | streamer stream waits a **fresh** compute event `after` (`pipeline.rs:2225`, `streamer.rs:170-184`) | **kept, re-pointed**: it waits `clears_done`, recorded on the copy stream after edge 1's wait and the plan's victim clears | same deadlock; `clears_done` implies `e_route(n)` transitively, so the streamer is ordered after every reader edge 1 covers *and* after its victims' entries are zero |
| 3 | compute waits `classified.fence` and the prefetch fence ring before computing (`pipeline.rs:2497-2554`, `types.rs:290-301`) | **removed** — the fill-after-bytes order (§3.3) plus the spin does this job | the forward thread enqueues the GEMM before the pipeline thread has recorded these fences; there is nothing to wait on yet. The ring stays, for `is_complete` late-load accounting only |
| 4 | compute waits the streamer's done fence at the target layer (`pipeline.rs:2273-2277` → 2498) | **kept, re-pointed**: the **copy stream** waits it at `join_stream_for`, which runs before classify | same reason as 3 for compute. The copy stream must be ordered after a streamed install before it may clear or re-tenant that slot (otherwise a late streamer write lands on the new tenant, or a late streamer fill resurrects an evicted pointer). Placing it before classify also closes an existing race, §18 |
| 5 | compute waits copy after relocations (`order_compute_after_copies`, `pipeline.rs:1834-1843`, 2943-2945) | **kept as is** | relocation runs only with no pass in flight (§9) |
| 6 | routing stream waits `e1` on compute before the routing DtoH (`quantized_qwen3_moe.rs:697-704`) | **kept**: it waits `e_route(n)`, recorded after bucketize instead of after `moe_route` | it copies the summary rather than the indices |
| 7 | host waits copy-stream events in `ColdStaging::acquire` (`pipeline.rs:1197-1204`) | **unchanged** | — |
| 8 | `quiesce_before_handover` `ctx.synchronize` (`pipeline.rs:3071-3080`) | **unchanged**, reached under §9's entry rule | — |
| 9 | the hint path's fresh-event wait (`pipeline.rs:3185-3192`) | **removed with the hint** | the forward thread has no host routing to hint from |
| new | — | compute waits `clears_done` at the start of a pass (Rule P below) | the one new edge, and of a kind that exists today (compute waiting on a copy-stream event, as edge 3 did) |

New *operations* on the copy stream are only the 8-byte clears and fills of §3.3,
issued from the same thread into the same FIFO as the loads they bracket. The
streamer's jobs gain the fill copies, issued by the streamer thread on its own
stream after each job's bytes; its read, staging and upload logic is untouched.

The rules below show that, with edges re-pointed this way, no slot is overwritten
while a kernel can still read it.

**Fact F.** When the pipeline thread holds `C(n)` — and on the GPU, wherever the
copy stream has passed its wait on `e_route(n)` — bucketize(`n`) has completed,
and, the null stream being serial, so has every kernel enqueued before it,
including every GEMM of every invocation `< n`. Edge 1 makes this a GPU-enforced
property of every overwrite, not only a host-side inference.

**Rule D (demand evictions may take any unprotected, unpinned slot).** Processing
`Routed(n)` with at least one miss, the victim of a load may sit in any row. The
clear is queued before the load, the load before its fill, and the gate GEMM of
`n` cannot complete until that fill lands. Readers of the victim's old pointer:

- invocations `< n` — complete (Fact F);
- invocation `n` itself — reads only experts routed in row `r`, which are
  protected (the classify protect set, `pipeline.rs:1575-1589`);
- invocations `> n`, in this pass or any later one — their bucketize and GEMMs
  run after gate(`n`) completes, hence after the clear has landed; they read
  zero and the expert is a miss for them, loaded on their own message.

So demand eviction needs no lock and no knowledge of where the GPU is. This
covers every victim choice the current policy can make, including the
ahead-of-the-wave ones (`cache.rs:704-712, 885-916`).

**Rule S (speculative evictions only from rows behind, in the current pass).**
Prefetch and whole-layer streaming are issued while processing `Routed(n)` whether
or not `n` had misses, so the GPU may be anywhere past bucketize(`n`). Their
victims must therefore come from rows whose readers are known complete and which
nothing will read again before the clear lands:

- rows already passed **in this pass** (`row' < row`, invocation `< n`) —
  complete by Fact F, and not re-invoked in this pass by definition of a pass
  (§6.4);
- **not** rows ahead, and **not** the wrap into the previous pass's tail that
  `PREFETCH_EVICT_WINDOW` allows today (`cache.rs:219, 628-634, 832-848`). The
  window keeps its depth and its ranking; it loses the wrap. When `row < 5` it
  is simply shorter.

**Rule P (a pass may not start over un-landed speculative clears).** Rule S stops
speculative clears from racing *this* pass; the next pass re-reads every row. A
**pass lock** (a `Mutex<PassState>` in `ExpertCache`) closes that:

- the forward thread, at the first invocation of a new pass, takes the lock,
  advances `pass`, takes `spec_clear_event`, releases, and if there was one
  enqueues `null_stream.wait(spec_clear_event)` — a GPU-side wait on the copy
  stream, which never waits on compute, so it always completes;
- the pipeline thread, before a speculative eviction for pass `P`, takes the
  lock; if `pass != P` it skips the eviction (the forward has moved on; it may
  still prefetch into free slots); otherwise it queues the clears, records
  `spec_clear_event` (`clears_done`, edge 2) on the copy stream after them,
  releases, then queues the loads and fills. A whole-layer stream plan carries
  that event as its `after`.

The lock is held for a few asynchronous enqueues on one side and three field
updates on the other. A free-slot prefetch needs no lock: a free slot has no
reader, and a fill landing during the next pass is just an expert becoming
resident.

**Rule J (cross-stream clears).** A streamed install's fill is on the streamer
stream. Until its plan is joined it is in `stream_loads` and protected from
demand eviction exactly as today (`pipeline.rs:1575-1589`, `cache.rs:684-688`);
at the join the copy stream waits the plan's done fence (edge 4), so every later
copy-stream clear of that entry is ordered after the streamer's fill.

**Why bucketize never reads a pointer that is about to be overwritten.** A
non-zero entry bucketize reads was written by a fill, after the bytes. It could
only be stale if a clear for it was queued but not landed. Rule D clears land
before any later bucketize (above); Rule S clears are of rows the current pass
has finished; Rule P makes the next pass wait for them. So bookkeeping and the
table agree for every row a bucketize can be reading, apart from one benign case:
a fill queued but not landed reads as zero, the expert is reported routed, the
host sees it resident and does nothing, and the spin ends when the fill lands.

**Why the gate GEMM never spins on a fill that is not coming.** A routed expert
whose entry reads zero is, in bookkeeping, either not resident — the host loads
it on this message — or an in-flight install whose fill is already queued. A
streamed job that fails is uninstalled by `join_stream_for` *before* classify,
so it classifies as a miss and is loaded. A load that fails is the abort path
(§8). There is no fourth case.

### 6.3 The demand-slot guarantee

`minimum_resident_slots(E) = 3E + 1` (`cache.rs:156-158`) prices the pinned rows
(`PINNED_LAYERS = 2`) plus one working row plus one. Rule D shows that only
**one** row is ever being read when a miss is serviced — the GPU is held at that
row's gate GEMM and nothing behind it is reading — so the existing floor is
sufficient, and the concern that several in-flight layers need protecting does
not arise. The protect set is left exactly as it is. That keeps one existing
edge: at the floor, a wide row plus protected in-flight installs can leave no
victim (`cache.rs:717-719`), which today fails the wave and here is the abort
path (§8). Closing it would mean letting demand eviction take an in-flight
install, which changes how the copy and streamer sides interact; it is left as
Q5 rather than folded into this change.

### 6.4 Passes

A **pass** is a run of invocations with strictly increasing row. The forward
thread starts a new pass when an invocation's row is `≤` the previous one: every
trunk forward; each sub-forward of a slab-split prefill (`wave_driver.rs:398-437`);
each MTP draft step, which re-invokes the head row (`qwen4exp/draft.rs:467`,
`qwen35/mtp.rs:327-332`). This is decided inside the `ExpertCache` call the MoE
block already makes, so no driver changes. The current end-of-pass decay keys on
`moe_layer_idx + 1 == num_moe_layers` (`pipeline.rs:2699-2701`), which on models
with an MTP head fires on draft steps only; this design keeps that behaviour
unchanged (§13, Q3).

### 6.5 Does the outcome survive the conservative stream rules?

The goal is that **the forward thread never waits on the MoE, and the GPU waits
only for the bytes of a real miss.** Every wait this design keeps or adds,
checked against that:

| Wait | Who waits | Cost |
|---|---|---|
| copy stream on `e_route(n)` (edge 1) | copy stream | **none** — the pipeline thread only reaches it after holding `C(n)`, which follows `e_route(n)`, so it is already signalled |
| streamer on `clears_done` (edge 2) | streamer stream | the stream plan's uploads queue behind this layer's demand uploads on the bus. Bus throughput is unchanged — one H2D direction is shared either way — and demand, which the GPU is actually waiting on, goes first. The streamer's NVMe reads are host-side and still overlap, since `run_plan` only makes its *stream* wait (`streamer.rs:170-184`) |
| copy stream on the plan fence at join (edge 4) | copy stream | this row's demand uploads queue behind the rest of the row's stream plan — the same total the compute stream waited for today (edges 3 and 4), and only at prefill width, where plans exist |
| compute on `clears_done` at a pass start (Rule P) | GPU, once per pass | only copies queued *before* the last speculative clear: earlier rows' demand uploads (already consumed by their own spins) and prefetch uploads for rows the pass has passed. Expected to be already signalled when the GPU reaches it; measured as its own `gpu_span` (§15) |
| gate GEMM on a miss's fill | GPU, miss tiles only | the irreducible part: the bytes have to arrive. Hit tiles of the same row compute meanwhile |
| forward thread on `send` | host | only with `RING − 2` = 62 layers outstanding — more than a whole forward |
| forward thread on the pass lock | host | three field updates on the pipeline side; a boundary move only under §9's rule |

What the design deliberately gives up: speculative eviction loses the
wrap-around into the previous pass's tail (Rule S). That only affects prefetch
for the first rows after the pinned ones (rows 2–4 with
`PREFETCH_EVICT_WINDOW = 5`), whose victims must now come from free slots or the
rows already passed in this pass. Those rows still prefetch; they have a
smaller pool to take from. If the hit rate on rows 2–4 drops measurably, there
is a refinement that keeps the stream rules intact: when the message that
issues the prefetch had demand misses, queue its speculative clears *before* its
demand fills. The GPU is then held at that row's gate GEMM until those clears
have landed, so the tail rows' next bucketize must see them. That is a change
to the order in which one thread enqueues on one stream, nothing more.

What it does not change, and so does not regress:
- **Prefetch lead time.** Today a prefetch for row `L+1` overlapped the forward
  thread's next attention pass. Here it is issued while the GPU runs row `L`'s
  expert GEMMs and then `L+1`'s attention, which is at least the same window.
- **Host enqueue cost per layer.** The forward thread issues the same number of
  driver calls per layer that `route_indices` does today (two event records, a
  stream wait, a DtoH, plus one query in place of the synchronize). Run-ahead
  hides them for as long as the host enqueues a layer faster than the GPU
  executes one. Where it does not, the bound is launch overhead — the existing
  question `docs/deepseek/deepseek_decode_cuda_graphs.md` addresses. It is
  neither caused nor worsened by this design.

## 7. Deadlock freedom

The design is deadlock-free if nothing the gate GEMM waits for can itself wait on
the gate GEMM. Its only wait is a fill on the copy or streamer stream. Every wait
those streams can encounter:

| Waiter | Waits on | Can that wait on a spinning GEMM? |
|---|---|---|
| copy stream | `e_route(n)` (edge 1) | no — recorded before gate(`n`) was enqueued, after gate(`n−1`), whose fills are earlier in the same copy-stream FIFO |
| copy stream | streamer done fence (edge 4) | no — the streamer stream waits only on `clears_done`, a copy-stream event recorded earlier |
| streamer stream | `clears_done` (edge 2) | no — copy stream, see above |
| pipeline thread | `C(n)` | no — it follows bucketize(`n`), which precedes the gate GEMM |
| pipeline thread | `ColdStaging::acquire` (`pipeline.rs:1197-1204`) | no — copy-stream events |
| pipeline thread | NVMe reads, `join_stream_for` `recv` | no — the streamer depends only on the above |
| pipeline thread | `quiesce_before_handover` `ctx.synchronize` (`pipeline.rs:3071-3080`) | yes, transiently — every gate launch enqueued so far has its `Routed` message ahead in the FIFO (§5.1 step 4), so its fills are already queued; it completes |
| pipeline thread | KV pool mutex (in `renegotiate_boundary`) | only if another thread holds it across a device synchronize while a pass is open — §9 refuses before taking it |
| forward thread | channel `send` | no — the pipeline drains it unconditionally |
| forward thread | pass lock | no — the pipeline holds it only for enqueues, or for a boundary move with no pass in flight to wait on |

And nothing on the pipeline thread is ever enqueued on the null stream: it issues
no kernels and no allocations against the wave arena. Wave arenas reset in
stream order (`bump_arena.rs:83-105, 721-728`), which stays correct because every
expert kernel is now enqueued by the forward thread, in program order — the
`as_foreign_lease` leases and ticketed allocations that relied on "the submitter
blocks" (`quantized_qwen3_moe.rs:312-333`, `types.rs:409-419`,
`wave_buffers.rs:381-430`, `cuda.rs:5684-5688`) are deleted with the host path.

## 8. Failure

There is no host path to retreat to, so a miss that cannot be served is fatal,
exactly as a device fault is:

- **A load fails** on the pipeline thread (NVMe error, staging failure) → it
  writes the abort word (one pinned-constant H2D on a dedicated non-blocking
  **abort stream**, then `cuStreamQuery`) and stops serving. Every spinning
  block `__trap()`s; the sticky error surfaces at the sampler's synchronize and
  takes the existing poison path. No token computed from the aborted layer is
  returned.
- **The pipeline thread dies** → `DeadFlagGuard` (`pipeline.rs:3498-3509`) writes
  the abort word in its drop, before setting the flag. `pipeline_dead` stops
  gating dispatch.
- **A spin overruns `SPIN_LIMIT_NS`** (a bug, a stuck copy) → `__trap()`, same
  path, before the TDR would reset the device.

Errors that are recoverable today because they surface synchronously through
`submit_moe_work` were all one of these. The one that was not — "cannot evict
(all pinned)" — is made unreachable (§6.3).

## 9. Boundary moves

`renegotiate_boundary` (`pipeline.rs:2771-3023`) moves and relocates slots, so it
must not run under a bucketize that has read the old layout. Today it runs the
whole retraction and only then lets `set_weight_floor`'s `wave_is_live` latch
refuse, keeping the evictions (`region_pool.rs:2432-2439`, `pipeline.rs:3006-3009`).
Under this design it decides at entry, holding the pass lock:

- **Refuse at once** (answer 0, edit nothing) if a wave is live, or if the
  requester is not the forward thread and a pass is open. A pass opens at its
  first invocation and closes when the forward thread reaches phase 0
  (`reclaim_spare_ground` / `request_kv_ground`, `batched_model.rs:781, 813`,
  `qwen4exp/wave.rs:2264`, `latent_moe/wave.rs:1373`, `qwen35/forward.rs:781`).
- **Otherwise proceed.** Either no pass is open, or the requester *is* the
  forward thread and is blocked in the round trip, so nothing new is enqueued
  until it returns. The quiesce is safe (§7). Evictions clear entries;
  relocations copy D2D then fill the destination's entries; all of it lands
  before the closing quiesce, ahead of the floor publish.

That keeps every current caller working: phase 0, phase-1 KV claims before the
first MoE invocation, and the forward thread's own claims during a draft walk.
What it refuses that runs today is a boundary move requested by *another* thread
while a pass is in flight — today that edits the cache under the forward. And it
fixes the existing wart that a refused retraction still evicted.

Growing never touches the table: new slots are free.

## 10. Porting — every ExpertCache model

The device chain moves out of `SparseMoeBlock::forward_gpu_native`
(`quantized_qwen3_moe.rs:370-625`) into one function on the cache —
`ExpertCache::forward_routed(acts, weights, indices, decode_tokens, out_dtype, wave)`,
in its own file under `expert_lre/` — which does bucketize through scatter and
§5.1. Each model supplies its router and calls it.

| Model | Router | Acts at the MoE | Change |
|---|---|---|---|
| Qwen3-30B-A3B (`quantized_qwen3_moe.rs`) | `moe_route` | Int8 | `forward_dynamic` → `moe_route` → `forward_routed`; host fork, `route_indices`, `forward_with_indices` deleted |
| Qwen3.5-35B-A3B, Qwen3.6-35B-A3B incl. AntiLoop/StyleTune (`qwen35/quantized_moe.rs`) | via `SparseMoeBlock` | Int8; **Float** when a LoRA forces `Int8Mode::Off` on a DeltaNet layer (`qwen35/quantized_delta_net.rs:67-71`) | Float acts are quantized with `to_dynamic(Precision, SumScale::Raw)` at entry — the same step the host path's Float-KO arm takes (`compute.rs:523-587`). Quantization is per row and per 128-tile, so quantize-then-gather equals gather-then-quantize bit for bit |
| Qwen3.8-Flash-Next incl. MTP head (`qwen4exp/wave.rs:3370`, `draft.rs:467`) | via `SparseMoeBlock` | Int8 | none beyond the shared block; the head row is an ordinary row. Unblocked by §3.1's `hidden % 1024` removal |
| DeepSeek-V4-Flash (`latent_moe/engine.rs:505-586`) | `Gate::route` | Int8 (Performance quantize, `:529`) | `indices.to_vec2`, the host sort and `submit_moe_work` → `forward_routed(q8, weights, indices, …)`; the readback-budget test drops the routing term (`latent_moe/wave.rs:3413-3424`) |

`ExpertCache::new` already refuses `Int8Mode::Off` (`handle.rs:604-611`), so no
cache-backed model can reach the MoE with Float acts except the LoRA case above.

Every routing constraint holds for all four: experts 128 / 256 / 512 / 256,
top-k 8 / 8 / 10 / 6, hidden 2048 / 2048 / 2560 / 4096, expert widths that
satisfy N % 32 and K % 128, KO weights throughout (MXFP4_KO for DeepSeek, which
the int8 grouped kernel serves, `dispatcher.cu:1517-1563`).

**Routing capture** (`routing_capture`, `quantized_qwen3_moe.rs:819-831`) becomes
an observer on the one path: when enabled it reads the route's indices and
weights back after `moe_route` and records them. The compute path is identical
with it on or off; the device fork's refusal under capture (`:286`) goes.

## 11. WDDM

All three dev machines run WDDM (`docs/performance.md:215-224`), which batches
submissions and drains at sync points (`docs/deepseek/deepseek_decode_cuda_graphs.md:34-38`).
Today the per-layer synchronize is such a point; this design removes it. So the
two hand-offs flush explicitly with a non-blocking query: the forward thread
after enqueuing the summary copy (§5.1 step 3), the pipeline thread after
enqueuing fills (§6.1 step 6) and after an abort. Each is one driver call per
layer or per miss batch.

## 12. What is deleted

- `expert_lre/handle.rs`: `PipelineMode::Inline`, `submit_inline`,
  `new_prepopulated`, `submit_moe_work`, `classify_and_load` / `with_inner` /
  `wait_for_copies` / `is_threaded` / `gpu_dispatch()` (no external callers),
  `send_hint`, `set_prev_layer_experts` / `get_prev_layer_experts`,
  `routing_pinned_ptr` and `PinnedRoutingBuffer`, the `live_gpu_dispatch` chain.
- `expert_lre/compute.rs`: `compute_experts_grouped` (all three arms),
  `compute_expert_contribution_gpu_weights`. `extract_weight_info` stays.
- `expert_lre/types.rs`: `MoeInput`, `MoeWorkRequest`, `ClassifiedExperts`,
  `PipelineMessage::{Work, Hint}`; `CopyBatchFence` loses `wait` (it stays as the
  `is_complete` probe for late-load accounting, which no longer gates compute).
- `expert_lre/assignment_sort.rs` and its tests.
- `expert_lre/pipeline.rs`: the compute half of `process_request`,
  `process_hint`, and the compute-side `fence.wait` calls in
  `wait_prefetch_fences_through` / `drain_prefetch_fences` / the classify fence
  (edge 3). `order_copies_after_compute` stays and takes the event to wait on
  as an argument (edge 1); `order_compute_after_copies` stays as is (edge 5).
- `gpu_dispatch.rs`: the all-resident and complete-grid requirements,
  `zone_moved` and the concede-epoch refusal, `hidden % 1024`.
- `quantized_qwen3_moe.rs`: `route_indices`, `forward_with_indices`, the host
  fork, and the reader path `from_gguf` (`:1428-1651`), which has no callers.
- `streamer.rs`: nothing. Its `after` field carries `clears_done` instead of a
  fresh compute event (edge 2) and its jobs carry their fill copies; the thread,
  its stream, staging and fence are unchanged.
- Profile spans that measured the round trip: `fwd_routing_wait`,
  `fwd_cpu_assign`, `submit_roundtrip`, `submit_inbound`, `submit_outbound`,
  `pipe_worker_total`, `pipe_compute_experts`, `pipe_fence_wait`, `pipe_hint*`;
  DeepSeek's `moe:sort`, `moe:sort_readback`, `moe:submit`. New:
  `pipe_routed_wait` (`C(n)`), `pipe_fill`.

Kept: host-table `grouped_qmatmul` (`latent_moe/attention.rs:482` and the kernel
tests use it), `grouped_matmul_gemx` and the float `fused_moe_gather` (core
tests), `moe_bucketize`, `fused_deterministic_scatter`.

## 13. What else changes

- **Stats settle.** Counters the scheduler reads right after a forward
  (`admit_ground.rs:504-531`, `scheduler/decode.rs:57-83`, `prefill.rs:1144-1157`,
  `WeightPlan::from_stats`) were complete when the forward returned because the
  forward waited for the pipeline. `ExpertCache::expert_stats()` now first sends a
  `Settle` message and waits for its answer, which the FIFO delivers after every
  earlier `Routed`. Read after the sampler's sync, that costs one channel round
  trip. Gauges read after phase 0 are settled by the phase-0 round trip itself.
- **Forward timing.** `decode.rs:637-664` times the forward around
  `decode_forward_cobatched`, which with a non-blocking MoE measures enqueue,
  not execution, and trains the admission planner's `layer_secs` on it. It is
  timed through the sampler's synchronize instead. (This is already wrong on the
  all-resident device path.)
- **Shared-expert order.** Qwen3.5/3.6/Flash-Next enqueue the shared expert
  before the router (`qwen35/quantized_moe.rs:136`); with no sync left between
  them it no longer matters where it sits, so it stays.
- **CLAUDE.md hot-path invariant 3** sanctions the MoE routing readback
  "because the streaming ExpertCache schedules pinned→VRAM uploads by expert
  id". It becomes: the MoE routing *summary* goes to the host on a side stream
  and **nothing on the device waits for the host to read it**.
- **Docs and comments** — the full audit is §17.
- **Profiling** — §16.
- **`SlotIntegrity`** (`tensor-assert` only) fingerprints a grid that no longer
  stands still. Its baseline for a slot is taken from the host record it was
  loaded from, and its checks — each a stream synchronize — run with the pass
  closed, at phase 0.

## 14. Tests (written first)

Kernel, `candle-core/src/quantized/cuda_tests.rs`:

1. **bucketize resident-first** — `bucketize_ref` gains the residency input and
   the two-pass tile order; `tile_expert` / `tile_b_start` / `tile_b_cnt` compared
   exactly, and `tok_ids` / `perm` / `rw_ids` / `token_starts` unchanged against
   the all-resident run of the same routing.
2. **bucketize summary** — raw expected `count | decode << 31` words for a fixed
   routing with a known decode split, sentinels included.
3. **device GEMM order-invariance** — the existing
   `cuda_grouped_qmatmul_dev_matches_host_tables` (`:9659-9772`) over a
   resident-first table set, bit-exact against the ascending order.

Integration, each in its own test binary because a `__trap` poisons the context
for the process:

4. **spin released by a copy** — a gate launch over a table whose entries are
   zero, released by fills on a non-blocking stream after a host delay; output
   bit-exact against the pre-filled launch; no synchronize issued between launch
   and fills (this is also the WDDM flush test).
5. **abort traps** — the abort word set while a launch spins; the next
   synchronize returns the sticky error.
6. **ordering stress** — a free-running forward over a cache sized to the floor
   with demand and speculative eviction on every layer, run for minutes with
   `readonly_regions` armed over resident slots; it is the instrument for Rules
   D, S, P and J and must run at production throughput (CLAUDE.md, the fence
   that suppresses the race it hunts).

Pipeline, `expert_lre`:

7. Rule S window: victims never at or ahead of the row, no wrap.
8. Rule P: a speculative eviction is skipped when the pass has advanced; the
   forward side receives the event otherwise.
9. Boundary refusal at entry: nothing evicted when refused.
10. Edge table (§6.2): one test per re-pointed edge asserting the event a load
    batch, a stream plan and a join order against — `e_route(n)`,
    `clears_done`, the plan fence on the copy stream — so a later change that
    re-introduces a fresh compute event fails a test rather than a gate.
11. The §18 streamer race: a stream plan whose last job is still in flight at
    join, a deficit that selects that job's slot, `readonly_regions` armed over
    it — the copy stream's write must be ordered after the streamer's.

Model gates: the sweep (`.claude/skills/sweep`) — every gate and both engine
probes, story intact, on the 3090 (paged for every MoE model) and the 4090
Mobile; plus the existing bitwise repeatability gates
(`qwen3_moe_decode_replay_is_bitwise_repeatable`, `quantized_qwen3_moe.rs:2771`),
which must stay bitwise.

## 15. Acceptance

- Every gate and engine probe green on the 3090 and the 4090 Mobile.
- Decode output bit-identical to the current build on the same residency
  (tile order is the only change in the arithmetic path, and §4.3 shows it is
  invisible).
- Flash-Next single-stream warm decode and ×16 aggregate decode on the 3090
  against the 2026-10-04 baseline (64.4 / 234.6 t/s,
  `docs/results/baseline_rtx_3090_24gb_2026-10-04.md`), and `fwd_routing_wait`
  gone from the profile.
- The gate-launch `row_fast = 1` choice measured against `grouped_grid_row_fast`
  at decode width on an all-resident cache (the 72 GB card), where it is the only
  change.
- **Proof the stall is gone, by direct measurement rather than by a missing
  span:** a `gpu_span` from each row's bucketize to its gate GEMM's start,
  and from the gate GEMM's start to its end, on the 3090 at ×1 and ×16. On an
  all-hit row the first must be the gather's duration and the second the
  GEMM's; on a miss row the second carries the miss service time and nothing
  else. The pass-start wait (Rule P) gets its own span, expected ≈ 0.
- The forward thread's per-layer enqueue time (host) against the GPU's
  per-layer time at ×1 decode on an all-resident cache — the configuration the
  reverted −8% experiment (§5.1) would show up in, since there was no stall
  there to remove.
- Expert hit rate on rows 2–4 against the current build (the Rule S wrap
  loss, §6.5); the refinement there is applied only if this moves.

## 16. Profiling — stall versus overlap

After this change the copies run in parallel with inference, and a profile has
to say which of three things moved when decode gets slower or faster:

1. **The GPU waited on a copy** — streaming did not keep up (hit rate, prefetch
   lead, bus, NVMe).
2. **The copies ran but were hidden** — streaming is working; the bus is busy
   and costs nothing.
3. **The expert kernels themselves got slower** — nothing to do with streaming.

Today's `profile` feature cannot separate these. Its GPU spans time only the
null stream (`profile/gpu.rs`, `gpu_span(name, &Device)` → `dev.cuda_stream()`),
every copy-stream event is `DISABLE_TIMING`, `dma_h2d` times the host enqueue of
an async copy rather than the transfer (`pipeline.rs:1010, 1057`), no kernel
records time or counters, and `moe:gate_up` (`quantized_qwen3_moe.rs:473-532`)
merges the one GEMM that will wait with one that never does.

### 17.1 What is measured

**M1 — miss stall, measured inside the gate GEMM.** `MoeWait` (§4.2) gains a
pointer to a per-row device counter block, non-null only in `profile` builds,
so the kernel code is the same in both and the cost off is one null check.
Thread 0 reads `%globaltimer` around its wait; a block that waited adds to:

| counter (per row, u64, monotonic) | meaning |
|---|---|
| `stall_ns` | per launch, the **longest** wait of any block, summed over launches — the delay the stream saw, since the launch cannot end before its longest-waiting block |
| `wait_block_ns` | every waiting block's wait, summed — how much of the GPU sat spinning |
| `wait_tiles` | tiles that waited at all |
| `stalled_launches` / `launches` | launches with any wait, and all gate launches |

The per-launch maximum is folded by the **last block to finish**, the pattern
the split-K path already uses (`kernel.cuh:2335-2341, 2447-2449`): each block
`atomicMax`es into a launch scratch word and increments a done counter, and the
block that brings the counter to `gridDim` adds the maximum into `stall_ns[row]`
and resets both words itself. One scratch pair per device suffices because
launches on the null stream do not overlap; a sliced gate (§4.4) folds per
slice, and since slices run in sequence the sum is still the stream's delay. No
reset kernel and no memset — both would be unschedulable under spinning blocks
(§3.3) — and the counters are never reset: the host subtracts readings.

`stall_ns` is an upper bound on the wall time waiting cost: a block whose wait
ends while other blocks of the same launch are still computing cost less than
its wait. It is exact for the case that matters, a launch held open by its
misses.

**M2 — copy busy time, measured on the copy engine.** Timing-enabled event
pairs on the stream that carries the bytes:

| span | stream | brackets |
|---|---|---|
| `copy:demand` | copy | one message's clears through its last fill |
| `copy:prefetch` | copy | one speculative prefetch batch |
| `copy:stream` | streamer | one whole-layer plan, `after` wait excluded |

with the bytes moved counted alongside (`demand_bytes`, `prefetch_bytes`,
`stream_bytes` in `PipelineStats`), so achieved GB/s per kind is a division,
not an estimate. This is the first direct measurement of bus time in the tree;
`adapt_prefetch_depth`'s `achieved_gbps` divides bytes by a host-clock *pass*
duration (`pipeline.rs:2348-2378`) and is never reported.

**M3 — overlap, derived.** `copy hidden % = 1 − stall_ns / (demand + prefetch +
stream busy)`. A copy the GPU did not wait for was hidden; every stall is a copy
it did wait for (a speculative copy that landed late shows up as a stall on the
row that needed it). No timeline reconstruction is needed for this number.

**M4 — expert kernel time without the stall.** `moe:gate_up` splits into
`moe:gate` and `moe:up`. The table reports `moe:gate − stall_ns` as the gate's
own time. `moe:up`, `moe:silu`, `moe:down` and `moe:scatter` never wait, which
makes them the control: if they slowed too, the kernels slowed; if only the gate
moved and by the stall, streaming did.

**M5 — miss service latency.** For a message with misses, a cross-stream span
`moe:service` from `e_route(n)` (null stream) to the event after its last gate
fill (copy stream) — the GPU-clock time from "routing known" to "the last missing
expert is computable". `cuEventElapsedTime` accepts two timing-enabled events on
different streams of one context. Its host-side breakdown, all on the pipeline
thread: `pipe_inbound` (send → pickup), `pipe_routed_wait` (`C(n)`),
`cl_classify`, `cold_acquire`, `cold_read`, `pipe_issue` (enqueue of clears,
loads, fills), `stream_join`.

**M6 — pass-start wait.** `moe:pass_wait`, a null-stream span around Rule P's
wait on `clears_done`. Expected ≈ 0 (§6.5); non-zero says speculative clears
are landing late.

`e_route(n)` and the copy-side bracket events are timing-enabled only in
`profile` builds; the fences are otherwise unchanged (`record_event(None)`).

### 17.2 Profile plumbing

- **`gpu_span_on(name, &Arc<CudaStream>)`** and a cross-stream
  `gpu_span_between(name, start: &CudaEvent, stop_stream)` in `profile/gpu.rs`.
  The event-pair pool is already stream-agnostic (`open`/`close` take a stream);
  only the public API is device-only.
- **The pipeline and streamer threads drain their own pools** — the pool is per
  thread — non-blocking after each message and each plan, blocking on `Settle`.
- **The pipeline thread records into the process-wide store** (`pipeline_record`)
  instead of its private `ProfileAccumulator` (`pipeline.rs:1473`), so `pipe_*`,
  `cl_*`, `cold_*`, `copy:*` reach `pipeline_snapshot*`, `/v1/profile`
  (`zend/src/api/profile.rs`) and `kv_fragmentation`'s tables, which today never
  see them. `ExpertCache::forward_profile` and `record_profile`'s second write go
  (it double-counts every forward-thread span, `handle.rs:1916-1922`).
- **`snapshot_profiles`** becomes: `Settle` (the pipeline thread drains its spans
  blockingly, reads the M1 counters with an async copy on the routing stream it
  waits for, and returns their deltas as entries). **`Qwen4ExpBatched` overrides
  it**; today it inherits the empty default (`batched_inference.rs:5464`), so
  Flash-Next's pipeline spans appear in no table at all.
- **`ProfileAccumulator` / `ProfileSnapshot` gain `max` and a `device` flag.**
  Miss stalls are a tail; a total and a count hide it. The device flag is set by
  the GPU drain, which replaces `kv_fragmentation`'s hand-maintained
  `GPU_SPANS` list (`candle-conversation/tests/kv_fragmentation.rs:303-350`).
- **`HIGH_WATER`.** With no per-layer sync, a forward opens on the order of a
  thousand GPU spans before anything drains them, and the high-water mark of
  4096 (`gpu.rs:80, 159-164`) would block the forward thread mid-forward in a
  profile build — perturbing exactly the run-ahead being measured. It is raised
  to cover several forwards, the scheduler's per-wave non-blocking drain
  (`scheduler/run.rs:687`) keeps it down, and a blocking drain that does happen
  records `profile:drain_block` so a self-perturbed profile says so.
- **Printers that skip `gpu_drain_blocking`** — the `decode_profile` test
  (`quantized_qwen3_moe.rs:3483-3612`) and the DeepSeek wave printers
  (`latent_moe/wave.rs:4011, 4204`) — call it, or they report no GPU spans.

### 17.3 What is printed

A new table in `batch_test/utils.rs` beside `"=== Expert Pipeline Stats ==="`,
one column per config as that table has, printed by `validate_and_print`:

```text
=== MoE Streaming ===
Metric                          #1 BF16×1   #2 C10×16 …
Miss stall (GPU)                 ms/tok, % of decode step
Stalled launches                 n / launches
Wait block-time                  ms/tok
Gate GEMM excl. stall            ms/call
Up / down GEMM (control)         ms/call
Copy busy — demand               ms/tok  @ GB/s
Copy busy — prefetch             ms/tok  @ GB/s
Copy busy — stream               ms/tok  @ GB/s
Copy hidden                      %
Miss service (GPU)               avg / max ms
  of which routed wait           avg / max ms
  of which cold read             avg / max ms
Pass-start wait                  ms/tok
Late loads                       n
```

`kv_fragmentation`'s `print_pipeline_profile` and `/v1/profile` get the same
entries through the snapshot (the derived rows as computed entries), and a
per-row detail — the rows with the most `stall_ns`, from the per-row counters —
follows the table when any row stalled.

`"=== Expert Pipeline Stats ==="` (`batch_test/utils.rs:2791-2968`) changes with
the path: **"MoE dispatch"** (`device` / `host (readback)`) goes — there is one
path; **"Hint loads"** goes with the hint; **"Work requests"** becomes
**"Routed messages"**; **"Fence stalls"** goes — it counted "a load batch had an
event" (`pipeline.rs:2547-2553`), never a stall, and M1's stalled launches are
the real figure. **"Hit rate"** becomes correct: `classify_and_load` returns
before its telemetry on a layer with no misses (`pipeline.rs:1616-1622` vs
`1733-1736`), so today all-hit layers are not counted and the hit rate — which
also trains the admission planner (`admit_ground.rs:504-529`) — is biased low.
Counting from the routing summary fixes that by construction.

### 17.4 Reading it

| What moved | Means |
|---|---|
| stall ↑; gate excl. stall, up, down flat | streaming fell behind — look at hit rate, late loads, copy GB/s, cold reads |
| stall flat; gate excl. stall and up/down ↑ | the expert kernels got slower |
| copy GB/s ↓ | the bus or the source — `cold_read` and `cold_acquire` say which |
| miss service ↑, copy busy flat | the pipeline thread's host side — `pipe_inbound`, `pipe_routed_wait`, `cl_classify` |
| copy hidden ↓ with stall ↑ at equal hit rate | prefetch lead time shrank — the run-ahead is outpacing the predictor |
| pass-start wait > 0 | Rule P clears are landing late |

### 17.5 Tests

- Kernel: a gate launch whose fills are released after a known host delay —
  `wait_tiles` and `stalled_launches` exact, `stall_ns` at least the delay;
  a pre-filled launch leaves every counter but `launches` unchanged.
- `ProfileAccumulator` `max` and `device` flag: raw expected entries.
- `gpu_span_on` on a non-null stream and `gpu_span_between` across two streams
  return a duration bounded below by a timed copy.

## 17. Documentation to update

Audited across `docs/`, `CLAUDE.md`, the skills, the READMEs and every Rust and
CUDA doc comment. **A** = states behaviour this change makes false; rewrite with
the code. **B** = a historical record; leave it, add a superseded-by pointer
where noted. **C** = paper text; flagged, not edited here. Comments sitting on
code §12 deletes go with that code and are not repeated.

### 18.1 Repository guidance (A)

- **`CLAUDE.md`** — invariant 3 (lines 193-196) and invariant 4's "the two
  transfers in #3" (line 200), per §13; invariant 7's MoE-table example
  (lines 232-237) cites the concession count this design deletes — keep the
  rule, replace the example with clear-before-re-tenant; the `SlotIntegrity`
  row of the tensor-assert table (§13).
- **`.claude/skills/bed/SKILL.md:83-85`** — "two sanctioned readbacks": mirror
  invariant 3.
- **`.claude/skills/sweep/SKILL.md:161-169`** — says the `qwen4exp` wave loop
  never calls `reclaim_spare_ground()`; it does (`qwen4exp/wave.rs:2264`), and
  §9 relies on it. Already false today.
- **`candle-transformers/README.md:82-93, 134`** and
  **`candle-kernels/README.md:87-88`** — "two modes — threaded … and inline";
  links to `docs/gpu_native_moe_dispatch.md`, which has moved to `archived/`.
  Point at this document.
- **`docs/README.md:65`** — the `gpu_native_moe_dispatch.md` row is a broken
  link marked "Built and verified"; replace it with this document.

### 18.2 Design docs (A)

- **`docs/expert_cache_design.md`** — §5.7 line 376 ("in front of a forward that
  is waiting for it"), §6.3 line 479 ("the routing buffer" → the summary ring),
  §12.7 lines 890-892 (`post_compute`'s response plumbing).
- **`docs/vram_span_partition.md`** — §4 lines 205-210 and §7 rule 5
  lines 475-476: the cached-slot-address rule and concession count.
- **`docs/deepseek/deepseek_hot_path_invariants.md`** (authoritative per
  CLAUDE.md) — lines 243-247, 258-260, 276-281, and the invariant 3/4 tables at
  348, 349, 360: the routing readback is no longer sanctioned-and-load-bearing.
- **`docs/deepseek/deepseek_decode_cuda_graphs.md`** — §2 lines 66-81, §3 94,
  §4 118-120, §5.2 169-176, §5.4 226, §6 270-274, §7 289-290: its segment A / B
  split is built on the readback and the pipeline round trip. With both gone the
  whole MoE layer is one capturable segment, which is good news for that design
  and needs saying in it.
- **`docs/deepseek/deepseek_decode_launch_overhead.md:147-163`** — "why the
  readback cannot simply be removed". The rest of the doc is a measurement
  record (B), including the −8% lesson §5.1 answers.
- **`docs/deepseek/deepseek_hot_path_optimization_findings.md`** — lines 49-51,
  264-267 ("routing must be host-visible").
- **`docs/deepseek/deepseek_v4_speculative_decode.md`** — §1 52-59, §4.1 168-172,
  §4.2 208-209, §7 433-437: the "routing readback is the WDDM wall" premise.
- **`docs/deepseek/deepseek_v4_flash.md:486-499`** and
  **`docs/deepseek/deepseek_batched_paged_attention_plan.md:446, 535, 685`** —
  lower priority; the first is partly stale already.

### 18.3 Historical (B)

`docs/wave_feeder.md:1126-1128`; `docs/unified_wave_inference_engine.md` §3.3;
`docs/deepseek/deepseek_perf_optimization_report.md:78-83, 310-319` (its
"`gpu_dispatch` today requires `all_resident`" is exactly the gap this closes —
add a pointer); `docs/deepseek/deepseek_decode_reproducibility.md:38-40,
171-178`; the "MoE dispatch … host (readback)" rows of
`docs/results/performance_rtx_3090_24gb_rows.tsv`; and in `docs/archived/`:
`gpu_native_moe_dispatch.md` (**add superseded-by this document** — its Phase B
is this design), `expert_pipeline_dataflow.md` (still cited by
`expert_lre/mod.rs:9`), `qwen38_flash_next.md`, `qwen36_prefill_profile.md`,
`qwen35_qwen38_models.md` (incl. the `hidden % 1024` row),
`tiered_cache_paper.md`, `arena_unification_results.md`,
`elastic_vram_partition.md`, and single lines in `npc_api_gui_design.md`,
`overnight_int8_and_kernels_report.md`, `qwen36_performance_plan.md`.

### 18.4 Paper (C)

- **`docs/unbounded_agents.md` (v2, in preparation)** — abstract line 11 and §1
  line 72 claim stall-free / zero-idle MoE inference under partial residency;
  the current profile contradicts that, and this design is what makes it closer
  to true — the claim should be re-measured, not assumed. §4.1 line 172; §4.2
  line 191 and §4.3 line 207 ("speculative prefetch never evicts" — already
  untrue, and Rule S changes it again); §4.4 lines 225-238, the "Phase 1 / Fence
  / Phase 2 / Join" dispatch and "the CPU never blocks", which this design
  replaces with per-tile waits.
- **`docs/unbounded_agents_v1.md`** — the same passages (lines 15, 73, §4.1,
  §4.2, §4.4 at 222-235). Published and served; an erratum is the only route.

### 18.5 Code comments that survive the change (A)

- **`expert_lre/`**: `mod.rs` module doc (1-32, 99-106 wrap, 137-175 modes and
  tables, link at 9); `handle.rs` (1-9, 102, 430-480, 1021-1034, 1755-1799);
  `gpu_dispatch.rs` (1-17, 60-130, 155-165, 236-241, 341-362, 443-447, 662-669,
  and the "fell to the host path" test messages at 842-951); `zone_geometry.rs`
  (whole module doc — it exists for the deleted `zone_moved`; either it goes or
  it is re-scoped); `cache.rs` (114-119, 213-222, 259-277, 485-505); `types.rs`
  (62-63, 86-87, 117-131, 257-258, 270-289, 428-460); `pipeline.rs` (1-6, 117,
  164-170, 1402-1458, 1814-1825, 2250, 3289-3318, 3498-3511); `streamer.rs`
  (4-11, 24-34: protocol now `clears_done`, joined on the routed message);
  `compute.rs` module doc; `slot_integrity.rs:6-11`; `weight_plan.rs:9-13`.
- **Models**: `quantized_qwen3_moe.rs` (216-276, 345-363 → `forward_routed`,
  592, 2197 dangling "reader path above", 3475); `qwen35/quantized_moe.rs:1-10`
  and `qwen35/expert_loader.rs:13-15`; `qwen4exp/engine.rs:20-23` (already
  wrong: "bucketize declines >256"); `qwen4exp/wave.rs:3202-3207, 3359-3365`;
  `latent_moe/readback.rs:1-9`, `engine.rs:498-501`, `wave.rs:929-930,
  1044-1047, 1089-1090, 2953-2956` and the readback-budget test 3413-3424;
  `latent_moe/dspark_experts.rs:1-43, 212-216, 335` (lower priority);
  `batched_inference.rs:4743, 5490-5491` and
  `candle-conversation/src/models/builder.rs:257-258` (fixed cost attributed
  to the routing readback).
- **Elsewhere**: `candle-conversation/src/scheduler/decode.rs:639-658` (forward
  timing and "counters this forward just moved", §13);
  `candle-nn/src/kv_cache/chunked/region_pool.rs:40-45` (when `W` moves, §9);
  `candle-core/src/quantized/cuda.rs` 5567-5571, 5760-5762, 6669-6684 (device
  GEMM: "every expert VRAM-resident", "same ascending-expert tile order"),
  7490-7498 (bucketize); `candle-kernels/src/simple/moe_bucketize.cu` 4-9,
  34-45, 158-159; `batch_test/utils.rs:2856-2867` (the "MoE dispatch" row,
  §16.3).

## 18. Open questions

- **Q1 — scope.** "Every MoE model" here means every `ExpertCache` model: Qwen3-30B,
  3.5, 3.6 (and variants), Flash-Next, DeepSeek-V4. Two MoE forwards in the tree
  never used this path and are not touched: Mixtral through `quantized_llama.rs`
  (`MlpOrMoe`, its own CPU routing at `:106-175`, non-KO experts, no cache), and
  the upstream float models (`mixtral.rs`, `qwen2_moe.rs`, `qwen3_moe.rs`,
  `deepseek2.rs`, `granitemoehybrid`). Bringing Mixtral onto the cache is a KO
  repack plus a cache build — a port of its own.
- **Q2 — boundary moves from other threads during a pass** are refused (§9).
  Whether any caller depends on them succeeding today needs one look at
  `claim_region`'s refusal handling in the persistence path.
- **Q3 — end-of-pass decay** fires only on draft steps for models with an MTP
  head (§6.4). This design preserves it; it reads like a defect.
- **Q4 — `GATE_SLICE_BYTES` and `SPIN_LIMIT_NS`** need a measured cold-read rate
  per machine to size. The slicing is the mechanism; the numbers are a
  measurement.
- **Q5 — "cannot evict (all pinned)" at the floor** (§6.3) is an existing edge
  that becomes fatal rather than a failed wave. Closing it touches the
  copy/streamer protection, which this change deliberately leaves alone.

**A probable existing copy race, closed by edge 4.** `join_stream_for` removes
the joined layer's installs from `stream_loads` (`pipeline.rs:2291`) *before*
classify computes its protect set (`:1575-1589`), and `join` waits only for the
streamer to finish *enqueueing* (`:2250-2262`), not for its copies to land. A
streamed expert of the current row that the route did not select is then
unprotected, and `demand_eviction` admits the current row at factor 1.0
(`cache.rs:895-903`, distance 0). Its re-tenant goes out on the copy stream
ordered only after a fresh compute event — and the compute stream has not yet
waited this plan's fence (that wait is at `pipeline.rs:2498`, after classify).
So the new tenant's bytes and the streamer's still-landing bytes can hit the same
slot unordered. It needs a prefill-width wave whose stream plan mispredicted an
expert and a deficit that picks exactly that slot, which fits the
"intermittent, wide-wave only" shape of past expert corruption. Not reproduced;
found by reading. Edge 4 orders the copy stream after the plan before classify,
which closes it, and it is worth a regression test of its own (§14, test 11).

Pre-existing defects found on the way and out of this scope: `router_topk`
indexes out of bounds when every score is `-inf` (`router_topk.cu:104-123`);
DeepSeek's routed experts skip the `swiglu_limit` clamp the reference applies
(`latent_moe/moe.rs:65-89`); the Float non-KO arm of `compute_experts_grouped`
cannot have worked for BF16 activations (the scatter requires F32,
`cuda.rs:7348`) — moot once deleted.
