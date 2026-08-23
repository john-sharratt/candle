# CUDA Graphs for the DeepSeek Decode Wave

Design document. Scope: the **decode** wave (plain decode + speculative verify)
of `BatchedEngine::forward_wave`. Prefill is explicitly out of scope (its
shapes are chaotic per wave and it already sits near its throughput target);
anything a decode graph happens to share with prefill is a bonus, not a goal.

---

## 1. Why: the measured wall

Decode on this box is **host-launch-bound by ~8×**, not GPU-bound. nsys over
one decode-heavy gate (the counting test):

| measurement | value |
|---|---|
| kernel launches | 427,956 |
| GPU busy time (all kernels) | 3.7 s |
| host time in `cudaLaunchKernel` | **28.7 s** (67 µs avg/launch — the WDDM submission tax; TCC/Linux is 3–5 µs) |
| launches that are elementwise glue | 43% (doing 9% of the GPU work) |
| kernels per MoE layer-step | ~63 (427,956 / 6,786 layer-steps) |

Two structural facts follow:

1. **Per-op fixes have a hard ceiling.** The fused router epilogue + batched
   decode rope removed ~20 launches per layer-step and bought +11–14% decode.
   The remaining ~45 launches per layer are the *useful* work — norms, GEMMs,
   quantizes, select, gather, attention, combine — each individually
   irreducible. Only collapsing their **submission** (one graph launch instead
   of ~45) attacks the remaining 60+ µs × 45 × 43 layers ≈ **120 ms of host
   submission per decode step**, against a plain-wave wall of ~50–105 ms
   (submission overlaps GPU work partially, so the realizable win is a large
   fraction, not all, of that number).
2. **Host profile spans cannot steer this work.** Under WDDM the submission
   queue drains at sync points, so enqueue-only spans lie about where time
   goes (measured: the `dprep:push` "38%" span was queue-drain, and two
   batching designs against it both regressed). Every claim in this document
   is grounded in API counters (launch counts) or like-for-like profiled
   t/s, and every phase gates the same way.

---

## 2. Anatomy of one decode layer (launch inventory)

Per layer `l`, per decode wave of `n_dec` rows, in wave-thread program order
(`wave.rs` decode block):

```
── SEGMENT A — compute stream, wave thread, ~40–45 launches ──────────────────
 hc_pre            mhc_pre_gates + mhc_pre_reduce + sinkhorn        ~3
 attn norm         rmsnorm                                           1
 projections       shared_int8_pair (quantize + wq_a + wkv GEMMs),
                   q_norm, wq_b (quantize+GEMM), rms_scale,
                   bf16 casts ×2                                    ~9
 compressor proj   project_rows ×2 families (2 GEMM pairs + quant)  ~6
 indexer query     query_gemm_batched (2 GEMMs) + rope_query_at     ~8
 corpus push       Vec pushes (host-only); every `ratio`-th row
                   emits: cat/ape/pool chain                        0–10
 select            sign_pack + bdp_recall_batched + topm_select
                   + padded bmm rescore + argsort + gather          ~6–8
 gather            corpus gather kernel (or arange for HCA)         1–2
 attention         latent_decode + latent_combine                    2
 out-proj          quantize + grouped int8 GEMM                      2
 hc_post_attn      mhc_post                                          1
 ffn norm + route  rmsnorm + gate GEMM (quantize+GEMM) + router_topk ~4
── SEGMENT BOUNDARY — the sanctioned readback ────────────────────────────────
 indices.to_vec2   DtoH sync (the ONE decode readback; hot-path
                   invariant #3a) + host counting-sort (µs)
── SEGMENT B — pipeline thread round-trip ────────────────────────────────────
 submit_moe_work   classify (host) + expert H2D DMA (copy stream,
                   data-dependent misses) + q8a128 quantize +
                   mxfp4_ko grouped GEMM (weight ptrs via TableRing)
                   + deterministic scatter + combine               ~8–12
 shared expert     3 GEMMs + activation (wave thread, after submit) ~6
 hc_post           mhc_post                                          1
──────────────────────────────────────────────────────────────────────────────
```

No hidden syncs inside segment A: the batched select's per-session `k` comes
from host-side gallery bookkeeping (`galleries[s].len`), not a device
readback. The layer's only host-visible sync is the routing readback.

---

## 3. Dynamism taxonomy — what moves, and when it is frozen

The user-visible concern — experts move, KV moves, transients move — resolves
into parameter classes with very different graph implications:

| class | examples | changes | graph implication |
|---|---|---|---|
| **Static per run** | dense weights, rope tables, sinks, `LatentWorkspace`, layer geometry | never | baked into each layer's graph |
| **Frozen per wave** | KV slot headers / chunk tables (the forward-entry reconciler freezes offsets and the header snapshot is taken at wave entry), wave-transient claims | at wave boundary | safe to bake **per capture**, refreshed by re-capture/update — this is the property that makes per-wave graphs possible at all |
| **Per layer, behind indirection** | expert slot pointers (move with eviction) | per layer | **already graph-proof**: the grouped GEMM reads its weight pointers from the TableRing's device-visible pinned table; re-pointing the table re-points the same captured kernel. The all-resident dispatch (`GpuDispatchTables`) is the same shape. |
| **Host-known per wave** | `n_dec`, per-slot select counts `cnt[i]` (from gallery lens + `top_k`), gather total `total_k`, emit phase `l0` | per wave, but computable before any launch | these set tensor SHAPES → they define the graph cache key (§5.3) |
| **Data-dependent** | expert misses → H2D DMA set; draft acceptance → verify block lengths | per wave, only knowable from device results | stays OUTSIDE graphs (segment B / eager verify in phase 1) |

Two accommodations this codebase already built, which the design leans on:

- **TableRing** (`candle-core/src/quantized/table_ring.rs`): device-mapped
  pinned launch tables, written by the host, read by the kernel. A captured
  kernel that consumes a table pointer is *re-targeted by writing the table*,
  no graph update needed.
- **Bound-grid + device-count kernels**: `moe_bucketize` pads its tile tables
  to the `n_tokens × k` bound with `b_cnt = 0` early-out tiles precisely so
  the host needs no data-dependent grid; the latent decode kernel reads
  per-slot `cnt/offset` from device tensors with `grid.x = n_dec`. Kernels of
  this shape are stable topology under a fixed `n_dec` regardless of what the
  counts do — this is the "argument-based" property, and §7 Phase 2 extends
  it to the select/gather block.

---

## 4. Graph-breaker audit

| breaker | where | disposition |
|---|---|---|
| Routing readback (`indices.to_vec2`) | engine.rs `moe_forward_batch` | Segment boundary. Stays eager (sanctioned readback). Splits the layer into A / B. |
| Pipeline-thread launches | `submit_moe_work` round-trip: classify + DMA + grouped GEMM run on the expert pipeline thread | Segment B stays eager in phases 1–2. Phase 3 restructures: the GEMM/scatter/combine launches move to the wave thread (post-sort), leaving only classify + DMA on the pipe thread — then B is capturable, with expert pointers already behind the TableRing. |
| Expert H2D DMA | copy stream, miss-dependent | Never in-graph. Already fenced (`CopyBatchFence` ring); the graph's kernels consume slots whose fences were waited before launch, unchanged. |
| Per-op `cuMemAllocAsync` | every eager op output | **Capturable.** candle allocates from the device DEFAULT mempool (`cuDeviceGetDefaultMemPool` is what the trim path manipulates) — stream-ordered allocation capture turns these into graph MEM_ALLOC nodes with VA-stable replays. Instantiate with `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` so each replay owns-and-frees its transients. |
| VRAM governor pool trim | `cuMemPoolTrimTo` in the relief path | **Risk**: trimming while a graph exec holds VA reservations mirrors the arena-topology trim hazard fixed once already (`cuda-fault-root-governor-trim`). Mitigation: register live graph execs with the same governor guard used for arena topology; trim invalidates (drops) cached graphs first. |
| cuBLAS calls (`gemmSN`) | lm_head/drafter F32 GEMMs | Capture-safe (workspace pre-allocated at handle creation; no internal syncs on these paths). Mostly outside the per-layer segment anyway. |
| cudarc per-arg events | everywhere | Already disabled (02369a3c). Would otherwise have put event record/wait nodes in every graph. |
| Shape variance | `n_dec` (verify blocks vary with adaptive K), gallery growth (`total_k` steps up every `ratio` rows), emit-phase `l0` cycle | The cache key (§5.3). Phase 1 restricts to plain decode (stable `n_dec`); Phase 2 buckets the corpus block so keys survive gallery growth. |
| Capture-mode illegal ops | device sync, `cuMemcpyDtoH` sync, event query inside capture | None exist inside segment A (audited §2); the capture wrapper asserts `capture_status` on exit as a tripwire. |
| Profiling spans / `pipeline_record` | host-side timers | Harmless (host code runs at capture time, not replay). NOTE: under replay the per-span timings become meaningless — the profile feature should mark graphed waves. |

---

## 5. Design: capture-and-replay per-layer segment graphs

### 5.1 Mechanism — capture the EXISTING eager code

The graph for (layer, shape-key) is built by running the **unmodified eager
enqueue path** under stream capture:

```
stream.begin_capture(ThreadLocal)
  ── existing segment-A code for layer l ──   // enqueues ~45 kernels: cheap
stream.end_capture(AUTO_FREE_ON_LAUNCH)       //  bookkeeping, no WDDM submit
→ CudaGraph { graph, exec }                   // instantiate once
```

Replay on subsequent waves with a matching key = **one** `cuGraphLaunch`.

This is deliberately NOT a hand-built graph with per-node parameter updates:
the eager path is the single source of truth, so the graph can never drift
from it (the same argument as the mirror-gate philosophy). "Argument-based"
enters through the two indirections of §3 (device tables, device counts) —
they make the captured topology insensitive to the values that move — not
through manual `cuGraphExecKernelNodeSetParams` surgery, which would couple
the graph code to every kernel's parameter struct.

Two costs, and why they are acceptable:

- **Capture cost ≈ eager enqueue cost, minus the WDDM submission.** Enqueue
  into a capturing stream is driver bookkeeping (~2–5 µs/node), not an OS
  submission (~67 µs). So even a wave that captures-and-launches (cold key)
  beats today's eager wave.
- **Instantiate cost** (~100 µs–1 ms for ~45 nodes) is paid once per (layer,
  key). Steady-state plain decode re-uses keys across steps (§5.3), so
  instantiates amortize to noise. `cuGraphExecUpdate_v2` (present in cudarc's
  sys layer; needs a thin wrapper) is the phase-2 fast path for
  topology-identical recaptures, skipping re-instantiation.

### 5.2 What is graphed, per phase

- **Phase 1**: segment A only, plain decode waves only. Everything from
  `hc_pre` through `router_topk`, ending BEFORE the readback. Segment B,
  shared expert, hc_post, and all verify waves stay eager.
- **Phase 2**: bound-shaped select/gather (below) extends key lifetimes and
  admits verify waves bucketed by block-length vector.
- **Phase 3**: segment B (grouped GEMM → scatter → combine + shared expert +
  hc_post) after the launch-site restructure; the shared expert can move
  ahead of `submit` (it is independent of routing) and join segment A sooner.

### 5.3 Graph cache and keying

```
GraphKey {
    layer: usize,
    n_dec: usize,                       // rows in the wave
    slots: Vec<SlotShape {              // per decode slot, in order
        gallery_len_bucket: u32,        // phase 1: exact len; phase 2: bucket
        sel: SelKind,                   // TwoStage / AllEntries / None
        write_chunk_epoch: u32,         // slot table generation (KV moved)
    }>,
    emit_phase: u8,                     // l0 mod ratio (push topology)
}
```

- Cache: per-layer `HashMap<GraphKey, CudaGraph>` with an LRU cap (a few
  hundred entries; each holds its transient VA reservation — the cap bounds
  that footprint and the governor guard can flush it).
- **Key miss ⇒ capture** (which is itself cheaper than today's eager wave).
  There is no "fallback path" to maintain: the eager code IS the capture
  body. A capture-unsafe situation (asserted via `capture_status`) aborts
  capture and runs the wave eagerly — same code, zero drift.
- Phase-1 reality check on key churn, plain decode at width W: `n_dec = W`
  constant; gallery lens advance in lockstep every `ratio` (=4) steps;
  `emit_phase` cycles 0..4. So the steady state cycles through ~4 keys whose
  lens advance every 4th step → 3 of 4 steps replay a cached graph, 1 of 4
  captures a new one (still cheaper than eager). Phase 2's bucketing
  (`gallery_len` rounded up to 64-group boundaries + gather block padded to
  the bucket, counts read from device) turns that into ~255 of 256 steps
  replaying.

### 5.4 Where the wave calls it

In the decode block of `forward_wave`, per layer:

```
let key = GraphKey::for_wave(l, &decode_slots, ...);   // host-only, cheap
match graph_cache.get(l, &key) {
    Some(g) => g.launch()?,                            // 1 submission
    None    => {
        stream.begin_capture(ThreadLocal)?;
        let r = enqueue_segment_a(l, ...);             // the existing code
        match stream.end_capture(AUTO_FREE)? {
            Some(g) if r.is_ok() => { g.launch()?; graph_cache.put(l, key, g); }
            _ => { /* capture aborted: the enqueue already ran eagerly */ }
        }
    }
}
// readback + host sort + segment B: unchanged
```

One subtlety the implementation must honor: segment A's HOST side effects
(compressor Vec pushes, sel capture, row snapshots) execute at capture time
and NOT at replay. Replayed waves must run the host-side state advance
separately from the device enqueue — i.e. segment A's code needs its host
mutations factored so a replay performs them without re-enqueueing. This is
the main refactor of phase 1 (and it is behavior-preserving: the same
statements, split by side-effect class). Device-visible values that change
per wave WITHIN a key (decode positions, comp-idx contents, projected rows)
are already tensor CONTENTS, not shapes — they flow into the replayed graph
through the same buffers the capture recorded, provided those inputs are
written into per-slot staging tensors the wave owns (`x` itself, `decode_pos`
upload, slot headers) rather than freshly-allocated ones. The audit confirms
the inputs at segment-A's boundary are exactly: `x` (wave input tensor),
`decode_pos` upload, slot header tables, gallery arenas — all wave-owned or
frozen; the capture-internal allocas are outputs and intermediates.

### 5.5 Memory semantics

- Captured `cuMemAllocAsync` → graph-owned allocations; with
  `AUTO_FREE_ON_LAUNCH`, each replay reallocates at the SAME virtual
  addresses (CUDA graph memory pools reserve the VA range for the exec's
  lifetime) and frees at the next launch. Kernel pointer arguments baked at
  capture stay valid across replays.
- Consequences to enforce:
  1. Segment-A outputs consumed AFTER the graph (attention out rows feeding
     the readback-side code) must be copied/landed in wave-owned buffers
     before graph end, or the consuming code must run before the next replay
     frees them. Phase 1 lands segment-A outputs into wave-plan transients
     (already the case for `attn_rows` accumulation).
  2. The governor's `cuMemPoolTrimTo` and the graph VA reservations must be
     mutually excluded (registered guard, §4).

---

## 6. Expected wins (bounded arithmetic)

Per plain-decode step, 43 layers:

- Segment A ≈ 40–45 launches/layer × 43 × ~62 µs saved submission ≈ **100–120
  ms of host submission removed per step**, replaced by 43 graph launches
  (~3 ms) + key computation (µs).
- The wall does not drop by the full amount: GPU work (~15–25 ms/step) and
  segment B (readback + MoE round-trip, ~45% of today's wall) remain. Bounded
  estimate for phase 1: plain-wave step time 50–60 ms → **~30–35 ms**
  (trailing single ~21 → ~28–33 t/s). Phase 3 (segment B graphed, readback
  remains) is what approaches the GPU-bound floor.
- Verify waves (the spec-decode path) inherit phase-2+ benefits; their walls
  are also expert-DMA-bound, so gains there are partial by design.

These are estimates to be falsified by the phase gates, not commitments.

---

## 7. Phases

| phase | contents | gate |
|---|---|---|
| **0. Infra** | candle-core: `exec_update_v2` sys wrapper; capture-abort guard (`capture_status` tripwire); graph-exec registry + governor-guard integration; graph cache scaffold | unit tests: capture→replay of a synthetic kernel chain is bit-identical across 3 replays with moved table contents |
| **1. Segment A, plain decode** | host/device side-effect split of segment A; GraphKey; cache; capture-or-replay dispatch in `forward_wave` | counting **bit-lossless**; realtext in band; sweep all-valid; trailing single measurably up; llama/qwen suites untouched (deepseek-only code) |
| **2. Bound-shaped select/gather** | bucket `total_k` (round to 64), gather kernel writes count-bound rows into the bucketed block, attention consumes device counts (already does); key drops exact lens for buckets; `exec_update_v2` fast path on topology-identical recaptures; admit verify waves keyed by block-length vector | same gates + key-churn telemetry (captures per 1k steps) |
| **3. Segment B** | move grouped-GEMM/scatter/combine launches to the wave thread post-sort; TableRing already carries expert pointers; graph B per (layer, expert-tile-bound); shared expert into segment A | same gates + expert-stats table unchanged (loads/evictions identical) |
| **4. Optional stitching** | child-graph composition of A+B per layer, or whole-wave graphs with the readback as the only break | only if 1–3 leave submission dominant |

Each phase lands independently with the standing gate discipline (counting
lossless is non-negotiable at every step; a phase that can't hold it reverts).

---

## 8. Risks

1. **Replay-vs-host-state divergence** (§5.4) — the highest-risk item: a
   host side effect left inside the captured closure silently stops
   happening on replays. Mitigation: the side-effect split is reviewed as
   its own commit; a debug assertion mode runs capture-every-wave (no
   replays) and compares outputs against replayed waves on the gates.
2. **Pool/graph VA interplay** — trim vs reservations (§4, §5.5); governor
   guard + cache flush on relief.
3. **WDDM capture quirks** — capture itself is submission-free, but the
   first replay after instantiate may compile/upload (one-time hitch);
   `cuGraphUpload` on the idle stream pre-warms.
4. **Key-space blowup on verify waves** — adaptive-K makes block-length
   vectors vary; this killed the assemble-geometry cache. Phase 2's answer
   is bucketing; the telemetry (captures/1k steps) is the tripwire, and
   verify waves stay eager until it reads low.
5. **The 16 GB card** — nothing here is card-specific; graph VA reservations
   add ~per-layer-transient × cache size, bounded by the LRU cap and priced
   into the wave plan.

---

## 9. Non-goals

- Prefill graphs (shape-chaotic; prefill is near target).
- Drafter graphs (its own launch storm is a later, separate audit).
- Manual per-node parameter update surgery (fragile against the eager path;
  the indirection tables make it unnecessary).
- Multi-GPU.
