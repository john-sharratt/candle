# Qwen3.6-35B-A3B — where the prefill time goes

**Measured 2026-08-23 on RTX PRO 5000 Blackwell 72 GB (sm_120, 110 SM), CUDA 12.4,
`nsys` 2026.1.3, `ncu` 2026.2.0.**

> **Revision note.** The first version of this document concluded that 53.6% of
> prefill was the KV-compression selection kernel. **That was wrong**, and the
> way it was wrong is worth recording: the figure was a share of *GPU-busy time*
> taken over a capture window whose config mix was dominated by C-level
> compression runs. Two things falsify it — selection is 5.6% of GPU time in the
> BF16 configs and 88.5% in the C-level configs (so it is not a constant), and
> **the GPU is idle ~50% of the wall clock**, so shares of GPU-busy time were
> never the right denominator. The corrected finding is below.

## Summary

**Prefill on this model is launch-bound, not compute-bound.** In a matched BF16
window the GPU runs at **47–50% utilisation** against Qwen3-30B-A3B's 59–72%, and
the idle is not a few large syncs — it is **199,703 inter-operation gaps
averaging 9.9 µs** across 4 seconds of wall time.

The model issues **2.6× more GPU operations than Qwen3-30B for comparable work**,
and the extra operations are small:

| 4 s BF16 window | Qwen3.6-35B | Qwen3-30B |
|---|---|---|
| GPU ops | **199,704** | 75,986 |
| GPU busy | 1.95 s (**~48%**) | 2.36 s (~59%) |
| idle gaps | 199,703 (mean 9.9 µs) | 75,985 (mean 17.4 µs) |
| int8 GEMM | **46,119 @ 23.6 µs** | 9,274 @ **101.8 µs** |
| elementwise/"other" | **107,211 @ 2.2 µs** | 28,598 @ 17.5 µs |

The GEMM line is the clearest statement of the problem: **5× the launches at
¼ the size, for the same total GEMM time.** Nothing is slower per unit of work —
the work is chopped into pieces too small to fill the machine or amortise a
launch.

## 1. What motivated this

Qwen3.6-35B-A3B passes its whole sweep (14/14 configs valid, C10 included), but:

| | BF16×1 | BF16×4 | C7×1 | C10×10 |
|---|---|---|---|---|
| t/s (bulk) | 4433.9 | 4307.8 | 4512.0 | 4303.8 |
| t/s (single) | 50.5 | 162.9 | 50.0 | 310.2 |

Single-session decode scales properly with batching (50.5 → 310.2). **Bulk
throughput does not move at all** across compression 2.21×→7.67× and contexts
1→10. Against Qwen3-30B-A3B on the same harness and machine: single decode is
10% down (50.5 vs 56.3 — DeltaNet's genuine per-token cost), bulk is **34% down**
(4434 vs 6752).

Note for interpretation: bulk t/s appears to be prefill-dominated and does not
include seal-time work, which is why it is flat even as selection cost swings
from 5.6% to 88.5% of GPU time between configs (§4).

## 2. The measurement that matters: utilisation, not GPU-time share

`nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --duration=260` over the
test binary directly, then windowed against wall time.

GPU busy per 2 s window:

```
Qwen3.6-35B                          Qwen3-30B
  16-18s   47.0%                       14-16s   59.0%
  18-20s   50.5%                       16-18s   71.8%
  20-22s   56.4%                       18-20s   53.0%
  22-24s   78.6%   <- C-levels start   20-22s   56.0%
  26-28s   89.9%                       26-28s   77.9%
```

**Any percentage taken over GPU-busy time is describing the composition of half
the wall clock.** The other half is inter-op gaps.

Idle distribution, matched BF16 windows (4 s each):

| | total idle | gaps | mean | <10 µs | 10–100 µs | 0.1–1 ms | >1 ms |
|---|---|---|---|---|---|---|---|
| Qwen3.6 (16–20 s) | 1.980 s | 199,703 | 9.9 µs | 0.407 s | **0.728 s** | 0.586 s | 0.259 s |
| Qwen3-30B (14–18 s) | 1.319 s | 75,985 | 17.4 µs | 0.180 s | 0.486 s | 0.331 s | 0.322 s |

The mass is in the 10–100 µs band — per-launch overhead, not blocking syncs.
This is the same shape as the documented WDDM launch-overhead floor.

## 3. Where the operations come from

Matched BF16 window, by **launch count** (the quantity that matters when
launch-bound):

```
Qwen3.6-35B  16-20s                    Qwen3-30B  14-18s
  TOTAL       199,704 ops                TOTAL        75,986 ops
  other       107,211  @   2.2us         other        28,598  @  17.5us
  int8 GEMM    46,119  @  23.6us         int8 GEMM     9,274  @ 101.8us
  memcpy       27,146  @   0.5us         memcpy       20,787  @   4.0us
  DeltaNet      9,531  @  37.5us         attention     2,784  @ 279.6us
  memset        7,367  @   0.7us         memset       12,267  @   0.6us
  attention     1,850  @  79.2us         KV-select     2,276  @ 133.1us
  KV-select       480  @ 224.0us
```

Top kernels by launch count in that window:

```
  31,522   0.634s   20.1us  <templated: delta_net / paged / fused_attn>
  19,130   0.178s    9.3us  q8_ko_int8_bf16_dense
  12,600   0.168s   13.3us  q8_ko_int8_f16_dense
   8,640   0.011s    1.3us  cast_f16_f32
   8,340   0.019s    2.3us  copy2d_f32
   7,894   0.018s    2.3us  bmul_bf16
   7,834   0.326s   41.6us  q4_ko_int8_f32_grouped
   5,760   0.007s    1.2us  badd_f16_inplace
   4,744   0.020s    4.1us  cast_f32_bf16
   3,977   0.005s    1.2us  usigmoid_bf16
   3,917   0.018s    4.7us  moe_gather_u8
   3,917   0.015s    3.8us  moe_route_bf16
```

### 3a. Hot-path invariant violations, by the numbers

Two CLAUDE.md invariants are being broken at scale in this window alone:

- **Invariant 1 (no `to_dtype` in the loop — kernels emit the final type):**
  `cast_f16_f32` 8,640 + `cast_f32_bf16` 4,744 = **13,384 conversion launches**.
  They cost only ~0.03 s of GPU time between them, but ~13,400 × ~10 µs of gap
  is ~0.13 s of *idle*. The GPU work is negligible; the launches are not.
- **Invariant 2 (no allocate-plus-copy to materialise a layout):**
  `copy2d_f32` 8,340 launches.

Together with `bmul_bf16` (7,894), `badd_f16_inplace` (5,760) and
`usigmoid_bf16` (3,977), that is **~35,000 launches of 1–4 µs elementwise work**
— roughly 0.07 s of GPU time bought at roughly 0.35 s of launch gap.

### 3b. The in-process profiler agrees — and names the stall

`candle-transformers` has its own counting profiler
(`models/profile.rs`: `ProfileAccumulator`, `pipeline_record`, span names) which
reports **total, count and average per named span**, per config. Build with
`--features cuda,profile`.

Prefill ("Bulk (Prompt) Profile"), abridged:

```
 Span                    #1 BF16×1     #3 Q8_0×1    #6 C10×10
 fwd_routing_wait       20.2ms (×80)  10.8ms (×80)  203.0ms (×520)
 pipe_compute_experts   32.4ms (×80)  18.5ms (×80)  153.8ms (×520)
 pipe_worker_total      48.9ms (×80)  33.2ms (×80)  267.0ms (×520)
 submit_roundtrip       50.2ms (×80)  34.3ms (×80)  275.7ms (×520)
 fwd_routing             2.8ms (×80)   2.8ms (×80)   20.0ms (×520)
 gemm_down               9.4ms (×80)   0.5ms (×80)    3.6ms (×520)
 gemm_gate               4.9ms (×80)   0.7ms (×80)    5.4ms (×520)
 pipe_fence_wait         0.0ms (×80)   0.0ms (×80)    0.1ms (×520)
```

Decode, abridged:

```
 decode:kernel          66.6ms (×100)   dn:ffn        95.3ms (×300)
 decode:qkv_proj        11.1ms (×100)   dn:proj       37.6ms (×300)
 decode:out_proj         3.0ms (×100)   dn:mix        17.8ms (×300)
 qmatmul_q8             40.7ms (×3510)  dn:out_proj   10.1ms (×300)
```

Two things fall out:

- **`fwd_routing_wait` is a named host stall on the forward thread** — 20.2 ms
  per config at BF16×1, rising to **203.0 ms** at C10×10. `pipe_fence_wait` is
  ~0, so this is not expert DMA (experts are resident); it is the forward thread
  blocking on the MoE routing round-trip. Same shape as the documented
  MoE-routing readback wall.
- **`qmatmul_q8` fires 3,510 times per config** at ~11.6 µs each. That is the
  §3c fragmentation counted by the engine itself, independent of nsys.

**The profiler was rebuilt on CUDA events** (see below). The numbers above are
from the event build.

#### The instrument itself: host sync → CUDA events

The original profiler measured GPU time by calling `device.synchronize()` at
every span boundary (`profile_sync`, 16 sites). That made each span accurate and
the run a fiction — it serialised the pipeline and, worse for this
investigation, folded the inter-op gaps of §2 into the span totals. Its own
source admitted the failure at one site:

> *"without this, everything queued and not yet awaited (on a hybrid stack,
> whole DeltaNet layers) drains inside the span below and is reported as Q/K/V
> projection time."*

`profile_sync` is gone. `GpuSpan` now brackets work with two `cuEventRecord`
calls enqueued **into the stream**; the host never waits. Elapsed time is read at
`gpu_drain_blocking()`, placed on boundaries that already synchronise (the
prompt→generate split and the end of a config), so the measurement adds no
synchronisation of its own.

Events are pooled and recycled per thread, and the pool bounds itself: a pair is
only returned at a drain, and a drain belongs on a boundary that already
synchronises — a phase, not a layer — so without a cap every span between two
boundaries would allocate a fresh pair and a long prefill would create driver
events without bound. Past 4,096 un-harvested pairs, opening a span first runs a
non-blocking harvest. That costs a few `cuEventQuery` calls once every 4,096
spans, never a stall, and it is what makes this a pool rather than an allocator.

Measured overhead, BF16×1 bulk t/s:

| build | bulk t/s | overhead |
|---|---|---|
| no profiling | 4433.9 | — |
| old host-sync profiler | 3562.4 | **−19.7%** |
| **event profiler** | **4363.6** | **−1.6%** |

**And removing the sync changed the answer.** `fwd_routing_wait` went
**20.2 ms → 79.0 ms** at BF16×1 and **203.0 ms → 937.6 ms** at C10×10. The old
profiler was not measuring it small — it was *draining the backlog before
reaching it*. With real async execution the forward thread blocks on genuine
outstanding work, and `fwd_routing_wait` becomes the largest span in the prefill
table, exceeding `pipe_worker_total` (47.2 ms) — the work it is waiting for.

Prefill, BF16×1, with the counts exposing the 3:1 layer schedule:

| | total | invocations | per layer |
|---|---|---|---|
| DeltaNet (`dn:ffn`+`mix`+`proj`+`out_proj`) | **153.0 ms** | ×60 | 2.55 ms |
| attention (`prefill:kernel`+`qkv_proj`+`out_proj`) | 17.2 ms | ×20 | 0.86 ms |
| `fwd_routing_wait` | 79.0 ms | ×80 | — |

A DeltaNet layer costs ~3× an attention layer, and there are 3× as many of them.

#### Using it on other models

The profiler is a general opt-in facility, not qwen3.6 scaffolding. It lives in
`candle-transformers/src/models/profile/`, one concern per file:

| file | holds |
|---|---|
| `mark.rs` | `ProfileMark` / `profile_now` — the host clock, or `()` when off |
| `accumulator.rs` | `ProfileAccumulator` (name → total, count) + `report()`, and `ProfileSnapshot` for crossing threads |
| `pipeline.rs` | the thread-local accumulator the hot path records into |
| `span.rs` | `span` / `span_if` — **host**-timed, plus NVTX ranges |
| `gpu.rs` | `gpu_span` / `gpu_span_if` / `gpu_span_phase` / `gpu_drain{,_blocking}` — **device**-timed |
| `tests.rs` | the contract: no-op under every feature set, real timing under `cuda,profile` |

To instrument a new model, bracket the region and name it:

```rust
let g = gpu_span("prefill:qkv_proj", x.device());
let qkv = self.qkv.forward(&x)?;
g.end();                       // or let it close at end of scope
```

`gpu_span_phase(decode, "decode:…", "prefill:…", dev)` picks the name by wave
phase — a host-timed span can choose its name at the *record*, but an event span
needs it at the *open*, and without the helper every phase-dependent site grows a
seven-line `if`. Then call `gpu_drain_blocking()` at a boundary that already
synchronises and read `pipeline_snapshot_and_reset()`.

Three properties worth relying on:

- **A CPU device is not an error.** `gpu_span` takes a `&Device` and silently
  records nothing on CPU, so a mixed-device path needs no `cfg` and no match.
- **Both `gpu_span` and `gpu_drain` exist in every feature combination** as
  zero-sized no-ops, so call sites are never `#[cfg]`-gated. Only the crate's
  `profile` + `cuda` features decide whether anything is measured; all four
  combinations build and are lint-clean.
- **An untimeable span is dropped, not recorded as zero** — a silent zero would
  read as "this phase is free", the one wrong answer a profiler must not give.

**Time device regions with `gpu_span`, host regions with `span`.** Both record
into the same table, so the choice is purely about what the number means. An
event pair around a region that enqueues nothing measures the host gap when the
stream happens to be idle and the outstanding backlog when it is not — the same
backlog-dependent answer the host-sync timer gave, reintroduced. `decode:alloc`
(cache validation) and `decode:meta` (metadata assembly, whose only possible
launches are dtype casts that do not fire when Q already matches the arena) are
host spans for this reason. Host time is also the answer that matters for them:
this model is launch-bound, and those are launch-side costs.

Placement is the other thing that needs care: the events measure the *stream*
interval, so a span that opens before an unrelated backlog attributes that
backlog to itself. That is exactly the failure the host-sync profiler had
everywhere, and it is why the `prefill:entry` span — which existed only to absorb
the caller's in-flight tail — was deleted rather than converted.

### 3c. GEMM fragmentation

`46,119 @ 23.6 µs` vs `9,274 @ 101.8 µs` for near-identical total time is the
single biggest structural difference. The likely cause is the hybrid layer's
projection set: an attention layer runs one fused QKV projection, whereas a
DeltaNet layer runs **five separate projections** (`wqkv`, `wz`, `w_beta`,
`w_alpha`, `w_out`), two of which (`w_beta`, `w_alpha`) are `[n_v_heads, hidden]`
— extremely thin matrices. At a 3:1 DeltaNet:attention schedule most layers pay
the five-projection cost.

**This is stated as a hypothesis, not a measurement.** The op counts are
measured; the attribution to DeltaNet's projection set is inferred from the
architecture and has not been confirmed by attributing launches to source sites
(see §6 — there is no NVTX coverage to do that with).

Note also that DeltaNet's *true* share is larger than the `DeltaNet` row
suggests: that row counts only the `delta_net_*` recurrence/conv kernels. Its
projections are GEMMs and land in the `int8 GEMM` row.

## 4. The selection kernel — real, but a different problem

`select_kv_format_palette4_paged<256>` is **5.6% of GPU time in BF16 configs and
88.5% in C-level configs**. It is not what pins bulk throughput (bulk t/s is
flat across that swing), but it dominates the compressed configs, which is what
zend will actually run.

`ncu`, `HB = 256`, 4 launches sampled mid-run:

| metric | value |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | **0.44 %** |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **8.33 %** |
| `launch__occupancy_limit_shared_mem` | 4 blocks |
| `launch__occupancy_limit_registers` | 4 blocks |
| `launch__shared_mem_per_block_static` | 24,304 byte/block |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | **320,804** |
| `gpu__time_duration.sum` | 16.3 ms / 20.0 ms |

**0.44% of peak SM throughput.** The kernel is not compute-limited; it is
starved. Three compounding reasons, all confirmed:

1. **Occupancy capped at 4 blocks/SM** by shared memory *and* registers
   simultaneously — the 24,304 B/block matches the ~23.8 KB the kernel's own
   header predicts for `HB = 256`.
2. **The grid cannot fill the machine.** `grid = n_chunks × n_kv_head`, one block
   per `(chunk, head)`. Measured grids: **42 blocks** for prefill seals and
   **2 blocks** for decode seals, on a **110-SM** card. The 2-block population is
   260 launches totalling **1.882 s** — using under 2% of the GPU.
3. **320,804 shared-memory bank conflicts** per launch.

The kernel header documents the `HB = 256` penalty (2× serial work per thread,
half the occupancy → 4×), and matched prefill launches measure exactly that:
4.2 µs/block at `HB = 128` vs 16.9 µs/block at `HB = 256`. But the ncu numbers
say the deeper problem is that **neither** width comes close to using the GPU —
`HB = 128` is merely cheaper at being starved.

Per-launch duration also grows ~42× over a run at *fixed* grid (0.57 ms early to
24 ms later). That tracks the config progression BF16 → Q8_0 → C0…C10, i.e. the
Phase 4/5 candidate search scaling with the compression candidate list — not head
dim.

## 5. Ruled out

- **Expert streaming.** All expert-pipeline counters are zero across all 14
  configs. The 22 GB checkpoint is fully resident in 72 GB; the cache never
  engages. (The `quantized_qwen36_moe` doc comment about "the 16 GB dev card"
  describes the old RTX 4090 Mobile. The "100.0%" hit rate in that table is a
  divide-by-zero artifact of zero traffic, not a measurement.)
- **Post-selection readback serialisation.** Four `memcpy_dtov` calls follow each
  selection launch (`sampled_selection/gpu.rs:1272-1275`). Measured GPU idle
  immediately after all 520 selection launches: **0.025 s total**, mean 48.7 µs.
  Not a stall.
- **Memset in the inference loop (invariant 6).** `cuMemsetD8_v2` shows 2.745 s
  of *API* time, but GPU-side memset is **0.089 s (0.8%)** — the API time is
  load-phase. Not a violation.
- **DeltaNet's recurrence kernels.** 0.357 s / 18.5% of GPU-busy time in the
  BF16 window, at 37.5 µs average — the largest average of any bucket, i.e. the
  *well-sized* work. The single-session decode gap of 10% vs Qwen3-30B is
  consistent with this and nothing more.

## 6. Instrumentation gap

**`qwen35/forward.rs`, `qwen35/wave.rs` and `delta_net/cuda.rs` contain zero NVTX
spans**, against 30 in `latent_moe/wave.rs`. Coverage exists only in shared code
they call: `batched_layer` (11), `prefill_utils` (9), `quantized_matmul` (4),
`qwen35/quantized_delta_net` (2).

Consequence: `nsys stats --report nvtx_kern_sum` cannot self-attribute this
model's kernels to phases. Everything above was reconstructed from kernel names,
launch geometry and timeline windowing — which is also why §3c's attribution of
the GEMM fragmentation to DeltaNet's projections remains a hypothesis rather than
a measurement.

The in-process profiler (§3b) covers the MoE/expert pipeline and the `dn:*` and
`decode:*` phases well, but it times by host sync, so it cannot see the gap
structure. Between them the two instruments answer different halves: **counts and
per-phase GPU work** from the profiler, **op sizes and idle** from the nsys
timeline. Neither alone would have produced this diagnosis.

Adding `span()` / `span_if()` marks mirroring `latent_moe/wave.rs` would make
this a one-run diagnosis and would let launches be attributed to source sites
directly.

## 7. Suggested work, in priority order

1. **Cut launch count on the elementwise path.** ~35,000 launches of 1–4 µs work
   in a 4 s window. The casts (13,384) are invariant-1 violations with a known
   remedy: have the producing kernel emit the consumer's dtype. `copy2d_f32`
   (8,340) is invariant 2. This is the cheapest large win and needs no new
   kernels.
2. **Batch the DeltaNet projections.** Five separate GEMMs per DeltaNet layer,
   two of them `[n_v_heads, hidden]`-thin. Fusing `wqkv`/`wz` and the
   `w_beta`/`w_alpha` pair into single launches would attack the 5×-launches-at-
   ¼-size finding directly. Confirm §3c first by attributing launches to sites.
3. **Batch the decode seals.** 260 launches × 7.2 ms at **2 blocks** on a 110-SM
   card. One launch across sequences/layers via a descriptor table would put real
   work in the grid — the pattern already exists (`bdp_recall_batched`'s
   per-gallery pointer table; CLAUDE.md invariant 2b).
4. **Rework the selection kernel's block mapping.** 0.44% SM throughput, 4
   blocks/SM, 320k bank conflicts. One block per `(chunk, head)` cannot fill the
   GPU at these chunk counts regardless of `HB`. Only worth doing after 1–3, but
   it is 88.5% of GPU time in the compressed configs zend will run.
5. **Chase `fwd_routing_wait`** — **79.0 ms/config at BF16×1, 937.6 ms at
   C10×10**, larger than the worker time it waits on, with `pipe_fence_wait`
   at ~0. Experts are resident, so this is the forward thread blocking on the MoE
   routing round-trip, not DMA. Same shape as the documented MoE-routing readback
   wall; worth checking whether the routing readback can be deferred or made
   device-resident as it was for DeepSeek-V4's `moe_bucketize`. **This is now the
   largest single span in prefill** and was invisible until the profiler stopped
   synchronising.
6. **Add NVTX spans to the qwen35 wave** (§6) — makes 1–5 measurable per phase,
   and would let launches be attributed to source sites.

*(The former item 6 — convert the profiler to CUDA-event timing — is done; see
§3b.)*

## Reproduction

```bash
cargo build --release --features cuda,nvtx --lib -p candle-transformers --tests

# timeline
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --duration=260 \
  --force-overwrite=true -o q36prof \
  target/release/deps/candle_transformers-<hash>.exe \
  quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b \
  -- --ignored --nocapture --test-threads=1

nsys stats --report cuda_gpu_trace --format csv q36prof.nsys-rep   # per-op start/dur/grid
nsys stats --report cuda_api_sum   --format csv q36prof.nsys-rep

# in-process counting profiler (counts exact; times perturbed by profile_sync)
cargo test --release --features cuda,profile --lib -p candle-transformers \
  quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b \
  -- --ignored --nocapture --test-threads=1

# kernel counters
ncu --kernel-name regex:select_kv_format --launch-skip 300 --launch-count 4 \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
  --csv <test binary> <test filter> --ignored --nocapture --test-threads=1
```

**Analyse against wall time, not GPU-busy time** — window `cuda_gpu_trace` by
timestamp and compute busy/idle per window, as in §2. Summing kernel durations
and taking percentages is what produced the wrong first answer.

Neither tool is on `PATH`:

```
/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe
/c/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.0/target/windows-desktop-win7-x64/ncu.exe
```

Baseline is `quantized_qwen3_moe::tests::test_parallel_batched_forwarding`
(Qwen3-30B-A3B, `head_dim 128`), same flags. Match windows by config, not by
elapsed time — the two sweeps progress at different rates.
