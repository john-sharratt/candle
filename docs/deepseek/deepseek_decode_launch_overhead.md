# DeepSeek-V4 decode: where the time actually goes

**Status: measured, first two optimisation attempts NEUTRAL and one reverted.**
Target is decode aggregate 48.5 → 100 t/s. This records the measurement and,
more importantly, the two ways the instrumentation misleads — because both
sent an optimisation attempt in the wrong direction.

## Baseline (cfg16, `test_parallel_batched_forwarding`, `--features profile`)

960 decode tokens (60 waves × 16 sessions) in 19.79 s = **48.5 t/s aggregate**.
Aggregate scales sub-linearly with wave width:

| sessions | decode t/s (aggregate) |
|---|---|
| 1 | 3.4 |
| 4 | 23.0 |
| 8 | 34.7 |
| 16 | 48.5 |

That shape is the signature of a **fixed per-layer cost amortised across
sessions**, which is the single most useful fact here.

## Span breakdown (cfg16 decode, 19790 ms)

| span | time | calls |
|---|---|---|
| `moe:sort` | 8684 ms | 2580 |
| `decode:prep` (of which `dprep:push` 3080 ms / 44806 calls) | 3452 ms | 2580 |
| `moe:submit` | 3006 ms | 2580 |
| `decode:select` | 2579 ms | 2580 |
| `decode:kernel` + `outproj` + `gather` + `cache` | ~400 ms | 2580 each |

`moe:sort` splits — measured, not assumed — as:

```
moe:sort           8545.4 ms
moe:sort_readback  8540.1 ms   ← 99.94%
```

So the **host counting sort is free** (~5 ms across 2580 calls) and the whole
cost is `indices.to_vec2()`, the sanctioned routing readback: 3.31 ms per layer,
43 times per token.

### That readback is GPU WORK, not sync overhead — measured

An earlier draft of this document claimed the 3.31 ms was "overwhelmingly WDDM
round-trip latency". **That was wrong.** Draining the stream first and then
timing a 512-byte synchronous readback isolates the latency exactly:

```
[d2h] empty-queue sync readback (512 B) = 9 us
```

**9 µs.** So 3.31 ms of readback is 99.7% waiting for the GPU to finish what
this layer already queued, and ~0.3% WDDM tax. The readback is not a cost to be
removed — it is an accidental, and accurate, GPU-execution probe.

Consequence: decode is **GPU-bound on real work** (~3.3 ms of GPU per layer plus
the expert GEMMs, against 7.6 ms of wall time per layer). Making decode faster
means making that GPU work cheaper — i.e. **kernel fusion** — not removing
syncs, not reducing launch counts, and not deepening prefetch.

## Phase 0: the PCIe / prefetch question — ANSWERED, and it is a dead end

Measured on this machine (`measure_h2d_bandwidth_at_expert_slot_size`, real
14.2 MB expert-slot geometry, `cuMemAllocHost_v2` source, stream copies):

```
pinned, back-to-back   = 57.2 GB/s      ← PCIe 5.0 at ~90% of theoretical
pinned, sync per copy  = 54.8 GB/s      ← 0.26 ms per expert
pageable, back-to-back = 24.7 GB/s      ← 2.3× penalty for missing the pinned tier
```

And in the real engine, the decode-phase pipeline profile at cfg16:

| span | time | calls |
|---|---|---|
| `pipe_classify_load` | 2649 ms | 2580 |
| ↳ `cold_read` (NVMe pack file) | 2049 ms | 1547 |
| `pipe_compute_experts` | 259 ms | 2580 |
| **`pipe_fence_wait`** | **2.7 ms** | 2580 |

**The copy fence never stalls** — 2.7 ms across 2580 layers, ~1 µs each. Expert
DMA is already completely hidden behind compute, at ~40% bus utilisation.

Therefore **prefetching further ahead cannot speed up decode**: there is no
stall to hide. The prediction system's low coverage (508 predicted loads against
41725 misses, `PREFETCH_MAX_K = 8`, hint path dead at 0 loads) is a real
inefficiency in *bandwidth spent*, but it is not costing tokens, because the
bandwidth it wastes is bandwidth nothing else wanted.

The one genuine load-path cost is `cold_read`: 2049 ms of NVMe reads for the
35% of experts the warm tier (7118 of 11008 slots) cannot hold. That is ~10% of
decode and is disk, not PCIe.

## TRAP 1 — the async spans measure submission, not execution

`decode:kernel` reads 83 ms across 2580 launches. That is **launch time**, not
GPU time: the work is asynchronous, so the span closes when the launch is
queued. Reading these spans as a cost breakdown says "attention is 2% of decode,
so ~80% of decode is bubble", which is wrong and led directly to the two failed
attempts below.

The readback is the only span that measures real GPU progress, because a
synchronous D2H waits for everything already queued on the stream. So the honest
reading is:

* GPU-busy ≈ `moe:sort_readback` (8.5 s) + `moe:submit` (3.0 s, includes the
  expert GEMMs) ≈ **11.5 s of 19.7 s ≈ 58%**
* host-only ≈ `dprep:push` (3.1 s) + `decode:select` (2.6 s) + misc ≈ 6 s

Which caps what host-side batching alone can buy: removing *all* host overhead
takes 19.7 s → ~13.7 s, i.e. 48.5 → **~70 t/s**. Reaching 100 t/s requires
reducing GPU work or widening the wave, not just cutting launches.

This is the mechanical explanation for the standing note that "WDDM spans lie —
trust API counters".

## TRAP 2 — you cannot hide GPU work behind a later same-stream sync

The shared expert is always-on and depends only on `normed`, never on routing,
so issuing it *before* the routing readback looks like free overlap: give the
GPU real work to chew on during the stall.

It does nothing, and the reason is structural. `memcpy_dtov` is a **stream**
sync (cudarc: async copy then `stream.synchronize()`), so it waits for
everything already queued on that stream — including the shared expert just
queued. The stall does not shrink; it grows by exactly the work moved into it.

Measured: `decode_total` 19790 → 19687 ms (noise), `moe:shared` 219 → 206 ms.
**Reverted**, because the code was unchanged in effect and the comment
justifying it was false.

Real overlap would need the shared expert on a *different* stream with the
readback waiting on a specific event — which is the same side-stream machinery
that was already tried for the readback itself and reverted at −8% on cfg8.

## What was tried

| change | result | kept? |
|---|---|---|
| One shared `arange` per layer, narrowed per session (was up to 16 allocs+launches) | `decode:select` 2579 → 2562 ms — neutral | kept (fewer allocations, no perf claim) |
| Shared-expert FFN issued before the routing readback | `decode_total` 19790 → 19687 ms — neutral | **reverted**, rationale disproven |

All five configs stayed valid; cfg16 bulk 971 → 973 t/s, single 51.7 → 52.0.

## Why the readback cannot simply be removed

`GpuDispatchTables::build` — the GPU-native path that eliminated this readback
for fully-resident models — refuses DeepSeek on two independent counts:

```rust
if n_experts > 128 || keys.len() != n_layers * n_experts { return None; }
```

DeepSeek routes 256 experts per layer, and the streaming cache holds 3632 of
11008 slots, so the residency grid is permanently incomplete. The host needs
the expert ids to schedule pinned→VRAM DMA; that is invariant 3's sanctioned
readback, and it is load-bearing for the cache, not an oversight.

A previously-tried variant (dedicated routing stream + async DtoH into a pinned
buffer) was reverted at −8% on cfg8: the per-layer event/side-stream overhead
over 43 layers × N sessions outweighed the pinned-copy saving.

## Where the remaining headroom is

Fixed cost per decode wave: 43 layers × 3.3 ms ≈ **142 ms of sync**, independent
of how many sessions share it. At 16 sessions that is 8.9 ms/token; at 32 it is
4.4 ms/token. This is why aggregate t/s scales with wave width, and it is the
cheapest remaining lever — pure batching, no new kernels.

### Wider waves: TESTED AND REFUTED

The obvious lever — amortise the fixed 142 ms/wave sync over more sessions —
**makes things worse on this hardware**:

| | bulk t/s | decode t/s (aggregate) | peak KV tokens |
|---|---|---|---|
| cfg16 | 972.9 | 48.5 | 10792 |
| cfg32 | **105.8** | **42.1** | 21548 |

Both configs stayed valid, so this is throughput, not correctness. Prefill
collapses ~9× and its routing readback goes to 109666 ms over 258 calls — 425 ms
each, i.e. an enormous GPU backlog rather than sync latency. At 21548 KV tokens
the elastic partition cedes ground to KV and starves the expert cache, which
then thrashes. 16 sessions is already at or past the sweet spot; do not spend
more time here without changing the VRAM split first.

Ranked by expected value:

1. ~~Wider waves~~ — refuted above.
2. **`dprep:push`, 44806 calls** — per-session compressor push + per-gallery
   `append_batch`. Each session owns its own gallery arena, so batching needs a
   multi-gallery append taking a device pointer table (the shape
   `gather_corpus_batched` already uses on the read side). Worth ~3 s of host
   time, but see Trap 1: some of that is queue backpressure, not pure host work.
3. **`decode:select`** — the per-session `arange` is already gone and it did not
   move the number, so the cost is `two_stage_select_batched` (sign_pack →
   `bdp_recall_batched` → `topm_select_batched`) or the small per-layer H2D
   uploads. Needs isolating before any more work goes into it.
4. **Reducing GPU work itself** — at 58% GPU-busy this is where a 2× ultimately
   has to come from. The expert GEMM path (`moe:submit`) is the largest single
   consumer.

## The nsys kernel breakdown — what decode GPU time is actually spent on

`nsys profile --trace cuda` over `decode_step_bitwise_probe` (1 session, 43
layers, ~25 decode steps ⇒ instances ÷ 1075 = per-layer count):

| kernel | % GPU | total | instances | per layer | avg |
|---|---|---|---|---|---|
| `mxfp4_ko_int8_f32_grouped` | 33.1% | 518 ms | 4257 | ~4 | 122 µs |
| `q8_ko_int8_f32_dense` | 22.1% | 345 ms | 18163 | **~17** | 19 µs |
| cublas `gemmSN_NN` | 6.5% | 102 ms | 688 | — | 148 µs |
| `sinkhorn_f32_kernel` | 5.4% | 84 ms | 2838 | ~2.6 | 30 µs |
| `mhc_post_kernel` | 4.5% | 70 ms | 2838 | ~2.6 | 25 µs |
| `q8_ko_int8_f32_grouped` | 3.9% | 61 ms | 1075 | 1 | 57 µs |
| `latent_decode_kernel` | 2.6% | 40 ms | 1075 | 1 | 37 µs |
| `mhc_pre_gates` / `mhc_pre_reduce` | 3.2% | 50 ms | 2838 | ~2.6 | 12 / 5 µs |
| `ucopy_f32` | 2.2% | 34 ms | 12313 | ~11 | 2.8 µs |
| `quantize_q8a128` | 1.1% | 16 ms | 18644 | ~17 | 0.9 µs |

Three conclusions:

* **The attention kernel is 2.6% of decode.** Further latent-attention tuning
  cannot move the number.
* **17 dense int8 GEMMs per layer at 19 µs each** (with 17 matching quantize
  calls). At one decode token these are GEMVs — microseconds of real math — so
  19 µs is launch and occupancy. 22% of GPU time in dispatch.
* **The mHC chain is 4 kernels per layer totalling 13%.**

**Ceiling:** `mxfp4_ko_int8_f32_grouped` (33%) is irreducible model compute.
Perfectly fusing everything else roughly halves GPU time, and decode is ~60%
GPU-bound, so the realistic target is **48.5 → ~75 t/s**. Beyond that needs the
expert GEMM itself to get cheaper.

### First fusion: fold `sinkhorn` into `mhc_pre_gates`

`sinkhorn_f32_kernel` is **one thread per matrix**, and at decode `n` is a
handful of matrices — so it is a ~1-thread kernel costing 30 µs, essentially
all launch overhead around a few dozen flops on a tiny `[hc, hc]` matrix.

`mhc_pre_gates_kernel` runs **one block per row** and already writes that row's
whole `comb_raw`. So the normalisation can happen in the block that produced it:
`__syncthreads()` after the `comb_row` loop, then one thread runs the existing
softmax → column-normalise → `(iters-1)` × (row, column) loop out of
`sinkhorn.cu` verbatim and writes `comb`. Copying the loop body unchanged keeps
it **bit-exact** — same values, same operation order, only a launch removed.

Wiring: `mhc_pre_gates` gains `comb_out`, `iters`, `eps` and emits the
normalised matrix; `HyperConnection::pre` drops its `self.sinkhorn(&comb_raw)`
call. Worth ~5.4% of decode GPU time and one launch per sub-block per layer.

## Method note

Measure GPU utilisation with `nsys` before optimising further. Every span-based
inference in this file that was not cross-checked against the readback turned
out to be misleading, and two changes were built on such an inference before it
was checked. The readback span is an accidental GPU-progress probe; it is the
only trustworthy one in the current instrumentation.
