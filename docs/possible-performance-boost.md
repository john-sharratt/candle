# Possible performance boosts — Flash-Next single-session decode

Candidates for closing the gap to Strata's published single-session decode
(`quantized_qwen38_moe::tests::strata_bench_single_session`, median of three runs),
measured 2026-10-07 on the RTX PRO 5000 (sm_120, 72 GB, 96 MiB L2) at commit `7aecbeab`
plus the stacked DeltaNet replay.

| prompt | ours (t/s) | Strata RTX 5090 (t/s) |
|---:|---:|---:|
| 4K | 152.0 | 179.4 |
| 32K | 149.0 | 175.7 |
| 128K | 125.1 | 165.0 |

At 4K a speculative step is 18.9 ms at 3.07 accepted tokens/step; 179.4 t/s needs
~17.1 ms, so **~1.8 ms/step** has to come out. The rules of the chase still hold:
no prefix speculation, draft ceiling 4, no quantization downgrade, full-vocab draft
head — the gain has to be ms/step at unchanged tokens.

Figures below are per 4K verify step, from an nsys node trace (`--cuda-graph-trace=node`).
"Real" time is the time a kernel adds to the critical path: its end minus the later of
its start and the previous op's end. Raw nsys durations overstate kernels that are
launched with PDL, because those include the wait on their predecessor. Host-side gaps
in node traces are inflated ~10× and are not trusted.

## 1. Small decode-width GEMMs — ≈1.65 ms/step above their DRAM floor

The largest lever. At 5 rows these GEMMs are bounded by a per-launch fixed latency of
~4–5 µs, not by bytes, and there are ~230 of them a step.

| GEMM | kernel / grid | calls/step | real µs | floor µs | lost ms/step |
|---|---|---:|---:|---:|---:|
| DeltaNet out-proj (+ attention o-proj) | `q8_ko_int8_f32_dense_sk` 1×80×6 | 44.8 | 20.3 | ~8.3 | 0.54 |
| Hyper-connection `down` | `dense_sk` 1×13×27 | 88.8 | 7.8 | ~3.4 | 0.40 |
| Hyper-connection `up` (fused SiLU) | `q8_ko_int8_f32_dense_silu` 1×320 | 92.6 | 6.2 | ~3.1 | 0.29 |
| Shared-expert `down` | `dense_sk` 1×80×5 | 51.1 | 5.6 | ~1.3 | 0.22 |
| Shared gate/up + router (stacked) | `dense_sk` 1×57×7 | 44.0 | 8.4 | ~3.7 | 0.20 |

Observations:
- The out-proj did not speed up when its weights were prefetched into L2 30 µs
  early, so it is not DRAM-bound. Its time is structure: prologue, a short K walk,
  the per-K-tile partial store, `__threadfence`, the counter atomic, and the last
  block's re-read of every partial.
- An earlier ncu run of HC `down` (cold): 33% of DRAM peak, 0.53 waves (shared memory
  caps it at 6 blocks per SM), 81% of cycles with no eligible warp.
- The shared-expert `down` (K = 640, 5 K tiles) is split 5 ways, so each block walks
  one tile and then pays the whole reduction.

Directions:
- Re-derive `q8a128_dense_k_splits` for small K: fewer splits where a block would walk
  only one or two tiles.
- A cheaper reduction, or none at all for shapes that fill the card unsplit.
- A dedicated kernel for 8 rows or fewer.
- Any of these must keep the tile-ordered sum (`((0 + f₀) + f₁) + …`) so results stay
  bit-identical to the unsplit kernel.

**Measure in the micro-harness** (`qwen4exp/hyper/bench_ko.rs`, `gr_hyper_bench`),
**never with ncu on the full model**: kernel replay backs up the ~60 GB resident model
to host memory per pass, and stalls on the first kernel.

## 2. Hyper-connection read chain — ≈0.4–0.6 ms/step

One hyper-connection read runs norm → down → up → mix and costs ~20 µs real, against
~8–10 µs of work. There are about 104 per step: two per layer, plus the draft head.
- `gr_norm_q8_kernel` launches one block per (row, stream): 4 blocks at draft width,
  20 at verify width. It takes ~6.8 µs where ~2 µs should do. Spread each stream's
  reduction over more blocks, or more rows per block.
- Merging `gr_combine` into the next `gr_norm_q8` would save one launch boundary per
  module, ~0.2 ms/step. Both are per-(row, stream) passes over the same residual.
- A cooperative kernel for the whole read half is possible, but it is a large
  project. Do the two items above first.

## 3. `moe_bucketize` — ≈0.5 ms/step

`moe_bucketize_kernel` runs as a single 512-thread block at ~14 µs per MoE layer
(0.75 ms/step). A multi-block form should be ~3 µs. This is the live MoE dispatch's
area (`docs/moe_live_dispatch_design.md`); agree ownership before touching it.

## 4. Draft walk — ≈0.15–0.3 ms/step

- Greedy draft sampling (`batched_penalty_sampling_kernel`, 1 block of 1,024 threads
  over a 248,320-wide row) takes ~42 µs, four times a step. A multi-block argmax with
  the same lowest-index tie-break would be a few µs, and acceptance would be unchanged.
- Each draft step makes a 32-byte eager H2D upload from `build_decode_metadata_at`,
  which pauses the capture and cuts the graph segment. Building all four steps'
  headers up front would remove three cuts. The real idle this costs is unclear,
  because traced host gaps are inflated.
- The draft step's Q4 LM head is at its floor (285 µs against ~267 µs) and stays
  full-vocab.

## 5. A segment boundary at every MoE layer — ≈0.2–0.4 ms/step (uncertain)

`MoeDispatch::after` flushes, ending the graph segment at every MoE invocation. That is
51 boundaries a step, each a graph-to-graph gap (~23 µs traced; the untraced cost is
unknown). Holding back every second flush is safe: `hold_for_ring` waits only on
invocations 64 back. Watch the start of a step, where the GPU waits for the first
segment. This is a cheap experiment.

## Smaller or later

- **Verify attention**, `paged_prefill_int8_kernel`, ~1.05 ms/step. Not DRAM-bound,
  and L2 prefetch measured zero. Revisit only for a kernel-structure change.
- **DeltaNet verify kernels**: state 0.5 ms, intra 0.3 ms per step. These are
  latency-bound tiny grids, sequential across layers.
- **128K.** The gap widens (125 against 165), partly through expert evictions. That
  is the expert cache's area.

## Already at the floor — not worth chasing

| piece | measured | floor | efficiency |
|---|---:|---:|---:|
| Routed experts (`q4_ko_int8_f32_grouped`) | ~91 µs/layer | — | ≈ floor |
| DeltaNet in-proj (`dense_m2` 1×515) | 38.7 µs | 33.4 µs | 86% |
| Verify LM head (Q8) | 552 µs | 504 µs | 91% |
| Draft LM head (Q4) | 285 µs | 267 µs | 94% |
| Attention qkv (`bf16_dense` 1×416) | 31 µs | 27 µs | 86% |

## Tried and measured negative — do not retry

| idea | result |
|---|---|
| L2 prefetch of dense weights, in the launch stream | 152 → 128 t/s: the prefetch serialised for its whole transfer |
| L2 prefetch of dense weights, on a forked side stream in the graph | 152 → 139 t/s: idle +1.7 ms/step, HC GEMMs slowed under contention |
| Draft ceiling 5 | 148 → 143 t/s at 4K |
| Larger graph exec cache (16 → 64) | reshapes fell 15×, zero t/s change |
| L2 prefetch in verify attention | ncu −6%, production 0 |

## Done in this round, for reference

- **Stacked DeltaNet replay.** The speculative rewind replays every recurrent layer in
  one launch triple (`delta_net/replay_stack.rs`) instead of 36 per-layer triples. The
  per-layer replay had measured ~767 µs of GPU time per rewind; the stacked form was
  estimated, not traced. Measured +3 t/s at 4K, 149.0 → 152.0, at unchanged tokens.
