# Qwen3.6-35B-A3B — where the prefill time goes

**Measured 2026-08-23 on RTX PRO 5000 Blackwell 72 GB (sm_120, 110 SM), CUDA 12.4,
`nsys` 2026.1.3.** Working tree at `5da6d80d` + the uncommitted merge fixes.

## Summary

The bulk-throughput deficit against Qwen3-30B-A3B is **not DeltaNet**. DeltaNet's
three prefill kernels are 8.7% of GPU time. **53.6% of all GPU kernel time is the
adaptive-KV-compression selection path**, and the single kernel
`select_kv_format_palette4_paged<256>` is **51.4%** of it.

Two independent causes, both structural, both measurable:

1. **The `HB = 256` instantiation costs ~4× per block** what `HB = 128` costs.
   This is documented behaviour of the kernel, not a bug — see §3 — but the
   qwen3.5/3.6 lineage runs `head_dim 256`, so it pays it on every seal.
2. **The launch grid is far too small to fill the GPU.** Prefill seals launch
   **42 blocks** and decode seals launch **2 blocks**, on a **110-SM** card. Half
   the kernel's wall time is in the 2-block launches.

The second is the more actionable of the two, and is independent of head dim.

## 1. The measurement that motivated this

Full sweep results are in the run log; the relevant shape:

| | BF16×1 | BF16×4 | C8×5 | C9×2 | C10×10 |
|---|---|---|---|---|---|
| t/s (bulk) | 4433.9 | 4307.8 | 4368.4 | 4996.4 | 4303.8 |
| t/s (single) | 50.5 | 162.9 | 194.2 | 91.5 | 310.2 |

Single-session decode scales properly with batching (50.5 → 310.2, +514%).
**Bulk throughput does not move at all** — 4304…4996 across compression 2.21×
to 7.67× and contexts 1 to 10. C10 at ten contexts is *slower* in bulk than BF16
at one.

A bulk rate that is invariant to a 10× change in contexts is a **fixed per-wave
cost**, not a per-token one. Against Qwen3-30B-A3B on the same harness and
machine: single-session decode is only 10% down (50.5 vs 56.3 t/s — DeltaNet's
genuine per-token cost), but bulk is **34% down** (4434 vs 6752).

## 2. GPU time attribution

`nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --duration=260`, over
the test binary directly (not through `cargo`), then
`nsys stats --report cuda_gpu_kern_sum`.

### Qwen3.6-35B-A3B — 9.704 s total GPU kernel time

| subsystem | time | share |
|---|---|---|
| **KV-compression select/convert** | **5.203 s** | **53.6%** |
| int8/quant GEMM (weights) | 2.465 s | 25.4% |
| DeltaNet | 1.020 s | 10.5% |
| other | 0.604 s | 6.2% |
| attention kernels | 0.412 s | 4.2% |

Top kernels:

```
 51.4%    4.988s     520     9592.4us  select_kv_format_palette4_paged<256>
  7.6%    0.735s   15016       49.0us  q4_ko_int8_f32_grouped
  6.8%    0.663s    1860      356.6us  delta_net_prefill_state_f32_kernel
  5.2%    0.504s    6945       72.6us  q5_ko_int8_f32_grouped
  3.7%    0.355s   26520       13.4us  q8_ko_int8_f16_dense
  3.6%    0.348s    4070       85.4us  q8_ko_int8_f16_dense_m2
  3.0%    0.289s   32000        9.0us  q8_ko_int8_bf16_dense
  2.6%    0.247s     410      603.6us  paged_prefill_int8_kernel<__half,256>
  1.5%    0.145s    4200       34.5us  delta_net_decode_step_f32_kernel
  1.4%    0.133s    1860       71.5us  delta_net_prefill_intra_f32_kernel
```

DeltaNet total = 0.663 + 0.145 + 0.133 + 0.044 (`conv_prefill`) = **1.020 s, 10.5%**.

### Qwen3-30B-A3B baseline — 14.193 s total GPU kernel time

| subsystem | time | share |
|---|---|---|
| KV-compression select/convert | 6.500 s | 45.8% |
| int8/quant GEMM (weights) | 3.361 s | 23.7% |
| attention kernels | 2.904 s | 20.5% |
| other | 1.429 s | 10.1% |

```
 38.5%    5.458s    5952      917.0us  select_kv_format_palette4_paged<128>
 10.9%    1.543s    2064      747.7us  paged_prefill_int8_kernel
  7.1%    1.005s   21662       46.4us  q4_ko_int8_f32_grouped
  6.2%    0.879s    6240      140.8us  int8_decode_bmma_kernel
  5.1%    0.717s    2162      331.8us  q4_ko_int8_f32_grouped_m4
  3.6%    0.506s    5952       85.0us  approximate_q_relevance_quantiles
```

Selection is heavy on **both** models. What differs is the per-launch cost.

## 3. Why `HB = 256` is 4× per block

`candle-kernels/src/quantize/select_kv_format.cuh` documents this directly:

> 4 warps × 32 lanes = 128 threads per (chunk, head), **regardless of head_dim**.
> … **Thread count does NOT scale with HB** — at HB = 256 each thread owns 2
> entries where it owned 1.

and

> The HB = 128 layout (~12.6 KB smem/block) sustains **8 blocks/SM** inside the
> ~102.4 KB MaxShared carveout; the HB = 256 layout (~23.8 KB) fits **4**.

So `HB = 256` pays **2× serial work per thread × ½ the occupancy = 4×**.

Measured, normalising by grid size on the early prefill launches:

| | grid | block | duration | per-block |
|---|---|---|---|---|
| Qwen3-30B `HB=128` | 80 | 128 | ~323–358 µs | **~4.2 µs** |
| Qwen3.6-35B `HB=256` | 42 | 128 | ~676–860 µs | **~16.9 µs** |

**4.0×** — the structural prediction exactly.

## 4. The grid is too small (independent of head dim)

Grid-size distribution of the selection kernel over the captured window:

| model | grid=2 or 4 | grid=42/44 or 80/84 |
|---|---|---|
| Qwen3.6-35B | 260 launches (grid **2**) | 260 launches (grid 42/44) |
| Qwen3-30B | 2976 launches (grid **4**) | 2976 launches (grid 80/84) |

`grid = n_chunks × n_kv_head`, one block per `(chunk, head)`, 128 threads each.
The small-grid population is the per-step decode seal (1 chunk); the large-grid
population is the prefill seal.

Time split:

| model | small-grid launches | large-grid launches |
|---|---|---|
| Qwen3.6-35B | 260 → **1.882 s** (7239 µs each) | 260 → 3.106 s (11946 µs each) |
| Qwen3-30B | 2976 → 2.397 s (805 µs each) | 2976 → 3.061 s (1029 µs each) |

**A 2-block launch taking 7.2 ms on a 110-SM GPU.** That population alone is
1.882 s — 19% of all GPU kernel time in the capture — while using under 2% of the
machine. Qwen3.6 has fewer KV heads than Qwen3-30B (inferred from grid 2 vs 4),
so it launches *half* as many blocks *and* each costs ~4× more.

This is the finding I would act on first. It is a launch-shape problem, not a
head-dim problem: the same under-occupancy exists on Qwen3-30B (4 blocks), it is
merely cheaper there.

## 5. What this is not

- **Not expert streaming.** All expert-pipeline counters are zero across all 14
  configs — 0 hits, 0 misses, 0 DMA, 0 fence stalls. The 22 GB checkpoint is
  fully resident in 72 GB; the cache never engages. (The `quantized_qwen36_moe`
  doc comment saying "the 16 GB dev card runs it through the three-tier expert
  cache" describes the *old* RTX 4090 Mobile, not this machine. The "100.0%" hit
  rate in that table is a divide-by-zero artifact of no traffic, not a
  measurement.)
- **Not DeltaNet.** 10.5% of GPU time, and the single-session decode gap vs
  Qwen3-30B is only 10% — consistent with DeltaNet's per-token cost and nothing
  more.
- **Not attention.** 4.2% on Qwen3.6 against 20.5% on Qwen3-30B.

## 6. Caveats on these numbers

Stated plainly so nothing here is over-read:

- **The two captures cover different config mixes.** Both windows were 260 s.
  Qwen3-30B completes its whole 16-config sweep inside that; Qwen3.6 reaches only
  the first ~4 of 14. The Phase 4/5 candidate search in the selection kernel
  scales with the candidate list, which differs per compression level — so the
  *per-launch averages* (9592 µs vs 917 µs) mix different work. **The per-block
  numbers in §3 are taken from matched early prefill launches and are the sound
  comparison; the aggregate ratios are not.**
- Per-launch duration varies widely on Qwen3.6 (min 556 µs, max 25.2 ms,
  median 8.37 ms). That spread is unexplained by head dim alone and is worth
  isolating.
- `--duration` bounds collection and kills the app, so neither capture includes
  the sweep's tail.

## 7. Instrumentation gap

**`qwen35/forward.rs`, `qwen35/wave.rs` and `delta_net/cuda.rs` contain zero NVTX
spans**, against 30 in `latent_moe/wave.rs`. Coverage exists only in shared code
the hybrid path calls: `batched_layer` (11), `prefill_utils` (9),
`quantized_matmul` (4), `qwen35/quantized_delta_net` (2).

Consequence: `nsys stats --report nvtx_kern_sum` cannot self-attribute this
model's kernels to phases the way it can for DeepSeek-V4. Everything above had to
be reconstructed from kernel names and launch geometry.

Adding `span()` / `span_if()` marks to the qwen35 wave phases — mirroring
`latent_moe/wave.rs` — would make this a one-run diagnosis in future, and would
let per-config attribution resolve the §6 caveat directly.

## 8. Suggested next measurements

In the order that would settle the most per unit effort:

1. **`ncu` on `select_kv_format_palette4_paged` for both `HB` values** —
   `achieved_occupancy`, `sm__throughput`, `smsp__warps_active`. Settles whether
   the 4× is purely the documented occupancy/work split or whether something else
   (smem bank conflicts, the Phase 2 single-warp sort) dominates at 256.
   Note the standing lesson from the decode-fusion campaign: *"memory-bound" was
   an unmeasured assumption — ncu said FP64 84.7%.* Profile before theorising.
2. **Attack the 2-block launches.** 260 launches × 7.2 ms for 2 blocks of work is
   the clearest waste in the trace. Batching decode seals across sequences (or
   across layers) into one launch would put real work in the grid. Compare with
   the existing precedent: `bdp_recall_batched`'s per-gallery pointer table, and
   the descriptor-table pattern in CLAUDE.md invariant 2b.
3. **Add NVTX spans to the qwen35 wave** (§7), then re-profile per config to
   separate compression-level search cost from head-dim cost.
4. **Check the host readback after selection.** `sampled_selection/gpu.rs:1272-1275`
   issues four `memcpy_dtov` calls immediately after the launch. Those are
   synchronous host readbacks per seal; they did not show as GPU kernel time here,
   but they would serialise the pipeline and are worth a look on the CPU timeline
   given CLAUDE.md invariant 3.

## Reproduction

```bash
# Build the NVTX-enabled test binary
cargo build --release --features cuda,nvtx --lib -p candle-transformers --tests

# Capture (bounded; --duration kills the app at expiry)
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --duration=260 \
  --force-overwrite=true -o q36prof \
  target/release/deps/candle_transformers-<hash>.exe \
  quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b \
  -- --ignored --nocapture --test-threads=1

# Attribute
nsys stats --report cuda_gpu_kern_sum --format csv q36prof.nsys-rep
nsys stats --report cuda_gpu_trace   --format csv q36prof.nsys-rep   # grid/block per launch
```

`nsys` and `ncu` are not on `PATH` on this machine:

```
/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe
/c/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.0/target/windows-desktop-win7-x64/ncu.exe
```

Baseline for comparison is `quantized_qwen3_moe::tests::test_parallel_batched_forwarding`
(Qwen3-30B-A3B, `head_dim 128`), same flags.
