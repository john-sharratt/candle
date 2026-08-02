# GPU-native MoE dispatch — implementation report

**Goal:** eliminate the per-layer expert-routing GPU→CPU readback (the dominant
decode stall on WDDM) with a fully GPU-native dispatch, per the Option-1 design.

**Result: built, verified, and it beats the target.**

| metric (same binary, same substrate snapshot, same prompt) | host path | GPU-native |
|---|---|---|
| raw decode forward (`fwd avg`) | 26–28 ms | **18–19 ms** |
| end-to-end tokens/s (`tps`) | 25–29 | **44** (22.7 ms/token) |
| GPU utilisation during decode | ~50 % | ~70 % |

On the earlier bigger-context conversation (~3.5 K KV) the host path measured
31–33 ms; the same ~8–9 ms saving lands that at ~23–25 ms — at or under the
25 ms target there too. The host path remains intact for the paged/partial
regime (Phase B) and as the `ZEND_MOE_HOST_DISPATCH=1` diagnostic fallback
(any value but `0`/empty forces the host path).

## Post-implementation review hardening (multi-agent review, all fixed)

An 8-angle review (line-by-line, invariants, lifetime/concurrency, reuse,
simplification, efficiency, altitude, conventions) confirmed the design and
surfaced the following, all now fixed and re-verified:

- **Gate hardening:** the fast path additionally requires
  `gd.n_experts == router width` (a mismatch would index other layers' expert
  rows), `k ≤ 32` (the bucketize sort bound — degrades to host, not an error),
  a live pipeline thread (see below), and the gate no longer needs an
  `unreachable!` (single `if let` destructure).
- **GEMM full-row bounds:** `grouped_qmatmul_dev_q8a128` now takes `n_experts`
  and validates the whole `[expert_base, expert_base + n_experts)` row.
- **Pipeline dead-flag:** the pipeline thread sets a flag on exit (incl.
  panic-unwind, which frees the slot weights the tables point at); the gate
  checks it, so a dead pipeline degrades to the host path's loud error instead
  of silently reading freed VRAM.
- **Build-time safety:** tables refuse to build for non-KO weights (no int8
  kernel twin — host path instead of per-forward errors) and when the compute
  stream is not the legacy null stream (the ordering invariant the whole
  single-stream design rests on; verified true today, now asserted).
- **Prefill grid bound:** the GEMM grid launches `⌈a_ub/32⌉ + n_experts` tiles
  (the provable maximum) instead of `a_ub` — at 2048-token prefill that is ~25×
  fewer padding blocks (~1.7 M no-op blocks/layer avoided).
- **Shared `GROUPED_GEMM_TILE_W` constant** across the host tile builder and
  the bucketize call (was 3 uncoordinated literals guarding the bit-exactness
  invariant); mirrored `MAX_EXPERTS`/`MAX_TOPK` consts between the FFI and the
  wrapper so their validation can never drift (a drifted wrapper would skip the
  launch and leave the workspace holding the previous layer's tables).
- **Deterministic padding:** the gather zeroes sentinel rows (sanitizer/golden
  hygiene); eager workspace (no lazy-init dance); import-rule cleanups; 3 new
  `GpuDispatchTables::build` unit tests (complete grid + expert_base math,
  sparse-grid rejection, non-KO rejection).

**Known/accepted (documented, deliberate):** `PipelineStats`/transition-matrix
counters read 0 on the GPU-native path (no classification actually runs — the
hit-rate tables in batch_test measure the host path only); the bucketize's
single-block phases cost ~µs at decode and ~5 ms per 2048-token prefill chunk
(parallelizable later if prefill profiling warrants); GEMM-output padding rows
remain uninitialized when router sentinels occur (rare; no consumer reads
them).

## How it works

Per MoE layer the old path did: `moe_route` (GPU) → **read indices back to CPU**
(pipeline-draining `to_vec2`) → CPU counting-sort → CPU pointer arrays →
re-upload → grouped GEMM. 48 layers of serialized CPU↔GPU ping-pong.

The new path keeps everything on-device:

1. **`moe_bucketize`** (new kernel, `candle-kernels/src/simple/moe_bucketize.cu`)
   — one 128-thread block, **no atomics**, turns `moe_route`'s `[n,k]` index
   tensor into every downstream table: expert-grouped gather lists, grouped-GEMM
   tile tables, and the deterministic scatter's token-major segment tables
   (ordered by ascending grouped row = the host path's exact accumulation
   order). All outputs padded to the `n×k` upper bound so **no data-dependent
   value ever returns to the host** (padding tiles `b_cnt=0`, padding rows `!0`,
   skipped by one-line guards in the existing kernels).
2. **Resident pointer tables** (`expert_lre/gpu_dispatch.rs`) — with every
   expert VRAM-resident, weight addresses are static; captured once at load
   (both cache constructors: threaded/mmap after the synchronous prewarm, and
   reader `new_prepopulated`) into flat `[n_layers × 128]` device tables.
3. **`grouped_qmatmul_dev_q8a128`** (candle-core) — the grouped GEMM launched
   straight off the device tables (raw expert ids + per-layer `expert_base`),
   no host pack, no `memcpy_stod`, upper-bound grid.
4. **`forward_gpu_native`** (`quantized_qwen3_moe.rs`) — route → bucketize →
   gather → gate/up → fused SwiGLU → down → deterministic scatter. Also
   bypasses the per-layer pipeline-thread handoff. Gated on: dispatch tables
   present (all-resident) ∧ int8 activations ∧ not routing-capture ∧ not
   `ZEND_MOE_HOST_DISPATCH`. Everything else falls back to the host path.

## Correctness — the equivalence chain (all bit-exact, no tolerances)

End-to-end output hashing can't gate this: the daemon is **inherently
non-reproducible run-to-run even on the unmodified host path** (verified:
identical snapshot + fresh conversation, host↔host runs diverge — wave-timing
ULP amplified by temp-0.8 sampling). The gate is component equivalence:

1. `cuda_moe_bucketize_matches_cpu_reference` — every bucketize output buffer
   **bit-identical** to a CPU reference mirroring `forward_with_indices`' sort
   (decode, duplicate experts, router sentinels, one-expert multi-tile,
   all-sentinel, k=1, fuzz to 2048×8, repeat-run determinism). Includes the
   scatter ordering (ascending grouped row per token — caught and fixed a
   would-be ULP divergence during development).
2. `cuda_grouped_qmatmul_dev_matches_host_tables` — the device-table GEMM
   **bit-identical** to the host-table GEMM on the same operand and
   production-faithful Q6_KO weights (decode-1tok, mixed, prefill-64,
   prefill-200 multi-tile).
3. The gather / SwiGLU / scatter kernels are shared verbatim and consume tables
   proven identical in (1) ⇒ the GPU-native forward is bit-identical to the
   host forward per invocation.

Live: coherent outputs across all runs; existing moe/grouped/scatter/gather
suite green (the one failure, `grouped_int8_outlier_stress`, **fails
identically on the clean committed tree** — pre-existing, unrelated).

## Benefits decode AND prefill AND glue

The dispatch is in the shared expert path — all three run through it. Decode
and glue (sync-bound) gain the most; prefill gains the removed serialization
plus the eliminated CPU counting-sort at large batch.

## Phase B (partial residency) — unchanged design, ready foundation

Tables + bucketize are the Phase-A foundation. Partial residency (2×5090 box)
needs: residency table with NULL sentinels + doorbell miss-service + the
existing Markov prefetcher deciding hit rate. The threaded/paged host path is
untouched and remains the fallback regime.

## Files

- NEW `candle-kernels/src/simple/moe_bucketize.cu` + `.rs` (registered in
  `build_utils.rs` 41→42, `simple/mod.rs`)
- `candle-kernels/src/quantized/kernel.cuh` — grouped-entry `b_cnt==0` early-out
- `candle-kernels/src/simple/moe_scatter.cu` — gather `!0` row skip
- `candle-core/src/quantized/cuda.rs` — `MoeBucketizeWorkspace`, `moe_bucketize`,
  `grouped_qmatmul_dev_q8a128`
- `candle-core/src/quantized/cuda_tests.rs` — the two bit-exact test suites
- NEW `candle-transformers/src/models/expert_lre/gpu_dispatch.rs` — resident tables
- `expert_lre/handle.rs` — table build in both ctors + accessor;
  `expert_lre/compute.rs` — `extract_weight_info` pub(crate); `expert_lre/mod.rs`
- `candle-transformers/src/models/quantized_qwen3_moe.rs` — `forward_gpu_native`
  + gate

Pre-existing issues surfaced (not mine, not fixed here): clippy errors in
`causal_mask_cache.rs:134` (`mask_seq_len == mask_seq_len` tautology — looks
like a real bug worth a look) and `expert_lre/handle.rs` `routing_pinned_mut`
(`mut_from_ref`); `grouped_int8_outlier_stress` stale vs the KO pairing rule.

Uncommitted, per standing practice. `.perf_sub/` + `/tmp/perf_pristine` are
throwaway test substrates.
