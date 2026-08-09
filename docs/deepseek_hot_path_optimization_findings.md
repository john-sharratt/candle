# DeepSeek-V4-Flash — hot-path optimization findings (next round)

**Date:** 2026-08-09 (Tier 1–3 measured 2026-08-10). **Branch:** `deepseek-flash`.
**Status:** Tiers 1–3 implemented + measured (see per-item **OUTCOME** notes).

## Tier-2/3 measured outcomes (2026-08-10)
Landed (all 100% correct at [1,4,8] StoryRewrite, uncommitted): **T2.1** (`signs_cat` → per-gallery device
pointer table in `bdp_recall_batched` — kernel+FFI+wrapper; bit-exact vs the concatenated path per
`select_harness_smoke` at entries 256/2048/8192, all 100% match; removes the one O(depth) selection copy),
**T2.3** (hoisted `q_pos_dec`/prefill `q_pos`/`flat_ids` out of the layer loop), **T2.4** (validity mask
built on-device, host double-loop removed), **T2.5** (MoE double-RMSNorm removed — q8 derived from the
routed `normed` via a quantize-only launch instead of re-normalizing), **Tier-3 decode narrow-then-cat**
(one `reshape((1,n_dec,dim))` view instead of `n_dec` narrow slices). Model-level t/s flat in the shallow
[1,4,8] dev regime (all are launch-count/compute reductions masked by the WDDM floor + MoE bound; T2.1's
win is depth-scaling and not visible until production depth).
**Not landed:** **T2.2** (dead zero-inits) — needs a new public candle-core uninit API AND risks
uninit-memory garbage if any kernel doesn't write every lane, for a `cuMemset`-removal win unmeasurable
at the model level; net-negative EV, skipped. **Tier-3 dead-kernel deletion** — the non-batched
`bdp_recall`/`topm_select`/`two_stage_select` are NOT dead (live on the single-session engine decode path
`kernel_attention.rs:323`, the out-of-regime prefill fallback `:504`, and the streaming reference
`engine.rs:948`) — genuine alternate paths, not dual-path shims; doc premise was wrong. **Tier-3
`norm_w` cast** — already a no-op (`dequant_f32` loads it F32). Remaining Tier-3 (single-session
micro-uploads, ascending re-sort, Phase-B pad+cat+stack, HCA arange) are low-value / non-product-path /
conditional — assessed, not landed.

## Tier-1 measured outcomes (2026-08-10)
Landed: **T1.1** (shared q8a128 activation, bit-identical), **T1.3** (fused `rms_scale`/`rms_norm_entry`),
**T1.2d** (padded `sK` + ldmatrix B-load, +0.6% isolated decode-kernel, bit-exact). Reverted as measured
non-wins on the `latent_decode_bench` (200k-depth, 640 keys/query, baseline 0.833 ms/call): **T1.2a** int4
packs (0.837, within noise), **T1.2b** PV single-read (**0.911, −10%** — `beta[2][8]` registers cost
occupancy on the `__launch_bounds__(256,4)` kernel), **T1.2c** A-frag hoist (0.843, −1%, +8 regs occupancy).
Lesson confirmed by the kernel's own top comment: this decode kernel is occupancy/latency-bound with load
latency already hidden by the 640 independent blocks, so **memory-traffic and store-width cuts buy nothing
and register/smem additions actively hurt**. Only T1.2d (which removes an MMA *bank conflict*, not traffic)
helped. Full model [1,4,8] gate stayed 100% correct (StoryRewrite); model-level t/s flat (decode is
WDDM-launch-floor + MoE bound, so the sub-percent kernel win is invisible end-to-end).
**Model:** 284B / 13B-active MoE, single-latent K≡V attention (HEAD_DIM=512, NOPE_DIM=448, ROPE_DIM=64),
streaming MXFP4 experts, 43 layers. RTX PRO 5000 Blackwell 72 GB (sm_120), WDDM.

This document is the backlog for the round *after* the landed prefill batching (`pprep:select` 124×,
`pprep:push` batched, writeback dead-work) — see `docs/deepseek_perf_optimization_report.md`. It is a
read-only static sweep of the forward hot path across five lenses: (1) VRAM↔RAM copies, (2) GPU syncs,
(3) dtype conversions, (4) inefficient Rust/kernel code, (5) formula/op-sequence rewrites.

---

## The drain-relocation filter (read first)

Empirically proven on this codebase: **a per-layer D2H sync often only *attributes* an unavoidable
pipeline drain — removing it relocates the wait to the next sync, for ~0 wall-clock gain.** Confirmed
mirages: the append_batch f32-archive D2H, and the per-layer MoE routing readback (`indices.to_vec2`).
Every finding below was screened against this filter; the ones that survive are **genuine work
reduction** (redundant quantizes/casts/copies/allocs removed, or fewer launches on the launch-bound
decode floor), not sync relocations.

**Cross-cutting theme:** the launch-bound **decode** path never received the optimizations the
**prefill** path already has — kernel vectorization, RMS fusion, shared-activation quant. Most Tier-1/2
wins are "port the prefill treatment to decode."

Decode is WDDM launch-overhead bound (~74 ms/token floor is driver launch cost); the decode attention
kernel is memory-bound (~64% memory vs 27% SM). So launch-count cuts and memory-traffic cuts are the
real levers; pure parallel-compute cuts are hidden under the launch floor at decode.

---

## Tier 1 — highest real value

### T1.1 — Share the q8a128 activation across attention projections
- **Category:** 1 (redundant activation quant) + 3 (dtype)
- **Where:** `candle-transformers/src/models/deepseek4/linear.rs:78-85` (`QLinear::Int8::forward`),
  invoked per projection at `attention.rs:389` (`wq_a`), `attention.rs:398` (`wkv`),
  `compressor.rs:90-91` (`wkv`/`wgate`), `indexer.rs:85`/`:108` (`weights_proj`); prefill mirror at
  `attention.rs:100/113/118`. Root cost: `to_dynamic` → `quantize_acts_q8a128`
  (`candle-core/src/quantized/cuda.rs:4457`, alloc `:4467` + kernel launch `:4472`).
- **Issue:** the same `x [*, dim]` is quantized to q8a128 **independently for every projection** — ≥2
  full redundant quantizes/layer/token (`wq_a` + `wkv`), more on CSA/group-boundary tokens, ×43 layers ×
  every token. Each is a fresh VRAM alloc + kernel launch; at M=1 decode this launch overhead dominates.
- **Fix (no int8-KO weight surgery — the API already exists and the MoE path already uses it):**
  quantize `x` once, share the operand.
  - Fused/best: `QMatMul::qkv_segmented(&op, &[wq_a_q, wkv_q, ...], DType::F32)`
    (`candle-core/src/quantized/mod.rs:2337` → `cuda.rs:5117 qkv_segmented_matmul`) runs one shared
    operand against multiple KO weights **of differing formats in a SINGLE launch**, returning the
    concatenated output. Build `op = cuda::to_dynamic(&x, Int8Mode::Performance, &dev)?` once — or fold
    the pre-attention RMSNorm+quant via `cuda::rms_norm_q8a128(pre_x, attn_norm, eps, dev)` exactly as
    MoE does at `engine.rs:394`.
  - Minimal: build `op` once, then `wq_a_q.forward_dynamic(DynamicTensor::Int8(&op))` etc.
    (`mod.rs:2313`) — no new kernels, just stop routing each weight through `forward_via_int8`.
  - Plumbing: `QLinear::Int8` already holds a `QMatMul` (`linear.rs:18`); add a
    `QLinear::forward_dynamic(&self, &DynamicTensor)` and a caller that builds the shared op once per
    attention block. `Q8a128Operand` is explicitly reusable (`cuda.rs:4344 with_device_ptr`).
- **Confidence:** HIGH — removes N-1 of N quantize allocs+launches; proven pattern; not a drain
  relocation. **This is the reframed "x-proj concat" — the earlier int8-KO-weight-surgery worry was
  unfounded; `qkv_segmented` does the multi-weight-single-launch already.**
- **Hotness:** MAXIMAL (per projection × 43 × every token).
- **OUTCOME (LANDED):** `shared_int8_forward(x, &[&QLinear])` in `linear.rs` (the "minimal"
  `forward_dynamic` shared-operand path, not `qkv_segmented`). Provably bit-identical: `forward_via_int8`
  *is* `to_dynamic + forward_dynamic`, so sharing the operand only elides the duplicate deterministic
  quantize. Wired into wq_a+wkv (prefill/decode/kernel_attention/wave) and wkv+wgate (compressor
  pool/project_row/project_rows). 100% correct at [1,4,8].

### T1.2 — Port the prefill attention kernel's vectorization to the decode kernel
The decode kernel (`candle-kernels/src/paged-latent/latent_decode_kernel.cuh`) is the hottest kernel and
memory-bound, but does scalar/strided memory where `latent_prefill_kernel.cuh` already went wide. All
bit-exact with a tested prefill precedent.

> **OUTCOME:** only **T1.2d LANDED** (+0.6%, bit-exact). **T1.2a/b/c REVERTED** as measured non-wins —
> this kernel is occupancy-bound with load latency already hidden by its 640 independent blocks, so the
> traffic/store cuts (a/b) buy nothing and the register-adding hoists (b/c) drop below 4 blocks/SM. See
> the Tier-1 measured-outcomes block at the top. Do not re-chase T1.2a/b/c without new hardware.

- **T1.2a — Pack int8 stores into `int4`.** `sQ`/`sK` int8 stores are 16–32 scalar `STS.S8`
  (`latent_decode_kernel.cuh:225-229` `sQ`, `:398-402` `sK`); prefill packs them
  (`latent_prefill_kernel.cuh:399-408`). Pack the lane's DPT(=16) bytes into `uint32_t pk[4]` and emit
  one `*(int4*)&sK[key][lane*DPT] = make_int4(...)`; `sQ` = two `int4` stores. **Confidence: HIGH**,
  bit-exact (same `__float2int_rn`). Category 1/4. Lowest-risk, do first.
- **T1.2b — PV accumulate: read `kv_f` once, vectorized.** The PV loop is head-outer, so identical
  `kv_f` values are read twice (once per head) as scalar bf16 loads + convert
  (`latent_decode_kernel.cuh:486-497`) — `2 × KEYS_TILE × DPT = 256` reads/tile/thread. Restructure to
  Phase A (per head: `new_m`, `alpha`, rescale `out_reg`/`l_i`, store `beta[2][KEYS_TILE]`) + Phase B
  (single key loop: load each key's 16 bf16 once via two `int4` loads, both heads' FMA from registers).
  Halves `kv_f` smem loads + replaces 16 scalar loads with 2 vector loads/key. **Confidence: MED-HIGH**,
  bit-exact (per-head accumulation order preserved). Category 1. Biggest memory-traffic cut on the bound
  loop.
- **T1.2c — Hoist the tile-invariant QK A-fragment.** `sQ` is written once but the A-fragment is rebuilt
  from smem every tile (`latent_decode_kernel.cuh:436-445`), and the `HEAD_DIM=512` stride collapses all
  8 row-groups onto banks {0,1,2,3} → 8-way bank conflict every tile. Prefill builds it once
  (`latent_prefill_kernel.cuh:498-527`). Build `qa_frag[NPAL/WARPS][SUB/32][4]` once after the Q-stage
  `__syncthreads()`; the tile loop then does only `load_b_frag` + `mma`. **Confidence: MED-HIGH**,
  bit-exact. Category 1/4. Caveat: +8 regs/thread vs `__launch_bounds__(256,4)` — measure occupancy.
- **T1.2d — Pad `sK` + `ldmatrix` for the QK B-load.** `sK` changes per tile (can't hoist), and the
  `HEAD_DIM=512` stride gives the same 8-way B-operand bank conflict every tile
  (`latent_decode_kernel.cuh:443`). Prefill stages a padded `sK[PF_KEYS][HEAD_DIM+16]` +
  `load_b_frag_n8k32_ldmatrix` (`latent_prefill_kernel.cuh:622`). Declare decode
  `sK[KEYS_TILE][HEAD_DIM+16]` (stride not a multiple of 128) + switch to the ldmatrix loader (`sK` is
  QK-only, so padding is safe). **Confidence: MED**, bit-exact. Category 4. +256 B smem/block — verify
  4th resident block survives.

### T1.3 — Fuse `rms_scale` (and the compressor's `rms_norm_entry`)
- **Category:** 5 (formula → fewer launches)
- **Where:** `attention.rs:466-469` (`rms_scale`), `compressor.rs:654-658` (`rms_norm_entry`),
  `compressor.rs:160-164` (`Compressor::rms_norm`, reference path).
- **Issue:** `rms_scale` is an eager `sqr → mean → add → sqrt → div` (~5 launches) — but the sibling
  `rms_norm` two lines up (`attention.rs:462`) is **already** the single-launch
  `candle_nn::ops::rms_norm`. `rms_scale` is exactly `rms_norm` with unit weight. `rms_norm_entry` is
  the same eager pattern + a per-call `norm_w.to_dtype(F32)` (~7 launches). Verified against source.
- **Fix:** `rms_scale` → cache a `ones([head_dim])` F32 once, call `candle_nn::ops::rms_norm(x, &ones, eps)`.
  `rms_norm_entry` → `candle_nn::ops::rms_norm(x, &norm_w_f32, eps)` with `norm_w_f32` cached at
  construction (also kills the per-call weight cast).
- **Confidence:** HIGH (`rms_scale`, per-token × 43) / MEDIUM (`rms_norm_entry`, per-group × 43). Pure
  launch reduction, no sync to relocate. Callers of `rms_scale`: `attention.rs:102/391`,
  `kernel_attention.rs:284/406`, `wave.rs:777/1158` (more sites than first listed).
- **OUTCOME (LANDED):** `rms_scale` is now an `Attention` method over a cached unit-weight `ones_hd`;
  compressor `rms_norm`/`rms_norm_entry` fused too. CPU equivalence tests (fused vs eager, <1e-5) added.
  100% correct at [1,4,8]. (Kept even though model-level t/s is flat — fewer launches is a real
  work reduction that will surface once the WDDM launch floor lifts.)

---

## Tier 2 — solid, real

### T2.1 — Kill the `signs_cat` depth-scaling D2D copy
- **Category:** 1 (VRAM↔VRAM copy). **Where:** `gallery.rs:1289-1295` (`q_signs_cat`) and
  `gallery.rs:1309-1313` (`signs_cat`) in `two_stage_select_batched`.
- **Issue:** concatenates each deep session's **entire** `[len, sign_words]` resident sign index into one
  contiguous buffer per CSA layer per token, purely to hand `bdp_recall_batched` one base pointer. The
  sign index is **never spilled — GPU-resident at any depth** (`gallery.rs:47`), so this is an
  `O(Σ len_s · words)` D2D copy that **grows without bound with context depth** (~8 MB/layer at 64×8192,
  ~350 MB/token). The one selection cost that scales with unbounded context.
- **Fix:** give `bdp_recall_batched` a per-gallery **device pointer table** reading each gallery's
  `packed_signs()` in place — the same pointer-table shape `gather_corpus_batched` /
  `run_corpus_gather_rows_batched` already use (`gallery.rs:~1530-1564`, though that one builds the table
  from `hot_region_ptrs()`, not `packed_signs()`). Needs a kernel-signature change.
- **Note:** gallery.rs line numbers in this doc are ~15 low vs the current file (the file drifted); the
  cited regions are otherwise accurate. wave/paged/hyper/engine/attention/compressor refs are exact.
- **Confidence:** HIGH for the unbounded-context product regime; pure copy, not a drain relocation.
- **Hotness:** very high (per CSA layer × token × wave).

### T2.2 — Stop zero-initializing fully-overwritten kernel outputs
- **Category:** 4 (needless work / driver memsets). **Where:** `paged.rs:375` + `:562` (attention `out`);
  `hyper.rs:376-378` (`pre`/`post`/`comb_raw`), `:418` (`y`), `:456` (mHC `out`); also `wave.rs:916-919`
  (the four gather output buffers).
- **Issue:** these are `Tensor::zeros(...)` then **fully written** by their kernels (split-KV combine,
  mHC, `gather_corpus_batched`), so the memset is dead — ~430 needless `cuMemset` driver ops/token across
  43 layers, plus fresh pool allocations that feed the known **VRAM-pool fragmentation → decode-rate
  decay** issue.
- **Fix:** allocate uninitialized (an `empty`/uninit alloc path) or from a reused caller-owned workspace
  (as `LatentWorkspace` already does for partials). **Keep** `paged.rs:625` `comp_vmax`'s zero-init — it
  is a genuine `atomicMax` accumulator.
- **Confidence:** MEDIUM — real driver-op + pool-churn cut in a launch/fragmentation-sensitive path.
  Caveat: confirm each kernel writes all lanes (padding tiles). Candle lacks a clean uninit primitive, so
  the fix has some surface.
- **Hotness:** high (per layer × token).

### T2.3 — Hoist per-layer constant uploads out of the layer loop
- **Category:** 1 (H2D) + 4 (alloc). All layer-invariant but rebuilt + re-uploaded ×43:
  - `q_pos_dec` (`wave.rs:964-970`) — `decode_pos` is fixed for the wave. **HIGH** (decode launch-bound).
  - prefill `q_pos` (`wave.rs:1094-1098`) — `base`/`s_len` fixed per prefill seq. MEDIUM (prefill one-shot).
  - MoE `ids` (`engine.rs:385`) and `flat_ids` (`wave.rs:1205-1208`) — `token_ids` identical across layers;
    build once per step. LOW-MEDIUM.
  - glue `sl`/`ib`/`q_pos_t`/`empty_idx`/`empty_cnt` (`wave.rs:738-739`, `:1164-1166`) — LOW-MED (rare path).
- **Fix:** precompute before the `for l` loop; index inside. `comp_idx`/`comp_cnt` (`wave.rs:959-960`) do
  genuinely vary per layer — leave those.
- **Confidence:** HIGH (q_pos_dec) → LOW; all genuine upload/alloc removals, none drain relocations.

### T2.4 — Build selection masks on-device instead of host-uploading them
- **Category:** 1 + 4. **Where (remaining):** `two_stage_select_batched` validity mask (~`gallery.rs:1438-1444`,
  host double loop `validv` then `affine`). The `batched_causal_select` mask (`gallery.rs:629-643`) is
  **already on-device** (`arange`/`broadcast_lt`/`affine`, landed in commit 5d3928aa) — do NOT redo it.
- **Issue:** the validity mask builds an `O(big·mm)` mask on the host in a double loop and H2D-uploads it
  every call. It is a pure function of a small per-row count: `col_index < a.m_s`.
- **Fix:** upload only the `[big]` count vector, form the mask on-device via
  `arange(mm).broadcast_lt(&count.unsqueeze(1))?.affine(...)`. Removes the host loop + large H2D.
- **Confidence:** MEDIUM (per CSA layer × token for the batched select). Genuine host-work + copy removal.

### T2.5 — Remove the double RMSNorm in `moe_forward_batch`
- **Category:** 5. **Where:** `engine.rs:384` (`normed = rms_norm(x2, ffn_norm)`) + `engine.rs:394-399`
  (`rms_norm_q8a128(x2, ffn_norm, ...)` re-normalizes then quantizes).
- **Issue:** `x2` is RMS-normalized twice with the same `ffn_norm` (the comment at `:388` even says "same
  normalization"). The second reduction over `[nt, dim≈7168]` is duplicate work.
- **Fix:** compute `normed` once, derive the q8a128 operand from `normed` with a **quantize-only** kernel
  (group-amax + int8), not a re-norm. (Composes with T1.1's shared-operand idea.)
- **Confidence:** MEDIUM for prefill/wave (compute-bound, large `nt`); LOW for single-token decode (the
  duplicate norm is inside one already-fused launch → hidden under the launch floor).

---

## Tier 3 — small / conditional (verify before spending time)

- **Decode output narrow-then-cat round-trip** — `wave.rs:1004-1006` splits the contiguous
  `[n_dec,1,dim]` into `n_dec` narrows, `wave.rs:1194` re-`cat`s them. Push one `reshape((1,n_dec,dim))`
  view instead (free); on decode-only waves the `cat` collapses to a no-op. MEDIUM/small, bit-identical.
- **Ascending re-sort in `two_stage_select_causal`** — `gallery.rs:568-576` does a u32→f32 cast + argsort
  + `index_select` (~6 launches) to reorder `k` ids ascending, but the **batched** sibling proved order
  is immaterial (the decode reader attends the SET; softmax is order-independent; each entry carries its
  own RoPE position — `gallery.rs:1338-1342`). Drop the re-sort **iff** the single-session
  `two_stage_select` path is still live. MEDIUM conditional.
- **Phase-B `pad+cat+stack` churn** — `gallery.rs:1390-1401` (`keys_pad`) + `:1437-1448` (`sl_pad`) pad
  each ragged session then `stack` (which allocs the result anyway). Alloc `keys_all`/`sl_all` zeros once
  and `slice_set` each row. LOW-MED (tensors small: mm ≤ 1024, ihd = 128).
- **Single-session engine decode-step micro-uploads** — `kernel_attention.rs:343/344/352` rebuild
  `comp_idx`/`cnt`/`q_pos_t` per layer per token (`q_pos_t` = same value ×43). Hoist `q_pos_t`, derive the
  dense `0..k` walk in-kernel from `comp_cnt`, drop `comp_idx`. Only the single-session `decode_step`
  path (the batched wave uses `decode_capture` + one batched kernel).
- **`ape` / `norm_w` per-call `to_dtype(F32)`** — `compressor.rs:448/536/94`, `:163/657`. Real cast+alloc
  per call **iff** stored non-F32; a storage-sharing no-op if already F32. **Verify the GGUF load dtype
  first** — likely no-ops (candle's `to_dtype` short-circuits on match).
- **Out-of-regime prefill select** (`kernel_attention.rs:489-507`) — per-token `query_space` + a
  per-token `to_vec1::<u32>()` readback inside the loop. Batch the projection (as the in-regime branch
  does with `query_gemm_batched`/`rope_query_batched`); the `Vec<Vec<u32>>` return contract caps the
  readback removal. Deep-prompt fallback only, not the common path.
- **HCA `arange(0..n_entries)` grows per token** (`kernel_attention.rs:325`) — dominated by the O(depth)
  gather it feeds. LOW.
- **Dead non-batched selection kernels** — `bdp.cu`'s non-batched `bdp_recall_kernel` (`:45-70`,
  re-reads signs per head, no smem staging), `topm_threshold_kernel` (`:93-107`, single-block
  single-thread serial scan — the known landmine), and `sign_pack_kernel` (`:18-40`, scalar reads +
  per-bit branch). The **batched** versions already fixed all of these. **Confirm these entry points are
  dead on the wave path and delete them** (no-dual-path rule); do NOT leave the serial `topm_threshold`
  scan on any live path.

---

## Explicitly NOT wins — parked artifacts (do not chase)

- **MoE routing readback** — `engine.rs:409` `indices.to_vec2::<u32>()`. Intrinsic: the streaming
  `ExpertCache` schedules pinned→VRAM uploads by expert id, so routing must be host-visible. Removing it
  relocates the per-layer drain. The counting-sort host loop (`engine.rs:411-440`) hangs off it and is
  cheap vs the expert GEMM.
- **append_batch f32-archive D2H** — `gallery.rs:275-278` (`attn.to_device(Cpu)`). Proven
  sync-attribution artifact (removing it just moved the drain to `prefill:writeback`). The `attn` field is
  bench-only (`gather_selected` has no production caller) — but the D2H is not the real cost.
- **Greedy-decode readback** — `engine.rs:481/595` `argmax(...).to_scalar::<u32>()`. Required per
  generated token.
- **q8a128 quant ops** (`quantize_acts_q8a128`, `rms_norm_q8a128`, `silu_mul_q8a128`,
  `dequantize_q8a128`, `q8a128_dense_matmul`, `grouped_qmatmul_dev_q8a128`, `qkv_segmented_matmul`,
  `fused_moe_gather_q8a128`) — audited clean: no `.to_vec*`/`synchronize`/D2H inside; per-call output
  alloc is inherent. The only redundancy is the operand quantized N times (T1.1).
- **mHC / sinkhorn `expf`** — `hyper_mhc.cu:84/86`, `sinkhorn.cu:51` use the SFU, and the codebase ships
  an SFU-free `fast_exp` — but these are **bit-exact mirrors** of the Rust `cpu_*` scalar refs asserted by
  a GPU-parity test. Swapping the approximation breaks parity unless the Rust mirror changes in lockstep,
  and at decode these are tiny (n = rows, hc = a handful). Leave as-is.
- **FP64 on the attention hot path** — none. `rope_angle`/`ds_sincos` (`latent_common.cuh:150-180`) run
  once at load in `latent_rope_table_kernel`; the per-key path reads the factored table in pure f32.

---

## Recommended implementation order

**Tier 1 — DONE (2026-08-10):** T1.3 ✅, T1.1 ✅, T1.2d ✅ landed; T1.2a/b/c reverted (measured non-wins).

Remaining (Tier 2):
1. **T2.1 (`signs_cat` pointer table)** — matters most at production depth (the one selection cost that
   scales unbounded with context). HIGH confidence.
2. **T2.3 (hoist per-layer uploads)** — `q_pos_dec` HIGH (decode launch-bound); rest LOW-MED.
3. **T2.2 (dead zero-inits)** — driver-memset + pool-churn cut; confirm each kernel fully overwrites.
4. **T2.4 (validity mask on-device)** — only the `two_stage_select_batched` half remains.
5. **T2.5 (MoE double-RMSNorm)** / **Tier 3** as time allows; confirm-and-delete the dead non-batched
   selection kernels.

**Validation discipline (reinforced by the Tier-1 results):** measure every kernel change on
`latent_decode_bench` / `latent_prefill_bench` (isolated, seconds) AND the [1,4,8] model gate before
keeping it — static "obvious win" reasoning failed for 3 of 4 T1.2 items because the decode kernel is
occupancy-bound. Revert anything that improves a span but not the wall clock.

**Validation discipline (from the last round):** each change must show a real **wall-clock** t/s
improvement at `[1,4,8]` (full path filter `models::deepseek4::wave::tests::test_parallel_batched_forwarding`
— the bare name pulls in unrelated Qwen/Llama tests). Bit-exact-gate every numeric change with a CPU/CUDA
unit test first; then the model run. Revert any change that improves a span but not the wall clock (the
drain-relocation trap).
