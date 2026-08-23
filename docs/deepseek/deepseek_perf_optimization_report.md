# DeepSeek-V4-Flash inference optimization — session report

**Date:** 2026-08-09. **Branch:** `deepseek-flash`. **All changes uncommitted** (for review).
**Model:** DeepSeek-V4-Flash-0731 (284B total, 13B active, MXFP4 streaming experts), RTX PRO 5000
Blackwell 72 GB. Bench: `latent_moe::wave::tests::test_parallel_batched_forwarding` (StoryRewrite,
64-token decode, contexts ×1/×4/×8), `--features cuda,verbose,profile`.

---

## Mandate

Get single-session **decode to 50 t/s** and **prefill to 1000 t/s**. Starting point: decode ~5.5 t/s,
prefill ~21 t/s. The user's key steer mid-session: *"the main problem here is your batching — the cost
of the launch must be minimized by doing paged batching of all the key kernel calls"* (Qwen3-30B-A3B,
also MoE, hits 35–50 t/s single-session on the same box, so it is **not** a hardware wall), and
*"make sure the fused kernels you are writing are fast — we don't want to kill the patient with the cure."*

Both steers were correct and shaped the work below.

---

## The diagnosis (and a correction)

Decode and prefill are **launch-count bound**, not compute- or PCIe-bound:

- **PCIe is a non-issue:** the streaming expert cache's `pipe_fence_wait` is **0.7 ms total** across a
  64-token decode (0.01 ms/token). Uploads fully overlap compute despite a 42.8% VRAM hit rate.
- **Real compute is tiny:** the attention kernel is ~7 ms/token, the expert GEMM ~5 ms — ~15 ms of the
  163 ms/token decode. The rest is **thousands of tiny kernel launches × 43 layers**.
- **Correction:** my first instrumentation used device-synced spans, which *serialized* the pipeline and
  made MoE look like the bottleneck (~117 ms). With syncs off, the **real** decode split is
  **`decode_attn` 95.6 ms (59%)**, MoE only 42.6 ms, wave_metadata 15.5 ms. The decode lever is the
  **attention path**, not MoE. (Instrumentation now supports both: sync helpers in `profile.rs`,
  per-phase pipeline tables in `batch_test/utils.rs`, and the expert-worker profile drained via
  `BatchedEngine::snapshot_profiles`.)

**The fusion rule (learned by measuring):** a fusion helps only when the ops it removes are
*tiny/launch-overhead-dominated* (the mHC elementwise chains) or *multi-pass over the same data*
(rms_norm: 6 memory passes → 1). It does **not** help when the removed ops move hidden-sized data once
(o_lora reorder) or replace a fast parallel primitive with a single-block kernel (indexer_select). Every
change below was measured; two were reverted.

---

## Landed and validated (100% correct on StoryRewrite ×1/×4/×8)

### 1. Prefill projection batching
The per-token prep loop was **86% of prefill** — and only the attention *kernel* had been batched, not
the *projections*. Every stateless projection (attention `wq_a`/`wq_b`/`wkv` + both streaming compressors'
`wkv`/`wgate`) is now hoisted out of the token loop into a few batched GEMMs, keeping the exact
incremental pooling + per-token causal selection. Bit-exact (each matmul output row is independent of
batching). Files: `compressor.rs` (`project_rows`, `push_projected`, `push_projected_roped`),
`kernel_attention.rs` (`kernel_attn_prefill_prepare_batched`), `wave.rs`.

### 2. mHC hyper-connection fusion
The manifold-constrained hyper-connection (`hc_pre`/`hc_post`, around **every** attention and FFN
sub-block — Qwen has no analogue) was ~70 tiny eager tensor-op launches/layer. New
`candle-kernels/src/simple/hyper_mhc.cu` fuses the rms-rsqrt · sigmoid gate split, the weighted residual
reduction, and the post recombination into **3 kernels** (`hc_pre` = fn_w matmul + gates + the existing
sinkhorn kernel + reduce ≈ 4 launches vs ~28). CUDA-dispatched; the eager path stays the CPU reference and
the bit-exact oracle (`fused_pre_post_matches_eager` unit test, green). Files: `hyper.rs`, `hyper_mhc.cu/.rs`.

### 3. rms_norm fusion
`attention.rs::rms_norm` now calls `candle_nn::ops::rms_norm` (one fused launch vs ~6 eager
sqr/mean/div/mul passes — same math, `mean = Σx²/hidden`, f32 for f32 input). Called 3–4×/layer. 38/38
deepseek4 CPU unit tests green. Reduces memory traffic, so it helps decode.

### 4. Removed the redundant per-call attention-kernel device sync (data-flow fix)
Studying the CPU↔GPU data flow surfaced the biggest single issue: `paged_latent_decode_raw` and
`paged_latent_prefill_raw` each did a **full `device.synchronize()` after every kernel launch**
(paged.rs:448/669), no explanatory comment — a debugging leftover. That's **43 device drains per decode
token** (one per layer), each blocking the host for no correctness benefit (kernel + downstream consumers
share the stream, so stream ordering already guarantees completion-before-read). **Validated properly** via
the seconds-scale microbench (`latent_moe::bench::run_decode`/`run_prefill` — correctness gate + isolated
timing): both gates pass without the sync (decode rel-err 0.0060, prefill 0.0106), *then* the full model
confirmed 100% correct. `decode:kernel`'s host span collapsed **47.6 → 1.09 ms/token** (43×).

The net decode gain (~+5%) is smaller than that span drop because the cost **relocates**: with the
attention sync gone, the per-layer **MoE routing readback** (`indices.to_vec2()`, intrinsic to the
streaming 284B expert set) becomes the sole serializer and now drains the layer's async GPU work the
attention sync used to drain. (`deepseek:moe` "rising" 44.6→87.6ms is that artifact — the work is
unchanged.) Eliminating that second per-layer drain is the next decode lever — it needs GPU-native
streaming dispatch (below).

### Measured results (final `[1,4,8]` run, all 100% correct)

| Config | Prefill t/s | vs baseline | Decode t/s | vs baseline |
|--------|-------------|-------------|------------|-------------|
| ×1     | **43.9**    | 21.0 → **+109%** | **6.6**  | 5.5 → **+20%** |
| ×4     | **51.4**    | 22.9 → **+124%** | **11.5** | 8.7 → **+32%** |
| ×8     | **52.8**    | 23.0 → **+130%** | **13.7** | 10.5 → **+30%** |

**Prefill more than doubled; decode +20–32% (gain grows with batch), all 100% correct.** Rough
per-change contribution (single-session): prefill projection batching drove the prefill doubling; mHC
fusion ~+11% decode / ~+34% prefill; rms_norm ~+2–7% decode (grows with batch, −2.6% prefill, kept as
net-positive on the launch-bound decode target); the attention-sync removal ~+5% decode / +2% prefill.

---

## Session 2 — batched decode selection + per-session launch batching (+32% decode at n=8)

The mandate here was to **properly** fuse the select this time — build individual-kernel benchmarks,
study the micro-kernels, then write fast batched versions (full occupancy, vector math) — after the
first session's `indexer_select` attempt regressed by being single-block.

### Method: a per-kernel selection microbench first (`latent_moe::select_bench`)
`run_select` synthesizes a realistic decode batch (64 sessions × their own galleries at a chosen depth)
and times each selection micro-kernel **in isolation** over the whole batch, with a rigorous gate: the
batched `bdp_recall` counts must be **bit-exact** vs the per-session kernel, the batched `topm` must be
a **valid top-M** (tie-agnostic), and the end-to-end selected-gid set must match the per-session loop on
correlated (realistic) data. Running it (`select_harness_smoke`, seconds) gave the decisive measurement:

| stage (64-session batch, per iter) | ms | nature |
|-----------------------------------|----|--------|
| sign_pack | ~1.0 | minor |
| bdp_recall | ~1.4 | already efficient |
| **topm_select** | **~15.0** | **dominant — pure launch/serial-scan overhead** |
| rescore (matmul) | ~5.2 | cuBLAS, launch-bound |
| argsort×2 | ~3.7 | launch-bound |

`topm_select` was **58% of the selection cost** — its `threshold` kernel is a single-warp serial scan
over ~8193 histogram bins, launched once **per session**. (This is exactly what the first session's
single-block fusion missed: it fused the 5 ms rescore matmul, not the 15 ms `topm`.)

### Batched selection (`bdp.cu` + `two_stage_select_batched`)
New `bdp_recall_batched` / `topm_select_batched` kernels fold the whole wave into **one launch per
Stage-1 kernel** (session = a grid dimension; the query heads staged in shared memory). Stage 2 batches
too: one padded `bmm` + one batched argsort + one batched gather across sessions (padding columns masked
to −∞; the selected set is order-independent for the decode reader). A **shallow-skip** partition runs
Stage 1 only for sessions past the shortlist width, so short conversations pay nothing (their per-session
all-pass fast path is preserved). Microbench: **2.8× shallow, 4.1–4.3× at depth, 100% gid-set match**,
counts bit-exact.

### The measurement redirected the bigger win: batch every per-session decode launch
Wiring the batched select into the wave and profiling showed selection is only ~4.5% of StoryRewrite
decode (short story → shallow galleries → the loop's all-pass path, no `topm`) — but it exposed the real
decode cost: **the wave ran the attention projections, the corpus-push compressor projections, and the
o_lora output projection in per-session loops.** Each is a stateless GEMM run once per session; batching
them across sessions is bit-identical (rows independent) and collapses the launch count:

- **Attention projections** (`wq_a`/`wq_b`/`wkv` + norms): one GEMM over all decode rows. `dprep:proj`
  3226 → 422 ms (n=8).
- **Output projection** (`output_proj` is already batch-parametrized — its inner loop is over the 8
  o_lora groups, not sessions): one call with `b = n_dec`. `decode:outproj` 7121 → 861 ms.
- **Compressor-push projections** (attn-comp + indexer-comp, shared layer weights): batched
  `project_rows`, then per-session `push_projected*` (the stateful pooling stays per-session).

### Measured results (StoryRewrite, `--features cuda,verbose,profile`)

| Config | Decode t/s (session 1) | vs session-1 baseline |
|--------|------------------------|-----------------------|
| ×1     | 6.63                   | 6.6 (flat — 1 session, nothing to batch) |
| ×8     | **18.1**               | 13.7 → **+32%** |

`decode_total` (n=8) 37.4 → 28.2 s; `deepseek:decode_attn` (the span wrapping the whole decode
attention block) 24.9 → 12.5 s. **All 100% correct** (StoryRewrite n=1 1/1, n=8 8/8, including the
historically-flaky n=8). The win **scales with concurrency** — n=1 is flat (nothing to batch), n=8 is
+32%, and the production target is 64 concurrent sessions, where it grows further. The batched selection
additionally wins **4× at unbounded depth** (where `topm` actually runs), which StoryRewrite is too
short to exercise.

### Kernel-level tuning of the two batched Stage-1 kernels (ncu)

With the batched kernels wired, the two Stage-1 kernels were profiled individually with Nsight Compute
(`select_kernel_bench` example loops just `bdp_recall_batched` → `topm_select_batched` at a deep
64-session × 8192-entry batch). ncu named the real cost immediately:

| kernel | before | after | how |
|--------|--------|-------|-----|
| `bdp_recall_batched` | 90 µs | **37 µs** (2.4×) | `uint4` (128-bit) vectorized smem+global sign loads — L1/TEX-bound (95%) → compute-bound (89%); L1/TEX 95%→26%. Entry signs hoisted to registers. |
| `topm_threshold_batched` | **301 µs** | **7.3 µs** (41×) | the launch was `<<<64, 32>>>` with only thread 0 live — 64 threads total each serially scanning ~8193 histogram bins. Replaced with a 256-thread block scan: chunked partial sums + a Hillis-Steele block scan locates the one chunk straddling the m-th element; only that chunk's `⌈bins/256⌉` bins are scanned serially. O(bins) → O(bins/blockDim + log blockDim). |
| `topm_hist` / `topm_compact` | 14 / 10 µs | 13 / 9.5 µs | atomic-bound, already small |
| **Stage-1 total (isolated)** | **325 µs** | **79 µs** | **4.1×** |

`topm_threshold` was 74% of the Stage-1 cost purely because of a lazy launch config — the single most
impactful fix here. `bdp` is now compute-bound near the roofline (256 popcounts/entry is the algorithm's
inherent work at 88% occupancy). All correctness gates (counts bit-exact, valid top-M, 100% gid-set
match) stay green. New ncu target: `cargo run --example select_kernel_bench` + `ncu --kernel-name
"regex:batched_kernel"`.

### Batched indexer query GEMM + the `wave_metadata` analysis

The last per-session projection in the decode loop — the Indexer's `query_space` (`wq_b` + `weights_proj`,
inside `decode:prep`) — was batched the same way: `Indexer::query_gemm_batched` does both GEMMs over all
decode rows in one call, `rope_query` keeps only the cheap position-dependent RoPE per session
(bit-identical per row). Validated 100% correct (n=1/8); `decode:prep` 7107 → 6641 ms. Modest, because the
profile shows `decode:prep` is now dominated by **`dprep:push` (4164 ms, 63%)** — the *stateful* compressor
pooling + gallery append, which is staggered per session (groups complete on different steps) and does not
batch cleanly — plus per-session loop overhead.

The single largest remaining decode cost is **`wave_metadata` (8124 ms, 29% of decode)**, unchanged by the
batching. It is `build_decode_metadata_at`: per step it rebuilds each session's `position_map` from scratch
(O(context) per session), then runs a **43-layer loop** each doing `resolve_arena_info` (storage read-lock +
arena walk), `sync_decode_gpu_chunks` (state write-lock + `validate_decode_state` + slot reuse), and
`ensure_for_batch_entries`. The cost is death-by-a-thousand-cuts across many small **locked, correctness-critical
KV-cache scheduler operations** — this metadata drives the paged-attention kernel's memory addressing, and
StoryRewrite is the only validation. The clean *structural* win is caching the `position_map` incrementally
(the `[0, offset)` prefix is identical between steps; only the write-slot entry changes) — O(context²) →
O(context) — but that pays off at production depth, not on a 64-token rewrite, and needs careful invalidation
on chunk-boundary/eviction. Deliberately **not** attempted unsupervised (per the "don't kill the patient"
steer); mapped here as a dedicated, well-gated follow-up.

### `wave_metadata` → near zero: the live persistent slot buffer (77×)

The single largest decode cost, `wave_metadata` (8067 ms, 29% of decode), was **`build_decode_metadata_at`
with `snapshot_slices=true`** — a per-layer, per-step *copy* of each session's slot-state into the wave's
pinned generation (the `decode:slot_reuse` = 7471 ms). The other models already avoid this: Qwen/Llama's
standard decode calls `build_decode_metadata` with **`snapshot_slices=false`** — the header points at the
**live, persistent, self-incrementing** `gpu_chunks` buffer, and the decode kernel commits the write-len
on-device (`commit_write_len=true`) so it advances for the next step. The snapshot copy exists only to let
prefill's up-front per-token snapshots survive later chunk-boundary reallocations; a **pure-decode wave
needs none of it**.

Fix (per **sequence**, so it applies to EVERY wave — pure-decode *and* mixed/continuous-batching, not just
pure-decode): `build_decode_metadata_at` now takes `snapshot_seqs` instead of a single `snapshot_slices`
bool, and `sync_decode_gpu_chunks_snapshot` takes a per-entry mask. A **decode row always uses the live
path** — its write chunk is pre-ensured, so it never reallocs during the layer loop — with the kernel
committing the write-len on-device. Only the rows that mutate the arena mid-forward — **prefill** (absorbs
across chunk boundaries in one launch) and **glue** (gap-chunk scatter) — still snapshot, so their
`slices_ptr` survives the reallocation. A prefill-only wave (mask all-true) is byte-identical to the old
snapshot path; a decode-only wave (mask all-false) is the cheap live path; a mixed wave gets live decode
rows + snapshot prefill/glue rows. Gated by a new fast correctness test,
**`decode_live_buffer_matches_snapshot_multistep`**: two independent
sessions run 72 identical decode steps across 3 chunk boundaries (positions 32/64/96), one on each path,
asserting **bit-identical** output every step (the live buffer's on-device write-len self-increment must
track the snapshot's offset-derived write-len exactly). The existing `arena_backed_matches_synthetic`
already proved the single-step live path bit-exact.

Measured (n=8): **`wave_metadata` 8067 → 104.9 ms (77×, near zero)**, `decode:slot_reuse` 7471 → 23 ms,
`decode_total` 28.0 → **19.1 s**. **Decode throughput n=8: 13.7 → 26.8 t/s (+96%, nearly doubled)** over
this session's baseline; n=1 6.6 → 7.5. **100% correct** (StoryRewrite 1/1, 8/8). The residual ~105 ms is
the position_map rebuild + per-layer header/arena resolve; caching the position_map incrementally
(O(context²) → O(context)) is the remaining lever, and it grows in importance at production depth.

### Fused corpus gather (`decode:gather` + `decode:cache`, 3.4×)

After the metadata fix, the next host cost was assembling the selected corpus rows: the wave did four
per-region `index_select`s per session (`gather_corpus`) then four cross-session `Tensor::cat`s to build
the attention kernel's contiguous block — `decode:gather` 1.7s + `decode:cache` 1.0s (n=8). New kernel
`corpus_gather.cu` (`run_corpus_gather_rows`) gathers a session's `k` selected rows across all four
hot-cache regions (`nope_i8`/`nope_scale`/`rope_bf`/`pos`) in ONE launch, writing straight into a
pre-allocated shared block at the session's row offset — so there is no per-region select and no cat. Tier-
aware: hot galleries gather on-GPU, a spilled gallery re-heats its `k` rows from CPU RAM and `slice_set`s
them in. Gated by `gather_corpus_into_matches_reference` (bit-identical to `gather_corpus` across all four
regions at a non-zero offset). Measured (n=8): `decode:gather` 1697 → **392 ms** (4.3×), `decode:cache`
998 → **395 ms** (2.5×), `decode_total` 19.5 → **17.8 s**; **decode n=8 26.8 → 28.8 t/s** (100% correct).

**Then tuned the kernel to death (ncu, `corpus_gather_bench`).** The first cut fired the gather PER SESSION
(one launch/session), which ncu showed underfilled the GPU: 3.8 µs × 64 launches, **0.39 waves/SM, 23%
occupancy**, ~290 µs of pure launch overhead. Batched it to ONE launch across all hot sessions via a device
pointer table, then iterated: `uint4`-vectorized region copies, 4 warps/block (one warp/row) for occupancy,
**host-cached region base pointers** (invalidated on `grow_to`/`maybe_spill`), **per-session gid pointers in
the table** (the biggest wall win — killed the `O(sessions)` gid `cat`), and packed metadata (2 small
uploads). Result at 64 sessions: **533 → 29 µs (18×)**, kernel **90.5% occupancy, 66.6% memory throughput,
26 µs**, and the wall is now **flat in session count** (26 µs at 8, 16, 64). Gated by
`gather_corpus_batched_matches_reference` (bit-identical per session across 5 galleries of different sizes at
distinct offsets). In the wave (n=8): `decode:gather` 392 → **169 ms**, 100% correct; the 18× shows at the
64-session production batch.

### Lesson refined
The first session's rule ("fuse only launch-overhead / multi-pass work") holds, but the decisive step
was **measuring per-kernel cost first**: it named `topm` (not the rescore matmul) as the selection
bottleneck, and it exposed that the *projections* — not the selection — were the dominant per-session
decode launch cost. Batch the per-session loops (bit-identical GEMMs), and validate the one genuinely
approximate kernel (recall shortlist) with a tie-agnostic gate, not a false determinism assertion.

---

## Attempted and reverted (kept for the lessons)

- **`indexer_select` fusion** — fused `two_stage_select`'s stage-2 (rescore + top-k + ascending sort)
  into one kernel. The parity unit test passed, but the model **regressed −29% prefill** (44→31 t/s): a
  single block uses one SM, replacing a fast cuBLAS matmul (all SMs) + a parallel argsort. Prefill fires
  ~16,800 selects, so each slower call dominated. Reverted; kernel files removed.
- **o_lora `output_proj` reorder** — group-major permute to drop the 8 per-group `.contiguous()` copies.
  Numerics-identical but **neutral-to-negative**: those copies move real data, not launch overhead.
  Reverted.

---

## Remaining levers (mapped, not attempted unsupervised — correctness risk)

1. **Prefill Stage-2 batched selection** — the current per-token select loop (~16,800 `two_stage_select`
   calls) is the prefill bottleneck after Stage 1. Batch it: build the corpus fully (recording each
   token's visible group count `G_t`), then one batched rescore matmul `[s·h, ih]×[ih, G]` + a causal
   mask (`g ≥ G_t → −inf`) + a batched per-row argsort. **Risk:** exact only when `G ≤ m` (no recall);
   for `G > m` the batched exact top-k *diverges* from the per-token recall-approximate path, breaking
   prefill≡decode parity on long prompts the tests don't cover. Needs a size-regime fallback and a
   self-validating `assert(batched == per-token)`. This is the single biggest remaining prefill win and
   the right next focused task.
2. **MoE routing readback — now the #1 decode serializer** (confirmed after the sync removal). The
   per-layer `indices.to_vec2()` drains the whole layer's async GPU work; with it gone, host (~64ms/token)
   and GPU (~87ms/token) would overlap → decode ceiling ~max(64,87) ≈ 87ms ≈ 11.5 t/s single-session.
   It's intrinsic to the streaming 284B set: to dispatch on-GPU without the readback the routed experts
   must be resident before the GEMM, but at a 42.8% hit rate the misses must be scheduled from host.
   Removing it needs GPU-native dispatch tolerant of partial residency (`gpu_dispatch` today requires
   `all_resident`, which Qwen has and DeepSeek cannot) — plus, for single-session, a cache policy that
   keeps the (roughly-fitting ~100–150 experts/layer) working set resident so misses become rare. Deep
   `expert_lre` work. Note PCIe itself is *not* the cost (`pipe_fence_wait` = 0.7ms total); it's the
   compute+launch of the expert GEMM, serialized by the readback.
3. **x-projection concat** — fuse `wq_a`/`wkv` + the compressors' projections into one matmul via int8-KO
   weight concatenation at load (keeps the matmul fast). Delicate quant-weight surgery.

Honest assessment: the three landed fusions + the sync fix + prefill batching roughly **doubled prefill
and lifted decode ~20%**, but the headline targets (50 t/s decode, 1000 t/s prefill) require levers 1–2
above. Those are genuinely deep and correctness-critical; they were deliberately not attempted without a
human to validate a subtle break (per the "don't kill the patient" steer). The **method** is now proven:
study the data flow / existing kernels, write the change, and validate it in the seconds-scale microbench
(correctness gate + isolated timing) *before* the 20-minute model run — that's how the sync fix was
de-risked and how the reverted `indexer_select` should have been caught earlier.

---

## Files changed (uncommitted)
- **candle-kernels:** `src/simple/hyper_mhc.cu`, `src/simple/hyper_mhc.rs` (new); `src/simple/mod.rs`,
  `build_utils.rs` (register the mHC kernel).
- **candle-transformers/src/models/latent_moe:** `hyper.rs` (mHC fusion + parity test), `attention.rs`
  (rms_norm fusion), `compressor.rs` + `kernel_attention.rs` + `wave.rs` (prefill projection batching +
  fine-grained instrumentation + `snapshot_profiles`/`expert_stats` overrides), `engine.rs` (MoE
  instrumentation + `experts()` accessor).
- **candle-transformers/src/models:** `profile.rs` (sync-bracketed span helpers, currently un-synced for
  real-throughput measurement), `batch_test/utils.rs` (per-phase prefill/decode profile tables).
- (Also present in the tree: the earlier turn-seal persistence work in `gallery.rs`/`indexer.rs`/
  `kernel_attention.rs`/`mod.rs`/`wave.rs`.)
