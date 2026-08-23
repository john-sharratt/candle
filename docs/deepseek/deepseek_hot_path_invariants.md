# DeepSeek-V4-Flash — Hot-Path Invariants

**Goal:** compute-bound single-session **prefill ≥ 1000 t/s** and **decode ≥ 50 t/s**.
Prefill sidesteps the expert-PCIe wall (it amortises the streaming-expert copy over a
large token count), so it *should* approach the ~1400 t/s compute bound. It is currently
capped at ~558 t/s. This document is the study that produces the invariants the hot path
must satisfy to get there — every violation below is a place the current code leaves the
GPU idle, round-trips through the host, or does work the architecture says is unnecessary.

The hot path is `forward_wave` (`wave.rs`) and everything it calls per layer, per wave:
projections → compressor assemble/pool → gallery append → select → gather → attention
kernel → out-proj → MoE. Two phases share it: **prefill** (bulk, ragged multi-token
prompt slots) and **decode** (one token per session, batched across sessions).

---

## Copy-elimination pass (invariant 2 / 2b)

Invariant 2 was originally worded as "no `contiguous` / `force_contiguous`", naming two
functions rather than the operation. `Tensor::cat` and `slice_set` matched neither name
and were therefore never audited, even though each allocates and copies for exactly the
same reason and costs **one launch per argument**. A profile of the post-fusion sweep
found **1,079,568 copy launches (6.1 % of GPU, ~120 per layer-step)** sitting entirely
outside the rule — `copy2d_f32` 746 k, `ucopy_f32` 186 k, `copy2d_u32` 146 k — with every
median at the WDDM launch floor (~0.9–1.9 µs), i.e. launch overhead rather than bandwidth.
For scale, the three fused kernels of the preceding campaign total 0.2 % of GPU.

The invariant is now worded against the OPERATION, and gains a corollary — **2b: a kernel
consuming per-session or per-row data takes a DESCRIPTOR TABLE, not a packed block**.
Requiring one dense base pointer is what forced callers to concatenate, so the copy was
the kernel's API bug, not the caller's.

Three sites were converted:

1. **Compressor pool** (`simple/compressor_pool.cu`). The kernel now takes a per-group
   list of SEGMENTS — `{kv, score, ape} × {ptr, row stride, rows}` — walked in order, and
   folds the `ape` bias in as it reads. Consequences: the overlapping compressor's two
   column windows need no join; a decode group's `ratio` independently-allocated rows are
   read where they lie and retained as the next group's previous half by moving a `Vec`;
   ragged groups (a turn-sealed partial) need no pad; a group 0 with no previous group
   omits the segment instead of pooling an all-`−∞` pad; and several sessions pool
   together by widening the table. `assemble_group_raw` went from ~13 launches per group
   to **zero device work at all**. The batched prefill assemble lost its prev-source
   concatenation, its `pi_t` gather and its pad rows — the prev-group shift is just the
   same block read one group earlier — which also removed `cd`, `overlap` and
   prev-presence from the geometry cache key.
2. **Gallery append** (`simple/rows_scatter.cu`, new). Appending a wave's groups is a
   scatter: six arrays × one destination per session per compression layer, i.e. six
   `slice_set` launches × width × layers, each moving a few kilobytes. It is now one
   sign-pack, one position upload, one latent seal and ONE scatter whose descriptor table
   names every destination by address — a fixed handful of launches for the whole wave,
   independent of session count. Each gallery's tier is settled (`grow_to`, `maybe_spill`)
   *before* any row is written, so every row is written once to its final home.
3. **Warm-tier re-heat** (`FloatGallery::gather_corpus_into`). The spilled float pair lives
   in **pageable** host RAM, which no kernel can address, so the row gather itself must
   stay on the host — the one place invariant 4 does not reach, because there is no kernel
   that could do it. What it no longer is, is a copy per region: all three regions pack
   into one staging buffer, so the re-heat is a single transfer plus a single scatter
   instead of three uploads and four `slice_set`s. `gather_corpus` now routes through the
   same function, so the tier logic exists once. Making the kernel read this tier in place
   needs the **pinned warm pool** of `kv_tier_migration.md`; that is deliberately a bounded
   *pool*, because page-locking every spilled arena wholesale would make an unbounded
   amount of host memory unswappable.

`keys.t()?.contiguous()?` (four `gallery.rs` sites) was **measured out**, not fixed:
candle's matmul accepts the strided transpose and all gates pass without the copy, but the
model A/B lands inside run-to-run variance (~2 % same-code spread), which points at cuBLAS
selecting a slower kernel for the strided operand and roughly cancelling the saved traffic.
It belongs with the GEMM work, under `nsys` attribution — not with deletion.

Measured: model sweep `[1,4,8,16,1]` 5/5 valid throughout; **decode cfg16 56.6 → 61.2 t/s
(+8.1 %), cfg8 41.3 → 43.8 (+6.1 %)**, bulk flat at ~1000 t/s. The decode gain is entirely
site 1 — sites 2 and 3 measured flat on this benchmark, because the sweep's galleries never
reach the spill threshold and an append fires only on a group close. They are structural
fixes whose payoff is at production depth and width.

One trap worth recording: the batched append first assumed one wave slot per sequence. A
**speculative verify wave carries a whole drafted block as consecutive decode rows of the
SAME sequence**, so one gallery legitimately receives several slots' groups in one wave.
Only `wave_paris_speculative_dspark` caught it — the unit gates and the non-speculative
sweep both passed. Slots are now grouped by sequence and appended back to back.

### `ncu` on the two new kernels — the bottleneck is the TABLE, not the kernels

Both kernels got an isolation harness (`bench_compressor_pool`, `bench_rows_scatter`,
min-of-five-rounds) and an `ncu` pass. The result redirected the whole optimisation:

| shape                              | kernel  | wall    | SM %  | DRAM % | warps % |
|------------------------------------|---------|---------|-------|--------|---------|
| `compressor_pool` decode ×1        | 5.2 us  | 15.6 us | 0.13  | 0.65   | 8.1     |
| `compressor_pool` prefill (g = 256)| 11.4 us | 22.4 us | 15.6  | 58.0   | 52      |
| `rows_scatter` 64 sessions         | 2.9 us  | 26.8 us | 6.1   | 2.8    | 32      |

At every shape but one the kernel is a minority of the wall, and at the decode and
scatter shapes it is doing essentially nothing (0.13 % / 6.1 % SM). Occupancy, register
pressure and vector width cannot move a kernel that is 0.13 % SM-busy — **the wall is the
descriptor upload**. Measured directly: `Tensor::from_vec` of a 928-word table costs
**7.6 us**, against a 5.2 us kernel, because it allocates a fresh device buffer per call
and copies from pageable host memory. `desc.rs` now stages both kernels' tables through a
persistent per-device buffer; the per-call allocation is gone (prefill 22.4 → ~20.5 us,
scatter 26.8 → 25.5 us) and the remaining ~7 us is the pageable H2D itself, which is
irreducible without either shrinking the table into kernel parameters (32 KB limit, the
prefill table is 22–33 KB) or batching the whole wave's uploads into one — both open.

**The table now rides the wave's pinned arena.** `PinnedStager` — a bump arena mapped
`DEVICEMAP | WRITECOMBINED`, already used by the paged-attention path for its slice and
header metadata — was already open for the whole layer loop (`begin_stager_generation`,
`wave.rs`). Staging the descriptor from it is a pointer increment, and `submit` on a bump
buffer hands back a device address into the SAME pinned pages: **the GPU reads the table
in place over PCIe, so there is no copy, no allocation and no driver call**, and freeing
is by lifetime when the generation drops. Isolated:

| shape                        | arena   | upload  | vs `slice_set` |
|------------------------------|---------|---------|----------------|
| scatter, 1 session           | 8.1 us  | 13.2 us | 5.0x           |
| scatter, 16 sessions         | 10.2 us | 16.6 us | 68.0x          |
| scatter, 64 sessions         | 17.1 us | 25.2 us | 159.8x         |
| pool, 64 streaming pools     | 22.2 us | 30.1 us | —              |
| pool, 16 streaming pools     | 16.9 us | 17.7 us | —              |

The win scales with table size — 1.5–1.7x for the scatter at every width, 1.36x for the
pool at 64 pools — and vanishes below ~16 pools, where `submit`'s arena lookup costs about
what the copy did.

There is NO second staging path. A caller outside a wave — the streamed reference, a
turn-seal `close`, the gates — does not LACK an arena, it simply has not opened a SCOPE
yet, so it opens one with `desc::scope`. Generations are refcounted, so a scope opened
inside a wave nests harmlessly and a standalone one resets at its own end. The alternative
was an `Option<&Generation>` with a copy-based fallback, and a shared fallback buffer is
only sound when at most ONE table is staged per launch — which stopped being true the
moment `gather_corpus_batched` staged its pointer table and its metadata before a single
kernel. The second overwrote the first (`CUDA_ERROR_ILLEGAL_ADDRESS`, half the gallery
gates). **Staging lifetime cannot be inferred from call order**; the arena gives every
`alloc` a distinct region, so the hazard cannot arise.

The wave's arena draw is now COMPUTED (`desc::wave_desc_bytes`) from its own width, layer
count and ratio, and `debug_assert`ed against half the 128 MB arena at
`begin_stager_generation` — the paged metadata shares it, so a future width or ratio trips
the assert instead of silently allocating overflow arenas mid-wave.

**Two optimisations were tried and MEASURED WORSE. Do not re-attempt either.**

- **Pinned write-combined staging for the table: 15 → 80 us, a 5× regression.**
  `PinnedHostSlice::as_mut_slice` calls `event.synchronize()`, and with one reusable
  buffer that drains the entire queued stream on every call, throttling the host to GPU
  rate. A ring deep enough to avoid it cannot be bounded, because the host legitimately
  runs hundreds of iterations ahead.
- **`float4` in the pool: prefill 20.1 → 21.1 us, HCA 19.1 → 33.5 us.** `d` is fixed at
  512, so four channels per thread means a quarter of the lanes and the channel tiling
  collapses from 4 blocks per group to 1. Narrowing the block to win the blocks back made
  it far worse — `ncu` measured block size 32, grid 64, **2.08 % of warps active**, 62 us —
  because an SM seats a bounded number of resident BLOCKS, so one warp per block caps
  occupancy. The general rule: vectorising a loop whose trip count is fixed by the data
  trades parallel work items for per-thread work, and on a kernel short of warps rather
  than short of bandwidth that trade is always backwards. It would need `d` ≥ 2048.

What the harness DID confirm is the batching itself: against the `slice_set` storm it
replaced, `rows_scatter` is **3.0× at one session, 41.6× at sixteen and 105.7× at
sixty-four** — the launch count stops scaling with width, which is the whole point.

### Second pass — the remaining per-sequence uploads

An audit of every per-call pageable H2D on the hot path found ~14 `Tensor::from_vec` sites
inside the layer loop plus 6 in the gallery helpers. The corpus gather's pointer table and
metadata, the BDP recall's per-gallery sign table and the two-stage selector's `off`/`cnt`
segments were converted to arena staging directly (the selector's pair is now staged ONCE
and shared by the recall and the top-M, which also removed a duplicated upload). Three
further sites needed structural changes rather than a staging swap:

1. **`comp_idx` expansion** (`simple/comp_idx.cu`, new). Every decode slot's selection is
   the dense range `[offset, offset+k)`, so the whole `[n_dec, max_sel]` matrix follows
   from two numbers per slot — yet it was built with two uploads plus a five-launch tensor
   chain (`arange`, `broadcast_add`, `broadcast_lt`, `full`, `where_cond`), every layer,
   every step, around about thirty words of input. One kernel now expands a staged
   `{offset, count}` table and emits `comp_cnt` in the same pass, because the decode kernel
   wants the counts as a device array anyway. Seven launches and two uploads → one launch,
   no upload. **This is the site my first framing got wrong**: `offsets_t`/`comp_cnt` were
   filed as "consumed as tensors by the paged wrappers, same conversion as the rest", but
   they never reach a kernel pointer at all — they feed candle ops. A staging swap would
   not have compiled, let alone helped. Check what actually consumes a tensor before
   classifying it.
2. **Prefill slot metadata** (`paged_latent_prefill_raw`). `seq_of` (one u32 per query) and
   `new_meta` (four per seq) are read once by the kernel and never by a tensor op, so the
   wrapper now takes them as device addresses alongside the `headers_ptr` that already was
   one. The single-seq convenience wrapper stages its own 1-seq tables — notably `seq_of`,
   which as a tensor was an allocation plus a memset for a block of zeros.
3. **Arena writeback** (`latent_glue_scatter_kernel`, now batched). Three separate wave
   phases scatter latents into reserved chunks — glue islands, prefill prompt writeback and
   speculative-verify writeback — and each ran one launch per SEQUENCE plus two uploads for
   its `{slice, in_blk}` arrays. Every run names its own destination slot by address, so
   there was nothing to serialise: the kernel now takes a run table (`{kv, headers, slices,
   in_blk, rows}`, `grid.y` = run) and the whole wave's writeback is one launch. The index
   arrays stay HOST vectors and go straight into the arena rather than becoming tensors.

The budget (`desc::wave_desc_bytes`) now covers all of it and takes a `max_tokens`
argument, because the writeback and prefill metadata are the first draws that scale with
TOKENS rather than slots — a long prompt stages two words per prompt token.

Measured: sweep `[1,4,8,16,1]` 5/5 valid, **cfg16 decode 61.8 t/s, bulk 1002.4** against
61.7 / 1000.1 before — i.e. **flat, inside run-to-run variance**. The launches are
genuinely gone, so the honest reading is that they were not on the critical path at these
shapes: the decode step is ~16 ms and these sites were order 2–3 ms of *driver* overhead
that evidently overlapped with GPU work rather than gating it. The value banked here is
structural (invariant 2b holds across the whole wave now, and the writeback stops scaling
with sequence count), not a rate win. **Where the remaining time goes is now an open
question that only the re-profile can answer** — which is the next task, not more
conversions.

---

## Campaign status (measured)

The invariant campaign landed **every** structural fix; only three residuals remain,
each with a measured/analysed reason. Commits on `deepseek-flash`: 4d5246d2,
c9305f8a, 30a05446, 74abb3e6. Every step held the bit gates
(mirror/gather/compressor) and model StoryRewrite [1,4,8,1] at 100% valid; config-8
prefill 558 → 593 t/s.

**Fixed:** Inv 6 (`Tensor::empty` — no zero-then-overwrite, 5 hot sites); Inv 2
(`force_contiguous` per-row storm consolidated to one bounded retained-state copy;
no-op `contiguous` dropped); Inv 3/4 decode comp-idx built on-device (broadcast);
Inv 5 batched `lm_head` + batched prefill out-proj (span 1448→46 ms) + **multi-slot
prefill attention kernel** (below); Inv 1 emit-type **output** (attention kernels
emit F32 → int8 out-proj, no widening cast); Inv 1 #8 (`ape`/`norm_w` F32 at load →
compressor casts are proven no-ops).

**Inv 5 multi-slot prefill attention kernel — BUILT (74abb3e6).** Prefill attention
was the last per-seq loop; it now fires ONE kernel launch per wave across the whole
prompt fleet (a per-query `seq_of[qi]` picks each query's arena slot + fresh-diagonal
slice; the corpus is packed via `gather_corpus_batched`). `prefill:kernel` collapses
from ~65 ms (×688 per-seq launches) to ~40 ms (×129 = one/wave). As predicted the
**wall is flat** (~593 t/s config-8 — prefill is MoE/readback-drain bound, not the
attention kernel), so this closes the invariant *structurally*, not as a speed win.
One numerics note: at multi-seq the int8-V scale `comp_vmax` is global over the packed
corpus (was per-seq) — inside the int8-PV tolerance envelope; 1-seq mirror gates stay
bit-exact (1 seq ⇒ global == per-seq).

**Residuals — analysed, not productive to fix:**
- **Inv 1 emit-type INPUT** (q/kv F32→BF16) — necessary boundary conversion. The
  kernel wants bf16 input (an f32-input kernel would 2× the per-tile re-read); the
  cast is the f32-norm→bf16-kernel rounding. Eliminating it needs a candle-nn
  `rms_norm` that stores bf16 — a fork for marginal gain.
- **Inv 4 MoE counting-sort** (`engine.rs:411-444`) — rides the **sanctioned**
  expert-id readback (`indices.to_vec2`, needed to schedule streaming pinned→VRAM
  uploads). The sort post-processes already-host data; `submit_moe_work` consumes host
  `assignments`. Moving it to GPU means re-uploading + reworking the bit-exact
  streaming submit for a host sort whose visible cost is GPU drain, not sort compute.
- **Inv 3/4/5 out-of-regime recall** (`kernel_attention.rs:621` + wave Host arm) —
  deep-prompt only (corpus wider than the shortlist), NOT exercised by the gate. The
  per-token *recall* (not exact top-k) is required for prefill≡decode parity, so it
  can't be naively batched; a parity-safe batched on-device recall is a large,
  unvalidatable-by-current-gate effort for an uncommon path.

**Doc corrections found while verifying:** `paged.rs:208-210` (`CorpusCache::build`)
and `gallery.rs:1704/1766` are test/bench-only, not hot sites (Inv 6 had 5 real
sites, not 7).

The priority list at the bottom predates these measurements: the "multi-slot prefill
kernel" is now BUILT (structural, wall-flat, above), and "GPU MoE bucketize" is the
MoE-sort residual (rides the sanctioned readback).

---

## The invariants (the destination)

1. **No `to_dtype` in the loop — kernels emit the final type.** Every dtype conversion on
   the hot path is a full-tensor pass over memory that a kernel could have avoided by
   writing its output in the type the next consumer wants. Norms emit the kernel's input
   type; attention kernels emit the out-proj's input type.

2. **No `contiguous` / `force_contiguous` in the loop — add paged/strided support where a
   copy is currently forced.** A `contiguous()` is an allocate-plus-copy of the whole
   tensor. If a consumer needs a specific layout, teach it to read the layout that exists
   (offset + stride), or produce the layout directly from the kernel that made the data.

3. **No unnecessary GPU→CPU transfers.** Two sanctioned exceptions: (a) MoE expert routing
   (`indices` → host), because the streaming `ExpertCache` schedules pinned→VRAM uploads by
   expert id; and (b) the embedding lookup (token ids → host, CPU `index_select`, transfer
   in), because it is a pure index + transfer with no compute and keeps the embed table off
   VRAM. Everything else — comp-idx assembly, select remaps, gather indices — stays on the
   GPU.

4. **Run as much as possible on the GPU.** No host-side compute that a kernel can do:
   no host counting-sort, no host set-union/dedup/remap, no host embedding gather. Host
   code issues launches; it does not compute over per-token data.

5. **Everything in prefill and decode runs fully batched.** One launch over all slots /
   all sessions, not a per-seq or per-token loop. Decode already batches select, gather,
   attention, and out-proj across sessions; prefill must reach the same shape — a
   multi-slot attention kernel over all prompt slots, not one launch per sequence.

6. **Never zero memory in the inference loop — it will be written anyway.** A buffer that
   a kernel fully overwrites must be allocated uninitialised (`alloc_uninit`), not with
   `Tensor::zeros`. `zeros` is a second full-width memory pass (a `memset`) on top of the
   allocation, on the exact bytes the kernel is about to stamp. The *only* buffers that
   may be zeroed are ones whose zero value is semantically read before being written:
   atomic accumulators (`atomicAdd`/`atomicMax` targets), scatter bases, and ragged
   padding.

The rest of this document is the code study that catalogues where each invariant is
violated today, across both phases, with the fix direction.

---

## Invariant 1 — no `to_dtype`; kernels emit the final type

The attention kernels emit **BF16** (`paged.rs:375` decode, `paged.rs:562` prefill), but
out-proj quantises to **int8** and needs **F32** input, so every attention result is cast
BF16→F32 before out-proj. Symmetrically, the projection norms emit **F32** but the kernels
want **BF16**, so q/kv are cast F32→BF16 on the way in. Both casts are full passes the
kernel could absorb.

| Phase | Site | Conversion | Fix |
|---|---|---|---|
| prefill | `wave.rs:1332`, `kernel_attention.rs:375` | kernel BF16 → `out.to_dtype(F32)` for out-proj | attention kernel writes F32 (or `to_dynamic` accepts BF16) |
| decode | `wave.rs:1087` | kernel BF16 → F32 | same |
| glue | `wave.rs:1383` | kernel BF16 → F32 | same |
| prefill | `kernel_attention.rs:447,449` | `q_bf`/`kv_bf` F32 → BF16 for the kernel | `rms_norm`/`rms_scale` emit BF16 |
| decode | `wave.rs:865,867` | `q_bf_all`/`kv_bf_all` F32 → BF16 | same |
| glue | `wave.rs:820,1360` | F32 → BF16 | same |
| decode (single) | `kernel_attention.rs:279,286,288` | F32/BF16 round-trips | engine single-session path, same treatment |
| prefill/pool | `compressor.rs:524,737,738` | `ape`/`norm_w`/`x` `to_dtype(F32)` | store `ape`/`norm_w` as F32 at load; delete per-call casts |

**No-ops to leave alone** (source is already F32 and it is a view, not a copy):
`kernel_attention.rs:436`, `compressor.rs:238`, `wave.rs:859`.

---

## Invariant 2 — no `contiguous` / `force_contiguous`

`force_contiguous` = `alloc_uninit` + `copy_strided_src` (an async D2D copy). It is not a
sync, but it is an allocation + full copy, and at config-8 the compressor issues thousands
of them.

| Phase | Site | What | Fix |
|---|---|---|---|
| **prefill (dominant drain)** | `compressor.rs:512,515,581,587,591,594` | **6× `force_contiguous` per seq** in `assemble_groups` (carried rows + prev group) | this is the bulk of the ~3205 ms `pprep:assemble` — ~4000 tiny alloc+copies. Replace the cat/copy assemble with **one fused pooling-input kernel** |
| prefill | `wave.rs:71` (`pool_prefill_across_seqs`) | `pooled.narrow().contiguous()` split-copy per seq | `append_batch` takes a `(tensor, row_offset, len)` view, no compacted copy |
| both | `gallery.rs` `gather_corpus` (`pos` + `sel` `.contiguous()`) | `index_select` results forced contiguous | gather kernel writes the corpus cache directly |
| decode | `paged.rs:242` (`CorpusCache::from_gathered`) | `comp_pos.contiguous()` | same gather-kernel fix |

---

## Invariant 3 — no unnecessary GPU→CPU transfers

| Phase | Site | Transfer | Verdict |
|---|---|---|---|
| MoE | `engine.rs:413` `indices.to_vec2` | routing readback | **SANCTIONED** — expert streaming schedules by id |
| both | `wave.rs:371` `token_ids` (`to_vec1`) | input token-id readback | **SANCTIONED** — MoE needs host ids, and it feeds the embed gather (also sanctioned, see below) |
| prefill | `kernel_attention.rs:621` `g.to_vec1::<u32>()` | per-token GID readback (out-of-regime recall) | violation (deep-prompt fallback) — keep on GPU |
| decode | `wave.rs:995-1046` `idx_flat` host build + `from_vec` | comp-idx assembly | build comp-idx on-device from the batched select's offsets |
| both | `gallery.rs` `gather_corpus` spilled arm `to_device(Cpu)` | warm-tier re-heat | **NEEDED** (tiering) — keep |

---

## Invariant 4 — run as much as possible on the GPU

| Phase | Site | Host compute | Fix |
|---|---|---|---|
| MoE | `engine.rs:411-444` | **counting-sort + `assignments` built on host** then re-uploaded | GPU bucketize (reuse `moe_bucketize`); the readback at 413 stays, the *sort* moves to the GPU |
| prefill | `wave.rs:1227-1267` (Host-select) | `union.sort/dedup` + `remap` HashMap + `idx_flat` host build + `from_vec` | extend `batched_causal_select_device` to the out-of-regime/HCA/SWA arms |
| both | `wave.rs:377` `embed_rows` | `from_vec(Cpu)` + `index_select(Cpu)` | **SANCTIONED** — this is an index + host→device transfer, not compute; the embed table stays CPU-resident to save VRAM |

---

## Invariant 5 — everything fully batched

**Decode** is ~90 % batched: projections (`wave.rs:859`), select
(`two_stage_select_batched`), gather (`gather_corpus_batched`), attention (one
`paged_latent_decode`), out-proj — all batched across sessions. Residual:

- `wave.rs:899-935` — per-session `kernel_attn_decode_capture` (stateful corpus push +
  query capture).
- `wave.rs:995-1046` — host comp-idx build (also Invariant 4).
- `wave.rs:1442-1447` — per-seq `lm_head`.

**Prefill** is the core gap — everything after the (now-batched) projection is per-seq:

- `wave.rs:1120` — pass-1 per-seq `assemble` (stateful; the `force_contiguous` storm).
- `wave.rs:1158` — pass-2 per-seq loop: **append, select, gather, attention kernel,
  writeback, out-proj — all single-seq.** `paged_latent_prefill_raw` is **single-slot**;
  decode's kernel is multi-slot. Batching this across all prompt slots is the single
  biggest win.
- `wave.rs:1247` — per-token remap loop.
- `wave.rs:1442-1447` — per-seq `lm_head` (shared with decode).

---

## Invariant 6 — never zero memory in the inference loop

`Tensor::zeros` allocates **and** memsets. For a buffer a kernel fully overwrites, the
memset is a wasted full-width memory pass on the bytes the kernel is about to stamp. The
fix is a public `Tensor::empty(shape, dtype, device)` wrapping the existing
`Device::alloc_uninit` (`device.rs:424`, today `pub(crate)`), swapped in at every
pure-output site. `alloc_uninit` is already what `force_contiguous` and every internal op
uses — the primitive exists; only a public constructor is missing.

### Pure kernel outputs — must become `Tensor::empty` (violations)

| Phase | Site | Buffer | Written by |
|---|---|---|---|
| decode | `paged.rs:375` | attention `out` BF16 `[num_slots, n_q_head, HEAD_DIM]` | `paged_latent_decode` |
| prefill | `paged.rs:562` | attention `out` BF16 `[total_q, n_q_head, HEAD_DIM]` | `paged_latent_prefill` |
| prefill | `paged.rs:208-210` | `nope_i8`/`nope_scale`/`rope_bf` | dequant/seal kernel |
| prefill | `paged.rs:619,620,624` | `comp_i8`/`comp_scale`/`comp_v8` (baked corpus) | pre-pass rope+quant kernel |
| decode | `wave.rs:1003-1006` | `out_nope`/`out_scale`/`out_rope`/`out_pos` | gather kernel |
| seal | `gallery.rs:249-251` | `nope_tmp`/`scale_tmp`/`rope_tmp` | append-seal kernel |
| gather | `gallery.rs:1704-1707,1766-1769` | batched corpus-gather outputs | gather kernel |

### Legitimately zeroed — the zero value is read before it is written (keep)

| Site | Buffer | Why zero is required |
|---|---|---|
| `paged.rs:625` | `comp_vmax` | `atomicMax` accumulator init (comment already notes this) |
| `paged.rs:99-100` | `acc`/`ml` workspace | accumulators (also one-time, not per-loop) |
| `gallery.rs:1033,1133,1200,1285,1286` | histogram / counts | `atomicAdd` targets |
| `gallery.rs:1477,1527` | `pad` | ragged-batch padding must read as zero |
| `attention.rs:232`, `moe.rs:148` | `scatter_add` bases | reference path; base read by scatter |

### One-time (outside the loop — either is fine, low priority)

`kernel_attention.rs:130-135` (empty_* sentinels), `gallery.rs:95-100` (arena capacity,
grown occasionally), `paged.rs:121` (rope table). These allocate once or on grow, not per
token; leave them unless a sweep is cheap.

---

## Priority (the through-line)

The wall is flat at ~558 t/s because prefill attention is per-seq while decode's is
batched, and the compressor assemble is a `force_contiguous` storm. In order of impact:

1. **Fused assemble/pool-input kernel** — kills the ~3.2 s `pprep:assemble`
   (Invariants 2 + 6, prefill).
2. **Multi-slot batched prefill attention kernel + on-device comp-idx** — kills the
   per-seq pass-2 loop (Invariant 5, prefill).
3. **GPU MoE bucketize** — kills the host counting-sort (Invariant 4).
4. **Kernels emit final type** (drop q/kv and out casts) + **`Tensor::empty` sweep** for
   pure outputs (Invariants 1 + 6) — second-order (100s of ms) but broad.
5. **Batched `lm_head`** (Invariant 5). (Embed stays a CPU gather — sanctioned.)

Invariants 1–4 remove per-step overhead; 5 removes the per-seq serialisation that caps
prefill; 6 removes a full memset from every buffer the kernels already fill. Together they
are what turns the current 558 t/s into the compute-bound target.
