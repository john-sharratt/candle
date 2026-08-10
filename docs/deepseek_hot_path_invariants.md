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
