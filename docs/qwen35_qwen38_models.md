# Qwen3.5 / Qwen3.8 bring-up — model research and integration design

Branch: `qwen35-qwen38`. This document records (1) what the engine runs today
and the exact contracts a new model must satisfy, (2) what the Qwen3.5 and
Qwen3.8 families are, (3) which variants fit our two compute envelopes, and
(4) the integration design and phasing for bringing them up on the
unbounded-context engine.

---

## 1. What we run today (in-repo)

### 1.1 Qwen3 dense — `quantized_qwen3.rs` (production) and `qwen3.rs` (reference)

The production path is GGUF-loaded, rides `BatchedModelCore` +
`BatchedAttentionLayer`, and is the dense gate
(`quantized_qwen3::tests::test_parallel_batched_forwarding`, Qwen3-8B-Q6_K).

- GQA full attention, `head_dim 128` explicit in metadata
  (`qwen3.attention.key_length`), per-head **q_norm/k_norm** between the QKV
  projection and RoPE — the defining Qwen3 attention difference vs Qwen2
  (`quantized_qwen3.rs:562-574`).
- RoPE theta default 1e6, non-interleaved, tables grown on demand
  (`ROPE_EXTEND_CHUNK = 1024`); scaling is a flat inv_freq divide inferred as
  `context_length / 32768` when the GGUF advertises a longer context with no
  explicit factor (`quantized_qwen3.rs:47-70`) — not YaRN.
- `hidden_size` is read from the checkpoint, never derived: Qwen3 ships
  `hidden 2048` against `32×128 = 4096`, and `wave_shapes()` reads
  `down_proj.weight_dims()` for the same reason (`quantized_qwen3.rs:683-691`).
- Int8 path: q/k/v deliberately **not** concatenated — three KO twins fused at
  launch by `qkv_segmented_matmul`; gate+up concatenated at load on CUDA;
  float weights (router, lm_head) quantized to Q8_0 so they get a Q8_KO twin
  (`quantized_matmul.rs:61-76`).
- Per-model KV threshold factors: `QWEN3_8B_KV_FACTORS`
  (`sampled_selection/params.rs:422-427`).
- The safetensors `qwen3.rs`/`qwen3_moe.rs` are reference-only — no batched
  engine, no paged KV, no int8; `qwen3.rs` hard-refuses sliding window. Do not
  model a new production integration on them.

### 1.2 Qwen3 MoE — `quantized_qwen3_moe.rs` (production; the MoE gate)

Qwen3-30B-A3B: 128 experts, top-8, `norm_topk_prob`, per-expert intermediate
from `{p}.expert_feed_forward_length`. Architecture prefix is detected by
probing (`qwen3moe` → `qwen2moe` → `qwen3` → `llama`), and MoE-vs-dense is
decided per layer by **tensor presence** (`ffn_gate_inp.weight` /
`ffn_gate_exps.weight`), so mixed stacks load correctly
(`quantized_qwen3_moe.rs:2092-2099`).

- Experts are never loaded as tensors: 3-D merged `ffn_*_exps.weight` is
  sliced into `MmapExpertRef` byte ranges served by the three-tier
  `expert_lre::ExpertCache` (VRAM slots / pinned warm / mmap cold pack).
- Two dispatch paths: GPU-native (`moe_route` → `moe_bucketize` → fused
  gather → grouped q8a128 GEMMs → deterministic scatter) and the production
  host-orchestrated path (async index readback + O(A+E) bucketing +
  `submit_moe_work`). The GPU-native dispatch table refuses to build above
  **128 experts** (`expert_lre/gpu_dispatch.rs:84-87`).
- Load order is load-bearing: dense weights → measure → reserve span →
  expert cache; the cache registers as the **ground broker** for the elastic
  KV/weight boundary.

### 1.3 DeepSeek — `deepseek4/` (production) and `deepseek2.rs` (reference)

`deepseek4` (DeepSeek-V4-Flash) is the most instructive precedent for this
bring-up because it already solves, in production, most of what Qwen3.5/3.8
need:

- **Non-uniform per-layer attention kinds** driven by config
  (`compress_ratios[layer]` → SWA / CSA / HCA `LayerKind`,
  `deepseek4/config.rs:12-38`) — the exact shape of a hybrid layer schedule.
- **A dedicated paged kernel family** (`candle-kernels/src/paged-latent/`)
  for an attention geometry the generic kernels don't serve (single-latent
  512), sharing the slot/arena record layout with `paged-decode` but
  templated at a different band count. Precedent for adding a family rather
  than force-fitting the generic kernels.
- **256 routed experts + 1 shared expert, host-orchestrated dispatch** with a
  counting sort by expert id feeding the same grouped-GEMM contract
  (`deepseek4/engine.rs:470-563`) — directly reusable for Qwen3.5's 256-expert
  MoE, which exceeds the 128-expert GPU-native table.
- **One shared `ExpertCache` across all MoE layers** (vs qwen3-moe's
  per-layer caches) — the right default for 256-expert models.
- Host-resident token embedding, multi-split GGUF loader
  (`NAME-00001-of-000NN.gguf`), MXFP4 expert slicing, speculative drafter
  (`dspark`) — all reusable pieces.
- `deepseek2.rs` is upstream Candle's V2 MLA reference (standalone, naive KV,
  zero tests) — useful only as an MLA math reference.

### 1.4 The contracts a new model must satisfy

From `BatchedModelCore` / `BatchedAttentionLayer` (details in
`batched_model.rs:194-368`, `batched_layer.rs:253-351`):

- Implement `wave_shapes` (from the checkpoint), `maybe_change_dtype`
  (must reach **every** RmsNorm — `weight_for` refuses rather than converts),
  `attention_norm`/`project_qkv`/`ffn_norm`/`ffn_forward` with wave-lifetime
  (`'w`) discipline, `rope_interleaved`, `prune`.
- `ManagedBatchedModel` comes free via the blanket impl over
  `BatchedInference<M>`; deepseek4 shows the alternative of implementing it
  directly when the layer loop itself is non-standard — **which is the right
  choice for a hybrid stack** (see §4.3).

Hard numeric constraints (all verified in-code by the research pass):

| Constraint | Value | Where |
|---|---|---|
| Paged **decode** head_dim | 64, 96 (fp16 only), **128**, **256** | `paged_decode_api_*.cu:35-41` |
| Paged **prefill** head_dim | **64, 128 only** (256's staging slabs exceed the 25.6 KB smem budget) | `paged_prefill_int8_fp16.cu:28-43`, `prefill_utils.rs:1189-1191` |
| q8a1024 decode context (B2) | 128 only | `paged_decode_api_fp16.cu:60` |
| Paged glue (reprojection) | 128 only | `paged_glue_api_fp16.cu:48-50` |
| GQA fast path | `n_head/n_kv_head ∈ 1..=8` (BMMA only at head_dim 128; else stripe) | `int8_decode_kernel.cuh:1689-1774` |
| Int8 weights | every projection `N % 32 == 0`, `K % 128 == 0` | `quantized/cuda.rs` (multiple) |
| Fused RMSNorm→q8a128 | `hidden % 128 == 0` and `hidden ≤ 8192` | `cuda.rs:5021-5056` |
| MoE gather | `hidden % 1024 == 0` | `cuda.rs:6158-6164` |
| `moe_route` | `n_experts ≤ 256`, `k ≤ 16` | `cuda.rs:6062-6070` |
| GPU-native MoE dispatch | `n_experts ≤ 128` (else host path) | `gpu_dispatch.rs:84-87` |
| KV chunk | `CHUNK_SIZE = 32`, shared Rust/CUDA | `candle-kernels/src/lib.rs:22` |

---

## 2. The new families (online research, 2026-08)

Both families are **hybrid Gated DeltaNet + full attention** generations in
the Qwen3-Next lineage: blocks of **3 Gated DeltaNet (linear-attention,
recurrent-state) layers + 1 full-attention layer**, each layer followed by a
sparse MoE block on the MoE variants. Apache 2.0. GGUFs exist on HF (unsloth
and others); llama.cpp support has landed (recent builds required; the
DeltaNet GPU kernels key off a `delta_net_gpu_compat` metadata flag).

### 2.1 Qwen3.5 (February 2026)

Dense lineup 0.8B–27B plus three MoE variants:

| Variant | Total / active | Layers | Attention layers | MoE | Context |
|---|---|---|---|---|---|
| **35B-A3B** | 35B / ~3B | 40 = 10×(3 DN + 1 attn) | 16 Q / 2 KV heads @ **head_dim 256** (hpg 8) | 256 experts, top-8 + 1 shared, expert FFN 512, hidden 2048 | 262K |
| **122B-A10B** | 122B / ~10B | 48, 3:1 hybrid | (same family pattern) | 256 experts, top-8 + 1 shared | 262K (1M YaRN) |
| 397B-A17B | 397B / 17B | 60, hybrid | — | 512 experts | 262K |

DeltaNet geometry (35B): 32 linear-attention V heads / 16 QK heads @ 128.
A 9B variant exists with an **MTP** (multi-token prediction) release
(`unsloth/Qwen3.5-9B-MTP-GGUF`) — a drafter head for speculative decode.

### 2.2 Qwen3.8 (August 12/14, 2026 — days old)

| Variant | Total / active | Layers | Attention layers | Notes |
|---|---|---|---|---|
| **27B** | 27B dense | 64 = 16×(3 DN + 1 attn) | 24 Q / 4 KV @ head_dim 256 (hpg 6); DN 48 V / 16 QK @ 128; hidden 5120 | MTP; 262K ctx |
| 2.4T-A95B | 2.4T / 95B | — | — | out of scope |

A Qwen3.8-9B also has community GGUFs. (A Qwen3.6 point-release family of the
same architecture exists between the two; noted for completeness — same
integration surface.)

### 2.3 What the hybrid means for this engine

- Only **1 in 4 layers holds a KV cache**. KV memory and the paged-attention
  surface shrink ~4×; the provenance scan (BDP) has ¼ the layers to index.
- The other ¾ of layers carry a **fixed-size recurrent state** per session
  (delta-rule state `[n_v_heads, d_k, d_v]` + a short causal-conv tail):
  ~1 MB/layer/session at BF16 for the 35B — ~30 MB/session total, **constant
  in context depth**. This is a new state class our engine has never stored.
- Attention layers move to **head_dim 256** with 2–4 KV heads — supported by
  paged decode today, *not* by paged prefill / B2 / glue (128-only).

---

## 3. Candidate selection vs our compute

Envelopes: dev = RTX 4090 Mobile 16 GB + 32 GB RAM; production (~mid-2026) =
2× RTX 5090 32 GB + 512 GB DDR5 + NVMe cold tier.

| Candidate | Weights (Q4_K_M-class) | Fit | Verdict |
|---|---|---|---|
| **Qwen3.5-35B-A3B** | ~19–21 GB | Dev: yes, via expert tiering (3B active; same posture as today's 30B-A3B at 18 GB). Prod: trivial. | **Primary dev + first bring-up target.** Direct successor of our current gate model; same test harness applies. |
| **Qwen3.5-122B-A10B** | ~65–70 GB | Prod: exactly what the three-tier expert cache was built for (64 GB VRAM + warm RAM). Dev: build-only. | **Production Zen Code target — build now, test on the workstation.** A derivative of the 35B implementation (same arch, larger dims); its gate test is authored alongside it and runs when the hardware lands. |
| Qwen3.5-9B (+MTP) / Qwen3.8-9B | ~5–6 GB | Everywhere. | Ablation + kernel-validation model; also the MTP/speculative testbed. |
| **Qwen3.8-27B** | ~15–16 GB | Dev: build-only (dense — no expert relief on 16 GB). Prod: yes. | **Build now, test on the workstation.** A derivative of the same hybrid implementation (dense = the MoE block replaced by a plain MLP, hidden 5120, hpg 6); gate authored now, run later. |
| Qwen3.5-397B-A17B | ~210 GB | Prod: RAM+NVMe stretch; interactive latency doubtful. | Deferred; revisit after 122B works. |
| Qwen3.8-2.4T-A95B | — | No. | Out. |

**Build-only targets.** Qwen3.5-122B-A10B and Qwen3.8-27B are implemented in
the same bring-up (they are dimensional/dense derivatives of the 35B hybrid
code path — no new subsystems), but they cannot be *validated* on the 16 GB
dev card. The rule for them: the loader, the model wiring, and the gate tests
are all written and kept compiling; the gates run on the production
workstation. Nothing about them is deferred except the GPU run itself.

Constraint check for the two headline picks against §1.4 (all pass):
hidden 2048 (35B) / 5120 (27B): `%128` ✓, `%1024` ✓, `≤8192` ✓. Projections:
q `[4096/6144, hidden]`, kv `[512/1024, hidden]`, expert gate/up
`[512, 2048]`, down `[2048, 512]` — all `N%32`, `K%128` ✓. `moe_route` caps:
256 experts = the exact warp bound ✓, top-8 ≤ 16 ✓. GQA hpg 8 / 6 ∈ 1..=8 ✓
(stripe path at head_dim 256; BMMA is 128-only). GPU-native MoE dispatch is
out (>128 experts) — host-orchestrated path per deepseek4 precedent.

---

## 4. Integration design

### 4.1 Gated DeltaNet layers (the new subsystem)

Per DeltaNet layer: fused input projection (q/k/v/gate + per-head decay `a`
and mixing `β`), short causal conv over a few-token tail, L2-normed q/k,
gated delta-rule recurrence over the state `S`, normed + gated output
projection.

- **Kernels** (new family `candle-kernels/src/delta-net/`, following the
  `paged-latent` precedent of a dedicated family):
  - *Prefill*: chunked parallel delta rule (chunk 64), BF16 in / FP32 state
    accumulate. llama.cpp's implementation is the reference for tensor
    semantics; our version targets the wave engine's batched rows.
  - *Decode*: fused single-token step (conv tail shift + state update + out)
    per session row — one launch per layer per wave, batched over rows like
    the paged decode kernel.
- **State store** (new `candle-nn/src/kv_cache/recurrent/` or sibling
  module): per-session, per-layer `[n_v_heads, d_k, d_v]` FP32/BF16 state +
  conv tail. Constant-size → lives in a small dedicated arena class, not the
  chunked KV system. Wave rules apply unchanged: states are pre-allocated
  before the wave. **Wave failure atomicity** follows KV's discipline, which
  is *never destroying the entering value* — KV gets that free by being
  append-only, so its rollback is a truncation. The recurrent state has no
  such structure (`s` is a fixed-size accumulator every token rewrites), so
  each slot holds **two `s` buffers** and a wave reads one and writes the
  other: `begin_wave` costs nothing, `commit_wave` swaps the pair for every
  layer the sweep advanced, rollback is doing nothing. Only `s` ping-pongs —
  the conv tail advances in place, so a commit swaps `s` alone.

  *This replaces "snapshot the entering state per row … and restore on
  `rollback_wave_kv`". Read literally, that produced a device copy of every
  layer's state on every wave — ~2 MB and two launches per layer, ~60 MB per
  wave — insuring against a rollback that almost never fires. The wording had
  named KV's words rather than its mechanism.*
- **Fork/branch semantics (substrate)**: KV is append-only, so forks share
  prefixes for free; recurrent state is destructive. Every **sealed turn
  boundary** snapshots the state alongside the sealed KV — the full record
  design (single live tail per conversation, supersede-and-tombstone, resume)
  is §5. `truncate_sequence` must restore a state consistent with the
  truncation offset — same shape as deepseek4's compressor-state rollback
  (`wave.rs:799-807`); with a single-tail snapshot that means recomputing the
  state over the surviving prefix when the tail is newer than the cut (§5.5).
- **Provenance/O(1)-error story**: provenance-selected attention applies to
  the attention layers only (10/40). DeltaNet layers summarize lossily by
  construction; the paper's error accounting gains a section, and the BDP
  scan indexes ¼ the layers (cheaper), with Q capture on attention layers
  only.

### 4.2 Attention layers at head_dim 256

- **Paged decode**: already dispatched (64/128/256); hpg 8/6 ride the stripe
  path. Band layout: `N_PALETTE = 4` × 32-dim bands covers 128; 256 needs an
  8-band GQA variant — the per-backing `n_palette()` mechanism and the
  16-band latent precedent (`arena_table.rs:294`, `backing.rs:211-220`) show
  exactly where it plugs in. New size classes for 256-dim K/V chunk slots
  (`size_class.rs`).
- **Paged prefill**: 128-only today for a real reason (smem budget). Plan:
  a 256 instantiation with halved block occupancy (2 blocks/SM) and split-D
  staging; until it lands, attention-layer prefill runs the float fallback —
  acceptable initially because only ¼ of layers pay it.
- **B2 (q8a1024 context output) and glue**: 128-only; both take their
  generic fallbacks at 256 to start. Glue matters less here (¼ the layers);
  revisit after profiling.
- KV compression: C0–C10 candidate tables are head-dim-agnostic (32-element
  blocks), but **thresholds are model-specific** — full `PRODUCTION_*`
  re-derivation per new model (per CLAUDE.md), plus new `*_KV_FACTORS`
  entries. Pin the GGUF revision in every new gate test — the 2026-08-16
  C10 incident (upstream re-upload silently invalidating a tuning) must not
  be repeatable.

### 4.3 Model implementation shape

A hybrid stack does not fit `BatchedModelCore`'s uniform-layer loop —
**confirmed by reading the trait rather than assumed**. `BatchedModelCore`
has a single `type Layer`, and `BatchedAttentionLayer`'s contract is
"project Q/K/V, and the generic code attends them against a KV cache"
(`project_qkv` → `forward_attn_batched`). A DeltaNet layer has neither a
Q/K/V triple nor a cache: it mixes by a recurrence over a per-sequence
state matrix. No implementation of that trait can express it.

So implement `ManagedBatchedModel` directly (`deepseek4/wave.rs:665`
precedent) with a per-layer `LayerKind` (`DeltaNet | Attention`) derived
from GGUF metadata.

**But not by restating the per-layer machinery.** The uniform loop's body
is a *free function* over the trait —
`forward_layer_batched_mixed(layer, groups, x, …)` — not a method on the
loop, so the hybrid sweep calls it per attention layer and the paged
attention path is shared, not duplicated.

**A DeltaNet layer implements no engine trait at all.** It cannot implement
`BatchedAttentionLayer` (no Q/K/V, no cache), and its FFN — an ordinary
SwiGLU over the same combined buffer — is driven by ten lines local to
`quantized_delta_net.rs` instead.

An earlier revision of this section proposed splitting
`BatchedAttentionLayer` into a `BatchedFfnLayer` supertrait so the
recurrent layer could reuse the shared FFN driver. That was **built and
then reverted (2026-08-21)**: it restructured a trait across four
production models, and turned a defaulted `int8mode` into a required
method, to avoid duplicating ten lines. The blast radius was not worth the
saving. The rule it leaves behind is worth keeping: *a new model kind pays
for itself locally unless sharing genuinely removes duplication rather
than merely relocating it.* The four models now differ from mainline by
exactly one line each — the gate field below.

The gated FFN both layer kinds need is now shared rather than copied a
fifth time. `down(silu(gate(x)) · up(x))` appeared in four models
(`quantized_llama`, `quantized_qwen2`, `quantized_qwen3`,
`quantized_qwen3_moe`) and **twice inside `quantized_qwen3` alone** — once
in the plain loader and again in the int8 one, differing only by the
numeric mode. It is now `models::quantized_mlp::QuantizedMlp`, which owns
the three details that are easy to get subtly wrong and were being
maintained in parallel: gate/up row-fusion on CUDA for quantized dtypes
only; casting the fused output in place *before* splitting it (the owned
contiguous buffer casts without allocating, two aliasing narrows do not);
and running silu/mul/down in `out_dtype` because MLP intermediates can
exceed F16's range. `quantized_qwen3` and the Qwen3.5 path both use it;
the remaining three copies are pre-existing and are candidates for the
same treatment, not made worse here.

Beyond that, the hybrid shares:

- the wave engine (phases, transient tier, admit/rollback) unchanged;
- `expert_lre` with **one shared cache across layers** and host-orchestrated
  dispatch (deepseek4's counting-sort contract) for the 256-expert MoE;
- the chunked KV backing for attention layers only (`n_kv_head 2/4`,
  `head_dim 256`), plus the new recurrent state store for DeltaNet layers;
- the shared-expert add and q8a128 activation path (hidden sizes all comply).

### 4.4 GGUF loading

New arch prefix (llama.cpp `qwen35`-lineage; exact string and tensor names to
be read off the actual GGUFs in Phase 0). Expected deltas vs `qwen3moe`:
per-layer DeltaNet tensor set (fused qkvz/ba projections, conv weight, decay
`A`/`dt` parameters, output-norm gate) alongside the familiar attention set on
every 4th layer; layer-schedule metadata (full-attention interval or
per-layer type array); `delta_net_gpu_compat` flag; MTP head tensors on MTP
releases (loadable-but-ignored until the drafter phase). Multi-split GGUF
support already exists (deepseek4 loader). Reuse arch-prefix probing and
tensor-presence layer classification from `quantized_qwen3_moe`.

### 4.4a Two contract gaps a gated, partially-rotary model exposes

The generic attention machinery assumes two things this family breaks. Both
are fixed in the shared contract rather than worked around per model,
because the whole Qwen3-Next/Qwen3.5 lineage shares them.

**The output gate had nowhere to travel.** `QkvProjection` carried only
q/k/v, but a gated layer's gate is produced *with* q/k/v (same `wq`, same
fused activation) and consumed at the far end of the attention block. It is
now a `gate: Option<LiveTensor>` on that struct, applied as
`sigmoid(gate) ⊙ context` immediately before the output projection —
matching `ggml_mul(cur, ggml_sigmoid(gate))`. A `None` gate is every
classic layer and costs nothing. The q8a1024 decode context refuses a gate
explicitly (an elementwise gate cannot land in a packed U8 buffer without
undoing the fused path); that branch is unreachable today because it
requires head_dim 128 while this lineage is 256, so the bail exists to stop
a future widening silently dropping the gate.

**Partial rotary against full-rotary kernels.** The paged kernels rotate
every head dim, pairing `d` with `d + head_dim/2` from a `[pos, head_dim]`
table. This family rotates 64 of 256, pairing `j` with `j + 32`. Two
things disagree — *which* dims rotate and *which* dims pair — and a
`cos = 1, sin = 0` table only fixes the first.

`models/rotary_layout.rs` fixes the pairing by permuting the head dims so the
kernel's own `(d, d + head_dim/2)` pairing lands on the dims this model
wants paired, with identity table entries on every pass-through pair. This
is exact because attention only ever contracts Q against K over the head
dim, so any permutation applied to *both* cancels; V is deliberately not
permuted, since its dims flow through to the context and out through the
output projection.

It is applied to the projection *outputs*, not folded into the weights:
rows are independent block sequences so a weight-side permutation is
meaningful on a quantized tensor, but candle has no row-gather for
`QTensor` (only `concat_rows_cuda`), and dequantizing to permute would
cost either a requantization's worth of quality or a resident F32 copy of
every Q/K projection. The gather index is built once at load, not per
forward — rebuilding it would be a host→device transfer on the hot path.

The standing optimisation is to template the paged kernels on `n_rot` and
delete the module: it exists only because those kernels hard-code the
pairing at `head_dim/2`.

### 4.4b The MoE path is Qwen3-MoE's, plus one expert

The 35B's routed MoE needed **nothing** built for it. Checked against the
code rather than assumed:

* `ExpertCache` takes `experts_per_layer` as a runtime parameter — the
  `128` in its docs is an example, not a bound;
* the only `> 128` test is in `gpu_dispatch::build`, and it returns `None`
  → *host path*, which is the documented behaviour for an oversized id
  space, not a failure;
* `moe_bucketize.cu` declares `MAX_EXPERTS 256`, so 256 is exactly covered;
* `ExpertCache::new(ExpertCacheSetup { mmap, host_refs, zone, … })` is a
  stable API over mmap references, and the expert tensor names are the same
  frozen `ffn_{gate,up,down}_exps` schema.

So `Qwen35MoeBlock` is `SparseMoeBlock` reused verbatim plus the piece
Qwen3-MoE genuinely lacks: a **shared expert** — an ordinary SwiGLU every
token passes through, scaled by a per-token `sigmoid(w·x)` and summed with
the routed output. There is no `shexp` tensor anywhere in Qwen3-MoE.

Reuse is by **visibility only** — `SparseMoeBlock`, its fields and
`forward_dynamic` became `pub(crate)`. No fields moved, no trait reshaped;
`quantized_qwen3_moe.rs` is otherwise untouched. That is the line §4.3's
reverted trait split established: sharing that removes duplication is
worth it, sharing that merely relocates it is not, and here the
alternative was duplicating ~300 lines of GPU dispatch.

**Ordering constraint.** `SparseMoeBlock::forward_dynamic` *consumes* its
activation — the expert gather is the activation's last reader, so it is
moved rather than borrowed. The shared expert and its gate therefore run
first, off a borrow, and the routed call takes ownership last.

**Load order.** The expert cache is a *parameter* to the model loader, not
built inside it: it is sized from a live measurement taken once the dense
weights are resident (§4 of `elastic_vram_partition.md`), so the sequence
is dense weights → measure the span, carve the zone, place the boundary →
fill the cache. `qwen35/expert_loader.rs` is that half. Experts are keyed
`(moe_layer_idx, expert)` where the layer index counts MoE layers among
themselves, so a stack mixing dense and MoE layers still indexes densely.

### 4.5 MTP / speculative decode (optional, later)

Qwen3.5-9B-MTP and Qwen3.8 ship MTP heads. The `ManagedBatchedModel`
speculative hooks are already in place with safe defaults, and deepseek4's
`dspark` drafter is the in-repo template. Deferred until after the main
bring-up; the 9B-MTP model is the testbed.

---

## 5. Turn-seal snapshots of the Gated DeltaNet state (single tail per conversation)

Every model in this bring-up is a Gated DeltaNet design, so resuming a
conversation requires more than the sealed KV: the recurrent state (delta-rule
matrix `S` per DeltaNet layer + conv tail) as it stood **at the end of the
last sealed turn**. The substrate gains one record kind for this. The design
below is grounded in a full read of `candle-conversation/src/persistence/`
(citations inline) and reuses the log's existing supersede machinery rather
than inventing any.

### 5.1 Semantics

- On every **turn seal**, one `Snapshot` record is appended for the
  conversation carrying the post-turn recurrent state of all DeltaNet layers.
- The newest snapshot **supersedes** all previous ones for that conversation:
  exactly one snapshot is live per conversation at all times — a single tail.
  Superseded copies become dead bytes and are reclaimed by normal segment
  maintenance/compaction; no separate tombstone record is written for
  supersede (the log's last-writer-wins accounting *is* the tombstone).
- Resume restores the tail snapshot; the conversation continues from the last
  sealed turn with bit-faithful recurrent state.

### 5.2 The record

- **Type**: `RecordType::Snapshot = 20` (`Unknown` moves to 21); wire-tag and
  byte-exact codec tests pinned like `TurnCoupling`'s
  (`record.rs:1022-1025`, `:1001-1008`).
- **Key**: a synthetic per-conversation stream id in the header's
  `stream_id` field — `snapshot_stream_id(timeline)` derived purely, sibling
  to `turn_stream_id(timeline, idx)` (`streams.rs:149-154`). Header-keying is
  what makes the two existing supersede mechanisms apply *mechanically*:
  - `RecordAccounting`'s `(type, stream_id, 0)` arm (`accounting.rs:50-54`)
    gives O(1) dead-byte credit the moment a newer snapshot lands;
  - `is_tracked_metadata` (`mod.rs:170-178`) + `metadata_locs` makes segment
    liveness count only the tail as live and makes `need_resident_reemit`
    (`maintenance.rs:515-521`) refuse to drop a segment holding it — the
    single-tail invariant, enforced by the accounting layer that already
    enforces it for `StreamDecl`/`WideQSig`/`Commit`.
- **Payload** (binary, `ByteWriter` LE like `ChunkPayload`,
  `record.rs:619-756`): `{ timeline_id, turn_index, layer schedule hash,
  n_deltanet_layers, per-layer [state dtype, dims, bytes], conv tails }`.
  `turn_index` is load-bearing: it binds the snapshot to the turn whose seal
  produced it (see §5.5). ~30 MB/conversation for the 35B at BF16.
- **Checksum**: GPU Fletcher-32 golden taken on-device before the DtoH copy,
  exactly as `Chunk` does (`gather_chunks_with_goldens`, `transfer.rs:82`;
  `verify_record_crc` skip + `recompute_*_golden` sibling,
  `record.rs:374-396`).

### 5.3 Write path

- **Producer hook**: the turn-seal site, `SealAction::Turn`
  (`scheduler/mod.rs:6309-6428`) — the same place that enqueues the turn's
  `WideQSig` and `Tokens` and then fires `persist_trigger`. The state is
  device-resident, so the gather runs where the CUDA copy stream and pinned
  scratch already live: the persistence thread's warm→cold phase
  (`thread.rs:1085-1121`), via a "snapshot pending" queue mirroring
  `snapshot_pending_cold`. One `gather`-style DtoH (it is a single contiguous
  region per layer, far simpler than a chunk grid), then enqueue.
- **Writer job**: `WriteJob::Snapshot` on the off-thread substrate writer
  (`writer.rs:51-82`), marked **`is_bulk = true`** so a 30 MB payload is
  gated by the byte cap alongside `KvCold`, never the event cap
  (`writer.rs:100-102`).
- **Ordering**: the snapshot is enqueued after the turn's decl/tokens/chunks
  in the same seal, and its payload names `(timeline, turn_index)`. Reload
  therefore never trusts a snapshot newer than the last recovered turn — if
  `snapshot.turn_index` exceeds the recovered tail (torn shutdown between the
  two appends), the snapshot is discarded and §5.5 recompute applies.
- The in-RAM mirror is a **`RecordLoc` only** — the `Tokens` shape
  (`apply_tokens_loc`, `substrate.rs:3110-3112`), never eager bytes: the
  payload stays on disk until restore. `Snapshot` joins the
  `payload_needed = false` skip list (`recovery.rs:64-74`) so boot never
  reads snapshot payloads.

### 5.4 Maintenance, compaction, deletion

- Compaction and incremental maintenance treat `Snapshot` as a **`Raw`**
  item — relocated verbatim from `(segment, offset, size)` like `Tokens`
  and `Chunk` (`compaction.rs:252-269`, `maintenance.rs:598-634`,
  relocation worklist `maintenance.rs:790-803` with the supersession guard
  in `MaintenanceResult::apply_to_substrate`). Only the tail is ever in the
  live set, so at most one snapshot per conversation survives any rewrite.
- **Timeline tombstone** kills the conversation's snapshot with everything
  else (`compaction.rs:201-203`); a **turn tombstone** that removes the tail
  turn leaves the snapshot pointing at a dead turn — restore detects the
  mismatch via `turn_index` and falls back to recompute. A snapshot of a
  distilled timeline is dropped (unresumable by construction).
- Inspector: `substrate_inspect` gains the new type in its histogram
  (`ALL_TYPES` is already stale — it omits `TurnCoupling`; fix both), the
  payload-skip lists, and a `snapshots` view proving "one live tail per
  conversation" against a real store (`substrate_inspect.rs:4399`, `:5091-5160`,
  `:173-207`).

### 5.5 Resume, forks, truncation

- **Reload** (metadata-only): the walker arm stores the `RecordLoc`; turn
  reconstruction (`resolver.rs:1990-2007`) resolves the tail per timeline and
  keeps it cold, consistent with the cold-only restart contract.
- **Materialization** happens on demand at `elevate_to_hot` / first
  `submit_turn` after `fork_resuming` (`elevate.rs:107`,
  `conversation.rs:2483-2487`): read the payload
  (`read_tokens`-style, `mod.rs:1064`), HtoD scatter into the recurrent
  state arena, verify the golden.
- **Fork from the latest turn** (the normal resume): O(1) — restore the tail.
- **Fork from an earlier turn / truncation below the tail**: the intermediate
  snapshots were superseded, so the state is **recomputed** by running the
  DeltaNet layers over the surviving prefix (a prefill-shaped pass with no
  attention-layer KV writes). This is the deliberate cost of single-tail:
  resume-latest is the hot path and stays O(1); rewinds pay a bounded
  recompute. It is also the only correct answer for forks from *reprojected*
  timelines, whose KV no longer matches the original stream (risk #1 in §8).
- Old readers skip the record cleanly (`walker.rs:139-140`); a missing
  snapshot (old writer, discarded torn tail, pre-DeltaNet conversations)
  degrades to recompute — the restore path must tolerate `None` exactly as
  `recover_turn_cold_refs` tolerates a chunk-less turn.

### 5.6 Insertion checklist (compile-error-driven)

`encode_record`, `RecordAccounting::record`, `Manifest::ingest`, and
`apply_walker_entry` are exhaustive matches — adding the variant produces
compile errors at precisely the sites that must change. The full set:
enum + `from_tag` + codec/pin tests (`record.rs`); accounting arm
(`accounting.rs:50-54`); `is_tracked_metadata` (`mod.rs:170-178`); manifest
no-op arm (`manifest.rs:163-177`); walker arm + `RecordLoc` storage
(`substrate.rs:3444`, `:3110`); compaction Raw item (`compaction.rs`);
maintenance relocation (`maintenance.rs`); recovery payload skip
(`recovery.rs:64-74`); `WriteJob` + `is_bulk` + `process_one`
(`writer.rs`); `SubstratePersistence::write_snapshot` (`mod.rs:627`
pattern); `Conversation::enqueue_snapshot` (`resolver.rs:2675` pattern);
seal-site producer (`scheduler/mod.rs:6390-6408`); persistence-thread gather
(`thread.rs:1085-1121`); reload resolve (`resolver.rs:1990-2007`); device
restore (`transfer.rs:138` pattern); tombstone gates; inspector histogram +
skip lists + `snapshots` view; doc updates (`docs/segmented_substrate_log.md`,
`docs/kv_tier_migration.md`).

---

## 6. Iterating gate tests (one per model)

Each new model gets the same iterating gate the current models have — an
`#[ignore]`d `test_parallel_batched_forwarding` with a config ladder
(unbatched baseline, F16/BF16 ×1/×N, Q8_0, Q4_0, C0…C10 StoryRewrite ladder),
per-session fixture validation, the performance table, and expert-pipeline
stats — driven by the same command shape:

```
cargo test --release --features cuda,verbose --lib --package candle-transformers \
  quantized_qwen35_moe::tests::test_parallel_batched_forwarding      -- --ignored --nocapture   # 35B-A3B (dev + prod)
cargo test --release --features cuda,verbose --lib --package candle-transformers \
  quantized_qwen35_moe::tests::test_parallel_batched_forwarding_122b -- --ignored --nocapture   # 122B-A10B (prod only)
cargo test --release --features cuda,verbose --lib --package candle-transformers \
  quantized_qwen38::tests::test_parallel_batched_forwarding          -- --ignored --nocapture   # Qwen3.8-27B (prod only)
```

Rules, carrying the lessons of the existing gates:

- **Pinned GGUF revisions.** Every gate passes an explicit revision hash to
  `hf_hub::Repo::with_revision` — never `"main"`. The 2026-08-16 C10 incident
  (upstream re-upload silently invalidating a tuned threshold row) is the
  standing reason.
- **Fixture-derived expectations** (name-substituted story prompt), so any
  config subset can run standalone — this is what made 16-second
  single-config iteration possible during the C10 walk, and it is preserved
  deliberately.
- **Prod-only gates still build everywhere.** The 122B and 27B tests compile
  in every workspace build; on hardware below their envelope they fail fast
  at model-load with an explicit capacity message rather than an OOM
  backtrace. They are run — and their compression ladders tuned — on the
  production workstation.
- **DeltaNet coverage inside the gate**: the C-ladder configs exercise KV
  compression on the ¼ attention layers; the gates add a multi-turn config
  (seal → snapshot → resume → continue) so recurrent-state sealing (§5) is
  exercised end-to-end in the same iterating loop, plus a fork-from-earlier
  config asserting the recompute path.
- **Threshold rows are per-model** (`PRODUCTION_*` re-derivation +
  `QWEN35_*_KV_FACTORS`), derived on the machine that runs the gate: 35B
  rows on dev, 122B/27B rows on the workstation when first run there.

---

## 7. Phasing

- **Phase 0 — acquisition & inspection.** *COMPLETE 2026-08-21.* Reference
  sources fetched and the GGUF schema frozen (§7.1); 0.8B, 9B and 35B-A3B
  downloaded at pinned revisions and their schemas verified against the
  actual files (§7.2, §7.6). The 9B-MTP variant is not fetched — it is
  only needed if the optional drafter work happens (§4.5).
- **Phase 1 — reference forward.** *Built 2026-08-18
  (`candle-transformers/src/models/qwen35/`, 27 unit tests): config +
  metadata parsing, the full DeltaNet reference layer (sequential delta rule,
  causal conv with carried tail, l2-norm, gated norm), gated attention with
  RoPE and KV carry, MoE with gated shared expert, hybrid model assembly
  with per-layer session state, and the dequantize-to-F32 GGUF loader over
  the frozen name schema. The segmentation property (segments-from-carried-
  state ≡ one-shot) is proven at recurrence, conv, layer, and whole-model
  level, plus clone-state-resume identity — the §5 snapshot contract in
  miniature.* Remaining for exit: per-layer parity against llama.cpp on real
  9B weights, once the GGUFs download.
- **Phase 2 — DeltaNet CUDA + state store.** *Complete 2026-08-20.* The §5
  snapshot record end-to-end in the persistence layer (single tail, supersede
  accounting, relocation, reload-cold, inspector; proof test). The
  `RecurrentStateStore` with wave-atomic begin/commit/rollback, enforced
  sequencing, and the byte-exact export/import bridge (schedule-hash
  validated, all-or-nothing). CUDA decode-step + conv-step kernels
  (`candle-kernels/src/delta-net/`), parity-locked to the sequential
  reference on GPU. The **chunked prefill** (`delta_chunked`, chunk 64 —
  the llama.cpp parallel-scan algorithm) implemented tensor-level and
  batched over heads, proven equal to the sequential rule at every chunk
  width, and live inside the layer forward; a fused single-launch prefill
  kernel remains a Phase-4 *optimization* if profiling demands it — the
  algorithm and its oracle are done. Deferred to Phase 4: seal-site
  producer + resume hooks (need the engine model), VRAM-arena residency for
  the state store.
- **Phase 3 — attention path at 256.** *Float milestone complete 2026-08-20.*
  The survey found head_dim 256 GQA already built at `N_PALETTE = 4` with
  64-dim bands (the ladder's 256 rungs, band-generic alloc, the decode
  kernel's `<HD=256, NP=4>` instantiation); an 8-band variant is structurally
  ruled out by the 2-bit pal_map and deliberately not built. Landed: decode
  correctness coverage in the Qwen3.5 (16/2) and Qwen3.8 (24/4) shapes with
  chunk-straddling history + an FP8-arena 256 test; and
  `paged_prefill_float_fallback` — prefill at 256 previously failed loudly —
  which preserves the unrotated-K arena convention (reference-tested,
  including that invariant) and refuses glue explicitly. Deferred: the
  fused 256 prefill instantiation (profiling-driven), the C-ladder quant
  enablement at 256 (ladder rungs 896/1280/1408/1536/2176/2304,
  `palette4_convert` HD templating, the compress/cpu_selection 128 gates) —
  riding Phase 5 with the threshold work; provenance sign-pack widening to
  `sub ≤ 64` if retrieval at 256 needs it.
- **Phase 4 — 35B-A3B production integration + all gates authored.**
  `ManagedBatchedModel` impl with hybrid layer loop; shared ExpertCache
  (host dispatch); the §6 gate tests for **all three** models written and
  compiling — the 35B gate green on dev, the 122B/27B gates build-verified
  with fail-fast capacity guards. The 122B and 27B model wirings themselves
  land here too (dimensional/dense derivatives of the 35B path).
- **Phase 5 — quality.** `PRODUCTION_*`/`*_KV_FACTORS` derivation for the
  35B; C-ladder gate rows; snapshot/resume and fork-recompute gate configs
  green; Zen Code live validation on the dev card.
- **Phase 6 — scale.** Run + tune the 122B-A10B and Qwen3.8-27B gates; their
  threshold rows derive on whichever machine runs them; optional MTP
  drafter; 397B feasibility note.

  *Corrected 2026-08-21:* an earlier revision said these could only run on
  the 2×5090 workstation. That was wrong, and wrong in a way worth naming:
  **model size is not bounded by VRAM here.** Qwen3-30B-A3B already runs on
  the 16 GB dev card because the expert cache streams its experts through
  three tiers (VRAM → pinned RAM → mmap), so a MoE model's *resident*
  footprint is its dense weights plus whatever expert working set fits, not
  its parameter count. The 122B-A10B is a MoE and is therefore feasible
  here — slowly, with a smaller resident expert set — and Qwen3.8-27B is
  dense at roughly the card's size. The workstation buys *speed of tuning*,
  not feasibility. Nothing in Phase 6 is gated on hardware that does not
  exist; it is gated on Phase 4's `ManagedBatchedModel` impl, because the
  gate harness (`TestParams::run<M: ManagedBatchedModel>`) is generic and a
  model plugs into it by implementing that one trait.

### 7.1 Phase-0 findings (2026-08-18): reference code + GGUF schema

Reference implementations fetched from `ggml-org/llama.cpp` master
(re-pin to a specific llama.cpp commit when the loader work starts):
`src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`,
`src/models/delta-net-base.cpp` (the shared chunked delta-rule graph),
`src/models/qwen3next.cpp` (the predecessor for comparison), and
`gguf-py/gguf/constants.py` (authoritative tensor/metadata names).

**Arch names**: `qwen35` and `qwen35moe`. **Per-layer GGUF tensors**
(`blk.{i}.` prefix): the familiar attention set (`attn_norm`, `attn_q/k/v`
+ `attn_q_norm`/`attn_k_norm`, `attn_output`, optional fused `attn_qkv`)
**plus** `attn_gate` (gated attention) and `post_attention_norm`; the
DeltaNet set reuses the SSM namespace — `ssm_conv1d`, `ssm_dt`, `ssm_a`,
`ssm_alpha`, `ssm_beta`, `ssm_norm`, `ssm_out`; the MoE set matches
qwen3moe (`ffn_gate_inp`, `ffn_{gate,up,down}_exps`) **plus a shared
expert** (`ffn_{gate,up,down}_shexp`, `ffn_gate_inp_shexp`) and an optional
fused `ffn_gate_up_exp`. MTP tensors ship under `blk.{i}.nextn.*`
(`eh_proj`, `embed_tokens`, `enorm`, `hnorm`, `shared_head.{head,norm}`) —
"preserved but unused" by llama.cpp, i.e. loadable-and-ignorable until our
drafter phase, exactly as §4.5 assumed.

**Chunked delta rule** (from `delta-net-base.cpp`): chunk size **64**
(16 for the KDA variant), per-chunk log-decay cumsum → pairwise decay mask
(`tri(lower) ∘ exp`), β-weighted K/V, an **intra-chunk unit-lower-triangular
solve** (`ggml_solve_tri`) forming the transition, then a sequential
inter-chunk state carry — i.e. the standard parallel-scan formulation. Our
CUDA port targets this exact graph, with the FP32 state-accumulation rule
from §8 (risk 2); the llama.cpp graph is also the parity oracle for
Phase 1's CPU reference.

### 7.2 Phase-4 findings (2026-08-20): the checkpoint is the authority

First contact with real weights (`unsloth/Qwen3.5-0.8B-GGUF` @
`6ab46149…`, and the source repo `Qwen/Qwen3.5-0.8B` @ `2fc06364…` for
`config.json` + `tokenizer.json`) settled several geometry questions that
the llama.cpp source read alone left open.

**Rotary width is partial.** `config.json` declares
`partial_rotary_factor: 0.25` and the GGUF carries
`qwen35.rope.dimension_count = 64` against `attention.key_length = 256`:
**64 of the 256 head dims rotate, the remaining 192 pass through
untouched**. The MRoPE sections corroborate it —
`rope.dimension_sections = [11, 11, 10, 0]` sums to 32 pairs = 64/2, so
the sections tile the *rotary width*, not the head. Both facts are now
config fields (`rope_dim`, `rope_sections`) with the cross-check enforced
at parse time: sections that do not sum to `rope_dim / 2` are refused,
because a disagreement means one of the two keys was misread and roping
the wrong number of dims is silent, not fatal. NeoX half-split applies
*within* the rotary width — pair `(i, i + rope_dim/2)` at
`theta^(−2i/rope_dim)` — matching ggml's `rope_neox`/`rope_multi`, which
rotates `[0, n_rot)` and memcpys `[n_rot, ne0)`.

This class of error is the reason the real-weights test carries a
**semantic** assertion and not only structural ones: a full-width rotary
still yields finite logits, still satisfies segmented ≡ one-shot, and
still produces deterministic non-degenerate tokens. Only the decoded text
separates a correct stack from a self-consistently wrong one.

**Confirmed against `config.json` (0.8B):** hybrid schedule
`[linear ×3, full] ×6` over 24 layers — attention on layers 3/7/11/15/19/23,
i.e. the `(i+1) % 4 == 0` interval rule, 6 attention / 18 DeltaNet;
`attn_output_gate: true`; `head_dim 256`, `num_attention_heads 8`,
`num_key_value_heads 2`; DeltaNet `linear_num_{key,value}_heads 16` at
`linear_{key,value}_head_dim 128` (GGUF: `ssm.group_count`/
`ssm.time_step_rank` = 16, `ssm.state_size` = 128,
`ssm.inner_size` = 2048); `rope_theta 1e7`; `tie_word_embeddings: true`;
`mamba_ssm_dtype: float32` (independent confirmation of §8 risk 2).

**Vocabulary changed.** 248320, a new tokenizer generation — Qwen2/3-era
token ids are meaningless against these checkpoints, so every fixture
must tokenize rather than hardcode ids. The GGUF ships
`tokenizer.ggml.{tokens,merges}` (`model = 'gpt2'`, `pre = 'qwen35'`) but
no `tokenizer.json`; the gate tests take the tokenizer from the source
repo at a pinned revision and **prove the pairing** (same vocab size, same
string at sampled ids) before trusting an id.

**Not present in the 0.8B GGUF:** `nextn_predict_layers` (the MTP block is
not converted — `block_count 24` is the whole trunk) and
`{arch}.vocab_size` (vocab is derived from the token-table length).
The 0.8B is also a VL model (`mmproj-*.gguf` siblings, image/video token
ids); the text trunk is unaffected.

**The DeltaNet read scale is load-bearing.** `build_delta_net_chunking` and
`build_delta_net_autoregressive` both open with
`q = ggml_scale(q, 1/√S_k)`. It reads like a no-op — `o = S q` is linear in
`q`, and the gated RMSNorm immediately downstream is scale-invariant — and
it was implemented as one. It is not:

```text
  rms(o/√d) = (o/√d) / √(mean(o²)/d + ε) = o / √(mean(o²) + d·ε)
```

Dropping the scale is exactly equivalent to shrinking that norm's epsilon
by `d` (128×). Because `o` here is `β(q·k)v` — small, since `q` and `k` are
L2-normalised and `β = sigmoid(·)` — the epsilon is a *live term* rather
than a divide-by-zero guard, and the difference moves the argmax. The
resulting model produced fluent-looking word salad while passing every
structural check (finite logits, segmented ≡ one-shot, deterministic,
non-degenerate). `ggml_l2_norm`'s floor sits on the root rather than the
sum (`max(√Σ, ε)`, not `√max(Σ, ε)`) for the same class of reason and is
now matched exactly too.

Generalising: in this family the epsilon terms are part of the arithmetic,
not defensive padding. Any transformation justified by "the normalisation
downstream cancels it" is wrong unless it also cancels through the epsilon.

### 7.3 The bring-up oracle (2026-08-20)

Reading the reference source is not sufficient to validate an
implementation — every one of the checks above passed while the model was
wrong, and the defect was in a line of the reference that had been read and
consciously dismissed. What found it was a numerical oracle, assembled from
three independently-derived implementations:

1. **llama.cpp** (`b10514`, prebuilt `llama-completion`) run against the
   very same GGUF, giving ground-truth greedy continuations.
2. **A layer-truncating GGUF rewriter** — `block_count` patched and
   tensor-info entries for dropped blocks removed, with the kept tensors
   repacked (ggml validates that offsets are tightly packed). Paired with a
   tensor-zeroing patcher, this yields models in which exactly one
   subsystem is live, *identically for every implementation reading the
   file*. Zeroing an output projection (`ssm_out`, `attn_output`) neutralises
   a mixer without touching anything else.
3. **An independent NumPy transcription** of `qwen35.cpp` +
   `delta-net-base.cpp`, written from the reference rather than from the
   Rust, so that agreement between the two is evidence rather than a shared
   assumption.

The bisect that resulted: FFN-only (all mixers zeroed) matched exactly,
clearing the embedding, norms, FFN, residual skeleton and tied head;
attention-only matched exactly at every prompt width, clearing the
attention layer; DeltaNet-only diverged, and diverged at a single token,
which excluded the chunked scan and the decay gate and left the read.
Toggling the scale in the NumPy reference reproduced the Rust's exact
token, which identified the defect conclusively.

Note llama.cpp cannot load a truncated model with **zero** attention layers
(it asserts on an empty attention buffer), so DeltaNet-only probes use a
4-block model with `attn_output` zeroed rather than a 3-block one.

### 7.4 Production path — one mixer core, two projection paths

The reference layer is split at the projection boundary
(`delta_net.rs`): `DeltaNetProjections` + `DeltaNetConstants` in,
`delta_net_mix` does conv → split → l2-norm → GQA broadcast → read scale →
recurrence → gated norm, and the caller applies the output projection. The
F32 reference projects with `Tensor::matmul`; the production path
(`quantized_delta_net.rs`) projects with `QMatMul` — and both then call the
*same* `delta_net_mix`.

This is not merely tidiness. §7.2 is a record of that algebra being
transcribed once and getting a load-bearing epsilon wrong; a second
transcription for the production path would be a second opportunity to do
so, and the two would then have to be kept in agreement forever. One
function, two callers.

Weights split by **role**, not size (`quantized_weights.rs`): projections
are matmuls and become `QMatMul` under a uniform `Int8Mode`; norm gains,
the conv kernel, `ssm_a` and the `dt` bias are elementwise constants read
by hand-written tensor algebra and are dequantized to F32 once at load (the
recurrence accumulates, and the checkpoint itself declares
`mamba_ssm_dtype: float32`). The embedding table stays **host-resident** —
the sanctioned CPU `index_select` + single upload of hot-path invariant 3,
which keeps 4 GB of `vocab × hidden` out of VRAM at the 9B's geometry —
while the tied LM head is a device-side `QMatMul`, because that one is a
matmul.

### 7.5 What a hybrid has to correct in its self-description

Three of the numbers the engine asks a model for mean something different
on a hybrid than on a uniform transformer, and each is wrong in a way that
costs memory or correctness if simply inherited (`kv_layout.rs`,
`engine.rs`):

**KV layers ≠ transformer layers.** The 9B has 32 layers and 8 that
attend. The engine already keeps the distinction — `session.num_layers()`
is the count of per-layer KV chunk sets (it sizes chunk headers and is what
`recover_section_cold_refs` reads a sequence back with), while
`model.num_layers()` is transformer depth (it bounds `forward_wave`'s layer
range) — but on a uniform model they are the same number, so nothing has
had to say which it means. `KvLayerMap` is that statement, made once.
Sessions are therefore built with the *attention*-layer count. Passing
depth would allocate 4× the backings, and — because admission prices a
wave's KV as per-layer row cost × layer count — would refuse four times
more prefill than the cache can actually hold.

**The priced intermediate is not the FFN's.** The wave plan sizes its span
from a single "intermediate" width, but a hybrid has two layer kinds with
unrelated widths: a dense FFN carries `intermediate` per row, a DeltaNet
layer carries its fused `[Q|K|V]` projection at `2·key_dim + value_dim`.
The span must hold the larger. On the 9B the FFN wins (12288 vs 8192), so
FFN-only pricing would happen to be right here and silently under-reserve
on a stack with a narrow FFN and wide heads — which is why it is a `max`
with a test that pins the case the naive version gets wrong.

**Provenance depths must land where there is a Q.** The default picks three
depth fractions of the stack; on a 3:1 hybrid those hit a DeltaNet layer
three times in four, and a DeltaNet layer has no Q to capture. Each band is
snapped *down* onto an attention layer (down, because the deepest index is
derived as `n − 1` and must not walk off the end) and its lower endpoint
becomes the previous attention layer — a real two-point window over layers
that have signatures. A stack with no attention layers returns `None`
rather than indices that cannot capture, and is refused a KV session
outright. This is the concrete form of §8 risk 5.

**First production target is Qwen3.5-9B, not the 35B.** It is dense (no
`expert_count`), so it brings up the hybrid wave loop, head_dim-256
attention and the DeltaNet path without the expert cache as a
simultaneous variable; and at 16 K heads against 32 V heads it is the
smallest cached model that actually exercises the mixer's GQA broadcast
(the 0.8B is 16/16 and cannot). Verified on the real Q6_K checkpoint
against the F32 reference built by dequantizing the same weights — so the
only variable under test is the quantized projection kernel — at 2.9%
output / 1.1% state relative error over a 40-token block.

---

### 7.6 Phase-0 closed (2026-08-21): the 35B verified against the doc

`unsloth/Qwen3.5-35B-A3B-GGUF` @ `bc014a17…`, `Q4_K_M`, 22,016,023,168
bytes (exactly its `Content-Length`). Its schema matches every prediction
§2–§3 made from the online research:

`general.architecture = qwen35moe`; 40 blocks at
`full_attention_interval 4` → **10 attention / 30 DeltaNet**; hidden 2048;
`attention.head_count 16` / `head_count_kv 2` at `key_length 256`;
`expert_count 256`, `expert_used_count 8`, `expert_feed_forward_length 512`
with a shared expert at 512; DeltaNet `state_size 128`, `group_count 16`
(K heads), `time_step_rank 32` (V heads), `inner_size 4096` → head_v 128;
`rope.dimension_count 64` against `key_length 256` — the same 0.25 partial
rotary as the 0.8B and 9B; `rope.freq_base 1e7`;
`rope.dimension_sections [11, 11, 10, 0]`. No `nextn_predict_layers`: this
conversion carries no MTP block.

Per-layer tensors match the frozen name schema of §7.1 exactly, including
the merged 3-D experts (`ffn_{gate,up,down}_exps` at
`[n_expert, expert_ffn, hidden]`) and the gated shared expert
(`ffn_gate_inp_shexp` as a `[hidden]` vector).

**One finding that changes Phase-4 sequencing: MoE is on *every* layer**,
DeltaNet and attention alike — there are no dense layers in this model. So
the 35B cannot be loaded at all until the shared ExpertCache path exists;
it is not a loader detail that can be deferred behind a dense bring-up.
The 9B remains the right first engine target for exactly that reason.

**It also confirms `priced_intermediate`'s `max`** (§7.5) on real numbers:
this model's per-expert FFN is 512 while its DeltaNet projection is 8192,
so pricing the wave span on the FFN alone would have under-reserved by
16×. The 9B has the opposite ordering (12288 against 8192), so a
single-model check would have concluded either rule was fine.

### 7.7 The engine wiring (2026-08-21): one wave driver, two sweeps

Making the hybrid drivable by the scheduler needed `ManagedBatchedModel`,
whose one substantial method is `forward_wave`. The uniform implementation
of it was ~530 lines on `BatchedInference<M>` plus a ~390-line
`forward_wave_contexts`, and DeepSeek's own is 1,527 — so the obvious move,
writing a third copy, would have put the same bookkeeping in three places.

It splits cleanly instead. Everything **around** the layer sweep is
model-agnostic: bounding a forward's token count, routing 1-token prefills
to the decode kernel (the paged prefill kernel diverges at `q_len == 1`),
building the three groups' metadata, the caller↔internal token permutation,
the KV rollback that makes a failed wave a retryable one, and the decode
rows' single per-step usage advance. None of it reads anything about a model
but its depth and its device.

So that half is now `models/wave_driver.rs` — `drive_wave` over a `WaveSweep`
trait — and each model supplies only its sweep:

* `BatchedInference<M>` → `forward_wave_contexts` (unchanged, one layer body
  at every index);
* `HybridBatched` → `qwen35/forward.rs` (dispatch on layer kind).

`ManagedBatchedModel::forward_wave` is then two lines on both. DeepSeek keeps
its own override, untouched.

Two things the driver had to learn from the hybrid, both of which were latent
bugs waiting for the second model:

* **`WaveSweep::kv_layer_range`.** The rollback truncated `caches[i]` for `i`
  in the *trunk* layer range. On a 3:1 hybrid the cache vector has a quarter
  as many entries, so the first failed wave would have indexed past its end.
  The translation is now asked for, with an identity default.
* **`WaveGroups::seq_ids`.** A `SequenceContext` carries a sequence's KV and
  offset but not its identity, which is all a uniform transformer needs. A
  recurrent mixer has to key `S` and the conv tail by *something*.

The hybrid sweep's own four departures are the ones §7.5 predicted: caches
indexed by KV layer, recurrent state lifted for the sweep and committed or
rolled back with the wave, `rope_cs` from `RotaryLayout` rather than
`compute_rope_cs`, and glue refused outright.

**The loader's ordering is now structural, not a caller's obligation.** The
expert cache's capacity is `(span − reserve) / slot_bytes` and the span is
*measured*; the measurement only means what it says once every dense tensor
is resident. `load_quantized_model` therefore takes the cache **builder** as
a callback and calls it at that one point, then grafts the result onto the
layers that route (`PendingLayer::resolve`). Passing a finished cache in — the
previous signature — required the caller to size the weight zone against a
span still containing none of the model, which is the failure Qwen3-MoE's
loader carries a comment about having made.

**One numerical boundary the wave forced.** `S` is a running sum with no bound
on how many additions it accumulates, so it must stay F32 while the wave's
activations are F16/BF16. The projections keep the activation dtype (their
kernels want it) and the mixer's inputs are widened at the layer boundary,
narrowed again before the output projection — two conversions per DeltaNet
layer, and they are the arithmetic rather than an oversight.

With F32 guaranteed through the mixer, the fused decode kernels became
reachable: a one-token span now takes `delta_net_decode_step` and
`delta_net_conv_step` — one launch each, state updated in place — instead of
the chunked scan's op graph. Same numbers either way (`cuda.rs` locks both to
the sequential reference); it is which kernel runs, never which arithmetic.

### 7.8 Three defects the gate found, and what each one taught

The hybrid ran end to end — 80+ t/s, fluent English — while answering the
wrong question. Getting from there to a passing gate took three fixes, and
none of them was visible as a crash.

**1. The chunked scan overflowed its own mask.** `D[i][j] = exp(G[i] − G[j])`
was evaluated over the whole matrix and masked to the lower triangle
afterwards. `G` is a cumulative sum of negative log-decays, so the *discarded*
half's exponent grows positive with token distance; past ~88 it overflows to
`+inf` and `inf × 0` is `NaN`, which `unit_lower_inverse` then spreads across
the chunk. Content-dependent, so a random-activation probe over 40 tokens
stayed finite while the **reference** went non-finite at token 23 of one real
prompt and 27 of another. Clamping the exponent at zero costs nothing on the
kept half, which is already ≤ 0.

**2. Recurrent state outlived its session.** The map is keyed by sequence id
and owned by the *model*, because a DeltaNet layer has no per-session storage
to put it in. A new session hands the same ids back with an empty KV — so
attention was correct, every shape matched, nothing errored, and the model
answered as though it remembered a conversation the prompt never had. The
invariant is now explicit: **a sequence standing at offset 0 has no history,
so its recurrent state is reset, not merely created.** Tying the reset to the
offset makes the recurrence follow the KV, which is what the scheduler
actually manages.

**3. The DeltaNet GQA broadcast was blocked where ggml tiles.** V head `j`
reads K head `j % h_k`, not `j / group`. `ggml_repeat` tiles its source; an
`expand` over an inserted axis blocks it. Both give the same shapes, and the
two are **identical whenever `h_v == h_k`** — which is true of the 0.8B, the
only model the reference had ever been validated against on llama.cpp. The
9B is 16 K heads to 32 V, and there the difference cost the model its ability
to follow an instruction while leaving it fluent: it still knew Paris was the
capital of France, it just could not carry out the rewrite.

The through-line is that **all three were invisible to the oracle in use.**
What broke the deadlock was building better ones, in this order:

* *wave vs reference*, swept by prefix length and across KV dtypes — turns
  "wrong answer" into a number per length, and separates storage precision
  from engine fault (F32 KV as bad as BF16 ⇒ not the cache);
* *wave one-shot vs wave segmented* — needs **no** oracle at all, which is
  what makes it decisive when the reference is itself suspect. It found
  defect 2 outright, and the reference passing the same split at 1.000000
  while the wave scored 0.76 is what localised it to the engine;
* *decode compared step by step*, because prefill and decode share almost
  nothing — a prefill that matches exactly says nothing about the fused
  one-token recurrent step;
* *a factual one-liner on the real checkpoint* — "does this model know Paris"
  separates a mis-read checkpoint from a model that will not follow the
  instruction, and those two look identical in a gate diff.

Defect 3 in particular is a standing warning about single-model validation:
the 0.8B is the family's cheapest oracle **and** its least discriminating,
because `h_v == h_k` collapses a distinction every larger sibling depends on.
Anything derived on it alone must be re-checked at 16/32.

### 7.9 The mixer on the wave span (2026-08-22)

The hybrid brought a second recurrent buffer per sequence and forty-odd
temporaries per DeltaNet layer, and all of it was reaching the CUDA allocator.
Measured with `--features forbidden_allocations` over the 0.8B gate's decode
window at four contexts: **143 distinct sites, 43,344 allocations, 1.19 GB**.
The transient tier was reserved and idle throughout — the attention arena's
high-water mark was 64 KB.

Two independent causes, and the second turned out not to be about DeltaNet at
all.

**The wave snapshot allocated.** `RecurrentStateStore::begin_wave` copied every
layer's `S` and conv tail into a *fresh* allocation and `commit_wave` dropped
it. At decode a wave is one token, so that is an allocate/free pair per layer
per session per token: 679 MB of the 1.19 GB, the single largest allocator in
the loop. The entry copy is now a buffer the slot owns for its whole life and
`begin_wave` writes into it; commit drops nothing and rollback copies back.
Peak footprint is unchanged — every session in a wave holds a backup at once
either way — and rollback now *restores* the live buffer rather than swapping a
different tensor into the slot, which the fused decode kernels (which write `S`
in place) had a right to expect.

**Four op paths did not carry provenance.** Seeding the mixer at its ln1 put
the chain on the span, and it went nowhere: the first projection dropped it.
Operand provenance is only as good as its weakest op, and a single
non-inheriting site silently relocates *everything downstream of it*:

| site | what it did |
|---|---|
| `CudaStorage::matmul` | allocated with a bare `dev.alloc`, stamping `Backing::Owned` |
| `QMatMul::forward_live`, float-weight arms | copied the activation off the wave before the matmul, for a `Tensor` return the signature had already stopped requiring |
| `mul_mat_{,vec_}via_q8_1` | took no backing, so every non-int8 quantized projection landed on the pool |
| `CudaStorage::affine` | same as `matmul` — and it is what every scalar `x * k` lowers to |
| `RmsNorm::forward_dynamic`, float arm | accepted a `root` and ignored it, so with int8 off the FFN generation opened, took nothing, and reset |

That last one is why the FFN arena also read zero. None of these were
DeltaNet-specific; the hybrid just exercised them harder than anything before
it.

After both: **26 sites, 1,557 allocations, 89 MB** — 96% fewer allocations, 93%
fewer bytes — and the arenas carry the traffic the tier was already reserved
for, so the saving is against the *pool*, on top of a reservation that was
being paid for regardless.

Two things worth keeping:

* **A phase reporting a zero peak is not evidence of a phase doing nothing.**
  It reads identically to a phase whose chain was never seeded. The
  `wave arenas:` line and the forbidden-allocation report answer different
  halves of the question and neither is sufficient alone.
* **`WaveBuffer::AttnNorm`/`FfnNorm` are priced dense, not q8a128.** The two
  encodings are alternatives fixed at session creation, and the plan has to
  bound whichever runs. Pricing the smaller would under-size a float session's
  span, and what gets squeezed out at the end of a phase is a buffer a *kernel
  wrapper* allocates — which refuses rather than falling back to the pool.

Still on the pool, deliberately: `l2_norm`'s `maximum(eps)` uploads its scalar
per call (`softplus`'s became `relu`, which is the same function without the
upload; the l2 floor has no unary equivalent and its epsilon is bit-locked to
the llama.cpp reference, so it wants a scalar-max kernel rather than an algebraic
rewrite), and the logits copy out of the head span, which predates this work.

### 7.10 The mixer runs per wave, not per sequence (2026-08-22)

Against the dense models in the same harness the hybrid was not slow so much as
**flat**: decode from one context to four gained 1.37×, where Llama-3B gains
3.19× and Qwen2-0.5B 3.06×. A model that does not batch is a different kind of
problem from a model that is slow, and it is the one that matters at the
64-session target.

The cause was that `delta_net_mix_wave` called the *whole layer* once per
sequence. Only two steps in a DeltaNet layer are per sequence — the causal conv,
which carries a tail, and the delta rule, which carries `S`. The other forty-odd
ops compute each row from that row alone. Running everything per sequence
therefore re-did all of it N times and, decisively, **re-read all five
projection weights N times**: a decode step is weight-bandwidth-bound, so N
sessions cost N× one session and batching bought nothing.

[`delta_net_mix_spans`] now takes the whole wave and a `DeltaNetSeq` per
sequence. The row-wise majority runs once over all `T` rows; the two carried
steps run per sequence over their own slice, and their results are concatenated
back in span order.

Two things this cost, and one of them was a mistake worth recording:

* **The first version wrote each span's result into a preallocated shared
  buffer** (`empty_beside` + `slice_set`). Correct, and a 35% regression at one
  context — because with a single span that is a full copy of a buffer the old
  code used in place. `cat` is the same work for several sequences and *free*
  for one (it hands a single argument straight back), and a wave carrying one
  sequence is the whole of a single-session decode and most of prefill. The
  measurement that found it was the gate's own 1-context column; the
  forbidden-allocation detector showed allocations *falling* (1,557 → 423) the
  whole time, which is exactly why "fewer allocations" is not a proxy for
  "faster".
* **The spans must tile the buffer, in order.** Both carried steps slice by
  `start`/`len` and their outputs are concatenated positionally, so a gap, an
  overlap or a reordering feeds one sequence another's rows — a plausible wrong
  answer, not a crash. Checked at the top of `delta_net_mix_spans` and pinned by
  `spans_must_tile_the_buffer`.

Result (gate, decode t/s at 1 → 4 contexts):

| model | before | after |
|---|---|---|
| 0.8B | 45.3 → 62.0 (1.37×) | 53.0 → 159.8 (**3.01×**) |
| 9B | (1.26–1.39×) | (**2.69–2.90×**) |
| 35B | 5.9 → 20.6 (3.49×) | 5.2 → 24.5 (**4.71×**) |

**Read the ratio, not the columns.** Absolute throughput on the 4090 Mobile
carries a large run-to-run band — the 0.8B's engine-only prefill measured 2764,
1578, 1448 and 1357 t/s across four runs of the same binary, decaying with GPU
temperature, and a gate at the same position in the same suite has come back 30%
apart on consecutive days. The 1 → 4 ratio is measured minutes apart under
near-identical conditions and is the number that survives that. Single-sequence
throughput is unchanged by this work within the noise: the 9B's engine-only
figures sit at 737–777 t/s prefill and 32.4–34.7 t/s decode both before and
after.

`spans_equal_running_each_sequence_alone` pins the property on CPU with two
sequences of different lengths, comparing both the activations and each
sequence's carried state; `batched_mixing_equals_mixing_each_sequence_alone`
does the same on the real 9B checkpoint and measures `rel 8.5e-8`.

**Still per sequence:** the two carried steps themselves. At decode both are
single fused kernels, and both already take a leading `seqs` axis — the blocker
is that each session's state lives in its own store, so batching them means
passing an array of state pointers rather than one. That is the next step, and
it is what takes the remaining per-sequence launch pair per layer to zero.

### 7.11 int8 was off, and the table said so without anyone reading it

The 9B and 35B gates ran every projection on the **float diagnostic path** —
`quantized_matmul::forward_live` casts activations to F32 and multiplies against
a dequantized weight when the mode is `Off` — for the whole bring-up. The 0.8B
and 9B gates pinned `Some(Int8Mode::Off)`; the 35B passed `None` and got auto.

The reason it stayed invisible is worth more than the fix: **the table's `int8`
column is set by `TestParams::with_int8mode` and the loader's mode is set
separately**, so a gate that pins one and defaults the other prints a numeric
path it is not running. The 35B had been on auto for months and printed `off`.
Both gates now resolve one value and pass it to both places.

`Int8Mode::Performance` on the 9B, four matched runs against four int8-off runs
under the same rebuild-then-repeat protocol and the same temperature ramp:

| run | prefill off → perf | decode @4 ctx off → perf |
|---|---|---|
| 1 | 524.6 → 617.6 (+17.7%) | 113.7 → 125.5 (+10.4%) |
| 2 | 503.7 → 561.4 (+11.5%) | 109.1 → 116.8 (+7.1%) |
| 3 | 495.9 → 542.6 (+9.4%) | 104.7 → 112.4 (+7.4%) |
| 4 | 478.5 → 558.1 (+16.6%) | 91.8 → 94.8 (+3.3%) |

**~+14% prefill, ~+7% decode**, at 100% gate correctness. `Performance` rather
than `auto_sized`: auto picks `Precision` on this card, the two differ only in
the weight twin (the q8a128 activation is identical), so it is a
throughput/accuracy dial and a gate should hold it steady rather than let it
move with the device.

The 0.8B stays `Off` and that is a statement about the checkpoint, not a default:
it is unquantized BF16, so every projection is a float weight and there is no KO
twin for an int8 mode to select.

### 7.12 The chunked scan solves instead of inverting (2026-08-22)

The scan needs `T·(βv)` and `T·(βk⊙e^G)` with `T = (I + A)⁻¹`.
[`unit_lower_inverse`] formed `T` by recursive 2×2 block inversion — `log₂ c`
levels of a handful of launches each, about **eighty serially dependent
host-issued ops per chunk** — and then multiplied twice.

Both right-hand sides share the left side, so they concatenate into a single
triangular solve, and a solve never forms the inverse. On CUDA in F32 that is
one `cublasStrsmBatched`. cuBLAS has no *strided*-batched trsm, so the per-head
address arrays are written by a small device kernel
(`run_delta_net_batch_ptrs`) rather than uploaded from the host — the loop is
exactly where a host round trip must not be.

The subtle part is that cuBLAS reads column-major and every tensor here is
row-major, so the buffers are **reinterpreted, not transposed**: a row-major
`[c, c]` with leading dimension `c` *is* a column-major `Aᵀ`, and the transpose
of a lower triangle is an upper one. `A X = B` becomes `Xᵀ Aᵀ = Bᵀ`, which is
trsm's right-side form. Hence `SIDE_RIGHT` and `FILL_MODE_UPPER` on data that is
neither — they describe the column-major view. A wrong `side` or `uplo` there
produces a plausible matrix rather than an error, which is why
`solve_matches_the_explicit_inverse` pins the solve against the reference form
directly.

**Measured by normalising against the attention layers**, which this change does
not touch — end-to-end gate t/s could not resolve it at all, the machine band
being wider than the effect:

| prefill, per layer | attention | deltanet | ratio |
|---|---|---|---|
| before, 0.8B | 12.1 ms | 16.7 ms | 1.38× |
| after, 0.8B | 19.7 ms | 12.9 ms | **0.65×** |
| before, 9B | 35.9 ms | 22.6 ms | 0.63× |
| after, 9B | 42.0 ms | 19.0 ms | **0.45×** |

A DeltaNet layer went from costing more than an attention layer to about half
of one. The wave oracle stays at cosine 1.000000 to 649 tokens.

**A coverage gap this found.** The first version handed `matmul` a strided
`narrow` (splitting the solved buffer back into `u` and `kcd` leaves a gap
between rows) and it reached a gate run, not a unit test — because *every* scan
test in `delta_net.rs` runs on the CPU, where `solve_pseudo_values` takes the
explicit-inverse fallback. Nothing exercised the solve. `chunked_scan_matches_
sequential_on_cuda` now asserts the same property where the solve actually runs.

**What this reframes.** The claim that "the chunked scan is ~80% of prefill" was
a misreading of the per-layer profile: DeltaNet *layers* are 80% of prefill
because there are three times as many of them, and each includes its own
projections and FFN. Each was only 38% more expensive than an attention layer,
and is now cheaper than one. On the 9B the eight attention layers take 336 ms of
a 792 ms prefill — **42% of the time from 25% of the layers** — so the
`head_dim` 256 fallback below is now the dominant per-layer prefill cost.

### 7.13 The prefill KV *write* was the largest span, not the attention

With the mixer batched and the scan solving instead of inverting, the remaining
prefill cost looked like the `head_dim` 256 attention fallback. It was not.
Instrumenting `paged_prefill_float_fallback` (permanent spans; zero-cost with
the `profile` feature off) on the 9B:

| span | 1 ctx | 4 ctx |
|---|---|---|
| `prefill_fb:attention` | 165.6 ms (×16) | 30.2 ms (×64) |
| `prefill_fb:kv_write` | **184.1 ms (×16)** | **792.2 ms (×64)** |
| `prefill_fb:rope_repeat` | 4.8 ms | 19.4 ms |

The KV write cost **more than the attention it feeds**. The tell is that its
per-call cost barely moved (11.5 → 12.4 ms) while tokens per call fell four-fold:
launch-bound, not bandwidth-bound. `write_contiguous_float` walks 32-token block
× KV head × palette band, each step a `narrow` + `contiguous` + a slot write —
about 2,700 tiny ops per attention layer at 649 tokens. The paged prefill
*kernel* never pays this because it scatters K/V itself; only the float fallback
writes from Rust, so `head_dim` 256 costs the write path as well as the
attention path.

**Fixed by generalising the primitive that already existed.**
`kv_migrate_copy` is the tier-migration scatter/gather: a host-built plan of
`(src, dst, len)` records, one block per record, one launch. It could not serve
the band write because it is contiguous-to-contiguous and a band's source is
`head_dim / N_PALETTE` elements per token, `head_dim` apart — which is precisely
the contiguity the per-band `.contiguous()` was manufacturing, one tiny copy at
a time.

`MigrationRecord` now carries `rows`/`src_stride`/`dst_stride`, and the three
arrays are **nullable**: an all-contiguous plan — every migration plan — uploads
exactly the three arrays it always did, and the kernel reads one row per record,
which is the copy it always was. The generalisation costs the migration path
three null pointers and no memory. `MigrationPlan::push` sets `rows = 1`;
`push_strided` marks the plan.

`try_plan_batched_write` builds one strided record per (block, head, band) and
returns `None` — deferring to the original walk — when a band is not a plain
copy: a quantized tag (`quantize_into_slot` computes scales), a per-band dtype
cast, or a non-GPU arena.

| span | before | after |
|---|---|---|
| `prefill_fb:kv_write` @1 ctx | 184.1 ms | **3.5–4.3 ms** |
| `prefill_fb:kv_write` @4 ctx | 792.2 ms | **15.9 ms** |

**43–50×**, taking the 9B gate's prefill from ~560–620 to ~790–820 t/s at 100%
correctness, oracle unchanged at cosine 1.000000. `prefill_fb:attention` is now
unambiguously the dominant prefill span, and the O(T²) materialisation below is
the next item.

**The other structural gap is not DeltaNet's.** `head_dim` 256 falls off the
optimized attention path in three places: the int8 prefix-attention kernel is
not instantiated at 256, so prefill takes `paged_prefill_float_fallback` →
`standard_attention_prefill`, which materialises the full `[B,H,T,T]` score
matrix and rebuilds its causal mask every layer; `want_q8` is gated on
`head_dim == 128`, so decode never gets the packed context and `o_proj` pays a
standalone quantize; and glue is 128-only and refused outright. That is why
prefill is ~8× a comparable dense model rather than ~2×.

### 7.14 The int8 prefill kernel at head_dim 256 (2026-08-22)

The first of those three is closed: `paged_prefill_int8_kernel.cuh` is now
instantiated at 256 and the float fallback is unreachable for this stack.

**What actually blocked it was occupancy, not correctness.** Partial rotary
was never the obstacle — `RotaryLayout` (§7.5) permutes the head dims so the
kernel's full-width RoPE reproduces Qwen3.5's 64-of-256 rotary exactly. The
obstacle was that both of the kernel's budgets break at 256: the tile overlay
is ~41 KB (`s_v8t` and the raw staging scratch are linear in `HEAD_DIM`), far
past the 25.6 KB union arena that 4 blocks/SM needs; and `q_frag` alone
doubles to 32 registers, so the 64-register cap is unreachable however the
output dims are split.

The generalisation is the warp partition. The 8 warps were hard-split
`(row_tile = warp >> 1, dim_half = warp & 1)` — two warps per m16 row-tile,
each accumulating half the output dims to keep `o_acc` at 32 registers. That
becomes `i8_dim_split(HEAD_DIM)`: **2** through 128 and **4** at 256, so a warp
always owns at most 64 output dims and `o_acc` stays at 32 regardless of head
width. `i8_min_blocks` and `i8_smem_budget` follow it (4 blocks / 25.6 KB
through 128, 2 blocks / 47 KB at 256), and the launcher's grid and split-KV
fan-out read the same functions.

Measured with `nvcc -Xptxas -v` on sm_89:

| HEAD_DIM | regs | smem | spill | blocks/SM |
|---|---|---|---|---|
| 128 | 64 | 21,480 B | 468 B | 4 |
| 256 | 128 | 41,768 B | **24 B** | 2 |

At 256 the register and smem limits both land on 2 blocks/SM, so neither is
wasted, and the residual spill is *lower* than the 128 path's. **128 is
byte-identical to before** — same registers, same smem, same spill — because
`i8_dim_split` returns the value that was hard-coded.

The cost is that a 4-way split duplicates QK across four warps instead of two,
so a compute-bound shape pays twice the redundant QK. It is bought back many
times over: the fallback ran per sequence, expanded K/V for GQA, materialised
`[1,H,T,T]`, and rebuilt the causal mask every layer.

0.8B, engine-only spans (`--features cuda,profile`):

| span | fallback | kernel |
|---|---|---|
| `prefill_fb:attention` | 119.4 ms (×12) | — |
| `prefill_fb:kv_write` | 3.0 ms (×12) | — (scattered in-kernel) |
| `prefill_fb:rope_repeat` | 3.2 ms (×12) | — (applied in-kernel) |
| `prefill:kernel` | — | **6.9 ms (×12)** |

Gate correctness unchanged: 4/4 sessions exact on all three configs.

**Routing, and a latent hole it closed.** The old branch tested
`head_dim != 64 && head_dim != 128` and bailed on glue *inside* it — so glue at
head_dim 64 fell through the `match` to plain prefill and silently lost its
masking. The refusal is now stated once, up front, against the condition that
actually matters (`is_cuda_paged && head_dim == 128`), before any routing
decision. `int8_prefill_head_dim` is the single source of truth for the
instantiation set, read by both the router and the kernel binding's own guard.
The float fallback stays for head widths outside {64, 128, 256}, which is real
capability coverage — `head_dim` comes from GGUF metadata, not from a literal.

**And for F32 reference sessions**, which is what the oracle caught. The kernel
is instantiated over F16 and BF16 and emits its context in the dtype it
computed in; the arena's `collapse_compute` maps an F32/F64 reference arena to
BF16 for the kernel's *operands*, but nothing maps the result back, so the
first RMSNorm downstream met an F32 weight against a BF16 activation. Casting
per layer would be a full-tensor pass on the hot path for the one mode that
exists to be exact rather than fast, so `int8_prefill_act_dtype` routes an F32
session to the float fallback, which computes at the session's own dtype
throughout. The glue guard carries the same condition — glue would break
identically, and there is no fallback that masks it.

### 7.15 The wave plan was 21% short, and the region pad was hiding it

Landing the kernel exposed a **pre-existing** mis-pricing. The attention span
carved 96,768,000 B of a 96,883,200 B budget — 99.88% — *before* the change;
adding one more declared buffer (`AttnOutput`, which the fallback had been
putting on the pool) tipped it over.

`KV_WAVE_CENSUS=labels` itemised the failing generation, and at rows = 2100,
hidden = 1024 it decodes exactly:

| carve | bytes | what |
|---|---|---|
| `RmsNorm` | 4,300,800 | AttnNorm, hidden × BF16 |
| `to_dtype` | 8,601,600 | hidden × **F32** — upcast |
| `matmul` | 34,406,400 | attn_cols × **F32** — Q projected in F32 |
| `to_dtype` | 17,203,200 | attn_cols × BF16 — cast back |
| …same three again for K, and for V | | |

The Q/K/V projections **round-trip through F32**: upcast the operand, run the
dequantized GEMM there, cast the result back. Six buffers, 68.8 MB at this
width, none of them declared — the whole gap between the planned span and the
carved one, absorbed until now by the 16 MiB `REGION_BYTES` pad.

`WaveBuffer::QkvProjOperand` and `QkvProjAccum` price it, mirroring what
`GateGemm`/`UpGemm`/`DownGemm` already do for the FFN. They are **conditional
on `ModelGeometry::projection_accum_roundtrip`**, because a packed session's
projections consume the norm's q8a128 output and emit `act_dtype` directly —
charging both unconditionally is a 95% over-bound on Qwen3-30B-A3B's attention
span, against a chain whose census shows no upcast at all. Only this stack sets
the flag, so no other model's reservation moves by a byte; the 30B's pinned
union margin is unchanged at 17–19%.

Two instruments were sharpened in the process, both permanent:
`wave_census`'s `ALLOCATOR_FRAMES` now skips `CudaStorage`, which was
absorbing every matmul-family carve under one uninformative label; and a span
that *exhausts* now itemises its generation, since the ordinary report only
fires when a generation completes and sets a high-water mark — which a failing
one never does.

**This makes `prefill:qkv_proj` the dominant prefill span** — 736.0 ms of a
1287.1 ms 4-context prefill. Eliminating the round trip is hot-path invariant
1 ("no `to_dtype` in the loop — kernels emit the final type") and is the next
prefill item, ahead of anything remaining in attention.

**Correction (same day):** that conclusion was read off a mismeasured span.
`profile_now` is a bare `Instant::now()` and `prefill:qkv_proj` was the first
*synced* span in an attention layer, so everything queued and not yet awaited —
on this stack, whole DeltaNet layers — drained inside it. With `dn:proj` /
`dn:mix` / `dn:out_proj` / `dn:ffn` spans added and a `profile_sync` at the
drain point, the real numbers (9B, BF16×1): `prefill:qkv_proj` **8.2 ms**, not
145.1; the dominant span is **`dn:mix` at 136.6 ms of a 226.1 ms prefill —
~60%**. The F32 round trip in `dn:proj` (45.1 ms) is deliberate (§7.16 below);
the mixer is the target.

### 7.16 The fused prefill scan (design)

`dn:mix` is `delta_net_mix_spans` — the recurrent mixer, written entirely as
individual candle ops. Decode's carried steps are two fused kernels
(`delta_net_decode_step_f32`, `delta_net_conv_step_f32`); prefill's are ~220
launches per layer per sequence: ~65 per 256-token chunk in `delta_chunked`
plus ~25 outer. At ~2.5 GFLOP/layer (≈0.2 ms of arithmetic) against a measured
2.85 ms/layer, the mixer is **launch-bound**, with three layout costs stacked
on top: two full-width transposes for the conv (`[T,conv_dim]` ↔
`[conv_dim,T]`, ~32 MB each way), the `s_fla` state copy **per chunk**
(`state.transpose(1,2).contiguous()`, 8 MB/chunk on the 9B), and the GQA
`repeat` materialising q/k at `h_v` width.

**Why not the decode kernel at t>1:** it holds `S` in global and touches it
once — correct at one token by definition. Looping it over T re-reads and
re-writes the full state per token (64 KB/head at d=128, past any smem/register
budget), which is exactly what the chunked form exists to amortise. Prefill
needs a kernel for the *chunked* algorithm; what carries over from decode is
the F32-only state math, the `[rows, d_v, d_k]` stored orientation, the
in-place update, and the parity oracle.

**Three kernels, chunk width 64** (`delta_net_prefill_kernel.cuh`):

- **K0 — conv + SiLU, token-parallel.** `y[t,c] = Σⱼ kern[c,j]·window[t,c,j]`
  with the window drawn from `[tail | x]`; new tail written to a **separate**
  buffer (blocks handling `t < K−1` read the old tail concurrently, so an
  in-place shift races; the caller copies the fresh tail into the state).
  Reads `x` token-major `[T, conv_dim]` directly — the channel-major layout
  existed only for candle's conv, so both transposes die with this kernel.
  SiLU folded into the epilogue.

- **K1 — intra-chunk build + solve, parallel over (V-head × chunk).** Per
  block: warp-scan `G = cumsum(g)`; build
  `A[i][j] = β_i(k_i·k_j)·exp(min(0, G_i−G_j))` strictly-lower in smem (the
  exponent clamp is load-bearing — §7.8); forward-substitution solve of
  `(I+A)X = [βv | βk⊙e^G]` with **one column per thread** (d_v+d_k = 256
  columns = 256 threads, X register-resident, 64 fully-unrolled steps,
  `A[i][j]` reads are smem broadcasts); also the inclusive-mask dot grid
  `kq[t][s] = (q_t·k_s)·D[t,s]` while k is already in smem. Emits `u`, `w`
  `[H, T_pad, d]`, `kq [H, T_pad, 64]`, `G [H, T_pad]`. c = 64 because the
  whole triangle then lives in smem — DELTA_CHUNK = 256 was tuned for the
  launch-bound cuBLAS path (fewer chunks = fewer launches), a constraint the
  fused kernel does not have. `chunked_matches_sequential_for_all_chunk_widths`
  already validates every width.

- **K2 — sequential state pass with fused output.** Grid (V-head × d_v-tiles);
  each block owns a register-resident tile of `S` **in stored orientation**
  and loops chunks in order: `v_new[:,tile] = u[:,tile] − w·Sᵀ[tile]`
  (block-local, never written to global); output
  `o = (q⊙e^G)·Sᵀ + kq·v_new` scattered directly into token-major
  `[T, H, d_v]`; `S[tile] ← e^{G_last}·S[tile] + v_newᵀ(k⊙e^{G_last−G})` in
  place. One 32 KB smem buffer is reused in three phases per chunk (w → q → k).
  The `s_fla` copy dies — the block reads `S` as stored. The FLA-style
  3-kernel split (checkpoint per-chunk states + separate parallel output pass)
  is the fallback if K2's occupancy measures poorly; it costs a 23 MB
  checkpoint buffer and a third launch.

**Deliberately unchanged:** all F32, FFMA only, no TF32 — the state is the
unbounded running sum and kernel semantics match the reference modulo
reduction order. The decode `t == 1` path, l2_norm, the gate elementwise ops,
and `rms_norm_per_head · silu(z)` stay as they are. `delta_chunked` stays,
unchanged, as the CPU/non-F32 reference. Phase 1 keeps the GQA `repeat`
(kernels index `h` directly, decode path untouched); dropping it is a
follow-up that touches the decode kernel too.

**Expected shape:** ~220 launches → ~12 per layer per sequence; the s_fla,
transpose, and `[H,c,c]` intermediate traffic replaced by `u`/`w`/`kq`
(~29 MB per layer-seq at 649 tokens, wave-first with pool fallback). Target
dn:mix ≲0.5 ms/layer from 2.85. Validation: A/B against `delta_chunked` on
identical inputs, segmented-equals-one-shot through the kernel (the state
carry across waves is the fork/glue guarantee), the 12-test GPU suite, the
three gates, and a census re-check of the attention span.

**Built and measured (same day).** Kernels in
`candle-kernels/src/delta-net/delta_net_prefill_kernel.cuh`, wrappers in
`models/delta_net/cuda.rs` (`delta_net_conv_prefill`, `delta_net_prefill_scan`), routed
from `conv_silu_spans` / `delta_advance` on CUDA + F32 (+ d = 128 for the
scan). All 7 CUDA parity tests passed on the first run — the fused scan
against the sequential rule at d = 128 with a ragged tail, the mid-chunk
segmented state carry, and the conv against `causal_conv1d` across chained
segments including one shorter than the tail. Full GPU suite 12/12; 9B gate
4/4 exact on all three configs; decode untouched.

Measured on the 9B (within-run spans, both runs on battery):

| span (BF16×1) | before | after |
|---|---|---|
| `dn:mix` | 136.6 ms (×48) | **84.7 ms** (×48) |
| `dn:mix` @4ctx | 689.0 ms (×72) | **538.5 ms** (×72) |
| `bench:bulk_total` | 226.1 ms | **197.0 ms** |

Short of the ≲0.5 ms/layer target: 1.8 ms/layer, from 2.85. The scan is now
three launches (conv + intra + state), so the remainder of `dn:mix` is the
tensor ops *around* the kernels — in likely order of cost: the GQA `repeat`
materialising q/k at `h_v` width (~2 × 10 MB per layer at 649 tokens), the
`rms_norm_per_head · silu(z)` epilogue, l2_norm, the sigmoid/softplus gates,
and the per-sequence slicing. Follow-ups, in that order: drop the `repeat`
(kernels index `kh = h % h_k` — touches the decode kernel too), fold the
gates into the intra kernel (it already reads g/β per chunk), and fold the
epilogue into the state kernel's output store.

**Follow-ups built (same day): the kernels read the mixer's buffers through
strides.** All three landed at once, plus two the list missed, because they
are one interface change: the kernels (decode step included) now take the
l2-normed Q|K stack, the conv output's V columns as a strided view, and the
*raw* gate projections with `dt_bias`/`a` —

- the GQA broadcast is an index (`kh = h % h_k`), never a materialisation;
- the gates are computed in-kernel (the reference `max(x,0)+log1p(e^{−|x|})`
  softplus and sigmoid, exactly);
- the read scale is applied on q loads, so no scaled copy of q exists —
  and it is load-bearing (the epilogue's epsilon floor, §7.8);
- each span writes its own rows of one whole-wave output, so a
  multi-sequence wave needs no concatenation;
- `delta_net_norm_gate` does `rms_norm_per_head · silu(z)` as one launch
  over the wave.

The v-reshape copy and the per-span `rows()` copies died as a side effect —
a span is a base-pointer offset. `delta_advance` is gone; the fused/fallback
branch lives once, in `delta_net_mix_spans`, and the tensor-op fallback is
byte-for-byte the old arithmetic. Parity: 8/8 CUDA tests first run — the
in-kernel tiling is pinned by an `h_k = 2, h_v = 4` case against the tiled
reference, the epilogue against the op forms. Suite 12/12; 9B gate 4/4 exact.

Measured on the 9B (BF16×1, within-run, battery):

| span | ops (§7.16 start) | kernels v1 | strided kernels |
|---|---|---|---|
| `dn:mix` prefill | 136.6 ms | 84.7 ms | **61.2 ms** |
| `dn:mix` @4ctx | 689.0 ms | 538.5 ms | **429.4 ms** |
| prefill total | 226.1 ms | 197.0 ms | **174.0 ms** |
| `dn:mix` decode (×240) | ~105 ms | ~105 ms | **45.3 ms** |

2.2× on the prefill mixer end to end, 1.3× on total prefill — and the decode
mixer **halved**, which v1 never touched: a decode step was paying the
repeat, the gate launches, and the ~6-launch epilogue per token, and all
three fold into the same strided interface. Per-layer prefill cost is now
1.28 ms against the ≲0.5 target; what remains inside `dn:mix` is the conv +
SiLU pass, l2_norm, and the scan kernels themselves — the next cut is
folding l2_norm and the Q|K|V split into the conv kernel's epilogue, which
is also the last full-width copy standing.

**That cut landed (same day): the conv output IS the operand buffer.** Both
conv kernels (prefill and the decode step) gained a SiLU + Q|K-norm
epilogue — `dn_silu_norm_epilogue` in `delta_net_common.cuh`, the reference
`l2_norm` exactly (`x / max(sqrt(Σx²), eps)`, floor on the root, over each
128-dim head of the SiLU'd values). The reduction is block-local by an
alignment fact: `qk_channels = 2·h_k·128 = h_k·256`, so a 256-thread block
is always wholly Q|K (two complete head groups) or wholly V, never a
fragment. The separate `qk` tensor — the last full-width copy in the mixer —
no longer exists: `DeltaNetFused` carries only the conv output, `h_k` is
derived from `conv_dim/d − h_v`, and the split, SiLU, and l2_norm launches
are gone from the fused path. The fallback keeps the op forms; the conv
parity tests compare the kernels against `causal_conv1d` + `silu` + `l2_norm`
across chained segments, including one shorter than the carried tail.

Measured (9B, BF16×1, within-run, battery), cumulative over the three
rounds:

| span | ops | kernels v1 | strided | conv fold |
|---|---|---|---|---|
| `dn:mix` prefill | 136.6 ms | 84.7 | 61.2 | **53.3** |
| `dn:mix` @4ctx | 689.0 ms | 538.5 | 429.4 | **305.1** |
| prefill total | 226.1 ms | 197.0 | 174.0 | **168.5** |
| `dn:mix` decode (×240) | ~105 ms | ~105 | 45.3 | **20.0** |

The decode mixer is 5.2× the ops form — a decode step is now two launches
(conv+epilogue, decode step) plus the shared norm-gate, ~83 µs per layer
call — and decode layers are now dominated by `dn:proj`/`dn:ffn`, not the
mixer. Prefill `dn:mix` sits at 1.11 ms/layer; the remainder is the conv
pass itself (memory-bound), the scan kernels, and per-span launch overhead —
further cuts are tuning, not structure.

### 7.17 Component-level pass over the scan kernels (2026-08-22)

The tuning, done with instruments rather than guesses. Three of them, all
now permanent: `fused_mixer_kernel_timing` (an `#[ignore]` benchmark in
`models/delta_net/cuda.rs` that splits the mixer into per-kernel µs at the 9B's real
geometry, free of `profile_sync` pollution), `nvcc -Xptxas -v` (registers /
smem / **spills** per kernel), and `ncu` (achieved occupancy, SM and LSU
throughput). What they showed, in order:

1. **Zero spills anywhere** — the intra solve's `xr[64]` holds in registers
   (84 regs/thread), which was the biggest unverified risk. The elementwise
   kernels (conv, decode step, norm_gate) already ran at 100% occupancy.
2. **The state pass was 83% of the scan** (2.37 ms of 2.84 under ncu) at
   14% achieved warp occupancy — the register-distributed S tile forced
   every output through a 32-FMA fragment plus two warp shuffles inside a
   serial t-loop, 4 warps/block, nothing to hide the latency.
3. **S tile → smem** flipped the mapping to warp → t, lane → r: every
   output one long independent dot, no shuffles, 8 warps. SM throughput
   34.5 → 60% — and wall time barely moved, exposing the next ceiling:
4. **The LSU, not the FMA pipe.** Two smem operand reads per FMA. Fix:
   4-way register tiling in every dot (one `s_tile`/`k_i` operand feeds
   4–16 FMAs) — scan 2.03 → 1.35 ms across the state phases and the intra
   A/kq grid.
5. **Residency.** 59 KB smem = 1 block/SM = a 2-wave grid for a
   sequential-per-block kernel. Staging half a chunk at a time (stage
   [32][129]) brings the footprint to ~42 KB — two blocks per SM, all 128
   blocks resident in one wave. Scan 1.35 → 1.15 ms.

Net: the prefill mixer 2,336 → **1,320 µs/layer** at T = 649 (1.77×), on
top of §7.16's three rounds; decode kernels untouched and unchanged. The
8-test parity suite ran green after every step. Remaining known levers, in
descending value: the intra kernel's smem diet (83.5 KB → 1 block/SM; its
sq staging could yield to `__ldg`), `float4` staging loads, and the kq
intra-sum's bank-friendly transpose — all tuning of a structure that is now
measured rather than assumed.

Gate confirmation (9B, BF16×1, within-run): `dn:mix` prefill 53.3 →
**32.5 ms** (×48), @4ctx 305.1 → 238.7, prefill total 168.5 → **156.6 ms**,
decode mixer 20.0 → 17.5 ms — 4/4 exact, 13/13 suite. **Day total on the
mixer: 136.6 → 32.5 ms prefill (4.2×) and ~105 → 17.5 ms decode (6.0×).**

### 7.18 Decode batched across sessions (2026-08-22)

The last per-sequence launches. A decode wave ran the conv step and the
recurrence step once per session — at the 64-session target, 128 launches
plus 128 wrapper resolutions per DeltaNet layer per token. Both kernels now
take the whole wave in one launch each (`grid.y = n_decode`), reading each
sequence's row of the shared wave buffers and its own state through a
**pointer table**: states live in per-session allocations, so their device
addresses must be materialised — `[n_dn_layers, 2, n_decode]` I64 (tails,
states) plus the wave rows.

**The table is ONE host upload per forward**, built in `sweep_layers` before
the layer loop where the launch queue is empty, and sliced per layer
(`DeltaNetWaveTable::layer_slice` — a narrow, no copy). Never per layer: a
host→device copy syncs the stream, and 24 of them per token would serialise
the launch pipeline. Pointer stability across the forward is the store's
standing rule — a sequence's state buffers are allocated once and mutated in
place. The reference path and unit tests, which call the mixer without a
wave table, build a single-layer table at the call site — same kernels, same
layout, the upload merely lands closer to the launch.

Two more things fell out of the same change: the prefill conv now writes
each span's rows **directly into the shared conv buffer** (the per-span
allocations and the concatenation are gone — a multi-sequence wave's conv is
zero-copy into one `[T, conv_dim]`), and the single-sequence
`conv_step`/`decode_step` wrappers no longer exist — one batched form serves
n = 1 identically.

Parity: the decode test now steps TWO sequences through the batched kernels
on interleaved wave rows with separate in-place states, against two
independent sequential references — the pointer scattering itself is what is
pinned. 8/8, first run.

Measured (9B, within-run): 4-ctx decode `dn:mix` 63.8 → **35.1 ms** (×240),
1-ctx 17.5 → 16.3 (a batch of one is the same work, as intended); 4-ctx
decode throughput 135–140 → **158.7 t/s**, the best of any run. The 1→4-ctx
mixer growth collapsed from 3.6× to 2.2×; the remainder is the recurrence's
own state traffic (4 MB read+write per session per layer), which no batching
removes. Gate 4/4 exact, 13/13 suite. From the ops-form baseline the same
morning, the 4-ctx decode mixer is 151.5 → 35.1 ms — 4.3×.

### 7.19 Decode q8 context at head_dim 256 — with the gate folded in

The last head_dim-256 gap in the attention layers. At decode, the paged
combine kernel can emit the attention context directly as q8a1024 blocks
(`want_q8` in `forward_attn_batched`), so `o_proj` runs int8 with no FP
context store and no standalone quantize. That path was gated to head_dim
128, and this family's attention could not have used it anyway: the output
gate (`sigmoid(g) ⊙ ctx`, §3) needs an elementwise landing spot, and a packed
U8 context has none — the guard at the o_proj branch refused gated q8
outright. So at 256 a decode step paid, per attention layer: an FP context
write + read, a sigmoid launch, a multiply launch, and a quantize launch.

All four fold into the combine kernel, which already holds every context
element in a register:

- **Per-tile reduction.** The q8 emit assumed one block = one q8a128 tile
  (128 threads). Generalised to `HEAD_DIM % 128 == 0`: a block covers
  `HEAD_DIM/128` whole tiles (`flat_tile = row·(HEAD_DIM/128) + d/128`), and
  the amax/Σx reduction becomes per-tile — a 32-lane warp butterfly, then a
  pairwise `(w0+w2)+(w1+w3)` combine of the tile's four warp results, which
  is the butterfly's own summation order, so the bytes at 128 are unchanged.
- **The gate rides in as a nullable pointer** (same dtype as Q, flat
  `[slots × n_q_head × head_dim]`, model dim order — the gate is not
  rotary-permuted and neither is the context, V's order). `sigmoid` in F32,
  rounded through O before the multiply: the exact arithmetic of the unfused
  chain (sigmoid materialised in O, O-precision elementwise multiply), so
  fused-vs-unfused differ only in launch count, not numerics.
- **The combine is now guaranteed to run whenever q8 is requested.** The
  emit lives only in the combine kernel and `out` is null on the q8 path, so
  a launch shape that used to direct-write (no stripe, one split) would have
  dereferenced null. `q8_out != nullptr` now forces the partials + combine
  route in the launcher.

`want_q8` widens to `head_dim ∈ {128, 256}` and the gated-refusal guard is
gone — a gated layer passes its gate down `paged_decode_attention →
paged_decode_attn_q8` instead. The FP fallback (Int8Mode::Off, non-paged,
prefill) still applies its gate on the FP context as before.

Parity (`paged_decode_q8_tests.rs`) is raw-byte: a real prefilled slot, one
decode step through the production wrappers, FP context vs q8 bytes. The CPU
reference reproduces the kernel's reduction order operation for operation
(IEEE f32 adds in identical order are bit-identical), so qs and ds bytes are
asserted with no tolerance, at 128 and 256, gated and ungated. The gate's
`expf` — the only transcendental — is rounded to bf16 before use, and the
test asserts every sigmoid sits robustly inside its bf16 rounding interval,
so a ≤2-ulp CUDA-vs-libm `expf` difference cannot move a byte.

### 7.20 The lineage restructure: model files over shared machinery (2026-08-22)

The `qwen35` module carried the whole hybrid implementation — generic
DeltaNet machinery, lineage code, and pinned-checkpoint gates — in one
directory. Restructured to the repo's model-file convention
(`quantized_qwen3.rs` is the template):

**Generic, extracted:**

- `models/delta_net/` — the model-agnostic Gated-DeltaNet subsystem,
  mirroring the kernel family `candle-kernels/src/delta-net/`: `types.rs`
  (`DeltaNetDims`, `LayerKind`), `mix.rs` (the mixer algebra, reference
  scans, span types — the kernels' parity oracle), `cuda.rs` (kernel
  wrappers), `quantized.rs` (`QuantDeltaNetWeights` + the quantized layer
  driver), `state_store.rs` (per-session recurrent state, wave atomicity),
  `kv_layout.rs` (`KvLayerMap`). Nothing in it reads a GGUF name or a model
  config.
- `models/rotary_layout.rs` — the partial-rotary permutation, generic beyond
  DeltaNet (any partial-rotary model on the full-width paged kernels).

**The lineage, kept as `models/qwen35/`** — named by its arch strings: the
GGUFs of Qwen3.5, the 3.6 point release, *and* Qwen3.8 all declare
`general.architecture = "qwen35"` / `"qwen35moe"` (verified on the actual
uploads), the way `llama` covers every Llama generation. It holds the
config parser, GGUF schema, the reference oracle stack, the gated-attention
layer, the shared-expert MoE block, the sweep, and the loaders. The
DeltaNet layer's FFN driver — ten lines dispatching the family's `QuantFfn`
— stays here per §4.3's locality rule.

**Model files, one per checkpoint family, gates inside** (the
`quantized_qwen3.rs` convention — machinery keeps synthetic unit tests,
model files own every pinned-checkpoint gate):

| File | Model | Checkpoint | Status |
|---|---|---|---|
| `quantized_qwen35.rs` | 0.8B / 9B dense | unsloth 0.8B-BF16, 9B-Q6_K (pinned revs) | gates run on dev |
| `quantized_qwen35_moe.rs` | 35B-A3B | unsloth Q4_K_M `bc014a17` | gates run on dev |
| `quantized_qwen36_moe.rs` | 3.6-35B-A3B | unsloth UD-Q4_K_M `a483e9e6` | gate authored; same arch (`qwen35moe`), same geometry — runs on dev when triggered |
| `quantized_qwen38.rs` | 3.8-27B dense | unsloth UD-Q4_K_M `4ca72078` | gate authored; **build-only on the 16 GB card** (§3), runs on the workstation |

Each model file is thin but concrete: pinned constants, a `from_gguf_path`
that loads through the lineage loader (`load_hybrid_gguf`, which returns the
bare `QuantModel`), refuses the wrong checkpoint kind (dense↔routed), and
constructs the scheduler-facing `HybridBatched` with **that model's own KV
threshold factor row** (`QWEN35_0_8B_KV_FACTORS` and siblings — the dense
entry tells its two pinned models apart by `hidden_size`). The 3.6/3.8
entries assert the geometry the engine was audited for at load, so a
silently changed point release fails in the loader rather than in a kernel.

Validation after the move: 77/77 unit tests (identical totals to
pre-restructure, including the 8 CUDA mixer-parity tests now under
`delta_net::cuda`), and the 13 pinned-checkpoint tests re-run green from
their new homes.

### 7.21 Quantized KV at head_dim 256 + the 0.8B compression ladder (2026-08-22)

The C-ladder (adaptive KV compression) reached this lineage. §4.2's plan
assumed an 8-band palette variant would be needed; the audit found the
**read side already validated at the 4×64 band geometry** (decode stripe
path, int8 prefill, size classes, policy — nothing to build), and the write
side 128-locked in four places, now generalized:

1. **`palette4_convert.cuh`** — templated `<HD, IS_K>` (128/256). The
   load-bearing change: palette identity is `d / PAL_DIM`, never the warp id
   (at 256 a palette spans two warps; the warp-cooperative encode stays
   palette-pure because quant blocks are per-dim and `PAL_DIM % 32 == 0`).
   Host side (`candle-core` quantized/cuda.rs) genericized: max-width
   `PalMapBytes`, `kvhead_size(hd)` offsets, `identity_pal_map(head_dim)`.
2. **`select_kv_format.cuh` fused pass** — templated `<HB>`: `u64[HB/64]`
   alive masks, warp-local bitonic over `HB/32` elements per lane, `HB/4`
   per-slot claims, segmented ballot compaction; ~23.7 KB smem at 256 →
   4 blocks/SM. Previously a **silent no-op** at 256 (early-return reporting
   success); the loud bail now lives on the Rust side.
3. The three `compress.rs` bails and `quantize_and_seal_sequences`'s
   `head_dim == 128` gate (the fourth site, in `batched_inference.rs` —
   found because the first real run sealed everything as float and reported
   `%Quantized 0.0`) widened to 128|256.
4. 128-path bit-identity held through the templating: palette4 GPU tests
   30/30, compress byte tests 9/9.

**The wave-plan under-pricing the ladder exposed** (span exhaustion at
C8×10, and 93–98% margins on *every* mode): the gated lineage's attention
chain carried five allocation classes the plan never priced — the gate half
of the QKV projection (`qkv_cols` didn't know `[q|gate]`), the two
partial-rotary permute gathers, the gate split/sigmoid/apply, and
`o_proj`'s float-session round trip. `ModelGeometry` gained `gated_qkv` +
`partial_rotary`; eight `WaveBuffer` variants price them conditionally
(~216 MB at the failing width), pinned byte-for-byte against the census in
`the_projection_round_trip_prices_the_measured_carves`. The census also
corrected the fixture geometry: the 0.8B attends with **8** Q heads — the
old 16-ungated fixture priced the same `qkv_cols` by accident and hid the
gate's whole chain.

**Ladder result (0.8B, identity `QWEN35_0_8B_KV_FACTORS`):** F16/BF16/Q8_0 and
C0–C9 all pass 100% at 100% quantized KV — compression 1.94× (C0) to 4.64×
(C9), Q8_0 at 1.88× — and **C10 just fails** (4/5 sessions; one
single-name-token divergence at 5.14×). That is the designed calibration
("C10 only just fails"), so identity is the derived factor row, C9 is the
gate's top pass-required rung, and a C10 row turning green after a factor
change would mean the thresholds went loose. The gate lives in
`quantized_qwen35.rs` alongside the float rows. **(Superseded by §7.22:**
the calibration target was later changed to "C0–C10 all pass, with C10 just
under the breaking edge" — the just-fails scheme and its edge probes are
retired.)

### 7.22 Concrete model entries, per-model factor rows, and the VRAM-governor fix (2026-08-23)

**`HybridBatched` + concrete entries.** `Qwen35Batched` was renamed to
`HybridBatched` — it serves the whole lineage, and a struct named for one
model being returned by four model files misread as model-specific. The
lineage loader (`load_hybrid_gguf`) now returns the bare `QuantModel`; each
model file's `from_gguf_path` is the concrete constructor, doing the
dense↔routed refusal and building `HybridBatched` with **that model's own
`KvErrorThresholdFactors` row** (the dense entry distinguishes its two pins
by `hidden_size`: 1024 → 0.8B, 4096 → 9B). The rename also surfaced a real
wiring bug: `HybridBatched::create_batched_session` overrides the
`ManagedBatchedModel` default (the KV layer count differs on a hybrid) but
did not replicate the default's factor fold into `BatchedConfig`, so the
lineage's factor row had **never reached the compression policy** —
invisible while the row was identity, caught the first time a non-identity
value changed nothing.

**The calibration target: the complete range passes.** Every gate runs the
full ladder C0–C10 as pass-required rungs, and the per-model factor rows
are tuned so **C0–C10 all pass with the C10×10 rung sitting just under the
breaking edge** — the range is complete, and C10's ten contexts make its
edge a measurement rather than a coin toss. A red C10 row means the
thresholds drifted past the edge: retighten the factor row, never widen
tolerances. (An earlier iteration of this work calibrated to "C9 passes,
C10 just *fails*" with dedicated `c10_edge_*` must-fail probes; that scheme
is retired — the probes are deleted and C10 is an ordinary pass-required
rung.) One protocol tool worth keeping from the derivation: wide rungs draw
the hard tail of the harness name list ("ChristopherJames" is real), so
when a wide C-rung fails, run a BF16 row at the same width to attribute
between compression error and a name the model cannot reproduce at any
fidelity — and note that the first wide config of a process pays the
wide-wave establishment cost, so only steady rows compare.

**Derived rows** (all in `sampled_selection/params.rs` with per-row
derivation notes; hi and low swept separately per model, retuned
2026-08-23 to the complete-range target):

| Model | k_hi/k_low | v_hi/v_low | status |
|---|---|---|---|
| 0.8B | 0.85 / 0.85 | 0.6 / 0.6 | derived (C0–C10 green) |
| 9B | 1.1 / 1.1 | 1.9 / 1.9 | derived (C0–C10 green) |
| 3.5-35B | 1.5 / 1.5 | 2.3 / 2.3 | derived (C0–C10 green) |
| 3.6-35B | 1.5 / 1.5 | 2.3 / 2.3 | derived (C0–C10 green) |
| 3.8-27B | 1.5 / 1.5 | 2.0 / 2.0 | extrapolated (between the 9B and the MoE pair; workstation derives) |

Sweep findings worth keeping: the critical blocks respond to the
**geometric mean** of an axis's hi·lo pair (single-sided probes barely move
them — they live in the scaled middle band, not at a clamp); K-tightening
alone makes the 0.8B *worse* (a second session diverges); and at the top of
the ladder the C9→C10 separation is narrower than proportional K+V
stepping, so the C10 edge is steered with the **V-differential** — C10's V
candidate floor (Q0/Q1) is strictly worse than C9's (which keeps Q4
fallbacks), so V movement shifts C10 while C9 holds.

**The VRAM-governor fix.** The lineage loader never called
`ensure_vram_governor`, so `span_target` fell back to the 3,024 MiB
governor-less test constant — the entire qwen35 lineage had been running
its KV span *and* expert zone inside 3 GiB of a 16 GB card (expert zone
hard-capped at 529 slots / 1.0 GiB, VRAM hit rate 6–17%, 30–40% of expert
loads served from the NVMe pack). One call in `load_hybrid_gguf` (the same
call every other loader makes) widens the span to ~11.3 GiB: expert zone
opens at 3,193 slots (limit 4,998), hit rate 33–40%, C8×20 cold loads
73,792 → 20,053, single-session decode roughly doubles and C-rung prefill
gains ~55%. Derivation caution the fix taught: widened spans change wave
widths and therefore accumulation order, which moves the marginal
calibration sessions — **re-derive the factor rows after any admission or
width change** (the rows above post-date the fix).

**Against Qwen3-30B-A3B (same night, same card):** the structural remainder
is host RAM, not code: the 30B's 17.8 GB pack gets 85% warm coverage
(15.2 GB pinned) on the 32 GB box, while the 3.6's 19.8 GB pack gets
43–57% (the warm tier is sized from *available* RAM at load). Post-fix the
3.6 reaches ~590 t/s prefill / ~68 t/s aggregate decode at ×4 (per-session
decode parity with the 30B's ×10 row), against the 30B's 2,896 / 210.9 at
×20. The apparent "C8×20 prefill cap at 209 t/s" resolved under the float
control: the ~30 s fixed cost belongs to the **first wide config of a run**
(wide-wave arena and warm establishment), and with BF16×20 preceding it
C8×20 prefills at ~590–600 t/s — 1.7× the pre-fix rate. What remains open
(§8) is **decode at width**: aggregate decode at ×20 is mode-independent
(BF16×20 ≈ C8×20) but per-session decode at width ≥5 sits well below the
×4 rate, and the profile spans put the cost in the DeltaNet mixer path
(`dn:mix` at C9×5 runs ~89× the per-call cost of ×1).

---

## 8. Risks and open questions

1. **DeltaNet state × substrate forks** is the only genuinely novel
   persistence problem: checkpoint cadence (per sealed turn) vs storage cost,
   and glue/reprojection interplay (a reprojected context changes attention
   layers' KV but the recurrent state was computed over the *original*
   stream — forks from re-projected timelines need a defined answer:
   recompute states through the projected prefix at fork time).
2. **Chunked delta-rule numerics**: FP32 state accumulation is mandatory;
   the C-ladder quality gates must cover long-context drift, not just 40
   tokens.
3. **Prefill-256 smem budget** may force a meaningfully different tiling
   (the 25.6 KB union-arena limit was already binding at 128).
4. **GGUF churn**: the families are weeks old; tensor names/metadata may
   shift between llama.cpp releases — pin revisions everywhere.
5. **Provenance retrieval coverage** drops to attention layers only; whether
   10-layer Q capture retains current retrieval quality needs an early
   measurement (Phase 4 exit criterion).
6. Qwen3.8-27B being dense makes it a poor fit for the 16 GB card despite
   its recency; it is deliberately not the first target.
7. **The 3.6 ↔ Qwen3-30B throughput gap** (~4.5× wide prefill, unstable
   wide decode) has a dedicated ordered work list with the measured
   evidence per item: `docs/qwen36_performance_plan.md` (T1 fused-mixer
   dispatch cliff at width, T2 wide prefill, T3 first-wide-config ~30 s
   establishment, T4 frequency-aware warm fill, T5 expert-zone growth,
   T6 prefetch depth, T7 single-session decode).

## 9. References

- In-repo research (2026-08-18): Qwen3 contract map and DeepSeek-V4 map —
  file:line citations inline above.
- [Qwen3.5 family overview](https://www.morphllm.com/qwen-3-5) ·
  [QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8) ·
  [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ·
  [35B-A3B architecture overview](https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview)
- GGUFs: [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) ·
  [unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) ·
  [unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF) ·
  [Qwen3.8-27B GGUF guide](https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/)
- [122B-A10B specs](https://apxml.com/models/qwen35-122b-a10b) ·
  [Qwen3.8-27B analysis](https://local-ai-zone.github.io/blog/qwen3-8-27b-comprehensive-analysis.html) ·
  [Qwen3-Next local guide (DeltaNet)](https://unsloth.ai/docs/models/tutorials/qwen3-next) ·
  [DeltaNet GPU offload compat discussion](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF/discussions/6)
