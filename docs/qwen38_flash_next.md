# Qwen3.8-Flash-Next — bring-up principles and integration surface

Branch: `qwen38-moe`. Released **2026-08-26** (open weights, Qwen Community
License 1.0). This is the model the engine moves to as its primary target.

§0 is binding: it states the decisions this bring-up is built on. §1–§9 are the
research the decisions rest on — what the model is, what already exists here,
and what does not. §10–§11 are the plan: phases with exit conditions, and the
gate that judges them. §12 is the **Phase 0 record**: the schema frozen from
the actual checkpoint and the llama.cpp implementation, and the corrections it
forced on the sections above it — read a section's correction there before
trusting the section.

It is *not* the Qwen3.8 covered by `docs/qwen35_qwen38_models.md`. That document
covers the Qwen3.8 point release of 2026-08-12/14 — a 27B dense member of the
Qwen3.5 hybrid lineage (arch `qwen35`). Qwen3.8-Flash-Next is a different
architecture generation: HF `model_type` **`qwen4_exp`**, GGUF arch
**`QWEN4EXP`**, described by Qwen as a preview of Qwen4. The two share a
lineage, and §5 below is a per-component account of exactly how much of the
Qwen3.5 bring-up carries over — the answer is "most of the hard parts".

Throughout: **GDN** = Gated DeltaNet, **QSA** = Qwen Sparse Attention,
**GR** = Gated Residual, **PLE** = Per-Layer Embedding, **MTP** = Multi-Token
Prediction. Several secondary write-ups expand these wrongly (one renders QSA as
"Quantum-Skip Attention", another MTP as "Multi-Tower Preference"); the
expansions here are the ones used by the HF model card and the transformers
docs.

---

## 0. Key principles

These are decisions, not options. Everything below is written to serve them, and
a change to any of them is a change to the design rather than an implementation
detail.

### 0.1 The embedding tier lives on NVMe, and its lookups are CPU-side

The 51B PLE table (~95 GiB at BF16) is never resident and is never paged into
VRAM. It lives on NVMe as the cold tier, fronted by a **non-pinned RAM cache
capped at 2 GB**, and every lookup happens **on the CPU**. Only the gathered
result — 16 rows × 160 values = 5 KB per token — crosses PCIe.

This is a **new module**, because nothing existing has this shape.
`models/host_embedding.rs` is a host-resident *table* with a CPU `index_select`;
this is a disk-resident table with a bounded cache in front of it, an n-gram
hash on the key side, and a per-session convolution state on the output side.
The token-embedding table (248,320 × 2,560 ≈ 1.27 GB at BF16) is small enough to
stay exactly where it is, in `host_embedding.rs` — do not fold it into the new
module for symmetry's sake.

The cost model that makes this work: a token needs 16 rows, and each head reads
only its own 160-wide slice, so the demand is 16 random reads of ~320 B — 4 KB
after page granularity, ~64 KB/token uncached. At 50 tok/s that is ~800 IOPS,
which is nothing for NVMe. **This tier is IOPS-shaped, not bandwidth-shaped**,
and the 2 GB cache is sized against n-gram frequency skew (common bigrams
dominate), not against the table. Hit rate is the number to measure first;
bandwidth is not at risk.

The upload is a **sanctioned exception to hot-path invariant 3**, of the same
kind as the existing embedding lookup — a pure index-plus-transfer that keeps a
table off VRAM — and it is the *only* new exception this model introduces.
Non-pinned is deliberate: pinning 2 GB competes with the expert cache's warm
tier for the same pool, and this transfer is small enough that pageable is fine.

### 0.2 Expert quantization is chosen per machine

The trunk is ~125B parameters of which **~121B are experts**
(512 experts × 3 matrices × 2560 × 640 × 48 layers ≈ 120.8B). Everything else —
attention, GDN, norms, router, LM head — is ~4B. So expert format *is* the
memory decision, and it is per machine:

| Machine | VRAM | Expert format |
|---|---|---|
| RTX PRO 5000 Blackwell | 72 GB | **Q4_K** |
| RTX 3090 | 24 GB | **Q3_K** |
| RTX 4090 Mobile | 16 GB | **Q2_K** |

**This reinstates a ladder that was deliberately removed, and the removal is
documented.** `zend/src/model_choice.rs` currently says, in its module docs:

> There is no VRAM ladder here. The lineage ships one quant (`UD-Q4_K_M`), and
> a MoE model's resident footprint is its dense weights plus whatever expert
> working set fits — the three-tier expert cache pages the rest — so parameter
> count does not decide whether a card can run it.

That reasoning is about *feasibility*, and it still holds — every machine can
run this model at any of the three formats. The ladder here is about *quality
per resident byte*: at 512 experts the residency spread is 28% to 88% (below),
which is a large enough difference in how often a token's experts are already
resident that the format is worth choosing per machine rather than fixed.

The mechanism already exists and needs no invention: the `Model` enum in
`candle-conversation/src/models/mod.rs` carries one variant per quant, and
`Qwen3_30B_A3B_Q4`/`_Q6` are already documented as "zend selects Q4 vs Q6 by
measured VRAM". So this is three `Model` variants plus a VRAM-keyed `model()`.
**The `model_choice.rs` module docs must be rewritten in the same change** — a
module whose documentation denies the ladder it implements is worse than either
alternative.

**The ladder also decides whether routing leaves the card.** Measured in Phase
3: the GPU-native MoE dispatch captures raw slot addresses and is therefore
built only when the expert grid is *fully* resident. At Q4_K on the 72 GB card
the grid is ~68 GB against a 53.6 GB zone — ~80% resident — so every layer
reads its routing back to the host to schedule that layer's uploads. At ~3.4
bpw the grid is ~51 GB and fits. So the format choice is not only quality per
resident byte; below the residency cliff it also buys the device dispatch path
and costs the expert precision. Neither side of that trade has been measured on
this model yet, and it should be, before the ladder's 72 GB row is treated as
settled.

Rough residency at those formats (**estimates to be replaced by measurement**,
using ~4.9M params/expert and 24,576 total expert slots):

| Machine | Per-expert | Expert budget | Resident slots | Coverage |
|---|---|---|---|---|
| 72 GB @ Q4_K | ~2.8 MB | ~60 GB | ~21,700 | ~88% |
| 24 GB @ Q3_K | ~2.1 MB | ~18 GB | ~8,600 | ~35% |
| 16 GB @ Q2_K | ~1.6 MB | ~11 GB | ~6,900 | ~28% |

The dense trunk is ~2.1 GB at Q4_K, and less than that on the device — the
token-embedding table (~0.64B of it) is host-resident per §0.1. So on every
machine VRAM is almost entirely expert working set. That is precisely what
`expert_lre/` was built for, and it is why a 180B model is a reasonable target
on a 16 GB card.

**The per-rung artifact is a prepared hybrid, identified by its recipe.** No
repository publishes an engine file at any rung, so each machine builds one
(`qwen4exp::prepare`): the pinned `Q8_0` split's trunk and n-gram (PLE) table
verbatim at `Q8_0`, the MTP head's dense weights at `Q8_0`, and the routed
experts at the rung `quant_ladder::expert_format` names — `Q4_KO` by the
bit-exact W4A16 import, `Q3_KO`/`Q2_KO` requantized from the `Q8_0` experts on
the GPU in one pass (no intermediate `Q*_K`), and the draft head's experts at
the trunk's width (`quant_ladder::drafter_format`). The recipe — every source
file's repo, revision, path, length and LFS SHA-256, plus each of those
choices and a converter version — hashes to the artifact's identity: stamped
into its GGUF metadata (`zen.prepare.recipe`, with the canonical text beside it)
and tagged into its filename. A present artifact with the matching stamp is
used without touching any source; anything else is built — sources fetched and
checked against their pins, converted, merged, its header read back and checked
(stamp, every block's three expert tensors at the recipe's width, the head's
block, the directory within the file) — and the sources are then deleted, on
every resolve that finds them, because the artifact carries
everything the engine reads. A new quant level, pin or converter version is a
new recipe, so every machine rebuilds rather than loading stale bytes. The
expert pack is kept beside the artifact and keyed on its identity, so it
follows.

### 0.3 Every hot-path invariant holds, without exception

`docs/deepseek/deepseek_hot_path_invariants.md` is binding on every line of this
bring-up. Specifically, and non-negotiably:

- **No `to_dtype` in the loop.** Kernels emit the type the next consumer wants.
  Where two types are supposed to agree, `operand_guard::expect_dtype` asserts
  it — a validation, never a conversion.
- **No allocate-plus-copy to materialise a layout**, by any spelling:
  `contiguous`, `force_contiguous`, `Tensor::cat`, `slice_set`. Per-row and
  per-session kernels take a **descriptor table** (`arena_table.cuh`) and read in
  place; where data must be rearranged, that is a **gather/scatter kernel**, not
  a copy to satisfy a consumer's preferred layout.
- **Everything runs GPU-side** — routing, selection, index expansion, remaps —
  **except the embedding lookups of §0.1**, which are the sanctioned exception.
- Fully batched: one launch over all slots, prefill and decode alike.
- Never zero memory a kernel will overwrite (`alloc_uninit`, not `zeros`).
- Span partition boundaries re-checked in both directions by every tenant.

The 512-expert question in §6.1 must be answered *inside* these rules: if the
histogram bucketize cannot be widened, the answer is a different GPU-side
dispatch, not a host-side sort.

### 0.4 No composed operations — one fused, batched, paged kernel

An operation on the hot path is **never** expressed as a chain of eager tensor
ops. Every link in such a chain is a kernel launch and a full-tensor memory pass,
and the chain's cost is invisible at the call site because each line reads like
arithmetic rather than like traffic. `indexer_score.cu` exists precisely because
the expression it replaces was *eight* launches, and eight was a floor rather
than an estimate.

The rule, in order:

1. **Check whether the kernel already exists.** `candle-kernels/src/simple/` has
   roughly sixty, plus the `paged-decode/`, `paged-prefill/`, `paged-latent/`,
   `delta-net/` and `quantized/` families. §5 is the inventory for this model;
   extend it rather than trusting it, because the cheapest kernel is the one
   already written and already profiled.
2. **Reuse and extend it. Writing a new one is the exception**, and there are
   exactly two grounds for it: the existing kernel genuinely cannot express the
   operation, or extending it would **regress its existing callers**. Nothing
   else qualifies — not "it would be cleaner", not "our case is special".
   *Regress* is a measured claim, and the instrument is the kernel's own harness
   (rule 4): run it on the existing geometry before and after the extension.
   The mechanism matters as much as the rule — extend by **template parameter**,
   which costs existing instantiations nothing, rather than by runtime branch,
   which puts our predicate in everyone's inner loop. The kernels are already
   templated this way (`<HEAD_DIM, WARPS_PER_BLOCK, STAGES, …>`), so this is the
   grain of the code, not a concession to it. If a fork is unavoidable, say in a
   comment which caller each variant serves, so the next reader does not have to
   diff them to find out.
3. **If it does not exist, write it fused, batched, and paged.** Fused: the
   whole expression in one launch. Batched: one launch over all sessions and all
   rows, never a per-sequence loop (invariant 5). Paged: it takes a **descriptor
   table** of `{ptr, offset, stride, len}` and reads rows in place
   (`arena_table.cuh`), never a dense base pointer that forces the caller to
   `cat`/`slice_set` rows together — that copy is the kernel's API bug, not the
   caller's (invariant 2b).
4. **Every kernel gets a performance harness**, built with it, not after — and
   this applies to kernels we **reuse**, not only ones we write. A kernel that is
   fast at the geometry it was tuned for is an unknown at a new one; §6.2 is
   exactly that case, an existing decode path whose behaviour at 24/2 @ 256 is
   not established by its behaviour anywhere else. Reusing a kernel at a new
   geometry means extending its harness to cover that geometry.
   The precedent is `candle-transformers/examples/{latent_decode,latent_prefill,
   corpus_gather,select_kernel,prov_sign_pack}_bench.rs`: runs in seconds, and
   carries a **per-run correctness gate** so a fast wrong kernel cannot pass.
   That property is load-bearing — correctness gates are not performance gates
   and the converse also holds; a decode-fusion first cut once halved throughput
   with every unit gate green.
5. **Iterate against the profiler, not against intuition.** The targets are
   occupancy, full warps (no idle warps, no divergent tails), vectorised loads
   and math (`float4`/`int4`, the `VEC` pattern the decode kernels use), and low
   register pressure — spills cost occupancy, which costs more than the
   arithmetic they were spilled to save. Shared-memory budget is part of the same
   trade: §6.2 is a live example of a two-stage pipeline that does not fit
   48 KiB at head_dim 256.
6. **Profile with `ncu`** (Nsight Compute) for the kernel and **`nsys`** for the
   timeline. Both are present on this machine. NVTX spans are already wired
   (`candle-kernels/src/simple/nvtx.cu`, `models/profile/`), so
   `nsys stats --report nvtx_kern_sum` self-attributes — use it rather than
   hand-attributing launches to phases.

Two failure modes to design against, both of which have already cost time here:

- **"Memory-bound" is a measurement, not a description.** A decode fusion was
  optimised on the assumption it was memory-bound; `ncu` reported FP64 at 84.7%.
  Nobody had looked.
- **Size the benchmark past the L2 or you are measuring cache.** The Blackwell
  card has 96 MiB of it. A microbench that fits is a microbench that lies.

A kernel that has not been measured is not finished, and a kernel whose harness
does not exist cannot be measured.

### 0.5 Reuse before building

§0.4 rule 2 states this for kernels, where the extend-vs-fork decision has a
measurable answer. It holds for Rust modules too, with the same two exceptions —
*cannot express it*, or *extending it would regress its existing users* — but
without the harness to arbitrate, so the judgement is the reviewer's.

§5 is the inventory of what already exists, and it is long. The default is to
use it. A new file is justified only when nothing existing expresses the thing —
which in this bring-up is true exactly twice: the NVMe embedding tier (§0.1) and
whatever QSA's block-selection shape turns out to need beyond the existing
two-stage selector (§8, item 2).

The counter-rule from §4.3 of the sibling doc still binds: *sharing that removes
duplication is worth it; sharing that merely relocates it is not.* A trait
reshaped across four production models to save ten lines was built and reverted
once already. Do not repeat it.

### 0.6 `forward_batched` is the oracle

Build the batched reference forward **first**: whole prompt in, plain tensor ops,
no paging, no wave engine, no int8. It is the artifact that

- establishes parity against llama.cpp on real weights, using the
  layer-truncation and tensor-zeroing bisect method of §7.3 of the sibling doc;
- every optimized path (paged, int8, wave-batched, speculative) is diffed
  against, for the life of the model;
- makes a defect *locatable* — the sibling doc's §7.2 records a wrong model that
  passed every structural check, and only a numerical oracle found it.

It is batched from the start, not a per-sequence reference batched later:
principle 0.3 says the production path is fully batched, so an unbatched oracle
would not be comparable to the thing it is supposed to validate.

**One mixer core, two projection paths** (§7.4 of the sibling doc): the oracle
and the production path share the mixer algebra and differ only in how they
project (`Tensor::matmul` vs `QMatMul`). Transcribing the algebra twice is how a
load-bearing epsilon gets lost — that has already happened once in this lineage.

### 0.7 Test-driven, alongside the code

Per CLAUDE.md: tests are built as the code is written, not after. Every building
block is testable in isolation — the n-gram hash, the cache eviction, the NVMe
gather, the QSA block selection, the GR gates, the PLE conv state, the delta
rule. Codec- and layout-level tests assert **raw expected bytes**, never error
tolerances. The segmentation property that the Qwen3.5 work proved at
recurrence/conv/layer/model level (segments-from-carried-state ≡ one-shot) is
re-proven here, because this model has *three* carried states rather than one
(§6.3).

### 0.8 Keep it simple

No speculative generality, no configuration surface for cases that do not exist,
no dual paths. Per CLAUDE.md: no backward compatibility, no environment-variable
feature flags, no `TODO`s, no stubs. When a new path is correct it replaces the
old one; when it is not correct, it does not land.

### 0.9 Fork `qwen35/` as the base

`candle-transformers/src/models/qwen35/` is the base. It already is a hybrid
3:1 Gated-DeltaNet stack at head_dim 256 with partial rotary, a sigmoid output
gate, a shared expert, an MTP head, a wave engine, and a loader over the same
tokenizer generation. That is the *skeleton* of this model — 36 of its 48 layers
are GDN, and the schedule, the state store, and the turn-seal snapshot are the
large, hard, already-solved parts.

It is also already the *family* module rather than one model's: `ModelArch::
Qwen35Hybrid` is what zend runs Qwen3.6-35B-A3B through today. Adding a third
member is the pattern it was shaped for, not a new use of it.

What it lacks is lifted from `latent_moe/`, which has each piece working in
production:

| Need | Lift from |
|---|---|
| QSA indexer + selection | `latent_moe/indexer.rs`, `candle-kernels/src/simple/indexer_score.cu`, `two_stage_select_batched` |
| Gated Residual (hyper-connection) | **corrected by §12.3** — qwen4exp's HC is a low-rank sigmoid-gated mixer with *no Sinkhorn*; `latent_moe/hyper.rs`'s mHC (Sinkhorn-normalised combine) is a different algebra and is *not* the lift. The kernel discipline of `hyper_mhc.cu` (fused pre/post with CPU parity references) is the pattern to mirror; the arithmetic is qwen4exp's own. |
| Multi-split GGUF loading | `latent_moe/loader.rs::SplitGguf` — `qwen35/quantized_loader.rs` reads a single `Content::read`, and these checkpoints ship split (Q8_0 in 6 shards, metadata isolated in shard 1, the PLE table isolated in shard 3) |

The alternative — forking `latent_moe/` and adding DeltaNet — was considered and
rejected: it would mean porting the hybrid schedule, the recurrent state store,
the wave-atomic begin/commit/rollback, and the snapshot record into a family
built around a uniform latent-attention stack. That is the larger half of the
work, and it is the half `qwen35/` already has.

---

## 1. Links

### Model and weights

| What | Where |
|---|---|
| **BF16 weights** (canonical) | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| **FP8 weights** (official) | https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 |
| Official repo (README + tech report PDF, **no code**) | https://github.com/QwenLM/Qwen3.8-Flash-Next |
| Announcement blog | https://qwen.ai/blog?id=qwen3.8-flash-next |
| ModelScope mirror | https://www.modelscope.cn/organization/Qwen |

### Reference implementations

| What | Where | Status |
|---|---|---|
| **HF transformers** — `Qwen4ExpForConditionalGeneration` | https://huggingface.co/docs/transformers/main/en/model_doc/qwen4_exp | Contributed 2026-08-26; `transformers_version: 5.8.0.dev0` in the checkpoint |
| **llama.cpp** — `MODEL_ARCH.QWEN4EXP` | https://github.com/ggml-org/llama.cpp/pull/27742 | **Merged 2026-08-27** (danielhanchen) |
| llama.cpp — first attempt | https://github.com/ggml-org/llama.cpp/pull/27739 | Closed unmerged (JJJYmmm), deferred to #27742 |
| **SGLang** day-0 support + kernel writeup | https://www.lmsys.org/blog/2026-08-26-qwen-flash-next | Model-support PR, not in a tagged release (`qwen4-main @ e17062a1d`) |
| SGLang cookbook (launch flags) | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | — |

The LMSYS post is the most implementation-dense source available and is the
reference for §3.1 (QSA), §3.3 (PLE offload) and §3.4 (MTP IndexShare). The
GitHub repo carries only a README and `tech_report.pdf` — there is no official
reference implementation to read, so **llama.cpp PR #27742 is our primary
oracle**, exactly as `qwen35.cpp`/`delta-net-base.cpp` were for the Qwen3.5
bring-up (`docs/qwen35_qwen38_models.md` §7.1).

---

## 2. The checkpoint, exactly

From `Qwen/Qwen3.8-Flash-Next/config.json`. Reproduced rather than summarised,
because §7.2 of the Qwen3.5 doc is a record of what summarising a geometry
costs.

```
architectures:            ["Qwen4ExpForConditionalGeneration"]
model_type:               qwen4_exp          (text: qwen4_exp_text)
language_model_only:      false
transformers_version:     5.8.0.dev0
```

**Trunk**

| Key | Value |
|---|---|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 48 |
| `full_attention_interval` | 4 |
| `layer_types` | 48 entries: `[linear_attention ×3, full_attention] ×12` |
| `vocab_size` | 248320 |
| `rms_norm_eps` | 1e-06 |
| `tie_word_embeddings` | false |
| `dtype` | bfloat16 |
| `max_position_embeddings` | 262144 |

**Full-attention layers (QSA), 12 of 48**

| Key | Value |
|---|---|
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 → **64 of 256 dims rotate** |
| `rope_parameters.rope_theta` | 10000000 |
| `rope_parameters.mrope_section` | `[11, 11, 10]` (sums to 32 pairs = 64/2) |
| `rope_parameters.mrope_interleaved` | true |
| `output_gate_type` | sigmoid |
| `indexer_n_heads` / `indexer_kv_heads` | 4 / 1 |
| `indexer_head_dim` | 128 |
| `indexer_budget` | 2048 |
| `indexer_compress_ratio` | 4 |

**Linear-attention layers (Gated DeltaNet), 36 of 48**

| Key | Value |
|---|---|
| `linear_num_value_heads` | 48 |
| `linear_num_key_heads` | 16 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `linear_conv_kernel_dim` | 4 |
| `mamba_ssm_dtype` | float32 |

**MoE (every layer)**

| Key | Value |
|---|---|
| `num_experts` | **512** |
| `num_experts_per_tok` | **10** routed |
| `moe_intermediate_size` | 640 |
| `shared_expert_intermediate_size` | 640 (1 shared expert) |
| `router_aux_loss_coef` | 0.001 |

**Gated Residual (hyper-connection)**

| Key | Value |
|---|---|
| `hc_count` | 4 residual streams |
| `hc_lowrank` | 320 |

**PLE (n-gram per-layer embedding)**

| Key | Value |
|---|---|
| `ple_layer_ids` | `[2]` — **one-based**, so decoder index 1 |
| `ple_embed_dim` | 2560 |
| `ngram_size` | 3 |
| `heads_per_ngram` | 8 |
| `ngram_vocab_size_base` | 20000000 |
| `split_ngram_parts` | 128 |
| `ple_conv_kernel_size` | 4 |
| `make_ngram_vocab_size_divisible_by` | 128 |

**MTP**

| Key | Value |
|---|---|
| `mtp.num_hidden_layers` | 1, `layer_types: [full_attention]`, `hybrid: true` |
| `mtp.rope_theta` | 10000000 |
| `mtp_use_dedicated_embeddings` | false |

**Vision encoder**

| Key | Value |
|---|---|
| `depth` | 27 |
| `hidden_size` / `out_hidden_size` | 1152 / 2560 |
| `num_heads` | 16 |
| `intermediate_size` | 4304 |
| `patch_size` / `spatial_merge_size` / `temporal_patch_size` | 16 / 2 / 2 |
| `num_position_embeddings` | 2304 |
| `deepstack_visual_indexes` | `[]` |
| token ids | image 248056, video 248057, vision start/end 248053/248054 |

### 2.1 Parameter accounting

The published "125B + 51B + 4B = 180B, 6B active" resolves as:

- **125B** trunk (48 layers × (GDN or QSA) + 512-expert MoE + GR).
- **51B** PLE table: `20,000,000 rows × 2,560 = 51.2B`. The parameter figure and
  the row count are the same object counted two ways — worth stating, because
  the model card lists the row count under a heading that reads like a
  parameter count.
- **4B** MTP head.
- **6B** active: 10 routed + 1 shared expert at intermediate 640, plus the
  attention/GDN mixers, plus the 16 PLE rows a token actually reads.

Disk: **BF16 335.28 GiB**, **FP8 172.78 GiB** (both per the vendor's deployment
notes). The PLE table dominates: `20e6 × 2560 × 2 B = 102.4 GB = 95.4 GiB` of
the 335.

**A 2.4% discrepancy here is unexplained and worth resolving**, because it bears
on §6.3's layout question: llama.cpp #27742 reports the PLE tensor group at
**97.7 GiB**, not 95.4. Vocabulary padding does not account for it —
`make_ngram_vocab_size_divisible_by: 128` and 20,000,000 / 128 = 156,250 exactly,
so there is nothing to pad. Either the table carries rows the config's
`ngram_vocab_size_base` does not describe, or the group includes the PLE
projection and convolution weights as well. Read the converter.

---

## 3. The four new components

### 3.1 QSA — Qwen Sparse Attention

The 12 full-attention layers do not attend over the whole prefix. A lightweight
indexer scores *compressed blocks* and the attention reads only the winners.

- **Compression** is 4:1 (`indexer_compress_ratio`). Four raw index keys are
  averaged in FP32, normalised, and rotated with the *first* token's MRoPE
  position to form one compressed key. So the indexer scans ~L/4 small keys
  rather than L.
- **Scoring** is MQA: 4 query heads of dim 128 against 1 shared key head,
  `s(t,b) = (1/√128) · Σ_h ReLU(⟨q_th, k̄_b⟩)`. The ReLU-then-sum over heads is
  the same expression `indexer_score.cu` already fuses for the DeepSeek CSA path
  (see §5).
- **Selection** keeps the best **512 blocks**, expands them to 2048 logical
  positions (`indexer_budget`), and appends the 0–3 tokens of the current
  incomplete block — at most **2051 positions** attended regardless of context
  length. That is the whole long-context story: KV *read* per QSA layer is
  O(1) in depth, and only the index scan is O(L/4).
- **Cache cost**: one BF16 compressed index key per four tokens, with the raw
  keys of the unfinished block in a four-slot per-request ring. LMSYS reports
  that as an 80% reduction in index-side cache versus keeping raw keys.
- **Kernels** (SGLang): prefill runs a custom score kernel → fast top-k →
  Triton index expansion → sparse GQA. Decode runs a paged scorer, compacts the
  selected K/V, and dispatches to TRTLLM-Gen on Blackwell or packed
  FlashAttention elsewhere.

This is architecturally the same idea as DeepSeek-V4-Flash's compressed-select
attention, which the engine already runs in production.

### 3.2 GR — Gated Residual

The residual stream is widened to **4 parallel branches** (`hc_count`), with an
element-wise read gate and a per-branch scalar write gate at bottleneck rank
**320** (`hc_lowrank`), applied before each attention and each MoE block. HF
describes it as Hyper-Connection combined with GatedNorm.

**There is no Sinkhorn.** An earlier draft of this section claimed the combine
matrix was Sinkhorn-normalised, from secondary write-ups; the merged llama.cpp
implementation and the actual GGUF metadata (§12.1) refute it — the only HC
keys are `hyper_connection.count` and `hyper_connection.low_rank`, and the
algebra (§12.3) is a grouped RMSNorm, a low-rank silu/sigmoid gate, a mean
collapse, and `2·sigmoid` scatter weights. Nothing iterates.

### 3.3 PLE — Per-Layer Embedding

At decoder layer index 1 (`ple_layer_ids: [2]`, one-based) the hidden state is
enriched with hashed n-gram features:

- 8 bigram hash heads over `(x_{t-1}, x_t)` and 8 trigram hash heads over
  `(x_{t-2}, x_{t-1}, x_t)` → **16 row ids** per token.
- Each head contributes **160 values**; concatenated head-major,
  `16 × 160 = 2560` = `ple_embed_dim`.
- **The gathered vector is not added to the stream directly** — see §12.4 for
  the actual block. It is projected two ways (`ple_key` → the 4-stream width,
  `ple_value` → hidden), the key is dotted per-stream against the normed
  residual to form a signed-sqrt sigmoid gate, and the gated value plus a
  **dilated** depthwise convolution of it (kernel 4, dilation = `ngram_size`
  3 → **9 tokens of carried history**) are added to all four residual streams.
  That convolution is the per-session PLE state of §6.3.
- The hash is `mixed_n = (t·m₀) ⊕ (t₋₁·m₁) ⊕ …` with the checkpoint's
  `ple.layer_multipliers`, an EOS in the window resetting everything at or
  before it; `row = mixed_n mod vocab[h] + offset[h]`, with **per-head** vocab
  sizes and offsets from metadata (§12.1) — the 16 head regions are stacked in
  one table and are *not* uniformly 20M rows.

**Offload is the intended deployment, not a fallback.** SGLang keeps each rank's
vocabulary-parallel shard in **pinned host memory** and gathers the 16 selected
rows into a small BF16 GPU buffer with a Triton UVA kernel, on a dedicated CUDA
stream overlapping the first decoder block. Measured on H200 TP4: target-model
weights down **23.46 GiB**, KV capacity up **78.54%**, throughput **−0.07%**.
llama.cpp exposes the same thing as `-ot "ple_ngram_embd=CPU"`.

### 3.4 MTP and speculative decode

A 1-layer full-attention MTP head (4B). SGLang's **IndexShare** optimisation is
worth recording: the draft loop does not run the QSA indexer at all — the
target's verify pass captures each request's last accepted index row and the
whole draft loop reuses it. Reported: **540 tok/s at batch 1**, TP4 on B200,
NVFP4, accept length **3.3**.

llama.cpp #27742 lists MTP as work-in-progress; #27739 implemented it with dense
attention for the head and its author measured no throughput gain that way,
which is consistent with IndexShare being the thing that makes it pay.

Note in passing: the SGLang cookbook lists `--speculative-algorithm DSPARK`
alongside EAGLE and NGRAM. Our DeepSeek drafter is also called `dspark`
(`latent_moe/dspark.rs`); the name collision appears to be coincidental and is
not evidence of a shared design.

---

## 4. Quantized releases

### Official

| Repo | Format | Notes |
|---|---|---|
| https://huggingface.co/Qwen/Qwen3.8-Flash-Next | BF16 | 335.28 GiB |
| https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 | FP8 | Fine-grained, **block size 128**; 172.78 GiB; tensor types BF16 + F8_E4M3 + I64. "Nearly identical" metrics claimed, no table published. Plain TP8 is incompatible with the 128-wide blocks on an 8×H200 node — TEP8 required. |

### GGUF (the format we load)

Requires a llama.cpp build carrying PR #27742 (merged 2026-08-27).

> **Measured 2026-08-30: every community sub-8-bit quant of this model is
> unloadable by this codebase.** Unsloth's `UD-Q4_K_XL` carries IQ-family
> tensors (dtype code 20 found in shard 2), and AtomicChat's
> `AD-4.27bpw-Q4_K_M-M64` likewise (dtype 21 in shard 5);
> `GgmlDType::from_gguf_file_code` has no IQ arms, so `Content::read` fails on
> the first such tensor. This is the same trap the 27B ladder already
> documents — "no rung is a `UD-` file" (`quantized_qwen38.rs`) — now
> confirmed on this model from the actual shard headers. **The bring-up
> checkpoint is therefore `Q8_0`** (pure Q8_0 + F32/BF16, verified across all
> six shard headers), pinned at revision `c8b5954a88c2775c546b92593eda40ea041d3176`.
>
> **The 4-bit rung comes from a W4A16 import, not a requant** (§12.9):
> compressed-tensors `W4A16 g128 symmetric` is the *same quantization* as
> `Q4_KO` (int4, one scale per 128 along K), so the AWQ release
> `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125b` imports **bit-exactly**
> into our resident format, carrying its AWQ calibration with it —
> `qwen4exp/convert.rs`. The Q3/Q2 rungs for the smaller cards have no W4A16
> analogue and remain local requants from Q8_0.

**https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF** — Unsloth Dynamic 3.0:

| Quant | Size | Quant | Size |
|---|---|---|---|
| UD-IQ1_S | 72.5 GB | UD-IQ4_XS | 93.7 GB |
| UD-IQ1_M | 74.5 GB | UD-Q4_K_XL | 111 GB |
| UD-Q2_K_XL | 78.9 GB | UD-Q5_K_XL | 158 GB |
| UD-IQ3_XXS | 82 GB | UD-Q6_K_XL | 169 GB |
| UD-Q3_K_XL | 90 GB | Q8_0 | 188 GB |
| | | BF16 | 354 GB |

**https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF** — self-quantized
with a published `imatrix.gguf` (4000 chunks). `AD-3.84bpw-IQ4_XS-M64` 84.9 GB,
`AD-4.27bpw-Q4_K_M-M64` 92.9 GB (recommended), `AD-5.00bpw-Q5_K_M-M64`
110.5 GB. Split into **33 shards**; a separate `mmproj-Qwen3.8-Flash-Next-F16.gguf`
carries the vision tower; **the n-gram table is isolated in shard 2** so it can
be paged or offloaded independently. Requires `--jinja` for the chat template.

Note the floor: even UD-IQ1_S is 72.5 GB, because the PLE table is most of the
file and does not compress like a weight matrix. **The trunk is small and the
table is enormous** — which is why the offload story in §3.3 is the whole
deployment question, not an optimisation.

### Other formats

| Repo | Format |
|---|---|
| https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 | NVFP4, day-0, SGLang-targeted (the checkpoint LMSYS benchmarks) |
| https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4 | NVFP4 |
| https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 | NVFP4, abliterated |
| https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit | MLX 4-bit (also 6bit, 8bit, mixed-4_8bit) |

No AWQ or MXFP4 release found as of 2026-08-30. MXFP4 matters to us because the
engine already has a bit-exact MXFP4 codec for DeepSeek-V4-Flash
(`candle-kernels/src/quantized/impl/mxfp4_f32.cu`); absent an upstream release
we would produce our own, as we do today for `MXFP4_KO`.

---

## 5. What this engine already has

The favourable finding of this research pass. Each row is checked against the
code, not assumed.

| Component | Existing machinery | Fit |
|---|---|---|
| **Hybrid 3:1 layer schedule** | `delta_net/`, `qwen35/` (`forward.rs`, `wave.rs`), `KvLayerMap` in `delta_net/kv_layout.rs` | Direct. Same `[linear ×3, full]` pattern, same `full_attention_interval = 4`. 12 KV-bearing layers of 48. |
| **Gated DeltaNet** | `delta_net/{mix,cuda,state_store,quantized}.rs` + `candle-kernels/src/delta-net/`; chunked prefill scan, decode step, wave-atomic state store, turn-seal snapshot record | Direct. Geometry (48 V heads / 16 QK heads @ 128, conv kernel 4, FP32 state) is **identical to Qwen3.8-27B**, already implemented. |
| **head_dim 256 + partial rotary 64** | `models/rotary_layout.rs`, paged decode `<HD=256, NP=4>`, `paged_prefill_float_fallback`, the int8 256 prefill kernel (§7.14 of the sibling doc) | Direct. `partial_rotary_factor 0.25`, `rope_theta 1e7`, `mrope_section [11,11,10]` all match the Qwen3.5 geometry already parsed and cross-checked at load. |
| **Sigmoid output gate** | `QkvProjection.gate` (§4.4a of the sibling doc) | Direct. `output_gate_type: sigmoid`. |
| **QSA indexer scoring** | `candle-kernels/src/simple/indexer_score.cu`, `latent_moe/indexer.rs` | Strong. The fused kernel computes `Σ_h relu(score) · w_h` with a `-1e30` pad mask — QSA's scoring function with per-head weights set to 1/√128. The two-stage selector (`two_stage_select_batched`, `batched_causal_select_device`) is the selection path. |
| **Gated Residual** | `latent_moe/hyper.rs` (`HyperConnection`, mHC), `candle-kernels/src/simple/hyper_mhc.cu`, `sinkhorn.cu` | Strong. `hc_pre`/`hc_post` with Sinkhorn-normalised combine and fused CUDA kernels already exist, with bit-exact CPU parity references. `hc_count 4` is well inside `MHC_MAX_HC 16`. |
| **Host-resident embedding** | `models/host_embedding.rs`; hot-path invariant 3 sanctions exactly this transfer | Strong basis for PLE, but not the same shape — see §6.3. |
| **Shared expert** | `qwen35/moe.rs`, `latent_moe/moe.rs` | Direct. |
| **Three-tier expert cache** | `expert_lre/` — VRAM slots / pinned warm / mmap cold | Direct, and it is the reason a 180B model is feasible at all here. |
| **Multi-split GGUF loader** | `latent_moe/loader.rs` (`NAME-00001-of-000NN.gguf`) | Direct — needed, the GGUFs ship in 33 shards. |
| **MTP drafter** | `qwen35/mtp.rs`, `qwen35/draft.rs`, `latent_moe/dspark.rs`, `models/draft_ladder.rs` | Strong. |
| **Vocabulary 248320** | Qwen3.5 tokenizer generation | Direct — same vocab size, so the Qwen3.5 tokenizer-pairing discipline applies unchanged. |

---

## 6. What is genuinely new

### 6.1 512 experts exceeds a hard kernel bound — the one real blocker

`num_experts: 512`. Both GPU-side MoE primitives cap at 256:

- `candle-kernels/src/simple/moe_bucketize.cu:53` — `#define MAX_EXPERTS 256`,
  and `sh_counts`/`sh_offsets`/`sh_tile_pref` are `__shared__` arrays sized by
  it. Enforced at `candle-core/src/quantized/cuda.rs:6734`.
- `moe_route` refuses `n_experts > 256` at `cuda.rs:6408`.

This is not the `gpu_dispatch.rs` 128-expert table (which returns `None` and
falls back to the host path by design — see §4.4b of the sibling doc). This is a
hard bail. The shared-memory arithmetic is the question: at 512 the three
`int32_t` arrays cost `512·4 · 3 ≈ 6 KB`, which still fits comfortably in a
48 KiB block, so raising `MAX_EXPERTS` looks tractable rather than structural —
but it must be measured, not assumed, since occupancy at the larger shared
footprint is what decides whether the histogram path stays worth using.
`num_experts_per_tok: 10` is inside `MAX_TOPK` and needs nothing.

This is the first thing to settle, because it decides whether the MoE runs the
GPU-native dispatch (committed in `2db1ea12`) or the host-orchestrated path.

### 6.2 heads_per_group 12 at head_dim 256 — an exotic path becomes the main one

`24 / 2 = 12` heads per KV group. The decode kernel handles `hpg > 8` via the
wide warp=head path (`int8_decode_kernel.cuh:1743`, `WARPS_PER_BLOCK = 16`), so
this is supported — but at `HEAD_DIM = 256` that path is forced single-stage,
and the comment at `int8_decode_kernel.cuh:817-823` says why:

> This path is exotic (no target model has hpg>8 at hd256; real hd256 models
> like Gemma have small GQA ratios and take the full-perf stripe path), so the
> lost load/compute overlap is irrelevant.

Qwen3.8-Flash-Next makes that comment stale: it is a target model with hpg 12 at
hd256, so the decode kernel loses load/compute overlap on the path it will
actually run. Two stages at HD=256 is ~55 KB against the 48 KiB static
shared-memory cap, so the fix is dynamic shared memory (`cudaFuncSetAttribute`,
which the sampling kernels already do) or a split-D stage — not a constant
change. **Measure before building**: only 12 of 48 layers attend, and QSA caps
the read at ~2051 positions, so this kernel's share of decode is far smaller
here than in a dense model.

**Measured 2026-08-31, and the answer is don't build it.** The measurement is
`test_engine_qsa_at_depth --features profile`, which runs the real geometry at
the QSA-capped read — this model's steady state at *any* depth, which is what
makes one number sufficient. `decode:kernel` is **0.52 ms per layer-call**
(43.5 ms over 84 calls), so a decode step's 12 attention layers are ~6 ms of a
64–69 ms step: **about a tenth**. Whatever a second stage recovers is bounded
by that tenth, and it does not pay for moving a kernel every model shares onto
dynamic shared memory. The stale comment is corrected in place with the
measurement and the condition for revisiting (more attention layers, or a model
without QSA's cap).

For scale, the same profile puts `fwd_routing_wait` at **30.9%** and
`submit_roundtrip` at **14.8%** — the decode kernel is not where this model's
decode time goes.

### 6.3 PLE is a new state class, and it is not our host embedding

`host_embedding.rs` reads one row per token from a host table and uploads it —
the sanctioned transfer of hot-path invariant 3. PLE differs in three ways that
each matter:

1. **16 rows per token, not 1**, selected by hash rather than by token id, from
   a 20M-row table — a gather, not an index.
2. **It sits at decoder layer 1, not at the embedding**, so the gather must
   overlap with layer 0's compute or it is a serial stall inside the layer loop
   (SGLang's dedicated stream does exactly this).
3. **It carries convolution state across tokens** (`ple_conv_kernel_size: 4`),
   which makes it a *third* per-session recurrent state alongside the GDN state
   and the QSA index ring. llama.cpp #27742 records the corresponding limitation
   — PLE conv state does not carry across ubatches for chunked prefill — which
   is a wave-boundary correctness question for us, not a performance one.

The `RecurrentStateStore` and the turn-seal `Snapshot` record (§5 of the sibling
doc) are the right home for (3): the snapshot payload gains the PLE conv tail
and the QSA index ring alongside the GDN state, and the schedule hash covers
them. That is an extension of an existing record, not a new one.

Points (1) and (2) are answered by §0.1: the table is NVMe-resident behind a
2 GB non-pinned RAM cache, the hash and the gather run on the CPU, and only the
5 KB result is uploaded. What §0.1 leaves to the implementation is *where the
gather is issued from* — it must be in flight before the layer loop reaches
layer 1, or it is a serial stall. Since the row ids depend only on the token ids
(`x_{t-1}, x_t` and `x_{t-2}, x_{t-1}, x_t`), they are computable the moment the
wave's token set is known, which is before layer 0 runs. **The gather is
therefore issued at wave admission, not at layer 1** — earlier than SGLang's
overlap-with-layer-0, and it needs no dedicated CUDA stream because the work is
CPU-side and the upload is a single small transfer.

**The layout question is answered from the file (§12.2)**: one tensor,
`per_layer_token_embd.weight` at **[320,001,536 × 160]** — the 16 head
regions stacked row-major at width 160, each head's rows contiguous at its
`ple.head_offsets[h]`, with **per-head** vocab sizes summing to 320,001,536
(not uniformly 20M). The NVMe read pattern is therefore 16 row reads of 320 B
(BF16) into 16 separate regions — the second of the two candidate layouts,
and the friendlier one for the row cache. The table is isolated in its own
shard (shard 3 of the Q8_0 split), so the loader routes one whole file to the
tier without slicing anything.

### 6.4 Multimodality

First vision-capable model the engine would run. `mmproj` GGUFs exist. The text
trunk is unaffected — `language_model_only: false` is a checkpoint property, and
a text-only bring-up simply does not load the tower, exactly as the Qwen3.5 0.8B
(also a VL model) was brought up. Out of scope for the first pass; recorded so
the loader does not treat the vision tensors as unknown.

---

## 7. Sizing against our hardware

Per CLAUDE.md, model size is not bounded by VRAM: the expert cache streams
VRAM → pinned RAM → mmap, so the resident footprint is dense weights plus the
expert working set. Two things make this model unusually favourable on that
axis and one makes it unusually hostile.

Favourable: **6B active of 180B**, and only 12 of 48 layers hold KV — and those
12 read at most ~2051 positions each regardless of depth. The KV footprint at
1M context is a *fraction* of what a uniform-attention model of this size would
demand, and the O(1)-in-depth read is the same property our provenance-selected
attention argues for, arrived at from the other direction.

Hostile: **the 95 GiB PLE table**, which is 4× the RTX 3090's VRAM and 3× the
4090 Mobile's entire host RAM. §0.1 is the answer — NVMe cold tier, 2 GB
non-pinned RAM cache, CPU-side lookup, 5 KB/token uploaded. The consequence for
sizing is that the table costs **no VRAM and 2 GB of RAM on every machine**, so
it does not differentiate them; what differentiates them is expert residency
(§0.2). PCIe is not the constraint here either — 5 KB/token is nothing even on
the 3090's PCIe 3.0 link.

| Machine | VRAM | Expert format (§0.2) | Verdict |
|---|---|---|---|
| RTX 4090 Mobile 16 GB (32 GB RAM) | 16 | Q2_K | ~28% experts resident. Slowest, but the correctness gate runs here. 2 GB PLE cache out of 32 GB RAM is comfortable. |
| RTX 3090 24 GB (64 GB RAM, PCIe 3.0) | 24 | Q3_K | ~35% resident. No native FP8 (sm_86) and no b1 BMMA / INT8 IMMA provenance backends — GGUF is the path, and provenance-scan numbers here are not comparable to the other two. |
| RTX PRO 5000 Blackwell 72 GB | 72 | Q4_K | ~88% resident. The development target. |

The 33-shard GGUF layout with the table isolated in shard 2 is what makes all
three tractable — it lets the loader route that shard to the NVMe tier while
everything else loads normally, without slicing a shard.

### 7.1 Storage

Measured 2026-08-30 on the Blackwell box.

| Drive | Device | Size | Free |
|---|---|---|---|
| C: | Kingston SNV3S2000G NVMe | 1,862.8 GB | 618.3 GB |
| D: | 2× Gigabyte AG514K4TB NVMe, spanned | 7,451.5 GB | 3,963.8 GB |

**Budget ~1.9× the GGUF size per model, not 1×.** The loader writes an
`experts.pack` beside each checkpoint and it runs ~90% of the file: the DeepSeek
tree holds a 152.8 GB GGUF beside a 145.1 GB pack, and the Qwen3.6 cache holds
20.6 GB beside 18.4 GB. On a model that is ~121B experts out of a ~125B trunk
that ratio holds, so:

| Format | GGUF | + `experts.pack` | Total |
|---|---|---|---|
| UD-Q4_K_XL (Blackwell) | 111 GB | ~74 GB | ~185 GB |
| UD-Q3_K_XL (3090) | 90 GB | ~55 GB | ~145 GB |
| UD-Q2_K_XL (4090 Mobile) | 78.9 GB | ~45 GB | ~124 GB |
| All three co-resident | 280 GB | ~174 GB | **~454 GB** |

`HF_HOME` is unset, so downloads land on **C:** and stay there by decision. One
consequence to hold in view: all three quants on C: leaves it at roughly 164 GB
free. Single-quant-per-machine is the normal case and costs ~185 GB here.

Reclaimable if headroom is wanted, in order of size: `target/` at 1,441.6 GB
(`cargo prune`, which keeps the two newest generations — CLAUDE.md notes cargo
never collects it); ~530 GB of DeepSeek redundancy in `D:\models`, where the
4-way split set, the merged GGUF and `MXFP4_KO` are three forms of one model
alongside a stale 84.7 GB partial experts pack; and a 20.6 GB orphaned blob in
the HF cache, which is the only blob in a 302 GB cache that is otherwise all
snapshots.

Note for §0.1: the PLE cold tier wants the NVMe array, and D: is that array —
two spanned Gigabyte NVMe drives with 3.96 TB free. Where the *download* cache
lives and where the *served* table lives are separate decisions.

---

## 8. Open questions

1. **Raising `MAX_EXPERTS` to 512** — does the histogram bucketize hold its
   throughput at the larger shared-memory footprint, or does the host path win
   at 512? Decides §6.1. First thing to measure, and per §0.4 rule 2 the
   measurement must also show no regression at 256 for its existing callers.
2. ~~**QSA against our selection machinery**~~ — **Answered (Phase 4): it
   needs a sibling.** `two_stage_select_batched` selects once per *session*
   against a BDP sign index; QSA selects once per *query* against per-block
   scores, which is a different shape at every level. The sibling is
   `simple/qsa_topk.cu`. See Phase 4 for the ordering key that makes it exact.
3. **PLE table layout on disk** — ~~one 20M × 2560 table or 16 × (20M × 160)?~~
   **Answered (§12.2)**: one [320,001,536 × 160] tensor, 16 stacked head
   regions with per-head vocab sizes; 160-wide row reads.
4. **PLE cache hit rate at 2 GB.** The cache is sized against n-gram frequency
   skew, so the hit rate is an empirical property of real traffic, not of the
   table. Measure it on Zen Code's own corpus before trusting the 800-IOPS
   figure in §0.1 — that number assumes misses are the common case, so a good
   hit rate only improves it, but a *bad* locality profile (long-tail trigrams
   dominating) is the case worth knowing about early.
   **Partly answered (Phase 2):** on gate traffic, 96.6% with **zero
   evictions** — the cache never fills, so the mechanism is sound and cheap
   (~200 IOPS). The gate repeats similar prompts, so that is an upper bound;
   this question stays open for the corpus that matters.
5. ~~**Compressed-index cache tiering**~~ — **Answered (Phase 4): a dedicated
   per-sequence buffer**, doubling on growth, grown at wave admission (never
   inside the layer loop). A completed block's key is prepared once and stored
   ready to score, so the tier holds one 128-wide F32 key per 4 tokens per
   full-attention layer — ~3% of this model's KV — plus the ≤3 raw rows of the
   open block. What remains open is only its *migration*: the cache is hot-only
   today, with no warm/cold tier of its own.
6. **Threshold re-derivation** — `PRODUCTION_*` and `*_KV_FACTORS` rows are
   per-model and per-machine (see the `kv-calibration-is-machine-specific`
   finding). Nothing carries over from Qwen3.5. Note this now has a **per-machine
   expert format** (§0.2) interacting with it: the rows are derived on the
   machine that runs the gate, at that machine's format.
7. **Pinned GGUF revisions in every gate.** Standing rule from the 2026-08-16
   C10 incident; this model is four days old and its GGUF repos are still
   uploading, so re-upload risk is at its highest right now.
8. **Which throughput claim is real.** The vendor blog cites 7.6× prefill /
   4.9× decode at 1M; the SGLang cookbook and vLLM recipes cite 10.2× / 6.6×.
   Both are relative to Qwen3.7-Plus on their hardware, neither is a number we
   can inherit.

---

## 9. Reported benchmarks

Recorded for reference; not independently verified.

| Benchmark | Qwen3.8-Flash-Next |
|---|---|
| SWE-bench Pro | 62.5 |
| DeepSWE 1.1 | 58.7 |
| CoWorkBench (agentic) | 73.9 |
| AndroidWorld (multimodal) | 84.5 |
| MathVision (with code interpreter) | 95.7 |
| HLE | 35.9 (Claude Opus 4.6 Max leads at 40.0) |
| NL2Repo-Bench | 48.1 (DeepSeek-V4-Flash-0731 leads at 54.2) |

Training cost is stated as ~1/9 of Qwen3.7-Plus.

llama.cpp #27742 reports its implementation at **4.0068 perplexity vs 4.0126**
for the reference on wikitext-2, with 98% top-1 agreement on prose — which is
the number that matters to us, since it means the GGUF path is a trustworthy
oracle for our own bring-up.

---

## 10. Phasing

Each phase names its **exit condition**. A phase is not done when the
interesting part works; it is done when its exit condition is demonstrated.
Phases are ordered by dependency, and the ordering is deliberate: the oracle
precedes the tier, the tier precedes the kernels, the kernels precede the
engine. Building in any other order means validating against nothing.

### Phase 0 — Acquisition and schema freeze — **DONE 2026-08-30**

Download the GGUF at a **pinned revision** (§8 item 7 — these repos are days old
and still uploading). Read the llama.cpp #27742 converter and freeze, in this
document, the tensor-name and metadata-key schema for `QWEN4EXP` — the sibling
doc's §7.1 is the shape to produce. Verify the §2 config against the actual file
rather than against the model card.

*Exit met:* §12 is the frozen record — metadata from the actual shard-1 header,
tensor names and shapes from all six Q8_0 shard headers, and the reference
algebra from the merged `src/models/qwen4exp.cpp`. The PLE layout (§8 item 3)
is answered; the checkpoint had to move to Q8_0 (§4) after both community
Q4-class quants measured IQ-poisoned.

### Phase 1 — The `forward_batched` oracle — **BUILT 2026-08-30, gate green**

The reference forward of §0.6: plain tensor ops, whole prompt, no paging, no
wave engine, no int8. PLE reads a plain mmap at this stage — correctness before
the tier. QSA starts dense (score everything, select nothing) so the attention
is validated before the sparsity is, then turns on block selection. GDN is
reused from `delta_net/` rather than rewritten.

*Status:* `models/qwen4exp/` (config, hyper, ple, qsa, loader, model) +
`models/quantized_qwen38_moe.rs` (§11.1), per §12.8's split. 22 unit tests plus the
real-weights gate `test_forward_batched_oracle`, green on the pinned Q8_0:
load 11.3 s, batched ×2 prefill 32.3 s, ×1 ≡ ×2 at max |Δ| 2.1e-5, segmented ≡
one-shot at 1.9e-5 across all three carried states, and greedy ×2 decode
completing "The capital of France is" → " Paris. Paris is a city. Therefore,"
and "Water is made of hydrogen and" → " oxygen."

`test_forward_batched_qsa_depth` is green too: a 2,111-token prompt (past the
2,051 selection width, so all 12 attention layers genuinely select), 344 s
CPU prefill, deterministic across two cold runs — and the greedy next token
is " meticulous", the needle planted 2,000 tokens earlier, which is sparse
*retrieval* working, not merely finiteness. What remains for the exit below:
the llama.cpp token-level comparison and the truncation bisect — the semantic
checks are necessary, not sufficient.

Parity is established by the sibling doc's §7.3 method, which is the only thing
that has ever actually found a defect in this lineage: llama.cpp on the same
GGUF for ground truth, a layer-truncating GGUF rewriter plus tensor-zeroing to
isolate one subsystem at a time, and an independent transcription so agreement
is evidence rather than a shared assumption.

*Exit:* greedy continuation matches llama.cpp on real weights, and each
subsystem matches in isolation under truncation.

### Phase 2 — The NVMe embedding tier — **BUILT + MEASURED 2026-08-31**

The new module of §0.1: NVMe-backed store, 2 GB non-pinned RAM cache, n-gram
hash, dilated-conv state. Unit-tested per §0.7, with a harness reporting hit
rate and IOPS against real traffic (§8 item 4).

It swaps into the Phase-1 oracle and must be **bit-identical** to the mmap path
it replaces. That equivalence is the whole point of building the tier second:
the oracle is already trusted, so any difference is the tier's.

*Status:* `qwen4exp/ple_cache.rs` — a slot-arena cache of **quantized** row
records (2 GB holds ~12.6M Q8_0 rows vs ~1.6M dequantized; the row's consumer
is an upload whose cheapest form *is* the record), CLOCK eviction (one byte of
metadata per slot at 12.6M entries), hit/miss/eviction counters. Wired under
the oracle's `PleSource`, serving the **same bytes** the direct-disk path
read, so bit-identity is by construction and the gates re-prove it end to
end. The transfer half is `simple/ple_gather_dequant.cu`: gathered records
upload **quantized** (~2.7 KB/token) and widen on the card —
GPU-vs-CPU-dequant parity pinned **bit-for-bit** by
`ple_cache::gpu_dequant_matches_cpu_dequant_bit_for_bit`.

**Measured on the §11 gate ladder** (five configs, 21,520 tokens × 16 rows):

```
lookups 344,320 | hits 332,705 (96.6%) | misses 11,615 | evictions 0
```

The counters existed from the first commit and **nothing read them** — the
tier's sizing was a design estimate for the whole bring-up because
`cache_stats()` had no caller. It is now a reported gauge:
`ManagedBatchedModel::row_cache_stats` and an `Embedding Row Cache` section in
the gate's report, beside the expert table.

Two things the number says. **Zero evictions**: the 2 GB cache never filled, so
every miss is a first touch and the tier is oversized for this traffic rather
than under. And 11,615 row reads across a multi-minute run is ~200 IOPS against
the ~800 §0.1 budgets for — the tier is nowhere near its shape's limit.

What it does **not** say: gate traffic is five repeats over similar prompts, so
its n-gram locality is unrealistically good, and a hit rate measured there is an
upper bound. §8 item 4 stays open for exactly this reason — the number to trust
is Zen Code's own corpus, where long-tail trigrams are the case worth knowing
about. This measurement bounds the *mechanism* (it works, it is cheap, nothing
evicts at this scale), not the deployment.

**Where the served table lives, decided:** the merged KO artifact on the D:
NVMe array, which is the tier. That is where it already sat, by accident of
where the conversion wrote it; it is now the stated choice. The HF *download*
cache is a separate decision and stays where it is (§7) — conflating the two is
what the note there warns against.

*Exit met:* bit-identical to the mmap path by construction, and hit rate + tier
reads recorded.

### Phase 3 — MoE at 512 experts — **DONE 2026-08-31, exit met**

§6.1. Extend `MAX_EXPERTS` to 512 by template parameter, and per §0.4 rule 2
measure the histogram bucketize at **both** widths — 512 for us, 256 to prove no
regression for its existing callers. If occupancy at the wider shared footprint
loses to the alternative, the answer is a different GPU-side dispatch, not a
host-side sort (§0.3).

*Status:* `moe_bucketize` serves 512 experts, bit-exact against the CPU
reference at 512, 257 and every width below (`cuda_moe_bucketize_matches_cpu_reference`
gained top-half, single-expert-at-511, sentinel and fuzz rungs).

**What the raise actually cost was not shared memory.** The arithmetic §6.1
predicted holds — three int32 arrays over the expert axis at 512 is ~6 KB, and
the block totals ≈39 KB against the 48 KiB cap. The real work was that three
phases were written **one thread per expert** (offsets, the per-expert prefix
across chunks, tile emission) against a 256-thread block. At 512 that form
leaves the upper half of the experts with no scatter base and no tiles —
silently, because their assignments land at stale positions rather than
faulting. All three are grid-stride loops now; at 256 the two forms coincide.

**Measured, both widths** (`bench_moe_bucketize`, µs/call):

| config | before | after |
|---|---|---|
| decode-64 (e128) | 17.73 | 16.99 |
| prefill-2048 (e128) | 179.35 | 177.37 |
| prefill-8192 (e128) | 684.60 | 674.32 |
| prefill-4096 **e256** | 472.26 | 459.39 |
| prefill-2048 e512 k10 | — | 426.94 |
| prefill-4096 e512 k10 | — | 872.50 |

No regression at 128 or 256 — every shared row is flat or marginally faster,
which is what a single-block kernel whose occupancy cannot change should do.
At equal assignment count 512 costs ~28% more than 256 (872 vs 681 µs at
a_ub 40960): `NCHUNK = 8192/n_experts` halves to 16, so the phase-3 scatter is
16-way parallel instead of 32. Widening `sh_cc` past 32 KB needs dynamic shared
memory; measure the model before spending it.

**The finding that matters more than the raise.** Qwen3.8-Flash-Next still
takes the **host** dispatch path, and the 256 cap was never why. Device
dispatch requires `all_resident` — `GpuDispatchTables` captures raw slot
addresses, which only hold if nothing ever streams — and at Q4_KO the grid is
24,576 experts × ~2.8 MB ≈ 68 GB against a 53.6 GB zone, so ~80% fits and the
rest streams. A streaming cache's per-layer routing readback is the *sanctioned*
exception (a) of hot-path invariant 3, not a fallback.

That puts the residency cliff, not the kernel bound, in charge: at ~3.4 bpw the
grid is ~51 GB and would fit, so the §0.2 expert-format ladder decides whether
routing leaves the card at all. Recorded in §0.2.

The path a run took is now **reported** rather than inferred: `PipelineStats`
carries `device_dispatch` and the gate's expert table prints `MoE dispatch:
device | host (readback)`. It was previously a `tracing::warn` that every gate
harness dropped on the floor for want of a subscriber — qwen3-MoE reads
`device` at 100% hit rate, qwen4exp reads `host (readback)`.

*Exit met:* the kernel runs at 512, bit-exact, with no measured regression at
256.

### Phase 4 — QSA on the engine — **BUILT 2026-08-31, exit met**

The indexer and block selection as a fused, batched, paged kernel path (§0.4),
reusing `indexer_score.cu` and the two-stage selector or extending them (§8
item 2). The compressed-index cache tier is decided here (§8 item 5): one BF16
key per four tokens plus the four-slot ring, riding the chunked KV backing or a
dedicated arena.

*Status:* the engine selects. A 2,434-token prompt recovers a needle planted
in its first sentence, deterministically, with 29,292 query rows narrowed
(`test_engine_qsa_at_depth`) — the regime the engine could not reach at all
while it attended densely.

**The selection is defined once** (`qwen4exp/qsa_select.rs`) and read by both
consumers: the CPU oracle expands it into its additive mask, and the engine
packs it as `(block << 2) | (cells − 1)` entries, ascending, that the paged
kernels binary-search (`candle-kernels/src/qsa_select.cuh`). That is what makes
"the engine matches the oracle" a property of one function rather than of two
implementations that happen to agree — and the reference's `(score desc, cell
asc)` cell ranking is proved equal to ranking *blocks* and expanding them,
which is why 2051 cells fit in ≤514 entries.

**Three tests pin it, none with a tolerance:**

- `qsa_select` — the packed selection expands to the reference cell ranking,
  at every phase of `qpos mod ratio`, including the partial block the budget
  cuts and the all-equal-scores case the tie rule exists for.
- `qwen4exp::indexer` — the device selection kernel reproduces that definition
  exactly (ties included, and past the streaming buffer's first trim), and the
  whole device path (project → pool → norm → rope → score → select) reproduces
  the CPU oracle's `qsa_selection_mask` row for row.
- `tests/qsa_kernel_tests.rs` — attention honours the selection *exactly*:
  selecting everything is bit-identical to no selection, and negating the K/V
  of every unselected token moves no bit of the output (while the same
  comparison unmasked moves it, so the test can tell the difference). All three
  decode routes (warp=head, batched-M, warp-stripe) and the prefill kernel.

Findings a successor should not re-derive:

- **§8 item 2, answered by reading:** `two_stage_select_batched` does not
  express QSA's shape and should not be bent into it. It selects once per
  *session* against a BDP sign index; QSA selects once per *query* against
  per-block scores. The sibling is `simple/qsa_topk.cu`: one CUDA block per
  query, streaming its candidates through a shared buffer against a running
  threshold, with the `(score desc, block asc)` order carried as a single u64
  key — sound because scores are sums of ReLUs and therefore never negative,
  so a float's bit pattern orders like its value.
- **§8 item 5, answered:** the compressed index rides a **dedicated
  per-sequence buffer**, not the chunked KV backing. A completed block's key
  never changes (its pool, norm and rotation are all functions of the block),
  so it is prepared once at completion and stored ready to score — which makes
  the scan a plain dot product and the cache 1/4 the raw keys' size. Only the
  `ratio − 1` raw rows of the block still filling are held raw.
- **The unselected-token test had to be sign-only**, and that is a real
  property of the kernels rather than a testing convenience: the prefill
  kernel's FP-fallback V requant takes its scale as a max-abs per (dim,
  32-token tile), so a perturbed *magnitude* anywhere in a tile moves the
  quantization of the selected tokens beside it. The mask cannot prevent that,
  and the kernel already has it for causally-dead columns.
- **Measured cost, and it is not zero:** the index keys must be cached at every
  depth (a wave that crosses the budget scores blocks the waves below it
  built), so a shallow context now pays the append it did not pay before. On
  the §11 gate — entirely inside the budget, so *no* row selects — bulk is
  unchanged to slightly better (550/1600/1655/1673/1523 t/s) but single-session
  decode is **~14% down** (150 vs 174 t/s at ×16), from ~1.3k small launches
  per step: 12 layers × 16 sequences × the per-sequence pool/append chain.
  Hoisting both projections to one GEMM per wave recovered none of it, which
  locates the cost in the per-sequence chain itself. The fix is the same shape
  as invariant 2b: one append kernel over a descriptor table of per-sequence
  cursors, replacing the per-sequence `cat`/pool/`slice_assign`. Recorded, not
  hidden.

*Exit met:* the device selection matches the oracle's exactly, the attention
provably reads exactly it, and the ≤2051-position read at depth is covered end
to end on the real checkpoint.

### Phase 5 — Production integration — **ENGINE RUNNING, GATE GREEN 2026-08-31**

`ManagedBatchedModel` wired into `qwen35/`'s hybrid sweep; the shared
`ExpertCache`; the three `Model` variants and the VRAM-keyed `model()` of §0.2,
**including the `model_choice.rs` module-doc rewrite** that the ladder makes
necessary. Decode at hpg 12 @ head_dim 256 is measured here (§6.2), harness
first, and the stale kernel comment is corrected whichever way the measurement
lands.

*Status — the measurement item is done, the ladder is blocked (see Phase 6).*
§6.2's decode-kernel question was measured and answered "don't build it", and
the stale kernel comment is corrected in place. The §0.2 three-variant ladder
could not be written: a `Model` variant carries download coordinates and this
engine loads a locally prepared artifact, so the honest half — `ModelArch::Qwen4Exp`
plus its builder arm — landed here and the ladder moved to the cutover, where
the decision it needs belongs. The `model_choice.rs` doc rewrite travels with
it, as §0.2 requires, rather than arriving before the thing it documents.

*Status:* the production engine is built and its gate is green — a
model-specific [`WaveSweep`] on `drive_wave` (the DeepSeek position, not a
`QuantModel` fork; the GR seam note below explains why), in
`qwen4exp/{engine,wave,batched_attention}.rs` + the gate in
`quantized_qwen38_moe.rs`. Loaded from ONE merged Q4KOEXP GGUF
(`convert::merge_gguf_split`, 124 GB on `D:\models\qwen38-flash-next`): F32
trunk (HC/PLE/GDN constants; embed stored BF16), KO twins for every dense
projection, Q4_KO experts through the `ExpertCache` (54.9 GiB zone, 48→99 %
hit, zero cold loads), PLE through the §0.1 row cache. The gate ladder
(BF16 ×[1,4,8,16,1], StoryRewrite, `Int8Mode::auto` = Precision) is 5×100 %
valid at **534 / 1533 / 1615 / 1607 / 1359 bulk t/s** and 22.2 / 70.5 / 112.0
/ 174.2 / 24.6 per-session decode — ahead of DeepSeek-V4 on the same box on
nine of ten figures (its ladder: 283/713/940/1116/335 bulk).

Bring-up findings a successor should not re-derive:

- **Kernel extensions landed by template parameter** (§0.4 rule 2), each with
  its harness rung: the delta-net norm-gate epilogue gained the sigmoid
  z-gate (`<SIGMOID_GATE>`, both instantiations parity-gated), and
  `moe_route` gained 16-slot `_x512` instantiations (512-expert × top-10 and
  257-boundary rungs in `cuda_moe_route_matches_reference`).
- **KO twins have no float GEMM loader.** A `MoeInput::Float` against a KO
  expert pack read garbage (~1e38) through the GEMX float path; the fix is a
  third arm in `expert_lre/compute.rs` — float-gather, quantize the stacked
  block once, run the int8 grouped chain. The q8a1024 byte-gather stays
  reserved for hidden widths that tile 1024 (2560 does not).
- **`sum(1)` over the hc axis was 73 % of bulk.** A middle-axis reduction
  over 4 elements takes candle's generic strided-reduce (~9.6 ms per call at
  prefill width); `hc − 1` strided slice-adds are sub-millisecond
  (`hyper.rs::hc_mix`). The grouped norm likewise runs the fused
  `candle_nn::ops::rms_norm` kernel, not a five-pass op chain.
- Deliberately deferred, recorded not hidden: the fused GR pre-mix kernel
  (the eager chain is now ~6 % of bulk; `GR_EAGER_ROW_CAP = 2048` in
  `prefill_width_cap` bounds its pool peak until the fusion lands), int8 MoE
  *activations* (needs the 2560-tiling gather), the batched index-key append
  (Phase 4's measured ~14 % single-session decode cost), and the C-ladder +
  snapshot/resume (Phase 6). The profiled next wall is `fwd_routing_wait`
  (the 48 per-layer routing syncs; 3.6 s of 5.2 s at ×16). QSA selection at
  depth is no longer on this list — Phase 4 built it.

**What the DeepSeek port (2026-08-31) already settled for this phase.** The
generic loop is now the *only* wave loop: DeepSeek-V4 runs
`drive_wave`/`WaveSweep` like every other model, so this phase assembles from
proven parts rather than porting around a private precedent. Inherited
directly: the sweep owns its session (headers built at the sweep's own phase
order — required by any model whose arena state moves before the layer loop
reads it); speculative verify blocks ride the prefill slot as one `[1,k]`
member per sequence (`verify_wave.rs` documents the encoding; both hybrid and
latent stacks now use it); the three `WaveSweep` offset/advance/rollback hooks
state each model's KV-bookkeeping contract; the corpus-state failure bracket
(capture → sweep → restore-on-error) is the lifecycle the PLE conv tail and
QSA index ring adopt; and the grouped-GEMM dispatch sort is one function
(`expert_lre::sort_assignments_by_expert`) serving both router families — the
512-expert host path needs only the gate math, which stays model-side.

**The GR seam's true shape — recorded so it is built once, not re-derived.**
The obvious seam ("replace `attention_norm` and the residual `add_mut` in
`forward_layer_batched_mixed`") is wrong: the layer norm is producer-fused
*inside* the per-group attention path (each group calls `attention_norm` on
its own row-slice, emitting the fused norm→q8a128 `DynamicActs`). A 4-stream
GR layer must instead run its pre-mix ONCE over the whole wide buffer *above*
the group loop (grouped RMSNorm → low-rank gate → mean collapse → the narrow
block input, plus the inject weights carried to the post-combine), with the
per-group norm hook becoming a pass-through. That raises one genuinely open
implementation question, to be answered by measurement when this phase builds
it: whether the whole-buffer pre-mix emits one fused `DynamicActs` the groups
then *slice* (needs row-sliceable q8a128 operands) or emits the narrow float
buffer and lets each group keep its fused norm→quant epilogue with a unit
norm weight. Per §0.5 the seam is deliberately NOT pre-built against guessed
requirements — it lands here, with GR's real algebra (§12.3) in hand.

*Exit:* the §11 gate green on this machine at the unquantized-KV rungs.

### Phase 6 — Quality and cutover — **thresholds + C-ladder DONE 2026-08-31**

`PRODUCTION_*` and `*_KV_FACTORS` derivation (§8 item 6), per machine and per
expert format. The full C-ladder green. Snapshot/resume proven across **three**
carried states — GDN, PLE conv tail, QSA index ring (§6.3). Then the zend
cutover: `model_choice.rs` moves off Qwen3.6-35B-A3B, which is a substrate and
threshold migration, not a constant change.

*Status — the threshold half is done.* The gate ladder gained C0, **C5** and C8
plus C10 at two widths, and `QWEN4EXP_KV_FACTORS` was derived by bracketing
both axes on it (six runs; the table and both edges are recorded on the
constant). The shipped row is **k 1.8 / v 3.0**, whole ladder green:

| rung | ratio | |
|---|---|---|
| C0 | 2.30× | |
| **C5** | **4.25×** | **the level zend runs** |
| C8 | 5.62× | |
| C10 ×2 / ×8 | 7.36× / 7.38× | the calibration probe |

In line with the lineage (3.5 at 7.13×, 3.6 at 6.8×) even though only 12 of 48
layers hold K/V at all.

**The row sits one notch under the C10 edge deliberately**, because C10 is not
the operating point — zend runs `compression_level(5)`. Held to C10's margin
this row would pay real ratio at C5 to buy headroom at a level nothing runs,
and C5 is demonstrably nowhere near an edge: **C8 still passed at 6.87× under
thresholds loose enough to break C10 entirely** (k 2.2 / v 4.5).

One caveat stands and belongs here as much as on the constant: the row is
derived at gate depth (~713 tokens), where QSA's selection is the identity and
every cell is read. Above 2051 cells a block's quantization error reaches the
output only when the indexer selects that block, so up there this row is a
bound rather than a measurement — the deep-context C-rung is the outstanding
piece.

*Outstanding:* snapshot/resume, and the cutover.

**Snapshot/resume now spans four states, not three.** `RecurrentStateStore`
already exports and imports the GDN half behind a `schedule_hash`, and
candle-conversation's persistence record is where it lands. The PLE conv tail
and the QSA index cache have to join that payload and that hash — an extension
of an existing record, as §6.3 says. Until they do, a resumed conversation
continues from an empty conv tail and an empty index cache, which is silent:
the model answers, just not from the state it left.

**The cutover is blocked on a question the plan did not anticipate**, and it
should be settled before anyone starts it. §0.2 says the ladder "needs no
invention: three `Model` variants plus a VRAM-keyed `model()`". But a `Model`
variant carries *download* coordinates (`model_repo` / `model_filename` /
`model_bytes`), and this engine loads a **locally prepared** merged KO artifact
that no repo serves — the same posture as DeepSeek-V4, which is why that model
has a `ModelArch` and no named `Model` variant at all, reachable only through
`Model::Custom`. So the ladder's three variants cannot be written as three
specs today, and two of the three rungs have no artifact either: §4 records
that the Q3/Q2 rungs "have no W4A16 analogue and remain local requants from
Q8_0", which nobody has built or evaluated.

What *did* land for the cutover: `ModelArch::Qwen4Exp` and its builder arm, so
the conversation layer can construct this engine from a path like any other
model. That is the part that needed no decision.

The artifact half of the ladder is built — every rung has a recipe and a build
(§0.2, *The per-rung artifact*), the gate prepares the rung for the card it runs
on, and zend's preset resolves its artifact by the `Q4_KO` recipe's name. What
`model_choice` runs is a separate decision: it runs Flash-Next above 60 GiB and
Qwen3.6-35B-A3B below.

Measured on the RTX 4090 Mobile (16 GB, 32 GB RAM) at the `Q2_KO` rung, 88 GiB
artifact, BF16 ×1/×4/×8 and C0/C5/C8/C10 all 100% valid, twice:

| rung | bulk t/s | decode t/s | ratio |
|---|---|---|---|
| BF16 ×1 | 117 | 11.4 | — |
| BF16 ×4 | 402 | 30.8 | — |
| BF16 ×8 | 303 | 44.0 | — |
| C5 ×2 | 240 | 17.2 | 4.06× |
| C10 ×8 | 261 | 28.1 | 6.38× |

The expert zone sits on its floor (2 GiB, ~6% of 25,088 slots), the warm tier
holds 30% (9.3 GiB of pinned RAM), and the hit rate is 10–18%: every wave
streams most of its experts from RAM or the pack. BF16 ×16 does not fit — 36
GDN layers carry 256 MiB of recurrent state per sequence — and is gated at
24 GiB.

### Phase 7 — Optional, after the above

MTP with IndexShare (§3.4), and the vision tower (§6.4). Neither blocks
anything; both are recorded so the loader does not treat their tensors as
unknown in the meantime.

---

## 11. Gate tests

The model gets the same iterating gate every production model here has — an
`#[ignore]`d `test_parallel_batched_forwarding` with a config ladder, per-session
fixture validation, a performance table, and expert-pipeline stats. The sibling
doc's §6 is the template and its rules carry over unchanged.

### 11.1 The naming collision, settled

`quantized_qwen38.rs` **already exists** and is the *other* Qwen3.8 — the 27B
dense member of the `qwen35` lineage. This model cannot take that name.

**Decided (2026-08-31, by the project owner): the model file is
`quantized_qwen38_moe.rs`** — the same dense/`_moe` split the 3.5 and 3.6
siblings use (`quantized_qwen35.rs`/`quantized_qwen35_moe.rs`,
`quantized_qwen36_moe.rs`), so the family reads uniformly. The *machinery*
module keeps the checkpoint's own name, `models/qwen4exp/`, per §7.20 of the
sibling doc — model files over shared machinery, no machinery naming a model
version. (An earlier draft of this section argued for `quantized_qwen4_exp.rs`;
consistency with the sibling files won.)

### 11.2 The ladder

```
cargo test --release --features cuda,verbose --lib --package candle-transformers \
  quantized_qwen38_moe::tests::test_parallel_batched_forwarding -- --ignored --nocapture
```

Configs, in the order they should first go green:

| Rung | What it proves |
|---|---|
| Unbatched baseline | The oracle path end to end |
| BF16 ×1 / ×N | Batching changes nothing (§0.6) |
| Q8_0, Q4_0 | Weight quantization |
| C0…C10 StoryRewrite ladder | KV compression at head_dim 256, on the 12 attention layers only |
| Seal → snapshot → resume → continue | The **three** carried states survive a turn boundary |
| Fork from an earlier turn | The recompute path when the tail snapshot is newer than the cut |
| Q4_K / Q3_K / Q2_K expert formats | The §0.2 ladder, each on its own machine — rungs **requantized locally from the pinned Q8_0** (§4: no community sub-8-bit file is loadable) |

### 11.3 Rules

- **Pinned GGUF revisions, never `"main"`.** The 2026-08-16 C10 incident — an
  upstream re-upload silently invalidating a tuned threshold row — is the
  standing reason, and §8 item 7 says the exposure is at its peak right now.
- **Fixture-derived expectations**, so any single config runs standalone. This
  is what made 16-second iteration possible during the C10 walk and it is
  preserved deliberately.
- **Every gate builds on every machine.** Where a rung cannot run within an
  envelope it fails fast at model load with an explicit capacity message, not an
  OOM backtrace.
- **Threshold rows are derived on the machine that runs the gate**, at that
  machine's expert format (§8 item 6).
- **Kernel work carries its own harness**, separately from this gate (§0.4
  rule 4). The gate proves correctness at the model level; it is not a
  performance instrument and must not be used as one.

---

## 12. Phase 0 record — the frozen `qwen4exp` schema

Frozen 2026-08-30 from three primary sources, none of them a write-up: the
actual GGUF shard headers (`unsloth/Qwen3.8-Flash-Next-GGUF` @
`c8b5954a88c2775c546b92593eda40ea041d3176`, all six Q8_0 shards), and the
merged llama.cpp implementation (`src/models/qwen4exp.cpp` on master, PR
#27742). Where this section disagrees with §2–§6, this section is right and
the earlier section carries a pointer here.

### 12.1 Metadata

`general.architecture = "qwen4exp"` (one token, no underscore — the HF
`model_type` `qwen4_exp` and the llama.cpp enum `QWEN4EXP` both differ from
the on-disk string). 67 entries; the load-bearing ones:

| Key (`qwen4exp.` prefix) | Value |
|---|---|
| `block_count` / `context_length` / `embedding_length` | 48 / 262144 / 2560 |
| `full_attention_interval` | 4 (no `attention.recurrent_layers` array) |
| `attention.head_count` / `head_count_kv` / `key_length` / `value_length` | 24 / 2 / 256 / 256 |
| `attention.layer_norm_rms_epsilon` | 1e-6 |
| `attention.compress_ratios` | i32[48] — per-layer QSA ratio; 0 ⇒ dense, layers sharing a ratio share selection inputs |
| `attention.indexer.head_count` / `key_length` / `top_k` | 4 / 128 / 2048 |
| `rope.dimension_count` / `dimension_sections` / `freq_base` | 64 / i32[4] / 1e7 |
| `ssm.state_size` / `group_count` / `time_step_rank` / `inner_size` / `conv_kernel` | 128 / 16 / **48** / 6144 / 4 |
| `expert_count` / `expert_used_count` / `expert_feed_forward_length` / `expert_shared_feed_forward_length` | 512 / 10 / 640 / 640 |
| `hyper_connection.count` / `low_rank` | 4 / 320 — **the only HC keys; no Sinkhorn keys exist** |
| `ple.layers` | i32[1] — llama.cpp asserts exactly one PLE layer |
| `ple.ngram_size` / `heads_per_ngram` / `conv_kernel` | 3 / 8 / 4 |
| `ple.layer_multipliers` | u64[3] — the hash multipliers |
| `ple.head_offsets` / `head_vocab_sizes` | u64[16] each — per-head regions of the one table |
| `ple.eos_token_id` / `image_token_id` | 248044 / 248056 — EOS resets the hash window |
| `embedding_length_per_layer_input` | 160 |
| `split.count` / `split.no` / `split.tensors.count` | 4 / 0 / 1224 (Q8_0: 6 splits) |

Absent, with consequences: no `nextn_predict_layers` (**this conversion
carries no MTP head** — llama.cpp #27742 lists MTP as WIP; speculation waits
for a sidecar or a later conversion), no `vocab_size` (fall back to
`tokenizer.ggml.tokens` length, 248320), no `expert_weights_norm` (top-k
renorm on, per the reference call), no Sinkhorn keys. Tokenizer:
`tokenizer.ggml.pre = "qwen35"` — the lineage's generation, as §5 assumed.

### 12.2 Tensors (from the Q8_0 shard headers, all 1224 accounted)

Split layout: shard 1 metadata only (0 tensors); shard 2 `token_embd.weight`,
`output.weight` [248320, 2560], `output_hc_{norm,down,up}`; shard 3
**`per_layer_token_embd.weight` [320,001,536 × 160] alone** (the isolation
§7's tiering wants); shards 4–6 the 48 `blk.*` layers.

Per layer, every layer: `hc_attn_{norm [10240], down [320,10240], up
[10240,320], inject [4,10240]}` and the same four `hc_ffn_*` — **there are no
`attn_norm`/`post_attention_norm` tensors; the HC norms are the layer norms**,
and no `output_norm` — the final HC mixer (`output_hc_*`, no inject) is the
output norm. MoE per layer exactly as `qwen35moe`: `ffn_gate_inp` [512,2560]
F32, `ffn_{gate,up}_exps` [512,640,2560], `ffn_down_exps` [512,2560,640],
`ffn_gate_inp_shexp` [2560] F32, `ffn_{gate,up,down}_shexp`.

GDN layers (36): the exact `qwen35` names and shapes at the wider geometry —
`attn_qkv` [10240,2560], `attn_gate` [6144,2560], `ssm_conv1d` [10240,4] F32,
`ssm_dt.bias` [48], `ssm_a` [48], `ssm_{beta,alpha}` [48,2560], `ssm_norm`
[128], `ssm_out` [2560,6144].

Attention layers (12): `attn_q` [12288,2560] (interleaved `[q|gate]`, 24×2×256),
`attn_k`/`attn_v` [512,2560], `attn_output` [2560,6144], `attn_{q,k}_norm`
[256], plus the indexer: `indexer.q_proj` [512,2560] BF16, `indexer.k_proj`
[128,2560] BF16, `indexer.{q,k}_norm` [128] F32.

PLE layer (blk.1 only): `ple_key` [10240,2560], `ple_value` [2560,2560],
`ple_norm_{key,query,conv}` [10240] F32, `ple_conv1d` [10240,4] F32.

### 12.3 The Gated Residual, as implemented (`build_hc_mix` / `build_hc_combine`)

The residual is `[n_embd, 4, T]`, initialised as four copies of the embedding.
Per block:

```
xn    = rms_norm(x, per stream) ⊙ w_norm          w_norm already folded to (1+γ)
lo    = silu(down(xn) / hc)                        hc_dim → 320
gate  = sigmoid(up(lo))                            320 → hc_dim, element-wise read gate
mixed = mean over streams of (xn ⊙ gate)           → [n_embd, T], the block input
inject = w_inject(xn)                              → [hc, T]
combine: res += block_out ⊗ (2·sigmoid(inject/hc)) per-stream scatter weight,
         centred on 1 so a zero injection is a plain residual add
```

No Sinkhorn, no iteration, no epsilon beyond the RMS eps. The final
`output_hc_*` mix (no inject) **is** the output norm; logits are
`output.weight · mixed`.

### 12.4 PLE, as implemented (`build_ple` + host-side hash)

At layer 1, *before* that layer's `hc_attn` mix:

```
rows  = 16 hashed row ids per token (host-side; §3.3 hash; EOS resets)
emb   = gather → [2560, T]  (head-major concat of 16×160)
key   = grouped_norm(ple_key·emb,  ple_norm_key)    → [n_embd, 4, T]
query = grouped_norm(res_hc,       ple_norm_query)
s     = Σ_embd (key ⊙ query) / √2560                → [1, 4, T] per-stream
gate  = sigmoid( sgn(s) · √max(|s|, 1e-6) )
gated = (ple_value·emb) broadcast to 4 streams ⊙ gate
conv  = silu( depthwise causal conv over time of grouped_norm(gated, ple_norm_conv),
              kernel 4, dilation 3 )                → 9 tokens of carried history
res_hc += gated + conv
```

The conv history `[9 × 10240]` F32 per sequence is the third carried state of
§6.3, alongside the GDN state and the QSA index cache.

### 12.5 QSA, as implemented (`build_qsa_top_k` / `build_attn_qsa`)

- The indexer **caches raw keys** (`index_k_proj·x`, [128] per token);
  pooling (mean over `ratio` cells), RMS norm, and rotation are applied at
  read time, with the **block's first position** as the rope position.
- Queries: `index_q_proj·x` → [128, 4, T], normed, roped at token positions.
- Score: `ReLU(q·k̄)` summed over the 4 heads — **no 1/√128 scale** (only the
  rank matters, so the reference drops it; §3.1's scale is cosmetic).
- Every cell of a block inherits the block score; the token-level causal mask
  is added; per-token `top_k` of width `min(n_kv, 2048 + ratio − 1)` = **2051**
  selects cells; attention runs dense GQA with all unselected cells masked
  to −inf. At `n_kv ≤ 2051` the selection is the identity and QSA *is* dense
  attention — a gate exercising real selection needs a >2051-token context.
- The attention half is `qwen35`'s exactly: interleaved `[q|gate]`, q/k norm,
  partial rotary 64/256 (text-only IMRoPE reduces to classic NeoX, as the
  lineage's reference already argues), scale 1/√256, sigmoid gate before
  `attn_output`.

### 12.6 GDN: one numerical difference from Qwen3.5

`build_norm_gated` applies `rms_norm_per_head(o) ⊙ sigmoid(z)` — the
Qwen3.5/3.6/3.8-27B lineage uses `silu(z)` (`delta_net/mix.rs:1581`). The
comment in qwen4exp.cpp calls this "the one numerical difference from
Qwen3.5's GDN". Everything else — conv→silu, l2-norm q/k, β sigmoid, α
softplus × −exp(A), the 48V/16K@128 delta rule — is the existing `delta_net`
subsystem unchanged. The z-gate becomes a parameter of the mixer, not a fork
of it.

### 12.7 MoE

`build_moe_ffn(..., LLM_FFN_SILU, norm_topk=true, gating=SOFTMAX)` plus the
sigmoid-scalar-gated shared expert — **identical semantics to `qwen35moe`**
(`qwen35/moe.rs` is the oracle's block as-is) at 512 experts, top-10.

### 12.8 What the oracle build does with this (Phase 1 shape)

The reference stack cannot dequantize a 180B checkpoint to F32 (~700 GB); the
box has 194 GB of RAM. So the oracle splits the weights three ways:

- **Dequantized F32 at load (~19 GB)**: everything except the routed experts
  and the PLE table — embeddings, all GDN/attention/indexer/HC/PLE-projection
  weights, routers, shared experts, the LM head.
- **Routed experts stay on disk** (Q8_0, ~123 GB): per layer, the router runs
  first, then only the union of routed experts is read (each expert's rows are
  contiguous in the 3-D expert tensors) and dequantized for that forward.
  Host-side routing is fine *here* — the oracle is the CPU reference; the
  invariants govern the production path.
- **The PLE table stays on disk** (Q8_0, 54 GB in its own shard): 16 row
  reads per token, dequantized per row — the mmap precursor of the §0.1 tier,
  which must later be bit-identical to it.

`forward_batched` packs all sequences' tokens into one `[ΣT, hidden]` row
block: embeddings, HC mixes, routers, expert GEMMs, and the LM head run over
the packed rows in one op each; only the three stateful mixers (GDN scan,
attention+QSA, PLE conv) iterate per sequence inside the layer. That is the
same row-packed shape the production wave runs, which is what makes the two
comparable (§0.6).

### 12.9 The W4A16 → Q4_KO expert import

Three facts compose into a 4-bit expert rung with **zero conversion loss**:

1. **KO twins are re-quantized from F32, never byte-permuted** (`ko_quant.rs`) —
   so the resident format is a repack-time choice, decoupled from the file.
2. **compressed-tensors `W4A16 group_size=128 symmetric` is the same
   quantization as `Q4_KO`**: int4, one scale per 128 along K. Affine per-128
   `(scale, min)` represents symmetric per-128 exactly via `min = −8·scale`,
   codes shifted `+8` — which is the offset the pack-quantized format already
   stores.
3. `quantize_ko`'s min-max observer would re-round any 128-group whose codes
   don't span the grid, so the import packs the **source's own codes and
   scales** via `pack_q4_ko` (added beside `quantize_ko`; the layout stays
   owned by one file). BF16 scales in the f16 normal range store exactly; the
   converter refuses one that would round.

`qwen4exp/convert.rs` rewrites the pinned Q8_0 split into a `Q4KOEXP` sibling:
expert-free shards hard-linked, expert shards re-emitted streaming with every
`ffn_{gate,up,down}_exps` converted — and **bit-exactness asserted per
expert** (`dequant_ko` ≡ the AWQ release's own dequantization) during the
conversion, so a green run *is* the proof. Source pinned:
`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @
`0939125b929543a783ce700c90e36dd1a575c00c` (shards 2–5 only, 73 GB — shard 1
is the BF16 PLE table and is never fetched). Packing semantics pinned from
`compressed-tensors`' `pack_to_int32`: 8 nibbles per i32, little-endian nibble
order, element `i` at bits `[4i, 4i+4)`.

The converted split loads through the standard `GgufModel` path; the oracle's
disk reader dispatches KO tensors through `dequant_ko` (the lane-major layout
has no CPU block codec, deliberately). The engine loads the same file
pre-repacked, exactly as `prepare_ko_gguf`'s MXFP4_KO output does. That
release also carries an **MTP head** (`model_mtp.safetensors`, 5.2 GB) the
GGUF conversion lacks — recorded for the speculation phase.

*The repack is a GPU kernel* (`simple/w4a16_repack.cu`, per §0.4): one fused
launch over every expert of a tensor — a pure byte permutation (each thread
emits one aligned u32 of a chunk's ql plane from two 16-bit nibble quads) plus
the f16 `(scale, min)` stores, with an atomic counter standing in for the CPU
path's per-group f16-exactness bail. Harness:
`examples/w4a16_convert_bench.rs`, per-run gates = the untouched `dequant_ko`
bit-exactness check AND GPU↔CPU byte identity. Measured on the Blackwell box:
CPU 0.31 ms/expert (decode 5.2 / pack 5.8 / verify 8.6 ms per 64-expert
iter, 5.35 G codes/s) → **GPU 0.144 ms/expert including H2D+D2H** (11.4 G
codes/s, PCIe-bound at ~6 GB/s out). The converter's per-expert full-dequant
verify was over half the first cut's time and is now a codes+dm roundtrip
(`unpack_q4_ko`), equivalence pinned against the untouched `dequant_ko`.

*Executed 2026-08-30, gate green* (`test_forward_batched_oracle_q4ko_experts`):
all **73,728 expert slabs** converted with the per-expert bit-exactness assert
holding (~2 min/shard); the trunk shards fell **123 GiB → 64 GiB**
(= 121B × 4.25 bpw, as computed). The oracle on the converted split passes
×1 ≡ ×2 and segmented ≡ one-shot, completes both probe prompts ("Paris",
"oxygen"), and its last-position **argmax agrees with the Q8_0 reference**
(max |Δlogit| 3.36 across the vocab — the expected scale for an 8→4-bit
expert swap, recorded, not asserted). Two operational rules the run itself
taught: rewritten shards go **`.tmp`-then-rename** so a killed conversion can
never leave a truncated shard that resumes as "done" (that happened, and
surfaced 200 GB later as a load error); and the HF cache needs a
`refs/<sha>` entry per pinned revision or `hf_get` silently walks past a
fully-populated cache onto the network.

## 13. Performance baseline — 2026-08-31

Everything the optimisation work is measured against, taken **after** Phases
2/3/4/5 and the threshold derivation, so a later run diffs against a state
where correctness is settled. RTX PRO 5000 Blackwell (sm_120, 72 GB); merged
Q4KOEXP artifact; `Int8Mode::auto` = Precision; `QWEN4EXP_KV_FACTORS`
k 1.8 / v 3.0.

### 13.1 Throughput — the §11 gate, production build

| KV | ctx | bulk t/s | single t/s | ratio | peak tokens |
|---|---|---|---|---|---|
| BF16 | 1 (cold) | 535.5 | 20.2 | — | 713 |
| BF16 | 4 | 1522.4 | 65.3 | — | 2894 |
| BF16 | 8 | 1634.4 | 99.4 | — | 5748 |
| BF16 | 16 | 1615.0 | 153.5 | — | 11482 |
| BF16 | 1 (warm) | 1368.9 | 24.6 | — | 713 |
| C0 | 2 | 1692.8 | 40.5 | 2.30× | 1466 |
| **C5** | 2 | 1630.0 | 40.4 | **4.25×** | 1466 |
| C8 | 2 | 1657.8 | 39.3 | 5.62× | 1466 |
| C10 | 2 | 1665.6 | 39.9 | 7.36× | 1466 |
| C10 | 8 | 1645.8 | 101.3 | 7.38× | 5748 |

Run-to-run spread on this machine is ±8% at the middle widths (WDDM), so treat
a single figure as evidence only outside that band.

### 13.2 Where the time goes — profile build, at QSA depth

`test_engine_qsa_at_depth --features profile`: one 2,434-token prefill plus 7
decode steps, 48 layers × 8 forwards. Summed span time 3,084.6 ms.

**The percentages are shares of summed span time, not of wall clock, and the
`hc_mix:*` rows are NESTED inside `q4e:gr_pre` / `q4e:gr_pre_ffn`** — they are
that span's internal breakdown, not four more line items. Summing the column
without knowing that double-counts the Gated Residual.

| span | ms | share | calls | |
|---|---|---|---|---|
| `fwd_routing_wait` | 953.7 | **30.9%** | 384 | the per-layer routing readback |
| `submit_roundtrip` | 456.5 | **14.8%** | 384 | |
| `q4e:gr_pre` | 240.9 | 7.8% | 384 | GR attn pre-mix |
| `q4e:gr_pre_ffn` | 192.0 | 6.2% | 384 | GR ffn pre-mix |
| `dn:mix` | 157.9 | 5.1% | 288 | DeltaNet |
| `prefill:kernel` | 123.3 | 4.0% | 12 | attention, prefill |
| `dn:proj` | 105.0 | 3.4% | 288 | |
| `q4e:ple` | 75.5 | 2.4% | 8 | |
| `q4e:qsa_select` | 58.4 | 1.9% | 96 | indexer + selection |
| `qmatmul_q8` | 53.3 | 1.7% | 2792 | |
| `decode:kernel` | 47.7 | **1.5%** | 84 | attention, decode |
| `q4e:gr_combine_ffn` | 40.0 | 1.3% | 384 | |
| `q4e:gr_combine` | 39.7 | 1.3% | 384 | |
| `dn:out_proj` | 31.5 | 1.0% | 288 | |
| ↳ `hc_mix:lowrank` | 197.2 | *6.4%* | 776 | nested in the two pre-mixes |
| ↳ `hc_mix:norm` | 87.2 | *2.8%* | 776 | nested |
| ↳ `hc_mix:gate_mean` | 75.0 | *2.4%* | 776 | nested |
| ↳ `hc_mix:inject` | 62.3 | *2.0%* | 776 | nested |

Un-nested, the Gated Residual is **16.6%** of summed spans (7.8 + 6.2 + 1.3 +
1.3) and `hc_mix:*` accounts for 13.6 of the 14.0 inside its two pre-mixes.

Decode, timed on its own in the same run: **64–69 ms/step** (14.5–15.6 tok/s)
at the QSA-capped read. Prefill of 2,434 tokens: 1.6–1.8 s.

### 13.3 Component baselines

| | |
|---|---|
| MoE dispatch | **host (readback)** — the grid is ~80% resident, so this is the sanctioned invariant-3 exception, not a fallback |
| Expert cache | hit 48.3% (cold) → 98.5% (warm), **0 cold loads**, 53,647 MiB resident |
| PLE row cache | 529,856 lookups, **97.8% hit**, 11,615 misses, **0 evictions** |
| `moe_bucketize` @ 512 | 35.1 µs (decode ×16), 426.9 µs (prefill 2048), 872.5 µs (prefill 4096) |
| QSA selection | 96 selection builds over the depth run, 1.9% of summed spans |

### 13.4 Optimisation run log

One row per optimisation run, measured the same way each time: the §11 gate
(production build) for throughput, `test_engine_qsa_at_depth --features profile`
for span shares. A change that cannot be seen in either is recorded as such —
"correct but invisible" is a result, and hiding it is how a backlog fills with
work nobody can justify.

| run | change | gate bulk ×16 | gate single ×16 | `q4e:qsa_select` | verdict |
|---|---|---|---|---|---|
| baseline | — | 1615.0 | 153.5 | 58.4 ms | §13.1/13.2 |
| Tier A | transpose copy dropped; `cat` only on block completion; `zeros`→`empty` | 1659.3 | 153.7 | 58.5 ms | **no measurable change** |
| GR fusion (first cut) | three kernels, but with a hand-rolled sigmoid | 1789.8 | 163.3 | 57.1 ms | fast, but **changed the arithmetic** |
| **GR fusion (faithful)** | same kernels using `fast_exp::sigmoid` | **1759.7** | **164.4** | 57.1 ms | **+6–12% bulk, +7–18% decode, KV row unchanged** |
| MoE scatter defines its target | kernel seeds from 0; four call sites allocate uninitialised | 1775.5 | 164.8 | — | **correct but invisible at gate depth** — see below |
| **inject stacked under down** | one GEMM instead of two over the same operand | 1768.2 | **179.8** | — | **+9–21% decode on every rung**, bulk flat |
| DeltaNet + attention stacked, split by `contiguous` | 4→1 and 3→1 GEMMs, one compaction per part | 1775.5 | 183.6 | — | +2% end to end, but `prefill:qkv_proj` +15.8% |
| **…split by one ragged scatter** | `rows_scatter` for the whole split, descriptor in kernel params | **1829.5** | **186.1** | — | `dn:proj` **below** the unstacked baseline |

**The projection groups, and the three things that were wrong on the way.**
`[Q|K|V]·z·β·α` and `q·k·v` are each one contraction against one activation, so
their weights row-concatenate (before the KO repack — see
`QuantDeltaNetWeights::proj`) and the layer issues one GEMM instead of four or
three. The split that follows is where all the difficulty was:

1. **A `contiguous` per part** cost what the stacking saved. `dn:proj` came out
   flat (106.4 → 107.5 ms) and `prefill:qkv_proj` lost outright (22.1 → 25.6 ms).
   The justification in the code had been measured once at `[t, 320]` — 9 µs —
   and reused unmeasured at `[t, 10240]` and `[t, 12288]`, which is §0.4 rule 4's
   warning about geometry applied to a copy instead of a kernel.
2. **One ragged scatter** (`rows_scatter`, invariant 2b's descriptor table) does
   every part in one launch into a single arena bump. But that kernel was built
   for the gallery append — *"the runs are small"* — and at 40M elements it was
   **437 GB/s, 38% slower than the copies it replaced**. Three fixes took it to
   1057 GB/s: a 64-bit division per element removed by giving the row its own
   grid axis, `uint4` decided per run, and the grid sized on both extents.
3. **The descriptor was the real cost, twice over.** `ncu` put the fixed kernel
   at 54.8 µs and **0.73% SM / 1.51% DRAM** at decode width — every block
   stalling on an uncached PCIe read of its six descriptor words. Carrying the
   table in kernel parameters fixed the device side; the host side was worse
   still, because `desc::scope` takes a **global mutex** and a wave opens one per
   layer. Skipping the staging entirely (`rows_scatter_inline`) took `dn:proj`
   from 124.6 ms to **101.1 ms — below the 106.4 ms it cost unstacked**.

Neither the harness nor the span profile could have found #3; only `ncu` did.
And the span profile actively misled once: `dn:proj` flat plus
`prefill:qkv_proj` +15.8% read as "revert this", while the ladder said +2%. The
ladder is the outcome and the spans are an explanation of it — that ordering
holds even when the spans look damning.

**The inject merge, and what it actually bought.** `down [low_rank, hc_dim]` and
`inject [hc, hc_dim]` are the same contraction against the same operand, so
stacking them once at load makes one GEMM whose `N` grows from 320 to 324.
11/11 rungs valid, every compression ratio held (C0 2.29×, C5 4.24×, C8 5.63×,
C10 7.43×), and the oracle agrees with the engine (`×1 ≡ ×2` 2.22e-5,
`segmented ≡ one-shot` 2.23e-5, argmax unchanged).

Single-session throughput rose on **all ten** rungs, in a 9–21% band:

| rung | before | after | |
|---|---|---|---|
| BF16 ×1 warm | 27.7 | **33.5** | +20.9% |
| BF16 ×8 | 108.1 | **122.3** | +13.1% |
| BF16 ×16 | 164.8 | **179.8** | +9.1% |
| C10 ×8 | 109.8 | **126.1** | +14.8% |

Bulk was flat. Ten of ten in one direction is well outside the ±8% band, and
the *shape* names the mechanism: **this is launch count, not GEMM work.** There
are 95 injecting hc-modules per forward (48 layers × 2, less the head), so the
merge removes 95 launches per step; at ×16 the step goes 6.07 → 5.56 ms, which
is ~5.3 µs per launch — this box's WDDM launch overhead. Prefill is
throughput-bound on a GEMM `ncu` puts at 49.5% SM and 13.5% DRAM, so the same
change cannot show there.

That distinction is the transferable part. The harness measured −198 µs/call at
2,048 tokens and the model gained nothing at prefill; the model gained 9–21% at
decode, where the GEMM does almost no work and the *launch* is nearly the whole
cost. **A kernel-level harness cannot see a launch-count win, and a gate at
prefill width cannot either.**

The one copy it costs is stated in `hc_mix` rather than hidden: splitting the
stacked output leaves both halves strided, candle's matmul refuses a strided
operand outright, so the gate half is compacted — 9 µs at 2,048 tokens against
215 µs saved. Two ways of avoiding even that were rejected as worse: padding
`up` with zero columns to swallow the stride, and giving `gr_combine` a stride
argument. Each bends a shared component to fit one model's weight layout, which
is the altitude failure the copy is not.

**The MoE scatter change, measured: +0.9% bulk / +0.2% decode at ×16, which is
nothing.** The gate's own run-to-run spread is ±8%, so both figures sit well
inside the noise, and the honest reading is that this gate cannot see the change.
All 11 rungs stayed valid and every compression ratio held exactly (C5 4.25×,
C10 7.41×), which is the result that mattered: the change is to *what the kernel
reads*, and a wrong answer would have shown as a broken rung rather than a slow
one.

It is invisible for a reason that arithmetic gives in advance, and the estimate
should have been made before the run rather than after it. The saving is two
passes over `[num_tokens, hidden]` per MoE layer, and `num_tokens` at this gate
is the *cohort width* on a decode step — 16 rows, 80 KiB, a rounding error — and
the prompt length on a prefill step, ~700 rows for ~336 MiB across 48 layers, or
roughly 200 µs against a 1.3 s prefill. Neither is measurable here.

Where it earns its place is the prefill widths the §13.5 harnesses actually run
at: at 2,048–4,096 tokens the same two passes are 10–20 MiB per layer, and a
memset alone was measured at 54 µs per 80 MiB on this card. It is also owed on
invariant grounds regardless of the number — a buffer a kernel fully overwrites
may not be zeroed (invariant 6), and the kernel-side read it removes was pure
waste at every width.

**The GR fusion, in full.** Every gate row rose on both metrics — and unlike
Tier A the *shape* is right, because these kernels sit on the prefill and
decode paths alike. Cold ×1 decode +23%, warm ×1 bulk +18%, ×16 bulk +11% /
decode +6%. At depth: prefill 1.7 s → **1.4 s**, decode 60.2 → **50.6 ms/step**,
summed spans 3146.9 → **2674.1 ms**.

Per-span, which is where the design shows through:

| span | before | after | |
|---|---|---|---|
| `hc_mix:norm` | 89.2 ms | **31.4 ms** | −65% — one pass instead of `rms_norm` + a broadcast multiply |
| `hc_mix:gate_mean` | 78.4 ms | **21.9 ms** | −72% — sigmoid, multiply and collapse in one launch |
| `q4e:gr_combine{,_ffn}` | 79.3 ms | ~40 ms | −50% — one read and one write, not four passes |
| `hc_mix:lowrank` | 205.3 ms | 190.1 ms | −7% — **as predicted: this is the two GEMMs, and fusion cannot touch them** |

That last row is the useful one. The prize was estimated at 7–9% of the
profile precisely because `lowrank` was known to be real GEMM work rather than
launch overhead, and it moved 7% while its neighbours moved 65–72%. The
estimate held.

**The KV re-derivation this appeared to need, and did not.** C10×2 went red on
the first fused gate. The row was retightened (k 1.8 → 1.65, v 3.0 → 2.8),
which restored the ladder at the cost of 1.6% of ratio **at C5, the level
production runs** — paying at the operating point to fix a probe.

That was the wrong diagnosis and the wrong currency. The cause was not
reassociation: the first kernels rolled their own `1/(1 + __expf(-x))` where
the eager path calls `fast_exp::sigmoid`, a cubic polynomial with ~0.009%
error. The fused path was therefore **~400× more accurate** than the production
path it replaced — a ~9e-5 relative change on every gate value, orders of
magnitude above any last-ulp effect. Switching to the shared primitive put the
arithmetic back, and the row held at k 1.8 / v 3.0 with **every ratio restored**
(C5 4.25×, C10 7.41×) and no measurable throughput cost.

Three things worth keeping:

- **§0.4 rule 1 applies to math primitives, not just kernels.**
  `fast_exp.cuh` exists so kernels agree on what `sigmoid` means. Rolling a
  local one silently changed the model's precision as a side effect of a
  performance change, and made both effects unattributable.
- **A GPU-vs-CPU parity test cannot police this.** candle's CPU sigmoid is a
  precise `exp` and its CUDA sigmoid is the polynomial, so the two eager paths
  already differ by ~2e-5 — a tolerance wide enough to admit that is wide
  enough to admit a changed formula. The parity tests now compare fused
  against eager **on the same device**, at 2e-6, which is where reassociation
  lives.
- **A red KV rung is a symptom, not a diagnosis.** Establish whether the
  arithmetic actually changed before spending compression on it.

Tier A's reading in full: every gate bulk row rose 2.4–7.5% and every single row
stayed flat. The bulk movement is **not** attributed to the change — these edits
touch the decode-side append and cannot affect prefill throughput, so the shape
is wrong for the cause and machine state is the likelier explanation. The one
span that *could* have shown the transpose removal did not move (58.4 → 58.5 ms),
which is consistent with the prediction on the item: 3.7 MB/wave of copy at
2.4k tokens is invisible, and the fix is insurance against 2.5 GB/wave at 100k
× 16 sequences.

What Tier A did establish is negative and useful: it removed launches from
`q4e:qsa_select` and moved it not at all, so **the QSA decode cost is not launch
count**. That span is 0.61 ms per layer-call — ~7.3 ms of a decode step across
12 attention layers, the right size for the ~12 ms/step the regression costs —
and it currently wraps three different things (`project_keys`, `append`,
`select_rows`). Splitting it into sub-spans is the next measurement.

### 13.5 The `ncu` sprint — which kernels are finished, and which are not

§0.4 rule 4 owes every kernel a harness with a per-run correctness gate, and
rule 5 owes it a profiler pass. Two harnesses now exist, both sized past this
card's 96 MiB L2 and both gating fused output against the reference before they
time anything:

| harness | example | gate |
|---|---|---|
| `qwen4exp::hyper::bench` | `gr_hyper_bench` | fused vs **eager on the same device**, 2e-6 |
| `delta_net::bench` | `delta_net_bench` | fused CUDA vs the tensor-op reference, 2e-4 |

The Gated-Residual gate is device-vs-device deliberately, for the reason §13.4
records: a CPU reference cannot police these kernels because candle's two
sigmoids already disagree by ~2e-5.

**The Gated Residual is finished.** All three kernels run at the memory floor,
so the 2.2–3.9× the fusion won is the whole prize and no further kernel work
exists on them:

| kernel | duration | DRAM | SM | occupancy (theor / ach) |
|---|---|---|---|---|
| `gr_norm` | 98.7 µs | **89.5%** | 15.0% | 100% / 84.0% |
| `gr_mix` | 148.0 µs | **93.1%** | 20.8% | 100% / 84.7% |
| `gr_combine` | 119.8 µs | **90.2%** | 5.2% | 100% / 81.5% |

(2,048 tokens × 4 streams × 2,560. Low SM throughput is the correct reading for
a kernel at 90% of DRAM, not a finding.)

What the harness did surface is **outside** the kernels. Each call allocates its
own output, and at this width that allocation is a quarter of the call:

| | 2,048 tokens | 4,096 tokens |
|---|---|---|
| `gr_norm` kernel (`ncu`) | 98.7 µs | ~197 µs |
| `Tensor::empty`, wide `[T, 10240]` | **37.0 µs** | **~75 µs** |
| `Tensor::zeros`, same shape | 91.4 µs | — |

Two things follow. The `empty`/`zeros` gap is one memset over the wide residual
— **54 µs**, hot-path invariant 6 priced at this geometry. And the 37 µs is a
synchronising call that cannot overlap the previous launch, so it is serialised
in front of every one of the Gated Residual's calls per sub-block.

**DeltaNet's scan is the opposite: nowhere near any hardware limit.** At 4,096
tokens over 2 sequences `dn:mix` is 7,047 µs, and it divides sharply:

| kernel | share | duration | DRAM | SM | occupancy | what caps it |
|---|---|---|---|---|---|---|
| `..._state_f32` | **71%** | ~5,000 µs | **6.6%** | 59.3% | 33.3% / 30.3% | 96 registers **and** 41 KB smem — both give exactly 2 blocks/SM |
| `..._intra_f32` | 22% | 1,550 µs | 22.7% | 37.5% | **16.7%** / 16.6% | 83.2 KB smem → **1 block/SM** |
| `..._conv_prefill_f32` | 6% | 427 µs | 54.9% | 69.5% | — / 73.2% | balanced; the closest to done |

The state kernel's 59% "Compute (SM) Throughput" is the **L1/shared pipe**, not
the FMA pipe — `Mem Pipes Busy` 59.3% and `L1/TEX Cache Throughput` 60.9% agree,
against 6.6% of DRAM. So it is shared-memory-bandwidth bound while running at a
third of occupancy: 3.46 active and **0.33 eligible** warps per scheduler out of
a possible 12, with 47% of stall cycles waiting on an L1TEX scoreboard
dependency. There is no latency-hiding parallelism resident to cover its own
shared-memory traffic.

Both caps are structural and both are named in the source. `intra` holds `sk`,
`sq` and `sA` for a whole 64-token chunk — 83.2 KB, one block per SM — and its
64-entry `xr[]` register array (the fully-unrolled forward substitution, which
must stay in registers or spill to local) is what independently puts it at 96
registers. `state` was already sized to ~41 KB *precisely so two blocks fit*, so
its smem cap and its register cap now land on the same number: relieving either
one alone moves its occupancy not at all.

**The MoE combine target was zeroed for a reason that had already gone.**
Chasing the memset above through the model found the real one. Every MoE layer
allocated its combine target with a `zeros`, and `deterministic_scatter` opened
each output element with `float sum = scatter_load(dst[col])` — so per layer,
per forward, the model paid a full memset over its largest transient *and* the
kernel paid a full read of it, to add a value that was always zero.

The seed was load-bearing once: the callers ran the scatter twice, for cache
hits and then for newly-loaded experts. That shape was removed when
residency-dependent grouping turned out to make decode non-deterministic
(`docs/deepseek/deepseek_decode_reproducibility.md`) and the callers merged into one
canonically-ordered pass — but the accumulate-from-target seed outlived it, and
with it every caller's obligation to zero.

The kernel now seeds from `0.f`, which is sound because it *defines* its target:
the grid is one block per token and the column loop strides the whole row, so
every `(token, column)` is stored exactly once, and a token with no
contributions stores an explicit zero rather than relying on the absent memset.
All four call sites — `expert_lre`'s pipeline and inline arms,
`quantized_qwen3_moe`, and `latent_moe`'s dspark experts — allocate the target
uninitialised. The one path that does not reach the scatter, nothing routed at
all, zeroes explicitly and says why. `wave_zeros` had no other caller and is
gone; `wave_empty_ticketed` is its replacement for the pipeline thread.

The saving is two full passes over `[tokens, hidden]` per MoE layer per forward.
The gate is
`cuda_deterministic_scatter_defines_its_target_rather_than_accumulating`, and it
is built the only way this contract can be tested: **the target is filled with
poison, not zeros.** A test that pre-zeroes passes under both kernels and would
have caught nothing. Reverting the seed makes it fail with `got 1000002, want 2`
— the poison, exactly.

### 13.6 What this baseline does not cover

- **Decode-only profile.** §13.2 is one prefill plus seven decode steps, so
  prefill-shaped spans are over-represented in the share column. A
  decode-only capture is the missing companion.
- **Profile-build inflation.** §13.2's absolute times are from a
  `--features profile` build and are not production times; §13.1 is.
- **Depth beyond the KV row's derivation.** §13.1's C-rungs run at ~713
  tokens, where QSA's selection is the identity.

## 14. Speculative decode — the head exists upstream and the GGUF does not carry it

**Status: blocked on weights, not on code.** The engine reports `draft budget 0`
on every gate rung because `ManagedBatchedModel::draft_budget` defaults to zero
and `qwen4exp` does not override it — it has no drafter. The reason is not an
oversight, and the reason matters, because it decides what unblocks it.

### 14.1 What was checked

`the_checkpoint_draft_head_inventory` (in `quantized_qwen38_moe`) reads the
weights rather than reasoning about them, so this is a **checked** no:

| source | result |
|---|---|
| merged Q4KOEXP engine artifact | `block_count = 48`, highest `blk.N` = 48, **0** `nextn`/`mtp` tensors |
| pinned upstream Q8_0 split (`unsloth/…`, rev `c8b5954a`) | 1,224 tensors, **0** `nextn`/`mtp` tensors, highest `blk.N` = 48 |

So the GGUF lineage this engine runs has no draft head anywhere.

### 14.2 But the released model has one

`Qwen/Qwen3.8-Flash-Next`'s own `config.json` declares it:

```json
"mtp": { "hybrid": true, "layer_types": ["full_attention"], "num_hidden_layers": 1 },
"mtp_num_hidden_layers": 1,
"mtp_use_dedicated_embeddings": false
```

and `model.safetensors.index.json` lists the tensors — `mtp.fc_embedding`,
`mtp.fc_hidden`, `mtp.hyper_connection_mixer.*`, `mtp.layers.0.self_attn.*`
(including its own QSA `indexer`), `mtp.layers.0.mlp.experts.*`. The GGUF
conversion dropped them; the weights are published.

`mtp_use_dedicated_embeddings: false` is the same arrangement
`qwen35::mtp` already documents and implements — the head reuses the trunk's
embedding and `lm_head`, which is what makes a NextN head cost one block instead
of a second model.

### 14.3 Why it is not an evening's work

Three things are larger than they look:

1. **The weights have to be fetched and converted.** The MTP tensors are spread
   across ~30 of the release's 131 safetensors shards (sharding is by size, not
   by module), so "download the head" means either the whole 131-shard release
   or a range-fetcher that parses each shard's header and pulls only the MTP byte
   spans. The head itself is ~4–5 GB in bf16 — it carries a **512-expert MoE**
   (`num_experts` is inherited, not overridden). Then it needs converting to the
   sidecar GGUF `Qwen35LoadOptions::mtp_path` already knows how to load.
2. **This head is not qwen35's head.** qwen35's MTP block is a plain transformer
   block. This one is a whole `qwen4exp` layer: Gated-Residual hyper-connections
   at both sub-blocks, a 512-expert MoE, and its own QSA indexer. Implementing it
   means a one-layer qwen4exp forward with a second expert-streaming tenant, not
   reusing `qwen35::mtp` as-is.
3. **The rewind is three recurrences here, not one.** `qwen35::spec` replays the
   DeltaNet mixer over the accepted prefix from the ping-pong's untouched half.
   `qwen4exp` additionally carries the **PLE conv history** and the **per-layer
   QSA `IndexCache`**, both of which advance per token and neither of which the
   KV can reconstruct. `can_rewind_speculative_block` may not be turned on until
   all three rewind, because declaring it with one of them stale is a silent
   wrong answer rather than a failure.

### 14.4 What does not work, and why — so it is not re-tried

- **A generic text/n-gram proposer** is refused by design, and the refusal is
  written into `ManagedBatchedModel::speculative_draft`: it "can only re-propose
  what the sequence already said, which is worth nothing on the reasoning and
  first-draft tokens a decode loop actually spends its time on".
- **Self-drafting with reduced MoE routing** (top-1 draft, top-10 verify) does
  not pay, by the profile's own arithmetic: MoE-attributable spans are ~50% of a
  step (`fwd_routing_wait` 33.1% + `submit_roundtrip` 16.6%), so cutting 10
  experts to 1 saves at most ~20% of a step. A draft token costing ~0.8 of a real
  one needs ~5 accepted per draft to break even.
- **A small-Qwen cross-model drafter** founders on the vocabulary: this release
  is `Qwen4ExpForConditionalGeneration` with `vocab_size` and special tokens
  (`bos`/`eos` 248044, `image_token_id` 248056) that no Qwen3/3.5 sibling in the
  local cache shares.

### 14.5 The head's KV layer, and why an unwritten one is not free

The head is block `num_layers` and needs a KV layer of its own: it is
full-attention by declaration, attends over the sequence's history at the
sequence's own length, and a drafted position that later becomes a real one must
find its keys where the verify wrote them. So the session allocates
`n_attention_layers() + num_mtp_layers` — 13, not 12.

Adding that layer, with the head not yet stepping it, changed the smoke
continuation from `" Paris. The capital of Germany is Berlin. The capital of"`
to JSON fragments. The head degraded quality without ever running, and the test
passed throughout because it only checked that the output contained `"Paris"` —
a token produced by prefill, before the damage lands.

**The mechanism is a stream-wide aggregate reading a layer nothing writes.**
`BatchedInferenceSession::sequence_backing_tokens` reports the token prefix as
the **minimum across layers**, which is right for the skew it was built for — a
windowed creep prefill advances layers incrementally and they converge. A layer
no pass ever writes does not converge: the minimum is pinned at 0 for the life
of the sequence. `wave_driver::reconcile_entry_offsets` then compares the
sequence's offset against it at every wave entry, finds `backing < offset`,
reads that as "offset ran ahead of the backing" and **clamps the offset to 0**.
The trunk re-attends from the start of its history on every step. Nothing
faults; the only trace is a `tracing::warn!` no test captures.

`build_decode_metadata` had the same shape of bug: its default group is every
layer, and a group is reconciled to a common block structure
(`unify_decode_layout`), so a permanently-short layer drags that repair across
the whole group on every decode step, moving the trunk's own write slices.

**The fix is `KvLayers { stream, draft }`.** Stream layers step together on
every wave and are the only set a stream-wide aggregate covers; a draft layer is
allocated so the head has somewhere to write, and excluded so it cannot speak
for layers that are being written. Blast radius is exactly this model — every
other stack is `stream_only(num_layers)`, and its aggregates are unchanged.
With the fix the 13th layer is genuinely free: 27.2 t/s against 26.4 at twelve.

**`draft` is the bring-up state, not the destination.** qwen35 keeps its head's
layer as a *stream* layer deliberately, and `qwen35::draft`'s module docs argue
why: `head_wave_pass` runs the head over the same rows at the same positions in
the same wave, so its layer always stands at the same length as its siblings,
`HeadWave::kv_layer` indexes straight into the wave's own metadata, and every
session-wide operation that assumes "a sequence's layers describe one stream at
one length" — fork, view, prefix injection, turn sealing, truncation — keeps
working without knowing the head exists. Marking qwen35's layer `draft` would
have broken its head pass. This model moves to `stream_only` when its own head
steps in lockstep.

The smoke now asserts the **exact twelve token ids**, not a substring: decode
here is greedy and deterministic, so the continuation is a fixed expected value,
and a substring probe is what hid this for a full session.

### 14.6 End of turn — and why `eos_token_id` is not it

The engine terminates correctly. A chat turn asking for a one-word answer
returns a single token (`"Paris"`, id 57590) and stops on `<|im_end|>`
(`test_engine_stops_on_end_of_turn`).

**The trap is that this checkpoint's `eos_token_id` is not its turn stop.**
`config.json` declares `bos_token_id == eos_token_id == 248044`, which is
`<|endoftext|>` — a *document* terminator. An assistant turn ends with
`<|im_end|>`, id **248046**, which appears nowhere in the config. A stop list
built from `eos_token_id` therefore never fires on a chat turn, and the symptom
is a model that appears not to stop while being entirely healthy. The
conversation layer is already immune: `ModelBuilder::engine_config` resolves stop
tokens **by name** from the tokenizer (`<|im_end|>`, `<|endoftext|>`,
`<|end_of_text|>`, …), so it picks up both ids without knowing this model. Any
new path that needs stop tokens must do the same rather than read the config
scalar.

`ple.eos_token_id` is a separate concern and is correct at 248044: it is the
n-gram hash-window reset (§6), where a document boundary is exactly the intended
cut, and it is read from the checkpoint rather than assumed.

Two properties this gate needs that are easy to get wrong, since either failure
looks like "the model won't stop":

- **Thinking must be closed.** The checkpoint carries `<think>`/`</think>`
  (248068/248069). The prompt uses Qwen3's non-thinking form — an assistant turn
  pre-filled with an already-closed think block — or the model reasons for
  hundreds of tokens first.
- **A completion prompt has no end.** `test_engine_wave_paris_smoke` prompts
  with the bare prefix `"The capital of France is"` and runs a fixed 11
  iterations; continuing into "The capital of Germany is Berlin" is the correct
  response to that input. Termination cannot be observed there at all, which is
  why it is a separate test rather than an assertion added to that one.

### 14.7 How the head is wired, and the one thing measurement could not settle

The head is `blk.48`, a full routed block of this architecture, plus its
`nextn` tensors — three input tensors and one hyper-connection mixer. Their
widths are what assign the roles, because `[hc_dim]` and `[n_embd]` are
different *kinds* of norm in this stack:

```
nextn.enorm        [2560]   plain RMSNorm over the token embedding
nextn.hnorm        [10240]  grouped norm (per-stream reduction, flat gain) over the wide residual
nextn.eh_proj      [2560, 5120]   fuses fc_embedding | fc_hidden side by side
nextn.hc_head_*    {norm, down, up}, no inject — the head's own OUTPUT mix
```

So: `hnorm` the wide residual, run `eh_proj` **per hyper-connection stream**,
run the block, collapse its output through the head's own mixer, and score with
the trunk's `lm_head`. The mixer **is** the output norm — `hc_mix` opens with the
grouped norm over `nextn.hc_head_norm` — so nothing precedes it, exactly as
nothing precedes the trunk's `output_hc_*`. That is the reference graph's wiring
(llama.cpp #28243, `graph_mtp`: *"the final mixer is the output norm: there is no
separate one"*), and the pinned head file (`MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`
at `38bb39ee`) publishes exactly these tensors under exactly these names, with no
`shared_head_norm`.

**`eh_proj` runs per stream, and this is the one part reasoning got wrong.**
The natural-looking assembly collapses the wide residual to `n_embd`, projects,
and broadcasts the result back across the streams the way a trunk block enters
from a token embedding. That type-checks, runs, and produces fluent proposals —
and they are never the trunk's. Three wirings were built and measured before the
reference settled it (llama.cpp PR #27836): *"the combiner must be run per
hyper-connection stream on the wide hidden state; if you do mean pooling first,
the acceptance rate drops catastrophically."* The head folds its embedding
**into** the trunk's carried streams rather than starting fresh ones, which is
what makes a proposal conditional on the state the trunk actually built.

The same source confirms the two conventions the shapes cannot check: the concat
is `[e ; h]` (measured strictly worse reversed), and the head's mixer is its
*output* mix.

### 14.8 The rewind: three recurrences, one that replays

A speculative block runs `k+1` rows and then learns how many the model agrees
with. The paged K/V truncates to the accepted prefix exactly; the three carried
states cannot be truncated at all, and each is silent when it is wrong.

| Carried state | Why truncation cannot reach it | How it rewinds |
|---|---|---|
| GDN `S` | a running sum with no per-token decomposition | replay the mixer over the accepted rows from the entering state |
| PLE conv history | a sliding window of *derived* rows, not of tokens | slice the window the accepted rows would have left |
| QSA index cache | pooled block keys, appended as blocks complete | restore the entering snapshot, re-append the accepted rows' keys |

Only the GDN half replays arithmetic, through `replay_accepted_prefixes` — the
same function the hybrid runs, generalized to take the four per-layer constants
plus `eps` and the device rather than a `QuantModel`, so one implementation
serves both. The other two are bookkeeping over rows the wave already computed:
the PLE window is a `narrow`, and the index cache re-appends through the same
`IndexCache::append` the wave called, because a cache rebuilt by different
arithmetic is a selection that drifts from the wave's.

**The entering state costs nothing.** All three are already snapshotted at wave
entry for the failure bracket that rolls a failed wave back; a verify wave keeps
those snapshots instead of dropping them. What a rewind adds is the block's own
per-row operands, because the wave arena reclaims them when the forward ends —
so a verifying span stashes them as it goes, and `ple_apply` grew a `capture`
parameter for exactly the rows nothing else could reconstruct.

Three details that are easy to get wrong and produce no error:

- **A verify block is a prefill span**, because all three recurrences are
  sequential within a sequence — two rows of one sequence cannot decode in
  parallel against a single carried state. A prefill normally scores only its
  last row; a verifying one must score **every** row, since each position is a
  proposal checked against the prediction before it.
- **The head's own index cache rewinds with the trunk's.** It runs the same rows
  in the same wave, so rewinding twelve caches and not the thirteenth leaves the
  rejected tokens' keys steering the head's selection for the rest of the
  sequence.
- **A draft walk is not a forward and must be bracketed like one**
  (`plan_wave_transient` + `begin_forward`), or the head's attention writes into
  ground no tier owns — hot-path invariant 7, surfacing as an illegal access
  inside the out-projection rather than anywhere near the cause. Two neighbours
  of the same kind: the head's index cache must be grown for the whole walk
  before it starts, and the slot-header index is **group-relative**, so a walk
  that builds a one-layer group passes `0` rather than the absolute KV layer.

### 14.9 Measuring it — two tests, one variable each

Speculation has two independent axes and conflating them produces a table that
cannot be read, so there is a test per axis and they share one ladder
definition:

| test | varies | holds | answers |
|---|---|---|---|
| `test_parallel_batched_forwarding` | the KV rungs | no speculation | the standing calibration gate |
| `test_speculative_ladder` | the KV rungs | the **production** budget | what speculation is worth at each operating point |
| `test_speculative_decode` | the draft depth | widths 1 and 4 | where depth stops paying — how the brackets were derived |

`test_speculative_ladder` runs **every** rung C0–C10, not the calibration
gate's C0/C5/C8/C10 sample. The plain gate can sample because its four points
pin `QWEN4EXP_KV_FACTORS` and the levels between them are bounded by the ends.
Acceptance is not interpolable that way: it depends on how far the compressed
K/V has moved the *target's own* argmaxes, so a level where the drafter and the
target stop agreeing shows up only at that level.

It pins no budget. The harness's default is `DraftBudget::Adaptive`, which asks
the model at each config's width — production behaviour, and the whole point of
the table. A fixed budget would measure a configuration nothing runs, and would
hide the one thing the wide rows exist to show: `affordable_draft_budget`
clamping the depth a 32-wide cohort can stash for.

Both are measured against a budget-0 baseline taken in the same harness rather
than a figure carried over from another run.

The model is **loaded once for the whole sweep** (`TestParams::run_loaded`).
Reloading per budget spent most of a sweep's wall clock re-reading a 124 GB
artifact that never changed, and it also weakened the comparison: expert-cache
warmth and arena shape now carry across the budgets instead of being rebuilt
differently for each.

Speculation is lossless, so every budget must report the same validity — on a
rung plain decode holds. That qualification is not pedantry; see §14.10.

### 14.10 C9/C10 drift without speculation, and the gate cannot see it

C9 and C10 lose the reference around **110–130 generated tokens** under **plain
decode**. `test_top_rung_divergence_vs_budget` measures it at every draft
budget, width 2, 256 tokens: divergence at the same characters (505/506 for C9,
505/440 for C10) at budgets 0, 1, 2 and 4 alike, while C8 stays exact
throughout. Depth changes nothing and budget 0 shows it, so it is not the
drafter.

**The standing gate is too short to reach it.**
`test_parallel_batched_forwarding` generates 64 tokens — roughly 250 characters
— and the divergence starts past 440. Its C10 row passing is a true statement
about the first 64 tokens and no statement at all about the rung. Any comparison
between the two gates has to account for that: they do not generate the same
amount of text.

### 14.11a The draft head's KV layer seals at C0 — one layer of thirteen

The head is **one block with one KV layer**. A trunk spreads every read across
twelve, so a level's key error is partly averaged over the depth; the head has
no depth to average over and takes it whole. And a proposal is only worth
something if it reproduces the trunk's *argmax*, which is a far sharper test
than "the text still reads correctly" — so the drafter is the most sensitive
consumer of KV error in the stack.

`cap_layer_seal_level(mtp_kv_layer, DRAFT_HEAD_MAX_COMPRESSION)`, trunk
untouched:

| rung | uncapped | **capped C3** | acceptance | throughput |
|---|---|---|---|---|
| C6 | 2.23 / 25.0 | **9.44 / 117.2** | 2.9% → 93.0% | 4.7× |
| C7 | 2.39 / 29.0 | **9.44 / 117.8** | 4.9% → 93.0% | 4.1× |
| C8 | 2.12 / 24.8 | **9.62 / 120.8** | 1.5% → 95.3% | **4.9×** |
| C9 | 2.12 / 25.2 | **9.27 / 114.5** | 1.5% → 90.9% | 4.5× |
| C10 | 2.23 / 26.1 | **7.18 / 83.0** | 2.9% → 61.8% | 3.2× |
| C5 ×8 | 37.78 / 242.7 | **39.23 / 262.2** | 94.4% → 98.1% | 1.08× |
| C10 ×8 | 8.33 / 49.2 | **31.38 / 202.6** | 1.0% → 63.4% | **4.1×** |

C0–C3 and every BF16 row are byte-unchanged — the check that the ceiling binds
only where it should, rather than the numbers merely coming out higher.

**C3 measures at or above C0 on every rung above C7**, so the cheaper cap is
also the better one: what matters is staying above the sub-3-bit K formats, not
reaching near-lossless. Worth having run rather than reasoned — C0 was the
defensible-sounding choice and it is not the right one.

**C10's partial recovery is the confirmation, not the exception.** C10 has the
most aggressive *trunk* compression, so the head's **input** — the trunk's
residual, computed by attending over its own twelve layers — is most degraded
there, and pinning the head's own layer cannot repair what arrives already
wrong. The two mechanisms separate cleanly: the head's own KV dominates at
C6–C9, its input dominates at C10.

This does **not** touch the C9/C10 output divergence (§14.10): that is the
trunk's own text, which the head plays no part in producing.

**A ceiling, not a setting.** `min(session, cap)`, so a session at C0–C3 keeps
its own level and only the deeper rungs are held back; the caller does not have
to know what the session was configured with, and capping *deeper* than the
session — compressing one layer harder than the stack — cannot happen by
accident. `DRAFT_HEAD_MAX_COMPRESSION = 3` because C3 is the deepest level whose
K candidate list contains no sub-3-bit format (§14.11).

**A level, not a format, and via the seal.** Two traps, both hit:
`quantize_and_seal_sequences` reads `self.compression_policy()` — the
*session's* — so a per-`KvCache` policy is a silent no-op that looks exactly
like a null result; and excluding the layer from the seal instead walks into
"re-injecting Arc-shared chunks under a lone sequence corrupted the decode
read". Pinning a level keeps the layer on the same code path as every other,
only less aggressive.

### 14.11 ROOT CAUSE: the sub-3-bit K formats do not work

`test_which_k_format_breaks_drafting` pins each candidate as the ONLY K format
and runs the gate's own StoryRewrite at C5 — so the single variable is which
format the keys land in:

| K format | bits | accepted/step | drafted-token acceptance | validity |
|---|---|---|---|---|
| `Q8_1` | 8 | 9.81 | **97.6%** | pass |
| `Q4_0` (control, in C5) | 4 | 9.27 | 90.9% | pass |
| `Q1_S` | 1 | 2.39 | 4.9% | **fail @ char 23** |
| `Q2_A` | 2 | 2.12 | 1.5% | **fail @ char 394** |
| `Q2_S` | 2 | 2.00 | **0%** | **fail @ char 357** |

A clean split at three bits. `Q8_1` is the best rung measured — C6's other
addition is sound.

**C6 is the first level whose K candidate list contains `Q1_S`, `Q2_A` and
`Q2_S`.** That one fact explains both symptoms, which look unrelated only
because the two consumers have different tolerances:

* **Acceptance collapses at once.** A drafter has to reproduce the target's
  *argmax*; a handful of poisoned key blocks is enough. The three formats
  measure 4.9%, 1.5% and 0%, which brackets the 1.5–4.9% band that C6 through
  C10 all occupy.
* **Trunk output degrades slowly.** Fluent text absorbs a few bad blocks, so it
  only loses the reference past char ~440 — which is why C6–C8 still validate
  and only C9/C10 fail at 256 tokens.

It also explains the flatness that made "graduated precision loss" untenable:
C6–C10 are not progressively worse because they are all selecting the same
broken formats, not because they are progressively lossier. Under the adaptive
policy those formats take a *minority* of blocks, which is why pinning one is so
much more violent than running the rung.

**The fix is the candidate lists, not the thresholds.** No threshold makes a
0%-acceptance format acceptable; `Q1_S`, `Q2_A` and `Q2_S` should not be K
candidates at any level. They may well be fine for V — keys are sensitive by
channel and the V side is untested here — so the change is to drop them from
`PRODUCTION_K_CANDIDATE_FORMATS` for C6–C10 and re-measure, which should recover
drafting and the C9/C10 validity together.

Still open: whether they are **broken** or merely **unusable for keys**. `Q1_S`
failing at char 23 reads as breakage; `Q2_A`/`Q2_S` at ~390 read as extreme
loss. That distinction decides kernel repair versus removal from the ladder, and
one round-trip test per format against a reference dequant settles it.

The Q0 family (`Q0`, `Q0_V`, `Q0_X`, `Q0_M2`, `Q0_M4`), which C9 admits and C8
does not, was the earlier suspect and is **not** implicated: the collapse
already exists at C6, three levels below where Q0 becomes selectable.

**What this costs the speculative claim:** nothing that was measured, and one
thing that cannot be. Speculation is exact on every rung where plain decode is
exact — C0–C8 at every budget, and BF16 at widths 1 through 32. C9 and C10
cannot be used to judge it either way, because the reference does not hold there
to begin with.

The route to establishing this was three broken controls, each worth naming
because each looked conclusive:

- a **substring test filter** that ran twelve models' gates, so another model's
  partition OOM nearly went into the report as a regression here (`--exact`);
- a run that **aborted before validating** — `validate_and_print_results` runs
  only after every config, so a grep for divergences returned zero *lines* and
  was read as zero *divergences*;
- and the **generation-length mismatch** above, which made a passing gate look
  like a control for a failing one.

## 15. References

- `docs/qwen35_qwen38_models.md` — the Qwen3.5/3.8 hybrid bring-up. §4.4a
  (output gate, partial rotary), §5 (recurrent-state turn-seal snapshots), §7.2
  (the checkpoint is the authority), §7.3 (the bring-up oracle method) apply
  here essentially unchanged.
- `docs/deepseek/deepseek_v4_flash.md`, `docs/deepseek/deepseek_batched_paged_attention_plan.md`
  — the sparse-selection attention precedent.
- `docs/deepseek/deepseek_hot_path_invariants.md` — the seven invariants any new
  hot path must satisfy.
- `docs/expert_cache_design.md` — three-tier expert streaming.
- `docs/archived/kv_tier_migration.md` — GPU/RAM/NVMe tiering, the home for the
  PLE table.
- `docs/deltanet_state_persistence.md` — the recurrent-state record §6.3 extends.
