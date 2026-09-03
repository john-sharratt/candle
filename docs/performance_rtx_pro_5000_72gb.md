# Performance — RTX PRO 5000 Blackwell 72 GB

Measured figures for every model with a batched-forward path in this tree, on
one machine, with an emphasis on **how cost behaves as the KV cache grows**.
Every number here was produced by a test in this repository; nothing is
extrapolated, and §4 — the limits — is as load-bearing as the tables.

> **Machine:** see §1 · **Branch:** `qwen38-moe`
>
> **One measurement epoch.** Every table in §3 comes from a single sequential
> sweep on one build: twelve depth gates, twelve width ladders, and the
> flagship's two depth curves, run one `cargo test` invocation at a time so
> exactly one model was ever resident.

---

## 1. The machine

Every figure below is from this one machine. None of it transfers to another
card without re-measurement — the elastic VRAM partition sizes itself from what
it finds, so both the compression ladder's headroom and the expert cache's
residency are properties of *this* 72 GB card.

| | |
|---|---|
| **GPU** | NVIDIA RTX PRO 5000 Blackwell, 72 GB GDDR7 (73,415 MiB reported) |
| **Compute capability** | 12.0 (sm_120) |
| **Max SM / memory clock** | 3,090 MHz / 14,001 MHz |
| **Driver** | 596.59 |
| **CPU** | AMD Ryzen 9 9950X3D, 16 cores / 32 threads |
| **System RAM** | 189 GB |
| **OS** | Windows 11 Pro, build 26200 |
| **GPU driver model** | WDDM (not TCC) |
| **CUDA toolkit** | 12.9 (V12.9.86) |
| **Rust** | 1.98.0 (88d9e12ae 2026-08-18) |
| **Build** | `--release --features cuda` |

Two properties of this machine shape several results and are worth stating
before the tables rather than after:

- **WDDM, not TCC.** Kernel launches carry the Windows display-driver model's
  submission overhead, which is the floor under single-session decode on every
  small model here. A Linux/TCC host would move the decode column and leave the
  prefill column roughly alone.
- **72 GB on one card.** Every model in this report except the 284B fits its
  weights resident, so the depth curves below are *not* contaminated by weight
  paging. That is the point of running them here.

---

## 2. What is being measured

### 2.1 Every rung is inside the checkpoint's own context window

This governs which rows exist at all, so it comes first.

A model asked for more positions than it was trained to address does not report
a deeper measurement of this engine. It reports whatever its rope tables
extrapolate to out there — a number that moves with the position-scaling scheme
and not with anything the cache or the kernels do. Such a row cannot be set
beside an in-window row, because the two are not measuring the same thing.

**This fleet runs native windows only. No YaRN, no position interpolation, no
scaling applied at load.** Each gate declares its checkpoint's window and
`long_context_gate` refuses any rung where `prompt + generated > window`,
before the model loads. Three unit tests pin the check, including the boundary
that actually bites: a prompt that fits with fewer than `generate` positions to
spare does not fit once decoding starts.

The window is a property of the checkpoint **as this code loads it**, which is
not always the number the file advertises:

| Checkpoint | Window used | Basis |
|---|---:|---|
| Qwen3.5-0.8B / 9B / 35B-A3B | 262,144 | `context_length`; native, no scaling involved |
| Qwen3.6-35B-A3B, Qwen3.8-27B | 262,144 | as above |
| Qwen3.8-Flash-Next | 262,144 | as above |
| Qwen3-30B-A3B-Instruct-2507 | 262,144 | as above (the 2507 release; the original is 32K) |
| DeepSeek-V4-Flash-0731 | 1,048,576 | as above |
| Qwen3-8B | 40,960 | `context_length`; 131K needs YaRN, which is not used |
| Qwen2-0.5B-Instruct | 32,768 | `context_length` |
| **Llama-3.2-3B (Nidum)** | **8,192** | **not** its declared 131,072 — see below |
| Llama-2-7B-Chat | 4,096 | `context_length`; honest, no scaling of any kind |

**Llama-3.2-3B is the case that shows why the declared number is not
sufficient.** Its GGUF advertises `context_length = 131072`, but carries no rope
scaling metadata at all — no `rope.scaling.type`, no `factor`, no
`original_context_length`. Llama 3.2 reaches 128K *only* through llama3 rope
scaling, factor 32 over a base window of 8,192. The loader reads exactly those
keys, finds nothing, and builds plain RoPE at theta = 500,000. The file
therefore declares 131,072 and supplies the machinery for 8,192, and 8,192 is
the number that describes what runs. This is measurable, not inferential: asked
for 32K, that checkpoint emits `". \n\n"` repeated to the token limit,
identically under BF16, C5 and C10.

A consequence worth stating: the deep tables below are the **Qwen3.5-and-later
checkpoints plus DeepSeek**, because those are the models whose windows reach
128K without scaling anything. The earlier-generation models are measured at 8K,
which is inside every window in the fleet.

### 2.2 The three tests

**The width ladder** (`test_parallel_batched_forwarding*`) — a ~700-token prompt
at several batch widths and several points on the KV compression ladder. The
**width** axis: what a session costs, and what concurrency buys.

**The depth gate** (`long_context_gate`) — the same batched forward with a
prompt long enough to put 4K–128K tokens in the KV cache. The **depth** axis. It
takes an explicit `DepthTask`, which decides what the model is asked to do once
it has read the padding:

- `Coherence` — answer briefly about the material just read.
- `Rewrite` — reproduce a story that follows the padding, with a character
  renamed. This is the width ladder's own task, so a `Rewrite` depth row and a
  ladder row differ **only** in context length.

The task is a parameter rather than a constant because it moves the decode
number more than most engine changes do: on the flagship the draft head accepts
4.85 tokens a step on `Rewrite` against ~2.3 on `Coherence`, and decode rate is
steps/sec × accepted/step. **Rows from different tasks are not comparable**, and
naming the task at each call site is what keeps that from being rediscovered.

### 2.3 Columns

| Column | Meaning |
|---|---|
| `prefill t/s` | Prompt tokens processed per second (the harness's `t/s (bulk)`) |
| `decode t/s` | Generated tokens per second, summed across the batch (`t/s (single)`) |
| `%Quantized` | Share of KV blocks that ended up in a quantized format |
| `Compress` | Float-equivalent bytes ÷ actual KV bytes |
| `Peak tokens` | Total tokens resident across all sessions |
| `Valid` | Output passed the config's validation mode |

> **The harness's column names are misleading and are renamed here.** `t/s
> (bulk)` is `prompt_tokens_per_sec` — prefill — and `t/s (single)` is
> `generate_tokens_per_sec` — decode. They are *not* a batched-vs-single
> distinction.

### 2.4 Instrumentation is not free

Building with `--features profile` costs **5–24% of decode** and 1–3% of
prefill: the spans fence a pipeline whose remaining cost is largely issue
overhead. Profiled builds are for **attribution** — which span dominates — and
uninstrumented builds are what the engine actually does. Every throughput figure
in this document is uninstrumented; quoting a profiled decode figure as
throughput understates it by up to a quarter.

### 2.5 The filler, and why it is not a tiled corpus

A depth measurement needs a prompt of a given length. The obvious way to build
one — tile a passage until it is long enough — would make a 128K prompt roughly
forty copies of the same text, and **two of the numbers in these tables are
compression ratios**. A cache holding forty copies of one passage compresses
like nothing real ever will.

So the filler (`batch_test/long_context.rs`) is assembled instead: paragraphs
from eight unrelated domains, walked in a rotating order so no two consecutive
paragraphs come from the same template, and *perturbed every cycle* — the names,
places and numbers differ each time a template comes round, so no two cycles
present the same token sequence. Unit tests pin all three properties.

It is still synthetic prose, and that is the honest caveat on every compression
number below: **these ratios are indicative of varied natural-language text, not
a measurement of any particular corpus.** Throughput figures are essentially
insensitive to content and carry no such caveat.

### 2.6 Prefill is chunked to the model's own width cap

Prefill honours `ManagedBatchedModel::prefill_width_cap` and submits a prompt in
cap-sized slices. A wave's transient tier is sized by its row count, so a
128K-token prompt submitted whole would ask for ~9.4 GB of transient against a
3.2 GB span.

The measurement consequence: **a chunked deep prefill figure is lower than the
same model submitting the same tokens in one wave.** The chunked path is the one
reported because it is the only configuration that spans every depth, and
because it is what the production scheduler does. Prompts below a model's cap —
every width-ladder row, and the 8K depth rows — take exactly one slice and are
unaffected.

### 2.7 The span must be sized before the weights load

The KV reservation is sized from a governor's balloon measurement of the card;
without one it falls back to a small test constant (3,170,893,824 B) with the
weight floor at its top, and a wave needing 872 MB of transient is refused on a
card with 72 GB free. `long_context_gate` calls `ensure_vram_governor` before
loading. At ladder depths nothing notices; at depth it is the difference between
a measurement and an error.

---

## 3. Results

### 3.1 The models

| Model | Params | Weights | Attention | Window | Depths measured |
|---|---|---|---|---:|---|
| Qwen2-0.5B-Instruct | 0.5B dense | Q4_0 | full | 32,768 | 8K |
| Llama-2-7B-Chat | 7B dense | Q4_0 | full, **no GQA** | 4,096 | 4K |
| Llama-3.2-3B (Nidum) | 3B dense | Q4_K_M | full (GQA) | 8,192 | 8K |
| Qwen3-8B | 8B dense | Q6_K | full (GQA) | 40,960 | 8K |
| Qwen3-30B-A3B-2507 | 30B / 3B active | Q4_K_M | full (GQA), MoE | 262,144 | 8K |
| Qwen3.5-0.8B | 0.8B dense | Q6_K | 3:1 DeltaNet hybrid | 262,144 | 32K, 128K |
| Qwen3.5-9B | 9B dense | Q6_K | 3:1 DeltaNet hybrid | 262,144 | 32K, 128K |
| Qwen3.5-35B-A3B | 35B / 3B active | Q6_K | hybrid + MoE | 262,144 | 32K, 128K |
| Qwen3.6-35B-A3B | 35B / 3B active | Q6_K | hybrid + MoE | 262,144 | 32K, 128K |
| Qwen3.8-27B | 27B dense | Q6_K/Q8 | hybrid | 262,144 | 32K, 128K |
| Qwen3.8-Flash-Next | 250B / 13B active | Q4_KOEXP | hybrid + **QSA** + MoE | 262,144 | 32K, 128K |
| DeepSeek-V4-Flash-0731 | 284B / 13B active | MXFP4_KO | native-sparse | 1,048,576 | none — §4 |

### 3.2 Depth: 32K → 128K, one context

Every row valid. `Coherence` task, 64 generated tokens.

| Model | Mode | Prefill 32K | Prefill 128K | Decode 32K | Decode 128K | Compress 128K |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next** | BF16 | 1,282.4 | **1,269.5** | 26.5 | **29.5** | — |
| | C10 | 1,341.2 | 1,277.1 | 25.6 | 23.8 | 6.97× |
| Qwen3.5-0.8B | BF16 | 7,309.1 | 1,859.0 | 157.3 | **135.6** | — |
| | C10 | 7,149.6 | 1,913.1 | 158.5 | 132.1 | 4.63× |
| Qwen3.5-9B | BF16 | 1,987.2 | 592.3 | 75.0 | 35.7 | — |
| | C10 | 2,000.8 | 603.2 | 52.1 | 18.5 | 6.37× |
| Qwen3.5-35B-A3B | BF16 | 1,692.7 | 441.4 | 61.5 | 28.7 | — |
| | C10 | 1,641.4 | 449.8 | 42.8 | 14.1 | **7.63×** |
| Qwen3.6-35B-A3B | BF16 | 1,692.8 | 443.4 | 56.8 | 27.9 | — |
| | C10 | 1,637.4 | 456.4 | 41.4 | 14.5 | 7.17× |
| Qwen3.8-27B | BF16 | 691.7 | 192.0 | 30.6 | 17.7 | — |
| | C10 | 656.7 | 196.7 | 26.0 | 11.5 | 5.44× |

C5 was also measured at 32K on every model: 2.85×–4.20× compression at 1–22% of
decode. Full rows are in the provenance file.

### 3.3 What survives a 4× depth increase

Fraction of BF16 throughput kept from 32K to 128K.

| Model | Attention | Prefill kept | Decode kept |
|---|---|---:|---:|
| **Qwen3.8-Flash-Next** | hybrid + QSA + MoE | **99%** | **111%** |
| Qwen3.8-27B | hybrid | 28% | 58% |
| Qwen3.6-35B-A3B | hybrid + MoE | 26% | 49% |
| Qwen3.5-9B | hybrid | 30% | 48% |
| Qwen3.5-35B-A3B | hybrid + MoE | 26% | 47% |
| Qwen3.5-0.8B | hybrid | 25% | **86%** |

**Flash-Next is flat.** Prefill 1,282 → 1,270 t/s and decode 26.5 → **29.5** t/s
across a 4× depth increase — decode is *higher* at 128K than at 32K. This is the
architecture's central claim and it is what the rest of the fleet's 26–30%
prefill retention exists to be read against. QSA attends a selected subset rather
than the full prefix, so the depth-dependent term the other models pay is not
merely reduced, it is absent within measurement noise.

#### The gap is a checkpoint property, not an engine setting

The table above invites the reading that the engine treats Flash-Next better
than its siblings. It does not: **they run the same attention kernel, and at
shallow depth it costs them the same.** Profiled per call (`--features profile`,
`prefill:kernel`, the kernel the speculative verify step runs):

| | shallow | deep | growth |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next | 0.382 ms @32K | 0.405 ms @128K | **+6%** |
| Qwen3.5-35B-A3B | 0.394 ms @8K | 1.463 ms @32K | **3.71× for 4× depth** |

Same kernel, same starting cost, and then one is flat and the other is linear in
prefix. The difference is that QSA caps the attended set at `top_k + ratio − 1`
= 2,051 positions however deep the cache is, and the hybrids read all of it.

The second half of the gap is how much of a step that term occupies:

| | attention kernel | share of `spec:verify` |
|---|---:|---:|
| Qwen3.5-35B-A3B @32K | 418.3 of 729.8 ms | **57%** |
| Qwen3.8-Flash-Next @32K | 129.0 of 2,007.0 ms | **6.4%** |
| Qwen3.8-Flash-Next @128K | 131.6 of 2,079.5 ms | 6.3% |

Flash-Next's step is dominated by depth-independent MoE weight work over 13B
active parameters, so the depth term is a twentieth of it before QSA bounds it
at all. On the 35B the same term is the majority of the step *and* growing.

**This cannot be enabled on the other models.** The indexer is trained weights
carried in the checkpoint — `{layer}.indexer.{q_proj,k_proj,q_norm,k_norm}.weight`,
with its geometry in the GGUF metadata — and only Flash-Next's has them. The
qwen35 lineage passes `qsa: None` at every call site. So the retention column
separates checkpoints that ship an indexer from checkpoints that do not, and no
configuration change moves a model between those groups.

**Qwen3.5-0.8B keeps 86% of decode** — far above its 9B–35B siblings. At 0.8B
the per-token cost is dominated by weight-bound work and by WDDM launch
overhead, both depth-independent, so the growing attention term is a small share
of a small total. Prefill, which has no such floor, retains 25% like everything
else. **Retention is confounded by model size and is only an architectural
statement between size-matched models** — which is why the flat row above is
stated in absolute terms rather than as a ratio.

### 3.4 The shallow group

Every model at a depth inside its own window, one context, `Coherence`, all
rows valid.

| Model | Depth | Mode | Prefill t/s | Decode t/s | Compress |
|---|---:|---|---:|---:|---:|
| Qwen2-0.5B | 8K | BF16 / C5 / C10 | 26,516.7 / 26,312.6 / 25,674.0 | 142.1 / 137.5 / 138.3 | *none — §4* |
| Llama-3.2-3B | 8K | BF16 / C5 / C10 | 6,195.9 / 5,963.8 / 5,936.8 | 69.9 / 69.5 / 66.1 | 3.29× / 4.62× |
| Llama-2-7B | 4K | BF16 / C5 / C10 | 3,684.7 / 3,852.0 / 3,851.1 | 47.1 / 47.2 / 44.1 | 3.33× / 4.81× |
| Qwen3-30B-A3B | 8K | BF16 / C5 / C10 | 3,444.7 / 3,388.6 / 3,367.6 | 51.2 / 47.5 / 47.2 | 3.70× / 5.94× |
| Qwen3-8B | 8K | BF16 / C5 / C10 | 3,235.1 / 3,213.9 / 3,212.7 | 42.8 / 40.1 / 41.8 | 3.71× / 6.31× |

At this depth compression is close to free on every model — 0–8% of decode for
3.3×–6.3× — which is the contrast that makes §3.5 worth stating separately.

### 3.5 Compression costs more decode as the cache grows

The C10 rung's decode penalty **roughly doubles** from 32K to 128K, on every
model that has enough KV for it to matter. Prefill is unaffected throughout — at
128K, five of six models prefill marginally *faster* under C10 than BF16.

| Model | C10 decode cost @8K | @32K | @128K |
|---|---:|---:|---:|
| Qwen2-0.5B | −3% | — | — |
| Qwen3-8B | −2% | — | — |
| Qwen3-30B-A3B | −8% | — | — |
| Qwen3.5-0.8B | — | +1% | −3% |
| **Qwen3.8-Flash-Next** | — | **−3%** | **−19%** |
| Qwen3.8-27B | — | −15% | −35% |
| Qwen3.6-35B-A3B | — | −27% | −48% |
| Qwen3.5-9B | — | −31% | −48% |
| Qwen3.5-35B-A3B | — | −30% | −51% |

The effect scales with the model's KV volume: the 0.8B, which has the least KV
per token, barely shows it; the 9B and the two 35B MoEs, which have the most,
lose about half their decode at 128K. Flash-Next is again the outlier, paying
19% where its architectural siblings pay ~50%.

#### The per-read cost is a constant; the number of reads is not

The column above reads as though compression gets more expensive with depth. It
does not. Profiling the attention kernel under each mode gives a **flat
multiplier**:

| Qwen3.5-35B-A3B, `prefill:kernel` | BF16 | C10 | ratio |
|---|---:|---:|---:|
| @8K | 117.1 ms | 227.0 ms | **1.94×** |
| @32K | 418.3 ms | 825.1 ms | **1.97×** |

A quantized KV position costs about twice a BF16 one to read, at every depth.
What grows is how many positions are read, and the product is what the decode
column shows. The same multiplier on Flash-Next is 1.13× at 32K and 1.36× at
128K — smaller because QSA has already bounded the reads, and because the term
is 6% of its step rather than 57%.

That the multiplier is constant is why this is not a codec defect. Measured in
isolation on the decode A/B harness at head_dim 256, every C10 format lands
between 1.26× and 1.71× of BF16 and the adaptive mix at 1.6×, with no format
dominating; `ncu` puts the kernel at 12% DRAM, 39% L1/TEX and 37% compute under
C10 against 58% DRAM under BF16 — compression removes the bandwidth it promises
to remove and the path is latency-bound either way.

**The practical consequence: C5 is the deep-context operating point, not C10.**
At 32K, C5 costs 6% of decode on the 35Bs for 3.8×, where C10 costs 30% for
7.5× — the last 2× of compression is bought at five times the decode price, and
that ratio worsens with depth because the read count does. C10 remains the right
choice where cache footprint is the binding constraint rather than latency.

This is guidance rather than a workaround: the models it applies to cannot
acquire selection (§3.3), so the read count is not reducible for them and the
per-read cost is the only term left to choose.

Compression itself does not degrade with depth. Several models compress slightly
*better* at 128K (Qwen3.5-35B-A3B 7.48× → 7.63×; Qwen3.5-9B 6.29× → 6.37×;
Qwen3.6-35B 7.01× → 7.17×), and every model that compresses at all reaches
**100% of KV blocks quantized** at every depth.

### 3.6 Qwen3.8-Flash-Next in detail

The flagship's cost is independent of context length over the measured range.
One context, uninstrumented, speculative decode (draft budget 4).

**`Coherence`**, five depths, 32 generated tokens:

| depth | prompt tokens | prefill t/s | decode t/s | accepted/step |
|---:|---:|---:|---:|---:|
| 8K | 7,990 | 1,098.8 | 24.7 | 2.07 |
| 16K | 16,020 | 1,368.9 | 28.3 | 2.21 |
| 32K | 32,128 | 1,342.8 | 24.6 | 1.94 |
| 64K | 64,249 | 1,291.8 | 26.3 | 2.07 |
| 128K | 128,433 | 1,257.5 | 26.4 | 2.21 |

**`Rewrite`** — the width ladder's own task, 64 generated tokens, so these rows
and the ladder's differ **only** in context length:

| depth | prompt tokens | prefill t/s | decode t/s | accepted/step |
|---:|---:|---:|---:|---:|
| 8K | 8,454 | 1,108.1 | 62.3 | 4.85 |
| 16K | 16,484 | 1,347.9 | 66.6 | 4.85 |
| 32K | 32,592 | 1,307.7 | 57.8 | 4.85 |
| 64K | 64,713 | 1,284.3 | 63.5 | 4.85 |
| 128K | 128,897 | 1,258.7 | 53.9 | 4.85 |

Three things this pair establishes.

**Neither curve has a depth term the measurement can see.** Prefill *rises* from
8K to 16K and then holds within 8% to 128K on both tasks — the 8K row is the
slowest, not the fastest. Decode over the same 16× range keeps 87% on `Rewrite`
(62.3 → 53.9) and is flat within noise on `Coherence` (24.7 → 26.4).

**Acceptance is constant at 4.85 on every `Rewrite` row**, which is what makes
those rows comparable to each other and to the width ladder: the fixture holds
the task fixed, so the drafter's hit rate does not drift with depth and the
decode column measures the engine rather than the task. On `Coherence` it sits
near 2.1 with no trend.

**The two tasks differ by ~2.3× at equal depth** — 53.9 against 26.4 t/s at
128K — entirely because of what is being asked. A rewrite is largely copying
tokens the drafter can see; free continuation it must guess. This is the
clearest illustration of why the task is a parameter and why rows from different
tasks are never set side by side.

The deepest `Rewrite` row is also a retrieval result: at 128,897 tokens the story
sits behind ~128K tokens of unrelated padding, and the rename still validates.

#### Width

The flagship's ladder, aggregate across the batch:

| Mode | Ctx | Prefill t/s | Decode t/s | Compress |
|---|---:|---:|---:|---:|
| BF16 | 1 (cold) | 579.4 | 68.1 | — |
| BF16 | 1 (warm) | 1,628.1 | 86.9 | — |
| BF16 | 4 | 1,798.7 | 233.5 | — |
| BF16 | 8 | 1,751.2 | 310.6 | — |
| BF16 | 16 | 1,841.5 | 333.3 | — |
| C0 | 2 | 1,808.6 | 146.4 | 2.29× |
| C5 | 2 | 1,771.2 | 145.4 | 4.21× |
| C8 | 2 | 1,838.2 | 139.5 | 5.43× |
| C10 | 2 | 1,824.6 | 126.9 | 6.93× |
| C10 | 8 | 1,857.7 | 295.9 | 6.89× |

All rows validate at 100%. Decode returns **3.8× single-session throughput at 8
contexts** and is flat from 8 to 16; prefill is already near the device's limit
at one warm context (1,628 → 1,842 at ×16), so width buys decode, not prefill.
The ladder's C0→C10 span costs **13% of decode** for **3.0× more compression**.

### 3.7 Width across the fleet

BF16 at one context against each model's widest measured point. Prompts are
~700 tokens, so this axis is unaffected by context windows.

| Model | ctx=1 prefill / decode | widest measured | prefill / decode |
|---|---|---|---|
| Qwen2-0.5B | 30,359.9 / 244.0 | ×60 | 75,846.1 / 4,562.4 |
| Qwen3.5-0.8B | 21,078.9 / 168.8 | ×32 (C8) | 30,948.3 / 2,399.5 |
| Llama-3.2-3B | 11,987.9 / 122.9 (C0) | ×10 (C8) | 12,797.2 / 729.2 |
| Qwen3-30B-A3B | 8,165.7 / 80.0 | ×10 | 9,709.2 / 505.8 |
| Qwen3.6-35B-A3B | 6,747.2 / 107.8 | ×64 (C10) | 6,645.6 / 1,144.9 |
| Qwen3.5-35B-A3B | 6,731.3 / 109.4 | ×64 (C10) | 6,549.8 / 1,170.7 |
| Qwen3-8B | 5,599.3 / 62.5 | ×10 (C8) | 6,006.0 / 460.1 |
| Llama-2-7B | 5,518.9 / 91.5 | ×48 | 3,486.3 / 834.3 |
| Qwen3.5-9B | 5,231.9 / 121.0 | ×20 (C8) | 5,537.6 / 869.9 |
| Qwen3.8-27B | 1,527.1 / 57.1 | ×40 (C10) | 1,607.2 / 458.1 |
| Qwen3.8-Flash-Next | 1,628.1 / 86.9 (warm) | ×16 | 1,841.5 / 333.3 |
| DeepSeek-V4-Flash | 333.3 / 15.0 (warm) | ×16 | 1,095.9 / 73.5 |

Two shapes appear here. **Prefill saturates early** on every model — most are
within 20% of their ×1 rate by ×4, and the 35Bs are flat from ×1 to ×64 — while
**decode scales nearly linearly with width** until it too flattens. The 35B MoEs
reach 1,145–1,171 t/s aggregate decode at 64 concurrent sessions against ~110 at
one, a 10× return on concurrency.

DeepSeek-V4-Flash is the exception whose prefill is still climbing at ×16
(333 → 1,096 t/s), having not yet reached the saturation the others hit by ×4.

The ladders are not run at a common set of widths, so this table gives each
model's own widest point rather than a shared column.

---

## 4. Limits and open items

### Qwen2-0.5B reports 0% quantized

Every C-mode row for Qwen2-0.5B-Instruct comes back `%Quantized = 0.0%` with no
compression ratio, while its sibling on the same code path (Qwen3-8B, also
`BatchedInference` + `from_gguf_by_path_with_int8`) reports 100% at 3.71×/6.31×.
The rows are otherwise healthy — output validates, and C5/C10 decode differs
from BF16, so the modes are not identical paths.

The zeros are model-specific rather than path-specific, and reproduce across
every run. The cause is not established, so **Qwen2-0.5B's C-mode rows are
reported as measured and excluded from every compression comparison in this
document.** Its throughput figures are unaffected and are used.

### DeepSeek-V4-Flash-0731 cannot reach 32K on this card

`CUDA_ERROR_OUT_OF_MEMORY` at the shallowest depth rung, with the GPU at
**72,691 of 73,415 MiB**. Its ~700-token width ladder runs fine (§3.7), so this
is specifically a depth limit, and it is not the ~6 GB DSpark drafter: removing
it changes nothing. The 284B model's expert working set fills the card on its
own, and the elastic partition has nothing left to concede to a 32K KV span.

This is the fleet's most interesting missing measurement — native-sparse
attention is precisely the architecture whose depth curve would be worth setting
against QSA and the DeltaNet hybrids. It needs a larger card or a smaller
resident expert budget.

### The slot-state buffer has a 16 MiB class ceiling

A model without GQA overruns it. Llama-2-7B-Chat has 32 KV heads, so its KV per
token is roughly 4× a GQA model's, and at 128K it produced `slot-state buffer of
17017152 B exceeds the 16777216 B top class`. This is a genuine **engine**
ceiling rather than a model limit.

Nothing in the current suite triggers it — Llama-2 is measured at 4K, well
inside the class — which is exactly why it is recorded here. Every other model
in the fleet has GQA and stays inside the ceiling regardless of depth.

### Validation at extreme width under maximum compression

Qwen3.6-35B-A3B's width ladder fails its 100% threshold at the two widest C10
rungs: **31/32 sessions at ×32 and 61/64 at ×64**. Throughput is healthy at both
(6,519.6 / 973.9 and 6,645.6 / 1,144.9 t/s), and the same model passes every
depth rung and every narrower width rung. Its sibling Qwen3.5-35B-A3B passes the
same ×32 and ×64 C10 rungs outright.

The rows are reported with their measured validity. One or two sessions in
sixty-four degrading under maximum compression at maximum concurrency is a
narrow enough failure that it is recorded as an open item rather than treated as
a general result about either C10 or width.

### The validation column is weaker than the throughput column

`CoherenceCheck` asks whether output is non-degenerate — at minimum, that it is
20% alphanumeric — not whether it is correct. It catches a model emitting
punctuation, and it does not catch fluent nonsense. The filler is also eight
rotating templates of dry technical prose, far from any instruction-tuned
model's training distribution.

**Throughput and compression figures are unaffected** — those tokens were
genuinely processed — but this document should not be read as evidence about
output *quality* at depth. `StoryRewrite`, which the flagship's rewrite curve
uses, is the stronger check: it is a concrete retrieval-and-rewrite task with a
verifiable answer.

### The quantized read multiplier is understood but not reduced

The ~2× per-position cost of a quantized KV read at decode (§3.5) is the largest
lever remaining for the models that cannot use selection, and it is parked
rather than solved.

What is established: it is not one bad codec — the whole C10 tier measures
1.26–1.71× of BF16 in isolation and the adaptive mix 1.6×, and `ncu` shows the
kernel latency-bound in both modes (C10 at 12% DRAM / 39% L1/TEX / 37% compute;
BF16 at 58% DRAM). Compression removes the bandwidth it is supposed to remove.
The quantized path issues 3.5× the global-load instructions and 2.9× the shared
loads of the float path, because quantized blocks are addressed per dim through
the block path while float takes a vector load — but **instruction counts are
not latency here**: the dequant runs on CUDA cores alongside tensor-core work,
so a higher count can be fully hidden, and neither path is near a roofline.
Which of the block path's fixed costs actually sits on the critical path — the
K-side shared-memory stage and read-back, the V-side shared `atomicMax` and its
barrier, or the all-or-nothing warp votes that make one quantized palette cost
what four would — is **not established**, and would need per-stall attribution
rather than the throughput counters collected so far.

`candle-examples/examples/decode_ab` is the harness for this work: CUDA-event
kernel timing with an FP32 golden gate, a `--formats c10` group covering the
level's whole candidate set, and an adaptive row (`rq-adaptive-L10`) for the
mixed case that uniform rows cannot represent.

### Q0_V decodes at 7.35× BF16, and cannot simply be dropped

The C10 tier's costliest format by a wide margin. In a uniform arena it decodes
at **332.9 µs against 45.3 µs for BF16 — 7.35×**, where every other candidate at
that level sits between 1.26× and 1.71×; its per-element `__constant__` table
lookups serialise across a warp whose lanes are each on a different block.

It nonetheless has to stay a candidate, which is the useful part of the finding.
Removing it from C6/C8/C9/C10 leaves the compression ratio **identical** —
7.10× on Qwen3.5-35B — because those blocks fall back to Q0_X at the same two
bytes; but the reconstruction differs, and three models lose output validation
at C10 under high concurrency: Qwen3.5-35B (31/32 and 63/64 sessions),
Qwen3.8-Flash-Next (7/8), Qwen3.6-35B (60/64 against its own standing 61/64).
Restoring it returns all three to their prior state exactly, reproducibly.

So the cost is a decode-path wart to fix in the kernel, not a candidate to
retire. The measurement trap is worth recording alongside it: an adaptive C10
cache on the decode harness measures 73.6 µs with Q0_V and 73.2 µs without,
which reads as "never selected" — but that fixture is **synthetic** K/V, and on
real model KV the selector plainly does pick it. A uniform-format bench cannot
answer a question about an adaptive cache, and a synthetic fixture cannot answer
one about production selection.

### Not measured

- **Full-attention against hybrid at matched depth.** The clean size-matched
  pair — Qwen3-8B and Qwen3.5-9B, same Q6_K, one full-attention and one hybrid —
  cannot be compared at depth, because the 8B's 40,960 window does not reach
  where the hybrids are measured. Making this comparison needs a
  full-attention checkpoint with a native 262K window.
- **Selection against no selection at matched depth.** Same problem from the
  other side: Flash-Next is the only model here that ships an indexer, and it is
  also the largest and the only one with QSA, MoE and a 250B parameter count at
  once. Its flat curve is measured beyond doubt (§3.3), but the fleet contains
  no second selecting model to separate QSA's contribution from everything else
  that is unique to that checkpoint. DeepSeek-V4-Flash is native-sparse and
  would be exactly that control — it cannot reach 32K on this card.
- **Multi-context depth.** Every depth row is one context; width × depth
  interaction is not covered.
- **Warm/cold KV tiers.** All runs are hot-tier; nothing exercised the RAM or
  NVMe tiers.
- **Non-speculative decode at depth.** The depth gate drives the speculative
  loop on models that support it; the plain path is not separately measured
  there.
- **Anything but this machine.** See §1.

---

## 5. Provenance

Every row measured in the sweep — including the ones the tables above omit — is
in `performance_rtx_pro_5000_72gb_rows.tsv` beside this file: 146 rows of
`test, label, depth, prompt_tokens, mode, int8, contexts, valid, prefill_tps,
decode_tps, quantized_pct, compress, peak_tokens`, scraped from the run logs.
Reproduce any row with the command in its test's `#[ignore]` attribute.

| Table | Test |
|---|---|
| §3.2, §3.3, §3.4 | each model's `long_context_*` gate |
| §3.5 | the same gates, C-mode rows |
| §3.6 Coherence | `quantized_qwen38_moe::tests::profile_decode_vs_depth` |
| §3.6 Rewrite | `quantized_qwen38_moe::tests::profile_story_rewrite_vs_depth` |
| §3.6 Width, §3.7 | `test_parallel_batched_forwarding*` |

All runs were strictly sequential — one `cargo test` invocation per model, so
exactly one model was ever resident and no run's VRAM sizing was perturbed by
another's. Run-to-run variation is 1–4% on the width ladder and ~5% on the depth
sweep, which is the noise floor any comparison in this document has to clear;
differences smaller than that are not claimed as results.
