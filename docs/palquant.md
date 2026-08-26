# PalQuant: Calibration-Free 7× KV Cache Compression via Adaptive Per-Block Quantization

**John Sharratt**

---

## Abstract

Existing online KV cache quantization methods optimise the compression *primitive* — uniform 2-bit (KIVI), outlier-aware fixed precision (KVQuant), random orthogonal rotation (TurboQuant) — and assign one format to all blocks at the population level. We present **PalQuant**, an adaptive system that optimises the compression *architecture*: at inference time, each (layer, head, chunk) selects 4 active formats from a fixed 16-format codebook spanning 0.25–16 bits per element, with each block dynamically routed to one of the 4 based on its measured reconstruction error. Selection uses structurally distinct K and V error metrics matching their attention paths: mean-of-top-4 weighted absolute error for K (capturing softmax-amplified outliers), warp-mean squared error for V (the L2 budget for V's attention-weighted contribution), with parameter-free sink protection via tanh-weighted Q·K alignment statistics.

Deployed identically across three models with no per-model calibration data, PalQuant achieves **7.42× cache compression on Qwen3-8B**, **7.04× on Qwen3-30B-A3B (MoE)**, and **5.02× on Llama-3.2-3B**, validated end-to-end by a multi-session story rewrite test where each concurrent session must preserve a distinct assigned character identity (name and gender) under aggressive compression. **PalQuant passes the multi-session test at all 11 compression levels (C0–C10) up to 7.42× CR**, while uniform Q4_0 fails at 3.56× CR on Llama-3.2-3B and Qwen3-30B-A3B — even though uniform Q4_0 holds competitive bulk PPL, demonstrating a structural dissociation between perplexity and the multi-session quality metric (§4.5). The selection mechanism's decisions transfer cleanly across MoE/dense, model families, and model sizes because pre-RoPE K/V activation structure is universal at the per-block level. PalQuant approaches Kitty's offline-calibrated compression band (7.42× vs ~8×) without calibration data and substantially exceeds TurboQuant (4.6× at FP16-equivalent quality). Native CUDA kernels on a single RTX 4090 Mobile (16 GB) support up to 256/120/64 concurrent Llama-3.2-3B/Qwen3-30B-A3B/Qwen3-8B sessions at 168K/75K/41K total KV cache tokens.

---

## 1. Introduction

The KV cache is the dominant memory bottleneck for long-context LLM inference: it grows linearly with context length while weights remain fixed, dominating VRAM on persistent workloads within tens of thousands of tokens. Inference-time KV cache quantization has produced three methodological lines — per-channel/per-token quantization with offline calibration (KIVI [Liu et al., 2024], KVQuant [Hooper et al., 2024]), rotation-based uniform quantization (TurboQuant [Zandieh et al., 2026]), and offline-calibrated mixed-precision selection (Kitty [Xia et al., 2025], KVTuner [Li et al., 2025]).

These methods share a structural property: **they optimise the compression *primitive***. Each proposes a single quantization formula — uniform 2-bit, outlier-aware 3-bit, rotated Lloyd-Max codebooks, magnitude-ranked channel boost — and applies it uniformly across every block in the cache, with format choice determined either by population statistics or by offline magnitude rankings. A fixed-primitive approach must either be conservative enough to handle the worst block (sacrificing compression on easy blocks) or aggressive enough to compress the easy blocks (failing on hard ones).

We present **PalQuant**, a system that optimises the compression *architecture*. Rather than picking one format and applying it everywhere, PalQuant picks four formats per (layer, head, chunk) from a fixed 16-format codebook spanning 0.25–16 bits per element, then routes each block to one of the four based on its measured reconstruction error. The codebook contains predetermined mathematical quantization formats — Q0 through Q8_KS — that span a structurally complete BPE ladder.

The selection mechanism is the technical core. It runs entirely at inference time, evaluates per-block reconstruction error with K- and V-specific metrics matched to the attention error paths, and produces palette assignments without any per-model calibration data. Deployed across three models — Qwen3-30B-A3B (MoE), Qwen3-8B (dense), and Llama-3.2-3B (dense) — it achieves 7.04×, 7.42×, and 5.02× compression respectively, validated by a multi-session story rewrite test that requires each session to preserve a distinct character identity under aggressive compression.

**Contributions.** (1) A **fixed 16-format codebook** spanning 0.25–16 BPE with three novel formats (Q0_V, Q0_X, Q0); no training, no calibration data. (2) **Palette-4 per-(layer, head, chunk) selection**: 128 blocks per chunk sorted by amax and partitioned into 4 slots of 32, each slot claiming a quota from the residual its predecessors could not cover, so BPE escalates across the slots and the demanding blocks land in the conservative ones; 88 B per-(chunk, head) overhead. (3) **Asymmetric K/V error metrics** matched to attention propagation — top-4-mean weighted absolute error for K, warp-mean MSE for V — grounded in the AsymKV bound [Tao et al., 2025]. (4) **Parameter-free runtime sink protection** via tanh-weighted Q·K alignment statistics; no fixed thresholds. (5) **Cross-model validation without calibration data**: passes the multi-session story rewrite test at all 11 levels (C0–C10) on Qwen3-30B-A3B (MoE), Qwen3-8B, and Llama-3.2-3B with no quality cliff. (6) **Native fused CUDA kernel**: 256/120/64 concurrent Llama-3.2-3B/Qwen3-30B-A3B/Qwen3-8B sessions at 168K/75K/41K total KV cache tokens on a single RTX 4090 Mobile (16 GB).

One mechanism — adaptive selection from a fixed codebook — yields what prior systems each calibrate separately: KVQuant's outlier protection from the per-block error metric, KIVI's K/V awareness from metric asymmetry, Kitty's mixed precision from palette-4 slot allocation. No calibration data needed.

---

## 2. Background and Related Work

### 2.1 LLM Inference and the KV Cache Bottleneck

Decoder-only transformer inference has two phases: parallel prefill over the input, then autoregressive decode that attends to all previously cached K and V tensors. The cache stores 2 tensors per layer per head per token at the model's native precision. A 30B-parameter MoE with 48 layers, 8 KV heads, and head dim 128 costs 192 KB per token — 6 GB for a 32K context at FP16. On 16–24 GB consumer GPUs, the cache is the binding memory constraint at long context. Quantization is the dominant compression lever, cutting per-token memory 4×–8× while preserving attention structure.

### 2.2 KV Cache Quantization

Four prior systems define the inference-time training-free landscape. **KIVI** [Liu et al., 2024] uses uniform 2-bit per-channel K and per-token V, reaching ~2.6× reduction. **KVQuant** [Hooper et al., 2024] adds pre-RoPE K quantization with sensitivity-weighted datatypes and dense-sparse outlier handling, holding sub-0.1 PPL at 3-bit. **TurboQuant** [Zandieh et al., 2026] uses Walsh-Hadamard rotation plus Lloyd-Max quantization — 4.6× CR at FP16-equivalent quality, calibration-free. **Kitty** [Xia et al., 2025] boosts 12.5–25% of K channels by offline magnitude rank to ~8×, and flags adaptive selection as future work.

All four assign formats at the population level — uniformly (KIVI, TurboQuant), by fixed rule (KVQuant), or by offline ranking (Kitty). PalQuant addresses the gap Kitty identifies, replacing binary precision-boost with a 16-format codebook, offline magnitude ranking with runtime per-block error scoring, and population averages with per-block validation.

Two adjacent works extend the design space orthogonally: PatternKV [Zhang et al., 2025] flattens activations to expand quantization headroom, and Staniszewski & Łańcucki [2026] apply transform coding for compact KV storage. Both compose with per-block adaptive selection.

### 2.3 Asymmetric K/V Error Propagation and Attention Sinks

The structural difference between K and V error propagation in attention is foundational to PalQuant's metric design. Attention computes $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})\mathbf{V}$. Let $\hat{\mathbf{K}} = \mathbf{K} + \Delta\mathbf{K}$ and $\hat{\mathbf{V}} = \mathbf{V} + \Delta\mathbf{V}$.

**K error** enters through the softmax: at low attention entropy (peaked distributions), small logit perturbations cause large changes in the weight distribution, and single-element K outliers can dominate the softmax score for a given query. K errors are *outlier-amplified*; the appropriate metric captures the worst few elements without being all-or-nothing on a single spike.

**V error** scales linearly by the softmax weights. For each query $t$ with weights $\boldsymbol{\alpha}_t$ satisfying $\|\boldsymbol{\alpha}_t\|_1 = 1$, Jensen's inequality on $x^2$ gives $\|\hat{\mathbf{a}}_t - \mathbf{a}_t\|_2 \leq \|\Delta\mathbf{V}\|_F$. V error propagates with amplification factor exactly 1 per query, regardless of attention entropy: the L2 budget is structurally correct and the appropriate metric is MSE.

AsymKV [Tao et al., 2025] proved this bound and demonstrated that up to 75% of decoder layers tolerate 1-bit V without output degradation, while 1-bit K causes near-total failure; KVTuner [Li et al., 2025] confirmed via layer-wise analysis that K is generally more sensitive than V. PalQuant operationalises this asymmetry directly in metric design (§3.3).

**Attention sinks.** StreamingLLM [Xiao et al., 2024] established that the first few tokens carry disproportionate attention weight; truncating them catastrophically degrades generation. KVSink [Su & Yuan, 2025] showed that sinks can also emerge mid-sequence. PalQuant's sink protection (§3.4) is parameter-free and fully runtime, catching both registration and emergent sinks via per-token Q·K alignment statistics. Token eviction (H2O, SnapKV, StreamingLLM) and trained compression (DMS) are orthogonal compression classes, out of scope for inference-time training-free quantization.

---

## 3. Method

PalQuant decomposes KV cache quantization into four mechanisms operating per (chunk, head): (1) a fixed 16-format codebook, (2) palette-4 selection with budgeted slot allocation, (3) K/V error metrics matched to softmax-amplified vs L2-bounded propagation, and (4) parameter-free sink protection via Q·K alignment statistics.

### 3.1 The Fixed 16-Format Codebook

The codebook contains 16 active formats spanning 0.25–16 BPE (64×), plus 2 floating-point fallbacks (F16, BF16) used when no active format passes the per-block error threshold. All formats encode 32-element blocks. Active formats span five encoding families: symmetric/asymmetric integer quantization, sub-block-scaled int (K-only), sign-encoded, mask-selected constants, and three novel sub-1-BPE templates described below. Full bit layouts and reconstruction formulas for all 18 formats appear in Appendix A.

Three formats are novel. **Q0_V** (parametric curve, 0.50 BPE) reconstructs blocks as $x[e] = c[v][w] + s[v] \cdot \text{curve}[u][e]$ via three indices into precomputed constant-memory tables (8.5 KB per arena). Q0_V's tables are the only model-derived component anywhere in the codebook; calibrated once from population statistics of pre-RoPE K/V activations and shipped as constants. **Q0_X** (outlier-aware constant, 0.50 BPE) stores an INT8 bulk anchor with one outlier escape (5-bit position, 3-bit signed delta), compressing near-constant blocks with a single anomalous spike. **Q0** (constant, 0.25 BPE) stores a single INT8 centroid for blocks where Q-projected attention is approximately zero.

### 3.2 Palette-4 Per-(Chunk, Head) Selection

Each (chunk, head) holds 128 blocks (32 elements per block × 128 blocks = one 32-token chunk per head). PalQuant partitions these into 4 palette slots of 32 blocks. Blocks within a slot share a single (format, outer scale) pair, selected to satisfy the per-block reconstruction-error threshold. K and V have independent palettes. Per-(chunk, head) palette metadata, per side: 4 × format tag (4 B) + 4 × FP16 outer scale (8 B) + 128 × 2-bit palette index (32 B) = **44 bytes**. Combined K+V overhead is 88 B per (chunk, head), contributing 0.086 BPE across the K+V cache (88 × 8 / (32 × 128 × 2)). All compression ratios in §4 include this overhead. During autoregressive decode, tokens accumulate at FP16 in the active chunk's scratch buffer; the selection kernel runs at each 32-token chunk boundary and commits the completed chunk to a codebook format.

**Slot filling order: cheapest format first, over a shrinking residual.** Blocks are sorted by absolute maximum descending, which concentrates the demanding blocks at the head of the array and lets the candidate scan terminate early once a format reaches the 32-block slot quota. Slot 0 then searches the codebook in BPE-ascending order against the full block set, so it settles on an aggressive format — one that can still claim a full quota — and the blocks it claims leave the pool. Each later slot searches only the residual its predecessors could not cover, which is strictly harder, and therefore escalates: BPE increases monotonically from slot 0 to slot 3, and slot 3 holds the conservative format for whatever nothing cheaper could handle. Per-head precision is therefore a 4-element distribution over the codebook, not a population-level scalar: a head with bimodal block difficulty gets three aggressive slots for easy blocks and one conservative slot for the outliers. Appendix L gives a real-data trace of this behaviour at C5; Tables 16 and 17 show the resulting monotone slot ladder directly.

The selection algorithm walks each slot in BPE-ascending order through the candidate format list (defined per operating point in §3.6), returning the first format where 32 blocks pass the per-block error threshold. If no format reaches 32, the slot falls back to the format with lowest worst-case error across the slot's blocks.

### 3.3 K/V Error Metrics

The two attention error paths require structurally distinct metrics.

**K metric — mean-of-top-4 weighted absolute error.** For the 32 elements of a block:
$$\varepsilon_K(\mathbf{k}, \hat{\mathbf{k}}) = \frac{\text{mean}_4 \big[|k_i - \hat{k}_i| \cdot w_i\big]}{\text{head amax}}$$
where $\text{mean}_4$ is the mean of the four largest values across the 32 lanes, and $w_i$ is an optional per-element weight (set to 1 for the F32 K arena; set to $|q_i|$, the magnitude of the Q element at the same head-dim position as $k_i$ in the same token, for the R16 K arena that stores Q alongside K). Head-amax normalisation makes the metric dimensionless and transferable across heads, layers, and models — a single threshold applies uniformly. We use top-4 mean rather than max because softmax washes out single-element spikes (max over-rejects benign candidates) and because empirically, K errors affecting attention output concentrate in 1–4 dimensions per block under the GQA topology used by Qwen3 and Llama-3.

**V metric — warp-mean squared error.** Across the 32 lanes:
$$\varepsilon_V(\mathbf{v}, \hat{\mathbf{v}}) = \frac{\text{mean}_{32}[(v_i - \hat{v}_i)^2]}{\text{head amax}^2}$$
V's contribution to attention output is the linear combination $\sum_t \alpha_t \mathbf{v}_t$, so the output error budget is exactly L2. Top-k metrics would over-penalise outlier elements that don't move the actual L2; MSE gives the exact budget.

The asymmetry — top-4 mean for K, warp-mean MSE for V — is grounded in the AsymKV bound (§2.3). The PalQuant kernel operationalises both metrics in a single reduction per block, with no additional cost over a uniform error metric. An offline-analysis variant of the K metric, used during threshold-curve derivation (Appendix D) but not in the runtime kernel, projects K vectors onto the top-30 PCA components of the per-(layer, head) Q distribution; this Q-subspace-projected error and the resulting two-sided gating analysis are documented in Appendix C.

### 3.4 Parameter-Free Attention Sink Protection

Attention sinks degrade quality if their V is quantized loosely (their attention contribution dominates the output). PalQuant detects sinks at runtime via per-token Q·K alignment scoring: for each of the 32 tokens in a chunk we compute $\text{score}_t = \mathbf{q}_{\text{mean}} \cdot \mathbf{k}_t / \sqrt{d}$ where $\mathbf{q}_{\text{mean}}$ is the chunk-mean Q vector (a pre-softmax proxy for "attention received by token $t$"), z-score against chunk statistics $z_t = (\text{score}_t - \mu) / \max(\sigma, \epsilon)$, and tanh-weight $w_t = \max(0, \tanh(z_t)) \in [0,1]$. The chunk's maximum sink weight $w_{\max}$ then lerps the V threshold:
$$\tau_V^{\text{eff}} = \tau_V^{\text{lo}} + w_{\max} \cdot (\tau_V^{\text{hi}} - \tau_V^{\text{lo}}),$$
where $\tau_V^{\text{lo}} > \tau_V^{\text{hi}}$ numerically (lo = lenient, hi = strict). One strong sink in the chunk forces stricter quality on every V block of that chunk. Detection threshold, sharpness, and floor are all derived from chunk-local statistics — no fixed thresholds, no calibration data, no per-model tuning. Both registration sinks (sequence start) and emergent sinks (mid-sequence) produce above-average alignment scores and are detected uniformly.

### 3.5 Per-Slot Outer-Scale Search

For each slot's chosen format, PalQuant searches six outer-scale candidates derived from the slot's per-block amax distribution: $\{1.0,\ 1/\text{amax},\ 1/p_{95},\ 1/p_{80},\ 1/\text{mean},\ 1/p_{25}\}$. For FP16-block-scale formats (Q4_0, Q4_1, Q8_0, etc.), the outer scale algebraically cancels in the round-trip and the search picks a canonical scale for metadata. For INT8-scale formats (Q1_S, Q2_S, Q2_A) and Q0-family formats, the outer scale *is* the quantization scale; the search picks the candidate that admits the most blocks under the slot's per-block error threshold.

### 3.6 Operating Points (C0–C10)

PalQuant exposes 11 operating points C0–C10. Each level is defined by (a) a candidate format list per side (K and V independently) drawn from the 16-format codebook, and (b) per-side error thresholds. Base error thresholds (Appendix D, Table 6) are shared across all three models; per-model API scaling factors (§4.1) shift the user-facing ladder for consistent semantics without modifying the algorithm or codebook. Full per-side candidate lists for all 11 levels appear in Appendix B.

**Ladder design.** C0–C5 use standard Q3–Q8 integer formats; C6 introduces sub-2-BPE; C7–C8 drop Q8; C9 admits the full sub-1-BPE template ladder keeping Q4 on V as backstop; C10 strips Q4 from V, producing the largest single CR step (5.24× → 7.04× on Qwen3-30B-A3B).

### 3.7 Native CUDA Kernel Implementation

PalQuant runs as a single fused CUDA kernel per (chunk, head): a five-phase selection algorithm produces palette metadata that a companion attention kernel uses to dispatch per-slot dequantization. Each palette holds **4 formats with a 2-bit per-block selector** — the namesake palette mechanism — routing each of the 128 blocks to one of four candidate formats. The palette-4 constraint comes within 1–3% of the per-block ideal CR (Appendix K, Table 12), substantially outperforming any single-format-per-head baseline. Phase-by-phase implementation, shared-memory layout, and SM occupancy appear in Appendix F.

---

## 4. Experimental Evaluation

### 4.1 Setup

**Models.** Three models span the architectural diversity required for cross-model generalization: **Qwen3-30B-A3B** (MoE, 48 layers, 128 experts, 8 KV heads, head dim 128), **Qwen3-8B** (dense, 36 layers, 8 KV heads, head dim 128), and **Llama-3.2-3B** (dense, 28 layers, 8 KV heads, head dim 128). All use grouped-query attention; weights at Q4_K_M throughout.

**Hardware.** NVIDIA RTX 4090 Mobile, 16 GB GDDR6, Ada Lovelace (sm_89), native FP8 tensor cores.

**Implementation.** Custom Rust + CUDA inference engine with native quantized matmul kernels. Full hardware specifications, software stack, and reproducibility instructions are documented in Appendix H.

**Evaluation methodology.** Q4_K_M weights are used throughout the inference stack, so reported PPL values include weight-quantization contribution and are not directly comparable to prior work that evaluates KV quantization in isolation against FP16 weights. PPL is reported at 2048-token context — the field convention used by KIVI, KVQuant, TurboQuant, Kitty, and prior Qwen3 quantization studies — for direct comparability. Context-length sensitivity of PPL across configurations is examined in Appendix I, which surfaces a structural problem with the 2048 convention as a basis for evaluating KV cache compression. Variation across Table 3 rows isolates the KV cache contribution at fixed weights and fixed context.

### 4.2 Compression Ratio: Three-Model Curve

All three models use the identical PalQuant algorithm, codebook, base threshold curve, and CUDA kernel. Per-model API scaling factors (described next) shift the user-facing C0–C10 ladder for consistent semantics across model families. Table 1 reports measured compression ratios at all 11 operating points (C0–C10).

**Per-model API scaling factors.** Different models have different intrinsic compression headroom; to keep C0–C10 ladder semantics comparable across model families, PalQuant applies four dimensionless scaling factors per model (`k_hi`, `k_low`, `v_hi`, `v_low`) that multiply the base thresholds (Appendix D, Table 6). Llama-3.2-3B uses IDENTITY (1.0 throughout); the Qwen models use factors that admit more aggressive compression at the same level number, reflecting their larger compression headroom. The factors are not data-derived, do not modify the selection algorithm or the codebook, and are equivalent to deploying a wider ladder in which each model accesses a feasible subset. Per-model values are listed in Appendix D, Table 7.

**Table 1.** Effective compression ratio per model per level. All numbers include palette overhead (§3.2). Bold values mark each model's maximum CR (at C10).

| Level | Qwen3-8B | Qwen3-30B-A3B (MoE) | Llama-3.2-3B |
|---|---:|---:|---:|
| C0 | 1.90× | 1.98× | 1.88× |
| C1 | 2.54× | 2.54× | 2.28× |
| C2 | 2.66× | 2.74× | 2.49× |
| C3 | 2.91× | 2.98× | 2.87× |
| C4 | 3.31× | 3.41× | 3.18× |
| C5 | 3.68× | 3.67× | 3.42× |
| C6 | 4.29× | 4.16× | 3.62× |
| C7 | 4.46× | 4.22× | 3.92× |
| C8 | 4.81× | 4.69× | 3.96× |
| C9 | 5.39× | 5.24× | 4.28× |
| C10 | **7.42×** | **7.04×** | **5.02×** |

Two observations:

**Cross-architecture generalization.** The same algorithm, codebook, and CUDA kernel deliver quality-preserving compression across three architectures (Qwen MoE, Qwen dense, Llama dense), three parameter scales (3B, 8B, 30B), and two model families. The 5.02–7.42× CR variation reflects intrinsic compression headroom in each model's pre-RoPE K/V activations.

**Head-dimension independence.** The three models in Table 1 share head dim 128 with 8 KV heads. The mechanism is not tied to that geometry: selection operates on 32-element blocks, so head dimension determines only how many blocks a palette band covers — 32 per band at head dim 128, 64 at head dim 256. The same selection is deployed unchanged on attention layers at **head dim 256 with 2–4 KV heads** (Qwen3.5 family), and on the 16-band latent geometry of DeepSeek-V4-Flash, with compression ratios and throughput holding in both. This is the per-block architecture argument (§11.3 of the companion report) reaching a case it was not derived on: a scheme fitted to one head geometry requires re-derivation when the geometry moves, whereas per-block selection requires a different band count.

**Monotone CR through C9, large jump to C10.** The CR sequence is monotone-increasing across all three models. The C9→C10 step is the largest (e.g., 5.24× → 7.04× on Qwen3-30B-A3B) because the C10 candidate list strips Q4 from V, forcing V into the structural-template ladder. All reported CRs are *effective* — they include the 88 B per-(chunk, head) palette metadata (§3.2; ~0.086 BPE, ~2.7% CR impact at C9, negligible at C0–C2).

### 4.3 Format Distribution: How Selection Behaves

Table 2 summarises K and V format-family usage on Qwen3-30B-A3B at five representative levels (Q8_KS is K-only, hence the 0 in the V row). Full per-format K and V breakdowns and parallel cross-model views appear in Appendix E.

**Table 2.** Combined format-family usage on Qwen3-30B-A3B by compression level (% of all blocks, K | V split).

| Level | Q8 (K\|V) | Q4 (K\|V) | Q3 (K\|V) | Q2 (K\|V) | Q1 (K\|V) | Q0 (K\|V) |
|---|---|---|---|---|---|---|
| C0 | 100\|65 | 0\|35 | 0\|0 | 0\|0 | 0\|0 | 0\|0 |
| C3 | 26\|18 | 74\|60 | 0\|22 | 0\|0 | 0\|0 | 0\|0 |
| C5 | 21\|1 | 14\|29 | 65\|70 | 0\|0 | 0\|0 | 0\|0 |
| C7 | 20\|0 | 13\|24 | 24\|57 | 26\|12 | 17\|7 | 0\|0 |
| C10 | 0\|0 | 0\|0 | 43\|34 | 20\|24 | 4\|18 | 24\|20 |

The selection mechanism behaves as a compression waterfall: K transitions Q8_KS → Q4 → Q3 → structural template ladder, culminating in ~24% Q0-family usage at C10. V follows in parallel but retains Q4 fallback through C7. Q0-family formats (Q0, Q0_V, Q0_X, Q0_M2, Q0_M4) appear primarily at C8–C10 where they capture per-block structure the integer ladder cannot represent at sub-2-BPE.

### 4.4 Quality Validation: WikiText-2 Perplexity

Perplexity on WikiText-2 [Merity et al., 2017] is the standard reference benchmark for KV cache compression. Table 3 reports cross-model results at five representative PalQuant levels plus uniform-quantization and F16 baselines. Per-level Pareto curves, long-context sweeps, and full-corpus comparisons appear in Appendix I.

**Table 3.** Cross-model WikiText-2 perplexity at 2048-token context (field convention), 50K tokens, Q4_K_M weights throughout. All variation across rows is in KV cache representation. Lower is better. Qwen3-14B API factors are not yet tuned (Table 7), so it uses default factors; it is included for PPL evidence only and excluded from §4.6 multi-session testing. C10 PPL elevations: Llama-3.2-3B +41.4%, Qwen3-30B-A3B +39.8%, Qwen3-8B +28.7%, Qwen3-14B +25.2% over F16.

| KV configuration | BPE | Llama-3.2-3B | Qwen3-8B | Qwen3-14B | Qwen3-30B-A3B |
|---|---:|---:|---:|---:|---:|
| F16 KV | 16.00 | 16.74 | 9.88 | 8.78 | 7.32 |
| Q8/Q8 KV | 8.50 | 16.75 | 9.88 | 8.78 | 7.32 |
| PalQuant C1  | 6.30–7.02 | 16.44 | 9.90 | 8.81 | 7.25 |
| Q4/Q4 KV | 4.50 | 16.50 | 9.96 | 8.89 | 7.27 |
| PalQuant C5  | 4.35–4.68 | 17.65 | 10.38 | 9.15 | 8.09 |
| PalQuant C8  | 3.33–4.04 | 17.60 | 10.88 | 9.21 | 8.18 |
| PalQuant C9  | 2.97–3.74 | 17.89 | 11.07 | 10.25 | 8.49 |
| PalQuant C10 | 2.16–3.19 | 23.67* | 12.72 | 10.99 | 10.23 |

**C1 Pareto-dominance.** PalQuant C1 *improves* PPL over F16 on Llama-3.2-3B (16.44 vs 16.74) and Qwen3-30B-A3B (7.25 vs 7.32) at 60% lower BPE, and is Pareto-equivalent on Qwen3-8B (9.90 vs 9.88) and Qwen3-14B (8.81 vs 8.78). §4.5 carries the discriminating evidence between configurations.

### 4.5 Quality Metric Dissociation

PPL and the multi-session story rewrite test (§4.6) can give opposite verdicts on the same configuration.

**Worked example: uniform Q4/Q4 vs PalQuant C5.** At comparable BPE (4.50 vs 4.35–4.68), uniform Q4/Q4 produces *better* PPL than PalQuant C5 at the field convention 2048 ctx — Llama-3.2-3B: 16.50 vs 17.65 (-7.0%, Q4/Q4 lower); Qwen3-8B: 9.96 vs 10.38 (-4.0%); Qwen3-14B: 8.89 vs 9.15 (-2.9%); Qwen3-30B-A3B: 7.27 vs 8.09 (-11.3%, the largest gap of the four). On a PPL-only assessment Q4/Q4 dominates PalQuant in this band. Yet uniform Q4_0 *fails* the §4.6 multi-session story rewrite test at 3.56× CR on Llama-3.2-3B and Qwen3-30B-A3B, while PalQuant passes at all 11 levels through C10. The two metrics give opposite verdicts on the same configuration.

**Second worked example: Llama-3.2-3B at C10.** The dissociation can occur within a *single* configuration. At 5.02× CR, Llama-3.2-3B's PPL elevates to 23.67 (+41% over F16) while the §4.6 multi-session test passes. PPL alone would label C10 a quality cliff; structural validation says the configuration is intact at deployment-relevant compression. The dissociation generalises across all three §4.6-tested models: Qwen3-30B-A3B (+39.8% PPL) and Qwen3-8B (+28.7%) show the same pattern at C10, all passing §4.6. Qwen3-14B's matching +25.2% PPL is consistent but not §4.6-tested (untuned API factors; Table 3 caption). RULER on Qwen3-8B (§4.7) provides a third confirming metric.

**The dissociation is structural.** PPL averages over all blocks: a configuration with most blocks well-quantized and a small fat tail of badly-approximated blocks holds acceptable PPL because the tail amortises across the corpus. PalQuant routes adaptively — easy blocks to aggressive slots, hard blocks to conservative slots — so some easy blocks pay a small PPL cost that uniform Q4_0 (spending 4 bits everywhere) avoids. The multi-session test is binding on the opposite tail: a single mis-quantized K block in a sink position routes one session's query into another session's cache, contaminating identity. PalQuant's per-block selection protects exactly these blocks; uniform Q4_0 at 3.56× CR has no such headroom. The offline Q-subspace-projected K metric (Appendix C) underpins threshold derivation; the runtime kernel uses a simpler magnitude metric for throughput.

### 4.6 Quality Validation: Multi-Session Story Rewrite

**Test protocol.** Multiple concurrent sessions run on a single GPU instance. Each session is assigned a distinct character identity (name and gender). The model is instructed to rewrite a narrative passage using the assigned character. A session passes iff: (a) the correct assigned name appears in the rewrite; (b) gender pronouns are consistent with the assignment; (c) no other session's assigned name appears. The test stresses the high-relevance K sub-block population — blocks where Q·K dot products concentrate — which carry the session-discriminating signal under aggressive compression.

**Headline result.** PalQuant passes the story rewrite test at *all 11 compression levels (C0–C10)* on *all three models*; no quality cliff is observed. **Comparison vs uniform quantization:** uniform Q4_0 fails the test on Llama-3.2-3B and Qwen3-30B-A3B at 3.56× CR (sessions confuse character identities) but passes on Qwen3-8B — pass/fail varies by model. Uniform Q4_1 (3.20× CR) likewise fails on Llama-3.2-3B. PalQuant passes uniformly at all 11 levels: strictly higher compression than the uniform formats that fail, and uniform quality across models where the primitives produce inconsistent verdicts.

### 4.7 Long-Context Retrieval (RULER)

We additionally evaluate PalQuant on the RULER benchmark [Hsieh et al., 2024] (NIAH-Single, NIAH-MultiKey-2, variable tracking) on Qwen3-8B at 4K context. CWE aggregation is excluded as a model-capability ceiling: Qwen3-8B fails CWE at F16. Weights are Q4_K_M throughout — matching the rest of the paper, so RULER pass rates compose with the §4.2 compression ratios. Per-cell sample counts and longer-context cells are reported in Appendix M.

**Table 4.** RULER pass rates at 4K, Qwen3-8B. NIAH = pooled NIAH-Single + NIAH-MultiKey-2 (retrieval-structural). VT = variable tracking (multi-step reasoning). Higher is better. Per-task pass counts and sample sizes in Appendix M, Table 19.

| Configuration | BPE | CR | NIAH | VT |
|---|---:|---:|---:|---:|
| F16 KV | 16.00 | 1.00× | 95% | 100% |
| Q4/Q4 KV | 4.50 | 3.56× | 97% | 100% |
| Q8/Q4 KV (asymm) | 6.50 | 2.46× | 87% | 83% |
| PalQuant C5  | 4.35 | 3.68× | 98% | 86% |
| PalQuant C9  | 2.97 | 5.39× | 90% | 75% |
| PalQuant C10 | 2.16 | 7.41× | 80% | 71% |

RULER confirms the §4.5 dissociation. F16 NIAH is 95% (App M discusses the Qwen3-8B output-formatting failures). C5 matches F16 NIAH within noise at 3.68× CR; C9 retains 90% NIAH at 5.39× CR; C10 at 7.41× drops to 80%/71% — the same Qwen3-8B configuration whose PPL elevates +28.7% (Table 3) yet the §4.6 multi-session test passes.

The Q8/Q4 asymmetric baseline at 6.50 BPE sits at 87% NIAH and 83% VT. PalQuant matches/exceeds Q8/Q4 retrieval at less than half the BPE (C5: 98%/4.35 BPE; C9: 90%/2.97 BPE), but Q8/Q4 retains an 8-point VT advantage at 2.2× the bits — fixed higher-BPE quantization remains competitive on multi-step reasoning below 3 BPE. C5 at 4.35 BPE matches Q8/Q4 on both metrics while compressing 1.5× more.

### 4.8 Comparison with Published Systems

**Table 5.** Comparison of PalQuant against published online KV cache quantization systems on dimensions of compression ratio, format-assignment mechanism, calibration requirement, and quality-validation methodology.

| System | Max CR | Format assignment | Calibration | Quality validation |
|---|---:|---|---|---|
| KIVI [Liu et al., 2024] | ~2.6× | Uniform 2-bit | None | Population-level (PPL) |
| KVQuant [Hooper et al., 2024] | ~3× | Fixed 3-bit + outliers | Sensitivity profiling | Population-level (PPL) |
| TurboQuant [Zandieh et al., 2026] | 4.6× | Orthogonal rotation + Lloyd-Max | None | Population-level (PPL) |
| Kitty [Xia et al., 2025] | ~8× | Mixed-precision K, magnitude rank | Offline calibration | Population-level (PPL) |
| **PalQuant (this work)** | **7.42×** | **Per-block adaptive, palette-4** | **None** | **Per-block; story rewrite C0–C10** |

PalQuant approaches Kitty's compression band (7.42× vs ~8×) without calibration data, and substantially exceeds TurboQuant (4.6×, calibration-free). The architectural difference: PalQuant's selection runs at inference time on each block's measured reconstruction error; the four prior systems assign formats at the population level. PalQuant additionally provides per-block quality validation through a structural failure mode (multi-session entity discrimination) that population-level perplexity does not detect.

**Methodological divergence.** Prior systems disagree on the right metric: KIVI, KVQuant, and TurboQuant report PPL; Kitty rejects PPL and uses task accuracy (GSM8K, MATH, AIME, MMLU). The §4.6 multi-session test extends this trajectory: neither corpus PPL nor task accuracy isolates the per-block structural failure — a mis-quantized K block routing one session's query into another's — that breaks deployment regardless of bulk statistics. §4.5 dissociation is the empirical signature.

### 4.9 Throughput

On a single RTX 4090 Mobile (16 GB), peak operating points are: Llama-3.2-3B C5 at 256 sessions / 887 t/s aggregate decode / 168K KV tokens; Qwen3-8B C8 at 64 sessions / 573 t/s / 41K tokens; Qwen3-30B-A3B C8 at 120 sessions / 603 t/s / 75K tokens. F16 OOMs before any of these session counts. Selection adds <1% overhead vs the underlying paged-attention kernel; Q4-family configurations reach similar throughput but fail §4.6 (Appendix J).

### 4.10 Ablations

Three components are individually necessary. **Adaptive selection:** uniform Q4_0 fails §4.6 at 3.56× CR while PalQuant passes through C10 at 7.42× — adaptive selection strictly improves the quality-compression Pareto. **Palette-4:** assigning one format per head ("worst-1") delivers 1.83×–4.61× CR vs PalQuant's 1.97×–6.93× — a 7.6–56.2% gain (App K, Table 12). **Sink protection and outer-scale search:** disabling sink protection produces pronoun drift at C7–C10; removing per-slot outer-scale search reduces achievable CR by 5–15% at C5–C7.

---

## 5. Conclusion

PalQuant delivers 5.02–7.42× calibration-free KV cache compression across three models, passing §4.6 at all 11 levels and supporting 256/120/64 concurrent sessions on a single RTX 4090 Mobile (16 GB). Adaptive per-block selection from a fixed codebook recovers prior-system advantages while eliminating calibration. The selection operates on 32-element blocks and is therefore indifferent to head dimension: head dim only sets how many blocks a palette band covers (32 per band at head dim 128, 64 at 256), and the mechanism is deployed unchanged across both. **Limitations:** GQA.

---

## References

[1] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*, 2024. arXiv:2401.18079.

[2] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, volume 235 of PMLR, pp. 32332–32344, 2024. arXiv:2402.02750.

[3] Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. TurboQuant: Online vector quantization with near-optimal distortion rate. In *International Conference on Learning Representations (ICLR 2026)*, 2026. arXiv:2504.19874.

[4] Haojun Xia, Xiaoxia Wu, Jisen Li, Robert Wu, Junxiong Wang, Jue Wang, Chenxi Li, Aman Singhal, Alay Dilipbhai Shah, Alpay Ariyak, Donglin Zhuang, Zhongzhu Zhou, Ben Athiwaratkun, Zhen Zheng, and Shuaiwen Leon Song. Kitty: Accurate and efficient 2-bit KV cache quantization with dynamic channel-wise precision boost. *arXiv preprint arXiv:2511.18643*, 2025.

[5] Qian Tao, Wenyuan Yu, and Jingren Zhou. AsymKV: Enabling 1-bit quantization of KV cache with layer-wise asymmetric quantization configurations. In *Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)*, pp. 2316–2328, Abu Dhabi, UAE, 2025. arXiv:2410.13212.

[6] Xing Li, Zeyu Xing, Yiming Li, Linping Qu, Hui-Ling Zhen, Wulong Liu, Yiwu Yao, Sinno Jialin Pan, and Mingxuan Yuan. KVTuner: Sensitivity-aware layer-wise mixed-precision KV cache quantization for efficient and nearly lossless LLM inference. In *International Conference on Machine Learning (ICML 2025)*, 2025. arXiv:2502.04420.

[7] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In *International Conference on Learning Representations (ICLR 2024)*, 2024. arXiv:2309.17453.

[8] Zunhai Su and Kehong Yuan. KVSink: Understanding and enhancing the preservation of attention sinks in KV cache quantization for LLMs. In *Conference on Language Modeling (COLM 2025)*, 2025. arXiv:2508.04257.

[9] Ji Zhang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Xinglin Wang, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, and Kan Li. PatternKV: Flattening KV representation expands quantization headroom. *arXiv preprint arXiv:2510.05176*, 2025.

[10] Konrad Staniszewski and Adrian Łańcucki. KV cache transform coding for compact storage in LLM inference. In *International Conference on Learning Representations (ICLR 2026)*, 2026. arXiv:2511.01815.

[11] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In *International Conference on Learning Representations (ICLR 2017)*, 2017. arXiv:1609.07843.

[12] An Yang, Anfeng Li, Baosong Yang, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.

[13] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

---

## Appendix

### A. Format Specifications

Block layouts and reconstruction formulas for the 16 codebook formats. All formats encode 32-element blocks. Standard integer formats follow conventional GGUF/llama.cpp block-quantization (FP16 or INT8 scales, integer codes); novel formats are described in the table directly.

| Format | BPE | Block layout (per 32 elements; total bytes) | Reconstruction |
|---|---:|---|---|
| Q8_KS | 9.00 | FP16 scale (2 B), 2× INT8 sub-block offsets (2 B), 32× INT8 (32 B); **36 B** | $x_i = s \cdot (1 + \delta_{k(i)}/127) \cdot q_i$ — sub-block-scaled across positions 0–3 vs 4–31 |
| Q8_1 | 9.00 | FP16 scale (2 B), FP16 min (2 B), 32× UINT8 (32 B); **36 B** | $x_i = s \cdot q_i + m$ |
| Q8_0 | 8.50 | FP16 scale (2 B), 32× INT8 (32 B); **34 B** | $x_i = s \cdot q_i$ |
| Q4_1 | 5.00 | FP16 scale (2 B), FP16 min (2 B), 32× UINT4 (16 B); **20 B** | $x_i = s \cdot q_i + m$ |
| Q4_0 | 4.50 | FP16 scale (2 B), 32× INT4 packed (16 B); **18 B** | $x_i = s \cdot q_i$ |
| Q3_1 | 4.00 | FP16 scale (2 B), FP16 min (2 B), 32× UINT3 packed (12 B); **16 B** | $x_i = s \cdot q_i + m$ |
| Q3_0 | 3.50 | FP16 scale (2 B), 32× INT3 packed (12 B); **14 B** | $x_i = s \cdot q_i$ |
| Q2_A | 2.50 | INT8 scale (1 B), INT8 bias (1 B), 32× UINT2 (8 B); **10 B** | $x_i = s \cdot q_i + b$ |
| Q2_S | 2.25 | INT8 scale (1 B), 32× INT2 (8 B); **9 B** | $x_i = s \cdot q_i$ |
| Q0_M4 | 2.00 | 4× INT8 centroids (4 B), 16× 2-bit pair selector (4 B); **8 B** | $x_i = c_{m_{\lfloor i/2 \rfloor}}$ — 4-centroid pair-mask |
| Q1_A | 1.50 | INT8 $s_+$ (1 B), INT8 $s_-$ (1 B), 32× sign bit (4 B); **6 B** | $x_i = s_+$ if $\text{sgn}(i){=}{+}$, else $-s_-$ — asymmetric sign-encoded |
| Q1_S | 1.25 | INT8 scale (1 B), 32× sign bit (4 B); **5 B** | $x_i = s \cdot \text{sgn}(i)$, $\text{sgn}(i) \in \{-1,+1\}$ |
| Q0_M2 | 0.75 | 2× INT8 centroids (2 B), 8× 1-bit quartet selector (1 B); **3 B** | $x_i = c_{m_{\lfloor i/4 \rfloor}}$ — 2-centroid quartet-mask |
| Q0_V | 0.50 | 8 b curve idx $u$, 5 b scale idx $v$, 3 b centroid idx $w$; **2 B** | $x_i = c[v][w] + s[v] \cdot \text{curve}[u][i]$ from constant tables |
| Q0_X | 0.50 | INT8 anchor $a$ (1 B), 5 b outlier position $p$, 3 b signed delta $d$; **2 B** | $x_i = a$ for $i \neq p$; $x_p = a + d \cdot s_{\text{step}}(a)$ |
| Q0 | 0.25 | INT8 centroid $c$ (1 B); **1 B** | $x_i = c$ for all $i$ — constant block |

Q0_V's tables (256 curves × 32 scales × 8 centroids per arena, 8.5 KB constant-memory per K or V arena) are calibrated once from population statistics of pre-RoPE K/V activations and shipped as constants — the only model-derived component anywhere in the codebook. F16 and BF16 (16 BPE, 64 B per block) are used as quality fallback when no codebook format passes the per-block error threshold; they are not part of the 16-format codebook itself.

### B. Per-Level Candidate Format Lists

Each operating point $C_n$ defines a candidate format list per side (K and V independently), drawn from the 16-format codebook of §3.1. The selection kernel walks each list in BPE-ascending order and picks the first format where the slot's 32 blocks pass the per-block error threshold. Lists below are in BPE-descending order (most conservative first).

| Level | K candidates | V candidates |
|---|---|---|
| C0 | Q8_KS | Q8_0, Q4_0 |
| C1 | Q8_KS, Q4_1, Q4_0 | Q8_0, Q4_0 |
| C2 | Q8_1, Q8_0, Q4_1, Q4_0 | Q8_0, Q4_1, Q4_0, Q3_1, Q3_0 |
| C3 | Q8_1, Q8_0, Q4_1, Q4_0 | Q8_0, Q4_1, Q4_0, Q3_1, Q3_0 |
| C4 | Q8_1, Q8_0, Q4_1, Q4_0, Q3_1, Q3_0 | Q8_0, Q4_1, Q4_0, Q3_1, Q3_0 |
| C5 | Q8_0, Q4_1, Q4_0, Q3_1, Q3_0 | Q8_0, Q4_1, Q4_0, Q3_1, Q3_0 |
| C6 | Q8_1, Q8_0, Q4_1, Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q1_S, Q0_V | Q8_1, Q8_0, Q4_1, Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q1_S |
| C7 | Q8_1, Q8_0, Q4_1, Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q1_S | Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q1_S |
| C8 | Q4_1, Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q1_S, Q0_V | Q4_1, Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q1_S |
| C9 | Q3_1, Q3_0, Q2_A, Q2_S, Q0_M4, Q1_A, Q1_S, Q0_M2, Q0_V, Q0_X, Q0 | Q4_1, Q4_0, Q3_1, Q3_0, Q2_A, Q2_S, Q0_M4, Q1_A, Q1_S, Q0_M2, Q0_V, Q0_X, Q0 |
| C10 | Q3_1, Q3_0, Q2_A, Q2_S, Q0_M4, Q1_A, Q1_S, Q0_M2, Q0_V, Q0_X, Q0 | Q3_1, Q3_0, Q2_A, Q2_S, Q0_M4, Q1_A, Q1_S, Q0_M2, Q0_V, Q0_X, Q0 |

### C. K-Side Q-Subspace-Projected Error (Offline Analysis Path)

This appendix documents the offline-analysis variant of the K-side error metric used to (i) derive the threshold curve in Appendix D and (ii) validate the two-sided gating heuristic in §3.4. The runtime kernel uses a simpler form (summarised at the end of this appendix).

**Q-subspace projection.** Per-(layer, head) Q distributions are summarised by their top-30 PCA components, computed offline on the calibration corpus. For each block $b$, K-relevance is
$$r_b = \frac{\|\mathbf{P}_{30} \mathbf{k}_b\|_2}{\|\mathbf{k}_b\|_2}$$
where $\mathbf{P}_{30}$ projects onto the top-30 Q-PCA subspace and $\mathbf{k}_b$ is the per-block mean K vector.

**Two-sided gating.** The empirical distribution of $r$ across the calibration corpus is bimodal: most blocks lie either outside the Q-relevant subspace ($r \to 0$, "drain") or substantially within it ($r \to 1$, "sink"). The cutoffs $r < 0.20$ (drain — block can absorb aggressive compression without affecting attention output) and $r > 0.95$ (sink — block demands strict compression because its K perturbations propagate through softmax) cleanly separate the two modes.

**Production runtime.** The selection kernel does not compute the PCA projection. The R16 K arena uses per-element $Q^2$-weighted error with the actual stored query vectors $\mathbf{q}_i$; the F32 K arena (which does not store Q) falls back to magnitude-weighted absolute error using the head's amax as a weight surrogate. Both are functionally equivalent to the offline projection for the purpose of distinguishing high-relevance from low-relevance blocks, at a fraction of the cost.

### D. Threshold Derivation and Per-Model API Scaling

**Derivation procedure.** Base thresholds at each compression level are anchored to percentiles of specific formats' empirical error distributions on the calibration corpus. C0 anchors to a near-lossless operating point (a strict percentile on the cheapest acceptable format). Each subsequent level $c$ anchors to the next-most-aggressive admitted format's distribution, such that admitting that format under the level-$c$ threshold preserves the per-block error guarantee for that format's typical case. Values are dimensionless — expressed as a fraction of the head's absolute maximum — making the same threshold transferable across heads, layers, and models without rescaling.

**Two-sided thresholds.** For each side (K, V), each level defines a HIGH (strict) and LOW (lenient) threshold. Sink protection (§3.4) lerps between the two: blocks whose chunk-mean Q·K alignment indicates an attention sink land near the strict bound; blocks identified as drain land near the lenient bound. The lerp coefficient is computed from chunk-local statistics, so this is parameter-free at runtime.

**Table 6.** Base error thresholds at C0–C10 (dimensionless, fraction of per-head $|\text{amax}|$). Same values used across all three models.

| Level | $\tau_K^{\text{HIGH}}$ | $\tau_K^{\text{LOW}}$ | $\tau_V^{\text{HIGH}}$ | $\tau_V^{\text{LOW}}$ |
|---|---:|---:|---:|---:|
| C0  | 0.003096 | 0.011315 | 0.012232 | 0.012730 |
| C1  | 0.004725 | 0.051794 | 0.018664 | 0.025898 |
| C2  | 0.008944 | 0.072130 | 0.015596 | 0.022541 |
| C3  | 0.014703 | 0.102114 | 0.019366 | 0.030230 |
| C4  | 0.018199 | 0.136622 | 0.022474 | 0.050119 |
| C5  | 0.020700 | 0.216643 | 0.023001 | 0.153698 |
| C6  | 0.020758 | 0.232942 | 0.023768 | 0.170920 |
| C7  | 0.021735 | 0.248296 | 0.024000 | 0.187766 |
| C8  | 0.018771 | 0.284827 | 0.022167 | 0.215390 |
| C9  | 0.025236 | 0.274433 | 0.023852 | 0.250035 |
| C10 | 0.031321 | 0.453389 | 0.024824 | 0.653093 |

**Non-monotonicity.** The HIGH thresholds are not strictly monotone: $\tau_K^{\text{HIGH}}$ drops at C7→C8, $\tau_V^{\text{HIGH}}$ drops at C1→C2 and C7→C8, and $\tau_V^{\text{LOW}}$ drops at C1→C2. These dips occur at level boundaries where the candidate list (Appendix B) admits a new format that fills a structural gap — for example, C8 admits Q0_V on the K side and removes Q8 from V, so the K-strict threshold at C8 anchors to a tighter percentile of the new candidate set than C7 did to the old. The thresholds are monotone on average, with localised relaxations at format-admission boundaries. The user-facing semantics: a level number controls *which* aggressive formats are admitted into the candidate set, while the HIGH threshold controls *which blocks qualify* for the most aggressive format in that set. When a more aggressive format is admitted (e.g., Q0_V at C8), the threshold tightens so that only blocks that can actually tolerate it are routed there — leaving the bulk of blocks to the next-tier format. A "more aggressive level with a stricter HIGH threshold" therefore reads as "the new aggressive format is now available, but admission to it is selective"; this is the intended composition.

**Table 7.** Per-model API scaling factors (defined in §4.1). The applied threshold at level $c$ for model $m$ is $\tau_m(c) = \text{factor}_m \cdot \tau_{\text{base}}(c)$.

| Model | `k_hi` | `k_low` | `v_hi` | `v_low` |
|---|---:|---:|---:|---:|
| Llama-3.2-3B  | 1.000 | 1.000 | 1.000 | 1.000 |
| Qwen3-8B      | 0.900 | 1.450 | 0.900 | 2.600 |
| Qwen3-30B-A3B | 0.475 | 1.200 | 1.225 | 2.700 |

Smaller `k_hi` tightens the K-strict threshold (more conservative selection at sink positions); larger `k_low` relaxes the K-lenient threshold (more aggressive selection at drain positions). The Qwen models' factor combinations admit more aggressive formats per slot at the same level number, the desired UX behaviour given their larger compression headroom. Factors were selected by sweeping a small grid against the multi-session story rewrite test on a held-out passage set, retaining the most aggressive combination that still passed at all 11 levels. The procedure does not use the WikiText-2 evaluation corpus and is consistent with the calibration-free deployment claim: a new model can use IDENTITY (1.0 throughout) at the cost of operating at a less aggressive point on its own ladder.

### E. Format Distribution on Qwen3-8B Across All 11 Compression Levels

§4.3 Table 2 reports K and V format distributions on Qwen3-30B-A3B at five representative levels with per-side breakdown. This appendix provides the parallel cross-model view: Qwen3-8B format usage at all 11 operating points (C0–C10), aggregated across both sides into format-family bins to show the compression-ladder transition explicitly.

**Table 8.** Qwen3-8B format usage by compression level (% of all elements, combined K+V). Each format family aggregates its symmetric and asymmetric variants (Q8 = Q8_KS + Q8_0; Q4 = Q4_0 + Q4_1; Q3 = Q3_0 + Q3_1; Q2 = Q2_S + Q2_A; Q1 = Q1_S + Q1_A; Q0 = Q0 + Q0_V + Q0_X + Q0_M2 + Q0_M4). The "Scratch" column is the active in-flight chunk — tokens accumulate at native FP16 (R16 for K, F16 for V) until they reach the 32-token chunk boundary, at which point the selection kernel commits them to a codebook format. The ~5% rate is geometric (1 of 20 chunks per context) and fixed by chunk granularity, not a tunable parameter. Rows sum to 100%.

| Level | Q8 family | Q4 family | Q3 family | Q2 family | Q1 family | Q0 family | Scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0  | 87.7 |  7.3 |  0.0 |  0.0 |  0.0 |  0.0 | 5.0 |
| C1  | 39.8 | 55.2 |  0.0 |  0.0 |  0.0 |  0.0 | 5.0 |
| C2  | 35.0 | 58.4 |  1.6 |  0.0 |  0.0 |  0.0 | 5.0 |
| C3  | 23.5 | 67.3 |  4.2 |  0.0 |  0.0 |  0.0 | 5.0 |
| C4  | 17.4 | 32.3 | 45.3 |  0.0 |  0.0 |  0.0 | 5.0 |
| C5  |  9.9 | 23.4 | 61.7 |  0.0 |  0.0 |  0.0 | 5.0 |
| C6  |  9.3 | 22.0 | 32.3 | 19.4 |  6.6 |  5.4 | 5.0 |
| C7  |  6.4 | 23.2 | 32.2 | 19.9 | 13.4 |  0.0 | 5.0 |
| C8  |  0.0 | 29.8 | 31.7 | 19.8 |  7.6 |  6.0 | 5.0 |
| C9  |  0.0 | 14.7 | 41.0 | 19.3 |  6.0 | 13.9 | 5.0 |
| C10 |  0.0 |  0.0 | 34.5 | 20.2 | 11.8 | 28.5 | 5.0 |

**Three transitions visible in the data, each interpretable.** The Q8 family — entirely a K-side phenomenon at C0 (Q8_KS) plus a V-side fallback (Q8_0) — fully exits at C8, consistent with the C8 candidate-list change that strips Q8 from V (§3.6, Appendix B). The Q4 family rises through C3 (where it dominates at 67.3%), then declines as Q3 takes over the bulk of the bandwidth-balanced regime; Q4 fully exits at C10 when V's candidate list strips it as well. The structural template family (Q0, Q0_V, Q0_X, Q0_M2, Q0_M4) emerges at C6 (5.4% of total), grows monotonically through C9 (13.9%), and reaches 28.5% at C10 — the largest single family at maximum compression. This is the empirical signature of the codebook's structural diversity: as the per-block error threshold tightens, the selection mechanism increasingly routes blocks to formats that exploit known activation structure (parametric template Q0_V, outlier-aware constant Q0_X, centroid-mask Q0_M2/Q0_M4) rather than to lower-bit integer formats that would fail the threshold.

**Per-side and full per-model breakdowns.** §4.3 Table 2 gives the per-side (K and V separately) breakdown for Qwen3-30B-A3B at five representative levels. Tables 13 and 14 below provide per-side breakdowns for Qwen3-8B and Llama-3.2-3B at all 11 levels (family-aggregated). Qwen3-30B-A3B family-level combined distributions paralleling Table 8 will be provided as supplementary material with the camera-ready version.

**Table 13.** K-side palette format family distribution by compression level on Qwen3-8B and Llama-3.2-3B (% of (head, slot) pairs; rows sum to 100 modulo rounding). Each (chunk, head, K) palette holds 4 (format, outer-scale) pairs; this distribution counts how often each format family is selected. p4_CR is the per-side palette-4 compression ratio.

| Level | Model | Q8 | Q4 | Q3 | Q2 | Q1 | Q0 | p4_CR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0  | Qwen3-8B     | 100.0 | —    | —    | —    | —    | —    | 1.76× |
| C0  | Llama-3.2-3B | 100.0 | —    | —    | —    | —    | —    | 1.76× |
| C1  | Qwen3-8B     |  54.5 | 45.6 | —    | —    | —    | —    | 2.26× |
| C1  | Llama-3.2-3B |  74.7 | 25.2 | —    | —    | —    | —    | 2.00× |
| C2  | Qwen3-8B     |  38.0 | 62.1 | —    | —    | —    | —    | 2.60× |
| C2  | Llama-3.2-3B |  61.7 | 38.3 | —    | —    | —    | —    | 2.24× |
| C3  | Qwen3-8B     |  26.2 | 73.8 | —    | —    | —    | —    | 2.82× |
| C3  | Llama-3.2-3B |  48.8 | 51.2 | —    | —    | —    | —    | 2.40× |
| C4  | Qwen3-8B     |  22.7 | 17.1 | 60.3 | —    | —    | —    | 3.21× |
| C4  | Llama-3.2-3B |  25.6 | 36.9 | 37.5 | —    | —    | —    | 2.94× |
| C5  | Qwen3-8B     |  20.6 | 13.8 | 65.6 | —    | —    | —    | 3.31× |
| C5  | Llama-3.2-3B |  21.8 | 38.1 | 40.2 | —    | —    | —    | 3.03× |
| C6  | Qwen3-8B     |  20.5 | 13.6 | 24.2 | 26.0 |  5.3 | 10.3 | 3.91× |
| C6  | Llama-3.2-3B |  15.0 | 28.5 | 32.2 | 22.4 |  1.5 |  0.4 | 3.57× |
| C7  | Qwen3-8B     |  19.9 | 13.0 | 23.5 | 26.5 | 17.0 | —    | 3.91× |
| C7  | Llama-3.2-3B |  13.5 | 28.7 | 32.3 | 23.6 |  2.0 | —    | 3.64× |
| C8  | Qwen3-8B     | —     | 35.5 | 22.4 | 25.2 |  5.6 | 11.2 | 4.76× |
| C8  | Llama-3.2-3B | —     | 46.6 | 28.0 | 23.2 |  1.8 |  0.5 | 4.02× |
| C9  | Qwen3-8B     | —     | —    | 51.7 | 22.7 |  4.5 | 21.0 | 5.71× |
| C9  | Llama-3.2-3B | —     | —    | 67.9 | 24.2 |  2.5 |  5.4 | 4.73× |
| C10 | Qwen3-8B     | —     | —    | 43.3 | 20.2 |  5.1 | 31.4 | 6.50× |
| C10 | Llama-3.2-3B | —     | —    | 56.4 | 22.9 |  7.4 | 13.3 | 5.28× |

**Table 14.** V-side palette format family distribution by compression level on Qwen3-8B and Llama-3.2-3B (% of (head, slot) pairs). p4_CR is the per-side palette-4 compression ratio. Note that V CR can decrease at level boundaries where the candidate list tightens (e.g., C7→C8 on Llama as Q4_1 becomes the dominant format).

| Level | Model | Q8 | Q4 | Q3 | Q2 | Q1 | Q0 | p4_CR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0  | Qwen3-8B     | 65.4 | 34.6 | —    | —    | —    | —    | 2.23× |
| C0  | Llama-3.2-3B | 90.4 |  9.6 | —    | —    | —    | —    | 1.95× |
| C1  | Qwen3-8B     | 26.7 | 73.3 | —    | —    | —    | —    | 2.84× |
| C1  | Llama-3.2-3B | 75.5 | 24.5 | —    | —    | —    | —    | 2.11× |
| C2  | Qwen3-8B     | 27.9 | 61.9 | 10.2 | —    | —    | —    | 2.83× |
| C2  | Llama-3.2-3B | 65.2 | 32.1 |  2.7 | —    | —    | —    | 2.22× |
| C3  | Qwen3-8B     | 18.2 | 60.1 | 21.7 | —    | —    | —    | 3.10× |
| C3  | Llama-3.2-3B | 59.5 | 34.9 |  5.6 | —    | —    | —    | 2.30× |
| C4  | Qwen3-8B     |  5.6 | 51.7 | 42.7 | —    | —    | —    | 3.55× |
| C4  | Llama-3.2-3B | 56.9 | 31.6 | 11.5 | —    | —    | —    | 2.35× |
| C5  | Qwen3-8B     |  1.1 | 29.2 | 69.8 | —    | —    | —    | 3.99× |
| C5  | Llama-3.2-3B | 35.0 | 42.4 | 22.6 | —    | —    | —    | 2.72× |
| C6  | Qwen3-8B     |  0.8 | 25.4 | 57.5 | 10.3 |  6.0 | —    | 4.33× |
| C6  | Llama-3.2-3B | 29.2 | 43.5 | 24.9 |  1.1 |  1.3 | —    | 2.87× |
| C7  | Qwen3-8B     | —    | 24.1 | 57.1 | 12.2 |  6.6 | —    | 4.46× |
| C7  | Llama-3.2-3B | —    | 70.5 | 26.9 |  1.3 |  1.3 | —    | 3.72× |
| C8  | Qwen3-8B     | —    | 23.9 | 55.0 | 13.6 |  7.5 | —    | 4.49× |
| C8  | Llama-3.2-3B | —    | 72.1 | 25.1 |  1.5 |  1.3 | —    | 3.53× |
| C9  | Qwen3-8B     | —    | 18.5 | 53.8 | 17.1 |  5.2 |  5.3 | 4.79× |
| C9  | Llama-3.2-3B | —    | 67.0 | 29.4 |  2.0 |  0.3 |  1.2 | 3.62× |
| C10 | Qwen3-8B     | —    | —    | 33.4 | 24.0 | 18.7 | 23.8 | 7.41× |
| C10 | Llama-3.2-3B | —    | —    | 84.5 |  9.7 |  3.2 |  2.6 | 4.42× |

**Cross-model reading.** Llama-3.2-3B and Qwen3-8B follow similar K-side trajectories through C5 but diverge above. At C10, Qwen3-8B routes 31.4% of K to the Q0 family (sub-1-BPE structural templates) while Llama-3.2-3B uses 13.3%. The V side shows the more dramatic divergence: at C10, Qwen3-8B reaches 23.8% Q0 + 18.7% Q1 (42.5% sub-1-BPE total) while Llama-3.2-3B keeps 84.5% in the Q3 family. The combined effect explains the headline CR variation in Table 1: Qwen3-8B reaches 7.42× while Llama-3.2-3B reaches 5.02× — the difference is intrinsic compression headroom in V activations, not algorithm tuning. Llama-3.2-3B uses IDENTITY API factors (Table 7); Qwen3-8B's tuned factors admit more aggressive formats per slot at the same level number.

### F. CUDA Kernel Implementation Details

The fused selection kernel runs one CUDA block per (chunk, head) pair. Each block uses 128 threads organised as 4 warps of 32 lanes; one chunk contains 32 tokens and one head holds 128 blocks (32 elements each). The kernel runs five sequential phases.

**Phase 1 — load and per-block amax.** Each warp loads 32 of the head's 128 blocks (one block per lane) into shared memory, computes the per-block absolute maximum via warp shuffle reduction, and writes amax to a 128-entry shared buffer. The bulk of the smem footprint is established at this phase.

**Phase 2 — sink detection.** A single warp computes the chunk-mean Q vector across all blocks of the head and per-token Q·K alignment scores (§3.4). The 32 alignment scores are z-scored against chunk statistics; tanh-weighted sink weights $w_t$ are written to a 32-entry shared buffer. Phase 2 cost is dominated by the per-token dot product and is amortised across all 128 blocks. The chunk's $w_{\max}$ — a single FP32 value — is broadcast to all warps for use in Phase 4.

**Phase 3 — bitonic sort by amax.** All 128 blocks are sorted in shared memory by descending amax using a warp-cooperative bitonic sort. Sorting concentrates the high-amax blocks (those most likely to need conservative formats) at the head of the array, allowing the slot-search phase to terminate early once all easy blocks have been claimed. Sort cost is $O(\log^2 n)$ exchanges across 128 elements, amortised across 128 threads.

**Phase 4 — per-block threshold computation.** Each block's K and V thresholds are computed as $\tau^{\text{eff}} = \tau^{\text{lo}} + w_{\max} \cdot (\tau^{\text{hi}} - \tau^{\text{lo}})$ using the Phase 2 sink weight and the per-model API-scaled base thresholds (Appendix D). The 128 effective thresholds (K and V) are written to shared memory.

**Phase 5 — iterative slot search.** The selection algorithm walks a two-dimensional search space: the candidate format list (Appendix B) crossed with the per-slot outer-scale ladder (§3.5). For each of the four palette slots, the kernel searches in BPE-ascending order for the (format, outer-scale) pair that claims the maximum number of remaining unclaimed blocks under the per-block error thresholds. A claimed block is removed from the candidate pool; the next slot searches against a shrinking residual set. The search terminates when either (a) all 128 blocks are claimed, or (b) all four slots are filled. The remaining unclaimed blocks (if any) fall back to F16 passthrough — a rare event, and one that bounds the worst-case per-(chunk, head) error to FP16-equivalent.

**Shared-memory budget.** Total smem footprint per (chunk, head) is ~12.7 KB across all five phases, dominated by the Phase 1 block-storage buffer with smaller contributions from amax + sink weights, threshold tables, and sort scratch / slot-search bookkeeping. MaxShared carveout (Ada-class hardware) enables 5–8 (chunk, head) pairs to run concurrently per SM, yielding the throughput numbers reported in §4.9. The selection mechanism adds <1% wall-clock overhead vs the underlying paged-attention kernel (§4.9).

### G. K Selection Metric: Design Rationale

The K-side error metric (§3.3) is mean-of-top-4 weighted absolute error, normalised by the head's amax. This appendix compares it against two natural alternatives — cosine distance and magnitude-weighted absolute error — in the framework of the multi-session story rewrite test (§4.6), highlighting the failure modes that motivated the production choice.

**Compared metrics (K side only).** All three operate on the same 32-element block; all are normalised by the head amax for cross-head comparability:

- *Cosine distance:* $\varepsilon_K^{\text{cos}} = 1 - \langle \mathbf{k}, \hat{\mathbf{k}} \rangle / (\|\mathbf{k}\| \cdot \|\hat{\mathbf{k}}\|)$
- *Magnitude-weighted abs:* $\varepsilon_K^{\text{magw}} = \frac{1}{32} \sum_i |k_i - \hat{k}_i| \cdot |k_i|$
- *Top-4 weighted abs (production):* $\varepsilon_K^{\text{top4}} = \text{mean}_{\text{top-4}}\big[|k_i - \hat{k}_i| \cdot w_i\big]$

**Failure modes observed.** All three metrics produce comparable C0–C7 behaviour. Divergence appears at C8–C10:

- **Cosine distance failed cross-session character preservation at C8 on Llama-3.2-3B.** In the multi-session test, one session was assigned a character named "Marcus." Under cosine-distance K selection, palette assignment treated direction-preserving but magnitude-collapsing approximations as low-error. The result was that "Marcus" tokens from session 4's KV cache produced near-aligned but heavily-rescaled K vectors at retrieval time; the attention layer routed queries from session 7 (a different character) to session 4's "Marcus" K cluster, producing cross-session character contamination. The story rewrite output for session 7 began naming its character "Marcus." This failure was reproducible at C8 on Llama-3.2-3B across multiple trials and was eliminated by switching to magnitude-aware error metrics. We refer to this failure mode as **Marcus contamination**: cosine-only metrics cannot distinguish "small magnitude error" from "completely wrong magnitude in the relevant direction," and the latter is a dispositive failure for retrieval.

- **Magnitude-weighted absolute error** does not produce Marcus contamination but over-rejects benign blocks. The averaging across all 32 elements washes out the structural fact that K errors propagate through softmax via 1–4 dominant elements per block; benign blocks with one moderate element-error get penalised the same as malicious blocks with one extreme element-error. The result is conservative format selection and meaningfully higher mean BPE at the same quality threshold.

- **Top-4 weighted absolute error** captures both: it preserves magnitude (eliminating Marcus contamination) and concentrates on the elements that actually propagate through softmax (eliminating over-rejection of benign blocks). It is the production choice.

**Scope.** Quantitative comparison across the three metrics on a uniform corpus and operating-point grid is left for future work; the qualitative failure modes documented above motivated the production choice and were verified to be eliminated by it on the §4.6 multi-session story rewrite test at C8 across all three models.

### H. Reproducibility

**Hardware.** All measurements use a single NVIDIA RTX 4090 Mobile (16 GB VRAM, Ada-class, compute capability 8.9). Results on data-centre hardware (H100, A100) are expected to scale similarly: the dominant cost is the per-block error reduction inside the fused selection kernel, which is bandwidth-bound on Ada and tensor-bound on Hopper.

**Models.** Three checkpoints, all weights at Q4_K_M, KV cache FP16 in baselines:

- Qwen3-30B-A3B — MoE, 48 layers, 128 experts, 8 KV heads, head dim 128, GQA group size 8.
- Qwen3-8B — dense, 36 layers, 8 KV heads, head dim 128, GQA group size 8.
- Llama-3.2-3B — dense, 28 layers, 8 KV heads, head dim 128, GQA group size 8.

All three are publicly available; see [Yang et al., 2025] for Qwen3 and [Grattafiori et al., 2024] for Llama 3.

**Software stack.** Custom Rust/Candle inference engine with native CUDA kernels: a fused selection kernel per (chunk, head) and a palette-aware paged-attention kernel. Kernel structure and shared-memory layout are documented in §3.7 and Appendix F.

**Calibration-corpus-free deployment.** PalQuant requires no calibration data at deployment time. The Q0_V template tables (Appendix A) — the only model-derived component anywhere in the codebook — are calibrated once from population statistics of pre-RoPE K/V activations on a reference corpus and shipped as constants. To deploy on a new transformer architecture, the user supplies (a) a checkpoint and (b) optional API scaling factors (§4.1, Appendix D Table 7). Default factors of `(1.0, 1.0, 1.0, 1.0)` (IDENTITY) are appropriate for any model.

**Code availability.** Source code for the fused selection kernel, the palette-aware paged-attention kernel, the 16-format codebook implementations, and the inference-engine integration is available as supplementary material accompanying this paper. The archive contains:

- `*.cu` and `*.cuh` — CUDA kernel sources and headers (selection kernel, attention kernel, format reconstruction)
- `*.rs` — Rust inference-engine sources (palette-4 selection logic, format encode/decode, runtime sink detection, threshold computation)
- `Cargo.toml` and `build.rs` — build configuration
- `README.md` — file-to-paper-section mapping showing which source files implement which §3 / §4 / Appendix claims

Build: `cargo build --release --features cuda`. Hardware requirement: Ada-class GPU (sm_89) or later for the FP8 tensor core paths; Hopper (sm_90) is supported. The supplementary archive is self-contained; it does not include model weights (publicly available from the cited references) or evaluation corpora (WikiText-2 from [Merity et al., 2017]).

### I. Extended Perplexity Analysis

This appendix extends the cross-model summary of §4.4 (Table 3) along two dimensions: a per-level Pareto curve on Qwen3-8B against uniform-quantization baselines (Table 9, Figure 1) and a context-length sweep from 512 to 4096 tokens (Table 10, Figure 2).

**Detailed Pareto curve on Qwen3-8B.** To make the quality-vs-compression trade-off explicit, we evaluate all ten PalQuant operating points against three uniform-quantization configurations on Qwen3-8B (representative dense model). The uniform-quantization baselines are: the symmetric **Q8/Q8** configuration (Q8_0 for both K and V, 8.50 BPE), the asymmetric **Q8/Q4** configuration (Q8_0 for K, Q4_0 for V — the standard llama.cpp recommendation, 6.50 BPE), and the symmetric **Q4/Q4** configuration (4.50 BPE). Table 9 reports the full level-by-level grid; Figure 1 plots PPL against BPE through this grid. **PalQuant C1 strictly Pareto-dominates Q8/Q4** (6.30 BPE / 9.90 PPL vs 6.50 / 9.94) and is **Pareto-equivalent with Q8/Q8** at -26% BPE (9.90 vs 9.88, +0.02 PPL within measurement noise). Below Q4/Q4's 4.50 BPE floor, PalQuant C5–C10 extend the achievable compression range to 2.16 BPE; the PPL gaps in this region (e.g., Q4/Q4 9.96 vs C5 10.38 at comparable BPE) constitute the §4.5 dissociation case rather than quality regression — the multi-session test (§4.6) passes at all 11 levels through C10 while Q4_0 fails at 3.56× CR.

**Table 9.** WikiText-2 perplexity on Qwen3-8B, 2048-token context (field convention; see §4.1), 50K tokens, Q4_K_M weights. KV configurations listed in compression-ascending order. PalQuant BPE values are the per-Qwen3-8B values from Table 1 (BPE = 16 / CR).

| KV configuration | BPE | CR | PPL |
|---|---:|---:|---:|
| F16 KV       | 16.00 | 1.00× |  9.88 |
| Q8/Q8 KV     |  8.50 | 1.88× |  9.88 |
| Q8/Q4 KV     |  6.50 | 2.46× |  9.94 |
| Q4/Q4 KV     |  4.50 | 3.56× |  9.96 |
| PalQuant C1  |  6.30 | 2.54× |  9.90 |
| PalQuant C2  |  6.02 | 2.66× | 10.11 |
| PalQuant C3  |  5.50 | 2.91× | 10.16 |
| PalQuant C4  |  4.83 | 3.31× | 10.32 |
| PalQuant C5  |  4.35 | 3.68× | 10.38 |
| PalQuant C6  |  3.73 | 4.29× | 10.88 |
| PalQuant C7  |  3.59 | 4.46× | 10.93 |
| PalQuant C8  |  3.33 | 4.80× | 10.89 |
| PalQuant C9  |  2.97 | 5.39× | 11.07 |
| PalQuant C10 |  2.16 | 7.41× | 12.72 |

**Figure 1** (referenced from Table 9, deferred to typesetting): Qwen3-8B PPL-vs-BPE curve traced through PalQuant C1–C10 at 2048 context, with horizontal reference lines at the PPL achieved by Q8/Q8 (8.50 BPE), Q8/Q4 (6.50 BPE), and Q4/Q4 (4.50 BPE). The cleanest Pareto wins: C1 strictly below Q8/Q4 on both axes (6.30/9.90 vs 6.50/9.94); C1 at -26% BPE relative to Q8/Q8 with PPL within noise (9.90 vs 9.88). C5–C10 extend the curve below Q4/Q4's 4.50 BPE floor at higher PPL — the §4.5 dissociation case rather than quality regression, since the multi-session test (§4.6) passes throughout while Q4_0 fails at 3.56× CR.

**Context-length sweep.** Tables 3 and 9 report at the field convention 2048-token context. The trajectory of PPL across context lengths reveals whether configurations behave consistently as sequences grow, or whether the convention point is a structurally special regime. We sweep context length geometrically from 512 to 4096 tokens on Qwen3-8B across all configurations and report PPL at each operating point. Table 10 reports the sweep; Figure 2 plots the trajectories.

**Table 10.** Qwen3-8B WikiText-2 perplexity vs context length, across KV configurations. Q4_K_M weights, evaluation over 50K tokens at each context length. Lower is better. The 2048 row reproduces Tables 3 and 9 anchor values (field convention); the 4096 row uses a chunked-window methodology rather than the single-window methodology of the rest of this table, with absolute values differing from a hypothetical chunked-2048 by at most ~0.25 PPL.

| Context | Q8/Q8 | Q8/Q4 | Q4/Q4 | C1 | C5 | C9 | C10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512   | 13.28 | 13.37 | 13.43 | 13.31 | 14.12 | 14.81 | 16.84 |
| 1024  | 11.22 | 11.30 | 11.32 | 11.23 | 11.94 | 12.64 | 14.33 |
| 2048  |  9.88 |  9.94 |  9.96 |  9.90 | 10.38 | 11.07 | 12.72 |
| 4096  | 10.74 | 10.62 | 10.77 | 10.12 | 10.77 | 12.45 | 15.21 |

**Trajectory shape and the 2048 convention.** F16 KV — full-precision baseline, no compression — sits at 9.88 PPL at 2048 vs 9.14 at 4096. The shorter context is not "easier" for the model; the gap reflects Q4_K_M weight quantization calibrated against 2048-context activation distributions, putting the weight stack at its tuned operating point. The consequence: at 2048, every KV configuration sits within 0.1 PPL of F16 (Q8/Q8: +0.00, Q4/Q4: +0.08, PalQuant C1: +0.02). At 4096, the same configurations cost 0.5–6 PPL.

KIVI, KVQuant, TurboQuant, and Kitty all evaluate at 2048 — the regime where 2-bit, 3-bit, and 4-bit formats can claim near-losslessness. Combined with §4.5: PPL evaluated where it cannot discriminate between configurations is doubly non-informative, and the deployment regime sits outside it.

**Figure 2** (referenced from Table 10, deferred to typesetting): Qwen3-8B PPL vs context length (log x-axis from 512 to 4096), one curve per KV configuration. The expected reading: configurations are vertically offset by their bitrate cost but share trajectory shape; the dramatic gap between 2048 (within 0.1 PPL of F16) and 4096 (0.5–6 PPL above F16) is shared across all configurations, evidence that the 2048 minimum is a property of the weight-quantization stack rather than any specific KV cache format.

### J. Full Throughput Sweep

This appendix reports the full throughput measurement set referenced from §4.9. All measurements are taken on a single RTX 4090 Mobile (16 GB) with Q4_K_M weights, fused PalQuant CUDA kernel, paged-attention prefill + decode. Table 11 lists every (model, KV configuration, concurrent-session count) cell measured. The *Valid* column reports the §4.6 multi-session story rewrite test outcome at that level: ✓ = passes, − = fails. *Bulk t/s* is the prefill processing rate (input-side); *Single t/s* is the aggregate decode throughput summed across the concurrent sessions; per-session decode rate is *Single t/s* divided by the session count.

The Q4_0, Q4_1, and Q4_KS rows on Llama-3.2-3B and Qwen3-30B-A3B reach throughput regimes comparable to PalQuant's C5–C7 levels but fail the multi-session test at those compression ratios — illustrating the §4.5 PPL/structural dissociation as a deployment concern rather than a measurement artifact: a configuration can be fast and still wrong.

**Table 11.** Combined throughput sweep across Llama-3.2-3B, Qwen3-8B, and Qwen3-30B-A3B at every measured (KV configuration, session count) cell. RTX 4090 Mobile, 16 GB, Q4_K_M weights. *Valid* indicates §4.6 multi-session test outcome.

| Model | KV mode | Sessions | Valid | Bulk t/s | Single t/s | Compress | Peak tokens |
|---|---|---:|:---:|---:|---:|---:|---:|
| Llama-3.2-3B | F32 | 1 | ✓ | 3785 | 71.5 | – | 654 |
| Llama-3.2-3B | F16 | 1 | ✓ | 5460 | 61.4 | – | 654 |
| Llama-3.2-3B | F16 | 10 | ✓ | 5300 | 466 | – | 6,550 |
| Llama-3.2-3B | R16 | 1 | ✓ | 5843 | 57.8 | – | 654 |
| Llama-3.2-3B | BF16 | 4 | ✓ | 5979 | 230 | – | 2,618 |
| Llama-3.2-3B | Q8_0 | 1 | ✓ | 5166 | 59.7 | 1.88× | 654 |
| Llama-3.2-3B | Q8_0 | 40 | ✓ | 4829 | 866 | 1.88× | 26,250 |
| Llama-3.2-3B | Q8_1 | 4 | ✓ | 5659 | 192 | 1.78× | 2,618 |
| Llama-3.2-3B | Q8_KS | 4 | ✓ | 5837 | 208 | 1.78× | 2,618 |
| Llama-3.2-3B | Q8_Q4 | 1 | ✓ | 5169 | 52.8 | 2.29× | 654 |
| Llama-3.2-3B | Q8_Q4 | 4 | ✓ | 5635 | 190 | 2.29× | 2,618 |
| Llama-3.2-3B | Q4_0 | 4 | − | 5879 | 213 | 3.56× | 2,618 |
| Llama-3.2-3B | Q4_1 | 4 | − | 5652 | 190 | 3.20× | 2,618 |
| Llama-3.2-3B | Q4_KS | 4 | − | 5894 | 206 | 3.20× | 2,618 |
| Llama-3.2-3B | C0 | 10 | ✓ | 4262 | 370 | 1.88× | 6,550 |
| Llama-3.2-3B | C1 | 5 | ✓ | 5281 | 249 | 2.28× | 3,272 |
| Llama-3.2-3B | C2 | 5 | ✓ | 5169 | 247 | 2.49× | 3,272 |
| Llama-3.2-3B | C3 | 64 | ✓ | 4177 | 868 | 2.87× | 41,980 |
| Llama-3.2-3B | C4 | 128 | ✓ | 3929 | 897 | 3.18× | 83,968 |
| Llama-3.2-3B | C5 | 256 | ✓ | 3854 | 887 | 3.42× | 167,904 |
| Llama-3.2-3B | C6 | 10 | ✓ | 3484 | 296 | 3.62× | 6,550 |
| Llama-3.2-3B | C7 | 10 | ✓ | 4632 | 356 | 3.92× | 6,550 |
| Llama-3.2-3B | C8 | 10 | ✓ | 2926 | 308 | 3.96× | 6,550 |
| Llama-3.2-3B | C9 | 10 | ✓ | 1786 | 238 | 4.28× | 6,550 |
| Llama-3.2-3B | C10 | 5 | ✓ | 1788 | 164 | 5.02× | 3,272 |
| Qwen3-8B | BF16 | 1 | ✓ | 1327 | 29.4 | – | 636 |
| Qwen3-8B | F16 | 1 | ✓ | 2065 | 31.8 | – | 636 |
| Qwen3-8B | F16 | 2 | ✓ | 1801 | 61.0 | – | 1,272 |
| Qwen3-8B | BF16 | 4 | ✓ | 2128 | 121 | – | 2,546 |
| Qwen3-8B | Q8_0 | 4 | ✓ | 1954 | 116 | 1.88× | 2,546 |
| Qwen3-8B | Q8_0 | 16 | ✓ | 2133 | 385 | 1.88× | 10,212 |
| Qwen3-8B | Q4_0 | 32 | ✓ | 2061 | 537 | 3.56× | 20,428 |
| Qwen3-8B | C0 | 10 | ✓ | 1981 | 215 | 1.90× | 6,370 |
| Qwen3-8B | C1 | 10 | ✓ | 1953 | 211 | 2.54× | 6,370 |
| Qwen3-8B | C2 | 10 | ✓ | 2107 | 209 | 2.66× | 6,370 |
| Qwen3-8B | C3 | 10 | ✓ | 2107 | 214 | 2.91× | 6,370 |
| Qwen3-8B | C4 | 10 | ✓ | 2074 | 194 | 3.31× | 6,370 |
| Qwen3-8B | C5 | 10 | ✓ | 1860 | 195 | 3.68× | 6,370 |
| Qwen3-8B | C6 | 10 | ✓ | 1650 | 172 | 4.29× | 6,370 |
| Qwen3-8B | C7 | 32 | ✓ | 1712 | 459 | 4.46× | 20,428 |
| Qwen3-8B | C8 | 64 | ✓ | 1431 | 573 | 4.81× | 40,830 |
| Qwen3-8B | C9 | 10 | ✓ | 1140 | 131 | 5.39× | 6,370 |
| Qwen3-8B | C10 | 5 | ✓ | 1263 | 97.4 | 7.42× | 3,182 |
| Qwen3-30B-A3B | F16 | 1 | ✓ | 422 | 9.4 | – | 626 |
| Qwen3-30B-A3B | BF16 | 1 | ✓ | 496 | 14.6 | – | 626 |
| Qwen3-30B-A3B | BF16 | 10 | ✓ | 1741 | 110 | – | 6,270 |
| Qwen3-30B-A3B | Q8_0 | 20 | ✓ | 1984 | 176 | 1.88× | 12,580 |
| Qwen3-30B-A3B | Q8_0 | 32 | ✓ | 2152 | 262 | 1.88× | 20,108 |
| Qwen3-30B-A3B | Q4_0 | 4 | − | 1744 | 62.1 | 3.56× | 2,506 |
| Qwen3-30B-A3B | Q4_0 | 48 | − | 2026 | 398 | 3.56× | 30,148 |
| Qwen3-30B-A3B | C0 | 10 | ✓ | 1754 | 137 | 1.98× | 6,270 |
| Qwen3-30B-A3B | C1 | 10 | ✓ | 1665 | 130 | 2.54× | 6,270 |
| Qwen3-30B-A3B | C2 | 10 | ✓ | 1712 | 131 | 2.74× | 6,270 |
| Qwen3-30B-A3B | C3 | 10 | ✓ | 1748 | 132 | 2.98× | 6,270 |
| Qwen3-30B-A3B | C4 | 10 | ✓ | 1710 | 129 | 3.41× | 6,270 |
| Qwen3-30B-A3B | C5 | 10 | ✓ | 1680 | 129 | 3.67× | 6,270 |
| Qwen3-30B-A3B | C6 | 10 | ✓ | 1478 | 122 | 4.16× | 6,270 |
| Qwen3-30B-A3B | C7 | 64 | ✓ | 1771 | 349 | 4.22× | 40,190 |
| Qwen3-30B-A3B | C8 | 120 | ✓ | 1512 | 603 | 4.69× | 75,366 |
| Qwen3-30B-A3B | C9 | 10 | ✓ | 1205 | 123 | 5.24× | 6,270 |
| Qwen3-30B-A3B | C10 | 5 | ✓ | 1086 | 26.8 | 7.04× | 3,132 |

### K. Palette-4 vs Per-Block Ideal Compression

PalQuant restricts each (chunk, head) palette to 4 formats with a 2-bit per-block selector (§3.2). This appendix quantifies the CR cost of that constraint against two reference baselines: **per-block ideal** — each of the 128 blocks individually picks its best codebook format, the upper bound on per-block adaptation — and **worst-1** — one format per head, conservative enough to handle the most demanding block, the per-head-sharing lower bound.

**Table 12.** Palette-4 compression vs per-block ideal and worst-1 baselines across operating levels. Per-block ideal is the upper bound (every block picks independently); worst-1 is the lower bound (one format per head). All numbers include palette overhead (4 × format tag + 4 × FP16 outer scale + 128 × 2-bit selector = 44 bytes per side, §3.2). Absolute CRs vary modestly across models (Table 1); per-level relative gaps are similar.

| Level | Per-block ideal CR | Worst-1 CR | Palette-4 CR | Pal-4 vs ideal | Pal-4 vs worst-1 |
|---|---:|---:|---:|---:|---:|
| C0  | 1.99× | 1.83× | 1.97× | -1.0% | +7.6%  |
| C1  | 2.54× | 1.84× | 2.52× | -0.8% | +36.8% |
| C2  | 2.75× | 1.91× | 2.71× | -1.5% | +42.1% |
| C3  | 2.99× | 2.05× | 2.96× | -1.0% | +44.3% |
| C4  | 3.42× | 2.37× | 3.37× | -1.5% | +42.5% |
| C5  | 3.68× | 2.60× | 3.62× | -1.6% | +39.3% |
| C6  | 4.19× | 2.63× | 4.11× | -1.9% | +56.2% |
| C7  | 4.24× | 2.70× | 4.17× | -1.7% | +54.5% |
| C8  | 4.71× | 3.46× | 4.62× | -1.9% | +33.4% |
| C9  | 5.33× | 3.93× | 5.21× | -2.3% | +32.7% |
| C10 | 7.14× | 4.61× | 6.93× | -2.9% | +50.4% |

Two readings. Palette-4 captures 97–99% of the per-block ideal CR — the 4-format constraint costs at most 2.9% (at C10) and typically <2% at C0–C8. Against the worst-1 baseline (one format per head, equivalent to no palette), palette-4 delivers 7.6–56.2% gain, the §4.10 ablation result. The combination is the architectural claim made concrete: palette-4 is close to the per-block ideal in compression efficiency while substantially outperforming any single-format-per-head choice — at fixed 44 B per-side metadata.

### L. Palette-4 Selection Trace: Qwen3-8B at C5

This appendix illustrates the palette-4 mechanism in action with a real 20-chunk inference trace on Qwen3-8B at C5 compression. The data shows the per-(chunk, head, side) format heterogeneity that motivates the §3.2 budgeted-slot allocation. K palettes are nearly uniform across heads (mostly Q3_0 with Q3_1 fallback), while V palettes vary, with some heads admitting Q4_0 or Q4_1 in conservative slots.

**Table 15.** Per-chunk palette format selection across the 20-chunk trace (n_kv_head=8, palette size 4 per (chunk, head, side)). Each cell shows the multiset of formats across all 32 (head, slot) palette entries for that chunk and side. Chunk 19 is the active scratch buffer (K=R16, V=F16) before commit at the next chunk boundary.

| Chunk | K palette (32 (head, slot) pairs) | V palette (32 (head, slot) pairs) |
|---:|---|---|
|  0 | Q3_0×24, Q3_1×8 | Q3_0×23, Q3_1×6, Q4_0×2, Q4_1×1 |
|  1 | Q3_0×25, Q3_1×7 | Q3_0×17, Q3_1×6, Q4_0×6, Q4_1×1, Q8_0×2 |
|  2 | Q3_0×24, Q3_1×8 | Q3_0×20, Q3_1×8, Q4_0×2, Q4_1×2 |
|  3 | Q3_0×24, Q3_1×8 | Q3_0×21, Q3_1×6, Q4_0×5 |
|  4 | Q3_0×24, Q3_1×8 | Q3_0×20, Q3_1×8, Q4_0×2, Q4_1×2 |
|  5 | Q3_0×24, Q3_1×8 | Q3_0×17, Q3_1×4, Q4_0×9, Q4_1×1, Q8_0×1 |
|  6 | Q3_0×24, Q3_1×8 | Q3_0×19, Q3_1×6, Q4_0×6, Q8_0×1 |
|  7 | Q3_0×24, Q3_1×8 | Q3_0×19, Q3_1×3, Q4_0×8, Q4_1×2 |
|  8 | Q3_0×24, Q3_1×8 | Q3_0×14, Q3_1×5, Q4_0×9, Q4_1×2, Q8_0×2 |
|  9 | Q3_0×24, Q3_1×8 | Q3_0×20, Q3_1×4, Q4_0×5, Q4_1×3 |
| 10 | Q3_0×24, Q3_1×8 | Q3_0×21, Q3_1×4, Q4_0×6, Q8_0×1 |
| 11 | Q3_0×24, Q3_1×8 | Q3_0×19, Q3_1×5, Q4_0×5, Q4_1×3 |
| 12 | Q3_0×24, Q3_1×8 | Q3_0×19, Q3_1×6, Q4_0×6, Q4_1×1 |
| 13 | Q3_0×24, Q3_1×8 | Q3_0×14, Q3_1×6, Q4_0×7, Q4_1×3, Q8_0×2 |
| 14 | Q3_0×25, Q3_1×7 | Q3_0×20, Q3_1×6, Q4_0×5, Q4_1×1 |
| 15 | Q3_0×25, Q3_1×7 | Q3_0×22, Q3_1×6, Q4_0×1, Q4_1×3 |
| 16 | Q3_0×24, Q3_1×8 | Q3_0×16, Q3_1×6, Q4_0×6, Q4_1×3, Q8_0×1 |
| 17 | Q3_0×24, Q3_1×8 | Q3_0×19, Q3_1×3, Q4_0×7, Q4_1×3 |
| 18 | Q3_0×24, Q3_1×8 | Q3_0×21, Q3_1×5, Q4_0×3, Q4_1×3 |
| 19 | (scratch: R16) | (scratch: F16) |

K is essentially constant across the trace: 24–25 of the 32 (head, slot) pairs select Q3_0 (3.5 BPE) and the remainder Q3_1 (4.0 BPE). V palettes show real per-chunk variation in conservative-format escalation (Q4_0, Q4_1, occasional Q8_0): some chunks fit V cleanly into a tight Q3_0/Q3_1 mix, others spread across five formats with a few Q8_0 escalations.

**Table 16.** Per-head palette layout for chunk 18, showing all 4 palette slots p0–p3 for K and V independently. Format names abbreviated: Q30 = Q3_0, Q31 = Q3_1, Q40 = Q4_0, Q41 = Q4_1.

| Head | K palette (p0, p1, p2, p3) | V palette (p0, p1, p2, p3) |
|---:|---|---|
| 0 | Q30, Q30, Q30, Q31 | Q30, Q30, Q30, Q40 |
| 1 | Q30, Q30, Q30, Q31 | Q30, Q30, Q30, Q31 |
| 2 | Q30, Q30, Q30, Q31 | Q30, Q30, Q30, Q31 |
| 3 | Q30, Q30, Q30, Q31 | Q30, Q30, Q30, Q40 |
| 4 | Q30, Q30, Q30, Q31 | Q30, Q30, Q30, Q31 |
| 5 | Q30, Q30, Q30, Q31 | Q30, Q30, Q30, Q41 |
| 6 | Q30, Q30, Q30, Q31 | Q30, Q31, Q40, Q41 |
| 7 | Q30, Q30, Q30, Q31 | Q30, Q30, Q31, Q41 |

All eight heads' K palettes converge on the same pattern (three slots of Q3_0, one slot of Q3_1). V palettes diverge: heads 1, 2, 4 keep an all-Q3 V palette; heads 0, 3, 5 escalate one slot to a 4-bit format; head 6 escalates from p1 onward; head 7 from p2. This per-head V variation is exactly what palette-4 captures and worst-1 (one format per head) cannot — quantified at the aggregate level in Appendix K.

**Table 17.** Per-block palette index grids for chunk 18, showing which of the 4 palette slots each of the 128 blocks routes to. Each cell is a 2-bit index $\in\{0, 1, 2, 3\}$. Both palettes here share the same key — slots p0–p2 hold Q3_0, slot p3 holds Q3_1 — so cell value 3 selects the more conservative format. By construction each slot holds exactly 32 blocks (§3.2), so 32 of the 128 cells in each grid are 3.

```
   K (head 1)              V (head 4)
   1 3 1 1 2 2 2 2         0 0 1 1 0 1 3 2
   2 0 0 1 1 0 2 3         0 3 2 0 1 3 3 2
   0 3 0 3 1 1 2 0         3 2 2 1 3 1 2 0
   2 3 0 3 2 2 0 1         1 0 3 2 3 2 0 3
   1 0 3 0 0 3 0 1         1 0 1 2 1 1 3 0
   2 3 1 2 1 2 1 0         2 3 0 2 2 3 3 1
   0 0 2 3 3 1 3 3         3 1 0 2 0 1 1 2
   0 3 1 1 2 3 2 3         2 0 3 3 3 0 1 0
   0 0 1 2 0 3 1 2         0 0 0 0 0 1 3 0
   1 3 1 3 3 1 3 3         0 2 1 0 0 3 3 0
   3 0 2 1 1 0 3 0         2 1 3 1 3 0 1 0
   3 1 1 0 2 0 3 0         0 0 2 3 3 0 1 3
   1 2 2 0 3 3 2 2         2 0 2 1 1 2 3 1
   1 2 0 2 1 3 2 2         1 3 2 1 1 1 3 1
   3 3 0 2 1 0 1 0         3 2 2 2 2 1 2 1
   0 1 2 0 2 2 3 1         2 2 3 3 3 2 2 2
```

Block-level routing is structurally heterogeneous: the 32 conservative-slot blocks are scattered (not contiguous), reflecting that block difficulty does not correlate simply with token position. The K and V grids differ in the spatial arrangement of conservative blocks despite both committing 32 to slot p3.

**Table 18.** Aggregate compression distribution across the 20-chunk trace, by element count. The two scratch families (R16 K, F16 V) account for the active in-flight chunk 19 before commit.

| Format | BPE | Elements | Share |
|---|---:|---:|---:|
| Q3_0 |  3.5 | 239,997 | 52.1% |
| Q4_0 |  4.5 |  78,681 | 17.1% |
| Q8_0 |  8.5 |  45,488 |  9.9% |
| Q3_1 |  4.0 |  44,393 |  9.6% |
| Q4_1 |  5.0 |  29,201 |  6.3% |
| F16  | 16.0 |  11,520 |  2.5% (V scratch) |
| R16  | 16.0 |  11,520 |  2.5% (K scratch) |

Q3_0 dominates at 52% of elements; the structural-template ladder (Q1_S, Q0_*) does not appear at C5, consistent with the §3.6 ladder design — those formats first admit at C6. The two scratch families together account for 5%, the geometric 1-in-20 chunk rate (Appendix E).

### M. RULER Detailed Results

This appendix reports the full RULER cell-by-cell results referenced from §4.7. RULER [Hsieh et al., 2024] generates synthetic long-context test instances; we evaluate Qwen3-8B with PalQuant and uniform-quantization baselines on four task types: NIAH-Single (single needle in haystack), NIAH-MultiKey-2 (two-key retrieval), VT (variable tracking, multi-step reasoning), and CWE (common-words extraction, aggregation). Sample counts vary per cell because the test harness samples randomly from the full grid; cells with low sample counts have wide confidence intervals.

**Table 19.** RULER per-task pass counts at 4096-token context, Qwen3-8B. Format: passes / total = pass-rate. CWE row excluded from §4.7 averaging because Qwen3-8B fails CWE at the F16 baseline (0/2) — a model-capability ceiling unrelated to KV compression.

| Configuration | NIAH-Single | NIAH-MultiKey-2 | VT | CWE |
|---|---:|---:|---:|---:|
| F16 KV | 30/33 = 91% | 29/29 = 100% | 29/29 = 100% | 0/2 = 0%* |
| Q8/Q8 KV | 23/26 = 88% | 17/18 = 94% | 19/20 = 95% | — |
| Q8/Q4 KV (asymm) | 20/23 = 87% | 21/24 = 88% | 20/24 = 83% | — |
| Q4/Q4 KV | 23/24 = 96% | 13/13 = 100% | 15/15 = 100% | — |
| PalQuant C5 | 17/18 = 94% | 23/23 = 100% | 12/14 = 86% | — |
| PalQuant C9 | 31/35 = 89% | 34/37 = 92% | 36/48 = 75% | 0/2 = 0%* |
| PalQuant C10 | 23/27 = 85% | 21/28 = 75% | 15/21 = 71% | 0/17 = 0%* |

*CWE failure is model-baseline-bound: Qwen3-8B cannot solve CWE at F16. Sample counts on CWE are low because the harness budget allocated more samples to tasks that produced discriminating signal across configurations.

**Failure mode analysis.** Both F16 and C10 NIAH failures share a digit-truncation pattern: the model finds the correct region of context but drops the final token of the retrieved answer, often falling into a repetition loop on the truncated form. Representative F16 failures observed during data collection (NIAH-MultiKey-2): predicted "9921 47977" vs expected "99211 47977"; "5026 5026 5026 12971" vs "50265 12971"; "4708 53758 4708 53758" vs "47088 53758". Representative C10 failures: "3442" vs "34482"; "5342" vs "53423"; "6368" vs "63686"; "9849" vs "98193"; "3057 89324" vs "30577 89324". On the matched-prompt case (sample 53, expected "47088"), F16 and C10 both fail with the same truncation (F16: "4708 53758 4708 53758"; C10: "4708 4708"). The pattern is a Qwen3-8B output-formatting behaviour — present at F16 baseline and amplified by aggressive compression rather than introduced by it. A small subset of C10 failures show structural breakdown unique to maximum compression: collapsed answers ("0" expecting a 5-digit number; "VAR_A = 1" in VT) and token-degeneration sequences ("ffffffff..."). This is consistent with the §4.5 dissociation: PalQuant's most aggressive operating point amplifies the base model's existing failure mode without introducing a new catastrophic regime; the §4.6 multi-session test passes at C10 because session-discriminating identity tokens occupy higher-relevance K slots that PalQuant routes to conservative palette positions.

**Table 20.** Long-context cells (8K and above) measured during the run. These are too sparsely sampled to support a per-cell pass-rate claim but are reported for completeness; their sole purpose here is to demonstrate that the harness exercised long-context paths and that no configuration produced systematic crashes or pathological outputs at 8K–16K on Qwen3-8B.

| Configuration | Context | Task | Result |
|---|---:|---|---|
| F16 KV  | 8192  | NIAH-Single | 1/1 |
| F16 KV  | 8192  | NIAH-MultiKey-2 | 1/1 |
| F16 KV  | 16384 | NIAH-Single | 2/2 |
| Q8/Q8 KV | 8192 | NIAH-Single | 1/1 |
| Q8/Q8 KV | 8192 | VT | 1/1 |
| Q4/Q4 KV | 8192 | NIAH-MultiKey-2 | 0/1 |
| PalQuant C5  | 8192  | VT | 0/1 |
| PalQuant C9  | 8192  | NIAH-Single | 1/1 |
| PalQuant C9  | 8192  | VT | 1/1 |
| PalQuant C9  | 16384 | NIAH-Single | 1/1 |
| PalQuant C10 | 8192  | CWE | 0/3 |
| PalQuant C10 | 8192  | NIAH-Single | 0/1 |
| PalQuant C10 | 8192  | NIAH-MultiKey-2 | 2/3 |
| PalQuant C10 | 16384 | NIAH-Single | 0/1 |
| PalQuant C10 | 16384 | VT | 0/1 |

**Reading.** The 4K data in Table 19 supports the §4.7 dissociation reading: PalQuant C5 matches F16 NIAH within noise (98% vs 95%) at 3.68× CR; C9 retains 90% NIAH at 5.39× CR; VT (multi-step reasoning) costs ~14 points at C5 and ~25 points at C9 relative to F16. C10 at 7.41× CR drops retrieval to 80% and VT to 71%, the same Qwen3-8B configuration that elevates PPL +28.7% (Table 3) yet passes the §4.6 multi-session test. The asymmetric Q8/Q4 baseline at 6.50 BPE shows the partial Pareto trade discussed in §4.7: PalQuant matches or exceeds Q8/Q4 retrieval (87%) at less than half the BPE (C5 at 98%/4.35 BPE; C9 at 90%/2.97 BPE), but Q8/Q4's multi-step reasoning (83%) outperforms C9 (75%) and C10 (71%) at the cost of 2.2× the bits. Adaptive selection wins on retrieval at all BPE; on reasoning, fixed higher-BPE quantization is competitive in the sub-3-BPE compression band.

**Scope.** This evaluation is Qwen3-8B-only and 4K-dominant. Without YaRN scaling enabled in the inference engine, the Qwen3 family cannot evaluate beyond its native 32K position-embedding range; YaRN integration was not deployed in time for these results. Extension to Llama-3.2-3B, Qwen3-30B-A3B, and longer contexts is straightforward and is left for the camera-ready or rebuttal phase.
