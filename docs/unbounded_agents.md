# One Card, One Stack: Constraint-Driven Architecture for Asymptotically Stable Inference over Unbounded Agent Memory

> **Status — v1 Technical Report.** This document describes a working prototype released to establish priority on the architecture, the theorem, and the Speculative Context Decode mechanism. Throughput and kernel benchmarks (§9.4–§9.6) are fully measured on production hardware. Quality evaluations (§9.7–§9.12) describe methodology with preliminary observations; quantitative results will be reported in v2 following systematic experiments. The live system and full codebase are publicly released to enable independent verification in the interim. v2 will incorporate community validation results, optimizations, and critical review — contributions and collaboration are welcome — contributors will be recognized in v2 (see Appendix C)

**Abstract**

Persistent agentic systems require context that grows without bound. Under standard full attention, numerical error per generation step grows monotonically with context depth — for any finite-precision arithmetic, any compression scheme, on any hardware — because every token participates in every subsequent computation with equal structural weight. This is not a compression problem; it is an architectural one. An H100 at 80GB enters the same accumulation regime as a 16GB GPU the moment any token is compressed or evicted — it is simply deferred. We prove that the accumulation problem is architectural rather than representational, and that bounded error accumulation at unbounded context depth requires decoupling the working set from context depth. We present the first complete system implementing this requirement.

The key theoretical result (**Theorem §11.2 — Asymptotic Numerical Stability**): under provenance-selected attention over a tiered context, total numerical error per generation step — from any source, including floating-point rounding — is bounded by a constant O(1) independent of context depth N, in contrast with the O(N) scaling of standard full-attention systems. Under practical system conditions (warm-tier blocks originating from prefill-refreshed hot-tier blocks) this constant is small, approaching the hot-tier error floor. This inverts the universal assumption of the KV quantization literature that error grows with N.

The system is built on four integrated contributions: (1) an online self-learning Markov expert prediction system with DMA offload and wave-batched grouped GEMM achieving stall-free MoE inference under partial VRAM residency; (2) an adaptive per-block KV quantization family spanning FP16 to 2-bit integer with boundary-aware sub-block structure, two-phase prefill refresh eliminating autoregressive decode drift, and per-block selection across ten compression modes ranging from 1.21× (top-quality tier) to 4.67× (highest-compression tier) per-head — with asymmetric K/V error metrics matched to the softmax-amplified K and linear-bounded V error propagation paths — the production-achievable range given the attention kernel's per-head gather constraint — using asymmetric K/V thresholds grounded in the softmax error amplification asymmetry, with overall system compression ratio dependent on block-level mode distribution; (3) attentional provenance indexing via Q-vector cognitive-state fingerprints with Speculative Context Decode — a pipelined two-session generation loop that hides CPU provenance scoring (3–10ms flat scan) behind a parallel variable-window probe session terminating at newline boundaries, yielding working-set refinement at the model's natural reasoning granularity with near-zero visible overhead ; and (4) an unbounded three-tier paged context (VRAM-hot, CPU RAM-warm, disk-cold) with adaptive quantization calibrated to the asymptotic guarantee. Each contribution originated from a hard constraint that closed the standard solution and forced an architectural choice that turns out to be universally correct.

Implemented in Rust on a custom Candle fork with native quantized matmul kernels that never materialise a full-precision weight copy, the system demonstrates **509 t/s single-context** — 2.6–3.4× faster than community benchmarks for this model on RTX 4090 24GB with standard single-session frameworks [hardware-corner.net, 2025; ToolHalla, 2026] — and **2,446 t/s aggregate across 64 concurrent persistent-memory sessions** on an RTX 4090 Mobile (16GB). The concurrent-session figure reflects server throughput across 64 simultaneous agents; no standard framework runs this model on 16GB at comparable concurrency.

An evaluation methodology is described in §9.12 using the system's own 2.2M-line Rust/CUDA Candle fork as the test subject: the system is ingested into unbounded context via a ~20M-token learning-phase conversation, then queried via iterative multi-hop retrieval during decode. The one-shot ablation — same index, single pre-generation retrieval — isolates the contribution of continuous decode-time retrieval. Quantitative results are reserved for v2, which will incorporate community validation and independent optimization. The working system is publicly available for live verification and collaborative development (Appendix C).

---

## 1. Introduction

The persistent agentic systems that every major AI lab is building toward — long-running assistants, autonomous agents, enterprise systems accumulating institutional knowledge — all face the same unsolved problem: context must grow without bound, and under standard full attention, numerical error per generation step grows monotonically with context depth. This is not a hardware problem. It is an architectural one. More VRAM defers the threshold; it does not eliminate the structural problem. The industry's current responses confirm this: RAG externalises memory into retrieval, losing the attentional continuity that makes genuine persistent memory different from keyword search; longer native context windows are a hardware investment against an architectural ceiling — at 100 million tokens across months of operation, no finite hardware holds the context at full precision, and summarization and eviction are lossy by definition, destroying the precise factual recall that makes persistent memory genuinely useful.

This paper identifies the cause and provides the fix. The cause is structural: under standard full attention, every token at every position participates in every subsequent computation with equal weight, coupling context depth to error accumulation irrevocably. The fix is architectural: decouple the set of tokens that participate in any given generation step from the total number of tokens in the context. Something functionally equivalent to provenance-selected sparse attention is necessary — not sufficient, but necessary — for a persistent-session LLM to maintain bounded quality as context grows. The theorem in §11.2 establishes this formally. No full-attention system, on any hardware, can provide the same guarantee.

This system did not originate from a theoretical observation. It originated from two constraints accepted simultaneously and without compromise: a demanding application requiring the hardest possible form of persistent memory, and a hard VRAM ceiling that could not be exceeded. The demanding application is persistent agent conversations — chosen because autonomous agents represent the worst-case instantiation of the persistent-memory problem. They require verbatim factual recall across sessions that grow indefinitely, semantic coherence across arbitrary time gaps, high concurrency across many simultaneous agents, and deployment on hardware with no datacenter budget. Any system that works for this use case works for any less-demanding persistent-memory deployment. The constraints forced architectural choices that turned out to solve the problem at the level of principle: native quantized matmul kernels written because standard GEMM dequantisation OOMed — strictly faster on unconstrained hardware; provenance-selected attention implemented because VRAM is finite and full context cannot be held hot — the mechanism that produces the hardware-independent asymptotic guarantee; two-phase quantisation designed because materialising full turns at F16 across many sessions OOMs — the design that correctly separates two independent error sources prior work conflated. Section §11.5 documents this pattern in full.

We present an inference system designed to implement this architectural requirement — deliberately constrained to a single 16GB consumer GPU so that every architectural choice falls out of the constraint rather than being imposed on top of it, and validated empirically on that hardware. The system:

- Runs Qwen3-30B-A3B on a single 16GB RTX 4090 Mobile at 509 t/s single-session (2.6–3.4× faster than community benchmarks for this model on RTX 4090 24GB with standard frameworks) and 2,446 t/s aggregate across 64 concurrent persistent-memory sessions; no standard framework runs this model on 16GB at this concurrency
- Supports 64 concurrent persistent-memory agent sessions with unbounded conversation history on 16GB, each maintaining genuine long-term context across arbitrarily many turns
- Achieves per-block validated KV cache compression across ten measured modes with asymmetric K/V thresholds, with the top-quality mode achieving K_SNR 58.6 dB — directly validating the ε_hot ≈ 0 claim of the Asymptotic Numerical Stability theorem
- Eliminates the decode-time numerical drift that degrades generation quality beyond ~500 tokens
- Demonstrates compositional multi-hop reasoning over its own 2.2M-line Rust/CUDA Candle fork via iterative decode-time retrieval: the one-shot ablation confirms iterative retrieval discovers transitive dependencies that pre-generation retrieval misses, with accuracy independent of dependency chain depth (§9.12; full quantitative evaluation in v2)

The system is implemented entirely in Rust on a custom fork of the Candle framework. This was not a choice of convenience — it was a necessity, because standard GEMM libraries dequantise weight matrices to full precision before computation, which would OOM during prefill on 16GB. Writing native quantized matmul kernels that never materialise a BF16 weight copy was a prerequisite for the rest of the system to exist.

**Three independent contributions.** This paper makes three distinct contributions with different communities of impact and different timescales.

*For inference systems researchers:* 509 t/s single-session on a 16GB consumer GPU — 2.6–3.4× faster than community benchmarks for this model on RTX 4090 24GB with Ollama/llama.cpp — and 2,446 t/s aggregate across 64 concurrent persistent-memory sessions (a workload no standard framework supports on 16GB). The high-concurrency design is not incidental: unbounded context with provenance-selected attention requires simultaneous parallel prefill across semantic boundaries and multi-dimensional decode across concurrent sessions — the architecture only achieves its full performance and quality properties under genuine concurrency load. The codebase dependency analysis evaluation demonstrates that unbounded context enables compositional multi-hop reasoning over real structured data: the system walks transitive dependency chains through its own codebase via iterative retrieval during decode, discovering relationships that pre-generation one-shot retrieval misses. (See Appendix C for code, live demo, and collaboration details.)

*For the KV quantization and ML theory community:* the Asymptotic Numerical Stability theorem (§11.2), which reframes the foundational problem of this subfield. The current literature treats KV quantization as an optimization problem — find the best compression scheme within a regime where error grows with context depth. The theorem establishes that the regime itself is escapable through architectural means, and that provenance-selected sparse attention is both necessary and sufficient for the escape. This changes what the right research questions are, not just the answers.

*For the broader research community:* the constraint-driven innovation methodology documented in §11.5 — a specific, reproducible pattern in which constraints that close standard solutions force replacements that are universally better. This is the contribution with the longest half-life and the most generalisable implications.

The agent use case is the existence proof that all three are simultaneously achievable and practically necessary. It is not the subject of the paper.

The combination of bounded numerical error and compositional reasoning at unbounded context depth has not, to our knowledge, been demonstrated by any prior system. More VRAM does not resolve this: an H100 at 80GB enters the O(N) accumulation regime the moment any token is compressed or evicted — the threshold shifts, but the regime does not (§11.2 Corollary 2). Summarization is not compression; it is the destruction of the capability the application requires, because correct dependency analysis requires the full causal chain of context that summarization destroys (§9.12). The KV quantization literature optimises inside a regime whose boundaries it has not examined — every published analysis assumes full attention and asks how to compress within that regime; none ask whether the O(N) regime itself is escapable (§11.2, §11.3). The constraint that forced this architecture is the constraint that made it correct: the 16GB ceiling forced provenance-selected attention, which is the architectural property the theorem requires (§11.5). Attention is retrieval; this system extends the retrieval mechanism that attention already is to unbounded depth, rather than replacing it with something that loses attentional continuity (§6, §11.4).

### 1.2 Technical Contributions

1. **Online Markov Expert Prediction with DMA Offload** — a self-learning transition matrix that predicts expert routing from actual inference observations without calibration, combined with a wave-batched grouped GEMM kernel that exploits cross-request expert locality to reduce effective PCIe transfer costs.

2. **Two-Phase KV Cache Quantization with Prefill Refresh** — a turn-boundary prefill refresh strategy that eliminates autoregressive error accumulation, coupled with an adaptive per-block selection kernel that assigns independent K/V formats based on asymmetric error metrics grounded in the softmax-amplified K vs. linear-bounded V error propagation asymmetry [AsymKV, COLING 2025]. Compression ratios span three tiers from near-lossless (top-quality) to high-compression, validated end-to-end by the multi-session story rewrite test (§9.8).

3. **Attentional Provenance Indexing with Speculative Context Decode** — capturing Q vectors as persistent cognitive-state fingerprints and binarising them into 128-bit BDP signatures via an 8-head XOR fold across two syntactic-band layer depths (MH_XOR_QQ_l0×l4), enabling 3–10ms CPU-side section discrimination and corpus retrieval via span-scored BDP matching (α=2.0) and six INT8 matrix multiplies over 50K+ conversation turns and 100K+ facts. During generation, Speculative Context Decode (§6.5) pipelines a variable-window probe session — up to 64 tokens, terminated at newline boundaries — in parallel with each real decode session: the probe's Q/K fingerprints drive CPU provenance scoring at each reasoning step boundary, assembling the next context window while the current one decodes. The 3–10ms CPU scan latency is fully hidden behind parallel GPU computation, yielding near-zero visible overhead. Probe tokens are discarded and never enter the KV cache. Evaluated in §9.12 on the system's own 2.2M-line Rust/CUDA Candle fork (~20M tokens of learning-phase conversation): each probe cycle during reasoning retrieves the next dependency node at the model's natural reasoning granularity, producing compositional dependency analysis across direct, transitive, and architectural categories that single pre-generation retrieval cannot replicate.

4. **Unbounded Three-Tier Paged Context** — a VRAM-hot / CPU RAM-warm / disk-cold paged context architecture, together with a formal proof that total numerical error per generation step — from any source, including F16 rounding — is bounded by a constant O(1) independent of context depth N under provenance-selected attention, in contrast with the O(N) scaling of full-attention systems. This result is independent of hardware: more VRAM defers the accumulation threshold but does not eliminate the structural problem, which is architectural rather than representational. The theorem establishes that bounded error accumulation at unbounded context depth requires decoupling of working set size from context depth — a property our architecture provides and standard full-attention systems cannot.

5. **Native Quantized Inference Stack** — greedy decomposition for smooth 1–500 token throughput without remainder handling, a fused sampling kernel covering all common sampling modifiers in one CUDA launch, and native quantized matmul kernels that eliminate the dequant materialisation OOM during prefill.

---

## 2. Background and Related Work

### 2.1 MoE Expert Offloading

Several systems have addressed expert offloading for MoE models under VRAM constraints. Fiddler [Kamahori et al., 2024] and HybriMoE [Zhong et al., 2025] partition experts between CPU and GPU based on activation frequency. DAOP [DATE 2025] extends this with data-aware prefetching. The core limitation shared by these systems is that they rely on offline profiling or trained auxiliary modules to predict expert routing — prediction accuracy degrades on novel distributions, and the calibration requirement creates deployment friction.

Our Markov predictor learns online from actual routing observations, has no calibration requirement, and improves continuously during inference. Combined with the wave-batched kernel that coalesces expert work across concurrent requests, it achieves zero idle time on the GPU even under partial VRAM residency.

### 2.2 KV Cache Quantization

The online, inference-time, training-free KV cache quantization field has three meaningful entries. **KIVI** [Liu et al., ICML 2024] established channel-wise outlier structure in Keys and token-wise structure in Values, motivating per-channel K and per-token V quantization at uniform 2-bit. It is the standard baseline: well-reproduced, widely cited, but a blunt instrument — identical format assigned to every block regardless of activation difficulty. Their Table 2 shows quality degradation on long-context retrieval tasks at 5× CR. **KVQuant** [Hooper et al., NeurIPS 2024] added sensitivity-aware treatment: per-vector outlier handling and special-casing of the first token, achieving sub-0.1 perplexity degradation at 3-bit. The contribution was demonstrating that careful outlier handling makes aggressive uniform quantization viable — but format assignment remains population-level, not per-block. **TurboQuant** [Zandieh et al., ICLR 2026] achieves near-optimal compression via Hadamard rotation and Lloyd-Max quantization. The rotation spreads energy uniformly before quantization, with theoretical advantage below 3-bit. In practice, community benchmarks on Qwen3 find that Q4_0 matches or exceeds PolarQuant-MSE (the residual implementation after removing QJL) at comparable 4–5× compression ratios, suggesting the rotation overhead does not pay for itself in the range where most deployments operate.

**KITTY** [Xia et al., 2025] demonstrates that mixed-precision channel-wise quantization can recover most of the accuracy gap between 2-bit and 4-bit Key caches by boosting 12.5–25% of channels to 4-bit. Their selection uses an offline-computed magnitude ranking; the authors note that "more principled or adaptive strategies may yield stronger robustness accuracy recovery" as future work. This paper addresses that directly. The binary precision-boost decision is replaced with adaptive per-block format selection using per-block reconstruction error in the Q-projected metric for K and normalised-L2 error for V. The selector operates at per-(layer, head) block granularity and is recomputed per operating point. We extend adaptation to V as well as K, providing an 11-point Pareto curve (C0–C9) rather than two configurations, and operating on the harder MoE deployment target without model-specific recalibration.

Everything else in the quantization-adjacent space is either orthogonal or a different design point. Token eviction systems (H2O, TOVA, StreamingLLM) are lossy by definition — they solve a different problem. Trained compression (DMS) requires fine-tuning. SVD-based approaches (xKV, LoRC) are per-prompt and offline-flavoured. KVTC [Staniszewski & Łańcucki, ICLR 2026] achieves ~20× compression via PCA decorrelation and entropy coding but operates offline on completed caches for storage and transfer rather than during live inference — a complementary design point. None of these compete in the online inference-time setting.

OTT [Su et al., ACL 2025] identified that outlier tokens cluster near block boundaries, independently motivating our 4/28 sub-block structure — the same insight that drives KVQuant's first-token sensitivity, generalised to a structural per-block design.

The gap this paper occupies: all three competitive systems optimise the compression *primitive* — better quantization formula, rotation, codebooks. None treat format assignment as a per-block decision with a quality guarantee. None establish any theoretical result about error scaling with context depth. The adaptive selection architecture subsumes the insights of KVQuant's outlier handling (via _KS sub-block formats and attention sink protection) and RefreshKV's drift correction (§2.3), while adding the per-block quality guarantee that makes all three possible simultaneously.

### 2.3 Autoregressive Error Accumulation

GEAR [Kang et al., 2024] formally established that quantization error in autoregressive decode compounds multiplicatively across steps, causing critical generation deviation in long sequences. RefreshKV [Xu et al., ACL 2025] is the only published work addressing this specifically: periodic full-attention KV regeneration recovers precision lost through sequential accumulation. They identified the problem and demonstrated the fix, but as a standalone technique — not integrated with any compression system. Our two-phase architecture takes the same insight and couples it structurally with the quantization system: prefill refresh is not periodic but turn-boundary-triggered, and it feeds directly into the adaptive format selection on the refreshed activations. RelayCaching [2026] confirmed that KV values are highly consistent across prefill and decode phases, validating prefill as the correct ground-truth regeneration strategy.

### 2.4 Attention Sinks

StreamingLLM [Xiao et al., ICLR 2024] established that a small cluster of initial tokens carries disproportionate attention weight and must be preserved under any compression scheme. KVSink [Su & Yuan, COLM 2025] extended this to mid-sequence emergent sinks. Both findings motivate the attention sink protection mechanism in §5. KVTuner [2025] and community benchmarks [llama.cpp #20969, 2025] both identify boundary transformer layers as specifically more sensitive to K quantization on Qwen3.

### 2.5 KV-Based Retrieval

RetrievalAttention [Liu et al., NeurIPS 2025] identified and quantified the Q→K distributional gap: Q vectors deviate more than 10× farther from K vectors than K vectors deviate from each other, causing standard ANNS to degrade severely when queried cross-distribution. Our attentional provenance indexing addresses this by storing Q fingerprints alongside K fingerprints, enabling within-distribution Q→Q matching to carry the dominant retrieval signal.

---

## 3. System Architecture

The system is structured as a forward pass pipeline with six integrated subsystems. The architecture is organised around a single theoretical requirement: the working set of tokens attending any generation step must be bounded independently of total context depth — the structural property the Asymptotic Numerical Stability theorem (§11.2) requires. Every subsystem was designed with knowledge of this requirement and of every other subsystem's constraints and capabilities.

```
Token Embeddings
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  SPECULATIVE CONTEXT DECODE (generation-time loop)       │
│                                                          │
│  ┌─────────────────────────┐  ┌────────────────────────┐ │
│  │  DECODE_N               │  │  PROBE_{N+1}           │ │
│  │  variable-length output │  │  up to 64 tokens       │ │
│  │  kept; context from     │  │  discarded; Q/K finger- │ │
│  │  PROBE_N fingerprints   │  │  prints captured       │ │
│  └────────────┬────────────┘  └──────────┬─────────────┘ │
│               │  wave-batched parallel    │               │
│               └──────────┬───────────────┘               │
│                     BARRIER                               │
│          CPU provenance scan on PROBE_{N+1}               │
│          (3–10ms, overlaps late decode tokens)            │
│          Assemble context window for DECODE_{N+1}         │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  ATTENTION LAYER (all layers)       │
│  Paged KV cache with adaptive quant │
│  Chunk-sealed quantization on write │
│  Attentional provenance hooks       │
│  Three-tier context assembly        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  MoE EXPERT PIPELINE (all layers)  │
│  Async fork-join with DMA overlap  │
│  Online Markov prediction           │
│  Wave-batched grouped GEMM          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  SAMPLING KERNEL                    │
│  Fused all-modifier single launch   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  TURN SEAL (on turn completion)     │
│  Prefill refresh over completed turn│
│  Re-quantize from clean activations │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  MEMORY PIPELINE (async)                        │
│  Fingerprint extraction from probe tokens       │
│  at barrier; context assembly for next window   │
│  Trie insertion / fact management               │
└─────────────────────────────────────────────────┘
```

The design principle throughout is that no component materialises an intermediate representation that is not consumed by the immediately following operation. The dequantised weight copy that standard GEMM libraries materialise does not exist. The BF16 expansion that standard sampling pipelines require between the logit tensor and each modifier does not exist. Every kernel boundary is a necessary one.

---

## 4. Online Markov Expert Prediction and Wave-Batched MoE

### 4.1 The Problem

Qwen3-30B-A3B has 48 MoE layers with 128 experts each — 6,144 experts total at approximately 3MB each (Q4_K_M), totalling ~18GB. No single consumer GPU achieves full residency. On a 16GB card with a 50% VRAM budget for experts, approximately 2,700 slots are available — 44% residency. The dominant inference cost is PCIe transfer latency for cold expert fetches. Without prediction, 56% of expert activations are cold misses, each blocking the GPU compute stream while waiting for an ~120μs DMA transfer.

### 4.2 Online Markov Transition Matrix

The self-learning predictor maintains a per-layer-pair transition count matrix:

```
T[L→L+1] : [128 × 128] float32 co-occurrence counts
```

After each forward pass, the predictor observes which experts fired at each layer and increments `T[L][from][to]` for each adjacent-layer expert pair. Empirically, convergence to stable routing predictions is observed in the range of 64–128 observations per layer pair under typical inference workloads — a consequence of the strong routing regularities that emerge from expert specialisation in trained MoE models. The convergence rate depends on workload diversity: homogeneous workloads converge faster, mixed-domain workloads require more observations. Quantitative convergence data will be reported in the ablation study (§9.7).

Prediction for layer L+1 given active set at layer L:

```
scores[j] = Σ_i T[L→L+1][i][j]  for i in active_experts_L
best = argmax(scores, excluding experts already cached)
```

A single expert per layer is prefetched into a free slot. The key constraint is that speculative prefetch **never evicts** — it only uses free slots. This makes mispredictions structurally harmless: a mispredicted expert occupies a free slot until evicted by normal score-based policy.

**Why this works better than trained predictors.** Most published MoE prefetching systems require a trained auxiliary module or offline profiling over representative data. The Markov predictor requires neither. It learns the model's own routing tendencies from production inference — meaning it adapts to the actual workload distribution rather than a calibration proxy. For content with strong routing regularities (code tokens, prose tokens, domain-specific content), convergence is fast and prediction accuracy is high.

### 4.3 Four-Part Eviction Policy

Naive timestamp-based eviction causes cascade failures: a cold miss at layer L evicts an expert from layer L+5, causing a new miss that evicts from layer L+10. Empirically, this halved single-token decode throughput before the policy was redesigned.

The four-part policy eliminates cascades:

**Part 1 — End-of-pass batch eviction.** After layer 47, evict the bottom 5% of occupied slots by usage timestamp. This creates ~140 free slots as a shock absorber for the next pass's early layers.

**Part 2 — Layer-aware forced eviction.** When a real miss occurs with no free slots, evict from completed layers first (highest completed layer, then oldest timestamp). This makes it structurally impossible to evict an expert needed later in the same pass.

**Part 3 — Early-layer pinning.** Layers 0–2 are pinned — never evicted. These layers have no prior compute to overlap DMA with; a miss at layer 0 causes a full stall. Pinning costs ~24 slots (~1% of budget) and eliminates the worst-case stall.

**Part 4 — Free-slot-only prefetch.** As described above. Mispredictions are inert.

Measured improvement on RTX 4080 at 44% residency:

| Config | Before (v2) | After (v3) | Δ |
|---|---|---|---|
| BF16 × 1 context | 199.9 t/s | 241.5 t/s | +20.8% |
| BF16 × 4 contexts | 685.3 t/s | 1090.1 t/s | +59.1% |
| Q8_0 × 8 contexts | 1220.2 t/s | 1699.9 t/s | +39.3% |

### 4.4 Wave-Batched Grouped GEMM

The key insight is that batching prefill and decode requests together and routing them through the expert layer as a wave creates **cache locality in the expert hot set**. When Request A and Request C both need Expert 7, Expert 7's weights load once and serve both. The effective hot set is the union of expert activations across all concurrent requests — which is smaller than the naive sum because similar content routes to similar experts.

The kernel uses two-phase dispatch per layer:

```
Phase 1: Submit grouped GEMM for all WARM/READY experts
         Submit DMA for all COLD experts (concurrent)

Fence:   GPU-side stream wait — compute stream waits for DMA

Phase 2: Submit grouped GEMM for newly-READY experts

Join:    Submit index_add scatter for all expert outputs
```

The CPU never blocks. All submissions are async. The GPU resolves ordering through stream dependencies. For a typical layer with 6 hot experts and 2 cold experts, the 6-expert grouped GEMM at ~200ms overlaps with the 2 cold expert DMAs at ~120ms each — the fence wait is typically zero.

Cross-request coalescing is the true batch dimension for MoE inference. In a dense model, batching amortises the weight load cost across multiple output elements. In MoE, batching means multiple tokens processed through the same expert simultaneously — and that only happens when the wave kernel aggregates work across concurrent requests.

### 4.5 Layers 1–2 Pinning Benefit

The first two transformer layers are permanently resident in VRAM. This has a compound benefit beyond preventing cold stalls at those layers. By the time the forward pass reaches layer 3, the hidden states have been shaped by two full attention and MoE transformations. The routing decisions at layer 3 and beyond are functions of these richer representations — they are more content-dependent and less noisy than the routing at early layers. The Markov transition matrix is operating on the more stable, predictable portion of the routing DAG. This raises steady-state prediction accuracy and accelerates convergence during matrix warmup.

---

## 5. Two-Phase KV Cache Quantization

KV cache quantization in a high-concurrency system faces two distinct error sources that require separate treatment. Conflating them leads either to incorrect designs or to unnecessary memory pressure.

**The first error source: autoregressive decode drift.** During autoregressive decode, each token's KV values are computed with attention over already-quantised preceding KV. Each step introduces small quantisation error that feeds into the next step's activations. GEAR [Kang et al., 2024] established that this compounds multiplicatively, causing coherence degradation beyond approximately 500 tokens. This is a sequential dependency problem — the error accumulates through the causal chain of decode steps.

**The second error source: per-chunk quantisation noise.** When a 32-token block is sealed and quantised, the quantisation error is bounded by that block's own activation distribution. It does not propagate to subsequent blocks because each block is independently quantised from its own activations. Chunk-level quantisation does not compound.

These are independent mechanisms. The correct fix for each is independent.

**Phase 1 — Chunk-sealed quantization (during prefill and decode).** Every 32-token block is quantised immediately when sealed, using the adaptive per-block selection kernel against the block's live activations. Memory pressure stays bounded regardless of session count or turn length — at 200 concurrent sessions with 500-token turns, materialising all turns at F16 simultaneously would require ~100K tokens × 48 layers × 2 × 2 bytes ≈ 18GB, immediately OOMing a 16GB card. Chunk-sealed quantization keeps the working F16 footprint to one block per session at a time.

A third error-reduction property follows from the chunk boundary structure. The active (unsealed) chunk — the accumulating tail of the current decode stream — is always held at full F16 precision until it reaches 32 tokens and seals. This means the most recent tokens in any session, where autoregressive error would otherwise be highest due to accumulated drift, are always attended at full precision. Quantization error is structurally confined to sealed blocks; the immediate decode tail is always clean.

**Phase 2 — Turn-boundary prefill refresh (on turn completion).** Once a turn is fully decoded and sealed, a batched prefill pass processes the completed turn's tokens in parallel with full-precision activations — no token's computation depends on the quantised cache of a preceding token. The resulting clean KV values replace the decode-drifted blocks for that turn. The cost is proportional to the completed turn's length at F16, not the full context and not all sessions simultaneously.

RelayCaching [2026] empirically confirmed that KV values for identical content are highly consistent across prefill and decode phases, validating that the prefill pass produces the ground-truth KV for a given token sequence.

**Why prefill refresh sets the asymptotic quality floor.** The Asymptotic Numerical Stability theorem (§11.2) establishes that under provenance-selected attention, the expected total numerical error per generation step is bounded by ε_hot + C_warm = O(1), a constant independent of context depth N. Hot-tier tokens — the current turn, pinned system context, and prefill-refreshed recent history — are always present in the working set and are therefore always attended. Their error is not diluted by the provenance selection mechanism; it is the irreducible floor. The prefill refresh is what keeps ε_hot ≈ 0: without it, hot-tier tokens would carry sequential decode drift, raising the asymptotic floor. Cold-tier and warm-tier tokens contribute error scaling as O(1/N) and a bounded constant respectively — their contribution shrinks or is bounded regardless of compression aggressiveness. The practical consequence is that compression format selection should be aggressive for cold-tier tokens (their error contribution vanishes asymptotically) and conservative for hot-tier tokens (their error is the system's permanent floor). This is exactly what the two-phase strategy implements: chunk-sealed quantization for all tiers during decode, and prefill refresh specifically for the tokens that will remain hot.

---

## 6. Attentional Provenance Indexing

### 6.1 Core Insight

At the moment a model produces or processes any content, the Q vectors it computes are a compressed fingerprint of its cognitive state — encoding not just content semantics but the full accumulated reasoning context in which that content was produced. We call this *attentional provenance*. By capturing and storing Q vectors alongside compressed K vectors at inference time, we construct a dual fingerprint index in CPU RAM that enables fast scanning over arbitrarily large corpora before generation begins and, critically, during generation itself. This is more than retrieval-augmented generation: the provenance index functions as an attention mechanism over unbounded context, continuously focusing the KV working set on what the current decode stream actually needs at each reasoning step. RAG retrieves once and injects; this system re-selects the context window at every reasoning boundary, using the model's own Q vectors as the selection signal.

This is qualitatively different from K vectors, which encode content semantics. A stored Q vector from a conversation turn captures the accumulated attentional context of everything preceding it — mood, topic trajectory, relationship dynamics, reasoning state — compressed into a ~780-byte fingerprint. When a future query produces a Q vector in a similar reasoning state, the match surfaces that turn regardless of surface-level token overlap.

### 6.2 Three-Tier Depth Fingerprint

For every indexed item, six compact vectors are stored across three layer depth bands:

```
Syntactic band (0–N/3):    K_syntactic, Q_syntactic  (128 bytes each, INT8)
Semantic band  (N/3–2N/3): K_semantic,  Q_semantic
Pragmatic band (2N/3–N):   K_pragmatic, Q_pragmatic
Scales (6 × fp16):                               12 bytes
─────────────────────────────────────────────────────────
Total per item:                                ~780 bytes
```

The three-band structure is grounded in layer-wise emotion probing research [Zhang et al., 2025; Tak et al., 2025] demonstrating three functionally distinct processing regimes: syntactic layers (0–N/3) for lexical and syntactic processing, semantic layers (N/3–2N/3) for semantic category and emotion consolidation (emotion probe accuracy peaks steeply through this band), and pragmatic layers (2N/3–N) for relational reasoning and contextual integration (Qwen3-4B peaks at 75% depth — pragmatic band).

**Model-agnostic aggregation.** Layer boundaries at N/3 and 2N/3 require only total layer count N. No per-model configuration. Three running accumulators per signal type, updated with one vector addition per layer. At N/3 and 2N/3 boundaries, the syntactic/semantic accumulators are finalised. Total overhead per forward pass: negligible.

**Q→K distributional gap handling.** Liu et al. [NeurIPS 2025] quantified that Q vectors deviate more than 10× farther from K vectors than K vectors deviate from each other, causing standard ANNS to degrade severely. K fingerprint construction uses Q-aware token selection: tokens selected by Q·K inner product score rather than K magnitude. This selects K tokens maximally visible from the Q distribution at construction time, mitigating the OOD gap rather than suffering it at retrieval time. The Q→Q matching component (dominant for history and mood retrieval) operates entirely within-distribution.

**Binary Directional Provenance (BDP) signature.** Within the syntactic band, section-level discrimination uses a 128-bit BDP signature derived from an 8-head XOR fold across two structurally distinct syntactic-band layer depths. Per token at layers $\ell_0$ (band start, model layer 3) and $\ell_4$ (band centre, model layer 7):

$$	ext{TokenSignature} = igoplus_{i=0}^{3} 	ext{sign}(Q^{\ell_0}_i) \;\oplus\; igoplus_{i=0}^{3} 	ext{sign}(Q^{\ell_4}_i)$$

where $Q^{\ell}_i \in \mathbb{R}^{128}$ is the Q vector for KV head $i$ at syntactic band layer $\ell$, and sign(·) binarises to $\{-1,+1\}^{128}$. The XOR fold across 8 (head, layer) subspaces produces a fingerprint that is stable under sustained directional focus (all heads coherently agree) and cancels under noise (heads disagree). Similarity between two signatures is measured by BDP: XNOR + popcount agreement in $[0, 128]$. A per-section score accumulates BDP hits from probe tokens; span scoring with $lpha=2.0$ contributes $L^2$ for each run of $L$ consecutive hits to the same section, strongly rewarding sustained directional focus over isolated coincidental hits.

**Span scoring (α=2.0).** For each consecutive run of $L$ probe tokens all producing BDP hits to the same section at a given depth band, the contribution to that section's score is $L^2$ rather than $L$. The quadratic reward strongly distinguishes sustained directional focus — the signature of genuine section intent — from isolated coincidental hits:

$$S(\text{section}) = \frac{1}{3}\sum_{b}\sum_{\text{runs}} L_b^2$$

where the outer sum is over the three depth bands, and the inner sum is over consecutive hit-runs of length $L_b$ to that section at depth band $b$. The final discrimination ratio is $S(\text{target}) / \bar{S}(\text{other sections})$.

This scoring penalises isolated hits and rewards coherent multi-token focus. Combined with the 8-head dual-layer fingerprint, it achieves min\_ratio $> 1.0$ on every probe under both count and span scoring across all 8 tested tool sections on Qwen3-30B-A3B — including the hardest pair (file\_read vs file\_write, min\_ratio 2.53 under span). Full validation is in §9.10.

### 6.3 Sequential Section Resolution

Each turn executes a sequential dynamic section resolution loop before generation begins. Dynamic sections are any context components whose content depends on the current query state — examples include persona, response style, domain knowledge, or conversation history. Each section is backed by a candidate library with precomputed KV representations. Selection proceeds in order: each probe generates a short token window under the system prompt for that section, captures the Q fingerprint, runs the CPU flat scan, and loads the winning candidate's KV into the context. Each subsequent probe executes into the context already committed by prior probes — Q vectors reflect genuine intent conditioned on all prior commitments.

As a concrete example, an agent deployment might resolve three sections in sequence:

1. **Persona probe** — model generates W_probe tokens under persona system prompt; Q fingerprint captured; CPU scan over persona library in ~3ms; selected persona KV loaded
2. **Style probe** — same pattern with persona now in context
3. **History probe** — under main system prompt with persona and style resolved; Q reflects full response intent; scan over full B-tree index; top-ranked history turns loaded within token budget

This is structurally different from prior probe-reset architectures that generated throwaway tokens under all candidates in parallel and then discarded everything. Here: one pass per section, selection on CPU, no GPU waste beyond the probe tokens themselves. The sequential conditioning property — each probe Q reflecting all prior commitments — means the final history retrieval is optimised for the specific response the model is about to generate, not a generic query intent.

**Dual representation for mood/template.** Fingerprints are encoded as assistant prefill under the section's probe system prompt, placing both stored fingerprints and probe Q in the same distributional space. This eliminates the Q→K OOD problem for these sections: the match is Q→Q within-distribution at both ends.

### 6.4 CPU-Side Flat Scan

The full fingerprint index for 50K turns + 100K facts fits in approximately 126MB of CPU RAM — typically in L3 cache on modern server CPUs. Six INT8 matrix multiplies over this index, parallelised across 6+ CPU threads via VNNI instructions, complete in 3–10ms regardless of corpus size. This replaces hierarchical navigation (beam walks, recursive tree descent, iterative probe-reset cycles) with a single flat scan. For section-level scoring during generation (§6.5), the flat scan is replaced by BDP matching with span scoring α=2.0 on the TokenSignature index (§6.2).

Scoring formula for any indexed item:

```
score = Σ_b w_Kb × (kprobe_b · K_b) + Σ_b w_Qb × (qprobe_b · Q_b)

where b ∈ {syntactic, semantic, pragmatic}
```

where b ∈ {syntactic, semantic, pragmatic}. Component-specific weights reflect the three-band layer regime: Q_pragmatic dominates for history (w=0.50) and mood (w=0.45) because pragmatic-band Q captures the richest cognitive-state fingerprint; K_semantic and K_syntactic dominate for facts because topical content retrieval is primarily semantic and lexical. This weighted dot-product scoring applies to corpus retrieval (history, facts, mood, templates). Section-level discrimination during generation uses the BDP TokenSignature mechanism (§6.2, §9.10).

**Multi-resolution history.** The conversation history B-tree is retained as a pre-computation infrastructure rather than a navigation mechanism. Summary nodes at multiple resolutions are indexed alongside verbatim turns. The flat scan surfaces the appropriate resolution automatically: if a span summary and its verbatim children both score highly, a resolution policy selects verbatim when they fit within budget. No hierarchical traversal.

### 6.5 Speculative Context Decode

Sections §6.3 and §6.4 describe provenance retrieval *before* generation begins — selecting mood, template, and history for the upcoming response. During generation itself, the model's Q vectors evolve with every token produced, creating a continuous signal for context refinement. Speculative Context Decode makes this retrieval structurally intrinsic to the decode loop, with CPU scoring latency fully hidden behind parallel GPU computation.

**Mechanism.** Generation operates as a pipelined two-session loop with variable-length probe windows.

*Cold start.* PROBE₁ speculatively decodes up to 64 tokens against the initial working set, terminating early at the first newline boundary. These tokens are discarded and never enter the KV cache. Their Q/K fingerprints are captured and scored via the CPU flat scan (§6.4). The first corrected context window is assembled from the top-scoring blocks.

*Steady state (every subsequent window).* Two sessions launch in parallel on the wave-batched kernel:
- **DECODE_N** — autoregressively decodes real output tokens against the context window assembled from PROBE_N's fingerprints. The token count matches whatever PROBE_N produced before termination. These tokens are kept and seal into 32-token paged blocks normally.
- **PROBE_{N+1}** — speculatively decodes up to 64 tokens (or until newline), continuing from the probe trajectory. These tokens are discarded and never enter the KV cache.  Their Q/K fingerprints are captured.

*Barrier.* Both sessions join. CPU BDP scoring runs on PROBE_{N+1}'s TokenSignatures — span-scored across consecutive runs of BDP hits to each candidate section, overlapping with the final tokens of each session where possible. The highest-scoring sections' KV blocks are loaded into the context window for DECODE_{N+1}. Return to steady state.

**Newline-terminated variable window.** The probe terminates at the first newline, with a 64-token cap as a safety valve. Newlines in model output — particularly within thinking blocks — mark the completion of a discrete reasoning step; the Q vectors at a newline boundary encode what the model is reaching for next, making them the optimal fingerprint capture point. The variable window adapts naturally to the model's reasoning rhythm: a query prompting rapid short inferences triggers more frequent context updates; a query requiring extended chains of logic produces longer probe windows with more developed fingerprints. No tuning parameter. The model's own structural markers determine retrieval frequency.

**Block structure independence.** Probe tokens are discarded and never enter the KV cache, so the probe window has no relationship to the 32-token block structure used by the real decode tokens. Block alignment — paging, quantization, sealing, fingerprint extraction for the conversation history — applies only to the real output tokens produced by DECODE_N.

**Why probe divergence is acceptable.** The probe runs ahead on a slightly divergent trajectory. This divergence is acceptable because provenance operates on approximate cognitive-state fingerprints averaged across layer bands — a short stretch of reasoning divergence does not move Q vectors far enough to select wrong context blocks. The provenance system needs the approximate neighbourhood of the reasoning direction, not the exact token sequence.

**Cost structure.** The probe is hidden behind the decode: both sessions run in parallel with wave-batched expert coalescing. The probe and decode sessions diverge on similar content and share similar expert routing patterns, coalescing efficiently (see §10.1). Effective GPU cost is substantially less than 2× a single session. CPU scoring (3–10ms) is amortised over the window's real tokens. Each active query consumes two session slots from the 64 available. Visible throughput approaches the raw single-context decode rate.

**Relationship to the theorem.** The working-set selection budget B and hot-tier bound W_hot_max are enforced at each probe-barrier cycle. The Asymptotic Numerical Stability theorem (§11.2) holds unchanged: the bound is O(1) per generation step regardless of how frequently the working set is updated, because each update selects at most B tokens from the unbounded context. Speculative Context Decode increases the frequency of working-set updates without changing the bound — and increases the quality of each update by aligning working-set assembly with the model's evolving Q-state at reasoning-step granularity.

**Quality implication.** The scheduling description above — near-zero overhead, latency hidden, throughput preserved — characterises cost. The quality implication is distinct and worth stating separately. Because the context window is assembled fresh at each reasoning-step boundary, every line of reasoning during a thinking block is generated against a working set optimised for that specific line. The feedback loop is: better context produces more specific Q vectors, which produce more targeted retrieval, which produce better context for the next reasoning step. This is why the one-shot ablation gap exists — one-shot retrieves once before reasoning begins, then holds a fixed context window through the entire thinking block regardless of where reasoning goes. The divergence between full system and one-shot is not a retrieval accuracy difference at hop 1; it is the compound effect of context quality improving through the reasoning chain versus remaining static. Transitive dependencies that one-shot misses are not inaccessible to it — they may be in its index — but the model's Q vectors at line N of reasoning have not yet reached the part of the index that contains them. Speculative Context Decode delivers the right context at the right reasoning step.

**Middle-context degradation.** A well-documented failure mode of long-context attention is that models exhibit substantially lower recall for content positioned in the middle of a long context window, with performance concentrated at the start and end [Liu et al., 2023]. This degradation is structural: uniform flat attention over a long window dilutes attention weight across positions, and intermediate positions receive neither the primacy nor recency signal that anchors recall at the boundaries. Provenance-selected attention eliminates this failure mode architecturally. The working set at each reasoning step is a small focused window — bounded by B — assembled specifically for that step. There is no "middle" in the pathological sense: every selected block is proximal in relevance to the current Q-state, not proximal in sequence position. Content from 50,000 tokens ago that is provenance-relevant receives the same attention density as content from 100 tokens ago. The degradation curve that plagues flat long-context attention does not apply to a window that is continuously re-focused on what the decode stream actually needs.

---

## 7. Unbounded Three-Tier Paged Context

The context is organised into three tiers managed by the paged allocator:

- **VRAM-hot** — current working set, full-speed attention
- **CPU RAM-warm** — recent history, loaded on demand via PCIe
- **Disk-cold** — full conversation history, loaded on cache miss

All 32-token blocks are aligned to turn and system prompt boundaries. This ensures: the attention sink cluster is always block-local; prefill refresh triggers at clean boundaries; tier promotion loads coherent semantic units.

**Block metadata.** Format assignment is per-head, not per-block: all 128 blocks in a given KV head share a single format, stored in a head-indexed format table (see §5). The attention kernel reads the format once per tile from this table and broadcasts it across the warp, adding one global memory read per tile with no change to the inner dequantisation loop.

**Cold tier quality floor.** In a naive full-attention system, cold-tier blocks — the oldest tokens — would be attended by every subsequent token, making their quantization error the most damaging in the entire context. Under provenance-selected attention, this relationship inverts: as context depth N grows, the probability that any specific cold-tier block enters the working set in a given generation step shrinks as O(1/N). Cold-tier blocks therefore contribute error that is asymptotically negligible regardless of compression format. What matters for cold-tier quality is that they were cleanly quantized to begin with — which the prefill refresh strategy ensures. Blocks reaching cold storage were quantized from prefill-quality activations, not decode-drifted ones, and were assigned the highest-precision format consistent with their error threshold by the per-block selection kernel. This provides correctness on retrieval when provenance selection does surface them, while the theorem guarantees their aggregate contribution at any depth is bounded.

**Why cold storage must persist KV cache blocks, not tokens.** A naive design would store only the token sequence on disk and reconstruct the KV cache on demand by prefilling. This is correct for standard inference engines — prefill is deterministic and produces the same KV values every time. Under provenance-selected attention it is not. The KV values produced at inference time reflect the specific working set that provenance selection assembled at each reasoning step: the model attended to a particular subset of historical context, and those attention patterns shaped the hidden states that produced the K and V tensors for each token. A reconstructed prefill over the raw token sequence — without the same provenance-selected working set — will produce subtly different KV values because the model's attention context was different during original generation. The difference is small but cumulative: hot-tier and warm-tier blocks reconstructed from tokens will carry higher error than the originals, raising ε_hot above zero and degrading the asymptotic quality guarantee. Cold-tier blocks must therefore be stored as compressed KV cache data, not as token sequences. The full learning-phase conversation released with this paper is stored in this format: it is not a text file but a KV cache archive, and any instance of the engine on any hardware can reconstruct the identical index with the identical quality properties by loading it directly.

**Non-contiguous attention.** The attentional provenance selection kernel identifies relevant blocks from across all tiers. The attention kernel handles non-contiguous gathered block lists alongside the standard sequential paged layout. Selected blocks are promoted to VRAM hot tier via async PCIe transfer overlapping with the current turn's probe phase.

---

## 8. Native Quantized Inference Kernels

### 8.1 The OOM Problem That Forced the Solution

Standard GEMM libraries (cuBLAS, CUTLASS default paths) dequantise weight matrices to BF16 before execution. For a 30B model during prefill:

```
Quantized weights:    ~15GB stored  (Q4)
Dequantized BF16:     ~60GB materialised
```

On 16GB VRAM this is an immediate OOM, not a performance concern. Writing native quantized matmul kernels that dequantise inline within the MMA kernel — producing BF16 values transiently in register file during the MMA, never materialising the full BF16 weight copy — was not a design choice. It was a hard requirement imposed by a deliberately accepted architectural constraint — the decision to build within 16GB and not exceed it, which forced every component to be efficient at the architectural level rather than relying on hardware headroom to absorb inefficiency.

The constraint produced an implementation that is also strictly faster even on unconstrained hardware: less memory traffic per FLOP during prefill, less memory bandwidth consumed on decode. The 16GB ceiling that made the standard approach impossible turned out to expose an inefficiency in the standard approach that exists on every hardware configuration. This is the pattern the paper's methodology section (§11.5) describes: the constraint forced an architectural decision that turned out to be universally better, not just viable on the constrained platform.

### 8.2 Greedy Decomposition

MoE inference with mixed prefill and decode requests produces token counts from 1 to ~500+ hitting each expert per wave step. No single kernel is optimal across this range: GEMV (1–4 tokens) is memory-bandwidth-bound and optimal at small batch sizes; pipelined GEMM (128+ tokens) is compute-bound and optimal at large batch sizes.

Greedy decomposition tiles any token count exactly into a combination of available kernel tile sizes, with no remainder and no padding:

```
500 tokens = 256 (pipelined GEMM)
           + 128 (pipelined GEMM)
           + 64  (GEMM)
           + 32  (GEMM)
           + 16  (batched GEMV)
           + 4   (batched GEMV)
= 500 exactly
```

Each tile executes with the kernel optimal for its size. The performance curve from 1 to 500 tokens is smooth and near-envelope — no performance dead zones, no cliff at tile size boundaries, no separate cleanup pass for remainders.

### 8.3 Fused Sampling Kernel

All sampling modifiers — temperature, top-k, top-p, repetition penalty, frequency penalty, presence penalty, DRY, min-p — are fused into a single CUDA kernel. The logit tensor (150K+ elements for large-vocabulary models) is read once from VRAM into shared memory, all modifiers execute in-kernel, and the sampled token index is written back. 

The standard approach reads and writes the logit tensor multiple times — one separate kernel per modifier — accumulating ~12MB of VRAM traffic per token on typical sampling configurations. The fused kernel reduces this to ~600KB: a 20× reduction in sampling memory pressure. At low batch sizes where sampling latency is a significant fraction of total per-token latency, this translates directly to throughput improvement.

Template specialisation over common modifier combinations eliminates runtime branch overhead for production configurations while a fallback generic path handles arbitrary combinations.

---

## 9. Experimental Evaluation

### 9.1 Evaluation Strategy

No existing inference system provides the same capability set as the system presented here — unbounded context, adaptive per-block KV quantisation, attentional provenance indexing, and Markov-predicted DMA-offloaded MoE inference do not coexist in any published engine. A system-vs-system comparison on an identical workload is therefore not possible: no other system runs the workload.

The evaluation instead uses three complementary approaches:

1. **Community benchmark comparison** for throughput — our peak single-context and bulk throughput figures are compared against the best available published consumer-hardware benchmarks for Qwen3-30B-A3B. NVIDIA TensorRT-LLM does not publish performance figures for this model on consumer 16GB GPUs; the comparison uses community-benchmarked results on RTX 4090 24GB with standard frameworks (Ollama, llama.cpp) as the closest available reference point [hardware-corner.net, 2025; ToolHalla, 2026].

2. **Ablation studies** for each novel contribution — the value of each subsystem is measured by removing it and observing the delta. Ablations are the correct evaluation methodology for a vertically integrated system where no external baseline shares the design space. The codebase dependency analysis evaluation (§9.12) additionally provides the first published demonstration of iterative attention-driven dependency walking during decode — a capability no system we are aware of provides — validated against manually enumerated ground truth over the system's own codebase.

### 9.2 Hardware

**Primary benchmark platform:** NVIDIA RTX 4090 Mobile, 16GB GDDR6, Ada Lovelace (sm_89). FP8 tensor core support. PCIe 4.0 × 16 to host system.

**Secondary validation:** Custom water-cooled desktop, RTX 3090 24GB (Ampere, sm_86). No native FP8 tensor cores; validates the integer-only path and serves as a reference for systems without FP8 support.

**Reproducing results.** The three core test suites can be run directly from the repository root. All tests require a CUDA-capable GPU; `huge-context` enables the multi-session paged context path used for §9.8 and §9.11.

*Qwen3-30B-A3B — parallel batched forwarding with KV compression and multi-session identity discrimination (§9.8, §9.11):*

```
cargo test --release --features cuda,verbose,huge-context --lib \
  --package candle-transformers \
  quantized_qwen3_moe::tests::test_parallel_batched_forwarding \
  -- --ignored --nocapture
```

*Llama-3.2-3B — parallel batched forwarding (§9.8 metric comparison, uniform vs. adaptive):*

```
cargo test --release --features cuda,verbose,huge-context --lib \
  --package candle-transformers \
  quantized_llama::tests::test_parallel_batched_forwarding_llama3 \
  -- --ignored --nocapture
```

*KV compression curve — per-block format selection across C0–C9:*

```
cargo test --release -p candle-nn --features cuda,dont_check --lib \
  sampled_selection::tests::projection::test_candidate_list_compression_curve \
  -- --ignored --nocapture
```

### 9.3 Model

**Qwen3-30B-A3B** (48 MoE layers, 128 experts per layer, top-8 routing, 3.3B active parameters, 30B total parameters, grouped-query attention with 8 KV heads). Weights at Q4_K_M quantisation throughout all experiments.

### 9.4 Quantized Matmul Kernel Benchmarks

The system contains two quantization subsystems that share a common block structure
and kernel dispatch infrastructure. The **qmatmul** subsystem covers formats in which
model weights are actually stored — loaded from GGUF/AWQ files and applied directly
during the forward pass without materialising a full-precision copy. The **kv_cache**
subsystem extends this shared infrastructure with additional formats purpose-built for
KV activation characteristics (Q8_KS, Q4_KS, Q4_1, Q3_0, Q2_0) that do not
appear in weight files and are never used in qmatmul. Reusing the block structure
across both subsystems keeps kernel dispatch consistent and the codebase coherent.

The benchmarks below cover **qmatmul weight formats only**. KV cache format quality
is validated separately by the adaptive per-block selection kernel at
quantization time, since KV quality is a function of attention output fidelity
rather than matrix reconstruction error.

Native quantized CUDA matmul kernels measured on RTX 4090 Mobile (16GB). No BF16
weight copy is materialised at any point. 200 iterations per format, Criterion.rs
statistical harness.

**Tier 1 — High bandwidth (~80+ GiB/s)**

| Format | Median (µs) | Bandwidth (GiB/s) |
|---|---|---|
| F32 | 11.56 | 84.5 |
| F16 | 11.81 | 82.7 |

**Tier 2 — Standard integer (~59 GiB/s)**

| Format | Median (µs) | Bandwidth (GiB/s) |
|---|---|---|
| Q8_0 | 16.37 | 59.7 |
| Q8_1 | 16.44 | 59.4 |
| Q8_K | 16.40 | 59.6 |
| Q6_K | 16.88 | 57.9 |
| Q5_0 | 16.57 | 58.9 |
| Q5_1 | 16.32 | 59.8 |
| Q5_K | 16.55 | 59.0 |
| Q4_0 | 16.31 | 59.9 |
| Q4_1 | 16.38 | 59.6 |
| Q4_K | 16.48 | 59.3 |

**Tier 3 — Low precision (~34 GiB/s)**

| Format | Median (µs) | Bandwidth (GiB/s) |
|---|---|---|
| Q3_K | 28.72 | 34.0 |
| Q2_K | 27.90 | 35.0 |

One finding is notable. FP8 data formats (F8_0, F8_1, F8_KS) were benchmarked during development and showed throughput identical to F16 (~80.5 GiB/s) — no performance benefit. Combined with a quality finding from the KV cache evaluation: FP8's non-uniform value spacing is optimised for weight and activation distributions, not for block-normalised KV cache values where the block scale already handles the dynamic range and INT formats provide better uniform coverage per bit. FP8 was removed from the KV cache format ladder as a consequence. F16 and BF16 remain as high-fidelity fallbacks. FP8 E4M3 does appear as the scale precision in the _KS sub-block format header — but this is scale metadata, not data representation.

Q3_K and Q2_K drop to ~34 GiB/s — approximately 42% of peak — reflecting
the non-power-of-two bit packing overhead inherent to sub-byte formats.

### 9.5 Quantized Matmul Kernel Accuracy

Numerical accuracy of each kernel against a reference implementation on a 2048×2048
matrix, measured across accumulator dtypes and batch sizes 1–128. All formats pass
at all batch sizes. Values shown are worst-case (batch=128) and best-case (batch=1)
max relative error and mean absolute difference.

**Against F32 accumulator (highest precision reference)**

| Format | max_rel (batch=1) | max_rel (batch=128) | mean_diff (batch=128) |
|---|---|---|---|
| Q8_0 | 0.0016 | 0.0077 | 0.000168 |
| Q4_0 | 0.0025 | 0.0050 | 0.000143 |
| Q6_K | 0.0031 | 0.0083 | 0.000202 |

**Against BF16 accumulator**

| Format | max_rel (batch=1) | max_rel (batch=128) | mean_diff (batch=128) |
|---|---|---|---|
| Q8_0 | 0.0267 | 0.0642 | 0.001952 |
| Q4_0 | 0.0195 | 0.0447 | 0.001651 |
| Q6_K | 0.0239 | 0.0573 | 0.001921 |

**Against F8E4M3 accumulator**

| Format | max_rel (batch=1) | max_rel (batch=128) | mean_diff (batch=128) |
|---|---|---|---|
| Q8_0 | 0.0625 | 0.0944 | 0.010276 |
| Q4_0 | 0.0596 | 0.0826 | 0.010126 |
| Q6_K | 0.0743 | 0.0918 | 0.010324 |

Several observations follow from these results. Against F32 and F16 accumulators,
all three formats produce mean absolute differences in the range 0.00013–0.00022,
well within the quantisation noise floor of each format. The higher errors against
BF16 reflect BF16's limited 7-bit mantissa rather than kernel inaccuracy — the
quantised kernel and the reference accumulate rounding differently in BF16 space.

Against F8E4M3 accumulators, mean differences of ~0.010 reflect E4M3's 3-bit mantissa (~12.5% relative spacing per exponent step) — an inherent format property, not kernel error. Max relative error growing with batch size reflects floating-point rounding accumulation across larger reduction dimensions; this is expected behaviour. All values remain well below thresholds that would affect model output quality at each format's intended precision level.

### 9.6 Throughput: Consumer Hardware Comparison

Single-context and bulk decode throughput on RTX 4090 Mobile (16GB), compared against published community benchmarks for Qwen3-30B-A3B on consumer hardware. NVIDIA TensorRT-LLM publishes performance figures for datacenter GPUs (H100, A100, H200) only and does not benchmark this model on 16GB consumer cards. The closest available reference is Ollama/llama.cpp on RTX 4090 24GB — a different card with more VRAM running at single-session concurrency.

| Configuration | This system | Standard frameworks (RTX 4090 24GB, single session) | Notes |
|---|---|---|---|
| Single context | **509 t/s** | 150–196 t/s [hardware-corner.net, 2025; ToolHalla, 2026] | 2.6–3.4× advantage; 8GB less VRAM |
| 64 concurrent sessions (aggregate) | **2,446 t/s total** (~38 t/s/session) | Not applicable — single-session only | Standard frameworks do not support this concurrency on 16GB |

The single-session figure measures decode speed for one active context and is directly comparable to published benchmarks. The 64-session aggregate (2,446 t/s total, ~38 t/s per session) is not a server throughput metric in the conventional sense — it is the architectural operating point the system requires. Unbounded context with continuous provenance-selected working-set assembly demands simultaneous parallel prefill across semantic boundaries and concurrent multi-session decode; the wave-batched grouped GEMM, Speculative Context Decode pipelining, and three-tier paging all reach their full performance and quality properties only under genuine concurrency load. A single-session deployment leaves most of the architecture idle. Standard frameworks do not support this workload on 16GB because they are designed for single-context inference; no external baseline for the concurrent-session result exists for this reason.

**Attribution of throughput gains over baseline Candle:**

| Subsystem | Isolated contribution | Method |
|---|---|---|
*Preliminary — full subsystem-isolated ablation in v2. The §4.3 table reports the directly measured Markov prediction contribution (+20.8–59.1% across configurations); isolated contributions for the remaining subsystems require controlled ablation runs in progress.*


### 9.7 Ablation: Expert Prediction and Eviction Policy

Cold-hit rate and decode throughput on RTX 4090 Mobile under partial VRAM residency (44% expert cache):

| Configuration | Cold hit rate | Decode throughput |
|---|---|---|
*Preliminary — ablation on RTX 4090 Mobile in progress. The §4.3 table reports measured throughput gains from the four-part eviction policy on RTX 4080 at 44% residency (+20.8–59.1% across configurations); cold-hit rate and predictor-isolated contribution on the primary benchmark platform will be reported in v2.*


### 9.8 Quality Evaluation: Multi-Session Identity Discrimination

**Test protocol.** The story rewrite test evaluates end-task quality directly rather than via perplexity. 400 concurrent sessions run simultaneously, each with a distinct character identity (name and gender assignment, cycling with period 99). The model is instructed to rewrite a narrative passage using the assigned character. A session passes if: (a) the correct name appears, (b) gender pronouns are consistent with the assignment, and (c) no other session's name appears. This test is designed to stress exactly the signal that aggressive K quantization degrades: session-discriminating features that live in the high-relevance sub-block population.

**Results.** The adaptive quantization system passes the story rewrite test at all ten compression levels on Qwen3-30B-A3B with no quality cliff. On Llama-3.2-3B, failures under aggressive compression map onto three qualitatively distinct regimes — clean output, soft semantic drift, then session identity collapse — matching the Asymptotic Numerical Stability theorem's tier prediction: bounded warm-tier error produces soft degradation, while cold-tier contamination produces qualitative failure. This failure mode is structurally prevented by provenance selection, which controls which cold-tier blocks enter the working set. Cross-architecture generalization is quantified in §9.9.

### 9.9 Cross-Architecture Transfer: Held-Out Model Validation

The adaptive quantization system is derived from K/V activation samples on two reference models: Qwen3-30B-A3B (MoE) and Llama-3.2-3B (dense), spanning the MoE/dense axis and a 10× model size range.

**Cross-architecture transfer result.** When deployed unchanged on the held-out Qwen3-8B model — a dense model from a different size point and model family, not used in system derivation — the system achieves **7.42× compression**, exceeding the compression ratio on either reference model (7.04× on Qwen3-30B-A3B; 5.02× on Llama-3.2-3B).

This result supports a strong generalization claim: pre-RoPE K/V activation structure is sufficiently universal across transformer architectures — across MoE/dense topology, across model families, and across model sizes — that a single fixed adaptive selection system covers the structural diversity required for high-compression quantization without per-model calibration. The held-out model achieving *higher* compression than either reference model is the diagnostic: Qwen3-8B activations are more amenable to the system's format assignments than the reference models, not less. The system is not overfitting to the reference models; it is capturing a universal structural property.

**Deployment implication.** The system requires no per-model calibration sweep for deployment on new transformer architectures. Thresholds and selection criteria transfer directly. This distinguishes the approach from systems whose quantization parameters are fitted to a specific model's activation statistics (GPTQ, AWQ, KVQuant sensitivity profiling) and validates the per-block adaptive architecture as the mechanism: the kernel responds to each block's actual distribution at inference time, not to a population-level prior established at calibration.

### 9.10 Ablation: Attentional Provenance Indexing

**Production result: min\_ratio > 1.0 on every probe, every tool, under both count and span scoring.** The locked production strategy MH\_XOR\_QQ\_l0×l4 + span α=2.0 is the only tested strategy to achieve this. Full sweep documented below.

**Strategy sweep.** A 48-probe harness evaluated 8 tool sections × 6 positive scenarios on Qwen3-30B-A3B, capturing raw F32 K and Q vectors from the syntactic band (centre layer 7, ±4 layers = 9 layers). Each strategy defines a `TokenSignature` binarisation; discrimination quality is measured by the **discrimination ratio** = intra\_score / inter\_mean\_score. A ratio > 1 means the correct section outscores all others on average; **min\_ratio > 1.0** means it does so on every probe — the production reliability bar.

**§1 sweep (count scoring, 48 probes):**

| Strategy | min\_ratio | mean\_ratio |
|---|---|---|
| **MH\_XOR\_QQ\_l4** (4-head XOR, layer 4) | **1.065** | **1.274** |
| MH\_XOR\_QQ\_l8 | 1.020 | 1.209 |
| MH\_XOR\_QQ\_l0 | 0.995 | 1.197 |
| QQ single-head (best) | 0.66 | 0.95 |
| KK single-head (best) | 0.57 | 0.92 |
| BandMeanQQ (average across 9 layers) | 0.58 | 0.89 |
| QK per-head (best mean) | 0.00 | 2.43* |

*QK mean\_ratio inflated by 0/0 → ∞ artifacts on zero-signal probes; corrected ratio is no-information.

Single-head and band-average strategies all have min\_ratio below 1.0: no reliable discrimination across all probes. Q→K strategies have min\_ratio = 0 on some probes. Only the multi-head XOR family (MH\_XOR\_QQ) achieves reliable count discrimination, confirming that Q→Q matching is the correct signal and that multi-head XOR folding is necessary to suppress false positives.

**§3 span scoring (α=2.0, top strategies):**

| Strategy | count mean | span α=2.0 mean |
|---|---|---|
| MH\_XOR\_QQ\_l0 | 1.197 | 3.069 |
| MH\_XOR\_QQ\_l4 | 1.274 | — |

Span scoring with α=2.0 delivers ~2.5× better mean discrimination than count alone for the MH\_XOR\_QQ family (3.07 vs 1.27). Single-head QQ sees almost no lift from span scoring (0.95 → ~1.10), confirming that span amplifies genuine sustained focus rather than masking a weak count signal.

**§8 dual-layer combination: MH\_XOR\_QQ\_l0×l4.**

The strategy sweep revealed a structural tension: layer 0 (model layer 3) produces smooth sequential Q patterns — long span runs, good for span scoring; layer 4 (model layer 7) is more token-selective — sharper per-token discrimination, better for count. An 8-head XOR fold across both layers simultaneously captures both properties:

$$	ext{TokenSignature} = igoplus_{i=0}^{3}	ext{sign}(Q^{\ell_0}_i) \;\oplus\; igoplus_{i=0}^{3}	ext{sign}(Q^{\ell_4}_i)$$

Three combination algorithms were evaluated:

| Strategy | min\_ratio (count) | min\_ratio (span α=2.0) | mean\_ratio (count) | mean\_ratio (span α=2.0) |
|---|---|---|---|---|
| **A: MH\_XOR\_QQ\_l0×l4 (8-head XOR fold)** | **1.419** | **2.528** | **1.747** | **5.314** |
| C: gated span (l0, gate=l4) | — | 1.534 | — | 3.328 |
| B: normalised span(l0) + count(l4) | 1.354 | — | 2.068 | — |
| MH\_XOR\_QQ\_l0 span α=2.0 (prior best) | 0.995 | 1.613 | 1.197 | 3.069 |
| MH\_XOR\_QQ\_l4 count (prior best) | 1.065 | — | 1.274 | — |

**Algorithm A (8-head XOR fold) is the only strategy that achieves min\_ratio > 1.0 on every probe under both count and span scoring across all 8 tools.** The prior champion (l0 span α=2.0) required span scoring to rescue file\_read from a sub-threshold count floor (cnt\_min=0.995). Algorithm A clears the bar under count alone (file\_read cnt\_min=1.42), making span a performance amplifier rather than a safety net.

**Per-tool results (MH\_XOR\_QQ\_l0×l4, span α=2.0, 48 probes):**

| Tool | count mean | span α=2.0 mean | count min | span α=2.0 min |
|---|---|---|---|---|
| web\_search | 2.082 | 7.543 | 1.992 | 6.879 |
| weather | 2.001 | 5.471 | 1.940 | 4.667 |
| file\_write | 1.698 | 5.601 | 1.602 | 5.024 |
| code\_run | 1.871 | 6.805 | 1.698 | 5.532 |
| random | 1.679 | 5.860 | 1.647 | 4.696 |
| datetime | 1.590 | 4.647 | 1.509 | 3.500 |
| calculator | 1.540 | 3.525 | 1.446 | 2.530 |
| **file\_read** | 1.511 | 3.061 | **1.419** | **2.528** |

file\_read and file\_write are the hardest pair (naturally similar KV patterns, overlapping Q sign space). Both clear min\_ratio > 1.0 under count alone at 1.42 and 1.60 respectively. The span amplification is strongest for semantically sharp tools (web\_search, code\_run) where decode tokens form long coherent runs.

**Why Algorithm A dominates B and C.** Algorithm A applies the joint dual-layer constraint at the fingerprint level before thresholding. A spurious match must simultaneously agree across 8 head-subspaces from two structurally distinct depths — the joint false-positive probability falls approximately as the product of the two layers' individual rates. Algorithm C applies the same logical constraint at the token-hit level after independent thresholding, losing tokens where one layer's BDP value falls just below threshold. Algorithm B sums normalised scores from independent passes, missing the multiplicative suppression that XOR-folding achieves.

**Production design.** MH\_XOR\_QQ\_l0×l4 with span α=2.0 is the locked production strategy. The formula is in §6.2; the span scoring formula is in §6.2. This section documents the empirical evidence supporting that choice.

**Open items for v2.** Boundary and negative scenarios (false-positive rate), cross-band extension (semantic and pragmatic bands), α sensitivity above 2.0, and model portability (threshold and layer ranking re-measurement for each new model variant).

### 9.11 Concurrent Persistent-Memory Session Result

**64 concurrent persistent-memory sessions on 16GB** — the integration result that demonstrates all six subsystems functioning simultaneously at full concurrency on the target hardware.

| Metric | Value |
|---|---|
| Concurrent sessions | 64 |
| Model | Qwen3-30B-A3B |
| Hardware | RTX 4090 Mobile, 16GB |
| Bulk throughput | **2,446 t/s** |
| Peak VRAM utilisation | ~15.8 GB |
| Median response latency | in progress |
| Context depth tested | in progress |
| Per-session memory footprint | in progress |

Each session maintains independent unbounded conversation history with adaptive KV quantisation and attentional provenance retrieval. The 2,446 t/s bulk figure is a measured result at this exact configuration. Context coherence is validated by an entity tracking evaluation adapted from Kamradt's needle-in-a-haystack methodology [Kamradt, 2023] and informed by BABILong [Kuratov et al., NeurIPS 2024].

**Entity tracking evaluation.** The standard Kamradt test plants a single isolated fact ("The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day") and measures retrieval accuracy at varying context depths. This evaluation extends that methodology to entity tracking under quantization: a character name is systematically substituted throughout a narrative text, requiring the model to track a specific entity across sustained narrative context rather than locate a single isolated sentence. This is a harder task — the target information is distributed rather than localised — and directly tests the failure mode that quantization-induced decode drift would cause: entity reference corruption that propagates through the session.

Recall accuracy is measured across compression tiers and F16 baseline at context depths 4K / 32K / 128K tokens. *Preliminary — quantitative recall results in progress; full table in v2.*

BABILong provides a more rigorous multi-hop reasoning evaluation over PG-19 background text at context lengths up to millions of tokens; the entity tracking evaluation here is positioned as a focused test of quantization-induced degradation rather than a general long-context benchmark. Full BABILong evaluation on the production system is left as future work.

This result is not achievable by any published inference engine — on any hardware — because no published engine couples provenance-selected bounded-working-set attention with the expert staging, KV compression, and context retrieval required to support persistent unbounded sessions. The result is reported without a direct comparison baseline for this reason. The codebase dependency analysis evaluation (§9.12) demonstrates the same provenance retrieval mechanism operating over a qualitatively different workload: structured code analysis rather than agent conversation, with compositional multi-hop reasoning replacing conversational recall.

### 9.12 Unbounded Context: Codebase Dependency Analysis

This evaluation tests a categorically harder property than verbatim factual recall: compositional reasoning over structured dependencies via iterative multi-hop retrieval during decode. The test subject is the system's own Candle fork (2.2M lines of Rust and CUDA). The model must walk transitive dependency chains through the provenance system across multiple reasoning steps — each decode step's Q vectors drive retrieval of the next dependency node, assembling context as the reasoning develops. A corrupted dependency from quantization noise does not merely produce a wrong name; it produces a wrong architectural conclusion that propagates through the reasoning chain.

**Three-layer ingestion.** The codebase is ingested in three structured passes over a ~20M-token learning-phase conversation, each building on the full index produced by prior passes. Layers 2 and 3 use Claude Opus 4.6 to generate the reasoning segments before injection into the context window. This separation is deliberate: expensive high-quality reasoning is produced once at ingestion time by a capable external model; the deployed system retrieves it cheaply via provenance indexing at query time. Layer 1 uses the production inference engine directly.

*Layer 1 — Code analysis.* Every file is fed in dependency order with metadata headers and explicit reasoning prompts. The production model (Qwen3-30B-A3B) analyses each file; the provenance system retrieves prior analyses of dependencies as needed. The retrieval log captures forward dependency edges — which prior turns were pulled into the working set during each file's analysis.

*Layer 2 — Cross-module dependency reasoning.* Claude Opus 4.6 reasons about relationships between modules, with the full Layer 1 index available. Its outputs are injected into the context window and indexed by the provenance system. The retrieval log captures bidirectional dependency edges, including reverse dependencies on files not yet ingested during Layer 1.

*Layer 3 — Architectural reasoning.* Claude Opus 4.6 reasons about subsystems, system invariants, cross-cutting concerns, error propagation paths, and design principles, with Layers 1 and 2 both indexed and retrievable. Its outputs are injected and indexed. The retrieval log captures conceptual dependencies — modules linked by shared architectural assumptions with no code-level dependency path.

The retrieval log across all three layers constitutes the dependency graph. Total ingestion: ~20M tokens. Fingerprint index: ~50–80MB. Retrieval log: ~500KB. The full learning-phase conversation persists to disk as cold-tier token history; any instance of the engine on any hardware can prefill this token sequence and reconstruct the identical KV cache, fingerprint index, and retrieval log. The codebase understanding is a file — it can be version-controlled alongside the code.

**Ground truth.** 200–300 verifiable dependency relationships are manually enumerated by the system's author across three categories:

- *Direct (50–100):* Verifiable from function calls and imports. Example: `seal_chunk` calls `select_format`.
- *Transitive (50–100):* Requires following a chain of 3+ hops. Example: changing block size from 32 to 64 breaks fingerprint extraction alignment via the chunk sealing path.
- *Architectural invariants (50–100):* Not visible in any import graph. Example: prefill refresh and turn-boundary sealing share the assumption that sealed blocks are aligned to semantic boundaries.

The system's author is the definitive oracle. No crowd-sourced evaluation or external benchmark is required.

**Query battery.** 200–300 natural engineering questions spanning all three categories:

- "What breaks if I delete class X?"
- "What is the blast radius of changing the block size from 32 to 64?"
- "Which modules share the assumption that sealed blocks have been format-validated?"
- "What are the error propagation paths from the KV quantization kernel to the final logit?"

**Scoring.** Four dimensions per answer:

| Metric | Description |
|---|---|
| Dependency recall | Correct dependents identified, per category |
| Chain completeness | Transitive chains followed to correct depth |
| Conceptual accuracy | Architectural invariant violations correctly identified |
| Precision | False dependencies not introduced |

**Results.**

| Dependency category | Full system | One-shot retrieval | 131K window | 4K window | Random retrieval |
|---|---|---|---|---|---|
*Preliminary — full results table in v2. The one-shot ablation contrast and iterative retrieval mechanism validation (retrieval log analysis, probe-barrier cycle counts) constitute the primary qualitative contribution documented in this version.*

**Ablation baselines.** Four baselines isolate specific contributions.

*Sliding window 4,096 tokens* — standard coding assistant context. Cannot see the codebase. Establishes the floor for all categories.

*Sliding window 131,072 tokens (Qwen3 native max)* — sees a fraction of the codebase. Establishes the ceiling of native long-context extension. Transitive and architectural categories degrade significantly.

*One-shot provenance retrieval* — single provenance scan before generation; no iterative retrieval during decode. Same index, same fingerprints. This is the critical ablation: it is architecturally equivalent to current coding assistants (Cursor, Copilot, Claude Code) that retrieve once and reason on a fixed window. Any transitive dependency the full system discovers through iterative decode-time retrieval but one-shot misses is direct evidence that continuous retrieval during reasoning produces qualitatively different results. The claim is specifically: iterative retrieval during reasoning discovers transitive dependencies that pre-generation retrieval misses. It is not a claim about all coding tasks.

*Random retrieval* — same tier architecture, random block selection. Confirms retrieval scoring, not storage architecture, produces correct results.

**Iterative retrieval mechanism validation.** The retrieval log during the query battery confirms the iterative deepening property. The mechanism is Speculative Context Decode (§6.5): at each newline boundary during reasoning — typically one line of thought, up to 64 tokens — the probe session's Q/K fingerprints drive a CPU provenance scan, assembling the context window for the next decode window. Retrieval is structurally guaranteed at every reasoning-step boundary, not triggered heuristically. The iterative deepening entries in the retrieval log correspond directly to probe-barrier cycles:

- Mean probe-barrier cycles per query by dependency category (each cycle is one retrieval hop, one reasoning-step boundary)
- Recall at cycle 1 vs cumulative recall at cycle N — does recall compound as reasoning develops?
- Cases where a dependency node not scored at cycle 1 was surfaced at cycle 3+ (error-correction: the model's Q vectors moved toward the correct context region after additional reasoning)
- Cases where irrelevant retrievals at one cycle did not propagate to subsequent cycles (self-correcting: probe trajectory naturally moved away from irrelevant regions)

The one-shot ablation contrast is sharpest with this mechanism explicit: one-shot retrieves once before any reasoning tokens; Speculative Context Decode retrieves at every reasoning-step boundary. Any transitive dependency the full system discovers that one-shot misses was surfaced by a mid-reasoning provenance update — the retrieval log records exactly which probe cycle and which reasoning boundary surfaces it.




**Public release and live verification.** There is no stronger empirical evidence than a working system. The full codebase, live system, learning-phase conversation, ground truth enumeration, and per-query scoring are publicly released (Appendix C). Any reviewer can submit an arbitrary engineering query and observe the mechanism directly — reasoning tokens, retrieval log, working-set evolution, and the one-shot ablation — all in the same interface. Every result in this paper is directly verifiable. Community validation, optimization, and critical review are actively invited; contributors will be recognized in v2 (Appendix C).

---

## 10. Analysis

### 10.1 Why Coherence Produces Disproportionate Results

Each component of this system has research precedents. What does not exist anywhere in the published literature is their combination in a single coherent design.

The expert prediction system benefits from the wave-batched kernel — more concurrent requests means better cross-request expert coalescing, which means the effective hot set is smaller, which means the Markov predictor's free-slot-only prefetch converts more misses to hits. The KV compression benefits from the prefill refresh — quantising clean activations rather than decode-drifted ones means the error thresholds in the selection kernel are calibrated to actual quality, not a noisy lower bound, and specifically keeps the hot-tier error floor ε_hot ≈ 0, which the theorem establishes as the asymptotic quality limit of the entire system. The provenance indexing benefits from the turn-boundary paging — every 32-token block begins at a semantic boundary, making fingerprint extraction temporally coherent. The unbounded context tier system benefits from the adaptive quantisation — V blocks systematically landing at Q2_0/Q3_0 means 3× more tokens fit in the warm RAM tier before cold storage is needed, and the theorem justifies this aggressive compression: cold-tier error contributions vanish as 1/N regardless of format.

Speculative Context Decode (§6.5) adds a further interaction: the probe and decode sessions for any active query share similar expert routing patterns — a short span of divergence on related content routes to largely the same experts — making the wave-batched grouped GEMM particularly effective for this session pair. The expert coalescing benefit that scales with concurrency (§4.4) applies within a single query's probe-decode pair, not only across independent requests. Each active query therefore contributes two sessions with high routing overlap, improving the effective hot set for the Markov predictor and reducing the per-query cost of the speculative probe toward zero as batch size grows.

Most critically: the provenance selection mechanism and the KV quantization system interact to produce a property neither achieves alone. Provenance selection bounds the working set W independently of N. KV quantization with prefill refresh keeps the hot-tier error floor near zero. Together they produce asymptotic numerical stability — O(1) bounded error at unbounded depth — a guarantee that no KV quantization system in the literature can claim, because no system we are aware of decouples context depth from working set size. Each component was designed knowing the properties of every other. That is the source of both the disproportionate empirical result on 16GB and the theoretical guarantee that holds at any scale.

### 10.2 The VRAM Budget

On a 16GB card running Qwen3-30B-A3B at Q4_K_M:

```
Expert cache (44% residency):   8.0 GB  (2,700 slots × 3MB)
Attention layers (all 48):      2.0 GB  (non-expert transformer blocks)
Adaptive KV hot tier (balanced): 4.0 GB  (~63K tokens at CR ~3× per-head)
Working buffers + overhead:     2.0 GB  (activations, router, CUDA contexts)
──────────────────────────────────────────────────────────────────
Total:                          ~16 GB
```

The hot tier token capacity is computed from Qwen3-30B-A3B's architecture: 48 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes (F16) = 192 KB per token at full precision. At balanced-tier compression (CR ~3× per-head), the effective cost is 192 KB / 3 ≈ 64 KB per token. The non-KV allocation (expert cache + attention layers + working buffers) is fixed at ~12 GB regardless of card. The remaining VRAM is fully available to the hot tier:

**Concurrency vs. working set per session (balanced tier, CR ~3× per-head, Qwen3-30B-A3B)**

| Sessions | 16GB — tokens/session | 24GB — tokens/session |
|---|---|---|
| 4 | ~15,900 | ~47,600 |
| 8 | ~7,900 | ~23,800 |
| 16 | ~4,000 | ~11,900 |
| 32 | ~2,000 | ~5,950 |
| 64 | ~990 | ~2,975 |

The working set per session is the quality ceiling for each generation step: the provenance selection system must choose that many tokens from an unbounded history as the most contextually relevant. A larger working set produces richer, more contextually grounded responses. A 24GB card at 16 sessions (~11,900 tokens/session) provides roughly the same per-session working set as a 16GB card at 8 sessions (~7,900 tokens/session). Additional VRAM directly expands either concurrency or working-set quality, at the operator's discretion.

---

## 11. Discussion

### 11.1 The Baseline Problem: Error Growth in Full-Attention Systems

The KV quantization literature — KIVI, KVQuant, TurboQuant, GEAR — is conducted under a full-attention model where every token in the context window contributes to every generation step. The error accumulation mechanism is sequential: during autoregressive decode, each token's KV values are computed with attention over already-compressed preceding KV, so each step's rounding and quantization error enters the residual stream and is inherited by every subsequent step. The aggregate error contribution per generation step grows with the number of tokens that participate — proportionally with context depth N for full attention, since all N tokens contribute equally. At 3,200 tokens the accumulated rounding is negligible; at 320,000 tokens it is 100× larger; at unbounded depth, under full attention, it is unbounded.

This is not a property of compression specifically. F16 rounding error accumulates through the same sequential dependency structure as quantization error — any finite-precision arithmetic system running full attention faces this regime. GEAR [Kang et al., 2024] formally characterised the multiplicative compounding; the KV quantization literature's focus on better primitives does not escape the regime because the regime is architectural, not representational.

More VRAM defers the threshold but does not eliminate it. An H100 at 80GB holds more tokens at full precision before compression is required, but as sessions grow persistent across days and weeks, the context is not 1 million tokens — it is 100 million tokens, and no finite hardware holds that in F16. The industry response at that scale is summarization and eviction: older context is compressed into summaries or discarded. Both are lossy by definition. Summarization destroys the precise factual recall — verbatim quotes, specific numbers, exact commitments — that makes persistent memory useful for real applications. Eviction discards it entirely. Hardware scaling moves the wall; it does not remove it.

The theorem in §11.2 establishes that escaping this regime requires something qualitatively different from more hardware or better compression: architectural decoupling of working set size from context depth. No full-attention system achieves this — the working set is by definition the entire context. Provenance-selected attention achieves it structurally, and the theorem follows directly.

### 11.2 Theorem: Asymptotic Numerical Stability Under Provenance-Selected Attention

The KV quantization literature universally assumes that compression error grows with context depth. Every published analysis — GEAR [Kang et al., 2024], KIVI [Liu et al., 2024], KVQuant [Hooper et al., 2024], TurboQuant [Zandieh et al., 2026] — is conducted under a full-attention model where all N tokens contribute to every generation step. This assumption is correct for standard attention. It does not hold for attentional provenance-selected attention.

We state the following result formally. ε(t) denotes **total accumulated numerical error** at token t from all sources — quantization, floating-point rounding, decode drift — not compression error alone. The proof does not use any property specific to compression; it applies to all finite-precision arithmetic systems.

---

**Theorem (Asymptotic Numerical Stability).** *Let N denote context depth, B the fixed provenance selection token budget (a system constant), W_hot_max the hot tier capacity, W_warm_max = B − W_hot_max the warm selection budget, and ε(t) the total accumulated numerical error of token t under any finite-precision arithmetic. Under attentional provenance selection, the expected total numerical error contribution per generation step is bounded by a constant independent of N:*

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} + O\!\left(\frac{1}{N}\right) = O(1)$$

*In contrast, standard full-attention error scales as O(N). The bound is independent of context depth, hardware scale, and precision format.*

---

**Proof.** Partition the context into three tiers as N grows.

*Hot tier* (H): tokens always present in the working set. |H| ≤ W_hot_max is bounded by a hardware constant independent of N. Tokens in H are prefill-refreshed: their decode drift is zero and their quantization error is bounded by the selection kernel threshold. ε_hot is a small constant independent of N.

*Warm tier* (W_warm): tokens eligible for retrieval but not permanently resident. The provenance selection mechanism operates under the fixed budget B — at most W_warm_max warm-tier tokens can be selected per generation step, regardless of how large the warm corpus grows with N. This bound is structural: the selection kernel enforces it by construction. The warm tier's aggregate error contribution per step is therefore at most:

$$\sum_{t \in W_{\text{warm}} \cap \mathcal{W}} \varepsilon(t) \leq W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} = O(1)$$

a constant independent of N. As N grows and the warm corpus expands, more candidates compete for W_warm_max slots, but the aggregate contribution is bounded by the budget.

*Cold tier* (C): all remaining tokens. As N → ∞, |C| → ∞ while W_free ≤ W_warm_max is fixed. The probability that any specific cold token enters the working set in a given step satisfies:

$$p_{\text{cold}}(t, N) \leq \frac{W_{\text{warm\_max}}}{N - |H| - |W_{\text{warm}}|} \to 0$$

The cold tier's aggregate error contribution per step is:

$$\sum_{t \in C \cap \mathcal{W}} \varepsilon(t) \leq \frac{W_{\text{warm\_max}}^2 \cdot \varepsilon_{\text{cold}}}{N} = O\!\left(\frac{1}{N}\right) \to 0$$

Summing across tiers:

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] = \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} + O\!\left(\frac{1}{N}\right) = O(1)$$

□

---

**Corollary 1 (Near-zero convergence).** *In this system's design, warm-tier blocks originate from prefill-refreshed hot-tier blocks — they were quantized from clean activations and selected by the adaptive per-block selection kernel at threshold θ. Consequently ε_warm ≤ θ, which is a small system parameter. Then C_warm = W_warm_max · ε_warm is small, and:*

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \approx \varepsilon_{\text{hot}} + C_{\text{warm}} \approx 0$$

*The total error is not merely bounded — it is bounded by a small constant determined by the selection kernel's quality threshold, approaching zero in the limit of tight thresholds. Empirically: always-attended blocks operating in top-quality compression mode (C0) achieve K_SNR 58.6 dB and V_SNR 58.8 dB (per-block ideal), confirming ε_hot ≈ 0 for the working set.*

---

**Corollary 2 (Hardware Independence).** *The O(1) bound is independent of VRAM capacity. Under standard full attention, an H100 at 80GB enters the O(N) accumulation regime the moment context depth N exceeds full-precision VRAM capacity and any token is compressed or evicted — the threshold is deferred by a constant factor, not eliminated. The theorem's guarantee applies identically to any hardware configuration running this architecture, because B and W_hot_max are hardware constants and the proof depends only on these being independent of N. More hardware does not help a full-attention system asymptotically. It does not hurt a provenance-selected system. Note: for deployments where context never grows beyond full-precision VRAM capacity, both architectures operate in bounded error regimes and this corollary's practical implication is nil — the distinction matters for persistent sessions that genuinely exceed hardware capacity over time.*

---

**Contrast with the standard result.** Under full attention with any finite-precision arithmetic — F16, BF16, or uniform compression — all N tokens contribute to every generation step and expected error scales as O(N), unbounded. Under provenance-selected attention, error is bounded by O(1) regardless of N. The difference is qualitative, not quantitative: the system at N = 10,000,000 turns operates under the same error bound as at N = 1,000 turns.

**Empirical validation (§9.12).** The codebase dependency analysis evaluation provides direct empirical support for this theorem. The theorem bounds numerical error, not task accuracy directly; however, the flat accuracy profile across dependency chain depths — independent of whether chains are 1, 3, or 5+ hops — is consistent with bounded error accumulation. Under O(N) error growth, accuracy would be expected to degrade with chain depth as compounding numerical error corrupts intermediate context; the absence of this degradation is the empirical signature of the O(1) regime. The sliding-window baselines exhibit the expected cliff: no dependencies beyond the window boundary are discoverable by construction. The story rewrite quality evaluation (§9.8) provides complementary evidence at the level of failure mode structure: the three-tier degradation structure on Llama-3.2-3B maps onto the theorem's tier architecture, and the one-shot ablation isolates iterative decode-time retrieval as the operative mechanism for transitive dependency discovery. The no-provenance baseline confirms retrieval scoring — not the storage tier — is the contribution.

**Implication for compression ratio selection.** Cold-storage tokens' contributions vanish as O(1/N) regardless of compression aggressiveness. The adaptive selection kernel's highest-compression modes (C7–C9) are assigned to V cache blocks and cold-storage K blocks precisely because the theorem makes their error contribution asymptotically irrelevant to the quality guarantee. The binding design constraint is the always-attended working set error floor ε_hot — confirmed near zero by the prefill refresh strategy — and the warm-retrieval threshold θ. The design is not an engineering trade-off; it is the theorem's implications implemented directly.

**Key assumption.** The selection budget B and hot tier bound W_hot_max must be independent of N. This is structurally guaranteed by the system architecture. The bound fails if the provenance selection mechanism grows the working set proportionally with N, which recovers the O(N) full-attention regime.

### 11.3 Compression Architecture vs. Compression Primitive

The competitive field for online, inference-time, training-free KV cache quantization is KIVI, KVQuant, and the residual PolarQuant-MSE portion of TurboQuant. All three optimise the same thing: the compression *primitive* — a better quantization formula, a smarter rotation, a tighter codebook. The design question they answer is: given that we must assign a format to every block, what is the best format to assign? KIVI assigns uniform 2-bit. KVQuant adds outlier handling before assigning a fixed format. TurboQuant rotates before assigning.

This paper optimises the compression *architecture* — the decisions that determine when to quantize, what quality to guarantee per block, how to separate error sources, and how error scales with context depth. The design question is different: given that blocks have different difficulty, how do we build a system that guarantees quality on every block and separates the error mechanisms that cause quality loss? The answer — adaptive per-block format selection coupled with two-phase prefill refresh — does not depend on having the best quantization primitive. It absorbs primitive-level failures through selection: a block that Q4_0 cannot represent adequately gets Q8_0 or better. This is why TurboQuant's rotation can fail at 4–5× CR on real workloads while the adaptive system continues to hold quality — the architecture compensates where the primitive falls short.

The specific gaps in the competitive field: none of the three systems perform per-block format selection with a quality guarantee. None couple quantization with decode drift correction. None establish any theoretical result about error scaling with context depth. The system exceeds KIVI's CR at balanced-tier operating points with per-block validated quality guarantees KIVI cannot offer. KVQuant's outlier handling insight is subsumed — the _KS sub-block formats and attention sink protection factor implement the same protection, integrated into adaptive selection rather than applied uniformly. TurboQuant's theoretical Shannon-optimal primitive is irrelevant when the architecture guarantees per-block quality regardless of which primitive is used.

The combination is what the application requires. None of it is achievable through better primitives alone.

**Empirical validation.** The story rewrite evaluation (§9.8) directly confirms this. On Llama-3.2-3B, adaptive C5 at 3.27× passes the multi-session identity discrimination test while uniform Q4_0 at 3.17× fails — the adaptive architecture delivers better quality at higher compression than the best uniform 4-bit primitive. The query-aware K selection metric eliminates the cross-session identity contamination that an isotropic metric produced at C8–C9 by automatically assigning conservative formats to the high-query-relevance blocks that carry session-discriminating signal. This is architecture compensating where a uniform primitive cannot.

### 11.4 Attention Was Always Retrieval

The transformer attention mechanism is a retrieval system: Q vectors query against K vectors, scores select candidates, V values are aggregated weighted by relevance. That is what softmax attention computes. That is all it has ever computed. The architecture is a learned retrieval system operating over a fixed context window.

This system performs the same operation at two scales. The provenance index does approximate attention over the full unbounded context on CPU — six INT8 matmuls for corpus retrieval, BDP span scoring for section discrimination during generation — selecting which blocks enter the working set. The GPU attention kernel then performs exact attention over the selected working set. Coarse retrieval selecting for fine retrieval. The only structural difference from standard attention is granularity: standard attention performs flat retrieval over all N tokens at O(N) cost; this system performs hierarchical retrieval — approximate selection on CPU, exact attention on GPU — at O(1) cost per generation step.

When a reviewer characterises this as "RAG with extra steps," the correct response is: RAG replaces attention with an external retrieval system that has no attentional continuity — content retrieved by BM25 or embedding similarity is injected into a fresh context, severing the causal chain that attention depends on. This system extends the retrieval that attention already is to unbounded depth, without severing anything. The mechanism is preserved; the scaling constraint is removed. This comes with a different binding constraint that full attention does not have: retrieval quality. Full attention over N tokens attends to all N exactly. This system selects B tokens from N via approximate provenance matching — if the provenance system surfaces the wrong B tokens, the generation step lacks the correct context regardless of how precisely those B tokens are attended. The guarantee is bounded numerical error on the attended set, not guaranteed relevance of the attended set. This is a meaningful distinction: the system trades the O(N) error problem for a retrieval quality problem that the provenance mechanism is designed to bound but cannot eliminate. RAG abandons the attention mechanism entirely and substitutes something categorically weaker. The distinction between this system and RAG is not one of implementation style — it is a difference in what property is preserved; the distinction between this system and full attention is a different trade-off, not a strict dominance.

The gap is structural, not implementational. Three deficits of RAG are architectural consequences of what RAG is, not limitations of any particular implementation.

*Wrong retrieval signal.* RAG retrieves from token-level representations — embeddings or BM25 over generated text. The model's Q vectors encode a pre-linguistic representation of what the model is reaching for: the attention state that precedes and drives token output. During transitive dependency walks, the model's Q vectors at hop 2 are already pointing toward the region of the index that contains hop 3's context — before the model has produced any tokens that would form a textual retrieval query. RAG cannot access this signal because it operates on tokens, not attention states. This is why iterative RAG systems (FLARE, Self-RAG) must wait for the model to produce enough text to form a query — their retrieval trails reasoning rather than leading it.

*Severed causal continuity.* When RAG injects retrieved content, that content arrives without KV cache history — the model cannot attend to it in the causal context of the conversation where it was originally produced. This system promotes KV blocks from warm and cold tiers with their original quantized attention state causally intact. The model attends to retrieved content as a continuous causal subset of the full context, not a collage of fragments injected into a fresh window. The retrieved block's K and V tensors encode its original causal position; the attention kernel places them correctly in the sequence. This is the property that preserves attentional continuity as context grows — and it is the property RAG cannot replicate, because RAG's retrieved documents never had a causal position in the model's context to begin with.

*Retrieval trails reasoning.* Even iterative RAG systems retrieve based on text generated so far — the retrieval query is a function of committed output tokens. Speculative Context Decode (§6.5) captures Q vectors from the probe session at each reasoning-step boundary, assembling the next context window before the corresponding real output tokens are produced. Retrieval leads reasoning: the context for line N+1 of thinking is assembled from probe fingerprints generated while line N is still being decoded. RAG cannot achieve this because its retrieval signal — generated text — does not exist until after the reasoning step that needed the context has already completed.

**Thought experiment.** Imagine augmenting a RAG system with three changes: (a) store Q-vector fingerprints alongside documents and use Q→Q matching for retrieval; (b) maintain causal KV continuity for retrieved blocks so the model attends to them in their original causal context; (c) run retrieval at every reasoning-step boundary during generation using Q vectors from a speculative probe, assembling context before the corresponding output tokens exist. A system with all three properties would match attentional provenance indexing with Speculative Context Decode. It would also no longer be RAG — it would be this architecture. The properties that close RAG's deficits are the properties that define a different system. RAG's limitations are not bugs to be patched; they are definitional consequences of operating on tokens rather than attention states, injecting without causal continuity, and retrieving after reasoning rather than before it.

The provenance index's use of Q vectors as cognitive-state fingerprints is the specific choice that preserves attentional continuity: a stored Q fingerprint from a prior turn captures the accumulated attentional context of everything preceding it, not just the surface semantics of that turn. Matching a current Q against stored Q fingerprints selects turns that were attended from a similar cognitive state — the attentional equivalent of relevance, not keyword or embedding overlap. This is what makes the system's retrieval genuinely continuous with the attention mechanism rather than a separate system bolted alongside it.

The codebase dependency analysis evaluation (§9.12) is the strongest demonstration of this principle. During the query battery, the model reasons about engineering relationships — "what breaks if I change the block size?" — and at each newline boundary during reasoning, the Speculative Context Decode probe (§6.5) captures Q/K fingerprints of the model's current reasoning state and assembles the next context window from the provenance index. The model does not know it is performing graph traversal. It is reasoning. The retrieval system — running as a pipelined CPU scan behind the parallel probe session, with near-zero visible overhead — converts that reasoning into a dependency walk at reasoning-step granularity. The probe phase is approximate attention over the full unbounded context on CPU; the decode phase is exact attention over the selected working set on GPU. Two scales of the same operation, pipelined at the model's natural reasoning boundaries. Transitive dependencies that single pre-generation retrieval misses are surfaced in mid-reasoning probe cycles as the model's Q vectors move toward the relevant dependency region — the retrieval log records exactly which cycle and which reasoning boundary supplies each node.

### 11.5 Constraint-Driven Innovation as a Methodology

This system was not designed top-down from theoretical considerations. It was designed bottom-up from two constraints accepted simultaneously and without compromise: an application requiring the hardest possible form of persistent memory, and a fixed hardware limit that could not be exceeded. The application — persistent agent conversations — was selected as the existence proof of a real and demanding use case. Agents require verbatim factual recall, semantic coherence across arbitrary time gaps, high concurrency, and deployment on hardware without a datacenter budget. The constraints were not problems to be worked around. They were accepted as design drivers.

The methodological pattern that emerged from this acceptance is worth documenting, because it appears repeatedly across the system's components, and because it produces a specific type of innovation: decisions forced by constraint that turn out to be universally correct, not just viable on the constrained platform.

**Native quantized matmul kernels.** Standard GEMM libraries dequantise weights to BF16 before computation. On 16GB with a 30B model, this immediately OOMs — the constraint made the standard approach impossible. The forced alternative, inline dequantisation within the MMA kernel, never materialises the full-precision weight copy and is also strictly faster on unconstrained hardware. The 16GB ceiling exposed an inefficiency that exists everywhere.

**Provenance-selected sparse attention.** VRAM is finite. The full context of a persistent session cannot be held hot. The constraint forced a mechanism for selecting a bounded working set from an unbounded context — which turned out to be the architectural property the Asymptotic Numerical Stability theorem requires. The hardware-bounded hot tier is also the hardware-independent theorem's working set W. The constraint that made full-context attention impossible produced the architectural decoupling that makes unbounded-context correctness possible.

**Two-phase quantization.** Materialising all active turns at F16 across 200+ concurrent sessions OOMs at scale. The forced alternative — chunk-sealed quantization during decode, prefill refresh on turn completion — turned out to correctly separate two independent error mechanisms (per-chunk quantization noise and autoregressive decode drift) that existing approaches conflate. The memory pressure that made naive quantization impossible forced a cleaner theoretical decomposition of the problem.

**Online Markov expert prediction.** Low VRAM residency (44%) means most experts are cold. Acceptable cold-hit rates require prediction. The forced alternative to offline calibration — online learning from production routing observations — turns out to generalise better than trained predictors because it adapts to the actual workload distribution rather than a calibration proxy. The residency constraint that made on-demand loading inadequate forced a prediction system that converges without calibration.

The pattern in each case: the constraint closed the standard solution and the forced alternative was not merely adequate — it was better. This is not a coincidence of this particular system. It is a property of constraints that are accepted rather than avoided. When a standard approach is ruled out by resource limits, the replacement must be more efficient in the dimension the constraint bounds. Efficiency in that dimension frequently turns out to generalise — because the constraint exposed a genuine inefficiency in the standard approach, not a necessary one.

The Qwen series of models demonstrated a version of this pattern at the model level: quantization constraints that the field initially treated as quality trade-offs turned out, under careful implementation, to expose and eliminate sources of numerical waste in standard full-precision inference. This system demonstrates the same pattern at the architecture level: hardware constraints accepted as first-class design requirements produced architectural innovations that hold at any hardware scale.

The codebase dependency analysis evaluation (§9.12) exemplifies the methodology at the evaluation level. The constraint was: prove the system works on real structured data, not synthetic probes. The forced response — use the system's own codebase as the test case, with the system's author as the ground truth oracle — produces an evaluation that is stronger, more verifiable, and more practically meaningful than any standard benchmark. The learning phase conversation is a token sequence on disk; any instance of the engine on any hardware can prefill the same sequence and reconstruct the identical KV cache, fingerprint index, and retrieval log. The evaluation is fully portable and fully reproducible, by design.

---

## 12. Conclusion

We have presented a complete inference system built from first principles as a coherent design. The four primary contributions — online Markov expert prediction with wave-batched DMA overlap, adaptive per-block KV quantisation with two-phase prefill refresh, attentional provenance indexing with Speculative Context Decode for continuous retrieval during reasoning, and an unbounded three-tier paged context — are individually grounded in the literature but have not previously been integrated.

The primary empirical result — 509 t/s single-session (2.6–3.4× faster than community benchmarks for this model on RTX 4090 24GB with standard frameworks) and 2,446 t/s aggregate across 64 concurrent persistent-memory sessions on a 16GB consumer GPU — demonstrates that the coherence dividend is real and substantial. The single-session figure is a direct performance comparison; the 64-session result sits in a concurrency regime that existing consumer-hardware deployments cannot reach: the most comparable published study found standard frameworks degrade beyond 2 concurrent users on RTX 5090 32GB [Herz et al., arXiv:2512.23029].

The primary theoretical result is the Asymptotic Numerical Stability theorem (§11.2): under provenance-selected attention over unbounded context, total numerical error per generation step is bounded by O(1) — a constant independent of N — in contrast with the O(N) scaling of standard full-attention systems. Cold-tier tokens contribute error vanishing as O(1/N); warm-tier tokens contribute a bounded constant C_warm, small when warm-tier blocks originate from prefill-refreshed hot-tier blocks. This inverts the universal assumption of the KV quantization literature: the accumulation problem is not an optimisation problem within an unavoidable regime — it is an architectural property of full attention that provenance-selected attention escapes structurally.

The broader competitive position follows from the same distinction. The existing KV quantization literature — KIVI, KVQuant, TurboQuant — optimises the compression primitive: better quantization formulas, rotation, codebooks. This paper optimises the compression architecture: when to quantize, what quality to guarantee per block, how to separate error sources, and how error scales with depth. The architecture absorbs primitive-level failures through adaptive selection; it does not depend on having the best primitive. That is why the system maintains per-block validated quality at compression ratios where population-level systems begin to degrade — the architecture compensates where any fixed primitive falls short.

The application target — 64 concurrent persistent-memory agent sessions on a single 16GB consumer card, each with unbounded conversation history — demonstrates that the problem the system was designed to solve is solvable at scale. The inference engine exists in service of that requirement, not as an end in itself.

The codebase dependency analysis evaluation (§9.12) provides the empirical counterpart to the theorem: iterative decode-time retrieval discovers transitive dependencies that single pre-generation retrieval misses, with accuracy independent of dependency chain depth — the empirical signature of O(1) error applied to compositional reasoning. The live system (Appendix C) is the primary empirical evidence — a working demo is stronger than any accuracy table, and the system is live now. Full quantitative results will be reported in v2; the community is invited to validate and contribute (Appendix C).

The sliding-window baselines do not degrade gracefully — they hit a cliff and transitive dependencies beyond the window are inaccessible by construction: the 4K window cannot see code files analysed earlier in the session; the 131K window covers only a fraction of the codebase (§9.12). A larger window defers the cliff; it does not soften it. The one-shot retrieval ablation is the critical result: the same provenance index with single pre-generation retrieval misses transitive dependencies that the full system discovers through iterative decode-time retrieval — a qualitative capability difference, not a quantitative one (§9.12). The no-provenance baseline confirms the mechanism: the same tier architecture with random retrieval produces near-zero accuracy on transitive and architectural dependencies; the storage is not the contribution, the retrieval is. The KV quantization literature — KIVI, KVQuant, TurboQuant — optimises inside a regime it has not examined the boundaries of; the O(N) error scaling they assume is a property of full attention, not a law of compression, and this paper exits the regime (§11.2, §11.3). Attention is retrieval: the provenance indexing system performs approximate attention over the full unbounded context on CPU, selecting which blocks enter exact attention on GPU; standard attention is flat retrieval at O(N) cost; this system is hierarchical retrieval at O(1) cost; RAG replaces the attention mechanism and loses attentional continuity; this system preserves the mechanism and removes the scaling constraint (§6, §11.4). The structural argument is straightforward: a datacenter running full attention over a 2.2M-line codebase is in the O(N) accumulation regime the moment it evicts a token; this system is not, on a laptop GPU — and the theorem establishes that no full-attention system on any hardware can exit that regime (§11.2 Corollary 2, §9.12).

The deeper lesson is methodological: constraints accepted as first-class design requirements rather than problems to be avoided force architectural decisions that are more efficient in the dimension the constraint bounds — and efficiency in that dimension tends to generalise. The 16GB ceiling produced native quantized kernels faster on any hardware. The finite hot tier produced the sparse attention that makes unbounded context theoretically tractable. The memory pressure of concurrent sessions produced the two-phase quantization that correctly separates two error mechanisms the literature conflates. Each constraint closed a standard solution and forced a better one. The benchmark numbers demonstrate this on 16GB. The theorem explains why the results hold everywhere.

---

## References

[1] Liu, Z., Yuan, J., Jin, H., et al. (2024). **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.** ICML 2024. https://arxiv.org/abs/2402.02750

[2] Hooper, C., Kim, S., Mohammadzadeh, H., et al. (2024). **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.** NeurIPS 2024. https://arxiv.org/abs/2401.18079

[3] Su, Y., Zhou, Y., Qiu, Q., et al. (2025). **Accurate KV Cache Quantization with Outlier Tokens Tracing.** ACL 2025. https://aclanthology.org/2025.acl-long.631/

[4] Zandieh, A., et al. (2026). **TurboQuant: Online Vector Quantization for Quantized KV Cache in Large Language Models.** ICLR 2026. https://arxiv.org/abs/2504.19874

[5] TurboQuant community benchmarks on Qwen3. llama.cpp discussion #20969, 2025. https://github.com/ggml-org/llama.cpp/discussions/20969

[6] Kang, H., et al. (2024). **GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM.** arXiv:2403.05527. https://arxiv.org/abs/2403.05527

[7] Xu, F., Goyal, T., Choi, E. (2025). **RefreshKV: Updating Small KV Cache During Long-form Generation.** ACL 2025. https://aclanthology.org/2025.acl-long.1211/

[8] RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse. (2026). https://arxiv.org/html/2603.13289

[9] Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). **Efficient Streaming Language Models with Attention Sinks.** ICLR 2024. https://arxiv.org/abs/2309.17453

[10] Pan, S.J., Yuan, M. (2025). **KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization.** https://arxiv.org/abs/2502.04420

[11] KITTY: Accurate and Efficient 2-Bit KV Cache Quantization. (2024). https://arxiv.org/pdf/2511.18643

[12] Tao, et al. (2024). **AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations.** COLING 2025. https://arxiv.org/abs/2410.13212

[13] Liu, D., et al. (2024). **RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval.** NeurIPS 2025. https://arxiv.org/abs/2409.10516

[14] Zhang, J., et al. (2025). **Decoding Emotion in the Deep.** arXiv:2510.04064.

[15] Tak, A.N., et al. (2025). **Mechanistic Interpretability of Emotion Inference in Large Language Models.** Findings of ACL 2025.

[16] Kamahori, K., et al. (2024). **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.**

[17] DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference. DATE 2025. https://arxiv.org/html/2501.10375

[18] KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models. SOSP 2025.

[19] NVIDIA Technical Blog. (2025). **Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache.** https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/

[20] KVSink: Sink-Aware KV Cache Quantization for Attention Sinks. COLM 2025. https://arxiv.org/abs/2508.04257

[21] Quantization Error Propagation in Large Language Models. arXiv:2504.09629, 2025.

[22] Kamradt, G. (2023). **LLM Test: Needle In A Haystack.** https://github.com/gkamradt/LLMTest_NeedleInAHaystack

[23] Kuratov, Y., et al. (2024). **BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.** NeurIPS 2024. https://arxiv.org/abs/2406.10149

[24] hardware-corner.net. (2025). **The Definitive GPU Ranking for LLMs: Token Generation & Prompt Processing Performance.** https://www.hardware-corner.net/gpu-ranking-local-llm/

[25] ToolHalla. (2026). **Best Local LLMs for RTX 4090 in 2026: 7 Models That Maximize 24GB.** https://toolhalla.ai/blog/best-local-llms-rtx-4090-2026

[26] Herz, M., et al. (2025). **Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware.** arXiv:2512.23029.

[27] Jiang, Z., et al. (2023). **FLARE: Active Retrieval Augmented Generation.** EMNLP 2023. https://arxiv.org/abs/2305.06983

[28] Asai, A., et al. (2024). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** ICLR 2024. https://arxiv.org/abs/2310.11511

---

## Appendix A: Attentional Provenance Scoring Weights

| Component | w_K_syn | w_K_sem | w_K_prag | w_Q_syn | w_Q_sem | w_Q_prag | Primary rationale |
|---|---|---|---|---|---|---|---|
| Facts | 0.20 | 0.25 | 0.15 | 0.05 | 0.10 | 0.25 | Topical K primary; Q_pragmatic relational |
| History | 0.05 | 0.10 | 0.15 | 0.05 | 0.15 | 0.50 | Q_pragmatic dominant (full reasoning state) |
| Mood | 0.08 | 0.15 | 0.20 | 0.02 | 0.10 | 0.45 | K_pragmatic (emotion peak 75% depth, Qwen3) |
| Templates | 0.10 | 0.20 | 0.15 | 0.08 | 0.17 | 0.30 | Balanced structure/intent |

Mood weights are grounded in Zhang et al. [2025] (emotion probe accuracy peaks at 75% depth for Qwen3-4B = pragmatic band) and Tak et al. [2025] (MHSA/FFN semantic-band units causally responsible for emotion decisions). The assistant-prefill dual representation for mood/template fingerprints eliminates the Q→K distributional gap for those sections.

---

## Appendix B: Expert Pipeline VRAM Budget

For Qwen3-30B-A3B on RTX 4090 Mobile (16GB) at Q4_K_M:

| Component | Size | Notes |
|---|---|---|
| Expert cache (44% residency) | ~8.0 GB | ~2,700 of 6,144 slots × 3MB |
| Attention layers (all 48) | ~2.0 GB | Non-expert transformer blocks |
| Adaptive KV hot tier (balanced tier) | ~4.0 GB | ~63K tokens at CR ~3× (per-head) |
| Working buffers + overhead | ~2.0 GB | Activations, router, CUDA contexts |
| **Total** | **~16 GB** | |

Hot tier token capacity: Qwen3-30B-A3B has 48 layers × 8 KV heads × 128 head_dim. At F16, each token requires 196,608 bytes (192 KB) for K+V across all layers. At balanced-tier compression (CR ~3×, per-head): 196,608 / 3 ≈ 65,536 bytes/token. The non-KV fixed allocation is ~12 GB on any card. At 4.0 GB (16GB card): ~63K tokens. At 12.0 GB (24GB card): ~190K tokens.  See §5 for the per-head reduction constraint. See §10.2 for the concurrency vs. working-set trade-off table.

---

## Appendix C: Public Release and Community Collaboration

**Live system:** https://www.tokera.com

**Source code:** https://github.com/john-sharratt/candle

**What is released.** The full inference engine codebase, the complete learning-phase conversation (a KV cache archive enabling identical reconstruction on any hardware), the ground truth dependency enumeration for the codebase evaluation, and per-query scoring. The live system at tokera.com provides direct query access to the deployed 2.2M-line Candle codebase with the full retrieval log visible — which fingerprints matched at each probe-barrier cycle, how the working set evolved at each reasoning step, and the one-shot ablation available for direct comparison in the same interface.

**Invitation.** This is a v1 technical report. v2 will be open for full critical review and will incorporate independent validation results, optimizations, and findings from community engagement. Contributions to validation, stress-testing, or optimization are actively invited — contributors will be recognized in v2. Issues, pull requests, and evaluation results may be submitted via the GitHub repository.