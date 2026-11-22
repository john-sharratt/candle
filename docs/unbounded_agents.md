# One Card, One Stack: Constraint-Driven Architecture for Asymptotically Stable Inference over Unbounded Agent Memory

> **Version 2.** v1 was published on 13 May 2026 ([10.5281/zenodo.20156060](https://zenodo.org/records/20156060)) and is archived unchanged, with an erratum, at `docs/unbounded_agents_v1.md`. v2 replaces v1's single-machine throughput section with a measured evaluation across three machines (RTX 4090 Laptop GPU 16 GB, RTX 3090 24 GB, RTX PRO 5000 Blackwell 72 GB) and twelve models from 0.5B to 284B parameters, including context-depth sweeps to 128K tokens and a comparison against every published figure for the same models on the same classes of card (§9.6). Every number in §9.6 is produced by a test in this repository and recorded, with its source row, in `docs/performance.md`; the raw rows are in `docs/results/`. v1's headline figures of "509 t/s single-session" and "2,446 t/s across 64 sessions" were prefill rates at one and twenty contexts and are withdrawn (see the v1 erratum). Kernel benchmarks (§9.4–§9.5) are carried over from v1, and v1's provenance calibration is summarised in §9.10 for the principle it established. Sections that describe the v1 design on Qwen3-30B-A3B rather than the production system are marked where they begin (§4.1–§4.5, §10.2, Appendix A); provenance as production runs it — signature, scoring, normalization, the pool and slot, and reprojection — is §6.
>
> **The system runs in production, not in prototype.** The substrate, the provenance index and the gateway are live: the retrieval gates in §9.11 run against the daemon's resident gallery over HTTP (`POST /v1/substrate/zend/scan`), and the journals, the tool catalog and the code-reading ingest described in §9.13 are the same store those gates read. The assistant that reads and writes that store is served by whichever model is attached to the gateway — the local engine for ingestion and for the fleet measurements in §9.6, external models for some reasoning passes — and this paper does not claim otherwise. The dependency battery of §9.12 is run on production as seven arms over 128 questions, each part of provenance retrieval switched off in turn, and the theorem's direct test — selection from a history twelve times the working set — is §9.11.2.

**Abstract**

Persistent agentic systems require context that grows without bound. Under standard full attention every cached token takes part in every subsequent step, so the stored error that can reach a step's output — and the number of perturbed scores competing with the relevant tokens for attention mass — grows with context depth. More VRAM defers the point at which tokens must be compressed or evicted; it does not change that coupling. We argue that the coupling is architectural rather than representational, and show that decoupling the working set from context depth is sufficient to bound that exposure. We present a complete system built around that decoupling.

The theoretical result (**Theorem §11.2 — Asymptotic Numerical Stability**): under provenance-selected attention over a tiered context, the summed stored error of the tokens that enter a generation step — from any source, including floating-point rounding — is bounded by a constant independent of context depth N, where the same bound under full attention grows linearly in N. When retrieved blocks were sealed from prefill-refreshed activations under a per-block threshold, the constant is small. The theorem bounds what a step is exposed to, not the error in its output; it does not show that full-attention output error grows, and §11.2 says what it does and does not establish.

The system is built on four integrated contributions: (1) an online self-learning Markov expert prediction system with DMA offload and wave-batched grouped GEMM for MoE inference under partial VRAM residency; (2) an adaptive per-block KV quantization family spanning FP16 down to sub-2-bit formats, with boundary-aware sub-block structure, two-phase prefill refresh replacing decode-drifted K/V at every turn seal, and per-block selection across eleven compression levels (C0–C10) with asymmetric K/V error metrics matched to the softmax-amplified K and linear-bounded V error propagation paths — the top level compressing the KV cache 4.1×–7.6× across twelve models with every session's output validated; (3) attentional provenance indexing via Q-vector cognitive-state fingerprints, re-selecting the working set during decode — every 64 decoded tokens or at a line break, from the signatures of the last 128 decoded tokens — over a GPU tensor-core scan of a VRAM-resident gallery (~9.5 ms on a 16 GB laptop GPU) whose result feeds the next reprojection rather than blocking the current forward; and (4) an unbounded three-tier paged context (VRAM-hot, CPU RAM-warm, disk-cold) with adaptive quantization calibrated to the bound. Each contribution originated from a hard constraint that closed the standard solution and forced an architectural choice that, we argue in §11.5, also holds on unconstrained hardware.

Implemented in Rust on a custom Candle fork with native quantized matmul kernels that never materialise a full-precision weight copy, the system runs a **180B-parameter mixture-of-experts model (Qwen3.8-Flash-Next) on a 16 GB laptop GPU with 32 GB of host RAM**, serving eight concurrent sessions at 108.5 t/s aggregate decode — above every published single-GPU run of the model, each on a machine with 64–128 GB of RAM. Its throughput is flat in context depth: from 32K to 128K tokens it keeps **99% of its prefill and 111% of its decode**, because the model's own trained sparse-attention indexer bounds the positions each token attends — the bounded-working-set principle this paper builds on, applied inside one context by the checkpoint rather than across contexts by provenance. The top compression level holds the KV cache at **up to 7.6×**, written inline as each block is sealed, at nearly 4× the compression of llama.cpp's q8_0 cache while slowing decode less. Serving concurrent conversations, one card's aggregate decode exceeds the best published llama.cpp single-stream rate for the same model and card class by **1.3–37.9×**, reaching **2,869.9 t/s across 64 sessions** of Qwen3.5-35B-A3B on one 72 GB card; and a **284B model (DeepSeek-V4-Flash) serves sixteen concurrent sessions from a single GPU** at 1,177.1 t/s prefill and 114.4 t/s aggregate decode, 4.1× the best published single-GPU decode (§9.6).

An evaluation protocol is specified in §9.12 using the system's own 2.2M-line Rust/CUDA Candle fork as the test subject: the codebase is ingested into unbounded context, then queried by iterative multi-hop retrieval during decode. Retrieval finds needles in a large haystack: probed with a question and the first stretch of the model's reasoning about it, the answer file ranks first of 6,854 files — 40M signature tokens — on 65% of questions and in the top five on 83%, by Q-signature agreement alone. Run on production with tools forbidden, a 24-question pilot is answered from memory at 57.3% recall over three runs, and **holding the working set fixed while the model reasons drops it to 38.8%** — every held run below every reprojecting one (p = 0.05; per question 14 wins to 4, p = 0.031) — with the loss concentrated where the answer lives in a file the question does not name: the reasoning is what moves retrieval to the files that answer it. *[PENDING: battery-128]* Over 128 questions and seven arms, the memory is worth __ points of recall, reprojection during reasoning __, and continuous reprojection __ over a single retrieval, every interval clear of zero. Testing the theorem directly, mid-decode reprojection selects every fact planted in the middle of a history twelve times the working set, at the top compression level as at near-lossless, where a single projection and a rolling window select none (§9.11.2): the bounded working set is delivered at depth. The source and the raw measurement rows are public, and the live system is open to signed-in users (Appendix B).

---

## 1. Introduction

The persistent agentic systems that every major AI lab is building toward — long-running assistants, autonomous agents, enterprise systems accumulating institutional knowledge — all face the same unsolved problem: context must grow without bound, and under standard full attention every token held grows the set of stored, compressed or rounded values that every later step reads. This is not a hardware problem. It is an architectural one. More VRAM defers the threshold; it does not eliminate the structural problem. The industry's current responses confirm this: RAG externalises memory into retrieval, losing the attentional continuity that makes genuine persistent memory different from keyword search; longer native context windows are a hardware investment against an architectural ceiling — at 100 million tokens across months of operation, no finite hardware holds the context at full precision, and summarization and eviction are lossy by definition, destroying the precise factual recall that makes persistent memory genuinely useful.

This paper names the cause and builds a fix. The cause is structural: under standard full attention, every token at every position takes part in every subsequent step, so the error a step is exposed to grows with the context. The fix is architectural: decouple the set of tokens that take part in any given generation step from the total number of tokens in the context. The theorem in §11.2 shows that this decoupling is sufficient to bound the exposure at any depth. We do not prove it necessary — full attention's output is a convex combination of its values, and whether its error grows in practice depends on the attention distribution — but a full-attention system cannot offer the same bound, because the bound's terms are the tokens it attends.

This system did not originate from a theoretical observation. It originated from two constraints accepted simultaneously and without compromise: a demanding application requiring the hardest possible form of persistent memory, and a hard VRAM ceiling that could not be exceeded. The demanding application is persistent agent conversations — chosen because autonomous agents represent the worst-case instantiation of the persistent-memory problem. They require verbatim factual recall across sessions that grow indefinitely, semantic coherence across arbitrary time gaps, high concurrency across many simultaneous agents, and deployment on hardware with no datacenter budget. Any system that works for this use case works for any less-demanding persistent-memory deployment. The constraints forced architectural choices that address the problem at the level of principle: native quantized matmul kernels written because standard GEMM dequantisation OOMed — which also save the dequantised copy's memory traffic on unconstrained hardware; provenance-selected attention implemented because VRAM is finite and full context cannot be held hot — the mechanism behind the hardware-independent bound; two-phase quantisation designed because materialising full turns at F16 across many sessions OOMs — the design that correctly separates two independent error sources prior work conflated. Section §11.5 documents this pattern in full.

We present an inference system designed to implement this architectural requirement — deliberately constrained to a single 16GB consumer GPU so that every architectural choice falls out of the constraint rather than being imposed on top of it, and validated empirically on that hardware and on two larger cards. The system:

- Runs a 180B-parameter mixture-of-experts model (Qwen3.8-Flash-Next, 6B active) on a 16GB RTX 4090 Laptop GPU with 32GB of host RAM, streaming its experts VRAM → RAM → NVMe, and serves eight concurrent sessions at 108.5 t/s aggregate decode with every session's output validated — above every published single-GPU run of the model, each on a host with 64–128GB of RAM
- Keeps Flash-Next's throughput flat in context depth: from 32K to 128K tokens it retains 99% of its prefill and 111% of its decode, with speculative acceptance fixed at 4.85 tokens per step from 8K to 128K — the model's own trained sparse-attention indexer bounding each token's attended set, which the engine runs without adding a depth-dependent cost of its own (§9.6.2)
- Compresses the KV cache up to 7.6× inline, each 32-token block choosing its own format as it is sealed, with prefill within 5% of uncompressed at every depth; at 8K the top level costs 0–8% of decode for 3.3×–6.3×
- Serves concurrent sessions at an aggregate decode 1.3–37.9× the best published llama.cpp single-stream rate for the same model and card class, reaching 2,869.9 t/s across 64 sessions on one 72GB card and 256 concurrent sessions on a 0.8B model
- Runs a 284B model (DeepSeek-V4-Flash) at sixteen-way concurrency on a single GPU: 1,177.1 t/s prefill and 114.4 t/s aggregate decode, 4.1× the best published single-GPU decode
- Achieves per-block validated KV cache compression across eleven levels with asymmetric K/V thresholds, with the top-quality level achieving K_SNR 58.6 dB — consistent with the ε_hot ≈ 0 assumption of the Asymptotic Numerical Stability theorem
- Replaces each turn's decode-drifted K/V with K/V re-prefilled from the finished turn at its seal, so no decoded token's stored state inherits the drift of the tokens decoded before it (§5)
- Retrieval over a live, shared, growing corpus: held-out questions over 2,206 files and ~25.8M signature tokens give R@1 0.310 / R@10 0.554 / MRR 0.397 for a single scan, rising to R@1 0.392 / R@10 0.708 / MRR 0.504 under the working set's momentum (§9.11.1)
- Finds the file that answers a question about its own 2.2M-line Rust/CUDA Candle fork among 6,854 ingested files (40M signature tokens): first on 65% of questions, top five on 83%, from the question and the model's first reasoning about it (§9.12)
- Answers that battery's 24 questions from memory alone, tools forbidden, at 57.3% recall over three runs, declining rather than inventing where it has not seen the source — and at 38.8% when the working set is held fixed while the model reasons, every held run below every reprojecting run: re-selecting the working set *during reasoning* is worth 18.5 points (§9.12)
- Selects every fact planted in the middle of a conversation twelve times its working set into that working set, at C10 as at C0, where a single projection and a rolling window select none (§9.11.2)

The system is implemented entirely in Rust on a custom fork of the Candle framework. This was not a choice of convenience — it was a necessity, because standard GEMM libraries dequantise weight matrices to full precision before computation, which would OOM during prefill on 16GB. Writing native quantized matmul kernels that never materialise a BF16 weight copy was a prerequisite for the rest of the system to exist.

**Three audiences.** The four technical contributions of §1.2, and the inference stack that supports them, bear on three communities on different timescales.

*For inference systems researchers:* a 180B model served to eight concurrent sessions from a 16GB laptop GPU; throughput that does not fall as context grows from 32K to 128K; up to 7.6× KV compression written inline; and one card's aggregate decode at 1.3–37.9× the best published llama.cpp rate for the same model and card class, up to 2,869.9 t/s across 64 sessions (§9.6). The high-concurrency design is not incidental: unbounded context with provenance-selected attention requires simultaneous parallel prefill across semantic boundaries and multi-dimensional decode across concurrent sessions — the architecture only achieves its full performance and quality properties under genuine concurrency load. The system also runs in production as its own workload: a persistent coding assistant whose sessions outlive any single model, whose memory is the substrate described here, and which is the source of the retrieval measurements in §9.11 and the daily record in §9.13.

*For the KV quantization and ML theory community:* the Asymptotic Numerical Stability theorem (§11.2), which reframes the foundational problem of this subfield. The current literature treats KV quantization as an optimization problem — find the best compression scheme within a regime where error grows with context depth. The theorem shows that the regime itself is escapable through architectural means: provenance-selected sparse attention with a working set bounded independently of depth is sufficient to bound the error each step is exposed to at any depth. That changes which questions are worth asking — how good the selection is, rather than how small each token's error can be made — not just the answers.

*For the broader research community:* the constraint-driven innovation methodology documented in §11.5 — a pattern, observed across this system's components, in which constraints that close standard solutions force replacements that hold up beyond the constraint. It is an observation from one system rather than a result, and it is offered as such.

The agent use case is the existence proof that the system's parts can be built and run together. It is not the subject of the paper.

A bound on per-step error exposure that holds at unbounded context depth has not, to our knowledge, been stated for any prior serving system. More VRAM does not supply one: an H100 at 80GB holds more tokens at full precision, but once a token is compressed or evicted it is in the same position as a 16GB card — the threshold shifts, but the structure does not (§11.2 Corollary 2). Summarization is not compression; it discards the verbatim detail — exact numbers, quotes, commitments — that persistent memory exists to keep. The KV quantization literature optimises inside the full-attention regime and asks how to compress within it; it has not asked whether the regime itself can be left (§11.2, §11.3). The constraint that forced this architecture is the one that produced its central property: the 16GB ceiling forced provenance-selected attention, which is the property the theorem uses (§11.5). Attention is retrieval; this system extends the retrieval mechanism that attention already is to unbounded depth, rather than replacing it with something that loses attentional continuity (§6, §11.4). Whether that extension reaches what it should is measured in §9.11.2, and whether the model reasons over what it retrieves in §9.12.

### 1.2 Technical Contributions

1. **Online Markov Expert Prediction with DMA Offload** — a self-learning transition matrix that predicts expert routing from actual inference observations without calibration, combined with a wave-batched grouped GEMM kernel that exploits cross-request expert locality to reduce effective PCIe transfer costs.

2. **Two-Phase KV Cache Quantization with Prefill Refresh** — a turn-boundary prefill refresh strategy that removes autoregressive decode drift from every sealed turn, coupled with an adaptive per-block selection kernel that assigns independent K/V formats based on asymmetric error metrics grounded in the softmax-amplified K vs. linear-bounded V error propagation asymmetry [AsymKV, COLING 2025]. Compression ratios span three tiers from near-lossless (top-quality) to high-compression, validated end-to-end by the multi-session story rewrite test (§9.8).

3. **Attentional Provenance Indexing with Mid-Decode Reprojection** — capturing Q vectors as persistent cognitive-state fingerprints and binarising them into Binary Directional Provenance signatures — packed Q sign bits folded per head — matched Q→Q by sign agreement (XNOR + popcount). The signature design was calibrated in v1 on a 128-bit 8-head XOR fold across two syntactic-band layer depths (MH_XOR_QQ_l0×l4, §9.10); the production signature is its successor over every attention layer — three layer-fold groups of one 128-bit signature per KV head, 192 bytes per token on a four-KV-head model (§6.6). The corpus is scanned on the GPU over a VRAM-resident paged gallery by tensor-core kernels — 1-bit BMMA on Ada at ~9.5 ms steady-state on a 16GB laptop GPU, INT8 IMMA on Blackwell — bit-identical to a scalar reference. During generation the working set is re-selected from the decode's own Q signatures — every 64 decoded tokens, or at a line break once more than 16 have passed, scoring the last 128 decoded tokens (§6.5) — and the scan's result feeds the next reprojection rather than blocking the current forward, so its latency stays off the decode's critical path. A design that would lead reasoning rather than follow it, a parallel probe session decoding ahead of the real one, is specified in §6.5 and is not built. What the memory answers is measured by §9.12's battery, and whether turns far past the working set are selected by §9.11.2. What re-selection during reasoning adds is measured too: holding the working set through the think block takes the battery from 57.3% to 38.8% over three runs per arm (§9.12). The scan's retrieval quality is §9.11.1 and §9.12's scoring lab — the answer file first of 6,854 on 65% of questions.

4. **Unbounded Three-Tier Paged Context** — a VRAM-hot / CPU RAM-warm / disk-cold paged context architecture, together with a proof that the summed stored error of the tokens entering a generation step — from any source, including F16 rounding — is bounded by a constant independent of context depth N under provenance-selected attention, where the same bound under full attention grows linearly in N. The bound is independent of hardware: more VRAM defers the point at which tokens must be compressed or evicted, but does not change which tokens a full-attention step reads. Decoupling working-set size from context depth is sufficient for the bound; §11.2 states what it does not establish.

5. **Native Quantized Inference Stack** (supporting, not one of the four) — greedy decomposition for smooth 1–500 token throughput without remainder handling, a fused sampling kernel covering all common sampling modifiers in one CUDA launch, and native quantized matmul kernels that eliminate the dequant materialisation OOM during prefill.

---

## 2. Background and Related Work

### 2.1 MoE Expert Offloading

Several systems have addressed expert offloading for MoE models under VRAM constraints. Fiddler [Kamahori et al., 2024] and HybriMoE [Zhong et al., 2025] partition experts between CPU and GPU based on activation frequency. DAOP [DATE 2025] extends this with data-aware prefetching. The core limitation shared by these systems is that they rely on offline profiling or trained auxiliary modules to predict expert routing — prediction accuracy degrades on novel distributions, and the calibration requirement creates deployment friction.

Our Markov predictor learns online from actual routing observations, has no calibration requirement, and improves continuously during inference. Combined with the wave-batched kernel that coalesces expert work across concurrent requests, it keeps the GPU working while experts stream under partial VRAM residency (§4.3, §9.7).

### 2.2 KV Cache Quantization

The online, inference-time, training-free KV cache quantization field has three meaningful entries. **KIVI** [Liu et al., ICML 2024] established channel-wise outlier structure in Keys and token-wise structure in Values, motivating per-channel K and per-token V quantization at uniform 2-bit. It is the standard baseline: well-reproduced, widely cited, but a blunt instrument — identical format assigned to every block regardless of activation difficulty. Their Table 2 shows quality degradation on long-context retrieval tasks at 5× CR. **KVQuant** [Hooper et al., NeurIPS 2024] added sensitivity-aware treatment: per-vector outlier handling and special-casing of the first token, achieving sub-0.1 perplexity degradation at 3-bit. The contribution was demonstrating that careful outlier handling makes aggressive uniform quantization viable — but format assignment remains population-level, not per-block. **TurboQuant** [Zandieh et al., ICLR 2026] achieves near-optimal compression via Hadamard rotation and Lloyd-Max quantization. The rotation spreads energy uniformly before quantization, with theoretical advantage below 3-bit. In practice, community benchmarks on Qwen3 find that Q4_0 matches or exceeds PolarQuant-MSE (the residual implementation after removing QJL) at comparable 4–5× compression ratios, suggesting the rotation overhead does not pay for itself in the range where most deployments operate.

**KITTY** [Xia et al., 2025] demonstrates that mixed-precision channel-wise quantization can recover most of the accuracy gap between 2-bit and 4-bit Key caches by boosting 12.5–25% of channels to 4-bit. Their selection uses an offline-computed magnitude ranking; the authors note that "more principled or adaptive strategies may yield stronger robustness accuracy recovery" as future work. This paper addresses that directly. The binary precision-boost decision is replaced with adaptive per-block format selection using per-block reconstruction error in the Q-projected metric for K and normalised-L2 error for V. The selector operates at per-(layer, head) block granularity and is recomputed per operating point. We extend adaptation to V as well as K, providing an 11-point Pareto curve (C0–C10) rather than two configurations, operating on dense, MoE, DeltaNet-hybrid and native-sparse models up to 284B parameters with no per-model calibration data — a model carries at most a scale on the thresholds, set at its validation edge (§9.9).

Everything else in the quantization-adjacent space is either orthogonal or a different design point. Token eviction systems (H2O, TOVA, StreamingLLM) are lossy by definition — they solve a different problem. Trained compression (DMS) requires fine-tuning. SVD-based approaches (xKV, LoRC) are per-prompt and offline-flavoured. KVTC [Staniszewski & Łańcucki, ICLR 2026] achieves ~20× compression via PCA decorrelation and entropy coding but operates offline on completed caches for storage and transfer rather than during live inference — a complementary design point. None of these compete in the online inference-time setting.

OTT [Su et al., ACL 2025] identified that outlier tokens cluster near block boundaries, independently motivating our 4/28 sub-block structure — the same insight that drives KVQuant's first-token sensitivity, generalised to a structural per-block design.

The gap this paper occupies: all three competitive systems optimise the compression *primitive* — better quantization formula, rotation, codebooks. None treat format assignment as a per-block decision with a quality guarantee. None establish any theoretical result about error scaling with context depth. The adaptive selection architecture subsumes the insights of KVQuant's outlier handling (via _KS sub-block formats and attention sink protection) and RefreshKV's drift correction (§2.3), while adding the per-block quality guarantee that makes all three possible simultaneously.

### 2.3 Autoregressive Error Accumulation

GEAR [Kang et al., 2024] formally established that quantization error in autoregressive decode compounds multiplicatively across steps, causing critical generation deviation in long sequences. RefreshKV [Xu et al., ACL 2025] is the only published work addressing this specifically: periodic full-attention KV regeneration recovers precision lost through sequential accumulation. They identified the problem and demonstrated the fix, but as a standalone technique — not integrated with any compression system. Our two-phase architecture takes the same insight and couples it structurally with the quantization system: prefill refresh is not periodic but turn-boundary-triggered, and it feeds directly into the adaptive format selection on the refreshed activations. RelayCaching [2026] confirmed that KV values are highly consistent across prefill and decode phases, validating prefill as the correct ground-truth regeneration strategy.

### 2.4 Attention Sinks

StreamingLLM [Xiao et al., ICLR 2024] established that a small cluster of initial tokens carries disproportionate attention weight and must be preserved under any compression scheme. KVSink [Su & Yuan, COLM 2025] extended this to mid-sequence emergent sinks. The system's treatment of attention sinks reframes this generatively rather than defensively: on attention-only models, turn-boundary sinks are recomputed in the current frame at every assembly rather than stored, closing a class of stale-frame error that defensive protection cannot address (§5). KVTuner [2025] and community benchmarks [llama.cpp #20969, 2025] both identify boundary transformer layers as specifically more sensitive to K quantization on Qwen3.

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
│  MID-DECODE REPROJECTION (generation-time loop)          │
│                                                          │
│  Decode captures each token's Q signature (192 B)        │
│  Trigger: every 64 tokens, or a line break after > 16    │
│  GPU provenance scan over the resident gallery,          │
│    probe = last 128 decoded tokens' signatures           │
│    (~9.5 ms on the laptop, off the critical path)        │
│  Result assembles the working set of the NEXT forward    │
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
│  Device-side routing, no readback  │
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
│  Signatures persisted beside K/V at turn seal   │
│  Gallery pages resident in VRAM, rebuilt on     │
│  demand from the substrate                      │
└─────────────────────────────────────────────────┘
```

The design principle throughout is that no component materialises an intermediate representation that is not consumed by the immediately following operation. The dequantised weight copy that standard GEMM libraries materialise does not exist. The BF16 expansion that standard sampling pipelines require between the logit tensor and each modifier does not exist. Every kernel boundary is a necessary one.

---

## 4. Online Markov Expert Prediction and Wave-Batched MoE

### 4.1 The Problem

*§4.1–§4.3's four-part policy and §4.5 describe the v1 design on Qwen3-30B-A3B. The production cache and dispatch that replaced parts of it are given at the end of §4.3 and in §4.4.*

Qwen3-30B-A3B has 48 MoE layers with 128 experts each — 6,144 experts total at approximately 3MB each (Q4_K_M), totalling ~18GB. No single consumer GPU achieves full residency. On a 16GB card with a 50% VRAM budget for experts, approximately 2,700 slots are available — 44% residency. The dominant inference cost is PCIe transfer latency for cold expert fetches. Without prediction, 56% of expert activations are cold misses, each blocking the GPU compute stream while waiting for a ~116μs DMA transfer from pinned memory (`docs/expert_cache_design.md`).

### 4.2 Online Markov Transition Matrix

The self-learning predictor maintains a per-layer-pair transition count matrix:

```
T[L→L+1] : [128 × 128] float32 co-occurrence counts
```

After each forward pass, the predictor observes which experts fired at each layer and credits `T[L][from][to]` for each pair — **arrival-specialised**: a transition counts only when `to` was not already active at the source layer, so the table learns the cold experts a prefetch must cover. Production keeps one direct table per hop ahead rather than chaining the one-hop table, and emits nothing from a table until it has seen 64 arrivals (`expert_lre/transition.rs`).

Prediction for layer L+h given the active set at layer L scores each successor by pointwise mutual information, which discounts the globally popular experts that are already cached in favour of those the current routing specifically implies:

```
PMI(α)[j] = Σ_i  P(j | i) / P(j)^α     for i in active_experts_L,  α = 0.5
candidates = successors by PMI, excluding the active set, above a confidence floor
```

The candidates are staged into free slots. Speculative prefetch **never evicts** — it only uses free slots — so a misprediction is structurally harmless: it occupies a free slot until normal eviction reclaims it.

**Measured out of sample** (§9.7): over a captured Qwen3-30B-A3B routing trace of 21 diverse prompts, scored by 21-fold leave-one-out cross-validation so no prompt is predicted by a model trained on it, the PMI transition model covers **4.7×** the cold experts a popularity baseline does at a four-expert fan-out, with top-1 precision of 80–91% against popularity's 22% (`docs/archived/markov_expert_prediction_eval.md`). Learning within the session is the dominant lever: a session's own transitions predict its next layers better than any cross-prompt average.

**Why this works better than trained predictors.** Most published MoE prefetching systems require a trained auxiliary module or offline profiling over representative data. The Markov predictor requires neither. It learns the model's own routing tendencies from production inference — meaning it adapts to the actual workload distribution rather than a calibration proxy. For content with strong routing regularities (code tokens, prose tokens, domain-specific content), convergence is fast and prediction accuracy is high.

### 4.3 Four-Part Eviction Policy

Naive timestamp-based eviction causes cascade failures: a cold miss at layer L evicts an expert from layer L+5, causing a new miss that evicts from layer L+10. Empirically, this halved single-token decode throughput before the policy was redesigned.

The four-part policy eliminates cascades:

**Part 1 — End-of-pass batch eviction.** After layer 47, evict the bottom 5% of occupied slots by usage timestamp. This creates ~140 free slots as a shock absorber for the next pass's early layers.

**Part 2 — Layer-aware forced eviction.** When a real miss occurs with no free slots, evict from completed layers first (highest completed layer, then oldest timestamp). This makes it structurally impossible to evict an expert needed later in the same pass.

**Part 3 — Early-layer pinning.** The first two MoE layers, 0 and 1, are pinned — their resident experts are never evicted. These layers run first in every pass with no prior compute to overlap a reload with; a miss at layer 0 is a full stall. Pinning removes that worst case at the cost of the slots the two layers hold.

**Part 4 — Free-slot-only prefetch.** As described above. Mispredictions are inert.

The eviction ordering was chosen by measurement: in the same held-out evaluation, a frequency-weighted recency score cut the expert miss rate from LRU's 22.9% to 14.7% at a VRAM budget of half the working set, where a learned recurrence model did worse than LRU (39.5%) — the reuse signal that matters is frequency and recency, not a model of it (§9.7).

**The production cache: three tiers.** The four-part policy was designed for a two-tier cache in which every expert not in VRAM sat in pinned RAM. The production cache treats the on-disk expert pack as authoritative and VRAM and pinned RAM as independent faster copies of it, so an expert can be resident in both and eviction never writes bytes back. Its eviction key keeps Parts 2–3's intent: access frequency weighted by the wrapped forward distance to the layer's next use — Bélády's ordering, computed exactly because the layer wave is a cycle — evaluated in passes by tier, so a pack-only expert, whose miss costs an NVMe read, is protected over a warm-backed one whose miss is a ~116μs copy from pinned memory. Measured on Qwen3-30B-A3B, RTX 4090 Laptop GPU (16GB), aggregate prefill t/s, the two-tier cache against the three-tier cache it was replaced by (design and full history: `docs/expert_cache_design.md` §12.5):

| Config | Two-tier | Three-tier | Δ |
|---|---|---|---|
| F16 × 1 context | 377.5 t/s | 708.1 t/s | +87% |
| BF16 × 1 context | 509.4 t/s | 780.6 t/s | +53% |
| BF16 × 10 contexts | 1,931.4 t/s | 2,726.1 t/s | +41% |
| Q8_0 × 20 contexts | 2,448.1 t/s | 2,732.5 t/s | +12% |
| C9 × 2 contexts | 951.4 t/s | 1,463.1 t/s | +54% |

The same gate on the laptop now measures **6,139.1 t/s at twenty contexts** with its experts streaming (§9.6, I10) — the three-tier cache is not costing the model its throughput, it is what allows the model to run at all on 16 GB.

### 4.4 Wave-Batched Grouped GEMM

The key insight is that batching prefill and decode requests together and routing them through the expert layer as a wave creates **cache locality in the expert hot set**. When Request A and Request C both need Expert 7, Expert 7's weights load once and serve both. The effective hot set is the union of expert activations across all concurrent requests — which is smaller than the naive sum because similar content routes to similar experts.

The production dispatch (`docs/moe_live_dispatch_design.md`) keeps routing on the device. Per layer:

```
Bucketize: a kernel classifies each routed expert VRAM / pinned RAM / cold,
           and writes a routing summary into a mapped host ring

Experts:   the grouped GEMM runs every assignment; a VRAM-resident expert is
           read in place, a pinned-RAM expert is copied by the expert kernel
           itself, and only a cold expert waits — on a mapped word the stager
           thread publishes once the expert is in pinned memory

Scatter:   the expert outputs are combined into the residual stream
```

No host CUDA call reads the routing or waits on it: the host threads read the summary off the critical path, to stage cold experts, score residency and promote, and nothing on the device waits for them to do so. The wait that remains is the cold miss, which is what the Markov prefetch and the eviction policy exist to make rare: modelled on the held-out trace at 60% VRAM residency, the two together leave ≈13% of expert accesses as stalling misses, and tuned for stall cost rather than hit rate — about 2% forced churn and sixteen prefetch candidates a layer — 7.8%, half the stall cost of the hit-rate optimum (§9.7). In production the predictor's live precision is 66–85%, 74% typically (`docs/expert_cache_design.md` §7).

Cross-request coalescing is the true batch dimension for MoE inference. In a dense model, batching amortises the weight load cost across multiple output elements. In MoE, batching means multiple tokens processed through the same expert simultaneously — and that only happens when the wave kernel aggregates work across concurrent requests.

### 4.5 Early-Layer Pinning and Prediction

The first two MoE layers' experts stay resident in VRAM. They run first in every pass with no compute ahead of them to overlap a transfer against, so a miss there is a full stall; pinned, they never miss, and the predictor and the warm tier both skip them — a host copy of an expert that never misses serves nothing. Prediction therefore runs from layer 2 onward, where it has two layers of compute to hide each transfer behind.

---

## 5. Two-Phase KV Cache Quantization

KV cache quantization in a high-concurrency system faces two distinct error sources that require separate treatment. Conflating them leads either to incorrect designs or to unnecessary memory pressure.

**The first error source: autoregressive decode drift.** During autoregressive decode, each token's KV values are computed with attention over already-quantised preceding KV. Each step introduces small quantisation error that feeds into the next step's activations. GEAR [Kang et al., 2024] established that this compounds multiplicatively, causing coherence degradation beyond approximately 500 tokens. This is a sequential dependency problem — the error accumulates through the causal chain of decode steps.

**The second error source: per-chunk quantisation noise.** When a 32-token block is sealed and quantised, the quantisation error is bounded by that block's own activation distribution. It does not propagate to subsequent blocks because each block is independently quantised from its own activations. Chunk-level quantisation does not compound.

These are independent mechanisms. The correct fix for each is independent.

**Phase 1 — Chunk-sealed quantization (during prefill and decode).** Every 32-token block is quantised immediately when sealed, using the adaptive per-block selection kernel against the block's live activations. Memory pressure stays bounded regardless of session count or turn length — at 200 concurrent sessions with 500-token turns, materialising all turns at F16 simultaneously would require ~100K tokens × 96 KB (Qwen3-30B-A3B: 48 layers × 4 KV heads × 128 × K+V × 2 bytes) ≈ 9.8GB on top of the weights, OOMing a 16GB card. Chunk-sealed quantization keeps the working F16 footprint to one block per session at a time.

A third error-reduction property follows from the chunk boundary structure. The active (unsealed) chunk — the accumulating tail of the current decode stream — is always held at full F16 precision until it reaches 32 tokens and seals. This means the most recent tokens in any session, where autoregressive error would otherwise be highest due to accumulated drift, are always attended at full precision. Quantization error is structurally confined to sealed blocks; the immediate decode tail is always clean.

**Phase 2 — Turn-boundary prefill refresh (on turn completion).** Once a turn is fully decoded and sealed, a batched prefill pass processes the completed turn's tokens in parallel with full-precision activations — no token's computation depends on the quantised cache of a preceding token. The resulting clean KV values replace the decode-drifted blocks for that turn. The cost is proportional to the completed turn's length at F16, not the full context and not all sessions simultaneously.

RelayCaching [2026] empirically confirmed that KV values for identical content are highly consistent across prefill and decode phases, validating that the prefill pass produces the ground-truth KV for a given token sequence.

**Why prefill refresh sets the asymptotic quality floor.** The Asymptotic Numerical Stability theorem (§11.2) establishes that under provenance-selected attention, the expected total numerical error per generation step is bounded by ε_hot + C_ret = O(1), a constant independent of context depth N. Hot-tier tokens — the current turn, pinned system context, and prefill-refreshed recent history — are always present in the working set and are therefore always attended. Their error is not diluted by the provenance selection mechanism; it is the irreducible floor. The prefill refresh is what keeps ε_hot ≈ 0: without it, hot-tier tokens would carry sequential decode drift, raising the asymptotic floor. Warm- and cold-tier tokens share the fixed retrieval budget and together contribute at most C_ret = W_warm_max · ε_ret — bounded at any depth, but not vanishing: a retrieved token contributes its full per-token error however old it is. The practical consequence is that compression format selection can be aggressive outside the hot tier only as far as the per-block threshold bounds the error (it sets ε_ret, and with it the constant), and must be conservative for hot-tier tokens (their error is the system's permanent floor, attended at every step). This is exactly what the two-phase strategy implements: chunk-sealed quantization for all tiers during decode, and prefill refresh specifically for the tokens that will remain hot.

### Boundary Reconstruction via Projection Segments

The two-phase quantization above applies to **interior content tokens** — the tokens whose K/V faithfully summarises position-portable content. The system handles **structural template tokens** — the 4–6 tokens of chat-template scaffolding at each turn seam (e.g. `<|im_start|>role\n` and `<|im_end|>\n` in ChatML), system-block delimiters, tool-block envelopes, and role markers — separately.

**Why structural tokens cannot be pre-baked.** Structural template tokens carry near-zero semantic content but are disproportionately attended to by every content token downstream. The model uses them as orientation markers — "here's where the system block closes," "here's where the user turn begins." The K vectors at those positions encode "I am a closing marker after the preceding content." If the preceding content the K was conditioned on does not match what's actually there at runtime, the orientation signal is noisy. Under provenance-selected attention this mismatch is the common case rather than an edge case: every projection selects a different combination of sections (different tools materialised, different turns selected), so a structural token's K/V baked against any single ingest-time prefix is wrong for any projection whose prefix differs.

**Three structurally independent error mechanisms.** The system faces three distinct error classes, each closed by a different mechanism:

1. *Quantization noise* on interior content — bounded by the theorem (§11.2), O(1) per generation step.
2. *Autoregressive decode drift* — eliminated by Phase 2 prefill refresh (above).
3. *Semantic frame drift via stale structural K/V* — closed by live structural-segment prefill against the current assembly (this section).

A structural token can have ε(t) = 0 (numerically perfect) and still carry attention summary referring to scaffolding that no longer exists on the attention surface. The assumption that a stored token's K/V faithfully summarises its actual attended context holds for interior content tokens; it is violated for stale-frame structural tokens. The architectural response: structural template K/V is never written to the persistent substrate.

**Mechanism — projection segments.** Each projection emits a sequence of three segment kinds in declaration order:

- **`Sealed`** segments reference content K/V already in the substrate (authored system-prompt sections, persisted turn parts). Application is metadata-only: Arc-clone the per-layer sealed sequences and patch block-range entries onto the slot. Microseconds per segment.
- **`Generated`** segments reference structural template tokens by name from the model's dialect catalog (system delimiters, role markers, block envelopes). They carry no persistent K/V. At assembly time they are live-prefilled against whatever sealed content actually precedes them on the slot.
- **`NewUserMessage`** segments carry the current turn's incoming user tokens. They are live-prefilled, captured to a slot-local buffer, and committed to the substrate at turn-seal time as the user half of the resulting turn record.

Adjacent `Generated` segments between two `Sealed` segments form a single **prefill run** — concatenated and executed as one `forward_batched` call. Each structural token's K/V is computed under its actual runtime causal prefix; the orientation signal is correct by construction. Boundaries become deterministic functions of (template token, current causal prefix) — computable, reproducible, never stale.

**Cached by causal prefix, not by position.** A slot keeps the K/V of the structural runs it has computed, keyed by a hash of the run's whole causal context — the identity and length of every piece before it, its own tokens, and the content it bridges to. A run whose key matches is injected like a sealed section instead of being recomputed; any change anywhere in its prefix changes the key, so a cached run is never stale. Structural K is stored un-rotated and rotated at read from its chunk's position base, so a cached run stays exact under a uniform position shift. Entries are retained for a fixed number of projections rather than pruned to the current one, because reprojection can alternate between two prefix shapes, and pruning to the current shape would make the two evict each other forever.

**Recurrent lineages bake it instead.** A model whose per-sequence memory is a recurrence — Qwen3.8-Flash-Next's DeltaNet layers, the production model — cannot compute a token inserted mid-sequence, because its state has already accumulated past the hole. For that lineage every structural producer is sealed: dialect markers and turn boundaries are baked into the turn's own K/V, template sections are sealed against a canonical prefix, and the live turn's opener is folded into its `NewUserMessage`; a projection of that model emits no live structural runs at all, and the engine refuses one that would (`docs/deltanet_state_persistence.md` §4.6–§4.7d). What follows on frame correctness applies to the attention-only models; on the recurrent lineage the structural tokens are stored K/V like any other and fall under the interior bound of §11.2.

**Cost scales with selected boundaries.** A projection's structural cost is the number of prefill runs — one per cross-turn / section boundary in the selection — times their forward cost; the token cost per run is the structural template length, single-digit tokens per boundary. Across slots these prefill runs batch into the standard `forward_batched` calls already used for user prefills and decode steps, so structural prefills are first-class participants in the same pipeline with no new kernel.

**Format quality is locked in.** `Generated` K/V is produced by a fresh prefill pass against live R16/F16 arenas — the same format the model uses during decode. Structural tokens never visit the warm tier, never round-trip through the background quantizer, never pick up Q4/Q3/Q2 quantization error. The most attention-loaded positions are guaranteed uncompressed.

**What this closes.** On the attention-only models, five problems are addressed by the single decision to live-prefill structural segments rather than persist them:

1. Dynamic system-prompt sections rotating in and out across turns no longer corrupt downstream structural K/V.
2. Gathered-turn frame contamination is eliminated: gathered blocks contain no structural K/V to be stale.
3. Compounding projection drift across many turns is closed: no stale frame ever enters a newly-sealed turn's structural computation.
4. Structural quantization edge cases vanish: structural tokens are never compressed.
5. BDP signature signal sharpens (§6.6): structural template ranges previously contributed high self-similarity noise to retrieval; sigs now cover content tokens only.

The substrate gets smaller as a byproduct on those models — structural templates are a per-process detail of the inference engine, materialised from the dialect catalog on demand, never persisted to disk.

---

## 6. Attentional Provenance Indexing

### 6.1 Core Insight

At the moment a model produces or processes any content, the Q vectors it computes are a compressed fingerprint of its cognitive state — encoding not just content semantics but the full accumulated reasoning context in which that content was produced. We call this *attentional provenance*. By capturing each token's Q signature as it is computed and storing it beside the token's K/V, we construct an index, held resident in VRAM (§6.6), that is scanned over the whole unbounded corpus before generation begins and, critically, during generation itself: the provenance index functions as an attention mechanism over unbounded context, re-selecting the working set at every reasoning boundary from the model's own Q vectors.

This is qualitatively different from K vectors, which encode content semantics. A stored Q signature from a conversation turn captures the accumulated attentional context of everything preceding it — topic trajectory, reasoning state, what the model was reaching for. When a future query produces a Q vector in a similar reasoning state, the match surfaces that turn regardless of surface-level token overlap.

### 6.2 From the v1 Fingerprint to the Production Signature

v1 indexed each item by six INT8 vectors — K and Q averaged over three layer bands, ~780 bytes — scored by a weighted dot product on the CPU, with a 128-bit sign signature folded from two early layers for section discrimination, span-scored over runs of consecutive hits, and resolved one dynamic section at a time by short probes (`docs/unbounded_agents_v1.md` §6). Its calibration (§9.10) established the principle the production design keeps — **Q→Q sign agreement over a flat scan**, decode signatures against decode signatures — and two findings that still hold: Q vectors read while prefilling a prompt and Q vectors produced while decoding occupy different regions of signature space, so a probe must be matched against stored text of its own kind; and a binary signature's 50% random agreement is a pedestal that must be removed before scores are summed. Everything else was replaced: the signature by `WideQSig`, every attention layer folded into three groups (§6.6); the dot product and span scoring by per-token votes (§6.3) read against each candidate's own background (§6.4); the CPU scan by the resident GPU gallery (§6.6); and the per-section probes by reprojection during decode (§6.5).

### 6.3 Scoring: Every Probe Token Votes

A scan compares every probe token's signature with every gallery token's by sign agreement and turns the agreements into votes — **late fusion**: per probe token and fold group, each candidate's best-agreeing token gives the candidate an agreement, the agreements are read against their own distribution across the gallery, and the candidate a token picks gets a vote of `z × margin` — how far it stands out, times its lead over the next. A candidate's raw score is its summed votes. Four learnings set the production formula (§9.12 states it in full and measures it):

- **A token votes for its best several, not its single best.** Under one vote per token, a code question's probe gave any vote to only 20–67 of 6,854 files, and seven answer files scored exactly zero — a ceiling no later stage can lift. Voting for the best eight, each vote square-rooted so one spiky match does not outweigh many moderate ones, raised the answer file's mean reciprocal rank from 0.508 to 0.713. Tool selection keeps one vote per token: a tool probe has one clean match, and spreading its vote lets wrong tools over the gate.
- **Only the probe's strongest tokens count.** A needle gate keeps the quarter of the probe's tokens with the largest summed vote; most of a probe's tokens are function words and template, and their votes are noise.
- **Probes find their own kind.** A question probe matches stored questions and a thinking probe matches stored thinking, so every file is stored with calibration questions shaped like the probes that will ask for it, and those are scored as a competition of their own and added at a weight — pooled with the file's content, the questions took 96.5% of a background's vote mass at 2% of the gallery and drowned it. For tools the calibration questions were the single largest lever: Tool-1 rose 74.5% → 87.9% with the scorer unchanged (§9.10).
- **Repairing signatures does not work.** Bit-level centring is a no-op, and a correction that marks down hub files marks down the files real conversations ask about.

### 6.4 Normalization: A Score Read Against Its Own Background

Raw vote totals are not comparable across candidates. A generic file — the workspace root, whose listing names every crate; a crate's `lib.rs` — scores on almost every probe, so under a shared scale the hubs take the slots and the specific answer is crowded out; for one identical question the right folder's raw rank swung from first to tenth across turns. Nothing computable from the gallery alone fixes it: a candidate's self-match, its affinity to other candidates and a cross-probe z-score each failed, because loudness is a property of how a signature responds to the *query* distribution, which gallery-against-gallery comparisons never see (`docs/provenance_score_normalization.md` §2).

The production normalizer learns each candidate's **background** — the scores it reaches on probes that are not about it — from query traffic, and reads a score as how far into that background's tail it sits:

```
surprise(c | x)   = −ln P(background(c) ≥ x)
normalized(c | x) = 1000 × surprise(c | x) / ln(1 / 0.01)
```

so 1,000 is a full hit (beyond 99 in 100 of the candidate's own background), and under the null every candidate — loud or quiet — wins equally rarely by chance. Backgrounds are log-binned counts, so they commute and read the same whatever order their evidence came in; a probe's own answer is left out of its background; a score is read only against the samples that scored, since late fusion leaves most candidates at zero on most probes; and a candidate too new for a background of eight scoring samples reads against its scope's pooled one. A candidate's score is normalized within its score competition — one scope of the projection schema — and a file is read against the files of its own kind.

Measured on 37,284 calibration questions over 1,740 files, each with its own file as the answer: the answer file's R@1 rose from 0.107 raw to **0.239**, and the largest single winner's share of every file's wins on other files' questions fell from 54.5% to **0.1%** (`docs/provenance_score_normalization.md` §7.1). The hit-level normalizer it replaced — an asymmetric moving average meant to settle at a candidate's hit score — reached 0.225, but every probe not about a candidate pulled its level toward that probe's score, so a tool's normalized score came to follow its place in the alphabet in which the warm-up visited the catalog.

**What provenance decides.** Provenance proposes; the model's own attention disposes. On a model whose attention selects (Flash-Next's indexer), a reprojection admits whole conversations — a file's reading, a guide, the dialogue — into a pool several times the attended window, by their normalized scores; the window itself is then filled with rounds of those conversations, ranked by how often the model's indexer selected their blocks or, for pooled rounds not yet loaded, would have (a shadow scan over their index rows). A round's score is `base × (1 + h)`, its provenance score lifted by a hit momentum `h ← 0.8 h + d` over the stride's hit density `d`, so the indexer's hits can bring in a turn provenance did not rank and retire one it ranked but the model never reads (`docs/hit_driven_projection.md` §7). Belief carries across a turn's reprojections the same way: on held-out probes momentum lifted the answer file's R@10 from 0.554 to 0.708 over three turns on a file, compounding the scoring gain rather than overlapping it.

### 6.5 Retrieval During Decode: Reprojection, and Speculative Context Decode

Sections §6.3 and §6.4 describe provenance retrieval *before* generation begins. During generation itself, the model's Q vectors evolve with every token produced, creating a continuous signal for context refinement.

**What runs: mid-decode reprojection.** The production engine re-selects a decoding session's working set from the session's own decode. Every decoded token's Q signature is captured as it is produced (§6.6). A reprojection is triggered every 64 decoded tokens, or at a line break once more than 16 tokens have been decoded since the last one, inside the think block as in the answer; its probe is the signatures of the last 128 decoded tokens, scored against the resident gallery by the GPU scan, with the working set's momentum carrying belief across a turn's reprojections (§9.11.1). The scan's result assembles the working set for the next forward, so the scan never blocks the current one. Retrieval therefore *follows* reasoning by at most one reprojection interval: the probe is what the model has just written, not what it is about to write. The configuration is `candle-conversation/src/config/sequence_config.rs` (`REPROJECT_EVERY_N_TOKENS`, and the probe's reach and the think-block gate, `reproject_max_probe_tokens` and `reproject_in_thinking`, which zend sets as `reproject_probe_tokens` and `reproject_in_thinking`).

**Where a reply begins: the question read twice, and short early strides.** A query's own Q vectors depend on the context it is read against. Every conversation starts on a small set of configured background files, and a question read against them scores the corpus differently from the same question read without them — measured on one battery question, its answer file ranked second of 6,854 alone and fifth against the background (§9.12). So a request is read twice on the conversation's own slot: first prefilled to the token before its last against the working set with the background files held out, and scored, its fresh scores feeding the working set and its hits and K/V discarded; then submitted again unchanged, its opening working set ranked by the first reading's scores beside the background files, prefilled once more, and decoded with no further scan at its opening's end. The reply's first reasoning tokens then take reprojections at 8, 16 and 32 decoded tokens before the 64-token cadence, because what the model writes early in its reasoning — what it believes it has read — stays in its context whatever the working set holds afterwards. The two readings cost one extra prefill of the question and one scan before the first token; the early strides cost three reprojections in a reply's first 32 tokens and none once its reasoning closes.

**A direction, not built: Speculative Context Decode.** Reprojection follows the reasoning by one interval. A probe session decoding a line *ahead* of the real one — its tokens discarded, its signatures scored — would let the working set for the next line of reasoning be assembled from where the reasoning is going, so retrieval would lead it. The two sessions would share the wave and most of their expert routing. It is not in the engine, and no figure in this paper is measured with it.

**Relationship to the theorem.** The working-set selection budget B and hot-tier bound W_hot_max are enforced at every reprojection. The bound of §11.2 is unchanged by how often the working set is updated, because each update selects at most B tokens from the unbounded context: reprojection changes the frequency and the aim of updates, not the bound.

**Structural segments at a reprojection.** Every reprojection resolves its structural segments against the new assembly (§5): on attention-only models adjacent `Generated` segments between sealed content form prefill runs that batch into the existing prefill pipeline, cached by causal prefix; on the recurrent production lineage they are sealed K/V and cost nothing at reprojection.

**Quality implication.** Cost aside, the expected effect on quality is distinct. With the working set re-selected during a thinking block, later lines of reasoning are generated against a working set chosen from what the earlier lines were about — and, with Speculative Context Decode, from where the next line is going. The hypothesised feedback loop is: better context produces more specific Q vectors, which produce more targeted retrieval, which produce better context for the next step. If it holds, a system with a fixed window through the whole thinking block would miss dependencies that are in its index but that the model's Q vectors have not yet reached when it retrieves. §9.12 measures that gap with the window held through the reasoning alone — every other reprojection kept, one taken at `</think>` — and it holds: over three runs per arm the battery falls from 57.3% to 38.8%, every held run below every reprojecting one.

**Middle-context degradation.** A well-documented failure mode of long-context attention is that models recall content positioned in the middle of a long context window markedly worse than content at the start and end [Liu et al., 2023]. Provenance-selected attention changes the conditions under which it arises: the working set at each step is a small window — bounded by B — assembled for that step, so content selected from 50,000 tokens back sits in the same short window as content from 100 tokens back. Whether that removes the effect, or only moves it inside the working set, is not measured here; the expectation is that a short, relevance-selected window is less exposed to it than a long sequential one.

### 6.6 The Production Signature and the Resident Gallery

The production signature is `WideQSig`: each token's `sign(Q)` over every attention layer is folded into three layer groups — the first absorbing all but the top two attention layers, the top two kept alone (`[n − 2, 1, 1]`, derived from the model's geometry by `FoldParams::derive`; `[46, 1, 1]` on a 48-layer stack), each layer bit-rotated before the XOR so correlated layers do not cancel — of one 128-bit head signature per KV head: with four KV heads, 3 × 4 × 128 = 1,536 bits, 192 bytes per token. Retrieval is a decode→decode Q·Q consensus: a query token's signature is compared against every gallery token's by sign agreement (XNOR + popcount), a vote is cast per fold group, and the groups are weighted per score competition (§6.3). It is the successor of the 8-head two-layer fold calibrated in §9.10: the same binary directional principle, over every layer and with no K term. Signatures cover content tokens only — the user's text, the model's decoded text, a file's reading — and never the chat template's structural tokens, whose attention patterns are self-similar orientation signals that would pollute every ranking. At turn seal a turn's full signature window is persisted on the substrate beside its K/V.

Re-uploading every candidate turn's signatures on every scan was the bottleneck of the first GPU implementation, so signatures stay **resident in VRAM** between reprojections in a paged gallery arena modelled on the paged KV cache: fixed 32-token pages, a free-list pool and persistent storage slabs, with the scan kernel reading pre-resolved device pointers rather than gathering into a packed copy. The warm and cold tiers of the gallery are the substrate's own persisted signatures; the arena holds only the hot tier and rebuilds an evicted turn on demand.

Three scan backends are verified **bit-identical** by an adversarial parity test: a scalar reference; a 1-bit tensor-core backend (`BMMA.88128.XOR.POPC`, sm_75–sm_89), ~9.5 ms steady-state on the RTX 4090 Laptop GPU; and an INT8 tensor-core backend (`mma.m16n8k32.s8`), the production path on Blackwell, which dropped the 1-bit instruction. The scan's result feeds the next reprojection, so selection and forward execution are decoupled by a full wave and the scan never blocks a forward pass.

---

## 7. Unbounded Three-Tier Paged Context

The context is organised into three tiers managed by the paged allocator:

- **VRAM-hot** — current working set, full-speed attention
- **CPU RAM-warm** — recent history, loaded on demand via PCIe
- **Disk-cold** — full conversation history, loaded on cache miss

All 32-token blocks are aligned to turn and system prompt boundaries. This ensures: the attention sink cluster is always block-local; prefill refresh triggers at clean boundaries; tier promotion loads coherent semantic units.

**Block metadata.** Format assignment is per (32-token block, KV head, K or V palette): each block's heads choose their K and V formats independently when the block is sealed, and the choice is recorded in the per-head descriptor tables the paged attention kernels read (`ArenaTable` / `PerHeadTable`, `candle-nn/src/kv_cache/arena_table.rs`). The kernel reads a block's format once per tile from its table entry and broadcasts it across the warp, adding one global memory read per tile with no change to the inner dequantisation loop.

**Cold tier quality floor.** In a naive full-attention system, cold-tier blocks — the oldest tokens — would be attended by every subsequent token, making their quantization error the most damaging in the entire context. Under provenance-selected attention, this relationship changes: a cold-tier block is attended only in the steps that select it, and the fixed retrieval budget caps how many cold blocks any one step can take in. Averaged over a growing cold tier, the chance that a given block is selected falls as N grows — but a block relevant to the current step is selected at any depth, and then contributes its full quantization error. The cold tier's contribution per step is bounded by the budget, not diminished by depth, so its quality sets part of the theorem's constant. What matters for cold-tier quality is that they were cleanly quantized to begin with — which the prefill refresh strategy ensures. Blocks reaching cold storage were quantized from prefill-quality activations, not decode-drifted ones, and were assigned the highest-precision format consistent with their error threshold by the per-block selection kernel. This provides correctness on retrieval when provenance selection does surface them, while the theorem guarantees their aggregate contribution at any depth is bounded.

**Why cold storage must persist KV cache blocks, not tokens.** A naive design would store only the token sequence on disk and reconstruct the KV cache on demand by prefilling. This is correct for standard inference engines — prefill is deterministic and produces the same KV values every time. Under provenance-selected attention it is not. The KV values produced at inference time reflect the specific working set that provenance selection assembled at each reasoning step: the model attended to a particular subset of historical context, and those attention patterns shaped the hidden states that produced the K and V tensors for each token. A reconstructed prefill over the raw token sequence — without the same provenance-selected working set — will produce subtly different KV values because the model's attention context was different during original generation. The difference is small but cumulative: hot-tier and warm-tier blocks reconstructed from tokens will carry higher error than the originals, raising ε_hot above zero and degrading the asymptotic quality guarantee. Cold-tier blocks must therefore be stored as compressed KV cache data, not as token sequences. The persisted data consists of **interior content K/V cache blocks** only (authored system-prompt sections and the user/assistant parts of completed turns); structural template positions carry no persistent K/V and are reconstructed live from the dialect catalog against the current assembly's causal prefix at retrieval time (§5). The substrate's redo log (`substrate/substrate.log`) is stored this way: each sealed turn's K/V, its tokens and its provenance signatures. The tokens are kept as the rebuild source for a turn whose K/V is lost or invalidated — a reload re-prefills it — which is a recovery path, not the storage format; a substrate is valid only for the model whose K/V it holds, so a model change costs a re-ingest (§9.13).

**Non-contiguous attention.** The attentional provenance selection kernel identifies relevant blocks from across all tiers. The attention kernel handles non-contiguous gathered block lists alongside the standard sequential paged layout. Selected blocks are promoted to the VRAM hot tier on demand (`elevate_to_hot`), and hot blocks migrate to the warm tier on the persistence thread. Warm residency is pageable CPU memory, so warm→hot runs at roughly half the PCIe bandwidth a pinned pool would reach.

---

## 8. Native Quantized Inference Kernels

### 8.1 The OOM Problem That Forced the Solution

Standard GEMM libraries (cuBLAS, CUTLASS default paths) dequantise weight matrices to BF16 before execution. For a 30B model during prefill:

```
Quantized weights:    ~15GB stored  (Q4)
Dequantized BF16:     ~60GB materialised
```

On 16GB VRAM this is an immediate OOM, not a performance concern. Writing native quantized matmul kernels that dequantise inline within the MMA kernel — producing BF16 values transiently in register file during the MMA, never materialising the full BF16 weight copy — was not a design choice. It was a hard requirement imposed by a deliberately accepted architectural constraint — the decision to build within 16GB and not exceed it, which forced every component to be efficient at the architectural level rather than relying on hardware headroom to absorb inefficiency.

The constraint produced an implementation that also saves work on unconstrained hardware: it never writes or reads back a dequantised copy, so it moves fewer bytes per FLOP during prefill and per token on decode. We have not run a head-to-head against a dequantise-then-GEMM path on a card large enough to hold the copy, so the saving is argued from the traffic it removes rather than measured; the 16GB ceiling that made the standard approach impossible exposed a cost the standard approach pays on every card. This is the pattern the paper's methodology section (§11.5) describes: the constraint forced an architectural decision that holds beyond the constrained platform rather than being merely viable on it.

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

1. **Published-figure comparison** for throughput and compression — every measured row is set against the best figure published for the same model on the same class of card by llama.cpp, vLLM, KTransformers and SGLang, with each outside figure's card, quantization, context length and engine recorded beside it (§9.6). Where our figure is an aggregate across concurrent sessions and the published one is single-stream, the comparison says so: it answers how much work one card does, which is the question this architecture is built for.

2. **Output validation at every measured point** — every throughput row in §9.6 comes from a gate that replays a story-rewrite task across all of its sessions and checks each session's output text against its expected rewrite (§9.8). A row is reported only if it passes. Compression and throughput are therefore never measured on a configuration whose output has degraded.

3. **Ablation studies** for each novel contribution — the value of each subsystem is measured by removing it and observing the delta. Ablations are the correct evaluation methodology for a vertically integrated system where no external baseline shares the design space. The codebase dependency analysis evaluation (§9.12) tests iterative attention-driven dependency walking during decode, scored against ground truth enumerated from the system's own source, as seven arms that remove the memory, the seed files, reprojection during reasoning and continuous reprojection in turn. The other measured ablations are §4.3's cache, §9.7's dispatch, and §9.11.2's arms with and without provenance.

### 9.2 Hardware

Three machines, each measured separately. The engine sizes its VRAM partition from the card it finds, so no row is transferred between machines.

| | RTX 4090 Laptop GPU | RTX 3090 | RTX PRO 5000 Blackwell |
|---|---|---|---|
| VRAM | 16 GB | 24 GB | 72 GB |
| Architecture | Ada, sm_89, native FP8 | Ampere, sm_86, **no native FP8** | Blackwell, sm_120 |
| Host | Core Ultra 9 185H, **32 GB RAM** | i7-10700K, 64 GB RAM | Ryzen 9 9950X3D, 189 GB RAM |
| Host link | PCIe 4.0, **trained ×8**, ~12 GB/s | **PCIe 3.0 ×16**, ~12 GB/s | PCIe 5.0 ×16 |
| Driver model | WDDM | WDDM (GeForce: no TCC) | WDDM for the 2026-10-10 sweep; TCC from 2026-11 |
| Measured | width ladder, engine probe | width ladder | width ladder and depth to 128K |

The 16GB laptop is the design constraint; the RTX 3090 validates the integer-only path (no native FP8, and neither tensor-core provenance backend applies) on a host whose PCIe 3.0 link gives it the same ~12 GB/s as the laptop's ×8 link; the Blackwell card, which holds every model but the 284B resident, is where the depth sweeps are run free of weight paging. The figures in §9.6 are WDDM figures on all three cards — the driver model whose submission overhead is the floor under single-session decode on the smallest models — except where a row names the TCC build of §9.6.6–§9.6.7. WDDM demotes VRAM past capacity and charges system commit for every allocation; TCC refuses an over-capacity allocation and charges nothing (`docs/driver_model.md`).

**Reproducing results.** Every row in §9.6 is a test in this repository, run one model per `cargo` process with the card to itself (no daemon running); the full command list and the per-row source files are in `README.md` and `docs/performance.md`. `huge-context` enables the multi-session paged context path used for §9.8 and §9.11. The three commands below are representative; each feature they name exists in this revision of the workspace.

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

*KV compression curve — per-block format selection across the ladder:*

```
cargo test --release -p candle-nn --features cuda,dont_check --lib \
  sampled_selection::tests::projection::test_candidate_list_compression_curve \
  -- --ignored --nocapture
```

### 9.3 Models

Twelve models, spanning dense, mixture-of-experts, DeltaNet-hybrid and native-sparse latent attention, from 0.5B to 284B parameters. Each runs inside its checkpoint's own native context window, with no position scaling applied.

| Model | Size | Architecture | Weights |
|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 284B / 13B active | native-sparse latent attention, MoE | MXFP4_KO |
| Qwen3.8-Flash-Next | 180B / 6B active | hybrid + QSA + MoE, MTP drafter | Q4_KOEXP (72 GB); Q2_KO experts (16 GB) |
| Qwen3.5-35B-A3B, Qwen3.6-35B-A3B | 35B / 3B active | DeltaNet hybrid + MoE | Q6_K |
| Qwen3-30B-A3B | 30B / 3B active | MoE, 48 layers × 128 experts, top-8 | Q4_K_M |
| Qwen3.8-27B | 27B dense | hybrid | Q6_K/Q8 |
| Qwen3.5-9B, Qwen3.5-0.8B | 9B, 0.8B dense | DeltaNet hybrid | Q6_K |
| Qwen3-8B, Llama-2-7B, Llama-3.2-3B, Qwen2-0.5B | 0.5–8B dense | full attention | Q6_K, Q4_0, Q4_K_M, Q4_0 |

§9.4–§9.5 and §9.10 were measured on Qwen3-30B-A3B at Q4_K_M, as in v1.

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

### 9.6 Throughput, Depth and Compression Across the Fleet

Every figure in this section is produced by a test in this repository: the width ladder (`test_parallel_batched_forwarding*`, a ~700-token prompt at several batch widths and points on the compression ladder), the depth gate (`long_context_gate`, 4K–128K tokens of KV), or the engine probe (`kv_fragmentation`, which drives the full conversation engine under load). Every row validates each session's output text (§9.8) and is reported only if it passes. Source rows, dates and builds are in `docs/performance.md`; the raw rows are in `docs/results/`. **Prefill** is prompt tokens per second; **decode** is generated tokens per second, **summed across the batch** where a row has more than one context.

#### 9.6.1 Width: what concurrency buys on one card

RTX PRO 5000 Blackwell 72 GB, build `21c36efbd` (2026-10-10): fourteen width gates, 212 rows, every session of every validated row reproducing its reference. Each cell is the model's best validated row on that card, with the format and batch width that produced it.

| Model | Best prefill t/s | Best decode t/s | Best compression |
|---|---:|---:|---:|
| Qwen2-0.5B | 139,750.4 (F16 ×60) | 6,586.7 (BF16 ×60) | — |
| Qwen3.5-0.8B | 84,235.6 (C10 ×10) | 10,805.3 (C8 ×256) | 4.38× (C10) |
| Llama-3.2-3B | 25,575.1 (C10 ×5) | 1,153.9 (C9 ×10) | 4.43× (C10) |
| Llama-2-7B | 11,676.0 (F16 ×8) | 1,638.4 (BF16 ×48) | 3.56× (Q4_0) |
| Qwen3.5-9B | 12,348.5 (C9 ×5) | 1,607.7 (C8 ×20) | 5.57× (C10) |
| Qwen3-8B | 11,269.1 (F16 ×2) | 622.3 (C8 ×10) | 5.82× (C10) |
| Qwen3-30B-A3B | 17,019.5 (C5 ×8) | 939.2 (BF16 ×10) | 5.45× (C10) |
| Qwen3.5-35B-A3B | 19,092.3 (C8 ×5) | 2,869.9 (C10 ×64) | 7.03× (C10) |
| Qwen3.6-35B-A3B | 18,516.5 (C8 ×5) | 2,761.0 (C10 ×64) | 6.43× (C10) |
| Qwen3.6-35B hybrid, Precision | 18,819.9 (C8 ×5) | 2,829.5 (C10 ×64) | 6.47× (C10) |
| Qwen3.6-35B hybrid, Performance | 19,577.9 (C8 ×5) | 2,892.5 (C10 ×64) | 6.45× (C10) |
| Qwen3.8-Flash-Next | 7,024.1 (BF16 ×4) | 1,163.9 (BF16 ×16) | 7.13× (C10) |
| DeepSeek-V4-Flash | 1,177.1 (BF16 ×16) | 114.4 (BF16 ×16) | — |

Prefill saturates early — the 35B MoEs barely move from ×1 to ×64 — while aggregate decode scales nearly linearly with width. Qwen3.5-35B-A3B goes from 230.2 t/s at one session to **2,869.9 t/s across sixty-four**, a 12.5× widening within one ladder, and Qwen3.6-35B-A3B from 476.6 to **2,761.0**, 5.8×, with the KV cache at C10 and prefill flat across the same range (`docs/results/sweep_rtx_pro_5000_72gb_2026-10-10_main.md`). This is the wave-batched grouped GEMM of §4.4 measured: one expert load serves every session routed to it, so additional sessions ride weight traffic that was already in flight.

#### 9.6.2 Depth: throughput does not fall with context

The same card, one context, 32K → 128K, BF16. Share of the 32K rate kept at 128K:

| Model | Prefill kept | Decode kept |
|---|---:|---:|
| **Qwen3.8-Flash-Next** | **99%** | **111%** |
| Qwen3.8-27B | 28% | 58% |
| Qwen3.5-9B | 30% | 48% |
| Qwen3.6-35B-A3B | 26% | 49% |
| Qwen3.5-35B-A3B | 26% | 47% |
| Qwen3.5-0.8B | 25% | 86% |

Flash-Next is the one model whose cost per token does not depend on how much history there is, and the reason is in its checkpoint, not in this engine's provenance: its full-attention layers carry a trained sparse-attention indexer (QSA, `{layer}.indexer.*` weights) that caps each token's attended set at 2,051 positions, where the hybrids read the whole prefix. Profiled per call, the attention kernel costs the same at shallow depth on both (0.382 ms at 32K on Flash-Next, 0.394 ms at 8K on Qwen3.5-35B), then stays within 6% to 128K on Flash-Next and grows 3.71× for 4× depth on the 35B. The other models in this table are not a failure of the engine; they cannot acquire the indexer without training, and their retention is also confounded by size — Qwen3.5-0.8B keeps 86% of decode because its step is dominated by depth-independent work (`docs/performance.md` §3.3). Across the full depth sweep on the rewrite task its speculative head accepts exactly **4.85 tokens a step at every depth from 8K to 128K**, and the story is reproduced correctly from behind 128,897 tokens of padding. On the coherence task its prefill runs 1,098.8 / 1,368.9 / 1,342.8 / 1,291.8 / 1,257.5 t/s and its decode 24.7 / 28.3 / 24.6 / 26.3 / 26.4 t/s at 8K / 16K / 32K / 64K / 128K. The published llama.cpp run of the same model on an RTX 4090 24GB keeps 64% of its prefill and 65% of its decode between 6K and 110K [33], so the engine contributes something too: its paged attention and compression add no depth-dependent cost of their own on top of the indexer's. What this row shows is the principle §11.2 rests on — a working set bounded independently of depth costs the same at any depth — demonstrated by a model-native mechanism inside one context. It is not a measurement of provenance-selected attention, which a single-context depth gate does not exercise.

#### 9.6.3 Compression: the top level, inline, validated

Each 32-token block is compressed in the forward that produced it; there is no separate compression pass. C10 against BF16 at 8K, one context, from the Blackwell depth sweep (`docs/performance.md` §3.4, the models measured at 8K or 4K only). Per-card C10 ratios on the current width builds are I6 and I14 in `docs/performance.md`:

| Model | Prefill BF16 / C10 (t/s) | Decode BF16 / C10 (t/s) | C10 compression |
|---|---|---|---:|
| Qwen3-8B | 3,235.1 / 3,212.7 | 42.8 / 41.8 | 6.31× |
| Qwen3-30B-A3B | 3,444.7 / 3,367.6 | 51.2 / 47.2 | 5.94× |
| Llama-2-7B (4K) | 3,684.7 / 3,851.1 | 47.1 / 44.1 | 4.81× |
| Llama-3.2-3B | 6,195.9 / 5,936.8 | 69.9 / 66.1 | 4.62× |

At 8K the top level costs **0–8% of decode for 3.3×–6.3×** across the models measured there. At 32K the largest hybrids pay 27–31% of decode for 6.3×–7.5×; at 128K the cache compresses 4.63×–7.63×, 100% of blocks quantized, with Flash-Next paying 19% for 6.97×. Prefill with C10 runs within 5% of BF16 at every depth measured, and at 128K it is marginally *faster* than BF16 on all six models, because there are fewer bytes to read back. For comparison, llama.cpp's q8_0 KV cache gives ~1.9× and costs 35% of decode at 32K and 45% at 64K; its q4_0 gives ~3.56× and costs 39% and 50% [37]; vLLM's FP8 cache gives 2× [38], and TurboQuant 2.4–3.4× for 20–27% of throughput [39]. At 32K the top level holds **nearly 4× the compression of q8_0 while slowing decode less**.

#### 9.6.4 Against the published record

For each model, our aggregate decode serving concurrent sessions against the best published llama.cpp single-stream decode for the same model on the same class of card. The Blackwell rows compare our RTX PRO 5000 with published RTX 5090 figures.

| Card class | Model | Ours, aggregate (t/s) | llama.cpp best, single-stream (t/s) | Multiple |
|---|---|---:|---:|---:|
| RTX 3090 | Qwen3.5-35B-A3B, C10 ×16 | 1,228.8 | 111.2 [30] | 11.1× |
| RTX 3090 | Qwen3.6-35B-A3B, C10 ×16 | 1,346.3 | 157.66 [35] | 8.5× |
| RTX 3090 | Llama-2-7B, BF16 ×48 | 934.9 | 161.89 [29] | 5.8× |
| RTX 3090 | Qwen3-8B, C8 ×10 | 383.8 | 115.3 [30] | 3.3× |
| RTX 3090 | Qwen3.8-27B, C8 ×20 | 358.7 | 65.28, with MTP [32] | 5.5× |
| RTX 3090 | Qwen3-30B-A3B, BF16 ×10 | 400.5 | 153.6 [30] | 2.6× |
| RTX 3090 | Qwen3.8-Flash-Next, BF16 ×16 | 543.8 | 15 [34] | 36.3× |
| 16 GB | Qwen3.6-35B-A3B, C10 ×16 | 336.7 | 183.29 / 249.33 MTP, IQ3_S resident [40] | 1.3–1.8× |
| 16 GB | Qwen3-8B, C8 ×10 | 308.4 | 102.7 (RTX 4080) [31] | 3.0× |
| 16 GB | Qwen3.8-Flash-Next, BF16 ×8 | 108.5 | 27.5–29 (RTX 5080) [34] | 3.7× |
| Blackwell | Qwen3.5-35B-A3B, C10 ×64 | 2,869.9 | 194.0 [36] | 14.8× |
| Blackwell | Qwen3.6-35B-A3B, C10 ×64 | 2,761.0 | 333.55, with MTP [40] | 8.3× |
| Blackwell | Qwen3.8-Flash-Next, BF16 ×16 (TCC build, §9.6.7) | 1,819.3 | 48.02 [44] | 37.9× |
| Blackwell | Llama-2-7B, BF16 ×48 | 1,675.0 | 300.40 [29] | 5.6× |
| Blackwell | Qwen3-30B-A3B, BF16 ×10 | 997.4 | 226.1 [41] | 4.4× |
| Blackwell | Qwen3-8B, C8 ×10 | 632.9 | 200.4 [41] | 3.2× |

Every row clears the published single-stream rate, by 1.3× to 37.9×. Aggregate against single-stream is the comparison the published record allows — llama.cpp can serve parallel requests, but no aggregate for these models on these cards has been published — and it is the comparison that answers how much work one card does. The published single-card serving runs of the 35B models on a larger RTX PRO 6000 report 5–10 concurrent requests [42, 43]; this engine runs them at 64 on a 72GB card and Qwen3.5-0.8B at 256.

Two rows in the table are not like-for-like and are marked here rather than left to the reader. The Flash-Next row against llama.cpp's 15 t/s is a published run at a ~130K context on UD-Q4_K_XL against our ~700-token prompt on Q2_KO experts; the same model against the published RTX 5090 figure at the same short prompt is 37.9×, and against the RTX 5080 figure at 16 GB it is 3.7×. The 16 GB Qwen3.6-35B row is the weakest in the table on purpose: a 3-bit quant that fits wholly in VRAM decodes one stream faster than streamed Q6_K experts can, and the honest multiple there is 1.3–1.8×, not the 8–14× the same model shows on the larger cards.

#### 9.6.5 The 16GB laptop

The RTX 4090 Laptop GPU with 32GB of host RAM is the constraint the architecture was designed under, and where its results are least expected.

- **A 180B model.** Qwen3.8-Flash-Next (180B total: a 125B trunk with 512 experts, a 51B n-gram table and a 4B MTP head; 6B active) runs from a Q2_KO-expert artifact of ~88GB, streaming its experts VRAM → pinned RAM → NVMe. Every rung of its ladder validates 8/8. On the 2026-10-10 sweep one warm session prefills at 269.1 t/s and decodes at 34.9; eight prefill at 1,127.5 and decode at **108.5 t/s aggregate**, with C10 at 5.79–5.82× and costing only 6.6% of that decode (1,128.7 / 101.3). Every published single-GPU run of the model that states its host uses 64–128GB of RAM; the fastest, an RTX 5090 with 128GB, decodes one stream at 48.02 t/s [44], and the only published 16GB-class run, an RTX 5080 with 64GB, at 27.5–29 [34].
- **A 30B MoE streaming its experts.** Qwen3-30B-A3B at Q8_0 ×20 prefills at **6,139.1 t/s** on the laptop with its experts streaming through the three-tier cache.
- **A 35B MoE serving sixteen users.** Qwen3.5-35B-A3B and Qwen3.6-35B-A3B at Q6_K — ~28GB of weights on a 16GB card — serve ×16 at **330.8 and 336.7 t/s aggregate** with C10 at 7.01× and 6.42×, every session validated. The published 16GB runs of these models are single-stream, at smaller quants that fit resident [40].
- **A small model at workstation rate.** Qwen3.5-0.8B at C8 ×32 prefills at **46,056.8 t/s** and decodes at **3,810.7 t/s aggregate** — above the 24GB RTX 3090's 42,982.2 t/s on the same rung, on a card with half the memory and the same ~12 GB/s host link.

C10 compresses the 35B to 7.01× on the laptop and 7.03× on the RTX 3090, and Qwen3.6-35B to 6.42× and 6.45×; the same selector, the same thresholds, making the same decisions on two different GPUs and two different PCIe links. The selector's per-model thresholds are the only thing that varies between the cards, and they vary by model family, not by silicon.

#### 9.6.6 A 284B model on one GPU

DeepSeek-V4-Flash (284B total, 13B active) on one RTX PRO 5000 72GB, sixteen concurrent sessions: **1,177.1 t/s prefill and 114.4 t/s aggregate decode**, build `21c36efbd`. The best published single-GPU prefill for the model is 748.4 t/s (RTX PRO 6000 Max-Q, all experts on the CPU, 8K) [45], and the best published single-GPU decode is 28 t/s (RTX 5090, KTransformers + SGLang, INT4 experts on the CPU) [46] — every published run holding the routed experts in host RAM, as this one does. The aggregate is 4.1× that decode, and the prefill is above every published single-GPU figure for the model.
#### 9.6.7 Where published engines lead

Single-stream decode on a model that fits wholly in VRAM is llama.cpp's strength: on a 16GB card, a 3-bit Qwen3.6-35B-A3B that fits resident decodes one stream at 183.29 t/s (249.33 with MTP) [40], where this engine's sixteen-session aggregate with Q6_K experts streamed is 336.7 — more work in total, less per stream. vLLM prefills Qwen3.6-35B-A3B at 41,105 t/s on an RTX PRO 6000 [43], against ~7,200 here at one session.

The closest published head-to-head is Strata's RTX 5090 single-session benchmark, replayed on this engine through the conversation engine at C5 as a daemon runs it (`kv_fragmentation::qwen38_flash_next_strata`, median of three runs, warm-up excluded). Ours leads prefill at 4K — 6,064.3 t/s against Strata's 4,269.8 — and trails on decode: 168.4 / 159.1 / 125.3 t/s against Strata's 179.4 / 175.7 / 165.0 at 4K / 32K / 128K, i.e. 82–95% of Strata at 4K and 32K and 69–76% at 128K, with a fastest single run of 181.9 t/s at 4K above Strata's median there. Strata runs on a PCIe 5.0 RTX 5090 with a dedicated CUDA graph path; this engine runs on a workstation card, where one segment submission per MoE layer is the floor. Both mark directions for the next version of the kernels; neither changes the architecture's claim, which is about how much one card can serve, from how large a model, at how much context — and Strata's published run is single-session.
#### 9.6.8 Output fidelity against depth

The throughput rows say compression is cheap; they do not say how much it moves the output, or whether that grows with what a step reads. This gate measures it directly (`batch_test::fidelity`; `docs/results/theorem_depth_tests_2026-11-10.md`). At each depth a needle prompt — four codes of random characters spread through the long-context filler, asked back at the end — is prefilled, sealed at the mode's level as the substrate seals a turn, and the answer read back teacher-forced, so every mode is compared with BF16 over the same prefix at every step. RTX PRO 5000, the card alone:

| Model, attention | Mode | ~4K | ~16K | ~64K | ~124K |
|---|---|---:|---:|---:|---:|
| Qwen3-8B, full (to 32K) | C10 mean KL, nats | 3.2e-2 | 1.1e-2 | 3.9e-2 (32K) | — |
| Qwen3.5-9B, whole prefix | C10 mean KL | 6.5e-3 | 1.2e-3 | 1.7e-3 | 3.5e-3 |
| Qwen3.5-9B | C5 mean KL | 6.6e-3 | 1.0e-3 | 9.4e-4 | 3.6e-3 |
| Flash-Next, QSA-selected | C10 mean KL | 5.9e-2 | 3.0e-2 | 1.08e-1 | 1.16e-1 |
| Flash-Next | C5 mean KL | 9.8e-3 | 1.1e-2 | 2.1e-2 | 7.6e-3 |

Top-1 agreement with BF16 is 95.8–100% in every cell; the run-to-run spread is 0 on the 8B and 9B and at most 4.3e-5 on Flash-Next. On the models whose attention reads the whole prefix, the divergence does not grow with depth across a 30× range — the exposure §11.2 bounds grows, the output error does not (§11.1). The one growth measured is C10 on Flash-Next, about 2× from 4K to 124K while its C5 and C0 rows stay flat: the selecting model, at the top level only, plausibly because its indexer selects over stored keys and a compressed key can move a selection. That attribution is not instrumented.

### 9.7 Ablation: Expert Prediction and Eviction Policy

The three-tier expert cache's measured effect against the two-tier cache it replaced is reported in §4.3 (+12% to +87% prefill across the configurations shown there, RTX 4090 Laptop GPU).

A second, cleaner ablation came from replacing the MoE dispatch itself rather than adding to it. When the live dispatch landed — routing classified on the device, expert kernels copying their own misses, no host round-trip — the dense models in the same sweep moved by −1.2% to +0.5% on best prefill and −1.6% to +0.2% on best decode, with compression identical everywhere: they do not touch the dispatch, so they are a control, and the control reads as a control should. The MoE models moved: Qwen3.5-35B-A3B +24–36% best prefill and +11–23% best decode, Qwen3.6-35B-A3B likewise, and Qwen3-30B-A3B's single-session decode from 45.8 to 54.9 t/s (+20%) on the same card and the same artifact (`docs/results/moe_live_dispatch_rtx_3090_24gb_2026-10-04.md`). The same sweep is the first run on which the Qwen3-30B-A3B engine probe passed on the 24 GB card.

**The predictor and the eviction policy, isolated.** Both were evaluated offline against a captured routing trace of Qwen3-30B-A3B (48 MoE layers, 128 experts, top-8) over 21 prompts spanning code, prose, facts, mathematics and lists, by 21-fold leave-one-out cross-validation: each prompt is predicted by a model trained on the other twenty and learning causally as the held-out prompt decodes, decode-phase routing only, pinned layers excluded from every denominator (`docs/archived/markov_expert_prediction_eval.md`; harness `candle-transformers/src/models/expert_lre/eval/`). Coverage is the share of *cold* experts a prefetch of fan-out k would have staged — the misses it converts.

| Promotion predictor (held out) | Top-1 precision | Cold coverage, k = 1 | k = 2 | k = 4 |
|---|---:|---:|---:|---:|
| Popularity | 21.8% | 2.9% | 5.2% | 9.0% |
| Raw co-occurrence counts | 67.5% | 9.0% | 16.7% | 29.4% |
| PMI(0.5) | 80.4% | 10.7% | 19.9% | 34.9% |
| PMI(0.5), cross-prompt prior + live session matrix | **91.1%** | **12.1%** | **23.4%** | **42.7%** |

At eight candidates a layer the best model stages 67.7% of the cold experts. Per-layer-pair tables beat every coarser sharing (a single shared table falls to 15.2% at k = 4), and weighting by router mass collapses coverage to 23.6%: what predicts is how often a transition occurs, not how heavily it is routed.

| Eviction policy (held out) | Miss rate, budget ½ of working set | Budget 0.7 |
|---|---:|---:|
| LRU | 22.9% | 21.0% |
| LFU | 28.4% | 13.0% |
| Learned recurrence model | 39.5% | 29.1% |
| **Frequency-weighted recency** (`last_used + ln(1 + freq) · 16`) | **14.7%** | **12.7%** |

Together, at 60% residency: ≈82% of expert accesses hit, ≈5% are converted to overlapped loads by prediction, and ≈13% stall; re-tuned for stall cost — a stall counted at ten times a converted load — the stalling share falls to 7.8% and the stall cost by 52%, with no change to the hardware. Modelled as the batched wave, where one expert load serves every session routed to it, aggregate throughput grows with the batch, and prediction pays in the regime between the working set fitting and the link saturating; past saturation, at about 256 concurrent decodes, prefetching every missing expert of the next layer matches the fully tuned optimum within 1–4%, and the predictor is not needed. In production the predictor's live precision is 66–85%, 74% typically.


### 9.8 Quality Evaluation: Multi-Session Identity Discrimination

**Test protocol.** The story rewrite test evaluates end-task quality directly rather than via perplexity. Up to 256 concurrent sessions run simultaneously in one batch, each with a distinct character identity (name and gender assignment, cycling with period 99). The model is instructed to rewrite a narrative passage using the assigned character. A session passes if: (a) the correct name appears, (b) gender pronouns are consistent with the assignment, and (c) no other session's name appears. This test is designed to stress exactly the signal that aggressive K quantization degrades: session-discriminating features that live in the high-relevance sub-block population.

**Results.** Every throughput row in §9.6 passes this test with every session correct — across the 213 rows of the RTX 3090 sweep, the 192 of the laptop sweep and the 212 of the Blackwell sweep, not one session failed. The top compression level, C10, passes on every model in the fleet at 4.1×–7.6× (§9.6.3), sitting just inside the edge where the test starts failing; the one exception on record is Flash-Next's C10 ×8 row under the TCC driver build, which validated 7 of 8 sessions on all three runs of that sweep (§9.6.7) and is reported as a failure rather than rounded away. On Llama-3.2-3B, failures under compression past the validation edge fall into three qualitatively distinct regimes — clean output, soft semantic drift, then session identity collapse. That progression is what the theorem's constant would lead one to expect as per-token error grows: small errors move the text, large ones move which session's content the model attends. The test runs in one forward with no tiering, so it says nothing about which tier the error came from; what bounds the error a selected block can carry in production is the per-block threshold it was sealed under (§11.2). Cross-architecture generalization is quantified in §9.9.

### 9.9 Cross-Architecture Transfer: Held-Out Model Validation

The adaptive quantization system is derived from K/V activation samples on two reference models: Qwen3-30B-A3B (MoE) and Llama-3.2-3B (dense), spanning the MoE/dense axis and a 10× model size range.

**Cross-architecture transfer result.** In v1, deployed unchanged on the held-out Qwen3-8B model — a dense model from a different size point and model family, not used in system derivation — the system achieved 7.42× compression, exceeding either reference model (7.04× on Qwen3-30B-A3B; 5.02× on Llama-3.2-3B). The ordering holds on the current production gates at 8K: **6.31× on Qwen3-8B**, above 5.94× on Qwen3-30B-A3B and 4.62× on Llama-3.2-3B (§9.6.3). Since v1 the same selector has been deployed across twelve models — dense, MoE, DeltaNet-hybrid and native-sparse latent attention, 0.5B to 284B — reaching 4.1×–7.6× at the top level on every one.

This result supports a generalization claim: pre-RoPE K/V activation structure is similar enough across the transformer architectures measured — MoE and dense, several families, 0.5B to 284B — that one adaptive selection system covers them with no per-model calibration data. The held-out model achieving *higher* compression than either reference model is the diagnostic: Qwen3-8B activations are more amenable to the system's format assignments than the reference models, not less, which is not what overfitting to the reference models would produce.

**Deployment implication.** The system needs no calibration data for a new architecture. The selection algorithm, candidate formats and error metrics transfer directly; a model carries at most a scale on the thresholds (its `*_KV_FACTORS` row), set by stepping the top level until the story gate's validation edge is found, which is how C10 sits just inside each model's edge. This distinguishes the approach from systems whose quantization parameters are fitted to a specific model's activation statistics (GPTQ, AWQ, KVQuant sensitivity profiling) and validates the per-block adaptive architecture as the mechanism: the kernel responds to each block's actual distribution at inference time, not to a population-level prior established at calibration.

### 9.10 Attentional Provenance Indexing: The Signal, and Tool Selection

**Which signal: the v1 sweep.** v1's calibration on Qwen3-30B-A3B (`docs/unbounded_agents_v1.md` §9.10) compared every family of signature on a 48-probe, 8-tool harness and fixed the principle the production design keeps. Only sign signatures folded across several heads discriminated the right tool on every probe; single heads, band averages and every Q→K strategy failed on some probes (Q→K scored zero on some), confirming **Q→Q sign agreement** as the signal and multi-head folding as what suppresses false matches. The production signature carries the same principle across every attention layer (§6.6).

A finding of the same calibration explains why absolute thresholds were abandoned: within a content type, the right item's scores and every other item's overlapped in their lower tails, so no threshold separated them and ranking was the only discrimination. The production design takes that to its end — every score is read against the candidate's own background (§6.4) rather than against a threshold.

**Prefill against decode.** Tool selection in v1 first failed outright: the model wrote tool calls with no tool definitions in context. The tools' corpus had been built from decode-phase signatures, while the probe was the user's request read during prefill — and the two occupy different regions of signature space. Over 8 tools, a request's prefill produced one discriminative token, hitting every tool equally; the decode that followed produced 16, every one of them on the right tool. A probe must be matched against stored signatures of its own kind, which is why production stores calibration questions beside every tool and file (§6.3).

**Tool selection in production.** The shipped tool selector (`docs/tool_selection_provenance_results.md` §23–§26) drops the prefill/decode split: at each reprojection the last 128 decoded tokens' `WideQSig`s (§6.6) are scored against a gallery of tagged tool turns, each query token casting a vote per fold group weighted by how far it stands out from that group's agreement distribution and by the margin between the leading tool and the runner-up, with only the strongest quarter of votes kept; belief is carried and decayed across a turn's reprojections, and one to three tools are put in scope. Measured leave-one-out with a question as the probe — the call the probe would otherwise contain names its own tool, so scoring a whole stored turn matches on the answer and overstates the rate (97.4–97.8% Top-1) — over 766 tool turns and 93 tools, with eight to twenty-four plain calibration questions per tool added to the gallery (chance ≈ 1.1% Top-1):

| Gallery | Tool-1 | Tool-3 | Tool-5 | MRR | Recall in scope |
|---|---:|---:|---:|---:|---:|
| 766 tool turns | 74.5% | 92.7% | 93.7% | 0.833 | 77.8% |
| + 8 questions a tool (1,510) | 85.1% | 96.5% | 96.9% | 0.907 | 87.1% |
| + 24 questions a tool (2,998) | **87.9%** | **97.5%** | **97.8%** | **0.925** | **89.0%** |

The lever was the corpus, not the scorer: calibration questions shaped like the live probe lifted Tool-1 by 13.4 points with the probe set held fixed. The code-corpus equivalent is the file-retrieval result of §9.12.

### 9.11 Concurrent Persistent-Memory Session Result

The width ladder of §9.6 drives the batched forward from a clean slate; it never constructs the conversation engine, so it cannot exercise admission, per-turn context projection, the persistence thread, KV compaction or tier migration. The **engine probe** (`candle-conversation/tests/kv_fragmentation.rs`) does: it runs a conversational workload through the full engine on the target hardware — admission, per-turn projection, sealing, compaction and tier migration all live — and checks every session's story rewrite against its expected text.

| Card | Model | Stories correct |
|---|---|---:|
| RTX 4090 Laptop GPU 16 GB | Qwen3.8-Flash-Next | 8/8 |
| RTX 3090 24 GB | Qwen3-30B-A3B | 20/20 |
| RTX 3090 24 GB | Qwen3.6-35B-A3B | 16/16 |
| RTX 3090 24 GB | Qwen3.8-Flash-Next | 8/8 |
| RTX PRO 5000 72 GB | Qwen3-30B-A3B | 20/20 |
| RTX PRO 5000 72 GB | Qwen3.6-35B-A3B | 16/16 |
| RTX PRO 5000 72 GB | Qwen3.8-Flash-Next | 8/8 |

Rows are the latest probe run on each card: laptop `bf291341c` (2026-09-30), RTX 3090 `c375834c8` and RTX PRO 5000 `21c36efbd` (both 2026-10-10). No probe log has a `non-finite` layer, and every story in every probe is correct. Flash-Next is the stronger test: it is the one model in the fleet carrying per-sequence recurrent state outside the paged K/V, so a pass that corrupts either shows as non-finite recurrent layers, and it returns every story correct on all three cards.

The direction of travel is measurable. On the RTX 3090, Flash-Next's gate went from 24.3 to 138.1 t/s single-session decode and from 95.6 to 361.6 t/s at C10 ×8 in ten days (`e596fad8d` → `5776799ac` → `7aecbeab3` → `c375834c8`), every row validated on all four builds, with the probe's story gate green throughout — the throughput work and the correctness work did not trade against each other.

**Recall under compression and depth** is measured three ways: output fidelity against depth under compression, teacher-forced on four planted codes from 4K to 124K (§9.6.8); selection of planted facts from the middle of a conversation up to twelve times the engine's working set (§9.11.2); and multi-hop questions over the system's own codebase memory (§9.12).

#### 9.11.1 Retrieval on the live gallery

Every retrieval figure in this paper is measured on the running daemon's resident gallery over HTTP (`POST /v1/substrate/zend/scan`), not on a copy, with held-out probes: a question never present in the background or the gallery, plus the stretch of thinking it starts — the probe a live projection actually makes.

- **Tools:** the answer tool first on 87.9% of held-out questions over 93 tools, in the top three on 97.5% (§9.10).
- **Code:** the answer file first of 6,854 on 65% of questions, in the top five on 83% (§9.12).
- **Normalization:** read against its background, the answer file's R@1 over 37,284 calibration questions rises from 0.107 to 0.239, and no file takes more than 0.1% of the wins on other files' questions (§6.4).
- **Momentum across a turn:** carrying belief from one reprojection to the next lifted the answer file's R@10 from 0.554 to 0.708 over three turns on a file — compounding the scoring gain rather than overlapping it — at the cost of slower topic switches: with the leak at 0.2, a new file leads in 70% of dialogues after three turns (`docs/results/provenance_gates_2026-10-26.md`).
- **A real conversation:** replayed on the paper conversation itself (`docs/paper_conversation_replay.md`), the scorer finds 91% of the files the conversation should place and puts all 14 of its tool calls' tools in the top ten (`docs/results/provenance_gates_2026-10-25.md`).

We know of no published engine that couples a bounded working set with the expert staging, KV compression and context retrieval this workload needs, so the result is reported without a direct comparison baseline. The codebase dependency analysis evaluation (§9.12) puts the same retrieval mechanism to a harder use: compositional multi-hop reasoning rather than recall.

#### 9.11.2 Selection past the working set

This test asks the theorem's premise directly: whether the bounded working set is delivered when the history is many times larger than it (`candle-conversation/tests/provenance_depth.rs`; `docs/results/theorem_depth_tests_2026-11-10.md`). The engine runs Flash-Next with its slot capped at 32,768 tokens. A conversation is built from the repository's own source files, one file a turn, to 30K, 131K and 393K tokens, with four turns stating vault codes placed together at 45% of its depth — the middle, where neither a fill from the oldest turns nor one from the newest reaches; the last turn asks for the codes back. Each cell is a fresh conversation, and each records the projections its question's decode emitted.

| Needle turns selected into the working set | 30K (19 turns) | 131K (64 turns) | 393K (195 turns) |
|---|---|---|---|
| Provenance with reprojection, C10 | 4/4 | 4/4 | 4/4 |
| Provenance with reprojection, C0 | 4/4 | 4/4 | 4/4 |
| One-shot projection, C10 | 3/4 | 0/4 | 0/4 |
| Rolling window (no provenance) | 3/4 | 0/4 | 0/4 |

**Selection works at depth.** Mid-decode reprojection brought every needle turn into a working set of 15–18 turns drawn from 64 and from 195 — a history twelve times the slot — at C10 and at near-lossless C0 alike, and the arms without it selected none past 30K: a one-shot projection and the append path both fill from the oldest turns. The bounded working set the theorem assumes is delivered at every depth measured, and compression does not change what is selected. What the model then does with what is selected is measured on the codebase memory itself, in §9.12.

### 9.12 Unbounded Context: Codebase Dependency Analysis

This evaluation tests a categorically harder property than verbatim factual recall: compositional reasoning over structured dependencies via iterative multi-hop retrieval during decode. The test subject is the system's own Candle fork (2.2M lines of Rust and CUDA). The model must walk transitive dependency chains through the provenance system across multiple reasoning steps — each decode step's Q vectors drive retrieval of the next dependency node, assembling context as the reasoning develops. A corrupted dependency from quantization noise does not merely produce a wrong name; it produces a wrong architectural conclusion that propagates through the reasoning chain.

**Ingestion as run.** The gallery the battery would query exists and is the one §9.11.1 measures: the production code-reading ingest has the daemon's model, Qwen3.8-Flash-Next on the local engine, read the repository file by file as turns of its own, sealing each turn's K/V and signatures into the substrate — 2,206 files, ~25.8M signature tokens, ~5 GB of signatures at 192 bytes a token — with a calibration question per file stored beside it (§9.11.1). Nothing in it was written by an external model.

The ingested conversation persists as the substrate does (§7) — K/V, tokens and signatures in the redo log — so the codebase understanding is a file kept alongside the code, valid for the model that wrote it. By 2026-11-20 the code-reading layer held 6,854 files and 39,954,048 signature tokens.

**The question.** One claim, on Qwen3.8-Flash-Next over its own codebase memory: *provenance projection supplies the memory the answers come from, and re-selecting the working set while the model reasons is what brings in the files a question does not name.* The battery isolates each part — the memory, the seed files a conversation opens on, reprojection during reasoning, and continuous reprojection against a single retrieval — with every other condition held equal.

**Questions.** 128 questions (`zend/examples/dependency_battery/battery.yaml`), in the proportions of the 24-question pilot below: 43 direct, 43 transitive, 42 architectural. Each lists the files, identifiers or values a correct answer must name, its answer files, and whether the question names its answer's file — all enumerated from the source and committed before the first run, with the analysis below. The direct questions are generated mechanically from the source (a symbol's definition and its non-test callers), so their selection carries no judgement; the transitive and architectural questions are written by hand and checked against the code.

- *Direct:* where a thing is defined and what calls it. Example: where `tier_fits` is defined (`span_geometry.rs`) and what calls it in production (`region_pool/transient_tier.rs`).
- *Transitive:* a chain of three or more hops. Example: how a MoE layer's routing reaches the expert kernels with no host readback — the bucketize kernel, the mapped pinned ring it writes, the word the stager thread publishes, the cold expert's wait on it.
- *Architectural:* an assumption several components share with no single call path. Example: the elastic boundary between the KV side and the expert weights (`weight_floor`) and the check that refuses a transient tier crossing it (`tier_fits`).

**Arms.** Three request-level settings, each a dial of the turn as its thinking and tools are, rather than a restart of the daemon. `projection` is *Full* (every layer provenance selects from), *Limited* (the journals alone) or *Minimal* (the tool catalog alone — and so nothing, with tools off, as every battery question runs). `seed` opens the conversation on its configured seed files, or on none. Reprojection re-selects through the think block (`reproject_in_thinking`), holds it, or selects only once at the opening — the retrieve-once turn. Reprojection exists only where something is projected, so the arms are seven:

| Arm | Seed | Projection | Reprojection | What it isolates |
|---|---|---|---|---|
| A0 | off | Minimal | — | the model alone |
| A1 | on | Minimal | — | the seed files alone |
| D1 | on | Full | opening only | retrieve once, before reasoning |
| B0 | off | Full | held in thinking | — |
| B1 | on | Full | held in thinking | reasoning-time reprojection removed |
| C0 | off | Full | full | — |
| C1 | on | Full | full | the production system |

**Running.** On BlackWorld at one build, through private chats — one fresh chat per question, deleted once scored — every question prefixed with an instruction to answer from memory and call no tool. Sixteen questions run at once, and every batch of sixteen mixes the arms, so whatever drifts across the evening — the gallery, ingest, the card's load — falls on every arm alike. Each arm runs the 128 questions three times: 2,688 answers.

**Scoring and statistics.** Recall: the share of a question's expected items the answer names, any of an item's spellings found case-insensitively; a 10% sample of answers is graded by hand against the source to show that recall tracks correct answers, and every answer is read for invented names. A question's score in an arm is its recall averaged over the three runs. Every interval is a 95% bootstrap interval over questions (10,000 resamples; for a difference, of the per-question paired differences). Every contrast is tested by Wilcoxon signed-rank on the per-question differences, Holm-corrected across the five contrasts below, and reported beside its effect size in points of recall.

**Figure 9.12-1 — Recall by arm.** *[PENDING: battery-128 — figure]* One point per arm with its 95% interval, the three run means beside it; arms ordered A0, A1, D1, B0, B1, C0, C1.

*[PENDING: battery-128]* Recall climbs the ladder the design predicts. The model alone (A0) answers little beyond what any reader of the question could; the seed files (A1) add only what they hold; projection with a single retrieval (D1) adds the memory; and re-selection during reasoning (C1) adds the files the question does not name — the production system scores __% against the model's own __%.

**Figure 9.12-2 — Effect sizes.** *[PENDING: battery-128 — figure]* A forest plot of the five contrasts, each a point estimate in points of recall with its 95% interval and its Holm-adjusted p, against a line at zero:

| Contrast | Arms | Effect | 95% interval | p (Holm) |
|---|---|---:|---:|---:|
| The memory | C1 − A1 | *[PENDING]* | | |
| Reasoning-time reprojection | C1 − B1 | *[PENDING]* | | |
| Continuous against retrieve-once | C1 − D1 | *[PENDING]* | | |
| Reasoning-time reprojection without the seed | C0 − B0 | *[PENDING]* | | |
| The seed files | C1 − C0 | *[PENDING]* | | |

*[PENDING: battery-128]* Every contrast that removes a part of provenance retrieval lowers recall, with an interval clear of zero. The memory is worth __ points; reprojection during reasoning is worth __, with or without the seed; continuous reprojection beats a single retrieval before reasoning by __. The seed files help the opening and change none of the conclusions above.

**Figure 9.12-3 — Question by question.** *[PENDING: battery-128 — figure]* Each question a point: its recall held in thinking (B1) across, with reprojection (C1) up, coloured by category, the diagonal drawn.

*[PENDING: battery-128]* The gain is broad: __ of 128 questions sit above the diagonal and __ below, so the effect is not carried by a few questions.

**Figure 9.12-4 — Where the gain falls.** *[PENDING: battery-128 — figure]* The reasoning-time gain (C1 − B1) with its interval, for questions that name their answer's file and for those that do not — a split fixed in the battery before any run.

*[PENDING: battery-128]* The gain lies on the questions that do not name their answer's file, __ points, and is near zero, __, on those that do. That is the mechanism the design predicts (§6.5): what the question names, the opening finds; what it does not, the model's reasoning reaches, and only a reprojection inside the reasoning places it while the reasoning can use it.

**Figure 9.12-5 — Retrieval as the reasoning proceeds.** *[PENDING: battery-128 — figure]* The share of questions whose answer file is in the working set, against tokens decoded, for C1, B1 and D1 with bootstrap bands, read from each turn's projection records.

*[PENDING: battery-128]* With reprojection the share climbs as the model reasons, from __% at the opening to __% by `</think>`; held and retrieve-once it stays at the opening's __% until the reasoning is over. The reasoning moves retrieval toward the answer.

**Figure 9.12-6 — Recall when the answer file is placed.** *[PENDING: battery-128 — figure]* Recall for answers decoded with their answer file in the working set and without it, pooled over the projecting arms.

*[PENDING: battery-128]* Placement is what the answers turn on: with the answer file placed, recall is __%; without it, __%. Retrieval and the answers are one measurement seen twice, which ties every contrast above to the selection it changed.

**The pilot battery.** The 128 questions extend a 24-question pilot, eight in each category, run on production from 2026-11-10 as the retrieval changed. The first run (2026-11-10, `docs/results/dependency_battery_2026-11-10.md`) and the run after the retrieval work below (2026-11-20, `docs/results/dependency_battery_2026-11-20_b.md`):

| Category | Mean recall, 11-10 | Full marks | Mean recall, 11-20 | Full marks |
|---|---:|---:|---:|---:|
| Direct | 33.3% | 1 of 8 | 47.9% | 2 of 8 |
| Transitive | 54.2% | 2 of 8 | 51.0% | 1 of 8 |
| Architectural | 85.4% | 5 of 8 | 87.5% | 6 of 8 |
| All 24 | 57.6% | 8 | **62.2%** | **9** |

Between them the battery ran nine more times as the retrieval changed, at 46.2–59.7%; with 24 questions one question moves a category by up to 12 points, so it is the direct questions' rise and the ranking measurements below, not the total alone, that carry the result. Seven later runs on 2026-11-23 measure what reprojection during reasoning contributes, three runs per arm on one build; their reprojecting arm averages 57.3% and its best run scores 64.2% with 11 full marks (below, *The pilot reasoning-hold ablation*).

**The scoring formula.** Which files a question's probe should place was measured apart from the model's answers, in a scoring lab (`candle-conversation/examples/provenance_gates`, `lab`): the battery's 23 file-answered questions, each probed once through the daemon — the question prefilled and decoded through its thinking — and its tokens' votes scanned against every one of the 6,854 ingested files on the resident gallery, 39,954,048 signature tokens, then re-ranked on the host under formulas read from a file, so every variant scores the same probes and the same votes (`docs/results/provenance_scoring_lab_2026-11-16.md`). The binding limit it found was votes per token, not normalization: under one vote per token a question's probe gave any vote to only 20–67 files, and seven answer files scored exactly zero, which no normalization can lift. The formula production runs on code reading:

- **Top-k votes.** Each probe token votes for its best 8 files, file `r` by `z_r × (a_r − a_9)` — its z-score times its margin over the ninth — not for its single best.
- **Vote power ½.** Each kept vote is square-rooted before summing, so one spiky match does not outweigh many moderate ones.
- **A needle gate.** Only the quarter of the probe's tokens with the largest summed vote count.
- **Every fold group alike** (`[1, 1, 1]`), and the file's **calibration** questions (§9.11.1) as a second competition at weight 1, one vote a token — a calibration question matches its own file cleanly, and spreading its vote dilutes it.
- **Kind-normalized.** A file scores as a z-score among the files of its own kind on the same question, which replaces a hand-set down-weighting of documents.
- **The lineage out.** The conversation's own inherited documents, placed anyway, are not ranked.

Every non-linear alternative tried — reciprocal-rank and softmax vote laws, token weighting, log or count aggregates, content/calibration blends, token runs — scored below summed `z × margin`; the formula plateaus there. Tool selection keeps one vote per token: a tool probe has one clean match, and spreading its vote raised wrong tools over the selection gate (Top-1 80.0% → 63.7%).

| Probe, on the recalibrated substrate (2026-11-20) | hit@1 | hit@5 | hit@10 | MRR |
|---|---:|---:|---:|---:|
| The question and its decoded thinking | 0.65 | **0.83** | **0.87** | **0.713** |
| The decoded thinking alone | 0.61 | 0.83 | 0.83 | 0.676 |
| The question alone | 0.43 | 0.78 | 0.78 | 0.573 |
| One vote per token, the whole probe (2026-11-18) | 0.43 | 0.57 | 0.61 | 0.508 |

The answer file is in the top five of 6,854 on 19 of 23 questions, and first on 15 (`docs/results/provenance_scoring_lab_2026-11-20.md`). Each is a needle-in-a-haystack search at a scale no context window holds: the answer is one file of 6,854, on average 0.015% of a 40M-token corpus, located by sign agreement between the probe's Q signatures and every stored token's, with no text index, no embedding model and no file name in the query — and the probe is a question plus the first stretch of the model's own reasoning about it, not a keyword. Ranking the answer first two times in three, and in the top five (the top 0.07% of the corpus) more than four times in five, is the selection the bounded working set depends on. The production gallery has grown since: on 2026-11-23 the projection galleries preloaded on BlackWorld held 29,747 turns and 54.7M resident signature tokens, across every layer.

**Figure 9.12-7 — The needle in the haystack.** *[PENDING: battery-128 — figure]* The share of the battery's file-answered questions whose answer file ranks at or above k of every ingested file, for k from 1 to 100 on a log axis, with a bootstrap band, over the line a random ranking draws (k over the file count).

*[PENDING: battery-128]* The curve leaves the random line at its first point: the answer file is first of __ files on __% of questions and in the top five on __%, where a random ranking puts it there __% of the time — a ratio of __ to one.

**What raised the hit rate.** The ranking and the answers are two measurements: the lab ranks, and what the model reads is what the projection places from that ranking. The work between the two runs closed gaps between them, each found by reading a failing question's projection records:

- **Calibrations that saw their file.** A calibration is drafted by a conversation forked on the file's reading; when append-only conversations stopped carrying their lineage in their frame, every draft was asked for its file's questions knowing only the file's name, and 1,227 calibrations were written blind. The frame carries the lineage again, a draft that holds none of its reading is refused, and a reading past the 64,000-token lineage cap — the battery's six largest answer files among them — is drafted over windows of its turns. The whole substrate was recalibrated; the scoring formula's MRR rose 0.663 → 0.713 with nothing in the formula changed.
- **Hits that survive a reprojection.** A reprojection moves the turn onto a rebuilt view; the hit counters stayed with the old one, so every file's hit momentum read zero after the first boundary. They now move with the state.
- **The question's leaders keep their places.** The pool's minimum per kind is filled first by the files the boundary's scan scores highest, so a file lifted only by hits earned on whatever the slot held cannot push out a file the question names.
- **Short strides early in the reasoning.** A reply still reasoning reprojects at 8, 16 and 32 decoded tokens before the 64-token cadence. Measured on one direct question, its answer file entered at token 64 before — after the reasoning had written that it had read nothing holding the symbol — and at token 8 after.
- **A first question that opens its slot.** The first boundary after a new question chooses its rounds whole; a conversation's set, cleared when it opens, ignored the mark, and the first question's slot was filled a third of a budget at a time.
- **A request read twice.** Every conversation starts on configured seed files as background, and the question read against them is a different probe: one direct question scored its answer file second of 6,854 read alone and fifth read against the seeds. A request is now prefilled first without the seeds and scored, that reading discarded, and the turn submitted again — its opening ranked by the first reading's scores beside the seeds, prefilled once more, and decoded (§6.5).

What remains is placement, not ranking: three questions still declined with files they do not concern in context, and the first reading's own score of an answer file still moves with what the question is read against.

**The pilot reasoning-hold ablation.** Before the 128-question battery, the pilot measured the contrast at the centre of the claim — whether the working set must follow the model *while it reasons* — with a daemon-wide setting changed between runs. `reproject_in_thinking: false` (`scheduler/think_gate.rs`) passes over every boundary inside the think block and keeps everything else: the turn's opening projection from the question read twice, every boundary through the answer, and one reprojection on the committed `</think>`, so the answer is decoded against a selection informed by the whole thought. The held arm therefore differs from the full system in one thing only: during its reasoning, the model reads what the question alone selected.

Seven runs on BlackWorld at `e6ebbb188`, each after a restart with the normalization replay drained, the last two, (q) and (r), on a store with every earlier conversation archived or deleted first (`docs/results/dependency_battery_2026-11-23_l_m.md`). The probe's reach was shortened from 256 decoded tokens to 128 for the same runs; (l), at 256 on the build before, is the control for that change.

| Run | Probe | Think block | Direct | Transitive | Architectural | All 24 | Full marks |
|---|---:|---|---:|---:|---:|---:|---:|
| (l) | 256 | reprojects | 25.0% | 56.3% | 66.7% | 49.3% | 7 |
| (n) | 128 | reprojects | 26.0% | 64.6% | 70.8% | 53.8% | 7 |
| (p) | 128 | reprojects | 27.1% | 69.8% | 95.8% | 64.2% | 11 |
| (q) | 128 | reprojects | 25.0% | 67.7% | 68.8% | 53.8% | 7 |
| (m) | 128 | held | 20.8% | 51.0% | 47.9% | 39.9% | 4 |
| (o) | 128 | held | 14.6% | 50.0% | 47.9% | 37.5% | 6 |
| (r) | 128 | held | 18.8% | 39.6% | 58.3% | 38.9% | 5 |
| **Reprojecting, mean of (n)(p)(q)** | 128 | reprojects | 26.0% | 67.4% | 78.5% | **57.3%** | 8.3 |
| **Held, mean of (m)(o)(r)** | 128 | held | 18.1% | 46.9% | 51.4% | **38.8%** | 5.0 |

- **Reprojecting during reasoning is worth 18.5 points of recall.** With every other reprojection in place, holding the think block on its opening projection takes the battery from 57.3% to 38.8%.
- **The arms do not overlap.** Every held run scores below every reprojecting run: under no effect, the three held runs take the three lowest places of six with probability 1/20 (one-sided permutation, p = 0.05). No run that reprojects in thinking — every battery run since 2026-11-10, 46.2% at the lowest — has scored as low as any held run.
- **Question by question,** summing each question's recall over the three runs of each arm, the reprojecting arm scores higher on 14 questions, the held arm on 4, with 6 tied (sign test, p = 0.031 two-sided).
- **The probe's reach is not the cause.** At 128 tokens the reprojecting arm averages 57.3%, above (l)'s 49.3% at 256, and production runs at 128.

The loss has a shape. A held answer declines rather than errs: it says it has not read the file, and the files in its context are the ones the question selected. It is largest where the answer lives in a file the question does not name — the glue-island cache (t7, 2.67 against 0.33 summed over three runs), the reprojection configuration (t6, 2.0 against 0), the recurrent model's refusal of live glue and the un-rotated key store (a3, a4) — and absent where the question names what it needs (a1, a5 level). That is the mechanism the design predicts (§6.5): reasoning about a question moves the model's Q vectors toward the files that answer it, a reprojection inside the reasoning places those files while the reasoning can still use them, and one at `</think>` places them after the thought that needed them is over. The one repetition loop in the seven runs came from a held run.

The held arm still re-selects at `</think>` and through the answer, so it is an upper bound on what retrieval before reasoning alone would score; the battery's retrieve-once arm (D1) measures that score itself.

What the memory holds is the documentation and the module structure more than the call graph: the architectural answers are in `AGENTS.md` and the design documents, and five of eight came back whole; the direct questions ask for call sites, which documents rarely name, and one did. The answers cite where they came from — a design document's section, a module header — and when the model had not seen the source it said so: five declined outright, more named what they knew and declined the rest, and one withdrew identifiers it had just named because it had not read the file that would confirm them. The errors found on reading are near-misses (the CUDA constant placed in the arena-table header beside the one that defines it; an out-of-date caller), not invented names.

**What is public.** The source and the raw measurement rows are public (Appendix B); every throughput, compression and retrieval figure in this paper is reproducible from a named test in the repository and a named row in `docs/performance.md`. The live system is open to signed-in users, but it does not expose a retrieval log, and the ingested conversation is not released; the battery, its ground truth, its analysis and its per-question scoring are in the repository (`zend/examples/dependency_battery/`, the pilot's `docs/results/dependency_battery_2026-11-10.md`, `docs/results/dependency_battery_2026-11-20_b.md` and `docs/results/dependency_battery_2026-11-23_l_m.md`, and the 128-question battery's *[PENDING: battery-128 — results file]*), and so is the scoring lab with its formulas and measurements (`candle-conversation/examples/provenance_gates`, `docs/results/provenance_scoring_lab_2026-11-16.md`, `docs/results/provenance_scoring_lab_2026-11-20.md`).

---

### 9.13 The System as Its Own Workload

Everything above is measured on a system that is also in use, and the use is itself the most interesting evidence, because it exercises the parts no benchmark can: memory that outlives the process that wrote it, and memory written by one model and read by another.

**The substrate is the estate's memory, not one session's.** The journal store is a separate repository — one master journal per group, `people/`, `conversations/`, `topics/`, `threads/`, `bots/` — whose entries are minute-stamped YAML records written by whichever agent is speaking: `claude-code`, `claude.ai`, Grok, this engine's own workers, and the game's NPC bots. A note written by one agent at 00:39 is read by another the next morning, and the note's `by:` and `account:` fields are the record of who said it. Memory is therefore portable across models in the strongest sense that matters operationally: the memory is not in any model's weights, nor in a model-specific embedding space, and swapping the model that reads it does not orphan it.

**Folding is what keeps it readable.** The journal entries fold the way the KV cache folds: a day's steps fold into the day, a week's days into the week, up to years, with the folded entry replacing its children in place and carrying a one-line summary plus a digest kept behind the line. The same discipline that keeps a 128K-token working set bounded keeps a years-long journal readable in one prompt, and the same single-flight discipline that keeps a fold and a draft from racing in the KV cache keeps them from racing in the journal.

**Retrieval is scoped, not global.** Each group's journal declares what its members and its bots may read, and there is a deliberately public slice — a group whose entries restate only what the public site already publishes, each line naming its public source. Which groups a reader may see is decided by those declarations, not by retrieval: provenance ranks what a reader is allowed to see, and the access question is asked once, at the boundary, rather than assumed.

**The measurement follows the deployment.** The retrieval numbers in §9.11.1 are not from a corpus assembled for the paper; they are from the gallery the running daemon holds — 2,206 files, ~25.8M signature tokens — and the gates run against it over HTTP while it is being written to. The engine probe (§9.11) drives admission, projection, the persistence thread, KV compaction and the three tiers together, which is the shape of the production workload rather than the shape of a benchmark.

**What this buys.** Sessions that survive restarts; a model swap that costs a re-ingest rather than a memory transplant; a body of work — decisions, errata, design reversals — that accumulates across agents and months; and an audit trail in which every entry is attributable to a named account at a named minute. None of that is a throughput number or a measured quality result, and none of it is claimed as one; it is what the deployment is for, reported as operating experience.

#### 9.13.1 This revision, as a workload

Much of this version was revised by the system itself. The author asked a Qwen3.8-Flash-Next session on the production daemon for a critique of the v2 draft, read it, and then asked it to revise: fix what the critique found, bring the figures to the current sweeps, add the live-operation results, and drop the disclaimer that the system was still being engineered. The session ran 85 tool rounds over ~320K tokens of conversation, made thirty-one edits to this paper and brought the README's figures with it (`docs/results/paper_revision_session_2026-11-09.md`).

**How it did.** The edits are the ones this version carries — §9.6's current figures, §9.7's live-dispatch ablation, §9.11.1, §9.13 itself, and the removal of most of the sentences that claimed a result for the then-unrun battery of §9.12 — and the quality of the judgement in them is the finding worth recording. Asked to fill the draft's gaps, among them §9.12's empty results table, it searched the repository for a measured table, found none, and wrote that the battery is not run rather than supplying numbers. It then went looking for every other place the paper claimed that battery's results — the abstract, §1, §11.2, §11.4, the conclusion, Appendix B's list of what is released — and corrected them. It did not find them all: a later review by the author found such claims still standing in §1.2, §6.5, §9.1, §9.11 and §11.5, and an attribution of Flash-Next's flat depth to provenance alone, which this subsection's first draft also made; those are corrected in this version. It caught its own citation error (reference numbers from `docs/performance.md`'s own list written into a table that cites the paper's), and it softened its own new status banner when it noticed that the banner said more than the evidence it had read could support. It kept the engine probe's failing rows in §9.11's table rather than leaving them out. Where it failed, the failures were mechanical: anchors gone stale after neighbouring edits, and one edit reported as failed that had in fact applied, together costing a dozen rounds. That is a defect in the editing tools, and it was the tools, not the model, that lost the time.

**The ratio, and what explains it.** The author's own reading of the session was that it decoded unusually much for the amount it read, and that provenance selection is the reason. The round log bears out the first half. Across the session one token was decoded for every 2.9 prefilled — 82,963 decoded against 238,619 prefilled — and after the paper was read whole once, in the first four rounds, each round's new input averaged ~2,100 tokens against ~970 decoded. The model edited thirty-odd parts of a 1,400-line document across eighty rounds without reading it whole again; its later reads were of single parts, the one it was about to replace.

Three mechanisms are in play, and the session separates only the first of them cleanly:

- **Every token is prefilled once.** The 238,619 tokens prefilled are the session's new input and nothing more: the earlier turns' K/V is held by the substrate and projected back, never prefilled again, though the conversation grew to ~320K tokens. A client that re-sends its history each round would have sent on the order of eighty times as many input tokens over the same session.
- **Decode does not slow as the conversation grows.** Decode ran at ~102 t/s while the conversation was 48K–130K tokens, ~110 t/s at 162K–237K and ~106 t/s at 309K–321K — the working set does not grow with the history it is selected from, so neither does the cost of a decoded token. Two bounds are at work and the session does not separate them: provenance caps how much of the conversation is projected into the working set, and Flash-Next's own sparse-attention indexer caps how much of the working set each token attends (§9.6.2). Either alone would flatten decode across this range; the session is evidence for the bounded working set of §11.2 in live use, not for provenance's share of it. (One stretch, rounds 47–57, dips to ~74 t/s and recovers; the longer rounds after it run at full rate, so the dip is not the conversation's length.)
- **The model reads less because what it read comes back.** Earlier reads retrieved into the working set when they become relevant would let the model edit a part it read sixty rounds before without reading it again — which is what it did. But the file tools address a part by its anchor, which also lets a model edit without reading the file around it, and the session ran no control. The author's attribution of the ratio to provenance is the right hypothesis, and this session is consistent with it; the run that would establish it is the same task with provenance selection replaced by a full window, and with the file tools reading whole files.

**Against the published record.** The largest published picture of agent traffic is OpenRouter's: in September 2026, cached prompt tokens were just under 90% of all the tokens agents used, novel prompt tokens most of the rest, and output plus reasoning around 1% — its worked example is 87% cached, 12% novel prompt, 1% output — against around 12% output for human traffic, because "an agent re-sends its whole context on every step" [OpenRouter, *Cache is Crucial*, October 2026; 50]. A study of eight models running SWE-bench Verified under OpenHands finds the same shape: input tokens, not output, dominate the cost of agentic coding even with caching, and an agentic task consumes on the order of a thousand times the tokens of a single-round one [Bai et al., arXiv:2604.22750; 51]. The comparable figure from this session is decoded tokens against genuinely new input, since the substrate holds everything earlier: **1 : 2.9 here, against about 1 : 12 for output against novel prompt in OpenRouter's agentic mix** — roughly four times as much written for each token newly read. Two things keep that a comparison rather than a result. OpenRouter's novel prompt tokens include re-reads a provider failed to cache, which makes the average agent's denominator larger than its genuinely new input. And counted the way a re-sending client counts it — the whole history at every round, ~19M tokens — this session's ratio would be about 1 : 230, more input-heavy than OpenRouter's average of about 1 : 99, because a 320K-token session would re-read far more at each step than the average agent does. What the architecture changes is not how much a long session would have to re-read, but that it does not.

**What else it showed.** A restart during the session resumed its open tool round on a second member while the first was still running it, and the two members wrote the conversation at once; the second's duplicate rounds reverted one of the first's edits. Nothing was lost: the committed files were rebuilt edit by edit from the session's own record, and the fault — a resumed turn that took no claim on its conversation — is fixed. It is a small instance of §9.13's point: the record outlived both processes that wrote it.

## 10. Analysis

### 10.1 Why Coherence Produces Disproportionate Results

Each component of this system has research precedents. What does not exist anywhere in the published literature is their combination in a single coherent design.

The expert prediction system benefits from the wave-batched kernel — more concurrent requests means better cross-request expert coalescing, which means the effective hot set is smaller, which means the Markov predictor's free-slot-only prefetch converts more misses to hits. The KV compression benefits from the prefill refresh — quantising clean activations rather than decode-drifted ones means the error thresholds in the selection kernel are calibrated to actual quality, not a noisy lower bound, and specifically keeps the hot-tier error floor ε_hot ≈ 0, which the theorem establishes as the asymptotic quality limit of the entire system. The provenance indexing benefits from the turn-boundary paging — every 32-token block begins at a semantic boundary, making fingerprint extraction temporally coherent. The unbounded context tier system benefits from the adaptive quantisation — V blocks systematically landing at Q2_0/Q3_0 means 3× more tokens fit in the warm RAM tier before cold storage is needed, and the theorem bounds what this aggressive compression costs: however deep the cold tier grows, no more than the fixed retrieval budget of its blocks enters any one step, each within its per-block threshold.
Most critically: the provenance selection mechanism and the KV quantization system interact to produce a property neither achieves alone. Provenance selection bounds the working set W independently of N. KV quantization with prefill refresh keeps the hot-tier error floor near zero. Together they give the bound of §11.2 — per-step error exposure independent of depth — which no KV quantization system in the literature states, because none we are aware of decouples context depth from working set size. Each component was designed knowing the properties of every other; we credit that, rather than any single component, for the results on 16GB.

### 10.2 The VRAM Budget

*This is the v1 design budget for Qwen3-30B-A3B at Q4_K_M, kept for the trade-off it illustrates. The production engine does not use fixed shares: it reserves one contiguous span per card and partitions it between persistent tensors, KV regions, a per-wave transient tier and expert weights, with an elastic boundary between the KV side and the expert side that each side re-checks before crossing; §9.11's engine probe measures how fully the KV side of that span is used.*

On a 16GB card running Qwen3-30B-A3B at Q4_K_M:

```
Expert cache (44% residency):   8.0 GB  (2,700 slots × 3MB)
Attention layers (all 48):      2.0 GB  (non-expert transformer blocks)
Adaptive KV hot tier (balanced): 4.0 GB  (~125K tokens at CR ~3×)
Working buffers + overhead:     2.0 GB  (activations, router, CUDA contexts)
──────────────────────────────────────────────────────────────────
Total:                          ~16 GB
```

The hot tier token capacity is computed from Qwen3-30B-A3B's architecture: 48 layers × 4 KV heads × 128 head_dim × 2 (K+V) × 2 bytes (F16) = 96 KB per token at full precision. At balanced-tier compression (CR ~3×), the effective cost is 96 KB / 3 ≈ 32 KB per token. The non-KV allocation (expert cache + attention layers + working buffers) is fixed at ~12 GB regardless of card. The remaining VRAM is fully available to the hot tier:

**Concurrency vs. working set per session (balanced tier, CR ~3×, Qwen3-30B-A3B)**

| Sessions | 16GB — tokens/session | 24GB — tokens/session |
|---|---|---|
| 4 | ~31,300 | ~93,800 |
| 8 | ~15,600 | ~46,900 |
| 16 | ~7,800 | ~23,400 |
| 32 | ~3,900 | ~11,700 |
| 64 | ~1,950 | ~5,860 |

The working set per session is the quality ceiling for each generation step: the provenance selection system must choose that many tokens from an unbounded history as the most contextually relevant, and a larger working set gives it more room. Because the non-KV share is fixed, a 24GB card triples the hot tier: it serves three times the sessions at the same per-session working set (×12 there against ×4 on 16GB, ~31K tokens each), or the same sessions with three times the working set. Additional VRAM buys concurrency or working-set size, at the operator's discretion.

---

## 11. Discussion

### 11.1 The Baseline Problem: Error Growth in Full-Attention Systems

The KV quantization literature — KIVI, KVQuant, TurboQuant, GEAR — is conducted under a full-attention model where every token in the context window contributes to every generation step. Two mechanisms connect stored error to context depth. The first is sequential: during autoregressive decode, each token's KV values are computed with attention over already-compressed preceding KV, so each step's rounding and quantization error enters the residual stream and is inherited by later steps; GEAR [Kang et al., 2024] characterised its multiplicative compounding. The second is exposure: every one of the N cached tokens enters every step, so the number of stored values carrying error that a step reads grows with N, and so does the number of perturbed attention scores competing with the relevant tokens for mass.

Exposure is not output error. Softmax attention's output is a convex combination of its values, so per-token value errors are averaged by the attention weights rather than summed, and a perturbation of a low-weight token's key moves little mass. Whether full-attention output error actually grows with N therefore depends on the attention distribution — on how much mass the step spreads, and on how many distractors sit near the relevant tokens in score — and we do not prove that it does. Measured directly, it does not grow within the windows we can reach (§9.6.8): teacher-forced on the same answer, C10's next-token divergence from BF16 on Qwen3.5-9B, whose full-attention layers read the whole prefix, is 6.5e-3 nats at 4K and 3.5e-3 at 124K, and on Qwen3-8B it moves between 1.1e-2 and 3.9e-2 from 4K to 32K with no trend; and every model of §9.6 whose attention layers read the whole prefix passes the story gate at C10 at every depth it was run to. What the literature has not offered is a bound on exposure that is independent of depth, and that is what §11.2 supplies. Neither mechanism is specific to compression: F16 rounding is stored error too.

More VRAM defers the threshold but does not eliminate it. An H100 at 80GB holds more tokens at full precision before compression is required, but as sessions grow persistent across days and weeks, the context is not 1 million tokens — it is 100 million tokens, and no finite hardware holds that in F16. The industry response at that scale is summarization and eviction: older context is compressed into summaries or discarded. Both are lossy by definition. Summarization destroys the precise factual recall — verbatim quotes, specific numbers, exact commitments — that makes persistent memory useful for real applications. Eviction discards it entirely. Hardware scaling moves the wall; it does not remove it.

The theorem in §11.2 shows that leaving this regime does not need more hardware or better compression: architectural decoupling of working set size from context depth is sufficient. No full-attention system has that property — its working set is by definition the entire context. Provenance-selected attention has it structurally, and the bound follows directly.

### 11.2 Theorem: Asymptotic Numerical Stability Under Provenance-Selected Attention

Every published KV quantization analysis — GEAR [Kang et al., 2024], KIVI [Liu et al., 2024], KVQuant [Hooper et al., 2024], TurboQuant [Zandieh et al., 2026] — is conducted under a full-attention model where all N tokens contribute to every generation step, so the error a step is exposed to grows with N. That premise is correct for standard attention. It does not hold for provenance-selected attention.

We state the following result formally. ε(t) denotes the **total stored numerical error** of token t's K/V from all sources — quantization, floating-point rounding, decode drift — not compression error alone. The quantity bounded is the sum of ε(t) over the tokens that enter a step: an upper bound on that step's exposure, not its output error. The proof does not use any property specific to compression; it applies to any finite-precision arithmetic.

---

**Theorem (Asymptotic Numerical Stability).** *Let N denote context depth, B the fixed provenance selection token budget (a system constant), W_hot_max the hot tier capacity, W_warm_max = B − W_hot_max the retrieval budget — the tokens selected per step from outside the hot tier, whether they reside in the warm or the cold tier — and ε(t) the total accumulated numerical error of token t under any finite-precision arithmetic. Let ε_warm and ε_cold bound the per-token error of a warm-tier and a cold-tier token, and ε_ret = max(ε_warm, ε_cold). Under attentional provenance selection, the expected total numerical error contribution per generation step is bounded by a constant independent of N:*

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{ret}} = O(1)$$

*Under full attention the same sum is Θ(N · ε̄), with ε̄ the mean per-token error over the context, and grows without limit. The bound's independence of N holds at any hardware scale and in any precision format; the format sets the constant, through ε_hot and ε_ret.*

---

**Proof.** Partition the context into three tiers as N grows.

*Hot tier* (H): tokens always present in the working set. |H| ≤ W_hot_max is bounded by a hardware constant independent of N. Tokens in H are prefill-refreshed: their decode drift is zero and their quantization error is bounded by the selection kernel threshold. ε_hot is a small constant independent of N.

*Warm tier* (W_warm): tokens eligible for retrieval but not permanently resident. The provenance selection mechanism operates under the fixed budget B — at most W_warm_max tokens from outside the hot tier can be selected per generation step, from the warm and cold tiers together, regardless of how large either grows with N. This bound is structural: the selection kernel enforces it by construction. Let k_warm and k_cold be the numbers of warm- and cold-tier tokens a step selects; the budget enforces k_warm + k_cold ≤ W_warm_max. The warm tier's aggregate error contribution per step is therefore at most:

$$\sum_{t \in W_{\text{warm}} \cap \mathcal{W}} \varepsilon(t) \leq k_{\text{warm}} \cdot \varepsilon_{\text{warm}} \leq W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} = O(1)$$

a constant independent of N. As N grows and the warm corpus expands, more candidates compete for W_warm_max slots, but the aggregate contribution is bounded by the budget.

*Cold tier* (C): all remaining tokens. As N → ∞, |C| → ∞ while the retrieval budget W_warm_max stays fixed. Summed over the cold tier, the selection probabilities are bounded by the budget:

$$\sum_{t \in C} p_{\text{cold}}(t, N) = E[k_{\text{cold}}] \leq W_{\text{warm\_max}}$$

so the *mean* per-token selection probability, at most W_warm_max / |C|, tends to zero. The mean does not bound any specific token. Selection is relevance-driven: a cold token relevant to the current step is selected with high probability at any depth — that is the purpose of retrieval — and it then contributes its full per-token error. Depth dilutes the cold tier's selection probability only on average; it does not dilute the error of what is selected.

**Scope of the bound.** The theorem bounds numerical error contributions from stored token K/V. On the attention-only models it covers the full system: interior tokens are bounded directly by the proof; boundary tokens carry no stored frame, because their K/V is constructed against the current assembly (§5), so there is no frame-mismatch error class to bound. On the recurrent production lineage, boundary tokens are sealed K/V like interior ones (§5) and fall under the same per-token bound; what the theorem does not cover there is frame mismatch in a boundary sealed against a canonical prefix, which is a different error class from numerical error.

The cold tier's aggregate error contribution per step is bounded by the tokens it can take from the shared budget, not by depth:

$$\sum_{t \in C \cap \mathcal{W}} \varepsilon(t) \leq k_{\text{cold}} \cdot \varepsilon_{\text{cold}}$$

It does not vanish as N grows. A bound that decays as O(1/N) would follow only if every cold token were equally likely to be selected — the opposite of relevance-driven selection.

Summing across tiers, with k_warm + k_cold ≤ W_warm_max:

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \leq \varepsilon_{\text{hot}} + E[k_{\text{warm}}] \cdot \varepsilon_{\text{warm}} + E[k_{\text{cold}}] \cdot \varepsilon_{\text{cold}} \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{ret}} = O(1)$$

□

---

**Corollary 1 (Near-zero convergence).** *In this system's design, warm- and cold-tier blocks alike originate from prefill-refreshed hot-tier blocks — they were quantized from clean activations and assigned a format by the adaptive per-block selection kernel at threshold θ, θ being the loosest threshold any retrievable block was sealed under. Consequently ε_ret ≤ θ, which is a small system parameter. Then C_ret = W_warm_max · ε_ret is small, and:*

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \approx \varepsilon_{\text{hot}} + C_{\text{ret}} \approx 0$$

*The total error is not merely bounded — it is bounded by a small constant determined by the selection kernel's quality threshold, approaching zero in the limit of tight thresholds. Empirically: always-attended blocks operating in top-quality compression mode (C0) achieve K_SNR 58.6 dB and V_SNR 58.8 dB (per-block ideal), confirming ε_hot ≈ 0 for the working set.*

---

**Corollary 2 (Hardware Independence).** *The bound is independent of VRAM capacity. Under standard full attention, an H100 at 80GB holds more tokens at full precision, but once context depth N exceeds that capacity and tokens are compressed or evicted, its exposure grows with N exactly as a smaller card's does — the threshold is deferred by a constant factor, not removed. The theorem's bound applies identically to any hardware configuration running this architecture, because B and W_hot_max are system constants and the proof depends only on these being independent of N. More hardware does not give a full-attention system a depth-independent bound. Note: for deployments where context never grows beyond full-precision VRAM capacity, both architectures operate in bounded error regimes and this corollary's practical implication is nil — the distinction matters for persistent sessions that genuinely exceed hardware capacity over time.*

---

**Contrast with the standard result.** Under full attention with any finite-precision arithmetic — F16, BF16, or uniform compression — all N tokens enter every generation step and the exposure this proof bounds is Θ(N · ε̄), unbounded. Under provenance-selected attention it is bounded regardless of N: the system at N = 10,000,000 turns operates under the same bound as at N = 1,000 turns. The contrast is between bounds. It is not a demonstration that full-attention output degrades: §11.1 gives the reasons it may not, and §9.6.8 measures that within 128K it does not. What full attention lacks is the bound, not the behaviour at the depths a single context can reach; how its output behaves past them, where every token held must be compressed or evicted, is the regime the bound is about and that no single-context measurement reaches.

**Empirical status.** The theorem is a proof, and its premises — a fixed selection budget, a hot tier of fixed size, a per-block threshold on everything sealed — are properties of the engine that hold by construction. What the measurements show is consistent with it, not a test of it. The depth measurement of §9.6.2 shows the operating signature of a bounded working set — Qwen3.8-Flash-Next keeps 99% of its prefill and 111% of its decode from 32K to 128K — but the bound there is the model's own trained indexer inside one context, not provenance, so it supports the principle, not this mechanism. The live session of §9.13.1 shows decode flat from 48K to 321K tokens of conversation with provenance in the loop, with the indexer as a confound. The story evaluation (§9.8) shows every throughput row reproducing its reference text at the compression levels in use, and on Llama-3.2-3B a progression of failure past the validation edge consistent with error growing per token. The direct test of the premise — the model fixed, N far past the working set, provenance on and off — is §9.11.2: with the engine's slot capped at 32K and a history twelve times that, mid-decode reprojection selected every planted turn into a working set of 15–18 turns, at C10 and C0 alike, where the one-shot and rolling-window arms selected none. The dependency battery of §9.12 tests what the model does with the selected set, at the level of compositional reasoning; its results are given there, and the theorem does not rest on them.

**Implication for compression ratio selection.** The theorem bounds how many cold-storage tokens can enter a step, not how much error each carries: a selected cold block contributes its full per-token error at any depth. Compression aggressiveness on cold storage therefore moves ε_cold, and with it the constant — it never threatens the O(1) bound, but it is not free. The adaptive selection kernel's highest-compression modes (C7–C9) are assigned to V cache blocks and cold-storage K blocks as a capacity trade-off with a bounded quality cost: every block still passes the per-block threshold of the level it was sealed under, which is what bounds ε_cold. The binding design constraints are the always-attended working set error floor ε_hot — confirmed near zero by the prefill refresh strategy — and the retrieval threshold θ, which bounds ε_ret for warm and cold blocks alike. The design is an engineering trade-off the theorem makes explicit: capacity against the size of the constant, never against the bound itself.

**Key assumption.** The selection budget B and hot tier bound W_hot_max must be independent of N. This is structurally guaranteed by the system architecture. The bound fails if the provenance selection mechanism grows the working set proportionally with N, which recovers the O(N) full-attention regime.

### 11.3 Compression Architecture vs. Compression Primitive

The competitive field for online, inference-time, training-free KV cache quantization is KIVI, KVQuant, and the residual PolarQuant-MSE portion of TurboQuant. All three optimise the same thing: the compression *primitive* — a better quantization formula, a smarter rotation, a tighter codebook. The design question they answer is: given that we must assign a format to every block, what is the best format to assign? KIVI assigns uniform 2-bit. KVQuant adds outlier handling before assigning a fixed format. TurboQuant rotates before assigning.

This paper optimises the compression *architecture* — the decisions that determine when to quantize, what quality to guarantee per block, how to separate error sources, and how error scales with context depth. The design question is different: given that blocks have different difficulty, how do we build a system that guarantees quality on every block and separates the error mechanisms that cause quality loss? The answer — adaptive per-block format selection coupled with two-phase prefill refresh — does not depend on having the best quantization primitive. It absorbs primitive-level failures through selection: a block that Q4_0 cannot represent adequately gets Q8_0 or better. This is why TurboQuant's rotation can fail at 4–5× CR on real workloads while the adaptive system continues to hold quality — the architecture compensates where the primitive falls short.

The specific gaps in the competitive field: none of the three systems perform per-block format selection with a quality guarantee. None couple quantization with decode drift correction. None establish any theoretical result about error scaling with context depth. The system exceeds KIVI's CR at balanced-tier operating points with per-block validated quality guarantees KIVI cannot offer. KVQuant's outlier handling insight is subsumed — the _KS sub-block formats and attention sink protection factor implement the same protection, integrated into adaptive selection rather than applied uniformly. TurboQuant's theoretical Shannon-optimal primitive is irrelevant when the architecture guarantees per-block quality regardless of which primitive is used.

The combination is what the application requires. None of it is achievable through better primitives alone.

**Empirical validation.** The story rewrite evaluation (§9.8) directly confirms this. On Llama-3.2-3B, adaptive C5 at 3.27× passes the multi-session identity discrimination test while uniform Q4_0 at 3.17× fails — the adaptive architecture delivers better quality at higher compression than the best uniform 4-bit primitive. The query-aware K selection metric eliminates the cross-session identity contamination that an isotropic metric produced at C8–C9 by automatically assigning conservative formats to the high-query-relevance blocks that carry session-discriminating signal. This is architecture compensating where a uniform primitive cannot.

### 11.4 Attention Was Always Retrieval

The transformer attention mechanism is a retrieval system: Q vectors query against K vectors, scores select candidates, V values are aggregated weighted by relevance. That is what softmax attention computes. That is all it has ever computed. The architecture is a learned retrieval system operating over a fixed context window.

This system performs the same operation at two scales. The provenance index does approximate attention over the full unbounded context — a binary sign-agreement scan over every stored token's Q signature (§6.6) — selecting which blocks enter the working set. The attention kernel then performs exact attention over the selected working set. Coarse retrieval selecting for fine retrieval. The only structural difference from standard attention is granularity: standard attention performs flat retrieval over all N tokens at O(N) cost; this system performs hierarchical retrieval — approximate 1-bit selection, then exact attention over what it selects — at O(1) attention cost per generation step.

When a reviewer characterises this as "RAG with extra steps," the correct response is: RAG replaces attention with an external retrieval system that has no attentional continuity — content retrieved by BM25 or embedding similarity is injected into a fresh context, severing the causal chain that attention depends on. This system extends the retrieval that attention already is to unbounded depth, without severing anything. The mechanism is preserved; the scaling constraint is removed. This comes with a different binding constraint that full attention does not have: retrieval quality. Full attention over N tokens attends to all N exactly. This system selects B tokens from N via approximate provenance matching — if the provenance system surfaces the wrong B tokens, the generation step lacks the correct context regardless of how precisely those B tokens are attended. The guarantee is bounded numerical error on the attended set, not guaranteed relevance of the attended set. This is a meaningful distinction: the system trades exposure that grows with N for a retrieval quality problem that the provenance mechanism is designed to bound but cannot eliminate. RAG abandons the attention mechanism entirely and substitutes something categorically weaker. The distinction between this system and RAG is not one of implementation style — it is a difference in what property is preserved; the distinction between this system and full attention is a different trade-off, not a strict dominance.

The gap is structural, not implementational. Three deficits of RAG are architectural consequences of what RAG is, not limitations of any particular implementation.

*Wrong retrieval signal.* RAG retrieves from token-level representations — embeddings or BM25 over generated text. The model's Q vectors encode a pre-linguistic representation of what the model is reaching for: the attention state that precedes and drives token output. During a transitive dependency walk, the model's Q vectors at hop 2 may already point toward the region of the index that holds hop 3's context before the model has produced any tokens that would form a textual retrieval query — the hypothesis §9.12 is designed to test. RAG cannot access this signal because it operates on tokens, not attention states. This is why iterative RAG systems (FLARE, Self-RAG) must wait for the model to produce enough text to form a query — their retrieval trails reasoning rather than leading it.

*Severed causal continuity.* When RAG injects retrieved content, that content arrives without KV cache history — the model cannot attend to it in the causal context of the conversation where it was originally produced. This system promotes KV blocks from warm and cold tiers with their original quantized attention state causally intact. The model attends to retrieved content as a continuous causal subset of the full context, not a collage of fragments injected into a fresh window. The retrieved block's K and V were computed under its original causal context; K is stored un-rotated and given its position in the assembled sequence when it is read. This is the property that preserves attentional continuity as context grows — and it is the property RAG cannot replicate, because RAG's retrieved documents never had a causal position in the model's context to begin with.

*Retrieval trails reasoning.* Even iterative RAG systems retrieve based on text generated so far — the retrieval query is a function of committed output tokens. Mid-decode reprojection (§6.5) also retrieves from what has been decoded, so in the engine as built retrieval trails reasoning too — by at most one reprojection interval, and on the model's attention state rather than its text. Speculative Context Decode, sketched in §6.5 and not built, would remove the lag: a probe session decoding ahead would assemble the context for line N+1 of thinking while line N is still being decoded, so retrieval would lead reasoning. Text-based retrieval cannot do that, because its signal does not exist until after the reasoning step that needed the context has completed.

**Thought experiment.** Imagine augmenting a RAG system with three changes: (a) store Q-vector fingerprints alongside documents and use Q→Q matching for retrieval; (b) maintain causal KV continuity for retrieved blocks so the model attends to them in their original causal context; (c) run retrieval repeatedly during generation from the model's Q vectors rather than from its text. A system with all three properties would match attentional provenance indexing with mid-decode reprojection. It would also no longer be RAG — it would be this architecture. The properties that close RAG's deficits are the properties that define a different system: operating on attention states rather than tokens, and injecting with causal continuity.

The provenance index's use of Q vectors as cognitive-state fingerprints is the specific choice that preserves attentional continuity: a stored Q fingerprint from a prior turn captures the accumulated attentional context of everything preceding it, not just the surface semantics of that turn. Matching a current Q against stored Q fingerprints selects turns that were attended from a similar cognitive state — the attentional equivalent of relevance, not keyword or embedding overlap. This is what makes the system's retrieval genuinely continuous with the attention mechanism rather than a separate system bolted alongside it.

The codebase dependency analysis evaluation (§9.12) is the strongest demonstration this architecture is built to support. The mechanism is real and runs: every 64 decoded tokens, or at a line break, reprojection (§6.5) scores the signatures of the model's last 128 decoded tokens against the provenance index and assembles the next working set. The scan finds its needle: the file that answers a battery question ranks first of 6,854 — 40M signature tokens — on 65% of questions and in the top five on 83%. And the reasoning is what aims it: with the working set held through the think block, the battery's recall falls from 57.3% to 38.8%, every held run below every reprojecting one, and the loss is largest where the answer lives in a file the question never names. The model does not know it is performing graph traversal; it is reasoning, and its Q vectors carry the reasoning to the files it needs. The scan is approximate attention over the full unbounded context, the forward exact attention over the selected working set — two scales of the same operation, alternating during decode. *[PENDING: battery-128]* The 128-question battery confirms it at scale: a single retrieval before reasoning scores __ points below continuous reprojection, and the share of answers with their file in the working set climbs through the reasoning only when the reasoning reprojects.

### 11.5 Constraint-Driven Innovation as a Methodology

This system was not designed top-down from theoretical considerations. It was designed bottom-up from two constraints accepted simultaneously and without compromise: an application requiring the hardest possible form of persistent memory, and a fixed hardware limit that could not be exceeded. The application — persistent agent conversations — was selected as the existence proof of a real and demanding use case. Agents require verbatim factual recall, semantic coherence across arbitrary time gaps, high concurrency, and deployment on hardware without a datacenter budget. The constraints were not problems to be worked around. They were accepted as design drivers.

The methodological pattern that emerged from this acceptance is worth documenting, because it appears repeatedly across the system's components, and because it produces a specific type of innovation: decisions forced by constraint that turn out to hold beyond the constrained platform, not just to be viable on it.

**Native quantized matmul kernels.** Standard GEMM libraries dequantise weights to BF16 before computation. On 16GB with a 30B model, this immediately OOMs — the constraint made the standard approach impossible. The forced alternative, inline dequantisation within the MMA kernel, never materialises the full-precision weight copy, and so also saves that copy's memory traffic on unconstrained hardware (§8.1). The 16GB ceiling exposed a cost that exists everywhere.

**Provenance-selected sparse attention.** VRAM is finite. The full context of a persistent session cannot be held hot. The constraint forced a mechanism for selecting a bounded working set from an unbounded context — which turned out to be the architectural property the Asymptotic Numerical Stability theorem uses. The hardware-bounded hot tier is also the hardware-independent theorem's working set W. The constraint that made full-context attention impossible produced the architectural decoupling that makes unbounded-context correctness possible.

**Two-phase quantization.** Materialising all active turns at F16 across 200+ concurrent sessions OOMs at scale. The forced alternative — chunk-sealed quantization during decode, prefill refresh on turn completion — turned out to correctly separate two independent error mechanisms (per-chunk quantization noise and autoregressive decode drift) that existing approaches conflate. The memory pressure that made naive quantization impossible forced a cleaner theoretical decomposition of the problem.

**Online Markov expert prediction.** Low VRAM residency (44%) means most experts are cold. Acceptable cold-hit rates require prediction. The forced alternative to offline calibration — online learning from production routing observations — adapts to the actual workload distribution rather than a calibration proxy, and held out it covers 4.7× the cold experts popularity does, with learning within the session as its dominant lever (§9.7). The residency constraint that made on-demand loading inadequate forced a prediction system that needs no calibration.

**Structural segments via projection.** Dynamic system-prompt sections exist in real deployments — persona, response style, and tool catalogs rotate between turns. The standard solution (store K/V uniformly for every token) closes a class of error invisibly: stored structural K/V references the ingest-time prefix, and when that prefix differs from the assembly prefix, attention scores degrade silently and drift compounds across turns. The constraint — structural orientation markers must remain semantically correct under dynamic frame rotation — closed the standard storage approach and forced a separable representation: content segments stored, structural segments live-prefilled from the dialect catalog against the current assembly's causal prefix and cached per slot by a hash of that prefix (§5). On the attention-only models the forced alternative is structurally cleaner: storage holds only what is portable, the chat template becomes a structural primitive of the inference engine rather than a prompt-engineering convention, the substrate shrinks, and the theorem (§11.2) covers boundary tokens without a frame-mismatch exception. A cached run is injected rather than recomputed, so a reprojection that keeps its prefixes pays only for the runs whose prefix changed. The constraint has a limit, which §5 records: a recurrent model cannot gap-fill, so on the production lineage the same markers are sealed K/V instead.

The pattern in each case: the constraint closed the standard solution, and the forced alternative was not merely adequate but, in the dimension the constraint bounds, better. Our reading is that this is a property of constraints that are accepted rather than avoided: when a standard approach is ruled out by resource limits, the replacement must be more efficient in the dimension the constraint bounds, and efficiency in that dimension often generalises, because the constraint exposed an avoidable cost in the standard approach. It is a pattern observed in one system, not a law, and its counterexample is in the same list — the structural-segment design had to be set aside for the recurrent lineage.

The codebase dependency analysis evaluation (§9.12) applies the methodology at the evaluation level. The constraint was: show the system works on real structured data, not synthetic probes. The response — the system's own codebase as the test case, with its author as the ground-truth oracle — gives an evaluation that is checkable against a codebase anyone can read. Its portability is bounded by what is released: the source is public, the ingested substrate is valid only for the model that wrote it and is not released, and the battery and its ground truth are in the repository beside its results.

---

## 12. Conclusion

We have presented a complete inference system built from first principles as a coherent design. The four primary contributions — online Markov expert prediction with wave-batched DMA overlap, adaptive per-block KV quantisation with two-phase prefill refresh, attentional provenance indexing with mid-decode reprojection, and an unbounded three-tier paged context — are individually grounded in the literature but have not previously been integrated.

The empirical results demonstrate that the coherence dividend is real and substantial. A 180B mixture-of-experts model serves eight concurrent sessions from a 16GB laptop GPU with 32GB of host RAM, above every published single-GPU run of it on machines with two to four times the memory. Its throughput is flat in context depth — 99% of prefill and 111% of decode kept from 32K to 128K — because the model's own sparse-attention indexer bounds each token's attended set and the engine adds no depth-dependent cost on top: the bounded-working-set principle, measured as time, though not by provenance. The KV cache compresses up to 7.6× inline with every output validated, at nearly 4× the compression of llama.cpp's q8_0 cache while slowing decode less. One card's aggregate decode exceeds the best published llama.cpp single-stream rate for the same model and card class by 1.3–37.9×, reaching 2,869.9 t/s across 64 sessions; and a 284B model serves sixteen sessions from a single GPU at 4.1× the best published single-GPU decode (§9.6). The concurrency regime these occupy is one existing consumer deployments do not reach: the most comparable published study found standard frameworks degrade beyond 2 concurrent users on an RTX 5090 32GB [Herz et al., arXiv:2512.23029], and the published single-card serving runs of the 35B models stop at 5–10.

The primary theoretical result is the Asymptotic Numerical Stability theorem (§11.2): under provenance-selected attention over unbounded context, the summed stored error of the tokens entering a generation step is bounded by a constant independent of N, where the same bound under full attention grows linearly in N. Warm- and cold-tier tokens share a fixed retrieval budget and together contribute a bounded constant C_ret, small when retrieved blocks originate from prefill-refreshed hot-tier blocks and pass the per-block threshold. The result bounds exposure, not output error, and it does not show that full attention degrades; what it changes is the framing of the KV quantization literature, which has optimised within the full-attention regime without asking whether the regime could be left. Provenance-selected attention leaves it structurally.

The broader competitive position follows from the same distinction. The existing KV quantization literature — KIVI, KVQuant, TurboQuant — optimises the compression primitive: better quantization formulas, rotation, codebooks. This paper optimises the compression architecture: when to quantize, what quality to guarantee per block, how to separate error sources, and how error scales with depth. The architecture absorbs primitive-level failures through adaptive selection; it does not depend on having the best primitive. That is why the system maintains per-block validated quality at compression ratios where population-level systems begin to degrade — the architecture compensates where any fixed primitive falls short.

The application target — many concurrent persistent-memory agent sessions on a single consumer card, each with unbounded conversation history — is met at every scale measured: sixteen users of a 35B model and eight of a 180B model on a 16GB laptop, sixty-four of a 35B model on one 72GB card. The problem the system was designed to solve is solvable at scale. The inference engine exists in service of that requirement, not as an end in itself.

The codebase dependency battery (§9.12) is the evidence that projection and reprojection with provenance work together. Retrieval finds the needle — the answer file first of 6,854, in a 40M-token corpus, on 65% of questions, top five on 83%. The model then answers from memory with tools forbidden at 57.3% recall over three runs, documented architecture well and call sites less well, and declines rather than invents. Holding the working set fixed while the model reasons takes that to 38.8%, every held run below every reprojecting run: re-selection during reasoning is worth 18.5 points, most of them on answers in files the question does not name. *[PENDING: battery-128]* The 128-question battery measures each part on its own: against the model alone the memory is worth __ points, reprojection during reasoning __, continuous reprojection __ over a single retrieval, and the seed files __, each with its interval and Holm-corrected p. The theorem's direct test (§9.11.2) shows its premise holds — a bounded working set selected, needle for needle, from a history twelve times its size, at the top compression level as at near-lossless. The deployment itself — §9.13 — is operating experience of what the architecture is for: memory that outlives the process, the session and the model that wrote it. That experience is not a substitute for the battery's table, and is not offered as one.

The sliding-window alternative degrades by construction rather than gracefully: a window that cannot see past its boundary cannot discover a dependency that lives beyond it, and a 131K window covers only a fraction of a 2.2M-line codebase. The contrast that matters is measured in its sharpest form: the same provenance index, the same opening retrieval and every other reprojection, with only the reasoning's re-selection held — and the battery loses 18.5 points (§9.12). *[PENDING: battery-128]* The retrieve-once turn — the same index and the same opening retrieval, then a fixed window through the reasoning — scores __ points below the production system over 128 questions. What is measured is retrieval quality on the live gallery (§9.11.1) and against the 40M-token code corpus (§9.12), selection far past the working set (§9.11.2), the battery from memory and its reasoning-hold ablation (§9.12), output fidelity against depth (§9.6.8), and the absence of a depth cliff in the engine's own throughput (§9.6.2). The KV quantization literature — KIVI, KVQuant, TurboQuant — optimises inside a regime whose boundaries it has not examined; the growth of exposure with N is a property of full attention, not a law of compression, and this architecture leaves the regime (§11.2, §11.3). Attention is retrieval: the provenance index performs approximate attention over the full unbounded context, selecting which blocks enter exact attention; standard attention is flat retrieval at O(N) cost; this system is hierarchical retrieval whose exact-attention cost per step does not grow with N, plus a scan that does — flat over the resident gallery, ~9.5 ms steady-state on the laptop, and off the critical path; RAG replaces the attention mechanism and loses attentional continuity; this system preserves the mechanism and removes the scaling constraint on exact attention (§6, §11.4). The structural argument is straightforward: a system running full attention over a 2.2M-line codebase reads every compressed token at every step once it holds more than it can keep at full precision; this one reads a bounded working set, on a laptop GPU — and more hardware does not give full attention a depth-independent bound (§11.2 Corollary 2).

The deeper lesson is methodological: constraints accepted as first-class design requirements rather than problems to be avoided force architectural decisions that are more efficient in the dimension the constraint bounds — and efficiency in that dimension tends to generalise. The 16GB ceiling produced native quantized kernels that never pay for a dequantised copy. The finite hot tier produced the sparse attention that makes unbounded context tractable to bound. The memory pressure of concurrent sessions produced the two-phase quantization that separates two error mechanisms the literature conflates. Each constraint closed a standard solution and forced one that holds up beyond it. The measurements show it on three cards from 16GB to 72GB; the theorem explains why the bound, though not every figure, carries to any card.

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

*Published throughput figures compared against in §9.6. Each is recorded with its card, quantization, context length and engine in `docs/performance.md` §6.*

[29] llama.cpp Discussion #15013. **Performance of llama.cpp on Nvidia CUDA** (Llama-2-7B Q4_0). https://github.com/ggml-org/llama.cpp/discussions/15013

[30] Hardware Corner. (2026). **RTX 3090 LLM benchmarks.** https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-3090/

[31] Hardware Corner. (2026). **RTX 4080 LLM benchmarks.** https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4080/

[32] jonidimo. (2026). **Qwen3.8-27B on One RTX 3090: 14h Measured Benchmark.** https://jonidimo.github.io/qwen38-3090-benchmark/benchmark.html

[33] ryan4yin. (2026). **Best llama.cpp config for Qwen3.8-Flash-Next (RTX 4090 24GB).** https://gist.github.com/ryan4yin/48617bbddacc7067f10799770b7cc33f

[34] unsloth/Qwen3.8-Flash-Next-GGUF discussion #3. (2026). **Share your model speed here.** https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3

[35] InsiderLLM. (2026). **Best Way to Run Qwen 3.6 35B MoE Locally.** https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/

[36] llama.cpp Discussion #19890. (2026). **RTX 5090 (CUDA) vs Radeon AI PRO R9700 (Vulkan) — Qwen3.5-35B-A3B.** https://github.com/ggml-org/llama.cpp/discussions/19890

[37] SOTAAZ. (2026). **llama.cpp KV Cache Quantization, Measured on One A100.** https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en

[38] vLLM Blog. (2026). **The State of FP8 KV-Cache and Attention Quantization in vLLM.** https://vllm.ai/blog/2026-04-22-fp8-kvcache

[39] vLLM Blog. (2026). **A First Comprehensive Study of TurboQuant: Accuracy and Performance.** https://vllm.ai/blog/2026-05-11-turboquant

[40] ByteShape. (2026). **If It Fits, It Sits: Qwen 3.6 35B.** https://byteshape.com/blogs/Qwen3.6-35B-A3B/

[41] Hardware Corner. (2026). **RTX 5090 LLM benchmarks.** https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5090/

[42] Millstone AI. (2026). **Qwen3.5-35B-A3B FP8 on 1× RTX Pro 6000.** https://www.millstoneai.com/inference-benchmark/qwen3-5-35b-a3b-fp8-1x-rtx-pro-6000-blackwell

[43] Millstone AI. (2026). **Qwen3.6-35B-A3B FP8 on 1× RTX Pro 6000.** https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-fp8-1x-rtx-pro-6000-blackwell

[44] holy_fox. (2026). **Running Qwen3.8-Flash-Next on an RTX 5090 with 128GB RAM using llama.cpp.** https://zenn.dev/holy_fox/articles/04887ff8177b87?locale=en

[45] llama.cpp PR #24162. (2026). **DeepSeek V4.** https://github.com/ggml-org/llama.cpp/pull/24162

[46] RockmSockmJesus. (2026). **DeepSeek-V4-Flash (284B MoE) at ~28 tok/s on 1x RTX 5090.** https://gist.github.com/RockmSockmJesus/30a195ccd9b62e981ec2676a99a57b7e

*Further works cited in the text by author and year.*

[47] Liu, N.F., Lin, K., Hewitt, J., et al. (2024). **Lost in the Middle: How Language Models Use Long Contexts.** TACL 2024. https://arxiv.org/abs/2307.03172

[48] Staniszewski, K., Łańcucki, A. (2026). **KV Cache Transform Coding for Compact Storage in LLM Inference.** ICLR 2026. https://arxiv.org/abs/2511.01815

[49] Zhong, S., Sun, Y., Liang, L., et al. (2025). **HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference.** DAC 2025. https://arxiv.org/abs/2504.05897

[50] OpenRouter. (2026). **Cache is Crucial.** *Inferred*, no. 3. https://openrouter.ai/data/newsletter/cache-is-crucial

[51] Bai, L., Huang, Z., Wang, X., et al. (2026). **How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks.** arXiv:2604.22750. https://arxiv.org/abs/2604.22750

[19], [21], [24] and [25] are background reading and are not cited in the text.

---

## Appendix A: Expert Pipeline VRAM Budget

The v1 design budget for Qwen3-30B-A3B on the 16GB laptop, with its derivation, is §10.2 (an expert cache of ~2,700 of 6,144 slots, ~8 GB; ~2 GB of non-expert blocks; ~4 GB of KV hot tier at ~3× compression, ~125K tokens at Qwen3-30B-A3B's 96 KB a token at F16; ~2 GB of working buffers). It is not repeated here. The production engine's span partition is described at the head of §10.2, its expert side in `docs/expert_cache_design.md`, and its measured KV fill in §9.11.

---

## Appendix B: Public Release and Community Collaboration

**Source code:** https://github.com/john-sharratt/candle (public)

**Live system:** https://code.tokera.com — the deployed coding assistant over this repository, on the cluster that produced §9.11.1 and §9.13. It requires sign-in; access is by allowlist, with a queue for guests. The project site, https://www.tokera.com, carries v1 of this paper.

**What is released.** The full inference engine codebase, and the raw measurement rows behind every figure in this paper — `docs/performance.md` and `docs/results/`, each row naming its build, its machine and the test that produced it — with the tests that produce them. What is not released: the substrate the live gallery holds (a KV archive valid only for the model that wrote it, and holding conversations that are not this paper's to publish), and an interface to the retrieval record — the live system shows a conversation's turns and tool calls, not which signatures matched at each reprojection. A turn's `projection` (Full, Limited to the journals, or Minimal to the tools), `seed` and reprojection are dials of the request and of the MCP tools that start a turn, as its thinking and tools are, so every arm of §9.12 is a setting a caller can choose — and a task can limit what it is shown. The §9.12 battery's questions and ground truth (`zend/examples/dependency_battery/battery.yaml`), its harness and analysis, and every answer's per-question score (the pilot's `docs/results/dependency_battery_2026-11-10.md`, `docs/results/dependency_battery_2026-11-20_b.md` and `docs/results/dependency_battery_2026-11-23_l_m.md`; the 128-question battery's *[PENDING: battery-128 — results file]*) are in the repository, as are the tests behind §9.6.8 and §9.11.2.

**Invitation.** This is v2, and it is open for full critical review. The v1 erratum is part of the record rather than quietly corrected, and the same discipline applies here: every figure in this version names its build, its machine and its test, and every claim that is not measured says so. Contributions to validation, stress-testing, or optimisation are invited — contributors will be recognised in the next revision. Issues, pull requests, and evaluation results may be submitted via the GitHub repository.