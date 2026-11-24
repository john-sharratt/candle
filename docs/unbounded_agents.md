# One Card, One Stack: Constraint-Driven Architecture for Asymptotically Stable Inference over Unbounded Agent Memory

> **Version 2.** v1 (13 May 2026, [10.5281/zenodo.20156060](https://zenodo.org/records/20156060)) is archived with an erratum at `docs/unbounded_agents_v1.md`; its "509 t/s" and "2,446 t/s" headlines were prefill rates and are withdrawn. v2 adds a measured evaluation across three machines and twelve models from 0.5B to 284B parameters (§9.6), every number produced by a test in this repository and recorded in `docs/performance.md`, and runs the retrieval evaluation on the production system (§9.11–§9.12). §4.1–§4.5 describe the v1 design on Qwen3-30B-A3B; provenance as production runs it is §6.

**Abstract**

Persistent agents need context that grows without bound. Under full attention every cached token takes part in every later step, so the stored error a step is exposed to grows with context depth, and more VRAM only defers the point where tokens must be compressed or evicted. We argue the coupling is architectural, and that decoupling the working set from context depth is sufficient to bound the exposure.

**Theorem (§11.2).** Under provenance-selected attention over a tiered context, the summed stored error of the tokens entering a generation step is bounded by a constant independent of context depth N; under full attention the same bound grows linearly in N. The theorem bounds exposure, not output error, and §11.2 states what it does not establish.

The system combines (1) online Markov expert prediction with wave-batched grouped GEMM for MoE inference under partial VRAM residency; (2) per-block adaptive KV quantization across eleven levels (C0–C10) with asymmetric K/V error metrics and a prefill refresh at every turn seal, compressing the KV cache 4.1–7.6× at the top level; (3) attentional provenance indexing — Q-vector sign signatures scanned on the GPU, re-selecting the working set during decode; and (4) an unbounded three-tier context, VRAM-hot, RAM-warm and disk-cold.

Implemented in Rust on a Candle fork, the system runs a **180B-parameter MoE model on a 16 GB laptop GPU**, serving eight sessions at 108.5 t/s aggregate decode, above every published single-GPU run of the model. From 32K to 128K tokens it keeps **99% of its prefill and 111% of its decode**. One card's aggregate decode is **1.3–37.9×** the best published llama.cpp single-stream rate for the same model and card class, reaching **2,869.9 t/s across 64 sessions**, and a **284B model serves sixteen sessions from one GPU** at 4.1× the best published single-GPU decode.

On the system's own 2.2M-line codebase, retrieval ranks a question's answer file first among 7,943 files on 49% of 121 questions and in the top five on 69%, by Q-signature agreement alone. Answering 128 questions from memory with tools forbidden, over seven arms and three runs, the memory is worth **36.7 points of recall**, re-selection during reasoning 4.6, and continuous re-selection 6.5 over a single retrieval, every interval clear of zero (§9.12). Mid-decode reprojection selects every fact planted in a history twelve times the working set, at the top compression level as at near-lossless, where a single projection and a rolling window select none (§9.11.2).

---

## 1. Introduction

Long-running assistants and autonomous agents need context that grows without bound, and under full attention every token held grows the set of stored, compressed or rounded values every later step reads. More VRAM defers the threshold; it does not remove the coupling. RAG externalises memory and loses attentional continuity; longer native windows are a hardware investment against an architectural ceiling; summarization and eviction discard the verbatim detail persistent memory exists to keep.

The fix here is to decouple the tokens that take part in a step from the tokens in the context. §11.2 shows the decoupling is sufficient to bound a step's exposure at any depth. We do not prove it necessary — full attention's output is a convex combination of its values, and whether its error grows depends on the attention distribution — but a full-attention system cannot offer the same bound, because the bound's terms are the tokens it attends.

The system came from two constraints held together: persistent agent conversations, the hardest form of the memory problem (verbatim recall across unbounded sessions, many agents at once), and a single 16 GB consumer GPU. Each constraint closed the standard solution: GEMM dequantisation OOMed, so the matmul kernels read quantized weights natively; full context could not be held hot, so attention became provenance-selected; materialising turns at F16 OOMed, so quantization became two-phase (§11.5). The result, on that card and two larger ones:

- A **180B-parameter MoE model** (Qwen3.8-Flash-Next, 6B active) on a 16 GB laptop GPU with 32 GB of RAM, experts streamed VRAM → RAM → NVMe, serving eight sessions at 108.5 t/s aggregate decode — above every published single-GPU run of it, each on a host with 64–128 GB (§9.6.5).
- **Throughput flat in depth**: 99% of prefill and 111% of decode kept from 32K to 128K, the model's own sparse-attention indexer bounding each token's attended set (§9.6.2).
- **KV compression up to 7.6×**, each 32-token block choosing its format as it is sealed; at 8K the top level costs 0–8% of decode for 3.3–6.3× (§9.6.3).
- **Concurrency**: aggregate decode 1.3–37.9× the best published llama.cpp rate for the same model and card class, 2,869.9 t/s across 64 sessions on one card; a 284B model at sixteen sessions on one GPU, 4.1× the best published single-GPU decode (§9.6.4, §9.6.6).
- **Retrieval over its own codebase**: the answer file first of 7,943 on 49% of questions and top-five on 69%; the memory worth 36.7 points of recall with tools forbidden, re-selection during reasoning 4.6 (§9.12).
- **Selection at depth**: every fact planted in a history twelve times the working set is selected, at C10 as at C0 (§9.11.2).

The agent use case is the existence proof that the parts run together; it is not the subject of the paper. To our knowledge no prior serving system states a bound on per-step error exposure at unbounded depth. An 80 GB card holds more tokens at full precision, but once a token is compressed or evicted it is in the same position as on a 16 GB card (§11.2, Corollary 2).

### 1.2 Technical Contributions

1. **Online Markov expert prediction** (§4) — a transition matrix learned from live routing, with no calibration, and a wave-batched grouped GEMM that amortises each expert's transfer across the sessions that route to it.
2. **Two-phase KV quantization** (§5) — K/V re-prefilled from the finished turn at its seal, removing decode drift, then a per-block format chosen independently for K and V by error metrics matched to K's softmax-amplified and V's linear error paths [AsymKV, COLING 2025].
3. **Attentional provenance indexing** (§6) — Q-vector sign signatures matched Q→Q by XNOR and popcount, scanned on the GPU over a VRAM-resident gallery by tensor-core kernels (~9.5 ms on a 16 GB laptop GPU) and re-selecting the working set every 64 decoded tokens, off the decode's critical path. The think block is the probe that decodes ahead of the answer.
4. **Unbounded three-tier context** (§7, §11.2) — VRAM-hot, RAM-warm and disk-cold paging, with the proof that the stored error entering a step is bounded independent of depth under provenance-selected attention.
5. **Native quantized kernels** (§8, supporting) — matmuls that never materialise a full-precision weight copy, greedy batch decomposition, and a fused sampling kernel.

---

## 2. Background and Related Work

### 2.1 MoE Expert Offloading

Fiddler [Kamahori et al., 2024] and HybriMoE [Zhong et al., 2025] partition experts between CPU and GPU by activation frequency; DAOP [DATE 2025] adds data-aware prefetching. All rely on offline profiling or trained predictors, which degrade on novel distributions. Our Markov predictor learns online from live routing, and the wave-batched kernel coalesces expert work across concurrent requests (§4.3, §9.7).

### 2.2 KV Cache Quantization

**KIVI** [Liu et al., ICML 2024] established channel-wise outliers in K and token-wise structure in V, quantizing both uniformly at 2 bits. **KVQuant** [Hooper et al., NeurIPS 2024] added per-vector outlier handling and first-token special-casing, under 0.1 perplexity at 3 bits. **TurboQuant** [Zandieh et al., ICLR 2026] rotates and Lloyd-Max quantizes, with its advantage below 3 bits; community benchmarks on Qwen3 find Q4_0 matching its residual implementation at 4–5×. **KITTY** [Xia et al., 2025] boosts 12.5–25% of K channels to 4 bits by an offline magnitude ranking and names adaptive selection as future work. Token eviction (H2O, TOVA, StreamingLLM) is lossy by design, DMS needs fine-tuning, SVD approaches (xKV, LoRC) are per-prompt, and KVTC [Staniszewski & Łańcucki, ICLR 2026] compresses ~20× offline for storage rather than live inference. OTT [Su et al., ACL 2025] finds outlier tokens cluster at block boundaries, independently motivating our 4/28 sub-block structure.

Every one of these optimises the compression *primitive*. None treats format assignment as a per-block decision with a quality guarantee, and none states a result on error scaling with depth. Here each (layer, head) block chooses its own K and V formats, by Q-projected error for K and normalised L2 for V, along an 11-point Pareto curve (C0–C10) that runs on dense, MoE, DeltaNet-hybrid and native-sparse models up to 284B with no per-model calibration data beyond a threshold scale (§9.9).

### 2.3 Autoregressive Error Accumulation

GEAR [Kang et al., 2024] showed quantization error in autoregressive decode compounds across steps. RefreshKV [Xu et al., ACL 2025] regenerates the KV periodically with full attention, as a standalone technique. Our refresh is turn-boundary-triggered and feeds the per-block format selection directly; RelayCaching [2026] confirms prefill and decode K/V agree closely, so prefill is the right ground truth.

### 2.4 Attention Sinks

StreamingLLM [Xiao et al., ICLR 2024] showed initial tokens carry disproportionate attention; KVSink [Su & Yuan, COLM 2025] extended this to mid-sequence sinks. Here, on attention-only models, turn-boundary sinks are recomputed in the current frame at every assembly rather than stored (§5). KVTuner [2025] and [llama.cpp #20969, 2025] find boundary layers more sensitive to K quantization on Qwen3.

### 2.5 KV-Based Retrieval

RetrievalAttention [Liu et al., NeurIPS 2025] measured the Q→K gap: Q vectors sit more than 10× farther from K than K from K, degrading cross-distribution ANNS. Provenance indexing stores Q signatures, so the dominant signal is a within-distribution Q→Q match.

---

## 3. System Architecture

One requirement organises the system: the tokens attending any step must be bounded independently of context depth, the property §11.2 uses. Figure 3-1 shows one decode step and the two loops that feed it. No component materialises an intermediate the next operation does not consume: there is no dequantised weight copy and no BF16 expansion between the logits and the sampling modifiers.

<figure class="fig">
<svg viewBox="0 0 640 292" role="img" aria-label="Figure 3-1. One decode step: embedding, attention over a working set, MoE experts streamed from VRAM, RAM and NVMe, a fused sampling kernel, and the next token. Every 64 decoded tokens the last 128 Q signatures are scanned against the VRAM gallery in about 9.5 ms, off the critical path, and the result assembles the next forward's working set. When a turn ends it is re-prefilled and each 32-token block is sealed in its own format, C0 to C10, into three tiers: VRAM hot, RAM warm, disk cold, which also hold the gallery.">
  <text class="ttl" x="16" y="20">Figure 3-1 · One decode step, and the loops that feed it</text>
  <text class="ttl-sub" x="16" y="36">each forward attends a working set, never the whole context</text>
  <rect class="box" x="16" y="56" width="96" height="52" rx="6"/>
  <text class="t-lbl mid" x="64" y="78">Embedding</text>
  <text class="t-dim mid" x="64" y="94">host lookup</text>
  <rect class="box weights" x="132" y="56" width="132" height="52" rx="6"/>
  <text class="t-lbl mid" x="198" y="78">Attention</text>
  <text class="t-dim mid" x="198" y="94">paged KV, working set</text>
  <rect class="box" x="284" y="56" width="132" height="52" rx="6"/>
  <text class="t-lbl mid" x="350" y="78">MoE experts</text>
  <text class="t-dim mid" x="350" y="94">VRAM → RAM → NVMe</text>
  <rect class="box" x="436" y="56" width="96" height="52" rx="6"/>
  <text class="t-lbl mid" x="484" y="78">Sampling</text>
  <text class="t-dim mid" x="484" y="94">one fused launch</text>
  <rect class="box" x="552" y="56" width="72" height="52" rx="6"/>
  <text class="t-lbl mid" x="588" y="86">token</text>
  <rect class="box" x="132" y="140" width="400" height="50" rx="6"/>
  <text class="t-lbl mid" x="332" y="161">Provenance scan, every 64 decoded tokens</text>
  <text class="t-dim mid" x="332" y="178">last 128 Q signatures vs the VRAM gallery · ~9.5 ms</text>
  <rect class="box" x="132" y="222" width="284" height="50" rx="6"/>
  <text class="t-lbl mid" x="274" y="243">Turn seal: re-prefill the finished turn</text>
  <text class="t-dim mid" x="274" y="260">fresh K/V; each block's format C0–C10</text>
  <rect class="box weights" x="436" y="222" width="188" height="50" rx="6"/>
  <text class="t-lbl mid" x="530" y="243">Three tiers</text>
  <text class="t-dim mid" x="530" y="260">VRAM hot · RAM warm · disk</text>
  <path class="ax" d="M112 82 H132 M264 82 H284 M416 82 H436 M532 82 H552 M588 108 V165 H532 M198 140 V108 M480 222 V190 M416 247 H436 M88 247 H132"/>
  <path class="t-dim" d="M132 82 l-6 -3.5 v7 z M284 82 l-6 -3.5 v7 z M436 82 l-6 -3.5 v7 z M552 82 l-6 -3.5 v7 z M532 165 l6 -3.5 v7 z M198 108 l-3.5 6 h7 z M480 190 l-3.5 6 h7 z M436 247 l-6 -3.5 v7 z M132 247 l-6 -3.5 v7 z"/>
  <text class="t-dim" x="542" y="182">Q signatures</text>
  <text class="t-dim" x="206" y="128">next forward's working set</text>
  <text class="t-dim" x="486" y="210">gallery, K/V</text>
  <text class="t-dim" x="16" y="251">turn ends</text>
</svg>
</figure>

---

## 4. Online Markov Expert Prediction and Wave-Batched MoE

### 4.1 The Problem

*§4.1–§4.3 describe the v1 design on Qwen3-30B-A3B; the production cache and dispatch close §4.3 and fill §4.4.*

Qwen3-30B-A3B has 48 MoE layers of 128 experts, ~18 GB at Q4_K_M. With half of a 16 GB card for experts, 44% are resident, and without prediction 56% of activations are cold misses, each stalling the stream for a ~116 μs copy from pinned memory (`docs/expert_cache_design.md`).

### 4.2 Online Markov Transition Matrix

After each forward the predictor credits a `[128 × 128]` count table per layer pair, `T[L][from][to]`, only when `to` was not already active at the source layer, so it learns the *cold* experts a prefetch must cover. Production keeps one table per hop ahead and stays silent until a table has seen 64 arrivals. Successors are scored by pointwise mutual information, which discounts globally popular experts already cached:

```
PMI(α)[j] = Σ_i P(j | i) / P(j)^α,   i ∈ active experts at L,   α = 0.5
```

Candidates are staged into **free slots only**, so a misprediction is inert. Out of sample — 21-fold leave-one-out over a 21-prompt routing trace — PMI covers **4.7×** the cold experts a popularity baseline does at a fan-out of four, with top-1 precision of 80–91% against 22% (§9.7). A session's own transitions predict its next layers better than any cross-prompt average.

### 4.3 Eviction Policy

Timestamp eviction cascades — a miss at L evicts an expert L+5 needs — and halved single-token decode. Four rules remove the cascade: evict the bottom 5% by timestamp at the end of each pass, leaving ~140 free slots; on a forced eviction, evict from completed layers first; pin layers 0 and 1, which have no compute ahead to hide a reload; prefetch into free slots only. Frequency-weighted recency cuts the miss rate from LRU's 22.9% to 14.7% at half the working set, where a learned recurrence model did worse than LRU (39.5%) (§9.7).

**Production: three tiers.** The production cache treats the on-disk expert pack as authoritative and VRAM and pinned RAM as independent copies of it, so eviction never writes back. Its key is access frequency weighted by the wrapped distance to the layer's next use — Bélády's ordering, exact because the layer wave is a cycle — evaluated by tier, so an expert whose miss costs an NVMe read is protected over one whose miss is a pinned copy (Figure 4-1). The same gate on the laptop now measures 6,139.1 t/s at twenty contexts with its experts streaming (§9.6).

<figure class="fig">
<svg viewBox="0 0 640 262" role="img" aria-label="Figure 4-1. Aggregate prefill on Qwen3-30B-A3B, RTX 4090 Laptop GPU, two-tier against three-tier expert cache. F16 one context 377.5 to 708.1 t/s, plus 87 percent. BF16 one context 509.4 to 780.6, plus 53 percent. BF16 ten contexts 1,931.4 to 2,726.1, plus 41 percent. Q8_0 twenty contexts 2,448.1 to 2,732.5, plus 12 percent. C9 two contexts 951.4 to 1,463.1, plus 54 percent.">
  <text class="ttl" x="16" y="20">Figure 4-1 · The three-tier expert cache, against two tiers</text>
  <text class="ttl-sub" x="16" y="36">Qwen3-30B-A3B · RTX 4090 Laptop 16 GB · aggregate prefill t/s · amber two-tier, green three-tier</text>
  <path class="grid" d="M335.7 54 V226 M471.4 54 V226"/>
  <path class="grid base" d="M200 54 V226"/>
  <text class="cat" x="16" y="72">F16 × 1 context</text>
  <text class="cat-sub" x="16" y="85">+87%</text>
  <rect class="them" x="200" y="60" width="51.2" height="11"/>
  <rect class="us" x="200" y="73" width="96.1" height="11"/>
  <text class="v-them" x="257.2" y="69">377.5</text>
  <text class="v-us" x="302.1" y="83">708.1</text>
  <text class="cat" x="16" y="106">BF16 × 1 context</text>
  <text class="cat-sub" x="16" y="119">+53%</text>
  <rect class="them" x="200" y="94" width="69.1" height="11"/>
  <rect class="us" x="200" y="107" width="105.9" height="11"/>
  <text class="v-them" x="275.1" y="103">509.4</text>
  <text class="v-us" x="311.9" y="117">780.6</text>
  <text class="cat" x="16" y="140">BF16 × 10 contexts</text>
  <text class="cat-sub" x="16" y="153">+41%</text>
  <rect class="them" x="200" y="128" width="262.1" height="11"/>
  <rect class="us" x="200" y="141" width="370" height="11"/>
  <text class="v-them" x="468.1" y="137">1,931.4</text>
  <text class="v-us" x="576" y="151">2,726.1</text>
  <text class="cat" x="16" y="174">Q8_0 × 20 contexts</text>
  <text class="cat-sub" x="16" y="187">+12%</text>
  <rect class="them" x="200" y="162" width="332.2" height="11"/>
  <rect class="us" x="200" y="175" width="370.8" height="11"/>
  <text class="v-them" x="538.2" y="171">2,448.1</text>
  <text class="v-us" x="576.8" y="185">2,732.5</text>
  <text class="cat" x="16" y="208">C9 × 2 contexts</text>
  <text class="cat-sub" x="16" y="221">+54%</text>
  <rect class="them" x="200" y="196" width="129.1" height="11"/>
  <rect class="us" x="200" y="209" width="198.6" height="11"/>
  <text class="v-them" x="335.1" y="205">951.4</text>
  <text class="v-us" x="404.6" y="219">1,463.1</text>
  <text class="tick mid" x="200" y="244">0</text>
  <text class="tick mid" x="335.7" y="244">1,000</text>
  <text class="tick mid" x="471.4" y="244">2,000 t/s</text>
</svg>
</figure>

### 4.4 Wave-Batched Grouped GEMM

Routing prefill and decode requests through the expert layer together as a wave makes the expert hot set the *union* of the requests' activations, smaller than their sum because similar content routes alike: when two requests need expert 7, its weights load once. In MoE this coalescing is the real batch dimension. The production dispatch (`docs/moe_live_dispatch_design.md`) keeps routing on the device. Per layer, a kernel classifies each routed expert as VRAM, pinned RAM or cold and writes a summary into a mapped host ring; the grouped GEMM reads VRAM experts in place and copies pinned ones itself; only a cold expert waits, on a mapped word the stager publishes. No host CUDA call reads the routing. Modelled on the held-out trace at 60% residency, prefetch and eviction leave ≈13% of accesses as stalling misses, and 7.8% when tuned for stall cost rather than hit rate (§9.7); the live predictor's precision is 66–85%, typically 74%.

### 4.5 Early-Layer Pinning and Prediction

Layers 0 and 1 stay resident: they have no compute ahead of them to hide a transfer behind, so the predictor and the warm tier skip them, and prediction runs from layer 2.

---

## 5. Two-Phase KV Cache Quantization

Two error sources need separate fixes. **Decode drift**: each decoded token's K/V is computed over already-quantized predecessors, so error compounds along the causal chain [GEAR, Kang et al., 2024]. **Per-block noise**: a sealed 32-token block's error is bounded by its own activations and does not propagate. The two phases answer each (Figure 5-1).

**Phase 1 — seal each block as it fills.** Every 32-token block is quantized the moment it seals, its K and V formats chosen against its live activations, so the F16 working footprint is one block per session. Materialising whole turns instead — 200 sessions of 500 tokens on Qwen3-30B-A3B, 96 KB a token — would need ~9.8 GB beyond the weights. The unsealed tail is always F16, so the most recent tokens are always read at full precision.

**Phase 2 — re-prefill the finished turn.** At turn completion one batched prefill recomputes the turn's tokens in parallel, none depending on a quantized predecessor, and the clean K/V replaces the drifted blocks. It costs the turn's length, not the context's; RelayCaching [2026] confirms prefill gives the ground-truth K/V.

<figure class="fig">
<svg viewBox="0 0 640 214" role="img" aria-label="Figure 5-1. Two-phase quantization. During decode each 32-token block is sealed and quantized as it fills, its format chosen per block, while the unsealed tail stays F16. When the turn ends it is re-prefilled in parallel, giving fresh K/V with no decode drift, and every block is quantized again.">
  <text class="ttl" x="16" y="20">Figure 5-1 · Two phases: seal as you go, refresh at the turn's end</text>
  <text class="ttl-sub" x="16" y="36">one turn, 32-token blocks · colour = the block's own format, cheapest first</text>
  <text class="cat" x="16" y="76">during decode</text>
  <rect class="q2" x="150" y="58" width="50" height="28" rx="3"/>
  <rect class="q3" x="204" y="58" width="50" height="28" rx="3"/>
  <rect class="q2" x="258" y="58" width="50" height="28" rx="3"/>
  <rect class="q4" x="312" y="58" width="50" height="28" rx="3"/>
  <rect class="q3" x="366" y="58" width="50" height="28" rx="3"/>
  <rect class="q2" x="420" y="58" width="50" height="28" rx="3"/>
  <rect class="q3" x="474" y="58" width="50" height="28" rx="3"/>
  <rect class="box weights" x="528" y="58" width="34" height="28" rx="3"/>
  <text class="t-dim" x="568" y="76">tail, F16</text>
  <text class="t-dim" x="150" y="102">each block sealed as it fills · K/V computed over quantized predecessors</text>
  <path class="ax" d="M356 108 V128"/>
  <path class="t-dim" d="M356 134 l-3.5 -6 h7 z"/>
  <text class="t-dim" x="364" y="124">turn ends: re-prefill the whole turn at once</text>
  <text class="cat" x="16" y="156">after the seal</text>
  <rect class="q2" x="150" y="138" width="50" height="28" rx="3"/>
  <rect class="q2" x="204" y="138" width="50" height="28" rx="3"/>
  <rect class="q3" x="258" y="138" width="50" height="28" rx="3"/>
  <rect class="q2" x="312" y="138" width="50" height="28" rx="3"/>
  <rect class="q4" x="366" y="138" width="50" height="28" rx="3"/>
  <rect class="q2" x="420" y="138" width="50" height="28" rx="3"/>
  <rect class="q3" x="474" y="138" width="50" height="28" rx="3"/>
  <rect class="q1" x="528" y="138" width="34" height="28" rx="3"/>
  <text class="t-dim" x="150" y="182">fresh K/V, no decode drift · every block quantized again from it</text>
  <g class="legend">
    <rect class="q1" x="150" y="194" width="10" height="9"/><text x="164" y="202">sub-4-bit</text>
    <rect class="q2" x="240" y="194" width="10" height="9"/><text x="254" y="202">4-bit</text>
    <rect class="q3" x="310" y="194" width="10" height="9"/><text x="324" y="202">5-bit</text>
    <rect class="q4" x="380" y="194" width="10" height="9"/><text x="394" y="202">8-bit</text>
  </g>
</svg>
</figure>

**Why the refresh sets the floor.** Under §11.2 a step's error is bounded by ε_hot + C_ret. Hot tokens — the current turn, pinned system context, refreshed recent history — are attended at every step, so their error is the permanent floor; the refresh keeps ε_hot ≈ 0. Retrieved warm and cold tokens share a fixed budget and contribute at most C_ret = W_warm_max · ε_ret: bounded, not vanishing. Compression can therefore be aggressive outside the hot tier only as far as the per-block threshold bounds ε_ret.

### Boundary Reconstruction via Projection Segments

The two phases cover **interior content**. The 4–6 **structural template tokens** at each seam — `<|im_start|>role\n`, `<|im_end|>\n`, block envelopes, role markers — carry no content but are heavily attended as orientation markers, and their K encodes the content before them. Under provenance selection every projection assembles a different prefix, so a structural token's K/V baked against one prefix is wrong for the next. This is a third error class, frame drift, independent of quantization noise and decode drift: a structural token can be numerically perfect and still describe scaffolding that is no longer there.

On attention-only models, structural K/V is therefore never persisted. A projection is a sequence of segments: **`Sealed`** segments patch stored K/V onto the slot by reference, in microseconds; **`Generated`** segments name template tokens from the model's dialect catalog and are live-prefilled against whatever actually precedes them, adjacent ones in one `forward_batched` run batched with the other slots' prefills; **`NewUserMessage`** carries the incoming turn, committed at its seal. Computed runs are cached by a hash of their whole causal prefix, so a match is reused and any change misses; K is stored un-rotated, so a cached run stays exact under a position shift. Structural K/V is always fresh F16, never quantized, and the gallery's signatures cover content tokens only (§6.6). A recurrent model — Flash-Next's DeltaNet layers, the production model — cannot compute a token inserted mid-sequence, so there every structural producer is sealed into the turn's own K/V and falls under the interior bound (`docs/deltanet_state_persistence.md` §4.6–§4.7d).

---

## 6. Attentional Provenance Indexing

### 6.1 Core Insight

The Q vectors a model computes while producing or reading content are a compressed record of its state — not only what the content says but the reasoning it was produced in. We call this *attentional provenance*. Each token's Q signature is captured as it is computed and stored beside its K/V, building an index held resident in VRAM (§6.6) that is scanned over the whole corpus before generation and, crucially, during it: re-selecting the working set at every reasoning boundary from the model's own Q vectors. Where K encodes content, a stored Q records what the model was reaching for, so a later query in a similar state matches it with no surface overlap.

### 6.2 From the v1 Fingerprint to the Production Signature

v1 indexed each item by six INT8 K and Q band averages, scored by a CPU dot product (`docs/unbounded_agents_v1.md` §6). Its calibration (§9.10) set the principle production keeps — **Q→Q sign agreement over a flat scan** — and two findings that still hold: prefill and decode Q occupy different regions of signature space, so a probe must match stored text of its own kind; and a binary signature's 50% chance agreement must be removed before scores are summed. Everything else was replaced: the signature by `WideQSig` (§6.6), the dot product by per-token votes (§6.3) read against each candidate's background (§6.4), the CPU scan by the GPU gallery, and per-section probes by reprojection during decode (§6.5). Figure 6-1 shows the production path.

<figure class="fig">
<svg viewBox="0 0 640 228" role="img" aria-label="Figure 6-1. The provenance path. Each token's Q vectors over every attention layer are reduced to signs and folded into three layer groups per KV head, a 192-byte WideQSig. The gallery holds these in VRAM in 32-token pages. A probe, the last 128 decoded tokens, is compared with every gallery token on the GPU by XNOR and popcount; each probe token votes for its best eight files by z times margin; scores are normalized against each file's own background; whole conversations enter a pool, and the model's own sparse-attention indexer decides which of their blocks fill the window.">
  <text class="ttl" x="16" y="20">Figure 6-1 · From a token's Q to the working set</text>
  <text class="ttl-sub" x="16" y="36">stored at every token · scanned every 64 decoded tokens · the result feeds the next forward</text>
  <rect class="box" x="16" y="56" width="180" height="46" rx="6"/>
  <text class="t-lbl mid" x="106" y="76">Q vectors</text>
  <text class="t-dim mid" x="106" y="92">every layer, every token</text>
  <rect class="box" x="222" y="56" width="180" height="46" rx="6"/>
  <text class="t-lbl mid" x="312" y="76">Sign and fold</text>
  <text class="t-dim mid" x="312" y="92">[n−2, 1, 1] per KV head</text>
  <rect class="box weights" x="428" y="56" width="196" height="46" rx="6"/>
  <text class="t-lbl mid" x="526" y="76">WideQSig, 192 B a token</text>
  <text class="t-dim mid" x="526" y="92">3 groups × 4 heads × 128 bits</text>
  <rect class="box" x="16" y="140" width="140" height="50" rx="6"/>
  <text class="t-lbl mid" x="86" y="161">GPU scan</text>
  <text class="t-dim mid" x="86" y="178">XNOR + popcount</text>
  <rect class="box" x="172" y="140" width="140" height="50" rx="6"/>
  <text class="t-lbl mid" x="242" y="161">Votes</text>
  <text class="t-dim mid" x="242" y="178">best 8, z × margin</text>
  <rect class="box" x="328" y="140" width="140" height="50" rx="6"/>
  <text class="t-lbl mid" x="398" y="161">Normalize</text>
  <text class="t-dim mid" x="398" y="178">vs its own background</text>
  <rect class="box weights" x="484" y="140" width="140" height="50" rx="6"/>
  <text class="t-lbl mid" x="554" y="161">Pool → window</text>
  <text class="t-dim mid" x="554" y="178">indexer fills window</text>
  <path class="ax" d="M196 79 H222 M402 79 H428 M526 102 V120 H86 V140 M156 165 H172 M312 165 H328 M468 165 H484"/>
  <path class="t-dim" d="M222 79 l-6 -3.5 v7 z M428 79 l-6 -3.5 v7 z M86 140 l-3.5 -6 h7 z M172 165 l-6 -3.5 v7 z M328 165 l-6 -3.5 v7 z M484 165 l-6 -3.5 v7 z"/>
  <text class="t-dim" x="96" y="115">gallery: resident in VRAM, 32-token pages · probe: the last 128 decoded tokens</text>
  <text class="t-dim" x="16" y="214">each file's best-agreeing token gives it an agreement, read against the gallery's spread</text>
</svg>
</figure>

### 6.3 Scoring: Every Probe Token Votes

Per probe token and fold group, each candidate's best-agreeing token gives it an agreement, the agreements are read against their distribution over the gallery, and each candidate the token picks gets a vote of `z × margin`. A candidate's raw score is its summed votes. Four findings set the formula (§9.12 states and measures it):

- **Vote for the best several.** One vote per token gave any vote to only 20–67 of 6,854 files, and seven answer files scored exactly zero. Voting for the best eight, each vote square-rooted, raised the answer file's MRR from 0.508 to 0.713. Tool selection keeps one vote: a tool probe has one clean match.
- **Only the strongest tokens count.** A needle gate keeps the quarter of probe tokens with the largest summed vote; most are function words and template.
- **Probes find their own kind.** Every file is stored with calibration questions shaped like the probes that will ask for it, scored as a separate competition — pooled with content they took 96.5% of the vote mass at 2% of the gallery. For tools they were the largest lever: Tool-1 74.5% → 87.9% (§9.10).
- **Repairing signatures does not work.** Bit centring is a no-op, and marking down hubs marks down the files conversations ask about.

### 6.4 Normalization: A Score Read Against Its Own Background

A generic file — the workspace root, a crate's `lib.rs` — scores on almost every probe, so on a shared scale hubs crowd out the answer. No gallery-only correction works (self-match, inter-candidate affinity and cross-probe z-scores all failed), because loudness is a response to the *query* distribution (`docs/provenance_score_normalization.md` §2). The normalizer learns each candidate's **background** — its scores on probes not about it — from live traffic, and reads a score by its tail:

```
surprise(c | x)   = −ln P(background(c) ≥ x)
normalized(c | x) = 1000 × surprise(c | x) / ln(1 / 0.01)
```

1,000 is a full hit, beyond 99 in 100 of the candidate's background, and under the null every candidate wins equally rarely. Backgrounds are log-binned counts that commute; a probe's own answer is left out; only scoring samples count; a candidate with fewer than eight reads against its scope's pooled background; files compete within their own kind. On 37,284 calibration questions over 1,740 files the answer file's R@1 rose from 0.107 raw to **0.239**, and the largest single winner's share of wins on other files' questions fell from 54.5% to **0.1%** (§7.1 of that document).

**Provenance proposes; the model's attention disposes.** On Flash-Next, a reprojection admits whole conversations — a file's reading, a guide, the dialogue — into a pool several times the attended window; the window is filled with their rounds ranked by how often the model's own indexer selected their blocks, or would have (a shadow scan). A round scores `base × (1 + h)`, with hit momentum `h ← 0.8 h + d`, so the indexer can bring in what provenance missed and retire what the model never reads (`docs/hit_driven_projection.md` §7). Momentum across a turn's reprojections lifted held-out R@10 from 0.554 to 0.708 over three turns.

### 6.5 Retrieval During Decode: Reprojection, and the Think Block as a Speculative Probe

**Mid-decode reprojection.** Every decoded token's Q signature is captured. Every 64 decoded tokens, or at a line break more than 16 tokens after the last, inside the think block as in the answer, the last 128 decoded tokens' signatures are scanned against the gallery, and the result assembles the next forward's working set, so retrieval follows the reasoning by at most one interval and never blocks a forward (`sequence_config.rs`: `REPROJECT_EVERY_N_TOKENS`, `reproject_max_probe_tokens`, `reproject_in_thinking`).

**The opening.** A question's Q depends on the context it is read in, and every conversation opens on configured background files: one battery question's answer file ranked second of 6,854 alone and fifth against them. So a request is read twice — first with the background held out, scored and discarded; then again, its opening working set ranked by the first reading beside the background. The first reasoning tokens then reproject at 8, 16 and 32 tokens, because what the model writes early about what it has read stays in its context. The cost is one extra prefill and one scan before the first token.

**The think block is the speculative probe.** A probe that decodes ahead of the answer, never enters memory, and chooses the answer's context is what the reasoning block already is. Each boundary inside it reprojects from what it has just reasoned, the reprojection on `</think>` assembles the answer's working set from the whole thought, and at seal its K/V is dropped and the exchange re-prefilled without it (§5). By the first answer token the context has been steered by everything the model thought. From the projection records of the 128-question battery (§9.12):

<figure class="fig">
<svg viewBox="0 0 640 200" role="img" aria-label="Figure 6-2. Share of 384 answers per arm with an answer file in the working set at the opening and at the first answer token. C1, reprojecting through its reasoning, 51.6 percent to 57.8 percent, with 17.1 reprojections inside the think block and 25.8 percent of the openings' misses brought in. B1, held, 51.6 to 51.8 percent. D1, retrieved once, 48.7 percent throughout.">
  <text class="ttl" x="16" y="20">Figure 6-2 · The reasoning brings in the answer's file</text>
  <text class="ttl-sub" x="16" y="36">answer file in the working set · faint: at the opening · solid: at the first answer token</text>
  <path class="grid" d="M300 52 V168 M400 52 V168 M500 52 V168 M600 52 V168"/>
  <path class="grid base" d="M200 52 V168"/>
  <text class="cat" x="16" y="72">C1, reprojecting</text>
  <text class="cat-sub" x="16" y="85">17.1 in thinking · 25.8% of misses brought in</text>
  <path class="line-us" d="M432 74 H556"/>
  <circle class="us-soft" cx="432" cy="74" r="5.5"/>
  <circle class="dot-us" cx="556" cy="74" r="6.5"/>
  <text class="t-dim mid" x="432" y="64">51.6%</text>
  <text class="v-us mid" x="556" y="63">57.8%</text>
  <text class="cat" x="16" y="112">B1, held</text>
  <text class="cat-sub" x="16" y="125">0.6 in thinking · 6.5%</text>
  <circle class="us-soft" cx="432" cy="114" r="5.5"/>
  <circle class="dot-us" cx="436" cy="114" r="6.5"/>
  <text class="t-dim" x="448" y="118">51.6% → 51.8%</text>
  <text class="cat" x="16" y="152">D1, retrieve once</text>
  <text class="cat-sub" x="16" y="165">0 in thinking · 0%</text>
  <circle class="dot-us" cx="374" cy="154" r="6.5"/>
  <text class="t-dim" x="386" y="158">48.7% throughout</text>
  <text class="tick mid" x="200" y="186">40%</text>
  <text class="tick mid" x="300" y="186">45%</text>
  <text class="tick mid" x="400" y="186">50%</text>
  <text class="tick mid" x="500" y="186">55%</text>
  <text class="tick mid" x="600" y="186">60%</text>
</svg>
</figure>

B1's 0.6 is the reprojection on `</think>` itself, stamped on the block's last token. The bound of §11.2 is unchanged by how often the working set is updated: each update selects at most B tokens, so reprojection changes the aim of updates, not the bound. On attention-only models each reprojection re-resolves its structural segments (§5); on the recurrent production model they are sealed and cost nothing. Re-selecting during reasoning lets later reasoning be decoded against what the earlier reasoning reached for; §9.12 measures what that is worth.

### 6.6 The Production Signature and the Resident Gallery

`WideQSig` folds each token's `sign(Q)` over every attention layer into three groups — all but the top two layers, then each of the top two alone (`[n − 2, 1, 1]`, `FoldParams::derive`), each layer bit-rotated before the XOR so correlated layers do not cancel — of one 128-bit signature per KV head: 3 × 4 × 128 bits, 192 bytes a token. It is the successor of v1's two-layer fold (§9.10), over every layer and with no K term. Signatures cover content tokens only, never the template's self-similar structural tokens, and are persisted beside the turn's K/V at its seal.

Re-uploading signatures every scan was the first GPU implementation's bottleneck, so they stay **resident in VRAM** in a paged gallery arena modelled on the KV cache — 32-token pages, a free-list pool, a pre-resolved pointer table the kernel reads in place. Its warm and cold tiers are the substrate's persisted signatures. Three scan backends are **bit-identical** under an adversarial parity test: a scalar reference; 1-bit tensor cores (`BMMA.88128.XOR.POPC`, sm_75–sm_89), ~9.5 ms on the RTX 4090 Laptop GPU; and INT8 tensor cores (`mma.m16n8k32.s8`), the production path on Blackwell, which dropped the 1-bit instruction.

---

## 7. Unbounded Three-Tier Paged Context

The context is paged in 32-token blocks aligned to turn and system-prompt boundaries, so sinks stay block-local, refresh triggers on clean boundaries and promotion moves whole units, across three tiers (Figure 7-1). Each block's KV heads choose their K and V formats at seal, recorded in the descriptor tables the paged attention kernels read (`ArenaTable` / `PerHeadTable`), one read per tile.

<figure class="fig">
<svg viewBox="0 0 640 214" role="img" aria-label="Figure 7-1. The three tiers. VRAM hot holds the working set, attended at full speed. RAM warm holds recent history in CPU arenas, promoted to hot on demand. Disk cold is the append-only redo log holding every sealed turn's K/V, tokens and signatures. Selected blocks are promoted to hot; hot blocks migrate to warm on the persistence thread; every sealed turn is written to the log. Under full attention every block is read every step; under provenance selection a step reads at most its budget, from any tier.">
  <text class="ttl" x="16" y="20">Figure 7-1 · Three tiers, one budget</text>
  <text class="ttl-sub" x="16" y="36">a step reads at most its budget of blocks, from any tier, however deep the history</text>
  <rect class="box weights" x="16" y="58" width="184" height="78" rx="6"/>
  <text class="t-lbl mid" x="108" y="80">VRAM · hot</text>
  <text class="t-dim mid" x="108" y="98">the working set</text>
  <text class="t-dim mid" x="108" y="114">attended at full speed</text>
  <rect class="box" x="228" y="58" width="184" height="78" rx="6"/>
  <text class="t-lbl mid" x="320" y="80">RAM · warm</text>
  <text class="t-dim mid" x="320" y="98">recent history, CPU arenas</text>
  <text class="t-dim mid" x="320" y="114">promoted on demand</text>
  <rect class="box" x="440" y="58" width="184" height="78" rx="6"/>
  <text class="t-lbl mid" x="532" y="80">Disk · cold</text>
  <text class="t-dim mid" x="532" y="98">append-only redo log</text>
  <text class="t-dim mid" x="532" y="114">K/V, tokens, signatures</text>
  <path class="ax" d="M200 90 H228 M412 90 H440 M228 108 H200 M440 108 H412"/>
  <path class="t-dim" d="M228 90 l-6 -3.5 v7 z M440 90 l-6 -3.5 v7 z M200 108 l6 -3.5 v7 z M412 108 l6 -3.5 v7 z"/>
  <text class="t-dim" x="16" y="160">→ hot blocks migrate down on the persistence thread; every sealed turn is logged</text>
  <text class="t-dim" x="16" y="176">← provenance-selected blocks are promoted to hot (elevate_to_hot)</text>
  <text class="t-dim" x="16" y="198">full attention reads every block every step · provenance selection reads at most B</text>
</svg>
</figure>

**Cold blocks are read only when selected.** Under full attention the oldest blocks are read by every later token. Under provenance selection a cold block is read only by the steps that select it, and the budget caps how many any step takes. A relevant block is still selected at any depth and contributes its full error, so cold quality sets part of the theorem's constant — which is why blocks reach cold storage quantized from refreshed, prefill-quality activations.

**The log stores K/V, not tokens.** Re-prefilling tokens does not reproduce the stored K/V under provenance selection, because the original activations were computed against the working set selected at the time, so a token-only store would raise ε_hot. The redo log (`substrate/substrate.log`) holds each sealed turn's interior K/V, its tokens and its signatures; structural tokens are rebuilt live (§5), and the tokens are a recovery path for a turn whose K/V is lost. A substrate is valid only for the model whose K/V it holds (§9.13). Warm residency is pageable CPU memory, so warm→hot runs at roughly half the PCIe bandwidth a pinned pool would reach.

---

## 8. Native Quantized Inference Kernels

### 8.1 The OOM Problem That Forced the Solution

Standard GEMM libraries (cuBLAS, CUTLASS default paths) dequantise weight matrices to BF16 before execution. For a 30B model during prefill:

```
Quantized weights:    ~15GB stored  (Q4)
Dequantized BF16:     ~60GB materialised
```

On 16 GB this is an immediate OOM. The matmul kernels therefore dequantise inside the MMA, in registers, and never materialise the BF16 copy. The saving holds on every card: weights stay at their stored size, and the memory a dequantised path would spend is left to the context, which the KV cache's own quantization (§5) then multiplies (§9.6.9).

### 8.2 Greedy Decomposition

Mixed prefill and decode put anywhere from 1 to 500+ tokens on an expert per wave step. GEMV is best at 1–4 tokens and pipelined GEMM from 128; greedy decomposition tiles any count exactly into the available tile sizes, each run by the kernel best for it, with no remainder and no padding (Figure 8-1), so throughput is smooth from 1 to 500 tokens.

<figure class="fig">
<svg viewBox="0 0 640 120" role="img" aria-label="Figure 8-1. Greedy decomposition of 500 tokens: 256 and 128 by pipelined GEMM, 64 and 32 by GEMM, 16 and 4 by batched GEMV, exactly 500 with no padding.">
  <text class="ttl" x="16" y="20">Figure 8-1 · 500 tokens, tiled exactly</text>
  <text class="ttl-sub" x="16" y="36">each tile by the kernel best for its size · no remainder, no padding</text>
  <rect class="b0" x="16" y="50" width="306.2" height="30"/>
  <rect class="b0" x="323.2" y="50" width="152.6" height="30"/>
  <rect class="b1" x="476.8" y="50" width="75.8" height="30"/>
  <rect class="b1" x="553.6" y="50" width="37.4" height="30"/>
  <rect class="b2" x="592" y="50" width="18.2" height="30"/>
  <rect class="b2" x="611.2" y="50" width="4.8" height="30"/>
  <text class="tick mid" x="169.1" y="96">256</text>
  <text class="tick mid" x="400" y="96">128</text>
  <text class="tick mid" x="514.7" y="96">64</text>
  <text class="tick mid" x="572.8" y="96">32</text>
  <text class="tick" x="586" y="110">16 + 4</text>
  <g class="legend">
    <rect class="b0" x="16" y="102" width="10" height="9"/><text x="30" y="110">pipelined GEMM</text>
    <rect class="b1" x="140" y="102" width="10" height="9"/><text x="154" y="110">GEMM</text>
    <rect class="b2" x="210" y="102" width="10" height="9"/><text x="224" y="110">batched GEMV</text>
  </g>
</svg>
</figure>

### 8.3 Fused Sampling Kernel

Temperature, top-k, top-p, repetition, frequency and presence penalties, DRY and min-p run in one kernel: the logits are read once into shared memory and the sampled index written back. One kernel per modifier moves ~12 MB per token; the fused kernel moves ~600 KB. Templates specialise the common combinations, and a generic path handles the rest.

---

## 9. Experimental Evaluation

### 9.1 Evaluation Strategy

No published engine combines unbounded context, per-block KV quantization, provenance indexing and predicted expert streaming, so no system runs the same workload. The evaluation instead (1) sets every throughput and compression row against the best published figure for the same model and card class — llama.cpp, vLLM, KTransformers, SGLang — saying where ours is aggregate and theirs single-stream (§9.6); (2) validates the output of every measured row, reporting it only if every session reproduces its reference (§9.8); and (3) ablates each contribution: the expert cache (§4.3, §9.7), selection at depth (§9.11.2), and seven arms over the system's own codebase that remove the memory, the seed files and reprojection in turn (§9.12).

### 9.2 Hardware

Three machines, each measured separately; the engine sizes its VRAM partition from the card it finds.

| | RTX 4090 Laptop GPU | RTX 3090 | RTX PRO 5000 Blackwell |
|---|---|---|---|
| VRAM | 16 GB | 24 GB | 72 GB |
| Architecture | Ada, sm_89, native FP8 | Ampere, sm_86, **no native FP8** | Blackwell, sm_120 |
| Host | Core Ultra 9 185H, **32 GB RAM** | i7-10700K, 64 GB RAM | Ryzen 9 9950X3D, 189 GB RAM |
| Host link | PCIe 4.0, **trained ×8**, ~12 GB/s | **PCIe 3.0 ×16**, ~12 GB/s | PCIe 5.0 ×16 |
| Driver model | WDDM | WDDM (GeForce: no TCC) | WDDM for the 2026-10-10 sweep; TCC from 2026-11 |
| Measured | width ladder, engine probe | width ladder | width ladder and depth to 128K |

The laptop is the design constraint; the RTX 3090 runs the integer-only path (no native FP8, neither tensor-core provenance backend) over a link no faster than the laptop's; the Blackwell card holds every model but the 284B resident and runs the depth sweeps. §9.6's figures are WDDM figures except where a row names the TCC build; WDDM demotes VRAM past capacity and charges system commit for every allocation, TCC refuses and charges nothing (`docs/driver_model.md`). Every row is a test in this repository, run one model per `cargo` process with the card to itself; the commands and source rows are in `README.md` and `docs/performance.md`.

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

The weight formats (**qmatmul**) and the KV formats share one block structure and dispatch; the KV-only formats (Q8_KS, Q4_KS, Q4_1, Q3_0, Q2_0) are validated by the per-block selector instead, since KV quality is attention-output fidelity, not reconstruction error. Figure 9.4-1 measures the weight kernels on the RTX 4090 Laptop GPU, 200 iterations a format under Criterion.rs, with no BF16 weight copy at any point. FP8 data formats ran at F16's ~80.5 GiB/s with no gain, and their non-uniform spacing suits block-normalised KV worse than integer formats, so FP8 left the KV ladder; E4M3 survives only as the `_KS` sub-block scale.

<figure class="fig">
<svg viewBox="0 0 640 302" role="img" aria-label="Figure 9.4-1. Native quantized matmul bandwidth on the RTX 4090 Laptop GPU, GiB per second. F32 84.5, F16 82.7. Q8_0 59.7, Q8_1 59.4, Q8_K 59.6, Q6_K 57.9, Q5_0 58.9, Q5_1 59.8, Q5_K 59.0, Q4_0 59.9, Q4_1 59.6, Q4_K 59.3. Q3_K 34.0, Q2_K 35.0.">
  <text class="ttl" x="16" y="20">Figure 9.4-1 · Quantized matmul bandwidth, no weight copy</text>
  <text class="ttl-sub" x="16" y="36">RTX 4090 Laptop GPU · GiB/s · three tiers: float, integer ≥4-bit, sub-4-bit packing</text>
  <path class="grid" d="M333.3 48 V276 M466.7 48 V276 M600 48 V276"/>
  <path class="grid base" d="M200 48 V276"/>
  <text class="tick" x="190" y="61" text-anchor="end">F32</text><rect class="b0" x="200" y="52" width="375.5" height="11"/><text class="tick" x="581.5" y="61">84.5</text>
  <text class="tick" x="190" y="77" text-anchor="end">F16</text><rect class="b0" x="200" y="68" width="367.5" height="11"/><text class="tick" x="573.5" y="77">82.7</text>
  <text class="tick" x="190" y="93" text-anchor="end">Q8_0</text><rect class="b1" x="200" y="84" width="265.3" height="11"/><text class="tick" x="471.3" y="93">59.7</text>
  <text class="tick" x="190" y="109" text-anchor="end">Q8_1</text><rect class="b1" x="200" y="100" width="264" height="11"/><text class="tick" x="470" y="109">59.4</text>
  <text class="tick" x="190" y="125" text-anchor="end">Q8_K</text><rect class="b1" x="200" y="116" width="264.9" height="11"/><text class="tick" x="470.9" y="125">59.6</text>
  <text class="tick" x="190" y="141" text-anchor="end">Q6_K</text><rect class="b1" x="200" y="132" width="257.3" height="11"/><text class="tick" x="463.3" y="141">57.9</text>
  <text class="tick" x="190" y="157" text-anchor="end">Q5_0</text><rect class="b1" x="200" y="148" width="261.8" height="11"/><text class="tick" x="467.8" y="157">58.9</text>
  <text class="tick" x="190" y="173" text-anchor="end">Q5_1</text><rect class="b1" x="200" y="164" width="265.8" height="11"/><text class="tick" x="471.8" y="173">59.8</text>
  <text class="tick" x="190" y="189" text-anchor="end">Q5_K</text><rect class="b1" x="200" y="180" width="262.2" height="11"/><text class="tick" x="468.2" y="189">59.0</text>
  <text class="tick" x="190" y="205" text-anchor="end">Q4_0</text><rect class="b1" x="200" y="196" width="266.2" height="11"/><text class="tick" x="472.2" y="205">59.9</text>
  <text class="tick" x="190" y="221" text-anchor="end">Q4_1</text><rect class="b1" x="200" y="212" width="264.9" height="11"/><text class="tick" x="470.9" y="221">59.6</text>
  <text class="tick" x="190" y="237" text-anchor="end">Q4_K</text><rect class="b1" x="200" y="228" width="263.5" height="11"/><text class="tick" x="469.5" y="237">59.3</text>
  <text class="tick" x="190" y="253" text-anchor="end">Q3_K</text><rect class="b2" x="200" y="244" width="151.1" height="11"/><text class="tick" x="357.1" y="253">34.0</text>
  <text class="tick" x="190" y="269" text-anchor="end">Q2_K</text><rect class="b2" x="200" y="260" width="155.6" height="11"/><text class="tick" x="361.6" y="269">35.0</text>
  <text class="tick mid" x="200" y="292">0</text>
  <text class="tick mid" x="333.3" y="292">30</text>
  <text class="tick mid" x="466.7" y="292">60</text>
  <text class="tick mid" x="600" y="292">90 GiB/s</text>
</svg>
</figure>

Q3_K and Q2_K run at ~42% of peak, the cost of non-power-of-two packing.

### 9.5 Quantized Matmul Kernel Accuracy

Each kernel against a reference on a 2048×2048 matrix, batch 1–128; every format passes at every batch size. Max relative error at batch 1 and 128, and mean absolute difference at 128:

| Accumulator | Q8_0 | Q4_0 | Q6_K |
|---|---|---|---|
| F32 | 0.0016 / 0.0077 / 0.000168 | 0.0025 / 0.0050 / 0.000143 | 0.0031 / 0.0083 / 0.000202 |
| BF16 | 0.0267 / 0.0642 / 0.001952 | 0.0195 / 0.0447 / 0.001651 | 0.0239 / 0.0573 / 0.001921 |
| F8E4M3 | 0.0625 / 0.0944 / 0.010276 | 0.0596 / 0.0826 / 0.010126 | 0.0743 / 0.0918 / 0.010324 |

Against F32 the mean differences sit at 0.00013–0.00022, inside each format's quantization noise. The larger BF16 and E4M3 figures are those formats' 7- and 3-bit mantissas, not kernel error, and max error grows with batch as rounding accumulates over longer reductions.

### 9.6 Throughput, Depth and Compression Across the Fleet

Every figure here is a test in this repository — the width ladder (`test_parallel_batched_forwarding*`, a ~700-token prompt at several widths and compression levels), the depth gate (`long_context_gate`, 4K–128K), or the engine probe (`kv_fragmentation`, the full conversation engine under load) — and every row validates each session's output (§9.8). Sources are in `docs/performance.md` and `docs/results/`. **Prefill** is prompt tokens per second; **decode** is generated tokens per second, **summed across the batch** where a row has more than one context.

#### 9.6.1 Width: what concurrency buys on one card

RTX PRO 5000 Blackwell 72 GB, build `21c36efbd` (2026-10-10): fourteen width gates, 212 rows. Each cell is the model's best validated row, with the format and width that produced it.

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

Prefill saturates early while aggregate decode scales nearly linearly with width: Qwen3.5-35B-A3B goes from 230.2 t/s at one session to **2,869.9 t/s across sixty-four** (12.5×) and Qwen3.6-35B-A3B from 476.6 to **2,761.0** (5.8×), at C10 with prefill flat. This is §4.4's wave batching measured: one expert load serves every session routed to it.

#### 9.6.2 Depth: throughput does not fall with context

<figure class="fig">
<svg viewBox="0 0 640 258" role="img" aria-label="Figure 9.6-1. Share of the 32K rate kept at 128K, one context, BF16, RTX PRO 5000. Qwen3.8-Flash-Next keeps 99 percent of prefill and 111 percent of decode. Qwen3.8-27B 28 and 58. Qwen3.5-9B 30 and 48. Qwen3.6-35B-A3B 26 and 49. Qwen3.5-35B-A3B 26 and 47. Qwen3.5-0.8B 25 and 86.">
  <text class="ttl" x="16" y="20">Figure 9.6-1 · What a 4× deeper context costs</text>
  <text class="ttl-sub" x="16" y="36">share of the 32K rate kept at 128K · faint bar prefill, solid bar decode · RTX PRO 5000, BF16</text>
  <path class="grid" d="M360 50 V232"/>
  <path class="grid base" d="M200 50 V232"/>
  <path class="parity" d="M520 50 V232"/>
  <text class="cat" x="16" y="66">Qwen3.8-Flash-Next</text>
  <text class="cat-sub" x="16" y="79">sparse-attention indexer</text>
  <rect class="us-soft" x="200" y="56" width="316.8" height="10"/>
  <rect class="us" x="200" y="68" width="355.2" height="10"/>
  <text class="t-dim" x="522.8" y="65">99%</text>
  <text class="v-us" x="561.2" y="78">111%</text>
  <text class="cat" x="16" y="96">Qwen3.8-27B</text>
  <rect class="us-soft" x="200" y="86" width="89.6" height="10"/>
  <rect class="us" x="200" y="98" width="185.6" height="10"/>
  <text class="t-dim" x="295.6" y="95">28%</text>
  <text class="t-dim" x="391.6" y="107">58%</text>
  <text class="cat" x="16" y="126">Qwen3.5-9B</text>
  <rect class="us-soft" x="200" y="116" width="96" height="10"/>
  <rect class="us" x="200" y="128" width="153.6" height="10"/>
  <text class="t-dim" x="302" y="125">30%</text>
  <text class="t-dim" x="359.6" y="137">48%</text>
  <text class="cat" x="16" y="156">Qwen3.6-35B-A3B</text>
  <rect class="us-soft" x="200" y="146" width="83.2" height="10"/>
  <rect class="us" x="200" y="158" width="156.8" height="10"/>
  <text class="t-dim" x="289.2" y="155">26%</text>
  <text class="t-dim" x="362.8" y="167">49%</text>
  <text class="cat" x="16" y="186">Qwen3.5-35B-A3B</text>
  <rect class="us-soft" x="200" y="176" width="83.2" height="10"/>
  <rect class="us" x="200" y="188" width="150.4" height="10"/>
  <text class="t-dim" x="289.2" y="185">26%</text>
  <text class="t-dim" x="356.4" y="197">47%</text>
  <text class="cat" x="16" y="216">Qwen3.5-0.8B</text>
  <rect class="us-soft" x="200" y="206" width="80" height="10"/>
  <rect class="us" x="200" y="218" width="275.2" height="10"/>
  <text class="t-dim" x="286" y="215">25%</text>
  <text class="t-dim" x="481.2" y="227">86%</text>
  <text class="tick mid" x="200" y="248">0%</text>
  <text class="tick mid" x="360" y="248">50%</text>
  <text class="tick mid" x="520" y="248">100%</text>
</svg>
</figure>

Flash-Next is the one model whose cost per token does not depend on history, and the reason is its checkpoint, not this engine's provenance: its full-attention layers carry a trained sparse-attention indexer (QSA) that caps each token's attended set at 2,051 positions, where the hybrids read the whole prefix. The attention kernel costs the same at shallow depth on both (0.382 ms at 32K on Flash-Next, 0.394 ms at 8K on Qwen3.5-35B), then stays within 6% to 128K on Flash-Next and grows 3.71× for 4× depth on the 35B. Qwen3.5-0.8B keeps 86% of decode because its step is dominated by depth-independent work. Flash-Next's speculative head accepts exactly **4.85 tokens a step at every depth from 8K to 128K**, and the story is reproduced from behind 128,897 tokens of padding. The published llama.cpp run of the same model on an RTX 4090 24 GB keeps 64% of prefill and 65% of decode from 6K to 110K [31], so the engine adds no depth-dependent cost of its own. This shows the principle §11.2 rests on — a working set bounded independently of depth costs the same at any depth — by a model-native mechanism inside one context; it is not a measurement of provenance selection.

#### 9.6.3 Compression: the top level, inline, validated

Each 32-token block is compressed in the forward that produced it. C10 against BF16 at 8K, one context, Blackwell (`docs/performance.md` §3.4):

| Model | Prefill BF16 / C10 (t/s) | Decode BF16 / C10 (t/s) | C10 compression |
|---|---|---|---:|
| Qwen3-8B | 3,235.1 / 3,212.7 | 42.8 / 41.8 | 6.31× |
| Qwen3-30B-A3B | 3,444.7 / 3,367.6 | 51.2 / 47.2 | 5.94× |
| Llama-2-7B (4K) | 3,684.7 / 3,851.1 | 47.1 / 44.1 | 4.81× |
| Llama-3.2-3B | 6,195.9 / 5,936.8 | 69.9 / 66.1 | 4.62× |

At 8K the top level costs **0–8% of decode for 3.3–6.3×**. At 32K the largest hybrids pay 27–31% for 6.3–7.5×; at 128K the cache compresses 4.63–7.63× with every block quantized, Flash-Next paying 19% for 6.97×. C10 prefill runs within 5% of BF16 at every depth and is marginally *faster* at 128K, with fewer bytes to read. llama.cpp's q8_0 cache gives ~1.9× for 35% of decode at 32K and 45% at 64K, its q4_0 ~3.56× for 39% and 50% [35]; vLLM's FP8 gives 2× [36], TurboQuant 2.4–3.4× for 20–27% [37]. At 32K the top level holds **nearly 4× q8_0's compression while slowing decode less**.

#### 9.6.4 Against the published record

Figure 9.6-2 sets our aggregate decode, serving concurrent sessions, against the best published llama.cpp single-stream decode for the same model on the same class of card; the Blackwell rows compare our RTX PRO 5000 with published RTX 5090 figures. Aggregate against single-stream is the comparison the record allows — no aggregate for these models on these cards has been published — and it answers how much work one card does. Published single-card serving of the 35B models on an RTX PRO 6000 reports 5–10 concurrent requests [42, 43]; this engine runs them at 64, and Qwen3.5-0.8B at 256.

<figure class="fig">
<svg viewBox="0 0 640 494" role="img" aria-label="Figure 9.6-2. Our aggregate decode against the best published llama.cpp single-stream decode, same model and card class, on a log scale. RTX 3090: Qwen3.5-35B-A3B 11.1 times, Qwen3.6-35B-A3B 8.5, Llama-2-7B 5.8, Qwen3-8B 3.3, Qwen3.8-27B 5.5, Qwen3-30B-A3B 2.6, Flash-Next 36.3. 16 GB: Qwen3.6-35B-A3B 1.3 to 1.8, Qwen3-8B 3.0, Flash-Next 3.7. Blackwell: Qwen3.5-35B-A3B 14.8, Qwen3.6-35B-A3B 8.3, Flash-Next 37.9, Llama-2-7B 5.6, Qwen3-30B-A3B 4.4, Qwen3-8B 3.2.">
  <text class="ttl" x="16" y="20">Figure 9.6-2 · One card's aggregate decode, against the published record</text>
  <text class="ttl-sub" x="16" y="36">× the best published llama.cpp single-stream decode, same model and card · log scale</text>
  <path class="grid" d="M313.8 46 V466 M398.1 46 V466 M461.9 46 V466 M525.7 46 V466 M610 46 V466"/>
  <path class="parity" d="M250 46 V466"/>
  <text class="cat" x="16" y="62">Qwen3.5-35B-A3B, C10 ×16</text><text class="cat-sub" x="16" y="74">RTX 3090 · 1,228.8 vs 111.2 [28]</text>
  <path class="line-us" d="M250 64 H471.5"/><circle class="dot-us" cx="471.5" cy="64" r="5"/><text class="v-us" x="481.5" y="69">11.1×</text>
  <text class="cat" x="16" y="88">Qwen3.6-35B-A3B, C10 ×16</text><text class="cat-sub" x="16" y="100">RTX 3090 · 1,346.3 vs 157.66 [33]</text>
  <path class="line-us" d="M250 90 H446.9"/><circle class="dot-us" cx="446.9" cy="90" r="5"/><text class="v-us" x="456.9" y="95">8.5×</text>
  <text class="cat" x="16" y="114">Llama-2-7B, BF16 ×48</text><text class="cat-sub" x="16" y="126">RTX 3090 · 934.9 vs 161.89 [27]</text>
  <path class="line-us" d="M250 116 H411.8"/><circle class="dot-us" cx="411.8" cy="116" r="5"/><text class="v-us" x="421.8" y="121">5.8×</text>
  <text class="cat" x="16" y="140">Qwen3-8B, C8 ×10</text><text class="cat-sub" x="16" y="152">RTX 3090 · 383.8 vs 115.3 [28]</text>
  <path class="line-us" d="M250 142 H359.9"/><circle class="dot-us" cx="359.9" cy="142" r="5"/><text class="v-us" x="369.9" y="147">3.3×</text>
  <text class="cat" x="16" y="166">Qwen3.8-27B, C8 ×20</text><text class="cat-sub" x="16" y="178">RTX 3090 · 358.7 vs 65.28 with MTP [30]</text>
  <path class="line-us" d="M250 168 H406.9"/><circle class="dot-us" cx="406.9" cy="168" r="5"/><text class="v-us" x="416.9" y="173">5.5×</text>
  <text class="cat" x="16" y="192">Qwen3-30B-A3B, BF16 ×10</text><text class="cat-sub" x="16" y="204">RTX 3090 · 400.5 vs 153.6 [28]</text>
  <path class="line-us" d="M250 194 H337.9"/><circle class="dot-us" cx="337.9" cy="194" r="5"/><text class="v-us" x="347.9" y="199">2.6×</text>
  <text class="cat" x="16" y="218">Flash-Next, BF16 ×16</text><text class="cat-sub" x="16" y="230">RTX 3090 · 543.8 vs 15 at ~130K [32]</text>
  <path class="line-us" d="M250 220 H580.5"/><circle class="dot-us" cx="580.5" cy="220" r="5"/><text class="v-us" x="570.5" y="214" text-anchor="end">36.3×</text>
  <text class="cat" x="16" y="244">Qwen3.6-35B-A3B, C10 ×16</text><text class="cat-sub" x="16" y="256">16 GB · 336.7 vs 183.3 / 249.3 MTP [38]</text>
  <path class="line-us" d="M250 246 H304.1"/><circle class="dot-us" cx="274.1" cy="246" r="4"/><circle class="dot-us" cx="304.1" cy="246" r="5"/><text class="v-us" x="314.1" y="251">1.3–1.8×</text>
  <text class="cat" x="16" y="270">Qwen3-8B, C8 ×10</text><text class="cat-sub" x="16" y="282">16 GB · 308.4 vs 102.7, RTX 4080 [29]</text>
  <path class="line-us" d="M250 272 H351.1"/><circle class="dot-us" cx="351.1" cy="272" r="5"/><text class="v-us" x="361.1" y="277">3.0×</text>
  <text class="cat" x="16" y="296">Flash-Next, BF16 ×8</text><text class="cat-sub" x="16" y="308">16 GB · 108.5 vs 27.5–29, RTX 5080 [32]</text>
  <path class="line-us" d="M250 298 H370.4"/><circle class="dot-us" cx="370.4" cy="298" r="5"/><text class="v-us" x="380.4" y="303">3.7×</text>
  <text class="cat" x="16" y="322">Qwen3.5-35B-A3B, C10 ×64</text><text class="cat-sub" x="16" y="334">Blackwell · 2,869.9 vs 194.0 [34]</text>
  <path class="line-us" d="M250 324 H498"/><circle class="dot-us" cx="498" cy="324" r="5"/><text class="v-us" x="508" y="329">14.8×</text>
  <text class="cat" x="16" y="348">Qwen3.6-35B-A3B, C10 ×64</text><text class="cat-sub" x="16" y="360">Blackwell · 2,761.0 vs 333.6 MTP [38]</text>
  <path class="line-us" d="M250 350 H444.8"/><circle class="dot-us" cx="444.8" cy="350" r="5"/><text class="v-us" x="454.8" y="355">8.3×</text>
  <text class="cat" x="16" y="374">Flash-Next, BF16 ×16, TCC</text><text class="cat-sub" x="16" y="386">Blackwell · 1,819.3 vs 48.02 [42]</text>
  <path class="line-us" d="M250 376 H584.5"/><circle class="dot-us" cx="584.5" cy="376" r="5"/><text class="v-us" x="574.5" y="370" text-anchor="end">37.9×</text>
  <text class="cat" x="16" y="400">Llama-2-7B, BF16 ×48</text><text class="cat-sub" x="16" y="412">Blackwell · 1,675.0 vs 300.40 [27]</text>
  <path class="line-us" d="M250 402 H408.5"/><circle class="dot-us" cx="408.5" cy="402" r="5"/><text class="v-us" x="418.5" y="407">5.6×</text>
  <text class="cat" x="16" y="426">Qwen3-30B-A3B, BF16 ×10</text><text class="cat-sub" x="16" y="438">Blackwell · 997.4 vs 226.1 [39]</text>
  <path class="line-us" d="M250 428 H386.4"/><circle class="dot-us" cx="386.4" cy="428" r="5"/><text class="v-us" x="396.4" y="433">4.4×</text>
  <text class="cat" x="16" y="452">Qwen3-8B, C8 ×10</text><text class="cat-sub" x="16" y="464">Blackwell · 632.9 vs 200.4 [39]</text>
  <path class="line-us" d="M250 454 H357"/><circle class="dot-us" cx="357" cy="454" r="5"/><text class="v-us" x="367" y="459">3.2×</text>
  <text class="tick mid" x="250" y="484">1×</text>
  <text class="tick mid" x="313.8" y="484">2×</text>
  <text class="tick mid" x="398.1" y="484">5×</text>
  <text class="tick mid" x="461.9" y="484">10×</text>
  <text class="tick mid" x="525.7" y="484">20×</text>
  <text class="tick mid" x="610" y="484">50×</text>
</svg>
</figure>

Two rows are not like-for-like. The RTX 3090 Flash-Next row is against a published run at ~130K context on UD-Q4_K_XL, where ours is a ~700-token prompt on Q2_KO experts; at a matched short prompt it is 37.9× on Blackwell and 3.7× at 16 GB. The 16 GB Qwen3.6-35B row is the weakest on purpose: a 3-bit quant that fits wholly in VRAM decodes one stream faster than streamed Q6_K experts can, so the honest multiple there is 1.3–1.8×.

#### 9.6.5 The 16GB laptop

The RTX 4090 Laptop GPU with 32 GB of RAM is the constraint the architecture was designed under.

- **A 180B model.** Qwen3.8-Flash-Next (a 125B trunk of 512 experts, a 51B n-gram table and a 4B MTP head; 6B active) runs from a ~88 GB Q2_KO-expert artifact, streaming VRAM → pinned RAM → NVMe, every rung validating 8/8. One session prefills at 269.1 t/s and decodes at 34.9; eight decode at **108.5 t/s aggregate**, and C10 at 5.79–5.82× costs 6.6% of that. Every published single-GPU run that states its host uses 64–128 GB of RAM: an RTX 5090 with 128 GB decodes one stream at 48.02 t/s [42], an RTX 5080 with 64 GB at 27.5–29 [32].
- **A 30B MoE streaming its experts** prefills at **6,139.1 t/s** (Qwen3-30B-A3B, Q8_0 ×20).
- **A 35B MoE serving sixteen.** Qwen3.5- and Qwen3.6-35B-A3B at Q6_K, ~28 GB of weights, serve ×16 at **330.8 and 336.7 t/s aggregate**, C10 at 7.01× and 6.42×. Published 16 GB runs are single-stream at quants that fit resident [38].
- **A small model at workstation rate.** Qwen3.5-0.8B at C8 ×32 prefills at **46,056.8 t/s** and decodes at **3,810.7 t/s aggregate**, above the RTX 3090's 42,982.2 on the same rung.

C10 compresses the 35B to 7.01× on the laptop and 7.03× on the RTX 3090: the same selector and thresholds make the same decisions on different GPUs and links.

#### 9.6.6 A 284B model on one GPU

DeepSeek-V4-Flash (284B total, 13B active) on one RTX PRO 5000 72GB, sixteen concurrent sessions: **1,177.1 t/s prefill and 114.4 t/s aggregate decode**, build `21c36efbd`. The best published single-GPU prefill for the model is 748.4 t/s (RTX PRO 6000 Max-Q, all experts on the CPU, 8K) [43], and the best published single-GPU decode is 28 t/s (RTX 5090, KTransformers + SGLang, INT4 experts on the CPU) [44] — every published run holding the routed experts in host RAM, as this one does. The aggregate is 4.1× that decode, and the prefill is above every published single-GPU figure for the model.
#### 9.6.7 Where published engines lead

Single-stream decode on a model that fits wholly in VRAM is llama.cpp's strength: a 3-bit Qwen3.6-35B-A3B resident on a 16 GB card decodes one stream at 183.29 t/s (249.33 with MTP) [38], where our sixteen-session aggregate with Q6_K experts streamed is 336.7 — more work in total, less per stream. vLLM prefills Qwen3.6-35B-A3B at 41,105 t/s on an RTX PRO 6000 [41], against ~7,200 here at one session. Against Strata's RTX 5090 single-session benchmark, replayed through the conversation engine at C5 (`kv_fragmentation::qwen38_flash_next_strata`, median of three), ours leads prefill at 4K — 6,064.3 t/s against 4,269.8 — and trails decode: 168.4 / 159.1 / 125.3 t/s against 179.4 / 175.7 / 165.0 at 4K / 32K / 128K, 82–95% of Strata to 32K and 69–76% at 128K. Strata has a PCIe 5.0 card and a dedicated CUDA graph path; here one segment submission per MoE layer is the floor.

#### 9.6.8 Output fidelity against depth

Throughput says compression is cheap, not how far it moves the output or whether that grows with what a step reads. At each depth a needle prompt — four random codes spread through filler, asked back at the end — is prefilled, sealed at the mode's level, and the answer read back teacher-forced against BF16 over the same prefix (`batch_test::fidelity`; `docs/results/theorem_depth_tests_2026-11-10.md`).

<figure class="fig">
<svg viewBox="0 0 640 246" role="img" aria-label="Figure 9.6-3. Mean KL divergence from BF16, in nats, against context depth, RTX PRO 5000. Flash-Next at C10: 5.9e-2 at 4K, 3.0e-2 at 16K, 1.08e-1 at 64K, 1.16e-1 at 124K. Flash-Next at C5: 9.8e-3, 1.1e-2, 2.1e-2, 7.6e-3. Qwen3.5-9B at C10: 6.5e-3, 1.2e-3, 1.7e-3, 3.5e-3. Qwen3.5-9B at C5: 6.6e-3, 1.0e-3, 9.4e-4, 3.6e-3. Qwen3-8B at C10 to 32K: 3.2e-2, 1.1e-2, 3.9e-2.">
  <text class="ttl" x="16" y="20">Figure 9.6-3 · Divergence from BF16 does not grow with depth</text>
  <text class="ttl-sub" x="16" y="36">mean KL, nats, teacher-forced needle answer · log scales · amber: C10 · green: C5</text>
  <path class="grid" d="M210 197.5 H600 M210 136 H600 M210 74.5 H600"/>
  <path class="grid base" d="M210 52 V216"/>
  <text class="tick" x="204" y="200.5" text-anchor="end">1e-3</text>
  <text class="tick" x="204" y="139" text-anchor="end">1e-2</text>
  <text class="tick" x="204" y="77.5" text-anchor="end">1e-1</text>
  <path class="line-them" d="M220 88.6 L340 106.7 L460 72.5 L517.2 70.5"/>
  <path class="line-us" d="M220 136.5 L340 133.5 L460 116.2 L517.2 143.3"/>
  <path class="line-us dash" d="M220 147.5 L340 192.6 L460 183.3 L517.2 164"/>
  <path class="line-us" opacity=".45" d="M220 147.1 L340 197.5 L460 199.1 L517.2 163.3"/>
  <path class="line-them dash" d="M220 104.9 L340 133.5 L400 99.7"/>
  <text class="t-them" x="525" y="74.5">Flash-Next C10</text>
  <text class="t-flat" x="525" y="147">Flash-Next C5</text>
  <text class="t-flat" x="525" y="160">Qwen3.5-9B C10</text>
  <text class="t-flat" x="525" y="172" opacity=".6">Qwen3.5-9B C5</text>
  <text class="t-them" x="404" y="112">Qwen3-8B C10, to 32K</text>
  <text class="tick mid" x="220" y="232">4K</text>
  <text class="tick mid" x="340" y="232">16K</text>
  <text class="tick mid" x="400" y="232">32K</text>
  <text class="tick mid" x="460" y="232">64K</text>
  <text class="tick mid" x="517.2" y="232">124K</text>
</svg>
</figure>

Top-1 agreement with BF16 is 95.8–100% in every cell, with run-to-run spread 0 on the 8B and 9B and at most 4.3e-5 on Flash-Next. Where attention reads the whole prefix, divergence does not grow across a 30× range of depth: the exposure §11.2 bounds grows, the output error does not (§11.1). On Flash-Next at C5, the level production seals at, it stays flat from 4K to 124K; C10 is the one series that grows, about 2×, the price of its extra compression on a model whose attention selects.

#### 9.6.9 Memory: what keeps a large context on one card

The native kernels keep weights at their stored size (§8.1) and per-block quantization keeps every sealed block at the format its own error allows (§5); both leave the card to the context. On Flash-Next the batched forward gate (`docs/performance.md` §3.6, RTX PRO 5000, every row validated) reads, per level:

| Level | Compression | Prefill ×2 (t/s) | Decode ×2 (t/s) |
|---|---:|---:|---:|
| C0 | 2.22× | 5,033.6 | 468.9 |
| C5 | 4.15× | 5,903.2 | 466.8 |
| C8 | 5.52× | 5,942.2 | 436.2 |
| C10 | 7.14× | 5,936.9 | 429.8 |

From C0 to C10 the cache shrinks 3.2× further for 8% of decode; C5, the level everything is sealed at, holds 4.15× the tokens of BF16 at C0's throughput.

**The substrate.** The dialogue and every ingest layer seal at C5. On 2026-11-23 BlackWorld's redo log held 103 segments, 409 GiB, ≈ 8.0 KB on disk per resident token with every record included; the galleries held 54.7M signature tokens over 29,747 turns and 7,830 code files. The K/V census (`GET /v1/kv/census`, `docs/results/battery_128/census_start_5470.json`) — exact chunk bytes from the chunk index, the sub-band format mix from a 4,096-chunk sample — finds 16,006,522 chunks holding 504.9M layer-tokens, 42.1M tokens over Flash-Next's 12 K/V layers, in 275.3 GiB of K/V standing for 976.7 GiB at BF16: **3.55×**. By lead format 68.7% of chunks are Q4_0, 18.8% Q4_1, 10.5% Q5_0 and 1.3% Q8_0; 46% of the V sub-bands are Q1_S, Q2_A or Q2_S, while no K sub-band falls below Q4_0.

**The span.** The daemon reserves one contiguous span per card and divides it between persistent tensors, KV regions, a per-wave transient tier and expert weights, with an elastic boundary between the KV side and the experts (§10.2; Figure 9.6-4). Behind it the warm tier holds 14.4 GiB of payload in host RAM.

<figure class="fig">
<svg viewBox="0 0 640 128" role="img" aria-label="Figure 9.6-4. BlackWorld's 68.1 GiB span at rest with Flash-Next loaded: expert zone 51.0 GiB, dense weights 5.3, recurrent state 4.9, and 6.9 GiB for the KV side, the transient tier and held index pages.">
  <text class="ttl" x="16" y="20">Figure 9.6-4 · The span at rest</text>
  <text class="ttl-sub" x="16" y="36">BlackWorld, 72 GB card, Flash-Next · 68.1 GiB · the boundary between KV and experts moves under load</text>
  <rect class="b0" x="16" y="50" width="448.4" height="32"/>
  <rect class="b1" x="465.4" y="50" width="45.7" height="32"/>
  <rect class="b2" x="512.1" y="50" width="42.2" height="32"/>
  <rect class="q2" x="555.3" y="50" width="60.7" height="32"/>
  <path class="parity" d="M555.3 44 V88"/>
  <text class="t-lbl" x="24" y="71">Expert zone · 51.0 GiB</text>
  <text class="tick mid" x="488.3" y="100">dense 5.3</text>
  <text class="tick mid" x="533.2" y="114">state 4.9</text>
  <text class="tick mid" x="585.7" y="100">KV side 6.9</text>
  <text class="t-them" x="551" y="126" text-anchor="end">elastic boundary →</text>
</svg>
</figure>

**Under load.** The battery's load is sampled every five seconds from `/v1/telemetry` and `/v1/memory` beside the answers in flight (`docs/results/battery_128/fig_9_6_telemetry.csv`, 6,786 loaded samples). VRAM is the newest wave's accounting by tenant; the tokens are the format census of every VRAM-resident residence, a token counted once across its layers; their BF16 size is 24.9 KB a token, 2,076 bytes a layer over 12 K/V layers. The KV figure is the arenas the span holds, at least the payload in them, so compression against it is a floor.

<figure class="fig">
<svg viewBox="0 0 640 324" role="img" aria-label="Figure 9.6-5. The card's memory by tenant over the battery's 10.2 hours, stacked, in GiB: dense 5.3 GiB on average, 5.3 at peak, experts 36.0 GiB on average, 50.1 at peak, KV 6.2 GiB on average, 11.2 at peak, activations 0.4 GiB on average, 3.7 at peak, state 15.5 GiB on average, 23.1 at peak; with 8.5 answers in flight on average and 16 at most.">
  <text class="ttl" x="16" y="20">Figure 9.6-5 · The card under the battery</text>
  <text class="ttl-sub" x="16" y="38">GiB each tenant of the span held, stacked · answers in flight on the right scale</text>
  <path class="grid" d="M56 250.3 H584 M56 220.6 H584 M56 190.9 H584 M56 161.1 H584 M56 131.4 H584 M56 101.7 H584 M56 72 H584"/>
  <text class="tick" x="48" y="283.5" text-anchor="end">0</text>
  <text class="tick" x="48" y="253.8" text-anchor="end">10</text>
  <text class="tick" x="48" y="224.1" text-anchor="end">20</text>
  <text class="tick" x="48" y="194.4" text-anchor="end">30</text>
  <text class="tick" x="48" y="164.6" text-anchor="end">40</text>
  <text class="tick" x="48" y="134.9" text-anchor="end">50</text>
  <text class="tick" x="48" y="105.2" text-anchor="end">60</text>
  <text class="tick" x="48" y="75.5" text-anchor="end">70</text>
  <path class="grid base" d="M56 280 H584"/>
  <path class="grid" d="M56 72 V280 M144 72 V280 M232 72 V280 M320 72 V280 M408 72 V280 M496 72 V280 M584 72 V280"/>
  <text class="tick mid" x="56" y="296">0</text>
  <text class="tick mid" x="144" y="296">2</text>
  <text class="tick mid" x="232" y="296">4</text>
  <text class="tick mid" x="320" y="296">6</text>
  <text class="tick mid" x="408" y="296">8</text>
  <text class="tick mid" x="496" y="296">10</text>
  <text class="tick mid" x="584" y="296">12</text>
  <text class="t-dim mid" x="320" y="314">hours into the battery</text>
  <path class="q4" d="M58 264 503 264 503 280 58 280 Z"/>
  <path class="q1" d="M58 158 63 177 68 179 72 181 100 181 105 175 110 161 114 161 119 163 124 163 129 166 147 166 152 169 157 170 183 170 219 136 222 142 227 143 231 141 236 148 241 149 246 149 250 152 255 151 260 152 264 152 269 150 274 147 278 148 283 156 288 162 292 162 297 164 302 173 306 171 311 171 316 169 320 160 325 154 330 153 334 155 344 157 348 157 353 147 358 146 367 164 372 158 377 144 381 150 386 148 391 152 395 150 400 147 405 142 409 144 414 150 419 143 423 145 428 146 433 145 437 146 442 144 447 144 451 147 456 144 461 145 465 145 470 146 475 148 479 149 484 145 489 145 493 146 498 149 503 149 503 264 58 264 Z"/>
  <path class="b0" d="M58 139 63 152 68 156 72 158 77 159 82 159 86 154 91 155 96 157 100 159 105 156 110 144 114 145 119 147 124 146 129 144 133 148 138 148 143 150 147 148 152 148 157 151 171 151 175 153 180 150 183 155 219 123 222 126 227 128 231 125 236 129 241 131 246 132 250 132 255 131 260 134 264 135 269 133 274 130 278 130 283 139 288 146 292 145 297 147 302 156 306 152 311 151 316 147 320 142 325 138 330 136 334 137 339 136 344 140 348 139 353 129 358 129 362 136 367 146 372 144 377 127 381 131 386 132 391 135 395 131 400 129 405 126 409 127 414 130 419 126 423 128 428 127 433 127 437 128 442 127 447 128 461 128 465 127 470 128 475 130 479 130 489 126 493 128 498 130 503 131 503 149 498 149 493 146 489 145 484 145 479 149 475 148 470 146 465 145 461 145 456 144 451 147 447 144 442 144 437 146 433 145 428 146 423 145 419 143 414 150 409 144 405 142 400 147 395 150 391 152 386 148 381 150 377 144 372 158 367 164 358 146 353 147 348 157 344 157 334 155 330 153 325 154 320 160 316 169 311 171 306 171 302 173 297 164 292 162 288 162 283 156 278 148 274 147 269 150 264 152 260 152 255 151 250 152 246 149 241 149 236 148 231 141 227 143 222 142 219 136 183 170 157 170 152 169 147 166 129 166 124 163 119 163 114 161 110 161 105 175 100 181 72 181 68 179 63 177 58 158 Z"/>
  <path class="q2" d="M58 137 63 150 68 153 72 155 77 155 82 157 86 153 96 153 100 156 105 153 110 142 114 143 119 144 129 142 133 144 138 147 143 148 147 146 152 146 157 148 161 149 166 149 171 148 175 151 180 149 183 153 219 122 222 125 227 127 231 125 241 131 255 131 260 134 264 134 269 133 274 130 278 129 283 138 288 144 292 144 297 146 302 154 306 151 311 150 316 146 325 137 330 135 334 136 339 135 344 139 348 138 353 129 358 129 362 135 367 145 372 143 377 126 381 130 386 132 391 133 395 130 405 126 409 126 414 130 419 126 423 127 433 127 437 128 442 127 447 128 451 127 456 127 461 128 465 126 475 130 479 129 484 127 489 126 493 127 498 129 503 130 503 131 498 130 493 128 489 126 479 130 475 130 470 128 465 127 461 128 447 128 442 127 437 128 433 127 428 127 423 128 419 126 414 130 409 127 405 126 400 129 395 131 391 135 386 132 381 131 377 127 372 144 367 146 362 136 358 129 353 129 348 139 344 140 339 136 334 137 330 136 325 138 320 142 316 147 311 151 306 152 302 156 297 147 292 145 288 146 283 139 278 130 274 130 269 133 264 135 260 134 255 131 250 132 246 132 241 131 236 129 231 125 227 128 222 126 219 123 183 155 180 150 175 153 171 151 157 151 152 148 147 148 143 150 138 148 133 148 129 144 124 146 119 147 114 145 110 144 105 156 100 159 96 157 91 155 86 154 82 159 77 159 72 158 68 156 63 152 58 139 Z"/>
  <path class="b3" d="M58 87 63 92 68 92 72 94 77 94 82 102 86 104 91 104 96 97 100 103 105 101 110 93 119 93 124 92 129 90 133 90 138 95 143 97 147 94 152 90 157 92 161 94 166 96 171 95 175 97 180 97 183 101 219 85 222 86 227 88 231 86 246 89 250 89 255 88 260 93 264 94 269 93 274 90 278 87 283 88 288 93 292 92 297 90 302 95 306 97 311 102 316 102 320 98 325 94 330 91 339 91 344 90 348 91 353 88 358 88 362 87 367 94 372 91 377 87 381 90 386 88 391 88 395 90 400 89 405 87 409 87 414 89 419 87 428 87 433 88 437 89 442 88 451 88 456 87 461 88 465 87 470 88 475 90 479 89 484 89 489 87 493 88 498 89 503 88 503 130 498 129 493 127 489 126 484 127 479 129 475 130 465 126 461 128 456 127 451 127 447 128 442 127 437 128 433 127 423 127 419 126 414 130 409 126 405 126 395 130 391 133 386 132 381 130 377 126 372 143 367 145 362 135 358 129 353 129 348 138 344 139 339 135 334 136 330 135 325 137 316 146 311 150 306 151 302 154 297 146 292 144 288 144 283 138 278 129 274 130 269 133 264 134 260 134 255 131 241 131 231 125 227 127 222 125 219 122 183 153 180 149 175 151 171 148 166 149 161 149 157 148 152 146 147 146 143 148 138 147 133 144 129 142 119 144 114 143 110 142 105 153 100 156 96 153 86 153 82 157 77 155 72 155 68 153 63 150 58 137 Z"/>
  <text class="tick t-them" x="592" y="283.5">0</text>
  <text class="tick t-them" x="592" y="231.5">4</text>
  <text class="tick t-them" x="592" y="179.5">8</text>
  <text class="tick t-them" x="592" y="127.5">12</text>
  <text class="tick t-them" x="592" y="75.5">16</text>
  <path class="line-them" d="M58 72 77 72 82 104 86 176 498 176 503 180"/>
  <text class="tick" x="48" y="60" text-anchor="end">GiB</text>
  <text class="tick t-them" x="592" y="60">in flight</text>
  <g class="legend">
  <rect class="q4" x="56" y="50" width="10" height="10"/>
  <text x="74" y="58">dense</text>
  <rect class="q1" x="121" y="50" width="10" height="10"/>
  <text x="139" y="58">experts</text>
  <rect class="b0" x="198.4" y="50" width="10" height="10"/>
  <text x="216.4" y="58">KV</text>
  <rect class="q2" x="244.8" y="50" width="10" height="10"/>
  <text x="262.8" y="58">activations</text>
  <rect class="b3" x="347" y="50" width="10" height="10"/>
  <text x="365" y="58">state</text>
  <path class="line-them" d="M412 55 H426"/>
  <text x="430" y="58">answers in flight</text>
  </g>
</svg>
</figure>

<figure class="fig">
<svg viewBox="0 0 640 324" role="img" aria-label="Figure 9.6-6. The share of the resident K/V tokens each format held over the battery, stacked to 100%; over the whole run: Q1_S 14.0%, Q3_0 13.3%, Q4_0 22.2%, Q4_1 8.4%, Q5_0 10.5%, Q8_0 11.4%, other 20.3%.">
  <text class="ttl" x="16" y="20">Figure 9.6-6 · The format mix does not move with load</text>
  <text class="ttl-sub" x="16" y="38">share of the resident tokens held in each K/V format, per 32-token block</text>
  <path class="grid" d="M56 228 H584 M56 176 H584 M56 124 H584 M56 72 H584"/>
  <text class="tick" x="48" y="283.5" text-anchor="end">0%</text>
  <text class="tick" x="48" y="231.5" text-anchor="end">25%</text>
  <text class="tick" x="48" y="179.5" text-anchor="end">50%</text>
  <text class="tick" x="48" y="127.5" text-anchor="end">75%</text>
  <text class="tick" x="48" y="75.5" text-anchor="end">100%</text>
  <path class="grid base" d="M56 280 H584"/>
  <path class="grid" d="M56 72 V280 M144 72 V280 M232 72 V280 M320 72 V280 M408 72 V280 M496 72 V280 M584 72 V280"/>
  <text class="tick mid" x="56" y="296">0</text>
  <text class="tick mid" x="144" y="296">2</text>
  <text class="tick mid" x="232" y="296">4</text>
  <text class="tick mid" x="320" y="296">6</text>
  <text class="tick mid" x="408" y="296">8</text>
  <text class="tick mid" x="496" y="296">10</text>
  <text class="tick mid" x="584" y="296">12</text>
  <text class="t-dim mid" x="320" y="314">hours into the battery</text>
  <path class="q4" d="M58 251 63 250 68 250 72 253 77 252 82 247 86 249 91 253 96 255 100 252 105 250 110 252 114 249 119 248 129 248 133 249 138 251 143 252 147 250 152 251 161 251 166 247 171 247 175 250 180 252 183 254 219 252 222 252 227 251 231 253 236 248 241 252 246 252 250 254 255 252 260 252 264 253 269 250 274 251 278 252 283 252 288 249 292 251 297 251 302 254 306 255 311 252 316 252 320 251 325 255 330 250 334 252 339 255 344 251 348 247 353 246 358 248 362 249 367 248 372 251 377 251 381 252 386 251 391 252 395 252 400 251 405 249 409 251 414 253 419 250 423 251 428 250 433 254 437 253 442 249 447 252 451 251 456 255 461 253 465 251 470 250 475 251 479 251 484 252 489 250 493 252 498 248 503 251 503 280 58 280 Z"/>
  <path class="q3" d="M58 223 63 222 68 222 72 225 77 224 82 220 86 221 91 225 96 226 100 224 105 223 110 224 114 221 129 221 133 222 143 224 147 223 152 224 157 223 161 223 166 220 171 220 175 223 180 224 183 226 219 224 222 224 227 223 231 225 236 221 241 224 246 224 250 226 255 224 260 224 264 225 269 222 274 223 278 224 283 224 288 222 292 223 297 223 302 226 306 227 311 224 316 224 320 223 325 226 330 223 334 224 339 226 344 224 348 220 353 219 358 221 367 221 372 223 377 223 381 224 386 223 391 224 395 224 405 222 409 223 414 225 419 222 423 223 428 223 433 226 437 224 442 221 447 224 451 223 456 226 461 225 465 223 470 222 475 224 479 223 484 224 489 223 493 224 498 221 503 223 503 251 498 248 493 252 489 250 484 252 479 251 475 251 470 250 465 251 461 253 456 255 451 251 447 252 442 249 437 253 433 254 428 250 423 251 419 250 414 253 409 251 405 249 400 251 395 252 391 252 386 251 381 252 377 251 372 251 367 248 362 249 358 248 353 246 348 247 344 251 339 255 334 252 330 250 325 255 320 251 316 252 311 252 306 255 302 254 297 251 292 251 288 249 283 252 278 252 274 251 269 250 264 253 260 252 255 252 250 254 246 252 241 252 236 248 231 253 227 251 222 252 219 252 183 254 180 252 175 250 171 247 166 247 161 251 152 251 147 250 143 252 138 251 133 249 129 248 119 248 114 249 110 252 105 250 100 252 96 255 91 253 86 249 82 247 77 252 72 253 68 250 63 250 58 251 Z"/>
  <path class="b0" d="M58 177 63 175 68 175 72 180 77 178 82 171 86 174 91 180 96 182 100 178 105 176 110 178 114 174 124 172 129 173 133 175 143 179 147 176 152 178 157 177 161 177 166 172 171 171 175 176 180 178 183 182 219 179 222 179 227 177 231 180 236 173 241 179 246 178 250 182 255 178 260 178 264 179 269 175 274 177 278 178 283 179 288 175 292 176 297 177 302 181 306 183 311 179 316 179 320 177 325 182 330 176 334 178 339 182 344 178 348 171 353 171 358 174 362 174 367 173 372 177 377 177 381 179 386 177 391 178 395 178 400 177 405 175 409 177 414 180 419 175 423 177 428 176 433 181 437 179 442 173 447 178 451 177 456 182 461 180 465 177 470 175 475 177 479 177 484 178 489 177 493 178 498 173 503 177 503 223 498 221 493 224 489 223 484 224 479 223 475 224 470 222 465 223 461 225 456 226 451 223 447 224 442 221 437 224 433 226 428 223 423 223 419 222 414 225 409 223 405 222 395 224 391 224 386 223 381 224 377 223 372 223 367 221 358 221 353 219 348 220 344 224 339 226 334 224 330 223 325 226 320 223 316 224 311 224 306 227 302 226 297 223 292 223 288 222 283 224 278 224 274 223 269 222 264 225 260 224 255 224 250 226 246 224 241 224 236 221 231 225 227 223 222 224 219 224 183 226 180 224 175 223 171 220 166 220 161 223 157 223 152 224 147 223 143 224 133 222 129 221 114 221 110 224 105 223 100 224 96 226 91 225 86 221 82 220 77 224 72 225 68 222 63 222 58 223 Z"/>
  <path class="q1" d="M58 159 63 158 68 158 72 163 77 161 82 154 86 157 91 162 96 165 100 161 105 158 110 160 114 156 119 155 129 155 133 157 138 160 143 161 147 159 152 161 157 160 161 159 166 155 171 154 175 159 180 161 183 164 219 161 222 161 227 159 231 162 236 155 241 161 246 161 250 164 255 161 260 161 264 162 269 158 274 159 278 161 283 161 288 158 292 159 297 159 302 164 306 165 311 161 316 161 320 160 325 165 330 159 334 161 339 165 344 161 348 154 353 153 358 156 367 156 372 160 377 160 381 161 386 159 391 161 395 160 400 160 405 157 409 160 414 162 419 158 423 159 428 159 433 163 437 162 442 156 447 160 451 159 456 164 461 162 465 159 470 158 475 160 479 159 484 161 489 159 493 161 498 156 503 159 503 177 498 173 493 178 489 177 484 178 479 177 475 177 470 175 465 177 461 180 456 182 451 177 447 178 442 173 437 179 433 181 428 176 423 177 419 175 414 180 409 177 405 175 400 177 395 178 391 178 386 177 381 179 377 177 372 177 367 173 362 174 358 174 353 171 348 171 344 178 339 182 334 178 330 176 325 182 320 177 316 179 311 179 306 183 302 181 297 177 292 176 288 175 283 179 278 178 274 177 269 175 264 179 260 178 255 178 250 182 246 178 241 179 236 173 231 180 227 177 222 179 219 179 183 182 180 178 175 176 171 171 166 172 161 177 157 177 152 178 147 176 143 179 133 175 129 173 124 172 114 174 110 178 105 176 100 178 96 182 91 180 86 174 82 171 77 178 72 180 68 175 63 175 58 177 Z"/>
  <path class="b2" d="M58 138 63 137 68 137 72 139 77 138 82 136 86 137 91 139 96 140 100 138 105 137 110 138 114 136 129 136 133 137 138 138 147 138 152 139 157 138 161 138 166 136 171 136 175 138 180 138 183 139 219 138 227 138 231 139 236 136 241 138 246 138 250 139 255 138 260 138 264 139 269 137 274 138 283 138 288 137 292 138 297 138 302 139 306 140 311 138 320 138 325 140 330 138 334 138 339 140 344 138 348 136 353 136 358 137 367 137 372 138 400 138 405 137 409 138 414 139 419 137 423 138 428 137 433 139 437 138 442 136 447 138 451 138 456 139 461 139 465 138 470 137 475 138 484 138 489 137 493 138 498 136 503 137 503 159 498 156 493 161 489 159 484 161 479 159 475 160 470 158 465 159 461 162 456 164 451 159 447 160 442 156 437 162 433 163 428 159 423 159 419 158 414 162 409 160 405 157 400 160 395 160 391 161 386 159 381 161 377 160 372 160 367 156 358 156 353 153 348 154 344 161 339 165 334 161 330 159 325 165 320 160 316 161 311 161 306 165 302 164 297 159 292 159 288 158 283 161 278 161 274 159 269 158 264 162 260 161 255 161 250 164 246 161 241 161 236 155 231 162 227 159 222 161 219 161 183 164 180 161 175 159 171 154 166 155 161 159 157 160 152 161 147 159 143 161 138 160 133 157 129 155 119 155 114 156 110 160 105 158 100 161 96 165 91 162 86 157 82 154 77 161 72 163 68 158 63 158 58 159 Z"/>
  <path class="q2" d="M58 114 68 114 72 115 77 114 82 112 86 113 91 115 96 116 100 114 110 114 114 113 133 113 143 115 147 114 152 115 157 114 161 114 171 112 175 114 180 114 183 115 219 115 222 114 227 114 231 115 236 113 241 115 246 114 250 116 255 114 260 115 264 115 269 114 278 114 283 115 288 114 297 114 302 115 306 116 311 115 316 115 320 114 325 116 330 114 334 115 339 116 344 114 348 113 353 112 358 113 367 113 372 114 377 114 381 115 386 114 391 115 395 114 400 114 405 113 409 114 414 115 419 114 428 114 433 115 437 115 442 113 447 114 451 114 456 116 461 115 465 114 493 114 498 113 503 114 503 137 498 136 493 138 489 137 484 138 475 138 470 137 465 138 461 139 456 139 451 138 447 138 442 136 437 138 433 139 428 137 423 138 419 137 414 139 409 138 405 137 400 138 372 138 367 137 358 137 353 136 348 136 344 138 339 140 334 138 330 138 325 140 320 138 311 138 306 140 302 139 297 138 292 138 288 137 283 138 274 138 269 137 264 139 260 138 255 138 250 139 246 138 241 138 236 136 231 139 227 138 219 138 183 139 180 138 175 138 171 136 166 136 161 138 157 138 152 139 147 138 138 138 133 137 129 136 114 136 110 138 105 137 100 138 96 140 91 139 86 137 82 136 77 138 72 139 68 137 63 137 58 138 Z"/>
  <path class="b3" d="M58 72 503 72 503 114 498 113 493 114 465 114 461 115 456 116 451 114 447 114 442 113 437 115 433 115 428 114 419 114 414 115 409 114 405 113 400 114 395 114 391 115 386 114 381 115 377 114 372 114 367 113 358 113 353 112 348 113 344 114 339 116 334 115 330 114 325 116 320 114 316 115 311 115 306 116 302 115 297 114 288 114 283 115 278 114 269 114 264 115 260 115 255 114 250 116 246 114 241 115 236 113 231 115 227 114 222 114 219 115 183 115 180 114 175 114 171 112 161 114 157 114 152 115 147 114 143 115 133 113 114 113 110 114 100 114 96 116 91 115 86 113 82 112 77 114 72 115 68 114 58 114 Z"/>
  <g class="legend">
  <rect class="q4" x="56" y="50" width="10" height="10"/>
  <text x="74" y="58">Q1_S</text>
  <rect class="q3" x="114.8" y="50" width="10" height="10"/>
  <text x="132.8" y="58">Q3_0</text>
  <rect class="b0" x="173.6" y="50" width="10" height="10"/>
  <text x="191.6" y="58">Q4_0</text>
  <rect class="q1" x="232.4" y="50" width="10" height="10"/>
  <text x="250.4" y="58">Q4_1</text>
  <rect class="b2" x="291.2" y="50" width="10" height="10"/>
  <text x="309.2" y="58">Q5_0</text>
  <rect class="q2" x="350" y="50" width="10" height="10"/>
  <text x="368" y="58">Q8_0</text>
  <rect class="b3" x="408.8" y="50" width="10" height="10"/>
  <text x="426.8" y="58">other</text>
  </g>
</svg>
</figure>

<figure class="fig">
<svg viewBox="0 0 640 324" role="img" aria-label="Figure 9.6-7. Over the battery, the resident tokens would take 13.2 GiB at BF16 on average and 21.9 at peak; the KV arenas the span held for them took 6.2 GiB on average and 11.2 at peak, 2.16 times smaller, out of 10.7 GiB of KV ground on average.">
  <text class="ttl" x="16" y="20">Figure 9.6-7 · The context against its BF16 size</text>
  <text class="ttl-sub" x="16" y="38">GiB · resident tokens at BF16, 24,916 B each, against the KV arenas that hold them</text>
  <path class="grid" d="M56 228 H584 M56 176 H584 M56 124 H584 M56 72 H584"/>
  <text class="tick" x="48" y="283.5" text-anchor="end">0</text>
  <text class="tick" x="48" y="231.5" text-anchor="end">5</text>
  <text class="tick" x="48" y="179.5" text-anchor="end">10</text>
  <text class="tick" x="48" y="127.5" text-anchor="end">15</text>
  <text class="tick" x="48" y="75.5" text-anchor="end">20</text>
  <path class="grid base" d="M56 280 H584"/>
  <path class="grid" d="M56 72 V280 M144 72 V280 M232 72 V280 M320 72 V280 M408 72 V280 M496 72 V280 M584 72 V280"/>
  <text class="tick mid" x="56" y="296">0</text>
  <text class="tick mid" x="144" y="296">2</text>
  <text class="tick mid" x="232" y="296">4</text>
  <text class="tick mid" x="320" y="296">6</text>
  <text class="tick mid" x="408" y="296">8</text>
  <text class="tick mid" x="496" y="296">10</text>
  <text class="tick mid" x="584" y="296">12</text>
  <text class="t-dim mid" x="320" y="314">hours into the battery</text>
  <path class="area-us" d="M58 214 63 193 68 199 72 201 77 204 82 205 86 187 91 189 96 197 100 204 105 213 110 223 114 225 119 224 124 219 129 203 133 216 138 218 143 224 147 218 152 208 157 211 161 213 166 212 171 214 175 221 180 211 183 226 219 236 222 223 227 225 231 224 236 215 241 219 246 221 250 212 255 213 260 219 264 221 269 220 274 222 278 217 283 220 288 222 292 220 297 220 302 219 306 212 311 208 316 205 320 217 325 223 330 221 334 214 339 210 344 220 348 218 353 218 358 222 362 218 367 219 372 228 377 221 381 215 386 224 391 223 395 214 400 215 405 225 409 218 414 210 419 222 423 217 428 214 433 217 437 218 447 224 451 214 456 221 461 220 465 216 475 216 479 213 484 219 489 214 493 216 498 212 503 216 503 280 58 280 Z"/>
  <path class="line-us dash" d="M58 183 63 144 68 149 72 144 77 147 82 120 86 95 91 97 96 130 100 115 105 133 110 171 114 174 119 171 124 171 129 159 133 175 138 159 143 159 147 161 152 165 161 156 166 148 171 153 175 153 180 144 183 147 219 211 222 193 227 191 231 195 236 183 246 181 250 173 255 178 260 166 264 163 269 167 274 181 278 184 283 183 288 170 292 170 297 177 302 158 306 144 311 122 316 119 320 147 325 167 330 174 334 167 339 163 344 179 348 171 353 183 358 186 362 187 367 164 372 182 377 189 381 171 386 190 391 187 395 173 400 177 405 195 409 188 414 173 419 191 423 185 428 181 433 183 437 180 442 186 447 187 451 179 456 189 461 184 465 185 470 182 475 174 479 174 484 181 489 183 493 181 498 175 503 179"/>
  <path class="line-them" d="M58 128 63 80 68 98 72 107 77 120 82 155 86 145 91 143 96 128 100 154 105 157 110 152 114 163 119 158 124 141 129 123 133 145 138 145 143 165 147 144 152 126 157 136 161 133 166 148 171 143 175 150 180 134 183 168 219 180 222 154 227 158 231 154 236 127 241 145 246 152 250 129 255 127 260 145 264 153 269 145 274 149 278 140 283 147 288 151 292 145 297 155 302 149 306 130 311 140 316 136 320 139 325 152 330 147 334 130 339 130 344 147 348 140 353 142 358 149 362 138 367 157 372 164 377 151 381 131 386 155 391 154 395 141 400 135 405 156 409 139 414 143 419 145 423 138 428 129 433 143 437 144 442 155 447 158 451 139 456 153 461 152 465 137 470 139 475 151 479 135 484 148 489 129 493 139 498 128 503 138"/>
  <path class="line-us" d="M58 214 63 193 68 199 72 201 77 204 82 205 86 187 91 189 96 197 100 204 105 213 110 223 114 225 119 224 124 219 129 203 133 216 138 218 143 224 147 218 152 208 157 211 161 213 166 212 171 214 175 221 180 211 183 226 219 236 222 223 227 225 231 224 236 215 241 219 246 221 250 212 255 213 260 219 264 221 269 220 274 222 278 217 283 220 288 222 292 220 297 220 302 219 306 212 311 208 316 205 320 217 325 223 330 221 334 214 339 210 344 220 348 218 353 218 358 222 362 218 367 219 372 228 377 221 381 215 386 224 391 223 395 214 400 215 405 225 409 218 414 210 419 222 423 217 428 214 433 217 437 218 447 224 451 214 456 221 461 220 465 216 475 216 479 213 484 219 489 214 493 216 498 212 503 216"/>
  <g class="legend">
  <path class="line-them" d="M56 55 H70"/>
  <text x="74" y="58">tokens at BF16</text>
  <path class="line-us" d="M176.8 55 H190.8"/>
  <text x="194.8" y="58">KV arenas held</text>
  <path class="line-us dash" d="M297.6 55 H311.6"/>
  <text x="315.6" y="58">KV ground</text>
  </g>
</svg>
</figure>

| Under the battery's load | Mean | Peak |
|---|---:|---:|
| Sessions in flight | 8.5 | 16 |
| KV arenas held (GiB) | 6.19 | 11.17 |
| Tokens resident | 570,909 | 945,212 |
| The same tokens at BF16 (GiB) | 13.25 | 21.93 |
| Effective compression (BF16 / arenas held) | 2.16× | 2.70× |
| Expert zone (GiB) | 36.0 | 50.1 |

The card holds 571 thousand tokens of context on average and 945 thousand at its peak — 13.3 and 21.9 GiB at BF16 — in 6.2 and 11.2 GiB of KV arenas, while the expert zone gives up ground, from 51.0 GiB at rest to 36.0 on average, and takes it back as sessions finish. The larger tenant under load is not the KV but the recurrent state of Flash-Next's DeltaNet layers, one per session and fork: 15.5 GiB on average and 23.1 at peak, against 4.9 at rest. **The format mix does not move with load**: with eight answers in flight and with sixteen, 22% of resident tokens are Q4_0, 14% Q1_S and 11% Q8_0, to within a point (`table_9_6_load_bands.csv`), because a block's format is set as it is sealed, by its level and its own error, never by the pressure on the card. Compression against the arenas is 2.16× at eight answers and 2.20× at sixteen, below the stored K/V's 3.55×; the difference is ground the span holds beyond the payload, at a load far from the card's limit. (The battery ran sixteen at once for its first 142 answers and eight for the rest.)

### 9.7 Ablation: Expert Prediction and Eviction Policy

The three-tier cache's effect is Figure 4-1 (+12% to +87% prefill). A cleaner ablation came from replacing the MoE dispatch with the live, device-side one: the dense models in the same sweep, which do not touch it, moved −1.2% to +0.5% on prefill and −1.6% to +0.2% on decode — a control reading as a control should — while Qwen3.5-35B-A3B gained 24–36% prefill and 11–23% decode, Qwen3.6-35B-A3B likewise, and Qwen3-30B-A3B's single-session decode went from 45.8 to 54.9 t/s (+20%) on the same card and artifact (`docs/results/moe_live_dispatch_rtx_3090_24gb_2026-10-04.md`).

**The predictor and eviction, isolated** on a captured Qwen3-30B-A3B routing trace of 21 prompts (code, prose, facts, maths, lists) by 21-fold leave-one-out, each prompt predicted by a model trained on the other twenty and learning causally as it decodes (`docs/archived/markov_expert_prediction_eval.md`). Coverage is the share of *cold* experts a prefetch of fan-out k would have staged.

| Promotion predictor (held out) | Top-1 precision | Cold coverage, k = 1 | k = 2 | k = 4 |
|---|---:|---:|---:|---:|
| Popularity | 21.8% | 2.9% | 5.2% | 9.0% |
| Raw co-occurrence counts | 67.5% | 9.0% | 16.7% | 29.4% |
| PMI(0.5) | 80.4% | 10.7% | 19.9% | 34.9% |
| PMI(0.5), cross-prompt prior + live session matrix | **91.1%** | **12.1%** | **23.4%** | **42.7%** |

At eight candidates a layer the best model stages 67.7% of cold experts. Per-layer-pair tables beat every coarser sharing (one shared table: 15.2% at k = 4), and weighting by router mass collapses coverage to 23.6%: what predicts is how often a transition occurs, not how heavily it is routed.

| Eviction policy (held out) | Miss rate, budget ½ of working set | Budget 0.7 |
|---|---:|---:|
| LRU | 22.9% | 21.0% |
| LFU | 28.4% | 13.0% |
| Learned recurrence model | 39.5% | 29.1% |
| **Frequency-weighted recency** (`last_used + ln(1 + freq) · 16`) | **14.7%** | **12.7%** |

At 60% residency ≈82% of accesses hit, ≈5% become overlapped loads by prediction and ≈13% stall; tuned for stall cost (a stall weighed at ten converted loads) the stalling share falls to 7.8% and the stall cost by 52%. In the batched wave, prediction pays between the working set fitting and the link saturating; past about 256 concurrent decodes, prefetching every missing expert of the next layer matches the tuned optimum within 1–4%. Live precision in production is 66–85%, typically 74%.

### 9.8 Quality Evaluation: Multi-Session Identity Discrimination

Up to 256 sessions run in one batch, each assigned a distinct character (name and gender, period 99) and asked to rewrite a passage with it. A session passes if its name appears, its pronouns match, and no other session's name appears — stressing the session-discriminating signal aggressive K quantization degrades first. Every throughput row of §9.6 passes with every session correct: 213 rows on the RTX 3090, 192 on the laptop, 212 on Blackwell. C10 passes on every model at 4.1–7.6×, just inside the edge where the test fails; the one exception on record, Flash-Next's C10 ×8 under the TCC build, validated 7 of 8 sessions and is reported as a failure (the WDDM build's 885.8 t/s is the validated figure). Past the edge, Llama-3.2-3B fails in three regimes — clean, soft drift, then identity collapse: small errors move the text, large ones move which session the model attends.

### 9.9 Cross-Architecture Transfer: Held-Out Model Validation

The selector was derived from two reference models, Qwen3-30B-A3B (MoE) and Llama-3.2-3B (dense). Deployed unchanged on held-out Qwen3-8B it compressed *more* than either — 7.42× in v1 against 7.04× and 5.02×, and **6.31×** against 5.94× and 4.62× on the current 8K gates (§9.6.3) — the opposite of what overfitting would produce. It now runs on twelve models, dense, MoE, DeltaNet-hybrid and native-sparse, 0.5B to 284B, reaching 4.1–7.6× at the top level on each, with no calibration data: a model carries at most a scale on its thresholds (`*_KV_FACTORS`), set by stepping C10 to the story gate's edge. Unlike GPTQ, AWQ or KVQuant's sensitivity profiles, the kernel responds to each block's distribution at inference time.

### 9.10 Attentional Provenance Indexing: The Signal, and Tool Selection

v1's calibration (`docs/unbounded_agents_v1.md` §9.10), a 48-probe, 8-tool harness, fixed the signal: only sign signatures folded across several heads found the right tool on every probe, while single heads, band averages and every Q→K strategy failed on some. Within a content type the right item's scores and the rest overlapped in their lower tails, so no threshold separated them — which is why production reads every score against its own background (§6.4). v1's tool selection first failed outright because its corpus held decode signatures while the probe was a prefilled request: over 8 tools the prefill produced one discriminative token, the decode that followed sixteen, all on the right tool. Hence calibration questions shaped like the probe beside every tool and file (§6.3).

The production tool selector (`docs/tool_selection_provenance_results.md` §23–§26) scores the last 128 decoded tokens' signatures against a gallery of tagged tool turns at every reprojection, votes by `z × margin` with the strongest quarter kept, carries belief across the turn, and scopes one to three tools. Leave-one-out with a question as the probe over 766 tool turns and 93 tools (chance ≈ 1.1%), the calibration questions were the lever (Figure 9.10-1):

<figure class="fig">
<svg viewBox="0 0 640 176" role="img" aria-label="Figure 9.10-1. Tool selection over 93 tools, leave-one-out. 766 tool turns: Tool-1 74.5 percent, Tool-3 92.7, recall in scope 77.8, MRR 0.833. Adding 8 calibration questions a tool: 85.1, 96.5, 87.1, MRR 0.907. Adding 24 a tool: 87.9, 97.5, 89.0, MRR 0.925.">
  <text class="ttl" x="16" y="20">Figure 9.10-1 · The corpus, not the scorer, lifted tool selection</text>
  <text class="ttl-sub" x="16" y="36">93 tools, leave-one-out · solid Tool-1 · faint Tool-3 · chance ≈ 1.1%</text>
  <path class="grid" d="M300 48 V150 M400 48 V150 M500 48 V150"/>
  <path class="grid base" d="M200 48 V150"/>
  <text class="cat" x="16" y="66">766 tool turns</text>
  <text class="cat-sub" x="16" y="79">MRR 0.833 · in scope 77.8%</text>
  <rect class="us-soft" x="200" y="56" width="370.8" height="10"/>
  <rect class="us" x="200" y="68" width="298" height="10"/>
  <text class="t-dim" x="576.8" y="65">92.7%</text>
  <text class="v-us" x="504" y="78">74.5%</text>
  <text class="cat" x="16" y="100">+ 8 questions a tool</text>
  <text class="cat-sub" x="16" y="113">MRR 0.907 · in scope 87.1%</text>
  <rect class="us-soft" x="200" y="90" width="386" height="10"/>
  <rect class="us" x="200" y="102" width="340.4" height="10"/>
  <text class="t-dim" x="592" y="99">96.5%</text>
  <text class="v-us" x="546.4" y="112">85.1%</text>
  <text class="cat" x="16" y="134">+ 24 questions a tool</text>
  <text class="cat-sub" x="16" y="147">MRR 0.925 · in scope 89.0%</text>
  <rect class="us-soft" x="200" y="124" width="390" height="10"/>
  <rect class="us" x="200" y="136" width="351.6" height="10"/>
  <text class="t-dim" x="596" y="133">97.5%</text>
  <text class="v-us" x="557.6" y="146">87.9%</text>
  <text class="tick mid" x="200" y="166">0%</text>
  <text class="tick mid" x="300" y="166">25%</text>
  <text class="tick mid" x="400" y="166">50%</text>
  <text class="tick mid" x="500" y="166">75%</text>
</svg>
</figure>

Scoring a whole stored turn instead overstates the rate (97.4–97.8% Top-1), because the call inside it names its own tool. The code-corpus equivalent is §9.12's file retrieval.

### 9.11 Concurrent Persistent-Memory Session Result

The width ladder never builds the conversation engine. The **engine probe** (`candle-conversation/tests/kv_fragmentation.rs`) does — admission, per-turn projection, sealing, compaction and tier migration all live — and checks every session's story. On the latest run on each card (laptop `bf291341c`, RTX 3090 `c375834c8`, RTX PRO 5000 `21c36efbd`) every story is correct: Flash-Next 8/8 on all three cards, Qwen3-30B-A3B 20/20 and Qwen3.6-35B-A3B 16/16 on both desktops, with no non-finite layer in any log. Flash-Next is the stronger test, the one model carrying recurrent state outside the paged K/V. On the RTX 3090 its gate went from 24.3 to 138.1 t/s single-session decode, and from 95.6 to 361.6 at C10 ×8, in ten days with every row validated: the throughput work did not trade against correctness.

#### 9.11.1 Retrieval on the live gallery

Every retrieval figure here is measured on the running daemon's resident gallery over HTTP (`POST /v1/substrate/zend/scan`), with held-out probes — a question never in the gallery plus the thinking it starts:

- **Tools:** first on 87.9% over 93 tools, top three on 97.5% (§9.10).
- **Code:** the answer file first of 7,943 on 49% of the 128-question battery, top five on 69% (§9.12).
- **Normalization:** R@1 over 37,284 calibration questions 0.107 → 0.239, no file taking more than 0.1% of the wins on other files' questions (§6.4).
- **Momentum:** R@10 0.554 → 0.708 over three turns on a file, at the cost of slower topic switches — a new file leads in 70% of dialogues after three turns (`docs/results/provenance_gates_2026-10-26.md`).
- **A real conversation:** replayed on the paper conversation itself, the scorer finds 91% of the files it should place and puts all 14 of its tool calls' tools in the top ten (`docs/results/provenance_gates_2026-10-25.md`).

#### 9.11.2 Selection past the working set

Is the bounded working set delivered when the history is many times larger than it (`candle-conversation/tests/provenance_depth.rs`; `docs/results/theorem_depth_tests_2026-11-10.md`)? Flash-Next runs with its slot capped at 32,768 tokens. A conversation is built from the repository's source, one file a turn, to 30K, 131K and 393K tokens, with four turns stating vault codes together at 45% of its depth, where neither an oldest-first nor a newest-first fill reaches; the last turn asks for the codes. Each cell is a fresh conversation:

| Needle turns selected into the working set | 30K (19 turns) | 131K (64 turns) | 393K (195 turns) |
|---|---|---|---|
| Provenance with reprojection, C10 | 4/4 | 4/4 | 4/4 |
| Provenance with reprojection, C0 | 4/4 | 4/4 | 4/4 |
| One-shot projection, C10 | 3/4 | 0/4 | 0/4 |
| Rolling window (no provenance) | 3/4 | 0/4 | 0/4 |

Mid-decode reprojection brought every needle turn into a working set of 15–18 turns drawn from 195 — a history twelve times the slot — at C10 and C0 alike; the arms without it selected none past 30K, both filling from the oldest turns. The working set the theorem assumes is delivered at every depth measured, and compression does not change what is selected.

### 9.12 Unbounded Context: Codebase Dependency Analysis

A harder property than recall: multi-hop reasoning over the system's own Candle fork (2.2M lines of Rust and CUDA), each decode step's Q vectors retrieving the next node of a dependency chain. The production code-reading ingest has the daemon's model, Flash-Next, read the repository file by file as turns of its own, sealing their K/V and signatures with calibration questions beside each file; nothing in it was written by an external model. It persists as the substrate does (§7), valid for the model that wrote it.

**The claim.** *Provenance projection supplies the memory the answers come from, and re-selecting the working set while the model reasons brings in the files a question does not name.* The battery isolates the memory, the seed files a conversation opens on, reprojection during reasoning, and continuous reprojection against a single retrieval, holding all else equal.

**Questions.** 128 (`zend/examples/dependency_battery/battery.yaml`): 43 direct, 43 transitive, 42 architectural, each listing the items a correct answer must name, its answer files, and whether it names its answer's file — all from the source and committed before the first run with the analysis; no question spells an item it expects. The direct questions are generated mechanically (`direct.rs`: a public function defined once and called from one to three other non-test files, picked by a seeded hash), so their selection carries no judgement; the rest are the 24-question pilot's and 69 more, written by hand against the lines they cite. *Direct:* where `tier_fits` is defined and what calls it. *Transitive:* how MoE routing reaches the expert kernels with no host readback — the bucketize kernel, the mapped ring, the stager's word, the cold expert's wait. *Architectural:* the elastic boundary between KV and experts (`weight_floor`) and the check that refuses a tier crossing it.

**Arms.** Three request-level dials: `projection` *Full* or *Minimal* (the tool catalog alone, and so nothing with tools off); `seed` on or off; and reprojection through the think block, held in it, or at the opening only:

| Arm | Seed | Projection | Reprojection | What it isolates |
|---|---|---|---|---|
| A0 | off | Minimal | — | the model alone |
| A1 | on | Minimal | — | the seed files alone |
| D1 | on | Full | opening only | retrieve once, before reasoning |
| B0 | off | Full | held in thinking | — |
| B1 | on | Full | held in thinking | reasoning-time reprojection removed |
| C0 | off | Full | full | — |
| C1 | on | Full | full | the production system |

**Running.** On BlackWorld through the daemon's chat API (`dependency_battery run`), the first 748 answers at build 5470-6eb469ce5 and the rest at 5475-3c73da839, which changes no retrieval dial. One fresh conversation per question, its arm's dials on the request, deleted once its answer and projection points (`GET /v1/conversations/{id}/placements`) are read; every question asks for an answer from memory with no tool. The first 142 answers ran sixteen at once and the rest eight, and the order mixes the arms throughout, so drift falls on every arm alike. Each arm answers the 128 questions three times: 2,688 answers.

**Scoring.** Recall is the share of a question's expected items the answer names, any spelling, case-insensitive; a question's score in an arm is its mean over three runs. Intervals are 95% bootstrap intervals over questions (10,000 resamples, paired for differences); contrasts are Wilcoxon signed-rank on per-question differences, Holm-corrected across five. Every figure's data and `summary.md` are in `docs/results/battery_128/`.

<figure class="fig">
<svg viewBox="0 0 640 306" role="img" aria-label="Figure 9.12-1. Mean recall by arm over 128 questions, each with its 95% bootstrap interval: A0 0.4% (0.1% to 0.9%); A1 0.5% (0.1% to 1.0%); D1 30.6% (25.0% to 36.4%); B0 31.9% (26.1% to 37.7%); B1 32.5% (26.8% to 38.3%); C0 36.7% (30.8% to 42.6%); C1 37.2% (31.0% to 43.2%). The reprojecting arms C0 and C1 recall the most.">
  <text class="ttl" x="16" y="20">Figure 9.12-1 · Recall by arm</text>
  <text class="ttl-sub" x="16" y="38">mean recall over 128 questions · bar: 95% bootstrap interval · small dots: each run's mean</text>
  <path class="grid" d="M70 212.4 H624 M70 174.8 H624 M70 137.2 H624 M70 99.6 H624 M70 62 H624"/>
  <text class="tick" x="62" y="253.5" text-anchor="end">0%</text>
  <text class="tick" x="62" y="215.9" text-anchor="end">10%</text>
  <text class="tick" x="62" y="178.3" text-anchor="end">20%</text>
  <text class="tick" x="62" y="140.7" text-anchor="end">30%</text>
  <text class="tick" x="62" y="103.1" text-anchor="end">40%</text>
  <text class="tick" x="62" y="65.5" text-anchor="end">50%</text>
  <path class="grid base" d="M70 250 H624"/>
  <rect class="them" opacity="0.35" x="102.6" y="246.7" width="14" height="3.1"/>
  <circle class="q1" cx="125.6" cy="248.3" r="2.5"/>
  <circle class="q1" cx="125.6" cy="248.4" r="2.5"/>
  <circle class="q1" cx="125.6" cy="248.8" r="2.5"/>
  <circle class="dot-them" cx="109.6" cy="248.5" r="6"/>
  <text class="v-them mid" x="109.6" y="238.7">0.4%</text>
  <text class="cat mid" x="109.6" y="268">A0</text>
  <text class="cat-sub mid" x="109.6" y="282">no memory</text>
  <text class="cat-sub mid" x="109.6" y="295">no seed</text>
  <rect class="them" opacity="0.35" x="181.7" y="246.3" width="14" height="3.3"/>
  <circle class="q1" cx="204.7" cy="247.6" r="2.5"/>
  <circle class="q1" cx="204.7" cy="248.4" r="2.5"/>
  <circle class="q1" cx="204.7" cy="248.7" r="2.5"/>
  <circle class="dot-them" cx="188.7" cy="248.2" r="6"/>
  <text class="v-them mid" x="188.7" y="238.3">0.5%</text>
  <text class="cat mid" x="188.7" y="268">A1</text>
  <text class="cat-sub mid" x="188.7" y="282">no memory</text>
  <text class="cat-sub mid" x="188.7" y="295">seed</text>
  <rect class="them" opacity="0.35" x="260.9" y="113" width="14" height="43"/>
  <circle class="q1" cx="283.9" cy="134.3" r="2.5"/>
  <circle class="q1" cx="283.9" cy="132.3" r="2.5"/>
  <circle class="q1" cx="283.9" cy="137.9" r="2.5"/>
  <circle class="dot-them" cx="267.9" cy="134.8" r="6"/>
  <text class="v-them mid" x="267.9" y="105">30.6%</text>
  <text class="cat mid" x="267.9" y="268">D1</text>
  <text class="cat-sub mid" x="267.9" y="282">once</text>
  <text class="cat-sub mid" x="267.9" y="295">seed</text>
  <rect class="them" opacity="0.35" x="340" y="108.3" width="14" height="43.5"/>
  <circle class="q1" cx="363" cy="130.9" r="2.5"/>
  <circle class="q1" cx="363" cy="135.8" r="2.5"/>
  <circle class="q1" cx="363" cy="123.8" r="2.5"/>
  <circle class="dot-them" cx="347" cy="130.2" r="6"/>
  <text class="v-them mid" x="347" y="100.3">31.9%</text>
  <text class="cat mid" x="347" y="268">B0</text>
  <text class="cat-sub mid" x="347" y="282">held</text>
  <text class="cat-sub mid" x="347" y="295">no seed</text>
  <rect class="them" opacity="0.35" x="419.1" y="106.1" width="14" height="43.1"/>
  <circle class="q1" cx="442.1" cy="124.3" r="2.5"/>
  <circle class="q1" cx="442.1" cy="127.4" r="2.5"/>
  <circle class="q1" cx="442.1" cy="131.5" r="2.5"/>
  <circle class="dot-them" cx="426.1" cy="127.8" r="6"/>
  <text class="v-them mid" x="426.1" y="98.1">32.5%</text>
  <text class="cat mid" x="426.1" y="268">B1</text>
  <text class="cat-sub mid" x="426.1" y="282">held</text>
  <text class="cat-sub mid" x="426.1" y="295">seed</text>
  <rect class="us-soft" x="498.3" y="89.9" width="14" height="44.2"/>
  <circle class="q1" cx="521.3" cy="111.4" r="2.5"/>
  <circle class="q1" cx="521.3" cy="108.4" r="2.5"/>
  <circle class="q1" cx="521.3" cy="115.7" r="2.5"/>
  <circle class="dot-us" cx="505.3" cy="111.8" r="6"/>
  <text class="v-us mid" x="505.3" y="81.9">36.7%</text>
  <text class="cat mid" x="505.3" y="268">C0</text>
  <text class="cat-sub mid" x="505.3" y="282">reprojecting</text>
  <text class="cat-sub mid" x="505.3" y="295">no seed</text>
  <rect class="us-soft" x="577.4" y="87.5" width="14" height="45.9"/>
  <circle class="q1" cx="600.4" cy="112.6" r="2.5"/>
  <circle class="q1" cx="600.4" cy="102.3" r="2.5"/>
  <circle class="q1" cx="600.4" cy="116.1" r="2.5"/>
  <circle class="dot-us" cx="584.4" cy="110.3" r="6"/>
  <text class="v-us mid" x="584.4" y="79.5">37.2%</text>
  <text class="cat mid" x="584.4" y="268">C1</text>
  <text class="cat-sub mid" x="584.4" y="282">reprojecting</text>
  <text class="cat-sub mid" x="584.4" y="295">seed</text>
</svg>
</figure>

Recall climbs the ladder the design predicts. The model alone (A0, 0.4%) says it has no record of what is asked rather than invent it; the seed files alone (A1, 0.5%) add nothing; a single retrieval (D1, 30.6%) adds the memory; re-selection during reasoning (C1, 37.2%) adds the files the question does not name. Full marks: 0, 0, 9, 13, 10, 11 and 12 questions across A0–C1. The direct questions are drawn mechanically, so they include call sites no document describes, which makes the 128 harder than the pilot's 24.

<figure class="fig">
<svg viewBox="0 0 640 294" role="img" aria-label="Figure 9.12-2. The effect of each contrast on mean recall over 128 paired questions, with its 95% bootstrap interval and Holm-corrected Wilcoxon p-value: The memory (C1 − A1 · Holm p = 1.8e-15), +36.7 points (+30.7 to +42.7); Reasoning-time reprojection (C1 − B1 · Holm p = 0.047), +4.6 points (+1.2 to +8.2); Continuous against retrieve-once (C1 − D1 · Holm p = 0.001), +6.5 points (+2.8 to +10.3); Reasoning-time reprojection without the seed (C0 − B0 · Holm p = 0.010), +4.9 points (+1.7 to +8.2); The seed files (C1 − C0 · Holm p = 0.670), +0.4 points (−2.7 to +3.5).">
  <text class="ttl" x="16" y="20">Figure 9.12-2 · What each part of the memory buys</text>
  <text class="ttl-sub" x="16" y="38">paired difference in mean recall, points · 128 questions · 95% interval · faded: Holm p ≥ 0.05</text>
  <path class="grid" d="M372 66 V270 M410.7 66 V270 M449.3 66 V270 M488 66 V270 M526.7 66 V270 M565.3 66 V270 M604 66 V270"/>
  <text class="tick mid" x="372" y="284">−10</text>
  <text class="tick mid" x="410.7" y="284">0</text>
  <text class="tick mid" x="449.3" y="284">+10</text>
  <text class="tick mid" x="488" y="284">+20</text>
  <text class="tick mid" x="526.7" y="284">+30</text>
  <text class="tick mid" x="565.3" y="284">+40</text>
  <text class="tick mid" x="604" y="284">+50</text>
  <path class="parity" d="M410.7 66 V270"/>
  <text class="cat" x="16" y="82">The memory</text>
  <text class="cat-sub" x="16" y="96">C1 − A1 · Holm p = 1.8e-15</text>
  <path class="line-us" d="M529.2 84 H575.7"/>
  <circle class="dot-us" cx="552.5" cy="84" r="6"/>
  <text class="v-us mid" x="552.5" y="74">+36.7</text>
  <text class="cat" x="16" y="122">Reasoning-time reprojection</text>
  <text class="cat-sub" x="16" y="136">C1 − B1 · Holm p = 0.047</text>
  <path class="line-us" d="M415.4 124 H442.3"/>
  <circle class="dot-us" cx="428.6" cy="124" r="6"/>
  <text class="v-us mid" x="428.6" y="114">+4.6</text>
  <text class="cat" x="16" y="162">Continuous against retrieve-once</text>
  <text class="cat-sub" x="16" y="176">C1 − D1 · Holm p = 0.001</text>
  <path class="line-us" d="M421.6 164 H450.4"/>
  <circle class="dot-us" cx="435.9" cy="164" r="6"/>
  <text class="v-us mid" x="435.9" y="154">+6.5</text>
  <text class="cat" x="16" y="202">Reasoning-time reprojection without the seed</text>
  <text class="cat-sub" x="16" y="216">C0 − B0 · Holm p = 0.010</text>
  <path class="line-us" d="M417.2 204 H442.3"/>
  <circle class="dot-us" cx="429.5" cy="204" r="6"/>
  <text class="v-us mid" x="429.5" y="194">+4.9</text>
  <text class="cat" x="16" y="242">The seed files</text>
  <text class="cat-sub" x="16" y="256">C1 − C0 · Holm p = 0.670</text>
  <path class="line-us" opacity="0.45" d="M400.2 244 H424.1"/>
  <circle class="dot-us" opacity="0.45" cx="412.2" cy="244" r="6"/>
  <text class="v-them mid" x="412.2" y="234">+0.4</text>
</svg>
</figure>

| Contrast | Arms | Effect | 95% interval | p (Holm) |
|---|---|---:|---:|---:|
| The memory | C1 − A1 | 36.7 | 30.7 to 42.7 | 1.8 × 10⁻¹⁵ |
| Reasoning-time reprojection | C1 − B1 | 4.6 | 1.2 to 8.2 | 0.047 |
| Continuous against retrieve-once | C1 − D1 | 6.5 | 2.8 to 10.3 | 0.0015 |
| Reasoning-time reprojection without the seed | C0 − B0 | 4.9 | 1.7 to 8.2 | 0.010 |
| The seed files | C1 − C0 | 0.4 | −2.7 to 3.5 | 0.67 |

Every contrast that removes a part of provenance retrieval lowers recall, with an interval clear of zero; the seed files change nothing measurable.

<figure class="fig">
<svg viewBox="0 0 640 384" role="img" aria-label="Figure 9.12-3. Each of the 128 questions' mean recall under reprojection during the reasoning (C1) against the held projection (B1): 51 questions lie above the diagonal, where C1 recalls more, 29 below it, where B1 does, and 48 on it.">
  <text class="ttl" x="16" y="20">Figure 9.12-3 · Question by question</text>
  <text class="ttl-sub" x="16" y="38">each question's mean recall over its runs · 51 above the diagonal, 29 below, 48 on it</text>
  <path class="grid" d="M70 270 H350 M70 200 H350 M70 130 H350 M70 60 H350"/>
  <text class="tick" x="62" y="343.5" text-anchor="end">0%</text>
  <text class="tick" x="62" y="273.5" text-anchor="end">25%</text>
  <text class="tick" x="62" y="203.5" text-anchor="end">50%</text>
  <text class="tick" x="62" y="133.5" text-anchor="end">75%</text>
  <text class="tick" x="62" y="63.5" text-anchor="end">100%</text>
  <path class="grid base" d="M70 340 H350"/>
  <path class="grid" d="M70 60 V340 M140 60 V340 M210 60 V340 M280 60 V340 M350 60 V340"/>
  <text class="tick mid" x="70" y="356">0%</text>
  <text class="tick mid" x="140" y="356">25%</text>
  <text class="tick mid" x="210" y="356">50%</text>
  <text class="tick mid" x="280" y="356">75%</text>
  <text class="tick mid" x="350" y="356">100%</text>
  <path class="parity" d="M70 340 L350 60"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="153.3" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="303.3" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="132.2" cy="215.6" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="182" cy="172" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="116" r="4"/>
  <circle class="us" opacity="0.3" cx="280" cy="83.3" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="144.7" cy="228" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="223.3" r="4"/>
  <circle class="us" opacity="0.3" cx="331.3" cy="78.7" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="275.3" cy="172" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="270" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="130" r="4"/>
  <circle class="us" opacity="0.3" cx="194.4" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="130" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="223.3" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="153.3" r="4"/>
  <circle class="us" opacity="0.3" cx="182" cy="302.7" r="4"/>
  <circle class="us" opacity="0.3" cx="144.7" cy="116" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="106.7" r="4"/>
  <circle class="us" opacity="0.3" cx="238" cy="153.3" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="316.7" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="130" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="316.7" r="4"/>
  <circle class="us" opacity="0.3" cx="186.7" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="153.3" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="280" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="106.7" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="83.3" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="223.3" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="106.7" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="223.3" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="101.1" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="293.3" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="101.1" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="101.1" cy="308.9" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="293.3" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="293.3" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="293.3" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="293.3" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="316.7" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="293.3" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="176.7" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="176.7" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="130" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="215.6" r="4"/>
  <circle class="us" opacity="0.3" cx="194.4" cy="184.4" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="101.1" cy="215.6" r="4"/>
  <circle class="us" opacity="0.3" cx="194.4" cy="184.4" r="4"/>
  <circle class="us" opacity="0.3" cx="326.7" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="130" r="4"/>
  <circle class="us" opacity="0.3" cx="294" cy="97.3" r="4"/>
  <circle class="us" opacity="0.3" cx="326.7" cy="83.3" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="228" r="4"/>
  <circle class="us" opacity="0.3" cx="238" cy="116" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="134.7" r="4"/>
  <circle class="us" opacity="0.3" cx="238" cy="190.7" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="302.7" r="4"/>
  <circle class="us" opacity="0.3" cx="126" cy="302.7" r="4"/>
  <circle class="us" opacity="0.3" cx="144.7" cy="302.7" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="182" cy="265.3" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="312.7" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="107.3" cy="153.3" r="4"/>
  <circle class="us" opacity="0.3" cx="88.7" cy="302.7" r="4"/>
  <circle class="us" opacity="0.3" cx="238" cy="302.7" r="4"/>
  <circle class="us" opacity="0.3" cx="275.3" cy="116" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="316.7" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="270" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="316.7" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="176.7" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="153.3" r="4"/>
  <circle class="us" opacity="0.3" cx="140" cy="316.7" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="223.3" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <text class="t-dim" x="82" y="76">C1 higher</text>
  <text class="t-dim" x="338" y="330" text-anchor="end">B1 higher</text>
  <text class="t-dim mid" x="210" y="372">B1 recall · projection held</text>
  <text class="t-dim" x="70" y="52">C1 recall · reprojecting</text>
  <text class="v-us" x="396" y="110">51</text>
  <text class="cat-sub" x="396" y="126">above the diagonal: C1 recalls more</text>
  <text class="v-them" x="396" y="166">29</text>
  <text class="cat-sub" x="396" y="182">below it: B1 recalls more</text>
  <text class="v-them" x="396" y="222">48</text>
  <text class="cat-sub" x="396" y="238">on it: the same recall</text>
</svg>
</figure>

The gain is broad: 51 of 128 questions sit above the diagonal, 29 below and 48 on it.

<figure class="fig">
<svg viewBox="0 0 640 174" role="img" aria-label="Figure 9.12-4. The gain in mean recall of reprojecting during the reasoning (C1) over the held projection (B1), split by whether the question names its answer file, with 95% bootstrap intervals: Names its file (n=17), −3.1 points (−11.2 to +4.1); Does not (n=111), +5.8 points (+2.2 to +9.7).">
  <text class="ttl" x="16" y="20">Figure 9.12-4 · Where the gain falls</text>
  <text class="ttl-sub" x="16" y="38">C1 − B1 in mean recall, points · 95% bootstrap interval over questions</text>
  <path class="grid" d="M372 66 V150 M418.4 66 V150 M464.8 66 V150 M511.2 66 V150 M557.6 66 V150 M604 66 V150"/>
  <text class="tick mid" x="372" y="164">−15</text>
  <text class="tick mid" x="418.4" y="164">−10</text>
  <text class="tick mid" x="464.8" y="164">−5</text>
  <text class="tick mid" x="511.2" y="164">0</text>
  <text class="tick mid" x="557.6" y="164">+5</text>
  <text class="tick mid" x="604" y="164">+10</text>
  <path class="parity" d="M511.2 66 V150"/>
  <text class="cat" x="16" y="82">Names its file (n=17)</text>
  <text class="cat-sub" x="16" y="96">the question names its answer file</text>
  <path class="line-us" opacity="0.45" d="M407.5 84 H549.4"/>
  <circle class="dot-us" opacity="0.45" cx="482.1" cy="84" r="6"/>
  <text class="v-them mid" x="482.1" y="74">−3.1</text>
  <text class="cat" x="16" y="122">Does not (n=111)</text>
  <text class="cat-sub" x="16" y="136">the question leaves its answer file unnamed</text>
  <path class="line-us" d="M531.3 124 H601.3"/>
  <circle class="dot-us" cx="565.3" cy="124" r="6"/>
  <text class="v-us mid" x="565.3" y="114">+5.8</text>
</svg>
</figure>

The gain lies on the 111 questions that do not name their answer's file, 5.8 points (2.2 to 9.7); on the 17 that do it is −3.1 (−11.2 to 4.1), indistinguishable from zero — a split fixed before any run. What the question names, the opening finds; what it does not, the reasoning reaches, and only a reprojection inside the reasoning places it while the reasoning can use it.

<figure class="fig">
<svg viewBox="0 0 640 324" role="img" aria-label="Figure 9.12-5. The share of answers whose projection held an answer file, by decoded token, with 95% bootstrap bands: C1 from 51.6% at the first token to 57.8% at token 4,096; B1 from 51.6% at the first token to 54.9% at token 4,096; D1 from 48.7% at the first token to 54.9% at token 4,096.">
  <text class="ttl" x="16" y="20">Figure 9.12-5 · Retrieval as the reasoning proceeds</text>
  <text class="ttl-sub" x="16" y="38">answers with an answer file placed, by decoded token · band: 95% bootstrap interval</text>
  <path class="grid" d="M62 210 H616 M62 140 H616 M62 70 H616"/>
  <text class="tick" x="54" y="283.5" text-anchor="end">40%</text>
  <text class="tick" x="54" y="213.5" text-anchor="end">50%</text>
  <text class="tick" x="54" y="143.5" text-anchor="end">60%</text>
  <text class="tick" x="54" y="73.5" text-anchor="end">70%</text>
  <path class="grid base" d="M62 280 H616"/>
  <path class="grid" d="M62 70 V280 M200.5 70 V280 M339 70 V280 M477.5 70 V280 M616 70 V280"/>
  <text class="tick mid" x="62" y="296">0</text>
  <text class="tick mid" x="200.5" y="296">1,024</text>
  <text class="tick mid" x="339" y="296">2,048</text>
  <text class="tick mid" x="477.5" y="296">3,072</text>
  <text class="tick mid" x="616" y="296">4,096</text>
  <text class="t-dim mid" x="339" y="314">decoded token</text>
  <path class="area-us" d="M62 142.6 L70.7 144.4 L79.3 98.8 L88 102.4 L96.6 102.4 L105.3 104.3 L113.9 111.6 L122.6 111.6 L131.3 109.7 L139.9 118.9 L148.6 106.1 L157.2 113.4 L165.9 109.7 L174.5 111.6 L183.2 102.4 L191.8 107.9 L200.5 117 L209.2 115.2 L217.8 106.1 L226.5 111.5 L235.1 104.3 L243.8 97 L252.4 107.9 L261.1 104.3 L269.8 107.9 L278.4 104.3 L287.1 100.6 L295.7 102.4 L304.4 106.1 L313 100.6 L321.7 100.6 L330.3 98.8 L339 102.4 L347.7 102.4 L356.3 104.3 L365 104.2 L373.6 104.3 L382.3 106.1 L390.9 106.1 L399.6 102.4 L408.3 104.3 L416.9 102.4 L425.6 104.3 L434.2 104.3 L442.9 102.4 L451.5 102.4 L460.2 104.3 L468.8 100.6 L477.5 104.3 L486.2 102.4 L494.8 104.3 L503.5 106.1 L512.1 106.1 L520.8 106.1 L529.4 104.3 L538.1 104.3 L546.8 104.3 L555.4 104.3 L564.1 104.3 L572.7 104.3 L581.4 102.4 L590 104.3 L598.7 104.3 L607.3 102.4 L616 102.4 L616 210 L607.3 210 L598.7 211.8 L590 211.8 L581.4 210 L572.7 211.8 L564.1 211.8 L555.4 211.8 L546.8 211.8 L538.1 211.8 L529.4 211.8 L520.8 213.6 L512.1 213.6 L503.5 213.6 L494.8 211.8 L486.2 210 L477.5 211.8 L468.8 208.2 L460.2 211.8 L451.5 210 L442.9 210 L434.2 211.8 L425.6 211.8 L416.9 210 L408.3 211.8 L399.6 210 L390.9 213.6 L382.3 213.6 L373.6 213.6 L365 211.8 L356.3 211.8 L347.7 210 L339 210 L330.3 204.5 L321.7 208.2 L313 206.4 L304.4 213.6 L295.7 210 L287.1 208.2 L278.4 211.8 L269.8 215.5 L261.1 211.8 L252.4 215.5 L243.8 200.9 L235.1 211.8 L226.5 219.1 L217.8 213.6 L209.2 222.8 L200.5 224.6 L191.8 213.6 L183.2 206.4 L174.5 215.5 L165.9 215.5 L157.2 219.1 L148.6 211.8 L139.9 224.6 L131.3 215.5 L122.6 217.3 L113.9 219.1 L105.3 210 L96.6 208.2 L88 208.2 L79.3 208.2 L70.7 255.6 L62 253.8 Z"/>
  <path class="them" opacity="0.12" d="M62 144.4 L70.7 144.4 L79.3 144.4 L88 142.6 L96.6 142.6 L105.3 138.9 L113.9 140.7 L122.6 133.4 L131.3 137.1 L139.9 135.3 L148.6 131.6 L157.2 124.3 L165.9 126.1 L174.5 128 L183.2 126.1 L191.8 133.4 L200.5 133.4 L209.2 128 L217.8 122.5 L226.5 120.7 L235.1 122.5 L243.8 118.9 L252.4 120.7 L261.1 118.9 L269.8 118.9 L278.4 118.9 L287.1 117 L295.7 118.9 L304.4 118.9 L313 117 L321.7 122.5 L330.3 122.5 L339 120.7 L347.7 118.9 L356.3 120.7 L365 126.1 L373.6 122.5 L382.3 118.9 L390.9 120.7 L399.6 122.5 L408.3 122.5 L416.9 122.5 L425.6 120.7 L434.2 118.9 L442.9 120.7 L451.5 120.7 L460.2 120.7 L468.8 120.7 L477.5 120.7 L486.2 120.7 L494.8 122.5 L503.5 120.7 L512.1 120.7 L520.8 122.5 L529.4 122.5 L538.1 122.5 L546.8 120.7 L555.4 118.9 L564.1 122.5 L572.7 118.9 L581.4 118.9 L590 118.9 L598.7 118.9 L607.3 118.9 L616 120.7 L616 230.1 L607.3 228.2 L598.7 228.2 L590 228.2 L581.4 228.2 L572.7 228.2 L564.1 231.9 L555.4 228.2 L546.8 230.1 L538.1 231.9 L529.4 231.9 L520.8 231.9 L512.1 230.1 L503.5 230.1 L494.8 231.9 L486.2 228.2 L477.5 228.2 L468.8 230.1 L460.2 230.1 L451.5 230.1 L442.9 230.1 L434.2 228.2 L425.6 230.1 L416.9 231.9 L408.3 231.9 L399.6 231.9 L390.9 230.1 L382.3 226.4 L373.6 230.1 L365 233.7 L356.3 228.2 L347.7 226.4 L339 228.2 L330.3 230.1 L321.7 230.1 L313 224.6 L304.4 226.4 L295.7 226.4 L287.1 224.6 L278.4 226.4 L269.8 226.4 L261.1 226.4 L252.4 226.4 L243.8 226.4 L235.1 230.1 L226.5 228.2 L217.8 230.1 L209.2 235.5 L200.5 241 L191.8 241 L183.2 233.7 L174.5 235.5 L165.9 231.9 L157.2 231.9 L148.6 239.2 L139.9 242.8 L131.3 244.6 L122.6 242.8 L113.9 250.1 L105.3 248.3 L96.6 251.9 L88 251.9 L79.3 253.8 L70.7 253.8 L62 253.8 Z"/>
  <path class="line-them dash" d="M62 219.1 L70.7 219.1 L79.3 219.1 L88 219.1 L96.6 219.1 L105.3 219.1 L113.9 213.6 L122.6 213.6 L131.3 213.6 L139.9 211.8 L148.6 211.8 L157.2 206.4 L165.9 206.4 L174.5 206.4 L183.2 204.5 L191.8 202.7 L200.5 200.9 L209.2 199.1 L217.8 197.2 L226.5 197.2 L235.1 193.6 L243.8 193.6 L252.4 193.6 L261.1 193.6 L269.8 191.8 L278.4 191.8 L287.1 189.9 L295.7 189.9 L304.4 188.1 L313 184.5 L321.7 184.5 L330.3 182.7 L339 182.7 L347.7 182.7 L356.3 179 L365 177.2 L373.6 177.2 L382.3 177.2 L390.9 177.2 L399.6 177.2 L408.3 177.2 L416.9 177.2 L425.6 177.2 L434.2 177.2 L442.9 177.2 L451.5 177.2 L460.2 177.2 L468.8 177.2 L477.5 175.4 L486.2 175.4 L494.8 175.4 L503.5 175.4 L512.1 175.4 L520.8 175.4 L529.4 175.4 L538.1 175.4 L546.8 175.4 L555.4 175.4 L564.1 175.4 L572.7 175.4 L581.4 175.4 L590 175.4 L598.7 175.4 L607.3 175.4 L616 175.4"/>
  <path class="line-them" d="M62 199.1 L70.7 199.1 L79.3 199.1 L88 197.2 L96.6 197.2 L105.3 193.6 L113.9 195.4 L122.6 188.1 L131.3 191.8 L139.9 189.9 L148.6 186.3 L157.2 179 L165.9 179 L174.5 182.7 L183.2 180.8 L191.8 188.1 L200.5 188.1 L209.2 182.7 L217.8 177.2 L226.5 175.4 L235.1 177.2 L243.8 173.5 L252.4 173.5 L261.1 173.5 L269.8 173.5 L278.4 173.5 L287.1 171.7 L295.7 173.5 L304.4 173.5 L313 171.7 L321.7 177.2 L330.3 177.2 L339 175.4 L347.7 173.5 L356.3 175.4 L365 180.8 L373.6 177.2 L382.3 173.5 L390.9 175.4 L399.6 177.2 L408.3 177.2 L416.9 177.2 L425.6 175.4 L434.2 173.5 L442.9 175.4 L451.5 175.4 L460.2 175.4 L468.8 175.4 L477.5 175.4 L486.2 175.4 L494.8 177.2 L503.5 175.4 L512.1 175.4 L520.8 177.2 L529.4 177.2 L538.1 177.2 L546.8 175.4 L555.4 173.5 L564.1 177.2 L572.7 173.5 L581.4 173.5 L590 173.5 L598.7 173.5 L607.3 173.5 L616 175.4"/>
  <path class="line-us" d="M62 199.1 L70.7 200.9 L79.3 153.5 L88 155.3 L96.6 155.3 L105.3 157.1 L113.9 164.4 L122.6 164.4 L131.3 162.6 L139.9 171.7 L148.6 159 L157.2 166.2 L165.9 162.6 L174.5 162.6 L183.2 153.5 L191.8 160.8 L200.5 169.9 L209.2 168.1 L217.8 159 L226.5 164.4 L235.1 157.1 L243.8 148 L252.4 160.8 L261.1 157.1 L269.8 160.8 L278.4 157.1 L287.1 153.5 L295.7 155.3 L304.4 159 L313 153.5 L321.7 153.5 L330.3 151.7 L339 155.3 L347.7 155.3 L356.3 157.1 L365 157.1 L373.6 159 L382.3 159 L390.9 159 L399.6 155.3 L408.3 157.1 L416.9 155.3 L425.6 157.1 L434.2 157.1 L442.9 155.3 L451.5 155.3 L460.2 157.1 L468.8 153.5 L477.5 157.1 L486.2 155.3 L494.8 157.1 L503.5 159 L512.1 159 L520.8 159 L529.4 157.1 L538.1 157.1 L546.8 157.1 L555.4 157.1 L564.1 157.1 L572.7 157.1 L581.4 155.3 L590 157.1 L598.7 157.1 L607.3 155.3 L616 155.3"/>
  <g class="legend">
  <path class="line-us" d="M330 53 H344"/>
  <text x="348" y="56">C1 reprojecting</text>
  <path class="line-them" d="M457 53 H471"/>
  <text x="475" y="56">B1 held</text>
  <path class="line-them dash" d="M534.4 53 H548.4"/>
  <text x="552.4" y="56">D1 once</text>
  </g>
</svg>
</figure>

With reprojection the share of answers with an answer file in the working set jumps from 51.6% at the opening to 58.1% by the 128th decoded token, inside the reasoning's first two reprojection intervals, and holds there. Held or retrieved once, it stays at the opening's 51.6% or 48.7% until the reasoning ends, and only then climbs — to 54.9% by token 4,096 — as those arms' answers reproject: the file arrives, but after the thought that needed it.

<figure class="fig">
<svg viewBox="0 0 640 188" role="img" aria-label="Figure 9.12-6. Mean recall of the answers by whether their projection held an answer file for the first answer token, with 95% bootstrap intervals: answer file placed (1,025 answers): 51.1% (49.0% to 53.3%); not placed (895 answers): 13.9% (12.0% to 15.8%).">
  <text class="ttl" x="16" y="20">Figure 9.12-6 · Recall when the answer file is in view</text>
  <text class="ttl-sub" x="16" y="38">mean recall by whether an answer file was placed for the first answer token · 95% interval</text>
  <path class="grid" d="M250 66 V162 M308.3 66 V162 M366.7 66 V162 M425 66 V162 M483.3 66 V162 M541.7 66 V162 M600 66 V162"/>
  <text class="tick mid" x="250" y="176">0%</text>
  <text class="tick mid" x="308.3" y="176">10%</text>
  <text class="tick mid" x="366.7" y="176">20%</text>
  <text class="tick mid" x="425" y="176">30%</text>
  <text class="tick mid" x="483.3" y="176">40%</text>
  <text class="tick mid" x="541.7" y="176">50%</text>
  <text class="tick mid" x="600" y="176">60%</text>
  <text class="cat" x="16" y="84">Answer file placed</text>
  <text class="cat-sub" x="16" y="98">1,025 answers</text>
  <rect class="us" x="250" y="75" width="298.3" height="22"/>
  <path class="grid base" d="M535.6 80 V92 M535.6 86 H561.2 M561.2 80 V92"/>
  <text class="v-us" x="569.2" y="90">51.1%</text>
  <text class="cat" x="16" y="130">Not placed</text>
  <text class="cat-sub" x="16" y="144">895 answers</text>
  <rect class="us-soft" x="250" y="121" width="81.2" height="22"/>
  <path class="grid base" d="M320.2 126 V138 M320.2 132 H342.3 M342.3 126 V138"/>
  <text class="v-us" x="350.3" y="136">13.9%</text>
</svg>
</figure>

Over 1,920 answers in the projecting arms, recall is 51.1% (49.0–53.3%) with the answer file placed for the first answer token and 13.9% (12.0–15.8%) without it: retrieval and the answers are one measurement seen twice.

**Ranking, apart from the answers.** A scoring lab (`provenance_gates lab`) probes each question once through the daemon — prefilled and decoded through its thinking — scans its votes against every ingested file on the resident gallery, and re-ranks on the host under formulas read from a file, so every variant scores the same votes. It found the binding limit was votes per token (§6.3); every non-linear alternative tried scored below summed `z × margin` (`docs/results/provenance_scoring_lab_2026-11-16.md`). Over the whole battery, on the gallery of 2026-11-24 — 7,943 files, 36.1M tokens, every probe fresh (`docs/results/battery_128/needle.md`) — 121 questions rank; seven name only files since split away:

| Probe | Questions | hit@1 | hit@5 | hit@10 | hit@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Question and decoded thinking, 128-question battery | 121 | 0.49 | 0.69 | 0.74 | 0.88 | 0.578 |
| Same, the pilot's 23 file-answered questions (2026-11-20) | 23 | 0.65 | 0.83 | 0.87 | | 0.713 |
| The pilot's 23, decoded thinking alone | 23 | 0.61 | 0.83 | 0.83 | | 0.676 |
| The pilot's 23, the question alone | 23 | 0.43 | 0.78 | 0.78 | | 0.573 |
| A random ranking of 7,943 files | | 0.0003 | 0.0016 | 0.0032 | 0.032 | |

<figure class="fig">
<svg viewBox="0 0 640 334" role="img" aria-label="Figure 9.12-7. The share of the 121 questions whose best-ranked answer file lies within the top k of the gallery's 7,943 files, with a 95% bootstrap band over questions: 49% at k = 1, 69% at k = 5, 74% at k = 10, 88% at k = 100; a random ranking of the gallery would reach 3% at k = 100.">
  <text class="ttl" x="16" y="20">Figure 9.12-7 · The needle in the haystack</text>
  <text class="ttl-sub" x="16" y="38">share of 121 questions with an answer file in the top k of 7,943 files · band: 95% bootstrap</text>
  <path class="grid" d="M62 235 H616 M62 180 H616 M62 125 H616 M62 70 H616"/>
  <text class="tick" x="54" y="293.5" text-anchor="end">0%</text>
  <text class="tick" x="54" y="238.5" text-anchor="end">25%</text>
  <text class="tick" x="54" y="183.5" text-anchor="end">50%</text>
  <text class="tick" x="54" y="128.5" text-anchor="end">75%</text>
  <text class="tick" x="54" y="73.5" text-anchor="end">100%</text>
  <path class="grid base" d="M62 290 H616"/>
  <path class="grid" d="M62 70 V290 M145.4 70 V290 M255.6 70 V290 M339 70 V290 M422.4 70 V290 M532.6 70 V290 M616 70 V290"/>
  <text class="tick mid" x="62" y="306">1</text>
  <text class="tick mid" x="145.4" y="306">2</text>
  <text class="tick mid" x="255.6" y="306">5</text>
  <text class="tick mid" x="339" y="306">10</text>
  <text class="tick mid" x="422.4" y="306">20</text>
  <text class="tick mid" x="532.6" y="306">50</text>
  <text class="tick mid" x="616" y="306">100</text>
  <text class="t-dim mid" x="339" y="324">k, the rank cut-off (log scale)</text>
  <path class="area-us" d="M62 162.7 L145.4 142.7 L194.2 137.3 L228.8 120.9 L255.6 119.1 L277.5 117.3 L296.1 117.3 L312.2 113.6 L326.3 110 L339 110 L350.5 110 L360.9 108.2 L370.6 102.7 L379.5 100.9 L387.8 99.1 L395.5 99.1 L402.8 95.5 L409.7 93.6 L416.2 93.6 L422.4 93.6 L428.3 93.6 L433.9 93.6 L439.2 93.6 L444.3 93.6 L449.2 91.8 L453.9 91.8 L458.5 91.8 L462.9 91.8 L467.1 91.8 L471.2 91.8 L475.1 91.8 L478.9 91.8 L482.6 91.8 L486.2 91.8 L489.7 91.8 L493.1 91.8 L496.4 91.8 L499.6 91.8 L502.7 91.8 L505.8 90 L508.7 90 L511.6 90 L514.5 90 L517.2 90 L519.9 90 L522.6 90 L525.2 90 L527.7 90 L530.2 90 L532.6 90 L535 90 L537.3 90 L539.6 90 L541.9 90 L544.1 90 L546.2 90 L548.4 90 L550.5 90 L552.5 90 L554.5 90 L556.5 90 L558.5 88.2 L560.4 88.2 L562.3 88.2 L564.2 88.2 L566 88.2 L567.8 88.2 L569.6 88.2 L571.4 88.2 L573.1 88.2 L574.8 88.2 L576.5 88.2 L578.1 88.2 L579.8 88.2 L581.4 86.4 L583 86.4 L584.6 86.4 L586.1 86.4 L587.6 86.4 L589.2 86.4 L590.7 86.4 L592.1 86.4 L593.6 86.4 L595 86.4 L596.4 84.5 L597.9 84.5 L599.2 84.5 L600.6 84.5 L602 84.5 L603.3 84.5 L604.7 84.5 L606 84.5 L607.3 84.5 L608.6 84.5 L609.8 84.5 L611.1 84.5 L612.3 84.5 L613.6 84.5 L614.8 84.5 L616 84.5 L616 108.2 L614.8 108.2 L613.6 108.2 L612.3 108.2 L611.1 108.2 L609.8 108.2 L608.6 108.2 L607.3 111.8 L606 111.8 L604.7 111.8 L603.3 111.8 L602 111.8 L600.6 111.8 L599.2 111.8 L597.9 111.8 L596.4 111.8 L595 113.6 L593.6 113.6 L592.1 113.6 L590.7 113.6 L589.2 113.6 L587.6 113.6 L586.1 113.6 L584.6 113.6 L583 113.6 L581.4 113.6 L579.8 115.5 L578.1 115.5 L576.5 115.5 L574.8 115.5 L573.1 115.5 L571.4 115.5 L569.6 115.5 L567.8 115.5 L566 115.5 L564.2 115.5 L562.3 115.5 L560.4 115.5 L558.5 115.5 L556.5 117.3 L554.5 117.3 L552.5 117.3 L550.5 117.3 L548.4 117.3 L546.2 117.3 L544.1 117.3 L541.9 117.3 L539.6 117.3 L537.3 117.3 L535 117.3 L532.6 117.3 L530.2 117.3 L527.7 117.3 L525.2 117.3 L522.6 117.3 L519.9 117.3 L517.2 117.3 L514.5 117.3 L511.6 117.3 L508.7 117.3 L505.8 117.3 L502.7 119.1 L499.6 120.9 L496.4 120.9 L493.1 120.9 L489.7 120.9 L486.2 120.9 L482.6 120.9 L478.9 120.9 L475.1 120.9 L471.2 120.9 L467.1 120.9 L462.9 120.9 L458.5 120.9 L453.9 120.9 L449.2 120.9 L444.3 124.5 L439.2 124.5 L433.9 124.5 L428.3 124.5 L422.4 124.5 L416.2 124.5 L409.7 124.5 L402.8 126.4 L395.5 130 L387.8 130 L379.5 131.8 L370.6 135.5 L360.9 140.9 L350.5 142.7 L339 142.7 L326.3 142.7 L312.2 146.4 L296.1 151.8 L277.5 151.8 L255.6 155.5 L228.8 157.3 L194.2 173.6 L145.4 180.9 L62 202.7 Z"/>
  <path class="line-them dash" d="M62 289.9 L145.4 289.9 L194.2 289.8 L228.8 289.7 L255.6 289.6 L277.5 289.6 L296.1 289.5 L312.2 289.4 L326.3 289.4 L339 289.3 L350.5 289.2 L360.9 289.2 L370.6 289.1 L379.5 289 L387.8 288.9 L395.5 288.9 L402.8 288.8 L409.7 288.7 L416.2 288.7 L422.4 288.6 L428.3 288.5 L433.9 288.4 L439.2 288.4 L444.3 288.3 L449.2 288.2 L453.9 288.2 L458.5 288.1 L462.9 288 L467.1 288 L471.2 287.9 L475.1 287.8 L478.9 287.7 L482.6 287.7 L486.2 287.6 L489.7 287.5 L493.1 287.5 L496.4 287.4 L499.6 287.3 L502.7 287.3 L505.8 287.2 L508.7 287.1 L511.6 287 L514.5 287 L517.2 286.9 L519.9 286.8 L522.6 286.8 L525.2 286.7 L527.7 286.6 L530.2 286.5 L532.6 286.5 L535 286.4 L537.3 286.3 L539.6 286.3 L541.9 286.2 L544.1 286.1 L546.2 286.1 L548.4 286 L550.5 285.9 L552.5 285.8 L554.5 285.8 L556.5 285.7 L558.5 285.6 L560.4 285.6 L562.3 285.5 L564.2 285.4 L566 285.3 L567.8 285.3 L569.6 285.2 L571.4 285.1 L573.1 285.1 L574.8 285 L576.5 284.9 L578.1 284.9 L579.8 284.8 L581.4 284.7 L583 284.6 L584.6 284.6 L586.1 284.5 L587.6 284.4 L589.2 284.4 L590.7 284.3 L592.1 284.2 L593.6 284.1 L595 284.1 L596.4 284 L597.9 283.9 L599.2 283.9 L600.6 283.8 L602 283.7 L603.3 283.7 L604.7 283.6 L606 283.5 L607.3 283.4 L608.6 283.4 L609.8 283.3 L611.1 283.2 L612.3 283.2 L613.6 283.1 L614.8 283 L616 282.9"/>
  <path class="line-us" d="M62 182.7 L145.4 162.7 L194.2 155.5 L228.8 139.1 L255.6 137.3 L277.5 133.6 L296.1 133.6 L312.2 130 L326.3 126.4 L339 126.4 L350.5 126.4 L360.9 124.5 L370.6 119.1 L379.5 115.5 L387.8 113.6 L395.5 113.6 L402.8 110 L409.7 108.2 L416.2 108.2 L422.4 108.2 L428.3 108.2 L433.9 108.2 L439.2 108.2 L444.3 108.2 L449.2 106.4 L453.9 106.4 L458.5 106.4 L462.9 106.4 L467.1 106.4 L471.2 106.4 L475.1 106.4 L478.9 106.4 L482.6 106.4 L486.2 106.4 L489.7 106.4 L493.1 106.4 L496.4 106.4 L499.6 106.4 L502.7 104.5 L505.8 102.7 L508.7 102.7 L511.6 102.7 L514.5 102.7 L517.2 102.7 L519.9 102.7 L522.6 102.7 L525.2 102.7 L527.7 102.7 L530.2 102.7 L532.6 102.7 L535 102.7 L537.3 102.7 L539.6 102.7 L541.9 102.7 L544.1 102.7 L546.2 102.7 L548.4 102.7 L550.5 102.7 L552.5 102.7 L554.5 102.7 L556.5 102.7 L558.5 100.9 L560.4 100.9 L562.3 100.9 L564.2 100.9 L566 100.9 L567.8 100.9 L569.6 100.9 L571.4 100.9 L573.1 100.9 L574.8 100.9 L576.5 100.9 L578.1 100.9 L579.8 100.9 L581.4 99.1 L583 99.1 L584.6 99.1 L586.1 99.1 L587.6 99.1 L589.2 99.1 L590.7 99.1 L592.1 99.1 L593.6 99.1 L595 99.1 L596.4 97.3 L597.9 97.3 L599.2 97.3 L600.6 97.3 L602 97.3 L603.3 97.3 L604.7 97.3 L606 97.3 L607.3 97.3 L608.6 95.5 L609.8 95.5 L611.1 95.5 L612.3 95.5 L613.6 95.5 L614.8 95.5 L616 95.5"/>
  <circle class="dot-us" cx="62" cy="182.7" r="5.5"/>
  <text class="v-us" x="72" y="174.7">49% at 1</text>
  <circle class="dot-us" cx="255.6" cy="137.3" r="5.5"/>
  <text class="v-us" x="265.6" y="129.3">69% at 5</text>
  <g class="legend">
  <path class="line-us" d="M330 53 H344"/>
  <text x="348" y="56">best answer-file rank</text>
  <path class="line-them dash" d="M494.2 53 H508.2"/>
  <text x="512.2" y="56">random ranking</text>
  </g>
</svg>
</figure>

The answer file is first of 7,943 on 49% of questions and in the top five on 69%, about 1,500 and 430 times a random ranking — by sign agreement alone, with no text index, no embedding model and no file name in the query. By kind it is first on 43% of direct, 52% of transitive and 50% of architectural questions. Most misses lose their top places to the same few files that score on every question (`tests_grid.rs`, `file_grep.yaml`, `tasks.rs`, `AGENTS.md`).

**The pilot.** A 24-question pilot, run on production from 2026-11-10 as retrieval changed (57.6% → 62.2% by 2026-11-20; `docs/results/dependency_battery_2026-11-20_b.md`), first measured reasoning-time reprojection with a daemon-wide switch, `reproject_in_thinking: false`, which passes over every boundary inside the think block and keeps the rest — the opening, the answer's boundaries, and one reprojection on `</think>`. Over three runs per arm at one build (`docs/results/dependency_battery_2026-11-23_l_m.md`), reprojecting averaged **57.3%** and holding **38.8%**; every held run scored below every reprojecting run (p = 0.05, one-sided permutation), and per question reprojection won 14 to 4 (sign test, p = 0.031). Its loss fell where the answer lives in a file the question does not name. The 128-question battery measures the same contrast at 4.6 points: smaller, and with an interval clear of zero.

**What the answers show.** The memory holds documentation and module structure more than the call graph. Answers cite their sources — a design document's section, a module header — and when the model had not read the source it said so, declining rather than inventing; the errors found on reading are near-misses (a constant placed in the header beside the one that defines it, an out-of-date caller), not invented names.

**What is public.** The source, the raw measurement rows, the battery with its ground truth and analysis (`zend/examples/dependency_battery/`, `docs/results/battery_128/`), the pilot's results and the scoring lab with its formulas (`candle-conversation/examples/provenance_gates`) are in the repository (Appendix A). The live system is open to signed-in users; the ingested conversation is not released.

---

### 9.13 The System as Its Own Workload

The system is also in use, and the use exercises what no benchmark can: memory that outlives the process that wrote it, written by one model and read by another. The journal store — one master journal per group, with `people/`, `conversations/`, `topics/`, `threads/` and `bots/` — holds minute-stamped records written by whichever agent is speaking, each naming its author and account, so the memory is in no model's weights and swapping the reader does not orphan it. Entries fold as the KV cache does, steps into days into weeks into years, so a years-long journal reads in one prompt. Each group declares what its members may read; provenance ranks only what a reader may see. The retrieval numbers of §9.11.1 and §9.12 come from this live gallery, scanned over HTTP while it is written to. What this buys — sessions that survive restarts, a model swap that costs a re-ingest, an attributable record across agents and months — is operating experience, not a measured result, and is reported as such.

#### 9.13.1 This revision, as a workload

Much of an earlier draft of this version was revised by the system: a Flash-Next session on the production daemon, asked to critique and then revise the paper, ran 85 tool rounds over ~320K tokens and made thirty-one edits (`docs/results/paper_revision_session_2026-11-09.md`). Asked to fill §9.12's then-empty results table, it searched for a measured table, found none, wrote that the battery was not yet run, and corrected the other places that claimed its results; it missed some, which a later review found. Its failures were mechanical — stale anchors after neighbouring edits — a defect of the editing tools.

It decoded one token for every 2.9 prefilled (82,963 against 238,619). Every token was prefilled once, the earlier turns' K/V projected back rather than resent, where a client re-sending its history would have sent on the order of eighty times as much. Decode held at ~102 t/s at 48K–130K tokens, ~110 at 162K–237K and ~106 at 309K–321K: the working set does not grow with the history, though the session does not separate provenance's bound from Flash-Next's own indexer (§9.6.2). That the model read less because what it read came back is the right hypothesis, and the session is consistent with it; the run that would establish it replaces provenance with a full window.

OpenRouter's picture of agent traffic is ~87% cached prompt, 12% novel prompt and 1% output, "because an agent re-sends its whole context on every step" [OpenRouter, *Cache is Crucial*, October 2026; 50]; on SWE-bench Verified under OpenHands, input tokens dominate the cost of agentic coding even with caching [Bai et al., arXiv:2604.22750; 51]. Against genuinely new input, this session wrote about four times as much per token read (Figure 9.13-1). OpenRouter's novel prompt includes re-reads a provider failed to cache, so this is a comparison, not a result; counted as a re-sending client counts it, ~19M tokens, the session would be 1 : 230. What the architecture changes is not how much a long session would re-read, but that it does not.

<figure class="fig">
<svg viewBox="0 0 640 128" role="img" aria-label="Figure 9.13-1. Tokens written per token of new input. This revision session decoded one token for every 2.9 prefilled, 0.345. OpenRouter's agentic mix, output against novel prompt, about one in twelve, 0.083.">
  <text class="ttl" x="16" y="20">Figure 9.13-1 · Written per token newly read</text>
  <text class="ttl-sub" x="16" y="36">decoded or output tokens ÷ new input tokens · amber: published agent traffic</text>
  <path class="grid base" d="M200 48 V104"/>
  <text class="cat" x="16" y="64">This session</text>
  <text class="cat-sub" x="16" y="77">82,963 ÷ 238,619</text>
  <rect class="us" x="200" y="54" width="310.3" height="16"/>
  <text class="v-us" x="516.3" y="67">1 : 2.9</text>
  <text class="cat" x="16" y="94">OpenRouter, agents</text>
  <text class="cat-sub" x="16" y="107">1% output ÷ 12% novel</text>
  <rect class="them" x="200" y="84" width="75" height="16"/>
  <text class="v-them" x="281" y="97">≈ 1 : 12</text>
</svg>
</figure>

A restart during the session resumed its open tool round on a second member while the first still ran it, and the duplicate rounds reverted one edit; the files were rebuilt edit by edit from the session's own record, and the fault — a resumed turn that took no claim on its conversation — is fixed. The record outlived both processes that wrote it.

## 10. Analysis

### 10.1 Why Coherence Produces Disproportionate Results

Each component has precedents; their combination does not. More concurrent requests coalesce more expert work, shrinking the hot set the Markov prefetch must cover. Refreshed activations make the selector's thresholds measure real quality and keep ε_hot ≈ 0. Turn-aligned blocks make signatures temporally coherent. V blocks landing at Q2–Q3 fit 3× more tokens in the warm tier, and the theorem bounds what that costs: however deep the cold tier grows, at most the retrieval budget enters a step, each block within its threshold. Above all, provenance selection bounds the working set independently of N and the refresh keeps the hot floor near zero; together they give §11.2's bound, which no KV quantization system states, because none decouples depth from the working set.

### 10.2 The VRAM Budget

The engine has no fixed shares: one span per card, sized before the weights load, divided between `| persistent tensors | KV regions | per-wave transient tier | expert weights |` with one elastic boundary between the KV side and the experts, which each side re-checks before crossing. Concurrent sessions take ground from the expert zone, which streams more experts through the three-tier cache, and an idle card gives it back (Figures 9.6-4 and 9.6-5). The working set per session is each step's quality ceiling, so more VRAM buys concurrency or working set at the operator's choice, and compression multiplies either: at C5 a GiB of KV holds 4.15× the tokens of BF16.

---

## 11. Discussion

### 11.1 The Baseline Problem: Error Growth in Full-Attention Systems

KIVI, KVQuant, TurboQuant and GEAR all work under full attention, where two mechanisms tie stored error to depth. *Sequential*: each decoded token's K/V is computed over already-compressed predecessors, so error enters the residual stream and is inherited [GEAR, Kang et al., 2024]. *Exposure*: all N cached tokens enter every step, so the stored error a step reads — and the number of perturbed scores competing for attention mass — grows with N. Neither is specific to compression; F16 rounding is stored error too.

Exposure is not output error. Attention's output is a convex combination of values, so value errors are averaged, not summed, and whether full-attention output error grows with N depends on the attention distribution; we do not prove that it does. Measured, it does not within the windows we reach (§9.6.8, Figure 9.6-3): C10's divergence from BF16 on Qwen3.5-9B is 6.5e-3 nats at 4K and 3.5e-3 at 124K, and every whole-prefix model of §9.6 passes the story gate at C10 at every depth run. What the literature lacks is a bound on exposure independent of depth. More VRAM defers the threshold, but a context persistent across months is 100 million tokens, which no card holds at F16, and summarization and eviction discard the verbatim detail. §11.2 shows decoupling the working set from depth is sufficient; no full-attention system has that property, because its working set is the whole context.

### 11.2 Theorem: Asymptotic Numerical Stability Under Provenance-Selected Attention

ε(t) denotes the **total stored numerical error** of token t's K/V from every source — quantization, rounding, decode drift. The quantity bounded is the sum of ε(t) over the tokens entering a step: that step's exposure, not its output error. The proof uses nothing specific to compression.

<figure class="fig">
<svg viewBox="0 0 640 224" role="img" aria-label="Figure 11-1. The bound. Under full attention a step's exposure, the summed stored error of the tokens it reads, grows linearly with context depth N; more VRAM only moves the point where tokens must be compressed or evicted. Under provenance-selected attention it is bounded by epsilon hot plus the retrieval budget times epsilon ret, at any depth.">
  <text class="ttl" x="16" y="20">Figure 11-1 · What a step is exposed to, as the context grows</text>
  <text class="ttl-sub" x="16" y="36">summed stored error of the tokens a step reads · schematic</text>
  <path class="ax" d="M70 190 H610 M70 190 V52"/>
  <path class="ax thin" d="M210 56 V190 M330 56 V190"/>
  <path class="c-rise" d="M70 186 L600 58"/>
  <path class="c-flat" d="M70 178 C110 156 140 150 180 148 H600"/>
  <text class="t-rise" x="470" y="78">full attention: Θ(N · ε̄)</text>
  <text class="t-flat" x="390" y="140">provenance: ε_hot + W_warm_max · ε_ret</text>
  <text class="t-dim mid" x="210" y="204">16 GB fills</text>
  <text class="t-dim mid" x="330" y="204">80 GB fills</text>
  <text class="t-dim" x="580" y="206">depth N →</text>
  <text class="t-dim" x="76" y="64">Σ ε(t)</text>
</svg>
</figure>

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

**The contrast is between bounds** (Figure 11-1). Under full attention the exposure is Θ(N · ε̄); under provenance selection the system at 10 million turns works under the same bound as at a thousand. This is not a demonstration that full-attention output degrades — §11.1 gives the reasons it may not, and within 128K it does not (§9.6.8). What full attention lacks is the bound, for the regime past a single context where every token held must be compressed or evicted.

**Empirical status.** The premises — a fixed budget, a fixed hot tier, a per-block threshold on everything sealed — hold by construction, so the measurements are consistent with the theorem rather than tests of it. Flash-Next's flat depth (§9.6.2) is its own indexer inside one context, supporting the principle, not the mechanism; the live session of §9.13.1 keeps decode flat from 48K to 321K with the indexer as a confound. The direct test of the premise is §9.11.2: with the slot capped at 32K and a history twelve times that, reprojection selected every planted turn at C10 and C0 alike, where one-shot and rolling-window arms selected none. §9.12 tests what the model does with the selected set; the theorem does not rest on it.

**Compression moves the constant, never the bound.** A selected cold block contributes its full error at any depth, so aggressive levels on cold K and on V raise ε_cold — a capacity trade bounded by the threshold each block was sealed under. The binding constraints are ε_hot, kept near zero by the refresh, and the retrieval threshold θ. The bound fails only if selection grows the working set with N, which recovers the full-attention regime.

### 11.3 Compression Architecture vs. Compression Primitive

KIVI, KVQuant and TurboQuant optimise the compression *primitive*: given that every block gets a format, which format is best. This paper optimises the *architecture*: when to quantize, what quality to guarantee per block, how to separate error sources, and how error scales with depth. Per-block selection absorbs a primitive's failures — a block Q4_0 cannot represent gets Q8_0 — which is why the adaptive system holds quality where a rotation fails at 4–5×. KVQuant's outlier insight is subsumed by the `_KS` sub-block formats and sink protection. On Llama-3.2-3B, adaptive C5 at 3.27× passes the multi-session identity test (§9.8) where uniform Q4_0 at 3.17× fails, and the query-aware K metric removed the cross-session contamination an isotropic metric produced at C8–C9.

### 11.4 Attention Was Always Retrieval

Attention is retrieval: Q queries K, scores select, V is aggregated by relevance. This system runs it at two scales — approximate 1-bit attention over the whole unbounded context to select the working set (§6.6), then exact attention over what it selected — hierarchical retrieval at constant attention cost per step, where standard attention is flat retrieval over N. The trade is real: full attention reads all N exactly, while this system can surface the wrong B tokens. The guarantee is bounded error on the attended set, not the relevance of the set; it trades exposure that grows with N for a retrieval-quality problem the provenance mechanism bounds but cannot eliminate. That is a different trade-off from full attention, not a strict improvement on it.

It is not RAG, and the difference is structural. *The signal*: RAG retrieves from text, so iterative RAG (FLARE, Self-RAG) must wait for the model to write a query; Q vectors carry what the model is reaching for before any text forms. *Causal continuity*: RAG injects text into a fresh context, while this system promotes stored K/V computed in its original causal context, K stored un-rotated and positioned on read. *Leading the answer*: reprojection inside the think block, whose K/V is dropped at seal, steers the answer's context by everything the model thought (§6.5). A RAG system given Q→Q matching, causal K/V continuity and retrieval from Q during generation would no longer be RAG; it would be this architecture. §9.12 shows it working: the answer file found among 7,943 by sign agreement alone, and a single retrieval before reasoning 6.5 points below continuous reprojection.

### 11.5 Constraint-Driven Innovation as a Methodology

The system was designed bottom-up from two constraints accepted together: the hardest form of persistent memory, and a hardware ceiling that could not be exceeded. Each closed a standard solution and forced a replacement that holds beyond the constraint:

- **Native quantized kernels.** Dequantising to BF16 OOMed at 16 GB; inline dequantisation never materialises the copy, and saves its traffic on every card (§8.1).
- **Provenance-selected attention.** The full context could not be held hot; selecting a bounded working set turned out to be the property the theorem uses.
- **Two-phase quantization.** Materialising turns at F16 across hundreds of sessions OOMed; sealing per block and refreshing per turn separates two error mechanisms the literature conflates.
- **Online expert prediction.** At 44% residency prediction was required; learning online from live routing covers 4.7× the cold experts popularity does, with no calibration (§9.7).
- **Live structural segments.** Rotating system-prompt sections made stored template K/V silently wrong; storing only portable content and prefilling the template against the live prefix closes the error on attention-only models (§5).

In each case the replacement is more efficient in the dimension the constraint bounds, and that efficiency generalised because the constraint exposed an avoidable cost. It is a pattern from one system, not a law, and its counterexample is in the list: the structural-segment design had to be set aside for the recurrent production model. The evaluation followed the same rule — the system's own codebase as the test, checkable by anyone who reads it.

---

## 12. Conclusion

The four contributions — online Markov expert prediction with wave-batched GEMM, per-block adaptive KV quantization with two-phase refresh, attentional provenance indexing with mid-decode reprojection, and an unbounded three-tier context — have precedents individually and had not been integrated.

Together they run a 180B MoE model for eight sessions on a 16 GB laptop GPU, above every published single-GPU run of it on hosts with two to four times the memory; keep throughput flat from 32K to 128K, through the model's own sparse-attention indexer with no depth-dependent cost from the engine; compress the KV cache up to 7.6× inline with every output validated, nearly 4× llama.cpp's q8_0 for less decode; and serve concurrent sessions at 1.3–37.9× the best published single-stream rate for the same model and card class, up to 2,869.9 t/s across 64 sessions, with a 284B model at sixteen sessions on one GPU at 4.1× the best published single-GPU decode. The memory behind it is a 42M-token code substrate held at C5; under sixteen concurrent answers the card's resident context peaks at 945 thousand tokens, 21.9 GiB at full precision, while the expert zone streams the rest of the model. The most comparable published study found standard frameworks degrade beyond two concurrent users on an RTX 5090 [Herz et al., arXiv:2512.23029].

The Asymptotic Numerical Stability theorem bounds the summed stored error entering a step by a constant independent of depth, where full attention's grows linearly. It bounds exposure, not output error, and it does not show full attention degrades; what it changes is the question the KV quantization literature asks — not how small each token's error can be made inside the full-attention regime, but whether the regime can be left. Provenance selection leaves it, and §9.11.2 shows the premise holds: every planted turn selected from a history twelve times the working set, at the top compression level as at near-lossless.

On the system's own codebase, retrieval ranks the answer file first among 7,943 on 49% of questions by sign agreement alone, and answering 128 questions from memory with tools forbidden, the memory is worth 36.7 points of recall, re-selection during reasoning 4.6, and continuous re-selection 6.5 over a single retrieval, each clear of zero after Holm correction; the gain falls on the questions that do not name their answer's file, the ones only the reasoning can reach. Attention is retrieval: this system runs it hierarchically, approximate over the unbounded context and exact over what it selects, preserving the mechanism RAG replaces and removing the scaling constraint on exact attention. The deeper lesson is methodological: constraints accepted as design requirements forced decisions more efficient in the dimension they bound, measured here on three cards from 16 GB to 72 GB, with a bound that carries to any card.

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

[14] Kamahori, K., et al. (2024). **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.**

[15] DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference. DATE 2025. https://arxiv.org/html/2501.10375

[16] KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models. SOSP 2025.

[17] NVIDIA Technical Blog. (2025). **Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache.** https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/

[18] KVSink: Sink-Aware KV Cache Quantization for Attention Sinks. COLM 2025. https://arxiv.org/abs/2508.04257

[19] Quantization Error Propagation in Large Language Models. arXiv:2504.09629, 2025.

[20] Kamradt, G. (2023). **LLM Test: Needle In A Haystack.** https://github.com/gkamradt/LLMTest_NeedleInAHaystack

[21] Kuratov, Y., et al. (2024). **BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.** NeurIPS 2024. https://arxiv.org/abs/2406.10149

[22] hardware-corner.net. (2025). **The Definitive GPU Ranking for LLMs: Token Generation & Prompt Processing Performance.** https://www.hardware-corner.net/gpu-ranking-local-llm/

[23] ToolHalla. (2026). **Best Local LLMs for RTX 4090 in 2026: 7 Models That Maximize 24GB.** https://toolhalla.ai/blog/best-local-llms-rtx-4090-2026

[24] Herz, M., et al. (2025). **Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware.** arXiv:2512.23029.

[25] Jiang, Z., et al. (2023). **FLARE: Active Retrieval Augmented Generation.** EMNLP 2023. https://arxiv.org/abs/2305.06983

[26] Asai, A., et al. (2024). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** ICLR 2024. https://arxiv.org/abs/2310.11511

*Published throughput figures compared against in §9.6. Each is recorded with its card, quantization, context length and engine in `docs/performance.md` §6.*

[27] llama.cpp Discussion #15013. **Performance of llama.cpp on Nvidia CUDA** (Llama-2-7B Q4_0). https://github.com/ggml-org/llama.cpp/discussions/15013

[28] Hardware Corner. (2026). **RTX 3090 LLM benchmarks.** https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-3090/

[29] Hardware Corner. (2026). **RTX 4080 LLM benchmarks.** https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4080/

[30] jonidimo. (2026). **Qwen3.8-27B on One RTX 3090: 14h Measured Benchmark.** https://jonidimo.github.io/qwen38-3090-benchmark/benchmark.html

[31] ryan4yin. (2026). **Best llama.cpp config for Qwen3.8-Flash-Next (RTX 4090 24GB).** https://gist.github.com/ryan4yin/48617bbddacc7067f10799770b7cc33f

[32] unsloth/Qwen3.8-Flash-Next-GGUF discussion #3. (2026). **Share your model speed here.** https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3

[33] InsiderLLM. (2026). **Best Way to Run Qwen 3.6 35B MoE Locally.** https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/

[34] llama.cpp Discussion #19890. (2026). **RTX 5090 (CUDA) vs Radeon AI PRO R9700 (Vulkan) — Qwen3.5-35B-A3B.** https://github.com/ggml-org/llama.cpp/discussions/19890

[35] SOTAAZ. (2026). **llama.cpp KV Cache Quantization, Measured on One A100.** https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en

[36] vLLM Blog. (2026). **The State of FP8 KV-Cache and Attention Quantization in vLLM.** https://vllm.ai/blog/2026-04-22-fp8-kvcache

[37] vLLM Blog. (2026). **A First Comprehensive Study of TurboQuant: Accuracy and Performance.** https://vllm.ai/blog/2026-05-11-turboquant

[38] ByteShape. (2026). **If It Fits, It Sits: Qwen 3.6 35B.** https://byteshape.com/blogs/Qwen3.6-35B-A3B/

[39] Hardware Corner. (2026). **RTX 5090 LLM benchmarks.** https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5090/

[40] Millstone AI. (2026). **Qwen3.5-35B-A3B FP8 on 1× RTX Pro 6000.** https://www.millstoneai.com/inference-benchmark/qwen3-5-35b-a3b-fp8-1x-rtx-pro-6000-blackwell

[41] Millstone AI. (2026). **Qwen3.6-35B-A3B FP8 on 1× RTX Pro 6000.** https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-fp8-1x-rtx-pro-6000-blackwell

[42] holy_fox. (2026). **Running Qwen3.8-Flash-Next on an RTX 5090 with 128GB RAM using llama.cpp.** https://zenn.dev/holy_fox/articles/04887ff8177b87?locale=en

[43] llama.cpp PR #24162. (2026). **DeepSeek V4.** https://github.com/ggml-org/llama.cpp/pull/24162

[44] RockmSockmJesus. (2026). **DeepSeek-V4-Flash (284B MoE) at ~28 tok/s on 1x RTX 5090.** https://gist.github.com/RockmSockmJesus/30a195ccd9b62e981ec2676a99a57b7e

*Further works cited in the text by author and year.*

[45] Staniszewski, K., Łańcucki, A. (2026). **KV Cache Transform Coding for Compact Storage in LLM Inference.** ICLR 2026. https://arxiv.org/abs/2511.01815

[46] Zhong, S., Sun, Y., Liang, L., et al. (2025). **HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference.** DAC 2025. https://arxiv.org/abs/2504.05897

[47] OpenRouter. (2026). **Cache is Crucial.** *Inferred*, no. 3. https://openrouter.ai/data/newsletter/cache-is-crucial

[48] Bai, L., Huang, Z., Wang, X., et al. (2026). **How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks.** arXiv:2604.22750. https://arxiv.org/abs/2604.22750

[17], [19], [22] and [23] are background reading and are not cited in the text.

---

## Appendix A: Public Release and Community Collaboration

**Source code:** https://github.com/john-sharratt/candle (public)

**Live system:** https://code.tokera.com — the deployed coding assistant over this repository, on the cluster that produced §9.11.1 and §9.13. It requires sign-in; access is by allowlist, with a queue for guests. The project site, https://www.tokera.com, carries v1 and this version.

**What is released.** The engine, and the raw measurement rows behind every figure — `docs/performance.md` and `docs/results/`, each naming its build, machine and test — with the tests that produce them; the §9.12 battery's questions, ground truth, harness, analysis and per-answer scores (`zend/examples/dependency_battery/`, `docs/results/battery_128/`, and the pilot's `docs/results/dependency_battery_2026-11-*.md`); and the tests behind §9.6.8 and §9.11.2. Not released: the live substrate, a KV archive valid only for its model and holding conversations that are not this paper's to publish, and a view of which signatures matched at each reprojection. A turn's `projection`, `seed` and reprojection are dials of the request, so every arm of §9.12 is a setting a caller can choose.

**Invitation.** v2 is open for critical review. The v1 erratum stays part of the record; every figure here names its build, machine and test, and every unmeasured claim says so. Validation, stress tests and optimisations are welcome through the GitHub repository, and contributors will be recognised in the next revision.