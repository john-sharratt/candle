# One Card, One Stack: Constraint-Driven Architecture for Asymptotically Stable Inference over Unbounded Agent Memory

> **Version 2**, as published 25 November 2026 ([10.5281/zenodo.24529143](https://zenodo.org/records/24529143)). v1 ([10.5281/zenodo.20156060](https://zenodo.org/records/20156060)) and its erratum: Appendix A.

**Abstract**

Persistent agents need context that grows without bound. Under full attention every cached token takes part in every later step, so the stored error a step is exposed to grows with context depth, and more VRAM only defers the point where tokens must be compressed or evicted. We argue the coupling is architectural, and that decoupling the working set from context depth is sufficient to bound the exposure.

**Theorem (§11.2).** Under provenance-selected attention over a tiered context, the change stored K/V error makes to each attention head's output is bounded by a constant, and a token the step needs keeps an attention weight bounded below by a constant, both independent of context depth N, so the step's signal-to-perturbation ratio is bounded at any depth. Under full attention the needed token's weight falls as 1/N against the same perturbation, and the summed stored error a step reads grows linearly in N.

The system combines (1) online Markov expert prediction with wave-batched grouped GEMM for MoE inference under partial VRAM residency; (2) per-block adaptive KV quantization across eleven levels (C0–C10) with asymmetric K/V error metrics and a prefill refresh at every turn seal, compressing the KV cache 4.1–7.6× at the top level across the fleet's compressing models; (3) attentional provenance indexing — Q-vector sign signatures scanned on the GPU, re-selecting the working set during decode; and (4) an unbounded three-tier context, VRAM-hot, RAM-warm and disk-cold.

Implemented in Rust on a Candle fork, the system runs a **180B-parameter MoE model on a 16 GB laptop GPU**, decoding one session at 34.9 t/s — above the 27.5–29 published for an RTX 5080, the same 16 GB class with twice the host RAM — and eight at 108.5 t/s aggregate, above every published llama.cpp run of the model and every published multi-session figure for it. On a 72 GB RTX PRO 5000, from 32K to 128K tokens it keeps **96–99% of its prefill and 93–111% of its decode** across two tasks. One card's aggregate decode is **1.4–37.9×** the best published llama.cpp single-stream rate for the same model and card class, reaching **2,869.9 t/s across 64 sessions**, and a **284B model serves sixteen sessions from one GPU** at 4.1× the best published single-GPU decode.

On the system's own 2.2M-line codebase, retrieval ranks one of a question's answer files first among 7,943 files on 49% of 121 questions and in the top five on 69%, by Q-signature agreement alone. Answering 128 questions from memory with tools forbidden, over seven memory arms and three to seven runs each, the memory is worth **37.0 points of recall**, re-selection during reasoning 5.3, and continuous re-selection 6.9 over a single retrieval, every interval clear of zero (§9.12). Mid-decode reprojection selects every fact planted in a history twelve times the working set, at the top compression level as at near-lossless, where a single projection and a rolling window select none (§9.11.2).

---

## 1. Introduction

Long-running assistants and autonomous agents need context that grows without bound, and under full attention every token held grows the set of stored, compressed or rounded values every later step reads. More VRAM defers the threshold; it does not remove the coupling. RAG externalises memory and loses attentional continuity; longer native windows are a hardware investment against an architectural ceiling; summarization and eviction discard the verbatim detail persistent memory exists to keep.

The fix here is to decouple the tokens that take part in a step from the tokens in the context. §11.2 shows the decoupling bounds, at any depth, how far stored error moves each attention head's output against the weight a needed token keeps, and what a step is exposed to; a full-attention system cannot offer the same bounds, because its attended set is the whole context.

The system came from two constraints held together: persistent agent conversations, the hardest form of the memory problem (verbatim recall across unbounded sessions, many agents at once), and a single 16 GB consumer GPU. Each constraint closed the standard solution: GEMM dequantisation OOMed, so the matmul kernels read quantized weights natively; full context could not be held hot, so attention became provenance-selected; materialising turns at F16 OOMed, so quantization became two-phase (§11.5). The result, on that card and two larger ones:

- A **180B-parameter MoE model** (Qwen3.8-Flash-Next, 6B active) on a 16 GB laptop GPU with 32 GB of RAM, experts streamed VRAM → RAM → NVMe, decoding one session at 34.9 t/s against the 27.5–29 published for an RTX 5080 with 64 GB [32], and eight at 108.5 t/s aggregate — above every published llama.cpp run of it, each on a host with 64–128 GB, and above Strata's published 63.1 t/s across four sessions (§9.6.5).
- **Throughput flat in depth**: 96–99% of prefill and 93–111% of decode kept from 32K to 128K across two tasks on the RTX PRO 5000 (§9.6.2).
- **KV compression up to 7.6×**, each 32-token block choosing its format as it is sealed; at 8K the top level costs 2–8% of decode for 4.6–6.3× (§9.6.3).
- **Concurrency**: aggregate decode 1.4–37.9× the best published llama.cpp rate for the same model and card class, 2,869.9 t/s across 64 sessions on one card; a 284B model at sixteen sessions on one GPU, 4.1× the best published single-GPU decode (§9.6.4, §9.6.6).
- **Retrieval over its own codebase**: an answer file first of 7,943 on 49% of 121 questions and top-five on 69%; the memory worth 37.0 points of recall with tools forbidden, re-selection during reasoning 5.3 (§9.12).
- **Selection at depth**: every fact planted in a history twelve times the working set is selected, at C10 as at C0 (§9.11.2).

The agent use case is the existence proof that the parts run together; it is not the subject of the paper. To our knowledge no prior serving system states a bound on per-step error exposure at unbounded depth. An 80 GB card holds more tokens at full precision, but once a token is compressed or evicted it is in the same position as on a 16 GB card (§11.2, Corollary 2).

### 1.2 Technical Contributions

1. **Online Markov expert prediction** (§4) — a transition matrix learned from live routing, with no calibration, and a wave-batched grouped GEMM that amortises each expert's transfer across the sessions that route to it.
2. **Two-phase KV quantization** (§5) — K/V re-prefilled from the finished turn at its seal, removing decode drift, then a per-block format chosen independently for K and V by error metrics matched to K's softmax-amplified and V's linear error paths [AsymKV, COLING 2025].
3. **Attentional provenance indexing** (§6) — Q-vector sign signatures matched Q→Q by XNOR and popcount, scanned on the GPU over a VRAM-resident gallery by tensor-core kernels (~9.5 ms over a 591K-token gallery on a 16 GB laptop GPU; a median 83 ms over the 36.1M-token production gallery with sixteen sessions in flight) and re-selecting the working set every 64 decoded tokens, off the decode's critical path. The think block is the probe that decodes ahead of the answer.
4. **Unbounded three-tier context** (§7, §11.2) — VRAM-hot, RAM-warm and disk-cold paging, with the proof that the stored error entering a step is bounded independent of depth under provenance-selected attention.
5. **Native quantized kernels** (§8, supporting) — matmuls that never materialise a full-precision weight copy, greedy batch decomposition, and a fused sampling kernel.

---

## 2. Background and Related Work

### 2.1 MoE Expert Offloading

Fiddler [Kamahori et al., 2024] and HybriMoE [Zhong et al., 2025] partition experts between CPU and GPU by activation frequency; DAOP [DATE 2025] adds data-aware prefetching. All rely on offline profiling or trained predictors, which degrade on novel distributions. Our Markov predictor learns online from live routing, and the wave-batched kernel coalesces expert work across concurrent requests (§4.3, §9.7).

### 2.2 KV Cache Quantization

**KIVI** [Liu et al., ICML 2024] established channel-wise outliers in K and token-wise structure in V, quantizing both uniformly at 2 bits. **KVQuant** [Hooper et al., NeurIPS 2024] added per-vector outlier handling and first-token special-casing, under 0.1 perplexity at 3 bits. **TurboQuant** [Zandieh et al., ICLR 2026] rotates and Lloyd-Max quantizes, with its advantage below 3 bits; community benchmarks on Qwen3 find Q4_0 matching its residual implementation at 4–5×. **KITTY** [Xia et al., 2025] boosts 12.5–25% of K channels to 4 bits by an offline magnitude ranking and names adaptive selection as future work. Token eviction (H2O, TOVA, StreamingLLM) is lossy by design, DMS needs fine-tuning, SVD approaches (xKV, LoRC) are per-prompt, and KVTC [Staniszewski & Łańcucki, ICLR 2026] compresses ~20× offline for storage rather than live inference. OTT [Su et al., ACL 2025] finds outlier tokens cluster at block boundaries, independently motivating our 4/28 sub-block structure.

Every one of these optimises the compression *primitive*. None treats format assignment as a per-block decision with a quality guarantee, and none states a result on error scaling with depth. Here each (layer, head) block chooses its own K and V formats, by Q-projected error for K and normalised L2 for V, along an 11-point Pareto curve (C0–C10) that runs on dense, MoE and DeltaNet-hybrid models up to 180B with no per-model calibration data beyond a threshold scale (§9.9).

### 2.3 Autoregressive Error Accumulation

GEAR [Kang et al., 2024] showed quantization error in autoregressive decode compounds across steps. RefreshKV [Xu et al., ACL 2025] regenerates the KV periodically with full attention, as a standalone technique. Our refresh is turn-boundary-triggered and feeds the per-block format selection directly; RelayCaching [2026] confirms prefill and decode K/V agree closely, so prefill is the right ground truth.

### 2.4 Attention Sinks

StreamingLLM [Xiao et al., ICLR 2024] showed initial tokens carry disproportionate attention; KVSink [Su & Yuan, COLM 2025] extended this to mid-sequence sinks. Here, on attention-only models, turn-boundary sinks are recomputed in the current frame at every assembly rather than stored (§5). KVTuner [2025] and [llama.cpp #20969, 2026] find boundary layers more sensitive to K quantization on Qwen3.

### 2.5 KV-Based Retrieval

RetrievalAttention [Liu et al., NeurIPS 2025] measured the Q→K gap: Q vectors sit more than 10× farther from K than K from K, degrading cross-distribution ANNS. Provenance indexing stores Q signatures, so the dominant signal is a within-distribution Q→Q match.

---

## 3. System Architecture

One requirement organises the system: the tokens attending any step must be bounded independently of context depth, the property §11.2 uses. Figure 3-1 shows one decode step and the two loops that feed it. No component materialises an intermediate the next operation does not consume: there is no dequantised weight copy and no BF16 expansion between the logits and the sampling modifiers.

<figure class="fig">
<svg viewBox="0 0 640 292" role="img" aria-label="Figure 3-1. One decode step: embedding, attention over a working set, MoE experts streamed from VRAM, RAM and NVMe, a fused sampling kernel, and the next token. Every 64 decoded tokens the last 128 Q signatures are scanned against the VRAM gallery, in about 9.5 ms over 591 thousand tokens and a median 83 ms over the 36.1 million-token production gallery, off the critical path, and the result assembles the next forward's working set. When a turn ends it is re-prefilled and each 32-token block is sealed in its own format, C0 to C10, into three tiers: VRAM hot, RAM warm, disk cold, which also hold the gallery.">
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
  <text class="t-dim mid" x="332" y="178">last 128 Q signatures vs the VRAM gallery · off the critical path</text>
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

Candidates are staged into **free slots only**, so a misprediction is inert. Out of sample — 21-fold leave-one-out over a 21-prompt routing trace — PMI with a cross-prompt prior and the live session's matrix covers **4.7×** the cold experts a popularity baseline does at a fan-out of four (3.9× for plain PMI), with top-1 precision of 80–91% against 22% (§9.7). A session's own transitions predict its next layers better than any cross-prompt average.

### 4.3 Eviction Policy

Timestamp eviction cascades — a miss at L evicts an expert L+5 needs — and halved single-token decode. Four rules remove the cascade: evict the bottom 5% by timestamp at the end of each pass, leaving ~140 free slots; on a forced eviction, evict from completed layers first; pin layers 0 and 1, which have no compute ahead to hide a reload; prefetch into free slots only. Frequency-weighted recency cuts the miss rate from LRU's 22.9% to 14.7% at half the working set, where a learned recurrence model did worse than LRU (39.5%) (§9.7).

**Production: three tiers.** The production cache treats the on-disk expert pack as authoritative and VRAM and pinned RAM as independent copies of it, so eviction never writes back. Its key is access frequency weighted by the wrapped distance to the layer's next use — Bélády's ordering, exact because the layer wave is a cycle — evaluated by tier, so an expert whose miss costs an NVMe read is protected over one whose miss is a pinned copy (Figure 4-1). The same gate on the laptop now measures 6,139.1 t/s at twenty contexts with its experts streaming (§9.6).

<figure class="fig">
<svg viewBox="0 0 640 262" role="img" aria-label="Figure 4-1. Aggregate prefill on Qwen3-30B-A3B, RTX 4090 Laptop GPU, two-tier against three-tier expert cache. F16 one context 377.5 to 708.1 t/s, plus 88 percent. BF16 one context 509.4 to 780.6, plus 53 percent. BF16 ten contexts 1,931.4 to 2,726.1, plus 41 percent. Q8_0 twenty contexts 2,448.1 to 2,732.5, plus 12 percent. C9 two contexts 951.4 to 1,463.1, plus 54 percent.">
  <text class="ttl" x="16" y="20">Figure 4-1 · The three-tier expert cache, against two tiers</text>
  <text class="ttl-sub" x="16" y="36">Qwen3-30B-A3B · RTX 4090 Laptop 16 GB · aggregate prefill t/s · amber two-tier, green three-tier</text>
  <path class="grid" d="M335.7 54 V226 M471.4 54 V226"/>
  <path class="grid base" d="M200 54 V226"/>
  <text class="cat" x="16" y="72">F16 × 1 context</text>
  <text class="cat-sub" x="16" y="85">+88%</text>
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

- **Vote for the best several.** On the 6,854-file gallery of 2026-11-16, one vote per token gave any vote to only 20–67 files, and seven answer files scored exactly zero. Voting for the best eight, each vote square-rooted, raised the answer file's MRR from 0.508 to 0.713. Tool selection keeps one vote: a tool probe has one clean match.
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

**The opening.** A question's Q depends on the context it is read in, and every conversation opens on configured background files: one battery question's answer file ranked second of the 6,854 files then ingested alone, and fifth against them. So a request is read twice — first with the background held out, scored and discarded; then again, its opening working set ranked by the first reading beside the background. The first reasoning tokens then reproject at 8, 16 and 32 tokens, because what the model writes early about what it has read stays in its context. The cost is one extra prefill and one scan before the first token.

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
  <text class="t-dim" x="386" y="158">48.7% → 48.7%</text>
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

Re-uploading signatures every scan was the first GPU implementation's bottleneck, so they stay **resident in VRAM** in a paged gallery arena modelled on the KV cache — 32-token pages, a free-list pool, a pre-resolved pointer table the kernel reads in place. Its warm and cold tiers are the substrate's persisted signatures. Three scan backends are **bit-identical** under an adversarial parity test: a scalar reference; 1-bit tensor cores (`BMMA.88128.XOR.POPC`, sm_75–sm_89), ~9.5 ms over a 591K-token gallery on the RTX 4090 Laptop GPU; and INT8 tensor cores (`mma.m16n8k32.s8`), the production path on Blackwell, which dropped the 1-bit instruction. On BlackWorld's 36.1M-token production gallery, with sixteen sessions in flight, 1,277 reprojections scanned in a median 83 ms (p90 182 ms; `docs/results/battery_128/reproject_scans.md`), every 64 decoded tokens and off the decode's critical path.

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
Quantized weights:    ~18GB stored  (Q4_K_M)
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

No published engine combines unbounded context, per-block KV quantization, provenance indexing and predicted expert streaming, so no system runs the same workload. The evaluation instead (1) sets every throughput and compression row against the best published figure for the same model and card class — llama.cpp, vLLM, KTransformers, SGLang — saying where ours is aggregate and theirs single-stream (§9.6); (2) validates the output of every measured row, reporting it only if every session reproduces its reference (§9.8); and (3) ablates each contribution: the expert cache (§4.3, §9.7), selection at depth (§9.11.2), and seven memory arms over the system's own codebase that remove the memory, the seed files and reprojection in turn (§9.12).

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

The laptop is the design constraint; the RTX 3090 runs the integer-only path (no native FP8, neither tensor-core provenance backend) over a link no faster than the laptop's; the Blackwell card holds every model but the 284B resident and runs the depth sweeps. BlackWorld's driver moved from WDDM to TCC between its two sweeps, and §9.6 names the sweep a Blackwell row comes from; WDDM demotes VRAM past capacity and charges system commit for every allocation, TCC refuses and charges nothing (`docs/driver_model.md`). Every row is a test in this repository, run one model per `cargo` process with the card to itself; the commands and source rows are in `README.md` and `docs/performance.md`.

### 9.3 Models

Twelve models, spanning dense, mixture-of-experts, DeltaNet-hybrid and native-sparse latent attention, from 0.5B to 284B parameters. Each runs inside its checkpoint's own native context window, with no position scaling applied.

| Model | Size | Architecture | Weights |
|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 284B / 13B active | native-sparse latent attention, MoE | MXFP4_KO |
| Qwen3.8-Flash-Next | 180B / 6B active: a 125B trunk, a 51B n-gram table, a 4B MTP head | hybrid + QSA + MoE, MTP drafter | Q4_KOEXP (72 GB); Q2_KO experts (16 GB) |
| Qwen3.5-35B-A3B, Qwen3.6-35B-A3B | 35B / 3B active | DeltaNet hybrid + MoE | Q6_K |
| Qwen3-30B-A3B | 30B / 3B active | MoE, 48 layers × 128 experts, top-8 | Q4_K_M |
| Qwen3.8-27B | 27B dense | hybrid | Q6_K/Q8 |
| Qwen3.5-9B, Qwen3.5-0.8B | 9B, 0.8B dense | DeltaNet hybrid | Q6_K |
| Qwen3-8B, Llama-2-7B, Llama-3.2-3B, Qwen2-0.5B | 0.5–8B dense | full attention | Q6_K, Q4_0, Q4_K_M, Q4_0 |

§9.4–§9.5 and §9.10 were measured on Qwen3-30B-A3B at Q4_K_M, as in v1.

### 9.4 Quantized Matmul Kernel Benchmarks

The weight formats (**qmatmul**) and the KV formats share one block structure and dispatch; the KV-only formats (Q8_KS, Q4_KS, Q4_1, Q3_0, Q2_0) are validated by the per-block selector instead, since KV quality is attention-output fidelity, not reconstruction error. Figure 9.4-1 measures the weight kernels on the RTX 4090 Laptop GPU, 200 iterations a format under Criterion.rs, with no BF16 weight copy at any point. FP8 data formats ran at ~80.5 GiB/s, level with F16's 82.7, with no gain, and their non-uniform spacing suits block-normalised KV worse than integer formats, so FP8 left the KV ladder; E4M3 survives only as the `_KS` sub-block scale.

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

Every figure here is a test in this repository — the width ladder (`test_parallel_batched_forwarding*`, a ~700-token prompt at several widths and compression levels), the depth gate (`long_context_gate`, 4K–128K), or the engine probe (`kv_fragmentation`, the full conversation engine under load). Every width and engine row checks each session's rewrite against its reference (§9.8); the depth gate checks its output is well-formed, and quality at depth is measured on its own — Flash-Next's rewrite reproduced behind 128,897 tokens (§9.6.2) and C5/C10 against BF16 teacher-forced to 124K (§9.6.8). Sources are in `docs/performance.md` and `docs/results/`. **Prefill** is prompt tokens per second; **decode** is generated tokens per second, **summed across the batch** where a row has more than one context.

#### 9.6.1 Width: what concurrency buys on one card

RTX PRO 5000 Blackwell 72 GB: each cell is the model's best validated row on the card across its two full sweeps, with the format and width that produced it — build `21c36efbd` under WDDM (2026-10-10, fourteen width gates, 212 rows) and, marked †, `3e51cb8c0` under TCC (2026-11-10, `docs/results/sweep_rtx_pro_5000_72gb_2026-11-10.md`).

| Model | Best prefill t/s | Best decode t/s | Best compression |
|---|---:|---:|---:|
| Qwen2-0.5B | 139,750.4 (F16 ×60) | 6,586.7 (BF16 ×60) | — |
| Qwen3.5-0.8B | 84,235.6 (C10 ×10) | 10,805.3 (C8 ×256) | 4.38× (C10) |
| Llama-3.2-3B | 25,575.1 (C10 ×5) | 1,153.9 (C9 ×10) | 4.43× (C10) |
| Llama-2-7B | 11,676.0 (F16 ×8) | 1,675.0 (BF16 ×48)† | 3.56× (Q4_0) |
| Qwen3.5-9B | 12,348.5 (C9 ×5) | 1,607.7 (C8 ×20) | 5.57× (C10) |
| Qwen3-8B | 11,269.1 (F16 ×2) | 632.9 (C8 ×10)† | 5.82× (C10) |
| Qwen3-30B-A3B | 17,019.5 (C5 ×8) | 997.4 (BF16 ×10)† | 5.45× (C10) |
| Qwen3.5-35B-A3B | 19,092.3 (C8 ×5) | 2,869.9 (C10 ×64) | 7.03× (C10) |
| Qwen3.6-35B-A3B | 18,516.5 (C8 ×5) | 2,761.0 (C10 ×64) | 6.43× (C10) |
| Qwen3.6-35B hybrid, Precision | 18,819.9 (C8 ×5) | 2,829.5 (C10 ×64) | 6.47× (C10) |
| Qwen3.6-35B hybrid, Performance | 19,577.9 (C8 ×5) | 2,892.5 (C10 ×64) | 6.45× (C10) |
| Qwen3.8-Flash-Next | 7,024.1 (BF16 ×4) | 1,819.3 (BF16 ×16)† | 7.13× (C10) |
| DeepSeek-V4-Flash | 1,177.1 (BF16 ×16) | 114.4 (BF16 ×16) | — |

Prefill saturates early while aggregate decode keeps climbing with width: Qwen3.5-35B-A3B goes from 230.2 t/s at one session to **2,869.9 t/s across sixty-four** (12.5×) and Qwen3.6-35B-A3B from 476.6 to **2,761.0** (5.8×), at C10 with prefill flat. This is §4.4's wave batching measured: one expert load serves every session routed to it.

#### 9.6.2 Depth: throughput does not fall with context

<figure class="fig">
<svg viewBox="0 0 640 258" role="img" aria-label="Figure 9.6-1. Share of the 32K rate kept at 128K, one context, BF16, RTX PRO 5000. Qwen3.8-Flash-Next keeps 99 percent of prefill and 111 percent of decode. Qwen3.8-27B 28 and 58. Qwen3.5-9B 30 and 48. Qwen3.6-35B-A3B 26 and 49. Qwen3.5-35B-A3B 26 and 47. Qwen3.5-0.8B 25 and 86.">
  <text class="ttl" x="16" y="20">Figure 9.6-1 · What a 4× deeper context costs</text>
  <text class="ttl-sub" x="16" y="36">share of the 32K rate kept at 128K · faint prefill, solid decode · RTX PRO 5000, BF16, coherence task</text>
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

On Flash-Next the cost per token does not depend on history: the engine runs its full-attention layers through their trained sparse-attention indexer (QSA), which caps each token's attended set at 2,051 positions, where the hybrids read the whole prefix. The attention kernel costs the same at shallow depth on both (0.382 ms at 32K on Flash-Next, 0.394 ms at 8K on Qwen3.5-35B), then stays within 6% to 128K on Flash-Next and grows 3.71× for 4× depth on the 35B. Qwen3.5-0.8B keeps 86% of decode because its step is dominated by depth-independent work. Flash-Next's speculative head accepts exactly **4.85 tokens a step at every depth from 8K to 128K**, and the story is reproduced from behind 128,897 tokens of padding, decode on that rewrite task keeping 93% from 32K (57.8 → 53.9 t/s) and prefill 96%. The published llama.cpp run of the same model on an RTX 4090 24 GB keeps 64% of prefill and 65% of decode from 6K to 110K [31], so the engine adds no depth-dependent cost of its own. This is the principle §11.2 rests on, measured: a working set bounded independently of depth costs the same at any depth.

#### 9.6.3 Compression: the top level, inline, validated

Each 32-token block is compressed in the forward that produced it. C10 against BF16 at 8K, one context, Blackwell (`docs/performance.md` §3.4):

| Model | Prefill BF16 / C10 (t/s) | Decode BF16 / C10 (t/s) | C10 compression |
|---|---|---|---:|
| Qwen3-8B | 3,235.1 / 3,212.7 | 42.8 / 41.8 | 6.31× |
| Qwen3-30B-A3B | 3,444.7 / 3,367.6 | 51.2 / 47.2 | 5.94× |
| Llama-2-7B (4K) | 3,684.7 / 3,851.1 | 47.1 / 44.1 | 4.81× |
| Llama-3.2-3B | 6,195.9 / 5,936.8 | 69.9 / 66.1 | 4.62× |

At 8K the top level costs **2–8% of decode for 4.6–6.3×**. At 32K the largest hybrids pay 27–31% for 6.3–7.5×; at 128K the cache compresses 4.63–7.63× with every block quantized, Flash-Next paying 19% for 6.97× and the whole-prefix hybrids 35–51%, the cost of reading quantized K/V growing with the number of reads (`docs/performance.md` §3.5). C10 prefill runs within 5% of BF16 at every depth and is marginally *faster* at 128K, with fewer bytes to read. llama.cpp's q8_0 cache gives ~1.9× for 35% of decode at 32K and 45% at 64K, its q4_0 ~3.56× for 39% and 50% [35]; vLLM's FP8 gives 2× [36], TurboQuant 2.4–3.4× for 20–27% [37]. At 32K the top level holds **nearly 4× q8_0's compression while slowing decode less**.

#### 9.6.4 Against the published record

Figure 9.6-2 sets our aggregate decode, serving concurrent sessions, against the best published llama.cpp single-stream decode for the same model on the same class of card; the Blackwell rows compare our RTX PRO 5000, each at its best validated row of §9.6.1, with published RTX 5090 figures. Aggregate against single-stream is the comparison the llama.cpp record allows, and it answers how much work one card does. The published aggregates are larger engines on larger cards: vLLM serves Qwen3.6-35B-A3B at 449.0 t/s across five concurrent requests on a 96 GB RTX PRO 6000 [41], where this engine serves it at 2,761.0 across sixty-four on a 72 GB card, 6.1×; Strata serves Flash-Next at 63.1 t/s across four on a 12 GB card [49], where the 16 GB laptop serves eight at 108.5. Published single-card serving of the 35B models stops at 5–10 concurrent requests [40, 41]; this engine runs them at 64, and Qwen3.5-0.8B at 256.

<figure class="fig">
<svg viewBox="0 0 640 468" role="img" aria-label="Figure 9.6-2. Our aggregate decode against the best published llama.cpp single-stream decode, same model and card class, on a log scale. RTX 3090: Qwen3.5-35B-A3B 11.1 times, Qwen3.6-35B-A3B 8.5, Llama-2-7B 5.8, Qwen3-8B 3.3, Qwen3.8-27B 5.5, Qwen3-30B-A3B 2.6. 16 GB: Qwen3.6-35B-A3B 1.4 to 1.8, Qwen3-8B 3.0, Flash-Next 3.7. Blackwell: Qwen3.5-35B-A3B 14.8, Qwen3.6-35B-A3B 8.3, Flash-Next 37.9, Llama-2-7B 5.6, Qwen3-30B-A3B 4.4, Qwen3-8B 3.2.">
  <text class="ttl" x="16" y="20">Figure 9.6-2 · One card's aggregate decode, against the published record</text>
  <text class="ttl-sub" x="16" y="36">× the best published llama.cpp single-stream decode, same model and card · log scale</text>
  <path class="grid" d="M313.8 46 V440 M398.1 46 V440 M461.9 46 V440 M525.7 46 V440 M610 46 V440"/>
  <path class="parity" d="M250 46 V440"/>
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
  <text class="cat" x="16" y="218">Qwen3.6-35B-A3B, C10 ×16</text><text class="cat-sub" x="16" y="230">16 GB · 336.7 vs 183.3 / 249.3 MTP [38]</text>
  <path class="line-us" d="M250 220 H304.1"/><circle class="dot-us" cx="274.1" cy="220" r="4"/><circle class="dot-us" cx="304.1" cy="220" r="5"/><text class="v-us" x="314.1" y="225">1.4–1.8×</text>
  <text class="cat" x="16" y="244">Qwen3-8B, C8 ×10</text><text class="cat-sub" x="16" y="256">16 GB · 308.4 vs 102.7, RTX 4080 [29]</text>
  <path class="line-us" d="M250 246 H351.1"/><circle class="dot-us" cx="351.1" cy="246" r="5"/><text class="v-us" x="361.1" y="251">3.0×</text>
  <text class="cat" x="16" y="270">Flash-Next, BF16 ×8</text><text class="cat-sub" x="16" y="282">16 GB · 108.5 vs 27.5–29, RTX 5080 [32]</text>
  <path class="line-us" d="M250 272 H370.4"/><circle class="dot-us" cx="370.4" cy="272" r="5"/><text class="v-us" x="380.4" y="277">3.7×</text>
  <text class="cat" x="16" y="296">Qwen3.5-35B-A3B, C10 ×64</text><text class="cat-sub" x="16" y="308">Blackwell · 2,869.9 vs 194.0 [34]</text>
  <path class="line-us" d="M250 298 H498"/><circle class="dot-us" cx="498" cy="298" r="5"/><text class="v-us" x="508" y="303">14.8×</text>
  <text class="cat" x="16" y="322">Qwen3.6-35B-A3B, C10 ×64</text><text class="cat-sub" x="16" y="334">Blackwell · 2,761.0 vs 333.6 MTP [38]</text>
  <path class="line-us" d="M250 324 H444.8"/><circle class="dot-us" cx="444.8" cy="324" r="5"/><text class="v-us" x="454.8" y="329">8.3×</text>
  <text class="cat" x="16" y="348">Flash-Next, BF16 ×16, TCC</text><text class="cat-sub" x="16" y="360">Blackwell · 1,819.3 vs 48.02 [42]</text>
  <path class="line-us" d="M250 350 H584.5"/><circle class="dot-us" cx="584.5" cy="350" r="5"/><text class="v-us" x="574.5" y="344" text-anchor="end">37.9×</text>
  <text class="cat" x="16" y="374">Llama-2-7B, BF16 ×48</text><text class="cat-sub" x="16" y="386">Blackwell · 1,675.0 vs 300.40 [27]</text>
  <path class="line-us" d="M250 376 H408.5"/><circle class="dot-us" cx="408.5" cy="376" r="5"/><text class="v-us" x="418.5" y="381">5.6×</text>
  <text class="cat" x="16" y="400">Qwen3-30B-A3B, BF16 ×10</text><text class="cat-sub" x="16" y="412">Blackwell · 997.4 vs 226.1 [39]</text>
  <path class="line-us" d="M250 402 H386.4"/><circle class="dot-us" cx="386.4" cy="402" r="5"/><text class="v-us" x="396.4" y="407">4.4×</text>
  <text class="cat" x="16" y="426">Qwen3-8B, C8 ×10</text><text class="cat-sub" x="16" y="438">Blackwell · 632.9 vs 200.4 [39]</text>
  <path class="line-us" d="M250 428 H357"/><circle class="dot-us" cx="357" cy="428" r="5"/><text class="v-us" x="367" y="433">3.2×</text>
  <text class="tick mid" x="250" y="458">1×</text>
  <text class="tick mid" x="313.8" y="458">2×</text>
  <text class="tick mid" x="398.1" y="458">5×</text>
  <text class="tick mid" x="461.9" y="458">10×</text>
  <text class="tick mid" x="525.7" y="458">20×</text>
  <text class="tick mid" x="610" y="458">50×</text>
</svg>
</figure>

The 16 GB Qwen3.6-35B row sets our streamed Q6_K experts against a 3-bit quant that fits wholly in VRAM, which decodes one stream faster: 1.4–1.8×. Every Flash-Next figure here decodes with the model's own MTP head, part of its 180B; [42]'s llama.cpp build had not implemented it, and the four n-gram speculation modes it tried instead were each slower than plain decoding, while [32]'s RTX 5080 drafts from the model's n-gram table.

#### 9.6.5 The 16GB laptop

The RTX 4090 Laptop GPU with 32 GB of RAM is the constraint the architecture was designed under.

- **A 180B model.** Qwen3.8-Flash-Next (a 125B trunk of 512 experts, a 51B n-gram table and a 4B MTP head; 6B active) runs from a ~88 GB Q2_KO-expert artifact, streaming VRAM → pinned RAM → NVMe, every rung validating 8/8. One session prefills at 269.1 t/s and decodes at 34.9; eight decode at **108.5 t/s aggregate**, and C10 at 5.79–5.82× costs 6.6% of that. Every published single-GPU run that states its host uses 64–128 GB of RAM: an RTX 5090 with 128 GB decodes one stream at 48.02 t/s [42], an RTX 5080 with 64 GB at 27.5–29 [32], the latter drafting from the n-gram table as ours drafts from the MTP head (§9.6.4).
- **A 30B MoE streaming its experts** prefills at **6,139.1 t/s** (Qwen3-30B-A3B, Q8_0 ×20).
- **A 35B MoE serving sixteen.** Qwen3.5- and Qwen3.6-35B-A3B at Q6_K, ~28 GB of weights, serve ×16 at **330.8 and 336.7 t/s aggregate**, C10 at 7.01× and 6.42×. Published 16 GB runs are single-stream at quants that fit resident [38].
- **A small model at workstation rate.** Qwen3.5-0.8B at C8 ×32 prefills at **46,056.8 t/s** and decodes at **3,810.7 t/s aggregate**, above the RTX 3090's 42,982.2 on the same rung.

C10 compresses the 35B to 7.01× on the laptop and 7.03× on the RTX 3090: the same selector and thresholds make the same decisions on different GPUs and links.

#### 9.6.6 A 284B model on one GPU

DeepSeek-V4-Flash (284B total, 13B active) on one RTX PRO 5000 72GB, sixteen concurrent sessions: **1,177.1 t/s prefill and 114.4 t/s aggregate decode**, build `21c36efbd`. The best published single-GPU prefill for the model is 748.4 t/s (RTX PRO 6000 Max-Q, all experts on the CPU, 8K) [43], and the best published single-GPU decode is 28 t/s (RTX 5090, KTransformers + SGLang, INT4 experts on the CPU) [44] — every published run holding the routed experts in host RAM, as this one does. The aggregate is 4.1× that decode, and the prefill is above every published single-GPU figure for the model.
#### 9.6.7 Where published engines lead

Single-stream decode on a model that fits wholly in VRAM is llama.cpp's strength: a 3-bit Qwen3.6-35B-A3B resident on a 16 GB card decodes one stream at 183.29 t/s (249.33 with MTP) [38], where our sixteen-session aggregate with Q6_K experts streamed is 336.7 — more work in total, less per stream. vLLM prefills Qwen3.6-35B-A3B at 41,105 t/s on an RTX PRO 6000 [41], against 10,582.4 here at one session. Against Strata's RTX 5090 single-session benchmark [49], replayed through the conversation engine at C5 (`kv_fragmentation::qwen38_flash_next_strata`, median of three), ours leads prefill at 4K — 5,794.4 t/s against 4,269.8 — and trails decode: 168.4 / 159.1 / 125.3 t/s against 179.4 / 175.7 / 165.0 at 4K / 32K / 128K, 94%, 91% and 76% of Strata. Strata has a PCIe 5.0 card and a dedicated CUDA graph path; here one segment submission per MoE layer is the floor.

#### 9.6.8 Output fidelity against depth

Throughput says compression is cheap, not how far it moves the output or whether that grows with what a step reads. At each depth a needle prompt — four random codes spread through filler, asked back at the end — is prefilled, sealed at the mode's level, and the answer read back teacher-forced against BF16 over the same prefix (`batch_test::fidelity`; `docs/results/theorem_depth_tests_2026-11-10.md`).

<figure class="fig">
<svg viewBox="0 0 640 246" role="img" aria-label="Figure 9.6-3. Mean KL divergence from BF16, in nats, against context depth, RTX PRO 5000. Flash-Next at C10: 5.9e-2 at 4K, 3.0e-2 at 16K, 1.08e-1 at 64K, 1.16e-1 at 124K. Flash-Next at C5: 9.8e-3, 1.1e-2, 2.1e-2, 7.6e-3. Qwen3.5-9B at C10: 6.5e-3, 1.2e-3, 1.7e-3, 3.5e-3. Qwen3.5-9B at C5: 6.6e-3, 1.0e-3, 9.4e-4, 3.6e-3. Qwen3-8B at C10 to 32K: 3.2e-2, 1.1e-2, 3.9e-2.">
  <text class="ttl" x="16" y="20">Figure 9.6-3 · Divergence from BF16 against depth</text>
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
  <text class="t-them" x="404" y="110">8B C10, to 32K</text>
  <text class="tick mid" x="220" y="232">4K</text>
  <text class="tick mid" x="340" y="232">16K</text>
  <text class="tick mid" x="400" y="232">32K</text>
  <text class="tick mid" x="460" y="232">64K</text>
  <text class="tick mid" x="517.2" y="232">124K</text>
</svg>
</figure>

Top-1 agreement with BF16 is 95.8–100% in every cell, with run-to-run spread 0 on the 8B and 9B and at most 4.3e-5 on Flash-Next. Where attention reads the whole prefix, divergence does not grow across a 30× range of depth: the exposure §11.2 bounds grows, the output error does not (§11.1). On Flash-Next at C5, the level production seals at, it stays flat from 4K to 124K; C10 is the one series that grows, about 2×, the price of its extra compression on a model whose attention selects.

#### 9.6.9 Memory: what keeps a large context on one card

The native kernels keep weights at their stored size (§8.1) and per-block quantization keeps every sealed block at the format its own error allows (§5); both leave the card to the context. On Flash-Next the batched forward gate (`docs/performance.md` §3.6, RTX PRO 5000, the 2026-11-10 sweep, every row validated) reads, per level:

| Level | Compression | Prefill ×2 (t/s) | Decode ×2 (t/s) |
|---|---:|---:|---:|
| C0 | 2.22× | 5,033.6 | 468.9 |
| C5 | 4.15× | 5,903.2 | 466.8 |
| C8 | 5.52× | 5,942.2 | 436.2 |
| C10 | 7.14× | 5,936.9 | 429.8 |

From C0 to C10 the cache shrinks 3.2× further for 8% of decode; C5, the level everything is sealed at, holds 4.15× the tokens of BF16 at C0's throughput.

**The substrate.** The dialogue and every ingest layer seal at C5. On 2026-11-23 BlackWorld's redo log held 103 segments, 409 GiB, ≈ 8.0 KB on disk per resident token with every record included; the galleries held 54.7M signature tokens over 29,747 turns and 7,830 code files. The K/V census (`GET /v1/kv/census`, `docs/results/battery_128/census_start_5470.json`) — exact chunk bytes from the chunk index, the sub-band format mix from a sample of 4,096 chunk heads (4,157 chunks) — finds 16,006,522 chunks holding 504.9M layer-tokens, 42.1M tokens over Flash-Next's 12 K/V layers, in 275.3 GiB of K/V standing for 976.7 GiB at BF16: **3.55×**. By lead format 68.7% of chunks are Q4_0, 18.8% Q4_1, 10.5% Q5_0 and 1.3% Q8_0; 46% of the V sub-bands are Q1_S, Q2_A or Q2_S, while no K sub-band falls below Q4_0.

**The span.** The daemon reserves one contiguous span per card and divides it between persistent tensors, KV regions, a per-wave transient tier and expert weights, with an elastic boundary between the KV side and the experts (§10.2; Figure 9.6-4). Behind it the warm tier holds 14.4 GiB of payload in host RAM.

<figure class="fig">
<svg viewBox="0 0 640 128" role="img" aria-label="Figure 9.6-4. BlackWorld's 68.1 GiB span at rest with Flash-Next loaded, in span order: dense weights 5.3 GiB, recurrent state 4.9, 6.9 GiB for the KV side, the transient tier and held index pages, then the expert zone, 51.0, across the elastic boundary.">
  <text class="ttl" x="16" y="20">Figure 9.6-4 · The span at rest</text>
  <text class="ttl-sub" x="16" y="36">BlackWorld, 72 GB card, Flash-Next · 68.1 GiB · the boundary between KV and experts moves under load</text>
  <rect class="b1" x="16" y="50" width="45.7" height="32"/>
  <rect class="b2" x="62.7" y="50" width="42.2" height="32"/>
  <rect class="q2" x="105.9" y="50" width="60.3" height="32"/>
  <rect class="b0" x="167.2" y="50" width="448.8" height="32"/>
  <path class="parity" d="M166.7 44 V88"/>
  <text class="t-lbl" x="176" y="71">Expert zone · 51.0 GiB</text>
  <text class="tick mid" x="38.9" y="100">dense 5.3</text>
  <text class="tick mid" x="83.8" y="114">state 4.9</text>
  <text class="tick mid" x="136" y="100">KV side 6.9</text>
  <text class="t-them" x="172" y="126">← elastic boundary</text>
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
  <rect class="hatch" x="184" y="72" width="34" height="208"/>
  <text class="t-dim mid" x="201" y="276">new build</text>
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
  <text class="ttl-sub" x="16" y="38">GiB, 96 bucket means · resident tokens at BF16, 24,916 B each, against the KV arenas</text>
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

The three-tier cache's effect is Figure 4-1 (+12% to +88% prefill). A cleaner ablation came from replacing the MoE dispatch with the live, device-side one: the dense models in the same sweep, which do not touch it, moved −1.2% to +0.5% on prefill and −1.6% to +0.2% on decode — a control reading as a control should — while Qwen3.5-35B-A3B gained 24–36% prefill and 11–23% decode, Qwen3.6-35B-A3B likewise, and Qwen3-30B-A3B's single-session decode went from 45.8 to 54.9 t/s (+20%) on the same card and artifact (`docs/results/moe_live_dispatch_rtx_3090_24gb_2026-10-04.md`).

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

Up to 256 sessions run in one batch, each assigned a distinct character (name and gender, period 99) and asked to rewrite a passage with it. A session passes if its name appears, its pronouns match, and no other session's name appears — stressing the session-discriminating signal aggressive K quantization degrades first. Every throughput row of §9.6 passes with every session correct: 213 rows on the RTX 3090, 192 on the laptop, 212 on Blackwell. C10 passes on every model that compresses at 4.1–7.6×, just inside the edge where the test fails; two exceptions are on record and reported as failures. Flash-Next's C10 ×8 under the TCC build validated 7 of 8 sessions; the WDDM build's 885.8 t/s is the validated figure for that rung. Qwen3.6-35B-A3B's C10 at ×32 and ×64 validated 31 of 32 and 61 of 64 sessions on one sweep and every session on the next. Past the edge, Llama-3.2-3B fails in three regimes — clean, soft drift, then identity collapse: small errors move the text, large ones move which session the model attends.

### 9.9 Cross-Architecture Transfer: Held-Out Model Validation

The selector was derived from two reference models, Qwen3-30B-A3B (MoE) and Llama-3.2-3B (dense). Deployed unchanged on held-out Qwen3-8B it compressed *more* than either — 7.42× in v1 against 7.04× and 5.02×, and **6.31×** against 5.94× and 4.62× on the current 8K gates (§9.6.3) — the opposite of what overfitting would produce. It now runs on ten of the fleet's twelve models, dense, MoE and DeltaNet-hybrid, 0.8B to 180B, reaching 4.1–7.6× at the top level on each, with no calibration data (Qwen2-0.5B's ladder has no compressed rung, and DeepSeek-V4-Flash runs its latent K/V at BF16): a model carries at most a scale on its thresholds (`*_KV_FACTORS`), set by stepping C10 to the story gate's edge. Unlike GPTQ, AWQ or KVQuant's sensitivity profiles, the kernel responds to each block's distribution at inference time.

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
- **Code:** an answer file first of 7,943 on 49% of the battery's 121 ranked questions, top five on 69% (§9.12).
- **Normalization:** R@1 over 37,284 calibration questions 0.107 → 0.239, no file taking more than 0.1% of the wins on other files' questions (§6.4).
- **Momentum:** R@10 0.554 → 0.708 over three turns on a file, at the cost of slower topic switches — a new file leads in 70% of dialogues after three turns (`docs/results/provenance_gates_2026-10-26.md`).
- **A real conversation:** replayed on the paper conversation itself, the scorer finds 91% of the files it should place and puts all 14 of its tool calls' tools in the top ten (`docs/results/provenance_gates_2026-10-25.md`).

#### 9.11.2 Selection past the working set

Is the bounded working set delivered when the history is many times larger than it (`candle-conversation/tests/provenance_depth.rs`; `docs/results/theorem_depth_tests_2026-11-10.md`)? Flash-Next runs with its slot capped at 32,768 tokens. A conversation is built from the repository's source, one file a turn, to 30K, 131K and 393K tokens, with four turns stating vault codes together at 45% of its depth, where neither an oldest-first nor a newest-first fill reaches; the last turn asks for the codes. Each cell is a fresh conversation:

| Needle turns selected · in the final working set · codes recalled | 30K (19 turns) | 131K (64 turns) | 393K (195 turns) |
|---|---|---|---|
| Provenance with reprojection, C10 | 4 · 4 · **4/4** | 4 · 4 · 0/4 | 4 · 3 · 0/4 |
| Provenance with reprojection, C0 | 4 · 4 · **4/4** | 4 · 4 · 1/4 | 4 · 2 · 0/4 |
| One-shot projection, C10 | 3 · 3 · 3/4 | 0 · 0 · 0/4 | 0 · 0 · 0/4 |
| Rolling window (no provenance) | 3 · 3 · 3/4 | 0 · 0 · 0/4 | 0 · 0 · 0/4 |

**Selection holds at depth.** Mid-decode reprojection brought every needle turn into the working set at 131K and 393K — from 64 and 195 turns into 15–18, a history twelve times the slot — at C10 and C0 alike, while the arms without it selected none past 30K, both filling from the oldest turns. Compression does not change what is selected.

**Reading a selected turn needs its sealing context.** Past 30K the model holds the needle turns and still answers that no codes were given. The same test's diagnosis isolates why, by elimination: QSA reads the needle pieces at rank 0 on 77–85% of queries; dense attention, fresh placement and moving the needles beside the question all fail the same way; and the failing projection's own 17 turns, re-prefilled into a fresh slot at the same positions and the same C10, recall every code they hold. What differs is the context the K/V was computed in. Moving the needles through the history shows a cliff — 4/4 with up to 18K tokens of conversation behind them at their seal, 1/4 at 31K, 0/4 from 44K — and a 2×2 of sealing against reading context recalls 4/4 exactly where the two match, in either direction: a sealed turn carries its context in its residual stream, and its K/V is read as what it was in that context and not in another. This is the cross-context reuse problem of position-independent KV caching: CacheBlend [50] and EPIC [51] repair a reused chunk by recomputing a few of its tokens, and Block-Attention [52] and APE [53] by training or rescaling the model to read blocks encoded apart; production's arrangement, each file sealed behind the system prompt alone, is Block-Attention's independent block. How far a model reads K/V sealed in another context is a property of the model, and it has grown with every model this system has been upgraded to.

**Production memory is built so the contexts match.** Every ingested file is its own conversation, sealed behind the system prompt alone and never parented on the priming chain (`zend/src/code_read/process.rs`); only the chain's own links are read behind it, and every dialogue opens on that chain. Measured directly, four turns sealed behind the 32K chain recall 4/4 when read with the chain ahead and 0/4 without it; sealed alone and read behind the chain — production's arrangement — 4/4 (`recall_with_and_without_the_priming_prefix`). The 128-question battery measures that arrangement end to end: an answer decoded with its answer file placed recalls **51.4%**, without it 13.7% (§9.12, Figure 9.12-6). The needle test's failure is the one arrangement production does not use — turns sealed 60–100K tokens deep inside one long conversation and read out of it.

### 9.12 Unbounded Context: Codebase Dependency Analysis

A harder property than recall: multi-hop reasoning over the system's own Candle fork (2.2M lines of Rust and CUDA), each decode step's Q vectors retrieving the next node of a dependency chain. The production code-reading ingest has the daemon's model, Flash-Next, read the repository file by file as turns of its own, sealing their K/V and signatures with calibration questions beside each file; nothing in it was written by an external model. It persists as the substrate does (§7), valid for the model that wrote it.

**The claim.** *Provenance projection supplies the memory the answers come from, and re-selecting the working set while the model reasons brings in the files a question does not name.* The battery isolates the memory, the seed files a conversation opens on, reprojection during reasoning, and continuous reprojection against a single retrieval, holding all else equal, and measures the memory against retrieval-augmented generation over the same repository.

**Questions.** 128 (`zend/examples/dependency_battery/battery.yaml`): 43 direct, 43 transitive, 42 architectural, each listing the items a correct answer must name, its answer files, and whether it names its answer's file — all from the source and committed before the first run with the analysis; no question spells an item it expects. The battery and its results are never ingested: `zend.ignore` (`motherd/motherd.yaml`) has kept them out of the memory since 2026-11-16, a week before the first run, as they are kept out of R1's index. The direct questions are generated mechanically (`direct.rs`: a public function defined once and called from one to three other non-test files, picked by a seeded hash), so their selection carries no judgement; the rest are the 24-question pilot's 16 transitive and architectural questions and 69 more, written by hand against the lines they cite. *Direct:* where `tier_fits` is defined and what calls it. *Transitive:* how MoE routing reaches the expert kernels with no host readback — the bucketize kernel, the mapped ring, the stager's word, the cold expert's wait. *Architectural:* the elastic boundary between KV and experts (`weight_floor`) and the check that refuses a tier crossing it.

**Arms.** Three request-level dials: `projection` *Full* or *Minimal* (the tool catalog alone, and so nothing with tools off); `seed` on or off; and reprojection through the think block, held in it, or at the opening only:

| Arm | Seed | Projection | Reprojection | What it isolates |
|---|---|---|---|---|
| A0 | off | Minimal | — | the model alone |
| A1 | on | Minimal | — | the seed files alone |
| D1 | on | Full | at the opening; none in thinking or at `</think>` | retrieve once, before reasoning |
| B0 | off | Full | held in thinking | — |
| B1 | on | Full | held in thinking | reasoning-time reprojection removed |
| C0 | off | Full | full | — |
| C1 | on | Full | full | the production system |
| R1 | off | Minimal | — | retrieval-augmented generation: BM25's top files as text |
| P1 | off | Minimal | — | the memory's own selection, as text under R1's budget |
| T1 | off | Minimal | — | reading the repository with tools, for reference |

The two baselines project no memory. **R1** ranks every text file of the repository by BM25 over identifier terms of path and body (`zend/examples/dependency_battery/rag/`), the battery and its results left out of the index, and packs the best-ranked files, whole, into the prompt ahead of the question until 24,000 tokens, cutting the file that reaches the budget; its prompt asks for an answer from those files. **P1** asks C1 the question first, takes the files of the reading turns its governing projection point placed, in that point's own order, and packs them whole under R1's budget and prompt, so it differs from R1 in the selection alone and from C1 in reading text instead of K/V. **T1** may call the workspace's read and search tools and answers once it has read what it needs.

**Running.** On BlackWorld through the daemon's chat API (`dependency_battery run`), the first 748 answers at build 5470-6eb469ce5 and the rest at 5475-3c73da839, which changes no retrieval dial. One fresh conversation per question, its arm's dials on the request, deleted once its answer and projection points (`GET /v1/conversations/{id}/placements`) are read; every memory arm's question asks for an answer from memory with no tool. The first 142 answers ran sixteen at once and the rest eight, and the order mixes the arms throughout, so drift falls on every arm alike. Each arm answers the 128 questions three times: 2,688 answers. The runs added after — R1 and T1 three times each, B1 runs 4–6, C1 runs 4–7, R1 run 4 — ran sixteen at once, interleaved across the arms still owed, on the builds deployed while they ran; none changed retrieval code (no commit to `candle-conversation/src/projection`, `provenance` or `scheduler/reproject*` in the period). P1 ran once, sixteen at once, at build 5564-10dd9ff2b, which changes no retrieval code either, started only once the member reported its memory settled — every file ingested, calibrated and normalized. 4,608 answers in all.

**Scoring.** Recall is the share of a question's expected items the answer names, any spelling, case-insensitive; a question's score in an arm is its mean over the arm's runs. Intervals are 95% bootstrap intervals over questions (10,000 resamples, paired for differences); contrasts are Wilcoxon signed-rank on per-question differences, Holm-corrected across the five pre-registered, and the four against the baselines as a family of their own. Every figure's data and `summary.md` are in `docs/results/battery_128/`.

<figure class="fig">
<svg viewBox="0 0 640 306" role="img" aria-label="Figure 9.12-1. Mean recall by arm over 128 questions, each with its 95% bootstrap interval: A0 0.4% (0.1% to 0.9%); A1 0.5% (0.1% to 1.0%); D1 30.6% (25.0% to 36.4%); B0 31.9% (26.1% to 37.7%); B1 32.2% (26.8% to 37.6%); C0 36.7% (30.8% to 42.6%); C1 37.5% (31.7% to 43.1%); R1 52.6% (46.1% to 58.9%); P1 44.0% (37.4% to 50.9%); T1 95.3% (93.3% to 97.1%). Of the memory arms, the reprojecting C0 and C1 recall the most; R1 is handed BM25's files as text, P1 the memory's own selection as text, and T1 reads the repository with tools.">
  <text class="ttl" x="16" y="20">Figure 9.12-1 · Recall by arm</text>
  <text class="ttl-sub" x="16" y="38">mean recall over 128 questions · bar: 95% bootstrap interval · small dots: each run's mean</text>
  <path class="grid" d="M70 231.2 H624 M70 212.4 H624 M70 193.6 H624 M70 174.8 H624 M70 156 H624 M70 137.2 H624 M70 118.4 H624 M70 99.6 H624 M70 80.8 H624 M70 62 H624"/>
  <text class="tick" x="62" y="253.5" text-anchor="end">0%</text>
  <text class="tick" x="62" y="234.7" text-anchor="end">10%</text>
  <text class="tick" x="62" y="215.9" text-anchor="end">20%</text>
  <text class="tick" x="62" y="197.1" text-anchor="end">30%</text>
  <text class="tick" x="62" y="178.3" text-anchor="end">40%</text>
  <text class="tick" x="62" y="159.5" text-anchor="end">50%</text>
  <text class="tick" x="62" y="140.7" text-anchor="end">60%</text>
  <text class="tick" x="62" y="121.9" text-anchor="end">70%</text>
  <text class="tick" x="62" y="103.1" text-anchor="end">80%</text>
  <text class="tick" x="62" y="84.3" text-anchor="end">90%</text>
  <text class="tick" x="62" y="65.5" text-anchor="end">100%</text>
  <path class="grid base" d="M70 250 H624"/>
  <rect class="them" opacity="0.35" x="90.7" y="248.4" width="14" height="1.5"/>
  <circle class="q1" cx="113.7" cy="249.1" r="2.5"/>
  <circle class="q1" cx="113.7" cy="249.2" r="2.5"/>
  <circle class="q1" cx="113.7" cy="249.4" r="2.5"/>
  <circle class="dot-them" cx="97.7" cy="249.3" r="6"/>
  <text class="v-them mid" x="97.7" y="240.4">0.4%</text>
  <text class="cat mid" x="97.7" y="268">A0</text>
  <text class="cat-sub mid" x="97.7" y="282">none</text>
  <text class="cat-sub mid" x="97.7" y="295">no seed</text>
  <rect class="them" opacity="0.35" x="146.1" y="248.1" width="14" height="1.7"/>
  <circle class="q1" cx="169.1" cy="248.8" r="2.5"/>
  <circle class="q1" cx="169.1" cy="249.2" r="2.5"/>
  <circle class="q1" cx="169.1" cy="249.3" r="2.5"/>
  <circle class="dot-them" cx="153.1" cy="249.1" r="6"/>
  <text class="v-them mid" x="153.1" y="240.1">0.5%</text>
  <text class="cat mid" x="153.1" y="268">A1</text>
  <text class="cat-sub mid" x="153.1" y="282">none</text>
  <text class="cat-sub mid" x="153.1" y="295">seed</text>
  <rect class="them" opacity="0.35" x="201.5" y="181.5" width="14" height="21.5"/>
  <circle class="q1" cx="224.5" cy="192.1" r="2.5"/>
  <circle class="q1" cx="224.5" cy="191.1" r="2.5"/>
  <circle class="q1" cx="224.5" cy="194" r="2.5"/>
  <circle class="dot-them" cx="208.5" cy="192.4" r="6"/>
  <text class="v-them mid" x="208.5" y="173.5">30.6%</text>
  <text class="cat mid" x="208.5" y="268">D1</text>
  <text class="cat-sub mid" x="208.5" y="282">once</text>
  <text class="cat-sub mid" x="208.5" y="295">seed</text>
  <rect class="them" opacity="0.35" x="256.9" y="179.1" width="14" height="21.8"/>
  <circle class="q1" cx="279.9" cy="190.4" r="2.5"/>
  <circle class="q1" cx="279.9" cy="192.9" r="2.5"/>
  <circle class="q1" cx="279.9" cy="186.9" r="2.5"/>
  <circle class="dot-them" cx="263.9" cy="190.1" r="6"/>
  <text class="v-them mid" x="263.9" y="171.1">31.9%</text>
  <text class="cat mid" x="263.9" y="268">B0</text>
  <text class="cat-sub mid" x="263.9" y="282">held</text>
  <text class="cat-sub mid" x="263.9" y="295">no seed</text>
  <rect class="them" opacity="0.35" x="312.3" y="179.2" width="14" height="20.4"/>
  <circle class="q1" cx="335.3" cy="187.2" r="2.5"/>
  <circle class="q1" cx="335.3" cy="188.7" r="2.5"/>
  <circle class="q1" cx="335.3" cy="190.8" r="2.5"/>
  <circle class="q1" cx="335.3" cy="189.7" r="2.5"/>
  <circle class="q1" cx="335.3" cy="190.3" r="2.5"/>
  <circle class="q1" cx="335.3" cy="190.4" r="2.5"/>
  <circle class="dot-them" cx="319.3" cy="189.5" r="6"/>
  <text class="v-them mid" x="319.3" y="171.2">32.2%</text>
  <text class="cat mid" x="319.3" y="268">B1</text>
  <text class="cat-sub mid" x="319.3" y="282">held</text>
  <text class="cat-sub mid" x="319.3" y="295">seed</text>
  <rect class="us-soft" x="367.7" y="169.9" width="14" height="22.1"/>
  <circle class="q1" cx="390.7" cy="180.7" r="2.5"/>
  <circle class="q1" cx="390.7" cy="179.2" r="2.5"/>
  <circle class="q1" cx="390.7" cy="182.9" r="2.5"/>
  <circle class="dot-us" cx="374.7" cy="180.9" r="6"/>
  <text class="v-us mid" x="374.7" y="161.9">36.7%</text>
  <text class="cat mid" x="374.7" y="268">C0</text>
  <text class="cat-sub mid" x="374.7" y="282">full</text>
  <text class="cat-sub mid" x="374.7" y="295">no seed</text>
  <rect class="us-soft" x="423.1" y="169" width="14" height="21.3"/>
  <circle class="q1" cx="446.1" cy="181.3" r="2.5"/>
  <circle class="q1" cx="446.1" cy="176.1" r="2.5"/>
  <circle class="q1" cx="446.1" cy="183" r="2.5"/>
  <circle class="q1" cx="446.1" cy="176.9" r="2.5"/>
  <circle class="q1" cx="446.1" cy="180.5" r="2.5"/>
  <circle class="q1" cx="446.1" cy="178.1" r="2.5"/>
  <circle class="q1" cx="446.1" cy="180.7" r="2.5"/>
  <circle class="dot-us" cx="430.1" cy="179.5" r="6"/>
  <text class="v-us mid" x="430.1" y="161">37.5%</text>
  <text class="cat mid" x="430.1" y="268">C1</text>
  <text class="cat-sub mid" x="430.1" y="282">full</text>
  <text class="cat-sub mid" x="430.1" y="295">seed</text>
  <rect class="them" opacity="0.35" x="478.5" y="139.3" width="14" height="24"/>
  <circle class="q1" cx="501.5" cy="148" r="2.5"/>
  <circle class="q1" cx="501.5" cy="153" r="2.5"/>
  <circle class="q1" cx="501.5" cy="151.8" r="2.5"/>
  <circle class="q1" cx="501.5" cy="151.7" r="2.5"/>
  <circle class="dot-them" cx="485.5" cy="151.1" r="6"/>
  <text class="v-them mid" x="485.5" y="131.3">52.6%</text>
  <text class="cat mid" x="485.5" y="268">R1</text>
  <text class="cat-sub mid" x="485.5" y="282">BM25</text>
  <text class="cat-sub mid" x="485.5" y="295">no seed</text>
  <rect class="them" opacity="0.35" x="533.9" y="154.4" width="14" height="25.4"/>
  <circle class="q1" cx="556.9" cy="167.2" r="2.5"/>
  <circle class="dot-them" cx="540.9" cy="167.2" r="6"/>
  <text class="v-them mid" x="540.9" y="146.4">44.0%</text>
  <text class="cat mid" x="540.9" y="268">P1</text>
  <text class="cat-sub mid" x="540.9" y="282">selected</text>
  <text class="cat-sub mid" x="540.9" y="295">no seed</text>
  <rect class="them" opacity="0.35" x="589.3" y="67.4" width="14" height="7.3"/>
  <circle class="q1" cx="612.3" cy="71.4" r="2.5"/>
  <circle class="q1" cx="612.3" cy="70.7" r="2.5"/>
  <circle class="q1" cx="612.3" cy="70.2" r="2.5"/>
  <circle class="dot-them" cx="596.3" cy="70.8" r="6"/>
  <text class="v-them mid" x="596.3" y="59.4">95.3%</text>
  <text class="cat mid" x="596.3" y="268">T1</text>
  <text class="cat-sub mid" x="596.3" y="282">tools</text>
  <text class="cat-sub mid" x="596.3" y="295">no seed</text>
</svg>
</figure>

Recall climbs the ladder the design predicts. The model alone (A0, 0.4%) says it has no record of what is asked rather than invent it; the seed files alone (A1, 0.5%) add nothing; a single retrieval (D1, 30.6%) adds the memory; re-selection during reasoning (C1, 37.5%) adds the files the question does not name. Full marks in every run: 0, 0, 9, 13, 3, 11 and 6 questions across A0–C1, B1's over six runs and C1's over seven. The direct questions are drawn mechanically, so they include call sites no document describes, which makes the 128 harder than the pilot's 24.

<figure class="fig">
<svg viewBox="0 0 640 294" role="img" aria-label="Figure 9.12-2. The effect of each contrast on mean recall over 128 paired questions, with its 95% bootstrap interval and Holm-corrected Wilcoxon p-value: The memory (C1 − A1 · Holm p = 1.9e-17), +37.0 points (+31.3 to +42.6); Reasoning-time reprojection (C1 − B1 · Holm p = 2.8e-6), +5.3 points (+3.2 to +7.4); Continuous against retrieve-once (C1 − D1 · Holm p = 1.7e-4), +6.9 points (+3.6 to +10.1); Reasoning-time reprojection without the seed (C0 − B0 · Holm p = 0.007), +4.9 points (+1.7 to +8.2); The seed files (C1 − C0 · Holm p = 0.539), +0.7 points (−1.7 to +3.3).">
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
  <text class="cat-sub" x="16" y="96">C1 − A1 · Holm p = 1.9e-17</text>
  <path class="line-us" d="M531.8 84 H575.4"/>
  <circle class="dot-us" cx="553.8" cy="84" r="6"/>
  <text class="v-us mid" x="553.8" y="74">+37.0</text>
  <text class="cat" x="16" y="122">Reasoning-time reprojection</text>
  <text class="cat-sub" x="16" y="136">C1 − B1 · Holm p = 2.8e-6</text>
  <path class="line-us" d="M423 124 H439.4"/>
  <circle class="dot-us" cx="431.2" cy="124" r="6"/>
  <text class="v-us mid" x="431.2" y="114">+5.3</text>
  <text class="cat" x="16" y="162">Continuous against retrieve-once</text>
  <text class="cat-sub" x="16" y="176">C1 − D1 · Holm p = 1.7e-4</text>
  <path class="line-us" d="M424.7 164 H449.8"/>
  <circle class="dot-us" cx="437.2" cy="164" r="6"/>
  <text class="v-us mid" x="437.2" y="154">+6.9</text>
  <text class="cat" x="16" y="202">Reasoning-time reprojection without the seed</text>
  <text class="cat-sub" x="16" y="216">C0 − B0 · Holm p = 0.007</text>
  <path class="line-us" d="M417.2 204 H442.3"/>
  <circle class="dot-us" cx="429.5" cy="204" r="6"/>
  <text class="v-us mid" x="429.5" y="194">+4.9</text>
  <text class="cat" x="16" y="242">The seed files</text>
  <text class="cat-sub" x="16" y="256">C1 − C0 · Holm p = 0.539</text>
  <path class="line-us" opacity="0.45" d="M404 244 H423.3"/>
  <circle class="dot-us" opacity="0.45" cx="413.6" cy="244" r="6"/>
  <text class="v-them mid" x="413.6" y="234">+0.7</text>
</svg>
</figure>

| Contrast | Arms | Effect | 95% interval | p (Holm) |
|---|---|---:|---:|---:|
| The memory | C1 − A1 | 37.0 | 31.3 to 42.6 | 1.9 × 10⁻¹⁷ |
| Reasoning-time reprojection | C1 − B1 | 5.3 | 3.2 to 7.4 | 2.8 × 10⁻⁶ |
| Continuous against retrieve-once | C1 − D1 | 6.9 | 3.6 to 10.1 | 1.7 × 10⁻⁴ |
| Reasoning-time reprojection without the seed | C0 − B0 | 4.9 | 1.7 to 8.2 | 0.0069 |
| The seed files | C1 − C0 | 0.7 | −1.7 to 3.3 | 0.54 |

Every contrast that removes a part of provenance retrieval lowers recall, with an interval clear of zero; the seed files change nothing measurable.

<figure class="fig">
<svg viewBox="0 0 640 384" role="img" aria-label="Figure 9.12-3. Each of the 128 questions' mean recall under reprojection during the reasoning (C1) against the held projection (B1): 71 questions lie above the diagonal, where C1 recalls more, 29 below it, where B1 does, and 28 on it.">
  <text class="ttl" x="16" y="20">Figure 9.12-3 · Question by question</text>
  <text class="ttl-sub" x="16" y="38">each question's mean recall over its runs · 71 above the diagonal, 29 below, 28 on it</text>
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
  <circle class="us" opacity="0.3" cx="303.3" cy="86.7" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="80" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="260" r="4"/>
  <circle class="us" opacity="0.3" cx="186.7" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="303.3" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="280" cy="180" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="273.3" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="196" r="4"/>
  <circle class="us" opacity="0.3" cx="228.7" cy="164" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="80" r="4"/>
  <circle class="us" opacity="0.3" cx="268.3" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="172.7" cy="196" r="4"/>
  <circle class="us" opacity="0.3" cx="186.7" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="331.3" cy="116" r="4"/>
  <circle class="us" opacity="0.3" cx="140" cy="220" r="4"/>
  <circle class="us" opacity="0.3" cx="247.3" cy="164" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="240" r="4"/>
  <circle class="us" opacity="0.3" cx="233.3" cy="150" r="4"/>
  <circle class="us" opacity="0.3" cx="178.9" cy="246.7" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="130" r="4"/>
  <circle class="us" opacity="0.3" cx="151.7" cy="230" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="220" r="4"/>
  <circle class="us" opacity="0.3" cx="172.7" cy="292" r="4"/>
  <circle class="us" opacity="0.3" cx="219.3" cy="148" r="4"/>
  <circle class="us" opacity="0.3" cx="186.7" cy="180" r="4"/>
  <circle class="us" opacity="0.3" cx="228.7" cy="156" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="280" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="220" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="310" r="4"/>
  <circle class="us" opacity="0.3" cx="198.3" cy="220" r="4"/>
  <circle class="us" opacity="0.3" cx="245" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="175" cy="220" r="4"/>
  <circle class="us" opacity="0.3" cx="291.7" cy="190" r="4"/>
  <circle class="us" opacity="0.3" cx="128.3" cy="280" r="4"/>
  <circle class="us" opacity="0.3" cx="315" cy="100" r="4"/>
  <circle class="us" opacity="0.3" cx="338.3" cy="70" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="151.7" cy="190" r="4"/>
  <circle class="us" opacity="0.3" cx="175" cy="210" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="300" r="4"/>
  <circle class="us" opacity="0.3" cx="163.3" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="85.6" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="300" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="326.7" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="260" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="326.7" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="260" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="320" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="300" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="300" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="140" cy="260" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="326.7" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="280" r="4"/>
  <circle class="us" opacity="0.3" cx="81.7" cy="330" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="260" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="151.7" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="170" r="4"/>
  <circle class="us" opacity="0.3" cx="268.3" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="334.4" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="180" r="4"/>
  <circle class="us" opacity="0.3" cx="128.3" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="85.6" cy="220" r="4"/>
  <circle class="us" opacity="0.3" cx="194.4" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="338.3" cy="70" r="4"/>
  <circle class="us" opacity="0.3" cx="256.7" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="294" cy="108" r="4"/>
  <circle class="us" opacity="0.3" cx="291.7" cy="80" r="4"/>
  <circle class="us" opacity="0.3" cx="172.7" cy="244" r="4"/>
  <circle class="us" opacity="0.3" cx="219.3" cy="180" r="4"/>
  <circle class="us" opacity="0.3" cx="284.7" cy="148" r="4"/>
  <circle class="us" opacity="0.3" cx="210" cy="204" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="324" r="4"/>
  <circle class="us" opacity="0.3" cx="126" cy="292" r="4"/>
  <circle class="us" opacity="0.3" cx="144.7" cy="252" r="4"/>
  <circle class="us" opacity="0.3" cx="175" cy="230" r="4"/>
  <circle class="us" opacity="0.3" cx="191.3" cy="188" r="4"/>
  <circle class="us" opacity="0.3" cx="280" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="294" cy="92" r="4"/>
  <circle class="us" opacity="0.3" cx="228.7" cy="140" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="268" r="4"/>
  <circle class="us" opacity="0.3" cx="154" cy="268" r="4"/>
  <circle class="us" opacity="0.3" cx="294" cy="116" r="4"/>
  <circle class="us" opacity="0.3" cx="294" cy="100" r="4"/>
  <circle class="us" opacity="0.3" cx="81.7" cy="330" r="4"/>
  <circle class="us" opacity="0.3" cx="105" cy="300" r="4"/>
  <circle class="us" opacity="0.3" cx="140" cy="280" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="240" r="4"/>
  <circle class="us" opacity="0.3" cx="93.3" cy="290" r="4"/>
  <circle class="us" opacity="0.3" cx="198.3" cy="180" r="4"/>
  <circle class="us" opacity="0.3" cx="128.3" cy="210" r="4"/>
  <circle class="us" opacity="0.3" cx="70" cy="340" r="4"/>
  <circle class="us" opacity="0.3" cx="268.3" cy="110" r="4"/>
  <circle class="us" opacity="0.3" cx="151.7" cy="300" r="4"/>
  <circle class="us" opacity="0.3" cx="350" cy="60" r="4"/>
  <circle class="us" opacity="0.3" cx="116.7" cy="200" r="4"/>
  <circle class="us" opacity="0.3" cx="291.7" cy="60" r="4"/>
  <text class="t-dim" x="82" y="76">C1 higher</text>
  <text class="t-dim" x="338" y="330" text-anchor="end">B1 higher</text>
  <text class="t-dim mid" x="210" y="372">B1 recall · projection held</text>
  <text class="t-dim" x="70" y="52">C1 recall · reprojecting</text>
  <text class="v-us" x="396" y="110">71</text>
  <text class="cat-sub" x="396" y="126">above the diagonal: C1 recalls more</text>
  <text class="v-them" x="396" y="166">29</text>
  <text class="cat-sub" x="396" y="182">below it: B1 recalls more</text>
  <text class="v-them" x="396" y="222">28</text>
  <text class="cat-sub" x="396" y="238">on it: the same recall</text>
</svg>
</figure>

The gain is broad: 71 of 128 questions sit above the diagonal, 29 below and 28 on it.

<figure class="fig">
<svg viewBox="0 0 640 174" role="img" aria-label="Figure 9.12-4. The gain in mean recall of reprojecting during the reasoning (C1) over the held projection (B1), split by whether the question names its answer file, with 95% bootstrap intervals: Names its file (n=17), +2.6 points (−2.3 to +7.2); Does not (n=111), +5.7 points (+3.5 to +8.0).">
  <text class="ttl" x="16" y="20">Figure 9.12-4 · Where the gain falls</text>
  <text class="ttl-sub" x="16" y="38">C1 − B1 in mean recall, points · 95% bootstrap interval over questions</text>
  <path class="grid" d="M372 66 V150 M449.3 66 V150 M526.7 66 V150 M604 66 V150"/>
  <text class="tick mid" x="372" y="164">−5</text>
  <text class="tick mid" x="449.3" y="164">0</text>
  <text class="tick mid" x="526.7" y="164">+5</text>
  <text class="tick mid" x="604" y="164">+10</text>
  <path class="parity" d="M449.3 66 V150"/>
  <text class="cat" x="16" y="82">Names its file (n=17)</text>
  <text class="cat-sub" x="16" y="96">the question names its answer file</text>
  <path class="line-us" opacity="0.45" d="M413.7 84 H560"/>
  <circle class="dot-us" opacity="0.45" cx="489.6" cy="84" r="6"/>
  <text class="v-them mid" x="489.6" y="74">+2.6</text>
  <text class="cat" x="16" y="122">Does not (n=111)</text>
  <text class="cat-sub" x="16" y="136">the question leaves its answer file unnamed</text>
  <path class="line-us" d="M503.3 124 H573.8"/>
  <circle class="dot-us" cx="537.8" cy="124" r="6"/>
  <text class="v-us mid" x="537.8" y="114">+5.7</text>
</svg>
</figure>

The gain lies on the 111 questions that do not name their answer's file, 5.7 points (3.5 to 8.0); on the 17 that do it is 2.6 (−2.3 to 7.2), indistinguishable from zero — a split fixed before any run. What the question names, the opening finds; what it does not, the reasoning reaches, and only a reprojection inside the reasoning places it while the reasoning can use it.

<figure class="fig">
<svg viewBox="0 0 640 324" role="img" aria-label="Figure 9.12-5. The share of answers whose projection held an answer file, by decoded token, with 95% bootstrap bands: C1 from 51.5% at the first token to 58.6% at token 4,096; B1 from 51.6% at the first token to 56.2% at token 4,096; D1 from 48.7% at the first token to 54.9% at token 4,096.">
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
  <path class="area-us" d="M62 145.9 70.7 145.9 79.3 96.7 88 100.6 96.6 102.9 105.3 108.4 113.9 109.2 122.6 108.4 131.3 105.3 139.9 112.3 148.6 103 157.2 106.9 165.9 109.2 174.5 110 183.2 101.4 191.8 104.5 200.5 109.2 209.2 107.6 217.8 99.8 226.5 103.8 235.1 102.2 243.8 98.3 252.4 104.5 261.1 103 269.8 106.1 278.4 103 287.1 100.6 295.7 100.6 304.4 103.7 313 100.6 321.7 101.4 330.3 100.6 339 102.2 347.7 99.8 356.3 100.6 365 102.2 373.6 99.1 382.3 99.8 390.9 99.1 399.6 98.3 408.3 99.1 416.9 99.8 425.6 101.4 434.2 100.6 442.9 98.3 451.5 99.1 460.2 99.8 468.8 99.1 486.2 99.1 494.8 99.8 503.5 99.1 512.1 99.1 520.8 99.8 529.4 99.1 538.1 99.8 555.4 99.8 564.1 101.4 572.7 99.8 581.4 98.3 590 99.1 598.7 99.1 607.3 99.8 616 99.1 616 201.4 590 201.4 581.4 200.6 572.7 202.2 564.1 203.8 555.4 203 546.8 202.2 529.4 202.2 520.8 203 512.1 202.2 503.5 202.2 494.8 203 486.2 202.2 477.5 202.2 468.8 201.4 460.2 203 451.5 203 442.9 201.4 434.2 203.7 425.6 204.5 416.9 203 408.3 202.2 399.6 201.4 390.9 202.2 382.3 203 373.6 202.2 365 205.3 356.3 203.7 347.7 203.7 339 206.1 330.3 204.5 321.7 205.3 313 204.5 304.4 206.9 295.7 203 287.1 203.7 278.4 206.1 269.8 209.2 261.1 206.1 252.4 206.9 243.8 199.1 235.1 203.8 226.5 205.3 217.8 200.6 209.2 208.4 200.5 211.6 191.8 206.9 183.2 203.8 174.5 212.3 165.9 211.6 157.2 208.4 148.6 205.3 139.9 213.9 131.3 206.9 122.6 209.2 113.9 212.3 105.3 210.8 96.6 205.3 88 204.5 79.3 199.8 70.7 252.2 62 252.2 Z"/>
  <path class="them" opacity="0.12" d="M62 144.4 88 144.4 96.6 143.4 105.3 139.8 113.9 139.8 122.6 135.3 131.3 140.7 139.9 138 148.6 134.3 157.2 129.8 165.9 128.9 174.5 131.6 183.2 129.8 191.8 130.7 200.5 128 209.2 122.5 217.8 120.7 226.5 121.6 235.1 118.9 243.8 118.9 252.4 120.7 261.1 119.7 269.8 117 278.4 117 287.1 115.2 295.7 117 313 117 321.7 121.6 330.3 120.7 339 117.9 347.7 116.1 356.3 116.1 365 118.9 373.6 117.9 382.3 116.1 390.9 117 399.6 117.9 408.3 117 416.9 117 425.6 116.1 434.2 115.2 442.9 116.1 486.2 116.1 494.8 115.2 503.5 114.3 512.1 114.3 520.8 116.1 538.1 116.1 546.8 115.2 555.4 114.3 564.1 115.2 572.7 115.2 581.4 114.3 607.3 114.3 616 115.2 616 217.3 607.3 216.4 581.4 216.4 572.7 217.3 564.1 217.3 555.4 216.4 546.8 217.3 538.1 218.2 520.8 218.2 512.1 216.4 503.5 216.4 494.8 217.3 477.5 217.3 468.8 218.2 442.9 218.2 434.2 217.3 425.6 218.2 416.9 219.1 408.3 219.1 399.6 220.9 390.9 220 382.3 218.2 373.6 220.9 365 221.8 356.3 218.2 347.7 218.2 330.3 221.8 321.7 222.8 313 219.1 304.4 218.2 295.7 218.2 287.1 216.4 278.4 218.2 269.8 218.2 261.1 221.8 252.4 221 243.8 219.1 235.1 220 226.5 222.8 217.8 220.9 209.2 224.6 200.5 230.1 191.8 232.8 183.2 233.7 174.5 234.6 165.9 232.8 157.2 233.7 148.6 238.3 139.9 242.8 131.3 246.5 122.6 241.9 113.9 246.5 105.3 246.5 96.6 250.1 88 251 79.3 251.9 62 251.9 Z"/>
  <path class="line-them dash" d="M62 219.1 105.3 219.1 113.9 213.6 131.3 213.6 139.9 211.8 148.6 211.8 157.2 206.4 174.5 206.4 183.2 204.5 191.8 202.7 209.2 199.1 217.8 197.2 226.5 197.2 235.1 193.6 261.1 193.6 269.8 191.8 278.4 191.8 287.1 189.9 295.7 189.9 304.4 188.1 313 184.5 321.7 184.5 330.3 182.7 347.7 182.7 356.3 179 365 177.2 468.8 177.2 477.5 175.4 616 175.4"/>
  <path class="line-them" d="M62 199.1 79.3 199.1 88 198.2 96.6 197.2 105.3 193.6 113.9 193.6 122.6 189 131.3 193.6 139.9 190.9 148.6 187.2 157.2 182.7 165.9 181.7 174.5 183.6 183.2 182.7 191.8 182.7 200.5 179.9 209.2 174.5 217.8 171.7 226.5 172.6 235.1 170.8 243.8 169.9 252.4 171.7 261.1 170.8 269.8 168.1 278.4 168.1 287.1 166.3 295.7 168.1 304.4 168.1 313 169 321.7 172.6 330.3 171.7 347.7 168.1 356.3 168.1 365 170.8 373.6 169.9 382.3 168.1 390.9 169 399.6 169.9 408.3 169 416.9 169 425.6 168.1 434.2 167.2 442.9 168.1 468.8 168.1 477.5 167.2 486.2 167.2 494.8 166.3 503.5 165.3 512.1 165.3 520.8 167.2 538.1 167.2 546.8 166.3 555.4 165.3 564.1 166.3 572.7 166.3 581.4 165.3 607.3 165.3 616 166.3"/>
  <path class="line-us" d="M62 199.8 70.7 199.8 79.3 148.3 88 152.2 96.6 153.8 105.3 159.2 113.9 160.8 122.6 158.4 131.3 156.1 139.9 163.1 148.6 153.7 157.2 156.9 165.9 160 174.5 160.8 183.2 152.2 191.8 155.3 200.5 160 209.2 157.7 217.8 149.8 226.5 153.8 235.1 152.2 243.8 148.3 252.4 155.3 261.1 153.8 269.8 156.9 278.4 153.8 287.1 151.4 295.7 151.4 304.4 154.5 313 152.2 321.7 153 330.3 152.2 339 153.8 347.7 151.4 356.3 151.4 365 153 373.6 149.8 382.3 150.6 390.9 149.8 399.6 149.1 408.3 149.8 416.9 150.6 425.6 152.2 434.2 151.4 442.9 149.1 451.5 150.6 460.2 150.6 468.8 149.8 486.2 149.8 494.8 150.6 503.5 149.8 512.1 149.8 520.8 150.6 529.4 149.8 538.1 150.6 555.4 150.6 564.1 152.2 572.7 150.6 581.4 149.1 590 149.8 616 149.8"/>
  <g class="legend">
  <path class="line-us" d="M330 53 H344"/>
  <text x="348" y="56">C1 full</text>
  <path class="line-them" d="M407.4 53 H421.4"/>
  <text x="425.4" y="56">B1 held</text>
  <path class="line-them dash" d="M484.8 53 H498.8"/>
  <text x="502.8" y="56">D1 once</text>
  </g>
</svg>
</figure>

With reprojection the share of answers with an answer file in the working set jumps from 51.5% at the opening to 58.8% by the 128th decoded token, inside the reasoning's first two reprojection intervals, and holds there. Held or retrieved once, it stays at the opening's 51.6% or 48.7% until the reasoning ends, and only then climbs — to 56.2% and 54.9% by token 4,096 — as those arms' answers reproject: the file arrives, but after the thought that needed it.

<figure class="fig">
<svg viewBox="0 0 640 188" role="img" aria-label="Figure 9.12-6. Mean recall of the answers by whether their projection held an answer file for the first answer token, with 95% bootstrap intervals: answer file placed (1,534 answers): 51.4% (49.6% to 53.2%); not placed (1,282 answers): 13.7% (12.1% to 15.3%).">
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
  <text class="cat-sub" x="16" y="98">1,534 answers</text>
  <rect class="us" x="250" y="75" width="299.8" height="22"/>
  <path class="grid base" d="M539.4 80 V92 M539.4 86 H560.5 M560.5 80 V92"/>
  <text class="v-us" x="568.5" y="90">51.4%</text>
  <text class="cat" x="16" y="130">Not placed</text>
  <text class="cat-sub" x="16" y="144">1,282 answers</text>
  <rect class="us-soft" x="250" y="121" width="80" height="22"/>
  <path class="grid base" d="M320.9 126 V138 M320.9 132 H339.1 M339.1 126 V138"/>
  <text class="v-us" x="347.1" y="136">13.7%</text>
</svg>
</figure>

Over 2,816 answers in the projecting arms, recall is 51.4% (49.6–53.2%) with the answer file placed for the first answer token and 13.7% (12.1–15.3%) without it: retrieval and the answers are one measurement seen twice.

**Against retrieval-augmented generation.** R1 hands the model the files BM25 ranks best for the question, as text, under a 24,000-token budget; P1 hands it the memory's own selection the same way; T1 reads the repository with tools. Their contrasts form a second Holm family, so the five above are corrected without them:

<figure class="fig">
<svg viewBox="0 0 640 254" role="img" aria-label="Figure 9.12-2b. The effect of each contrast on mean recall over 128 paired questions, with its 95% bootstrap interval and Holm-corrected Wilcoxon p-value: the memory against BM25 retrieval (C1 − R1 · Holm p = 8.7e-4), −15.1 points (−23.0 to −7.1); the memory's selection against BM25's, both read as text (P1 − R1 · Holm p = 0.047), −8.5 points (−16.9 to −0.1); the memory against tool use (C1 − T1 · Holm p = 4.0e-21), −57.8 points (−64.1 to −51.5); BM25 retrieval against the seed files (R1 − A1 · Holm p = 2.0e-18), +52.1 points (+45.6 to +58.4).">
  <text class="ttl" x="16" y="20">Figure 9.12-2b · The memory against the baselines</text>
  <text class="ttl-sub" x="16" y="38">paired difference in mean recall, points · 128 questions · 95% interval · faded: Holm p ≥ 0.05</text>
  <path class="grid" d="M372 66 V230 M389.8 66 V230 M407.7 66 V230 M425.5 66 V230 M443.4 66 V230 M461.2 66 V230 M479.1 66 V230 M496.9 66 V230 M514.8 66 V230 M532.6 66 V230 M550.5 66 V230 M568.3 66 V230 M586.2 66 V230 M604 66 V230"/>
  <text class="tick mid" x="372" y="244">−70</text>
  <text class="tick mid" x="407.7" y="244">−50</text>
  <text class="tick mid" x="443.4" y="244">−30</text>
  <text class="tick mid" x="479.1" y="244">−10</text>
  <text class="tick mid" x="496.9" y="244">0</text>
  <text class="tick mid" x="532.6" y="244">+20</text>
  <text class="tick mid" x="568.3" y="244">+40</text>
  <text class="tick mid" x="604" y="244">+60</text>
  <path class="parity" d="M496.9 66 V230"/>
  <text class="cat" x="16" y="82">The memory against BM25 retrieval</text>
  <text class="cat-sub" x="16" y="96">C1 − R1 · Holm p = 8.7e-4</text>
  <path class="line-us" d="M455.9 84 H484.2"/>
  <circle class="dot-us" cx="470" cy="84" r="6"/>
  <text class="v-us mid" x="470" y="74">−15.1</text>
  <text class="cat" x="16" y="122">The memory's selection against BM25's, as text</text>
  <text class="cat-sub" x="16" y="136">P1 − R1 · Holm p = 0.047</text>
  <path class="line-us" d="M466.8 124 H496.8"/>
  <circle class="dot-us" cx="481.7" cy="124" r="6"/>
  <text class="v-us mid" x="481.7" y="114">−8.5</text>
  <text class="cat" x="16" y="162">The memory against tool use</text>
  <text class="cat-sub" x="16" y="176">C1 − T1 · Holm p = 4.0e-21</text>
  <path class="line-us" d="M382.6 164 H405"/>
  <circle class="dot-us" cx="393.7" cy="164" r="6"/>
  <text class="v-us mid" x="393.7" y="154">−57.8</text>
  <text class="cat" x="16" y="202">BM25 retrieval against the seed files</text>
  <text class="cat-sub" x="16" y="216">R1 − A1 · Holm p = 2.0e-18</text>
  <path class="line-us" d="M578.4 204 H601.2"/>
  <circle class="dot-us" cx="589.9" cy="204" r="6"/>
  <text class="v-us mid" x="589.9" y="194">+52.1</text>
</svg>
</figure>

BM25 retrieval recalls 52.6% (46.1–58.9%) against the memory's 37.5%, 15.1 points (7.1 to 23.0) more. To find where, each expected item of C1's seventh run, R1's fourth and P1's run was looked for in the text the model held as its answer began — for C1, the bodies of the turns the governing projection point injected; for R1 and P1, the files their prompts carried — by the same spellings recall matches (`zend/examples/dependency_battery/view.rs`):

| Arm | Items | In an answer file's text | Named, when there | Named, when not | Anywhere in view | Named, when anywhere |
|---|---:|---:|---:|---:|---:|---:|
| C1 (the memory) | 445 | **48.5%** | 69.9% | 14.0% | **60.4%** | 66.2% |
| R1 (BM25 files) | 445 | 38.2% | **97.6%** | 22.5% | 56.0% | **90.0%** |
| P1 (the memory's selection, as text) | 445 | 39.8% | 96.0% | 15.3% | 50.8% | 92.0% |

The memory puts the answer in front of the model more often than BM25 does — 48.5% of expected items were in the text of an answer file it injected, against 38.2% for the files BM25 retrieved, and 60.4% against 56.0% anywhere in view. It placed 48.3% of an answer file's reading turns, and those carried the item more often than BM25's whole files did. The difference is in what the model makes of an item once it holds it: from BM25's plain text it names 97.6%, from the memory's K/V 69.9% (`docs/results/battery_128/summary_run11.md`, `table_9_12_in_view.csv`). Handing the model the memory's own selection as text (P1, under R1's budget and prompt) lifts recall from 37.5% to 44.0%, and the model then names 96.0% of what an answer file carries: 6.5 of the 15.1 points are the model's reading of K/V, not provenance's selection. How well a model reads K/V placed from memory is a capability of the model, and it has grown with every model this system has been upgraded to. Provenance's selection puts an answer file first more often than BM25 does (hit@1 40.6% against 28.9%, `table_9_12_retrieval.csv`); what remains sits with BM25's wider spread of whole files under a fixed text budget. T1, reading the source with 12.3 tool calls an answer, recalls 95.3%, the reference for what the questions allow.

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

An answer file is first of 7,943 on 49% of questions and in the top five on 69%, about 1,500 and 430 times a random ranking — by sign agreement alone, with no text index and no embedding model. By kind it is first on 43% of direct, 52% of transitive and 50% of architectural questions. Most misses lose their top places to the same few files that score on every question (`tests_grid.rs`, `file_grep.yaml`, `tasks.rs`, `AGENTS.md`).

**The pilot.** A 24-question pilot, run on production from 2026-11-10 as retrieval changed (57.6% → 62.2% by 2026-11-20; `docs/results/dependency_battery_2026-11-20_b.md`), first measured reasoning-time reprojection with a daemon-wide switch, `reproject_in_thinking: false`, which passes over every boundary inside the think block and keeps the rest — the opening, the answer's boundaries, and one reprojection on `</think>`. Over three runs per arm at one build (`docs/results/dependency_battery_2026-11-23_l_m.md`), reprojecting averaged **57.3%** and holding **38.8%**; every held run scored below every reprojecting run (p = 0.05, one-sided permutation), and per question reprojection won 14 to 4 (sign test, p = 0.031). Its loss fell where the answer lives in a file the question does not name. The 128-question battery measures the same contrast at 5.3 points: smaller, and with an interval clear of zero.

**What the answers show.** The memory holds documentation and module structure more than the call graph. Answers cite their sources — a design document's section, a module header — and when the model had not read the source it said so, declining rather than inventing; the errors found on reading are near-misses (a constant placed in the header beside the one that defines it, an out-of-date caller), not invented names.

**What is public.** The source, the raw measurement rows, the battery with its ground truth and analysis (`zend/examples/dependency_battery/`, `docs/results/battery_128/`), the pilot's results and the scoring lab with its formulas (`candle-conversation/examples/provenance_gates`) are in the repository (Appendix A). The live system is open to signed-in users; the ingested conversation is not released.

---

### 9.13 The System as Its Own Workload

The system is also in use, and the use exercises what no benchmark can: memory that outlives the process that wrote it, written by one model and read by another. The journal store — one master journal per group, with `people/`, `conversations/`, `topics/`, `threads/` and `bots/` — holds minute-stamped records written by whichever agent is speaking, each naming its author and account, so the memory is in no model's weights and swapping the reader does not orphan it. Entries fold as the KV cache does, steps into days into weeks into years, so a years-long journal reads in one prompt. Each group declares what its members may read; provenance ranks only what a reader may see. The retrieval numbers of §9.11.1 and §9.12 come from this live gallery, scanned over HTTP while it is written to. What this buys — sessions that survive restarts, a model swap that costs a re-ingest, an attributable record across agents and months — is operating experience, not a measured result, and is reported as such.

#### 9.13.1 This revision, as a workload

Much of an earlier draft of this version was revised by the system: a Flash-Next session on the production daemon, asked to critique and then revise the paper, ran 85 tool rounds over ~320K tokens and made thirty-one edits (`docs/results/paper_revision_session_2026-11-09.md`). Asked to fill §9.12's then-empty results table, it searched for a measured table, found none, wrote that the battery was not yet run, and corrected the other places that claimed its results; it missed some, which a later review found. Its failures were mechanical — stale anchors after neighbouring edits — a defect of the editing tools.

It decoded one token for every 2.9 prefilled (82,963 against 238,619). Every token was prefilled once, the earlier turns' K/V projected back rather than resent, where a client re-sending its history would have sent on the order of eighty times as much. Decode held at ~102 t/s at 48K–130K tokens, ~110 at 162K–237K and ~106 at 309K–321K: the working set does not grow with the history.

OpenRouter's picture of agent traffic is ~87% cached prompt, 12% novel prompt and 1% output, "because an agent re-sends its whole context on every step" [OpenRouter, *Cache is Crucial*, October 2026; 47]; on SWE-bench Verified under OpenHands, input tokens dominate the cost of agentic coding even with caching [Bai et al., arXiv:2604.22750; 48]. Against genuinely new input, this session wrote about four times as much per token read (Figure 9.13-1). OpenRouter's novel prompt includes re-reads a provider failed to cache, so this is a comparison, not a result; counted as a re-sending client counts it, ~19M tokens, the session would be 1 : 230. What the architecture changes is not how much a long session would re-read, but that it does not.

<figure class="fig">
<svg viewBox="0 0 640 128" role="img" aria-label="Figure 9.13-1. Tokens written per token of new input. This revision session decoded one token for every 2.9 prefilled, 0.348. OpenRouter's agentic mix, output against novel prompt, about one in twelve, 0.083.">
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

Exposure is not output error. Attention's output is a convex combination of values, so value errors are averaged, not summed, and whether full-attention output error grows with N depends on the attention distribution. Measured, it does not within the windows we reach (§9.6.8, Figure 9.6-3): C10's divergence from BF16 on Qwen3.5-9B is 6.5e-3 nats at 4K and 3.5e-3 at 124K, and every whole-prefix model of §9.6 passes the story gate at C10 at every depth run. What the literature lacks is a bound on exposure independent of depth. More VRAM defers the threshold, but a context persistent across months is 100 million tokens, which no card holds at F16, and summarization and eviction discard the verbatim detail. §11.2 shows decoupling the working set from depth is sufficient; no full-attention system has that property, because its working set is the whole context.

### 11.2 Theorem: Asymptotic Numerical Stability Under Provenance-Selected Attention

ε(t) denotes the **total stored numerical error** of token t's K/V from every source — quantization, rounding, decode drift. Lemma 1 bounds the sum of ε(t) over the tokens entering a step, that step's exposure; the theorem bounds what each attention head outputs from them. Neither uses anything specific to compression.

<figure class="fig">
<svg viewBox="0 0 640 224" role="img" aria-label="Figure 11-1. The bound. Under full attention a step's exposure, the summed stored error of the tokens it reads, grows linearly with context depth N; more VRAM only moves the point where tokens must be compressed or evicted. Under provenance-selected attention it is bounded by epsilon hot plus the retrieval budget times epsilon ret, at any depth.">
  <text class="ttl" x="16" y="20">Figure 11-1 · What a step is exposed to, as the context grows</text>
  <text class="ttl-sub" x="16" y="36">summed stored error of the tokens a step reads · schematic</text>
  <path class="ax" d="M70 190 H610 M70 190 V52"/>
  <path class="ax thin" d="M210 56 V190 M330 56 V190"/>
  <path class="c-rise" d="M70 186 L600 58"/>
  <path class="c-flat" d="M70 178 C110 156 140 150 180 148 H600"/>
  <text class="t-rise" x="420" y="118">full attention: Θ(N · ε̄)</text>
  <text class="t-flat" x="300" y="168">provenance: ε_hot + W_warm_max · ε_ret</text>
  <text class="t-dim mid" x="210" y="204">16 GB fills</text>
  <text class="t-dim mid" x="330" y="204">80 GB fills</text>
  <text class="t-dim" x="580" y="206">depth N →</text>
  <text class="t-dim" x="76" y="64">Σ ε(t)</text>
</svg>
</figure>

---

**Lemma 1 (Bounded exposure).** *Let N denote context depth, B the fixed provenance selection token budget (a system constant), W_hot_max the hot tier capacity, W_warm_max = B − W_hot_max the retrieval budget — the tokens selected per step from outside the hot tier, whether they reside in the warm or the cold tier — and ε(t) the total accumulated numerical error of token t under any finite-precision arithmetic. Let ε_h, ε_warm and ε_cold bound the per-token error of a hot-, warm- and cold-tier token, ε_hot = W_hot_max · ε_h the hot tier's summed error, and ε_ret = max(ε_warm, ε_cold). Under attentional provenance selection, the total numerical error contribution of every generation step is bounded by a constant independent of N:*

$$\sum_{t \in \mathcal{W}} \varepsilon(t) \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{ret}} = O(1)$$

*Under full attention the same sum is Θ(N · ε̄), with ε̄ the mean per-token error over the context, and grows without limit. The bound's independence of N holds at any hardware scale and in any precision format; the format sets the constant, through ε_hot and ε_ret.*

---

**Proof of Lemma 1.** Partition the context into three tiers as N grows.

*Hot tier* (H): tokens always present in the working set. |H| ≤ W_hot_max is bounded by a hardware constant independent of N. Tokens in H are prefill-refreshed: their decode drift is zero and their quantization error is bounded by the selection kernel threshold, so each carries at most ε_h and together at most ε_hot = W_hot_max · ε_h, a small constant independent of N.

*Warm tier* (W_warm): tokens eligible for retrieval but not permanently resident. The provenance selection mechanism operates under the fixed budget B — at most W_warm_max tokens from outside the hot tier can be selected per generation step, from the warm and cold tiers together, regardless of how large either grows with N. This bound is structural: the selection kernel enforces it by construction. Let k_warm and k_cold be the numbers of warm- and cold-tier tokens a step selects; the budget enforces k_warm + k_cold ≤ W_warm_max. The warm tier's aggregate error contribution per step is therefore at most:

$$\sum_{t \in W_{\text{warm}} \cap \mathcal{W}} \varepsilon(t) \leq k_{\text{warm}} \cdot \varepsilon_{\text{warm}} \leq W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} = O(1)$$

a constant independent of N. As N grows and the warm corpus expands, more candidates compete for W_warm_max slots, but the aggregate contribution is bounded by the budget.

*Cold tier* (C): all remaining tokens. As N → ∞, |C| → ∞ while the retrieval budget W_warm_max stays fixed. A cold token is read only when a step selects it, and then within the budget, so the cold tier's contribution per step is at most:

$$\sum_{t \in C \cap \mathcal{W}} \varepsilon(t) \leq k_{\text{cold}} \cdot \varepsilon_{\text{cold}}$$

Boundary tokens fall under the same per-token bound: on the attention-only models their K/V is constructed against the current assembly (§5), and on the recurrent lineage they are sealed like interior ones (§5).

Summing across tiers, with k_warm + k_cold ≤ W_warm_max at every step — the budget is enforced per step, not on average, so the bound holds for each step and not only in expectation:

$$\sum_{t \in \mathcal{W}} \varepsilon(t) \leq \varepsilon_{\text{hot}} + k_{\text{warm}} \cdot \varepsilon_{\text{warm}} + k_{\text{cold}} \cdot \varepsilon_{\text{cold}} \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{ret}} = O(1)$$

□

---

**Theorem (Asymptotic Numerical Stability).** *Take one attention head of one layer at one step: query q, scale 1/√d, an attended set 𝒲 of n tokens, logits s_t = q · k_t / √d and weights p = softmax(s). Let each attended token's stored key and value differ from their exact values by δk_t and δv_t, with ‖δv_t‖ ≤ ε_V and a logit perturbation |q · δk_t| / √d ≤ η for every t ∈ 𝒲, and let R_V = max_t ‖v_t − c‖ be the radius of the exact values about any centre c.*

*(a) Bounded perturbation. The head's output o = Σ_t p_t v_t moves by at most*

$$‖\tilde{o} - o‖ \leq \varepsilon_V + \left(e^{2\eta} - 1\right) R_V \approx \varepsilon_V + 2\eta R_V$$

*(b) Bounded dilution. Let r ∈ 𝒲 be a token the step needs, its logit ahead of every other attended token's by a margin of at least Δ. Its weight, read from the stored K/V, keeps*

$$\tilde{p}_r \geq \frac{1}{1 + (n - 1)\, e^{-(\Delta - 2\eta)}}$$

*and if every other attended token's logit lies within Δ_max of r's, its exact weight is at most* $1 / (1 + (n - 1) e^{-\Delta_{max}})$.

*(c) Under provenance selection, n ≤ B at any depth and every attended token was sealed at the selector's threshold (Corollary 1): the perturbation of (a) is bounded by a constant, and a needed token's weight is bounded below by a constant, so the ratio of the signal a step needs to the perturbation it carries is bounded below independently of N. Under full attention n = N: the perturbation bound of (a) is the same, but among distractors within Δ_max of it the needed token's weight falls as Θ(1/N), so the signal the perturbation is measured against vanishes with depth — and in fixed memory M each token's share falls as M/N, so past the depth at which the threshold's format no longer fits, ε_V and η grow with N as well.*

**Proof.** Write s̃_t = s_t + δ_t, with |δ_t| ≤ η. For (a), for each t,

$$\frac{\tilde p_t}{p_t} \;=\; e^{\delta_t}\,\frac{\sum_u e^{s_u}}{\sum_u e^{s_u + \delta_u}} \;\in\; \left[e^{-2\eta},\, e^{2\eta}\right],$$

so $|\tilde{p}_t - p_t| \leq p_t (e^{2\eta} - 1)$, and summing, $‖\tilde{p} - p‖_1 \leq e^{2\eta} - 1$. The output changes by

$$\tilde o - o \;=\; \sum_t \tilde p_t\, \delta v_t \;+\; \sum_t (\tilde p_t - p_t)\, v_t .$$

The first term is a convex combination of the δv_t, so its norm is at most ε_V. In the second, Σ_t (p̃_t − p_t) = 0, so each v_t may be replaced by v_t − c, and its norm is at most $‖\tilde{p} - p‖_1 R_V \leq (e^{2\eta} - 1) R_V$.

For (b), $\tilde{p}_r = 1 / (1 + \sum_{t \neq r} e^{\tilde{s}_t - \tilde{s}_r})$, and s̃_r − s̃_t ≥ Δ − 2η for each of the n − 1 other tokens, so each term of the sum is at most $e^{-(\Delta - 2\eta)}$. The upper bound on the exact weight is the same sum with each term at least $e^{-\Delta_{max}}$.

(c) is (a) and (b) with n ≤ B under selection and n = N under full attention: under selection the ratio of the needed token's weight to the perturbation is at least $1 / [(1 + (B - 1) e^{-(\Delta - 2\eta)}) (\varepsilon_V + (e^{2\eta} - 1) R_V)]$, which names no N. □

**Averaging does not rescue full attention.** If the value errors are independent and zero-mean with spread σ, the value term Σ_t p̃_t δv_t is smaller than (a)'s worst case: its size is about σ‖p̃‖₂, which for weight spread over N comparable tokens is σ/√N. But the needed token's weight in that regime is Θ(1/N), so the ratio of signal to perturbation still falls as Θ(1/√N) with depth. Under selection both are set by B and the thresholds, at any N.

**Recurrent layers.** On the DeltaNet-hybrid models — Flash-Next is 36 Gated DeltaNet layers and 12 attention layers — the recurrent layers store no per-token K/V and read no working set: each holds one fixed-size state per sequence, `[h_v, d_v, d_k]`, kept in F32 and never quantized by the ladder, so what a step reads from them is the same size at any N. Each update is non-expansive — S ← e^g · S, g ≤ 0, then S ← S + β(v − Sk)kᵀ with k l2-normed and β ∈ (0, 1), so S(I − βkkᵀ) has operator norm at most ‖S‖ — and adds one F32 rounding (`candle-kernels/src/delta-net/delta_net_decode_kernel.cuh`). An error already in the state is therefore never amplified and decays wherever the gate is below one; the arithmetic is the reference model's own, and the bound neither adds to it nor relies on it. Measured with all 36 recurrent layers in the path, Flash-Next's divergence from BF16 is flat from 4K to 124K at C0 (2.2e-4 → 1.4e-4 nats) and at C5 (9.8e-3 → 7.6e-3) (§9.6.8).

**Cost is not error.** The bound is on what a step attends; the work of choosing it is separate. Exact attention reads at most B tokens a step at any depth. The 1-bit provenance scan that selects them reads every signature in the gallery, linear in N, once per reprojection — every 64 decoded tokens, off the decode's critical path (§6.6) — and its result is a selection, which the bound covers through the budget whatever the scan's size.

---

**Corollary 1 (The threshold bounds every attended token).** *In this system's design, warm- and cold-tier blocks alike originate from prefill-refreshed hot-tier blocks — they were quantized from clean activations and assigned a format by the adaptive per-block selection kernel at threshold θ, θ being the loosest threshold any block was sealed under. Measure ε(t) in the selector's own per-block metric — Q-projected relative error for K, normalised L2 for V — so that θ bounds it directly. Then ε_h ≤ θ and ε_ret ≤ θ, and since W_hot_max + W_warm_max = B:*

$$\sum_{t \in \mathcal{W}} \varepsilon(t) \leq B \cdot \theta, \qquad \text{every attended token's error} \leq \theta$$

*Every token a step attends carries at most the selection kernel's quality threshold, at any depth. The K metric is the error as the query sees it and the V metric the value's, so the thresholds bound the theorem's η and ε_V, relative to each block's scale: each attention head's output moves by a constant the thresholds set. Empirically, at C0 the always-attended blocks reach K_SNR 58.6 dB and V_SNR 58.8 dB (per-block ideal), and Flash-Next's output stays within 4.4e-4 nats of BF16 at C0 and 2.1e-2 at C5, production's sealing level, from 4K to 124K (§9.6.8).*

---

**The contrast is between bounds** (Figure 11-1). Both architectures bound how far stored error moves a head's output; what depth changes is the signal that perturbation competes with. Under provenance selection the attended set is at most B tokens, so a needed token's weight, and with it the signal-to-perturbation ratio, is bounded below at any depth. Under full attention the attended set is the whole context: a needed token's weight falls as Θ(1/N) among comparable distractors, the summed exposure of Lemma 1 grows as Θ(N · ε̄), and in fixed memory the per-token precision falls too. Neither bound names VRAM: a larger card moves the depth at which full attention must compress or evict, and does not remove it, while B and the thresholds are the same on every card.

**Empirical status.** The premises — a fixed budget, a fixed hot tier, a per-block threshold on everything sealed — hold by construction. Throughput is flat in depth from 32K to 128K (§9.6.2), and the live session of §9.13.1 keeps decode flat from 48K to 321K. The direct test of the premise is §9.11.2: with the slot capped at 32K and a history twelve times that, reprojection selected every planted turn at C10 and C0 alike, where one-shot and rolling-window arms selected none. §9.12 tests what the model does with the selected set; the theorem does not rest on it.

**Compression moves the constant, never the bound.** A selected cold block contributes its full error at any depth, so aggressive levels on cold K and on V raise ε_cold — a capacity trade bounded by the threshold each block was sealed under. The binding constraints are ε_hot, kept near zero by the refresh, and the retrieval threshold θ. The bound holds over whatever selection attends, and is lost only if selection grows the working set with N, which recovers the full-attention regime. Part (b) speaks of a needed token the selection holds; whether selection holds it is retrieval, which §9.12 measures (Figure 9.12-6).

### 11.3 Compression Architecture vs. Compression Primitive

KIVI, KVQuant and TurboQuant optimise the compression *primitive*: given that every block gets a format, which format is best. This paper optimises the *architecture*: when to quantize, what quality to guarantee per block, how to separate error sources, and how error scales with depth. Per-block selection absorbs a primitive's failures — a block Q4_0 cannot represent gets Q8_0 — which is why the adaptive system holds quality where a rotation fails at 4–5×. KVQuant's outlier insight is subsumed by the `_KS` sub-block formats and sink protection. On Llama-3.2-3B, adaptive C5 at 3.27× passes the multi-session identity test (§9.8) where uniform Q4_0 at 3.17× fails, and the query-aware K metric removed the cross-session contamination an isotropic metric produced at C8–C9.

### 11.4 Attention Was Always Retrieval

Attention is retrieval: Q queries K, scores select, V is aggregated by relevance. This system runs it at two scales — approximate 1-bit attention over the whole unbounded context to select the working set (§6.6), then exact attention over what it selected — hierarchical retrieval at constant exact-attention cost per step, the scan's linear 1-bit work running every 64 tokens off the decode's critical path, where standard attention is flat retrieval over N at every step.

It is not RAG, and the difference is structural. *The signal*: RAG retrieves from text, so iterative RAG (FLARE, Self-RAG) must wait for the model to write a query; Q vectors carry what the model is reaching for before any text forms. *Causal continuity*: RAG injects text into a fresh context, while this system promotes stored K/V computed in its original causal context, K stored un-rotated and positioned on read. *Leading the answer*: reprojection inside the think block, whose K/V is dropped at seal, steers the answer's context by everything the model thought (§6.5). A RAG system given Q→Q matching, causal K/V continuity and retrieval from Q during generation would no longer be RAG; it would be this architecture. §9.12 shows it working: the answer file found among 7,943 by sign agreement alone, and a single retrieval before reasoning 6.9 points below continuous reprojection.

### 11.5 Constraint-Driven Innovation as a Methodology

The system was designed bottom-up from two constraints accepted together: the hardest form of persistent memory, and a hardware ceiling that could not be exceeded. Each closed a standard solution and forced a replacement that holds beyond the constraint:

- **Native quantized kernels.** Dequantising to BF16 OOMed at 16 GB; inline dequantisation never materialises the copy, and saves its traffic on every card (§8.1).
- **Provenance-selected attention.** The full context could not be held hot; selecting a bounded working set turned out to be the property the theorem uses.
- **Two-phase quantization.** Materialising turns at F16 across hundreds of sessions OOMed; sealing per block and refreshing per turn separates two error mechanisms the literature conflates.
- **Online expert prediction.** At 44% residency prediction was required; learning online from live routing, with a cross-prompt prior, covers 4.7× the cold experts popularity does, with no calibration (§9.7).
- **Live structural segments.** Rotating system-prompt sections made stored template K/V silently wrong; storing only portable content and prefilling the template against the live prefix closes the error on attention-only models (§5).

In each case the replacement is more efficient in the dimension the constraint bounds, and that efficiency generalised because the constraint exposed an avoidable cost. It is a pattern from one system, not a law, and its counterexample is in the list: the structural-segment design had to be set aside for the recurrent production model. The evaluation followed the same rule — the system's own codebase as the test, checkable by anyone who reads it.

---

## 12. Conclusion

The four contributions — online Markov expert prediction with wave-batched GEMM, per-block adaptive KV quantization with two-phase refresh, attentional provenance indexing with mid-decode reprojection, and an unbounded three-tier context — have precedents individually and had not been integrated.

Together they run a 180B MoE model for eight sessions on a 16 GB laptop GPU, above every published llama.cpp run of it, on hosts with two to four times the memory; keep throughput flat from 32K to 128K; compress the KV cache up to 7.6× inline, nearly 4× llama.cpp's q8_0 for less decode at 32K, with every session of every width row reproducing its reference; and serve concurrent sessions at 1.4–37.9× the best published single-stream rate for the same model and card class, up to 2,869.9 t/s across 64 sessions, with a 284B model at sixteen sessions on one GPU at 4.1× the best published single-GPU decode. The memory behind it is a 42M-token substrate held at C5, its code gallery 36.1M tokens; under sixteen concurrent answers the card's resident context peaks at 945 thousand tokens, 21.9 GiB at full precision, while the expert zone streams the rest of the model. The most comparable published study found standard frameworks degrade beyond two concurrent users on an RTX 5090 [Herz et al., arXiv:2512.23029].

The Asymptotic Numerical Stability theorem bounds a step's signal-to-perturbation ratio below, and its lemma the summed stored error entering a step above, independently of depth, where under full attention a needed token's weight falls as 1/N and the exposure grows linearly. What it changes is the question the KV quantization literature asks — not how small each token's error can be made inside the full-attention regime, but whether the regime can be left. Provenance selection leaves it, and §9.11.2 shows the premise holds: every planted turn selected from a history twelve times the working set, at the top compression level as at near-lossless.

On the system's own codebase, retrieval ranks an answer file first among 7,943 on 49% of 121 questions by sign agreement alone, and answering 128 questions from memory with tools forbidden, the memory is worth 37.0 points of recall, re-selection during reasoning 5.3, and continuous re-selection 6.9 over a single retrieval, each clear of zero after Holm correction; the gain falls on the questions that do not name their answer's file, the ones only the reasoning can reach. Attention is retrieval: this system runs it hierarchically, approximate over the unbounded context and exact over what it selects, preserving the mechanism RAG replaces and removing the scaling constraint on exact attention. The deeper lesson is methodological: constraints accepted as design requirements forced decisions more efficient in the dimension they bound, measured here on three cards from 16 GB to 72 GB, with a bound that carries to any card.

---

## References

[1] Liu, Z., Yuan, J., Jin, H., et al. (2024). **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.** ICML 2024. https://arxiv.org/abs/2402.02750

[2] Hooper, C., Kim, S., Mohammadzadeh, H., et al. (2024). **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.** NeurIPS 2024. https://arxiv.org/abs/2401.18079

[3] Su, Y., Zhou, Y., Qiu, Q., et al. (2025). **Accurate KV Cache Quantization with Outlier Tokens Tracing.** ACL 2025. https://aclanthology.org/2025.acl-long.631/

[4] Zandieh, A., et al. (2026). **TurboQuant: Online Vector Quantization for Quantized KV Cache in Large Language Models.** ICLR 2026. https://arxiv.org/abs/2504.19874

[5] TurboQuant community benchmarks on Qwen3. llama.cpp discussion #20969, 2026. https://github.com/ggml-org/llama.cpp/discussions/20969

[6] Kang, H., et al. (2024). **GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM.** arXiv:2403.05527. https://arxiv.org/abs/2403.05527

[7] Xu, F., Goyal, T., Choi, E. (2025). **RefreshKV: Updating Small KV Cache During Long-form Generation.** ACL 2025. https://aclanthology.org/2025.acl-long.1211/

[8] Geng, Y., Gao, Y., Wu, W., Liu, G., & Liu, J. (2026). **RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse.** ICML 2026, PMLR 306. https://arxiv.org/abs/2603.13289

[9] Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). **Efficient Streaming Language Models with Attention Sinks.** ICLR 2024. https://arxiv.org/abs/2309.17453

[10] Pan, S.J., Yuan, M. (2025). **KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization.** https://arxiv.org/abs/2502.04420

[11] Xia, H., et al. (2025). **Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost.** arXiv:2511.18643. https://arxiv.org/abs/2511.18643

[12] Tao, et al. (2024). **AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations.** COLING 2025. https://arxiv.org/abs/2410.13212

[13] Liu, D., et al. (2024). **RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval.** NeurIPS 2025. https://arxiv.org/abs/2409.10516

[14] Kamahori, K., Tang, T., Gu, Y., Zhu, K., & Kasikci, B. (2025). **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.** ICLR 2025. https://arxiv.org/abs/2402.07033

[15] DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference. DATE 2025. https://arxiv.org/html/2501.10375

[16] Chen, H., et al. (2025). **KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models.** SOSP 2025. https://doi.org/10.1145/3731569.3764843

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

[49] Niko1221 and contributors. (2026). **Strata.** Community RTX 5090 run, Strata 0.1.29 (d6708a4), IQ2_XS, MTP, median of 3, https://github.com/Niko1221/Strata/tree/main/bench/results/2026-09-30-community-rtx-5090; batching, RTX 5070 12 GB, https://github.com/Niko1221/Strata/blob/main/docs/BATCHING.md

[50] Yao, J., Li, H., Liu, Y., et al. (2025). **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion.** EuroSys 2025. https://arxiv.org/abs/2405.16444

[51] Hu, J., Huang, W., Wang, W., et al. (2025). **EPIC: Efficient Position-Independent Caching for Serving Large Language Models.** ICML 2025. https://arxiv.org/abs/2410.15332

[52] Ma, D., Wang, Y., et al. (2025). **Block-Attention for Efficient Prefilling.** ICLR 2025. https://arxiv.org/abs/2409.15355

[53] Yang, X., Chen, T., Chen, B. (2025). **APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding.** ICLR 2025. https://arxiv.org/abs/2502.05431

[17] and [19]–[23] are background reading and are not cited in the text.

---

## Appendix A: Public Release and Community Collaboration

**Source code:** https://github.com/john-sharratt/candle (public)

**Live system:** https://code.tokera.com — the deployed coding assistant over this repository, on the cluster that produced §9.11.1 and §9.13. It requires sign-in; access is by allowlist, with a queue for guests. The project site, https://www.tokera.com, carries v1 and this version.

**What is released.** The engine, and the raw measurement rows behind every figure — `docs/performance.md` and `docs/results/`, each naming its build, machine and test — with the tests that produce them; the §9.12 battery's questions, ground truth, harness, analysis and per-answer scores (`zend/examples/dependency_battery/`, `docs/results/battery_128/`, and the pilot's `docs/results/dependency_battery_2026-11-*.md`); and the tests behind §9.6.8 and §9.11.2. Not released: the live substrate, a KV archive valid only for its model and holding conversations that are not this paper's to publish, and a view of which signatures matched at each reprojection. A turn's `projection`, `seed` and reprojection are dials of the request, so every arm of §9.12 is a setting a caller can choose.

**Versions.** v1 (13 May 2026, [10.5281/zenodo.20156060](https://zenodo.org/records/20156060)) is archived with an erratum at `docs/unbounded_agents_v1.md`; its "509 t/s" and "2,446 t/s" headlines were prefill rates and are withdrawn. v2 adds a measured evaluation across three machines and twelve models from 0.5B to 284B parameters (§9.6), every number produced by a test in this repository and recorded in `docs/performance.md`, and runs the retrieval evaluation on the production system (§9.11–§9.12). §4.1–§4.3 describe the v1 design on Qwen3-30B-A3B; provenance as production runs it is §6.

**Invitation.** v2 is the final version of this paper, and critical review of it is welcome. The v1 erratum stays part of the record; every figure here names its build, machine and test. Validation, stress tests and optimisations are welcome through the GitHub repository.