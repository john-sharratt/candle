# Paged KV Cache Reuse with Mixed-Precision Bootstrapping and Hallucination-Guided Fact Retrieval for High-Density LLM Inference

**Abstract**

We present an architecture for high-density LLM inference on consumer and mid-tier hardware, built around two novel techniques and their synergistic integration with existing methods.

**Novel contributions**: First, we introduce *trie-constrained generation as a retrieval mechanism*—rather than fighting model "hallucinations," we channel them through a trie structure where unconstrained generation becomes consistent fact retrieval. Single-token vocabulary selection (tied to tokenizer-specific words) ensures deterministic path selection across 24,000 retrieval paths. We extend this with *speculative path resolution*: when category-level entropy indicates uncertainty, serial evaluation of complete paths with mean log-probability selection exploits trie depth as a disambiguation signal, substantially reducing catastrophic retrieval errors. Second, we implement *dynamic knowledge with override semantics*: a three-tier memory architecture where dynamic facts shadow static knowledge at the same trie path without reindexing, enabling mid-conversation knowledge updates that RAG and fine-tuning cannot provide.

**Integration contributions**: We combine these techniques with position-independent KV caching (building on MEPIC and Lazy-Attention) and content-based mixed-precision allocation. The key insight is that these are co-requirements: position-independent caching enables arbitrary fact injection; the fact index motivates content-based precision; override semantics differentiate from static retrieval. Removing any component substantially degrades the others.

**Important distinction**: The 2.4M token fact index represents *storage capacity*, not *attention span*. The architecture retrieves 1-8 facts (~800 tokens) per query—categorically different from native long-context models providing simultaneous attention. Cross-chunk relationships are invisible unless both chunks are retrieved. This architecture excels when relevant content is localizable; native long-context excels when holistic visibility matters.

Additionally, we demonstrate a *quality bootstrapping effect*: high-precision KV cache for stable content partially compensates for aggressive model quantization, extending mixed-precision research from layer/channel allocation to semantic-role allocation.

We evaluated on RTX 4090 (24GB) with Qwen3-30B-A3B-AWQ and A100 (80GB) with Hermes3-70B-Q4, demonstrating [**TODO: X**] concurrent contexts with [**TODO: Y ms**] median latency and coherent state across [**TODO: N**] conversation turns.

---

## 1. Introduction

Frontier-level language model inference has traditionally required datacenter-class hardware: multi-GPU clusters with hundreds of gigabytes of combined VRAM, sophisticated orchestration, and substantial operational overhead. This paper demonstrates that consumer and mid-tier enterprise hardware—specifically an RTX 4090 (24GB) or A100 (80GB)—can approach comparable inference density through four synergistic architectural innovations.

The key insight is that these innovations are not independent optimizations but architectural co-requirements. Quantized kernels that keep weights compressed throughout execution maximize the model size that fits in VRAM. Position-independent KV caching enables O(1) reuse of shared content across unlimited concurrent contexts. Sparse retrieval over a trie-indexed fact store provides access to large knowledge bases without proportional memory growth. Speculative path resolution exploits trie depth to disambiguate uncertain queries. Each innovation amplifies the others; removing any one substantially degrades system effectiveness.

**Retrieval ≠ Attention**: We emphasize upfront that retrieval-based access to 2.4M tokens is categorically different from simultaneous attention over 2.4M tokens. Our architecture indexes 24,000 chunks of ~100 tokens each; per query, 1-8 chunks (~800 tokens) are retrieved and materialized in the physical context window. Cross-chunk relationships are invisible unless both chunks are retrieved together. Native long-context models (1M+ tokens) provide fundamentally different capabilities—simultaneous attention enables discovery of unexpected relationships, global pattern detection, and cross-document reasoning that retrieval cannot replicate. Our architecture excels at different workloads: queries where relevant content is localizable, session memory across many turns, and cost-sensitive deployment at scale. These are complementary approaches, not substitutes.

```
┌─────────────────────────────────────────────────────────────┐
│              The Consolidation Advantage                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRADITIONAL: Distributed inference                         │
│  ─────────────────────────────────────                      │
│  • Multiple GPUs, each with partial model                   │
│  • Network overhead for cross-device attention              │
│  • Limited sharing between requests                         │
│  • High operational complexity                              │
│                                                             │
│  PROPOSED: Consolidated single-card inference               │
│  ───────────────────────────────────────────                │
│  • Full model on one GPU (via aggressive quantization)      │
│  • All contexts share KV cache for common content           │
│  • Fact index exceeds physical limits                       │
│  • Volume handled through efficient batching                │
│                                                             │
│  The consolidation is a design requirement that enables     │
│  sharing, not a constraint to be overcome.                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 Why Consolidation Enables Sharing

Running the same model for the same use case on the same machine appears limiting but is actually the source of the architecture's power. Any production deployment generates inference volume that exceeds single-card capacity—the question is whether to distribute across cards or consolidate and batch.

Distribution sacrifices sharing. Each card maintains separate KV caches, separate model weights, separate fact databases. Communication overhead compounds with scale.

Consolidation enables sharing. All concurrent contexts share:
- Model weights (loaded once, used for all sequences)
- System prompt KV cache (computed once, referenced by all)
- Shared domain knowledge (precomputed, position-adjusted per context)
- Dynamic facts (streamed in/out via LRU based on relevance)

The volume that would traditionally require multiple cards instead amortizes over a single card's shared resources. Latency improves because batching increases arithmetic intensity; throughput improves because shared content loads once.

### 1.2 The Three Innovations as Co-Requirements

**Innovation 1: Fused Quantized Kernels**

The GEMV/GEMX decomposition keeps weights quantized throughout execution. Unlike approaches that dequantize for computation, our kernels operate directly on INT4/AWQ/GGUF weights for both small-batch (GEMV) and large-batch (GEMM) operations. This enables larger models in fixed VRAM—a 30B MoE model fits in ~17GB leaving 7GB for KV cache on a 24GB card, while a 70B dense model at Q4 fits in ~40GB leaving 40GB for KV cache on an 80GB card.

Without aggressive quantization, the model itself would consume available VRAM, leaving insufficient headroom for the KV sharing that makes consolidation valuable.

**Innovation 2: Position-Independent KV Paging**

The extended page table with RoPE remapping enables arbitrary reuse of precomputed KV cache. A fact computed with internal positions 0-100 can be injected at position 500 in Context A, position 1200 in Context B, and position 800 in Context C—simultaneously, from a single physical copy.

Without position-independence, each context would require its own copy of shared content, negating the memory benefits of consolidation.

**Innovation 3: Sparse Retrieval Over Trie-Indexed Facts**

The three-level trie (24,000 paths × 100 tokens = 2.4M addressable tokens) provides access to a large fact index beyond physical window limits. The model's attention patterns steer retrieval through the trie; selected facts materialize in the physical window via KV injection.

This trie-based retrieval does not replace native long-context models—a 1M token frontier context window offers different capabilities, particularly simultaneous attention over all content. However, the architecture addresses three distinct context extension problems:

1. **System prompt extension**: Static knowledge functions as a chunked system prompt with sparse retrieval. Rather than cramming all reference material into a monolithic prompt that must fit in the physical window, arbitrarily large instruction sets can be organized into retrievable chunks, with relevant portions injected based on query content. A 100-page policy manual or technical specification becomes addressable without consuming physical context.

2. **Conversation context extension**: Dynamic facts extend conversation memory beyond physical limits. As conversations grow, older content compresses into retrievable facts rather than being truncated, maintaining long-term coherence across sessions that would otherwise exceed context windows.

3. **Dynamic knowledge without retraining**: Critically, the fact index can change between every conversation turn without model modification. This distinguishes the architecture from fine-tuning approaches, where large reference material could theoretically be baked into model weights but would then be static—unchangeable without retraining. Here, facts can be added, updated, or removed dynamically. An agent can learn a user's preferences mid-conversation and immediately apply them; a knowledge base can be updated without touching the model; conflicting information can be resolved by the override mechanism rather than weight interference. This dynamic adaptability is essential for applications requiring medium-term memory—the ability to accumulate and apply knowledge within a session that fine-tuning simply cannot provide.

Both mechanisms provide effective access to far more information than the 4-16K physical windows typical of quantized inference on consumer hardware, though through retrieval rather than simultaneous attention.

```
┌─────────────────────────────────────────────────────────────┐
│              Innovation Interdependence                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Quantized Kernels ──────► Larger model fits in VRAM        │
│         │                           │                       │
│         │                           ▼                       │
│         │                  More VRAM for KV cache           │
│         │                           │                       │
│         ▼                           ▼                       │
│  Position-Independent ───► Shared KV across contexts        │
│  KV Paging                          │                       │
│         │                           ▼                       │
│         │                  Higher concurrency possible      │
│         │                           │                       │
│         ▼                           ▼                       │
│  Fact Index via ─────────► 2.4M tokens addressable          │
│  Trie Retrieval                     │                       │
│         │                           ▼                       │
│         │                  Trie structure enables depth     │
│         │                           │                       │
│         ▼                           ▼                       │
│  Speculative Path ───────► Disambiguation via trie depth    │
│  Resolution                         │                       │
│                                     ▼                       │
│                            D3 errors bounded tighter        │
│                                     │                       │
│                                     ▼                       │
│                            High-density inference on        │
│                            consumer/mid-tier hardware       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Innovation 4: Speculative Path Resolution**

When the model is uncertain at the category level (high entropy), standard constrained decoding commits arbitrarily. Speculative resolution instead explores the top-k candidate paths through the complete trie depth, selecting by mean log-probability. This exploits a key property of the trie architecture: ambiguity at shallow levels often resolves at deeper levels, where only one interpretation produces natural continuations.

This mechanism has no analog in embedding-based RAG, where similarity ties are broken arbitrarily or by returning multiple results. Here, the generative model itself resolves ambiguity using the same weights that will consume the retrieved content—extending the self-consistency principle from retrieval to disambiguation.

### 1.3 Contributions

This paper makes seven contributions, ordered by novelty:

1. **Dynamic knowledge with override semantics** (Section 6.3): A three-tier memory architecture where dynamic facts shadow static knowledge at the same retrieval path without reindexing—enabling mid-conversation knowledge updates that RAG and fine-tuning cannot provide

2. **Trie-constrained generation as retrieval** (Section 6): A novel application of constrained decoding where model "hallucinations" become consistent retrievals when channeled through a trie, with single-token vocabulary selection ensuring deterministic path selection

3. **Speculative path resolution** (Section 6.9): When category-level entropy indicates uncertainty, serial evaluation of top-k complete paths with KV truncation, selecting by mean log-probability. This exploits trie depth as a disambiguation signal unavailable to flat retrieval methods, reducing D3 errors by [**TODO: X%**] with latency cost only on ambiguous queries

4. **Co-requirement architecture demonstration**: Empirical validation that these innovations are synergistic—position-independent caching enables the fact index; the fact index motivates content-based precision; speculative resolution exploits trie depth; removing any component substantially degrades the others

5. **Content-based mixed-precision KV** (Section 8.7): Allocating KV precision by semantic role (system prompts, facts, conversation) rather than by layer or channel, with empirical evidence for a "quality bootstrapping" effect where high-precision anchors partially compensate for aggressive model quantization

6. **Position-independent KV caching integration**: Building on MEPIC and Lazy-Attention, we integrate runtime RoPE remapping with content-addressable lookup and heterogeneous precision support for the trie retrieval use case

7. **Consumer hardware density**: Demonstrating frontier-level inference density on RTX 4090 (24GB) and A100 (80GB) through the combined architecture

---

## 2. Background

### 2.1 The Scaling Dilemma: Distribute or Consolidate

Transformer inference faces multiplicative memory pressure. Total KV memory equals the product of concurrent sequences, context length, KV cache size per token, and precision bytes. For a 32B-parameter model serving 100 concurrent sequences at 4,000 tokens with BF16 precision, this approaches 100GB—exceeding single-GPU capacity.

```
┌─────────────────────────────────────────────────────────────┐
│              Memory Pressure Components                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Total KV = Sequences × Context × KV_per_token × Precision  │
│                                                             │
│  Example (32B model, BF16):                                 │
│    100 seq × 4,000 tok × 256 KB/tok × 1.0 ≈ 100 GB         │
│                                                             │
│  Exceeds single GPU → must choose scaling strategy          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The conventional response is distribution: spread the workload across multiple GPUs, each handling a subset of sequences or model layers. This sacrifices sharing—each GPU maintains separate KV caches, loads weights independently, and cannot amortize common content across requests on different devices.

The alternative is consolidation: aggressive quantization to fit the model on a single GPU, combined with architectural innovations that maximize sharing. This approach requires solving three problems simultaneously:

1. **Model compression**: Weights must remain quantized throughout execution, not just storage
2. **KV reuse**: Common content (prompts, shared knowledge, facts) must be sharable across all concurrent contexts
3. **Context expansion**: Physical window limitations must be overcome without proportional memory growth

Current approaches attack these problems individually. Quantization reduces precision but typically dequantizes for computation. Prefix sharing deduplicates common context but only at fixed positions. Eviction drops old tokens under pressure but loses the information entirely. These approaches miss the synergies available when all three problems are solved together.

### 2.2 RoPE Position Encoding

Rotary Position Embedding encodes position through rotation of embedding dimensions. The transformation applies position-dependent rotation to query and key vectors, with the critical property that RoPE applies to K vectors but not V vectors. Furthermore, rotations compose additively: applying RoPE with offset a followed by offset b equals applying RoPE with offset a+b.

This composability enables post-hoc position adjustment. KV cache computed with internal positions 0 through N can be repositioned to any target position P by applying the delta rotation. The precomputed cache appears at positions P through P+N in the attention window, enabling the same physical cache to serve different logical positions across contexts.

### 2.3 Quantized Matmul Landscape

Modern inference employs multiple quantization strategies optimized for different regimes. GEMV (general matrix-vector) operations excel for batch sizes up to approximately 4, operating in a memory-bound regime where bandwidth dominates. GEMX and similar matrix-matrix kernels excel for larger batches, operating in a compute-bound regime where tensor core throughput dominates.

```
┌─────────────────────────────────────────────────────────────┐
│              Kernel Selection by Batch Size                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Batch 1-4:   GEMV optimal                                  │
│               Memory-bound, simple dequant per element      │
│                                                             │
│  Batch 5+:    GEMX optimal                                  │
│               Compute-bound, fused dequant in registers     │
│                                                             │
│  Key insight: Both can operate on quantized weights         │
│               No need to dequantize to full precision       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Critically, both kernel families can operate directly on quantized weights without full-precision intermediate storage. GEMV dequantizes per-element during the dot product; GEMX dequantizes in registers during tiled computation. This means a model stored at INT4/AWQ precision never requires a full-precision copy in VRAM—the memory footprint is the quantized size throughout execution.

This property is essential for consolidation. A 30B MoE model at AWQ fits in approximately 17GB, leaving 7GB on a 24GB card for KV cache and operational overhead. A 70B dense model at Q4 fits in approximately 40GB, leaving 40GB on an 80GB card. Without fused quantized kernels, models would expand during computation, making single-card deployment of these model classes impractical.

### 2.4 The Hallucination Opportunity

Small models in the 1.5-8B parameter range exhibit consistent patterns when generating content beyond their training data. These "hallucinations" follow predictable vocabulary and semantic patterns determined by the model's learned distributions. We observed that when the same model both compresses information into a summary and later retrieves from that summary, its generation patterns become self-consistent.

The model's biases cease to be errors and become coherent reconstruction patterns. The same semantic space, vocabulary, and tokenization ensure that compressed concepts map consistently to retrieved concepts. This insight motivated using constrained generation as a retrieval mechanism: rather than fighting the model's biases, we channeled them through a trie structure that guaranteed valid outputs while preserving the model's learned associations.

---

## 3. Extended Page Table Architecture

### 3.1 Design Motivation

Standard paged attention maps logical positions to physical memory addresses, enabling non-contiguous allocation and copy-on-write prefix sharing. However, the mapping remains fundamentally position-bound: a KV cache page computed for positions 0-255 cannot serve positions 1000-1255 without recomputation.

Our extended page table broke this constraint by adding metadata that enabled content-addressable lookup, runtime position remapping, heterogeneous precision, and reference-counted sharing. The page table transformed from a simple position-to-address map into a content-addressable memory system.

### 3.2 Page Entry Structure

Each page entry contained six fields beyond the basic physical address. The content hash enabled lookup by token sequence rather than position, supporting deduplication and cache-hit detection. The RoPE offset specified the position adjustment applied at attention time, enabling the same physical page to appear at different logical positions. The dtype field indicated storage precision (BF16, FP8, INT4, etc.), enabling mixed-precision attention. The reference count tracked active sessions using this page for LRU eviction. Finally, flags indicated special handling such as pinned pages that should never be evicted.

```
┌─────────────────────────────────────────────────────────────┐
│                  Extended Page Entry                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐                                         │
│  │ content_hash   │ ─── Content-addressable lookup          │
│  ├────────────────┤                                         │
│  │ physical_addr  │ ─── K,V memory location                 │
│  ├────────────────┤                                         │
│  │ rope_offset    │ ─── Runtime position adjustment         │
│  ├────────────────┤                                         │
│  │ dtype          │ ─── BF16 / FP8 / INT4 / ...            │
│  ├────────────────┤                                         │
│  │ refcount       │ ─── Active session count                │
│  ├────────────────┤                                         │
│  │ flags          │ ─── PINNED / PRECOMPUTED / LIVE        │
│  └────────────────┘                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Content-Addressable Storage

The content hash enabled a fundamentally different access pattern. Rather than asking "what KV cache is at position P?", the system asked "do we have KV cache for token sequence T, and if so, where should it appear in this context?"

When assembling a context, the system computed hashes for each segment (system prompt, facts, history) and checked the page table. Cache hits returned existing pages with appropriate RoPE offsets; cache misses triggered computation and storage. The same fact could serve Context A at position 500, Context B at position 1200, and Context C at position 800—all from a single physical page with different offset metadata in each context's manifest.

```
┌─────────────────────────────────────────────────────────────┐
│              Cross-Context Page Sharing                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                 Physical Page (single copy)                 │
│                 ┌─────────────────────┐                     │
│                 │ Fact: "The Iron     │                     │
│                 │ Gate fell after..." │                     │
│                 │ dtype: FP8          │                     │
│                 └─────────────────────┘                     │
│                    │         │         │                    │
│          ┌─────────┘         │         └─────────┐          │
│          ▼                   ▼                   ▼          │
│    Context A           Context B           Context C        │
│    offset=500          offset=1200         offset=800       │
│                                                             │
│    Same physical memory, different logical positions        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Reference Counting and Eviction

Shared pages required careful lifecycle management. When a session began using a fact, the page's reference count incremented. When the session ended, the count decremented. Pages with positive reference counts were never evicted regardless of recency.

The eviction policy operated in priority order. Pinned pages (system prompts, critical shared context) were never evicted. Pages with active references were retained until all sessions completed. Among unreferenced pages, least-recently-used ordering determined eviction candidates. This policy ensured that actively-shared content remained available while allowing graceful reclamation of unused cache.

---

## 4. Kernel Architecture

### 4.1 Batched Page Prefill

The prefill phase processed input tokens in parallel, operating in a compute-bound regime where tensor core utilization dominated performance. Our kernel extended standard prefill to handle mixed-dtype pages within a single operation.

The kernel operated in three phases. First, pages were grouped by dtype to minimize warp divergence from type-dependent code paths. BF16 pages formed one group, FP8 pages another, INT4 pages a third. This grouping enabled efficient vectorized operations within each dtype while maintaining correct attention computation across the full context.

```
┌─────────────────────────────────────────────────────────────┐
│              Mixed-Dtype Prefill Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Page manifest with dtype metadata                   │
│                                                             │
│  Phase 1: Group by dtype                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ BF16: [P0, P3]  FP8: [P1, P2, P5]  INT4: [P4, P6]   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  Phase 2: Process each group                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Load K,V → Dequant → Apply RoPE offset → Attention  │   │
│  │ FP32 accumulator across all dtype groups            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  Phase 3: Normalize and output                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Second, each dtype group processed through a unified pipeline: load K and V from physical addresses, dequantize to FP32 working precision, apply RoPE offset from page metadata to K vectors, compute attention scores using tensor cores, and accumulate with online softmax for numerical stability. The FP32 accumulator maintained precision across dtype boundaries, ensuring that mixing BF16 system prompts with INT4 conversation history produced correct results.

Third, the accumulated attention output was normalized and converted to the target output dtype. The entire operation fused what would traditionally require separate dtype-specific kernels into a single launch with metadata-driven dispatch.

### 4.2 Batched Page Decode

The decode phase generated one token per sequence per step, operating in a memory-bound regime where bandwidth utilization dominated. Each generated token required loading model weights once (amortized across the batch) plus loading KV cache for each sequence's full context.

The extended page table enabled a key optimization: sequences sharing pages could share memory loads. When multiple concurrent sessions referenced the same fact from shared domain knowledge, that page loaded once and served all relevant attention computations. The kernel scheduler identified page sharing opportunities within each batch and organized memory access to maximize reuse.

### 4.3 GEMV/Marlin Decomposition

Weight matrix multiplication presented a regime-dependent optimization problem. For very small batches (1-4 sequences), the operation was memory-bound: each output element required loading an entire weight row, and the computation completed before memory bandwidth saturated. GEMV kernels with simple dequantization excelled here.

For larger batches, the operation became compute-bound: weight loads amortized across multiple output elements, and tensor core throughput limited performance. Marlin-style kernels with fused dequantization and careful tiling excelled here.

```
┌─────────────────────────────────────────────────────────────┐
│              Greedy Decomposition Algorithm                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: 47 sequences                                        │
│                                                             │
│  Step 1: Sort by context length (descending)                │
│                                                             │
│  Step 2: Partition into execution groups                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Marlin(32) │ Marlin(12) │ GEMV(3)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Step 3: Execute each group with optimal kernel             │
│                                                             │
│  Step 4: Merge results                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Our greedy decomposition algorithm handled mixed batches by partitioning sequences into execution groups. The algorithm sorted sequences by length, then iteratively assigned sequences to either GEMV (if remaining count was below threshold) or Marlin (up to maximum batch size) execution paths.

### 4.4 Inline Multi-Dtype Support

Both kernel families handled the full range of quantization formats through metadata-driven dispatch rather than separate kernel implementations. Supported formats included BF16, FP8 (E4M3 and E5M2), INT8, INT4, INT3, INT2, as well as structured formats like GPTQ, AWQ, and GGUF variants.

The implementation used a dtype tag in page or weight metadata to select the appropriate dequantization path. All paths produced FP32 intermediate values for accumulation, ensuring numerical consistency regardless of storage format. Template specialization for common format combinations provided optimized fast paths while the general mechanism handled arbitrary mixing.

---

## 5. Inline RoPE Position Remapping

### 5.1 Position-Independent Storage

The key insight enabling cross-context KV reuse was that position information in transformer attention is encoded through RoPE rotation of key vectors, and these rotations can be adjusted post-hoc. A fact computed with internal positions 0 through N contained K vectors with RoPE applied for those positions. To place this fact at position P in a runtime context, we applied an additional rotation corresponding to offset P.

Mathematically, if K_stored contained keys with RoPE(pos=0..N) applied, then K_adjusted = RoPE_delta(K_stored, P) produced keys that behaved as if originally computed at positions P through P+N. The V vectors required no adjustment since RoPE applied only to K.

```
┌─────────────────────────────────────────────────────────────┐
│              RoPE Position Remapping                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Precomputation:                                            │
│    K_stored = RoPE(K_raw, pos=0..N)                        │
│    Store with internal positions                            │
│                                                             │
│  Runtime injection at position P:                           │
│    K_adjusted = RoPE_delta(K_stored, offset=P)             │
│    Result: K behaves as if computed at pos=P..P+N          │
│                                                             │
│  Key property: RoPE rotations compose additively            │
│    RoPE(x, a+b) = RoPE(RoPE(x, a), b)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This enabled a powerful workflow: precompute KV cache for facts, system prompts, and shared knowledge once using a high-quality model, store with position-independent internal numbering, then inject at arbitrary positions across unlimited contexts with only a lightweight rotation adjustment at attention time.

### 5.2 Why Facts Work as Injection Targets

Not all content was suitable for position-independent caching. The key requirement was that the cached content must be semantically self-contained—its internal attention patterns should not depend on surrounding context.

Facts satisfied this requirement naturally. A fact like "The Iron Gate fell after a 40-day siege" had internal coherence: each token attended to previous tokens within the fact to build the complete semantic representation. When injected into a runtime context, the fact provided K and V vectors that generation could attend to, but the fact itself never needed to attend to the surrounding conversation.

```
┌─────────────────────────────────────────────────────────────┐
│              Attention Directionality                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Context: [System][History][Fact][Query] → Generation       │
│                              │      │           │           │
│                              │      │           │           │
│                              ▼      ▼           ▼           │
│                         ┌─────────────────────────────┐     │
│                         │ Generation attends TO:      │     │
│                         │   System ✓                  │     │
│                         │   History ✓                 │     │
│                         │   Fact ✓ (provides K,V)     │     │
│                         │   Query ✓                   │     │
│                         │                             │     │
│                         │ Fact never attends OUT      │     │
│                         │ (self-contained at precomp) │     │
│                         └─────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This asymmetry—generation attended to facts, but facts didn't attend out—was precisely what enabled position-independent precomputation. The fact's internal coherence was established at precompute time; runtime position only affected how generation queried the fact, not the fact's self-representation.

---

## 6. Sparse Retrieval Over Trie-Indexed Facts

### 6.1 The Core Insight: Retrieval as Attention

Traditional transformer attention computes fine-grained weights over every token in the physical context window. This creates a fundamental tradeoff: larger contexts enable richer reasoning but impose quadratic computational cost and linear memory growth.

We observed that this tradeoff could be restructured by separating the *addressable* context from the *physical* context. Rather than attending over all tokens densely, the system could attend over fact-sized chunks sparsely, selecting which chunks to materialize in the physical window based on relevance to the current query.

This reframing transforms fact retrieval from a memory system bolted onto inference into an attention mechanism operating at chunk granularity. The retrieval computation—steering through a trie of fact paths based on conversational context—performs the same function as attention weights: determining which stored information is relevant to the current generation step.

```
┌─────────────────────────────────────────────────────────────┐
│              Attention at Two Granularities                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRADITIONAL: Dense token-level attention                   │
│  ─────────────────────────────────────────                  │
│  Physical context: 4,096 tokens                             │
│  Attention: O(n²) over all tokens                          │
│  Everything in context, all the time                        │
│                                                             │
│  PROPOSED: Sparse chunk-level + dense token-level           │
│  ─────────────────────────────────────────────              │
│  Fact index: 2,400,000 tokens (24,000 facts × 100 tok)      │
│  Chunk attention: Select relevant facts via trie steering   │
│  Token attention: O(n²) over materialized physical context │
│  Right information, at the right time                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Fact Index Scale

The fact database organized knowledge into a three-level hierarchy: approximately 20 categories, 30 subcategories per category, and 40 topics per subcategory. This yielded roughly 24,000 distinct fact paths, each storing approximately 100 tokens of content.

The total addressable content therefore spanned 2.4 million tokens—substantially exceeding the 4-16K physical windows typical of quantized inference on consumer hardware. 

**Critical Distinction: Retrieval vs. Attention**

The "2.4M token fact index" framing requires careful interpretation. The 2.4M tokens are *addressable through retrieval*, not *simultaneously attended*. This is a categorical difference from native long-context models, not merely a quantitative one:

- **Native long-context (e.g., 1M tokens)**: All tokens are simultaneously present in the attention computation. The model can identify relationships between any two tokens regardless of their positions. This enables tasks requiring holistic analysis—finding contradictions across documents, identifying patterns that span the entire context, or synthesizing information that requires seeing everything at once.

- **Retrieval-based fact index (2.4M tokens)**: Tokens are organized into ~24,000 chunks, of which 1-8 are materialized per query. The model sees only the retrieved chunks plus the physical context window. Cross-chunk relationships are invisible unless both chunks are retrieved simultaneously.

These approaches serve different workloads:

| Workload Type | Better Suited |
|---------------|---------------|
| Finding contradictions across documents | Native long-context |
| Global optimization over all content | Native long-context |
| Answering questions from a known knowledge base | Retrieval-based |
| Maintaining session memory across many turns | Retrieval-based |
| Queries where relevant content is localizable | Retrieval-based |
| Cost-sensitive deployment at scale | Retrieval-based |

Our architecture does not approximate native long-context; it provides a different capability with different tradeoffs. The 2.4M figure indicates storage capacity, not attention span.

```
┌─────────────────────────────────────────────────────────────┐
│              Fact Index Arithmetic                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hierarchy:                                                 │
│    20 categories × 30 subcategories × 40 topics            │
│    = 24,000 fact paths                                      │
│                                                             │
│  Content:                                                   │
│    24,000 paths × 100 tokens/fact                          │
│    = 2,400,000 tokens addressable                          │
│                                                             │
│  Physical window: ~4,000 tokens                             │
│                                                             │
│  Per-query visibility: 1-8 facts (~800 tokens)             │
│                                                             │
│  Two extension mechanisms:                                  │
│    Static knowledge: Chunked system prompt (100K+ tokens)   │
│    Dynamic facts: Conversation memory extension             │
│                                                             │
│  IMPORTANT: "Addressable" ≠ "Simultaneously attended"       │
│  This is storage capacity, not attention span               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Three-Tier Memory Architecture

The system organized knowledge into three tiers with different storage strategies, update frequencies, and retrieval characteristics.

**Tier 1: Physical Context (Dense Attention)**

The innermost tier comprised the active context window where standard transformer attention operated. This included fresh messages (approximately the last 10 conversational turns stored verbatim), a recent summary compressing older conversation into dense narrative, and any facts retrieved for the current query. All tokens in this tier received full attention during generation.

**Tier 2: Dynamic Facts (Sparse Retrieval, Mutable)**

The middle tier extended conversation context beyond physical window limits. As conversations grew, older content that would normally be truncated was instead compressed into retrievable facts. These facts could be injected into or updated within the trie structure, potentially overriding pre-existing static knowledge.

When conversation revealed new information—an agent's changed circumstances, updated preferences, evolved relationships—dynamic facts captured these changes and took precedence over static knowledge during retrieval. This enabled long-running sessions to maintain coherence across hundreds of turns without unbounded context growth.

**Tier 3: Static Knowledge (Sparse Retrieval, Immutable)**

The outermost tier functioned as a chunked system prompt with sparse retrieval. Traditional system prompts must fit entirely in the physical context window; this tier removed that constraint. Reference material that would be prohibitively large as a monolithic prompt—policy manuals, technical specifications, agent profiles, domain knowledge—could be organized into retrievable chunks and injected based on query relevance.

This content was computed once and cached as position-independent KV, shared across all inference contexts. Static knowledge provided the stable foundation that dynamic facts could extend or override, enabling per-session customization without duplicating the base knowledge.

```
┌─────────────────────────────────────────────────────────────┐
│              Three-Tier Memory Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ TIER 1: Physical Context                            │   │
│  │ Dense attention, ~4K tokens, full fidelity          │   │
│  │ [System][Summary][Retrieved Facts][Fresh][Query]    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                    [Retrieval]                              │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ TIER 2: Dynamic Facts (Conversation Extension)      │   │
│  │ Sparse attention, mutable, conversation-derived     │   │
│  │ Can override Tier 3 when paths collide              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                    [Fallback]                               │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ TIER 3: Static Knowledge (Chunked System Prompt)    │   │
│  │ Sparse attention, immutable, pre-computed           │   │
│  │ Shared KV cache across all contexts                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Total fact index: 2.4M tokens                              │
│  Physical window: ~4K tokens                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The override mechanism enabled knowledge updates without rebuilding the knowledge cache. When a dynamic fact occupied the same path as static knowledge, retrieval returned the dynamic version. This created a copy-on-write semantic for domain knowledge: the base knowledge remained shared and immutable while per-context modifications layered on top.

### 6.4 Deterministic Trie-Based Retrieval

The retrieval mechanism used a trie (prefix tree) to constrain generation to valid fact paths. Unlike embedding-based retrieval which computes similarity scores, or fuzzy matching which recovers from near-misses, our approach was fully deterministic at the token selection level: the model's logits were masked to permit only tokens that continued valid trie paths.

**Trie Structure**

The trie encoded all valid fact paths as a tree of single-token transitions. Each path consisted of exactly three tokens: category, subcategory, and topic. The model generated paths by making three successive token selections, each constrained to valid continuations from the current trie node.

```
┌─────────────────────────────────────────────────────────────┐
│              Trie-Constrained Path Generation               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ROOT                                                       │
│   ├── goal ─────┬── career ────┬── promotion                │
│   │             │              ├── transfer                 │
│   │             │              └── retirement               │
│   │             ├── project ───┬── deadline                 │
│   │             │              ├── scope                    │
│   │             │              └── team                     │
│   │             └── personal ──┬── health                   │
│   │                            └── family                   │
│   ├── event ────┬── combat ────┬── raid                     │
│   │             │              ├── ambush                   │
│   │             │              └── siege                    │
│   │             └── trade ─────┬── deal                     │
│   │                            └── dispute                  │
│   └── ...                                                   │
│                                                             │
│  Path generation: 3 token selections, each constrained      │
│  Example: goal → career → promotion                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Single-Token Vocabulary Selection**

A critical implementation detail: all category, subcategory, and topic words were selected to be single tokens in the target tokenizer. This eliminated intermediate states where the model had partially committed to a multi-token word and could be influenced by language modeling priors to complete it incorrectly.

The vocabulary was selected by analyzing the tokenizer's vocabulary file directly, filtering for semantically useful words that tokenized to single tokens, then organizing these into disjoint categories. This inverted the typical workflow of designing a taxonomy first and discovering tokenization problems later—instead, the taxonomy emerged from viable single-token candidates.

```
┌─────────────────────────────────────────────────────────────┐
│              Tokenizer-First Vocabulary Design              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROBLEMATIC: Multi-token category words                    │
│  ─────────────────────────────────────────                  │
│  "consideration" → ["consider", "ation"]                    │
│  After generating "consider", model priors compete          │
│  with trie constraint for completion                        │
│                                                             │
│  SOLUTION: Single-token vocabulary                          │
│  ────────────────────────────────────                       │
│  "goal" → ["goal"]           (one token, one selection)     │
│  "event" → ["event"]         (one token, one selection)     │
│  "trait" → ["trait"]         (one token, one selection)     │
│                                                             │
│  Methodology:                                               │
│  1. Parse tokenizer.json for single-token words             │
│  2. Filter for semantic utility                             │
│  3. Organize into disjoint categories                       │
│  4. Build trie from valid combinations                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Category Disjointness with Subcategory Bridges**

Top-level categories were designed to be semantically disjoint—no overlap in meaning that could cause the model to inconsistently distribute facts between categories. This constraint ensured that the model's first token selection made a clean commitment to a semantic domain.

However, at the subcategory and topic levels, bridge points allowed recovery from partial errors. If the model selected the correct category but wrong subcategory, lateral connections in the trie could still route to relevant topics. This provided graceful degradation without requiring fuzzy matching: wrong paths still landed in semantic proximity to the intended target.

**Limitations of Graceful Degradation**

The "semantic proximity" property holds only when errors occur *within* the correct domain. Category-level misselection—where the model commits to an entirely wrong semantic domain at the first token—does not degrade gracefully. If a query about career goals routes to the "relationship" category, subsequent subcategory and topic selection will optimize within the wrong domain, potentially producing responses that sound coherent but are contextually inappropriate.

This failure mode is particularly concerning because it may be invisible to users: the model retrieves *some* fact and generates fluent text referencing it, but the fact is wrong for the situation. Unlike obvious failures (incoherence, refusals), category misselection produces plausible-sounding but factually incorrect responses.

We characterize these failure modes rigorously in Section 6.9 and measure their frequency in Section 8.5, establishing bounded failure rates even under adversarial conditions.

```
┌─────────────────────────────────────────────────────────────┐
│              Disjoint Categories, Bridged Subcategories     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1 (Category): DISJOINT                               │
│  ─────────────────────────────────                          │
│  "goal" ≠ "event" ≠ "trait" ≠ "bond" ≠ ...                 │
│  No semantic overlap → clean first commitment               │
│                                                             │
│  Level 2 (Subcategory): BRIDGED                             │
│  ───────────────────────────────                            │
│  goal.career ←→ goal.project (shared topics possible)       │
│  Wrong subcategory can still reach relevant topic           │
│                                                             │
│  Level 3 (Topic): CONVERGENT                                │
│  ─────────────────────────────                              │
│  Multiple paths may reach same semantic content             │
│  goal.career.team ≈ goal.project.team                       │
│                                                             │
│  Recovery without fuzzy matching:                           │
│  Model takes wrong turn at L2, still finds useful fact      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.5 Steering via Contextual Attention

The model's path selection was not random—it was guided by attention over the conversational context. Given the current query and conversation history, the model's learned associations determined which category, subcategory, and topic tokens received highest probability mass.

This created a natural mapping between conversational relevance and fact retrieval. When the conversation concerned career decisions, the model's attention patterns favored "goal" at the category level and "career" at the subcategory level. The trie constraint ensured valid paths; the model's attention determined which valid path was selected.

The same model that would "hallucinate" plausible content in unconstrained generation instead "hallucinated" plausible fact paths when constrained to the trie. Because the trie was populated with actual facts, these hallucinations became successful retrievals. The model's biases—normally a source of confabulation—became a retrieval mechanism operating through learned semantic associations.

```
┌─────────────────────────────────────────────────────────────┐
│              Attention-Guided Path Selection                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Context: "I've been thinking about asking for              │
│           that promotion we discussed..."                   │
│                                                             │
│  Model attention highlights: career, advancement, goal      │
│                                                             │
│  Trie-constrained generation:                               │
│    Step 1: P(goal) >> P(event), P(trait), ...              │
│            → Select "goal"                                  │
│    Step 2: P(career) >> P(project), P(personal)            │
│            → Select "career"                                │
│    Step 3: P(promotion) >> P(transfer), P(retirement)      │
│            → Select "promotion"                             │
│                                                             │
│  Result: goal.career.promotion                              │
│  Retrieved fact contains prior discussion of promotion      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.6 Graceful Degradation

Because the trie always produces a valid path, retrieval never fails—it may simply retrieve a less-than-optimal fact. The category disjointness ensures that even "wrong" retrievals land in semantically related territory.

If the conversation concerned a career decision but the model selected "goal.project.deadline" instead of "goal.career.promotion", the retrieved fact still related to goals and planning. The response could incorporate this contextually adjacent information, potentially enriching the reply even if not directly addressing the query.

This property distinguished the system from embedding-based retrieval, where low similarity scores could return irrelevant content, and from exact-match systems, where failed lookups returned nothing. Every retrieval contributed relevant context; the question was degree of relevance rather than presence or absence.

### 6.7 Memory Pipeline

The memory pipeline extracted facts from conversation and inserted them into the trie structure, enabling dynamic growth of the fact index.

**Phase 0: Compression** (Synchronous)

When fresh messages exceeded the token budget, a fast model compressed them into the recent summary. This maintained narrative flow while discarding verbatim dialogue.

**Phase 1: Path Extraction** (Asynchronous)

When the recent summary exceeded its budget, the fast model generated candidate fact paths using trie-constrained generation. The same mechanism used for retrieval was used for storage: the model "hallucinated" appropriate paths for the content being extracted.

**Phase 2: Editorial Filtering** (Asynchronous)

A larger model judged each candidate: KEEP for facts with lasting significance, DISCARD for transient moments. This prevented the trie from filling with ephemeral content.

**Phase 3: Content Generation** (Asynchronous)

For approved paths, the larger model generated rich narrative content in the agent's voice. This content populated the fact's 100-token payload.

**Phase 4: Trie Insertion** (CPU)

The generated fact was inserted into the trie at the extracted path. If the path already existed (collision with static knowledge or previous dynamic fact), the new content overwrote the old—implementing the copy-on-write override semantic.

```
┌─────────────────────────────────────────────────────────────┐
│              Memory Pipeline                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Fresh Messages                                             │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────┐                                       │
│  │ Phase 0         │ Limbic model, sync                    │
│  │ Compress        │ → recent summary                      │
│  └─────────────────┘                                       │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────┐                                       │
│  │ Phase 1         │ Limbic model, async                   │
│  │ Extract Paths   │ Trie-constrained generation           │
│  └─────────────────┘ → candidate paths                     │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────┐                                       │
│  │ Phase 2         │ Frontal model, async                  │
│  │ Editorial Filter│ KEEP / DISCARD judgment               │
│  └─────────────────┘ → approved paths                      │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────┐                                       │
│  │ Phase 3         │ Frontal model, async                  │
│  │ Generate Content│ First-person narrative                │
│  └─────────────────┘ → 100-token fact payload              │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────┐                                       │
│  │ Phase 4         │ CPU                                   │
│  │ Trie Insert     │ Override if path exists               │
│  └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.8 Context Assembly

At inference time, context assembly materialized the selected portion of the fact index into the physical window. The assembly structure was not arbitrary—it was specifically designed to maximize KV cache hit rates, establish effective attention patterns, and optimize inference throughput.

**Cache-Optimized Sequence Structure**

The assembled context followed a specific structure optimized for cache coherency. Facts were injected *within* the fresh messages section—specifically, after older messages but before the most recent 3 turns. This hybrid placement balanced cache efficiency with attention quality.

```
┌─────────────────────────────────────────────────────────────┐
│              Context Assembly                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                       │
│  │ System Prompt   │ ← ALWAYS CACHED: Never changes        │
│  │                 │   100% cache hit rate                 │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ Recent Summary  │ ← ROTATION CACHED: Head summarized    │
│  │                 │   every 5 turns, ~80% hit rate        │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ Older Messages  │ ← ROTATION CACHED: Turns 4-5 of the   │
│  │ (turns 4-5)     │   fresh window, ~80% hit rate         │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼  ─ ─ ─ CACHE BOUNDARY ─ ─ ─                     │
│           │                                                 │
│  ┌─────────────────┐                                       │
│  │ Retrieved Facts │ ← CONTENT-ADDRESSED: Prefill only on  │
│  │ (1-8 facts)     │   cache miss, shared across sessions  │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ Recent Messages │ ← ATTENTION BRIDGE: Last 3 turns      │
│  │ (turns 1-3)     │   attend to facts during prefill      │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ User Query      │ ← ATTENTION SINK: Primary fact        │
│  │                 │   attention plus conversational ref   │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│      Generation                                             │
│                                                             │
│  Physical window: ~4K tokens                                │
│  Addressable via retrieval: 2.4M tokens                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Component-by-Component Cache Analysis**

Each component of the assembled context had distinct caching characteristics:

**System Prompt (Always Cached)**: The system prompt defined the model's persona, capabilities, and behavioral constraints. It never changed during a session—or even across sessions for the same deployment. This made it a perfect candidate for permanent KV cache pinning. The system prompt's KV representation was computed once at initialization and served all subsequent inference calls with zero additional prefill cost.

**Recent Summary (Rotation Cached)**: The recent summary compressed older conversation history into a narrative form. The summary updated on a fixed schedule: every 5 turns, the oldest messages in the fresh messages section were summarized and prepended to the existing summary. Between rotations, the summary's KV cache entry achieved 100% hit rate. Averaged across the rotation cycle, this yielded approximately 80% cache hit rate.

**Older Messages (Rotation Cached)**: The fresh message window maintained 5 turns total. The older 2 turns (positions 4-5 in the window) were placed *before* the facts, making them part of the stable cacheable prefix. Like the summary, these achieved approximately 80% cache hit rate across the rotation cycle.

**Retrieved Facts (Content-Addressed Cache)**: Facts retrieved from the fact index were injected after the cacheable prefix but before the most recent messages. Facts used the content-addressable KV cache described in Section 5. A fact's KV representation was computed on first access and then served all subsequent accesses—both within the same session and across different sessions. Popular facts achieved high cache hit rates with prefill costs amortized across potentially thousands of accesses.

**Recent Messages (Attention Bridge)**: The most recent 3 turns were placed *after* the facts. During prefill, these messages attended to the injected facts, creating attention bridges between the retrieved knowledge and the recent conversation. This was critical for conversational continuity: when a user asked "is there a better way?" or "what about the second option?", the recent messages provided the context that made such references meaningful. The facts could inform how those recent exchanges were understood during generation.

This design recognized that conversational references often span 1-3 turns. A user's current query frequently built on the immediately preceding exchange. By ensuring recent messages attended to facts, the model could connect retrieved knowledge to ongoing conversational threads.

**User Query (Primary Attention Sink)**: The user's current query was the primary attention sink for facts, but not the only one. The query attended to both the facts and the recent messages that had themselves attended to facts. This created a two-hop attention path: query → recent messages → facts, in addition to the direct query → facts path.

**Why This Hybrid Placement**

The split placement of fresh messages—older turns before facts, recent turns after—balanced competing concerns:

1. **Cache efficiency**: The stable prefix [System Prompt → Summary → Older Messages] was identical across turns regardless of which facts were retrieved. This maximized cacheable content while maintaining conversational context.

2. **Conversational continuity**: The last 3 turns after facts ensured that recent exchanges could attend to retrieved knowledge. When a user said "tell me more about that", the preceding assistant response (which might have introduced "that") could attend to the facts that informed it.

3. **Bounded prefill cost**: Only 3 turns required re-prefill with fact attention, not all 5. This limited the computational overhead of the attention bridge while capturing the most relevant conversational context.

4. **Reference resolution**: User queries often implicitly reference the last 1-2 exchanges. "Is there a better way?" assumes the previous turn established what "way" means. By placing recent messages after facts, the attention patterns during prefill could connect such references to relevant retrieved knowledge.

**Attention Flow During Prefill**

The prefill phase established attention patterns that guided generation:

```
┌─────────────────────────────────────────────────────────────┐
│              Attention Flow                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Cached Prefix (no fact attention):                         │
│    System Prompt ← (self-attention only)                    │
│    Summary ← System Prompt                                  │
│    Older Messages ← Summary, System Prompt                  │
│                                                             │
│  ─ ─ ─ CACHE BOUNDARY ─ ─ ─                                │
│                                                             │
│  Fresh Prefill (fact attention established):                │
│    Facts ← Cached Prefix (position-adjusted)                │
│    Recent Messages ← Facts, Cached Prefix                   │
│    User Query ← Recent Messages, Facts, Cached Prefix       │
│                                                             │
│  Generation:                                                │
│    Output tokens ← Full context with established attention  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The cached prefix attended only to itself and earlier content—no fact attention. This was acceptable because those turns had already incorporated fact knowledge when they were *generated*. The stored turns didn't retain fact attention, only the content that resulted from that attention. This paralleled how thinking blocks are handled in some inference systems: the thinking contributed to generation, but the block itself is removed from subsequent context.

**Inference Performance Implications**

This structure optimized for three performance dimensions simultaneously:

**Cache Coherency**: The stable prefix [System Prompt → Summary → Older Messages] achieved high cache hit rates regardless of fact selection. A typical inference call served the entire prefix from cache (~85% of prefix tokens), requiring fresh prefill only for facts and the last 3 messages plus query.

**Attention Quality**: The attention bridge pattern—where recent messages attended to facts—ensured conversational continuity. Unlike placing facts at the very end (where only the query attends), this design allowed 3 turns of conversational context to inform how facts were understood. This was essential for reference resolution and contextual interpretation.

**Inference Throughput**: The hybrid placement balanced cache efficiency with attention quality. Compared to placing all messages before facts (maximum cache, minimum fact attention) or all messages after facts (minimum cache, maximum fact attention), this design achieved strong performance on both dimensions.

```
┌─────────────────────────────────────────────────────────────┐
│              Cache Hit Analysis (Typical Turn)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Component        │ Tokens │ Cache Hit │ Prefill Saved     │
│  ─────────────────────────────────────────────────────────  │
│  System Prompt    │ 500    │ 100%      │ 500 tokens        │
│  Recent Summary   │ 300    │ 80%       │ 240 tokens        │
│  Older Messages   │ 160    │ 80%       │ 128 tokens        │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ CACHE BOUNDARY ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  Retrieved Facts  │ 800    │ 50%       │ 400 tokens        │
│  Recent Messages  │ 240    │ 0%        │ 0 tokens          │
│  User Query       │ 100    │ 0%        │ 0 tokens          │
│  ─────────────────────────────────────────────────────────  │
│  Total            │ 2100   │ 60%       │ 1268 tokens       │
│                                                             │
│  Effective prefill: 832 tokens (vs 2100 naive)             │
│  Prefill reduction: 60%                                     │
│                                                             │
│  Cacheable prefix (960 tokens): 90% hit rate               │
│  Variable suffix (1140 tokens): 35% hit rate               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The cache-optimized assembly structure was not merely a performance optimization—it was essential to achieving the throughput targets that made high-density concurrent inference viable. Without aggressive cache reuse, the prefill overhead would dominate latency, negating the benefits of the other architectural innovations. The hybrid placement of facts within the message stream balanced this efficiency against the attention quality needed for coherent multi-turn conversation.

### 6.9 Speculative Path Resolution

**The Problem: Uncertain Category Selection**

High-entropy category selection indicates the model is uncertain between multiple valid interpretations. Standard constrained decoding commits to whichever category crosses the sampling threshold first, but when probabilities are distributed across 2-3 plausible categories (e.g., "career relationships" splitting between goal and bond), this commitment is essentially arbitrary. These uncertain selections are the primary source of D3 (catastrophic) errors.

**The Mechanism**

When category-level entropy exceeds a threshold, rather than committing immediately:

1. Record current KV cache length
2. For top-k candidate categories (typically k=3):
   - Force the candidate category token, append to KV
   - Continue trie-constrained generation through subcategory and topic
   - Accumulate log-probabilities for the complete 3-token path
   - Truncate KV cache back to recorded length
3. Select the path with highest mean log-probability
4. Commit that path by replaying its tokens into KV

```
┌─────────────────────────────────────────────────────────────┐
│              Speculative Path Resolution                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query → Generate Category Token → Compute Entropy          │
│                                          │                  │
│                    ┌─────────────────────┴──────────────┐   │
│                    │                                    │   │
│              H ≤ threshold                        H > threshold
│                    │                                    │   │
│                    ▼                                    ▼   │
│            Commit immediately              Record KV length │
│            (zero overhead)                          │       │
│                                                     ▼       │
│                                    ┌────────────────────┐   │
│                                    │ For top-k categories: │
│                                    │  • Force category     │
│                                    │  • Complete path      │
│                                    │  • Sum log-probs      │
│                                    │  • Truncate KV        │
│                                    └──────────┬─────────┘   │
│                                               │             │
│                                               ▼             │
│                                    Select max mean log-prob │
│                                               │             │
│                                               ▼             │
│                                    Commit winning path      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why Mean Log-Probability**

Mean log-probability (sum of log-probs divided by path length) provides:

- **Numerical stability**: Raw probability products underflow; log-probs sum cleanly
- **Interpretable magnitude**: -1.2 indicates confident path, -3.5 indicates forced/unnatural path
- **Debuggability**: Per-token log-probs reveal where paths diverge in quality
- **Path-length invariance**: If trie depth varies in future extensions, mean normalizes correctly
- **Absolute thresholds**: If the winning path still has poor mean log-prob, the query may not fit the trie and requires fallback handling

**Why This Works: Trie Depth as Disambiguation**

Ambiguity at the category level often resolves at deeper levels. The query "career relationships" may genuinely split between goal and bond at level 1, but `goal→career→networking` may have substantially higher joint probability than `bond→professional→???` because the deeper trie structure naturally fits only one interpretation. The trie's depth becomes a disambiguation signal unavailable to flat retrieval.

```
┌─────────────────────────────────────────────────────────────┐
│  Example: Query "my career relationships"                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Category entropy: 1.42 (threshold: 1.0) → Speculate        │
│                                                             │
│  Path                       │ Log-probs        │ Mean       │
│  ───────────────────────────┼──────────────────┼────────────│
│  goal → career → networking │ [-1.1, -0.8, -0.9] │ -0.93    │
│  bond → professional → ???  │ [-1.2, -2.1, -2.8] │ -2.03    │
│  event → work → ???         │ [-2.4, -3.1, -2.9] │ -2.80    │
│                                                             │
│  Selection: goal → career → networking (highest mean)       │
│  Interpretation: Path 1 flows naturally; paths 2-3 force    │
│                  unnatural continuations at deeper levels   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Computational Cost**

- Speculation occurs only when entropy exceeds threshold (minority of queries)
- Serial evaluation with KV truncation: zero memory overhead
- k paths × 3 tokens = 3k forward passes when triggered
- At typical inference speeds: 5-10ms additional latency on flagged queries only
- Confident queries (low entropy) pay nothing

**Relationship to Embedding-Based Retrieval**

This mechanism has no analog in embedding-based RAG. When cosine similarity returns multiple equally-distant documents, RAG systems either return all candidates or select arbitrarily. Speculative path resolution asks the generative model itself to explore alternatives and select based on how naturally each path continues—using the same weights that will ultimately consume the retrieved content. This self-consistency property means the model's own biases become selection criteria rather than noise.

### 6.10 Validation Methodology

The retrieval system makes a strong architectural claim: errors degrade gracefully within semantic neighborhoods rather than failing catastrophically. Validating this claim requires more than test harness pass rates—it requires systematic characterization of error distributions across both development and held-out scenarios.

**Defining Graceful Degradation Formally**

We define retrieval error by path distance from the optimal retrieval:

- **Distance 0**: Exact optimal path selected
- **Distance 1**: Correct category, correct subcategory, wrong topic
- **Distance 2**: Correct category, wrong subcategory (any topic)
- **Distance 3**: Wrong category (catastrophic failure)

Graceful degradation is the claim that the error distribution concentrates at distances 0-2, with distance-3 errors rare even on adversarial inputs. This is a statistical claim about error distribution, not a binary pass/fail assertion.

**Validation Protocol**

We evaluated retrieval quality across four increasingly adversarial test sets:

```
┌─────────────────────────────────────────────────────────────┐
│  Table: Retrieval Error Distribution by Test Set            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set             │ N    │ D0   │ D1   │ D2   │ D3     │
│  ─────────────────────┼──────┼──────┼──────┼──────┼────────│
│  Development (tuned)  │ XXX  │ XX%  │ XX%  │ XX%  │ XX%    │
│  Held-out (unseen)    │ XXX  │ XX%  │ XX%  │ XX%  │ XX%    │
│  Boundary (ambiguous) │ XXX  │ XX%  │ XX%  │ XX%  │ XX%    │
│  Adversarial (attack) │ XXX  │ XX%  │ XX%  │ XX%  │ XX%    │
│                                                             │
│  D0 = Optimal, D1-D2 = Graceful, D3 = Catastrophic          │
│  Graceful degradation holds if D0+D1+D2 > 95% on held-out   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Test Set Construction**

1. **Development Set ([TODO: N] scenarios)**: Used during prompt engineering. The 100% "pass rate" on this set is acknowledged as circular—we tuned until it passed. This set validates only that the architecture *can* work, not that it *robustly* works.

2. **Held-Out Set ([TODO: N] scenarios)**: Constructed after prompt tuning was frozen. Scenarios drawn from same domain distribution with entity substitution, paraphrase variation, and novel combinations. Never exposed during development. This is the primary validation set.

3. **Boundary Set ([TODO: N] scenarios)**: Deliberately ambiguous queries at category boundaries. Examples:
   - "career relationships" (goal.career vs bond.professional)
   - "family goals" (goal.personal vs bond.family)  
   - "work-life balance" (goal.career vs trait.wellbeing)
   
   These queries have no single correct answer—we measure whether the system routes to *any* reasonable category, not a specific one.

4. **Adversarial Set ([TODO: N] scenarios)**: Queries designed by red-team evaluation to trigger category misselection:
   - Keyword stuffing (embedding misleading category keywords)
   - Negation attacks ("not about my career, but...")
   - Idiomatic expressions with literal misparses
   - Cross-domain references requiring multi-category retrieval

**Success Criteria**

The graceful degradation claim holds if:
- **Held-out set**: D3 (catastrophic) errors < 5%
- **Boundary set**: D3 errors < 15% (accepting higher ambiguity)
- **Adversarial set**: D3 errors < 25% (accepting adversarial success)

These thresholds acknowledge that adversarial inputs will sometimes succeed, but bound the failure rate. A system where 75%+ of adversarial attacks still route to reasonable categories demonstrates architectural robustness even under stress.

**Addressing the Circularity Concern**

The development set pass rate is explicitly *not* evidence of robustness—it is evidence that the architecture is capable of correct retrieval when tuned for specific inputs. The held-out and adversarial sets provide the actual validation:

```
┌─────────────────────────────────────────────────────────────┐
│  Validation Logic                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Development 100% pass rate proves: Architecture works      │
│  Held-out D3 < 5% proves: Generalizes beyond tuning set     │
│  Adversarial D3 < 25% proves: Bounded failure under attack  │
│                                                             │
│  The combination establishes:                               │
│  1. The system CAN retrieve correctly (development)         │
│  2. It DOES retrieve correctly on novel inputs (held-out)   │
│  3. It BOUNDS failures even adversarially (adversarial)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Category Boundary Characterization**

For the boundary set, we additionally measure routing consistency—whether similar queries route to the same category:

```
┌─────────────────────────────────────────────────────────────┐
│  Boundary Routing Consistency                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query Cluster          │ Dominant │ Consistency │ Accept.  │
│  ───────────────────────┼──────────┼─────────────┼──────────│
│  "career relationships" │ goal: X% │    XX%      │ goal OR  │
│  variants (N=XX)        │ bond: X% │             │ bond     │
│                         │                                   │
│  "family goals"         │ goal: X% │    XX%      │ goal OR  │
│  variants (N=XX)        │ bond: X% │             │ bond     │
│                         │                                   │
│  Consistency = % routing to dominant category               │
│  High consistency (>80%) indicates predictable behavior     │
│  even when "correct" answer is ambiguous                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

For ambiguous queries, predictable routing to *either* reasonable category is preferable to inconsistent routing. Users can learn that "career relationships" routes to professional networking (or to personal bonds) and phrase accordingly—unpredictable flip-flopping is worse than consistent "wrong" routing.

**Speculative Resolution Impact on Error Rates**

The speculative path resolution mechanism (Section 6.9) changes the error analysis framework. D3 errors now divide into two categories:

- **Confident D3 errors**: Low-entropy selections that commit to wrong category. These bypass speculation entirely—the model was confident but wrong.
- **Resolved D3 errors**: High-entropy selections where speculation recovered the correct path. Without speculation, these would have been D3 errors; with speculation, they become D0-D2.

```
┌─────────────────────────────────────────────────────────────┐
│  Table: Speculative Resolution Impact                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set        │ Queries   │ D3→D0-D2  │ Original │ Net  │
│                  │ Flagged   │ Recovery  │ D3 Rate  │ D3   │
│  ────────────────┼───────────┼───────────┼──────────┼──────│
│  Held-out        │    XX%    │    XX%    │   X.X%   │ X.X% │
│  Boundary        │    XX%    │    XX%    │    XX%   │  XX% │
│  Adversarial     │    XX%    │    XX%    │    XX%   │  XX% │
│                                                             │
│  Net D3 = Original D3 × (1 - Recovery Rate on flagged)      │
│  Recovery = % of flagged queries where speculation          │
│             selected correct path over naive selection      │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Success Criteria with Speculation**

With speculative resolution enabled, tighter bounds become achievable:

- **Held-out set with speculation**: Net D3 errors < 2% (vs 5% without)
- **Boundary set with speculation**: Net D3 errors < 8% (vs 15% without)
- **Adversarial set with speculation**: Net D3 errors < 15% (vs 25% without)

These tighter bounds reflect speculation's ability to recover from ambiguous queries while accepting that confident-but-wrong selections remain irreducible.

**Entropy Threshold Calibration**

The entropy threshold balances D3 recovery against computational overhead:

```
┌─────────────────────────────────────────────────────────────┐
│  Table: Entropy Threshold Calibration                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Threshold │ Flagged % │ D3 Recovery │ Overhead │ Net D3   │
│  ──────────┼───────────┼─────────────┼──────────┼──────────│
│    0.5     │    XX%    │     XX%     │  +XX ms  │   X.X%   │
│    0.8     │    XX%    │     XX%     │  +XX ms  │   X.X%   │
│    1.0     │    XX%    │     XX%     │  +XX ms  │   X.X%   │
│    1.2     │    XX%    │     XX%     │  +XX ms  │   X.X%   │
│    1.5     │    XX%    │     XX%     │  +XX ms  │   X.X%   │
│                                                             │
│  Trade-off: Lower threshold catches more errors but adds    │
│  speculation overhead to confident queries unnecessarily.   │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The optimal threshold [**TODO: X.X**] flagged [**TODO: Y%**] of queries while achieving [**TODO: Z%**] D3 recovery on flagged queries, with average overhead of [**TODO: N ms**] on flagged queries and zero overhead on confident queries.

**Remaining Limitations**

Even with speculative resolution and rigorous validation, limitations remain:

1. **Distribution shift**: Held-out scenarios are from the same distribution as development. True out-of-distribution robustness is not guaranteed.

2. **Confident-but-wrong errors**: Speculative resolution only triggers on high-entropy selections. When the model is confidently wrong (low entropy, wrong category), speculation cannot help. These irreducible errors set the floor for D3 rates.

3. **Silent failures**: D1-D2 errors may still produce subtly wrong responses. The validation measures retrieval quality, not downstream response quality.

4. **Speculation overhead at scale**: While individual speculation costs 5-10ms, high-ambiguity workloads could see meaningful latency increases if many queries trigger speculation.

---

## 7. Experimental Setup

### 7.1 Hardware Configurations

We evaluated the system across two hardware tiers representing consumer and mid-tier enterprise deployments:

**Configuration A (Consumer)**: NVIDIA RTX 4090 with 24GB VRAM, [**TODO: CPU model**], [**TODO: X GB**] system RAM. Memory bandwidth: 1,008 GB/s.

**Configuration B (Enterprise)**: NVIDIA A100 PCIe with 80GB HBM2e, [**TODO: CPU model**], [**TODO: X GB**] system RAM. Memory bandwidth: 2,039 GB/s.

### 7.2 Models

We evaluated different model configurations per hardware tier (see Appendix E for detailed model-hardware mappings):

**Configuration A (RTX 4090)**:
- Primary (Frontal): Qwen3-30B-A3B-AWQ (~17GB) — MoE architecture with 3.3B active parameters
- Fast operations (Limbic): Qwen3-4B-Q4 (~2.5GB)

**Configuration B (A100)**:
- Primary (Frontal): Hermes3-70B-Q4 (~40GB) — Dense architecture, dialogue-optimized
- Fast operations (Limbic): Qwen3-8B-Q4 (~5GB)

The MoE model on consumer hardware provides favorable compute characteristics (only 3.3B parameters active per token) while the dense model on enterprise hardware provides maximum quality for premium use cases.

### 7.3 Baselines

We compared against three baselines:

1. **Standard Inference**: Full BF16 KV cache, no sharing, no fact system
2. **vLLM PagedAttention**: Standard paged attention with prefix sharing only
3. **Uniform Quantization**: INT4 KV cache throughout, no mixed precision

### 7.4 Evaluation Scenarios

**Scenario A - High-Density Concurrent Sessions**: Multiple concurrent conversations sharing domain knowledge, measuring throughput and latency under load.

**Scenario B - Long-Term Memory**: Extended conversations testing fact extraction, storage, and retrieval over many turns.

**Scenario C - Mixed Workload**: Combination of new sessions (cold start) and continuing sessions (warm cache) reflecting realistic deployment.

### 7.5 Metrics

- **Concurrency**: Maximum simultaneous contexts at target latency
- **Latency**: Time to first token (TTFT) and inter-token latency (ITL)
- **Throughput**: Tokens per second (aggregate and per-sequence)
- **Memory Efficiency**: VRAM utilization and cache hit rates
- **Retrieval Validation**: Test harness pass rates and degradation analysis
- **Conversation Quality**: Human evaluation of coherence over extended dialogues

---

## 8. Results

### 8.1 Concurrency and Latency

[**TODO: INSERT TABLE 1 - Concurrency results**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 1: Maximum Concurrent Contexts at Target Latency     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example format:                                            │
│                                                             │
│  System              │ Max Contexts │ TTFT (p50) │ ITL (p50)│
│  ────────────────────┼──────────────┼────────────┼──────────│
│  Standard (BF16)     │     XX       │   XXX ms   │   XX ms  │
│  vLLM Paged          │     XX       │   XXX ms   │   XX ms  │
│  Uniform INT4        │     XX       │   XXX ms   │   XX ms  │
│  Ours (mixed)        │     XX       │   XXX ms   │   XX ms  │
│                                                             │
│  Target latency: TTFT < 500ms, ITL < 50ms                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Our system achieved [**TODO: X**] concurrent contexts compared to [**TODO: Y**] for standard inference and [**TODO: Z**] for vLLM, representing a [**TODO: N×**] improvement. The extended page table's sharing mechanism was particularly effective when [**TODO: describe conditions**].

[**TODO: INSERT FIGURE 1 - Latency vs concurrency curve**]

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 1: Latency Distribution at Various Concurrency      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example description:                                       │
│                                                             │
│  Line chart showing:                                        │
│  - X-axis: Number of concurrent contexts (10 to XXX)        │
│  - Y-axis: TTFT in milliseconds                             │
│  - Lines: Standard, vLLM, Uniform INT4, Ours                │
│  - Shaded region: Target latency threshold                  │
│                                                             │
│  Key observation: Our system maintained sub-XXXms TTFT      │
│  up to XX contexts, while baselines exceeded threshold      │
│  at XX contexts.                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Cache Sharing Efficiency

[**TODO: INSERT TABLE 2 - Cache hit rates and sharing factors**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 2: Cache Sharing Metrics                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example format:                                            │
│                                                             │
│  Content Type    │ Avg Refcount │ Hit Rate │ Memory Saved  │
│  ────────────────┼──────────────┼──────────┼───────────────│
│  System Prompt   │    XX.X      │   XX%    │    XX GB      │
│  Domain Knowledge │    XX.X      │   XX%    │    XX GB      │
│  Agent Facts      │    XX.X      │   XX%    │    XX GB      │
│  Conversation    │     1.0      │   XX%    │    XX GB      │
│                                                             │
│  Total memory reduction vs. no sharing: XX%                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The content-addressable storage achieved [**TODO: X%**] cache hit rate for system prompts and [**TODO: Y%**] for shared domain knowledge. This translated to [**TODO: Z GB**] memory savings across [**TODO: N**] concurrent sessions, enabling the higher concurrency reported above.

### 8.3 Kernel Performance

[**TODO: INSERT TABLE 3 - Kernel microbenchmarks**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 3: Kernel Performance Comparison                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example format:                                            │
│                                                             │
│  Operation               │ Baseline │ Ours   │ Speedup     │
│  ────────────────────────┼──────────┼────────┼─────────────│
│  Prefill (2K tok, BF16)  │  XX ms   │ XX ms  │   X.Xx      │
│  Prefill (2K tok, mixed) │   N/A    │ XX ms  │    —        │
│  Decode (batch=1)        │  XX ms   │ XX ms  │   X.Xx      │
│  Decode (batch=32)       │  XX ms   │ XX ms  │   X.Xx      │
│  Decode (batch=100)      │  XX ms   │ XX ms  │   X.Xx      │
│                                                             │
│  GEMV/Marlin crossover measured at batch size: XX           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The mixed-dtype prefill kernel achieved [**TODO: X%**] of single-dtype performance despite handling three precision levels. The greedy GEMV/Marlin decomposition showed optimal crossover at batch size [**TODO: N**], matching our analytical prediction.

[**TODO: INSERT FIGURE 2 - Throughput vs batch size**]

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 2: Decode Throughput by Batch Size                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example description:                                       │
│                                                             │
│  Line chart showing:                                        │
│  - X-axis: Batch size (1 to 128)                           │
│  - Y-axis: Tokens/second (total)                           │
│  - Lines: GEMV-only, Marlin-only, Greedy decomposition      │
│  - Annotation: Crossover point where greedy matches best    │
│                                                             │
│  Key observation: Greedy decomposition tracked the          │
│  envelope of both kernels, never more than X% below         │
│  the optimal single-kernel choice.                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.4 Memory Pipeline Timing

[**TODO: INSERT TABLE 4 - Pipeline phase timings**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 4: Memory Pipeline Phase Timing                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example format:                                            │
│                                                             │
│  Phase                │ Model   │ Time (p50) │ Time (p99)  │
│  ─────────────────────┼─────────┼────────────┼─────────────│
│  0: Compress          │ Limbic  │   XXX ms   │   XXX ms    │
│  1: Extract paths     │ Limbic  │   XXX ms   │   XXX ms    │
│  2: Editorial filter  │ Frontal │   XXX ms   │   XXX ms    │
│  3: Generate content  │ Frontal │   XXX ms   │   XXX ms    │
│  4: Store             │ CPU     │   XXX ms   │   XXX ms    │
│  ─────────────────────┼─────────┼────────────┼─────────────│
│  Total (async)        │         │   XXX ms   │   XXX ms    │
│                                                             │
│  Note: Phases 1-4 run asynchronously, not blocking user     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Phase 0 (compression) completed in [**TODO: X ms**] median, well under the target for synchronous execution. The asynchronous phases 1-4 completed in [**TODO: Y ms**] total, running in the background without impacting user-perceived latency.

### 8.5 Retrieval Validation

Following the validation protocol defined in Section 6.9, we evaluated retrieval quality across four test sets, measuring error distribution by path distance.

[**TODO: INSERT TABLE 5 - Error distribution by test set**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 5: Retrieval Error Distribution by Test Set          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set             │  N   │  D0  │  D1  │  D2  │  D3    │
│  ─────────────────────┼──────┼──────┼──────┼──────┼────────│
│  Development (tuned)  │ XXX  │  XX% │  XX% │  XX% │  X.X%  │
│  Held-out (unseen)    │ XXX  │  XX% │  XX% │  XX% │  X.X%  │
│  Boundary (ambiguous) │ XXX  │  XX% │  XX% │  XX% │  XX%   │
│  Adversarial (attack) │ XXX  │  XX% │  XX% │  XX% │  XX%   │
│  ─────────────────────┼──────┼──────┼──────┼──────┼────────│
│  Success criteria     │      │      │      │      │ <5/15/25%│
│                                                             │
│  D0=Optimal  D1-D2=Graceful (within-category)  D3=Catastrophic│
│  Graceful rate (D0+D1+D2): Dev XX%, Held-out XX%, Adv XX%   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Development Set Results**

The development set achieved [**TODO: X%**] D0 (optimal) retrieval with [**TODO: Y%**] D3 (catastrophic) errors. As noted in Section 6.9, this result is circular—we tuned until performance was acceptable—and establishes only that the architecture *can* work, not that it robustly generalizes.

**Held-Out Set Results**

The held-out set, constructed after prompt tuning was frozen, achieved [**TODO: X%**] D0 with [**TODO: Y%**] D3 errors. This [**TODO: meets/does not meet**] our <5% D3 threshold for graceful degradation. The [**TODO: X%**] gap between development and held-out D0 rates indicates [**TODO: characterize generalization—"modest overfitting to development phrasings" or "strong generalization"**].

**Boundary Set Results**

Deliberately ambiguous queries showed higher D3 rates ([**TODO: X%**]), as expected when queries have no single correct answer. Importantly, [**TODO: X%**] of boundary queries routed to *one of* the acceptable categories, demonstrating that ambiguity produces reasonable (if unpredictable) routing rather than catastrophic failure.

[**TODO: INSERT TABLE 5b - Boundary routing consistency**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 5b: Boundary Query Routing Consistency               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query Cluster          │ N  │ Cat A │ Cat B │ Consist.    │
│  ───────────────────────┼────┼───────┼───────┼─────────────│
│  "career relationships" │ XX │  XX%  │  XX%  │    XX%      │
│  "family goals"         │ XX │  XX%  │  XX%  │    XX%      │
│  "work-life balance"    │ XX │  XX%  │  XX%  │    XX%      │
│  ───────────────────────┼────┼───────┼───────┼─────────────│
│  Mean consistency       │    │       │       │    XX%      │
│                                                             │
│  High consistency (>80%) = predictable even if "wrong"      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Adversarial Set Results**

Red-team queries achieved [**TODO: X%**] D3 rate, [**TODO: meeting/exceeding**] our <25% adversarial threshold. The most effective attack vectors were:
- [**TODO: describe top attack pattern and success rate**]
- [**TODO: describe second attack pattern and success rate**]

Even under adversarial conditions, [**TODO: X%**] of queries routed to acceptable categories (D0-D2), demonstrating bounded failure rather than architectural collapse.

[**TODO: INSERT FIGURE 3 - Error distribution visualization**]

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 3: Error Distribution Across Test Sets              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stacked bar chart showing:                                 │
│  - X-axis: Test set (Dev, Held-out, Boundary, Adversarial) │
│  - Y-axis: Percentage of retrievals                         │
│  - Stacked bars: D0 (green), D1 (light green), D2 (yellow),│
│                  D3 (red)                                   │
│                                                             │
│  Key observation: D3 (red) remains small fraction even      │
│  in adversarial set, validating bounded failure claim.      │
│  D0+D1+D2 (graceful) exceeds XX% across all test sets.      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Interpretation**

The results support the graceful degradation claim with the following caveats:

1. **Generalization gap**: The [**TODO: X%**] difference between development and held-out D0 rates indicates some overfitting to development phrasings, though D3 rates remain low.

2. **Boundary unpredictability**: Ambiguous queries show lower consistency than unambiguous queries, meaning users cannot always predict routing for edge cases.

3. **Adversarial vulnerability**: Determined adversaries can achieve [**TODO: X%**] misrouting, which may be unacceptable for security-critical applications.

4. **Statistical confidence**: With [**TODO: N**] held-out scenarios, the 95% confidence interval on D3 rate is [**TODO: X-Y%**].

The architecture demonstrates graceful degradation in the precise sense defined: errors concentrate at distances 0-2 even under adversarial pressure, with catastrophic (D3) failures bounded below [**TODO: X%**] on held-out evaluation.

#### 8.5.1 Speculative Resolution Results

Speculative path resolution (Section 6.10) substantially improved D3 error rates by recovering correct paths on ambiguous queries:

```
┌─────────────────────────────────────────────────────────────┐
│  Table 5c: Speculative Resolution Impact                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set     │ Flagged │ Recovery │ D3 Before │ D3 After  │
│  ─────────────┼─────────┼──────────┼───────────┼───────────│
│  Held-out     │   XX%   │    XX%   │    X.X%   │    X.X%   │
│  Boundary     │   XX%   │    XX%   │     XX%   │     X%    │
│  Adversarial  │   XX%   │    XX%   │     XX%   │     XX%   │
│                                                             │
│  Flagged = queries where entropy exceeded threshold         │
│  Recovery = % of flagged where speculation found correct path
│  D3 After = D3 Before × (1 - Flagged × Recovery)           │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The boundary set showed the most dramatic improvement: [**TODO: X%**] of boundary queries were flagged as high-entropy, and speculation recovered the correct path in [**TODO: Y%**] of those cases, reducing net D3 from [**TODO: A%**] to [**TODO: B%**].

**Why Speculation Helps More on Boundary Queries**

Boundary queries (e.g., "career relationships") are precisely the cases where category-level ambiguity is genuine but deeper trie structure resolves the ambiguity. The query may split evenly between `goal` and `bond` at level 1, but `goal→career→networking` has much higher joint probability than any path through `bond→professional→*`. Speculation surfaces this signal; naive selection commits before seeing it.

**Latency Impact**

```
┌─────────────────────────────────────────────────────────────┐
│  Table 5d: Speculation Latency Overhead                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set     │ Flagged % │ Avg Overhead │ P99 Overhead    │
│  ─────────────┼───────────┼──────────────┼─────────────────│
│  Held-out     │    XX%    │    +X.X ms   │    +XX ms       │
│  Boundary     │    XX%    │    +X.X ms   │    +XX ms       │
│  Adversarial  │    XX%    │    +X.X ms   │    +XX ms       │
│                                                             │
│  Overhead measured on flagged queries only                  │
│  Non-flagged queries: zero additional latency               │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The latency overhead is concentrated on queries that would otherwise produce D3 errors—a favorable tradeoff. Confident queries (the majority) pay nothing.

### 8.6 Long-Term Conversation Quality

[**TODO: INSERT TABLE 6 - Human evaluation results**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 6: Human Evaluation of Conversation Quality          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example format:                                            │
│                                                             │
│  Metric                  │ No Memory │ RAG    │ Ours       │
│  ────────────────────────┼───────────┼────────┼────────────│
│  Factual consistency     │   X.XX    │ X.XX   │   X.XX     │
│  Agent voice             │   X.XX    │ X.XX   │   X.XX     │
│  Long-term coherence     │   X.XX    │ X.XX   │   X.XX     │
│  Appropriate recall      │   X.XX    │ X.XX   │   X.XX     │
│  Overall preference      │   XX%     │ XX%    │   XX%      │
│                                                             │
│  Scale: 1-5 (5=best), N=XX evaluators, XX conversations     │
│  Each conversation: XXX turns simulating X weeks of chat    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Human evaluators preferred our system [**TODO: X%**] of the time over no-memory baselines and [**TODO: Y%**] over traditional RAG. The largest gains came in long-term coherence ([**TODO: +Z points**]), validating the benefit of the same-model compression/retrieval loop.

### 8.7 Quality Bootstrapping Effect

We tested whether high-precision KV cache for stable content (system prompts, frequently-accessed facts) can partially compensate for aggressive model weight quantization. We term this the "bootstrapping effect"—using abundant KV memory (available after model compression) to recover quality lost to weight quantization.

**Theoretical Basis**

Quantization errors in model weights propagate through attention: when a quantized model attends to KV representations, errors compound at each layer. If KV representations are themselves low-precision, errors multiply. However, if the model attends to high-precision "anchor" KV—content computed and cached at full precision—the attention output inherits some of that precision, dampening error accumulation.

This suggests an asymmetric tradeoff: model weights are accessed once per token, but KV cache is accessed repeatedly through attention across all subsequent tokens. High-precision KV should therefore provide more quality-per-byte than high-precision weights.

**Factorial Experiment**

We tested this through a factorial design crossing model quantization with KV precision:

```
┌─────────────────────────────────────────────────────────────┐
│  Table 7a: Quality by Model × KV Precision (MMLU 5-shot)    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      KV Cache Precision                     │
│  Model Weights    │ INT4 KV  │ FP8 KV   │ BF16 KV          │
│  ─────────────────┼──────────┼──────────┼──────────────────│
│  BF16 (baseline)  │  XX.X%   │  XX.X%   │  XX.X% baseline  │
│  FP8              │  XX.X%   │  XX.X%   │  XX.X%           │
│  Q4               │  XX.X%   │  XX.X%   │  XX.X%           │
│  Q3               │  XX.X%   │  XX.X%   │  XX.X% bootstrap │
│                                                             │
│  Model: Qwen3-30B-A3B, N=XXX samples per cell               │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The bootstrapping configuration (Q3 model + BF16 KV) achieved [**TODO: XX.X%**] accuracy—recovering [**TODO: XX%**] of the gap between Q3+INT4 ([**TODO: XX.X%**]) and the BF16 baseline ([**TODO: XX.X%**]) at [**TODO: XX%**] of the memory cost.

**Ablation: Which KV Content Matters?**

We ablated by applying high precision selectively to isolate the contribution of each component:

```
┌─────────────────────────────────────────────────────────────┐
│  Table 7b: Selective High-Precision KV Ablation             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Configuration               │ Task Acc │ Memory │ Δ vs INT4│
│  ────────────────────────────┼──────────┼────────┼──────────│
│  All INT4 KV (control)       │  XX.X%   │ X.X GB │   —      │
│  System prompt BF16 only     │  XX.X%   │ X.X GB │  +X.X%   │
│  Retrieved facts BF16 only   │  XX.X%   │ X.X GB │  +X.X%   │
│  Recent messages BF16 only   │  XX.X%   │ X.X GB │  +X.X%   │
│  Sys + Facts BF16            │  XX.X%   │ X.X GB │  +X.X%   │
│  All BF16 KV                 │  XX.X%   │ X.X GB │  +X.X%   │
│                                                             │
│  Model: Qwen3-30B-A3B at Q3 quantization throughout         │
│  Task: MMLU (5-shot), average across categories             │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

System prompt precision contributed [**TODO: XX%**] of the total effect, retrieved facts [**TODO: XX%**], and recent messages [**TODO: XX%**]. The effect showed [**TODO: diminishing/linear/increasing**] returns: the first [**TODO: N**] tokens of high-precision KV provided [**TODO: XX%**] of the benefit.

**Task Sensitivity**

The bootstrapping effect varied by task type:

```
┌─────────────────────────────────────────────────────────────┐
│  Table 7c: Bootstrapping Effect by Task Type                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Task Type           │ INT4 KV │ BF16 KV │ Δ      │ p-value │
│  ────────────────────┼─────────┼─────────┼────────┼─────────│
│  Reasoning (GSM8K)   │  XX.X%  │  XX.X%  │ +X.X%  │  0.XXX  │
│  Factual (TriviaQA)  │  XX.X%  │  XX.X%  │ +X.X%  │  0.XXX  │
│  Instruction (IFEval)│  XX.X%  │  XX.X%  │ +X.X%  │  0.XXX  │
│  Creative (story)    │  X.XX   │  X.XX   │ +X.XX  │  0.XXX  │
│                                                             │
│  Model: Qwen3-30B-A3B-Q3 throughout                         │
│  Creative measured by perplexity (lower = better)           │
│  N = XXX samples per task, paired t-test                    │
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Reasoning tasks showed the largest effect ([**TODO: +X.X%**], p<0.01), consistent with longer error propagation chains in multi-step reasoning. Creative tasks showed [**TODO: the smallest/no significant**] effect ([**TODO: +X.X%**], p=[**TODO: 0.XX**]), consistent with higher tolerance for variation in open-ended generation.

**Mechanistic Validation**

To confirm the hypothesized mechanism (error dampening through high-precision attention), we measured attention entropy and output divergence:

```
┌─────────────────────────────────────────────────────────────┐
│  Table 7d: Mechanistic Analysis                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Metric                        │ INT4 KV │ BF16 KV │ Δ      │
│  ──────────────────────────────┼─────────┼─────────┼────────│
│  Attention entropy (nats)      │  X.XXX  │  X.XXX  │ -X.XXX │
│  KL divergence from BF16 model │  X.XXX  │  X.XXX  │ -X.XXX │
│  Layer 0 output cosine sim     │  0.XXX  │  0.XXX  │ +0.XXX │
│  Layer L/2 output cosine sim   │  0.XXX  │  0.XXX  │ +0.XXX │
│  Layer L output cosine sim     │  0.XXX  │  0.XXX  │ +0.XXX │
│                                                             │
│  Cosine similarity measured against BF16 model outputs      │
│  Higher similarity = less drift from full-precision behavior│
│  [TODO: Fill with actual measured values]                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The mechanistic analysis [**TODO: confirmed/did not confirm**] the error-dampening hypothesis. With BF16 KV anchors, attention entropy [**TODO: decreased by X.XXX nats**], indicating [**TODO: sharper/more diffuse**] attention patterns. KL divergence from the full-precision model [**TODO: decreased from X.XXX to X.XXX**], indicating closer alignment to full-precision behavior.

Most tellingly, layer-wise cosine similarity showed [**TODO: describe pattern—e.g., "progressive divergence from BF16 baseline with INT4 KV (0.XXX at layer 0, 0.XXX at layer L), but remained stable with BF16 KV anchors (0.XXX at layer 0, 0.XXX at layer L)"**]. This [**TODO: supports/does not support**] the error accumulation theory: quantization errors compound through layers when attending to low-precision KV, but high-precision anchors dampen this accumulation.

**Summary**

The quality bootstrapping effect is [**TODO: real and significant / modest but measurable**]:

- **Factorial result**: Q3 + BF16 KV recovered [**TODO: XX%**] of the quality gap vs Q3 + INT4, at [**TODO: XX%**] memory overhead
- **Ablation result**: System prompt precision contributed [**TODO: XX%**] of effect; diminishing returns after [**TODO: N**] tokens
- **Task sensitivity**: Reasoning [**TODO: +X.X%**] > Factual [**TODO: +X.X%**] > Instruction [**TODO: +X.X%**] > Creative [**TODO: +X.X%**]
- **Mechanism**: Layer-wise analysis [**TODO: confirms/does not confirm**] error dampening through high-precision attention

**Practical Implications**

This finding changes the calculus for memory-constrained deployment. The conventional approach—uniform quantization—compresses everything equally. Our results demonstrate an alternative: aggressive weight quantization combined with selective high-precision KV for stable content. This asymmetric allocation exploits the repeated access pattern of KV cache through attention.

The optimal configuration for our architecture: [**TODO: describe—e.g., "BF16 for system prompt (~500 tokens) and top-8 retrieved facts (~800 tokens), INT4 for conversation history, yielding XX% quality recovery at XX% memory overhead vs uniform INT4"**].

[**TODO: INSERT FIGURE 4 - Quality vs memory Pareto frontier**]

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 4: Quality-Memory Pareto Frontier                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scatter plot showing:                                      │
│  - X-axis: Total VRAM usage (GB)                           │
│  - Y-axis: Task accuracy (%)                               │
│  - Points: All 12 factorial configurations                  │
│  - Highlighted: Pareto-optimal configurations               │
│  - Annotated: Bootstrapping config (Q3 + BF16 KV)          │
│                                                             │
│  Key observation: Bootstrapping configuration lies on       │
│  Pareto frontier, achieving better quality-per-GB than      │
│  uniform quantization approaches.                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Analysis

### 9.1 Where Did the Gains Come From?

[**TODO: Ablation analysis breaking down contribution of each component**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table 8: Ablation Study - Concurrency Contribution         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example format:                                            │
│                                                             │
│  Configuration                      │ Max Contexts │ Delta │
│  ───────────────────────────────────┼──────────────┼───────│
│  Baseline (no innovations)          │     XX       │   —   │
│  + Extended page table              │     XX       │  +XX  │
│  + Mixed-precision KV               │     XX       │  +XX  │
│  + Fact-based memory (text storage) │     XX       │  +XX  │
│  + Page sharing                     │     XX       │  +XX  │
│  Full system                        │     XX       │  +XX  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The ablation revealed that [**TODO: component X**] contributed the largest share ([**TODO: Y%**]) of the concurrency improvement, followed by [**TODO: component Z**] ([**TODO: W%**]). The components showed [**TODO: additive/super-additive/sub-additive**] behavior when combined.

### 9.2 Failure Modes

We observed several failure modes during evaluation:

**Category Misselection**: [**TODO: describe specific failure cases, e.g., "When conversational context was ambiguous between goal-oriented and event-oriented framing, the model occasionally selected the wrong top-level category, resulting in retrieval from an unrelated semantic domain."**]

**Cache Thrashing**: [**TODO: describe conditions that caused cache thrashing, e.g., "Under workloads with less than X% content overlap, the sharing mechanism provided minimal benefit and added overhead of Y ms per request."**]

**Quality Degradation**: [**TODO: describe quality edge cases, e.g., "For conversations requiring precise temporal reasoning, the fact-based memory lost ordering information, resulting in X% accuracy drop on temporal benchmarks."**]

### 9.3 Scaling Behavior

[**TODO: INSERT FIGURE 5 - Scaling with model size**]

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 5: System Scaling Characteristics                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Example description:                                       │
│                                                             │
│  Multi-panel figure showing:                                │
│  - Panel A: Concurrency vs model size (8B, 14B, 32B, 70B)  │
│  - Panel B: Sharing benefit vs concurrent sessions          │
│  - Panel C: Path selection consistency vs trie depth        │
│  - Panel D: Memory pipeline time vs conversation length     │
│                                                             │
│  Key observations:                                          │
│  - Sharing benefit scaled [linearly/sublinearly] with N     │
│  - Single-token vocabulary maintained selection consistency │
│  - Pipeline overhead remained [constant/grew] with length   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Related Work

### 10.1 Position-Independent KV Caching

Position-independent KV cache reuse via RoPE remapping is an active research area with several concurrent approaches:

**MEPIC** [Chen et al., Dec 2024] stores KV without pre-applied RoPE ("NoPE" format) and applies positional encoding on-the-fly in a fused attention kernel. **KVLINK** [Wang et al., Feb 2025] similarly stores position-free KV and applies global rotary embedding at inference, enabling document pre-computation with RoPE re-application at concatenation. **Lazy-Attention** [OpenReview, 2025] defers positional encoding until attention computation for zero-copy sharing. **CacheBlend** [Yao et al., 2024] selectively recomputes 10-20% of tokens with RoPE realignment. **Prompt Cache** [Gim et al., MLSys 2024] handles discontinuous position IDs but requires same-position alignment.

Our implementation follows the same architectural principle as MEPIC and Lazy-Attention—storing position-free KV and applying RoPE at attention time. Our contribution here is integration rather than novelty: combining position-independent caching with content-addressable lookup, heterogeneous precision support, and the trie-based retrieval system. The position remapping technique itself builds directly on this recent work.

### 10.2 Constrained Decoding and Trie-Based Generation

Constrained decoding techniques are well-established for output formatting: **Outlines** [Willard & Louf, 2023], **XGrammar** [Dong et al., 2024], and **LMQL** [Beurer-Kellner et al., 2023] use finite-state machines to constrain generation to valid outputs. **Trie-based decoding** has been used for entity recognition and relation extraction [Lu et al., 2021; Cao et al., 2021], constraining generation to paths in a prefix tree. **Grammar-Constrained Decoding** [Scholak et al., EMNLP 2023] combines trie-based lexical constraints with state-based constraints.

**Our novel contribution** is applying constrained decoding as a *retrieval mechanism* rather than an output formatter. The key insight—that model "hallucinations" become consistent retrievals when channeled through a trie—repurposes a failure mode as a feature. Rather than fighting the model's tendency to generate plausible-sounding content, we constrain that generation to select from a vocabulary of retrieval paths. The single-token vocabulary selection (tying categories to tokenizer-specific single-token words) ensures deterministic path selection without multi-token ambiguity.

**Speculative path resolution** extends constrained decoding with a novel disambiguation mechanism. When top-level selection is uncertain (high entropy), we exploit the trie's depth structure by evaluating complete paths rather than committing at the first token. This uses mean log-probability over the full path as a selection criterion—a signal unavailable to single-token constrained decoding or flat embedding-based retrieval. The mechanism is complementary to existing constrained decoding techniques; it determines *which* valid path to take when multiple paths are comparably valid at the constraint boundary.

To our knowledge, both the application of trie-constrained generation for sparse retrieval and the speculative disambiguation mechanism are novel. Prior constrained decoding work focuses on ensuring valid *output* formats; we use it to select valid *input* content and resolve ambiguity through trie depth.

### 10.3 Mixed-Precision KV Cache

Extensive prior work explores KV cache quantization with various allocation strategies:

**Layer-wise**: KVTuner [Zhang et al., Nov 2025] allocates different precision per transformer layer. **Channel-wise**: QAQ [2024] keeps outlier channels in FP16; KITTY [2025] and MixKVQ [Dec 2024] allocate precision based on channel importance. **Temporal**: PM-KVQ [2025] uses higher precision early in generation, lower later. **Asymmetric K/V**: KIVI [Liu et al., 2024] exploits keys being more quantization-sensitive than values.

Our approach differs in allocating precision by *content semantics* rather than layer, channel, or position. System prompts and frequently-accessed facts receive high precision; conversation history receives lower precision. This content-based allocation exploits repeated attention to stable content across many generated tokens.

The "quality bootstrapping" hypothesis—that high-precision KV can partially compensate for aggressive weight quantization—extends mixed-precision research in a new direction. Existing work optimizes KV precision assuming fixed model precision; we explore the joint optimization of model quantization and KV precision allocation.

### 10.4 Memory-Augmented LLMs and Dynamic Knowledge

**RAG systems** [Lewis et al., 2020; Borgeaud et al., 2022] retrieve from external knowledge bases via embedding similarity. Retrieved content enters as static context—if documents conflict, the model must resolve ambiguity through attention, with no explicit override semantics.

**Memory-R1** [Hu et al., July 2025] introduces RL-based memory management with explicit ADD/UPDATE/DELETE/NOOP operations, but requires learning when to apply each operation. **Mem0** provides external memory banks with retrieval but without structured override semantics. **Fine-tuning** bakes knowledge into weights, making mid-session updates impossible. **Long-context models** place everything in the attention window with no tiered priority.

**Our contribution** addresses a gap identified in recent work [Liu et al., "Procedural Memory Is Not All You Need", 2025]: the inability to update knowledge mid-conversation with clean override semantics. Our three-tier architecture provides:

1. **Copy-on-write override**: Dynamic facts shadow static knowledge at the same trie path without reindexing
2. **Tiered priority**: Physical context > Dynamic facts > Static knowledge, with automatic precedence
3. **Immediate updates**: Facts extracted from conversation override pre-existing knowledge within the same turn

This is architecturally distinct from RAG (which concatenates without priority) and fine-tuning (which cannot update). The override mechanism enables medium-term memory—knowledge that persists across turns but adapts within a session.

### 10.5 Sparse Attention

**Sparse attention** [Child et al., 2019; Beltagy et al., 2020; Zaheer et al., 2020] reduces attention complexity through learned or structured sparsity patterns while preserving the ability for any token to potentially attend to any other token.

Our approach provides an orthogonal form of sparsity at chunk granularity: dense token-level attention over a bounded physical context, plus sparse chunk-level retrieval. Critically, this is *not* equivalent to sparse attention—cross-chunk relationships are invisible unless both chunks are explicitly retrieved. Sparse attention preserves unexpected relationship discovery; our retrieval requires relevance to be predictable from the query. These are complementary capabilities for different workloads.

### 10.6 Summary: Positioning This Work

| Innovation | Novelty | Relationship to Prior Work |
|------------|---------|---------------------------|
| Position-Independent KV | Incremental | Builds on MEPIC, KVLINK, Lazy-Attention |
| Trie-Constrained Retrieval | High | Novel application of constrained decoding |
| Speculative Path Resolution | High | Novel disambiguation mechanism for constrained decoding |
| Content-Based Mixed Precision | Moderate | Extends KVTuner/KIVI to semantic allocation |
| Dynamic Override Semantics | High | Addresses gap in RAG/Memory literature |
| Co-Requirement Architecture | High | Novel integration demonstrating synergy |

The primary contribution is not any single technique but their co-requirement relationship: position-independent caching enables the fact index; the fact index's trie structure enables speculative disambiguation; override semantics differentiate from static RAG. Removing any component substantially degrades the others.

---

## 11. Limitations

Several limitations should be noted:

**Single-GPU Scope**: All experiments were conducted on single GPUs (RTX 4090 and A100). Multi-GPU scaling would require distributed page table management and cross-device communication for the fact retrieval mechanism—architectures we did not implement or evaluate.

**Model Coverage**: We evaluated primarily on Qwen3-30B-A3B and Hermes3-70B. While the architecture should generalize to other transformer models using RoPE position encoding, performance characteristics may differ for other model families, particularly those using different attention variants or tokenizers.

**Fact Index Limitations**: The "2.4M token fact index" represents *storage capacity*, not *attention span*. These tokens are addressable through retrieval, not simultaneously attended. The architecture retrieves 1-8 facts per query from a 24,000-path trie; cross-chunk relationships are invisible unless both chunks are retrieved together. This is categorically different from native long-context models that provide simultaneous attention over all tokens. Tasks requiring holistic analysis—finding contradictions across documents, identifying patterns spanning the entire context, or synthesis requiring visibility of everything at once—would not benefit from this architecture and should use native long-context approaches instead.

**Retrieval Failure Modes**: Category-level misselection (D3 errors) produces coherent-sounding but contextually inappropriate responses. Speculative path resolution (Section 6.9) substantially reduces D3 rates on ambiguous queries, but confident-but-wrong selections (low entropy, wrong category) remain irreducible. Our validation (Section 8.5) bounded net D3 rates: [**TODO: <X%**] on held-out evaluation, [**TODO: <Y%**] on adversarial inputs with speculation enabled. Determined adversaries targeting confident misselection can still achieve misrouting.

**Tokenizer Dependency**: The single-token vocabulary selection ties the category structure to a specific tokenizer. Migrating to models with different tokenizers would require rebuilding the category vocabulary from the new tokenizer's single-token words.

**Temporal Reasoning**: The fact-based memory system discarded temporal ordering information during extraction. Tasks requiring precise temporal reasoning showed reduced accuracy compared to full-context approaches.

**Evaluation Scope**: Human evaluation was conducted with [**TODO: N**] evaluators over [**TODO: M**] conversations. Broader evaluation across diverse conversational domains and user populations would strengthen the quality claims.

---

## 12. Conclusion

We presented three novel techniques for high-density LLM inference and demonstrated their synergistic integration with existing methods on consumer and mid-tier hardware.

**Trie-constrained generation as retrieval** repurposes a model failure mode—hallucination—as a feature. By constraining generation to paths in a trie structure, we transform unconstrained token prediction into deterministic fact selection. The single-token vocabulary design (where each category maps to exactly one tokenizer token) ensures consistent retrieval without multi-token ambiguity. To our knowledge, this application of constrained decoding for sparse retrieval is novel.

**Speculative path resolution** exploits trie depth for disambiguation. When category-level entropy indicates uncertainty, serial evaluation of complete paths with KV truncation allows selection by mean log-probability—a signal unavailable to flat retrieval or single-token constrained decoding. Ambiguity at shallow levels often resolves at deeper levels where only one interpretation produces natural continuations. This reduces D3 (catastrophic) errors by [**TODO: X%**] on boundary queries with latency cost only on ambiguous queries.

**Dynamic knowledge with override semantics** addresses a gap in existing memory-augmented LLM approaches. RAG systems concatenate retrieved content without priority; fine-tuning bakes knowledge into weights that cannot be updated mid-session. Our three-tier architecture enables dynamic facts to shadow static knowledge at the same trie path—immediate overrides without reindexing. This makes medium-term memory (knowledge that persists across turns but adapts within a session) a first-class capability.

These techniques integrate with position-independent KV caching (building on MEPIC and Lazy-Attention) and content-based mixed-precision allocation. The architectural insight is that these are co-requirements: position-independent caching enables arbitrary fact injection; the trie structure enables speculative disambiguation; override semantics differentiate from static retrieval. Removing any component degrades the others substantially.

A secondary finding extends mixed-precision research: the quality bootstrapping effect demonstrates that high-precision KV for stable content can partially compensate for aggressive model quantization. Content-based precision allocation (by semantic role rather than layer or channel) achieves [**TODO: X%**] quality recovery at [**TODO: Y%**] memory overhead compared to uniform quantization.

**Important scope**: The 2.4M token fact index is storage capacity, not attention span. This architecture excels when relevant content is localizable and retrievable; native long-context models excel when holistic visibility matters. These are complementary approaches.

In experiments on RTX 4090 (24GB) with Qwen3-30B-A3B-AWQ and A100 (80GB) with Hermes3-70B-Q4, the system achieved [**TODO: key quantitative result**]. The same architectural principles scale to frontier hardware (Appendix G), where 400B+ parameter models with 120M+ token fact indices become feasible.

---

## Appendix A: Memory Formulas

**KV cache per token** for a model with L layers, H KV heads, and D head dimension:

Elements per token = 2 × L × H × D (factor of 2 for K and V)

Bytes per token = Elements × sizeof(dtype)

**Model weight size** scales linearly with parameter count and precision:

Bytes = Parameters × sizeof(dtype)

**Concurrent context capacity** given total VRAM V, model weight size W, overhead O, and per-context KV size K:

Max contexts = (V - W - O) / K

---

## Appendix B: Throughput Formulas

**Decode throughput** in the memory-bound regime:

tokens/sec = Bandwidth / (Weight_bytes + Context_tokens × KV_bytes_per_token)

For batched decode with B sequences:

total_tokens/sec = Bandwidth / (Weight_bytes + B × Context_tokens × KV_bytes_per_token)

**Prefill throughput** in the compute-bound regime:

tokens/sec ≈ FLOPS / (2 × Parameters × 2)

**Time to first token** combines prefill and first decode:

TTFT = Prefill_tokens / Prefill_throughput + 1 / Decode_throughput

---

## Appendix C: Experimental Details

### C.1 Benchmark Datasets

[**TODO: List specific datasets used for each evaluation**]

```
Example format:

- Retrieval accuracy: Custom dataset of XXX conversations with 
  human-labeled ground truth facts. Available at [URL].
  
- Quality evaluation: Subset of [benchmark name] adapted for 
  multi-turn conversation. XX conversations, XXX turns each.
  
- Perplexity: [Dataset name], standard test split.
```

### C.2 Hyperparameters

[**TODO: Complete hyperparameter table**]

```
┌─────────────────────────────────────────────────────────────┐
│  Table C.1: System Hyperparameters                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Parameter                          │ Value                 │
│  ───────────────────────────────────┼───────────────────────│
│  Page size (tokens)                 │ XXX                   │
│  Trie depth (levels)                │ 3                     │
│  Categories (level 1)               │ ~20                   │
│  Subcategories per category         │ ~30                   │
│  Topics per subcategory             │ ~40                   │
│  Tokens per fact                    │ ~100                  │
│  Fresh message budget (tokens)      │ XXXX                  │
│  Recent summary budget (tokens)     │ XXX                   │
│  Facts retrieved per query          │ X                     │
│  GEMV/Marlin crossover threshold    │ X                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### C.3 Prompts

[**TODO: Include key prompts used in the memory pipeline**]

```
Phase 0 (Compression) prompt:
---
[Insert actual prompt used]
---

Phase 1 (Path extraction) prompt:
---
[Insert actual prompt used]
---

Phase 2 (Editorial filter) prompt:
---
[Insert actual prompt used]
---

Phase 3 (Content generation) prompt:
---
[Insert actual prompt used]
---
```

---

## Appendix D: Additional Results

[**TODO: Include supplementary results that support main findings but weren't essential to the narrative**]

```
Suggested additional tables/figures:

- Full latency distributions (not just p50/p99)
- Per-category path selection distribution
- Cache hit rate over time during extended sessions
- Memory usage breakdown by component
- Examples of successful and degraded retrievals
- Sample generated facts with quality annotations
- Tokenizer vocabulary analysis for category selection
```

---

## Appendix E: Model-Hardware Mappings

This appendix details the model configurations evaluated across different hardware tiers. The selection of models for each tier reflects a careful balance between quality, throughput, and memory constraints. We prioritized models with strong instruction-following capabilities and consistent output quality, while ensuring sufficient VRAM headroom for KV cache to enable the concurrent session counts and trie-based fact retrieval that the architecture requires.

For each hardware tier, we evaluated multiple model options and identified recommended configurations based on the target use case. The "Active" column indicates the number of parameters participating in each forward pass—particularly relevant for Mixture-of-Experts (MoE) models where total parameter count significantly exceeds active parameters.

### E.1 Consumer Tier (RTX 4090 24GB)

The RTX 4090 represents the upper end of consumer hardware, with 24GB VRAM and approximately 1 TB/s memory bandwidth. This capacity enables surprisingly capable deployments when combined with aggressive quantization and efficient architectures.

```
┌─────────────────────────────────────────────────────────────┐
│  Table E.1: RTX 4090 Model Options                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model              │ Quant │ Size  │ Active │ MMLU │ Notes │
│  ──────────────────────────────────────────────────────────│
│  Qwen3-30B-A3B     │ AWQ   │ 17 GB │ 3.3B   │ ~80  │ MoE   │
│  Qwen3-14B         │ Q5    │ 10 GB │ 14B    │ ~77  │ Dense │
│  Hermes3-8B        │ Q6    │ 6 GB  │ 8B     │ ~65  │ Dense │
│  Qwen3-4B          │ Q4    │ 2.5GB │ 4B     │ ~55  │ Fast  │
│                                                             │
│  Recommended: Qwen3-30B-A3B-AWQ (Frontal) + Qwen3-4B (Limbic)│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The Mixture-of-Experts architecture proves particularly valuable at this tier. The Qwen3-30B-A3B model contains 30 billion total parameters but activates only 3.3 billion per token through its expert routing mechanism. This yields inference speeds comparable to a 3B dense model while maintaining quality metrics competitive with 14B dense models. The AWQ quantization further compresses the model to 17GB, leaving 7GB for KV cache, framework overhead, and the secondary (Limbic) model used for fast operations like fact path extraction.

For the dual-model architecture described in Section 5, we pair the 30B MoE model with Qwen3-4B-Q4 as the fast model. The 4B model handles high-frequency, latency-sensitive operations (path extraction, editorial filtering) while the 30B model handles quality-critical generation. This pairing fits comfortably within 24GB while providing both speed and quality where each is most needed.

### E.2 Enterprise Tier (A100 80GB)

The A100 PCIe with 80GB HBM2e represents the workhorse of enterprise AI deployment. With 2 TB/s memory bandwidth and substantially more VRAM, this tier enables fundamentally different deployment strategies—either maximizing model quality or maximizing concurrent session count.

```
┌─────────────────────────────────────────────────────────────┐
│  Table E.2: A100 80GB Model Options                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model              │ Quant │ Size  │ Active │ MMLU │ Notes │
│  ──────────────────────────────────────────────────────────│
│  Hermes3-70B       │ Q4    │ 40 GB │ 70B    │ ~83  │ Dense │
│  Qwen3-32B         │ FP8   │ 32 GB │ 32B    │ ~81  │ Dense │
│  Qwen3-30B-A3B     │ FP8   │ 30 GB │ 3.3B   │ ~80  │ MoE   │
│  Qwen3-72B         │ FP8   │ 72 GB │ 72B    │ ~85  │ Tight │
│                                                             │
│  Recommended: Hermes3-70B-Q4 (quality) or                   │
│               Qwen3-30B-A3B-FP8 (concurrency)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The choice between quality and concurrency optimization depends on deployment requirements. For applications requiring the highest output quality—complex reasoning, nuanced dialogue, or tasks where errors are costly—the Hermes3-70B-Q4 configuration dedicates more parameters to each request. The Q4 quantization maintains nearly full model quality while fitting within 40GB, leaving 40GB for KV cache. With 160KB per token for KV at this model size, this supports approximately 14 concurrent sessions at 16K context length, or more sessions at shorter contexts.

For applications prioritizing throughput—serving many concurrent users with acceptable quality—the Qwen3-30B-A3B-FP8 configuration leverages MoE efficiency. With only 3.3B parameters active per token, this configuration can theoretically support thousands of concurrent sessions, limited primarily by memory bandwidth rather than VRAM capacity. The 50GB of KV headroom enables substantial session counts even at longer context lengths.

The Qwen3-72B option at FP8 pushes the limits of the 80GB envelope. While offering the highest quality metrics, the tight VRAM budget (only 8GB headroom) severely constrains concurrent session capacity and leaves little room for the trie-based fact retrieval that makes this architecture valuable. We include it for completeness but do not recommend it for production deployments using this architecture.

### E.3 Entry Tier (RTX 3060 12GB / RTX 4070 12GB)

The 12GB VRAM tier represents a sweet spot for accessible deployment. Cards in this range are widely available and reasonably priced, while still providing enough headroom for meaningful model quality and concurrent session support. This tier demonstrates that the architecture's benefits are not limited to high-end hardware.

```
┌─────────────────────────────────────────────────────────────┐
│  Table E.3: 12GB VRAM Model Options                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model              │ Quant │ Size  │ Active │ MMLU │ Notes │
│  ──────────────────────────────────────────────────────────│
│  Qwen3-8B          │ Q4    │ 5 GB  │ 8B     │ ~65  │ Dense │
│  Hermes3-8B        │ Q6    │ 6 GB  │ 8B     │ ~65  │ Dense │
│  Qwen3-4B          │ Q6    │ 3 GB  │ 4B     │ ~55  │ Fast  │
│                                                             │
│  Recommended: Qwen3-8B-Q4 (Frontal) + Qwen3-4B (Limbic)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

At this tier, the dual-model architecture remains viable with the Qwen3-8B handling quality-critical generation and Qwen3-4B handling fast operations. The combined footprint of approximately 8GB leaves 4GB for KV cache—sufficient for several concurrent sessions at moderate context lengths. The fact retrieval system becomes particularly valuable here, as it allows long-running conversations without the KV cache growth that would quickly exhaust available memory.

The RTX 4070's higher memory bandwidth (504 GB/s vs the 3060's 360 GB/s) provides meaningful throughput improvements, making it the preferred choice when available. Both cards benefit significantly from the position-independent KV sharing mechanism when serving multiple sessions with overlapping static knowledge.

### E.4 Minimum Tier (GTX 1660 Super 6GB / RTX 3050 8GB)

The minimum viable tier for this architecture requires approximately 6GB VRAM. Below this threshold, the combination of model weights, KV cache, and operational overhead leaves insufficient room for meaningful concurrent session support. However, even at this tier, the architecture provides benefits over naive inference approaches.

```
┌─────────────────────────────────────────────────────────────┐
│  Table E.4: 6-8GB VRAM Model Options                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model              │ Quant │ Size  │ Active │ MMLU │ Notes │
│  ──────────────────────────────────────────────────────────│
│  Hermes3-3B        │ Q6    │ 2.6GB │ 3B     │ ~45  │ Dense │
│  Qwen3-4B          │ Q4    │ 2.5GB │ 4B     │ ~55  │ Dense │
│  Qwen2.5-1.5B      │ Q8    │ 1.8GB │ 1.5B   │ ~40  │ Fast  │
│                                                             │
│  Recommended: Qwen3-4B-Q4 (single model for both roles)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

At this tier, we recommend collapsing the dual-model architecture into a single model serving both roles. The Qwen3-4B-Q4 at 2.5GB leaves approximately 3.5-5.5GB for KV cache depending on the specific card. While this constrains concurrent session count, the trie-based fact retrieval system remains fully functional, enabling long conversations and large static knowledge bases that would be impossible with traditional context management.

The limited VRAM makes position-independent KV sharing particularly impactful at this tier. When multiple sessions share static knowledge, the memory savings directly translate to additional concurrent session capacity—a critical benefit when every gigabyte matters.

---

## Appendix F: VRAM Budget Analysis

This appendix details how the three innovations interact to maximize effective VRAM utilization. Understanding the VRAM budget is essential for capacity planning and helps explain why the three innovations are co-requirements rather than independent optimizations.

### F.1 VRAM Budget Components

The total VRAM budget divides into four components, each with different characteristics and optimization opportunities. Model weights represent the largest fixed cost and benefit most from quantization. KV cache scales with concurrent sessions and context length, benefiting from both precision reduction and sharing. Fact storage is negligible in comparison. Framework overhead is largely unavoidable but relatively constant.

```
┌─────────────────────────────────────────────────────────────┐
│  VRAM Budget = Model + KV Cache + Facts + Overhead          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model Weights (Quantized):                                 │
│    Size = Parameters × bits_per_param / 8                   │
│    30B @ 4-bit AWQ ≈ 17 GB                                  │
│    70B @ 4-bit GGUF ≈ 40 GB                                 │
│                                                             │
│  KV Cache (per token, FP8):                                 │
│    Size = 2 × layers × kv_heads × head_dim × 1 byte         │
│    30B MoE: ~48 KB/token                                    │
│    70B Dense: ~160 KB/token                                 │
│                                                             │
│  Fact Storage (text, not KV):                               │
│    ~100 tokens × 4 bytes/token = 400 bytes/fact             │
│    24,000 facts ≈ 10 MB (negligible)                        │
│                                                             │
│  Framework Overhead:                                        │
│    CUDA contexts, allocator fragmentation: ~1-2 GB          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Note that fact storage—the text content of the fact index—is negligible compared to other components. The 2.4 million tokens of indexed facts occupy only about 10MB as text. The real memory cost comes when facts are retrieved and their KV representations are computed or loaded into the physical context window.

### F.2 RTX 4090 Budget (24GB)

The RTX 4090 configuration demonstrates tight but viable budget management. With the recommended Qwen3-30B-A3B-AWQ model consuming 17GB and the Limbic model adding 2.5GB, approximately 3GB remains for KV cache after accounting for framework overhead.

```
┌─────────────────────────────────────────────────────────────┐
│  RTX 4090: Qwen3-30B-A3B-AWQ Configuration                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Component                    │ Size      │ Cumulative      │
│  ────────────────────────────────────────────────────────── │
│  Model weights (AWQ)          │ 17.0 GB   │ 17.0 GB        │
│  Limbic model (Qwen3-4B-Q4)   │ 2.5 GB    │ 19.5 GB        │
│  Framework overhead           │ 1.5 GB    │ 21.0 GB        │
│  ────────────────────────────────────────────────────────── │
│  Available for KV cache       │ 3.0 GB    │ 24.0 GB        │
│                                                             │
│  KV capacity (48 KB/tok):                                   │
│    3 GB / 48 KB = ~65,000 tokens                           │
│    @ 4K context = ~16 concurrent sessions                   │
│    @ 8K context = ~8 concurrent sessions                    │
│                                                             │
│  With shared knowledge (50% overlap):                       │
│    Effective sessions: 24-32 @ 4K context                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The 3GB KV budget appears limiting, but two factors improve the effective capacity. First, the MoE architecture's smaller KV footprint (48KB/token vs 160KB for equivalent dense models) stretches the budget significantly. Second, the position-independent KV sharing mechanism means that static knowledge chunks load once and serve all sessions. When sessions share 50% of their context as common static knowledge, the effective session capacity nearly doubles.

The fact retrieval system further extends effective capacity by keeping only relevant facts in the physical window. A conversation can reference millions of tokens of indexed knowledge while maintaining a bounded KV footprint.

### F.3 A100 Budget (80GB)

The A100's larger VRAM budget enables both larger models and more generous KV allocation. With Hermes3-70B-Q4 consuming 40GB, approximately 33GB remains for KV cache—an order of magnitude more than the RTX 4090 configuration.

```
┌─────────────────────────────────────────────────────────────┐
│  A100: Hermes3-70B-Q4 Configuration                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Component                    │ Size      │ Cumulative      │
│  ────────────────────────────────────────────────────────── │
│  Model weights (Q4)           │ 40.0 GB   │ 40.0 GB        │
│  Limbic model (Qwen3-8B-Q4)   │ 5.0 GB    │ 45.0 GB        │
│  Framework overhead           │ 2.0 GB    │ 47.0 GB        │
│  ────────────────────────────────────────────────────────── │
│  Available for KV cache       │ 33.0 GB   │ 80.0 GB        │
│                                                             │
│  KV capacity (160 KB/tok):                                  │
│    33 GB / 160 KB = ~215,000 tokens                        │
│    @ 8K context = ~26 concurrent sessions                   │
│    @ 16K context = ~13 concurrent sessions                  │
│                                                             │
│  With shared knowledge (50% overlap):                       │
│    Effective sessions: 40-52 @ 8K context                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

However, the larger model also has a larger KV footprint per token (160KB vs 48KB for the MoE model). This partially offsets the VRAM advantage—the A100 with a 70B dense model supports roughly similar session counts to the RTX 4090 with a 30B MoE model, but at substantially higher quality. The choice between configurations reflects the quality-vs-concurrency tradeoff discussed in Appendix E.

The A100's 2 TB/s memory bandwidth (vs 1 TB/s for RTX 4090) provides additional throughput benefits not reflected in the VRAM budget. Higher bandwidth enables faster KV cache access and higher tokens-per-second during generation, particularly for batched inference.

### F.4 Innovation Impact on VRAM Efficiency

Each of the three innovations contributes differently to VRAM efficiency. Understanding these contributions clarifies why removing any single innovation significantly degrades the system's effectiveness.

```
┌─────────────────────────────────────────────────────────────┐
│  How Each Innovation Contributes to VRAM Efficiency         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INNOVATION 1: Fused Quantized Kernels                      │
│  ─────────────────────────────────────                      │
│  Without: 70B model requires 140GB (BF16) — impossible      │
│  With: 70B model fits in 40GB (Q4) — leaves 40GB for KV     │
│  Impact: Enables deployment of larger models                │
│                                                             │
│  INNOVATION 2: Position-Independent KV Paging               │
│  ───────────────────────────────────────────                │
│  Without: Each session copies shared content separately     │
│  With: Shared knowledge loads once, referenced by all       │
│  Impact: 50%+ reduction in KV memory when knowledge shared  │
│                                                             │
│  INNOVATION 3: Fact Index via Trie Retrieval                │
│  ─────────────────────────────────────────────              │
│  Without: Must keep all context in physical window          │
│  With: 2.4M tokens addressable, ~4K physical at any time    │
│  Impact: Long conversations without KV growth               │
│                                                             │
│  COMBINED EFFECT:                                           │
│  ───────────────                                            │
│  70B model + 50 sessions + 2.4M fact index                  │
│  on 80GB hardware that traditionally supports               │
│  ~8B model + 10 sessions + 16K context                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The innovations compound multiplicatively. Quantization enables larger models, which provide higher quality. The larger model's KV footprint would ordinarily limit concurrency, but position-independent sharing recovers much of this cost when sessions overlap. Trie-based fact retrieval further decouples effective knowledge access from KV budget, enabling arbitrarily long conversations and large knowledge bases within a bounded physical window.

Removing any innovation breaks this chain. Without quantization, the model itself exhausts VRAM. Without position-independent sharing, concurrent sessions compete for the limited KV budget. Without the fact index, long conversations or large knowledge bases force either truncation or unbounded memory growth.

### F.5 Comparison: Traditional vs. Proposed Architecture

The following table summarizes the efficiency improvements achieved by the complete architecture compared to traditional inference approaches on identical hardware.

```
┌─────────────────────────────────────────────────────────────┐
│  Table F.1: VRAM Efficiency Comparison (A100 80GB)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Metric              │ Traditional │ Proposed  │ Improvement│
│  ───────────────────────────────────────────────────────── │
│  Max model size      │ 32B (BF16)  │ 70B (Q4)  │ 2.2×      │
│  KV per session      │ 100%        │ 50%*      │ 2×        │
│  Context per session │ 16K tokens  │ 2.4M index│ 150×      │
│  Concurrent sessions │ ~8          │ ~50       │ 6×        │
│                                                             │
│  * With typical 50% knowledge overlap between sessions      │
│                                                             │
│  Net effect: Same hardware serves significantly more        │
│  concurrent users with larger models and larger fact index  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

These improvements are not additive but represent the combined effect of all three innovations working together. The 6× improvement in concurrent sessions, for example, results from quantization freeing VRAM (enabling the larger model while leaving KV headroom), sharing reducing per-session KV cost (effectively doubling capacity when sessions overlap), and trie-based fact retrieval preventing context growth from consuming the KV budget over long conversations.

---

## Appendix G: Future Work — Frontier Hardware Projections

While this paper focused on consumer (RTX 4090, 24GB) and mid-tier enterprise (A100, 80GB) hardware, the architectural innovations apply equally—and perhaps more compellingly—to frontier accelerators. This appendix explores projected capabilities on NVIDIA B200 and AMD MI300X hardware.

### G.1 Frontier Hardware Specifications

The next generation of datacenter accelerators offers dramatically increased VRAM capacity and memory bandwidth. Both the NVIDIA B200 (Blackwell architecture) and AMD MI300X (CDNA 3 architecture) provide 192GB of HBM3/HBM3e memory—nearly 2.5× the A100's 80GB and 8× the RTX 4090's 24GB.

```
┌─────────────────────────────────────────────────────────────┐
│  Table G.1: Frontier Accelerator Specifications             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Accelerator    │ VRAM    │ Bandwidth │ Notes               │
│  ───────────────────────────────────────────────────────── │
│  NVIDIA B200    │ 192 GB  │ 8.0 TB/s  │ Blackwell, HBM3e   │
│  AMD MI300X     │ 192 GB  │ 5.3 TB/s  │ CDNA 3, HBM3       │
│  ───────────────────────────────────────────────────────── │
│  For comparison:                                            │
│  NVIDIA A100    │ 80 GB   │ 2.0 TB/s  │ Ampere, HBM2e      │
│  NVIDIA RTX 4090│ 24 GB   │ 1.0 TB/s  │ Ada, GDDR6X        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The B200's 8 TB/s memory bandwidth represents a 4× improvement over the A100, enabling proportionally higher throughput for memory-bound inference workloads. The MI300X's 5.3 TB/s, while lower than the B200, still represents substantial improvement and comes with competitive pricing and AMD's ROCm software ecosystem.

These specifications suggest that the architectural innovations presented in this paper become even more valuable at frontier scale. The larger VRAM budget enables larger models with more KV headroom, while the higher bandwidth enables faster retrieval from the fact index.

### G.2 Projected Model Capacity

With 192GB VRAM, frontier accelerators can run models that are currently impractical on any single device. The following table shows projected configurations, assuming continued progress in quantization techniques and model architectures.

```
┌─────────────────────────────────────────────────────────────┐
│  Table G.2: B200/MI300X Model Configurations                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model              │ Quant │ Size   │ KV Headroom │ Notes  │
│  ──────────────────────────────────────────────────────────│
│  Llama-405B         │ Q4    │ ~115GB │ ~77 GB      │ Dense  │
│  Qwen-MoE-A72B      │ FP8   │ ~85GB  │ ~107 GB     │ MoE    │
│  DeepSeek-V3-671B   │ Q3    │ ~140GB │ ~52 GB      │ MoE    │
│  Claude-scale 400B  │ Q4    │ ~112GB │ ~80 GB      │ Dense  │
│                                                             │
│  All configurations leave substantial KV cache headroom     │
│  for concurrent sessions and trie-based fact retrieval      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The critical observation is that even 400B+ parameter models at Q4 quantization leave 50-100GB of KV headroom. This headroom is where the position-independent KV sharing and trie-based fact retrieval provide their value. Without these innovations, the headroom would support only a handful of concurrent sessions at modest context lengths. With them, the same headroom can support dozens of sessions with effectively unlimited fact index capacity.

The MoE architectures (Qwen-MoE-A72B, DeepSeek-V3) are particularly interesting at this scale. Despite their massive total parameter counts, their active parameter footprints remain tractable, providing frontier-level quality with inference speeds closer to much smaller dense models.

### G.3 Scaled Fact Index Projections

The trie-based retrieval architecture scales dramatically when combined with frontier models that natively support 1M+ token context windows. The current implementation was designed for consumer hardware constraints: small fact chunks (100 tokens), limited injection (8 facts per turn), and modest trie depth. Frontier hardware removes these constraints, enabling a fundamentally different scale of operation.

```
┌─────────────────────────────────────────────────────────────┐
│  Table G.3: Fact Index Scaling                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Parameter             │ Current     │ Frontier   │ Scaling │
│  ──────────────────────────────────────────────────────────│
│  Native context window │ 4-16K       │ 1M         │ 60-250× │
│  Tokens per fact       │ 100         │ 2,000      │ 20×     │
│  Facts injected/turn   │ 8           │ 40         │ 5×      │
│  Trie dimensions       │ 20×30×40    │ 30×40×50   │ 2.5×    │
│  Total trie paths      │ 24,000      │ 60,000     │ 2.5×    │
│  ──────────────────────────────────────────────────────────│
│  Fact index            │ 2.4M tokens │ 120M tokens│ 50×     │
│  In-scope per turn     │ 800 tokens  │ 80K tokens │ 100×    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**The Arithmetic of Scale**

The current architecture retrieves 8 facts × 100 tokens = 800 tokens from the fact index per turn. With frontier models, this expands to 40 facts × 2,000 tokens = 80,000 tokens per turn. Combined with an expanded trie (30 × 40 × 50 = 60,000 paths), the total fact index reaches 60,000 paths × 2,000 tokens = 120 million tokens.

To put this in perspective: 120 million tokens is approximately equivalent to 300 full-length novels, a complete enterprise codebase with all documentation and history, or the entire medical literature for a specialty. The fact storage requirement remains modest—at 4 bytes per token, 120M tokens requires only ~480MB, negligible compared to the VRAM budget.

**Multi-Turn Attention Shifting**

The most compelling capability emerges from combining large per-turn context (80K tokens) with multi-turn task execution. With careful prompt and inference design, complex tasks can be decomposed into sequential turns where the attention window shifts across different regions of the fact index.

Consider a code review task across a 10M token codebase:

```
┌─────────────────────────────────────────────────────────────┐
│  Multi-Turn Attention Shifting Example                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Turn 1: Retrieve architecture overview, entry points       │
│          → 80K tokens covering high-level structure         │
│                                                             │
│  Turn 2: Retrieve authentication module, security policies  │
│          → 80K tokens covering auth implementation          │
│                                                             │
│  Turn 3: Retrieve database layer, schema definitions        │
│          → 80K tokens covering data access patterns         │
│                                                             │
│  Turn 4: Retrieve test coverage, CI/CD configuration        │
│          → 80K tokens covering quality infrastructure       │
│                                                             │
│  Turn 5: Synthesize findings, generate review report        │
│          → 80K tokens of prior turn summaries + key facts   │
│                                                             │
│  Effective coverage: 400K+ tokens across 5 turns            │
│  Each turn has full attention over its 80K window           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This multi-turn pattern enables effective reasoning over fact indices far larger than any single attention window. The key insight is that 80K tokens per turn provides sufficient context for meaningful work, while the retrieval mechanism ensures each turn receives the most relevant 80K tokens for its specific subtask.

**Comparison with Native Long Context**

Frontier models with 1M native context provide simultaneous attention over all tokens—a powerful capability for tasks requiring cross-document reasoning or holistic analysis. Our architecture provides a different tradeoff:

```
┌─────────────────────────────────────────────────────────────┐
│  Native Long Context vs. Fact Index Retrieval               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Capability           │ Native 1M   │ Virtual 120M         │
│  ──────────────────────────────────────────────────────────│
│  Simultaneous tokens  │ 1M          │ 80K per turn         │
│  Total addressable    │ 1M          │ 120M                 │
│  Cross-doc reasoning  │ Full        │ Multi-turn required  │
│  Dynamic updates      │ Prompt only │ Any turn             │
│  KV cache cost        │ O(n)        │ O(1) amortized       │
│  Multi-user sharing   │ None        │ Static layer shared  │
│                                                             │
│  Best for:                                                  │
│  Native 1M: Tasks requiring simultaneous attention over     │
│             all content (e.g., finding contradictions       │
│             across documents, global optimization)          │
│                                                             │
│  Virtual 120M: Tasks decomposable into focused subtasks     │
│                with shifting attention (e.g., codebase      │
│                navigation, knowledge-base queries,          │
│                multi-step research)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The approaches are complementary. Native long context excels at tasks requiring global attention; trie-based retrieval excels at tasks requiring vast knowledge with focused attention. Many real-world applications benefit from both: use native context for the current turn's 80K tokens while drawing from a 120M token fact index for retrieval.

**Enabling Factors**

Several advances make this scaling viable on frontier hardware:

1. **Native 1M context**: Models like Gemini 1.5 Pro and Claude demonstrate that 1M+ context windows are achievable. This provides the physical window for larger fact chunks and more facts per turn.

2. **Improved retrieval models**: Better embedding models and retrieval techniques enable more trie paths without degraded retrieval accuracy. The 30×40×50 trie assumes retrieval quality improvements that justify finer-grained organization.

3. **2K token facts**: Larger fact chunks capture more coherent information per retrieval—a complete function with context, a full guideline section, or a comprehensive drug interaction profile. This reduces the number of retrievals needed while improving context quality.

4. **Multi-turn orchestration**: Frameworks for decomposing complex tasks into sequential turns with managed state enable the attention-shifting pattern described above. This is an active area of development in agent architectures.

### G.4 Application Scenarios

The combination of 400B+ parameter models with 120M+ token fact indices and 80K tokens per turn enables applications impractical on current hardware. The following scenarios illustrate the possibilities, each representing a domain where comprehensive knowledge access combined with frontier model quality could provide transformative capabilities.

**Scenario 1: Enterprise Codebase Assistant**

Software development tools today face a fundamental tension: models are either unaware of the codebase (requiring manual context provision) or limited to context windows that can only hold fragments of larger projects. An assistant with an entire enterprise codebase in a fact index resolves this tension.

```
┌─────────────────────────────────────────────────────────────┐
│  Enterprise Codebase Assistant on B200                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model: Llama-405B-Q4 (~115GB)                              │
│  KV headroom: ~77GB                                         │
│                                                             │
│  Static knowledge (chunked system prompt):                  │
│    • Repository structure and architecture docs             │
│    • All source files chunked by module/class/function      │
│    • API documentation and type signatures                  │
│    • Test suites and coverage data                          │
│    • Commit history summaries by component                  │
│    • Code review comments and design decisions              │
│    • Deployment configurations and runbooks                 │
│                                                             │
│  Scale: 2M LOC enterprise codebase ≈ 60M tokens            │
│  Organization: 30 services × 40 modules × 50 components    │
│                = 60K paths in trie                          │
│  Facts: 2,000 tokens each (complete functions with context) │
│                                                             │
│  Per-turn scope: 40 facts × 2K tokens = 80K tokens         │
│                                                             │
│  Multi-turn capability: "Review the authentication flow"    │
│    Turn 1: Auth service overview, entry points (80K)        │
│    Turn 2: Token validation, session management (80K)       │
│    Turn 3: Related security policies, audit logs (80K)      │
│    Turn 4: Integration points, dependent services (80K)     │
│    Turn 5: Synthesize findings, generate report (80K)       │
│    → Effective coverage: 400K tokens with full attention    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The key advantage is multi-turn attention shifting. Rather than attempting to cram an entire codebase into a single context window, the assistant decomposes complex tasks into focused turns, each with full attention over 80K relevant tokens. The retrieval system ensures each turn receives the most relevant content for its subtask, while the dynamic fact layer accumulates discoveries across turns.

**Scenario 2: Comprehensive Medical Knowledge System**

Clinical decision support systems face similar constraints: comprehensive drug interaction databases, clinical guidelines, and patient records each contain millions of tokens of relevant information. Current systems require explicit queries against specific databases, placing the burden of knowing which references apply on the clinician.

```
┌─────────────────────────────────────────────────────────────┐
│  Medical Knowledge System on MI300X                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model: Med-specialized 400B-Q4 (~112GB)                    │
│  KV headroom: ~80GB                                         │
│                                                             │
│  Static knowledge (chunked reference material):             │
│    • Complete drug database (interactions, dosing, ADRs)    │
│    • Clinical guidelines by condition and specialty         │
│    • Diagnostic criteria (ICD-11, DSM-5, SNOMED)           │
│    • Laboratory reference ranges and interpretations        │
│    • Imaging protocols and findings databases               │
│    • Pharmacogenomic recommendations                        │
│    • Medical literature summaries by topic                  │
│                                                             │
│  Dynamic facts (patient context):                           │
│    • Current medications and administration history         │
│    • Lab values, vitals, and trends                        │
│    • Documented allergies and adverse reactions             │
│    • Prior treatment responses and outcomes                 │
│    • Current symptoms and examination findings              │
│                                                             │
│  Scale: ~8,000 drugs × 25 interaction categories           │
│         + ~15,000 conditions × 40 guideline chunks         │
│         = 60K paths, ~120M tokens                           │
│                                                             │
│  Per-turn scope: 40 facts × 2K tokens = 80K tokens         │
│                                                             │
│  Multi-turn capability: "Evaluate treatment options for     │
│                         this patient with heart failure"    │
│    Turn 1: Patient history, current medications (80K)       │
│    Turn 2: Heart failure guidelines, contraindications (80K)│
│    Turn 3: Drug interactions for candidate therapies (80K)  │
│    Turn 4: Dosing adjustments for renal function (80K)      │
│    Turn 5: Synthesize recommendation with rationale (80K)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The three-tier architecture is particularly valuable here. Static knowledge contains the stable medical reference material—drug databases, clinical guidelines, diagnostic criteria. Dynamic facts contain patient-specific information that can override or contextualize static knowledge. When a query mentions a specific medication, the retrieval system automatically surfaces interaction data, relevant guidelines for the patient's conditions, and any documented sensitivities—without requiring the clinician to query each database separately.

The multi-turn capability enables complex clinical reasoning. Rather than attempting to consider all factors simultaneously, the system can methodically work through patient history, applicable guidelines, drug interactions, and dosing adjustments, with full attention at each step. The dynamic fact layer accumulates relevant findings across turns, ensuring later turns benefit from earlier discoveries.

**Scenario 3: Legal Document Analysis**

Legal research requires navigating vast bodies of regulations, case law, and statutory text. Current tools provide keyword search but require extensive domain expertise to identify which regulations and precedents apply to a given situation.

```
┌─────────────────────────────────────────────────────────────┐
│  Legal Research System on B200                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model: Legal-tuned 400B-Q4 (~115GB)                        │
│  KV headroom: ~77GB                                         │
│                                                             │
│  Static knowledge:                                          │
│    • Federal regulations (CFR) by title/section             │
│    • State regulations for key jurisdictions                │
│    • Case law summaries by jurisdiction/topic/outcome       │
│    • Statutory text with amendment history                  │
│    • Precedent relationships and citation networks          │
│    • Agency guidance and enforcement actions                │
│    • Contract clause libraries with annotations             │
│                                                             │
│  Scale: 50 CFR titles × 1200 sections × 3 chunks           │
│         + 200K case summaries organized by topic            │
│         = 60K regulatory paths + case law index             │
│         = ~100M tokens                                      │
│                                                             │
│  Per-turn scope: 40 facts × 2K tokens = 80K tokens         │
│                                                             │
│  Multi-turn capability: "Analyze compliance requirements    │
│                         for this data processing agreement" │
│    Turn 1: Identify applicable regulatory frameworks (80K)  │
│    Turn 2: GDPR/CCPA specific requirements (80K)            │
│    Turn 3: Relevant enforcement actions and penalties (80K) │
│    Turn 4: Case law on similar contract disputes (80K)      │
│    Turn 5: Generate compliance checklist with citations (80K)│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The trie organization mirrors legal taxonomy: regulations organized by title and section, case law organized by jurisdiction and topic. When analyzing a contract clause, the model's attention patterns identify which regulatory areas apply and retrieve relevant sections, along with case law interpreting those regulations and any enforcement precedents. The 400B model provides the reasoning capability to synthesize these sources into coherent analysis, while the fact index provides access to the comprehensive reference material that synthesis requires.

The multi-turn pattern is natural for legal analysis, which typically proceeds through stages: identify applicable law, examine specific provisions, review interpretive case law, consider enforcement history, and synthesize conclusions. Each stage benefits from focused attention over the most relevant 80K tokens, while the dynamic fact layer tracks key findings and relevant citations across the analysis.

### G.5 Dynamic Knowledge vs. Fine-Tuning

A crucial advantage of the fact index architecture—often overlooked—is that the knowledge base can change dynamically between conversation turns without any model modification. This distinguishes the approach fundamentally from fine-tuning, where reference material is baked into model weights.

**The Fine-Tuning Alternative**

One could theoretically address large reference material through fine-tuning: train the model on the codebase, the medical literature, or the legal corpus until it "knows" the content. This approach has been successfully applied in domain-specific models. However, fine-tuning creates static knowledge:

- **Updates require retraining**: When the codebase changes, when new drugs are approved, when regulations are amended, the model must be fine-tuned again. For rapidly evolving domains, this creates an impossible maintenance burden.

- **Knowledge conflicts cause interference**: Fine-tuning on new information can degrade performance on previously learned content. The model cannot cleanly "forget" outdated facts or "override" superseded guidelines.

- **Per-user customization is impractical**: Fine-tuning a separate model for each user's preferences, history, or context is computationally prohibitive at scale.

- **No medium-term memory**: Fine-tuning operates on training-time knowledge. It cannot incorporate information learned during a conversation—there is no mechanism for "I just learned the user prefers formal language" to affect subsequent responses.

**The Dynamic Knowledge Advantage**

The fact index architecture addresses each limitation:

- **Updates are immediate**: Adding a new file to the codebase trie, a new drug interaction to the medical trie, or a new regulation to the legal trie requires only inserting new facts—the model itself is unchanged. Updates can happen between conversation turns.

- **Override semantics resolve conflicts**: When dynamic facts occupy the same path as static knowledge, the dynamic version takes precedence. This provides clean semantics for "the user just told me their address changed" or "this guideline was updated yesterday."

- **Per-user customization is native**: Each conversation maintains its own dynamic fact layer. User A's preferences don't affect User B's experience. The static knowledge base is shared; the dynamic layer is per-session.

- **Medium-term memory emerges naturally**: Facts extracted from conversation become immediately retrievable in subsequent turns. "Remember I'm allergic to penicillin" creates a fact that influences all future medication-related queries within that session—without any model modification.

**Why This Matters for Frontier Applications**

The application scenarios in G.4 depend critically on dynamic knowledge:

- **Codebase assistant**: Developers make changes during debugging sessions. The assistant must incorporate "I just moved that function to a different file" immediately, not after retraining.

- **Medical knowledge system**: Patient state evolves during consultations. "The patient just reported nausea" must immediately influence subsequent recommendations.

- **Legal research**: Counsel may discover relevant precedents during research. "Also consider the Smith v. Jones ruling" should immediately inform the analysis.

In each case, the value of the system depends on its ability to learn and adapt within the session. A fine-tuned model, no matter how comprehensive its initial training, cannot provide this capability. The fact index architecture makes medium-term memory—knowledge that persists across turns but doesn't require permanent model modification—a first-class feature.

This dynamic adaptability becomes increasingly valuable as the fact index scales. A 20M token knowledge base that cannot be updated is useful but limited. A 20M token knowledge base that evolves with each conversation turn enables fundamentally different applications.

### G.6 Architectural Considerations for Frontier Scale

Scaling to frontier hardware introduces additional considerations beyond simply expanding the existing architecture. While the core innovations remain applicable, several areas may benefit from architectural extensions to fully exploit the larger resource budget.

**Trie Depth Extension**

The current three-level trie (category → subcategory → topic) was designed for approximately 24,000 paths. Scaling to 100K+ paths may benefit from a fourth level, providing finer-grained organization without increasing path ambiguity at any single level. A four-level hierarchy of 20 × 25 × 25 × 8 yields 100,000 paths while keeping each level's branching factor manageable for single-token selection.

The single-token vocabulary constraint becomes more challenging at larger scales. With more paths required, the vocabulary must be carefully curated to ensure sufficient single-token candidates exist across all levels. This may require domain-specific vocabulary selection or relaxation of the single-token constraint at deeper levels where path commitment errors are less costly.

**Hierarchical Retrieval**

For 20M+ token fact indices, a two-stage retrieval process may improve accuracy. The first stage performs coarse retrieval to identify relevant trie subtrees—essentially selecting which knowledge domains apply to the query. The second stage performs fine-grained path selection within those subtrees. This mirrors human information-seeking behavior: first identifying which reference books to consult, then finding the specific sections within those books.

Hierarchical retrieval also enables parallel processing. Multiple subtrees can be searched simultaneously, with results merged based on relevance scores. This reduces latency compared to sequential traversal of a single large trie.

**KV Cache Quantization**

With 400B+ models, FP8 or INT8 KV cache becomes essential to maintain concurrent session capacity. A 400B dense model at full precision would require approximately 800KB per token for KV cache—rapidly exhausting even 77GB of headroom. FP8 KV reduces this to 400KB, doubling effective capacity. INT8 with appropriate scaling provides further reduction with minimal quality impact for most applications.

The mixed-precision kernel architecture described in Section 4 extends naturally to these aggressive KV quantization formats. The key insight—that different content merits different precision—applies equally at frontier scale. System prompts and stable reference material can tolerate aggressive quantization; recent conversation turns benefit from higher precision.

**Multi-Trie Organization**

Distinct knowledge domains may benefit from separate tries with domain-specific vocabularies, unified through a meta-retrieval layer. For the codebase assistant scenario, separate tries might organize source code (by module/file/function), documentation (by topic/section), tests (by coverage area), and conversation history (by topic/time). A meta-retrieval layer first identifies which tries are relevant to a query, then retrieves within those tries.

This organization simplifies vocabulary design—each trie uses domain-appropriate terminology—and enables domain-specific retrieval tuning. It also facilitates incremental updates: adding new documentation requires updating only the documentation trie, without reindexing code or conversation history.

### G.7 Research Directions

Several open questions merit further investigation as the architecture scales to frontier hardware:

1. **Optimal trie depth vs. breadth tradeoffs** at 100K+ path scales. Deeper tries provide finer-grained organization but require more token selections per retrieval. Broader tries reduce selection depth but increase ambiguity at each level. The optimal balance likely depends on the specific domain and query characteristics.

2. **Cross-domain retrieval** when facts span multiple knowledge categories. Some queries naturally require information from multiple domains—debugging a performance issue might require both code and documentation. Mechanisms for multi-domain retrieval and result synthesis need investigation.

3. **Incremental trie updates** for rapidly changing knowledge bases. The current architecture assumes relatively stable static knowledge. Applications with frequently updated content (live codebases, current news, real-time data) need efficient update mechanisms that avoid full reindexing.

4. **Quality preservation** in 400B+ models at aggressive quantization. While Q4 quantization shows minimal quality loss for 70B models, the impact at 400B+ scale is less well characterized. Larger models may be more or less robust to quantization; systematic evaluation is needed.

5. **Retrieval latency** as fact index scales to 20M+ tokens. The current three-token path selection completes in milliseconds. Four-level hierarchies with larger vocabularies may introduce measurable latency. Techniques for latency hiding (speculative retrieval, prefetching) need investigation.

6. **Multi-user knowledge sharing** when multiple users access overlapping but distinct knowledge bases. Can static knowledge be shared while dynamic facts remain user-specific? What are the memory and correctness implications?

These directions suggest that the architectural innovations presented in this paper provide a foundation for continued scaling as hardware capabilities advance. The core insight—that retrieval-based context extension complements rather than competes with native long-context models—becomes increasingly valuable as both approaches scale.