# Adaptive Block-Oriented Quantization for LLM Inference

## INT4/INT8 MMA Pipeline with Token-Oriented KV Cache

**Version:** 5.0  
**Date:** February 2026

---

## Executive Summary

### The Problem

Running large language models efficiently on consumer GPUs requires aggressive quantization of both weights and activations. The KV cache — which stores key and value tensors for all previous tokens — becomes the dominant memory consumer for long contexts and high concurrency. A naive approach might simply quantize everything to 4-bit integers, but this destroys model quality due to a well-documented phenomenon: activation outliers.

Research from SmoothQuant, LLM.int8(), KIVI, and AWQ has established that 1-2% of embedding dimensions consistently carry values 10-100× larger than typical activations. These outlier channels appear in the same dimensions across all tokens — they are a property of the model, not the input. When outliers share a quantization block with normal values, the block's scale must accommodate the outlier magnitude, crushing the normal values into a tiny fraction of the available precision. The result: what should be 8-bit quantization delivers only 4-5 effective bits.

### The Solution

This specification addresses the outlier problem through **token-oriented blocking**: instead of grouping 32 consecutive dimensions into a quantization block (traditional approach), we group 32 consecutive tokens for a single dimension. Each channel gets its own quantization scale. Outlier channels get large scales; normal channels get small scales. No contamination occurs.

However, token-oriented blocking creates a new problem: misalignment with matrix multiplication. Modern GPUs achieve peak throughput using tensor core MMA operations, which require the quantization block to align with the reduction dimension. Token-oriented blocks run perpendicular to most reduction dimensions, preventing efficient INT4/INT8 MMA.

The key insight of this specification is that **requantization is an opportunity, not just overhead**. Every matrix multiplication produces FP32 outputs that must be requantized for the next operation. At each requantization point, we can choose the block orientation that best serves the next operation:

- For KV cache storage: token-oriented blocks (outlier isolation)
- For activations feeding weight matmuls: channel-oriented blocks (MMA alignment)

This adaptive approach achieves both goals: 7.0 effective bits in the KV cache through perfect outlier isolation, and INT4/INT8 MMA acceleration for 5 of 6 major matrix multiplications.

### The One Exception

The Q @ K^T attention score computation cannot use INT MMA because K uses token-oriented blocks (for cache storage) but the reduction dimension is head_dim. We accept this single FP16 fallback because:

1. The benefits of token-oriented K cache (outlier isolation) far outweigh the cost
2. K is read-only from cache — no requantization overhead
3. The scores proceed to FP32 softmax regardless of MMA precision
4. FP16 MMA still uses tensor cores effectively

### Pipeline Overview

| Operation | Compute Path | A Orientation | B Type |
|-----------|--------------|---------------|--------|
| QKV projection | INT4/INT8 MMA | channels ✓ | Q4_K weights |
| Q @ K^T | FP16 MMA | channels ✓ | tokens ✗ (dequant) |
| Scores @ V | FP32 × Q4 | FP32 | tokens ✓ |
| Output projection | INT4/INT8 MMA | channels ✓ | Q4_K weights |
| FFN gate/up | INT4/INT8 MMA | channels ✓ | Q4_K weights |
| FFN down | INT4/INT8 MMA | channels ✓ | Q4_K weights |

### Key Metrics

| Metric | Value |
|--------|-------|
| KV cache compression | 31.3% of FP16 |
| Effective KV precision | 7.0 bits (vs 4.3 with naive blocking) |
| INT MMA utilization | 5 of 6 matmuls |
| Memory savings (100 contexts) | 432 MB |

### Why This Matters

For the target application of 100+ concurrent NPC conversations on a single RTX 4090:

- **Memory**: Q4_1 KV cache uses 197 MB vs 629 MB for FP16 — the difference between fitting in VRAM and running out of memory
- **Precision**: Token-oriented blocking preserves 7.0 effective bits vs 4.3 bits with naive quantization — the difference between coherent dialogue and degraded output quality
- **Throughput**: INT4/INT8 MMA provides up to 2× the throughput of FP16 MMA for weight-bound operations — enabling higher tokens/second across all contexts

### Architecture Overview

```
                                    LAYER INPUT
                                   [channel blocks]
                                         │
                         ┌───────────────┼───────────────┐
                         │               │               │
                         ▼               ▼               ▼
                   ┌─────────┐     ┌─────────┐     ┌─────────┐
                   │    Q    │     │    K    │     │    V    │
                   │ channel │     │  token  │     │  token  │
                   └────┬────┘     └────┬────┘     └────┬────┘
                        │               │               │
                        │               ▼               ▼
                        │          ┌─────────────────────────┐
                        │          │        KV CACHE         │
                        │          │   (outlier isolation)   │
                        │          └────┬───────────────┬────┘
                        │               │               │
                        ▼               ▼               │
                   ┌─────────────────────────┐          │
                   │        Q @ K^T          │          │
                   │      FP16 MMA ░░░       │          │
                   │    (K dequantized)      │          │
                   └───────────┬─────────────┘          │
                               │                        │
                               ▼                        │
                        ┌─────────────┐                 │
                        │   SOFTMAX   │                 │
                        └──────┬──────┘                 │
                               │                        │
                               ▼                        ▼
                        ┌───────────────────────────────────┐
                        │           Scores @ V              │
                        │    (V blocks aligned with sum)    │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   OUTPUT PROJECTION   │
                              │     INT MMA ████      │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │      FFN BLOCK        │
                              │     INT MMA ████      │
                              └───────────┬───────────┘
                                          │
                                          ▼
                                     NEXT LAYER


    Legend:   ████ = INT4/INT8 MMA (5 ops)    ░░░ = FP16 fallback (1 op)
```

*Block orientation pivots at each stage: channel-oriented for weight matmuls (INT MMA), token-oriented for KV cache (outlier isolation).*

---

## 1. Research Foundation

### 1.1 Channel-Correlated Outliers

When large language models process text, the hidden activations flowing through the network exhibit a peculiar but consistent pattern: certain embedding dimensions carry values that are dramatically larger than others. These "outlier" dimensions don't appear randomly — they occur in the same channels across every token the model processes. If dimension 47 fires with large values for the word "cat," it will also fire with large values for "dog," "the," and every other token. This phenomenon has profound implications for quantization.

The pattern was first systematically documented by four landmark research papers:

**SmoothQuant** (Xiao et al., ICML 2023) found that 1-2% of channels consistently carry values 10-100× larger than typical activations. Critically, these outliers appear "in fixed channels across all tokens and layers" — they are a property of the model's learned representations, not the input data.

**LLM.int8()** (Dettmers et al., NeurIPS 2022) identified approximately 0.1% of dimensions as persistent outliers, noting they emerge after roughly 50% of the layers have been processed. Their solution — using FP16 precision for outlier channels while quantizing the rest — validated that channel-specific treatment is essential for maintaining model quality.

**KIVI** (Liu et al., 2024) extended this analysis to KV cache quantization specifically, demonstrating that per-channel scales substantially improve quality. They also found the K cache is more sensitive to quantization error than the V cache.

**AWQ** (Lin et al., 2023) approached the problem from a weight quantization perspective, showing that preserving salient channels is critical for model quality. Their work reinforced that some dimensions carry disproportionate importance.

The convergent finding across all four papers is clear: LLM activations are not uniformly distributed across dimensions. Any quantization strategy that ignores this structure will sacrifice precision unnecessarily.

### 1.2 Outlier Characteristics

The research establishes several consistent properties of activation outliers:

| Property | Value | Source |
|----------|-------|--------|
| Channel prevalence | 1-2% of dimensions | SmoothQuant |
| Magnitude ratio | 10-100× normal | SmoothQuant, LLM.int8() |
| Persistence | Same channels across all tokens | LLM.int8() |
| Layer emergence | After ~50% of layers | LLM.int8() |
| Sign consistency | Often consistent within channel | Empirical |

Understanding these properties is essential for designing an effective quantization strategy. The most important insight is that outliers are **channel-correlated** rather than token-correlated. When dimension 47 fires hot, it fires hot for ALL tokens in the sequence. This means we can predict which channels will have outliers before we even see the data — they're determined by the model architecture and training, not by the specific input.

This predictability opens a powerful optimization: if we organize our quantization blocks so that each block contains values from only one channel, then outlier channels will be completely isolated in their own blocks with their own scales. Normal channels won't be affected at all. This is the core principle behind token-oriented blocking.

### 1.3 The Alignment Problem

There is an inherent tension between what's optimal for outlier isolation and what's optimal for computation. Modern GPUs achieve their highest throughput using tensor core matrix multiply-accumulate (MMA) operations, which work on fixed tile sizes. For these operations to be efficient, all elements within a tile should share a common quantization scale — otherwise, the hardware must perform expensive per-element scale corrections.

The reduction dimension of a matrix multiplication determines which elements get accumulated together. If quantization blocks align with the reduction dimension, all elements in a MMA tile share one scale, and the hardware can operate at full efficiency. If they don't align, each element has a different scale, creating a computational nightmare.

Here's the dilemma: token-oriented blocking achieves perfect outlier isolation, but the blocks run perpendicular to most reduction dimensions in a transformer. Channel-oriented blocking aligns with reduction dimensions for efficient computation, but it mixes outlier and normal values together, destroying precision.

The solution developed in this specification is **adaptive block orientation**: we don't commit to one orientation. Instead, we pivot the block orientation at each requantization point to align with whatever the next operation needs. KV cache storage uses token-oriented blocks for outlier isolation. Activations feeding into matrix multiplications use channel-oriented blocks for compute efficiency. The requantization step between operations is the pivot point where we can change orientation.

---

## 2. Block Orientation Analysis

### 2.1 Traditional vs Token-Oriented Blocking

Quantization works by grouping floating-point values into blocks that share a common scale factor. The choice of which values to group together — the block's orientation — has profound effects on precision.

Traditional quantization schemes, including the standard GGUF formats used by llama.cpp, organize blocks along the embedding dimension. A single block might contain 32 consecutive values from dimensions 0-31 for a single token. This makes intuitive sense for weight matrices and aligns naturally with how data is typically laid out in memory.

However, for activations with channel-correlated outliers, this orientation is catastrophic. When a block contains values from 32 different dimensions, and even one of those dimensions is an outlier channel, the entire block's scale must accommodate the outlier's magnitude. The 31 normal values in that block get crushed into a tiny fraction of the available quantization levels.

Token-oriented blocking flips the orientation: instead of grouping 32 dimensions for one token, we group 32 tokens for one dimension. Each block now contains values from a single channel across multiple sequence positions.

**Traditional blocking** (along head_dim):

```
                        ◄─────── one block ───────►
                        
              d₀   d₁   d₂   d₃   d₄   d₅   d₆   d₇
            ┌────┬────┬────┬────┬────┬────┬────┬────┐
       t₀   │ ░░ │ ░░ │ ░░ │ ██ │ ░░ │ ░░ │ ░░ │ ░░ │
            └────┴────┴────┴────┴────┴────┴────┴────┘
                            ▲
                            │
                      ┌─────┴─────┐
                      │  outlier  │
                      │  channel  │
                      └───────────┘
```

*Legend: ░░ = normal value, ██ = outlier value. Single outlier forces large scale for entire block.*

**Token-oriented blocking** (along tokens):

```
              d₀        d₁        d₂        d₃
            ┌────┐    ┌────┐    ┌────┐    ┌────┐
       t₀   │ ░░ │    │ ░░ │    │ ██ │    │ ░░ │
       t₁   │ ░░ │    │ ░░ │    │ ██ │    │ ░░ │
       t₂   │ ░░ │    │ ░░ │    │ ██ │    │ ░░ │    ◄── one block
       t₃   │ ░░ │    │ ░░ │    │ ██ │    │ ░░ │        per column
            └────┘    └────┘    └────┘    └────┘
              │         │         │         │
              ▼         ▼         ▼         ▼
            s=0.1     s=0.1     s=5.0     s=0.1
```

*Legend: Each column is one block with its own scale (s). Outlier channel d₂ gets large scale; others unaffected.*

The difference is stark: with traditional blocking, the outlier contaminates the entire row of blocks. With token-oriented blocking, the outlier is quarantined to its own column, leaving all other channels pristine.

### 2.2 Precision Impact

The quantitative impact of block orientation is dramatic. Consider a typical scenario with a 128-dimensional head where 2 channels (~1.5%) are outliers. With traditional blocking, each row of blocks spans all 128 dimensions, so every block has a roughly 50% chance of containing an outlier. In practice, about half of all blocks will be contaminated.

When a block is contaminated, the normal values within it suffer severe precision loss. If the outlier has magnitude 50 and normal values have magnitude 1, the scale must accommodate the outlier. The normal values, now represented relative to this large scale, can only use a tiny fraction of the available quantization levels. What should be 8-bit precision becomes effectively 4 bits or worse.

With token-oriented blocking, only the 2 outlier channels (out of 128) have their own dedicated blocks. These blocks have large scales, but that's fine — all values within them are outliers and genuinely need that scale. The remaining 126 channels each get blocks with small, appropriate scales. No contamination occurs.

```
    Traditional (1×32 blocks along head_dim):
    
    Channels:  0         16        32        48        64
               ├─────────┼─────────┼─────────┼─────────┤
               │░░░░░░░░░█████████░░░░░░░░░░░░░░░░░░░░│
               │░░░░░░░░░█████████░░░░░░░░░░░░░░░░░░░░│
               │░░░░░░░░░█████████░░░░░░░░░░░░░░░░░░░░│
               │░░░░░░░░░█████████░░░░░░░░░░░░░░░░░░░░│
               └─────────┴─────────┴─────────┴─────────┘
    
    ░ = clean block    █ = contaminated by outlier
    
    
    Token-oriented (32×1 blocks along tokens):
    
    Channels:  0         16        32        48        64
               ├─────────┼─────────┼─────────┼─────────┤
               │░░░░░░░░░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░│
               │░░░░░░░░░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░│
               │░░░░░░░░░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░│
               │░░░░░░░░░░█░░░░░░░░░░░░░░░░░░░░░░░░░░░│
               └─────────┴─────────┴─────────┴─────────┘
    
    █ = isolated outlier channel (own scale)
```

| Orientation | Affected Blocks | Normal Values Crushed | Effective Bits |
|-------------|-----------------|----------------------|----------------|
| Along head_dim | 64 of 128 (50%) | 31 per block | 4.33 |
| Along tokens | 2 of 128 (1.5%) | 0 per block | **7.00** |

The numbers tell the story: token-oriented blocking achieves **+2.67 bits** of effective precision by completely isolating outliers. For Q4_1 quantization, this is the difference between 4.3 effective bits (barely usable) and 7.0 effective bits (near-lossless for most applications).

### 2.3 Within-Channel Variance

A natural concern with token-oriented blocking is whether values from the same channel but different tokens will have similar enough magnitudes to share a scale effectively. After all, the same word might appear in different contexts with different activation patterns.

Empirical analysis shows this concern is unfounded. Within a single channel, the variance across tokens is remarkably low — typically a ratio of only 1.5× between the minimum and maximum values in a block. This is because each channel represents a learned feature detector, and that detector tends to fire at consistent magnitudes regardless of context.

```
    Outlier channel (dim 47):        Normal channel (dim 48):
    
    ┌────────────────────┐           ┌────────────────────┐
    │ 45 52 38 55 48 51  │           │ 0.9 1.1 0.8 1.2    │
    │ 42 49 47 53 44 50  │           │ 1.0 0.95 1.05 0.88 │
    │ ...                │           │ ...                │
    └────────────────────┘           └────────────────────┘
    
    Range: 38-55 (1.45× ratio)       Range: 0.8-1.2 (1.5× ratio)
    All "outlier magnitude"          All "normal magnitude"
```

Compare this to the cross-channel variance when outliers are present: a 50× ratio or more between outlier and normal channels. Token-oriented blocking exploits this asymmetry. By grouping values that naturally have low variance (same channel, different tokens) rather than values with potentially high variance (different channels, same token), we achieve better utilization of the available quantization range.

---

## 3. Format Specification

### 3.1 Why Asymmetric Quantization

Token-oriented blocking introduces a secondary consideration: the distribution of values within a single-channel block. Because each block now contains values from only one channel, the distribution often exhibits a strong bias toward positive or negative values.

This bias arises from several sources. Post-activation functions like SiLU and GELU produce predominantly positive outputs. Feature detectors often encode presence rather than absence, firing positive when a feature is detected. Training dynamics can cause certain channels to specialize in one direction.

Symmetric quantization formats (Q4_0, Q8_0) assume values are centered around zero and allocate quantization levels equally to positive and negative ranges. When the actual data is clustered in one region, most of these levels go unused.

Consider a positive-biased outlier channel where all values fall between +38 and +55:

```
    Positive-biased outlier channel (values clustered around +45):
    
    Value distribution:
    
    ─────────────────────────────────────────────────────────────────►
    -55       -30        0         30        55
                                   ├────────────┤
                                  38           55
                                  (actual data)
    
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  Q4_0 (symmetric): must span [-55, +55]                         │
    │                                                                 │
    │  ◄──────────────────────────────────────────────────────────►   │
    │  -55                        0                              55   │
    │  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤                 │
    │   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15                │
    │                                      ▲  ▲  ▲                    │
    │                                      └──┴──┘                    │
    │                                    only 3 levels used           │
    │                                    = 1.6 effective bits         │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  Q4_1 (asymmetric): spans [38, 55] only                         │
    │                                                                 │
    │                                   ├────────────┤                │
    │                                  38           55                │
    │                                   ├──┼──┼──┼──┤                 │
    │                                    0  4  8 12 15                │
    │                                   ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲              │
    │                                   all 16 levels used            │
    │                                   = 4.0 effective bits          │
    └─────────────────────────────────────────────────────────────────┘
```

With symmetric Q4_0, the scale must span from -55 to +55 to accommodate the most extreme possible value. But our actual data only uses a small slice of this range, so we get only 3 of 16 possible quantization levels — a mere 1.6 effective bits.

With asymmetric Q4_1, we store both a scale and a minimum value. The quantized range maps directly to [38, 55], using all 16 levels for a full 4.0 effective bits. The asymmetric format recovers **2.4 bits** of precision for shifted distributions.

This makes asymmetric formats (Q4_1, Q8_1) essential for token-oriented blocking. The "_1" suffix in GGUF nomenclature indicates the presence of a minimum parameter alongside the scale.

### 3.2 Q8_1 Block Structure

Standard GGUF Q8_1: 32 signed 8-bit integers with F16 scale and F16 min.

```
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                           Q8_1 Block (36 bytes)                          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐   │
    │   │v₀│v₁│v₂│v₃│v₄│v₅│v₆│v₇│v₈│v₉│..│..│..│..│..│..│..│..│..│..│..│v₃₁  │
    │   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   │
    │   └────────────────────── 32 × INT8 ──────────────────────┘ ┌────┬────┐ │
    │                          (32 bytes)                         │ sc │ mn │ │
    │                                                             └────┴────┘ │
    │                                                              2B    2B   │
    └──────────────────────────────────────────────────────────────────────────┘
    
    Compression: 56.3% of FP16 (36B vs 64B for 32 elements)
```

To convert a quantized value back to floating point, multiply the stored integer by the scale and add the minimum: `value = data × scale + min`. This simple formula allows the full range of the 8-bit integer (0-255) to map to any arbitrary range of floating-point values.

The quantization process finds the minimum and maximum values in the block, computes a scale that maps this range to 0-255, then rounds each value to the nearest integer. This achieves 7.0 effective bits of precision when the block is well-utilized.

### 3.3 Q4_1 Block Structure

Q4_1 is the 4-bit counterpart to Q8_1, packing two values into each byte. Each block still represents 32 values but requires only 16 bytes of data storage, plus the same 4 bytes of metadata.

```
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                           Q4_1 Block (20 bytes)                          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐             │
    │   │v₀│v₁ │v₂│v₃ │v₄│v₅ │v₆│v₇ │ .... │ .... │ .... │v₃₀│v₃₁            │
    │   └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘             │
    │   └──────────────── 32 × UINT4 (packed pairs) ───────────┘ ┌────┬────┐  │
    │                          (16 bytes)                        │ sc │ mn │  │
    │                                                            └────┴────┘  │
    │   Each byte: [lo nibble | hi nibble]                        2B    2B    │
    └──────────────────────────────────────────────────────────────────────────┘
    
    Compression: 31.3% of FP16 (20B vs 64B for 32 elements)
```

The packing scheme places even-indexed values in the low nibble (bits 0-3) and odd-indexed values in the high nibble (bits 4-7) of each byte. Dequantization extracts the nibble, multiplies by scale, and adds the minimum: `value = nibble × scale + min`.

With only 16 quantization levels (0-15), Q4_1 provides 4.0 effective bits of precision. This is sufficient for KV cache storage where token-oriented blocking ensures good utilization of the available range.

### 3.4 Block Size Comparison

```
    FP16 (baseline):
    ┌────────────────────────────────────────────────────────────────┐
    │████████████████████████████████████████████████████████████████│
    └────────────────────────────────────────────────────────────────┘
                                64 bytes
    
    Q8_1:
    ┌────────────────────────────────────────┬────┐
    │████████████████████████████████████████│░░░░│
    └────────────────────────────────────────┴────┘
                   32 bytes                   4B
                                           (meta)
    
    Q4_1:
    ┌──────────────────────┬────┐
    │██████████████████████│░░░░│
    └──────────────────────┴────┘
           16 bytes          4B
                          (meta)
```

### 3.5 Format Comparison

| Format | Data | Metadata | Total | Bytes/Elem | vs FP16 | Eff. Bits |
|--------|------|----------|-------|------------|---------|-----------|
| FP16 | 64B | — | 64B | 2.000 | 100% | 16.0 |
| Q8_1 | 32B | 4B | 36B | 1.125 | 56.3% | 7.0 |
| Q4_1 | 16B | 4B | 20B | 0.625 | 31.3% | 4.0 |

---

## 4. Adaptive Orientation Pipeline

### 4.1 Core Insight

The central innovation of this specification is recognizing that **requantization is not just overhead — it's an opportunity**. Every time we convert from the FP32 accumulator of a matrix multiplication back to a quantized format, we have a choice: which orientation should the new blocks have?

Traditional approaches treat this choice as fixed by the data layout. But if we're going to requantize anyway (which we must, to maintain compression), we can choose the orientation that best serves the next operation. This is the essence of adaptive block orientation.

The decision rule is straightforward: for data that will be stored long-term (KV cache), use token-oriented blocking for outlier isolation. For data that will immediately feed into a matrix multiplication, use channel-oriented blocking for compute alignment.

```
    ┌─────────────┐
    │   MatMul    │
    │  (FP32 acc) │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     Choose orientation
    │ Requantize  │ ◄── based on NEXT op's
    │   Q4_1      │     reduction dimension
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Next Op    │     Blocks now aligned
    │  (INT MMA)  │     with reduction dim
    └─────────────┘
```

This approach requires thinking ahead: at each requantization point, we must know what the data will be used for next. In a transformer layer, this is completely predictable — the computation graph is fixed. We can hardcode the optimal orientation for each stage.

### 4.2 Full Pipeline

```
    Layer Input [channel-oriented]
           │
           ▼
    ╔═══════════════════════════════════════╗
    ║         QKV PROJECTION                ║
    ║   Reduction: hidden_dim               ║
    ║   Input blocks: channels ✓ aligned    ║
    ║   Path: INT4/INT8 MMA                 ║
    ╚═══════════════════════════════════════╝
           │
           ├──────────────────┬─────────────────────┐
           ▼                  ▼                     ▼
    ┌─────────────┐    ┌─────────────┐      ┌─────────────┐
    │      Q      │    │      K      │      │      V      │
    │  (channels) │    │  (tokens)   │      │  (tokens)   │
    │  for Q@K^T  │    │  for cache  │      │  for cache  │
    └──────┬──────┘    └──────┬──────┘      └──────┬──────┘
           │                  │                    │
           │                  ▼                    ▼
           │           ┌─────────────────────────────────┐
           │           │          KV CACHE               │
           │           │     (token-oriented)            │
           │           │   Perfect outlier isolation     │
           │           └─────────────────────────────────┘
           │                  │                    │
           ▼                  ▼                    │
    ╔═══════════════════════════════════════╗     │
    ║            Q @ K^T                    ║     │
    ║   Reduction: head_dim                 ║     │
    ║   Q: channels ✓  K: tokens ✗          ║     │
    ║   Path: FP16 MMA (dequant K)          ║     │
    ╚═══════════════════════════════════════╝     │
           │                                      │
           ▼                                      │
    ┌─────────────┐                               │
    │   Softmax   │                               │
    │   (FP32)    │                               │
    └──────┬──────┘                               │
           │                                      │
           ▼                                      ▼
    ╔═══════════════════════════════════════════════╗
    ║              Scores @ V                       ║
    ║   Reduction: seq_kv (tokens)                  ║
    ║   Scores: FP32   V: tokens ✓ aligned          ║
    ║   Path: FP32 × Q4 dequant                     ║
    ╚═══════════════════════════════════════════════╝
           │
           ▼
    ┌─────────────┐
    │ Requantize  │
    │ (channels)  │
    └──────┬──────┘
           │
           ▼
    ╔═══════════════════════════════════════╗
    ║       OUTPUT PROJECTION               ║
    ║   Reduction: head_dim                 ║
    ║   Input blocks: channels ✓ aligned    ║
    ║   Path: INT4/INT8 MMA                 ║
    ╚═══════════════════════════════════════╝
           │
           ▼
    ┌─────────────┐
    │  Residual   │
    │ + LayerNorm │
    └──────┬──────┘
           │
           ▼
    ╔═══════════════════════════════════════╗
    ║         FFN GATE + UP                 ║
    ║   Reduction: hidden_dim               ║
    ║   Input blocks: channels ✓ aligned    ║
    ║   Path: INT4/INT8 MMA                 ║
    ╚═══════════════════════════════════════╝
           │
           ▼
    ┌─────────────┐
    │ SiLU × mul  │
    │   (FP32)    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Requantize  │
    │ (channels)  │
    └──────┬──────┘
           │
           ▼
    ╔═══════════════════════════════════════╗
    ║           FFN DOWN                    ║
    ║   Reduction: intermediate_dim         ║
    ║   Input blocks: channels ✓ aligned    ║
    ║   Path: INT4/INT8 MMA                 ║
    ╚═══════════════════════════════════════╝
           │
           ▼
    ┌─────────────┐
    │  Residual   │
    │ + LayerNorm │
    └──────┬──────┘
           │
           ▼
      Next Layer
```

### 4.3 Block Orientation Summary

The table below summarizes the orientation choices at each stage of the pipeline. Notice the pattern: everything feeding into a weight matmul uses channel-oriented blocks, while KV cache storage uses token-oriented blocks.

| Tensor | Block Orientation | Reason |
|--------|-------------------|--------|
| Layer input | channels | Aligns with QKV projection reduction |
| Q (after projection) | channels | Aligns with Q @ K^T reduction |
| K (in cache) | tokens | Outlier isolation |
| V (in cache) | tokens | Outlier isolation + Scores@V alignment |
| Attention output | channels | Aligns with output projection reduction |
| Post-attention | channels | Aligns with FFN reduction |
| FFN intermediate | channels | Aligns with down projection reduction |
| FFN output | channels | Aligns with next layer |

The Q tensor is an interesting case: it uses channel-oriented blocking because Q @ K^T reduces along head_dim, and channel-oriented blocks align with this reduction. This is different from K and V, which prioritize outlier isolation for long-term cache storage.

The V cache happens to benefit from double alignment: token-oriented blocking provides outlier isolation, and the Scores @ V operation reduces along the token dimension, which is exactly how V is blocked. This serendipity means V can use INT MMA without dequantization.

---

## 5. KV Cache Architecture

### 5.1 Design Philosophy

The KV cache is the largest consumer of memory in long-context inference. For 100 concurrent NPC conversations at 96 tokens each, the cache can exceed 600 MB with FP16 storage. Quantization is essential for memory efficiency, but naive quantization destroys precision due to outlier contamination.

Token-oriented blocking solves this problem by giving each channel its own quantization scale. Outlier channels get large scales that accommodate their magnitude; normal channels get small scales that preserve their precision. No contamination occurs because values from different channels never share a block.

The page structure aligns naturally with this approach. A page holds 32 tokens worth of data — exactly one block per channel. When we append a new token to the cache, we're adding one element to each of 128 blocks. When we read a page for attention computation, we load 128 complete blocks with 128 independent scales.

### 5.2 Page Structure

With page_size=32 and 32-element blocks along tokens, one page equals exactly one block row. This alignment is deliberate: it ensures that page boundaries coincide with block boundaries, simplifying memory management.

```
    Single page (32 tokens × 128 dimensions):
    
              d₀     d₁     d₂     d₃           d₁₂₆   d₁₂₇
            ┌──────┬──────┬──────┬──────┬─────┬──────┬──────┐
       t₀   │      │      │      │      │     │      │      │
       t₁   │      │      │      │      │     │      │      │
       t₂   │      │      │      │      │     │      │      │
       ⋮    │  B₀  │  B₁  │  B₂  │  B₃  │ ⋯   │ B₁₂₆ │ B₁₂₇ │
       t₂₉  │      │      │      │      │     │      │      │
       t₃₀  │      │      │      │      │     │      │      │
       t₃₁  │      │      │      │      │     │      │      │
            └──────┴──────┴──────┴──────┴─────┴──────┴──────┘
    
    Each column = one Q4_1 block (32 tokens × 1 dim)
    with its own scale + min
```

Each column in this diagram represents one quantization block. The block contains 32 values (one per token) from a single dimension, plus a scale and minimum for dequantization. If dimension d₂ is an outlier channel with large activations, its block will have a large scale — but this doesn't affect any other dimension.

### 5.3 Memory Layout

Pages are stored as contiguous arrays of blocks. Within a page, blocks are ordered by dimension: all 32 tokens for dimension 0, then all 32 tokens for dimension 1, and so on. This layout enables efficient sequential reads during attention computation.

```
    Page memory layout:
    
    ┌────────┬────────┬────────┬────────┬─────┬────────┬────────┐
    │   B₀   │   B₁   │   B₂   │   B₃   │ ⋯   │  B₁₂₆  │  B₁₂₇  │
    └────────┴────────┴────────┴────────┴─────┴────────┴────────┘
    
    
    Q4_1 block structure (20 bytes):
    
    ┌─────────────────────────────────────┬────────┬────────┐
    │        16 bytes (32 × INT4)         │ scale  │  min   │
    └─────────────────────────────────────┴────────┴────────┘
    
    
    Q8_1 block structure (36 bytes):
    
    ┌─────────────────────────────────────┬────────┬────────┐
    │        32 bytes (32 × INT8)         │ scale  │  min   │
    └─────────────────────────────────────┴────────┴────────┘
```

The block metadata (scale and min) is stored immediately after the quantized data. This colocation ensures that when we load a block for dequantization, both the data and its parameters arrive in the same cache line.

### 5.4 Memory Footprint

For Qwen3-8B (4 KV heads, 128 head_dim, 32 layers):

| Format | Bytes/Page/Head | Per 32 Layers | vs FP16 |
|--------|-----------------|---------------|---------|
| FP16 | 8,192 | 2,097,152 | 100% |
| Q8_1 | 4,608 | 1,179,648 | 56.3% |
| Q4_1 | 2,560 | 655,360 | 31.3% |

For 100 concurrent NPC contexts at 96 tokens (3 pages) each:

| Format | Total KV Cache | Savings vs FP16 |
|--------|----------------|-----------------|
| FP16 | 629 MB | — |
| Q8_1 | 354 MB | 275 MB |
| Q4_1 | 197 MB | 432 MB |

```
    Memory footprint comparison (100 contexts, 96 tokens each):
    
    FP16:  ████████████████████████████████████████████████████████████  629 MB
    
    Q8_1:  ██████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  354 MB
                                             ▲
                                             └─ 275 MB saved
    
    Q4_1:  ███████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  197 MB
                              ▲
                              └─ 432 MB saved (69%)
```

---

## 6. Tensor Core Integration

Modern NVIDIA GPUs achieve their highest throughput using tensor cores — specialized hardware units designed for matrix multiply-accumulate operations. Understanding how to feed data to these units efficiently is critical for realizing the benefits of quantized inference.

### 6.1 MMA Tile Alignment

Tensor cores operate on fixed-size tiles. For INT8 operations on Ada and Hopper architectures, the fundamental tile shape is M×N×K where K (the reduction dimension) must be 32 elements. For INT4 operations, K must be 64 elements.

This has a direct implication for our quantization blocks: when the reduction dimension of a matmul aligns with our block orientation, all 32 (or 64) elements in a tile share a single quantization scale. The MMA produces an integer accumulator, and we can apply the scale correction once at the end rather than per-element.

When alignment fails, each element in the tile has a potentially different scale, requiring either per-element correction (expensive) or dequantization before the MMA (sacrificing INT precision).

```
    INT8 MMA (Ada/Hopper):
    
    mma.sync.m16n8k32.s8.s8.s32
    
    ┌─────────────────────────────────┐
    │                                 │
    │         M = 16 rows             │
    │                                 │
    └─────────────────────────────────┘
              K = 32 elements
              (1 Q8_1 block)
    
    
    INT4 MMA (Ada/Hopper):
    
    mma.sync.m16n8k64.s4.s4.s32
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │                        M = 16 rows                              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
                            K = 64 elements
                            (2 Q4_1 blocks)
```

One Q8_1 block exactly fills one INT8 K-tile. Two Q4_1 blocks fill one INT4 K-tile. This precise alignment enables clean scale handling at block boundaries, making the overhead of quantization metadata negligible.

### 6.2 Q @ K^T Handling

The Q @ K^T operation computes attention scores by multiplying queries against keys. The reduction dimension is head_dim (typically 128), which means the MMA tiles should span head_dim elements.

Q uses channel-oriented blocks, with each block spanning 32 consecutive dimensions. This aligns perfectly with the reduction: when we process dimensions 0-31, all values come from one Q block with one scale.

K, however, uses token-oriented blocks for outlier isolation in the cache. Each K block spans 32 tokens for a single dimension. This is perpendicular to the reduction dimension — when we need dimensions 0-31 for the MMA, we must gather from 32 different K blocks, each with its own scale.

```
    Q @ K^T operation:
    
         Q                    K^T                    Scores
    ┌─────────┐         ┌───────────┐           ┌───────────┐
    │         │         │ ░ ░ ░ ░ ░ │           │           │
    │ channel │    @    │ ░ ░ ░ ░ ░ │     =     │           │
    │ blocks  │         │ ░ ░ ░ ░ ░ │           │           │
    │    ✓    │         │ ░ ░ ░ ░ ░ │           │           │
    └─────────┘         └───────────┘           └───────────┘
                         token blocks
                         (misaligned ✗)
    
    
    Solution: Dequant K to FP16
    
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  K (Q4_1)    │ ──► │  K (FP16)    │ ──► │  FP16 MMA    │
    │ token blocks │     │  dequantized │     │              │
    └──────────────┘     └──────────────┘     └──────────────┘
```

The pragmatic solution is to dequantize K to FP16 before the MMA. This adds one dequantization step per K load, but K is read-only from the cache (we don't requantize it), and the scores proceed to FP32 softmax regardless of the MMA precision. The FP16 MMA still uses tensor cores effectively, so throughput impact is modest.

This is the one operation in our pipeline that cannot use INT MMA. We accept this tradeoff because the benefits of token-oriented K cache blocking (outlier isolation, 7.0 effective bits) far outweigh the cost of FP16 dequantization for this single operation.

### 6.3 Scores @ V Handling

The Scores @ V operation multiplies the softmax attention weights against the values to produce the attention output. The reduction dimension is seq_kv — we sum across all key-value positions.

Here, token-oriented blocking in the V cache actually aligns with the reduction. Each V block spans 32 tokens for one dimension, and the MMA reduces across tokens. When we process tokens 0-31, all values come from one V block with one scale.

```
    Scores @ V operation:
    
       Scores                  V                    Output
    ┌───────────┐         ┌─────────┐           ┌─────────┐
    │           │         │ ░ ░ ░ ░ │           │         │
    │   FP32    │    @    │ ░ ░ ░ ░ │     =     │         │
    │           │         │ ░ ░ ░ ░ │           │         │
    │           │         │ ░ ░ ░ ░ │           │         │
    └───────────┘         └─────────┘           └─────────┘
     (seq_q × seq_kv)      token blocks         (seq_q × head_dim)
                          (aligned ✓)
    
    Reduction along seq_kv = along token blocks
    Each K-tile spans one complete block with one scale+min
```

This fortuitous alignment means V cache can potentially use INT MMA paths, though in practice the scores are FP32 (from softmax), so the operation typically uses mixed-precision multiply-accumulate rather than pure INT MMA.

### 6.4 Scale Application

When both operands of a matmul use aligned quantization, the scale handling is elegant. Consider INT4 MMA where both A and B blocks have scale and min parameters:

```
    INT accumulator flow:
    
    ┌─────────┐   ┌─────────┐
    │ A block │   │ B block │
    │ (INT4)  │   │ (INT4)  │
    │ scale_a │   │ scale_b │
    │ min_a   │   │ min_b   │
    └────┬────┘   └────┬────┘
         │             │
         ▼             ▼
    ┌─────────────────────┐
    │    INT4 × INT4      │
    │    MMA (exact)      │
    │    INT32 accum      │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Apply scales:     │
    │   result = acc ×    │
    │   scale_a × scale_b │
    │   + corrections     │
    └─────────────────────┘
```

The MMA operates on pure integers, producing an exact INT32 accumulator. The scale correction is applied once per tile in the epilogue. For asymmetric quantization with min parameters, there are cross-terms involving the min values that must be computed, but these are also per-tile rather than per-element.

The critical insight is that block alignment enables this clean separation: do the expensive MMA in integer arithmetic, then apply the cheap floating-point scale correction at the end.

---

## 7. Implementation Considerations

### 7.1 Memory Layout Strategy

Implementing adaptive block orientation requires careful attention to memory layout. The key insight is that the same physical data can be interpreted with different orientations — what changes is how we traverse it.

For token-oriented blocks (KV cache), memory is organized so that consecutive addresses contain values from consecutive tokens within the same channel. A complete block of 32 values for dimension d occupies a contiguous 20-byte (Q4_1) or 36-byte (Q8_1) region. Adjacent blocks in memory represent adjacent dimensions.

For channel-oriented blocks (activations), consecutive addresses contain values from consecutive channels within the same token. This layout ensures that when we load a tile for matrix multiplication, all loaded values share a common quantization scale.

The reorientation operation occurs during requantization. When we requantize the FP32 output of a matrix multiplication, we gather values along the appropriate dimension for the new orientation, compute per-block scales, and store in the new layout. This gather-quantize-scatter pattern is the pivot point where orientation changes.

### 7.2 Block Indexing

For token-oriented tensors with shape (tokens, channels), the block index for position (t, c) is computed as:
- Page index: t / 32
- Block within page: c
- Element within block: t % 32

For channel-oriented tensors, the indexing inverts:
- Row index: t
- Block within row: c / 32
- Element within block: c % 32

This simple integer arithmetic enables efficient access patterns. The 32-element block size aligns naturally with warp-level operations on NVIDIA GPUs, allowing entire blocks to be processed by a single warp.

### 7.3 Tensor Core Alignment

NVIDIA tensor cores impose strict alignment requirements on input data. For INT8 MMA, the K-dimension must be a multiple of 32 elements. For INT4 MMA, it must be a multiple of 64 elements (two Q4_1 blocks).

The 32-element block size of Q8_1 matches the INT8 MMA tile exactly — one block fills one K-tile, and all elements share a single scale. This enables clean scale handling: perform the integer MMA, then apply a single scale multiplication to the entire tile's accumulated result.

For Q4_1 with INT4 MMA, two blocks must be processed together to fill the 64-element K-tile. The two blocks may have different scales, requiring either: (a) dequantization to a common scale before MMA, or (b) scale correction in the accumulator. Option (a) is simpler and often preferred.

### 7.4 Requantization Kernels

The requantization operation fuses naturally into the matmul epilogue. After the MMA produces FP32 accumulators, the requantization kernel:

1. Gathers 32 values along the target dimension
2. Finds the minimum and maximum values
3. Computes scale = (max - min) / 15 for Q4_1 or / 255 for Q8_1
4. Quantizes each value: quantized = round((value - min) / scale)
5. Packs the quantized values into the output block
6. Stores scale and min as F16 metadata

This operation has negligible cost compared to the MMA itself. The dominant overhead is the memory traffic for writing the requantized output, which is reduced by the compression ratio of the quantized format.

### 7.5 K Cache Dequantization

The Q @ K^T operation presents the one case where block orientation misaligns with the reduction dimension. The K cache uses token-oriented blocks for outlier isolation, but the matmul reduces along head_dim.

The solution is to dequantize K to FP16 before the MMA. For each K-tile load:

1. Load the Q4_1 block (token-oriented)
2. Extract scale and min
3. Unpack and dequantize each element: value = nibble × scale + min
4. Store as F16 in registers
5. Proceed with FP16 MMA

This dequantization adds latency but not additional memory bandwidth — we're transforming data already loaded from the cache. The FP16 MMA path uses tensor cores effectively, and the output proceeds to FP32 softmax regardless.

---

## 8. Performance Analysis

### 8.1 Compute Paths Summary

```
    Operation          Reduction    A Block      B Block       Compute Path
    ─────────────────────────────────────────────────────────────────────────
    
    QKV projection     hidden_dim   channels ✓   weights ✓     ████ INT MMA
                                                               
    Q @ K^T            head_dim     channels ✓   tokens ✗      ░░░░ FP16 MMA
                                                               (dequant K)
                                                               
    Scores @ V         seq_kv       FP32         tokens ✓      ▓▓▓▓ FP32×Q4
                                                               
    Output proj        head_dim     channels ✓   weights ✓     ████ INT MMA
                                                               
    FFN gate           hidden_dim   channels ✓   weights ✓     ████ INT MMA
                                                               
    FFN up             hidden_dim   channels ✓   weights ✓     ████ INT MMA
                                                               
    FFN down           interm_dim   channels ✓   weights ✓     ████ INT MMA
    
    ─────────────────────────────────────────────────────────────────────────
    
    ████ = INT4/INT8 MMA (5 operations)
    ░░░░ = FP16 MMA fallback (1 operation)
    ▓▓▓▓ = Mixed precision
```

| Operation | Reduction Dim | A Aligned | B Aligned | Compute Path |
|-----------|---------------|-----------|-----------|--------------|
| QKV proj | hidden_dim | ✓ | ✓ (weights) | INT4/INT8 MMA |
| Q @ K^T | head_dim | ✓ | ✗ (tokens) | **FP16 MMA** |
| Scores @ V | seq_kv | FP32 | ✓ (tokens) | FP32 × dequant |
| Output proj | head_dim | ✓ | ✓ (weights) | INT4/INT8 MMA |
| FFN gate | hidden_dim | ✓ | ✓ (weights) | INT4/INT8 MMA |
| FFN up | hidden_dim | ✓ | ✓ (weights) | INT4/INT8 MMA |
| FFN down | intermediate | ✓ | ✓ (weights) | INT4/INT8 MMA |

**5 of 6 matmuls use INT MMA.** Only Q @ K^T requires FP16 dequant path.

### 8.2 Memory Bandwidth

Primary bottleneck for decode is weight and KV cache reads:

| Component | FP16 | Q4_1 | Savings |
|-----------|------|------|---------|
| Weights | ~8 GB | ~4 GB (Q4_K) | 50% |
| KV cache | 65 MB/1K tok | 20 MB/1K tok | 69% |
| Activations | 22 KB/tok | 7 KB/tok | 69% |

### 8.3 Expected Throughput

For RTX 4090 (504 GB/s bandwidth):

| Metric | FP16 Baseline | Q4_1 Pipeline | Improvement |
|--------|---------------|---------------|-------------|
| Single decode | ~45 t/s | ~70 t/s | +55% |
| 100 concurrent | OOM | ~30 t/s total | Enabled |

```
    Single stream throughput:
    
    FP16:  ████████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░  45 t/s
    
    Q4_1:  ██████████████████████████████████████████████████████████████████████  70 t/s
                                                                    ▲
                                                                    └─ +55%
    
    
    Concurrent contexts (RTX 4090, 24GB VRAM):
    
           ┌────────────────────────────────────────────────────────────────────┐
     FP16: │██████████████████████████ OOM at ~40 contexts                      │
           └────────────────────────────────────────────────────────────────────┘
           
           ┌────────────────────────────────────────────────────────────────────┐
     Q4_1: │████████████████████████████████████████████████████████████████████│
           │                       100+ contexts @ 30 t/s total                 │
           └────────────────────────────────────────────────────────────────────┘
```

---

## 9. Design Decisions Summary

### 9.1 Key Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| KV cache format | Q4_1/Q8_1, token-oriented | Outlier isolation (7.0 bits) |
| Activation format | Q4_1/Q8_1, channel-oriented | INT MMA alignment |
| Q @ K^T path | FP16 dequant | Accept one unaligned op for cache benefits |
| Requantization | At each matmul output | Pivot orientation for next op |
| Page size | 32 tokens | Matches block size |
| Asymmetric quant | Yes (min parameter) | Handles shifted distributions |

### 9.2 Tradeoffs

| Benefit | Cost |
|---------|------|
| 7.0-bit effective KV precision | FP16 path for Q @ K^T |
| INT MMA for 5/6 ops | Requant overhead at each stage |
| 69% KV memory reduction | Orientation tracking complexity |
| 100+ concurrent contexts | — |

### 9.3 Core Insight

**Requantization is an opportunity, not just overhead.** By choosing block orientation at each requant point, we achieve:

- Optimal outlier isolation in KV cache (token blocking)
- Optimal compute alignment in activations (channel blocking)
- INT4/INT8 MMA for weight-dominated operations
- One simple exception (Q @ K^T via FP16)

### 9.4 Decision Flow

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         BLOCK ORIENTATION DECISION                       │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │  Is this KV cache?     │
                           └───────────┬────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
               ┌─────────────┐                   ┌─────────────┐
               │     YES     │                   │     NO      │
               └──────┬──────┘                   └──────┬──────┘
                      │                                 │
                      ▼                                 ▼
         ┌────────────────────────┐       ┌────────────────────────┐
         │    TOKEN-ORIENTED      │       │  What is next op's     │
         │                        │       │  reduction dimension?  │
         │  ┌─┐ ┌─┐ ┌─┐ ┌─┐      │       └───────────┬────────────┘
         │  │░│ │░│ │█│ │░│      │                   │
         │  │░│ │░│ │█│ │░│      │                   ▼
         │  │░│ │░│ │█│ │░│      │       ┌────────────────────────┐
         │  └─┘ └─┘ └─┘ └─┘      │       │   CHANNEL-ORIENTED     │
         │                        │       │                        │
         │  • Outlier isolation   │       │  ┌─┬─┬─┬─┬─┬─┬─┬─┐    │
         │  • 7.0 effective bits  │       │  │░│░│░│░│░│░│░│░│    │
         │  • Per-channel scale   │       │  └─┴─┴─┴─┴─┴─┴─┴─┘    │
         └────────────────────────┘       │                        │
                                          │  • INT MMA alignment   │
                                          │  • Reduction-aligned   │
                                          │  • Compute optimal     │
                                          └────────────────────────┘
```

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         FORMAT SELECTION DECISION                        │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │  Precision requirement │
                           └───────────┬────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
       ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
       │   Maximum   │          │  Balanced   │          │   Minimum   │
       │  precision  │          │             │          │   memory    │
       └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
              │                        │                        │
              ▼                        ▼                        ▼
       ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
       │    Q8_1     │          │    Q8_1     │          │    Q4_1     │
       │  activations│          │  KV cache   │          │  KV cache   │
       │             │          │             │          │             │
       │  36B / 32   │          │  36B / 32   │          │  20B / 32   │
       │  = 1.125B/e │          │  = 1.125B/e │          │  = 0.625B/e │
       │             │          │             │          │             │
       │  7.0 bits   │          │  7.0 bits   │          │  4.0 bits   │
       └─────────────┘          └─────────────┘          └─────────────┘
```

---

## References

1. Xiao, G., et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML 2023.

2. Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS 2022.

3. Liu, Z., et al. (2024). "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." arXiv:2402.02750.

4. Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978.

5. GGML/llama.cpp. (2024). "GGUF Quantization Format Specification."

6. NVIDIA. (2023). "CUDA C++ Programming Guide: Warp Matrix Functions."

---

*Document revision 5.0 — Consolidated adaptive block-oriented quantization with INT4/INT8 MMA pipeline*