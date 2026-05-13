# Two-Tier Expert Caching for Mixture-of-Experts Inference on Consumer GPUs: Scaling Concurrent Sessions Beyond Unified Memory Architectures

*February 2026*

---

## Abstract

We present a two-tier expert caching framework for serving large Mixture-of-Experts (MoE) language models on consumer-grade discrete GPUs with limited VRAM. Our system maintains a managed cache of expert weight tensors in GPU VRAM backed by a secondary store in pinned host memory, employing speculative prefetch via learned transition matrices, smart eviction policies, and fused scatter kernels for MoE dispatch. Implemented in Rust with custom CUDA kernels, the framework uses a dual-thread architecture where a forward thread handles attention and routing while a dedicated pipeline thread manages expert cache operations and GEMM dispatch.

Evaluated on Qwen3-30B-A3B (30.5B total parameters, 3.3B active) running on an NVIDIA RTX 4090 laptop GPU with only 16 GB of VRAM, our framework achieves 308.5 tokens/second aggregate decode throughput across 24 concurrent sessions, compared to approximately 130 tokens/second reported for single-session inference using llama.cpp on a desktop RTX 4090 with 24 GB of VRAM. We provide architectural analysis comparing discrete GPU, unified memory (Apple Silicon), and CPU-offload (KTransformers) approaches, demonstrating that while unified memory architectures achieve superior single-session latency, our batched expert caching approach delivers dramatically higher aggregate throughput. We further analyse how hardware trends in PCIe bandwidth, VRAM capacity, and GPU compute scaling will amplify this advantage over coming hardware generations, positioning expert caching as a durable architectural pattern for MoE inference on discrete GPU platforms.

---

## 1. Introduction

Mixture-of-Experts (MoE) architectures have emerged as a dominant paradigm for scaling language model capacity while controlling inference cost. Models such as Qwen3-235B-A22B (235B total, 22B active), DeepSeek-V3 (671B total, 37B active), and Qwen3-30B-A3B (30.5B total, 3.3B active) achieve performance competitive with dense models many times their active parameter count by routing each token through a small subset of specialised expert sub-networks.

This sparsity creates a unique deployment challenge: the full model is far too large for GPU VRAM, yet only a fraction of parameters are needed for any given token. On datacenter hardware with multi-GPU tensor parallelism, all experts reside in aggregate VRAM. On consumer hardware with a single discrete GPU, this is impossible. The Qwen3-30B-A3B model at Q4_K_M quantisation occupies approximately 17 GB, with 6,144 total expert tensors across 48 layers, of which only 8 are active per token per layer. At Q4_K_M, each expert occupies approximately 2.9 MB. On a 16 GB GPU, only 3,394 expert slots can be maintained in VRAM at any time, covering 55% of the total expert population.

Existing frameworks approach this problem in three ways. **Full GPU offload** (llama.cpp, vLLM, SGLang) requires all active weights in VRAM, limiting model scale to what fits. **CPU compute offload** (KTransformers) keeps experts in host memory and executes expert GEMMs on the CPU using Intel AMX or AVX-512 instructions, avoiding the PCIe transfer entirely but sacrificing GPU compute throughput. **Unified memory architectures** (Apple Silicon) eliminate the memory hierarchy entirely, providing 800 GB/s bandwidth to all memory from both CPU and GPU cores, but are limited in compute capacity and cannot batch across concurrent sessions.

We propose a fourth approach: **two-tier expert caching with batched concurrent inference.** Our framework maintains a managed VRAM cache of expert tensors, backed by a pinned host memory tier, with smart eviction, speculative prefetch, and fused dispatch kernels. Critically, we batch token processing across multiple independent inference sessions, amortising the fixed per-layer overhead of cache management across many contexts and exploiting expert reuse patterns between contexts with correlated routing distributions.

This paper makes the following contributions:

1. We describe the architecture of a two-tier expert cache with learned eviction, speculative prefetch, and a dual-thread GPU execution model implemented in Rust with custom CUDA kernels.
2. We present empirical results demonstrating 2.4x the aggregate throughput of the best published single-session baseline, on hardware with 33% less VRAM.
3. We analyse the compute-versus-bandwidth characteristics of three competing inference architectures across decode and prefill phases.
4. We project how hardware trends in interconnect bandwidth, VRAM capacity, and GPU compute will affect the relative competitiveness of each approach over the next 3–5 hardware generations.

---

## 2. System Architecture

### 2.1 Model and Hardware Configuration

Our primary evaluation target is Qwen3-30B-A3B, a Mixture-of-Experts transformer with 48 layers, 128 experts per layer with top-8 routing, 4 KV heads (28:4 GQA ratio), and 128-dimensional heads. The model is quantised to Q4_K_M format with total weights of approximately 17 GB. Each expert consists of three projection matrices (gate, up, down) totalling approximately 2.9 MB at Q4_K. The total expert population is 48 × 128 = 6,144 experts. The framework is implemented in Rust using custom CUDA kernels for routing, scatter-gather, and expert GEMM dispatch.

Primary hardware is an NVIDIA RTX 4090 laptop GPU with 16 GB GDDR6X VRAM (1,008 GB/s bandwidth) connected via PCIe 4.0 x16 (approximately 13 GB/s effective DMA throughput). Host memory is 256 GB DDR5 with approximately 90 GB/s CPU-local bandwidth. Comparative measurements were also collected on an NVIDIA RTX 3090 with 24 GB GDDR6X VRAM.

### 2.2 Dual-Thread Execution Model

The core of the framework is a dual-thread architecture that separates the transformer forward pass from expert cache management and GEMM execution. This design is motivated by the observation that MoE inference involves two fundamentally different workload types: the attention and routing computations (regular, predictable, GPU-friendly) and the expert dispatch (irregular, cache-dependent, requiring DMA coordination).

```
┌─────────────────────────────┐     ┌──────────────────────────────────┐
│      FORWARD THREAD         │     │        PIPELINE THREAD           │
│                             │     │                                  │
│  for layer in 0..48:        │     │  loop:                           │
│    ├─ RMSNorm               │     │    ├─ recv(request) from channel │
│    ├─ Self-Attention (GQA)  │     │    ├─ classify_and_load()        │
│    ├─ Residual Add          │     │    │   ├─ lookup cache slots     │
│    ├─ RMSNorm               │     │    │   ├─ identify misses        │
│    ├─ Gate → Softmax → Sort │     │    │   ├─ evict if needed        │
│    ├─ Async DtoH routing    │     │    │   ├─ DMA misses (stream 1)  │
│    │   indices (event E1)   │     │    │   └─ record fence event     │
│    ├─ Record event E2       │     │    ├─ compute_hits (stream 0)    │
│    ├─ Read indices from     │     │    ├─ fence.wait() [GPU-side]    │
│    │   pinned buffer        │     │    ├─ compute_loaded (stream 0)  │
│    ├─ Build assignments     │     │    ├─ fused_scatter_add()        │
│    ├─ submit_roundtrip() ──────────>   └─ send(result) via channel  │
│    │   [blocks until done]  │     │                                  │
│    ├─ Residual Add          │     │                                  │
│    └─ Send hint for N+1     │     │                                  │
└─────────────────────────────┘     └──────────────────────────────────┘
         │                                        │
         ▼                                        ▼
  ┌─────────────┐                    ┌────────────────────────┐
  │  Stream 0   │                    │  Stream 1 (copy)       │
  │  (compute)  │                    │  DMA: pinned RAM→VRAM  │
  │  Attention  │                    │  DMA: VRAM→pinned RAM  │
  │  Expert     │                    │  (evictions)           │
  │  GEMMs      │                    └────────────────────────┘
  └─────────────┘
```

**Forward thread.** Iterates through the 48 transformer layers sequentially. For each layer, it executes RMSNorm, grouped-query attention with RoPE, and the MoE routing network (gate projection → softmax → top-8 selection). The routing indices are transferred from GPU to pinned host memory asynchronously using a CUDA event pair (E1/E2) to avoid draining the compute stream. Once indices are available on the CPU, the forward thread builds expert assignment structures and submits a work request to the pipeline thread via a bounded channel. It then blocks until the pipeline thread returns the computed MoE output.

**Pipeline thread.** Receives work requests containing the batched hidden states and expert assignments. For each request, it performs three operations:

1. **Classify and load.** Looks up each required expert in the VRAM cache. Hits are flagged for immediate computation. Misses trigger asynchronous DMA transfers from pinned host memory to VRAM on a dedicated copy stream (stream 1), with a CUDA fence event recorded after the last transfer.

2. **Compute.** Expert GEMMs for cache hits execute immediately on the compute stream (stream 0). After the DMA fence resolves (GPU-side wait, no CPU blocking), expert GEMMs for newly loaded experts execute on the same stream.

3. **Fused scatter-add.** A custom CUDA kernel performs the weighted accumulation of expert outputs back into the token hidden states, applying router weights in-place.

### 2.3 Async Routing Protocol

A critical optimisation is the asynchronous transfer of routing indices from GPU to CPU. In the naive approach, reading routing indices requires a `cudaDeviceSynchronize()` or equivalent, which drains all pending work from the compute stream — potentially stalling on several milliseconds of queued attention and normalisation kernels.

Our approach uses a CUDA event pair:

```
Forward thread:                         GPU streams:
                                        
  Launch gate matmul ─────────────────> Stream 0: [gate][softmax][sort]
  Record event E1 on stream 0          Stream 0: ... record E1
  Launch async DtoH on stream 1 ──────> Stream 1: wait(E1) → memcpy DtoH
  Record event E2 on stream 1          Stream 1: ... record E2
  cuEventSynchronize(E2) ◄──────────── [CPU blocks only until DtoH done]
  Read routing indices from pinned buf
```

Event E1 ensures the sort kernel completes before the DtoH transfer begins. Event E2 ensures the transfer completes before the CPU reads the pinned buffer. The CPU never waits for the full compute stream to drain — only for the specific DtoH transfer. This reduced the routing wait from 6.13 ms/layer (full drain) to 0.19 ms/layer (event sync only) during single-token decode.

### 2.4 Two-Tier Expert Cache

The expert cache operates as a two-tier hierarchy. The fast tier resides in GPU VRAM, holding up to 3,394 expert slots on 16 GB hardware (after accounting for backbone weights, KV cache, and compute buffers). The slow tier holds all 6,144 experts in pinned (page-locked) host memory, pre-staged at model load time by reading from the GGUF model file and repacking into GPU-optimal memory layout. This pre-staging eliminates mmap page faults during inference — a critical detail, since mmap-backed loads can incur unpredictable latency from OS page cache eviction.

**Expert repacking.** At load time, each expert's three projection matrices (gate, up, down) are read from the GGUF file and repacked into contiguous, GPU-aligned buffers in pinned memory. This ensures that DMA transfers move contiguous blocks rather than gathering from scattered mmap regions, maximising PCIe throughput. Each expert transfer is approximately 2.9 MB, completing in approximately 0.22 ms at 13 GB/s effective PCIe bandwidth.

### 2.5 Eviction Policy

Eviction uses an LRU-based policy with two mechanisms:

**Drip eviction** maintains 2% VRAM headroom by evicting the least-recently-used experts when the cache approaches capacity. This runs continuously during inference, ensuring that incoming DMA transfers always have destination slots available without blocking.

**End-of-pass eviction** reclaims 7% of occupied slots at the end of each inference pass, writing evicted experts back to pinned host memory via DMA on the copy stream. This is capped by pinned pool write-back capacity to prevent overwhelming the PCIe bus.

A critical optimisation discovered during development was preventing **over-eviction** within a single forward pass. The initial implementation exhibited an eviction-to-miss ratio of 8.7× (694 evictions for 80 misses), aggressively removing experts that subsequent layers within the same pass would require. Correcting this to a near-1:1 ratio reduced cache misses by 66% and improved decode throughput by 24%.

| Metric (Q8_0×8) | Before fix | After fix | Change |
|------------------|-----------|-----------|--------|
| Cache misses | 9,614 | 3,315 | −66% |
| Hit rate | 29.5% | 70.0% | +40pp |
| Eviction ratio | 8.7× | 1.08× | Healthy |
| Decode t/s | 96.3 | 119.4 | +24% |

### 2.6 Speculative Prefetch

We maintain a per-layer transition matrix recording the empirical probability of each expert being active at layer N+1 given the expert set active at layer N. During decode, after dispatching work for layer N, the forward thread issues prefetch hints for the predicted expert set of layer N+1 on the dedicated copy stream. The transition matrix achieves 98.1% prediction accuracy under warm-cache conditions.

However, during prefill, the forward thread sends hints for layer N+1 then immediately dispatches work for that same layer, giving insufficient time for prefetch transfers to complete. Consequently, prefetch provides minimal benefit during prefill (86 hint loads out of 9,614 total misses) and primarily assists decode-phase performance where the working set is stable and the inter-layer gap provides sufficient DMA time.

### 2.7 Fused Weighted Scatter-Add Kernel

Standard MoE dispatch involves separate operations: (1) gather token hidden states by expert assignment, (2) perform the expert GEMM, (3) multiply by router weights, (4) scatter-accumulate back into the output tensor. Each operation involves a separate kernel launch with its own memory traffic.

We implement a fused CUDA kernel that performs scatter-gather, router weight multiplication, and accumulation in a single pass:

```
__global__ void fused_weighted_scatter_add(
    float* output,              // [batch, hidden_dim] — accumulated result
    const float* expert_out,    // [n_assigned, hidden_dim] — expert GEMM results
    const int32_t* token_ids,   // [n_assigned] — which token each result belongs to
    const float* weights,       // [n_assigned] — router weights per assignment
    int n_assigned,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_assigned * hidden_dim) return;
    
    int assign = idx / hidden_dim;
    int dim = idx % hidden_dim;
    int token = token_ids[assign];
    float w = weights[assign];
    
    // Fused: read expert output, multiply by router weight, 
    // atomically accumulate into the correct token position
    atomicAdd(&output[token * hidden_dim + dim], 
              expert_out[assign * hidden_dim + dim] * w);
}
```

This eliminates intermediate buffer allocations and reduces per-layer dispatch time from 61.6 ms to 22.4 ms for the scatter operation (64% reduction) and overall submit roundtrip from 256.5 ms to 165.8 ms (35% reduction). The fused kernel was the single largest per-operation improvement in the framework.

| Metric (Q8_0×8 single decode) | Before fusion | After fusion | Change |
|-------------------------------|--------------|-------------|--------|
| `gemm_scatter` time | 61.6 ms | 22.4 ms | −64% |
| `pipe_compute_hits` | 179.1 ms | 121.2 ms | −32% |
| `submit_roundtrip` | 256.5 ms | 165.8 ms | −35% |
| Decode throughput | 119.4 t/s | 145.7 t/s | +22% |

### 2.8 DMA/Compute Overlap

An important and initially accidental performance characteristic is the overlap between DMA transfers and GPU computation. In the pipeline thread's `process_request`, the execution follows three phases:

```rust
// Phase A: compute cache-hit experts on compute stream (stream 0)
compute_experts_grouped(&xs, &mut ys, &hit_experts)?;

// Phase B: GPU-side wait for copy stream DMA to finish
classified.fence.wait(&device)?;   // stream 0 waits on stream 1's event

// Phase C: compute newly-loaded experts on compute stream (stream 0)
compute_experts_grouped(&xs, &mut ys, &loaded_experts)?;
```

When cache misses trigger DMA transfers of 3+ ms duration on stream 1, Phase A's hit-expert GEMMs execute concurrently on stream 0, effectively hiding the DMA latency. With 630 prefill tokens, the hit GEMMs are substantial enough to fill most of the DMA window. By the time Phase A finishes, the DMA fence is nearly ready.

This overlap is most pronounced during prefill where both DMA traffic and compute volume are high. During single-token decode with warm caches, DMA is minimal and the overlap provides little benefit.

### 2.9 KV Cache Quantisation

Independent of expert caching, we quantise the key-value cache to reduce VRAM pressure and free capacity for additional expert cache slots.

For Qwen3-30B-A3B with 4 KV heads, 128-dimensional heads, and 48 layers:

| KV Format | Bytes/token/layer | Per token (48 layers) | 24ctx × 630tok |
|-----------|-------------------|----------------------|----------------|
| F16 | 2,048 B | 96 KB | 1.45 GB |
| BF16 | 2,048 B | 96 KB | 1.45 GB |
| Q8_0 | 1,024 B | 48 KB | 0.73 GB |
| Q4_0 | 512 B | 24 KB | 0.36 GB |

Quantising KV from F16 to Q4_0 frees over 1 GB of VRAM for additional expert slots. This is the mechanism that enables scaling from 8 concurrent sessions (Q8_0 KV) to 24 sessions (Q4_0 KV) within the same 16 GB VRAM budget.

---

## 3. Experimental Results

### 3.1 Performance Across Configurations

Table 1 presents the performance progression across KV cache formats and concurrent session counts on the RTX 4090 16 GB laptop GPU.

| KV Mode | Contexts | Bulk (t/s) | Decode (t/s) | Per-ctx (t/s) | Peak Tokens |
|---------|----------|-----------|-------------|--------------|-------------|
| F16 | 1 | 471.3 | 9.4 | 9.4 | 630 |
| BF16 | 1 | 547.3 | 17.3 | 17.3 | 630 |
| BF16 | 4 | 1,786.6 | 77.1 | 19.3 | 2,522 |
| Q8_0 | 8 | 2,607.6 | 145.7 | 18.2 | 5,042 |
| Q4_0 | 24 | 2,519.8 | 308.5 | 12.9 | 15,186 |

*Table 1: Performance across KV cache formats and concurrent session counts. RTX 4090 laptop GPU, 16 GB VRAM. Qwen3-30B-A3B at Q4_K_M. Bulk = prefill throughput, Decode = aggregate autoregressive decode.*

The aggregate decode throughput scales superlinearly from 1 to 8 contexts (9.4 → 145.7 t/s, 15.5× increase for 8× contexts), then sublinearly from 8 to 24 (145.7 → 308.5 t/s, 2.1× increase for 3× contexts) as VRAM bandwidth saturates.

### 3.2 Cumulative Impact of Optimisations

Table 2 isolates the contribution of each major optimisation to the Q8_0×8-context configuration.

| Optimisation | Decode (t/s) | Hit Rate | Misses | Δ |
|-------------|-------------|---------|--------|---|
| Baseline (initial) | 96.3 | 29.5% | 9,614 | — |
| + Smart eviction | 119.4 | 70.0% | 3,315 | +24% |
| + Fused scatter kernel | 145.7 | 70.2% | 3,362 | +22% |

*Table 2: Cumulative impact of optimisations on Q8_0×8-context decode throughput.*

### 3.3 Comparison with Published Baselines

Table 3 compares our framework against published benchmarks for Qwen3-30B-A3B inference.

| System | GPU / Hardware | VRAM | Contexts | Aggregate (t/s) |
|--------|---------------|------|----------|-----------------|
| Ollama | RTX 4090 Desktop | 24 GB | 1 | ~30 |
| llama.cpp | RTX 4090 Desktop | 24 GB | 1 | ~130 |
| llama.cpp | RTX 3090 | 24 GB | 1 | 87 |
| llama.cpp | RTX 5090 | 32 GB | 1 | 110 |
| MLX | M4 Max | 48 GB | 1 | ~50 |
| **This work (Q8_0×8)** | **RTX 4090 Laptop** | **16 GB** | **8** | **145.7** |
| **This work (Q4_0×24)** | **RTX 4090 Laptop** | **16 GB** | **24** | **308.5** |

*Table 3: Comparison with published single-session baselines on Qwen3-30B-A3B (Q4_K_M). Our framework exceeds the best published aggregate throughput by 2.4× on hardware with 33% less VRAM.*

No mainstream framework attempts batched multi-session MoE inference on a single consumer GPU. The entire llama.cpp / Ollama / vLLM ecosystem assumes either that the model fits entirely in VRAM, or accepts catastrophic slowdown from CPU offloading.

### 3.4 Per-Layer Timing Decomposition

Table 4 provides the per-layer timing breakdown during single-token decode at Q8_0×8 contexts after all optimisations.

| Phase | Time/layer | Calls | Description |
|-------|-----------|-------|-------------|
| `fwd_routing` | 0.14 ms | 48 | CPU launching attention + routing kernels |
| `fwd_routing_wait` | 0.19 ms | 48 | GPU executing attention + routing (event sync) |
| `fwd_cpu_assign` | 0.002 ms | 48 | Building expert assignments from indices |
| `submit_roundtrip` | 0.35 ms | 48 | Pipeline: classify + GEMM + scatter + return |
| Untracked overhead | ~17 ms | — | Embed, norms, residuals, lm_head, sampling |
| **Total wall time** | **~55 ms** | — | **Full decode step (8 contexts)** |

*Table 4: Per-layer timing decomposition during decode (Q8_0×8 contexts, RTX 4090 16 GB). The forward thread and pipeline thread alternate — never overlap — creating a serialisation tax that is the primary remaining bottleneck.*

Inside `submit_roundtrip` (0.35 ms/layer):

| Sub-phase | Time | Description |
|-----------|------|-------------|
| `pipe_compute_hits` | 0.25 ms | Expert GEMMs for cached experts |
| `pipe_compute_loaded` | 0.02 ms | Expert GEMMs for DMA'd experts |
| `pipe_classify_load` | 0.01 ms | Cache lookup + DMA |
| Thread overhead | 0.07 ms | Channel wake/sleep/signalling |

### 3.5 Cache Hit Rate vs VRAM Capacity

Comparative measurements on the RTX 3090 (24 GB) versus RTX 4090 laptop (16 GB) demonstrate that cache capacity is the dominant factor for decode throughput.

| Metric (Q8_0×8) | 3090 (24 GB) | 4090 (16 GB) |
|-----------------|-------------|-------------|
| Cache slots | ~5,500 | 3,394 |
| Bulk hit rate | 34.3% | 29.5% |
| Decode DMA loads | 2 | 592 |
| Decode DMA loads/layer | 0.004 | 1.23 |
| Decode t/s | 134 | 96.3 (before eviction fix) |

The 3090 holds the entire decode working set with essentially zero DMA during generation. The 4090's smaller cache forces continuous expert swapping during decode. After the eviction fix, the 4090's decode DMA drops from 592 to 233 loads (0.49/layer), closing the gap to 119.4 t/s.

---

## 4. Architectural Analysis: Three Paradigms for MoE Inference

### 4.1 Memory Hierarchy Comparison

The critical distinction between the three paradigms is the bandwidth available to expert weight data:

| Architecture | Expert Bandwidth | Compute (FP16) | Key Constraint |
|-------------|-----------------|----------------|----------------|
| Discrete GPU + cache (this work) | 1,008 GB/s (hit) / 13 GB/s (miss) | 82.6 TFLOPS | Cache hit rate |
| Unified memory (M2 Ultra) | 800 GB/s (all experts) | 27 TFLOPS | Compute ceiling |
| CPU offload (KTransformers) | ~350 GB/s (DDR5) | ~10 TFLOPS (AMX) | CPU compute |

The bandwidth ratio between tiers is the fundamental quantity governing the value of expert caching. On current PCIe 4.0 hardware, the hit-to-miss bandwidth ratio is approximately 77× (1,008 GB/s vs 13 GB/s). This extreme ratio makes cache hit rate the dominant performance lever. By contrast, unified memory architectures have a ratio of 1:1, rendering caching unnecessary but sacrificing the GPU's superior compute density.

### 4.2 Single-Session Decode: Why Unified Memory Wins

During autoregressive decode, each token produces a single vector that passes through the routing network, selecting 8 of 128 experts per layer. The arithmetic intensity is approximately 4 FLOPs per byte of expert weights loaded, placing all architectures firmly in the memory-bound regime. Under these conditions, effective bandwidth to expert data determines throughput.

Our measured per-layer wall-clock time of approximately 1.2 ms on a single context (BF16 KV, 17.3 t/s) versus approximately 0.2 ms on the Mac M2 Ultra (28 t/s) decomposes as follows:

| Component | Discrete GPU (per layer) | Unified Memory (per layer) |
|-----------|------------------------|--------------------------|
| Useful bandwidth work | ~0.15 ms | ~0.15 ms |
| Forward/pipeline thread alternation | ~0.33 ms | 0 ms |
| Event signalling overhead | ~0.07 ms | 0 ms |
| Classify step | ~0.02 ms | 0 ms |
| GPU idle during pipeline | ~0.63 ms | 0 ms |
| **Total** | **~1.2 ms** | **~0.2 ms** |

The Mac isn't beating the hardware — it's beating the threading model. The RTX 4090 has 25% more raw bandwidth (1,008 vs 800 GB/s) but spends approximately 70% of per-layer time on architectural overhead that does not exist on unified memory, where the GPU reads expert weights from the same memory controller as the CPU with no bus crossing, no protocol overhead, and no cache management.

### 4.3 Multi-Session Decode: Where Batching Dominates

The competitive dynamics reverse when serving multiple concurrent sessions:

| Concurrent Contexts | Unified Memory (sequential) | This Work (batched) |
|--------------------|---------------------------|-------------------|
| 1 | 28 t/s | 5–7 t/s |
| 4 | 28 t/s | 20–24 t/s |
| 8 | 28 t/s | 145.7 t/s |
| 24 | 28 t/s | 308.5 t/s |

Unified memory architectures lack batching capability. No mainstream inference framework on Apple Silicon (MLX, Ollama, LM Studio) supports batched multi-session decode. Requests are processed sequentially, and aggregate throughput is bounded by single-session speed regardless of hardware capability.

Three mechanisms drive our superlinear aggregate scaling:

1. **Overhead amortisation.** Routing kernel launch, thread synchronisation, and pipeline bookkeeping are paid once per layer regardless of batch size. At 24 contexts, the effective overhead per context drops from 1.05 ms to approximately 0.044 ms.

2. **Expert reuse.** With 128 experts and top-8 routing per token, each expert has a 6.25% activation probability per token. At 24 concurrent contexts, the probability that at least one context activates a given expert per layer is approximately 78%. Experts loaded for one context serve others for free.

3. **Increased arithmetic intensity.** Batched expert GEMMs process matrices rather than vectors, improving tensor core utilisation and shifting the workload closer to the compute-bound regime.

### 4.4 Prefill: The Compute-Bound Phase

During prefill, each expert receives approximately `batch_size / 16` tokens (given 128 experts with top-8 routing). At 4 contexts of 630 tokens each, approximately 156 tokens route to each expert, producing an arithmetic intensity of approximately 624 FLOPs/byte — well above the compute-bound crossover for all architectures.

| Architecture | Peak Compute | Bandwidth | Crossover (FLOPs/byte) | Compute-bound when |
|-------------|-------------|-----------|----------------------|-------------------|
| Mac M2 Ultra | 27 TFLOPS FP16 | 800 GB/s | 34 | batch > 9 tok/expert |
| RTX 4090 FP16 | 82.6 TFLOPS | 1,008 GB/s | 82 | batch > 21 tok/expert |
| RTX 4090 INT8 | 330 TFLOPS | 1,008 GB/s | 327 | batch > 82 tok/expert |
| A100 FP16 | 312 TFLOPS | 2 TB/s | 156 | batch > 39 tok/expert |

With 156 tokens per expert during 4-context prefill, all architectures are deeply compute-bound. The RTX 4090's 82.6 TFLOPS delivers 3× the throughput of the M2 Ultra's 27 TFLOPS — but only when cache hits are high. On the 30B model where the full working set fits in cache, this advantage is fully realised (2,608 t/s prefill). On the 235B model with 10–50% cache coverage, DMA penalties during prefill partially offset the compute advantage.

### 4.5 Decode Remains Memory-Bound at All Practical Scales

A natural question is whether batching enough contexts makes decode compute-bound, fully utilising the GPU's tensor cores. The answer is no — not at practical concurrency levels.

With 128 experts and top-8 routing, each expert sees `N/16` tokens on average at N concurrent contexts. For decode to become compute-bound:

| Architecture | Compute-bound when N > |
|-------------|----------------------|
| Mac M2 Ultra | 144 contexts |
| RTX 4090 FP16 | 336 contexts |
| RTX 4090 INT8 | 1,312 contexts |
| A100 FP16 | 624 contexts |

All architectures remain memory-bound during decode at practical concurrency levels. The only thing that matters for decode throughput is **effective bandwidth to expert data**, which is determined by cache hit rate and VRAM bandwidth for our framework, versus raw memory bandwidth for unified memory.

---

## 5. Scaling to Larger Models: Qwen3-235B-A22B

### 5.1 Architecture

Qwen3-235B-A22B shares the same MoE architecture as the 30B variant (128 experts/layer, top-8 routing, 4-head GQA) but with 94 layers, 4,096 hidden dimension, and expert FFN dimension of approximately 1,536. Each expert is approximately 11.6 MB at Q4_K. The total expert population is 94 × 128 = 12,032.

### 5.2 VRAM Budget Analysis

| Component | 16 GB (4090 Laptop) | 24 GB (4090 Desktop) | 80 GB (A100) |
|-----------|-------------------|---------------------|-------------|
| Backbone (attn + embed) | ~6 GB | ~6 GB | ~6 GB |
| KV cache (Q4_0 × 4ctx) | ~0.3 GB | ~0.3 GB | ~0.7 GB |
| Compute buffers | ~2.5 GB | ~2.5 GB | ~2.5 GB |
| **Available for expert cache** | **~7.2 GB** | **~15.2 GB** | **~70.8 GB** |
| Expert slots | ~621 | ~1,310 | ~6,103 |
| Cache coverage | 5.2% | 10.9% | 50.7% |

### 5.3 Projected Performance

Using empirically measured per-layer overhead (~1.2 ms/layer wall-clock on a single context for the 30B model) and scaling for the 235B's increased layer count and expert GEMM size:

| Configuration | Single-ctx (t/s) | 4-ctx aggregate | 24-ctx aggregate |
|--------------|-----------------|----------------|-----------------|
| 24 GB (4090 Desktop) | 5–7 | 20–24 | — |
| 80 GB (A100) | 9–10 | 36–40 | 144–168 |
| M2 Ultra (132 GB unified) | 28 | 28 (sequential) | 28 (sequential) |

The crossover point where our framework overtakes the M2 Ultra in aggregate throughput shifts from ~8 concurrent sessions (30B) to ~4–5 sessions (235B on A100), because the larger model magnifies the cache advantage and batching benefits compound with more layers.

---

## 6. Hardware Trends and Future Implications

### 6.1 The Persistent Bandwidth Tier Gap

A natural question is whether hardware evolution will eliminate the need for expert caching. We argue it will not.

| Generation | VRAM BW | Interconnect | Tier Ratio |
|-----------|---------|-------------|-----------|
| Current (GDDR6X + PCIe 4.0) | 1.0 TB/s | ~13 GB/s | 77× |
| Next (GDDR7 + PCIe 5.0) | 1.8 TB/s | ~30 GB/s | 60× |
| 2027 (GDDR7+ + PCIe 6.0) | 2.5 TB/s | ~60 GB/s | 42× |
| Datacenter (HBM4 + PCIe 6.0) | 8.0 TB/s | ~60 GB/s | 133× |

The tier gap does not converge. For consumer GPUs, the ratio shrinks slowly from 77× to 42× over three generations. For datacenter GPUs, it actually **widens** because HBM bandwidth scales faster than PCIe.

This persistence is driven by physics: high bandwidth requires proximity between compute and memory. HBM achieves 4+ TB/s by stacking memory dies on the GPU interposer with thousands of pins over millimetre distances. PCIe achieves its bandwidth over centimetre-to-metre traces using serialised lanes. These are fundamentally different physical regimes. They converge only when compute and memory are co-located on the same die, as in unified memory architectures — but unified memory carries its own tradeoffs in compute density and thermal constraints.

### 6.2 GPU Compute Scaling Divergence

Consumer GPU compute scales at approximately 2.5× per generation, significantly faster than Apple Silicon at approximately 1.5× per generation:

| Hardware | FP16 TFLOPS | Generation | Scaling vs Baseline |
|---------|-------------|-----------|-------------------|
| RTX 4090 | 82.6 | Current | 1.0× |
| RTX 5090 | ~210 | Next | 2.5× |
| RTX 6090 (proj.) | ~450 | 2027–28 | 5.5× |
| M2 Ultra | 27 | Current | 1.0× |
| M3 Ultra | ~40 | Next | 1.5× |
| M4 Ultra (proj.) | ~60 | 2027 | 2.2× |

This divergence directly impacts prefill throughput. Projected prefill performance for the 235B model (compute-bound phase, high cache hit rate):

| Hardware | Projected Prefill (t/s) |
|---------|----------------------|
| M4 Ultra (~60 TFLOPS) | ~600 |
| RTX 4090 (82.6 TFLOPS) | ~850 |
| RTX 5090 (~210 TFLOPS) | ~2,100 |
| RTX 6090 (~450 TFLOPS) | ~4,500 |

The consumer GPU pulls further ahead every generation. Prefill speed directly determines how fast NPCs, agents, or users can process their input context before the model begins generating — a critical latency metric for interactive applications.

### 6.3 Three Compounding Forces

The hardware trends create three compounding forces favouring the batched caching approach:

**Force 1: Interconnect bandwidth improves.** PCIe 5.0 and 6.0 reduce the per-miss DMA penalty from 0.9 ms to 0.2 ms, raising single-context decode speed above usability thresholds (~8–10 t/s for real-time dialogue) for larger models. Once single-context crosses this threshold, the competitive question flips from "how fast is one session" to "how many sessions can I serve."

**Force 2: VRAM bandwidth improves.** GDDR7 at 1.8 TB/s increases memory-bound decode throughput, providing more headroom per context before bandwidth saturation. This enables packing more concurrent sessions before per-context speed drops below usable levels.

**Force 3: GPU compute grows faster than unified memory compute.** The FLOPS gap widens every generation (5.5× vs 2.2× after three generations), making prefill and compute-dense phases increasingly favour the discrete GPU. The same FLOPS that go unused during single-session decode become fully utilised when batching across 24, 48, or 128 concurrent sessions.

### 6.4 Projected System Performance (2027–28)

Two or three generations from now, a consumer GPU has 48 GB GDDR7 at 3 TB/s, approximately 450 TFLOPS FP16, and PCIe 6.0 at 60 GB/s:

| Metric | Projected (RTX 6090 + 235B equivalent) | Mac (M5 Ultra equivalent) |
|--------|----------------------------------------|--------------------------|
| Single-ctx decode | 15–20 t/s | ~40 t/s |
| 48-ctx aggregate decode | 400–600 t/s | ~40 t/s (sequential) |
| Prefill | 4,000+ t/s | ~600 t/s |
| Expert cache coverage | 50%+ | N/A (all in memory) |

The unified memory approach delivers excellent single-session quality but cannot scale aggregate throughput with concurrency. The batched caching approach delivers 10–15× higher aggregate throughput by exploiting the structural properties of discrete GPU architectures.

---

## 7. Related Work

**MoE architectures.** Modern gated MoE designs including GShard, Switch Transformer, and the DeepSeekMoE architecture demonstrate the effectiveness of sparse expert routing for scaling language models. The Qwen3 model family employs top-K routing with 128 experts per layer, following the design validated in Qwen2.5-MoE.

**Expert offloading.** KTransformers computes expert GEMMs on the CPU using Intel AMX instructions, achieving 13.8 t/s single-context and 24.4 t/s at 4 contexts on Qwen3-235B-A22B with a Xeon Platinum and RTX 4090. This avoids the PCIe bottleneck entirely but is limited by CPU compute throughput. Our approach keeps expert computation on the GPU where tensor cores provide 8–30× higher throughput.

**Layer-level offloading.** llama.cpp supports coarse-grained layer offloading via the `-ngl` and `-ot` flags, assigning entire layers or specific tensor types to CPU or GPU. This does not exploit dynamic expert-level caching or cross-context batching.

**Unified memory inference.** MLX, Ollama, and LM Studio on Apple Silicon provide excellent single-session latency. However, none implement batched multi-session inference, limiting aggregate throughput to single-session rates.

**Virtual memory for ML.** FlexGen and similar systems implement virtual memory abstractions for large model inference, but target throughput-oriented batch processing rather than latency-sensitive concurrent serving. Our framework targets the real-time interactive regime where per-context latency and aggregate throughput must both be optimised.

---

## 8. Future Work

**Multi-stream expert compute.** The current implementation serialises attention (forward thread) and expert GEMM (pipeline thread). Overlapping layer N+1 attention on stream 0 with layer N expert computation on a dedicated stream 3, synchronised via GPU-side events before the residual add, could reduce per-layer time from approximately 0.85 ms to approximately 0.40 ms:

```
Layer N:   [attention]──────────────────[residual add]──[attention N+1]
                    \                  /
Stream 3:            [expert GEMMs]──[done event]
```

This is the single largest remaining architectural optimisation.

**Adaptive model routing.** In applications with heterogeneous quality requirements (e.g., major characters vs background actors in a game), a two-model architecture could route high-importance requests to a 235B model and low-importance requests to a 30B model, both served by the same framework on a single GPU with distinct expert cache partitions.

**Cross-context expert reuse tracking.** Explicit tracking of which contexts share expert activations could inform eviction policy, preferentially retaining experts that serve multiple active contexts. This is particularly relevant for applications where concurrent sessions share environmental context.

**Flash attention integration.** Integrating flash attention kernels for the attention computation during prefill would reduce the quadratic memory complexity and improve prefill throughput, particularly beneficial for long-context scenarios.

---

## 9. Conclusion

We have presented a two-tier expert caching framework that achieves 308.5 tokens/second aggregate decode throughput across 24 concurrent sessions of the Qwen3-30B-A3B model on a 16 GB consumer laptop GPU, exceeding the best published single-session throughput by 2.4× on hardware with 33% less VRAM. The framework combines smart eviction policies (reducing cache misses by 66%), fused scatter kernels (improving per-layer dispatch by 35%), KV cache quantisation (enabling 3× session scaling), and speculative prefetch via learned transition matrices, all orchestrated through a dual-thread execution model with asynchronous GPU-to-CPU routing via CUDA event synchronisation.

Our analysis across three competing paradigms — discrete GPU caching, unified memory, and CPU offload — reveals that while unified memory architectures achieve superior single-session latency due to their flat memory hierarchy, the batched caching approach delivers dramatically higher aggregate throughput through overhead amortisation, expert reuse across contexts, and improved tensor core utilisation via batched GEMMs.

Hardware trend analysis demonstrates that the bandwidth tier gap between VRAM and host memory is a **structural feature** of discrete GPU architectures, not a transitional limitation. VRAM bandwidth scales at least as fast as interconnect bandwidth, maintaining ratios of 42–133× across projected hardware generations. Simultaneously, GPU compute scaling (2.5× per generation) outpaces unified memory compute (1.5× per generation), widening the discrete GPU's advantage for batched workloads.

The convergence of these trends positions two-tier expert caching as a durable architectural pattern for MoE inference. As single-session speed crosses application-specific usability thresholds over the next 1–2 hardware generations, the competitive axis shifts from latency to throughput — where batched expert caching provides a decisive and widening advantage. The framework we describe is, in essence, a **virtual memory system for MoE inference**; and just as virtual memory did not become obsolete when physical RAM grew larger, expert caching will not become obsolete as VRAM grows, because model scale grows in tandem, and the physics of tiered memory hierarchies in discrete GPU systems endures.