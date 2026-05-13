# Async Fork-Join Expert Pipeline

## Design Philosophy

The MoE expert pipeline is not a synchronous layer-by-layer execution
model. It is an **async submission loop** where the CPU runs flat out
enqueuing work, and the GPU resolves execution order through stream
dependencies. The pipeline never stalls. There are no barriers. The only
synchronisation points are **conditional joins** — and even those are
fully async operations resolved on the GPU side.

The CPU is a **submission engine**. The GPU is a **dataflow machine**.

---

## The Problem Space

### Model geometry

Qwen3-30B-A3B: 48 transformer layers, **all 48 of which are MoE**
(`decoder_sparse_step: 1`, `mlp_only_layers: []`). Each MoE layer
has 128 experts with top-8 routing. Each expert is a SwiGLU FFN
(~3 MB at Q4_K_M quantisation). Total expert count:
48 × 128 = **6,144 experts**. Total expert weight:
6,144 × 3 MB ≈ **18 GB**.

All expert weights reside in system RAM at all times. The question is
never "where is the data" but "is it also in VRAM right now". At
18 GB of expert weight alone, no consumer GPU can cache every expert.
The DMA overlap machinery is always the primary execution path.

### VRAM budget and the cache regime

The VRAM budget allocated for expert caching (50% of available VRAM)
determines which regime the pipeline operates in:

```
Total experts:          6,144
Expert size (Q4_K_M):   ~3 MB each
Total expert weight:    ~18 GB

Scenario A — 24 GB VRAM card (e.g. RTX 4090):
  50% budget:           ~12 GB
  Cache slots:          ~4,000
  Slots vs experts:     4,000 < 6,144 (65% residency)
  Regime:               PARTIAL — DMA overlap active

Scenario B — 16 GB VRAM card (e.g. RTX 4080):
  50% budget:           ~8 GB
  Cache slots:          ~2,700
  Slots vs experts:     2,700 < 6,144 (44% residency)
  Regime:               PARTIAL — DMA overlap critical

Scenario C — 8 GB VRAM card (e.g. RTX 4060):
  50% budget:           ~4 GB
  Cache slots:          ~1,300
  Slots vs experts:     1,300 < 6,144 (21% residency)
  Regime:               PARTIAL — heavy DMA, overlap essential
```

With 6,144 experts at ~3 MB each, **no consumer GPU achieves full
expert residency**. Even a 24 GB card at 50% budget caches only 65%
of experts. The DMA overlap machinery is always the primary execution
path — this is not a fallback mode but the design centre.

The design should still detect the (unlikely) case where cache
slots ≥ total experts — e.g. a multi-GPU setup or a smaller model
variant — and elide the DMA/fence machinery. But for Qwen3-30B-A3B
on any single consumer GPU, the full pipeline is always active.

### The input

A batch of `T` tokens across `S` concurrent sessions:

```
         session 0    session 1    session 2    session 3
token 0  ┌─────────┬─────────┬─────────┬─────────┐
token 1  │         │         │         │         │
token 2  │  block  │  block  │  block  │  block  │
  ...    │  of     │  of     │  of     │  of     │
token N  │  tokens │  tokens │  tokens │  tokens │
         └─────────┴─────────┴─────────┴─────────┘
```

This 2D grid flows through 48 transformer layers. **All 48 are MoE
layers**. At each layer, the router scatters the tokens across
~60–100 of the 128 experts. Each expert processes its subset and
the results are gathered back. Then the next layer begins.

### The data dependency

Layer L+1 depends on layer L's output:

```
h_L+1 = LayerNorm(h_L + attn(h_L) + MoE(attn_out))
```

This is an irreducible sequential dependency. A token cannot enter
layer L+1 until layer L is complete. This is not a problem to solve —
it is a constraint to design around.

---

## The Fork-Join Model

### Each layer is a fork and a join

At each MoE block, every token **forks** to 8 experts simultaneously.
Each fork is an independent computation. The layer completes when all
forks **join** — their weighted outputs are summed back into the token's
hidden state.

```
token ──── fork ──┬──► E_3  ──► compute ──┐
                  ├──► E_7  ──► compute ──┤
                  ├──► E_12 ──► compute ──┤
                  ├──► E_45 ──► compute ──┤
                  ├──► E_51 ──► compute ──┤
                  ├──► E_88 ──► compute ──┤
                  ├──► E_99 ──► compute ──┤  (may need DMA first)
                  └──► E_101 ─► compute ──┤  (may need DMA first)
                                          │
                                          ▼
                                   join (weighted sum)
                                          │
                                   layer output
```

The critical insight: **the fork paths are independent**. The 6 experts
already in VRAM execute immediately. The 2 that need DMA transfer
concurrently. The pipeline never stalls — it processes whatever is
ready, and the join is just the natural point where all paths converge.

### The join is not a barrier

The join (weighted sum via `index_add`) is itself an async GPU operation.
The CPU submits it and moves on. The GPU executes it in stream order
after all expert compute kernels complete. No CPU wait. No pipeline
drain. Just another kernel in the submission queue.

```
CPU timeline:
  submit(route) → submit(compute E_3) → submit(compute E_7) → ...
  → submit(DMA E_99) → submit(stream_wait_event) → submit(compute E_99)
  → submit(index_add for all) → submit(next layer attention) → ...
  ▲
  │ CPU never blocks. Every call returns immediately.

GPU timeline:
  [route][E_3 compute][E_7 compute]...[fence]...[E_99 compute][index_add][attn L+1]
                                         ▲
                                         │ GPU stream waits here for DMA
                                         │ but CPU is already submitting L+1
```

### Why the join is naturally hidden

In the partial-residency regime (cache slots < total experts), not all
8 selected experts will be in VRAM. But the probability that *all 8*
are cold is vanishingly small. In practice, 5–7 of the 8 are already
resident.

While the hot experts compute sequentially (or as a grouped GEMM), the
1–2 cold experts transfer via DMA on a separate stream. The DMA and
compute run **in parallel on different hardware**: the PCIe controller
handles the transfer while the GPU SMs handle the matmuls.

```
GPU compute stream:  [E_3][E_7][E_12][E_45][E_51][E_88]
                                                        ▲ fence wait
GPU DMA stream:      [═══ E_99 DMA ═══][═ E_101 DMA ═]  │
                                                         ▼
GPU compute stream:                                     [E_99][E_101][join]
```

Each cold expert is ~3 MB at Q4_K_M. PCIe 4.0 x16 transfers at ~25 GB/s,
so one expert loads in ~120μs. Six hot expert SwiGLU matmuls at small
batch sizes take ~120–300μs total. The DMA completes within the time
the hot experts are computing — the join wait is typically zero.

---

## Expert State Machine

Each of the 6,144 experts (48 MoE layers × 128 experts) has two
independent properties: where are its weights (RAM only, transferring,
or in VRAM) and does it have work (idle, queued, or executing). These
combine into five valid states.

### States

| State | Weights | Work | Description |
|-------|---------|------|-------------|
| **COLD** | RAM only | none | Default. Not in VRAM, not needed. |
| **LOADING** | DMA in flight | queued | Selected by router, DMA started, work waiting. |
| **WARM** | in VRAM | none | Loaded but idle. Eviction candidate. |
| **READY** | in VRAM | queued | Work waiting, weights present. Can dispatch immediately. |
| **RUNNING** | in VRAM | executing | GPU actively computing through this expert. |

There is no "has work but no weights and no DMA" state. The moment work
is assigned to a cold expert, DMA is initiated — the transition is
COLD → LOADING, never COLD → stuck.

### Transitions

```
                         ┌───────────────────────────────────────┐
                         │                                       │
                         ▼                                       │
                   ┌─────────┐                                   │
         ┌─────────│  COLD   │◄──── evict (score-based, VRAM pressure)  │
         │         └────┬────┘                                   │
         │              │                                        │
         │              │ router selects + DMA starts            │
         │              ▼                                        │
         │         ┌──────────┐                                  │
         │         │ LOADING  │  weights transferring,           │
         │         │          │  work accumulating in queue      │
         │         └────┬─────┘                                  │
         │              │                                        │
         │              │ DMA complete (fence signals)           │
         │              ▼                                        │
         │         ┌─────────┐   more work   ┌─────────┐         │
         │    ┌───►│  READY  │◄──────────────│ RUNNING │───┐     │
         │    │    └────┬────┘               └────▲────┘   │     │
         │    │         │                         │        │     │
         │    │         │ dispatch                │        │     │
         │    │         └─────────────────────────┘        │     │
         │    │                                            │     │
         │    │    work arrives                            │     │
         │    │    for WARM expert                         │     │
         │    │         │                                  │     │
         │    │         │        ┌─────────┐               │     │
         │    └─────────┼─────── │  WARM   │◄──────────────┘     │
         │              │        └────┬────┘  queue drained      │
         │              │             │                          │
         │              │             │ evict                    │
         │              └─────────────┴──────────────────────────┘
         │                            │
         └────────────────────────────┘
```

### The fast path: WARM → READY → RUNNING → WARM

For frequently-used experts, weights stay resident. Work arrives, the
expert dispatches immediately, completes, returns to idle. No DMA, no
allocation. Pure compute throughput.

### The slow path: COLD → LOADING → READY → RUNNING

A cache miss. DMA transfers the expert from RAM to VRAM. But this path
runs **concurrently with** the fast path — cold expert DMA overlaps
with hot expert compute. The slow path's latency is hidden by the fork
parallelism.

### Eviction constraints

Only WARM experts may be evicted. LOADING, READY, and RUNNING experts
have active work or in-flight transfers and must not be disturbed.

```
Evictable:     WARM only (ordered by eviction score)
Protected:     LOADING, READY, RUNNING
```

---

## The Async Submission Loop

The CPU executes a tight loop that never blocks. Every GPU operation
is submitted asynchronously and returns immediately. The GPU resolves
ordering through stream dependencies.

### Per-layer iteration

```
for each layer L:

    1. ROUTE
       Submit: gate matmul → softmax → top-k
       All async GPU kernels. CPU does not wait for results.

    2. READ ROUTING (the one sync point)
       Pull expert IDs from GPU → CPU.
       This is a small transfer (~1 KB) that tells the CPU
       which experts each token selected. The CPU needs this
       to make cache scheduling decisions.

    3. CLASSIFY
       For each selected expert, check its state:
       - WARM → READY (instant, just enqueue work)
       - COLD → LOADING (kick DMA, enqueue work)
       Group experts into ready_set and loading_set.

    4. SUBMIT PHASE 1 — ready experts
       Submit compute kernels for all READY experts.
       Submit DMA transfers for all LOADING experts.
       Both run concurrently on separate GPU streams.
       CPU returns immediately from all submissions.

    5. SUBMIT FENCE
       Submit stream-wait-event: compute stream waits for
       DMA stream completion. This is a GPU-side dependency —
       the CPU submits it and moves on instantly.

    6. SUBMIT PHASE 2 — newly ready experts
       Submit compute kernels for experts that were LOADING.
       These will execute after the fence resolves.

    7. SUBMIT JOIN
       Submit index_add kernels that scatter expert outputs
       back into the layer output tensor.
       The next layer's attention is queued after this.

    → loop to layer L+1
```

Every step except step 2 is a non-blocking submission. The CPU races
ahead, building up a deep queue of GPU work. The GPU executes it in
order, resolving data dependencies through stream events.

### What the sync point actually costs

Step 2 pulls a `[num_tokens, 8]` tensor of u32 expert indices from GPU
to CPU. For a batch of 496 tokens: 496 × 8 × 4 bytes = 15.9 KB.

The transfer itself is near-instant over PCIe. The cost is not the
transfer — it's the **pipeline drain**: the GPU must finish the routing
kernels before the data is available, so the CPU waits for the routing
to complete. This is typically ~10–50μs.

However, this drain happens while the GPU has no expert compute queued
yet (it just finished routing), so there is no useful work being
preempted. The drain is unavoidable but not wasteful.

---

## The Routing Topology

### Each router is a fixed linear map

The gate at each layer is a learned `Linear(hidden_dim, 128)` with
frozen weights at inference time. It maps a hidden state to 128 expert
logits, then softmax + top-k selects the winners.

```
                     gate_L weights: [hidden_dim, 128]
                     (fixed at inference time)

hidden state ──► matmul ──► [128 logits] ──► softmax ──► top-8
   [1, D]        gate_L                                   │
                                                          ▼
                                               expert IDs + weights
```

### Experts produce characteristic output subspaces

Each expert is a fixed SwiGLU transform:

```
Expert E_i at layer L:
   input x ──► gate_proj(x) ──► silu ──► * up_proj(x) ──► down_proj ──► output
               [D → D_ff]                 [D → D_ff]      [D_ff → D]
```

Because the projection matrices are frozen, expert E_i maps its input
to a **characteristic output subspace**. Different experts produce
outputs that cluster in different regions of the hidden dimension.

### Static expert-to-expert transition probabilities

Layer L+1's gate is also frozen. If expert E_i at layer L produces
outputs in subspace S_i, and layer L+1's gate maps that subspace to
logits, then the transition probability from E_i at layer L to each
expert at layer L+1 is statistically predictable.

```
Layer L                          Layer L+1
┌──────┐                         ┌──────┐
│ E_3  │──── outputs in ────────►│gate  │──► P(E_7)  = 0.31  ← likely
│      │     subspace S_3        │ L+1  │──► P(E_22) = 0.18
└──────┘                         │      │──► P(E_91) = 0.12
                                 │      │──► P(E_*)  = tiny  ← the other 125
┌──────┐                         │      │
│ E_12 │──── outputs in ────────►│      │──► P(E_44) = 0.28  ← different set
│      │     subspace S_12       │      │──► P(E_7)  = 0.22
└──────┘                         └──────┘
```

This defines a static transition matrix per adjacent MoE layer pair:

```
T[M→M+1] : [128 × 128]
T[M→M+1][i][j] = P(token routed to E_j at M+1 | routed to E_i at M)
```

This matrix is an emergent property of the frozen weights, measurable
empirically by recording routing decisions over representative inputs.

**Complication**: the MoE output is added to the residual stream, not
substituted. The residual carries information from all prior layers,
diluting the expert-to-expert correlation. Between layer L and L+1,
attention transforms the hidden state before the next router sees it.
But the expert's contribution is additive and shifts the routing logits
in a consistent direction, so the transition probabilities remain
peaked. The transition matrix captures the net effect of expert L's
output *plus* the intervening attention.

### The routing DAG

Each token traces a path through the expert graph across all 48
layers:

```
Token 0:  E_3  ──► E_7  ──► E_91 ──► E_12 ──► ... ──► E_44
          L0       L1       L2       L3              L47

Token 1:  E_3  ──► E_22 ──► E_5  ──► E_12 ──► ... ──► E_88
          L0       L1       L2       L3              L47
```

(L0–L47 = all 48 transformer layers, each of which is MoE)

Across all tokens, this forms a directed acyclic graph:

- Nodes: `(layer, expert)` pairs — 48 × 128 = 6,144 possible
- Edges: token flows between adjacent MoE layers
- Each token produces 8 edges per MoE layer (top-8 routing)
- The graph is sparse — the transition matrix is peaked, not uniform

```
M0          M1          M2          M3          ...
E_0  ─────► E_0  ─────► E_0  ─────► E_0
  \          ╲           │ ╲         ╱
   \          ╲          │  ╲       ╱
E_1  ──────► E_1  ─────► E_1  ──► E_1
   ╲    ╱╲     ╲     ╱    ╲
    ╲  ╱  ╲     ╲   ╱      ╲
E_2  ╳    ►E_2  ──► E_2  ──► E_2
    ╱ ╲      ╱       │╲
   ╱   ╲    ╱        │ ╲
E_3  ──► E_3  ─────► E_3  ──► E_3
 ...     ...         ...       ...
E_127 ─► E_127 ───► E_127 ─► E_127

(L0–L3 = all layers are MoE; 47 transition matrices L→L+1 total)
```

### What the topology enables

The transition matrix makes the pipeline **predictable**:

1. **Prefetching**: if the active set at layer L is known, the transition
   matrix predicts which experts layer L+1 will likely need. Their DMA
   can begin while layer L computes — converting cold misses into warm
   hits before they happen.

2. **Pinning**: if the transition graph has a "hot core" of experts that
   appear in most routing paths, those experts can be permanently pinned
   in VRAM. The cache only manages the long tail.

3. **Clustering**: groups of experts that co-occur across layers form
   natural pathways. Cache eviction can respect cluster boundaries to
   avoid thrashing within a pathway.

---

## Grouped GEMM: The Expert-Centric Inversion

### The duality

There are two equivalent ways to view the MoE computation:

```
Token-centric:
  "I have T tokens. For each token, which 8 experts does it need?"
  Inner loop: for each expert, gather its tokens, compute one matmul.
  Result: 60–100 small sequential kernel launches per layer.

Expert-centric:
  "I have E ready experts. Each has a queue of tokens to process."
  Inner loop: gather all ready experts, launch one grouped GEMM.
  Result: 1–2 large kernel launches per layer.
```

### Sequential dispatch

```
Time ──────────────────────────────────────────────────────────────►

GPU:  [E_3 matmul][E_7 matmul][E_12 matmul][E_45 matmul] ...
       ▲            ▲            ▲            ▲
       2–8 tokens    2–8 tokens   2–8 tokens   2–8 tokens
```

Each expert processes a tiny batch. The GPU is underutilised — the
matmul is memory-bandwidth-bound at these sizes, and kernel launch
overhead is significant relative to the work.

### Grouped dispatch

```
Time ──────────────────────────────────────────────────────────────►

GPU:  [═══════ grouped GEMM: E_3, E_7, E_12, E_45, ... ═══════]
       ▲
       all expert matmuls fused into one kernel launch
       GPU fully utilised across all SMs
```

The grouped GEMM handles **ragged batches** — each expert has a different
number of tokens. The kernel receives a list of (expert_weights,
token_batch, routing_weights, scatter_indices) segments and executes
them all in a single launch.

### Two-phase dispatch with the submission loop

The grouped GEMM integrates naturally with the async fork-join model:

```
for each layer L:

    1. ROUTE → top-k (async)
    2. READ ROUTING → classify experts (one small sync)

    3. PHASE 1:
       Submit grouped GEMM for all READY experts    ─┐ concurrent
       Submit DMA for all COLD→LOADING experts       ─┘

    4. SUBMIT FENCE (async GPU-side wait)

    5. PHASE 2:
       Submit grouped GEMM for newly-READY experts

    6. SUBMIT JOIN (async index_add scatter)
```

Phase 1 and DMA run concurrently. By the time phase 1 completes, phase 2's
experts are ready. The GPU compute stream is never idle — it always has
a grouped GEMM to execute.

### Why this matters for offloaded MoE

For a model that fits entirely in VRAM, sequential per-expert dispatch
is adequate — the GPU stays busy. But for offloaded MoE where experts
cycle through VRAM:

- **Fewer kernel launches**: 2 per layer instead of 60–100. Less CPU
  scheduling overhead, less kernel launch latency.

- **Better SM utilisation**: a grouped GEMM uses all SMs across all
  expert segments, rather than one small matmul underutilising the GPU.

- **Natural batching window**: while phase 1 computes, DMA completes
  for loading experts. The DMA latency becomes a natural gather window
  for accumulating work.

- **Scales with concurrency**: more concurrent requests means bigger
  per-expert token batches, which means better arithmetic intensity
  per GEMM segment.

---

## Cross-Request Expert Coalescing

With multiple concurrent requests at different points in the pipeline,
the same expert may be needed by multiple requests simultaneously:

```
Request A at layer 5 ──► needs experts [3, 7, 12]
Request B at layer 3 ──► needs experts [7, 44, 12]
                                        ▲       ▲
                                   shared experts
```

The expert-centric model handles this naturally: expert 7's queue
has work from both requests. It processes them together in one matmul,
amortising the cost of having the weights in VRAM.

This is the **true batch dimension** for MoE inference. In a dense model,
batching means processing multiple tokens through the same weight matrix.
In MoE, batching means processing multiple tokens through the same
expert — and that only happens when the expert queue aggregates work
from multiple sources.

### Device-level stall freedom

Even if an individual request hits a cold expert, the device never
stalls. While request A waits for a DMA at layer 5, requests B, C, D
are computing at layers 3, 7, 12 using hot experts. The GPU compute
stream always has READY work to dispatch. The DMA stream is always
filling experts that some future request will need.

Individual requests experience latency from cold hits. The GPU itself
experiences zero idle time. This is latency hiding through concurrency —
the same principle as GPU warp scheduling hiding memory access latency.

---

## The Background Submission Thread

### Why a background thread

The async submission loop described above — route, classify, submit
phase 1, fence, submit phase 2, join — is a CPU-side state machine
that runs through all 48 MoE layers sequentially. This loop has no
reason to live on the caller's thread. The caller (inference engine,
HTTP handler, chat session) wants to submit a chunk of tokens and
receive the result when it's ready. It does not want to drive the
GPU scheduling loop itself.

The submission loop is a **background thread** that owns the GPU
streams, the expert cache, and the state machine. The caller interacts
with it through two boundaries:

```
                 ┌──────────────────────────────────────┐
  Caller         │       Expert Pipeline Thread         │
  thread         │                                      │
                 │  owns:                               │
  submit ───────►│    - compute stream                  │
  (flat tensor   │    - DMA stream                      │
   + metadata)   │    - expert cache (6,144 experts)    │
                 │    - per-expert state machines       │
                 │    - routing gates for all 48 layers │
                 │                                      │
                 │  runs:                               │
                 │    the async submission loop         │
                 │    (route → classify → dispatch →    │
                 │     fence → dispatch → join) × 48    │
                 │                                      │
  result ◄───────│  produces:                           │
  (future/       │    output tensor, same shape as      │
   oneshot)      │    input                             │
                 └──────────────────────────────────────┘
```

### The submission boundary

The caller submits a **work chunk**: a flat tensor of hidden states
that needs to flow through all 48 layers (every layer is MoE in
this model, so the pipeline processes the full forward pass).

The simplest design: the caller submits the full token batch for one
forward pass and gets back a future that resolves when the pipeline
has processed all 48 layers.

```
Caller:
    let future = pipeline.submit(hidden_states, attention_mask);
    // ... do other work, handle other requests ...
    let output = future.await;   // or future.get() in sync context
```

Since every layer is MoE in Qwen3-30B-A3B, the pipeline naturally
owns the full forward pass. Two architectural options:

```
Option A — pipeline owns the full forward pass:
    Caller submits raw token embeddings.
    Pipeline runs attention + MoE for all 48 layers.
    Pipeline returns final hidden states.
    Clean separation. The pipeline owns the entire layer loop.

Option B — pipeline owns only MoE, caller drives the loop:
    Caller submits hidden states for layer 0's MoE block.
    Gets a future. When it resolves, feeds result into attention
    for layer 1, then submits to layer 1's MoE block. Repeat.
    The background thread processes MoE blocks from multiple
    callers concurrently — true pipelining.
```

For a model where every layer is MoE, Option A is the cleaner fit.
The pipeline thread owns the layer loop and the expert cache. The
caller submits token embeddings and receives the final hidden states.
No interleaving of caller-driven attention and pipeline-driven MoE.

For a mixed-density model (some MoE, some dense), Option B would
be preferred — the caller drives the layer loop and only hands off
MoE blocks. The pipeline multiplexes MoE work from multiple callers.

### What the caller provides per submission

The caller provides a flat tensor that the pipeline must divide up:

```
Input:
    hidden_states:  [num_tokens, hidden_dim]   — GPU tensor
    moe_layer_idx:  usize                      — which of the 48 layers

Output (via future):
    moe_output:     [num_tokens, hidden_dim]   — GPU tensor
```

The pipeline internally:
1. Runs the routing gate to produce `[num_tokens, 8]` expert indices
   and `[num_tokens, 8]` routing weights
2. Divides the flat tensor into per-expert slices
3. Dispatches through the state machine
4. Joins results back into the output tensor

The caller never sees expert indices, routing weights, or per-expert
slices. The MoE block is a black box: tensor in, tensor out.

### Thread safety and the ownership boundary

The background thread **exclusively owns** the mutable state:
- The expert cache (score-based eviction, slot allocation)
- The DMA stream and its fences
- The per-expert state machines (COLD/LOADING/WARM/READY/RUNNING)

The caller **exclusively owns**:
- The hidden state tensors (until submitted)
- The future/oneshot handle

The submission channel transfers ownership of the input tensor to the
pipeline thread. The completion channel transfers ownership of the
output tensor back. No shared mutable state. No locks on the hot path.

```
Caller thread                Pipeline thread

  hidden_states ──────send──────► receives work
  (moves ownership)               routes, dispatches, joins
                                   output_tensor ──────send──────►
  future.await ◄──────────────────────────────────────────────────
  (receives ownership)
```

### Multiple concurrent callers

With multiple requests in flight, the pipeline thread has a queue of
pending MoE work items. It processes them in whatever order maximises
GPU utilisation — typically by coalescing experts shared across
requests (see Cross-Request Expert Coalescing above).

```
Request A submits MoE layer 3 ──►┐
Request B submits MoE layer 7 ──►├──► pipeline thread
Request C submits MoE layer 3 ──►┘     │
                                       │ coalesces A and C (same layer)
                                       │ interleaves B (different layer)
                                       │ maximises expert reuse
                                       ▼
                              dispatches grouped GEMMs
```

---

## Expert Registration and Construction

### The builder abstraction

The model loading code needs to construct experts and register them
with the pipeline. Today, this happens in two paths:

1. **Reader path** (all experts fit in VRAM): load each expert's three
   weight tensors (gate_proj, up_proj, down_proj) directly to VRAM,
   wrap in quantised matmul handles, store in the cache.

2. **Mmap path** (experts larger than VRAM): compute byte offsets into
   the memory-mapped file for each expert's three projections, store
   the offsets, and load on demand via DMA.

Both paths produce the same thing: a set of expert weight descriptors
that the pipeline can use to compute SwiGLU. The builder should
abstract over the storage backend.

### What the builder receives

For each MoE layer, the model loading code knows:
- The number of experts (128 for Qwen3-30B-A3B)
- The top-k routing parameter (8)
- The gate weights (a `[hidden_dim, num_experts]` tensor)
- For each expert, the three projection weight matrices

The weight matrices arrive in one of two forms:

```
Form A — pre-loaded tensors (reader path):
    For expert j:
        gate_proj: QMatMul   (already in VRAM)
        up_proj:   QMatMul   (already in VRAM)
        down_proj: QMatMul   (already in VRAM)

Form B — mmap byte references (mmap path):
    For expert j:
        gate_proj: (offset, length, shape, dtype) into mmap
        up_proj:   (offset, length, shape, dtype) into mmap
        down_proj: (offset, length, shape, dtype) into mmap
```

The GGUF file may store experts as:
- **3D merged tensors**: `ffn_gate_exps.weight` with shape
  `[num_experts, intermediate_size, hidden_size]`, which the loader
  splits into per-expert byte slices
- **2D per-expert tensors**: `ffn_gate.{j}.weight` with shape
  `[intermediate_size, hidden_size]`, one per expert

The builder handles both layouts transparently.

### The registration flow

```
model loading code                     pipeline builder

  for each MoE layer (0..48):
    ┌──────────────────────────┐
    │ load gate weights        │
    │ determine expert layout  │──────► builder.register_moe_layer(
    │ (merged 3D or per-expert)│           layer_idx,
    │                          │           gate_weights,
    └──────────────────────────┘           expert_weights_or_refs
                                       )
                                           │
                                           ▼
                                       builder accumulates:
                                         - gate for this layer
                                         - 128 expert descriptors
                                           (either VRAM tensors
                                            or mmap references)

  after all 48 layers registered:
    ┌────────────────────┐
    │ builder.build()    │──────────► creates:
    │                    │             - ExpertPipeline (the thread)
    │                    │             - ExpertCache (shared across layers)
    │                    │             - per-expert state machines (×6,144)
    │                    │             - submit/result channels
    └────────────────────┘

  caller receives:
    pipeline_handle: ExpertPipelineHandle
      .submit(hidden_states, moe_layer_idx) → Future<Tensor>
      .shutdown() → join the background thread
```

### Regime detection at build time

The builder knows:
- Total expert count (num_moe_layers × num_experts)
- Expert size (from weight shapes and dtypes)
- Available VRAM budget (from device query)
- Whether all experts were provided as VRAM tensors (reader path)
  or as mmap references (mmap path)

At `build()` time, it can detect the cache regime:

```
if cache_slots >= total_experts:
    // FULL RESIDENCY — skip DMA machinery
    // Pre-load all experts at startup
    // State machine is trivially WARM → READY → RUNNING → WARM
    // No eviction, no DMA stream, no fences
    // Pipeline thread still exists (for the submission abstraction)
    // but its inner loop is simplified

else:
    // PARTIAL RESIDENCY — full pipeline
    // Allocate DMA stream, set up fences
    // Expert state machine tracks all 5 states
    // Score-based eviction active
    // Transition matrix prefetching available (if calibrated)
```

This means the model loading code doesn't need to know which regime
it's in. It calls the same builder API regardless. The builder
inspects the budget and constructs the appropriate pipeline variant.

### What the builder does NOT own

The builder constructs the expert pipeline, but does **not** own:
- The attention layers (self-attention, KV cache)
- The embedding and output projection layers

These remain in the model's `forward()` method. For Qwen3-30B-A3B
where every layer is MoE, the loop is uniform:

```
Model forward():
    for layer in 0..48:
        x = attention(x)                              // model owns this
        x = pipeline.submit(x, layer_idx).await        // pipeline owns this
    return x
```

---

## Locking, Queues, and the Zero-Contention Architecture

### What the background thread eliminates

The current implementation has a `RwLock<ExpertCacheInner>` that every
`SparseMoeBlock::forward()` call must acquire — a write-lock for score
promotion on cache hits, a write-lock for slot reservation on misses,
and a write-lock again to install newly-loaded slots. The `Arc<ExpertSlot>`
wrappers exist specifically to let compute proceed after releasing the
lock, but the lock acquisitions themselves are unavoidable contention
points.

With a single background thread owning all mutable state, **every lock
disappears**:

```
Current architecture (original, pre-pipeline):
    ExpertCache {
        mmap:      Arc<Mmap>,              // immutable, shared
        host_refs: Vec<Vec<MmapExpertRef>>, // immutable, shared
        device:    Device,                  // immutable, shared
        inner:     RwLock<ExpertCacheInner>, // ← contention point
    }
    ExpertCacheInner {
        slots:       Vec<Option<Arc<ExpertSlot>>>,  // Arc to survive lock release
        free_slots:  Vec<usize>,
        lru:         LruCache<(usize, usize), usize>,  // (original design)
        slot_to_key: Vec<Option<(usize, usize)>>,
    }

Background thread architecture (implemented — score-based eviction):
    ExpertCacheInner {
        slots:       Vec<Option<ExpertSlot>>,   // no Arc needed — sole owner
        free_slots:  Vec<usize>,
        key_to_slot: HashMap<(usize, usize), usize>,  // score-based eviction
        slot_to_key: Vec<Option<(usize, usize)>>,
        scores:      Vec<f32>,                  // per-expert eviction scores
    }
    // No RwLock. No Arc on slots. The thread owns it all.
    // &mut self everywhere. The compiler enforces exclusivity.
```

The `Arc<ExpertSlot>` wrapper was necessary because multiple callers
on multiple threads could hold references to the same slot while the
lock was released for compute. With a single thread, the slot is used,
the compute is submitted (async), and the thread moves on. No concurrent
access. No reference counting. No atomic operations on the hot path.

### What still needs to cross thread boundaries

Two things cross the boundary between caller threads and the pipeline
thread:

1. **Inbound**: the work submission (input tensor + metadata)
2. **Outbound**: the result (output tensor)

Everything else — cache lookups, score promotion, slot allocation,
eviction, DMA submission, fence management, expert state transitions —
happens exclusively on the pipeline thread with plain `&mut self`.

### The submission channel

The submission must be:
- **Non-blocking for the caller** (submit and return immediately)
- **Non-allocating on the hot path** (no heap allocation per submission)
- **Single-producer or multi-producer** (depending on concurrency model)
- **Wake the pipeline thread** if it's idle (but cheaply)

The natural primitive is a bounded MPSC channel. The caller sends a
work item, the pipeline thread drains all pending items in a batch.

But even a channel has overhead: mutex on the queue, allocation for
the item, a condvar or futex for wake-up. For the single-caller case
(one inference thread), this is overkill.

#### Lightweight option: ring buffer + atomic + futex

```
                    ┌──────────────────────────────────┐
Caller thread:      │   Ring buffer (fixed capacity N) │
                    │                                  │
  submit() ────────►│  [slot 0][slot 1]...[slot N-1]   │
    │               │     ▲                            │
    │ write item    │     │ read cursor (atomic)       │
    │ advance       │     │                            │
    │ write cursor  │  write cursor (atomic)           │
    │ (atomic)      └──────────────────────────────────┘
    │                         │
    │ futex_wake ─────────────┼──────► pipeline thread
    │ (only if thread         │         │
    │  was sleeping)          │         │ drain all items
    │                         │         │ between read and write cursor
    └─────────────────────────┘         │ (no lock, no allocation)
                                        ▼
                                    process batch
```

The ring buffer slots are pre-allocated at build time. Each slot holds:

```
WorkItem {
    hidden_states:  Tensor,       // moved in, not cloned
    moe_layer_idx:  u16,          // which MoE layer
    result_slot:    *mut Option<Tensor>,  // where to write the output
    wake:           Arc<Notify>,   // or a raw futex / eventfd
}
```

The hot path:
1. Caller writes into `ring[write_cursor % N]` (plain store)
2. Caller advances write_cursor (single `AtomicUsize::store(Release)`)
3. Caller wakes pipeline thread (futex_wake, **only if it was sleeping**)

No mutex. No allocation. Two atomic operations in the fast case (one
store, one compare-exchange on the futex). If the pipeline thread is
already running (processing a previous batch), step 3 is skipped
entirely — the thread will see the new item on its next drain pass.

#### The drain loop

The pipeline thread runs:

```
loop {
    // 1. Drain all pending submissions
    let read = read_cursor.load(Acquire);
    let write = write_cursor.load(Acquire);

    if read == write {
        // Nothing pending — sleep until woken
        futex_wait(&write_cursor, write);
        continue;
    }

    // 2. Process all items [read..write)
    for i in read..write {
        let item = ring[i % N].take();
        // ... route, classify, dispatch ...
    }

    // 3. Advance read cursor
    read_cursor.store(write, Release);
}
```

No lock anywhere in this loop. The atomics provide the necessary
ordering. The pipeline thread processes items in submission order,
and can coalesce multiple submissions that arrived while it was busy.

### The completion path

The result must flow back to the caller. Two options:

#### Option 1: Oneshot channel (simple, one allocation)

```
Caller:
    let (tx, rx) = oneshot::channel();
    pipeline.submit(hidden_states, moe_layer_idx, tx);
    let output = rx.await;

Pipeline thread:
    // ... compute ...
    tx.send(output_tensor);
```

One heap allocation for the channel. The caller blocks (or awaits)
on the receiver. Simple. The allocation is per-submission but amortised
over the entire MoE layer computation (~100μs+).

#### Option 2: Pre-allocated result slot (zero allocation)

```
Caller:
    let mut result: Option<Tensor> = None;
    pipeline.submit(hidden_states, moe_layer_idx, &mut result);
    pipeline.wait_for(submission_id);  // blocks until slot is filled
    let output = result.take().unwrap();

Pipeline thread:
    // ... compute ...
    *item.result_slot = Some(output_tensor);
    item.wake.notify();   // wake the caller
```

Zero allocation. The result is written directly into the caller's
stack frame. But this requires the caller's stack frame to outlive the
pipeline's processing — which it does, since the caller blocks until
the result arrives. The `wake` is a futex/condvar/Notify that the
caller sleeps on.

For the single-caller case, Option 2 is ideal. For multi-caller with
async runtimes, Option 1 integrates better with `await`.

### Expert queues: per-expert or per-layer?

Within the pipeline thread, after routing produces expert assignments,
the thread must group tokens by expert. Two designs:

#### Per-expert persistent queues (6,144 queues)

```
expert_queues: [Vec<WorkSlice>; 6144]

// After routing:
for (token_idx, expert_id, weight) in routing_decisions {
    expert_queues[moe_layer * 128 + expert_id].push(WorkSlice {
        token_idx,
        weight,
    });
}

// Dispatch:
for q in expert_queues.iter_mut().filter(|q| !q.is_empty()) {
    dispatch_expert(q);
    q.clear();   // reuse allocation
}
```

The `Vec` allocations persist across forward passes — `clear()` keeps
the capacity, so after the first pass there are zero allocations. Each
queue is a plain `Vec` (no synchronisation needed — single thread).

But 6,144 `Vec`s is a lot of bookkeeping. Most will be empty on any
given pass (only ~60–100 experts are active per layer). Iterating
over 6,144 queues to find the non-empty ones is wasteful.

#### Flat gather array (one per MoE layer)

Instead of per-expert queues, build a flat assignment array:

```
// Pre-allocated, reused every pass:
expert_assignments: Vec<(expert_id, token_idx, weight)>

// After routing — just append:
for (token_idx, expert_id, weight) in routing_decisions {
    expert_assignments.push((expert_id, token_idx, weight));
}

// Sort by expert_id (puts same-expert tokens together):
expert_assignments.sort_unstable_by_key(|a| a.expert_id);

// Now iterate contiguous expert groups:
for group in expert_assignments.group_by(|a, b| a.expert_id == b.expert_id) {
    dispatch_expert(group);
}

expert_assignments.clear();  // keep capacity
```

One `Vec`, one sort, one scan. No per-expert data structure. The sort
is `O(n log n)` where n = num_tokens × top_k (e.g. 496 × 8 = 3,968
items). A radix sort on 13-bit expert IDs (0..6144) would be `O(n)`.

This is the better design. No 6,144 empty `Vec`s. No sparse iteration.
Just a flat array, a sort, and contiguous groups ready for grouped GEMM.

#### But wait — the sort is already done on the GPU

The routing step produces `[num_tokens, top_k]` expert indices on the
GPU. If the GPU sorts these by expert_id (which it already does for
the grouped dispatch), the CPU just reads back the sorted assignment
list. No CPU-side sort needed.

```
GPU routing output (already sorted by expert_id):

  expert_id:  [3, 3, 3, 7, 7, 12, 12, 12, 12, 45, 45, ...]
  token_idx:  [0, 4, 7, 1, 5,  0,  2,  3,  6,  1,  4, ...]
  weight:     [.31, .28, .15, .42, .33, .18, .22, .19, .11, ...]

  group boundaries: [0, 3, 5, 9, 11, ...]   ← prefix sum
```

The CPU pulls this pre-sorted flat array and iterates contiguous
groups. The per-expert "queue" is just a slice into the sorted array.
No data structure at all — just pointer arithmetic.

### What about the expert state lookups?

The pipeline thread needs to check each active expert's state
(COLD/WARM/LOADING/READY/RUNNING) to decide what to dispatch. With
6,144 total experts, this is a flat array:

```
expert_states: [ExpertState; 6144]

// ExpertState is a single byte (5 variants)
// The full array is 6,144 bytes — fits in L1 cache
```

Checking an expert's state is a single array index. Transitioning
state is a single array write. No hash map. No tree. No lock. The
array is so small it stays cache-hot across the entire forward pass.

### The eviction-policy question

The original implementation used `lru::LruCache` — a hash map + doubly
linked list. With sole ownership, this was replaced by score-based eviction:

```
Current:
    lru: LruCache<(usize, usize), usize>
    // HashMap<K, *mut LruEntry> + doubly-linked list
    // Each access: hash + pointer chase + 4 pointer writes (unlink + relink)

Alternative — timestamp array:
    last_used: [u32; 6144]     // generation counter per expert
    generation: u32            // bumped each forward pass

    // Promote: last_used[expert] = generation;  (one write)
    // Evict: find min(last_used) among WARM experts (scan)
```

The timestamp approach trades `O(1)` eviction (linked list) for
`O(n)` scan, but n = 6,144 and the scan is a linear read over 24 KB
of contiguous memory — roughly 1μs. Eviction is rare relative to
promotion (every expert is promoted every time it's used; eviction
only happens on cache misses). The timestamp approach eliminates the
hash map, the linked list, and the pointer chasing, making promotion
(the hot path) a single array write instead of a hash lookup + four
pointer writes.

Whether this is worthwhile depends on eviction frequency. In the
full-residency regime, eviction never happens and the tracking structure
is pure overhead. In partial residency with 85% cache, eviction
happens ~15% of the time per MoE layer. The linear scan is fine.

### Summary: zero-lock, zero-allocation hot path

```
Operation             Current (multi-thread)        Pipeline thread (single-thread)
─────────────────────────────────────────────────────────────────────────────────────
Cache lookup          RwLock write-lock              Array index (1 cycle)
                      + HashMap get                  
Score promotion        RwLock write-lock              Array store (1 cycle)
                      + linked list unlink/relink    
Slot access           Arc::clone()                   Plain &slot reference
                      (atomic increment)             
Slot install          RwLock write-lock              Array store
                      + HashMap insert               
Eviction decision     RwLock write-lock              Linear scan over 6,144 bytes
                      + EvictionTracker::pop_lowest()          
Expert state          Not tracked (implicit)          Array index (1 cycle)
Token grouping        Per-expert Vec + push           Slice into GPU-sorted array
Work submission       —                               Ring buffer + 1 atomic
Result delivery       —                               Oneshot or result slot + 1 futex
```

The pipeline thread's inner loop touches:
- One 6,144-byte array (expert states)
- One 24 KB array (eviction scores)
- One flat array (GPU-sorted assignments — read only)
- The expert slot array (VRAM pointers — read only for dispatch)

Total working set: ~34 KB. Fits in L1. No locks, no atomics, no
reference counting, no hash maps, no linked lists. The thread is a
tight sequential state machine processing a flat data structure.

---

## Design Decisions (Resolved)

These questions were originally open. Each is now answered based on
analysis of the actual model code, the Qwen3-30B-A3B config, and the
design constraints established in this document.

---

### 1. How sparse is the transition matrix T[L→L+1]?

**Decision: assume moderately sparse, but do not rely on it.**

The transition matrix T[L→L+1] is defined by the composition of
expert L's output subspace with layer L+1's frozen gate weights. Each
expert produces outputs in a characteristic region of the hidden space,
and the next gate linearly maps that region to 128 logits. The sparsity
depends on how clustered those output subspaces are.

In practice, MoE routing in trained models shows peaked distributions
— a given expert at layer L will feed preferentially into ~20–40 of the
128 experts at L+1, not all 128 uniformly. This is well-established
empirically in MoE research and follows from the fact that expert
specialisation creates structured output subspaces.

However, **we do not need the transition matrix at all for the initial
pipeline**. The two-phase dispatch already hides DMA latency behind
hot-expert compute. Prefetching based on transition probabilities is a
second-order optimisation — it converts some cold misses to warm hits,
but the pipeline is already stall-free without it. Transition matrix
profiling and speculative prefetch can be added later as a measured
improvement, not a design prerequisite.

**Simplification**: omit transition matrix profiling from v1. The score-based
cache with two-phase DMA overlap handles the common case. Measure
actual cold-hit rates before investing in prediction machinery.

---

### 2. How stable is the transition matrix across domains?

**Decision: moot for v1 (no transition matrix). But if later added:
single calibration over mixed data is sufficient.**

Routing patterns do shift between domains (code vs prose vs
multilingual), but the shift is in the *popularity distribution* of
experts, not in the *structure* of the transition graph. The same
experts tend to be "adjacent" in the routing DAG regardless of domain
— what changes is how often each path is taken, not which paths exist.

A single calibration over representative mixed data would capture the
dominant transitions. Per-domain calibration adds complexity for
marginal gain. The expert cache naturally adapts to domain shifts by
evicting stale experts and loading popular ones — this is a self-
correcting mechanism that doesn't require explicit domain awareness.

**Simplification**: if transition-based prefetching is ever added, use
a single calibration run over diverse data. The cache itself is the
primary domain-adaptation mechanism.

---

### 3. Can layer L's routing predict L+1's?

**Decision: yes in theory, not worth the complexity for v1.**

Given that transition probabilities are peaked (Decision 1), speculative
prefetch is technically feasible: while layer L's grouped GEMM runs,
begin DMA for the experts that L+1's router will likely select. The
prediction would be: take the top-8 experts from layer L, look up
their transition rows in a cached matrix, union the top-N successors,
and start DMA for any that are COLD.

The problem is that this prediction happens *before* L+1's router runs.
Any misprediction wastes DMA bandwidth loading experts that won't be
needed, and may evict experts that were about to be needed. With
6,144 experts and only 44–65% residency, eviction mistakes are costly.

The two-phase dispatch already provides a natural prefetch window:
hot-expert compute at layer L is the DMA window for cold experts at
layer L. Speculative cross-layer prefetch would try to extend this
window to L+1, but the gain is small (the router at L+1 runs ~200μs
later, and DMA per expert is ~120μs — there's barely enough time for
1 speculative DMA before the router resolves the truth).

**Simplification**: no speculative cross-layer prefetch in v1. The
intra-layer two-phase overlap is the primary latency-hiding mechanism.
Cross-layer prefetch can be measured and added later if cold-hit rates
remain high after warm-up.

---

### 4. Does the graph have a hot core?

**Decision: yes, but handle it through score-based eviction, not pinning.**

MoE models consistently show non-uniform expert popularity. Some
experts activate on nearly every input (general-purpose feature
detectors), while others are highly specialised and rarely invoked.
This Zipf-like distribution means a "hot core" exists — perhaps 20–30%
of experts handle 60–70% of activations.

Pinning the hot core (marking certain experts as non-evictable) is
tempting but introduces several problems:
- It requires knowing which experts are hot *a priori*, which depends
  on the workload mix and may shift over time.
- Pinned experts consume fixed VRAM that can never be reclaimed, even
  if the workload shifts.
- The boundary between "hot enough to pin" and "not hot enough" is
  fuzzy and hard to calibrate.

The score-based cache already handles this naturally:
hot experts are promoted on every use, so they never reach the eviction
frontier. They are *effectively* pinned by usage frequency, without
any explicit pinning mechanism. The cache self-organises: the hot core
stays resident because it's constantly refreshed, and the long tail
cycles through the remaining slots.

**Simplification**: no explicit pinning. The eviction-score mechanism
is the hot-core management strategy. If profiling later shows that
certain experts are evicted and immediately reloaded (thrashing), a
pinning heuristic can be added — but this is a reactive fix, not a
proactive design element.

---

### 5. What is the right queue granularity?

**Decision: no queues. Flat GPU-sorted array, sliced by expert group.**

This was already resolved in the "Locking, Queues, and the Zero-
Contention Architecture" section above. The answer:

The GPU's `sort_last_dim` on the `[num_tokens, 128]` routing logits
produces expert indices already sorted by expert ID within each token.
The CPU reads back the `[num_tokens, k]` index tensor, builds a flat
`Vec<(expert_id, token_idx, weight_idx)>`, sorts by `expert_id`, and
iterates contiguous groups. Each group is a "queue" — a slice of the
sorted array. No persistent per-expert data structure needed.

This is visible in the current code: `SparseMoeBlock::forward()` builds
a `HashMap<usize, (Vec<u32>, Vec<u32>)>` mapping expert → (token list,
weight index list). In the pipeline thread, this becomes a flat sorted
array with group-by, eliminating the HashMap allocation.

**Simplification**: one flat `Vec`, one sort (or CPU-side radix sort
on 7-bit expert IDs 0..128 within a single layer), one linear scan.
No per-expert queues, no 6,144-element sparse structure.

Note: each layer has 128 experts, not 6,144. The pipeline processes
one layer at a time, so the routing assignment array per layer has at
most `num_tokens × 8` entries, and expert IDs range 0..127. A 7-bit
radix sort is trivially O(n).

---

### 6. At what VRAM threshold does the pipeline degenerate to trivial?

**Decision: never on consumer hardware for this model. But detect and
optimise the case anyway.**

With 6,144 experts at ~3 MB each (18 GB total), no single consumer
GPU can cache them all:

| GPU VRAM | 50% budget | Slots  | Residency |
|----------|-----------|--------|-----------|
| 24 GB    | 12 GB     | ~4,000 | 65%       |
| 16 GB    | 8 GB      | ~2,700 | 44%       |
| 8 GB     | 4 GB      | ~1,300 | 21%       |

Full residency requires ~36 GB of VRAM budget (72 GB total), which
is multi-GPU territory. For any single consumer card, the full DMA
overlap pipeline is always the primary execution path.

The code already handles the degenerate case: `num_slots =
(expert_budget / max_expert_size).min(total_experts)` caps at the
total expert count, so if a hypothetical GPU has enough VRAM, the
cache holds everything and the cache never evicts. The state machine
collapses to WARM → READY → RUNNING → WARM with zero DMA overhead.
The two-phase dispatch still runs but phase 2 is always empty (no
LOADING experts), so it's a no-op.

**Simplification**: no special "trivial mode" code path. The existing
pipeline gracefully degrades to zero overhead when cache ≥ experts.
The `load_batch()` call returns an empty set, the fence is a no-op,
phase 2 has no work. The cost is one empty-vec check per layer — 
negligible.

---

## Eviction Policy (Implemented)

The original design decisions (1–4 above) deferred transition matrix
prefetching and explicit pinning to later iterations.  During
implementation, empirical testing revealed that naive timestamp-based eviction
combined with speculative prefetch caused severe eviction cascades —
a single mispredicted prefetch could evict an expert needed by a later
layer in the same pass, triggering 3–5 downstream misses and halving
single-token decode throughput.

The solution is a four-part eviction policy that separates concerns:
batch eviction creates headroom, forced eviction respects layer
ordering, pinning protects critical layers, and prefetch can never
evict.  Each piece is simple; together they eliminate cascades.

### The eviction cascade problem

MoE layers execute sequentially: layer 0, then 1, then 2, ..., then
47.  Once layer L completes, its experts are not needed again until
the *next* forward pass.  But experts for layers L+1, L+2, ... are
still needed in *this* pass.

With naive timestamp-based eviction, a cache miss at layer L could evict an expert
from layer L+5 — an expert about to be needed in 5 layers.  That
creates a new miss at layer L+5, which evicts from layer L+10, and so
on.  One miss cascades into many.

Speculative prefetch made this catastrophically worse: prefetching 8
experts per layer (the v1 approach) could inject up to 384 evictions
per pass (8 × 48 layers), each capable of triggering a cascade.

### Part 1: End-of-pass batch eviction (proactive headroom)

After the last MoE layer (layer 47) completes, evict the bottom 5%
of occupied slots by usage timestamp.  Respects pinning — never evicts
experts from pinned layers.

```
After layer 47:
  candidates = all occupied slots where layer >= PINNED_LAYERS
  sort by last_used ascending (stalest first)
  evict bottom 5% → push to free_slots
```

On a 2,805-slot budget, this creates ~140 free slots.  The next pass's
early layers find free slots instead of triggering forced evictions.
This is the primary mechanism for preventing eviction stalls at
layers 3–10, where there are few completed layers to evict from.

### Part 2: Layer-aware forced eviction (behind-layer bias)

When a real cache miss occurs and no free slots are available,
eviction follows a two-tier priority:

```
Tier 1 — behind-layer candidates:
  layer >= PINNED_LAYERS AND layer < current_layer
  (already executed this pass, not needed again until next pass)
  Pick: highest layer first (furthest from next-pass reuse)
  Tie-break: lowest last_used timestamp

Tier 2 — global score-based fallback:
  any occupied slot where layer >= PINNED_LAYERS
  Pick: lowest last_used timestamp globally
  (only reached when no behind-layer candidates exist,
   e.g. at layer 3 before any layers are "behind")
```

This ensures that forced eviction never displaces an expert from a
layer that has not yet executed in this pass.  Cascades are
structurally impossible because every evicted expert is from a
completed layer.

### Part 3: Early-layer pinning

Experts in the first `PINNED_LAYERS` (currently 3) MoE layers are
never evicted by any mechanism — not by forced eviction, not by
batch eviction.

```
Layers 0, 1, 2:  pinned — never evicted
Layers 3–47:     normal eviction rules apply
```

Rationale: these layers run first every pass.  There is no prior
compute to overlap with DMA, so a cache miss at layer 0–2 causes a
full stall with zero latency hiding.  Pinning costs ~24 slots
(top-8 × 3 layers), less than 1% of the slot budget.

This is a targeted exception to the "no explicit pinning" decision
(Decision 4 above).  The cache naturally keeps popular experts hot,
but layers 0–2 are not "popular" in the eviction-score sense — they execute
once per pass like every other layer.  Their criticality comes from
their position (first in the pipeline), not their frequency.

### Part 4: Free-slot-only speculative prefetch

Speculative prefetch **never evicts**.  It only loads into free slots:

```
slot = free_slots.pop()
if no free slot:
    skip prefetch entirely (noop)
else:
    load predicted expert into slot
```

This makes prefetch structurally incapable of causing eviction
cascades.  A mispredicted prefetch simply occupies a free slot that
would otherwise sit empty.  The worst case is that a future real miss
must evict via forced eviction instead of finding a free slot — but
that eviction follows the layer-aware policy (Part 2) and is safe.

Free headroom for prefetch is created by end-of-pass batch eviction
(Part 1).  As the pass progresses and free slots are consumed by
real misses and prefetches, the free pool depletes.  Once empty,
prefetch auto-disables — no special cutoff logic needed.

### How the four parts interact

```
End of pass N:
  batch-evict 5% → ~140 free slots
         ↓
Pass N+1, layers 0–2:
  pinned, always hit (zero DMA stall)
         ↓
Pass N+1, layers 3–10:
  misses consume free slots (no eviction scan)
  prefetch fills remaining free slots
         ↓
Mid-pass (layers ~15+):
  free slots depleted → prefetch auto-disables
  forced evictions use behind-layer bias
  (evict layer 3 experts to load layer 15 expert)
         ↓
Late-pass (layers 40–47):
  large behind-layer pool → cheap eviction
         ↓
End of pass N+1:
  batch-evict 5% → cycle repeats
```

The policy creates a natural back-pressure mechanism: free slots are
a finite resource that drains during the pass.  Early layers get free
slots (cheap), middle layers get behind-layer eviction (safe), and
prefetch gracefully degrades as resources deplete.

### Measured performance impact

Benchmark: Qwen3-30B-A3B, Q4_K_M quantisation, RTX 4080 (16 GB),
2,805 cache slots (44% residency), 20 decode tokens per context.

```
                   Before eviction    After eviction
Config             policy (v2)        policy (v3)       Δ
────────────────── ────────────────── ────────────────── ──────
F16  × 1 context        57.3 t/s          61.8 t/s     +7.9%
BF16 × 1 context       199.9 t/s         241.5 t/s    +20.8%
BF16 × 4 contexts      685.3 t/s        1090.1 t/s    +59.1%
Q8_0 × 8 contexts     1220.2 t/s        1699.9 t/s    +39.3%
```

The gains are largest for batched configs because more contexts means
more unique experts per pass, which means more cache pressure.  The
old policy thrashed under pressure; the new policy absorbs it because
the free-slot pool from batch eviction acts as a shock absorber.

---

## Transition Matrix and Speculative Prefetch (Implemented)

Design decisions 1–3 originally deferred transition matrix prefetching.
It was subsequently implemented with an **online learning** approach
that requires no calibration pass.

### Online learning (no calibration required)

The transition matrix is built incrementally during inference:

```
For each pair of adjacent MoE layers (L, L+1):
  T[L→L+1] : [128 × 128] float32 co-occurrence counts

During each forward pass:
  observe(layer_idx, expert_ids):
    if previous layer was layer_idx - 1:
      for each (from, to) in prev_experts × current_experts:
        T[pair][from][to] += 1.0
    store current experts as "previous"

  reset_pass():
    clear previous-layer state (prevents cross-pass transitions)
```

After ~64 total observations (configurable via `min_observations`),
the matrix has enough data to make useful predictions.

### Prediction: single best non-cached expert

Given the active expert set at layer L, prediction works as follows:

```
scores = [0.0; 128]    // accumulator for each expert at L+1
for from in active_experts_at_L:
  for to in 0..128:
    scores[to] += T[L→L+1][from][to]

best = argmax(scores, excluding experts already in active set)
return best   // single expert, or empty if not enough observations
```

Only a single expert is predicted per layer to minimise DMA pressure.
The active set is excluded because those experts are likely cache hits
at the next layer anyway.

### Integration with the eviction policy

Speculative prefetch runs **after** the current layer's compute
completes (not before — running CPU-side mmap/parsing work before
compute would starve the GPU).  It loads at most one expert into a
free slot (Part 4 of the eviction policy — never evicts).

```
process_request(layer L):
  1. classify_and_load(L)   → hits + loaded experts
  2. compute hits           → GPU busy
  3. fence wait             → DMA completes
  4. compute loaded         → GPU busy
  5. speculative_prefetch(L)  → predict L+1's expert, DMA into free slot
  6. prefetch fence wait    → near-instant if DMA finished during step 4
  7. end_of_pass_eviction() → only if L == last layer
```

Correct predictions at step 5 mean the predicted expert is already in
cache when layer L+1 runs — its classify step finds a hit instead of
a miss.  This converts one DMA stall into a free cache hit.

### Why mispredictions are harmless

- Prefetch only uses free slots — no expert is evicted.
- A mispredicted expert sits in cache as a WARM slot.
- It will eventually be evicted by normal score-based or batch eviction.
- Cost: one wasted DMA (~120μs) that ran concurrently with compute.
- Benefit when correct: one avoided DMA stall at the next layer.
