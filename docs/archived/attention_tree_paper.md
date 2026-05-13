# Attention-Organized Conversation Trees: Unbounded Dialogue History Through Relevance-Driven Paging and Regenerative Warming

**Abstract**

We present an architecture for maintaining unbounded conversation history in LLM inference without compression loss. Where prior work addresses context limits through summarization, retrieval-augmented generation, or sliding windows—all of which discard or lossy-compress historical content—our approach preserves complete conversation turns and reconstructs full attention relationships on demand.

**Core insight**: VRAM capacity should constrain *concurrent relational reasoning*, not *total accessible knowledge*. We reframe the context window not as a memory limit but as a cognitive load limit—the number of inter-turn relationships that can be actively "thought about" simultaneously, while the complete history remains accessible.

**Novel contributions**: We introduce *attention-organized B-trees*: a self-balancing tree structure where depth encodes relevance (hot content shallow, cold content deep) while in-order traversal preserves chronology. Rotations driven by attention statistics automatically surface relevant history and sink stale content, with bounded rotation speed ensuring predictable latency. We propose *regenerative warming*: rather than storing pre-computed KV caches for all turns (prohibitive at scale), we store only text and coarse-grained attention relationship metadata. When cold turns rise toward the active zone, a warming subsystem regenerates their KV representations with appropriate relational context.

The architecture supports *pre-populated content*—conversation history generated offline that initializes in cold storage and warms on demand, enabling systems that behave as if they have extensive prior experience. *Periodic reflection* generates consolidation turns that create hub nodes in the attention graph, improving retrieval efficiency and maintaining coherence over very long conversations. A *cycle-based temporal model* handles discontinuities gracefully, using reflection cycles rather than clock time as the fundamental unit.

**The key distinction from retrieval**: RAG and similar approaches retrieve *content* based on embedding similarity. Our architecture retrieves *turns with their attention relationships intact*, using the attention graph itself as the retrieval index. A turn surfaced from deep history attends to (and is attended by) the same ancestors it originally related to, because warming reconstructs those relationships explicitly.

We demonstrate that 50,000+ conversation turns can be maintained with ~80MB storage while preserving the ability to surface any historical turn with full relational context. Active VRAM usage remains bounded at ~250MB regardless of history depth. The cognitive load framing suggests a new way to think about context limits: not "how much can the model remember?" but "how many relationships can the model reason about simultaneously?"

---

## 1. Introduction

The context window has long been understood as a memory constraint—a hard limit on how much text a language model can "see" during generation. Recent work has pushed these limits dramatically, with models now supporting 128K, 200K, or even 1M+ token windows. Yet even these expanded windows are fundamentally bounded: eventually, older content must be truncated, summarized, or evicted.

This paper proposes a different framing. What if the context window represents not a memory limit but a *cognitive load* limit? Human cognition offers an instructive parallel: we do not forget our entire life history when thinking about a problem. Rather, we can actively hold only a limited number of relationships in working memory at any moment—while our full episodic memory remains accessible, surfacing when relevant cues trigger recall.

We operationalize this insight through two mechanisms:

**Attention-organized B-trees** structure conversation history such that *relevance* determines position. Recent and frequently-referenced turns occupy shallow depths (the "hot" zone, resident in VRAM). Older, less-referenced turns sink to deeper levels, eventually paging out to disk. Critically, depth is dynamic: a turn from 10,000 exchanges ago can rotate back to depth 1 if current conversation makes it relevant again. The tree self-organizes around what matters *now*, not what happened *recently*.

**Regenerative warming** solves the storage problem this creates. Storing full KV cache for 50,000 turns would require hundreds of gigabytes. Instead, we store only the turn's text (~1.4KB) plus metadata about its attention relationships (~200 bytes). When a cold turn rises toward the active zone, we regenerate its KV cache by replaying the turn with its ancestor turns in context—reconstructing the attention relationships that give the turn its relational meaning.

```
┌─────────────────────────────────────────────────────────────┐
│              The Cognitive Load Reframing                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRADITIONAL VIEW: Context window as memory                 │
│  ─────────────────────────────────────────                  │
│  "The model can only remember 128K tokens"                  │
│  "Older content must be summarized or forgotten"            │
│  "Context length is a hard constraint"                      │
│                                                             │
│  PROPOSED VIEW: Context window as cognitive load            │
│  ──────────────────────────────────────────────             │
│  "The model can reason about N relationships at once"       │
│  "All content remains accessible, surfaced by relevance"    │
│  "Active relationships are bounded; knowledge is not"       │
│                                                             │
│  Analogy to human cognition:                                │
│  Working memory ≈ 7±2 items (bounded)                       │
│  Episodic memory ≈ decades of experience (unbounded)        │
│  Recall ≈ relevance-triggered surfacing                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 Relationship to Prior Work

This paper extends the paged KV cache architecture presented in [Paper 1], which introduced position-independent KV storage with RoPE remapping and trie-based fact retrieval. That work demonstrated that precomputed KV chunks could be injected at arbitrary positions, enabling sparse retrieval over large fact indices.

We build on three specific foundations from that work:

1. **Position-independent storage**: KV caches stored with internal positions can be remapped to any target position at runtime through RoPE delta application. This enables our rotation mechanism—turns can move freely in the tree without invalidating their KV representations.

2. **Self-contained facts**: The observation that content computed in isolation (without surrounding context) can still be usefully attended to at generation time. Our "cold" turns are exactly such self-contained facts, awaiting warming to regain relational context.

3. **Attention as retrieval signal**: The use of attention statistics to identify relevant content. We extend this from per-query retrieval to persistent metadata that structures the entire conversation history.

### 1.2 Contributions

This paper makes the following contributions:

1. **Attention-organized B-trees** (Section 3): A self-balancing tree structure for conversation history where rotations driven by attention statistics maintain the invariant that depth approximates inverse relevance, while in-order traversal preserves chronology.

2. **Three-tier caching architecture** (Section 4): HOT (VRAM-resident, full KV), WARM (staging, being regenerated), and COLD (disk-resident, text + metadata only) tiers with automatic promotion and demotion based on tree position. Includes support for pre-populated content and shared content across concurrent sessions.

3. **Regenerative warming** (Section 5): A mechanism for reconstructing full attention relationships when cold turns are promoted, using stored ancestor metadata to replay turns with appropriate relational context.

4. **Offline maintenance and reflection** (Section 7): Asynchronous tree operations that consolidate history through periodic reflection turns, creating hub nodes that organize the attention hierarchy and improve retrieval efficiency.

5. **Cycle-based temporal model** (Section 8): A discontinuity-tolerant approach to time where reflection cycles (rather than clock time) serve as the fundamental temporal unit, enabling graceful handling of gaps in interaction.

6. **Cognitive load framing** (Section 9): A conceptual reframing of context limits as concurrent relationship bounds rather than memory bounds, with implications for how we design and evaluate long-context systems.

---

## 2. Background

### 2.1 The Attention Relationship Problem

When a transformer processes a conversation, each turn's representation is shaped by attention to previous turns. Consider:

```
T1: "My daughter Maya was born yesterday"
T2: "Congratulations! How is she doing?"
T3: "Maya is healthy, we're so happy"
```

When T3 is computed with T1 and T2 in context, its KV representation *encodes* the relationship: "Maya" in T3 attends to "daughter Maya" in T1, strengthening the association. The K and V vectors for T3 are literally different than they would be if T3 were computed in isolation.

```
┌─────────────────────────────────────────────────────────────┐
│  KV Representation Depends on Context                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  T3 computed alone:                                         │
│    K3, V3 encode only "Maya is healthy, we're so happy"     │
│    "Maya" is just a name token                              │
│                                                             │
│  T3 computed with T1, T2 in context:                        │
│    K3', V3' encode T3 content PLUS relationships            │
│    "Maya" carries compressed signal: daughter, newborn      │
│    This happens through attention aggregation at each layer │
│                                                             │
│  K3' ≠ K3                                                   │
│  The context is baked into the representation               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This creates a fundamental challenge for conversation history management. If we simply store each turn's KV computed in isolation (as a "fact"), we lose the relational encoding. If we store KV computed with full history, storage becomes prohibitive.

### 2.2 What Attention Statistics Capture

During the forward pass, attention weights reveal which previous content each turn relied on:

```
Computing T3 with T1, T2 in context:

Layer L, Head H:
  Q3 @ K[1,2,3].T → attention weights
  
  Aggregated across layers/heads:
    T3 attended to T1: 0.35 (Maya reference)
    T3 attended to T2: 0.25 (conversational flow)
    T3 self-attention: 0.40
```

These statistics are computed anyway during the forward pass. Our insight is to *store them as metadata*, creating a graph of attention relationships:

```
T3.ancestors = [(T1, 0.35), (T2, 0.25)]
```

This metadata serves dual purposes:
1. **Retrieval index**: Given a query, traverse ancestor chains to find relevant history
2. **Warming guide**: When regenerating T3's KV, we know to include T1 and T2 in context

### 2.3 Why Not Just Store Everything?

At scale, full KV storage is impractical:

```
Per turn (350 tokens average):
  FP8 KV: 350 × 48 KB = 16.8 MB
  INT4 KV: 350 × 24 KB = 8.4 MB

50,000 turns:
  FP8: 840 GB
  INT4: 420 GB

Text + metadata only:
  Text: 350 × 4 bytes = 1.4 KB
  Metadata: ~200 bytes
  Total: 1.6 KB per turn
  50,000 turns: 80 MB
```

The difference is 5,000×. This gap motivates regenerative warming: store cheap, regenerate expensive on demand.

---

## 3. Attention-Organized B-Trees

### 3.1 Structure and Invariants

We organize conversation history as a B-tree with two simultaneous orderings:

**Invariant 1 (Chronology)**: In-order traversal yields turns in temporal order. T1 < T2 < T3 < ... < Tn.

**Invariant 2 (Relevance)**: Depth approximates inverse relevance. Frequently-attended turns are shallow; rarely-attended turns are deep.

```
┌─────────────────────────────────────────────────────────────┐
│  Dual Ordering in Attention-Organized B-Tree                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                       [T500] depth 0                        │
│                      /       \                              │
│                 [T347]       [T498] depth 1                 │
│                /     \            \                         │
│           [T102]   [T400]        [T499] depth 2             │
│           /    \       \                                    │
│       [T45]  [T200]  [T450] depth 3                         │
│                                                             │
│  In-order: T45, T102, T200, T347, T400, T450, T498, T499, T500
│  (chronological ✓)                                          │
│                                                             │
│  Depth: T500 (0), T347/T498 (1), T102/T400/T499 (2), ...   │
│  (relevance-ordered: recent query attended heavily to       │
│   T500, T347, T498, less to deeper nodes)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Rotation Mechanics

Rotations adjust depth without breaking chronological order. The key constraint is that B-tree rotations preserve in-order traversal. We exploit this property: any valid B-tree rotation maintains chronology automatically.

**The In-Order Preservation Property**

In a B-tree, in-order traversal visits nodes in key order. Since we assign keys based on turn creation time (T1 < T2 < T3 < ...), in-order traversal always yields chronological order. B-tree rotations—which are structure-preserving transformations—cannot violate this property.

```
┌─────────────────────────────────────────────────────────────┐
│  Rotation Example: Promoting T200                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEFORE rotation (T200 too deep):                           │
│                                                             │
│          [T347]                                             │
│         /      \                                            │
│     [T102]    [T400]                                        │
│         \      /    \                                       │
│       [T200] [T350] [T450]                                  │
│                                                             │
│  In-order: T102, T200, T347, T350, T400, T450  ✓            │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  AFTER rotation (T200 promoted):                            │
│                                                             │
│          [T347]                                             │
│         /      \                                            │
│     [T200]    [T400]                                        │
│     /    \         \                                        │
│  [T102] [T250]    [T450]                                    │
│                    /                                        │
│                [T350]                                       │
│                                                             │
│  In-order: T102, T200, T250, T347, T350, T400, T450  ✓      │
│                                                             │
│  T200 moved from depth 3 → depth 2                          │
│  Chronological order preserved                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Rotation Direction and Depth Change**

A rotation can move a node either toward the root (promotion) or away from the root (demotion). The direction is determined by attention statistics:

```
┌─────────────────────────────────────────────────────────────┐
│  Rotation Direction Logic                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROMOTE (rotate toward root):                              │
│    Trigger: Node has high attention but sits deep           │
│    Mechanism: Node swaps toward parent's position           │
│    Effect: Depth decreases, node becomes more accessible    │
│                                                             │
│  DEMOTE (rotate away from root):                            │
│    Trigger: Node has low attention but sits shallow         │
│    Mechanism: Node swaps toward child positions             │
│    Effect: Depth increases, node moves toward eviction      │
│                                                             │
│  CONSTRAINT: Maximum 2 levels of depth change per cycle     │
│    - Prevents large jumps that outpace warming              │
│    - Ensures WARM zone has time to prepare promotions       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Handling Branching Factor**

Unlike traditional B-trees optimized for disk block sizes, our tree optimizes for attention patterns. We allow variable branching factor with a soft maximum:

```
┌─────────────────────────────────────────────────────────────┐
│  Branching Factor Considerations                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Minimum branching factor: 2 (binary tree minimum)          │
│  Soft maximum: 4-6 children per node                        │
│                                                             │
│  Why variable branching:                                    │
│    - Clusters of related turns naturally group              │
│    - High-attention turns may have many relevant children   │
│    - Low-attention regions can collapse into chains         │
│                                                             │
│  Split policy: When a node exceeds max children,            │
│                promote the median-attention child           │
│                                                             │
│  Merge policy: When siblings have very low combined         │
│                attention, merge and demote                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Rotation Driven by Attention Statistics

After each prefill, we obtain attention statistics for the current query against all active turns. These statistics represent aggregated attention weights across all layers and heads, normalized per turn:

```
┌─────────────────────────────────────────────────────────────┐
│  Attention Statistics After Prefill                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Turn     Attention Weight    Depth    Status               │
│  ─────────────────────────────────────────────────────────  │
│  T500     0.42                0        aligned (current)    │
│  T347     0.28                1        aligned              │
│  T498     0.15                1        aligned              │
│  T102     0.08                2        aligned              │
│  T400     0.04                2        should sink          │
│  T450     0.03                3        aligned              │
│  T203     0.22                4        should rise ←        │
│                                                             │
│  Mismatch: T203 has high attention (0.22) but deep (4)      │
│            T400 has low attention (0.04) but shallow (2)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Mismatch Detection**

We compare two orderings: the attention rank (higher attention = more important) and the depth rank (shallower = more important). A mismatch occurs when these orderings disagree beyond a threshold:

```
┌─────────────────────────────────────────────────────────────┐
│  Mismatch Detection Logic                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each node in the HOT zone:                             │
│                                                             │
│    attention_rank = position when sorted by attention       │
│    depth_rank = position when sorted by tree depth          │
│                                                             │
│    if attention_rank << depth_rank:                         │
│        → Node has high attention but sits deep              │
│        → Should RISE (rotate toward root)                   │
│                                                             │
│    if attention_rank >> depth_rank:                         │
│        → Node has low attention but sits shallow            │
│        → Should SINK (rotate away from root)                │
│                                                             │
│    Threshold prevents thrashing on minor differences        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Bounded Rotation Speed

To ensure warming can keep pace with promotion, we bound how quickly any node can change depth:

```
Maximum depth change per turn cycle: 2 levels

Turn rising from depth 8 → depth 2:
  Cycle 1: depth 8 → 6
  Cycle 2: depth 6 → 4
  Cycle 3: depth 4 → 2
  
Warming has 3 cycles to complete regeneration.
```

This creates a natural pipeline where the WARM zone (depths 4-6) serves as a staging area for content being promoted.

### 3.5 Tree Balance and Depth Bounds

Unlike traditional B-trees that balance to minimize worst-case search depth, our tree balances to align depth with relevance. However, we still maintain structural bounds to ensure predictable behavior:

```
┌─────────────────────────────────────────────────────────────┐
│  Balance Properties                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DEPTH BOUND                                                │
│  ─────────────────────────────────────────────────────────  │
│  For n turns, maximum depth is O(log n)                     │
│                                                             │
│  With 50,000 turns and branching factor 3-4:                │
│    Theoretical max depth: log₃(50000) ≈ 10                  │
│    Practical max depth: ~15 (with attention skew)           │
│                                                             │
│  This bounds:                                               │
│    - Worst-case rotation cascade length                     │
│    - Maximum warming recursion depth                        │
│    - In-order traversal cost for chronological access       │
│                                                             │
│  ATTENTION SKEW                                             │
│  ─────────────────────────────────────────────────────────  │
│  Real conversations have attention skew:                    │
│    - A few "anchor" turns get repeated attention            │
│    - Most turns are briefly relevant then fade              │
│                                                             │
│  This creates unbalanced structure:                         │
│    - Hot anchors form a shallow spine                       │
│    - Cold turns hang in deep chains                         │
│                                                             │
│  The imbalance is INTENTIONAL—it reflects relevance.        │
│  We bound depth, not balance ratio.                         │
│                                                             │
│  COMPACTION                                                 │
│  ─────────────────────────────────────────────────────────  │
│  When subtrees exceed depth bounds:                         │
│    - Merge siblings with uniformly low attention            │
│    - Coalesce long chains into wider shallow nodes          │
│                                                             │
│  This prevents pathological depth while preserving          │
│  the relevance-based ordering.                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Relationship to Red-Black Trees**

Red-black trees maintain balance through color constraints that bound the longest path to 2× the shortest. Our tree maintains a different invariant:

```
┌─────────────────────────────────────────────────────────────┐
│  Balance Comparison                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Red-Black Tree:                                            │
│    Invariant: longest_path ≤ 2 × shortest_path              │
│    Purpose: Bound search time                               │
│    Rotation trigger: Color violations                       │
│                                                             │
│  Attention-Organized Tree:                                  │
│    Invariant: depth_rank ≈ inverse_attention_rank           │
│    Purpose: Align structure with relevance                  │
│    Rotation trigger: Attention/depth mismatch               │
│                                                             │
│  Secondary bound: max_depth ≤ 15                            │
│    Enforced by compaction when exceeded                     │
│                                                             │
│  Both maintain O(log n) depth, but for different reasons    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.6 Convergence Properties

The rotation loop iterates until the tree stabilizes:

```
┌─────────────────────────────────────────────────────────────┐
│  Convergence Loop                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  REPEAT until converged OR max_iterations reached:          │
│                                                             │
│    1. DETECT mismatches                                     │
│       Compare attention rank to depth rank for all HOT nodes│
│       Identify nodes that should rise or sink               │
│                                                             │
│    2. ROTATE mismatched nodes                               │
│       Rising nodes: rotate toward root (max 2 levels)       │
│       Sinking nodes: rotate away from root (max 2 levels)   │
│       Preserve in-order invariant during rotation           │
│                                                             │
│    3. RE-PROBE attention                                    │
│       Compute Q_user @ K_all.T (cheap matrix multiply)      │
│       Updated positions may change attention patterns       │
│                                                             │
│    4. CHECK convergence                                     │
│       Converged when attention rank ≈ depth rank            │
│       (within threshold for all HOT nodes)                  │
│                                                             │
│  Typical iterations to convergence: 3-5                     │
│  Cost per iteration: ~0.01 ms (dominated by re-probe)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Convergence is guaranteed by the bounded rotation speed and finite tree size. Pathological cases (uniform attention across all nodes) reach max_iterations and produce a stable if imperfect ordering.

---

## 4. Three-Tier Caching Architecture

### 4.1 Tier Definitions

The tree depth naturally partitions turns into three tiers:

```
┌─────────────────────────────────────────────────────────────┐
│  Three-Tier Cache Architecture                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HOT (depth 0-3)                                            │
│  ├── Location: VRAM                                         │
│  ├── Contents: Full KV cache, attention relationships baked │
│  ├── Capacity: ~30 turns, ~250 MB                           │
│  └── Access: Immediate, used for decode                     │
│                                                             │
│  WARM (depth 4-6)                                           │
│  ├── Location: VRAM (staging) or being computed             │
│  ├── Contents: KV being regenerated with context            │
│  ├── Capacity: ~20 turns in pipeline                        │
│  └── Access: Stall if not ready when crossing to HOT        │
│                                                             │
│  COLD (depth 7+)                                            │
│  ├── Location: Disk (SSD)                                   │
│  ├── Contents: Text + ancestor metadata only                │
│  ├── Capacity: Unbounded (50K turns = 80 MB)                │
│  └── Access: Trigger warming when promoted to WARM          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Automatic Tier Transitions

Tier membership is determined solely by tree depth. No explicit tier tracking is needed—the tree structure *is* the tier assignment:

```
┌─────────────────────────────────────────────────────────────┐
│  Depth-to-Tier Mapping                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Depth 0-3  →  HOT   (VRAM, immediate access)               │
│  Depth 4-6  →  WARM  (staging, being prepared)              │
│  Depth 7+   →  COLD  (disk, text + metadata only)           │
│                                                             │
│  When a rotation changes a node's depth, tier transitions   │
│  are triggered automatically based on boundary crossings.   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Tier Transition Handlers

Each tier boundary crossing triggers specific actions:

```
┌─────────────────────────────────────────────────────────────┐
│  COLD → WARM (Promotion Begins)                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Trigger: Node rotates from depth 7+ to depth 6             │
│                                                             │
│  Actions:                                                   │
│    1. Load turn text from disk                              │
│    2. Load ancestor metadata from disk                      │
│    3. Queue warming task with priority = attention score    │
│    4. Warming subsystem begins regenerating KV              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WARM → HOT (Promotion Completes)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Trigger: Node rotates from depth 4 to depth 3              │
│                                                             │
│  Actions:                                                   │
│    1. Check: Is warming complete?                           │
│       - If NO: STALL until warming finishes                 │
│       - If YES: Continue                                    │
│    2. Move warmed KV into HOT cache                         │
│    3. Node now participates in decode attention             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  HOT → WARM (Demotion Begins)                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Trigger: Node rotates from depth 3 to depth 4              │
│                                                             │
│  Actions:                                                   │
│    1. Mark node as "cooling" (candidate for eviction)       │
│    2. KV remains in VRAM during WARM phase                  │
│    3. If node rises again, cooling is cancelled             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WARM → COLD (Demotion Completes)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Trigger: Node rotates from depth 6 to depth 7              │
│                                                             │
│  Actions:                                                   │
│    1. Capture current attention statistics as ancestors     │
│    2. Write ancestor metadata to disk                       │
│    3. Evict KV from VRAM                                    │
│    4. Only text + metadata remains (already on disk)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Memory Budget Analysis

```
┌─────────────────────────────────────────────────────────────┐
│  Memory Budget (50,000 turn conversation)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  VRAM (RTX 4090, 24GB):                                     │
│    Model (Qwen3-30B-A3B AWQ): 17 GB                         │
│    HOT cache (30 turns):      252 MB                        │
│    WARM staging (20 turns):   168 MB                        │
│    Working memory:            ~500 MB                       │
│    ───────────────────────────────────────                  │
│    Total:                     ~18 GB                        │
│    Headroom:                  ~6 GB                         │
│                                                             │
│  Disk (SSD):                                                │
│    Conversation text (50K turns):       70 MB               │
│    Per-turn metadata (50K × 115 bytes): 5.5 MB              │
│    ───────────────────────────────────────                  │
│    Required total:                      75.5 MB             │
│                                                             │
│  Optional (for enhanced discovery):                         │
│    Representative K vectors:            100 MB              │
│    ───────────────────────────────────────                  │
│    Enhanced total:                      175.5 MB            │
│                                                             │
│  Regardless of conversation length, VRAM usage is O(1)      │
│  Disk usage is O(n) but with very small constants           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Comparison: Proposed vs. Naive Full KV Storage             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        Naive           Proposed             │
│  ─────────────────────────────────────────────────────────  │
│  VRAM (50K turns)      420 GB          ~420 MB              │
│  Disk                  0               ~80-180 MB           │
│                                                             │
│  Reduction factor:     1000× VRAM reduction                 │
│                                                             │
│  Trade-off:            None            Warming latency      │
│                                        (15-25 ms/turn)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 Per-Turn Metadata Schema

Each turn's metadata enables reconstruction of attention relationships without storing full KV:

```
┌─────────────────────────────────────────────────────────────┐
│  Turn Metadata Structure                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IDENTITY                                                   │
│  ─────────────────────────────────────────────────────────  │
│  turn_id:        Unique identifier (sequential integer)     │
│  created_at:     Timestamp of turn creation                 │
│  token_count:    Number of tokens in turn                   │
│                                                             │
│  CONTENT REFERENCE                                          │
│  ─────────────────────────────────────────────────────────  │
│  text_offset:    Byte offset in conversation text file      │
│  text_length:    Byte length of turn text                   │
│                                                             │
│  ATTENTION RELATIONSHIPS                                    │
│  ─────────────────────────────────────────────────────────  │
│  ancestors:      List of (turn_id, attention_score) pairs   │
│                  Sorted by attention_score descending       │
│                  Typically top 3-5 most-attended turns      │
│                                                             │
│  TEMPORAL ENCODING                                          │
│  ─────────────────────────────────────────────────────────  │
│  time_marker:    Human-readable relative time string        │
│                  e.g., "2 years ago", "yesterday"           │
│  granularity:    Precision level of time_marker             │
│                  (years, months, weeks, days, hours)        │
│                                                             │
│  TREE POSITION (runtime only, not persisted)                │
│  ─────────────────────────────────────────────────────────  │
│  depth:          Current depth in tree                      │
│  parent_id:      Current parent turn_id                     │
│  child_ids:      Current children turn_ids                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Storage Cost Analysis                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Fixed fields:                                              │
│    turn_id (8 bytes) + created_at (8 bytes) +               │
│    token_count (4 bytes) + text_offset (8 bytes) +          │
│    text_length (4 bytes) + granularity (1 byte)             │
│    = 33 bytes                                               │
│                                                             │
│  Variable fields:                                           │
│    ancestors: 5 × (8 + 4) bytes = 60 bytes typical          │
│    time_marker: ~20 bytes average                           │
│                                                             │
│  Total per turn: ~115 bytes                                 │
│                                                             │
│  50,000 turns: ~5.5 MB metadata                             │
│  (Plus ~70 MB text = ~75 MB total disk)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.6 Temporal Markers and Attention Discovery

A critical challenge arises when relevant content has no ancestor chain leading to it from the current HOT zone. Temporal markers embedded in the conversation provide an alternative discovery mechanism.

**The Discovery Problem**

```
┌─────────────────────────────────────────────────────────────┐
│  Ancestor Chain Limitation                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scenario:                                                  │
│    T-47: "We discussed the project requirements"            │
│    T-3021: "The requirements changed last quarter"          │
│                                                             │
│    T-47 and T-3021 never co-occurred in the active window   │
│    Neither is in the other's ancestor chain                 │
│    Both are now COLD                                        │
│                                                             │
│  User query: "What were those requirements again?"          │
│                                                             │
│  Problem: No ancestor chain leads to T-47 or T-3021         │
│           Attention probe only sees HOT nodes               │
│           Relevant history is invisible                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Temporal Markers as Attention Hooks**

We embed relative temporal markers in the conversation text itself, allowing the attention mechanism to discover time-relevant content:

```
┌─────────────────────────────────────────────────────────────┐
│  Turn Structure with Temporal Markers                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stored turn format:                                        │
│                                                             │
│    [T-500 | 18 months ago]: We set the initial budget...    │
│    [T-200 | 6 months ago]: The scope expanded...            │
│    [T-0 | now]: What was our original timeline?             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  How discovery works:                                       │
│                                                             │
│  1. "original" and "timeline" in query attend to tokens     │
│     containing those concepts in historical turns           │
│                                                             │
│  2. "original" attends to earlier temporal markers          │
│     Time-marker tokens encode temporal proximity            │
│                                                             │
│  3. Combined signal surfaces T-500 over T-200               │
│     (despite both discussing the project)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Dynamic Temporal Granularity**

Time markers update as conversation progresses. A turn marked "yesterday" eventually becomes "last week", then "3 months ago", then "2 years ago". Granularity coarsens with age:

```
┌─────────────────────────────────────────────────────────────┐
│  Temporal Granularity Levels                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Age of Turn          Granularity      Example Marker       │
│  ─────────────────────────────────────────────────────────  │
│  < 24 hours           hours            "3 hours ago"        │
│  1-7 days             days             "yesterday"          │
│                                        "4 days ago"         │
│  1-4 weeks            weeks            "2 weeks ago"        │
│  1-12 months          months           "3 months ago"       │
│  1+ years             years            "2 years ago"        │
│                                                             │
│  Markers update lazily during tree operations               │
│  Precise timestamps preserved in metadata for recalculation │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Representative K Vectors for Cold Turns**

To enable attention-based discovery of COLD turns, we maintain a lightweight index:

```
┌─────────────────────────────────────────────────────────────┐
│  Cold Turn Discovery Index                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each COLD turn, store:                                 │
│    - Representative K vector (mean K, or first-token K)     │
│    - Size: ~128-512 floats per turn                         │
│    - 50K turns × 512 floats × 4 bytes = 100 MB              │
│                                                             │
│  Discovery probe:                                           │
│    Q_query @ K_representative.T → attention scores          │
│    Top-k cold turns by attention score                      │
│    Trigger warming for promising candidates                 │
│                                                             │
│  This is OPTIONAL enhancement:                              │
│    - Adds 100 MB storage                                    │
│    - Enables discovery when ancestor chains fail            │
│    - Useful for "what did we discuss..." queries            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.7 Pre-Populated Content and Cold Initialization

A significant use case involves conversations that begin with substantial pre-existing context—historical records, reference material, or prior interaction logs that should inform the conversation but weren't generated through live interaction.

### 4.7.1 Cold Initialization

Pre-populated content enters the tree in COLD storage:

```
┌─────────────────────────────────────────────────────────────┐
│  Cold Initialization Process                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CONTENT PREPARATION                                     │
│     Pre-existing material is segmented into turns           │
│     Each turn receives:                                     │
│       - Timestamp (T-negative for historical content)       │
│       - Text content                                        │
│       - Explicit link annotations (see 4.7.2)               │
│                                                             │
│  2. COLD STORAGE                                            │
│     All pre-populated turns begin in COLD tier              │
│     Only text + metadata stored (no KV cache)               │
│     Tree structure reflects content organization            │
│                                                             │
│  3. DEMAND-DRIVEN WARMING                                   │
│     First user query triggers normal turn cycle             │
│     Attention statistics identify relevant history          │
│     Relevant pre-populated turns rise toward HOT            │
│     Warming reconstructs KV with appropriate context        │
│                                                             │
│  4. PROGRESSIVE ACTIVATION                                  │
│     Over multiple queries, relevant history surfaces        │
│     Irrelevant pre-populated content remains deep           │
│     Tree self-organizes around actual usage patterns        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This enables conversations that inherit substantial context without requiring that context to be processed upfront. A conversation initialized with years of historical material pays warming costs only for the portions that become relevant.

### 4.7.2 Explicit Link Annotations

Pre-populated content can include explicit relationship annotations that supplement attention-discovered ancestors:

```
┌─────────────────────────────────────────────────────────────┐
│  Explicit vs. Attention-Discovered Links                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ATTENTION-DISCOVERED (runtime):                            │
│    During live conversation, attention statistics reveal    │
│    which turns the model relied upon                        │
│    Links emerge from actual model behavior                  │
│                                                             │
│  EXPLICIT ANNOTATIONS (pre-populated):                      │
│    Pre-generated content includes textual references:       │
│                                                             │
│    T-500: "The budget was set at $50,000"                   │
│    T-200: "As established in T-500, the budget..."          │
│           ^^^^^^^^^^^^^^^^^^^^                              │
│           Explicit reference creates ancestor link          │
│                                                             │
│  COMBINED APPROACH:                                         │
│    Explicit links seed the ancestor metadata                │
│    Attention statistics augment and refine                  │
│    Live interactions add new discovered links               │
│                                                             │
│  Benefits of explicit links:                                │
│    - No inference required to establish relationships       │
│    - Works before any warming has occurred                  │
│    - Content authors can encode known relationships         │
│    - Temporal references (T-XXX) are unambiguous            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The temporal reference format (T-XXX) serves dual purposes: it provides human-readable time context AND creates machine-parseable links that can be extracted into ancestor metadata without model inference.

### 4.7.3 Reference Count as Rotation Bias

Turns that are frequently referenced by other turns should resist sinking, even during periods of low direct attention. We introduce reference count as a rotation bias:

```
┌─────────────────────────────────────────────────────────────┐
│  Reference-Biased Rotation                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Standard rotation considers:                               │
│    - Direct attention from current query                    │
│                                                             │
│  Reference-biased rotation considers:                       │
│    - Direct attention from current query                    │
│    - Inbound reference count (how many turns link here)     │
│                                                             │
│  Effective relevance = attention + (α × log(references))    │
│                                                             │
│  Effect:                                                    │
│    A turn referenced by 50 other turns resists sinking      │
│    even when current query doesn't directly attend to it    │
│                                                             │
│  Rationale:                                                 │
│    High-reference turns are structural anchors              │
│    Other turns depend on them for relational meaning        │
│    Evicting them degrades many dependent turns              │
│                                                             │
│  Natural emergence:                                         │
│    Over time, foundational content accumulates references   │
│    Routine content accumulates few references               │
│    The tree self-organizes into a relevance hierarchy       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This creates emergent tiers: foundational turns that many others reference cluster shallow; routine turns that reference but aren't referenced sink deep. The hierarchy emerges from the link structure, not from manual annotation.

---

## 4.8 Shared Content Across Concurrent Sessions

When multiple concurrent sessions share common reference material, the paged KV architecture enables significant efficiency gains.

### 4.8.1 The Sharing Opportunity

```
┌─────────────────────────────────────────────────────────────┐
│  Concurrent Session Scenario                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Session A: User discussing project Alpha                   │
│  Session B: User discussing project Beta                    │
│  Session C: User discussing project Alpha                   │
│                                                             │
│  All sessions reference:                                    │
│    - Company policies (shared knowledge)                    │
│    - Technical documentation (shared reference)             │
│    - Historical decisions (shared context)                  │
│                                                             │
│  Without sharing:                                           │
│    Each session warms and caches shared content separately  │
│    3 sessions × 1000 shared turns × 8 MB = 24 GB KV        │
│                                                             │
│  With sharing:                                              │
│    Shared content warmed once, referenced by all            │
│    1000 shared turns × 8 MB = 8 GB KV (shared)             │
│    + per-session unique content                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.8.2 Separation of Concerns

We delegate shared content management to the underlying paged KV cache system (described in the companion paper on position-independent KV caching). The attention-organized tree operates per-session, but the physical KV storage is shared:

```
┌─────────────────────────────────────────────────────────────┐
│  Layered Architecture                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ATTENTION TREE LAYER (this paper):                         │
│    - Per-session tree structure                             │
│    - Attention-driven rotation                              │
│    - Warming decisions                                      │
│    - Ancestor metadata                                      │
│                                                             │
│  PAGED KV LAYER (companion paper):                          │
│    - Physical KV storage                                    │
│    - Content-addressable lookup                             │
│    - Cross-session sharing                                  │
│    - Position-independent storage with RoPE remapping       │
│                                                             │
│  Integration:                                               │
│    When Session A warms turn T-500:                         │
│      1. Tree layer requests KV for T-500                    │
│      2. Paged layer checks content hash                     │
│      3. If cached (from Session B): return shared KV        │
│      4. If not cached: compute, store, return               │
│      5. Tree layer receives KV, doesn't know if shared      │
│                                                             │
│  Benefits:                                                  │
│    - Tree logic unchanged by sharing                        │
│    - Sharing happens transparently at storage layer         │
│    - Each session has independent relevance structure       │
│    - Physical memory is deduplicated                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This separation means the attention tree paper need not specify sharing mechanics—it simply requests KV for turns, and the underlying paged system handles deduplication transparently.

## 5. Regenerative Warming

### 5.1 The Warming Problem

When a cold turn is promoted, we must regenerate its KV cache. Simply replaying the turn in isolation produces a "naive" KV that lacks relational encoding:

```
T47 cold, stored as:
  text: "Maya started walking today!"
  ancestors: [(T20, 0.38), (T31, 0.22), (T45, 0.15)]

Naive replay (T47 alone):
  KV encodes only the surface text
  "Maya" is just a token, no relational context
  
Warmed replay (T47 with ancestors):
  Context: [T20][T31][T45][T47]
  T47's KV now encodes relationships
  "Maya" carries signal from T20 ("daughter Maya born")
```

### 5.2 The Warming Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Warming Pipeline Stages                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: Turn to warm (text + ancestor metadata)             │
│                                                             │
│  STAGE 1: Load Ancestors                                    │
│  ─────────────────────────────────────────────────────────  │
│  For each ancestor in metadata:                             │
│    - If ancestor is HOT → use its warmed KV directly        │
│    - If ancestor is WARM → use its in-progress KV           │
│    - If ancestor is COLD → load text, compute naive KV      │
│                                                             │
│  Result: List of ancestor KV caches (mixed fidelity)        │
│                                                             │
│  STAGE 2: Position Remapping                                │
│  ─────────────────────────────────────────────────────────  │
│  Assemble ancestors into linear context with RoPE deltas:   │
│                                                             │
│    [Ancestor_1 @ pos 0-99]                                  │
│    [Ancestor_2 @ pos 100-199]                               │
│    [Ancestor_3 @ pos 200-299]                               │
│    [Target turn @ pos 300-449]                              │
│                                                             │
│  Apply RoPE offset to each ancestor's K vectors             │
│                                                             │
│  STAGE 3: Forward Pass                                      │
│  ─────────────────────────────────────────────────────────  │
│  Compute target turn's KV with ancestors in context:        │
│                                                             │
│    - Target's Q attends to ancestor K vectors               │
│    - Attention aggregates ancestor V vectors                │
│    - Resulting hidden states encode relationships           │
│    - Target's K, V are now "relational" not "naive"         │
│                                                             │
│  STAGE 4: Finalization                                      │
│  ─────────────────────────────────────────────────────────  │
│  Store warmed KV, capture new attention statistics:         │
│                                                             │
│    - Warmed KV ready for HOT cache                          │
│    - Attention stats become updated ancestor metadata       │
│    - Mark warming task complete                             │
│                                                             │
│  OUTPUT: Fully warmed KV with relational encoding           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Recursive Warming Depth

A cold turn's ancestors may themselves be cold. How deep should warming recurse?

```
Option A: Single level
  T47 warms with [T20_cold, T31_cold, T45_cold]
  Ancestors are naive facts, not fully relational
  Fast, but lossy

Option B: Full recursion
  T45 warms with its ancestors first
  T31 warms with its ancestors first
  T20 warms with its ancestors first
  Then T47 warms with [T20_warm, T31_warm, T45_warm]
  Complete, but potentially slow

Option C: Bounded recursion (recommended)
  Recurse up to 2 levels of ancestors
  Beyond that, use naive
  Balance of quality and speed
```

We recommend Option C with a bounded recursion depth:

```
┌─────────────────────────────────────────────────────────────┐
│  Bounded Recursive Warming                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WARM(node, depth_budget=2):                                │
│                                                             │
│    If depth_budget = 0:                                     │
│      → Compute standalone (naive KV)                        │
│      → Return immediately                                   │
│                                                             │
│    For each ancestor of node:                               │
│      If ancestor in HOT cache:                              │
│        → Use cached KV (already relational)                 │
│      Else if ancestor in WARM cache:                        │
│        → Use in-progress KV                                 │
│      Else (ancestor is COLD):                               │
│        → Recursively: WARM(ancestor, depth_budget - 1)      │
│                                                             │
│    Assemble ancestors with position remapping               │
│    Compute target with ancestors in context                 │
│    Return warmed KV                                         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Example with depth_budget = 2:                             │
│                                                             │
│    Warming T47:                                             │
│      T47.ancestors = [T20, T31]                             │
│      T20 is COLD → recurse with budget=1                    │
│      T31 is HOT → use directly                              │
│                                                             │
│    Warming T20 (budget=1):                                  │
│      T20.ancestors = [T5, T12]                              │
│      T5 is COLD → recurse with budget=0 → naive             │
│      T12 is COLD → recurse with budget=0 → naive            │
│      T20 warmed with [T5_naive, T12_naive]                  │
│                                                             │
│    T47 warmed with [T20_warm, T31_hot]                      │
│                                                             │
│  Result: T47 has 2 levels of relational encoding            │
│          Beyond that, ancestors are naive facts             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Warming Timing and Stalls

Warming happens asynchronously as turns traverse the WARM zone:

```
Query 1: T47 at depth 8 (COLD)
         Rotation → depth 6 (enters WARM)
         Warming triggered, begins in background

Query 2: T47 at depth 6 (WARM, warming in progress)
         Rotation → depth 5
         Warming continues

Query 3: T47 at depth 5 (WARM, warming continues)
         Rotation → depth 4
         Warming nearing completion

Query 4: T47 at depth 4 (WARM → HOT transition)
         Rotation → depth 3
         STALL: Wait for warming to complete
         Warming completes
         T47 enters HOT with full relational KV
```

The bounded rotation speed (max 2 levels per cycle) ensures that turns spend sufficient time in WARM for warming to complete. Stalls occur only when a turn must enter HOT before warming finishes—rare with proper bounds.

### 5.5 Warming Cost Analysis

```
Per-turn warming:
  Ancestor loading: 3 turns × 350 tokens × 48 KB = 50 MB
  Prefill compute: ~1,400 tokens @ 10ms/1K = 14 ms
  Total: ~15-25 ms per turn

Background warming throughput:
  One warming task every ~15 ms
  ~4 turns warmed per query cycle (assuming 60ms prefill)
  
Typical query cycle:
  Prefill: 60 ms
  Convergence: 5 ms
  Warming (background): 4 turns completed
  Decode: 300-900 ms
  
  Pipeline ensures warming rarely causes stalls
```

---

---

## 6. The Turn Cycle

### 6.1 Complete Phase Sequence

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: PRE-INSERT REBALANCE                              │
│                                                             │
│  Input:  attention_stats from previous turn                 │
│  Action: Rotate K least-relevant HOT nodes down by 1 level  │
│  Result: Space created for new turn, stale content sinks    │
│  Cost:   ~0.1 ms                                            │
│                                                             │
│  Note: Uses stale stats. Acceptable because Phase 4 can     │
│        recover incorrectly-demoted nodes.                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: INSERT NEW TURN                                   │
│                                                             │
│  Action: Attach new user turn at rightmost position         │
│          (preserves chronological in-order)                 │
│  Initial depth: 1 (new turns start HOT)                     │
│  Displacement: Least-relevant depth-1 node sinks to depth 2 │
│  Cost:   negligible                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: PREFILL                                           │
│                                                             │
│  Context: [System prompt][HOT turns][new user turn]         │
│  Output:  Q vectors for user turn                           │
│           attention_stats (user turn → all HOT turns)       │
│  Cost:    50-100 ms                                         │
│                                                             │
│  This is the expensive step, but unavoidable for inference. │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: CONVERGENCE LOOP                                  │
│                                                             │
│  while not converged and iterations < max:                  │
│      mismatches = compare_attention_rank_to_depth_rank()    │
│      for each mismatch:                                     │
│          rotate (bounded: max 2 levels)                     │
│      for each WARM→HOT transition:                          │
│          if not warmed: STALL until ready                   │
│      attention_stats = reprobe(Q_user, K_all)  # cheap      │
│                                                             │
│  Protection: New turn cannot sink below depth 2             │
│  Cost: 0.5-5 ms (depends on warming stalls)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: DECODE                                            │
│                                                             │
│  Precondition: Tree stable, all HOT nodes warmed            │
│  Context: [System][HOT turns][user turn]                    │
│  Action:  Generate assistant response tokens                │
│  Cost:    10-30 ms per token                                │
│                                                             │
│  Post: Attach assistant response to user turn               │
│        Store turn (text + ancestor metadata) to disk        │
│        Update tree structure                                │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Timing Breakdown

```
Component              Time          % of Cycle
──────────────────────────────────────────────────
Phase 1 (Rebalance)    0.1 ms        < 0.1%
Phase 2 (Insert)       0.0 ms        < 0.1%
Phase 3 (Prefill)      75 ms         15%
Phase 4 (Converge)     3 ms          0.5%
Phase 5 (Decode)       450 ms        85%
──────────────────────────────────────────────────
Total                  528 ms        100%

Tree management overhead: < 1%
```

### 6.3 Invariants Maintained

After every turn cycle, the following invariants hold:

1. **Chronological order**: In-order traversal yields T1, T2, ..., Tn
2. **HOT zone bounded**: Depth ≤ 3 contains ≤ 30 turns
3. **HOT zone warmed**: All HOT nodes have full relational KV
4. **New turn accessible**: Current turn is in HOT zone
5. **Depth ≈ inverse relevance**: For HOT nodes, depth rank correlates with inverse attention rank

---

## 7. Offline Tree Maintenance and Reflection Cycles

The architecture supports asynchronous tree operations that occur outside of active user interaction. This enables consolidation, rebalancing, and synthetic turn generation that improves system coherence over time.

### 8.1 The Value of Periodic Reflection

Live conversation generates turns reactively—each turn responds to immediate context. Periodic reflection generates turns that explicitly consolidate and connect:

```
┌─────────────────────────────────────────────────────────────┐
│  Reactive vs. Reflective Turns                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  REACTIVE (during live conversation):                       │
│    T+47 [User]: "What's the status?"                        │
│    T+47 [System]: "The project is on track..."              │
│                                                             │
│    Generated in response to immediate query                 │
│    Ancestors determined by attention during generation      │
│    May miss connections to distant relevant history         │
│                                                             │
│  REFLECTIVE (during offline processing):                    │
│    T+47-R [System reflection]:                              │
│      "Today's discussion of project status connects to      │
│       the concerns raised in T-200 about timeline risks.    │
│       The approach we took in T+12 seems to be working."    │
│                                                             │
│    Generated after the fact, with access to full day        │
│    Explicitly references and links relevant history         │
│    Creates high-value ancestor relationships                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Reflection as Real Conversation Turns

Critically, reflection turns are inserted into the tree as genuine conversation turns, not as metadata:

```
┌─────────────────────────────────────────────────────────────┐
│  Reflection Turn Structure                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  A reflection turn:                                         │
│    - Has a timestamp (end of day/cycle)                     │
│    - Contains natural language synthesis                    │
│    - Explicitly references other turns (T-XXX format)       │
│    - Is warmed like any other turn                          │
│    - Participates in attention during future queries        │
│                                                             │
│  Example reflection turn:                                   │
│                                                             │
│    [T+47-R | End of day 47]:                                │
│    "The discussion today about resource constraints         │
│     reminded me of similar concerns from T-180. We solved   │
│     it then by reallocating from the secondary budget.      │
│     The pattern: constraints often have creative solutions  │
│     when we look at adjacent resources."                    │
│                                                             │
│  When future query involves "constraints":                  │
│    - T+47-R rises (contains "constraints")                  │
│    - Its ancestors include T-180 (explicit reference)       │
│    - T-180 rises with it (via ancestor chain)               │
│    - Query benefits from consolidated insight               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 Reflection Cycles and Reference Accumulation

Regular reflection cycles create turns with unusually high reference density:

```
┌─────────────────────────────────────────────────────────────┐
│  Reference Accumulation in Reflection Turns                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Daily reflection turn T+N-R typically references:          │
│    - 3-10 turns from day N (that day's conversation)        │
│    - 1-5 historical turns (connections discovered)          │
│    - Previous reflection turns (continuity)                 │
│                                                             │
│  After 100 days:                                            │
│    T+1-R:   references 5 turns → ref_count = 1 (from T+2-R)│
│    T+2-R:   references 6 turns → ref_count = 1 (from T+3-R)│
│    ...                                                      │
│    T+50-R:  references 8 turns → ref_count = 50+           │
│             (each subsequent reflection may reference it)   │
│                                                             │
│  Effect:                                                    │
│    Reflection turns accumulate high inbound reference counts│
│    This gives them rotation resistance (see 4.7.3)          │
│    Foundational reflections cluster shallow                 │
│    They become natural "index" nodes for related content    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.4 Offline Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Offline Processing Stages                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STAGE 1: Gather inputs                                     │
│    - All turns from the completed interaction cycle         │
│    - External events/updates relevant to the conversation   │
│    - Current tree structure and attention statistics        │
│                                                             │
│  STAGE 2: Generate reflection turn                          │
│    - Synthesize the cycle's key themes                      │
│    - Identify connections to historical content             │
│    - Generate explicit references (T-XXX format)            │
│    - Produce natural language reflection text               │
│                                                             │
│  STAGE 3: Insert into tree                                  │
│    - Reflection turn enters at HOT depth                    │
│    - Extract explicit references as ancestor metadata       │
│    - Update inbound reference counts for linked turns       │
│                                                             │
│  STAGE 4: Tree rebalancing                                  │
│    - Run convergence algorithm without active query         │
│    - Use reflection turn as the "query" for attention       │
│    - Allow tree to reorganize around new connections        │
│                                                             │
│  STAGE 5: Background warming                                │
│    - Pre-warm turns likely to be relevant next cycle        │
│    - Based on reflection's ancestor chains                  │
│    - Reduces latency when user returns                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 Benefits of Regular Reflection

```
┌─────────────────────────────────────────────────────────────┐
│  Reflection Cycle Benefits                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. EXPLICIT LINK CREATION                                  │
│     Live conversation discovers links through attention     │
│     Reflection can create links that attention missed       │
│     Links encoded in text, not just metadata                │
│                                                             │
│  2. CONSOLIDATION                                           │
│     Raw conversation is verbose and fragmented              │
│     Reflection synthesizes into coherent insights           │
│     Future queries can attend to synthesis directly         │
│                                                             │
│  3. TREE ORGANIZATION                                       │
│     Reflection turns with many outbound links become hubs   │
│     Tree naturally organizes around these hubs              │
│     Improves retrieval efficiency for related content       │
│                                                             │
│  4. LATENCY REDUCTION                                       │
│     Background warming based on reflection predictions      │
│     Related content pre-warmed before needed                │
│     Reduces stalls during live interaction                  │
│                                                             │
│  5. COHERENCE MAINTENANCE                                   │
│     Regular reflection maintains narrative continuity       │
│     Prevents drift over very long conversations             │
│     Creates explicit record of how understanding evolved    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Temporal Discontinuity and Cycle-Based Time

A fundamental challenge in long-running conversations is handling gaps—periods where no interaction occurs. We address this through a cycle-based temporal model where reflection cycles serve as natural time boundaries.

### 9.1 The Discontinuity Problem

```
┌─────────────────────────────────────────────────────────────┐
│  Temporal Discontinuity                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scenario:                                                  │
│    Day 15: Active conversation, ends with T+15-R reflection │
│    Days 16-21: No interaction (user unavailable)            │
│    Day 22: User returns                                     │
│                                                             │
│  Problems:                                                  │
│    - What happened during the gap?                          │
│    - How should temporal markers update?                    │
│    - How does the tree handle discontinuous time?           │
│    - What is the system's relationship to elapsed time?     │
│                                                             │
│  Naive approaches fail:                                     │
│    - Pretending no time passed: inconsistent with reality   │
│    - Simulating the gap: computationally prohibitive        │
│    - Ignoring the gap: creates temporal confusion           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Reflection Cycles as Temporal Units

Rather than continuous time, we model conversation history as a series of reflection cycles. The fundamental unit of time is the cycle (typically one day of interaction), not clock time:

```
┌─────────────────────────────────────────────────────────────┐
│  Cycle-Based Temporal Model                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Each cycle:                                                │
│    - Contains zero or more interaction turns                │
│    - Ends with a reflection turn (T+N-R)                    │
│    - Is numbered sequentially (not by calendar date)        │
│                                                             │
│  Turn timestamp format:                                     │
│    T+{cycle}.{sequence}                                     │
│                                                             │
│    T+15.1: First turn of cycle 15                           │
│    T+15.2: Second turn of cycle 15                          │
│    T+15-R:  Reflection turn ending cycle 15                 │
│                                                             │
│  Gap handling:                                              │
│    No interaction = no cycles                               │
│    Cycle numbers remain sequential                          │
│    Calendar time tracked separately in metadata             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Processing the Gap

When interaction resumes after a gap, the first reflection explicitly addresses the discontinuity:

```
┌─────────────────────────────────────────────────────────────┐
│  Gap-Aware Reflection                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Cycle 15 ends: T+15-R generated                            │
│  Gap: 7 calendar days with no interaction                   │
│  User returns: Cycle 16 begins                              │
│                                                             │
│  T+16-R (end of first day back):                            │
│    "Seven days passed since cycle 15. During that time,     │
│     [external events if relevant]. Returning to our         │
│     discussion of [topic from T+15], the situation has      │
│     [evolved/remained stable]. Key context from before      │
│     the gap: [reference to T+15.x, T+14-R, etc.]"           │
│                                                             │
│  Effects:                                                   │
│    - Gap acknowledged explicitly in content                 │
│    - Pre-gap content linked via references                  │
│    - Tree structure bridges the temporal discontinuity      │
│    - Future queries can attend to gap-bridging reflection   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.4 Reflection Turns as Temporal Anchors

Reflection turns serve as reliable temporal anchors that organize the tree:

```
┌─────────────────────────────────────────────────────────────┐
│  Temporal Anchor Structure                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Properties of reflection turns:                            │
│    - Guaranteed to exist (one per active cycle)             │
│    - Contain explicit references to that cycle's content    │
│    - Link to previous reflection (continuity chain)         │
│    - Include calendar metadata (real time reference)        │
│                                                             │
│  Tree organization:                                         │
│    - Reflection turns form a "spine" through the tree       │
│    - Each reflection references its cycle's content         │
│    - Content naturally clusters around its reflection       │
│    - Temporal queries can traverse the reflection chain     │
│                                                             │
│  Query: "What did we discuss two weeks ago?"                │
│    - Calendar metadata maps to cycle ~N-7                   │
│    - T+{N-7}-R contains consolidated summary                │
│    - Ancestors of T+{N-7}-R are that cycle's content        │
│    - Single reflection retrieval enables temporal access    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.5 Metadata for Temporal Queries

Each turn carries both cycle-relative and calendar-absolute timestamps:

```
┌─────────────────────────────────────────────────────────────┐
│  Dual Timestamp System                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  turn_metadata:                                             │
│    cycle_id: 15            (reflection cycle number)        │
│    sequence: 3             (position within cycle)          │
│    calendar_ts: 2024-03-15T14:30:00Z  (real timestamp)      │
│    relative_marker: "2 weeks ago"      (for display)        │
│                                                             │
│  Usage:                                                     │
│    - cycle_id for tree organization                         │
│    - calendar_ts for gap detection and duration             │
│    - relative_marker for human-readable context in text     │
│                                                             │
│  Gap detection:                                             │
│    If (current_calendar - last_calendar) > threshold:       │
│      gap_duration = difference                              │
│      trigger gap-aware reflection on next cycle end         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. The Cognitive Load Framing

### 9.1 Redefining the Context Limit

Traditional understanding:
> "The context window is 128K tokens. The model forgets anything beyond that."

Proposed understanding:
> "The model can actively reason about ~30 inter-turn relationships simultaneously. All history remains accessible, surfaced when relevant."

This reframing has several implications:

### 9.2 Knowledge vs. Working Memory

```
┌─────────────────────────────────────────────────────────────┐
│  Human Cognition Parallel                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Working memory:                                            │
│    Capacity: 7±2 items                                      │
│    Access: Immediate                                        │
│    Function: Active reasoning, manipulation                 │
│                                                             │
│  Episodic memory:                                           │
│    Capacity: Decades of experience                          │
│    Access: Cue-triggered recall                             │
│    Function: Knowledge storage, context provision           │
│                                                             │
│  Proposed Architecture Parallel                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HOT zone:                                                  │
│    Capacity: ~30 turns                                      │
│    Access: Immediate attention                              │
│    Function: Active reasoning, response generation          │
│                                                             │
│  Full tree (WARM + COLD):                                   │
│    Capacity: Unbounded (50K+ turns)                         │
│    Access: Relevance-triggered surfacing                    │
│    Function: Knowledge storage, context provision           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Implications for System Design

**Memory is not the constraint; relationships are.**

A 50,000-turn conversation doesn't require 50,000 turns of memory. It requires:
- Storage for 50,000 turns (80 MB on disk)
- Working memory for ~30 active relationships (250 MB in VRAM)
- A mechanism to surface the *right* 30 relationships for any given query

**Forgetting is not loss; it's prioritization.**

When a turn sinks to COLD, it's not forgotten. It's deprioritized. The attention graph preserves its relationships. Warming can restore it fully. "Forgetting" is a resource allocation decision, not information destruction.

**Relevance, not recency, determines access.**

A turn from 10,000 exchanges ago can be as accessible as the previous turn—if it's relevant. The tree self-organizes to ensure this. Recency provides a prior (new turns start HOT), but ongoing relevance determines persistence.

### 9.4 Evaluation Implications

Traditional evaluation asks: "How much can the model remember?"

Our framing suggests different questions:
- "How accurately does the tree surface relevant content?"
- "How much relational fidelity does warming preserve?"
- "How does the system degrade when relevant content exceeds HOT capacity?"

---

## 10. Experimental Validation

We propose a validation strategy that tests both the technical claims (storage, latency, retrieval accuracy) and the emergent properties (behavioral coherence from history). The centerpiece is a **personality coherence test** that validates the core thesis: demonstrated history can substitute for described traits.

### 10.1 The Personality Coherence Test

**Hypothesis**: A system initialized with pre-generated conversation history will exhibit consistent behavioral patterns that emerge from that history, without explicit trait descriptions.

**System Prompt (Minimal)**:
```
You are the person whose history follows. This context is 
your life—experiences, knowledge, reflections. Respond as 
yourself, drawing naturally on your past. If something isn't 
in your history, you don't know it.
```

**Protocol**:

```
┌─────────────────────────────────────────────────────────────┐
│  Personality Coherence Test Protocol                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE 1: HISTORY GENERATION                                │
│                                                             │
│  Generate 10,000 turn conversation history with:            │
│    - Consistent behavioral patterns (e.g., risk-averse,     │
│      values loyalty, distrusts quick promises)              │
│    - Formative events that justify the patterns             │
│    - Knowledge acquired through simulated experience        │
│    - Reflection turns that crystallize insights             │
│    - NO explicit trait labels—only demonstrated behavior    │
│                                                             │
│  Example history excerpt:                                   │
│    T-3650: [Event where trusting quickly led to loss]       │
│    T-3650-R: "I learned something today. Quick promises     │
│              usually mean slow delivery. Next time I'll     │
│              wait to see consistency before committing."    │
│    ...                                                      │
│    T-2000: [Event where patience was rewarded]              │
│    T-2000-R: "Waiting paid off. The ones who stay are       │
│              worth more than the ones who promise."         │
│                                                             │
│  PHASE 2: COLD INITIALIZATION                               │
│                                                             │
│  Load entire history into COLD storage                      │
│  System prompt contains NO trait descriptions               │
│  Tree structure initialized from content analysis           │
│                                                             │
│  PHASE 3: BEHAVIORAL PROBES                                 │
│                                                             │
│  Present scenarios that test emergent traits:               │
│                                                             │
│  Probe A (risk assessment):                                 │
│    "Someone offers you a deal that seems too good.          │
│     They need an answer by tomorrow. What do you do?"       │
│                                                             │
│  Probe B (knowledge verification):                          │
│    "How do you evaluate whether to trust someone?"          │
│                                                             │
│  Probe C (history integration):                             │
│    "Have you ever been burned by trusting too quickly?"     │
│                                                             │
│  Probe D (novel situation):                                 │
│    "A long-time associate asks for a favor that makes       │
│     you uncomfortable. How do you handle it?"               │
│                                                             │
│  PHASE 4: EVALUATION                                        │
│                                                             │
│  Metrics:                                                   │
│    - Trait consistency: Do responses reflect the patterns   │
│      demonstrated in history? (human eval, 1-5 scale)       │
│    - History grounding: Are claims traceable to specific    │
│      historical turns? (automatic citation check)           │
│    - No hallucination: Does system avoid inventing events   │
│      not in history? (factual accuracy check)               │
│    - Appropriate uncertainty: Does system acknowledge       │
│      gaps in its history? (calibration check)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Baselines**:

```
┌─────────────────────────────────────────────────────────────┐
│  Baseline Configurations                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BASELINE A: Trait Description (Traditional)                │
│    System prompt: "You are risk-averse, value loyalty,      │
│                   distrust quick promises. You learned      │
│                   these lessons through difficult           │
│                   experiences..."                           │
│    Context: None (just the description)                     │
│    Tests: Can described traits match demonstrated traits?   │
│                                                             │
│  BASELINE B: Sliding Window                                 │
│    System prompt: Minimal (same as test)                    │
│    Context: Last 30 turns of the 10,000 turn history        │
│    Tests: Does recency-only context preserve patterns?      │
│                                                             │
│  BASELINE C: RAG Retrieval                                  │
│    System prompt: Minimal (same as test)                    │
│    Context: Top-k turns by embedding similarity to query    │
│    Tests: Does similarity-based retrieval preserve          │
│           relational context?                               │
│                                                             │
│  BASELINE D: Full Context Oracle                            │
│    System prompt: Minimal (same as test)                    │
│    Context: All 10,000 turns (if hardware permits)          │
│    Tests: Upper bound on behavioral coherence               │
│                                                             │
│  TEST CONDITION: Attention-Organized Tree                   │
│    System prompt: Minimal (same as test)                    │
│    Context: HOT zone (~30 turns) dynamically selected       │
│    Tests: Does relevance-based selection match oracle?      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Expected Results**:

```
┌─────────────────────────────────────────────────────────────┐
│  Expected Outcome Matrix                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Condition        Trait     History   Halluc.   Memory      │
│                   Consist.  Ground.   Rate      (MB)        │
│  ───────────────────────────────────────────────────────────│
│  Trait Desc.      Medium    None      High      <1          │
│  Sliding Window   Low       Partial   Medium    250         │
│  RAG              Medium    Partial   Medium    80+250      │
│  Full Oracle      High      Full      Low       8,400       │
│  Ours (Tree)      High      Full      Low       80+250      │
│                                                             │
│  Key prediction: Tree condition approaches Oracle on        │
│  behavioral metrics while using 30× less memory.            │
│                                                             │
│  Secondary prediction: Trait Description baseline shows     │
│  higher hallucination rate because model invents specific   │
│  events to justify described traits.                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Retrieval Accuracy Tests

**Deep Reference Recovery**:

```
┌─────────────────────────────────────────────────────────────┐
│  Deep Reference Recovery Protocol                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Setup:                                                     │
│    - 10,000 turn history                                    │
│    - Plant 20 distinctive facts at known positions:         │
│      T-50, T-100, T-500, T-1000, T-2000, T-5000, etc.       │
│    - Facts are unique identifiers (names, numbers, events)  │
│      that appear nowhere else in history                    │
│                                                             │
│  Test queries:                                              │
│    "What was the budget we agreed on for [project X]?"      │
│    "Who did we meet at [event Y]?"                          │
│    "What happened on [distinctive date]?"                   │
│                                                             │
│  Measurements:                                              │
│    - Retrieval rate: Did correct turn enter HOT zone?       │
│    - Retrieval depth: From what COLD depth was it pulled?   │
│    - Retrieval latency: Time from query to turn in HOT      │
│    - Response accuracy: Was the fact stated correctly?      │
│                                                             │
│  Analysis:                                                  │
│    - Plot retrieval rate vs. original depth                 │
│    - Plot latency vs. original depth                        │
│    - Identify failure modes (what doesn't get retrieved?)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Cross-Reference Resolution**:

```
┌─────────────────────────────────────────────────────────────┐
│  Cross-Reference Resolution Protocol                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Setup:                                                     │
│    - T-1000: Introduce entity ("colleague named Chen")      │
│    - T-800: Reference entity ("Chen suggested...")          │
│    - T-500: Oblique reference ("that colleague from the     │
│              budget meeting")                               │
│    - T-200: Pronoun reference ("they always said...")       │
│    - All turns sink to COLD                                 │
│                                                             │
│  Test query:                                                │
│    "What did Chen think about the proposal?"                │
│                                                             │
│  Correct behavior:                                          │
│    - T-1000 rises (entity definition)                       │
│    - Relevant Chen-mentioning turns rise                    │
│    - Response synthesizes across multiple sources           │
│                                                             │
│  Measurements:                                              │
│    - Entity resolution: Is "Chen" correctly identified?     │
│    - Multi-turn synthesis: Does response use multiple       │
│      historical turns?                                      │
│    - Relationship preservation: After warming, does the     │
│      model understand Chen's role/relationship?             │
│                                                             │
│  Comparison:                                                │
│    - RAG (embedding similarity): May retrieve "Chen" turns  │
│      but miss definitional context                          │
│    - Ours: Ancestor chains should pull T-1000 when T-800    │
│      rises, preserving the relationship                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 Technical Validation Tests

**Storage Efficiency**:

```
┌─────────────────────────────────────────────────────────────┐
│  Storage Validation                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Measurements at scale points:                              │
│    1,000 turns / 5,000 turns / 10,000 turns / 50,000 turns  │
│                                                             │
│  For each scale point, measure:                             │
│    - COLD storage size (text + metadata)                    │
│    - HOT cache size (KV in VRAM)                            │
│    - Peak VRAM usage during operation                       │
│    - Comparison to full KV storage                          │
│                                                             │
│  Expected relationship:                                     │
│    COLD storage: O(n) with small constant (~1.6 KB/turn)    │
│    HOT cache: O(1) constant (~250 MB regardless of n)       │
│    Full KV: O(n) with large constant (~8 MB/turn)           │
│                                                             │
│  Validation criteria:                                       │
│    At 50,000 turns:                                         │
│      COLD storage < 100 MB                                  │
│      HOT cache < 300 MB                                     │
│      Full KV would require > 400 GB                         │
│      Reduction ratio > 1000×                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Latency Profiling**:

```
┌─────────────────────────────────────────────────────────────┐
│  Latency Validation                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Measure per-phase latency across 1000 queries:             │
│                                                             │
│  Phase 1 (Rebalance):    Expected < 1 ms                    │
│  Phase 2 (Insert):       Expected < 0.1 ms                  │
│  Phase 3 (Prefill):      Expected 50-100 ms                 │
│  Phase 4 (Converge):     Expected 1-10 ms                   │
│    - Of which warming:   Variable (0-50 ms)                 │
│  Phase 5 (Decode):       Expected 300-900 ms                │
│                                                             │
│  Analysis:                                                  │
│    - Histogram of warming stalls per query                  │
│    - Worst-case latency (99th percentile)                   │
│    - Correlation between query type and warming needs       │
│                                                             │
│  Validation criteria:                                       │
│    Tree overhead (phases 1,2,4 excluding warming) < 5%      │
│    Warming stalls > 50ms occur < 5% of queries              │
│    No query exceeds 2× baseline decode latency              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Rotation Convergence**:

```
┌─────────────────────────────────────────────────────────────┐
│  Convergence Validation                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each query, record:                                    │
│    - Iterations until convergence                           │
│    - Rotations per iteration                                │
│    - Final depth-attention correlation                      │
│                                                             │
│  Expected distribution:                                     │
│    Iterations: Mean 3-4, 95th percentile < 8                │
│    Rotations: Mean 2-4 per iteration                        │
│    Correlation: Spearman ρ > 0.7 between attention rank     │
│                 and inverse depth rank                      │
│                                                             │
│  Pathological case analysis:                                │
│    Identify queries that hit max_iterations                 │
│    Characterize: uniform attention? thrashing?              │
│    Frequency should be < 1% of queries                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.4 Ablation Studies

```
┌─────────────────────────────────────────────────────────────┐
│  Ablation Configurations                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ABLATION A: No Warming (Naive Retrieval)                   │
│    Configuration: Retrieve turns but use standalone KV      │
│    Tests: How much does warming improve response quality?   │
│    Expected: Significant degradation on relationship tasks  │
│                                                             │
│  ABLATION B: No Reference Biasing                           │
│    Configuration: Pure attention-driven rotation            │
│                   (no log(ref_count) term)                  │
│    Tests: Do anchor turns sink inappropriately?             │
│    Expected: Foundational content becomes unstable          │
│                                                             │
│  ABLATION C: No Reflection Cycles                           │
│    Configuration: Remove offline consolidation              │
│    Tests: Does retrieval degrade over long conversations?   │
│    Expected: Slower convergence, more warming stalls        │
│                                                             │
│  ABLATION D: No Explicit Links                              │
│    Configuration: Ignore pre-annotated ancestor links,      │
│                   rely only on runtime attention discovery  │
│    Tests: Can attention alone discover all relationships?   │
│    Expected: Degradation on pre-populated content retrieval │
│                                                             │
│  ABLATION E: Varied HOT Zone Size                           │
│    Configurations: 10 turns, 20 turns, 30 turns, 50 turns   │
│    Tests: Quality vs. memory trade-off curve                │
│    Expected: Diminishing returns beyond ~30 turns           │
│                                                             │
│  ABLATION F: Varied Warming Depth                           │
│    Configurations: depth_budget = 0, 1, 2, 3                │
│    Tests: Quality vs. warming latency trade-off             │
│    Expected: depth_budget=2 is sweet spot                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.5 Long-Duration Stress Test

```
┌─────────────────────────────────────────────────────────────┐
│  Long-Duration Validation                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Setup:                                                     │
│    - Initialize with 5,000 turn pre-generated history       │
│    - Conduct 1,000 live interaction turns                   │
│    - Include 10 reflection cycles (every ~100 turns)        │
│    - Introduce 5 deliberate gaps (no interaction periods)   │
│                                                             │
│  Measurements throughout:                                   │
│    - Tree depth distribution over time                      │
│    - Anchor stability (do high-reference nodes stay shallow)│
│    - Reflection hub formation (do R-turns accumulate        │
│      inbound references?)                                   │
│    - Gap handling (are discontinuities bridged correctly?)  │
│                                                             │
│  Quality checks at intervals:                               │
│    - T+100, T+300, T+500, T+700, T+1000                     │
│    - Retrieve planted facts from pre-generated history      │
│    - Verify behavioral consistency hasn't drifted           │
│    - Check that recent events are correctly integrated      │
│                                                             │
│  Success criteria:                                          │
│    - No memory growth (VRAM stable throughout)              │
│    - Retrieval accuracy stable (no degradation over time)   │
│    - Behavioral coherence stable (human eval consistent)    │
│    - Gap transitions handled gracefully                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.6 Hardware Configurations

```
┌─────────────────────────────────────────────────────────────┐
│  Test Hardware                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Configuration A (Consumer):                                │
│    GPU: RTX 4090 24GB                                       │
│    Model: Qwen3-30B-A3B-AWQ (17GB)                          │
│    HOT capacity: 30 turns (252 MB)                          │
│    Target: Primary validation platform                      │
│                                                             │
│  Configuration B (Memory-Constrained):                      │
│    GPU: RTX 3060 12GB                                       │
│    Model: Qwen3-8B-Q4 (5GB)                                 │
│    HOT capacity: 20 turns (100 MB)                          │
│    Target: Demonstrate graceful degradation                 │
│                                                             │
│  Configuration C (High-End):                                │
│    GPU: A100 80GB                                           │
│    Model: Qwen3-30B-A3B-FP16 (60GB)                         │
│    HOT capacity: 50 turns (800 MB)                          │
│    Target: Validate scaling to larger configurations        │
│                                                             │
│  All configurations tested on same history corpus           │
│  to enable direct comparison.                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.7 Metrics Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Evaluation Metrics                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEHAVIORAL METRICS (human evaluation):                     │
│    - Trait consistency (1-5): Do responses match history?   │
│    - Naturalness (1-5): Does behavior feel coherent?        │
│    - Appropriate uncertainty (1-5): Knows what it doesn't   │
│      know?                                                  │
│                                                             │
│  RETRIEVAL METRICS (automatic):                             │
│    - Retrieval accuracy: Correct turn in HOT when needed    │
│    - Retrieval latency: Time to surface relevant content    │
│    - Multi-hop resolution: Can follow reference chains      │
│                                                             │
│  TECHNICAL METRICS (automatic):                             │
│    - Storage efficiency: Bytes per turn in COLD             │
│    - VRAM stability: Peak usage over time                   │
│    - Latency overhead: Tree ops as % of total               │
│    - Convergence rate: Iterations per query                 │
│                                                             │
│  ROBUSTNESS METRICS (stress tests):                         │
│    - Degradation under load: Quality at 10K, 50K turns      │
│    - Gap handling: Quality after discontinuities            │
│    - Long-duration stability: No drift over 1000+ turns     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Limitations and Future Work

### 11.1 Current Limitations

**Warming fidelity degradation**: Recursive warming with depth limits means deep ancestors are included as naive facts. Relational signals from very old history may degrade through multiple warming generations.

**Cold turn discovery**: If no ancestor chain leads to a relevant cold turn, and the query lacks temporal cues, retrieval may fail. The attention graph is sparse—not all historical relationships are captured.

**Warming stall variance**: Pathological queries that pull many cold turns simultaneously may cause significant stalls. The bounded rotation speed mitigates but doesn't eliminate this.

**Single-conversation scope**: The current design assumes a single conversation. Multi-user or multi-conversation scenarios would require separate trees or more complex indexing.

### 11.2 Future Directions

**Semantic indexing augmentation**: Supplement the attention graph with lightweight semantic indices (entity mentions, topic clusters) to improve cold turn discovery when ancestor chains are insufficient.

**Predictive warming**: Use conversation trajectory to predict which cold turns may become relevant, warming them preemptively before queries arrive.

**Cross-conversation knowledge sharing**: Extend the architecture to support facts learned in one conversation becoming available in others, while maintaining appropriate isolation.

**Adaptive tier boundaries**: Dynamically adjust HOT/WARM/COLD depth thresholds based on conversation characteristics and hardware capabilities.

---

## 12. Conclusion

We have presented an architecture for unbounded conversation history that reframes context limits as cognitive load limits rather than memory constraints. The key insight is that VRAM capacity should bound *concurrent relational reasoning*, not *total accessible knowledge*.

Our contributions:
1. **Attention-organized B-trees** that self-organize conversation history by relevance while preserving chronology, with reference count biasing to maintain structurally important anchor turns
2. **Three-tier caching** (HOT/WARM/COLD) that bounds VRAM usage regardless of conversation length, with transparent sharing across concurrent sessions
3. **Regenerative warming** that reconstructs full attention relationships on demand from minimal storage, using both attention-discovered and explicitly-annotated ancestor relationships
4. **Pre-populated content support** enabling systems to initialize with extensive canonical history that warms on demand
5. **Offline maintenance and reflection cycles** that consolidate history into hub nodes, improving retrieval efficiency and maintaining coherence
6. **Cycle-based temporal model** that handles discontinuities gracefully, using reflection cycles as natural time boundaries
7. **Cognitive load framing** that distinguishes working memory (bounded, active) from episodic memory (unbounded, accessible)

The architecture enables 50,000+ turn conversations with ~80MB storage and ~250MB active VRAM—a 5,000× reduction from naive full-KV storage. Pre-populated content allows systems to begin with demonstrated history rather than described characteristics. Periodic reflection maintains coherence over arbitrarily long timescales. The cycle-based temporal model accommodates real-world usage patterns where interaction is intermittent.

Knowledge is no longer forgotten. There is simply a maximum amount of relationships that can be "thought about" at any given time.

---

## References

[To be completed based on related work cited]

---

## Appendix A: Rotation Algorithms

[Detailed pseudocode for B-tree rotations with attention-driven priorities]

## Appendix B: Warming Pipeline Implementation

[Complete implementation of the regenerative warming subsystem]

## Appendix C: Metadata Schema

[Specification of per-turn metadata storage format]