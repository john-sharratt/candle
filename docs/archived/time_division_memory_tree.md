# Time-Division Summary Tree for Unbounded NPC Life Recall (v2)

**Abstract**

This document specifies a summary tree architecture for unbounded NPC life histories (100,000+ turns). The design uses fixed budget allocation across tree levels, compression-aware summarization, and density-based selection to retrieve an optimal mix of broad context, period narrative, and specific detail. Each level serves a distinct purpose: structural routing (L0-L3), era framing (L4), period context (L5), and specific detail (leaf turns).

**Core insight**: Different compression ratios serve different purposes. A 400:1 summary orients the reader to an era; a 20:1 summary tells the story of a period; raw turns provide exact quotes and specific scenes. The final context should contain all three, with budget allocation ensuring each purpose is served.

---

## 1. Introduction

Prior approaches to NPC memory retrieval treat all content uniformly—either retrieving raw turns by relevance or summarizing everything to fit context limits. This creates a tension: summaries lose specific detail; raw turns lack broader context.

We propose a different framing: **content at different compression ratios serves different purposes**, and effective retrieval combines them deliberately.

Consider a query about "Maya's feelings about the Portland move":

- **Era framing** (400:1): "Maya Chen, age 7, moved from Seattle to Portland. Close friend Jamie Santos. Emotional arc: resistant → grieving → accepting."
- **Period context** (20:1): "The hardest day came while packing her bookshelf. She'd been quiet all morning, but broke down holding their shared book..."
- **Specific detail** (1:1): "'Jamie and I were going to read all of these together,' she said through tears. 'We made a promise.'"

A good response draws from all three levels. The architecture ensures this through fixed budget allocation and level-specific summarization.

**Contributions:**

1. **Fixed budget allocation by level**: Pre-determined token budgets ensure representation from each compression tier
2. **Compression-aware summarization**: Distinct prompts for each level, optimized for what that compression ratio can preserve
3. **Density-based selection**: Attention score per token as universal ranking within each level's budget
4. **Structural routing separation**: L0-L3 used only for pruning during descent, never included in final context

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    SUMMARY TREE ARCHITECTURE                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  STRUCTURAL ZONE (routing only, no summaries):               │
│  ─────────────────────────────────────────────────────────── │
│  L0 (root):     [1 node]         100,000 turns               │
│                      │                                       │
│  L1:            [10 nodes]       10,000 turns each           │
│                      │                                       │
│  L2:            [100 nodes]      1,000 turns each            │
│                      │                                       │
│  L3:            [1,000 nodes]    100 turns each              │
│                                                              │
│  CONTENT ZONE (summaries for final context):                 │
│  ─────────────────────────────────────────────────────────── │
│  L4:            [5,000 nodes]    20 turns each               │
│                 Era summaries (400:1 compression)            │
│                 → 20% of token budget                        │
│                                                              │
│  L5:            [5,000 nodes]    20 turns each               │
│                 Period summaries (20:1 compression)          │
│                 → 20% of token budget                        │
│                                                              │
│  LEAF TURNS:    [100,000 turns]  Raw turn storage            │
│                 Specific detail (1:1)                        │
│                 → 40% of token budget                        │
│                                                              │
│  RECENT:        [last 5 turns]   Conversation continuity     │
│                 → 20% of token budget                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Level Roles

| Level | Nodes | Turns/Node | Compression | Role | In Context? |
|-------|-------|------------|-------------|------|-------------|
| L0-L3 | 1-1000 | 100-100K | — | Structural routing | No |
| L4 | ~5,000 | ~20 | 400:1 | Era framing | Yes (20%) |
| L5 | ~5,000 | ~20 | 20:1 | Period context | Yes (20%) |
| Leaf | ~100K | 1 | 1:1 | Specific detail | Yes (40%) |
| Recent | 5 | 1 | 1:1 | Continuity | Yes (20%) |

### 2.2 Why This Structure?

**L0-L3 (Structural)**: These levels exist only for routing efficiency. With fanout 10, we can prune 90% of branches at each level during descent. No summaries needed—we never include them in final context.

**L4 (Era Framing)**: At 400:1 compression, these summaries cannot preserve specific moments or quotes. They CAN preserve: who was involved, what major events occurred, the emotional arc, key places and objects. Purpose: orient the reader to the era.

**L5 (Period Context)**: At 20:1 compression, these summaries can preserve 3-5 verbatim quotes, specific scenes, and narrative flow. Purpose: tell the story of a period in a way that could answer questions directly.

**Leaf Turns (Specific Detail)**: Raw turns contain exact dialogue, specific actions, nuanced emotional moments. Purpose: provide the precise detail that summaries cannot.

**Recent Turns (Continuity)**: The last few turns maintain conversational coherence regardless of what historical content is retrieved.

---

## 3. Tree Structure

### 3.1 Node Structure

Each node contains:

```
Node {
    time_range: [min_turn, max_turn]    // temporal coverage
    level: int                          // 0-5 or LEAF
    children: Node[]                    // child pointers (null for leaves)
    summary: string | null              // null for L0-L3, content for L4-L5
    turn_ids: int[] | null              // only for leaf nodes
}
```

### 3.2 Fanout and Depth

For 100K turns with target ~20 turns per leaf:

```
┌──────────────────────────────────────────────────────────────┐
│  Tree Parameters (100K turns)                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Leaf target: 20 turns each → 5,000 leaf groups              │
│  Fanout: 10 (each node has 10 children)                      │
│                                                              │
│  Level    Nodes       Turns/Node                             │
│  ─────────────────────────────────────────────────────────── │
│  L0       1           100,000                                │
│  L1       10          10,000                                 │
│  L2       100         1,000                                  │
│  L3       1,000       100                                    │
│  L4       5,000       20        ← L4 and L5 may be same     │
│  L5       5,000       20          physical level with        │
│  LEAF     100,000     1           different summary types    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Note**: L4 and L5 can be the same physical tree level, with each node containing BOTH an L4 summary (era framing) and an L5 summary (period context). This simplifies the tree while maintaining distinct summary purposes.

### 3.3 Alternative: Merged L4/L5

```
┌──────────────────────────────────────────────────────────────┐
│  Merged Summary Node (L4 + L5 at same level)                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Node {                                                      │
│      time_range: [T-8450, T-8470]                            │
│      level: SUMMARY_LEVEL                                    │
│      children: [leaf_node_1, leaf_node_2, ...]               │
│                                                              │
│      l4_summary: "Maya Chen, age 7, Portland move period.    │
│                   Key people: Jamie Santos (best friend),    │
│                   parents David and Lin. Emotional arc:      │
│                   resistant → grieving. Major events:        │
│                   packing breakdown [T-8472], ..."           │
│                   (400-600 tokens, 400:1 compression)        │
│                                                              │
│      l5_summary: "The hardest day of the move came while     │
│                   packing Maya's room. She'd been quiet      │
│                   all morning, methodically wrapping         │
│                   trinkets, until she reached the            │
│                   bookshelf. 'Jamie and I were going to      │
│                   read all of these together,' she said..."  │
│                   (400-800 tokens, 20:1 compression)         │
│  }                                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

This merged approach:
- Simplifies tree structure (fewer levels)
- Both summaries cover same time range but serve different purposes
- Selection chooses between them based on attention density

---

## 4. Budget Allocation

### 4.1 Fixed Allocation

```
┌──────────────────────────────────────────────────────────────┐
│  TOKEN BUDGET ALLOCATION                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Total context budget: 12,000 tokens                         │
│                                                              │
│  Layer              Budget    Tokens    Purpose              │
│  ─────────────────────────────────────────────────────────── │
│  Recent turns       20%       2,400     Continuity           │
│  L4 summaries       20%       2,400     Era framing          │
│  L5 summaries       20%       2,400     Period context       │
│  Leaf turns         40%       4,800     Specific detail      │
│                                                              │
│  Rationale:                                                  │
│  - Leaf turns get most budget (contain actual answers)       │
│  - L4/L5 get equal budget (different compression, same value)│
│  - Recent turns ensure conversational coherence              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Why Fixed Allocation?

Dynamic allocation risks bias toward one level. With attention-based selection, summaries often score higher than turns (they're optimized for hooks), but turns contain the actual answers.

Fixed allocation ensures:
1. Specific detail (turns) always represented even when summaries score higher
2. Era context (L4) included even when period detail (L5) dominates
3. Predictable context structure for consistent generation quality

### 4.3 Tunable Parameters

```
┌──────────────────────────────────────────────────────────────┐
│  Configuration Options                                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  PARAMETER           DEFAULT     RANGE       NOTES           │
│  ─────────────────────────────────────────────────────────── │
│  total_budget        12,000      8K-16K      Model dependent │
│  recent_ratio        0.20        0.10-0.25   Continuity need │
│  l4_ratio            0.20        0.15-0.25   Era importance  │
│  l5_ratio            0.20        0.15-0.25   Period depth    │
│  leaf_ratio          0.40        0.35-0.50   Specificity     │
│                                                              │
│  Ratios must sum to 1.0                                      │
│                                                              │
│  QUERY-ADAPTIVE ALLOCATION (optional):                       │
│  - Factual queries: increase leaf_ratio to 0.50              │
│  - "Tell me about" queries: balance across levels            │
│  - "What happened during" queries: increase l5_ratio         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Density-Based Selection

### 5.1 Core Metric

Selection within each level uses attention density:

```
density(candidate) = attention_score(candidate) / token_count(candidate)
```

This favors content that is both relevant (high attention) and concise (few tokens).

### 5.2 Selection Algorithm

```python
def select_context(tree, query, budget=12000):
    """
    Fixed budget allocation with density-based selection within each level.
    """
    # Allocate budgets
    budgets = {
        'recent': int(budget * 0.20),
        'l4':     int(budget * 0.20),
        'l5':     int(budget * 0.20),
        'leaf':   int(budget * 0.40),
    }
    
    # Phase 1: Descent to collect candidates
    candidates = descend_tree(tree, query)
    
    # Partition by level
    l4_candidates = [c for c in candidates if c.level == 'L4']
    l5_candidates = [c for c in candidates if c.level == 'L5']
    leaf_candidates = [c for c in candidates if c.level == 'LEAF']
    
    # Phase 2: Score all candidates in one probe
    all_candidates = l4_candidates + l5_candidates + leaf_candidates
    scores = attention_probe(all_candidates, query)
    for c, s in zip(all_candidates, scores):
        c.score = s
        c.tokens = count_tokens(c.text)
        c.density = s / c.tokens
    
    # Phase 3: Select by density within each level's budget
    selected = {
        'recent': get_recent_turns(budgets['recent']),
        'l4':     select_by_density(l4_candidates, budgets['l4']),
        'l5':     select_by_density(l5_candidates, budgets['l5']),
        'leaf':   select_by_density(leaf_candidates, budgets['leaf']),
    }
    
    return selected


def select_by_density(candidates, budget):
    """
    Greedy selection by attention density until budget exhausted.
    """
    candidates.sort(key=lambda c: c.density, reverse=True)
    
    selected = []
    remaining = budget
    
    for c in candidates:
        if c.tokens > remaining:
            continue
        selected.append(c)
        remaining -= c.tokens
    
    return selected
```

### 5.3 Variable-Length Content

With density-based selection, content length is unconstrained:

```
┌──────────────────────────────────────────────────────────────┐
│  VARIABLE LENGTH BY CONTENT RICHNESS                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  L4 SUMMARIES (era framing):                                 │
│  - Eventful era (major life transition): 500-600 tokens      │
│  - Routine era (steady state): 200-300 tokens                │
│  Density handles this: routine era is shorter but also less  │
│  relevant, so density may be similar to eventful era.        │
│                                                              │
│  L5 SUMMARIES (period context):                              │
│  - Rich emotional period: 600-800 tokens                     │
│  - Quiet routine period: 150-300 tokens                      │
│  Density favors rich periods when query needs depth.         │
│                                                              │
│  LEAF TURNS (specific detail):                               │
│  - Eventful day with dialogue: 400-700 tokens                │
│  - Brief routine day: 100-200 tokens                         │
│  Density naturally selects eventful days for relevant queries│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.4 Minimum Density Cutoff

To avoid filling budget with low-value content:

```python
MIN_DENSITY = 0.00005  # tunable threshold

def select_by_density(candidates, budget):
    candidates.sort(key=lambda c: c.density, reverse=True)
    
    selected = []
    remaining = budget
    
    for c in candidates:
        if c.density < MIN_DENSITY:
            break  # remaining candidates are noise
        if c.tokens > remaining:
            continue
        selected.append(c)
        remaining -= c.tokens
    
    return selected
```

Better to return under-budget with high-relevance content than to fill budget with filler.

---

## 6. Compression-Aware Summarization

### 6.1 Design Principle

Each compression level can preserve different information. Summarization prompts should be explicit about what CAN and CANNOT be preserved at each ratio.

### 6.2 L4 Summary Prompt (400:1 Compression - Era Framing)

```
┌──────────────────────────────────────────────────────────────┐
│  L4 SUMMARIZATION PROMPT                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Summarize these turns into a narrative passage.             │
│                                                              │
│  COMPRESSION: ~400 turns → 400-600 tokens                    │
│  PURPOSE: Orient the reader to this ERA of the NPC's life    │
│                                                              │
│  At this compression ratio, you CANNOT preserve:             │
│  - Individual scenes or specific moments                     │
│  - Verbatim quotes or exact dialogue                         │
│  - Day-to-day progression or sequence                        │
│  - Subtle emotional nuances                                  │
│                                                              │
│  At this compression ratio, you MUST preserve:               │
│  - All significant PEOPLE (full names, relationship to NPC)  │
│  - Major EVENTS (with turn citations [T-###])                │
│  - Overall EMOTIONAL ARC (starting state → ending state)     │
│  - Key PLACES the NPC frequented                             │
│  - Significant OBJECTS that mattered                         │
│  - SKILLS learned or demonstrated                            │
│  - RELATIONSHIP changes (who got closer, who drifted)        │
│                                                              │
│  STYLE: Dense factual overview. Every sentence should        │
│  contain queryable hooks (names, places, events, emotions).  │
│  Think "chapter summary" or "era overview."                  │
│                                                              │
│  EXAMPLE OUTPUT:                                             │
│  "Maya Chen (age 7) relocated from Seattle to Portland with  │
│  parents David and Lin [T-8400]. This period centered on     │
│  separation from best friend Jamie Santos—they'd been        │
│  inseparable since age 4, bonding over fantasy books and     │
│  backyard adventures at Jamie's house. Maya's emotional arc: │
│  resistant and angry → grieving → slowly accepting. Key      │
│  moments: breakdown while packing bookshelf [T-8472], first  │
│  positive reaction to new room [T-8550], receiving Jamie's   │
│  letter [T-9340]. Significant objects: The Lion the Witch    │
│  and the Wardrobe (shared reading with Jamie), telescope     │
│  (dad's gift for new room). Ended period cautiously          │
│  optimistic about Portland but still missing Jamie deeply.   │
│  Relationship with parents strained by move decision but     │
│  recovering."                                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 L5 Summary Prompt (20:1 Compression - Period Context)

```
┌──────────────────────────────────────────────────────────────┐
│  L5 SUMMARIZATION PROMPT                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Summarize these turns into a narrative passage.             │
│                                                              │
│  COMPRESSION: ~20 turns → 400-800 tokens                     │
│  PURPOSE: Tell the STORY of this period, usable as answer    │
│                                                              │
│  At this compression ratio, you CAN preserve:                │
│  - 3-5 verbatim QUOTES with turn citations [T-###]           │
│  - Specific scenes and emotional moments                     │
│  - Sequence and causation between events                     │
│  - Character reactions and dialogue                          │
│                                                              │
│  At this compression ratio, you MUST preserve:               │
│  - All people who appear (names, roles in the scene)         │
│  - The emotional through-line (what changed, why it matters) │
│  - Key dialogue that reveals character or advances story     │
│  - Cause-and-effect between moments                          │
│  - Sensory details that ground the scene                     │
│                                                              │
│  STYLE: Narrative prose that could directly answer a         │
│  question about this period. Think "detailed scene summary"  │
│  or "key moments with context."                              │
│                                                              │
│  EXAMPLE OUTPUT:                                             │
│  "The hardest day of the move came while packing Maya's      │
│  room [T-8470..T-8478]. She'd been quiet all morning,        │
│  methodically wrapping trinkets in newspaper with unusual    │
│  focus for a seven-year-old, until she reached the           │
│  bookshelf. Picking up The Lion, the Witch and the Wardrobe, │
│  she froze. 'Jamie and I were going to read all of these     │
│  together,' she said, tears starting. 'We made a promise'    │
│  [T-8472]. Her mother Lin sat with her on the floor for      │
│  nearly an hour, not pushing, just present. 'We can visit,'  │
│  Lin offered finally. 'And Jamie can come stay in the        │
│  summer.' Maya shook her head—at seven, promises felt        │
│  permanent and distances infinite [T-8474]. By evening,      │
│  she'd packed the books herself, carefully, arranged in      │
│  the order she and Jamie had planned to read them. She       │
│  asked for extra tape to seal that box shut—she didn't       │
│  want anyone else touching it [T-8478]."                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.4 Prompt Design Principles

```
┌──────────────────────────────────────────────────────────────┐
│  SUMMARIZATION PROMPT PRINCIPLES                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. EXPLICIT COMPRESSION CONSTRAINTS                         │
│     Tell the model what it CANNOT preserve at this ratio.    │
│     This prevents attempts to cram everything in.            │
│                                                              │
│  2. EXPLICIT PRESERVATION REQUIREMENTS                       │
│     Tell the model what it MUST preserve.                    │
│     This ensures queryable hooks are present.                │
│                                                              │
│  3. PURPOSE STATEMENT                                        │
│     Tell the model HOW this summary will be used.            │
│     "Orient the reader" vs "Answer questions directly"       │
│                                                              │
│  4. STYLE GUIDANCE                                           │
│     "Dense factual" vs "Narrative prose"                     │
│     Affects readability and information density.             │
│                                                              │
│  5. CONCRETE EXAMPLE                                         │
│     Show, don't just tell.                                   │
│     Example calibrates length, tone, and content balance.    │
│                                                              │
│  6. CITATION REQUIREMENTS                                    │
│     [T-###] citations enable grounding and fact-checking.    │
│     L4: cite major events. L5: cite quotes and key moments.  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Query-Time Pipeline

### 7.1 Complete Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  RETRIEVAL PIPELINE                                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT: query, tree, budget=12000                            │
│                                                              │
│  PHASE 1: STRUCTURAL DESCENT (L0-L3)                         │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: Prune irrelevant branches, identify candidate      │
│           regions without loading summaries into context     │
│                                                              │
│  for level in [L0, L1, L2, L3]:                              │
│      scores = routing_probe(current_nodes, query)            │
│      current_nodes = expand_above_threshold(scores)          │
│                                                              │
│  Output: ~50-200 candidate L4 regions                        │
│                                                              │
│  PHASE 2: CONTENT COLLECTION                                 │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: Gather all content that might appear in context    │
│                                                              │
│  l4_candidates = [node.l4_summary for node in candidate_regions]
│  l5_candidates = [node.l5_summary for node in candidate_regions]
│  leaf_candidates = [turn for node in candidate_regions       │
│                          for turn in node.turns]             │
│                                                              │
│  PHASE 3: ATTENTION SCORING                                  │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: Compute relevance scores for all candidates        │
│                                                              │
│  all_candidates = l4 + l5 + leaf                             │
│  scores = attention_probe(all_candidates, query)             │
│  for c, s in zip(all_candidates, scores):                    │
│      c.density = s / c.tokens                                │
│                                                              │
│  PHASE 4: BUDGET-ALLOCATED SELECTION                         │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: Select best content within each level's budget     │
│                                                              │
│  selected = {                                                │
│      'recent': get_recent(budget * 0.20),                    │
│      'l4':     select_by_density(l4_candidates, budget*0.20),│
│      'l5':     select_by_density(l5_candidates, budget*0.20),│
│      'leaf':   select_by_density(leaf_candidates, budget*0.40)
│  }                                                           │
│                                                              │
│  PHASE 5: CONTEXT ASSEMBLY                                   │
│  ─────────────────────────────────────────────────────────── │
│  Purpose: Order selected content for generation              │
│                                                              │
│  context = [                                                 │
│      system_prompt,                                          │
│      sorted(selected['l4'], by=time),    # era framing       │
│      sorted(selected['l5'], by=time),    # period context    │
│      sorted(selected['leaf'], by=time),  # specific detail   │
│      selected['recent'],                  # continuity       │
│      query                                                   │
│  ]                                                           │
│                                                              │
│  OUTPUT: assembled context for generation                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Structural Descent Detail

```python
def descend_tree(tree, query, prune_threshold=0.01):
    """
    Navigate L0-L3 using lightweight routing probes.
    No summaries loaded—just structure for pruning.
    """
    current_level = [tree.root]
    
    for level in range(4):  # L0 through L3
        # Lightweight probe: just node metadata, not full summaries
        # Can use node's time_range and child count as features
        scores = routing_probe(current_level, query)
        
        # Expand nodes above threshold
        next_level = []
        for node, score in zip(current_level, scores):
            if score >= prune_threshold:
                next_level.extend(node.children)
        
        current_level = next_level
    
    # current_level now contains L4 nodes (summary nodes)
    return current_level
```

### 7.3 Chunked Attention Probing

For large candidate pools:

```python
def attention_probe_chunked(candidates, query, chunk_size=100):
    """
    Probe candidates in chunks to manage memory.
    """
    all_scores = []
    
    for chunk in chunks(candidates, chunk_size):
        # Build probe prompt
        prompt = build_probe_prompt(chunk, query)
        
        # FP8 routing probe (Q·K only, no V, no MLP)
        scores = fp8_attention_scores(prompt, chunk)
        
        all_scores.extend(scores)
    
    return all_scores
```

---

## 8. Context Assembly

### 8.1 Assembly Order

```
┌──────────────────────────────────────────────────────────────┐
│  CONTEXT ASSEMBLY ORDER                                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [System prompt]                                             │
│      ↓                                                       │
│  [L4 summaries - chronological]                              │
│      Purpose: Reader knows who, what era, major events       │
│      ↓                                                       │
│  [L5 summaries - chronological]                              │
│      Purpose: Reader knows detailed narrative of periods     │
│      ↓                                                       │
│  [Leaf turns - chronological]                                │
│      Purpose: Reader has specific quotes and scenes          │
│      ↓                                                       │
│  [Recent turns - chronological]                              │
│      Purpose: Conversation continuity                        │
│      ↓                                                       │
│  [Query]                                                     │
│                                                              │
│  This order: broad → specific → recent → query               │
│  Mimics natural narrative structure.                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Why This Order Works

The model reads context sequentially. This order ensures:

1. **L4 first**: Model knows who Maya is, what era we're discussing
2. **L5 second**: Model knows the narrative arc of relevant periods
3. **Leaf third**: Model has specific quotes and scenes to draw from
4. **Recent fourth**: Model has immediate conversational context
5. **Query last**: Model knows what to answer, with full context available

### 8.3 Alternative: Interleaved Chronological

```
┌──────────────────────────────────────────────────────────────┐
│  ALTERNATIVE: FULLY INTERLEAVED                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [System prompt]                                             │
│  [L4 for era A][L5 for period A1][Turns from A1]             │
│  [L4 for era B][L5 for period B1][Turns from B1]             │
│  [Recent turns]                                              │
│  [Query]                                                     │
│                                                              │
│  Pro: More natural timeline, era→period→detail per region    │
│  Con: More complex assembly logic                            │
│  Con: Same era might appear multiple times if periods span   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

The simpler level-grouped approach is recommended unless evaluation shows interleaved performs better.

---

## 9. Summary Construction (Sleep Phase)

### 9.1 When to Summarize

Summaries are constructed during "sleep" (offline processing), not during query time:

```
┌──────────────────────────────────────────────────────────────┐
│  SUMMARY CONSTRUCTION TRIGGERS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INITIAL CONSTRUCTION:                                       │
│  - When NPC is first created with history                    │
│  - When bulk historical content is imported                  │
│                                                              │
│  INCREMENTAL UPDATE:                                         │
│  - When a leaf node accumulates 20 turns (create L5/L4)      │
│  - During periodic "sleep" cycles (consolidation)            │
│                                                              │
│  NEVER during query time—use existing summaries.             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 Bottom-Up Construction

```
┌──────────────────────────────────────────────────────────────┐
│  SUMMARY CONSTRUCTION ORDER                                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Group raw turns into leaf nodes (~20 turns each)         │
│                                                              │
│  2. For each leaf node:                                      │
│     - Generate L5 summary (20:1, period context)             │
│     - Generate L4 summary (400:1, era framing)               │
│     Both from the same raw turns, different prompts          │
│                                                              │
│  3. Build tree structure (L0-L3) without summaries           │
│     These levels are structural only.                        │
│                                                              │
│  4. Store summaries with their nodes                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 9.3 Incremental Updates

When new turns arrive:

```python
def add_turns(tree, new_turns):
    """
    Add new turns and update summaries as needed.
    """
    # Add to current leaf node
    current_leaf = tree.get_current_leaf()
    current_leaf.turns.extend(new_turns)
    
    # If leaf is full, finalize and create new leaf
    if len(current_leaf.turns) >= LEAF_SIZE:
        # Generate summaries for completed leaf
        current_leaf.l5_summary = generate_l5_summary(current_leaf.turns)
        current_leaf.l4_summary = generate_l4_summary(current_leaf.turns)
        
        # Create new leaf for future turns
        new_leaf = create_leaf_node(parent=current_leaf.parent)
        tree.current_leaf = new_leaf
        
        # Rebalance tree if needed
        tree.rebalance_if_needed()
```

---

## 10. Evaluation Metrics

### 10.1 Retrieval Quality

```
┌──────────────────────────────────────────────────────────────┐
│  RETRIEVAL METRICS                                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  COVERAGE: Did we retrieve content from the right regions?   │
│  - Plant facts at known turns, query for them                │
│  - Measure: was the relevant region in candidate set?        │
│                                                              │
│  SELECTION: Did density select the best content?             │
│  - Compare selected vs oracle (human-chosen) content         │
│  - Measure: overlap between selected and oracle sets         │
│                                                              │
│  ANSWER QUALITY: Did the response use the right information? │
│  - Ask questions with known answers in history               │
│  - Measure: accuracy, citation correctness, hallucination    │
│                                                              │
│  LEVEL UTILIZATION: Is budget allocation effective?          │
│  - Track which level's content appears in responses          │
│  - Adjust ratios if one level consistently unused            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 10.2 Latency Breakdown

```
┌──────────────────────────────────────────────────────────────┐
│  LATENCY TARGETS                                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase                     Target      Notes                 │
│  ─────────────────────────────────────────────────────────── │
│  Structural descent (L0-3) < 10ms      Lightweight probes    │
│  Content loading           < 5ms       Disk/memory read      │
│  Attention scoring         < 50ms      FP8 chunked probe     │
│  Selection                 < 1ms       In-memory sort        │
│  Context assembly          < 1ms       String concat         │
│  ─────────────────────────────────────────────────────────── │
│  Total retrieval           < 70ms                            │
│                                                              │
│  Generation (prefill)      50-100ms    Depends on context    │
│  Generation (decode)       varies      Token count dependent │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. Comparison to Other Approaches

```
┌──────────────────────────────────────────────────────────────┐
│  APPROACH COMPARISON                                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Approach          │ Strength        │ Weakness              │
│  ──────────────────┼─────────────────┼─────────────────────  │
│  Sliding window    │ Simple, fast    │ No long-term memory   │
│  RAG (embedding)   │ Semantic search │ Loses narrative arc   │
│  Full summarize    │ Compact         │ Loses specific detail │
│  This approach     │ Multi-level     │ More complex          │
│                                                              │
│  KEY DIFFERENCES:                                            │
│                                                              │
│  vs RAG:                                                     │
│  - RAG retrieves by embedding similarity (semantic)          │
│  - We retrieve by attention (relational, contextual)         │
│  - RAG returns isolated chunks; we return layered context    │
│                                                              │
│  vs Single-level summarization:                              │
│  - Single level must choose compression ratio                │
│  - We use multiple ratios for different purposes             │
│  - Single level loses either context or detail; we keep both │
│                                                              │
│  vs Sliding window:                                          │
│  - Window keeps only recent; we keep entire history          │
│  - Window has no retrieval; we retrieve by relevance         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 12. Implementation Notes

### 12.1 Storage Requirements

```
┌──────────────────────────────────────────────────────────────┐
│  STORAGE ESTIMATE (100K turns)                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw turns:      100K × 400 tokens × 4 bytes = 160 MB        │
│  L5 summaries:   5K × 600 tokens × 4 bytes = 12 MB           │
│  L4 summaries:   5K × 500 tokens × 4 bytes = 10 MB           │
│  Tree structure: ~10K nodes × 100 bytes = 1 MB               │
│  ─────────────────────────────────────────────────────────── │
│  Total:          ~183 MB per NPC                             │
│                                                              │
│  With compression (gzip ~3:1): ~60 MB per NPC                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 12.2 Memory During Query

```
┌──────────────────────────────────────────────────────────────┐
│  MEMORY DURING QUERY                                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Loaded for attention probe:                                 │
│  - L4 candidates: ~200 × 500 tokens = 100K tokens            │
│  - L5 candidates: ~200 × 600 tokens = 120K tokens            │
│  - Leaf candidates: ~500 × 400 tokens = 200K tokens          │
│  Total: ~420K tokens for scoring                             │
│                                                              │
│  With chunking (100 candidates per chunk):                   │
│  - Max ~50K tokens loaded at once                            │
│  - Multiple chunks processed sequentially                    │
│                                                              │
│  Final context: 12K tokens (fixed budget)                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 13. Future Extensions

### 13.1 Query-Adaptive Budget

```
┌──────────────────────────────────────────────────────────────┐
│  ADAPTIVE ALLOCATION (future work)                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Classify query type, adjust ratios:                         │
│                                                              │
│  FACTUAL ("What did Maya say when..."):                      │
│    leaf_ratio = 0.55, l5_ratio = 0.15, l4_ratio = 0.10       │
│                                                              │
│  NARRATIVE ("Tell me about Maya's friendship with..."):      │
│    leaf_ratio = 0.30, l5_ratio = 0.35, l4_ratio = 0.15       │
│                                                              │
│  OVERVIEW ("Who is Maya?"):                                  │
│    leaf_ratio = 0.20, l5_ratio = 0.25, l4_ratio = 0.35       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 13.2 Cross-NPC Retrieval

```
┌──────────────────────────────────────────────────────────────┐
│  CROSS-NPC QUERIES (future work)                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Query: "What does Maya know about Jamie's family?"          │
│                                                              │
│  Requires:                                                   │
│  - Retrieve from Maya's tree (what she experienced)          │
│  - Filter to content involving Jamie's family                │
│  - Maya can only know what she directly experienced          │
│                                                              │
│  Implementation:                                             │
│  - Entity index: which NPCs appear in which turns            │
│  - Cross-reference during retrieval                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 13.3 Integration with KV Rematerialization

```
┌──────────────────────────────────────────────────────────────┐
│  COMBINED PIPELINE (with KV rematerialization paper)         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  This paper: DISCOVERY (find relevant content)               │
│  KV remat paper: ENCODING (reconstruct attention relations)  │
│                                                              │
│  Combined:                                                   │
│  1. Summary tree descent → identify relevant turns           │
│  2. KV rematerialization → reconstruct relational encoding   │
│  3. Generation → respond with full context                   │
│                                                              │
│  The summary tree solves "what to retrieve"                  │
│  KV rematerialization solves "how to encode it"              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Complete Pseudocode

```python
# ============================================================
# SUMMARY TREE: COMPLETE IMPLEMENTATION
# ============================================================

class SummaryTree:
    def __init__(self, turns, config):
        self.config = config
        self.root = self._build_tree(turns)
    
    def _build_tree(self, turns):
        """Build tree with L4/L5 summaries at leaf level."""
        # Group turns into leaf nodes
        leaf_groups = list(chunks(turns, self.config.leaf_size))
        
        # Create leaf nodes with summaries
        leaves = []
        for group in leaf_groups:
            node = Node(
                level='SUMMARY',
                time_range=(group[0].time, group[-1].time),
                turns=group,
                l4_summary=self._generate_l4(group),
                l5_summary=self._generate_l5(group),
            )
            leaves.append(node)
        
        # Build structural levels (L3 → L0)
        current_level = leaves
        for level in reversed(range(4)):  # L3, L2, L1, L0
            parent_level = []
            for group in chunks(current_level, self.config.fanout):
                parent = Node(
                    level=f'L{level}',
                    time_range=(group[0].time_range[0], group[-1].time_range[1]),
                    children=group,
                    # No summaries for structural levels
                )
                parent_level.append(parent)
            current_level = parent_level
        
        return current_level[0]  # Root
    
    def retrieve(self, query, budget=12000):
        """Main retrieval pipeline."""
        # Phase 1: Structural descent
        candidate_regions = self._descend(query)
        
        # Phase 2: Collect content
        l4_candidates = [r.l4_summary for r in candidate_regions]
        l5_candidates = [r.l5_summary for r in candidate_regions]
        leaf_candidates = [t for r in candidate_regions for t in r.turns]
        
        # Phase 3: Score all candidates
        all_candidates = l4_candidates + l5_candidates + leaf_candidates
        scores = self._attention_probe(all_candidates, query)
        for c, s in zip(all_candidates, scores):
            c.score = s
            c.density = s / c.tokens
        
        # Phase 4: Budget-allocated selection
        budgets = {
            'recent': int(budget * self.config.recent_ratio),
            'l4': int(budget * self.config.l4_ratio),
            'l5': int(budget * self.config.l5_ratio),
            'leaf': int(budget * self.config.leaf_ratio),
        }
        
        selected = {
            'recent': self._get_recent(budgets['recent']),
            'l4': self._select_by_density(l4_candidates, budgets['l4']),
            'l5': self._select_by_density(l5_candidates, budgets['l5']),
            'leaf': self._select_by_density(leaf_candidates, budgets['leaf']),
        }
        
        # Phase 5: Assemble context
        return self._assemble(selected, query)
    
    def _descend(self, query, threshold=0.01):
        """Navigate L0-L3, pruning irrelevant branches."""
        current = [self.root]
        
        while current and current[0].level.startswith('L'):
            scores = self._routing_probe(current, query)
            next_level = []
            for node, score in zip(current, scores):
                if score >= threshold:
                    next_level.extend(node.children)
            current = next_level
        
        return current  # Summary-level nodes
    
    def _select_by_density(self, candidates, budget):
        """Greedy selection by attention density."""
        candidates.sort(key=lambda c: c.density, reverse=True)
        
        selected = []
        remaining = budget
        
        for c in candidates:
            if c.density < self.config.min_density:
                break
            if c.tokens > remaining:
                continue
            selected.append(c)
            remaining -= c.tokens
        
        return selected
    
    def _assemble(self, selected, query):
        """Order content for generation."""
        context = []
        context.append(self.config.system_prompt)
        context.extend(sorted(selected['l4'], key=lambda x: x.time))
        context.extend(sorted(selected['l5'], key=lambda x: x.time))
        context.extend(sorted(selected['leaf'], key=lambda x: x.time))
        context.extend(selected['recent'])
        context.append(query)
        return '\n\n'.join(str(c) for c in context)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class TreeConfig:
    # Tree structure
    leaf_size: int = 20
    fanout: int = 10
    
    # Budget allocation
    total_budget: int = 12000
    recent_ratio: float = 0.20
    l4_ratio: float = 0.20
    l5_ratio: float = 0.20
    leaf_ratio: float = 0.40
    
    # Selection
    min_density: float = 0.00005
    prune_threshold: float = 0.01
    
    # System
    system_prompt: str = "..."
```

---

## Appendix B: Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Structural levels | L0-L3, no summaries | Only needed for routing/pruning |
| Summary levels | L4 + L5 at same tree level | Different purposes, same coverage |
| Budget allocation | Fixed ratios | Prevents bias toward summaries |
| Selection metric | Attention density | Balances relevance and conciseness |
| Context order | Level-grouped | Simpler than interleaved, broad→specific |
| Summary construction | Offline (sleep) | Avoid query-time latency |
| Coverage exclusion | None needed | Levels serve different purposes |

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| Attention density | `attention_score / token_count` — selection metric |
| Era framing (L4) | High-compression summary for orientation |
| Period context (L5) | Medium-compression summary with quotes |
| Leaf turns | Raw turns at full detail |
| Structural descent | Navigating L0-L3 to prune branches |
| Budget allocation | Pre-determined token limits per level |
| Compression ratio | Turns summarized : summary tokens |