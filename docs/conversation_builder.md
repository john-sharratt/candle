# candle-conversation: Multi-Layer Projection Design

## 1. Purpose and Scope

The `candle-conversation` crate is a projection engine. It compresses an unbounded, layered substrate of conversation content into a fixed-size context window for the LLM, applying declared rules at every level of the hierarchy.

The substrate grows without bound — turns accumulate across cognitive cycles, conversations span sessions, every layer of the substrate produces content over time. The window does not grow. Every cognitive step requires a fresh projection: which substrate content survives the budget, in which order, framed by which system prompt.

### Responsibilities

The crate owns:

- A declared schema describing what content can exist where
- Runtime tracking of turn existence and the identifiers needed to address content
- Budget reconciliation across layers, groups, and sections under declared rules
- Emission of a structured system prompt and an ordered linear turn list at projection time

### Non-responsibilities

The crate does not own:

- Turn or section *content* — the caller stores content and resolves it from `(GroupId, TurnIndex)` or `SectionId` keys after projection emits them
- Tokenization — token counts are supplied through the resolver; no tokenizer touches this crate
- Scoring — scores are computed externally (typically from a Binary Directional Provenance scan against a live query) and supplied through the resolver per projection

The crate is a pure structural reconciler. Everything dynamic flows through the resolver trait.

## 2. Substrate Model

A *schema* is a one-time declaration of intent. It names the layers, the groups within them, the sections of the system prompt, and the rules each node carries. After construction the schema is immutable. Content is appended into groups over time; the schema itself never changes.

```
Schema
├── window: { total, system_prompt, turns }
├── SystemPrompt
│   └── Section, Section, Section, ...      [static, ordered by declaration]
└── Layers                                   [ordered by declaration, layer 0 first]
    ├── Layer { name, description, ... }
    │   ├── score_formula                    [turn-score → group-score aggregation]
    │   ├── budget defaults
    │   └── Groups
    │       ├── Group { selection, budget, score_threshold }
    │       │   └── Turns                    [append-only, opaque to crate]
    │       └── Group { ... }
    └── Layer { ... }
```

Layers are ordered. Groups within a layer are unordered structurally — at projection time they are sorted by their derived group score. Turns within a group are ordered by insertion (which is time order, since appends are append-only). Sections within the system prompt are ordered by declaration.

### Identifiers

The crate hands out two kinds of stable identifiers:

- `LayerId`, `GroupId`, `SectionId` — assigned at construction time when the schema is parsed. Stable for the lifetime of the builder.
- `TurnIndex` — assigned at append time, scoped to a single group, monotonically increasing.

The caller uses these identifiers when resolving content and when interpreting the projection output. The crate uses them internally to address budget pressure and selection.

## 3. YAML Schema

The schema is authored as YAML. A complete worked example for a small substrate:

```yaml
window:
  total: 32000
  system_prompt: 4000
  turns: 28000

system_prompt:
  sections:
    - id: frame
      priority: 100
      min_percent: 30
      content: |
        You are operating as a tactical layer in a multi-layer cognitive substrate.
        Your role is to integrate substrate content into a coherent first-person report.
    - id: values
      priority: 80
      min_percent: 20
      content: |
        Honesty over fluency. Precision over reassurance.
    - id: guidance
      priority: 40
      max_percent: 30
      content: |
        Reach for substrate content over fabrication. When uncertain, say so.

layers:
  - name: perceptual_ground
    description: |
      Ground-truth specialists. Type, structure, low-level facts.
      Read by all higher layers; reads nothing.
    score_formula: max
    score_threshold: 0.1
    budget:
      priority: 30
    groups:
      - id: type_specialist
        selection: { kind: top_k, k: 3 }
        score_threshold: 0.2
      - id: structure_specialist
        selection: { kind: top_k, k: 3 }

  - name: motivational
    description: |
      Goal- and mission-level reasoning. Biases mission attention.
    score_formula: top_k_mean
    score_formula_k: 3
    budget:
      priority: 60
    groups:
      - id: active_mission
        selection: { kind: top_k, k: 5 }
        budget: { priority: 80, min_percent: 30 }
      - id: goal_pressure
        selection: { kind: single }
        budget: { priority: 40 }

  - name: dialogue
    description: |
      The current conversation with the user. Only one active per session.
    score_formula: max
    budget:
      priority: 100
      min_percent: 50
    groups:
      - id: primary_conversation
        selection:
          kind: conversation
          recent: 8
          historical_top_k: 12
        budget: { priority: 100 }
```

Everything numeric is static at parse time. Everything dynamic enters through the resolver: turn token counts, turn scores, derived group scores via the layer's formula. Defaults for absent budget fields are crate-defined (even distribution, no min, no max), not declared in YAML.

## 4. Selection Rules

Selection determines which turns within a group survive into the projection. After selection, surviving turns are emitted in insertion order regardless of score (with one exception, noted below).

The closed set of selection rules:

**`always_visible`** — every turn in the group survives selection. The group still competes for budget; oversized turns are dropped at the budget pass.

**`top_k(k)`** — the k highest-scored turns survive, after the score threshold filters out ineligible turns. Ties broken by insertion order (earlier wins).

**`single`** — the single highest-scored turn survives. Used for groups where only one entry is ever relevant at a time (a single goal pressure, a single active threat).

**`conversation { recent, historical_top_k }`** — composite for the natural shape of an ongoing conversation. The most recent N turns survive unconditionally; the next-best K turns by score from the rest of the group also survive. The full set emits in insertion order, which gives the LLM a chronologically coherent stream with gaps where unselected turns fell out.

The score threshold is applied first as an eligibility gate: turns with score below the threshold are invisible to selection entirely. The same applies to groups within layers — a group whose derived score falls below the layer threshold is removed from the layer before group-level selection runs.

### Why insertion order on emission

Selection orders by score because attention is the scarce resource — the highest-scoring content deserves to be in the window. Emission orders by insertion because the LLM reads sequentially and reordering by score destroys the temporal coherence of dialogue. The split is deliberate.

The Conversation rule is the same machinery: selection picks recent-N + top-K, emission walks insertion order. Recent and historical halves interleave naturally because both come from the same group's append history.

## 5. Budget Reconciliation

Reconciliation distributes the available token budget across the schema. The model is borrowed from CSS flexbox: priorities are weights, min and max are bounds.

### Per-pass distribution

Within a parent's budget, each child gets:

```
ideal_share = (child.priority / sum_of_sibling_priorities) * parent_budget
allocated   = clamp(ideal_share, child.min, child.max)
```

After the first pass, summed allocations may not equal the parent's budget — clamping leaves leftover or shortfall. Leftover is redistributed by re-running the share calculation with the remaining budget over the remaining unsaturated siblings (those not at their max). Shortfall (sum of mins exceeds budget) is caught at construction time as a static error where determinable, and at projection time as proportional shrink of mins where the dynamic case demands it.

### The full algorithm

Reconciliation is recursive and iterative:

```
reconcile(node, budget):
  visible_children = [c for c in node.children if not masked(c) and not threshold_filtered(c)]
  if visible_children is empty: return 0
  
  remaining = budget
  for iteration in 0..MAX_ITERATIONS:
    allocations = flexbox_distribute(visible_children, remaining)
    
    consumed = 0
    for child, alloc in allocations:
      if child is leaf-group:
        consumed += run_selection_under_budget(child, alloc)
      else:
        consumed += reconcile(child, alloc)
    
    freed = remaining - consumed
    if freed <= EPSILON: break
    remaining = freed  # released budget reopens selection for under-saturated subtrees
  
  return consumed
```

Released budget from under-consumption (a group whose selection produced less content than its allocation) returns to the *global* pool, not just to siblings — the next iteration redistributes across the whole tree, letting other layers admit content they had previously dropped for capacity reasons.

`MAX_ITERATIONS` is a fixed cap. `EPSILON` provides early exit on convergence.

### Single-turn overflow

A turn whose token count exceeds the budget allocated to its group is dropped from selection. Selection then continues with the next-highest-scored turn. Turns are indivisible.

### Top-k overflow

If the top-k turns selected by score sum to more tokens than the group's allocated budget, trim from the lowest-scored end iteratively until the set fits. Effectively `max` is non-binding — when selection demands the budget, selection wins; if even the single highest-scored turn exceeds the budget, it is dropped.

### Empty groups and layers

A group whose selection returns zero turns (everything threshold-filtered, everything too large to fit, or simply no turns appended yet) is filtered out of its layer before layer-level reconciliation runs. A layer whose groups are all empty is filtered out of the projection. Empty nodes do not consume budget; their min reservations are released.

## 6. Score Formulas

Every layer declares one score formula. The formula composes turn scores into a single group score, used for ordering groups within the layer at emission time and for the eligibility gate.

The closed set, as a strongly-typed enum:

| Formula              | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `max`                | Maximum turn score in the group                                    |
| `sum`                | Sum of all turn scores in the group                                |
| `mean`               | Arithmetic mean of turn scores                                     |
| `top_k_mean(k)`      | Mean of the top-k turn scores (smooths against single outliers)   |
| `count`              | Number of eligible turns (score-independent salience)              |

`max` is the natural default — it matches the substrate semantic where one highly-attended turn elevates its containing group, and it's robust to noise from low-scoring tail content. `top_k_mean` is the smoothed variant for groups where consistency matters more than peaks.

## 7. Projection

A projection is parameterised by a target — `(LayerId, GroupId)` — and a resolver. The target identifies which layer and which group within that layer the projection is *for*. Masking semantics derive from the target.

### Masking

The projection visibility rule:

```
visible(node) = TRUE if:
  node is in a layer with index < target.layer
  OR (node is in target.layer AND node.group_id == target.group)

visible(node) = FALSE if:
  node is in a layer with index > target.layer
  OR (node is in target.layer AND node.group_id != target.group)
```

The system prompt is always visible (it has its own root and its own budget; masking applies only to the layer hierarchy).

Visualised for a target of `(layer: motivational, group: active_mission)`:

```
                                              visible?
SystemPrompt                                   YES (always)
Layer perceptual_ground                        YES (lower than target)
  ├── group type_specialist                    YES
  └── group structure_specialist               YES
Layer motivational                             partial
  ├── group active_mission     [TARGET]        YES
  └── group goal_pressure                      NO  (same layer, sibling)
Layer dialogue                                 NO  (higher than target)
  └── group primary_conversation               NO
```

Masked nodes are excluded from reconciliation entirely. Their declared minimums do not reserve budget; the visible nodes split the full window between themselves. This means the same schema produces structurally different projections depending on the target — the active mission's projection sees its own substrate priming but not its peers; the conversation layer's projection sees the full motivational and perceptual context.

### Output

A projection emits:

```rust
pub struct Projection {
    pub system_prompt: Vec<ResolvedSection>,
    pub turns: Vec<ResolvedTurn>,
}

pub struct ResolvedSection {
    pub id: SectionId,
    // Caller resolves content from id via the schema.
}

pub struct ResolvedTurn {
    pub group: GroupId,
    pub index: TurnIndex,
    // Caller resolves content from (group, index).
}
```

Sections emit in declaration order. Turns emit in the order:

1. All visible layers in declaration order (layer 0 first)
2. Within each layer, groups ordered by descending derived group score
3. Within each group, turns in insertion order

Higher-scored groups appear *later* in the emitted list within their layer — closer to the bottom of the LLM's input, where attention is typically strongest.

## 8. Rust API

### Schema types

```rust
pub struct Schema {
    pub window: WindowSplit,
    pub system_prompt: SystemPromptSchema,
    pub layers: Vec<LayerSchema>,
}

pub struct WindowSplit {
    pub total: usize,
    pub system_prompt: usize,
    pub turns: usize,
}

pub struct SystemPromptSchema {
    pub sections: Vec<SectionSchema>,
}

pub struct SectionSchema {
    pub id: SectionId,
    pub content: String,
    pub budget: Budget,
}

pub struct LayerSchema {
    pub name: String,
    pub description: String,
    pub score_formula: ScoreFormula,
    pub score_threshold: f32,
    pub budget: Budget,
    pub groups: Vec<GroupSchema>,
}

pub struct GroupSchema {
    pub id: GroupId,
    pub selection: SelectionRule,
    pub score_threshold: f32,
    pub budget: Budget,
}

#[derive(Debug, Clone)]
pub struct Budget {
    pub priority: f32,
    pub min_percent: Option<f32>,
    pub max_percent: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum SelectionRule {
    AlwaysVisible,
    TopK { k: usize },
    Single,
    Conversation { recent: usize, historical_top_k: usize },
}

#[derive(Debug, Clone)]
pub enum ScoreFormula {
    Max,
    Sum,
    Mean,
    TopKMean { k: usize },
    Count,
}
```

### Identifiers

```rust
pub struct LayerId(NonZeroU32);
pub struct GroupId(NonZeroU32);
pub struct SectionId(NonZeroU32);
pub struct TurnIndex(u32);  // scoped to a single group, monotonically increasing
```

Identifiers are opaque newtypes. Layers and groups carry human-readable names accessible via the schema for documentation and debugging; identifiers are the addressing mechanism.

### Builder

```rust
pub struct Builder {
    schema: Schema,
    turn_counts: HashMap<GroupId, u32>,  // tracks existence, not content
}

impl Builder {
    pub fn from_yaml(yaml: &str) -> Result<Self, ConstructionError>;
    pub fn from_schema(schema: Schema) -> Result<Self, ConstructionError>;

    /// Append a turn to a group. Returns the index assigned to this turn.
    /// Content storage is the caller's responsibility.
    pub fn append(&mut self, group: GroupId) -> TurnIndex;

    /// Schema accessors for documentation lookup.
    pub fn layer(&self, id: LayerId) -> &LayerSchema;
    pub fn group(&self, id: GroupId) -> &GroupSchema;
    pub fn section(&self, id: SectionId) -> &SectionSchema;

    /// Project the substrate for a given target.
    pub fn project<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
    ) -> Projection;
}

pub struct ProjectionTarget {
    pub layer: LayerId,
    pub group: GroupId,
}
```

### Resolver

```rust
pub trait ContentResolver {
    /// Token count for a turn. Stable per turn (set once at append time on the caller side).
    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize;

    /// Score for a turn. Recomputed per projection (typically BDP-derived against a live query).
    fn turn_score(&self, group: GroupId, index: TurnIndex) -> f32;

    /// Token count for a system prompt section. Stable for the lifetime of the schema.
    fn section_token_count(&self, section: SectionId) -> usize;
}
```

The resolver is the only dynamic input to projection. Section token counts are queried because sections carry static authored text but the crate has no tokenizer; the caller computes once and caches.

## 9. Reconciliation Flow

The full projection flow:

```
project(target, resolver):
  1. Apply target mask  →  visible_layers, visible_groups
  2. Score every visible turn via resolver
  3. Apply group score thresholds  →  eligible_turns per group
  4. Apply selection rules under unbounded budget  →  selected_turns per group
  5. Compute group scores via layer.score_formula(selected_turns)
  6. Apply layer score thresholds  →  surviving groups
  7. Filter out empty groups and empty layers
  8. Reconcile system prompt budget against system prompt sections
  9. Reconcile turn budget across surviving layers, then within each layer across groups
 10. Within each group, run selection under the allocated budget
     (top-k trims from low end if needed; single-turn overflow drops; conversation
      maintains recent-N inviolate, trims historical)
 11. If any group released budget, return to step 9 (capped at MAX_ITERATIONS,
     early exit on convergence within EPSILON)
 12. Emit:
     - system prompt sections in declaration order
     - layers in declaration order, groups within each ordered by descending score,
       turns within each group in insertion order
```

## 10. Validation

### Parse time

- YAML is well-formed
- Selection rule kinds and score formulas are recognised
- Identifiers (group ids, section ids) are unique within their scope
- Numeric fields are well-typed (priorities are positive floats, percentages are 0..=100)

### Construction time

- Sum of declared `min_percent` across siblings of any single parent does not exceed 100 (statically infeasible mins fail early)
- `max_percent >= min_percent` where both are declared
- `score_threshold >= 0`
- `window.system_prompt + window.turns <= window.total`
- Score formula `top_k_mean` has a valid `k` (positive integer)
- Selection rule `top_k` has a valid `k`; `conversation` has valid `recent` and `historical_top_k`

### Projection time

- Nothing — projection is pure given a valid schema and a working resolver. Failures here represent resolver bugs, not user errors.

## 11. Out of Scope

Explicitly not part of this crate:

- **Scoring mechanism.** The Binary Directional Provenance scan, query projection, attention-derived scores — all of that lives in the inference engine and reaches this crate only as a resolver implementation.
- **Tokenization.** Token counts are inputs.
- **Content storage.** Turn and section text live in the caller's structures.
- **Cross-target caching.** A single cognitive cycle may project for several targets in succession (one per layer being run). The naive implementation re-reconciles each call. A caching layer that reuses score evaluations across multi-target calls within a cycle is a future optimisation, not a v1 concern.
- **Multi-conversation arbitration at the dialogue layer.** The substrate may host multiple parallel conversations (different interlocutors); each is its own group, and projection masking already handles isolation between them. No additional machinery needed at the crate level.