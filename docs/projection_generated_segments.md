# Projection Generated Segments — attention-correct boundary K/V via live prefill

> **Status — Design v1, ready to build.** Specifies a structural change to the
> projection engine, scheduler, and substrate: pre-baked sealed K/V for
> *content* (sections, turns), live-prefilled K/V for *structural template*
> tokens (role markers, block envelopes), with a per-slot cache so the
> common case stays free. The change makes attention-pivot K/V correct
> against the actual runtime context instead of a frozen ingest-time
> approximation, eliminates structural tokens from the persistent
> substrate, and lets the schema declare conversation structure
> declaratively against the dialect catalog. Implementation is phased
> (§13); each phase compiles, tests pass, runs end-to-end, and is
> standalone-mergeable.

---

## 1. Abstract

Today's projection emits a flat list of `SectionId` + `(GroupId, TurnIndex)`
references and `apply_projection` Arc-clones their pre-baked `SealedSequence`
K/V onto the slot. Every system-prompt section's K/V is **frozen against the
prefix it was ingested with** at session start. When projection selects a
different combination of sections per turn (different tools materialised,
different turns selected), the K/V of any section whose ingest-time prefix
no longer matches its runtime neighbours is wrong — the model's attention
into those structural-token positions sees a K vector that was conditioned
on a context that isn't actually there. The structural envelope tokens
(`<|im_end|>`, `<tools>`, role markers) are the model's **primary attention
pivots**; bad K/V on those positions degrades response quality
proportionately, on every projection that touches them.

This design replaces the "store structural template K/V in the substrate
and Arc-clone it forever" pattern with **live prefill of template tokens
at projection time**, cached per-slot. Concretely:

1. The projection schema (YAML) declares two kinds of items:
   - `kind: section` — *content*. Authored text. Sealed once at session
     start, pinned in substrate, Arc-cloned per projection.
   - `kind: template` — *structural*. References a `DialectTemplate`
     catalog entry by name. Prefilled live on the slot under its actual
     runtime left context. Never enters the substrate or the redo log.

2. Projection emits `Vec<ProjectionSegment>` where each segment is either
   `Sealed` (substrate-backed, Arc-clone path) or `Generated` (prefilled
   live).

3. A per-slot `SlotProjectionCache` memoises live-prefilled K/V keyed by
   `hash(preceding_tokens ++ run_tokens)`. Cache hits Arc-clone the cached
   sealed run; cache misses submit a `PrefillRun` to the scheduler's
   batched prefill queue.

4. Turn boundaries (`UserStart`, `UserEnd`, `AssistantStart`,
   `AssistantEnd`) become `Generated` segments emitted around each
   selected turn, sourced from the dialect catalog. The current
   role-marker baking in the user-prefill goes away.

5. New user messages become `NewUserMessage` segments captured to a
   slot-attached pending-user-part buffer at prefill time; committed to
   the substrate as one half of a `TurnEntryData` at seal time alongside
   the decoded assistant half.

6. `TurnEntryData` keeps single-record identity (preserving BDP selection
   surface) but splits internally into `user: TurnPart` and `assistant:
   TurnPart`, each with its own residence, tokens, signatures.

7. BDP signatures cover only content — never structural template ranges,
   which were polluting retrieval signal anyway.

Net cost: tens of tokens prefilled per cache-miss projection, batched
across all active slots in the standard prefill loop. Steady-state
(same-conversation reprojection): zero — all generated segments hit the
cache. Net benefit: attention-correct boundary K/V always in R16/F16,
smaller substrate, smaller redo log, cleaner BDP signal, declarative
template structure.

---

## 2. Why this is the right cut

### 2.1 The attention-pivot K/V is what was wrong

Structural tokens (`<|im_end|>\n`, `<tools>\n`, `</tools>\n`, role markers)
carry near-zero semantic content but are **disproportionately attended to**
by every content token downstream. The model uses them as orientation
markers — "here's where the system block closes," "here's where the user
turn begins." The K vectors at those positions encode "I am a closing
marker after the preceding content." If the preceding content the K was
conditioned on doesn't match what's actually there at runtime, the
orientation signal is noisy.

Today's ingest path picks **one** plausible prefix for each structural
section's K/V and freezes it. zend's dialogue layer chooses two:

- `linear_prefix` — cumulative, all sections including all collection
  members.
- `fixed_prefix` — same but excluding collection members and
  `depends_on`-gated sections.

Then `__system_end` is ingested with `fixed_prefix` and `tools_close` is
ingested with `linear_prefix`. Both choices are wrong for any runtime
projection that isn't either "no tools selected" or "all tools selected
at once." A typical projection (3 tools selected from 50) matches neither
ingest-time prefix.

The fix is structural: don't pre-bake K/V for content-sensitive
structural tokens. Generate them live, every time, under their actual
runtime context.

### 2.2 The cost is bounded by what actually changed

A run of generated template tokens — `[Generated_A, Generated_B,
Generated_C]` between two sealed segments — gets one cache key:
`hash(preceding_tokens ++ A.tokens ++ B.tokens ++ C.tokens)`. Two
consecutive projections with the same selected sections produce the same
key for every run → all hits → zero prefill. Projections that differ
only in their tail produce hits up to the divergence point and one cache
miss for each remaining run. The token cost of a miss is the number of
template tokens in that run, which is tiny (single-digit tokens per
boundary, dozens per projection at most).

Across slots, prefill runs batch into one `forward_batched` call. Within a
slot, runs execute in order. The scheduler already does this for
user-prefills and decode steps; generated-run prefills become first-class
participants in the same pipeline.

Steady-state cost: ~zero (all cache hits). Worst case (cold slot, full
projection change): a couple dozen tokens of batched prefill, on the
order of a single decode step.

### 2.3 The storage benefit is incidental but real

Structural template tokens currently consume:
- A `SectionEntryData` per template item, per session.
- A `ResidenceIndex` slot in `Substrate::residence`.
- Redo-log bytes for the `Chunks` + `Tokens` records of each template
  section.
- Warm-tier RAM bytes when pinned.

Under the new model: zero. Templates are a per-process detail of the
inference engine. A daemon restart loads the schema's template fragments
from YAML and prefills them on demand against whatever is in the
substrate. The substrate carries only content.

### 2.4 The format quality is locked in

`Generated` K/V is produced by a fresh prefill pass against live arenas —
R16/F16, the same format the model uses during decode. It never visits
the warm tier, never round-trips through the bg-quantizer, never picks
up Q4_KS/Q8_KS quantization error. Pinned sections do stay R16/F16
today, but the bg-quantizer for warm-tier turns may quantize them; with
templates out of the substrate, that path concerns content only and the
most attention-loaded positions are guaranteed uncompressed.

---

## 3. Data model

### 3.1 `ProjectionSegment`

```rust
pub enum ProjectionSegment {
    /// Pre-baked, substrate-pinned K/V. Arc-cloned onto the slot.
    Sealed(SealedKind),

    /// Live-prefilled structural template tokens. K/V is cached on the
    /// slot; never written to the substrate or redo log.
    Generated {
        tokens: Arc<Vec<u32>>,
        identity: GeneratedIdentity,    // diagnostic; not in cache key
    },

    /// A new user message being submitted this turn. Prefilled onto the
    /// slot and captured into the slot's `pending_user_part` buffer.
    /// Committed to substrate at seal time as the `user` half of the
    /// resulting `TurnEntryData`.
    NewUserMessage { tokens: Arc<Vec<u32>> },
}

pub enum SealedKind {
    Section(SectionId),
    Turn {
        timeline: TimelineId,
        index: TurnIndex,
        part: Role,                     // User | Assistant
    },
}

pub struct GeneratedIdentity {
    /// Schema-level name for diagnostic logging.
    pub name: String,
    /// Where this segment came from in the schema.
    pub schema_position: SchemaPosition,
}
```

A `Projection` is `Vec<ProjectionSegment>` instead of separate
`Vec<ResolvedSection> + Vec<ResolvedTurn>`. Emission order is exactly the
order the segments appear on the slot.

### 3.2 Dialect catalog

```rust
pub enum DialectTemplate {
    SystemStart,
    SystemEnd,
    UserStart,
    UserEnd,
    AssistantStart,
    AssistantEnd,
    ToolBlockOpen,
    ToolBlockClose,
    ToolResponseOpen,
    ToolResponseClose,
    NoThinkPrefix,
}

pub struct Dialect {
    // ... existing atomic fields (user_start, user_end, etc.) ...
    // plus new fields populated to match the catalog enum above.
}

impl Dialect {
    pub fn template(&self, t: DialectTemplate) -> &str { ... }
}
```

The catalog is **atoms only** — no `UserToAssistant` composites. The
schema sequences atoms wherever it needs a transition; prefill-run
concatenation merges adjacent generated segments into one forward pass
at runtime.

Each model's dialect module (`qwen3.rs`, `qwen3_moe.rs`, `hermes3.rs`,
`qwen2.rs`) gets the new fields populated. Adding a new model requires
populating the full catalog or the builder construction fails with
`ConstructionError::DialectMissingTemplate { template }`.

### 3.3 YAML grammar

```yaml
layers:
  - name: dialogue
    window: 16000
    system_prompt:
      items:
        - kind: template
          id: system_open
          dialect: system_start

        - kind: template
          id: no_think
          dialect: no_think_prefix

        - kind: section
          id: frame
          content: |+
            You are a senior engineer working alongside the developer ...

        # ... more content sections ...

        - kind: template
          id: tools_open
          dialect: tool_block_open
          depends_on: tools

        - kind: collection
          name: tools
          selection: { kind: top_k, k: 5 }
          sections: []

        - kind: template
          id: tools_close
          dialect: tool_block_close
          depends_on: tools

        - kind: template
          id: system_close
          dialect: system_end
```

Item kinds:

| Kind | Behaviour | K/V provenance |
|---|---|---|
| `section` | Authored content. Cumulative-ingested at session start, pinned. | Substrate `SectionEntryData`. Arc-cloned per projection. |
| `template` | Dialect catalog reference. Resolved to tokens at builder time. | None at session start. Prefilled live at projection time, cached on slot. |
| `collection` | Bucket of `section`-kind members with a selection rule. | Members are sections. |

`depends_on: <collection_name>` works on either kind — it gates emission
at projection time based on whether the named collection materialised
≥ 1 member. Independent of the section-vs-template question.

Empty-content templates (e.g. `NoThinkPrefix` when thinking is enabled)
are filtered at build time, not projection time. The schema item is
dropped during construction.

`__system_start` / `__system_end` synthetic-section mechanism dies — the
dialect-aware caller (zend) declares them as ordinary `kind: template`
items at the head and tail of every layer's `system_prompt.items` via a
builder helper. The schema becomes self-describing.

### 3.4 `TurnEntryData` per-role split

```rust
pub struct TurnEntryData {
    block_range: (u64, u64),    // overall slot extent across both parts
    user: TurnPart,
    assistant: TurnPart,
}

pub struct TurnPart {
    text: String,
    token_count: usize,
    token_ids: TokenBuffer,
    sig_entries: Vec<SigEntry>,  // BDP sigs for this part's content tokens
    residence: ResidenceIndex,
}
```

Required fields, both halves present. A `TurnEntryData` is never written
to the substrate until both halves exist. There is no "half-turn" state
the substrate has to know about — submit_turn captures the user half to a
slot-local buffer; finish_turn assembles both halves and writes one atomic
record.

`Role` stays `User | Assistant | System` (System is currently unused for
turn records but retained for the role-marker handling that already
threads through).

BDP scoring is at turn granularity. `Substrate::turn_score(timeline,
index, formula, weights)` aggregates the per-part sig material:
concatenates `user.sig_entries` and `assistant.sig_entries`, scores once.
Selection rules (`Sequence { recent, historical_top_k }`) operate on whole
turns, unchanged.

Two residence slots per turn — one for each part. Independent in
principle, elevated/evicted together in practice (no use case for one
without the other). The existing `Substrate::snapshot_promotion_state`
classifier handles per-residence promotion plans without modification.

### 3.5 Cache identity

A **prefill run** is the unit cached. It corresponds to one
`forward_batched` invocation on the slot. A run contains the tokens of
all adjacent `Generated` segments between two `Sealed` segments (or
between a `Sealed` and the end of the projection). A `NewUserMessage`
segment is **its own run** — never combined with neighbouring
`Generated`s into a single cached entry, because its capture target
differs.

Cache key for a generated run:

```text
key = hash(preceding_tokens_in_projection ++ run_tokens)
```

`preceding_tokens_in_projection` is the concatenation of every prior
segment's tokens. Hashed incrementally via a rolling `AHasher` advanced
as the assembler walks the segment list. `run_tokens` is the
concatenation of the run's `Generated` segment token vectors.

Same preceding context AND same run tokens → same key → cache hit. Any
difference in either → miss.

Collision probability with `u64` ahash at any realistic slot lifetime is
vanishingly small. Treated as zero.

---

## 4. Projection engine

### 4.1 Output

`Builder::project` returns `Projection { segments: Vec<ProjectionSegment>
}`. The 12-step pipeline in `project.rs` is unchanged through step 11
(scoring, selection, reconcile, budgeting). Step 12 (emit) changes shape:

```text
Step 12 (new):
  out = Vec::new()
  for item in target_layer.system_prompt.items in declaration order:
      match item:
          Section(s):
              if s.depends_on satisfied:
                  out.push(Sealed(Section(s.id)))
          Template(t):
              if t.depends_on satisfied AND t.tokens not empty:
                  out.push(Generated { tokens: t.tokens.clone(), identity: ... })
          Collection(c):
              for member in selected_members(c) in declaration order:
                  out.push(Sealed(Section(member.id)))
  for layer in surviving_layers (by ascending group score):
      for turn in selected_turns_of_layer:
          out.extend(expand_turn(turn))
  return Projection { segments: out }

expand_turn(turn) -> Vec<ProjectionSegment>:
    [
        Generated(dialect.template(UserStart)),
        Sealed(Turn { timeline, index, part: User }),
        Generated(dialect.template(UserEnd)),
        Generated(dialect.template(AssistantStart)),
        Sealed(Turn { timeline, index, part: Assistant }),
        Generated(dialect.template(AssistantEnd)),
    ]
```

`NewUserMessage` is **not** emitted by `Builder::project`. It is appended
by the scheduler's submit-turn handler after the projection list is
assembled, as part of the user-prefill staging step (§5).

### 4.2 `depends_on` semantics unchanged

A schema item (section or template) with `depends_on: <collection_name>`
is emitted iff the named collection's selection rule materialised ≥ 1
member in this projection. This evaluation happens during the emit walk —
the collection's results are computed in a first pass over items, then
the second pass walks items in declaration order applying the predicate.

This carries over from today's `emit_system_prompt_items` two-pass walk
unchanged.

### 4.3 Empty-token templates

Resolved at builder time, not projection time. If `dialect:
no_think_prefix` resolves to an empty string for the current dialect (or
the model is configured `suppress_thinking: false`), the template item
is dropped from the schema during construction. Projection never sees it.

This avoids per-projection empty-tokens checks and keeps the segment list
free of no-op entries.

---

## 5. `ProjectionAssembler`

Lives in `candle-conversation/src/scheduler/projection_assembler.rs`,
extracted out of `Scheduler::apply_projection`'s current ~300-line
inline implementation.

### 5.1 Inputs

```rust
pub struct ProjectionAssembler<'s> {
    sequence_id: SequenceId,
    conversation: &'s Conversation,
    session: &'s mut BatchedInferenceSession,
    slot_state: &'s mut SlotState,    // includes SlotProjectionCache,
                                       // pending_user_part, previous segments
    prefill_queue: &'s mut PrefillQueue,
}

impl ProjectionAssembler<'_> {
    pub fn apply(
        &mut self,
        new_segments: Vec<ProjectionSegment>,
        new_user_message: Option<Arc<Vec<u32>>>,
    ) -> Result<()> { ... }
}
```

### 5.2 Walk

```text
1.  Snapshot the slot's writer tail (in-flight decode chunks) as a
    `WriterTail`. The tail is preserved across LCP truncate and
    re-attached at the end.

2.  Compute LCP between `slot_state.previous_segments` and
    `new_segments` by segment-identity equality:
      - Sealed(SealedKind) → equal iff SealedKind equal.
      - Generated { tokens, identity } → equal iff identity equal AND
        rolling-hash-at-this-position equal between old and new walks.
      - (Both halves of the rolling hash agree iff every preceding
        segment was identical, which the LCP comparison enforces by
        construction.)
    LCP = N means the first N segments of the new projection are
    already on the slot in the right form.

3.  Truncate slot to the block boundary at end-of-LCP. Drops Arc refs
    for everything past N; residence chunks beyond LCP return to the
    pool.

4.  Walk new_segments[N..]:
      rolling_hash = (cached at end-of-LCP from previous walk)
      current_run = empty
      for segment in new_segments[N..]:
          match segment:
              Sealed(kind):
                  flush_current_run()      // see below
                  resolve sealed bytes from substrate; inject onto slot
                  fold segment.tokens (looked up from substrate) into
                      rolling_hash
              Generated { tokens, identity }:
                  fold tokens into rolling_hash
                  current_run.tokens.extend(tokens)
                  current_run.identities.push(identity)
              NewUserMessage { tokens }:
                  flush_current_run()
                  fold tokens into rolling_hash
                  enqueue PrefillRun {
                      sequence_id: self.sequence_id,
                      tokens,
                      write_offset: slot.current_block_count,
                      capture: PendingUserPart,
                  }
                  advance slot.current_block_count by ceil(len(tokens) / CHUNK_SIZE)
      flush_current_run()

    flush_current_run():
        if current_run is empty: return
        key = rolling_hash               // includes the run's tokens (just folded)
        if slot_state.cache.contains_key(key):
            inject cache[key] onto slot
        else:
            enqueue PrefillRun {
                sequence_id: self.sequence_id,
                tokens: current_run.tokens,
                write_offset: slot.current_block_count,
                capture: SlotCache { key },
            }
            advance slot.current_block_count
        current_run = empty

5.  Restore writer tail at slot.current_block_count + queued-prefill-blocks.
    (The writer tail attaches AFTER all pending prefills resolve; the
    scheduler's prefill-batcher handles the ordering.)

6.  Update slot_state.previous_segments = new_segments.
```

### 5.3 Prefill-run capture targets

```rust
pub enum RunCapture {
    /// Generated template run — captured to slot.cache under `key`.
    SlotCache { key: u64 },

    /// New user message run — captured to slot.pending_user_part. Not
    /// cached. Committed to substrate as `TurnEntryData.user` at seal.
    PendingUserPart,
}
```

Capture happens in the scheduler's prefill-completion hook (§6). The
captured `Vec<SealedSequence>` is one entry per layer; the assembler
treats it identically to a substrate-sourced sealed sequence for the
purposes of Arc-cloning.

### 5.4 Cache lifetime and growth

`SlotProjectionCache::memo` is `AHashMap<u64, Vec<SealedSequence>>`. It
lives on the slot — populated as runs miss, queried as runs hit, dropped
entirely when the slot is freed. No eviction policy. The Arc refs inside
the cached `SealedSequence`s keep their arena chunks alive; dropping the
cache releases them.

In long-running slots the cache grows monotonically with the number of
unique `(preceding_context, run_tokens)` combinations the conversation
has produced. Typical bound: O(turns × structural-variations-per-turn).
For zend's dialogue layer with 5 tools and ~10 templates per projection,
this is a few hundred entries over a long conversation. Each entry holds
a small number of layers × small SealedSequences = on the order of
kilobytes per entry. Total: small megabytes. Acceptable.

---

## 6. Scheduler integration

### 6.1 `PrefillRun`

```rust
pub struct PrefillRun {
    pub sequence_id: SequenceId,
    pub tokens: Arc<Vec<u32>>,
    pub write_offset: BlockOffset,
    pub capture: RunCapture,
}
```

Peer to `ActivePrefill`. Both flow through the same prefill batcher,
both participate in `forward_batched` calls.

### 6.2 Queueing and ordering

Within a slot, prefills run in submitted order. The assembler submits
runs in segment-list order (mixed `SlotCache` and `PendingUserPart`
captures). The scheduler does NOT reorder within a slot.

Across slots, runs batch freely. The standard `run_one_prefill_chunk`
loop selects the next-ready prefill per slot, builds a batched
`forward_batched` call, and dispatches.

Adjacent prefill runs on the **same slot** with **compatible captures**
(both `SlotCache`, or `SlotCache` followed by `PendingUserPart`) are
combined into a single forward pass for throughput. The capture
machinery slices the produced K/V at sub-range boundaries to deposit
each segment's bytes in the right place. (Two `PendingUserPart` captures
on the same slot don't combine because each one is a distinct user
message — but per turn there's exactly one user message, so this never
arises in practice.)

### 6.3 Capture hook

When a `PrefillRun` completes its forward pass:

```text
let captured = extract_sealed_sequence_from_slot(
    slot, write_offset, write_offset + ceil(len(tokens) / CHUNK_SIZE)
);

match run.capture:
    SlotCache { key }:
        slot_state.cache.insert(key, captured);
    PendingUserPart:
        slot_state.pending_user_part = Some(captured);
```

The captured `Vec<SealedSequence>` is one per layer, sourced from the
slot's existing per-layer state. The Arc refs in the SealedChunks keep
the arena chunks alive even if the slot's prefix is later rebuilt past
them.

### 6.4 Error propagation

A failed forward pass marks the `PrefillRun` as errored; the slot's
turn-in-flight state propagates the error to the `TurnEvent` channel
and the slot's subsequent prefills are dropped (cascade abort). Any
runs that completed successfully before the failure leave behind valid
cache entries — they're keyed deterministically, so they're not
"corrupted," just incomplete coverage. The slot's pending-user-part is
**not** populated on cascade failure; finish_turn sees no half-turn
state.

### 6.5 Decode integration

Decode starts on the slot once all queued prefills (projection-prefills
+ user-prefill) have completed. The slot's `ActiveDecode` is initialised
with the writer tail attached at the position immediately after the last
prefill range — i.e., after `Generated(AssistantStart)` in the submit-turn
expansion.

---

## 7. Slot state

```rust
pub struct SlotState {
    pub cache: SlotProjectionCache,
    pub pending_user_part: Option<Vec<SealedSequence>>,
    pub previous_segments: Vec<ProjectionSegment>,
    pub previous_segment_hashes: Vec<u64>,    // rolling hash at end of each segment
}
```

`previous_segment_hashes[i]` is the rolling hash at the end of segment
`i`. Stored so LCP comparison can validate that two segments at the same
position have the same preceding context — i.e., the same effective K/V
input — without re-walking from scratch.

`pending_user_part` is populated by a `NewUserMessage` capture and
cleared at seal time when the resulting bytes are written to the
substrate. If the turn is aborted (decode error, EOS forced before seal,
slot freed mid-flight), the buffer drops without persistence.

`previous_segments` is the assembler's record of the slot's current
shape. Replaces the implicit "the slot has system blocks 0..N, then
turn-injected blocks N..M" tracking today.

---

## 8. Substrate / persistence consequences

### 8.1 Substrate

- `Substrate::sections` shrinks: no more `__system_start`,
  `__system_end`, `tools_open`, `tools_close`, or any other
  template-kind item.
- `SectionEntryData` stores authored content sections only.
- `TurnEntryData` splits internals into `user: TurnPart` + `assistant:
  TurnPart`, each with `residence: ResidenceIndex`.
- `Substrate::append_full` becomes `Substrate::append_complete(timeline,
  user_part, assistant_part)`. Both residences allocated; both filled
  from the seal-time data.
- `Substrate::turn_token_count(group, index)` sums
  `user.token_count + assistant.token_count + (role-marker
  tokens contributed by Generated boundaries in projection)`. The
  role-marker contribution is computed from the dialect catalog — not
  stored, derived.
- BDP scoring (`turn_score`) concatenates `user.sig_entries` and
  `assistant.sig_entries` before running the formula.

### 8.2 Persistence (redo log)

- `TurnDecl` carries two `(block_start, block_end)` ranges: one for the
  user part, one for the assistant part. Both required.
- Per-turn `Chunks` stream carries chunks for both parts. The simplest
  layout: chunks ordered (user part first, then assistant part). A
  `(user_chunk_count, assistant_chunk_count)` pair in TurnDecl
  disambiguates the split. Reload walks both regions in order.
- Per-turn `Tokens` record carries the concatenation of `user.token_ids
  ++ assistant.token_ids` with the split point recorded in TurnDecl
  (matching the chunk count partition).
- Per-turn `Signatures` record similarly split with the partition
  recorded.
- No template-section records. The redo log shrinks for every session.

### 8.3 Reload

`SubstrateReload::recover_turn` reconstructs both halves of each
`TurnEntryData` from the persisted records. Cold-marker turns (chunks
not yet landed when the daemon shut down) materialise both `TurnPart`s
with empty `Vec<SealedSequence>` cold residences until elevation
recovers them.

### 8.4 BDP signature scope

The sig extractor (`extract_prov_after_step` and the per-block prefill
sig capture) gains a "this block is structural" flag. Sigs are skipped
for any block range captured to `SlotCache`. Sigs are extracted for any
block range captured to `PendingUserPart` (user content) or written by
decode (assistant content) — i.e., everything destined for the substrate.

The persisted `Signatures` records carry only content sigs. The BDP
scanner's per-turn corpus becomes:

```text
per-turn corpus = user content tokens ++ assistant content tokens
```

— no role markers, no structural framing. Retrieval ranking improves
because the high-self-similarity noise from role-marker positions is
gone.

---

## 9. YAML migration (zend's `projection.yaml`)

Before (current):

```yaml
- name: dialogue
  ...
  system_prompt:
    items:
      - kind: section
        id: mode
        content: |+
          /no_think
      - kind: section
        id: frame
        content: |+
          You are a senior engineer ...
      # ... more sections ...
      - kind: section
        id: tools_open
        content: "<tools>\n"
        depends_on: tools
      - kind: collection
        name: tools
        sections: []
      - kind: section
        id: tools_close
        content: "</tools>\n"
        depends_on: tools
```

(With `__system_start` / `__system_end` installed programmatically via
`Builder::set_system_markers` outside the YAML.)

After (Phase 6):

```yaml
- name: dialogue
  ...
  system_prompt:
    items:
      - kind: template
        id: system_open
        dialect: system_start
      - kind: template
        id: no_think
        dialect: no_think_prefix
      - kind: section
        id: frame
        content: |+
          You are a senior engineer ...
      # ... more sections (frame, history_stance, grounding, tools_overview) ...
      - kind: template
        id: tools_open
        dialect: tool_block_open
        depends_on: tools
      - kind: collection
        name: tools
        selection: { kind: top_k, k: 5 }
        score_threshold: 140.70
        depth_weights: { syntactic: 0.0, semantic: 0.0, pragmatic: 1.0 }
        sections: []
      - kind: template
        id: tools_close
        dialect: tool_block_close
        depends_on: tools
      - kind: template
        id: system_close
        dialect: system_end
```

`mode` (which is literally `/no_think\n`) becomes a template referencing
`no_think_prefix`. zend no longer calls `Builder::set_system_markers` —
the schema author declares the boundaries.

---

## 10. Cutover

Clean break, no backwards compatibility. The user owns rebuilding the
redo log on first boot after each phase that changes the persistence
format (Phases 4, 5). Phases 1, 2, 3, 6 are persistence-compatible with
the prior phase's redo log.

Old conversations from before this change are not migrated. The redo log
is wiped between major schema versions during this work.

---

## 11. What this does NOT change

- The projection engine's pipeline (mask → score → select → reconcile)
  is unchanged. Only the emit step changes shape.
- BDP scanner, scoring formulas, depth weights, calibration: unchanged.
  The scope of what gets BDP'd shrinks (content only) but the scoring
  math is identical.
- Substrate's tier transitions (hot/warm/cold), LRU policy, eviction:
  unchanged.
- The cold-load pipeline: unchanged. Turns load the same way; their
  internal user/assistant split is invisible to the cold-load code
  except via the persistence record shape.
- Mid-decode reprojection cadence and triggers: unchanged.
- ChunkedKvBacking: unchanged. New segment types map to the same
  Arc-clone / inject_sealed_at_tail / split-off-tail primitives.

---

## 12. What this DOES change

Module-level diff summary:

| Module | Change |
|---|---|
| `candle-conversation/src/models/dialect.rs` | `DialectTemplate` enum, new dialect fields. |
| `candle-conversation/src/models/qwen3.rs`, etc. | Populate new dialect fields per model. |
| `candle-conversation/src/projection/schema.rs` | `SystemPromptItem::Template` variant, no synthetic-marker fields on `LayerSchema`. |
| `candle-conversation/src/projection/yaml.rs` | Parse `kind: template`, resolve `dialect:` refs at build. |
| `candle-conversation/src/projection/project.rs` | Emit `Vec<ProjectionSegment>`, turn expansion, depends_on unchanged. |
| `candle-conversation/src/projection/builder.rs` | `Builder::set_system_markers` deleted; helper for dialect-aware caller to inject template items. |
| `candle-conversation/src/scheduler/mod.rs` | `apply_projection` becomes thin wrapper around `ProjectionAssembler`. `SlotState` gains `cache`, `pending_user_part`, `previous_segments`. |
| `candle-conversation/src/scheduler/projection_assembler.rs` | **NEW.** Walk logic, LCP, rolling hash, prefill-run emission. |
| `candle-conversation/src/scheduler/prefill.rs` | `PrefillRun` peer to `ActivePrefill`. Combined-run forward-pass logic. Capture hook. |
| `candle-conversation/src/substrate.rs` | `TurnEntryData` split, `TurnPart`, `append_complete`. |
| `candle-conversation/src/persistence/record.rs`, `streams.rs`, `resume.rs` | `TurnDecl` two-range layout, `Chunks`/`Tokens`/`Signatures` split. |
| `candle-conversation/src/provenance/scan.rs` | Skip-structural-blocks flag for sig extraction. |
| `candle-conversation/src/conversation.rs` | Drop role-marker formatting from `submit_turn`. Drop synthetic-section ingest dance from `new`. |
| `zend/src/prompts/projection.yaml` | Migrate `__system_*`, `tools_*` to `kind: template`. |

---

## 13. Implementation phases

Each phase compiles, tests pass, the daemon boots, a conversation
completes end-to-end. Each ends in a commit. The redo log is rebuilt
between phases that change its format (Phases 4, 5).

### Phase 1 — Dialect catalog + YAML `kind: template` parsing

**Pure addition. No runtime behaviour change.**

Deliverables:
- `DialectTemplate` enum + new `Dialect` fields.
- Per-model dialect populations for Qwen3, Qwen3-MoE, Hermes3, Qwen2.
- YAML parser accepts `kind: template` with `dialect:` reference;
  unknown references error at parse.
- Builder resolves references to tokens at construction.
- `kind: template` items materialise as `SectionSchema` with a
  `is_template: true` marker. Treated identically to `kind: section`
  downstream — ingested at session start, pinned, Arc-cloned per
  projection.

Tests:
- Parsing: round-trip on every template kind. Unknown dialect name →
  `ConstructionError::UnknownDialectTemplate`. Empty content (e.g.
  no_think_prefix when disabled) is filtered.
- Dialect lookup: every variant resolves to non-empty for every model
  unless that model legitimately lacks it.
- Schema construction: template items present with correct tokens.
- Integration: zend conversation flow unchanged.

Risk: **low.** Additive. Existing tests stay green.

### Phase 2 — `ProjectionSegment` + `ProjectionAssembler` + LCP

**Behaviour-preserving refactor. Bit-identical K/V on slot. Measurable
reproject perf gain.**

Deliverables:
- `ProjectionSegment` enum added (`Sealed` and `Generated` variants;
  `NewUserMessage` added but unused until Phase 5).
- `Builder::project` returns `Projection { segments: Vec<ProjectionSegment>
  }`. All segments emit as `Sealed` at this phase, including template
  items (still routed through substrate ingest).
- `ProjectionAssembler` module extracted from
  `Scheduler::apply_projection`.
- LCP comparison + truncate-to-divergence-point. Behaviour identical
  when LCP = 0 (truncate-to-zero, full rebuild).
- `SlotState::previous_segments` and `previous_segment_hashes` added,
  populated, consulted.

Tests:
- Bit-identical: scripted conversation transcript, run with full-rebuild
  vs LCP path, per-layer SealedSequence bytes match.
- LCP coverage: no-change, prefix-only-change, mid-change cases.
- Performance: `apply_projection` time on same-projection reproject
  should drop from O(slot blocks) to O(LCP comparison).
- All existing scheduler tests pass.

Risk: **medium.** Output-shape change touches many callsites. LCP
correctness needs coverage. No quality-impact risk yet.

### Phase 3 — Slot cache + `Generated` for system-prompt + `PrefillRun`

**Behaviour change. System-prompt boundary K/V becomes attention-correct.**

Deliverables:
- `SlotProjectionCache` added to `SlotState`.
- `PrefillRun` added to scheduler, integrated into the prefill batcher.
- Adjacent runs on the same slot concatenate into one forward pass.
- Cross-slot batching as today.
- Capture hook: completed `PrefillRun` deposits captured K/V into
  `SlotProjectionCache` under `key`.
- `Builder::project` emits `Generated` segments for `kind: template`
  items. `Sealed` for `kind: section` (and turn) items.
- Template items NO LONGER ingested as sections at session setup.
- `conversation.rs` ingest loop drops template handling for system-prompt
  templates (content sections still ingest cumulatively).
- `__system_start` / `__system_end` synthetic-section mechanism retired.
  Dialect-aware caller (zend) installs them as `kind: template` items
  via a builder helper.
- Error in a `PrefillRun` cascades to turn abort.

Tests:
- Cache hit/miss: same `(preceding, run)` → byte-identical K/V on hit;
  miss-then-hit on second projection.
- K/V correctness under varying preceding: prefill produces correct K/V
  under different sealed-prefix combinations.
- Multi-slot batching: scenarios where N slots each have a generated
  run pending → one `forward_batched` invocation.
- Substrate residence count: drops by the number of template items per
  session.
- End-to-end quality smoke: fixed conversation script, eyeball assistant
  outputs for coherence.
- Mid-decode reproject: cache hits on unchanged tail, misses on changed
  tail.

Risk: **high.** Big behavioural change. Prefill batching must be exact
per-slot. Cache key correctness is load-bearing.

### Phase 4 — `TurnEntryData` per-role split storage

**Storage refactor. No projection or behavioural change. Scaffolding for
Phase 5.**

Deliverables:
- `TurnEntryData { user: TurnPart, assistant: TurnPart }`.
- `TurnPart { text, token_count, token_ids, sig_entries, residence }`.
- `Substrate::append_complete(timeline, user_part, assistant_part)`
  replaces `append_full`.
- Persistence: `TurnDecl` two-range layout. `Chunks` / `Tokens` /
  `Signatures` records split with partition counts in `TurnDecl`.
- Reload reconstructs both parts.
- At Phase 4 entry, the user `TurnPart` is empty; the assistant
  `TurnPart` carries today's full turn content (formatted prefill +
  decoded tokens + post-decode tokens, role markers and all). Existing
  behaviour preserved.
- `SealedKind::Turn` gains a `part: Role` field (only `Assistant` used
  at this phase).
- BDP scoring aggregates both parts' sigs (assistant-only at this phase).

Tests:
- Round-trip: append, read back, both parts present (one empty).
- Reload: `reconstruct_from_log` restores both parts correctly.
- Persistence record format: golden bytes for new TurnDecl + Chunks
  layout.
- BDP scoring matches Phase 3 baseline.
- Conversation flows: identical output text.

Risk: **medium.** Substrate format change. Reload is the main gate.
Redo log rebuilt on first boot.

### Phase 5 — `NewUserMessage` + pending-user-part + turn-boundary `Generated`s

**Behaviour change. Turn-boundary attention pivots become correct.**

Deliverables:
- `ProjectionSegment::NewUserMessage { tokens }` actively emitted by the
  scheduler's submit-turn handler at the projection-list tail.
- `RunCapture::PendingUserPart` capture target.
- `SlotState::pending_user_part: Option<Vec<SealedSequence>>` populated
  by `NewUserMessage` capture, cleared at seal.
- Turn expansion in `Builder::project`: each turn emits
  `[Generated(UserStart), Sealed::Turn { part: User }, Generated(UserEnd),
  Generated(AssistantStart), Sealed::Turn { part: Assistant },
  Generated(AssistantEnd)]`.
- Seal flow: `pending_user_part` + decoded slot range →
  `Substrate::append_complete` with both `TurnPart`s populated.
- Mid-decode reproject: `pending_user_part` survives LCP truncate;
  assembler re-injects it like a sealed segment after the new prefix is
  assembled.
- `conversation.rs` `submit_turn` formatting reduced to user-message
  tokens only (role markers move to projection).

Tests:
- End-to-end: submit_turn → decode → seal → reload → replay; output
  text coherent.
- Mid-decode reproject: trigger reproject during decode;
  `pending_user_part` survives; turn completes.
- Crash mid-decode: no substrate state for the in-flight turn; reload
  sees no orphan.
- Multi-turn: turn N attends correctly to turn N-1.
- Persistence: `TurnDecl` + both `TurnPart`s written atomically at seal.

Risk: **high.** Scheduler hot path. Pending-user-part interacts with
reproject and seal. Redo log rebuilt.

### Phase 6 — BDP scope + YAML migration + cleanup

**Consolidation. Signal-quality improvement. Code-debt removal.**

Deliverables:
- BDP sig extractor skips blocks captured to `SlotCache`. Sigs only
  cover `PendingUserPart` and decode-time content captures.
- zend's `projection.yaml` migrated: `__system_start`, `__system_end`,
  `tools_open`, `tools_close`, `mode` → `kind: template`.
- Delete `LayerSchema::system_start_section`, `system_end_section`,
  `Builder::set_system_markers`, `next_section_id_raw` synthetic-id
  machinery.
- Delete the `linear_prefix` / `fixed_prefix` cumulative ingest dance
  in `conversation.rs` for template handling; content sections still
  cumulative-ingest.

Tests:
- BDP retrieval MRR / Top-1 against canonical probe set: equal or
  better than Phase 3 baseline.
- YAML migration: zend boots, ingest completes, conversation works.
- Integration suite green.

Risk: **low.** Mostly mechanical. BDP scope change has measurable
signal-quality outcome.

---

## 14. Open issues

None blocking. Detail-level questions to settle during implementation:

- **Chunk-aligned user/assistant boundary.** The seal-time split between
  user content and assistant content lands at whatever block the
  prefill happened to finish on. Since user-message length is
  arbitrary, the split is generally not chunk-aligned. The two
  `TurnPart`s either pay padding bytes at the boundary or share a
  partial chunk via Arc semantics. The existing chunked-cache supports
  partial chunks; reuse that. Concrete mechanism worked out in Phase 4.

- **Reproject and pending_user_part.** When reprojection mid-decode
  rebuilds the prefix, `pending_user_part` must be re-injected after the
  new prefix is assembled. It sits **outside** the segment list (it's
  slot-attached state). The assembler's `apply` method takes both
  `new_segments` and a flag indicating whether to re-inject the pending
  user part. Worked out in Phase 5.

- **PrefillRun granularity for batch-throughput.** The single-slot
  concatenation rule says "adjacent runs on the same slot combine into
  one forward pass." For very long generated runs the per-pass cap
  (`max_prefill_chunk`) still applies — the run executes as multiple
  chunks through the same forward batcher. Capture hooks fire after the
  **final** chunk of a run completes. Standard prefill-batcher
  mechanics.

- **Cache key cross-conversation reuse.** Keys are deterministic across
  slots since they're hashes of token sequences. In principle a cache
  shared across slots in the same workspace could be useful. Out of
  scope for v1; revisit if profiling shows the per-slot cache misses
  are concentrated on common prefixes.

---

## 15. Cross-phase invariants

Every phase must hold:

1. `apply_projection` produces a slot whose per-layer K/V is correct
   under causal attention. No stale chunks, no missing role markers,
   no double-injection.
2. Mid-decode reproject preserves the writer tail.
3. Seal writes atomic substrate records. Crash mid-flight leaves no
   partial turn record.
4. The existing test suite stays green between phases.
5. The daemon boots, takes a turn, decodes, seals, reloads.

Phases that violate any of these are not landable.
