# Projection Generated Segments — attention-correct boundary K/V via live prefill

> **Status — Phases 1–6 shipped.** Phase 5 landed in a simplified shape
> and Phase 6's deliverables were absorbed into the earlier phases as
> they shipped — no separate Phase 6 commit was needed.  Specifies a
> structural change to the projection engine,
> scheduler, and substrate: pre-baked sealed K/V for *content* (sections,
> turns), live-prefilled K/V for *structural template* tokens (system
> framing, inter-turn role markers), with a per-slot cache so the common
> case stays free. The change makes attention-pivot K/V correct against
> the actual runtime context instead of a frozen ingest-time approximation,
> removes inter-turn role markers from the persistent substrate, and lets
> the schema declare conversation structure declaratively against the
> dialect catalog.
>
> **Implementation deviation from the original design (recorded here for
> future readers).**  The shipped Phase 5 is leaner than this doc's
> original spec.  Specifically:
>
> - Only the **inter-turn** boundary markers (`user_start` at a turn's head,
>   `assistant_end` at its tail) regenerate as `Generated` segments at
>   projection time.  The **intra-turn** pair (`user_end` +
>   `assistant_start`, plus an optional `/think_block`) stays baked in the
>   persisted prefill bytes — its K/V context is dominated by the turn's
>   own invariant content on both sides, so re-prefilling on every
>   projection would burn compute for no attention-correctness gain.
> - `TurnEntryData` is a single `TurnPart` carrying both `user_text` and
>   `assistant_text` as verbatim strings (not tokens, not K/V halves).
>   The K/V is one indivisible sealed block per turn.
> - `ProjectionSegment::NewUserMessage` and `SlotState::pending_user_part`
>   are retained as forward-looking infrastructure but no current code
>   path emits or populates them.  The current turn's user message
>   prefills as ordinary `prefill_tokens` (the bytes the substrate seals
>   anyway), not as a captured `NewUserMessage` run.
> - `BoundaryMarkers` lives on the scheduler and is threaded through the
>   assembler's `ApplyContext` — **not** on the `Schema`.  Dialect-specific
>   tokens are a runtime concern, not a schema concern.
> - `SlotProjectionCache::retain_keys()` fires at end-of-turn against
>   the working-set of cache keys the most recent `apply_segments` walk
>   actually touched.  Entries from prior projections that fell out of
>   the current window (or that lived under a now-divergent prefix) are
>   dropped; entries the new projection still depends on stay hot.
>   Without this trim the cache grows monotonically per turn over a
>   long conversation.
>
> Sections below describe the **as-built** shape; the original phased
> deliverable list in §13 is kept with a status annotation per phase.

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

5. **(Not shipped — reserved infrastructure.)**  The `NewUserMessage`
   segment variant exists in `ProjectionSegment` as a forward-looking
   path for a future feature (mid-turn speculative capture, parallel
   user-message ingestion).  In the current shipped path the user
   message prefills as ordinary `prefill_tokens` and lands directly in
   the slot's writable region ahead of decode.

6. `TurnEntryData` holds a single `TurnPart` carrying `user_text` and
   `assistant_text` as verbatim strings, plus a single combined K/V
   block.  The doc's original per-role internal split was implemented in
   Phase 4 and walked back during the Phase 5 simplification — see
   commit `44f915c7`.

7. BDP signatures cover only content — never the boundary-marker ranges
   that live as `Generated` segments (they're never persisted, so
   they're never sig'd).  This is a structural property of the cutover,
   not a separate filter.

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
`hash(preceding_tokens ++ A.tokens ++ B.tokens ++ C.tokens)`.  Two
consecutive projections with the same selected sections produce the
same rolling hash at every run boundary, so every run's key matches an
existing cache entry → all hits → zero prefill work on the GPU.
Projections that differ produce hits up to the first divergent token
and cache misses for every run downstream of it; the token cost of a
miss is the number of template tokens in that run, which is tiny
(single-digit tokens per boundary, dozens per projection at most).

Sealed segments are still re-resolved from substrate on every apply.
That cost is metadata-only — Arc-clone the per-layer sealed sequences
and patch `block_range` entries — measured in microseconds for a
typical projection.  The GPU forward pass for live-prefilled templates
is the cost that scales badly without the cache; the metadata cost
isn't.

Across slots, prefill runs batch into one `forward_batched` call.
Within a slot, runs execute in order.  The scheduler already does this
for user-prefills and decode steps; generated-run prefills become
first-class participants in the same pipeline.

Steady-state cost: ~zero (all cache hits).  Worst case (cold slot, full
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

    /// **Reserved infrastructure — not currently emitted.**  Originally
    /// designed as the projection-side handle for a new user message
    /// being captured for substrate commit at seal.  The shipped path
    /// prefills the user message as ordinary `prefill_tokens` instead.
    /// Retained for a future feature (mid-turn capture, speculative
    /// decoding, parallel ingestion).
    NewUserMessage { tokens: Arc<Vec<u32>> },
}

pub enum SealedKind {
    Section(SectionId),
    /// One sealed turn — single indivisible K/V block.  The `part`
    /// field is retained for forward-compatibility with the doc's
    /// original per-role split design but is always `Role::Assistant`
    /// in the shipped path.
    Turn {
        timeline: TimelineId,
        index: TurnIndex,
        part: Role,
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

### 3.4 `TurnEntryData` shape

```rust
pub struct TurnEntryData {
    block_range: (u64, u64),    // slot extent of the indivisible K/V block
    content: TurnPart,
}

pub struct TurnPart {
    user_text: String,            // verbatim user message
    assistant_text: String,       // verbatim decoded reply (special tokens skipped)
    token_count: usize,           // K/V token count = token_ids.len() (invariant)
    token_ids: TokenBuffer,       // [/no_think][user_msg][user_end][assistant_start]
                                  //   [/think_block][response]
    sig_entries: Vec<SigEntry>,   // BDP sigs over the turn's content tokens
    residence: ResidenceIndex,    // hot/warm/cold residence for the K/V block
}
```

A turn is one indivisible K/V block at the substrate layer.  The text
fields carry the human-readable strings exactly as the caller had them
at submit time (no role-marker envelope, no `/no_think` prefix); they're
stored verbatim so the sidebar reload path renders without any
re-tokenising or boundary scanning.  The `token_ids` field is the
slot's actual token sequence in slot order, used for cross-process
replay (`recover_turn`).  The invariant `token_count == token_ids.len()`
is enforced by a `debug_assert_eq!` at the seal site; see
`scheduler/mod.rs` `perform_seal_and_write`.

**Design evolution.**  An earlier iteration (commit `44b00b53`, Phase 4)
implemented a per-role internal split with separate `user: TurnPart` +
`assistant: TurnPart`, two residences, and a chunk-aligned boundary
between them.  The split was walked back in commit `44f915c7` because
(a) the K/V is one indivisible block anyway — the prefill writes the
whole `[/no_think][user_msg][user_end][assistant_start][...]` sequence
in one pass and seals it as one chunk grid; (b) splitting the residence
created bookkeeping overhead for no operational benefit; (c) the
sidebar reload path actually needs the strings, not the K/V halves.
The single-block shape with paired text fields delivers the same UX
without the split.

`Role` stays `User | Assistant | System` for the boundary-marker
machinery in `Dialect`; it is **not** stored on `TurnPart` (every turn
implicitly carries a user-then-assistant exchange).

BDP scoring is at turn granularity, against the single combined
`content.sig_entries`.  Selection rules (`Sequence { recent,
historical_top_k }`) operate on whole turns, unchanged.

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
(scoring, selection, reconcile, budgeting). Step 12 (emit) emits one
`Sealed::Turn` segment per selected turn — the projection engine does
not know about dialect tokens or boundary wrapping:

```text
Step 12:
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
          out.push(Sealed(Turn { timeline, index, part: Assistant }))
  return Projection { segments: out }
```

Inter-turn boundary wrapping happens in the **projection assembler**,
not the projection engine.  When the assembler walks a `Sealed::Turn`
segment it `extend`s the current `Generated` run with the dialect's
`user_start` tokens, flushes the run (one batched prefill at the
boundary), injects the sealed turn, then `extend`s the next run with
the dialect's `assistant_end` tokens.  The next `Sealed::Turn` (or the
trailing `Generated(UserStart)` the scheduler appends ahead of the
current turn's prefill) collapses with that `assistant_end` into one
5-token batched prefill at the cross-turn boundary.  See
`projection_assembler.rs` `apply_segments`.

The trailing `Generated(UserStart)` ahead of the current turn's prefill
is appended **by the scheduler's `SubmitTurn` handler** (see
`scheduler/mod.rs` around the comment `// Append a trailing
Generated(UserStart) so the current turn's user-side prefill begins
behind a live opener`), not by `Builder::project`.

`NewUserMessage` is not emitted by anything in the shipped path.

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

Lives in `candle-conversation/src/scheduler/projection_assembler.rs`.

### 5.1 Inputs

```rust
pub struct ProjectionAssembler<'s> {
    sequence_id: SequenceId,
    conversation: &'s Conversation,
    session: &'s mut BatchedInferenceSession,
    slot_state: &'s mut SlotState,    // SlotProjectionCache + pending_user_part
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

Every apply is a full rebuild from substrate.  The slot is truncated
to zero, every segment is resolved and injected in declaration order,
and the writer tail is re-attached at the end.  The performance
optimisation lives inside the `Generated` arm: a rolling hash committed
over every emitted token serves as a content-addressed cache key, so
runs of structural template tokens that have been live-prefilled before
reuse their captured K/V instead of running a fresh forward pass.

```text
1.  Snapshot the slot's writer tail (in-flight decode chunks) as a
    `WriterTail`.  Re-attached at the end so mid-decode reprojections
    don't drop already-generated tokens.

2.  Truncate the slot to zero blocks.  Drops Arc refs to whatever the
    slot previously held; arenas reclaim asynchronously.

3.  Walk new_segments:
      rolling_hash = 0
      current_run = empty
      for segment in new_segments:
          match segment:
              Sealed(kind):
                  flush_current_run()
                  resolve sealed bytes from substrate; inject onto slot
                  fold segment.tokens (looked up from substrate) into
                      rolling_hash
              Generated { tokens, identity }:
                  // Don't fold yet — the cache key covers the WHOLE
                  // run, not the partial prefix at this segment's
                  // start.
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
        // Cache key = preceding context + the run's full token
        // concatenation.  Both halves are load-bearing:
        //   - rolling_hash captures the attention prefix the K/V was
        //     computed under;
        //   - run_tokens commits to the exact batched forward pass
        //     that produced it.  A run [A,B,C] captured together is
        //     not interchangeable with [A,B,D] even though they share
        //     A,B at the head — B's K/V was computed in a batch that
        //     also contained C, with a different padding/position
        //     layout from the [A,B,D] batch.
        run_tokens = concat(current_run.tokens)
        key = fold(rolling_hash, run_tokens)
        if slot_state.cache.contains_key(key):
            inject cache[key] onto slot                  // Arc-clone, no GPU
        else:
            enqueue PrefillRun {
                sequence_id: self.sequence_id,
                tokens: run_tokens,
                write_offset: slot.current_block_count,
                capture: SlotCache { key },
            }
        advance slot.current_block_count by ceil(len(run_tokens) / CHUNK_SIZE)
        rolling_hash = key       // committed; next segment sees post-run hash
        current_run = empty

4.  Restore writer tail at slot.current_block_count + queued-prefill-blocks.
    The writer tail attaches AFTER all pending prefills resolve; the
    scheduler's prefill-batcher handles the ordering.
```

The assembler does not maintain a "previous projection" record.  The
cache is content-addressed and persists across reprojections under the
slot — that is the only state carried forward.  A reproject with an
identical projection produces identical run-hash keys, hits every
cached entry, and finishes without enqueuing any prefill work.  A
reproject that diverges at a sealed boundary re-resolves the diverging
sealed entries from substrate (cheap, metadata-only) and re-keys every
generated run downstream of the divergence — those see cache misses
and prefill fresh.

### 5.3 Prefill-run capture targets

The shipped path has only one active capture target — generated template
runs captured into the slot cache.  The `PendingUserPart` target
described in the original design remains reserved infrastructure (no
emitter, no consumer in the current code).

```rust
/// Reserved for future use — defined inline at the capture site,
/// not as a public enum in the shipped path.
enum RunCapture {
    SlotCache { key: u64 },     // shipped
    PendingUserPart,             // reserved
}
```

Capture happens directly inside the assembler's `drive_prefill_and_capture`
helper: after a successful `forward_batched` over a run's tokens, the
per-layer `Vec<SealedSequence>` covering the just-written block range is
sliced off the slot and inserted into `state.cache.memo` under the
content-addressed key.  The assembler treats a captured run identically
to a substrate-sourced sealed sequence on subsequent hits.

### 5.4 Cache lifetime and growth

`SlotProjectionCache::memo` is `HashMap<u64, Arc<Vec<SealedSequence>>>`. It
lives on `SlotState`, populated as runs miss, queried as runs hit.

**Trim policy.**  Every `apply_segments` walk records each key it
touches (hit or freshly captured) into `SlotState::last_apply_keys`.
At seal time (`perform_seal_and_write` after `persist_trigger.fire()`),
`SlotState::trim_post_turn` calls `SlotProjectionCache::retain_keys`
against that set, dropping every cache entry whose key wasn't part of
the most recent projection's working set.  The Arcs of the dropped
entries release their refcount; GPU arenas reclaim asynchronously
once no other reference holds the captured `SealedSequence`s.

The kept set is exactly the boundary K/V the slot's current
post-seal state depends on — every `user_start` and `assistant_end`
between past turns within the projection's window, plus the trailing
`user_start_current` (which becomes the inter-turn boundary opener
for this turn in next-turn projections).  Next-turn reprojection
with the same window shape hits every entry; cost is zero forward
passes for boundaries.

**What gets dropped:**
- Boundary runs from turns that fell out of the projection's window
  (the schema's per-target turn budget bounds the live working set).
- Runs whose rolling-hash prefix diverged when an upstream segment
  changed (e.g. a different system-prompt window selected, a section
  collection materialised differently) — these entries were stranded
  on a hash that no current segment list will ever reproduce.

**Memory bound.**  The cache stabilises at ~one projection's worth of
generated runs: window-bounded, not conversation-length-bounded.  For
zend's dialogue layer (~16 past-turn window) the working set is ~17
entries.  Each entry holds a per-layer `Vec<SealedSequence>` covering
a few-token boundary run — kilobytes per layer × layer count ≈ a few
hundred KB per entry on Qwen3-30B-A3B.  Total steady-state: low
single-digit MB of GPU per slot.

The mid-decode reprojection path also benefits: it happens within a
single turn, before `trim_post_turn` fires, so reprojection sees the
full cache.

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

The shipped path performs capture inline inside the assembler's
`drive_prefill_and_capture` (no separate hook stage):

```text
forward_batched over run.tokens → writes K/V into slot at [start, end)
captured = slice_per_layer_sealed(snapshot_sequence_per_layer(slot),
                                  start_block, end_block)
state.cache.insert(key, Arc::new(captured))
```

`PendingUserPart` has no live capture path today.  When (or if) a future
feature wires up `NewUserMessage` emission, the same inline capture
shape applies — write to `state.pending_user_part` instead of
`state.cache`.

The captured `Vec<SealedSequence>` is one per layer.  Wrapping in `Arc`
lets the cache hand the same captured bytes back via Arc-clone on
subsequent hits without further copies; the Arc refs inside the
`SealedChunk`s keep the arena chunks alive even if the slot's prefix is
later rebuilt past them.

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

Decode starts on the slot once the projection apply has finished and
the user-message prefill has landed.  In the shipped path the user
message is part of `prefill_tokens` (which the `submit_turn` formatter
concatenates as
`[/no_think][user_msg][user_end][assistant_start][/think_block]`), so
the writer tail naturally attaches at the position after the baked
`assistant_start` (+ optional `/think_block`) — exactly where the
model should start emitting its first response token.

---

## 7. Slot state

```rust
pub(super) struct SlotState {
    pub(super) cache: SlotProjectionCache,
    pub(super) pending_user_part: Option<Arc<Vec<SealedSequence>>>,
    pub(super) last_apply_keys: HashSet<u64>,
}
```

`cache` is the content-addressed memo of captured live-prefill runs.
Keyed by `hash(rolling_hash_at_run_start ++ concat(run_tokens))`, valued
as the per-layer `Arc<Vec<SealedSequence>>` captured from the slot after
the prefill that produced it.  Populated on cache miss after the prefill
completes; queried on every run-flush during the assembler walk.
**Trimmed at end-of-turn** to the most recent apply's working set
(`SlotState::trim_post_turn`, fired from `perform_seal_and_write`
after persistence has been triggered) so memory stays bounded while
still-relevant boundary K/V stays hot — see §5.4.

`pending_user_part` is reserved infrastructure (no current emitter or
consumer).  `trim_post_turn` resets it to `None` alongside the cache
trim, so when `NewUserMessage` emission is eventually wired up the
field will already participate in the end-of-turn reset contract.  The
original design's mid-decode-reproject survival contract still applies:
the field lives on `SlotState`, not on the slot, so a truncate-to-zero
rebuild during reprojection would preserve it.

`last_apply_keys` is the working set the slot's current K/V residency
reflects.  Reset to empty at the start of every `apply_segments`; each
`flush_run` inserts its computed key into the set.  Outlives the apply
that produced it because `trim_post_turn` consults it at seal time.
Next `apply_segments` overwrites with a fresh set.

---

## 8. Substrate / persistence consequences

### 8.1 Substrate

- `Substrate::sections` shrinks: template-kind items no longer ingest as
  sections (Phase 3).  No `__system_start` / `__system_end` synthetic
  sections (Phase 3 retired the synthetic marker mechanism).  The
  zend-side `tools_open` / `tools_close` / `mode` are already `kind:
  template` (Phase 1 YAML migration).
- `SectionEntryData` stores authored content sections only.
- `TurnEntryData` holds one `TurnPart` carrying `user_text` +
  `assistant_text` strings and one indivisible K/V block (§3.4).  The
  doc's original per-role residence split shipped in Phase 4 and was
  walked back in commit `44f915c7`.
- `Substrate::append_complete(timeline, write)` appends one turn.  The
  inter-turn role markers (`user_start` head, `assistant_end` tail) are
  **not** stored — they're re-emitted as `Generated` segments at every
  projection.
- `Substrate::turn_token_count(group, index)` returns
  `TurnPart::token_count`, which is the K/V block's exact token count.
  The boundary markers' tokens are not counted (they're not pinned).
- BDP scoring (`turn_score`) runs against the single combined
  `content.sig_entries`.

### 8.2 Persistence (redo log)

- `TurnDecl` carries one `(block_start, block_end)` range covering the
  turn's indivisible K/V block, plus verbatim `user_text` and
  `assistant_text` strings (`#[serde(default)]` for forward
  compatibility).
- Per-turn `Chunks` stream carries the turn's K/V chunks in slot order.
  The original `user_chunk_count` / `user_token_count` /
  `user_sig_count` partition fields are persisted as zeros (kept for
  forward compatibility with the per-role split design; no consumer
  reads them in the shipped path).
- Per-turn `Tokens` record carries the slot's full token sequence
  `[/no_think][user_msg][user_end][assistant_start][/think_block][response]`
  — without the inter-turn `user_start` head and `assistant_end` tail,
  which are projection-time only.  The trailing EOS that decode samples
  but never forwards is **trimmed at seal** so `token_ids.len() ==
  token_count` (the K/V chunk grid token count).  See
  `scheduler/mod.rs` `perform_seal_and_write` for the trim and the
  `debug_assert_eq!` that guards the invariant.
- Per-turn `Signatures` record is unsplit.
- No template-section records.  Inter-turn boundary markers don't
  persist.  Redo log shrinks accordingly.

### 8.3 Reload

`SubstrateReload::recover_turn` reconstructs each `TurnEntryData`'s
single `TurnPart` from the persisted records.  Cold-marker turns
(chunks not yet landed when the daemon shut down) materialise with an
empty `Vec<SealedSequence>` cold residence until elevation recovers
them.  The `user_text` + `assistant_text` strings come straight off the
`TurnDecl` — no re-tokenising, no boundary scanning.

### 8.4 BDP signature scope

Sigs are extracted from the decode path (assistant content) and the
prefill path (user content + intra-turn role markers).  Inter-turn
boundary markers (`user_start` head, `assistant_end` tail) are never
sig'd because they're never persisted — they only exist as transient
`Generated` segments at projection time, and the sig extractor doesn't
look at the boundary-cache region.

The Phase 6 "BDP sig extractor skips blocks captured to SlotCache"
deliverable described in the original design is therefore satisfied
**structurally** rather than via an explicit skip flag: the cache
region doesn't intersect the sealed-turn region the sig extractor
walks.

---

## 9. YAML migration (zend's `projection.yaml`)

This migration shipped alongside Phase 1's dialect catalog +
`kind: template` parser; `zend/src/prompts/projection.yaml` is
already in the **After** shape below.  Recorded here for the
before/after diff history.

Before (pre-Phase-1):

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

After (as shipped):

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

Module-level diff summary (as shipped through Phase 5):

| Module | Change |
|---|---|
| `candle-conversation/src/models/dialect.rs` | `DialectTemplate` enum, new dialect fields. |
| `candle-conversation/src/models/qwen3.rs`, etc. | Populate new dialect fields per model. |
| `candle-conversation/src/projection/schema.rs` | `SystemPromptItem::Template` variant. **No `BoundaryMarkers` on `Schema`** — moved to scheduler. |
| `candle-conversation/src/projection/yaml.rs` | Parse `kind: template`, resolve `dialect:` refs at build. |
| `candle-conversation/src/projection/project.rs` | Emit `Vec<ProjectionSegment>`.  One `Sealed::Turn` per past turn (boundary wrapping is the assembler's job). |
| `candle-conversation/src/projection/builder.rs` | `Builder::tokenize_boundary_markers` removed; markers built at engine construction instead. |
| `candle-conversation/src/scheduler/mod.rs` | `apply_projection` calls `ProjectionAssembler`.  `Scheduler` owns `BoundaryMarkers` + `SlotProjectionCache` per slot.  `perform_seal_and_write` trims trailing EOS, fires `trim_post_turn`, asserts `token_ids.len() == token_count`. |
| `candle-conversation/src/scheduler/projection_assembler.rs` | Walk logic, rolling-hash cache lookup, inline prefill+capture.  Wraps every `Sealed::Turn` with `user_start` / `assistant_end`. |
| `candle-conversation/src/scheduler/prefill.rs` | Carries `user_text` through to `DecodeState` for verbatim seal storage. |
| `candle-conversation/src/substrate.rs` | `TurnPart { user_text, assistant_text, token_count, token_ids, sig_entries, residence }` — single block, paired strings. |
| `candle-conversation/src/persistence/streams.rs`, `resume.rs` | `TurnDecl` carries `user_text` + `assistant_text` strings (`#[serde(default)]`).  Partition fields persisted as zeros (forward-compat). |
| `candle-conversation/src/conversation.rs` | `submit_turn` formats prefill without the inter-turn boundary tokens.  `recovered_history` returns `Vec<(Role, String)>` straight from substrate. |
| `candle-conversation/src/engine.rs` | Pre-tokenises `BoundaryMarkers` once at engine construction and passes them to `Scheduler::new`. |
| `zend/src/session.rs` | No longer tokenises boundary markers; no decode call on `recovered_history`. |
| `zend/src/prompts/projection.yaml` | `system_open`, `system_close`, `mode`, `tools_open`, `tools_close` migrated to `kind: template` (landed in Phase 1's YAML migration). |

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

### Phase 2 — `ProjectionSegment` + `ProjectionAssembler`

**Behaviour-preserving refactor.  Bit-identical K/V on slot.**

Deliverables:
- `ProjectionSegment` enum added (`Sealed`, `Generated`, and
  `NewUserMessage` variants; only `Sealed` is emitted by the projection
  engine at this phase).
- `Builder::project` returns `Projection { segments:
  Vec<ProjectionSegment> }`.  All segments emit as `Sealed`, including
  template items (still routed through substrate ingest).
- `ProjectionAssembler` module extracted from
  `Scheduler::apply_projection`.  Always-rebuild semantics: snapshot
  writer tail, truncate slot to zero, resolve every segment from
  substrate, inject in declaration order, re-attach tail.  Equivalent
  to the prior inline implementation, factored into one module.

Tests:
- All existing scheduler tests pass.
- Unit-level fixture tests for the new segment types and helpers.

Risk: **medium.**  Output-shape change touches many callsites.  No
quality-impact risk.

### Phase 3 — Slot cache + `Generated` for system-prompt + `PrefillRun`

**Behaviour change. System-prompt boundary K/V becomes attention-correct.**

Deliverables:
- `SlotProjectionCache` added to `SlotState` — `AHashMap<u64,
  Vec<SealedSequence>>` keyed by `hash(rolling_hash ++ run_tokens)`.
- Rolling-hash machinery inside the assembler walk — fold sealed
  segments' tokens into `rolling_hash` as they're injected; on a
  generated-run flush, compute the cache key over `rolling_hash` and
  the run's full token concatenation.
- `PrefillRun` added to scheduler, integrated into the prefill batcher.
- Adjacent generated segments on the same slot collect into one
  `PrefillRun` and run as one forward pass.
- Cross-slot batching as today.
- Capture hook: a completed `PrefillRun` extracts the captured per-layer
  `Vec<SealedSequence>` from the slot and inserts it into
  `SlotProjectionCache` under the key the run was enqueued with.
- `Builder::project` emits `Generated` segments for `kind: template`
  items.  `Sealed` for `kind: section` (and turn) items.
- Template items NO LONGER ingested as sections at session setup.
- `conversation.rs` ingest loop drops template handling for
  system-prompt templates (content sections still ingest cumulatively).
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

### Phase 5 — inter-turn boundary `Generated`s (shipped, simplified)

**Status: shipped in commits `b16cc728` + `fdd058a4`.**

**Behaviour change: inter-turn attention pivots become correct.**

The shipped Phase 5 is leaner than the original spec.  It delivers
attention-correct inter-turn boundary K/V (the actual goal — see §2.1)
without the per-role `TurnPart` split, the `NewUserMessage` capture
path, or the four-marker turn expansion the original design called for.
The trade-off is recorded in commit `44f915c7` (which walked back the
Phase 4 per-role split) and reflected in §3.4 and §1 above.

Deliverables (as shipped):
- The projection engine emits one `Sealed::Turn` segment per past turn.
  Inter-turn boundary markers are **not** segments at this layer.
- The projection assembler wraps each `Sealed::Turn` it walks with
  `user_start` (joined into the run that flushes before the inject) and
  `assistant_end` (opened on a fresh run after the inject).  Adjacent
  runs at cross-turn boundaries collapse into a single 5-token batched
  prefill via the assembler's run accumulator.  See
  `projection_assembler.rs` `apply_segments`.
- The scheduler's `SubmitTurn` handler appends a trailing
  `Generated(UserStart)` after the projection's past-turn segments and
  before the current turn's prefill.  See `scheduler/mod.rs`.
- `BoundaryMarkers` (pre-tokenised `user_start` + `assistant_end`) is
  built once at engine construction and lives on the scheduler.
  Threaded into the assembler via `ApplyContext.boundary_markers`.
  **Not on `Schema`** — the schema is dialect-agnostic.
- `TurnPart` carries `user_text` + `assistant_text` as verbatim
  strings.  Single indivisible K/V block (Phase 4's per-role split was
  walked back).
- `conversation.rs` `submit_turn` formats prefill as
  `[/no_think][user_msg][user_end][assistant_start][/think_block]` —
  the inter-turn `user_start` head and `assistant_end` tail are
  stripped from the persisted prefill bytes.  The intra-turn pair
  (`user_end` + `assistant_start`) stays baked because its hidden
  state is dominated by the turn's own invariant content on both sides.
- `recovered_history` returns `Vec<(Role, String)>` straight from the
  substrate's text fields — no re-tokenising, no marker scanning.
- Trailing-EOS trim at seal: `state.generated_tokens` always ends with
  one unforwarded terminator (EOS or max-tokens edge); the seal-site
  assembly drops it so `token_ids.len() == token_count`.  Guarded by
  `debug_assert_eq!` (see `scheduler/mod.rs` `perform_seal_and_write`).
- `SlotState::trim_post_turn` at end-of-turn: retains only the cache
  entries whose keys were touched by the most recent `apply_segments`
  walk, dropping stranded entries from prior projections.  Bounds GPU
  arena memory at ~one projection's working set while keeping next-turn
  reprojection cache-hot.  See §5.4.

Deferred (originally Phase 5, intentionally left for later):
- `ProjectionSegment::NewUserMessage` emission.  Variant exists in the
  enum as forward-looking infrastructure; nothing currently emits it.
- `SlotState::pending_user_part` population.  Field exists and
  participates in `clear_post_turn`; nothing currently populates it.
- Per-role `TurnPart` split with two residences.  Walked back; see
  `44f915c7`.

Tests (shipped):
- End-to-end: submit_turn → decode → seal → reload → replay.  Output
  text coherent; sidebar renders user + assistant turns in the right
  roles after daemon resume.
- Multi-turn: turn N attends correctly to turn N-1 (inter-turn boundary
  K/V is computed under the actual runtime causal prefix at every
  projection).
- 360 `candle-conversation` lib tests pass on every commit.

Risk: **medium** as built.  Scheduler hot path touched, but the
simplification removed the riskiest pieces (pending_user_part /
reproject interaction, the two-range `TurnDecl`).  Redo log rebuilt.

### Phase 6 — YAML migration + cleanup (absorbed into Phases 1–5)

**Status: complete — every code deliverable landed during an earlier
phase rather than as a separate Phase 6 commit.**  The original list
anticipated cleanup work that would still be needed after Phase 5
landed; in practice the relevant pieces were retired in-flight as
each phase rewrote its surrounding code.  Recorded here per
deliverable.

Deliverables (as shipped):
- ✅ **YAML migration.**  `zend/src/prompts/projection.yaml` already
  uses `kind: template` with `dialect:` references for `system_open`,
  `mode` (`no_think_prefix`), `tools_open`, `tools_close`,
  `system_close`.  Landed alongside Phase 1's dialect catalog +
  `kind: template` parser.
- ✅ **Synthetic-marker cleanup.**  `LayerSchema::system_start_section`,
  `system_end_section`, and `Builder::set_system_markers` no longer
  exist (Phase 3 retired the synthetic marker mechanism when the
  schema became self-describing via dialect templates).
  `next_section_id_raw` survives but is **not** synthetic-marker
  machinery — it's the section-ID allocator backing `add_section` /
  `add_section_to_collection` builder methods, which remain
  legitimate user-facing API.  Kept by design.
- ✅ **Template ingest skip.**  `conversation.rs`'s cumulative-prefix
  ingest loop filters out template-kind items via `is_template`
  guards at every step (sec setup, byte accounting, batch ingest).
  Templates contribute nothing to `linear_prefix` /
  `fixed_prefix`; content sections still cumulative-ingest, as
  intended.
- ✅ **BDP sig extractor structural-block skip.**  Satisfied
  structurally: the boundary-marker `Generated` runs live on the
  slot cache, never enter the substrate, and the sig extractor
  doesn't walk the cache region.  No explicit skip flag needed.

Remaining (measurement, not code):
- ⏳ **BDP retrieval MRR / Top-1 against canonical probe set: equal
  or better than Phase 3 baseline.**  Part of the broader benchmark
  backlog for the May 2026 arXiv submission — runs against the
  current shipped state, no further code changes required to
  measure.

Risk: closed.

---

## 14. Open issues

- ~~**Chunk-aligned user/assistant boundary.**~~  Moot — the per-role
  `TurnPart` split was walked back in commit `44f915c7`.  Turns are
  one indivisible K/V block at the substrate layer.

- ~~**Reproject and pending_user_part.**~~  Reserved infrastructure;
  nothing currently emits `NewUserMessage` so the mid-decode survival
  contract doesn't fire in the shipped path.  The contract is still
  documented in §5.4 / §7 for the future feature that wires it up.

- **PrefillRun granularity for batch-throughput.** The single-slot
  concatenation rule says "adjacent runs on the same slot combine into
  one forward pass." For very long generated runs the per-pass cap
  (`max_prefill_chunk`) still applies — the run executes as multiple
  chunks through the same forward batcher.  In the shipped assembler
  this is handled by the chunked-loop inside `drive_prefill_and_capture`.

- **Cache key cross-conversation reuse.** Keys are deterministic across
  slots since they're hashes of token sequences. In principle a cache
  shared across slots in the same workspace could be useful. Out of
  scope for v1; revisit if profiling shows the per-slot cache misses
  are concentrated on common prefixes.

- **Cache trim policy.**  The end-of-turn `trim_post_turn` retains
  cache entries by the working set of the most recent apply.  This
  keeps next-turn reprojection cache-hot (same window shape → all
  kept entries hit) while bounding the cache at ~one projection's
  worth of entries.  An alternative would be a hard `clear_post_turn`
  that drops everything — simpler but pays `O(window × ~5)` tokens
  of batched re-prefill on every next-turn projection.  Trim won
  on cost/complexity grounds.  See §5.4.

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
