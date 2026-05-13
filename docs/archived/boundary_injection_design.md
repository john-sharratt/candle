# Boundary Token Injection via Static Chunk Cache — Design Document

## 1. Goal

Use the static chunk cache infrastructure (built in the previous phase) to
eliminate repeated prefill passes for the structural boundary tokens that appear
between dynamic content in every conversation turn.

**Target boundaries:**

| Boundary | When it occurs | Tokens (ChatML) |
|----------|---------------|-----------------|
| **document+system start** | `initial_handle`: very first tokens in sequence | `<\|im_start\|>system\n` (+ optional `/no_think\n`) |
| **system→user** | `initial_handle`: after system prompt content | `<\|im_end\|>\n<\|im_start\|>user\n` |
| **user→assistant** | Inside `submit_turn`: after user text, before decode | `<\|im_end\|>\n<\|im_start\|>assistant\n` (+ optional think block) |
| **assistant→user** | In `finish_turn`: after decode completes, before next turn | `<\|im_end\|>\n<\|im_start\|>user\n` |

Note: for ChatML and Llama 3, **system→user** and **assistant→user** produce
identical token sequences (`turn_end + user_start`), so they share a single
cache entry. For Llama 2 they differ (`\n<</SYS>>\n\n` vs ` </s>`) and need
separate entries.

At server scale with hundreds of concurrent conversations generating many turns,
these 5–15 token prefills are repeated thousands of times with identical token
sequences. Replacing them with O(1) chunk injection would eliminate the
corresponding forward passes entirely.

## 2. KV Context Dependency and the Approximation

Cached boundary KV is an approximation for mid-sequence injection. In a
multi-layer transformer, K and V at layer L are projections of the hidden state
from layer L−1, which depends on the full causal context. A prototype sequence
generates boundary KV attending to a representative context, not the exact
runtime conversation — so the cached KV is close but not identical to an
in-context prefill.

**Why this is acceptable:** Structural tokens (`<|im_end|>`, role headers) carry
near-zero semantic content. Their hidden states are highly stereotyped across
different preceding content — the model treats them as format markers, not
information. Attention weights from content tokens to structural positions are
typically very small, so minor KV differences have negligible effect on logits.
RoPE re-rotation (`chunk_rope_shifts`) is always exact.

**Contextual prototype generation (§7.1)** minimises error by prefilling a
representative preceding context before the boundary tokens, so boundary KV
reflects realistic causal attention at every layer. `write_offset_shift` aligns
boundary tokens to a clean block boundary without padding tokens.

`DocumentSystemStart` is **exact** — position 0, no preceding context.

## 3. Boundary Types

The four structural boundaries. Concrete token sequences are resolved by
`BoundaryResolver` at engine startup — callers only use `Boundary` variants
(ChatML shown for illustration):

| Boundary | Tokens | Cache type |
|----------|--------|------------|
| `DocumentSystemStart` | `<\|im_start\|>system\n` [+ `/no_think\n`] | Exact (position 0) |
| `SystemToUser` | `<\|im_end\|>\n<\|im_start\|>user\n` | Contextual-approx |
| `UserToAssistant` | `<\|im_end\|>\n<\|im_start\|>assistant\n` [+ think block] | Contextual-approx |
| `AssistantToUser` | `<\|im_end\|>\n<\|im_start\|>user\n` | Contextual-approx |

For ChatML, `SystemToUser` and `AssistantToUser` share one token sequence and
one cache entry. For Llama 2 they differ. Only boundaries that produce zero
tokens for the current dialect are skipped; all others are always cached.

### 3.4 Boundary Abstraction

Callers never deal with token strings, `ChunkKey`s, or dialect details. A
`Boundary` enum captures the semantic position in the conversation structure.
The dialect + config resolve each variant to its concrete token sequence at
cache construction time.

```rust
/// Semantic boundary positions in a conversation.
///
/// Each variant represents a fixed structural transition between
/// dynamic content regions. The actual token sequence is determined
/// by the model's dialect and thinking configuration at engine
/// construction time — callers never see dialect-specific tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Boundary {
    /// Start of document + system header.
    /// Tokens: document_start + system_start [+ no_think prefix].
    /// Position: 0 (exact injection — no preceding context).
    DocumentSystemStart,

    /// Transition from system prompt content to user turn.
    /// Tokens: system_end + user_start.
    /// Position: mid-sequence (contextual-approximate injection — see §2.3).
    SystemToUser,

    /// Transition from user content to assistant response.
    /// Tokens: user_end + assistant_start [+ think/no_think block].
    /// Position: mid-sequence (contextual-approximate injection — see §2.3).
    UserToAssistant,

    /// Transition from assistant response to next user turn.
    /// Tokens: turn_end + user_start.
    /// Position: mid-sequence (contextual-approximate injection — see §2.3).
    AssistantToUser,
}
```

**Key design points:**

1. **No thinking variants.** Whether the model uses `<think>\n`,
   `<think>\n\n</think>\n\n`, or nothing is resolved at construction time by
   the dialect + `suppress_thinking` config. `UserToAssistant` always maps to
   the correct variant for this engine session. Callers never branch on thinking
   mode.

2. **No dialect variants.** The same `Boundary::SystemToUser` works for ChatML
   (`<|im_end|>\n<|im_start|>user\n`), Llama 3
   (`<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n`), and Llama 2
   (`\n<</SYS>>\n\n`). The mapping is internal.

3. **Deduplication is automatic.** If two variants produce the same token
   sequence (e.g., ChatML's `SystemToUser` and `AssistantToUser` both resolve
   to `<|im_end|>\n<|im_start|>user\n`), the cache stores one `ChunkEntry` and
   both variants point to it.

4. **All boundaries required.** If a boundary's text tokenizes to zero tokens
   the dialect is misconfigured and `BoundaryResolver::new` panics at engine
   startup rather than silently skipping it. This makes configuration errors
   immediately visible.

### 3.5 Dialect Resolution

A `BoundaryResolver` is constructed once at engine startup from the dialect and
thinking config. It maps each `Boundary` variant to its token sequence and
`ChunkKey`:

```rust
/// Resolves Boundary variants to token sequences using the session's
/// dialect and thinking configuration. Constructed once at engine startup.
pub struct BoundaryResolver {
    entries: HashMap<Boundary, ResolvedBoundary>,
}

struct ResolvedBoundary {
    key: ChunkKey,
    tokens: Vec<u32>,
    text: String,
}

impl BoundaryResolver {
    pub fn new(
        dialect: &Dialect,
        suppress_thinking: bool,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Self {
        let thinking_capable = dialect.supports_no_think();
        let mut entries = HashMap::new();

        // DocumentSystemStart: document_start + system_start [+ /no_think]
        let mut doc_sys = format!("{}{}", dialect.document_start, dialect.system_start);
        if suppress_thinking && !dialect.no_think.is_empty() {
            doc_sys.push_str(dialect.no_think);
        }
        Self::insert_required(
            &mut entries, tokenizer,
            Boundary::DocumentSystemStart, &doc_sys, true,
        );

        // SystemToUser: system_end + user_start
        let sys_to_user = format!("{}{}", dialect.system_end, dialect.user_start);
        Self::insert_required(
            &mut entries, tokenizer,
            Boundary::SystemToUser, &sys_to_user, false,
        );

        // UserToAssistant: user_end + active_assistant_start(...)
        //   Think block is flattened here based on config — caller never sees it.
        let user_to_asst = format!(
            "{}{}",
            dialect.user_end,
            dialect.active_assistant_start(suppress_thinking, thinking_capable),
        );
        Self::insert_required(
            &mut entries, tokenizer,
            Boundary::UserToAssistant, &user_to_asst, false,
        );

        // AssistantToUser: turn_end + user_start
        let asst_to_user = format!("{}{}", dialect.turn_end, dialect.user_start);
        Self::insert_required(
            &mut entries, tokenizer,
            Boundary::AssistantToUser, &asst_to_user, false,
        );

        Self { entries }
    }

    /// Token count for a boundary.
    pub fn token_count(&self, boundary: Boundary) -> usize {
        self.entries[&boundary].tokens.len()
    }

    /// The ChunkKey for a boundary (infallible � all boundaries are required
    /// at construction time).
    pub fn key_for(&self, boundary: Boundary) -> &ChunkKey {
        &self.entries[&boundary].key
    }

    /// All resolved boundaries for cache generation.
    /// Deduplicates entries with identical token sequences.
    pub fn cache_entries(&self) -> Vec<(ChunkKey, Vec<u32>, bool)> {
        let mut seen_tokens: HashMap<Vec<u32>, ChunkKey> = HashMap::new();
        let mut out = Vec::new();
        for (&boundary, entry) in &self.entries {
            if let Some(_existing) = seen_tokens.get(&entry.tokens) {
                // Duplicate token sequence �?? alias will be set up in the cache
                continue;
            }
            seen_tokens.insert(entry.tokens.clone(), entry.key.clone());
            out.push((
                entry.key.clone(),
                entry.tokens.clone(),
                matches!(boundary, Boundary::DocumentSystemStart),
            ));
        }
        out
    }

    fn insert_required(
        entries: &mut HashMap<Boundary, ResolvedBoundary>,
        tokenizer: &tokenizers::Tokenizer,
        boundary: Boundary,
        text: &str,
    ) {
        let tokens: Vec<u32> = tokenizer
            .encode(text, false)
            .expect("tokenizer error")
            .get_ids()
            .to_vec();
        assert!(
            !tokens.is_empty(),
            "boundary {boundary:?} produced zero tokens � check dialect configuration",
        );
        entries.insert(boundary, ResolvedBoundary {
            key: ChunkKey(format!("boundary:{boundary:?}").to_lowercase()),
            tokens,
            text: text.to_string(),
        });
    }
}
```

**All boundaries are required.** Construction panics if any boundary tokenizes
to zero tokens � a misconfigured dialect is caught immediately at engine startup.

In practice, **Llama 2 benefits least** from boundary injection because its
boundaries are short. ChatML and Llama 3 benefit most (5–12 tokens per boundary).

**Position-0 entries are exact, not approximate.** The `DocumentSystemStart`
boundary is injected at position 0 of a fresh sequence — there is no
preceding context, so the cached KV is identical to what a full prefill would
produce. These use the existing `InjectPrefix` mechanism (section 5 of the
static chunk cache design doc) and do not require the approximation discussed
in section 2.

## 4. Current Turn Flow (Before Changes)

### Initial Prefill (Turn 0):

```
 initial_handle() prefills:
     format_system_prompt() + turn_end + user_start
     = "<|im_start|>system\n/no_think\nYou are helpful.<|im_end|>\n<|im_start|>user\n"
       ^^^^^^^^^^^^^^^^^^^^^^^^                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       doc_start + system_start                       system_end + user_start
       (+ optional /no_think)                         (= turn_end + user_start for ChatML)

     → SchedulerRequest::Prefill { tokens: [...all...] }
     → Single forward_batched pass
     → KV cache now has: system_header system_content system→user_boundary  [ready]
```

### Turn N (N ≥ 1):

```
                    ┌─────────────────────────────────────────────┐
 Already in KV:     │ ...previous_turns... turn_end  user_start   │
                    └─────────────────────────────────────────────┘
                                                                   ↑ cursor here

 submit_turn("hello") formats:
     "hello" + user_end + active_assistant_start(...)
     = "hello<|im_end|>\n<|im_start|>assistant\n<think>\n"

     → SchedulerRequest::SubmitTurn { prefill_tokens: [...all...] }
     → Single forward_batched pass (prefill)
     → Decode loop generates response

 finish_turn() prefills:
     turn_end + user_start
     = "<|im_end|>\n<|im_start|>user\n"

     → SchedulerRequest::Prefill { tokens: [...] }
     → Single forward_batched pass
     → KV cache now has: ...response... turn_end user_start  [ready for next turn]
```

## 5. Proposed Turn Flow (After Changes)

### Initial Prefill (Turn 0):

```
 initial_handle():
     Step 1: Inject Boundary::DocumentSystemStart
             → InjectPrefix at position 0 (EXACT — no preceding context)
             → No forward pass

     Step 2: Prefill system prompt content (dynamic)
             → SchedulerRequest::Prefill { tokens: [...system_content...] }
             → forward_batched

     Step 3: Inject Boundary::SystemToUser
             → InjectBoundary (contextual-approximate — see §2.3)
             → No forward pass
             → KV cache now has: system_header system_content system→user  [ready]
```

### Turn N (N ≥ 1):

```
                    ┌─────────────────────────────────────────────┐
 Already in KV:     │ ...previous_turns... turn_end  user_start   │
                    └─────────────────────────────────────────────┘
                                                                   ↑ cursor here

 submit_turn("hello"):
     [Phase 1 — unchanged from current flow]
     Prefill "hello" + user_end + active_assistant_start in one pass
     → SchedulerRequest::SubmitTurn { prefill_tokens: [...all...] }
     → forward_batched (prefill) → decode loop

     [Phase 2 — split prefill, requires solving logits problem (§8)]
     Step 1: Prefill only "hello" (dynamic content)
             → SchedulerRequest::Prefill { tokens: [...user_tokens...] }
             → forward_batched
     Step 2: Inject Boundary::UserToAssistant
             → No forward pass — just block table pointer writes + refcount
     Step 3: Logits pass for last boundary token (1 token forward)
     Step 4: Decode loop

 finish_turn():
     Step 1: Inject Boundary::AssistantToUser
             → No forward pass
             → KV cache now has: ...response... turn_end user_start  [ready]
```

**Key difference:** The boundary tokens' KV comes from the static chunk cache
instead of a forward pass. The model never sees these tokens during the turn —
their KV is injected directly into the block table.

## 6. New Concept: Mid-Sequence Chunk Injection

The existing `InjectPrefix` operates on a **fresh sequence slot** — it creates
a new sequence and populates it with cached chunks before any content exists.

Mid-sequence injection is different: we append cached chunks to an **existing
sequence** that already has content in its KV cache. This requires a new
primitive.

### 6.1 `InjectBoundary` Scheduler Request

```rust
SchedulerRequest::InjectBoundary {
    /// The existing sequence to append boundary chunks to.
    sequence_id: SequenceId,
    /// Which boundary to inject (dialect/thinking details are opaque).
    boundary: Boundary,
    /// The position in the sequence where the boundary starts.
    /// Used for RoPE shift: rope_shift = target_pos for each chunk.
    target_pos: usize,
    /// Reply with the new cursor position after injection.
    reply: oneshot::Sender<Result<usize>>,
}
```

### 6.2 Scheduler Handler

```rust
fn handle_inject_boundary(
    &mut self,
    sequence_id: SequenceId,
    boundary: Boundary,
    target_pos: usize,
) -> Result<usize> {
    let key = self.boundary_resolver.key_for(boundary);
    let entry = self.static_cache.entry(key);

    // For each chunk in the boundary entry:
    // 1. Write chunk_id into block_table[seq][block_idx]
    // 2. Store ChunkRef::with_rope_shift(handle, arena_chunks, target_pos)
    // 3. Increment block_count
    //
    // This is similar to InjectPrefix but operates on an existing
    // sequence instead of a fresh one.
    self.session.append_borrowed_chunks_cow(
        sequence_id.0,
        &entry.chunk_ids,
        &vec![target_pos as i32; entry.chunk_ids.len()],
        entry.token_count,
    )?;

    let new_pos = target_pos + entry.token_count;
    Ok(new_pos)
}
```

### 6.3 `append_borrowed_chunks_cow` on Chunked Backing

New method on the chunked backing that appends borrowed chunk references to an
existing sequence's block table. Unlike `inject_prefix_chunks` (which operates
on a fresh, empty slot), this method extends an existing slot that already has
content.

```rust
pub fn append_borrowed_chunks_cow(
    &mut self,
    batch_idx: usize,
    chunk_ids: &[i64],
    rope_shifts: &[i32],
    token_count: usize,
) -> Result<()> {
    // Validate slot is allocated
    // Determine starting block index from current block_count
    // For each chunk_id:
    //   - Look up the ChunkHandle from global registry
    //   - Arc-clone and store ChunkRef::with_rope_shift in refs[start + i]
    //   - Write chunk_id into block table host[batch * max_blocks + start + i]
    // Advance block_count and seq_len
    // If last chunk is partially filled:
    //   - Mark it for CoW (on first decode write, allocate fresh block,
    //     copy token_count % chunk_size positions, replace borrowed ref)
    // NOTE: seq_start_offset is NOT changed — it was set at sequence creation
    // and applies to the entire sequence uniformly.
}
```

**Layout:** Boundary prototypes are generated with `sso = 0` (left-packed), so
within-block positions match what the host sequence kernel expects when injected
mid-sequence.

**CoW on the trailing partial block.** Boundaries shorter than `chunk_size` have
one partial block. The first decode write triggers CoW: allocate a fresh block,
copy the partial KV, replace the borrowed ref. For a 12-token boundary this is
768 KB of memcpy -- trivial compared to a forward pass.

## 7. Code Changes

### 7.0 Boundary KV Quantization Bypass

Boundary chunks must remain in **full-precision float (BF16/F16)** rather than
being quantized to Q8_0/Q4_0. Rationale:

1. **Maximum model quality.** Boundary KV is already an approximation (§2).
   Adding quantization error on top of context approximation error compounds
   quality loss. Keeping boundaries at full precision eliminates one source
   of error entirely.

2. **Negligible VRAM cost.** Boundary chunks are massively deduplicated — every
   conversation shares the same physical chunks via Arc refcount. The total
   VRAM for all boundary prototypes is a fixed constant regardless of batch
   size: ~3 boundaries × ~1 chunk each × `chunk_size × head_dim × n_kv_heads ×
   2 (K+V) × n_layers × sizeof(BF16)` ≈ 768 KB for a 32-layer model. This is
   dwarfed by the arena pool for active sequences.

3. **Same rationale as system prompt BF16.** The existing codebase already uses
   BF16 for system prompt KV (`node.rs` L244–253) because it's the
   most-attended tensor. Boundary chunks follow the same principle.

**Primary mechanism: explicit `no_quantize` flag on `SlotState`**

Add a `no_quantize: bool` field to `SlotState` in `types.rs`:

```rust
pub(super) struct SlotState {
    pub(super) block_count: usize,
    pub(super) refs: Vec<Option<ChunkRef>>,
    pub(super) free_chunks: Vec<i64>,
    pub(super) seq_start_offset: usize,
    /// When true, reconcile skips this slot entirely — used for prototype
    /// sequences whose chunks must remain at full float precision.
    pub(super) no_quantize: bool,
}
```

The reconcile loop in `chunk_ops.rs` checks this before any block work:

```rust
// Hard guard: prototype sequences are never quantized
if slot.no_quantize {
    return Ok(0);
}
// Existing COW guard for borrowed chunks
if let Some(ref chunk_ref) = slot.refs[blk] {
    if chunk_ref.is_shared() {
        continue;
    }
}
```

Set `no_quantize = true` at prototype sequence creation inside
`generate_static_chunks_with_context()`:

```rust
let raw_id = self.proto_session.create_sequence()?;
self.proto_session.set_no_quantize(raw_id, true)?;
```

Why explicit rather than relying on `is_shared()` alone:

- **`is_shared()` cannot guard the prototype's own copy.** Before any loan is
  issued, `Arc::strong_count == 1`, so `is_shared() == false`. A reconcile pass
  running between the prototype prefill and the first `loan()` call would
  silently quantize the chunks.
- **`is_shared()` is an incidental property, not intent.** Refcount is a
  sharing detail; `no_quantize` documents that this sequence exists precisely to
  hold permanent float KV. The code is self-documenting and immune to future
  refcount changes.

**Secondary mechanism: float-only prototype session (belt-and-suspenders)**

The prototype session is also created with `StoragePolicy::GpuFloat(BF16)`:

```rust
let proto_session = BatchedInferenceSession::new_with_format(
    &model,
    config,
    KvFormat::Float(DType::BF16),
)?;
```

With a float-only `StoragePolicy`, `reconcile` is a no-op even without
`no_quantize` because source and target format are identical. The two
mechanisms stack: `no_quantize` is checked first (fast path, always
explicit), and the float session ensures correct arena type. If either guard
alone misfires, the other still prevents quantization.

**Third layer: COW protection for borrower sequences**

Borrowed boundary chunks always have `Arc::strong_count > 1` (prototype +
borrower), so `chunk_ref.is_shared() == true`. The reconcile loop already
skips them regardless of the borrower session's `StoragePolicy`. This means
borrower sequences do not need any special treatment — it is handled
automatically.

**Implementation:**

```rust
// At engine startup, create a dedicated float session for prototype generation
let proto_session = BatchedInferenceSession::new_with_format(
    &model,
    config,
    KvFormat::Float(DType::BF16),  // No quantization (float session)
)?;

// Generate all boundary prototypes in this float session
for (key, tokens, is_pos_zero, context_tokens) in resolver.generation_plan(&system_prompt, &tokenizer) {
    let result = self.generate_static_chunks_with_context(
        &context_tokens,
        &tokens,
        is_pos_zero,
    )?;
    cache.store(key, result);
}
// proto_session lives for the lifetime of the engine (chunks must persist)
```

### 7.1 `StaticChunkCache` — Contextual Boundary Entry Generation

**File:** `candle-conversation/src/static_chunk_cache.rs`

Replace isolated prototype generation with contextual multi-pass generation.
The `BoundaryResolver` provides a `generation_plan()` method that returns
each boundary's tokens along with its required preceding context:

```rust
impl BoundaryResolver {
    /// Returns generation plan: (ChunkKey, boundary_tokens, context_tokens).
    /// context_tokens is the representative preceding context for contextual generation.
    /// Empty for DocumentSystemStart (position 0 � exact injection, no preceding context).
    pub fn generation_plan(
        &self,
        system_prompt: &str,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Vec<(ChunkKey, Vec<u32>, Vec<u32>)> {
        let mut plan = Vec::new();
        let dialect = &self.dialect;

        for (&boundary, entry) in &self.entries {
            let context_tokens = match boundary {
                Boundary::DocumentSystemStart => {
                    // Position 0 — no context needed (exact injection)
                    vec![]
                }
                Boundary::SystemToUser => {
                    // Needs: document_start + system_start + system_content
                    let ctx = format!(
                        "{}{}{}",
                        dialect.document_start,
                        dialect.system_start,
                        system_prompt,
                    );
                    self.tokenize_context(&ctx, tokenizer)
                }
                Boundary::UserToAssistant => {
                    // Needs: full system prompt + user_start + representative user msg
                    let ctx = format!(
                        "{}{}{}{}Hello",
                        dialect.document_start,
                        dialect.format_system_prompt(system_prompt),
                        dialect.system_end,
                        dialect.user_start,
                    );
                    self.tokenize_context(&ctx, tokenizer)
                }
                Boundary::AssistantToUser => {
                    // Needs: system + user + assistant_start + representative response
                    let ctx = format!(
                        "{}{}{}{}Hello{}{}I'd be happy to help.",
                        dialect.document_start,
                        dialect.format_system_prompt(system_prompt),
                        dialect.system_end,
                        dialect.user_start,
                        dialect.user_end,
                        dialect.active_assistant_start(
                            self.suppress_thinking,
                            dialect.supports_no_think(),
                        ),
                    );
                    self.tokenize_context(&ctx, tokenizer)
                }
            };

            plan.push((
                entry.key.clone(),
                entry.tokens.clone(),
                context_tokens,
            ));
        }
        plan
    }

    /// Tokenize context tokens without padding.
    /// Alignment to block boundaries is handled in `generate_static_chunks_with_context`
    /// via `write_offset_shift` rather than by appending filler tokens.
    fn tokenize_context(
        &self,
        text: &str,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .expect("tokenize context")
            .get_ids()
            .to_vec()
    }
}
```

**File:** `candle-conversation/src/scheduler/mod.rs`

New generation method that prefills context then boundary:

```rust
fn generate_static_chunks_with_context(
    &mut self,
    context_tokens: &[u32],
    boundary_tokens: &[u32],
) -> Result<GenerateStaticResult, ConversationError> {
    let raw_id = self.proto_session.create_sequence()
        .map_err(ConversationError::Model)?;

    if context_tokens.is_empty() {
        // Position-0: no context, right-packed as before
        let sso = self.chunk_size.saturating_sub(boundary_tokens.len());
        let input = Tensor::new(boundary_tokens, &self.device)?.unsqueeze(0)?;
        self.model.forward_batched_with_write_shifts(
            &mut self.proto_session, &[raw_id], &[input], &[sso as u32],
        )?;
        self.proto_session.advance_sequence(raw_id, boundary_tokens.len())?;
    } else {
        // Mid-sequence: prefill context first, then boundary.
        let ctx_input = Tensor::new(context_tokens, &self.device)?.unsqueeze(0)?;
        self.model.forward_batched(
            &mut self.proto_session, &[raw_id], &[ctx_input],
        )?;
        self.proto_session.advance_sequence(raw_id, context_tokens.len())?;

        // Block close-off: compute the write_offset_shift that causes boundary
        // token 0 to land at physical position `next_block_start`, i.e., within
        // slot 0 of block `ceil(C / chunk_size)`.  No padding tokens are
        // generated; the gap positions (C .. next_block_start-1) are never
        // written and are outside the causal window (seq_len = C at this point).
        let c = context_tokens.len();
        let context_blocks = c.div_ceil(self.chunk_size);
        let next_block_start = context_blocks * self.chunk_size;
        let boundary_write_shift = (next_block_start - c) as u32;

        // Ensure blocks are allocated for the write-shifted range before the
        // kernel runs.  The shift can extend writes up to one block past what
        // offset-based allocation would predict.
        self.proto_session.ensure_capacity(
            &[raw_id],
            boundary_write_shift as usize + boundary_tokens.len(),
        )?;

        // Boundary tokens attend to the full context via causal mask.
        let bnd_input = Tensor::new(boundary_tokens, &self.device)?.unsqueeze(0)?;
        self.model.forward_batched_with_write_shifts(
            &mut self.proto_session,
            &[raw_id],
            &[bnd_input],
            &[boundary_write_shift],
        )?;
        // Advance by boundary_tokens.len() only — the gap is never in the
        // logical sequence, so the offset does not include it.
        self.proto_session.advance_sequence(raw_id, boundary_tokens.len())?;
    }

    // Extract only the boundary chunks (skip context blocks).
    // context_blocks == 0 when context is empty (DocumentSystemStart), so all chunks are extracted.
    // For mid-sequence entries, context_blocks = ceil(C / chunk_size).
    let all_chunk_ids = self.proto_session.slot_chunk_ids(raw_id)?;
    let context_blocks = context_tokens.len().div_ceil(self.chunk_size);
    let boundary_chunk_ids = all_chunk_ids[context_blocks..].to_vec();

    Ok(GenerateStaticResult {
        sequence_id: SequenceId(raw_id),
        chunk_ids: boundary_chunk_ids,
    })
}
```

### 7.2 `Conversation::initial_handle` — Split System Prefill (Phase 1)

**File:** `candle-conversation/src/conversation.rs`

**Before:**
```rust
let text = format_system_prompt(dialect, system_prompt, suppress_thinking);
let text = format!("{}{}{}", text, dialect.turn_end, dialect.user_start);
let tokens = self.tokenize(&text)?;
// → Single Prefill of everything
```

**After:**
```rust
// Step 1: Inject Boundary::DocumentSystemStart at position 0 (exact)
self.scheduler_tx.send(SchedulerRequest::InjectBoundary {
    sequence_id,
    boundary: Boundary::DocumentSystemStart,
    target_pos: 0,
    reply,
})?;

// Step 2: Prefill system prompt content only (dynamic)
let content_tokens = self.tokenize(&system_prompt_content)?;
self.scheduler_tx.send(SchedulerRequest::Prefill {
    tokens: content_tokens, ...
})?;

// Step 3: Inject Boundary::SystemToUser (contextual-approximate)
let target_pos = self.current_seq_len();
self.scheduler_tx.send(SchedulerRequest::InjectBoundary {
    sequence_id,
    boundary: Boundary::SystemToUser,
    target_pos,
    reply,
})?;
```

### 7.3 `Conversation::finish_turn` — Inject Instead of Prefill (Phase 1)

**File:** `candle-conversation/src/conversation.rs`

**Before:**
```rust
let text = format!("{}{}", turn_end, user_start);
let tokens = self.tokenize(&text)?;
// → Prefill request
```

**After:**
```rust
let target_pos = self.current_seq_len();
// → InjectBoundary { boundary: Boundary::AssistantToUser, target_pos }
//   No forward pass — resolver maps to the correct dialect tokens internally.
```


### 7.4 (Phase 2) `SubmitTurnSplit`

Deferred to Phase 2. See �8 and �9. `submit_turn` keeps the single-pass full
prefill (user tokens + boundary tokens) in Phase 1.

## 8. The Logits Problem

Injecting boundary KV without a forward pass produces no logits. The decode loop
needs logits from the last boundary token to sample the first generated token.

**Resolution:** Use injection only where no logits are needed. For `submit_turn`,
keep the full single-pass prefill (user tokens + boundary tokens) -- saving 5-15
tokens in a prefill pass is negligible and split-submission adds complexity not
worth taking in Phase 1.

Phase 1 injection points are all logits-free: `initial_handle` (steps 1 and 3)
and `finish_turn`. No decode follows injection at these points.

## 9. Scope

### Phase 1
- `Boundary` enum + `BoundaryResolver`
- Contextual prototype generation with `write_offset_shift` block close-off
- `no_quantize` flag on `SlotState` + float prototype session
- `append_borrowed_chunks_cow` with CoW on trailing partial block
- `InjectBoundary` scheduler request + handler
- Wire `initial_handle`: inject `DocumentSystemStart`, prefill system content, inject `SystemToUser`
- Wire `finish_turn`: inject `AssistantToUser`
- Opt-in flag: `ConversationConfig::use_boundary_injection: bool` (default false)

### Phase 2 (if Phase 1 validates)
- `Boundary::UserToAssistant` in `submit_turn` via `SubmitTurnSplit`
  (requires single-token logits pass � see �8)
- Thinking block injection automatic (folded into `UserToAssistant` by resolver)

### Out of Scope
- System prompt content caching; cross-conversation KV sharing

## 10. Testing

### Unit Tests (no model -- CI on every commit)

**`BoundaryResolver`** (`candle-conversation/src/config/`)
- All 3 dialects x 2 thinking modes: non-empty plan, distinct `ChunkKey`s per entry
- `DocumentSystemStart` has empty context tokens (position 0, exact injection)
- `SystemToUser`/`UserToAssistant`/`AssistantToUser` have non-empty context tokens
- Identical token sequences collapse to one entry (deduplication)
- `generation_plan(system_prompt, tokenizer)` returns `(ChunkKey, tokens, context_tokens)` � no bool in tuple

**`no_quantize` flag** (`candle-nn/src/kv_cache/chunked/tests/`)
- `no_quantize = true` skips reconcile regardless of storage policy
- `no_quantize = false` reconciles normally
- All three guards (float session, `no_quantize`, `is_shared()`) verified independently

**`generate_static_chunks_with_context`** (CPU mock model)
- First extracted chunk ID starts at `context_blocks` (not `context_blocks - 1`)
- Gap positions `context_len..context_blocks*chunk_size` are never written
- `sequence_length == context_len + boundary_len` (gap excluded from logical length)
- Empty context tokens (i.e., `DocumentSystemStart`) gives `seq_start_offset = chunk_size - M`, single chunk
- Extracted chunk count equals `boundary_len.div_ceil(chunk_size)` for all context sizes

**`append_borrowed_chunks_cow`** (`candle-nn/src/kv_cache/chunked/tests/`)
- Block table and `block_count` correct after append
- `chunk_rope_shifts` set to expected rope offset for injected blocks
- CoW fires on first decode write to borrowed block; original KV unchanged
- CoW does NOT fire on writes to unshared blocks
- `sequence_length` advances by `token_count`, not by `blocks * chunk_size`

### Integration Tests (requires `Qwen2_0_5B` weights -- `#[ignore]` gated)

**Startup generation** (`candle-conversation/tests/`)
- All expected `Boundary` variants present in cache after startup
- `DocumentSystemStart` prototype: `seq_start_offset = chunk_size - M`, block 0
- Mid-sequence entries: first boundary token lands at a `chunk_size` multiple
- Prototype chunks are BF16 (no quantization)

**Scheduler `InjectBoundary`**
- Request dispatched; sequence offset advances by `token_count`
- Returns `Err` if cache entry is missing (startup assertion: cache must be fully populated before first use)
- Generates tokens normally after injection (no attention corruption)

**`initial_handle` / `finish_turn`**
- `initial_handle`: block 0 = `DocumentSystemStart` chunk; post-system block = `SystemToUser` chunk
- `finish_turn`: turn boundary block = borrowed `AssistantToUser` chunk

### Output Consistency (nightly, greedy sampling)

- KL divergence (injected vs. non-injected first-token logits) < 0.01 nats per boundary
- First 20 greedy tokens identical for single-turn: injection on vs. off
- First 10 greedy tokens identical per turn for 5-turn scripted conversation
- All 3 dialects tested; divergence logged by turn and token position

### Benchmarks (pre-release)

- `finish_turn` injection vs. prefill at batch sizes [1, 8, 32, 128]: median + p99 latency
- End-to-end tokens/sec for 20-turn conversation at batch 64: injection on vs. off

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Output quality degradation | Low | High | Contextual generation (§2.3), explicit `no_quantize` + BF16 session (§7.0), opt-in flag, greedy consistency test |
| Drift over many turns | Low | Medium | Multi-turn drift test, monitoring |
| CoW complexity bugs | Low | Medium | Unit tests, existing CoW patterns |
| Block fragmentation waste | Low | Low | 27 wasted slots per injection is ~0.1% of typical sequence |
| Logits problem blocks phase 2 | High | Low | Phase 1 (finish_turn only) doesn't need logits |
| Prototype VRAM from context blocks | None | None | ~1 MB total for all boundaries — negligible |

## 12. Implementation Order

1. **`Boundary` enum + `BoundaryResolver`** — implement §3.4–3.5 including
   `generation_plan()` method that produces raw (unpadded) context tokens per
   boundary. Unit test that resolver produces correct token sequences for all
   3 dialects � thinking modes, including deduplication; panics on zero-token
   boundary as a configuration check.
2. **`no_quantize` flag + float prototype session** — add `no_quantize: bool`
   to `SlotState`, check in reconcile loop (§7.0). Create a dedicated
   `BatchedInferenceSession` with `KvFormat::Float(BF16)`. Verify reconcile
   is a no-op and that prototype chunks stay at BF16 even before first loan.
3. **`generate_static_chunks_with_context`** — new scheduler method (§7.1) that
   prefills context then boundary in two passes using `write_offset_shift` to
   close off the partial context block; extracts only boundary chunk IDs.
   Unit test that boundary KV reflects causal attention to context and that
   no padding tokens appear in the sequence.
4. **`append_borrowed_chunks_cow`** on chunked backing (+ unit tests for
   block table writes, rope shift, CoW on trailing partial block).
5. **`InjectBoundary` scheduler request + handler** — resolve `Boundary` →
   `ChunkKey` via `BoundaryResolver`, delegate to `append_borrowed_chunks_cow`.
6. **Boundary entry generation at startup** — pass `BoundaryResolver` +
   system prompt to `StaticChunkCache`. Position-0 entries (`DocumentSystemStart`)
   right-packed via `generate_static_chunks_with_context` with empty context.
   Mid-sequence entries (`SystemToUser`, `AssistantToUser`) generated with
   context close-off via `write_offset_shift` (§7.1). Deduplication handled by
   `cache_entries()`.
7. **Wire `initial_handle`** — inject `Boundary::DocumentSystemStart` at
   position 0 (exact), prefill system prompt content, then inject
   `Boundary::SystemToUser` (contextual-approximate). No fallback branch �
   all non-empty boundaries are always present in the resolver.
8. **Wire `finish_turn`** — inject `Boundary::AssistantToUser` when
   `use_boundary_injection = true`, else fall back to normal prefill.
9. **Unit tests** — §10.1 (`BoundaryResolver`), §10.2 (`no_quantize`), §10.3
   (write_offset_shift generation), §10.4 (`append_borrowed_chunks_cow`).
   All run without a loaded model; target is CI green on every commit.
10. **Integration + wiring tests** — §10.5 (scheduler), §10.6 (startup cache),
    §10.7 (`initial_handle` / `finish_turn`). Require `Qwen2_0_5B` weights.
11. **Output consistency tests** (§10.8) — greedy-match and KL divergence pass
    before enabling by default.
12. **Performance benchmarks** (§10.9) — run manually before each release.
13. **Decision point**: enable by default or keep opt-in based on §10.8 results.
