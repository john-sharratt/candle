# Stencil Tree — Schema-Guided Constrained Decoding

**Status:** implemented (this doc captures the original design; see *As-built
deltas* below for what shipped beyond it).
**Module:** `candle-conversation/src/stencil/` (self-contained, 100% unit-tested
without a GPU; only the final end-to-end check needs hardware).
**Touch-points:** a sample-stage mask hook in `batched_sampler.rs` and a
per-sequence step hook in `scheduler/decode.rs`. Nothing in the KV cache,
projection, or persistence paths changes.
**Primary use case:** forcing a tool call to a tool catalog's exact names and
JSON schema.

### As-built deltas (beyond this design)

- **A fourth front-end, D — think steering** (`think.rs`, `ThinkMode`,
  `compile_think_tree`): per-effort-dial trees that steer the `<think>` block with
  an opener + continuation phrases, closed via token-closed `suppress_close` spans.
  §8's "three front-ends" is now four.
- **`StencilDriver`** (`driver.rs`): the online walker the scheduler drives one
  decoded token at a time (the `Healed` signal contract with `scheduler/decode.rs`).
- **`TriggerRegistry`** (`trigger.rs`): token → tree dispatch with atomic,
  idempotent `with_trigger` / `without_trigger`, used to swap the per-turn think
  tree onto the tool-call base.
- The `§15` module layout and the `§13` public-API sketch predate `driver.rs`,
  `think.rs`, and `trigger.rs` and are illustrative, not exhaustive.

---

## 1. Motivation

A tool call must be **exactly** valid: the tool name must be one of the
registered tools, the JSON must parse, every required parameter must be present,
no hallucinated parameters may appear, and enum-valued parameters must be one of
the allowed strings. A freely-decoded model — even one trained on tool use,
like Qwen3 — drifts: it invents a plausible-but-wrong tool name, forgets a
required field, emits a trailing comma, closes a brace early, or spells an enum
value differently. Recovering after the fact (parse → detect → retry) wastes a
whole generation and can loop.

The **stencil tree** exploits one observation: a tool call is mostly **rigid
scaffolding** (`{"name": "`, `", "arguments": {`, `}}`, quotes, commas, keys)
with a few genuine **decision points** (which tool, which enum value, whether to
include an optional field) and a few **free spans** (string and number
*values*). So:

- **Prefill the rigid scaffolding.** The tokens are known, so we inject them in
  one forward pass instead of decoding them one at a time. This is *guaranteed*
  speculative decoding — the "speculation" is the stencil, which is always
  right — so the structural tokens are not just correct but **faster** than free
  decode.
- **Constrain the sampler only at decision points.** A handful of masked
  decodes, not one per token.
- **Let the model decode values freely**, watching for the terminator that ends
  the span — escape- and nesting-aware (§6) — with optional soft/hard limits
  (§6.3) so a value can't ramble forever.

The name is the mental model: a stencil is solid where the pattern is fixed and
cut away where paint goes through. The **solid** parts are prefilled; the
**holes** are where the model decodes (branches and free-text).

### Non-goals

- A general context-free-grammar engine. The tree is expressive enough for the
  tool-call / JSON-schema shape and composes recursively over it.
- Changing what the model *wants* to say. The stencil enforces *form*, not
  *content*: the model still chooses the tool, the values, and which optionals
  to include.

---

## 2. Concepts and vocabulary

| Term | Meaning |
|------|---------|
| **Stencil tree** | An immutable compiled tree of nodes alternating *static* runs, *branches*, and *free-text* spans, ending at a *terminal*. Built once per `(tokenizer, source spec)`. |
| **Trigger** | A token (e.g. `<tool_call>`) that, emitted during free decode, *enters* a tree at its root. |
| **Stencil session** | The per-sequence runtime walker: a single cursor into one tree. No side buffers (§4.1). |
| **Stencil (sampling) mask** | A logit mask applied at the sample stage restricting the next token to an allowed set. |
| **Guided phase** | Walking static runs and branches — the model is being steered. |
| **Free-text phase** | Inside a `FreeText` span — the model decodes unconstrained until the terminator fires. |
| **Terminator** | A small, escape- and nesting-aware byte lexer that ends a free-text span (§6). |

The whole feature is a **decode-time** construct. It produces ordinary tokens
that seal into the substrate exactly like any other decoded tokens (§11).

---

## 3. The augmented decode loop

The scheduler's decode loop today is, per sequence: forward pass → sample →
append token → health/EOS checks → repeat. The stencil adds one hook after the
token is sampled, plus a small inner loop that drains atomic prefills:

```text
// per sequence, each outer decode iteration:
token = decode_step(seq, seq.next_mask)     // forward + sample (mask if Some)
seq.health.observe(token)
seq.stencil.observe(token)                  // advance cursor on the decoded token

loop {                                       // drain non-decoding actions
    match seq.stencil.next_action() {
        Free               => break,         // not stenciling; also scan triggers
        Prefill(static_buf)=> decode_prefill(seq, static_buf),   // inject; loop
        MaskedDecode(set)  => { seq.next_mask = Some(set); break }
        FreeDecode(boost)  => { seq.next_mask = None; seq.eos_bias = boost; break }
        Exit               => { seq.stencil.clear(); break }
    }
}
```

Key property — **prefills are atomic and self-contained**: a `Static` node owns
its token buffer (§4.1), so `Prefill` injects that one buffer and the inner loop
immediately re-queries `next_action`. Consecutive structure never accumulates in
the session; the compiler already fuses adjacent static runs into single
maximal nodes (§8.1), so the tree strictly alternates `Static ↔ (Branch |
FreeText)` and each prefill is one node. No `pending` buffer exists.

Entering: in the `Free` state, `observe` checks the decoded token against the
**trigger registry** (§9). A hit begins a session at the tree root; the trigger
token has already been emitted, so walking starts after it.

`Prefill` is a single forward pass over the node's whole buffer (the existing
`run_prefill`), not a decode per token — this is the §1 speedup.

---

## 4. Node model

A compiled tree is a flat arena addressed by `NodeId`; sessions share the tree by
`Arc`, so 64 concurrent sessions are nearly free.

```rust
pub struct StencilTree { nodes: Vec<StencilNode>, root: NodeId, label: String }
#[derive(Clone, Copy)] pub struct NodeId(u32);

pub enum StencilNode {
    /// A fixed token run that owns its buffer. Prefilled atomically; exactly one
    /// successor. The compiler guarantees no two `Static` nodes are adjacent.
    Static { tokens: Vec<TokenId>, next: NodeId },

    /// A constrained choice. The sampler is masked to the live frontier of
    /// `trie`; successive masked decodes walk it to an accepting node, whose
    /// `next` is taken. Single-arm "branches" never reach here — folded to
    /// `Static` (§8.1).
    Branch { trie: TokenTrie },

    /// Unconstrained decode until `term` fires (§6). Optional soft/hard limits
    /// and an optional EOS exit, mirroring the sampler's graceful/forced EOS.
    FreeText(FreeTextSpan),

    /// Leave stencil mode; resume normal free decode.
    End,
}

pub struct FreeTextSpan {
    /// Escape- and nesting-aware terminator (§6).
    term: Terminator,
    /// When set, an EOS sample also ends the span (for free prose with no
    /// natural delimiter). When unset, EOS is masked out for the span's
    /// duration — the span must end via `term`.
    eos_ends: bool,
    /// Span-scoped limits mirroring `SamplingConfig` (§6.3). `None` soft = no
    /// ramp; `hard` is always present (the runaway guard).
    limits: FreeTextLimits,
    next: NodeId,
}
```

### 4.1 No session-side buffering

The session is just a cursor:

```rust
pub struct StencilSession {
    tree: Arc<StencilTree>,
    cursor: Cursor,
}
enum Cursor {
    At(NodeId),
    InBranch { pos: TrieNodeId },
    InFreeText { node: NodeId, term: TerminatorState, emitted: u32 },
    Done,
}
```

There is no `pending: Vec<TokenId>`. A `Static` node carries its own buffer; the
session emits `Prefill(&node.tokens)` and advances to `next` in one step. This is
what the §3 inner loop drains: prefill a node, loop, prefill the next… except the
compiler's fusion (§8.1) means a `Static` is always followed by a `Branch`,
`FreeText`, or `End`, so the loop does exactly one prefill before a decode or
exit. Atomic and clean.

---

## 5. Walking the tree

Two methods drive it: `next_action()` (what to do at the current cursor, without
consuming a token) and `observe(token)` (consume a decoded token, advance).

```text
fn next_action() -> StencilAction:
    match cursor:
        At(node):
            match tree[node]:
                Static{tokens, next} -> cursor = At(next); Prefill(tokens)
                Branch{trie}         -> cursor = InBranch{trie.root};
                                        MaskedDecode(frontier(trie.root))
                FreeText(span)       -> cursor = InFreeText{node, term: span.term.start(), 0};
                                        FreeDecode(span.eos_bias(0))   // §6.3
                End                  -> cursor = Done; Exit
        InBranch{pos}    -> MaskedDecode(frontier(pos))
        InFreeText{..}   -> FreeDecode(span.eos_bias(emitted))
        Done             -> Exit

fn observe(token):
    match cursor:
        InBranch{pos}:
            child = trie[pos].edge_to(token)        // guaranteed legal (masked)
            if trie[child].is_leaf(): cursor = At(trie[child].next)
            else:                     cursor = InBranch{child}
        InFreeText{node, term, emitted}:
            emitted += 1
            match term.feed(token):                 // §6 byte lexer
                Open  -> stay                        // span continues
                Close{consumed} -> cursor = At(span.next)   // (heal if mid-token §7.3)
                Eos   if span.eos_ends -> cursor = At(span.next)
            apply_limits(span, emitted)              // §6.3 soft/hard
        _ -> /* tokens only arrive after MaskedDecode / FreeDecode */
```

### 5.1 At a branch

A `Branch`'s decode was masked to `frontier(pos)` (the sorted edge list on the
trie node — a slice, not a computation), so the decoded token is always a legal
edge. `observe` descends one edge; a leaf transitions to the arm's `next`. A node
that both accepts and has further edges (a name that is a prefix of another) puts
the accept transition in the frontier as well; in the tool-call grammar this
never arises because every arm is followed by a fixed delimiter (`"`) that
disambiguates.

### 5.2 Worked trace — `read_file({"path":"src/main.rs"})`

Catalog: `read_file(path: string)`, `write_file(path, content)`,
`list_dir(path)`. Qwen3 envelope:
`<tool_call>\n{"name": "<NAME>", "arguments": <ARGS>}\n</tool_call>`.

```text
free decode … "<tool_call>"          TRIGGER → begin(root)
  Static  '\n{"name": "'             Prefill(6)            ; loop
  Branch  tool-name trie             MaskedDecode{read,write,list}
  decode -> "read_file"  (one masked step; '_file' was a single child → folded
            into the arm's static suffix, so the arm completes here)
  Static  '", "arguments": {"path": "'   Prefill(N)        ; loop
  FreeText JsonString                FreeDecode
  decode -> "src" / "/main" / ".rs"  lexer: no close       (3 free decodes)
  decode -> "\""                     lexer: unescaped '"' → close ; advance
  Static  '}}\n</tool_call>'         Prefill(M)            ; loop
  End                                Exit
free decode resumes …
```

---

## 6. Terminators — escape, nesting, and limits

The terminator is the brain of the free-text phase. It must be **escape-aware**
(a `\"` inside a string does not close it), **nesting-aware** (a value that is
itself a structured object/array closes only at the matching bracket at depth 0,
and brackets *inside* nested strings don't count), and it must offer **soft/hard
limits** so a value can be forced to end. It runs over decoded **bytes**, not
token identity (§7.3), which is what makes it robust to tokenization.

### 6.1 The terminator kinds

```rust
pub enum Terminator {
    /// JSON string value. Ends at the first UNESCAPED `"`. A `\` escapes the
    /// next byte (`\"`, `\\`, `\n`, …). The closing quote is consumed.
    JsonString,

    /// JSON number value. Lookahead-terminated: ends at the first byte that
    /// can't extend a number (digit, '-', '+', '.', 'e', 'E'; `integer_only`
    /// drops '.'/'e'). The terminator byte is NOT consumed — it belongs to the
    /// following static run (a ',' or '}').
    JsonNumber { integer_only: bool },

    /// A balanced `open`/`close` structure — a raw object/array value. Ends when
    /// nesting **depth returns to 0**. STRING-AWARE: a `"…"` opened inside
    /// suspends bracket counting, and within it `\` escaping applies, so a `}`
    /// or `]` inside a string never affects depth. The closing bracket is
    /// consumed. This is the "handle nesting" case.
    Balanced { open: u8, close: u8 },
}
```

### 6.2 The lexer state

```rust
pub struct TerminatorState {
    kind: Terminator,
    depth: u32,        // bracket nesting depth (Balanced)
    in_string: bool,   // inside a nested "…" (Balanced / string-aware)
    escaped: bool,     // previous byte was an unconsumed backslash
}

pub enum Feed { Open, Close { byte_in_token: usize }, Eos }
```

`feed(token)` decodes the token to bytes and runs the machine byte-by-byte:

- **`JsonString`**: `if escaped { escaped=false } else if b=='\\' { escaped=true }
  else if b=='"' { return Close }`.
- **`Balanced{open,close}`**: maintain `in_string`/`escaped` exactly as above for
  any `"` seen; while **not** `in_string`, `b==open → depth+=1`,
  `b==close → { depth-=1; if depth==0 return Close }`. The first `open` is either
  part of the preceding static run (depth starts at 1) or the span's first byte
  (depth starts at 0 then →1); the compiler picks one convention and records the
  starting depth on the span.
- **`JsonNumber`**: `if !is_number_byte(b) { return Close{byte_in_token: i} }` —
  the close byte is *not* consumed (lookahead); §7.3 push-back applies.

Escaping and nesting are therefore handled in one ~20-line machine, shared by all
three kinds, with no regex engine.

### 6.3 Soft / hard limits (mirrors the sampler's EOS ramp)

`FreeTextLimits` scopes the sampler's whole-turn EOS mechanism
(`SamplingConfig::{eos_boost, eos_ramp_start, eos_ramp_len, graceful_eos_after,
forced_eos_after}`, see `batched_sampler.rs`) to a single span, closing with the
span's terminator (or EOS when `eos_ends`) instead of the turn's EOS:

```rust
pub struct FreeTextLimits {
    /// Below this many span tokens, no pressure. At/after it, ramp a boost onto
    /// the span's CLOSE token (the quote, the ']'/'}' , or EOS when eos_ends),
    /// linearly to `ramp_len`. Mirrors `eos_ramp_start` / `eos_boost`.
    ramp_start: Option<u32>,
    ramp_len: u32,
    boost: f32,
    /// Hard: at this many tokens, force-close unconditionally — emit the
    /// terminator's canonical close tokens (healing if needed, §7.3) and advance.
    /// The runaway backstop; always set. Mirrors `forced_eos_after`.
    forced_after: u32,
}
```

- **Ramp** (`FreeDecode(boost)` in §5): the action carries a per-step boost the
  sampler adds to the close token's logit — a *soft* nudge that lets the model
  close naturally, exactly like `eos_boost`. Zero before `ramp_start`.
- **Forced**: at `forced_after`, the session injects the close tokens and
  advances regardless. Bounded; never infinite.
- **`eos_ends`**: when set, an EOS sample is a legal close (and is what the ramp
  boosts); when unset, EOS is masked out for the span so it cannot end early.

A pure JSON string value typically sets `forced_after` only (a high runaway
guard) and leaves the rest `None` — the model reliably emits its own closing
quote. A free-prose span (e.g. a tool's `reason` field) sets the full ramp +
graceful so it wraps up like a normal turn.

---

## 7. The sampling stencil and tokenization

### 7.1 The mask

`MaskedDecode(set)` sets every disallowed logit to `-inf` **after** penalties and
**before** temperature/top-p, so the model's relative preference among legal
tokens survives and a penalty can never resurrect an illegal token or override
the stencil. Per-sequence in the wave batch (`seq -> Option<AllowedSet>`); rows
with `None` decode normally. A one-token mask never occurs (folded to `Static`,
§8.1); an empty mask is a compile-time invariant violation (`debug_assert`).

### 7.2 Static runs and tries are tokenized in context

A `Static` run's `tokens` are not `encode(piece)` in isolation — the compiler
tokenizes the **canonical full call** and *segments* it, so boundary merges
(`{"`, `": "`, `"}`) match what a correct free decode produces. Branch tries are
built from each arm **in its envelope position** (`{"name": "read_file"` → take
the slice after `"`), so shared tokenized prefixes share trie paths and the
frontier is exactly the set the model could legally pick. The tree carries a
tokenizer fingerprint; a mismatch at load fails loudly.

### 7.3 The free→guided boundary: token healing

The dangerous seam is the end of a free span: the model may merge the closing
delimiter with following bytes (`rs"}` as one token). Because the terminator
lexer works on **bytes** (§6), it detects the close *inside* the token and
reports `byte_in_token`. If the close is mid-token, the session **heals**:
truncate the KV by that one token (`truncate_sequence_to_tokens`), and re-tokenize
the value tail + the stencil's close run as one canonical run, then prefill it.
For `JsonNumber` the same machinery handles lookahead push-back. Healing is rare
(models emit the quote as its own token) but must be correct; §12 turns
"healing impossible" into a clean abort, never corruption.

---

## 8. Construction — three front-ends, one backend

Every tree is a `TreeSpec` (string-space, untokenized) compiled by a single
backend. This keeps the three required construction methods thin and guarantees
they produce identical trees for identical logical input.

```rust
// String-space intermediate — what every front-end produces.
pub struct TreeSpec { nodes: Vec<NodeSpec>, root: NodeRef, label: String }
pub enum NodeSpec {
    Static(String, NodeRef),
    Branch(Vec<(String, NodeRef)>),                 // (arm literal, successor)
    FreeText { term: Terminator, eos_ends: bool,
               limits: FreeTextLimits, next: NodeRef },
    End,
}

/// The one backend: tokenize-in-context, build tries, FOLD single-arm branches
/// to Static, FUSE adjacent Static runs into maximal nodes, verify invariants.
pub fn compile(spec: TreeSpec, tok: &Tokenizer) -> Result<StencilTree, BuildError>;
```

### 8.1 The compile backend (`compile.rs`)

1. **Tokenize in context** (§7.2): walk the spec from `root`, accumulating the
   canonical string prefix so each `Static`/arm is tokenized at its true
   position.
2. **Build tries** for `Branch` arms from their in-context tokenizations.
3. **Single-arm fold**: a `Branch` with one arm becomes that arm's tokens as a
   `Static` spliced to its successor.
4. **Static fusion**: merge adjacent `Static` runs into one maximal node.
5. **Invariant check**: every reachable `Branch` has ≥2 live arms; no two
   `Static` nodes adjacent; `FreeText.limits.forced_after > 0`; every path
   reaches `End`. Violations are `BuildError`, surfaced loudly.

This pass is where "atomic prefill" (§3) and "mask never one-hot" (§7.1) are
*made true*, independent of how sloppy a front-end's spec was.

### 8.2 Front-end A — the builder

A fluent, programmatic builder for tests and bespoke trees:

```rust
let tree = StencilTreeBuilder::new("tool_call")
    .static_str("<tool_call>\n{\"name\": \"")
    .branch()
        .arm("read_file").goto("read_args")
        .arm("write_file").goto("write_args")
    .end_branch()
    .at("read_args")
        .static_str("\", \"arguments\": {\"path\": \"")
        .free_string(FreeTextLimits::json_string())   // forced_after only
        .static_str("\"}}\n</tool_call>")
        .finish()
    // … write_args …
    .build(&tokenizer)?;     // -> TreeSpec -> compile()
```

It emits a `TreeSpec` and calls `compile`. Labels (`goto`/`at`) resolve forward
references. This is the most direct way to exercise every node kind in tests.

### 8.3 Front-end B — a JSON tool description

The production path: a tool catalog (names + JSON-schema parameter specs) → a
tool-call tree. The catalog is the **same** source that feeds the projection's
`tools` collection (the prompt the model reads), so the prompt and the enforced
grammar can never disagree.

```rust
pub fn compile_tool_call_tree(catalog: &[ToolSpec], dialect: &Dialect,
                              tok: &Tokenizer) -> Result<StencilTree, BuildError>;
```

Layout (emitting a `TreeSpec`):

```text
Static  dialect.tool_call_open               // "<tool_call>\n{\"name\": \""
Branch  over tool names → each tool's schema
Static  "\", \"arguments\": {"
<argument object per tool>
Static  "}}\n" + dialect.tool_call_close
End
```

An **argument object** (required `R₁…Rₘ` in fixed order, optional `O₁…Oₙ` in
fixed order):

```text
for each required Rᵢ:  Static "\"<key>\": " · <value(Rᵢ.type)>   (+ separator)
for each optional Oⱼ:  Branch { "\"<key>\": " → <value> → next-gate
                              |  <close/next-key>     → next-gate }
```

Required fields have **no** gate (cannot be omitted); optionals are binary
include/skip gates; comma placement is laid out so no path produces a leading or
trailing comma. **Value sub-trees** by schema type:

| type | sub-tree |
|---|---|
| string (free) | `Static "\""` · `FreeText{JsonString}` |
| string enum | `Static "\""` · `Branch` over enum tries · `Static "\""` |
| integer / number | `FreeText{JsonNumber{integer_only}}` |
| boolean | `Branch { "true" \| "false" }` |
| nullable T | `Branch { <value(T)> \| "null" }` |
| array&lt;T&gt; | `Static "["` · `Branch{ <value(T)> \| "]" }` · ( `Branch{ "," → value \| "]" }` )* |
| object | recurse |
| raw/any (object value) | `FreeText{Balanced{'{','}'}}` (string-aware, §6.1) |

Enum/name tries reuse §7.2; nesting recurses.

### 8.4 Front-end C — a YAML node spec

A declarative format for hand-authoring trees directly (sibling to
`projection.yaml`). It maps 1:1 to `TreeSpec`:

```yaml
label: tool_call
root: open
nodes:
  - id: open
    static: "<tool_call>\n{\"name\": \""
    next: name
  - id: name
    branch:
      - match: "read_file"   ; next: read_args
      - match: "write_file"  ; next: write_args
  - id: read_args
    static: "\", \"arguments\": {\"path\": \""
    next: path_val
  - id: path_val
    free_text:
      terminator: json_string        # | json_number | balanced: {open: '{', close: '}'}
      eos_ends: false
      limits: { forced_after: 256 }  # ramp_start/ramp_len/boost optional
    next: close
  - id: close
    static: "\"}}\n</tool_call>"
    next: done
  - id: done
    end: true
```

`StencilTree::from_yaml(yaml, tok)` parses to `TreeSpec` and compiles. Parser
errors (unknown node id in `next`, unreachable node, no `End`, bad terminator
tag) are `BuildError`s, not panics.

All three front-ends share §8.1, so a builder tree, the JSON-derived tree, and
the YAML tree for the *same logical grammar* compile to byte-identical
`StencilTree`s — which is itself a test (§14.3).

---

## 9. Triggers

```rust
pub struct TriggerRegistry { by_token: HashMap<TokenId, Arc<StencilTree>> }
```

The decode loop's `observe` in the `Free` state does one hash lookup per decoded
token (the `<tool_call>` fast path is a single special token in Qwen3). A trigger
fires only in **free** decode, never inside an active session — the session owns
the stream until `End`, and a `FreeText` span does **not** consult the registry
(§12), so a `<tool_call>`-like substring inside a value cannot recursively
trigger. Nested calls are expressed as nested *tree* structure, not re-entry.
Multi-token triggers (a short literal that isn't a special token) are supported by
a suffix automaton over the recent-token window, but the tool-call case needs
only the single token.

---

## 10. Scheduler integration

Per sequence: an `Option<StencilSession>` beside its `DecodeState` (small; the
tree is shared `Arc`). The trees compile once at model/catalog load and live on
the scheduler in the `TriggerRegistry`. Only two scheduler changes:

1. **Sample-stage mask + EOS bias hook** in `batched_sampler.rs`: a per-row
   `Option<AllowedSet>` mask and a per-row close-token boost (§6.3), applied
   after penalties.
2. **Pre-wave prefill drain** in `scheduler/decode.rs`: a sequence owing a
   `Prefill` runs it on its slot (`run_prefill`) before rejoining the
   next decode wave; the §3 inner loop drains chained actions.

KV cache, projection, and persistence are untouched. The rare healing rewind
(§7.3) uses the existing `truncate_sequence_to_tokens`. **Decode-health guards**
(`decode_health.rs` entropy/repetition floors) are suspended while a session is
active — stenciled regions are deliberately low-entropy/repetitive — via a
`stencil_active` flag the checks short-circuit on, symmetric with how they handle
`inside_think_block`.

---

## 11. Persistence and provenance

A session produces **ordinary tokens** — prefilled static runs, masked samples,
and free samples all land in the sequence's KV at real positions and seal as part
of the turn's `[response]`, exactly like any other decoded tokens. `token_ids`
records the actual emitted tokens, so cross-process replay reconstructs the
identical KV; the GUI shows the verbatim tool call; provenance/BDP signatures
extract normally. The stencil is invisible downstream of the decode loop — it
changes *which* tokens are produced and *how fast*, not their representation.

---

## 12. Edge cases and failure modes

| Case | Handling |
|---|---|
| **Runaway free-text** | `limits.forced_after` force-closes; the close-token `ramp` nudges it to close naturally first (§6.3). Bounded. |
| **Escaped terminator** (`\"`, `\}` in a string) | The byte lexer's `escaped`/`in_string` state ignores them (§6.2). |
| **Nested structure value** (`{…{…}…}`) | `Balanced` depth counter, string-aware so brackets inside strings don't count (§6.1). |
| **Token healing impossible** | Session aborts to free decode, logs `WARN`; the partial call fails the caller's JSON parse loudly — never silent corruption (§7.3). |
| **EOS sampled in a session** | Masked out at every `Branch` and ignored by `FreeText` unless `eos_ends`; a session only ends at `End`. |
| **Model "fights" the stencil** | The mask is the last word at branches, so an out-of-grammar token shouldn't occur. As a failsafe, if one ever escapes the mask the session **bails** — logs `DEBUG`, emits the tree's configurable bail tokens to terminate the call cleanly, then exits. |
| **Empty / one-arm branch** | Invariant violations folded/checked at compile (§8.1); empty `debug_assert`s, release logs `ERROR` and aborts the session. |
| **Empty tool catalog** | No tree registered; `<tool_call>` isn't a trigger; tool calls decode freely. Logged once. |
| **Tokenizer ≠ compiled tree** | Fingerprint mismatch fails loudly at load. |

---

## 13. Public API and configuration

```rust
// candle-conversation/src/stencil/mod.rs
pub use tree::{StencilTree, StencilNode, NodeId, FreeTextSpan, FreeTextLimits};
pub use terminator::{Terminator, TerminatorState};
pub use session::{StencilSession, StencilAction};
pub use mask::AllowedSet;
pub use trigger::TriggerRegistry;
pub use builder::StencilTreeBuilder;
pub use spec::{TreeSpec, NodeSpec, BuildError};
pub use compile::compile;
pub use tool_call::{compile_tool_call_tree, ToolSpec};
pub use sim::StencilSimulator;   // test/diagnostic harness (§14)
```

Stenciling is **on by default** when a non-empty catalog and a tool-call trigger
exist, otherwise a no-op. A `SequenceConfig` flag disables it for ablation (free
tool-call decode) without removing the catalog from the prompt.

---

## 14. Testing — 100% coverage, simulator-driven

The module is **fully unit-testable without a GPU**: everything except the
forward pass is pure (compiler, session, terminator, mask construction). The bar
is **100% line/branch coverage** of `stencil/`, with extensive scenario and
edge-case suites. A **simulator** drives sessions deterministically.

### 14.1 The simulator (`sim.rs`)

```rust
/// Drives a StencilSession without a model: an oracle supplies the next token at
/// each MaskedDecode/FreeDecode; the simulator records the full action stream and
/// the emitted token sequence for assertions.
pub struct StencilSimulator { session: StencilSession, oracle: Oracle }
pub enum Oracle {
    Scripted(Vec<TokenId>),        // exact tokens (deterministic edge cases)
    PickArm(Box<dyn Fn(&AllowedSet) -> TokenId>),  // policy: choose an arm/value
    Adversarial(AdversaryPlan),    // escapes, nesting, runaway, healing, EOS
}
pub struct SimRun { actions: Vec<StencilAction>, tokens: Vec<TokenId>, healed: u32 }
```

`Scripted` asserts an exact `(action, token)` trace; `PickArm` enumerates *all*
paths through a tree (every tool, every optional combination, every enum value)
and asserts each emits parseable JSON matching the schema; `Adversarial` injects
the nasty inputs. The simulator also re-decodes `tokens` to a string and runs it
through a real JSON parser + schema validator for an end-to-end assertion — all
on CPU.

### 14.2 Component suites (byte-exact)

- **Terminator** (`terminator.rs`): for `JsonString`, `JsonNumber{int,float}`,
  `Balanced`: plain close; `\"` and `\\` escapes; `\\"` (escaped backslash then
  real quote); brackets inside strings; deeply nested `{[{}]}`; number lookahead
  (`123,` and `1.5e-3}`); UTF-8 multi-byte values; a terminator split across two
  tokens; empty value. Assert exact `Feed` and `byte_in_token`.
- **Compile backend** (`compile.rs`): single-arm fold; static fusion; in-context
  tokenization (boundary merges); trie sharing of common prefixes; invariant
  failures (empty arm, adjacent statics not fused, missing `End`, `forced_after
  == 0`) → `BuildError`. Raw token-id assertions, not tolerances.
- **Session** (`session.rs`): the §5 state machine via `Scripted` — minimal call;
  optional included/skipped; enum; number lookahead push-back; string with
  escapes; `Balanced` value; `eos_ends` true/false; soft ramp / graceful / forced
  limit firing; healing boundary; abort-on-heal-impossible.
- **Mask** (`mask.rs`): `frontier` correctness at each trie step; EOS exclusion;
  one-arm never reaches the sampler; per-row application over a synthetic logits
  matrix; close-token boost wiring.

### 14.3 Construction equivalence

For a fixed logical grammar, assert the **builder**, the **JSON tool
description**, and the **YAML** front-ends compile to byte-identical
`StencilTree`s (§8). YAML and builder also round-trip through `TreeSpec`.

### 14.4 Tool-library construction tests

Take the **actual tool definitions from the tool library** (the same JSON the
projection's `tools` collection uses), compile each to a tree, and run the
simulator across scripted scenarios per tool: every required field; each optional
present and absent (and combinations, capped); each enum value; arrays empty,
singleton, and multi; nested objects; nullable fields as value and as `null`.
Assert every produced byte stream parses and validates against that tool's
schema, and that the **tool name and every enum are exactly from the catalog**.
This is the regression net that guarantees the stencil never lets a real tool be
mis-called.

### 14.5 End-to-end (GPU, integration — the only non-CPU test)

With the real model + tokenizer, prompt tool calls and assert the produced bytes
parse, name ∈ catalog, required present, enums ∈ allowed — across all catalog
tools. Everything else (14.1–14.4) runs on CI CPU.

---

## 15. Module layout

```text
candle-conversation/src/stencil/
  mod.rs          public API, StencilAction, re-exports
  tree.rs         StencilTree, StencilNode, NodeId, FreeTextSpan, FreeTextLimits
  trie.rs         TokenTrie, frontier(), in-context name/enum trie builder
  terminator.rs   Terminator, TerminatorState (escape + nesting byte lexer)
  session.rs      StencilSession, Cursor, next_action()/observe()
  mask.rs         AllowedSet, logit masking + close-token boost, per-row apply
  trigger.rs      TriggerRegistry
  spec.rs         TreeSpec, NodeSpec, BuildError (the string-space intermediate)
  compile.rs      compile(): tokenize-in-context, fold, fuse, verify invariants
  builder.rs      StencilTreeBuilder (front-end A)
  tool_call.rs    compile_tool_call_tree + ToolSpec → TreeSpec (front-end B)
  yaml.rs         TreeSpec::from_yaml (front-end C)
  sim.rs          StencilSimulator + Oracle (test/diagnostic harness)
  tests/          extensive scenario + edge-case suites (or #[cfg(test)] per file)
```

One concern per file. The scheduler/sampler touch-points are additive hooks in
`scheduler/decode.rs` and `batched_sampler.rs`; no existing KV, projection, or
persistence code changes.

---

## 16. Implementation order

Dependency-ordered, every step CPU-testable until the final hardware hook:

1. **`terminator`** — the escape/nesting byte lexer + its full edge-case suite
   (§14.2). Pure, no dependencies.
2. **`tree` + `trie` + `spec`** — the data model and the string-space
   intermediate.
3. **`compile`** — tokenize-in-context, fold, fuse, invariants, byte-exact tests.
4. **`builder` + `yaml` + `tool_call`** — the three front-ends, with the
   equivalence test (§14.3) and the tool-library construction tests (§14.4).
5. **`session`** — `next_action`/`observe`, driven by `sim` (§14.1). No scheduler.
6. **`mask`** — `AllowedSet` + per-row logit mask + close-token boost, against a
   synthetic logits matrix.
7. **`trigger`** + engine wiring — register trees, hand the registry to the
   scheduler.
8. **Scheduler hook** — pre-wave prefill drain + sample-stage mask/boost +
   health-guard suspension.
9. **Token healing** — the rewind/reprefill boundary path, behind the limit guards.
10. **End-to-end GPU validation** (§14.5) across the full catalog.

Steps 1–6 land and are 100% covered before any scheduler change; the
hardware-coupled work (7–10) sits on a proven, CPU-verified core.
