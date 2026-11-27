# Stencil Tree — Schema-Guided Constrained Decoding

**Status:** built. The sections below describe the code as it is; where the
shipped design differs from the first sketch, the section says so in a
*Replaced design* note. Pieces designed and not built are marked **Not built**
(§10, §13, §14.3, §14.5).
**Module:** `candle-conversation/src/stencil/` (the walker, compiler, masks and
simulator run without a GPU; the live-tokenizer checks in
`zend/tests/stencil_tool_call.rs` need a cached tokenizer).
**Touch-points:** a sample-stage mask hook in `batched_sampler/allow_list.rs`, a
per-sequence step hook in `scheduler/decode/` (`commit.rs`, `stencil.rs`), and the
wave rows of `scheduler/stencil_wave.rs`. Nothing in the KV cache, projection, or
persistence paths changes.
**Primary use case:** forcing a tool call to a tool catalog's exact names and
JSON schema.

### As-built deltas (beyond the first sketch)

- **A fourth front-end, D — think steering** (`think.rs`, `ThinkMode`,
  `compile_think_tree`): per-effort-dial trees that steer the `<think>` block with
  an opener + continuation phrases, closed via token-closed `suppress_close` spans.
  §8 has four front-ends.
- **`StencilDriver`** (`driver/mod.rs`): the online walker the scheduler drives one
  decoded token at a time (the `Healed` signal contract with `scheduler/decode/`).
- **`TriggerRegistry`** (`trigger.rs`): token → tree dispatch with atomic,
  idempotent `with_trigger` / `without_trigger`, used to swap the per-turn think
  tree onto the tool-call base.
- Signatures as built (§13, §15 list them): `compile(&TreeSpec, &dyn Vocab)` takes
  a `Vocab`, not a tokenizer; `compile_tool_call_tree(&[ToolSpec],
  &ToolCallEnvelope)` returns a `TreeSpec` (the dialect is the envelope);
  `TreeSpec::from_yaml(yaml)` is the YAML front-end; the builder's `build()`
  returns a `TreeSpec`; the simulator is `simulate` with `Oracle` and `SimRun`
  (`sim.rs`); and `StencilAction::FreeDecode` carries `{ close_boost, opens_value, barred }`.
  `Terminator` also has `JsonStringValue`, `JsonValue`, `Never` and `Until`.
- **Value sub-trees as shipped** (§8.3). Scalars (`integer`/`number`) and any
  container the grammar does not guide are one `FreeText{JsonValue}` span —
  any structurally-valid JSON value, lookahead-terminated at the enclosing
  `,`/`}`/`]` — rather than `JsonNumber` / `Balanced`. **Guided containers**
  follow the table: an `object` with `properties` is recursed (its fields under
  the same required/optional gates as the arguments, closed by `}`), and an
  `array` whose elements the grammar can begin (string, boolean, enum, object
  with properties) is `[` · `Branch{ <open> element | "]" }` ·
  (`Branch{ ", <open>" element | "]" }`)*. The repetition is **unrolled** to
  `MAX_ARRAY_ELEMENTS` (64) levels, the last offering only `]`, because the
  compiler rejects cycles (§8.1). Left free: arrays of numbers (an element the
  model opens has nothing for an arm to carry), and an array inside a guided
  array's elements (it would repeat per enclosing level).
- **Lowering shares nodes** (§8.1). The spec is a DAG — optional gates share
  successors, every unrolled array level shares the value after the array — and
  the compiler lowers each `(spec node, left context)` once, so the runtime
  tree is a DAG too. Lowered per path, an array of objects with optional
  fields grows exponentially in the array bound.
- **A lookahead delimiter the successor does not continue with is dropped**
  (§7.3), not pushed back and bailed on: the session returns
  `Observe::DelimiterDropped`, the driver `Healed::Drop` (counted in
  `PathStats::dropped_delimiters`), and the successor writes the structure in
  its place. Live, a free-array `read_files` call ended
  `"end_line": 3420]}}}` — the element's `}` never written — and was not JSON.
- **Free JSON spans validate and repair** (§6, §7.3, §12). `JsonString` and
  `JsonValue` run a full incremental JSON lexer (`json_lexer.rs`: container
  stack, escapes, number and literal states) instead of a quote finder and a
  depth counter. A free decode cannot be masked, so a token that would break
  the value is **rewritten**: a raw control character in a string becomes its
  escape, an escape JSON lacks becomes a literal backslash, a short `\u` is
  padded; and at the first byte that cannot continue the value, the value ends
  with the smallest completion of what was written — the string closed, the
  containers closed in the order they opened (`[1, [2}` → `[1, [2]]`), the
  missing digit or value supplied (`"end_line":,` → `"end_line": null`). An EOS
  inside a span, or the hard limit, completes it the same way, so the fixed
  per-span `close_run` is gone: the session hands the decode loop the bytes to
  commit (`StencilSession::take_rewrite` → `Healed::Rewrite`, re-tokenized in
  the heal path that already committed mid-token closes). The design's
  "healing impossible → abort" (§12) does not arise: every cut has a completion.
- **A repair keeps the meaning.** What another language spells differently is
  respelled, not replaced: `True`/`FALSE`/`None`/`nil`/`undefined` (any case) are
  the JSON literals; `NaN`/`Infinity`/`inf` are `null`, as `JSON.stringify`
  writes them; single-quoted strings and keys, bare keys (`{path: 1}`), a
  missing `:` or `,`, `.5`/`5.`, and `\'`/`\xHH`/`\a`/`\v` escapes all keep their
  value. `null` is supplied only where nothing recoverable was written. The one
  cut-dependent repair is a trailing comma: dropped when it shares a token with
  the closer, completed after (`[1, 2, null]`) when an earlier token committed
  it. `edge_tests::intent` asserts the parsed value under every cut of each case
  and under 600 generated values, spellings and cuts.
- **A closed set is a choice, with `null` where the schema allows it** (§8.3's
  `nullable T` row, built). A boolean, an enum, and every nullable value are
  offered as branch arms — `true`/`false`/`null`, `"a"`/`"b"`/`null`,
  `"`/`null`, `[`/`null`, `{`/`null` — so a closed value is never free text and a
  nullable field can be `null`. Nullable is read from a `null` in `type` or
  `enum`, `nullable: true`, and `anyOf`/`oneOf` of one schema and
  `{"type": "null"}`. Arrays of booleans and enums are guided like arrays of
  strings, their elements chosen per level (`true`, `false`, `]`, then
  `, true`, …). Numbers, and values with no schema, stay free.
- **A value can bring its own fields** (§8.3; `Param::shapes`, built). A
  required, non-nullable enum carries `(value, fields)` pairs: once the model has
  chosen a value, the object continues with exactly those fields — required,
  typed as given, and replacing any same-named sibling — so the choice decides
  what may be written beside it. The compiler lays this out as a branch over the
  enum's token trie in which each arm leads to the tail for its value's fields.
  The same machinery serves `requires` (the schema's `if`/`then`: fields a value
  makes required). The effector's `invoke` uses it: `url` brings the `body` its
  address takes (`the_effector.md` §11), which is what lets one tree type every
  address a character can reach without a second compile per address.
  - **An optional enum decides too.** Written, its value goes on to the fields
    it brings and the fields it requires, then the gates over the optionals
    after it, from the standing the object had reached (`builder::value_tails`
    with an `Owed`); left out, the object goes on as if it had no shapes. Only a
    required enum once decided: npcd's `order_post` takes `job` or its `steps`,
    so `job` is optional, and choosing `coding-task` brought nothing — Zen
    wrote a `brief` beside it and the post was refused for want of `what`.
  - **Equal shapes share one tail.** The spec is a DAG and the compile memoises on
    `(spec node, left context)`, so values whose field lists are equal point at
    one tail `SpecId` and lower it once. The tree grows with the number of
    *distinct* shapes, not the number of values: two hundred addresses that take
    `{}` cost their url arms and one body.
  - **Measured** (npcd `invoke_body` tests, `TestVocab` byte-level vocabulary, one
    turn of `ACTS_PER_TURN` calls over the whole catalogue): a room offering 286
    station addresses — two of every part, everything a body could name in reach —
    compiles to 612 nodes in about 6 ms, and 572 addresses compile to the same
    612 (node count does not track the url count in this range); adding an
    address costs its url, not its body, and the test holds that cost under 400
    nodes.
  - **The budget counts paths, not nodes.** `estimated_paths` (npcd
    `engine/tools/compile.rs`) sums a shaped parameter over its shapes and multiplies over
    properties, then raises to `ACTS_PER_TURN`. It is the larger number and is
    quadratic in the address count: it bounds what a change to the catalogue can
    multiply, while the node count above is what the compile costs.
  - **Limit.** A required free-text string can still be written as `""`; the
    grammar has no non-empty terminator, and the handler's prescriptive error
    (`the_effector.md` §12) is what catches it.

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
- **Let the model decode scalar values freely** (a container with a schema is
  scaffolding and decision points too, §8.3), watching for the terminator that ends
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
| **Stencil session** | The per-sequence runtime walker: a single cursor into one tree. No buffer of structure still to emit (§4.1). |
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
        FreeDecode{close_boost, ..} => { seq.next_mask = None; seq.eos_bias = close_boost; break }
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

`Prefill` writes the node's buffer in the decode step's own wave, not a decode
per token — this is the §1 speedup. The buffers up to the next choice, behind
the sequence's pending token, make one row (`scheduler/stencil_wave.rs`) that
rides the wave's prefill slot as a full-sweep member beside the decode rows and
verify blocks. A run costs no forward of its own, which on a card that streams
its experts is a second round of expert loads saved per stenciled step. A wave
refused for ground keeps the row for the next step; the run's tokens are already
recorded, so it is written by the retry rather than failing the turn.

**A wave writes a row a piece at a time.** A piece is at most the model's short
span (`ManagedBatchedModel::short_span_rows`) and never runs past a page break
with something after it; the break's page closes before the next piece. Only
the piece that completes the row is scored, and the token after the run is
sampled from it. The span is the recurrent stack's timeline (8 rows): a longer
span writes a second copy of the sequence's whole recurrent state, its prefill
scratch (144 MiB on Flash-Next), which a decode would take at any step with no
admission of its own. Cut to the span, a stencil decode claims nothing a cohort
or admission sets aside. Measured over a minimal call of every tool in the live
catalog on the live tokenizer, 80% of rows are within it; the rest are the row
after the tool's name, its last token and `", "arguments": {"`, at 9 or 10, and
each costs one more step as two pieces (`zend/tests/stencil_tool_call.rs`,
`the_rows_a_tool_call_plays_on_the_live_tokenizer`).

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
    pub term: Terminator,
    /// When set, an EOS sample also ends the span (byte-terminator spans). When
    /// unset, EOS is never committed inside the span: it is replaced by the
    /// completion of the value as written (§12).
    pub eos_ends: bool,
    /// Span-scoped limits mirroring `SamplingConfig` (§6.3). `forced_after` is
    /// always present (the runaway guard).
    pub limits: FreeTextLimits,
    /// A token whose sample also closes the span (the thinking block's
    /// `</think>`), and whether that token is dropped (`suppress_close`) so the
    /// successor prefills in its place.
    pub close_token: Option<TokenId>,
    pub suppress_close: bool,
    /// What the span's value may spell (`ValueRule`, §9.2); `None` leaves it free.
    pub rule: Option<Arc<ValueRule>>,
    /// One successor per tail of a `Terminator::Until`, the tail that fired
    /// choosing (§12); empty otherwise.
    pub forks: Arc<[NodeId]>,
    pub next: NodeId,
}
```

### 4.1 No session-side buffering

The session is just a cursor:

```rust
pub struct StencilSession {
    tree: Arc<StencilTree>,
    cursor: Cursor,
    // plus the walk's small side state: a lookahead delimiter awaiting its
    // successor, the bytes of a pending rewrite, the value's position against
    // its rule, the held values and the call written so far (§9.2) — none of it
    // a buffer of tokens still to emit
}
enum Cursor {
    At(NodeId),
    InBranch { node: NodeId, pos: TrieNodeId },
    InFreeText { node: NodeId, term: TerminatorState, emitted: u32 },
    Bailing,     // the failsafe: emit the tree's bail tokens, then finish
    Done,
}
```

There is no `pending: Vec<TokenId>` of structure still to emit. A `Static` node carries its own buffer; the
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
                                        FreeDecode{span.limits.boost_at(0), ..}   // §6.3
                                        // or MaskedDecode(rule.mask(state)) under a ValueRule (§9.2)
                End                  -> cursor = Done; Exit
        InBranch{pos}    -> MaskedDecode(frontier(pos))
        InFreeText{..}   -> FreeDecode{span.limits.boost_at(emitted), ..}
        Bailing          -> Prefill(tree.bail) ; cursor = Done
        Done             -> Exit

fn observe(token, bytes) -> Observe:
    match cursor:
        InBranch{pos}:
            child = trie[pos].edge_to(token)        // legal by the mask; a token
                                                    // that escaped it → Bailed
            if trie[child].is_leaf(): cursor = At(trie[child].next)   // ArmComplete
            else:                     cursor = InBranch{child}
        InFreeText{node, term, emitted}:
            emitted += 1
            match term.feed(bytes):                 // §6 byte lexer
                Continue        -> stay                      // span continues
                Close{consumed} -> cursor = At(span.after(fired))   // leftover bytes: §7.3
                Rewrite{bytes, closed} -> hold the bytes for take_rewrite()   // §7.3
            close_token / eos / forced_after close it too   // §6.3
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
    /// JSON string value whose opening quote is already written. Ends at the
    /// first UNESCAPED `"`, which is consumed; validated and repaired by the
    /// incremental JSON lexer (below).
    JsonString,
    /// A JSON string value INCLUDING its opening quote, written by the model
    /// (a plain string parameter: Qwen spells an empty string as one token,
    /// ` ""`, so the grammar must not have chosen "non-empty" for it).
    JsonStringValue,
    /// JSON number value. Lookahead-terminated: ends at the first byte that
    /// can't extend a number; `integer_only` drops '.'/'e'. The terminator byte
    /// is NOT consumed — it belongs to the following node (a ',' or '}').
    JsonNumber { integer_only: bool },
    /// A balanced `open`/`close` structure, string-aware: ends when nesting
    /// depth returns to 0; the closing bracket is consumed.
    Balanced { open: u8, close: u8 },
    /// Any JSON value (the shipped form of an unguided scalar or container):
    /// lookahead-terminated at the enclosing `,`/`}`/`]`, validated byte by byte.
    JsonValue,
    /// No byte delimiter; the span ends by a close token, EOS or `forced_after`
    /// (the thinking block).
    Never,
    /// Raw text ending at a literal `marker` followed by one of `tails` (a
    /// function block's `</parameter>` value, §12).
    Until { marker: &'static str, tails: &'static [&'static str] },
}
```

### 6.2 The lexer state

```rust
pub struct TerminatorState { /* kind, depth, in_string, escaped, started,
                                spaced, matched, recent, fired, json: Option<JsonLexer> */ }

pub enum Feed {
    Continue,
    Close { consumed: usize },                       // consumed < len ⇒ mid-token (§7.3)
    Rewrite { bytes: Vec<u8>, closed: bool },        // commit these bytes instead (§7.3)
}
```

`feed(bytes)` runs the machine byte-by-byte over the decoded token:

- **`JsonString` / `JsonValue`**: the full incremental JSON lexer
  (`json_lexer.rs`: container stack, escapes, number and literal states). A
  token that cannot be committed as written comes back as `Rewrite` (§7.3).
- **`Balanced{open,close}`**: maintain `in_string`/`escaped` for any `"` seen;
  while **not** `in_string`, `b==open → depth+=1`,
  `b==close → { depth-=1; if depth==0 return Close }`. A span that begins at its
  first `open` starts at depth 0 and `started` is set on the first `open`.
- **`JsonNumber`**: `if !is_number_byte(b) { return Close{consumed: i} }` — the
  close byte is *not* consumed (lookahead); §7.3 applies.
- **`Until`**: the marker is matched across token boundaries; with tails, the
  span closes only when the marker and then one tail have been written, and
  `fired` names the tail.

No regex engine is involved; `Terminator::is_lookahead` says which kinds leave
their delimiter to the successor.

### 6.3 Soft / hard limits (mirrors the sampler's EOS ramp)

`FreeTextLimits` scopes the sampler's whole-turn EOS mechanism
(`SamplingConfig::{eos_boost, eos_ramp_start, eos_ramp_len, graceful_eos_after,
forced_eos_after}`, see `batched_sampler/`) to a single span, closing with the
span's terminator (or EOS when `eos_ends`) instead of the turn's EOS:

```rust
pub struct FreeTextLimits {
    /// Below this many span tokens, no pressure. At/after it, ramp a boost onto
    /// the span's CLOSE token (the quote, the ']'/'}' , or EOS when eos_ends),
    /// linearly to `ramp_len`. Mirrors `eos_ramp_start` / `eos_boost`.
    ramp_start: Option<u32>,
    ramp_len: u32,
    boost: f32,
    /// Hard: at this many tokens, force-close unconditionally — commit the
    /// token with the text that completes the value as written (§7.3) and
    /// advance. The runaway backstop; always set. Mirrors `forced_eos_after`.
    forced_after: u32,
}
```

- **Ramp** (`FreeDecode{close_boost, ..}` in §5): the action carries a per-step
  boost the sampler adds to the close token's logit — a *soft* nudge that lets
  the model close naturally, exactly like `eos_boost`. Zero before `ramp_start`.
- **Forced**: at `forced_after`, the session reports `Observe::SpanForcedClosed`:
  the token is committed with the span's completion after it, and the walk
  advances regardless. Bounded; never infinite.
- **`eos_ends`**: when set, an EOS sample is a legal close (and is what the ramp
  boosts); when unset, an EOS sample is never committed inside the span (§12).

A pure JSON string value sets `forced_after` only (`MAX_STRING_VALUE_TOKENS`, a
high runaway guard) and leaves the rest unset — the model reliably emits its own
closing quote. A free-prose span (e.g. a tool's `reason` field) sets the full ramp +
graceful so it wraps up like a normal turn.

---

## 7. The sampling stencil and tokenization

### 7.1 The mask

`MaskedDecode(set)` restricts the sample to the allowed tokens. The sampler does
not write `-inf` over the vocabulary: `batched_sampler/allow_list.rs` gathers only
the allowed logits on the device and runs the row's strategy (penalties,
temperature, top-p, or argmax) over them, so the cost is the size of the list,
never the vocabulary, and the model's relative preference among legal tokens
survives. Every allow-list row of a step runs in one launch; a row whose allowed
logits hold a NaN or +inf is refused before anything is drawn. Per-sequence in
the wave batch; rows with no mask decode normally. A one-token mask never occurs
(folded to `Static`, §8.1); an empty mask is a compile-time invariant violation.

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
reports `Close { consumed }` with `consumed` short of the token's length. The
heal is a rewrite, not a rewind of the KV: the token is not yet committed when
`StencilDriver::accept` sees it, so the driver answers `Healed::Rewrite { bytes }`
with the valid prefix, the commit path re-tokenizes `bytes` and swaps the sampled
token for the last token of that encoding. When the encoding is several tokens,
all but the last are recorded as forwarded and prefilled once the step's KV is
final (`HealPrefill`), and the last rides the next decode. The successor node
writes the rest of the structure. (Replaced design: the first sketch truncated
the KV by one token with `truncate_sequence_to_tokens` and re-prefilled; the
scheduler's stencil path no longer calls it.)

A lookahead span (`JsonNumber`, `JsonValue`) leaves its delimiter to the
successor. A delimiter that is its own token but is not what the successor starts
with is dropped (`Healed::Drop`, `Observe::DelimiterDropped`) rather than pushed
back, so the successor writes the right one. A token that cannot be committed as
written inside a span is `Feed::Rewrite`, handled by the same path. Every cut has
a completion, so "healing impossible" does not arise (§12).

### 7.4 The guided→free boundary: spills

The opposite seam is the **start** of a free span, where the grammar's last
token meets the model's first. A tokenizer merges across it too: Qwen3.8 writes
an array element `"-p"` as ` "-` then `p"` — the separator's opening quote and
the flag's dash in one token. A branch arm that ends on ` "` offers only ` "`,
so that token was masked out, and the model continued from a boundary its
encoding never produces. Measured live: asked for `cargo tree -p agent --depth
1`, every `run_command` came out `["tree", "depth"]` — the dashes, and the words
they led, gone — and after five tries the array decoded as `"p"` forty times.

So an arm's last token is offered beside its **spills**
(`stencil::trie`): every token that starts with that token's bytes and runs on
past it, when the arm's successor is a `JsonString` span and the bytes past the
arm are plain string content — no `"`, no `\`, no control byte, no partial
character. Taking one completes the arm and feeds the bytes past it to the
span's lexer as its first (`StencilSession::observe`). A static run that ends
right before such a span (an array's required first element, whose `"` is the
grammar's alone) holds its last token back as a one-arm branch with its spills,
for the same reason.

The candidates come from `Vocab::extensions`, which for the live tokenizer is an
index of every token's bytes, sorted (`TokenBytes`), built once per engine and
shared by every compile. A plain string *parameter* needs none of this: its
opening quote is already the model's (`Terminator::JsonStringValue`, §6.1).

---

## 8. Construction — four front-ends, one backend

Every tree is a `TreeSpec` (string-space, untokenized) compiled by a single
backend. This keeps the construction methods thin and guarantees they produce
identical trees for identical logical input. The front-ends are the builder (A),
the tool catalog (B), YAML (C) and think steering (D, `think.rs`).

```rust
// String-space intermediate — what every front-end produces (spec.rs).
pub struct SpecId(pub usize);
pub struct TreeSpec {
    pub nodes: Vec<NodeSpec>, pub root: SpecId, pub label: String,
    pub bail: String,                       // text emitted if a token escapes the mask
    pub keys: HashMap<SpecId, Arc<str>>,    // value span → "tool.field" (§9.2)
    pub name_branches: HashSet<SpecId>,     // branches whose arms are tool names
    pub chosen: HashMap<SpecId, ChosenArms>,// branches choosing an argument's value
    pub fixed: HashMap<Arc<str>, Vec<(Arc<str>, Arc<str>)>>, // single-valued required args
}
pub enum NodeSpec {
    Static { text: String, next: SpecId },
    Branch { arms: Vec<(String, SpecId)> },         // (arm literal, successor)
    FreeText { term: Terminator, eos_ends: bool, limits: FreeTextLimits,
               close_token: Option<TokenId>, suppress_close: bool, next: SpecId },
    Value { pattern: String, opened: bool, limits: FreeTextLimits, next: SpecId },
    Element { marker: &'static str, tails: &'static [&'static str],
              limits: FreeTextLimits, forks: Vec<SpecId>, next: SpecId },
    End,
}

/// The one backend: tokenize-in-context, build tries, FOLD single-arm branches
/// to Static, FUSE adjacent Static runs into maximal nodes, verify invariants.
pub fn compile(spec: &TreeSpec, vocab: &dyn Vocab) -> Result<StencilTree, BuildError>;
```

### 8.1 The compile backend (`compile.rs`)

1. **Tokenize in context** (§7.2): walk the spec from `root`, accumulating the
   canonical string prefix so each `Static`/arm is tokenized at its true
   position.
2. **Build tries** for `Branch` arms from their in-context tokenizations.
3. **Single-arm fold**: a `Branch` with one arm becomes that arm's tokens as a
   `Static` spliced to its successor.
4. **Static fusion**: merge adjacent `Static` runs into one maximal node.
5. **Validation and invariant check**: references resolve, the spec has no
   cycle, every path reaches `End` (`validate`); then every reachable `Branch`
   has ≥2 live arms (or one and a spill), no two `Static` nodes are adjacent, and
   `FreeText.limits.forced_after > 0` (`verify_invariants`). Violations are
   `BuildError`, surfaced loudly.

Lowering is memoised on `(spec node, left context)`, so a spec node reached along
several paths is lowered once and the runtime tree is a DAG (the *Lowers shares
nodes* delta above).

This pass is where "atomic prefill" (§3) and "mask never one-hot" (§7.1) are
*made true*, independent of how sloppy a front-end's spec was.

### 8.2 Front-end A — the builder

A programmatic builder for tests and bespoke trees. Nodes are declared by string
id and name their successors by id, so forward references are natural:

```rust
let spec = StencilTreeBuilder::new("tool_call")
    .root("open")
    .static_node("open", "<tool_call>\n{\"name\": \"", "name")
    .branch("name", &[("read_file", "read_args"), ("write_file", "write_args")])
    .static_node("read_args", "\", \"arguments\": {\"path\": \"", "path_val")
    .free_string("path_val", "close")             // JsonString, forced_after only
    .static_node("close", "\"}}\n</tool_call>", "done")
    .end("done")
    // … write_args …
    .build()?;                                    // -> TreeSpec
let tree = compile(&spec, &vocab)?;
```

`build()` resolves the ids (`LabeledTree::resolve`, shared with the YAML
front-end) to a `TreeSpec`; `compile` lowers it. Also available: `free_text`
(any terminator, limits and EOS behaviour) and `free_text_token_closed` (a span
that also closes on a sampled token). This is the most direct way to exercise
every node kind in tests.

*Replaced design:* the first sketch was a chained `.arm(..).goto(..)` /
`.at(..)` builder that took the tokenizer in `build`. The shipped builder is
id-based, returns a `TreeSpec`, and leaves tokenization to `compile`.

### 8.3 Front-end B — a JSON tool description

The production path: a tool catalog (names + JSON-schema parameter specs) → a
tool-call tree. The catalog is the **same** source that feeds the projection's
`tools` collection (the prompt the model reads), so the prompt and the enforced
grammar can never disagree.

```rust
pub fn compile_tool_call_tree(tools: &[ToolSpec], env: &ToolCallEnvelope)
    -> Result<TreeSpec, BuildError>;
// and the loops a turn runs: compile_tool_call_loop(tools, env, max_calls, close_turn),
// compile_action_loop(.., think), compile_action_loop_with_body(..)
```

The envelope (`ToolCallEnvelope::qwen3()` for the JSON block,
`ToolCallEnvelope::qwen35()` for the function block) carries the dialect's
markers. Layout (emitting a `TreeSpec`, JSON-block style):

```text
Static  env.open                             // "<tool_call>\n{\"name\": \""
Branch  over tool names → each tool's schema
Static  env.args_open                        // ", \"arguments\": {"
<argument object per tool>
Static  env.close                            // "}}\n</tool_call>"
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
| integer / number | `FreeText{JsonValue}` (the *Value sub-trees as shipped* delta) |
| boolean | `Branch { "true" \| "false" }` |
| nullable T | `Branch { <value(T)> \| "null" }` |
| array&lt;T&gt; | `Static "["` · `Branch{ <value(T)> \| "]" }` · ( `Branch{ "," → value \| "]" }` )*, unrolled to `MAX_ARRAY_ELEMENTS` |
| object | recurse |
| raw/any (unguided value) | `FreeText{JsonValue}` (string-aware, §6.1) |

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
      - { match: "read_file", next: read_args }
      - { match: "write_file", next: write_args }
  - id: read_args
    static: "\", \"arguments\": {\"path\": \""
    next: path_val
  - id: path_val
    free_text:
      terminator: json_string        # | json_number | json_integer | balanced (with open: / close: bytes)
      eos_ends: false
      limits: { forced_after: 256 }  # ramp_start/ramp_len/boost optional
    next: close
  - id: close
    static: "\"}}\n</tool_call>"
    next: done
  - id: done
    end: true
```

`TreeSpec::from_yaml(yaml)` parses to a `TreeSpec`, which `compile` lowers. Parser
errors (unknown node id in `next`, duplicate id, unknown root, a node declaring
none or several of `static`/`branch`/`free_text`/`end`, bad terminator tag, no
reachable `End`) are `BuildError`s, not panics. A token-closed span (the thinking
block) is built programmatically, not in YAML, because its close token is a
resolved id.

The builder and YAML front-ends resolve through the same `LabeledTree`, and all
front-ends share §8.1, so a builder tree and the YAML tree for the same grammar
compile to the same `StencilTree` — pinned by `tests::builder_and_yaml_agree`
(§14.3).

---

## 9. Triggers

```rust
pub struct TriggerRegistry { by_token: HashMap<TokenId, Binding>, tail: Option<Arc<StencilTree>> }
```

A `Binding` is a tree plus what a turn holds it to (§9.1, §9.2) and whether it
fires once. The decode loop's commit path in the free state does one hash lookup
per decoded token (`<tool_call>` is a single special token in Qwen3).
`with_trigger` / `without_trigger` swap a binding atomically and idempotently;
`with_once_trigger` binds a token that fires at most once a turn (the think
block). A trigger fires only in **free** decode, never inside an active session
— the session owns the stream until `End`, and a `FreeText` span does **not**
consult the registry, so a `<tool_call>`-like substring inside a value cannot
recursively trigger. Nested calls are expressed as nested *tree* structure, not
re-entry. Triggers are single tokens; a multi-token trigger (a suffix automaton
over the recent-token window) is **Not built**, and no catalog needs one.

### 9.1 Holding a turn's calls to the tools it was given

A tool-call tree is compiled once per tools mode from that mode's whole catalog,
and each call level of its loop has a name branch offering every tool in it. A
turn given fewer — a layer unit with two tools, a file's reading with
`file_read` — narrows those branches per turn, without compiling anything:

- `held_branches(tree, names, decode)` finds every **name branch** (a branch
  whose arms spell exactly the catalog — the first branch's names — so an enum
  or the loop's go-again choice never qualifies) and keeps the arms that spell a
  given name. It decodes only branches with as many arms as the catalog.
- `TriggerRegistry::holding(token, held)` returns the turn's registry with that
  table on the trigger's binding; `ConversationEngine::holding_tools(triggers,
  names, first_call)` does both, and errors when no given name is in the catalog.
- The driver `driver_for` starts carries the table. At each masked step it keys
  on the branch the cursor is in (`StencilSession::in_branch`): entering a held
  branch at its first token starts a hold, each step is masked to the tokens a
  still-live held arm writes next, and the hold ends when an arm completes.
  Later branches — an argument's enum — are untouched.
- Named-tool steering (`scheduler/named_tool.rs`) picks among the held arms, so
  reasoning that names a tool the turn was not given steers nothing.

**A tool called at most once a turn** (`never_twice` on its definition) is held
out of every later name branch of the turn once a call has named it
(`NeverTwice`, `stencil/never_twice.rs`). Which tools are spent is the walk's
own record — the session notes each call's tool as its name branch completes —
so the driver narrows each name branch as it enters it: the turn's held arms
when it has them, the branch's whole catalog when it does not, less the spent
tools' arms. A branch is never narrowed to nothing. `journal_note` is the case:
a round's opening call is held to it, the calls after it to the turn's tools,
and those offered the note again — a model wrote the same note twice, word for
word, where its thinking had planned a listing. zend sets the flag in the tool's
YAML and holds every name and alias of a flagged tool out
(`tools::never_twice_names`).

**Cost.** The compiled tree is shared and unchanged. A held registry adds one
table of `(branch node, arms)` — a few tokens per given tool per call level —
built once for the conversation that runs under it (a layer unit, or once at
boot for file reading). Per call, a hold is one index list over the held arms;
per masked step of a name, a filter of the frontier against them.

Before holding, a unit shown two tools could still decode any tool of its mode's
catalog, and each such call was answered `not_available` after the fact — a
wasted round, ten or so a day in ingest. The refusal stays in
`zend::layers::run::only_given` as a check on the hold: a call it refuses got
past the grammar, a stencil defect, and is logged at `ERROR` rather than run.

### 9.2 Holding a call's values to what exists

A pattern holds a value to a *shape*: `part` to four anchor characters, `from`
to `+N`. A shape admits values that do not exist — an anchor the file does not
have, `+40` of a 37-line part — and every one is a refused call, read and
retried. A model that has decided on a wrong value retries it: live, a coder
whose reasoning said `+37` wrote `"to": "last"` seven rounds running (a stale
overlay had given `to` an anchor's pattern, which "last" fits), and only the
repeat guard ended it. **The grammar, not a guard after it, holds a call to what
can succeed**: each round, the fields whose values are known are held to those
values, so a value that does not exist cannot be written, and neither can the
call that was just refused for one.

- **Every value span carries a key.** The tool-call front-end records, beside
  each argument's span, `tool.field` (`file_edit.part`; an array's items
  `tool.field[]`) in the spec's `keys` table, and `compile` carries it to the
  tree's (`StencilTree::key_of(node)`). Nothing else about the span changes.
- **A turn's held values** (`HeldValues`, `stencil/held_values.rs`) map a key to
  the values its span may take, and a value may depend on fields already written
  in the same call: `file_edit.part` on `repo` and `path`, `from` and `to` on
  `repo`, `path` and `part`. Each is a `ValueRule` over the set of values
  (`ValueRule::of_set`, §9.2.1) — never `ValueRule::shared`, whose cache keeps
  every pattern for the process's life and would grow by one per round.
- **The session holds a keyed span to its held rule** in place of the span's
  own: entering a span, it looks up the key with the call's tool and written
  fields (`CallSoFar`, `stencil/call_so_far.rs`), and walks the held rule when
  there is one. A key with no held values, or whose fields name nothing held (a
  file the conversation never read), keeps the span's own rule. Each keyed
  span's value is recorded as the call writes it, so a later field can depend on
  it.
- **A function block's raw value is held as a quoted one is.** Qwen3.5/3.8
  write a call as elements (the edge-case table's raw-value row), each
  value an `Element` span closed by `</parameter>` and the tag after it, with
  no quotes. The front-end keys that span like any other
  (`ToolTreeBuilder::element_value`), and the session walks the held set's
  *raw form* there (`ValueRule::closed_by`): the set's trie, then — from a
  node where a value ends — only the strings that close the span, the marker
  followed by each tail its terminator closes on, so the rule closes on the
  same byte the terminator fires on. The call records the value up to its
  `</parameter>`, one framing line break taken off (`ValueText::Raw`). Until
  2026-11-14 these spans were not keyed, so on the live checkpoint every hold
  in this section held nothing: a reading of
  `candle-conversation/src/cancel.rs` wrote `src/cancel.rs` on every pass and
  was read again on every restart (pinned by
  `a_reading_writes_only_its_own_path_in_a_function_block`).
- **A catalog of one tool names its calls where it writes the name.** Every
  hold is looked up by the call's tool, and a name branch of one arm is folded
  into the run around it (§4), so a grammar over a lone tool — a file's
  reading offers `file_read` and nothing else — had no branch to name its calls
  and held none of them, in either dialect. `compile` marks the node such a run
  lowers to with the tool it names (`TreeMarks::names_on_entry`), and the walk
  begins a call of that tool on reaching it, as a name branch's arm begins one
  (pinned by `a_lone_tools_raw_values_are_held`).
- **A value chosen by a branch is recorded too.** An enum's value is a choice,
  not a span — and every file tool's `repo` is an enum, the workspace's
  repositories written into it at startup — so nothing a span records says
  which repository a call names. The front-end marks each enum's branch in the
  spec's `chosen` table with its field and, arm by arm, the value it writes (a
  nullable enum's `null` writes none); `compile` maps each arm's tokens to its
  value, as it maps a name branch's (`StencilTree::value_chosen`), and the
  session records the value when the arm completes, however the branch is left.
  A required enum with one value has no branch — its value is folded into the
  key — so the front-end records it per tool in the spec's `fixed` table, and
  naming the call writes it. Without this, every hold on `repo` found its
  field unwritten and held nothing: live, a listing of `crates` in `candle`,
  whose crates sit at its root, came back `not_found` while the holds were in
  place (pinned by `a_turn_of_listings_names_only_directories_that_exist`).
- **A spill into a held span keeps to the held rule.** Which spills a branch
  offers (§7.4) is settled at compile time against the span's *own* rule, and a
  spill's bytes past the arm are the value's first — committed the moment the
  token is taken. So the branch's mask leaves out each spill whose bytes the
  held rule for the span it runs into refuses; a spill that finishes a name is
  checked under the tool its arm names. Without this, `"path": ` closed by
  `"src` began a value the held set did not have, the rule could only leave the
  rest free, and a held reading still wrote `src/turn_layout.rs`.
- **The call's tool is the arm its name branch took**, mapped at compile time
  from each arm's tokens to the tool it names. However the branch is left — its
  last token, a token running past the arm into a value (a spill, which carries
  the arm it completed), or a delimiter pushed back from the span before it —
  the call is named the same way.
- **What a turn holds is the conversation's own** (`zend/src/held_values.rs`),
  read each round from the conversation's store, its own edits included:
  - **which repository:** every file tool's `repo` is one of the workspace's
    repositories (`RepoFiles::names`), and `file_grep.repo` and
    `file_search.repo` may also be `*`;
  - **which file:** the `path` of every tool that takes an existing file —
    `file_read`, `file_edit`, `file_delete`, `file_move`, `file_upload`,
    `file_image` — is one of the files of the repository named
    (`VfsStore::paths`, which leaves out protected paths);
  - **which directory:** `file_list.path` is one of the repository's
    directories or `""`, its root; `file_grep.prefix` and `file_search.prefix`
    one of its directories, with or without the closing `/`, or for `*` one
    of any repository's;
  - **which part and line:** for each file the conversation has read or edited,
    the anchors it carves into now and each part's line count.
    `file_read.part`, `file_edit.part`, `file_edit.through` and `file_edit.dest`
    are held to the anchors of the file named; `file_edit.from` and
    `file_edit.to` to `+1` through the length of the part named.

  A path the tool will create — `write.path`, `file_download.path`,
  `file_move.to`, `file_image`'s destination — names a file that does not
  exist yet, so it is not held. Neither is a git tool's path, which names a
  file at a revision, possibly one the working tree has since lost.

  This is what ended the repository-prefix collision: a file whose first
  directory is, or starts with, a repository's name (`candle-conversation/…`,
  `journals/…` in `candle`) was written without it, as though it were the
  repository prefix every read fence shows (`file=candle/…`), and the read came
  back `not_found`. Held, `src/turn_layout.rs` cannot be written in `candle`,
  whose root has no `src`.
- **An ingest's calls are held like a dialogue's.** A `repo_map` folder
  conversation answers its listing in a tool loop, and each decode runs under
  the restricted catalog's call grammar held to the chain's tools (`file_list`,
  `file_read`), with each folder's paths held to its commit-pinned stores
  (`zend/src/repo_scan/ingest_plan.rs`, `IngestPlan`). The pass refuses steering with
  no grammar on `<tool_call>` rather than run on it: holding passes such a
  registry through unchanged, and the repo map ran on one — the `<think>`
  trigger alone — until 2026-11-14, decoding its calls free. Live, a folder
  wrote `file_list` with no arguments three times, and sealed "I have enough
  evidence from the listing to summarize" as its summary.
- **A refused value is not offered again**, because it is not offered at all:
  the held values are read again each round from the file as it stands, so an
  anchor or a line that does not exist — the repeat the guard used to end — is
  never among them. The files a conversation has read or edited are noted
  after each round (`FilesSeen::note`), its 32 most recent held.

**Cost.** A round builds only what changed since the last one: a file is
carved again only when its content's digest differs from the one it was carved
from, and a value set's rule is taken from the last round's (`HeldRules`) when
the same set was built there. Both keep only what the current round used, so
neither grows past one round's files. A file the store cannot read is held to
nothing, and logged.

#### 9.2.1 A held set of any size: the trie rule

A pattern's `ValueRule` compiles every automaton state's mask when it is built,
so a turn's only cost is an index — and it caps a rule at 512 states, since each
state holds a mask. A literal set past that cap could not be held: one
repository's files (`candle`: 6,099 paths, 254 KB) is hundreds of thousands of
states.

A held set is therefore not a pattern. `ValueRule::of_set` builds a **byte trie**
of the set (`stencil/value_set.rs`), and a trie is already a deterministic
automaton: its nodes are the states (`ValueState::Inside(node)`), a node's
children its transitions, a value's last node accepting. What differs from a
pattern's rule is when the masks are made:

- **A node's mask is made the first time a walk stands on it**, by the same
  byte-order scan a pattern's build runs (§7.1): tokens sharing a prefix
  resume from it, and a prefix the trie refuses skips every token under it by
  binary search, so a node with a few children visits only the few tokens that
  could continue one. It is kept for the rule's life: a walk through
  `candle-conversation/src/` makes each of those nodes' masks once, for every
  later call.
- **No state cap**, because only the nodes walked are ever made. A path is ten
  to twenty tokens, so a call makes at most that many masks, each a scan of a
  narrow prefix of the vocabulary.
- **Finishable as a pattern's rule is**, but worked out over the trie at build:
  a node is finishable when it accepts, or when one of its children is reached
  by a token of that one byte and is finishable itself. That is exact for a
  byte-level vocabulary, which has a token for every byte; on any other it may
  hold back a node only multi-byte tokens reach, never offer one that strands
  the walk. A set none of whose values can be written is refused, and the
  field decodes under its own rule.
- **A value holding `"` or `\`, or a byte below a space, is left out of the
  set**: a rule admits no escape (§7.1), so no such value could be written.
- **One rule per set, shared while anything holds it.** The rules built over a
  vocabulary are kept by the set's digest behind a weak reference, so every
  conversation holding the same repository's files walks one trie and shares
  its masks, and a rule no round holds any longer is dropped.

#### 9.2.2 A turn that begins inside a grammar holds its values too

A turn armed by a trigger takes its registry's held values as the walk starts
(§9). A turn that begins inside a grammar (`TurnOptions::turn_grammar`, §9.1's
opening call) is armed before anything is sampled, with no trigger, so it
carries its own: `TurnGrammar { tree, held }`. A file's reading
(`zend/src/code_read`) uses it to hold its opening call to the one file it
reads: `repo` to its repository, `path` to its path, `lines` to its ranges. The
reading's prompt names all three, and the model — not the grammar — used to
write them, which is how a reading came to read `src/turn_layout.rs` for
`candle-conversation/src/turn_layout/` nine times over.

### 9.3 A tail: what an answer's end hands over to

A trigger starts a walk on a token the model writes. A **tail** starts one on
the token that would end the turn (`TriggerRegistry::with_tail`). An end token
sampled in free decode, with a tail bound, is dropped, not committed, and the
tail's walk writes in its place. The tail's own last static is the turn's
close, so the turn ends inside the stencil. A tail fires once, and never on a
replayed turn, whose recording already holds what it wrote. It fires only
after an answer (`scheduler/answered.rs`): an end token with nothing but
whitespace since the reasoning ends the turn as it stands. A reviewer
reasoned until its think block was closed for it, then sampled the end at
once. The tail wrote its `Steps:` onto the empty answer, which then passed
the check that sends back an answer that leaves its steps out.

A task's `Steps:` section is the tail (`stencil/answer_steps.rs`). It writes
the heading and, for each step, `N. <its words> — `. Each line is held to
`done`, `not done` or `could not`, then `: ` and the model's own line, which
ends at its newline. A section the model was only asked to write was left out
of 3 answers in 8 (`docs/archived/zen_coding_challenge_fixes.md` Cause 3).

---

## 10. Scheduler integration

Per sequence: an `Option<StencilSession>` beside its `DecodeState` (small; the
tree is shared `Arc`). The trees compile once at model/catalog load and live on
the scheduler in the `TriggerRegistry`. Only two scheduler changes:

1. **Sample-stage mask + EOS bias hook** in `batched_sampler/`: a per-row
   `Option<AllowedSet>` mask and a per-row close-token boost (§6.3), applied
   after penalties.
2. **Static-run drain** in `scheduler/decode/` (`play_stencil_runs`,
   `decode/stencil.rs`): a sequence owing a `Prefill` plays it as a row that
   rides the decode step's own wave (`scheduler/stencil_wave.rs`, §3) rather
   than a forward of its own; the §3 inner loop drains chained actions.

KV cache, projection, and persistence are untouched. The healing path (§7.3)
swaps the sampled token and forwards a prefix in the same step; it rewinds
nothing.

**Not built: decode-health suspension.** The design suspends the
`decode_health/` entropy and repetition floors while a session is active, since
stenciled regions are deliberately low-entropy and repetitive, through a
`stencil_active` flag the checks short-circuit on. No such flag exists. The
health checks in `scheduler/decode/commit.rs` run on a stenciled sequence like
any other, skipping only a token the stencil dropped (`dropped[i]`), and
`inside_think_block` is the only grammar-related state they read. The checks
run only when `health_config.enabled`, which is off by default.

### 10.1 Speculative decode inside a free span

A sequence whose walk sits in a free-text span — a thinking block's body, a
tool-call value — drafts like free decode does. Nothing is masked there, so a
drafted token is judged under exactly the rules a plain step would apply. A walk
at a branch, a static run or its exit takes a one-token row: its next token is
constrained or written, so there is nothing to propose.

A drafted block stops at the first token after which the next position would be
sampled under different rules (`scheduler/block_guard.rs`):

- the token leaves the span (a closing quote, a dropped `</think>` or EOS);
- the walk commits it as other bytes (a heal, §7.3), even when the span stays
  open;
- it opens a walk (a trigger token, in free decode);
- it follows a page-break token.

The accept walk stops there itself, rather than the commit loop discarding the
tail, because the sampler records every token it draws into the sequence's
penalty history. A heal's re-tokenized prefix is written into the KV after the
step's rollback, since until then the KV still holds the whole verified block.

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
| **Nested structure value** (`{…{…}…}`) | With a schema, the grammar writes every bracket (§8.3). Without one, the `JsonValue` lexer keeps a container stack, string-aware so brackets inside strings don't count: a mismatched closer ends the value and closes what was actually opened; a wrong delimiter *after* the value is dropped (§7.3). |
| **Malformed free value** (empty, `True`, `'x'`, `{k: 1}`, `[1 2]`, `[1,]`, `01`, a raw newline in a string, `\q`) | Where another language spells the same value differently it is respelled, keeping the meaning (`True` → `true`, never `null`). Otherwise the value ends at the first byte that cannot continue it and is completed as written (`null` only for a value that is missing). Never committed as invalid JSON. |
| **Closed or nullable field** (boolean, enum, `["T","null"]`, `anyOf` with null) | A choice under the mask, `null` among the arms where allowed — the wrong spelling is never offered. |
| **Token healing impossible** | Does not arise: every cut of a free span has a completion (§7.3), and a token the lexer cannot commit is rewritten, not rejected. |
| **EOS sampled in a session** | Masked out at every `Branch`; inside a `FreeText` without `eos_ends` it is never committed — replaced by the completion of the value as written (a closing quote, `]}`, ` null`, a `</parameter>` marker), or dropped when there is nothing to close. A session only ends at `End`. |
| **A raw value holding its own close tag** (a function-block value containing `</parameter>`, `</function>` or `</tool_call>` — a file about the call format) | A function block's value is an `Element` node (`NodeSpec::Element`): free text closed by `</parameter>` followed by a separator (`\n`, `\r\n`, `\n\n`, ` \n`, none) and the next element's tag, `<parameter=` or `</function>` — only the tags the tree offers next (`element_close::Follows`). Any other `</parameter>` is content. All of the close is consumed and the span forks on which tail fired (`FreeTextSpan::forks`, `TerminatorState::fired`): each fork is the continuation with that tag already written (`ToolTreeBuilder::after_written`). A value cut short is completed with `</parameter>` and goes on through the whole continuation. Every reader of the block — `function_blocks_to_json`, zend's splitter and parts, npcd's act parser — ends a value by the same rule (`element_close::value_end`, `function_end`), and the translated JSON writes a value's `<` as its JSON escape (`element_close::LT_ESCAPE`), so no tag survives inside a value. |
| **Model "fights" the stencil** | The mask is the last word at branches, so an out-of-grammar token shouldn't occur. As a failsafe, if one ever escapes the mask the session **bails** — logs `DEBUG`, emits the tree's configurable bail tokens to terminate the call cleanly, then exits. |
| **Empty / one-arm branch** | Invariant violations folded/checked at compile (§8.1); empty `debug_assert`s, release logs `ERROR` and aborts the session. |
| **Empty tool catalog** | No tree registered; `<tool_call>` isn't a trigger; tool calls decode freely. Logged once. |
| **Tokenizer ≠ compiled tree** | Fingerprint mismatch fails loudly at load. |

### 12.1 Traps: what can block, stick or loop a conversation

A grammar can do three kinds of harm a refused call cannot: leave a call with no
way on, keep a model repeating a call it cannot fix, or change what the model
decided without saying so. Each is held off by a named rule, and each rule is
tested where it lives.

| Trap | What holds it off |
|---|---|
| **A gate with no way on** | Every `Branch` has at least two ways on, or one and a spill (`compile.rs`). A choice among optionals ([`Pick`]) offers nothing past its last member while one is owed, so `opt_gates` can always pay; a gate left with one way on becomes a `Static`. |
| **A value that cannot finish** | A `ValueRule` needs a finishable fixpoint: every state of the walk reaches a close through the vocabulary, or the rule is refused at compile time, not during a call. End tokens spell no value. |
| **A value that never ends** | Every free span has `forced_after > 0`, refused at compile otherwise. A string's limit is `maxLength` in tokens plus its delimiters, and `MAX_STRING_VALUE_TOKENS` above that. |
| **A required value written blank** | A required free string's span is `nonblank` (`FreeTextLimits`). While nothing but whitespace is written in it, the step leaves out of its draw the tokens that would close it there (`blank.rs`, one scan per vocabulary): before its opening quote the tokens that open and close at once (` ""`, `"",`), after it the tokens that close (`"`, `",`). The session reads where the value stands from its terminator (`TerminatorState::blank`) and hands them on as `StepMask::Free`'s `barred`. An optional string, a value under a rule, and every other terminator are untouched. Lifting the presence penalty at a value's first step made the empty value less likely and left it possible: a model that had planned a field wrote `""` for it anyway. |
| **An array that runs on** | Arrays unroll to `maxItems`, at most `MAX_ARRAY_ELEMENTS`, and the last level writes only the close. `run_command`'s `args` stops at 24. |
| **A turn that loops on calls** | `MAX_TOOL_CALLS_PER_TURN` bounds a turn. Across turns the repeat guard answers the same failing call with a warning, and an argument repeated past `FLAG_REPEATS` is refused before anything runs. |
| **A heal that changes the decision** | A token that closes a value and carries more (`",`, `"]`) commits only its close, because the next node writes the rest. What it carried is kept (`carried`) and narrows the next branch to the arms that continue it, so the model is not asked again from a bare quote. A rest that leads no arm narrows nothing, so this can never empty a mask. |
| **A field order the model does not expect** | The grammar writes required fields first, in `required` order, then optionals. A tool's calibration examples must write its fields in that order, or a model that planned another order loses the field: `file_image`'s `source_repo` was shown second and could only be written last, and twelve calls in one run dropped it. |
| **A constraint stricter than the tool** | An overlay in `tool_constraints.yaml` may only block what the executor refuses, from the same source of truth. Known exceptions are deliberate and cost no capability: `credential_save` leaves out the two aliases `api_key` and `ed25519_key`. |
| **A tool that cannot succeed** | `tool_availability` takes it out of the round's name branch and says why in a sealed line, so a model is not left guessing ids. Each rule names a state in which every call to the tool fails. |
| **A stencil fault** | An out-of-grammar token bails: the tree's bail tokens close the call and the session exits. A hold refused after the fact is logged at `ERROR` (§9.1). |

**What a rule cannot see.** A pattern depends on one value and on values before
it, so a rule needing a later field or a file's contents stays with the
executor, whose refusal names the problem: `kdf_derive`'s salt, `bytes_pack`'s
value count, a regular expression's validity. A rule that depends on a value
works only when that value comes first, which is why `bytes_transcode` writes
`from` before `data` in its grammar and its examples.

---

## 13. Public API and configuration

```rust
// candle-conversation/src/stencil/mod.rs (the core of the re-exports)
pub use tree::{StencilTree, StencilNode, NodeId, FreeTextSpan, FreeTextLimits, TreeMarks};
pub use terminator::{Terminator, TerminatorState, Feed};
pub use session::{StencilSession, StencilAction, Observe};
pub use driver::{StencilDriver, Healed, StepMask, PathStats};
pub use mask::AllowedSet;
pub use trigger::TriggerRegistry;
pub use builder::StencilTreeBuilder;
pub use spec::{TreeSpec, NodeSpec, SpecId};
pub use error::{BuildError, WalkError};
pub use compile::compile;
pub use tool_call::{compile_tool_call_tree, compile_tool_call_loop, compile_action_loop,
                    ToolSpec, ToolCallEnvelope, Param, ParamType, Pick};
pub use think::{compile_think_tree, ThinkMode};
pub use sim::{simulate, Oracle, SimRun, SimError};   // test/diagnostic harness (§14)
// plus the held-value machinery of §9 (HeldValues, ValueRule, NeverTwice, ...)
```

Stenciling is on whenever a non-empty catalog and a tool-call trigger exist,
otherwise a no-op (an empty catalog registers no tree, §12).

**Not built: the ablation flag.** The design calls for a `SequenceConfig` flag
that disables stenciling (free tool-call decode) without removing the catalog
from the prompt. No such flag exists. `TriggerRegistry::without_trigger` removes
a tree's trigger, which has the same effect on a registry a caller builds.

---

## 14. Testing — 100% coverage, simulator-driven

The module is **fully unit-testable without a GPU**: everything except the
forward pass is pure (compiler, session, terminator, mask construction). The
suites are scenario and edge-case tests beside the code (`tests.rs`,
`edge_tests/`, `spill_tests.rs`, `held_tests.rs`, `held_values_tests.rs`,
`shaped_params_tests.rs`, `sub_stencil_tests.rs`, `driver/tests_*.rs`,
`tool_call/tests_*.rs`). A **simulator** drives sessions deterministically. No
coverage gate is enforced; a 100% line/branch bar was the first sketch's target
and is **Not built** as a measured gate.

### 14.1 The simulator (`sim.rs`)

```rust
/// Drives a StencilSession without a model: an oracle supplies the next token at
/// each MaskedDecode/FreeDecode; the run records the action stream and the
/// tokens that entered the KV.
pub enum Oracle {
    Scripted(Vec<TokenId>),                              // exact tokens
    Policy(Box<dyn FnMut(Option<&AllowedSet>) -> TokenId>),  // Some at a branch, None in free text
}
pub struct SimRun { actions, observes, tokens, dropped, healed_bytes, forced_closes }
pub fn simulate(tree: Arc<StencilTree>, vocab: &dyn Vocab, oracle: Oracle,
                max_steps: usize) -> Result<SimRun, SimError>;
pub fn lowest_arm_policy(free_token: TokenId) -> Oracle;
```

`Scripted` asserts an exact `(action, token)` trace; a `Policy` picks the arm or
value, so a test enumerates paths through a tree and asserts each emits parseable
JSON matching the schema. `SimRun::text` re-decodes `tokens` to a string for an
end-to-end assertion against a real JSON parser — all on CPU. A dedicated
adversarial oracle type is **Not built**; the adversarial cases are scripted and
fuzzed in `edge_tests/` (`fuzz.rs`, `intent.rs`).

### 14.2 Component suites (byte-exact)

- **Terminator** (`terminator.rs`): for `JsonString`, `JsonNumber{int,float}`,
  `Balanced`, `JsonValue`, `Until`: plain close; `\"` and `\\` escapes; `\\"`
  (escaped backslash then real quote); brackets inside strings; deeply nested
  `{[{}]}`; number lookahead (`123,` and `1.5e-3}`); UTF-8 multi-byte values; a
  terminator split across two tokens; empty value. Assert exact `Feed`
  (`consumed`, `Rewrite` bytes).
- **Compile backend** (`compile.rs`): single-arm fold; static fusion; in-context
  tokenization (boundary merges); trie sharing of common prefixes; invariant
  failures (empty arm, adjacent statics not fused, missing `End`, `forced_after
  == 0`) → `BuildError`. Raw token-id assertions, not tolerances.
- **Session** (`session.rs`): the §5 state machine via `Scripted` — minimal call;
  optional included/skipped; enum; number lookahead push-back; string with
  escapes; `Balanced` value; `eos_ends` true/false; soft ramp / graceful / forced
  limit firing; the healing boundary (`Observe::SpanClosed { leftover }`,
  `Repaired`, `DelimiterDropped`); the bail failsafe.
- **Mask** (`mask.rs`): `frontier` correctness at each trie step; EOS exclusion;
  one-arm never reaches the sampler; per-row application over a synthetic logits
  matrix; close-token boost wiring.

### 14.3 Construction equivalence

`tests::builder_and_yaml_agree` asserts the **builder** and the **YAML**
front-ends compile to the same tree for one grammar. The same assertion for the
**JSON tool description** front-end is **Not built**: the tool compiler lowers
through its own `ToolTreeBuilder` and is covered by the tool-library tests
(§14.4) rather than by comparison with a hand-written grammar.

### 14.4 Tool-library construction tests

`zend/tests/stencil_tool_call.rs` takes the **actual tool definitions from the
tool library** (the same JSON the projection's `tools` collection uses),
compiles them to one tree, and runs the simulator (`every_tool_minimal_call_is_valid`
and the masking tests beside it). The scenario set the design asks of every tool
is: every required field; each optional
present and absent (and combinations, capped); each enum value; arrays empty,
singleton, and multi; nested objects; nullable fields as value and as `null`.
Assert every produced byte stream parses and validates against that tool's
schema, and that the **tool name and every enum are exactly from the catalog**.
This is the regression net that guarantees the stencil never lets a real tool be
mis-called.

### 14.5 End-to-end (GPU, integration)

**Not built** as a test: prompt a real model for tool calls and assert the
produced bytes parse, name ∈ catalog, required present, enums ∈ allowed, across
all catalog tools. What exists is the live *tokenizer* tier
(`full_catalog_compiles_against_real_qwen3_tokenizer`,
`the_rows_a_tool_call_plays_on_the_live_tokenizer`,
`natural_calls_drive_the_live_grammar` in `zend/tests/stencil_tool_call.rs`),
which drive the grammar with natural token streams and need a cached tokenizer,
and the live daemon's own logs (`PathStats`, `log_stencil_finish`).

---

## 15. Module layout

```text
candle-conversation/src/stencil/
  mod.rs          public API, re-exports
  tree.rs         StencilTree, StencilNode, NodeId, FreeTextSpan, FreeTextLimits, TreeMarks
  trie.rs         TokenTrie, frontier(), spills, in-context name/enum trie builder
  terminator.rs   Terminator, TerminatorState, Feed (byte lexer)
  json_lexer*     the incremental JSON lexer behind JsonString / JsonValue
  session.rs      StencilSession, Cursor, StencilAction, Observe, next_action()/observe()
  driver/         StencilDriver: the online walker the scheduler drives (Healed, StepMask, hold)
  mask.rs         AllowedSet, logit masking + close-token boost
  trigger.rs      TriggerRegistry (trigger / once / tail / held bindings)
  spec.rs         TreeSpec, NodeSpec, SpecId (the string-space intermediate)
  error.rs        BuildError, WalkError
  compile.rs      compile(): validate, tokenize-in-context, fold, fuse, verify invariants
  builder.rs      StencilTreeBuilder (front-end A)
  tool_call/      compile_tool_call_tree + ToolSpec/Param/ToolCallEnvelope → TreeSpec (front-end B)
  yaml.rs         TreeSpec::from_yaml (front-end C)
  think.rs        compile_think_tree, ThinkMode (front-end D)
  value_rule.rs, value_set.rs, held_values.rs, held_arms.rs, call_so_far.rs,
  never_twice.rs, named_arm.rs, opening_call.rs, turn_grammar.rs,
  element_close.rs, function_block.rs, answer_steps.rs   §9 and §12 machinery
  vocab.rs        Vocab, TestVocab, HfVocab, TokenBytes
  sim.rs          simulate, Oracle, SimRun (test/diagnostic harness)
  *_tests.rs, edge_tests/, tests.rs   scenario and edge-case suites
```

One concern per file. The scheduler/sampler touch-points are hooks in
`scheduler/decode/` (`commit.rs`, `stencil.rs`), `scheduler/stencil_wave.rs`,
`scheduler/block_guard.rs`, `scheduler/named_tool.rs`, `scheduler/answered.rs`
and `batched_sampler/allow_list.rs`; no existing KV, projection, or persistence
code changes.

---

## 16. Layers and dependencies

The module is layered so each layer tests without the next:

1. **Pure core** — `terminator` and `json_lexer` (byte lexers), `tree`, `trie`,
   `spec`, `compile`: no scheduler, no model.
2. **Front-ends** — `builder`, `yaml`, `tool_call`, `think` produce a `TreeSpec`.
3. **Walk** — `session` and `driver`, driven by `sim` in tests; `mask` applies
   `AllowedSet`s and close-token boosts to a logits matrix.
4. **Triggers and holds** — `trigger`, `held_values`, `held_arms`, `never_twice`,
   handed to the scheduler through the engine's registry.
5. **Scheduler hooks** — the decode commit path (`Healed` signals, heals,
   drops), the wave rows of `stencil_wave.rs` and the block guard.

Not built: decode-health suspension (§10), the ablation flag (§13), the
model-driven end-to-end test (§14.5).
