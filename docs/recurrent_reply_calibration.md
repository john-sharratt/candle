# Recurrent reply calibration: a served read the recurrence has seen

**Status: withdrawn (2026-11-04).** The stored records cost ~20.8 MB per page
on every member (§3), which the substrates could not carry. The calibration,
the records and the live replay are removed; a served page's reply prefills as
any other again. Each calibration pass ends whatever records remain
(`zend/src/calibrate/retire_replies.rs`: the object ceased, its blob retired).
The span arms in the engine stay: the diagnostic decode captures and merges
rows in memory. What follows is the design as it was built, kept for the
measurement in §1.

## 1. The problem, and what was measured

A `file_read` the corpus already holds is served by locking the file's
`code_reading` conversation into the dialogue's working set and answering
`{"status":"in_context","anchor":"file=<repo>/<path> page=N pages=M"}`
(`zend/src/fast_path.rs`, `docs/zend_working_set.md` §4.6). The page's K/V is
placed ahead of the dialogue, and every attention layer reads it.

The recurrent layers never do. Flash-Next is 36 Gated DeltaNet layers and 12
attention layers. A served page enters the projection as K/V only, so the
recurrent state the dialogue carries has run over its own lineage and nothing
the working set placed. Measured with the capped diagnostic decode
(`POST /v1/conversations/{id}/diag/decode`, conversation
`candle-incontext-probe/small-guide`), 2026-11-02 and 2026-11-03:

- The page is placed (2,930 tokens) and attended: its best block ranks first and
  is read by 1,136 of 1,209 layer queries. Dense attention gives the same
  answer, so it is not selection. Forced to open a doc comment from the page,
  the model completes it verbatim; with the page removed it invents one.
- Asked whether it can see the page, it says no: 6 of 6 seeds before any change.
- Eight rewordings of the tool reply: 17 of 18 runs still say no.
- The system prompt's `in_context` rule rewritten to "find it by that anchor,
  never call it missing" (`eff34c54d`, reverted `3b539b47b`): the model obeys,
  searches for the anchor string, and says it genuinely cannot find it in 8 of 8
  continued decodes.
- Every list it makes of what it has read names only its lineage: the priming
  chain and `primary_conversation`, positions 9k–50k. The page sits at ~64.6k in
  `scopes`, in the same `Read the entire contents of …` + fenced-page form, and
  is never named.

Reading works and introspection fails, and what introspection reaches is exactly
what the recurrence ran over. Replaying the page's tokens through the dialogue's
own recurrence after the call (`DiagRun.catch_up`, `757ce6599`) changed the state
and did not change the answer: the recurrence was given the page, but never
processed the reply as a reader of the page would.

## 2. The idea

Calibrate, per page, **how a reader's recurrence takes the `in_context` reply**,
and give a live conversation that recurrence when it is served that page.

At calibration the unit's own reading is live: a fork of the layer's base with
the unit as its lineage projects the reading, and `seed_recurrent_from_lineage`
gives the fork the recurrent state the reading left. Prefilling the call and the
`in_context` reply there runs the reply through a recurrence that has read the
page. The reply's per-token inputs to the delta rule are recorded. A live
conversation served the same page runs those inputs through its own state at the
reply, so the state it carries on is what a reader's would be, applied to
everything else it knows.

## 3. The record

### 3.1 Span: the reply block

The record covers the served result's block and nothing else:
`<tool_response>` + the reply JSON + `</tool_response>\n`, about 28 tokens. The
reply is byte-identical on every serve of a page (`served_response`); the call
is not — the model writes its arguments in different orders, with or without
`page` — so the call is prefilled in the fork as context and is not recorded.

### 3.2 Rows: the scan's own inputs

Per DeltaNet layer, per token of the span, the record holds exactly what the
fused prefill scan reads for that row (`delta_net_prefill_scan`,
`candle-transformers/src/models/delta_net/cuda.rs`):

| Field | Width (Flash-Next) | What it is |
|---|---|---|
| `qkv` | `conv_dim` = 2·16·128 + 48·128 = 10,240 | the convolved `[Q\|K\|V]` row |
| `alpha` | `n_v_heads` = 48 | the raw `ssm_alpha` projection, before bias and softplus |
| `beta` | `n_v_heads` = 48 | the raw `ssm_beta` projection, before sigmoid |

The decay `g` and the write strength `β` are derived from these by the scan with
the layer's own `dt` bias and `A`, which are weights and are not stored.

**Not the timeline.** The short-span timeline (`delta_net/timeline.rs`) stores
`G`, `k` and `v_new = β·(v − S·k)`. `v_new` is the correction against the state
it was computed on, so folding it is exact only onto that same state. A record
is applied to another conversation's state, so it stores the raw value `v`
(inside `qkv`) and the raw `beta`: the delta rule is run again against the
target's own `S`, and the correction is the target's.

`Q` is stored with `K` and `V` so the scan runs unchanged: its input layout is
the fused `[Q|K|V]` row. Dropping `Q` would save a fifth of the record and need
a scan variant that reads `K` and `V` alone; this design keeps one scan.

### 3.3 Precision and size

Rows are stored BF16, and widened to F32 as they are uploaded for a replay
(§7.4) — the narrowing is the storage format's, by design. Per token:
10,336 values × 36 layers × 2 bytes = **744 KB**. For a 28-token reply,
**~20.8 MB per page.** The sweep (§6) writes one record per page of every
`code_reading` unit, and the substrate replicates it to every member: a corpus
of N pages costs ~20.8·N MB on each member's disk.

### 3.4 Layout

One blob per record, little-endian, field-major per layer so a layer's rows are
one contiguous run the upload copies in one piece:

```text
header:  magic "RRC1" | n_layers u32 | n_tokens u32 | conv_dim u32 | n_v_heads u32
         | layer indices [n_layers] u32
layer l: qkv   [n_tokens][conv_dim]   bf16
         alpha [n_tokens][n_v_heads]  bf16
         beta  [n_tokens][n_v_heads]  bf16
```

The layer indices are the model's DeltaNet layers in order (`KvLayerMap`), so a
record names which layer each run belongs to rather than assuming the order.

## 4. What a record depends on

Two halves, treated differently.

**Identity — must match, or the record is never used.**

- The unit's content key (`CONTENT_KEY`: path and blob) and the page. A new
  version of the file is a new unit: its old record is never looked up, so no
  hash is needed for content.
- The model. A substrate is bound to the model that wrote it and refuses any
  other, so a record in it is this model's. The rows also carry their DeltaNet
  geometry — the layer indices, `conv_dim`, `n_v_heads` — and an arm refuses
  rows whose geometry is not the model's (`SpanArms::arm`).

**Recipe — may be stale; the old record is used until the new one lands.**

```text
recipe = SHA-256( request ‖ 0 ‖ call ‖ 0 ‖ reply ‖ 0 ‖ frame system prompt ‖ 0 )[..8]
```

The texts are exactly what the fork prefills after the unit: the layer's reply
request, the call, and the reply's tool-round text (`format_tool_blocks`, the
function a live round's results turn is built with). A change to the reply's
format, the anchor's, the call's canonical form, the wrapper or the request is a
new recipe. The chat template and tokenizer belong to the model, which the
substrate already pins.

## 5. Where a record lives

A custom object holds the record's identity and recipe and names its rows; the
rows are a blob. Custom objects are held in RAM (`Substrate::custom_object`);
blobs are content-addressed on disk and only indexed in RAM
(`SubstratePersistence::write_blob`, `docs/replicated_substrate.md` §10.3). Both
replicate.

```text
key:    recurrent/<content key>/<page>
value:  { "recipe": "<16 hex>", "rows": "<blob sha256, hex>" }
```

The store is `zend::reply_record`: the object through the `Records` trait every
`/mcp` record uses, the rows through the substrate's blob store.

Reading a slot is one key lookup (`custom_object`), never a scan of the prefix —
the request path's no-full-scan rule. A later write of the same key supersedes
the earlier one, so a record is replaced atomically. The blob it superseded is
tombstoned (`retire_blob`) once the new object is committed, never before.

## 6. The calibration step

A new step of the `code_reading` layer's calibration, run where calibration
already runs:

- **At commit** (`calibrate_committed`), after the questions, while the reading
  is still resident — one record per page of the unit just read.
- **In the calibration pass** (`calibrate_layer`), as a sweep: every unit of the
  layer that is ready by the questions' rules (`units_of`: current `KEY_VERSION`,
  chain whole, every turn held) and has a page whose slot is missing or whose
  recipe differs from the current one. Units read before this step existed are
  calibrated by the first pass, without being read again.

The layer opts in with a `reply:` request in its `calibrate:` block:

```yaml
- name: code_reading
  calibrate:
    questions: 24
    think: 64
    prompt: …
    reply: |
      Read page <<page>> of `<<path>>` in the `<<repo>>` repository using file_read.
```

`reply` is the user turn the call answers in the fork (§6.2). It is part of the
recipe, so rewording it recalibrates every page; it is not part of the
questions' marker, so it recalibrates no questions.

### 6.1 The plan

The plan is the questions' plan rule (`calibrate/plan.rs`) over slots:

| Slot state | What the pass does | What a live serve does |
|---|---|---|
| missing | calibrate | no injection |
| recipe differs | calibrate | inject the old record (§7.3) |
| recipe matches | nothing | inject |
| unit no longer present | tombstone the object and its blob | — |

A record of a present unit stands until its replacement is committed; a unit is
never left without one in between.

**One member calibrates a slot.** The pass takes the calibration lease on the
slot's key and recipe (`ingest_lease::share_calibration`), and looks again after
taking it: a record another member wrote and replication brought here is not
written twice.

### 6.2 Calibrating one page

1. Fork the layer's ingest base onto the unit, as the questions' draft does
   (`draft`): transient, unsummarised, `set_forked_from(unit)`, then
   `seed_recurrent_from_lineage`. The fork projects the unit's reading and
   carries the state the reading left. Each page gets its own fork, so every
   record starts from the reading's state, not from another page's reply.
2. Prefill turn A with `submit_prefilled_turn`: the user half is the layer's
   `reply` request for this page, and the assistant half is the canonical call,
   `<tool_call>\n{"name":"file_read","arguments":{"repo":R,"path":P,"page":N}}\n</tool_call>`,
   with nothing decoded.
3. Run turn B: the user half is the served reply for this page, built by
   `format_tool_blocks` from `served_response` exactly as a live round's is,
   with the block armed for capture (§6.3). It decodes one argmax token, which
   is the draft's and goes with it.
4. Write the rows as a blob, then the slot's object, then tombstone the
   superseded blob. Tombstone the fork.

The frame is the ingest frame the questions' draft runs under — the layer's base
and its system prompt, with tools off. The recipe carries that prompt's digest,
so a change to it recalibrates.

### 6.3 Capture

A turn names a span of its user half by its bytes (`TurnOptions::recurrent_spans`,
`<tool_response>` through `</tool_response>`). The conversation maps the bytes
to tokens (`turn_text::token_span`: both ends fall on a tag, so each prefix of
the text encodes to a prefix of the whole — checked, not assumed), and shifts
them past the turn's baked head. The scheduler cuts the turn's prefill so the
span is a chunk of its own (`recurrent_span::snap_advance`, the last step of
forming a wave group) and arms the model for exactly that chunk, by its absolute
position and length (`arm_recurrent_span`). The arms live in a model-level map
beside the recurrent stores (`delta_net::SpanArms`), committed and rolled back
with each wave, so a creep that sweeps the layers a window at a time keeps its
arm until every DeltaNet layer has run the span.

A capture arm allocates each layer's rows on the device; the mixer copies the
span's `conved`, `alpha_lin` and `beta_lin` rows into them, on the scan's
stream (`capture_spans`). When the turn's last chunk lands the scheduler takes
the rows (`take_recurrent_capture`) — BF16 on the host — and sends them on the
turn's stream as `TurnEvent::RecurrentCapture`, before `Done`. Only a turn that
armed a capture copies anything, and only calibration arms one.

## 7. Applying a record in a live conversation

### 7.0 The page is placed before the reply prefills

The injection rests on the live turn running in the calibration fork's order:
the page, then the call, then the reply. It does, and §10 pins it with a test:

1. The round's calls are screened before any runs (`session.rs`, the
   `fast_path::screen` call ahead of dispatch). A hit is returned only once
   `working_set_lock` has locked the page's unit into the dialogue's working set.
2. The tools run, and the results become the next turn's user half
   (`current_message = format_tool_responses(&results)`).
3. That turn is submitted, and its **opening projection** runs at submit, before
   anything of it is prefilled (`SchedulerRequest::SubmitTurn`,
   `project_with_mode_and_sink` at decode position 0, `scheduler/mod.rs`). The
   `working_set` selection rule reads the working set as it stands then
   (`ContentResolver::working_set_members`, `docs/zend_working_set.md` §4.4), so
   the lock from step 1 is in it. The submit projection is skipped only for
   ingest timelines (`skip_projection` requires `disable_reprojection`), never for
   a dialogue.
4. The projection's turns are elevated and placed into the slot
   (`apply_projection`). A promised turn that cannot be placed fails the
   projection instead of dropping out of it (`is_promised`,
   `projection_assembler.rs`), so the page is never silently missing.
5. Only then does the turn's user half — the reply — prefill, and the reply
   span's replay (§7.2) runs in that prefill.

So the page's K/V is in place when the reply prefills, and the reply's state
comes from the record in the same pass; nothing waits for a mid-decode
reprojection. On the probe conversation the opening projection of the served
turn (`/projections/1/0`) carries the page's reading.

### 7.1 Serve

When the fast path serves a page (`Screen::read_hit`), the hit carries the unit's
content key, the page and the reply it was answered with (`Served`). Once the
round has run, the session builds the results turn with `format_tool_blocks` —
its text and each result's block by bytes — and, on the blocking pool, gives
every block a served read answered a replay span with its page's record
(`zend::reply_spans::replay_spans`, one key read and one blob read per served
read). The served result is found by its reply, not by its position, so a
notice the round appends moves nothing. A served page with no record replays
nothing.

### 7.2 The span's state comes from the record

The results turn is submitted with one span per served block that has a record:
the block's piece range and the record's rows. The scheduler makes each such
span a prefill span of its own — cut at the block's first and last tokens — and
runs it on the long path whatever its length.

In that span's wave, each DeltaNet layer runs the prefill scan twice:

1. **The live scan**, as always: the span's own rows give the span's outputs —
   the activations every later layer and the span's attention K/V are built from
   — and write the span's advanced state to its state-out.
2. **The replay scan**, after it on the same stream: the record's rows, entering
   from the same entering state, written to the **same** state-out.

The entering state is not touched until the wave commits, so both scans read
the state at the span's start. The replay overwrites the live scan's state, and
`commit_wave` exchanges it in as it would any prefill span's. The state the
conversation carries past the reply is the record's rows applied to its own
state: `S_end = scan(record rows, S_start)`. The conv tail is the live span's —
the replay does not run the conv — so the tokens after the reply convolve
against what this conversation actually prefilled.

The attention layers are unchanged: the reply's K/V is what this conversation
prefilled.

### 7.3 Stale and missing records

Replacing the state does not depend on the record's length. A stale record — a
reply whose token ids have changed since it was calibrated — still replays onto
the span's entering state and still gives the state a reader of that page would
carry; it is applied until the sweep replaces it. A missing slot applies
nothing, and the reply prefills as it does today.

### 7.4 Batching and the upload

A replay arm uploads each layer's rows once, when the span is armed — BF16
widened to the F32 the scan reads, the narrowing being the storage format's.
In the span's wave each DeltaNet layer runs one extra launch per replayed span
(`replay_spans` in `delta_net/mix.rs`): the same `delta_net_prefill_scan`, over
a one-span table built by the same `build_span_tables`, pointed at the record's
rows. A replay is one per served read, and a wave carries one about as often as
a turn serves a read, so the launches are not batched across spans.

### 7.5 Seal, restart, replay

The sealed turn's recurrent snapshot is the state after the injection, so a
conversation resumed from it carries it and nothing is applied again. A
diagnostic decode applies it when asked (`DiagRun::inject`): each `in_context`
block of the target's user half replays its page's current record, calibrated
on the spot when the page has none. A recorded-turn replay of history does not
re-apply records.

## 8. Generality

Two operations carry the mechanism, and neither knows about files:

- **Calibrate** `(unit, span text) → rows`: fork onto the unit, seed its state
  from the lineage, prefill the span with its block captured.
- **Apply** a turn's span with rows: its state comes from the rows.

They live in `candle-conversation`, model-agnostic: a model with no recurrent
layers records nothing, and a span with rows applies nothing, as
`seed_recurrent_from_lineage` returns `Ok(false)` there. What to calibrate and
when to apply belong to zend. A served answer that stands for a placed unit
names `(unit, slot, reply text)`. Today that is a `file_read` page; another kind
of placed content adds a slot of its own, and the calibration step, the record,
the plan and the scheduler hook stay the same.

## 9. Invariants

- **Hot path.** No host compute over rows: the upload is one H2D copy and one
  widening kernel per wave, and the replay is one launch per layer over every
  injected span (invariants 4 and 5, 2b). No device-to-host copy on a live turn;
  capture runs only for a calibration turn (invariant 3). The replay's state-out
  is the span's existing prefill scratch; its transients are wave-arena
  allocations the kernel fully writes (invariant 6).
- **Request path.** The serve does one key read and no scan; the rows are read
  off the request path while the round runs (MCP request-path rules 2, 4, 6).
- **Persistence.** Records are substrate objects and blobs: written through the
  redo log, replicated, compacted like any other.

## 10. Tests

Unit tests, written with the code:

- The record's encoding against raw expected bytes, one layer of two tokens.
- The recipe: equal for equal ids and prompt, different for one changed id or a
  changed prompt, against fixed expected digests.
- The plan: missing, stale, fresh and gone slots, including that a stale record
  stands until its replacement commits.
- Piece range to token range for a single-result and a three-result round,
  served block in the middle.

Kernel parity, on the card:

- **Self-replay is exact.** Replaying a record onto the fork's own entering state
  reproduces the fork's state-out, bit for bit, at F32 rows; at BF16 rows, it
  matches the scan run on the same BF16-rounded rows bit for bit.
- **Replay onto another state** matches `delta_recurrence` (`delta_net/mix.rs`)
  over the same rows from that state, within the reference's parity tolerance.
- **Live outputs are untouched.** A span with a record and one without produce
  the same span outputs and K/V bit for bit; only the state past the span
  differs.

Order, in `candle-conversation` against a real substrate:

- **The served page precedes the reply.** Lock a unit into a dialogue's working
  set, submit a turn whose user half is the served reply, and assert from the
  turn's opening projection event and its placed pieces that the unit's turns are
  placed and that every one of their positions precedes the user half's first
  token. A unit locked after the submit is not in that opening projection.

Behavioural gate, through the diagnostic decode:

- `DiagRun.inject`: the target's served block applies the slot's current
  record. On `candle-incontext-probe/small-guide`, six seeds: the model names the
  page as read and quotes `INVALID_ARGUMENTS` and `SUGGESTIONS`, where today it
  says NOT VISIBLE in 6 of 6. The same six seeds under a probe that does not
  invite refusal ("Quote only what you can see …" removed), as the control.
- A conversation that is never served a page decodes bit for bit as before.

## 11. Order of work

1. Capture and apply, with `DiagRun.inject` computing the record on the spot —
   the fork, the capture and the replay scan, no store — and the behavioural gate
   run on it. This is the mechanism the sweep stores; if it does not change the
   answer, the store is not built.
2. The record, its store and the plan; the step at commit and in the sweep.
3. The serve, the marks, the spans, and the replay on restart.

## 12. Decisions

Taken 2026-11-03:

- **The span is the reply block only**, not the call: the reply is byte-identical
  per page; the call is not.
- **Rows are BF16.**
- **The fork runs under the ingest frame** — the questions' draft, as it is.
- **A stale record replaces the state**, so it is always usable until the sweep
  replaces it.
- **The sweep calibrates every page of every unit**, and the storage of §3.3 is
  accepted.

## 13. Where it is built

| Piece | Where |
|---|---|
| Rows and their blob | `candle-transformers/src/models/delta_net/span_rows.rs` |
| Arms and hooks | `candle-transformers/src/models/delta_net/span_arm.rs`; `DeltaNetSeq::hook` |
| Capture and replay in the mixer | `capture_spans`, `replay_spans` in `delta_net/mix.rs` (fused path only; the ops path refuses a hook) |
| Model trait | `arm_recurrent_span`, `take_recurrent_capture`; Flash-Next in `qwen4exp/wave.rs` |
| Byte span → tokens | `turn_text::token_span`, `TurnText::prefix` |
| Spans, the cut | `candle-conversation/src/recurrent_span.rs`; `Scheduler::cut_at_recurrent_spans` |
| The record | `zend/src/reply_record.rs` |
| Calibration | `zend/src/calibrate/reply.rs`; at commit and in `sweep_replies` |
| Live apply | `zend/src/reply_spans.rs`; the session's results turn |
| Diagnostic | `DiagRun::inject` |

Each one rests on code that existed: the draft fork (`on_a_draft`, shared with
the questions), `submit_prefilled_turn` for turn A, the fused scan and its span
tables, the calibration lease, and the `Records` and blob stores.
