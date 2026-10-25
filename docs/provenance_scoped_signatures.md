# Provenance signatures scoped like projection

**Status:** §2.1 built — `provenance::boundaries` is the one rule, and the
decode loop steps it; the gallery windows (§2.2) and the aggregations (§2.3)
are next. Supersedes the per-turn gallery windows the belief scan uses today
for ingested conversations. The acceptance test is
`docs/paper_conversation_replay.md`; this document adds the gates that come
before it.

## 1. The problem

Provenance compares a **probe** — what a projection looks for — with a
**gallery** — what each ingested conversation is. The two are cut by different
rules, and the gallery's rule does not match the unit projection selects.
Measured on the zend substrate, 2026-10-25.

### 1.1 How projection cuts its probe

A boundary fires during decode (`scheduler/decode.rs`):

- on a line-feed token (`reproject_trigger_texts: ["\n"]`), once more than 16
  non-trigger tokens have been decoded since the last boundary;
- every `reproject_every_n_tokens` (64) decoded tokens, the cadence;
- inside a `<tool_call>` body never: one boundary fires at `<tool_call>`, and
  `</tool_call>` re-enables the triggers.

At a boundary the probe is (`scheduler/mod.rs`, the reproject probe):

- the turn's first `QUERY_HEAD_CHUNKS` (2) chunks — 64 tokens, the query head —
  once the trailing window has slid past them;
- the turn's trailing chunks, at most `reproject_max_probe_tokens` (256)
  tokens, aligned to 32-token chunks;
- and a second probe, the turn's user half, at most 256 tokens (Concept F).

So a projection asks one line-bounded question of at most 256 tokens plus its
head.

### 1.2 How ingest cuts its gallery

- At seal, every real token of the turn gets a folded `sign(Q)` signature,
  1:1 with the turn's `Tokens` record (`Scheduler::gather_wide_sigs`). The
  reasoning block is stripped; a block whose R16 K/V is already gone at seal
  contributes nothing.
- A turn is split into windows only at **seams**, the start tokens of the
  turn's own self-referencing projection events (`Substrate::decoded_seams`) —
  the points where that turn reprojected while it was decoded.
- **Prefilled content never reprojects, so it has no seams.** A file's pages,
  a folder's listing and the ingest request are each one window covering the
  whole turn. Decoded parts (a closing summary) are cut into the disjoint spans
  between their boundaries, which is not the probe's shape either.
- `belief_files::assemble_file_scans` then merges a tool call and its response
  into one case (an *exchange*), and the scorer (late fusion, `z × margin`)
  scores each case by its **single best-matching token** against each probe
  token.

### 1.3 What that does

| | |
|---|---|
| `code_reading` conversations | 2,091; 1,792 of them two turns, one exchange — one case per file |
| tokens per file's case | median 4,507, p90 22,918, max 204,426 (`substrate.rs`) |
| the paper's case | 45,564 tokens: the request and six calls (414), pages 1–6 as **one 42,170-token window**, page 7 and the summary (2,980) |
| the probe | at most 256 tail tokens plus the 64-token head, and a ≤256-token question |

A case scores by the best of its tokens, and the best of more tokens is higher
by chance, so file size decides rank: on every question of the paper
conversation the working set's provenance budget filled with the same large
test files (`flat_prefill_probe.rs` 37K, `paged_glue_tests.rs` 22K,
`decode_layout_exact.rs` 15K). The request template and the tool-call JSON are
near-identical in every file's case and match the dialogue's own tool-round
tails, which crowned test files whenever the probe was a listing.

## 2. The design

**Ingest's windows are projection's probes.** Every window the gallery scores
is cut by the same formula, to the same shape, as a probe a projection would
build at the same point.

### 2.1 One boundary formula

The trigger rule leaves `decode.rs` as a pure function over a token stream:

```rust
/// Where a projection's boundaries fall in `tokens`.
pub fn boundaries(tokens: &[u32], rule: &BoundaryRule) -> Vec<usize>;

pub struct BoundaryRule {
    pub triggers: Arc<Vec<u32>>,     // reproject_trigger_texts, tokenized
    pub content_gate: usize,         // 16: non-trigger tokens before a trigger fires
    pub cadence: usize,              // reproject_every_n_tokens
    pub tool_call_open: Option<u32>, // one boundary here, then suppressed
    pub tool_call_close: Option<u32>,
}
```

The live decode loop asks it token by token (its state is the count since the
last boundary and whether it is inside a call); the gallery asks it over a
stored turn. One rule, one implementation: a decoded turn's live boundaries and
the boundaries computed over its stored tokens are equal, and a test holds
them so.

### 2.2 Windows mirror the probe

For every turn of an ingested conversation — decoded or prefilled — the gallery
computes its boundaries, and at each boundary `b` its window is what the probe
would have been at `b`:

- the turn's first 64 tokens, when `b` is past them;
- the tokens before `b`, back to at most 256, chunk-aligned as the probe is.

Windows overlap, as successive probes do. They are offsets into the turn's
stored signatures, so they cost no signature storage. A massive prefill — a
file's pages — captures its per-token `sign(Q)` exactly as any prefill does,
and is carved by the same rule afterwards: capture never changes, only the
cut. They are computed at
gallery assembly from the turn's stored tokens and memoized per stream, as
seams are today; **no re-ingest** is needed to adopt them.

The question probe has its gallery counterpart in the turn's user half, cut
the same way.

### 2.3 A file's score

The working set selects a whole conversation (a file or a folder), and an
analysis layer a whole unit, so the gallery must give each one a single score
that does not grow with how many windows it has. Three candidates are built
and measured against each other by the gates in §3; the winner becomes the
rule.

1. **Best window** — the file scores as its best-matching window.
2. **Window-count-normalised** — the best window's score corrected for the
   number of windows it was the best of, so a long file is not ranked up for
   drawing more chances.
3. **Identity windows** — the file scores on the windows of its first turn's
   request (the line naming its path) and its last turn's closing summary
   only. A conversation's first and last turns are likely its most relevant:
   the request says what it is about, the summary what it found.

Turn layouts record each segment's span, and signatures are 1:1 with tokens,
so the identity windows are selected from what is stored.

### 2.4 Analysis units

An analysis unit — `dependency_analysis`, `code_review`, `security_review`,
each `after: code_reading` — forks from the file's reading, so its own
conversation inherits the whole file. Its gallery is its **own** turns — the
question asked of it, its tool rounds, and its answer — cut by §2.1 and §2.2,
never the inherited reading. A `security_review` unit's web searches are tool
rounds like any other and are cut the same way.

## 3. Gates

Each gate must hold before the next is run. All are `#[test]`s.

### 3.1 Boundary parity

`boundaries` over a recorded decoded turn's tokens equals the boundaries its
decode fired live — including a turn with a tool call, a run of line feeds, a
line under the content gate and a span past the cadence.

### 3.2 Capture coverage

For every file the test conversation needs: each turn's token count against
its signature-token count, and which segments are covered (request, call,
response, summary). A turn whose signatures fall short is re-ingested alone.

### 3.3 Self-retrieval

Over the whole `code_reading` group: a window of file X, used as a probe, ranks
X first. Reported per §2.3 candidate as recall at 1, 5 and 10 over many files
and windows, and as the correlation between a file's size and its rank, which
must be near zero. A candidate that fails this cannot pass §3.4.

### 3.4 The per-projection harness

The paper conversation (`docs/paper_conversation_replay.md`) is broken into
every projection it builds: each user turn, each boundary of each thinking
block, each tool call and each tool response. For each, the expected
injections are written down — which files, folders and analysis units the
working set and each layer should hold, and which they must not. **The user
writes the expectations.** The harness compares each projection's injections
with them and reports hits, misses and noise per projection.

## 4. The harness

The harness runs **offline, without the full engine, on the GPU**.

- **Recorded once:** the conversation's turns — their tokens and folded
  signatures — and a snapshot of the substrate's galleries.
- **Replayed:** for each projection, the probe is rebuilt from the recorded
  signatures by the §2.1 formula and scored against the snapshot through the
  same paged gallery arena and kernels the daemon uses
  (`provenance::gallery_arena`, the IMMA/BMMA/scalar backends). No model is
  loaded and nothing is decoded.
- **Compared:** the working-set momentum, the per-layer selections and their
  injected tokens, against the written expectations.

So a change to a boundary rule, a window shape, an aggregation or a formula is
judged in seconds, deterministically, on any machine with a CUDA card; a live
replay of the conversation remains the end-to-end check after the gate passes.

## 5. What changes, and what does not

- **Changes:** the gallery's windows (seams become §2.1/§2.2 windows); the
  per-file score (§2.3); the decode trigger, which calls the shared
  `boundaries` rule rather than holding its own.
- **Unchanged:** what ingest stores. Signatures stay per token, which is what
  lets windows be cut at assembly. A turn's capture is re-done only where §3.2
  finds it short.
- **The full re-ingest** waits until the gates pass; until then only the files
  the gates name are re-ingested.

## 6. Code map

| Piece | Where |
|---|---|
| The boundary rule | `candle-conversation/src/provenance/boundaries.rs` (new), called by `scheduler/decode.rs` |
| Windows from a stored turn | `candle-conversation/src/projection/belief_files.rs`, replacing `subwindow_bounds` over seams |
| Per-file aggregation | `candle-conversation/src/projection/belief_files.rs` / the scan wrappers |
| Probe construction | `scheduler/mod.rs` (the reproject probe) — the shape the windows mirror |
| Gates 3.1–3.3 | tests beside the rule and the scan |
| The harness | a recorded fixture of the paper conversation and a GPU `#[test]` over `gallery_arena` |

## 7. Open

- Whether boilerplate — the request template, tool-call JSON, line-number
  gutters — needs masking once windows are the probe's shape, or whether
  window-shaped comparison and identity windows already remove its effect.
  Measured at §3.3.
- The expected injections, per projection (§3.4).
