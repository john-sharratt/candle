# Provenance signatures scoped like projection

**Status:** §2.1 built — `provenance::boundaries` is the one rule, and the
decode loop steps it. §2.2's windows (`provenance::probe_windows`) and §2.3's
aggregations (`provenance::window_score`) are built and measured offline by
gates 2–3 (`candle-conversation/examples/provenance_gates`); gate 4's harness
and expectations are written and wait for the replay. The live scan has not
switched yet. Supersedes the per-turn gallery windows the belief scan uses today
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

**What the gates found** (`docs/results/provenance_gates_2026-10-25.md`):

- Scoring a window by its own per-token z-scores fails outright: it has no
  competition between files, so boilerplate every file matches swamps it.
- What works keeps late fusion's competition and adds coherence: each probe
  token's `z × margin` vote — won by one file, its margin over the runner-up
  file — is credited to the windows around the gallery token it was won at
  (`provenance::window_score::late_fusion_votes`, `window_tally`), and a file
  scores by its windows. Of the three candidates, the better of a file's
  identity windows and its count-normalised best window, each standardised
  (`FileAggregate::IdentityOrCount`), is the best all-round rule so far.
- Re-scoring a shortlist of today's best files by window loses the gain: the
  runner-up a margin is taken against has to be the whole gallery's.

**On the GPU**, with the arena's kernels unchanged. The paged scan emits, per
probe token, fold group and segment, the winning case and its vote. Two
launches over the same resident gallery give the vote and where it was won:

1. today's — one case per file — the winning file and its `z × margin` vote;
2. one segment per file, one case per 32-token chunk — each file's best chunk.

The first launch's vote is credited to the second's best chunk in the winning
file; a window's tally is the votes credited to the chunks it covers. That is
the CPU tally at chunk resolution, for about twice the scan.

**What ships** is the summed vote, with every file one case of one
competition (`ScanCases::File`), every fold group voting alike: the window
tallies helped request and summary probes but not analysis units' answers,
which decide. A file-scored group's warm-up teaches its pool scope nothing —
a file's self-match would teach it only that file's ceiling — so its levels
are live traffic's.

Two corrections that won on the analysis-unit probes lost on the paper
conversation (gate 4), and neither ships:

- **Hub correction** — each file over its *background peak*, the most it
  scored on probes of other files, floored at 100. MRR over the analysis-unit
  probes rose 0.40–0.44 → 0.45–0.50, but the paper conversation's first
  message found 38% of the files it should (91% without), and 9 of its 14
  tool calls' files in the top ten (14 without): the files a real
  conversation asks about are often the hubs.
- **Top fold group alone** (`[0, 0, 1]`): 75% and 11 of 14 on the same turn.

Both are kept as rules in the harness (`provenance_gates/rules.rs`) to be
measured against again.

### 2.4 Analysis units

An analysis unit — `dependency_analysis`, `code_review`, `security_review`,
each `after: code_reading` — forks from the file's reading, so its own
conversation inherits the whole file. Its gallery is its **own** turns — the
question asked of it, its tool rounds, and its answer — cut by §2.1 and §2.2,
never the inherited reading. A `security_review` unit's web searches are tool
rounds like any other and are cut the same way.

### 2.5 Calibration questions

A unit's own windows are its **content**: the request that read it, the tool
calls, the file, its summary. A live probe is a **question**. The tool catalog
measured the gap: its exemplars were ~95% think block and call, and when plain
questions joined them as exemplars Top-1 rose 74.5% → 87.9%
(`docs/tool_selection_provenance_results.md` §26.3). A unit is given the same.

- **Declared per layer.** A layer opts in with a `calibrate:` block — how many
  questions, the decode budget of each question's thinking (`think`), and the
  prompt that asks for them (`<<n>>`, `<<subject>>`, `<<file>>`, `<<repo>>`,
  `<<path>>`). Nothing names a layer; `code_reading` and the three analyses of
  a file (`dependency_analysis`, `code_review`, `security_review`, each asked
  about what its own analysis answers) declare 24 questions and 64 thinking
  tokens — one projection's worth: a turn's first
  reprojection fires by the 64-token cadence, so the probe that decides what
  is pulled in has seen at most that much of the model's thinking.
- **Written by the unit.** Each unit is asked in a draft forked from its
  layer's base with the unit as its lineage — so it writes its questions with
  its whole content in front of it — without reasoning or tools. The draft is
  tombstoned once its answer is read; one question per line is kept.
- **Thought about as a live turn is.** A live probe is the trailing window of
  what the model decodes, and for most of a turn that is its own thinking,
  which no question exemplar resembles. Each question is asked again alone —
  a conversation of the hidden calibration layer, the system prompt every
  conversation opens with, nothing projected, every dial at the dialogue's
  default — and the thinking the model starts, up to `think` tokens, is kept.
  All of a unit's questions are asked at once and decode in one wave.
- **Calibrated as tools' questions are.** All of a unit's questions go in as
  one turn group in a conversation of the hidden calibration layer, under the
  system prompt every conversation opens with, prefilled lossless
  (`kv_lossless`): each masked to itself, sealed one turn apiece, its answer
  its thinking as a closed reasoning block, tagged `calibration`. The answer
  is assistant content, not a reasoning segment, so the seal keeps its
  signatures: a turn's own reasoning is not retrievable, but an exemplar is
  never retrieved — it votes for its unit. The
  conversation names the unit (`Substrate::CALIBRATES_KEY`), is archived —
  the commit — and is distilled `ProvenanceOnly`: K/V dropped, signatures
  kept.
- **Scored in a competition of their own.** A calibration's exemplars and a
  live probe share a distribution: in one pool with the units' content they
  took 96.5% of a background's vote mass at 2% of the gallery, and a question
  found its unit less often than with no questions at all
  (`docs/results/provenance_gates_2026-10-26.md`). So every calibrated unit is
  a case of one more competition, scored on its calibration windows alone
  with its own `z` and margin, in the same launch
  (`belief_files::scan_file_scans`); the unit's vote there, weighed by its
  group's `calibration_weight`, is added to its head exchange's score — under
  file-scored cases, the unit's. `code_reading` weighs it 0.5. A calibration
  is never itself a candidate, a working-set member, or a dialogue probe.
- **Resumed like the tools' corpus.** Ahead of each ingest pass, a layer's
  ready units — finished without a chain break, every turn holding content
  here — without an archived calibration carrying their current marker
  (layer, digest of what is asked, the unit's content key) are calibrated. A
  unit present but not ready keeps the calibration its marker names and waits:
  a member that holds another member's reading hollow must not retire a
  calibration every member shares. A present unit's finished calibration
  under an earlier marker — a changed prompt, count or budget — stands until
  one carrying its current marker is archived, and the pass after retires it,
  so the hours a layer takes to calibrate again never leave a unit with none.
  Every other calibration of the layer — of no present unit, never finished,
  or a second of the same marker — is retired. Units read before the block existed are calibrated
  without being read again, and a unit read again is calibrated again.

### 2.6 Anomalous-chunk exclusion

Some gallery chunks win votes on every file's questions: the request template,
the tool-call JSON and its closing tags, a pager's footer, file text worded
like a question. Each vote one of them wins is a vote the right file does not
get. They are cut from the live scan, so each such vote goes to the probe
token's next-best agreement; down-weighting them only shrinks the vote and
loses (`docs/results/provenance_gates_2026-10-26.md`).

- **Learned from the questions, never from the gallery alone.** A layer's
  calibration questions — `EXCLUSION_QUESTIONS_PER_FILE` (6) of each file —
  are scanned raw against the group's content alone: no calibration in the
  gallery, nothing cut, so the exclusion never learns from a gallery it already
  shaped. Each kept vote for a file not the question's own is credited to the
  32-token chunk holding the token that won it: the best agreement in the
  vote's fold group among that file's own tokens. A chunk's mass is what it
  wins on other files' questions. Verbatim shared runs, a gallery statistic,
  lost at every threshold measured.
- **Cut at the outliers.** A chunk is cut when its log mass is more than five
  robust deviations (`1.4826 × MAD`) above the median of the chunks that won
  anything, each file keeping at least half its chunks — on the measured
  gallery, ~140 of ~800,000 chunks.
- **Both competitions.** The calibration competition (§2.5) has generic
  chunks of its own — the thinking every question opens with, the questions
  near-identical files ask. The same questions are scanned against the
  calibrations alone, every file's own turns masked, and its outliers are
  cut at three robust deviations: its chunks are few per file and alike, so
  its outliers sit nearer the median. Measured over the first files
  calibrated with thinking: 17 of ~8,000 chunks cut, MRR 0.333 → 0.401 on
  question probes and 0.424 → 0.463 on thinking probes.
- **Where.** Learned ahead of each layer's calibration pass
  (`GroupScanner::learn_chunk_exclusion`, the scan at ingest priority, the
  attribution on the calling thread holding no lock), held in memory per group
  (`Substrate::chunk_exclusion`), and cut from the group's own windows where
  the live scan assembles them (`belief_files::cut_excluded`), calibration
  windows and its own alike.
- **Measured.** Held-out questions, content alone: MRR 0.172 → 0.186; stacked
  on the calibration competition at 15 questions a file: 0.220 → 0.228, R@10
  0.364 → 0.379.

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
| The harness | `candle-conversation/examples/provenance_gates` — the sweep, gate 4 and the questions gate on the GPU gallery (`GalleryArena::scan_votes`) |
| Calibration questions (§2.5) | `zend/src/calibrate/` (spec, plan, questions, run); `Substrate::{CALIBRATES_KEY, calibrations, calibrations_of}`; the merge in `belief_files::assemble_file_scans` |

## 7. Open

- The expected injections, per projection (§3.4).
