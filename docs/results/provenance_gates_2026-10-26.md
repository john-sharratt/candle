# Provenance gates — 2026-10-26

The gates of `docs/provenance_scoped_signatures.md` §3, run by
`candle-conversation/examples/provenance_gates` on BlackWorld (RTX PRO 5000,
Ryzen 9 9950X3D). Every scan runs on the live daemon's resident gallery
(`POST /v1/substrate/zend/scan`); the harness weighs, gates and folds the
votes. The gallery grows between runs (2,085 → 2,206 files, ~25.8 M signature
tokens); compare rows within a table, not across tables.

The probe set throughout: each calibrated file's question 0 (1,501–1,661
questions), asked of every `code_reading` file. It is in no background and no
gallery.

## Headline

What the day's calibration work bought `code_reading`'s file retrieval, each
row measured on the live gallery against held-out calibration questions
(R@1 / R@10 / MRR). Rows of one table share a gallery; the gallery grew
between tables, so the two tables are compared row to row within themselves.

**Bare-question probes** (the question's words alone; 15 questions a file):

| scoring | R@1 | R@10 | MRR |
|---|---|---|---|
| calibration questions pooled with content (the scorer at the day's start) | — | — | 0.152 |
| content alone | 0.125 | 0.264 | 0.172 |
| calibration as a competition of its own, `+ 0.5 ×` | — | 0.365 | 0.221 |

**Live-turn probes** (a held-out question and the ≤64 tokens of thinking it
starts — the probe a live projection makes; 24 questions a file, 999 files):

| scoring | R@1 | R@10 | MRR |
|---|---|---|---|
| content alone, every fold group | 0.270 | 0.457 | 0.337 |
| `+ 0.5 ×` calibration competition, every fold group | 0.197 | 0.561 | 0.323 |
| content groups 1–2 `+ 0.25 ×` calibration top group (**shipped**) | **0.310** | 0.554 | **0.397** |
| the same through the working set's momentum, three turns on a file | **0.392** | **0.708** | **0.504** |

Single-scan R@1 rises 0.270 → 0.310 over content alone while R@10 rises 0.457
→ 0.554; over a short conversation, under the same momentum, R@1 nearly
doubles against the competition weighed alike (0.213 → 0.392 at the same
three turns, `momentum`). The findings that carry beyond this gallery:

1. **A calibration question and a live probe share a distribution**, so in
   one pool the questions take nearly every vote (96.5% of a background's
   vote mass at 2% of the gallery); scored as a competition of their own and
   added at a fraction, they lift recall instead of drowning content.
2. **Probes find their own kind.** Question probes match stored questions,
   thinking probes stored thinking; storing both and probing with the live
   turn (both) is best, and summing two half-probes is worse than one whole.
3. **Identity lives in different fold groups for content and for
   calibration.** Fold group 0 (layers 0–45) carries none of a file's
   identity; content sits in groups 1–2, calibration in the top group alone.
4. **Momentum compounds the scoring gain rather than overlapping it**, at the
   cost of slower topic switches (β 0.2: a new file leads in 70% of
   dialogues after three turns).
5. **Repairing signatures does not work; cutting chunks barely matters.**
   Bit-level centring is a no-op; the anomalous-chunk cut moves MRR by
   ±0.005 once calibration competes apart.

## Anomalous-chunk exclusion (`chunk-factor`)

Each vote is credited to the 32-token gallery chunk it was won at
(`attribution.rs`); a chunk's background mass is what it wins on other files'
calibration questions. The most generic chunks are the request template, the
tool-call JSON and its closing tags, and the pager line — and file text worded
like a question.

| rule | chunks cut | R@1 | R@10 | MRR |
|---|---|---|---|---|
| none | 0 | 0.135 | 0.278 | 0.184 |
| smooth factor `(1 + w/w0)^-α`, `1/(1 + ln(1 + w/w0))` | — | 0.077–0.126 | 0.253–0.300 | 0.123–0.179 |
| top 0.02% by mass | 150 | 0.153 | 0.305 | 0.205 |
| distinct-file frequency ≥ 40 | 786 | 0.150 | 0.300 | 0.199 |
| log-mass outlier, k = 5, each file keeping half | 131 | 0.147 | 0.299 | 0.200 |
| token runs in ≥ 2–50% of files (no scan) | 605–1,863 k tok | 0.123–0.136 | 0.264–0.274 | 0.173–0.181 |

- **Cut from the scan, never down-weighted.** A cut chunk's votes go to the
  probe token's next-best agreement; a smaller factor only shrinks them.
- **From the questions, never from the gallery alone.** Verbatim shared runs —
  a statistic of the gallery — lose at every share, as every gallery-only
  normalizer has (`docs/provenance_score_normalization.md` §2).
- Run-to-run noise is about ±0.01: the gallery grows while it is measured.

## Bit-level centring — closed

Agreement with the background questions' per-bit majority taken out
(`popcount((q⊕m) & (g⊕m))`). The 150 most generic chunks sit 570.8 bits of
1,536 from the majority, the rest 593.5: they are not distinguished by
closeness to the centroid, and a per-bit mask cannot touch the joint
common-mode (`docs/tool_selection_provenance_results.md`, bitmask whitening a
no-op). Not built.

## Question density (`density`)

Each file's question 0 against galleries holding every file's questions
`1..=k` in the files' own pool.

| questions per file | turn vs turn | body vs body |
|---|---|---|
| 0 | 0.172 | 0.155 |
| 1 | 0.084 | 0.076 |
| 4 | 0.119 | 0.102 |
| 8 | 0.132 | 0.107 |
| 15 | 0.153 | 0.124 |

More questions compound, but the first halves MRR and fifteen never recover
what none gives. Cutting the shared chat framing from probe and gallery
changes nothing: it is not the scaffold.

## Why: one pool (`question-cut`)

With questions `1..=8` in the gallery and `9..` as background, **96.5% of the
background's vote mass is won at question chunks, 2% of the gallery's
chunks.** Question exemplar and question probe share a distribution, and `z`
is taken over a background that is mostly content, so a question's match to
any other question outvotes its file's content. The outlier exclusion cannot
separate them — question chunks are uniformly loud (MRR 0.132 → 0.136 at most).

## The fix: a competition of their own (`competitions`)

Each file's calibration questions scored alone, every calibrated file a case,
its own `z` and margin; the file's score is its content's plus `w ×` its
calibration's.

| questions per file | one pool | questions alone | content + 0.25× | content + 0.5× | content + 1× |
|---|---|---|---|---|---|
| 4 | 0.118 | 0.107 | 0.191 / 0.317 | 0.189 / 0.338 | 0.179 / 0.339 |
| 8 | 0.132 | 0.130 | 0.205 / 0.339 | 0.203 / 0.352 | 0.193 / 0.355 |
| 15 | 0.152 | 0.150 | 0.218 / 0.352 | **0.221 / 0.365** | 0.209 / 0.371 |

MRR / R@10 for the combined rows; content alone is 0.172 / 0.264. Normalizing
each competition to its probe's best file before combining does worse. At 15
questions a file and `w = 0.5`, MRR rises 0.152 → 0.221 over today's pool and
0.172 → 0.221 over content alone, R@10 0.264 → 0.365, and it still climbs
with density.

**What ships**: `scan_file_scans` scores the calibration competition as one
more segment of the same paged launch (and the same competition on the CPU
path); `code_reading`'s `scopes` declares `calibration_weight: 0.5`. Gate 4,
the paper conversation, waits for a replay.

## The cut as production learns it (`calibration-cut`)

The calibration competition's chunk cut learned as production learns it —
over every calibrated file, from questions held out of both the learning and
the gallery; 1,637 files asked, 443 of them calibrated with thinking at the
time.

| scoring | R@1 | R@10 | MRR |
|---|---|---|---|
| content alone | 0.155 | 0.301 | 0.205 |
| `+ 0.5 ×` competition, no cut | 0.163 | 0.413 | 0.251 |
| `+ 0.5 ×` competition, k 5 (3 chunks cut) | 0.163 | 0.412 | 0.251 |
| `+ 0.5 ×` competition, k 4 (51 chunks cut) | 0.163 | 0.412 | 0.251 |

The competition is worth +0.046 MRR and +0.112 R@10 here; the cut on top of it
is neutral. Production learns the cut anyway (content k 5, calibration k 3):
on the live gallery it cuts ~85 of 806 k content chunks and ~450 of 57 k
calibration chunks — and was re-measured neutral on the thinking-calibrated
set below (k 3–5: ≤ +0.004 MRR).

## Thinking and the live turn (`thinking`)

Calibrated at 24 questions a file, each with up to 64 tokens of thinking; 979
files carried thinking on all of questions `0..=8`. Question 0 held out, asked
three ways: its words, its thinking, and both as one turn — the probe a live
projection makes once the turn has thought.

| probe | content alone | content + 0.5 × competition (question + thinking) |
|---|---|---|
| question | 0.143 / 0.294 / 0.191 | 0.132 / 0.377 / 0.214 |
| thinking | 0.201 / 0.361 / 0.255 | 0.205 / 0.472 / 0.296 |
| turn | 0.271 / 0.454 / 0.336 | 0.189 / 0.555 / 0.316 |

R@1 / R@10 / MRR. A probe finds its own kind: question probes score on the
competition's questions (its thinking alone gives 0.015), thinking probes on
its thinking (its questions alone give 0.004). Asking the two halves apart and
summing their scores is worse than asking the turn whole at every weight. At
`w = 0.5` the competition buys the turn ten points of R@10 and costs it eight
of R@1.

The calibration weight on the turn probe, every fold group alike (979 files):

| w | one turn R@1 / R@10 / MRR | halves summed R@1 / R@10 / MRR |
|---|---|---|
| 0 (content) | 0.271 / 0.454 / 0.336 | 0.254 / 0.438 / 0.319 |
| 0.1 | 0.263 / 0.496 / **0.353** | 0.250 / 0.484 / 0.338 |
| 0.25 | 0.223 / 0.540 / 0.340 | 0.217 / 0.520 / 0.328 |
| 0.35 | 0.201 / 0.551 / 0.327 | 0.199 / 0.533 / 0.320 |
| 0.5 | 0.189 / **0.555** / 0.316 | 0.183 / 0.535 / 0.307 |
| 0.75 | 0.170 / 0.541 / 0.298 | 0.163 / 0.529 / 0.288 |

Weighed alike, the competition trades top hits for coverage at every weight —
the trade the fold groups below remove.

## Fold groups (`layers`)

Content and competition weighed by fold group apart, re-summed from one scan
each; 999 files, turn probes.

| content | competition | R@1 | R@10 | MRR |
|---|---|---|---|---|
| [1,1,1] | — | 0.270 | 0.457 | 0.337 |
| [0,1,1] | — | 0.283 | 0.483 | 0.354 |
| [1,0,0] | — | 0.000 | 0.003 | 0.003 |
| [1,1,1] | 0.5 × [1,1,1] (production) | 0.197 | 0.561 | 0.323 |
| [0,1,1] | 0.25 × [0,1,1] | 0.280 | 0.560 | 0.382 |
| [0,1,1] | 0.25 × [0,0,1] | **0.310** | 0.554 | **0.397** |
| [0,1,1] | 0.5 × [0,0,1] | 0.280 | **0.575** | 0.389 |

Fold group 0 carries nothing of a file's identity and dilutes the rest; the
calibration competition's identity sits in the top group alone. Question and
thinking probes order the settings the same way (question 0.214 → 0.240,
thinking 0.296 → 0.347 at their best).

## Momentum (`momentum`)

The working set's `m ← (1 − β)·m + fresh` over a dialogue asking three
held-out questions of one file, each turn scanned at its opening (question)
and once thought (the turn); 1,019 files. MRR after each scan, R@1 / R@10
after the last.

| scoring | β | q0 open | q0 thought | q1 thought | q2 thought | R@1 | R@10 |
|---|---|---|---|---|---|---|---|
| production | 1 (none) | 0.234 | 0.334 | 0.285 | 0.289 | 0.186 | 0.482 |
| production | 0.2 | 0.234 | 0.329 | 0.363 | 0.375 | 0.213 | 0.692 |
| [0,1,1] + 0.25 × [0,0,1] | 1 (none) | 0.252 | 0.405 | 0.336 | 0.335 | 0.258 | 0.473 |
| [0,1,1] + 0.25 × [0,0,1] | 0.2 | 0.252 | 0.398 | 0.474 | **0.504** | **0.392** | **0.708** |

Momentum compounds the fold-group gain instead of overlapping it: R@1 0.186
→ 0.392 over three turns. Its cost is the switch to another file — after
three turns on A, three on B leave B ahead of A in 70% of dialogues at β 0.2,
87% at β 0.5, 50% at β 0 — and B's MRR reaches 0.475 by its sixth scan at
β 0.2. Production's β 0.2 stands.

## Calibrating as the ingest commits

Each unit is calibrated by the ingest worker that read it, between its commit
and its eviction to cold (`calibrate::calibrate_committed`), so the draft and
its 24 questions fork onto the reading while it is still resident. Over the
`code_reading` re-ingest on BlackWorld, 603 files were calibrated against 604
committed — one a commit — at ~4–4.5 files a minute, each file's reading,
24 questions and their thinking together; 2 drafts wrote no question and fell
to the catch-up round. Decode is the bound, not the lift from cold the
catch-up round paid.

## Costs found in production

- **The exclusion learner froze the engine.** Learning scanned all ~9,800
  sampled questions in two scheduler jobs over the whole gallery, ~40 s
  each; every live conversation stalled ~3.5 minutes after each restart and
  ahead of each pass. It now scans 16 questions a job, between waves, and
  relearns only when its questions move by a tenth.
- **A per-probe harness gate starves production.** The `questions` gate
  (≈3,300 probes, each two serial scans of the whole gallery) against the
  live daemon held decode near 2 tokens/s for half an hour. Gates against a
  serving daemon batch their probes.

**What ships**: a group policy's `calibration_layer_weights` weighs the
calibration competition's fold groups apart from the content's
(`layer_weights`), per segment in the one launch's tally. `code_reading`,
`dependency_analysis`, `code_review` and `security_review` take
`layer_weights: [0, 1, 1]`, `calibration_layer_weights: [0, 0, 1]` and
`calibration_weight: 0.25` — the analysis layers carrying code_reading's
measurement until their own units are measured.

Gate 4 did not run: the paper conversation's dialogue turns (`e7bca43c…`)
hold their tokens but no signatures, so it has no probe. Weighing the top
group alone of *content* lost that conversation a quarter of its tool calls
(2026-10-25); content here keeps groups 1 and 2, and only the calibration
competition is the top group's. The replay is owed on a fresh paper
conversation.
