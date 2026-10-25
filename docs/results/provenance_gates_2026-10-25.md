# Provenance gates — 2026-10-25

The gates of `docs/provenance_scoped_signatures.md` §3.2–§3.3, run offline by
`candle-conversation/examples/provenance_gates.rs` on BlackWorld (RTX PRO 5000,
Ryzen 9 9950X3D), CPU only, against the live zend substrate **read-only** while
it re-ingests from a wipe. Every figure names the gallery it was measured on;
the gallery grows between runs.

## Gate 2 — capture coverage

Every `code_reading` turn's signature tokens against its `Tokens` record.

| run | files | turns | exact | short | long | no signature |
|---|---|---|---|---|---|---|
| 1 | 77 | 156 | 156 | 0 | 0 | 0 |
| 5 | 200 | 434 | 434 | 0 | 0 | 0 |

One run (128 files) caught a turn with 658 signatures and no `Tokens` record
yet — a turn mid-write in the daemon, read by the read-only open between its
records. **Capture is complete**: signatures are 1:1 with tokens, prefilled
pages included, so every turn can be windowed with no re-ingest.

## Gate 3 — retrieval

Probes are cut by the shared boundary rule (`provenance::boundaries`) to the
live probe's shape (`provenance::probe_windows`): the last window of a turn.

- **self (control)** — a window of a file's own pages, nothing left out. Every
  rule must rank the file first.
- **summary-to-rest** — the file's last turn's last window (its closing
  summary), that turn left out of the gallery.
- **request-to-rest** — the file's first turn's last window (the request and
  its tool call), that turn left out.

Ranked among every `code_reading` file. Size bias is Spearman's ρ between a
file's signature tokens and how often it is a false top-10 hit (target: ≈ 0).

### What failed: coherent z-scoring

A window scored by, for each probe token, its best agreement inside the window
as a z over the whole gallery — reduced by mean, top quarter, or excess over a
threshold, with or without standardising each probe token across windows —
reached R@10 ≤ 0.10 where late fusion reached 0.55–0.74 (runs 1–4, 97–160
files). The self control passed at 1.00, so the scoring is not broken: it
lacks late fusion's **competition**. Late fusion gives each probe token to one
file, weighted by its margin over the runner-up file, so a boilerplate token
that matches every file equally — the request template, tool-call JSON —
votes for nothing. A per-window z cannot see that.

### What works: late fusion's votes, tallied per window

The same per-token `z × margin` votes and needle gate as today
(`window_score::late_fusion_votes`, equal to `score_slots_weighted` per file —
unit-tested), each credited to the windows holding the gallery token it was
won at (`window_tally`). A file then scores by its windows
(`file_scores`): its best window, that less `σ·√(2 ln W)` for its `W` windows,
or its best identity window (first turn's opening, last turn's closing).

Run 5 — 200 files, 1,683,868 tokens, 71,313 windows, 37 probes a source:

| source | rule | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| request-to-rest | late fusion (today) | 0.05 | 0.32 | 0.62 | 0.209 |
| request-to-rest | window votes / best-window | 0.05 | 0.32 | 0.59 | 0.201 |
| request-to-rest | window votes / count-normalised | 0.05 | 0.32 | 0.59 | 0.203 |
| request-to-rest | window votes / identity | **0.14** | **0.43** | 0.54 | **0.273** |
| summary-to-rest | late fusion (today) | 0.08 | 0.35 | 0.46 | 0.217 |
| summary-to-rest | window votes / best-window | 0.14 | 0.35 | 0.46 | 0.245 |
| summary-to-rest | window votes / count-normalised | **0.14** | **0.43** | **0.68** | **0.299** |
| summary-to-rest | window votes / identity | 0.00 | 0.00 | 0.08 | 0.023 |
| self (control) | every rule | 1.00 | 1.00 | 1.00 | 1.000 |

| rule | size bias ρ |
|---|---|
| late fusion (today) | +0.212 |
| window votes / best-window | +0.184 |
| window votes / count-normalised | −0.266 |
| window votes / identity | −0.057 |

Reading it:

- Tallying votes per window loses nothing against today and gains where the
  probe's partner is a coherent region (summary → request: MRR 0.22 → 0.30).
- Count normalisation over-corrects: long files are now penalised (ρ −0.27).
- Identity windows are nearest unbiased, and best for a request-shaped probe,
  but read nothing when the left-out turn held the file's only closing window —
  summary-to-rest's 0.00 is that artefact, not the rule.

### Run 6 — the combined rule

302 files, 2,221,127 tokens, 92,227 windows, 57 probes a source; coverage
644/644 exact. Two rules added: count normalisation at half strength, and
**identity-or-count** — the better of a file's identity score and its
count-normalised best window, each standardised over the files.

| source | rule | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| request-to-rest | late fusion (today) | 0.04 | 0.30 | 0.49 | 0.178 |
| request-to-rest | best-window | 0.04 | 0.30 | 0.49 | 0.169 |
| request-to-rest | count-normalised ×1 | 0.04 | 0.30 | 0.51 | 0.171 |
| request-to-rest | identity | 0.12 | 0.35 | 0.47 | 0.218 |
| request-to-rest | identity-or-count ×1 | 0.07 | 0.32 | 0.51 | 0.199 |
| summary-to-rest | late fusion (today) | 0.07 | 0.30 | 0.44 | 0.184 |
| summary-to-rest | best-window | 0.11 | 0.32 | 0.44 | 0.206 |
| summary-to-rest | count-normalised ×1 | 0.11 | 0.35 | 0.54 | 0.244 |
| summary-to-rest | identity | 0.00 | 0.02 | 0.05 | 0.019 |
| summary-to-rest | identity-or-count ×1 | 0.11 | 0.35 | 0.53 | 0.228 |
| self (control) | every rule | 1.00 | 1.00 | 1.00 | 1.000 |

| rule | size bias ρ |
|---|---|
| late fusion (today) | +0.152 |
| best-window | +0.120 |
| count-normalised ×1 | −0.171 |
| count-normalised ×0.5 | −0.117 |
| identity | +0.041 |
| identity-or-count ×1 | −0.115 |
| identity-or-count ×0.5 | −0.085 |

**identity-or-count ×1 is the best all-round rule so far**: above today on
both sources (MRR 0.18 → 0.20 and 0.18 → 0.23, R@10 0.49 → 0.51 and
0.44 → 0.53), with the size bias halved and of the opposite sign. The margins
are modest and the probes few; the analysis-unit source decides.

### Run 7 — two stages do not carry the gain

401 files, 2,782,797 tokens, 114,271 windows, 60 probes a source; coverage
853/853 exact. Added: **today's top 32, re-scored** — today's per-file scan
picks 32 files and only they compete again by window (identity-or-count ×1),
the production shape that would leave the GPU scan unchanged.

| source | rule | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| request-to-rest | late fusion (today) | 0.08 | 0.47 | 0.62 | 0.241 |
| request-to-rest | today's top 32, re-scored | 0.08 | 0.43 | 0.60 | 0.231 |
| request-to-rest | identity | 0.17 | 0.53 | 0.58 | 0.320 |
| request-to-rest | identity-or-count ×1 | 0.15 | 0.48 | 0.65 | 0.292 |
| summary-to-rest | late fusion (today) | 0.05 | 0.43 | 0.60 | 0.209 |
| summary-to-rest | today's top 32, re-scored | 0.05 | 0.35 | 0.67 | 0.200 |
| summary-to-rest | count-normalised ×1 | 0.05 | 0.50 | 0.67 | 0.237 |
| summary-to-rest | identity-or-count ×1 | 0.05 | 0.47 | 0.63 | 0.211 |

| rule | size bias ρ |
|---|---|
| late fusion (today) | +0.337 |
| best-window | +0.304 |
| count-normalised ×1 | +0.079 |
| identity-or-count ×1 | +0.107 |
| today's top 32, re-scored | +0.129 |

- **Re-scoring a shortlist loses the gain** (MRR at or below today): the
  votes are margins over the runner-up file, and with the field cut to 32 the
  runner-up is a different file than over the whole gallery. Window tallies
  have to come from the full scan — the GPU arena must return each vote's
  winning token, not only its file.
- Over the whole gallery identity-or-count holds its lead on request probes
  (MRR 0.24 → 0.29) and ties on summary probes; count normalisation leads on
  summary probes. Both cut today's size bias by two thirds or more.
- The CPU scan took 1,842 s for 180 probes at 2.8 M gallery tokens — ~10 s a
  probe: offline only.

### Sweeps — the scorer's settings and hubness

`provenance_gates sweep` scores each probe once (`group_votes`) and weighs,
gates and aggregates every setting on the host: per-fold-group vote weights,
the needle gate's keep fraction, whether the gallery holds each file's request
turn, and the fold from votes to file scores. The source that decides is the
**analysis units**: a `dependency_analysis` unit's answer windows, each finding
the file it analyses (5.2 M gallery tokens, 192 probes).

| setting | MRR |
|---|---|
| today: weights [1, 1, 1], keep 0.25, summed | 0.436 |
| best weights and keep, summed | 0.456 |
| [0.5, 1, 2], keep 0.25, over-peak (floor 100) | 0.469 |
| [0, 0, 1], keep 0.5, over-peak (floor 100) | **0.502** |

- **Window tallies do not carry over to analysis units.** Count-normalised
  and identity-or-count, the best rules on request and summary probes, sit at
  or below the summed votes here.
- **Hubness is what costs the most.** The same few files —
  `regen_summaries.rs`, `battle-cities/CLAUDE.md`, `coding_assistant.md`,
  `batch_norm.rs` — win part of every probe's votes whatever it asks. Scored
  over each file's **background** (what it wins on probes of other files, its
  own left out) they lose that lead: the share less the background mean helps,
  and the raw score over the file's background *peak*, both at a 256-token
  probe and floored at 100, helps most on these probes. It does not survive
  the paper conversation (below), and does not ship.

The sweeps run on the GPU gallery (`GalleryArena::scan_votes`, the raw
per-token votes): 62 s a sweep, against 16–33 minutes scanned on the host.

## Calibration questions

Each `code_reading` file wrote 16 questions its content answers
(`zend::calibrate`, design §2.5); all 1,005 files calibrated, 16,038
questions. **The questions gate** holds each file's questions out in turn and
asks them of the gallery: each question should find its own file.

| rule | MRR |
|---|---|
| today: [1, 1, 1], summed | 0.125 |
| today + questions | 0.180 |
| top group [0, 0, 1] + questions | 0.187 |
| over-peak (floor 100) + questions | 0.173 |
| top group, over-peak, + questions | 0.182 |

With its file's other questions in the gallery, a question finds its file
nearly half again as often (0.125 → 0.180). The other rules move it less than
a hundredth either way.

## Gate 4 — the paper conversation

`paper e7bca43ccfa75cbd`, checked against
`docs/provenance_expectations/paper_conversation.yaml`. Its first message asks
about the paper. The table gives the share of the files it should find that
are found, and how many of its 14 tool calls' files are in the top ten:

| rule | should-hit | tool calls in top 10 |
|---|---|---|
| today | 88% | 14 / 14 |
| today + questions | **91%** | **14 / 14** |
| top group + questions | 75% | 11 / 14 |
| over-peak + questions | 38% | 9 / 14 |
| top group, over-peak, + questions | 50% | 9 / 14 |

The hub correction, which won on the analysis units' answers, loses a real
conversation more than half the files it should find: the files it asks
about are often the hubs, and the correction marks them down. The top group
alone loses less, but still loses: 75% and 11 of 14. **What ships** is
today's scorer — every fold group alike, no background peak — with the
calibration questions in the gallery. `code_reading`'s `scopes` keeps its
cold-scope floor (`level_prior: { floor_base: 2, floor_cap: 16 }`); the
analysis groups have none. No file-scored group is warmed by self-match;
their levels are live traffic's.

## Open

- The analysis-unit probes and the paper conversation disagree on what helps.
  A rule is measured against both before it ships.
- Next rule to measure: identity and best-window combined (the better of a
  file's identity score and its count-normalised best window, each
  standardised), and a milder count correction.
