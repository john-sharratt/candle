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

## Open

- The analysis-unit source (a `dependency_analysis` / `code_review` /
  `security_review` unit's own answer finding its file) has no probes yet: no
  analysis layer has run on the fresh substrate. It is the gate's real test.
- 37 probes a source is small; rerun as the gallery grows.
- Next rule to measure: identity and best-window combined (the better of a
  file's identity score and its count-normalised best window, each
  standardised), and a milder count correction.
