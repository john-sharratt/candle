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
