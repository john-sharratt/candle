# Battery-128 analysis

48 answers, 0 failed after every try.

## Figure 9.12-1 — recall by arm

| Arm | Questions | Mean recall | 95% interval | Run means | Full marks |
|---|---:|---:|---|---|---:|
| C1 | 24 | 84.7% | 72.9–95.1% | 86.8, 82.6 | 18 |

## Figure 9.12-2 — effect sizes (the pre-registered family)

| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |
|---|---|---:|---:|---|---:|---:|
| The memory | C1 − A1 | 0 | — | — to — | 1.00e0 | 1.00e0 |
| Reasoning-time reprojection | C1 − B1 | 0 | — | — to — | 1.00e0 | 1.00e0 |
| Continuous against retrieve-once | C1 − D1 | 0 | — | — to — | 1.00e0 | 1.00e0 |
| Reasoning-time reprojection without the seed | C0 − B0 | 0 | — | — to — | 1.00e0 | 1.00e0 |
| The seed files | C1 − C0 | 0 | — | — to — | 1.00e0 | 1.00e0 |

## Figure 9.12-2b — the memory against the baselines (secondary family)

Holm-corrected across this family alone; the pre-registered family above is corrected without it.

No R1 or T1 answers, so no contrast to test.

## The files handed as text — an answer file among them (R1 by BM25, P1 by provenance)

No R1 or P1 answers.

## Table 9.12 — grounding and cost by arm

| Arm | Answers | Citing a name | Mean grounding | Mean completion tokens | Tool calls read | Mean tool calls |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 48 | 41 | 99.8% | 288.5 | 48 | 0.0 |

## Table 9.12 — an expected item in view as the answer began

Each expected item of each answer, looked for in the text the model had at its answer's first token: in an answer file's text (the injected turns of its reading, or R1's block of it), or anywhere it was given.

| Arm | Answers | Items | In an answer file's text | Named, when there | Named, when not | Anywhere in view | Named, when anywhere | Answer-file turns placed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 48 | 72 | 0.0% | —% | 81.9% | 16.7% | 100.0% | —% |

## Figure 9.12-3 — question by question

0 questions above the diagonal (C1 > B1), 0 below.

## Figure 9.12-4 — where the gain falls (C1 − B1)

| Question names its file | Questions | Gain (points) | 95% interval |
|---|---:|---:|---|
| true | 0 | — | — to — |
| false | 0 | — | — to — |

## Figure 9.12-6 — recall when the answer file is placed

| Answer file placed for the first answer token | Answers | Recall | 95% interval |
|---|---:|---:|---|
| true | 0 | —% | —–—% |
| false | 48 | 84.7% | 76.0–92.7% |

## §6.5 — how the reasoning steers retrieval

| Arm | Answers | Placed at opening | At reasoning's end | For first answer token | Reprojections in thinking | Brought in, of those the opening missed |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0 | —% | —% | —% | NaN | —% |
| B0 | 0 | —% | —% | —% | NaN | —% |
| B1 | 0 | —% | —% | —% | NaN | —% |
| C0 | 0 | —% | —% | —% | NaN | —% |
| C1 | 48 | 0.0% | 0.0% | 0.0% | 8.1 | 0.0% |

21 telemetry samples in `fig_9_6_telemetry.csv` (Figures 9.6-1 to 9.6-3).

## §9.6.9 — the card under load

BF16 bytes per token: 25379.3.

| Under the battery's load | Mean | Peak |
|---|---:|---:|
| Sessions in flight | 14.40 | 16.00 |
| KV arenas held (GiB) | 1.30 | 1.72 |
| Tokens resident | 75343.85 | 85102.00 |
| The same tokens at BF16 (GiB) | 1.78 | 2.01 |
| Effective compression (BF16 / arenas held) | 1.26 | 1.61 |
| Expert zone (GiB) | 56.44 | 60.70 |

| In flight | Samples | Tokens resident | Compression | Q2_A | Q3_0 | Q4_0 | Q4_1 | Q5_0 | Q8_0 | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–4 | 1 | 83342 | 1.56× | 9.6% | 15.5% | 13.1% | 8.1% | 22.1% | 12.8% | 18.8% |
| 5–8 | 1 | 84407 | 1.20× | 9.5% | 15.4% | 13.0% | 8.0% | 21.9% | 13.0% | 19.2% |
| 9–12 | 1 | 84387 | 1.26× | 9.5% | 15.4% | 13.0% | 8.0% | 21.9% | 13.0% | 19.3% |
| 13–16 | 17 | 73808 | 1.24× | 9.4% | 15.4% | 13.0% | 8.0% | 21.9% | 13.0% | 19.3% |
