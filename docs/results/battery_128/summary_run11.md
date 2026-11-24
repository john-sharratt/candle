# Battery-128 analysis

7513 answers, 3033 failed after every try.

## Figure 9.12-1 — recall by arm

| Arm | Questions | Mean recall | 95% interval | Run means | Full marks |
|---|---:|---:|---|---|---:|
| A0 | 128 | 0.4% | 0.1–0.9% | 0.5, 0.4, 0.3 | 0 |
| A1 | 128 | 0.5% | 0.1–1.0% | 0.7, 0.4, 0.4 | 0 |
| D1 | 128 | 30.6% | 25.0–36.4% | 30.8, 31.3, 29.8 | 9 |
| B0 | 128 | 31.9% | 26.1–37.7% | 31.7, 30.4, 33.6 | 13 |
| B1 | 128 | 32.2% | 26.8–37.6% | 33.4, 32.6, 31.5, 32.1, 31.8, 31.7 | 3 |
| C0 | 128 | 36.7% | 30.8–42.6% | 36.8, 37.7, 35.7 | 11 |
| C1 | 128 | 37.5% | 31.7–43.1% | 36.5, 39.3, 35.6, 38.9, 37.0, 38.2, 36.9 | 6 |
| R1 | 128 | 52.6% | 46.1–58.9% | 54.3, 51.6, 52.2, 52.3 | 31 |
| T1 | 128 | 95.3% | 93.3–97.1% | 95.0, 95.4, 95.7 | 99 |

## Figure 9.12-2 — effect sizes (the pre-registered family)

| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |
|---|---|---:|---:|---|---:|---:|
| The memory | C1 − A1 | 128 | 37.0 | 31.3 to 42.6 | 3.88e-18 | 1.94e-17 |
| Reasoning-time reprojection | C1 − B1 | 128 | 5.3 | 3.2 to 7.4 | 7.05e-7 | 2.82e-6 |
| Continuous against retrieve-once | C1 − D1 | 128 | 6.9 | 3.6 to 10.1 | 5.79e-5 | 1.74e-4 |
| Reasoning-time reprojection without the seed | C0 − B0 | 128 | 4.9 | 1.7 to 8.2 | 3.46e-3 | 6.92e-3 |
| The seed files | C1 − C0 | 128 | 0.7 | -1.7 to 3.3 | 5.39e-1 | 5.39e-1 |

## Figure 9.12-2b — the memory against the baselines (secondary family)

Holm-corrected across this family alone; the pre-registered family above is corrected without it.

| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |
|---|---|---:|---:|---|---:|---:|
| Provenance memory against retrieval-augmented generation | C1 − R1 | 128 | -15.1 | -23.0 to -7.1 | 4.33e-4 | 4.33e-4 |
| Provenance memory against tool use | C1 − T1 | 128 | -57.8 | -64.1 to -51.5 | 9.96e-22 | 2.99e-21 |
| Retrieval-augmented generation against the seed files | R1 − A1 | 128 | 52.1 | 45.6 to 58.4 | 6.69e-19 | 1.34e-18 |

## R1's retrieval — an answer file among the files it was handed

| Arm | Questions | Answers | Answer file retrieved (hit@k) | Top file an answer file (hit@1) | Files per prompt | Files cut at the budget |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 128 | 512 | 53.9% | 28.9% | 3.2 | 31.1% |

## Table 9.12 — grounding and cost by arm

| Arm | Answers | Citing a name | Mean grounding | Mean completion tokens | Tool calls read | Mean tool calls |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 384 | 372 | 99.6% | 480.4 | 0 | — |
| A1 | 384 | 375 | 99.4% | 473.1 | 0 | — |
| D1 | 384 | 384 | 95.5% | 1402.2 | 0 | — |
| B0 | 384 | 383 | 94.7% | 1397.2 | 0 | — |
| B1 | 768 | 766 | 95.2% | 1459.2 | 384 | 0.0 |
| C0 | 384 | 384 | 94.4% | 1368.7 | 0 | — |
| C1 | 896 | 894 | 95.8% | 1431.0 | 512 | 0.0 |
| R1 | 512 | 511 | 99.6% | 1208.9 | 512 | 0.0 |
| T1 | 384 | 384 | 97.6% | 2032.8 | 384 | 12.3 |

T1 called 12.3 tools an answer on average, over the 384 answers whose calls were read back.

## Table 9.12 — an expected item in view as the answer began

Each expected item of each answer, looked for in the text the model had at its answer's first token: in an answer file's text (the injected turns of its reading, or R1's block of it), or anywhere it was given.

| Arm | Answers | Items | In an answer file's text | Named, when there | Named, when not | Anywhere in view | Named, when anywhere | Answer-file turns placed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 128 | 445 | 48.5% | 69.9% | 14.0% | 60.4% | 66.2% | 48.3% |
| R1 | 128 | 445 | 38.2% | 97.6% | 22.5% | 56.0% | 90.0% | —% |

## Figure 9.12-3 — question by question

71 questions above the diagonal (C1 > B1), 29 below.

## Figure 9.12-4 — where the gain falls (C1 − B1)

| Question names its file | Questions | Gain (points) | 95% interval |
|---|---:|---:|---|
| true | 17 | 2.6 | -2.3 to 7.2 |
| false | 111 | 5.7 | 3.5 to 8.0 |

## Figure 9.12-6 — recall when the answer file is placed

| Answer file placed for the first answer token | Answers | Recall | 95% interval |
|---|---:|---:|---|
| true | 1534 | 51.4% | 49.6–53.2% |
| false | 1282 | 13.7% | 12.1–15.3% |

## §6.5 — how the reasoning steers retrieval

| Arm | Answers | Placed at opening | At reasoning's end | For first answer token | Reprojections in thinking | Brought in, of those the opening missed |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 384 | 48.7% | 48.7% | 48.7% | 0.0 | 0.0% |
| B0 | 384 | 50.0% | 52.3% | 52.3% | 0.6 | 8.9% |
| B1 | 768 | 51.6% | 52.9% | 52.9% | 0.6 | 7.3% |
| C0 | 384 | 51.0% | 56.2% | 56.2% | 17.7 | 22.9% |
| C1 | 896 | 51.5% | 58.5% | 58.5% | 18.3 | 26.0% |

17645 telemetry samples in `fig_9_6_telemetry.csv` (Figures 9.6-1 to 9.6-3).

## §9.6.9 — the card under load

BF16 bytes per token: 24916.1.

| Under the battery's load | Mean | Peak |
|---|---:|---:|
| Sessions in flight | 12.91 | 16.00 |
| KV arenas held (GiB) | 5.71 | 11.17 |
| Tokens resident | 469362.01 | 945212.00 |
| The same tokens at BF16 (GiB) | 10.89 | 21.93 |
| Effective compression (BF16 / arenas held) | 1.88 | 2.80 |
| Expert zone (GiB) | 35.12 | 60.45 |

| In flight | Samples | Tokens resident | Compression | F16 | Q1_S | Q2_S | Q3_0 | Q3_1 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_KS | R16 | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–4 | 135 | 129004 | 1.47× | 0.0% | 19.0% | 4.2% | 11.8% | 0.0% | 23.3% | 7.9% | 3.8% | 0.0% | 12.6% | 0.0% | 0.0% | 17.4% |
| 5–8 | 6412 | 560304 | 2.15× | 0.0% | 14.0% | 0.0% | 13.3% | 0.0% | 22.2% | 8.4% | 10.5% | 0.0% | 11.4% | 0.0% | 0.0% | 20.3% |
| 9–12 | 126 | 221410 | 1.57× | 0.0% | 17.1% | 3.2% | 12.4% | 0.2% | 23.0% | 7.9% | 6.1% | 0.0% | 12.2% | 0.0% | 0.0% | 18.0% |
| 13–16 | 10849 | 422728 | 1.73× | 0.0% | 15.1% | 0.0% | 13.0% | 0.0% | 22.3% | 8.2% | 9.6% | 0.0% | 11.8% | 0.0% | 0.0% | 20.0% |
