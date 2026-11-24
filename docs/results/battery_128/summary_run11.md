# Battery-128 analysis

7257 answers, 3033 failed after every try.

## Figure 9.12-1 — recall by arm

| Arm | Questions | Mean recall | 95% interval | Run means | Full marks |
|---|---:|---:|---|---|---:|
| A0 | 128 | 0.4% | 0.1–0.9% | 0.5, 0.4, 0.3 | 0 |
| A1 | 128 | 0.5% | 0.1–1.0% | 0.7, 0.4, 0.4 | 0 |
| D1 | 128 | 30.6% | 25.0–36.4% | 30.8, 31.3, 29.8 | 9 |
| B0 | 128 | 31.9% | 26.1–37.7% | 31.7, 30.4, 33.6 | 13 |
| B1 | 128 | 32.2% | 26.8–37.6% | 33.4, 32.6, 31.5, 32.1, 31.8, 31.7 | 3 |
| C0 | 128 | 36.7% | 30.8–42.6% | 36.8, 37.7, 35.7 | 11 |
| C1 | 128 | 37.6% | 31.8–43.3% | 36.5, 39.3, 35.6, 38.9, 37.0, 38.2 | 6 |
| R1 | 128 | 52.7% | 46.4–59.0% | 54.3, 51.6, 52.2 | 31 |
| T1 | 128 | 95.3% | 93.3–97.1% | 95.0, 95.4, 95.7 | 99 |

## Figure 9.12-2 — effect sizes (the pre-registered family)

| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |
|---|---|---:|---:|---|---:|---:|
| The memory | C1 − A1 | 128 | 37.1 | 31.4 to 42.7 | 8.30e-18 | 4.15e-17 |
| Reasoning-time reprojection | C1 − B1 | 128 | 5.4 | 3.2 to 7.6 | 3.17e-6 | 1.27e-5 |
| Continuous against retrieve-once | C1 − D1 | 128 | 7.0 | 3.7 to 10.3 | 6.01e-5 | 1.80e-4 |
| Reasoning-time reprojection without the seed | C0 − B0 | 128 | 4.9 | 1.7 to 8.2 | 3.46e-3 | 6.92e-3 |
| The seed files | C1 − C0 | 128 | 0.9 | -1.8 to 3.5 | 4.67e-1 | 4.67e-1 |

## Figure 9.12-2b — the memory against the baselines (secondary family)

Holm-corrected across this family alone; the pre-registered family above is corrected without it.

| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |
|---|---|---:|---:|---|---:|---:|
| Provenance memory against retrieval-augmented generation | C1 − R1 | 128 | -15.1 | -23.0 to -7.1 | 3.87e-4 | 3.87e-4 |
| Provenance memory against tool use | C1 − T1 | 128 | -57.7 | -64.0 to -51.4 | 4.31e-21 | 1.29e-20 |
| Retrieval-augmented generation against the seed files | R1 − A1 | 128 | 52.2 | 45.8 to 58.5 | 6.76e-19 | 1.35e-18 |

## R1's retrieval — an answer file among the files it was handed

| Arm | Questions | Answers | Answer file retrieved (hit@k) | Top file an answer file (hit@1) | Files per prompt | Files cut at the budget |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 128 | 384 | 53.9% | 28.9% | 3.2 | 31.1% |

## Table 9.12 — grounding and cost by arm

| Arm | Answers | Citing a name | Mean grounding | Mean completion tokens | Tool calls read | Mean tool calls |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 384 | 372 | 99.6% | 480.4 | 0 | — |
| A1 | 384 | 375 | 99.4% | 473.1 | 0 | — |
| D1 | 384 | 384 | 95.5% | 1402.2 | 0 | — |
| B0 | 384 | 383 | 94.7% | 1397.2 | 0 | — |
| B1 | 768 | 766 | 95.2% | 1459.2 | 384 | 0.0 |
| C0 | 384 | 384 | 94.4% | 1368.7 | 0 | — |
| C1 | 768 | 766 | 95.7% | 1420.8 | 384 | 0.0 |
| R1 | 384 | 383 | 99.6% | 1204.7 | 384 | 0.0 |
| T1 | 384 | 384 | 97.6% | 2032.8 | 384 | 12.3 |

T1 called 12.3 tools an answer on average, over the 384 answers whose calls were read back.

## Figure 9.12-3 — question by question

62 questions above the diagonal (C1 > B1), 19 below.

## Figure 9.12-4 — where the gain falls (C1 − B1)

| Question names its file | Questions | Gain (points) | 95% interval |
|---|---:|---:|---|
| true | 17 | 1.6 | -3.4 to 6.5 |
| false | 111 | 6.0 | 3.7 to 8.4 |

## Figure 9.12-6 — recall when the answer file is placed

| Answer file placed for the first answer token | Answers | Recall | 95% interval |
|---|---:|---:|---|
| true | 1460 | 51.4% | 49.5–53.2% |
| false | 1228 | 13.6% | 12.0–15.2% |

## §6.5 — how the reasoning steers retrieval

| Arm | Answers | Placed at opening | At reasoning's end | For first answer token | Reprojections in thinking | Brought in, of those the opening missed |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 384 | 48.7% | 48.7% | 48.7% | 0.0 | 0.0% |
| B0 | 384 | 50.0% | 52.3% | 52.3% | 0.6 | 8.9% |
| B1 | 768 | 51.6% | 52.9% | 52.9% | 0.6 | 7.3% |
| C0 | 384 | 51.0% | 56.2% | 56.2% | 17.7 | 22.9% |
| C1 | 768 | 51.3% | 58.6% | 58.6% | 18.1 | 25.7% |

15945 telemetry samples in `fig_9_6_telemetry.csv` (Figures 9.6-1 to 9.6-3).

## §9.6.9 — the card under load

BF16 bytes per token: 24916.1.

| Under the battery's load | Mean | Peak |
|---|---:|---:|
| Sessions in flight | 12.62 | 16.00 |
| KV arenas held (GiB) | 5.82 | 11.17 |
| Tokens resident | 482123.58 | 945212.00 |
| The same tokens at BF16 (GiB) | 11.19 | 21.93 |
| Effective compression (BF16 / arenas held) | 1.92 | 2.80 |
| Expert zone (GiB) | 34.85 | 60.45 |

| In flight | Samples | Tokens resident | Compression | Q1_S | Q2_S | Q3_0 | Q3_1 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_KS | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–4 | 126 | 121033 | 1.50× | 19.6% | 4.8% | 11.6% | 0.0% | 23.5% | 7.9% | 2.9% | 0.0% | 12.8% | 0.0% | 16.9% |
| 5–8 | 6403 | 560358 | 2.15× | 14.0% | 0.0% | 13.3% | 0.0% | 22.2% | 8.4% | 10.5% | 0.0% | 11.4% | 0.0% | 20.3% |
| 9–12 | 105 | 177011 | 1.51× | 18.7% | 4.8% | 11.9% | 0.2% | 23.4% | 7.7% | 3.8% | 0.0% | 12.6% | 0.0% | 16.9% |
| 13–16 | 9282 | 436508 | 1.77× | 15.2% | 0.0% | 13.0% | 0.0% | 22.4% | 8.2% | 9.5% | 0.0% | 11.8% | 0.0% | 20.0% |
