# Battery-128 analysis

2688 answers, 0 failed after every try.

## Figure 9.12-1 — recall by arm

| Arm | Questions | Mean recall | 95% interval | Run means | Full marks |
|---|---:|---:|---|---|---:|
| A0 | 128 | 0.4% | 0.1–0.9% | 0.5, 0.4, 0.3 | 0 |
| A1 | 128 | 0.5% | 0.1–1.0% | 0.7, 0.4, 0.4 | 0 |
| D1 | 128 | 30.6% | 25.0–36.4% | 30.8, 31.3, 29.8 | 9 |
| B0 | 128 | 31.9% | 26.1–37.7% | 31.7, 30.4, 33.6 | 13 |
| B1 | 128 | 32.5% | 26.8–38.3% | 33.4, 32.6, 31.5 | 10 |
| C0 | 128 | 36.7% | 30.8–42.6% | 36.8, 37.7, 35.7 | 11 |
| C1 | 128 | 37.2% | 31.0–43.2% | 36.5, 39.3, 35.6 | 12 |

## Figure 9.12-2 — effect sizes

| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |
|---|---|---:|---:|---|---:|---:|
| The memory | C1 − A1 | 128 | 36.7 | 30.7 to 42.7 | 3.66e-16 | 1.83e-15 |
| Reasoning-time reprojection | C1 − B1 | 128 | 4.6 | 1.2 to 8.2 | 2.35e-2 | 4.70e-2 |
| Continuous against retrieve-once | C1 − D1 | 128 | 6.5 | 2.8 to 10.3 | 3.73e-4 | 1.49e-3 |
| Reasoning-time reprojection without the seed | C0 − B0 | 128 | 4.9 | 1.7 to 8.2 | 3.46e-3 | 1.04e-2 |
| The seed files | C1 − C0 | 128 | 0.4 | -2.7 to 3.5 | 6.70e-1 | 6.70e-1 |

## Figure 9.12-3 — question by question

51 questions above the diagonal (C1 > B1), 29 below.

## Figure 9.12-4 — where the gain falls (C1 − B1)

| Question names its file | Questions | Gain (points) | 95% interval |
|---|---:|---:|---|
| true | 17 | -3.1 | -11.2 to 4.1 |
| false | 111 | 5.8 | 2.2 to 9.7 |

## Figure 9.12-6 — recall when the answer file is placed

| Answer file placed for the first answer token | Answers | Recall | 95% interval |
|---|---:|---:|---|
| true | 1025 | 51.1% | 49.0–53.3% |
| false | 895 | 13.9% | 12.0–15.8% |

## §6.5 — how the reasoning steers retrieval

| Arm | Answers | Placed at opening | At reasoning's end | For first answer token | Reprojections in thinking | Brought in, of those the opening missed |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 384 | 48.7% | 48.7% | 48.7% | 0.0 | 0.0% |
| B0 | 384 | 50.0% | 52.3% | 52.3% | 0.6 | 8.9% |
| B1 | 384 | 51.6% | 51.8% | 51.8% | 0.6 | 6.5% |
| C0 | 384 | 51.0% | 56.2% | 56.2% | 17.7 | 22.9% |
| C1 | 384 | 51.6% | 57.8% | 57.8% | 17.1 | 25.8% |

6787 telemetry samples in `fig_9_6_telemetry.csv` (Figures 9.6-1 to 9.6-3).

## §9.6.9 — the card under load

BF16 bytes per token: 24916.1.

| Under the battery's load | Mean | Peak |
|---|---:|---:|
| Sessions in flight | 8.50 | 16.00 |
| KV arenas held (GiB) | 6.19 | 11.17 |
| Tokens resident | 570908.85 | 945212.00 |
| The same tokens at BF16 (GiB) | 13.25 | 21.93 |
| Effective compression (BF16 / arenas held) | 2.16 | 2.70 |
| Expert zone (GiB) | 36.04 | 50.14 |

| In flight | Samples | Tokens resident | Compression | Q1_S | Q3_0 | Q4_0 | Q4_1 | Q5_0 | Q8_0 | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–4 | 3 | 448487 | 1.71× | 14.7% | 13.2% | 22.6% | 8.5% | 9.9% | 11.3% | 19.8% |
| 5–8 | 6358 | 562084 | 2.16× | 14.0% | 13.3% | 22.2% | 8.4% | 10.5% | 11.4% | 20.3% |
| 13–16 | 425 | 703786 | 2.20× | 14.0% | 13.4% | 22.3% | 8.4% | 10.5% | 11.3% | 20.2% |
