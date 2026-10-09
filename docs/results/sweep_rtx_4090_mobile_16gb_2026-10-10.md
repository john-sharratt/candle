# Sweep — RTX 4090 Laptop 16 GB, 2026-10-10 (origin/main `21c36efbd`)

One sweep of the thirteen width gates, one `cargo` process per model, card to itself, on
`21c36efbd`: `origin/main` after it merged this card's expert-cache work (`a08f3ee35`:
blended read-ahead prediction, eviction behind the wave by expected miss cost) with the
72 GB card's recovery work (`41481e374`, `21c36efbd`: decode slot uploads in one scatter,
the read-ahead link-time gate, `blend::BREAK_EVEN`, skipped hops for begun rows). Core
Ultra 9 185H, 31.5 GiB RAM, PCIe 4.0 ×16. `deepseek4` is skipped on this card (a partial
sweep). The engine probes were not run. Every row is in
`performance_rtx_4090_mobile_16gb_rows_2026-10-10.tsv`.

The reference is the previous full sweep on this card,
`sweep_rtx_4090_mobile_16gb_2026-10-06.md` (tree `fa4fa870d`), and the recorded rows of
2026-09-30 (`performance_rtx_4090_mobile_16gb_rows.tsv`).

## Result

**Thirteen of thirteen gates pass.** `EXIT=0` and `test result: ok. 1 passed` on each, every
config of every gate at 100% of its sessions, no `✗` line, no `non-finite` and no `!!!!` in
any log. These are single runs, so a gap under about 5% is noise.

**Qwen3.8-27B ran twice.** Its first run read BF16 ×1 decode 8.1 and C8 ×20 25.9 t/s, which
looked like thermal throttling, so the sweep was stopped, the card left to cool to a flat
46 °C, and the sweep restarted from the 27B. The second run read 8.2 and 28.9 with the SM
clock at 1,877 MHz mean and 70 °C peak, so these two rows are not heat: they are the same two
slow rows the recorded 2026-09-30 ladder has (7.0 and 28.6). Every 27B figure below is the
second run; the 30B and later gates ran after the restart.

## Gates

Best prefill and best decode over each ladder, and the best compression of a validated row.

| # | Model | best prefill t/s | best decode t/s | best compression | time |
|---|---|---:|---:|---:|---:|
| 1 | Qwen2-0.5B | 86,038.3 (BF16 ×60) | 5,123.9 (BF16 ×60) | — | 175 s ¹ |
| 2 | Qwen3.5-0.8B | 52,030.7 (Q8_0 ×4) | 3,810.7 (C8 ×32) | 4.38× (C10) | 178 s |
| 3 | Llama-3.2-3B | 15,194.5 (Q8_1 ×4) | 688.7 (C8 ×10) | 4.43× (C10) | 57 s |
| 4 | Llama-2-7B | 6,573.9 (F16 ×8) | 904.6 (BF16 ×48) | 3.56× (Q4_0) ² | 75 s |
| 5 | Qwen3-8B | 6,561.7 (BF16 ×4) | 308.4 (C8 ×10) | 5.82× (C10) | 68 s |
| 6 | Qwen3.5-9B | 7,601.7 (Q8_0 ×4) | 873.0 (C8 ×20) | 5.57× (C10) | 43 s |
| 7 | Qwen3.8-27B | 2,318.9 (BF16 ×4) | 219.7 (BF16 ×4) | 5.03× (C10) | 70 s |
| 8 | Qwen3-30B-A3B | 6,460.6 (C5 ×8) | 310.0 (Q4_0 ×20) | 5.45× (C10) | 145 s |
| 9 | Qwen3.5-35B-A3B | 3,307.3 (C8 ×5) | 395.2 (C10 ×8) | 7.03× (C10) | 82 s |
| 10 | Qwen3.6-35B-A3B | 2,649.6 (C8 ×5) | 336.7 (C10 ×16) | 6.43× (C10) | 143 s |
| 11 | Qwen3.6-35B hybrid, Precision | 2,949.4 (C8 ×5) | 384.3 (C5 ×8) | 6.47× (C10) | 90 s |
| 12 | Qwen3.6-35B hybrid, Performance | 3,886.7 (C8 ×5) | 539.2 (C10 ×8) | 6.45× (C10) | 78 s |
| 13 | Qwen3.8-Flash-Next (Q2_KO experts) | 1,128.7 (C10 ×8) | 108.5 (BF16 ×8) | 5.82× (C10) | 215 s |

¹ Includes the release build. ² From a row the gate does not validate for reproduction
(`-`); its sessions passed the gate's non-empty and distinct-output checks.

## Against the 2026-10-06 sweep

The widest C10 row of each ladder (Llama-2: BF16 ×48; Qwen2: F16 ×60), prefill / decode t/s.
Peak tokens are identical to the 2026-10-06 run on every row here, and so is compression.

| Model | row | prefill 10-06 → now | Δ | decode 10-06 → now | Δ |
|---|---|---|---:|---|---:|
| Qwen2-0.5B | F16 ×60 | 64,587.8 → 85,233.7 | +32.0% | 5,125.0 → 4,983.6 | −2.8% |
| Qwen3.5-0.8B | C10 ×10 | 26,275.7 → 50,135.7 | +90.8% | 2,058.1 → 2,017.5 | −2.0% |
| Llama-3.2-3B | C10 ×5 | 9,179.8 → 14,052.5 | +53.1% | 499.0 → 504.4 | +1.1% |
| Llama-2-7B | BF16 ×48 | 4,645.0 → 6,322.2 | +36.1% | 1,007.7 → 904.6 | **−10.2%** |
| Qwen3-8B | C10 ×5 | 4,048.4 → 6,393.0 | +57.9% | 234.5 → 228.0 | −2.8% |
| Qwen3.5-9B | C10 ×10 | 3,780.7 → 7,274.2 | +92.4% | 729.5 → 792.8 | +8.7% |
| Qwen3.8-27B | C10 ×10 | 1,179.9 → 2,225.6 | +88.6% | 77.6 → 79.7 | +2.7% |
| Qwen3-30B-A3B | C10 ×2 | 2,472.3 → 3,062.5 | +23.9% | 116.8 → 130.0 | +11.3% |
| Qwen3.5-35B-A3B | C10 ×8 | 2,143.1 → 2,728.8 | +27.3% | 199.5 → 395.2 | +98.1% |
| Qwen3.6-35B-A3B | C10 ×8 | 1,946.0 → 2,131.5 | +9.5% | 128.4 → 252.1 | +96.3% |
| Qwen3.6 hybrid, Precision | C10 ×8 | 2,014.7 → 2,460.0 | +22.1% | 189.5 → 313.4 | +65.4% |
| Qwen3.6 hybrid, Performance | C10 ×8 | 2,410.1 → 3,182.1 | +32.0% | 208.4 → 539.2 | +158.7% |
| Qwen3.8-Flash-Next | C10 ×8 | 987.5 → 1,128.7 | +14.3% | 51.8 → 101.3 | +95.6% |

- **Prefill rose on every gate**, by 9–92%; the dense models' gain is the larger.
- **Decode on the Qwen3.5/3.6 and Flash-Next gates rose 65–159%** at their widest C10 row,
  and Qwen3-30B's by 11%. The single-session rows rose further: Qwen3.5-35B C0 ×1
  52.2 → 156.1, the Precision hybrid C0 ×1 48.3 → 160.5, the Performance hybrid C1 ×1
  60.7 → 192.9.
- **Dense decode held**, within ±3% on five of seven dense gates at their widest row.
- **Llama-2-7B BF16 ×48 decode fell 10.2%** (1,007.7 → 904.6), while its prefill on the same
  row rose 36.1% and its Q4_0 ×32 decode rose 3.6% (475.0 → 492.0). This is one run; it is
  recorded as open, not attributed.

## Qwen3.8-Flash-Next across the pull

Before the sweep, the merged tree (`21c36efbd`) and this branch's pre-merge commit
(`611901d1d`) were built as two test binaries and run alternately on the gate — merged,
pre, pre, merged — so that neither always ran first or hot. Decode t/s, per run:

| row | pre-merge | merged |
|---|---|---|
| BF16 ×1 (cold) | 21.7 / 15.6 | 9.2 ³ / 28.9 |
| BF16 ×4 | 71.0 / 69.0 | 35.1 ³ / 86.3 |
| BF16 ×8 | 84.2 / 92.8 | 81.6 / 103.6 |
| BF16 ×1 (warm) | 25.3 / 23.5 | 29.9 / 32.1 |
| C0 ×2 | 36.8 / 36.8 | 53.4 / 51.2 |
| C5 ×2 | 36.1 / 35.6 | 53.9 / 50.2 |
| C5 ×8 | 69.5 / 73.2 | 91.5 / 98.4 |
| C8 ×2 | 18.4 / 30.8 | 51.0 / 50.5 |
| C10 ×2 | 28.1 / 26.1 | 42.2 / 44.0 |
| C10 ×8 | 60.4 / 65.2 | 64.7 / 73.8 |

³ The first run after the build, against a model artifact not yet in the page cache; the same
rows are the day's best in the merged binary's second run.

The merged tree is ahead on every row once warm: the ×2 rows by 42–59% (C8 ×2 by more,
against one low pre-merge run), C5 ×8 by 33%, BF16 ×8 by 5% and C10 ×8 by 10%. The sweep's own Flash-Next run, on a cooler machine,
is the best ladder this card has recorded for the model:

| row | 2026-10-09, pre-merge | this sweep | Δ decode |
|---|---|---|---:|
| BF16 ×1 (cold) | 123.4 / 26.1 | 145.4 / 33.4 | +28.0% |
| BF16 ×4 | 613.6 / 81.0 | 793.4 / 89.5 | +10.5% |
| BF16 ×8 | 911.2 / 106.7 | 1,127.5 / 108.5 | +1.7% |
| BF16 ×1 (warm) | 212.3 / 32.8 | 269.1 / 34.9 | +6.4% |
| C0 ×2 | 380.8 / 53.3 | 463.3 / 61.6 | +15.6% |
| C5 ×2 | 379.0 / 53.8 | 499.2 / 60.7 | +12.8% |
| C5 ×8 | 899.0 / 109.5 | 1,127.6 / 107.9 | −1.5% |
| C8 ×2 | 374.8 / 49.3 | 483.8 / 56.3 | +14.2% |
| C10 ×2 | 372.9 / 38.0 | 499.1 / 51.0 | +34.2% |
| C10 ×8 | 890.1 / 83.0 | 1,128.7 / 101.3 | +22.0% |

Prefill / decode t/s. Prefill rose 18–34% on every row.

## Against the recorded rows (2026-09-30)

At each streamed-expert MoE gate's widest C10 row, decode is now 1.7–10.6× the recorded
figure: Qwen3-30B C10 ×2 12.3 → 130.0, Qwen3.5-35B C10 ×16 131.2 → 330.8, the Precision
hybrid C10 ×8 71.5 → 313.4, the Performance hybrid C10 ×8 77.6 → 539.2, Flash-Next C10 ×8
59.5 → 101.3. The recorded decode rows predate the gate's decode-clock fix (`b705689b2`) and
read low, so not all of that gap is the engine; prefill is the cleaner comparison, and it is
up on every gate (Qwen3.5-0.8B C8 ×32 21,847.0 → 46,056.8; Qwen3-30B Q8_0 ×20 3,999.7 →
6,139.1).

## Open

- **Llama-2-7B BF16 ×48 decode −10.2%** against 2026-10-06, single run; unattributed.
- **Qwen3.8-27B's two slow rows** (BF16 ×1 8.2, C8 ×20 28.9) stand as recorded since
  2026-09-30; the other ×1 rows run at 82–83.
- **The engine probes were not run** in this sweep. The last laptop probe figures are the
  2026-10-06 sweep's.
