# Sweep — RTX 4090 Laptop 16 GB, 2026-10-06 (origin/main with the CUDA graph subsystem), against 2026-10-05

One full sweep — thirteen width gates and both engine probes, one `cargo` process per
model, card to itself, `deepseek4` skipped on this card (a partial sweep) — on tree
`fa4fa870d` (origin/main on 2026-10-06, which already carries the KV claim fix
`6d97c3460`). Core Ultra 9 185H, 31.5 GiB RAM, PCIe 4.0 ×16. Compared with the previous
sweep on this card, `6d97c3460` plus the `86c91146a` merge
(`sweep_rtx_4090_mobile_16gb_2026-10-05_kv_claim.md`), and with the recorded rows
(`performance_rtx_4090_mobile_16gb_rows.tsv`, 2026-09-30).

## Result

**Sweep: FAIL, on one gate.** Thirteen of thirteen width gates passed (`EXIT=0`, `1 passed`,
every session of every config validated at 100%). Probe 14 (`qwen3_30b_a3b_q4`) failed its
VRAM-efficiency gate only — story 20/20, weight zone at its limit — and panicked at
`probe.rs:251`; the user parked that failure earlier. Probe 15 (`qwen38_flash_next`) passed.
The previous sweep ended the same way: 13/13 gates, probe 14 failing efficiency (71%),
probe 15 passing.

Single runs, so a gap under ~5% is noise on this laptop. Rows that moved more than 10% are in
**bold** below.

Two things to read before the tables:

- **Gates 10 and 13 are not like-for-like.** Their `Peak Tokens` column changed between the
  runs (Qwen3.6-35B BF16 ×1 659 → 905; Flash-Next BF16 ×1 713 → 905), so those rows ran
  longer sequences. Every other gate's `Peak Tokens` is identical to the previous run. See
  Caveats.
- **Decode moved the most, and not in one direction.** Dense and small-MoE decode rose by
  4–90%; the ×4-wide BF16 decode rows and the single-stream decode rows of the Qwen3.5/3.6
  MoE gates fell by 12–43%. Prefill rose 3–7% on most gates, fell 25–36% across gate 12, and
  rose 2–4× at the widest C10 rungs of the MoE gates.

## What changed

origin/main brought, between the two sweeps (inferred from `docs/decode_graphs.md` and
`docs/results/baseline_rtx_3090_24gb_2026-10-05_decode_graphs.md`, not from a diff; none of
the deltas below is attributed to a specific change):

- **A CUDA graph subsystem** (`candle-core/src/cuda_backend/graph/`): a wave's forward is
  recorded as a chain of graphs with no host sync and replayed or updated in place, for
  models that mark where recording begins. It acts on decode and prefill launch overhead, so
  it could plausibly touch every gate's decode column; `decode_graphs.md` §3 lists recording
  for Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, and (measured on the 3090) Qwen2-0.5B,
  Llama-3.2-3B, Llama-2-7B, Qwen3-8B and Qwen3-30B-A3B.
- **A zero-allocation forward pass** and a **PLE fused kernel**: per-forward allocation and
  per-layer kernel count, so decode rows at small batch.
- **A `wave_plan` rework**: how a wave is composed, so the wide (×8, ×16) prefill rungs and
  the mixed decode/prefill waves.
- **QSA stratified selection** (`qsa_topk.cu`): Flash-Next's attention selection (gate 13,
  probe 15).
- **`expert_lre` copier/started changes** and a **kernel dispatcher rework**: expert
  streaming and kernel selection, so the streamed-expert models (gates 8–13, both probes).
- Gates 10 and 13 generate longer sequences than in the previous run (see Caveats).

## Gates, best row of each

Best prefill / best decode is the highest value in that column of the gate's box, whichever
row it is (rows differ between runs where noted). Compression is the widest C10 row (Q4_0 for
Llama-2, which has no C-ladder).

| # | Model | Result | Best prefill t/s (row) now / prev, Δ | Best decode t/s (row) now / prev, Δ | Best compression now / prev | Time now / prev |
|---|---|---|---|---|---|---:|
| 1 | Qwen2-0.5B | pass | 65,502.4 (BF16 ×60) / 61,998.5 (F16 ×60), +5.7% | 5,141.7 (BF16 ×60) / 4,255.1 (F16 ×60), **+20.8%** | — | 6.49 s / 8.24 s |
| 2 | Qwen3.5-0.8B | pass | 26,768.0 (C1 ×2) / 24,983.4 (C1 ×2), +7.1% | 4,164.3 (C8 ×32) / 3,564.2 (C8 ×32), **+16.8%** | 4.38× / 4.38× | 11.46 s / 15.62 s |
| 3 | Llama-3.2-3B | pass | 9,636.5 (Q8_KS ×4) / 9,357.4 (Q8_KS ×4), +3.0% | 745.2 (C8 ×10) / 638.2 (C8 ×10), **+16.8%** | 4.43× / 4.43× | 45.41 s / 62.72 s |
| 4 | Llama-2-7B | pass | 4,921.9 (F16 ×4) / 4,772.5 (F16 ×4), +3.1% | 1,007.7 (BF16 ×48) / 946.9 (BF16 ×48), +6.4% | 3.56× / 3.56× | 54.93 s / 71.30 s |
| 5 | Qwen3-8B | pass | 4,334.2 (BF16 ×1, unbatched) / 4,118.1 (BF16 ×4), +5.2% | 352.8 (C8 ×10) / 298.2 (C8 ×10), **+18.3%** | 5.82× / 5.82× | 46.78 s / 60.60 s |
| 6 | Qwen3.5-9B | pass | 3,906.6 (F16 ×1) / 3,876.6 (BF16 ×4), +0.8% | 837.0 (C8 ×20) / 827.4 (C8 ×20), +1.2% | 5.57× / 5.57× | 31.56 s / 40.81 s |
| 7 | Qwen3.8-27B | pass | 1,230.5 (BF16 ×4) / 1,229.0 (BF16 ×4), +0.1% | 163.3 (BF16 ×4) / 232.1 (BF16 ×4), **−29.6%** | 5.03× / 5.03× | 85.97 s / 91.96 s |
| 8 | Qwen3-30B-A3B | pass | 5,095.9 (C5 ×8) / 4,821.4 (C5 ×8), +5.7% | 397.9 (BF16 ×10) / 290.5 (Q4_0 ×20), **+37.0%** | 5.45× / 5.45× | 117.18 s / 146.11 s |
| 9 | Qwen3.5-35B-A3B | pass | 2,535.4 (C10 ×16) / 2,368.1 (C8 ×5), +7.1% | 239.2 (C10 ×16) / 261.3 (C10 ×16), −8.5% | 7.03× / 7.03× | 70.66 s / 87.29 s |
| 10 | Qwen3.6-35B-A3B (longer run, see Caveats) | pass | 2,281.1 (C8 ×5) / 2,145.0 (C8 ×5), +6.3% | 146.2 (C10 ×16) / 165.8 (C5 ×8), **−11.8%** | 6.43× / 6.45× | 172.41 s / 103.42 s |
| 11 | Qwen3.6-35B AntiLoop+StyleTune, Precision | pass | 2,362.4 (C10 ×16) / 2,234.5 (C8 ×5), +5.7% | 206.3 (C10 ×16) / 213.2 (C10 ×16), −3.2% | 6.47× / 6.47× | 78.18 s / 100.45 s |
| 12 | Qwen3.6-35B AntiLoop+StyleTune, Performance | pass | 2,873.7 (C10 ×16) / 3,246.9 (C8 ×5), **−11.5%** | 290.7 (C10 ×16) / 320.9 (C10 ×16), −9.4% | 6.45× / 6.45× | 64.62 s / 80.67 s |
| 13 | Qwen3.8-Flash-Next (longer run on non-C10 rows) | pass | 1,040.9 (BF16 ×8) / 671.4 (BF16 ×4), **+55.0%** | 58.7 (BF16 ×4) / 57.2 (BF16 ×8), +2.6% | 5.82× / 5.82× | 288.72 s / 184.05 s |
| 14 | probe: Qwen3-30B-A3B | **FAIL** (efficiency) | — | — | — | 229.38 s / 285.79 s |
| 15 | probe: Qwen3.8-Flash-Next | pass | — | — | — | 356.21 s / 369.25 s |

Compression ratios are identical to the previous run on every gate but gate 10, where they
differ by at most 0.02× (C10 ×8 6.43× against 6.45×, C2 3.11× against 3.13×). No gate's
compression moved by more than that.

Wall time of the thirteen gates: 1,074.4 s against 1,053.2 s (+2.0%). Without gates 10 and
13, whose sequences are longer, 613.2 s against 765.8 s (−19.9%); each of the other eleven
gates ran 6–28% faster. Probes: 585.6 s against 655.0 s.

## Wide dense prefill, Llama-2-7B

Prompt phase t/s. *Recorded* is the 2026-09-30 TSV; *prev* is the 2026-10-05 sweep. Peak
tokens are identical across the three.

| Llama-2-7B row | recorded | prev | now | vs prev | vs recorded |
|---|---:|---:|---:|---:|---:|
| F32 ×1 | 4,172.6 | 4,334.0 | 4,530.0 | +4.5% | +8.6% |
| BF16 ×1 | 3,808.7 | 4,437.5 | 4,385.4 | −1.2% | +15.1% |
| BF16 ×8 | 3,686.6 | 4,593.0 | 4,872.1 | +6.1% | **+32.2%** |
| BF16 ×16 | 3,190.5 | 4,546.7 | 4,772.1 | +5.0% | **+49.6%** |
| BF16 ×48 | 1,902.2 | 4,430.6 | 4,645.0 | +4.8% | **+144.2%** |
| Q8_0 ×32 | 2,464.6 | 3,859.7 | 4,049.9 | +4.9% | **+64.3%** |
| Q4_0 ×32 | 2,339.0 | 3,825.5 | 4,005.2 | +4.7% | **+71.2%** |

The wide-prefill gain of the KV claim work holds and is 5% higher on every row. Decode at the
same rows: BF16 ×48 1,007.7 (prev 946.9, recorded 693.5), Q8_0 ×32 506.2 (prev 543.4,
recorded 436.0), Q4_0 ×32 475.0 (prev 548.8, recorded 448.2).

## Per-gate tables

Prefill = `t/s (bulk)`, decode = `t/s (single)`, exactly as printed; Δ is now against prev.
Both runs validated every row that carries a ✓. Compression is the same as the previous run
unless noted.

### Gate 1 — Qwen2-0.5B

Prefill +4 to +10% on every row (within noise to slightly above). Decode moved on every row.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| F32 ×1 | 29,884.5 / 27,943.2 | +6.9% | 419.6 / 245.5 | **+70.9%** |
| BF16 ×1 | 30,883.7 / 28,436.1 | +8.6% | 432.4 / 251.8 | **+71.7%** |
| F16 ×1 | 30,460.4 / 27,808.6 | +9.5% | 403.2 / 221.0 | **+82.4%** |
| F16 ×4 | 60,444.9 / 55,490.0 | +8.9% | 1,323.5 / 762.5 | **+73.6%** |
| F16 ×60 | 64,587.8 / 61,998.5 | +4.2% | 5,125.0 / 4,255.1 | **+20.4%** |
| BF16 ×60 | 65,502.4 / 61,357.1 | +6.8% | 5,141.7 / 4,205.2 | **+22.3%** |

### Gate 2 — Qwen3.5-0.8B

Compression identical on every row (1.88×, 1.87×, 2.13×, 2.33×, 2.82×, 2.66×, 2.93×, 3.23×,
3.59×, 3.98×, 4.16×, 4.38×). Prefill is +3 to +8% on all rows but the two that sat low last
time (C3, C7), and within this run C4–C7 read 18–20k against C0–C3 and C8–C10 at 26k — the
same two-level pattern the previous run had (C3 and C7 at 17.9k and 12.4k).

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| F16 ×1 | 18,102.1 / 17,000.5 | +6.5% | 208.8 / 122.6 | **+70.3%** |
| BF16 ×1 | 18,275.4 / 17,657.8 | +3.5% | 216.2 / 120.6 | **+79.3%** |
| BF16 ×16 | 24,917.6 / 24,064.6 | +3.5% | 2,906.0 / 2,154.4 | **+34.9%** |
| Q8_0 ×4 | 26,118.9 / 24,640.3 | +6.0% | 1,001.4 / 562.5 | **+78.0%** |
| C0 ×2 | 26,678.9 / 24,838.5 | +7.4% | 529.4 / 280.8 | **+88.5%** |
| C1 ×2 | 26,768.0 / 24,983.4 | +7.1% | 528.2 / 282.2 | **+87.2%** |
| C2 ×2 | 26,522.4 / 24,765.8 | +7.1% | 524.7 / 277.2 | **+89.3%** |
| C3 ×2 | 26,673.2 / 17,944.2 | **+48.6%** | 528.6 / 285.4 | **+85.2%** |
| C4 ×2 | 20,146.7 / 18,829.6 | +7.0% | 456.0 / 270.1 | **+68.8%** |
| C5 ×2 | 19,377.7 / 17,904.0 | +8.2% | 444.0 / 282.1 | **+57.4%** |
| C6 ×2 | 18,681.9 / 17,524.8 | +6.6% | 436.0 / 271.8 | **+60.4%** |
| C7 ×2 | 18,010.9 / 12,416.6 | **+45.1%** | 432.5 / 276.2 | **+56.6%** |
| C8 ×32 | 26,096.4 / 24,574.2 | +6.2% | 4,164.3 / 3,564.2 | **+16.8%** |
| C9 ×5 | 25,924.2 / 24,423.8 | +6.1% | 1,218.1 / 625.7 | **+94.7%** |
| C10 ×10 | 26,275.7 / 24,902.3 | +5.5% | 2,058.1 / 1,289.5 | **+59.6%** |

### Gate 3 — Llama-3.2-3B

Compression identical on every row. Prefill +0.3 to +8.6% on every row (a uniform small
rise); single-stream decode +4 to +10% on every ×1 row, wider rows more.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| F32 ×1 | 8,889.4 / 8,571.3 | +3.7% | 129.2 / 123.7 | +4.4% |
| F16 ×1 | 8,845.4 / 8,647.4 | +2.3% | 136.0 / 124.8 | +9.0% |
| F16 ×4 | 9,340.0 / 9,260.8 | +0.9% | 485.4 / 432.5 | **+12.2%** |
| R16 ×1 | 9,349.9 / 8,797.3 | +6.3% | 121.1 / 111.7 | +8.4% |
| Q8_0 ×1 | 9,163.1 / 9,063.5 | +1.1% | 128.8 / 118.1 | +9.1% |
| Q8_Q4 ×1 | 9,273.1 / 8,989.7 | +3.2% | 125.1 / 115.3 | +8.5% |
| BF16 ×4 | 9,322.0 / 8,871.3 | +5.1% | 487.2 / 431.1 | **+13.0%** |
| Q8_1 ×4 | 9,411.7 / 8,972.3 | +4.9% | 412.3 / 376.4 | +9.5% |
| Q8_KS ×4 | 9,636.5 / 9,357.4 | +3.0% | 401.4 / 366.3 | +9.6% |
| Q8_Q4 ×4 | 9,286.0 / 9,254.3 | +0.3% | 417.8 / 376.2 | **+11.1%** |
| Q4_0 ×4 | 9,524.7 / 8,953.5 | +6.4% | 435.9 / 392.9 | **+10.9%** |
| Q4_1 ×4 | 9,156.8 / 9,090.0 | +0.7% | 420.0 / 378.9 | **+10.8%** |
| Q4_KS ×4 | 9,615.3 / 9,007.6 | +6.7% | 430.2 / 387.7 | **+11.0%** |
| C0 ×1 | 9,164.1 / 8,909.8 | +2.9% | 124.6 / 114.3 | +9.0% |
| C1 ×1 | 8,588.6 / 8,233.4 | +4.3% | 126.6 / 115.9 | +9.2% |
| C2 ×1 | 8,614.3 / 8,157.8 | +5.6% | 128.5 / 117.6 | +9.3% |
| C3 ×1 | 9,356.4 / 8,973.9 | +4.3% | 127.4 / 116.9 | +9.0% |
| C4 ×1 | 9,137.0 / 8,823.9 | +3.5% | 127.3 / 115.8 | +9.9% |
| C5 ×1 | 9,516.5 / 8,762.8 | +8.6% | 126.9 / 116.3 | +9.1% |
| C6 ×1 | 9,163.6 / 8,754.0 | +4.7% | 125.9 / 115.5 | +9.0% |
| C7 ×1 | 9,295.0 / 8,807.4 | +5.5% | 126.4 / 116.1 | +8.9% |
| C8 ×10 | 9,243.8 / 8,928.0 | +3.5% | 745.2 / 638.2 | **+16.8%** |
| C9 ×10 | 9,314.5 / 8,862.8 | +5.1% | 729.8 / 599.0 | **+21.8%** |
| C10 ×5 | 9,179.8 / 8,875.7 | +3.4% | 499.0 / 449.2 | **+11.1%** |

### Gate 4 — Llama-2-7B

Peak tokens identical. Prefill in the wide-prefill table above; decode is +6 to +10% on every
row but the two quantised ×32 rows, which fell.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| F32 ×1 | 4,530.0 / 4,334.0 | +4.5% | 94.0 / 86.4 | +8.8% |
| F16 ×1 | 4,347.7 / 4,210.3 | +3.3% | 93.4 / 86.7 | +7.7% |
| F16 ×4 | 4,921.9 / 4,772.5 | +3.1% | 312.6 / 285.3 | +9.6% |
| F16 ×8 | 4,897.1 / 4,560.8 | +7.4% | 516.3 / 480.1 | +7.5% |
| BF16 ×1 | 4,385.4 / 4,437.5 | −1.2% | 96.7 / 89.4 | +8.2% |
| BF16 ×8 | 4,872.1 / 4,593.0 | +6.1% | 518.5 / 482.6 | +7.4% |
| BF16 ×16 | 4,772.1 / 4,546.7 | +5.0% | 768.4 / 720.5 | +6.6% |
| BF16 ×48 | 4,645.0 / 4,430.6 | +4.8% | 1,007.7 / 946.9 | +6.4% |
| Q8_0 ×32 | 4,049.9 / 3,859.7 | +4.9% | 506.2 / 543.4 | −6.8% |
| Q4_0 ×32 | 4,005.2 / 3,825.5 | +4.7% | 475.0 / 548.8 | **−13.4%** |

### Gate 5 — Qwen3-8B

Compression identical on every row. The first row (BF16 ×1, unbatched) read 2,034.1 prefill in
the previous run against 3,891.4 on the next row; it is the first row of the process (a
cold-start effect is likely, not shown) and is the only large prefill delta below. Decode is +6 to +7% on every ×1 C row.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| BF16 ×1 (first row) | 4,334.2 / 2,034.1 | **+113.1%** | 63.0 / 48.5 | **+29.9%** |
| F16 ×1 | 3,835.2 / 3,891.4 | −1.4% | 62.3 / 58.3 | +6.9% |
| F16 ×2 | 4,086.0 / 4,046.1 | +1.0% | 122.3 / 108.9 | **+12.3%** |
| BF16 ×4 | 4,184.5 / 4,118.1 | +1.6% | 224.4 / 209.7 | +7.0% |
| Q8_0 ×4 | 4,145.8 / 4,104.4 | +1.0% | 201.5 / 187.6 | +7.4% |
| C0 ×1 | 3,801.9 / 4,046.0 | −6.0% | 59.2 / 55.6 | +6.5% |
| C1 ×1 | 3,834.4 / 3,870.5 | −0.9% | 60.4 / 56.2 | +7.5% |
| C2 ×1 | 3,928.7 / 3,790.8 | +3.6% | 60.7 / 56.7 | +7.1% |
| C3 ×1 | 3,810.5 / 3,860.7 | −1.3% | 60.3 / 56.4 | +6.9% |
| C4 ×1 | 3,856.0 / 4,063.2 | −5.1% | 60.2 / 56.4 | +6.7% |
| C5 ×1 | 3,920.5 / 3,861.5 | +1.5% | 60.2 / 56.4 | +6.7% |
| C6 ×1 | 3,866.4 / 3,865.1 | +0.0% | 59.7 / 55.9 | +6.8% |
| C7 ×1 | 3,849.4 / 3,992.3 | −3.6% | 59.6 / 55.7 | +7.0% |
| C8 ×10 | 3,971.3 / 3,920.6 | +1.3% | 352.8 / 298.2 | **+18.3%** |
| C9 ×5 | 4,085.0 / 4,075.8 | +0.2% | 237.8 / 208.5 | **+14.1%** |
| C10 ×5 | 4,048.4 / 4,081.1 | −0.8% | 234.5 / 207.0 | **+13.3%** |

### Gate 6 — Qwen3.5-9B

Compression identical on every row. Nothing moved more than 10%: prefill within ±7%, decode
+4 to +10%.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| F16 ×1 | 3,906.6 / 3,651.8 | +7.0% | 96.5 / 91.4 | +5.6% |
| BF16 ×1 | 3,640.2 / 3,584.6 | +1.6% | 96.9 / 92.8 | +4.4% |
| BF16 ×4 | 3,859.9 / 3,876.6 | −0.4% | 395.0 / 370.8 | +6.5% |
| Q8_0 ×4 | 3,853.7 / 3,806.8 | +1.2% | 399.0 / 375.7 | +6.2% |
| C0 ×1 | 3,881.8 / 3,745.4 | +3.6% | 106.2 / 102.1 | +4.0% |
| C1 ×1 | 3,671.0 / 3,555.1 | +3.3% | 108.0 / 101.1 | +6.8% |
| C2 ×1 | 3,598.4 / 3,594.6 | +0.1% | 107.9 / 101.1 | +6.7% |
| C3 ×1 | 3,780.5 / 3,629.2 | +4.2% | 107.5 / 100.5 | +7.0% |
| C4 ×1 | 3,656.0 / 3,686.1 | −0.8% | 106.7 / 98.6 | +8.2% |
| C5 ×1 | 3,749.2 / 3,562.2 | +5.2% | 106.8 / 98.3 | +8.6% |
| C6 ×1 | 3,767.5 / 3,573.2 | +5.4% | 106.9 / 98.8 | +8.2% |
| C7 ×1 | 3,614.1 / 3,662.3 | −1.3% | 106.9 / 99.3 | +7.7% |
| C8 ×20 | 3,799.6 / 3,680.5 | +3.2% | 837.0 / 827.4 | +1.2% |
| C9 ×5 | 3,824.2 / 3,829.4 | −0.1% | 444.2 / 424.0 | +4.8% |
| C10 ×10 | 3,780.7 / 3,651.8 | +3.5% | 729.5 / 664.5 | +9.8% |

### Gate 7 — Qwen3.8-27B

Compression identical on every row. Prefill flat on every row but C8 ×20. Decode: the two
×4 rows fell ~27–30%; the single-stream rows rose 2–10% with the usual alternation between
~55 and ~75 (C1 and C3 low, the rest high, in both runs); C9 and C10 rose ~40% and sit inside
this card's known bimodal range for the C10 ×10 row (19.9–117.4 across the previous session).

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| BF16 ×1 | 956.0 / 945.7 | +1.1% | 9.2 / 9.0 | +2.2% |
| BF16 ×4 | 1,230.5 / 1,229.0 | +0.1% | 163.3 / 232.1 | **−29.6%** |
| Q8_0 ×4 | 1,224.6 / 1,220.5 | +0.3% | 162.3 / 221.6 | **−26.8%** |
| C0 ×1 | 1,176.7 / 1,169.5 | +0.6% | 73.2 / 71.9 | +1.8% |
| C1 ×1 | 1,175.6 / 1,173.2 | +0.2% | 56.1 / 51.2 | +9.6% |
| C2 ×1 | 1,174.7 / 1,171.6 | +0.3% | 76.3 / 70.9 | +7.6% |
| C3 ×1 | 1,177.4 / 1,171.6 | +0.5% | 55.4 / 51.4 | +7.8% |
| C4 ×1 | 1,178.0 / 1,173.0 | +0.4% | 74.8 / 70.7 | +5.8% |
| C5 ×1 | 1,174.9 / 1,168.2 | +0.6% | 75.0 / 70.7 | +6.1% |
| C6 ×1 | 1,180.0 / 1,173.5 | +0.6% | 75.0 / 70.5 | +6.4% |
| C7 ×1 | 1,178.7 / 1,176.0 | +0.2% | 74.7 / 69.8 | +7.0% |
| C8 ×20 | 919.0 / 824.0 | **+11.5%** | 28.2 / 30.6 | −7.8% |
| C9 ×5 | 1,201.2 / 1,212.0 | −0.9% | 70.0 / 49.5 | **+41.4%** |
| C10 ×10 | 1,179.9 / 1,197.1 | −1.4% | 77.6 / 55.5 | **+39.8%** |

### Gate 8 — Qwen3-30B-A3B

Compression identical on every row. Prefill +0 to +11% on the C ladder; decode +20 to +76% on
every row. The ×1 BF16/F16 rows are the noisiest here (BF16 ×1 appears twice, first and last).

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| F16 ×1 | 878.8 / 718.5 | **+22.3%** | 21.6 / 11.8 | **+83.1%** |
| BF16 ×1 | 1,299.8 / 1,662.3 | **−21.8%** | 64.0 / 59.3 | +7.9% |
| BF16 ×10 | 3,911.8 / 4,566.1 | **−14.3%** | 397.9 / 201.2 | **+97.8%** |
| Q8_0 ×20 | 4,660.3 / 4,310.9 | +8.1% | 345.8 / 261.9 | **+32.0%** |
| Q4_0 ×4 | 3,599.9 / 3,658.7 | −1.6% | 207.5 / 140.0 | **+48.2%** |
| C0 ×2 | 2,329.5 / 2,337.5 | −0.3% | 113.0 / 77.8 | **+45.2%** |
| C1 ×2 | 2,369.1 / 2,244.5 | +5.6% | 114.9 / 77.6 | **+48.1%** |
| C2 ×2 | 2,278.5 / 2,211.2 | +3.0% | 107.8 / 78.7 | **+37.0%** |
| C3 ×2 | 2,400.5 / 2,236.4 | +7.3% | 101.5 / 73.6 | **+37.9%** |
| C4 ×2 | 2,311.6 / 2,280.9 | +1.3% | 116.6 / 72.0 | **+61.9%** |
| C5 ×2 | 2,410.8 / 2,336.8 | +3.2% | 106.9 / 80.8 | **+32.3%** |
| C5 ×8 | 5,095.9 / 4,821.4 | +5.7% | 268.4 / 197.8 | **+35.7%** |
| C6 ×2 | 2,456.2 / 2,202.7 | **+11.5%** | 102.6 / 68.5 | **+49.8%** |
| C7 ×2 | 2,453.7 / 2,217.4 | **+10.7%** | 115.0 / 65.2 | **+76.4%** |
| C8 ×2 | 2,330.3 / 2,249.3 | +3.6% | 100.5 / 83.7 | **+20.1%** |
| C9 ×2 | 2,484.1 / 2,254.8 | **+10.2%** | 118.4 / 93.1 | **+27.2%** |
| C10 ×2 | 2,472.3 / 2,259.7 | +9.4% | 116.8 / 91.8 | **+27.2%** |
| BF16 ×1 (last) | 1,680.8 / 1,452.0 | **+15.8%** | 74.2 / 29.5 | **+151.5%** |
| Q4_0 ×20 | 4,916.7 / 4,524.8 | +8.7% | 356.2 / 290.5 | **+22.6%** |

### Gate 9 — Qwen3.5-35B-A3B

Compression identical on every row. Single-stream decode on the C ladder fell 9–20% on every
row while prefill held; the ×4 and ×2 wide rows fell on both columns. The C10 ×16 prefill
in the previous run (652.8) was far below its own C10 ×8 (2,011.2) and below the recorded
752.7; it is 2,535.4 now.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| BF16 ×1 | 616.8 / 462.5 | **+33.4%** | 15.0 / 14.8 | +1.4% |
| BF16 ×4 | 1,931.5 / 2,202.1 | **−12.3%** | 91.8 / 139.3 | **−34.1%** |
| Q8_0 ×2 | 1,191.5 / 1,397.7 | **−14.8%** | 67.5 / 70.6 | −4.4% |
| C0 ×1 | 737.2 / 756.0 | −2.5% | 52.2 / 57.4 | −9.1% |
| C1 ×1 | 743.4 / 755.6 | −1.6% | 53.2 / 62.7 | **−15.2%** |
| C2 ×1 | 745.6 / 766.1 | −2.7% | 54.3 / 67.1 | **−19.1%** |
| C3 ×1 | 751.1 / 764.8 | −1.8% | 57.4 / 68.4 | **−16.1%** |
| C4 ×1 | 749.4 / 743.0 | +0.9% | 54.4 / 63.8 | **−14.7%** |
| C5 ×1 | 748.4 / 750.8 | −0.3% | 51.2 / 63.5 | **−19.4%** |
| C6 ×1 | 750.7 / 756.0 | −0.7% | 51.2 / 62.9 | **−18.6%** |
| C7 ×1 | 759.8 / 759.8 | 0.0% | 56.8 / 68.3 | **−16.8%** |
| C8 ×5 | 2,476.3 / 2,368.1 | +4.6% | 147.9 / 179.7 | **−17.7%** |
| C9 ×2 | 1,240.4 / 1,177.0 | +5.4% | 88.3 / 110.1 | **−19.8%** |
| C10 ×8 | 2,143.1 / 2,011.2 | +6.6% | 199.5 / 234.7 | **−15.0%** |
| C10 ×16 | 2,535.4 / 652.8 | **+288.4%** | 239.2 / 261.3 | −8.5% |

### Gate 10 — Qwen3.6-35B-A3B (longer sequences; see Caveats)

Peak tokens now 905 / 3,662 / 7,284 / 4,567 / 1,466 / 5,748 / 11,482 against 659 / 2,678 /
5,316 / 3,337 / 1,358 / 5,316 / 10,618, so decode here is not measured over the same
sequence. Compression differs by ≤0.02×.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| BF16 ×1 | 506.4 / 414.2 | **+22.3%** | 37.4 / 13.2 | **+183.3%** |
| BF16 ×4 | 1,940.8 / 1,893.5 | +2.5% | 131.5 / 95.4 | **+37.8%** |
| Q8_0 ×1 | 687.2 / 703.5 | −2.3% | 51.2 / 31.6 | **+62.0%** |
| C0 ×1 | 709.0 / 699.1 | +1.4% | 52.1 / 42.9 | **+21.4%** |
| C1 ×1 | 712.9 / 672.0 | +6.1% | 52.5 / 48.3 | +8.7% |
| C2 ×1 | 715.5 / 670.5 | +6.7% | 52.7 / 51.7 | +1.9% |
| C3 ×1 | 718.0 / 676.4 | +6.2% | 52.8 / 52.1 | +1.3% |
| C4 ×1 | 726.8 / 669.4 | +8.6% | 52.6 / 51.1 | +2.9% |
| C5 ×1 | 726.4 / 683.1 | +6.3% | 53.0 / 50.5 | +5.0% |
| C5 ×8 | 1,946.2 / 1,828.8 | +6.4% | 138.5 / 165.8 | **−16.5%** |
| C6 ×1 | 693.5 / 637.7 | +8.8% | 47.7 / 49.3 | −3.2% |
| C7 ×1 | 707.7 / 699.4 | +1.2% | 49.0 / 51.8 | −5.4% |
| C8 ×5 | 2,281.1 / 2,145.0 | +6.3% | 130.5 / 139.0 | −6.1% |
| C9 ×2 | 1,146.1 / 1,074.3 | +6.7% | 71.6 / 80.4 | **−10.9%** |
| C10 ×8 | 1,946.0 / 1,850.9 | +5.1% | 128.4 / 149.5 | **−14.1%** |
| C10 ×16 | 2,210.6 / 571.1 | **+287.1%** | 146.2 / 121.2 | **+20.6%** |

### Gate 11 — Qwen3.6-35B AntiLoop+StyleTune, Precision

Compression identical on every row; peak tokens identical. Prefill on the ×1 rows fell 2–10%
(six rows from −5% to −10%), decode fell 7–27% on every row but BF16 ×1 and Q8_0 ×1.

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| BF16 ×1 | 564.0 / 495.2 | **+13.9%** | 14.5 / 14.7 | −1.4% |
| BF16 ×4 | 1,756.8 / 2,059.0 | **−14.7%** | 87.6 / 119.6 | **−26.8%** |
| Q8_0 ×1 | 674.3 / 751.2 | **−10.2%** | 37.9 / 38.0 | −0.3% |
| C0 ×1 | 693.4 / 754.9 | −8.1% | 48.3 / 54.4 | **−11.2%** |
| C1 ×1 | 695.9 / 740.6 | −6.0% | 49.7 / 63.1 | **−21.2%** |
| C2 ×1 | 697.0 / 757.3 | −8.0% | 51.1 / 63.6 | **−19.7%** |
| C3 ×1 | 702.1 / 742.6 | −5.5% | 54.1 / 63.5 | **−14.8%** |
| C4 ×1 | 705.0 / 746.8 | −5.6% | 54.0 / 63.2 | **−14.6%** |
| C5 ×1 | 702.8 / 718.8 | −2.2% | 53.6 / 65.1 | **−17.7%** |
| C5 ×8 | 1,965.8 / 1,895.6 | +3.7% | 190.2 / 203.1 | −6.4% |
| C6 ×1 | 719.6 / 714.9 | +0.7% | 59.2 / 68.0 | **−12.9%** |
| C7 ×1 | 713.6 / 781.6 | −8.7% | 55.6 / 65.6 | **−15.2%** |
| C8 ×5 | 2,302.9 / 2,234.5 | +3.1% | 143.4 / 175.5 | **−18.3%** |
| C9 ×2 | 1,153.8 / 1,137.1 | +1.5% | 90.7 / 109.3 | **−17.0%** |
| C10 ×8 | 2,014.7 / 1,912.6 | +5.3% | 189.5 / 204.1 | −7.2% |
| C10 ×16 | 2,362.4 / 630.2 | **+274.9%** | 206.3 / 213.2 | −3.2% |

### Gate 12 — Qwen3.6-35B AntiLoop+StyleTune, Performance

Compression identical on every row; peak tokens identical. This is the one gate where prefill
fell on nearly every row, by 9–36%, and decode on every row, by 9–43%. In the previous run
this gate sat well above its recorded rows (C0 ×1 1,350.5 against 971.3 recorded); now it is
close to or below them (859.3).

| mode ×ctx | prefill now / prev | Δ | decode now / prev | Δ |
|---|---:|---:|---:|---:|
| BF16 ×1 | 715.2 / 672.9 | +6.3% | 16.6 / 19.1 | **−13.1%** |
| BF16 ×4 | 2,120.5 / 2,879.1 | **−26.3%** | 99.6 / 174.6 | **−43.0%** |
| Q8_0 ×1 | 849.2 / 1,242.2 | **−31.6%** | 40.3 / 48.9 | **−17.6%** |
| C0 ×1 | 859.3 / 1,350.5 | **−36.4%** | 55.6 / 70.0 | **−20.6%** |
| C1 ×1 | 862.0 / 1,284.3 | **−32.9%** | 60.7 / 74.6 | **−18.6%** |
| C2 ×1 | 872.5 / 1,207.4 | **−27.7%** | 59.6 / 75.0 | **−20.5%** |
| C3 ×1 | 876.3 / 1,251.9 | **−30.0%** | 62.9 / 75.5 | **−16.7%** |
| C4 ×1 | 871.9 / 1,243.1 | **−29.9%** | 59.9 / 70.1 | **−14.6%** |
| C5 ×1 | 877.8 / 1,285.9 | **−31.7%** | 59.2 / 75.2 | **−21.3%** |
| C5 ×8 | 2,361.5 / 2,691.1 | **−12.2%** | 214.8 / 272.5 | **−21.2%** |
| C6 ×1 | 897.7 / 1,203.8 | **−25.4%** | 61.6 / 70.1 | **−12.1%** |
| C7 ×1 | 893.5 / 1,353.9 | **−34.0%** | 64.0 / 75.4 | **−15.1%** |
| C8 ×5 | 2,714.2 / 3,246.9 | **−16.4%** | 161.5 / 200.0 | **−19.3%** |
| C9 ×2 | 1,424.9 / 1,827.0 | **−22.0%** | 96.4 / 124.1 | **−22.3%** |
| C10 ×8 | 2,410.1 / 2,645.6 | −8.9% | 208.4 / 261.8 | **−20.4%** |
| C10 ×16 | 2,873.7 / 968.1 | **+196.8%** | 290.7 / 320.9 | −9.4% |

## Qwen3.8-Flash-Next, full ladder (gate 13)

Prefill / decode t/s, now / prev / recorded (2026-09-30 TSV), every row validated. **The
C0, C5, C8 and the BF16 rows ran longer sequences than in the previous run; the two C10 rows
did not** (peak tokens: BF16 ×1 905 against 713, ×4 3,662 against 2,894, ×8 7,284 against
5,748, C0/C5/C8 ×2 1,850 against 1,466, C5 ×8 7,284 against 5,748; C10 ×2 1,466 and C10 ×8
5,748 in both). The C10 rows are therefore the like-for-like ones.

| row | ctx | prefill now / prev / recorded | prefill Δ vs prev / vs rec | decode now / prev / recorded | decode Δ vs prev / vs rec | compression now / prev / recorded |
|---|---:|---:|---:|---:|---:|---:|
| BF16 (cold) | 1 | 162.9 / 129.9 / 134.2 | **+25.4%** / **+21.4%** | 21.3 / 16.3 / 16.1 | **+30.7%** / **+32.3%** | — |
| BF16 | 4 | 739.8 / 671.4 / 654.7 | **+10.2%** / **+13.0%** | 58.7 / 48.1 / 55.8 | **+22.0%** / +5.2% | — |
| BF16 | 8 | 1,040.9 / 490.8 / 495.4 | **+112.1%** / **+110.1%** | 54.2 / 57.2 / 64.7 | −5.2% / **−16.2%** | — |
| BF16 (warm) | 1 | 251.9 / 239.4 / 244.2 | +5.2% / +3.2% | 19.9 / 18.5 / 20.0 | +7.6% / −0.5% | — |
| C0 | 2 | 498.5 / 429.8 / 422.8 | **+16.0%** / **+17.9%** | 32.3 / 30.7 / 32.8 | +5.2% / −1.5% | 2.19× / 2.19× / 2.18× |
| C5 | 2 | 498.0 / 427.2 / 423.1 | **+16.6%** / **+17.7%** | 32.9 / 28.8 / 34.3 | **+14.2%** / −4.1% | 3.80× / 3.80× / 4.01× |
| C5 | 8 | 1,035.1 / 481.3 / — | **+115.1%** / — | 51.0 / 55.1 / — | −7.4% / — | 3.79× / 3.79× / — |
| C8 | 2 | 456.9 / 415.4 / 424.1 | +10.0% / +7.7% | 31.9 / 28.7 / 33.4 | **+11.1%** / −4.5% | 4.87× / 4.87× / 4.74× |
| C10 | 2 | 486.8 / 425.7 / 430.2 | **+14.4%** / **+13.2%** | 27.5 / 25.0 / 34.9 | +10.0% / **−21.2%** | 5.82× / 5.82× / 5.44× |
| C10 | 8 | 987.5 / 487.1 / 497.9 | **+102.7%** / **+98.3%** | 51.8 / 53.4 / 59.5 | −3.0% / **−12.9%** | 5.80× / 5.80× / 5.43× |

The recorded C5 ×8 row does not exist. Prefill at the ×8 rungs roughly doubled on all three
of BF16, C5 and C10, including the C10 row whose sequence length did not change. Decode at
×8 and C10 ×2 did not improve and remains below the recorded rows (−13% to −21%), as in the
previous run; the previous write-up recorded that gap (6–28% on every row but the cold ×1) as
not attributed, and it still is, though it is smaller on the ×1 and ×4 rows now.

## Engine probes

| Probe | story | worst sustained eff% | worst single | weight uptake | clean C5 ×8 baseline | result | time |
|---|---:|---:|---:|---:|---|---|---:|
| Qwen3-30B-A3B, now | 20/20 | 66% | 58% | 96% of 4,400 MiB released (at its limit; 229 at-limit answers in the drain) | 8/8 at 4,769.1 prefill, 355.6 decode | **FAIL** (VRAM efficiency, threshold 90%) | 229.38 s |
| Qwen3-30B-A3B, prev | 20/20 | 71% | 62% | 90% of 2,416 MiB released (at its limit; 69 at-limit answers) | 8/8 at 919.1 prefill, 139.0 decode | FAIL (VRAM efficiency) | 285.79 s |
| Qwen3-30B-A3B, recorded 2026-09-30 | 20/20 | 43% | — | 75% | — | FAIL (VRAM efficiency) | — |
| Qwen3.8-Flash-Next, now | 8/8 | 100% (no sample both large and persistent enough to judge) | 89% | 97% of 2,976 MiB released | 8/8 at 208.1 prefill, 37.4 decode | PASS | 356.21 s |
| Qwen3.8-Flash-Next, prev | 8/8 | 100% (same) | 92% | 93% of 2,912 MiB released (at its limit) | 8/8 at 118.1 prefill, 34.2 decode | PASS | 369.25 s |
| Qwen3.8-Flash-Next, recorded | 8/8 | 100% | — | 73% (2026-09-30), 88% (2026-09-29) | — | PASS | — |

The panic, exactly as printed (`14_qwen3_30b_a3b_q4.log`):

```
thread 'qwen3_30b_a3b_q4' (4396) panicked at candle-conversation\src\fragmentation_probe\probe.rs:251:9:
1 probe gate(s) failed:
  VRAM efficiency fell to 66% (threshold 90%): 2048 MiB of the ground denied to the weight side was not holding KV
```

Probe 14's efficiency table, now / prev: worst sustained frontier 386 / 310, live 361 / 310,
packed 256 / 221, efficiency 66% / 71%, loss 2,048 MiB / 1,408 MiB; worst single sample
387 / 308, 58% / 62%, loss 2,544 / 1,840 MiB; phase B 57%. The probe's story and weight-uptake
gates passed in both runs, so the K/V is correct; what falls short is the share of the
ground below the frontier that holds KV, as `performance.md` records for this card. The
efficiency figure has now read 31–61% (before compaction ran), 88%, 71% and 66% on this card
in successive runs and is not resolved by this sweep. No `CUDA_ERROR` and no `non-finite`
line appeared in any log of this sweep.

## Ranked changes, now against previous

Like-for-like rows only (peak tokens identical), by percentage; whole-table counts at the
end.

**Largest gains**

| # | Row | Column | prev → now | Δ |
|---|---|---|---|---:|
| 1 | Gate 9 Qwen3.5-35B-A3B, C10 ×16 | prefill | 652.8 → 2,535.4 | +288.4% |
| 2 | Gate 11 Qwen3.6 Precision, C10 ×16 | prefill | 630.2 → 2,362.4 | +274.9% |
| 3 | Gate 12 Qwen3.6 Performance, C10 ×16 | prefill | 968.1 → 2,873.7 | +196.8% |
| 4 | Gate 8 Qwen3-30B-A3B, BF16 ×1 (last row) | decode | 29.5 → 74.2 | +151.5% |
| 5 | Gate 5 Qwen3-8B, BF16 ×1 (first row of a cold process) | prefill | 2,034.1 → 4,334.2 | +113.1% |
| 6 | Gate 13 Flash-Next, C10 ×8 | prefill | 487.1 → 987.5 | +102.7% |
| 7 | Gate 8 Qwen3-30B-A3B, BF16 ×10 | decode | 201.2 → 397.9 | +97.8% |
| 8 | Gate 2 Qwen3.5-0.8B, C9 ×5 | decode | 625.7 → 1,218.1 | +94.7% |
| 9 | Gate 2 Qwen3.5-0.8B, C2 ×2 | decode | 277.2 → 524.7 | +89.3% |
| 10 | Gate 8 Qwen3-30B-A3B, F16 ×1 | decode | 11.8 → 21.6 | +83.1% |

Longer-sequence rows with larger moves, not ranked: gate 10 BF16 ×1 decode 13.2 → 37.4
(+183.3%) and C10 ×16 prefill 571.1 → 2,210.6 (+287.1%); gate 13 BF16 ×8 prefill 490.8 →
1,040.9 (+112.1%) and C5 ×8 prefill 481.3 → 1,035.1 (+115.1%).

Other gains: the C10 ×16 prefill rungs of all four hybrid/MoE gates (9–12) rose 2–4× from
values (571–968) that were also the recorded values (688–900), so the previous run's low
figure was not an outlier of that run; and Qwen2-0.5B and Qwen3.5-0.8B single-stream decode
rose 57–89% on every ×1 and ×2 row.

**Largest regressions**

| # | Row | Column | prev → now | Δ |
|---|---|---|---|---:|
| 1 | Gate 12 Qwen3.6 Performance, BF16 ×4 | decode | 174.6 → 99.6 | −43.0% |
| 2 | Gate 12 Qwen3.6 Performance, C0 ×1 | prefill | 1,350.5 → 859.3 | −36.4% |
| 3 | Gate 9 Qwen3.5-35B-A3B, BF16 ×4 | decode | 139.3 → 91.8 | −34.1% |
| 4 | Gate 12 Qwen3.6 Performance, C7 ×1 | prefill | 1,353.9 → 893.5 | −34.0% |
| 5 | Gate 12 Qwen3.6 Performance, C1 ×1 | prefill | 1,284.3 → 862.0 | −32.9% |
| 6 | Gate 12 Qwen3.6 Performance, C5 ×1 / Q8_0 ×1 | prefill | 1,285.9 → 877.8 / 1,242.2 → 849.2 | −31.7% / −31.6% |
| 7 | Gate 7 Qwen3.8-27B, BF16 ×4 | decode | 232.1 → 163.3 | −29.6% |
| 8 | Gate 11 Qwen3.6 Precision, BF16 ×4 | decode | 119.6 → 87.6 | −26.8% |
| 9 | Gate 7 Qwen3.8-27B, Q8_0 ×4 | decode | 221.6 → 162.3 | −26.8% |
| 10 | Gate 12 Qwen3.6 Performance, BF16 ×4 | prefill | 2,879.1 → 2,120.5 | −26.3% |

Patterns behind the list, by count of rows moving more than 10%:

- **BF16 ×4 decode fell on four gates** (7: −29.6%, 9: −34.1%, 11: −26.8%, 12: −43.0%) and
  Q8_0 ×4 on gate 7 (−26.8%); it rose on gates 3, 4, 5 and 6.
- **Single-stream decode fell on nearly every C-ladder row of gates 9, 11 and 12** (the
  Qwen3.5/3.6 MoE family): 9–20% on gate 9, 7–21% on gate 11, 12–23% on gate 12, mostly
  beyond 10%, while prefill held on gate 9 and fell 2–10% on gate 11's ×1 rows.
- **Gate 12 prefill fell on every row but BF16 ×1 and C10 ×16** (−9% to −36%).
- Elsewhere: gate 8 BF16 ×1 prefill −21.8% and BF16 ×10 −14.3%; gate 9 BF16 ×4 prefill
  −12.3% and Q8_0 ×2 −14.8%; gate 11 BF16 ×4 prefill −14.7%; gate 4 Q4_0 ×32 decode −13.4%.

The gains are concentrated in decode of gates 1–3, 5 and 8 and the C10 ×16 prefill rungs;
the regressions in gates 9, 11 and 12 and the ×4 rows of gate 7.

## Against the recorded rows (performance.md, §3.9, 2026-09-30)

Widest C10 row of each ladder (Llama-2 BF16 ×48), t/s. The recorded decode rows predate the
gate's decode-clock fix (`b705689b2`) and read low, so decode gaps against them are not all
engine; prefill is the cleaner comparison. Gates 10 and 13 are marked where their sequence
differs from the recorded run.

| Model | row | prefill now / recorded | Δ | decode now / recorded | Δ | compression now / recorded |
|---|---|---:|---:|---:|---:|---:|
| Qwen2-0.5B | F16 ×60 | 64,587.8 / 46,762.1 | **+38.1%** | 5,125.0 / 3,565.1 | **+43.8%** | — |
| Qwen3.5-0.8B | C10 ×10 | 26,275.7 / 22,228.5 | **+18.2%** | 2,058.1 / 407.5 | **+405.1%** | 4.38× / 4.11× |
| Llama-3.2-3B | C10 ×5 | 9,179.8 / 7,850.8 | **+16.9%** | 499.0 / 266.7 | **+87.1%** | 4.43× / 4.35× |
| Llama-2-7B | BF16 ×48 | 4,645.0 / 1,902.2 | **+144.2%** | 1,007.7 / 693.5 | **+45.3%** | — |
| Qwen3-8B | C10 ×5 | 4,048.4 / 3,708.2 | +9.2% | 234.5 / 143.7 | **+63.2%** | 5.82× / 5.84× |
| Qwen3.5-9B | C10 ×10 | 3,780.7 / 3,340.9 | **+13.2%** | 729.5 / 193.7 | **+276.6%** | 5.57× / 5.13× |
| Qwen3.8-27B | C10 ×10 | 1,179.9 / 1,109.9 | +6.3% | 77.6 / 60.5 | **+28.3%** | 5.03× / 4.76× |
| Qwen3-30B-A3B | C10 ×2 | 2,472.3 / 2,546.7 | −2.9% | 116.8 / 12.3 | **+849.6%** | 5.45× / 5.42× |
| Qwen3.5-35B-A3B | C10 ×8 | 2,143.1 / 2,088.2 | +2.6% | 199.5 / 78.6 | **+153.8%** | 7.03× / 6.23× |
| Qwen3.6-35B-A3B (longer run) | C10 ×8 | 1,946.0 / 2,179.3 | **−10.7%** | 128.4 / 77.8 | **+65.0%** | 6.43× / 5.96× |
| Qwen3.6-35B Precision | C10 ×8 | 2,014.7 / 1,924.1 | +4.7% | 189.5 / 71.5 | **+165.0%** | 6.47× / 6.00× |
| Qwen3.6-35B Performance | C10 ×8 | 2,410.1 / 2,323.9 | +3.7% | 208.4 / 77.6 | **+168.6%** | 6.45× / 6.00× |
| Qwen3.8-Flash-Next | C10 ×8 | 987.5 / 497.9 | **+98.3%** | 51.8 / 59.5 | **−12.9%** | 5.80× / 5.43× |

Rows below the recorded figure: prefill only on Qwen3-30B-A3B C10 ×2 (−2.9%) and the
Qwen3.6-35B C10 ×8 row of gate 10 (−10.7%, longer run); decode only on Flash-Next
(×8 −16.2% BF16, −12.9% C10; C10 ×2 −21.2%). On gate 12 several single-stream prefill rows
are now below their recorded rows (C0 ×1 859.3 against 971.3, −11.5%; C7 ×1 893.5 against
991.2, −9.9%); the previous run's rows on that gate were 39% above recorded.

The recorded Flash-Next probe row (efficiency 100%, uptake 73% on 2026-09-30 and 88% on
2026-09-29) and the 30B probe row (43% sustained, uptake 75%) are in the probes table.
No recorded row exists for gate 13 C5 ×8, for the clean-baseline throughputs of either
probe, or for the probes' worst single samples.

## Caveats

- **Single runs on a noisy laptop.** Each figure is one run; this write-up has no clock
  sampling inside the runs, so a thermal or clock-cap state cannot be separated from the
  engine here. Deltas under ~5% are noise; 5–10% are not flagged but several gates move
  uniformly in that range (decode on gates 3, 6; prefill on gates 3, 4, 10), which is more
  than random noise.
- **Gates 10 and 13 ran longer sequences.** Peak tokens differ by +246 per session on gate
  10's BF16/C5/C8 rows and +54 on its C9/C10 rows, and by +192 on gate 13's BF16/C0/C5/C8
  rows. This matches `decode_graphs.md`'s statement that those two gates were moved to
  256 generated tokens (C9/C10 at 64), but I did not read the test source to confirm that
  is the change. Their decode deltas, wall times (+66.7% and +56.9%) and compression
  figures are therefore not like-for-like; prefill (a prompt-phase figure) is reported but
  I have not shown it independent of the change.
- **Previous-run outliers.** Gate 5's first row (2,034.1), the C10 ×16 prefill rungs of
  gates 9–11 (571–968, equal to the recorded ones), gate 2's C3/C7 prefill (17.9k, 12.4k)
  and gate 12's whole prefill column (39% above recorded) are unusual in the previous run
  and drive the largest percentages above. The Δ against the recorded rows is the check on
  whether a delta is the engine or the previous run.
- **Bimodal rows.** Qwen3.8-27B's decode alternates between ~55 and ~75 across C1 to C10 in
  both runs and its C10 ×10 row is known to range 19.9–117.4 on this card; do not read its
  +40% as a gain.
- **No attribution.** The what-changed list names categories of change from the design docs;
  no delta here is attributed to one of them. Gate 12 and the BF16 ×4 decode rows are the
  regressions that most want a bisect.
- **Compression on gate 10** differs from the previous run by ≤0.02×; the other twelve are
  identical to the printed digit.
- **Partial sweep.** `deepseek4` was skipped; its gate is not part of this comparison.
- **Probe 14's efficiency failure is parked**, and it still fails; this sweep does not
  resolve it.

## Verification

Chain exit codes (`summary.txt`, identical layout to the previous run): 01–13 `EXIT=0`;
14 `qwen3_30b_a3b_q4` `EXIT=101`; 15 `qwen38_flash_next` `EXIT=0`; `CHAIN DONE`.

Pass lines:

- Gates 1–13: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1236 filtered out`,
  finished in 6.49, 11.46, 45.41, 54.93, 46.78, 31.56, 85.97, 117.18, 70.66, 172.41, 78.18,
  64.62 and 288.72 s. (The previous run read `1180 filtered out`; the filtered count differs by 56.)
- Per-config session lines: 189 `✓` lines across the thirteen gates (6, 15, 23, 10, 16, 15,
  14, 17, 15, 16, 16, 16, 10), every one `100% pass`; no line below 100% and no `✗`. The
  previous run has the same 189 lines, all 100%. Gates with a story-rewrite validation read
  `sessions matched expected output (100% pass, threshold 100%)`; gate 1 reads
  `sessions produced non-empty output (100% pass, threshold 80%)`.
- Probe 14: `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 11 filtered out;
  finished in 229.38s`, with `PASS  20/20 sessions rewrote the story correctly.`,
  `PASS  the weight side reached its limit (229 at-limit answers in the drain, 4248 MiB
  taken on the way, 96% of the 4400 MiB released)` and
  `FAIL  VRAM efficiency fell to 66% (threshold 90%)`.
- Probe 15: `test result: ok. 1 passed; 0 failed; ... finished in 356.21s`, with
  `PASS  8/8 sessions rewrote the story correctly.`,
  `PASS  VRAM efficiency held at or above 90% (worst 100%).` and
  `PASS  the weight side took 97% of the 2976 MiB released.`
- No `CUDA_ERROR`, `launch failed` or `non-finite` line in any of the fifteen logs.
- No tests, builds or GPU work were run for this write-up; every number is read from the
  sweep logs (`sweep5` now, `sweep4` previous) and from the recorded TSV.
