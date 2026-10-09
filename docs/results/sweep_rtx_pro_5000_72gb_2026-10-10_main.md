# Sweep — RTX PRO 5000 Blackwell 72 GB, 2026-10-10 (main at `21c36efb`)

One full sweep of `main` as pushed (`21c36efb`, the merge of `origin/main` with the decode
recovery), with nothing uncommitted. It covers fourteen width gates and three engine probes.
Each model ran as its own `cargo` process, with zend stopped so the card was free. The
code is the same as the final code of `sweep_rtx_pro_5000_72gb_2026-10-10_post_merge.md`.
The baseline for code changes is the 2026-10-09 sweep (`sweep_rtx_pro_5000_72gb_2026-10-09.md`),
taken before the merge.

All 206 gate rows validated, for every session. No probe log has a `non-finite` layer or a
`!!!!`. These are single runs, so a gap under about 5% is noise.

## Gates

| # | Model | Best prefill t/s | Best decode t/s | Best compression | Time |
|---|---|---:|---:|---:|---:|
| 1 | Qwen2-0.5B | 139,750.4 (F16 ×60) | 6,586.7 (BF16 ×60) | — | 4.6 s |
| 2 | Qwen3.5-0.8B | 84,235.6 (C10 ×10) | 10,805.3 (C8 ×256) | 4.38× (C10) | 31.0 s |
| 3 | Llama-3.2-3B | 25,575.1 (C10 ×5) | 1,153.9 (C9 ×10) | 4.43× (C10) | 45.7 s |
| 4 | Llama-2-7B | 11,676.0 (F16 ×8) | 1,638.4 (BF16 ×48) | 3.56× (Q4_0) | 33.1 s |
| 5 | Qwen3-8B | 11,269.1 (F16 ×2) | 622.3 (C8 ×10) | 5.82× (C10) | 22.3 s |
| 6 | Qwen3.5-9B | 12,348.5 (C9 ×5) | 1,607.7 (C8 ×20) | 5.57× (C10) | 26.5 s |
| 7 | Qwen3.8-27B | 3,602.6 (BF16 ×4) | 734.3 (C10 ×40) | 5.09× (C10) | 68.8 s |
| 8 | Qwen3-30B-A3B | 17,019.5 (C5 ×8) | 939.2 (BF16 ×10) | 5.45× (C10) | 57.1 s |
| 9 | Qwen3.5-35B-A3B | 19,092.3 (C8 ×5) | 2,869.9 (C10 ×64) | 7.03× (C10) | 37.4 s |
| 10 | Qwen3.6-35B-A3B | 18,516.5 (C8 ×5) | 2,761.0 (C10 ×64) | 6.43× (C10) | 47.6 s |
| 11 | Qwen3.6-35B hybrid, Precision | 18,819.9 (C8 ×5) | 2,829.5 (C10 ×64) | 6.47× (C10) | 37.0 s |
| 12 | Qwen3.6-35B hybrid, Performance | 19,577.9 (C8 ×5) | 2,892.5 (C10 ×64) | 6.45× (C10) | 34.2 s |
| 13 | Qwen3.8-Flash-Next | 7,024.1 (BF16 ×4) | 1,163.9 (BF16 ×16) | 7.13× (C10) | 70.9 s |
| 14 | DeepSeek-V4-Flash | 1,177.1 (BF16 ×16) | 114.4 (BF16 ×16) | — | 179.0 s |

Every gate passed.

## Engine probes

| Probe | story | worst sustained eff% | weight uptake | clean C5×8 prefill / decode | result |
|---|---:|---:|---|---|---|
| Qwen3-30B-A3B | 20/20 | 91% | at its limit (0%) | 15,449.9 / 584.2 | PASS |
| Qwen3.6-35B-A3B | 16/16 | **66%** | at its limit (0%) | 15,874.7 / 1,950.9 | FAIL (efficiency) |
| Qwen3.8-Flash-Next | 8/8 | 99% | 70% | 6,518.1 / 993.1 | PASS |

The Qwen3.6 efficiency gate read 66% (threshold 90%): 2,240 MiB of the ground denied to the
weight side was not holding KV. The story gate passed. This gate flips from run to run on this
card and is recorded as open, separately from this work: the same probe read 66% on 2026-10-09
before the merge and 92% in the post-merge sweep. Flash-Next read 80% on the post-merge re-run
and 99% here.

## Strata through the engine

`kv_fragmentation::qwen38_flash_next_strata`, the median of 3 runs with the warm-up excluded.
The workload is a single session with greedy sampling and MTP at ceiling 4. Pairs are
prefill / decode t/s.

| prompt tokens | engine, standalone | engine, straight after the sweep | forward bench | Strata, RTX 5090 |
|---:|---|---|---|---|
| 4,096 | 6,019.5 / 156.6 | 5,987.7 / 136.5 | 6,078.6 / 146.5 | 4,269.8 / 179.4 |
| 32,768 | 5,397.9 / 152.5 | 5,368.0 / 137.5 | 5,414.2 / 155.3 | 5,543.2 / 175.7 |
| 128,000 | 4,776.9 / 141.6 | 4,775.7 / 132.4 | 4,839.5 / 140.8 | 5,778.7 / 165.0 |

**The engine delivers 156.6 / 152.5 / 141.6 t/s decode** on this code, at 17.1–19.3 ms a step
and identical step counts (82/95/94, 81/89/88, 83/89/86) in every run below. That is level
with the engine figure `docs/wave_feeder.md` records for 2026-10-09 (153.2 / 148.5 / 138.7),
and at or above the forward bench.

The run straight after the sweep read 136.5 / 137.5 / 132.4, at 19.7–23.2 ms a step. The same
binary, re-run later, read 156.2 / 152.5 / 141.3; so did a rebuild of the committed code
(the "standalone" column, with the GPU at P1, 2,370 MHz and no throttle reason throughout).
The 2–3 ms a step is a state of the machine after a full sweep, not the code. The
post-merge sweep (130.4) and the 2026-10-09 sweep (130.8) took Strata in the same position
and read low for the same reason; the 2026-10-09 sweep doc's attribution of its 13% to
acceptance alone is wrong, since its step time had grown as well.

A temporary per-phase probe on the step (host time, µs a step at 4,096) found no host
overhead in the engine loop: draft 2,600–3,300, forward launch 4,600–6,100, verify wait
9,200–10,000, everything else under 300, and 20–50 between steps.

## Against the 2026-10-09 sweep (before the merge)

Widest-point decode and BF16 ×1 decode, in t/s. The ×1 column is warm for Flash-Next and
DeepSeek.

| Model | widest | before | now | change | ×1 before | ×1 now | change |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2-0.5B | ×60 | 5,884.1 | 6,586.7 | +11.9% | 539.9 | 560.3 | +3.8% |
| Qwen3.5-0.8B | ×256 C8 | 10,164.1 | 10,805.3 | +6.3% | 341.8 | 347.1 | +1.6% |
| Llama-3.2-3B | ×10 C8 | 1,098.9 | 1,152.2 | +4.9% | cold row | cold row | — |
| Qwen3-30B-A3B | ×20 Q8_0 | 843.4 | 874.3 | +3.7% | 135.2 | 136.6 | +1.0% |
| Qwen3.5-35B-A3B | ×64 C10 | 2,784.7 | 2,869.9 | +3.1% | 230.8 | 230.2 | −0.3% |
| Qwen3.6-35B-A3B | ×64 C10 | 2,724.4 | 2,761.0 | +1.3% | 472.8 | 476.6 | +0.8% |
| Qwen3.6 hybrid, Precision | ×64 C10 | 2,751.2 | 2,829.5 | +2.8% | 259.6 | 253.3 | −2.4% |
| Qwen3.6 hybrid, Performance | ×64 C10 | 2,790.5 | 2,892.5 | +3.7% | 267.6 | 258.8 | −3.3% |
| Qwen3-8B | ×10 C8 | 593.4 | 622.3 | +4.9% | 99.7 | 102.2 | +2.5% |
| Llama-2-7B | ×48 | 1,400.7 | 1,638.4 | **+17.0%** | 157.1 | 163.4 | +4.0% |
| Qwen3.5-9B | ×20 C8 | 1,523.0 | 1,607.7 | +5.6% | 192.7 | 199.8 | +3.7% |
| Qwen3.8-27B | ×40 C10 | 704.2 | 734.3 | +4.3% | cold row | 75.4 | — |
| Qwen3.8-Flash-Next | ×16 | 1,160.6 | 1,163.9 | +0.3% | 293.5 | 295.2 | +0.6% |
| DeepSeek-V4-Flash | ×16 | 62.3 | 114.4 | **+83.6%** | 13.6 | 18.9 | **+39.0%** |

Compression is unchanged on every row.

Against the post-merge sweep, on identical code, every gate row reads level or 2–7% higher.
That lift is the card's state, not code. It does confirm footnote ² of the post-merge sweep:
the Qwen3.5/3.6 ×1 rows that read 2–5% under on that run read 230.2 and 476.6 here, level
with the 2026-10-09 sweep. The two hybrid ×1 rows read 2.4% and 3.3% under 2026-10-09 in
this run, inside single-run noise.

**Sweep result: FAIL on one probe gate** (Qwen3.6 × VRAM efficiency). All fourteen gates pass.
