# Sweep — RTX PRO 5000 Blackwell 72 GB, 2026-10-10 (after merging origin/main)

One full sweep on `more-optimizations`, with the merge of `origin/main` into `41481e37` staged
but not committed, and the recovery work below on top of it. It covers fourteen width gates,
three engine probes, and the Strata benchmark run through the engine. Each model ran as its
own `cargo` process, with zend stopped so the card was free. The reference is the 2026-10-09
sweep (`sweep_rtx_pro_5000_72gb_2026-10-09.md`), taken before the merge.

The merge brought, from `origin/main`: the blended read-ahead predictor (`expert_lre/blend.rs`),
router look-ahead votes, the five-hop Markov tables, the device fault word, and the read-ahead
walk in bucketize. The recovery work on top of it:

- **Host meeting points with the expert pipeline thread.** These waits returned with the
  merged predictor's heavier per-invocation work. The fixes:
  - `observe_expert_hit_rate` reads unsettled hit counts.
  - `reclaim_spare_ground` asks its growth question on the caller's thread.
  - The fault check waits only on a cache that can leave an expert cold (`can_go_cold`).
  - Router look-ahead runs only while the link-time gate (`read_ahead_gate`) says reading
    ahead pays.
- **The predictor's own cost.** `P(to)^α` is taken once per target. Hops are skipped for
  rows the device has already begun. On Flash-Next ×4 these restored decode from 627 to
  797 t/s.
- **A one-slot decode upload batch inside a recorded wave** goes to the staging ring.
  A batch spanning several slots is scattered.
- **Read-ahead lists only candidates at even odds or better** (`blend::BREAK_EVEN`); see
  DeepSeek below.

Every gate row validated, for every session. No probe log has a `non-finite` layer or a
`!!!!`. These are single runs, so a gap under about 5% is noise.

## Gates

BF16 at one context against each model's widest measured point. Pairs are prefill / decode
t/s. Flash-Next and DeepSeek are from their re-runs on the final code. Every other row is
from the sweep itself: the read-ahead floor added after it does not reach a model whose
experts are all resident, or a dense model.

| Model | ctx=1 prefill / decode | widest measured | prefill / decode | C10 compression |
|---|---|---|---|---:|
| Qwen2-0.5B | 53,213.5 / 537.4 | ×60 (F16) | 134,540.1 / 6,400.4 | — |
| Qwen3.5-0.8B | 41,583.2 / 342.6 | ×256 (C8) | 75,058.9 / 10,142.6 | 4.38× |
| Llama-3.2-3B | 5,647.0 / 110.6 (F16) ¹ | ×10 (C8) | 24,536.7 / 1,103.4 | 4.43× |
| Qwen3-30B-A3B | 10,045.8 / 132.9 | ×20 (Q8_0) | 16,130.1 / 848.7 | 5.45× |
| Qwen3.5-35B-A3B | 10,661.0 / 219.1 | ×64 (C10) | 17,618.8 / 2,746.6 | 7.03× |
| Qwen3.6-35B-A3B | 10,140.4 / 453.8 ² | ×64 (C10) | 17,000.1 / 2,655.9 | 6.43× |
| Qwen3.6-35B hybrid, Precision | 10,255.0 / 250.9 | ×64 (C10) | 17,282.0 / 2,709.0 | 6.46× |
| Qwen3.6-35B hybrid, Performance | 11,036.8 / 254.9 | ×64 (C10) | 17,985.4 / 2,775.8 | 6.43× |
| Qwen3-8B | 9,440.1 / 102.0 | ×10 (C8) | 10,560.1 / 596.1 | 5.82× |
| Llama-2-7B | 7,644.0 / 155.8 | ×48 | 10,829.2 / 1,579.1 | — |
| Qwen3.5-9B | 9,792.8 / 195.1 | ×20 (C8) | 11,278.0 / 1,540.4 | 5.57× |
| Qwen3.8-27B | 3,166.5 / 73.1 | ×40 (C10) | 3,304.4 / 705.6 | 5.09× |
| Qwen3.8-Flash-Next | 4,701.4 / 294.0 (warm) | ×16 | 6,821.8 / 1,134.9 | 7.13× (C10 ×8) |
| DeepSeek-V4-Flash | 306.5 / 18.0 (warm) | ×16 | 1,124.8 / 108.8 | — |

¹ **Llama-3.2-3B's first rows are its cold rows**, recorded as open on 2026-10-06. From Q8_0 ×1
on, every row reads 20,000–24,500 t/s prefill, and the C0–C7 ×1 rows decode at 178–184 t/s.

² **The Qwen3.5/3.6 MoE single-stream rows read 2–5% under the 2026-10-09 sweep.** The sweep
just after the merge (before the recovery work) read them level: 231.0 / 475.1 / 256.0 /
267.4. Every file these models run through is unchanged since then: their experts are all
resident (hit 100%, warm tier empty), so the pipeline thread never predicts for them. A
re-run read 462.9. This is run-to-run variance, not code.

## DeepSeek: read-ahead under even odds

The merge took DeepSeek-V4-Flash's warm ×1 decode from 13.7 to 9.8 t/s. Its hit rate rose
at the same time, from 88% to 93%. The gate's decode table named the cause:

- the warm ×1 run made 12,912 read-ahead claims at 8.6% precision, moving 170 GiB of copies
  against 1,205 real misses;
- every width read 3.6–10.8% precision and moved 140–274 GiB.

DeepSeek's sweeps cast no look-ahead votes, so the list was filled from Markov and recent
cells measured at 4–33%. A read-ahead copy is made by the gate launch's workers, in the
stream, over the same link a demand miss uses. A wrong claim therefore costs as much link
time as a right one saves, and the link-time gate never switched off: 13.5 MiB images price
even a few misses well past its 80 µs threshold.

`CellTable::select` now lists only candidates whose cell is at even odds or better. Every
candidate is still judged, so a cell that climbs past one half is listed again.

| DeepSeek BF16 decode | pre-merge (2026-10-09 22:55) | merged | floor |
|---|---:|---:|---:|
| ×1 | 9.6 | 9.1 | 15.7 |
| ×4 | — | 25.7 | 43.6 |
| ×8 | — | 42.6 | 69.4 |
| ×16 | 76.4 | 70.3 | 108.8 |
| ×1 warm | 13.7 | 9.6 | 18.0 |

On this card, Flash-Next's read-ahead was already off after its first 35 claims, so the floor
changes nothing for it here. **Open:** the read-ahead gain that `origin/main` measured for
Flash-Next on the RTX 3090 (×1 warm decode from 28.5–31.6 to 35.3 t/s) needs re-measuring
on that card. Its listed cells should sit at 50% or above.

## Engine probes

| Probe | story | worst sustained eff% | weight uptake | clean C5×8 prefill / decode | result |
|---|---:|---:|---|---|---|
| Qwen3-30B-A3B | 20/20 | 95% | at its limit (0%) | 14,959.4 / 559.5 | PASS |
| Qwen3.6-35B-A3B | 16/16 | 92% | at its limit (0%) | 15,453.1 / 1,882.5 | PASS |
| Qwen3.8-Flash-Next (sweep) | 8/8 | 94% | 71% | 6,239.3 / 959.9 | PASS |
| Qwen3.8-Flash-Next (final code) | 8/8 | **80%** | 71% | 6,264.5 / 956.0 | FAIL (efficiency) |

The story gate passed on every probe run. The Flash-Next efficiency gate read 94% in the sweep
and 80% on the re-run after the read-ahead floor. The floor changes which experts are copied
ahead, not where KV is placed. This gate flips between about 83% and 99% from run to run on
this card, and it is recorded as open, separately from this work. On 2026-10-09 the probes
read 99 / 66 / 78%.

## Strata through the engine

`kv_fragmentation::qwen38_flash_next_strata`, the median of 3 runs with the warm-up excluded.
The workload is a single session with greedy sampling and MTP at ceiling 4, on the final code.

| prompt tokens | engine, this sweep | engine, 2026-10-09 | Strata, RTX 5090 |
|---:|---|---|---|
| 4,096 | 5,970.0 / 130.4 | 5,984.9 / 130.8 | 4,269.8 / 179.4 |
| 32,768 | 5,333.3 / 133.5 | 5,163.3 / 130.8 | 5,543.2 / 175.7 |
| 128,000 | 4,747.6 / 131.6 | 4,623.3 / 131.2 | 5,778.7 / 165.0 |

## Against the 2026-10-09 sweep

Widest-point decode and BF16 ×1 decode, in t/s.

| Model | widest | before | now | change | ×1 before | ×1 now | change |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2-0.5B | ×60 F16 | 5,884.1 | 6,400.4 | +8.8% | 539.9 | 537.4 | −0.5% |
| Qwen3.5-0.8B | ×256 C8 | 10,164.1 | 10,142.6 | −0.2% | 341.8 | 342.6 | +0.2% |
| Llama-3.2-3B | ×10 C8 | 1,098.9 | 1,103.4 | +0.4% | cold row | cold row | ¹ |
| Qwen3-30B-A3B | ×20 Q8_0 | 843.4 | 848.7 | +0.6% | 135.2 | 132.9 | −1.7% |
| Qwen3.5-35B-A3B | ×64 C10 | 2,784.7 | 2,746.6 | −1.4% | 230.8 | 219.1 | −5.1% ² |
| Qwen3.6-35B-A3B | ×64 C10 | 2,724.4 | 2,655.9 | −2.5% | 472.8 | 453.8 | −4.0% ² |
| Qwen3.6 hybrid, Precision | ×64 C10 | 2,751.2 | 2,709.0 | −1.5% | 259.6 | 250.9 | −3.4% ² |
| Qwen3.6 hybrid, Performance | ×64 C10 | 2,790.5 | 2,775.8 | −0.5% | 267.6 | 254.9 | −4.7% ² |
| Qwen3-8B | ×10 C8 | 593.4 | 596.1 | +0.5% | 99.7 | 102.0 | +2.3% |
| Llama-2-7B | ×48 | 1,400.7 | 1,579.1 | **+12.7%** | 157.1 | 155.8 | −0.8% |
| Qwen3.5-9B | ×20 C8 | 1,523.0 | 1,540.4 | +1.1% | 192.7 | 195.1 | +1.2% |
| Qwen3.8-27B | ×40 C10 | 704.2 | 705.6 | +0.2% | cold row | 73.1 | — |
| Qwen3.8-Flash-Next | ×16 | 1,160.6 | 1,134.9 | −2.2% | 293.5 | 294.0 | +0.2% |
| DeepSeek-V4-Flash | ×16 | 62.3 | 108.8 | **+74.6%** | 13.6 | 18.0 | **+32.4%** |

Compression is unchanged on every row.

**Sweep result: FAIL on one probe gate.** All fourteen gates pass on the final code, and so
do the Qwen3-30B and Qwen3.6 probes. The Flash-Next probe passed in the sweep and failed its
efficiency gate (80%) on the final-code re-run; see the engine probes above.
