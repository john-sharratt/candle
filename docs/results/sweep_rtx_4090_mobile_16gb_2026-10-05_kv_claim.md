# Sweep — RTX 4090 Laptop 16 GB, 2026-10-05 (after the wide-prefill KV claim work)

One full sweep — thirteen width gates and both engine probes, one `cargo` process
per model, card to itself, `deepseek4` skipped on this card — on `5776799ac` plus
the uncommitted work below. Core Ultra 9 185H, 31.5 GiB RAM, PCIe 4.0 ×16.

Single runs, so a gap under ~5% is noise on this laptop. Every validated gate row
validated, every session. Probe 14's VRAM-efficiency gate failed (below).

## What changed

A prefill wave claims every KV chunk it will write, for every layer and every
sequence, before anything computes (`wave_admit`). At Llama-2-7B's 32 KV heads a
chunk is 256 band slots, and that claim was 90% of the 48-context prompt phase
(`admit:ensure`, 5.8 of 6.4 s). The causes, found with per-stage host spans and a
claim benchmark (`claim_cost_tests`):

- **The run claim walked every exhausted arena.** `ArenaPool::allocate_run` visited
  each arena from the lowest on every call, and a chunk makes 2·n_kv_head run claims,
  so a claim's cost grew with the arenas in the pool: 19 µs a chunk over ten arenas,
  about 495 µs in the gate over hundreds. An arena's never-used tail only shrinks, so
  the pool now keeps a floor under which every arena is exhausted (`run_floor`) and a
  run claim starts there. Placement is unchanged — lowest arena with a tail first — and
  an arena registered below the floor lowers it.
- **A chunk's 2·n_kv_head runs are claimed as one ordered batch**
  (`alloc_chunk_runs_for_keys`): the same slots in the same order, with the arena and
  dry-class checks made once per arena.
- **The slot-state rebuild** that follows (every `(layer, slot)` per wave) was 1.1 s of
  the phase: the upload coalescer sent a full rebuild as two copies where one covers it
  (`upload_ranges`), and the record serializer cost 6–22× what it needed to
  (`serialize_kv_heads`: no per-head allocations, the identity palette map built once).
  The rebuild fell from 1,100 to 214 ms.
- The expert cache's retired-slot list is released before a weight-zone retraction
  (`release_retired`), which fixes a stager panic after a concession.

## Wide dense prefill

Prompt phase, t/s. *Recorded* is `performance_rtx_4090_mobile_16gb_rows.tsv`
(2026-09-30); *before* is this session's sweep on `5776799ac` without the work.

| Llama-2-7B row | recorded | before | now | vs before |
|---|---:|---:|---:|---:|
| BF16 ×8 | 3,686.6 | 3,441.4 | 4,593.0 | +33% |
| BF16 ×16 | 3,190.5 | 2,737.9 | 4,546.7 | +66% |
| BF16 ×48 | 1,902.2 | 1,331.1 | 4,430.6 | **+233%** |
| Q8_0 ×32 | 2,464.6 | 2,170.1 | 3,859.7 | +78% |
| Q4_0 ×32 | 2,339.0 | 1,907.0 | 3,825.5 | +102% |

At ×48 the prompt phase (`bench:bulk_total`) went from 6.9 s to 2.3 s; `admit:ensure`
from 5.8 s to 1.7 s. Single-stream and narrow rows did not move (F32 ×1 4,334.0,
BF16 ×1 4,437.5): the cost grows with slots and arenas.

## Gates

Widest C10 row of each ladder (BF16 / F16 where noted), t/s, against the recorded rows.
The recorded decode rows predate the gate's decode-clock fix (`b705689b2`) and read
low, so decode gaps against them are not all engine.

| Model | row | prefill now / recorded | decode now / recorded | compression now / recorded |
|---|---|---:|---:|---:|
| Qwen2-0.5B | F16 ×60 | 61,998.5 / 46,762.1 | 4,255.1 / 3,565.1 | — |
| Qwen3.5-0.8B | C10 ×10 | 24,902.3 / 22,228.5 | 1,289.5 / 407.5 | 4.38× / 4.11× |
| Llama-3.2-3B | C10 ×5 | 8,875.7 / 7,850.8 | 449.2 / 266.7 | 4.43× / 4.35× |
| Llama-2-7B | BF16 ×48 | 4,430.6 / 1,902.2 | 946.9 / 693.5 | — |
| Qwen3-8B | C10 ×5 | 4,081.1 / 3,708.2 | 207.0 / 143.7 | 5.82× / 5.84× |
| Qwen3.5-9B | C10 ×10 | 3,651.8 / 3,340.9 | 664.5 / 193.7 | 5.57× / 5.13× |
| Qwen3.8-27B | C10 ×10 | 1,197.1 / 1,109.9 | 55.5 / 60.5 | 5.03× / 4.76× |
| Qwen3-30B-A3B | C10 ×2 | 2,259.7 / 2,546.7 | 91.8 / 12.3 | 5.45× / 5.42× |
| Qwen3.5-35B-A3B | C10 ×8 | 2,011.2 / 2,088.2 | 234.7 / 78.6 | 7.03× / 6.23× |
| Qwen3.6-35B hybrid, Precision | C10 ×8 | 1,912.6 / 1,924.1 | 204.1 / 71.5 | 6.47× / 6.00× |
| Qwen3.6-35B hybrid, Performance | C10 ×8 | 2,645.6 / 2,323.9 | 261.8 / 77.6 | 6.45× / 6.00× |
| Qwen3.8-Flash-Next | C10 ×8 | 487.1 / 497.9 | 53.4 / 59.5 | 5.80× / 5.43× |

The 27B's C10 ×10 decode varies run to run on this card: 55.5 here, 117.4 on an
immediate rerun of the gate on the same build, and 19.9–117.4 across this session's
runs. Its prefill and compression did not move.

Compression moved only where the KV factor rows were retuned (Q2_KO factors, committed
earlier); the ratios are otherwise identical to the previous sweep.

## Qwen3.8-Flash-Next, full ladder

Prefill / decode t/s against the recorded rows (every row validated):

| row | ctx | prefill now / recorded | decode now / recorded | decode Δ | compression now / recorded |
|---|---:|---:|---:|---:|---:|
| BF16 (cold) | 1 | 129.9 / 134.2 | 16.3 / 16.1 | +1% | — |
| BF16 | 4 | 671.4 / 654.7 | 48.1 / 55.8 | −14% | — |
| BF16 | 8 | 490.8 / 495.4 | 57.2 / 64.7 | −12% | — |
| BF16 (warm) | 1 | 239.4 / 244.2 | 18.5 / 20.0 | −8% | — |
| C0 | 2 | 429.8 / 422.8 | 30.7 / 32.8 | −6% | 2.19× / 2.18× |
| C5 | 2 | 427.2 / 423.1 | 28.8 / 34.3 | −16% | 3.80× / 4.01× |
| C5 | 8 | 481.3 / — | 55.1 / — | — | 3.79× / — |
| C8 | 2 | 415.4 / 424.1 | 28.7 / 33.4 | −14% | 4.87× / 4.74× |
| C10 | 2 | 425.7 / 430.2 | 25.0 / 34.9 | −28% | 5.82× / 5.44× |
| C10 | 8 | 487.1 / 497.9 | 53.4 / 59.5 | −10% | 5.80× / 5.43× |

Prefill matches the record on every row. Decode is 6–28% below it on every row but the
cold ×1, with speculation unchanged (draft depth capped at 4, near-full acceptance). The
sweep's two earlier same-day runs read the same, so the gap is not noise. Not yet
attributed.

## Engine probes

| Probe | story | worst sustained eff% | weight uptake | result |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B | 20/20 | 71% | 90% | FAIL (VRAM efficiency, threshold 90%) |
| Qwen3.8-Flash-Next | 8/8 | 100% | 93% | PASS |

Probe 14's efficiency read 88% in an earlier run this session (before the allocator work
above) and 71% in this sweep, against 31–61% in the sweeps before compaction ran; two
runs do not say whether the allocator work moved it. The intermittent `CUDA_ERROR_LAUNCH_FAILED` after KV compaction seen on
earlier runs of this probe did not occur in this run, and the stager panic fixed by
`release_retired` did not recur. Neither the fault nor the efficiency figure is
resolved.

## Verification

`cargo test -p candle-nn --features cuda --lib kv_cache`: 713 passed, 0 failed (the new
pool placement tests, the upload-range tests and the byte-exact record serializer tests
included). No `non-finite` layer in any gate log.
