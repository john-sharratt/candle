# Sweep — RTX PRO 5000 Blackwell 72 GB, 2026-10-04 (after the kernel-size work)

One full sweep — fourteen width gates and both engine probes, one `cargo` process
per model, card to itself — on the post-merge build (`490d64c6`) plus the
uncommitted kernel-size work that followed it:

- paged decode: one call site per format-dispatch body (prologues folded into
  their loops, merged walks, the ninth-quad round and the multi-pass sweeps as
  loop iterations), rare palette loops rolled, the tile kernel's Q0_V
  read-through as one nine-slot copy;
- head_dim 96 removed from paged decode (prefill never had it);
- quantized matmul: the `iter` loops rolled, and the sixteen tc16/tc32 remainder
  kernels per tile width collapsed to `_0` / `_r` with the remainder read from
  `batch_size`;
- paged prefill: the per-element V window loop rolled;
- `select_kv_format`: the tid-0 per-scale reduction rolled.

Compiled kernel size (sm_120 SASS instructions, every archive) went from
10.49 M to 4.79 M (−54%); the paged-decode archive from 4.16 M to 0.98 M.

Single runs, so a gap under ~5% is noise. Every gate row validated, every session.
**This is the fastest sweep recorded on this card.**

## Gates

BF16 at one context against each model's widest measured point, as in the
earlier sweeps.

| Model | ctx=1 prefill / decode | widest measured | prefill / decode | C10 compression |
|---|---|---|---|---:|
| Qwen2-0.5B | 32,725.4 / 286.7 | ×60 (F16) | 98,571.2 / 5,562.1 | — |
| Qwen3.5-0.8B | 25,634.5 / 191.6 | ×256 (C8) | 39,655.0 / 9,222.2 | 4.38× |
| Llama-3.2-3B | 13,486.8 / 190.2 (F16) | ×10 (C8) | 15,661.8 / 1,081.6 | 4.43× |
| Qwen3-30B-A3B | 8,667.6 / 92.9 | ×20 (Q8_0) | 10,955.3 / 720.1 | 5.45× |
| Qwen3.5-35B-A3B | 7,252.7 / 141.8 | ×64 (C10) | 9,814.6 / 2,333.2 | 7.02× |
| Qwen3.6-35B-A3B | 7,376.6 / 140.2 | ×64 (C10) | 9,837.2 / 2,341.3 | 6.42× |
| Qwen3.6-35B hybrid, Precision | 7,375.3 / 139.3 | ×64 (C10) | 9,927.9 / 2,346.8 | 6.46× |
| Qwen3.6-35B hybrid, Performance | 7,460.0 / 141.2 | ×64 (C10) | 10,020.5 / 2,341.1 | 6.43× |
| Qwen3-8B | 6,578.5 / 96.0 | ×10 (C8) | 6,941.3 / 594.5 | 5.82× |
| Llama-2-7B | 6,200.0 / 147.0 | ×48 | 4,082.1 / 1,530.4 | — |
| Qwen3.5-9B | 5,848.1 / 159.5 | ×20 (C8) | 6,498.4 / 1,458.2 | 5.57× |
| Qwen3.8-27B | 1,816.4 / 67.3 | ×40 (C10) | 1,854.8 / 698.2 | 5.09× |
| Qwen3.8-Flash-Next | 2,473.7 / 116.7 (warm) | ×16 | 3,273.9 / 685.9 | 7.33× (C10 ×8) |
| DeepSeek-V4-Flash | 351.6 / 16.5 (warm) | ×16 | 1,052.0 / 78.0 | — |

Wall clock: 56, 90, 95, 90, 72, 82, 139, 108, 96, 94, 93, 90, 125 and 260 s.

## Engine probes

| Probe | story | worst sustained eff% | weight uptake | result |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B | 20/20 | 100% | weights fully resident | PASS |
| Qwen3.8-Flash-Next | 8/8 | 97% | 91% | PASS |

No `non-finite` layer and no `!!!!` in either log.

## Against the earlier sweeps

Widest-point decode, t/s. *Evening* is `baseline_rtx_pro_5000_72gb_2026-10-03_evening.md`;
*post-merge* is `sweep_rtx_pro_5000_72gb_2026-10-04_post_merge.md`.

| Model | widest | evening | post-merge | now | vs post-merge | vs evening |
|---|---|---:|---:|---:|---:|---:|
| Qwen2-0.5B | ×60 F16 | 5,762.9 | 5,659.9 | 5,562.1 | −1.7% | −3.5% |
| Qwen3.5-0.8B | ×256 C8 | 9,079.1 | 8,500.3 | 9,222.2 | +8.5% | +1.6% |
| Llama-3.2-3B | ×10 C8 | 1,034.6 | 960.4 | 1,081.6 | +12.6% | +4.5% |
| Qwen3-30B-A3B | ×20 Q8_0 | 683.1 | 692.8 | 720.1 | +3.9% | +5.4% |
| Qwen3.5-35B-A3B | ×64 C10 | 2,228.9 | 2,219.9 | 2,333.2 | +5.1% | +4.7% |
| Qwen3.6-35B-A3B | ×64 C10 | 2,247.2 | 2,223.1 | 2,341.3 | +5.3% | +4.2% |
| Qwen3.6-35B hybrid, Precision | ×64 C10 | 2,268.2 | 2,223.5 | 2,346.8 | +5.5% | +3.5% |
| Qwen3.6-35B hybrid, Performance | ×64 C10 | 2,262.6 | 2,214.9 | 2,341.1 | +5.7% | +3.5% |
| Qwen3-8B | ×10 C8 | 552.2 | 527.1 | 594.5 | +12.8% | +7.7% |
| Llama-2-7B | ×48 | 1,504.3 | 1,417.1 | 1,530.4 | +8.0% | +1.7% |
| Qwen3.5-9B | ×20 C8 | 1,437.3 | 1,393.3 | 1,458.2 | +4.7% | +1.5% |
| Qwen3.8-27B | ×40 C10 | 687.7 | 668.9 | 698.2 | +4.4% | +1.5% |
| Qwen3.8-Flash-Next | ×16 | 668.4 | 704.4 | 685.9 | −2.6% | +2.6% |
| DeepSeek-V4-Flash | ×16 | 78.0 | 77.7 | 78.0 | +0.4% | 0.0% |

Single-stream decode moved most — against post-merge, BF16 ×1: Qwen3.5-0.8B
162.1 → 191.6 (+18%), Qwen3.5-9B 136.1 → 159.5 (+17%), Qwen3.5-35B-A3B
124.6 → 141.8 (+14%), Qwen3-8B 85.8 → 96.0 (+12%), Llama-2-7B 133.6 → 147.0
(+10%), Qwen3.8-27B 62.9 → 67.3 (+7%); Llama-3.2-3B F16 ×1 172.7 → 190.2
(+10%).

The post-merge decode regression is recovered and the dense and hybrid models
are now ahead of the evening baseline; prefill and C10 compression are level
everywhere. The gain over post-merge carries both the decode-regression fix
(`8abb8714` — the tile kernel's I-cache bloat and the BMMA kernel's local memory)
and the size work after it; these single runs do not separate the two, but they
show the size work gave none of it back. Qwen2-0.5B −1.7% and Flash-Next −2.6%
are inside single-run noise.

Repeated runs of the Qwen3.5-0.8B gate during the size work put that noise on
measure: on identical decode kernels the ×2 rows' row means moved up to ±1.5%
over six runs, and single C×2 runs ranged 382–416 t/s.
