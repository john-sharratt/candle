# Sweep — RTX PRO 5000 Blackwell 72 GB, 2026-10-04 (after the origin/main merge)

One full sweep — fourteen width gates and both engine probes, one `cargo` process
per model, card to itself — on `bca99a5c`: the evening baseline's build
(`b456c72a`, `baseline_rtx_pro_5000_72gb_2026-10-03_evening.md`) merged with
origin/main (`56ec7184`), which brought in:

- the CPU block codecs held bit-exact to the KV kernels (INT8 block parameters,
  `warp_mirror`) and the reworked Q0_V encode and read path;
- the KV selection rework: flat-only Q0, the sink-capped head scale, partial
  chunks read without their dead slots, and wider C3–C5 candidate ladders;
- the band codec for host-side band reads and writes, and window-aware write
  placement;
- speculative decode inside a stencil's free spans, with the adaptive draft
  depth fed from each turn's `BlockGuard`;

plus the BMMA decode kernel's Q0_V staging as force-inlined helpers (`74f88985`).

Single runs, so a gap under ~5% is noise. Every gate row validated, every session.

## Gates

BF16 at one context against each model's widest measured point, as in the
evening baseline.

| Model | ctx=1 prefill / decode | widest measured | prefill / decode | C10 compression |
|---|---|---|---|---:|
| Qwen2-0.5B | 33,146.6 / 282.2 | ×60 (F16) | 99,431.2 / 5,659.9 | — |
| Qwen3.5-0.8B | 25,040.0 / 162.1 | ×256 (C8) | 39,591.4 / 8,500.3 | 4.38× |
| Llama-3.2-3B | 13,382.4 / 172.7 (F16) | ×10 (C8) | 15,483.5 / 960.4 | 4.43× |
| Qwen3-30B-A3B | 8,501.3 / 92.5 | ×20 (Q8_0) | 10,870.5 / 692.8 | 5.45× |
| Qwen3.5-35B-A3B | 7,286.0 / 124.6 | ×64 (C10) | 9,814.7 / 2,219.9 | 7.02× |
| Qwen3.6-35B-A3B | 7,366.1 / 122.0 | ×64 (C10) | 9,815.4 / 2,223.1 | 6.42× |
| Qwen3.6-35B hybrid, Precision | 7,386.3 / 122.1 | ×64 (C10) | 9,928.5 / 2,223.5 | 6.46× |
| Qwen3.6-35B hybrid, Performance | 7,448.5 / 121.0 | ×64 (C10) | 10,025.1 / 2,214.9 | 6.43× |
| Qwen3-8B | 6,567.0 / 85.8 | ×10 (C8) | 6,894.0 / 527.1 | 5.82× |
| Llama-2-7B | 6,074.1 / 133.6 | ×48 | 4,071.7 / 1,417.1 | — |
| Qwen3.5-9B | 5,839.9 / 136.1 | ×20 (C8) | 6,502.0 / 1,393.3 | 5.57× |
| Qwen3.8-27B | 1,816.7 / 62.9 | ×40 (C10) | 1,855.4 / 668.9 | 5.09× |
| Qwen3.8-Flash-Next | 2,443.5 / 117.4 (warm) | ×16 | 3,278.6 / 704.4 | 7.33× (C10 ×8) |
| DeepSeek-V4-Flash | 351.5 / 16.6 (warm) | ×16 | 1,141.0 / 77.7 | — |

Wall clock: 79, 93, 98, 95, 76, 85, 142, 113, 101, 95, 96, 93, 129 and 256 s.

## Engine probes

| Probe | story | worst sustained eff% | weight uptake | result |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B | 20/20 | 98% | weights fully resident | PASS |
| Qwen3.8-Flash-Next | 8/8 | 77% | 94% | **FAIL (efficiency)** |
| Qwen3.8-Flash-Next, re-run alone | 8/8 | 99% | 91% | PASS |

The sweep's Flash-Next probe failed its efficiency gate. Its story was intact (no
non-finite layer, no `!!!!`), and the loss was not air inside arenas (16 arena
regions, 15 packed) but free regions stranded below the frontier: worst sustained
frontier 284 over 221 live, 1,024 MiB the weight side was denied, with compaction
running (99 attempts, 631 arenas released). Re-run alone on the same build it read
99% (frontier 244 over 195 live), against 94% for the evening probe before the
merge — the 77% is the run-to-run spread of where the frontier sits when the
sample lands, not a change the merge made.

## Against the evening baseline

Widest-point decode, t/s.

| Model | widest | evening | now | change |
|---|---|---:|---:|---:|
| Qwen2-0.5B | ×60 | 5,762.9 | 5,659.9 | −1.8% |
| Qwen3.5-0.8B | ×256 C8 | 9,079.1 | 8,500.3 | −6.4% |
| Llama-3.2-3B | ×10 C8 | 1,034.6 | 960.4 | −7.2% |
| Qwen3-30B-A3B | ×20 Q8_0 | 683.1 | 692.8 | +1.4% |
| Qwen3.5-35B-A3B | ×64 C10 | 2,228.9 | 2,219.9 | −0.4% |
| Qwen3.6-35B-A3B | ×64 C10 | 2,247.2 | 2,223.1 | −1.1% |
| Qwen3.6-35B hybrid, Precision | ×64 C10 | 2,268.2 | 2,223.5 | −2.0% |
| Qwen3.6-35B hybrid, Performance | ×64 C10 | 2,262.6 | 2,214.9 | −2.1% |
| Qwen3-8B | ×10 C8 | 552.2 | 527.1 | −4.5% |
| Llama-2-7B | ×48 | 1,504.3 | 1,417.1 | −5.8% |
| Qwen3.5-9B | ×20 C8 | 1,437.3 | 1,393.3 | −3.1% |
| Qwen3.8-27B | ×40 C10 | 687.7 | 668.9 | −2.7% |
| Qwen3.8-Flash-Next | ×16 | 668.4 | 704.4 | +5.4% |
| DeepSeek-V4-Flash | ×16 | 78.0 | 77.7 | −0.4% |

Decode on the dense and hybrid models fell 2–7%, float KV as well as compressed
(Llama-2 BF16 ×48 −5.8%); the MoE models at width held, and Flash-Next rose.
Prefill is level everywhere. The cause is not yet attributed.

## Compression at C10

| Gate | row | evening | now |
|---|---|---:|---:|
| Qwen3.8-Flash-Next | C10 ×8 | 6.43× | 7.33× |
| Qwen3.6-35B-A3B | C10 ×64 | 5.73× | 6.42× |
| Qwen3.6-35B hybrid, Precision | C10 ×64 | 5.77× | 6.46× |
| Qwen3.6-35B hybrid, Performance | C10 ×64 | 5.76× | 6.43× |

The wider C3–C5 ladders and the selection rework raise every rung (Flash-Next C5
3.55× → 4.20×), with every session still validating; Flash-Next's C10 passes on
the `QWEN4EXP_KV_FACTORS` 1.7 / 2.8 re-set made before the merge.
