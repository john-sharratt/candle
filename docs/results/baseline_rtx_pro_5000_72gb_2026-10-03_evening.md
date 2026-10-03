# Baseline — RTX PRO 5000 Blackwell 72 GB, 2026-10-03 (evening)

One full sweep — the fourteen width gates and both engine probes, one `cargo`
process per model, card to itself — on the build after `148423b1`:

- the warp-per-row DeltaNet decode step;
- greedy picks through the fused `batched_sample_argmax` kernel, and the
  deferred allow-list rows in `BatchedSampler`;
- the wave-wide Flash-Next PLE block;
- the DeltaNet decode pointer table read off the state-arena slots, built once
  per forward on Flash-Next as on the Qwen3.5 lineage;
- verify-plan token rows uploaded once per group, and device token ids read
  back in one transfer (Flash-Next, DeepSeek);
- `Tensor::cat` copying runs of adjacent views as one range, and the accept
  walk reading the verify wave's logits in place on the head's span;
- `QWEN4EXP_KV_FACTORS` at 1.7 / 2.8 and `QWEN36_MOE_KV_FACTORS` at 0.90 / 1.9.

Single runs, so a gap under ~5% is noise. Every row validated, every session.
Two first rows read low in the sweep (Qwen3-8B unbatched decode 59.9, Flash-Next
cold prefill 572.1) and were re-run alone; the table carries the re-runs, which
read normal (87.8 and 1,832.2).

Same layout as the morning baseline (`baseline_rtx_pro_5000_72gb_2026-10-03.md`,
build `91fb4da8`) and `docs/performance.md` §3.7: BF16 at one context against
each model's widest measured point.

| Model | ctx=1 prefill / decode | widest measured | prefill / decode |
|---|---|---|---|
| Qwen2-0.5B | 32,866.1 / 285.9 | ×60 (F16) | 99,592.3 / 5,762.9 |
| Qwen3.5-0.8B | 24,910.6 / 170.7 | ×256 (C8) | 39,013.6 / 9,079.1 |
| Llama-3.2-3B | 13,029.8 / 165.3 (C0) | ×10 (C8) | 15,230.9 / 1,034.6 |
| Qwen3-30B-A3B | 8,399.3 / 98.8 | ×20 (Q8_0) | 10,705.8 / 683.1 |
| Qwen3.5-35B-A3B | 7,172.0 / 128.4 | ×64 (C10) | 9,679.1 / 2,228.9 |
| Qwen3.6-35B-A3B | 7,243.3 / 129.6 | ×64 (C10) | 9,688.6 / 2,247.2 |
| Qwen3.6-35B hybrid, Precision | 7,260.8 / 123.2 | ×64 (C10) | 9,786.3 / 2,268.2 |
| Qwen3.6-35B hybrid, Performance | 7,333.2 / 127.8 | ×64 (C10) | 9,892.4 / 2,262.6 |
| Qwen3-8B | 6,485.5 / 87.8 | ×10 (C8) | 6,870.0 / 552.2 |
| Llama-2-7B | 6,110.8 / 138.1 | ×48 | 4,058.5 / 1,504.3 |
| Qwen3.5-9B | 5,770.9 / 144.2 | ×20 (C8) | 6,463.8 / 1,437.3 |
| Qwen3.8-27B | 1,801.6 / 64.1 | ×40 (C10) | 1,846.2 / 687.7 |
| Qwen3.8-Flash-Next | 2,411.0 / 117.8 (warm) | ×16 | 3,247.8 / 668.4 |
| DeepSeek-V4-Flash | 353.5 / 16.6 (warm) | ×16 | 1,170.8 / 78.0 |

**Engine probes** on the same build: Qwen3-30B-A3B story 20/20, worst sustained
efficiency 97%, weights fully resident. Flash-Next story 8/8, worst sustained
efficiency 94%, weight uptake 91%. Both pass all three gates.

## Flash-Next across its ladder

The standalone re-run, Q4_KO experts on the int8 Precision path, ~700-token
prompts. Decode is the aggregate across sessions; compression is the KV cache
against BF16.

| KV mode | sessions | prefill t/s | decode t/s | per session | compression |
|---|---:|---:|---:|---:|---:|
| BF16 (cold) | 1 | 1,832.2 | 104.8 | 104.8 | — |
| BF16 (warm) | 1 | 2,411.0 | 117.8 | 117.8 | — |
| BF16 | 4 | 3,308.3 | 377.6 | 94.4 | — |
| BF16 | 8 | 3,215.0 | 581.8 | 72.7 | — |
| BF16 | 16 | 3,229.3 | 664.8 | 41.6 | — |
| C0 | 2 | 3,030.6 | 209.9 | 105.0 | 2.24× |
| C5 | 2 | 3,026.5 | 204.8 | 102.4 | 3.55× |
| C8 | 2 | 3,076.9 | 207.1 | 103.6 | 5.18× |
| C10 | 2 | 3,061.1 | 206.0 | 103.0 | 6.45× |
| C10 | 8 | 3,315.4 | 558.4 | 69.8 | 6.43× |

## Widest-point decode against the morning and `docs/performance.md` §3.7

| Model | widest | morning (`91fb4da8`) | §3.7 | now | vs morning | vs §3.7 |
|---|---|---:|---:|---:|---:|---:|
| Qwen2-0.5B | ×60 | 4,549.9 | 4,924.7 | 5,762.9 | +26.7% | +17.0% |
| Qwen3.5-0.8B | ×256 C8 | 3,122.7 | 3,353.4 | 9,079.1 | +190.7% | +170.7% |
| Llama-3.2-3B | ×10 C8 | 978.4 | 745.1 | 1,034.6 | +5.7% | +38.9% |
| Qwen3-30B-A3B | ×20 Q8_0 | 618.8 | 595.7 | 683.1 | +10.4% | +14.7% |
| Qwen3.5-35B-A3B | ×64 C10 | 1,179.7 | 1,187.7 | 2,228.9 | +88.9% | +87.7% |
| Qwen3.6-35B-A3B | ×64 C10 | 1,138.5 | 1,201.6 | 2,247.2 | +97.4% | +87.0% |
| Qwen3-8B | ×10 C8 | 505.1 | 460.1 | 552.2 | +9.3% | +20.0% |
| Llama-2-7B | ×48 | 1,440.6 | 917.3 | 1,504.3 | +4.4% | +64.0% |
| Qwen3.5-9B | ×20 C8 | 880.8 | 877.5 | 1,437.3 | +63.2% | +63.8% |
| Qwen3.8-27B | ×40 C10 | 441.9 | 458.1 | 687.7 | +55.6% | +50.1% |
| Qwen3.8-Flash-Next | ×16 | 587.4 | 421.5 | 668.4 | +13.8% | +58.6% |
| DeepSeek-V4-Flash | ×16 | — | 73.5 | 78.0 | — | +6.1% |

Nothing is under §3.7 any more: Qwen2-0.5B ×60 (5,762.9 against 4,924.7),
Qwen3.5-0.8B at one context (170.7 against 168.8) and DeepSeek-V4-Flash ×16
prefill (1,170.8 against 1,120.6) all clear it.

The widest rows move most where the per-step host work scaled with the cohort:
the DeltaNet pointer table (18 recurrent layers × 4 pointers × every sequence),
the per-sequence token-id uploads and their concatenation, and the scored-row
copies and gather — on the 0.8B at ×256 those were 9 ms of a 41 ms step with the
device idle, and ~8 ms of row copies beside it. The DeltaNet decode step is the
other large move, on the hybrids.

## Compression the KV-factor re-sets cost

| Gate | row | before | now |
|---|---|---:|---:|
| Qwen3.8-Flash-Next | C10 ×8 | 6.75× | 6.43× |
| Qwen3.6-35B-A3B | C10 ×64 | 5.94× | 5.73× |
| Qwen3.6-35B hybrid, Performance | C10 ×64 | 5.97× | 5.76× |
| Qwen3.6-35B hybrid, Precision | C10 ×64 | 5.99× | 5.77× |

Flash-Next's re-set pays for the wave-wide PLE projection, which changes its
F32 summation order (see the `QWEN4EXP_KV_FACTORS` comment); the Qwen3.6 row
re-set brought the hybrid on `Int8Mode::Performance` back from 63/64 at C10 ×64.
