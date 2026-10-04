# MoE live dispatch and KV compaction — RTX 3090 24 GB, 2026-10-04

One run of each width gate and both engine probes on the `npc-engine` working tree
after `a9889aeca` (Add live MoE dispatch design): revision 2 of the live MoE
dispatch (`docs/moe_live_dispatch_design.md` §0) plus the KV compaction and
partition fixes listed below. Partial sweep for this card: every gate except
DeepSeek-V4-Flash (284B). Compared against the morning baseline on build
`8d061e4fc` (`baseline_rtx_3090_24gb_2026-10-04.md`), same card, same day. Single
runs, so a gap under ~5% is noise. Every row is in
`performance_rtx_3090_24gb_rows_2026-10-04_live_dispatch.tsv` (188 rows).

**Result: all 13 gates pass, every validated session. The Qwen3-30B-A3B engine
probe passes for the first time on this card. The Qwen3.8-Flash-Next probe passes
story and uptake and reads 89% efficiency against a 90% threshold** (below).

## The gates

| Model | ctx=1 prefill / decode | widest measured | prefill / decode |
|---|---|---|---|
| Qwen2-0.5B | 25,953.4 / 239.3 | ×60 (BF16) | 42,920.5 / 6,111.4 |
| Qwen3.5-0.8B | 16,730.6 / 125.3 | ×32 (C8) | 19,925.0 / 3,224.6 |
| Llama-3.2-3B | 6,723.0 / 159.3 | ×10 (C8) | 5,378.5 / 696.2 |
| Llama-2-7B | 3,590.3 / 102.9 | ×48 (BF16) | 1,617.0 / 858.6 |
| Qwen3-8B | 3,178.4 / 77.1 | ×10 (C8) | 2,330.7 / 363.9 |
| Qwen3.5-9B | 3,169.5 / 105.1 | ×20 (C8) | 2,422.7 / 875.9 |
| Qwen3.8-27B | 1,045.7 / 60.2 | ×20 (C8) | 745.2 / 344.0 |
| Qwen3-30B-A3B | 3,584.5 / 54.9 (BF16) | ×20 (Q8_0) | 4,780.4 / 349.1 |
| Qwen3.5-35B-A3B | 1,319.1 / 49.0 | ×16 (C10) | 3,874.0 / 827.8 |
| Qwen3.6-35B-A3B | 1,380.9 / 48.0 | ×16 (C10) | 4,095.8 / 876.4 |
| AntiLoop+StyleTune (auto) | 1,043.1 / 37.1 | ×16 (C10) | 2,663.0 / 764.1 |
| AntiLoop+StyleTune (Performance) | 1,543.3 / 53.2 | ×16 (C10) | 4,288.5 / 916.1 |
| Qwen3.8-Flash-Next (Q2_KO) | 515.2 / 72.3 (warm) | ×16 (BF16) | 1,065.8 / 223.0 |
| DeepSeek-V4-Flash | not run (284B, partial sweep) | | |

## Against the morning baseline

The live dispatch touches only the MoE path, so the dense models are a control.
They sit within noise of the baseline: best prefill −1.2% to +0.5%, best decode
−1.6% to +0.2%, and compression is identical. The one larger gap is Qwen2-0.5B
F32 ×1 decode (239.3 against 261.6, −9%). That row is ~4 ms a token and host-bound;
the same gate's BF16 ×1 row reads 266.4.

| Model | ctx=1 decode | widest prefill | widest decode | best prefill | best decode |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 49.0 vs 46.3 (+6%) | 3,874.0 vs 3,019.8 (+28%) | 827.8 vs 741.9 (+12%) | 5,319.7 vs 4,260.2 (+25%) | 827.8 vs 741.9 (+12%) |
| Qwen3.6-35B-A3B | 48.0 vs 44.5 (+8%) | 4,095.8 vs 3,116.3 (+31%) | 876.4 vs 712.7 (+23%) | 5,334.6 vs 4,285.3 (+24%) | 876.4 vs 712.7 (+23%) |
| AntiLoop+StyleTune (auto) | 37.1 vs 36.3 (+2%) | 2,663.0 vs 2,008.6 (+33%) | 764.1 vs 634.3 (+20%) | 4,918.2 vs 3,617.1 (+36%) | 764.1 vs 634.3 (+20%) |
| AntiLoop+StyleTune (Performance) | 53.2 vs 47.0 (+13%) | 4,288.5 vs 3,382.0 (+27%) | 916.1 vs 828.3 (+11%) | 5,598.6 vs 4,413.1 (+27%) | 916.1 vs 828.3 (+11%) |
| Qwen3-30B-A3B | 54.9 vs 45.8 (+20%) | 4,780.4 vs 4,823.0 (−1%) | 349.1 vs 359.6 (−3%) | 4,816.6 vs 4,855.0 (−1%, BF16 ×10) | 349.1 vs 359.6 (−3%, Q8_0 ×20) |
| Qwen3.8-Flash-Next | 72.3 vs 64.4 (+12%, warm) | 1,065.8 vs 991.7 (+7%) | 223.0 vs 234.6 (−5%) | 1,226.0 vs 1,048.8 (+17%, ×4) | 228.1 vs 234.6 (−3%, C10 ×8) |

The best columns use validated rows. Compression is unchanged everywhere (6.42–7.03×
at C10 on the 35B models, 5.98× on Flash-Next, 5.45× on the 30B).

**The four Qwen3.5/3.6-35B rows improve across the board**: +24–36% best prefill and
+11–23% best decode.

**Qwen3-30B-A3B** is level at width and faster narrow. Prefill at ×10–×20 is within
1% of the baseline. The C-modes at ×2 read ~4,520 against ~4,415 prefill (+2%) and
~92 against ~81 decode (+13%). Single-stream decode is +20%. Decode at ×20 is 3%
under: noise-level, and the only 30B row below baseline.

**Qwen3.8-Flash-Next** is faster on prefill at every width: ×4 +17%, ×8 +12%
(1,013.2 against 903.5), ×16 +7%, C10 ×8 +2%, and the C-modes at ×2 +8%. Decode at
×2 is +40–47% (C-modes 140.4–154.2 against 100.4–105.8) and C10 ×8 is +25% (228.1
against 181.9). BF16 decode at ×4 and ×16 is 5% under (125.6 against 131.9, 223.0
against 234.6). The cold first ×1 row reads 32.9 decode against 44.5 at baseline.
That row runs while the expert cache is still filling from pinned RAM. The warm ×1
row, which runs the same prompt again, is +12%.

### What fixed the wide-prefill regression

The first sweep of this work lost 11% of the 30B's prefill at width, and 25–33% of
Flash-Next's at ×8 and above. Two causes, both fixed here:

- **The live launch ran every token tile at 32 wide.** The host-side grouped int8
  GEMM picks its tile width from the average rows per expert: 32, 64 or 128, and
  the wide modes need KO weights. The live launch was pinned to 32 and gave up 2–4×
  weight reuse at prefill. Both paths now use `grouped_int8_n_sub`. The live
  bucketize pads to that width, and the launch passes it to the kernel. The 30B's
  BF16 ×10 prefill went from 4,344.5 to 4,816.6.
- **Flash-Next's look-ahead duplicated PCIe traffic at prefill.** A launch of more
  than 256 tokens touches most of a layer's experts, so promoting the next layer
  ahead of time only re-copied what the workers were about to stream. In one ×16
  run it moved 52.6 GiB and landed late. Speculation (look-ahead and the Markov
  prefetch) now runs only for launches of 256 tokens or fewer.
- **The workers' blocks sit in their own grid rows.** The grid is
  `(row_tiles, worker_rows + launch_tiles)` with `worker_rows = ⌈W / row_tiles⌉`.
  Before, it was widened by W columns, which multiplied every tile row; at 128
  workers that cost more than the extra workers bought. The worker count at prefill
  is 64.

## Engine probes

| Probe | story | worst sustained efficiency | weight uptake | result |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B | 20/20 | 98% (single sample 62%) | at its limit (whole checkpoint resident, 124 at-limit answers in the drain) | PASS |
| Qwen3.8-Flash-Next | 8/8 | **89%** (single sample 69%) | at its limit, 2,631 MiB = 90% of 2,912 MiB released | FAIL (efficiency, 1 point under 90%) |

Earlier runs in this work, on the same compaction code with the 32-wide tiles, put
the Flash-Next probe at 98%, 98% and 97%. This run's 89% is one sustained pair with
a 1,408 MiB loss, at frontier 292 over 204 live regions. Story and uptake are green,
so K/V and the recurrent state are intact. The probe was not re-run.

The 30B probe failed both memory gates at baseline (62% efficiency, 8% uptake), and had
failed them on every earlier build and card (32–62%). It is fixed by these changes:

- **The compaction pass orders its moves across pools by source region, highest
  first.** Concatenated pool by pool, a time-budgeted pass emptied one pool's top arena
  and spent the rest on that pool's low arenas: 40,000–115,000 moves per pass for one
  region of frontier.
- **The planner is O(moves + arenas).** It binary-searched every slot it passed. At a
  frontier of ~300 regions planning alone outran the 80 ms budget, and passes moved 512
  chunks.
- **Each pool takes as many fresh low arenas as it has arenas' worth of chunks above
  them**, up to 64 a pass. With one per pool, a full top arena could drop only one arena
  per pass, however many holes stood below it.
- **Pressure relief packs the pools before evicting turns or conceding expert ground.**
  Without this rung it conceded 193, 240 and 290 MiB of experts in two seconds while
  287 arenas held what 185 would. The weight zone now holds 15.6–17.0 GB through the
  churn, against 12.9–15.7 GB before.
- **A pass that clips is followed at once, up to four times.** The next wave-loop
  iteration was 1–3 s away. Recurrent-state and gallery compaction stay once per
  wave-loop consideration. Running them in the follow-ups as well dropped the
  Flash-Next probe to 76–87%.
- **Hot→warm migrations release the compaction exclusion every 8 residences**, not once
  per compression class. A class can be the whole backlog after a mass retirement.
- **The uptake gate's "at its limit" exemption reads the drain window**, not the totals
  since boot. Those include the load-time fill, so the exemption never applied.

**A latent fault the probe exposed.** A wave's transient tier is placed against the arena
frontier. The ground it lacks is bought from the weight side with the pool lock
released. Regions claimed by the persistence thread during the purchase could leave the
tier one region short, and the wave then failed with "too wide for a partition"
(4 phase-B decodes in one run; churn turns in 4 of 12). The placement now buys again
while purchases keep landing. The error now reports what was conceded instead of
blaming the weight floor.

## Reproduce

```bash
cargo test --release --features cuda -p candle-transformers --lib \
  models::<module>::tests::<gate> -- --exact --ignored --nocapture --test-threads=1
cargo test --release -p candle-conversation --features hub --test kv_fragmentation \
  <qwen3_30b_a3b_q4|qwen38_flash_next> -- --exact --ignored --nocapture --test-threads=1
```
