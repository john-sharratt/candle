# Flash-Next single-session decode — RTX PRO 5000 Blackwell 72 GB, 2026-10-06/07

Two full sweeps — fourteen width gates and two engine probes each, one `cargo`
process per model, card to itself (zend stopped) — after a round aimed at
Qwen3.8-Flash-Next's single-session decode. The first (2026-10-06) ran one session
at a draft ceiling of 12; the second and final one (2026-10-07) at the ceiling of 4
the build ships, the same ceiling as the morning's sweep. Against that sweep
(`sweep_rtx_pro_5000_72gb_2026-10-06.md`) this build carries:

- **A Q4_KO draft head.** The MTP drafter scores its logits through its own copy of
  the LM head narrowed to Q4_KO instead of the trunk's (`qwen4exp/mtp.rs`,
  `DRAFT_LM_HEAD_FORMAT`). The draft's argmax only has to agree with the trunk's
  where the trunk accepts, and verify is lossless, so this trades no output.
- **The draft walk recorded as graphs.** `draft_walk` records its step loop
  (`docs/decode_graphs.md` §3.2), which needed the slot-state and slot-header
  uploads recorded through the wave staging ring (§2.3: `GpuChunks` staging,
  `copy_bytes`, headers carved from the wave arena and priced as
  `WaveBuffer::PrefillSlotHeaders`) and no allocation inside the recording
  (`ensure_for_batch_entries_all` goes eager only when a layer needs work).
- **A per-position draft depth rule** (`draft_depth.rs`): acceptance is tracked
  per draft position and the depth maximises (1 + q₁ + q₁q₂ + …)/(1 + r·d) with
  the model's measured token cost r (`DraftLadder::token_cost`, 0.125 on
  Flash-Next); the accept loop chooses all rows at once for a stateless chooser.
- **The walk's transient tier no longer re-buys a past forward's plan**
  (`bump_arena::cover_wave_transient`): with no tier standing, the walk places
  only what it asked for. Before it, a walk after the prefill merged in the
  prefill's 4.0 GB plan and failed Flash-Next C5×8 into live KV.

Single runs, so a gap under ~5% is noise. **Both sweeps: every gate row validated,
every session; both probes passed all three gates.**

## Qwen3.8-Flash-Next — final sweep, one-session draft ceiling 4

| mode | ctx | prefill t/s | decode t/s | compression | pass |
|---|---:|---:|---:|---:|---|
| BF16 | 1 | 2,875.9 | 214.9 | — | 1/1 |
| BF16 | 4 | 4,175.3 | 725.8 | — | 4/4 |
| BF16 | 8 | 4,244.4 | 962.6 | — | 8/8 |
| BF16 | 16 | 4,142.7 | 1,087.8 | — | 16/16 |
| BF16 | 1 (warm) | 3,555.6 | 234.8 | — | 1/1 |
| C0 | 2 | 4,027.8 | 391.5 | 2.22× | 2/2 |
| C5 | 2 | 4,028.4 | 386.7 | 4.15× | 2/2 |
| C5 | 8 | 4,217.5 | 886.0 | 4.15× | 8/8 |
| C8 | 2 | 4,024.1 | 385.8 | 5.53× | 2/2 |
| C10 | 2 | 4,019.5 | 382.4 | 7.14× | 2/2 |
| C10 | 8 | 4,208.9 | 867.7 | 7.13× | 8/8 |

| row | morning | now | change |
|---|---:|---:|---:|
| BF16 ×1 warm decode | 165.9 | 234.8 | **+41.5%** |
| BF16 ×16 decode | 905.6 | 1,087.8 | **+20.1%** |
| engine probe, clean C5×8 decode | 770.7 | 922.9 | **+19.7%** |

Engine probe: story 8/8, worst sustained efficiency 97% (single sample 79%),
weight zone at its limit (uptake 59%); clean C5×8 4,081.8 / 922.9 t/s.

**What the ceiling is worth.** At one session the speculative rate is set by how
deep the head drafts and how much of it is accepted, so the ceiling has to be named
beside the number. `profile_single_session_decode` (the gate's rewrite, three warm
runs) and `profile_single_session_essay` (free continuation, ~4K context, two warm
runs), on the same build:

| task | ceiling | accepted / step | warm t/s |
|---|---:|---:|---:|
| rewrite | 4 | 4.90 | 235.4 / 235.9 / 236.2 |
| rewrite | 12 | 9.44 | 314.6 (first sweep's gate row) |
| essay | 4 | 2.14 | 87.9 / 88.0 |
| essay | 12 | ~2.2 | 92 |

A ceiling of 12 buys a third more only on a near-verbatim rewrite, where the head
is accepted nine tokens deep; on free text the depth rule settles near 2 under
either ceiling. The build ships 4, and the like-for-like gain over the morning —
same ceiling, same task — is the +41.5% above. The first sweep, at 12, read warm
×1 3,536.6 / 314.6 and ×16 4,138.7 / 1,106.6.

## Strata's benchmark, on this build (2026-10-07)

`quantized_qwen38_moe::tests::strata_bench_single_session` rebuilds the requests of
Strata's published RTX 5090 run (`batch_test::strata_bench`): a synthetic Python
module cut to 4,096 / 32,768 / 128,000 rendered prompt tokens and a 600-word
explanation request, greedy, reasoning off, 256 tokens, one warm-up then three runs
per length. Every prompt landed on its target exactly (5 of its tokens an empty
system turn the harness always prefills). BF16 KV, draft ceiling 4.

| prompt tokens | prefill t/s, median (runs) | decode t/s, median (runs) | Strata RTX 5090 prefill / decode |
|---:|---:|---:|---:|
| 4,096 | 3,303.7 (range 2,275.1–3,304.1) | 103.1 (99.4, 103.1, 115.5) | 4,269.8 / 179.4 |
| 32,768 | 3,138.6 (range 3,012.9–3,143.6) | 98.6 (98.0, 98.6, 108.9) | 5,543.2 / 175.7 |
| 128,000 | 2,973.1 (range 2,971.5–2,977.1) | 91.2 (91.2, 89.0, 101.0) | 5,778.7 / 165.0 |

- **Drafting is not the gap.** Our MTP head committed 2.87–3.19 tokens per verify
  step on these prompts; Strata's `engine.log` gives ~2.7 (e.g. 256 tokens with
  160 of 226 drafts accepted = 96 steps). Its prompt-lookup drafter fired 0–3
  windows per answer.
- **Step cost is.** Ours 27–33 ms per step (mean of the non-first steps), Strata's
  ~14–15 ms (1,385–1,427 ms of decode over 96–98 steps at 4K).
- Experts were mostly, not wholly, resident: 20.6–24.6K of 25,088, hit rate
  98.8–100%, 25–1,389 pinned misses a run, none cold.
- The third run of each length was its fastest, so three runs are still warming.

## The rest of the fleet, BF16 (or F16) ×1 decode — first sweep

The ceiling change touches only Flash-Next's one-session bracket; the final sweep
passed every one of these gates again.

| Model | morning | now | change |
|---|---:|---:|---:|
| Qwen2-0.5B | 546.2 | 545.0 | −0.2% |
| Qwen3.5-0.8B | 318.1 | 320.6 | +0.8% |
| Llama-3.2-3B (F16) | 180.1 | 208.6 | **+15.8%** ¹ |
| Qwen3-30B-A3B | 111.2 | 130.4 / 137.0 (warm) | **+17.3% / +23.2%** |
| Qwen3.5-35B-A3B | 154.7 | 196.0 | **+26.7%** |
| Qwen3.6-35B-A3B | 221.4 | 300.0 | **+35.5%** |
| Qwen3.6-35B hybrid, Precision | 156.1 | 194.1 | **+24.3%** |
| Qwen3.6-35B hybrid, Performance | 159.0 | 198.5 | **+24.8%** |
| Qwen3-8B | 105.8 | 105.1 | −0.7% |
| Llama-2-7B | 166.1 | 164.5 | −1.0% |
| Qwen3.5-9B | 163.4 | 167.2 | +2.3% |
| Qwen3.8-27B | 66.5 | 63.9 | −3.9% |
| DeepSeek-V4-Flash (warm) | 14.1 | 13.2 | −6.4% |

The speculative models gain with Flash-Next: the depth rule, the recorded draft walk
and the batched accept are shared by the `qwen35` lineage. The dense models without
a drafter sit within noise. DeepSeek-V4-Flash's ×1 rows are 9.2 and 13.2 t/s from
one run each, and its changed code paths are only the slot-state uploads it shares.

¹ Llama-3.2-3B's first-rows warm-up from the morning sweep is down to its very first
row: F32 ×1 (unbatched) reads 4,834.9 t/s prefill, and F16 ×1 after it reads
12,543.8 against the morning's 3,963.1.

Engine probe Qwen3-30B-A3B: story 20/20, efficiency 99%, zone at its limit; clean
C5×8 11,502.2 / 567.8 t/s (morning 11,358.9 / 533.0).

Wall clock: 63, 98, 108, 95, 80, 90, 147, 120, 104, 121, 102, 99, 140 and 258 s;
probes 244 and 248 s.
