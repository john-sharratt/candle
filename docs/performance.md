# Performance — the fleet

Measured figures for every model with a batched-forward path in this tree,
across the machines it is developed on (§1), with an emphasis on **how cost
behaves as the KV cache grows**. Every number here was produced by a test in
this repository; nothing is extrapolated, nothing is transferred between cards,
and §4 — the limits — is as load-bearing as the tables.

## Headline results

Ten results, each measured by a test in this repository and set against what has
been published elsewhere. Notation: **[I*n*]** is one of our own results
(internal references, below); **[E*n*]** is an outside result (external
references, below and in full in §6.8).

1. **A 180B-parameter model runs on a 16 GB laptop GPU with 32 GB of system
   RAM — and every output validates.** Qwen3.8-Flash-Next (180B total, 6B
   active) runs on an RTX 4090 Laptop GPU with a 31.5 GiB host, streaming its
   experts VRAM → RAM → NVMe: 8/8 sessions correct at every rung, 108.5 t/s
   aggregate decode at ×8, 33.4 t/s at one session, 5.79–5.82× KV compression
   at C10 **[I1 · I2]**. No
   laptop-GPU run of this model has been published, and the published
   single-GPU runs that state their host use 64–128 GB of RAM
   **[E49 · E51 · E52]**. On a 24 GB RTX 3090 with 64 GB of RAM the same
   artifact decodes **138.1 t/s at one session and 543.8 t/s aggregate at
   ×16**, every session validated, where the published llama.cpp run on an RTX
   3090 with 128 GB decodes 15 t/s single-stream — one session on the 3090 is
   above every published llama.cpp run of the model, the RTX 5090's 48.0
   included, and at four sessions its 320.5 t/s aggregate is five times the only
   published single-card aggregate (Strata's 63.1 at four sessions) **[I18 · E51 ·
   E52 · E92]**. Single-stream it is now ahead of Strata at an equal expert
   footprint, on a slower bus: 138.1 t/s on a PCIe 3.0 RTX 3090 with 36–49% of
   the experts in VRAM, against 93–94 on a PCIe 5.0 RTX 5070 with 14% **[E90]**
   and 93.0 on a PCIe Gen4 RTX 3090 at 3-bit **[E93]**. On Strata's own benchmark
   requests on Blackwell, Strata still leads single-session decode by 13–14%:
   179.4 / 175.7 / 165.0 t/s on an RTX 5090 against our 156.6 / 152.5 / 141.6
   through the engine on the RTX PRO 5000 at 4K / 32K / 128K, while our prefill
   leads at 4K, 6,019.5 against 4,269.8 (§6.5).

2. **Context length is free.** Flash-Next keeps **99% of its prefill and 111% of
   its decode from 32K to 128K** — decode is *faster* at 128K — while the other
   hybrid models keep 25–30% of prefill over the same range **[I3]**.
   Speculative acceptance holds at exactly 4.85 tokens a step from 8K to 128K
   **[I4]**. Published llama.cpp runs of the same model lose a third of
   their decode between 6K and 110K **[E49]**.

3. **Up to 7.6× KV-cache compression with every output validated.** The
   adaptive per-block ladder's top rung compresses the KV cache 4.1× to 7.6× on
   every model in the fleet that compresses, quantizes 100% of blocks, and still
   reproduces the rewrite correctly —
   including a story retrieved from behind 128,897 tokens of padding **[I4 · I5 ·
   I6]**. The production engines' compressed formats stop at ~1.9× (llama.cpp
   q8_0) and 2× (vLLM FP8) **[E64 · E72]**.

4. **Nearly 4× the compression of llama.cpp's q8_0 KV cache — while slowing
   decode less, done inline, at no measurable prefill cost.** Every
   32-token block is compressed as it is written, in the forward that produced
   it; there is no separate compression pass. Prefill under C10 lands within 5%
   of uncompressed at every depth measured, and at 128K all six models prefill
   marginally *faster* under C10 than uncompressed **[I5 · I7]**. Decode pays
   0–8% at 8K for 3.3×–6.3× on every model measured there **[I7]**, 3–31% at 32K
   for 4.6×–7.5×, and just 19% on Flash-Next at 128K for ~7× **[I5]**. Per unit
   of compression that is cheaper than the shipping formats: llama.cpp's q8_0
   gives ~1.9× for 35% of decode at 32K and 45% at 64K **[E64]**, vLLM's
   TurboQuant gives 2.4–3.4× for 20–34% of throughput **[E73]**, and a llama.cpp
   K/V type pair without a compiled flash-attention kernel drops its prefill
   10–45× **[E66 · E67]**.

5. **In aggregate, one card out-decodes llama.cpp's best published rate — by
   24× like for like, and up to 36×.** Serving concurrent conversations from one card, this engine's
   aggregate decode beats the best published llama.cpp single-stream figure for
   the same model on the same class of card: **2.6–36.3× on the RTX 3090**
   (Qwen3-8B 383.8 vs 115.3 t/s; Qwen3.5-35B-A3B 1,228.8 vs 111.2; Qwen3.8-27B
   358.7 vs 65.3 with MTP; Flash-Next 543.8 at ×16 vs 15, the published run at
   a 130K context on UD-Q4_K_XL against our ~700-token prompt on Q2_KO experts, so
   the least like-for-like row),
   **1.4–3.7× on 16 GB cards** (Qwen3-8B; Qwen3.6-35B-A3B; Flash-Next), and
   **3.1–24.2× on Blackwell** (Flash-Next 1,163.9 at ×16 vs 48.02 on an RTX 5090,
   both from a short prompt, ours on 4-bit experts to its 2-bit — the most
   like-for-like row; Qwen3.5-35B-A3B 2,869.9 vs 194.0) **[I8 · I16 · I18 ·
   E1 · E17 · E19 · E21 · E27 · E45 · E46 · E51]**. On 16 GB that now includes the
   35B MoEs: a 3-bit quant that fits wholly in VRAM decodes a single stream at
   183–249 t/s, and ours reaches 336.7 aggregate with its Q6_K experts streamed
   **[I11 · E45]**. The 35B MoEs reach
   **2,761–2,870 t/s aggregate at ×64** on one 72 GB card, 6–12 times their
   single-session rate, and Qwen3.5-0.8B serves **256 concurrent sessions**
   **[I8]**; the published single-card serving runs of the same models stop at
   5–10 concurrent requests **[E47 · E48]**.

6. **Workstation-class throughput from a laptop.** On the 16 GB laptop,
   Qwen3.5-0.8B prefills 32 concurrent sessions at **46,057 t/s** — above the
   24 GB RTX 3090's 42,982 t/s on the same rung — and decodes them at **3,811 t/s
   aggregate** under C8 compression, and Qwen3-30B-A3B prefills at **6,139 t/s
   with its experts streaming** **[I9 · I10]**. One wave engine carries
   prefill rows and decode rows in the same forward, which is what the engine
   probe runs under load **[I15]**.

7. **A 35B MoE serving sixteen users on a laptop.** Qwen3.5/3.6-35B-A3B at Q6_K —
   ~28 GB of weights — serves ×16 at **~330 t/s aggregate with 6.4–7.0× KV
   compression** on the 16 GB laptop, every session validated **[I11]**. The
   published 16 GB runs of these models are single-stream, at smaller quants
   **[E43 · E44]**.

8. **A 284B model at sixteen-way concurrency on one GPU.** DeepSeek-V4-Flash on a
   single 72 GB card prefills at **1,177 t/s** — above every published
   single-GPU figure for the model, whose best is 748 t/s — and decodes
   **114.4 t/s aggregate**, 4.1× the best published single-GPU decode
   (28 t/s, single-stream) **[I12 · E59 · E63]**.

9. **One engine, every card.** The same gates pass on a PCIe 3.0 RTX 3090 with
   no native FP8, an Ada laptop and a Blackwell workstation card — 213 ladder
   rows on the 3090 and 192 on the laptop, no session failing on either — with
   the engine sizing its own memory partition to each card. On the 3090's current
   build C10 compresses the Qwen3.5-35B to **7.03×**, every session validated
   **[I13 · I14]**.

10. **The whole engine, proven under load — not just its kernels.** The engine
    probe drives admission, per-turn context projection, the persistence thread,
    KV compaction and the three memory tiers together. On the RTX 3090 all three
    probes tell their story in full: Qwen3-30B-A3B 20/20, Qwen3.6-35B-A3B 16/16
    under speculative decode with its recurrent state rewound on every rejected
    draft, and Flash-Next 8/8, with the expert weights taking all the ground they
    may. At `7aecbeab3` all three held **97–99% VRAM efficiency**; on the
    2026-10-10 build two of them fall to 59% and 79% **[I15]**. In ten days
    Flash-Next's decode on that card went from **24.3 to 138.1 t/s at one session
    and from 95.6 to 361.6 t/s at C10 ×8**, with zero loss of validation **[I17]**.

### Internal references — our results

| Ref | Result | Where |
|---|---|---|
| **I1** | Flash-Next on the RTX 4090 Laptop GPU: full ladder, 8/8 validated. Build `21c36efbd` (2026-10-10): BF16 ×8 1,127.5 / **108.5** t/s, C5 ×8 1,127.6 / 107.9, BF16 ×1 145.4 / **33.4** (cold) and 269.1 / 34.9 (warm), C10 ×8 1,128.7 / 101.3, C10 5.79–5.82×. Build `856a0fc40` (2026-10-07): C5 ×8 71.5, BF16 ×1 26.2 (cold). Build `bf291341c` (2026-09-30): BF16 ×8 495.4 / 64.7, C10 ×8 59.5 at 5.43–5.44× | §3.9 *Qwen3.8-Flash-Next*; `results/sweep_rtx_4090_mobile_16gb_2026-10-10.md`; `results/performance_rtx_4090_mobile_16gb_rows.tsv` (last nine rows, 2026-09-30) |
| **I2** | The laptop's host: Core Ultra 9 185H, 31.5 GiB RAM, 16 GB VRAM; Flash-Next's 180B / 6B-active size | §1 machine table; §3.1 and its note ¹ |
| **I3** | Flash-Next depth retention 32K → 128K: prefill 99%, decode 111%; the rest of the fleet 25–30% prefill | §3.2, §3.3 |
| **I4** | `Rewrite` at 8K–128K: acceptance 4.85 on every row; the story validated behind 128,897 tokens | §3.6 *Rewrite* |
| **I5** | C10 at 128K: 4.63×–7.63× compression; Flash-Next 6.97× at −19% decode; 100% of blocks quantized | §3.2, §3.5 |
| **I6** | C10 across the fleet's ladders: 4.11× (Qwen3.5-0.8B) to 7.03× (Qwen3.5-35B, RTX 3090) at width; 7.63× at 128K depth | §3.7–§3.9 ladders; §3.2 |
| **I7** | Compression at 8K: 0–8% of decode for 3.3×–6.3× | §3.4 |
| **I8** | Width, RTX PRO 5000, build `21c36efbd` (2026-10-10): Qwen3.5-35B / Qwen3.6-35B 2,869.9 / 2,761.0 t/s aggregate at ×64 C10 against 230.2 / 476.6 at ×1 BF16 (12.5× / 5.8×); Qwen3.6-35B prefill 10,582.4 → 17,642.4 over the same range; Qwen3.5-0.8B at ×256 | §3.7; `results/sweep_rtx_pro_5000_72gb_2026-10-10_main.md` |
| **I9** | Qwen3.5-0.8B C8 ×32 on the laptop: prefill 46,056.8, decode 3,810.7 t/s (2026-10-10, `21c36efbd`) | §3.9 *Qwen3.5-0.8B* |
| **I10** | Qwen3-30B-A3B Q8_0 ×20: 6,139.1 t/s prefill on the laptop with its experts streamed; Qwen3.5-0.8B C8 ×32 prefill 46,056.8 on the laptop (2026-10-10, `21c36efbd`) against 42,982.2 on the RTX 3090 (2026-10-10, `c375834c8`) | §3.9, §3.8; `results/sweep_rtx_4090_mobile_16gb_2026-10-10.md`, `results/sweep_rtx_3090_24gb_2026-10-10.md` |
| **I11** | Qwen3.5/3.6-35B-A3B C10 ×16 on the laptop: 330.8 / 336.7 t/s aggregate, 7.01× / 6.42× (2026-10-10, `21c36efbd`) | §3.9 *Qwen3.5-35B-A3B*, *Qwen3.6-35B-A3B* |
| **I12** | DeepSeek-V4-Flash on the 72 GB card, build `21c36efbd` (2026-10-10): ×16 1,177.1 t/s prefill, 114.4 t/s aggregate decode | §3.7; `results/sweep_rtx_pro_5000_72gb_2026-10-10_main.md` |
| **I13** | Thirteen gates: 213 rows on the RTX 3090 and 192 on the laptop with no failing session; the same gates on the 72 GB card | §3.7–§3.9; §5 provenance table |
| **I14** | C10 on the RTX 3090's current build: Qwen3.5-35B 7.03×, Qwen3.6-35B 6.45×, Qwen3-8B 5.82×, Flash-Next 5.80× at ×8, every session validated | §3.8 |
| **I15** | Engine probes. RTX 3090 (2026-10-05): Qwen3-30B-A3B 20/20, Qwen3.6-35B-A3B 16/16 with speculative decode, Flash-Next 8/8; worst sustained efficiency 99% / 99% / 98%; weight zone at its limit in all three. Again on 2026-10-07 at `7aecbeab3`, the same stories, 99% / 98% / 97%. On 2026-10-10 at `c375834c8`, the same stories, 59% / 98% / 79% — the Qwen3-30B and Flash-Next probes failing the efficiency gate. RTX 4090 Mobile, Flash-Next: story 8/8, efficiency 100%, uptake 73% (2026-09-30) and 88% (2026-09-29) | §3.8 *Engine probes*; §3.9 *Engine probes* |
| **I16** | Aggregate decode against llama.cpp's best published single-stream rate, same model and card class. Each of ours is the best validated decode row of the latest sweep on that card. RTX 3090 (2026-10-10, `c375834c8`): Qwen3-8B C8 ×10 383.8 vs 115.3; Qwen3-30B-A3B BF16 ×10 400.5 vs 153.6; Qwen3.5-35B C10 ×16 1,228.8 vs 111.2; Qwen3.6-35B C10 ×16 1,346.3 vs 157.66; Qwen3.8-27B C8 ×20 358.7 vs 65.28 (MTP); Llama-2-7B BF16 ×48 934.9 vs 161.89; Flash-Next BF16 ×16 543.8 vs 15 (published at 130K context, UD-Q4_K_XL; ours Q2_KO experts, ~700-token prompt; I18). 16 GB (RTX 4090 Mobile, 2026-10-10, `21c36efbd`): Qwen3-8B C8 ×10 308.4 vs 102.7 (RTX 4080); Qwen3.6-35B C10 ×16 336.7 vs 183.29 / 249.33 MTP (RTX 4080, IQ3_S resident); Flash-Next BF16 ×8 108.5 vs 27.5–29 (RTX 5080). Blackwell (our RTX PRO 5000 vs a published RTX 5090, 2026-10-10, `21c36efbd`): Flash-Next BF16 ×16 1,163.9 vs 48.02 (E52, UD-Q2_K_XL, short prompt; ours Q4_KOEXP, ~700-token prompt); Qwen3-8B C8 ×10 622.3 vs 200.4; Qwen3-30B-A3B BF16 ×10 939.2 vs 226.1; Qwen3.5-35B C10 ×64 2,869.9 vs 194.0; Qwen3.6-35B C10 ×64 2,761.0 vs 333.55 (MTP); Llama-2-7B BF16 ×48 1,638.4 vs 300.40 | §3.7–§3.9 ladders; §6.2–§6.5 |
| **I17** | Flash-Next gate on the RTX 3090, build `e596fad8d` (2026-09-30) → `5776799ac` (2026-10-05) → `7aecbeab3` (2026-10-07) → `c375834c8` (2026-10-10): warm ×1 decode 24.3 → 73.3 → 122.8 → 138.1 t/s, ×8 113.2 → 233.8 → 413.7 → 477.2, C10 ×8 95.6 → 255.1 → 306.3 → 361.6, BF16 ×4 prefill 1,033.4 → 1,229.5 → 1,351.8 → 1,570.5, every row validated on all four. On the laptop, builds `9be7b182c` → `a475e852c`: prefill 430.7 → 627.0 t/s (BF16 ×4) | §3.8 *Qwen3.8-Flash-Next*; `results/performance_rtx_3090_24gb_rows_2026-09-30.tsv`, `results/performance_rtx_3090_24gb_rows_2026-10-05.tsv`, `results/sweep_rtx_3090_24gb_2026-10-07.md`, `results/sweep_rtx_3090_24gb_2026-10-10.md`; §7.3 |
| **I18** | Flash-Next on the RTX 3090 (i7-10700K, 64 GB RAM), Q2_KO experts, build `c375834c8` (2026-10-10): full ladder incl. ×16, every row validated; BF16 ×1 warm 556.5 / 138.1 t/s, ×4 1,570.5 prefill and 320.5 aggregate decode, ×8 477.2, ×16 2,301.9 / 543.8, C10 ×8 361.6, C10 5.80–5.83×; engine probe story 8/8, efficiency 79% (97% at `7aecbeab3`), 110% weight uptake | §3.8 *Qwen3.8-Flash-Next*; `results/sweep_rtx_3090_24gb_2026-10-10.md` |

### External references — published results

| Ref | Source | Link |
|---|---|---|
| **E1** | llama.cpp Discussion #15013, "Performance of llama.cpp on Nvidia CUDA" (Llama-2-7B Q4_0) | https://github.com/ggml-org/llama.cpp/discussions/15013 |
| **E17** | Hardware Corner, RTX 3090 LLM benchmarks (2026-03) | https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-3090/ |
| **E19** | Hardware Corner, RTX 4080 LLM benchmarks (2026) | https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4080/ |
| **E21** | Hardware Corner, RTX 5090 LLM benchmarks (2026-03) | https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5090/ |
| **E27** | jonidimo, "Qwen3.8-27B on One RTX 3090: 14h Measured Benchmark" (2026-08) | https://jonidimo.github.io/qwen38-3090-benchmark/benchmark.html |
| **E43** | InsiderLLM, "Best Way to Run Qwen 3.6 35B MoE Locally" (2026-07) | https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/ |
| **E44** | Magnus919, "Running a 35B MoE Model on a 16GB Consumer GPU" (2026-05-27) | https://magnus919.com/2026/05/running-a-35b-moe-model-on-a-16gb-consumer-gpu/ |
| **E45** | ByteShape, "If It Fits, It Sits: Qwen 3.6 35B" (2026-05-19) | https://byteshape.com/blogs/Qwen3.6-35B-A3B/ |
| **E46** | llama.cpp Discussion #19890, "RTX 5090 (CUDA) vs Radeon AI PRO R9700 (Vulkan) — Qwen3.5-35B-A3B…" (2026-02) | https://github.com/ggml-org/llama.cpp/discussions/19890 |
| **E47** | Millstone AI, Qwen3.5-35B-A3B FP8 on 1× RTX Pro 6000 (2026-02-26) | https://www.millstoneai.com/inference-benchmark/qwen3-5-35b-a3b-fp8-1x-rtx-pro-6000-blackwell |
| **E48** | Millstone AI, Qwen3.6-35B-A3B FP8 on 1× RTX Pro 6000 (2026-05-25) | https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-fp8-1x-rtx-pro-6000-blackwell |
| **E49** | ryan4yin, "Best llama.cpp config for Qwen3.8-Flash-Next (RTX 4090 24GB)" (2026-08-29) | https://gist.github.com/ryan4yin/48617bbddacc7067f10799770b7cc33f |
| **E51** | unsloth/Qwen3.8-Flash-Next-GGUF discussion #3, "Share your model speed here" (2026-08/09) | https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3 |
| **E52** | holy_fox, "Running Qwen3.8-Flash-Next on an RTX 5090 with 128GB RAM using llama.cpp" (2026-08-27) | https://zenn.dev/holy_fox/articles/04887ff8177b87?locale=en |
| **E59** | RockmSockmJesus, "DeepSeek-V4-Flash (284B MoE) at ~28 tok/s on 1x RTX 5090…" (2026-07) | https://gist.github.com/RockmSockmJesus/30a195ccd9b62e981ec2676a99a57b7e |
| **E63** | llama.cpp PR #24162, "DeepSeek V4" (2026-06/07) | https://github.com/ggml-org/llama.cpp/pull/24162 |
| **E64** | SOTAAZ, "llama.cpp KV Cache Quantization, Measured on One A100…" (2026-09-15) | https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en |
| **E66** | Kauan Lopes, "The One llama.cpp Setting That Made My RTX 3090 10× Faster" (2026-06-09) | https://kauanlopes.com/blog/llama-cpp-setting-rtx-3090-10x-faster/ |
| **E67** | llama.cpp issue #24485 (2026-06-11) | https://github.com/ggml-org/llama.cpp/issues/24485 |
| **E72** | vLLM Blog, "The State of FP8 KV-Cache and Attention Quantization in vLLM" (2026-04-22) | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| **E73** | vLLM Blog, "A First Comprehensive Study of TurboQuant: Accuracy and Performance" (2026-05-11) | https://vllm.ai/blog/2026-05-11-turboquant |

---

> **The numbers do not transfer between machines.** The elastic VRAM partition
> sizes itself from what each card has, so a model's compression headroom and
> its expert residency are properties of the card it ran on. Every table names
> its machine, and a row from one card is never set beside a row from another
> except in the comparisons that say they are cross-machine.
>
> **What each machine has measured.**
> - **RTX PRO 5000 Blackwell 72 GB** — the reference, and the source of every
>   depth measurement here. Three sequential sweeps: a depth+width sweep on
>   2026-09-03, a width-only sweep on 2026-09-13 (build `2c5f065c` + working
>   tree, same machine and toolchain as §1), and a width-only sweep run twice on
>   2026-09-15 (build `23623c6b`, the decode-slot refresh regression fixed — §4).
>   The depth tables (§3.2–§3.5, §3.6's curves) are the first sweep; the width
>   tables (§3.6 *Width*, §3.7) report the **highest** measurement per cell,
>   **†** marking a 2026-09-13 cell and **◆** a 2026-09-15 cell. A best-of-several
>   sits above any one run by up to the 1–4% noise floor (§5).
> - **RTX 3090 24 GB** — a width/throughput gate sweep on 2026-10-10 (§3.8),
>   build `c375834c8`: the same `test_parallel_batched_forwarding*` gates as
>   §3.7, run one model at a time, plus all three `kv_fragmentation` engine
>   probes (§2.2). Ten of the fleet's models, the two AntiLoop+StyleTune hybrids,
>   and **Qwen3.8-Flash-Next** from the same Q2_KO-expert artifact the 4090
>   Mobile runs; the 284B DeepSeek was not run. **No depth curves** — the
>   `long_context_*` and `profile_*` gates were not run on this card, so the 3090
>   appears in the width tables only.
> - **RTX 4090 Mobile 16 GB** — a width/throughput gate sweep on 2026-10-10
>   (§3.9), build `21c36efbd`: the same `test_parallel_batched_forwarding*` gates,
>   one model at a time. Ten of the fleet's models, the two AntiLoop+StyleTune
>   hybrids, and **Qwen3.8-Flash-Next**, which runs here from its Q2_KO-expert
>   artifact through the expert cache's streaming tiers; the 284B DeepSeek was not
>   run. The two `kv_fragmentation` engine probes (§2.2) are from the 2026-09-30
>   sweep (build `bf291341c`). **No depth curves**, as on the 3090.
>
> Published figures from other engines on comparable cards, for comparison, are
> §6. What remains to be measured on each machine is collected in §7.

---

## 1. The machines

Three machines carry this work, and **none of it transfers between them without
re-measurement** — the elastic VRAM partition sizes itself from what each card
finds, so both the compression ladder's headroom and the expert cache's
residency are properties of the card a row ran on.

| | RTX PRO 5000 Blackwell | RTX 3090 | RTX 4090 Mobile |
|---|---|---|---|
| **VRAM** | 72 GB GDDR7 (73,415 MiB) | 24 GB (24,576 MiB) | 16 GB (16,376 MiB) |
| **Compute capability** | 12.0 (sm_120) | 8.6 (sm_86, GA102) | 8.9 (sm_89, Ada) |
| **Native FP8** | yes | **no** (sm < 8.9) | yes |
| **CPU** | AMD Ryzen 9 9950X3D, 16C/32T | Intel i7-10700K, 8C/16T | Intel Core Ultra 9 185H, 16C/22T |
| **System RAM** | 189 GB | 64 GB | 32 GB (31.5 GiB) |
| **Host↔GPU link** | PCIe 5.0 ×16 | **PCIe 3.0 ×16 (~12 GB/s)** | PCIe 4.0 ×16 (~25 GB/s) |
| **Max SM / mem clock** | 3,090 / 14,001 MHz | — | 3,105 / 9,001 MHz |
| **OS** | Windows 11 Pro (26200) | Windows 11 Pro (26200) | Windows 11 Pro (26200) |
| **GPU driver model** | WDDM (not TCC) | WDDM (not TCC) | WDDM (not TCC), driver 596.08 |
| **Build** | `--release --features cuda` | `--release --features cuda` | `--release --features cuda` |
| **Measured here** | depth + width (§3.2–§3.7) | width gate sweep + engine probes (§3.8) | width gate sweep + engine probes (§3.9) |

Properties that shape several results, worth stating before the tables:

- **WDDM, not TCC**, on all three cards. Kernel launches carry the Windows
  display-driver model's submission overhead, which is the floor under
  single-session decode on every small model here. Recording each forward as a
  chain of CUDA graphs (`docs/decode_graphs.md`) removes most of the per-launch
  cost; what remains is one segment submission per MoE layer. A Linux/TCC host
  would move the decode column and leave the prefill column roughly alone — by
  how much is unmeasured here, and Strata's paper also names it without a number.
- **72 GB on one card.** Every model in the reference report except the 284B and
  Qwen3.8-Flash-Next fits its weights resident, so their depth curves below are
  *not* contaminated by weight paging. That is the point of running them there.
  Flash-Next holds most but not all of its 25,088 experts (48 layers × 512 plus
  the MTP head's 512): 20.6–24.6K resident across a run, the zone shrinking as KV
  grows, hit rate 98.8–100%, with the misses copied from pinned RAM by the expert
  kernels.
- **The 3090's two traps** (both from `CLAUDE.md`'s fleet notes, and both real
  in the numbers). Its host caps the link at **PCIe 3.0** — ~12 GB/s, roughly
  half the 4090 Mobile's despite 50% more VRAM — so any warm↔hot KV or
  expert-stream cost is paid at that rate; and **sm_86 has no native FP8**, so
  the int8-MMA and the FP8 provenance/expert fast paths degrade to the next
  rung. Neither changes a row's validity, but both shape the 3090's absolute
  rates against the Blackwell card, so its numbers are never subtracted from the
  72 GB card's to claim an architectural result.

---

## 2. What is being measured

### 2.1 Every rung is inside the checkpoint's own context window

This governs which rows exist at all, so it comes first.

A model asked for more positions than it was trained to address does not report
a deeper measurement of this engine. It reports whatever its rope tables
extrapolate to out there — a number that moves with the position-scaling scheme
and not with anything the cache or the kernels do. Such a row cannot be set
beside an in-window row, because the two are not measuring the same thing.

**This fleet runs native windows only. No YaRN, no position interpolation, no
scaling applied at load.** Each gate declares its checkpoint's window and
`long_context_gate` refuses any rung where `prompt + generated > window`,
before the model loads. Three unit tests pin the check, including the boundary
that actually bites: a prompt that fits with fewer than `generate` positions to
spare does not fit once decoding starts.

The window is a property of the checkpoint **as this code loads it**, which is
not always the number the file advertises:

| Checkpoint | Window used | Basis |
|---|---:|---|
| Qwen3.5-0.8B / 9B / 35B-A3B | 262,144 | `context_length`; native, no scaling involved |
| Qwen3.6-35B-A3B, Qwen3.8-27B | 262,144 | as above |
| Qwen3.8-Flash-Next | 262,144 | as above |
| Qwen3-30B-A3B-Instruct-2507 | 262,144 | as above (the 2507 release; the original is 32K) |
| DeepSeek-V4-Flash-0731 | 1,048,576 | as above |
| Qwen3-8B | 40,960 | `context_length`; 131K needs YaRN, which is not used |
| Qwen2-0.5B-Instruct | 32,768 | `context_length` |
| **Llama-3.2-3B (Nidum)** | **8,192** | **not** its declared 131,072 — see below |
| Llama-2-7B-Chat | 4,096 | `context_length`; honest, no scaling of any kind |

**Llama-3.2-3B is the case that shows why the declared number is not
sufficient.** Its GGUF advertises `context_length = 131072`, but carries no rope
scaling metadata at all — no `rope.scaling.type`, no `factor`, no
`original_context_length`. Llama 3.2 reaches 128K *only* through llama3 rope
scaling, factor 32 over a base window of 8,192. The loader reads exactly those
keys, finds nothing, and builds plain RoPE at theta = 500,000. The file
therefore declares 131,072 and supplies the machinery for 8,192, and 8,192 is
the number that describes what runs. This is measurable, not inferential: asked
for 32K, that checkpoint emits `". \n\n"` repeated to the token limit,
identically under BF16, C5 and C10.

A consequence worth stating: the deep tables below are the **Qwen3.5-and-later
checkpoints plus DeepSeek**, because those are the models whose windows reach
128K without scaling anything. The earlier-generation models are measured at 8K,
which is inside every window in the fleet.

### 2.2 The three tests

**The width ladder** (`test_parallel_batched_forwarding*`) — a ~700-token prompt
at several batch widths and several points on the KV compression ladder. The
**width** axis: what a session costs, and what concurrency buys.

**The depth gate** (`long_context_gate`) — the same batched forward with a
prompt long enough to put 4K–128K tokens in the KV cache. The **depth** axis. It
takes an explicit `DepthTask`, which decides what the model is asked to do once
it has read the padding:

- `Coherence` — answer briefly about the material just read.
- `Rewrite` — reproduce a story that follows the padding, with a character
  renamed. This is the width ladder's own task, so a `Rewrite` depth row and a
  ladder row differ **only** in context length.

The task is a parameter rather than a constant because it moves the decode
number more than most engine changes do: on the flagship the draft head accepts
4.85 tokens a step on `Rewrite` against ~2.3 on `Coherence`, and decode rate is
steps/sec × accepted/step. **Rows from different tasks are not comparable**, and
naming the task at each call site is what keeps that from being rediscovered.

**The engine probe** (`candle-conversation/tests/kv_fragmentation.rs`) — the only
test here that constructs a `ConversationEngine`, so the only one that exercises
admission, per-turn projection, the persistence thread and KV compaction. The two
tests above drive the batched forward from a clean slate and cannot see any of
that. A probe runs one conversational workload rather than a ladder and reports
three gates, each read on its own:

- **Story** — the width ladder's rewrite, per session. The correctness gate.
- **Worst sustained VRAM efficiency** — how much of the ground below the KV
  arena frontier is actually holding KV, threshold 90%.
- **Weight uptake** — how much of the ground the frontier gives up the expert
  weights take back.

It produces no throughput figures, and its rows are not in the ladder TSVs.

### 2.3 Columns

| Column | Meaning |
|---|---|
| `prefill t/s` | Prompt tokens processed per second (the harness's `t/s (bulk)`) |
| `decode t/s` | Generated tokens per second, summed across the batch (`t/s (single)`) |
| `%Quantized` | Share of KV blocks that ended up in a quantized format |
| `Compress` | Float-equivalent bytes ÷ actual KV bytes |
| `Peak tokens` | Total tokens resident across all sessions |
| `Valid` | Output passed the config's validation mode |

> **The harness's column names are misleading and are renamed here.** `t/s
> (bulk)` is `prompt_tokens_per_sec` — prefill — and `t/s (single)` is
> `generate_tokens_per_sec` — decode. They are *not* a batched-vs-single
> distinction.

### 2.4 Instrumentation is not free

Building with `--features profile` costs **5–24% of decode** and 1–3% of
prefill: the spans fence a pipeline whose remaining cost is largely issue
overhead. Profiled builds are for **attribution** — which span dominates — and
uninstrumented builds are what the engine actually does. Every throughput figure
in this document is uninstrumented; quoting a profiled decode figure as
throughput understates it by up to a quarter.

### 2.5 The filler, and why it is not a tiled corpus

A depth measurement needs a prompt of a given length. The obvious way to build
one — tile a passage until it is long enough — would make a 128K prompt roughly
forty copies of the same text, and **two of the numbers in these tables are
compression ratios**. A cache holding forty copies of one passage compresses
like nothing real ever will.

So the filler (`batch_test/long_context.rs`) is assembled instead: paragraphs
from eight unrelated domains, walked in a rotating order so no two consecutive
paragraphs come from the same template, and *perturbed every cycle* — the names,
places and numbers differ each time a template comes round, so no two cycles
present the same token sequence. Unit tests pin all three properties.

It is still synthetic prose, and that is the honest caveat on every compression
number below: **these ratios are indicative of varied natural-language text, not
a measurement of any particular corpus.** Throughput figures are essentially
insensitive to content and carry no such caveat.

### 2.6 Prefill is chunked to the model's own width cap

Prefill honours `ManagedBatchedModel::prefill_width_cap` and submits a prompt in
cap-sized slices. A wave's transient tier is sized by its row count, so a
128K-token prompt submitted whole would ask for ~9.4 GB of transient against a
3.2 GB span.

The measurement consequence: **a chunked deep prefill figure is lower than the
same model submitting the same tokens in one wave.** The chunked path is the one
reported because it is the only configuration that spans every depth, and
because it is what the production scheduler does. Prompts below a model's cap —
every width-ladder row, and the 8K depth rows — take exactly one slice and are
unaffected.

### 2.7 The span must be sized before the weights load

The KV reservation is sized from a governor's balloon measurement of the card;
without one it falls back to a small test constant (3,170,893,824 B) with the
weight floor at its top, and a wave needing 872 MB of transient is refused on a
card with 72 GB free. `long_context_gate` calls `ensure_vram_governor` before
loading. At ladder depths nothing notices; at depth it is the difference between
a measurement and an error.

---

## 3. Results

**§3.2 through §3.7 are the RTX PRO 5000 72 GB reference machine** — the only
card with a depth sweep. The **RTX 3090 24 GB** gate sweep is **§3.8** and the
**RTX 4090 Mobile 16 GB** gate sweep **§3.9**, each with its own methodology note;
where the cards ran the same `test_parallel_batched_forwarding*` gate, those
sections set them side by side.

### 3.1 The models

| Model | Params | Weights | Attention | Window | Depths measured |
|---|---|---|---|---:|---|
| Qwen2-0.5B-Instruct | 0.5B dense | Q4_0 | full | 32,768 | 8K |
| Llama-2-7B-Chat | 7B dense | Q4_0 | full, **no GQA** | 4,096 | 4K |
| Llama-3.2-3B (Nidum) | 3B dense | Q4_K_M | full (GQA) | 8,192 | 8K |
| Qwen3-8B | 8B dense | Q6_K | full (GQA) | 40,960 | 8K |
| Qwen3-30B-A3B-2507 | 30B / 3B active | Q4_K_M | full (GQA), MoE | 262,144 | 8K |
| Qwen3.5-0.8B | 0.8B dense | Q6_K | 3:1 DeltaNet hybrid | 262,144 | 32K, 128K |
| Qwen3.5-9B | 9B dense | Q6_K | 3:1 DeltaNet hybrid | 262,144 | 32K, 128K |
| Qwen3.5-35B-A3B | 35B / 3B active | Q6_K | hybrid + MoE | 262,144 | 32K, 128K |
| Qwen3.6-35B-A3B | 35B / 3B active | Q6_K | hybrid + MoE | 262,144 | 32K, 128K |
| Qwen3.8-27B | 27B dense | Q6_K/Q8 | hybrid | 262,144 | 32K, 128K |
| Qwen3.8-Flash-Next | 180B / 6B active ¹ | Q4_KOEXP | hybrid + **QSA** + MoE | 262,144 | 32K, 128K |
| DeepSeek-V4-Flash-0731 | 284B / 13B active | MXFP4_KO | native-sparse | 1,048,576 | none — §4 |

¹ Qwen3.8-Flash-Next's published size is **125B trunk + 51B n-gram embedding
table + 4B MTP head = 180B, 6B activated** (the Qwen model card, and
`docs/archived/qwen38_flash_next.md`); of the trunk, ~121B are the 512 experts.
Earlier revisions of this document gave "250B / 13B active", which was wrong, and
the sweep TSVs up to 2026-09-15 still carry that label in their `label` column.

The weights are the 72 GB card's. The RTX 4090 Mobile runs Qwen3.8-Flash-Next
from a **Q2_KO-expert** artifact instead (§3.9); every other model loads the same
checkpoint on every card. "Depths measured" is the 72 GB card's alone — the other
two have no depth sweep.

### 3.2 Depth: 32K → 128K, one context

Every row valid. `Coherence` task, 64 generated tokens.

| Model | Mode | Prefill 32K | Prefill 128K | Decode 32K | Decode 128K | Compress 128K |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next** | BF16 | 1,282.4 | **1,269.5** | 26.5 | **29.5** | — |
| | C10 | 1,341.2 | 1,277.1 | 25.6 | 23.8 | 6.97× |
| Qwen3.5-0.8B | BF16 | 7,309.1 | 1,859.0 | 157.3 | **135.6** | — |
| | C10 | 7,149.6 | 1,913.1 | 158.5 | 132.1 | 4.63× |
| Qwen3.5-9B | BF16 | 1,987.2 | 592.3 | 75.0 | 35.7 | — |
| | C10 | 2,000.8 | 603.2 | 52.1 | 18.5 | 6.37× |
| Qwen3.5-35B-A3B | BF16 | 1,692.7 | 441.4 | 61.5 | 28.7 | — |
| | C10 | 1,641.4 | 449.8 | 42.8 | 14.1 | **7.63×** |
| Qwen3.6-35B-A3B | BF16 | 1,692.8 | 443.4 | 56.8 | 27.9 | — |
| | C10 | 1,637.4 | 456.4 | 41.4 | 14.5 | 7.17× |
| Qwen3.8-27B | BF16 | 691.7 | 192.0 | 30.6 | 17.7 | — |
| | C10 | 656.7 | 196.7 | 26.0 | 11.5 | 5.44× |

C5 was also measured at 32K on every model: 2.85×–4.20× compression at 1–22% of
decode. Full rows are in the provenance file.

### 3.3 What survives a 4× depth increase

Fraction of BF16 throughput kept from 32K to 128K.

| Model | Attention | Prefill kept | Decode kept |
|---|---|---:|---:|
| **Qwen3.8-Flash-Next** | hybrid + QSA + MoE | **99%** | **111%** |
| Qwen3.8-27B | hybrid | 28% | 58% |
| Qwen3.6-35B-A3B | hybrid + MoE | 26% | 49% |
| Qwen3.5-9B | hybrid | 30% | 48% |
| Qwen3.5-35B-A3B | hybrid + MoE | 26% | 47% |
| Qwen3.5-0.8B | hybrid | 25% | **86%** |

**Flash-Next is flat.** Prefill 1,282 → 1,270 t/s and decode 26.5 → **29.5** t/s
across a 4× depth increase — decode is *higher* at 128K than at 32K. This is the
architecture's central claim and it is what the rest of the fleet's 26–30%
prefill retention exists to be read against. QSA attends a selected subset rather
than the full prefix, so the depth-dependent term the other models pay is not
merely reduced, it is absent within measurement noise.

#### The gap is a checkpoint property, not an engine setting

The table above invites the reading that the engine treats Flash-Next better
than its siblings. It does not: **they run the same attention kernel, and at
shallow depth it costs them the same.** Profiled per call (`--features profile`,
`prefill:kernel`, the kernel the speculative verify step runs):

| | shallow | deep | growth |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next | 0.382 ms @32K | 0.405 ms @128K | **+6%** |
| Qwen3.5-35B-A3B | 0.394 ms @8K | 1.463 ms @32K | **3.71× for 4× depth** |

Same kernel, same starting cost, and then one is flat and the other is linear in
prefix. The difference is that QSA caps the attended set at `top_k + ratio − 1`
= 2,051 positions however deep the cache is, and the hybrids read all of it.

The second half of the gap is how much of a step that term occupies:

| | attention kernel | share of `spec:verify` |
|---|---:|---:|
| Qwen3.5-35B-A3B @32K | 418.3 of 729.8 ms | **57%** |
| Qwen3.8-Flash-Next @32K | 129.0 of 2,007.0 ms | **6.4%** |
| Qwen3.8-Flash-Next @128K | 131.6 of 2,079.5 ms | 6.3% |

Flash-Next's step is dominated by depth-independent MoE weight work — 10 routed
experts and a shared one out of 512, in each of 48 layers — so the depth term is a twentieth of it before QSA bounds it
at all. On the 35B the same term is the majority of the step *and* growing.

**This cannot be enabled on the other models.** The indexer is trained weights
carried in the checkpoint — `{layer}.indexer.{q_proj,k_proj,q_norm,k_norm}.weight`,
with its geometry in the GGUF metadata — and only Flash-Next's has them. The
qwen35 lineage passes `qsa: None` at every call site. So the retention column
separates checkpoints that ship an indexer from checkpoints that do not, and no
configuration change moves a model between those groups.

**Qwen3.5-0.8B keeps 86% of decode** — far above its 9B–35B siblings. At 0.8B
the per-token cost is dominated by weight-bound work and by WDDM launch
overhead, both depth-independent, so the growing attention term is a small share
of a small total. Prefill, which has no such floor, retains 25% like everything
else. **Retention is confounded by model size and is only an architectural
statement between size-matched models** — which is why the flat row above is
stated in absolute terms rather than as a ratio.

### 3.4 The shallow group

Every model at a depth inside its own window, one context, `Coherence`, all
rows valid.

| Model | Depth | Mode | Prefill t/s | Decode t/s | Compress |
|---|---:|---|---:|---:|---:|
| Qwen2-0.5B | 8K | BF16 / C5 / C10 | 26,516.7 / 26,312.6 / 25,674.0 | 142.1 / 137.5 / 138.3 | *none — §4* |
| Llama-3.2-3B | 8K | BF16 / C5 / C10 | 6,195.9 / 5,963.8 / 5,936.8 | 69.9 / 69.5 / 66.1 | 3.29× / 4.62× |
| Llama-2-7B | 4K | BF16 / C5 / C10 | 3,684.7 / 3,852.0 / 3,851.1 | 47.1 / 47.2 / 44.1 | 3.33× / 4.81× |
| Qwen3-30B-A3B | 8K | BF16 / C5 / C10 | 3,444.7 / 3,388.6 / 3,367.6 | 51.2 / 47.5 / 47.2 | 3.70× / 5.94× |
| Qwen3-8B | 8K | BF16 / C5 / C10 | 3,235.1 / 3,213.9 / 3,212.7 | 42.8 / 40.1 / 41.8 | 3.71× / 6.31× |

At this depth compression is close to free on every model — 0–8% of decode for
3.3×–6.3× — which is the contrast that makes §3.5 worth stating separately.

### 3.5 Compression costs more decode as the cache grows

The C10 rung's decode penalty **roughly doubles** from 32K to 128K, on every
model that has enough KV for it to matter. Prefill is unaffected throughout — at
128K, all six models prefill marginally *faster* under C10 than BF16.

| Model | C10 decode cost @8K | @32K | @128K |
|---|---:|---:|---:|
| Qwen2-0.5B | −3% | — | — |
| Qwen3-8B | −2% | — | — |
| Qwen3-30B-A3B | −8% | — | — |
| Qwen3.5-0.8B | — | +1% | −3% |
| **Qwen3.8-Flash-Next** | — | **−3%** | **−19%** |
| Qwen3.8-27B | — | −15% | −35% |
| Qwen3.6-35B-A3B | — | −27% | −48% |
| Qwen3.5-9B | — | −31% | −48% |
| Qwen3.5-35B-A3B | — | −30% | −51% |

The effect scales with the model's KV volume: the 0.8B, which has the least KV
per token, barely shows it; the 9B and the two 35B MoEs, which have the most,
lose about half their decode at 128K. Flash-Next is again the outlier, paying
19% where its architectural siblings pay ~50%.

#### The per-read cost is a constant; the number of reads is not

The column above reads as though compression gets more expensive with depth. It
does not. Profiling the attention kernel under each mode gives a **flat
multiplier**:

| Qwen3.5-35B-A3B, `prefill:kernel` | BF16 | C10 | ratio |
|---|---:|---:|---:|
| @8K | 117.1 ms | 227.0 ms | **1.94×** |
| @32K | 418.3 ms | 825.1 ms | **1.97×** |

A quantized KV position costs about twice a BF16 one to read, at every depth.
What grows is how many positions are read, and the product is what the decode
column shows. The same multiplier on Flash-Next is 1.13× at 32K and 1.36× at
128K — smaller because QSA has already bounded the reads, and because the term
is 6% of its step rather than 57%.

That the multiplier is constant is why this is not a codec defect. Measured in
isolation on the decode A/B harness at head_dim 256, every C10 format lands
between 1.26× and 1.71× of BF16 and the adaptive mix at 1.6×, with no format
dominating; `ncu` puts the kernel at 12% DRAM, 39% L1/TEX and 37% compute under
C10 against 58% DRAM under BF16 — compression removes the bandwidth it promises
to remove and the path is latency-bound either way.

**The practical consequence: C5 is the deep-context operating point, not C10.**
At 32K, C5 costs 6% of decode on the 35Bs for 3.8×, where C10 costs 30% for
7.5× — the last 2× of compression is bought at five times the decode price, and
that ratio worsens with depth because the read count does. C10 remains the right
choice where cache footprint is the binding constraint rather than latency.

This is guidance rather than a workaround: the models it applies to cannot
acquire selection (§3.3), so the read count is not reducible for them and the
per-read cost is the only term left to choose.

Compression itself does not degrade with depth. Several models compress slightly
*better* at 128K (Qwen3.5-35B-A3B 7.48× → 7.63×; Qwen3.5-9B 6.29× → 6.37×;
Qwen3.6-35B 7.01× → 7.17×), and every model that compresses at all reaches
**100% of KV blocks quantized** at every depth.

### 3.6 Qwen3.8-Flash-Next in detail

The flagship's cost is independent of context length over the measured range.
One context, uninstrumented, speculative decode (draft budget 4).

**`Coherence`**, five depths, 32 generated tokens:

| depth | prompt tokens | prefill t/s | decode t/s | accepted/step |
|---:|---:|---:|---:|---:|
| 8K | 7,990 | 1,098.8 | 24.7 | 2.07 |
| 16K | 16,020 | 1,368.9 | 28.3 | 2.21 |
| 32K | 32,128 | 1,342.8 | 24.6 | 1.94 |
| 64K | 64,249 | 1,291.8 | 26.3 | 2.07 |
| 128K | 128,433 | 1,257.5 | 26.4 | 2.21 |

**`Rewrite`** — the width ladder's own task, 64 generated tokens, so these rows
and the ladder's differ **only** in context length:

| depth | prompt tokens | prefill t/s | decode t/s | accepted/step |
|---:|---:|---:|---:|---:|
| 8K | 8,454 | 1,108.1 | 62.3 | 4.85 |
| 16K | 16,484 | 1,347.9 | 66.6 | 4.85 |
| 32K | 32,592 | 1,307.7 | 57.8 | 4.85 |
| 64K | 64,713 | 1,284.3 | 63.5 | 4.85 |
| 128K | 128,897 | 1,258.7 | 53.9 | 4.85 |

Three things this pair establishes.

**Neither curve has a depth term the measurement can see.** Prefill *rises* from
8K to 16K and then holds within 8% to 128K on both tasks — the 8K row is the
slowest, not the fastest. Decode over the same 16× range keeps 87% on `Rewrite`
(62.3 → 53.9) and is flat within noise on `Coherence` (24.7 → 26.4).

**Acceptance is constant at 4.85 on every `Rewrite` row**, which is what makes
those rows comparable to each other and to the width ladder: the fixture holds
the task fixed, so the drafter's hit rate does not drift with depth and the
decode column measures the engine rather than the task. On `Coherence` it sits
near 2.1 with no trend.

**The two tasks differ by ~2.3× at equal depth** — 53.9 against 26.4 t/s at
128K — entirely because of what is being asked. A rewrite is largely copying
tokens the drafter can see; free continuation it must guess. This is the
clearest illustration of why the task is a parameter and why rows from different
tasks are never set side by side.

The deepest `Rewrite` row is also a retrieval result: at 128,897 tokens the story
sits behind ~128K tokens of unrelated padding, and the rename still validates.

#### Width

The flagship's ladder on the RTX PRO 5000 (2026-10-10, build `21c36efbd`),
aggregate across the batch, speculative decode on:

| Mode | Ctx | Prefill t/s | Decode t/s | Compress | Peak tokens |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 (cold) | 3,719.5 | 274.2 | — | 905 |
| BF16 | 1 (warm) | 4,688.0 | 295.2 | — | 905 |
| BF16 | 4 | 7,024.1 | 828.3 | — | 3,662 |
| BF16 | 8 | 6,938.7 | 1,008.2 | — | 7,284 |
| BF16 | 16 | 6,857.5 | 1,163.9 | — | 14,554 |
| C0 | 2 | 5,648.0 | 462.2 | 2.22× | 1,850 |
| C5 | 2 | 5,657.6 | 455.5 | 4.15× | 1,850 |
| C5 | 8 | 7,002.6 | 926.8 | 4.15× | 7,284 |
| C8 | 2 | 5,648.8 | 453.1 | 5.53× | 1,850 |
| C10 | 2 | 5,650.2 | 441.9 | 7.13× | 1,466 |
| C10 | 8 | 7,000.8 | 885.8 | 7.13× | 5,748 |

Decode returns **3.4× single-session throughput at 8 contexts** (295.2 → 1,008.2)
and gains another 15% from 8 to 16; prefill is flat from ×4 to ×16 (6,858–7,024),
so width buys decode, not prefill. The ladder's C0→C10 span costs **4% of decode
at ×2** (462.2 → 441.9) for **3.2× more compression**, and C10 ×8 holds 88% of
BF16 ×8 decode (885.8 against 1,008.2) at 7.13×. The ×1 rows are the gate's story
rewrite, an easy task for the drafter; Strata's free-text requests decode 156.6 t/s
through the engine at 4K (§6.5).

### 3.7 Width across the fleet

BF16 at one context against each model's widest measured point. Prompts are
~700 tokens, so this axis is unaffected by context windows. Every cell is the
latest sweep, 2026-10-10 at build `21c36efbd`
(`results/sweep_rtx_pro_5000_72gb_2026-10-10_main.md`).

| Model | ctx=1 prefill / decode | widest measured | prefill / decode |
|---|---|---|---|
| Qwen2-0.5B | 54,944.4 / 560.3 | ×60 (F16) | 139,750.4 / 6,558.3 |
| Qwen3.5-0.8B | 43,068.5 / 347.1 | ×256 (C8) | 77,753.9 / 10,805.3 |
| Llama-3.2-3B | 7,312.9 / 144.3 (F16) ¹ | ×10 (C8) | 25,512.0 / 1,152.2 |
| Qwen3-30B-A3B | 10,446.8 / 147.0 (warm) | ×20 (Q8_0) | 16,668.1 / 874.3 |
| Qwen3.5-35B-A3B | 11,034.6 / 230.2 | ×64 (C10) | 18,237.2 / 2,869.9 |
| Qwen3.6-35B-A3B | 10,582.4 / 476.6 | ×64 (C10) | 17,642.4 / 2,761.0 |
| Qwen3-8B | 9,717.2 / 102.2 | ×10 (C8) | 10,962.5 / 622.3 |
| Llama-2-7B | 7,974.8 / 163.4 | ×48 | 11,353.2 / 1,638.4 |
| Qwen3.5-9B | 10,192.3 / 199.8 | ×20 (C8) | 11,620.1 / 1,607.7 |
| Qwen3.8-27B | 3,282.2 / 75.4 | ×40 (C10) | 3,420.7 / 734.3 |
| Qwen3.8-Flash-Next | 4,688.0 / 295.2 (warm) ² | ×16 | 6,857.5 / 1,163.9 |
| DeepSeek-V4-Flash | 319.0 / 18.9 (warm) | ×16 | 1,177.1 / 114.4 |

¹ The ladder's early rows run slow: from Q4_0 ×4 on every row prefills at
20,700–25,600 t/s, and the C0–C7 ×1 rows decode at 188–192 t/s.

² Speculative, at a draft ceiling of 4 on the gate's story rewrite, an easy task for
the drafter. On Strata's free-text requests the engine decodes 156.6 t/s at 4K on the
same build (§6.5).

Two shapes appear here. **Prefill saturates early** on every model — most are
within 20% of their ×1 rate by ×4, and the 35Bs gain about two thirds from ×1 to
×64 — while **decode scales nearly linearly with width** until it too flattens.
The 35B MoEs reach 2,761–2,870 t/s aggregate decode at 64 concurrent sessions
against 230–477 at one, a 6–12× return on concurrency.

DeepSeek-V4-Flash is the exception whose prefill is still climbing at ×16
(319 → 1,177 t/s), having not yet reached the saturation the others hit by ×4.

The ladders are not run at a common set of widths, so this table gives each
model's own widest point rather than a shared column.

### 3.8 RTX 3090 24 GB — the width gate sweep

Every gate and engine probe on **2026-10-10**, on the RTX 3090 (§1), build
`c375834c8`, of the same `test_parallel_batched_forwarding*` gates that produce
§3.6 *Width* and §3.7 — one `cargo test` invocation per model so exactly one was
ever resident, `--release --features cuda`, with the daemon stopped and the card
at its idle floor before each. This is a **width / throughput** sweep: each gate
runs its own ladder of KV modes and context counts at a fixed ~700-token prompt,
so it measures aggregate throughput and the compression ladder, **not** depth —
the `long_context_*` and `profile_*` depth gates were not run here. Numbers are
single measurements, so the 1–4 % noise floor (§5) applies to each cell alone.
Every gate passed: no session in any of the 213 rows failed its check. Two engine
probes failed their efficiency gate (below). The run and its comparison with the
2026-10-07 sweep (`7aecbeab3`) are in `results/sweep_rtx_3090_24gb_2026-10-10.md`;
the 2026-10-07 run, its earlier runs that day and the commits before it are in
`results/sweep_rtx_3090_24gb_2026-10-07.md`.

**Thirteen gates — ten of the fleet's models, the two AntiLoop+StyleTune
hybrids** (the production 3.6-35B npcd serves), **and Qwen3.8-Flash-Next**, from
the same Q2_KO-expert artifact the 4090 Mobile runs, including the ×16 rung that
card skips. The 284B DeepSeek was not run. Two card-specific notes carry into
these rows: **Llama-3.2-3B ran without flash-attn** (the sweep builds without
`--features flash-attn`, so the model's `#[cfg(not(feature = "flash-attn"))]`
path runs), and **Qwen2-0.5B's gate ladder has no compressed rung** — it runs
F32, BF16 and F16 only, so its compression column is empty by construction. The
plain Qwen3.6-35B-A3B gate loads at `Int8Mode::auto`, which is Precision on this
card; the hybrid's Performance gate prices the other posture.

**Summary** — best prefill, best decode, and the best validated compression over
each model's ladder in this run.

| Model | best prefill t/s | best decode t/s | best compression (mode) |
|---|---:|---:|---|
| Qwen2-0.5B | 72,012.4 | 7,258.1 | — (no ladder) |
| Qwen3.5-0.8B | 44,981.3 | 4,177.8 | 4.38× (C10) |
| Llama-3.2-3B † | 10,920.1 | 789.1 | 4.35× (C10) |
| Llama-2-7B | 5,737.5 | 934.9 | 3.56× (Q4_0) |
| Qwen3-8B | 5,285.5 | 383.8 | 5.82× (C10) |
| Qwen3.5-9B | 6,735.3 | 926.2 | 5.57× (C10) |
| Qwen3.8-27B | 2,067.0 | 358.7 | 5.05× (C10) |
| Qwen3-30B-A3B ‡ | 6,821.5 | 400.5 | 5.45× (C10) |
| Qwen3.5-35B-A3B | 10,041.6 | 1,228.8 | 7.03× (C10) |
| Qwen3.6-35B-A3B | 7,184.4 | 1,346.3 | 6.43× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (auto/Precision) | 8,348.6 | 1,166.7 | 6.47× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (Performance) | 10,221.6 | 1,269.0 | 6.45× (C10) |
| Qwen3.8-Flash-Next (Q2_KO experts) | 2,301.9 | 543.8 | 5.83× (C10) |

† ran on the no-flash-attn path (above).
‡ best over validated rows. The gate's unvalidated Q4_0 rows reach 6,709.1 prefill
(×4) and 390.4 decode (×20) here.

**Against 2026-10-07 (`7aecbeab3`).** Best prefill rose on every model, by 36%
(Qwen3-30B-A3B) to 101% (Qwen3.5-9B). Best decode rose 9–18% on the 35B MoEs and
Flash-Next, and moved −2.1% to +8.3% on the rest, inside the single-run noise.
Compression is unchanged on every row. Per-gate deltas are in
`results/sweep_rtx_3090_24gb_2026-10-10.md`.

**Against the 72 GB card, on the models both ran.** The comparison holds only on
the width gate, and only loosely: the 3090's ladders stop at a narrower widest
context (×16 on the 35Bs, where the 72 GB reached ×64), because 24 GB caps how
many concurrent sessions fit. On the 35B MoEs the 72 GB card prefills about
**1.9×** the 3090 at one context (~11,100 vs ~5,700 t/s for the 3.5-35B's C-mode
rows), and its aggregate decode at ×64 is **2.3×** the 3090's at ×16 (2,869.9
vs 1,228.8 t/s on the 3.5-35B, I8) — that gap being the extra 48 sessions the
bigger card holds. The 3090 runs the current C10 calibration, which compresses
the two 35Bs 7.03× and 6.43–6.47×. The sm_86 and PCIe-3.0 traps (§1) sit under
the 3090's absolute rates.

**Full ladders.** Each model's complete gate ladder — every KV mode and context
the gate ran, exactly as the run logs printed them. `Valid` is the per-config
reproduction check (`✓`, or `-` for a mode not validated for reproduction);
`int8` is the loader's int8 posture (`prec`/`perf`/`off`).

#### Qwen2-0.5B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | off | no | 1 | - | 33396.7 | 506.0 | - | - | 308 |
| BF16 | off | yes | 1 | - | 33891.2 | 529.1 | - | - | 308 |
| F16 | off | yes | 1 | - | 33892.3 | 478.6 | - | - | 308 |
| F16 | off | yes | 4 | - | 62291.3 | 1565.5 | - | - | 1272 |
| F16 | off | yes | 60 | - | 72012.4 | 7240.6 | - | - | 18612 |
| BF16 | off | yes | 60 | - | 70686.8 | 7258.1 | - | - | 18612 |

#### Qwen3.5-0.8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 32534.2 | 257.6 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 32462.0 | 273.9 | - | - | 659 |
| BF16 | prec | yes | 16 | ✓ | 43998.0 | 3072.3 | - | - | 10618 |
| Q8_0 | prec | yes | 4 | ✓ | 42489.4 | 1180.5 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 2 | ✓ | 40576.3 | 647.4 | 100.0% | 1.87x | 1358 |
| C1 | prec | yes | 2 | ✓ | 40737.7 | 648.2 | 100.0% | 2.13x | 1358 |
| C2 | prec | yes | 2 | ✓ | 24221.4 | 379.7 | 100.0% | 2.33x | 1358 |
| C3 | prec | yes | 2 | ✓ | 22229.3 | 364.2 | 100.0% | 2.82x | 1358 |
| C4 | prec | yes | 2 | ✓ | 20682.0 | 346.2 | 100.0% | 2.66x | 1358 |
| C5 | prec | yes | 2 | ✓ | 19866.4 | 331.6 | 100.0% | 2.93x | 1358 |
| C6 | prec | yes | 2 | ✓ | 18485.0 | 339.1 | 100.0% | 3.23x | 1358 |
| C7 | prec | yes | 2 | ✓ | 19440.7 | 346.8 | 100.0% | 3.59x | 1358 |
| C8 | prec | yes | 32 | ✓ | 42982.2 | 4177.8 | 100.0% | 3.98x | 21202 |
| C9 | prec | yes | 5 | ✓ | 44767.3 | 1418.7 | 100.0% | 4.16x | 3337 |
| C10 | prec | yes | 10 | ✓ | 44981.3 | 2231.0 | 100.0% | 4.38x | 6640 |

C2–C7 ×2 ran in a light-load clock window (every kernel 1.4–2× slower, recovering
at the ×32 row); C0–C1 show the rate: 647–648 t/s. The 2026-10-07 sweep showed the
same window on the same rows.

#### Llama-3.2-3B (no flash-attn)

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | prec | no | 1 | ✓ | 10510.0 | 184.7 | - | - | 654 |
| F16 | prec | yes | 1 | ✓ | 10508.9 | 183.4 | - | - | 654 |
| F16 | prec | yes | 4 | ✓ | 10920.1 | 559.3 | - | - | 2658 |
| R16 | prec | yes | 1 | ✓ | 10676.7 | 163.3 | 0.0% | - | 654 |
| Q8_0 | prec | yes | 1 | ✓ | 10643.7 | 169.2 | 100.0% | 1.88x | 654 |
| Q8_Q4 | prec | yes | 1 | ✓ | 10622.6 | 165.4 | 100.0% | 2.29x | 654 |
| BF16 | prec | yes | 4 | ✓ | 10769.8 | 555.2 | - | - | 2658 |
| Q8_1 | prec | yes | 4 | ✓ | 10770.3 | 458.9 | 100.0% | 1.78x | 2658 |
| Q8_KS | prec | yes | 4 | ✓ | 10708.4 | 446.9 | 100.0% | 1.78x | 2658 |
| Q8_Q4 | prec | yes | 4 | ✓ | 10714.3 | 462.7 | 100.0% | 2.29x | 2658 |
| Q4_0 | prec | yes | 4 | - | 10729.3 | 484.0 | 100.0% | 3.56x | 2658 |
| Q4_1 | prec | yes | 4 | - | 10694.2 | 466.8 | 100.0% | 3.20x | 2658 |
| Q4_KS | prec | yes | 4 | - | 10674.7 | 478.1 | 100.0% | 3.20x | 2658 |
| C0 | prec | yes | 1 | ✓ | 10550.7 | 164.0 | 100.0% | 1.88x | 654 |
| C1 | prec | yes | 1 | ✓ | 10299.0 | 165.6 | 100.0% | 2.24x | 654 |
| C2 | prec | yes | 1 | ✓ | 10319.0 | 167.0 | 100.0% | 2.45x | 654 |
| C3 | prec | yes | 1 | ✓ | 10580.4 | 166.3 | 100.0% | 3.02x | 654 |
| C4 | prec | yes | 1 | ✓ | 10518.0 | 165.4 | 100.0% | 2.80x | 654 |
| C5 | prec | yes | 1 | ✓ | 10584.7 | 165.8 | 100.0% | 3.08x | 654 |
| C6 | prec | yes | 1 | ✓ | 10523.8 | 165.4 | 100.0% | 3.53x | 654 |
| C7 | prec | yes | 1 | ✓ | 10562.2 | 166.0 | 100.0% | 3.73x | 654 |
| C8 | prec | yes | 10 | ✓ | 9966.1 | 789.1 | 100.0% | 3.92x | 6590 |
| C9 | prec | yes | 10 | ✓ | 9908.2 | 772.2 | 100.0% | 4.16x | 6590 |
| C10 | prec | yes | 5 | ✓ | 10618.2 | 552.8 | 100.0% | 4.35x | 3312 |

#### Llama-2-7B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | perf | no | 1 | - | 4960.4 | 118.5 | - | - | 283 |
| F16 | perf | yes | 1 | - | 4974.6 | 117.9 | - | - | 283 |
| F16 | perf | yes | 4 | - | 5737.5 | 355.9 | - | - | 1176 |
| F16 | perf | yes | 8 | - | 5511.0 | 536.6 | - | - | 2316 |
| BF16 | perf | yes | 1 | - | 4966.8 | 120.6 | - | - | 283 |
| BF16 | perf | yes | 8 | - | 5479.6 | 543.3 | - | - | 2316 |
| BF16 | perf | yes | 16 | - | 4869.2 | 741.0 | - | - | 4624 |
| BF16 | perf | yes | 48 | - | 4600.2 | 934.9 | - | - | 13796 |
| Q8_0 | perf | yes | 32 | - | 4309.8 | 605.9 | 100.0% | 1.88x | 9220 |
| Q4_0 | perf | yes | 32 | - | 4284.1 | 635.2 | 100.0% | 3.56x | 9220 |

#### Qwen3-8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | no | 1 | ✓ | 4914.3 | 85.6 | - | - | 636 |
| F16 | perf | yes | 1 | ✓ | 4913.4 | 84.9 | - | - | 636 |
| F16 | perf | yes | 2 | ✓ | 5285.5 | 146.3 | - | - | 1312 |
| BF16 | perf | yes | 4 | ✓ | 5168.3 | 259.8 | - | - | 2586 |
| Q8_0 | perf | yes | 4 | ✓ | 5139.6 | 230.7 | 100.0% | 1.88x | 2586 |
| C0 | perf | yes | 1 | ✓ | 4865.3 | 78.1 | 100.0% | 1.91x | 636 |
| C1 | perf | yes | 1 | ✓ | 4772.8 | 78.8 | 100.0% | 2.56x | 636 |
| C2 | perf | yes | 1 | ✓ | 4892.7 | 79.4 | 100.0% | 2.79x | 636 |
| C3 | perf | yes | 1 | ✓ | 4871.0 | 78.7 | 100.0% | 3.28x | 636 |
| C4 | perf | yes | 1 | ✓ | 4838.8 | 78.5 | 100.0% | 3.21x | 636 |
| C5 | perf | yes | 1 | ✓ | 4896.3 | 78.5 | 100.0% | 3.40x | 636 |
| C6 | perf | yes | 1 | ✓ | 4909.1 | 78.0 | 100.0% | 4.04x | 636 |
| C7 | perf | yes | 1 | ✓ | 4791.3 | 78.3 | 100.0% | 4.31x | 636 |
| C8 | perf | yes | 10 | ✓ | 4664.1 | 383.8 | 100.0% | 4.79x | 6410 |
| C9 | perf | yes | 5 | ✓ | 4987.3 | 270.6 | 100.0% | 5.17x | 3222 |
| C10 | perf | yes | 5 | ✓ | 5001.8 | 269.8 | 100.0% | 5.82x | 3222 |

#### Qwen3.5-9B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 5788.8 | 144.8 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 5804.9 | 146.1 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 6735.3 | 549.1 | - | - | 2678 |
| Q8_0 | prec | yes | 4 | ✓ | 6716.1 | 543.1 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 1 | ✓ | 5782.3 | 157.7 | 100.0% | 2.20x | 659 |
| C1 | prec | yes | 1 | ✓ | 5808.7 | 158.4 | 100.0% | 2.43x | 659 |
| C2 | prec | yes | 1 | ✓ | 5795.1 | 158.2 | 100.0% | 2.72x | 659 |
| C3 | prec | yes | 1 | ✓ | 5780.3 | 158.7 | 100.0% | 3.19x | 659 |
| C4 | prec | yes | 1 | ✓ | 5800.9 | 158.0 | 100.0% | 3.26x | 659 |
| C5 | prec | yes | 1 | ✓ | 5780.7 | 158.5 | 100.0% | 3.60x | 659 |
| C6 | prec | yes | 1 | ✓ | 5789.7 | 157.4 | 100.0% | 4.04x | 659 |
| C7 | prec | yes | 1 | ✓ | 5777.4 | 157.6 | 100.0% | 4.23x | 659 |
| C8 | prec | yes | 20 | ✓ | 6435.7 | 926.2 | 100.0% | 4.71x | 13278 |
| C9 | prec | yes | 5 | ✓ | 6687.4 | 588.9 | 100.0% | 5.17x | 3337 |
| C10 | prec | yes | 10 | ✓ | 6395.6 | 880.6 | 100.0% | 5.57x | 6640 |

#### Qwen3.8-27B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1854.7 | 70.8 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 2067.0 | 263.1 | - | - | 2678 |
| Q8_0 | perf | yes | 4 | ✓ | 2061.5 | 250.2 | 100.0% | 1.88x | 2678 |
| C0 | perf | yes | 1 | ✓ | 1851.7 | 81.0 | 100.0% | 2.17x | 659 |
| C1 | perf | yes | 1 | ✓ | 1852.2 | 80.4 | 100.0% | 2.33x | 659 |
| C2 | perf | yes | 1 | ✓ | 1853.6 | 82.2 | 100.0% | 2.57x | 659 |
| C3 | perf | yes | 1 | ✓ | 1855.3 | 81.3 | 100.0% | 3.10x | 659 |
| C4 | perf | yes | 1 | ✓ | 1846.2 | 81.9 | 100.0% | 3.03x | 659 |
| C5 | perf | yes | 1 | ✓ | 1856.0 | 80.6 | 100.0% | 3.42x | 659 |
| C6 | perf | yes | 1 | ✓ | 1847.3 | 81.3 | 100.0% | 3.81x | 659 |
| C7 | perf | yes | 1 | ✓ | 1845.9 | 81.3 | 100.0% | 3.96x | 659 |
| C8 | perf | yes | 20 | ✓ | 1999.7 | 358.7 | 100.0% | 4.39x | 13278 |
| C9 | perf | yes | 5 | ✓ | 2004.0 | 288.9 | 100.0% | 4.77x | 3337 |
| C10 | perf | yes | 10 | ✓ | 2007.6 | 341.6 | 100.0% | 5.05x | 6640 |

C10 ×10 read 390.1 t/s in the 2026-10-05 sweep (`5776799ac`) and has read 307–349 on
every build since.

#### Qwen3-30B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | perf | yes | 1 | ✓ | 2715.8 | 55.4 | - | - | 626 |
| BF16 | perf | yes | 1 | ✓ | 4117.2 | 69.9 | - | - | 626 |
| BF16 | perf | yes | 10 | ✓ | 6821.5 | 400.5 | - | - | 6310 |
| Q8_0 | perf | yes | 20 | ✓ | 6671.8 | 377.7 | 100.0% | 1.88x | 12620 |
| Q4_0 | perf | yes | 4 | - | 6709.1 | 193.7 | 100.0% | 3.56x | 2546 |
| C0 | perf | yes | 2 | ✓ | 6395.6 | 119.7 | 100.0% | 2.03x | 1292 |
| C1 | perf | yes | 2 | ✓ | 6402.1 | 120.8 | 100.0% | 2.49x | 1292 |
| C2 | perf | yes | 2 | ✓ | 6369.1 | 120.7 | 100.0% | 2.70x | 1292 |
| C3 | perf | yes | 2 | ✓ | 6362.2 | 119.5 | 100.0% | 3.20x | 1292 |
| C4 | perf | yes | 2 | ✓ | 6398.5 | 119.4 | 100.0% | 3.10x | 1292 |
| C5 | perf | yes | 2 | ✓ | 6372.7 | 119.2 | 100.0% | 3.35x | 1292 |
| C5 | perf | yes | 8 | ✓ | 6655.0 | 310.7 | 100.0% | 3.35x | 5052 |
| C6 | perf | yes | 2 | ✓ | 6371.8 | 121.2 | 100.0% | 3.86x | 1292 |
| C7 | perf | yes | 2 | ✓ | 6397.4 | 121.9 | 100.0% | 4.06x | 1292 |
| C8 | perf | yes | 2 | ✓ | 6397.8 | 122.0 | 100.0% | 4.57x | 1292 |
| C9 | perf | yes | 2 | ✓ | 6408.8 | 122.0 | 100.0% | 5.01x | 1292 |
| C10 | perf | yes | 2 | ✓ | 6419.6 | 120.9 | 100.0% | 5.45x | 1292 |
| BF16 | perf | yes | 1 | ✓ | 4370.4 | 72.7 | - | - | 626 |
| Q4_0 | perf | yes | 20 | - | 6662.5 | 390.4 | 100.0% | 3.56x | 12620 |

#### Qwen3.5-35B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1610.8 | 78.1 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 7094.1 | 618.4 | - | - | 2678 |
| Q8_0 | perf | yes | 2 | ✓ | 8775.2 | 339.8 | 100.0% | 1.88x | 1358 |
| C0 | perf | yes | 1 | ✓ | 5710.6 | 201.6 | 100.0% | 2.21x | 659 |
| C1 | perf | yes | 1 | ✓ | 5716.2 | 203.0 | 100.0% | 2.85x | 659 |
| C2 | perf | yes | 1 | ✓ | 5722.1 | 200.9 | 100.0% | 3.24x | 659 |
| C3 | perf | yes | 1 | ✓ | 5683.2 | 200.3 | 100.0% | 3.39x | 659 |
| C4 | perf | yes | 1 | ✓ | 5699.4 | 199.5 | 100.0% | 3.73x | 659 |
| C5 | perf | yes | 1 | ✓ | 5704.8 | 127.0 | 100.0% | 4.03x | 659 |
| C6 | perf | yes | 1 | ✓ | 4152.1 | 199.4 | 100.0% | 4.62x | 659 |
| C7 | perf | yes | 1 | ✓ | 5705.2 | 200.2 | 100.0% | 4.90x | 659 |
| C8 | perf | yes | 5 | ✓ | 10041.6 | 674.4 | 100.0% | 5.52x | 3337 |
| C9 | perf | yes | 2 | ✓ | 8806.0 | 351.4 | 100.0% | 6.23x | 1358 |
| C10 | perf | yes | 8 | ✓ | 9694.0 | 888.6 | 100.0% | 7.03x | 5316 |
| C10 | perf | yes | 16 | ✓ | 9439.8 | 1228.8 | 100.0% | 7.01x | 10618 |

#### Qwen3.6-35B-A3B

At `Int8Mode::auto`, Precision on this card. The gate decodes 256 tokens a row
(C9/C10 at 64) — the 2026-10-05 sweep decoded 64 — and drafts to depth 4 up to
eight sessions (`draft_ladder.rs`), at 4.90 accepted tokens a step.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 | ✓ | 1094.2 | 228.0 | - | - | 905 |
| BF16 | prec | yes | 4 | ✓ | 6024.3 | 911.6 | - | - | 3662 |
| Q8_0 | prec | yes | 1 | ✓ | 2777.8 | 324.5 | 100.0% | 1.88x | 905 |
| C0 | prec | yes | 1 | ✓ | 2791.8 | 322.1 | 100.0% | 2.21x | 905 |
| C1 | prec | yes | 1 | ✓ | 2818.4 | 320.5 | 100.0% | 2.66x | 905 |
| C2 | prec | yes | 1 | ✓ | 2833.0 | 319.4 | 100.0% | 3.11x | 905 |
| C3 | prec | yes | 1 | ✓ | 2844.2 | 319.0 | 100.0% | 3.34x | 905 |
| C4 | prec | yes | 1 | ✓ | 2857.4 | 316.0 | 100.0% | 3.58x | 905 |
| C5 | prec | yes | 1 | ✓ | 2894.5 | 317.9 | 100.0% | 3.91x | 905 |
| C5 | prec | yes | 8 | ✓ | 5862.9 | 1171.3 | 100.0% | 3.91x | 7284 |
| C6 | prec | yes | 1 | ✓ | 2943.1 | 310.0 | 100.0% | 4.40x | 905 |
| C7 | prec | yes | 1 | ✓ | 2943.2 | 310.1 | 100.0% | 4.61x | 905 |
| C8 | prec | yes | 5 | ✓ | 7184.4 | 976.1 | 100.0% | 5.24x | 4567 |
| C9 | prec | yes | 2 | ✓ | 4856.9 | 484.5 | 100.0% | 5.83x | 1466 |
| C10 | prec | yes | 8 | ✓ | 6079.7 | 991.4 | 100.0% | 6.43x | 5748 |
| C10 | prec | yes | 16 | ✓ | 6481.5 | 1346.3 | 100.0% | 6.42x | 11482 |

#### Qwen3.6-35B AntiLoop+StyleTune (auto/Precision)

The npcd production configuration — AntiLoop trunk under StyleTune's output head,
at `Int8Mode::auto` (Precision on this int8-MMA-less card).

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 | ✓ | 1198.4 | 63.3 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 5991.7 | 559.8 | - | - | 2678 |
| Q8_0 | prec | yes | 1 | ✓ | 2980.8 | 200.6 | 100.0% | 1.88x | 659 |
| C0 | prec | yes | 1 | ✓ | 3635.4 | 231.8 | 100.0% | 2.21x | 659 |
| C1 | prec | yes | 1 | ✓ | 3617.1 | 227.9 | 100.0% | 2.67x | 659 |
| C2 | prec | yes | 1 | ✓ | 3608.0 | 231.9 | 100.0% | 3.13x | 659 |
| C3 | prec | yes | 1 | ✓ | 3645.1 | 228.3 | 100.0% | 3.35x | 659 |
| C4 | prec | yes | 1 | ✓ | 3626.9 | 219.7 | 100.0% | 3.59x | 659 |
| C5 | prec | yes | 1 | ✓ | 3586.8 | 149.9 | 100.0% | 3.91x | 659 |
| C5 | prec | yes | 8 | ✓ | 7568.4 | 829.9 | 100.0% | 3.91x | 5316 |
| C6 | prec | yes | 1 | ✓ | 3809.5 | 219.2 | 100.0% | 4.41x | 659 |
| C7 | prec | yes | 1 | ✓ | 3915.2 | 163.4 | 100.0% | 4.64x | 659 |
| C8 | prec | yes | 5 | ✓ | 8348.6 | 711.0 | 100.0% | 5.24x | 3337 |
| C9 | prec | yes | 2 | ✓ | 6451.2 | 386.7 | 100.0% | 5.83x | 1358 |
| C10 | prec | yes | 8 | ✓ | 8121.1 | 900.5 | 100.0% | 6.47x | 5316 |
| C10 | prec | yes | 16 | ✓ | 7226.2 | 1166.7 | 100.0% | 6.46x | 10618 |

The gate decodes 10 tokens a row, at the same depth-4 drafting; C5 and C7 ×1 read
under their neighbours, as on 2026-10-07, where an early rejected proposal leaves a
three-step row no room to amortise it.

#### Qwen3.6-35B AntiLoop+StyleTune (Performance)

The same hybrid at `Int8Mode::Performance` — same-width KO twins — priced against
the `auto`/Precision row above.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1864.5 | 80.7 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 7590.5 | 650.3 | - | - | 2678 |
| Q8_0 | perf | yes | 1 | ✓ | 5827.0 | 213.9 | 100.0% | 1.88x | 659 |
| C0 | perf | yes | 1 | ✓ | 5853.2 | 240.0 | 100.0% | 2.21x | 659 |
| C1 | perf | yes | 1 | ✓ | 5836.0 | 153.5 | 100.0% | 2.66x | 659 |
| C2 | perf | yes | 1 | ✓ | 4440.6 | 237.9 | 100.0% | 3.12x | 659 |
| C3 | perf | yes | 1 | ✓ | 5899.2 | 242.8 | 100.0% | 3.35x | 659 |
| C4 | perf | yes | 1 | ✓ | 5830.4 | 112.9 | 100.0% | 3.58x | 659 |
| C5 | perf | yes | 1 | ✓ | 5696.0 | 221.8 | 100.0% | 3.91x | 659 |
| C5 | perf | yes | 8 | ✓ | 9917.7 | 947.4 | 100.0% | 3.91x | 5316 |
| C6 | perf | yes | 1 | ✓ | 5828.5 | 171.0 | 100.0% | 4.40x | 659 |
| C7 | perf | yes | 1 | ✓ | 5853.4 | 173.8 | 100.0% | 4.63x | 659 |
| C8 | perf | yes | 5 | ✓ | 10221.6 | 764.7 | 100.0% | 5.22x | 3337 |
| C9 | perf | yes | 2 | ✓ | 8961.1 | 404.2 | 100.0% | 5.81x | 1358 |
| C10 | perf | yes | 8 | ✓ | 9902.3 | 790.9 | 100.0% | 6.45x | 5316 |
| C10 | perf | yes | 16 | ✓ | 9907.9 | 1269.0 | 100.0% | 6.42x | 10618 |

#### Qwen3.8-Flash-Next (Q2_KO experts)

The card runs the same **Q2_KO-expert** artifact as the 4090 Mobile (both sit
under the ladder's 32 GiB rung, so the recipe and its digest tag `130076148f33`
are identical). The ×16 rung the 16 GB card skips runs here.

The 3090's host changes the expert tiers, not the format: its 64 GB of RAM pins
**all 24,064 evictable experts** (31.0 GiB warm tier, no shortfall), so every
miss is a host→device upload over PCIe 3.0 and the run reads nothing from the
NVMe pack. The 4090 Mobile's 31.5 GiB host cannot hold that tier.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 (cold) | ✓ | 270.6 | 97.3 | - | - | 905 |
| BF16 | prec | yes | 4 | ✓ | 1570.5 | 320.5 | - | - | 3662 |
| BF16 | prec | yes | 8 | ✓ | 2051.7 | 477.2 | - | - | 7284 |
| BF16 | prec | yes | 16 | ✓ | 2301.9 | 543.8 | - | - | 14554 |
| BF16 | prec | yes | 1 (warm) | ✓ | 556.5 | 138.1 | - | - | 905 |
| C0 | prec | yes | 2 | ✓ | 1036.1 | 222.0 | 100.0% | 2.19x | 1850 |
| C5 | prec | yes | 2 | ✓ | 1025.1 | 212.0 | 100.0% | 3.80x | 1850 |
| C5 | prec | yes | 8 | ✓ | 2053.8 | 417.1 | 100.0% | 3.79x | 7284 |
| C8 | prec | yes | 2 | ✓ | 1010.0 | 210.0 | 100.0% | 4.87x | 1850 |
| C10 | prec | yes | 2 | ✓ | 1024.5 | 191.1 | 100.0% | 5.83x | 1466 |
| C10 | prec | yes | 8 | ✓ | 2078.9 | 361.6 | 100.0% | 5.80x | 5748 |

**One warm session decodes 138.1 t/s, and sixteen reach 543.8 t/s aggregate.**
The gate decodes 256 tokens a row (C9/C10 at 64); the 2026-10-05 sweep decoded 64.
The model's first run on this card, on 2026-09-30 (build `e596fad8d`), read 24.3
t/s for one warm session, 113.2 at ×8, 147.0 at ×16 and 95.6 at C10 ×8, with
prefill at ×4 of 1,033.4. Against that run, decode is 5.7× at one session, 4.2× at
×8, 3.7× at ×16 and 3.8× at C10 ×8, and prefill at ×4 is up 52%. The 2026-10-05
sweep (`5776799ac`) stood at 73.3, 233.8, 226.6 and 255.1, and the 2026-10-07 sweep
(`7aecbeab3`) at 122.8, 413.7, 460.8 and 306.3. The decode hot-path work that landed
by 2026-10-04 (fused MoE shared-expert residual, fused hyper-connection producers,
host-side wave overhead, QSA index pages resident in the span) doubled decode across
the ladder, the live MoE dispatch added 40–47% to the compressed ×2 rows, and the
single-session decode work after it — a Q4 draft head with a recorded draft walk,
latency-bound split-K and verify-attention kernels, stacked MoE input projections —
took it to the 2026-10-07 figures. Router look-ahead, wider copy workers, a give-up
for cold waits and the blended read-ahead predictor added 12–18% more by 2026-10-10,
and prefill at ×4–×16 rose 16–38%. C10 here is the retuned row (K 1.35, V 1.9):
5.80× at ×8, 5.83× at ×2.

**Against the 4090 Mobile (§3.9, 2026-10-10), same artifact.** BF16 decode at
×8 is 477.2 against 108.5 t/s (4.4×), one warm session 138.1 against 34.9, and
the 3090 adds a ×16 row; prefill at ×4 is 1,570.5 against 793.4.

**Engine probes** (`kv_fragmentation`, §2.2), at the same build, under the same
card-to-itself rule:

| Probe | Story | Worst sustained VRAM efficiency | Weight uptake | Result |
|---|---:|---:|---|---|
| Qwen3-30B-A3B (`qwen3_30b_a3b_q4`) | 20/20 | **59%** (single sample 48%) | at its limit | fail (efficiency) |
| Qwen3.6-35B-A3B (`qwen36_35b_a3b_q4`) | 16/16 | 98% (single sample 56%) | at its limit, 100% uptake | pass |
| Qwen3.8-Flash-Next (`qwen38_flash_next`) | 8/8 | **79%** (single sample 71%) | at its limit, 110% uptake | fail (efficiency) |

The Qwen3.6-35B probe runs speculative decode with the model's NextN head, so
every rejected draft rewinds the DeltaNet recurrent state; all sixteen sessions
rewrote the story correctly. Every probe told its story in full, with no
non-finite layer. **The Qwen3-30B and Flash-Next probes failed the 90% efficiency
gate with KV compaction running**: 3,008 MiB and 1,264 MiB of the ground denied
to the weight side were not holding KV. At `7aecbeab3` (2026-10-07) they read 99%
and 97%. The 72 GB card recorded the same gate as open on 2026-10-10, flipping
between about 80% and 99% from run to run; 59% is below that range. The cause is
not yet established.

At `7aecbeab3` compaction stopped relocating every tenant's arenas into any hole
below the frontier — an arena under another tenant's live region was emptied into
a hole the next pass refilled, and Flash-Next's recurrent-state passes moved 912
slots each with the region count unchanged — and relocates only the arenas holding
the frontier (`region_pool::frontier_relocations`), with 24–28% fewer moves. Before
it these two probes read 85% and 76–78% on this card. The 2026-10-05 sweep read
99% / 99% / 98%.

### 3.9 RTX 4090 Mobile 16 GB — the width gate sweep

A single sequential sweep on **2026-10-10**, on the RTX 4090 Mobile (§1), build
`21c36efbd`, of the same `test_parallel_batched_forwarding*` gates as §3.7 and
§3.8 — one `cargo test` invocation per model so exactly one was ever resident,
`--release --features cuda`, with the daemon stopped and the card at its idle
floor before each. Like §3.8 it is a **width / throughput** sweep at the gates'
fixed ~700-token prompt, **not** depth, and every cell is a single measurement,
so the 1–4 % noise floor (§5) applies to each on its own. Every gate passed: no
session in any of the 192 rows failed its check. The run, its comparison with the
2026-10-06 sweep on this card, and a same-session A/B of Flash-Next across the
merge that produced this build are in `results/sweep_rtx_4090_mobile_16gb_2026-10-10.md`;
the ladders of the first sweep on this card (2026-09-30, `bf291341c`) are in
`results/performance_rtx_4090_mobile_16gb_rows.tsv`.

**Thirteen gates — ten of the fleet's models, the two AntiLoop+StyleTune
hybrids, and Qwen3.8-Flash-Next.** Flash-Next runs on this card from its
**Q2_KO-expert** artifact (`qwen4exp::prepare`), not the Q4_KOEXP experts the
72 GB card runs (§3.1), streaming its experts VRAM → pinned RAM → NVMe through
the expert cache; its ×16 rung needs 24 GiB and the gate skips it here. The 284B
DeepSeek was not run. **Qwen2-0.5B's gate ladder has no compressed rung**, as in
§3.8.

**Every MoE here streams its experts.** At 16 GB none of the MoE models holds its
whole expert set resident, so a routed miss is a host→device upload over PCIe
4.0, where the 3090's 24 GB holds more of each set. That is the expected reason
their rates sit under the 3090's; this sweep did not profile it, so the
attribution comes from the mechanism, not from a measurement. The dense models
do not stream experts.

**Summary** — best prefill, best decode, and the best validated compression over
each model's ladder:

| Model | best prefill t/s | best decode t/s | best compression (mode) |
|---|---:|---:|---|
| Qwen2-0.5B | 86,038.3 | 5,123.9 | — (no ladder) |
| Qwen3.5-0.8B | 52,030.7 | 3,810.7 | 4.38× (C10) |
| Llama-3.2-3B | 15,194.5 | 688.7 | 4.43× (C10) |
| Llama-2-7B | 6,573.9 | 904.6 | 3.56× (Q4_0) |
| Qwen3-8B | 6,561.7 | 308.4 | 5.82× (C10) |
| Qwen3.5-9B | 7,601.7 | 873.0 | 5.57× (C10) |
| Qwen3.8-27B | 2,318.9 | 219.7 | 5.03× (C10) |
| Qwen3-30B-A3B | 6,460.6 | 310.0 | 5.45× (C10) |
| Qwen3.5-35B-A3B | 3,307.3 | 395.2 | 7.03× (C10) |
| Qwen3.6-35B-A3B | 2,649.6 | 336.7 | 6.43× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (auto/Precision) | 2,949.4 | 384.3 | 6.47× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (Performance) | 3,886.7 | 539.2 | 6.45× (C10) |
| Qwen3.8-Flash-Next (Q2_KO experts) | 1,128.7 | 108.5 | 5.82× (C10) |

Llama-2-7B's best compression is from a row the gate does not validate for
reproduction (`-`); its sessions still passed. Qwen3-30B-A3B's best decode is
its unvalidated Q4_0 ×20 row; its best validated row is Q8_0 ×20 at 299.6.

**Qwen3.8-27B ran twice.** Its first run read BF16 ×1 decode 8.1 and C8 ×20 25.9
t/s; the sweep was stopped, the card cooled to a flat 46 °C, and the 27B rerun
read 8.2 and 28.9 with the SM clock at 1,877 MHz mean. These two rows are slow on
this card at every build recorded (7.0 and 28.6 on 2026-09-30) while its other ×1
rows decode at 82–83; they are not heat. The ladder below is the rerun.

**Against the RTX 3090, on the models both ran.** The 3090's sweep (§3.8) is on
build `c375834c8`, the same day as this one.

- **Prefill is higher on the 16 GB card on every dense model** — Qwen2-0.5B
  86,038 vs 72,012, Qwen3.5-0.8B 52,031 vs 44,981, Llama-3.2-3B 15,195 vs 10,920
  (the 3090's without flash-attn), Llama-2-7B 6,574 vs 5,738, Qwen3-8B 6,562 vs
  5,286, Qwen3.5-9B 7,602 vs 6,735, Qwen3.8-27B 2,319 vs 2,067. On the streamed
  MoEs the 3090 leads, by the most where it holds more of each expert set:
  Qwen3-30B 6,461 vs 6,822, Qwen3.5-35B 3,307 vs 10,042, Qwen3.6-35B 2,650 vs
  7,184, Flash-Next 1,129 vs 2,302.
- **Decode is lower here on every model**, least on the dense ones (Llama-2-7B
  904.6 vs 934.9, Qwen3.5-9B 873.0 vs 926.2, Qwen3.5-0.8B 3,810.7 vs 4,177.8) and
  most on the MoEs: Qwen3.5-35B 395.2 vs 1,228.8, Qwen3.6-35B 336.7 vs 1,346.3,
  Flash-Next 108.5 vs 543.8 (the 3090 adds a ×16 row). Expert streaming
  is the expected cause (above): a bigger card buys speed, not feasibility.
- **Compression at C10** is now the same calibration on both cards: 7.03× / 6.43×
  on the two 35Bs here against the 3090's 7.03× / 6.45×.

**Flash-Next against the 72 GB card** is the same build (`21c36efbd`) but not a
like-for-like comparison: that card runs Q4_KOEXP experts almost wholly resident
and this one Q2_KO experts streamed. So the gap — BF16 ×8 decode 108.5 here
against 1,008.2 there — is the expert tier more than the card.

**Engine probes.** They were not run in this sweep. The table is the 2026-09-30
sweep's, build `bf291341c`; the 2026-10-06 sweep on this card
(`results/sweep_rtx_4090_mobile_16gb_2026-10-06.md`) read Flash-Next 8/8 at 100%
efficiency and 97% uptake, and the 30B 20/20 at 66% efficiency. Both
`kv_fragmentation` probes (§2.2) ran after the gates, under the same
card-to-itself rule:

| Probe | Story | Worst sustained VRAM efficiency | Weight uptake | Result |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B (`qwen3_30b_a3b_q4`) | 20/20 | 43% (single sample 39%) | 75% of 1,600 MiB released | **fail — efficiency under the 90% threshold** |
| Qwen3.8-Flash-Next (`qwen38_flash_next`) | 8/8 | 100% (single sample 92%) | 73% of 3,136 MiB released | pass |

The 30B's efficiency shortfall is not new on this card — earlier runs of the same
probe read 36%, 32% and 61% — and its story passed, so the K/V it holds is
correct; the gap is ground the arena frontier keeps that is not holding KV.
§4 records it.

**Full ladders.** Each model's complete gate ladder — every KV mode and context
the gate ran, exactly as the run logs printed them. `Valid` is the per-config
reproduction check (`✓`, or `-` for a mode not validated for reproduction);
`int8` is the loader's int8 posture (`prec`/`perf`/`off`).

#### Qwen2-0.5B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | off | no | 1 | - | 40928.4 | 434.8 | - | - | 308 |
| BF16 | off | yes | 1 | - | 42648.1 | 456.6 | - | - | 308 |
| F16 | off | yes | 1 | - | 41779.3 | 420.1 | - | - | 308 |
| F16 | off | yes | 4 | - | 76880.3 | 1368.7 | - | - | 1272 |
| F16 | off | yes | 60 | - | 85233.7 | 4983.6 | - | - | 18612 |
| BF16 | off | yes | 60 | - | 86038.3 | 5123.9 | - | - | 18612 |

#### Qwen3.5-0.8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 30506.4 | 190.9 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 30447.8 | 220.4 | - | - | 659 |
| BF16 | prec | yes | 16 | ✓ | 47556.2 | 2935.0 | - | - | 10618 |
| Q8_0 | prec | yes | 4 | ✓ | 52030.7 | 1030.3 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 2 | ✓ | 48994.0 | 550.7 | 100.0% | 1.87x | 1358 |
| C1 | prec | yes | 2 | ✓ | 49036.7 | 430.6 | 100.0% | 2.13x | 1358 |
| C2 | prec | yes | 2 | ✓ | 27838.1 | 406.9 | 100.0% | 2.33x | 1358 |
| C3 | prec | yes | 2 | ✓ | 24286.7 | 365.5 | 100.0% | 2.82x | 1358 |
| C4 | prec | yes | 2 | ✓ | 21235.2 | 309.5 | 100.0% | 2.66x | 1358 |
| C5 | prec | yes | 2 | ✓ | 19248.2 | 303.8 | 100.0% | 2.93x | 1358 |
| C6 | prec | yes | 2 | ✓ | 19937.2 | 300.2 | 100.0% | 3.23x | 1358 |
| C7 | prec | yes | 2 | ✓ | 19951.9 | 302.3 | 100.0% | 3.59x | 1358 |
| C8 | prec | yes | 32 | ✓ | 46056.8 | 3810.7 | 100.0% | 3.98x | 21202 |
| C9 | prec | yes | 5 | ✓ | 50567.8 | 1260.8 | 100.0% | 4.16x | 3337 |
| C10 | prec | yes | 10 | ✓ | 50135.7 | 2017.5 | 100.0% | 4.38x | 6640 |

#### Llama-3.2-3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | prec | no | 1 | ✓ | 13132.3 | 138.7 | - | - | 654 |
| F16 | prec | yes | 1 | ✓ | 13319.6 | 138.5 | - | - | 654 |
| F16 | prec | yes | 4 | ✓ | 14841.7 | 497.3 | - | - | 2658 |
| R16 | prec | yes | 1 | ✓ | 14260.3 | 122.0 | 0.0% | - | 654 |
| Q8_0 | prec | yes | 1 | ✓ | 14303.1 | 129.6 | 100.0% | 1.88x | 654 |
| Q8_Q4 | prec | yes | 1 | ✓ | 14144.1 | 124.1 | 100.0% | 2.29x | 654 |
| BF16 | prec | yes | 4 | ✓ | 14299.4 | 490.5 | - | - | 2658 |
| Q8_1 | prec | yes | 4 | ✓ | 15194.5 | 412.9 | 100.0% | 1.78x | 2658 |
| Q8_KS | prec | yes | 4 | ✓ | 14528.7 | 408.8 | 100.0% | 1.78x | 2658 |
| Q8_Q4 | prec | yes | 4 | ✓ | 15106.9 | 422.3 | 100.0% | 2.29x | 2658 |
| Q4_0 | prec | yes | 4 | - | 14187.1 | 432.8 | 100.0% | 3.56x | 2658 |
| Q4_1 | prec | yes | 4 | - | 14227.1 | 420.5 | 100.0% | 3.20x | 2658 |
| Q4_KS | prec | yes | 4 | - | 15094.4 | 430.4 | 100.0% | 3.20x | 2658 |
| C0 | prec | yes | 1 | ✓ | 13921.7 | 125.6 | 100.0% | 1.89x | 654 |
| C1 | prec | yes | 1 | ✓ | 12738.2 | 126.9 | 100.0% | 2.28x | 654 |
| C2 | prec | yes | 1 | ✓ | 12679.6 | 127.8 | 100.0% | 2.50x | 654 |
| C3 | prec | yes | 1 | ✓ | 14049.1 | 127.7 | 100.0% | 3.06x | 654 |
| C4 | prec | yes | 1 | ✓ | 14080.8 | 127.4 | 100.0% | 2.85x | 654 |
| C5 | prec | yes | 1 | ✓ | 13993.0 | 128.4 | 100.0% | 3.13x | 654 |
| C6 | prec | yes | 1 | ✓ | 13945.3 | 126.7 | 100.0% | 3.58x | 654 |
| C7 | prec | yes | 1 | ✓ | 13966.4 | 127.0 | 100.0% | 3.78x | 654 |
| C8 | prec | yes | 10 | ✓ | 14240.2 | 688.7 | 100.0% | 3.99x | 6590 |
| C9 | prec | yes | 10 | ✓ | 14180.0 | 685.3 | 100.0% | 4.22x | 6590 |
| C10 | prec | yes | 5 | ✓ | 14052.5 | 504.4 | 100.0% | 4.43x | 3312 |

#### Llama-2-7B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | perf | no | 1 | - | 5694.8 | 94.3 | - | - | 283 |
| F16 | perf | yes | 1 | - | 5653.6 | 95.3 | - | - | 283 |
| F16 | perf | yes | 4 | - | 6477.7 | 315.2 | - | - | 1176 |
| F16 | perf | yes | 8 | - | 6573.9 | 499.3 | - | - | 2316 |
| BF16 | perf | yes | 1 | - | 5864.4 | 89.3 | - | - | 283 |
| BF16 | perf | yes | 8 | - | 6235.0 | 498.4 | - | - | 2316 |
| BF16 | perf | yes | 16 | - | 6211.8 | 716.1 | - | - | 4624 |
| BF16 | perf | yes | 48 | - | 6322.2 | 904.6 | - | - | 13796 |
| Q8_0 | perf | yes | 32 | - | 5847.1 | 479.8 | 100.0% | 1.88x | 9220 |
| Q4_0 | perf | yes | 32 | - | 5754.9 | 492.0 | 100.0% | 3.56x | 9220 |

#### Qwen3-8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | no | 1 | ✓ | 5651.7 | 60.5 | - | - | 636 |
| F16 | perf | yes | 1 | ✓ | 5938.5 | 63.0 | - | - | 636 |
| F16 | perf | yes | 2 | ✓ | 6428.5 | 124.7 | - | - | 1312 |
| BF16 | perf | yes | 4 | ✓ | 6561.7 | 231.1 | - | - | 2586 |
| Q8_0 | perf | yes | 4 | ✓ | 6407.1 | 183.5 | 100.0% | 1.88x | 2586 |
| C0 | perf | yes | 1 | ✓ | 5758.3 | 58.8 | 100.0% | 1.91x | 636 |
| C1 | perf | yes | 1 | ✓ | 6070.0 | 54.0 | 100.0% | 2.56x | 636 |
| C2 | perf | yes | 1 | ✓ | 5598.8 | 58.0 | 100.0% | 2.79x | 636 |
| C3 | perf | yes | 1 | ✓ | 5760.0 | 58.2 | 100.0% | 3.28x | 636 |
| C4 | perf | yes | 1 | ✓ | 5653.6 | 58.8 | 100.0% | 3.21x | 636 |
| C5 | perf | yes | 1 | ✓ | 5963.9 | 59.5 | 100.0% | 3.40x | 636 |
| C6 | perf | yes | 1 | ✓ | 5893.9 | 59.9 | 100.0% | 4.04x | 636 |
| C7 | perf | yes | 1 | ✓ | 5940.8 | 59.7 | 100.0% | 4.31x | 636 |
| C8 | perf | yes | 10 | ✓ | 6437.2 | 308.4 | 100.0% | 4.79x | 6410 |
| C9 | perf | yes | 5 | ✓ | 6500.6 | 224.6 | 100.0% | 5.17x | 3222 |
| C10 | perf | yes | 5 | ✓ | 6393.0 | 228.0 | 100.0% | 5.82x | 3222 |

#### Qwen3.5-9B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 6747.6 | 70.5 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 6983.6 | 74.4 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 7213.1 | 418.7 | - | - | 2678 |
| Q8_0 | prec | yes | 4 | ✓ | 7601.7 | 423.5 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 1 | ✓ | 6683.7 | 116.3 | 100.0% | 2.20x | 659 |
| C1 | prec | yes | 1 | ✓ | 6750.1 | 115.5 | 100.0% | 2.43x | 659 |
| C2 | prec | yes | 1 | ✓ | 7096.8 | 116.5 | 100.0% | 2.72x | 659 |
| C3 | prec | yes | 1 | ✓ | 6964.2 | 106.7 | 100.0% | 3.19x | 659 |
| C4 | prec | yes | 1 | ✓ | 5614.8 | 112.4 | 100.0% | 3.26x | 659 |
| C5 | prec | yes | 1 | ✓ | 6440.8 | 116.0 | 100.0% | 3.60x | 659 |
| C6 | prec | yes | 1 | ✓ | 6766.7 | 117.0 | 100.0% | 4.04x | 659 |
| C7 | prec | yes | 1 | ✓ | 6755.9 | 116.3 | 100.0% | 4.23x | 659 |
| C8 | prec | yes | 20 | ✓ | 7216.5 | 873.0 | 100.0% | 4.71x | 13278 |
| C9 | prec | yes | 5 | ✓ | 7316.5 | 474.2 | 100.0% | 5.17x | 3337 |
| C10 | prec | yes | 10 | ✓ | 7274.2 | 792.8 | 100.0% | 5.57x | 6640 |

#### Qwen3.8-27B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1259.1 | 8.2 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 2318.9 | 219.7 | - | - | 2678 |
| Q8_0 | perf | yes | 4 | ✓ | 2315.5 | 219.0 | 100.0% | 1.88x | 2678 |
| C0 | perf | yes | 1 | ✓ | 2134.6 | 82.7 | 100.0% | 2.17x | 659 |
| C1 | perf | yes | 1 | ✓ | 2109.6 | 82.4 | 100.0% | 2.32x | 659 |
| C2 | perf | yes | 1 | ✓ | 2116.6 | 82.7 | 100.0% | 2.56x | 659 |
| C3 | perf | yes | 1 | ✓ | 2109.1 | 81.7 | 100.0% | 3.10x | 659 |
| C4 | perf | yes | 1 | ✓ | 2100.9 | 83.2 | 100.0% | 3.02x | 659 |
| C5 | perf | yes | 1 | ✓ | 2104.7 | 83.1 | 100.0% | 3.41x | 659 |
| C6 | perf | yes | 1 | ✓ | 2133.5 | 82.8 | 100.0% | 3.80x | 659 |
| C7 | perf | yes | 1 | ✓ | 2123.3 | 82.3 | 100.0% | 3.94x | 659 |
| C8 | perf | yes | 20 | ✓ | 1333.2 | 28.9 | 100.0% | 4.38x | 13278 |
| C9 | perf | yes | 5 | ✓ | 2212.4 | 72.7 | 100.0% | 4.75x | 3337 |
| C10 | perf | yes | 10 | ✓ | 2225.6 | 79.7 | 100.0% | 5.03x | 6640 |

#### Qwen3-30B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | perf | yes | 1 | ✓ | 869.1 | 13.2 | - | - | 626 |
| BF16 | perf | yes | 1 | ✓ | 1807.3 | 26.7 | - | - | 626 |
| BF16 | perf | yes | 10 | ✓ | 5502.5 | 219.5 | - | - | 6310 |
| Q8_0 | perf | yes | 20 | ✓ | 6139.1 | 299.6 | 100.0% | 1.88x | 12620 |
| Q4_0 | perf | yes | 4 | - | 4357.6 | 191.4 | 100.0% | 3.56x | 2546 |
| C0 | perf | yes | 2 | ✓ | 2986.6 | 110.6 | 100.0% | 2.03x | 1292 |
| C1 | perf | yes | 2 | ✓ | 3085.7 | 110.3 | 100.0% | 2.49x | 1292 |
| C2 | perf | yes | 2 | ✓ | 3050.2 | 115.4 | 100.0% | 2.70x | 1292 |
| C3 | perf | yes | 2 | ✓ | 3066.0 | 102.3 | 100.0% | 3.20x | 1292 |
| C4 | perf | yes | 2 | ✓ | 3026.9 | 104.1 | 100.0% | 3.10x | 1292 |
| C5 | perf | yes | 2 | ✓ | 3107.1 | 106.8 | 100.0% | 3.35x | 1292 |
| C5 | perf | yes | 8 | ✓ | 6460.6 | 240.9 | 100.0% | 3.35x | 5052 |
| C6 | perf | yes | 2 | ✓ | 2989.1 | 112.7 | 100.0% | 3.86x | 1292 |
| C7 | perf | yes | 2 | ✓ | 3034.6 | 108.8 | 100.0% | 4.06x | 1292 |
| C8 | perf | yes | 2 | ✓ | 3073.3 | 111.6 | 100.0% | 4.57x | 1292 |
| C9 | perf | yes | 2 | ✓ | 3064.8 | 132.2 | 100.0% | 5.01x | 1292 |
| C10 | perf | yes | 2 | ✓ | 3062.5 | 130.0 | 100.0% | 5.45x | 1292 |
| BF16 | perf | yes | 1 | ✓ | 1884.7 | 29.9 | - | - | 626 |
| Q4_0 | perf | yes | 20 | - | 6022.3 | 310.0 | 100.0% | 3.56x | 12620 |

#### Qwen3.5-35B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 571.4 | 24.2 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 2824.7 | 236.9 | - | - | 2678 |
| Q8_0 | perf | yes | 2 | ✓ | 1693.6 | 157.2 | 100.0% | 1.88x | 1358 |
| C0 | perf | yes | 1 | ✓ | 920.7 | 156.1 | 100.0% | 2.21x | 659 |
| C1 | perf | yes | 1 | ✓ | 936.4 | 155.1 | 100.0% | 2.85x | 659 |
| C2 | perf | yes | 1 | ✓ | 929.6 | 150.8 | 100.0% | 3.24x | 659 |
| C3 | perf | yes | 1 | ✓ | 937.4 | 154.6 | 100.0% | 3.39x | 659 |
| C4 | perf | yes | 1 | ✓ | 939.1 | 136.0 | 100.0% | 3.73x | 659 |
| C5 | perf | yes | 1 | ✓ | 951.5 | 143.1 | 100.0% | 4.03x | 659 |
| C6 | perf | yes | 1 | ✓ | 964.2 | 146.8 | 100.0% | 4.62x | 659 |
| C7 | perf | yes | 1 | ✓ | 968.4 | 148.0 | 100.0% | 4.90x | 659 |
| C8 | perf | yes | 5 | ✓ | 3307.3 | 384.6 | 100.0% | 5.52x | 3337 |
| C9 | perf | yes | 2 | ✓ | 1654.9 | 228.2 | 100.0% | 6.23x | 1358 |
| C10 | perf | yes | 8 | ✓ | 2728.8 | 395.2 | 100.0% | 7.03x | 5316 |
| C10 | perf | yes | 16 | ✓ | 3056.4 | 330.8 | 100.0% | 7.01x | 10618 |

#### Qwen3.6-35B-A3B

At `Int8Mode::auto`, Precision on this int8-MMA card. The gate decodes 256 tokens
a row (C9/C10 at 64), so its peak tokens are longer than the 2026-09-30 ladder's.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 | ✓ | 439.1 | 71.6 | - | - | 905 |
| BF16 | prec | yes | 4 | ✓ | 2081.0 | 302.1 | - | - | 3662 |
| Q8_0 | prec | yes | 1 | ✓ | 667.5 | 109.8 | 100.0% | 1.88x | 905 |
| C0 | prec | yes | 1 | ✓ | 698.7 | 109.6 | 100.0% | 2.21x | 905 |
| C1 | prec | yes | 1 | ✓ | 693.4 | 108.7 | 100.0% | 2.66x | 905 |
| C2 | prec | yes | 1 | ✓ | 704.1 | 107.9 | 100.0% | 3.11x | 905 |
| C3 | prec | yes | 1 | ✓ | 704.0 | 107.2 | 100.0% | 3.34x | 905 |
| C4 | prec | yes | 1 | ✓ | 703.0 | 103.8 | 100.0% | 3.58x | 905 |
| C5 | prec | yes | 1 | ✓ | 699.3 | 103.9 | 100.0% | 3.91x | 905 |
| C5 | prec | yes | 8 | ✓ | 2193.8 | 307.1 | 100.0% | 3.91x | 7284 |
| C6 | prec | yes | 1 | ✓ | 636.7 | 97.6 | 100.0% | 4.40x | 905 |
| C7 | prec | yes | 1 | ✓ | 708.3 | 96.8 | 100.0% | 4.61x | 905 |
| C8 | prec | yes | 5 | ✓ | 2649.6 | 264.0 | 100.0% | 5.24x | 4567 |
| C9 | prec | yes | 2 | ✓ | 1186.6 | 140.3 | 100.0% | 5.83x | 1466 |
| C10 | prec | yes | 8 | ✓ | 2131.5 | 252.1 | 100.0% | 6.43x | 5748 |
| C10 | prec | yes | 16 | ✓ | 2466.3 | 336.7 | 100.0% | 6.42x | 11482 |

#### Qwen3.6-35B AntiLoop+StyleTune (auto/Precision)

The npcd production configuration — AntiLoop trunk under StyleTune's output head,
at `Int8Mode::auto` (Precision on this int8-MMA card).

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 | ✓ | 503.5 | 24.9 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 2227.5 | 167.3 | - | - | 2678 |
| Q8_0 | prec | yes | 1 | ✓ | 787.7 | 76.3 | 100.0% | 1.88x | 659 |
| C0 | prec | yes | 1 | ✓ | 812.8 | 160.5 | 100.0% | 2.21x | 659 |
| C1 | prec | yes | 1 | ✓ | 833.0 | 166.5 | 100.0% | 2.67x | 659 |
| C2 | prec | yes | 1 | ✓ | 838.5 | 164.9 | 100.0% | 3.13x | 659 |
| C3 | prec | yes | 1 | ✓ | 841.3 | 163.8 | 100.0% | 3.35x | 659 |
| C4 | prec | yes | 1 | ✓ | 846.2 | 156.4 | 100.0% | 3.59x | 659 |
| C5 | prec | yes | 1 | ✓ | 849.5 | 73.8 | 100.0% | 3.91x | 659 |
| C5 | prec | yes | 8 | ✓ | 2429.1 | 384.3 | 100.0% | 3.91x | 5316 |
| C6 | prec | yes | 1 | ✓ | 774.8 | 133.9 | 100.0% | 4.41x | 659 |
| C7 | prec | yes | 1 | ✓ | 824.6 | 89.7 | 100.0% | 4.64x | 659 |
| C8 | prec | yes | 5 | ✓ | 2949.4 | 316.4 | 100.0% | 5.24x | 3337 |
| C9 | prec | yes | 2 | ✓ | 1431.3 | 249.8 | 100.0% | 5.83x | 1358 |
| C10 | prec | yes | 8 | ✓ | 2460.0 | 313.4 | 100.0% | 6.47x | 5316 |
| C10 | prec | yes | 16 | ✓ | 2791.4 | 294.8 | 100.0% | 6.46x | 10618 |

#### Qwen3.6-35B AntiLoop+StyleTune (Performance)

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 701.8 | 34.2 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 3225.9 | 321.7 | - | - | 2678 |
| Q8_0 | perf | yes | 1 | ✓ | 1126.0 | 84.5 | 100.0% | 1.88x | 659 |
| C0 | perf | yes | 1 | ✓ | 1136.2 | 160.5 | 100.0% | 2.21x | 659 |
| C1 | perf | yes | 1 | ✓ | 1170.7 | 192.9 | 100.0% | 2.66x | 659 |
| C2 | perf | yes | 1 | ✓ | 1179.5 | 174.1 | 100.0% | 3.12x | 659 |
| C3 | perf | yes | 1 | ✓ | 1177.6 | 192.3 | 100.0% | 3.35x | 659 |
| C4 | perf | yes | 1 | ✓ | 1185.4 | 124.1 | 100.0% | 3.58x | 659 |
| C5 | perf | yes | 1 | ✓ | 1197.9 | 166.3 | 100.0% | 3.91x | 659 |
| C5 | perf | yes | 8 | ✓ | 3239.3 | 404.0 | 100.0% | 3.91x | 5316 |
| C6 | perf | yes | 1 | ✓ | 1050.0 | 100.5 | 100.0% | 4.40x | 659 |
| C7 | perf | yes | 1 | ✓ | 1144.9 | 103.3 | 100.0% | 4.63x | 659 |
| C8 | perf | yes | 5 | ✓ | 3886.7 | 348.5 | 100.0% | 5.22x | 3337 |
| C9 | perf | yes | 2 | ✓ | 2047.6 | 310.7 | 100.0% | 5.81x | 1358 |
| C10 | perf | yes | 8 | ✓ | 3182.1 | 539.2 | 100.0% | 6.45x | 5316 |
| C10 | perf | yes | 16 | ✓ | 3506.7 | 402.3 | 100.0% | 6.42x | 10618 |

#### Qwen3.8-Flash-Next (Q2_KO experts)

The gate runs BF16 ×1 twice — at the head of its ladder (cold) and again after
×8 (warm) — the same pair §3.6 *Width* labels.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 (cold) | ✓ | 145.4 | 33.4 | - | - | 905 |
| BF16 | prec | yes | 4 | ✓ | 793.4 | 89.5 | - | - | 3662 |
| BF16 | prec | yes | 8 | ✓ | 1127.5 | 108.5 | - | - | 7284 |
| BF16 | prec | yes | 1 (warm) | ✓ | 269.1 | 34.9 | - | - | 905 |
| C0 | prec | yes | 2 | ✓ | 463.3 | 61.6 | 100.0% | 2.19x | 1850 |
| C5 | prec | yes | 2 | ✓ | 499.2 | 60.7 | 100.0% | 3.80x | 1850 |
| C5 | prec | yes | 8 | ✓ | 1127.6 | 107.9 | 100.0% | 3.79x | 7284 |
| C8 | prec | yes | 2 | ✓ | 483.8 | 56.3 | 100.0% | 4.87x | 1850 |
| C10 | prec | yes | 2 | ✓ | 499.1 | 51.0 | 100.0% | 5.82x | 1466 |
| C10 | prec | yes | 8 | ✓ | 1128.7 | 101.3 | 100.0% | 5.79x | 5748 |

**One cold session decodes 33.4 t/s and eight reach 108.5 t/s aggregate.** This
build carries the expert cache's blended read-ahead prediction and its eviction
behind the wave by expected miss cost (`docs/moe_live_dispatch_design.md`
§0.7.4). Against the build before them on this card (2026-10-07, `856a0fc40`),
decode is up on every row: 26.2 → 33.4 at one cold session, 66.5 → 108.5 at
BF16 ×8, 71.5 → 107.9 at C5 ×8, 50.8 → 101.3 at C10 ×8, and 36.5 → 51.0 at
C10 ×2. Prefill at ×8 is 1,051.3 → 1,127.5.

---

## 4. Limits and open items

### The decode-slot refresh prefill regression (fixed)

`4a740f4a` (the same change reached `origin/main` separately as `d45e69ce`)
made every commit outside the decode kernel —
a prefill layer, a speculative verify block — bring the cached decode slot
buffer up to date, the fix for the MTP draft head's NaN. On the 72 GB card it
took Qwen2-0.5B's gate from 80,170.5 to 33,360.8 t/s at ×60 (−58 %) and from
32,220.1 to 17,699.5 at one context (−45 %), both measured 2026-09-15 against
`81e487b5` on the same card; decode never moved. Two causes were stacked:

1. **A fence per chunk.** The refresh waited for the previous upload — queued
   behind the layer's kernels — before rewriting the pinned buffer that upload
   read, so the host drained the GPU once per chunk per layer. Uploads now pack
   into two event-guarded staging buffers and the host copy is memory no copy
   reads (`234e037b`).
2. **An upload between layers.** Wherever the refresh ran — in the commit, or
   in the post-prefill prime right after it — it put one host→device copy
   between one layer's kernels and the next: about 90 µs of GPU time per layer
   with every host span flat (a `profile` build: 88.9 ms against 66.3 ms over
   ten one-context prefills of 24 layers, with and without it). A commit now
   only marks the buffer; the sync every reader goes through re-serialises the
   writer region first, and the prime builds only a buffer that is missing
   (`7dce6231`, `03794cac`).

A third loss, 12–22 % on the one- and two-context rows of most multi-config
gates, was the **harness**: each config's prompt timer started without
synchronising, and the work that config had just queued — its session setup,
which re-materialises the norm weights for the config's dtype, and its
system-prompt prefill — finished inside the timer; builds before `234e037b`,
which fenced slot-state uploads on the host, had drained it first. The gates
now synchronise before starting the timer (`23623c6b`), so a prompt row
measures the prompt alone. That is what the earlier rows measured too:
`81e487b5` gives the same rows with the synchronise as without it (Qwen3.5-9B's
one-context rows 5,472–5,527 t/s against 5,492–5,530; Qwen3-8B's C0–C7 ×1
5,703–5,812 against 5,792–5,818), so every row in §3 compares directly.

**Recovered, fleet-wide.** Twelve gates, best of two per build, both builds on
the synchronised harness (`81e487b5` against `23623c6b`, 2026-09-15, this
card). No prefill row of any model is more than 4 % below the reference, and
every run validated:

| Model | best prefill, ref / fix (t/s) | worst Δ prefill row | median Δ prefill |
|---|---:|---:|---:|
| Qwen2-0.5B | 79,901.4 / 80,430.5 | −2.7 % | −0.3 % |
| Qwen3.5-0.8B | 36,449.5 / 37,145.6 | +0.5 % | +1.1 % |
| Llama-3.2-3B | 14,094.6 / 14,612.6 | +1.0 % | +4.4 % |
| Llama-2-7B | 6,191.8 / 6,810.2 | +0.3 % | +5.3 % |
| Qwen3-8B | 6,139.8 / 6,241.6 | +1.5 % | +2.4 % |
| Qwen3.5-9B | 5,954.4 / 5,961.0 | −0.0 % | +0.1 % |
| Qwen3.8-27B | 1,758.5 / 1,761.6 | −0.0 % | +0.1 % |
| Qwen3-30B-A3B | 10,036.2 / 10,324.3 | +2.3 % | +3.6 % |
| Qwen3.5-35B-A3B | 7,304.1 / 7,315.0 | −0.2 % | +0.1 % |
| Qwen3.6-35B-A3B | 7,359.0 / 7,373.4 | −0.1 % | +0.2 % |
| Qwen3.8-Flash-Next | 1,968.2 / 2,060.9 | −1.9 % | +1.9 % |
| DeepSeek-V4-Flash | 1,083.9 / 1,120.6 | +0.9 % | +1.6 % |

Decode is within noise everywhere but one row: Qwen3.6-35B-A3B C10×16, 769.5 →
690.9 t/s (−10.2 %), both runs of each build agreeing — open item #11 in
`docs/open_items.md`.

### Qwen2-0.5B reports 0% quantized

Every C-mode row for Qwen2-0.5B-Instruct comes back `%Quantized = 0.0%` with no
compression ratio, while its sibling on the same code path (Qwen3-8B, also
`BatchedInference` + `from_gguf_by_path_with_int8`) reports 100% at 3.71×/6.31×.
The rows are otherwise healthy — output validates, and C5/C10 decode differs
from BF16, so the modes are not identical paths.

The zeros are model-specific rather than path-specific, and reproduce across
every run. The cause is not established, so **Qwen2-0.5B's C-mode rows are
reported as measured and excluded from every compression comparison in this
document.** Its throughput figures are unaffected and are used.

### DeepSeek-V4-Flash-0731 cannot reach 32K on this card

`CUDA_ERROR_OUT_OF_MEMORY` at the shallowest depth rung, with the GPU at
**72,691 of 73,415 MiB**. Its ~700-token width ladder runs fine (§3.7), so this
is specifically a depth limit, and it is not the ~6 GB DSpark drafter: removing
it changes nothing. The 284B model's expert working set fills the card on its
own, and the elastic partition has nothing left to concede to a 32K KV span.

This is the fleet's most interesting missing measurement — native-sparse
attention is precisely the architecture whose depth curve would be worth setting
against QSA and the DeltaNet hybrids. It needs a larger card or a smaller
resident expert budget.

### The slot-state buffer has a 16 MiB class ceiling

A model without GQA overruns it. Llama-2-7B-Chat has 32 KV heads, so its KV per
token is roughly 4× a GQA model's, and at 128K it produced `slot-state buffer of
17017152 B exceeds the 16777216 B top class`. This is a genuine **engine**
ceiling rather than a model limit.

Nothing in the current suite triggers it — Llama-2 is measured at 4K, well
inside the class — which is exactly why it is recorded here. Every other model
in the fleet has GQA and stays inside the ceiling regardless of depth.

### The 30B engine probe holds ground that is not KV (RTX 4090 Mobile)

The Qwen3-30B-A3B `kv_fragmentation` probe (§3.9) fails its VRAM-efficiency gate
on the 16 GB card: worst sustained 43% against a 90% threshold on 2026-09-30,
with 2,848 MiB of ground denied to the weight side not holding KV. Its story
(20/20) and weight uptake (75%) pass, so the K/V is correct and the frontier
does give ground back; what falls short is how much of the ground below the
frontier is in use. Earlier runs of the same probe on this card read 36%, 32%
and 61%, so it predates this sweep. The Flash-Next probe on the same card and
build holds 100%.

The 2026-09-30 run also logged KV compaction relocating slots that no holder it
reached names (up to 41,022 of a pass's moves), and a pass running out of
pre-provisioned record slots mid-sweep — relocations whose sources are then not
reclaimed. The earlier runs logged neither. Which change introduced them, and
whether they account for the shortfall, is not yet established.

### Validation at extreme width under maximum compression

Qwen3.6-35B-A3B's width ladder fails its 100% threshold at the two widest C10
rungs: **31/32 sessions at ×32 and 61/64 at ×64**. Throughput is healthy at both
(6,519.6 / 973.9 and 6,645.6 / 1,144.9 t/s), and the same model passes every
depth rung and every narrower width rung. Its sibling Qwen3.5-35B-A3B passes the
same ×32 and ×64 C10 rungs outright.

The rows are reported with their measured validity. One or two sessions in
sixty-four degrading under maximum compression at maximum concurrency is a
narrow enough failure that it is recorded as an open item rather than treated as
a general result about either C10 or width.

**It did not reproduce in the 2026-09-13 sweep.** Both rungs validated outright
— every session at ×32 (7,178.4 / 953.2 t/s) and at ×64 (7,140.5 / 1,150.7 t/s)
— so §3.7's ×64 cell for this model is a valid one. One clean run is not
evidence the failure is gone, since it was already intermittent at one or two
sessions in sixty-four; it stays open, recorded as not reproduced.

### Compression ratios moved between the two sweeps, in both directions

The width tables keep the **higher** ratio of the two sweeps, which for most
models is the first sweep's. So their `Compress` cells overstate what the
2026-09-13 build achieves at the top of the ladder on those models, and the
difference is recorded here rather than absorbed by the maximum:

| Model | C10, 09-03 → 09-13 | C8, 09-03 → 09-13 |
|---|---:|---:|
| Qwen3.5-0.8B | 4.68× → 4.11× | 3.88× → 3.83× |
| Llama-3.2-3B | 4.63× → 4.34× | 3.90× → 3.95× |
| Qwen3-8B | 6.21× → 5.85× | 4.85× → 4.85× |
| Qwen3.5-9B | 6.29× → 5.87× | 4.96× → 4.91× |
| Qwen3.5-35B-A3B | 7.10–7.14× → 6.20–6.23× | 5.40× → 5.13× |
| Qwen3.6-35B-A3B | 6.65–6.68× → 6.04–6.06× | 5.18× → 5.04× |
| Qwen3.8-Flash-Next | 6.89–6.93× → 6.73–6.75× | 5.43× → 5.43× |
| **Qwen3.8-27B** | **5.44–5.45× → 5.56×** | **4.53× → 4.77×** |

The movement is concentrated at C10 and runs both ways — Qwen3.8-27B compresses
*better*. A ratio is bytes stored, so a different ratio at the same level means
the adaptive ladder selected different formats: something in the compression
policy or its per-model thresholds changed between the builds, and which change
is **not established**. Throughput does not track it: prefill is higher in the
second sweep on almost every cell, while decode is mixed (Qwen3-8B's widest
point 460.1 → 432.3, Qwen3.8-27B's 458.1 → 436.7, against Flash-Next's
+26–32%).

### The validation column is weaker than the throughput column

`CoherenceCheck` asks whether output is non-degenerate — at minimum, that it is
20% alphanumeric — not whether it is correct. It catches a model emitting
punctuation, and it does not catch fluent nonsense. The filler is also eight
rotating templates of dry technical prose, far from any instruction-tuned
model's training distribution.

**Throughput and compression figures are unaffected** — those tokens were
genuinely processed — but this document should not be read as evidence about
output *quality* at depth. `StoryRewrite`, which the flagship's rewrite curve
uses, is the stronger check: it is a concrete retrieval-and-rewrite task with a
verifiable answer.

### The quantized read multiplier is understood but not reduced

The ~2× per-position cost of a quantized KV read at decode (§3.5) is the largest
lever remaining for the models that cannot use selection, and it is parked
rather than solved.

What is established: it is not one bad codec — the whole C10 tier measures
1.26–1.71× of BF16 in isolation and the adaptive mix 1.6×, and `ncu` shows the
kernel latency-bound in both modes (C10 at 12% DRAM / 39% L1/TEX / 37% compute;
BF16 at 58% DRAM). Compression removes the bandwidth it is supposed to remove.
The quantized path issues 3.5× the global-load instructions and 2.9× the shared
loads of the float path, because quantized blocks are addressed per dim through
the block path while float takes a vector load — but **instruction counts are
not latency here**: the dequant runs on CUDA cores alongside tensor-core work,
so a higher count can be fully hidden, and neither path is near a roofline.
Which of the block path's fixed costs actually sits on the critical path — the
K-side shared-memory stage and read-back, the V-side shared `atomicMax` and its
barrier, or the all-or-nothing warp votes that make one quantized palette cost
what four would — is **not established**, and would need per-stall attribution
rather than the throughput counters collected so far.

`candle-examples/examples/decode_ab` is the harness for this work: CUDA-event
kernel timing with an FP32 golden gate, a `--formats c10` group covering the
level's whole candidate set, and an adaptive row (`rq-adaptive-L10`) for the
mixed case that uniform rows cannot represent.

### Q0_V decodes at 7.35× BF16, and cannot simply be dropped

The C10 tier's costliest format by a wide margin. In a uniform arena it decodes
at **332.9 µs against 45.3 µs for BF16 — 7.35×**, where every other candidate at
that level sits between 1.26× and 1.71×; its per-element `__constant__` table
lookups serialise across a warp whose lanes are each on a different block.

It nonetheless has to stay a candidate, which is the useful part of the finding.
Removing it from C6/C8/C9/C10 leaves the compression ratio **identical** —
7.10× on Qwen3.5-35B — because those blocks fall back to Q0_X at the same two
bytes; but the reconstruction differs, and three models lose output validation
at C10 under high concurrency: Qwen3.5-35B (31/32 and 63/64 sessions),
Qwen3.8-Flash-Next (7/8), Qwen3.6-35B (60/64 against its own standing 61/64).
Restoring it returns all three to their prior state exactly, reproducibly.

So the cost is a decode-path wart to fix in the kernel, not a candidate to
retire. The measurement trap is worth recording alongside it: an adaptive C10
cache on the decode harness measures 73.6 µs with Q0_V and 73.2 µs without,
which reads as "never selected" — but that fixture is **synthetic** K/V, and on
real model KV the selector plainly does pick it. A uniform-format bench cannot
answer a question about an adaptive cache, and a synthetic fixture cannot answer
one about production selection.

What this document has not measured, and what it would take to measure it, is
§7.

---

## 5. Provenance

Every row measured in each sweep — including the ones the tables above omit —
is beside this file, one TSV per sweep, all with the columns `test, label,
depth, prompt_tokens, mode, int8, contexts, valid, prefill_tps, decode_tps,
quantized_pct, compress, peak_tokens`, scraped from the run logs:

| File | Machine · Sweep | Rows |
|---|---|---:|
| `performance_rtx_pro_5000_72gb_rows.tsv` | 72 GB · 2026-09-03 — depth and width | 146 |
| `performance_rtx_pro_5000_72gb_rows_2026-09-13.tsv` | 72 GB · 2026-09-13 — width only, build `2c5f065c` + working tree | 171 |
| `performance_rtx_3090_24gb_rows.tsv` | RTX 3090 · 2026-09-14 — width gate sweep | 276 |
| `performance_rtx_pro_5000_72gb_rows_2026-09-15_run1.tsv` | 72 GB · 2026-09-15 — width only, build `23623c6b`, run 1 | 171 |
| `performance_rtx_pro_5000_72gb_rows_2026-09-15_run2.tsv` | 72 GB · 2026-09-15 — width only, build `23623c6b`, run 2 | 171 |
| `performance_rtx_4090_mobile_16gb_rows.tsv` | RTX 4090 Mobile · 2026-09-30 — width gate sweep, build `bf291341c` | 187 |
| `performance_rtx_3090_24gb_rows_2026-09-30.tsv` | RTX 3090 · 2026-09-30 — Flash-Next gate only, build `e596fad8d` | 10 |
| `performance_rtx_3090_24gb_rows_2026-10-04.tsv` | RTX 3090 · 2026-10-04 — width gate sweep, build `8d061e4fc` (baseline: `baseline_rtx_3090_24gb_2026-10-04.md`) | 188 |
| `performance_rtx_3090_24gb_rows_2026-10-04_live_dispatch.tsv` | RTX 3090 · 2026-10-04 — width gate sweep, live MoE dispatch (`moe_live_dispatch_rtx_3090_24gb_2026-10-04.md`) | 188 |
| `performance_rtx_3090_24gb_rows_2026-10-05.tsv` | RTX 3090 · 2026-10-05 — width gate sweep, build `5776799ac`; §3.8 (`sweep_rtx_3090_24gb_2026-10-05.md`) | 193 |
| `performance_rtx_4090_mobile_16gb_rows_2026-10-10.tsv` | RTX 4090 Mobile · 2026-10-10 — width gate sweep, build `21c36efbd`; §3.9 (`sweep_rtx_4090_mobile_16gb_2026-10-10.md`) | 192 |

A † cell in §3.6 *Width* or §3.7 is the 72 GB 2026-09-13 file's value, a ◆ cell
the higher of the two 2026-09-15 files' values; every other 72 GB width cell is
the first file's. The 2026-09-15 files list gates in sweep order, which puts
Llama-2-7B before Qwen3-8B and Qwen3.8-27B before Qwen3-30B-A3B. The 3090 and
4090 Mobile TSVs hold the width-sweep axis only — their `depth` column is blank
and `prompt_tokens` is `~700`, the gate's fixed prompt. The 4090 Mobile's engine
probes (§3.9) are not rows of that file: they report story, efficiency and
uptake rather than a ladder, and §3.9 carries them; the same holds for the
3090's three probes, which §3.8 carries. Reproduce any row with the
command in its test's `#[ignore]` attribute.

| Table | Test |
|---|---|
| §3.2, §3.3, §3.4 | each model's `long_context_*` gate (72 GB) |
| §3.5 | the same gates, C-mode rows (72 GB) |
| §3.6 Coherence | `quantized_qwen38_moe::tests::profile_decode_vs_depth` (72 GB) |
| §3.6 Rewrite | `quantized_qwen38_moe::tests::profile_story_rewrite_vs_depth` (72 GB) |
| §3.6 Width, §3.7 | `test_parallel_batched_forwarding*` (72 GB) |
| §3.8 | `test_parallel_batched_forwarding*` and `kv_fragmentation::{qwen3_30b_a3b_q4, qwen36_35b_a3b_q4, qwen38_flash_next}` (RTX 3090) |
| §3.9 | `test_parallel_batched_forwarding*` (RTX 4090 Mobile, 2026-10-10) and `kv_fragmentation::{qwen3_30b_a3b_q4, qwen38_flash_next}` (RTX 4090 Mobile, 2026-09-30) |

All runs, in every sweep and on all three cards, were strictly sequential — one
`cargo test` invocation per model, so exactly one model was ever resident and no
run's VRAM sizing was perturbed by another's. Run-to-run variation is 1–4% on
the width ladder and ~5% on the depth sweep, which is the noise floor any
comparison in this document has to clear; differences smaller than that are not
claimed as results. The 3090's §3.8 and the 4090 Mobile's §3.9 cells are single
measurements, so that floor applies to each on its own rather than to a
best-of-two.

---

## 6. External reference figures

Published throughput for the same models — or the nearest published proxy — on
cards comparable to the fleet's, from other engines (llama.cpp, ik_llama.cpp,
vLLM, SGLang, KTransformers, ExLlamaV2, TensorRT-LLM), collected 2026-09-30 and 2026-10-05 for
comparison. **None of these numbers was measured here**; every one is quoted
from the source cited against it in §6.8, and each carries that source's own
conditions. Our own figures are set beside them only where the machine class
matches, and marked **(ours)**.

### 6.1 How to read these figures

- **Different workloads.** llama.cpp's `llama-bench` reports `pp512` (a
  512-token prompt) and `tg128` (128 generated tokens) at batch 1 with an empty
  or stated-depth cache; our width gate prefills a ~700-token prompt per session
  and decodes a story rewrite (§2.2). The two are the same order of work, not the
  same work. vLLM/SGLang "aggregate" figures are summed over N concurrent
  requests, like our ×N decode column.
- **Speculative decoding is marked.** A row with **MTP** decodes with the model's
  draft head, which multiplies decode by the acceptance rate; compare it only
  with other speculative rows. Our gate harness speculates by default
  (`DraftBudget::Adaptive`) for any model that carries a draft head, so our
  Flash-Next rows are speculative and marked; our other rows are marked only
  where the checkpoint is known to carry one.
- **Proxies are marked.** A GPU or model in *italics* stands in for the fleet's:
  no source measured the **RTX 4090 Laptop 16 GB** or the **RTX PRO 5000
  Blackwell 72 GB** with these models, so those columns lean on the desktop
  RTX 4090 / 4080 / 4070 Ti Super / 5080 (16 GB class) and on the RTX 5090 /
  RTX PRO 6000 Blackwell (Blackwell class).
- **KV cache** is `f16` unless the row says otherwise; `n/s` = not stated by the
  source (llama.cpp's default is f16).
- **Provenance quality.** Most sources are community reports without controlled
  setups, many on engine builds that have since moved. Quotes in §6.8 were
  collected through a page-extraction tool and may be close paraphrases; the
  numbers are the sources' own, but re-open the URL before citing a figure
  anywhere that matters. Rows read from bug reports, estimates or chart-reads
  are excluded unless marked.

### 6.2 Small dense models

**Llama-2-7B** — the most-benchmarked model in the fleet; `llama-2-7b.Q4_0`,
llama.cpp CUDA, pp512 / tg128, batch 1, f16 KV unless stated:

| GPU | Engine | Weights | KV | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---:|---:|---|
| RTX 3090 | llama.cpp CUDA, FA off / on | Q4_0 | f16 | 5,174.69 / 5,560.06 | 158.16 / 161.89 | E1 |
| RTX 3090 | llama.cpp Vulkan | Q4_0 | f16 | 4,666.15 | 164.05 | E2 |
| *RTX 3090 Ti* | ExLlamaV2 | EXL2 4.0 bpw | f16 | — | 185 | E3 |
| *RTX 3090 Ti* | llama.cpp (2023-12) | Q4_0 | K f16 / **K q8_0** / **K q4_0** | 3,853.05 / 3,504.47 / 3,461.67 | 123.62 / 65.26 / 65.44 | E6 |
| *RTX 4080 16 GB* | llama.cpp CUDA, FA off / on | Q4_0 | f16 | 8,031.64 / 9,205.93 | 142.49 / 143.47 | E1 |
| *RTX 4090 24 GB* | llama.cpp CUDA, FA off / on | Q4_0 | f16 | 11,992.70 / 14,770.63 | 186.21 / 188.96 | E1 |
| *RTX 4090 24 GB* | ExLlamaV2 | EXL2 4.0 bpw | f16 | — | 211 | E3 |
| *RTX 4090 24 GB* | LMDeploy TurboMind (1-token prompt, 512 out) | W4A16 | n/s | — | 206.4 | E5 |
| *RTX 4090 24 GB* / *RTX 3090 Ti* | MLC LLM / ExLlamaV2 / llama.cpp (2023-10; short prompt, 256 out) | 4-bit | n/s | — | 204.8 / 177.46 / 151.1; 186.7 / 161.67 / 144.93 | E4 |
| *RTX 5090 Laptop 24 GB* | llama.cpp CUDA, FA off / on | Q4_0 | f16 | 6,667.06 / 7,641.89 | 156.49 / 158.14 | E1 |
| *RTX 5090* | llama.cpp CUDA, FA off / on | Q4_0 | f16 | 14,073.41 / 14,970.15 | 290.02 / 300.40 | E1 |
| *RTX PRO 6000 Blackwell* | llama.cpp CUDA, FA off / on | Q4_0 | f16 | 14,854.63 / 16,618.98 | 274.20 / 281.11 | E1 |
| RTX 3090 **(ours, §3.8)** | this engine, BF16 ×1 / BF16 ×48 | Q4_0 | BF16 | 4,966.8 / 4,600.2 | 120.6 / 934.9 (aggregate) | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, BF16 ×1 / BF16 ×48 | Q4_0 | BF16 | 5,864.4 / 6,322.2 | 89.3 / 904.6 (aggregate) | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, BF16 ×1 / BF16 ×48 | Q4_0 | BF16 | 7,974.8 / 11,353.2 | 163.4 / 1,638.4 (aggregate) | — |

**Llama-3.2-3B:**

| GPU | Engine | Weights | KV | Workload | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---:|---:|---|
| *RTX 4090 24 GB* | llama.cpp CUDA 12.9 | Q4_K / Q4_0 / Q8_0 | f16 | 1-token step | — | 328.89 / 331.87 / 212.01 | E8 |
| *RTX 4090 24 GB* | ONNX Runtime GenAI + DirectML | AWQ INT4 | n/s | 100 in / 100 out, batch 1 / 4 | — | 253 / 615 (incl. TTFT) | E7 |
| *RTX 4090 24 GB* | same | AWQ INT4 | n/s | 4,000 in / 100 out, batch 1 / 4 | — | 165 / 251 | E7 |
| *RTX 5090* | llama.cpp CUDA 12.9 | Q4_K / Q4_0 / Q8_0 | f16 | 1-token step | — | 434.51 / 454.01 / 301.50 | E8 |
| *RTX PRO 6000 Blackwell* | llama.cpp | Q4_K_M | f16 | pp512 / tg128 | 21,970.62 | 405.95 | E9 |
| *RTX PRO 6000 Blackwell* | llama.cpp, FA on | Q4_K_M | f16 | tg128 | — | 464.85 | E10 |
| *RTX PRO 6000 Blackwell Max-Q* | llama.cpp | Q4_K_M | f16 | pp8096 / tg128 | 16,879.10 | 426.27 | E11 |
| *RTX 4090 Laptop 16 GB* — ***Llama-3.2-1B*** (low trust) | LocalScore (llamafile) | Q4_K_M | n/s | not stated | 11,965 | 207 | E12 |
| RTX 3090 **(ours, §3.8)** | this engine, F16 ×1 / C8 ×10 (no flash-attn) | Q4_K_M | F16 / C8 | ~700-tok prompt | 10,508.9 / 9,966.1 | 183.4 / 789.1 | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, F16 ×1 / C8 ×10 | Q4_K_M | F16 / C8 | ~700-tok prompt | 13,319.6 / 14,240.2 | 138.5 / 688.7 | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, C0 ×1 / C8 ×10 | Q4_K_M | C0 / C8 | ~700-tok prompt | 20,789.1 / 25,512.0 | 191.2 / 1,152.2 | — |

No source measured Llama-3.2-3B on an RTX 3090.

**Qwen2-0.5B and Qwen3.5-0.8B.** Neither Qwen2-0.5B itself nor Qwen3.5-0.8B on a
target card was found; the nearest published rows:

| GPU | Model | Engine | Weights | Workload | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---:|---:|---|
| *RTX 4080 16 GB* | *Qwen2.5-0.5B-Instruct* | llama.cpp CUDA | Q4_K_M | pp512 / tg128 | 39,230.93 | 496.01 | E13 |
| *RTX 4090 24 GB* | *Qwen2.5-0.5B* | vLLM, `vllm bench serve`, cold prefix cache | BF16 | 1,024 in / 64 out, concurrency 32 | — | 16,349.78 total tok/s | E14 |
| *RTX 5080 16 GB* | Qwen3.5-0.8B | llama.cpp | Q8_0 | tg128 | — | 460.86 | E16 |
| *RTX 5090* | Qwen3.5-0.8B | llama.cpp, MMVQ / MMQ kernel | NVFP4 | pp512 / tg128 | 23,799.93 / 32,858.97 | 392.92 / 389.75 | E15 |
| RTX 3090 **(ours)** | Qwen2-0.5B / Qwen3.5-0.8B | this engine, BF16 ×1 | Q4_0 / Q6_K | ~700-tok prompt | 33,891.2 / 32,462.0 | 529.1 / 273.9 | — |
| RTX 4090 Mobile **(ours)** | Qwen2-0.5B / Qwen3.5-0.8B | this engine, ×1 | Q4_0 / Q6_K | ~700-tok prompt | 42,648.1 / 30,447.8 | 456.6 / 220.4 | — |
| RTX PRO 5000 **(ours)** | Qwen2-0.5B / Qwen3.5-0.8B | this engine, BF16 ×1 | Q4_0 / Q6_K | ~700-tok prompt | 54,944.4 / 43,068.5 | 560.3 / 347.1 | — |

### 6.3 Mid-size dense models

**Qwen3-8B** — Hardware Corner's llama.cpp context sweep (CUDA, `-fa 1`,
Q4_K, batch 1, KV dtype not stated) is the one consistent series across all
three card classes:

| GPU | 4K prefill / decode | 16K | 32K | 64K | 128K | Ref |
|---|---|---|---|---|---|---|
| RTX 3090 | 4,049.6 / 115.3 | 2,572.5 / 87.5 | 1,714.6 / 67.9 | 1,014.3 / 46.6 | 570.0 / 28.1 | E17 |
| *RTX 4070 Ti Super 16 GB* | 5,220.1 / 96.3 | 3,050.7 / 72.2 | 1,616.6 / 54.5 | 829.4 / 36.6 | — | E20 |
| *RTX 4080 16 GB* | 6,177.9 / 102.7 | 3,809.8 / 77.9 | 1,968.4 / 59.0 | 937.9 / 39.0 | — | E19 |
| *RTX 4090 24 GB* | 9,250.5 / 141.3 | 5,530.5 / 108.0 | 3,560.1 / 82.3 | 2,028.6 / 56.1 | 1,059.9 / 33.8 | E18 |
| *RTX 5090* | 11,933.4 / 200.4 | 8,538.4 / 162.3 | 6,034.2 / 129.8 | 3,089.6 / 91.8 | 1,209.1 / 58.8 | E21 |
| *RTX PRO 6000 Blackwell* | 10,964.1 / 173.7 | 7,587.7 / 140.6 | 5,300.7 / 111.1 | 1,921.6 / 77.9 | 1,009.7 / 48.3 | E22 |

Batched: *RTX 5090*, vLLM 0.12, NVFP4 weights, 8 concurrent — aggregate decode
411 t/s at 8K and 232 t/s at 16K (E23). Ours at ×1 / widest: RTX 3090
4,914.3 / 85.6 and C8 ×10 383.8 aggregate; RTX 4090 Mobile 5,938.5 / 63.0 and C8
×10 308.4 aggregate; RTX PRO 5000 9,717.2 / 102.2 and C8 ×10 622.3 aggregate
(Q6_K weights, ~700-token prompt).

**Qwen3.5-9B.** No trustworthy measurement exists on any of the three card
classes. The nearest: *GB10 / DGX Spark*, llama.cpp Q4_K_M pp512 2,558.98 /
tg128 35.41 (E25). Ours: RTX 3090 5,804.9 / 146.1 at ×1;
RTX 4090 Mobile 6,983.6 / 74.4; RTX PRO 5000 10,192.3 / 199.8.

**Qwen3.8-27B** — well covered, much of it with MTP; batch 1 unless noted:

| GPU | Engine | Weights | KV | Context | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---:|---:|---|
| RTX 3090 | llama.cpp b10364 | Q4_K_S | n/s | 4K / 32K / 64K | 1,308.05 / 977.25 / 766.59 | 40.31 / 37.02 / 33.95 | E26 |
| RTX 3090 | llama.cpp b10217 | UD-Q4_K_XL | **q4_0** | 128,290-tok prompt | 705 | 36.88; 65.28 **MTP** | E27 |
| RTX 3090 | llama.cpp b10450 | Q4_K_M | **q8_0** | 131K | — | 41.3; 63.5 **MTP** | E28 |
| RTX 3090 | vLLM 0.30 + patches | INT8 Marlin | f16 | 1K prompt; 64 concurrent | ~1,850–1,940 | ~1,035 aggregate | E29 |
| *RTX 4090 24 GB* | llama.cpp b10364 | Q4_K_S | n/s | 4K / 32K / 64K | 2,962.59 / 2,367.35 / 1,918.13 | 46.16 / 42.17 / 38.41 | E26 |
| *RTX 4090 24 GB* | llama.cpp | Q4_K_M | n/s | 131K | — | 47.7; 76.3 **MTP** | E28 |
| *RTX 5090* | llama.cpp b10364 | Q4_K_S | n/s | 4K / 32K / 128K | 3,749.51 / 1,146.03 / 461.17 | 74.83 / 28.96 / 22.79 | E26 |
| *RTX 5090* | llama.cpp b10448, MTP n=2 | Q4_K_M | **f16 / q8_0 / q4_0** | 32K | — | 125.5 / 128.3 / 136.7 (MTP off 73.6) | E30 |
| *RTX 5090* | vLLM, NVFP4 | NVFP4 | **fp8** | 1 / 4 / 16 concurrent | — | 67 / 326 / 579 aggregate | E31 |
| *RTX PRO 6000 Blackwell* | vLLM 0.27.1 | FP8 | bf16 | 262,144 max | — | 46.8; 62.2 **MTP** | E32 |
| *RTX PRO 5000 **48 GB*** | vLLM | FP8 | bf16 | 200K | — | "approximately 80" (sub-version unstated) | E33 |
| RTX 3090 **(ours, §3.8)** | this engine, BF16 ×1 / C10 ×10 | Q6_K/Q8 | BF16 / C10 | ~700-tok prompt | 1,854.7 / 2,007.6 | 70.8 / 341.6 | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, BF16 ×1 / BF16 ×4 / C10 ×10 | Q6_K/Q8 | BF16 / C10 | ~700-tok prompt | 1,259.1 / 2,318.9 / 2,225.6 | 8.2 / 219.7 / 79.7 (aggregate at ×4, ×10) | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, BF16 ×1 / C10 ×40 | Q6_K/Q8 | BF16 / C10 | ~700-tok prompt | 3,282.2 / 3,420.7 | 75.4 / 734.3 | — |

### 6.4 Mid-size MoE models

**Qwen3-30B-A3B** (Qwen3-Coder-30B-A3B, same MoE shape, marked *Coder*):

| GPU | Engine | Weights | Offload | KV | Context / batch | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---|---:|---:|---|
| RTX 3090 | llama.cpp, `-fa 1` | Q4_K | none | n/s | 4K / 32K / 64K, batch 1 | 2,988.6 / 1,336.8 / 800.9 | 153.6 / 87.2 / 58.3 | E17 |
| *RTX 3090 Ti* | ik_llama.cpp sweep-bench | IQ4_K | none | f16 | up to 32K | "over 1600" | 105 | E34 |
| *RTX 4090 24 GB* | llama.cpp | Q4_K | none | n/s | 4K / 32K / 64K | 6,159.4 / 2,627.7 / 1,502.5 | 207.3 / 105.1 / 68.2 | E18 |
| *RTX 4090 24 GB* | KTransformers 0.5.3 | BF16 | CPU/GPU hybrid, 2× EPYC 7C13 | n/s | batch 1 | — | 19.4 | E35 |
| *RTX 4090 24 GB* | vLLM (*Coder*) | AWQ | none | **fp8** | 8K | — | 2,259 aggregate | E36 |
| *RTX 5090* | llama.cpp b8189 | Q4_K | none | n/s | 4K / 32K / 128K | 7,093.0 / 4,210.0 / 985.0 | 226.1 / 143.1 / 76.8 | E21 |
| *RTX 5090* | Ollama (*Coder*) | Q5_K_M | none | **f16** / **q8_0** | 32K / 64K | — | 231 / 223 | E37 |
| *RTX 5090* | vLLM (*Coder*) | AWQ | none | n/s | 16 / 24 concurrent | — | 1,157 / 1,186 aggregate | E38 |
| *RTX PRO 6000 Blackwell* | vLLM (*Coder*) | FP8 | none | full | 1K / 32K / 128K, 4 concurrent | 36,943 peak | 333.7 / 137.2 / 27.9 aggregate | E39 |
| *RTX PRO 6000 Blackwell* | vLLM (*Coder*) | AWQ | none | **fp8** | 8K, ~400 concurrent | — | 8,425 aggregate | E36 |
| RTX 3090 **(ours, §3.8)** | this engine, BF16 ×1 / BF16 ×10 / Q8_0 ×20 | Q4_K_M | expert cache | BF16 / Q8_0 | ~700-tok prompt | 4,117.2 / 6,821.5 / 6,671.8 | 69.9 / 400.5 / 377.7 aggregate | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, BF16 ×1 / Q8_0 ×20 | Q4_K_M | experts streamed | BF16 / Q8_0 | ~700-tok prompt | 1,807.3 / 6,139.1 | 26.7 / 299.6 aggregate | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, BF16 ×1 / BF16 ×10 / Q8_0 ×20 | Q4_K_M | expert cache | BF16 / Q8_0 | ~700-tok prompt | 10,450.6 / 16,966.7 / 16,668.1 | 136.6 / 939.2 / 874.3 aggregate | — |

No source ran Qwen3-30B-A3B with its experts offloaded on a 16 GB card.

**Qwen3.5-35B-A3B and Qwen3.6-35B-A3B**, batch 1 unless noted:

| GPU | Model | Engine | Weights | Offload | KV | Context | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---|---|---:|---:|---|
| RTX 3090 | 3.5 | llama.cpp, `-fa 1` | MXFP4 | none | n/s | 4K / 32K / 128K | 2,622.1 / 2,121.6 / 1,288.9 | 111.2 / 101.2 / 79.4 | E17 |
| RTX 3090 | 3.5 | llama.cpp / Ollama | Q4_K_M | none | **q8_0** 65K / **f16** 32K | short prompt | — | 142.2 / 98.9 (different engines) | E40 |
| RTX 3090 | 3.6 | llama.cpp CUDA | UD-IQ4_NL_XL | none / **FFNs of 10 layers on CPU** | n/s | 89,600 / 262,144 | 3,360.4 / 1,153.5 | 139.6 / 89.1 | E41 |
| RTX 3090 | 3.6 | llama.cpp b9008 | Q4_K_M / Q5_K_M | none / **`--n-cpu-moe 10`** | **q8_0** | 16K / n/s | — / 598.13 | 148.64 / 105.05 | E42 |
| RTX 3090 | 3.6 | llama.cpp b10088 | UD-Q4_K_M | none | n/s | 8K | 3,674 | 157.66 | E43 |
| RTX 3090 | 3.5 | llama.cpp b8155, `llama-bench -fa 1` | MXFP4 | none | n/s | pp512 at depth 1K / 4K / 10K | 2,310.5 / 2,193.1 / 2,080.9 | 98.45 / 96.44 / 94.22 | E24 |
| *RTX 5070 Ti 16 GB* | 3.6 | llama.cpp | 10.88 GB quant | none (`-ngl 40`) | **q8_0** | 32,768 | 407 | 121 | E44 |
| *RTX 3060 12 GB* | 3.6 | llama.cpp b10088 | UD-Q4_K_M | **`--n-cpu-moe 24`** | n/s | 8K | 413 | 38.9 | E43 |
| *RTX 4080 16 GB* / *RTX 4090 24 GB* | 3.6 | llama.cpp (ByteShape) | IQ3_S / IQ4_XS | none | n/s | n/s | — | 183.29 / 214.54; **MTP** 249.33 / 285.53 | E45 |
| *RTX 5090* | 3.5 | llama.cpp | UD-Q4_K_XL | none | **q8_0** | 512–32,768 | 6,461–6,960 | 194.0 | E46 |
| *RTX 5090* | 3.6 | NInfer (C++/CUDA), MTP3 | INT group-64 | none | INT8 group-64 | 8,192 generated; C=1 / 2 / 4 / 8 | — | 593.0 / 877.7 / 1,166.0 / 1,313.8 aggregate | E86 |
| *RTX PRO 6000 Blackwell* | 3.5 | vLLM | FP8 | none | full | 1K / 256K; 10 concurrent | 34,509 peak | 160.3 / 97.7; 598.5 aggregate | E47 |
| *RTX PRO 6000 Blackwell* | 3.6 | vLLM | FP8 | none | full | 1K / 32K / 256K; 5 concurrent | 41,105 | 196.4 / 183 / 116.3; 449.0 aggregate | E48 |
| RTX 3090 **(ours, §3.8)** | 3.5 / 3.6 | this engine, BF16 ×1 / C10 ×16 | Q6_K | expert cache | BF16 / C10 | ~700-tok prompt | 1,610.8 / 1,094.2 (×1) | 78.1 / 228.0 (×1); 1,228.8 / 1,346.3 (×16) | — |
| RTX 4090 Mobile **(ours, §3.9)** | 3.5 / 3.6 | this engine, BF16 ×1 / C10 ×16 | Q6_K | experts streamed | BF16 / C10 | ~700-tok prompt | 571.4 / 439.1 (×1) | 24.2 / 71.6 (×1); 330.8 / 336.7 (×16) | — |
| RTX PRO 5000 **(ours, §3.7)** | 3.5 / 3.6 | this engine, BF16 ×1 / C10 ×64 | Q6_K | expert cache | BF16 / C10 | ~700-tok prompt | 11,034.6 / 10,582.4 (×1) | 230.2 / 476.6 (×1); 2,869.9 / 2,761.0 (×64) | — |

### 6.5 Large MoE models with expert offload

Both of the fleet's large MoEs have direct published results on single consumer
and workstation cards, all with the routed experts in host RAM. **The public
Qwen3.8-Flash-Next is 125B + 51B n-gram + 4B MTP, 6B active** (§3.1, E60).

**Qwen3.8-Flash-Next**, batch 1:

| GPU | Host | Engine | Weights | Offload | KV | Context | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---|---|---:|---:|---|
| RTX 3090 | Ryzen 9 5950X, 128 GB DDR4 | llama.cpp PR #27742 | UD-Q4_K_XL | `--fit on` | **q8_0** | 130K | ~340 | 15 | E51 |
| *2× RTX 3090, PCIe 3.0* | 2× E5-2696 v4, 188 GiB DDR4-2133 | llama.cpp + expert-cache PR | UD-Q6_K_XL | all 48 expert layers in host RAM | f16 | 26K / 131K depth | 138 | 23.7 / 17.2 | E54 |
| *RTX 5080 16 GB* | Ryzen 7 9800X3D, 64 GB DDR5 | llama.cpp b10819 | UD-IQ3_XXS | n-gram table via mmap/NVMe; ngram speculation | **q8_0** | 32K | 14.5–20.5 | 27.5–29 | E51 |
| *RTX 4090 24 GB* | Core Ultra 7 270K, 96 GB DDR5 | llama.cpp | UD-IQ3_XXS | `--fit on` | **q8_0** | 6K / 90K / 110K | ~1,360 / ~905 / ~863 | ~30 / ~22 / ~19.5 | E49 |
| *RTX 4090 24 GB* | same | same | same | same | **q4_0** vs **q8_0** | 32K | — | 28.9 vs 30.1 | E49 |
| *RTX 4090 24 GB* | n/s | n/s | n/s | n/s | f16 ("no kv cache quantization") | 250K | 364 | 21 (no MTP) | E50 |
| *RTX 5090* | Ryzen 7 9700X, 128 GB DDR5 | llama.cpp PR #27742 | UD-Q2_K_XL | `-ncmoe 24` | n/s | 65,536 | 1,429 | 48.02 short; 43.28 after ~16K | E52 |
| *RTX PRO 6000 Blackwell* | n/s | n/s (Unsloth) | n/s | n/s | n/s | n/s | — | 100; 170 **MTP** | E53 |
| *RTX 5090* | Core Ultra 9 285K, 64 GB, PCIe 5 x16 | Strata 0.1.29 (d6708a4) | IQ2_XS | 17,463 expert slots (23.4 GiB, 71% of the experts) in VRAM, misses on the CPU | INT8 | 4K / 32K / 128K | 4,269.8 / 5,543.2 / 5,778.7 | 179.4 / 175.7 / 165.0 (**MTP**) | E91 |
| *RTX 5070 12 GB* | Ryzen 5 7600, 64 GB DDR5-5200 | Strata 0.1.36 | Q2_0 / IQ3_S | ~4.8 GiB expert cache (14% of the experts) in VRAM, misses on the CPU | n/s | 4K answers, 32K prompt | 2,650 / 1,620 | 94 / 53 (**MTP**) | E90 |
| *RTX 5070 12 GB* | same | Strata 0.1.36, HTTP server, 4 concurrent | Q2_0 | same | n/s | 32K | — | 63.1 aggregate (70.7 for the same four requests run one at a time) | E92 |
| RTX 3090 **(ours, §3.8)** | i7-10700K, 64 GB | this engine, BF16 ×1 warm / BF16 ×8 / C10 ×8 | Q2_KO experts | experts streamed VRAM→pinned RAM | BF16 / C10 | ~700-tok prompt | 556.5 / 2,051.7 / 2,078.9 | 138.1 / 477.2 / 361.6 (**MTP**) | — |
| RTX 4090 Mobile **(ours, §3.9, `21c36efbd`)** | Core Ultra 9 185H, 32 GB | this engine, BF16 ×1 cold / ×1 warm / ×8 / C5 ×8 / C10 ×8 | Q2_KO experts | experts streamed VRAM→RAM→NVMe | BF16 / C5 / C10 | ~700-tok prompt | 145.4 / 269.1 / 1,127.5 / 1,127.6 / 1,128.7 | 33.4 / 34.9 / 108.5 / 107.9 / 101.3 (**MTP**) | — |
| RTX PRO 5000 **(ours, §3.6)** | Ryzen 9 9950X3D, 189 GB | this engine, ×1 warm / ×8 | Q4_KOEXP | expert cache, mostly resident (20.6–24.6K of 25,088 experts, misses from pinned RAM) | BF16 | ~700-tok prompt | 4,688.0 / 6,938.7 | 295.2 / 1,008.2 (**MTP**) | — |
| RTX PRO 5000 **(ours, Strata's workload)** | same | this engine through `ConversationEngine` (`kv_fragmentation::qwen38_flash_next_strata`), ×1 median of 3 | Q4_KOEXP | same | C5 | 4K / 32K / 128K | 6,019.5 / 5,397.9 / 4,776.9 | 156.6 / 152.5 / 141.6 (**MTP**, ceiling 4) | — |

**Flash-Next at an equal expert footprint.** Every run below carries 2-bit
experts of about the same size — 1.32 MiB a slot here, ~1.38 MB a slot in Strata
(Q2_0 34 GB of experts, IQ2_XS 36 GB; our Q2_KO 31.0 GiB over 24,064 evictable
experts) — with MTP on, so what separates the rows is how much of the expert set
sits in VRAM and how misses are served:

| Engine | GPU, host | Experts in VRAM | Misses served by | Decode ×1 t/s | Ref |
|---|---|---|---|---:|---|
| Strata 0.1.26 | RTX 5070 12 GB (PCIe 5.0), Ryzen 5 7600, 64 GB DDR5-5200 | ~3,500 slots, 4.8 GiB, 14% | CPU, from pinned RAM | 93.0 (4K) | E90 |
| this engine | RTX 3090 24 GB, i7-10700K, 64 GB, PCIe 3.0 | 8,677 → 11,772 slots, 11.2 → 15.2 GiB, 36–49% | PCIe upload | 138.1 (warm, ~700-token prompt) | I18 |
| Strata 0.1.26 | RTX 3090 24 GB (one of two in the host; PCIe Gen4 x16, 23–26 GB/s probed), EPYC 7453, 165 GiB | IQ3_XXS experts (43 GB, ~1.75 MB a slot), share not stated | CPU, from pinned RAM | 93.0 (1K) / 89.2 (4K) / 90.6 (32K) / 79.3 (128K), draft acceptance 0.74 | E93 |
| Strata 0.1.29 | RTX 5090 32 GB, Core Ultra 9 285K, 64 GB | 17,463 slots, 23.4 GiB, 71% | CPU, from pinned RAM | 179.4 (4K) | E91 |

Our 3090 decodes 48% faster than the 5070 (138.1 against 93.0) with about three
times its resident share, on a quarter of its link bandwidth: the 5070 is a PCIe
5.0 card on a Ryzen 5 7600 with DDR5-5200, our 3090 a PCIe 3.0 card on an
i7-10700K with DDR4, with a different miss path (CPU compute from pinned RAM
against our upload). Residency and the bus pull in opposite directions, so no
row here isolates the engine from the bus. The 5090's 179.4 has twice our
resident share on PCIe 5.0 and is not a like-for-like row either.

**The Blackwell pair is the comparable one, and on the same workload we trail.**
Strata's RTX 5090 32 GB (E91) and our RTX PRO 5000 72 GB are close hardware: the
same architecture on PCIe 5.0 ×16, at different bit widths and residency.
Strata's IQ2_XS experts are 36 GB in total with 71% of them in VRAM and the misses
computed on the CPU; our Q4_KOEXP experts are about twice the bytes per expert,
mostly resident, with misses copied from pinned RAM by the expert kernels. On
Strata's own benchmark requests, rebuilt request for request and run through the
engine as a daemon runs them (`kv_fragmentation::qwen38_flash_next_strata`,
2026-10-10, build `21c36efbd`: the same synthetic-module prompts cut to the token,
greedy, 256 tokens, one warm-up then the median of three, KV at C5), we decode
**156.6 / 152.5 / 141.6 t/s at 4K / 32K / 128K against Strata's 179.4 / 175.7 /
165.0** — 86–87% of its single-session decode — and prefill 6,019.5 / 5,397.9 /
4,776.9 against 4,269.8 / 5,543.2 / 5,778.7, ahead at 4K and level at 32K. The
remaining gap is the cost of a verify step, not the drafter: our MTP head commits
2.7–3.2 tokens a step on these prompts against Strata's ~2.7 (its `engine.log`),
and a step costs us 17.1–19.3 ms against its ~14–15. Its prompt-lookup ("suffix")
drafter fired 0–3 times per 256-token answer there and is not what separates the
two. The 295.2 above is our story-rewrite row, a different and easier task for the
drafter, and is not comparable to the 179.4. Our ×8 aggregate on the PRO 5000 is
1,008.2; Strata publishes no 4-bit row (its largest, IQ3_S, is 50 GB) and no
batched run on a Blackwell card.

**Strata has never run on our 3090 machine** (PCIe 3.0, i7-10700K, 64 GB DDR4),
so no row compares the two engines on it. The nearest is Strata's own RTX 3090
24 GB (E93): the same card, with a Gen4 link, 2.0× ours, an EPYC 7453 host and
3-bit experts. Its experts are ~30% larger than ours, so the same VRAM holds a
smaller share of them than our 36–49%. We decode 138.1 against its 93.0 at 1K and
90.6 at 32K (+48% and above), while its prefill is 869.7 at 1K and 2,160.4 at 32K
against our 556.5. The card, the quant class and MTP match; the bus (Gen4 against
Gen3), the host (EPYC against an i7-10700K), the prompt (ours ~700 tokens) and the
miss path (CPU compute against PCIe upload) differ.

**DeepSeek-V4-Flash** (284B / 13B active), batch 1:

| GPU | Host | Engine | Weights | Offload | KV | Context | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---|---|---:|---:|---|
| RTX 3090 | 128 GB DDR5-5600 | llama.cpp | UD-IQ3_S (0731) | `--n-cpu-moe 39` | n/s | 384K configured | — | 12.5 | E56 |
| RTX 3090 | 192 GB DDR4-2400 | llama.cpp-style backend | n/s | experts in RAM | n/s | n/s | — | 10–11 | E55 |
| *RTX 4090 24 GB* | n/s | GGUF | "Q2" (0731) | CPU-MoE | f16 ("no kv cache quantization") | 250K | "650+" | 12 | E57 |
| *RTX 4090 24 GB* | 2× Xeon Platinum 8488C | KTransformers 0.6.2 | MXFP4 | CPU experts | n/s | n/s | — | 18.5 | E35 |
| *RTX 5090 Laptop 24 GB* | 192 GB | llama.cpp b9977 | UD-Q4_K_XL | `-ot` experts→CPU | **q8_0** | 50,536 | 40–80 | 8 | E61 |
| *RTX 5090* | Threadripper 5965WX, 512 GB | llama.cpp | UD-Q8_K_XL (0731) | `--cpu-moe`, `-b/-ub 8192` | f16 | n/s | ~700 average | 18 | E58 |
| *RTX 5090* | 2× Xeon Gold 6138, 256 GB DDR4 | KTransformers + SGLang | INT4 (AMX) | all routed experts on CPU | **fp8** | 8,192 | — | 27.9–28.0 | E59 |
| *RTX 5090* | x86 AVX2, ≥ 200 GB | SGLang + KT-Kernel | MXFP4 | 10 GPU / 60 CPU experts | n/s | 16,384 | — | "20+" | E62 |
| *RTX PRO 6000 Max-Q* | EPYC 9374F | llama.cpp PR #24162 | n/s | `-cmoe` (all experts CPU) | default | 8K / 32K / 65K / 524K | 748.4 / 699.9 / 637.3 / 281.8 | 10.74 / 11.12 / 10.80 / 7.59 | E63 |
| RTX PRO 5000 **(ours, §3.7)** | Ryzen 9 9950X3D, 189 GB | this engine, ×1 warm / ×16 | MXFP4_KO | expert cache | BF16 | ~700-tok prompt | 319.0 / 1,177.1 | 18.9 / 114.4 aggregate | — |

### 6.6 KV-cache compression

Engine-level KV quantization, with the same setup measured with and without it:

| Engine / method | KV format | Model | GPU | Context / batch | Uncompressed | Compressed | Ref |
|---|---|---|---|---|---|---|---|
| llama.cpp mainline | f16 → q8_0 / q4_0 | Qwen3-8B Q4_K_M | *A100 80 GB* (reference) | decode at 8K / 32K / 64K | 134.4 / 105.0 / 81.6 | q8_0 110.2 / 68.0 / 44.7; q4_0 107.2 / 63.6 / 41.2 | E64 |
| same | same | same | same | prefill at 8K | 4,539 | q8_0 4,411 | E64 |
| same | same | same | same | peak memory at 32K | 9.63 GiB | q8_0 7.52 / q4_0 6.39 GiB | E64 |
| llama.cpp, FA | f16 → q8_0 / q4_0 | LLaMA-7B Q4_0 | not stated | pp512 / tg128 | 4,946.49 / 138.23 | q8_0 3,177.33 / 141.48; q4_0 3,128.80 / 141.26 | E65 |
| llama.cpp, FA | q8_0 vs q4_0 | Qwen3.6-27B | RTX 3090 | 49K–131K | — | q8_0 ~100 t/s prompt eval; q4_0 ~1,150 prompt / ~60 decode | E66 |
| llama.cpp, missing FA quant kernel | q8_0 K / q4_0 V | Gemma 4 12B | *RTX 5070 Ti Laptop 12 GB* | prefill | 2,361 (all FA quant kernels compiled) | 96 (CPU fallback) | E67 |
| llama.cpp, FA | four asymmetric pairs | Qwen3.8-27B | RTX 4090 24 GB | 143,360 | — | prefill 2,409–2,529 across pairs; 21.0–22.4 GiB total | E68 |
| llama.cpp fork, MMA-FA on Q4_0 tiles | q4_0 / q4_0 | Qwen3.8-27B IQ3_S | 2× RTX 3090 | 8K prefill; 1 request | 1,735.46 / 44.73 | 1,700.70 / 44.69 | E69 |
| custom engine (NInfer) | INT8 → E8 4-bit | Qwen3.8-27B | RTX 4090 24 GB | 262K | 134.2 decode | 126.6 decode | E70 |
| ExLlamaV2 | FP16 → Q4 / Q8 | Llama-3-8B | not stated | 20K | 36.24 | Q4 24.06; Q8 25.07 | E71 |
| vLLM | BF16 → FP8 | Llama-3.1-8B | *H100* (reference) | ~20K in / 2K out, 8 concurrent | 450.3 out tok/s | 517.5 (+14.9%); "2x KV-cache capacity" | E72 |
| vLLM TurboQuant | k8v4 / 3-bit | Qwen3-30B-A3B-2507 | *2× H100* (reference) | up to 256K, load | BF16 = 100% | 80% / 73% of BF16 throughput; FP8 ≈ BF16 | E73 |
| TensorRT-LLM vs vLLM | FP8 / INT8 | Llama-3.1-8B | *H100 PCIe* (reference) | up to 256/512 batch | BF16 | TRT-LLM up to 1.09× (prefill-heavy), 1.45× (decode-heavy); vLLM FP8 "did not improve throughput" | E74 |
| NVFP4 KV (TRT Model Optimizer) | FP8 → NVFP4 | Qwen3-Coder-480B | Blackwell (reference) | — | FP8 | "up to 50%" less memory than FP8; "up to 3x better TTFT" | E75 |
| SGLang | BF16 → FP4 | 235B+ models | n/s | — | — | "~3.56× more tokens than BF16" | E76 |
| llama.cpp mainline, Hadamard rotation of Q/K/V (merged 2026-04-01) | f16 → q4_0 / q8_0 | Qwen3 0.6B | n/s | PPL | f16 baseline; q4_0 62.0161; q8_0 13.9115 (rotation off) | q4_0 46.2503; q8_0 13.6713 (rotation on); AIME25 q4_0 0.0% → 21.7% | E87 |
| SGLang | BF16 → FP8 | Qwen3.5-122B-A10B | 8× RTX PRO 6000 (SM120) | burst serving | correct | 1,985 tok/s burst; output silently corrupted ("exclamation marks, repetition"), no crash | E88 |
| vLLM FlashInfer FA2, split-KV gate | → NVFP4 | Gemma 3/4, MTP | 2× RTX 5060 Ti 16 GB (SM120) | 185K | ~16.6 engine steps/s, gate off | ~3.2 engine steps/s, gate on as shipped (5.2×) | E89 |

Research methods, as their papers report them:

| Method | KV format | Model | GPU | Result | Ref |
|---|---|---|---|---|---|
| KIVI | 2-bit (K per-channel, V per-token) | Llama-2-7B | *A100 80 GB* | 2.35×–3.47× throughput; 2.6× less peak memory; up to 4× larger batch | E77 |
| KVQuant | nuq4 (kernels); nuq2 | LLaMA-2-7B-32K | RTX A6000 (GA102, the 3090's die) | K 219.4 → 126.3 µs, V 203.7 → 124.5 µs at 16K | E78 |
| Atom | W4A4 + INT4 KV | Llama-7B | RTX 4090 24 GB | up to 7.73× vs FP16, 2.53× vs INT8 (weights, activations and KV together) | E79 |
| TailorKV | 1-bit layers + offload | Llama-3.1-8B | RTX 3090 | 128K context served on one 3090 at 82 ms/token decode | E80 |
| MagicPIG | LSH sampling (sparse) | Llama-3.1-8B-Instruct | RTX 4090 24 GB | 54 ms decode at 96K; 3.3× throughput | E81 |
| CommVQ | 2-bit / 1-bit VQ | LLaMA-3.1-8B | RTX 4090 24 GB | 87.5% KV reduction; 128K context on one 4090 with 1-bit | E82 |
| BitDecoding | 4-bit / 2-bit KV on tensor cores | LLaMA-3.1-8B | Blackwell / Hopper / Ampere | average 7.5× decode kernel vs FP16 FlashDecoding-v2; up to 8.6× (Blackwell NVFP4) | E83 |
| Rethinking KV compression (MLSys'25), LMDeploy | KIVI-4 / GEAR-4 / H2O / StreamingLLM | (Table 3) | 4× RTX A6000 | decode 0.98× / 1.02× / 1.34× / 1.34×; prefill 1.06× / 0.86× / 0.58× / 0.95× of baseline | E84 |
| TurboQuant | 3.5 / 2.5 bits per channel | — | — | "absolute quality neutrality" at 3.5 bits; "marginal" loss at 2.5 | E85 |

### 6.7 What the published figures say about ours

Read with §6.1's caveats; each point is a juxtaposition of published and measured
figures, not a controlled comparison.

- **Single-session dense decode is below llama.cpp's on the same class of
  card.** On the RTX 3090, for dense models both engines hold resident,
  llama.cpp decodes Qwen3-8B at 115.3 t/s at 4K (E17) against our 85.6 at ×1
  (Q4_K against our Q6_K), and Llama-2-7B at 158–162 (E1) against our 120.6 on
  the same Q4_0 file. Our one-context prefill is now the higher (Qwen3-8B 4,914
  vs 4,050). The width ladder is where this engine gains: its aggregate at
  ×10–×64 is the figure the batch-1 sources do not report.
- **With experts streamed, Qwen3.5-35B's single-session decode is the weakest
  number here.** On the RTX 3090 our Q6_K Qwen3.5-35B-A3B streams its experts and
  decodes at 78.1 t/s at ×1, where llama.cpp holds an MXFP4 quant resident at
  111.2 (E17) and 98.45 at depth 1K (E24); at ×16 ours reaches 1,228.8 aggregate.
  Qwen3.6-35B, with MTP, decodes 228.0 at ×1 on the same card, above llama.cpp's
  157.66 for a resident Q4_K_M (E43). On the RTX 4090 Mobile ours decodes 24.2 / 71.6 t/s at ×1
  (Qwen3.5 / Qwen3.6);
  published partial-offload runs on 12–16 GB cards reach 38.9 (RTX 3060 12 GB,
  `--n-cpu-moe 24`, E43), and a 16 GB card fully resident at a 10.88 GB quant
  121 (E44). At ×16 ours reaches 330.8–336.7 aggregate.
- **Flash-Next at one session now leads Strata at an equal expert footprint, and
  trails it on Blackwell; its width leads every published run.** Strata decodes
  93–94 t/s on a 12 GB RTX 5070 (PCIe 5.0) at Q2_0 with 14% of the experts in VRAM
  (E90), against our 138.1 at ×1 on the PCIe 3.0 RTX 3090 with 36–49% of ours
  resident — 2-bit experts of the same size on both, MTP on both, a quarter of the
  link bandwidth (§6.5). On the same card, a PCIe Gen4 RTX 3090 with an EPYC host,
  Strata decodes 93.0 at 1K and 90.6 at 32K on larger 3-bit experts (E93). Its
  prefill is 2,650 on the 5070 and 869.7–2,160.4 on that 3090, against our 556.5.
  **On its own 5090 workload, rebuilt and run through our engine on the 72 GB
  Blackwell card, we decode 141.6–156.6 t/s against its 165.0–179.4** (§6.5, *The
  Blackwell pair*): the like-for-like single-session row, and we trail it by
  13–14%, on step cost rather than drafting, while prefilling faster at 4K. The
  llama.cpp runs are far behind both: 15 t/s on an RTX 3090 with 128 GB of host
  RAM (E51, UD-Q4_K_XL, q8_0 KV at a 130K context) and 27.5–29 on an RTX 5080
  16 GB (E51) and 48.02 on an RTX 5090 (E52). Our 138.1 on the 3090 is above all
  of them, on Q2_KO experts and a ~700-token prompt. The one published Flash-Next
  aggregate on a single card is Strata's 63.1 at four sessions, below its own 70.7
  one at a time (E92); ours is 361.6 at C10 ×8 on the 3090 and 107.9 at C5 ×8 on
  the 16 GB laptop. On the 72 GB card our 295.2 t/s at ×1 with MTP (the story
  rewrite) is above the RTX PRO 6000's published 100 without MTP and 170 with it
  (E53, on an unstated artifact and task), and our ×8 aggregate is 1,008.2.
- **Quantized KV and decode depth.** llama.cpp's q8_0 KV costs Qwen3-8B 18% of
  decode at 8K and 45% at 64K (E64, A100); our C10 costs 3–31% at 32K and
  19–51% at 128K (§3.5) at 4.6–7.6× compression, against q8_0's ~1.9×. Both
  engines pay for quantized reads as the cache deepens; no source measured a
  4–7× KV format with its throughput on these cards. Since April 2026
  llama.cpp rotates Q/K/V before quantizing (E87), which lifts q4_0 KV quality
  well above the older figures; and the published SM120 runs of FP8 and NVFP4 KV
  show silent corruption (E88) and a 5.2× decode collapse at 185K (E89) with no
  error raised, so a compressed KV format is only as good as the output check
  that accompanies it.
- **FP8 KV in vLLM** is the published default for compressed KV: 2× capacity for
  +5–15% throughput at concurrency on H100 (E72), and sub-8-bit TurboQuant costs
  20–34% (E73). There is no FP8-KV A/B on the fleet's card classes.

### 6.8 References

All accessed 2026-09-30, except E24 and E86–E93, accessed 2026-10-05. Publication dates are the source's own where it gives
one; community posts are dated by their thread.

- **E1** — llama.cpp Discussion #15013, "Performance of llama.cpp on Nvidia
  CUDA" (opened 2025-08-01; entries undated).
  https://github.com/ggml-org/llama.cpp/discussions/15013 — e.g. "RTX 3090 …
  5174.69 ± 21.83 | 158.16 ± 0.21"; "RTX 4090 … 11992.70 ± 107.99 | 186.21 ±
  0.13".
- **E2** — llama.cpp Discussion #10879, "Performance of llama.cpp with Vulkan"
  (2024-12-18). https://github.com/ggml-org/llama.cpp/discussions/10879
- **E3** — turboderp, ExLlamaV2 README (undated).
  https://github.com/turboderp/exllamav2 — "Llama2 | EXL2 4.0 bpw | 7B | 185 t/s
  | 211 t/s" (3090 Ti | 4090).
- **E4** — sh1ng, llm-perf-bench (2023-10). https://github.com/sh1ng/llm-perf-bench
  — "Llama2-7B | RTX 4090 | 204.8 tok/sec | 177.46 tok/sec | 151.1 tok/sec";
  "decoding 256 tokens with a prompt 'What is the meaning of life?'".
- **E5** — LMDeploy v0.2.0 docs, "INT4 Weight-only Quantization and Deployment
  (W4A16)". https://lmdeploy.readthedocs.io/en/v0.2.0/quantization/w4a16.html —
  "Llama-2-7B-chat | 112.9 | 159.4 | 206.4".
- **E6** — llama.cpp PR #4312, "llama : support quantum K cache" (2023-12-06).
  https://github.com/ggml-org/llama.cpp/pull/4312 — "NVIDIA GeForce RTX 3090 Ti
  … q8_0 | tg 128 | 65.26 ± 0.35".
- **E7** — NVIDIA Technical Blog, "Llama 3.2 Full-Stack Optimizations Unlock
  High Performance on NVIDIA GPUs" (2024-11-19).
  https://developer.nvidia.com/blog/llama-3-2-full-stack-optimizations-unlock-high-performance-on-nvidia-gpus/
- **E8** — llama.cpp PR #26705, "CUDA: branchless Q4_K/Q5_K unpack…"
  (2026-08-07). https://github.com/ggml-org/llama.cpp/pull/26705 — "RTX 4090
  sm_89 … Q4_K | 1 | 328.89".
- **E9** — llama.cpp PR #22522, "Programmatic Dependent Launch (PDL)…"
  (2026-04-29). https://github.com/ggml-org/llama.cpp/pull/22522
- **E10** — llama.cpp PR #17795 (2025-12-05).
  https://github.com/ggml-org/llama.cpp/pull/17795 — "llama 3B Q4_K - Medium …
  tg128 | 464.85 ± 0.55".
- **E11** — llama.cpp PR #19053 (2026-01-23).
  https://github.com/ggml-org/llama.cpp/pull/19053 — "NVIDIA RTX PRO 6000
  Blackwell Max-Q Workstation Edition | llama 3B Q4_K_M | pp8096 | 16879.10".
- **E12** — LocalScore, Llama 3.2 1B leaderboard (undated; aggregated user
  submissions, low trust). https://www.localscore.ai/model/3
- **E13** — llama.cpp PR #12874, "llama-bench: enhance benchmark with improved
  token throughput measurements" (2025-04).
  https://github.com/ggml-org/llama.cpp/pull/12874 — "RTX 4080 … qwen2 1B Q4_K -
  Medium | pp512 | 39230.93".
- **E14** — vLLM PR #53920, "[Benchmark] Warn on warm prefix cache for random
  serve runs" (2026-08-27). https://github.com/vllm-project/vllm/pull/53920 —
  "0 / 131072 prefix-cache hits … 16349.78 tokens/s".
- **E15** — llama.cpp PR #21074, "ggml-cuda: Add generic NVFP4 MMQ kernel"
  (2026-03-27). https://github.com/ggml-org/llama.cpp/pull/21074
- **E16** — llama.cpp PR #20391 (2026-03-11).
  https://github.com/ggml-org/llama.cpp/pull/20391
- **E17** — Hardware Corner, "RTX 3090 Local LLM Benchmarks, Context Scaling…"
  (updated 2026-03). https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-3090/
- **E18** — Hardware Corner, RTX 4090 benchmarks (2026-03).
  https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4090/
- **E19** — Hardware Corner, RTX 4080 benchmarks (2026).
  https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4080/
- **E20** — Hardware Corner, RTX 4070 Ti Super benchmarks (2026).
  https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-4070-ti-super/
- **E21** — Hardware Corner, RTX 5090 benchmarks (2026-03; llama.cpp build
  8189). https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5090/
- **E22** — Hardware Corner, RTX PRO 6000 Blackwell benchmarks (2026-03).
  https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-pro-6000-blackwell/
- **E23** — arXiv 2601.09527, "Private LLM Inference on Consumer Blackwell
  GPUs…" (2026-01-14). https://arxiv.org/html/2601.09527v1
- **E24** — llama.cpp issue #19902, "Qwen3.5-35B-A3B MXFP4_MOE on RTX 3090 and
  RTX PRO 6000", build 8155 (832aa94).
  https://github.com/ggml-org/llama.cpp/issues/19902 — "build/bin/llama-bench
  -n 1024 -fa 1 -d 1024,4096,10240 -m …Qwen3.5-35B-A3B-MXFP4_MOE.gguf"
- **E25** — ewams.net, "How to Benchmark AI Performance with llama-bench"
  (2026-04-26).
  https://ewams.net/?date=2026%2F04%2F26&view=How_to_Benchmark_AI_Performance_with_llama-bench_%28llama.cpp%29
- **E26** — Hardware Corner, "We Tested Qwen3.8 27B…" (2026-08-17).
  https://www.hardware-corner.net/qwen3-8-27b-hardware-tests/ — "llama.cpp build
  153d324bc (10364)".
- **E27** — jonidimo, "Qwen3.8-27B on One RTX 3090: 14h Measured Benchmark"
  (2026-08-16/17). https://jonidimo.github.io/qwen38-3090-benchmark/benchmark.html
  — "705 tok/s over 128,290-token prompt"; "Without MTP 36.88 tok/s".
- **E28** — sudoingX/qwen38-mtp (2026-08/09).
  https://github.com/sudoingX/qwen38-mtp
- **E29** — syv-ai, HyperQwen (qwen38-27b-rtx3090) (2026).
  https://github.com/syv-ai/qwen38-27b-rtx3090 — "~1,035 tok/s at 64
  concurrent".
- **E30** — KGP Talkie, "Qwen 3.8 27B Speed Settings on llama.cpp"
  (2026-08-16, updated 2026-08-29).
  https://kgptalkie.com/tutorials/generative-ai/qwen-3-8-27b-llama-cpp-speed-settings
  — "f16 | 125.5 … q8_0 | 128.3 … q4_0 | 136.7".
- **E31** — Hugging Face, Qwen/Qwen3.8-27B discussion #132 (2026-08-18/23).
  https://huggingface.co/Qwen/Qwen3.8-27B/discussions/132 — slots "67 / 159 /
  326 / 521 / 579 t/s".
- **E32** — Hugging Face, Qwen/Qwen3.8-27B-FP8 discussion #9 (2026-08).
  https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/9
- **E33** — Startup Fortune, "A Single RTX 5000 PRO Is Running Qwen3 27B at 200k
  Context and 80 Tokens Per Second…" (2026-05-05; relays an unlinked Reddit
  post).
  https://startupfortune.com/a-single-rtx-5000-pro-is-running-qwen3-27b-at-200k-context-and-80-tokens-per-second-and-that-number-should-change-how-founders-think-about-local-inference-economics/
- **E34** — ubergarm/Qwen3-30B-A3B-GGUF model card (undated).
  https://huggingface.co/ubergarm/Qwen3-30B-A3B-GGUF — "over 1600 tok/sec PP
  and 105 tok/sec TG on my 3090TI".
- **E35** — KTransformers benchmark leaderboard (undated; decode only).
  https://ktransformers.net/en/benchmarks
- **E36** — CloudRift, "GPU Benchmarks for LLM Inference" (2025-10-09).
  https://www.cloudrift.ai/gpu-benchmarks
- **E37** — ai.rs, "How to Run Qwen3-Coder 30B-A3B on RTX 5090 with Ollama"
  (2026-05). https://ai.rs/ai-developer/qwen3-coder-30b-a3b-rtx-5090-ollama
- **E38** — CloudRift, "Optimizing Qwen3 Coder for RTX 5090 and PRO 6000"
  (2026-03-05). https://www.cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000
- **E39** — Millstone AI, Qwen3-Coder-30B-A3B-Instruct FP8 on 1× RTX Pro 6000
  (2026-02-02).
  https://www.millstoneai.com/inference-benchmark/qwen3-coder-30b-a3b-instruct-fp8-1x-rtx-pro-6000-blackwell
- **E40** — Amine Raji, "Qwen3.6 on 24GB VRAM: Benchmark, Config, and Every
  Mistake" (2026-04-18, updated 2026-09-19).
  https://aminrj.com/posts/llamacpp-qwen36-35b/
- **E41** — Giles Thomas, "Benchmarking Qwen 3.6 35B MoE (3B active) on an RTX
  3090" (2026-07-24).
  https://www.gilesthomas.com/2026/07/benchmarking-qwen-3-6-35b-moe-rtx-3090
- **E42** — zephel01, "[RTX 3090 Blazing Fast LLM Series Vol. 1]…" (note.com,
  2026-05-02). https://note.com/zephel01/n/n57ddf32000f2?hl=en
- **E43** — InsiderLLM, "Best Way to Run Qwen 3.6 35B MoE Locally" (2026-07).
  https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/
- **E44** — Magnus919, "Running a 35B MoE Model on a 16GB Consumer GPU"
  (2026-05-27).
  https://magnus919.com/2026/05/running-a-35b-moe-model-on-a-16gb-consumer-gpu/
- **E45** — ByteShape, "If It Fits, It Sits: Qwen 3.6 35B" (2026-05-19).
  https://byteshape.com/blogs/Qwen3.6-35B-A3B/
- **E46** — llama.cpp Discussion #19890, "RTX 5090 (CUDA) vs Radeon AI PRO R9700
  (Vulkan) — Qwen3.5-35B-A3B…" (2026-02-25 on).
  https://github.com/ggml-org/llama.cpp/discussions/19890
- **E47** — Millstone AI, Qwen3.5-35B-A3B FP8 on 1× RTX Pro 6000 (2026-02-26).
  https://www.millstoneai.com/inference-benchmark/qwen3-5-35b-a3b-fp8-1x-rtx-pro-6000-blackwell
- **E48** — Millstone AI, Qwen3.6-35B-A3B FP8 on 1× RTX Pro 6000 (2026-05-25).
  https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-fp8-1x-rtx-pro-6000-blackwell
- **E49** — ryan4yin, "Best llama.cpp config for Qwen3.8-Flash-Next (RTX 4090
  24GB)" (gist, 2026-08-29).
  https://gist.github.com/ryan4yin/48617bbddacc7067f10799770b7cc33f — "6K→≈30,
  90K→≈22, 110K→≈19.5 t/s"; "28.9 vs 30.1 t/s @32K".
- **E50** — @analogalok on X (≈ late 2026-08; figures from a search snippet,
  x.com was not fetchable).
  https://x.com/analogalok/status/2092697021790708148 — "21 tokens/sec decode.
  364 t/s prefill. no mtp. no dflash. no kv cache quantization!"
- **E51** — Hugging Face, unsloth/Qwen3.8-Flash-Next-GGUF discussion #3, "Share
  your model speed here" (2026-08-27 to ~09-13).
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3
- **E52** — holy_fox, "Running Qwen3.8-Flash-Next on an RTX 5090 with 128GB RAM
  using llama.cpp" (Zenn, 2026-08-27).
  https://zenn.dev/holy_fox/articles/04887ff8177b87?locale=en
- **E53** — Unsloth, "Qwen3.8-Flash-Next: How to Run Locally" (undated).
  https://unsloth.ai/docs/models/qwen3.8-next — "170 tokens/s on 1x RTX 6000 PRO
  GPU compared to the 100 token baseline".
- **E54** — Inovello, "Qwen3.8-Flash-Next on 2x3090 + DDR4…" (2026-09-03).
  https://inovello.dev/writeups/qwen3-flash-next-2x3090-expert-cache/
- **E55** — MindStudio, "DeepSeek V4 Flash on One RTX 3090: Real Tokens-Per-Second
  Numbers" (2026-08-25).
  https://www.mindstudio.ai/blog/freetoken-deepseek-v4-flash-single-3090
- **E56** — Ken Ashe, "DeepSeek-V4-Flash on a 3090 shifts the bottleneck to DDR5"
  (2026-08-02; relays an r/LocalLLaMA post).
  https://kenashe.ai/blog/2026-08-02-deepseek-v4-flash-on-a-3090-shifts-the-bottleneck-to-ddr5/
- **E57** — @analogalok on X (≈ early 2026-08; search snippet).
  https://x.com/analogalok/status/2084274615829102618 — "12 tokens/sec - Single
  RTX 4090 - 650+ tokens/sec prefill - 250k context - no kv cache quantization!"
- **E58** — 東リ屋, "DeepSeek V4 Flash Inference Optimization…" (note.com,
  2026-08-02; secondary, relays r/LocalLLaMA).
  https://note.com/samehadaonsen/n/n18477c290b02?hl=en — "average processing
  speed of 700pp/s and a generation speed of 18tg/s".
- **E59** — RockmSockmJesus, "DeepSeek-V4-Flash (284B MoE) at ~28 tok/s on 1x RTX
  5090 + dual Xeon Gold 6138" (gist, re-verified 2026-07-22).
  https://gist.github.com/RockmSockmJesus/30a195ccd9b62e981ec2676a99a57b7e
- **E60** — Qwen/Qwen3.8-Flash-Next model card (2026-08).
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next — "125B with 6B activated, plus
  51B n-gram embedding and 4B MTP".
- **E61** — Hugging Face, unsloth/DeepSeek-V4-Flash-GGUF discussion #6 (2026-07-12).
  https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/discussions/6
- **E62** — kvcache-ai/ktransformers, `doc/en/DeepSeek-V4-Flash.md` (undated).
  https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md
  — "Decode throughput: 20+ tok/s on a single RTX 5090."
- **E63** — llama.cpp PR #24162, "DeepSeek V4" (comments 2026-06-08 to
  2026-07-01). https://github.com/ggml-org/llama.cpp/pull/24162
- **E64** — SOTAAZ, "llama.cpp KV Cache Quantization, Measured on One A100 — q8_0
  Is Free at 4K and Costs Half Your Decode Speed at 64K" (2026-09-15).
  https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en
- **E65** — llama.cpp PR #7527, "CUDA: quantized KV support for FA vec" (merged
  2024-06-01). https://github.com/ggml-org/llama.cpp/pull/7527 — "Performance
  stays mostly the same with a quantized KV cache".
- **E66** — Kauan Lopes, "The One llama.cpp Setting That Made My RTX 3090 10×
  Faster" (2026-06-09).
  https://kauanlopes.com/blog/llama-cpp-setting-rtx-3090-10x-faster/
- **E67** — llama.cpp issue #24485 (2026-06-11).
  https://github.com/ggml-org/llama.cpp/issues/24485 — "Dramatically worse
  prefill speed (25-45x slower in our tests)".
- **E68** — MinskAndBoo/llama-kv-cache-compile README (undated).
  https://github.com/MinskAndBoo/llama-kv-cache-compile — "Prefill is flat
  within ~5% across all four pairs".
- **E69** — 2x4ever/llama.cpp PR #6, "cuda : load Q4_0 KV tiles directly in MMA
  flash attention" (2026-09-14). https://github.com/2x4ever/llama.cpp/pull/6
- **E70** — sergiuszm/ninfer-4090 README (undated).
  https://github.com/sergiuszm/ninfer-4090 — "a 5.7% decode tax (126.6 vs 134.2
  tok/s)".
- **E71** — exllamav2 issue #499, "Q-Cache - Token Generation Speed"
  (2024-06-09). https://github.com/turboderp/exllamav2/issues/499
- **E72** — vLLM Blog, "The State of FP8 KV-Cache and Attention Quantization in
  vLLM" (2026-04-22). https://vllm.ai/blog/2026-04-22-fp8-kvcache — "14.9% higher
  output throughput".
- **E73** — vLLM Blog, "A First Comprehensive Study of TurboQuant: Accuracy and
  Performance" (2026-05-11). https://vllm.ai/blog/2026-05-11-turboquant
- **E74** — SqueezeBits, "[vLLM vs TensorRT-LLM] #8. KV Cache Quantization"
  (2024-11-18).
  https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079
- **E75** — NVIDIA Technical Blog, "Optimizing Inference for Long Context and
  Large Batch Sizes with NVFP4 KV Cache" (2025-12-08).
  https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- **E76** — SGLang docs, "Quantized KV Cache" (undated).
  https://docs.sglang.io/docs/advanced_features/quantized_kv_cache
- **E77** — Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
  Cache", arXiv 2402.02750 (2024-02). https://arxiv.org/abs/2402.02750
- **E78** — Hooper et al., "KVQuant", arXiv 2401.18079 (2024-01-31; NeurIPS
  2024). https://arxiv.org/abs/2401.18079
- **E79** — Zhao et al., "Atom", arXiv 2310.19102 (2023-10-29).
  https://arxiv.org/abs/2310.19102
- **E80** — "TailorKV", arXiv 2505.19586 (2025-05; ACL Findings 2025).
  https://arxiv.org/abs/2505.19586
- **E81** — Chen et al., "MagicPIG", arXiv 2410.16179 (2024-10-21).
  https://arxiv.org/abs/2410.16179
- **E82** — "CommVQ", arXiv 2506.18879 (2025-06-23).
  https://arxiv.org/abs/2506.18879
- **E83** — "BitDecoding", arXiv 2503.18773 (2025-03-24; v3 2026-01-05).
  https://arxiv.org/abs/2503.18773
- **E84** — Gao et al., "Rethinking Key-Value Cache Compression Techniques for
  Large Language Model Serving", arXiv 2503.24000 (2025-03-31; MLSys 2025).
  https://arxiv.org/html/2503.24000v1
- **E85** — Zandieh et al., "TurboQuant", arXiv 2504.19874 (2025-04-28).
  https://arxiv.org/abs/2504.19874
- **E86** — Neroued, "NInfer" README, "Concurrent MTP3 decode" (RTX 5090, INT8
  group-64 KV, CUDA Graphs, MTP3, 8,192 tokens per request; accessed 2026-10-05).
  https://github.com/Neroued/ninfer — "At C=8, Qwen3.6-35B-A3B reaches **1,313.8
  aggregate decode tok/s**"
- **E87** — llama.cpp PR #21038, "llama : rotate activations for better
  quantization" (merged 2026-04-01).
  https://github.com/ggml-org/llama.cpp/pull/21038
- **E88** — SGLang issue #19603, "[Benchmark] Qwen3.5-122B-A10B FP8 weights /
  bf16 KV on 8x RTX PRO 6000 (SM120): 1,985 tok/s burst, MTP 2.75x, fp8 KV
  silent corruption finding" (2026-03-01).
  https://github.com/sgl-project/sglang/issues/19603
- **E89** — vLLM PR #46329, "[Attention][Quantization] NVFP4 KV cache on
  consumer/SoC Blackwell (sm120/sm121) for Gemma 3/4 via FlashInfer FA2"
  (2026-06-22). https://github.com/vllm-project/vllm/pull/46329
- **E90** — Niko1221, "Strata" README, engine 0.1.36 (MIT; accessed 2026-10-05).
  https://github.com/Niko1221/Strata — RTX 5070 12 GB, 64 GB RAM, 4K answers /
  32K prompts: Q2_0 94 t/s decode / 2,650 t/s prefill, IQ3_S 53 / 1,620
- **E91** — hagope, Strata community run, RTX 5090 (2026-09-30), Strata 0.1.29,
  commit d6708a4aae15b4860000d54c8af9e84d684bce09; IQ2_XS, MTP `--spec 4`,
  greedy, 256 output tokens, median of 3.
  https://github.com/Niko1221/Strata/tree/main/bench/results/2026-09-30-community-rtx-5090
- **E93** — Niko1221/Strata `bench/results/2026-09-29-rtx3090-epyc-milan`
  (issue #165), Strata v0.1.26 at ac8b251: IQ3_XXS on one RTX 3090 (2× 3090,
  EPYC 7453, 165 GiB, PCIe Gen4 x16 reported, 23–26 GB/s probed), int8 KV, MTP,
  median of 3. Decode 93.0 / 89.2 / 90.6 / 79.3 / 67.3 and prefill 869.7 /
  1,772.2 / 2,160.4 / 2,173.6 / 1,725.6 at 1K / 4K / 32K / 128K / 262K.
  https://github.com/Niko1221/Strata/tree/main/bench/results/2026-09-29-rtx3090-epyc-milan
- **E92** — Niko1221, Strata `docs/BATCHING.md`: RTX 5070 12 GB, Q2_0, 32K,
  `"parallel": 4` → 63.1 t/s total against 70.7 one request at a time.
  https://github.com/Niko1221/Strata/blob/main/docs/BATCHING.md

---

## 7. Next phase — outstanding measurements

What this document does not yet cover, split into what can be run with the
tests and checkpoints already in the tree and what is blocked on something the
fleet does not have. Each runnable item names the machine it needs, because
nothing transfers between cards (§1). Every sweep follows §5's rules: one
`cargo test` per model, the card to itself, results added as a new TSV and a
section beside the existing ones rather than overwriting them.

### 7.1 Runnable — RTX 4090 Mobile 16 GB

- **Depth.** Every model's `long_context_*` gate and Flash-Next's
  `profile_decode_vs_depth` / `profile_story_rewrite_vs_depth`, so this card
  gets §3.2–§3.6's depth axis. The interesting row is Flash-Next with its experts
  streamed: whether QSA's flat curve (§3.3) holds when the step is dominated by
  expert uploads rather than resident expert work.
- **The Qwen3-30B probe's efficiency shortfall** (§4, *The 30B engine probe
  holds ground that is not KV*). Bisect the compaction errors the 2026-09-30 run
  logged against the earlier runs that did not, then establish whether they
  account for the 43%.
- **DeepSeek-V4-Flash.** Never run on this card; whether its width ladder runs
  through the expert cache's streaming tiers at 16 GB is unmeasured.

### 7.2 Runnable — RTX 3090 24 GB

- **DeepSeek-V4-Flash.** No 3090 row. Size does not exclude it — Flash-Next
  runs here and on the 16 GB card through the expert cache (§3.8).
- **Depth**, as for the 4090 Mobile.
- **Llama-3.2-3B with flash-attn.** §3.8 builds without `--features
  flash-attn`, so it runs the model's fallback path; a flash-attn build
  measures the real one.

### 7.3 Runnable — RTX PRO 5000 Blackwell 72 GB

- **A re-sweep of the width gates on a current build.** The newest 72 GB width
  rows are from 2026-09-15. Qwen3.8-Flash-Next's hot path has changed since —
  hyper-connection kernels, a once-quantized MoE input, the head's mix — and on
  the 16 GB card those changes raised its gate prefill by 32–46% (BF16 ×4
  430.7 → 627.0 t/s, build `9be7b182c` against `a475e852c`), so §3.6 *Width* and
  §3.7 no longer describe the current code for that model.
- **Both engine probes**, which have no 72 GB result.
- **Qwen3.6-35B-A3B's C10 validation at ×32 and ×64** (§4, *Validation at
  extreme width under maximum compression*): intermittent, not reproduced on
  2026-09-13; it needs repeated runs to close or to confirm.
- **DeepSeek-V4-Flash at depth** (§4): out of memory at 32K on this card. It
  needs a smaller resident expert budget, so that the elastic partition has
  ground to concede to the KV span.

### 7.4 Runnable — any machine

- **Multi-context depth.** Every depth row is one context; width × depth
  interaction is not covered.
- **Warm/cold KV tiers.** Every gate row is hot-tier; no gate exercises the RAM or
  NVMe KV tiers. (The engine probes run the persistence thread but report no
  throughput.)
- **Non-speculative decode at depth.** The depth gate drives the speculative loop
  on models that support it; the plain path is not separately measured there.
- **The open causes in §4** — Qwen2-0.5B's 0% quantized C-mode rows, the
  compression-ratio movement between sweeps, the quantized read multiplier and
  Q0_V's decode cost. These are investigations rather than sweeps, but each
  would change figures in §3 when resolved.

### 7.5 Blocked

- **Full-attention against hybrid at matched depth.** The clean size-matched
  pair — Qwen3-8B and Qwen3.5-9B, same Q6_K, one full-attention and one hybrid —
  cannot be compared at depth, because the 8B's 40,960 window does not reach
  where the hybrids are measured. It needs a full-attention checkpoint with a
  native 262K window.
- **Selection against no selection at matched depth.** Flash-Next is the only
  model here that ships an indexer, and it is also the only one combining QSA,
  a 512-expert MoE and an n-gram embedding table. Its flat curve is measured
  beyond doubt (§3.3), but the fleet contains no second selecting model to
  separate QSA's contribution from everything else unique to that checkpoint.
  DeepSeek-V4-Flash is native-sparse and would be exactly that control once it
  reaches depth on some card (§7.3).
- **Any machine but these three.** See §1.
