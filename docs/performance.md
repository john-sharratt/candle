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
   experts VRAM → RAM → NVMe: 8/8 sessions correct at every rung, 64.7 t/s
   aggregate decode at ×8, 5.44× KV compression at C10 **[I1 · I2]**. No
   laptop-GPU run of this model has been published, and the published
   single-GPU runs that state their host use 64–128 GB of RAM
   **[E49 · E51 · E52]**. On a 24 GB RTX 3090 with 64 GB of RAM the same
   artifact decodes **24.3 t/s at one session and 147.0 t/s aggregate at ×16**,
   every session validated, where the published llama.cpp run on an RTX 3090
   with 128 GB decodes 15 t/s single-stream **[I18 · E51]**.

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

5. **In aggregate, one card out-decodes llama.cpp's best published rate — by up
   to 6×.** Serving concurrent conversations from one card, this engine's
   aggregate decode beats the best published llama.cpp single-stream figure for
   the same model on the same class of card: **1.8–3.4× on the RTX 3090**
   (Qwen3-8B 303.4 vs 115.3 t/s; Qwen3.8-27B 198.9 vs 65.3 with MTP), **2.2× on
   16 GB cards** (Qwen3-8B; Flash-Next), and **2.3–6.1× on Blackwell**
   (Qwen3.5-35B-A3B 1,187.7 vs 194.0) **[I8 · I16 · E1 · E17 · E19 · E21 · E27
   · E46 · E51]**. The one exception is the 35B MoEs on 16 GB, where a 3-bit
   quant that fits wholly in VRAM decodes a single stream at 183–249 t/s against
   our 130 aggregate with Q6_K experts streamed **[I11 · E45]**. The 35B MoEs reach
   **1,188–1,202 t/s aggregate at ×64** on one 72 GB card, eleven times their
   single-session rate, and Qwen3.5-0.8B serves **256 concurrent sessions**
   **[I8]**; the published single-card serving runs of the same models stop at
   5–10 concurrent requests **[E47 · E48]**.

6. **Workstation-class throughput from a laptop.** On the 16 GB laptop,
   Qwen3.5-0.8B prefills 32 concurrent sessions at **21,847 t/s** and decodes
   them at **993 t/s aggregate** under C8 compression, and Qwen3-30B-A3B prefills
   at **4,000 t/s with its experts streaming** — matching the 24 GB RTX 3090,
   which holds far more of the model **[I9 · I10]**. One wave engine carries
   prefill rows and decode rows in the same forward, which is what the engine
   probe runs under load **[I15]**.

7. **A 35B MoE serving sixteen users on a laptop.** Qwen3.5/3.6-35B-A3B at Q6_K —
   ~28 GB of weights — serves ×16 at **~130 t/s aggregate with 6.2× KV
   compression** on the 16 GB laptop, every session validated **[I11]**. The
   published 16 GB runs of these models are single-stream, at smaller quants
   **[E43 · E44]**.

8. **A 284B model at sixteen-way concurrency on one GPU.** DeepSeek-V4-Flash on a
   single 72 GB card prefills at **1,121 t/s** — above every published
   single-GPU figure for the model, whose best is 748 t/s — and decodes
   **73.5 t/s aggregate**, 2.6× the best published single-GPU decode
   (28 t/s, single-stream) **[I12 · E59 · E63]**.

9. **One engine, every card, identical compression.** The same gates pass on a
   PCIe 3.0 RTX 3090 with no native FP8, an Ada laptop and a Blackwell
   workstation card — 187 ladder rows on the laptop alone, no session failing —
   with the engine sizing its own memory partition to each card. C10 compresses
   the 35B to 6.23× on both the 3090 and the laptop, and every dense rung
   matches to within 0.1× **[I13 · I14]**.

10. **The whole engine, proven under load — not just its kernels.** The engine
    probe drives admission, per-turn context projection, the persistence thread,
    KV compaction and the three memory tiers together: on Flash-Next — which
    carries recurrent state outside the paged KV — **8/8 stories correct
    at 100% VRAM efficiency**, with the expert weights retaking 73–88% of the
    ground released **[I15]**. This week's hot-path work added **32–46% to
    Flash-Next's prefill** with zero loss of validation **[I17]**.

### Internal references — our results

| Ref | Result | Where |
|---|---|---|
| **I1** | Flash-Next on the RTX 4090 Laptop GPU: full ladder, 8/8 validated, BF16 ×8 495.4 / 64.7 t/s, C10 ×8 5.43–5.44× | §3.9 *Qwen3.8-Flash-Next*; `results/performance_rtx_4090_mobile_16gb_rows.tsv` (last nine rows) |
| **I2** | The laptop's host: Core Ultra 9 185H, 31.5 GiB RAM, 16 GB VRAM; Flash-Next's 180B / 6B-active size | §1 machine table; §3.1 and its note ¹ |
| **I3** | Flash-Next depth retention 32K → 128K: prefill 99%, decode 111%; the rest of the fleet 25–30% prefill | §3.2, §3.3 |
| **I4** | `Rewrite` at 8K–128K: acceptance 4.85 on every row; the story validated behind 128,897 tokens | §3.6 *Rewrite* |
| **I5** | C10 at 128K: 4.63×–7.63× compression; Flash-Next 6.97× at −19% decode; 100% of blocks quantized | §3.2, §3.5 |
| **I6** | C10 across the fleet's ladders: 4.11× (Qwen3.5-0.8B) to 6.23× (Qwen3.5-35B) at width; 7.63× at 128K depth | §3.7–§3.9 ladders; §3.2 |
| **I7** | Compression at 8K: 0–8% of decode for 3.3×–6.3× | §3.4 |
| **I8** | Width: 35B MoEs 1,187.7 / 1,201.6 t/s aggregate at ×64 against ~108–109 at ×1 (best of three sweeps); within one run, Qwen3.6-35B-A3B 104.9 → 1,201.6 t/s (×1 BF16 → ×64 C10), 11.5×; Qwen3.5-0.8B at ×256 | §3.7; `results/performance_rtx_pro_5000_72gb_rows_2026-09-15_run1.tsv` |
| **I9** | Qwen3.5-0.8B C8 ×32 on the laptop: prefill 21,847.0, decode 993.0 t/s | §3.9 *Qwen3.5-0.8B* |
| **I10** | Qwen3-30B-A3B Q8_0 ×20: 3,999.7 t/s prefill on the laptop, 3,873.0 on the RTX 3090 | §3.9, §3.8 *Qwen3-30B-A3B* |
| **I11** | Qwen3.5/3.6-35B-A3B C10 ×16 on the laptop: 131.2 / 129.7 t/s aggregate, 6.20× / 5.94× | §3.9 *Qwen3.5-35B-A3B*, *Qwen3.6-35B-A3B* |
| **I12** | DeepSeek-V4-Flash on the 72 GB card: ×16 1,120.6 t/s prefill, 73.5 t/s aggregate decode | §3.7 |
| **I13** | Thirteen gates, 187 rows on the laptop with no failing session; the same gates on the 3090 and 72 GB card | §3.7–§3.9; §5 provenance table |
| **I14** | C10 on the 35Bs: 6.23× / 5.96× (laptop) against 6.23× / 6.05× (3090); dense rungs within 0.1× | §3.9 *Against the RTX 3090* |
| **I15** | Engine probe, Flash-Next: story 8/8, worst sustained efficiency 100%, uptake 73% (2026-09-30) and 88% (2026-09-29) | §3.9 *Engine probes* |
| **I16** | Aggregate decode against llama.cpp's best published single-stream rate, same model and card class. RTX 3090: Qwen3-8B C8 ×10 303.4 vs 115.3; Qwen3-30B-A3B Q8_0 ×20 274.5 vs 153.6; Qwen3.5-35B C10 ×16 375.9 vs 111.2; Qwen3.6-35B C10 ×16 388.6 vs 157.66; Qwen3.8-27B C9 ×5 198.9 vs 65.28 (MTP); Llama-2-7B BF16 ×48 547.6 vs 161.89. 16 GB: Qwen3-8B C8 ×10 230.1 vs 102.7 (RTX 4080); Flash-Next BF16 ×8 64.7 vs 27.5–29 (RTX 5080); exception — Qwen3.6-35B C10 ×16 129.7 vs 183.29 / 249.33 MTP (RTX 4080, IQ3_S resident). Blackwell (our RTX PRO 5000 vs a published RTX 5090): Qwen3-8B 460.1 vs 200.4; Qwen3-30B-A3B 595.7 vs 226.1; Qwen3.5-35B 1,187.7 vs 194.0; Qwen3.6-35B 1,201.6 vs 333.55 (MTP); Llama-2-7B 917.3 vs 300.40 | §3.7–§3.9 ladders; §6.2–§6.5 |
| **I17** | Flash-Next gate prefill 430.7 → 627.0 t/s (BF16 ×4) and 354.5 → 472.0 (BF16 ×8), builds `9be7b182c` → `a475e852c`, every row validated | commits `91ef33339`, `eef0f44ab`, `0ce74ecad`; §7.3 |
| **I18** | Flash-Next on the RTX 3090 (i7-10700K, 64 GB RAM), Q2_KO experts: full ladder incl. ×16, every row validated; BF16 ×1 warm 524.0 / 24.3 t/s, ×4 1,033.4 prefill, ×16 147.0 aggregate decode, C10 ×2 5.46×; engine probe story 8/8, efficiency 99%, uptake 80% | §3.8 *Qwen3.8-Flash-Next*; `results/performance_rtx_3090_24gb_rows_2026-09-30.tsv` |

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
> - **RTX 3090 24 GB** — a width/throughput gate sweep on 2026-09-14 (§3.8),
>   the same `test_parallel_batched_forwarding*` gates as §3.7, run one model at
>   a time. Ten of the fleet's models plus the two AntiLoop+StyleTune hybrids;
>   the 180B Flash-Next and the 284B DeepSeek were not run in that sweep.
>   **Qwen3.8-Flash-Next** was added on 2026-09-30 (build `e596fad8d`): its gate
>   and its `kv_fragmentation` engine probe, from the same Q2_KO-expert artifact
>   the 4090 Mobile runs (§3.8 *Qwen3.8-Flash-Next*). **No depth curves** — the
>   `long_context_*` and `profile_*` gates were not run on this card, so the 3090
>   appears in the width tables only.
> - **RTX 4090 Mobile 16 GB** — a width/throughput gate sweep on 2026-09-30
>   (§3.9), build `bf291341c`: the same `test_parallel_batched_forwarding*` gates,
>   one model at a time, plus the two `kv_fragmentation` engine probes (§2.2). Ten
>   of the fleet's models, the two AntiLoop+StyleTune hybrids, and
>   **Qwen3.8-Flash-Next**, which runs here from its Q2_KO-expert artifact through
>   the expert cache's streaming tiers; the 284B DeepSeek was not run. **No depth
>   curves**, as on the 3090.
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
| **Measured here** | depth + width (§3.2–§3.7) | width gate sweep (§3.8) | width gate sweep + engine probes (§3.9) |

Properties that shape several results, worth stating before the tables:

- **WDDM, not TCC**, on all three cards. Kernel launches carry the Windows
  display-driver model's submission overhead, which is the floor under
  single-session decode on every small model here. A Linux/TCC host would move
  the decode column and leave the prefill column roughly alone.
- **72 GB on one card.** Every model in the reference report except the 284B fits
  its weights resident, so the depth curves below are *not* contaminated by
  weight paging. That is the point of running them there.
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

The flagship's ladder, aggregate across the batch — best of the three sweeps,
† = 2026-09-13, ◆ = 2026-09-15:

| Mode | Ctx | Prefill t/s | Decode t/s | Compress |
|---|---:|---:|---:|---:|
| BF16 | 1 (cold) | 589.7 † | 68.1 | — |
| BF16 | 1 (warm) | 1,705.7 ◆ | 86.9 | — |
| BF16 | 4 | 1,878.7 † | 241.1 ◆ | — |
| BF16 | 8 | 1,880.9 † | 393.1 † | — |
| BF16 | 16 | 1,969.8 ◆ | 421.5 † | — |
| C0 | 2 | 2,009.8 ◆ | 146.4 | 2.29× |
| C5 | 2 | 2,018.1 ◆ | 145.4 | 4.21× |
| C8 | 2 | 2,021.4 † | 139.5 | 5.43× |
| C10 | 2 | 2,017.8 ◆ | 135.4 † | 6.93× |
| C10 | 8 | 2,060.9 ◆ | 391.7 ◆ | 6.89× |

All rows validate at 100% in every sweep. The 2026-09-15 sweep sets six of the
ten prefill cells — C10 ×8 reaches 2,060.9 — and matches the earlier decode
cells within noise. The second sweep is the faster of the first two on every
prefill cell and on the widest decode cells — BF16 ×8 decode
310.6 → 393.1 and ×16 333.3 → 421.5, C10 ×8 295.9 → 391.4 (+26–32%) — while
the single-context and ×2 decode cells stay with the first. Decode returns
**4.5× single-session throughput at 8 contexts** (86.9 → 393.1) and gains only
another 7% from 8 to 16; prefill is already near the device's limit at one warm
context (1,686 → 1,960 at ×16), so width buys decode, not prefill. The ladder's
C0→C10 span costs **13% of decode in the first sweep and 4% in the second**
(each measured within its own run) for **~3× more compression**. The C10
ratios shown are the first sweep's; the second measured 6.75× and 6.73× — see
§4 on compression between the sweeps.

### 3.7 Width across the fleet

BF16 at one context against each model's widest measured point. Prompts are
~700 tokens, so this axis is unaffected by context windows. Each cell is the
highest of the three sweeps at the same mode and width, † = 2026-09-13,
◆ = 2026-09-15 (two runs of build `23623c6b`). The 2026-09-15 one-context
prefill cells come from the gates' synchronised prompt timer, which the older
builds measure identically (§4, *The decode-slot refresh prefill regression*).

| Model | ctx=1 prefill / decode | widest measured | prefill / decode |
|---|---|---|---|
| Qwen2-0.5B | 31,605.8 ◆ / 256.7 ◆ | ×60 | 79,653.3 ◆ / 4,924.7 ◆ |
| Qwen3.5-0.8B | 24,626.5 ◆ / 168.8 | ×256 (C8) | 36,290.6 ◆ / 3,353.4 |
| Llama-3.2-3B | 13,166.0 ◆ / 130.5 ◆ (C0) | ×10 (C8) | 13,977.4 ◆ / 745.1 ◆ |
| Qwen3-30B-A3B | 8,307.4 ◆ / 80.7 ◆ | ×20 (Q8_0) | 9,950.6 ◆ / 595.7 ◆ |
| Qwen3.5-35B-A3B | 7,231.9 † / 109.4 | ×64 (C10) | 7,153.5 ◆ / 1,187.7 ◆ |
| Qwen3.6-35B-A3B | 7,310.2 ◆ / 107.8 | ×64 (C10) | 7,219.2 ◆ / 1,201.6 ◆ |
| Qwen3-8B | 6,008.5 ◆ / 67.3 ◆ | ×10 (C8) | 6,065.2 ◆ / 460.1 |
| Llama-2-7B | 6,063.7 ◆ / 97.2 ◆ | ×48 | 3,679.8 † / 917.3 ◆ |
| Qwen3.5-9B | 5,534.1 ◆ / 121.0 | ×20 (C8) | 5,837.5 † / 877.5 † |
| Qwen3.8-27B | 1,729.8 ◆ / 61.0 ◆ | ×40 (C10) | 1,718.7 ◆ / 458.1 |
| Qwen3.8-Flash-Next | 1,705.7 ◆ / 86.9 (warm) | ×16 | 1,969.8 ◆ / 421.5 † |
| DeepSeek-V4-Flash | 333.3 / 15.1 ◆ (warm) | ×16 | 1,120.6 ◆ / 73.5 |

Two shapes appear here. **Prefill saturates early** on every model — most are
within 20% of their ×1 rate by ×4, and the 35Bs are flat from ×1 to ×64 — while
**decode scales nearly linearly with width** until it too flattens. The 35B MoEs
reach 1,188–1,202 t/s aggregate decode at 64 concurrent sessions against
~108–109 at one, an 11× return on concurrency.

DeepSeek-V4-Flash is the exception whose prefill is still climbing at ×16
(333 → 1,121 t/s), having not yet reached the saturation the others hit by ×4.

The ladders are not run at a common set of widths, so this table gives each
model's own widest point rather than a shared column — and two ladders run
wider from the second sweep on (Qwen3.5-0.8B to ×256, Qwen3-30B-A3B to ×20), so
those widest points are the best of the last two sweeps rather than all three.

### 3.8 RTX 3090 24 GB — the width gate sweep

A single sequential sweep on **2026-09-14**, on the RTX 3090 (§1), of the same
`test_parallel_batched_forwarding*` gates that produce §3.6 *Width* and §3.7 —
one `cargo test` invocation per model so exactly one was ever resident,
`--release --features cuda`. This is a **width / throughput** sweep: each gate
runs its own ladder of KV modes and context counts at a fixed ~700-token prompt,
so it measures aggregate throughput and the compression ladder, **not** depth —
the `long_context_*` and `profile_*` depth gates were not run here. Numbers are
single measurements (no best-of-two), so the 1–4 % noise floor (§5) applies to
each cell alone.

**Measured on a build that carries the decode-slot refresh regression.** The
sweep ran after `d45e69ce`, whose refresh cost the 72 GB card 58 % of Qwen2-0.5B's
widest prefill (§4, *The decode-slot refresh prefill regression*), so this
table's prefill column likely carries a share of that cost. Decode is
unaffected. The 3090 has not been re-swept on the fixed build.

**Ten of the fleet's models, plus the two AntiLoop+StyleTune hybrids** — the
production 3.6-35B npcd actually serves. The 180B Flash-Next and the 284B
DeepSeek were not run in this sweep. Size alone does not exclude them — the
expert cache streams a MoE's experts from host, and Flash-Next runs on the
16 GB card (§3.9) — and **Flash-Next was measured separately on 2026-09-30**,
on a current build (*Qwen3.8-Flash-Next* below, marked ‡ in the summary).
DeepSeek still has no 3090 row. Two card-specific notes carry
into these rows: **Llama-3.2-3B ran without flash-attn** (its build needs
`cl.exe` on PATH, absent in the sweep shell; the model's
`#[cfg(not(feature = "flash-attn"))]` fallback path was used instead), and
**Qwen2-0.5B's gate ladder has no compressed rung** — it runs F32, BF16 and F16
only, so its compression column is empty by construction.

**Summary** — best prefill, best decode, and the best validated compression over
each model's ladder:

| Model | best prefill t/s | best decode t/s | best compression (mode) |
|---|---:|---:|---|
| Qwen2-0.5B | 23,762.6 | 4,951.2 | — (no ladder) |
| Qwen3.5-0.8B | 18,468.5 | 1,149.9 | 4.11× (C10) |
| Llama-3.2-3B † | 5,326.9 | 542.7 | 4.27× (C10) |
| Llama-2-7B | 2,952.9 | 547.6 | 3.56× (Q4_0) |
| Qwen3-8B | 2,707.2 | 303.4 | 5.84× (C10) |
| Qwen3.5-9B | 2,907.7 | 491.3 | 5.13× (C10) |
| Qwen3.8-27B | 940.5 | 198.9 | 4.78× (C10) |
| Qwen3-30B-A3B | 3,886.6 | 274.5 | 5.31× (C9) |
| Qwen3.5-35B-A3B | 3,539.2 | 375.9 | 6.23× (C10) |
| Qwen3.6-35B-A3B | 3,662.1 | 388.6 | 6.05× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (auto/Precision) | 3,463.1 | 351.6 | 6.09× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (Performance) | 3,742.3 | 399.0 | 6.10× (C10) |
| Qwen3.8-Flash-Next (Q2_KO experts) ‡ | 1,033.4 | 147.0 | 5.46× (C10) |

† ran on the no-flash-attn fallback path (above).
‡ measured 2026-09-30 on build `e596fad8d`, which carries the refresh fix and the
synchronised prompt timer — not comparable with the other rows' prefill (above).

**Against the 72 GB card, on the models both ran.** The comparison holds only on
the width gate, and only loosely: the 3090's ladders stop at a narrower widest
context (×16 on the 35Bs, where the 72 GB reached ×64), because 24 GB caps how
many concurrent sessions fit. So the raw gaps are dominated by concurrency
headroom, not per-session speed. On the 35B MoEs the 72 GB card prefills about
**2×** the 3090 at one context (~7,200 vs ~3,600 t/s) and reaches about **3×**
the aggregate decode at its own widest (1,150–1,183 vs 376–389 t/s at ×16) —
most of that decode gap being the extra 48 sessions the bigger card holds.
**Compression tracks the model, not the card**: the 3090's C10 lands at
6.23×/6.05× on the two 35Bs against the 72 GB card's 2026-09-13 6.20×/6.04×
(§4) — the same ladder within noise, as expected, since a ratio is bytes stored
and the adaptive policy is identical on both. The sm_86 and PCIe-3.0 traps (§1)
sit under the 3090's absolute rates but leave the ratios untouched.

**Full ladders.** Each model's complete gate ladder — every KV mode and context
the gate ran, exactly as the run logs printed them. `Valid` is the per-config
reproduction check (`✓`, or `-` for a mode not validated for reproduction);
`int8` is the loader's int8 posture (`prec`/`perf`/`off`).

#### Qwen2-0.5B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | off | no | 1 | - | 15018.2 | 241.5 | - | - | 308 |
| BF16 | off | yes | 1 | - | 15752.4 | 247.2 | - | - | 308 |
| F16 | off | yes | 1 | - | 15642.9 | 240.2 | - | - | 308 |
| F16 | off | yes | 4 | - | 23762.6 | 902.6 | - | - | 1272 |
| F16 | off | yes | 60 | - | 22018.4 | 4455.0 | - | - | 18612 |
| BF16 | off | yes | 60 | - | 21985.8 | 4951.2 | - | - | 18612 |

#### Qwen3.5-0.8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 14961.7 | 103.4 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 15114.3 | 134.6 | - | - | 659 |
| BF16 | prec | yes | 16 | ✓ | 17779.6 | 897.6 | - | - | 10618 |
| Q8_0 | prec | yes | 4 | ✓ | 15875.2 | 232.5 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 2 | ✓ | 17615.3 | 324.0 | 100.0% | 1.85x | 1358 |
| C1 | prec | yes | 2 | ✓ | 17639.0 | 324.5 | 100.0% | 2.01x | 1358 |
| C2 | prec | yes | 2 | ✓ | 17659.5 | 318.5 | 100.0% | 2.26x | 1358 |
| C3 | prec | yes | 2 | ✓ | 17445.2 | 315.0 | 100.0% | 2.40x | 1358 |
| C4 | prec | yes | 2 | ✓ | 17613.2 | 323.5 | 100.0% | 2.58x | 1358 |
| C5 | prec | yes | 2 | ✓ | 17640.2 | 317.9 | 100.0% | 2.76x | 1358 |
| C6 | prec | yes | 2 | ✓ | 17603.8 | 325.2 | 100.0% | 2.86x | 1358 |
| C7 | prec | yes | 2 | ✓ | 17500.3 | 326.9 | 100.0% | 3.61x | 1358 |
| C8 | prec | yes | 32 | ✓ | 17632.3 | 772.4 | 100.0% | 3.83x | 21202 |
| C9 | prec | yes | 5 | ✓ | 18468.5 | 728.0 | 100.0% | 4.08x | 3337 |
| C10 | prec | yes | 10 | ✓ | 17796.4 | 1149.9 | 100.0% | 4.11x | 6640 |

#### Llama-3.2-3B (no flash-attn fallback)

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | prec | no | 1 | ✓ | 5261.6 | 130.3 | - | - | 654 |
| F16 | prec | yes | 1 | ✓ | 5279.1 | 130.7 | - | - | 654 |
| F16 | prec | yes | 4 | ✓ | 4980.5 | 375.3 | - | - | 2658 |
| R16 | prec | yes | 1 | ✓ | 5320.5 | 130.8 | 0.0% | - | 654 |
| Q8_0 | prec | yes | 1 | ✓ | 5297.1 | 135.3 | 100.0% | 1.88x | 654 |
| Q8_Q4 | prec | yes | 1 | ✓ | 5326.9 | 128.5 | 100.0% | 2.29x | 654 |
| BF16 | prec | yes | 4 | ✓ | 4972.2 | 373.4 | - | - | 2658 |
| Q8_1 | prec | yes | 4 | ✓ | 4968.3 | 313.1 | 100.0% | 1.78x | 2658 |
| Q8_KS | prec | yes | 4 | ✓ | 4968.6 | 348.0 | 100.0% | 1.78x | 2658 |
| Q8_Q4 | prec | yes | 4 | ✓ | 4946.9 | 312.0 | 100.0% | 2.29x | 2658 |
| Q4_0 | prec | yes | 4 | - | 4928.2 | 354.6 | 100.0% | 3.56x | 2658 |
| Q4_1 | prec | yes | 4 | - | 4932.4 | 317.1 | 100.0% | 3.20x | 2658 |
| Q4_KS | prec | yes | 4 | - | 4968.1 | 339.7 | 100.0% | 3.20x | 2658 |
| C0 | prec | yes | 1 | ✓ | 5288.7 | 133.5 | 100.0% | 1.87x | 654 |
| C1 | prec | yes | 1 | ✓ | 5240.4 | 132.0 | 100.0% | 2.19x | 654 |
| C2 | prec | yes | 1 | ✓ | 5252.0 | 133.3 | 100.0% | 2.37x | 654 |
| C3 | prec | yes | 1 | ✓ | 5211.1 | 132.3 | 100.0% | 2.75x | 654 |
| C4 | prec | yes | 1 | ✓ | 5253.5 | 132.4 | 100.0% | 3.07x | 654 |
| C5 | prec | yes | 1 | ✓ | 5281.4 | 132.2 | 100.0% | 3.28x | 654 |
| C6 | prec | yes | 1 | ✓ | 5257.8 | 131.4 | 100.0% | 3.44x | 654 |
| C7 | prec | yes | 1 | ✓ | 5270.8 | 131.8 | 100.0% | 3.80x | 654 |
| C8 | prec | yes | 10 | ✓ | 4179.3 | 542.7 | 100.0% | 3.90x | 6590 |
| C9 | prec | yes | 10 | ✓ | 4171.5 | 482.6 | 100.0% | 4.22x | 6590 |
| C10 | prec | yes | 5 | ✓ | 4790.5 | 402.3 | 100.0% | 4.27x | 3312 |

#### Llama-2-7B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | perf | no | 1 | - | 2952.9 | 87.7 | - | - | 283 |
| F16 | perf | yes | 1 | - | 2929.5 | 88.0 | - | - | 283 |
| F16 | perf | yes | 4 | - | 2909.6 | 244.4 | - | - | 1176 |
| F16 | perf | yes | 8 | - | 2555.7 | 359.4 | - | - | 2316 |
| BF16 | perf | yes | 1 | - | 2894.8 | 90.9 | - | - | 283 |
| BF16 | perf | yes | 8 | - | 2556.8 | 360.8 | - | - | 2316 |
| BF16 | perf | yes | 16 | - | 1998.9 | 466.3 | - | - | 4624 |
| BF16 | perf | yes | 48 | - | 1433.0 | 547.6 | - | - | 13796 |
| Q8_0 | perf | yes | 32 | - | 1813.8 | 464.7 | 100.0% | 1.88x | 9220 |
| Q4_0 | perf | yes | 32 | - | 1813.4 | 445.8 | 100.0% | 3.56x | 9220 |

#### Qwen3-8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | no | 1 | ✓ | 2697.4 | 69.2 | - | - | 636 |
| F16 | perf | yes | 1 | ✓ | 2707.2 | 69.8 | - | - | 636 |
| F16 | perf | yes | 2 | ✓ | 2675.7 | 114.4 | - | - | 1312 |
| BF16 | perf | yes | 4 | ✓ | 2351.2 | 213.3 | - | - | 2586 |
| Q8_0 | perf | yes | 4 | ✓ | 2355.2 | 208.8 | 100.0% | 1.88x | 2586 |
| C0 | perf | yes | 1 | ✓ | 2662.8 | 71.5 | 100.0% | 1.90x | 636 |
| C1 | perf | yes | 1 | ✓ | 2661.9 | 71.5 | 100.0% | 2.54x | 636 |
| C2 | perf | yes | 1 | ✓ | 2663.9 | 71.6 | 100.0% | 2.67x | 636 |
| C3 | perf | yes | 1 | ✓ | 2652.9 | 71.8 | 100.0% | 2.92x | 636 |
| C4 | perf | yes | 1 | ✓ | 2653.6 | 71.3 | 100.0% | 3.31x | 636 |
| C5 | perf | yes | 1 | ✓ | 2653.7 | 71.3 | 100.0% | 3.67x | 636 |
| C6 | perf | yes | 1 | ✓ | 2652.9 | 70.6 | 100.0% | 4.31x | 636 |
| C7 | perf | yes | 1 | ✓ | 2656.2 | 70.5 | 100.0% | 4.49x | 636 |
| C8 | perf | yes | 10 | ✓ | 2262.6 | 303.4 | 100.0% | 4.85x | 6410 |
| C9 | perf | yes | 5 | ✓ | 2260.9 | 222.8 | 100.0% | 5.47x | 3222 |
| C10 | perf | yes | 5 | ✓ | 2266.2 | 225.1 | 100.0% | 5.84x | 3222 |

#### Qwen3.5-9B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 2879.6 | 93.8 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 2907.7 | 93.7 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 2573.9 | 311.8 | - | - | 2678 |
| Q8_0 | prec | yes | 4 | ✓ | 2566.4 | 337.3 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 1 | ✓ | 2899.5 | 108.2 | 100.0% | 2.12x | 659 |
| C1 | prec | yes | 1 | ✓ | 2895.6 | 107.5 | 100.0% | 2.43x | 659 |
| C2 | prec | yes | 1 | ✓ | 2889.7 | 105.6 | 100.0% | 2.82x | 659 |
| C3 | prec | yes | 1 | ✓ | 2893.4 | 106.8 | 100.0% | 3.12x | 659 |
| C4 | prec | yes | 1 | ✓ | 2885.8 | 106.3 | 100.0% | 3.45x | 659 |
| C5 | prec | yes | 1 | ✓ | 2895.9 | 106.8 | 100.0% | 3.62x | 659 |
| C6 | prec | yes | 1 | ✓ | 2897.4 | 109.4 | 100.0% | 3.95x | 659 |
| C7 | prec | yes | 1 | ✓ | 2872.1 | 107.6 | 100.0% | 4.03x | 659 |
| C8 | prec | yes | 20 | ✓ | 2555.7 | 295.3 | 100.0% | 4.49x | 13278 |
| C9 | prec | yes | 5 | ✓ | 2458.3 | 381.2 | 100.0% | 5.00x | 3337 |
| C10 | prec | yes | 10 | ✓ | 2598.1 | 491.3 | 100.0% | 5.13x | 6640 |

#### Qwen3.8-27B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 940.5 | 50.6 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 860.8 | 179.8 | - | - | 2678 |
| Q8_0 | perf | yes | 4 | ✓ | 855.8 | 175.7 | 100.0% | 1.88x | 2678 |
| C0 | perf | yes | 1 | ✓ | 934.6 | 66.2 | 100.0% | 2.11x | 659 |
| C1 | perf | yes | 1 | ✓ | 933.1 | 65.2 | 100.0% | 2.33x | 659 |
| C2 | perf | yes | 1 | ✓ | 932.2 | 65.5 | 100.0% | 2.69x | 659 |
| C3 | perf | yes | 1 | ✓ | 933.6 | 65.7 | 100.0% | 3.05x | 659 |
| C4 | perf | yes | 1 | ✓ | 931.2 | 65.6 | 100.0% | 3.38x | 659 |
| C5 | perf | yes | 1 | ✓ | 931.0 | 65.3 | 100.0% | 3.56x | 659 |
| C6 | perf | yes | 1 | ✓ | 930.7 | 65.8 | 100.0% | 3.78x | 659 |
| C7 | perf | yes | 1 | ✓ | 929.6 | 65.6 | 100.0% | 3.84x | 659 |
| C8 | perf | yes | 20 | ✓ | 828.2 | 101.6 | 100.0% | 4.27x | 13278 |
| C9 | perf | yes | 5 | ✓ | 842.2 | 198.9 | 100.0% | 4.70x | 3337 |
| C10 | perf | yes | 10 | ✓ | 823.4 | 196.6 | 100.0% | 4.78x | 6640 |

#### Qwen3-30B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | perf | yes | 1 | ✓ | 1914.8 | 33.8 | - | - | 626 |
| BF16 | perf | yes | 1 | ✓ | 2987.2 | 40.3 | - | - | 626 |
| BF16 | perf | yes | 10 | ✓ | 3886.6 | 256.6 | - | - | 6310 |
| Q8_0 | perf | yes | 20 | ✓ | 3873.0 | 274.5 | 100.0% | 1.88x | 12620 |
| Q4_0 | perf | yes | 4 | - | 3883.1 | 107.7 | 100.0% | 3.56x | 2546 |
| C0 | perf | yes | 2 | ✓ | 3590.5 | 74.2 | 100.0% | 1.98x | 1292 |
| C1 | perf | yes | 2 | ✓ | 3598.7 | 74.5 | 100.0% | 2.54x | 1292 |
| C2 | perf | yes | 2 | ✓ | 3596.8 | 72.7 | 100.0% | 2.74x | 1292 |
| C3 | perf | yes | 2 | ✓ | 3581.9 | 74.3 | 100.0% | 2.99x | 1292 |
| C4 | perf | yes | 2 | ✓ | 3580.8 | 73.0 | 100.0% | 3.41x | 1292 |
| C5 | perf | yes | 2 | ✓ | 3574.1 | 73.5 | 100.0% | 3.67x | 1292 |
| C6 | perf | yes | 2 | ✓ | 3587.1 | 72.2 | 100.0% | 4.17x | 1292 |
| C7 | perf | yes | 2 | ✓ | 3566.3 | 66.4 | 100.0% | 4.24x | 1292 |
| C9 | perf | yes | 2 | ✓ | 3584.1 | 70.8 | 100.0% | 5.31x | 1292 |
| BF16 | perf | yes | 1 | ✓ | 3281.3 | 42.4 | - | - | 626 |
| Q4_0 | perf | yes | 20 | - | 3874.9 | 268.5 | 100.0% | 3.56x | 12620 |

#### Qwen3.5-35B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1267.1 | 41.8 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 3066.1 | 256.7 | - | - | 2678 |
| Q8_0 | perf | yes | 2 | ✓ | 3508.7 | 127.2 | 100.0% | 1.88x | 1358 |
| C0 | perf | yes | 1 | ✓ | 3525.8 | 80.5 | 100.0% | 2.20x | 659 |
| C1 | perf | yes | 1 | ✓ | 3528.6 | 81.8 | 100.0% | 2.85x | 659 |
| C2 | perf | yes | 1 | ✓ | 3532.7 | 81.3 | 100.0% | 3.26x | 659 |
| C3 | perf | yes | 1 | ✓ | 3483.6 | 81.2 | 100.0% | 3.41x | 659 |
| C4 | perf | yes | 1 | ✓ | 3539.2 | 79.7 | 100.0% | 3.72x | 659 |
| C5 | perf | yes | 1 | ✓ | 3538.0 | 80.6 | 100.0% | 3.83x | 659 |
| C6 | perf | yes | 1 | ✓ | 3516.2 | 80.2 | 100.0% | 4.51x | 659 |
| C7 | perf | yes | 1 | ✓ | 3521.2 | 79.9 | 100.0% | 4.61x | 659 |
| C8 | perf | yes | 5 | ✓ | 3492.9 | 292.5 | 100.0% | 5.13x | 3337 |
| C9 | perf | yes | 2 | ✓ | 3527.1 | 147.5 | 100.0% | 5.89x | 1358 |
| C10 | perf | yes | 8 | ✓ | 3391.9 | 342.6 | 100.0% | 6.23x | 5316 |
| C10 | perf | yes | 16 | ✓ | 2848.5 | 375.9 | 100.0% | 6.20x | 10618 |

#### Qwen3.6-35B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1339.8 | 41.7 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 3181.4 | 263.0 | - | - | 2678 |
| Q8_0 | perf | yes | 1 | ✓ | 3621.2 | 62.0 | 100.0% | 1.88x | 659 |
| C0 | perf | yes | 1 | ✓ | 3639.9 | 76.6 | 100.0% | 2.20x | 659 |
| C1 | perf | yes | 1 | ✓ | 3603.7 | 82.2 | 100.0% | 2.79x | 659 |
| C2 | perf | yes | 1 | ✓ | 3602.4 | 78.4 | 100.0% | 3.23x | 659 |
| C3 | perf | yes | 1 | ✓ | 3662.1 | 81.2 | 100.0% | 3.39x | 659 |
| C4 | perf | yes | 1 | ✓ | 3641.8 | 80.3 | 100.0% | 3.72x | 659 |
| C5 | perf | yes | 1 | ✓ | 3652.9 | 81.7 | 100.0% | 3.81x | 659 |
| C6 | perf | yes | 1 | ✓ | 3646.3 | 78.0 | 100.0% | 4.42x | 659 |
| C7 | perf | yes | 1 | ✓ | 3647.6 | 78.0 | 100.0% | 4.51x | 659 |
| C8 | perf | yes | 5 | ✓ | 3604.3 | 296.5 | 100.0% | 5.04x | 3337 |
| C9 | perf | yes | 2 | ✓ | 3629.5 | 119.1 | 100.0% | 5.75x | 1358 |
| C10 | perf | yes | 8 | ✓ | 3510.2 | 347.6 | 100.0% | 6.05x | 5316 |
| C10 | perf | yes | 16 | ✓ | 3115.7 | 388.6 | 100.0% | 6.04x | 10618 |

#### Qwen3.6-35B AntiLoop+StyleTune (auto/Precision)

The npcd production configuration — AntiLoop trunk under StyleTune's output head,
at `Int8Mode::auto` (Precision on this int8-MMA-less card).

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 | ✓ | 1021.8 | 36.8 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 2303.9 | 221.6 | - | - | 2678 |
| Q8_0 | prec | yes | 1 | ✓ | 2620.8 | 68.9 | 100.0% | 1.88x | 659 |
| C0 | prec | yes | 1 | ✓ | 2752.8 | 77.1 | 100.0% | 2.21x | 659 |
| C1 | prec | yes | 1 | ✓ | 2857.5 | 76.3 | 100.0% | 2.79x | 659 |
| C2 | prec | yes | 1 | ✓ | 2960.0 | 74.6 | 100.0% | 3.24x | 659 |
| C3 | prec | yes | 1 | ✓ | 3112.0 | 78.5 | 100.0% | 3.40x | 659 |
| C4 | prec | yes | 1 | ✓ | 3234.3 | 77.6 | 100.0% | 3.72x | 659 |
| C5 | prec | yes | 1 | ✓ | 3340.1 | 79.5 | 100.0% | 3.82x | 659 |
| C6 | prec | yes | 1 | ✓ | 3453.2 | 75.9 | 100.0% | 4.46x | 659 |
| C7 | prec | yes | 1 | ✓ | 3463.1 | 78.7 | 100.0% | 4.55x | 659 |
| C8 | prec | yes | 5 | ✓ | 3236.1 | 266.0 | 100.0% | 5.06x | 3337 |
| C9 | prec | yes | 2 | ✓ | 3254.2 | 112.3 | 100.0% | 5.79x | 1358 |
| C10 | prec | yes | 8 | ✓ | 3161.3 | 275.1 | 100.0% | 6.09x | 5316 |
| C10 | prec | yes | 16 | ✓ | 2371.2 | 351.6 | 100.0% | 6.08x | 10618 |

#### Qwen3.6-35B AntiLoop+StyleTune (Performance)

The same hybrid at `Int8Mode::Performance` — same-width KO twins — priced against
the `auto`/Precision row above.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 1476.5 | 43.3 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 3305.0 | 252.4 | - | - | 2678 |
| Q8_0 | perf | yes | 1 | ✓ | 3719.1 | 73.0 | 100.0% | 1.88x | 659 |
| C0 | perf | yes | 1 | ✓ | 3739.4 | 81.7 | 100.0% | 2.21x | 659 |
| C1 | perf | yes | 1 | ✓ | 3741.0 | 83.1 | 100.0% | 2.80x | 659 |
| C2 | perf | yes | 1 | ✓ | 3725.4 | 82.1 | 100.0% | 3.23x | 659 |
| C3 | perf | yes | 1 | ✓ | 3739.7 | 77.3 | 100.0% | 3.40x | 659 |
| C4 | perf | yes | 1 | ✓ | 3728.6 | 82.6 | 100.0% | 3.72x | 659 |
| C5 | perf | yes | 1 | ✓ | 3740.0 | 83.4 | 100.0% | 3.82x | 659 |
| C6 | perf | yes | 1 | ✓ | 3724.6 | 81.7 | 100.0% | 4.44x | 659 |
| C7 | perf | yes | 1 | ✓ | 3742.3 | 81.8 | 100.0% | 4.54x | 659 |
| C8 | perf | yes | 5 | ✓ | 3691.7 | 295.7 | 100.0% | 5.05x | 3337 |
| C9 | perf | yes | 2 | ✓ | 3731.4 | 148.5 | 100.0% | 5.77x | 1358 |
| C10 | perf | yes | 8 | ✓ | 3738.5 | 346.7 | 100.0% | 6.10x | 5316 |
| C10 | perf | yes | 16 | ✓ | 3384.8 | 399.0 | 100.0% | 6.07x | 10618 |

#### Qwen3.8-Flash-Next (Q2_KO experts)

Measured on **2026-09-30**, sixteen days after the rest of this section, on build
`e596fad8d` — with the daemon stopped and the card at its idle floor, like the
sweep. The card runs the same **Q2_KO-expert** artifact as the 4090 Mobile (both
sit under the ladder's 32 GiB rung, so the recipe and its digest tag
`130076148f33` are identical); this run used a copy of the 4090 Mobile's file,
which the gate verified against its own recipe before loading. The ×16 rung the
16 GB card skips runs here.

The 3090's host changes the expert tiers, not the format: its 64 GB of RAM pins
**all 24,064 evictable experts** (31.0 GiB warm tier, no shortfall), so every
miss is a host→device upload over PCIe 3.0 and the run read nothing from the
NVMe pack. The 4090 Mobile's 31.5 GiB host cannot hold that tier.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 (cold) | ✓ | 247.3 | 20.2 | - | - | 713 |
| BF16 | prec | yes | 4 | ✓ | 1033.4 | 67.0 | - | - | 2894 |
| BF16 | prec | yes | 8 | ✓ | 895.2 | 113.2 | - | - | 5748 |
| BF16 | prec | yes | 16 | ✓ | 976.5 | 147.0 | - | - | 11482 |
| BF16 | prec | yes | 1 (warm) | ✓ | 524.0 | 24.3 | - | - | 713 |
| C0 | prec | yes | 2 | ✓ | 851.7 | 44.2 | 100.0% | 2.18x | 1466 |
| C5 | prec | yes | 2 | ✓ | 857.7 | 43.4 | 100.0% | 4.01x | 1466 |
| C8 | prec | yes | 2 | ✓ | 862.6 | 43.4 | 100.0% | 4.74x | 1466 |
| C10 | prec | yes | 2 | ✓ | 863.9 | 44.0 | 100.0% | 5.46x | 1466 |
| C10 | prec | yes | 8 | ✓ | 1010.6 | 95.6 | 100.0% | 5.44x | 5748 |

**Against the 4090 Mobile (§3.9), same artifact.** Decode at ×8 is 113.2 against
64.7 t/s (1.7×), and the 3090 adds a ×16 row at 147.0; prefill at ×4 is 1,033.4
against 654.7. The 4090 Mobile's rows are from build `bf291341c`, before the
hot-path prefill work of I17, so the prefill gap is the build as much as the
card; decode is the column to read. Compression matches to the second decimal
place (C10 ×2 5.46× against 5.44×), as it should — the ratio is the model's.

**Engine probe** (`kv_fragmentation::qwen38_flash_next`, §2.2), run after the
gate under the same card-to-itself rule:

| Probe | Story | Worst sustained VRAM efficiency | Weight uptake | Result |
|---|---:|---:|---:|---|
| Qwen3.8-Flash-Next (`qwen38_flash_next`) | 8/8 | 99% (single sample 80%) | 80% of 3,152 MiB released | pass |

Its phase B — eight sequences prefilling and decoding together on the pool
phase A fragmented — delivered 4,024 prefill tokens in 17.90 s (224.8 t/s) and
384 decode tokens in 18.40 s (20.9 t/s aggregate), with the two phases
overlapping rather than summing. That is delivery under churn, at the engine's
own context rather than the gate's 262,144 (§2.2), mid-way through a KV pack and
a weight-zone regrowth; it is not a ceiling and is not comparable with the
gate's ×8 row. `qwen38_flash_next_combined` measures both from one load.

### 3.9 RTX 4090 Mobile 16 GB — the width gate sweep

A single sequential sweep on **2026-09-30**, on the RTX 4090 Mobile (§1), build
`bf291341c`, of the same `test_parallel_batched_forwarding*` gates as §3.7 and
§3.8 — one `cargo test` invocation per model so exactly one was ever resident,
`--release --features cuda`, with the daemon stopped and the card at its idle
floor before each. Like §3.8 it is a **width / throughput** sweep at the gates'
fixed ~700-token prompt, **not** depth, and every cell is a single measurement,
so the 1–4 % noise floor (§5) applies to each on its own. Every gate passed: no
session in any row failed its reproduction check.

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
| Qwen2-0.5B | 46,762.1 | 3,565.1 | — (no ladder) |
| Qwen3.5-0.8B | 22,485.3 | 993.0 | 4.11× (C10) |
| Llama-3.2-3B | 7,923.9 | 412.4 | 4.35× (C10) |
| Llama-2-7B | 4,172.6 | 693.5 | 3.56× (Q4_0) |
| Qwen3-8B | 3,772.0 | 230.1 | 5.84× (C10) |
| Qwen3.5-9B | 3,468.3 | 286.6 | 5.13× (C10) |
| Qwen3.8-27B | 1,118.2 | 76.9 | 4.76× (C10) |
| Qwen3-30B-A3B | 4,088.1 | 96.2 | 5.42× (C10) |
| Qwen3.5-35B-A3B | 2,305.7 | 131.2 | 6.23× (C10) |
| Qwen3.6-35B-A3B | 2,406.3 | 129.7 | 5.96× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (auto/Precision) | 2,141.5 | 115.9 | 6.00× (C10) |
| Qwen3.6-35B AntiLoop+StyleTune (Performance) | 2,526.7 | 131.6 | 6.00× (C10) |
| Qwen3.8-Flash-Next (Q2_KO experts) | 654.7 | 64.7 | 5.44× (C10) |

Llama-2-7B's best compression is from a row the gate does not validate for
reproduction (`-`); its sessions still passed.

**Against the RTX 3090, on the models both ran.** Read with care: the two sweeps
are sixteen days and many commits apart, and the 3090's was measured on a build
carrying the decode-slot refresh regression and before the gates synchronised
their prompt timer (§3.8, §4) — both of which understate its prefill, the
latter by 12–22% on one- and two-context rows. So a prefill gap between the two
cards is not an architectural result.

- **Dense-model prefill is higher on the 16 GB card** — Qwen2-0.5B 46,762 vs
  23,763, Qwen3.5-0.8B 22,485 vs 18,469, Llama-3.2-3B 7,924 vs 5,327 (the 3090's
  without flash-attn), Llama-2-7B 4,173 vs 2,953, Qwen3-8B 3,772 vs 2,707,
  Qwen3.5-9B 3,468 vs 2,908, Qwen3.8-27B 1,118 vs 941. The regression cost the
  72 GB card up to 58% of Qwen2-0.5B's prefill, which is enough to cover these
  gaps, so how much of each is the card is not separable from these two sweeps.
  Dense decode is mixed and mostly lower here at one context (e.g. Qwen3.5-0.8B
  41.9 vs 103–135 t/s), which this sweep does not explain; it is recorded, not
  attributed.
- **The MoE models are several times slower** — Qwen3.5/3.6-35B best decode
  129.7–131.6 vs 375.9–388.6 t/s, Qwen3-30B 96.2 vs 274.5 — and decode, which the
  regression did not touch, is the column that shows it. Expert streaming is the
  expected cause (above): a bigger card buys speed, not feasibility.
- **Compression tracks the model, not the card**, as §3.8 found against the
  72 GB card: C10 on the two 35Bs is 6.23×/5.96× here against the 3090's
  6.23×/6.05×, and every dense rung is within 0.1× of the 3090's.

**Flash-Next against the 72 GB card** is not a like-for-like comparison on two
counts. That card runs Q4_KOEXP experts almost wholly resident and this one
Q2_KO experts streamed, and its §3.6 *Width* ladder predates the Flash-Next
hot-path changes this build carries. So the gap — BF16 ×8 decode 64.7 here
against 393.1 † there — is the expert tier and the build more than the card.

**Engine probes.** Both `kv_fragmentation` probes (§2.2) were run after the
gates, under the same card-to-itself rule:

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
| F32 | off | no | 1 | - | 22879.5 | 187.9 | - | - | 308 |
| BF16 | off | yes | 1 | - | 25215.6 | 199.2 | - | - | 308 |
| F16 | off | yes | 1 | - | 24391.0 | 175.9 | - | - | 308 |
| F16 | off | yes | 4 | - | 44515.4 | 716.1 | - | - | 1272 |
| F16 | off | yes | 60 | - | 46762.1 | 3565.1 | - | - | 18612 |
| BF16 | off | yes | 60 | - | 46436.4 | 3469.7 | - | - | 18612 |

#### Qwen3.5-0.8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 15740.2 | 41.9 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 15566.1 | 41.5 | - | - | 659 |
| BF16 | prec | yes | 16 | ✓ | 21193.1 | 608.0 | - | - | 10618 |
| Q8_0 | prec | yes | 4 | ✓ | 21677.9 | 202.3 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 2 | ✓ | 22485.3 | 102.8 | 100.0% | 1.85x | 1358 |
| C1 | prec | yes | 2 | ✓ | 22405.7 | 105.6 | 100.0% | 2.01x | 1358 |
| C2 | prec | yes | 2 | ✓ | 22399.6 | 101.9 | 100.0% | 2.26x | 1358 |
| C3 | prec | yes | 2 | ✓ | 22363.1 | 109.8 | 100.0% | 2.40x | 1358 |
| C4 | prec | yes | 2 | ✓ | 20752.8 | 105.4 | 100.0% | 2.58x | 1358 |
| C5 | prec | yes | 2 | ✓ | 20260.9 | 103.0 | 100.0% | 2.76x | 1358 |
| C6 | prec | yes | 2 | ✓ | 19577.5 | 103.1 | 100.0% | 2.86x | 1358 |
| C7 | prec | yes | 2 | ✓ | 18960.0 | 104.3 | 100.0% | 3.61x | 1358 |
| C8 | prec | yes | 32 | ✓ | 21847.0 | 993.0 | 100.0% | 3.83x | 21202 |
| C9 | prec | yes | 5 | ✓ | 21768.4 | 242.9 | 100.0% | 4.08x | 3337 |
| C10 | prec | yes | 10 | ✓ | 22228.5 | 407.5 | 100.0% | 4.11x | 6640 |

#### Llama-3.2-3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | prec | no | 1 | ✓ | 7330.7 | 66.8 | - | - | 654 |
| F16 | prec | yes | 1 | ✓ | 7262.1 | 70.9 | - | - | 654 |
| F16 | prec | yes | 4 | ✓ | 7705.2 | 265.4 | - | - | 2658 |
| R16 | prec | yes | 1 | ✓ | 7749.9 | 70.2 | 0.0% | - | 654 |
| Q8_0 | prec | yes | 1 | ✓ | 7777.7 | 74.7 | 100.0% | 1.88x | 654 |
| Q8_Q4 | prec | yes | 1 | ✓ | 7923.9 | 65.2 | 100.0% | 2.29x | 654 |
| BF16 | prec | yes | 4 | ✓ | 7460.8 | 275.9 | - | - | 2658 |
| Q8_1 | prec | yes | 4 | ✓ | 7545.2 | 235.2 | 100.0% | 1.78x | 2658 |
| Q8_KS | prec | yes | 4 | ✓ | 7708.7 | 258.1 | 100.0% | 1.78x | 2658 |
| Q8_Q4 | prec | yes | 4 | ✓ | 7672.3 | 236.7 | 100.0% | 2.29x | 2658 |
| Q4_0 | prec | yes | 4 | - | 7714.2 | 258.3 | 100.0% | 3.56x | 2658 |
| Q4_1 | prec | yes | 4 | - | 7598.3 | 229.8 | 100.0% | 3.20x | 2658 |
| Q4_KS | prec | yes | 4 | - | 7634.9 | 245.5 | 100.0% | 3.20x | 2658 |
| C0 | prec | yes | 1 | ✓ | 7586.8 | 69.0 | 100.0% | 1.87x | 654 |
| C1 | prec | yes | 1 | ✓ | 7052.9 | 69.2 | 100.0% | 2.23x | 654 |
| C2 | prec | yes | 1 | ✓ | 7237.8 | 67.6 | 100.0% | 2.41x | 654 |
| C3 | prec | yes | 1 | ✓ | 7920.0 | 67.0 | 100.0% | 2.78x | 654 |
| C4 | prec | yes | 1 | ✓ | 7355.2 | 64.8 | 100.0% | 3.11x | 654 |
| C5 | prec | yes | 1 | ✓ | 7537.8 | 66.2 | 100.0% | 3.32x | 654 |
| C6 | prec | yes | 1 | ✓ | 7458.4 | 66.6 | 100.0% | 3.51x | 654 |
| C7 | prec | yes | 1 | ✓ | 7436.0 | 66.4 | 100.0% | 3.87x | 654 |
| C8 | prec | yes | 10 | ✓ | 7586.3 | 412.4 | 100.0% | 3.95x | 6590 |
| C9 | prec | yes | 10 | ✓ | 7408.5 | 411.3 | 100.0% | 4.27x | 6590 |
| C10 | prec | yes | 5 | ✓ | 7850.8 | 266.7 | 100.0% | 4.35x | 3312 |

#### Llama-2-7B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F32 | perf | no | 1 | - | 4172.6 | 52.3 | - | - | 283 |
| F16 | perf | yes | 1 | - | 3788.2 | 50.2 | - | - | 283 |
| F16 | perf | yes | 4 | - | 3973.5 | 179.4 | - | - | 1176 |
| F16 | perf | yes | 8 | - | 3723.4 | 296.2 | - | - | 2316 |
| BF16 | perf | yes | 1 | - | 3808.7 | 53.1 | - | - | 283 |
| BF16 | perf | yes | 8 | - | 3686.6 | 296.3 | - | - | 2316 |
| BF16 | perf | yes | 16 | - | 3190.5 | 449.7 | - | - | 4624 |
| BF16 | perf | yes | 48 | - | 1902.2 | 693.5 | - | - | 13796 |
| Q8_0 | perf | yes | 32 | - | 2464.6 | 436.0 | 100.0% | 1.88x | 9220 |
| Q4_0 | perf | yes | 32 | - | 2339.0 | 448.2 | 100.0% | 3.56x | 9220 |

#### Qwen3-8B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | no | 1 | ✓ | 3772.0 | 36.3 | - | - | 636 |
| F16 | perf | yes | 1 | ✓ | 3381.3 | 37.2 | - | - | 636 |
| F16 | perf | yes | 2 | ✓ | 3614.5 | 69.0 | - | - | 1312 |
| BF16 | perf | yes | 4 | ✓ | 3640.1 | 137.8 | - | - | 2586 |
| Q8_0 | perf | yes | 4 | ✓ | 3606.9 | 133.6 | 100.0% | 1.88x | 2586 |
| C0 | perf | yes | 1 | ✓ | 3519.6 | 35.5 | 100.0% | 1.90x | 636 |
| C1 | perf | yes | 1 | ✓ | 3573.7 | 36.9 | 100.0% | 2.54x | 636 |
| C2 | perf | yes | 1 | ✓ | 3407.4 | 35.7 | 100.0% | 2.67x | 636 |
| C3 | perf | yes | 1 | ✓ | 3539.9 | 35.5 | 100.0% | 2.92x | 636 |
| C4 | perf | yes | 1 | ✓ | 3501.3 | 34.2 | 100.0% | 3.31x | 636 |
| C5 | perf | yes | 1 | ✓ | 3445.8 | 33.3 | 100.0% | 3.67x | 636 |
| C6 | perf | yes | 1 | ✓ | 3400.7 | 33.5 | 100.0% | 4.30x | 636 |
| C7 | perf | yes | 1 | ✓ | 3544.6 | 33.3 | 100.0% | 4.48x | 636 |
| C8 | perf | yes | 10 | ✓ | 3482.1 | 230.1 | 100.0% | 4.85x | 6410 |
| C9 | perf | yes | 5 | ✓ | 3611.0 | 140.6 | 100.0% | 5.46x | 3222 |
| C10 | perf | yes | 5 | ✓ | 3708.2 | 143.7 | 100.0% | 5.84x | 3222 |

#### Qwen3.5-9B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | prec | yes | 1 | ✓ | 3312.3 | 24.8 | - | - | 659 |
| BF16 | prec | yes | 1 | ✓ | 3294.5 | 23.5 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 3422.1 | 95.1 | - | - | 2678 |
| Q8_0 | prec | yes | 4 | ✓ | 3402.3 | 97.5 | 100.0% | 1.88x | 2678 |
| C0 | prec | yes | 1 | ✓ | 3179.2 | 25.8 | 100.0% | 2.12x | 659 |
| C1 | prec | yes | 1 | ✓ | 3254.9 | 24.3 | 100.0% | 2.43x | 659 |
| C2 | prec | yes | 1 | ✓ | 3269.2 | 23.6 | 100.0% | 2.82x | 659 |
| C3 | prec | yes | 1 | ✓ | 3265.5 | 21.7 | 100.0% | 3.12x | 659 |
| C4 | prec | yes | 1 | ✓ | 3252.0 | 24.3 | 100.0% | 3.45x | 659 |
| C5 | prec | yes | 1 | ✓ | 3257.1 | 24.1 | 100.0% | 3.62x | 659 |
| C6 | prec | yes | 1 | ✓ | 2991.1 | 24.2 | 100.0% | 3.95x | 659 |
| C7 | prec | yes | 1 | ✓ | 3312.7 | 25.8 | 100.0% | 4.03x | 659 |
| C8 | prec | yes | 20 | ✓ | 3273.5 | 286.6 | 100.0% | 4.49x | 13278 |
| C9 | prec | yes | 5 | ✓ | 3468.3 | 107.1 | 100.0% | 5.00x | 3337 |
| C10 | prec | yes | 10 | ✓ | 3340.9 | 193.7 | 100.0% | 5.13x | 6640 |

#### Qwen3.8-27B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 890.0 | 7.0 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 1095.2 | 76.9 | - | - | 2678 |
| Q8_0 | perf | yes | 4 | ✓ | 1094.9 | 71.5 | 100.0% | 1.88x | 2678 |
| C0 | perf | yes | 1 | ✓ | 1065.5 | 18.3 | 100.0% | 2.11x | 659 |
| C1 | perf | yes | 1 | ✓ | 1068.6 | 20.3 | 100.0% | 2.32x | 659 |
| C2 | perf | yes | 1 | ✓ | 1074.0 | 19.3 | 100.0% | 2.68x | 659 |
| C3 | perf | yes | 1 | ✓ | 1065.4 | 20.9 | 100.0% | 3.04x | 659 |
| C4 | perf | yes | 1 | ✓ | 1069.7 | 19.3 | 100.0% | 3.37x | 659 |
| C5 | perf | yes | 1 | ✓ | 1070.8 | 20.2 | 100.0% | 3.56x | 659 |
| C6 | perf | yes | 1 | ✓ | 1063.9 | 20.3 | 100.0% | 3.76x | 659 |
| C7 | perf | yes | 1 | ✓ | 1077.4 | 20.3 | 100.0% | 3.83x | 659 |
| C8 | perf | yes | 20 | ✓ | 796.1 | 28.6 | 100.0% | 4.27x | 13278 |
| C9 | perf | yes | 5 | ✓ | 1118.2 | 38.1 | 100.0% | 4.68x | 3337 |
| C10 | perf | yes | 10 | ✓ | 1109.9 | 60.5 | 100.0% | 4.76x | 6640 |

#### Qwen3-30B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| F16 | perf | yes | 1 | ✓ | 782.9 | 6.1 | - | - | 626 |
| BF16 | perf | yes | 1 | ✓ | 1547.0 | 7.4 | - | - | 626 |
| BF16 | perf | yes | 10 | ✓ | 3836.7 | 66.8 | - | - | 6310 |
| Q8_0 | perf | yes | 20 | ✓ | 3999.7 | 96.2 | 100.0% | 1.88x | 12620 |
| Q4_0 | perf | yes | 4 | - | 3605.1 | 26.6 | 100.0% | 3.56x | 2546 |
| C0 | perf | yes | 2 | ✓ | 2495.2 | 13.3 | 100.0% | 1.98x | 1292 |
| C1 | perf | yes | 2 | ✓ | 2478.0 | 13.5 | 100.0% | 2.50x | 1292 |
| C2 | perf | yes | 2 | ✓ | 2484.2 | 12.6 | 100.0% | 2.71x | 1292 |
| C3 | perf | yes | 2 | ✓ | 2475.0 | 13.8 | 100.0% | 2.96x | 1292 |
| C4 | perf | yes | 2 | ✓ | 2491.6 | 13.3 | 100.0% | 3.37x | 1292 |
| C5 | perf | yes | 2 | ✓ | 2498.8 | 13.5 | 100.0% | 3.63x | 1292 |
| C6 | perf | yes | 2 | ✓ | 2510.8 | 13.2 | 100.0% | 4.06x | 1292 |
| C7 | perf | yes | 2 | ✓ | 2521.7 | 13.0 | 100.0% | 4.13x | 1292 |
| C8 | perf | yes | 2 | ✓ | 2526.0 | 12.9 | 100.0% | 4.60x | 1292 |
| C9 | perf | yes | 2 | ✓ | 2526.3 | 12.9 | 100.0% | 5.18x | 1292 |
| C10 | perf | yes | 2 | ✓ | 2546.7 | 12.3 | 100.0% | 5.42x | 1292 |
| BF16 | perf | yes | 1 | ✓ | 1713.9 | 6.5 | - | - | 626 |
| Q4_0 | perf | yes | 20 | - | 4088.1 | 83.8 | 100.0% | 3.56x | 12620 |

#### Qwen3.5-35B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 304.2 | 9.6 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 2046.5 | 47.6 | - | - | 2678 |
| Q8_0 | perf | yes | 2 | ✓ | 1371.5 | 24.1 | 100.0% | 1.88x | 1358 |
| C0 | perf | yes | 1 | ✓ | 824.8 | 13.5 | 100.0% | 2.20x | 659 |
| C1 | perf | yes | 1 | ✓ | 828.2 | 12.8 | 100.0% | 2.85x | 659 |
| C2 | perf | yes | 1 | ✓ | 827.6 | 13.1 | 100.0% | 3.26x | 659 |
| C3 | perf | yes | 1 | ✓ | 831.1 | 13.4 | 100.0% | 3.41x | 659 |
| C4 | perf | yes | 1 | ✓ | 828.6 | 12.9 | 100.0% | 3.72x | 659 |
| C5 | perf | yes | 1 | ✓ | 837.8 | 13.6 | 100.0% | 3.83x | 659 |
| C6 | perf | yes | 1 | ✓ | 835.8 | 13.4 | 100.0% | 4.51x | 659 |
| C7 | perf | yes | 1 | ✓ | 832.0 | 13.0 | 100.0% | 4.61x | 659 |
| C8 | perf | yes | 5 | ✓ | 2305.7 | 54.6 | 100.0% | 5.13x | 3337 |
| C9 | perf | yes | 2 | ✓ | 1375.6 | 23.4 | 100.0% | 5.89x | 1358 |
| C10 | perf | yes | 8 | ✓ | 2088.2 | 78.6 | 100.0% | 6.23x | 5316 |
| C10 | perf | yes | 16 | ✓ | 752.7 | 131.2 | 100.0% | 6.20x | 10618 |

#### Qwen3.6-35B-A3B

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 325.9 | 8.8 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 2185.7 | 42.7 | - | - | 2678 |
| Q8_0 | perf | yes | 1 | ✓ | 880.9 | 10.8 | 100.0% | 1.88x | 659 |
| C0 | perf | yes | 1 | ✓ | 893.7 | 11.5 | 100.0% | 2.20x | 659 |
| C1 | perf | yes | 1 | ✓ | 893.0 | 11.8 | 100.0% | 2.76x | 659 |
| C2 | perf | yes | 1 | ✓ | 895.4 | 11.5 | 100.0% | 3.22x | 659 |
| C3 | perf | yes | 1 | ✓ | 897.6 | 11.7 | 100.0% | 3.39x | 659 |
| C4 | perf | yes | 1 | ✓ | 894.1 | 11.3 | 100.0% | 3.70x | 659 |
| C5 | perf | yes | 1 | ✓ | 905.2 | 11.8 | 100.0% | 3.80x | 659 |
| C6 | perf | yes | 1 | ✓ | 904.6 | 12.0 | 100.0% | 4.37x | 659 |
| C7 | perf | yes | 1 | ✓ | 895.8 | 12.0 | 100.0% | 4.46x | 659 |
| C8 | perf | yes | 5 | ✓ | 2406.3 | 49.3 | 100.0% | 4.99x | 3337 |
| C9 | perf | yes | 2 | ✓ | 1454.5 | 22.0 | 100.0% | 5.68x | 1358 |
| C10 | perf | yes | 8 | ✓ | 2179.3 | 77.8 | 100.0% | 5.96x | 5316 |
| C10 | perf | yes | 16 | ✓ | 808.0 | 129.7 | 100.0% | 5.94x | 10618 |

#### Qwen3.6-35B AntiLoop+StyleTune (auto/Precision)

The npcd production configuration — AntiLoop trunk under StyleTune's output head,
at `Int8Mode::auto` (Precision on this int8-MMA card).

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 | ✓ | 259.6 | 8.3 | - | - | 659 |
| BF16 | prec | yes | 4 | ✓ | 1763.9 | 44.8 | - | - | 2678 |
| Q8_0 | prec | yes | 1 | ✓ | 737.6 | 12.1 | 100.0% | 1.88x | 659 |
| C0 | prec | yes | 1 | ✓ | 750.9 | 12.7 | 100.0% | 2.21x | 659 |
| C1 | prec | yes | 1 | ✓ | 750.2 | 12.6 | 100.0% | 2.76x | 659 |
| C2 | prec | yes | 1 | ✓ | 740.1 | 12.3 | 100.0% | 3.23x | 659 |
| C3 | prec | yes | 1 | ✓ | 742.9 | 12.8 | 100.0% | 3.40x | 659 |
| C4 | prec | yes | 1 | ✓ | 746.2 | 12.4 | 100.0% | 3.70x | 659 |
| C5 | prec | yes | 1 | ✓ | 748.2 | 13.1 | 100.0% | 3.80x | 659 |
| C6 | prec | yes | 1 | ✓ | 755.4 | 12.1 | 100.0% | 4.41x | 659 |
| C7 | prec | yes | 1 | ✓ | 755.7 | 12.4 | 100.0% | 4.49x | 659 |
| C8 | prec | yes | 5 | ✓ | 2141.5 | 51.8 | 100.0% | 5.01x | 3337 |
| C9 | prec | yes | 2 | ✓ | 1226.4 | 24.3 | 100.0% | 5.71x | 1358 |
| C10 | prec | yes | 8 | ✓ | 1924.1 | 71.5 | 100.0% | 6.00x | 5316 |
| C10 | prec | yes | 16 | ✓ | 688.1 | 115.9 | 100.0% | 5.98x | 10618 |

#### Qwen3.6-35B AntiLoop+StyleTune (Performance)

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | perf | yes | 1 | ✓ | 647.6 | 8.7 | - | - | 659 |
| BF16 | perf | yes | 4 | ✓ | 2248.8 | 41.1 | - | - | 2678 |
| Q8_0 | perf | yes | 1 | ✓ | 943.7 | 11.4 | 100.0% | 1.88x | 659 |
| C0 | perf | yes | 1 | ✓ | 971.3 | 13.9 | 100.0% | 2.21x | 659 |
| C1 | perf | yes | 1 | ✓ | 969.3 | 14.0 | 100.0% | 2.76x | 659 |
| C2 | perf | yes | 1 | ✓ | 959.7 | 12.5 | 100.0% | 3.22x | 659 |
| C3 | perf | yes | 1 | ✓ | 971.9 | 13.6 | 100.0% | 3.40x | 659 |
| C4 | perf | yes | 1 | ✓ | 983.2 | 14.2 | 100.0% | 3.70x | 659 |
| C5 | perf | yes | 1 | ✓ | 974.4 | 13.3 | 100.0% | 3.81x | 659 |
| C6 | perf | yes | 1 | ✓ | 986.8 | 13.4 | 100.0% | 4.39x | 659 |
| C7 | perf | yes | 1 | ✓ | 991.2 | 14.2 | 100.0% | 4.47x | 659 |
| C8 | perf | yes | 5 | ✓ | 2526.7 | 58.4 | 100.0% | 5.01x | 3337 |
| C9 | perf | yes | 2 | ✓ | 1542.7 | 23.0 | 100.0% | 5.70x | 1358 |
| C10 | perf | yes | 8 | ✓ | 2323.9 | 77.6 | 100.0% | 6.00x | 5316 |
| C10 | perf | yes | 16 | ✓ | 899.9 | 131.6 | 100.0% | 5.97x | 10618 |

#### Qwen3.8-Flash-Next (Q2_KO experts)

The gate runs BF16 ×1 twice — at the head of its ladder (cold) and again after
×8 (warm) — the same pair §3.6 *Width* labels.

| KvMode | int8 | Batched | Ctx | Valid | prefill t/s | decode t/s | %Quant | Compress | Peak tok |
|---|---|:-:|--:|:-:|--:|--:|--:|--:|--:|
| BF16 | prec | yes | 1 (cold) | ✓ | 134.2 | 16.1 | - | - | 713 |
| BF16 | prec | yes | 4 | ✓ | 654.7 | 55.8 | - | - | 2894 |
| BF16 | prec | yes | 8 | ✓ | 495.4 | 64.7 | - | - | 5748 |
| BF16 | prec | yes | 1 (warm) | ✓ | 244.2 | 20.0 | - | - | 713 |
| C0 | prec | yes | 2 | ✓ | 422.8 | 32.8 | 100.0% | 2.18x | 1466 |
| C5 | prec | yes | 2 | ✓ | 423.1 | 34.3 | 100.0% | 4.01x | 1466 |
| C8 | prec | yes | 2 | ✓ | 424.1 | 33.4 | 100.0% | 4.74x | 1466 |
| C10 | prec | yes | 2 | ✓ | 430.2 | 34.9 | 100.0% | 5.44x | 1466 |
| C10 | prec | yes | 8 | ✓ | 497.9 | 59.5 | 100.0% | 5.43x | 5748 |

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

A † cell in §3.6 *Width* or §3.7 is the 72 GB 2026-09-13 file's value, a ◆ cell
the higher of the two 2026-09-15 files' values; every other 72 GB width cell is
the first file's. The 2026-09-15 files list gates in sweep order, which puts
Llama-2-7B before Qwen3-8B and Qwen3.8-27B before Qwen3-30B-A3B. The 3090 and
4090 Mobile TSVs hold the width-sweep axis only — their `depth` column is blank
and `prompt_tokens` is `~700`, the gate's fixed prompt. The 4090 Mobile's engine
probes (§3.9) are not rows of that file: they report story, efficiency and
uptake rather than a ladder, and §3.9 carries them; the same holds for the
3090's Flash-Next probe, which §3.8 carries. Reproduce any row with the
command in its test's `#[ignore]` attribute.

| Table | Test |
|---|---|
| §3.2, §3.3, §3.4 | each model's `long_context_*` gate (72 GB) |
| §3.5 | the same gates, C-mode rows (72 GB) |
| §3.6 Coherence | `quantized_qwen38_moe::tests::profile_decode_vs_depth` (72 GB) |
| §3.6 Rewrite | `quantized_qwen38_moe::tests::profile_story_rewrite_vs_depth` (72 GB) |
| §3.6 Width, §3.7 | `test_parallel_batched_forwarding*` (72 GB) |
| §3.8 | `test_parallel_batched_forwarding*` and `kv_fragmentation::qwen38_flash_next` (RTX 3090) |
| §3.9 | `test_parallel_batched_forwarding*` and `kv_fragmentation::{qwen3_30b_a3b_q4, qwen38_flash_next}` (RTX 4090 Mobile) |

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
vLLM, SGLang, KTransformers, ExLlamaV2, TensorRT-LLM), collected 2026-09-30 for
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
| RTX 3090 **(ours, §3.8)** | this engine, BF16 ×1 / BF16 ×48 | Q4_0 | BF16 | 2,894.8 / 1,433.0 | 90.9 / 547.6 (aggregate) | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, BF16 ×1 / BF16 ×48 | Q4_0 | BF16 | 3,808.7 / 1,902.2 | 53.1 / 693.5 (aggregate) | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, ×1 / ×48 | Q4_0 | BF16 | 6,063.7 / 3,679.8 | 97.2 / 917.3 (aggregate) | — |

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
| RTX 3090 **(ours, §3.8)** | this engine, F16 ×1 / C8 ×10 (no flash-attn) | Q4_K_M | F16 / C8 | ~700-tok prompt | 5,279.1 / 4,179.3 | 130.7 / 542.7 | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, F16 ×1 / C8 ×10 | Q4_K_M | F16 / C8 | ~700-tok prompt | 7,262.1 / 7,586.3 | 70.9 / 412.4 | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, C0 ×1 / C8 ×10 | Q4_K_M | C0 / C8 | ~700-tok prompt | 13,166.0 / 13,977.4 | 130.5 / 745.1 | — |

No source measured Llama-3.2-3B on an RTX 3090.

**Qwen2-0.5B and Qwen3.5-0.8B.** Neither Qwen2-0.5B itself nor Qwen3.5-0.8B on a
target card was found; the nearest published rows:

| GPU | Model | Engine | Weights | Workload | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---:|---:|---|
| *RTX 4080 16 GB* | *Qwen2.5-0.5B-Instruct* | llama.cpp CUDA | Q4_K_M | pp512 / tg128 | 39,230.93 | 496.01 | E13 |
| *RTX 4090 24 GB* | *Qwen2.5-0.5B* | vLLM, `vllm bench serve`, cold prefix cache | BF16 | 1,024 in / 64 out, concurrency 32 | — | 16,349.78 total tok/s | E14 |
| *RTX 5080 16 GB* | Qwen3.5-0.8B | llama.cpp | Q8_0 | tg128 | — | 460.86 | E16 |
| *RTX 5090* | Qwen3.5-0.8B | llama.cpp, MMVQ / MMQ kernel | NVFP4 | pp512 / tg128 | 23,799.93 / 32,858.97 | 392.92 / 389.75 | E15 |
| RTX 3090 **(ours)** | Qwen2-0.5B / Qwen3.5-0.8B | this engine, ×1 | Q4_0 / Q6_K | ~700-tok prompt | 15,752.4 / 15,114.3 | 247.2 / 134.6 | — |
| RTX 4090 Mobile **(ours)** | Qwen2-0.5B / Qwen3.5-0.8B | this engine, ×1 | Q4_0 / Q6_K | ~700-tok prompt | 25,215.6 / 15,566.1 | 199.2 / 41.5 | — |
| RTX PRO 5000 **(ours)** | Qwen2-0.5B / Qwen3.5-0.8B | this engine, ×1 | Q4_0 / Q6_K | ~700-tok prompt | 31,605.8 / 24,626.5 | 256.7 / 168.8 | — |

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
2,707.2 / 69.8 and C8 ×10 303.4 aggregate; RTX 4090 Mobile 3,381.3 / 37.2 and C8
×10 230.1 aggregate; RTX PRO 5000 6,008.5 / 67.3 and C8 ×10 460.1 aggregate
(Q6_K weights, ~700-token prompt).

**Qwen3.5-9B.** No trustworthy measurement exists on any of the three card
classes. The nearest: *RTX 5090*, llama.cpp Q4_K_M at 4K, "~10,400" prefill /
"~186" decode (E24, a secondary compilation); *GB10 / DGX Spark*, llama.cpp
Q4_K_M pp512 2,558.98 / tg128 35.41 (E25). Ours: RTX 3090 2,907.7 / 93.7 at ×1;
RTX 4090 Mobile 3,294.5 / 23.5; RTX PRO 5000 5,534.1 / 121.0.

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
| RTX 3090 **(ours, §3.8)** | this engine, BF16 ×1 / C9 ×5 | Q6_K/Q8 | BF16 / C9 | ~700-tok prompt | 940.5 / 842.2 | 50.6 / 198.9 | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, BF16 ×1 / BF16 ×4 / C10 ×10 | Q6_K/Q8 | BF16 / C10 | ~700-tok prompt | 890.0 / 1,095.2 / 1,109.9 | 7.0 / 76.9 / 60.5 (aggregate at ×4, ×10) | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, ×1 / C10 ×40 | Q6_K/Q8 | BF16 / C10 | ~700-tok prompt | 1,729.8 / 1,718.7 | 61.0 / 458.1 | — |

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
| RTX 3090 **(ours, §3.8)** | this engine, BF16 ×1 / BF16 ×10 / Q8_0 ×20 | Q4_K_M | expert cache | BF16 / Q8_0 | ~700-tok prompt | 2,987.2 / 3,886.6 / 3,873.0 | 40.3 / 256.6 / 274.5 aggregate | — |
| RTX 4090 Mobile **(ours, §3.9)** | this engine, BF16 ×1 / Q8_0 ×20 | Q4_K_M | experts streamed | BF16 / Q8_0 | ~700-tok prompt | 1,547.0 / 3,999.7 | 7.4 / 96.2 aggregate | — |
| RTX PRO 5000 **(ours, §3.7)** | this engine, ×1 / Q8_0 ×20 | Q4_K_M | expert cache | BF16 / Q8_0 | ~700-tok prompt | 8,307.4 / 9,950.6 | 80.7 / 595.7 aggregate | — |

No source ran Qwen3-30B-A3B with its experts offloaded on a 16 GB card.

**Qwen3.5-35B-A3B and Qwen3.6-35B-A3B**, batch 1 unless noted:

| GPU | Model | Engine | Weights | Offload | KV | Context | Prefill t/s | Decode t/s | Ref |
|---|---|---|---|---|---|---|---:|---:|---|
| RTX 3090 | 3.5 | llama.cpp, `-fa 1` | MXFP4 | none | n/s | 4K / 32K / 128K | 2,622.1 / 2,121.6 / 1,288.9 | 111.2 / 101.2 / 79.4 | E17 |
| RTX 3090 | 3.5 | llama.cpp / Ollama | Q4_K_M | none | **q8_0** 65K / **f16** 32K | short prompt | — | 142.2 / 98.9 (different engines) | E40 |
| RTX 3090 | 3.6 | llama.cpp CUDA | UD-IQ4_NL_XL | none / **FFNs of 10 layers on CPU** | n/s | 89,600 / 262,144 | 3,360.4 / 1,153.5 | 139.6 / 89.1 | E41 |
| RTX 3090 | 3.6 | llama.cpp b9008 | Q4_K_M / Q5_K_M | none / **`--n-cpu-moe 10`** | **q8_0** | 16K / n/s | — / 598.13 | 148.64 / 105.05 | E42 |
| RTX 3090 | 3.6 | llama.cpp b10088 | UD-Q4_K_M | none | n/s | 8K | 3,674 | 157.66 | E43 |
| *RTX 5070 Ti 16 GB* | 3.6 | llama.cpp | 10.88 GB quant | none (`-ngl 40`) | **q8_0** | 32,768 | 407 | 121 | E44 |
| *RTX 3060 12 GB* | 3.6 | llama.cpp b10088 | UD-Q4_K_M | **`--n-cpu-moe 24`** | n/s | 8K | 413 | 38.9 | E43 |
| *RTX 4080 16 GB* / *RTX 4090 24 GB* | 3.6 | llama.cpp (ByteShape) | IQ3_S / IQ4_XS | none | n/s | n/s | — | 183.29 / 214.54; **MTP** 249.33 / 285.53 | E45 |
| *RTX 5090* | 3.5 | llama.cpp | UD-Q4_K_XL | none | **q8_0** | 512–32,768 | 6,461–6,960 | 194.0 | E46 |
| *RTX PRO 6000 Blackwell* | 3.5 | vLLM | FP8 | none | full | 1K / 256K; 10 concurrent | 34,509 peak | 160.3 / 97.7; 598.5 aggregate | E47 |
| *RTX PRO 6000 Blackwell* | 3.6 | vLLM | FP8 | none | full | 1K / 32K / 256K; 5 concurrent | 41,105 | 196.4 / 183 / 116.3; 449.0 aggregate | E48 |
| RTX 3090 **(ours, §3.8)** | 3.5 / 3.6 | this engine, BF16 ×1 / C10 ×16 | Q6_K | expert cache | BF16 / C10 | ~700-tok prompt | 1,267.1 / 1,339.8 (×1) | 41.8 / 41.7 (×1); 375.9 / 388.6 (×16) | — |
| RTX 4090 Mobile **(ours, §3.9)** | 3.5 / 3.6 | this engine, BF16 ×1 / C10 ×16 | Q6_K | experts streamed | BF16 / C10 | ~700-tok prompt | 304.2 / 325.9 (×1) | 9.6 / 8.8 (×1); 131.2 / 129.7 (×16) | — |
| RTX PRO 5000 **(ours, §3.7)** | 3.5 / 3.6 | this engine, ×1 / C10 ×64 | Q6_K | expert cache | BF16 / C10 | ~700-tok prompt | 7,231.9 / 7,310.2 | 109.4 / 107.8 (×1); 1,187.7 / 1,201.6 (×64) | — |

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
| RTX 3090 **(ours, §3.8)** | i7-10700K, 64 GB | this engine, BF16 ×1 warm / ×8 / ×16 | Q2_KO experts | experts streamed VRAM→pinned RAM | BF16 | ~700-tok prompt | 524.0 / 895.2 / 976.5 | 24.3 / 113.2 / 147.0 (**MTP**) | — |
| RTX 4090 Mobile **(ours, §3.9)** | Core Ultra 9 185H, 32 GB | this engine, BF16 ×1 warm / ×8 / C10 ×8 | Q2_KO experts | experts streamed VRAM→RAM→NVMe | BF16 / C10 | ~700-tok prompt | 244.2 / 495.4 / 497.9 | 20.0 / 64.7 / 59.5 (**MTP**) | — |
| RTX PRO 5000 **(ours, §3.6)** | Ryzen 9 9950X3D, 189 GB | this engine, ×1 warm / ×8 | Q4_KOEXP | resident | BF16 | ~700-tok prompt | 1,705.7 / 1,880.9 | 86.9 / 393.1 (**MTP**) | — |

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
| RTX PRO 5000 **(ours, §3.7)** | Ryzen 9 9950X3D, 189 GB | this engine, ×1 warm / ×16 | MXFP4_KO | expert cache | BF16 | ~700-tok prompt | 333.3 / 1,120.6 | 15.1 / 73.5 aggregate | — |

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

- **Single-session decode is well below llama.cpp's on the same class of
  card.** On the RTX 3090, for dense models both engines hold resident,
  llama.cpp decodes Qwen3-8B at 115.3 t/s at 4K (E17) against our 69.8 at ×1
  (Q4_K against our Q6_K), and Llama-2-7B at 158–162 (E1) against our 90.9 on
  the same Q4_0 file. Our one-context prefill is lower too (Qwen3-8B 2,707 vs
  4,050). The width ladder is where this engine gains: its aggregate at
  ×10–×64 is the figure the batch-1 sources do not report.
- **With experts streamed, our single-session MoE decode is the weakest number
  here.** On the RTX 3090 our Q6_K Qwen3.5-35B-A3B streams its experts and
  decodes at 41.8 t/s at ×1, where llama.cpp holds an MXFP4 quant resident at
  111.2 (E17). On the RTX 4090 Mobile ours decodes 9.6 / 8.8 t/s at ×1;
  published partial-offload runs on 12–16 GB cards reach 38.9 (RTX 3060 12 GB,
  `--n-cpu-moe 24`, E43), and a 16 GB card fully resident at a 10.88 GB quant
  121 (E44). At ×16 ours reaches 129.7–131.2 aggregate.
- **Flash-Next at one session trails the published runs too; its width does
  not.** On a 16 GB-class card the nearest published run (RTX 5080 16 GB, E51)
  decodes at 27.5–29 t/s with n-gram speculation, against our 20.0 at ×1 with
  MTP on the RTX 4090 Mobile — and our 64.7 aggregate at ×8, a width no source
  reports. On the RTX 3090, the one card class with a published run and ours,
  llama.cpp decodes 15 t/s single-stream with 128 GB of host RAM (E51, UD-Q4_K_XL,
  q8_0 KV at a 130K context) against our 24.3 at ×1 with MTP and 64 GB (Q2_KO
  experts, a ~700-token prompt) — the context and the expert width both differ,
  so the ×1 gap is not the engine alone — and our 147.0 aggregate at ×16. On the 72 GB card our 86.9 t/s at ×1 with MTP is below the
  RTX PRO 6000's published 100 without MTP and 170 with it (E53, on an
  unstated artifact), and our ×8 aggregate is 393.1.
- **Quantized KV and decode depth.** llama.cpp's q8_0 KV costs Qwen3-8B 18% of
  decode at 8K and 45% at 64K (E64, A100); our C10 costs 3–31% at 32K and
  19–51% at 128K (§3.5) at 4.6–7.6× compression, against q8_0's ~1.9×. Both
  engines pay for quantized reads as the cache deepens; no source measured a
  4–7× KV format with its throughput on these cards.
- **FP8 KV in vLLM** is the published default for compressed KV: 2× capacity for
  +5–15% throughput at concurrency on H100 (E72), and sub-8-bit TurboQuant costs
  20–34% (E73). There is no FP8-KV A/B on the fleet's card classes.

### 6.8 References

All accessed 2026-09-30. Publication dates are the source's own where it gives
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
- **E24** — InsiderLLM, "RTX 5090 Benchmarks: 5090 vs 4090 vs Used 3090"
  (2026-08; secondary compilation).
  https://insiderllm.com/guides/rtx-5090-local-ai-benchmarks/
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

- **A re-sweep of the width gates on a current build.** §3.8 predates the
  decode-slot refresh fix and the synchronised prompt timer (§4), so its prefill
  column understates the card, and it predates every change since. Until it is
  re-run, no prefill comparison against the 3090 is architectural (§3.9).
- **DeepSeek-V4-Flash.** No 3090 row. Size does not exclude it — Flash-Next
  runs here and on the 16 GB card through the expert cache (§3.8).
- **Depth**, as for the 4090 Mobile.
- **The Qwen3-30B engine probe** (`kv_fragmentation::qwen3_30b_a3b_q4`), which has
  no 3090 result; the Flash-Next probe does (§3.8).
- **Llama-3.2-3B with flash-attn.** §3.8 ran its no-flash-attn fallback because
  the sweep shell had no `cl.exe` on PATH; a shell with it measures the real
  path.

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
