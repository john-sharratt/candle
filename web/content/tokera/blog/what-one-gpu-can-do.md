---
title: "AI: What one GPU can do"
date: 2026-09-30
feature: 1
tint: ok
tags: [inference, performance, benchmarks]
summary: >-
  A 180B model on a 16 GB laptop. Context that gets longer without getting
  slower. KV compressed 7× inline, as it's written. And one card that
  out-serves llama.cpp's best published decode by 24×. Measured on three
  machines, set against everything published, with every source cited.
---

**A 180-billion-parameter model, serving eight people at once, from a 16 GB
laptop — out-serving every published run of it, all of them on desktops with two
to four times the memory.**

<figure class="fig">
<svg viewBox="0 0 640 252" role="img" aria-label="Every headline result as a multiple of the best published figure for the same thing, which is 1 times. Aggregate decode against llama.cpp 24.6 times; concurrent sessions on one card 6.4 times; KV-cache compression 4.0 times; a 284B model's decode on one GPU 2.6 times; speed kept at 128K context 1.7 times; a 180B model on a laptop 1.35 times.">
  <text class="ttl" x="16" y="20">Every result, against the best published figure</text>
  <text class="ttl-sub" x="16" y="38">best published = 1× · ours in green · details in each section below</text>
  <path class="grid base" d="M260 50 V226"/>
  <path class="grid" d="M327 50 V226 M394 50 V226 M461 50 V226 M528 50 V226"/>
  <path class="parity" d="M273.4 50 V226"/>
  <text class="cat" x="16" y="64">Aggregate decode vs llama.cpp</text>
  <text class="cat-sub" x="16" y="77">Flash-Next, RTX 3090: 368.8 against 15 t/s</text>
  <path class="line-us" d="M273.4 66 H589.6"/>
  <circle class="dot-them" cx="273.4" cy="66" r="5"/>
  <circle class="dot-us" cx="589.6" cy="66" r="6.5"/>
  <text class="v-us" x="601.6" y="71">24.6×</text>
  <text class="cat" x="16" y="94">Concurrent sessions, one card</text>
  <text class="cat-sub" x="16" y="107">64 against 10 published</text>
  <path class="line-us" d="M273.4 96 H345.8"/>
  <circle class="dot-them" cx="273.4" cy="96" r="5"/>
  <circle class="dot-us" cx="345.8" cy="96" r="6.5"/>
  <text class="v-us" x="357.8" y="101">6.4×</text>
  <text class="cat" x="16" y="124">KV-cache compression</text>
  <text class="cat-sub" x="16" y="137">7.6× against llama.cpp q8_0's 1.9×</text>
  <path class="line-us" d="M273.4 126 H313.6"/>
  <circle class="dot-them" cx="273.4" cy="126" r="5"/>
  <circle class="dot-us" cx="313.6" cy="126" r="6.5"/>
  <text class="v-us" x="325.6" y="131">4.0×</text>
  <text class="cat" x="16" y="154">284B model, one GPU, decode</text>
  <text class="cat-sub" x="16" y="167">73.5 against 28 t/s</text>
  <path class="line-us" d="M273.4 156 H294.8"/>
  <circle class="dot-them" cx="273.4" cy="156" r="5"/>
  <circle class="dot-us" cx="294.8" cy="156" r="6.5"/>
  <text class="v-us" x="306.8" y="161">2.6×</text>
  <text class="cat" x="16" y="184">Speed kept at 128K context</text>
  <text class="cat-sub" x="16" y="197">decode 111% against 65%</text>
  <path class="line-us" d="M273.4 186 H282.8"/>
  <circle class="dot-them" cx="273.4" cy="186" r="5"/>
  <circle class="dot-us" cx="282.8" cy="186" r="6.5"/>
  <text class="v-us" x="294.8" y="191">1.7×</text>
  <text class="cat" x="16" y="214">180B model on a laptop</text>
  <text class="cat-sub" x="16" y="227">64.7 against 48.0 t/s on an RTX 5090</text>
  <path class="line-us" d="M273.4 216 H278.1"/>
  <circle class="dot-them" cx="273.4" cy="216" r="5"/>
  <circle class="dot-us" cx="278.1" cy="216" r="6.5"/>
  <text class="v-us" x="290.1" y="221">1.35×</text>
  <text class="tick mid" x="260" y="244">0×</text>
  <text class="tick mid" x="273.4" y="244">1×</text>
  <text class="tick mid" x="327" y="244">5×</text>
  <text class="tick mid" x="394" y="244">10×</text>
  <text class="tick mid" x="461" y="244">15×</text>
  <text class="tick mid" x="528" y="244">20×</text>
</svg>
</figure>

That's the first of ten results. Every one is measured by a test in this
repository and set beside the best figure anyone has published for the same
model on the same class of card.

1. **A 180B model on a 16 GB laptop.** 64.7 t/s aggregate across eight
   sessions, with 32 GB of RAM, above every published llama.cpp run.
2. **Context length is free.** 16× more context and decode gets *faster*: 99% of
   prefill and 111% of decode kept from 32K to 128K.
3. **Up to 7.6× KV-cache compression, written inline, every output validated** —
   no calibration data, no per-model calibration.
4. **Nearly 4× the compression of llama.cpp's q8_0 cache — while slowing decode
   less.**
5. **24× llama.cpp's best published decode, from one card** — 2.2× to 24.6× on
   every model measured.
6. **Sixty-four conversations on one card:** 1,201.6 t/s aggregate, 11.5× a
   single session, with prefill untouched.
7. **A 284B model serving sixteen people from a single GPU**, at 2.6× the best
   published single-GPU decode.
8. **Workstation work from a laptop:** a small model prefills faster than on a
   24 GB RTX 3090, a 30B MoE prefills at 4,000 t/s with its experts streaming,
   and a 35B MoE serves sixteen users.
9. **One engine, every card:** 193 ladder rows on the RTX 3090 and 187 on the
   laptop, not one failing session.
10. **The whole engine, proven under load:** admission, projection, persistence,
    compaction and all three memory tiers at once — 8/8 correct at 100% VRAM
    efficiency.

And every single session, in every row, produced the right answer.

The machines — three cards anyone can buy:

<div class="machines">
<figure>
<img src="/img/blog/machine-rtx4090-laptop.webp" alt="The RTX 4090 laptop, open on a lap desk, with tokera.com on its screen" width="516" height="640" loading="lazy">
<figcaption><b>RTX 4090 Laptop GPU</b>16 GB VRAM · 32 GB RAM · PCIe 4.0</figcaption>
</figure>
<figure>
<img src="/img/blog/machine-rtx3090.webp" alt="The RTX 3090 desktop in a black and white case, an MSI GeForce card behind the glass" width="591" height="560" loading="lazy">
<figcaption><b>RTX 3090</b>24 GB VRAM · PCIe 3.0 · no native FP8</figcaption>
</figure>
<figure>
<img src="/img/blog/machine-rtx-pro-5000.webp" alt="The RTX PRO 5000 Blackwell workstation in a glass case lit with purple fans" width="596" height="560" loading="lazy">
<figcaption><b>RTX PRO 5000 Blackwell</b>72 GB VRAM · PCIe 5.0</figcaption>
</figure>
</div>

Most of our figures are the *aggregate* across concurrent conversations, and most
published ones are a single conversation; each chart says which. Every number,
ours and theirs, is in the [performance document](https://github.com/john-sharratt/candle/blob/main/docs/performance.md)
with the test or source that produced it.

## A 180B model, on a laptop

Qwen3.8-Flash-Next is 180 billion parameters: a 125B trunk with 512 experts, a
51B n-gram table and a 4B speculative head. On disk, even squeezed to a 2-bit
expert artifact, it's about 88 GB.

The laptop has 16 GB of VRAM and 31.5 GiB of system RAM. Add those together and
you've got roughly half the model.

It runs. Beautifully. Every rung of the gate ladder validates, 8/8 sessions
correct, and at eight concurrent conversations it decodes at **64.7 t/s
aggregate**, with the experts streaming VRAM → RAM → NVMe underneath it the
whole time.

<figure class="fig">
<svg viewBox="0 0 640 446" role="img" aria-label="Qwen3.8-Flash-Next decode rate by machine. This engine on an RTX 3090 with 64 GB RAM: 368.8 tokens per second aggregate at sixteen sessions and 85.9 single. This engine on a 16 GB laptop with 32 GB RAM: 64.7 aggregate at eight sessions and 20.0 single. Published runs: llama.cpp on an RTX 5090 with 128 GB RAM 48.0, RTX 4090 with 96 GB 30, RTX 5080 with 64 GB 29, an RTX 4090 with unstated engine and RAM 21, RTX 3090 with 128 GB 15.">
  <text class="ttl" x="16" y="20">Qwen3.8-Flash-Next (180B) · decode, tokens per second</text>
  <text class="ttl-sub" x="16" y="38">one GPU each · the host RAM behind every run</text>
  <rect class="us" x="452" y="11" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="467" y="20">This engine</text>
  <rect class="them" x="542" y="11" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="557" y="20">Published</text>
  <path class="grid base" d="M250 52 V396"/>
  <path class="grid" d="M335 52 V396 M420 52 V396 M505 52 V396 M590 52 V396"/>
  <text class="cat" x="16" y="72">This engine · 16 sessions</text>
  <text class="cat-sub" x="16" y="86">RTX 3090 24 GB · 64 GB RAM</text>
  <rect class="us" x="250" y="62" width="313.5" height="22" rx="4"/>
  <text class="v-us" x="571.5" y="78">368.8</text>
  <text class="cat" x="16" y="110">This engine · 1 session</text>
  <text class="cat-sub" x="16" y="124">RTX 3090 24 GB · 64 GB RAM</text>
  <rect class="us-soft" x="250" y="100" width="73" height="22" rx="4"/>
  <text class="v-us" x="331" y="116">85.9</text>
  <text class="cat" x="16" y="148">This engine · 8 sessions</text>
  <text class="cat-sub" x="16" y="162">RTX 4090 Laptop 16 GB · 32 GB RAM</text>
  <rect class="us" x="250" y="138" width="55" height="22" rx="4"/>
  <text class="v-us" x="313" y="154">64.7</text>
  <text class="cat" x="16" y="186">llama.cpp · RTX 5090</text>
  <text class="cat-sub" x="16" y="200">32 GB · 128 GB RAM</text>
  <rect class="them" x="250" y="176" width="40.8" height="22" rx="4"/>
  <text class="v-them" x="298.8" y="192">48.0</text>
  <text class="cat" x="16" y="224">llama.cpp · RTX 4090</text>
  <text class="cat-sub" x="16" y="238">24 GB · 96 GB RAM</text>
  <rect class="them" x="250" y="214" width="25.5" height="22" rx="4"/>
  <text class="v-them" x="283.5" y="230">30</text>
  <text class="cat" x="16" y="262">llama.cpp · RTX 5080</text>
  <text class="cat-sub" x="16" y="276">16 GB · 64 GB RAM · n-gram speculation</text>
  <rect class="them" x="250" y="252" width="24.7" height="22" rx="4"/>
  <text class="v-them" x="282.7" y="268">29</text>
  <text class="cat" x="16" y="300">RTX 4090, engine not stated</text>
  <text class="cat-sub" x="16" y="314">24 GB · RAM not stated · 250K context</text>
  <rect class="them" x="250" y="290" width="17.9" height="22" rx="4"/>
  <text class="v-them" x="275.9" y="306">21</text>
  <text class="cat" x="16" y="338">This engine · 1 session</text>
  <text class="cat-sub" x="16" y="352">RTX 4090 Laptop 16 GB · 32 GB RAM</text>
  <rect class="us-soft" x="250" y="328" width="17" height="22" rx="4"/>
  <text class="v-us" x="275" y="344">20.0</text>
  <text class="cat" x="16" y="376">llama.cpp · RTX 3090</text>
  <text class="cat-sub" x="16" y="390">24 GB · 128 GB RAM · 130K context</text>
  <rect class="them" x="250" y="366" width="12.8" height="22" rx="4"/>
  <text class="v-them" x="270.8" y="382">15</text>
  <text class="tick mid" x="250" y="414">0</text>
  <text class="tick mid" x="335" y="414">100</text>
  <text class="tick mid" x="420" y="414">200</text>
  <text class="tick mid" x="505" y="414">300</text>
  <text class="tick mid" x="590" y="414">400</text>
  <text class="t-dim" x="16" y="438">this engine decodes speculatively with the model's own MTP head; the RTX 5080 run uses n-gram speculation</text>
</svg>
<figcaption>Every published single-GPU run of this model that states its host uses
64–128 GB of RAM. The laptop has 32 and serves eight people at once; the RTX 3090
has 64, and one session on it out-decodes every llama.cpp run.</figcaption>
</figure>

Look at the laptop's two green bars, because together they tell the whole story.

The single session — 20 t/s — sits among the desktop runs, which is exactly where
you'd expect a laptop with half the model on an NVMe drive to sit.

Now give it eight conversations. The laptop climbs past an RTX 5090 with 128 GB
of RAM behind it. Every expert that crosses the bus serves every session that
routed to it, so the eighth conversation costs a fraction of the first. That's
the [wave](/blog/waves-and-the-pcie-bottleneck) doing its job, on a model six
times bigger than the one it was designed on.

Then there's the RTX 3090 — a six-year-old card on PCIe 3.0, with no native
FP8, and the one card where there's a published run to put ours directly beside.
llama.cpp on a 3090 with 128 GB of RAM decodes this model at 15 t/s. Ours has
half that RAM, and a single session decodes at **85.9 t/s** — faster than every
published llama.cpp run of this model, the RTX 5090 with 128 GB of RAM behind
it included. Give it sixteen conversations and it reaches **368.8 t/s
aggregate**, nearly twenty-five times the published figure, every session
validated. (Their run was at a 130K context and a 4-bit quant; ours is a short
prompt on the 2-bit expert artifact.)

One engine is faster at a single session: [Strata](https://github.com/Niko1221/Strata),
built for this model alone, keeps the hot experts in VRAM and computes the misses
on the CPU. On a PCIe 5.0 RTX 5070 with 12 GB, holding 14% of the experts against
our 36–49% on a PCIe 3.0 3090, it decodes 94 t/s to our 85.9, both on 2-bit
experts of the same size and both with MTP, so the bus is part of that gap. On
an RTX 3090 of its own, with a PCIe Gen4 link and an EPYC host, it decodes
93 t/s on larger 3-bit experts. Its one published batched run, four sessions on
the 5070, is 63.1 t/s aggregate, below its 70.7 one request at a time. Width is
where this engine pulls ahead.

As far as I can find, this is the first time anyone has published this model
running on a laptop GPU — or in 32 GB of host memory at all.

<div class="key">
<h4>Model size is not bounded by VRAM</h4>
<p>This is the sentence I'd most like people to take away. A mixture-of-experts
model's resident footprint is its dense weights plus whatever expert working set
fits. Everything else lives a tier down and streams up as the router asks for
it.</p>
<p>A bigger card buys speed. It doesn't buy feasibility. So the next time a
parameter count tells you a model "won't run here", try it anyway.</p>
</div>

## Context length is free

Now here's the one I'm proudest of, because it's the theorem from
[the first post](/blog/one-card-unbounded-context) turning up on a stopwatch.

Take Flash-Next on the big card and grow its context from 8K to 128K. Sixteen
times more history behind every token it generates.

What would you expect to happen to the speed?

<figure class="fig">
<svg viewBox="0 0 640 300" role="img" aria-label="Flash-Next throughput as context grows, as a percentage of each run's speed at its shallowest depth, in two panels. Prefill: this engine 100, 125, 122, 118 and 114 percent at 8K, 16K, 32K, 64K and 128K; llama.cpp 100, 67 and 64 percent at 6K, 90K and 110K. Decode: this engine 100, 115, 100, 107 and 107 percent; llama.cpp 100, 73 and 65 percent.">
  <text class="ttl" x="16" y="20">Qwen3.8-Flash-Next · speed as context grows</text>
  <text class="ttl-sub" x="16" y="38">each run as a % of its own speed at the shallowest depth it measured</text>
  <path class="line-us" d="M16 54 H40"/>
  <text class="cat-sub" x="46" y="58">this engine</text>
  <path class="line-them" d="M126 54 H150"/>
  <text class="cat-sub" x="156" y="58">llama.cpp, same model</text>
  <text class="cat" x="62" y="84">Prefill</text>
  <text class="cat" x="362" y="84">Decode</text>
  <path class="grid" d="M62 102 H312 M62 202 H312 M362 102 H612 M362 202 H612"/>
  <path class="grid base" d="M62 152 H312 M362 152 H612"/>
  <path class="grid" d="M62 252 H312 M362 252 H612"/>
  <text class="tick" x="54" y="106" text-anchor="end">125%</text>
  <text class="tick" x="54" y="156" text-anchor="end">100%</text>
  <text class="tick" x="54" y="206" text-anchor="end">75%</text>
  <text class="tick" x="54" y="256" text-anchor="end">50%</text>
  <path class="area-us" d="M77.6 152 L93.2 102.8 L124.5 107.6 L187 116.8 L312 123.2 L312 152 Z"/>
  <path class="them" style="opacity:.16" d="M73.7 152 L237.8 219 L276.8 225 L276.8 152 Z"/>
  <path class="line-them draw" pathLength="100" d="M73.7 152 L237.8 219 L276.8 225"/>
  <path class="line-us draw" pathLength="100" d="M77.6 152 L93.2 102.8 L124.5 107.6 L187 116.8 L312 123.2"/>
  <circle class="dot-them" cx="237.8" cy="219" r="4"/>
  <circle class="dot-them" cx="276.8" cy="225" r="4"/>
  <circle class="dot-us" cx="93.2" cy="102.8" r="4"/>
  <circle class="dot-us" cx="124.5" cy="107.6" r="4"/>
  <circle class="dot-us" cx="187" cy="116.8" r="4"/>
  <circle class="dot-us" cx="312" cy="123.2" r="4.5"/>
  <text class="v-us" x="312" y="114" text-anchor="end">114%</text>
  <text class="v-them" x="282" y="229">64%</text>
  <path class="area-us" d="M377.6 152 L393.2 122.8 L424.5 152.8 L487 139 L612 138.2 L612 152 Z"/>
  <path class="them" style="opacity:.16" d="M373.7 152 L537.8 205.4 L576.8 222 L576.8 152 Z"/>
  <path class="line-them draw" pathLength="100" d="M373.7 152 L537.8 205.4 L576.8 222"/>
  <path class="line-us draw" pathLength="100" d="M377.6 152 L393.2 122.8 L424.5 152.8 L487 139 L612 138.2"/>
  <circle class="dot-them" cx="537.8" cy="205.4" r="4"/>
  <circle class="dot-them" cx="576.8" cy="222" r="4"/>
  <circle class="dot-us" cx="393.2" cy="122.8" r="4"/>
  <circle class="dot-us" cx="424.5" cy="152.8" r="4"/>
  <circle class="dot-us" cx="487" cy="139" r="4"/>
  <circle class="dot-us" cx="612" cy="138.2" r="4.5"/>
  <text class="v-us" x="612" y="129" text-anchor="end">107%</text>
  <text class="v-them" x="583" y="226">65%</text>
  <text class="tick mid" x="77.6" y="270">8K</text>
  <text class="tick mid" x="124.5" y="270">32K</text>
  <text class="tick mid" x="187" y="270">64K</text>
  <text class="tick" x="312" y="270" text-anchor="end">128K</text>
  <text class="tick mid" x="377.6" y="270">8K</text>
  <text class="tick mid" x="424.5" y="270">32K</text>
  <text class="tick mid" x="487" y="270">64K</text>
  <text class="tick" x="612" y="270" text-anchor="end">128K</text>
  <text class="t-dim" x="16" y="292">ours: RTX PRO 5000 72 GB · llama.cpp: RTX 4090 24 GB, UD-IQ3_XXS, q8_0 KV, 6K → 110K [ryan4yin]</text>
</svg>
<figcaption>The published llama.cpp run loses a third of its decode between 6K
and 110K. Ours ends the range faster than it started.</figcaption>
</figure>

**99% of prefill kept from 32K to 128K. 111% of decode.**

For context, every other hybrid model in the fleet keeps 25–30% of its prefill
over the same range, on the same engine:

<figure class="fig">
<svg viewBox="0 0 640 306" role="img" aria-label="Share of 32K throughput kept at 128K, one context, same engine. Qwen3.8-Flash-Next prefill 99 percent, decode 111. Qwen3.8-27B 28 and 58. Qwen3.6-35B 26 and 49. Qwen3.5-9B 30 and 48. Qwen3.5-35B 26 and 47. Qwen3.5-0.8B 25 and 86.">
  <text class="ttl" x="16" y="20">Throughput kept from 32K to 128K, one context</text>
  <text class="ttl-sub" x="16" y="38">same engine, same card · Flash-Next selects what it attends to; the others attend to everything</text>
  <rect class="us" x="16" y="49" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="31" y="58">prefill kept</text>
  <rect class="us-soft" x="112" y="49" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="127" y="58">decode kept</text>
  <rect class="q1" x="208" y="49" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="223" y="58">the other hybrid models</text>
  <path class="grid" d="M80 224.5 H610 M80 187 H610 M80 149.5 H610"/>
  <path class="grid base" d="M80 112 H610 M80 262 H610"/>
  <text class="tick" x="74" y="266" text-anchor="end">0%</text>
  <text class="tick" x="74" y="191" text-anchor="end">50%</text>
  <text class="tick" x="74" y="116" text-anchor="end">100%</text>
  <rect class="us" x="95.2" y="113.5" width="26" height="148.5" rx="3"/>
  <rect class="us-soft" x="127.2" y="95.5" width="26" height="166.5" rx="3"/>
  <text class="v-us mid" x="108.2" y="107.5">99%</text>
  <text class="v-us mid" x="140.2" y="89.5">111%</text>
  <rect class="q1" x="183.5" y="220" width="26" height="42" rx="3"/>
  <rect class="q1" style="opacity:.45" x="215.5" y="175" width="26" height="87" rx="3"/>
  <text class="tick mid" x="196.5" y="214">28%</text>
  <text class="tick mid" x="228.5" y="169">58%</text>
  <rect class="q1" x="271.8" y="223" width="26" height="39" rx="3"/>
  <rect class="q1" style="opacity:.45" x="303.8" y="188.5" width="26" height="73.5" rx="3"/>
  <text class="tick mid" x="284.8" y="217">26%</text>
  <text class="tick mid" x="316.8" y="182.5">49%</text>
  <rect class="q1" x="360.2" y="217" width="26" height="45" rx="3"/>
  <rect class="q1" style="opacity:.45" x="392.2" y="190" width="26" height="72" rx="3"/>
  <text class="tick mid" x="373.2" y="211">30%</text>
  <text class="tick mid" x="405.2" y="184">48%</text>
  <rect class="q1" x="448.5" y="223" width="26" height="39" rx="3"/>
  <rect class="q1" style="opacity:.45" x="480.5" y="191.5" width="26" height="70.5" rx="3"/>
  <text class="tick mid" x="461.5" y="217">26%</text>
  <text class="tick mid" x="493.5" y="185.5">47%</text>
  <rect class="q1" x="536.8" y="224.5" width="26" height="37.5" rx="3"/>
  <rect class="q1" style="opacity:.45" x="568.8" y="133" width="26" height="129" rx="3"/>
  <text class="tick mid" x="549.8" y="218.5">25%</text>
  <text class="tick mid" x="581.8" y="127">86%</text>
  <text class="cat mid" x="124.2" y="282">Flash-Next</text>
  <text class="cat mid" x="212.5" y="282">27B</text>
  <text class="cat mid" x="300.8" y="282">35B</text>
  <text class="cat mid" x="389.2" y="282">9B</text>
  <text class="cat mid" x="477.5" y="282">35B</text>
  <text class="cat mid" x="565.8" y="282">0.8B</text>
  <text class="cat-sub mid" x="124.2" y="297">Qwen3.8</text>
  <text class="cat-sub mid" x="212.5" y="297">Qwen3.8</text>
  <text class="cat-sub mid" x="300.8" y="297">Qwen3.6</text>
  <text class="cat-sub mid" x="389.2" y="297">Qwen3.5</text>
  <text class="cat-sub mid" x="477.5" y="297">Qwen3.5</text>
  <text class="cat-sub mid" x="565.8" y="297">Qwen3.5</text>
</svg>
<figcaption>One model breaks away from the pack, and it is the one whose attention
works over a selected working set rather than the whole history.</figcaption>
</figure>

So this isn't the engine being generous to one model. It's what happens when a
model attends to a selected working set rather than the entire history — the cost
per token simply stops depending on how much history there is.

That's the O(1) claim. Here it is with a clock on it.

## 7.6× compression, written inline

Every 32-token block of KV is compressed as it's written, in the same forward
that produced it. No separate compression pass. No background job. No "quantize
the cache at turn end".

The block is born compressed. That's it. That's the whole trick.

The top rung of the [adaptive ladder](/blog/palquant-per-block) picks a format
per block and compresses the cache **4.1× to 7.6×** across the fleet, quantizes
100% of blocks, and still reproduces every session's output — with no
calibration data and no per-model calibration. And prefill with it switched on
lands within a few percent of uncompressed — at 128K it's marginally *faster* on
every model, because there are fewer bytes to read back.

So the interesting question is what it costs in decode — and how that compares
with everything else out there.

<figure class="fig">
<svg viewBox="0 0 640 324" role="img" aria-label="KV-cache compression against decode speed kept. This engine's C10 rung at 8K sits at 4.6 to 6.3 times compression keeping 92 to 98 percent of decode; at 32K 6.3 to 7.5 times keeping 70 to 73 percent; Flash-Next at 128K 7.0 times keeping 81 percent. llama.cpp q8_0 is 1.9 times keeping 82, 65 and 55 percent at 8K, 32K and 64K; q4_0 3.56 times keeping 80, 61 and 51 percent. vLLM FP8 2 times; TurboQuant 2.4 and 3.4 times keeping 80 and 73 percent of throughput.">
  <text class="ttl" x="16" y="20">KV-cache compression against the decode it costs</text>
  <text class="ttl-sub" x="16" y="38">up and to the right is better</text>
  <circle class="dot-us" cx="21" cy="54" r="5"/>
  <text class="cat-sub" x="31" y="58">this engine, top level (C10)</text>
  <rect class="them" x="200" y="49" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="215" y="58">shipping formats</text>
  <path class="grid" d="M70 70 H600 M70 118 H600 M70 166 H600 M70 214 H600 M145.7 70 V262 M297.1 70 V262 M448.6 70 V262 M600 70 V262"/>
  <path class="grid base" d="M70 262 H600 M70 70 V262"/>
  <text class="tick" x="62" y="74" text-anchor="end">120%</text>
  <text class="tick" x="62" y="122" text-anchor="end">100%</text>
  <text class="tick" x="62" y="170" text-anchor="end">80%</text>
  <text class="tick" x="62" y="218" text-anchor="end">60%</text>
  <text class="tick" x="62" y="266" text-anchor="end">40%</text>
  <text class="tick mid" x="70" y="280">1×</text>
  <text class="tick mid" x="145.7" y="280">2×</text>
  <text class="tick mid" x="297.1" y="280">4×</text>
  <text class="tick mid" x="448.6" y="280">6×</text>
  <text class="tick mid" x="600" y="280">8×</text>
  <text class="tick" x="600" y="296" text-anchor="end">compression →</text>
  <rect class="area-us" x="330" y="110" width="250" height="94" rx="14"/>
  <rect class="them" x="133.1" y="156.2" width="10" height="10" rx="2"/>
  <rect class="them" x="133.1" y="197.5" width="10" height="10" rx="2"/>
  <rect class="them" x="133.1" y="221.5" width="10" height="10" rx="2"/>
  <rect class="them" x="258.8" y="161.5" width="10" height="10" rx="2"/>
  <rect class="them" x="258.8" y="207.6" width="10" height="10" rx="2"/>
  <rect class="them" x="258.8" y="231.8" width="10" height="10" rx="2"/>
  <rect class="them" x="140.7" y="77.2" width="10" height="10" rx="2"/>
  <rect class="them" x="171" y="161" width="10" height="10" rx="2"/>
  <rect class="them" x="246.7" y="177.8" width="10" height="10" rx="2"/>
  <circle class="dot-us" cx="474.8" cy="123.5" r="6"/>
  <circle class="dot-us" cx="446.7" cy="136.7" r="6"/>
  <circle class="dot-us" cx="346.8" cy="131" r="6"/>
  <circle class="dot-us" cx="361.2" cy="133.4" r="6"/>
  <circle class="dot-us" cx="347.5" cy="124.2" r="6"/>
  <circle class="dot-us" cx="522" cy="164.3" r="6"/>
  <circle class="dot-us" cx="525" cy="183" r="6"/>
  <circle class="dot-us" cx="560.6" cy="191" r="6"/>
  <circle class="dot-us" cx="473.3" cy="191.2" r="6"/>
  <text class="v-us" x="336" y="103">at 8K: 4.6–6.3× for 0–8% of decode</text>
  <text class="cat-sub" x="512" y="160" text-anchor="end">Flash-Next · 128K</text>
  <text class="cat-sub" x="572" y="219" text-anchor="end">35B · 3.6-35B · 9B · 32K</text>
  <text class="t-them" x="156" y="86">vLLM FP8 · H100 throughput</text>
  <text class="t-them" x="184" y="156">TurboQuant</text>
  <text class="t-them mid" x="138" y="250">q8_0</text>
  <text class="t-them mid" x="264" y="256">q4_0</text>
  <text class="t-dim" x="16" y="316">llama.cpp q8_0 / q4_0 at 8K, 32K and 64K: Qwen3-8B on an A100 [SOTAAZ] · vLLM FP8 and TurboQuant: H100 [vLLM blog]</text>
</svg>
<figcaption>The shipping formats cluster on the left, below 3.6×. Ours start at
4.6×.</figcaption>
</figure>

The top-right cluster is 8K context: **3.3× to 6.3× compression for 0–8% of
decode**, on every model measured there. At 32K the bigger models pay 27–31% for
6.3–7.5×, and Flash-Next at 128K pays 19% for 7×.

Now the squares, which are some very good engineering by some very good people.
llama.cpp's q8_0 KV cache — the one everybody reaches for — gives **1.9×**, for
35% of decode at 32K and 45% at 64K. Its q4_0 reaches 3.56×. vLLM's FP8 cache
gives 2×, and TurboQuant, the newest thing in the literature, gets to 3.4× for
about a quarter of throughput.

So at 32K — our most expensive depth — we're at **nearly 4× the compression of
llama.cpp's q8_0**, and decode slows down *less* than theirs does. At 8K it's not
even close.

That's what per-block format selection buys you. The format is chosen for each
32 tokens on its own merits, so the cache compresses hard wherever it can, and
gently only where it has to.

<div class="key">
<h4>Why "validated" is doing real work in that sentence</h4>
<p>Anyone can compress a KV cache 7×. The hard bit is getting the same answer
afterwards. Every compressed row here comes from a gate that replays the same
prompts and checks each session's output against its uncompressed run — not a
perplexity number, not a cosine threshold, the actual text.</p>
<p>The top rung is deliberately set to sit just under the edge where that check
starts failing. That's what makes it the top rung.</p>
</div>

## One card out-decodes llama.cpp by 24×

This is the one I'd most like people to go and check, so here's exactly what it
claims.

For each model, take the **best single-stream decode figure anyone has published
for llama.cpp** on that model, on that class of card — RTX 3090 against RTX 3090,
16 GB against 16 GB, our Blackwell workstation card against a published RTX 5090.
Then take this engine's **aggregate** decode serving concurrent conversations on
the same class of card. Divide.

<figure class="fig">
<svg viewBox="0 0 640 446" role="img" aria-label="Our aggregate decode divided by llama.cpp's best published single-stream decode, same model and card class. Flash-Next on RTX 3090 24.6 times; Qwen3.5-35B on RTX 3090 7.74; Qwen3.5-35B Blackwell 6.12; Qwen3.8-27B RTX 3090 5.98; Llama-2-7B RTX 3090 5.84; Qwen3.6-35B RTX 3090 4.95; Qwen3.6-35B Blackwell 3.60; Qwen3-8B RTX 3090 3.31; Llama-2-7B Blackwell 3.05; Qwen3-30B Blackwell 2.63; Qwen3-30B RTX 3090 2.53; Qwen3-8B Blackwell 2.30; Qwen3-8B 16 GB 2.24; Flash-Next 16 GB 2.23.">
  <text class="ttl" x="16" y="20">Our aggregate decode ÷ llama.cpp's best published decode</text>
  <text class="ttl-sub" x="16" y="38">same model, same class of card · 1× is parity</text>
  <path class="grid base" d="M250 56 V422"/>
  <path class="grid" d="M316 56 V422 M382 56 V422 M448 56 V422 M514 56 V422"/>
  <path class="parity" d="M263.2 56 V422"/>
  <text class="t-them" x="267" y="52">llama.cpp's best = 1×</text>
  <text class="cat" x="16" y="75">Flash-Next</text>
  <text class="cat-sub" x="240" y="75" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="62" width="324.7" height="17" rx="4"/>
  <text class="v-us" x="582.7" y="76">24.6×</text>
  <text class="cat" x="16" y="101">Qwen3.5-35B</text>
  <text class="cat-sub" x="240" y="101" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="88" width="102.2" height="17" rx="4"/>
  <text class="v-us" x="360.2" y="102">7.74×</text>
  <text class="cat" x="16" y="127">Qwen3.5-35B</text>
  <text class="cat-sub" x="240" y="127" text-anchor="end">Blackwell</text>
  <rect class="us" x="250" y="114" width="80.8" height="17" rx="4"/>
  <text class="v-us" x="338.8" y="128">6.12×</text>
  <text class="cat" x="16" y="153">Qwen3.8-27B</text>
  <text class="cat-sub" x="240" y="153" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="140" width="78.9" height="17" rx="4"/>
  <text class="v-us" x="336.9" y="154">5.98×</text>
  <text class="cat" x="16" y="179">Llama-2-7B</text>
  <text class="cat-sub" x="240" y="179" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="166" width="77.1" height="17" rx="4"/>
  <text class="v-us" x="335.1" y="180">5.84×</text>
  <text class="cat" x="16" y="205">Qwen3.6-35B</text>
  <text class="cat-sub" x="240" y="205" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="192" width="65.3" height="17" rx="4"/>
  <text class="v-us" x="323.3" y="206">4.95×</text>
  <text class="cat" x="16" y="231">Qwen3.6-35B</text>
  <text class="cat-sub" x="240" y="231" text-anchor="end">Blackwell</text>
  <rect class="us" x="250" y="218" width="47.5" height="17" rx="4"/>
  <text class="v-us" x="305.5" y="232">3.60×</text>
  <text class="cat" x="16" y="257">Qwen3-8B</text>
  <text class="cat-sub" x="240" y="257" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="244" width="43.7" height="17" rx="4"/>
  <text class="v-us" x="301.7" y="258">3.31×</text>
  <text class="cat" x="16" y="283">Llama-2-7B</text>
  <text class="cat-sub" x="240" y="283" text-anchor="end">Blackwell</text>
  <rect class="us" x="250" y="270" width="40.3" height="17" rx="4"/>
  <text class="v-us" x="298.3" y="284">3.05×</text>
  <text class="cat" x="16" y="309">Qwen3-30B</text>
  <text class="cat-sub" x="240" y="309" text-anchor="end">Blackwell</text>
  <rect class="us" x="250" y="296" width="34.7" height="17" rx="4"/>
  <text class="v-us" x="292.7" y="310">2.63×</text>
  <text class="cat" x="16" y="335">Qwen3-30B</text>
  <text class="cat-sub" x="240" y="335" text-anchor="end">RTX 3090</text>
  <rect class="us" x="250" y="322" width="33.4" height="17" rx="4"/>
  <text class="v-us" x="291.4" y="336">2.53×</text>
  <text class="cat" x="16" y="361">Qwen3-8B</text>
  <text class="cat-sub" x="240" y="361" text-anchor="end">Blackwell</text>
  <rect class="us" x="250" y="348" width="30.4" height="17" rx="4"/>
  <text class="v-us" x="288.4" y="362">2.30×</text>
  <text class="cat" x="16" y="387">Qwen3-8B</text>
  <text class="cat-sub" x="240" y="387" text-anchor="end">16 GB</text>
  <rect class="us" x="250" y="374" width="29.6" height="17" rx="4"/>
  <text class="v-us" x="287.6" y="388">2.24×</text>
  <text class="cat" x="16" y="413">Flash-Next</text>
  <text class="cat-sub" x="240" y="413" text-anchor="end">16 GB</text>
  <rect class="us" x="250" y="400" width="29.4" height="17" rx="4"/>
  <text class="v-us" x="287.4" y="414">2.23×</text>
  <text class="tick mid" x="250" y="438">0×</text>
  <text class="tick mid" x="263.2" y="438">1×</text>
  <text class="tick mid" x="316" y="438">5×</text>
  <text class="tick mid" x="382" y="438">10×</text>
  <text class="tick mid" x="448" y="438">15×</text>
  <text class="tick mid" x="514" y="438">20×</text>
</svg>
<figcaption>Every row clears parity. The widest is Flash-Next on the RTX 3090:
368.8 t/s across sixteen sessions against a published 15 — their run at a 130K
context on a 4-bit quant, ours at a short prompt on 2-bit experts, so it is the
least like-for-like row. The widest like-for-like one is Qwen3.5-35B on the same
card: 860.3 against 111.2.</figcaption>
</figure>

**Between 2.2× and 24.6×, on every row.**

My favourite is the oldest card in the fleet. A 3090, behind a PCIe 3.0 bus, with
no native FP8, serves Qwen3.5-35B at **860.3 t/s aggregate** — against a
published 111.2. Qwen3.8-27B on the same card does 390.1 against a published
65.3, and that published figure already has speculative decoding switched on.
And the 180B Flash-Next, streaming its experts from 64 GB of RAM, serves sixteen
conversations at **368.8 t/s** against a published 15.

A six-year-old card, doing the work of nearly eight — and on the biggest model,
of twenty-four.

Aggregate against single-stream is exactly the comparison that matters when
you're deciding how many cards to buy. llama.cpp can serve parallel requests too,
and I'd love to see someone publish its aggregate on these models — that would
make a great follow-up chart. Until then, this is the best comparison the
published record allows, labelled as what it is.

## Sixty-four conversations on one card

So how far does it go?

Here's Qwen3.6-35B on one 72 GB card, doubling the session count until the gate
runs out of ladder:

<figure class="fig">
<svg viewBox="0 0 640 314" role="img" aria-label="Qwen3.6-35B aggregate decode by concurrent sessions on one RTX PRO 5000: 104.9 tokens per second at 1, 376.1 at 4, 573.2 at 8, 687.5 at 16, 974.2 at 32, 1,201.6 at 64. The published vLLM run on an RTX PRO 6000: 196.4 at 1 and 449.0 at 5.">
  <text class="ttl" x="16" y="20">Qwen3.6-35B-A3B on one RTX PRO 5000 · aggregate decode, t/s</text>
  <text class="ttl-sub" x="16" y="38">one run · every session validated · ×1–×4 BF16 KV, ×8–×64 compressed 6.04×</text>
  <path class="line-us" d="M16 54 H40"/>
  <text class="cat-sub" x="46" y="58">this engine</text>
  <path class="line-them" d="M126 54 H150"/>
  <text class="cat-sub" x="156" y="58">vLLM on a larger RTX PRO 6000, published</text>
  <path class="grid" d="M70 203.5 H600 M70 145.1 H600 M70 86.6 H600"/>
  <path class="grid base" d="M70 262 H600"/>
  <text class="tick" x="62" y="266" text-anchor="end">0</text>
  <text class="tick" x="62" y="207.5" text-anchor="end">400</text>
  <text class="tick" x="62" y="149.1" text-anchor="end">800</text>
  <text class="tick" x="62" y="90.6" text-anchor="end">1,200</text>
  <path class="area-us" d="M80 246.7 L250 207 L335 178.2 L420 161.5 L505 119.6 L590 86.4 L590 262 L80 262 Z"/>
  <path class="line-them" d="M80 233.3 L277.4 196.4"/>
  <path class="line-us draw" pathLength="100" d="M80 246.7 L250 207 L335 178.2 L420 161.5 L505 119.6 L590 86.4"/>
  <circle class="dot-them" cx="80" cy="233.3" r="4.5"/>
  <circle class="dot-them" cx="277.4" cy="196.4" r="4.5"/>
  <circle class="dot-us" cx="80" cy="246.7" r="4.5"/>
  <circle class="dot-us" cx="250" cy="207" r="4.5"/>
  <circle class="dot-us" cx="335" cy="178.2" r="4.5"/>
  <circle class="dot-us" cx="420" cy="161.5" r="4.5"/>
  <circle class="dot-us" cx="505" cy="119.6" r="4.5"/>
  <circle class="dot-us" cx="590" cy="86.4" r="5.5"/>
  <text class="v-us" x="90" y="258">105</text>
  <text class="v-them" x="90" y="228">196</text>
  <text class="v-us" x="244" y="199" text-anchor="end">376</text>
  <text class="v-them" x="283" y="214">449</text>
  <text class="v-us" x="329" y="170" text-anchor="end">573</text>
  <text class="v-us" x="414" y="153" text-anchor="end">688</text>
  <text class="v-us" x="499" y="111" text-anchor="end">974</text>
  <text class="v-us" x="584" y="78" text-anchor="end">1,202</text>
  <text class="hero-n" x="110" y="120">11.5×</text>
  <text class="cat-sub" x="110" y="138">1 → 64 sessions, one card</text>
  <text class="tick mid" x="80" y="280">×1</text>
  <text class="tick mid" x="250" y="280">×4</text>
  <text class="tick mid" x="335" y="280">×8</text>
  <text class="tick mid" x="420" y="280">×16</text>
  <text class="tick mid" x="505" y="280">×32</text>
  <text class="tick mid" x="590" y="280">×64</text>
  <text class="t-dim" x="16" y="306">prefill over the same range: 7,302 → 7,219 t/s (−1.1%) · vLLM: Millstone AI, FP8</text>
</svg>
<figcaption>11.5× the decode of a single session, and prefill doesn't notice.</figcaption>
</figure>

**104.9 t/s for one conversation. 1,201.6 t/s for sixty-four.** Eleven and a
half times the work from the same card, with the KV cache compressed 6× so they
all fit, and every one of the sixty-four sessions checked for the right answer.
Prefill, over the same range, moves by 1.1%.

Qwen3.5-0.8B goes further still and serves **256 concurrent sessions**. The
published single-card serving runs of these models stop at five to ten
concurrent requests.

And that flat prefill line deserves a second look. One wave engine carries prefill
rows and decode rows in the *same forward*, so a new conversation arriving
doesn't make the other sixty-three wait.

Everybody keeps talking. Nobody takes turns.

## 284 billion parameters, one GPU

DeepSeek-V4-Flash is 284B parameters with 13B active. It's the kind of model
that normally comes with a rack attached.

Here it is on a single 72 GB workstation card, beside every published single-GPU
run I could find:

<figure class="fig">
<svg viewBox="0 0 640 384" role="img" aria-label="DeepSeek-V4-Flash decode on a single GPU. This engine at sixteen sessions on an RTX PRO 5000: 73.5 tokens per second aggregate. Published single-stream: KTransformers with SGLang on RTX 5090 28; SGLang with KT-Kernel on RTX 5090 20; KTransformers on RTX 4090 18.5; llama.cpp on RTX 5090 18; llama.cpp on RTX 3090 12.5; GGUF on RTX 4090 12; llama.cpp on RTX PRO 6000 10.7.">
  <text class="ttl" x="16" y="20">DeepSeek-V4-Flash (284B) on one GPU · decode, t/s</text>
  <text class="ttl-sub" x="16" y="38">ours: 16 sessions, aggregate · published: single stream · experts in host RAM throughout</text>
  <rect class="us" x="452" y="11" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="467" y="20">This engine</text>
  <rect class="them" x="542" y="11" width="10" height="10" rx="2"/>
  <text class="cat-sub" x="557" y="20">Published</text>
  <path class="grid base" d="M250 52 V340"/>
  <path class="grid" d="M337.5 52 V340 M425 52 V340 M512.5 52 V340 M600 52 V340"/>
  <text class="cat" x="16" y="72">This engine · 16 sessions</text>
  <text class="cat-sub" x="16" y="86">RTX PRO 5000 72 GB</text>
  <rect class="us" x="250" y="62" width="321.6" height="22" rx="4"/>
  <text class="v-us" x="579.6" y="78">73.5</text>
  <text class="cat" x="16" y="108">KTransformers + SGLang</text>
  <text class="cat-sub" x="16" y="122">RTX 5090 · INT4 experts on CPU</text>
  <rect class="them" x="250" y="98" width="122.5" height="22" rx="4"/>
  <text class="v-them" x="380.5" y="113">28</text>
  <text class="cat" x="16" y="144">SGLang + KT-Kernel</text>
  <text class="cat-sub" x="16" y="158">RTX 5090</text>
  <rect class="them" x="250" y="134" width="87.5" height="22" rx="4"/>
  <text class="v-them" x="345.5" y="149">20+</text>
  <text class="cat" x="16" y="180">KTransformers</text>
  <text class="cat-sub" x="16" y="194">RTX 4090 · MXFP4</text>
  <rect class="them" x="250" y="170" width="80.9" height="22" rx="4"/>
  <text class="v-them" x="338.9" y="185">18.5</text>
  <text class="cat" x="16" y="216">llama.cpp</text>
  <text class="cat-sub" x="16" y="230">RTX 5090 · experts on CPU</text>
  <rect class="them" x="250" y="206" width="78.75" height="22" rx="4"/>
  <text class="v-them" x="336.75" y="221">18</text>
  <text class="cat" x="16" y="252">llama.cpp</text>
  <text class="cat-sub" x="16" y="266">RTX 3090</text>
  <rect class="them" x="250" y="242" width="54.7" height="22" rx="4"/>
  <text class="v-them" x="312.7" y="257">12.5</text>
  <text class="cat" x="16" y="288">GGUF, engine not stated</text>
  <text class="cat-sub" x="16" y="302">RTX 4090</text>
  <rect class="them" x="250" y="278" width="52.5" height="22" rx="4"/>
  <text class="v-them" x="310.5" y="293">12</text>
  <text class="cat" x="16" y="324">llama.cpp</text>
  <text class="cat-sub" x="16" y="338">RTX PRO 6000 Max-Q · 8K</text>
  <rect class="them" x="250" y="314" width="47" height="22" rx="4"/>
  <text class="v-them" x="305" y="329">10.7</text>
  <text class="tick mid" x="250" y="356">0</text>
  <text class="tick mid" x="337.5" y="356">20</text>
  <text class="tick mid" x="425" y="356">40</text>
  <text class="tick mid" x="512.5" y="356">60</text>
  <text class="tick mid" x="600" y="356">80</text>
  <text class="t-dim" x="16" y="378">prefill: 1,120.6 t/s at sixteen sessions, against a best published single-GPU figure of 748.4</text>
</svg>
<figcaption>2.6× the best published single-GPU decode, while serving sixteen
conversations.</figcaption>
</figure>

**Prefill at 1,120.6 t/s**, above every published single-GPU figure for the
model (the best is 748). **Decode at 73.5 t/s aggregate** — 2.6× the best
published single-GPU decode of 28 t/s.

Sixteen people, on a model of that class, from one card in an ordinary
workstation. No NVLink. No second node. No InfiniBand.

Just a card in a slot, doing what it was always capable of.

## Workstation work on a laptop

You'll have noticed the laptop keeps turning up. That's because it's the machine
this whole design grew up on — and it's where the results are most delightful.

| on the 16 GB laptop | result | against the 24 GB RTX 3090 |
|---|---:|---:|
| Qwen3.5-0.8B prefill, ×32 | **21,847 t/s** | 21,382 |
| Qwen3-30B-A3B prefill, ×20, experts streamed | **3,999.7 t/s** | 4,955.9 |
| Qwen3.5-0.8B aggregate decode, ×32 | 993 t/s | 3,999 |
| Qwen3.5-35B-A3B aggregate decode, ×16 | 131.2 t/s at **6.2×** compression | 860.3 |

The small model prefills *faster* on the laptop than on the 3090, because the
laptop has twice the bus. The 30B, streaming its experts over that bus, prefills
within 20% of a card that holds far more of the model in VRAM. And a 35B MoE
with about 28 GB of Q6_K weights serves **sixteen users** on a card with 16 GB
of memory, every one of them validated.

The 3090 takes the decode, as it should with 50% more VRAM to hold experts in.
What's remarkable is that the laptop is in the same conversation at all —
serving a model nearly twice its memory, to sixteen people, from a bag.

## One engine, every card

The last one isn't a speed number at all, and I think it matters more than any of
them.

The same thirteen gates pass on an Ada laptop, on a Blackwell workstation, and on
a 3090 behind a PCIe 3.0 bus with no native FP8 — where the fast provenance-scan
backends don't even apply and fall back a rung. **193 ladder rows on the 3090 and
187 on the laptop, and not one failing session.** The engine sizes its own memory
partition to each card; there's no per-machine tuning file.

On the 3090's current build, C10 compresses the Qwen3.5-35B **7.03×**, and every
session still reproduces its uncompressed answer.

And beyond the gates, the whole engine runs under load — admission, per-turn
context projection, the persistence thread, KV compaction, all three memory tiers
at once. On Flash-Next, the one model here that carries recurrent state outside
the paged KV, that probe comes back **8/8 correct at 100% VRAM efficiency** on
the laptop. On the 3090 all three probes pass — the 30B, Flash-Next, and the
Qwen3.6-35B under speculative decode, its recurrent state rewound on every
rejected draft — every story correct, at 98–99% VRAM efficiency.

And it's still getting faster. In the week since Flash-Next first ran on the
3090, its single-session decode there has **more than tripled, from 24.3 to
85.9 t/s**, eight sessions have gone from 113.2 to **311.9 t/s**, and sixteen
now reach **368.8** — with zero loss of validation. The latest step came from
recording every forward as a chain of CUDA graphs, so the GPU stops waiting on
the host between kernels, and from moving experts by where each wave actually
is. Nearly every model on the card got faster with it.

The ceiling is still moving, and it's moving up.

## Where the others shine

None of this happens in a vacuum. The engines in those amber bars are the work of
some of the best people in the field, and each is superb at the question it was
built to answer. So here's where to reach for them — and where this engine is
heading next.

**One conversation, on a model that fits.** That's llama.cpp's home ground, and
it's wonderful at it. On a 16 GB card, a 3-bit Qwen3.6-35B that fits wholly in
VRAM decodes one stream at 183–249 t/s, where our sixteen-session aggregate with
Q6_K experts streaming is 130. For one person and one model that fits, it's a
fantastic choice.

**Raw prefill on a big card.** vLLM on an RTX PRO 6000 prefills Qwen3.6-35B at
41,105 t/s; we're at about 7,200. vLLM's prefill kernels are a genuine
inspiration, and that's the next ceiling on my list.

**Compression at depth on the big hybrids.** At 32K the 35B models spend 27–31% of
their decode to hold the cache at 7× — already a better trade than q8_0's 35% for
1.9×, and the next rung of work is making it cheaper still.

Three clear directions. I'm looking forward to all of them.

## What it adds up to

Put the pieces side by side and the picture is bigger than anything I set out to
build.

A model twice the size of the machine's memory, serving eight people from a
laptop. Context that gets longer without getting slower. A KV cache seven times
smaller, compressed as it's written, with every answer checked. One card doing
the decode work of up to twenty-four llama.cpp instances. A 284B model in a single
workstation.

None of it came from a bigger GPU. All of it came from the constraint of not
having one — and, as [the first post](/blog/one-card-unbounded-context) argued,
a constraint that closes the easy road tends to open a better one.

So here's the thing I'd pass on. The hardware you already own can do far more
than its spec sheet suggests. It's waiting for software that asks it the right
question.

The code is in the public domain. The numbers are in the repository with the
tests that produced them. Go and try them — and if you beat them, I'd love to
hear about it.

One last word, about [Strata](https://github.com/Niko1221/Strata). It was built
for Flash-Next alone, and it is the fastest single-session decoder of that model
I've found. A community run on an RTX 5090 — close hardware to our RTX PRO 5000 —
makes the cleanest comparison:

- **One session:** Strata decodes 179.4 t/s at 4K context (175.7 at 32K), on
  2-bit experts. We decode 147.5 t/s from a short prompt, on 4-bit experts that
  are twice the bytes per expert. On one conversation, Strata is faster.
- **Many sessions:** our eight-session aggregate is 738.2 t/s, 5.0× our single
  session, and sixteen reach 792.8. Strata's one published batched run, four
  sessions on an RTX 5070, comes to 63.1 t/s, 0.89× its 70.7 one request at a
  time. That ratio doesn't depend on the card, and it's where the two designs
  part ways.
- **The KV cache:** ours is compressed 7.13× as it's written, while eight
  sessions still decode 681.5 t/s, 92% of uncompressed. Strata's int8 KV is
  about 2×.

Strata has never run on our 3090 machine, so the 3090 comparison earlier is only
indicative. Its benchmarks, hardware and method are all published in the open
for anyone to reproduce, and work like that moves the whole field. It's well
worth a look.

<div class="key">
<h4>Sources</h4>
<p>Our figures, and every external figure with its card, quantization and
context: the <a href="https://github.com/john-sharratt/candle/blob/main/docs/performance.md">performance
document</a>, §3 and §6.</p>
<p>Key outside sources:
<a href="https://gist.github.com/ryan4yin/48617bbddacc7067f10799770b7cc33f">ryan4yin, Flash-Next on an RTX 4090</a> ·
<a href="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3">unsloth Flash-Next speed thread</a> ·
<a href="https://zenn.dev/holy_fox/articles/04887ff8177b87?locale=en">holy_fox, Flash-Next on an RTX 5090</a> ·
<a href="https://sotaaz.com/post/llamacpp-kv-cache-quantization-bench-en">SOTAAZ, llama.cpp KV quantization on an A100</a> ·
<a href="https://vllm.ai/blog/2026-04-22-fp8-kvcache">vLLM, FP8 KV cache</a> ·
<a href="https://vllm.ai/blog/2026-05-11-turboquant">vLLM, TurboQuant</a> ·
<a href="https://github.com/ggml-org/llama.cpp/discussions/15013">llama.cpp CUDA performance discussion</a> ·
<a href="https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-3090/">Hardware Corner, RTX 3090</a> ·
<a href="https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5090/">Hardware Corner, RTX 5090</a> ·
<a href="https://jonidimo.github.io/qwen38-3090-benchmark/benchmark.html">jonidimo, Qwen3.8-27B on an RTX 3090</a> ·
<a href="https://byteshape.com/blogs/Qwen3.6-35B-A3B/">ByteShape, Qwen3.6-35B</a> ·
<a href="https://github.com/ggml-org/llama.cpp/discussions/19890">llama.cpp, Qwen3.5-35B on an RTX 5090</a> ·
<a href="https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-fp8-1x-rtx-pro-6000-blackwell">Millstone AI, Qwen3.6-35B on an RTX PRO 6000</a> ·
<a href="https://gist.github.com/RockmSockmJesus/30a195ccd9b62e981ec2676a99a57b7e">DeepSeek-V4-Flash at 28 t/s on an RTX 5090</a> ·
<a href="https://github.com/ggml-org/llama.cpp/pull/24162">llama.cpp DeepSeek V4 PR</a> ·
<a href="https://github.com/Niko1221/Strata/tree/main/bench/results">Strata, Flash-Next benchmark results</a></p>
</div>
