---
title: "AI: One Card, One Stack — the table I owed you"
date: 2026-11-25
feature: 1
tint: accent
tags: [inference, architecture, research]
summary: >-
  v2 of One Card, One Stack is published, and it is the final version. The
  claims v1 made are now measured: a model attending over a 42M-token memory of
  its own codebase, the memory worth 37.0 points of recall with tools
  forbidden, re-selecting while it reasons worth 6.9 over retrieving once, and
  a theorem that says exactly what is bounded — all on one card.
---

In August I wrote about [One Card, One Stack](/blog/one-card-unbounded-context),
and near the bottom of that post sat a box with one phrase in bold: **the full
quantitative table is held for v2**. That was an IOU, and today it's paid.

Version 2 is published on Zenodo —
[10.5281/zenodo.24529143](https://doi.org/10.5281/zenodo.24529143) — and it is
the final version. Not a preview, and not "v2, with a v3 to follow when the
numbers come in". The numbers came in: measured on the production system,
across three machines and twelve models, every figure naming the build, the
machine and the test that produced it.

August made three claims. That context is an architecture problem, not a budget
problem. That a model can attend over a memory far larger than anything it could
hold, and get *better* for re-selecting what it attends to as it reasons. And
that the error it's exposed to stays bounded at any depth. v2 proves all three.

## A 42-million-token memory, on one card

Here's the scale of it. The system's memory of its own Candle fork — 2.2 million
lines of Rust and CUDA, read by the model that serves it, file by file, as turns
of its own — is a **42M-token substrate**. At full precision that's 976.7 GiB
of K/V; held compressed, 275.3 GiB, **3.55×** smaller. The code gallery alone
is 36.1 million tokens.

No card holds that. None needs to. Every step attends to a bounded working set,
selected from the whole memory by the model's own attention — and with **sixteen
answers in flight at once**, the card's resident context peaks at 945 thousand
tokens. The memory grows without bound; what each step reads does not.

That's the architecture working at the scale it was built for, and the next
question is the one that matters: does it actually *remember*?

## The exam

**128 questions** about the codebase — 43 direct, 43 transitive, 42
architectural — each listing what a correct answer has to name, committed with
the analysis before the first run. The direct ones are drawn by a seeded hash,
so I don't get to pick favourites, and the battery is kept out of the memory, so
the model has never read its own exam paper. One rule: **no tools**. No grep, no
opening files. It answers from what it remembers.

Each part of the memory is switched on and off, one arm at a time, with three to
seven runs per arm.

<figure class="fig">
<svg viewBox="0 0 640 320" role="img" aria-label="Mean recall by arm over 128 questions. A0, the model alone, 0.4 percent. A1, the seed files alone, 0.5 percent. D1, retrieve once before reasoning, 30.6 percent. B0, projection held in thinking, no seed, 31.9 percent. B1, held in thinking, seed, 32.2 percent. C0, full reprojection, no seed, 36.7 percent. C1, the production system, 37.5 percent.">
  <text class="ttl" x="16" y="20">Recall by arm · 128 questions · tools forbidden</text>
  <text class="ttl-sub" x="16" y="38">solid: re-selecting as it reasons · faint: retrieved once, or held still</text>
  <path class="grid" d="M320 50 V292 M390 50 V292 M460 50 V292 M530 50 V292 M600 50 V292"/>
  <path class="grid base" d="M250 50 V292"/>
  <text class="cat" x="16" y="67">A0 · the model alone</text>
  <text class="cat-sub" x="16" y="80">no memory, no seed</text>
  <rect class="q1" x="250" y="59" width="2.8" height="18"/>
  <text class="v-them" x="260.8" y="73">0.4%</text>
  <text class="cat" x="16" y="101">A1 · the seed files alone</text>
  <text class="cat-sub" x="16" y="114">no memory projected</text>
  <rect class="q1" x="250" y="93" width="3.5" height="18"/>
  <text class="v-them" x="261.5" y="107">0.5%</text>
  <text class="cat" x="16" y="135">D1 · retrieve once</text>
  <text class="cat-sub" x="16" y="148">at the opening, before reasoning</text>
  <rect class="us-soft" x="250" y="127" width="214.2" height="18" rx="3"/>
  <text class="v-us" x="472.2" y="141">30.6%</text>
  <text class="cat" x="16" y="169">B0 · held while it thinks</text>
  <text class="cat-sub" x="16" y="182">no seed</text>
  <rect class="us-soft" x="250" y="161" width="223.3" height="18" rx="3"/>
  <text class="v-us" x="481.3" y="175">31.9%</text>
  <text class="cat" x="16" y="203">B1 · held while it thinks</text>
  <text class="cat-sub" x="16" y="216">seed</text>
  <rect class="us-soft" x="250" y="195" width="225.4" height="18" rx="3"/>
  <text class="v-us" x="483.4" y="209">32.2%</text>
  <text class="cat" x="16" y="237">C0 · full reprojection</text>
  <text class="cat-sub" x="16" y="250">no seed</text>
  <rect class="us" x="250" y="229" width="256.9" height="18" rx="3"/>
  <text class="v-us" x="514.9" y="243">36.7%</text>
  <text class="cat" x="16" y="271">C1 · the production system</text>
  <text class="cat-sub" x="16" y="284">full reprojection, seed</text>
  <rect class="us" x="250" y="263" width="262.5" height="18" rx="3"/>
  <text class="v-us" x="520.5" y="277">37.5%</text>
  <text class="tick mid" x="250" y="310">0%</text>
  <text class="tick mid" x="320" y="310">10%</text>
  <text class="tick mid" x="390" y="310">20%</text>
  <text class="tick mid" x="460" y="310">30%</text>
  <text class="tick mid" x="530" y="310">40%</text>
  <text class="tick mid" x="600" y="310">50%</text>
</svg>
<figcaption>Recall climbs the ladder the design predicts: nothing, then the
memory, then re-selection while it reasons. The intervals are in §9.12.</figcaption>
</figure>

The model alone scores **0.4%** — and that's a good sign: it says it has no
record of what's being asked rather than inventing an answer. Give it the memory
with one retrieval before it reasons and it gets **30.6%**. Let it re-select the
working set as it reasons — the production system — and it gets **37.5%**.

Paired question by question, and corrected for the number of comparisons:

| what's being measured | points of recall | 95% interval | p (Holm) |
|---|---:|---:|---:|
| the memory at all | **37.0** | 31.3 to 42.6 | 1.9 × 10⁻¹⁷ |
| re-selecting while it reasons | **5.3** | 3.2 to 7.4 | 2.8 × 10⁻⁶ |
| continuous, against retrieve-once | **6.9** | 3.6 to 10.1 | 1.7 × 10⁻⁴ |

Take away any part of provenance retrieval and recall drops, every interval
clear of zero.

## The ablation that matters

In August I called one experiment *the ablation that matters*: same index, same
fingerprints, same tiers, but retrieve **once**, before the model starts
reasoning. I said it would miss the dependencies you only reach by following one
fact to the next — not because they're absent from the index, but because at
line one the model hasn't reached them yet, and a single retrieval never looks
again. It does miss them, **by 6.9 points**.

v2 can also say *where*. The questions were split in advance by whether they
name the file their answer lives in. The gain from re-selecting lands on the
**111 that don't**: 5.7 points (3.5 to 8.0). On the 17 that do, it's 2.6 (−2.3
to 7.2), indistinguishable from zero.

> What the question names, the opening finds. What it doesn't, the reasoning
> has to reach.

You can watch it happen. With reprojection on, the share of answers with an
answer file in the working set climbs from 51.5% at the opening to **58.8% by
the 128th decoded token**. Held still, or retrieved once, it sits at 51.6% or
48.7% until the reasoning is over. And it matters: across 2,816 answers, recall
is **51.4%** when an answer file is in place for the first answer token, and
**13.7%** when it isn't.

<div class="key">
<h4>And the retrieval, on its own</h4>
<p>Over 121 of the questions, an answer file ranks <b>first out of 7,943 on
49%</b>, and in the top five on 69% — about 1,500 and 430 times a random
ranking. By sign agreement alone: no text index, no embedding model, just the
XNOR and popcount from August over a 36.1-million-token gallery.</p>
</div>

## The theorem, said properly

v1's headline was "O(1) error per token". A headline is not a theorem, so v2
says exactly what is bounded, per attention head, in two halves. **The
perturbation is bounded**: the change stored K/V error makes to the head's
output is at most a constant. **The dilution is bounded**: a token the step
needs keeps an attention weight bounded *below* by a constant. Both are
independent of context depth N, so the signal a step needs stays bounded against
the perturbation it carries, at any depth.

Under full attention the perturbation bound is **the same**. What changes is the
other half: among comparable distractors, the needed token's weight falls as
1/N. Depth doesn't make the noise louder. It makes the signal quieter.

That's sharper than August, and narrower. What grows with N under full attention
is a step's *exposure* — the summed stored error of everything it reads — and
exposure is not output error: attention averages its value errors, and measured
inside the windows we can reach, full-attention output error does not grow. What
full attention cannot offer is a *bound*.

The obvious objection is that the errors just average out. Grant it: with
independent, zero-mean errors spread over N comparable tokens, the perturbation
shrinks to about 1/√N. But the needed token's weight falls as 1/N, so the ratio
still falls as **1/√N**. Averaging slows the decline. It doesn't stop it.

<figure class="fig">
<svg viewBox="0 0 640 244" role="img" aria-label="Schematic of the theorem. As context depth N grows, the signal a step needs against the perturbation it carries stays bounded below under provenance-selected attention. Under full attention the needed token's weight falls as one over N, and even granting that errors average out the ratio still falls as one over root N.">
  <path class="ax" d="M70 40 V210 H610"/>
  <path class="c-rise" d="M70 60 L78.8 90 L87.7 110 L105.3 135 L123 150 L140.7 160 L176 172.5 L211.3 180 L246.7 185 L317.3 191.3 L423.3 196.4 L600 200.6"/>
  <path class="c-rise dash" d="M70 60 L78.8 75.8 L87.7 87.5 L105.3 103.9 L123 115.1 L140.7 123.4 L176 135 L211.3 142.9 L246.7 148.8 L317.3 157 L423.3 164.8 L600 172.5"/>
  <path class="c-flat" d="M70 60 H600"/>
  <text class="t-dim" x="70" y="30">the signal a step needs, against the perturbation it carries</text>
  <text class="t-flat" x="170" y="50">provenance-selected: bounded below, at any N</text>
  <text class="t-rise" x="600" y="142" text-anchor="end">full attention, errors averaged: still Θ(1/√N)</text>
  <text class="t-rise" x="600" y="183" text-anchor="end">full attention: the needed token's weight, Θ(1/N)</text>
  <text class="t-dim" x="70" y="232">schematic · relative to shallow depth · §11.2</text>
  <text class="t-dim" x="600" y="232" text-anchor="end">context depth N →</text>
</svg>
<figcaption>August drew this as an error line going up. Same argument, with the
axis labelled correctly: it's the signal that comes down.</figcaption>
</figure>

<div class="key">
<h4>And here's what it does not say</h4>
<p>The bound holds over whatever the selection attends, and is lost only if
selection grows the working set with N — which simply recovers full attention.
What the theorem can't tell you is whether the selection <em>holds the token
you need</em>. That's retrieval, which is why the exam is in the same paper as
the proof.</p>
</div>

## Twelve times the working set

v2 also tests the premise head-on. Cap the model's slot at 32,768 tokens, build
a conversation from the repository's own source to 30K, 131K and then **393K
tokens** — twelve times the slot — and plant four turns stating vault codes at
45% of its depth, too far in for an oldest-first fill and too far back for a
newest-first one. Then ask for the codes.

| needle turns selected, of four | 30K | 131K | 393K |
|---|:---:|:---:|:---:|
| provenance with reprojection, top compression (C10) | **4** | **4** | **4** |
| provenance with reprojection, near-lossless (C0) | **4** | **4** | **4** |
| one-shot projection, C10 | 3 | 0 | 0 |
| rolling window, no provenance | 3 | 0 | 0 |

Every planted turn, at every depth — and at the most aggressive compression
level exactly as at the gentlest, so compression doesn't change what is
selected. Without reprojection, nothing past 30K.

## What it runs on

- A **180B-parameter mixture-of-experts model on a 16 GB laptop GPU**, experts
  streamed VRAM → RAM → NVMe: **34.9 t/s** for one session and **108.5 t/s**
  aggregate across eight.
- **Flat in depth**: on a 72 GB RTX PRO 5000, from 32K to 128K tokens, **96–99%
  of prefill and 93–111% of decode** kept across two tasks.
- **2,869.9 t/s across 64 sessions** on one card, and a **284B model serving
  sixteen sessions from one GPU**.
- **KV compression of 4.1–7.6×** at the top level, every 32-token block choosing
  its own format as it's sealed.

The chart-by-chart version is [What one GPU can do](/blog/what-one-gpu-can-do).
And one correction, which you should hear from me: v1's "509 t/s" and "2,446
t/s" headlines were prefill rates, and they are withdrawn. v1 stays archived
with its erratum.

## What "final" means

It means the paper is finished — not the work. August's claim was that error
growth with depth is a property of full attention, not a law of compression, and
that you are allowed to leave. That is now a theorem that says precisely what it
bounds, an exam fixed before its first run with an interval on every contrast,
and a 42-million-token memory answering questions on one card.

Everything needed to check it is out: the engine, the raw rows behind every
figure, the tests that produce them, and the battery's questions, ground truth,
harness and scores. The code is [on GitHub](https://github.com/john-sharratt/candle),
and critical review is welcome. Next comes the 2027 conference round, and the
follow-up paper: reading sealed K/V as reliably as the
model reads text.

The KV-quantization literature keeps asking how small each token's error can be
made inside full attention. The better question was whether you can leave. v1
said you could; v2 is what it looks like from outside.

[Read the paper →](/papers/one-card-v2)
