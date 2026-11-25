---
title: "AI: One Card, One Stack — the final paper"
date: 2026-11-25
feature: 1
draft: true
tint: accent
tags: [inference, architecture, research]
summary: >-
  v2 of One Card, One Stack is published, and it is the final version. The
  battery v1 held back is measured: with tools forbidden, the memory is worth
  37.0 points of recall, re-selecting while the model reasons 5.3, and
  continuous reprojection 6.9 over a single retrieval — every interval clear of
  zero. A 180B model runs on a 16 GB laptop GPU. Here's what that means, and
  what it doesn't.
---

In August I wrote about [One Card, One Stack](/blog/one-card-unbounded-context),
and at the bottom of that post there was a box with a sentence in bold: **the
full quantitative table is held for v2**.

That was a promise, and today it's kept.

Version 2 of the paper is published on Zenodo —
[10.5281/zenodo.24529143](https://doi.org/10.5281/zenodo.24529143) — and it
is the final version. Not a draft, not a preview, not "v2 with a v3 to follow
when the numbers come in". The numbers came in. This is the paper.

> v1 argued that context is an architecture problem, not a budget problem.
> v2 **measures it** — on the production system, across three machines and
> twelve models, with every figure naming the build, the machine and the test
> that produced it.

## What v1 promised, and v2 delivered

v1 was honest about what it was: a technical report. The theorem was stated,
the kernels were measured, and the evaluation I actually cared about — whether
provenance retrieval holds up on a real codebase, against a real answer key —
was described in method and left unquantified.

v2 runs it. The test subject is still the system's own Candle fork, 2.2 million
lines of Rust and CUDA, and the battery is now 128 questions answered **with
tools forbidden** — the model can't grep, can't open a file, can't cheat. Seven
arms switch the memory's parts on and off, each run three to seven times, and
every contrast is corrected for the number of comparisons made.

Here's what each part is worth, in points of recall:

| what's switched on | recall gained |
|---|---:|
| the memory at all | **37.0** |
| re-selecting the working set while the model reasons | **5.3** |
| continuous reprojection, over a single retrieval | **6.9** |

Every one of those intervals is clear of zero after correction.

Read that last row again, because it's the one I've been arguing about since
August. In v1's post I called it *the ablation that matters*: keep the same
index, the same fingerprints, the same tiers, and retrieve **once** before the
model starts reasoning — which is architecturally what most coding assistants
do today. I said it would miss the dependencies that only come into view as the
reasoning moves.

It does. By 6.9 points, measured.

## The theorem, tightened

v1 stated the bound as "O(1) error per token". That's the right headline, but
a headline is not a theorem.

v2 states exactly what is bounded. Under provenance-selected attention, the
change stored K/V error can make to each attention head's output is bounded by
a constant, **and** a token the step actually needs keeps an attention weight
bounded below by a constant — both independent of context depth. So the
signal-to-perturbation ratio holds at any depth.

Under full attention, the needed token's weight falls as 1/N against the same
perturbation.

There's an obvious objection, and v2 answers it in the paper rather than in
the comments: *surely the errors average out?* If the per-token errors are
independent and zero-mean, the perturbation does shrink — to about 1/√N. But
the needed token's weight falls as 1/N, so the ratio still falls as **1/√N**
with depth. Averaging slows the decline. It doesn't stop it.

<div class="key">
<h4>And the honest edge of it</h4>
<p>The bound holds over whatever the selection attends. It says nothing about
whether the selection <em>holds the token you need</em> — that's retrieval, and
retrieval is an empirical question. Which is exactly why the battery exists,
and why it's in the same paper as the proof.</p>
</div>

## What one card does now

The architecture argument was always the point, but it had to run on
something, and the something got a great deal bigger between versions.

- A **180B-parameter mixture-of-experts model on a 16 GB laptop GPU**,
  experts streamed VRAM → RAM → NVMe, decoding one session at **34.9 t/s** —
  above the 27.5–29 published for an RTX 5080, the same 16 GB class with twice
  the host RAM — and eight sessions at **108.5 t/s** aggregate.
- **Flat in depth**: on a 72 GB RTX PRO 5000, from 32K to 128K tokens of
  context, it keeps **96–99% of its prefill and 93–111% of its decode** across
  two tasks.
- **2,869.9 t/s across 64 sessions** on one card, and a **284B model serving
  sixteen sessions from one GPU** at 4.1× the best published single-GPU decode.
- **KV compression of 4.1–7.6×** at the top level across the compressing
  models, every 32-token block choosing its own format as it's sealed.
- **Selection at depth**: every fact planted in a history twelve times the
  working set is selected — at the most aggressive compression level, exactly
  as at near-lossless.

On retrieval alone, with nothing but the model's own Q signatures, an answer
file comes first out of 7,943 on 49% of questions, and in the top five on 69%.

## The part I'd rather you heard from me

v1's headline throughput figures — 509 t/s and 2,446 t/s — were prefill rates.
They're withdrawn. The
erratum is in the v1 record and it stays there, and v2 says so in its own
appendix.

I'd rather publish a paper that carries its own correction than one that
quietly hopes nobody checks. Every number in v2 is produced by a test in the
repository, and the raw rows behind every figure are published beside it.

And there's one result in v2 that isn't a victory lap, and it's the most
interesting thing in the paper.

Give the same model the same repository through **BM25 retrieval** — plain
text, whole files, under a fixed budget — and it recalls more than it does from
the memory. Not because BM25 finds the answer more often. It doesn't: the
memory puts an answer file first **40.6%** of the time against BM25's
**28.9%**, and has an answer in view more often too.

The gap is in the *reading*. Shown an answer as text, the model names what it
carries almost every time. Shown the same content as sealed K/V, it names it
far less often. Hand the model the memory's own selection *as text*, under the
same budget, and recall climbs from 37.5% to 44.0% — **6.5 of those points are
the model reading K/V, not provenance choosing it**.

So the selection is right. The model just doesn't yet read its own memory as
well as it reads a page.

<div class="key">
<h4>Which is the next paper</h4>
<p>Reading sealed K/V as reliably as text is a problem with a sharp edge and a
measured size, and it's the follow-up. v2 doesn't paper over it — it puts the
arm in the figure, with its interval, and names what's left.</p>
</div>

## What publishing it means

It means the argument is on the record, with a timestamp nobody can move.

The claim in August was that the context window is the wrong thing to be
spending money on — that error growth with depth is a property of full
attention, not a law of compression, and that you're allowed to leave. That's
now a theorem with its objection answered, a battery with its intervals, and a
system running on hardware you can buy in a shop.

It means **one person and one card** can do work that gets compared, line by
line, with what datacentres publish — and hold up. Not because the card is
special. Because the constraint closed every easy answer in the first hour,
and what was left was the shape of the actual problem.

And it means the paper is finished, which is not the same as the work being
finished. Next is the conference round in 2027, and the follow-up on reading
sealed K/V.

v2 is final, and critical review of it is welcome — that's the whole reason
for publishing the rows, the harness and the tests alongside it. The code is
[on GitHub](https://github.com/john-sharratt/candle). If a number doesn't hold
up on your machine, I want to know.

[Read the paper →](/papers/one-card-v2)
