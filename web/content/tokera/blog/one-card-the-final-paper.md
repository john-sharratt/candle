---
title: "AI: One Card, One Stack — the table I owed you"
date: 2026-11-25
feature: 1
draft: true
tint: accent
tags: [inference, architecture, research]
summary: >-
  v2 of One Card, One Stack is published, and it is the final version. The
  battery v1 held back has been run: 128 questions about a 2.2M-line codebase,
  tools forbidden. The memory is worth 37.0 points of recall, and re-selecting
  while the model reasons beats retrieving once by 6.9. The theorem now says
  exactly what is bounded. And one result isn't a win at all — it's the next
  paper.
---

In August I wrote about [One Card, One Stack](/blog/one-card-unbounded-context),
and near the bottom of that post sat a box with one phrase in bold: **the full
quantitative table is held for v2**.

That's an IOU. Today it's paid.

Version 2 of the paper is published on Zenodo —
[10.5281/zenodo.24529143](https://doi.org/10.5281/zenodo.24529143) — and it is
the final version. Not a preview. Not "v2, with a v3 to follow when the numbers
come in". The numbers came in.

> v1 argued that context is an architecture problem, not a budget problem. v2
> **measures it** — on the production system, across three machines and twelve
> models, with every figure naming the build, the machine and the test that
> produced it.

Three things changed between the versions. The evaluation got run. The theorem
got sharper. And one result came back that I didn't order.

I'll take them in that order, and yes, the awkward one is last.

## The exam

The test subject hasn't changed: the system's own Candle fork, **2.2 million
lines of Rust and CUDA**. The model that serves it read the repository file by
file, as turns of its own, and sealed what it read. Nothing in that memory was
written by an external model.

Then it sat an exam.

**128 questions** — 43 direct, 43 transitive, 42 architectural — each listing
the items a correct answer has to name. They were written from the source and
committed, with the analysis, before the first run. The direct ones are
generated mechanically and picked by a seeded hash, so I don't get to choose my
favourites. And the battery itself is kept out of the memory, so the model has
never read its own exam paper.

One rule: **no tools**. It can't grep. It can't open a file. It answers from
what it remembers, or it doesn't answer.

Then switch the memory's parts on and off, one arm at a time, three to seven
runs for each memory arm — 4,608 answers in all, baselines included.

<figure class="fig">
<svg viewBox="0 0 640 424" role="img" aria-label="Mean recall by arm over 128 questions. A0, the model alone, 0.4 percent. A1, the seed files alone, 0.5 percent. D1, retrieve once before reasoning, 30.6 percent. B0, projection held in thinking, no seed, 31.9 percent. B1, held in thinking, seed, 32.2 percent. C0, full reprojection, no seed, 36.7 percent. C1, the production system, 37.5 percent. P1, the memory's own selection read as text, 44.0 percent. R1, BM25's files as text, 52.6 percent. T1, reading the repository with tools, 95.3 percent.">
  <text class="ttl" x="16" y="20">Recall by arm · 128 questions</text>
  <text class="ttl-sub" x="16" y="38">solid: re-selecting as it reasons · faint: selected once, or held · amber: handed text or tools, no memory</text>
  <path class="grid" d="M337.5 50 V396 M425 50 V396 M512.5 50 V396 M600 50 V396"/>
  <path class="grid base" d="M250 50 V396"/>
  <text class="cat" x="16" y="67">A0 · the model alone</text>
  <text class="cat-sub" x="16" y="80">no memory, no seed</text>
  <rect class="q1" x="250" y="59" width="1.4" height="18"/>
  <text class="v-them" x="259.4" y="73">0.4%</text>
  <text class="cat" x="16" y="101">A1 · the seed files alone</text>
  <text class="cat-sub" x="16" y="114">no memory projected</text>
  <rect class="q1" x="250" y="93" width="1.8" height="18"/>
  <text class="v-them" x="259.8" y="107">0.5%</text>
  <text class="cat" x="16" y="135">D1 · retrieve once</text>
  <text class="cat-sub" x="16" y="148">at the opening, before reasoning</text>
  <rect class="us-soft" x="250" y="127" width="107.1" height="18" rx="3"/>
  <text class="v-us" x="365.1" y="141">30.6%</text>
  <text class="cat" x="16" y="169">B0 · held while it thinks</text>
  <text class="cat-sub" x="16" y="182">no seed</text>
  <rect class="us-soft" x="250" y="161" width="111.7" height="18" rx="3"/>
  <text class="v-us" x="369.7" y="175">31.9%</text>
  <text class="cat" x="16" y="203">B1 · held while it thinks</text>
  <text class="cat-sub" x="16" y="216">seed</text>
  <rect class="us-soft" x="250" y="195" width="112.7" height="18" rx="3"/>
  <text class="v-us" x="370.7" y="209">32.2%</text>
  <text class="cat" x="16" y="237">C0 · full reprojection</text>
  <text class="cat-sub" x="16" y="250">no seed</text>
  <rect class="us" x="250" y="229" width="128.5" height="18" rx="3"/>
  <text class="v-us" x="386.5" y="243">36.7%</text>
  <text class="cat" x="16" y="271">C1 · the production system</text>
  <text class="cat-sub" x="16" y="284">full reprojection, seed</text>
  <rect class="us" x="250" y="263" width="131.3" height="18" rx="3"/>
  <text class="v-us" x="389.3" y="277">37.5%</text>
  <path class="ax thin" d="M16 291 H624"/>
  <text class="cat" x="16" y="305">P1 · the memory's picks, as text</text>
  <text class="cat-sub" x="16" y="318">R1's budget and prompt</text>
  <rect class="them" x="250" y="297" width="154" height="18" rx="3"/>
  <text class="v-them" x="412" y="311">44.0%</text>
  <text class="cat" x="16" y="339">R1 · BM25's files, as text</text>
  <text class="cat-sub" x="16" y="352">24,000-token budget</text>
  <rect class="them" x="250" y="331" width="184.1" height="18" rx="3"/>
  <text class="v-them" x="442.1" y="345">52.6%</text>
  <text class="cat" x="16" y="373">T1 · reading with tools</text>
  <text class="cat-sub" x="16" y="386">for reference</text>
  <rect class="them" x="250" y="365" width="333.6" height="18" rx="3"/>
  <text class="v-them" x="591.6" y="379">95.3%</text>
  <text class="tick mid" x="250" y="414">0%</text>
  <text class="tick mid" x="337.5" y="414">25%</text>
  <text class="tick mid" x="425" y="414">50%</text>
  <text class="tick mid" x="512.5" y="414">75%</text>
  <text class="tick mid" x="600" y="414">100%</text>
</svg>
<figcaption>Every arm in the paper, including the three I'd have an easier day
without. The intervals are in §9.12; the amber bars get a section of their own
further down.</figcaption>
</figure>

Read it from the top.

The model alone scores **0.4%**. That is not a bad score. It's a model that has
never read your code and has the manners to say so — it reports that it has no
record of what's being asked, rather than inventing one.

One retrieval, before it starts reasoning: **30.6%**.

Re-selecting the working set while it reasons — the production system:
**37.5%**.

And at the bottom, for reference, the same model allowed to open the files, at
12.3 tool calls an answer: 95.3%. That's what the questions allow.

Here's what each part is worth, paired question by question, corrected for the
number of comparisons made:

| what's being measured | arms | points of recall | 95% interval | p (Holm) |
|---|---|---:|---:|---:|
| the memory at all | C1 − A1 | **37.0** | 31.3 to 42.6 | 1.9 × 10⁻¹⁷ |
| re-selecting while it reasons | C1 − B1 | **5.3** | 3.2 to 7.4 | 2.8 × 10⁻⁶ |
| continuous, against retrieve-once | C1 − D1 | **6.9** | 3.6 to 10.1 | 1.7 × 10⁻⁴ |
| the seed files | C1 − C0 | 0.7 | −1.7 to 3.3 | 0.54 |

Take away any part of provenance retrieval and recall drops, with an interval
clear of zero.

The seed files — the ones a conversation opens on — change nothing measurable.
That row stays in the table because that's what came back.

## The ablation that matters, with a number on it

In August I called one experiment *the ablation that matters*. Keep the same
index, the same fingerprints, the same tiers — and retrieve **once**, before
the model starts reasoning. Retrieve, then think on a fixed window.

I said it would miss the dependencies that need one fact followed to the next.
Not because they're absent from the index. Because at line one of the reasoning
the model hasn't yet reached the part of the index that holds them, and a single
retrieval never gets another look.

It does miss them. **By 6.9 points.**

And v2 can say *where*. Before any run, the questions were split by whether
they name the file their answer lives in. The gain from re-selecting during
reasoning lands on the **111 that don't**: 5.7 points, 3.5 to 8.0. On the 17
that do, it's 2.6, with an interval from −2.3 to 7.2 — indistinguishable from
zero.

> What the question names, the opening finds. What it doesn't, the reasoning
> has to reach.

You can watch it happen, too. With reprojection on, the share of answers with
an answer file in the working set jumps from 51.5% at the opening to **58.8% by
the 128th decoded token** — and holds. Hold the projection still, or retrieve
once, and it sits at the opening's 51.6% or 48.7% until the reasoning is over.
Only then does it climb.

The file arrives. After the thought that needed it.

And it matters whether it arrives. Across 2,816 answers, recall is **51.4%**
when an answer file is in place for the first answer token, and **13.7%** when
it isn't. Retrieval and the answers are one measurement, seen twice.

<div class="key">
<h4>And the retrieval, on its own</h4>
<p>Strip the answers away and just ask where the right file ranks. Over 121 of
the questions, an answer file comes <b>first out of 7,943 on 49%</b>, and in the
top five on 69% — about 1,500 and 430 times a random ranking.</p>
<p>By sign agreement alone. No text index. No embedding model. The XNOR and the
popcount from August, pointed at a gallery of 36.1 million tokens.</p>
</div>

## The theorem, said properly

v1's headline was "O(1) error per token". That's a fine headline. A headline is
not a theorem.

v2 says exactly what is bounded, one attention head at a time, and there are
two halves to it.

**The perturbation is bounded.** The change that stored K/V error makes to each
head's output is at most a constant.

**The dilution is bounded.** A token the step actually needs keeps an attention
weight that is bounded *below* by a constant.

Both independent of context depth N. So the ratio of the signal a step needs to
the perturbation it carries is bounded below at any depth.

Now the part that's easy to miss. Under full
attention the perturbation bound is **the same**. What changes is the other
half: among comparable distractors, the needed token's weight falls as 1/N.

Depth doesn't make the noise louder. It makes the signal quieter.

That is a sharper claim than August's, and a narrower one, and I'd rather be
precise about the difference. What grows linearly with N under full attention is
a step's *exposure* — the summed stored error of everything it reads. Exposure
is not output error. Attention's output is a weighted average, so value errors
are averaged, not summed; and measured, inside the windows we can reach,
full-attention output error does not grow. The paper says so and puts the
figures beside it. What full attention cannot offer is a *bound*.

Which brings up the obvious objection, and v2 answers it in the paper rather
than in the comments: *surely the errors just average out?*

Grant it. Say the value errors are independent and zero-mean. Then with weight
spread over N comparable tokens, the perturbation does shrink — to about 1/√N.
But the needed token's weight is falling as 1/N. So the ratio still falls, as
**1/√N**.

Averaging slows the decline. It doesn't stop it.

<figure class="fig">
<svg viewBox="0 0 640 244" role="img" aria-label="Schematic of the theorem. As context depth N grows, the signal a step needs against the perturbation it carries stays bounded below under provenance-selected attention. Under full attention the needed token's weight falls as one over N, and even granting that errors average out the ratio still falls as one over root N.">
  <path class="ax" d="M70 40 V210 H610"/>
  <path class="c-rise" d="M70 60 L78.8 90 L87.7 110 L105.3 135 L123 150 L140.7 160 L176 172.5 L211.3 180 L246.7 185 L317.3 191.3 L423.3 196.4 L600 200.6"/>
  <path class="c-rise dash" d="M70 60 L78.8 75.8 L87.7 87.5 L105.3 103.9 L123 115.1 L140.7 123.4 L176 135 L211.3 142.9 L246.7 148.8 L317.3 157 L423.3 164.8 L600 172.5"/>
  <path class="c-flat" d="M70 60 H600"/>
  <text class="t-dim" x="70" y="30">the signal a step needs, against the perturbation it carries</text>
  <text class="t-flat" x="170" y="50">provenance-selected: bounded below, at any N</text>
  <text class="t-rise" x="296" y="142">full attention, errors averaged: still Θ(1/√N)</text>
  <text class="t-rise" x="330" y="183">full attention: the needed token's weight, Θ(1/N)</text>
  <text class="t-dim" x="70" y="232">schematic · relative to shallow depth · §11.2</text>
  <text class="t-dim" x="600" y="232" text-anchor="end">context depth N →</text>
</svg>
<figcaption>August drew this as an error line going up. Same argument, with the
axis labelled correctly: it's the signal that comes down.</figcaption>
</figure>

<div class="key">
<h4>And here's what it does not say</h4>
<p>The bound holds over whatever the selection attends, and it is lost only if
selection grows the working set with N — which simply recovers full attention.
The scan that does the choosing <em>is</em> linear in N: it reads every
signature. But it's 1-bit work, once every 64 decoded tokens, off the decode's
critical path.</p>
<p>What the theorem cannot tell you is whether the selection <em>holds the
token you need</em>. That's retrieval, and retrieval is an empirical question.
Which is exactly why the exam is in the same paper as the proof.</p>
</div>

## Twelve times the working set

There's a direct test of the premise in v2, and it's the meanest one I could
build.

Cap the model's slot at 32,768 tokens. Build a conversation out of the
repository's own source, one file a turn, to 30K, then 131K, then **393K
tokens** — twelve times the slot. At 45% of its depth, plant four turns stating
vault codes: too far in for an oldest-first fill, too far back for a
newest-first one. Then ask for the codes.

Needle turns selected, out of four:

| | 30K | 131K | 393K |
|---|:---:|:---:|:---:|
| provenance with reprojection, top compression (C10) | **4** | **4** | **4** |
| provenance with reprojection, near-lossless (C0) | **4** | **4** | **4** |
| one-shot projection, C10 | 3 | 0 | 0 |
| rolling window, no provenance | 3 | 0 | 0 |

Every planted turn, every depth, at the most aggressive compression level
exactly as at the gentlest. Compression does not change what is selected. And
the arms without reprojection select **none** past 30K.

Now the other half of that table, the half that would be very easy to leave out.

Past 30K the model holds needle turns in its working set — and recalls one code
out of four at best. Mostly it answers that no codes were given.

The paper chases that down by elimination, and the answer is not selection and
not compression. Take the failing projection's own turns, prefill them again
into a fresh slot, same positions, same C10, and the model recalls every code
they hold. What differs is the *context the K/V was computed in*. A sealed turn
carries its context with it, and is read as what it was in that context — not
in another.

Production memory is built so the two match: every ingested file is sealed as
its own conversation, behind the system prompt alone. Measured in that
arrangement, 4/4. The needle test fails in the one arrangement production
doesn't use — turns sealed deep inside one long conversation and read out of it.

So selecting is one problem and reading is another. Hold that thought. It comes
back.

## What it runs on now

The architecture was always the argument, but it has to run on something, and
the something grew.

- A **180B-parameter mixture-of-experts model on a 16 GB laptop GPU**, experts
  streamed VRAM → RAM → NVMe. One session decodes at **34.9 t/s** — above the
  27.5–29 published for an RTX 5080, the same 16 GB class with twice the host
  RAM — and eight at **108.5 t/s** aggregate.
- **Flat in depth.** On a 72 GB RTX PRO 5000, from 32K to 128K tokens, it keeps
  **96–99% of its prefill and 93–111% of its decode** across two tasks.
- **2,869.9 t/s across 64 sessions** on one card. Aggregate decode at
  **1.4–37.9×** the best published llama.cpp single-stream rate for the same
  model and card class.
- A **284B model serving sixteen sessions from one GPU**, at 4.1× the best
  published single-GPU decode.
- **KV compression of 4.1–7.6×** at the top level across the compressing
  models, every 32-token block choosing its own format as it's sealed.

That's the abstract's worth. The long version, chart by chart, is [What one GPU
can do](/blog/what-one-gpu-can-do).

And one correction, which you should hear from me. v1's "509 t/s" and
"2,446 t/s" headlines were prefill rates, and they are withdrawn. v1 stays
archived with its erratum, and v2 says so in its own appendix.

## The result I didn't order

There is one arm in that first chart that isn't a victory lap, and it's the
most interesting thing in the paper.

Give the same model the same repository through **BM25**. Plain text retrieval:
rank the files, pack the best ones whole into the prompt until a 24,000-token
budget runs out, and ask.

It recalls **52.6%**. The memory recalls 37.5%. That's 15.1 points — interval
7.1 to 23.0 — in BM25's favour.

Sorry, folks. I'd have liked that the other way round too.

So where does it come from? Not from BM25 finding the answer more often. It
doesn't:

- The memory's selection puts an answer file **first on 40.6%** of questions.
  BM25: 28.9%.
- Of the items a correct answer must name, **48.5%** were sitting in the text of
  an answer file the memory placed. For BM25's files, 38.2%. Anywhere in view at
  all: 60.4% against 56.0%.

The memory puts the answer in front of the model more often. The gap opens
*after* that:

| what the model was holding | item in an answer file's text | named, when it's there |
|---|---:|---:|
| the memory — sealed K/V | **48.5%** | 69.9% |
| BM25's files — text | 38.2% | **97.6%** |
| the memory's own selection — as text | 39.8% | 96.0% |

(One run of each, 445 expected items, looked for in what the model held as its
answer began.)

Shown an item as plain text, the model names it 97.6% of the time. Shown it as
sealed K/V, 69.9%.

The third row is the experiment that separates the two. Take exactly what the
memory selected, hand it over **as text** under BM25's budget and prompt, and
recall climbs from 37.5% to **44.0%** — and the model names 96.0% of what an
answer file carries.

**So 6.5 of the 15.1 points are the model's reading of K/V, not provenance's
selection.**

And the remainder is not hidden either. Read as text, the memory's selection is
still 8.5 points behind BM25's — interval 0.1 to 16.9, which only just clears
zero — and that sits with BM25's wider spread of whole files under a fixed text
budget.

Put the two halves together and it's the vault codes again, on a real workload.
The selection is right more often than the keyword search. The model just
doesn't yet read its own memory as reliably as it reads a page.

<div class="key">
<h4>Which is the next paper</h4>
<p>Reading sealed K/V as reliably as text is a problem with a sharp edge and a
measured size, and it's the follow-up. How well a model reads K/V placed from
memory is a capability of the model — and it has grown with every model this
system has been upgraded to.</p>
<p>v2 doesn't paper over it. The arm is in the figure, with its interval, next
to the ones that flatter me.</p>
</div>

## What "final" means

It means the paper is finished, which is not the same thing as the work being
finished.

The claim in August was that error growth with depth is a property of full
attention, not a law of compression — and that you are allowed to leave. That
is now a theorem that says precisely what it bounds, with its best objection
answered in the text. It's an exam with its questions fixed before the first
run and an interval on every contrast. And it's a system running on three cards,
from 16 GB to 72 GB.

Everything needed to check it is out: the engine, the raw rows behind every
figure, the tests that produce them, and the battery — questions, ground truth,
harness, analysis, per-answer scores. Every arm in that first chart is a
setting on the request, so you can run the ones I'd rather you didn't. The code
is [on GitHub](https://github.com/john-sharratt/candle).

v2 is the final version of this paper, and critical review of it is welcome.
Next comes the conference round in 2027 — MLSys, ACL, EMNLP, NeurIPS — and the
follow-up on reading sealed K/V.

The question the KV-quantization literature keeps asking is how small each
token's error can be made inside the full-attention regime. The question worth
asking was whether the regime can be left.

v1 said it could.

v2 is what it looks like from outside — measured, with the unfinished bit
marked on the map.

[Read the paper →](/papers/one-card-v2)
