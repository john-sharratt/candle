---
title: "AI: One Card, One Stack"
date: 2026-08-11
feature: 4
tint: accent
tags: [inference, architecture]
summary: >-
  Everybody treats the context window as a budget problem. It isn't — it's an
  architecture problem, and there's a theorem. Under provenance-selected
  attention, error per generated token is O(1) at any depth, on any hardware,
  under any compression scheme.
---

Here's a question that sounds like it has an obvious answer.

What actually stops a language model from having a memory that never runs out?

The obvious answer is "the context window", and it's wrong. A window is a budget,
and budgets get raised. The vendors have raised them from 4K to 128K to a million
— and every single time, the same thing happens. The model gets measurably worse
at using the far end of what it can now technically see.

If the window were the only obstacle, that wouldn't happen. So something else is
going on, and it isn't the thing everybody is spending money on.

> Attention error doesn't grow because your cache is too small. It grows because
> **every token participates in every subsequent step**. That's structural. No
> amount of VRAM fixes it — it just defers it.

## What actually accumulates

Attention is a weighted sum. Every token in context takes part in every
computation that follows it, and each of those is arithmetic in finite precision.
The error in any one of them is tiny. The *number* of them is not.

Carry twice the context and you don't just pay twice the memory. You pay twice
the arithmetic **per generated token**, and the error in that token grows with it.
A wider float slows this down. It does not stop it.

And note the important word in that paragraph. *Arithmetic* — not quantization.
This is not a compression argument that happens to also cover floating point. The
quantity being bounded is total accumulated error from **every** source:
quantization noise, floating-point rounding, autoregressive drift. The proof uses
no property specific to compression whatsoever.

Turn compression off entirely and the shape of the curve doesn't move.

<figure class="fig">
<svg viewBox="0 0 640 260" role="img" aria-label="Two curves. Standard attention's error climbs with context depth; provenance-selected attention stays flat.">
  <defs>
    <linearGradient id="oc-rise" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="var(--crit)" stop-opacity=".10"/>
      <stop offset="100%" stop-color="var(--crit)" stop-opacity=".45"/>
    </linearGradient>
  </defs>
  <path class="ax" d="M54 16 V214 H600"/>
  <path d="M54 208 C 220 200, 380 150, 600 34 L600 214 L54 214 Z" fill="url(#oc-rise)"/>
  <path class="c-rise draw" pathLength="100" d="M54 208 C 220 200, 380 150, 600 34"/>
  <path class="c-flat draw" pathLength="100" d="M54 176 H600"/>
  <text class="t-dim" x="54" y="10">error per token</text>
  <text class="t-dim" x="600" y="238" text-anchor="end">context depth →</text>
  <text class="t-rise" x="430" y="86">full attention</text>
  <text class="t-rise big" x="430" y="112">O(N)</text>
  <text class="t-flat" x="66" y="164">retrieved subset</text>
  <text class="t-flat big" x="66" y="204">O(1)</text>
</svg>
<figcaption>The whole argument, drawn. One of those lines is a product roadmap.
The other one is a different architecture.</figcaption>
</figure>

Now here's the part that annoys people, and I'd like you to sit with it for a
moment.

An 80 GB H100 is in **exactly the same regime** as a 16 GB laptop card. It simply
arrives later. The moment you compress or evict *anything*, you are on the same
curve as everybody else. More hardware defers the threshold by a constant factor.
It does not remove it — because the exponent is what's wrong, and a constant
doesn't touch an exponent.

Sorry, folks. You cannot buy your way out of an asymptote.

I'll state the honest limit of that claim, because it's the version worth
defending. If your context genuinely never grows past what fits in full precision
on your card, then both architectures sit in a bounded-error regime and none of
this matters to you. The distinction bites on **persistent sessions that exceed
hardware capacity over time** — which is to say, every agent that's supposed to
still be useful next month.

## And every standard fix leaves the shape alone

Quantise the cache. Evict old blocks. Summarise the history. Retrieve into a
fixed window.

Every one of those helps with *memory*. Not one of them changes the *shape of the
error*, because every one of them keeps the same assumption underneath — that
everything in context should be attended to.

That assumption is the bug.

## The theorem, stated properly

So suppose the model attends not to everything it has, but to a bounded subset
selected for relevance. And crucially — this is the whole thing — that the
selection is made by **the model's own attention**, not by some embedding model
bolted on the side.

Partition the context into three tiers as N grows, and bound each one separately.

**Hot tier.** Tokens always present in the working set. Its size is capped by a
hardware constant, independent of N. These are prefill-refreshed, so their decode
drift is zero and their quantization error is bounded by the selection kernel's
threshold. Contribution: a small constant, `ε_hot`.

**Warm tier.** Tokens eligible for retrieval but not permanently resident. At most
`W_warm_max` of them can be selected per generation step — and this is the
load-bearing sentence in the entire paper — **regardless of how large the warm
corpus grows**. That bound is structural. The selection kernel enforces it by
construction. It is not a policy somebody could tune wrong on a Friday.
Contribution: `W_warm_max · ε_warm`, a constant.

**Cold tier.** Everything else. As N → ∞ the cold tier grows without bound while
the slots available to it stay fixed, so the probability any specific cold token
enters the working set on a given step falls as `1/N`. Contribution: `O(1/N)`,
which vanishes.

Sum the three, and you get a constant:

$$E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} + O\!\left(\frac{1}{N}\right) = O(1)$$

<figure class="fig">
<svg viewBox="0 0 640 244" role="img" aria-label="Two context depths, one thousand turns and ten million turns. The corpus bar grows enormously; the working set selected from it is identical in both cases.">
  <text class="t-lbl" x="70" y="34">context depth N = 1,000 turns</text>
  <rect class="box" x="70" y="42" width="160" height="22" rx="3"/>
  <rect class="box weights" x="70" y="78" width="64" height="22" rx="3"/>
  <text class="t-mono" x="144" y="93">working set — B tokens</text>
  <path class="link" d="M70 68 V78 M134 68 V78"/>
  <text class="t-lbl" x="70" y="140">context depth N = 10,000,000 turns</text>
  <rect class="box" x="70" y="148" width="520" height="22" rx="3"/>
  <rect class="box weights" x="70" y="184" width="64" height="22" rx="3"/>
  <text class="t-mono" x="144" y="199">working set — B tokens, identical</text>
  <path class="link" d="M70 174 V184 M134 174 V184"/>
  <text class="t-dim" x="70" y="230">error per generation step is the same in both rows — that is the entire theorem</text>
</svg>
<figcaption>The grey bar is what the system knows. The accent bar is what it
attends to. Only one of them grows.</figcaption>
</figure>

And there's a corollary that makes it sharper than merely "bounded".

Warm-tier blocks in this system don't arrive from nowhere. They originate as
prefill-refreshed hot-tier blocks — quantized from clean activations, admitted by
the selection kernel at threshold θ. So `ε_warm ≤ θ`, which is a small system
parameter *you control*. The total isn't just constant, it's a **small** constant
that tightens as you tighten θ.

Measured: always-attended blocks in top-quality compression mode reach **58.6 dB
K-SNR and 58.8 dB V-SNR**. That's `ε_hot ≈ 0` in practice, not in principle.

<div class="key">
<h4>And here's what would break it</h4>
<p>The selection budget and the hot-tier bound must be independent of N. That's
the entire load. If your selection mechanism grows the working set in proportion
to context depth — a perfectly natural thing to do if nobody is watching for it —
you land straight back in O(N) and the theorem says precisely nothing.</p>
<p>A theorem worth trusting is one where you can name the thing that kills it.
There it is.</p>
</div>

## So how does the model decide what it wants?

The selection mechanism is the part that took longest to get right, and it turns
on one observation.

At the moment a model processes any content, the query vectors **Q** it computes
are a compressed fingerprint of its cognitive state. Not just the semantics of the
text — the whole accumulated reasoning context it was produced in.

And that is qualitatively different from what K vectors hold. K encodes *what the
words mean*. A stored Q vector encodes *what the model was thinking about when it
read them* — topic trajectory, reasoning state, where it had got to. So when a
future query produces a Q vector in a similar reasoning state, the match surfaces
that content **regardless of surface-level token overlap**.

That last clause is why this isn't embedding search wearing a hat.

### The fingerprint is three vectors, not one

Transformer layers are not interchangeable, and treating them as one soup throws
away most of the signal. Layer-wise probing establishes three functionally
distinct regimes, so the index stores a separate fingerprint for each:

| band | layers | what lives there |
|---|---|---|
| Syntactic | 0 – N/3 | lexical and syntactic processing |
| Semantic | N/3 – 2N/3 | semantic category, emotional consolidation |
| Pragmatic | 2N/3 – N | relational reasoning, contextual integration |

Six INT8 vectors of 128 bytes plus six FP16 scales — **~780 bytes per indexed
item**. That's the whole fingerprint. And the band boundaries need nothing but the
model's layer count, so there's no per-model configuration for anybody to get
wrong.

Default depth weighting is **1 : 1 : 4** in favour of the pragmatic band, derived
from a cross-corpus sweep over 250K tokens and eight content types.
Discourse-level Q patterns carry the most discriminative power, consistently,
across content types that have very little else in common.

### And the scoring is deliberately crude

Take the sign of each Q component. Take the sign of each K. Count agreements.

That's it. **Binary directional provenance** — an XNOR and a popcount.

The signature is built by XOR-folding across eight (head, layer) subspaces, and
that gives it a property worth pausing on: it is **stable when heads coherently
agree, and cancels when they disagree**. Sustained directional focus survives the
fold. Noise destroys itself. The fingerprint is self-filtering, which is a very
nice thing to get for free.

On top sits span scoring. For a run of `L` consecutive probe tokens all hitting
the same section, the contribution is `L²` rather than `L`. Quadratic,
deliberately — so sustained focus dominates coincidence, and a section that gets
eight scattered lucky hits loses to one that gets four consecutive deliberate
ones.

The whole index — 50K turns plus 100K facts — is about **126 MB**, which fits in
L3 cache on a server CPU. Six INT8 matrix multiplies across six threads with
VNNI, and the scan completes in **3–10 ms regardless of corpus size**.

No hierarchy. No approximate-nearest-neighbour structure. Nothing to rebuild when
the corpus grows. No index build step to schedule at 3am. Flat scan, every single
time.

<div class="key">
<h4>The failure mode this had to design around</h4>
<p>Q vectors deviate more than <b>10× farther from K vectors than K vectors
deviate from each other</b> (Liu et al., NeurIPS 2025). Standard approximate
nearest-neighbour search falls apart across a gap that size.</p>
<p>So K fingerprints are built with <em>Q-aware token selection</em> — tokens
chosen by Q·K score rather than K magnitude, picking the K tokens most visible
from the Q distribution at construction time. Mitigate the gap when you build the
index, rather than suffering it every time you query. And the Q→Q matching
component, which dominates history retrieval, is within-distribution at both
ends.</p>
</div>

### Prefill Q and decode Q are completely different animals

This one surprised me, and it's the kind of thing that only turns up when you
actually build the thing rather than write about building the thing.

Q vectors produced while the model *reads* input, and Q vectors produced while it
*generates* output, occupy different regions of the fingerprint space entirely.
During prefill, attention spreads over all input tokens at once and Q reflects
lexical and structural features of the prompt. During decode, attention is causal
and sequential, and Q reflects generative **intent**.

Concretely. The query *"use a tool, determine 123891283 + 123124"* produces
exactly **one** discriminative prefill Q token across eight candidate tools. The
decode stream for the same request produces **sixteen** probe tokens that hit the
correct tool exclusively.

That is not a quantitative difference you tune away. It's a distributional shift,
and it means corpus fingerprints built from decode-phase Q **cannot be queried
with prefill-phase Q at all**. So the engine keeps two scoring profiles — span
scoring for decode's long coherent runs, and a per-token excess-agreement measure
for prefill's weak-but-genuine signal. Tool selection from 86 candidates runs on
the prefill profile; history retrieval during generation runs on the decode one.

Nobody tells you this. You find it out.

## Speculative Context Decode

Three to ten milliseconds is small. But it lands on every reasoning step, so it
doesn't run on the critical path at all.

Generation runs as a **pipelined two-session loop**. One session decodes real
output tokens against the current working set. In parallel, a *probe* session
speculatively decodes ahead — up to 64 tokens, terminating early at the first
newline. The probe's tokens are thrown away and never enter the KV cache. Only
its Q/K fingerprints are kept, and those drive the CPU scan that assembles the
working set for the *next* window.

<figure class="fig">
<svg viewBox="0 0 640 224" role="img" aria-label="Two-lane pipeline. The probe session runs one window ahead of the decode session, with a CPU provenance scan at each newline barrier.">
  <text class="t-lbl" x="16" y="66">PROBE</text>
  <text class="t-lbl" x="16" y="116">DECODE</text>
  <text class="t-lbl" x="16" y="164">CPU</text>
  <rect class="box" x="90" y="48" width="88" height="26" rx="4"/>
  <text class="t-mono mid" x="134" y="65">probe₁</text>
  <rect class="box" x="200" y="48" width="108" height="26" rx="4"/>
  <text class="t-mono mid" x="254" y="65">probe₂</text>
  <rect class="box" x="350" y="48" width="128" height="26" rx="4"/>
  <text class="t-mono mid" x="414" y="65">probe₃</text>
  <rect class="box" x="520" y="48" width="76" height="26" rx="4"/>
  <text class="t-mono mid" x="558" y="65">probe₄</text>
  <rect class="box weights" x="200" y="98" width="108" height="26" rx="4"/>
  <text class="t-mono mid" x="254" y="115">decode₁</text>
  <rect class="box weights" x="350" y="98" width="128" height="26" rx="4"/>
  <text class="t-mono mid" x="414" y="115">decode₂</text>
  <rect class="box weights" x="520" y="98" width="76" height="26" rx="4"/>
  <text class="t-mono mid" x="558" y="115">decode₃</text>
  <rect class="box" x="180" y="148" width="18" height="18" rx="3"/>
  <rect class="box" x="310" y="148" width="18" height="18" rx="3"/>
  <rect class="box" x="480" y="148" width="18" height="18" rx="3"/>
  <path class="link" d="M180 40 V172 M310 40 V172 M480 40 V172"/>
  <text class="t-dim" x="90" y="196">↵ barriers fall at newlines — the model's own reasoning boundaries, not a tuned interval</text>
  <text class="t-dim" x="90" y="214">probe tokens are discarded; they never enter the KV cache</text>
</svg>
<figcaption>The probe runs one window ahead. The CPU scan for window N+1 happens
while the GPU decodes window N — so the retrieval is free in wall-clock
terms.</figcaption>
</figure>

**The newline is the whole trick.** Newlines in model output — especially inside
thinking blocks — mark the completion of a discrete reasoning step. The Q vectors
at that boundary encode what the model is *reaching for next*, which makes it the
optimal moment to fingerprint. And because the window is variable, retrieval
frequency adapts to the model's own rhythm: a query prompting rapid short
inferences updates context often; an extended chain of logic produces longer probe
windows with more developed fingerprints.

There is no tuning parameter. None. The model's own structural markers set the
rate, and I consider that the most elegant thing in the entire system.

Probe divergence turns out not to matter, which is worth stating because it sounds
like it absolutely should. The probe runs ahead on a slightly different trajectory
than the real decode will take. But provenance operates on approximate
cognitive-state fingerprints averaged across layer bands, and a short stretch of
reasoning divergence doesn't move Q far enough to select the wrong blocks. The
system needs the approximate *neighbourhood* of the reasoning direction — not the
exact token sequence.

And the cost is far less than you'd expect, for a reason specific to this stack.
Probe and decode diverge on closely related content, so they route to largely the
same experts and coalesce efficiently in the [wave-batched grouped
GEMM](/blog/waves-and-the-pcie-bottleneck). Two sessions, well under 2× the work.

### But the scheduling was never the point

That's all cost. The quality argument is separate, and it's the better one.

Because the working set is assembled fresh at every reasoning boundary, **every
line of reasoning is generated against context selected for that specific line.**
The loop is: better context produces more specific Q vectors, which produce more
targeted retrieval, which produces better context for the next step.

Retrieval quality **compounds through** the reasoning chain, instead of being
fixed before it starts.

<div class="key">
<h4>And lost-in-the-middle simply stops existing</h4>
<p>Long-context models famously recall the start and end of a window far better
than the middle. That degradation is structural — flat attention over a long
window dilutes weight across positions, and interior positions get neither primacy
nor recency.</p>
<p>Provenance selection doesn't mitigate that failure mode. It deletes it. The
working set at each step is a small focused window assembled for that step, so
there is no "middle" in the pathological sense. Every block in it is proximal in
<em>relevance</em>, not in sequence position. Content from 50,000 tokens ago that
is provenance-relevant gets exactly the same attention density as content from 100
tokens ago.</p>
</div>

## The ablation that matters

Now, it would be very easy to assume the win here comes from having a big tiered
store and pulling things out of it. It does not, and the evaluation was built
specifically to prove that.

Swap provenance selection for **random** retrieval, keeping the identical
three-tier architecture, and accuracy on transitive and architectural dependencies
collapses to near zero.

The storage is not the contribution. The retrieval is.

The second ablation is sharper. Keep the provenance index — same fingerprints,
same tiers — but do a *single* retrieval before generation begins. Which is
architecturally what Cursor, Copilot and Claude Code all do: retrieve once, then
reason on a fixed window.

It misses exactly the dependencies that require following one fact to the next.

And the reason is precise rather than hand-wavy: those dependencies are **not
absent from its index**. They're sitting right there. But the model's Q vectors at
line one of reasoning haven't yet reached the region of the index that contains
them, and one-shot retrieval never gets another look. The gap isn't retrieval
accuracy at hop one — it's the compound effect of context improving through a
reasoning chain versus staying frozen solid.

And here's the striking part. Accuracy comes out **independent of how deep the
dependency chain goes**. Under O(N) error growth you'd expect degradation with
depth as compounding numerical error corrupts the intermediate context. Its
absence is the empirical signature of the O(1) regime — the theorem, pointed at
compositional reasoning instead of at arithmetic.

### The evaluation is the codebase itself

The test subject is the system's own Candle fork: **2.2 million lines of Rust and
CUDA**, ingested through a ~20M-token learning-phase conversation in three passes,
each building on the index the previous one produced. File-level analysis, then
cross-module dependency reasoning, then architectural reasoning about invariants
and error-propagation paths. The retrieval log across all three passes *is* the
dependency graph.

Ground truth is 200–300 relationships enumerated by hand across three categories.
Direct (visible in the import graph). Transitive (3+ hops). And architectural
invariants — **visible in no import graph at all**, things like *"which modules
share the assumption that sealed blocks are aligned to semantic boundaries?"* The
system's author is the definitive oracle. No crowd-sourcing, no proxy benchmark,
nobody to argue with about the answer key.

And the whole thing is portable, which I think is the most underrated property
here. The learning-phase conversation is **a token sequence on disk**. Any
instance of the engine, on any hardware, can prefill that sequence and reconstruct
the identical KV cache, fingerprint index and retrieval log.

The model's understanding of a codebase is a file. You can put it in version
control, next to the code.

<div class="key">
<h4>What this version reports, and what it doesn't</h4>
<p>The paper is a v1 technical report. Throughput and kernel benchmarks are fully
measured; the provenance calibration is fully reported across 1,024 scenarios and
250K tokens. The codebase dependency battery is described with methodology and
qualitative results — <b>the full quantitative table is held for v2</b>, and the
paper says so in its own status line rather than burying it in a footnote.</p>
<p>The live system is released so anyone can run the query battery themselves. I'd
rather you did.</p>
</div>

## Attention *is* retrieval

This is the most useful idea in the whole project, and it's a reframe rather than
a mechanism.

Standard attention is **flat retrieval at O(N) cost**. It scores every token
against your query, every step, and that exhaustive scoring is precisely why the
error accumulates.

RAG is the usual escape hatch, and it works by *replacing* the attention mechanism
with an external retriever — which is exactly why RAG systems lose attentional
continuity and feel like they're reading notes about a conversation rather than
remembering it. You've all felt it. That's what you were feeling.

Provenance selection is neither. It's **hierarchical retrieval at O(1) cost** —
approximate attention over the unbounded corpus on CPU, choosing which blocks get
exact attention on GPU. The mechanism is preserved. Only the scaling constraint is
removed.

<div class="key">
<h4>Why a bigger window is not the same thing</h4>
<p>Sliding windows don't degrade gracefully. They hit a <em>cliff</em>. Anything
beyond the window isn't approximate, it's unreachable by construction. A 131K
window over a 2.2M-line codebase covers a fraction of it, and the dependency you
need is either inside or it isn't.</p>
<p>A larger window defers the cliff. It doesn't soften it.</p>
</div>

## Three error mechanisms, not one

The theorem bounds quantization noise. That is not the only thing that can be
wrong, and the genuinely interesting engineering is in separating what the
literature routinely mashes together.

**Decode drift** is sequential. Each token's KV is computed by attending over
already-quantised predecessors, so error feeds forward and compounds
multiplicatively — coherence degrades past roughly 500 tokens. **Per-chunk
quantization noise** is not sequential at all: a 32-token block is quantised from
its own activations, so its error is bounded by its own distribution and doesn't
propagate anywhere.

Two independent mechanisms. Two independent fixes. Conflate them and you'll fix
neither.

The fixes then fall out cleanly. Blocks are sealed and quantised at 32 tokens,
which keeps the F16 working footprint to one block per session — materialising 200
concurrent sessions' turns at full precision would want ~18 GB on a 16 GB card,
so that's not a preference, it's arithmetic. And a lovely property comes free: the
*unsealed* tail is always F16, so **the most recent tokens — where drift would
otherwise be worst — are always attended at full precision**.

Then at turn completion a batched prefill pass recomputes the whole turn in
parallel from clean activations and replaces the drifted blocks. That's what holds
`ε_hot ≈ 0`, which the theorem identifies as the system's permanent error floor.

### And a third one that isn't numerical at all

Here's the subtle one. It took a real failure to find it.

Chat templates put 4–6 structural tokens at every turn seam — role markers, block
delimiters, envelope tags. They carry almost no semantic content, and they are
*disproportionately attended to* by everything downstream, because the model uses
them as orientation markers. The K vector at one of those positions means
something like *"I am a closing marker, sitting after the preceding content."*

Now select your context by provenance. Every projection picks a different
combination of sections — different tools materialised, different turns retrieved.
So the content that structural token was conditioned on **is not the content that
precedes it at runtime**, and its orientation signal is quietly, invisibly wrong.

Think about what that means. A structural token can be numerically perfect —
ε(t) = 0, exactly — and still be **wrong**, because the assumption that a stored K
faithfully summarises its attended context has been violated.

Under flat attention that's an edge case. Under provenance selection it is the
**common** case, by construction.

So structural template K/V is never persisted at all. It's live-prefilled at
assembly time against whatever content actually precedes it, batching into the
same forward pass as everything else. The orientation signal becomes correct by
construction rather than by luck, the most attention-loaded positions in the whole
context are guaranteed uncompressed, and the substrate gets *smaller* — because
templates stop being data and go back to being a property of the engine.

Three error classes. Three mechanisms. Nobody else is even counting.

## What the whole thing costs

Four subsystems, on a 16 GB card running a 30B mixture-of-experts:

| | |
|---|---|
| Expert cache (44% residency) | 8.0 GB |
| Attention layers (all 48) | 2.0 GB |
| Adaptive KV hot tier | 4.0 GB |
| Working buffers, activations, CUDA contexts | 2.0 GB |

The non-KV allocation is fixed at ~12 GB regardless of card, and everything left
over goes to the hot tier. Which makes the real design dial explicit — and it's
not "how much context". It's **how you split a fixed budget between concurrency
and per-session working set**:

| sessions | 16 GB — tokens/session | 24 GB — tokens/session |
|---:|---:|---:|
| 4 | ~15,900 | ~47,600 |
| 8 | ~7,900 | ~23,800 |
| 16 | ~4,000 | ~11,900 |
| 32 | ~2,000 | ~5,950 |
| 64 | ~990 | ~2,975 |

Read that table carefully, because the numbers are not what people assume. The
working set is the quality ceiling for each generation step — the number of tokens
provenance selection gets to choose from an unbounded history. These are **not**
context lengths. The history is unbounded in every single row. This is how much of
it the model looks at per step.

The two supporting subsystems each earned their own post: [adaptive per-block KV
quantization](/blog/palquant-per-block), and [streaming a 30B MoE over
PCIe](/blog/waves-and-the-pcie-bottleneck) to make the weights fit at all.

And what falls out the other end, measured on that 16 GB laptop: **20 concurrent
sessions holding 12,620 tokens of cache between them, at 2,314 t/s prefill and
166 t/s aggregate decode** — with the KV compressed and the model itself not
resident on the card at all.

That's the demonstration. It was never the point.

## The assumption everybody is optimising underneath

The KV quantization literature — KIVI, KVQuant, TurboQuant, all of it — is aimed
squarely at making the accumulation cheaper. Better formulas, better rotations,
better codebooks. Each of them measurably improves on what came before, and the
results stand.

And all of it takes for granted that error grows with N. It's so universal that
it's barely even stated.

But O(N) error scaling is not a law of compression. It is a **property of full
attention**, and you are allowed to leave.

Which is the part worth dwelling on. It isn't that anybody made a mistake — it's
that a boundary went unexamined because nothing in the work required examining
it. The assumption was load-bearing and invisible at the same time, which is the
most durable kind.

And the corollary is uncomfortable if you happen to own a datacentre. A machine
running full attention over a large corpus enters the accumulation regime **the
moment it evicts a single token** — and the theorem says no full-attention system,
on any hardware, at any budget, can climb back out of it.

Sixteen gigabytes or eight hundred. The only difference is when.

## What the small card was actually for

None of this would exist on a bigger machine. I want to be completely clear about
that, because it's the actual lesson.

Give me hardware big enough to hold everything and I'd have held everything — and
the accumulation problem would have stayed invisible until it surfaced years later
as a quality complaint nobody could localise. Give me 16 GB and a 30B model, and
every easy answer is closed off inside the first hour. What's left is the shape of
the actual problem.

And the pattern repeated often enough that it's worth naming. In each case a
constraint closed the standard solution, and the forced alternative wasn't merely
adequate — it was **better on unconstrained hardware too**:

| the constraint | what it closed | what it forced |
|---|---|---|
| 16 GB, 30B model | cuBLAS dequantising weights to BF16 — ~60 GB materialised, instant OOM | inline dequantisation inside the MMA kernel; never materialises the full-precision copy, and is strictly faster anywhere |
| Finite hot tier | holding a persistent session's full context | bounded working-set selection — which turned out to be exactly what the theorem requires |
| 200+ concurrent sessions | materialising all active turns at F16 | two-phase quantization, which correctly separates two error mechanisms the literature conflates |
| 44% expert residency | on-demand expert loading | online prediction that learns from production routing — generalises better than offline calibration, because it adapts to the real distribution rather than a proxy |
| Rotating system-prompt sections | storing K/V uniformly for every token | separating stored content from live-prefilled structure — closing a silent error class and shrinking the substrate |

Five constraints. Five forced alternatives. Five things that turn out to be
correct at any size.

That is not luck, and it's not me being clever. When a resource limit rules out a
standard approach, the replacement **has** to be more efficient in the dimension
the limit bounds. And if the constraint exposed a *genuine* inefficiency rather
than a necessary cost, that efficiency generalises. Every time.

So here's the thing I'd actually pass on, and it has nothing to do with inference.

Take your hardest constraint and promote it from a problem to a **requirement**.
It closes the standard solution and forces a better one, and the efficiency you
win in that dimension tends to travel to every other machine you'll ever run on.

The benchmarks demonstrate this on 16 GB.

The theorem explains why it holds everywhere else.

[Read the paper →](/papers/one-card)
