---
title: "AI: one KV format per block, not per model"
date: 2026-08-21
feature: 2
tint: violet
tags: [quantization, kv-cache]
summary: >-
  Every KV cache quantizer argues about which precision is best. That's the
  wrong argument — the blocks aren't the same, so the answer shouldn't be
  either. 7.4× compression, no calibration data, and a benchmark convention
  that turns out to be measuring nothing.
---

Here's the shape of basically every paper on KV cache quantization ever written:

> We propose a new quantization *primitive*. It is better than the previous
> primitive. We apply it to every block in the cache.

2-bit uniform. Outlier-aware 3-bit. Rotate first, then Lloyd-Max. Boost the top
12% of channels by offline magnitude rank.

Every one of those is a real advance on the one before it. And every one of them
is having precisely the same argument — *which single format is best?* — while
agreeing on the premise sitting underneath it.

Which is that one format should be applied everywhere.

Nobody argues about that bit. It's not defended anywhere. It's just there.

<div class="key">
<h4>The premise is the problem</h4>
<p>A single format must be conservative enough for your <em>worst</em> block, or
aggressive enough to break your easiest ones. You price the whole cache at the
cost of its most demanding 1%.</p>
</div>

## Blocks are not alike

Go and look at real pre-RoPE activations, 32 elements at a time, and the variance
is enormous.

Some blocks are nearly constant — one centroid reconstructs them to within noise.
Some are constant with a single violent spike. And some carry the session's entire
discriminating signal and fall over if you look at them funny.

Now, spending four bits per element on every one of those is a genuinely strange
thing to do once you've actually seen the distribution. We've just all been doing
it for so long that nobody stops to say so.

## So: don't

So don't pick a format. Pick four, per head, per chunk, and let the data sort
itself out.

Give each `(layer, head, chunk)` a **palette of four formats** drawn from a fixed
codebook of sixteen — spanning 0.25 to 16 bits per element — and route each of its
128 blocks to one of the four by its *measured reconstruction error*. Two bits per
block records which one it went to.

That's the whole idea. Everything below is what it costs to make it real.

The selection is greedy set-cover, not first-fit. Blocks get sorted by absolute
magnitude descending, which concentrates the demanding ones at the head of the
array so the search can stop early. Then each slot in turn walks the codebook
cheapest-first and takes the (format, outer-scale) pair that **claims the most
blocks still unclaimed** — and those blocks leave the pool.

The direction that follows is the opposite of the one you'd guess. Slot 0 gets
the *cheapest* format, because it is picking from the full set and something
aggressive can still cover a quota of it. Slot 1 then faces only what slot 0
couldn't cover — a strictly harder residual — so it has to spend more bits. By
slot 3 what remains is whatever nothing cheaper could handle, and that slot is
the conservative one. The palette escalates.

<figure class="fig">
<svg viewBox="0 0 640 230" role="img" aria-label="128 blocks of one head claimed by four palette slots in turn, the format escalating from a sub-1-bit template up to a conservative 8-bit format as the residual gets harder.">
  <text class="t-dim" x="16" y="20">128 blocks of one head · each slot claims exactly 32</text>
  <g class="blocks">
    <rect class="q1" x="16"  y="34" width="150" height="22" rx="4"/>
    <rect class="q2" x="172" y="34" width="150" height="22" rx="4"/>
    <rect class="q3" x="328" y="34" width="150" height="22" rx="4"/>
    <rect class="q4" x="484" y="34" width="140" height="22" rx="4"/>
  </g>
  <path class="ax thin" d="M91 62 V96 M247 62 V96 M403 62 V96 M554 62 V96"/>
  <g class="slots">
    <rect class="s q1" x="16"  y="100" width="150" height="58" rx="8"/>
    <rect class="s q2" x="172" y="100" width="150" height="58" rx="8"/>
    <rect class="s q3" x="328" y="100" width="150" height="58" rx="8"/>
    <rect class="s q4" x="484" y="100" width="140" height="58" rx="8"/>
  </g>
  <text class="t-lbl" x="91"  y="126" text-anchor="middle">slot 0</text>
  <text class="t-mono" x="91"  y="146" text-anchor="middle">Q0 · 0.25 bpe</text>
  <text class="t-lbl" x="247" y="126" text-anchor="middle">slot 1</text>
  <text class="t-mono" x="247" y="146" text-anchor="middle">Q2 · 2.25 bpe</text>
  <text class="t-lbl" x="403" y="126" text-anchor="middle">slot 2</text>
  <text class="t-mono" x="403" y="146" text-anchor="middle">Q4 · 4.5 bpe</text>
  <text class="t-lbl" x="554" y="126" text-anchor="middle">slot 3</text>
  <text class="t-mono" x="554" y="146" text-anchor="middle">Q8 · 8.5 bpe</text>
  <text class="t-dim" x="16" y="192">cheapest format first · what it can't cover gets harder, so the format escalates</text>
  <text class="t-dim" x="16" y="212">2 bits per block records which slot it went to</text>
</svg>
<figcaption>A head with bimodal difficulty ends up with three aggressive slots
and one careful one — which no single per-head format can express.</figcaption>
</figure>

## What's actually in the codebook

Most of the sixteen are the integer quantizers you'd expect: 8-, 4-, 3-, 2-bit,
symmetric and asymmetric, FP16 or INT8 scales. The interesting end is the
bottom, where three formats exist because real activations have shapes that
integer quantization can't express cheaply:

- **Q0 — the constant block, 0.25 bpe.** One INT8 centroid for all 32 elements.
  A single byte, for blocks where the model is projecting essentially nothing.
- **Q0_X — constant with one escape, 0.50 bpe.** An INT8 anchor plus a single
  outlier: five bits of position, three of signed delta. The near-flat block
  with one violent spike — which integer quantization prices at the spike's
  magnitude across all 32 elements. Two bytes instead.
- **Q0_V — a parametric curve, 0.50 bpe.** Three indices into constant-memory
  tables: which curve, which scale, which centroid. The block is reconstructed
  as a *template* rather than as samples.

Those tables — 256 curves × 32 scales × 8 centroids, 8.5 KB — are the only
model-derived component anywhere in the system, calibrated once and shipped as
constants. Everything else is arithmetic.

By the most aggressive level, **28.5% of all elements** are being served by that
sub-1-bit family. It is not a rounding error in the ladder; it is where the last
of the compression comes from.

## What the palettes actually choose

This is the part I found most surprising, and it only shows up in a real trace.

Twenty chunks of Qwen3-8B at a mid-range level. On the **K side**, the palettes
are almost invariant: 24 or 25 of the 32 (head, slot) pairs pick Q3_0 every
single chunk, and the rest pick Q3_1. Chunk after chunk, near-identical.

On the **V side** they move constantly — some chunks fit cleanly into a tight
Q3 mix, others spread across five formats with occasional Q8_0 escalations.

Zoom into one chunk and it's sharper still. All eight heads have *identical* K
palettes. Their V palettes diverge completely:

<figure class="fig">
<svg viewBox="0 0 640 322" role="img" aria-label="One chunk of eight heads. Every head's K palette is the identical four formats; every head's V palette differs.">
  <defs>
    <!-- Every K row is the same four formats, so the source says so once. -->
    <g id="krow">
      <rect class="q1" width="50" height="19" rx="3"/>
      <rect class="q1" x="55" width="50" height="19" rx="3"/>
      <rect class="q1" x="110" width="50" height="19" rx="3"/>
      <rect class="q2" x="165" width="50" height="19" rx="3"/>
    </g>
  </defs>
  <text class="t-lbl" x="56" y="20">K — every head identical</text>
  <text class="t-lbl" x="372" y="20">V — every head different</text>
  <text class="t-mono" x="56" y="40">p0    p1    p2    p3</text>
  <text class="t-mono" x="372" y="40">p0    p1    p2    p3</text>
  <use href="#krow" x="56" y="52"/><use href="#krow" x="56" y="78"/>
  <use href="#krow" x="56" y="104"/><use href="#krow" x="56" y="130"/>
  <use href="#krow" x="56" y="156"/><use href="#krow" x="56" y="182"/>
  <use href="#krow" x="56" y="208"/><use href="#krow" x="56" y="234"/>
  <g>
    <rect class="q1" x="372" y="52"  width="50" height="19" rx="3"/><rect class="q1" x="427" y="52"  width="50" height="19" rx="3"/><rect class="q1" x="482" y="52"  width="50" height="19" rx="3"/><rect class="q3" x="537" y="52"  width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="78"  width="50" height="19" rx="3"/><rect class="q1" x="427" y="78"  width="50" height="19" rx="3"/><rect class="q1" x="482" y="78"  width="50" height="19" rx="3"/><rect class="q2" x="537" y="78"  width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="104" width="50" height="19" rx="3"/><rect class="q1" x="427" y="104" width="50" height="19" rx="3"/><rect class="q1" x="482" y="104" width="50" height="19" rx="3"/><rect class="q2" x="537" y="104" width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="130" width="50" height="19" rx="3"/><rect class="q1" x="427" y="130" width="50" height="19" rx="3"/><rect class="q1" x="482" y="130" width="50" height="19" rx="3"/><rect class="q3" x="537" y="130" width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="156" width="50" height="19" rx="3"/><rect class="q1" x="427" y="156" width="50" height="19" rx="3"/><rect class="q1" x="482" y="156" width="50" height="19" rx="3"/><rect class="q2" x="537" y="156" width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="182" width="50" height="19" rx="3"/><rect class="q1" x="427" y="182" width="50" height="19" rx="3"/><rect class="q1" x="482" y="182" width="50" height="19" rx="3"/><rect class="q4" x="537" y="182" width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="208" width="50" height="19" rx="3"/><rect class="q2" x="427" y="208" width="50" height="19" rx="3"/><rect class="q3" x="482" y="208" width="50" height="19" rx="3"/><rect class="q4" x="537" y="208" width="50" height="19" rx="3"/>
    <rect class="q1" x="372" y="234" width="50" height="19" rx="3"/><rect class="q1" x="427" y="234" width="50" height="19" rx="3"/><rect class="q2" x="482" y="234" width="50" height="19" rx="3"/><rect class="q4" x="537" y="234" width="50" height="19" rx="3"/>
  </g>
  <text class="t-mono hd" x="40" y="66">h0</text><text class="t-mono hd" x="40" y="92">h1</text>
  <text class="t-mono hd" x="40" y="118">h2</text><text class="t-mono hd" x="40" y="144">h3</text>
  <text class="t-mono hd" x="40" y="170">h4</text><text class="t-mono hd" x="40" y="196">h5</text>
  <text class="t-mono hd" x="40" y="222">h6</text><text class="t-mono hd" x="40" y="248">h7</text>
  <g class="legend">
    <rect class="q1" x="56"  y="286" width="14" height="14" rx="3"/><text class="t-mono" x="76"  y="298">Q3_0 · 3.5 bpe</text>
    <rect class="q2" x="196" y="286" width="14" height="14" rx="3"/><text class="t-mono" x="216" y="298">Q3_1 · 4.0</text>
    <rect class="q3" x="316" y="286" width="14" height="14" rx="3"/><text class="t-mono" x="336" y="298">Q4_0 · 4.5</text>
    <rect class="q4" x="436" y="286" width="14" height="14" rx="3"/><text class="t-mono" x="456" y="298">Q4_1 · 5.0</text>
  </g>
  <path class="ax thin" d="M56 272 H587"/>
</svg>
<figcaption>One chunk, eight heads, four palette slots each. The K side has
converged on a single answer and repeats it; the V side has converged on
nothing — which is where the adaptivity is actually doing work.</figcaption>
</figure>

Which inverts the intuition the literature gives you. K is the sensitive side —
its errors go through a softmax, 1-bit K destroys a model while 1-bit V is
survivable in most layers. But *sensitive* and *variable* are different
properties. K is uniformly demanding, so a fixed conservative choice serves it
well. **V is where the adaptivity actually earns its keep**, because V is where
the blocks differ from each other.

### The mapping underneath is not a mapping anyone would have designed

Go one level down and it gets stranger. A palette has four slots; a head has 128
blocks; the block index *is* the head-dimension channel. So every head carries a
128-entry table saying which channel is quantized how — and that table is the
thing the selector actually produces.

You would expect it to be banded. Low channels one way, high channels another;
some smooth gradient you could approximate with a formula and skip the search.
It isn't. Here is the real routing for one chunk, one K head and one V head,
every channel coloured by the slot it landed in:

<figure class="fig">
<svg viewBox="0 0 640 232" role="img" aria-label="Palette slot assignment for all 128 head-dimension channels of one K head and one V head. Both are scattered with no banding.">
  <text class="t-lbl" x="48" y="20">K · head 1 — 128 channels</text>
  <text class="t-mono hd" x="42" y="41" text-anchor="end">0</text>
  <text class="t-mono hd" x="42" y="56" text-anchor="end">32</text>
  <text class="t-mono hd" x="42" y="71" text-anchor="end">64</text>
  <text class="t-mono hd" x="42" y="86" text-anchor="end">96</text>
    <rect class="s1" x="48" y="30" width="16" height="14" rx="2"/><rect class="s3" x="65" y="30" width="16" height="14" rx="2"/><rect class="s1" x="82" y="30" width="16" height="14" rx="2"/><rect class="s1" x="99" y="30" width="16" height="14" rx="2"/><rect class="s2" x="116" y="30" width="16" height="14" rx="2"/><rect class="s2" x="133" y="30" width="16" height="14" rx="2"/><rect class="s2" x="150" y="30" width="16" height="14" rx="2"/><rect class="s2" x="167" y="30" width="16" height="14" rx="2"/><rect class="s2" x="184" y="30" width="16" height="14" rx="2"/><rect class="s0" x="201" y="30" width="16" height="14" rx="2"/><rect class="s0" x="218" y="30" width="16" height="14" rx="2"/><rect class="s1" x="235" y="30" width="16" height="14" rx="2"/><rect class="s1" x="252" y="30" width="16" height="14" rx="2"/><rect class="s0" x="269" y="30" width="16" height="14" rx="2"/><rect class="s2" x="286" y="30" width="16" height="14" rx="2"/><rect class="s3" x="303" y="30" width="16" height="14" rx="2"/><rect class="s0" x="320" y="30" width="16" height="14" rx="2"/><rect class="s3" x="337" y="30" width="16" height="14" rx="2"/><rect class="s0" x="354" y="30" width="16" height="14" rx="2"/><rect class="s3" x="371" y="30" width="16" height="14" rx="2"/><rect class="s1" x="388" y="30" width="16" height="14" rx="2"/><rect class="s1" x="405" y="30" width="16" height="14" rx="2"/><rect class="s2" x="422" y="30" width="16" height="14" rx="2"/><rect class="s0" x="439" y="30" width="16" height="14" rx="2"/><rect class="s2" x="456" y="30" width="16" height="14" rx="2"/><rect class="s3" x="473" y="30" width="16" height="14" rx="2"/><rect class="s0" x="490" y="30" width="16" height="14" rx="2"/><rect class="s3" x="507" y="30" width="16" height="14" rx="2"/><rect class="s2" x="524" y="30" width="16" height="14" rx="2"/><rect class="s2" x="541" y="30" width="16" height="14" rx="2"/><rect class="s0" x="558" y="30" width="16" height="14" rx="2"/><rect class="s1" x="575" y="30" width="16" height="14" rx="2"/>
    <rect class="s1" x="48" y="45" width="16" height="14" rx="2"/><rect class="s0" x="65" y="45" width="16" height="14" rx="2"/><rect class="s3" x="82" y="45" width="16" height="14" rx="2"/><rect class="s0" x="99" y="45" width="16" height="14" rx="2"/><rect class="s0" x="116" y="45" width="16" height="14" rx="2"/><rect class="s3" x="133" y="45" width="16" height="14" rx="2"/><rect class="s0" x="150" y="45" width="16" height="14" rx="2"/><rect class="s1" x="167" y="45" width="16" height="14" rx="2"/><rect class="s2" x="184" y="45" width="16" height="14" rx="2"/><rect class="s3" x="201" y="45" width="16" height="14" rx="2"/><rect class="s1" x="218" y="45" width="16" height="14" rx="2"/><rect class="s2" x="235" y="45" width="16" height="14" rx="2"/><rect class="s1" x="252" y="45" width="16" height="14" rx="2"/><rect class="s2" x="269" y="45" width="16" height="14" rx="2"/><rect class="s1" x="286" y="45" width="16" height="14" rx="2"/><rect class="s0" x="303" y="45" width="16" height="14" rx="2"/><rect class="s0" x="320" y="45" width="16" height="14" rx="2"/><rect class="s0" x="337" y="45" width="16" height="14" rx="2"/><rect class="s2" x="354" y="45" width="16" height="14" rx="2"/><rect class="s3" x="371" y="45" width="16" height="14" rx="2"/><rect class="s3" x="388" y="45" width="16" height="14" rx="2"/><rect class="s1" x="405" y="45" width="16" height="14" rx="2"/><rect class="s3" x="422" y="45" width="16" height="14" rx="2"/><rect class="s3" x="439" y="45" width="16" height="14" rx="2"/><rect class="s0" x="456" y="45" width="16" height="14" rx="2"/><rect class="s3" x="473" y="45" width="16" height="14" rx="2"/><rect class="s1" x="490" y="45" width="16" height="14" rx="2"/><rect class="s1" x="507" y="45" width="16" height="14" rx="2"/><rect class="s2" x="524" y="45" width="16" height="14" rx="2"/><rect class="s3" x="541" y="45" width="16" height="14" rx="2"/><rect class="s2" x="558" y="45" width="16" height="14" rx="2"/><rect class="s3" x="575" y="45" width="16" height="14" rx="2"/>
    <rect class="s0" x="48" y="60" width="16" height="14" rx="2"/><rect class="s0" x="65" y="60" width="16" height="14" rx="2"/><rect class="s1" x="82" y="60" width="16" height="14" rx="2"/><rect class="s2" x="99" y="60" width="16" height="14" rx="2"/><rect class="s0" x="116" y="60" width="16" height="14" rx="2"/><rect class="s3" x="133" y="60" width="16" height="14" rx="2"/><rect class="s1" x="150" y="60" width="16" height="14" rx="2"/><rect class="s2" x="167" y="60" width="16" height="14" rx="2"/><rect class="s1" x="184" y="60" width="16" height="14" rx="2"/><rect class="s3" x="201" y="60" width="16" height="14" rx="2"/><rect class="s1" x="218" y="60" width="16" height="14" rx="2"/><rect class="s3" x="235" y="60" width="16" height="14" rx="2"/><rect class="s3" x="252" y="60" width="16" height="14" rx="2"/><rect class="s1" x="269" y="60" width="16" height="14" rx="2"/><rect class="s3" x="286" y="60" width="16" height="14" rx="2"/><rect class="s3" x="303" y="60" width="16" height="14" rx="2"/><rect class="s3" x="320" y="60" width="16" height="14" rx="2"/><rect class="s0" x="337" y="60" width="16" height="14" rx="2"/><rect class="s2" x="354" y="60" width="16" height="14" rx="2"/><rect class="s1" x="371" y="60" width="16" height="14" rx="2"/><rect class="s1" x="388" y="60" width="16" height="14" rx="2"/><rect class="s0" x="405" y="60" width="16" height="14" rx="2"/><rect class="s3" x="422" y="60" width="16" height="14" rx="2"/><rect class="s0" x="439" y="60" width="16" height="14" rx="2"/><rect class="s3" x="456" y="60" width="16" height="14" rx="2"/><rect class="s1" x="473" y="60" width="16" height="14" rx="2"/><rect class="s1" x="490" y="60" width="16" height="14" rx="2"/><rect class="s0" x="507" y="60" width="16" height="14" rx="2"/><rect class="s2" x="524" y="60" width="16" height="14" rx="2"/><rect class="s0" x="541" y="60" width="16" height="14" rx="2"/><rect class="s3" x="558" y="60" width="16" height="14" rx="2"/><rect class="s0" x="575" y="60" width="16" height="14" rx="2"/>
    <rect class="s1" x="48" y="75" width="16" height="14" rx="2"/><rect class="s2" x="65" y="75" width="16" height="14" rx="2"/><rect class="s2" x="82" y="75" width="16" height="14" rx="2"/><rect class="s0" x="99" y="75" width="16" height="14" rx="2"/><rect class="s3" x="116" y="75" width="16" height="14" rx="2"/><rect class="s3" x="133" y="75" width="16" height="14" rx="2"/><rect class="s2" x="150" y="75" width="16" height="14" rx="2"/><rect class="s2" x="167" y="75" width="16" height="14" rx="2"/><rect class="s1" x="184" y="75" width="16" height="14" rx="2"/><rect class="s2" x="201" y="75" width="16" height="14" rx="2"/><rect class="s0" x="218" y="75" width="16" height="14" rx="2"/><rect class="s2" x="235" y="75" width="16" height="14" rx="2"/><rect class="s1" x="252" y="75" width="16" height="14" rx="2"/><rect class="s3" x="269" y="75" width="16" height="14" rx="2"/><rect class="s2" x="286" y="75" width="16" height="14" rx="2"/><rect class="s2" x="303" y="75" width="16" height="14" rx="2"/><rect class="s3" x="320" y="75" width="16" height="14" rx="2"/><rect class="s3" x="337" y="75" width="16" height="14" rx="2"/><rect class="s0" x="354" y="75" width="16" height="14" rx="2"/><rect class="s2" x="371" y="75" width="16" height="14" rx="2"/><rect class="s1" x="388" y="75" width="16" height="14" rx="2"/><rect class="s0" x="405" y="75" width="16" height="14" rx="2"/><rect class="s1" x="422" y="75" width="16" height="14" rx="2"/><rect class="s0" x="439" y="75" width="16" height="14" rx="2"/><rect class="s0" x="456" y="75" width="16" height="14" rx="2"/><rect class="s1" x="473" y="75" width="16" height="14" rx="2"/><rect class="s2" x="490" y="75" width="16" height="14" rx="2"/><rect class="s0" x="507" y="75" width="16" height="14" rx="2"/><rect class="s2" x="524" y="75" width="16" height="14" rx="2"/><rect class="s2" x="541" y="75" width="16" height="14" rx="2"/><rect class="s3" x="558" y="75" width="16" height="14" rx="2"/><rect class="s1" x="575" y="75" width="16" height="14" rx="2"/>
  <text class="t-lbl" x="48" y="114">V · head 4 — 128 channels</text>
  <text class="t-mono hd" x="42" y="135" text-anchor="end">0</text>
  <text class="t-mono hd" x="42" y="150" text-anchor="end">32</text>
  <text class="t-mono hd" x="42" y="165" text-anchor="end">64</text>
  <text class="t-mono hd" x="42" y="180" text-anchor="end">96</text>
    <rect class="s0" x="48" y="124" width="16" height="14" rx="2"/><rect class="s0" x="65" y="124" width="16" height="14" rx="2"/><rect class="s1" x="82" y="124" width="16" height="14" rx="2"/><rect class="s1" x="99" y="124" width="16" height="14" rx="2"/><rect class="s0" x="116" y="124" width="16" height="14" rx="2"/><rect class="s1" x="133" y="124" width="16" height="14" rx="2"/><rect class="s3" x="150" y="124" width="16" height="14" rx="2"/><rect class="s2" x="167" y="124" width="16" height="14" rx="2"/><rect class="s0" x="184" y="124" width="16" height="14" rx="2"/><rect class="s3" x="201" y="124" width="16" height="14" rx="2"/><rect class="s2" x="218" y="124" width="16" height="14" rx="2"/><rect class="s0" x="235" y="124" width="16" height="14" rx="2"/><rect class="s1" x="252" y="124" width="16" height="14" rx="2"/><rect class="s3" x="269" y="124" width="16" height="14" rx="2"/><rect class="s3" x="286" y="124" width="16" height="14" rx="2"/><rect class="s2" x="303" y="124" width="16" height="14" rx="2"/><rect class="s3" x="320" y="124" width="16" height="14" rx="2"/><rect class="s2" x="337" y="124" width="16" height="14" rx="2"/><rect class="s2" x="354" y="124" width="16" height="14" rx="2"/><rect class="s1" x="371" y="124" width="16" height="14" rx="2"/><rect class="s3" x="388" y="124" width="16" height="14" rx="2"/><rect class="s1" x="405" y="124" width="16" height="14" rx="2"/><rect class="s2" x="422" y="124" width="16" height="14" rx="2"/><rect class="s0" x="439" y="124" width="16" height="14" rx="2"/><rect class="s1" x="456" y="124" width="16" height="14" rx="2"/><rect class="s0" x="473" y="124" width="16" height="14" rx="2"/><rect class="s3" x="490" y="124" width="16" height="14" rx="2"/><rect class="s2" x="507" y="124" width="16" height="14" rx="2"/><rect class="s3" x="524" y="124" width="16" height="14" rx="2"/><rect class="s2" x="541" y="124" width="16" height="14" rx="2"/><rect class="s0" x="558" y="124" width="16" height="14" rx="2"/><rect class="s3" x="575" y="124" width="16" height="14" rx="2"/>
    <rect class="s1" x="48" y="139" width="16" height="14" rx="2"/><rect class="s0" x="65" y="139" width="16" height="14" rx="2"/><rect class="s1" x="82" y="139" width="16" height="14" rx="2"/><rect class="s2" x="99" y="139" width="16" height="14" rx="2"/><rect class="s1" x="116" y="139" width="16" height="14" rx="2"/><rect class="s1" x="133" y="139" width="16" height="14" rx="2"/><rect class="s3" x="150" y="139" width="16" height="14" rx="2"/><rect class="s0" x="167" y="139" width="16" height="14" rx="2"/><rect class="s2" x="184" y="139" width="16" height="14" rx="2"/><rect class="s3" x="201" y="139" width="16" height="14" rx="2"/><rect class="s0" x="218" y="139" width="16" height="14" rx="2"/><rect class="s2" x="235" y="139" width="16" height="14" rx="2"/><rect class="s2" x="252" y="139" width="16" height="14" rx="2"/><rect class="s3" x="269" y="139" width="16" height="14" rx="2"/><rect class="s3" x="286" y="139" width="16" height="14" rx="2"/><rect class="s1" x="303" y="139" width="16" height="14" rx="2"/><rect class="s3" x="320" y="139" width="16" height="14" rx="2"/><rect class="s1" x="337" y="139" width="16" height="14" rx="2"/><rect class="s0" x="354" y="139" width="16" height="14" rx="2"/><rect class="s2" x="371" y="139" width="16" height="14" rx="2"/><rect class="s0" x="388" y="139" width="16" height="14" rx="2"/><rect class="s1" x="405" y="139" width="16" height="14" rx="2"/><rect class="s1" x="422" y="139" width="16" height="14" rx="2"/><rect class="s2" x="439" y="139" width="16" height="14" rx="2"/><rect class="s2" x="456" y="139" width="16" height="14" rx="2"/><rect class="s0" x="473" y="139" width="16" height="14" rx="2"/><rect class="s3" x="490" y="139" width="16" height="14" rx="2"/><rect class="s3" x="507" y="139" width="16" height="14" rx="2"/><rect class="s3" x="524" y="139" width="16" height="14" rx="2"/><rect class="s0" x="541" y="139" width="16" height="14" rx="2"/><rect class="s1" x="558" y="139" width="16" height="14" rx="2"/><rect class="s0" x="575" y="139" width="16" height="14" rx="2"/>
    <rect class="s0" x="48" y="154" width="16" height="14" rx="2"/><rect class="s0" x="65" y="154" width="16" height="14" rx="2"/><rect class="s0" x="82" y="154" width="16" height="14" rx="2"/><rect class="s0" x="99" y="154" width="16" height="14" rx="2"/><rect class="s0" x="116" y="154" width="16" height="14" rx="2"/><rect class="s1" x="133" y="154" width="16" height="14" rx="2"/><rect class="s3" x="150" y="154" width="16" height="14" rx="2"/><rect class="s0" x="167" y="154" width="16" height="14" rx="2"/><rect class="s0" x="184" y="154" width="16" height="14" rx="2"/><rect class="s2" x="201" y="154" width="16" height="14" rx="2"/><rect class="s1" x="218" y="154" width="16" height="14" rx="2"/><rect class="s0" x="235" y="154" width="16" height="14" rx="2"/><rect class="s0" x="252" y="154" width="16" height="14" rx="2"/><rect class="s3" x="269" y="154" width="16" height="14" rx="2"/><rect class="s3" x="286" y="154" width="16" height="14" rx="2"/><rect class="s0" x="303" y="154" width="16" height="14" rx="2"/><rect class="s2" x="320" y="154" width="16" height="14" rx="2"/><rect class="s1" x="337" y="154" width="16" height="14" rx="2"/><rect class="s3" x="354" y="154" width="16" height="14" rx="2"/><rect class="s1" x="371" y="154" width="16" height="14" rx="2"/><rect class="s3" x="388" y="154" width="16" height="14" rx="2"/><rect class="s0" x="405" y="154" width="16" height="14" rx="2"/><rect class="s1" x="422" y="154" width="16" height="14" rx="2"/><rect class="s0" x="439" y="154" width="16" height="14" rx="2"/><rect class="s0" x="456" y="154" width="16" height="14" rx="2"/><rect class="s0" x="473" y="154" width="16" height="14" rx="2"/><rect class="s2" x="490" y="154" width="16" height="14" rx="2"/><rect class="s3" x="507" y="154" width="16" height="14" rx="2"/><rect class="s3" x="524" y="154" width="16" height="14" rx="2"/><rect class="s0" x="541" y="154" width="16" height="14" rx="2"/><rect class="s1" x="558" y="154" width="16" height="14" rx="2"/><rect class="s3" x="575" y="154" width="16" height="14" rx="2"/>
    <rect class="s2" x="48" y="169" width="16" height="14" rx="2"/><rect class="s0" x="65" y="169" width="16" height="14" rx="2"/><rect class="s2" x="82" y="169" width="16" height="14" rx="2"/><rect class="s1" x="99" y="169" width="16" height="14" rx="2"/><rect class="s1" x="116" y="169" width="16" height="14" rx="2"/><rect class="s2" x="133" y="169" width="16" height="14" rx="2"/><rect class="s3" x="150" y="169" width="16" height="14" rx="2"/><rect class="s1" x="167" y="169" width="16" height="14" rx="2"/><rect class="s1" x="184" y="169" width="16" height="14" rx="2"/><rect class="s3" x="201" y="169" width="16" height="14" rx="2"/><rect class="s2" x="218" y="169" width="16" height="14" rx="2"/><rect class="s1" x="235" y="169" width="16" height="14" rx="2"/><rect class="s1" x="252" y="169" width="16" height="14" rx="2"/><rect class="s1" x="269" y="169" width="16" height="14" rx="2"/><rect class="s3" x="286" y="169" width="16" height="14" rx="2"/><rect class="s1" x="303" y="169" width="16" height="14" rx="2"/><rect class="s3" x="320" y="169" width="16" height="14" rx="2"/><rect class="s2" x="337" y="169" width="16" height="14" rx="2"/><rect class="s2" x="354" y="169" width="16" height="14" rx="2"/><rect class="s2" x="371" y="169" width="16" height="14" rx="2"/><rect class="s2" x="388" y="169" width="16" height="14" rx="2"/><rect class="s1" x="405" y="169" width="16" height="14" rx="2"/><rect class="s2" x="422" y="169" width="16" height="14" rx="2"/><rect class="s1" x="439" y="169" width="16" height="14" rx="2"/><rect class="s2" x="456" y="169" width="16" height="14" rx="2"/><rect class="s2" x="473" y="169" width="16" height="14" rx="2"/><rect class="s3" x="490" y="169" width="16" height="14" rx="2"/><rect class="s3" x="507" y="169" width="16" height="14" rx="2"/><rect class="s3" x="524" y="169" width="16" height="14" rx="2"/><rect class="s2" x="541" y="169" width="16" height="14" rx="2"/><rect class="s2" x="558" y="169" width="16" height="14" rx="2"/><rect class="s2" x="575" y="169" width="16" height="14" rx="2"/>
  <path class="ax thin" d="M48 196 H591"/>
  <g class="legend">
    <rect class="s0" x="48"  y="208" width="14" height="14" rx="3"/><text class="t-mono" x="68"  y="220">slot 0</text>
    <rect class="s1" x="150" y="208" width="14" height="14" rx="3"/><text class="t-mono" x="170" y="220">slot 1</text>
    <rect class="s2" x="252" y="208" width="14" height="14" rx="3"/><text class="t-mono" x="272" y="220">slot 2</text>
    <rect class="s3" x="354" y="208" width="14" height="14" rx="3"/><text class="t-mono" x="374" y="220">slot 3 — the conservative one</text>
  </g>
</svg>
<figcaption>Every one of the 128 head-dimension channels, in index order, coloured
by the palette slot it was assigned. Each slot holds exactly 32 channels by
construction. There is no banding on either side — and the two sides disagree
about which channels are difficult.</figcaption>
</figure>

Nothing about that is smooth. The 32 channels that land in the conservative slot
are **scattered across the whole dimension**, not clustered at one end, and the K
head and the V head of the same chunk disagree about which channels those are.
Difficulty is a property of the individual channel's activation distribution, and
it does not track index, neighbourhood, or token position.

Which is why the search has to be a search. There is no closed form to fit here,
no "boost the top 12.5% of channels" heuristic that recovers this map — that is
precisely the offline-calibrated approximation Kitty makes, and it is why it
needs calibration data to make it. It also kills the obvious shortcut on the
other axis: "just protect the first N tokens" would miss most of these, because
the conservative blocks aren't where position would put them either.

## The attention sinks, without a magic number

Some tokens receive wildly disproportionate attention. Quantize their V loosely
and output degrades badly, because their contribution dominates the weighted
sum.

The usual fix is "protect the first four tokens" — a constant that happens to be
right on the models people tested, and misses emergent mid-sequence sinks
entirely.

Instead: for each of the 32 tokens in a chunk, score `q_mean · k / √d`, a
pre-softmax proxy for attention this token is about to receive. Z-score against
the chunk's own statistics, pass through `tanh`, and take the maximum. That one
number linearly interpolates the V threshold between a lenient and a strict
bound — so a single strong sink anywhere in the chunk pulls **every** V block in
it toward strict.

No fixed threshold, no calibration, no per-model tuning, and registration sinks
and emergent sinks are caught by the same three lines because both produce an
above-average alignment score. The chunk's own distribution is the only
reference.

## Two error metrics, because there are two error paths

K errors go through the softmax, where a single outlier element can dominate a
score. V errors are combined *linearly* by the attention weights, so the output
budget is exactly L2. Using one metric for both is measuring the wrong thing
half the time.

So: mean-of-top-4 weighted absolute error for K, plain mean-squared error for V,
both normalised by the head's absolute maximum so one dimensionless threshold
works across every head, layer and model.

<div class="key">
<h4>Cosine distance will betray you</h4>
<p>We tried it for K. It treats a direction-preserving, magnitude-collapsing
approximation as low error — and under concurrent load that means one session's
query gets routed into another session's cache.</p>
<p>In testing, a session assigned the character "Marcus" leaked into a
<em>different</em> session, which happily started calling its own character
Marcus. We named the failure mode after him. Magnitude is not optional.</p>
</div>

Plain magnitude-weighted error fixes Marcus but over-rejects: averaging across
all 32 elements ignores that K error propagates through 1–4 dominant elements,
so a benign block with one moderate error gets penalised like a malicious one
with an extreme error. Top-4 keeps the magnitude and concentrates on the
elements that actually matter.

## In the kernel

One fused CUDA kernel per (chunk, head), 128 threads as four warps, five phases:
load and compute per-block magnitudes; detect sinks; **bitonic-sort 128 blocks
by magnitude**; compute per-block thresholds from the sink weight; then the
iterative slot search.

Anything still unclaimed after four slots falls back to FP16 passthrough — rare,
and it bounds worst-case error per (chunk, head) to full precision rather than
to whatever the last slot happened to pick.

Total shared memory is about **12.7 KB**, which lets 5–8 (chunk, head) pairs run
concurrently per SM on Ada. The whole selection mechanism costs **under 1%** of
wall-clock against the paged-attention kernel it feeds.

## Paying for the bookkeeping

A compression number that excludes its own metadata is one you shouldn't trust.

Per (chunk, head), per side: four format tags, four FP16 outer scales, 128
two-bit indices. **44 bytes** — 88 for K and V together, or 0.086 bits per
element across the cache. Every ratio here includes it. About 2.7% of the
achievable ratio at the aggressive end; rounding error at the conservative end.

There's also a floor nobody can optimise away: the in-flight chunk sits at FP16
until it reaches 32 tokens and gets committed, so **5% of the cache is always
uncompressed**. That's geometric — one chunk in twenty — and fixed by chunk
granularity, not a knob.

Is four slots enough? Against two reference points:

| | vs palette-4 |
|---|---|
| Per-block ideal — all 128 choose independently | palette-4 is within **1–2.9%** |
| One format per head — no palette at all | palette-4 wins by **7.6–56%** |

Four captures nearly all the available adaptation for 44 bytes. Finer buys under
3% and costs a selector per block; coarser gives back half.

## The level ladder does something counterintuitive

Eleven operating points, C0 to C10. Each defines which formats are admitted to
the candidate list, plus strict and lenient error thresholds.

And the thresholds are **not monotone**. Going from C7 to C8 — a *more*
aggressive level — the K-strict threshold gets *tighter*.

That looks like a bug and isn't. C8 admits Q0_V to the K candidate list, so the
level number and the threshold are controlling different things: the level says
*which* aggressive formats exist in the set, and the strict threshold says
*which blocks qualify* for the most aggressive one. Admitting a sub-1-bit format
while tightening admission to it means "this is now available, but selectively"
— and the bulk of blocks fall through to the next tier instead.

## Where it's actually free

At the conservative end this isn't a trade at all.

| Configuration | BPE | PPL (Qwen3-8B) |
|---|---:|---:|
| F16 KV | 16.00 | 9.88 |
| Q8/Q8 | 8.50 | 9.88 |
| Q8/Q4 — the llama.cpp default | 6.50 | 9.94 |
| **PalQuant C1** | **6.30** | **9.90** |

C1 **strictly dominates** the standard asymmetric recommendation — better
perplexity at fewer bits, both axes, no trade. And it's within noise of Q8/Q8 at
26% fewer bits. On two of the four models tested it comes in *below* F16, which
is quantization-as-regularisation and not worth over-reading, but it certainly
isn't a loss.

## The bit nobody wants to hear about perplexity

Right. Time to say the quiet part.

At matched bitrate in the mid range, plain uniform Q4 gets **better** perplexity
than this scheme. On one model, by 11%.

I'm not going to bury that in an appendix. It's true, it's reproducible, and if
perplexity is your metric then uniform Q4 beats me.

It also **fails** the test that actually matters — the multi-session one, where
each concurrent session has to keep its own character straight — at 3.56×
compression. Per-block selection passes at every level right up to 7.4×.

So which of those two numbers do you care about? Because you have to pick, and the
field has picked wrong.

<div class="key">
<h4>Why both are true</h4>
<p>Perplexity averages over everything. A small fat tail of badly-approximated
blocks vanishes into a corpus average. But one mis-quantized K block in a sink
position routes a query into the wrong session's memory — total failure, costing
almost nothing in mean log-likelihood.</p>
<p>Adaptive routing sends easy blocks to aggressive slots, so some of them pay a
small perplexity cost that uniform Q4 — spending four bits everywhere — simply
doesn't. You lose on the average and win on the tail. The average is what gets
published.</p>
</div>

And it gets worse, in a way that took a context sweep to spot.

Everyone reports perplexity at 2048 tokens. Everyone. It's the convention. And at
2048, *every single configuration* we measured sits within 0.1 perplexity of FP16
— 8-bit, 4-bit, adaptive, the lot of them. At 4096 those same configurations cost
between 0.5 and 6.

And here's the mechanism. Full-precision KV scores **9.88 at 2048 but 9.14 at
4096**. Read that twice. The shorter context isn't *easier* — the weights are
Q4_K_M, quantized against 2048-context activation statistics, so 2048 is precisely
where the *weight* stack sits at its tuned operating point and swamps everything
else in the measurement.

So the standard KV cache benchmark is being run at exactly the one context length
where KV cache format cannot be distinguished at all.

The convention long predates any of the systems using it, and I used it myself
until the sweep showed me what it was measuring. But it is measuring the weights,
and every number reported under it — mine included — has to be read that way.

## A third opinion: RULER

Perplexity says one thing and the multi-session test says another, so a third
axis helps. RULER on Qwen3-8B at 4K — needle-in-haystack retrieval, and variable
tracking for multi-step reasoning:

| Configuration | BPE | CR | Retrieval | Reasoning |
|---|---:|---:|---:|---:|
| F16 | 16.00 | 1.00× | 95% | 100% |
| Q8/Q4 | 6.50 | 2.46× | 87% | 83% |
| **PalQuant C5** | 4.35 | 3.68× | **98%** | 86% |
| PalQuant C9 | 2.97 | 5.39× | 90% | 75% |
| PalQuant C10 | 2.16 | 7.41× | 80% | 71% |

C5 matches full precision on retrieval at 3.68× compression. C9 holds 90% at
5.39×.

But look at the reasoning column, because this is the honest part. **Q8/Q4 keeps
an 8-point advantage over C9 on multi-step reasoning** — at 2.2× the bits, which
is a bad deal, but it is a real one. Adaptive selection wins on retrieval at
every bitrate. Below about 3 bits per element, fixed higher-precision
quantization stays competitive on reasoning. If your workload is chains of
inference rather than recall, that gap is worth knowing about.

## What the failures actually look like

The most useful thing in the whole evaluation was reading the wrong answers.

Both full-precision and maximum-compression retrieval failures share one
pattern: the model finds the right region of context and **drops the last digit
of the answer**, then loops on the truncated form.

```
expected 99211 47977   →  predicted "9921 47977"
expected 50265 12971   →  predicted "5026 5026 5026 12971"
expected 34482         →  predicted "3442"
```

The first two are **F16**. The third is the most aggressive compression level.
On one matched prompt, F16 and C10 fail *identically* — both truncate 47088 to
4708 and loop.

That reframes the whole result. This is a Qwen3-8B output-formatting behaviour,
present at full precision, which aggressive compression **amplifies rather than
introduces**. A small subset of C10 failures are genuinely new — collapsed
answers, token degeneration into `ffffffff...` — but the bulk of the "damage"
was already there.

If you only had the pass rates you'd conclude compression broke retrieval. It
didn't. It made an existing crack wider.

## It doesn't care what shape the heads are

Here's the test that would break a scheme fitted to one geometry.

Qwen3-30B-A3B, Qwen3-8B and Llama-3.2-3B all run grouped-query attention at head
dimension 128 with eight KV heads. The Qwen3.5 family runs its attention layers at
**head dimension 256**, with two or four KV heads. Twice the dimension, a quarter
the head count — a different shape in both directions at once.

Same selection, no refitting. Compression ratios hold, the compression costs no
measurable throughput, and quality holds where it matters.

And that's structural rather than lucky. **The selection works on 32-element
blocks, and a block is a block.** Head dimension only decides how many bands cover
one head: 128 gives four, 256 gives eight, and the latent geometry on
DeepSeek-V4-Flash runs sixteen. Nothing underneath knows or cares which of those
it's sitting in.

Which is this post's whole argument arriving from a direction I wasn't looking in.
**An architecture that selects per block absorbs a change in geometry the same way
it absorbs a weak primitive** — by measuring what's in front of it instead of
trusting a prior that was fitted somewhere else. A scheme tuned to head dim 128
needs a new tuning pass when the shape moves. This one needs a different band
count.

## Where this is weakest

Two things, and I'd rather say them than have somebody find them.

**The "calibration-free" claim has an asterisk.** Llama runs with identity scaling
factors — genuinely nothing tuned, nothing fitted, nothing. The two Qwen models
use four dimensionless factors picked by sweeping a small grid against the
multi-session test on held-out passages. That is not calibration data in the
KVQuant sense, and a new model can absolutely deploy at identity. But it does mean
the headline 7.42× and the 5.02× aren't quite measured the same way, so: identity
works everywhere, and four numbers buy you a better point on your own ladder.

**Multi-step reasoning in the sub-3-bit band.** The RULER numbers above are what
they are: retrieval holds all the way down, multi-step reasoning at C9 doesn't
hold as well as spending more bits would. That's a real trade at a real operating
point — and it is a *choice*, not a defect. Nothing makes you run at C9. The
ladder exists precisely so you can sit at C5, take 3.7×, and match full precision
on retrieval while you're there.

## One mechanism instead of four

Here's what makes the architecture argument rather than an engineering one.

The prior systems each had to build something separately:

- KVQuant needed **outlier protection** — from sensitivity profiling
- KIVI needed **K/V awareness** — baked into the format design
- Kitty needed **mixed precision** — from offline magnitude ranking

Per-block selection produces all three as *consequences*. Outlier protection
falls out of the error metric, because a spiky block fails the threshold on
aggressive formats by itself. K/V asymmetry falls out of having two metrics for
two error paths. Mixed precision falls out of the palette slots. None of them
was designed in.

<div class="key">
<h4>The distinction worth stealing</h4>
<p>Everyone else is optimising the compression <b>primitive</b>. The lever is
the compression <b>architecture</b> — <em>when</em> to quantize, what to
guarantee per block, how to keep error sources separate.</p>
<p>An architecture that selects adaptively absorbs a weak primitive. It doesn't
need the best one. That's why per-block selection holds quality at ratios where
population-level schemes have started to slide — and why the two compose:
PatternKV's activation flattening or transform coding would give the selector a
better codebook to choose from, not compete with it.</p>
</div>

## Measure the failure, not the average

Which leaves one loose end, and this one is the field's rather than mine.

Nobody can agree on how to measure any of this. KIVI, KVQuant and TurboQuant
report perplexity. Kitty rejects perplexity outright and uses task accuracy.

And both camps miss the failure that actually breaks a deployment — one
mis-quantized block routing a query into the wrong session's memory — because it
costs nothing in a corpus average *and* nothing on a benchmark where sessions
never run concurrently. Meanwhile the conventional benchmark gets run at the one
context length where weight quantization dominates and nothing can be told apart
at all.

So here's the line I'd like people to take away from this whole thing.

**If your quality metric can't distinguish the thing that breaks you from the
thing that doesn't, it isn't a quality metric. It's a habit.**

Find the failure mode first. Then, and only then, pick what to measure.

## What the compression is actually for

None of this was ever about ratios.

The KV cache is the only thing in an inference stack that grows without bound.
Every property you'd want from a system that *remembers* — a coding assistant
that still knows the project next month, a character who remembers what you did
to them — reduces to the same question underneath: how much cache can you afford
to keep, and for how many people at once.

That question used to have a bad answer. On one 16 GB laptop GPU, full
precision runs out of memory before it can hold a useful number of sessions at
all. With per-block selection, the same card holds **256 concurrent
Llama-3.2-3B sessions**, 168K tokens of cache between them, 887 tokens/second
aggregate — or 120 sessions of a 30-billion-parameter mixture-of-experts. F16
reaches none of those numbers.

So the figure worth caring about isn't 7.42×. It's **256** — the gap between an
inference engine and something that can hold a few hundred people's histories
simultaneously, on hardware one person owns outright. Compression stopped being
a tax you pay in quality and became the thing that makes the memory affordable
in the first place.

Every factor of two is another hundred people who get to be remembered. That's
worth more than a benchmark table.

[Read the paper →](/papers/palquant)
