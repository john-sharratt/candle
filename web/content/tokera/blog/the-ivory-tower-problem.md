---
title: "The ivory tower problem"
date: 2023-08-16
tint: accent
tags: [architecture, teams, careers]
summary: >-
  Software architecture is thriving and diluting at the same time. The cause is
  stereotype and projection — and if you follow the smoke there is a real fire
  at the source. Four things that fix it.
---

<figure class="shot hero">
<img src="/img/blog/ivory-tower.jpg" alt="A tall stone tower rising above mist, a beam of light breaking from its peak" width="800" height="800">
</figure>

The legacy of ivory tower architecture hangs over our profession brighter than
ever. What causes this stigma, and how do we overcome it?

This may be a tough post for some to swallow, but it's long overdue, so let's get
stuck in.

Software architecture is **both thriving and slowly diluting away**.

It thrives, as the art of software design remains needed more than ever, with huge
and increasingly complex software systems literally running our lives. Tech
continues to advance at a fast pace. But the profession of architecture slowly
degrades, because the engineers that code those systems — at least in fintech —
grow ever more distant from the advice of the ultimate guardians of the
non-functionals. Eventually that results in systems that don't scale, and
operations that are expensive and not robust.

Why? Simply this.

**Stereotypes. Projection.** And of course, if you follow the smoke there is a real
fire at the source.

## The stereotyping runs both ways

On the one hand, some architects — but definitely not me — believe that engineers
themselves are not experienced or knowledgeable enough to design scalable and
robust complex systems.

While on the other end of the spectrum, engineers see architects as out-of-touch
dictators preaching non-pragmatic solutions that are disconnected from reality, or
using out-of-date ideas.

Projection then sets in after one experiences these stereotypes, which trains the
brain to project that onto both respective professions *generally*. This amplifies
the problem as communication lines erode and break down.

<figure class="fig">
<svg viewBox="0 0 640 212" role="img" aria-label="A loop: stereotype leads to projection, projection erodes communication, eroded communication produces worse designs, which reinforce the stereotype.">
  <rect class="box weights" x="60" y="34" width="150" height="42" rx="6"/>
  <text class="t-mono mid" x="135" y="60">stereotype</text>
  <rect class="box weights" x="430" y="34" width="150" height="42" rx="6"/>
  <text class="t-mono mid" x="505" y="60">projection</text>
  <rect class="box weights" x="430" y="130" width="150" height="42" rx="6"/>
  <text class="t-mono mid" x="505" y="150">communication</text>
  <text class="t-mono mid" x="505" y="164">breaks down</text>
  <rect class="box weights" x="60" y="130" width="150" height="42" rx="6"/>
  <text class="t-mono mid" x="135" y="150">designs get worse</text>
  <text class="t-mono mid" x="135" y="164">on both sides</text>
  <path class="ax" d="M210 55 H424"/>
  <path class="ax" d="M418 49 L430 55 L418 61"/>
  <path class="ax" d="M505 76 V124"/>
  <path class="ax" d="M499 118 L505 130 L511 118"/>
  <path class="ax" d="M430 151 H216"/>
  <path class="ax" d="M222 145 L210 151 L222 157"/>
  <path class="ax" d="M135 130 V82"/>
  <path class="ax" d="M129 88 L135 76 L141 88"/>
  <text class="t-dim mid" x="320" y="200">nobody in this loop is behaving irrationally, which is exactly why it never resolves</text>
</svg>
<figcaption>Every step here is a reasonable response to the one before it. That's
what makes the loop stable — and stable is the last thing you want it to
be.</figcaption>
</figure>

## Ultimately though, there is truth deep beneath the smoke

I have met architects that really don't know what they are talking about, and make
impractical, incomplete designs, and eventually hand it over to engineers who are
kind enough to pretend they are following it.

While I've also met engineers who play the risky game of giving huge estimates,
instead of helping the architect improve the design or at least give it a chance.
They gamble a wild estimate and hope that they'll never get caught out.

Both of those are real, both are common, and pretending otherwise is why this
conversation normally goes precisely nowhere. The stereotype isn't baseless. It's
an accurate observation about a minority, generalised onto everybody.

### And there's a structural reason this happened

Let me add something I didn't say at the time, because I think it's the actual
root.

The separate architect role is a **fossil of waterfall**. It exists because there
was once a phase — a real, scheduled, months-long phase — in which design happened
and code did not. Somebody had to own that phase, and that somebody became an
architect.

Then we deleted the phase. We kept the role.

So the role now has to justify itself continuously in an environment that no
longer has a slot for it, and the two failure modes we're all familiar with are
exactly what you'd predict from that: either the architect produces artefacts
nobody asked for in order to demonstrate value (the ivory tower), or the architect
becomes a reviewer who signs things and slows everybody down (the rubber stamp).

Notice also who *doesn't* have this problem. The organisations most people
consider best at large-scale system design mostly don't run a separate
architecture function at all — the senior engineers do the design, and it happens
continuously rather than up front. That isn't because design stopped mattering.
It's because design got given back to the people holding the compiler.

## So quite a problem then it would seem. But what do we all do about it?

It's time to face the reality and make some changes.

<div class="key">
<h4>1. Architects must continue — or start — to code, or at least script</h4>
<p>If you can't understand coding then you can't call the estimate bluff. But most
importantly you will command a better level of mutual respect from engineers when
you speak their language, and it will break the stigma of the ivory tower.</p>
<p>Nothing else does this. Not a better diagram, not a clearer standard, not a
certification. Being demonstrably able to do the work.</p>
</div>

<div class="key">
<h4>2. Architects must stop aiming for design completeness</h4>
<p>Designs are wrong the moment you finish writing them. So just write enough that
engineers can start the work, then iterate it with them.</p>
<p>A complete design isn't a more rigorous design. It's a design that spent its
entire budget before making contact with the problem — and it arrives at the team
too finished to argue with, which is the worst possible property.</p>
</div>

<div class="key">
<h4>3. Senior engineers will become the next architects</h4>
<p>Nurture them, socialise with them, delegate design work to them. They are the
future, so let's give them the respect they deserve.</p>
<p>This is also the only durable fix for point 1. The pipeline into architecture
should start with people who were writing code last year — not a parallel career
track that never did.</p>
</div>

<div class="key">
<h4>4. Become a part of the open source world</h4>
<p>It's not that scary, and it has wonders going on in there that are the future.
Dive in.</p>
<p>It's also the only place an architect's decisions meet unfiltered feedback from
people who owe them absolutely nothing — no reporting line, no politics, no
politeness. That is the fastest correction mechanism our profession has, and most
architects have never once been exposed to it.</p>
</div>

Do those four and the tower comes down on its own. Nobody has to be told to stop
projecting; they just stop, because the thing they were projecting onto isn't
there any more.

---

## Postscript

Point 2 is the one I've come to hold most strongly, and I've since written it into
a codebase as a hard rule:

> **Design docs are authoritative.** When a design document exists for the work, it
> takes precedence over discrepancies with the code. **If the document is itself
> wrong, fix the document in the same change.**

That second sentence is point 2 made enforceable. It refuses the false choice
between "the design is sacred" — which is the ivory tower — and "the design is
decoration", which is what an ignored document actually is. The document leads,
*and* it is expected to be wrong, *and* correcting it is part of the work rather
than an admission that anybody failed.

It has caught real errors too, including one in a paper I'd written myself: a
section describing the wrong direction of an algorithm that its own appendix
contradicted. The code was right, the document was wrong, and the rule meant the
document got fixed rather than quietly diverging for another year until somebody
built on it.

As for point 1 — I've spent the last few years writing CUDA kernels.

The estimate bluff is a great deal harder to call from a slide deck.
