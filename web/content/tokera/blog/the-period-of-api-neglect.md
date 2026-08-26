---
title: "The period of API neglect"
date: 2023-05-17
tint: crit
tags: [architecture, microservices, industry]
summary: >-
  31 million engineers. Somewhere north of 46 million APIs. Every one of them
  has to be secured, patched, refactored and deployed, forever. That ratio only
  moves one way and no amount of AI changes the arithmetic.
---

<figure class="shot hero">
<img src="/img/blog/api-neglect.jpg" alt="A rusted car abandoned in a barn with its bonnet propped open" width="800" height="800">
</figure>

Do you ever get the feeling that engineering is getting overwhelmed?

Well, it would seem the reason for that feeling is likely because... in fact we
are.

- There are **31 million engineers** in the world. That's it. That is the entire
  capacity of earth to write code.
- Estimates for the number of APIs in the world vary from **46 million** to
  **200 million**.

Now, as we digitalise the world at an exponential rate, the complexity of it all
is outstripping our ability to maintain it all. No wonder tech jobs are in such
high demand — the rest of the world is dependent on this subset of society to
maintain the whole thing.

Throw microservices into the mix and it creates a multiplier effect in both the
volume and the complexity, as many of these APIs are in fact complex HTTP servers
under the hood.

## The bill on every single one

Each one of these APIs has to be:

- made to be secure
- made to be reliable
- constantly refactored and patched
- enhanced with new features
- built somewhere
- deployed by something or someone

All of that with a ratio that is approaching **5:1 APIs to engineers**, and it
will only get worse over the next few years as the growth curve attempts to
escape earth's gravity.

<figure class="fig">
<svg viewBox="0 0 640 216" role="img" aria-label="Two bars: 31 million engineers against an API count between 46 and 200 million, with the gap widening.">
  <text class="t-lbl" x="60" y="46">engineers</text>
  <rect class="box weights" x="180" y="30" width="76" height="26" rx="4"/>
  <text class="t-mono" x="266" y="48">31M — and that is all of them</text>
  <text class="t-lbl" x="60" y="96">APIs (low estimate)</text>
  <rect class="box" x="180" y="80" width="112" height="26" rx="4"/>
  <text class="t-mono" x="302" y="98">46M</text>
  <text class="t-lbl" x="60" y="146">APIs (high estimate)</text>
  <rect class="box" x="180" y="130" width="400" height="26" rx="4"/>
  <text class="t-mono" x="590" y="148" text-anchor="end">200M</text>
  <text class="t-dim" x="60" y="190">the top bar is fixed by demographics. the bottom two are not.</text>
  <text class="t-dim" x="60" y="208">every unit of the lower bars needs securing, patching, refactoring and deploying — forever.</text>
</svg>
<figcaption>One of these quantities has a hard ceiling. The other one is
compounding. That's the whole problem in a single picture.</figcaption>
</figure>

## And it's worse than a ratio, because the costs aren't linear

Now here's where I'll push the argument a bit harder than I did on the day,
because the 5:1 figure actually *understates* it.

Maintenance cost doesn't scale with the count of your services. It scales with
the count of your **interactions**, and that is a square law. Ten services have
forty-five possible pairs. A hundred services have four thousand nine hundred and
fifty. You didn't multiply your problem by ten, you multiplied it by a hundred
and ten, and then you hired one extra person.

And every one of those APIs drags its own dependency tree behind it. When
[Log4Shell](https://nvd.nist.gov/vuln/detail/CVE-2021-44228) landed, the question
that broke organisations was not "can we patch it" — the
patch was trivial. The question was **"where is it?"** Companies spent weeks
answering that, and the ones with the most services took the longest, because the
blast radius of a single bad dependency is precisely the number of deployable
units that carry it.

Lehman worked this out back in the seventies and gave us laws for it: a system in
use must continuously change or become less useful, and as it changes its
complexity increases unless work is explicitly done to reduce it. Note that last
clause. Complexity reduction doesn't happen on its own — someone has to be paid to
do it, and nobody gets promoted for deleting a service.

<div class="key">
<h4>The bit that should frighten you</h4>
<p>Every API in that 46 million is a <b>perpetual liability</b>, not a one-off
build cost. It needs patching in 2025, in 2030, and on the day its last original
author leaves the company.</p>
<p>We have been counting the build. We have never once counted the maintenance,
and the maintenance is the part that compounds.</p>
</div>

## We've already spent all the delaying tactics

As an industry we've done a great deal to put this critical moment off until now
— massive code reuse through open source, SDKs, enormous automation efforts. All
of it genuinely worked, and all of it just **delayed** the moment rather than
preventing it.

Those delaying solutions don't change the fact that we are reaching the software
equivalent of **peak oil**. Not a cliff where everything stops. The point where
the cheap easy supply is behind us, and every additional barrel costs more than
the last one did.

And I'd note that our biggest delaying tactic has quietly become a liability of
its own. Open source let us stop writing the same thing 46 million times — but it
did so by concentrating the world's software onto a small number of maintainers
who are mostly unpaid, mostly unrecognised, and increasingly exhausted. That's not
a sustainability problem for them. It's a supply chain problem for you.

## Which brings me to the AI mania

Now, I've been trying to understand why the AI hype and the LLM enthusiasm is
reaching such mania on steroids, and I've come to think it's a **symptom**.

Engineers are reaching out for anything they can find to try and manage this
complexity, and hooking onto it like it's a miracle cure.

Sorry to burst your bubble folks, but many of us in the know have to break it to
you that while AI is a multiplier, it cannot do the jobs of those 31 million
engineers — and even if it could, it's already too late. Maybe in ten years I'll
look back and eat my hat. But there's a reason I see hype curves in my feed every
single day, and it still won't change the fact that all of them have a peak.

And here's the uncomfortable part nobody's costing properly. AI is very good at
*producing* code. Producing code was never the bottleneck. **Maintaining** code
was the bottleneck. If you use a multiplier on the cheap half of the problem, all
you have done is enlarge the expensive half — faster.

We're about to find out whether an industry that couldn't maintain 46 million
APIs can maintain 200 million.

## So then what is one to do?

**Rise-of-the-monolith**, baby.

It's time to collectively bring the number back to a sensible set by combining
things together again, and again, and again.

Contrary to how this may sound or be perceived, I am not advocating for a return
to the mainframe or the erosion of our massively HTTP-interconnected world. The
interconnection is good. The interconnection is the entire point.

All I am saying is this:

> If you have ten teams of engineers working on ten or more separate APIs, then
> it's well past the time you consider combining those teams and producing **a
> single binary that you deploy multiple times and that runs all those APIs
> together as one unit**.

And read that carefully, because it does *not* say fewer APIs. Your interfaces
stay exactly where they are. Your consumers notice nothing. What shrinks is the
number of **things to secure, patch, build and deploy** — because every one of
those costs attaches to the deployable unit, and none of them attach to the
interface.

Ten APIs in one binary is one pipeline, one dependency tree, one patch, one
rollback and one on-call rota. Ten APIs in ten binaries is ten of each, forever,
and you did not get anything for the other nine.

Either we start to leverage economies of scale, or **API neglect** will suck our
brains dry.

---

## Postscript

I took my own advice, at an extreme.

The inference engine I've spent the last few years building is one binary. Its web
front is [a single crate](/blog/one-card-unbounded-context) that is either the
authoritative server or a proxy, selected by one line of YAML per route — not two
codebases, not a service mesh, not an ingress controller with a lifecycle of its
own. The mock APIs live inside it, so the entire estate runs with no daemons at
all when you want it to.

The paper that came out of it is called **One Card, One Stack**, which is this
post's argument wearing a lab coat. The card is the hardware constraint. The stack
being *one* stack is the same claim as above: the cost lives in the number of
things you deploy and defend, so make that number small and spend everything you
save on the part that is genuinely hard.

Ten years isn't up yet, so the hat remains uneaten. But I'd note this — the way I
eventually used AI at scale was not as a replacement for those 31 million
engineers. It was to give **one** engineer a memory that doesn't run out.

That is a multiplier on the expensive half.
