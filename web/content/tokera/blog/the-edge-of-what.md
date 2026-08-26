---
title: "The Edge of What?"
date: 2023-10-24
tint: ok
tags: [edge, distributed-systems, architecture]
summary: >-
  Everyone talks about the mythical edge without ever defining it. So here is a
  definition with a hard boundary — the edge is your device plus thousands of
  small datacentres, and it can never, ever be the hyperscaler's.
---

<figure class="shot hero">
<img src="/img/blog/edge.jpg" alt="A city built along cliff tops, waterfalls falling away into the valley beneath it" width="800" height="800">
</figure>

When one talks of edge computing they refer to the mythical edge without defining
what that means. Read on then, and I'll share my view on what is Edge and what it
is not.

## Five computers to rule them all

Ultimately Edge is the counterbalancing reaction to Thomas Watson's prediction of
*"five computers to rule them all"* actually becoming true.

Of course I refer here to the top five cloud providers, who corner 73% of the
server hosting market — Amazon, Azure, Google, Alibaba and IBM. Maybe they don't
quite look like the computers we predicted in 1943, but make no mistake, they fit
the scope and outcome of said prediction.

Five computers to choose from. Isn't this just global capitalism sorting itself
out?

Not quite so fast there, fella. Maybe capitalism has hit an equilibrium, but
innovation and geopolitics certainly have not. If we constrain our thinking
pattern to the dominant design paradox then we are signing up for the same trouble
that held back electric cars for so long — a locally optimal design that is
murderously hard to leave, not because the alternative is worse but because the
incumbent has absorbed all the tooling, all the skills and all the supply chain.

And I'd add something I only half-said at the time, because it has become the
stronger half of the argument.

**Concentration is a correlated failure mode.** Five providers isn't five
independent systems, it's five single points of failure with most of the planet
downstream. Every time one region of one provider has a bad morning, a
statistically improbable proportion of the internet goes dark simultaneously —
banks, airlines, doorbells, hospital scheduling — and every one of those
organisations had a resilience strategy that said "multi-AZ". They were all in the
same building. They just had different desks.

You cannot buy your way out of that with an SLA. An SLA is a refund, not an
outcome.

## Distribution was a design requirement once

Nuclear armageddon is not your usual architecture design requirement, but least we
not forget that's what the original Internet was designed for back in 1983.

Paul Baran's
[work at RAND](https://www.rand.org/pubs/research_memoranda/RM3420.html) on
distributed communications was explicitly about
surviving a first strike, and it is the direct ancestor of packet switching. That
is the foresight I'm talking about — and without it we'd have some mighty
difficult engineering challenges holding us back today.

DNS is the proof it was real. A *distributed* database by design, where no central
server holds the answers, authority is delegated down a hierarchy, and every
resolver on earth takes part in the lookup. A deliberate choice made at a moment
when centralising it would have been far, far easier.

You see, the thing is, when you slice up those data centres and move the workloads
closer to their users, it removes a whole bunch of problems. And of course creates
a whole bunch of new ones.

### And one of the problems it removes cannot be bought

Here's the bit that makes Edge inevitable rather than merely fashionable.

**Latency has a floor, and the floor is physics.** Light in fibre does about 200
kilometres per millisecond, and real networks do considerably worse. Sydney to
Virginia is a round trip of roughly 200 milliseconds before a single line of your
code executes. No amount of money, no CPU generation, no clever protocol touches
that number. It is set by the speed of light and the shape of the planet.

Which means there exists a whole class of application that **cannot be built in a
mega-datacentre at all** — not "runs slowly", cannot be built. And every year we
invent more of them.

Then stack the second forcing function on top: **data sovereignty**. More
jurisdictions every year require that their citizens' data physically resides
within their borders. That isn't an engineering preference you can architect
around, it's a legal constraint with fines attached, and it is pushing the entire
industry toward geographic distribution whether anybody wanted it or not.

Physics from one side, lawyers from the other. Edge isn't a trend. It's a
squeeze.

## So let's jump back to the original question then — the Edge of What?

It's simple. Delete the public cloud data centres from your mind and replace them
with much smaller, highly distributed points-of-presence, and you get half the
answer.

The second half comes from the device you hold in your hand, or that sits on the
desk in front of you.

That's the edge.

<figure class="fig">
<svg viewBox="0 0 640 236" role="img" aria-label="Three tiers: your device and a nearby point of presence together form the edge; the hyperscaler datacentre sits outside it.">
  <rect class="box weights" x="50" y="56" width="150" height="72" rx="6"/>
  <text class="t-lbl mid" x="125" y="82">your device</text>
  <text class="t-mono mid" x="125" y="102">everything else</text>
  <text class="t-mono mid" x="125" y="118">thick client</text>
  <rect class="box weights" x="240" y="56" width="150" height="72" rx="6"/>
  <text class="t-lbl mid" x="315" y="82">a POP, &lt;20ms away</text>
  <text class="t-mono mid" x="315" y="102">shared state</text>
  <text class="t-mono mid" x="315" y="118">held by a third party</text>
  <rect class="box" x="440" y="56" width="150" height="72" rx="6"/>
  <text class="t-lbl mid" x="515" y="82">hyperscaler DC</text>
  <text class="t-mono mid" x="515" y="102">five of them</text>
  <text class="t-mono mid" x="515" y="118">73% of hosting</text>
  <path class="ax" d="M50 148 V158 H390 V148"/>
  <text class="t-flat mid" x="220" y="180">the edge</text>
  <path class="ax thin" d="M440 148 V158 H590 V148"/>
  <text class="t-dim mid" x="515" y="180">not the edge — and never can be</text>
  <text class="t-dim" x="50" y="216">the boundary is not distance. it is whether the data has to be trusted by more than one person.</text>
</svg>
<figcaption>Two tiers inside the line, one outside it. And the line is drawn by
trust, not by latency.</figcaption>
</figure>

## So how do you distinguish the two halves?

If the Edge is both the device you are using *plus* smaller data centres hosted
physically close to you, then which workload goes where?

It comes down to this.

No matter what your computing workload is, no matter what use case it runs,
eventually you will need **some shared data that multiple humans can trust**. My
bank account balance. Stock levels. Work contracts. Industrial measurements.

Those things can't run on your personal devices, as the data must be protected by
a 3rd party. That isn't a technical limitation you'll engineer away in a few
years, by the way — it is what the word "trusted" *means*. If I can edit my own
balance, it isn't a balance.

Hence Edge is:

**a)** thousands of small data centres around the world positioned as close to you
as possible, that hold shared data — social state.

**b)** your personal computing device, that holds and runs everything else, ideally
as much of the app as possible. A thick client.

And Edge computing is not, and can never be, the public cloud providers' super
data centres.

Not because they're too big, and not because they're too far away. Because
**they're the thing Edge exists as a reaction to**. A hyperscaler region with a CDN
bolted to the front is a centralised point of origin that has learned to cache.
Putting a copy near the user doesn't distribute a system — it makes a centralised
one feel quicker right up until somebody writes something.

<div class="key">
<h4>The test for whether you've actually built Edge</h4>
<p>Take one point of presence and cut its link to your central region entirely.</p>
<p>If it keeps serving its users — that's Edge. If it degrades to read-only, or
starts erroring on writes, or serves stale data with a shrug, then congratulations,
you've built a cache with excellent marketing.</p>
</div>

## The most successful Edge app on the planet

And by an absolute mile, it's DNS.

Half of it is running on your device right now, and the other half is less than
20ms away in a small DC. An unbreakable service, diligently turning human words
into IP addresses, at a scale and a reliability that nothing else in computing has
ever matched — with no owner, no single point of failure, and no vendor to call
when it breaks. Which it doesn't.

Thank you Paul Mockapetris, for showing us the way of the Edge.

---

## Postscript

The unfinished half of this argument is state.

You can distribute the compute all you like. The moment a workload needs shared
data that multiple humans can trust, you are straight back to asking where the
writes get ordered — and that place has gravity.

I picked that thread up the following week, and it turns out the answer isn't
about the data at all: [the journal is king](/blog/the-journal-is-king).
