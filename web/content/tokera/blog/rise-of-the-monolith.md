---
title: "Rise of the monolith"
date: 2023-07-19
tint: crit
tags: [architecture, microservices, industry]
summary: >-
  Microservices didn't win an argument, they won a conversion. And you don't
  beat a religion with a benchmark. To defeat one you have to create one — so
  here's the creed.
---

<figure class="shot hero">
<img src="/img/blog/monolith.jpg" alt="A vast dark monolith standing on a plain, split by a seam of molten lava" width="483" height="416">
</figure>

> *"To defeat a religion... one must create one."*

Microservices did not win an argument. They won a **conversion**, and that
distinction is the reason a decade of careful, well-evidenced pushback has
achieved approximately nothing.

You cannot refute a faith with a benchmark. Nobody holds this position because
they weighed the trade-offs and the numbers came out that way. They hold it
because it is what serious engineers are understood to do, and because the
alternative has been successfully framed as an admission of failure. Say
"monolith" in a design review and watch the room decide things about you.

So this is not another list of reasons fine-grained services are a bad default.
That list exists, it's correct, and it hasn't worked.

This is the other thing.

## First, let's name the mechanism out loud

The strongest force pushing services smaller has never been a technical
requirement.

It's **CV architecture**.

A distributed system is a career asset. Kubernetes on your CV is worth more than
"maintained a well-factored modular monolith", regardless of which one actually
delivered value to a human being. Every engineer reading this knows it. Almost
none of us will say it in a meeting.

And I want to be clear I'm not calling anybody cynical here. The incentive is
real, and responding to a real incentive is not a character flaw — it's what
incentives *are for*. But follow it through:

<div class="key">
<h4>Follow the incentive, not the argument</h4>
<p>An architecture decision that benefits the person making it, at a cost paid
three years later by somebody else entirely, gets made every single time. And it
always arrives dressed in technical language — because technical language is the
only language in which it can be proposed.</p>
<p>Which is why this debate feels unwinnable on the merits. Both sides are arguing
about latency and coupling. Only one of them is <em>about</em> latency and
coupling.</p>
</div>

Underneath the CV pressure sits a second mechanism, and this one I'll forgive
because we all did it. People followed blindly into fine-grained services
something that should always have been **coarse-grained**.

Service-oriented architecture had this right and then comprehensively lost the
plot. A service boundary is expensive. It costs you a network hop, a
serialisation format, a version negotiation, an independent failure mode, a
deployment lifecycle, a monitoring surface, an on-call rota and a team. Those
costs are absolutely worth paying at a boundary that genuinely needs to flex on
its own.

They are catastrophic when you pay them forty times inside a single product
because a conference talk said "single responsibility principle".

The unit of decomposition was supposed to be a **business capability**. It became
a noun in the domain model. That's the entire error — and it's a small one, which
is exactly why it spread like it did.

## The bit everybody gets backwards

Here's the argument I hear most, and it is simply false.

*"We need microservices for modularity."*

No you don't. **Modularity is free.** It's a property of how you write code, and a
well-factored monolith has strictly *more* of it than a service mesh does —
because the compiler enforces the boundaries and no amount of Friday-afternoon
deadline pressure can smuggle a violation past it.

At a network boundary, the enforcement mechanism is a code review and a hope.

<figure class="fig">
<svg viewBox="0 0 640 224" role="img" aria-label="One coarse-grained deployable containing many modules, versus many fine-grained services each carrying its own pipeline, monitoring, versioning and failure mode.">
  <text class="t-flat" x="40" y="30">one coarse-grained deployable</text>
  <rect class="box weights" x="40" y="40" width="240" height="96" rx="6"/>
  <rect class="box" x="54" y="54" width="98" height="30" rx="4"/>
  <rect class="box" x="166" y="54" width="98" height="30" rx="4"/>
  <rect class="box" x="54" y="92" width="98" height="30" rx="4"/>
  <rect class="box" x="166" y="92" width="98" height="30" rx="4"/>
  <text class="t-dim mid" x="160" y="156">4 modules · 1 pipeline · 1 failure mode</text>
  <text class="t-rise" x="360" y="30">four fine-grained services</text>
  <rect class="box" x="360" y="40" width="52" height="30" rx="4"/>
  <rect class="box" x="422" y="40" width="52" height="30" rx="4"/>
  <rect class="box" x="484" y="40" width="52" height="30" rx="4"/>
  <rect class="box" x="546" y="40" width="52" height="30" rx="4"/>
  <path class="link" d="M386 70 V136 M448 70 V136 M510 70 V136 M572 70 V136"/>
  <rect class="box" x="360" y="106" width="52" height="30" rx="4"/>
  <rect class="box" x="422" y="106" width="52" height="30" rx="4"/>
  <rect class="box" x="484" y="106" width="52" height="30" rx="4"/>
  <rect class="box" x="546" y="106" width="52" height="30" rx="4"/>
  <text class="t-dim mid" x="479" y="156">4 pipelines · 4 monitoring surfaces</text>
  <text class="t-dim mid" x="479" y="174">6 network paths · 4 version negotiations</text>
  <text class="t-dim" x="40" y="206">the modules on the left are just as separable. they are separated at compile time instead of over a network.</text>
</svg>
<figcaption>The left box isn't "less modular". It's exactly as modular, with the
boundaries enforced by something that cannot be argued with.</figcaption>
</figure>

What a service boundary actually buys you is **independent deployability** and
**independent scaling**. Both are real. Both are worth good money at the
boundaries that need them. Neither is worth a penny at a boundary that doesn't.

And most don't.

## And the receipts are piling up

Now here's the part that should have ended this debate, except that faiths don't
work like that.

The people who went furthest into fine-grained services are the ones walking
back.

- **Amazon Prime Video** rebuilt their audio/video monitoring service from
  distributed serverless components into a single process and
  [cut costs by 90%](https://www.thestack.technology/amazon-prime-video-microservices-monolith/).
  Their own engineers wrote it up. The bottleneck was orchestration overhead and
  shuffling frames through object storage between steps — costs that existed
  *only because the pieces were separate*.
- **Segment** ran 140-odd microservices, found the operational burden was eating
  the company, and consolidated back to a monolith. They published it under the
  title
  ["Goodbye Microservices"](https://www.twilio.com/en-us/blog/developers/best-practices/goodbye-microservices),
  which is about as unambiguous as our industry gets.
- **Istio** — the service mesh, the very cathedral of this religion — collapsed
  its own control plane from a set of microservices into
  [a single binary called `istiod`](https://istio.io/latest/blog/2020/istiod/).
  Their reasoning is worth reading in their own words: microservices are a good
  pattern when independent rollout and scale are worth more than the cost of
  orchestration, and for their control plane *none of that was the case*.
- Even **Martin Fowler**, who has done more than almost anyone to popularise this
  stuff, published
  [*MonolithFirst*](https://martinfowler.com/bliki/MonolithFirst.html) — start
  with the monolith, split when the seams announce themselves.

Notice the pattern in that list. It isn't small teams who couldn't cope. It's the
organisations who went deepest, hit the wall hardest, and had enough engineering
maturity to publish the retreat.

<div class="key">
<h4>The tell</h4>
<p>When a pattern is genuinely correct, the most sophisticated practitioners push
it <em>further</em>. When a pattern is a fashion, the most sophisticated
practitioners are the first ones out, and everybody else spends five years
catching up while insisting it's working fine.</p>
<p>We are somewhere in year four of that.</p>
</div>

## So: create the religion

Argument has been tried. Here's the creed instead — short enough to remember,
specific enough to act on. Those are the properties that make ideas actually
spread, and I'd rather be memorable than exhaustive.

<div class="key">
<h4>The creed</h4>
<p><b>One binary.</b> A deployable is not a component. It is the whole estate,
built once and deployed many times.</p>
<p><b>Boundaries in the compiler, not on the wire.</b> If a compiler can enforce
it, a network must never be asked to.</p>
<p><b>A service boundary is a debt, not an asset.</b> You justify creating one.
You never justify not creating one.</p>
<p><b>Coarse by default, fine by evidence.</b> Split when a piece has genuinely
divergent scaling or deployment needs. Then, and not before.</p>
<p><b>The estate moves as one.</b> Predictable change beats independent change —
you should be able to swap the whole thing like stepping from one working plane
to the next.</p>
</div>

That last one isn't theoretical either. [We ran an entire bank this
way](/blog/monolithic-deployments) — deterministic, monolithic deployments across
the whole estate — and the pattern has survived the test of time.

## Now let me head off the obvious objections

**"You just want to go back to the mainframe."** I don't. I'm not against HTTP,
not against distribution, and not against the interconnected world. The
interconnection is the good bit. I want fewer *deployables*, not fewer
*interfaces*.

**"Kubernetes is fine actually."** Kubernetes is a perfectly good answer to the
question it was built for, which is *"schedule heterogeneous workloads across a
fleet I don't want to think about"*. It is not, and was never advertised as, a way
to structure your application. We took an infrastructure tool and turned it into
an architectural philosophy, and then blamed the tool.

**"Some systems really are fine-grained."** Yes. Some boundaries genuinely need to
flex alone and I'll help you find them. The claim here is narrower and much harder
to wriggle out of: **fine-grained is a justification, not a default.** The industry
inverted that, and it inverted it for reasons that were social rather than
technical.

And the bill has now come due. Somewhere between [46 and 200 million APIs against
31 million engineers](/blog/the-period-of-api-neglect) is not a design problem any
more. It's an arithmetic one. And no quantity of people getting better at
operating microservices makes that subtraction come out differently.

Faiths are not defeated by evidence.

They're replaced by better ones.
