---
title: "Edge, now the hype cycle is over"
date: 2024-02-14
tint: ok
tags: [edge, fintech, distributed-systems]
summary: >-
  Four things that matter about Edge for fintech: it deletes the target
  attackers aim at, it makes Kubernetes unnecessary, it demands you shard state
  for real, and it forces your migration to run outside-in.
---

<figure class="shot hero">
<img src="/img/blog/edge-fintech.jpg" alt="A datacentre corridor lined with racks, one cabinet standing apart at the centre" width="800" height="800">
</figure>

I've been getting my hands dirty with edge computing over the last few years.
Boring tech courses and reading external reports are not how I learn — in my view
if you want to learn to drive, just get in a car in a safe environment and start
driving. Ideally with someone else beside you who already knows how to drive.

So, practical hands-on coding it is then. But that's a topic for another day.

Now that edge computing has survived the hype cycle, what do I think is relevant
for fintech engineers and executives about Edge?

Four things.

## 1. Defence from cyber attacks

If your front line services are everywhere and nowhere, then an attacker has no
easy target to focus on. This includes the growing threat that comes from APTs —
advanced persistent threats.

Servers and containers are your weak spot. **Delete them** from your design if you
want to survive the next decades of cyber war.

Now I want to be precise about why this is different from simply buying more
defence, because it's the strongest argument in the whole post and it usually gets
skipped.

Read the middle word: **persistent**. An APT's entire method is to establish a
foothold and then sit in it — for months, sometimes years — learning your estate,
harvesting credentials, waiting. Every capability they have downstream depends on
that first property.

And persistence requires something to persist *in*.

A workload with no long-lived host, redeployed constantly across thousands of
points of presence, doesn't have a stronger door. It has **nowhere to establish
residency at all**. That's not a control an attacker can defeat with better
tooling or a bigger budget — it's an absence, and you cannot exploit an absence.

<div class="key">
<h4>The difference nobody prices properly</h4>
<p>Hardening a server is a race. You buy better locks, they buy better picks, and
the score is settled by whoever spends more — which, against a state actor, is not
going to be your bank.</p>
<p>Removing the server ends the race. Not "wins" it. <b>Ends</b> it. Those are
completely different products and we keep buying the first one.</p>
</div>

This is the same argument as [shifting your security
left](/blog/shift-left-your-security), arriving from the opposite direction. There
the point was that runtime controls exist to compensate for design decisions
nobody made. Here the design decision is sitting right there: don't have the thing
that needs defending.

## 2. Edge is the Kubernetes killer

When your services automatically run on tens of thousands of servers — POPs — all
over the world, then scalability and robustness are not a choice. They are a
**side effect** of the different thinking pattern you are forced into.

That's a powerful concept when you use it right. The hard bit is getting over the
learning curve of those constraints, but when you master them the other costs
drop away.

And I'm looking at you, Mr. k8s, where an estimated 30% of corporate engineering
power is wasted on him.

Edge does not need Kubernetes. Not "can be made to work without" — **does not
need**. And the reason is worth spelling out, because people hear this as
tribalism when it's actually a category error.

Kubernetes solves scheduling. *Given a fleet of interchangeable machines and a pile
of heterogeneous workloads, decide what runs where and keep it running.* It is a
good answer to that question. It's a genuinely impressive piece of engineering.

But at the Edge, **the placement question was already answered by geography**
before anybody wrote a line of code. The workload runs at the POP nearest the
user. There is no bin-packing problem, no scheduler, no affinity rules, no taints,
no tolerations, no cluster autoscaler, and no team of six keeping all of it
upright. The hardest problem Kubernetes solves does not exist in this
architecture.

Which is why "let's run Kubernetes at the edge" is such a telling phrase. It's
importing the solution to a problem you have just designed away, and then hiring
people to operate it.

## 3. Sharding of state is the key to success

This is the big one, and it's where most fintech transitions to Edge will live or
die.

One of the key architectural changes needed for fintech to transition to Edge will
be its traditional view on state and how one makes it scale. As Cassandra, Kafka
and other distributed datastores infected the enterprise, we still treated them as
`SELECT WHERE` queries — a centralised mental model wearing a distributed
implementation like a coat.

That does not fly in Edge, as **data gravity is an Edge anti-pattern** that tries
to centralise what is decentralised.

In Edge you have to shard the database itself. No fancy cache models, no voting
transactions. You literally have to distribute the source of truth on all the
POPs.

<figure class="fig">
<svg viewBox="0 0 640 224" role="img" aria-label="Centralised state with edge caches in front of it, versus genuinely sharded state where each point of presence holds its own source of truth.">
  <text class="t-rise" x="40" y="28">cache at the edge — still centralised</text>
  <rect class="box" x="40" y="38" width="54" height="24" rx="4"/>
  <rect class="box" x="102" y="38" width="54" height="24" rx="4"/>
  <rect class="box" x="164" y="38" width="54" height="24" rx="4"/>
  <rect class="box" x="226" y="38" width="54" height="24" rx="4"/>
  <path class="link" d="M67 62 L160 92 M129 62 L160 92 M191 62 L160 92 M253 62 L160 92"/>
  <rect class="box weights" x="96" y="92" width="128" height="34" rx="5"/>
  <text class="t-mono mid" x="160" y="114">the truth</text>
  <text class="t-rise mid" x="160" y="152">one origin · one gravity well</text>
  <text class="t-rise mid" x="160" y="170">writes go home</text>
  <text class="t-flat" x="380" y="28">sharded — truth at every POP</text>
  <rect class="box weights" x="380" y="38" width="48" height="24" rx="4"/>
  <rect class="box weights" x="436" y="38" width="48" height="24" rx="4"/>
  <rect class="box weights" x="492" y="38" width="48" height="24" rx="4"/>
  <rect class="box weights" x="548" y="38" width="48" height="24" rx="4"/>
  <rect class="box weights" x="380" y="70" width="48" height="24" rx="4"/>
  <rect class="box weights" x="436" y="70" width="48" height="24" rx="4"/>
  <rect class="box weights" x="492" y="70" width="48" height="24" rx="4"/>
  <rect class="box weights" x="548" y="70" width="48" height="24" rx="4"/>
  <text class="t-flat mid" x="488" y="126">no single origin</text>
  <text class="t-flat mid" x="488" y="152">writes stay local</text>
  <text class="t-flat mid" x="488" y="170">every customer is a shard</text>
  <text class="t-dim" x="40" y="208">the left-hand picture is what most "edge" deployments actually are.</text>
</svg>
<figcaption>Putting a cache near the user does not distribute a system. It makes
a centralised one feel faster, right up until somebody writes.</figcaption>
</figure>

So far the Edge industry are heavily using S3 with customer-centric buckets to
pull this off. But in the banking world, how do we deal with the fact that **every
customer is a different database?**

And that question is not rhetorical, nor is it a complaint. It's the design.

Look at what per-customer isolation buys you, all at once and for free: data
residency falls out (the customer's shard lives in the customer's jurisdiction);
blast radius containment falls out (one corrupted shard is one customer, not one
outage); right-to-erasure falls out (delete the shard); noisy-neighbour disappears;
and per-tenant encryption keys stop being a research project.

Every one of those is something banks currently spend enormous money achieving
*despite* their architecture. Sharding by customer gets them as a side effect.

The reason it feels impossible is that traditional banking cores are built the
opposite way round — one enormous shared ledger with an access-control layer
bolted to the front, where "customer" is a `WHERE` clause rather than a boundary.
That model cannot express per-customer isolation at all, and no amount of Edge
infrastructure will fix a core that thinks a customer is a filter.

<div class="key">
<h4>The uncomfortable version</h4>
<p>"Every customer is a different database" isn't the problem Edge creates. It is
the answer to about six problems you already have, arriving in a shape your core
banking system cannot pronounce.</p>
<p>Which tells you where the real migration work is, and it isn't at the edge.</p>
</div>

## 4. Migrate from Outside to Inside

The benefits — and also the weaknesses — of Edge are its distributed nature. Thus
it only really works well and gives its benefits when your customers are also
distributed.

For the Web that's easy, with its 15 billion distributed devices. But when a
single region or location — such as a data centre — reaches out to the Edge, those
benefits drop off and start to become a headache. You have taken on the entire
coordination cost of a distributed system in order to serve traffic that arrives
from one building.

It's for this reason that it makes most sense to follow a simple rule:

> Internet devices go **first** to your Edge layer, then after to your traditional
> private or public cloud, after which you stay there.

This means your migration strategy will become an **outside-in** migration path.

Start at the boundary where the traffic genuinely arrives distributed, and work
inwards. Not the other way round — which is what every "cloud migration, then edge
later" roadmap on earth proposes, and which gets the economics backwards from day
one. Those roadmaps move everything to the centre first and then try to push it
back out, paying for both journeys and getting the benefit of neither.

---

## Postscript

Point 3 is the one I never let go of, and it's the one the industry has since
started proving for me.

Look at what the edge platforms actually shipped.
[Per-object state](https://developers.cloudflare.com/durable-objects/) where each
object is single-threaded and lives in exactly one place.
[SQLite databases](https://developers.cloudflare.com/d1/) created per customer, at
the edge, in the thousands. A whole product category built on
[database-per-tenant](https://turso.tech/), sold on precisely the properties I
listed above — residency, blast radius, per-tenant keys.

Nobody built the giant globally-consistent distributed store everyone assumed edge
would need. They built **millions of tiny single-origin ones** instead, and it
works.

Which is the interesting part, because it means they didn't defeat data gravity.
They **accepted** it and made the gravity wells small enough not to matter. Give
each customer one authoritative location and the ordering problem doesn't go away
— it gets divided by ten million until each instance of it is trivial.

That's a better answer than the one I was reaching for in 2024, and I'd take it
over a distributed journal if I could only have one. [The following
week](/blog/the-journal-is-king) I argued that "data gravity" is the wrong phrase
entirely, because data copies freely and a copy has no mass — what pulls is data
*at its origin*, and the origin is always a journal.

Millions of small journals, one per customer, each with its own uncontested
origin. Sharding by customer wasn't a workaround for the edge. It was the shape
the problem had all along.

The genuinely distributed journal — one with **no** point of origin — is still
unsolved, and I now suspect it stays that way. But it turns out you mostly don't
need it. You need the origin to be small, local, and yours.

What has to go first is still the ledger-shaped mental model, where a customer is
a `WHERE` clause. Everything else is available off the shelf.
