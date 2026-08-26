---
title: "Monolithic deployments"
date: 2023-06-21
tint: warn
tags: [architecture, devops, fintech]
summary: >-
  Deterministic systems give you the safety margin to change anything at any
  time. Technical debt stops being frightening the moment you can step from one
  working plane to the next. We ran an entire bank this way.
---

<figure class="shot hero">
<img src="/img/blog/monolithic.jpg" alt="A man leaping through open air beside an airliner in flight" width="800" height="665">
</figure>

Deterministic systems, as opposed to chaotic systems, give their owners extreme
reliability and the safety margin to make BIG change of anything at anytime.

That predictability is what gives a sustainable and reliable march forward with
your tech estate. And it comes down to this:

> Technical debt is just vastly easier to eliminate if you can swap it out as
> easily as jumping from one working plane to the next working plane.

## Component deployment versus monolithic deployment

Most estates practise **component deployment**. Each service, library or subsystem
ships on its own cadence, to its own environment, with its own version matrix. It
sounds like agility. What it actually produces is a combinatorial explosion of
states that has never been tested and never will be.

**Monolithic deployment** is the opposite discipline. The entire scope of your
estate is built together, versioned together and promoted together. There is one
thing in production and you know precisely what it is.

<figure class="fig">
<svg viewBox="0 0 640 232" role="img" aria-label="Component deployment produces many independently-versioned parts with an untested combination space; monolithic deployment produces one versioned estate promoted as a unit.">
  <text class="t-rise" x="40" y="30">component deployment</text>
  <rect class="box" x="40" y="42" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="63" y="58">v4</text>
  <rect class="box" x="94" y="42" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="117" y="58">v9</text>
  <rect class="box" x="148" y="42" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="171" y="58">v2</text>
  <rect class="box" x="202" y="42" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="225" y="58">v7</text>
  <rect class="box" x="40" y="76" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="63" y="92">v5</text>
  <rect class="box" x="94" y="76" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="117" y="92">v9</text>
  <rect class="box" x="148" y="76" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="171" y="92">v3</text>
  <rect class="box" x="202" y="76" width="46" height="24" rx="4"/>
  <text class="t-mono mid" x="225" y="92">v7</text>
  <text class="t-rise mid" x="144" y="126">which combination is in production?</text>
  <text class="t-rise mid" x="144" y="144">nobody tested this one</text>
  <text class="t-flat" x="380" y="30">monolithic deployment</text>
  <rect class="box weights" x="380" y="42" width="216" height="58" rx="6"/>
  <text class="t-lbl mid" x="488" y="66">the estate</text>
  <text class="t-mono mid" x="488" y="86">v1.4.0</text>
  <text class="t-flat mid" x="488" y="126">one version. one thing to test.</text>
  <text class="t-flat mid" x="488" y="144">one thing to roll back.</text>
  <text class="t-dim" x="40" y="184">the left-hand estate has a state space. the right-hand estate has a state.</text>
  <text class="t-dim" x="40" y="206">you cannot reason about a system whose current configuration has never existed in a test environment.</text>
</svg>
<figcaption>Component deployment doesn't remove integration risk. It moves it into
production and spreads it across every combination nobody ran.</figcaption>
</figure>

And let's do that arithmetic properly, because it's worse than it looks. Twenty
components with four supported versions each is a trillion possible
configurations. You test perhaps a dozen of them. Production then picks one at
random from the remaining trillion and you find out how it went from your
customers.

That is not a testing gap you can close by writing more tests. It's a **category**
of problem you can only close by refusing to create it.

## Why determinism is the thing that actually pays

The usual argument for independent deployment is speed — small changes ship
faster. That's true, and it's also beside the point, because shipping speed is
rarely what limits an estate.

What limits an estate is **fear**.

Fear is what stops you upgrading the framework. Fear is what leaves the old
authentication path in place for four years because nobody is certain what still
calls it. Fear is what turns a two-day change into a six-month programme with a
steering committee, three impact assessments and a go/no-go call at 2am on a
Sunday.

And that fear is entirely rational. It is the correct response to an estate whose
production configuration is a combination that has never existed anywhere else. A
sensible engineer facing that system *should* be frightened of it.

Determinism removes the fear, and removing the fear is what unlocks the big moves.
A team that can rebuild and redeploy its entire estate on demand will casually do
things on a Wednesday that a component-deployed estate treats as a once-a-decade
migration programme.

<div class="key">
<h4>The plane test</h4>
<p>Can you step from your current production estate to a completely rebuilt one,
in flight, and step back again if it goes wrong?</p>
<p>If yes — technical debt is a chore. If no — technical debt is a <em>risk</em>,
and risk compounds. Which is precisely why estates that can't do this get slowly,
permanently worse while everybody works very hard.</p>
</div>

### And the industry already proved this at the top end

Now I'm aware how this sounds to a room full of people who've been told the
opposite for a decade, so let me point at who else does it.

**Google** builds from a single monorepo where a change is tested against the
whole tree. **Facebook** shipped its main product as one enormous binary for
years. Both are meaningfully larger than your estate, both had every resource
required to do it the other way, and both chose determinism.

And look at where the tooling has drifted. **Trunk-based development** won the
argument against long-lived branches, and it won it for the same reason — one
integrated state beats many divergent ones. Every argument for trunk-based
development at the source level is an argument for monolithic deployment at the
estate level. We accepted it in one place and refused it in the other, which is
simply inconsistent.

## This is not a thought experiment

In ING Australia we mastered that strategy with **`vNext`** and the results speak
for themselves. It's a pattern that has survived the test of time, and a pattern
that has been proven to work.

And I'd stress the conditions, because everyone's first objection is that this
only works for greenfield systems or small teams. It was neither. It was an entire
bank, under the regulatory scrutiny that implies, with real customers and real
money and real auditors, on an estate nobody had the luxury of starting from
scratch.

If you want that same predictability, stability and speed... you need to stop
practising "Component Deployment" and start doing "Monolithic Deployments".

**If ING can do it with an entire bank, you can do it too.**

---

## Postscript

Years later the discipline followed me into a codebase with no bank, no regulator
and no steering committee — where it turned out to matter even more, because there
was nobody around to stop me from breaking things.

The contributor rules for the inference engine read like monolithic-deployment
doctrine one level down:

> **No backward compatibility.** It is fine — expected — to break everything
> before you. Do not write compatibility shims, dual code paths, or `Option`-typed
> feature flags that exist only to keep an old path alive. Replace the real thing.

Same claim, applied to source. A compatibility shim *is* component deployment
inside a single binary: two versions coexisting, one of them rarely exercised, and
a combination space nobody tested. Deleting the old path is frightening exactly
once, and then it's gone — and everything afterwards moves faster because there is
only ever one thing in the estate.

The plane test, it turns out, is scale-free.
