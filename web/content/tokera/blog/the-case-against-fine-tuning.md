---
title: "AI: the case against fine-tuning your NPCs"
date: 2026-08-03
tint: ok
tags: [npcs, architecture, games]
summary: >-
  You want a character with a personality, so you fine-tune a model on them. It
  demos beautifully and it is the wrong architecture — because it puts the
  character in the one place in the entire system that can never change.
---

So you want a game character with a real personality. Their voice, their history,
their opinions.

The obvious move is to fine-tune a model on them, and I'll give it this much: it
demos beautifully. Put it in front of a publisher and watch the room light up.

It's also the wrong architecture, and I'm going to be specific about why, because
"just fine-tune it" is the default answer in an awful lot of rooms right now and
almost nobody is asking what happens on day two.

Here's the whole objection in one line.

> Fine-tuning writes the character into the **weights**. And weights are the one
> part of your system that cannot change at runtime, cannot be per-player, and
> cannot remember last Tuesday.

## Four things it can never do

**It cannot remember.** A fine-tune teaches a *disposition*, not an episode. Your
character can sound like a bitter ex-sergeant all day long, but it has no idea it
met you yesterday, that you lied to it, or that you came back anyway.

And notice what happens when you fix that. You bolt a retrieval store on the side
— at which point you've conceded the entire argument. The memory now lives outside
the weights, and the interesting system is no longer the model. It's the thing you
just bolted on.

**It cannot change its mind.** Personality in the weights is frozen at training
time. A character who *becomes* suspicious of you, gradually, because of things
you actually did to them — that's the whole reason anybody wants this. And it is
precisely the one thing a static artefact cannot express. So you end up faking it
with state machines and quest flags, which is exactly where we came in.

**It cannot be per-player.** The entire point of a persistent character is that
they know *you*. Weights are shared across every player on the server. So either
everybody gets the same character, or you fine-tune per player. Nobody can afford
that. Not you, not Rockstar, not anyone.

**It cannot scale to a cast.** A game needs hundreds of characters. That's hundreds
of adapters to train, version, store and re-train every single time the base model
moves. And after all that, you still have none of the three above.

Four for four. That's not a tuning problem, that's the architecture telling you
something.

## So put the character somewhere it can actually move

One model. Shared. **Never fine-tuned.**

Every character is instead a **substrate** — a set of append-only layers recording
what they've perceived, what they've done, what they've come to believe, and how
they feel about everyone they've ever met.

<figure class="fig">
<svg viewBox="0 0 640 300" role="img" aria-label="One shared model reading many per-character substrates, each with an immutable core and mutable layers.">
  <rect class="box weights" x="204" y="16" width="232" height="52" rx="10"/>
  <text class="t-lbl mid" x="320" y="40">one model, shared</text>
  <text class="t-mono mid" x="320" y="58">never fine-tuned</text>
  <path class="link" d="M256 68 V104 M320 68 V104 M384 68 V104"/>
  <g>
    <rect class="box sub" x="16"  y="104" width="184" height="172" rx="10"/>
    <rect class="box sub" x="228" y="104" width="184" height="172" rx="10"/>
    <rect class="box sub" x="440" y="104" width="184" height="172" rx="10"/>
  </g>
  <text class="t-lbl" x="32"  y="128">Varek</text>
  <text class="t-lbl" x="244" y="128">Ilse</text>
  <text class="t-lbl" x="456" y="128">Hess</text>
  <g class="layers">
    <rect class="l a" x="32"  y="140" width="152" height="17" rx="4"/>
    <rect class="l b" x="32"  y="161" width="152" height="17" rx="4"/>
    <rect class="l c" x="32"  y="182" width="152" height="17" rx="4"/>
    <rect class="l d" x="32"  y="203" width="152" height="17" rx="4"/>
    <rect class="l core" x="32" y="230" width="152" height="24" rx="4"/>
    <rect class="l a" x="244" y="140" width="152" height="17" rx="4"/>
    <rect class="l b" x="244" y="161" width="152" height="17" rx="4"/>
    <rect class="l c" x="244" y="182" width="152" height="17" rx="4"/>
    <rect class="l d" x="244" y="203" width="152" height="17" rx="4"/>
    <rect class="l core" x="244" y="230" width="152" height="24" rx="4"/>
    <rect class="l a" x="456" y="140" width="152" height="17" rx="4"/>
    <rect class="l b" x="456" y="161" width="152" height="17" rx="4"/>
    <rect class="l c" x="456" y="182" width="152" height="17" rx="4"/>
    <rect class="l d" x="456" y="203" width="152" height="17" rx="4"/>
    <rect class="l core" x="456" y="230" width="152" height="24" rx="4"/>
  </g>
  <text class="t-mono" x="40" y="246">immutable core</text>
  <text class="t-mono" x="252" y="246">immutable core</text>
  <text class="t-mono" x="464" y="246">immutable core</text>
  <text class="t-dim" x="16" y="292">perception · relationships · beliefs · memory — append-only, drifting under selection</text>
</svg>
<figcaption>The character isn't in the weights. It's a stream of things that
happened, sitting over a fixed core that nothing is permitted to overwrite.</figcaption>
</figure>

The layers are **perception** (what reached them), **action** (what they did),
**agency** (what they're trying to achieve), **relationships** (per-entity
calibration), **beliefs**, and **memory**. Underneath sits an **immutable core** —
world knowledge and the archetype — read-only by construction, and shared as a
copy-on-write prefix across every character of that type.

## And then, one operation everywhere

And here's where it starts paying for itself.

Every behaviour in the system is the same move viewed from a different angle:
**gather the relevant substrate under a salience budget, attend over it through a
fixed lens, act, write the result back.**

That's it. No separate filtering system. No routing table. No
conditional-trigger machinery. No mood state machine. One loop, a layered
substrate it reads, and provenance attention deciding what's salient this tick.

Two consequences fall out of that, and both are load-bearing.

<div class="key">
<h4>Filtering is selection, not a gate</h4>
<p>"Should this belief affect behaviour?" is never evaluated by a predicate. The
belief block either wins the gather against the current cognitive state, or it
doesn't.</p>
<p>A filter that drops a signal <em>before</em> it lands is control. Letting it
land and lose the salience competition is influence. Build it the first way and
your characters lurch — one event trips a flag and the whole personality changes
between frames. Build it the second way and they drift... which is what actual
people do.</p>
</div>

<div class="key">
<h4>Convergence is attention, not reconciliation</h4>
<p>When several things land on the substrate at once they are never merged by a
reconciliation step. They co-occur in the working set, and the softmax over the
gathered set <em>is</em> the integration.</p>
<p>Gather-and-attend is the merge. There is no other merge.</p>
</div>

## The one rule that keeps it honest

One discipline holds this entire design together, and if you take nothing else,
take this:

> The model nudges its mutable substrate but never controls it — and it cannot
> touch its immutable core at all.

No layer is ever written by fiat. The model emits *mutation events*, they land on
the substrate as one more entry among many, and what a layer **becomes** is the
gather over all of them under salience.

Sounds pedantic, doesn't it. It is the entire difference between a character who
has opinions and a character who has variables.

## Beliefs move. They never flip.

A belief carries a confidence, a threshold at which it starts driving behaviour,
and a disconfirmation pressure that accumulates as evidence contradicts it.
Evidence **moves** the number. Nothing sets it.

And the consequences are exactly the things you actually wanted from a character
in the first place:

- They can be **wrong for a very long time**, and act on it
- They can be **gradually persuaded** over a hundred hours, and the transition is
  legible afterwards
- They can hold a belief **under pressure** — visibly straining, not yet changed —
  which is the most human state on this list
- Two characters given identical evidence can end up in different places, because
  they started from different confidences

Not one of those is a quest flag. All of them fall out of the same gather.

## The economics, since this is a game

Per character, a fine-tune costs you a training run, an artefact, a version, and a
re-train every time the base model moves.

Now multiply by two hundred.

Per character, a substrate costs you a directory of append-only records and some
KV cache. The model is the same one bytes-for-bytes across every character, so it
loads once. Characters cost **memory** — which the engine underneath was built
specifically to make unbounded and cheap, and [that's a whole argument of its
own](/blog/one-card-unbounded-context) — rather than costing **weights**, which
are the single most expensive thing in the building.

And the archetype layer is a shared copy-on-write prefix, so two hundred soldiers
who share a doctrine share those tokens too.

## Now, what you give up

Here's the honest bit, because there's a real cost and I'm not going to pretend
otherwise.

A fine-tune gives you **voice** almost for free. Vocabulary, cadence, verbal tics —
all the things that make a character feel *written*. A shared model reading a
substrate does not automatically sound like anyone in particular, and I'll tell
you exactly what the first version of this sounded like: six different people
doing an impression of the same assistant.

The fix is to split the job. The character emits **intent**. A narrator turns
intent into words. The character decides *to refuse, coldly, and mention the
debt* — the narrator writes the sentence.

Voice becomes a rendering concern with its own consistent lens, instead of
something you hope survived gradient descent. And the operator surface follows the
same rule in both directions — **inject down, narrate up, never fabricate**. An
operator speaking becomes an event in the character's inbox; the character's acts
become prose; the interaction layer never invents an act that didn't happen.

It also means your characters never write their own dialogue, which for anything
actually shipping to players is a property you will be extremely grateful for at
2am.

## Where fine-tuning genuinely wins

Credit where it's due. If you want **one** character, with a strong voice, no
memory of individual players and no capacity to change — fine-tune it. Less
machinery, and it'll sound better sooner. I'd do the same.

But the moment you want two hundred of them, each remembering a different player,
each capable of being wrong and then being talked round...

...the weights are the worst place in the entire system to have put any of it.
