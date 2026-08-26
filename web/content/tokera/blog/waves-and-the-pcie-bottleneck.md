---
title: "AI: Waves, or how to lose a fight with PCIe"
date: 2026-08-19
feature: 1
tint: info
tags: [moe, scheduling, cuda]
summary: >-
  Your mixture-of-experts model doesn't fit in VRAM, so you stream the weights.
  Now a bus is your bottleneck, every big prefill wrecks the cache for everyone
  else, and the metric you're optimising is the wrong one. With measured
  numbers from a 16 GB laptop and a Blackwell workstation.
---

Mixture-of-experts models are wonderful right up until you do the arithmetic.

Qwen3-30B-A3B has 30 billion parameters, of which 3 billion are active per token.
The *active* part fits a 16 GB card beautifully. The other 27 billion still have
to exist somewhere, and no amount of enthusiasm makes that go away.

So you stream them. 48 layers, 128 experts each, 6,144 experts in total, about
2.9 MB apiece. They live in host memory, get DMA'd into VRAM as the router asks
for them, and a cache keeps the hot ones resident.

Simple. And it works... right up until you notice you haven't removed the
bottleneck at all. You've moved it onto a bus.

Here's where this ends up, so you know whether to keep reading.

> **20 concurrent conversations on a 16 GB laptop, running a 30-billion-parameter
> model that does not fit on the card.** Aggregate decode goes from 9.5 to
> **166 t/s** — 17.5× — while every individual session still runs at **87% of the
> speed it got with the whole machine to itself**. Prefill hits **2,314 t/s**
> across the batch.
>
> The sessions aren't competing for the bus. They're sharing a load that was
> going to happen anyway.

Getting there took four mistakes, and the numbers at the end only mean anything
once you've seen them.

<div class="key">
<h4>Know your actual bus</h4>
<p>Two dev machines here. A 4090 Mobile on PCIe 4.0 measures <b>~25 GB/s</b>. A
3090 — bigger card, more VRAM, newer-looking — sits behind a Comet Lake CPU that
caps it at PCIe 3.0: <b>~12 GB/s</b>. Half.</p>
<p>Every sizing decision taken with the wrong number is wrong by 2×. Name the
machine when you quote a bus figure.</p>
</div>

What follows is four things I got wrong, in the order I found them.

Three were bugs in a data model or a metric. The fourth was a *correct* answer,
from a simulation that wasn't sloppy, which turned out to be the wrong answer on
real hardware. That's the most interesting kind and we'll get to it.

## Bug one: copying data that already exists

My first version had VRAM and pinned RAM as an *exclusive-or*. An expert is
either resident on the device or held in pinned host memory, and pinned was sized
to hold exactly the experts that weren't on the card. Neat, symmetric, obviously
sensible.

Now follow that through to eviction. An expert leaving VRAM has no home in RAM to
fall back to, because its pinned slot was never allocated. So eviction has to
**write the bytes back across the bus**.

```
DMA evicts (D2H), one gate config: 23,415 × 2.9 MB = 68 GB of PCIe traffic
```

Sixty-eight gigabytes. Per configuration. Competing on the copy stream against an
equal volume of loads going the other way.

And every single byte of it is bit-identical to what the pack file on disk already
contains. It's pure waste, and it exists entirely because somebody — me — modelled
residency as a choice between two places.

The exclusive-or did rather more damage than the traffic, too. Because
`pinned_occupied = total_experts − vram_slots` held by construction, and the
pinned pool was a single host allocation that never grew, **VRAM could never
hold fewer experts than `total − pinned_capacity`**. That is a hard floor on
device residency set by a host allocation — the dependency running backwards.
The floor moved twice while the defect stood, at one point costing 14.1 GB of
pinned host memory on a 31.5 GB machine, spent in proportion to how far the
VRAM boundary *might* move rather than how far it did.

And the free slots in that pool were doing two jobs at once: they were the churn
depth of the swap pipeline (a swap is evict-then-load) *and* the entire budget
for retracting the VRAM boundary. At the opening position that was 237 slots
serving both. A live rebuild asked for 4,436 regions and reported
`relocated=0 evicted=0` — the two consumers had eaten each other.

### The fix is a different data model, not a smarter policy

Stop treating disk as a degraded state and start treating it as the truth:

```
cold    <model>.experts.pack     authoritative, write-once, all 6144 experts
warm    pinned host RAM          static, immutable, stratified subset
hot     VRAM                     dynamic cache, eviction = drop
```

Disk isn't a fallback an expert decays into. It is where every expert **is**,
permanently, from the moment the pack is written. RAM and VRAM residency record
that a faster copy *also* exists. The two are independent facts over an
always-present base, not three alternatives — an expert is very often in VRAM
*and* pinned RAM at once.

The load path becomes a total function with no error case:

```rust
match (r.vram, r.ram) {
    (Some(_), _)    => already resident,
    (None, Some(s)) => H2D from pinned slot s,
    (None, None)    => read from the pack file,
}
```

and eviction collapses to one field assignment. `r.vram = None`. That lands the
expert on whichever tier still holds it, without consulting anything, and without
moving a single byte.

Sixty-eight gigabytes of PCIe traffic, deleted by changing a type.

<div class="key">
<h4>Why a product and not a sum</h4>
<p>The obvious encoding is the sum: <code>Vram { slot, ram_backed } | Ram { slot }
| Disk</code>. Same four states, same semantics. The product —
<code>{ vram: Option, ram: Option }</code> — wins for one reason.</p>
<p><code>ram</code> is an immutable per-expert fact decided at startup, and the
sum form makes every VRAM transition rewrite it. Five sites construct a VRAM
location. Under the sum, each must restate <code>ram_backed</code> correctly:
<b>the compiler forces them to supply <i>a</i> value, but not the <i>right</i>
one</b> — and slot relocation is exactly where a hand-written
<code>ram_backed: None</code> silently orphans a live pinned slot for the process
lifetime. Under the product, those sites assign <code>vram</code> and the
<code>ram</code> field isn't in the expression, so it can't be dropped.</p>
<p>The invalid state is unrepresentable rather than merely avoided. That is worth
more than match exhaustiveness when you aren't adding tiers.</p>
</div>

## The warm tier is random on purpose

Which raises the obvious question of what should actually be *in* pinned RAM. And
the obvious answer is: rank the experts by popularity, keep the hottest.

The obvious answer is wrong, and the reason is worth sitting with.

A uniform random subset of size *X*% yields **X% hit rate on the accesses that
reach it** — whatever the popularity distribution. Every access lands on some
expert, and every expert is resident with probability *X* independent of how hot
it is. It simply doesn't matter what the shape is.

Now, the usual objection here is that random ignores the skew. And I'd agree with
you, except for one thing: **VRAM has already done the popularity filtering.**
What reaches RAM is the *miss* stream. The long tail. And the tail is far flatter
than the head, so random sampling of it is close to optimal.

So the property that would normally make random naive is precisely the property
that makes it sound here. It sits behind a tier that already skimmed the skew.

But that same sentence names the pool the draw has to run over — and the first
build got it wrong. If VRAM has already skimmed the skew, then the experts VRAM
*holds* are not in the miss stream at all, and a warm slot spent on one of them
buys you absolutely nothing until that expert gets evicted.

Drawing uniformly over all 6,144 spent **36% of the tier on guaranteed hits.**
Thirty-six percent, doing nothing.

The draw belongs over VRAM's complement — 3,767 experts on this model, not 6,144
— which quietly turns "hold the whole model in RAM" from a requirement into a
nicety.

### Then stratify it, because the wave is sequential

Take *X*% of *each layer*, not *X*% of the whole set. A global draw gives
per-layer residency `~ Binomial(128, X)`. At 40%: mean 51.2, standard deviation
5.54 — so across 48 layers the unluckiest lands near 37. **29% residency against
a 40% average.**

<figure class="fig">
<svg viewBox="0 0 640 214" role="img" aria-label="A global random draw spreads per-layer warm residency around 40 percent with the unluckiest of 48 layers near 29 percent; a stratified draw puts every layer exactly at 40 percent.">
  <path class="ax" d="M80 30 V160 H612"/>
  <text class="t-dim" x="80" y="22">layers at this warm residency</text>
  <path class="c-rise draw" pathLength="100" d="M132 159.9 L158 159.5 L184 157.9 L210 153 L236 141.9 L262 121.7 L288 94.8 L314 70.1 L340 60 L366 70.1 L392 94.8 L418 121.7 L444 141.9 L470 153 L496 157.9 L522 159.5 L548 159.9"/>
  <rect class="hatch" x="80" y="30" width="117" height="130"/>
  <path class="ax thin" d="M197 30 V160"/>
  <path class="c-flat draw" pathLength="100" d="M340 160 V44"/>
  <text class="t-rise" x="86" y="176">unluckiest of 48</text>
  <text class="t-rise" x="86" y="192">≈ 29%</text>
  <text class="t-flat" x="352" y="52">stratified — every layer, exactly 40%</text>
  <text class="t-rise" x="470" y="112">global draw</text>
  <text class="t-mono" x="210" y="176" text-anchor="middle">30%</text>
  <text class="t-mono" x="340" y="176" text-anchor="middle">40%</text>
  <text class="t-mono" x="470" y="176" text-anchor="middle">50%</text>
</svg>
<figcaption>Binomial(128, 0.4) across 48 layers. Same fill, same immutability,
one different line in the sampler — and the variance goes to zero.</figcaption>
</figure>

Now that variance matters far more here than it would for an ordinary cache, and
for two reasons that are both properties of *this* architecture rather than of
caching in general.

First, the fill is **immutable** — so an unlucky layer is slow on every single
forward for the entire process lifetime, not just for one pass. It never heals.

Second, and this is the one people miss, the wave is **sequential**. Layer *N+1*
cannot start until layer *N*'s experts land. Which means lucky layers cannot
compensate for unlucky ones.

**The sweep runs at its worst stage, not at its mean.**

A cache with an eviction policy would heal an unlucky draw over time. An immutable
one never will. And a pipeline that is only ever as fast as its slowest stage
doesn't average out — it just waits.

Stratifying costs one different line in the sampler and deletes the entire failure
mode.

## Bug two: the fight over one cache

Now here's the failure that took me longest to see properly.

A decode step touches a handful of experts per layer. Cheap as chips — top-8
across 48 layers, so **384 expert references** and most of them already resident.
A prefill forward over a long prompt touches **every expert in every layer**, in
one shot. All 128, all 48 layers. **6,144 references.**

Sixteen times the working set, arriving in a single forward.

Run those two alternately — which is what any straightforward scheduler does, and
what mine did — and the big prefill loads all of it into the cache and evicts the
tail.

And that tail is *exactly* the working set the live decoding conversations were
riding. So the next several decode steps run stone cold, re-streaming their
experts from RAM at a fraction of steady state, while your users sit there
wondering what happened.

<figure class="fig">
<svg viewBox="0 0 640 250" role="img" aria-label="Taking turns: a prefill burst evicts the decode working set and throughput collapses until it re-warms. Continuous waves keep it flat.">
  <path class="ax" d="M56 16 V150 H612"/>
  <text class="t-dim" x="56" y="10">decode throughput</text>
  <path class="c-rise draw" pathLength="100" d="M56 40 H210 L232 132 L250 136 C300 120, 340 60, 400 44 H612"/>
  <rect class="hatch" x="210" y="16" width="40" height="134"/>
  <text class="t-rise" x="214" y="172">prefill lands here</text>
  <text class="t-rise" x="272" y="192">↓ everyone else goes cold, then slowly re-warms</text>
  <path class="c-flat draw" pathLength="100" d="M56 62 H612"/>
  <text class="t-flat" x="470" y="82">continuous waves</text>
  <text class="t-dim" x="612" y="176" text-anchor="end">time →</text>
</svg>
<figcaption>The dip is real, it's large, and a scheduler that alternates the two
kinds of work cannot avoid it — only recover from it.</figcaption>
</figure>

### This is sequential flooding, and the known fix doesn't work

Now, databases solved this exact problem decades ago and I went looking for their
answer first.

It's called **sequential flooding**: one big table scan walks through the buffer
pool, touches every page exactly once, and evicts an entire hot working set that
dozens of other queries were using. Same shape as mine, one layer down. And the
fix is well established — make the cache *scan-resistant*. LRU-K, 2Q, ARC. They
all work by recognising that a page touched once, in a sweep, is worth less than a
page touched repeatedly, and refusing to promote it.

That doesn't transfer here. And the reason it doesn't is the whole problem.

A scan-resistant policy works because the scan's pages are **genuinely low value**
— the query reads each one once and never comes back. But a prefill's experts
aren't low value. The prefill *needs* every one of them, right now, to produce a
correct forward pass. There's no signal to demote, because nothing about the
access is speculative or wasteful. Both workloads want their experts equally
badly.

So you cannot referee this by ranking the requests. Both sides are right.

<div class="key">
<h4>Which is when it clicked</h4>
<p>Decode and prefill have <b>the same access pattern</b>. Both of them sweep
layers 0 → N in order. They aren't fighting because they want different things —
they want the identical thing, in the identical order.</p>
<p>They're only fighting about <b>rate</b>. Decode wants to go round the loop
constantly with a small set. Prefill wants to go round once with a big one.</p>
<p>And turn-taking is what converts a difference in <i>rate</i> into a conflict
over <i>capacity</i>. That's not a property of the workloads. It's a property of
the scheduler I wrote.</p>
</div>

The usual patch is a **recovery cooldown**. After a big prefill, run decode-only
waves until throughput comes back, then allow the next prefill through.

I wrote one. It works, in the sense that a tourniquet works — and it has the
failure mode tourniquets have. It doesn't reduce the damage, it just spreads the
recovery out and makes it somebody else's problem: prefills now queue behind
cooldowns, ingest throughput becomes a function of how much dialogue is happening,
and interactive latency goes **bimodal**. Most steps fine, some steps terrible,
which users notice far more than a uniformly slower system.

Every knob I added made one of those worse.

That's the smell that tells you you're managing a structural problem rather than
fixing one. Decode and prefill are fighting over one resident-expert cache
**because they never share the layer traversal** — and if the traversal is the
thing they both want, then the traversal is the thing to share.

Fix that and there's nothing left to referee.

## Stop taking turns

The word "wave" was quietly doing two jobs:

- The **layer wave** — walking `layer 0 → 1 → … → N-1`. This is where the
  expense lives: per layer, prefetch that layer's experts, run one grouped GEMM
  over whatever tokens happen to be present.
- The **inference wave** — a unit of output. One decode token, or one completed
  prefill.

Conflate those two and you're forced to decide, on every single pass, whose work
this one belongs to. Separate them and the question simply evaporates.

Decode sweeps every layer on every wave, one token per sweep. A large prefill
**creeps** through the layers in the background at a throttled rate, riding along
in the same grouped GEMM as whatever decode happens to be doing.

And the whole novelty is two lines of intent. Lift the prefill's inter-layer
buffers out of the per-forward scope so they survive across waves. Give each
cursor its own layer-advance rate. That's the lot. The attention kernels, the slot
allocator, the sampler, the expert streaming — all untouched.

That's usually the sign you've found the right change, incidentally. It's small
and it deletes a whole category of problem rather than managing one.

### What the sweep actually looks like

A wave is not one forward any more. It is up to three, split at the creep's
window:

<figure class="fig">
<svg viewBox="0 0 640 236" role="img" aria-label="Three successive waves. The decode band spans all 48 layers every wave; the creep window advances three layers per wave and its residual is carried between them.">
  <text class="t-dim" x="16" y="30">layer</text>
  <text class="t-mono" x="70" y="30">0</text>
  <text class="t-mono" x="600" y="30" text-anchor="end">47</text>
  <text class="t-lbl" x="16" y="56">wave t</text>
  <rect class="box" x="70" y="44" width="530" height="16" rx="3"/>
  <rect class="box weights" x="70" y="44" width="33" height="16" rx="3"/>
  <text class="t-lbl" x="16" y="118">t+1</text>
  <rect class="box" x="70" y="106" width="530" height="16" rx="3"/>
  <rect class="box weights" x="103" y="106" width="33" height="16" rx="3"/>
  <text class="t-lbl" x="16" y="180">t+2</text>
  <rect class="box" x="70" y="168" width="530" height="16" rx="3"/>
  <rect class="box weights" x="136" y="168" width="33" height="16" rx="3"/>
  <path class="link" d="M103 60 V106"/>
  <path class="link" d="M136 122 V168"/>
  <text class="t-flat" x="112" y="80">residual held</text>
  <text class="t-flat" x="145" y="142">residual held</text>
  <text class="t-dim" x="180" y="212">grey = decode, every layer, every wave</text>
  <text class="t-flat" x="180" y="228">blue = the creep window, ⌈N/R⌉ layers wide</text>
</svg>
<figcaption>The creep pauses mid-model. Its residual stream — the activations
between layers — is held across waves so decode can overtake it, then handed
back when the window resumes.</figcaption>
</figure>

`[0, cursor)` carries decode alone. `[cursor, win_end)` carries decode **and** the
creep, co-batched in one grouped GEMM so they share the expert load. `[win_end, N)`
is decode alone again. The forward returns its residual in caller order
`[decode | creep | glue]`, so the boundaries split it by contiguous group: the
creep's slice is stored whole, the rest continues.

Because decode re-touches every layer's experts continuously, its hot set is
never the tail that gets evicted. The cold-decode-after-prefill penalty isn't
recovered from faster — it stops existing.

### The throttle is one number

Each layer declares a `decode_priority`, which is really a decode-to-prefill
airtime ratio `R` — how many decode tokens land per completed prefill. With `N`
layers, the creep window is `ceil(N/R)` layers wide:

| priority | R | window at N=48 | for |
|---|---:|---|---|
| Low | 1 | all 48 layers in one wave | bulk ingest, no decode to protect |
| Normal | 16 | 3 layers/wave, completing over 16 sweeps | background work |
| High | 64 | 1 layer/wave | dialogue, latency protected |

When no interactive decode is running at all, `R` collapses to 1 and the prefill
drains at full speed — there is nothing to shield. One knob, resolved per wave
from the live decode set rather than configured.

Boundary gap-fill — "glue" — rides the same cursor, because it *is* part of
getting a context prefilled; only the attention kernel at each layer differs.

### The part that bites

Holding a residual across waves means the creep's **membership cannot change
mid-sweep**. The held tensor is indexed by position in a fixed member order, and
a member joining or leaving between waves makes every subsequent layer read the
wrong rows — which produces exactly the failure you'd least like: not a crash,
but subtly wrong activations, and a sampler that emits token 0 forever.

So the cohort re-forms only at sweep boundaries, and a mismatch between the held
residual's width and the group's is caught by comparison and **recovered from**
rather than asserted on. The creep restarts at layer 0, which is safe because a
prefill member re-feeds its whole token block and only advances its slot offset
at completion — re-running the layers it had already done rewrites the same K/V
at the same positions.

And that is deliberately **not** an assert. An assert here took the whole daemon
down once, and the lesson stuck: a panic on the scheduler thread is not a failed
wave, it's a failed process. Recover, log loudly, carry on.

<div class="key">
<h4>The general lesson</h4>
<p>When two workloads fight over a shared cache, the fix is rarely a smarter
eviction policy or a fairer scheduler. It is noticing you built a structure that
forces them to take turns, and removing the turn.</p>
</div>

## The wave makes Belady computable

Now here's the thing that fell out of continuous waves that I genuinely did not
expect, and it's the property I'd keep if I had to throw the rest away.

Cache eviction is normally a **prediction** problem. Bélády's algorithm — evict
the line whose next use is furthest away — is provably optimal and completely
unimplementable, because you don't know the future. Every real policy you have
ever used, LRU, LFU, ARC, CLOCK, all of them, is a proxy for a distance nobody can
measure.

So here's the fun part.

In a continuous layer wave, **you can measure it.** The access pattern is a cycle:
layer 0, 1, 2, … 47, 0, 1, 2… forever. So the distance from the current layer to
any resident expert's next use is a subtraction with a wrap:

```rust
fn forward_distance(&self, layer: usize, current_layer: usize) -> usize {
    if layer >= current_layer {
        layer - current_layer
    } else {
        self.num_moe_layers - current_layer + layer
    }
}
```

That's it. That's the whole thing. **Not an estimate.**

The layer about to be routed is distance 0. The layer just executed is distance
`n-1` and won't be touched again for a full pass. So the eviction key carries a
position factor that falls with wrapped forward distance — Bélády's direction,
computed exactly, derived from the *shape of the loop* rather than from history.

Sixty years of cache research spent approximating a number, and the architecture
just hands it to you because the access pattern happens to be a cycle.

The full key is three multiplicative terms, and each one answers a different
question:

<figure class="fig">
<svg viewBox="0 0 640 236" role="img" aria-label="The eviction key as three multiplied terms: decayed access frequency, reload cost by tier, and a position factor falling with wrapped distance to next use.">
  <rect class="box weights" x="26" y="44" width="164" height="96" rx="8"/>
  <text class="t-lbl mid" x="108" y="72">score</text>
  <text class="t-mono mid" x="108" y="94">hit +1.0</text>
  <text class="t-mono mid" x="108" y="110">predicted +0.3</text>
  <text class="t-mono mid" x="108" y="126">decay ×0.85</text>
  <text class="big mid t-flat" x="212" y="102">×</text>
  <rect class="box weights" x="238" y="44" width="164" height="96" rx="8"/>
  <text class="t-lbl mid" x="320" y="72">reload_cost</text>
  <text class="t-mono mid" x="320" y="98">warm-backed → 1.0</text>
  <text class="t-mono mid" x="320" y="118">cold only → 4.0</text>
  <text class="big mid t-flat" x="424" y="102">×</text>
  <rect class="box weights" x="450" y="44" width="164" height="96" rx="8"/>
  <text class="t-lbl mid" x="532" y="72">position</text>
  <text class="t-mono mid" x="532" y="98">[0.5 … 1.0]</text>
  <text class="t-mono mid" x="532" y="118">by wrapped distance</text>
  <text class="t-dim mid" x="108" y="168">how much do we</text>
  <text class="t-dim mid" x="108" y="184">want this?</text>
  <text class="t-dim mid" x="320" y="168">what does being</text>
  <text class="t-dim mid" x="320" y="184">wrong cost?</text>
  <text class="t-dim mid" x="532" y="168">when will we</text>
  <text class="t-dim mid" x="532" y="184">find out?</text>
  <path class="ax thin" d="M26 206 H614"/>
  <text class="t-flat mid" x="320" y="228">lowest product is the victim — temperature, tier and time, in one multiply</text>
</svg>
<figcaption>Three questions, three terms, one number. Most eviction policies only
ever ask the first one.</figcaption>
</figure>

The `reload_cost` term is the one the old two-tier cache could not have had.
Under an exclusive-or, every expert not in VRAM was in pinned RAM by
construction, so every reload cost the same and the score could be pure
temperature. Under three tiers the two outcomes differ by an order of magnitude:
an expert the warm tier holds comes back as a **~116 µs H2D** from pinned memory;
one it doesn't comes back as a page-cache-bypassing **2.9 MB positioned NVMe
read, near a millisecond**.

And weighting by it makes the cache converge on a shape nobody designed: **VRAM
drifts toward holding the experts that are expensive to re-acquire, the warm tier
covers the ones that are cheap, and the experts that churn are the ones whose
churn is cheapest.** That is emergent from one multiplication.

### Except the honest ratio is the wrong constant

The reload times differ by about 8×. The constant in the code is **4**, and that
is measured, not derived.

At 8 the term stops being a tilt on the ordering and starts replacing it.
Cold-only experts get held past the point their temperature justifies, the
cache's hit rate *falls* (44.8% → 44.3% on one config), and the further cold
reads it saves cost more than they buy. Every configuration was slower at 8 than
at 4; the widest lost 55 t/s.

At 4, frequency still decides between experts of equal reload cost, and a truly
cold cold-backed expert still loses to a hot warm-backed one. What it stops is
the policy evicting the expensive one when the two are otherwise close — which,
at the margin an eviction scan actually operates on, is most of them.

So here's a rule I'd hand anyone tuning anything: **a physically-derived constant
is the starting point for a sweep, never the answer to it.**

The physics tells you the *sign* of the term. Only measurement tells you the
magnitude — because the term isn't operating in isolation, it's competing with
everything else in the key. Derive it from first principles and ship it and you
will be wrong by exactly the amount the rest of your system matters.

### One thing that is never evicted

The first three layers are pinned unconditionally. They run first every pass
with **zero compute ahead of them to overlap a DMA against**, so evicting them
guarantees a cold miss with maximum stall. On a wide card that reservation is
noise.

It is not always noise. Qwen3.5-35B-A3B carries 256 experts in every one of its
40 layers, and a 16 GB card affords about 567 slots: three pinned layers would
want 769 before serving a single token, and two would leave the working layer 55
slots against a possible 256. So the pinned depth is derived from capacity rather
than fixed — pin what you can afford, and shed pinned layers before the eviction
filter can starve.

## Bug three: optimising the wrong number

Now, prefetch. Expert routing isn't random — what a layer wants correlates
strongly with what the previous layer chose. So predict it.

**The published approach is to run the current hidden state through the next
layer's gate.** Pre-gated MoE, HOBBIT, AdapMoE, Mixtral-Offloading, ExpertFlow —
they all exploit cross-layer activation similarity this way, and it reaches about
96% top-1 accuracy. That is a strong result and it is the right approach when you
can afford it. It also costs a real gate and a linear over live activations, per
layer, per token, forever.

I deliberately didn't do that.

The predictor here is **ID-only**. A co-occurrence matrix over historical
expert-routing IDs. No hidden states, no extra GEMM, nothing. For each adjacent
layer pair it accumulates an `[E × E]` count table online and scores candidates by
pointwise mutual information with a marginal discount — which demotes the
globally-popular experts in favour of the ones *specifically* implied by the
current routing. It costs nothing to evaluate and it's trivially shared across a
whole batch, because expert IDs are identical for every token that routed to them.

And the question worth asking isn't "is it as accurate as the gate?" Of course it
isn't. The question is:

> Is a **zero-compute** model good enough, once you pair it with a well-tuned
> cache?

That turns out to have a far more interesting answer than I expected. But getting
to it meant throwing away the metric first.

### Hit rate is the wrong objective

The obvious move is to tune for hit rate. Sweep the eviction churn and the
prefetch fan-out `K`, find the configuration that misses least, ship it, go home.

That sweep says 5% forced churn with `K=4`.

And that configuration is close to the **worst one available**.

Not every miss costs the same. A **hard** miss stalls the pipeline for a full
PCIe round trip. A **soft** miss is one the prefetcher already has in flight, and
it overlaps with work. Weight them the way the hardware does —

> stall cost = 10 · hard-miss + 1 · soft-miss

— and sweep again:

| churn ＼ K | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|
| 0% | 94.2 | 89.1 | 86.1 | 84.7 |
| 1% | 96.7 | 88.7 | 84.2 | 82.5 |
| 2% | 112.9 | 97.1 | 85.2 | **80.7** |
| 3% | 132.5 | 110.6 | 91.9 | 81.0 |

The hit-rate optimum scores about **169**. The cost optimum is 2% forced churn
with `K=16`, at **80.7** — less than half the stall cost. The hard-miss rate
falls from 16.6% to 7.8%, for no extra hardware, purely by changing which number
you minimise.

Read the table by column and by row, because they say opposite things.

**Deeper prefetch always helps**, saturating around 12–16. It converts hard
misses into soft ones, and soft misses are nearly free because the bus was going
to be busy anyway. It is bandwidth-cheap precisely *because* it is headroom-
limited: it can only fill slots that are already free.

**Forced churn is mostly pure cost**, and this is where the hit-rate tuning went
most wrong. Every slot you evict on purpose is a demand expert that frequently
gets re-requested — a hard miss at 10× weight, plus a reload transfer. The soft
misses that headroom enables do not pay for it. So churn goes *down*, from 5% to
2%: enough to seed prefetch and no more. Both moves are large, and they point in
opposite directions.

So there it is. Hit rate measures how often you were *right*. Stall cost measures
what being *wrong* costs you.

Only one of those is a thing your users can feel, and it isn't the one everybody
reports.

## Bug four: the simulation was right and the answer was wrong

Now for my favourite mistake in the whole project.

The sweep above says `K≈16` is optimal at *every* bandwidth budget it tested.
Production caps prefetch at **8**.

And this is the failure mode genuinely worth writing down, because the simulation
wasn't sloppy at all. It was faithful — to an assumption that stops holding.

The model treats prefetch as bandwidth-cheap because it is headroom-limited: it
only fills free slots, so a wider fan-out costs bandwidth and nothing else. On a
card whose resident set is genuinely **capacity-bound**, that assumption
collapses. There is no standing headroom. Every speculative load displaces an
expert the wave will need later in the same sweep, so a wider fan-out doesn't
remove misses — it *moves* them, from a layer you predicted to a layer you
didn't.

Measured, not argued: demand-width prefetch **doubled the glue wave's wall
time** through eviction thrash and pipe serialisation. Re-measured at `K=24` on
the 72 GB card's ~3,500-slot resident set, prediction precision fell from **92%
to 53%** and single-session decode lost **~14%**. The knee is a property of the
predictor's tail — the candidates past rank 8 are mostly wrong — **not** of card
capacity. A bigger card does not buy you a deeper prefetch.

So the fixed `K` is gone, replaced by something better shaped:

- An expert is prefetched only if **some active source routes to it at ≥50% of
  that source's own strongest successor rate.**
- Relative, not absolute, so it survives a top-8 router diluting every
  conditional.
- Per-source, so one sticky pair can't raise the bar for everybody else's
  successors.
- Capped at 8, and the cap is deliberately fixed rather than scaled with demand.

The effect is that **depth tracks demand diversity**. A homogeneous batch — one
prompt fanned out across N sessions — implies a couple of sticky successors, so
prefetch stays shallow and precise. Diverse demand implies many, each source
nominating what it genuinely implies, and the cap bounds the total volume.

The same offline work recommended a two-tier predictor: a frozen cross-prompt
prior blended with a live per-session matrix, to fix the cold-start problem.
That didn't ship either — because in production **the process is the session**.
The matrix accumulates for the life of the daemon and converges within a few
thousand tokens. There was no cold start to solve. The offline harness had
invented one by evaluating across 21 separate prompts.

<div class="key">
<h4>What a simulation is for</h4>
<p>It told me the objective was wrong (hit rate → stall cost). That was worth
everything, and it survived contact with hardware completely.</p>
<p>It also told me a parameter value, and that value was wrong, because the
parameter's cost model depended on an assumption the real machine doesn't
satisfy. <b>Trust a simulation for the shape of the answer, never for the
constant.</b></p>
</div>

## And sometimes the predictor doesn't matter at all

And then push the batch wide enough and something very clean happens. The union of
experts a layer routes to grows toward *all of them*.

At which point there is nothing left to predict, and the predictor — mine
included, after all that work — is a rounding error.

The simulation put the crossover at a batch of about 256 — and here's the useful
part — the saturation batch is **residency-independent**. It's a property of the
workload, not of the card. Past it, wave time is simply
`total_stream / PCIe_bandwidth` regardless of *which* experts you prefetched,
because the streaming volume dwarfs the compute window entirely.

A deterministic policy — prefetch the next layer's whole missing set, evict
behind, ride the wave — lands within **1–4%** of the fully tuned optimum. All that
prediction machinery, worth three percent.

So there are three regimes, and the engine has to be two different programs:

| regime | what binds | policy |
|---|---|---|
| Working set fits | compute | prefetch almost nothing |
| The knee | spare bandwidth | **prediction pays** — this is where the predictor earns its keep |
| Saturated | bandwidth | stream the whole missing set, don't predict |

That split is visible in the codebase as two separate mechanisms. The predictor
serves the knee. A **separate streamer thread** serves saturation: during a
prefill-width wave the next layer will route most of its experts, so the win
isn't prediction, it's turning the layer's expert traffic from on-demand stalls
into copy-stream DMA overlapped with the current layer's compute.

That thread exists for a measured reason. Running the same loads on the pipeline
thread **regressed bulk throughput from 437 to 416 t/s** — the next layer's work
request queued behind ~170 loads instead of overlapping them. So the division of
labour is strict: the pipeline thread owns every piece of mutable cache state
with `&mut self` and no mutex on the hot path; the streamer thread moves **bytes
only**, on its own CUDA stream, and never touches the cache bookkeeping at all.

## Then batch the sessions

The last piece, and the one that makes the whole exercise pay for itself.

Step 64 conversations through the layers **coherently**. At each layer the wave
needs the *union* of experts its tokens route to — and here's the bit that
matters: a weight loaded over PCIe is shared by every batched token routing to it,
in one grouped GEMM.

So the PCIe cost is **per unique expert per wave**, amortised across every single
session riding in it.

Which quietly inverts the usual scaling story. `aggregate = B / wave_time`, and
once the per-wave streaming saturates, `wave_time` stops growing while `B` just
keeps going:

<figure class="fig">
<svg viewBox="0 0 640 216" role="img" aria-label="Modelled aggregate throughput against batch size: a dip at the cache-fitting knee near batch 8, then near-linear growth to batch 1024.">
  <path class="ax" d="M60 20 V170 H612"/>
  <text class="t-dim" x="60" y="14">aggregate t/s (log scale)</text>
  <rect class="hatch" x="150" y="20" width="90" height="150"/>
  <path class="c-flat draw" pathLength="100" d="M60 144.7 L114 124.4 L168 127.8 L222 142.6 L276 136.2 L330 122.9 L384 106.9 L438 90.4 L492 72.6 L546 54.1 L600 35.2"/>
  <text class="t-rise" x="156" y="190">the cache-fitting knee</text>
  <text class="t-flat" x="470" y="46">1313</text>
  <text class="t-mono" x="60" y="186" text-anchor="middle">1</text>
  <text class="t-mono" x="222" y="186" text-anchor="middle">8</text>
  <text class="t-mono" x="384" y="186" text-anchor="middle">64</text>
  <text class="t-mono" x="600" y="186" text-anchor="middle">1024</text>
  <text class="t-dim" x="612" y="206" text-anchor="end">batch size →</text>
</svg>
<figcaption>Modelled at 60% VRAM residency and 8 GB/s — a PCIe-limited upper
bound, since the compute floor is held constant. The shape is the point: it gets
worse before it gets better.</figcaption>
</figure>

Now note the dip, because it's a trap.

Aggregate throughput is **not monotone** in batch size. Around batch 8 the working
set stops fitting, the hit rate collapses from the low 90s to the low 70s, and
aggregate throughput goes *backwards* before streaming saturation starts paying
you.

So anybody tuning this by nudging the batch size upward and watching the number go
down would stop right there and conclude — reasonably, and completely wrongly —
that batching doesn't help. The win is on the other side of a valley, and you only
find it if you were willing to walk through the bad bit.

Per-session latency falls the whole way, 24.6 t/s down to 1.3. That's the trade
and it's deliberate: throughput for latency, with `decode_priority` as the dial
deciding which conversations are exempt from it.

## And here is the measured version

Right, enough modelling. Two machines, real runs.

And a note on the columns, because the harness names are misleading and I'd rather
you weren't. **Prefill** is the aggregate rate across every context in the batch —
and so is **decode**. Neither of them is a per-session figure. The per-session
column below is just the division, done for you.

First, the laptop. Qwen3-30B-A3B on an RTX 4090 Mobile, 16 GB. A machine you can
close and put in a bag, running a 30-billion-parameter mixture-of-experts that
does not fit in its VRAM and never will.

| KV | contexts | prefill t/s | decode t/s | decode per session | peak KV tokens |
|---|---:|---:|---:|---:|---:|
| F16 | 1 | 466.9 | 6.6 | 6.6 | 626 |
| BF16 | 1 | 665.1 | 9.5 | 9.5 | 626 |
| BF16 | 10 | 2,128.6 | 102.4 | 10.2 | 6,310 |
| Q8_0 | 20 | 2,314.4 | **166.1** | 8.3 | 12,620 |
| Q4_0 | 20 | 2,294.3 | 162.5 | 8.1 | 12,620 |
| BF16 | 1 *(same config, later in the sweep)* | 670.4 | **18.4** | 18.4 | 626 |

**Twenty sessions for the price of one.** Aggregate decode goes from 9.5 to 166.1
t/s — **17.5×** — while each individual session still runs at 8.3 t/s, which is
**87% of the speed it got with the whole card to itself**. That is the
amortisation argument, measured: the sessions are not competing for the bus,
they are sharing a load that was going to happen anyway.

Prefill scales too, but only 3.5×, and the gap between those two multipliers is
the whole design. Prefill was already batching work internally — one long prompt
saturates the machine on its own. Decode is where a single session leaves the
card idle waiting on PCIe, and decode is where filling that idle time pays 17×.

### The same row, twice

Look at the first `BF16 / 1 context` row and the last one. Identical
configuration. Identical context count. **9.5 t/s versus 18.4 t/s — the same work
ran 94% faster the second time**, purely because by then the expert cache was
warm.

I did not stage that. It's an artifact of the sweep order, and it is the entire
thesis of this post sitting in two rows of a table: on a streamed MoE, *what is
resident* matters roughly as much as everything else combined. Half your
throughput is a cache state. That is why so much of the work above is about
eviction policy and residency rather than about kernels — and why a scheduler
that lets a big prefill flush the decode working set is not making a small
mistake.

### Compression is free until it isn't

The same laptop, two contexts, walking the adaptive KV compression ladder:

| level | compression | prefill t/s | decode t/s |
|---|---:|---:|---:|
| C0 | 1.98× | 1,236.3 | 30.2 |
| C1 | 2.54× | 1,235.5 | 31.3 |
| C2 | 2.74× | 1,219.7 | 31.5 |
| C3 | 2.99× | 1,227.4 | 31.2 |
| C4 | 3.40× | 1,235.3 | 31.4 |
| C5 | 3.67× | 1,225.5 | 30.3 |
| C6 | 4.17× | 1,234.6 | 27.6 |
| C7 | 4.24× | 1,236.2 | 28.3 |
| C9 | 5.31× | 1,230.9 | 27.1 |

Compression ratio nearly **triples** from C0 to C9. Prefill throughput moves by
**1.4% across the entire ladder** — noise. Decode gives up 10%.

That is what a compression scheme should look like on a bandwidth-bound machine:
the bits you don't move are bits you don't pay for, and the selection work
disappears into time the GPU was going to spend waiting anyway. There is a whole
[separate argument](/blog/palquant-per-block) about whether those ratios cost you
*quality*; on throughput they cost essentially nothing.

### And the big one, on a big card

DeepSeek-V4-Flash at Q4 — 284 billion parameters — on an RTX PRO 5000 Blackwell,
72 GB:

| contexts | prefill t/s | decode t/s | decode per session | peak KV tokens |
|---:|---:|---:|---:|---:|
| 1 | 222.6 | 15.4 | 15.4 | 670 |
| 4 | 615.7 | 27.1 | 6.8 | 2,722 |
| 8 | 833.6 | 44.8 | 5.6 | 5,404 |
| 16 | 1,041.3 | **62.8** | 3.9 | 10,792 |
| 1 *(same config, later in the sweep)* | 242.4 | 20.8 | 20.8 | 670 |

Same shape, different position on it. Aggregate decode climbs 4.1× across 16
contexts and aggregate prefill 4.7×, but per-session decode falls from 15.4 to
3.9 — a quarter of what one session gets alone, where the laptop retained 87%.

Which is the modelled curve telling the truth. The 30B on 16 GB is deep in the
streaming regime, where the union of routed experts has saturated and an extra
session mostly rides loads that were already in flight. A 284B model at 16
contexts on a 72 GB card is nearer the **knee**: there is still enough headroom
that each new context pulls in experts nobody else asked for, so it pays for
them. More VRAM did not move the machine past the knee — it moved the knee.

And the warm-cache effect shows up here too: the repeat run is 35% faster on
decode for the identical configuration.

## Where this sits against everything else

Let me be straight about what these numbers are, because the obvious comparison is
the wrong one and I don't want anybody wasting their time on it.

This is not a raw tokens-per-second claim. Put a model on a card where it *fits*
and a well-tuned serving stack will beat my single-session figures comfortably.
Those stacks are excellent, they are built by people who know exactly what they're
doing, and single-session speed on resident weights is a competition I am not
entering.

The axis that matters is different: **how big a model can you run, at real
concurrency, on the hardware you've actually got?**

And on that axis the field runs out of road much earlier than people expect.

### Offloading is not the problem. Offloading *per session* is.

Every serious stack can push weights off the card in some form. What separates
them isn't whether they offload — it's what the offload costs as you add users.

Load a weight on demand for one session and that transfer serves one session. Two
sessions, two transfers. The bus traffic is **O(sessions)**, so throughput per
user falls roughly as fast as you add users, which is why "CPU offload" has the
reputation it has. It isn't a slow implementation. It's a scaling class.

Wave batching changes the class. Step every session through the layers
**coherently** and each layer needs the *union* of experts its tokens want — so a
weight crossing the bus serves every session that routed to it, in one grouped
GEMM. Bus traffic becomes **O(unique experts per wave)**, which stops growing long
before your session count does.

That's the whole difference, and you can see it in the table above. Going from 1
session to 20, aggregate decode rises 17.5× while per-session decode holds at
**87%**. Under per-session offload that column falls off a cliff. Under wave
batching the twentieth conversation is very nearly free.

Same hardware, same bus, same 2.9 MB per expert. Different complexity class.

### Which is a datacentre argument, not a laptop one

I keep framing this around a 16 GB laptop because that's where the constraint
forced the design. But the economics land far harder at the other end of the
market.

The standard answer to a large mixture-of-experts model is to shard it across
GPUs, and past a certain size, across *nodes*. Inside a node that's fine — NVLink
runs at hundreds of gigabytes per second and the sharding is close to free. Across
nodes you're on the network, and MoE's all-to-all is exactly the traffic pattern
that punishes you for it.

So look at what that network actually gives you:

| link | bandwidth |
|---|---:|
| PCIe 4.0 ×16 | ~31 GB/s |
| **400 Gb InfiniBand / Ethernet** | **50 GB/s** |
| **PCIe 5.0 ×16** | **63 GB/s** |
| 800 Gb | 100 GB/s |
| NVLink 4 (per GPU, each way) | 450 GB/s |

Read the two bold rows again.

**A single card in a PCIe 5.0 slot has more host bandwidth than a 400 Gb fabric
has network bandwidth.** People are building racks — multiple nodes, InfiniBand
NICs, switches, the power and cooling to match — around an interconnect that is
*slower than the bus already sitting inside one server*.

And the bus doesn't need a second node, a second set of NICs, or a network team.

<div class="key">
<h4>What that buys, concretely</h4>
<p>DeepSeek-V4-Flash is a <b>284-billion-parameter</b> model. At Q4 that's roughly
150 GB of weights — comfortably more than twice a 72 GB card.</p>
<p>The measured table above runs it on <b>one</b> RTX PRO 5000: 16 concurrent
contexts, 1,041 t/s prefill, 62.8 t/s aggregate decode, 10,792 tokens of cache.
Half the model is somewhere else at any given moment and the wave doesn't
care.</p>
<p>That is a model class that normally implies multi-GPU sharding, running usefully
on a single card in an ordinary server.</p>
</div>

Which reframes the whole purchasing question. The reason you buy 180 GB cards and
400 Gb fabric is to make a model **fit**. If a model no longer has to fit, then
capacity stops being a hard requirement and becomes a speed dial — and you can buy
exactly as much of it as your throughput target actually needs, rather than as
much as your parameter count demands.

Model size stops being a procurement decision.

### So how close is this to having it all resident?

That's the fair question, and here's the honest answer, because it isn't a
tokens-per-second one.

Resident weights and streamed weights converge when the streaming stops being what
limits you. And you can see that happening in the numbers: prefill runs **2,128
t/s at 10 contexts and 2,314 t/s at 20** — twice the sessions for 8.7% more
throughput. That flatness is not the bus scaling. That's the bus having *stopped*
being the constraint, and something else — compute, scheduling, launch overhead —
taking over.

Which is the same thing that limits a fully-resident deployment.

So below batch 16 or so, streaming is a compromise and you feel it. Above it,
you're in the regime where you're bounded by the same things everyone else is
bounded by, and the weights being elsewhere has stopped mattering.

You don't get resident *performance*. You get resident *scaling behaviour*, which
is the property that was worth the money in the first place.

<div class="key">
<h4>And it's doing that while compressing</h4>
<p>The 20-session row runs Q8_0 KV. Walk the adaptive ladder in the table above and
compression goes from <b>1.98× to 5.31×</b> while prefill throughput moves by 1.4%
across the entire range — noise.</p>
<p>So the concurrency isn't bought by spending memory recklessly somewhere else.
The KV cache is compressed 2–5× <em>at the same time</em>, and that's what pays for
the session count. Whole argument of its own, and it lives
<a href="/blog/palquant-per-block">over here</a>.</p>
</div>

For smaller models the same machinery goes much further again: the Llama-3.2-3B
configuration on that 16 GB laptop runs up to **256 sessions** holding 168K cached
tokens between them. And for scale on concurrency generally — the closest
[published study](https://arxiv.org/abs/2512.23029) I can find had standard
frameworks degrading beyond **two** concurrent users on an RTX 5090, a card with
twice the laptop's VRAM.

## What the bus was always going to cost

Every single one of those was the same mistake wearing a different costume.

The exclusive-or was a data model that made a copy mandatory — so the fix was a
type, not a policy. The turn-taking scheduler was a structure that forced two
workloads to fight — so the fix was removing the turn, not refereeing it better.
The hit-rate sweep optimised a number that isn't the one users feel. The prefetch
depth was a constant borrowed from a model whose assumptions the hardware doesn't
satisfy.

And notice what's absent from that list. **Not one of them was fixed by making
anything faster.** Every one was fixed by deleting work that should never have
existed in the first place — 68 GB of copies, a recovery cooldown, a churn budget,
half a prefetch fan-out.

That's the lesson I'd hand anybody fighting a bus. You are not going to out-engineer
physics. Go and find the work you're doing that didn't need doing, because there is
always more of it than you think, and it's always somewhere you stopped looking
years ago because that part "works".

The bus is still the bottleneck. It always was, and no amount of cleverness makes
2.9 MB cross PCIe faster than PCIe crosses it.

What changed is what you get for each crossing. On a 16 GB laptop, one expert load
now serves twenty conversations at 87% of the speed any one of them would get
alone — and while it's in flight, the GPU has something else to do.

Stop trying to win the fight with the bus. Have fewer of them, and make each one
count for more.

That's the whole game with streamed MoE. You are not trying to win a fight with
the bus. You are trying to have fewer of them, and make each one count for more.
