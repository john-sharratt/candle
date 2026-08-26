---
title: "The journal is king... long live the king"
date: 2023-11-01
tint: warn
tags: [distributed-systems, storage, architecture]
summary: >-
  Every serious data store on the planet keeps a redo log and treats it as the
  source of truth. Everything else is a copy in a costume. Which means the
  phrase "data gravity" is wrong, and I'm going to replace it.
---

<figure class="shot hero">
<img src="/img/blog/journal.jpg" alt="A crown resting on a closed leather-bound book, lit by candles" width="800" height="800">
</figure>

This hidden detail of distributed systems is often overlooked until one starts
digging into the guts of data stores, and then this silent workhorse keeps
reappearing from behind the curtains. Apparently the "master elected log journal"
is a bit more important than a small detail in the operations manual.

Perhaps it's about time we admit that the journal **is** the state.

Hear me out and I'll explain.

## The list is the argument

Take these technologies:

ext4, ZFS, Postgres, NTFS, Oracle DB, SQL Server, drbd, Cassandra, Kafka,
min.io, SAN, Active Directory, vmfs.

These are pivotal technologies that run our digital economy and they all use redo
logs — journals — as a source of truth. Different vendors. Different decades.
Different problem domains. No shared lineage to speak of. And they all converged
on the same primitive anyway.

| system | what it calls its journal |
|---|---|
| ext4 | the `jbd2` journal |
| NTFS | `$LogFile`, via the Log File Service |
| ZFS | the ZFS Intent Log |
| Postgres | the write-ahead log |
| Oracle DB | the redo log |
| SQL Server | the transaction log |
| Cassandra | the commitlog |
| Active Directory | ESE transaction logs |
| Kafka | the log *is* the product |

Now, when independent teams with nothing in common keep arriving at the same
answer, that answer is not an implementation detail. That's the thing itself —
and we file it in the operations manual anyway.

## Everything else is a copy in a costume

The importance of this only really shows when the chips are down and it's
recovery time. That's when the master journal really counts.

If you think about it, everything else around their journals of truth — query
abstractions, blocks, caches, indexes, performance optimisation tricks — they
*look* like the data. But in fact they're often just copies of what's written in
the journal.

<figure class="fig">
<svg viewBox="0 0 640 218" role="img" aria-label="Four derived structures — index, cache, block map, replica — sitting above a single journal bar, each connected to it by a dashed line.">
  <rect class="box" x="60" y="34" width="120" height="34" rx="5"/>
  <text class="t-mono mid" x="120" y="55">index</text>
  <rect class="box" x="200" y="34" width="120" height="34" rx="5"/>
  <text class="t-mono mid" x="260" y="55">cache</text>
  <rect class="box" x="340" y="34" width="120" height="34" rx="5"/>
  <text class="t-mono mid" x="400" y="55">block map</text>
  <rect class="box" x="480" y="34" width="120" height="34" rx="5"/>
  <text class="t-mono mid" x="540" y="55">replica</text>
  <path class="link" d="M120 68 V140 M260 68 V140 M400 68 V140 M540 68 V140"/>
  <text class="t-dim mid" x="320" y="106">derived — rebuildable</text>
  <rect class="box weights" x="60" y="140" width="540" height="42" rx="6"/>
  <text class="t-lbl mid" x="330" y="166">the journal — the only thing that is the state</text>
  <text class="t-dim" x="60" y="206">lose anything above the line and you rebuild it. lose the line and there is nothing to rebuild from.</text>
</svg>
<figcaption>Four things that look like your data, and one thing that is your
data. You only find out which is which on the worst day of the year.</figcaption>
</figure>

Here's the test, and it takes about ten seconds.

Delete an index — the database rebuilds it. Drop the buffer cache — it refills.
Corrupt a replica — it re-syncs. Every one of those is survivable, and nobody
even files an incident.

Now lose the journal.

There is no "somewhere else". That asymmetry is the whole ballgame, and it is
completely invisible on a healthy system... which is exactly why it gets designed
*around* rather than designed *for*.

### And the theory has been sitting there since 1978

And none of this is hidden. Leslie Lamport wrote it down decades ago; we filed it
under "consensus algorithms" and got on with our lives.

**State machine replication.** Take a deterministic state machine. Feed it an
ordered log of commands. Every replica that consumes the same log in the same
order arrives at the same state. That's it. That's the entire theory of
replicated systems.

Read it again and notice what it says: the *state* is not the thing you
replicate. The **log** is the thing you replicate, and the state is what falls
out the other end. Raft and Paxos are not "consensus algorithms" in any
meaningful sense — they are algorithms for **agreeing on the order of a log**,
and everybody calls them consensus because it sounds more impressive.

So when I say the journal is the state, I'm not being provocative. I'm reading
the field its own foundational result back to it.

And once you're looking for it you can't stop seeing it:

- **Git** is a journal. The commit graph is the truth; your working tree is a
  materialised view you can `rm -rf` without losing anything.
- **Event sourcing** is this idea, rediscovered by application developers who had
  never opened a database and were amazed by what they found.
- **Kafka** stopped pretending. It doesn't keep a journal behind a nicer
  abstraction — it sells you the journal and lets you build the abstraction. Which
  is why it ate the enterprise.
- **`fsck` versus journal replay.** One walks your entire disk hoping to infer
  what happened. The other reads the log and *knows*. Guess which one we all
  stopped using.

## So "data gravity" is the wrong phrase

Tech chat often talks of "data gravity" — for those new to it, the idea that
where you physically store your data in the world generates a gravity that pulls
other data, and the workloads that manipulate that data, to that single location.
CAP theorem demands it.

But I think that phrase is wrong.

Data is easily copied. And when that original data is copied, it loses its
consistency properties and hence has **no mass and no attraction**. Nobody in
their right mind builds a workload on a snapshot they can't trust to be current.

In practice this means data *at its origin* is what has the attractive effect,
rather than the data itself. And when you follow the rabbit to where that origin
physically is... you end up at the journal. Every single time.

Correct the concept then, and we have **journal gravity**.

Now this is not me renaming something for the fun of it. The new phrase makes a
prediction the old one can't. If you want to know where your workloads will
inevitably migrate, don't ask where the data is — data is everywhere, data is
cheap, data is on a laptop in a coffee shop. Ask **where the writes get ordered**.
That is the location with mass, and everything else in your architecture is
eventually going to fall into it whether you planned for that or not.

Data gravity tells you to look at your storage bill. Journal gravity tells you to
look at your write path. One of those is a description and the other is an
instruction.

## Which is why most "distributed systems" are nothing of the sort

Now to all those out there thinking they are building distributed systems when
they use stateless APIs, S3 buckets and Postgres instances.

Sorry fellas. Until you show me a technology you're using that utilises
**distributed journals with no single point of origin**, then it isn't a truly
distributed system.

What you've built is a distributed *compute* tier sitting in front of a
centralised point of truth. Which is a perfectly reasonable thing to build, and
most systems that need building are exactly that. It is just not the thing the
architecture diagram is claiming, and the gap between those two matters enormously
on the day your single origin has a bad afternoon.

And notice *why* stateless services scale so beautifully. It isn't clever
engineering. It's that they contain no state to distribute — every hard problem
was quietly pushed one layer down into the thing with the journal, and that layer
is where your system's real topology lives. Your API tier is a rounding error.
Your write path is your architecture.

<div class="key">
<h4>The question that exposes it</h4>
<p>Ask anyone showing you a distributed architecture one question: <b>where does a
write get ordered?</b></p>
<p>If there's an answer — a primary, a leader, a region, a partition owner —
that's your point of origin and your system is centralised with extra steps. If
they can't answer, they haven't thought about it, which is worse.</p>
</div>

## So how do we actually fix it?

The question then is this: how does one build useful apps on top of distributed
journals that are high performance and low latency?

Blockchain is out of the equation, given it's about the most inefficient, slow
and costly data store on the planet. No, that won't do at all.

But I want to be precise about *why* it won't do, because the reason is
instructive rather than dismissive. Blockchain is not the wrong primitive — it is
**exactly** the right primitive. It is a distributed journal with no single point
of origin, which is the thing I just said we need. It solves the ordering problem
properly and completely.

It just pays an absolutely obscene price for it. Global consensus on every
single write, burning energy as a substitute for trust, at a throughput that
would embarrass a fax machine. It is the correct answer to a question almost
nobody is asking, priced for a threat model almost nobody has.

Bittorrent's distributed hash tables got closer, but they were optimised for one
particular corner of the CAP triangle and they aren't general enough to run a
bank on.

So the state of play: we know exactly what we need, we have one implementation
that does it at a ludicrous price, and one that does it for a narrow case.

An opportunity for innovation then, it would seem.

---

## Postscript, three years on

I wrote that last line in 2023 and then went off and spent three years building
an inference engine, which is not where I expected any of this to matter.

It turned out to be load-bearing anyway. Three times.

**The expert cache.** A mixture-of-experts model too big for VRAM has to stream
its weights. My first design treated device memory and host memory as an
exclusive-or — an expert is in one place or the other — and that forced eviction
to copy bytes *back* across PCIe. 68 GB of traffic per configuration, every byte
of it identical to what a file on disk already held. Complete waste.

The fix wasn't a cleverer eviction policy. It was noticing that **the pack file on
disk is the journal** — the authoritative copy of every expert, always — and that
VRAM and RAM residency are merely records that a faster copy also exists.
Eviction stopped being a write and became a single field assignment. [The whole
story is here](/blog/waves-and-the-pcie-bottleneck).

**The KV cache.** The conversation substrate is backed by an append-only redo log
and there is deliberately no in-memory-only mode. Not for durability box-ticking
— because a three-tier cache whose cold tier is authoritative is *structurally
simpler* than one where it isn't. Every eviction path in the system collapses to
"forget it".

**The context itself.** And this is the one that surprised me. A system that has
ingested a 2.2-million-line codebase into unbounded context stores that
understanding as a **token sequence on disk**. Any instance of the engine, on any
hardware, can prefill that sequence and reconstruct a bit-identical KV cache,
fingerprint index and retrieval log.

The model's understanding of a codebase is a file you can put in version control
next to the code. Because the KV cache is a derived structure, and the token log
is the journal.

I didn't set out to apply any of this. It just kept being the answer.

**And the open question is still open.** My redo log has exactly one point of
origin, so it is not the distributed journal I was asking for above. Nobody has
solved that one yet.

But I'll tell you what I think now that I didn't think then. The reason isn't a
missing algorithm. A journal's entire value is **total ordering**, and total
ordering is precisely the one property you cannot have without agreeing on an
origin. That's not an engineering gap, it's the shape of the problem. Whoever
gets past it isn't solving storage — they're solving consensus, and they'd better
turn up with something better than proof-of-work.

Either way. Next time you see a journal, recognise and admire its silent
dominance.
