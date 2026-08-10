# The Expert Cache — VRAM ↔ RAM ↔ NVMe

> **Status — Built.** Replaces the two-tier VRAM↔pinned-RAM expert cache
> (`candle-transformers/src/models/expert_lre/`) with a three-tier one whose cold
> tier is authoritative. The change is motivated by a specific defect: the old
> tiers were **mutually exclusive**, which forced eviction to be a device-to-host
> *copy* and made VRAM residency a function of how much host RAM was allocated.
> §2 is that defect; §3 is the shape that removes it; §12 is what the build
> settled that the design left open.
>
> Every number in this document is measured on the RTX 4090 Mobile 16 GB dev
> machine with Qwen3-30B-A3B unless it says otherwise. §10 listed what had to be
> measured before building; §12.1 records what the tree already answered.

---

## 1. Abstract

A MoE model's experts do not fit in VRAM. Qwen3-30B-A3B has 48 layers × 128
experts = **6144 experts** at 2,899,968 B repacked, or **16.6 GiB** — more than
the whole card. So the expert cache exists to keep a working subset resident and
stream the rest.

Today it does that with two tiers and one rule: an expert is in VRAM **or** in
pinned host RAM, never both. That rule is the source of every problem in §2, and
it is not load-bearing for anything: **experts are read-only weights.** Nothing
in the engine writes a resident expert after it is loaded — there is no dirty
bit, no writeback path, and no concept of a modified expert anywhere in the
cache.

Once a cold tier holds an authoritative copy of every expert, eviction stops
being a data movement and becomes a bookkeeping change. Everything else in this
document follows from that.

---

## 2. What is wrong today

```rust
pub(crate) enum ExpertLocation {
    Vram { slot_idx: usize },
    Pinned { slot_idx: usize },
}
```

Exclusive-or. Four consequences, in the order they bite.

### 2.1 Eviction copies data that already exists

An expert leaving VRAM has no home in RAM to fall back to — its pinned slot was
never allocated, because pinned was sized to hold only the experts *not* in VRAM.
So eviction must write the bytes back:

```
DMA evicts (D2H), one gate config: 23,415 × 2.9 MB = 68 GB of PCIe traffic
```

Per config, competing on the copy stream with an equal number of H2D loads. The
data being copied is bit-identical to what the file it was repacked from already
contains. **It is pure waste, and it exists only because of the exclusive-or.**

### 2.2 VRAM's floor is set by how much RAM was allocated

Because `pinned_occupied = total_experts − vram_slots` always, and the pool is a
single `cuMemAllocHost` that never grows, VRAM can never hold fewer experts than
`total_experts − pinned_capacity`. That is a hard floor on VRAM residency set by
a host allocation, and it runs the dependency backwards: how far the *device*
boundary may move is decided by how much *host* RAM was claimed at startup.

The floor moved twice while this defect stood, which is itself the argument.
Originally the pool was sized `total − vram + vram/10` from the boundary's
**opening** position — 4004 slots, floor 6144 − 4004 = **2140**, and nothing in
the code stated or enforced it, so a retraction could walk straight past it and
destroy experts (`docs/elastic_vram_partition.md` §13c). The elastic-partition
commit `d2cecd84` then inverted the derivation, sizing the pool from a *chosen*
floor instead — `min_vram_expert_slots = total/4` plus churn, so 4864 slots and
**14.1 GB** of pinned host memory on a 31.5 GB machine. That is the better of the
two and still the wrong shape: the machine spends host RAM in proportion to how
far the boundary *might* move, whether it moves or not.

### 2.3 The eviction budget and the swap depth are the same slots

At the boundary's opening position, under the original sizing:

```
pinned capacity           4004
pinned occupied at open   3767   (6144 − 2377)
free                       237
```

Those 237 free slots are simultaneously:

- the **churn depth** of the swap pipeline — a swap is evict-then-load, and
  `evict_for_prefetch_batch` is asked for as many experts as a layer is short
  (up to 128), with prefetch running a layer ahead
- the **entire budget** for the elastic VRAM boundary
  (`docs/elastic_vram_partition.md`) to retract

They trade one for one, and the "10% headroom" is almost exactly the two layers
of churn the pipeline needs — leaving **nothing** for the boundary. A live
rebuild confirmed it: the retraction asked for 4436 regions and delivered
`relocated=0 evicted=0`.

Sizing from the floor bought headroom rather than removing the conflict:
4864 − 3767 = 1097 free, less `CHURN_RESERVE_LAYERS × 128 = 256` reserved, leaves
~841 slots — about 2.4 GB of concession, paid for with 2.5 GB *more* pinned RAM
than the version that could concede nothing. **The two uses still share one pool,
and the pool is still the price of both.**

### 2.4 It produces states with no answer

```rust
let pinned_slot = match self.expert_locations[moe_idx][expert_idx] {
    ExpertLocation::Pinned { slot_idx } => slot_idx,
    ExpertLocation::Vram { .. } => {
        candle::bail!("load_from_pinned: L{moe_idx}E{expert_idx} is already in VRAM");
    }
};
```

"Where do I load this from?" has no answer when the expert is in VRAM, so it is
an error branch. A model that has to raise an error for a question that should
always be answerable is the wrong model.

---

## 3. The shape

```text
cold    <gguf dir>/<model>.<id>.experts.pack   authoritative, write-once, all 6144
warm    pinned host RAM                        static, immutable, stratified subset
hot     VRAM                                   dynamic cache, eviction = drop
```

Disk is not a fallback state an expert degrades into. It is where every expert
*is*, permanently, from the moment the pack file is written. RAM and VRAM
residency record that a faster copy *also* exists — and the three are not
alternatives. An expert can be in VRAM **and** RAM at once, and usually is:
promoting a warm expert copies its bytes to the device, it does not move them,
because the pinned tier is immutable (§6) and would not reuse the vacated slot
anyway.

So residency is two independent facts over an always-present base, not a choice
between three:

```rust
/// Where an expert's bytes are. Disk is implicit — always, by §4.
pub(crate) struct ExpertResidency {
    /// Slot in the VRAM zone, while a device copy exists.
    vram: Option<usize>,
    /// Slot in the pinned host pool. Written once at fill (§6), never again.
    ram: Option<usize>,
}
```

The load source is then a total function with no error case:

```rust
match (r.vram, r.ram) {
    (Some(_), _)    => already resident,
    (None, Some(s)) => H2D from pinned slot s,
    (None, None)    => read from the pack file,
}
```

and eviction is one field assignment, `r.vram = None`, which lands the expert on
whichever tier still holds it without consulting anything.

### 3.1 Why a product and not a sum

The obvious encoding is the sum type — `Vram { slot_idx, ram_backed:
Option<usize> } | Ram { slot_idx } | Disk` — which carries exactly the same four
states and exactly the same semantics. The product form is preferred for one
reason: `ram` is an **immutable per-expert fact** decided at startup, and the sum
form makes every VRAM transition rewrite it.

There are five sites that construct a VRAM location (initial load, promotion,
prefetch, forced load, `relocate_slot`). Under the sum form each must restate
`ram_backed` correctly — the compiler forces them to supply *a* value, but not
the *right* one, and `relocate_slot` moving a hot expert between VRAM slots is
precisely where a `ram_backed: None` would be written by hand and silently orphan
a live pinned slot for the process lifetime. Under the product form those sites
assign `r.vram` and the `ram` field is not in the expression, so it cannot be
dropped. The invalid state is unrepresentable rather than merely avoided.

The cost is losing match exhaustiveness as a change detector, which is worth
little here: tiers are not being added, and the fallback chain `vram → ram →
disk` is naturally an option chain, not a match.

---

## 4. The invariant

> **The cold tier holds a valid copy of every expert, always.**

Everything else is a consequence:

- **Eviction from VRAM is a drop.** `r.vram = None`. No copy, no stream, no
  ordering, no destination slot to find, no failure mode. §2.1 deletes.
- **An evicted expert falls to RAM, not to disk, when RAM still holds it.** This
  is not a second mechanism — it is what the drop already means. The pinned copy
  was never surrendered on promotion and the immutable warm tier never reclaimed
  its slot, so clearing `vram` re-exposes a copy that has been there the whole
  time. The next miss on a warm expert is an H2D from pinned memory whether or
  not it has been through VRAM before, which is the point: **residency in the
  warm tier is a property of the expert, not a history of its promotions.**
  Without this the tier *drains*: every warm expert that is ever promoted would
  leave the warm set permanently, so the hit rate decays toward zero over the
  process lifetime and the warm tier is worth least exactly when the daemon has
  been up longest. Since promotion is driven by demand, the experts lost first
  would be the most useful ones.
- **The warm tier has no eviction at all.** §6 makes it immutable, so it needs no
  eviction policy *and* no eviction path.
- **VRAM has no floor.** Any expert can leave at any time, so the elastic
  boundary is bounded only by `MIN_ELASTIC_RESERVE` and by where throughput stops
  paying. §2.2 deletes.
- **There is no shared budget.** Nothing competes for free slots because nothing
  needs a free slot to evict into. §2.3 deletes.
- **Every load has a source.** §2.4 deletes.

The invariant is cheap to hold because experts are read-only (§1). It would be
expensive and delicate if they were not.

---

## 5. The cold tier — a repacked pack file

### 5.1 Repacked, not the original GGUF

The pack file stores experts in the **repacked K/128 layout the kernels consume**,
one contiguous blob per expert, not the original GGUF tensors. Three reasons:

1. **The repack is hot-path poison.** Startup repacks 6144 experts in ~42 s —
   about 7 ms each. A forward issues on the order of 1,150 expert loads
   (~24 misses × 48 layers). Repacking on load would cost seconds per forward.
2. **The repacked form is already one contiguous blob per expert** (gate + up +
   down concatenated). That makes a load an offset and a copy. The GGUF layout is
   per-*tensor* with experts stacked inside, so loading one expert from the
   original is a strided gather plus a dequantise.
3. **It decouples the hot path from the checkpoint format.** Reading the original
   would make GGUF packing decisions into cache performance regressions.

The record layout **is** the VRAM slot layout — same three projections at the
same aligned offsets — so a load is one read and three copies with nothing
rearranged in between, and the same code path serves a warm promotion and a cold
miss. Records are padded to a 4 KiB sector because the reads bypass the page
cache (§12.1), which also makes the warm pool's slots the right size to be read
into directly.

### 5.2 Where it lives — and why not `.substrate/`

**Not under `.substrate/`.** The pack is a pure function of the *checkpoint*; the
substrate directory holds conversation state, which is a different thing with a
different lifetime. Putting it there gets two things wrong at once:

- **`--wipe-substrate` would delete it.** That flag is used on nearly every
  iteration run, and it would silently cost 42 s of repack plus a 16.6 GiB write
  on every restart — for a file whose contents the wipe has no opinion about.
- **One copy per workspace.** Two workspaces on the same model would each carry
  their own 16.6 GiB, for byte-identical content.

The pack belongs **beside the checkpoint it is derived from** — `<gguf
dir>/<model-hash>.experts.pack` — where it is shared by every workspace using
that model, survives substrate wipes, and is deleted by the same act that deletes
the model. The hub cache is user-writable in both the HF-download and
`--model-dir` cases. The exposure is that a hub-cache cleaning tool may remove
it, which costs a repack and nothing else.

**When there is no persistent location, it is a temp file unlinked at exit.**
That is the default: an embedder, an example, or a test must never have a
16.6 GiB file appear beside its model without asking.

### 5.3 How the choice reaches the cache

The store is a **builder parameter on the conversation engine**, defaulting to
`None`:

```rust
impl ModelBuilder {
    /// Directory for the persistent repacked expert pack. `None` (default) uses
    /// a temp file unlinked at exit, paying the repack on every start.
    pub fn expert_pack_dir(mut self, dir: impl Into<PathBuf>) -> Self { .. }
}
```

It reaches `ExpertCache::new` down the path the model loader already takes:

```text
ModelBuilder::expert_pack_dir            builder.rs
  → ModelBuilder::load_model             takes &self already — no signature change
    → ModelWeights::from_gguf_by_path    quantized_qwen3_moe.rs
      → ExpertCache::new                 the only consumer
```

Three notes on that path:

1. **`load_model` needs no new argument.** It takes `&self`, so it reads the
   field directly, the same way it reads `max_seq_len`.
2. **The `_with_int8` pair collapses.** `from_gguf_by_path` and
   `from_gguf_by_path_with_int8` already exist as a pair for one load-time knob;
   a second knob would make a third variant. They become a single entry point
   taking a `GgufLoadOptions { int8mode, expert_pack }`, which is also where the
   next load-time decision goes instead of a fourth function.
3. **It is MoE-only.** Only the `Qwen3Moe` arm of `load_model` has an expert
   cache; the `Qwen3`, `Qwen2` and `Llama` arms ignore the field entirely. The
   parameter is on the builder rather than the arch because it configures the
   engine, but it is inert for three of the four arches.

`zend` passes the GGUF's parent directory, so its packs persist by default; every
other caller gets a temp file unless it opts in.

### 5.4 It makes restart cheap

The pack is a pure function of the checkpoint. Stamped with the model's hash and
the repack format version, **a restart can skip the 42-second repack entirely**
and map straight to serving. That is a material quality-of-life win on a daemon
that gets restarted while iterating, and it is free — the file has to exist
anyway. It is also the entire reason §5.2 fights for a location that survives a
substrate wipe: a pack that is deleted every iteration delivers none of this.

### 5.5 What it costs

16.6 GiB on disk, and a first boot that pays the repack *and* the write before
serving. Both are one-time; neither is free. In the temp-file case they are paid
on *every* boot, which is the price of not writing to someone's model directory
uninvited.

---

### 5.6 The pack is checked against the formula that wrote it

Every other field of the pack's identity describes the **input**: which
checkpoint, which numeric mode, which record geometry. None describes the
**function**. Change how the repack lays bytes out at unchanged sizes, offsets
and dtypes — a different permutation, a moved rounding step, a fixed bug — and
every one of those checks still passes. The stale pack is reused and the model
serves subtly wrong weights for its whole expert set, silently, across restarts
and substrate wipes, until someone notices the outputs drifted.

A version constant does not close this, because it depends on whoever changed
the formula remembering to bump it — and the failure of that memory is exactly
the case worth defending against. **So the check runs the formula instead.**

At startup, a fixed table of `(source dtype → target dtype)` pairs is swept: every
quantisation an expert weight can arrive as, repacked every way the engine can
repack it — straight to the gemx K/128 layout, and to each of the KO twins the
two int8 modes select. Each pair gets a deterministic reference matrix; the
outputs are hashed together into `repack_fp`, which the header carries and every
open compares.

Three properties make it worth its ~36 small repacks:

- **The inventory is part of the hash.** Adding a dtype, removing one, or
  changing which twin a mode selects moves the fingerprint even when every byte
  of every repack is unchanged. A pair the repack *refuses* hashes as a refusal,
  so gaining support for a type invalidates old packs too.
- **It covers types this model does not contain.** The sweep is a property of the
  build, not the run. A binary that repacks Q5_K differently is stale whether or
  not today's checkpoint has a Q5_K tensor in it — which matters because the next
  checkpoint might.
- **The reference data cannot go NaN.** Every byte of the pattern is ≤ 60, so no
  `f16` scale read out of it can have an all-ones exponent. A
  dequantise-requantise path would otherwise hash NaN payload bits, which are not
  guaranteed stable, and the fingerprint would drift on its own.

### 5.7 Records carry a checksum, checked in bulk

Each record's `fletcher32` goes in a trailer after the records — a trailer
because the writer streams 16.6 GiB and never seeks, so the only place a value
known *after* writing a record can go is the end.

It is verified on the **bulk** path: the startup fill, where thousands of records
land at once, the cores are idle waiting on the drive, and the work parallelises.
It is **not** verified on the per-miss cold read, and that is measured rather
than assumed. A `fletcher32` over 2.9 MB costs about as much as the read it
follows, on the pipeline thread, in front of a forward that is waiting for it:
with it there the gate lost **more than half its throughput** — 723 → 299 t/s on
the narrowest config — to insure ~850 records per config.

What the trailer defends is the *medium* — bit rot, a bad sector, a truncating
filesystem — on a file that lives beside the checkpoint for as long as the
checkpoint does. It is not defending against a half-written pack (the writer
publishes by rename, so an incomplete one is never visible) or against two
writers colliding (§12.7). The residual exposure is one unverified expert of
6,144 in a layer that sums eight of them, against a checkpoint that carries no
per-tensor checksum of its own.

## 6. The warm tier — static, stratified, immutable

The warm tier is filled **once**, at startup, and never changes. No eviction
policy, no ranking, no background filler, no demand-versus-speculative priority,
no thrash. It is a partial mirror of the pack file, pinned for DMA.

Immutability is what licenses §4's fall-back-to-RAM. Because no other expert can
ever claim a pinned slot, a promoted expert's slot is not worth reclaiming and
holding it costs nothing that anything else wants — so `ram` is written at fill
and never cleared. A warm tier with an eviction policy could not make that
promise, and the residency struct would need the two fields kept in step against
a policy that moves underneath it.

### 6.1 Why a random subset is the right default — over the right pool

A uniform random subset of size *X*% yields **X% hit rate on accesses**,
whatever the popularity distribution: each access lands on some expert, and each
expert is resident with probability *X* independent of how hot it is.

The usual objection — "pick the hottest, not random" — does not apply, because
**VRAM already does the popularity filtering.** The hot working set lives in
VRAM, so the accesses that reach RAM are *misses*: the long tail. The tail is far
flatter than the head, and random sampling of a flat distribution is close to
optimal. The property that would normally make random naive is exactly what makes
it sound here — it sits behind a tier that has already skimmed the skew.

**But that same sentence names the pool the draw must run over, and the first
build got it wrong.** If VRAM has already skimmed the skew, then the experts VRAM
*holds* are not part of the miss stream at all, and a warm slot spent on one buys
nothing until that expert is evicted. Drawing uniformly over all 6,144 spent 36 %
of the tier on guaranteed hits.

The startup fill takes the first `vram_slots` experts in flat `(layer, expert)`
order, so the pool the draw should run over is everything past that prefix — and
the tier only has to be the size of VRAM's *complement*, not of the model, to
cover every miss. On Qwen3-30B-A3B that is 3,767 experts rather than 6,144: it
turns "hold the whole model in RAM" from a requirement into a nicety. The
two-tier cache had this property for free, by construction, because pinned *was*
the complement; this is the one thing the exclusive-or was buying, and it has to
be re-bought deliberately.

Slots past the complement are not wasted — they insure VRAM-resident experts
against eviction — but they are the second call on the tier, not the first.

### 6.2 Why the sample is stratified per layer

Take *X*% **of each layer**, not *X*% of the whole set.

A global draw gives per-layer residency `~ Binomial(128, X)`. At *X* = 40%: mean
51.2, sd 5.54, so across 48 layers the unluckiest lands near 37 — **29%
residency against a 40% average.**

That matters more than it would for a normal cache, for two reasons:

- **The fill is immutable.** A globally-random draw does not make one layer
  unlucky for one forward; it makes it slow on *every* forward for the process
  lifetime. A permanent bottleneck decided by a coin flip at startup.
- **The wave is sequential.** Layer *N+1* cannot begin until layer *N*'s experts
  land, so lucky layers cannot compensate for unlucky ones. The sweep runs at its
  worst stage, not its mean.

Stratifying drives that variance to zero at no cost: same mean, same fill, same
immutability, one different line in the sampler.

### 6.3 Sizing

The warm tier is a **performance** choice, not a correctness one — the invariant
(§4) holds at any size, including zero. So it is sized by what the machine can
spare, and `cuMemAllocHost` refusing is the answer to "was that too much".

This is the change §2.3 makes possible: pinned capacity stops being
`total_experts − vram_slots` and becomes free.

**Free is not the same as arbitrary, and "what the machine can spare" has a
precise meaning that took two wrong answers to find.** It is not a *share* of
spare RAM — the first build took half, left 2,241 of 6,144 experts warm, and sent
64 % of every miss to disk. And it is not measured against *total* RAM either: a
dev box with an editor and a browser open is 12 GB down before the process
starts, so a tier sized against the total succeeds by consuming every free page
and leaves the next pinned allocation in the same process to fail with
`CUDA_ERROR_OUT_OF_MEMORY` — which is what a 46 MB staging ring did.

The rule that survives contact:

```text
ask for   every expert
bounded by min(what the machine is big enough for, what it has free now − headroom)
backed by cuMemAllocHost halving on refusal
```

with the headroom (3 GiB) reserved for everything the process allocates *after*
the warm tier — the staging ring, the routing buffer, the substrate's pinned
cold-load scratch, the `PinnedStager` arenas, and the warm KV tier. Pinned pages
cannot be reclaimed under pressure, so a tier that fits by exactly nothing does
not degrade, it fails.

### 6.3.1 The tier does not have to hold the model, and past a point should not

On the dev box the tier lands at **4,979 of 6,144 experts (81 %)** — 14.4 GB of
the 17.8 GB the whole set needs — because ~17.7 GB is free at load time with an
editor and a browser running, not because 31.5 GiB is too small. It would hold
everything on an idle machine, and does trivially on the production box.

**That shortfall costs almost nothing, and closing it further costs more than it
saves.** Two mechanisms make the difference: the draw covers VRAM's complement
first (§6.1), so every miss is covered at startup; and eviction weighs what a
reload costs (§12.6), so VRAM keeps the experts the tier could not take. Together
they hold cold reads to 4 % of loads at 81 % coverage.

Lowering the headroom to 1 GiB was measured. The tier grew to 5,090 slots and
cold loads halved (986 → 435) — and **throughput did not improve**: flat to 1–2 %
down, single-stream falling further. Past the knee, pinned pages come out of the
page cache and the warm KV tier, and the reads they save are no longer on the
critical path. The tier's job is to cover the miss stream, not to mirror the
model.

### 6.4 Refinements deliberately not taken yet

**Fill by observed popularity rather than randomly.** Run the first *N* forwards
cold, count accesses (the counters exist), then do the single fill from the
observed top-*X*%. Still one fill, still immutable. Worth it only if the tail
turns out to be skewed enough to beat *X*% — which §10 should measure before it
is built.

**Grade the strata by layer depth.** `cache.rs` already records the reason this
might pay, for the VRAM tier: the first layers "run first every pass and have
**zero compute to overlap with DMA**". A RAM slot given to layer 0 removes more
stall than one given to layer 40, which has the whole preceding sweep to prefetch
behind. So the strata need not be equal. Flat first: it removes the defect that
matters (§6.2) and costs one line, where a taper needs a curve and a curve needs
evidence. The extreme case is already handled — layers 0–2 are VRAM-pinned
regardless.

---

## 7. The hot tier — VRAM

Unchanged in policy: the existing frequency-plus-recency scoring, layer-aware
forced eviction, early-layer pinning and Markov prefetch all stand. The measured
prediction precision is **66–85% (74% typical)**, and that machinery is the
reason §6.1's argument works.

One change: **eviction is a drop.** `evict_to_pinned` and every path that
allocates a destination slot for it are deleted.

The scoring is unaffected but its *stakes* change. Today an eviction costs a
2.9 MB D2H copy and can fail outright, so a mispredicted eviction is expensive
and the policy is tuned partly around that. Under §4 a mispredicted eviction of a
warm-backed expert costs one H2D to undo, and of a cold expert one pack read — so
the policy can afford to be more aggressive at the boundary than it currently is.
Whether it *should* be is a measurement, not something to assume here.

---

## 8. What this deletes

From the expert cache:

- `evict_to_pinned` and the D2H eviction path entirely
- the free-slot accounting on the pinned pool, and its free list — an immutable
  fill has no free list, because nothing is ever returned to it
- the `Vram { .. } => bail!` branch in `load_from_pinned` (§2.4), which does not
  become unreachable but **unrepresentable**: the state it guarded, "in VRAM and
  therefore nowhere loadable from", is not expressible once `vram` and `ram` are
  independent

From the elastic VRAM partition (`docs/elastic_vram_partition.md` §13c), all of
which exist solely to work around §2.2 and §2.3:

- the retraction cap against free pinned slots
- `CHURN_RESERVE_LAYERS` and the churn reserve
- `min_vram_expert_slots` and the derived VRAM floor
- the "weight side cannot concede" refusal path

`relocate_slot` — the VRAM→VRAM move that keeps a hot expert alive when the
boundary retracts over it — becomes **optional** rather than necessary. Dropping
and reloading from RAM is now legal; whether the device-to-device copy still pays
is a measurement, not an argument.

---

## 9. Consequences for the elastic partition

The two designs meet at exactly one place: how far the weight side may retract.
Today that is capped by free pinned slots and measured at **0–109 usable slots**,
which is why the boundary could not move on a live rebuild. Under §4 the cap
disappears and the range becomes the full `[MIN_ELASTIC_RESERVE, max_slots]`
span — the elasticity that document was written to deliver.

**This is a prerequisite, not an optimisation.** The elastic partition cannot do
its job while eviction requires a destination slot.

---

## 10. What must be measured before building

Ordered by how much they could change the design.

**1. Is disk→VRAM actually direct on this platform?** There is **no cuFile
binding in the tree**, and GPUDirect Storage is Linux-first with thin-to-absent
Windows support. If it is unavailable, `Disk → VRAM` is really
`disk → OS page cache → host buffer → VRAM`, which changes the cost model and
means the page cache is already an uncontrolled warm tier underneath us.

**2. The NVMe : PCIe bandwidth ratio, which decides whether the warm tier is
worth building at all.**

| machine | NVMe | PCIe to GPU | warm tier value |
|---|---|---|---|
| dev (this box) | single, ~3–7 GB/s | ~25 GB/s | 4–8× — earns its keep |
| production (`CLAUDE.md`) | 16 TB PCIe 5.0 RAID 0 **@ 45 GB/s** | ~25 GB/s | disk is *faster than the bus* — buys ≈ nothing |

The warm tier may be a dev-box concession rather than a permanent architecture.
That is an argument for keeping it as simple as §6 describes and for making it
sizable to zero.

**3. Pinned RAM versus the page cache.** The pack file is 16.6 GiB and host RAM
is 31.5 GB, so the OS will cache a large fraction of it for free. An explicit
pinned tier buys **guaranteed residency** and **no bounce copy on DMA**, and
costs physical memory the page cache would otherwise use. The honest baseline for
§6 is therefore *no explicit warm tier at all*.

**4. Cold-read latency against the prefetch horizon.** 2.9 MB per expert. If the
existing Markov prefetch (74%) covers it a layer ahead, cold misses are nearly
free; if not, the warm tier has to be large enough to make them rare.

**5. Whether expert popularity is skewed enough to justify §6.4's warmup fill.**

---

## 11. Risks

**The pack file is a new artefact with a lifecycle.** 16.6 GiB in `.substrate`,
needing invalidation on checkpoint change, and a story for two daemons on one
workspace. Not hard; the kind of thing that bites later if it is not decided now.

**First boot is slower**, paying the repack and a 16.6 GiB write before serving.

**Pinned memory is non-pageable.** Whatever §6.3 settles on is memory the OS can
never reclaim, and it competes with the page cache holding the same file (§10.3).

**The warm tier could be redundant.** If §10.1 and §10.3 both come back
unfavourably, the right answer may be no warm tier: read the pack file and let
the OS cache it. That would be a *simpler* system than this document describes,
and it should be allowed to win.

---

## 12. What the build settled

### 12.1 §10's first three questions were already answered in the tree

The measurements §10 asks for were framed as unknowns. Two of them are not: the
substrate's cold-load path had already met them and written the answers down.

**§10.1 — disk→VRAM is not direct on this platform, and the tree says so.**
`candle-conversation/src/persistence/cold_load.rs` states it outright: GPUDirect
Storage depends on the `nvidia-fs` kernel module NVIDIA does not ship for
Windows, and `libcufile` is not in the Windows CUDA Toolkit. DirectStorage is a
D3D12 API and not reachable from CUDA. So `Disk → VRAM` is
`disk → host buffer → VRAM`, exactly as suspected.

**§10.3 — the page cache is not a free warm tier here; it is the thing that had
to be bypassed.** The honest baseline §10.3 proposes — no explicit warm tier,
let the OS cache the pack — is the configuration the substrate already tried and
abandoned. Through the ordinary buffered path a read into `cuMemAllocHost`'d
memory caps at **~7–10 MB/s** on Windows. `direct_io.rs` exists because of that:
`FILE_FLAG_NO_BUFFERING` / `O_DIRECT` positioned reads into a sector-aligned
pinned buffer, at the device's real sequential bandwidth.

That module is now `candle-core/src/direct_io.rs` rather than a private module of
the persistence layer. The expert cache sits *below* `candle-conversation` in the
crate graph and could not reach it where it was written, and a second
implementation of the Windows `ReadFile`+`OVERLAPPED` path was the alternative.
It moved unchanged, with its tests.

**§10.2, §10.4 and §10.5 remain open**, and none of them gates the build: they
size the warm tier, and §6.3 makes that a number the machine chooses rather than
a constant to get right. On the production box — NVMe RAID faster than the bus —
the answer may well be to size it to zero, which §11 already allowed.

### 12.2 The pack replaces the repack on every boot but the first

A consequence §5.4 predicted and the build confirms in shape: with a persistent
pack, startup reads only the experts it is going to place — the VRAM fill and the
warm membership — and touches the GGUF's expert regions not at all. The ~42 s
repack becomes a read. `zend` passes the GGUF's own directory, so its packs
persist; the batched-forwarding gate does the same, because it reloads the model
once per invocation while iterating.

### 12.3 The warm tier is sized from the machine, and that is where the RAM went

§6.3 says the warm tier is sized by what can be spared. What that turns out to
mean on the dev box is the point of the whole change:

| | before (`d2cecd84`) | after |
|---|---|---|
| pinned expert pool | 14.1 GB, fixed by `total − total/4 + churn` | half of what is spare after weights and the OS |
| what sets it | how far the boundary might retract | how much RAM the machine has |
| KV warm budget (`host_ram_budget`) | negative — pinned + mmap exceeded RAM | the other half |

`candle::vram::host_ram_budget` computes the warm-KV budget as
`total − pinned − weights_mmap − os_keep`. With a 14.1 GB pinned pool and a
17.3 GB mapped checkpoint on a 31.5 GB machine, that was negative: the expert
cache was consuming the KV warm tier's entire budget as a side effect of a
residency floor nobody had asked for. Sizing the warm tier from spare RAM instead
hands those bytes back, and `cuMemAllocHost` halving on refusal means the number
does not have to be right, only close.

### 12.4 Removing the cap exposed a missing quiesce at the boundary

The gate found it, once in four runs: `CUDA_ERROR_ILLEGAL_ADDRESS` on the widest
config (`Q4_0`, 20 contexts) — the only one that puts enough KV pressure to make
the boundary move a long way. The other fifteen passed, and so did three
subsequent runs of the same binary, which is the signature of an ordering hazard
rather than a wrong address.

**The bug is not in this document's design; it is one the design's success
reaches.** The KV side quiesces before re-tenanting a *recycled* region
(`region_pool::claim_region`), because a region on its free list may still be
under read by a kernel launched earlier. Ground arriving from the weight side is
not recycled — it is **fresh**, claimed by advancing `pool.next` past a ceiling
that has just moved — and that path has no wait at all. That was correct for as
long as it stood: while the concession was capped at free pinned slots, the
ceiling never moved over ground an expert had been sitting on, so a fresh region
had genuinely never been anyone's.

With the cap gone it moves over gigabytes of it. `renegotiate_boundary` runs at
end of pass, and "the pass ended" means its expert GEMMs were *issued*, not that
they retired — so publishing a lower floor lets the KV side memset bytes those
GEMMs are still reading. The fault then surfaces in whatever unrelated kernel is
running when it lands, which is why it looks random.

The fix is a device-wide synchronize on **both** directions of the handover,
immediately before the new floor is published and before newly-taken regions can
be filled. A GPU-side `cudaStreamWaitEvent` cannot do this job: it orders two
streams we know about, while the readers include the persistence thread's copy
stream, and the *host* is what has to know the ground is quiet before it tells
the other side it may have it. It is the same quiesce `claim_region` performs,
for the same reason, at the one place where memory changes side.

It costs a full drain, paid only when the boundary actually moves.

**The general form is worth stating**: a cap that keeps a mechanism from
operating also keeps its hazards untested. The three guards
`docs/elastic_vram_partition.md` §13c added were correct, and they were also the
reason that document could report the boundary as working while it had never
moved a byte.

### 12.5 The regression, and the three things that fixed it

The first build of this design was **30 % slower than the two-tier cache it
replaced**, on every config. It is now faster than that cache on every config,
and the gap between those two statements is the most useful thing in this
document.

| config | two-tier (before) | three-tier, first build | three-tier, now |
|---|---|---|---|
| F16 × 1 | 377.5 | 389.1 | **708.1** (+87 %) |
| BF16 × 1 | 509.4 | 455.1 | **780.6** (+53 %) |
| BF16 × 10 | 1931.4 | 1626.1 | **2726.1** (+41 %) |
| Q8_0 × 20 | 2448.1 | 1596.0 | **2732.5** (+12 %) |
| Q4_0 × 4 | 1800.5 | 1368.9 | **2329.9** (+29 %) |
| C0 × 2 | 961.0 | 818.4 | **1424.0** (+48 %) |
| C5 × 2 | 940.1 | 830.7 | **1442.3** (+53 %) |
| C9 × 2 | 951.4 | 833.7 | **1463.1** (+54 %) |
| Q4_0 × 20 | 2423.4 | 1655.0 | **2735.1** (+13 %) |

t/s aggregate, `test_parallel_batched_forwarding`, RTX 4090 Mobile,
Qwen3-30B-A3B. Single-stream t/s rose 20–35 % alongside.

**What was wrong was never the architecture — it was that the tier which makes
the architecture affordable had been sized by three separate guesses.** The
diagnostic that found it was a single counter pair, `warm_loads` / `cold_loads`,
added to the stats table: 15,993 cold against 9,089 warm on Q8_0 × 20 says
immediately that two thirds of every miss is a synchronous NVMe read, and no
amount of profiling around the edges says it as fast. That is why the warm
tier's *size* is now reported in the same table as its hit rate — a cold-load
count is a verdict on that number, and reading the two in different places is
exactly how this went unnoticed.

Three fixes, in order of what they bought:

1. **Size against available RAM, ask for every expert** (§6.3). 2,241 → 4,979
   slots. Cold loads on Q8_0 × 20: 15,993 → 4,038.
2. **Draw the membership from VRAM's complement** (§6.1). The tier now covers
   the entire miss stream at startup instead of spending a third of itself on
   experts VRAM already holds.
3. **Put the reload cost in the eviction score** (§12.6). Cold loads 4,038 →
   986, and +11 % on the widest config.

And one thing that was simply in the way: the loader was still calling
`register_mmap_cuda`, pinning all 18.6 GB of the GGUF so H2D copies out of it
would not bounce. For a dense model that is right — the mapping *is* the weight
source. Here the experts move to the pack at startup and what stays live is the
dense tensors and the embedding table, so it was locking 18.6 GB, non-pageable,
for the process lifetime, in direct competition with the tier that keeps expert
loads off the disk. The MoE loader no longer registers it, and separately no
longer declares the GGUF's expert regions to the host-RAM budget as resident
weight bytes — they are read once, to build the pack, and never again.

### 12.6 Eviction learned what a reload costs

A mechanism the design did not anticipate, and which only exists because the
tiers stopped being exclusive.

The two-tier cache had one reload cost: every expert not in VRAM was in pinned
RAM *by construction*, so every eviction was worth the same and the score could
be pure temperature. Under three tiers an expert either has a warm copy — ~116 µs
H2D at PCIe bandwidth — or it does not, and comes back as a 2.9 MB
page-cache-bypassing NVMe read near a millisecond. An eviction policy blind to
that difference is choosing at random between outcomes an order of magnitude
apart.

So `slot_eviction_score` is now `frequency × position × reload_cost`, with
`reload_cost` ∈ {1, 4} by whether the warm tier holds the expert. The cache then
converges on the right shape without anyone specifying it: **VRAM drifts toward
holding what is expensive to re-acquire, the warm tier covers what is cheap, and
the experts that churn are the ones whose churn costs least.**

The 4 is measured, not derived. The cost ratio is nearer 8, and at 8 the term
stops tilting the ordering and starts replacing it — cold-only experts are held
past the point their temperature justifies, hit rate falls (44.8 % → 44.3 % on
Q8_0 × 20), and every config was slower than at 4.

### 12.7 What the code review found

Six defects, none of them in the shape of the design and all of them in what the
build had to decide for itself.

**Mandatory-and-small must allocate before elastic-and-large.** The cold-tier
staging ring (46 MB, required) was allocated *after* the warm tier (~14 GB,
elastic, halves itself when refused). On a machine the warm tier had just filled
— the reuse path immediately after a run that wrote 16.6 GiB, so the page cache
was full and `avail_phys` read high — the 46 MB failed and the model load died
with `CUDA_ERROR_OUT_OF_MEMORY`. It should have been a slightly smaller warm
tier. Reversing the order is the fix and it made the tier *bigger*, not smaller
(5,100 slots against 4,979), because the warm sizing now runs against a machine
whose mandatory allocations are already accounted. A sector-aligned host
fallback inside the ring is the belt: it must never be a plain `Vec`, whose
alignment `FILE_FLAG_NO_BUFFERING` rejects.

**Geometry validates where the bytes go, not what they are.** §5's identity —
GGUF length, a 4 MiB checksum, and the record geometry — misses a repack that
emits *different bytes* at identical sizes, offsets and dtypes: a changed
permutation, moved rounding in the quantizer. That pack validates and serves
subtly wrong weights for the model's entire expert set, silently, and survives a
substrate wipe. §5.6 is the answer, and it is stronger than the one this review
first got: a reference sweep over every quantisation rather than a probe of the
one expert this model happens to start with.

**Nothing checked the record bytes against the medium.** §5.7 adds a per-record
checksum in a trailer, verified on the bulk fill. Deliberately *not* on the
per-miss cold read: with it there the gate lost more than half its throughput,
which is the wrong price for insuring a derived cache against a failure the
checkpoint it derives from is not insured against.

**A shared pack needs a private temp file.** §5.2 puts the pack beside the
checkpoint precisely so several workspaces share one, and the pack's name is a
pure function of that checkpoint — so the fixed `.partial` sibling was the same
path for every process that decided to build it. Two daemons starting together
interleave their writes into one file and both rename it into place; the header
is written first and is identical, so the result validates. The temp name now
carries the pid and a nanosecond stamp.

**An all-resident cache wants no warm tier.** When VRAM holds every expert
nothing is ever evicted — `post_compute` returns before the eviction and boundary
passes — so a warm slot could only be read by a load that misses, and none does.
The old two-tier code had this guard (`if all_resident { 0 }`) and the rewrite
dropped it, pinning the model's size in host RAM to serve nothing and paying a
full-pack read at startup for it.

**`O_DIRECT` does not exist on Apple targets.** `libc` does not define it, so
naming it under `cfg(unix)` is a build failure rather than a slow path — and
`direct_io` is now in `candle-core`, which is every crate's dependency. Linux
keeps `O_DIRECT`; other unixes open buffered, which is correct but cached.

**A cuda build has no CPU expert path, and should say so at load.** The rewrite
made this an error where it had been a silent success followed by a panic on the
first MoE layer — but the doc still advertised a non-CUDA path and the caller
still built a plausible-looking host-side `WeightZone` for it. Both are gone, and
the message now names the reason and the two ways out.

### 12.8 Ephemeral packs are unlinked while open, not deleted on drop

§5.2 asks for "a temp file unlinked at exit". The build takes the older and more
robust reading: the pack is unlinked **immediately after it is opened for
reading**, and stays fully readable through the open handles until the process
ends. Unix has always allowed this; Windows does too, because `std::fs` opens
with `FILE_SHARE_DELETE`. Nothing is left behind however the process ends —
including a kill, which no drop handler survives.
