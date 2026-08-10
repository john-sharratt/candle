# The Expert Cache — VRAM ↔ RAM ↔ NVMe

> **Status — Design, not built.** Replaces the two-tier VRAM↔pinned-RAM expert
> cache (`candle-transformers/src/models/expert_lre/`) with a three-tier one
> whose cold tier is authoritative. The change is motivated by a specific defect:
> the current tiers are **mutually exclusive**, which forces eviction to be a
> device-to-host *copy* and makes VRAM residency a function of how much host RAM
> was allocated. §2 is that defect; §3 is the shape that removes it.
>
> Every number in this document is measured on the RTX 4090 Mobile 16 GB dev
> machine with Qwen3-30B-A3B unless it says otherwise, and §10 lists what is
> **not** measured and must be before building.

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
`total_experts − pinned_capacity`. Measured: 6144 − 4004 = **2140 slots**, a
floor nothing in the code states or enforces.

### 2.3 The eviction budget and the swap depth are the same slots

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

They trade one for one. The pool was sized `total − vram + vram/10`, so the "10%
headroom" is almost exactly the two layers of churn the pipeline needs — leaving
**nothing** for the boundary. A live rebuild confirmed it: the retraction asked
for 4436 regions and delivered `relocated=0 evicted=0`.

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
cold    repacked file, .substrate/experts.pack     authoritative, write-once, all 6144
warm    pinned host RAM                            static, immutable, stratified subset
hot     VRAM                                       dynamic cache, eviction = drop
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

### 6.1 Why a random subset is the right default

A uniform random subset of size *X*% yields **X% hit rate on accesses**,
whatever the popularity distribution: each access lands on some expert, and each
expert is resident with probability *X* independent of how hot it is.

The usual objection — "pick the hottest, not random" — does not apply, because
**VRAM already does the popularity filtering.** The hot working set lives in
VRAM, so the accesses that reach RAM are *misses*: the long tail. The tail is far
flatter than the head, and random sampling of a flat distribution is close to
optimal. The property that would normally make random naive is exactly what makes
it sound here — it sits behind a tier that has already skimmed the skew.

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
