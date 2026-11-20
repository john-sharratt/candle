# The elastic warm tier — pinned expert RAM that follows the machine

**Status:** design (2026-11-21). Builds on the warm tier and the pad of
`docs/expert_cache_design.md` §6.3 and §6.5, which it changes from a size
chosen once at load into one the process keeps adjusting.

## 1. Why

The expert warm tier is the host copy of the experts VRAM cannot hold. Every
expert it misses is a read from the pack on disk, on the critical path of the
layer that routed it. It is sized **once**, mid-load, from the RAM free at that
instant less a headroom (`handle::warm_sizing_from`), and never revisited.

Measured on WhiteWorld (RTX 3090, 68.6 GB host, Flash-Next, 2026-11-20):

| | value |
|---|---|
| RAM free just before a restart | 26 GB |
| RAM free when the tier was sized, 3 min into the load | 11.5 GiB |
| tier taken | 8.3 GiB — 4,154 of 24,064 slots |
| RAM free an hour after the load | ~11–26 GB, unused |
| expert hit rate, VRAM | 3–54 % |
| one expert's pack read | 120–1,000 ms (disk shared with a 19 GB-busy pagefile) |
| prefill | ~1 token per sequence per forward, a 3,000-token backlog |
| decode | 300–1,500 ms a forward |

The load's own transients — staging buffers, the substrate opening, the VRAM
fill — take the RAM the tier is sized against and give it back minutes later.
A tier that cannot grow afterwards keeps the trough's size for the process's
life. One that cannot shrink has the opposite failure: a build started on the
machine pushes the process's own pageable memory to the pagefile, and every
fault on it stalls the wave.

## 2. What the tier must do

1. **Use the machine's RAM**, up to the point where the process itself starts
   paging, and no further.
2. **Count only its own paging.** Other processes page all the time — a
   minimized app is trimmed to the pagefile and that is the OS doing its job.
   Their paging says nothing about whether this process has too much pinned.
3. **React fast to a build**: when another process starts using RAM, give it
   back within seconds, so the machine recovers before the build thrashes.
4. **Not churn.** A grow that will be undone a minute later costs a
   `cuMemAllocHost` and a fill; a shrink costs the experts it held. The tier
   settles near its steady size and probes past it rarely.

## 3. The signals

All per tick (`TICK` = 1 s), read off the critical path on the controller's own
thread.

- **This process's hard faults**, the primary signal. A hard fault is a page
  read from disk into this process: its pageable memory had been paged out.
  Windows keeps the count per process in `SystemProcessInformation`
  (`HardFaultCount`); Linux in `/proc/self/stat` (`majflt`). Nothing the
  engine reads from the pack or the redo log is a fault: both go through
  unbuffered `DirectFile` reads (`expert_cache_design.md` §6.6), and the GGUF
  mapping is not read after load. So the count is the engine's own heap, the
  warm tier's pageable part and the warm KV arenas coming back from the
  pagefile — exactly the paging the tier exists to avoid causing.
- **Available RAM** (`available_physical_ram`) and its fall over the last
  `SLOPE_TICKS`, the early signal: a build allocating gigabytes a second empties
  the free list before the OS starts trimming anyone, and waiting for this
  process's faults would wait until the damage was done.
- **What the tier may still take from the driver**: the GPU-addressable room
  (`gpu_addressable_room`, WDDM's non-local budget) less `ADDRESSABLE_MARGIN`,
  and `cuMemAllocHost`'s own answer.

The system-wide pages-in rate (`host_perf`) is **not** an input: it counts every
process's paging (§2 item 2).

The fault count is smoothed twice: a fast EWMA (`FAST_ALPHA` = 0.5, about 2 s)
that reacts, and a slow one (`SLOW_ALPHA` = 0.1, about 10 s) that decides when
the process is quiet.

## 4. The controller

A pure function of the signals and its own state, `elastic::policy`, called once
a tick; it answers `Hold`, `Grow` or `Shrink(n)` in segments (§5). Every
constant below is a starting value the balloon runs (§8) tune.

### 4.1 Shrink — fast, multiplicative

| Trigger | Shrink |
|---|---|
| **Severe**: fast faults > `SEVERE_FAULTS` (2,000/s), or available < `FLOOR` (1 GiB) | a quarter of the segments held, at least one |
| **Build**: available fell by ≥ `SLOPE_BYTES` (2 GiB) over `SLOPE_TICKS` (3) and is below `HEADROOM` + one segment | as many segments as restore `HEADROOM` (3 GiB), at least one |
| **Pressure**: fast faults > `PRESSURE_FAULTS` (400/s) for `PRESSURE_TICKS` (3) ticks running | one segment |

A shrink is taken at most once per `SHRINK_DWELL` (2 s): the released pages take
a moment to reach the free list and the faults to stop, and shrinking again on a
signal that has not had time to change would empty the tier for one build.

### 4.2 Grow — slow, additive, and only while quiet

A grow is one segment, at most once per `GROW_DWELL` (5 s), when:

- the slow fault average is below `QUIET_FAULTS` (25/s), and has been for
  `QUIET_TICKS` (15 s);
- no shrink in the last `GROW_DWELL`;
- the tier is below its **ceiling** (§4.3) and below what the driver allows;
- available RAM ≥ `HEADROOM` + one segment.

Past that last bound — RAM full of other processes' idle pages — the tier may
still grow, by **pressing** (`page_pressure::press`): writing through a
pageable segment's worth of memory makes the OS trim other working sets and
hand their pages over. Pressing is the growth that takes RAM from minimized
apps (§2 item 1), and it may also trim this process — which its faults then
show, and §4.1 undoes. It is allowed once per `PRESS_DWELL` (30 s), after
`PRESS_QUIET_TICKS` (60) quiet ticks, and never under a ceiling.

### 4.3 Anti-churn — a ceiling with exponential back-off

A shrink taken for **pressure, severe or build** sets a ceiling: the size the tier
held before the shrink, less one segment. Growth stops at the ceiling for the
**back-off** period, which starts at `BACKOFF_MIN` (60 s). When the back-off
expires the ceiling lifts by one segment, a probe.

If pressure recurs within twice the back-off after a ceiling lifted, the next
back-off doubles, up to `BACKOFF_MAX` (30 min). After `RELAX_AFTER` (30 min)
quiet, the back-off halves, back towards the minimum.

So the tier finds its steady size by halving its probing rate each time it
meets the edge: on a machine whose steady size moves only when something else
starts or stops, it probes past the edge once an hour or less, one segment at a
time.

### 4.4 Bounds

- **Ceiling of the whole tier**: every evictable expert, the host partition's
  expert budget (`expert_pinned_budget`), and the addressable room less its
  margin — the bounds the startup sizing uses (`warm_slots_for`), applied at
  every grow.
- **Floor**: none. The pad's churn block makes every expert computable at any
  warm size (`expert_cache_design.md` §4).

## 5. The mechanism

### 5.1 Segments

The pinned part becomes a list of **segments**, each one `cuMemAllocHost` block
of `SEGMENT_BYTES` (1 GiB) cut to whole slots. Kernels read only memory from
`cuMemAllocHost` (`quantized/cuda.rs`, `alloc_host_mapped`: a registered range is
refused by `cuMemHostGetDevicePointer` on WDDM). The startup sizing takes its
pinned part as segments too, so the tier can shrink below its startup size.

Pad slot indices stay what they are: the churn block's `0..churn`, then the
resident region's slots in segment order. A slot's address is its segment's base
plus its offset, looked up in a table the pad holds behind a lock (readers take
it once per read).

### 5.2 The range table

The bucketize kernel and the CPU pool tell a pinned host copy from a VRAM one by
its address: today, inside one of two fixed ranges (`PinnedRanges`, the warm
block and the pad). With segments there are up to `MAX_RANGES` (128) ranges.

They live in a **range table in mapped pinned memory** — `u32 count` then
`(lo, hi)` pairs — whose device address the kernel takes in place of the two
pairs. The address is fixed for the process's life, so a captured graph replays
reading the table as it stands. The host writes it under two rules:

- **A range is added before any entry points into it**: the pair is written,
  then the count, with a system fence between. A segment's slots are published
  only after a read lands in them, long after.
- **A range is removed after no entry points into it and no invocation could
  have snapshotted one** (§5.3): the last pair is copied over it, a fence, then
  the count is decremented. A kernel reading between the two sees the moved pair
  twice, never a pair missing.

### 5.3 Growing and shrinking the pad

The stager owns the pad's book (`PadBook`), so it executes both.

- **Grow**: the controller allocates the segment, adds its range, and sends it
  to the stager, which appends its slots to the book as free churn slots and
  raises the resident capacity by as many. The resident set fills them by the
  ordinary relabel (`resident::plan`) as reads land.
- **Shrink**: the stager marks the last segment's slots **retiring**: off the
  free list, never a read's target. Each held one is evicted as a churn victim
  is (`Stager::evict`: entry retargeted, retire key read after a fence), once its
  pins and carry pins are released. A loading one is evicted when it lands. The
  segment is **drained** when every slot is free and every retire key it took
  is reclaimable (`reclaim.rs`). The stager then hands it back; the controller
  removes the range and frees the block. The resident capacity drops by the
  segment first, so the relabel never promotes into it.
- **Bands** (`resident::Bands`) scale with the capacity: each layer's band is its
  startup share of the *current* capacity, not of the startup one.

### 5.4 What stays

- The **pageable part** of the tier, and its static membership.
- The startup sizing, as the tier's **first** size: conservative, as it is, and
  now only a starting point.
- The driver's margins: `ADDRESSABLE_MARGIN` against the non-local budget is
  checked at every grow, and a refused `cuMemAllocHost` is a ceiling until the
  back-off lifts it.

## 6. Telemetry

On each change, and once a minute:

```
warm tier: elastic segments=N gib=… change=grow|shrink(reason) faults_fast=… faults_slow=…
           available_gib=… ceiling=… backoff_s=… pressed=…
```

The stats table carries the same fields, so `/v1/status` and the GUI can show
the tier moving.

## 7. Tests

- **The policy**, unit tests against signal sequences:
  - a quiet machine grows one segment per dwell up to its bound;
  - severe and build pressure shrink at once, by their amounts;
  - a ceiling holds growth for its back-off, then lifts by one;
  - pressure recurring right after a lift doubles the back-off;
  - a long quiet period halves it;
  - another process's paging, as available RAM falling with no fault of this
    process's, does not shrink below the build rule.
- **The range table**: an added range is seen; a removed one moves the last
  pair over it and is never missing from a concurrent read.
- **The pad book**: grown slots are free churn; retiring slots are never taken;
  a drained segment reports only once every slot is free.
- **The probes**: this process's hard-fault count is read and grows when its
  own pages are forced out and back.

## 8. The balloon harness

`ram-balloon` simulates the machine around the engine:

- **active** RAM: allocated and touched continuously, one byte a page each
  sweep — a build's working set, which the OS must keep resident;
- **idle** RAM: allocated, touched once, then left — a minimized app, which the
  OS may trim.

It runs a schedule of phases (`--phase active=16G,idle=8G,secs=120 …`) with
ramps, and logs once a second its own sizes and hard faults, available RAM, and
— given `--watch motherd` — the watched process's working set and hard faults.

Run on BlackWorld and WhiteWorld, beside a serving motherd:

| Scenario | What it shows | Pass |
|---|---|---|
| idle hog | the tier takes RAM from a minimized app | the tier grows while the balloon is idle; motherd's faults stay quiet |
| build ramp | the tier gives RAM back to a build | shrink within 3 s of the ramp; motherd's faults back under the pressure line within 10 s; the balloon's own faults low |
| square wave | no churn | under a repeating build, one grow and one shrink per cycle at most, back-off visible |
| release | it comes back | after the build ends, the tier regrows after its back-off |

## 9. Build order

1. The per-process fault probe, with its test.
2. The policy, pure, with its tests.
3. The segmented pinned part and the range table, the kernel taking the table.
4. Grow and shrink in the pad book and the stager.
5. The controller thread and the telemetry.
6. The balloon harness.
7. Runs on BlackWorld and WhiteWorld; constants tuned from them, recorded here.
