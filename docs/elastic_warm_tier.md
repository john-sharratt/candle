# The elastic warm tier — pageable expert RAM that follows the machine

**Status:** built (2026-11-21). Builds on the warm tier and the pad of
`docs/expert_cache_design.md` §6.3 and §6.5. The warm tier's **pinned part is
sized once at load and never touched after**; its **pageable part** changes
from a size chosen once at load into one the process keeps adjusting.

**Why not the pinned part.** A first build grew and shrank the pinned part in
`cuMemAllocHost` segments, behind a mapped range table the bucketize kernels
read (a97f820d5, reverted by 955760c89). Pinned memory is the wrong thing to
move: `cuMemFreeHost` may synchronise the device mid-wave, every pinned byte is
charged against WDDM's GPU-addressable budget and the driver's page-lock
ceiling that the uploads need, and it is the one memory the OS cannot take
back anyway — the pageable part is what the OS trims and what a build competes
for. So the kernels keep their two fixed ranges, and only memory no kernel
addresses moves.

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
   Their paging says nothing about whether this process holds too much.
3. **React fast to a build**: when another process starts using RAM, give it
   back within seconds, so the machine recovers before the build thrashes.
4. **Not churn.** A grow that will be undone a minute later costs an
   allocation and a gigabyte of pack reads; a shrink costs the experts it held.
   The tier settles near its steady size and probes past it rarely.
5. **Never touch the pinned part**, nor anything a kernel addresses.

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
- **The allocator's own answer**: a refused chunk is a ceiling (§4.3). The
  driver's budgets do not bound the pageable part: the device never maps it.

The system-wide pages-in rate (`host_perf`) is **not** an input: it counts every
process's paging (§2 item 2).

The fault count is smoothed twice: a fast EWMA (`FAST_ALPHA` = 0.5, about 2 s)
that reacts, and a slow one (`SLOW_ALPHA` = 0.1, about 10 s) that decides when
the process is quiet.

## 4. The controller

A pure function of the signals and its own state, `elastic::policy`, called once
a tick; it answers `Hold`, `Grow` or `Shrink(n)` in chunks of the pageable
part (§5; the policy's code calls them segments). Every
constant below is the value the balloon runs (§8.1) tuned it to.

### 4.1 Shrink — fast, multiplicative

| Trigger | Shrink |
|---|---|
| **Severe**: fast faults > `SEVERE_FAULTS` (2,000/s), or available < `FLOOR` (1 GiB) | a quarter of the segments held, at least one |
| **Build**: available fell by ≥ `SLOPE_BYTES` (2 GiB) over `SLOPE_TICKS` (3), and **projected** `LEAD_TICKS` (2, one shrink dwell) ahead at that rate is below `HEADROOM` + one segment | as many segments as restore `HEADROOM` (3 GiB) at the projected point, at least one |
| **Short**: available below `HEADROOM` for `SHORT_TICKS` (5) ticks running | as many segments as restore `HEADROOM`, at least one |
| **Pressure**: fast faults > `PRESSURE_FAULTS` (400/s) for `PRESSURE_TICKS` (3) ticks running | one segment |

The build rule sheds for where free RAM will be when the next shrink may be
taken, not where it is: shedding only the shortfall of the moment trailed a
1 GiB/s build by a step every dwell, and free RAM touched 0.23 GiB on WhiteWorld
(2026-11-21, balloon run 2) though motherd itself never paged past 56 faults/s.

The short rule is the build rule for a build too slow for the slope: measured on
WhiteWorld (2026-11-21), a 14 GiB balloon that took 84 s to ramp left free RAM at
2.2–4 GiB for 2.5 minutes with the tier untouched, until the floor rule shed a
quarter at 0.90 GiB. The headroom is the machine's, whoever took it.

A shrink is taken at most once per `SHRINK_DWELL` (2 s): the released pages take
a moment to reach the free list and the faults to stop, and shrinking again on a
signal that has not had time to change would empty the tier for one build.

### 4.2 Grow — slow, additive, and only while quiet

A grow is one segment, at most once per `GROW_DWELL` (5 s), when:

- the slow fault average is below `QUIET_FAULTS` (25/s), and has been for
  `QUIET_TICKS` (15 s);
- no shrink in the last `GROW_DWELL`;
- the tier is below its **ceiling** (§4.3) and its bound (§4.4);
- available RAM ≥ `HEADROOM` + one segment.

Past that last bound — RAM full of other processes' idle pages — the tier may
still grow, by **pressing** (`page_pressure::press`): writing through a
pageable segment's worth of memory makes the OS trim other working sets and
hand their pages over. Pressing is the growth that takes RAM from minimized
apps (§2 item 1), and it may also trim this process — which its faults then
show, and §4.1 undoes. It is allowed once per `PRESS_DWELL` (30 s), after
`PRESS_QUIET_TICKS` (60) quiet ticks, only with free RAM at or past `HEADROOM`,
and not within `PRESS_AFTER_SHRINK` (10 min) of a shrink: a shrink says
something is using the machine's RAM, and a press trims a build's live pages as
readily as an idle app's. **The press is a probe**: the controller re-reads free
RAM after it and grows only if the headroom and a whole chunk are free
(`controller::press_made_room`); otherwise the grow is refused, a ceiling for a
back-off.

### 4.3 Anti-churn — a ceiling with exponential back-off

Every shrink sets a ceiling: **the size the shrink leaves**. Growth stops at the
ceiling for the **back-off** period, which starts at `BACKOFF_MIN` (60 s). When
the back-off expires the ceiling lifts by one segment, a probe. (A ceiling one
under the size *before* the shrink let a severe shrink of 18 → 13 grow straight
back to 17 within 70 s while the build that caused it still held its RAM —
WhiteWorld, 2026-11-21.)

**Relief lifts a RAM ceiling whole.** A ceiling set by free RAM falling short —
build, short, or the floor — records it (`Ceiling::ram`). Once free RAM has stood
at `HEADROOM` + `RELIEF_SPARE` (4 GiB) or more, this process quiet, for half the
back-off (`relief_ticks`), the ceiling goes and the tier regrows a segment per
`GROW_DWELL`, the headroom rule still bounding every grow. Run 2 on WhiteWorld
had 24 GiB free once its build ended and took minutes to retake 6 GiB a segment
per back-off. A ceiling set by this process's own faults is never lifted by
free RAM: it marks where this process pages, plentiful RAM or not. A relief
counts as a lift, so pressure returning soon after doubles the back-off — and
with it the next relief window.

If pressure recurs within twice the back-off after a ceiling lifted, the next
back-off doubles, up to `BACKOFF_MAX` (30 min). After `RELAX_AFTER` (1 h, twice
the longest back-off) without pressure, the back-off halves, back towards the
minimum. Shorter, and the quiet a tier spends waiting out its own back-off would
relax it: measured in the policy's tests, a 30-minute relax undid every doubling
at the cap.

So the tier finds its steady size by halving its probing rate each time it
meets the edge: on a machine whose steady size moves only when something else
starts or stops, it probes past the edge once an hour or less, one segment at a
time.

### 4.4 Bounds

- **Ceiling of the pageable part**: the draw's experts past the pinned part, in
  whole chunks, and the host partition's expert budget
  (`expert_pinned_budget`) less the pinned part (`controller::max_chunks`).
- **Floor**: none. The pad's churn block makes every expert computable at any
  warm size (`expert_cache_design.md` §4).

## 5. The mechanism

### 5.1 Chunks

The pageable part is a list of **chunks**, each one ordinary heap allocation of
`CHUNK_BYTES` (1 GiB) cut to whole slots, sector-aligned for the direct reads
(`elastic::paged`). The chunk size is fixed for the process at the model's
(`chunk_slots(stride, evictable)`). The startup sizing takes its pageable part
as whole chunks; a remainder short of one is left for the controller to grow
into.

Warm slot `i` is still membership entry `i`: `0..pinned` the pinned part, then
the pageable slots in chunk order. The membership is drawn once over **every**
evictable expert (`stratified_membership`): the startup tier is its prefix, and
chunk `c` holds the next `per` entries past the pinned part, so a chunk given
back and taken again holds the same experts.

Nothing on the device addresses a pageable slot. Its two readers are host
threads: the stager's readers, which copy a pageable copy into the pad
(`Source::Warm`), and the CPU pool, which computes from it in place
(`cpu_pool::cold::PagedImages`). So no kernel, range table or driver call is
involved in moving one.

### 5.2 Growing and shrinking

The stager owns the residency places, so it publishes and withdraws; the
controller allocates, fills and frees.

- **Grow**: the controller appends a chunk and fills each slot from the pack
  with a positioned direct read (`ExpertPack::read_into`) **on a thread of its
  own** (`controller::Filling`), then, once it lands, sends the stager
  `PagedGrown`: each expert gets its warm slot (`Residency::set_warm`), its
  CPU-pool copy (`PagedImages::set`) and its host-backed mark for the VRAM
  eviction. Nothing names a slot until its bytes are written. The controller
  keeps ticking through a fill — a gigabyte of pack reads takes 1.5–12 s, and a
  fill on its own thread held the controller while a 14 GiB build emptied
  WhiteWorld's free RAM in 11 s, the first shed coming 13 s after the RAM was
  gone (balloon run 3, 2026-11-21). A shrink decided during a fill takes the
  filling chunk first: the fill stops at its next record and the chunk is freed
  unpublished.
- **Shrink**: the controller sends `PagedShrink` for the last chunk. The stager
  withdraws its experts — no warm slot, no CPU-pool copy, host-backed only if a
  resident pad slot still holds one — and reads the readers' key
  (`ReclaimClock::readers_key`) after the clears. The chunk is **drained** when
  no read it started out of the chunk is still in flight and every invocation up
  to the key is done: the pool reads a copy only inside the invocation that
  routed it. The stager answers; the controller frees the chunk, and drains the
  next one a multi-chunk shrink owes.
- **Idle drain.** The key is the latest invocation begun, and while inference is
  idle nothing later begins to prove it done. So a drain that waits a whole tick
  with no invocation begun has the controller read the latest started ticket,
  wait out a context-wide synchronise (only while no segment records,
  `try_without_recording`), and settle the clock past it
  (`ReclaimClock::settled`), then nudge the stager (`Drain`).

### 5.3 What stays

- The **pinned part**, fixed at its startup size for the process: the pad's
  resident region, and the bucketize kernels' two fixed ranges.
- The startup sizing, as the pageable part's **first** size.

## 6. Telemetry

On each change, and once a minute:

```
warm tier: elastic chunks=N paged_gib=… change=grow|refused|shrink(reason) faults_fast=…
           faults_slow=… available_gib=… ceiling=… backoff_s=… pressed=… draining=…
```

and `warm tier: elastic chunk filled chunk=… fill_ms=…`, `warm tier: elastic
chunks freed chunks=… reason=… drain_ms=…` (the controller's `report`, at target
`candle_transformers::expert_lre`; `draining` is the chunks a shrink still owes and
`ceiling` is −1 with none).

The controller also keeps gauges in the expert pipeline's `PipelineStats`:
`warm_paged_chunks`, `warm_paged_bytes`, `warm_paged_slots`, `warm_slots`, and the
counts of chunks grown and shrunk (`warm_grown`, `warm_shrunk`). The batched test's
expert table prints them. The controller's signals (faults, available RAM, ceiling,
back-off) are on the log line only: neither `/v1/memory` nor `/v1/telemetry` carries
the tier's elastic state.

## 7. Tests

- **The policy**, unit tests against signal sequences:
  - a quiet machine grows one segment per dwell up to its bound;
  - severe and build pressure shrink at once, by their amounts;
  - a ceiling holds growth for its back-off, then lifts by one;
  - pressure recurring right after a lift doubles the back-off;
  - a long quiet period halves it;
  - another process's paging, as available RAM falling with no fault of this
    process's, does not shrink below the build rule.
- **The chunks**: slots index across chunks in order, grow and shrink at the
  end, the slots below never move; a slot past the chunks is refused.
- **The membership**: chunk `c` holds the draw's next experts past the pinned
  part; the bound is the lesser of the draw and the budget.
- **The drain**: a shrink waits for reads out of its chunk and for its readers'
  invocations; a settled clock completes everything begun and nothing after.
- **The CPU pool's copies**: a cleared copy is gone, the others stay.
- **The probes**: the Windows process records parse at their offsets from raw
  bytes; this process's own record is found, its resident bytes grow with what
  it touches, and its fault count never falls. Paging a process out on demand is
  not something a unit test can make the OS do; the balloon runs (§8) measure
  the faults themselves.

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
| release | it comes back | after the build ends, the tier regrows once relief has held (§4.3), a segment per `GROW_DWELL` |

### 8.1 Runs

WhiteWorld (RTX 3090 box, 64 GB), 2026-11-20 and 2026-11-21, beside a serving
motherd. Each run's one-second CSV is the balloon's own log; the figures are
from those logs. Phases are written `A14` (14 GiB active, 15 s ramp),
`P10` (10 GiB active pulse, 10 s ramp), `I10` (10 GiB idle, 10 s ramp).

| Run | Build | Schedule | What it measured | What changed |
|---|---|---|---|---|
| 1 | 13d672a80 | A14 over 84 s, held | free RAM at 2.2–4 GiB for 2.5 min with the tier untouched, until the floor rule shed a quarter at 0.90 GiB (stopped early: the harness's schedule stalled) | the short rule (§4.1); the harness ends a phase on the clock |
| 2 | b18bd59e2 | A14 120 s, release 150 s, I10 120 s, release 90 s | the first shed on the first second free RAM was under 3 GiB, then 1–3 GiB steps to t=21 (8.8 GiB in all); free RAM touched **0.23 GiB**; the first regrow 59 s after the last shed, then 1 GiB a minute; motherd's faults ≤ 56/s | the build rule sheds for `LEAD_TICKS` ahead (§4.1); relief lifts a RAM ceiling (§4.3) |
| 3 | 09dd9f0c6 | A14 120 s, release 120 s, P10 60 s, release 60 s, P10 60 s, release 90 s, I10 120 s, release 60 s | the A14 ramp went unshed until t=24 and free RAM sat at **0.00 GiB** for 7 s (motherd's working set trimmed 34 → 17 GiB, faults ≤ 295/s); both pulses shed at ~5 GiB free (minimums 2.10 and 6.10 GiB) with no regrow after them; the final release regrew at 17 s, then 1 GiB per 5 s | a chunk's fill runs on its own thread, and a shrink decided mid-fill takes the filling chunk first (§5.2): the fill held the controller's ticks for 1.5–12 s |
| 4 | 65487f34f | as run 3 | the A14 ramp shed at t=7 with 5.36 GiB free, 8 GiB over 14 s; minimum free RAM **1.78 / 4.10 / 3.14 / 5.79 GiB** (A14 / P10 / P10 / I10); motherd's working set 38 → 31 GiB, faults ≤ 208/s; ended at t=592 by a cluster install, not by the tier | none: the constants above are final |

What run 4 shows of the design, as intended rather than as defects:

- **Relief under a held build.** With the 14 GiB still held, free RAM stood at
  10.2 GiB once the sheds and the OS's trim had made room, past `HEADROOM` +
  `RELIEF_SPARE`; the tier regrew a segment at a time from t=67 to t=131, back to
  its starting size, while free RAM fell to 6.4 GiB and never under the
  headroom. Using RAM the build left free, without faults or a second shed, is
  the steady state the tier is for.
- **The back-off doubles only for a build that recurs within twice it.** The
  first pulse came 181 s after the relief that preceded it — past 2 × 60 s — so
  the back-off stayed at its minimum and the tier regrew in the release after
  it. The second pulse came 60 s after the next lift, the back-off doubled, and
  nothing regrew in the 90 s release after that one.

## 9. Build order

1. The per-process fault probe, with its test.
2. The policy, pure, with its tests.
3. The pageable part in chunks, the membership drawn over every evictable
   expert.
4. Publish and withdraw in the stager, with the drain.
5. The controller thread and the telemetry.
6. The balloon harness.
7. Runs on BlackWorld and WhiteWorld; constants tuned from them, recorded here
   (§8.1).
