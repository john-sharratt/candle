# Reprojection after the pool-first revision: where the time goes, and the plan

**Status: plan.** Measured 2026-11-18. The pool-first projection
(`docs/hit_driven_projection.md` §7.2, 8088a2457) stays as it is: its
selection, the pool, the shares and the allowance are not reverted. Each step
below makes a part of a reprojection fast **where it runs**, on the scheduler
thread inside the decode quantum. None moves work to another thread, defers it
or slows its cadence. Each part is named in the reprojection census and the
step census, and is measured before and after.

## 1. The measurement

BlackWorld, build `5211-8088a2457`. One private chat, the only stream decoding:
*"read `projection/fill.rs` and `projection/dialogue_fill.rs`, then explain how
a dialogue's slot is filled when the room runs short"*. Two file reads and an
answer of about 600 tokens, 20:05–20:07 UTC.

**The reprojection census** (`scheduler::reproject`, one line a minute), for
the minute ending 20:06:04:

| | total | per reprojection | share of a rebuild |
|---|---:|---:|---:|
| reprojections | 48 rebuilt, 1 kept | | |
| rebuilt, end to end | 16,218 ms | 338 ms | 100% |
| `score` — the belief scan | 8,388 ms | 175 ms | 52% |
| `pool` — the pool refresh | 4,460 ms | 93 ms | 27% |
| `ws` — working-set observe | 380 ms | 7.9 ms | 2% |
| `lift` | 241 ms | 5.0 ms | 1.5% |
| `seal` | 160 ms | 3.3 ms | 1% |
| `project` | 146 ms | 3.0 ms | 1% |
| `probe` | 49 ms | 1.0 ms | |
| `hits` | 31 ms | 0.6 ms | |
| `snapshot` | 26 ms | 0.5 ms | |
| the rest: `elevate`, `whole`, `build`, `glue`, `finish`, `inject`, `view` | ≈ 2.3 s | ≈ 48 ms | ≈ 14% |

The last row is the remainder. The log line is cut before those fields, so
step 0 reads them.

**What that costs the decode.** The scheduler spent **16.2 s of the minute,
27%,** reprojecting.

> **Corrected 2026-11-18.** An earlier version of this section read the wave
> heartbeat as the decode rate and gave ~17 tokens/s against a forward bound of
> ~31. Neither number is a decode rate. The rate is committed tokens
> (`/v1/telemetry` `decode_tps`), and a turn's own rate is its tokens over the
> time its forwards ran (`turn complete … tps=`, `DecodeState::decode_busy_us`).
> On build `5228` at 22:17 UTC, one chat alone, the forwards ran at
> **111–115 tokens/s**. A tool round took 6–9 s from submit to completion:
> 1.7–2.3 s of prefill, 1.1–1.8 s of forwards, and 3–5 s between them. The
> census for that minute has 29 rebuilds at 467 ms each, about one per 32
> tokens decoded, which is more than those 32 tokens' forwards (~290 ms). The
> gap is the time between forwards. Reprojection is most of it.

**Before the chat, the same build:** reprojections that kept their selection
took about **1 ms each** (`kept=19 kept_ms=18.6`, `kept=15 kept_ms=13.1`). The
cost is in what a rebuild does, and in how often a reprojection rebuilds.

**The pool refresh, every reprojection** (`scheduler::pool`, once a second):

- `pool:index_read`: 8–155 ms, reading **2.8–127 MB** of index pages over
  15–101 turns, every time;
- 167–197 pieces pooled, of which 6–25 admitted and 3–37 dropped per refresh;
- the same strongest pieces read again each second: `dialogue_fill.rs` (24
  pages), `hit_driven_projection.md` (36), `fill.rs` (30);
- the fill itself: about 3 ms (`fill: each cap's use`), not a cost.

**Rebuilds, not keeps:** 48 of 49 reprojections rebuilt. The fill placed 23–28
pieces, a different set nearly every time.

## 2. The bottlenecks

### B1. The belief scan, 175 ms per reprojection (52%)

`Substrate::score_beliefs` (`projection/resolver.rs`) scores the target layer's
collections, plus every belief-driven turn group in every layer, against the
probe. The pool is now the working set: up to 2M tokens on BlackWorld
(`PoolSize::for_card`), 167–197 pieces here. A scan that covers the working-set
groups therefore covers far more turns than the locked working set it scanned
before.

**Not yet shown:** which side of the scan grew — collections or groups, the
GPU arena launches, a CPU fallback, or the host walk that assembles a group's
segments. The scan writes that split on every reprojection, as the `belief scan
phase split` event and the `beliefs:*` profile rows. It names `group_turns`,
`files`, `arena_launches`, `galleries_built`, `arena_index_built` / `stale`,
`walk_us`, `assemble_us` and the kernel and readback times. Step 0 reads it
before any change.

### B2. The pool refresh, 93 ms per reprojection (27%)

`refresh_pool` (`scheduler/pool_refresh.rs`) pools the slot's placed pieces and
the working set's members. It then reads every pooled piece's index pages again
(`pool:index_read`), unthrottled, on every reprojection. A sealed turn's index
pages never change, and most of the pool is the same from one reprojection to
the next: 6–25 pieces enter per refresh, out of 167–197. So nearly all of the
2.8–127 MB read per refresh was read a second earlier.

### B3. Every reprojection rebuilds (48 of 49)

A reprojection keeps its parent when the selection's identity is unchanged.
That took about 1 ms. Under the pool-first fill the placed set changes at
almost every boundary, so nearly every reprojection pays a full rebuild:

- lift, snapshot, seal, move and free;
- elevating the new working set hot;
- building and gluing the segments;
- finishing and re-injecting the tail;
- carving the view;
- the KV traffic behind them. In the log, within the same seconds: `kv_gap`
  installs of 13–2,252 rows, a KV compaction pass most seconds (up to 97,881
  moves), and hot→warm passes of 2–289 ms.

**Not yet shown:** how much of each rebuild is the set actually changing,
versus the whole slot being rebuilt for a change of a few units. Each boundary
admits at most a third of a working-set kind's cap (`Allowance`), so the change
per boundary is bounded, but the rebuild is whole.

### B4. The remaining gap

Reprojection explains about 0.6 s of the ~0.9 s per 2.1 s window that is not a
forward. The rest has to be named, by the step census's bands, before it is
worked on.

## 3. The plan

### Step 0 — attribute every millisecond (instrument, no behaviour change)

- **The census line in full.** `reprojection census` is cut in the log reader
  after `release_ms`. Write it as two lines: `prepare`, from `probe` to
  `truncate`, and `apply`, from `elevate` to `view`. Then every part is
  readable through `log_grep`.
- **The scan split as a log line**, once a minute, summed like the census:
  - `belief scan census`: every field of `belief scan phase split`, summed;
  - the counts: groups, turns and files scanned;
  - the arena launches, index builds and stale or absent indexes.
- **The rebuild's size:** units kept, entered and left per rebuild, and the
  tokens entered and left, summed in the census.
- **The step census bands** for the same minute, to name B4.
- **A fixed replay.** The chat of §1, on BlackWorld, quiet, as a scripted
  `zen_chat_send` with the question verbatim. It is run before and after every
  step, and the census lines are compared.

Gate: every part of a rebuild and every band of the minute is named, with
nothing left as "the rest".

### Step 1 — the pool keeps what it read (B2)

- **A pooled piece's index pages are read once, when it enters the pool, and
  held while it stays pooled.** They are released when it leaves. The pages are
  of sealed turns, so they cannot go stale while the piece is pooled. A piece
  whose timeline gains turns (a reading still being written) reads only the new
  turns' pages.
- `refresh_pool` reads only the entering pieces' pages: 6–25 pieces here, not
  167–197.
- Ranking by strength reads the held pages. It does not read through the index
  again.

This is the pool's own state, tied to membership: it is not a cache that can
go stale or be rebuilt. Memory is bounded by the pool's size, which
`PoolSize` already bounds per card.

Target: `pool` per reprojection from 93 ms to the cost of the entering pieces'
reads, under 10 ms. `pool:index_read` bytes per refresh fall to the newcomers'.

Tests: a piece's pages are read once across refreshes while it stays; an
entering piece's are read; a leaving piece's are released; a piece that grows
reads only its new turns; the ranking is identical, to the bit, to a full
re-read.

### Step 2 — the scan, fast where it is (B1)

Ordered by what step 0's split shows. The changes, each only where the split
points:

- **One launch over every scanned turn group**, reading each pooled piece's
  sign segments through a descriptor table (`AGENTS.md` invariant 2b), not one
  launch, gallery or upload per group or file.
- **A piece's gallery index is built when it enters the pool**, with step 1's
  pages, and kept with it. Its build is not repeated per reprojection, and a
  reprojection never sees it stale (`arena_index_stale`) or absent.
- **No host walk per reprojection.** The segments each group scans are
  assembled once per change of membership. They are not walked out of the
  substrate on every probe.
- **No CPU fallback on the live path.** A group that falls to the CPU per-file
  scan names itself in the split, and is moved to the arena path.

Target: `score` per reprojection from 175 ms to under 20 ms, the scan's cost
before the pool grew the working set.

Tests: scores are bit-for-bit those of the current scan for the same probe and
pool, the parity `examples/gpu_belief_parity.rs` already checks between the GPU
and CPU paths; one launch for any number of groups; no index build on a
reprojection whose pool did not change.

### Step 3 — a rebuild that costs what changed (B3)

The selection changes at almost every boundary, by a bounded amount. The fix is
to make a rebuild's cost follow that amount, not to keep the selection still.

- **Measure first.** Step 0's units kept, entered and left per rebuild. A
  rebuild whose set is unchanged and whose order alone moved must not happen at
  all: the identity compares what is placed and where. If the identity counts
  pool membership or scores rather than placement, that is fixed first.
- **Units that stay keep their K/V where it is.** Keys are stored un-rotated
  and RoPE'd at read (`docs/thinking_span_projection.md`). A unit that moves to
  a new position needs no copy, only its new base. The rebuild elevates,
  places and glues only the entering units and the boundaries they touch. It
  frees only the leaving units. It reseals only what the change cut.
- **The KV traffic follows.** The `kv_gap` installs, compaction passes and
  hot→warm passes of §2 B3 are counted per rebuild in the step census. Each is
  then sized to the change.

Target: a rebuild's apply half (`elevate` to `view`) proportional to the units
entered and left. A boundary that admits nothing costs what a keep costs today,
about 1 ms.

Tests: for a scripted pool and boundary sequence, the K/V the slot holds after
a delta rebuild is identical, row for row, to a full rebuild of the same
selection. A rebuild with nothing entered or left touches no K/V. One that
admits one unit elevates and glues that unit alone.

### Step 4 — the rest of the decode gap (B4)

Whatever step 0's step census names beyond reprojection, taken the same way:
named, measured, made fast in place.

## 4. Done when

On the fixed replay of §1, one stream on BlackWorld:

- reprojection under **5% of the minute** (from 27%);
- a rebuild under **60 ms** on average (from 338 ms), with `score` under 20 ms
  and `pool` under 10 ms;
- a tool round's time between forwards no more than its forwards' own (from
  3–5 s against 1.1–1.8 s), so the committed rate is within reach of the
  forward's ~110 tokens/s;
- the battery of `docs/results/dependency_battery_2026-11-19.md` not below its
  51.0%, because the selection has not changed.

Each step is committed with its census lines before and after, in
`docs/results/`.
