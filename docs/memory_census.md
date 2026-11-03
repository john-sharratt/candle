# The memory census

motherd's private memory grows by gigabytes an hour in memory nothing it
reports accounts for. This document designs the counting that finds where:
a census of what the process allocates, what the OS has committed for it, and
what each substrate holds. The counts appear in the usual places: `/v1/memory`,
the `/v1/telemetry` history, and the log. Every figure is cheap enough to stay
on in production, so the next growth is visible the day it starts, not weeks
later from a crash.

## 1. What is known

BlackWorld (RTX PRO 5000, 194 GB host, build 4344), 2026-11-02:

| | 20:20:07Z | 20:40:48Z | Change |
|---|---|---|---|
| motherd private bytes | 225.70 GB | 227.06 GB | +1.37 GB (~4.0 GB/h) |
| Machine commit / limit | 249.95 / 280.66 GB | 251.36 / 280.66 GB | ~7 h to the limit |
| Threads | 222 | 223 | flat |
| Handles | 1,235 | 1,289 | +54 |
| zend records applied from the other member | 253,402 | 296,813 | +43,411 |
| npcd records applied | 170,327 | 173,167 | +2,840 |

Over the same window every figure `/v1/memory` reports was flat or falling:
pinned experts 66.1 GB (fixed at load), held payloads 10.9 GB, warm KV 7.1 GB,
gallery 0.13 GB. Telemetry put the hour before at +6.6 GB of RSS.

Roughly 65 GB of BlackWorld's private bytes is unaccounted for: private 225.7
GB, less pinned experts 66.1, less the system-memory backing Windows commits
for VRAM in use (~74.7), less held payloads, warm KV and reserved weights
(~20.2).

The growth tracks records applied: ~30 KB per zend record. Two read-only code
sweeps (this session and BlackWorld's) costed every structure that grows per
record at 0.2–1 KB:

| Structure | Per record | Bounded? |
|---|---|---|
| Replica location index (`persistence/replica/location_index.rs`) | 48 B | No: never shrinks; `forget_segments` empties slots and keeps the length |
| Record accounting `live_sizes` (`persistence/accounting.rs`) | ~100 B per chunk key | Reset at compaction |
| `StreamRuntime.chunks` (`substrate.rs`) | ~56 B + node | Bounded by live chunks |
| `ResidenceSlab.slots` (`substrate/residence.rs`) | 72–80 B per turn | No: discard flags a slot dead and keeps it |
| Custom objects | 0.5–2 KB + text | Ceased keys expire (`forget_expired_tombstones`) |
| `PendingChildren` (`substrate/pending.rs`) | the record's payload | **No**: held until the timeline registers, so forever if it never does |

None explains 30 KB a record. The leading explanation is **heap retention**:
each applied record of about a chunk's size passes through several buffers its
own size (`Incoming::parse`, the padded encode in `append_carried`, the staged
`LogFile.pending`, the `WalkEntry` payload, the `Arc<[u8]>` copy in
`apply_walker_entry`), all allocated and freed. Allocations that size are served
from the Windows process heap's back end, and a heap churned by mixed lifetimes
can keep freed memory committed. The laptop's unexplained ~0.9 GB/h of
small-heap growth in October (`evidence/motherd-oom-2026-10-24/FINDINGS.md`)
would be the same thing.

Unproven, because nothing counts it. That is what this design fixes.

**Constraints**, from the user: no debugger attached to a live motherd, no WPR
heap trace. The counting must come from inside the process.

## 2. The question each count answers

| Count | Answers |
|---|---|
| **Live heap bytes**, counted by the allocator | How much our Rust code holds right now |
| **Live heap bytes per size class** | Which allocation size holds it, which narrows the owner |
| **Heap committed**, from the OS | What the process heap has committed to hold it |
| **Private bytes**, from the OS | The process's whole commit charge |
| **Substrate gauges**, per tenant | What each per-record structure holds |
| **Applied record mix** | What kind of record each member is applying, and how many bytes |

They combine into a verdict, which the log states (§7):

| Private | Heap committed | Live heap bytes | Verdict |
|---|---|---|---|
| grows | grows | flat | **Heap retention**: freed memory kept committed. The fix is the allocator. |
| grows | grows | grows | **A real owner**: the size class that grows, and the substrate gauge that grows with it, name it. |
| grows | flat | flat | **Outside the Rust heap**: a native library's own heap (libgit2, the CUDA driver), or direct `VirtualAlloc`. The untracked remainder grows. |

## 3. Change 1: the counting allocator

A `#[global_allocator]` that forwards every call to `std::alloc::System` and
counts it. Its own crate, `heap-census/`, with no heavy dependencies, because
every daemon binary links it and its tests must run without an engine or a
card. One concern per file:

| File | Holds |
|---|---|
| `classes.rs` | The size-class function and its table |
| `shards.rs` | `Census`: the sharded counters and their sum |
| `counting.rs` | `Counting`, the `GlobalAlloc` wrapper, and the process's `CENSUS` |
| `os.rs` | The OS's figures (§4) |
| `reading.rs`, `history.rs` | One reading, and an hour of them |
| `verdict.rs` | The §2 verdict over a window |
| `report.rs` | The census as `/v1/memory` carries it |

### 3.1 What it counts

- `live_bytes`, `live_count`: requested sizes allocated and not yet freed.
- `allocated_total`, `freed_total`: cumulative bytes, so a rate can be read
  between two samples (churn, not just level).
- Per **size class**, `live_bytes` and `live_count`. Class *k* holds requests of
  (2^(k−1), 2^k] bytes, for *k* = 0..48, with requests above 2^47 in the top
  class. A 30 KB record buffer lands in class 15 (16–32 KiB), and a 48 B
  location slot's vector growth lands wherever its capacity does.

`realloc` moves bytes between classes: subtract the old size from its class,
add the new size to its own. `alloc_zeroed` counts like `alloc`.

### 3.2 Keeping it cheap

The allocator is on every path, including threads that allocate in tight loops,
so a single shared counter would put 220 threads on one cache line. Instead:

- The counters are **sharded 64 ways by thread**, each shard aligned to 128
  bytes so no two share a line. A thread is given a shard on its first
  allocation, round-robin, and keeps it in a `const`-initialised thread-local
  with no destructor, which needs no allocation and is never torn down while
  the thread runs.
- An allocation is counted on the allocating thread's shard and a free on the
  freeing thread's, which need not match. A shard's own figures can therefore
  run negative, and are kept as wrapping `u64`s. Only the **sum across shards**
  means anything, and it is exact, because every add and subtract lands on
  some shard. A reading clamps a class at zero for the instant a free is seen
  before its allocation.
- Every update is a `Relaxed` `fetch_add` / `fetch_sub`: three per allocation
  (class bytes, class count, bytes ever allocated), two per free. A reading
  sums the shards while threads go on counting, so it can see a call
  half-counted; the error is bounded by the calls in flight.
- No allocation, lock, log or panic inside the allocator. The arithmetic wraps.

Cost, measured (`heap-census/tests/overhead.rs`, release, on the RTX 4090
Mobile laptop): 16 threads allocating and freeing 32 B, 4 KiB and 30 KB blocks,
median of five alternated runs.

| | ns per allocation or free |
|---|---|
| `System` | 336.3 |
| `Counting` | 344.4 |
| Overhead | **2.4%** |

The gate refuses more than 5%. The microbenchmark does nothing but allocate, so
the share of a real forward is smaller.

### 3.3 Wiring

`motherd/src/main.rs`:

```rust
use heap_census::Counting;

#[global_allocator]
static HEAP: Counting = Counting;
```

The standalone `zend` and `npcd` binaries get the same two lines in their own
mains, so every daemon that can run this engine is counted.

### 3.4 What it cannot see

Memory not allocated through Rust's global allocator: `cuMemAllocHost` (already
tallied as pinned), CUDA device allocations and the system-memory backing
Windows commits for them, memory-mapped files, and native libraries that call
the C runtime's `malloc` directly. libgit2, behind the journal sync, is one of
those. The remainder (§5) is what catches them.

### 3.5 Tests

- **Unit**: the size-class table at exact boundaries (1 → 0, 2 → 1, 3 → 2,
  4096 → 12, 4097 → 13, 30,000 → 15, 2^47 + 1 → 48). An allocation and its free
  counted exactly. A free on another shard than its allocation balancing in
  the sum. Sixteen threads leaving the census at zero live.
- **The positive leak test** (`tests/positive_leak.rs`, its own binary, the
  allocator installed as the global one): 2,000 buffers of 30,000 B kept. Class
  15 grows by exactly their bytes, the commit charge by at least 90% of them,
  and the verdict is an owner whose top class is 15.
- **The negative leak test** (`tests/negative_leak.rs`): the same buffers
  allocated and freed twenty times over. Live bytes stay within 1 MiB, the
  churn counts all 1.2 GB, the commit charge grows by less than two rounds'
  worth, and the verdict is never an owner.

## 4. Change 2: what the OS has committed

`heap-census/src/os.rs`, reading two things on request:

- **Private bytes**, the process's commit charge. On Windows,
  `GetProcessMemoryInfo` with `PROCESS_MEMORY_COUNTERS_EX`, field
  `PrivateUsage`. On Linux, the sum of `RssAnon` and `VmSwap` from
  `/proc/self/status`. `sysinfo`'s `virtual_memory()`, which `/v1/memory`
  reports today, is address space, not commit, and on BlackWorld reads 541 GB
  against 226 GB private.
- **The process heap's own summary.** On Windows, `HeapSummary(GetProcessHeap())`
  returns `cbAllocated` (bytes in allocated blocks) and `cbCommitted`. Their
  difference is committed memory the heap holds free: the retention §2 is
  asking about, measured directly by the OS. On Linux, glibc's
  `mallinfo2().uordblks` and `.fordblks`.

`HeapSummary` holds the heap's lock while it runs, and every allocation in the
process waits for it. So it is read **once a minute**, by the census task on a
blocking thread (`host/src/memory_census/`), and its own duration is reported
beside it (`read_us`). The telemetry samples and `/v1/memory` requests reuse
that minute's figure. Only the commit charge and the allocator's counts, which
lock nothing, are read fresh.

Dependency: `windows-sys`, features `Win32_System_ProcessStatus`,
`Win32_System_Memory` and `Win32_System_Threading`, target-gated to Windows.

Tests: the readings are non-zero and ordered (`cbAllocated ≤ cbCommitted`,
heap committed ≤ private bytes). After allocating and touching 256 MiB through
the counted allocator, private bytes and heap committed have both risen by at
least 256 MiB, and live heap bytes by exactly 256 MiB.

## 5. Change 3: the substrate gauges

Each tenant's substrate (zend's and npcd's) counts what its per-record
structures hold. Each structure gets a `gauge()` beside its own code, and two
small files gather them: `candle-conversation/src/substrate/gauge.rs` (the
substrate's maps) and `candle-conversation/src/persistence/gauge.rs` (the
persistence layer's). `candle-conversation/src/store_gauges.rs` is the
registry the gauges are published to:

| Gauge | Source | Counted |
|---|---|---|
| `location_index` | `LocationIndex` | per origin: `len`, `capacity`, `held` (non-empty slots), bytes = capacity × 48 |
| `pending_children` | `PendingChildren` | timelines waiting, records waiting, payload bytes — kept as running counts, so reading them walks nothing |
| `residences` | `ResidenceSlab` | slots ever allocated, hot, warm, dead |
| `custom_objects` | substrate maps | live, ceased, payload bytes, by key prefix up to the second `/` (`mcp/chat-said`, `mcp/note`, `npc/…`) |
| `streams` | `StreamRuntime` map | streams, chunk locations |
| `accounting` | `RecordAccounting.live_sizes` | entries |

Every gauge is a count over in-memory maps, O(entries), so it is not run on any
request path. The **persistence thread computes them once a minute**, in a
phase 2.7 right after the warm-tier purge (`persistence/thread.rs`), where it
already holds the store off every request path. It publishes the result, with
how long the gauge took, to `store_gauges`, which `/v1/memory` and the census
log read. A request never walks a map.

**The applied record mix** is counted at the source: both remote-apply paths
(`persistence/remote/apply.rs`) tally each applied record's type, count and
payload bytes into the persistence layer's `AppliedMix`
(`persistence/applied_mix.rs`), on the log's owner, under the `&mut` it already
holds. This tells us what the 43,411 zend records were. If they were mostly
`Chunk` records of ~30 KB, the heap-retention lead is confirmed in its own
terms.

Tests, all with exact figures:

- A `LocationIndex` given oseqs 1..=1000 for one origin reports `len` 1000 and
  `held` 1000. After `forget_segments` drops their segment, it reports `len`
  1000 and `held` 0. That is the unbounded case, asserted as such.
- `PendingChildren` holding three records of 100, 200 and 300 B under two
  timelines reports 2 timelines, 3 records, 600 B. Releasing one timeline
  reports the remainder.
- Custom objects: two live under `mcp/note/…` and one ceased under
  `mcp/chat-said/…` report exactly that, with their payload bytes.
- The record mix after applying 5 chunks of 30,000 B and 2 labels counts
  `Chunk: 5, 150,000 B` and `Label: 2`, with the labels' bytes.

## 6. Change 4: in `/v1/memory` and `/v1/telemetry`

### 6.1 `/v1/memory`

`host/src/telemetry/memory.rs`'s `process` block gains the census in two
parts, and a new `substrates` array carries each store's gauge:

```json
"process": {
  "working_set_bytes": 145234403328,
  "virtual_bytes": 542179999744,
  "heap": {
    "private_bytes": 227063132160,
    "os_heap": {"allocated_bytes": …, "committed_bytes": …,
                "free_committed_bytes": …, "read_us": …},
    "allocator": {
      "live_bytes": …, "live_count": …, "allocated_total": …, "freed_total": …,
      "classes": [{"class": 15, "upto_bytes": 32768, "live_bytes": …, "live_count": …}, …]
    }
  },
  "census": {
    "at_unix_ms": …, "untracked_bytes": …,
    "machine_committed_bytes": …, "machine_commit_limit_bytes": …,
    "growth": {"over_ms": …, "private_grew": …, "committed_grew": …,
               "live_grew": …, "verdict": "retention"}
  }
},
"substrates": [
  {"dir": "D:\\prog\\substrate", "gauged_unix_ms": …, "took_us": …,
   "substrate": {"streams": …, "chunk_locations": …, "pending": {…},
                 "residences": {…}, "custom_objects": [{"prefix": "mcp/note", …}, …]},
   "persistence": {"location_index": [{"origin": …, "len": …, "capacity": …,
                                        "held": …, "bytes": …}, …],
                   "location_index_bytes": …, "accounting_entries": …,
                   "applied": [{"record_type": "chunk", "count": …, "payload_bytes": …}, …]}},
  …
]
```

`heap.private_bytes` and the allocator's counts are read fresh on each request;
`os_heap` is the census task's last minute (§4). `allocator` is absent in a
binary that did not install the counting allocator, and `census` before the
first reading. `classes` lists only non-empty classes. A store is named by its
directory, which says whose it is.

`untracked_bytes` = private bytes − heap committed (OS) − pinned host bytes −
the VRAM backing estimate (`accounting.device_in_use_bytes`, the figure
FINDINGS.md showed Windows commits, untouched, for VRAM in use; zero off
Windows). It is what none of the census explains: native heaps, direct
`VirtualAlloc`, anything else.

### 6.2 `/v1/telemetry`

The history (`host/src/telemetry/ring.rs`, `Series`) gains five columns, sampled
every 2 s for the hour the ring keeps:

| Column | From |
|---|---|
| `private_mib` | private bytes |
| `heap_live_mib` | the allocator's live bytes |
| `heap_committed_mib` | the OS's heap committed |
| `untracked_mib` | §6.1 |
| `heap_churn_mib_s` | `allocated_total` delta per second |

With these on one chart the verdict in §2 is readable by eye: private and heap
committed climbing together over a flat `heap_live_mib` is retention.

## 7. Change 5: the log

Target `host::memory`, so `cluster_logs` and `/ws/logs` find it by name.

- **Every 5 minutes, one INFO line**, `memory census`: private, heap live, heap
  committed, untracked, each with its change over the last 30 minutes, the
  heap summary's own read time, and the three size classes whose live bytes
  grew most. Beside it, one `store census` line per tenant store: location
  index size, slots and held slots, accounting entries, streams, chunk
  locations, records pending, residence slots and dead slots, the largest
  custom-object prefixes, and the largest applied record types.
- **A WARN when private bytes have grown more than 2 GiB in 30 minutes**,
  `memory growing`, with the §2 verdict as a field: `retention`, `owner` (with
  the size classes that grew most), or `outside the heap`, and the untracked
  figure then and at the reading before. The `store census` lines beside it
  carry the gauges an owner is matched against. It repeats at most every 30
  minutes while the growth continues, and at once when the verdict changes.
- **A WARN at 85% of the machine's commit limit**, and an ERROR at 95%, with
  the time to the limit at the current slope. Said again when the level
  changes either way, and every 30 minutes while it holds. BlackWorld crossed
  89% on 2026-11-02 with nothing saying so.

The 30-minute window comes from a small ring of census readings (one a minute,
kept for an hour) in `heap-census/src/history.rs`, separate from the telemetry
ring, so the log's arithmetic does not depend on the page's. The task that
takes them, logs them and serves the last one is `host/src/memory_census/`.

## 8. Built behind a feature

The census is a Cargo feature, `memory-census`, like `profile` and
`tensor-assert`: compiled in or not, never switched at run time. It gates the
three things that do work — the counting allocator each binary installs, the
host's once-a-minute census task, and each store's gauge pass — and leaves the
types and the readers built, so `/v1/memory` and `/v1/telemetry` report the
census absent without it. `motherd` turns it on by default while BlackWorld's
growth is being found; `zend` and `npcd` take it when built with
`--features memory-census`. The feature runs down through `zend`, `npcd`,
`host`, `candle-conversation` and `substrate-git`, each of which names its
threads' scopes (§10) only with it.

## 8.1 The order of the changes, and what each one settles

| # | Change | Deployed alone, it tells us |
|---|---|---|
| 1 | Counting allocator (§3) | Whether live heap bytes grow |
| 2 | OS commit and heap summary (§4) | Whether the heap is holding freed memory committed |
| 3 | Substrate gauges and record mix (§5) | Which structure grows, and what is being applied |
| 4 | `/v1/memory` and `/v1/telemetry` (§6) | The same, in the usual tools, with history |
| 5 | The census log (§7) | The verdict, stated, and a warning before a commit crash |

Changes 1–4 are small and independent enough to land as four commits but ship
as **one build and one deploy**, because each deploy restarts both members and
every restart resets the growth being measured. Change 5 lands in the same
build.

After the deploy, two `/v1/memory` samples 30 minutes apart give the verdict.
The first `memory census` line gives it by itself.

## 9. What follows from each verdict

- **Retention**: the fix is the allocator. The first step is to turn on the
  Windows segment heap for motherd (an application-manifest setting, no code).
  It returns freed memory to the OS more readily than the NT heap's back end.
  If that is not enough, a Rust allocator built for this (mimalloc), behind the
  same `Counting` wrapper so the census stays. Either way, the census shows
  whether heap committed then tracks heap live. Separately, and whatever the
  verdict, the apply path's per-record copies (§1) get trimmed, one copy at a
  time, each change measured by the census's churn column.
- **An owner**: bound it at the structure that grows, with a test that the
  gauge stays flat under a replay of the same records.
- **Outside the heap**: `untracked_bytes` grows. The suspects are libgit2's heap
  (the journal sync, every 5 s), the CUDA driver's host allocations, and mapped
  files. Each can be counted from its own side: libgit2 accepts a custom
  allocator, which can route it through `Counting`.

Whatever the verdict, the location index and the residence slab are bounded
too. They grow with every record and never shrink, which is a slower version of
this same failure.

## 10. Scopes: which code made the blocks

An owner verdict names a size class, and a class names a band of sizes, not the
code. On BlackWorld (2026-11-03) the owner was ~600 MiB/h of 64–128 KiB blocks,
which no gauge matched. A stack trace per allocation would name it and cost far
more than the census itself, so the census names the code at compile time
instead.

- **A scope is a name in the code** (`heap-census/src/scope.rs`): one of the
  `Scope` enum's values — the scheduler, the persistence pass, the log's owner,
  a remote batch, its adoptions, the summariser, a section's seal, the
  repositories' follower, the journals, a tenant's load, the log's maintenance,
  the census itself, the cluster's sockets, and the async runtime. A thread sets
  its own as it starts (`set_thread`); a call narrows it while a guard lives
  (`enter`). The guard is not `Send`, so it cannot be held across an `.await`
  and name the next task's blocks.
- **A block of 4 KiB or more carries its scope** in a header in front of it:
  the alignment's size, at least 16 bytes, its last byte the scope. Whether a
  block has one is decided by its layout alone — at least 4 KiB, aligned to at
  most 64 — which the free is handed too, so the free reads the scope back and
  takes the block off it on whichever thread frees it. The header costs under
  0.4% of such a block; a smaller block is counted by class only, where a
  header would be a large share of it.
- **Counted per scope per class**, in the same shards as the classes, with the
  same `Relaxed` adds. The cost is one thread-local read per allocation, and
  for a labelled block one byte written and read; the overhead gate (§3) times
  it with 4 KiB and 30 KB blocks in the mix.
- **Reported** under `/v1/memory` `process.heap.allocator.scopes` (each scope's
  live bytes, count and classes), and named in the owner verdict and the
  `memory census` and `memory growing` lines as `grown_scopes`.

A scope names the code that **made** a block, not whatever holds it after: a
buffer built in a remote batch and kept by the substrate counts under
`remote_apply`. That is the question the narrowing asks; when a scope is too
broad, a guard inside it splits it, and the next deploy narrows again.

**The window starts when the process is loaded.** A load grows the process by
the tens of GB its model and stores take, and a window reaching back into it
named that as growth (BlackWorld, 2026-11-03: "+4.4 GB in 30 min", of which the
growth was ~0.4 GB). The census clears its readings when every tenant is proven
(`memory_census::loaded`), so a verdict is only ever drawn after the load.
