# KV Tier Migration & Persistence — VRAM ↔ RAM ↔ NVMe

> **Status — Design v7; BUILT (2026-07), with two divergences.** Specifies the
> full three-tier KV-cache storage path: the clean-slate VRAM↔RAM migration
> kernels, the RAM warm tier, and the append-only NVMe redo log of
> content-addressed streams — a **complete, self-contained substrate image**
> (§5.7). The persistence layer is a mandatory, generalized module in
> `candle-conversation` (§13); it replaces the existing ad-hoc
> `migrate_to_cpu` warm path and retires the redundant `store.rs` /
> `provenance/` persistence (§7). §16 is the executable, nine-phase
> implementation plan — **every feature is in scope; nothing is deferred,
> stubbed, or left as a `TODO`.**
>
> **What actually shipped vs. this design** (verified 2026-07-17): the
> `kv_pack`/`kv_unpack` migration kernel, the hot→warm→cold pipeline
> (`persistence/thread.rs`, `elevate.rs`), the redo log + skip-load + recovery +
> compaction, and warm-backed eviction (`evict_hot_to_free`) are all **built and
> live**. **Two deliberate divergences remain from the target shape:** (1) the
> warm tier lives in **pageable CPU arenas**, not the pinned host-buffer pool of
> §10 — `warm_pool.rs` was never created, so warm↔hot DtoH/HtoD runs at ~½ PCIe
> bandwidth; (2) the hot→warm copy runs on the **primary CUDA stream**, not the
> dedicated overlap copy stream of §11 (a correctness-motivated choice to avoid
> shared-state races — see `persistence/thread.rs`). Both are the remaining
> performance work; the tier is functional without them.

---

## 1. Abstract

The chunked KV cache has **no durable storage** — a conversation's KV state
is lost when the process exits. An ad-hoc VRAM→CPU path *does* exist for
system-prompt sections, but it is unbatched, slow, and incorrect; this
design **replaces it wholesale** (§7). This document specifies a
**three-tier storage system** that gives the KV cache a fast warm RAM tier
and a durable NVMe tier, so conversation state survives restarts and can be
reloaded — fast — into any inference slot.

```
   VRAM  (hot)   — the compute set; only what infers this tick
   RAM   (warm)  — recently-hot sequences + the redo-log write buffer
   NVMe  (cold)  — a complete, self-contained substrate image
```

The design rests on three pillars:

1. **Two migration kernels** — `kv_pack` (gather/evict) and `kv_unpack`
   (scatter/load) — that move a sequence's free-list-scattered arena chunks
   to/from a *contiguous* buffer so the PCIe/NVMe transfer is one large DMA,
   not thousands of tiny ones. Both reuse the existing `arena_table.cuh`
   chunk-indexing helpers.

2. **An append-only, single-file NVMe redo log of interleaved streams.**
   Chunks are appended sequentially, each tagged with a `stream_id`. A
   stream is one conversation *turn* or one system-prompt section; section
   streams are **content-addressed**, so an unchanged template section is a
   durable prefix-cache hit shared across every conversation that uses it.
   The log also carries the model spec, the projection template, per-turn
   structure, token IDs, and provenance signatures — it is a **complete,
   self-contained substrate image** (§5.7), so a restart needs no companion
   file. The log is *skip-loadable*: a reader walks fixed-size record
   headers without reading payload, reconstructing the index in one
   header-only pass.

3. **An asymmetric tier flow.** The **write path is `VRAM → RAM → disk`** —
   the RAM copy serves double duty as both the warm cache and the
   group-commit write buffer. The **cold-load path is `disk → pinned
   host → VRAM`** today (a bridge implementation), with the target shape
   being `disk → VRAM` directly via GPUDirect Storage (`cuFileReadAsync`)
   once the production Linux workstation is in place. The bridge uses a
   pre-allocated pinned host scratch + `cuMemcpyHtoDAsync`; the API is
   GDS-shaped so the Linux backend is a backend-swap, not a caller
   rewrite. The warm tier is re-populated as a side effect of normal
   eviction, not by cold loads. See §4.

**Goals:** (1) durable KV persistence — conversations survive process
restarts; (2) very fast cold-load of a stored conversation straight into
VRAM; (3) unify test-data generation with real data — one storage format,
fewer modes for error. This is the enabling infrastructure for the paper's
§9.12 unbounded-context evaluation (ingest once, snapshot, load-many).

**Scope assumption:** storage is **NVMe** (high bandwidth, RAID 0). The log
is **compacted in place** by a full defragment-and-rewrite pass (§5.8) — a
real, implemented operation, not a future task.

---

## 2. Design principles

### 2.1 What we optimize for

- **Saturate the wire.** One large contiguous DMA per migration, never
  thousands of small ones. Pinned host memory. Dedicated copy stream so
  transfers overlap decode.
- **Never block decode.** Every transfer is enqueued async and synchronised
  via a `CudaEvent` only when the data is actually needed.
- **Correct by construction.** A quantized chunk's bytes are
  *format-identical* across tiers — only the address space differs.
  Migration is a pure byte copy: no dequant, no re-quant, no RoPE pass.
- **Reuse existing indexing.** The kernels `#include arena_table.cuh` and
  address chunks with the *same* helpers the paged-attention kernels use.
  No bespoke chunk-addressing logic.

### 2.2 Database learnings we take

The cold tier is, structurally, an NVMe-optimized database. We borrow
directly:

- **Log-structured storage** (LSM-tree, Bitcask, WAL). Append-only writes
  never seek — the fastest possible NVMe write pattern. It also matches the
  KV cache's own shape: a conversation's sealed chunks only ever grow.
- **Write-back buffering + group commit.** The RAM warm buffer *is* the
  write buffer. Many sealed chunks accumulate, then flush as one large
  sequential append. Writes never sit on the decode critical path.
- **Buffer-pool hierarchy.** RAM is the page cache for disk; VRAM is the
  page cache for RAM. Eviction is **LRU** over each tier. Capacity is
  ~10× per step down: VRAM ≪ RAM ≪ disk.
- **Explicit async I/O, never `mmap`.** mmap reintroduces OS-page-cache
  double-caching of data we already hold in our own RAM tier, its
  page-fault stalls are unschedulable, and page-fault-driven writes give no
  clean durability barrier. (See *"Are You Sure You Want to Use MMAP in
  Your DBMS?"*, CIDR 2022.) We use explicit reads and writes at controlled
  offsets, with `fsync` for durability.
- **Durable, explicit I/O.** The log is a normal buffered file; durability
  is an explicit `fsync` / `sync_data` at each group-commit boundary — the
  standard, robust durable-logging mechanism (as used by SQLite and WALs).
  Records are 4 KB-aligned so a future move to unbuffered/`O_DIRECT` I/O
  remains open, but it is a perf option, not a correctness requirement.
- **Self-describing records + checksums.** Every record carries a header
  and a checksum; crash recovery walks the log and stops at the first
  torn record (torn-write detection).
- **Indexed recovery + compaction.** Recovery follows the backward
  `HeaderIndex` digest chain from a superblock hint (§5.6) — a handful
  of reads regardless of log size — and falls back to a filtered
  forward walk on any inconsistency; compaction (§5.8) bounds the log
  to the live record set. There is no manifest snapshot: the chain
  carries record *headers*, and all live state is still rebuilt from
  the records themselves.
- **Content-addressed storage** (Git objects, CAS dedup stores). A
  system-prompt section is keyed by a hash of its content and its prefix,
  so identical sections share storage automatically and any change forks a
  new object — invalidation without an invalidation protocol.

### 2.3 Asymmetries the design accounts for

- **Bandwidth.** NVMe RAID ≈ 45 GB/s; host→VRAM PCIe 5.0 ≈ 55–64 GB/s.
  These are *separate buses*. The bridge cold-load path uses both — the
  NVMe leg reads into a pinned host buffer (~45 GB/s), the PCIe leg
  HtoD's that buffer to the VRAM staging scratch (~55 GB/s); the slower
  NVMe leg sets the throughput and the PCIe leg is fully hidden through
  double-buffering. **The GDS target eliminates the PCIe-bounce leg**:
  the NVMe controller DMAs straight through the GPU's PCIe BAR into
  VRAM, so cold load runs at NVMe rate (~45 GB/s) over a single bus
  hop, freeing the host PCIe bandwidth + CPU memcpy work for other
  uses. The **eviction write path** (VRAM → RAM → log) keeps the
  pinned host buffer — that buffer doubles as the warm tier (a real
  RAM cache, not a transient staging region) so eliminating it is not
  a useful simplification.
- **Capacity.** VRAM is evicted aggressively every tick; RAM is an LRU
  working set; disk grows append-only between compaction passes (§5.8).

---

## 3. The write path — VRAM → RAM → disk

New tokens arrive and fill the active (partial) chunk. When it reaches
`CHUNK_SIZE` (32) tokens it **seals** — becomes immutable, quantized,
format-identical bytes. The active writer chunk stays **GPU-float** while
it fills.

Both kinds of chunk migrate and persist. The kernels are format-agnostic
byte copies, so a partial float chunk moves through the same path as a
sealed quantized one — only its `byte_len` differs (see §5.5 for how the
partial tail is superseded).

The path off the GPU has three stages:

1. **Gather (`kv_pack`).** A sequence's chunks are scattered across GPU
   arenas — the `GidPool` free-list does not place them contiguously.
   `kv_pack` gathers them (every sealed chunk, plus the partial tail) into
   the fixed VRAM staging scratch buffer, so the transfer that follows is
   one contiguous block.
2. **DtoH to the warm tier.** One async device-to-host copy on the
   dedicated copy stream lands the contiguous blob in a **pinned host
   buffer**. That buffer is the warm-tier representation.
3. **Group-commit append.** The pinned RAM buffer serves **double duty** —
   it is both the warm cache copy *and* the redo-log write buffer. Chunks
   accumulate in RAM; when the buffer crosses a flush threshold they are
   appended to the NVMe log as **one large sequential write**. After the
   write is durable a *commit record* is appended, and the RAM copy becomes
   a *clean* page — droppable under LRU pressure with no write-back.

The **partial tail chunk** is persisted too — losing it on restart would
truncate the end of every conversation. The live, still-mutable tail is
written as a normal `Chunk` record (float, `token_count < 32`),
force-flushed on eviction so an evicted or idle sequence is always fully
durable. A partial that belongs to a **sealed** turn or section is
different: the seal-time quantize pass compresses it like a full chunk
(its dead token slots are zero — arena chunks are zeroed at creation and
recycle — and the selection kernel receives the valid range for its
count-normalized metrics), so it persists in its quantized format. For a hot, actively-decoding
sequence the tail is flushed on the group-commit timer, so crash loss is
bounded by the q2 time-bound. During fast decode a chunk usually seals
faster than the timer fires, so most chunks are written once, sealed, with
no intermediate partial snapshot. See §5.5.

The RAM warm tier is **LRU**: when RAM is pressured, clean (already-durable)
sequences are dropped first; dirty (not-yet-flushed) sequences must flush
before they can be dropped.

> **Snapshot variant.** To persist a sequence that is *still hot* in VRAM
> (e.g. the §9.12 *ingest-once* case), `kv_pack` gathers and the blob takes
> the same `DtoH → pinned host → log append` path as eviction — but the
> VRAM `ChunkGid`s are **retained**, not dropped, so the sequence stays
> hot. Same path, the GPU copy is simply kept.

```
                          new tokens
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  VRAM · hot tier                                           │
   │     active chunk ──fills──▶ ■ ■ ■  sealed chunks           │
   │                             (scattered across free-list   │
   │                              arenas — not contiguous)      │
   └──────────────────────────────────────────────────────────┘
                              │  kv_pack
                              │  gather scattered ─▶ contiguous
                              ▼
                  ┌───────────────────────────┐
                  │  VRAM staging scratch      │  fixed, pre-allocated
                  └───────────────────────────┘
                              │  1× async DtoH  (copy stream, pinned dst)
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  RAM · warm tier  (LRU)                                    │
   │     pinned host buffers ───── double duty ─────┐           │
   │       • warm cache copy   (→ future warm hit)  │           │
   │       • redo-log write buffer                  │           │
   └────────────────────────────────────────────────┼──────────┘
                              │  group commit       │
                              │  many chunks ─▶ one large sequential append
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  NVMe · cold tier — append-only redo log                   │
   │   ┌──────┬──────┬────────┬──────┬──────┬─ ─ ─┐             │
   │   │chunk │chunk │ commit │chunk │chunk │     │◀── tail     │
   │   └──────┴──────┴────────┴──────┴──────┴─ ─ ─┘             │
   └──────────────────────────────────────────────────────────┘
```

---

## 4. The cold-load path — disk → VRAM

A sequence not resident in VRAM is loaded on demand. Two cases:

- **Warm hit** — the sequence is still in the RAM tier. Load is the fast
  RAM→VRAM path: one HtoD from the warm pinned buffer into the VRAM staging
  scratch, then `kv_unpack`. No disk touch. This is the common case and the
  reason the warm tier exists.
- **Cold load** — the sequence is on disk only. Its chunk records are
  streamed **NVMe → pinned host buffer → VRAM staging scratch**, then
  `kv_unpack` scatters. The pinned host buffer is a *transient cold-load
  scratch*, **not the warm pool** — a cold load does not populate the
  warm tier (the warm tier still has exactly one writer, eviction). The
  warm tier re-populates naturally later when the now-hot sequence is
  evicted VRAM → RAM.

  > **Bridge implementation note.** The target shape for this path is
  > **NVMe → VRAM directly via GPUDirect Storage** (`cuFileRead` →
  > controller DMAs straight through the GPU's PCIe BAR, no host bounce
  > buffer). GDS is a Linux-only NVIDIA technology (it depends on the
  > `nvidia-fs` kernel module); Microsoft DirectStorage is the Windows
  > analogue but isn't reachable from CUDA. Until the production Linux
  > workstation lands — and until/if a CUDA-reachable Windows
  > GPU-storage API ships — we use the pinned-staging path described
  > here as the bridge. The interface is shaped so the Linux backend
  > can be swapped to GDS without touching the cold-load caller; the
  > rest of this section describes the bridge path.

**Resuming a conversation loads a DAG of streams (§5.2).** A conversation is
an ordered set of turn streams anchored to the prompt-section streams that
form their prefix; loading it means resolving every stream in that DAG —
each a warm hit, a cold load, or (for an unchanged content-addressed
section) a stream already resident and shared by another live conversation.
Each stream keeps its own local chunk grid; the substrate composes them at
the right absolute positions on injection (`inject_sealed_at_tail`), with
RoPE applied from recomputed positions — no re-RoPE, no byte rewrite.

A cold load of a single stream — bridge (pinned-staging) implementation:

1. **Manifest lookup.** The in-RAM index gives the `(kv_bytes_offset,
   kv_bytes_len)` ranges of every chunk's bulk payload — the K/V byte
   blob, not the ChunkPayload prefix (formats / palettes / scales). That
   prefix is small and read host-side through the usual buffered I/O so
   the loader knows how to lay the bytes out; the prefix offset/length
   are pre-computed at append time and live in the manifest's
   `ChunkLoc`. Contiguous `kv_bytes` ranges are coalesced into larger
   reads.
2. **NVMe read into pinned host.** The coalesced ranges are `pread`'d
   into a **pre-allocated pinned host buffer** (allocated once at
   substrate open via `cudaHostAlloc`). Pinned memory means the next
   leg is a single DMA, not a pageable-bounce-then-DMA, and the
   userspace `Vec<u8>` copy is gone.
3. **`cuMemcpyHtoDAsync` to VRAM staging.** One async device-to-host
   copy on the dedicated copy stream lands the contiguous blob in the
   VRAM staging scratch. With double-buffered pinned + staging
   regions, the NVMe leg of batch *n+1* can overlap the HtoD of batch
   *n* and the `kv_unpack` of batch *n−1* — the slower NVMe leg sets
   the throughput (~45 GB/s, §2.3).
4. **Scatter (`kv_unpack`).** Freshly-allocated arena GIDs receive the
   chunks; the kernel scatters the contiguous VRAM staging blob into
   them.
5. **Provenance-driven streaming.** A cold load does not fetch the whole
   conversation — it fetches the chunks **provenance selects** first and
   streams the remainder. The working set off disk is far smaller than the
   full corpus; this is the §9.12 load-many path.

**Setup once at substrate open.** The pinned host buffer is allocated
(`cudaHostAlloc` via `PinnedStager`) and re-used across cold loads.
Compaction replaces the log file underneath; the file descriptor is
re-opened after the rename swap (§5.8).

**Future GDS swap.** Replacing steps 2–3 with a single `cuFileReadAsync`
call against a `cuFileBufRegister`'d region of the VRAM staging scratch
is the upgrade path — no more pinned host buffer, no more HtoD, the NVMe
controller DMAs through the GPU's BAR directly into VRAM. The bridge
interface is shaped so this is a backend-swap, not a caller-facing
change.

The sequence's **partial tail chunk** (if any) loads through the same path:
`kv_unpack` scatters its float bytes into a fresh GPU-float active writer
chunk. The conversation resumes simply by continuing to decode into it —
no prefill, no recompute. The load path stays pure data movement.

```
   ┌──────────────────────────────────────────────────────────┐
   │  NVMe · cold tier — redo log                               │
   │     manifest ──▶ (kv_bytes_offset, len) ranges per chunk  │
   │   ┌──────┬──────┬────────┬──────┬──────┐                   │
   │   │chunk │ ···  │ chunk  │ ···  │chunk │                   │
   │   └──┬───┴──────┴───┬────┴──────┴──┬───┘                   │
   └──────┼──────────────┼──────────────┼──────────────────────┘
          │              │              │
          └──────────────┴──────────────┘
                         │  pread() into pinned host buffer (bridge)
                         ▼
              ┌───────────────────────────┐
              │  pinned host scratch       │  double-buffered
              │  cudaHostAlloc'd           │
              └───────────────────────────┘
                         │  cuMemcpyHtoDAsync (copy stream)
                         ▼
              ┌───────────────────────────┐
              │  VRAM staging scratch      │  double-buffered, pipelined
              │                            │  with prior kv_unpack
              └───────────────────────────┘
                         │  kv_unpack
                         │  scatter contiguous ─▶ arena chunks
                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │  VRAM · hot tier                                           │
   │     ■ ■ ■  freshly-allocated arena chunks — ready to infer │
   └──────────────────────────────────────────────────────────┘

   ── warm hit (sequence already resident in RAM) ────────────────
      warm pinned buffer ──HtoD──▶ VRAM staging ──kv_unpack──▶ VRAM
      no disk touch — the fast PCIe-bound path

   ── future (GDS, Linux + nvidia-fs) ───────────────────────────
      NVMe ──cuFileReadAsync──▶ VRAM staging  (no host hop)
      then kv_unpack — same scatter as above
```

---

## 5. On-disk format: streams, records, and skip-loading

### 5.1 File layout

The cold tier is a **single, pre-grown file** — an append-only redo log.
(Segmented files were considered and rejected: segments only bought
bounded recovery scan and cold-tier-on-slow-media — both moot under
header-only recovery and the NVMe-only assumption; compaction is a
whole-file rewrite, §5.8, so it needs no per-segment unit.) The file is
grown ahead in large extents (e.g. 1–4 GB via
`SetFileValidData`/`set_len`) so appends write into already-allocated
space and never trigger per-append metadata churn.

```
┌──────────────┬─────────┬─────────┬──────────┬─────────┬─ ─ ─┐
│ file header  │ record  │ record  │  record  │ record  │     │
│ (log magic)  │         │         │          │         │     │
└──────────────┴─────────┴─────────┴──────────┴─────────┴─ ─ ─┘
```

**File header — minimal.** Just a log-format **magic** and a **log-format
version** — enough for a reader to recognise the file and know the record
header layout for skip-loading. It deliberately carries **no** model or
template fingerprint: that information is mutable over a log's life, so it
lives in updatable `ModelSpec` / `Template` records (§5.3) resolved by
last-writer-wins, not in a write-once header. The model-compatibility check
(§12 invariant 6) runs against the *latest* `ModelSpec` record.

**The log is a complete substrate image.** Everything needed to restart the
substrate from this one file is in it — model spec, projection template,
stream DAG, per-turn structure, token IDs, provenance signatures, and KV
chunks. See §5.7.

### 5.2 Streams

The substrate is not one monolithic token sequence — it is a set of
independently-addressable ones: conversation turn-histories, and each
section of the system prompt. The redo log represents every one of them as
a **stream**: an ordered chunk sequence with its own **local chunk grid**
(`chunk_index` starts at 0 per stream). Every `Chunk` record carries a
`stream_id`; the log **interleaves** records from all streams as
group-commit flushes them, and the skip-load walk demuxes by `stream_id`.

A stream maps directly onto a substrate `SealedSequence`, which is
**position-agnostic by construction** (`types.rs`): it carries no RoPE
base and no offset; K is stored un-rotated and RoPE is applied in the
attention kernel from positions recomputed from the cumulative usage of
preceding blocks in the destination slot. Consequence: a stream's stored
chunks can be composed at *any* absolute position — which is exactly what
makes a system-prompt section reusable across conversations.

There are **two kinds of stream**:

- **Turn stream** — one per conversation *turn*. In the substrate, every
  turn carries its own `SealedSequence` (`TurnEntryData.sealed`); a
  conversation is therefore **not** one monolithic stream but an *ordered
  set of turn streams*. A turn stream is **immutable once its turn seals**;
  only the single in-progress turn has the mutable partial tail.
  Identity-addressed by `(timeline_id, turn_index)`.
- **PromptSection stream** — one per system-prompt section. **Write-once and
  immutable**, **content-addressed** (see the hash chain below). Its last
  chunk is naturally **partial** (a section is rarely a multiple of 32
  tokens); the substrate already seals these partial section tails
  deliberately — `types.rs` notes that dropping them would "silently lose
  up to `(sections-1)*(CHUNK_SIZE-1)` tokens when sections are projected
  back-to-back." Sealed section tails quantize like full chunks and
  persist in their quantized block format (block bytes cover all 32 token
  slots, dead slots zeroed); only the live mutable tail uses the float
  no-padding `Chunk` form (§3, §5.5).

A **conversation** is an emergent grouping, not a stream: it is *all turn
streams sharing a `timeline_id`, ordered by `turn_index`*, anchored to the
prompt-section streams that form their prefix. Streams form a **DAG**:
section streams are shared roots; turn streams chain after them. N
conversations on the same prompt version store its section streams **once**.

```
   PromptSection streams (content-addressed, shared roots)
     ┌────────┐  ┌────────┐  ┌────────┐
     │ sec A  │  │ sec B  │  │ sec C  │
     └───┬────┘  └───┬────┘  └───┬────┘
         └─────┬─────┴───────────┘
               ▼
   Turn streams (identity-addressed, ordered per timeline)
     ┌────────┐   ┌────────┐   ┌────────┐
     │ turn 0 │──▶│ turn 1 │──▶│ turn 2 │◀── only this one
     └────────┘   └────────┘   └────────┘    has a mutable tail
        └──────── timeline T ────────┘
```

#### Content addressing — a hash chain

A PromptSection stream's ID is **derived from content**, so any change to
the template forks a new stream and leaves existing ones untouched. A
section's KV depends on its own tokens *and* on the entire prefix before it
(its hidden states attended over that prefix; same tokens ⇒ same length ⇒
same RoPE). Both dependencies are captured by a **hash chain** over the
sections:

```
chain[0]  = H(section_0_tokens)
chain[i]  = H(chain[i-1] ++ section_i_tokens)
stream_id(section_i) = ( chain[i-1] , H(section_i_tokens) )
```

- **Correct cascade.** Change section *i* → its hash changes → every
  `chain` value after it changes → every following section gets a new
  `stream_id`. The change invalidates exactly itself and everything
  downstream; earlier sections are reused.
- **Automatic reuse / persistent prefix cache.** An unchanged section with
  an unchanged prefix yields a byte-identical `stream_id`. This is a
  durable, NVMe-backed prefix cache — the same idea as vLLM automatic
  prefix caching / SGLang RadixAttention, but persisted.
- **Structural fork sharing.** Two developer forks of the same codebase
  independently compute the *same* content-addressed `stream_id`s and share
  the section streams with no dedup pass (this is what resolves §15 q3).

**Cache hit / miss lifecycle.** Before prefilling a section, compute its
content-addressed `stream_id` and check the manifest. **Hit** → cold-load
its chunks, skip prefill entirely. **Miss** → prefill, append a
`StreamDecl` + the section's `Chunk` records. This is the payoff: fast new
conversations and fast codebase re-ingestion (§9.12) because most sections
are hits.

### 5.3 Records

Every record is a **fixed-size header** followed by a variable payload.
The header is the key to skip-loading:

| Field          | Purpose                                            |
|----------------|----------------------------------------------------|
| `magic`        | record-boundary sentinel; resync after corruption  |
| `record_type`  | one of the eight types below                       |
| `length`       | payload byte length — **lets a reader skip it**    |
| `stream_id`    | which stream (turn or prompt section); 0 if N/A    |
| `chunk_index`  | position within the stream's local chunk grid      |
| `token_count`  | `Chunk` only — `32` = sealed, `<32` = tail          |
| `format`       | `Chunk` only — quantized (sealed) or float (tail)   |
| `checksum`     | covers header + payload; torn-write detection      |

There are **nine record types**. Four are *singletons / metadata*, the
rest are *per-stream*:

- **`ModelSpec`** (singleton, last-writer-wins) — the model and its
  properties: architecture, chat format / dialect, HF model + tokenizer
  coordinates, `max_seq_len`, sampling defaults, and the KV-critical dims
  (`n_layers`, `n_kv_head`, `head_dim`, RoPE params, quant format set,
  `CHUNK_SIZE`, provenance layer indices, engine code version). **Weights
  are not stored** — only the spec needed to re-load them. The latest
  `ModelSpec` is the compatibility authority (§12 invariant 6).
- **`Template`** (singleton, last-writer-wins) — the serialised projection
  `Schema`: layers, groups, `SelectionRule`s, `SystemPromptSchema`
  (sections + collections), `DepthWeights`, `Budget`s, `ScoreFormula`s.
  Loaded today from `projection.yaml`.
- **`Tokenizer`** (singleton, last-writer-wins) — the model's raw
  `tokenizer.json` bytes, embedded so the log can detokenize offline with no
  companion file. Large (~11 MB for Qwen3) but written at most once per
  distinct model via compare-and-insert; identical bytes are a no-op.
- **`StreamDecl`** — declares a stream and carries its structural metadata.
  For a **`PromptSection`**: the `(prefix_hash, section_hash)` content
  address + a debug name. For a **`Turn`**: `timeline_id`, `turn_index`,
  `TurnId{day,seq}`, `role`, the ordered anchored prefix `stream_id`s
  (section streams + prior turn streams), `block_range`, the persisted
  `PerDepthScores` (syn/sem/prag), and the projection `view` edges. A
  stream is declared once; an already-present content-addressed section is
  never re-declared (cache hit).
- **`Chunk`** — one sealed or partial KV chunk. Payload = a small
  **metadata prefix** (`offset` window skip-count; the host-side
  quantization metadata `k_pal`, `v_pal`, `k_scale`, `v_scale` —
  **mandatory: the arena blob is not self-describing and cannot be
  dequantised without these**) followed by the format-identical arena blob
  `kv_pack` gathers. `token_count`/`format` in the header say sealed vs
  partial. See §5.5 for supersession.
- **`Tokens`** — a stream's token IDs (`Vec<u32>`). Stored separately from
  `Chunk` so the skip-load walk can jump them; read on demand. Turn text is
  *not* stored — it is `detokenize(token_ids)`.
- **`Signatures`** — a stream's provenance signature vectors (the Binary
  Directional Provenance sign-bit data). This is the home the retired
  `provenance/` test-data stores (§7.2) fold into.
- **`Commit`** — "stream `stream_id` is durable through `chunk_index`." A
  stream is recoverable only up to its last commit; the group-commit
  boundary.
- **`HeaderIndex`** — a batch of fixed-width header digests for the
  records appended since the previous index, plus a link to that
  previous index record. The backward chain recovery follows instead of
  probing every record header (§5.6). Derived data: compaction drops
  every copy and the writer regenerates the chain in the new file.

All records are **4 KB-aligned** (payloads padded; the header's `length`
is the true unpadded size) for clean sector-aligned record boundaries.

### 5.4 Skip-loading — walking the log by headers only

The log is **skip-loadable**: a reader reconstructs the full index without
reading a single chunk payload.

```
   read header @ offset 0
        │
        ▼
   header.length tells the payload size
        │
        ▼
   next header @ offset + header_size + padded(length)
        │
        └──▶ repeat — only headers are read, payloads are jumped
```

The cost is **O(records) tiny reads**, not O(bytes). Each header read
yields `record_type`, `stream_id`, `chunk_index`, the current file offset,
`length`, `token_count`, and `format` — enough to populate the in-RAM
manifest without touching the (large) payloads. The small structural
payloads — `ModelSpec`, `Template`, `StreamDecl` — *are* read during the
walk (the model, template, and stream DAG are needed up front). The bulk
payloads — `Chunk`, `Tokens`, `Signatures` — are **skipped**, their offsets
recorded, and read only on demand: `Chunk` on a cold load (and only the
chunks provenance selects), `Tokens`/`Signatures` when text or a provenance
scan actually needs them.

### 5.5 Supersession — last-writer-wins

A chunk's state changes over its lifetime: a partial tail grows token by
token, then seals. Each group-commit flush appends an **immutable snapshot**
of the chunk *as it was at flush time*. So the log accumulates several
records for the same `(stream_id, chunk_index)` — e.g. a 20-token partial,
then a 30-token partial, then the sealed 32-token quantized chunk.

The loader resolves this with **last-writer-wins**, and it needs no
supersession detection: append order *is* recency order (a chunk's
`token_count` only ever grows; once sealed it never goes partial again).
The skip-load walk simply overwrites as it goes:

```
for each valid record in append order:
    manifest[(stream_id, chunk_index)] = (offset, len, token_count, format)
```

The last **checksum-valid** record for a key wins. A sealed chunk
supersedes an earlier partial because it was appended later; a 30-token
partial supersedes a 20-token one for the same reason. A torn final write
fails its checksum, the walk stops there, and the prior valid snapshot
stands. This is log-structured / MVCC semantics — the log is a sequence of
immutable snapshots, the live state is the newest valid one per key. Stale
superseded records are dead weight reclaimed by compaction (§5.8).

(A content-addressed `PromptSection` stream is immutable, and a turn stream
is immutable once its turn seals — so chunk supersession only ever fires
for the single **open turn stream** of a conversation. The `ModelSpec` and
`Template` singletons use the *same* last-writer-wins rule, keyed by record
type rather than `(stream_id, chunk_index)`: the walker simply keeps the
latest valid one. `Tokens` and `Signatures` are superseded per stream the
same way a stream's tail grows.)

### 5.6 Recovery — the header-index chain

Even a header-only forward walk is latency-bound: each header read
reveals where the *next* record starts, so a multi-GB log degenerates
into hundreds of thousands of serial queue-depth-1 sector reads. The
`HeaderIndex` chain removes the serial dependency. Every
`INDEX_FLUSH_ENTRIES` appends, the writer flushes one `HeaderIndex`
record whose payload holds a fixed-width **digest** — `(type, format,
stream_id, chunk_index, token_count, offset, record_size,
payload_len)` — of each record appended since the previous index, plus
the `(offset, size)` of that previous index record. The superblock
carries an advisory hint to the newest committed index (updated after
the index record's bytes are durably committed).

Recovery is then:

1. **Follow the chain backwards** from the superblock hint — a handful
   of CRC-verified reads for the whole log — and reverse the collected
   digests into append order.
2. **Replay the digests** through the same per-record dispatch a walk
   uses (manifest singletons, `Substrate::apply_walker_entry`, the
   dead-byte accounting), last-writer-wins in append order.
3. **Batch-fetch metadata payloads.** The digest types whose payload
   feeds in-RAM state (`StreamDecl`, `Label`, `TreeMetadata`,
   `ProjectionEvents`, …) expose their offsets up front, so their
   records are read in coalesced spans — no serial probing. The bulk
   types (`Chunk`, `Tokens`, `Signatures`) stay by-reference and their
   payloads are never read.
4. **Forward-walk only the un-indexed tail** — everything after the
   hinted index record, at most one flush interval plus the crash
   window — with the filtered walk (§5.4). Torn-tail detection and
   truncation live here, unchanged. The tail's digests seed the
   writer's accumulator so the next flush covers them: the chain heals
   forward across restarts.

Every failure mode of the fast path — a zero or garbage hint (old
superblocks carry the retired checkpoint offset in these bytes), a hint
into the pre-grown zero tail, a torn / wrong-typed / wrong-version
index record, a non-monotonic chain — degrades to the **full filtered
forward walk**: correct on any log, just slower. Index records are
derived data; nothing is ever reconstructed *from* them that the
records themselves don't also carry.

A stream is recoverable through its last `Commit`; the open turn stream's
tail is whatever partial `Chunk` snapshot most recently survived
supersession.

### 5.7 The log is a complete substrate image

The intent is that the substrate can be **fully restarted from this one
file** — no companion files. The audit (against `substrate.rs` /
`conversation.rs` / `tree/` / `projection/`) maps every piece of substrate
state to a record:

| Substrate state | Record |
|---|---|
| Model + properties (to re-load weights from HF) | `ModelSpec` |
| Projection schema / template | `Template` |
| Model `tokenizer.json` (offline detokenize) | `Tokenizer` |
| Stream DAG; per-turn `role`, `timeline_id`, `turn_index`, `TurnId`, `block_range`, `view`, `PerDepthScores` | `StreamDecl` |
| KV bytes + quantization metadata (`offset`, `k_pal`/`v_pal`/`k_scale`/`v_scale`) | `Chunk` |
| Token IDs (turn text = `detokenize`) | `Tokens` |
| Provenance signatures | `Signatures` |
| Durability boundary | `Commit` |

Runtime-only state (the `SubstrateCache` VRAM accounting, the warm-pool LRU
order) is **not** persisted — it is rebuilt on load. `PerDepthScores` *is*
persisted (it is accumulated, not cheaply recomputable). Because the log is
now self-sufficient, the legacy YAML transcript log `store.rs` is **retired**
(§7.2) — `detokenize(Tokens)` reproduces turn text.

### 5.8 Compaction

An append-only log only grows: every superseded partial-tail snapshot,
every stale `ModelSpec`/`Template`, every chunk of a deleted conversation
is dead weight that the skip-load walk still steps over. **Compaction**
reclaims it. It is a first-class, implemented operation — not a future
task.

Compaction is a **whole-file rewrite**:

1. **Quiesce.** Flush and commit all in-flight writes; the active log is
   now consistent on disk.
2. **Collect the live set** from the in-RAM state — the substrate's
   stream index and the manifest's singletons already resolve
   last-writer-wins, and tombstoned timelines' records are excluded.
3. **Rewrite.** Stream every *live* record — the latest singletons, then
   per stream its `StreamDecl` / `Chunk`s / `Tokens` / `Signatures` /
   `Commit` — into a new file `.substrate/substrate.log.compact`, in
   dependency order, interleaving a fresh `HeaderIndex` chain (§5.6)
   whose head lands in the new superblock. Dead records — including
   every old index record — are simply not copied.
4. **Swap.** `fsync` the new file and atomically rename it over
   `.substrate/substrate.log`. Every record now sits at a new offset, so
   the substrate's walker-built state is cleared and re-walked from the
   compacted file (§5.6), and each residence's cold-tier references are
   re-pointed at the rebuilt stream index. KV bytes in VRAM/RAM are
   untouched — only the on-disk references move.

**Triggering.** The dead weight is tracked incrementally, O(1) per
append: every record type that resolves last-writer-wins by header key
(`Chunk` by `(stream, chunk_index)`; `Tokens` / `Signatures` /
`StreamDecl` / `Commit` / `ProjectionEvents` per stream; the singletons
per type) charges its superseded predecessor's bytes to a dead-byte
counter, and tombstoned timelines' stream bytes are summed from the
in-RAM index. The persistence thread polls the resulting dead-byte
ratio every pass — pure in-RAM arithmetic — and compacts automatically
when it crosses the threshold on a log past the minimum size (small
logs never auto-compact; a rewrite there reclaims nothing worth the
pause). The daemon's startup flag forces a compaction regardless of the
ratio. Inherited logs (§13.5) are read-only and are **never** compacted
by a child; a base log is compacted only by a process that opens it as
its own active log.

---

## 6. Kernels and what is reused

### 6.1 The two kernels

| Kernel       | Direction        | Role                                      |
|--------------|------------------|-------------------------------------------|
| `kv_pack`    | arenas → staging | **gather** scattered chunks → contiguous  |
| `kv_unpack`  | staging → arenas | **scatter** contiguous → fresh arena slots|

They are **the same kernel body with src/dst swapped** — implement once,
parameterise the direction. One FFI entry point, `kv_migrate_chunks`.
Detailed thread-mapping and kernel bodies are in §9.

`kv_pack` serves both eviction and snapshot — its blob always exits via the
same DtoH. `kv_unpack` serves both warm-hit and cold loads — its source is
always the VRAM staging scratch after an HtoD. The kernels are indifferent
to where the host data came from (warm pool or an NVMe read); they only
ever touch the VRAM staging scratch and the VRAM arenas.

### 6.2 What is reused — not rebuilt

- **`arena_table.cuh`** — its `per_head_lookup` / `per_head_k_ptr` helpers
  remain the authority for chunk addresses, consulted **host-side** by the
  plan-builder. The migration kernel itself is index-free (§8).
- **The copy-stream + event template** from
  `select_and_summarize_kv_winners_paged_staged` (`quantized/cuda.rs`) —
  the proven pattern of an async DMA on a dedicated stream returning a
  `CudaEvent`. Reused verbatim for both DtoH and HtoD.
- **`pinned_staging.rs`** — the pinned (page-locked) host-memory primitive.
  The warm-tier pool is built from it.
- **`HeadGids::map_unique`** — the GID-remap primitive that preserves the
  K/V/palette sharing structure of a chunk across a tier move.
- **No kernel writes the file.** A CUDA kernel cannot do I/O — it only
  touches memory. Disk transfer is an explicit host-side file read or
  write; the host↔VRAM hop is a `cudaMemcpy`. The kernels' only job is to
  make the data *contiguous*.

---

## 7. Background — the existing implementation this replaces

This design is **not greenfield**. An ad-hoc warm-tier path already exists;
it is **redundant and incorrect**, and this document specifies its
wholesale replacement. Investigation (2026-05-18, re-verified against the
code) found:

### 7.1 The existing warm (CPU) path — redundant, replaced

A VRAM→CPU migration path **does exist today**, for system-prompt sections:

- `candle-conversation`'s substrate threads a
  `migrate_to_cpu: impl FnOnce(&[SealedSequence]) -> Result<Vec<SealedSequence>>`
  closure through `substrate.rs` (`record_turn`, `set_section_data`) and
  `projection/resolver.rs`. It converts GPU-resident `SealedSequence`s into
  CPU-resident ones stored as `sealed_cpu`; the substrate's own comments
  call this *"CPU (warm tier)"* / *"CPU warm-tier storage."*
- System-prompt sections *"live in CPU arenas pinned by the substrate"*
  (`conversation.rs`); `apply_projection` re-injects the selected ones per
  turn. `ArenaLocation::Cpu` and `ArenaKey::cpu_float()` are real,
  production-used.

This existing warm path is **the slow, buggy implementation the rebuild
exists to replace**. It is unbatched and per-head granular, uses pageable
host memory, and is synchronous — the four flaws below. This design
**discards it**: the new `kv_pack` / `kv_unpack` + pinned copy stream
become the `migrate_to_cpu` implementation, and the substrate seam is
otherwise unchanged.

`ChunkedKvBacking::read_raw_sealed_chunk` (`io.rs`) — a per-head eviction
reader — has **zero callers**: dead code. It is deleted, superseded by
`kv_pack`.

**Why the existing path is slow and buggy** — four compounding flaws
(`QStorage::data_range` → `cuMemcpyDtoH`, the shape `read_raw_sealed_chunk`
encodes):

1. **Pageable host memory** — `memcpy_dtov` returns an ordinary `Vec<u8>`;
   DtoH on pageable RAM runs at ~half PCIe bandwidth and cannot overlap
   compute.
2. **Synchronous, on the main stream** — serialised against decode.
3. **Per-head granularity** — a separate tiny DtoH per head's K and per
   head's V, `2 × n_kv_head` copies per chunk. Latency-bound.
4. **No batching, no pipelining.**

### 7.2 No correct cold (disk) KV tier exists — three things retired

KV state is **lost on process exit**. There is no `WarmPool`,
`EvictionManager`, `warm_pool.rs`, or `eviction.rs`. The redo log replaces
**three** pre-existing partial / redundant persistence mechanisms:

- **`provenance/raw_store.rs`** — an mmap-backed full-precision K/Q dump;
  and **`provenance/store.rs`** — the binarised-signature store. Test-only
  formats. Goal 3 (*unify test-data generation with real data*) folds them
  into the redo log's `Signatures` records (§5.3); they are then redundant
  and retired.
- **`store.rs`** — the YAML append-log of turn text / token IDs / `view` /
  signatures. Once the redo log carries `Tokens`, `Signatures`, and
  per-turn `StreamDecl` metadata it is a **complete substrate image**
  (§5.7) and reproduces turn text via `detokenize` — so `store.rs` is
  **redundant and retired** too. (A human-readable export can still be
  generated *from* the redo log on demand if one is wanted; it is no longer
  a source of truth.)

### 7.3 Scaffolding already in place

The substrate already carries the shape this design needs:
`SealedSequence.location: ArenaLocation`; `SealedChunk.byte_size`
documented *"preserved unchanged through CPU↔GPU migration"*;
`SealedSequence` is *"position-agnostic by construction"* (§5.2);
`inject_sealed_at_tail` composes sequences at arbitrary offsets. The
rebuild fills in the *mechanism* behind scaffolding that already exists.

---

## 8. Address resolution — host-side, not in-kernel

The migration kernel is **index-free**: it copies between device pointers
that the host has already resolved. Chunk-address resolution stays on the
host, in the plan-builder, which walks `SealedSequence.chunks[*].gids`,
decodes each `ChunkGid` (`arena_idx = raw / arena_gid_stride()`,
`chunk_idx = raw % arena_gid_stride()`), and resolves each sub-chunk's
device address from the arena tables.

`arena_table.cuh` and its `PerHeadTableEntry` / `per_head_lookup` helpers
remain the authority for those addresses, but they are consulted
**host-side** when the plan is built — the kernel itself never indexes an
arena table. Keeping resolution on the host makes the kernel a pure,
format-agnostic byte copier and avoids coupling it to the arena layout.

---

## 9. The migration kernel in detail

### 9.1 The kernel — `kv_migrate_copy`

A single kernel — `candle-kernels/src/simple/kv_migrate.cu`, in the
`simple` archive group — serves both `kv_pack` and `kv_unpack`. It takes
three parallel device arrays — `src_ptrs`, `dst_ptrs`, `byte_lens` — and
launches **one thread block per plan record**. Each block copies
`byte_lens[r]` bytes from `src_ptrs[r]` to `dst_ptrs[r]`: a 16-byte
vectorised copy when source, destination, and length are all aligned, a
correct byte copy otherwise.

Direction is just which side is the scattered set, so one kernel body
covers both operations:

- **`kv_pack`** (evict / gather) — `src_ptrs` are resolved arena-chunk
  addresses, `dst_ptrs` are offsets into a contiguous staging buffer.
- **`kv_unpack`** (load / scatter) — `src_ptrs` is the staging buffer,
  `dst_ptrs` are freshly-allocated arena chunks.

### 9.2 The plan — `MigrationPlan`

The host builds a `MigrationPlan`: a flat `Vec<MigrationRecord>`, each
record `(src_ptr, dst_ptr, byte_len)` — one per unique physical sub-chunk
(§15 q1). `kv_migrate` (candle-nn `kv_cache/chunked/migrate.rs`) uploads
the three arrays and launches the kernel once. The plan-builder that
derives a plan from a `SealedSequence`'s GIDs is built in P4, where
`SealedSequence` migration is exercised end-to-end.

### 9.3 One launch per phase, not per sequence

The plan is a flat record list; the kernel is one block per record and is
indifferent to which sequence a record belongs to. The scheduler builds a
single plan spanning every chunk of every sequence in the phase → **one
launch covers the whole eviction (or load) batch.** See §11.

---

## 10. Warm tier (RAM) representation

> Lives in `candle-conversation/src/persistence/warm_pool.rs` (§13.3).

- **Warm pool** = a pool of **pinned (page-locked) host buffers**, built on
  `pinned_staging.rs`. Pinned memory is mandatory: ~2× DtoH/HtoD bandwidth
  and the only way to overlap the copy with compute.
- An evicted sequence becomes a **new `SealedSequence`** with
  `location = ArenaLocation::Cpu`, its chunks re-pointed via
  `HeadGids::map_unique` to CPU-arena GIDs that index the warm pool.
  `SealedChunk.byte_size` carries over unchanged (format-identical).
- The original GPU `ChunkGid`s drop → their physical chunks return to the
  `GidPool` free-list (RAII). VRAM is reclaimed by construction.
- **LRU.** The warm pool tracks per-sequence recency. Under RAM pressure,
  *clean* (durable-on-disk) sequences are evicted first — a pure drop, no
  write-back. *Dirty* sequences (sealed but not yet flushed) must complete
  their log append before they can be dropped.

`map_unique` is the migration primitive — it maps each unique source GID to
one freshly-allocated destination GID while preserving the K/V/palette
sharing structure of the `HeadGids`.

---

## 11. Copy stream, overlap, and scheduler integration

> Note: "stream" in this section means a **CUDA stream**, unrelated to the
> redo-log *streams* of §5.2.

### 11.1 The dedicated copy stream

A **dedicated copy stream**, following the proven template of
`select_and_summarize_kv_winners_paged_staged`: record an event on the
compute stream, make the copy stream GPU-wait on it, enqueue the DMA,
record a completion event. The DMA is **enqueued, not waited on** — callers
hold a `CudaEvent` and synchronise only when the data is needed. Eviction
of a cold sequence thus overlaps fully with ongoing decode of hot
sequences.

### 11.2 The scheduler tick — migration is one-shot per phase

The tick is: **(1) queue work → (2) evict to free VRAM → (3) load
RAM/disk→VRAM → (4) run inference.** Steps 2 and 3 each migrate a *set* of
sequences as a **single batched kernel launch**:

- **Step 2 (evict):** one plan over every evicted chunk → one `kv_pack`
  launch → one DtoH.
- **Step 3 (load):** one plan over everything loaded → either one HtoD
  from the warm pinned buffer into the staging scratch (warm hit), or
  one NVMe `pread` into the cold-load pinned host scratch followed by
  one HtoD into the staging scratch (cold load — bridge; on Linux/GDS
  this collapses to a single `cuFileReadAsync` straight into the
  staging scratch) → one `kv_unpack` launch.

Per tick: **2 kernel launches + 2 DMAs**, independent of sequence count.

### 11.3 Within-tick ordering — the critical path

Load cannot allocate the VRAM that evict frees until evict has released it:

```
kv_pack (reads source arenas → staging)
   │  must COMPLETE before…
   ▼
drop evicted GIDs → GidPool reclaims source VRAM
   │
   ▼
load allocates fresh GIDs in the reclaimed VRAM
   │
   ▼
HtoD (→ staging)  →  kv_unpack (staging → new arenas)
   │
   ▼
inference
```

Critical path: `kv_pack → free → allocate → load DMA → kv_unpack →
inference`. The eviction **DtoH (the host save) and the subsequent log
append run in the shadow** of everything after them — they only hold the
staging buffer until complete; they do not block load or inference.

### 11.4 The staging buffer is fixed, pre-allocated scratch

Eviction's job is to *free* VRAM, so the staging buffer cannot scale with
the migration. It is a **small, fixed, pre-allocated VRAM scratch region**
(a permanent cost, sized for the expected per-tick volume).

- Batch **fits** the scratch → genuinely one-shot.
- Batch **exceeds** it → the phase **waves**: repeated `kv_pack`/DtoH (or
  load-DMA/`kv_unpack`) into the same reused scratch. Each wave is a single
  launch.

---

## 12. Correctness invariants

1. **Format-identical migration.** A quantized chunk's bytes are
   independent of tier. Migration copies bytes verbatim — never
   dequantises, re-quantises, or re-RoPEs. (K is stored un-rotated; RoPE is
   applied in the attention kernel from recomputed positions.)
2. **Sealed chunks are byte-stable before migration.** A sealed chunk is
   migrated/persisted only after the background quantiser has finished it —
   the evict phase calls `bg_quantizer.join()` first (see §15 q6). The
   partial tail chunk *is* migrated and persisted (as a float `Chunk`
   record), so a restart never truncates a conversation; its on-disk
   snapshots are resolved by last-writer-wins (§5.5).
3. **RAII reclaim.** VRAM is freed only when the last GPU `ChunkGid` drops.
   A sequence shared by multiple slots (forks) is not evicted while any hot
   holder remains.
4. **No partial states.** A migration either completes (new
   `SealedSequence`, old GIDs dropped) or aborts leaving the original
   untouched. The substrate swaps the `SealedSequence` pointer atomically.
5. **Durability is explicit.** A sequence is durable only through its last
   `Commit` record. The RAM copy stays *dirty* until that record is on
   stable storage; only then may LRU drop it without write-back.
6. **Model compatibility.** The KV-critical fields of the latest
   `ModelSpec` record (architecture, dims, RoPE params, quant formats,
   `CHUNK_SIZE`, provenance layer indices) must match the running engine,
   or the log is **refused** — never silently mis-interpreted. Benign
   fields (sampling defaults, HF coordinates) are simply adopted. The check
   is against the latest `ModelSpec` record, not a write-once file header.
7. **Self-sufficiency.** The log is a complete substrate image (§5.7): a
   restart needs no companion file. `ModelSpec` + `Template` + `StreamDecl`
   + `Chunk` + `Tokens` + `Signatures` reconstruct every piece of
   persisted substrate state; only runtime accounting is rebuilt fresh.

---

## 13. Module layout, API, and subsystem integration

### 13.1 Where the persistence layer lives

The persistence layer — redo log, warm tier, streams, manifest, recovery,
multi-log inheritance, and the tiering orchestration — lives **entirely in
`candle-conversation`**, as a new generalized module
`candle-conversation/src/persistence/`. It is **not** conversation-specific:
it persists any substrate (Zen Code conversations, Battle Cities agents).

`candle-nn` keeps only the **kernel primitive** — `kv_pack` / `kv_unpack`
and the host-side plan-builder — because those touch `pub(crate)` `chunked`
internals (`HeadGids`, `GidPool`, `SealedChunk`). The persistence layer
calls that primitive; everything else (disk I/O, warm pool, copy stream,
content-addressing, manifest, recovery, inheritance) is in
`candle-conversation`.

**Persistence is not optional.** There is no in-memory-only substrate — a
`Substrate` cannot be constructed without a backing log, and every turn
append and section ingest goes through the persistence layer. There is no
feature flag and no `Option<Persistence>` seam; the layer is a hard
dependency of the substrate. (Backward compatibility is explicitly *not* a
goal — see §13.7.)

### 13.2 On-disk location

On initialisation the layer ensures a **`.substrate/`** directory exists
(created if absent, relative to the process working directory). The active
redo log is **`.substrate/substrate.log`**. When an explicit list of logs
is supplied (§13.5) the active file is always the last in that list; the
default no-argument open uses `.substrate/substrate.log` as the sole log.

### 13.3 Module decomposition — every block testable in isolation

`persistence/` is split so each building block is **independently
unit-testable**, most with no GPU. One concern per file:

```
candle-conversation/src/persistence/
  mod.rs           public API — `SubstratePersistence`
  record.rs        the record types + header; encode/decode codec
  log_file.rs      append-only file: create / pre-grow extents,
                   buffered append, fsync durability, group commit
  walker.rs        skip-load header walk (§5.4)
  manifest.rs      in-RAM singleton index; last-writer-wins (§5.5)
  streams.rs       stream registry — StreamId, StreamKind, StreamDecl
  content_hash.rs  the prefix-hash chain (§5.2)
  header_index.rs  the batched record-digest chain (§5.6)
  recovery.rs      chain-first recovery + forward-walk fallback (§5.6)
  accounting.rs    O(1) live/dead byte accounting (§5.8)
  compaction.rs    whole-file dead-record rewrite (§5.8)
  warm_pool.rs     RAM warm tier — pinned buffers, LRU (§10)
  inherit.rs       multi-log loading, the inherited chain, shared reuse
  transfer.rs      drives candle-nn's kv_pack/kv_unpack; staging buffers,
                   copy stream, NVMe file read/write, HtoD/DtoH
```

(`transfer.rs` is the persistence-side orchestrator; the kernel itself is
`candle-nn`'s `kv_cache/chunked/migrate.rs` — see §13.4.)

`record.rs`, `walker.rs`, `manifest.rs`, `content_hash.rs`,
`header_index.rs`, `recovery.rs`, and `accounting.rs` are **pure CPU
logic** — tested with in-memory byte buffers, no GPU, no real files. `log_file.rs` and `compaction.rs` are tested against
a temp directory. `warm_pool.rs` and `transfer.rs` need CUDA, but their
non-GPU logic (LRU bookkeeping, plan-building) is factored into GPU-free
units. See §13.7.

### 13.4 Public API surface

```rust
use std::path::PathBuf;
use candle::Result;
use candle_nn::kv_cache::chunked::SealedSequence;

use crate::persistence::record::{ModelSpec, Template};
use crate::persistence::streams::{ContentAddress, StreamId, StreamRef};
use crate::persistence::transfer::ChunkBatch;

/// The persistence layer behind a substrate. Owns the active redo log, the
/// inherited read-only logs, the warm pool, and the layered manifest.
pub struct SubstratePersistence { /* … */ }

impl SubstratePersistence {
    /// Open `.substrate/substrate.log` (created if absent) under the
    /// process working directory, recovering the manifest.
    pub fn open() -> Result<Self>;

    /// Open an ordered list of logs. The last is the active, writable log;
    /// the rest are inherited and read-only. Inherited logs are loaded
    /// through the shared cache (§13.5), so a common base is loaded once.
    pub fn open_concat(logs: &[PathBuf]) -> Result<Self>;

    /// Append migrated chunks (sealed or partial) for a stream, declaring
    /// the stream and writing its Tokens / Signatures if new. Group commit
    /// decides when this physically flushes.
    pub fn append(&mut self, batch: ChunkBatch) -> Result<()>;

    /// Force a durable group-commit flush and a Commit record now.
    pub fn commit(&mut self) -> Result<()>;

    /// Compact the active log — whole-file rewrite dropping dead records
    /// (§5.8). Triggered automatically past the dead-byte-ratio threshold
    /// (`should_compact`, O(1) from the incremental accounting), or
    /// called explicitly. The in-RAM state is re-walked from the
    /// compacted file and cold-tier references are re-pointed in place.
    pub fn compact(&mut self) -> Result<()>;

    /// Resolve a content-addressed prompt section across the active log and
    /// every inherited log. `Some` is a prefix-cache hit; `None` means the
    /// section must be prefilled and appended.
    pub fn lookup_section(&self, addr: ContentAddress) -> Option<StreamRef>;

    /// Load a stream's chunks into VRAM — cold (bridge: NVMe → pinned
    /// host scratch → HtoD into VRAM staging scratch; targets GDS
    /// `cuFileReadAsync` once Linux production is in place) or warm
    /// (warm pinned buffer → VRAM).
    pub fn load_stream(&self, stream: StreamId) -> Result<SealedSequence>;

    /// Model spec / template — last-writer-wins records. The setters append
    /// a fresh record only when the value differs from what is on file.
    pub fn model_spec(&self) -> &ModelSpec;
    pub fn template(&self) -> &Template;
    pub fn set_model_spec(&mut self, spec: ModelSpec) -> Result<()>;
    pub fn set_template(&mut self, template: Template) -> Result<()>;
}
```

The candle-nn primitive the layer calls:

```rust
// candle-nn/src/kv_cache/chunked/migrate.rs
//
// One FFI entry point `kv_migrate_chunks`, parameterised by direction,
// registered in candle-kernels/build.rs and bound in candle-kernels/src/lib.rs.

pub fn kv_pack(/* per_head_table, plan, staging */) -> Result<()>;
pub fn kv_unpack(/* staging, dst_per_head_table, plan */) -> Result<()>;
```

### 13.5 Multi-log inheritance and shared substrate

`open_concat(&[base, … , active])` loads an ordered chain of logs:

- **Active log** — the last entry; the only **writable** one. All appends
  and `Commit`s go here.
- **Inherited logs** — every earlier entry; **read-only**. Their streams
  are visible to the child for resolution but never mutated.

The manifest is a **stack** — lookup walks active → … → oldest and takes
the first hit. Because the active log is newest, this *is* last-writer-wins
across logs: a child can extend an inherited conversation by appending new
turn streams (or superseding chunks) to its active log, and resolution
naturally prefers them.

**Shared-memory reuse.** Inherited logs are loaded through a process-wide
**`InheritedSubstrate` cache** keyed by canonical path + fingerprint.
`InheritedSubstrate::load(path) -> Arc<InheritedSubstrate>` returns the same
`Arc` for repeated opens, so when many child substrates inherit a common
base — a shared codebase context, a shared system prompt — that base's
manifest and resident streams exist **once** in memory and are referenced,
not copied. This is the mechanism for a common inherited substrate that
many child substrates build on. Content-addressed section streams (§5.2)
make the sharing exact: child and base compute identical `stream_id`s.

```
   InheritedSubstrate (Arc, loaded once)
        base.log  ──┐
                    ├──▶ child A  (active: A/.substrate/substrate.log)
                    └──▶ child B  (active: B/.substrate/substrate.log)
```

### 13.6 Substrate integration

The persistence layer is wired into `candle-conversation`'s substrate at
two seams:

- **Construction.** A `Substrate` is built *from* a `SubstratePersistence`;
  `recover()` rebuilds `ModelSpec`, `Template`, the stream DAG, and every
  turn's metadata from the log before the substrate is usable.
- **The `migrate_to_cpu` seam.** The substrate already calls a
  `migrate_to_cpu` closure at every seal boundary (§7.1). That closure's
  body becomes the persistence layer's evict path — `kv_pack` →
  pinned DtoH → warm pool → group-commit `append`. The old unbatched
  implementation and the dead `read_raw_sealed_chunk` are deleted; no new
  call site is introduced.

The scheduler tick (§11.2) drives evict/load through the same layer.

### 13.7 Testing strategy and conventions

- **Extensive unit tests, built as the layer is built.** Every file in
  §13.3 ships with its own tests. The codec (`record.rs`) is tested with
  **raw-byte round-trip assertions** — encode, compare against a fixed
  expected byte image, decode, assert structural equality — not tolerance
  checks. `walker.rs` / `manifest.rs` / `recovery.rs` are tested by
  constructing in-memory log byte buffers and asserting the recovered
  manifest. `content_hash.rs` is tested for the cascade property (§5.2).
- **No backward compatibility.** This design may break everything before
  it. There are no compatibility shims, no `Option`-typed feature seams, no
  dual code paths — the persistence layer is mandatory and the old
  `migrate_to_cpu` body, `read_raw_sealed_chunk`, `store.rs`, and the
  `provenance/` stores are deleted outright. Genuine `Option`s (a
  cache hit/miss) stay; optionality-as-a-feature-flag does not.
- **Coding conventions.** Never write a fully-qualified type path inline —
  every type is `use`-imported at the top of its file. Keep one concern per
  file inside the `persistence/` module subfolder (§13.3).

---

## 14. Phasing — overview

The build is **bottom-up**: leaf building blocks first, the CUDA kernel and
subsystem integration last. Nine phases (P0–P8), each gated by a hard test
pass. **§16 is the detailed, executable plan** — concrete tasks, tests, and
commit gates per phase. At a glance:

- **P0 — Persistence primitives.** `record.rs`, `content_hash.rs`,
  `streams.rs` — pure-CPU codec, checksums, content-hash chain.
- **P1 — Log file, walker, manifest, recovery.** The append-only file,
  skip-load walk, last-writer-wins manifest, recovery — still pure CPU.
- **P2 — `SubstratePersistence` + inheritance.** The public API, multi-log
  `open_concat`, the shared `InheritedSubstrate` cache.
- **P3 — The migration kernel.** `kv_pack` / `kv_unpack` in `candle-kernels`
  + `candle-nn` (GPU); byte-identical VRAM↔staging↔VRAM.
- **P4 — Warm tier + transfer.** `warm_pool.rs`, `transfer.rs` — VRAM↔RAM
  evict/load, pinned buffers, copy stream, LRU.
- **P5 — Disk path.** `Chunk` records carry real migrated KV; cold load
  disk→VRAM; inheritance wired end-to-end.
- **P6 — Substrate integration.** Persistence made mandatory; the
  `migrate_to_cpu` seam replaced; `store.rs` / `read_raw_sealed_chunk` /
  `provenance/` stores deleted; the eviction policy.
- **P7 — Log compaction.** The whole-file dead-record rewrite (§5.8).
- **P8 — `zend` daemon integration.** End-to-end persist + resume across
  daemon restarts.

---

## 15. Resolved design decisions

Every question raised during design review is settled. Each decision is
fully in scope and implemented by the plan in §16 — nothing here is
deferred.

1. **Plan granularity → fine-grained, one record per sub-chunk.** The
   migration plan carries one copy record per `(arena, chunk, kv, head,
   palette)` sub-chunk. Correct by construction and the plan-builder is
   trivial; the kernel body handles a variable `byte_len`. This is the
   final design — there is no coarser variant to fall back to or refine
   into later.

2. **Group-commit trigger → whichever-first.** A size threshold (sized to
   produce one large sequential write, tuned to the RAID stripe) bounds
   throughput; a time bound caps durability latency so a low-traffic
   conversation's tail does not sit un-durable. Whichever fires first
   flushes. The constants live in `log_file.rs`'s group-commit buffer (P1)
   and are exercised by P5's KV-append tests.

3. **Cross-fork sharing → content-addressed streams (§5.2).** Prompt-section
   streams are content-addressed by a prefix-hash chain, so two developer
   forks of the same codebase/prompt independently compute identical
   `stream_id`s and share the section streams structurally — no dedup pass,
   no `ChunkRef` indirection. Stale records left by supersession are
   reclaimed by compaction (§5.8).

4. **Cold-load transport → pinned-staging bridge, GDS-shaped API.** The
   target transport is GPUDirect Storage (`cuFileReadAsync`: NVMe
   controller DMAs straight through the GPU's PCIe BAR into VRAM, no
   host bounce buffer). GDS is **Linux-only** — it depends on the
   `nvidia-fs` kernel module that NVIDIA does not ship for Windows;
   Microsoft's DirectStorage is the Windows analogue but isn't
   reachable from CUDA. Until the production Linux workstation lands
   (and until/if a CUDA-reachable Windows GPU-storage API exists), the
   cold-load path is a **pinned-staging bridge** (`pread` into a
   `cudaHostAlloc`'d pinned host buffer + `cuMemcpyHtoDAsync` to the
   VRAM staging scratch + `kv_unpack` scatter). The interface is
   GDS-shaped — a single `read_kv_bytes_into_staging` entry point —
   so the Linux backend can swap to `cuFileReadAsync` without touching
   the cold-load caller (§4).

5. **Per-head table for the warm tier → host-side only.** A CPU-located
   arena needs no `PerHeadTableEntry`. The kernels only ever dereference
   the VRAM staging scratch and VRAM arenas — never CPU memory. The warm
   pool is addressed entirely host-side; a warm→VRAM load has the host read
   the pinned buffer and issue the HtoD.

6. **`bg_quantizer` interaction → reconcile barrier is `bg_quantizer.join()`.**
   The background quantiser is a worker thread; `enqueue_reconcile_batch`
   queues float→quant work without joining. `BackgroundQuantizer::join()`
   blocks until the worker drains all pending work *and* completes its
   bg-stream CUDA work. The evict phase calls `join()` before building the
   migration plan → every sealed chunk is then in its final, byte-stable
   quantized form. Combined with invariant 2 (only fully-sealed sequences'
   sealed chunks are quantized; the tail is float and generates no quant
   work), `kv_pack` can never read a chunk mid-recompression. Sufficient.

7. **`PerDepthScores` → persisted, not recomputed.** The per-turn relevance
   scores are accumulated state, not cheaply recomputable (recompute would
   need to replay provenance queries). They are persisted in the turn's
   `StreamDecl` record, last-writer-wins as they evolve.

8. **`store.rs` → retired.** Once the redo log carries `Tokens`,
   `Signatures`, and per-turn `StreamDecl` metadata it is a complete
   substrate image (§5.7) and reproduces turn text via `detokenize`. The
   legacy YAML transcript log is redundant and removed; a human-readable
   export can be generated from the log on demand if wanted.

---

## 16. Implementation plan

This is the executable, bottom-up build plan. It expands §14 into concrete
tasks, tests, and commit gates.

### 16.0 Build status

| Phase | State | Commit | Verified |
|-------|-------|--------|----------|
| P0 — persistence primitives | **done** | `4847bc92` | 27 CPU tests |
| P1 — log file / walker / manifest / recovery | **done** | `b877914d` | 21 CPU tests (on-disk crash recovery) |
| P2 — `SubstratePersistence` + inheritance | **done** | `b7e10f26` | 9 CPU tests |
| P3 — `kv_migrate` migration kernel | **done** | `7e3451cb` | `nvcc` build + GPU round-trip on RTX 4090 |
| P4 — warm tier + `transfer.rs` (evict/load) | **done** | `eba22394` `e4cffe1e` `4a6d77cc` | 5 CPU + GPU gather/scatter round-trip |
| P5 — disk path (`Chunk` records, cold load) | **done** | uncommitted (grouped) | 71 CPU tests, builds + GPU |
| P6 — substrate integration | **core done** | uncommitted (grouped) | `Conversation` carries mandatory persistence; turns persist; 289 candle-conversation tests pass |
| P7 — log compaction | **done** | uncommitted (grouped) | 73 persistence tests; compact → reopen-identical, dead-ratio trigger |
| P8 — `zend` daemon integration | **wiring done** | uncommitted (grouped) | daemon opens substrate at `<workdir>/.substrate/`; per-turn group-commit; graceful Ctrl-C/SIGTERM shutdown commits durably; GPU-gated e2e test asserts turns recover across restart |

> **Post-P8 revision.** The `Checkpoint` record and its superblock hint
> were removed: the checkpoint snapshot carried only singleton offsets
> (per-entity state had already moved to the substrate), so it never
> shortened the walk. Recovery was then rebuilt around the
> **`HeaderIndex` chain** (§5.6, `header_index.rs` + `recovery.rs`):
> batched header digests chained backwards from a superblock hint, with
> the filtered forward walk as the universal fallback. `accounting.rs`
> tracks the dead-byte ratio O(1) per append, and the persistence
> thread compacts automatically past the threshold (§5.8). References
> to `checkpoint.rs` / `checkpoint()` in the phase records below are
> historical.

> **P8 resume reconstruction — status.** The recording path is live (per-turn
> group-commit, durable commit on shutdown). The §5.6/§5.7 resume
> reconstruction is built **device-free-first**:
>
> - **Done (CPU-tested).** `persistence/resume.rs` — the Option-A on-disk
>   layout (`chunk_index = layer*C + chunk`), the `L×C` demux, the `Tokens`
>   codec, and `persist_turn_kv` / `recover_turn_grid` +
>   `recover_turn_meta` round-trips (the restore loop reads metadata only;
>   chunk payloads load on cold→hot elevation). `read_tokens`
>   on `SubstratePersistence`. 7 unit tests; persist→reopen→recover verified.
> - **Remaining (GPU-coupled).** The byte legs that move KV across the PCIe
>   bus, built on the **layout-agnostic migration primitives** so the
>   round trip is correct by construction — *not* on `read_raw_sealed_chunk`
>   / `write_raw_sealed_chunk`, which inline (mutually inconsistent) layout
>   arithmetic. (a) At seal, `resolve_sealed_chunk_ptrs` + `gather_chunks`
>   the layer's `SealedSequence` into one opaque host blob, split per chunk
>   by `SealedChunk.byte_size` into `ChunkImage`s, `persist_turn_kv`.
>   (b) On resume, allocate a destination `SealedSequence` in arenas of the
>   persisted `format`, `scatter_chunks` the recovered `kv_bytes` back
>   (exactly `cold_load_stream`'s path). Both directions go through the
>   *same* `resolve_sealed_chunk_ptrs` and the *same*
>   `arena.chunk_byte_stride` (a pure function of `format`) — the bytes are
>   opaque memcpy, so the layout cannot desync. (c) Repopulate the
>   `Substrate` turn entries and wire reconstruction into
>   `ConversationEngine::new`. (d) Persist the daemon's
>   `conv_id ↔ timeline_id` map. The GPU only verifies the kernel runs —
>   not that two layout calculations agree.

**67 tests pass** across the committed phases; `cargo fmt` + `cargo clippy`
clean on all new code. The persistence foundation (on-disk format, durable
log, recovery, content-addressed streams, inheritance, the public API), the
GPU migration kernel, and the warm tier are complete and verified.

**P4b key facts (resolved during study).** A `SealedChunk`'s device address
resolves host-side as
`arena_info[gid.arena_idx()].base_ptr + gid.chunk_idx() * chunk_byte_stride`,
where `arena_info` comes from `ChunkedKvBacking::resolve_arena_info()` and a
chunk's GIDs are reached via `cw.gids.k_gid_pal(h,p)` / `v_gid_pal(h,p)`
(pattern: `backing.rs::gather_r16_kv_probe`). The `SealedSequence` →
`MigrationPlan` builder therefore needs a method *on* `ChunkedKvBacking`
(it consumes `pub(crate)` arena internals); `load_to_hot` additionally
allocates fresh chunks through the `GidPool`.

### 16.1 Standing rules (apply to every phase)

- **Bottom-up.** Leaf building blocks first; the CUDA kernel and subsystem
  integration last. A phase only begins once the previous phase's exit gate
  is green.
- **Test-driven.** Each `persistence/` file ships with its unit tests in
  the same change. Phase boundaries add integration tests. Quant/codec
  tests use **raw-byte assertions**, never tolerance checks.
- **Design-doc precedence.** Every detail is cross-checked against the real
  code *and* this document. On a discrepancy the document wins — *unless*
  the document is demonstrably wrong, in which case **update the document**
  in the same change (and note it in the commit) and proceed.
- **Fail forward — no stubs, no `TODO`s.** Never leave the workspace
  non-compiling between phases, and never stall. "Fail forward" means: when
  a task turns out harder than expected, **solve it fully** and move on —
  it does *not* mean leaving a placeholder. A phase is not complete until
  every task in it is genuinely, completely implemented and tested. No
  `TODO` / `FIXME` / `unimplemented!()` / stub may be committed; no task is
  pushed to a later phase. Every feature this document describes is built.
- **Exit gate (every phase).** The phase's own code passes `cargo fmt`
  and `cargo clippy ... -D warnings` cleanly, and the phase's tests pass
  (GPU phases additionally run `cargo test --features cuda`). The gate is
  scoped to the code the phase writes or touches — pre-existing lint /
  format debt elsewhere in the workspace is **not** in scope and is not
  retroactively fixed (consistent with not dragging unrelated cleanup into
  a change).
- **Commit per phase.** The moment a phase's gate is green, `git commit`
  (the user has pre-authorised these per-phase commits for this run). One
  commit per phase; message prefix `kv-tier(Pn): …`.
- **Workspace.** `candle-conversation/src/persistence/` is the new module;
  the kernel work is in `candle-kernels/` + `candle-nn/`. Confirm
  `blake3` (content hashing) and a CRC32C implementation are workspace
  dependencies in P0; add them if absent.

### 16.2 Phase P0 — Persistence primitives (pure CPU)

**Goal.** The `persistence/` module skeleton and its leaf building blocks,
fully unit-tested, no GPU, no files.

**Files.** `persistence/mod.rs` (skeleton + `pub mod` wiring),
`record.rs`, `content_hash.rs`, `streams.rs`; register `mod persistence` in
`candle-conversation/src/lib.rs`.

**Tasks.**
1. `streams.rs` — `StreamId` (newtype over `u64`), `StreamKind`
   (`Turn` / `PromptSection`), `StreamRef`, `StreamDecl` (the per-kind
   metadata of §5.3), `ContentAddress` (`(prefix_hash, section_hash)`).
2. `content_hash.rs` — the prefix-hash chain of §5.2:
   `chain[i] = H(chain[i-1] ++ section_i_tokens)`; `ContentAddress`
   derivation; the turn/section `StreamId` derivation.
3. `record.rs` — the fixed record **header** struct (§5.3 table), the
   `RecordType` enum (8 variants), each record type's payload struct, and
   the **encode/decode codec**: 4 KB alignment + padding, `length`,
   checksum (CRC32C over header+payload), `magic`.

**Tests.** Raw-byte round-trip for every record type — encode → assert
against a fixed expected byte image → decode → assert structural equality;
checksum catches a flipped byte; padding/alignment is exactly 4 KB;
content-hash **cascade** (mutate section *i* → every downstream `StreamId`
changes; unchanged input → byte-identical `StreamId`).

**Exit gate.** `cargo test -p candle-conversation persistence::` green +
fmt + clippy. **Commit** `kv-tier(P0): persistence primitives — record codec, content hash, streams`.

### 16.3 Phase P1 — Log file, walker, manifest, recovery (pure CPU)

**Goal.** A working append-only log on disk that can be walked, indexed,
checkpointed, and recovered — with synthetic record payloads, no KV/GPU.

**Files.** `persistence/log_file.rs`, `walker.rs`, `manifest.rs`,
`checkpoint.rs`.

**Tasks.**
1. `log_file.rs` — create/open a log file; minimal file header (magic +
   log-format version, §5.1); **pre-grow** in large extents
   (`set_len`); append into a group-commit staging buffer; `sync_data`
   (`fsync`) on commit for durability. Records are 4 KB-aligned (§5.3) for
   clean boundaries. Buffered file I/O + `fsync` is the durable-logging
   mechanism (the standard SQLite/WAL approach); `O_DIRECT`-style
   unbuffered I/O is a deliberately deferred perf option, not used.
2. `walker.rs` — the skip-load header walk (§5.4): read header → jump
   `padded(length)` → repeat; stop at first bad checksum.
3. `manifest.rs` — the in-RAM index; build it from a walk; **last-writer-wins**
   per `(stream_id, chunk_index)` and per singleton record type; the
   **layered** (stack) structure (one layer now; inheritance in P2).
4. `checkpoint.rs` — serialise the manifest into a `Checkpoint` record;
   recovery = latest checkpoint + tail replay (§5.6); torn-tail truncation.

**Tests.** Append synthetic records to a temp-dir log, walk → assert
manifest; supersession (same key twice → last wins); checkpoint
serialise/deserialise round-trip; **crash recovery** (truncate the file
mid-record → recovery stops at the torn record and the prior state stands);
pre-grow leaves the logical length correct.

**Exit gate.** As P0. **Commit** `kv-tier(P1): append-only log file, skip-load walker, manifest, recovery`.

### 16.4 Phase P2 — `SubstratePersistence` + inheritance (pure CPU)

**Goal.** The public API working for everything that does not need a GPU;
multi-log loading with shared inherited substrate.

**Files.** `persistence/mod.rs` (the real `SubstratePersistence`),
`inherit.rs`.

**Tasks.**
1. `mod.rs` — `SubstratePersistence::open()` (ensures `.substrate/`,
   opens/creates `.substrate/substrate.log`, recovers) and `open_concat()`;
   `append` for non-`Chunk` records (`StreamDecl` / `Tokens` /
   `Signatures`); `commit`; `checkpoint`; `set_model_spec` / `set_template`
   (last-updated — append only when changed); `lookup_section`;
   `model_spec` / `template` accessors.
2. `inherit.rs` — `InheritedSubstrate::load(path) -> Arc<InheritedSubstrate>`
   with a **process-wide cache** keyed by canonical path + fingerprint;
   layered-manifest resolution across active + inherited logs.

**Tests.** `open()` creates `.substrate/substrate.log`; `ModelSpec` /
`Template` last-updated (no-op when unchanged, fresh record when changed);
`open_concat` of 3 logs — last is writable/active, earlier ones read-only;
**shared reuse** — load the same base twice → `Arc::ptr_eq` holds;
`lookup_section` resolves a hit located in an inherited log; recovery of a
multi-log chain.

**Exit gate.** As P0. **Commit** `kv-tier(P2): SubstratePersistence API + multi-log inheritance`.

### 16.5 Phase P3 — The migration kernel (`candle-kernels` + `candle-nn`, GPU)

**Goal.** The `kv_migrate` scatter/gather primitive; byte-identical
VRAM → staging → VRAM.

**Files.** `candle-kernels/src/simple/kv_migrate.cu` + `kv_migrate.rs`
(the `simple` archive group, `simple/mod.rs`), `candle-kernels/build_utils.rs`
(register the `.cu`), `candle-nn/src/kv_cache/chunked/migrate.rs`,
`candle-nn/src/kv_cache/mod.rs` (public re-export).

**Tasks.**
1. `kv_migrate.cu` — the index-free scatter/gather kernel (§9): one block
   per plan record, copies `byte_lens[r]` bytes `src_ptrs[r]→dst_ptrs[r]`,
   16-byte vectorised when aligned. Added to the `simple` archive group.
2. `kv_migrate.rs` — the FFI binding, exported via `simple/mod.rs`.
3. `migrate.rs` — `MigrationRecord` / `MigrationPlan` and the `kv_migrate`
   primitive that uploads the plan arrays and launches the kernel once;
   re-exported from `candle-nn` as public API for the persistence layer.
   The plan-builder that derives a plan from a `SealedSequence`'s GIDs is
   part of P4 (where `SealedSequence` migration is exercised).

**Tests.** `MigrationPlan` accumulation (CPU). GPU (`--features cuda`):
gather three scattered device chunks into a contiguous staging buffer via
`kv_migrate`, scatter them back into fresh buffers, assert **byte-identical**
round-trip.

**Exit gate.** As P0 **plus** `cargo build --features cuda` and
`cargo test --features cuda` for the migrate tests. **Commit**
`kv-tier(P3): kv_pack/kv_unpack migration kernel + plan-builder`.

### 16.6 Phase P4 — Warm tier + transfer orchestration (VRAM↔RAM, GPU)

**Goal.** `evict_to_warm` / `load_to_hot` — the full VRAM↔RAM path:
pinned buffers, dedicated copy stream, LRU.

**Files.** `persistence/warm_pool.rs`, `persistence/transfer.rs`.

**Tasks.**
1. `warm_pool.rs` — a pool of pinned host buffers (on
   `candle-core/.../pinned_staging.rs`); LRU recency; clean/dirty bits
   (§10). Non-GPU bookkeeping factored for isolated tests.
2. `transfer.rs` — the **`SealedSequence` → `MigrationPlan` builder**
   (walk `chunks[*].gids`, dedup via `HeadGids::unique_arena_indices` /
   `map_unique`, resolve device addresses host-side, §8); the dedicated
   copy stream (pattern from `select_and_summarize_kv_winners_paged_staged`);
   `evict_to_warm` (`bg_quantizer.join()` barrier → `kv_migrate` gather →
   async DtoH → warm pool); `load_to_hot` (HtoD → `kv_migrate` scatter);
   the fixed staging scratch + wave handling (§11.4).

**Tests.** `warm_pool` LRU/clean-dirty unit tests (CPU). GPU: evict a
sequence VRAM→RAM then load it back → byte-identical; the `bg_quantizer`
barrier holds; copy-stream overlap (event-gated, no main-stream stall).

**Exit gate.** As P3. **Commit** `kv-tier(P4): warm pool + VRAM↔RAM transfer orchestration`.

### 16.7 Phase P5 — The disk path: KV chunks + cold load + inheritance (GPU + files)

**Goal.** `Chunk` records carry real migrated KV; full disk round-trip;
cold load disk→VRAM; inheritance wired end-to-end.

**Files.** extend `mod.rs` (append `Chunk` records), `transfer.rs`
(cold-load pipeline), `inherit.rs` (cold-load from inherited logs).

**Tasks.**
1. `Chunk` record encode/decode: the metadata prefix (`offset`,
   `k_pal`/`v_pal`/`k_scale`/`v_scale`) + the arena blob (§5.3) — **without
   the prefix the KV is undecodable**; raw-byte tests.
2. `append` for warm chunks → `Chunk` records; group-commit flush; the
   partial-tail rule (§3, force-flush on eviction); and the **snapshot
   variant** (§3 — persist a still-hot sequence: the same `kv_pack` → DtoH
   → append path, but the VRAM `ChunkGid`s are retained, not dropped).
3. Cold load (§4, §15 q4) — **bridge** (NVMe `pread` into a pinned
   host scratch → `cuMemcpyHtoDAsync` into the VRAM staging scratch →
   `kv_unpack`). The pinned host and VRAM staging scratches are
   double-buffered so the NVMe leg, the HtoD leg, and `kv_unpack`
   pipeline; provenance-driven ordering loads the selected chunks
   first. The **target** is GDS (`cuFileReadAsync` into the VRAM
   staging scratch, no host hop) on Linux + nvidia-fs; the bridge
   interface is shaped so this is a backend swap.
4. `inherit.rs` — cold-load a stream resident only in an inherited log;
   resolve a conversation's full stream DAG across active + inherited logs.

**Tests.** Full round-trip — evict a sequence, flush to disk, drop the warm
copy, **cold-load from disk**, assert byte-identical KV; **restart
simulation** — a fresh `SubstratePersistence::open` on the same file
recovers and loads the sequence; cold-load of an inherited stream;
partial-tail survives a restart.

**Exit gate.** As P3. **Commit** `kv-tier(P5): disk path — Chunk records, cold load, inheritance`.

### 16.8 Phase P6 — Substrate integration (persistence made mandatory)

**Goal.** Wire the persistence layer into `candle-conversation`'s substrate;
delete the redundant code.

**Files.** `substrate.rs`, `conversation.rs`, `projection/resolver.rs`;
deletions of `store.rs`, `read_raw_sealed_chunk` (`kv_cache/chunked/io.rs`),
`provenance/raw_store.rs`, `provenance/store.rs`.

**Tasks.**
1. `Substrate` is constructed *from* a `SubstratePersistence`; `recover()`
   rebuilds `ModelSpec`, `Template`, the stream DAG, and per-turn metadata
   (§13.6). No in-memory-only path.
2. Replace the `migrate_to_cpu` closure body with the P4 evict path; delete
   the old unbatched implementation and the dead `read_raw_sealed_chunk`.
3. Map `TurnEntryData` / `SectionEntryData` ↔ records: `role`,
   `timeline_id`, `turn_index`, `TurnId`, `block_range`, `view`,
   `PerDepthScores` → `StreamDecl`; `token_ids` → `Tokens`; `sig_entries`
   → `Signatures`. Populate `ModelSpec` from `models/` + `config.rs` and
   `Template` from `projection/schema.rs`.
4. Delete `store.rs`, the dead `read_raw_sealed_chunk`, and the
   `provenance/` stores outright.
5. Implement the **eviction policy** (the absent `EvictionManager`): a VRAM
   high-water/low-water mark drives eviction; LRU over cold sequences picks
   victims; the scheduler tick (§11.2) runs evict → load → inference in the
   §11.3 order. This is the complete policy, not a placeholder.

**Tests.** Integration — create a conversation, add several turns and
sections, restart (reopen the substrate), assert the conversation
reconstructs **identically** (turn text via `detokenize`, scores, views);
section content-addressing produces a cache hit on re-ingest; eviction
under a forced VRAM watermark evicts the LRU sequence and inference
continues; the existing `candle-conversation` test suite still passes.

**Exit gate.** As P3, plus the full `candle-conversation` suite. **Commit**
`kv-tier(P6): substrate integration — mandatory persistence, eviction policy, redundant code removed`.

### 16.9 Phase P7 — Log compaction

**Goal.** Implement compaction (§5.8) — the whole-file rewrite that
reclaims dead records.

**Files.** `persistence/compaction.rs` (new); wiring in `mod.rs`.

**Tasks.**
1. `compaction.rs` — quiesce (flush + `Commit`); rebuild the manifest from
   a clean walk; stream every live record (latest `ModelSpec` / `Template`,
   reachable `StreamDecl` / `Chunk` / `Tokens` / `Signatures`, a fresh
   `Checkpoint`) into `.substrate/substrate.log.compact` in dependency
   order; `fsync`; atomic rename over `.substrate/substrate.log`.
2. The dead-record-ratio trigger, measured during the manifest rebuild;
   an explicit `compact()` entry point on `SubstratePersistence`.
3. Post-swap: flush VRAM and full-reload the substrate from the compacted
   file (§5.8). Inherited logs are never compacted by a child.

**Tests.** Build a log with known dead records (superseded partials, a
stale `Template`, an orphaned stream), compact, assert the new file
contains exactly the live set and recovers to an identical substrate;
atomic-rename crash safety (a crash mid-compaction leaves the original
intact); the dead-ratio trigger fires at the threshold.

**Exit gate.** As P3. **Commit** `kv-tier(P7): log compaction — whole-file rewrite`.

### 16.10 Phase P8 — `zend` daemon integration (end-to-end)

**Goal.** The `zend` daemon runs on the persistence layer; conversations
persist and resume across daemon restarts.

**Files.** `zend/src/` (daemon startup / conversation lifecycle).

**Tasks.**
1. Daemon startup opens/recovers the substrate at
   `<workdir>/.substrate/substrate.log`; optional inherited base via
   `open_concat`.
2. Conversation create / turn / section paths go through the persistence
   layer; group-commit + checkpoint cadence wired to the daemon loop.
3. **Graceful shutdown — including `Ctrl-C`.** Install a `SIGINT` /
   `Ctrl-C` handler (and the Windows console-close / `SIGTERM` paths) that
   runs the shutdown sequence: stop accepting work, `commit()` the
   group-commit buffer (durably flushing any un-flushed partial tails so
   the in-flight turn is not lost), `checkpoint()`, then exit. A second
   `Ctrl-C` aborts immediately. Shutdown must be idempotent and must not
   deadlock against the scheduler tick.

**Tests.** End-to-end — start the daemon, run a multi-turn conversation,
**restart the daemon**, resume the same conversation and assert continuity;
**`Ctrl-C` mid-turn then restart** — the interrupted turn's tail is durable
and resumes intact; measure cold-resume load time; a shared inherited base
across two daemon working directories; a compaction pass mid-session.

**Exit gate.** As P6, plus the end-to-end daemon test. **Commit**
`kv-tier(P8): zend daemon integration — persist and resume across restarts`.

### 16.11 Completion

After P8 the three-tier KV persistence system is **complete and live
end-to-end**: every component this document specifies — the migration
kernels, the warm tier, the nine-record redo log of content-addressed
streams, cold load, multi-log inheritance, compaction, mandatory substrate
integration, and the `zend` daemon — is implemented and tested. Nothing is
deferred and nothing is stubbed; the design carries no out-of-scope
remainder.

### 16.12 Corrected plan — the missing allocation keystone (P4b)

An audit of P4/P5/P8 during resume implementation found a structural gap
the original §16 plan did not name. It is recorded here as the authoritative
correction; **P4b** must land before P8 resume can work.

**The finding.** Every *restore-direction* primitive the project built —
`scatter_chunks` and `load_to_hot` (P4), `cold_load_stream` (P5) — takes a
**pre-allocated `&SealedSequence`** and only moves bytes into it. None of
them allocate. The design's §16-era API sketch declared
`load_stream(StreamId) -> Result<SealedSequence>` (a signature that *returns*
— i.e. allocates — a sealed sequence), but P5 shipped `cold_load_stream(…,
seq: &SealedSequence, …) -> Result<()>` instead. The allocation keystone was
specified and built by **zero phases**.

The only code that allocates a sealed chunk's GIDs is the pre-existing
`write_raw_sealed_chunk` (`chunk_ops.rs`), and it welds allocation to a
palette4-regime-specific byte split. Meanwhile `read_raw_sealed_chunk`
(`io.rs`) inlines a *different*, flat/R16-regime layout. The two are not an
inverse pair; the only layout-agnostic primitive is `resolve_sealed_chunk_ptrs`
(it walks `chunk.gids.0` and reads `arena.chunk_byte_stride`, a pure function
of `arena.format()`).

**Consequences.** `load_to_hot` (P4) can only scatter into a `SealedSequence`
whose GIDs are still allocated — so warm eviction that *frees* VRAM cannot be
reloaded; the P4 round-trip test passed only because it reused a still-live
`SealedSequence`. `cold_load_stream` (P5) cannot cold-load into a fresh
process. P8 resume is blocked outright.

**P4b — the allocation keystone.**

1. **`ChunkedKvBacking::alloc_sealed_block`** (`chunk_ops.rs`, new). The
   allocation half of `write_raw_sealed_chunk`, decoupled from byte I/O:
   allocate the chunk's per-`(head, palette)` GIDs in arenas of the given
   `KvFormat`, register them on the slot's block table with the chunk's
   `offset` / `token_count` / palettes / scales. **One source of truth** for
   "what GID shape does format `F` need." `write_raw_sealed_chunk` is
   refactored to `alloc_sealed_block` + a thin byte-write so no second
   allocation path can drift.
2. **`load_stream(StreamId) -> Result<SealedSequence>`** — the design's
   real signature, built on the keystone: allocate a scratch slot, call
   `alloc_sealed_block` per chunk (format from the persisted `Chunk` record),
   `record_turn` to snapshot a `SealedSequence` with real GIDs,
   `scatter_chunks` the persisted `kv_bytes` via `resolve_sealed_chunk_ptrs`.
   `cold_load_stream` becomes a thin caller of it.

   **P5 record gap (asymmetric K/V format).** Adaptive quantization picks K
   and V formats *independently* per block (C0 = K:R16, V:Q8_0). The `Chunk`
   record carries a single `format: u8` header field — it cannot express the
   pair. `alloc_sealed_block` already takes `k_format`/`v_format` separately;
   `ChunkPayload` must be extended with `k_format` / `v_format` bytes (it
   already carries the per-chunk dequant metadata `k_pal`/`v_pal`/`k_scale`/
   `v_scale`, so the formats belong there). The P5 `ChunkPayload` round-trip
   tests gain the two fields.
3. **`load_to_hot` (P4)** is re-pointed at the keystone so warm→hot reload
   works after a genuine VRAM free; its test is extended to free-then-reload.

**P8 resume, on the corrected base.**

4. **Seal-time gather.** In the scheduler seal path, per layer:
   `resolve_sealed_chunk_ptrs` + `gather_chunks` the turn's `SealedSequence`
   into one opaque host blob; split per chunk by `SealedChunk.byte_size`;
   `persistence::resume::persist_turn_kv` (Option-A flat grid, `Tokens`,
   `Commit`). Already CPU-tested device-free in `persistence/resume.rs`.
5. **Resume reconstruction.** `persistence::resume::recover_turn` →
   `load_stream` per layer → `Vec<SealedSequence>`; repopulate the
   `Substrate` turn entries (`append_full` shape) with recovered metadata,
   token ids, and scores.
6. **Engine wiring.** `ConversationEngine::new`, after opening persistence
   with a non-empty turn manifest, drives steps 4–5 to rebuild the substrate.
7. **Daemon `conv_id ↔ timeline_id`.** Persist the map (a small dedicated
   record, or reuse `TurnDecl.timeline_id` keyed by a `conv_id` the daemon
   records) so a client `conv_id` resolves to a recovered timeline.

**Layout-safety invariant.** Gather and scatter both route through the *same*
`resolve_sealed_chunk_ptrs` and the *same* `chunk_byte_stride`-from-`format`;
`alloc_sealed_block` is the *sole* allocator. The `kv_bytes` blob is opaque
memcpy. There is no second layout calculation for any of them to disagree
with — correctness is structural, not test-discovered.

**Implementation status.**

- **P4b.1 — done.** `alloc_sealed_block` (`chunk_ops.rs`); `write_raw_sealed_chunk`
  refactored to call it. 232 candle-nn chunked tests pass.
- **P4b.2 — done.** `ChunkPayload` carries `k_format`/`v_format`;
  `KvFormat::to_tag`/`from_tag` + `ArenaFormatTag::from_u8`/`to_kv_format`
  (single-source tag codec); `set_block_window`, `sealed_chunk_kv_formats`;
  `transfer::load_stream` — the design's missing primitive. 80 persistence
  tests pass.
- **P4b.3 — done.** `WarmPool` now stores `Vec<ChunkImage>` (the
  realloc-able representation) rather than opaque `Vec<u8>`; `evict_to_warm`
  gathers via the shared `seal_to_chunk_images`; `load_to_hot` rebuilds a
  fresh `SealedSequence` through `load_stream` / the allocation keystone, so
  a warm eviction that genuinely frees VRAM is reloadable. 80 persistence
  tests pass.
- **P8.4 — done.** Seal-time gather (`Scheduler::gather_turn_layers`) +
  `Conversation::persist_turn_kv`: every sealed turn persists its `L×C` KV
  grid + `Tokens` + `Commit`.
- **P8.5/6 — done.** `Substrate::restore_turn`, `Conversation::reconstruct_from_log`,
  `Scheduler::reconstruct_substrate`, wired into `ConversationEngine::new` — on
  startup the substrate is rebuilt from the redo log, each turn's KV
  cold-loaded into VRAM via `load_stream`. `TurnDecl` now persists
  `layer_id`/`group_id`; `LayerId`/`GroupId`/`TimelineId` gained public
  `from_raw`. Builds CPU + CUDA.
- **P8.7 — done.** `Sequence::fork_resuming(timeline)` forks onto a
  specific timeline (registered idempotently) instead of minting a fresh
  one; `Conversation::register_timeline` exposes the binding. The `zend`
  daemon derives a stable `timeline_id` from each `conv_id`
  (`timeline_for` — a content hash) and forks with `fork_resuming`, so a
  client reconnecting after a restart resolves to the timeline the
  substrate reload recovered. `LayerId`/`GroupId`/`TimelineId` gained
  public `from_raw`.

The three-tier KV persistence system is now complete: the redo log, the
allocation keystone, warm-tier evict/reload, cold-load `load_stream`,
seal-time persistence, substrate reconstruction on startup, and daemon
resume routing all build CPU + CUDA. GPU byte-path correctness
(`gather_chunks`/`scatter_chunks` round trip) is structural per the
layout-safety invariant above; end-to-end on-GPU exercise (and the P8
restart/resume integration test) needs the RTX 4090.
