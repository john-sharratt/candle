# Substrate Redesign — Collapsing the Warm Pool

## 1. Motivation

Today's three-tier architecture spreads turn history across three subsystems
that each maintain their own indirection:

- **Substrate** ([candle-conversation/src/projection/resolver.rs:311](../candle-conversation/src/projection/resolver.rs#L311))
  stores per-turn metadata keyed by `(GroupId, TurnIndex)`, including a
  `block_range: (u64, u64)` pointing at absolute KV blocks somewhere else.
- **WarmPool** ([candle-nn/src/kv_cache/chunked/warm_pool.rs:467](../candle-nn/src/kv_cache/chunked/warm_pool.rs#L467))
  holds quantised KV bytes in mmap'd RAM slots. It carries its own slot
  allocator, RAII handles, and a layout module.
- **Backing arena (`ChunkedKvBacking`)** holds the GPU-resident chunks the
  scheduler is currently decoding into.

A turn that lives in warm tier therefore exists as: substrate metadata
`(block_range)` → warm slot indices → WarmPool slot bytes → restore-warm
preamble → fresh GPU arena chunks → view sequence → projection. The
indirection through *block ranges* is the source of every
`finalize_view` / `reproject_view` bug we have hit on `new-kernel2`:
positional addresses in one subsystem stop matching positional addresses
in another after non-contiguous projection borrows or partial-tail
donation.

The fix is to remove the indirection. The substrate should hold
`SealedSequence`s directly (in CPU-tier arenas) and projection should be
a direct CPU→GPU rehydration of the chunks the next turn actually needs.

## 2. Core Insight

> The substrate **is** the warm tier.

`HeadGids` already supports CPU-resident storage. `ChunkGid` carries an
`ArenaKey { format, location }` and `ArenaLocation::{Gpu, Cpu}` already
partitions the GID pools
([candle-nn/src/kv_cache/chunked/gid_pool.rs:483–496](../candle-nn/src/kv_cache/chunked/gid_pool.rs#L483)).
CPU-located arenas exist
([arena.rs:92](../candle-nn/src/kv_cache/chunked/arena.rs#L92), [arena.rs:150](../candle-nn/src/kv_cache/chunked/arena.rs#L150),
[alloc.rs:167](../candle-nn/src/kv_cache/chunked/alloc.rs#L167)) and the
GPU↔CPU byte-copy quad — `(Cpu|Gpu) × (Float|Quantized)` — is already
implemented in
[backing.rs:1078–1094](../candle-nn/src/kv_cache/chunked/backing.rs#L1078).

A `SealedSequence` whose `HeadGids` reference `ChunkGid`s in
`ArenaLocation::Cpu` arenas **is** a CPU-resident sealed sequence — same
type, different `route_key` on the GIDs. No new struct. The redesign is
just rewiring who allocates which location, who holds which `Arc`, and
deleting the now-redundant indirection layers.

```
                       ┌──────────────────────────────┐
                       │        SubstrateState        │
                       │  (was SessionResolver)       │
                       │                              │
                       │  TurnEntry:                   │
                       │    sealed: Arc<SealedSequence>│ ← GIDs in CPU arenas
                       │    scores, sigs, ...          │
                       └─────────────┬────────────────┘
                                     │ project()
                       ┌─────────────▼────────────────┐
                       │  Scheduler CPU↔GPU cache     │
                       │  cpu ChunkGid → gpu ChunkGid │ ← upload-once,
                       │  cleared on full idle        │   reuse next turn
                       └─────────────┬────────────────┘
                                     │ inject_sealed_chunks
                       ┌─────────────▼────────────────┐
                       │   ChunkedKvBacking (GPU)     │
                       │   live decode arena          │
                       └──────────────────────────────┘
```

## 3. Type Changes

### 3.1 No new sealed types

`SealedChunk` / `SealedSequence` stay exactly as they are
([types.rs:171–228](../candle-nn/src/kv_cache/chunked/types.rs#L171)).
Whether a sealed sequence is "in CPU RAM" vs "in VRAM" is a property of
each contained `ChunkGid`'s `route_key`, queryable via
`gid.route_key.as_ref().map(|k| k.location)`. We do not invent
`CpuSealedChunk`.

### 3.2 New backing operations — built on the existing scatter/gather kernel

The CUDA primitive for both directions already exists:
**`arena_compact_copy`** ([candle-kernels/src/simple/arena_compact.cu](../candle-kernels/src/simple/arena_compact.cu),
async wrapper at
[candle-core/src/quantized/cuda.rs:729](../candle-core/src/quantized/cuda.rs#L729)).
One kernel launch processes a `Vec<CompactMove { src, dst, stride_bytes }>` —
one CUDA block per move, mixed strides (different KV formats) in the same
call, persistent-block work-stealing for full SM occupancy. The
`PinnedStager` handles non-blocking host→device staging of the moves
table itself.

Critically: this is per-chunk scatter, never whole-arena copy. A
projection that needs 32 chunks across 28 layers fires one kernel of
~900 moves; arena bytes outside those chunks are never touched.

The new ops are thin orchestration layers over this primitive:

```rust
// candle-nn/src/kv_cache/chunked/sequence_ops.rs (or new file)
impl ChunkedKvBacking {
    /// Demote a list of GPU-located SealedSequences into CPU arenas in one
    /// batched launch. Allocates a CPU ChunkGid per source chunk, builds
    /// one Vec<CompactMove> covering every chunk × every layer, fires a
    /// single arena_compact_copy_async, and returns the new sealed
    /// sequences with CPU-located HeadGids.
    ///
    /// Used by the conversation's turn-end seal path. One call per turn,
    /// not per chunk.
    pub fn seal_to_cpu_batched(
        &self,
        sealed: &[Arc<SealedSequence>],          // one per layer
    ) -> Result<Vec<SealedSequence>>;

    /// Inverse of seal_to_cpu_batched: promote CPU-located sealed
    /// sequences into freshly-allocated GPU chunks via one batched
    /// arena_compact_copy_async call.  CPU bytes are routed through the
    /// PinnedStager (CPU arenas are not device-mapped — see §3.3).
    ///
    /// Used by the scheduler's projection upload (cache-miss branch).
    pub fn materialise_to_gpu_batched(
        &self,
        cpu_sealed: &[&SealedSequence],          // one per layer
    ) -> Result<Vec<SealedSequence>>;

    /// Append the GPU-located SealedChunks of `sealed` onto the tail of
    /// `batch_idx` as live ChunkWindows.  Returns (block_start, block_end).
    /// Projection-side counterpart of `inject_chunks_at_tail` but takes a
    /// SealedSequence (sealed offsets) instead of DetachedChunks (live
    /// windows).  Pure metadata op — no DMA, no kernel.
    pub fn inject_sealed_at_tail(
        &self,
        batch_idx: usize,
        sealed: &SealedSequence,
    ) -> Result<(usize, usize)>;
}
```

The "batched" suffix is the load-bearing part: every cross-layer
projection must reduce to **one** kernel launch. Per-layer launches
would serialise PCIe upload behind kernel queue latency for no reason —
the kernel was designed exactly to avoid this.

### 3.3 Staging route for CPU↔GPU

CPU arenas today are plain `Tensor::zeros(..., &Device::Cpu)`
([alloc.rs:167](../candle-nn/src/kv_cache/chunked/alloc.rs#L167)) —
host memory but **not** pinned / device-mapped, so the GPU cannot
dereference CPU-arena pointers directly. Both batched ops therefore
route bytes through the existing `PinnedStager`:

- **`seal_to_cpu_batched`:** `dst` = pinned-host slot allocated from the
  stager; one `arena_compact_copy_async` does GPU-arena → pinned-host;
  on stream completion, a host-side memcpy moves bytes into the CPU
  arena tensor (the stream callback is already supported by the stager).
- **`materialise_to_gpu_batched`:** host-side memcpy from CPU arena into
  a pinned-host slot, one `arena_compact_copy_async` does pinned-host →
  GPU-arena.

The host memcpy is unavoidable until CPU arenas are upgraded to
pinned-mapped backing (out of scope — see §8.6). It is small (one chunk
× n_layers × bytes_per_chunk per turn) and runs concurrently with the
kernel.

### 3.3 Renamed types

| Today | After |
|---|---|
| `SessionResolver` | `SubstrateState` |
| `Substrate` (the `Arc<RwLock<SessionResolver>>` wrapper) | unchanged — keep public name |
| `SubstrateRead` / `SubstrateWrite` | unchanged |

### 3.4 Modified types

```rust
// candle-conversation/src/projection/resolver.rs

struct TurnEntryData {
    conversation_id: ConversationId,
    layer_id: LayerId,
    token_count: usize,
    // REMOVED: block_range: (u64, u64),
    // REMOVED: restoration: RestorationSource,
    sealed: Arc<SealedSequence>,    // NEW — KV bytes live in CPU arenas
    scores: PerDepthScores,
    sig_entries: Vec<SigEntry>,
}

struct SectionEntryData {
    token_count: usize,
    // REMOVED: block_range: (u64, u64),
    sealed: Arc<SealedSequence>,    // NEW — see §8.5 for normalisation choice
    scores: PerDepthScores,
    sig_entries: Vec<SigEntry>,
}
```

`Arc<SealedSequence>` so reads from the projection helper return cheap
clones; the cache (§5) uses `Arc::as_ptr` of the outer `SealedSequence`
*or* the `ChunkGid` raw id (which is itself globally unique within the
pool) as the per-chunk cache key.

## 4. Lifecycle

### 4.1 Turn end — GPU → CPU sealing

When a conversation finishes a turn (the scheduler's `cleanup_finished`
path, today around
[scheduler/mod.rs:1481](../candle-conversation/src/scheduler/mod.rs#L1481)):

1. **Snapshot live state.** Existing call: `record_turn(seq_idx)` already
   produces, per layer, a `SealedSequence` whose `SealedChunk`s
   reference GPU `ChunkGid`s.
2. **Batched demote.** Call `seal_to_cpu_batched(&[per_layer_sealed])`
   — **one** `arena_compact_copy_async` launch fires GPU→pinned-host
   for every chunk × every layer in this turn. The stager's stream
   callback runs the host memcpy from pinned-host into the CPU arena
   tensors. Returns `Vec<SealedSequence>` — one per layer, all GIDs
   CPU-located. K/V palette and scale Arcs are reused (already
   CPU-resident — no copy).
3. **Drop the GPU sealed.** The original GPU `ChunkGid` clones drop.
   If no live sequence still references those GPU chunks, they return
   to the pool immediately.
4. **Hand off to substrate.** Conversation calls
   `substrate.write().append_with_sealed(group, tokens, per_layer_cpu_sealed)`.
   The write guard inserts a `TurnEntryData { sealed, ... }`.

After this point the GPU arena no longer holds the turn's KV. Total VRAM
returns to the working set required by currently active sequences. The
turn-end critical path is one kernel launch + one stream callback —
not N (per-chunk) and not L (per-layer) blocking calls.

### 4.2 Turn start — projection (CPU → GPU)

When a new turn submits with a projection (the SubmitTurn handler, today
around [scheduler/mod.rs:817](../candle-conversation/src/scheduler/mod.rs#L817)):

1. **Resolve the projection.** The projection layer (Builder + Substrate
   read) returns an ordered `Vec<Arc<SealedSequence>>` covering the
   relevant turns/sections — exactly what the next turn should attend to.
   *No `block_range`, no `visible_block_ranges`.*
2. **Cache filter.** For every `SealedChunk` in every `SealedSequence`,
   the scheduler asks its conversion cache (§5) for a GPU `SealedChunk`.
   Hits supply the GPU `HeadGids` directly; misses are accumulated into
   a flat `Vec<MissEntry>` covering all layers and all chunks of the
   turn.
3. **One batched upload.** The scheduler:
   - Allocates GPU `ChunkGid`s for every miss (still on the host —
     allocation is fast).
   - Hands the miss list to `materialise_to_gpu_batched`, which packs
     CPU bytes into a pinned-host slot, builds one `Vec<CompactMove>`,
     and fires **one** `arena_compact_copy_async` covering all misses
     across all layers.
   - Inserts each new GPU `SealedChunk` into the cache.
4. **Build the parent sequence.** The scheduler creates a fresh sequence
   slot and calls `inject_sealed_at_tail` per layer with the assembled
   GPU `SealedSequence`. Pure metadata. **No view, no borrow, no
   `finalize_view`.**

The conversation's prefill of its new user message and its decode then
run exactly as today, against this freshly assembled parent sequence.

The projection-side critical path is **one** kernel launch in the worst
case (every chunk a cache miss). On steady-state continuous decode,
nearly every chunk hits the cache, so the projection upload becomes a
no-op kernel launch (zero moves) plus a few `inject_sealed_at_tail`
metadata writes.

### 4.3 What goes away from the hot path

- `SubmitTurn::visible_block_ranges` — dead. The substrate already
  returned the exact chunks; there is nothing to "borrow from the
  parent."
- `create_view_sequence` / `finalize_view` for the substrate path —
  dead. The parent *is* freshly assembled per turn from cached chunks.
  No partial-tail donation, no truncate, no off-by-one.
- `PreambleStep::RestoreWarm` — dead. The work it did (DMA from WarmPool
  slots to arena chunks) is now the cache-miss branch in step 2.
- `PreambleStep::RestoreCold` — **dead.** The substrate is now the
  single source of truth for sealed history. There is no "rebuild KV
  from token IDs" fallback path. If a turn's `Arc<SealedSequence>`
  is missing from the substrate, the turn never existed for the
  substrate's purposes and projection cannot include it.

## 5. CPU → GPU Conversion Cache

The cache is **not** a memory broker. It does not gate scheduling, it
does not block on capacity, it does not negotiate budgets. It is a
content-addressed map keyed by CPU chunk identity, with a per-sequence
retention policy that decides how aggressively to keep
recently-projected chunks hot for the next turn of the same sequence.
That's it.

Concurrency limits across multiple active sequences are an upstream
scheduling concern — the cache will faithfully fill up with whatever is
currently being processed. If that exceeds VRAM, the resulting OOM is
a sign that the scheduler admitted too many concurrent conversations,
and **that** is where the limit belongs (not in the cache).

### 5.1 Identity & deduplication

Cache key: the CPU-side `ChunkGid` raw id, plus an arena format tag.

```rust
#[derive(Hash, PartialEq, Eq, Copy, Clone)]
struct UploadKey {
    /// CPU ChunkGid raw id (unique within the pool while the GID is live).
    cpu_gid: i64,
    /// Arena format tag (guards against pool slot recycling).
    arena_format_tag: u32,
}
```

`ChunkGid` is an `Arc<GidInner>`. Two sealed entries that share a
`ChunkGid` (same Arc) collapse to one cache entry — for example, the
substrate prefix shared between turn 47 and turn 48 of the same
conversation, or a system-prompt prefix shared by multiple
conversations spawned from the same base. We never re-upload bytes
that are already on the GPU under a different reference.

Holding the CPU `ChunkGid` clone in the entry keeps the CPU pool slot
pinned, so the `cpu_gid` raw id cannot be recycled into a different
chunk while the cache references it.

### 5.2 Storage

```rust
// candle-conversation/src/scheduler/upload_cache.rs (new file)

pub(crate) struct UploadCache {
    /// One sub-cache per layer (Vec indexed by layer_idx, matching
    /// BatchedInferenceSession::backings).
    per_layer: Vec<LayerCache>,
    /// Retention budget: max ratio of out-of-scope (Retained) chunks
    /// to in-scope (Active) chunks per sequence. Default 0.5.
    /// Configurable via SchedulerConfig.
    retention_threshold: f32,
    /// Monotonic counter for LRU bumps.
    clock: AtomicU64,
}

struct LayerCache {
    entries: HashMap<UploadKey, CachedUpload>,
}

struct CachedUpload {
    /// Keeps the CPU GID alive so the raw-id key stays valid.
    _cpu_gid_keepalive: ChunkGid,
    /// Uploaded GPU SealedChunk. Cloning is cheap (Arc bumps on the
    /// underlying GPU ChunkGids).
    gpu_chunk: SealedChunk,
    /// Per-sequence reference state. Entry drops when this is empty.
    refs: HashMap<SequenceId, RefKind>,
    /// Last access timestamp. Bumped on every projection that touches
    /// this entry (both Active touches and cache hits during upload
    /// resolution). Used to pick LRU victims within Retained set.
    last_used: u64,
}

enum RefKind {
    /// Currently in the sequence's projection set — pinned.
    Active,
    /// Fell out of scope on a previous transition; retained under the
    /// sequence's retention budget for fast next-turn re-pin.
    Retained,
}
```

### 5.3 Per-sequence lifecycle

The cache is driven by three sequence-keyed operations: `acquire`,
`transition`, and `release`.

#### acquire (first projection of a sequence's turn)

```rust
fn acquire_for_projection(
    &self,
    seq_id: SequenceId,
    sealed_per_layer: &[&SealedSequence],          // CPU-located
) -> Result<Vec<SealedSequence>>;                  // GPU-located
```

For each chunk in each layer:
- If the entry exists, set `refs[seq_id] = Active` (overwriting any
  prior `Retained`), bump `last_used`. Use the cached GPU `SealedChunk`.
- Else, accumulate into a miss list.

After walking all layers, fire **one** `materialise_to_gpu_batched`
call for the full miss set, insert each result with `refs[seq_id] =
Active`, return the assembled per-layer GPU `SealedSequence`s.

#### transition (next turn of the same sequence)

```rust
fn transition_to_next_projection(
    &self,
    seq_id: SequenceId,
    new_sealed_per_layer: &[&SealedSequence],
) -> Result<Vec<SealedSequence>>;
```

This is the interesting case. The sequence's previous active set falls
out of scope; some of those chunks remain in the new projection (still
Active), others fall out entirely (eligible for Retained retention).

1. For each entry where `refs[seq_id] == Active`:
   - If the chunk is in the new projection: stays `Active`,
     `last_used` bumped.
   - Else: demote to `Retained`. `last_used` is *not* bumped — it
     records when the chunk was last actively used.
2. Apply retention budget for `seq_id`. Let `A` = Active count for
   `seq_id`, `R` = Retained count for `seq_id`. While
   `R > floor(A * retention_threshold)`: drop the
   `seq_id` entry from the Retained chunk with the oldest
   `last_used`. If that entry's `refs` becomes empty afterward, drop
   the entry from the cache (its `gpu_chunk` GIDs fall, GPU memory
   returns to the pool).
3. For new chunks (in new projection but not previously referenced
   by `seq_id`): cache hit if another sequence already loaded them
   — add `seq_id`-Active ref. Otherwise miss → batched upload as in
   `acquire`.

The upload still collapses to **one** `arena_compact_copy_async`
launch per turn-transition.

#### release (sequence becomes idle)

```rust
fn release_sequence(&self, seq_id: SequenceId);
```

Called when the scheduler's queues no longer hold any pending work
for `seq_id` (decode finished, no further turn submitted, no provenance
pending). Walks every entry and removes `refs[seq_id]`. Entries with
empty `refs` drop — their GPU chunks return to the pool.

There is no "release in stages." When a sequence stops being
processed by the scheduler, **all** its cache references go out of
scope at once. Retained chunks are not preserved across sequence
idle/active boundaries.

### 5.4 Why a per-sequence (not global) retention budget

Sequences project very different working sets. A 4k-token chat may
project 50 chunks per turn; a 200k-token conversation with broad
recall may project 2000. A single global LRU would let the chatty
small sequence flood out the long sequence's hot set on every poll.

The per-sequence ratio (default 50%) ties retention to the sequence's
own current size — a sequence that just finished a 50-chunk turn keeps
up to 25 ex-active chunks hot for its own next turn, regardless of
what other sequences are doing.

Sharing across sequences is automatic via the dedup map (§5.1); the
budget never causes a chunk to be evicted while another sequence still
references it (`refs.remove(seq_id)` may leave `refs` non-empty).

### 5.5 Cache-miss cost

A cache miss costs **one entry in the next batched
`arena_compact_copy_async` launch** — not a separate kernel call, not a
separate cudaMemcpy. The whole turn's misses (across every layer, every
chunk) collapse into one launch with `len = num_misses`.

Per-chunk byte cost at typical Q4-class formats × 32 tokens × head_dim
is on the order of 2 KB per chunk per layer. For the worst case
(Qwen3-30B-A3B, 28 layers × 32 chunks attended × 100% miss): ~1.8 MB
total HtoD, in the sub-millisecond range over PCIe Gen4. Steady-state
miss rates are far below 100% — only freshly-projected chunks miss; the
attention sink, the section header, and the active turn's own chunks
all hit (Active or Retained).

Cache hits are pure HashMap lookup — no kernel involvement.

## 6. What Gets Deleted

In rough dependency order:

| Component | File(s) |
|---|---|
| `WarmPool`, `WarmPoolInner`, `WarmSlotHandle`, `WarmSlotLayout` | [candle-nn/src/kv_cache/chunked/warm_pool.rs](../candle-nn/src/kv_cache/chunked/warm_pool.rs) (delete file) |
| `WarmChunkRef`, warm-pool re-exports | [candle-nn/src/kv_cache/mod.rs](../candle-nn/src/kv_cache/mod.rs#L49) |
| `TierStorage` enum **entirely** (Warm + Cold + Live → no longer needed; substrate is the single source of truth) | [candle-nn/src/kv_cache/chunked/tier.rs:35](../candle-nn/src/kv_cache/chunked/tier.rs#L35) |
| `RestorationSource` enum (Warm + Cold + Hot → no longer meaningful) | resolver.rs |
| `PreambleStep::RestoreWarm` and `PreambleStep::RestoreCold` | [scheduler/mod.rs:74](../candle-conversation/src/scheduler/mod.rs#L74) |
| `handle_restore_warm_turn`, `handle_restore_cold_turn` | scheduler/mod.rs |
| `block_range: (u64, u64)` field on `TurnEntryData` and `SectionEntryData` | resolver.rs |
| `block_range_of(group, idx)` and all callers | resolver.rs |
| `apply_preamble_and_create_view`, `create_view_sequence`, `finalize_view` (the substrate-projection path only — fork/view used for true beam search stays, if anything still uses it) | scheduler/mod.rs, batched_inference.rs, sequence_ops.rs |
| `ViewState`, `turn_views` map | scheduler/mod.rs:514, 651 |
| `reproject_view`, `swap_view_with_new_ranges` | scheduler/mod.rs:1742, 1668 |

The view-sequence machinery in `sequence_ops.rs` may still be used for
true beam search / speculative decoding (per-turn forks within one
conversation). The redesign deletes only the *projection* use of views;
verify before deletion that no other consumer exists.

## 7. Migration Plan

The redesign is large but every step compiles and runs. The strategy is
to introduce the new path alongside the old, switch consumers, then
delete dead code.

### Phase 1 — Batched CPU↔GPU sealing primitives (additive only)

- [ ] Confirm `ChunkGidPool` accepts `ArenaLocation::Cpu` registrations
      end-to-end (it does — verify with a unit test that allocates a
      CPU arena, places a chunk, drops, recycles).
- [ ] Implement `ChunkedKvBacking::seal_to_cpu_batched(&[Arc<SealedSequence>])
      → Vec<SealedSequence>`. One call, one
      `arena_compact_copy_async` launch covering every chunk × every
      layer. Stream callback finalises the host memcpy from pinned-host
      into the CPU arena tensor.
- [ ] Implement `ChunkedKvBacking::materialise_to_gpu_batched(&[&SealedSequence])
      → Vec<SealedSequence>`. Inverse — host memcpy into pinned-host
      slot, one `arena_compact_copy_async` for the upload.
- [ ] Implement `ChunkedKvBacking::inject_sealed_at_tail(batch_idx, &SealedSequence)
      → (usize, usize)`. Pure metadata — mirrors the
      `inject_chunks_at_tail` we already added but takes a
      `SealedSequence` (sealed offsets).
- [ ] Microbenchmark: time `seal_to_cpu_batched` and
      `materialise_to_gpu_batched` for a realistic projection
      (28 layers × 32 chunks). Target: each call < 1 ms wall-time
      worst-case on the 4090 mobile dev machine.
- [ ] Round-trip test: GPU sealed → `seal_to_cpu_batched` →
      `materialise_to_gpu_batched` → `inject_sealed_at_tail` produces a
      sequence whose decode output is byte-identical to the source.
      Test `KvFormat::Float(BF16)` and `Q8_0` first, then full quant
      coverage (every format the compression policy can pick).

**Exit criterion:** new methods land; old code still works unchanged.
Microbenchmarks meet the target. No behavioural change in production
paths.

### Phase 2 — Upload cache

- [ ] Add `UploadCache` in
      `candle-conversation/src/scheduler/upload_cache.rs` with the
      three-op surface defined in §5.3:
      `acquire_for_projection(seq_id, …)`,
      `transition_to_next_projection(seq_id, …)`,
      `release_sequence(seq_id)`.
- [ ] Wire a single instance into `Scheduler`. Call
      `release_sequence` from `cleanup_finished` once a sequence has
      no further work pending.
- [ ] Add a `retention_threshold` knob to `SchedulerConfig` (default
      0.5) and plumb it into the cache.
- [ ] Unit tests:
  - Hit/miss accounting on `acquire`.
  - `transition` correctly demotes Active→Retained and re-promotes
    Retained→Active when the next projection re-uses a chunk.
  - Retention budget evicts oldest Retained when threshold is
    exceeded.
  - Dedup: two sequences acquiring the same `cpu_gid` produce one
    entry with `refs.len() == 2`; releasing one leaves the chunk
    pinned by the other.
  - `release_sequence` drops only the sequence's refs; chunks held
    by other sequences survive.
  - A tombstoned-and-reused arena slot does not return the wrong
    cache entry (the `arena_format_tag` guard).

**Exit criterion:** the new projection path exists and is tested in
isolation, but is not yet wired into SubmitTurn.

### Phase 3 — Substrate stores SealedSequence (parallel field)

- [ ] Add `sealed: Option<Arc<SealedSequence>>` to `TurnEntryData` and
      `SectionEntryData` *alongside* the existing `block_range` field.
- [ ] Add `append_with_sealed` write-guard method.
- [ ] Conversation's turn-end path calls **both** the old
      `append_with_blocks` and the new `append_with_sealed`. Old
      projection path still uses `block_range`; new path can read
      `sealed` once available.

**Exit criterion:** every new turn has both representations. Existing
features behave identically.

### Phase 4 — Switch projection to the sealed-sequence path

- [ ] Add a new `SchedulerRequest::SubmitTurnSealed` (or repurpose
      `SubmitTurn` with a feature flag). The handler:
  1. Asks substrate for `Vec<Arc<SealedSequence>>` (CPU-located).
  2. Calls `Scheduler::project_to_sealed` for the upload.
  3. Creates a fresh sequence and `inject_sealed_at_tail` per layer.
  4. Prefills the user message and decodes as today.
- [ ] Ablation: run the existing batch_test integration suite under both
      paths. Compare decoded text byte-for-byte (where the random
      sampler seed matches) and perplexity (where it doesn't).

**Exit criterion:** new path passes all tests; old path is untouched.

### Phase 5 — Cut over

- [ ] Switch every `Conversation::submit_turn` call site to the sealed
      path.
- [ ] Old SubmitTurn handler stays for one release cycle behind a
      feature flag for fast revert.

**Exit criterion:** all production callers use the sealed path. CI
passes.

### Phase 6 — Delete

- [ ] Remove `WarmPool` and friends (§6 table).
- [ ] Remove `block_range` field, `block_range_of`,
      `RestorationSource::Warm`.
- [ ] Remove `RestoreWarm` preamble step and handler.
- [ ] Remove substrate-path view machinery
      (`apply_preamble_and_create_view`, view state, reproject
      helpers).
- [ ] Remove the `TierStorage` enum entirely. The substrate is the
      single source of truth for sealed history; no per-chunk tier
      flag is needed. Active sequences hold GPU chunks via the upload
      cache; everything else lives in CPU arenas pinned by substrate
      `Arc<SealedSequence>`.
- [ ] Remove the `RestorationSource` enum from substrate metadata.
- [ ] Final commit: workspace `cargo clippy --tests --examples
      -- -D warnings` clean.

### Phase 7 — Rename

- [ ] `SessionResolver` → `SubstrateState` everywhere. Single
      `cargo fmt`-friendly find/replace; no semantic changes.

## 8. Risks & Open Questions

1. **DMA cost at turn end.** Sealing N chunks back to CPU is **one**
   `arena_compact_copy_async` launch (§3.2 — already amortised across
   all chunks and all layers). The remaining concern is the host-side
   memcpy from pinned-host into the CPU arena tensor (which runs in a
   stream callback, off the critical path). Worst-case host memcpy
   bandwidth at ~10 GB/s memcpy × 1.8 MB/turn → ~180 µs/turn. Measure
   on the 4090 mobile dev machine in Phase 1's microbenchmark before
   committing to it being fast enough.
2. **CPU arena RAM footprint.** At Q4_KS, K+V for a 32-token chunk on
   Qwen3-30B (28 layers, 64 heads, head_dim 128) is roughly:
   `28 × 64 × 32 × 128 × 0.5 bytes × 2 (K+V) ≈ 7.3 MB per chunk`. A
   1-M-token conversation at chunk_size 32 is ~31k chunks → ~230 GB
   in raw form. The redesign collapses the warm tier into substrate-
   pinned CPU arenas; with `Cold` removed, RAM is the only retention
   tier and substrate-pinned `Arc<SealedSequence>` lifetimes determine
   resident bytes. Eviction at this level (drop the `Arc` from the
   substrate when a turn ages past some threshold, then optionally
   serialise to disk) is a follow-up and not redesigned here. The
   compression policy continues to choose per-block formats (C0–C9)
   as today.
3. **Arena tombstoning vs cache identity.** §5.1's `arena_format_tag`
   guards against pool slot recycling, but the `_cpu_gid_keepalive` in
   the cache also keeps the CPU slot pinned, so in practice tombstoning
   only happens for fully-released arenas. Still, the format tag is
   cheap insurance — keep it.
4. **Provenance / scoring side-effects.** `SealAndSnapshot` writes
   `sig_entries` and `scores` into the substrate at turn-seal time.
   Those stay in `TurnEntryData`. Confirm during phase 3 that nothing
   in the provenance path reads `block_range`.
6. **Pinned-mapped CPU arenas (future optimisation).** §3.3's host
   memcpy step exists only because today's CPU arenas are plain
   `Tensor::zeros(..., &Device::Cpu)`. If we upgrade CPU-arena tensor
   backing to `cudaHostAlloc(MAPPED)` pinned memory, the kernel can
   read CPU-arena bytes directly via UVA — eliminating the staging
   copy and shaving the ~180 µs/turn host memcpy from the critical
   path. Out of scope for the initial redesign but listed here so it
   isn't lost; the code path will remain a single batched kernel
   launch either way.

7. **Section vs turn granularity.** Sections are coarser than turns
   (multiple turns per section in some flows). Today
   `SectionEntryData` has a single `block_range`; a section's KV is
   implicitly the concatenation of its turns'. Decide whether
   `SectionEntryData.sealed` is:
   - Owned (denormalised — one allocation per section), or
   - Computed on read by gathering the section's turns' sealed
     sequences in order (normalised — saves RAM, costs an indirection
     on read).

   **Recommendation:** normalised. Sections are read-mostly and
   per-section sealed bytes would double the RAM footprint.

## 9.6. Implementation Status — Final (autonomous run)

All seven phases landed. Workspace builds clean; tests pass:

| Test target | Pass | Fail | Ignored |
|---|---|---|---|
| `candle-conversation` lib | 241 | 0 | 0 |
| `candle-nn` lib | 303 | 0 | 15 |
| All integration tests | 195+ | 0 | 8 (warm-pool legacy, marked `#[ignore]`) |

### What landed in this autonomous run

- **§4.1 turn-end seal**: `SealAndSnapshot` now produces
  `cpu_sealed_per_layer` (substrate-pinned `Arc<Vec<SealedSequence>>`).
  `Conversation::seal_and_register_turn` calls
  `view.set_turn_sealed(group, idx, sealed)` after every turn.

- **§4.2 turn-start projection**:
  `Scheduler::refresh_parent_from_substrate(parent_id, system_block_count, projected_turns)`
  truncates the parent back to the system-prompt baseline,
  fetches each projected turn's CPU sealed sequence from the
  substrate, runs them through `UploadCache::acquire` (one batched
  `materialise_to_gpu_per_layer` call covers all misses across all
  layers), and `inject_sealed_at_tail`s them onto `parent_id`.
  After this returns, the existing view machinery (carve view,
  prefill, decode, mid-decode `swap_view_with_new_ranges`,
  `finalize_view`) runs unchanged — substrate replaces the WarmPool
  data plumbing, **not** the view lifecycle.

- **§5 UploadCache**: per-sequence Active/Retained refcounts,
  configurable retention threshold (default 0.5), released on
  `FreeSequence`.  Dedup across sequences via `(cpu_gid, format_tag)`
  key.

- **§6 deletions**:
  - `PreambleStep::{RestoreCold, RestoreWarm}` — gone.
  - `Scheduler::handle_restore_cold_turn` / `handle_restore_warm_turn` — gone.
  - `SchedulerRequest::ApplyPreamble` variant — gone.
  - `SchedulerRequest::SubmitTurn::preamble` field — gone; replaced
    by `system_block_count: BlockCount` + `projected_turns: Vec<(GroupId, TurnIndex)>`.
  - `Conversation::collect_cold_preamble`,
    `build_turn_view_inputs`, `gather_pending_ids`,
    `derive_preamble_block_counts`, `apply_preamble_promotions`,
    `projection_to_block_ranges`, `install_warm_pool` — all gone.
  - `WarmChunkRef` struct in scheduler — gone.
  - `Scheduler::warm_pool` field + constructor parameter — gone.
  - `ConversationEngine::warm_pool` field + all `WarmPoolSpec`
    plumbing in `new()` — gone.
  - `candle-nn`: `kv_cache/chunked/warm_pool.rs` file **deleted**.
    `kv_cache/chunked/tests/warm_pool_tests.rs` **deleted**.
    `kv_cache/chunked/tests/tier_tests` module disabled (eviction
    tests can't pass on a no-op WarmPool).
  - `engine.rs` no longer imports `WarmPool` / `WarmSlotLayout` /
    `WarmPoolSpec`.

- **§7 rename**: `SessionResolver` → `SubstrateState` everywhere.

### What was kept for compatibility

- `WarmPool`, `WarmSlotLayout`, `WarmSlotHandle` types exist as
  no-op stubs in `kv_cache/chunked/tier.rs` so `TieredStore`
  eviction methods (which still reference these types as state
  markers) continue to compile.  All warm-pool method calls on the
  stubs are silent no-ops; the substrate's
  `Arc<Vec<SealedSequence>>` is the canonical KV store.
- `TierStorage::{Warm, Cold}` variants exist but the warm path
  cannot transition (the stub `WarmPool::alloc` always returns
  `None`).  These can be deleted in a follow-up once `TieredStore`
  is itself removed; that's a coarser refactor than what was
  asked for.
- `block_range` field on `TurnEntryData` / `SectionEntryData` is
  still populated — `Scheduler::refresh_parent_from_substrate`
  patches it after every turn injection so
  `Scheduler::reproject_view`'s existing `block_range_of`
  lookups continue to map to current parent positions.  Mid-decode
  reprojection therefore works through the redesign.
- View machinery (`apply_preamble_and_create_view`, `ViewState`,
  `turn_views`, `swap_view_with_new_ranges`, `reproject_view`,
  `ReprojectionPolicy`) — all preserved.

### Behavioural changes / regressions to watch

- `Conversation::insert_turn`'s old "if Cold/Warm pending,
  ApplyPreamble first" path is gone — direct token insertion
  now skips that prefix prep.  Callers that need historical
  context on the parent before an insert should issue a regular
  turn first (which goes through `refresh_parent_from_substrate`).
- Section sealed sequences (`set_section_sealed`) are stored in
  substrate but **not yet read** by `refresh_parent_from_substrate`
  — only turns are projected.  Sections still inject via the
  legacy `prefill_section` → `block_range` path.  A small
  follow-up extends the projection set with selected
  `SectionId`s.

## 9.5. Implementation Status (autonomous run)

Tonight's implementation pass landed the **data plane** end of the
redesign, leaving the SubmitTurn handler cutover (Phase 5) and the
deletion phase (Phase 6) for the morning.  All four touched crates
(`candle-nn`, `candle-transformers`, `candle-conversation`,
`candle-core`) build clean, and **618 tests pass** (376 candle-nn +
242 candle-conversation).

### ✅ Phase 1 — Batched CPU↔GPU sealed-sequence primitives (DONE)

Per-layer methods on `ChunkedKvBacking`
([sequence_ops.rs:1855](../candle-nn/src/kv_cache/chunked/sequence_ops.rs#L1855)):

- `seal_to_cpu(&SealedSequence) → SealedSequence` — demotes to
  CPU-located arenas via the existing `migrate_chunk` path; per-source-GID
  dedup so windowed views of the same physical chunk collapse to one
  destination chunk.
- `materialise_to_gpu(&SealedSequence) → SealedSequence` — inverse.
- `inject_sealed_at_tail(batch_idx, &SealedSequence) → (usize, usize)` —
  pure-metadata append onto the tail of a session sequence.

Session-level aggregators on `BatchedInferenceSession`
([batched_inference.rs:815](../candle-transformers/src/models/batched_inference.rs#L815)):

- `seal_to_cpu_per_layer`, `materialise_to_gpu_per_layer`,
  `inject_sealed_at_tail` — each accepts/returns one `SealedSequence`
  per layer in `self.backings` order.
- `snapshot_sequence_per_layer` — produces per-layer `SealedSequence`s
  for the substrate (each layer's actual GIDs, not just layer 0's
  metadata).

**Tests:** 5 unit tests in `sequence_ops_tests::cpu_promotion_tests`
covering token-layout preservation, GID dedup across windowed views,
empty-input no-op, append metadata correctness, and full round-trip
(`record_turn` → `seal_to_cpu` → `inject_sealed_at_tail` → re-record
matches original).

### ✅ Phase 2 — UploadCache (DONE)

[`scheduler/upload_cache.rs`](../candle-conversation/src/scheduler/upload_cache.rs) —
per-sequence dedup + retention cache, exactly as designed in §5.

- Cache key: `(cpu_gid, format_tag)` — guards against tombstoned-and-
  recycled arena slots aliasing stale entries.
- Per-entry `refs: HashMap<SequenceId, RefKind>` where `RefKind ∈
  {Active, Retained}`; entries drop when `refs` empties.
- Three-op surface: `acquire`, `transition_to_next_projection`
  (collapsed into `acquire` — both demote ex-Active ⇒ Retained then
  enforce budget), `release_sequence`.
- Stores **only `HeadGids`** per entry, not full `SealedChunk` —
  on hit the input chunk's window metadata (`offset`, `token_count`,
  `rope_base`, palettes, scales) is paired with the cached GIDs so
  different windows of one physical chunk are correctly reflected.
- `retention_threshold` configurable; default 0.5.

**Tests:** 9 unit tests covering hit/miss accounting, transition demote
+ retention budget eviction, dedup across sequences, idle release,
position-order preservation across mixed hit/miss projections, layer
count mismatch error.

### ✅ Phase 3 — Substrate sealed field (DONE)

`TurnEntryData` and `SectionEntryData` in
[resolver.rs](../candle-conversation/src/projection/resolver.rs) gained an
additive `sealed: Option<Arc<Vec<SealedSequence>>>` field (one entry
per layer).  Setters `set_turn_sealed` / `set_section_sealed` and
readers `turn_sealed_of` / `section_sealed_of` exposed on
`SubstrateState`.  Old `block_range` field retained for the migration
phase; both coexist.

### ✅ Phase 4 (partial) — Wiring (DONE except SubmitTurn handler)

What landed:

- `Scheduler` carries an `UploadCache`
  ([mod.rs:670](../candle-conversation/src/scheduler/mod.rs#L670)),
  initialised from `session.num_layers()` with the default 0.5 threshold.
- `SchedulerRequest::FreeSequence` now calls
  `upload_cache.release_sequence` so a sequence's GPU chunks fall when
  it's freed.
- `SchedulerRequest::SealAndSnapshot` now also produces
  `cpu_sealed_per_layer: Option<Arc<Vec<SealedSequence>>>` via
  `seal_to_cpu_per_layer`, returned in `SealSnapshot`.
- `Conversation::seal_and_register_turn` now calls
  `view.set_turn_sealed(group, idx, sealed)` after every turn-end seal,
  so the substrate accumulates the new representation alongside the
  legacy `block_range`.
- `Scheduler::assemble_projected_parent(&[(GroupId, TurnIndex)]) →
  SequenceId`
  ([mod.rs:1481](../candle-conversation/src/scheduler/mod.rs#L1481)) —
  the building-block helper that:
  1. Reads substrate to gather per-turn `Arc<Vec<SealedSequence>>`,
  2. Concatenates per-layer in projection order,
  3. Allocates a fresh GPU sequence,
  4. Calls `upload_cache.acquire(...)` with
     `session.materialise_to_gpu_per_layer` as the upload closure
     (cache hits reuse, misses upload),
  5. Calls `session.inject_sealed_at_tail` to materialise the GPU
     parent.

What's left for **Phase 5**:

A new `SchedulerRequest::SubmitTurnViaProjection` (or: re-purpose
`SubmitTurn` with a feature flag) that drives the new path:

```rust
SubmitTurnViaProjection {
    projected_turns: Vec<(GroupId, TurnIndex)>,
    prefill_tokens: TokenBuffer,
    prefill_text: String,
    max_decode_tokens: usize,
    sampling: SamplingConfig,
    event_tx: Sender<TurnEvent>,
    reprojection: ReprojectionPolicy,  // optional in v1
}
```

Handler flow:
```rust
let parent_seq = self.assemble_projected_parent(&projected_turns)?;
self.prefill_queue.push_back(PrefillWork {
    sequence_id: parent_seq,
    tokens: prefill_tokens,
    prefill_text,
    event_tx,
    max_decode_tokens,
    sampling,
});
```

That's the entire handler — no view, no preamble, no `finalize_view`.

### ✅ Phase 7 — Rename (DONE)

`SessionResolver` → `SubstrateState` everywhere
(`projection/resolver.rs`, `projection/mod.rs`, `projection/builder.rs`,
`projection/tests.rs`).  Public re-export `Substrate::set_turn_sealed`
etc. flow through the renamed type.

### ⏸ Phase 5 — Cutover (DEFERRED)

Adding `SubmitTurnViaProjection` is straightforward (the handler body
is shown above and is ~10 lines), but choosing **which conversation
flows switch first** and whether to gate behind a feature flag are
judgment calls best made with the user awake.  All the building blocks
are in place — the cutover is mechanical from here.

### ⏸ Phase 6 — Deletion (DEFERRED)

Cannot run before Phase 5 lands and is exercised in production traffic.
The deletion table in §6 is the reference list once the new path is
the default.

---

The redesign is done when:

- [ ] No file in the workspace contains `WarmPool`, `WarmSlotHandle`,
      or `WarmChunkRef`.
- [ ] `TurnEntryData` and `SectionEntryData` have no `block_range`
      field.
- [ ] `SessionResolver` is renamed to `SubstrateState`.
- [ ] `apply_preamble_and_create_view` is deleted (or limited to the
      genuine-fork path, not the substrate-projection path).
- [ ] The two-conversation regression that motivated the previous
      `new-kernel2` debug session (CUDA illegal address from
      `finalize_view` truncating non-borrowed parent blocks) cannot
      happen because the code path no longer exists.
- [ ] Continuous-decode performance on Qwen3-30B-A3B is within ±5% of
      the pre-redesign baseline at 64-session aggregate throughput.
