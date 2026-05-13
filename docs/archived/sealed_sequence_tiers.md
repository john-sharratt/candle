# SealedSequence Storage Tiers

## Overview

A `SealedSequence` is an immutable snapshot of a completed KV-cache sequence.
Sequences pass through four storage tiers driven by two independent pressures:
**VRAM pressure** (demote toward the mmap pool) and **restoration demand**
(promote toward GPU for use in inference).

---

## Tiers

| Tier | Location | Format | Used in decode | Description |
|------|----------|--------|---------------|-------------|
| **Live** | VRAM | BF16 | yes | Full precision. Default output of active inference. |
| **Hot** | VRAM | Q8_0 | yes | ≈50% VRAM. Quantised in background as inference runs. Also the landing tier for restored prefix dependencies. |
| **Warm** | mmap file (NVMe) | Q8_0 | no (load to VRAM first) | KV data in a large mmap pool. OS pages in/out of RAM as needed. Chunks freed from VRAM pool. |
| **Cold** | in-memory | token IDs only | no (prefill first) | KV data gone. Token IDs live in memory at all times. KV rebuilt via prefill on demand. |

---

## The Warm mmap Pool

The Warm tier is backed by a single memory-mapped file on NVMe rather than
ad-hoc heap allocations. The OS manages which pages are resident in RAM at any
given time — the pool behaves as a large, transparent swap space for KV data.

### Pool sizing and backing selection

```
disk_pool_size = min(256 GB, free_disk_space − 16 GB)
ram_threshold  = min(available_ram, vram_kv_budget)

if disk_pool_size < ram_threshold:
    pool = anonymous mmap (RAM-only, no file)
    pool_size = ram_threshold
else:
    pool = file-backed mmap (NVMe)
    pool_size = disk_pool_size
```

If the available disk headroom is too small to beat what RAM alone can provide,
there is no benefit to using disk — the pool is created as an anonymous
in-memory mapping instead. This handles machines with nearly-full disks, small
SSDs, or situations where the NVMe is slow enough that a RAM-only pool is
strictly better.

The RAM-only pool is created with `mmap(MAP_ANONYMOUS | MAP_PRIVATE)` (Linux)
or `CreateFileMapping(INVALID_HANDLE_VALUE, ...)` (Windows) — same API surface
as the file-backed case, just without a file descriptor. The rest of the pool
logic (slot free-list, demotion, eviction timestamps) is identical in both
cases.

The 16 GB disk headroom is reserved for the OS, model weights, and other
process data. On a machine with less than 17 GB free disk, `disk_pool_size`
will be negative or zero and the RAM fallback activates automatically.

### File lifecycle

The mmap file is created, then **immediately unlinked** (deleted from the
directory). The file descriptor remains open and the OS keeps the data on disk
until the last reference is closed. When the process exits or crashes, the OS
releases the disk space automatically. No cleanup step is required and no stale
files are left behind on an unclean shutdown.

The file is sized with `fallocate(FALLOC_FL_KEEP_SIZE)` (Linux) or
`SetFileValidData` / a sparse file (Windows) — the full size is reserved in the
filesystem without writing any bytes. This makes creation instantaneous
regardless of pool size. Pages are only written to NVMe when actual KV data is
first stored in a slot.

### Pool layout

The mmap file is a flat byte array. Each slot holds the full KV data for **one
physical chunk across all transformer layers** — K and V tensors for every
layer are packed contiguously, so a single slot index is the only pointer
needed to locate or free all the data for one chunk.

Slot layout (all layers, K then V per layer, Q8_0):

```
slot[i]:
  [layer_0_K: n_kv_heads × chunk_size × head_dim × q8_0_bytes]
  [layer_0_V: n_kv_heads × chunk_size × head_dim × q8_0_bytes]
  [layer_1_K: ...]
  [layer_1_V: ...]
  ...
  [layer_N_K: ...]
  [layer_N_V: ...]

layer_kv_bytes = n_kv_heads × chunk_size × head_dim × q8_0_bytes_per_element
layer_stride   = 2 × layer_kv_bytes
slot_size      = n_layers × layer_stride
slot_base(i)   = i × slot_size
K_offset(i, layer) = slot_base(i) + layer × layer_stride
V_offset(i, layer) = slot_base(i) + layer × layer_stride + layer_kv_bytes
```

This offset arithmetic is encapsulated in a `WarmSlotLayout` struct with the
geometry baked in at construction time. Every byte range is computed from slot
index and layer index alone — there is no per-entry metadata in the file.
Since the file is always created fresh, the layout can change between versions
without any migration concern.

**Unit tests are required** for `WarmSlotLayout` covering: offset correctness
for layer 0, last layer, and mid-layer; no overlap between adjacent slots; no
overlap between K and V within a slot; and total file size calculation.

A free-list of slot indices tracks available slots. When a sequence is demoted
from Hot to Warm, its Q8_0 bytes are written into one slot per physical chunk
and the VRAM chunk slots are returned to the VRAM pool. The `SealedSequence`
records its list of warm slot indices.

---

## Normal Lifecycle (Demotion)

```
  Active inference
        │
        ▼
  ┌──────┐  background ┌──────┐ VRAM pressure ┌──────┐ mmap pressure   ┌──────┐
  │ Live │ ─quantise──►│ Hot  │ ────────────► │ Warm │ ──────────────► │ Cold │
  │(BF16)│             │(Q8_0)│               │(mmap)│ (drop KV data,  │(toks)│
  └──────┘             └──────┘               └──────┘ keep token IDs) └──────┘
```

**Live → Hot**: background quantisation kernel runs on the KV chunks in VRAM.
No data movement. By the time a sequence is no longer the active decode target
it is already Q8_0.

**Hot → Warm**: Q8_0 chunk data is written into the mmap pool. The VRAM chunk
slot is freed and returned to the chunk pool. The sequence now holds only mmap
slot references.

**Warm → Cold**: under mmap pool pressure, the KV bytes are evicted from the
mmap file and the slot is freed. The token IDs are already in memory — the
`ConversationStore` loads all turns into RAM at startup and they stay there for
the lifetime of the process. The sequence simply drops its mmap slot reference
and becomes Cold. No disk write, no encoding, nothing.

---

## Restoration (Promotion from Cold)

Restoring a Cold sequence requires its prefix dependencies to be present in
VRAM first. Dependencies are loaded from the Warm mmap pool directly into VRAM
chunks — no prefill needed for sequences that are still Warm.

```
Cold target sequence
        │
        ▼
  1. Walk prefix chain. For each prefix chunk:
       - If Warm: load mmap slot → VRAM chunk (Hot, ephemeral)
       - If Cold: prefill from token IDs (requires its own prefix first)
        │
        ▼
  2. All prefix chunks now in VRAM as ephemeral Hot chunks.
        │
        ▼
  3. Prefill target sequence token IDs → Live (BF16) in VRAM.
        │
        ▼
  4. Evict ephemeral prefix chunks immediately (first eviction targets).
```

- **Decode target** → restored to **Live** (BF16).
- **Prefix dependencies from Warm** → loaded as ephemeral **Hot** (Q8_0). No
  prefill required — the KV data is intact in the mmap pool.
- **Prefix dependencies from Cold** → must be recursively prefilled at **Hot**
  (Q8_0), tagged ephemeral, evicted once no longer needed.

If a prefix is Warm rather than Cold, restoration is much cheaper: a DMA copy
from the mmap pool replaces a full prefill pass.

---

## Inference Behaviour by Tier

| Tier | Attention kernel path |
|------|-----------------------|
| Live | BF16 paged attention, direct |
| Hot | Q8_0 dequant-on-the-fly |
| Warm | stall: load mmap slot → VRAM first |
| Cold | stall: full prefill required |

---

## Cold Tier

Cold sequences have no KV data anywhere — no VRAM chunk, no mmap slot. They
are just a pair of IDs pointing into the in-memory `ConversationStore`:

```
Cold = (ConversationId: u64, TurnId: u64)
```

The `ConversationStore` keeps every turn's text in memory for the lifetime of
the process (loaded from the binary store file at startup). Token IDs are
**not** stored persistently — they are derived at prefill time by re-tokenizing
the stored text through the model's tokenizer. This makes the store
tokenizer-agnostic: if the tokenizer is updated, old conversations are
automatically re-tokenized correctly on next use rather than restoring stale
transformed into stale integer IDs.

Cold restoration is a re-tokenize + GPU prefill — no disk access at restoration
time (beyond the initial startup load). No separate Cold storage format exists.

### Startup loading

At startup, `ConversationStore::read_all()` reads the binary store into memory.
For every turn loaded, a `Cold` `SealedSequence` is immediately created pointing
at that turn's `(ConversationId, TurnId)`. The tiered cache is therefore fully
populated with Cold entries on startup — no separate resume logic is needed.

When a conversation is next used, the normal promotion path handles everything:
the Cold entries are prefilled in turn order (with each turn's prefix restored
first as ephemeral Hot), and the conversation continues exactly as if it had
never been evicted. Full conversation resumption is a free consequence of the
tier system — it is not a separate code path.

### Turn sealing and the partial tail chunk

Each turn (user message or assistant response) seals the current KV sequence
when it completes. The last physical chunk of a turn will almost always be
partial — it contains `token_count % chunk_size` valid tokens, with the
remainder of the chunk unused. This is a normal `SealedChunk` with
`token_count < chunk_size`.

The *next* turn begins filling a **new** chunk from position 0. Turns do not
share physical chunks across their boundary; each turn produces an independent
set of sealed chunks with a possible partial tail. This simplifies restoration:
each turn can be prefilled independently in order without cross-turn chunk
boundary arithmetic.

### Quantization on seal

When a turn seals its `SealedSequence`, Live (BF16) chunks should be
quantized to Hot (Q8_0) immediately — the background quantization that runs
during active inference is triggered on seal rather than waiting for
pressure. This ensures sequences are already Q8_0 before any demotion
decision is made, keeping the demotion path (Hot → Warm) from racing with
quantization. The `no_quantize` flag on `SlotState` is respected — sequences
marked no-quantize (e.g. system prompts) are sealed at BF16 and stay Live.

### Token store vs chunk granularity

The `ConversationStore` stores turns at **turn granularity** (one text string
per turn). The KV cache operates at **chunk granularity** (typically 32–64
tokens per physical block). These do not need to match.

At Cold restoration time, the turn text is re-tokenized and the scheduler
assigns chunks via the same `div_ceil(chunk_size)` path used for any prefill.
No chunk boundary metadata needs to be persisted.

### Conversation deletion and RAII cleanup

When a conversation is dropped (user deletes a chat), the `SealedSequence`
objects for all its turns are dropped with it. Each `SealedSequence` owns its
Warm mmap slot indices — `Drop` returns them to the pool free-list. VRAM
chunks are returned to the VRAM pool the same way. The `ConversationStore`
removes the turn records from its in-memory map, and the binary store file is
appended with a tombstone record (or compacted on restart).

No explicit cache scan or forced eviction is needed — ownership does the work.
Any Warm mmap slots that belonged to the deleted conversation are immediately
available for reuse. Any VRAM chunks still Live or Hot are returned to the
VRAM pool via the same Drop path.

---

## Key Constraints

- **Warm is mmap, not a Tensor**: Warm storage is a slot in a memory-mapped
  file. The OS manages RAM residency transparently. No candle `Tensor` tracking
  overhead. VRAM chunk pool is kept fully free for active inference.
- **Prefix ordering on restore**: a sequence cannot be prefilled until all of
  its prefix dependencies are in VRAM. Dependencies are resolved depth-first
  (earlier turns first) using the turn ordering already present in the binary
  store before the target is prefilled.
- **Ephemeral dependency chunks**: Hot chunks loaded as prefix dependencies
  (from either Warm or Cold) are tagged ephemeral and are the first eviction
  candidates under VRAM pressure. They do not pass through Warm on eviction —
  their KV data was either already in the mmap pool (Warm restore) or freshly
  computed (Cold prefill) and can be dropped immediately.
- **Warm preserves KV data, Cold does not**: the key difference between Warm and
  Cold is whether Q8_0 KV bytes exist anywhere. Warm → Live/Hot is a mmap load;
  Cold → Live/Hot requires a full prefill from in-memory token IDs.
- **System prompts are exempt from demotion**: the KV cache for system prompts
  always remains in VRAM at Live (BF16) precision. System prompt chunks are
  never quantised, offloaded to the Warm pool, or evicted to Cold. They are
  pinned residents and are excluded from all pressure-driven demotion logic.

---

## Eviction Policies

### Hot → Warm (VRAM eviction): score-based with tail protection

Each chunk carries an eviction score. Under VRAM pressure, the chunk with the
lowest score is demoted first. Session-pinned (system prompt) and ephemeral
dependency chunks are excluded from the candidate set before scoring.

```
score = last_access_time − (tail_penalty if chunk is within tail window)
```

**Tail window**: the last `T` chunks of any active session (default: last 4
chunks, roughly the last 2k tokens of active context) receive a large negative
penalty making them effectively un-evictable. These are the chunks most likely
to be needed in the next decode step.

The **unit of eviction is one `SealedSequence`** (one complete turn, all its
physical chunks together). Partial eviction of individual chunks within a turn
is not supported — a sequence's chunks are always used together for prefill, so
freeing half a sequence's chunks provides no benefit and complicates
restoration state.

**Eviction order**:
1. Ephemeral dependency sequences (tagged — always first, dropped directly to Cold)
2. Dead/closed sessions (no penalty, oldest-first within this set)
3. Sequences outside the tail window of idle sessions, scored by recency
4. Sequences outside the tail window of active sessions, scored by recency
5. Tail-window sequences are never voluntarily evicted (only evicted under
   extreme pressure if no other candidates exist)

Score promotion on every decode access is a single timestamp write per sequence —
O(1), no pointer chasing.

### Warm → Cold (mmap eviction): oldest by wall-clock time

When the mmap pool is full, the `SealedSequence` whose `demoted_at` timestamp
is oldest is evicted first — all its mmap slots freed at once. No scoring, no
frequency tracking: pure FIFO on demotion time.

This policy is appropriate here because:
- The pool is large (up to 256 GB) so eviction is rare and not on the hot path.
- The miss cost for Cold is high (full prefill), so a conservative policy that
  retains recently-demoted sequences and discards genuinely old ones is correct.
- Sequences that are accessed again will be promoted back to VRAM before they
  age to the front of the eviction queue.

Each `SealedSequence` records a single `u64` Unix timestamp at the moment it
was demoted to Warm. The eviction scan walks the sequence registry (not the
mmap data itself) to find the minimum timestamp — O(n_sequences), but the
sequence registry is tiny compared to the mmap data.
