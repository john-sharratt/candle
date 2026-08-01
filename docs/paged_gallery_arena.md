# Paged Gallery Arena — a resident VRAM store for the provenance scan

> **Status — BUILT, TARGET MET (2026-07). Live behind the reproject path.**
> A purpose-built VRAM arena holds the wide-Q provenance gallery *resident on the
> GPU* between reprojections; each turn's folded signatures are a run of fixed-size
> group-major pages from a free-list; a re-seal re-uploads only the changed turn;
> a fingerprint-keyed per-scan index cache (§13.Q4 — built) skips the
> O(scanned-tokens) rebuild across a turn's reprojections; the arena is registered
> with the VRAM governor at a cheap relief rung so KV always wins. The scan runs
> on **two backends** sharing one float finalize (`bdp_vote.cuh`): the production
> **b1 tensor-core kernel** (`bdp_bmma.cu`, §14 — `BMMA.88128.XOR.POPC`,
> sm_75..sm_89) and the **scalar kernel** (`bdp_scan.cu`, nullable
> `page_ptr`/`pos_map`), which is both the fallback for devices without b1 BMMA
> (Hopper/Blackwell dropped it — the 2×5090 box runs scalar until an IMMA INT8
> variant exists) and the differential-testing oracle. **Measured on the real
> substrate** (788 files, 2548 exchanges, 591 481 tokens, 128 MB resident):
> selection overlap **1.000**, max-rel **6.5e-5** vs CPU, steady-state scan
> **~9.5 ms** — inside the 3–10 ms flat-scan target — at **~41×** over the CPU
> per-file scan (BMMA accumulate ~5.7 ms; the scalar fallback runs the same scan
> in ~20.6 ms with bit-identical votes); cold first scan ~170 ms. **Remaining,
> deferred:** empty-slab reclamation (§13.Q3). The rest of this document is the
> as-designed spec it was built to, grounded in the chunked-KV `GidPool`, the
> paged-KV pointer interface, and the VRAM governor's relief ladder, with
> file:line references throughout; §14 documents the tensor-core backend.

---

## 1. Problem & target

The reproject belief scan is now on the GPU (`45f8617e`, segmented BDP, per-file
z), **8.9×** over the CPU per-file scan on the real substrate (788 files, 2548
exchanges, 591 481 gallery tokens). But the design target for the flat provenance
scan is **3–10 ms** (`docs/attention_provenance.md` §9.5: "six INT8 matrix
multiplies … 3–10 ms on CPU"; the same figure is the GPU-path budget), and we
measure **~40 ms**. The phase breakdown (measured with per-phase timers):

| phase | time | note |
|-------|------|------|
| **upload** | **~10 ms** | re-upload the **108 MB** gallery pinned→VRAM, *every scan* |
| kernel | ~25 ms | the BDP popcount (3.6 B agreements) — the tuned hot loop |
| download | ~1 ms | small result buffers |
| tally | ~3 ms | host per-segment needle gate (already parallelised) |

The upload is pure waste. The gallery is **append-mostly**: within a decode turn
*nothing seals*, so the gallery is byte-identical across every reprojection (the
resolver already exploits this with a resident *pinned-host* mirror keyed by an
`Arc::as_ptr` fingerprint — `resolver.rs:826-830`). Across turns, exactly **one**
turn grows by one window; the other ~591 k tokens are untouched. Re-uploading
108 MB on both axes is wrong.

**This document removes the upload** by keeping the token records resident in
VRAM in a purpose-built arena, uploading only a tiny per-scan index. It does
**not** touch the 25 ms kernel — see §13. Expected effect: **~40 ms → ~29 ms**,
and per-scan H2D from ~10 ms to **~0.4 ms**. Reaching 5–13 ms additionally
requires the kernel work tracked as a follow-on.

### 1.1 Why an arena, not just "keep the pinned buffer in VRAM"

A single resident VRAM buffer per group (the trivial fix) has two faults the
arena avoids:

1. **VRAM pressure.** 108 MB × (belief groups) held permanently competes with the
   model KV on the 16 GB card — the exact resource whose exhaustion caused the
   multi-turn admission wedge. The arena is governed: it yields VRAM to KV on
   demand (§9).
2. **Whole-buffer churn on every seal.** A monolithic buffer must be rebuilt and
   re-uploaded wholesale when any turn changes. Per-turn pages let a seal touch
   only the one changed turn (a **delta upload** of tens of KB), leaving 99.9% of
   the gallery resident and untouched (§6).

---

## 2. Design overview

```
                    per TURN, once (on first scan that needs it, or after re-seal)
  decoded_wide_sig(turn)                                  GalleryArena (VRAM)
  Arc<Vec<WideQSig>>  ──transpose→ group-major pages ──►  ┌──────────────────────┐
   (192 B/token, RAM)     (host)      (H2D delta)          │ 16 MiB slabs         │
        ▲                                                  │  page = 3×32×8 u64   │
        │ warm/cold tier ALREADY EXISTS:                   │  (6 KiB, group-major)│
        │  substrate wide_q_sigs blob + redo log           │  Treiber free-list   │
        │                                                  └──────────────────────┘
                                                                     │ page gids
   per SCAN (reproject), tiny:                                       ▼
     page_ptr[]   (u64 device addr per page, ~144 KB)   ───►  ┌──────────────┐
     pos_map[]    (u32 per scanned token, page<<5|in_pg) ───► │ bdp_scan.cu  │  z*margin
     case[]       (u32 per scanned token, exchange slot) ───► │ (paged)      │  per (tok,grp,seg)
     seg_tok/seg_case prefixes, probe                   ───►  └──────────────┘
```

- **The token records live in the arena, resident.** They are the only large
  thing, and they no longer move per scan.
- **Warm/cold is free.** The RAM copy is the substrate `wide_q_sigs` blob +
  `decoded_wide_sig` `Arc` memo (`substrate.rs:2646,1088`); the cold copy is the
  `WideQSig` redo-log record (`persistence/writer.rs:263`). The arena owns **only
  the hot VRAM tier** — an eviction just drops pages and re-materialises them from
  the `Arc` on the next scan. We build no new warm/cold path.
- **The per-scan input is tiny** (index arrays + probe, ~4.7 MB total ≈ 0.4 ms),
  so the "efficient input passing" the brief flags reduces to: keep the records
  resident, upload only the addressing. §8.
- **The kernel change is minimal** and matches the paged-KV precedent: one page
  lookup replaces the flat `g_base + j*gw` addressing; the popcount hot loop is
  byte-for-byte unchanged. §7.

---

## 3. Prior art in this codebase (what we reuse)

We invent almost nothing. Three existing subsystems supply the pattern.

### 3.1 The chunked-KV slab allocator — `candle-nn/src/kv_cache/chunked/gid_pool.rs`

A **lock-free, refcounted slab allocator** over fixed 16 MiB arenas. We model the
gallery pool directly on it:

- **`CHUNK_SIZE = 32`, `TARGET_ARENA_BYTES = 16 MiB`** (`chunked/types.rs:17,20`).
- **`ArenaRefcounts`** (`gid_pool.rs:77`): per-arena `counts: Vec<AtomicU16>`
  overlapped as *either* a refcount (occupied) *or* an intrusive free-list link
  (free), disambiguated by an `occupancy` bit vector; a `recycle_head` Treiber
  stack; an `hwm` high-water mark for never-used slots. Allocation is **O(1)**
  (`try_claim_one`, `:257`): pop the recycle stack, else bump `hwm`. Free is
  lock-free RAII (`dec`, `:328`) on the last handle drop.
- **`ChunkGid`** (`gid_pool.rs:435`): an RAII handle, `id = arena_idx * stride +
  chunk_idx`; `Clone` bumps a refcount, `Drop` recycles the slot. `arena_idx()`
  /`chunk_idx()` decode it (`:493,499`).
- **`ChunkGidPool`** (`gid_pool.rs:1067`): the shared façade —
  `allocate_for(key)`, `allocate_run_for(key, len)`, `register_arena(key)`,
  `next_tombstone(key)`, `force_release_arena(idx)`. A `CapacityBitmap`
  (`:583`) gives O(1) "which slab has room"; freed arena indices recycle FIFO.
- **`ResolvedArenaInfo`** (`arena_table.rs:631`): the per-arena snapshot `{
  base_ptr, chunk_byte_stride, … }` from which a `ChunkGid` resolves to a device
  address `base_ptr + chunk_idx * chunk_byte_stride`. This is exactly how a page
  gid becomes a kernel pointer.

The gallery arena is a **simplified** copy: one format (the 6 KiB group-major
page), one location (GPU), no palette, no K/V split, no quantization.

### 3.2 The paged-KV → CUDA interface — `candle-kernels/src/paged-decode/` + `candle-transformers/src/models/slot_state.rs`

The decisive precedent for the "tricky" hand-off. The codebase does **not** use a
vLLM-style `i32` block-index table multiplied by a stride in-kernel. It embeds
**pre-resolved absolute `u64` device pointers** in a serialized header stream and,
where the logical→physical map is sparse, a **`position_map`**:

- `SlotHeader` → `TokenSlice` → `KvHead.k_ptr[4]` are absolute chunk-base
  pointers (`paged-decode/slot_types.cuh:31-42`), computed **host-side** as
  `base_ptr + chunk_idx * chunk_byte_stride` (`slot_state.rs:147-154`).
- The **`position_map` (`u32[total_tokens]`, `(slice_idx<<16)|in_blk`)**
  translates a logical token position to `(page, offset)` for gapped/partial
  chunks (`slot_types.cuh:70-79`); `resolve_pos` reads one `u32`.
- Kernel arg lists are **spare**: paged-decode takes a single `headers_ptr`
  (`paged-decode/api.rs:11-25`); everything paged is reached through it.
- Tables are packed host-side into pinned buffers, uploaded, and the device base
  pointers patched back into the headers (`prefill_utils.rs:388-406`).
- **Dangling-pointer guard**: a freed arena leaves `base_ptr == 0`; the host
  refuses to launch and names the offending chunk (`prefill_utils.rs:136-167`).
  Any absolute-pointer design needs this.
- **Perf lesson**: the dominant cost was rebuilding the pointer records host-side
  per forward, *not* the kernel indirection — build the records **once, resident**
  and point at them (`slot_state.rs:362-367`).

Our gallery scan is *strictly simpler* than paged-KV: it reads **all** tokens of
a segment (no sparse logical remap for reads), so page addressing is a plain
divide, and the per-scan tables are minuscule next to KV's per-forward rebuild.

### 3.3 The VRAM governor — `candle-core/src/vram/{mod,budget,relief}.rs`

Capacity `C = min(frac·total, total − headroom_abs)` (`balloon.rs:30`), a descending
**relief ladder** (`Trivial → Cheap → Moderate → Costly → Critical`, `mod.rs:84`),
and a registration API `register_relief(class, tier, relief_fn, evictable_fn)`
(`relief.rs:103`). KV eviction is `Costly`; we register the gallery at the
**cheapest** rungs so it is reclaimed *before* KV is ever touched (§9).

### 3.4 The signature data flow — `wide_sig.rs` + `substrate.rs`

- A **token** is one `WideQSig { n_heads, words: Vec<u64> }` (`wide_sig.rs:24`);
  the stored/gallery form is **folded**: 12 heads × 2 words = **1536 bits = 192
  bytes/token** (`fold_provenance`, `wide_sig.rs:111`).
- A **turn** is a variable-length window persisted as `wide_q_sigs` (blob =
  12-byte header + n_tokens × 192 B, `wide_sig.rs:158-166`), produced at seal
  (`gather_wide_sigs`, `scheduler/mod.rs:6297`), stored via
  `set_wide_q_sigs_blob` (`substrate.rs:3975`).
- **`decoded_wide_sig(stream_id) -> Arc<Vec<WideQSig>>`** (`substrate.rs:2646`) is
  a memo serving a **stable `Arc`** whose identity changes *only* when the blob is
  rewritten (`evict_decoded_wide_sig`, `:2677`, per-stream/incremental). This
  `Arc` identity is our residency key (§6).
- The **logical gallery** the arena must reproduce is Phase A of
  `score_belief_groups` (`resolver.rs:671-800`): per file, a `FileScan { arc_turn,
  ex_ranges, n_slots, arcs_kept, windows: Vec<(k, s, e, slot)> }` — seam-bounded
  sub-windows tagged by exchange slot. **The arena changes only where the bytes
  live and how they are addressed; Phase A's windowing/exchange logic is
  untouched.**

---

## 4. Record & page layout

### 4.1 The record — one token

A gallery record is exactly the folded signature of one token: `wpt = n_groups ·
gw = 3 · 8 = 24` u64 = **192 bytes**. This is fixed by the locked fold geometry
(`PROV_FOLD_SIZES.len()=3`, `PROV_HEADS_PER_LAYER=4`, `HEADS_PER_GROUP=4`,
`gw = 8`).

### 4.2 The page — 32 tokens, group-major

**A page holds `PAGE_TOKENS = 32` tokens in group-major order:** `[group][token][word]`
= `3 · 32 · 8 = 768` u64 = **6 144 bytes**. `PAGE_TOKENS = 32` deliberately
mirrors `CHUNK_SIZE` so the arena machinery and the "one page = 32 tokens" mental
model line up with KV.

**Why group-major *within* the page** (not token-major) — this is the crux that
preserves memory coalescing:

- The kernel fixes a layer-group `g` per block and reads consecutive tokens' 8-word
  signatures for that group. Group-major places group `g`'s 32 tokens contiguously
  (`[g·256 .. g·256+256)`), so consecutive threads read consecutive 64-byte
  (8×u64) records → **coalesced**, identical to today's contiguous group-major
  buffer (`bdp_scan.cu:19-26` documents this as the measured-fastest layout).
- Token-major (`[token][group][word]`) would stride the same group by 192 B per
  token — the non-coalesced regression the current layout exists to avoid.

Coalescing breaks only at page boundaries (every 32 tokens), which is negligible.

The host transpose from a turn's token-major `Vec<WideQSig>` into group-major
pages is the same transpose `from_segments` does today (`gpu.rs:180-188`), but
**per turn** and **once** (when the turn becomes resident), not for the whole
corpus per scan.

### 4.3 A turn's allocation — a run of pages

A turn with `N` tokens owns **`ceil(N / 32)` pages** (the last page partial; its
unused tail slots are never addressed, so partial pages cost only VRAM, ~≤ 6 KiB
of slack per turn). The pages need not be contiguous — the free-list hands out
whatever slots are free, and the per-token page map (§7) resolves any layout.
This is the "allocate the range that represents the turn" step: a run of page
gids, held in the turn's residency record; on free they drop and recycle.

---

## 5. The `GalleryArena` allocator

A dedicated allocator modelled on `ChunkGidPool` but single-format. Proposed
surface (in `candle-conversation/src/provenance/gallery_arena/`):

```rust
/// One resident 6 KiB group-major page in VRAM. RAII: last drop recycles the slot.
pub struct PageGid { id: i64, pool: Arc<GalleryPoolInner> }   // id = slab_idx*STRIDE + page_idx
impl PageGid { pub fn slab_idx(&self)->usize; pub fn page_idx(&self)->usize; }

pub struct GalleryArena {
    pool:    GalleryGidPool,          // Treiber free-list over 16 MiB slabs (mirror gid_pool.rs)
    storage: RwLock<Slabs>,           // slab_idx -> device Tensor (arena_pages, PAGE_TOKENS, wpt) analog
    info:    RwLock<Vec<ResolvedSlab>>, // slab_idx -> { base_ptr, page_byte_stride=6144 }
    device:  Device,
}
```

`GalleryGidPool` reuses, verbatim in structure, the `ArenaRefcounts` design
(overlapped refcount/occupancy word, Treiber `recycle_head`, `hwm`,
`CapacityBitmap`) and the `register_arena`/`allocate_for`/`next_tombstone`
lifecycle (`gid_pool.rs:257,328,1067`). The public methods we need:

```rust
impl GalleryArena {
    fn alloc_pages(&self, n: usize) -> Option<Vec<PageGid>>;      // register a slab on exhaustion
    fn page_ptr(&self, gid: &PageGid) -> u64;                     // base_ptr + page_idx*6144
    fn upload_page(&self, gid: &PageGid, group_major: &[u64]);    // H2D one 6 KiB page (delta)
    fn release_empty_slabs(&self);                               // tombstone drained 16 MiB slabs
    fn resident_bytes(&self) -> u64;                             // for the governor's evictable()
}
```

**Why a dedicated pool, not a new `ArenaKey` in `ChunkGidPool`.** The KV pool is
partitioned by `KvFormat`/palette and its records are K/V pairs; threading a
"gallery" format through it entangles unrelated code (`GIDS_PER_HEAD`, palette
maps, `PerHeadTable`). A ~300-line dedicated pool that copies the *pattern* is
cleaner and independently testable. (Open question §13.Q1 revisits sharing the
raw slab allocator.)

**Growth / fragmentation / compaction** come for free from the mirrored design:
16 MiB slabs registered on demand (recycling tombstoned indices), lowest-first
packing, `release_empty_slabs` tombstoning drained slabs. Fragmentation is
tolerable here — pages are uniform 6 KiB and a scan reaches every page through
`page_ptr[]` regardless of physical scatter, so we likely **do not** need the
KV defrag/remap machinery in v1 (§13.Q3).

---

## 6. Per-turn residency

The arena's index maps a **turn** to its resident pages, keyed so a re-seal or
eviction is detected exactly:

```rust
struct ResidentTurn {
    fingerprint: u64,          // content key (see below)
    pages: Vec<PageGid>,       // ceil(N/32) pages, in token order
    n_tokens: usize,
    lru: u64,                  // eviction ordering
}
// keyed by StreamId (turn_stream_id(timeline, index))
residency: Mutex<HashMap<StreamId, ResidentTurn>>
```

**Residency key = the `decoded_wide_sig` `Arc` identity + a content sample.** The
memo serves a stable `Arc` that changes identity only on blob rewrite
(`substrate.rs:2677`); `Arc::as_ptr` is therefore a cheap content key. We fold in
a small content sample (first/last `u64`) as the ABA guard — the exact hardening
just shipped for the resolver's group cache (`resolver.rs:803-821`). This makes
re-seal detection precise: a rewritten turn gets a new `Arc` → fingerprint
mismatch → free old pages, re-upload.

**`ensure_resident(stream_id, arc: &Arc<Vec<WideQSig>>) -> &[PageGid]`:**
1. Compute the fingerprint. If the residency entry matches → reuse its pages.
2. Else (miss or stale): drop any stale pages (RAII recycle), `alloc_pages(ceil(N/32))`,
   transpose the turn's sigs into group-major pages host-side, `upload_page` each
   (a **delta H2D of `N·192` bytes** — tens of KB, not 108 MB), record.
3. Bump LRU.

**Lifecycle mapping** (all events already exist):
- **Create**: first scan that includes the turn calls `ensure_resident`.
- **Re-seal** (blob rewrite → `Arc` change): fingerprint miss → free + re-upload
  just that turn.
- **Drop** (timeline drop / `substrate.reset()` clears `wide_q_sigs` +
  `sig_cache`, `substrate.rs:3939-3948`): the turn's `Arc` is gone; the residency
  entry is evicted lazily (next scan won't reference it) or eagerly on a reset
  hook. Pages recycle.
- **Governor eviction** (§9): drop LRU turns' pages; rebuild on demand.

**Within a turn**, every reprojection finds all turns resident → zero uploads
except the tiny index (§8). **Across a seal**, only the sealed turn is re-uploaded.
This is the whole win.

---

## 7. The paged gallery kernel

The change to `bdp_scan.cu` is confined to **address resolution**; the popcount
hot loop, the shared-memory reductions, the z/margin math, the segment structure,
and the output layout are all unchanged.

### 7.1 New kernel inputs

Replace the single contiguous `gallery_words` + `g_base = gallery_words +
g·total_tokens·gw` (`bdp_scan.cu:119`) with:

- `const uint64_t* page_ptr` — `n_pages` absolute device addresses, one per
  resident page, in the scan's global page order. Built host-side from
  `GalleryArena::page_ptr(gid)` = `base_ptr + page_idx·6144` (the `k_ptr`
  precedent, `slot_state.rs:147`).
- `const uint32_t* pos_map` — `n_scanned_tokens` entries, `pos_map[j] = (page <<
  5) | in_pg`. Resolves scanned-token `j` to its resident page + offset. This is
  the `position_map` pattern (`slot_types.cuh:70-79`), simplified to a 5-bit
  offset (32 tokens/page).

`gallery_case` stays a flat `u32[n_scanned_tokens]` side array (tiny). The
segment prefixes `seg_tok_start`/`seg_case_start` stay exactly as today — they are
token-index ranges, orthogonal to physical paging.

### 7.2 The hot-loop address change

Today (`bdp_scan.cu:127-131`):
```c
for (int j = tok0 + tid; j < tok1; j += nthreads) {
    const uint64_t* tok = g_base + (size_t)j * gw;   // flat contiguous
    ...
}
```
Paged:
```c
// Cache this segment's page-base pointers in shared memory once per block.
for (int j = tok0 + tid; j < tok1; j += nthreads) {
    const uint32_t pm    = pos_map[j];
    const uint32_t page  = pm >> 5;
    const uint32_t in_pg = pm & 31u;
    const uint64_t* pg   = (const uint64_t*)page_ptr[page];   // shared-mem cached
    const uint64_t* tok  = pg + (size_t)g * (PAGE_TOKENS * gw) + in_pg * gw;  // group-major
    ...   // identical ulonglong4×2 coalesced load + popcount, unchanged
}
```
- One `u32` (`pos_map`) + one `u64` (`page_ptr`) read per token. Both are tiny
  arrays; `page_ptr` for the block's segment is cached in `__shared__` at block
  start (the KV kernels cache the block table across a slice's tiles,
  `int8_decode_kernel.cuh:242-255`), so the per-token cost is a shared read.
- The `g * PAGE_TOKENS * gw` term selects group `g`'s contiguous strip inside the
  page; `in_pg * gw` selects the token → the **coalesced `ulonglong4×2` load is
  unchanged**, coalescing preserved within the page.

### 7.3 FFI + launcher

`run_batched_bdp_scan` (`provenance/api.rs`) gains `page_ptr`, `pos_map`,
`n_pages` and drops `gallery_words`; `n_probe_tokens`, `n_groups`, `n_segments`,
`max_seg_cases`, `gw`, `wpt`, `seg_*`, `out_*` are unchanged. The grid, block
size, and dynamic-shared computation are unchanged (plus the small `page_ptr`
shared cache).

### 7.4 Dangling-page guard

Because `page_ptr` entries are absolute pointers, an evicted/freed turn must never
leave a stale address in a launched scan. The host builds `page_ptr` **only from
currently-resident pages** (it calls `ensure_resident` for every in-scope turn
first, §8), so a live scan cannot reference a freed page. We additionally assert
no entry is 0 before launch, mirroring `prefill_utils.rs:136-167`.

---

## 8. Efficient input passing (the "tricky" part)

The brief flags this as the hard bit. The resolution: **the large thing (records)
never moves; only the addressing is passed, and it is tiny.** Per scan, in Phase B
of `score_belief_groups`:

1. **Ensure residency.** For each in-scope turn (from Phase A's `arc_turn`/`arcs_kept`),
   `ensure_resident(stream_id, arc)`. Within a turn this is all hits (no upload);
   after a seal it uploads exactly the one changed turn's pages (delta).
2. **Assemble the per-scan index** from Phase A's `windows: Vec<(k, s, e, slot)>`
   — the seam-bounded sub-windows already computed today. Walk the windows in
   segment (file) order; for each scanned token emit:
   - `pos_map[j] = (global_page(turn, tok_in_turn) << 5) | (tok_in_turn & 31)`,
     where `global_page` indexes into `page_ptr[]` (a per-scan concatenation of
     the in-scope turns' resident page gids, resolved to addresses).
   - `case[j] = exchange slot` (the `slot` from the window tuple).
   - accumulate `seg_tok_start`/`seg_case_start` exactly as today.
3. **Upload the index** through the pinned stager (reuse `PinnedBuf`,
   `gpu.rs:30`): `page_ptr` (~144 KB), `pos_map` (~2.3 MB), `case` (~2.3 MB),
   `seg_*` (bytes), `probe` (KB). **Total ≈ 4.7 MB ≈ ~0.4 ms** vs 108 MB / 10 ms.
4. **Launch** the paged kernel (§7); download + parallel tally unchanged.

**Sizing** on the measured corpus (591 481 tokens, ~18 484 pages): `page_ptr` =
18 484 × 8 B ≈ 144 KB; `pos_map`/`case` ≈ 2.3 MB each. Even uploaded every scan
these are ~25× cheaper than the records. `pos_map` and `page_ptr` are stable
within a turn (turn set/order unchanged) and **may be cached** keyed by the same
per-group fingerprint the resolver already maintains — reducing per-scan upload to
`case` + `probe` — but v1 can rebuild them each scan; the win is already captured
by making the records resident.

**Host-cost discipline** (the KV lesson, `slot_state.rs:362-367`): the index
assembly is `O(scanned tokens)` of pointer/`u32` writes into preallocated pinned
buffers — no per-token heap allocation. The transpose-into-pages cost is paid once
per turn at `ensure_resident`, not per scan.

---

## 9. VRAM governor integration

The arena is a **low-priority, fully-reclaimable** citizen. Per the governor
research and `docs/vram_governor_design.md`, we do **not** add a new `AllocClass`
(that widens `COUNT`/`idx()`/`[_;COUNT]` arrays across the governor). Instead we
`register_relief` (`relief.rs:103`) at the **cheapest rungs** so the gallery is
shed before KV is ever touched at `Costly`:

```rust
governor.register_relief(
    AllocClass::Kv,                 // shares the KV variable region…
    Criticality::Cheap,             // …but relieved at a rung KV eviction never uses
    move |req| { let freed = arena.evict_lru(req.want); ReliefOutcome{ freed_est: freed } },
    move || arena.resident_bytes(), // evictable estimate for forecast/budget
);
```

- **Eviction is nearly free** because the warm/cold tier already exists: dropping
  a turn's pages loses nothing — the `Arc` window (RAM) and the `wide_q_sigs`
  blob + redo log (RAM/NVMe) remain, and the next scan re-materialises the turn
  via `ensure_resident`. So the gallery satisfies `Cheap`/`Moderate` relief (no
  data loss, no GPU sync) — strictly cheaper than KV's `Costly` hot→warm move.
- **The gate is the live free-VRAM measurement** (`available()`, `mod.rs:229`), so
  the model's `relieve_with` observes pressure and the descending ladder sheds the
  gallery first by construction.
- **`resident_bytes()`** feeds `evictable_estimate`/`forecast_units`
  (`relief.rs:136`, `managed.rs:81`) so concurrency sizing sees the reclaimable
  gallery.
- **Pins**: the turns referenced by the *current* scan are pinned for its duration
  (a scan-local `working_set` mirroring `working_set_pins`, `substrate.rs:187`) so
  relief never frees a page an in-flight launch reads. Allocation uses the
  `creation_pending` guard (`gid_pool.rs:145`) so a fresh slab can't be tombstoned
  mid-upload.

Net: on the 16 GB card the gallery lives only in the VRAM the model isn't using
and evaporates the instant KV needs it; on the 2×5090 box it simply stays
resident.

---

## 10. Correctness & parity

The kernel math is unchanged — only which bytes each token reads. The paged path
must remain **numerically equivalent to the CPU per-file scan up to fast-math ULP**
(the property just shipped, `resolver.rs:672-676`). Tests:

1. **Kernel parity (unit).** `paged == contiguous`: build a small gallery both as
   today's `from_segments` contiguous buffer and as arena pages, scan the same
   probe, assert bit-identical `(out_case, out_vote)`. (The two differ only in
   addressing, so this should be *exactly* equal, not just ULP-close.)
2. **Resolver parity (unit).** Extend `score_belief_groups_gpu_matches_cpu_and_caches`
   (`tests/turn_belief_scan.rs`) to run the arena path and assert CPU == arena
   scores, and that a second scan is an all-resident hit (no upload).
3. **Residency lifecycle (unit).** `ensure_resident` re-uploads exactly one turn
   after a simulated re-seal (new `Arc`); freed pages recycle; a partial last page
   scans correctly (N not a multiple of 32).
4. **Real-substrate (example).** Extend `examples/gpu_belief_parity.rs` to the
   arena path: assert top-K selection overlap 1.000 and max-rel ≤ 1e-3 vs CPU, and
   report the new upload/scan timings (target: upload → sub-ms, first-scan build
   amortised, steady-state scan ≈ 29 ms).
5. **Governor (integration).** Under induced VRAM pressure, the gallery evicts
   before KV; a post-eviction scan rebuilds and still matches.

Assertions follow the repo rule: **raw expected values, not error thresholds**,
for the paged-vs-contiguous equality (test 1); ULP tolerance only where fast-math
already applies (tests 2/4).

---

## 11. Module layout & touch points

New (`candle-conversation/src/provenance/gallery_arena/`, one concern per file):
- `mod.rs` — `GalleryArena`, `ResidentTurn`, `ensure_resident`, the per-scan
  index builder.
- `gid_pool.rs` — `GalleryGidPool`, `PageGid`, `ArenaRefcounts` analog (mirrors
  `chunked/gid_pool.rs`).
- `pages.rs` — the group-major transpose + `upload_page` (pinned staging) +
  `ResolvedSlab`/`page_ptr`.
- `residency.rs` — the `StreamId → ResidentTurn` map, fingerprinting, LRU, the
  governor `register_relief` hook.

Changed:
- `candle-kernels/src/provenance/bdp_scan.cu` — the address resolution (§7.2) +
  the `page_ptr` shared cache. Hot loop unchanged.
- `candle-kernels/src/provenance/api.rs` — FFI signature (§7.3).
- `candle-conversation/src/provenance/gpu.rs` — `scan_weighted` reads from the
  arena (page_ptr/pos_map) instead of a contiguous pinned buffer; the contiguous
  `from_segments` path is **removed** (no backward-compat shim — CLAUDE.md), its
  parity role assumed by test 1.
- `candle-conversation/src/projection/resolver.rs` — Phase B calls
  `ensure_resident` + the index builder instead of `BatchedGpuGallery::from_segments`;
  the per-group pinned-gallery cache (`gpu_gallery_cache`) is replaced by the
  arena's per-turn residency (finer-grained, delta-uploading).
- Arena construction is owned by the scheduler (holds the `Device` and the
  `VramGovernor`, `batched_inference.rs:1388`) and handed to the `Conversation`
  like the persistence writer.

---

## 12. Implementation phases

Every phase lands complete — no stubs, no deferred cleanup (CLAUDE.md).

- **Phase 0 — the arena allocator.** `GalleryGidPool`/`PageGid`/slab storage +
  unit tests (alloc/free/recycle/tombstone/hwm, mirror the `gid_pool.rs` tests).
  No kernel involvement yet.
- **Phase 1 — pages + residency.** Group-major transpose, `upload_page`,
  `ensure_resident`, fingerprint/LRU. Test: a turn's pages round-trip
  (upload → D2H) bit-identically; re-seal re-uploads one turn.
- **Phase 2 — the paged kernel.** `bdp_scan.cu` address change + FFI + launcher;
  the paged-vs-contiguous parity test (test 1). Keep the contiguous path alive
  only until this passes, then delete it.
- **Phase 3 — wire the resolver.** Phase B builds the per-scan index from the
  arena; delete `from_segments` and `gpu_gallery_cache`. Resolver parity +
  real-substrate example (tests 2/4). Measure upload → sub-ms.
- **Phase 4 — governor.** `register_relief` + `resident_bytes` + scan-local pins;
  induced-pressure integration test (test 5). Confirm KV is untouched until the
  gallery is fully shed.

---

## 13. Risks & open questions

**Necessary, not sufficient, for 5–13 ms.** The arena removes the ~10 ms upload
(→ ~29 ms) but the **25 ms kernel** is the remaining wall. That is load imbalance
across 788 wildly-sized segments and a 32× per-tile gallery re-read — a separate
kernel-restructuring effort (a follow-on doc), and one that touches the hot loop
we have so far kept frozen. This doc should ship knowing it gets us to ~29 ms, not
the target, alone.

**Q1 — dedicated pool vs. sharing the raw slab allocator.** The `gid_pool.rs`
allocator is *almost* format-agnostic; the palette/KV coupling is in the layers
above. Do we (a) copy ~300 lines for a clean, independent `GalleryGidPool`, or
(b) refactor `gid_pool.rs` to expose a format-parametric core both KV and the
gallery use? (a) is faster and lower-risk now; (b) is less duplication long-term.
**Recommendation: (a) for v1**, revisit (b) if a third arena appears.

**Q2 — page size.** `PAGE_TOKENS = 32` matches `CHUNK_SIZE` and keeps partial
waste ≤ 6 KiB/turn, but yields ~18 k `page_ptr` entries and a coalescing break
every 32 tokens. Larger pages (e.g. 128) shrink `page_ptr` 4× and the break
frequency but waste up to ~24 KiB per small turn. **Recommendation: 32 for v1**;
it's a one-constant sweep to retune once measured.

**Q3 — do we need defrag?** Uniform 6 KiB pages reached via `page_ptr[]` are
scatter-tolerant for the scan, so v1 likely skips the KV defrag/remap machinery
(`backing.rs:349`) and relies on `release_empty_slabs` alone. If long-run
fragmentation strands live pages thinly across slabs (wasting whole 16 MiB slabs
under governor pressure), add the greedy-drain compaction later. **Open: measure
slab occupancy over a long session before committing.**

**Q4 — `pos_map`/`case` residency.** v1 rebuilds+uploads them per scan (~4.6 MB,
0.4 ms). They are stable within a turn; caching them (keyed by the group
fingerprint) drops steady-state upload to `case`+probe. **Defer** unless 0.4 ms
matters after the kernel work.

**Q5 — cross-fork sharing.** Developer forks share the substrate/KV prefix. Their
gallery turns are the *same* `wide_q_sigs` streams, so the arena could share
resident pages across forks by keying residency on `StreamId` (already global).
The refcounted `PageGid` supports this directly (a shared page has refcount > 1).
**Opportunity, not v1 scope** — but the residency key choice (StreamId, not
per-conversation) keeps the door open.

**Q6 — first-scan latency after a cold start / mass eviction.** Re-materialising
the whole gallery (788 turns) costs one transpose+upload pass (~the current
build, ~150 ms). This hits only on a cold conversation or after a full
governor sweep, amortised over the turn's reprojections. Acceptable; note it in
the timing story so it isn't mistaken for per-scan cost.

---

## 14. The tensor-core backend (`bdp_bmma.cu`) — built

The scan's inner operation — `popcount(XNOR(query, token))` over 512-bit group
signatures — is *literally* the 1-bit tensor-core instruction:
`BMMA.88128.XOR.POPC` computes an 8×8 tile of `popcount(A XOR B)` over 128-bit
K-chunks in one warp op, and `agreement = 512 − xor_popc` is an exact integer
transform. The b1 path exists on **sm_75..sm_89** (Turing/Ampere/Ada — the dev
4090 is sm_89, instruction confirmed in SASS); **Hopper and Blackwell dropped
it**, so the 2×5090 production box runs the scalar backend until an INT8 IMMA
(±1 encoding) variant exists. `bdp_bmma_supported()` gates at runtime; device
code is `__CUDA_ARCH__`-guarded so the sm_120 compilation is a stub.

### Structure — two kernels, fused segmented reduction

- **`bdp_bmma_accum_kernel`** — grid `(64-token chunk, group)`; 256 threads = 8
  warps (4 query-blocks × 2 token-halves). Per CTA: stage the gallery chunk once
  in shared (K-chunk-major, so every wmma fragment pointer is 32-byte aligned at
  ldm = 128 bits), dense-rank the chunk's cases (two warp ballots — cases are
  non-decreasing over the scan order because the index builder sorts each
  segment's windows by case), hoist the warp's 16 gallery fragments into
  registers, then **loop the query tiles inside the CTA** (staging and ranks
  amortise across all tiles). Per tile: 4 `bmma_sync` per 8×8 output tile, a
  **run-merged epilogue** (16 lanes each merge a half-row's consecutive
  same-rank columns in registers → one shared atomic triple per run, not per
  element), and a per-(query, rank) flush to global `case_max`/`case_sum`/
  `case_sumsq` accumulators. Rank windows of 32 cap the shared accumulators at
  12 KB (4 CTAs/SM); a chunk with more distinct cases (all-tiny exchanges)
  reruns its tiles per window.
- **`bdp_bmma_finalize_kernel`** — one thread per (query, group, segment): scans
  the segment's case range for leader/runner-up (same ascending order and strict
  comparisons as the scalar kernel) and emits `(out_case, z*margin)` through the
  shared `bdp_vote` — so the two backends' votes **bit-match** (verified: the
  adversarial parity test asserts full bit equality across 1-token exchanges,
  case-id gaps, unsorted window input, dropped out-of-range cases, seam
  sub-windows, chunk-straddling segments, and non-tile-multiple probe/token
  counts).

The flattened chunk grid also removes both scalar structural limits: no
per-segment block tail (a 3-token file no longer occupies a whole CTA) and no
shared-memory dependence on the largest segment's case count — the 48 KB
`max_seg_cases` guard applies only when the scalar backend actually runs.

### Optimisation history (measured on the real substrate, Nsight-guided)

| step | accum kernel | scan steady | limiter addressed |
|------|-------------|-------------|-------------------|
| first correct version | 22.4 ms | 23.6 ms | — (per-element epilogue atomics, 8-way same-address serialization) |
| run-merged epilogue | 14.8 ms | 17.7 ms | shared-atomic contention |
| query-loop-in-CTA + ballot ranks | 7.4 ms | 10.6 ms | CTA-barrier stalls (45%) + 8× gallery re-staging |
| B-fragments hoisted to registers | 7.1 ms | ~10.6 ms | (neutral; kept — strictly less shared traffic) |
| rank windows → 4 CTAs/SM | **5.7 ms** | **~9.5 ms** | occupancy (76% no-eligible at 2 CTAs/SM) |

Final profile: memory-pipe 93% (saturated on fragment staging + store — the
structural floor of this design), DRAM ~5%, occupancy 66%. The remaining ~4 ms
of scan time outside the kernel is index upload (~0.7), result download (~1.5),
host tally (~1.3), finalize (~0.15) — candidates only if the target ever
tightens (on-GPU tally would remove ~2.8 ms).

### 14.1 The INT8 (IMMA) backend (`bdp_imma.cu`) — the Blackwell production path

Because Hopper/Blackwell dropped b1 BMMA, a third backend covers the 2×5090 box:
`mma.m16n8k32.s8` via inline PTX (documented fragment↔thread mappings, which is
what makes register-side bit expansion legitimate) — **built, parity-proven, and
ncu-optimized on the Ada dev card by forcing it over the locally-faster b1 path**
(`scan_weighted_imma`), and compiled as real SASS for sm_120 by the project's
gencode set, so the build itself is the Blackwell compile-check.

Structure: identical to the b1 backend (same chunk grid, ballot ranks, rank
windows, global accumulators, shared finalize kernel) with the MMA core swapped:
a warp computes a 16×8 tile over 16 K-steps; the sign bits stay **packed** in
the arena and each thread expands the nibbles it needs into fragment lanes in
registers — the 8× inflation never exists in memory. The winning encoding is
**0/1 (not ±1)**: the MMA then accumulates `m11 = popc(q AND t)` and

```
agreement = 512 − popc(q) − popc(t) + 2·m11
```

with the per-row popcounts staged next to the bits — exact integers, two ops
cheaper per fragment (measured ~25% off the whole kernel: accum 13.8 → 10.6 ms
on sm_89; a ±1 register-caching variant was tried and REVERTED — 93 regs halved
occupancy). The register epilogue run-merges each thread's 4 accumulator
elements (adjacent-token pairs) with no staging buffer at all.

Measured on the dev card (sm_89, real substrate): IMMA scan ~14–18 ms — slower
than b1 (~10 ms) as expected, faster than scalar (~18–21 ms), votes bit-identical
to both (the 3-way adversarial parity test asserts full bit equality). Projected
on one RTX 5090 (~2.9× SM×clock, PCIe 5, 7970X host): accum ~3.5 ms, **scan
~6 ms — inside the 3–10 ms target on the production hardware**. Backend ladder:
b1 (sm_75..89) → IMMA (sm_80+, incl. Hopper/Blackwell) → scalar (universal).
