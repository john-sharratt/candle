# Co-Resident KV-Head Metadata

## Summary

The per-chunk KV-head metadata that the paged attention kernels read (palette
maps, formats, outer scales, and resolved device pointers) is **KV-cache state**,
not per-forward scratch. Today it is reconstructed from scratch on the host for
every layer of every forward and shipped across PCIe. This document specifies
moving that metadata to a **device-resident per-chunk record that lives next to
the KV bytes it describes** — frozen at seal time, pointer-patched at tier
migration, and read directly by the kernels. The CPU stops touching sealed-chunk
metadata during a forward entirely.

> **Validation status.** This design was reviewed against the code in depth. The
> frozen-vs-location split, the migration pointer-patch hook, the position-agnostic
> sharing, the single-point kernel ABI change, and the per-slot ordering residue
> all hold. One early framing did **not** survive review and has been corrected
> below: a sealed chunk does **not** map to a single `(arena_idx, chunk_idx)`, so
> the metadata record is keyed by a **stable per-chunk handle**, not by an arena
> slot. The freeze hook is **chunk-window construction / sealed injection**, not
> `set_block_gids`. See the inline notes marked **[corrected]**.

This collapses the dominant cost of the reprojection glue forward
(`slot:build ≈ 270 ms` of a ~660 ms glue wave on Qwen3-30B-A3B) and the
equivalent cost in batched prefill, and unifies the metadata contract across
paged-decode, paged-prefill, and paged-glue.

## Implementation status — DONE, GPU-validated

The full design is implemented and validated on an RTX PRO 5000 (Blackwell,
sm_120). On the `reproject_control` 30B integration test the reprojection glue
forward's `slot:build` dropped from ~270–450 ms to **~108 ms** (≈3–4×),
`glue:hdr_meta` ~470→112 ms, and total `glue_ms` ~660→**403 ms** (−40%), with
**bit-identical decode output**.

**Pool + record (`candle-nn/src/kv_cache/chunked/meta_pool.rs`).** `MetaGid` RAII
handle + `MetaPool` (refcount slabs, lazy growth, CUDA device slab with
`write_record`/`device_addr`) and `serialize_kv_heads` (the 168 B/head record
body). 7 unit tests incl. a byte-exact golden and the multi-arena pointer case.
`meta: Option<MetaGid>` on `ChunkWindow` + `SealedChunk`, propagated through every
construction site (fork / inject / snapshot / **view borrow**); `MetaPool` on
`BackingInner`.

**Records are rebuilt at every GPU-residence point, not persisted.** A `MetaGid`
is a device handle that cannot survive the redo log or a tier demotion, so
`build_meta_record` runs wherever a sealed chunk *becomes* GPU-resident:
- **quantize** (`quantize_sealed_in_place`) — in-session seal;
- **cold-load** (`alloc_sealed_blocks_bulk`, after `set_block_gids` finalizes the
  placement) — the redo-log → hot path the reproject uses on a fresh process;
- **warm→hot elevate** (`migrate_sealed_to_gpu_batch_async` Phase 6) — eviction
  then promotion.

Built **once per residence**, then read by every subsequent reproject. Demote and
the rarely-used single `migrate_sealed_to_gpu` keep `meta = None` (records are
GPU-only; the scratch path is the correct fallback). `set_block_gids` clears
`meta` so a stale record can never outlive a re-placement.

**ABI (`slot_types.cuh` + serializers).** `get_head` dereferences a per-slice
`kvheads_ptr` and `token_slice_byte_size` is fixed at 16. Both serializers emit a
two-section layout `[ slice headers (16 B) | records ]`: decode (`gpu_chunks.rs`)
self-references the records section within its own buffer; prefill/glue
(`build_slot_headers`) submits a records buffer first (resident chunks point at
`device_addr(meta)` and skip the rebuild; transient/float chunks serialize a
scratch record). `from_sealed_chunk` skips building heads entirely for resident
chunks — that skip is the latency win.

**Validation:** `kernel_layout` decode+prefill, `paged_glue` quant (resident
path) + f16, 274 chunked + 7 meta_pool tests, and the `reproject_control` 30B
end-to-end run — all green.

## Motivation

### The metadata is immutable from seal, except for one field

A `SealedChunk` (`candle-nn/src/kv_cache/chunked/types.rs`) carries everything
the kernel needs to interpret a chunk's bytes: `gids`, `offset`, `token_count`,
`k_pal`, `v_pal`, `k_scale`, `v_scale`. The host materializes this per head into
a `KvHeadHost` (`candle-transformers/src/models/slot_state.rs`). Its on-device
form is **168 bytes/head at HD128**, grouped here *by lifetime* (not by byte
order — the actual byte offsets put the pointers in the middle, at 64–128):

| field | bytes | lifetime |
|-------|-------|----------|
| `k_pal`, `v_pal` | 32 + 32 | frozen at seal |
| `k_fmt`, `v_fmt` | 4 + 4 | frozen at seal |
| `k_scale`, `v_scale` | 16 + 16 | frozen at seal |
| `k_ptr[4]`, `v_ptr[4]` | 32 + 32 | location-dependent (one pointer **per palette**) |

> **[corrected] There is no C++ `KvHead`/`TokenSlice` struct.** Both are
> offset-addressed raw byte buffers via accessors in
> `candle-kernels/src/paged-decode/slot_types.cuh` (offset map mirrored in
> `arena_table.cuh`); only `SlotHeader` is a real `static_assert`-checked 24-byte
> struct. The frozen/location split below is by *byte ranges*, not struct fields.
> Note also that `k_ptr`/`v_ptr` are **`[u64; N_PALETTE]`** — 8 pointers total per
> head, one per palette per side — which matters for the table design below.

**104 of 168 bytes are frozen the moment the chunk is sealed** and never change
again for a given placement. Sealing has two flavours, and the frozen bytes are
well-defined in both:

- **Float seal** (`record_turn` with no reconcile, decode-primed via
  `prime_chunked_decode_slots_batch` → `sync_decode_gpu_chunks`): identity
  palettes, unity (1.0) scales, the arena's float `fmt` tag. Constant. This path
  **never calls `set_block_gids`**. *[corrected]* live float `ChunkWindow`s carry
  **populated** identity palette bytes and **populated** `1.0` scales (set at
  `alloc_block_chunks`, `alloc.rs:595`; `backing.rs:678`), not empty `Vec`s — the
  "empty ⇒ identity/unity" convention exists but is not what the live float path
  stores.
- **Quantize**: the real pal/scale values are produced by
  `quantize_sealed_in_place` (`chunked/compress.rs`), which builds **new**
  `SealedChunk`s carrying the quantized palette maps and outer scales and reaches
  a live block table via `inject_sealed_at_tail`. *[corrected]* this does **not**
  go through `set_block_gids` — the only `set_block_gids` finalization is
  cold-load reinjection (`alloc_sealed_block(s_bulk)`); `backing.rs:208` is the
  defrag remap, which **preserves** pal/scale and only swaps GID values.

After either finalization the bytes are immutable for the life of that placement;
the only event that changes them is a tier migration, which relocates the chunk
(new pointers) while carrying the frozen bytes forward unchanged.

The remaining 64 bytes — the 8 device pointers — are a pure function of *where
each sub-band currently lives*: `base_ptr + chunk_idx * chunk_byte_stride`,
resolved independently per palette per side. They change only when the chunk
moves between tiers.

The tier-migration code already proves this split. `migrate_sealed_to_gpu_batch_async`
(`chunked/chunk_ops.rs`) Phase 6 rebuilds every `SealedChunk` as:

```rust
Ok(SealedChunk {
    gids: mapped,        // NEW placement → new pointers
    ..chunk.clone()      // pal / fmt / scale / offset / token_count: unchanged
})
```

Only `gids` change across a migration; the entire frozen payload is carried
forward byte-for-byte. The pointer for the new placement is computed inside the
same function at Phase 4:
`dst_ptr = info.base_ptr + dst_chunk_idx * info.chunk_byte_stride`.

### What the forward path does today (the waste)

There is no co-resident metadata. `build_slot_headers`
(`candle-transformers/src/models/prefill_utils.rs`), shared by paged-prefill and
paged-glue, does the following **for every layer**:

1. `live_chunks_as_sealed` (`backing.rs:854`) clones all five `Arc<Vec<…>>`
   fields of every chunk into a fresh `Vec<SealedChunk>`. For a 126-block prefix
   × 48 layers that is ~30k allocations per reproject.
2. `SlotStateHost::from_sealed_chunks` rebuilds the entire `slices` array, and
   for each chunk `KvHeadHost::from_gids` re-resolves pointers and re-derives the
   104 frozen bytes that have not changed since the turn was quantized.
3. The whole thing is serialized and uploaded across PCIe through the pinned
   stager.

The decode path is faster only because it already pays this once and caches the
result per-sequence in a device buffer (`GpuChunks`, `chunked/gpu_chunks.rs`),
rebuilding solely at chunk boundaries (`sync_decode_gpu_chunks`,
`types.rs:826`). But that cache is **per-slot-view**, rebuilt from host chunk
state, and still re-resolves pointers; it does not exploit immutability and is
not shared across the prefill / glue paths.

### The constraint that makes this clean

Under the substrate, sealed sections/turns are immutable while resident, and
move between VRAM / RAM / NVMe as whole units via `elevate_to_hot` /
`evict_hot_except` (`candle-conversation/src/persistence/elevate.rs`). The
metadata that describes a resident sealed chunk is therefore constant for as
long as the chunk is resident, and the *only* event that invalidates its
pointers is the migration that relocates it — an event that already walks every
chunk and already computes the new pointers.

## Design

### The co-resident KV-head record

> **[corrected] Why this is not a per-arena table.** A sealed chunk does not own
> one physical chunk slot. Its `HeadGids` holds `N_PALETTE × 2 × n_kv_head`
> (`= 8 × n_kv_head`, N_PALETTE=4) **distinct GIDs** — one per (head, palette,
> K/V) — and per-head adaptive quantization routes those sub-bands into
> *different arenas* at *different `chunk_idx`* (`head_gids.rs:19`, `:156`;
> `arena_byte_size` exists precisely to sum strides across the distinct
> `(arena_idx, chunk_idx)` slots, and a regression test asserts the 16 sub-bands
> land at 16 different `chunk_idx`). So `kvhead_table[arena_idx][chunk_idx]` is
> undefined — there is no single GID, arena, or chunk_idx for "the chunk." Each
> head's on-device record itself holds 8 pointers into up to 8 arenas
> (`from_gids`, `slot_state.rs:138`).

The metadata is therefore a **per-chunk record** addressed by a **stable
per-chunk handle**, not an arena slot:

```
chunk → 8·n_kv_head KV sub-band GIDs scattered across arenas   (exists today)
chunk → ONE KvHead[n_kv_head] record in a dedicated metadata pool   (new)
```

A sealed chunk gains a `meta` handle (a `ChunkGid`-style RAII id from a dedicated
metadata pool, allocated once per chunk and stored on the `ChunkWindow` /
`SealedChunk`). The record is `n_kv_head × sizeof(KvHead)` and holds the full
per-palette pointer/pal/scale/fmt set for that chunk. The slice's `kvheads_ptr`
(below) resolves to this record's device address. The record is a normal VRAM
allocation that travels with the chunk across tiers; the CPU never reads it
during a forward.

The handle must be stable across the chunk's life **but not across migration** —
each elevate/demote reallocates KV GIDs anyway, so the record is re-placed (or
its 8·n_kv_head inner pointers rewritten) at the same point the KV bytes are
re-placed. The one invariant that must hold: every slot referencing the same
physical sealed chunk (same GIDs) resolves to the **same** record, so the
sharing in “Per-slot state” below is real.

### Lifecycle

**Build at chunk-window construction (freeze the 104 bytes).** *[corrected]*
There is no single `set_block_gids` chokepoint. A live block-table entry is a
`ChunkWindow`, and the record must be (re)built wherever a `ChunkWindow` with a
finalized `(gids, pal, scale, fmt)` tuple is installed:

- **float writer birth** — `alloc_block_chunks` (`alloc.rs:591`): identity/unity
  defaults;
- **sealed injection** — `inject_sealed_at_tail` (`sequence_ops.rs:1944`): the
  dominant path; carries both quantized reinjection (from
  `quantize_sealed_in_place`) **and** float section projection into a live slot;
- **cold-load reinjection** — `alloc_sealed_block(s_bulk)` (`chunk_ops.rs:1193`):
  rebuilds frozen bytes from the redo-log `ChunkPayload`.

The cleanest implementation gives `ChunkWindow` a single
`write_meta_record(meta_gid, &arena_info)` primitive and calls it from those
sites, rather than chasing each one — they all converge on `ChunkWindow`
construction. Defrag (`apply_gid_remap`, `backing.rs:208`) preserves pal/scale
and only changes pointers, so it patches pointers, not frozen bytes. Either way
the write runs once per chunk per placement — not once per forward. A purely
float, never-quantized sequence is fully supported.

**Patch at migration (the 8·n_kv_head pointers).** *[corrected]*
`migrate_sealed_to_gpu_batch_async` / `migrate_sealed_to_gpu` (and the symmetric
`migrate_sealed_to_cpu*`) dedup the chunk's GIDs into `unique_raws`, alloc a
fresh destination **per GID** (Phase 3), and compute `dst_ptr` **per GID** in one
loop (`chunk_ops.rs:2239`, the `dst_ptr = base_ptr + chunk_idx*stride` line). A
table-write hook attaches exactly there: it already computes every one of the
chunk's 8·n_kv_head destination pointers, each from that sub-band's own
destination arena. The hook writes the frozen 104 bytes (from the source
`SealedChunk`) plus those patched pointers into the chunk's record at its new
placement — one extra device scatter on the copy stream already moving the KV.
Demote drops the GPU record; the warm/cold tier keeps the frozen fields in
`SealedChunk` (and the redo log keeps them on NVMe), so re-elevation rebuilds the
record from data already in hand.

**Cold-load builds fresh (no prior record to patch).** On NVMe→hot, pal/scale/fmt
are reconstructed from the redo-log `ChunkPayload` and brand-new GIDs are
allocated (`pipeline.rs` → `alloc_sealed_blocks_bulk`). The record is **built
from scratch** there, not pointer-patched — the natural site is alongside the
bulk alloc.

**Read in the kernel.** The kernel obtains a chunk's `KvHead*` from the slice's
`kvheads_ptr` (the record's device address) rather than reading inline bytes from
the uploaded slot buffer.

### Per-slot state shrinks to ordering only

A sealed chunk's *heads* are shared across every slot that references it. The
genuinely per-slot quantity is the **RoPE base** (cumulative usage within that
slot's layout — `SealedChunk` is deliberately position-agnostic, `types.rs:242`,
and carries no rope; `rope_base` is recomputed per slot at `slot_state.rs:424`).
*[corrected]* `offset`/`token_count` are **fixed in the `SealedChunk` at seal**
and copied verbatim into every referencing slot (`inject_sealed_at_tail`,
`create_view_sequence` borrow whole blocks, not sub-windows) — so they are
identical across all sharers, which only makes sharing *easier*: a slice keeps
`(offset, len, rope)` and replaces its inline `KvHead[n_kv_head]` array (160+
bytes) with a single pointer/handle into the resident record:

```
TokenSlice (today): offset:u16, len:u16, rope:u32, heads:[KvHead; n_kv_head]   // 8 + n_kv_head*168
TokenSlice (new):   offset:u16, len:u16, rope:u32, kvheads_ptr:u64             // 16 bytes, fixed
```

The per-forward host upload for the sealed prefix drops from
`n_chunks × (8 + n_kv_head·168)` bytes to `n_chunks × 16` bytes — and the bytes
it does upload are pure ordering data (which resident chunk, what window, what
RoPE), which is layer-invariant and can be built once per forward exactly as
decode already builds its `position_map` once (`build_decode_metadata`,
`batched_inference.rs:744`).

### Uniform kernel path: resident vs. scratch heads

The writer region (and any not-yet-quantized partial tail) is mutating float
state with no resident table entry. Rather than branch in the kernel, every
slice carries a `kvheads_ptr` and the kernel always dereferences it:

- **sealed chunk** → `kvheads_ptr` points at the chunk's resident record
  (resident, zero per-forward cost).
- **writer / glue / partial / forked-CoW-tail chunk** → `kvheads_ptr` points into
  a small **per-forward scratch table** built and uploaded by the host for just
  those few chunks — the only metadata upload a forward performs, and exactly the
  transient float island reprojection already throws away. *[corrected]* a fork's
  copy-on-write partial tail allocates fresh GIDs (`sequence_ops.rs:935`) → new
  pointers, so it is a distinct placement and correctly falls here until it
  re-seals into its own record.

This is the single contract all three kernels converge on: **decode, prefill,
and glue read `KvHead` through a pointer; the sealed prefix resolves to the
co-resident record, the writer island resolves to per-forward scratch.** The ABI
change is single-point: every head access already funnels through `get_head` in
`slot_types.cuh` (shared by all three kernels), so only that accessor's base
arithmetic and `token_slice_byte_size` change.

## Affected components

| area | change |
|------|--------|
| `chunked/gid_pool.rs`, `arena.rs` | a dedicated metadata pool + RAII `meta` handle (per-chunk record allocation) |
| `chunked/types.rs` (`ChunkWindow`, `SealedChunk`) | carry the `meta` handle; one `write_meta_record` primitive |
| `chunked/sequence_ops.rs` (`inject_sealed_at_tail`), `alloc.rs` (`alloc_block_chunks`) | build/refresh the record at chunk-window construction (the real finalization sites — **not** `set_block_gids`) |
| `chunked/chunk_ops.rs` (migrate \*, `alloc_sealed_block(s_bulk)`) | patch 8·n_kv_head pointers at the per-GID `dst_ptr` loop; build fresh on cold-load |
| `chunked/backing.rs` (`apply_gid_remap` defrag) | patch pointers only (frozen bytes preserved) |
| `models/slot_state.rs` | `TokenSlice` carries `kvheads_ptr`; `KvHead` serialization split into frozen vs. per-palette-pointer halves |
| `models/prefill_utils.rs` (`build_slot_headers`) | build ordering-only slices for the prefix; build scratch heads only for the writer island |
| `models/batched_inference.rs` (`build_decode_metadata` / `sync_decode_gpu_chunks`) | resident-record pointers replace per-slot head rebuild; keep the position_map-once structure |
| kernels: `slot_types.cuh` (`get_head`, `token_slice_byte_size`) | read `KvHead` via the slice's `kvheads_ptr` — single shared accessor, covers all three kernels |

## Correctness

- **Format/pal/scale consistency.** The frozen bytes are written from the same
  `SealedChunk` fields (`k_pal/v_pal/k_scale/v_scale` + per-GID `fmt`) that define
  how the arena bytes were quantized, so the record can never disagree with the
  data it describes.
- **Pointer validity.** The 8·n_kv_head pointers are written only by the code that
  places each sub-band (alloc/migrate/defrag), under the same `state.write()` lock
  that updates the GID, so a record's pointers always match the chunk's current
  arena locations. Eviction drops the GPU record with the chunk.
- **Shared chunks.** A chunk referenced by N slots resolves to one record read by
  all N — correct because heads are position-agnostic and the GIDs (hence the 8
  pointers + pal/scale/fmt) are identical across Arc-sharers; only per-slot RoPE
  and windowing live in the slice. A fork's CoW partial tail is a *different*
  placement (new GIDs) and gets its own record, so the "one record per N slots"
  invariant is per-GID-placement, not per-logical-chunk.
- **Partial tail / writer.** Never resident; always resolved through per-forward
  scratch. A partial chunk that later fills is re-sealed as a **new** `SealedChunk`
  by `quantize_sealed_in_place` and injected via `inject_sealed_at_tail`, which is
  where its record is built — the same site as any other sealing chunk.

## Testing

Per repo convention, codec/layout assertions are byte-exact, not threshold-based.

- **Record-build golden (quantized).** Quantize a known chunk; assert the
  resident record's frozen 104 bytes/head equal the expected serialization,
  byte-for-byte, including a per-palette `fmt`/`scale` that differs across
  sub-bands (exercise the multi-arena case explicitly).
- **Record-build golden (float seal).** Seal a float (BF16/F16) turn without
  reconcile; assert the record holds the **populated** identity palette bytes,
  `1.0` scales, and the arena's float fmt tag — the design must not depend on
  quantization happening, and must match what live float `ChunkWindow`s store
  (not the empty-`Vec` encoding).
- **Migration pointer patch.** Elevate a sealed turn warm→hot; assert each of the
  record's 8·n_kv_head pointers equals `base_ptr + chunk_idx*stride` for that
  sub-band's new placement and the frozen bytes are unchanged from before.
- **Multi-arena chunk.** Build a chunk whose sub-bands span ≥2 arenas; assert the
  single record correctly carries pointers into both, and one slice's
  `kvheads_ptr` resolves all heads.
- **Kernel equivalence.** For all three kernels, assert bit-identical output
  between the inline-heads path and the resident-table path on the same KV
  (extends the existing `paged_glue_matches_golden_quant` /
  `kernel_layout_tests` gates).
- **Shared-chunk read.** Two slots referencing one sealed chunk at different tail
  positions produce correctly-RoPE'd attention from a single table entry.
- **Reproject end-to-end.** `reproject_control` / `reproject_wave` under
  `--features profile`: `slot:build` for the sealed prefix goes to ~0; only the
  writer-island scratch remains.

## Expected impact

- Reprojection glue forward: removes the ~30k per-reproject chunk-clone
  allocations and the per-layer 168-byte head rebuild for the sealed prefix.
  `slot:build` (~270 ms) collapses to the writer-island scratch (tens of
  chunks). The glue forward becomes kernel-bound.
- Batched prefill with a resident prefix: same `build_slot_headers` saving.
- Decode: the per-slot `GpuChunks` rebuild at chunk boundaries is replaced by a
  resident-record pointer; the hot path keeps its position_map-once structure with
  no per-slot head serialization. This also removes the duplicate head bytes two
  slots sharing a chunk store today (each holds its own serialized copy).
- PCIe: the sealed prefix's metadata stops crossing the bus. Only the small
  transient writer-island scratch is uploaded per forward.
