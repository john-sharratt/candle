# Arena Unification — Measurement Log

Companion to `docs/arena_unification.md`. One section per step: the gate
result, the numbers the step predicted, and what actually happened. Design
lives in the other file; **this file is evidence only**.

Gate command (every step):

```
cargo test --release --features cuda,verbose --lib --package candle-transformers \
  quantized_qwen3_moe::tests::test_parallel_batched_forwarding -- --ignored --nocapture
```

Second gate (from step 1): the four `gpu_cpu_gpu_round_trip_*_is_byte_identical`
tests in `chunk_ops.rs`.

Machine: RTX 4090 Mobile 16 GB, driver 596.08, WDDM. Model:
Qwen3-30B-A3B-Instruct-2507 Q4_K_M (GGUF, cached).

---

## Step 0 — Baseline

### 0.2 — Main gate: **GREEN**, 16/16 configs, 119.4 s total

The whole gate is **two minutes**, not the 20-minutes-per-config that was
budgeted. Iteration is far cheaper than the plan assumed — every step can
afford a full gate run rather than saving it for step boundaries.

| KvMode | Contexts | Valid | t/s (bulk) | t/s (single) | %Quant | Compress | Peak toks |
|---|---|---|---|---|---|---|---|
| F16 | 1 | ✓ | 489.4 | 10.5 | – | – | 626 |
| BF16 | 1 | ✓ | 539.8 | 19.2 | – | – | 626 |
| BF16 | 10 | ✓ | 1890.3 | 126.3 | – | – | 6310 |
| Q8_0 | 20 | ✓ | 2370.7 | 170.4 | 100.0 % | 1.88× | 12620 |
| Q4_0 | 4 | – | 1785.4 | 64.8 | 100.0 % | 3.56× | 2546 |
| C0 | 2 | ✓ | 1069.3 | 43.8 | 100.0 % | 1.98× | 1292 |
| C1 | 2 | ✓ | 1069.2 | 44.4 | 100.0 % | 2.54× | 1292 |
| C2 | 2 | ✓ | 1062.0 | 44.8 | 100.0 % | 2.74× | 1292 |
| C3 | 2 | ✓ | 1073.1 | 45.9 | 100.0 % | 2.99× | 1292 |
| C4 | 2 | ✓ | 1069.0 | 46.1 | 100.0 % | 3.41× | 1292 |
| C5 | 2 | ✓ | 1045.0 | 43.8 | 100.0 % | 3.67× | 1292 |
| C6 | 2 | ✓ | 1054.4 | 43.0 | 100.0 % | 4.18× | 1292 |
| C7 | 2 | ✓ | 1060.6 | 40.2 | 100.0 % | 4.24× | 1292 |
| C9 | 2 | ✓ | 1060.6 | 43.9 | 100.0 % | 5.31× | 1292 |
| BF16 | 1 | ✓ | 579.8 | 25.2 | – | – | 626 |
| Q4_0 | 20 | – | 2370.3 | 177.9 | 100.0 % | 3.56× | 12620 |

`Valid: –` on the two Q4_0 rows is `test_mode: Skip` by design, not a failure.

**Correction to the design doc**: §8 A2 says the gate spans "1–48 contexts".
Actual maximum is **20**. The coverage claim (BF16/F16/Q8_0/Q4_0 + C0–C7, C9,
C8 absent) is otherwise confirmed exactly.

### 0.3 — Second gate: **GREEN**, 4/4 in 0.73 s

`gpu_cpu_gpu_round_trip_{,_f16,_r16,_quantized}_is_byte_identical` all pass.

**Procedural note, learned the hard way here.** The first read of this log
looked like `0 passed; ... filtered out` and I called it a false green — but I
had only read `tail -10` of a 12-binary log and cut off the lib binary, which
did run 4/4. The real lesson survives the false alarm: **a gate must assert on
the count of tests that ran**, not on the exit code. `cargo test` with a filter
that matches nothing exits 0. From here on every gate records `N passed` and
`N` is checked against an expected value.

### 0.6 — Payload/stride pre-flight

**The pre-flight paid for itself immediately.** The design doc's step-1 list
named seven sites; the sweep found **nine, in two opposite-facing categories**,
and one of the doc's seven was misclassified.

**Category A — payload inferred from stride** (must switch to
`chunk_payload_bytes`):

| # | Site | Role |
|---|---|---|
| A1 | `chunk_ops.rs:1987` | DtoH gather copy length |
| A2 | `chunk_ops.rs:2206` | cross-layer gather copy length |
| A3 | `head_gids.rs:185` | `arena_byte_size` → `SealedChunk.byte_size` |
| A4 | `migrate.rs:199` | `resolve_sealed_chunk_ptrs` → `(ptr, len)` |
| A5 | `migrate.rs:234` | `resolve_sealed_chunk_ptrs_per_gid` → `(ptr, len)` |
| A6 | **`migrate.rs:423`** | `resolve_block_ptrs_from_hgids` → `(ptr, len)`; **NEW** — feeds cold-load via `pipeline.rs:589` |
| A7 | `transfer.rs:269` | `seal_to_chunk_images_cpu` blob slot reservation |
| A8 | **`chunk_ops.rs:2597`** | `chunk_byte_size_of(arena)` in the HtoD scatter — **NEW**, and not a `chunk_byte_stride` expression at all |

**Category B — stride inferred from payload** (must switch to the *class
stride*; these fail in the opposite direction):

| # | Site | Expression |
|---|---|---|
| B1 | `chunk_ops.rs:166` | `read_chunk_into_pinned_bytes`: `byte_offset = chunk_idx * dst.len()` |
| B2 | `chunk_ops.rs:2498` | `write_chunk_from_pinned_bytes`: `byte_offset = chunk_idx * bytes.len()` |

Category B is a class of bug the design document does not mention at all.
Both sites derive the *addressing step* from the *payload length* — sound
today because they are equal, silently wrong the moment a class stride exceeds
the format bytes. They would read and write the wrong slot, not a wrong
length, so the failure is data corruption rather than a size mismatch.

**Correction to the doc**: `chunk_ops.rs:2755` (listed as doc-site 3) is
**addressing-only** and must be left on `chunk_byte_stride`. Its length comes
from `gid_byte_range`, populated at A8 — so fixing A8 fixes the scatter
transitively, and "fixing" 2755 would break it.

Net: **8 sites to move to payload, 2 to move to stride, 1 to leave alone.**

### 0.5 — Test blast radius classification

| Verdict | Count | Sites |
|---|---|---|
| **MUST-PASS unmodified** | ~24 | `kv_stats_tests.rs:1734` (row-per-chunk table — the free oracle for the 1.5 re-index); all `ChunkGid` clone/drop/refcount tests in `types_tests.rs`; `gid_pool.rs` concurrency + tombstone tests; the four `chunk_ops.rs` round-trip byte-identity tests |
| **MUST-PASS, mechanical constant swap** | 2 | `types_tests.rs:93` `test_chunk_gid_arena_addressing`, `alloc_tests.rs:407-408` — both construct gids from `arena_gid_stride()`; swap to `GID_STRIDE` |
| **REWRITE** | ~22 | `arena_tests.rs` `arena_key_tests` (6) + `storage_policy_tests` (5) + `arena_storage_tests` (4) → assert **class** identity instead of format; `arena_tests::{test_arena_location, test_arena_to_arena_entry_cpu}`; `types_tests.rs` chunk-count formulas (2) → per-class budget; `chunk_ops_tests.rs` `ArenaKey::` ctors (11); `alloc_tests.rs` `ArenaKey::cpu_float` (3); `cold_warm_hot_path.rs:538` |
| **DELETE** | 9 | `backing_tests.rs::{test_k_arenas_empty_initially, test_v_arenas_empty_initially, test_float_arenas_empty_initially, test_quantized_arenas_returns_none_for_float}` (accessors deleted); `arena_tests.rs::arena_tests::{test_arena_float_creation, test_arena_kv_format_float, test_arena_float_kv_access, test_arena_as_float_k_v, test_arena_key_from_arena}` (Float/Quantized duality gone) |
| **DELETE at step 5** | 3 | `gid_pool.rs` `drain_plan_*` tests + `defrag_targets_exclude_protected_arenas` |

The `ArenaKey` rewrites are not busywork — they become the tests that
**formats sharing a class share a key**, which is the whole point of the
change and is currently untested because it is currently false.

---

## Step 1 — Size classes

### 1.1 — `size_class.rs`: **12/12 green**, first run

New module `candle-nn/src/kv_cache/chunked/size_class.rs`. Ladder, `SizeClass`,
`payload_bytes`, `class_for_format`, `GID_STRIDE = 1 << 16`. Pure addition —
nothing wired to it yet, so the gate is the unit tests alone.

| Test | Asserts |
|---|---|
| `ladder_is_strictly_increasing` | ordering, so `promote` is `+1` |
| `every_kv_format_maps_to_a_class` | **coverage** — all 22 quants + 4 float dtypes |
| `class_is_smallest_that_fits` | no format wastes a whole rung |
| `every_class_fits_u16_recycle_links` | `≤ 65_535`, **not** `< 65_536` — the sentinel is `chunks_per_region` itself |
| `gid_stride_exceeds_max_chunks_per_region` | 65_536 > 52_428 |
| `gid_stride_is_a_power_of_two` | decode is shift/mask |
| `payload_bytes_match_block_goldens` | **raw byte goldens**, all 26 formats, + exhaustiveness check against `all_kv_formats()` |
| `known_rounding_waste_is_unchanged` | F8E4M3 11.11 %, Q0 90 %, Q4_KS/Q8_KS/F16 0 % |
| `formats_sharing_a_class_are_fungible` | the whole point: the sub-320 tail collapses to one class; Q8_0≡Q8_KS; Q4_1≡Q4_KS; R16≡F32 |
| `promote_walks_up_and_stops_at_the_top` | scarcity promotion terminates |
| `region_capacity_never_overruns` | no slot straddles a region boundary |
| `payload_scales_with_sub_head_dim` | A9 — geometry is explicit, not assumed `== CHUNK_SIZE` |

Every hand-derived golden matched the real `bytes_per_block()` on the first
run, which independently confirms the §9 block-size table.

Max chunks/region is **52,428** (320 B class), leaving 13,108 of `GID_STRIDE`
headroom.

### 1.2 — Payload/stride split: **main gate GREEN, 16/16, 119.07 s**

Baseline was 119.37 s. Throughput is within noise on every config (largest
delta ‑2.2 % on F16×1 at 481.1 vs 489.4, offset by +5.9 % on the second BF16×1
at 613.8 vs 579.8 — run-to-run spread, not signal).

This is the payoff of the "introduce equal, migrate, then diverge" discipline:
the split landed across three crates and **provably changed nothing**, because
the two quantities were still equal when every consumer moved.

What moved:

- `ResolvedArenaInfo` gained `chunk_payload_bytes`, populated from the new
  `size_class::payload_bytes`. That makes 1.1's module load-bearing
  immediately — the gate passing is independent evidence that the ladder's
  byte arithmetic agrees with the hand-rolled ggml formula it replaced.
- **8 Category-A sites** switched to payload; **2 Category-B sites** switched
  to a stride derived from the arena via a new `slot_stride_of` helper (rather
  than a caller-passed value that could be wrong); `chunk_ops.rs:2755` left on
  stride, correctly, as addressing.
- New regression test `arena_byte_size_counts_payload_not_slot_stride` uses
  `arena_info_split(640, 320)` — stride ≠ payload — and asserts `byte_size`
  follows the payload. **This is the assertion the existing round-trip tests
  structurally cannot make** (A7): they read the same arena-derived length on
  both sides, so a symmetric error compares equal.

### 1.3 — Per-band format tags: compile + CPU green

`ChunkWindow` and `SealedChunk` gained `k_fmt`/`v_fmt` (`Arc<Vec<u8>>`,
`n_kv_head × N_PALETTE`), sharing the exact lifecycle of the existing
`k_pal`/`k_scale` fields — they describe how to read the bytes at `gids`, so
they travel with every gid mutation.

Population sites, by kind:

| Kind | Sites | Value |
|---|---|---|
| Fresh writer chunk | `alloc_block_chunks` | `BackingInner::active_{k,v}_fmt` — R16/F16 on GPU, shared `Arc`, no per-chunk allocation |
| Donor copy | 9 (scripted) | the donor's tags, same donor as `k_pal` |
| Cold-load / borrow | `ResolvedBlock`, `DonorMeta` | donor tags, falling back to the active formats |
| CoW partial, view | 2 tuples threaded | source tags (byte-for-byte copy into a same-format arena) |
| **Format decided** | `quantize_sealed_in_place` | the selection result, `override_{k,v}_quant` applied — the same expression `alloc_side` used to pick the destination arena |
| **Format decided** | `dequantize_sealed_in_place` | uniformly F16 |

**The safety net**: `ChunkWindow::debug_assert_tags_match_arenas`, called from
`write_record_for_chunk` — i.e. on every live chunk of every sequence on every
decode slot-state rebuild. During this window *both* answers to "what format is
this band?" exist and must agree; a construction site that forgot to propagate
a tag fails here rather than as a mis-decoded chunk after §1.5 removes the
arena's format. The helper is deleted in 1.5, when there is no second opinion
left to check against.

CPU suite 363/363. Workspace + tests compile under `--features cuda`. Clippy
warning count unchanged at **228 before and after** (a pre-existing repo-wide
baseline; this change adds none).

**Caveat, and it matters**: `debug_assert!` is compiled out under `--release`,
so the *main gate cannot exercise the tag check*. In 1.3 the tags are
write-only — nothing reads them yet — so the release gate has nothing to catch
either.

Resolved by testing the net itself rather than hoping a slow debug run reaches
it. `tag_assert_tests` (CPU, instant) proves all three behaviours:
`agreeing_tags_pass`, `unrecorded_tags_are_skipped` (empty tags mean "not yet
recorded" and must not trip — otherwise every `SealedChunk::for_test` would),
and **`disagreeing_{k,v}_tag_trips_the_assert`**, two `#[should_panic]` tests
proving the net actually fires. Without those the assert could have been
decorative and §1.5 would have landed blind.

**Main gate: GREEN, 16/16, 116.32 s** (baseline 119.37 s). Per-config deltas
vs baseline: F16 ‑0.2 %, BF16×1 +2.3 %, BF16×10 +1.0 %, Q8_0×20 ‑2.4 %,
Q4_0×4 +1.6 %, C0–C9 between ‑0.6 % and +1.9 %, Q4_0×20 ‑3.3 %. All noise.
CPU suite 367/367.

### 1.4 — Readers inverted: **main gate GREEN, 16/16, 120.00 s**

Baseline 119.37 s. Bulk t/s per config against the baseline table:

| Config | baseline | 1.4 | Δ |
|---|---|---|---|
| F16×1 | 489.4 | 493.7 | +0.9 % |
| BF16×1 | 539.8 | 548.4 | +1.6 % |
| BF16×10 | 1890.3 | 1893.2 | +0.2 % |
| Q8_0×20 | 2370.7 | 2325.2 | ‑1.9 % |
| Q4_0×4 | 1785.4 | 1853.0 | +3.8 % |
| C0–C9 | 1045–1073 | 1059–1086 | ‑0.8 % … +2.0 % |
| BF16×1 (2nd) | 579.8 | 603.0 | +4.0 % |
| Q4_0×20 | 2370.3 | 2325.9 | ‑1.9 % |

All inside the documented spread — the two BF16×1 configs differ by 7 % from
each other *within* the baseline run, so ±5 % on that config is intra-run
noise, not signal. Second gate 4/4. CPU suite 375/375 (367 + 8 new).

**This gate is the first real evidence the tags are correct**, and it is much
stronger than the 1.3 assert. `debug_assert!` is compiled out under
`--release`, so the tag check has never executed during a gate run. After 1.4
the attention and selection paths read band formats *only* from chunk tags —
so a wrong or missing tag mis-decodes KV and fails the config's validity
check. Sixteen green configs across `F16`/`BF16`/`Q8_0`/`Q4_0` and `C0–C7`,
`C9` therefore say the tags agree with reality on the production workload.
That is the proof 1.5 needs before deleting the arena's copy of the format;
without it, 1.5 would have landed on the strength of an assertion that never
ran.

Clippy, measured apples-to-apples (same command, both trees, `lib.rs` touched
so neither run served a cached result): **candle-nn 74 → 70 warnings**,
`lib test` 171 → 167, every other crate byte-identical. The reduction is real
— `ChunkRecordSrc` retired two `very complex type` warnings and one
`too many arguments`. (A whole-workspace count is *not* comparable between
runs: clippy re-emits warnings only for crates it recompiles, so editing
`candle-transformers` makes its cached test warnings reappear and the total
jumps ~10 for no reason. That artifact cost a detour; the per-crate line is
the number to read.)

#### Thirteen readers, not nine — and two of them were dead

The design doc listed nine; the resume brief corrected it to ten. The actual
sweep found **thirteen**, and the four the earlier counts missed are the
interesting ones:

| # | Site | Why it was missed |
|---|---|---|
| 2 | `serialize_kv_heads` (`meta_pool.rs`) | the resident-record twin of `KvHeadHost::from_gids`; identical `ai.k_format_tag` read, different file |
| 11 | `dequantize_sealed_in_place` bucketing (`compress.rs`) | the mirror of `bucket_quant_chunks`, in the *reverse* pass nobody was auditing |
| 12 | `resolve_src` closure (`compress.rs`) | resolved `(ptr, GgmlDType)` by matching on `arena.format()` — a format read that reads like a pointer resolve |
| 13 | the `[pal4]` verbose grid (`batched_inference.rs`) | a *diagnostic*, so out of every production-path sweep — but it is the tool that displays per-band format variety, and built from arena state it would have printed every band of a shared region identically. A grid that lies is worse than no grid |

The lesson repeats A1's: **audit the consumers of a value, not the paths you
expect it to travel.** Both misses in `compress.rs` are in the same file as a
site that *was* on the list.

**Site 4 collapsed into a deletion, and that is the strongest result of the
step.** `kv_formats_for_gids` walked arena keys to build `Vec<KvFormat>` for
three consumers — and all three immediately mapped the result through
`KvFormat::to_tag`, which is *defined* as
`ArenaFormatTag::from_kv_format(..).as_u8()`. That is byte-for-byte the
encoding the chunk already carries in `k_fmt`/`v_fmt`. So the persist path was
doing `arena → KvFormat → tag byte` to reconstruct a tag byte it was holding.
The function is deleted; the three sites copy `SealedChunk::format_tags()`.
Invariant 5 (byte-identical persistence) is preserved **by construction**
rather than by testing, and the new
`chunk_tag_bytes_are_the_persisted_format_encoding` test pins the identity
against raw byte values.

**Three deletions of code that should never have survived:**

- `ensure_writable_tail` (`sequence_ops.rs`) — **zero callers**, only a
  comment reference in a test. Worth recording *why* it was dead: its
  any-quantized probe asks whether any of the tail's gids sits in a
  `Quantized(_)` arena, and on GPU the active writer K arena is
  `Quantized(R16)`. The predicate was therefore unconditionally true, so the
  function would have pushed a fresh chunk on every call and never let a
  partial tail accumulate. It was not merely unused; it was unusable.
- `Cache::chunked_per_head_table_and_sync` (`cache.rs`) — zero callers, and a
  doc comment asserting it exists "for decode kernel consumption". False:
  attention reads `kvhead_k_ptr/k_fmt/k_scale` from the out-of-line `KvHead`
  record (`slot_types.cuh:187-221`) and never touches the per-head table. That
  sentence is the likely origin of the belief that the attention path depends
  on the selection table.
- `ArenaFormatTag::from_ggml_index` — zero callers, and a byte-for-byte
  duplicate of the tag-decode table I was about to add as `from_u8`. Its own
  doc comment contradicts its body (doc says GGML `Q4_0 = 2`, body maps
  `2 => BF16`). Two spellings of one table, one of them mislabelled, is
  exactly the drift the repo's single-source rule exists to prevent — so the
  correct move was to delete it rather than add a third.

#### Two sites deferred, with cause

Not everything on the list belongs in 1.4, and saying so beats forcing it:

- **`per_head_table_host`** — its row *is* the arena. Inverting it and
  re-keying it by `(chunk, head)` are the same edit, so it moves in the
  re-index (below).
- **`gpu_format_stats`** — aggregates over *pool keys*, not chunks. It has no
  chunk to read a tag from, and cannot be expressed until `ArenaKey` carries a
  class. It becomes class-occupancy logging in 1.9.

#### The 1.5 split — the re-index does **not** have to be atomic with the class switch

The design says the selection re-index "must land in the same commit as the
class switch, or selection silently reads wrong formats". That constraint is
**one-directional**: it forbids the class switch from *preceding* the
re-index. The re-index on its own is behaviour-neutral — populating each
palette sub-entry from that band's own gid produces the same pointers and the
same formats the palette-0 row produces today, just addressed by `(chunk,
head)` instead of `(arena, head)`. So 1.5 splits, and each half gates:

- **1.5a** — re-index only: `per_head_table_host` rows per `(chunk, head)`,
  `resolve_band_source` row key and `.palette[palette]`, `from_head_gids_multi`
  loses its `arena_offset`/`gid_off` rebasing.
- **1.5b** — the class switch: `ArenaKey → (SizeClass, ArenaLocation)`,
  `chunks_for_class`, `GID_STRIDE`, the validator port, and the deletion of
  the arena's format tags plus `debug_assert_tags_match_arenas`.

**And 1.5b cannot be split from 1.6.** This was not obvious from the plan, and
it is the one place the seven-step decomposition genuinely does not hold. The
moment `ArenaKey` stops carrying a `KvFormat`, `create_arena` has nothing to
build an `Arena::Float { dtype }` or `Arena::Quantized { format }` *with* — the
storage variant is chosen by the key's format today. Keeping a "representative
format" on the arena purely so the constructor still type-checks is precisely
the optionality-as-a-feature-flag the repo forbids, and it would leave a second
answer to "what format is this?" alive after 1.4 spent a whole step killing the
first. So the class key and the byte-slab collapse are one change: an arena
becomes `chunks × class_bytes` raw bytes, and the only thing the key says is
how wide a slot is. Budget them together.

**Correction to the resume brief: `kv_stats_tests.rs` is *not* a free oracle
that passes unmodified.** It does build row-per-chunk rows with `arena_idx`
aliased to `chunk_idx` — that part was right — but it populates `palette[0]`
with real data and **zeroes `palette[1..3]`**, with a comment saying only
palette 0 is read. That is true only because `per_head_lookup` currently
hard-codes `.palette[0]`. The moment 1.5a reads `.palette[palette]`, bands
1–3 dereference a zero pointer. The same pattern appears in **eleven** hand-built
table fixtures (`gpu_vs_cpu.rs`, `projection.rs` ×3, `projection_tables.rs` ×3,
`kv_selection_tests.rs` ×3, `kv_stats_tests.rs`): each ends
`row.extend_from_slice(&[0i64; 27])`. All eleven must replicate the real
sub-entry across all four palettes. They are not an oracle to lean on — they
are the first work item of 1.5a, and doing it makes them stronger tests than
they are today.

Blast radius re-measured for 1.5b/1.6, and A3's numbers hold:
`arena_gid_stride` 55 refs (8 in tests), `arena_chunks_for_format` 38 (4),
`ArenaKey::` 62 (28), `PagedKvArenas` 4 (1).

### 1.5a — Selection-table re-index: **main gate GREEN, 16/16, 119.04 s**

The selection table now emits **one row per `(chunk, head)`**, and each of the
row's four palette sub-entries is populated from *that band's own* gid (base
pointer + slot stride) and *that band's own* format tag. `resolve_band_source`
keys the row by the `chunk_idx` it was already being passed and takes
`.palette[palette]`; the gid supplies only the slot index within its region.

Behaviour-neutral by construction: with bands still one-arena-per-format, four
sub-entries built per band carry the same values the single palette-0 row
carried, so the kernel reads identical bytes. That is what makes this landable
*before* the class switch rather than atomically with it — see the note at the
end of the 1.4 section.

**Audit A8 said the sub-entry path "has never executed with non-identical
entries", so this gate is its first real exercise.** All 16 configs valid,
including every C-level (which is where adaptive per-band selection actually
produces per-band variety). 119.04 s against a 119.37 s baseline — the closest
match of the whole run; C-level bulk throughput 1039–1077 t/s against a
1045–1073 baseline.

New permanent tests, `tests/selection_table_tests.rs` (4/4, first run):

| Test | Asserts |
|---|---|
| `every_palette_sub_entry_carries_its_own_band_format` | **the load-bearing one** — one chunk with four *different* K formats and four *different* V formats must report all eight distinctly. A table built from arena state cannot pass it: the bands share an arena, so it would report one format four times |
| `rows_are_keyed_by_chunk_then_head` | two chunks do not alias; the row index is the one the kernel computes |
| `sub_entries_carry_a_real_slot_stride` | the stride is resolved from the arena, not defaulted — pinned to the exact byte value `32 × 8 × 2` for a BF16 backing, and byte offsets are zero because a slot *is* one band |
| `empty_job_list_yields_an_empty_table` | no garbage row |

#### Two bugs found by doing it, both invisible to the gate

**1. `select_chunks` shared the parent's table.** It narrows a
`PagedSelectionGpuInputs` to a chunk sub-range, slicing `head_gids` while
reusing `per_head_table_buf` **unchanged**. That was correct while rows were
keyed by arena — arena indices don't move when you drop chunks. Under a
`(chunk, head)` key it is silently wrong: the kernel's `chunk_idx` is relative
to the buffer it was handed, so chunk `start_chunk + i` would read the parent's
row `i`. Fixed by keeping the host table on the struct (exactly as `head_gids`
already was) and slicing both. **The gate would never have caught this** —
`select_chunks` has one caller, in a benchmark test.

This also removed a field: `_per_head_table_tensor` existed only to keep a
borrowed device allocation alive. Uploading the table the same way `head_gids`
is uploaded (owned, never the pinned-stager arena — same asynchronous-read
hazard, same reasoning) makes the tensor unnecessary.

**2. Eleven selection-table fixtures were stale, three of them by two years of
layout drift.** The resume brief said `kv_stats_tests.rs` was a "free oracle"
that would pass unmodified. It was not, and the reality was worse than the
correction:

- **Four fixtures** built correct 36-column rows but filled `palette[0]` and
  zeroed `palette[1..3]` (`row.extend_from_slice(&[0i64; 27])`), with a comment
  explaining that only palette 0 is read. True only because `per_head_lookup`
  hard-coded `.palette[0]`. Reading `.palette[palette]` turns bands 1–3 into a
  zero-pointer dereference.
- **Seven fixtures** (`gpu_vs_cpu.rs`, `projection.rs` ×3,
  `projection_tables.rs` ×3, `kv_selection_tests.rs` ×1) built **7-column**
  rows — the `PerHeadTableEntry` width from *before* the palette4 migration,
  which made rows 36 columns wide. Those fixtures have been feeding
  `select_kv_format_paged_batched_raw` a table whose stride is 5× too small,
  reading far out of bounds for every chunk after the first. They are all
  `#[ignore]`d and gated on an external data dump, so nothing has run them and
  nothing has complained.

All eleven now emit four populated sub-entries per row. The seven wide ones
also gained the two outer-scale columns the 7-value layout never had.

**The general lesson is about `#[ignore]`.** A test that cannot run is not a
test that is merely idle — it is a test that silently stops describing the
system, and it accumulates drift at exactly the rate the code changes. These
fixtures encoded a struct layout that had been dead for a major refactor. Worth
remembering when the deletion sweeps of step 5 start: an ignored test is not
evidence of anything.

#### 3. A real cold-load bug that 1.3 introduced and 1.4 armed

Chasing whether `SealedChunk::format_tags()`'s new fail-loud guard could fire in
production turned up a defect that had been sitting quietly since 1.3.

`SequenceState::set_block_gids` re-points a live window at different chunks. It
replaces `gids`, `k_pal`, `v_pal`, `k_scale`, `v_scale` and drops the resident
`meta` — and it **did not touch `k_fmt`/`v_fmt`**. Three call sites reach it:
the defrag remap (harmless: a remap moves a chunk between arenas of the same
format), and the two allocation paths that serve cold load and warm→hot
elevate, `alloc_sealed_block` and the bulk `ensure_blocks_from_specs`. Both of
those re-point a window that `alloc_block_chunks` had stamped with the **active
writer formats** — R16 for K, F16 for V on GPU — at sealed chunks in whatever
formats were persisted.

Before 1.4 that was invisible, because nothing read the tags. After 1.4 the
attention record and the selection table read *only* the tags, so every
cold-loaded chunk would have been decoded as raw R16/F16. Garbage KV, silently.

Neither gate covers it: audit A2 lists cold load as a blind spot of the model
test, and the round-trip tests do not exercise window re-pointing. The
`debug_assert_tags_match_arenas` net from 1.3 was built for exactly this bug —
and could not catch it, because release gates compile it out. That is now twice
that assert has been the right idea in the wrong place.

Fixed by threading the tags through `set_block_gids` and both sharded wrappers,
derived at the allocation sites from the specs' own `k_formats`/`v_formats`.
Because `KvFormat::to_tag` *is* the `ArenaFormatTag` discriminant, the cold-load
path now closes the loop from disk straight back to the window: the bytes the
substrate wrote are the bytes the window carries.

New regression test `alloc_sealed_block_stamps_the_destination_formats_on_the_window`
(`backing_tests.rs`), and — following the 1.3 discipline of proving the net
fires — the fix was temporarily reverted to confirm the test goes red on the
old behaviour. It does, with the message
`K tags must be the destination formats, not the active ones`.

**Final tree, everything landed: main gate GREEN 16/16 in 121.17 s**, second
gate 4/4, CPU suite 380/380, both cfg branches compile clean, clippy candle-nn
68 (from 74). Bulk throughput slightly *above* baseline on most configs
(C-levels 1055–1096 t/s vs 1045–1073; BF16×10 1939.6 vs 1890.3) — noise in the
favourable direction, not a claimed win. That run overlapped a clippy pass for
part of its duration, so treat 121.17 s as an upper bound; the clean 1.5a
measurement of 119.04 s is the timing of record.

#### 4. `--features cuda --tests` is not a compile check

The broken `set_block_gids` call in `types_tests.rs` compiled clean under
`cargo check --workspace --features cuda --tests` and failed only under a plain
`cargo test -p candle-nn --lib`. The enclosing test helper is
`#[cfg(not(feature = "cuda"))]`, so the cuda build never sees it.

**Both branches must be checked.** From here the routine is
`cargo check --workspace --features cuda --tests` *and*
`cargo check --workspace --tests`. This repo has enough `cfg(not(cuda))` test
scaffolding that a cuda-only check is a partial one, and the failure mode is a
non-cuda build that no gate exercises.

### 1.5b(i) — `Backing::Lease` in candle-core: built and **kept**

The first half of the class switch, and the only part that lives outside
`candle-nn`. E4 built this once as a throwaway and reverted it; this time it
stays.

Why it is a prerequisite rather than a nicety: once an arena is a byte slab, the
only consumers that still need a *typed* tensor over a slot are
`read_contiguous` / `write_contiguous` — and they are production
(`batched_inference.rs`, `prefill_utils.rs`, audit Q4). A slot is raw bytes;
those two need to see it as `(1, chunk_size, sub_head_dim)` of F16/BF16 or as a
quantized block run. `Tensor::from_raw_buffer` copies from a **host** slice, so
it would turn a device-local op into a round trip. A lease is the only
mechanism that gives a typed device view over bytes we own.

What landed:

- `CudaStorageSlice::Empty` — the tombstone. `CudaSlice::leak` takes `self` by
  value, so it is unreachable from `Drop::drop(&mut self)` without moving the
  slice out, and moving out of a `Drop` type requires putting something back.
- `Backing::{Owned, Lease}` on `CudaStorage`, and `impl Drop for CudaStorage`
  that `mem::replace`s the slice out and calls the per-variant `leak()`.
  Calling `leak` rather than merely suppressing the drop is load-bearing: it
  waits on the slice's read/write events, destroys them, and decrements the
  stream `Arc`. Bare suppression strands two `CudaEvent`s and a stream refcount
  **per lease**.
- `CudaStorage::from_leased_device_ptr` and `Tensor::from_leased_cuda_ptr`.

Measured cost, against E4's prediction of "18 match arms, 43 struct literals,
115 insertions":

| | E4 predicted | actual |
|---|---|---|
| non-exhaustive matches (E0004) | 18 | **20** (17 in `cuda_backend/mod.rs`, 3 in `utils.rs`) |
| struct literals (E0063) | 43 (35 core + 8 nn) | **52** (44 core + 8 nn) |
| files touched | 6 | 7 |

Close enough that the estimate was useful, and every single one was
compiler-enumerated — there is no silent-failure surface, exactly as E4 claimed.
Two arms needed a tuple pattern (`(CudaStorageSlice::Empty, _)`) rather than a
scalar one, and one match already had a `_` fallback; a mechanical sweep gets
those wrong, and the compiler said so both times.

One design choice worth recording: every match over the slice ends with
`CudaStorageSlice::Empty => CudaStorageSlice::unreachable_empty()`, a `#[cold]`
function returning `!`. Returning `!` lets **one** arm serve matches of every
result type, so the tombstone costs one line per match instead of a bespoke
error value each. It is genuinely unreachable — `Empty` exists only between the
`mem::replace` and the end of `drop` — so `unreachable!` is the honest
construct, not a placeholder.

**Five permanent tests** (`candle-core/tests/leased_storage_tests.rs`, 5/5 first
run):

| Test | Asserts |
|---|---|
| `lease_reads_the_owners_bytes` | a lease sees the memory it was pointed at |
| `writes_through_a_lease_reach_the_owner` | it is a *view*, not a copy — what `write_contiguous` depends on |
| `dropping_a_lease_leaves_the_owner_intact` | **the load-bearing one**: 64 lease/drop/allocate cycles against one owner, then every byte re-checked. A missing drop-suppression shows up as a use-after-free, and the interleaved allocation makes the freed slot likely to be handed straight back |
| `lease_travels_with_views` | `narrow`/reshape share the `Arc<Storage>`, so no derived tensor frees it |
| `a_slot_can_be_viewed_as_bytes` | the same memory read as `U8` yields the float's little-endian bytes — raw-byte golden `0x00 0x00 0x80 0x3F` for `1.0f32`. This is the property that lets **one** byte-slab arena serve every KV format |

`impl Drop` on `CudaStorage` is not free in principle — it forbids moving fields
out and adds a branch to every storage drop in the workspace — so this landed
behind a full gate rather than on the strength of "it compiles".

**Main gate: GREEN, 16/16, 114.21 s** — the fastest run of the whole initiative
(baseline 119.37 s; previous best 116.32 s). Every config at or above baseline:

| Config | baseline | with lease | Δ |
|---|---|---|---|
| F16×1 | 489.4 | 500.3 | +2.2 % |
| BF16×10 | 1890.3 | 1945.2 | +2.9 % |
| Q8_0×20 | 2370.7 | 2376.4 | +0.2 % |
| C0–C9 | 1045–1073 | 1055–1114 | all ≥ baseline |
| Q4_0×20 | 2370.3 | 2394.4 | +1.0 % |

So the `Drop` impl costs nothing measurable. Not "within noise" — *above*
baseline on every config, which is itself noise in the favourable direction.

#### The measurement trap fired a second time, and now has a signature

The **first** run of this gate reported **147.71 s** with Q8_0×20 at 1252 t/s —
a 47 % collapse — and BF16×10 down 14 %, while the low-context configs barely
moved. Entirely believable as a real `Drop` cost: more storages dropped at high
context, so a per-drop charge should hit exactly there.

It was self-inflicted. A `cargo check` and a second gate were running against
the same target directory during the measured window; one of them died with
`link.exe` error 1104 (file in use) and left two `cargo.exe` processes resident.
Re-measured on a verified-idle machine: 114.21 s.

That makes twice, and the two instances look identical:

| | 1.3 | 1.5b(i) |
|---|---|---|
| reported | 213.30 s (‑79 %) | 147.71 s (‑24 %) |
| worst config | Q8_0×20 ‑63 % | Q8_0×20 ‑47 % |
| low-context configs | barely moved | barely moved |
| actual cause | stray test binary on the GPU | concurrent cargo build |
| clean re-measure | 116.32 s | 114.21 s |

**The signature is: severe, non-uniform, monotonically worse with context
count, and Q8_0×20 the worst hit.** That is what CPU/GPU contention looks like
here, because the high-context configs are the longest-running and therefore
overlap a competing process for the largest fraction of their duration. A
genuine per-operation regression would scale with operation *count*, which also
rises with context — the two are not distinguishable from the table alone.

So the rule is not "interpret the shape", it is **"never interpret a gate that
shared the machine"**. Concretely, before every measurement:
`tasklist | grep -iE "candle_|cargo|rustc"` must be empty — not just
`candle_`, which is what the 1.3 rule said and which would have passed here.
And do not run *anything* — not a `cargo check`, not a second gate — while a
gate is measuring.

### 1.5b(ii-a) — u16 clamp + `GID_STRIDE`: **main gate GREEN, 16/16, 115.70 s**

Two of the class switch's prerequisites, landed and gated **independently** of
the arena collapse. Doing them separately is the point: inside a 177-error
change a failure would be ambiguous between the clamp, the stride, and the
byte-slab rewrite. Here each has its own green gate.

#### The §2.1 defect was real, and the design had it exactly right

`ArenaRefcounts.counts` is `Vec<AtomicU16>`; a **free** slot's word holds the
recycle-stack link, and the empty-stack sentinel is `arena_chunks` itself — so
the bound is `≤ 65_535`, **not** `< 65_536`. Recomputed from the real block
table before touching anything:

| Format | B/chunk | chunks | |
|---|---|---|---|
| Q0 | 32 | 524,288 | overflow |
| Q0_V, Q0_X | 64 | 262,144 | overflow |
| Q0_M2 | 96 | 174,762 | overflow |
| Q1_S | 160 | 104,857 | overflow |
| Q1_A | 192 | 87,381 | overflow |
| **Q0_M4** | 256 | **65,536** | **sentinel aliases slot 0** |

Seven formats, precisely the set §2.1 predicted. Any recycled slot above the
bound silently truncated its link — free-list corruption with no error and no
crash. **Q0_M4 is the one that matters for the bound's shape**: at exactly
65,536 the sentinel is unrepresentable in `u16` and wraps to 0, aliasing slot 0.
A `< 65_536` bound would have declared it safe.

Fixed with `MAX_CHUNKS_PER_ARENA = 65_535`, applied as a clamp inside
`arena_chunks_for_format`. The clamped formats get smaller-than-target arenas
(Q0: ~2 MiB rather than 16 MiB) — the right trade, since they are the
aggressive C9/C10 candidates and hold very few bytes per slot. The class ladder
retires the clamp entirely: a 320 B minimum caps chunks-per-region at 52,428 by
construction.

#### `GID_STRIDE` — the A15 win, now unblocked

With the clamp in place, `arena_gid_stride()` → `GID_STRIDE = 1 << 16` is safe
(65,535 < 65,536), so it landed in the same change. The old function folded
`arena_chunks_for_format` over all 22 `QuantFormat` variants plus three float
dtypes **through a `strum` iterator on every call**, and it is called from
`ChunkGid::clone`, `drop`, `arena_idx`, `chunk_idx` and `strong_count` — every
refcount operation in the system. It was not a `const fn`, so the fold was
plausible under LTO but unproven. 53 call sites converted; the duplicate
definition in `size_class.rs` collapsed into a re-export, so there is one
constant, guarded by
`const _: () = assert!(MAX_CHUNKS_PER_ARENA < GID_STRIDE)`.

A15 asked which of two things the win would be — a broad host-side drop (fold
was *not* happening) or nothing visible (fold *was* happening). The answer is
**nothing visible**: 115.70 s against a 119.37 s baseline, every config at or
above baseline (Q8_0×20 2387.7 vs 2370.7; C-levels 1040–1088 vs 1045–1073).
LLVM was almost certainly folding it. The change stands anyway — it removes the
question, and it is a precondition for the class switch regardless — but it
should not be recorded as a performance win, because it is not one.

#### Five permanent tests (CPU suite 380 → 385)

| Test | Asserts |
|---|---|
| `every_format_fits_a_u16_recycle_link` | every `QuantFormat` and float dtype is within the bound |
| `the_sentinel_is_representable` | pins `≤ 65_535` vs `< 65_536` — the distinction Q0_M4 turns on |
| `the_clamp_binds_exactly_the_known_hazard_set` | **the load-bearing one** — recomputes the *unclamped* counts from the real block table and asserts the over-bound set is exactly those seven names. If block sizes shift, the clamp's doc comment goes stale and this says so |
| `gid_stride_bounds_the_chunk_index` | power of two, strictly above the chunk bound |
| `clamped_formats_still_get_a_workable_arena` | the clamp shrinks the slab, it does not zero it; unclamped formats are untouched (F16 8192, Q8_0 15420) |

### Measurement hygiene — a real trap, hit and recorded

The **first** 1.3 gate run reported 213.30 s (‑79 %) with wildly non-uniform
per-config drops: Q8_0×20 fell 2370→867 (‑63 %) while Q4_0×4 barely moved
(‑6 %). That pattern — severe, non-uniform, worst at high context counts —
looked exactly like a per-chunk host cost from the new `Arc<Vec<u8>>` tag
allocations.

It was not. `TaskStop` on a `cargo test` task **kills the cargo wrapper but
leaves the spawned test binary running**: `candle_nn-bddceb0c0df9f32` (PID
64052, 513 MB) was still executing debug CUDA tests and contending for the GPU
throughout the measurement. Killed it; the identical build re-measured at
116.32 s.

Two rules follow, and they apply to every gate from here:

1. **Check for stray test binaries before measuring.** `tasklist | grep
   candle_` must be empty. Stopping a cargo task is not sufficient.
2. **Do not pipe a long background run through `grep`.** grep buffers when
   stdout is not a tty, so the log stays 0 bytes until the process exits and
   the run is completely opaque while it matters. Redirect to a file instead.

A 79 % regression that is really a dirty machine is the most expensive kind of
false signal: it is exactly believable enough to spend an afternoon
"optimising" a change that was never slow.

---

## Resume brief — everything needed to continue cold at 1.5b

Written so this file plus `arena_unification.md` are sufficient; nothing below
is recoverable from the code alone without re-deriving a day of reading.

### Working state

- **Nothing is committed.** The user's standing instruction for this build is:
  **do all the work, then present for review — do not commit per step.**
  `main` is 21 ahead of origin from the earlier doc-only commits.
- Line numbers in `arena_unification.md` predate 1.2/1.3/1.4 and have
  **shifted**. Re-derive by symbol name, never by line.
- Step 0, sub-steps 1.1–1.4, **1.5a**, and **1.5b(i) `Backing::Lease`** are done
  and gated green. What is left of step 1 is **1.5b(ii)** — `ArenaKey` → class
  and the `Arena` byte-slab collapse, which absorb 1.6 — then 1.7–1.9. The
  candle-core dependency is discharged; the rest is entirely inside
  `candle-nn/src/kv_cache/`.
- **Start 1.5b(ii) by writing the new `Arena` first**, then let the compiler
  enumerate. `Arena` becomes a struct, not an enum: `{ data: Tensor /* U8,
  (chunks, class_bytes) */, class: SizeClass, location, index }`. Address a slot
  as `base + idx * class.bytes()`; get a typed view for io.rs with
  `Tensor::from_leased_cuda_ptr`; `zero_chunk_at` becomes a byte write. The
  `float_data()` / `quantized_data()` accessors (39 + 23 references) collapse
  into one `byte_data()`, and `tensor_ptr_at_offset` / `qtensor_ptr_at_byte_offset`
  (8 + 9) collapse into one byte-offset helper.

### Environment facts worth not rediscovering

- Main gate is **~119 s**, not 20 min/config. Model is cached; no download.
- **The machine must be idle before any measurement**:
  `tasklist | grep -iE "candle_|cargo|rustc"` must be empty, and nothing may be
  started while a gate runs. Stopping a cargo task leaves its spawned test
  binary running; a concurrent `cargo check` fights for the target directory.
  Both have produced believable false regressions in this initiative — see the
  signature table in §1.5b(i).
- Never pipe a long background run through `grep` — it buffers and the log
  stays empty until exit. Redirect to a file.
- `cargo check -p candle-nn --features cuda --tests` is ~0.3 s warm; the CPU
  suite is ~0.6 s. Iteration is cheap. A **debug** CUDA build, by contrast, is
  many minutes — prefer testing an assert's own logic on CPU over waiting for a
  debug GPU run to reach it.
- **Clippy must be compared per crate, never as a workspace total.** Clippy
  re-emits warnings only for crates it recompiles, so a warm-cache workspace
  count moves by ~10 purely from *which* crates you touched. Measure by
  `touch`ing each crate's `lib.rs` and reading the
  `` `candle-nn` (lib) generated N warnings `` lines, on both trees, with the
  same command. Current: candle-nn 68 (down from 74 at the start of 1.4),
  candle-transformers 113, candle-conversation 16, candle-core 54. The bar is
  "unchanged"; this work has only removed warnings.
- `candle-examples/examples/perplexity-eval` **does not compile** (calls a
  `forward_batched` that no longer exists) and has not for some time. It is
  pre-existing; `cargo clippy --workspace --examples` reports 7 errors on a
  clean tree. Do not chase it.
- `cargo fmt` is dirty in `candle-core/src/vram/*` and `zend/repo_scan/*` from
  before this work — do not format those.

### 1.5b + 1.6 — the class switch, mechanics

**These are one change** (see the note at the end of the 1.4 section): the
moment `ArenaKey` drops its `KvFormat`, `create_arena` has no way to pick
between `Arena::Float { dtype }` and `Arena::Quantized { format }`, so the
byte-slab collapse has to land in the same commit. Everything below is one
work item.

What it consists of, all verified against the code:

**Both replacement types are already written**, staged in `docs/drafts/`
(see its README): `arena_byteslab.rs.draft` (the new `ArenaKey` + `Arena`) and
`arena_table_stripped.rs.draft` (`ArenaEntry` with its format tags removed).
Applying them and running `cargo check -p candle-nn --features cuda` yields the
enumeration below — **177 errors**, which is the real size of this step:

| File | Errors | Character |
|---|---|---|
| `chunk_ops.rs` | 80 | the hard half — migration/convert machinery, `quantize_into`, `dequantize`, per-format byte math. Each needs a decision, not a sweep |
| `arena.rs` | 35 | `ArenaStorageState` and leftovers of the old enum |
| `backing.rs` | 23 | mostly `key.format` → `key.class` |
| `alloc.rs` | 21 | `create_arena` shape + `ArenaKey::uniform` call sites |
| `compress.rs` | 17 | `alloc_side` and the `PalHeadDesc` pointer math |
| `io.rs` | 16 | the typed-view conversions — use `Arena::slot_view` |
| `sequence_ops.rs`, `gid_pool.rs` | 9 | mechanical |

1. **`ArenaKey` → `(SizeClass, ArenaLocation)`.** 62 references, 28 of them in
   test fixtures. `ArenaKey::gpu_quant(fmt)` / `gpu_float(dtype)` become
   `ArenaKey::for_format(fmt, elems_per_chunk, loc)?`. The one closure that
   creates the format⇒arena binding is `alloc_side` (`compress.rs`), run per
   `(chunk, head, side)`; because classes are coarser than formats, its
   contiguous-run eligibility test (`fmts.iter().all(|f| *f == fmts[0])`)
   *fires more often*, not less.
2. **`Arena` collapses to one byte-slab struct** with a class stride: `{ data:
   Tensor /* U8, (chunks, class_bytes) */, class, location, index }`.
   `zero_chunk_at` becomes a byte write to the **full stride** (invariant 4 —
   the next tenant may be any format), pointer resolution becomes
   `base + idx × stride`, and `Arena::slot_view(chunk_idx, dtype, shape)` hands
   io.rs a lease-backed typed view with a bounds check against the stride.
   Delete the `PagedKvArenas` trait and its impl — experiment E2 verified that
   removing both leaves `candle-nn` lib **and** tests compiling clean.
3. ~~**`Backing::Lease`** in candle-core.~~ **DONE — see §1.5b(i).** Landed,
   gated, and covered by five permanent tests including drop-safety and a
   raw-byte view golden. `Tensor::from_leased_cuda_ptr(ptr, dtype, shape, dev)`
   is the constructor the byte-slab arena will use to hand `read_contiguous` /
   `write_contiguous` a typed view over a slot. Nothing further is needed from
   candle-core.
4. ~~**`arena_gid_stride()` → `GID_STRIDE`**~~ **DONE — see §1.5b(ii-a)**,
   along with the `MAX_CHUNKS_PER_ARENA` clamp that made it safe. Still to do:
   **`arena_chunks_for_format` → `chunks_for_class`** (38 refs, 4 in tests).
   Note A15 is settled: the stride change measured as *no* win, so LLVM was
   folding the `strum` iterator. Do not budget it as a speedup.
5. **Port `validate_selection_gids`**, do not delete it — bound becomes
   `chunks_for_class`. Its `chunk_idx`-vs-capacity arm exists for a
   sanitizer-confirmed OOB read at slab end, and a region re-stamped to a
   different class has a different stride: the same re-tenancy hazard renamed
   (audit A14). It reads like format bookkeeping and is actually a
   memory-safety net — exactly the shape of thing a deletion sweep removes by
   association.
6. **Delete** `ArenaEntry`'s per-arena `k_format_tag` / `v_format_tag`,
   `actual_kv_format_tags`, and `ChunkWindow::debug_assert_tags_match_arenas`
   (plus its four `tag_assert_tests`). At that point there is no second opinion
   left to check against, which is the whole reason 1.3 was a separate step.

Test classification for this change is already done — see §0.5 above: ~24
must-pass unmodified, 2 mechanical constant swaps, ~22 rewrites, 9 deletions.
The `ArenaKey` rewrites are not busywork: they become the tests that **formats
sharing a class share a key**, which is the point of the change and is
currently untested because it is currently false.

### What makes it safe

- `resolve_arena_info` is the **only** producer of `ResolvedArenaInfo`, so the
  payload/stride pair has one authority — and 1.2 already split them, so the
  copy lengths are format-derived while the addressing is stride-derived.
- The selection table already carries per-band pointers, strides and formats
  (1.5a), so collapsing `arena_idx` changes nothing it reads.
- **The 1.4 gate, not the 1.3 assert.** An earlier draft of this brief claimed
  the `debug_assert_tags_match_arenas` net had "been live through a full green
  gate", so the tags were proven. That was wrong: `debug_assert!` is compiled
  out under `--release`, and every gate run is a release build — the net has
  never fired during a gate. The real proof arrived in 1.4: with the readers
  inverted, formats come *only* from chunk tags, so 16 green configs are direct
  evidence the tags are right on the production workload. This step deletes the
  arena's copy against that evidence.


---

## 1.5b(ii) — the class switch: **main gate GREEN, 16/16, 117.02 s**

The whole of step 1 is done. `ArenaKey` is `(SizeClass, ArenaLocation)`, an
arena is a flat run of untyped byte slots, and a band's format lives only on
its chunk.

| Gate | Result |
|---|---|
| Main gate | **GREEN**, 16/16 configs, **117.02 s** (baseline 119.37 s) |
| candle-nn GPU suite | **409 passed**, 0 failed, 21 ignored |
| CPU suite | **378 passed**, 0 failed (was 385 — see the test accounting below) |
| `leased_storage_tests` | 5/5 |
| `cargo check --workspace --features cuda --tests` | clean |
| `cargo check --workspace --tests` | clean |
| clippy | candle-core 54, candle-nn **68**, candle-transformers 113, candle-conversation 16 — **all at baseline** |
| `candle-core/src/vram/` | zero delta from HEAD |

### The blocker, and what it actually was

The captured 174-error enumeration listed `quantized_data()` as part of a
`byte_data()` sweep. **It is not.** `byte_data()` yields raw bytes;
`quantize_into` needs a `QTensor` to write blocks *through*. `Backing::Lease`
(1.5b(i)) covered plain `Tensor` only, so the quantize / dequantize paths had
no way to reach a slot at all.

That is the second time in this initiative a *count* of call sites was mistaken
for an *understanding* of what they require (the first was 1.4's "thirteen
readers, not nine"). The enumeration was right about where the work was and
wrong about what it was.

### `QTensor::from_leased_cuda_ptr` — the missing half

`QCudaStorage` is `{ PaddedCudaSlice, GgmlDType, CudaDevice }` and
`PaddedCudaSlice` wraps a `CudaSlice<u8>` — the same shape `Backing::Lease`
already handled. So the fix is symmetric with 1.5b(i): a `backing: Backing`
field, a `Drop` that `leak()`s rather than frees, and an `unsafe` constructor
over a device pointer. `Arena::qslot_view` builds one per slot.

Two details that are load-bearing:

- **`Clone` had to become manual, returning `Backing::Owned`.**
  `CudaSlice::clone` is a device-to-device **copy**, so a clone of a lease has
  its own allocation; carrying `Lease` across would leak it on every clone.
- **The view carries no matrix-row padding**, unlike `QCudaStorage::zeros`. A
  slot is exactly its own bytes and the next slot belongs to another chunk.
  That makes the view unusable as a matmul operand and correct for the block
  quantize / dequantize kernels, which do not read past the data.

### A live bug the lease exposes

`write_contiguous`'s quantized arm did:

```rust
let mut qt = data.clone();          // <- device-to-device COPY
qt.quantize_into(&k_band, elem_offset)?;
```

Since `CudaSlice::clone` deep-copies, that quantized into a **throwaway** and
dropped it. The write was lost. It survived because the arm is only reachable
when the active writer chunk is quantized and `write_contiguous` is itself a
test / CPU-fallback path — the production KV write goes through the prefill and
decode kernels. The new code writes through a lease, so the same expression is
now correct by construction rather than by not being reached.

### What collapsed

The byte-slab arena did not just replace code, it removed the *reason* for most
of it. Net: **‑1,057 lines** across the crate.

| Was | Is | Why |
|---|---|---|
| `migrate_chunk` + `copy_chunk_data_static` + `convert_chunk_data_static` (~320 lines of per-`(location, format)` dispatch) | `migrate_chunk` + `copy_slot_bytes` (~40 lines) | Every production caller — hot→warm demote, warm→hot elevate, fork — is **format-preserving**. The format-changing arms were reachable **only from tests**. A relocation now *requires* equal classes, which is what makes the verbatim copy safe |
| `write_raw_sealed_chunk`'s 280-line per-dtype upload ladder (R16 memcpy / ggml re-wrap / one arm per float width) | one byte write per slot | The bytes arriving are already the format's own image — that is what "raw" means. The ladder existed only because the destination was typed |
| `tensor_ptr_at_offset` (scaled an element offset by a dtype width) + `qtensor_ptr_at_byte_offset` (took a byte offset from a ggml block layout) | `Arena::slot_ptr` | Both were reconstructing `base + idx * stride`, which a byte slab hands over directly |
| `fork_sequence`'s two-pass copy with a Quantized→Float dequantize arm | `copy_slot_bytes` | A fork keeps each band's format, so it keeps its class, so it is a copy |
| `ArenaEntry` + `to_tensor_row` / `from_tensor_row` / `encode_metadata`, and the CUDA `ArenaTableEntry` struct with its eight accessors | deleted | After 1.5a the only surviving consumer wanted `.k_ptr`; the rest had **zero** callers on either side |
| `PagedKvArenas`, `float_arenas`, `quantized_arenas`, `k_arenas`, `v_arenas`, `count_quantized_arenas`, `actual_kv_format_tags`, `debug_assert_tags_match_arenas`, `chunked_live_chunks_as_sealed_with` | deleted | E2's prediction held: nothing outside tests referenced any of them |
| gid pool table: ~58 entries (2 locations × 29 formats) | **14** (2 × 7 classes) | *This collapse is the initiative.* A slot freed by any format is now allocatable by every format sharing its class |

### Two bugs found by doing it

**1. `read_raw_sealed_chunk` depended on an allocation coincidence.** It sized
its read as `chunk_size * head_dim` — a whole head — and issued it against
**palette 0's slot alone**, reaching the other three only because
`alloc_chunk_run_for_key` *usually* lays a head's palettes out contiguously. It
does not always: a mixed-format band group falls through to per-band
allocation, and the read then walked into unrelated chunks. The slot bounds
check turned that silent dependency into a loud failure on the first GPU run.
Fixed by reading each band from its own gid — the same correction
`resolve_band_source` already made kernel-side.

**2. `ResolvedArenaInfo` was answering three questions it had no business
answering.** `chunk_payload_bytes`, `k_format_tag` and `v_format_tag` are all
properties of a *chunk*, not of a run of byte slots. They are gone; the struct
is `base_ptr`, `chunk_byte_stride` (the class stride) and `chunk_capacity`.
Every copy length in `migrate.rs`, `chunk_ops.rs`, `head_gids.rs` and
`transfer.rs` now goes through `payload_bytes_for_tag`.

### New single-source pieces

- **`ArenaFormatTag::to_kv_format`** — the inverse of `from_kv_format`, and now
  load-bearing: with arenas untyped, the tag is the only path from a persisted
  byte back to a byte length. Three tests pin it: round-trip over
  `all_kv_formats()`, agreement with `payload_bytes` from the format side, and
  `None` for the tags that name no storage format (`Invalid`, the GGML
  K-quants, `P2`, the QAWQ pair, `F8E5M2`).
- **`head_gids::band_tags`** — the one place the `[h * N_PALETTE + p]` indexing
  is written down. `SealedChunk::bands` and `ChunkWindow::bands` both delegate,
  so a live chunk and the sealed chunk it becomes cannot disagree about which
  tag belongs to which slot.
- **`gpu_class_stats`** (1.9's item, pulled forward because the float/quant
  split is no longer computable): per-rung
  `ClassOccupancy { slot_bytes, arenas, reserved_bytes, live_bytes }`, wired
  through `memory_report.rs` and the scheduler's `kv-pool` line.

### The slab is one-dimensional, and that was a real decision

The draft had `(chunks, stride)`. A band's payload is generally shorter than
its slot, so **every** access is a byte *range*; on a 1-D slab
`narrow(0, off, len)` and `slice_set(src, 0, off)` express that contiguously on
either device. A 2-D slab forces every partial write to either pad up to the
full stride — moving pad over PCIe, which invariant 8 forbids — or to go
through raw pointer writes.

### Test accounting: 385 → 378 on CPU, and why that is not a loss

| Change | Δ |
|---|---|
| `arena_tests.rs` rewritten: format-identity tests became **class-identity** tests, including `formats_sharing_a_class_share_a_key` — the property the initiative exists for, which was previously untestable because it was false | ‑6 |
| `tag_assert_tests` (4) deleted with `debug_assert_tags_match_arenas`; replaced by `band_iteration_tests` (3) pinning the K/V interleave and live-vs-sealed agreement | ‑1 |
| `head_gids` byte-size tests rewritten against tags; gained `arena_byte_size_follows_each_band_own_format`, which an arena-derived length **cannot express at all** | +1 |
| The two format-conversion round-trips (R16, Q8_0) seeded themselves through a converting `migrate_chunk` and cannot exist | ‑2 |
| `a_band_narrower_than_its_slot_round_trips_its_payload` — a Q8_0 band (1088 B) in a 1152 B slot, seeded through `write_raw_sealed_chunk` and round-tripped hot→warm→hot→warm | +1 |
| Backing accessor tests deleted with their accessors | ‑4 |
| `size_class` tag round-trip / payload-agreement / unmapped-tag tests | +3 |

The last one is worth stating plainly: **the two tests that were removed could
not have caught the hazard size classes introduce.** They read the same
arena-derived length on both sides of the trip, so a symmetric error compared
equal (audit A7). The replacement asserts on the payload alone, with the stride
deliberately wider.

### Correction to the resume brief

It said the enumeration was "four repeated patterns, not 174 decisions". The
patterns were real, but two of the four were mis-specified: `quantized_data()`
needed a primitive that did not exist, and `float_data()`'s replacement needed
a CPU decode path (`decode_bytes` / `encode_bytes`) because a slab cannot be
reinterpreted in place on the host. Neither was visible from the error text.


---

## 1.7 — Scarcity-only class promotion: **main gate GREEN, 16/16**

`SizeClass::promote` is now wired into every claim path. The order is: a free
slot in the class; else stamp a region for it; else take a free slot from a
wider class that already has one.

### The correction that matters

The first implementation walked *up the ladder stamping regions* — try the
class, fail, try the next class, fail, and so on. That is wrong twice over:

1. **It can never succeed.** Every class's region is the same
   `TARGET_ARENA_BYTES`. If 16 MiB cannot be had for a 320 B class it cannot be
   had for a 4096 B class either.
2. **It amplifies the failure sevenfold.** `ensure_vram_budget` calls
   `request_global_compact()` on refusal, so one refused region became seven
   global compactions.

What §3.4 actually asks for is the opposite motion: *avoid stamping* by reusing
an already-stamped wider region — "this stops a trickle of rare formats from
stamping a whole 16 MiB region for a class that will never fill it". Promotion
takes an existing free slot or it does not happen.

The registration is also rolled back when a region cannot be materialised.
Without that, the pool advertises free slots storage cannot produce: every
later claim into that arena fails identically while `total_arenas` inflates the
occupancy diagnostic. That bug predates this step.

### Tests

`ordinary_allocation_never_promotes` is the gate on the gate: with regions
freely available, `class_promotion_count()` must not move however many chunks
are claimed. If promotion ever becomes a background mixing vector, the
per-class occupancy numbers stop meaning anything and step 6 has nothing to
tune against.

---

## 1.8 — `GpuChunks` to the region tier: **GREEN but 124.47 s — a 6 % REGRESSION**

Landed and correct, but it **did not deliver A13's predicted win**, and the
honest summary is that it costs more than it saves on the current allocator.
Recorded in full because the failure is more informative than the mechanism.

### What was built

`slot_state_arena` — a doubling class family (4 KiB … 1 MiB) with per-class
free lists over slabs, plus `GpuChunks` holding a slot instead of an
allocation. `clear` returns the slot to its free list; `resize` promotes to a
wider slot when the current one no longer fits.

**Promotion copies nothing.** A `copy_slot` helper was written and then deleted
when its only caller made it unreachable: `rebuild_decode` is the sole caller
of `resize` and rewrites every entry immediately afterwards. The buffer's two
sections — `[ slice headers | records ]` — mean the records half *moves*
whenever the entry count changes, so the old "preserve `min(old, new)` bytes,
zero the rest" was producing bytes nobody read.

**Releasing needs no fence.** Every sequence takes its stream from
`CudaDevice::cuda_stream()` — the device's *primary* stream — so a slot handed
straight back out cannot be written before the copies that drained it have run.
That is what let the `stream.synchronize()` disappear rather than merely move.

### The measurement, in three acts

| Attempt | Total | Q8_0 single | C0 single |
|---|---|---|---|
| Baseline (step 1 without 1.7/1.8) | **117.02 s** | 162.9 | 39.2 |
| First 1.8 | 152.71 s | 149.3 | 28.2 |
| + host capacity on the ladder, fence restored | 141.92 s | 168.4 | 36.5 |
| + slabs sized by slot count, not by region | **124.47 s** | 167.3 | 36.1 |

**Defect 1 — the host buffer still churned, and unsafely.** It grew to exactly
`byte_len`, so `push_chunk` still paid a `cuMemFreeHost` + `cuMemHostAlloc`
pair per layer — the churn the change existed to remove. Worse, the old pinned
buffer was dropped without the stream fence the previous code had, unpinning
pages an in-flight `memcpy_htod` may still have been sourcing. Fixed by sizing
the host capacity on the *same doubling ladder* as the device slot: the
reallocation becomes logarithmic in depth, which is what makes keeping the
fence affordable.

**Defect 2 — a whole 16 MiB region per class.** The entire slot-state working
set is a few MiB; giving each touched class a region put tens of MiB of
mostly-empty slabs in front of KV on a 16 GiB card, and the `ensure_vram_budget`
compaction that triggered cost far more than the churn removed. Slabs are now
sized by slot *count* (512, capped at a region).

### The 6 % that remains is unexplained

Per-config throughput is at or above baseline — Q8_0 single is **167.3 vs
162.9** — yet wall-clock is 124.47 s against a 114–120 s historical band. The
cost is therefore **outside token throughput**: setup, prefill, persistence or
teardown. Diagnosing it needs a profile, not another guess, and three guesses
have already been spent here.

### The strategic question this raises

The design places the region tier **inside the reservation** (§3.2, step 4).
Built before the reservation exists, it has nowhere to live but the CUDA pool —
so it competes with KV for exactly the memory step 4 stops contending over.
Both defects above were versions of that same problem.

So there is a real case that 1.8 is **premature**, and that its slabs should
come from the reservation rather than the pool. Two options, and this is a
judgement call for the author:

- **Keep it** and accept ~6 % until step 4 moves the slabs into the
  reservation, where the contention disappears by construction.
- **Revert it** to `docs/drafts/` and re-land after step 4, taking the gate
  back to 117 s in the meantime.

The mechanism is right — A13's ~3,000 alloc/free/sync cycles per 32 decoded
tokens are gone, and `class_promotion_count()` / `slot_state_stats()` make that
observable. What is wrong is the *home* its slabs currently have.


---

## 1.9 — A12 tally + doc reconciliation: **STEP 1 COMPLETE**

Final gate on the whole of step 1: **16/16, 132.07 s**. Suites: candle-nn GPU
**418 passed**, CPU **383 passed**, lease tests 5/5, both cfg branches clean,
clippy at baseline on every crate (core 54, nn 68, transformers 113,
conversation 16), `candle-core/src/vram/` zero delta.

### A12 is now answerable

`ChunkedKvBacking::class_histogram(batch_idx)` returns live band slots per size
class and, separately, those in formats **narrower than the 320 B floor** —
the numerator of A12's rule. Splitting the low end into {64, 160, 320} pays for
its two extra partial tails (≈16 MiB) only above ~65–84 K such slots, ≈2 % of a
~4.8 M-slot pool. Expected to fail at the C4/C5 production default and to be
worth re-checking at C9/C10; step 6 makes the call.

Paired with `gpu_class_stats` (per-rung occupancy from the pool) and
`class_promotion_count` / `slot_state_stats`, the ladder's shape is now fully
observable — which was the point of the step, since §3.4 explicitly calls the
small end "a bet, not a fact".

### The design doc now records where the plan and reality diverged

§5 step 1 gained a "Step 1 as built" block: predictions that held (class
coverage, the 58 → 14 pool collapse, run eligibility firing more often, clean
`PagedKvArenas` deletion), predictions that did not (`GID_STRIDE` bought
nothing; `GpuChunks` on the region tier is currently a cost), and the four
things the plan did not know about (the `QTensor` lease, `migrate_chunk` never
needing to convert, promotion having to reuse rather than re-stamp, and two
allocation-order dependencies the slot bounds check surfaced).

### Honest final numbers on 1.8's cost

The gate ran four times across the 1.8 work: **152.71 → 141.92 → 124.47 →
132.07 s** against a baseline band of 114–120 s over eight prior runs.

Two things follow, and both are worth stating rather than averaging away:

1. **The residual cost is ~8–12 %, not the 6 % a single run suggested.** One
   measurement was not enough to characterise it.
2. **The variance itself grew.** 124.47 and 132.07 came from identical trees on
   an idle machine; the baseline never spread that far. Slabs contending with
   KV for the same pool make timing less predictable, which is consistent with
   the diagnosis and is another reason the fix is step 4's reservation rather
   than more tuning here.

Per-config throughput remains at or near baseline throughout (Q8_0 single
155.7–168.4 against a 162.9 baseline), so the cost stays outside token
throughput. **Do not tune this further without a profile** — three guesses have
already been spent, two of them productive and the third not.

### Where step 1 leaves the engine

Formats are chunk metadata. The allocator hands out fixed-stride byte slots and
knows nothing about what occupies them. Free slots are fungible across every
format sharing a class. None of that has delivered a VRAM win yet — it is not
supposed to; §1.3's ~2 GiB comes from steps 2–4 replacing the allocator with a
static reservation, and step 1's own gate condition was "no regression".

On that gate condition, step 1 passes everywhere except 1.8, which is carried
deliberately: reverting it would mean rebuilding the same mechanism against a
moved target after step 4, and its cost is bounded and understood.


---

## Step 2 (partial) — `BumpArena` + the persistence domain: **GREEN, 139.57 s**

The transient tier exists and has its first real consumer. **This is not all of
step 2** — see "what is left" below.

| Gate | Result |
|---|---|
| Main gate | 16/16, 139.57 s |
| candle-nn GPU suite | **420 passed**, 0 failed |
| CPU suite | **383 passed**, 0 failed |
| Both cfg branches, workspace + tests | clean |
| clippy | candle-core 54, candle-nn 68 — at baseline |

### What landed

**`bump_arena`** — per-domain device bump allocators. A generation hands out
disjoint ranges by advancing a cursor and later resets it in one store; nothing
is ever individually freed.

Two things guard the reset, and they are the whole safety argument:

- **A counted generation.** The cursor cannot move while any `Generation` guard
  is live. Checked, not assumed — a silent early reset is a data race that
  reproduces as garbage output far from its cause (principle 7).
- **A stream fence.** The last guard to drop synchronises the domain's stream
  before rewinding, so the GPU has drained the ranges the host is about to hand
  out again. `PinnedStager`'s sync-then-reset discipline, applied to device
  memory. If the fence *fails*, the reset is skipped and the span leaks for one
  generation — refusing beats scribbling.

`BumpRange` is deliberately **not** RAII and carries no lifetime. A bump range
is freed by its generation's reset, so a `Drop` impl would be a lie, and the
compiler cannot express "valid until the generation resets" — pretending
otherwise with a borrow would force every consumer to thread a lifetime the
counted generation already enforces at run time.

**The persistence domain** is the first consumer: all three migration staging
sites (`migrate_sealed_to_cpu_batch_async`, the layer-batch variant, and
`migrate_sealed_to_gpu_batch_async`) now bump-allocate instead of calling
`copy_stream.alloc` and freeing at scope end. Its span is
`MIGRATION_STAGING_CAP_BYTES` — already the cap every batch bisected against,
so the budget is unchanged and now *ours* rather than the driver's. The
adaptive layer-batch bisect is untouched in shape; it just shrinks against a
declared budget instead of against a refused allocation.

**Per-domain peaks are logged** (`persistence_domain_stats`), which is step 2's
stated gate deliverable: step 4 sizes the transient span from them —
`S = 2·W_wave + W_persist + shelf`.

`BumpArena::seal` and the no-mid-wave-allocation assert were written and then
**removed**: they belong to the wave domain, which step 3 introduces. Carrying
them dead through a step would be exactly the placeholder the repo forbids.

### What is left of step 2

The design lists candidates in order of confidence; only the second group is
done. Still to port:

- kernel argument/metadata blobs: chunk-meta rows, per-head tables, head-gid
  uploads, selection tables (`PagedSelectionGpuInputs`), migration descriptors;
- **logits** (A11: wave-scoped, and `BatchedSampler` holds no device state);
- grow-only scratches (`ProvSignScratch`, `KvSamplerGpu`, MoE routing) to the
  static shelf;
- the **wave domain** itself: A/B halves, the buffer set allocated at wave
  start, and the seal assert.

### ⚠ The measurement has stopped being trustworthy

Gate totals across this session's later runs, all on an idle machine:

| Tree | Total |
|---|---|
| Step 1 without 1.7/1.8 | 117.02 s |
| + 1.7/1.8 (four runs) | 152.71 → 141.92 → 124.47 → 132.07 s |
| + step 1.9 | 132.07 s |
| + step 2 (this) | 139.57 s |

**The spread on identical trees (124.47 vs 132.07) is now comparable to the
differences between trees.** Per-config throughput is flat to noisy in the same
way — C0 single ranges 28.2–39.2 across runs of code that did not touch the
decode path.

That means the gate can no longer distinguish a real 5–10 % regression from
run-to-run variance, and every conclusion drawn from a single run below that
threshold is unsound. **Step 3 should not start until this is fixed**, because
its whole point is moving inference-loop intermediates and its gate is
performance.

Two candidates for the cause, in order of suspicion:

1. **Pool contention from the region-tier slabs and now the transient span** —
   both are new fixed allocations competing with KV on a 16 GiB card, and both
   disappear into the reservation at step 4.
2. **Thermal or clock drift** across a long session of back-to-back 2-minute
   CUDA runs — never characterised on this machine.

Distinguishing them is cheap and should be done first: re-run the *step-1
without 1.7/1.8* tree now and see whether it still measures 117 s. If it does
not, the drift is environmental and the last several comparisons need redoing
with interleaved A/B runs rather than sequential ones.


### Correction: the "1.8 regression" was mostly thermal drift

Three back-to-back gate runs on an **unchanged** tree: **117.01 → 133.13 →
137.46 s**, monotonically rising. The machine warms across a session of
back-to-back 2-minute CUDA runs, and the first run after an idle period is the
fast one.

That invalidates the attribution above. 1.8's four measurements
(152.71 → 141.92 → 124.47 → 132.07) were taken sequentially against a 117 s
baseline measured *cold*, so an unknown part of that gap — plausibly most of
it — is drift, not code. The two real defects found while chasing it (the
exact-size host regrowth without a fence, and whole-region slabs) were genuine
bugs worth fixing on their own merits, and the first version's 152 s was far
outside any drift band. But **"1.8 costs 8-12 %" is not supported by this
data**, and no decision should rest on it.

Rule for the rest of this initiative: **compare only interleaved A/B runs, or
runs from the same thermal state.** A sequential before/after on this machine
measures the clock as much as the code.


### Step 2 continued — selection-table uploads on the transient span

`stage_bytes_as_gpu_buf` now has three paths in preference order: a
caller-supplied pinned `Generation` (host-mapped, no device copy at all — still
the best option), the **persistence domain's transient span** (one bump + one
H2D), and a plain allocation only when there is no stream to bump against
(CPU-backed tests). The middle path replaces `dev.memcpy_stod`, which allocated
and freed a device buffer per selection table per call.

`GpuBuf::from_borrowed` already existed, so the buffers borrow the span rather
than owning anything. What that costs is a lifetime obligation:
`PagedSelectionGpuInputs` now holds a `bump_arena::Generation`, because its
`GpuBuf`s point into the span and the cursor must not rewind under them. The
guard is `Some` exactly when the bump path was taken — the three constructors
that stage with `None` always hold one; the one that may receive a pinned
generation holds one only when it did not.

**Gate: 16/16, 126.28 s.** candle-nn GPU suite 420/420, CPU 383/383, both cfg
branches clean, clippy at baseline (core 54, nn 68).

### Step 2's remaining work

- **The wave domain** — A/B halves, the buffer set allocated at wave start, and
  the `seal` assert. This needs a wave-start/wave-end hook in the scheduler's
  loop; `BumpArena` already supports everything else it requires.
- Chunk-meta rows, per-head tables and head-gid uploads on the *decode* path
  (the ones ported here are the persistence thread's).
- **Logits** (A11) and the grow-only scratches (`ProvSignScratch`,
  `KvSamplerGpu`, MoE routing) to the static shelf.

None of these is blocked on anything; they are the wave-lifecycle half of the
step, and they want the boundary hook built first.


### Step 2 reassessed after re-reading §3.6 / §3.7 — it is closer to done than the plan implies

The candidate list in step 2 reads as five groups of work. Checked against the
code, **two of them are already done and one belongs to a later step**:

| Candidate | State |
|---|---|
| Migration staging (`copy_stream.alloc`, ≤512 MiB cap) | **Done** — persistence domain |
| Selection tables (`PagedSelectionGpuInputs`) | **Done** — persistence domain |
| Kernel argument / metadata blobs: chunk-meta rows, per-head tables, head-gid uploads | **Already on a bump allocator** — `PinnedStager`, opened per forward by `begin_stager_generation` (`batched_inference.rs:3208`) |
| Logits (A11) | Step 3's territory — see below |
| Grow-only scratches → static shelf | Needs the reservation; step 4 |

The third row is the one the plan obscures. §3.7 already says it: "the pinned
instance keeps its role (zero-copy PCIe reads of small descriptors); the
unification is the allocator and its safety rules, not the memories." The
metadata blobs are **host-pinned and device-mapped** — the GPU reads them over
PCIe with no device buffer at all — which is strictly better than a device bump
range for descriptors of this size. Moving them would be a regression dressed
as progress.

So the device wave domain has, right now, **no step-2 consumer left**. Its
consumers are the per-forward kernel *output* buffers — `prefill_utils.rs`
lines 1038, 1368, 1781, 1875 are all `dev.alloc` for a kernel's destination —
and those are exactly step 3's "intermediate buffers in the inference loop".

**Decision: do not build the wave domain speculatively.** Its A/B halves, the
buffer set, and the `seal` assert land in step 3 *with* the consumers that give
them shape, rather than sitting dead through a step boundary. `BumpArena`
already carries everything they need; `begin_stager_generation` is the wave
boundary they will hook, and it is a candle-transformers call — **no scheduler
edit is required**, which the earlier note wrongly assumed.

### One thing step 3 must re-check before it starts

§3.7 drafts `ArenaLease = {generation_id, Arc<AtomicUsize> live_count}`.
`Backing::Lease` as built is a bare marker, with the count living on
`BumpArena::Generation` instead — sufficient for the region tier (a region is
never reset under a live chunk) and for staging (the guard's scope encloses the
copies).

Wave intermediates break that: handed to candle ops as `Tensor`s, they outlive
the scope that opened the generation. Either the lease starts carrying the
count as originally drafted, or the wave's buffer-set guard is held for the
whole wave — which the wave-domain design already implies, and which is
probably the cheaper answer. **Decide this before writing the first
intermediate**, not after. Recorded in §3.7 as an "as built" note.


---

## Step 3 — started, staged not landed. Tree unchanged and green.

### The §3.7 lease-counting question is settled: **the wave holds the count, not the lease.**

§3.7 drafted `ArenaLease = {generation_id, Arc<AtomicUsize> live_count}`, and
step 1 shipped `Backing::Lease` as a bare marker with the count living on
`BumpArena::Generation` instead. Step 3 forces the choice, because wave
intermediates are handed to candle ops as `Tensor`s and outlive the scope that
allocated them.

**Answer: hold one generation guard for the whole wave.** A per-lease refcount
would make every intermediate independently pin the cursor, which is both more
machinery and *weaker* — the cursor could then rewind mid-wave the moment the
last intermediate happened to drop, while a kernel enqueued earlier is still
draining. A wave-scoped guard is exactly the lifetime every intermediate needs
and no longer, and it is what the wave-domain design (A/B halves, buffer set
allocated at wave start) already implies. `Backing::Lease` stays a bare marker.

The design doc is corrected to state this rather than pose it as open.

### What is staged

`docs/drafts/wave_domain.rs` — the A/B-half wave domain (`WaveDomain`,
`begin_wave`, `wave_domain_stats`, `WAVE_HALF_BYTES = 64 MiB`), written against
the `bump_arena` API and ready to splice back in above `persistence_domain`. It
was written into `bump_arena.rs`, compiled, and then **removed again**: with no
consumer it is dead code, which the repo forbids, and its consumers are a step
of work away.

The 64 MiB half is deliberately modest, not the design's generous default — on a
16 GiB card two halves compete with KV, and this step's scope is only the
custom-kernel output buffers. The peak log (`wave_domain_stats`) is what turns
it into a measured number for step 4's `S = 2·W_wave + W_persist + shelf`.

### Why it did not land with a consumer

The reachable consumers — the four per-forward kernel output allocations in
`prefill_utils.rs` (the paged-decode-v2 q8 pack destination and its three
siblings) — are in **candle-transformers**, not candle-nn. Wiring them needs:

1. `BumpArena` / `Generation` / `BumpRange` promoted from `pub(crate)` to `pub`
   and re-exported from `kv_cache`, since the consumer crate is downstream;
2. a **wave plan** threaded from the wave boundary (`batched_inference.rs`,
   alongside `begin_stager_generation`) down through the forward to each
   allocation site. The alternative — an ambient "current wave arena" looked up
   at the allocation site — is worse: it hides the guard's lifetime from the
   type system, which is the one thing the counted generation exists to make
   explicit.

(2) is the real content of step 3 and is a session of work on its own. Each site
then swaps `dev.alloc::<u8>(n)` for a bump range plus
`CudaStorage::from_leased_device_ptr` (step 1.5b(i)'s primitive, already built
and tested) — baseline (a): our kernels take preallocated leased Tensors,
interior op outputs stay on the pool remnant.

### Tree state at this stopping point

Unchanged from the step-2 gate: 53 files changed, +5,386/−4,873, all
uncommitted. `cargo check -p candle-nn --features cuda` clean after the revert.
Last full gate 16/16 @ 126.28 s; 383 CPU tests, 420 candle-nn GPU tests, both
cfg branches clean, clippy at baseline (core 54, nn 68, transformers 113,
conversation 16).


---

## Step 3 — inference-loop intermediates on the wave's transient half

### The §3.7 lease-counting question, settled

§3.7 drafted `ArenaLease = {generation_id, Arc<AtomicUsize> live_count}`; step 1
shipped `Backing::Lease` as a bare marker with the count on
`BumpArena::Generation`. Step 3 forced the choice, because wave intermediates
are handed to candle ops as `Tensor`s and outlive the scope that allocated them.

**A scope holds the count, not the lease.** `Backing::Lease` stays a bare
marker; a generation guard bounds the intermediates instead. A per-lease
refcount would be more machinery *and* weaker — the cursor could rewind the
moment the last intermediate happened to drop, while a kernel issued earlier was
still draining.

The guard is **layer-scoped**, not wave-scoped. Getting that wrong is the one
real defect this step produced; see "the gate caught a design error" below.

### No wave plan is threaded, and that is the right answer

The plan was to thread a wave-plan object from the wave boundary down to each
allocation site. The caller set killed it: `paged_decode_attn`,
`paged_glue_attn` and `paged_prefill_batched` are also called by
`kernel_layout_tests`, `prefill_replay`, `test_fp8_hd128`, and the `decode_ab` /
`prefill_ab` fixtures — none of which have a wave. Threading would have forced
every harness to synthesise one.

`wave_alloc` instead answers **"is a wave in flight on this stream?"** from the
domain's own liveness count. That is real state, not a mode flag — inside a wave
there is a generation bounding the range's lifetime; outside one there is
nothing, and the caller must own its buffer. It also matches the precedent step
2 already set with `persistence_domain(stream)`, and the argument `BumpRange`'s
own doc comment makes: the compiler cannot express "valid until the generation
resets", so threading a lifetime buys nothing the counted generation does not
already enforce.

### What landed

**`candle-core`** — `CudaDType::wrap_leased_ptr` and
`CudaStorage::wrap_leased_ptr<T>`, the typed counterpart of
`from_leased_device_ptr` for generic kernel wrappers that know their output type
as a type parameter rather than a runtime `DType`.

**`bump_arena`** — the wave domain: A/B halves, `begin_wave`, `wave_alloc`,
`wave_domain_stats`. Halves are 64 MiB, deliberately modest rather than the
design's generous default: two halves compete with KV on a 16 GiB card, and the
peak log is what turns the number into a measured one for step 4's
`S = 2·W_wave + W_persist + shelf`.

**`models/wave_buffers.rs`** (new) — `KernelOutput<T>` for kernel destinations
and `wave_zeros` for destinations that are *accumulated into* rather than
overwritten.

**Consumers** — all four kernel output allocations in `prefill_utils.rs`
(paged-prefill-int8, paged-glue, paged-decode, paged-decode-q8). On the decode
path each was an alloc/free pair per layer per forward.

`ys`, the MoE combine target, was moved and then **moved back**. It is
*returned* from `fwd_expert_gpu`, so nothing inside the MoE forward bounds it,
and the only scope that would have — a wave-wide guard — is precisely what the
gate failure removed. It needs the explicit ping-pong buffer set, not an ambient
bump.

### The fence was going to be a regression, and the A/B halves are why it isn't

`Generation::drop` unconditionally called `stream.synchronize()`. Held for a
whole forward, that would have put a **full device sync on every forward** — a
pipeline stall introduced by this step, not an optimization deferred by it.

The A/B structure already makes it unnecessary. The two halves share one stream,
so by the time a half is handed out again an entire other wave's work sits
between its last read and its next write, and same-stream launches complete in
issue order. The fence buys nothing there.

So reclamation is now a declared per-domain policy:

| Domain | Policy | Why |
|---|---|---|
| persistence staging | `Fence` | stages on the copy stream while the compute stream runs — genuinely unordered |
| wave halves | `StreamOrdered` | double-buffered on one stream; a whole wave separates reads from writes |

This is a correction to the design doc, which described the event fence as
unconditional (§3.6, and the risk table row on cursor-reset races). Both are
fixed. The A/B halves are not merely an overlap optimization — they are what
makes the wave reset free.

### Verified by hand: no leased buffer escapes the forward

The hazard this step introduces is a leased tensor outliving the wave that owns
its memory. Traced on the decode path: `paged_decode_attention` returns the
leased attention context up to `forward_attn_batched_decode`, which feeds it
straight into `output_projection` — as an int8 operand when `want_q8`, otherwise
reshaped — and returns only the projection's own tensor. The leased storage dies
inside the layer. The prefill and glue paths converge on the same o_proj. `ys`
is reshaped and returned, but the reshape's consumer is the residual add, whose
output is a fresh pool tensor.

### What step 3 does *not* cover, and why

The design names three things to move: attention outputs, the MoE combine
output, and the **inter-layer hidden state**. The first two have explicit
allocation sites and are done. The hidden state does not — it is only ever the
result of a residual add, i.e. an output candle's own binary op allocates.
Redirecting that is exactly the `WaveAllocScope` of optimization (b), which the
design already schedules for step 6. Recorded here rather than silently skipped.

### Tests

Five new device tests on the wave domain, covering what the safety argument
actually rests on:

| Test | Guarantee |
|---|---|
| `no_wave_in_flight_means_no_range` | outside a wave, nothing is handed out |
| `ranges_within_a_wave_are_disjoint` | disjointness + alignment; the half closes again on drop |
| `consecutive_waves_alternate_halves` | A → B → A; the property the whole reset argument rests on |
| `a_concurrent_wave_is_refused` | two waves are fine, a third is refused (principle 7) |
| `peak_outlives_the_reset` | peaks survive the cursor rewind — step 4 sizes the span from them |

They serialise on a module mutex: the domain is process-global and cargo runs
tests in parallel, so without it `no_wave_in_flight_means_no_range` would see a
half another test still had open.

### Gating note: one test owns the candle-nn suite's wall clock

`kv_cache::chunked::tests::kv_stats_tests::test_q0_v_iterative_curve_selection`
runs for 20+ minutes and the other 445 tests finish in a couple of minutes. It
sweeps a curve family — 12 frequencies × shapes × 16 phases × 32 lanes, each
firing a CUDA roundtrip kernel through `par_iter` — across the real Qwen3 and
Llama KV dumps, in a **debug build**. It is guarded on the dump files existing,
so CI skips it in milliseconds and only this machine pays for it.

It is a threshold-derivation *measurement*, not a regression test, so it does not
belong in a per-step gate loop. Routine runs:

```
cargo test -p candle-nn --features cuda --lib -- --skip test_q0_v_iterative_curve_selection
```

Run it deliberately, in `--release`, when the Q0/V thresholds are actually in
play. Two suite runs were lost to it this session before it was identified —
the first invisibly, because `cargo test … | tail` buffers all output until
exit, so a long run and a hung one look identical. Never pipe a gate through
`tail`; redirect to a log and read that.

### The gate caught a design error: accumulation where the design said reuse

First gate run on the wave wiring:

```
Error: wave-b: transient span exhausted — 4030464 B at offset 64487424
       exceeds the 67108864 B budget.
```

Not a sizing miss. **One guard held for the whole forward, with a fresh bump per
kernel per layer, makes consumption O(layers)** — every layer's attention output
stays live until the forward ends, roughly 48x what any single layer needs. No
half size fixes that; it only moves the cliff.

The design said so and I read past it: step 3 specifies that the per-layer
tensors become "lease-backed views over the **ping-pong buffers**" — a fixed set
allocated once and *reused* each layer. I built accumulation where it specified
reuse.

**Fix: scope the guard to the layer.** It now spans exactly attention ->
`output_projection` on both the decode and prefill paths — the lifetime already
traced by hand, where the context is provably dead once o_proj has produced its
own tensor. Halves alternate per layer, so layer N's reads are separated from
layer N+2's writes by a full layer of same-stream work; the `StreamOrdered`
reclaim argument holds unchanged at layer granularity. The wave-wide guard in
`batched_inference` is gone.

This is worth stating plainly because the loud failure is the only reason it was
caught at all: a silent fallback to the pool on exhaustion would have hidden an
O(depth) allocation pattern behind ordinary-looking throughput, and it would have
surfaced as an OOM on a longer model rather than as an error naming its own
budget.

### Gate

| Gate | Result |
|---|---|
| MoE `test_parallel_batched_forwarding` (`--test-threads=1`) | **1 passed, 0 failed** — 143.20 s, then 122.99 s on an unchanged tree |
| candle-nn GPU suite | **424 passed, 21 ignored, 0 failed** (excl. the curve sweep) |
| Both cfg branches, workspace + tests | clean |
| clippy | 235 total (core 54, nn 68, transformers 113) — at baseline |

Two gate invocations were wasted first, both my error, both worth recording:
running without `--ignored` silently matched **zero** tests and still reported
`ok`; and running without `-p candle-transformers` rebuilt every workspace test
binary and example in release before reaching the test. A third run let cargo
execute all 5 matching `test_parallel_batched_forwarding` variants in parallel,
which failed on `cuMemAllocHost ... 13.5 GB` — host pinned memory, five CUDA
contexts deep, nothing to do with the code under test.

### Still open at the end of step 3

- **`W_wave` is measured: 30.8 MiB per half.** Qwen3-30B-A3B, batch 64. Both
  halves converge on the same peak, which is what per-layer alternation should
  produce. Decode-only layers sit at 3.8-6 MiB; the high-water mark is a
  wide-prefill attention output. The 64 MiB half therefore carries ~2.1x
  headroom, and is now a justified number rather than a guess.

  Getting it took one temporary `eprintln!` in `BumpArena::alloc`, run, read,
  removed. The two permanent log sites (`wave_domain_stats` in
  `scheduler/run.rs`, `log::debug!` in `alloc`) are worth keeping for the daemon
  but produce **nothing** under `cargo test` — a test binary initialises no
  logger or tracing subscriber, so the records are dropped. Reaching for the
  throwaway `eprintln!` first would have saved a round trip.

- **The persistence domain is over-provisioned by ~3 orders of magnitude.**
  Same run: `persist-staging` peaks at **29,696 B** against a
  `MIGRATION_STAGING_CAP_BYTES` span. Its budget was inherited from the cap the
  migration batches already bisected against, never from a measurement. Step 4
  should size it from this, not from the old cap — it is close to free space.

- **`ys` and the inter-layer hidden state** both need the explicit ping-pong
  buffer set (`ys` because it outlives its function, the hidden state because it
  is a residual add with no allocation site). The hidden state was always step
  6's `WaveAllocScope`; `ys` should join the buffer-set work rather than be
  forced onto an ambient bump.


---

## Step 4 — not started. Entry conditions and the two measured inputs.

Stopped here deliberately: step 4 is unsafe driver work spanning startup, the
allocator and the scheduler, and it does not decompose into a piece small enough
to land green in the remaining budget.

**Tree at this boundary:** 55 files, all uncommitted, both cfg branches 0
errors, clippy 234 (baseline 235), candle-nn GPU suite 424 passed / 21 ignored /
0 failed, MoE gate green (122.99 / 139.78 / 143.20 s across runs).

**Two inputs step 4 no longer has to guess**, both measured on Qwen3-30B-A3B at
batch 64:

| Term | Measured | Note |
|---|---|---|
| `2·W_wave` | **61.6 MiB** | 30.8 MiB per half; both halves converge, as per-layer alternation should |
| `W_persist` | **29,696 B** | ~3 orders of magnitude under its inherited `MIGRATION_STAGING_CAP_BYTES` |

So `S = 2·W_wave + W_persist + shelf` is dominated by the wave term and the
shelf; the persistence span should be re-sized from its peak rather than carried
over at the old cap.

**Order of work**, per §3.2 and the step-4 text:

1. Startup: reserve the VA span, balloon granules to the driver's refusal
   (measuring `C` and claiming it in one act), release the partition for dense
   weights + expert cache + small reserve, load, keep the rest as the
   reservation. Runtime probe falls back to the giant-`cuMemAlloc` order.
2. Carve regions, seed the free-region list. `create_arena` → carve,
   `release_arena` → push free list; neither touches the CUDA allocator.
3. Swap the bump side's provisional backing onto the reservation's right side at
   a fixed, region-aligned boundary sized from the table above.
   `drain_free_arenas_above(k)` keeps the free list on the KV side. **The
   boundary does not move in this step** (§9 S6).
4. Free-region counter as the pressure signal; rightmost-first
   evict-as-evacuation (§3.8) on the persistence thread's demotion pump.

Fold in here rather than in step 6: the **ping-pong buffer set** for `ys`, which
is returned from `fwd_expert_gpu` and so has no scope inside the MoE forward to
bound it.

*Gate*: correctness; startup time; and the one that matters —
**zero CUDA allocations in steady state** (`KV_ARENA_STATS` / pool counters
flat).


---

## Step 4 (partial) — the VA reservation is built, tested, and carrying the transient tier

| Gate | Result |
|---|---|
| MoE `test_parallel_batched_forwarding` | **1 passed, 0 failed, 130.33 s** |
| reservation + wave device tests | **9 passed, 0 failed** |
| Both cfg branches, workspace + tests | clean |
| clippy | 235 — at baseline |

### What landed

**`reservation.rs`** — a safe wrapper over the VMM API. cudarc exposes
`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/`cuMemSetAccess` only as raw FFI
in its `sys` layer, so this is ours to own. `reserve` takes address space;
`map_more` creates a granule, maps it, and grants access; `balloon` maps until
the driver refuses.

That refusal is the design's `C`: **measuring capacity and claiming it are one
act**, because any gap between the two is a window for another process — or our
own weight loading — to change the answer. `map_more` returns `Ok(false)` on
`CUDA_ERROR_OUT_OF_MEMORY` and errors on anything else, so ballooning stops
cleanly without swallowing real faults.

**Verified on the device, not merely compiled:**

| Test | What it proves |
|---|---|
| `reserving_more_than_the_card_costs_nothing` | 64 GiB reserved on a 16 GiB card, 0 B mapped — the property the whole design rests on |
| `reservations_are_whole_granules` | sizes round up, so region carving can assume alignment |
| `mapped_memory_is_usable_at_the_reserved_address` | a write through the reserved address reads back — the only real check that `cuMemSetAccess` was applied, since a mapped-but-inaccessible range faults later as an illegal address rather than failing at the call |
| `mapping_stops_at_the_span_end` | the balloon cannot run past its reservation |

**Wired to its consumer.** Both transient domains are now disjoint sub-ranges of
one per-device reservation rather than standalone `cuMemAlloc`s — §3.6's actual
wording, and what makes `BumpRange` legitimately a bare pointer: addresses are
fixed for the process lifetime by construction now, not by convention. A tested
module with no production consumer would have been exactly the dead code the
repo forbids, so the wiring landed with it.

### What remains in step 4

1. **Startup balloon + partition** — reserve, balloon to refusal, release the
   partition for dense weights + expert cache + small reserve, load, keep the
   remainder as the reservation. Runtime probe falls back to the
   giant-`cuMemAlloc` order.
2. **Region carve + free list** — `create_arena` → carve, `release_arena` →
   push; neither touching the CUDA allocator.
3. **Free-region counter as the pressure signal**, plus rightmost-first
   evict-as-evacuation (§3.8) on the demotion pump.
4. **`ys` ping-pong buffer set** (see step 3).
5. **Re-size `persist-staging`** from its measured 29,696 B peak.

Its gate — *zero CUDA allocations in steady state* — needs a measured run with
pool counters flat, not just a passing test.

---

## Step 4 — complete. The KV cache runs on the reservation.

| Gate | Result |
|---|---|
| MoE `test_parallel_batched_forwarding` | **1 passed**, 122.80 / 125.89 / 128.63 / 139.68 / 152.82 / 153.44 s across six runs |
| candle-nn GPU suite | **429 passed**, 21 ignored, 0 failed |
| `cargo check --workspace --tests` | 0 errors |
| `cargo check --workspace --tests --features cuda` | 0 errors |
| clippy (core+nn+transformers, `--features cuda --lib`) | **235** — exactly baseline |
| **Zero CUDA allocation growth in steady state** | **`pool_reserved` flat at 8,858,370,048 B across configs 1–15** |

### The gate that matters, measured

The step-4 gate is not "the test passes" — it is *does the allocator stop
growing*. Instrumented temporarily at each config boundary of the MoE gate:

```
cfg=0   pool_used=8434876672  pool_reserved=10435428352  regions=None
cfg=1   pool_used=8003666176  pool_reserved= 8858370048  regions=226 total, 0 live
cfg=2   pool_used=8116781312  pool_reserved= 8858370048
cfg=3   pool_used=7782793472  pool_reserved= 8925478912   <- +64 MiB, once
cfg=4..15                     pool_reserved= 8858370048
```

`cfg=0` is the model loaded with no KV cache yet, so no reservation exists and
the pool still holds the load's high-water mark. From the first KV cache onward
the pool **never grows again** — one 64 MiB excursion at cfg=3 and back. Every
byte of KV after that point comes from the reservation, and the 562 arena
creations the same run performs (`KV_ARENA_STATS`) average **0.029 ms**, because
an arena creation is now a free-list pop and a pointer.

`live: 0` at every config boundary is the other half of the proof: all 562
regions came back. The free list is not leaking.

### What the partition came out as

| Term | Value |
|---|---|
| KV side | **226 regions = 3,616 MiB** |
| Peak live regions | **167 of 226** (74 %) |
| Transient tier | **704 MiB**, of which 640 MiB carved |
| Wave halves | 30.75 MiB peak on A, 0 on B (see below) |

### Five corrections to the design

**1. The startup order is inverted from §3.2, and it has to be.** The design
claims the reservation is filled *first* — balloon to refusal, then release the
partition's granules for dense weights + expert cache + small reserve, then load
into the freed space. That is not buildable here, because **the partition is not
knowable before the load**. `expert_budget()` is by construction a live
measurement taken *during* the load, and the dense-weight total is only known
once the loader has walked the file — predicting either would mean duplicating
the loader's tensor walk and keeping the duplicate in sync forever.

As built: the governor's balloon measures `C` and frees, exactly as before; the
model loads; and the reservation is then claimed from `usable - scratch_margin`
at the first KV cache. The claim itself keeps the design's real insight —
granules are mapped **and written** one at a time, so the reservation ends
wherever the driver actually refuses. Measuring and claiming stay one act; only
the *order* relative to the model load changed.

**2. `cuMemCreate` never refuses, so the touch is the measurement.** §3.2 already
records the probe (`cuMemCreate` succeeded across a 32 GiB span on a 16 GiB
card); the code now honours it. `Reservation::map_granule` creates, maps, grants
access, and then `memset`s the granule **synchronously** — the write is the
capacity test, and a failure unmaps and releases that granule and stops the
balloon. The zero-fill is not waste: it is what makes a freshly-claimed region
zeroed, which is exactly the guarantee `Tensor::zeros` used to give the slab it
replaced.

**3. §3.3's "the free-region list already exists" is wrong.** It says
`GidPoolState.free_arenas` *is* the region free list, so the region tier adds no
lock. It cannot be: `register_arena` pops that queue **regardless of location**,
so a tombstoned CPU arena's index is handed to the next GPU arena and vice
versa. Arena indices are therefore not positions in the span, and using them as
region indices would scatter GPU regions across address space that was never
mapped. The region pool keeps its own free list (a min-heap, lowest-first per
principle 5) and its own mutex. The lock-order claim of audit A5 still holds —
the new mutex is a leaf, taken by nothing else.

**4. `W_persist` is a declared budget, not a watermark — the earlier measurement
was a sampling artefact.** The step-3 record says the persistence domain peaks at
29,696 B and should be re-sized from that. **Do not.** That number came from a
gate run that barely migrates; the domain carries the *bulk* hot→warm staging,
which bisects itself against `MIGRATION_STAGING_CAP_BYTES` (512 MiB). Shrinking
the span to the observed peak would not have saved memory, it would have cut the
migration batch by four orders of magnitude and turned a ~3-batch pass into
thousands of DtoH syncs. The transient tier is sized `2·WAVE_HALF + 512 MiB +
64 MiB shelf = 704 MiB`, built as a `const` expression from each domain's own
budget so the three cannot drift apart.

**5. Evacuation is positional by construction, not by a separate scan.** §3.8
asks for "demote the **rightmost** occupied regions". As built there is no
rightmost scan, and there should not be one: regions are handed out lowest-first,
so the high end of the span is the least-populated by construction, and the
scheduler's existing budget-aware eviction — now driven by the **exact**
free-region count instead of three disagreeing driver estimates — empties it
without needing to know that is what it is doing. §3.10 already says this ("with
lowest-first packing, rightmost ≈ emptiest"); the separate mechanism §3.8 implies
is redundant with the packing policy.

### The bug the gate caught: recycling a region has no stream order

Removing the anti-churn tombstone guard (below) turned the candle-nn suite red
with `CUDA_ERROR_ILLEGAL_ADDRESS` — 14 failures, all cascades of one sticky
context fault.

A region returns to the free list the instant its arena drops, and that says only
that **no host-side gid still names it**. Kernels launched earlier can still be
reading it, on the compute stream or on the persistence thread's copy stream. The
next claimant then memsets those bytes.

The allocator used to supply the missing ordering for free: an arena's slab was
released with `cuMemFreeAsync` and re-allocated with `cuMemAllocAsync` **on the
same stream**, so reuse could not overtake the reads. A free-list push has no
such ordering. So the wait is now explicit — a device-wide synchronise before a
*recycled* region is zeroed, device-wide because the copy stream is one of the
readers and draining the compute stream alone would not cover it.

This was latent before the guard came out, not introduced by it: the guard made
recycling rare enough that the race almost never landed. It is the second time in
this initiative that removing a conservative guard has been the thing that
surfaced the defect it was accidentally hiding.

### Deletions forced by the region model

Step 5 inventory items that became dead the moment the allocator stopped being
involved, so they landed here rather than being left as dead code:

| Deleted | Why it is dead |
|---|---|
| `ensure_vram_budget`, `vram_gate`, `VramGateFacts`, `gate_decide`, `vram_has_room` | The gate asked "will this arena fit?"; the answer is now whether a region is free, which the claim itself reports |
| `vram_reserve_bytes`, `eviction_reserve_bytes`, `EvictionScope`, `EVICTING` | Byte reserves defending a pool that no longer grows. The compress-to-free deadlock they existed for is closed by scarcity-only class promotion instead (§3.4) |
| `kv_alloc_headroom` | One of three competing availability estimates; the region counter replaces all three |
| `request_global_compact` | The allocation-failure retry. Under regions it would have to run while the storage lock is held, which is a deadlock, and the scheduler's wave-boundary sweep covers the same ground safely |
| `try_tombstone`'s 10 % headroom guard and its `force` bypass | Anti-churn for a create/destroy cycle that no longer exists. Its condition was "the pool is nearly full", i.e. exactly when reclaim was being asked for, which is why it needed a bypass at all |
| `release_empty_arenas_forced` (three layers of it) | Identical to the unforced twin once the guard is gone |

`vram_budget_available` survives with a new body: `free_regions × REGION_BYTES`.
Its old comment ran to 25 lines explaining which of three estimates to trust in
which regime; the replacement is one line of arithmetic on an exact counter.

### The wave halves stopped alternating, and that is fine

Half A peaks at 30.75 MiB; half B at **0**. Two things caused it, and neither is
a fault:

- A layer now opens **two** generations — attention and FFN — so with one
  attention group the parity is even and each call site keeps landing on the same
  half. Alternation is what makes the gap between a half's last read and its next
  write large; it is not what makes it correct. The correctness argument is
  same-stream issue order, and between two attention sections sits an entire FFN.
- Half B reads 0 because the FFN section allocates nothing **on this card**: the
  wave-backed MoE combine target is on the GPU-native dispatch path, which needs
  an all-resident expert cache. At 46 % residency the threaded pipeline runs
  instead.

### The MoE combine target: one of three sites moved, deliberately

`ys` is the layer's largest transient and there are three of them:

| Site | Disposition |
|---|---|
| `quantized_qwen3_moe.rs` GPU-native dispatch | **`wave_zeros`** — bounded by the FFN generation `forward_layer_batched_mixed` now opens around `ffn_forward` and the residual add that consumes it |
| `expert_lre/pipeline.rs` threaded | **Left alone, and it has to be.** It is produced on the pipeline thread and handed back over a channel, so the scope that would hold it open belongs to a different thread than the one that allocated it |
| `expert_lre/handle.rs` inline | Left alone — the caller's own thread and stream, so it *could* be wave-backed, but the mode only exists when there is no pipeline thread, which production never configures |

So "zero CUDA allocations in steady state" is met for arenas and for the
attention path, and the remaining per-layer allocation on this card is the
threaded pipeline's combine target. Moving it needs the expert pipeline to stop
returning ownership across a channel — a restructuring of who owns the result,
not a buffer change.

### Sizing without a governor

Test binaries have no governor, and the first attempt sized their KV side at
half of what the driver reported free. That is unstable by construction: a test
binary runs hundreds of GPU tests concurrently, so "half of free" depends on
which ones are in flight when the first KV cache is built, and one run lost the
race badly enough to claim nothing at all and fail 15 tests. It is a fixed
2 GiB now — above the suite's peak arena working set, below what would starve
the tests' own tensors.

### The giant-`cuMemAlloc` fallback: reviewed, and cut from the design

Step 4 left it unbuilt and flagged it. Reviewed 2026-08-08: **cut, and §3.2
rewritten to match** — it is no longer a deferred item, it is not part of the
design.

The reason it looked cheap was the seam: everything above `reservation.rs`
consumes one contract (`map_range` → bytes actually backed), so a second
implementation behind it seemed like a swap. It is not. Three things differ, and
each is load-bearing:

1. **Eviction granularity.** VMM sheds 2 MiB granules under WDDM pressure; a
   giant allocation's eviction unit is the whole buffer, so the entire
   reservation leaves for host and faults back over PCIe at the next touch.
2. **Partial release.** The probed release-then-reuse behaviour has no
   `cuMemAlloc` equivalent; a giant allocation can never return a byte.
3. **Capacity measurement.** Map-and-touch per granule is self-terminating and
   doubles as the region tier's zero-fill guarantee. `cuMemAlloc` is
   all-or-nothing, so the extent would have to be binary-searched over whole
   allocations — a different algorithm with its own failure modes, not a ported
   one.

So the fallback is a distinct implementation that only a machine we do not own
can exercise, and code like that is wrong by the time anyone needs it.

What replaced it is an explicit refusal. `Reservation::reserve` now queries
`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` before it reserves
anything and fails with a message naming the capability, the reason there is no
second path, and where VMM is commonly absent (WSL2, vGPU/MIG, older drivers).
Previously such a device would have failed at `cuMemAddressReserve` with the
driver's own error — correct, but it tells a reader nothing. A device test
(`the_device_supports_vmm`) asserts the attribute, so an unsupported target
reports it in the suite rather than at the first KV cache of a benchmark run.

Unverified and worth five minutes before step 7: **whether WSL2 on this driver
reports the attribute**. If it does, the whole concern is dead scope. If it does
not, WSL is out of bounds for the daemon and that should be known before it is
discovered by a failed run.

### Still open after step 4

- **`migrate_sealed_to_gpu` targeted `ArenaLocation::Gpu` unconditionally**, so a
  CPU-only backing produced an arena tagged `Gpu` whose bytes were on the host.
  Harmless while an arena was an ordinary allocation; a contradiction once `Gpu`
  means a region of the device reservation. Now it targets the backing's own
  device (`BackingInner::hot_location`).
- The device-wide synchronise on a recycled region claim is the one new cost on
  an allocation path. ~336 of the 562 claims in a gate run are recycled. Step 6
  should measure it and, if it shows, replace it with a per-region release event.
- The static shelf still has no allocator; `SHELF_BYTES` (64 MiB) is address
  space held for it. `ProvSignScratch`, the sampler scratch and the MoE routing
  buffers are still grow-only pool allocations.

---

## Step 5a — compaction, the pressure ladder and the topology guard are gone

| Gate | Result |
|---|---|
| MoE `test_parallel_batched_forwarding` | **1 passed**, 118.61 s (3m32 wall incl. build) |
| candle-nn GPU suite | **423 passed**, 21 ignored, 0 failed |
| `cargo check --workspace --tests` | 0 errors |
| `cargo check --workspace --tests --features cuda` | 0 errors |
| clippy (core+nn+transformers, `--features cuda --lib`) | **227** — 8 below the 235 baseline |
| candle-conversation lib tests | **950 passed**, 0 failed |

Tree at this point: 74 files changed, 6,330 insertions / 9,385 deletions. Step 5
is the first sub-step of this initiative where deletions outweigh insertions by
more than 3,000 lines.

### What the pressure signal became

`vram_under_pressure_for` was three gates in disjunction, each present because
the other two were wrong in some regime:

1. a byte budget from `init_free − pool_used − reserve` against a card-fraction
   band,
2. a driver-free floor, qualified by how much the CUDA pool could still absorb by
   reuse (`reserved − used < 512 MiB`, plus an over-subscription arm for
   `reserved > total`),
3. a footprint gate on `pool_reserved` against a compaction ceiling — which
   needed a 2 s cooldown *and* a futility latch with a doubling re-probe bar,
   because a fragmented gap the engine kept reusing reported pressure on every
   scheduler loop and pinned the admission window at the floor.

All three are one line now: `free_regions < setpoint`. The setpoint is
`max(span/8, 24 regions)` under load and `max(span/16, 8)` in decode, clamped to
half the span so a card too small to hold it cannot sit in permanent pressure.
It is scaled to the reservation rather than the card, so the same constants hold
on this 226-region KV side and on the workstation's.

**The band was measuring the wrong thing, not merely measuring it badly.** It
reserved headroom against a wide forward's *transient activation* peak — a
quantity the KV side no longer has anything to do with, because transients come
from the reservation's other end (§3.6). The one reserve still expressed in
bytes against capacity is `reserve_for_width`, inside admission, and it is about
activations. Its card-fraction base is deleted too: that base was this same
pressure band borrowed as though it were a physical requirement, and it once
saturated `available − reserve` to zero at every width.

### The ladder became call order

`relieve_vram_pressure` ran the governor's `Criticality` ladder through a
borrowed `SchedulerReliefDriver`, with the governor re-measuring driver headroom
between rungs to decide whether to climb. It is now a straight sequence, each
step conditional on pressure surviving the last: release empty arenas → evict
resident galleries → compress-to-free → flush pending hot→warm and evacuate.

Same priorities, no arbitration — which is the whole point. §5 predicted this for
the gallery specifically ("the same priority, expressed as call order instead of
rung numbers"); it generalises to every rung, because with an exact counter there
is nothing to re-measure between them.

Two things fell out:

- **`reclaim_footprint` is gone entirely**, with `VRAM_EVICT_*`, `cap_margin`,
  `FOOTPRINT_HYSTERESIS`, `FOOTPRINT_RELIEF_COOLDOWN`, `VRAM_MIN_COMPACT_GAP_*`,
  `defrag_futile`, and the `defrag_futile_at` / `last_footprint_relief` fields.
  Every one of them was hysteresis around a number that no longer moves.
- **`relieve_compression_starvation` is gone.** A starved background compressor
  now takes the ordinary relief path. Its escalated variant existed to *override*
  the "only evict when `used` is high" watermark — an override that was only
  needed because the watermark was a guess about the card.

### Admission stopped forecasting

`admit_budget_ceiling` was `available_bytes(headroom, evictable, pinned, …)`:
a driver measurement, plus what registered relievers claimed they could
reversibly free, minus the evictable-but-pinned working set the hot→warm drain
was skipping. Both corrections existed because the base term described the card.

It is now `(free_regions − setpoint) × REGION_BYTES`. Pinned KV holds live
regions, so it is excluded by construction rather than by discount; evictable KV
shows up as free regions the moment the relief pass ahead of admission evicts it.
Measured, not forecast. `available_bytes` and its two tests are deleted.

### Two defects found while deleting

**1. Every `#[cfg(feature = "cuda")]` block in `candle-conversation` is dead
code.** The crate has no `cuda` Cargo feature — it forces the feature on its
*dependencies* and its manifest says so explicitly ("there is no longer a `cuda`
Cargo feature on this crate"). So `#[cfg(feature = "cuda")]` is always false
there. Step 4's permanent `kv-regions:` telemetry sat inside such a block, along
with the transient-domain and slot-state lines: **none of it has ever emitted**.
The step-4 gate measurement is unaffected — it was taken through the MoE test
with a temporary `eprintln!` — but the daemon-side observability it was supposed
to leave behind did not exist. Now unconditional, in `log_kv_memory`.

**2. `demote_cold_ingest_if_pressured` was reading a watermark that had come
loose.** It fires when `pool_used` exceeds ~50 % of C. With KV out of the pool,
`pool_used` is the model plus the expert cache — a high, flat fraction of C
forever — so the gate would have fired on *every wave* regardless of how much
ingest was resident. Moved to KV-span occupancy (`live > total × pct/100`), which
is the same question asked of the right counter. This would have been very hard
to see from behaviour: the demote is bounded and mostly a no-op, so it would have
read as harmless churn.

Also fixed in passing: three `#[test]` attributes duplicated above a doc comment
(`compaction.rs`, `admission.rs`, `substrate.rs`), and a pre-existing red test —
`report_serializes_with_every_section` asserted a `float_live_bytes` field that
earlier uncommitted work had removed from the report struct.

### The topology guard: dead twice over

`migrate_guard.rs` was a process-global `RwLock` over arena topology — shared for
operations that captured raw arena base pointers, exclusive for operations that
invalidated them. Audit A4 predicted the reservation would kill it. It did, but
the audit understated the case: **the guard was already unreachable before step
4**, and for an independent reason.

The hazard it names is the migrate's *dense* per-head table, which addressed
every arena in a backing including empty ones, so the scheduler's sweep could
unmap a neighbour the kernel was about to dereference. Step 1's E5 change made
that table **job-list sized**: every pointer in it now comes from a gid the
caller has pinned, and a pinned arena cannot be tombstoned. The neighbour-arena
hazard had no route left even under the old allocator.

The reservation closes the other half — nothing unmaps at all now, and the
ordering that *is* still required (not re-tenanting a region while an earlier
kernel may still read it) belongs to `claim_region`, where step 4 put it.

What survives is `migrate_flight.rs`: an atomic counter and an RAII marker, no
lock. Its two consumers defer a section quantize while a migrate is converting
the same residences — which they described as a safety deferral against a stale
base pointer and is in fact a *work* deferral against doing the conversion twice.
Comments corrected to say so. The warm→hot elevate dropped its guard outright:
its destination arenas are freshly allocated and hold live gids, so nothing can
reclaim them mid-write.

**Removing this takes a process-global read lock off every migrate and every
elevate.**

### The per-wave device sync is gone

`BatchedInferenceSession::release_empty_arenas` runs every wave. It took the
topology write lock and then paid a full `device.synchronize()` before releasing,
because releasing an arena unmapped its slab. Nothing unmaps now, so both are
gone: a whole-device sync per wave, removed from the sweep path. The wait that is
genuinely needed is paid once per region claim instead.

### The pool trim is gone, and with it the guard around it

Every `trim_kv_pool` caller was inside the machinery deleted above. That is the
right answer rather than an accident: `cuMemPoolTrimTo` returned freed *pool*
blocks to the OS, which mattered when KV lived in the pool and its freed arenas
climbed `pool_reserved` away from `pool_used`. What is left in the pool is the
model, the expert cache and a few grow-only scratches — allocations that reach
their size and stay. There is nothing for a trim to return.

`CudaDevice::trim_pool` survives with exactly one caller: the **startup balloon**,
which must hand back the bytes it allocated to measure `C` before the model loads
into them. That is the governor's retained startup role. The doc now says the
trim is not a runtime path and why: it unmaps synchronously, which is safe before
any kernel has run and a hazard everywhere else.

`PoolTrimGuard` / `set_pool_trim_guard` / `guarded_pool_trim` are deleted, along
with the registration in `quantized_qwen3_moe.rs`. That registration existed only
because the sync-hook lives in candle-core, below candle-nn, and could not reach
candle-nn's topology guard — a layering workaround for a hazard that no longer
exists.

### Deletions

| Deleted | Where |
|---|---|
| `defragment_arenas`, `apply_gid_remap`, `compact_arenas{,_forced}`, `needs_compaction`, `defragment`, `compact{,_forced}`, `defragment_bounded`, `can_reclaim_arena`, `lock_all` + tests | `candle-nn` backing |
| `DrainPlan`, `drain_plan{,_for_key}`, `allocate_avoiding`, `allocate_for_avoiding`, `live_gids{,_for_arena}`, `arenas_sorted_by_live`, `defragmentable_ratio{,_for}`, `needs_defragmentation` + 4 tests | `candle-nn` gid_pool |
| `MigrateGuard`, `ReliefGuard`, `enter_migrate`, `try_enter_relief`, `ARENA_TOPOLOGY` | `candle-nn` (file replaced) |
| `compact_check`, `compact`, `compact_forced`, `defragment_bounded`, `can_reclaim_arena`, `trim_kv_pool` | session (`batched_inference`) |
| `CompactMove`, `arena_compact_copy{,_async}`, `arena_compact_patch{,_async}`, `run_arena_compact_*`, `arena_compact.cu`, `arena_compact_gpu_test.rs` | `candle-core` + `candle-kernels` |
| `PoolTrimGuard`, `set_pool_trim_guard`, `guarded_pool_trim` + test, the sync-hook trim | `candle-core` vram |
| `relieve_vram_pressure`'s ladder, `SchedulerReliefDriver`, `log_relief_event`, `reclaim_footprint`, `relieve_compression_starvation`, `defrag_futile`, `trim_kv_pool`, `vram_band_for`, `vram_base_band_for`, `combine_band`, `vram_budget_band`, `vram_decode_band`, `per_seq_decode_bytes`, `cap_margin`, `vram_evict_{high,low}_pct`, `compact_base_moves`, the gallery relief registration | scheduler |
| `available_bytes` + 2 tests | admission |

### Still standing after 5a

The **governor's runtime relief role** — `relief.rs` (`register_relief`,
`relieve_pressure`/`relieve_to`/`relieve_with`, `KvReliefDriver`, `Criticality`,
`evictable_estimate`) and its share of the 1,700-line `vram/tests.rs`. Nothing in
production registers a reliever any more (the gallery was the only one), and
`evictable_estimate` has no consumer left but diagnostics. That is step 5b.

---

## Step 5b — the governor keeps startup and gives up runtime

| Gate | Result |
|---|---|
| MoE `test_parallel_batched_forwarding` | **1 passed**, 120.32 s |
| candle-nn GPU suite | **423 passed**, 21 ignored, 0 failed |
| candle-core `vram::` suite | **30 passed**, 0 failed (incl. 3 real-CUDA balloon tests) |
| candle-conversation lib tests | **950 passed**, 0 failed |
| `cargo check --workspace --tests` | 0 errors |
| `cargo check --workspace --tests --features cuda` | 0 errors |
| clippy (core+nn+transformers, `--features cuda --lib`) | **227** |

Step 5 whole: **77 files, 6,395 insertions / 10,974 deletions**.

### What the governor is now

Its whole surviving job is the **startup partition**: balloon to find resident
capacity `C`, expose `usable()` so the expert cache and the KV reservation are
sized from one number, and hold the per-class tallies the memory report renders.
Everything that regulated at runtime is gone.

`relief.rs` is deleted entire — `register_relief`/`unregister_relief`, the
`ReliefRegistry`, `evictable_estimate`, `run_tier_with_sync`, `relieve_pressure`,
`relieve_to`, `relieve_with`, `KvReliefDriver`, `ReliefRequest`, `ReliefOutcome`,
`ReliefResult`, `ReliefHandle`, `last_relief`, `relief_count`, `sync_count` — and
with it `Criticality` itself, the five `LadderTier` trip points in `budget.rs`,
and `critical_min_interval_ms`.

Four more things fell out that the §5 inventory did not name, each because its
last caller was in the deleted set:

- **`VramGovernor::allocate`** — the OOM-retry path that escalated one rung per
  round. Its whole reason to exist was an allocation that had to survive
  transient exhaustion, and that allocation was KV. KV does not allocate now.
- **`VramGovernor::reserve`** — the class-tagged permanent allocation wrapper. It
  ran the closure and credited a tally; call sites credit the tally directly.
- **`forecast_units`** — "how many concurrent units fit", headroom plus evictable.
  Admission asks the region counter.
- **`spawn_budget_watcher`** — a thread blocking on the Windows budget-change
  event to shed KV the instant another process took VRAM. §3.9 gives that posture
  up explicitly: the reservation is held for the process lifetime and a hot
  reservation has maximal WDDM residency priority.

And the two hooks the governor was constructed with:

- **the sync hook**, which retired pending async frees before the ladder
  remeasured. There is no remeasure.
- **the reuse hook**, which fed `available()` the pool's `reserved − used` gap.
  That gap is the estimate that once read 3045 MiB free while `vram_free` was 0
  and the pool held 15168 of 16375 MiB. `available()` had no callers left, so it
  and the hook go together — the last of the reuse-gap-as-availability idea.

### What was kept, and why

**`AllocClass` and the per-class tallies** (`credit_class`/`debit_class`/
`set_class`/`class_reserved`). They are reporting, not gating: the memory report
renders them and the model loader credits weights and experts. Nothing decides
anything from them.

**`signal_starvation` / `take_starvation`.** A background compressor that cannot
get memory still needs to tell the scheduler; only the *response* changed (§5a).

**`kv_floor`, `scratch_margin`, `expert_budget`, `usable`.** The startup
partition, which is exactly what the design says the governor keeps.

**`CudaDevice::trim_pool`** — see §5a; the balloon needs it and nothing else may.

### The test file halved

`vram/tests.rs` went from 1,707 lines to 883. Twenty-nine test functions deleted:
the whole relief-ladder matrix (gentle-early, escalation, Critical-only sync,
no-spin, ladder-exhausted, relief-stops-when-recovered, driver-climbs-cheapest-
first, relieve-to-target, unregister), the forecast tests, the OOM-retry tests,
and the two real-CUDA ones that exercised the ladder against a live card.

The `scenarios` module was a simulated engine — a fake card, a KV/expert
residency model, and three registered relief closures — built to run
boot → load → grow KV → forecast → relieve → contend → steady state. Two tests
survive it, both about the startup partition (`startup_partitions_evolve_and_
leave_kv_floor`, `expert_budget_all_resident_vs_partial`), so `Sim` is now a
governor and a fake card with `load_weights`/`load_experts`/`load_scratch`.

`real_cuda` keeps its three balloon tests: measure-and-track, full-balloon
throughput, chunk cost. Those are the startup role, and they still pass on the
card.

### One report field went

`GovernorSection.evictable_moderate_bytes` — "what registered relievers report
they could reversibly free". With no relievers it was structurally zero. The
report's own header claimed to show "what the governor believes is evictable";
it now says how many KV regions are free, which is the number the throttles
actually reason about.

### The shape of the whole step

| | before | after |
|---|---|---|
| "is there room?" | three driver-derived estimates in disjunction, two hysteresis latches, a cooldown | one comparison against a free-region count |
| pressure response | a 5-rung governor ladder with re-measurement between rungs | four calls in order |
| admission ceiling | headroom + registered-evictable − pinned, clamped | `(free − setpoint) × 16 MiB` |
| reclaiming a region | chunk-moving GPU→GPU compaction, a batched copy kernel, a host-side gid remap | the arena's last chunk leaving |
| protecting a captured pointer | a process-global `RwLock` on every migrate and elevate | nothing — pointers cannot be invalidated |
| per-wave sweep cost | topology write lock + full `device.synchronize()` | a free-list push per region |

---

## Step 6 — steady state, measured

*Gate*: tokens/s meets or beats the step-0 baseline on every config.

| Config | baseline | before step 6 (5 runs) | after (3 runs) | best | vs base |
|---|---|---|---|---|---|
| F16×1 | 489.4 | 492.0–497.6 | 499.2–511.2 | 511.2 | +4.5 % |
| BF16×1 | 539.8 | 549.8–564.4 | 558.2–566.1 | 566.1 | +4.9 % |
| BF16×10 | 1890.3 | 1866.9–1947.9 | 1969.7–2034.7 | 2034.7 | +7.6 % |
| **Q8_0×20** | 2370.7 | **2160.3–2305.4** | **2401.9–2423.8** | 2423.8 | **+2.2 %** |
| Q4_0×4 | 1785.4 | 1720.9–1829.1 | 1826.2–1920.7 | 1920.7 | +7.6 % |
| C0×2 | 1069.3 | 1056.8–1080.7 | 1068.1–1092.5 | 1092.5 | +2.2 % |
| C1×2 | 1069.2 | 1060.0–1106.3 | 1084.6–1096.1 | 1096.1 | +2.5 % |
| C2×2 | 1062.0 | 1062.6–1079.9 | 1068.2–1091.3 | 1091.3 | +2.8 % |
| C3×2 | 1073.1 | 1072.4–1103.7 | 1088.4–1110.8 | 1110.8 | +3.5 % |
| C4×2 | 1069.0 | 1048.5–1090.1 | 1066.8–1107.1 | 1107.1 | +3.6 % |
| C5×2 | 1045.0 | 1058.2–1089.5 | 1067.3–1083.1 | 1083.1 | +3.6 % |
| C6×2 | 1054.4 | 1058.2–1117.7 | 1079.2–1094.0 | 1094.0 | +3.8 % |
| C7×2 | 1060.6 | 547.6–1084.4 | 1047.4–1099.6 | 1099.6 | +3.7 % |
| C9×2 | 1060.6 | 570.0–1082.5 | 1087.0–1106.8 | 1106.8 | +4.4 % |
| BF16×1 #15 | 579.8 | 568.1–621.8 | 613.3–631.5 | 631.5 | +8.9 % |
| **Q4_0×20** | 2370.3 | **2208.5–2338.1** | **2287.4–2404.0** | 2404.0 | **+1.4 %** |

**16/16 clear baseline.** Also green: candle-nn 426/0, candle-conversation 950/0,
both `cargo check` branches, clippy 227.

**On method, because the numbers demand it.** The baseline is one run; the
current tree is 3–5. Run-to-run spread on the wide configs is ~6 %, and the
"before" column contains a visibly degraded run (C7/C9 at ~550, half rate), so
part of the across-the-board lift is machine state and cannot be attributed to
this step. What *is* attributable is stated below, where the ranges separate
cleanly.

### The recycled-region sync: 2,837 ms → 0.5 ms

Step 4 recorded this cost as negligible — 0.029 ms per arena creation — and
deferred the per-region release event on that basis. **That measurement expired
in step 5.** The scheduler used to `device.synchronize()` every wave inside
`release_empty_arenas`, so the queue was always shallow by the time a claim
quiesced; step 5 removed that sweep-path sync (correctly — it was guarding an
unmap that no longer happens), and the claim stopped riding on it.

Re-measured with the flag on, one gate run:

```
before   n=395  quiesces=395  sync_total=2837.1 ms  avg=7.18 ms  max=35.3 ms
after    n=395  quiesces= 15  sync_total=   0.5 ms                max= 0.25 ms
```

2.4 % of the run, blocking, on an allocation path.

The fix is not the per-region event the design proposed. A `cuCtxSynchronize`
retires **every** kernel on every stream, so it discharges the debt of every
region released before it — not just the one that triggered it. Regions come
back in bulk (an eviction drops a turn's arenas together) and are re-claimed in
bulk, so stamping each release with a **quiesce epoch** turns a wait per claim
into a wait per batch: 395 → 15. No events, no stream registry, four lines of
state. `one_quiesce_covers_a_whole_batch_of_releases` asserts the zero guarantee
survives the skip, which is the property the fast path must not break.

The per-region event stays unbuilt, now for a measured reason rather than an
assumed one. `[region-recycle]` reports the quiesced/total ratio so the moment
batching stops working is visible rather than inferred.

### The class ladder was padding the two formats that a whole config is built on

The gate's Q8_0×20 and Q4_0×20 were the only two configs below baseline, and
they were below it in **five runs out of five** — not noise around it. Both are
20-context, i.e. the widest KV working set, where read bandwidth dominates.

The ladder was derived from the adaptive C-level formats, and every one of those
lands on a rung exactly. The two *fixed* formats did not:

| format | payload | old class | pad |
|---|---|---|---|
| Q8_0 | 1088 B | 1152 B | 5.6 % |
| Q4_0 | 576 B | 640 B | 10.0 % |

A slot's stride is what the kernels step by, so that pad is re-read on every
attention pass for as long as the chunk lives.

Writing the test first was worth it: it disproved the framing it was written
under. *Fourteen* formats padded, not two — so a zero-padding ladder would mean
one rung per format, which is per-format arenas again, which is what size
classes replaced. The rule the measurements actually support is **coarse where
stranding dominates, exact where bandwidth does**: keep one catch-all rung at
320 B absorbing the eight formats from 32 B to 288 B (§3.10 problem 3 — each
would otherwise stamp a whole region for a trickle), and give every format above
it its own rung. Ladder 7 → 13 rungs.

Result: Q8_0×20's five old runs peak at 2305.4 and its three new runs bottom at
2401.9 — disjoint, and the whole new range is above baseline. Q4_0×20 likewise.
Those two are the ones this change can explain; the rest of the table moved too,
and the confound above applies.

The trade it costs: Q8_0 (1088 B) no longer shares a class with Q8_KS (1152 B),
and C4/C5 offer both as K candidates, so a session using both may hold one extra
region. At ~15 k slots per region that is a rounding error against 5.6 % of every
band read. `formats_sharing_a_class_are_fungible` asserts the split explicitly
so it reads as a decision rather than an accident.

### Not built, and why

- **Boundary motion (§9 S6).** Measured, not earned: KV peaked at 167 of 226
  regions, so it never starved, and the transient side's 704 MiB is the sum of
  three declared budgets rather than slack. §9 S6 says build motion only if the
  fixed split strands real memory. It does not. **This is now never built.**
- **The per-region release event.** Superseded by the epoch stamp above.
- **A low-end class split (A12).** Its decision rule is "sub-320 formats exceed
  ~2 % of live slots". Zero class promotions fired across the whole gate, so no
  class was starved; the sub-320 occupancy tally the rule needs is still
  uninstrumented, and the split stays unbuilt until it exists.

### Still open

- `WaveAllocScope` for interior op outputs (step 3's deferred option (b)).
- The static shelf still has no allocator; `SHELF_BYTES` is address space held
  for `ProvSignScratch`, the sampler scratch and the MoE routing buffers, which
  remain grow-only pool allocations. This is unfinished step-2 work.
- The threaded expert pipeline's `ys` — the last per-layer allocation on this
  card. Moving it needs the pipeline to stop returning ownership over a channel.

---

## Step 7 — daemon; opened, not finished

### The conversion is a no-op, and that is the right answer

`zend` has no allocator touchpoints to move. Its only VRAM call is
`get_total_vram_device0`, for picking a model quantisation; everything else
reaches the reservation through `candle-conversation` → `candle-nn`. So "move
`zend` onto the unified allocator" was satisfied the moment step 4 landed —
there is no daemon-side allocator to convert, which is what one would hope from
a change made at the right altitude. Release build clean (2 pre-existing
warnings in `repo_scan`).

What remains is the half that needs a live card: the partition, and the
measurement suite.

### The partition double-books the forward's scratch

Found by reading the arithmetic end to end before spending a run on it.

```
expert_budget()  = usable − kv_floor − scratch_margin
kv_span_target() = usable' − scratch_margin           // usable' after experts
RegionPool::create reserves  kv_span + TRANSIENT_SPAN_BYTES
```

`scratch_margin` (1 GiB) is described as "the cushion the first forward's
scratch lands in, before any KV eviction". It is subtracted **twice** — once
when sizing experts, again when sizing the KV span — and then
`TRANSIENT_SPAN_BYTES` (704 MiB) is added on top of the result and claimed from
the same reservation.

But the forward's scratch **is** the transient tier now. That is what step 2 and
step 3 were for. So the cushion and the tier are the same memory, reserved in
two places that do not know about each other, and the reservation ends up asking
for `usable − scratch_margin + 704 MiB` when the honest ask is
`usable − (whatever scratch still lives outside the reservation)`.

This is the same error the design already caught once, in
`balloon_headroom_abs`: "That is true, and it is already reserved —
`scratch_margin` is subtracted in `expert_budget` and sits below every relief
rung. Reserving it here as well booked the same bytes twice." The same sentence
applies here, one layer down.

It is not pure double-booking — the CUDA pool still holds the gallery arena, the
grow-only scratches and the threaded pipeline's `ys`, so *some* cushion outside
the reservation is real. What is wrong is that the figure was sized for a world
where the whole forward allocated from the pool, and nobody re-derived it when
the forward moved.

**Corroborating measurement, from step 4:** the KV side came up at 226 regions =
3,616 MiB. The arithmetic above intends roughly `kv_floor` ≈ 5.1 GiB on this
card. The gap is not silent — `RegionPool::create` already logs `KV side claimed
X MiB of the Y MiB asked for` when the granule touch refuses — so the claim is
being **truncated by the card**, the refusal path working exactly as designed
while the partition asks for more than exists. That is the tuning target.

### What the gate still needs

Everything in step 7's gate is a live-daemon measurement, and none of it is done:
cold ingest, warm restart, concurrency probes with distinct `conv_id`s; expert
residency, decode t/s, aggregate ingest t/s; zero budget-exceeded / no-forward
waves. The suite exists (`zend/tests/*.rs`, all `#[ignore]`d, full
Qwen3-30B-A3B, ~5 min for the smoke tier alone) but a tuning campaign is many
runs, not one.

Two things make this cycle cheaper than it was:

- The `kv-regions` telemetry finally emits (step 5 found it had been inside a
  permanently-false `#[cfg]`), so the partition is now observable from the
  daemon log without instrumenting anything.
- `CANDLE_VRAM_KV_FLOOR_MB` is now the direct KV-side sizing knob, exactly as
  step 7 predicted ("`CANDLE_VRAM_KV_FLOOR_MB` and friends collapse into the
  partition knobs"): experts take `usable − floor − margin`, so the KV side gets
  `floor` less whatever the card refuses. The standing 16 GiB workaround
  (`CANDLE_VRAM_KV_FLOOR_MB=6144`) should be re-derived, not carried — it was
  compensating for a pool that no longer exists.

### Step 7a — the double-booking is fixed

`kv_span_target` now subtracts `TRANSIENT_SPAN_BYTES`, and the identity that
falls out is worth stating plainly: **the whole reservation is exactly
`kv_floor`**, both sides, with `scratch_margin` left outside it on the CUDA pool
where the gallery arena, the grow-only scratches and the threaded pipeline's
combine target still live. `the_reservation_claims_exactly_the_kv_floor` pins it
as pure arithmetic, so it holds without a GPU.

That makes `kv_floor` mean one thing — the VRAM the KV subsystem owns — and it
is no longer a floor in the evict-no-further sense, because nothing evicts
against a watermark any more. It is the partition knob, which is what step 7
tunes. The governor-side docs for both terms were re-derived to match; the old
`scratch_margin` wording ("the cushion the first forward's scratch lands in") is
precisely what let the same bytes be booked twice, since a forward's activations
have come from the transient tier since step 3.

**Gate re-run, green**: 122.59 s, and the KV side is now 704 MiB smaller than it
was. Recycled claims rose 395 → 413 and quiesces stayed at 15 (0.5 ms total);
zero class promotions. The region tier absorbed the smaller span through reuse,
which is the behaviour the free-list is for — the memory went back to the pool
rather than being asked for and refused.

The remaining question is the one this exposes rather than answers: step 4
measured the KV side at 226 regions where the arithmetic intends ~`kv_floor`, so
`usable()` appears to over-read the card by ~1.5 GiB. With the ask now honest, a
truncation warning means that and only that. It needs a daemon run to confirm.

### Step 7b — the partition, measured

The reservation now reports itself under `KV_ARENA_STATS`, on the one channel a
test binary can see (no tracing subscriber there, so the existing `log::` lines
are invisible). One gate run, this card:

```
[reservation] capacity_c=14592MiB usable=4640MiB kv_floor=5143MiB
              scratch_margin=1024MiB | asked=2912MiB claimed=2912MiB
              (182 regions) transient=704MiB | shortfall=0MiB
```

**`shortfall=0`, which refutes the hypothesis I recorded in 7a.** `usable()` is
not over-reading the card; the corrected arithmetic asks for exactly what is
there and the driver grants all of it. The step-4 gap was the double-booking,
and it is gone. (Worth recording that the first version of this instrumentation
read `usable` *after* the claim and reported 1024 MiB — the claim consumes
headroom as it maps. A measurement taken at the wrong moment looked like a
second bug.)

The real defect is one line up. At the first KV cache, `usable` is **4,640 MiB**
while `kv_floor` is **5,143 MiB** — the expert loader was supposed to leave
`kv_floor + scratch_margin` = 6,167 MiB and left 4,640. The KV side therefore
comes out at 182 regions / 2,912 MiB against the ~4,439 MiB (`kv_floor` less the
transient tier) the partition intends. **A 34 % shortfall.**

**Correcting my first reading of this.** I initially wrote that `kv_floor` is
*under*-computed at expert-sizing time. It is the opposite — with the `Weights`
tally still zero there, `kv_floor` reads `3072 + 0.15 × 14592 = 5,261 MiB`,
which is *larger* than the 5,143 MiB it settles at once the weights register.
The inflation protects KV slightly. The deficit is not from mis-sizing the
floor.

It is that **`expert_budget` reserves `kv_floor + scratch_margin` at a moment
when nothing else has been paid yet.** Everything that loads after it comes out
of that same reserve:

```
after experts        6,285 MiB   (= kv_floor_inflated + margin)
- dense weights     -1,049       (summed tensors; they load after the experts)
- gallery + scratch   -596       (gallery arena slabs, grow-only buffers)
= usable at KV       4,640 MiB   (measured)
```

The loader's own comment at the `set_class(Weights, ...)` site understands the
hazard — "leaving this unrecorded computes the floor against the whole card as
though the model were free" — but registers the tally at line 2119, and
`expert_budget()` was called at line 1859. The registration is real and correct;
it simply happens ~260 lines too late to inform the decision it was written for.

So the dense weights, and every post-expert allocation, are paid for out of the
KV reserve. That is what `CANDLE_VRAM_KV_FLOOR_MB=6144` has been compensating
for — the workaround is almost exactly the deficit, which is the sort of
coincidence that names its own cause.

**The fix is not a knob.** The expert budget has to know the dense-weight total
before it sizes itself, which means the loader declaring it up front rather than
crediting it as it goes. That is a real change to the load path and it is where
step 7 continues.

### Step 7 — where it stands, and the exact next change

Done: the conversion (a no-op — `zend` has no allocator of its own), the
double-booking fix, and the partition made observable and measured.

Not done: the fix for the deficit above, and the daemon measurement suite.

**The next change, precisely.** `expert_budget()` must reserve the dense weights
it knows are still coming. Today the sequence is:

```
quantized_qwen3_moe.rs:1859   expert_budget()      <- reserves floor + margin
        ...experts allocated, layers stream in...
quantized_qwen3_moe.rs:2101   base_weight_bytes = dense_bytes.get()
quantized_qwen3_moe.rs:2119   set_class(Weights, base_weight_bytes)
```

`dense_bytes` is a `Cell` accumulated by the tensor-loading closure
(`:1597`, `:1606`), so at `:1859` it holds only the embeddings. The dense total
*is* knowable before then — it is a walk of the GGUF tensor list, which the
loader already has open — so the fix is to compute it up front and
`set_class(Weights, …)` **before** `expert_budget()` rather than after.

That makes `kv_floor` correct at the one moment it is used for a decision, and
it makes the expert budget subtract weights it currently assumes are free. Both
budgets then describe the same card.

Expect it to move expert residency and KV capacity in *opposite* directions —
experts shrink by roughly the dense total, KV grows by it — which is the trade
step 7 exists to put under a knob rather than leave to load order. The gate
should be run with `CANDLE_VRAM_KV_FLOOR_MB` unset afterwards: the 6144
workaround was compensating for exactly this and should not survive it.

### Step 7c — the expert budget now knows what has not loaded yet

**Two attempts, and the first one was wrong in an instructive way.**

*Attempt 1* — declare the dense weights (summed from the GGUF tensor table,
experts and the embedding excluded) with `set_class(Weights, ...)` **before**
`expert_budget()` instead of ~260 lines after it. Measured: KV **182 → 176
regions**. The wrong direction.

Because `kv_floor` is `abs + pct × (C − Weights)`. Declaring the weights
*lowers* the floor, which hands the expert cache **more**, and the weights still
load out of the remainder afterwards. The declaration was necessary — every
later reader of `kv_floor`, including `kv_span_target`, was working from a tally
that said the model was free — but on its own it made the thing it was meant to
fix slightly worse.

*Attempt 2* — the pending bytes come off the **budget**, not out of the floor:

```rust
let gov_budget = candle::vram::get(gpu_id)
    .and_then(|g| g.expert_budget().ok())
    .map(|b| b.saturating_sub(pending_dense_bytes as u64));
```

What `expert_budget()` cannot know is that `usable` includes memory already
promised to tensors the loader has not reached. Only the loader knows that, so
the subtraction belongs at the call site.

| | before | after | Δ |
|---|---|---|---|
| `usable` at first KV cache | 4,640 MiB | **5,216 MiB** | +576 |
| KV side | 182 regions / 2,912 MiB | **218 regions / 3,488 MiB** | **+19.8 %** |
| shortfall | 0 | 0 | — |
| MoE gate | green | green, 127.58 s | — |

Both `set_class` calls stand, and they are not redundant: the early one is an
estimate that informs a decision, the late one replaces it with the true summed
total once every tensor is resident.

**Still ~950 MiB short of the ideal.** `usable` is 5,216 where
`kv_floor + scratch_margin` is 6,167. The residue is the gallery arena's slabs
and the grow-only sampler / provenance / MoE-routing scratches — the allocations
`scratch_margin` is *supposed* to cover, which is the cushion doing its job
rather than a leak. Whether 1 GiB is the right cushion for them is a
`scratch_margin` question and needs the daemon, where the gallery is actually
populated; in the gate it is nearly empty.

### Step 7 — status

Done: the conversion (a no-op); the scratch double-booking; the partition made
observable, measured, and corrected for load order — **+19.8 % KV capacity on
this card, with the ask honest and no truncation**.

Not done, and honestly named: the daemon measurement suite. Cold ingest, warm
restart, concurrency probes, expert residency and ingest t/s against the
pre-unification baselines all need `zend` under real load — the suite exists
(`zend/tests/*.rs`, `#[ignore]`d, full Qwen3-30B-A3B) but the cheapest entry
point walks the whole workspace with a 24 h timeout. `CANDLE_VRAM_KV_FLOOR_MB`
should be re-derived there with the workaround unset; it was compensating for
the deficit this step removed.

### Step 7d — the daemon, measured

`zend` booted on the reservation, `--wipe-substrate`, ingest layers disabled so
startup is the projection population rather than a workspace sweep. **This is
the first time the `kv-regions` telemetry has ever emitted** — it was inside the
permanently-false `#[cfg]` that step 5 found.

| run | binary | KV side | usable | shortfall | outcome |
|---|---|---|---|---|---|
| before the step-7 fixes | 09:11 | 226 regions / 3,616 MiB | 4,640 | — | boots |
| after, default floor | 09:58 | 218 regions / 3,488 MiB | 5,216 | 0 | **fails** |
| after, `KV_FLOOR_MB=6144` | 09:58 | **384 regions / 6,144 MiB** | 7,872 | 0 | **healthy** |

**The daemon's KV side went *down* 226 → 218, and that is correct.** The
double-booking fix returned 128 MiB to the CUDA pool, which now holds its
designed 1,024 MiB cushion instead of the 320 MiB it was left with. The gate
gained KV because its `usable` was the binding term; the daemon traded a little
KV for the cushion it was supposed to have. Reporting the gate's +19.8 % as the
result of this work without the daemon's −128 MiB would have been a half-truth.

**The middle row is the real finding.** With the shipped default floor the
daemon exhausts the reservation during boot:

```
decode-less wave step failed: kv-cache GPU VRAM budget exceeded:
every region of the KV reservation is occupied (218 live)
```

and the relief sequence fires (`want_mib=336`, `560`) without freeing anything —
correctly, because boot-time projection content is pinned and not warm-backed,
so there is nothing evictable yet. 40,857 arena creations at 0.002 ms each, so
the allocator is not the problem; the partition is simply too small.

`CANDLE_VRAM_KV_FLOOR_MB=6144` fixes it, and now it is *sized* rather than
folklore: the `[reservation]` line shows the knob producing exactly what it
asks for, no truncation, 384 regions with 325 free at steady boot. **The
workaround was not compensating only for the load-order deficit — it is a real
statement that this model on this card needs ~6 GiB of KV, and the shipped
`3 GiB + 15 % × (C − Weights)` default under-provides it by a factor of ~1.75.**
The default is what should change; step 7's tuning conclusion is that the floor
formula, not the knob, is wrong for a 16 GiB card running a 30B MoE.

**Not done**: the throughput half of the gate (cold-ingest and decode t/s,
expert residency against the pre-unification baselines). The daemon is healthy
and forwards run, but a like-for-like throughput comparison needs the ingest
layers enabled, which is the multi-hour sweep.

---

## Step 7 (continued) — the partition, measured

### First: the previous section's experiment was malformed

The three runs above are not comparable and two of their verdicts are wrong.
Checking the logs before building on them:

| run | wall | kv-regions samples | state at kill |
|---|---|---|---|
| "before" | 2 min | 32 | `live=peak=70`, arena 70 just created, ingest still firing |
| "after, default" | 6.5 min | 164 | wedged |
| "after, 6144" | 3 min | 45 | `live=peak=76`, ingest still firing |

The first and third were **killed while still climbing** — one region every
~2.3 s, no release ever observed. "Before boots" and "6144 is healthy" were
both readings of a run that had not yet reached the wall. Only the middle run
ran to a conclusion, and it is the one that failed. A duration confound, and it
produced two false verdicts in the same table.

The workspace was also empty of content (`.substrate` only), so what was being
ingested was not the real corpus.

### Re-run properly: cold ingest has a transient, and it is what binds

Real `mind` content (36 MB, `identities/` + `lore/` + `personalities/` +
`responses/`), fresh substrate, run to completion rather than to a timer.

```
[reservation] capacity_c=14592MiB usable=7872MiB kv_floor=8215MiB
  scratch_margin=1024MiB | asked=6144MiB claimed=6144MiB (384 regions)
  transient=704MiB | shortfall=0MiB
```

The region trace is the result:

```
7 … 41   slow climb, ~1 region per section        (initial sections)
41 → 233 fast climb                                (the five collections)
       ↳ peak 284
233 → 69 one bulk release                          (flush at end of ingest)
69 …     flat for the rest of the run
```

**Cold ingest peaks at 284 regions (4,544 MiB); steady state is 69 regions
(1,104 MiB).** Boot needs 4.1× what running needs, the peak decides whether the
daemon comes up, and relief fired **zero** times because 284 < 384 — nothing was
ever under pressure. 0 errors, 0 budget-exceeded.

That single number explains the failure above exactly: the default gave 218
regions, which is below 284, so boot exhausts the reservation and relief frees
nothing — the content is pinned and not yet warm-backed, so there is genuinely
nothing to evict.

### Why the peak is that large: the offload runs on one arm only

`OffloadCollectionMembers` — quantize the prefix-transparent members, then block
on a persistence flush so `install_cold` frees their VRAM before the next batch
prefills — exists precisely to bound this, and its comment says so: *"so the
native catalog never exceeds one branch"*. It is called from the
`SystemPromptItem::SectionTree` arm, per branch.

It is **not** called from the `SystemPromptItem::Collection` arm. The schema has
one `section_tree` and **five `collection` items**, all 133 inserts came from the
un-offloaded arm, and `grep -c offload` over the whole boot log returns **0**.
So the five collections accumulate hot for the entire system prompt and release
in one act at the end — which is the 41 → 284 → 69 trace above.

The peak is therefore proportional to the corpus, with nothing bounding it. A
larger workspace needs a larger reservation, forever, and the reservation must be
sized for the largest corpus that will ever be ingested. That is the same
"arbitrate between disagreeing estimates of is there room" shape the whole design
set out to remove — surviving in the one place the design never looked.

### The sweep

Same corpus, same cold start, `kv_floor_abs` varied:

| `kv_floor_abs` | kv_floor | usable@KV | KV span | expert slots | residency | boot |
|---|---|---|---|---|---|---|
| 3 GiB (old default) | 5143 MiB | 5216 | 218 regions / 3488 MiB | 2618 | 42.6 % | **fails** — retry storm, no forwards |
| 4 GiB | 6167 MiB | 6112 | 274 regions / 4384 MiB | 2267 | 36.9 % | comes up; one drain WARN at the peak |
| 6 GiB (the workaround) | 8215 MiB | 7872 | 384 regions / 6144 MiB | 1566 | 25.5 % | clean, 100 regions never used |

Slope: **1024 MiB of `kv_floor_abs` buys 56 KV regions and costs 351 expert
slots.** The KV side moves 896 MiB per 1024 MiB of floor; the rest is lost to
expert-cache slot granularity.

Note the 4 GiB row's failure mode is *not* the 3 GiB row's. 3 GiB is a hard
ERROR loop (`decode-less wave step failed`, repeated, daemon never ready). 4 GiB
reaches `daemon ready` and serves requests; the one casualty is a post-priming
quantize drain that could not stamp a class-1088 slot at the instant the
reservation was full. Marginal, not broken — but it means 274 regions sits
*inside* the peak, and the peak is not a number to sit inside.

### `kv_floor` is not achieved, and nothing said so

The 4 GiB row asks for `kv_floor=6167 MiB` and the KV side ends up owning
`4384 + 704 = 5088 MiB`. The floor is short by **1,079 MiB** and every
diagnostic reads clean, because `shortfall` measures only the granule touch
refusing — a floor that was never achievable in the first place is invisible.

`expert_budget` takes `usable − kv_floor − scratch_margin` *before* the dense
weights finish loading, so what actually survives to the KV claim is smaller than
the floor by whatever loaded in between. The step-7 `pending_dense_bytes` fix
narrowed this; it did not close it.

Added `floor_deficit` to the `[reservation]` line and a `log::warn!` naming the
trade when it is non-zero. This is the same rule the transient tier already
follows — a span that cannot be honoured says so and names its own budget —
applied to the one term the partition is actually tuned against.

### The offload cannot be extended to the collection arm — tried, refuted

The obvious reading of the section above is that the 284-region peak is a missing
call: `OffloadCollectionMembers` bounds the tree arm's hot set, the collection
arm never calls it, so call it there too and the peak collapses toward the
70-region steady state. The supporting evidence looked strong —
`insert_section_collection_with_progress` documents that members
*"don't attend to each other, so each member's `prefix_hash` is the chain state
before the collection"*, and that **"collection members are excluded from the
chain"**. That is exactly the prefix-transparency the tree arm's offload relies
on.

Built it, cold-booted it. **Both predictions failed:**

```
41 → 225 regions          identical trajectory; no reduction whatever
53 × budget exceeded
WARN prepare_section_ingest: prefix section SectionId(154)
     has no sealed substrate entry — skipping     (…155, 156, …)
ERROR inference engine failed to load: base conv create
```

The distinction that matters is one the tree arm states about itself and the
collection arm does not share: *"It never extends `linear_prefix`"*. Tree-embedded
members are sealed per branch and nothing prefills against them afterwards. Plain
collection members **are** pushed onto `linear_prefix`, precisely so *"subsequent
sections see them at projection time"* — so every section declared after a
collection prefills over its members. Offload them and that prefix is gone:
`prepare_section_ingest` finds no sealed substrate entry, silently skips it, and
the system prompt is built wrong before the reservation ever runs out.

Mutual prefix-transparency *within* a collection is not transparency to what
follows it, and only the second one licenses an offload.

**So the peak is structural, not a defect.** Every collection's members must stay
resident until the last section that prefills over them is built, which is the
end of the system prompt. The cold-boot hot set is therefore the whole collection
span, it scales with the corpus, and the KV reservation has to cover it. Reverted
to the single-line comment recording why, so the next reader does not re-derive
the same wrong inference from the same true comments.

Bounding it is a change to how the system prompt is *composed* — not putting
collection members in later sections' prefixes — which is a projection-schema
question, not an allocator one.

### Decode against expert residency — the curve the partition actually buys

Matched probe: same binary, same prompt, fresh conversation, three runs each,
compared at equal context depth (kv/fwd 490–736 in every arm).

| `kv_floor_abs` | KV span | expert slots | residency | decode median | cold boot |
|---|---|---|---|---|---|
| 3 GiB | 218 regions | 2618 | 42.6 % | — | **dies** — retry storm, never ready |
| **4 GiB** | 274 regions | **2267** | **36.9 %** | **57 ms/fwd** | ready; 1 drain casualty |
| 5 GiB | 328 regions | 1917 | 31.2 % | 67 ms/fwd | clean |
| 6 GiB | 384 regions | 1566 | 25.5 % | 80 ms/fwd | clean, 100 regions unused |

**The 6144 workaround was the worst point on this curve.** It was sized against
pre-unification arena slack that the unification removed, so it held 100 regions
that nothing ever occupied while starving the expert cache to pay for them.
Default is now **4 GiB and the knob is retired**: decode is **29 % faster per
forward** than the configuration this repo has been running.

Note the 3 GiB and 4 GiB failure modes are different things. 3 GiB is fatal —
`decode-less wave step failed` in a retry loop, `daemon ready` never reached.
4 GiB reaches ready and serves; its one casualty is the post-priming quantize
drain, which needs a slot to write a compressed section into *before* it can
release the native one, and at the end of a cold build there is no slot to be
had. Reproduced on two independent boots, so it is a property of the
configuration, not a flake.

Tried relieving before that drain, on the same `vram_under_pressure()` test every
other allocation site uses. **It does not help**: at that instant all 274 regions
are legitimately pinned by the priming projection, so relief has nothing to free
— it frees 41 a moment later, once the projection releases its refs. Reverted;
a call that cannot achieve its purpose is worse than none. The drain casualty is
under-provisioning by ~10 regions, not a scheduling mistake.

### The transient tier reserves 704 MiB and uses 14

Measured live on the daemon:

```
kv-transient wave:    peak=14MiB (a=14MiB b=0MiB)  cap=64MiB each
kv-transient persist: cursor=0MiB peak=0MiB        cap=512MiB
shelf:                                             64MiB, no allocator exists
```

`TRANSIENT_SPAN_BYTES = 2·WAVE_HALF(64) + MIGRATION_STAGING_CAP(512) + SHELF(64)
= 704 MiB`, against a measured high-water mark of **14 MiB**. Two of the three
terms are budgets rather than watermarks — the migration cap is a declared batch
size, and the shelf is address space held for an allocator that was never built
— but they are charged to the reservation all the same, at the exact boundary
where 10 more regions would remove the 4 GiB casualty.

**690 MiB is 43 regions.** Reclaiming it is the one lever that widens KV without
taking anything from the experts, and it is worth more than the whole 4→5 GiB
step. The persist term needs a real migration workload before it can be cut
honestly (this run had `residences=0`); the shelf's 64 MiB is unambiguous — it
backs nothing.

### Gates after the change

`cargo check --workspace --tests` and `--features cuda` both 0 errors; clippy
227 (baseline); candle-nn **422**/0, candle-conversation **950**/0, candle-core
`vram::` 27/0.

`a_released_region_comes_back_lowest_first` was failing intermittently and is
fixed. It asserted two exact region indices from a **process-global** pool while
other modules' tests claim from it in parallel — `SERIAL` only orders this
module against itself — so a concurrent claim could take a just-freed region
between the drop and the re-claim. Now asserts the invariant (claims come back
ascending, lowest-first) rather than the interleaving. Verified across parallel
and `--test-threads=1` runs.

The residual `CUBLAS_STATUS_NOT_INITIALIZED` burst when suites are launched
back-to-back is the known context teardown/setup contention, unrelated: those
runs abort in ~12 s against ~34 s for a real pass.

---

## Step 7 — the transient tier, and what the daemon suite found

### `S = 2·W_wave + W_persist` — 704 MiB → 192 MiB

The span was 44 regions taken off the KV side before one was carved, on the card
where those regions decide whether the expert cache is fed. Two of its three
terms were unearned:

| term | was | now | measured peak (batch 64) |
|---|---|---|---|
| `2·W_wave` | 128 MiB | **128 MiB** | 61.6 MiB — kept, ~2x headroom |
| `W_persist` | 512 MiB | **64 MiB** | 29,696 B |
| `shelf` | 64 MiB | **removed** | nothing allocates from it |

The wave halves are load-bearing and stay: 30.8 MiB per half measured, and
exhausting one fails the forward. The shelf reserved address space for a static-
shelf allocator that was never built — the sampler, provenance and MoE routing
scratches still grow from the CUDA pool, which `scratch_margin` already covers.

`W_persist` is the interesting one. Its 512 MiB was defended as "a declared
budget, not a watermark — the batch bisects itself to fit". That argues the span
can be *small*; it was being used to argue it should be *large*. The cost of a
big span is paid in KV regions on every boot; the cost of a small one is paid in
DtoH syncs on the hot→warm path only. Sized instead from the floor the existing
batch-halving retry already handles — a single ~30 MB layer — giving 64 MiB, two
layers plus headroom, ~22 syncs for a ~1.4 GiB pass against the per-layer 48
that made `copy_ms` the bottleneck.

**Measured, cold boot, `kv_floor` 3.5 GiB:** transient 704 → 192 MiB, expert
slots **2267 → 2443 (39.8 %)**, KV span **274 → 278 regions**. Both sides gained;
the 512 MiB came out of reserved-but-unoccupied space. Against the 6144
workaround: **1566 → 2443 slots, +56 % resident experts.**

### Decode saturates, and it saturates before the partition runs out

| expert slots | residency | decode median |
|---|---|---|
| 1566 | 25.5 % | 80 ms/fwd |
| 1917 | 31.2 % | 67 ms/fwd |
| 2267 | 36.9 % | 57 ms/fwd |
| **2443** | **39.8 %** | **57 ms/fwd** |

The curve is flat past ~2,300 slots on this model. Slots bought beyond that are
free to spend elsewhere — which reframes the whole partition question: the
binding constraint stopped being decode.

### The 284-region peak is a **first-boot-only** cost

The single most useful thing the daemon suite found. Arena creations by location:

| boot | GPU arenas | section-ingest phase |
|---|---|---|
| cold (fresh workspace) | **284** | 76.0 s over 133 inserts |
| warm (substrate present) | **11** | **0.0 s** over 133 inserts |

On restart the 133 collection members are *restored*, not re-prefilled — the
whole ingest phase completes inside one second and touches 11 GPU regions
instead of 284. Time-to-ready 132 s → 69 s, and the residual 69 s is almost
entirely the 18.8 GB weight load (`elapsed_ms=69271`), not KV work.

So sizing `kv_floor` to clear 284 taxes **every run for the daemon's lifetime**
to survive **one event per workspace**. At 278 regions the first boot of a fresh
workspace emits one `post-priming section quantize drain failed` warning and
comes up; every boot after that is clean. That is the right trade on this card,
and it is why the shipped default sits below the cold peak rather than above it.

(`substrate reload complete sections=0` is not evidence against this — sections
come back through `substrate.section.hot`, not the redo-log replay counter. The
arena-create counts are the measurement that settles it.)

### Concurrency — step 7's gate condition

Six concurrent conversations, distinct `conv_id`s, warm daemon at the shipped
default (278 regions):

```
6/6 HTTP 200, aggregate wall 50.2 s
budget exceeded: 0    ERROR: 0    relief events: 0
peak regions under load: 34 of 278   (12 %)
waves batched 3 sequences, kv/fwd ~1900-2150, 78-89 ms/fwd
```

**Zero budget-exceeded, zero no-forward waves, zero relief events** — the gate
condition, met. Concurrency capped at 3-4 by `max_concurrent`, not by VRAM: the
KV side never exceeded 12 % occupancy. Aggregate ~37 tok/s at 3-way against
~17.5 tok/s single-stream, so batching is returning ~2.1x.

**The reservation is sized for an event that happens once per workspace, while
serving uses an eighth of it.** That is the next thing worth attacking, and it is
a projection-schema question (how collection members enter later sections'
prefixes), not an allocator one.

### Still not done

Aggregate ingest t/s against the pre-unification baseline — it needs the ingest
layers enabled, which is the multi-hour workspace sweep. The free-region setpoint
still has never bound anything: it was not reached in any run here.

### `scratch_margin`: 1024 → 512 MiB, paired against `kv_floor`

The 1 GiB cushion held outside the reservation for what still allocates from the
CUDA pool. Its own doc comment asked for exactly this: *"Re-derive it against
what is actually left on the pool rather than against a forward's activation
peak."* Measured on the daemon under six concurrent conversations:

```
kv-pool reserved: 7424 MiB, flat from end-of-load onwards — it does not grow while serving
kv-pool used:     6510 → 6837 MiB, a ~330 MiB swing INSIDE what is already reserved
card at peak:     13,797 MiB of 16,376   (desktop baseline 453 MiB)
capacity_c:       14,592 MiB
```

Serving takes nothing from this cushion — the pool has its memory by the end of
load and only moves *within* it. What the cushion has to cover is pool growth
between `expert_budget` being computed and load finishing, and roughly a
gigabyte of the governor's own budget was never touched at peak. 512 MiB is
~2.2x the measured demand.

**Paired with a matching `kv_floor` raise so the expert cache is untouched.**
`expert_budget = usable − kv_floor − scratch_margin`, so +512 on one and −512 on
the other cancels there, while `kv_span = usable − scratch_margin − transient`
gains the whole 512 MiB. Measured:

| | before | after |
|---|---|---|
| `scratch_margin` | 1024 MiB | **512 MiB** |
| `kv_floor_abs` | 3840 MiB | **4352 MiB** |
| KV span | 278 regions | **324 regions** |
| expert slots | 2443 | 2355 |
| cold boot | 1 drain casualty | **0 exceeded, 0 drain, 0 errors** |

The expert side moved 88 slots rather than staying flat: `kv_floor` is
`abs + pct × (C − Weights)`, and the `Weights` tally differs between the two runs
at the moment the KV side is claimed, so the percentage term is not the constant
the cancellation assumed. The cold-boot peak (285 GPU arenas) is now fully
inside the span, which is what was being bought.

**One invariant fired and was right to.** `the_balloon_reserve_does_not_double_
book_the_scratch_cushion` failed immediately: `balloon_headroom_abs` (1 GiB) must
stay at or below the cushion, or the balloon reserves engine headroom a second
time — the same double-booking this work already fixed once. Moved to 512 MiB.
It does not bind on this card either way: `C = min(frac × total, total − this)`
targets 15,864 MiB while the balloon actually stops at 14,592 MiB where the
driver refuses.

### Decode is flat above ~2,300 slots — stated as a band, not a curve

| expert slots | 1566 | 1917 | 2267 | 2355 | 2443 |
|---|---|---|---|---|---|
| decode median | 80 ms | 67 ms | 57 ms | 63 ms | 57 ms |

The first three are a real trend. **The last three are one flat band inside
run-to-run noise**, and 2355 measuring *slower* than 2267 is what proves it —
residency cannot make decode worse. Earlier text in this document quoting
"2443 → 57 ms" as a point estimate was reading precision into noise; the honest
claim is that the expert cache stops paying near 2,300 slots on this model.

That is what licenses spending slots on the KV side instead, and it is why the
first-boot casualty could be removed for free.

Gates after: checks 0/0, clippy 227, candle-nn 422/0, candle-conversation 950/0,
candle-core `vram::` 27/0, fmt clean.

### `WaveAllocScope` and the threaded pipeline's `ys` — both unearned, measured

The two remaining allocator items both target the same thing: allocations that
still come from the CUDA pool rather than the reservation. Step 3 deferred
`WaveAllocScope` (interior op outputs — the temporaries candle's own ops
allocate, including the inter-layer hidden state) with the gate "step 6, if the
leak counter stays clean". The counter is clean, so the gate is open and the
question is whether the pool remnant costs anything.

**It does not.** Across a full cold boot — 284-region ingest peak, wide prefills,
then six concurrent conversations:

```
pool reserved: 30 MiB → 7,232 MiB, then FLAT
               (three distinct values across 60 samples, including all of ingest)
pool used:     6,276 → 6,645 MiB — a ~370 MiB swing INSIDE what is reserved
```

The pool reserves once during load and never grows again. Interior op outputs
and the threaded pipeline's `ys` recycle inside that flat allocation, so they
cost **no VRAM** — the reservation is what bounds the card, and the pool remnant
sits under `scratch_margin`, which the same run shows untouched at peak.

So moving either onto the wave would return nothing to the expert cache or the
KV side, which is the only currency this design trades in. `ys` additionally
needs the pipeline to stop handing its combine target back over a channel —
allocated on the pipeline thread, consumed on the caller's — so the scope that
would hold it open belongs to a different thread than the one that allocates it.
Real restructuring for a measured zero.

**Both are now unbuilt by measurement rather than by omission**, joining boundary
motion and the A12 low-end split. They become earned if either of two things
changes: `scratch_margin` needs to go below the ~370 MiB working swing (it is at
512 MiB, so there is one step of room left), or CUDA graph capture is pursued,
which wants stable allocation addresses across waves and is the case
`WaveAllocScope` was really designed for.

`wave_zeros` already places the *inline* MoE combine target on the wave
(`quantized_qwen3_moe.rs`); only the threaded path's `ys` remains on the pool.

---

## Code review — six defects fixed, one of them mine from this session

A `/code-review` pass over the whole change set found things the self-review
missed. Six were fixed; the arithmetic of the two most important is recorded
here because both were *masked by a comment or a test that asserted the wrong
thing*.

### `MIGRATION_STAGING_CAP_BYTES` 64 MiB → back to 512 MiB — regression, mine

Cutting this to 64 MiB earlier today was wrong and is reverted. Three sites
stage against it; **only one bisects**:

| site | batches? |
|---|---|
| `migrate_sealed_layers_to_cpu_batch` (hot→warm, cross-layer) | yes — grows a batch to the cap, halves on OOM |
| `migrate_sealed_to_gpu_batch_async` (warm→hot elevate) | **no** — one `bump.alloc(total_bytes)` |
| the hot→warm per-layer gather | **no** — same |

`persistence/elevate.rs` issues the elevate **once per layer across every warm
item**, so its `total_bytes` is bounded by nothing in this file. For those two
the cap is not a batch size that shrinks to fit, it is a hard ceiling, and at
64 MiB a deep elevate becomes a failed forward.

The justification was doubly unsound. The doc comment claimed 64 MiB was "two
layers plus headroom, and no batch is ever forced below what the code already
handles" — true of the bisecting site, false of the two that do not. And the
29,696 B measured peak that licensed the cut came from runs where **the elevate
path never executed** (`residences=0`, persist domain `cursor=0 peak=0`): a
measurement of the path not running, read as a measurement of it running
cheaply. The constant now carries both facts and the condition for lowering it.

Net after the revert: transient span **704 → 640 MiB** (the 64 MiB shelf removal
stands, the staging cut does not). Re-measured cold boot: **296 regions, 2355
expert slots, 0 budget-exceeded, 0 drain failures, 0 errors**, with the
285-arena cold peak still inside the span.

### The slot-state ladder's top rung is 40x smaller than its test claimed

`SLOT_STATE_LADDER`'s top rung is 1 MiB, and both the doc comment and
`the_top_class_holds_a_deep_sequence` computed its capacity as `top / 16` =
65,536 chunks ≈ 2 M tokens — dividing by the 16-byte *slice header* alone. A
real entry is the header **plus** one `KvHead` record per head:

```
kv_head_record_bytes(128) = (128/4)*2 + 32 + 32 + 4 + 4 + 16 + 16 = 168 B
entry = 16 + 4 * 168                                              = 688 B
top rung = 1 MiB / 688 B = 1,524 chunks = 48,768 tokens
```

So the real ceiling is **~48.7 K tokens for one sequence in one layer**, not
2 M — and past it `claim` errors and the forward fails, where the `stream.alloc`
this tier replaced had no cap at all. The test now asserts against
`chunk_record_bytes(4, 128)` and pins 1,524 / 48,768, so the number cannot drift
back into looking safe; both doc comments were carrying the same false
arithmetic and now state the ceiling.

**This is a real limit on an unbounded-context engine and it is not fixed** —
raising it is not free, because a slab is `SLOTS_PER_SLAB (512) x class`, so a
2 MiB rung implies a 1 GiB slab. The sizing needs rethinking, not another rung.

### The rest

- **`pending_dense_bytes` counted a 2D checkpoint's whole expert set as dense**
  (mine, this session). The filter matched only the 3D merged
  `*_exps.weight`; the loader also accepts a 2D fallback naming experts
  `blk.{i}.ffn_{gate,up,down}.{j}.weight`. On such a checkpoint every expert
  byte lands in `planned`, so `expert_budget().saturating_sub(planned)`
  saturates to **0** and the LRU cache is built with no GPU slots — a silently
  crippled load. Replaced with `is_expert_tensor`, which recognises both layouts
  and still counts a *dense* layer's `ffn_gate.weight` (no expert index) as
  dense; pinned by `expert_tensors_are_recognised_in_both_gguf_layouts`.
- **Ingest demote watermark truncated to zero.** `stats.total / 100 * pct`
  divided first. Harmless when the term was `capacity` in bytes (~1.6e10); on a
  region *count* it quantises to whole percent-of-100 steps, and on any span
  under 100 regions it is **0**, which `live <= watermark` can never satisfy —
  turning the gentle rung into a full demote of the ingest tail every wave.
  Now `total * pct / 100`.
- **Warm-starvation nudge was unconditionally true.** It compared
  `vram_pool_stats().used` (~6.5 GiB: model + experts + scratches, since KV
  moved to the reservation) against a region-derived watermark (~2.4 GiB), so
  `nudged` recorded nothing. It now tests `report.bytes < target_bytes` — whether
  step 1 could shed what it needed — which is what the comment describes.
- **The perf dashboard read four removed fields.** `zend/web/perf.html` still
  consumed `kv.float_*` / `kv.quant_*` and `g.evictable_moderate_bytes`, which
  the size-class `KvSection` replaced, rendering `undefined arenas` / `NaN` on
  the panel this work is tuned against. Now renders one row per occupied size
  class plus a total. A reminder that a Rust-only review misses the consumers.

Gates after: checks 0/0, clippy 227, candle-nn **422**/0, candle-conversation
**950**/0, fmt clean, and the daemon re-measured above.

### Second pass: the remaining review findings

**The slot-state ceiling is fixed, and the objection to fixing it was wrong.**
I reported that raising the ladder was blocked because "a slab is
`SLOTS_PER_SLAB(512) x class`, so a 2 MiB rung implies a 1 GiB slab". The code
already says otherwise:

```rust
let per_slab = SLOTS_PER_SLAB.min(TARGET_ARENA_BYTES / slot_bytes).max(1);
```

A slab is bounded in **bytes**, not slots — the 1 MiB rung already gets 16 slots,
not 512. So every rung from 1 MiB up carves one 16 MiB region holding 16/8/4/2/1
slots, and extending the ladder is nearly free. `SLOT_STATE_LADDER` now runs to
16 MiB: **48.7 K → 781 K tokens** for one sequence in one layer. Rungs are
claimed lazily, so a deployment that never runs deep never allocates them.

Two existing tests failed and both were right to. `every_class_tiles_a_slab_exactly`
required ≥ 16 slots per slab for *every* rung — the assumption that had capped
the ladder at 1 MiB in the first place; it now holds that floor only for the
shallow rungs a wave actually shares, while the divisibility invariant still
covers all of them. `class_for_picks_the_smallest_that_fits` hard-coded the old
top rung.

**Fixed alongside it:**

- **`meta_pool` defaulted unrecorded band tags to `BF16`** — a *valid float*
  format, so a band whose tag was missing would have had quantized bytes decoded
  as floats. `Invalid` exists precisely so such a band fails the kernel's format
  check; every other unrecorded-tag path already resolves to it. Unreachable
  today (producers always fill `n_kv_head * N_PALETTE` entries), which is exactly
  why it must fail loudly rather than plausibly.
- **`kv_span_target` treated a governor error as "no governor"**, so a transient
  `usable()` failure sized production KV at the 2 GiB `TEST_KV_SPAN_BYTES` and
  said nothing. Absence of a governor is a test binary; a governor that errors is
  a fault, and the two no longer share an arm.
- **`QCudaStorage::quantize` replaced the buffer without resetting `backing`.**
  On a leased storage the quantized bytes landed in a fresh allocation instead of
  the arena slot the lease pointed at, *and* `Drop` then `leak()`d that
  allocation — a permanent VRAM leak per call. `Clone` had already been taught to
  force `Owned`; this was the other place the buffer identity changes.
- **`QMatMul::fwd` now refuses a leased quantized view.** A lease is exactly its
  payload and carries no `MATRIX_ROW_PADDING`, while both matmul kernels gate on
  element count and then address `pad(ncols, MATRIX_ROW_PADDING)` columns — so a
  lease passed the guard and read up to 512 elements past the slot. The
  restriction was documented on both constructors and enforced by nothing.
- **Two stale doc comments corrected.** `per_head_table_host`'s safety note
  credited "the migrate-in-flight guard" with blocking arena free/relocate;
  `migrate_flight` is an advisory counter with no mutual exclusion, and what
  actually holds is gid pinning plus the reservation making arena bases
  permanently valid. And `gid_pool`'s `TEST_CLASS` said "2048 B … holds `Q8_0`"
  when rung 5 is 640 B and `Q8_0` is the 1088 B rung — wrong twice.

**Two were diagnosed rather than repaired, deliberately:**

- **Fork at a quantized boundary block.** Destinations come from the *active*
  key (R16, 4096 B class) so decode can append to the forked tail, while a
  sealed partial tail is quantized like any other chunk and can sit in the
  1088 B class — `copy_slot_bytes` then refuses the stride, correctly. The
  comment claiming size classes made the formats agree, and that the old
  `Quantized→Float` dequantize arm was therefore dead, had it backwards: the
  class follows the format, and fork deliberately changes the tail's format.
  Closing it needs that arm rebuilt against tag-driven band reads including the
  dim-major→token-major transpose. Writing that blind risks silent corruption
  where today there is a loud bail, so the mismatch is now detected at the fork
  site and named there instead of surfacing as a bare stride error.
- **Gallery eviction in the relief ladder.** `evict_lru` returns pages to the
  gallery's own `PagePool`; the VRAM is in `storage.slabs`, which is only ever
  appended to and comes from the CUDA pool — so it cannot move
  `region_stats().free`, the signal it is gated on, and it therefore fires on
  essentially every pressure episode, shedding belief-scan residency that the
  next scan rebuilds. But it cannot simply be removed: the arena adds a slab
  whenever its pool empties and **never evicts itself**, so this is the only
  bound on gallery growth. It is a gallery budget wearing a relief rung's
  clothes and wants its own footprint-vs-cap trigger; documented in place so
  `gallery_freed_mib` is not read as relief.

**Left alone, with the reason recorded:** batching the two non-bisecting staging
sites. That is the change that would make `MIGRATION_STAGING_CAP_BYTES` tunable
and hand ~28 KV regions back, but it is a refactor of the async migrate path —
and the elevate path was never exercised in any run here (`residences=0`), so
there is no harness to validate it against yet. It needs that harness first.

Gates: checks 0/0, clippy 227, candle-nn **423**/0, candle-conversation **950**/0,
candle-core 58/0, candle-transformers expert-layout test green, fmt clean on
every crate the standing rule permits formatting.

---

## Third pass: the three that were only diagnosed, now repaired

### The slot-state objection was wrong, and the ceiling is fixed

Reported as blocked because "a slab is `SLOTS_PER_SLAB(512) × class`, so a 2 MiB
rung implies a 1 GiB slab". The code already bounded a slab in **bytes**:

```rust
let per_slab = SLOTS_PER_SLAB.min(TARGET_ARENA_BYTES / slot_bytes).max(1);
```

The 1 MiB rung already got 16 slots, not 512. So `SLOT_STATE_LADDER` now runs to
16 MiB — **48.7 K → 781 K tokens** per sequence per layer — and every rung from
1 MiB up carves one 16 MiB region holding 16/8/4/2/1 slots, claimed lazily.
`no_rung_carves_a_slab_larger_than_a_region` pins that.

Two existing tests failed and both deserved to. `every_class_tiles_a_slab_exactly`
demanded ≥ 16 slots per rung — the assumption that had capped the ladder — now
scoped to the shallow rungs a wave shares, with divisibility still covering all.

### All three staging sites batch, so the cap is a batch size again

`staging_groups(lens, cap)` splits consecutive band lengths into groups that fit
the span, always taking at least one element so a band wider than the cap forms
its own group instead of looping forever. Both previously-unbatched sites use it:

- `migrate_sealed_to_gpu_batch_async` (warm→hot elevate) — offsets are contiguous
  in `unique_raws` order, so a group is one slice of the pinned scratch and
  `off - group_start` re-bases it onto that group's staging.
- the hot→warm gather — same shape, DtoH into the group's slice.

Each group opens its own generation, so the cursor rewinds between groups
(`Reclaim::Fence` fences the copy stream first) and peak transient use is one
group rather than a whole pass. `staging_groups_respect_the_cap` covers exact
fits, boundary crossings, the oversized-single-element escape, and
contiguity/coverage over 40 uneven lengths.

**`MIGRATION_STAGING_CAP_BYTES` is now 64 MiB** and the transient span is back to
**192 MiB**. Re-measured cold boot: **324 regions, 2355 expert slots, 0
budget-exceeded, 0 drain failures, 0 errors** — the 28 regions the revert had
cost are back, with the elevate path now safe at any depth.

### Fork across a format change

The destination comes from the *active* key while a sealed partial tail may be
quantized, so the classes differ and the bytes cannot travel verbatim. The band
is now read as floats through **its own tag** and re-encoded into the active
format — `KvFormat::Float` via `write_slot_typed`, `KvFormat::Quantized` (R16 on
GPU) via `quantize_into_slot` — and the band's tag is rewritten to match, since
the tag is the only record of how its bytes decode.

`read_band_chunk` returns the canonical `(CHUNK_SIZE, sub_head_dim)` token-major
tensor for either tag, so the dim-major→token-major transpose the deleted arm
did by hand is already handled — that was the part most likely to be got wrong.

**Coverage caveat, stated plainly:** the same-class path (every existing fork
test) is covered and green; the conversion branch itself is not, because
constructing a live partial block whose bands are quantized needs the
seal/quantize/inject cycle and the float test backings cannot reach it. The
branch is implemented and compiles, and the MoE gate plus the full GPU suite show
the common path is unaffected.

### Gallery eviction now has its own budget

`GalleryArena::cap_bytes` (`ZEN_GALLERY_CAP_MB`, default 512 MiB) enforced at
admission in `ensure_locked`, under the residency guard and before `inner` is
taken — respecting the residency→inner lock order, skipping pins so an active
scan's working set is never pulled out from under it. `evict_lru_locked` was
factored out because the residency mutex is not reentrant.

That makes the arena self-bounding, which it never was — it adds a slab whenever
its page pool empties and never evicted itself, so the scheduler's relief call
was the only limit. That call now fires **only when the arena is over its own
ceiling**, instead of on every KV pressure episode where it could not move the
signal it was gated on and merely shed belief-scan residency.

### `QCudaStorage::drop` does no CUDA work

`data` is `ManuallyDrop<PaddedCudaSlice>`, so the lease path moves the slice out
with `ManuallyDrop::take` instead of `mem::replace`-ing a stand-in built by
`upgrade_device_ptr(0, 0)` — which creates and destroys two `CudaEvent`s under
cudarc's event tracking and `unwrap`s, i.e. fallible CUDA work on a drop path,
which aborts if it fails during an unwind. This is `CudaStorage`'s `Empty`
tombstone trick for a struct with no spare variant.

It adds one obligation — every path that destroys or replaces `data` must dispose
of the old value — and there are exactly two: `Drop` (leak if leased, drop if
owned) and `quantize` (same choice before overwriting). `bytes_mut` needed
`len` read out first, since `DerefMut` makes the two halves one borrow.

### `wave_alloc` refuses an ambiguous half

It serves `domain.current`, the most recently begun half — not the half the
caller's `Generation` pins, which it cannot name (the ambient design is the point:
kernel wrappers allocate without threading a guard). Those coincide only while
one generation is live. With two, the older wave's allocations land in the
newer's half and are freed by its reset, under kernels still reading them.
`begin_wave` refuses a *third* wave; `wave_alloc` now refuses to guess between
two. Unreachable today, which is when a wrong answer goes unnoticed longest.

### `floor_deficit` is now attributable — and it is the pool's own gap

The `[reservation]` line carries the class tallies, and they close the question:

```
capacity 14592 − weights 784 − experts 6871 − usable 5888 = residual 1047
floor_deficit                                             =          1047
```

**The entire `kv_floor` shortfall is the residual**, and it matches the
`kv-pool reserved=7232 / used=6276` gap measured earlier: the CUDA pool holding
~1 GiB it is not using, which `usable()` cannot see because the pool has it
reserved. `kv_floor` names the reserve the expert budget must *leave*, not what
KV receives, and the gap between them is now printed with its parts rather than
left as an unexplained gigabyte.

That gap is the next lever — a post-load `cuMemPoolTrimTo` would return it, worth
~65 regions or the same again in experts — but the trim was removed in step 5
because it synchronously unmaps, and re-adding it needs the in-flight-kernel
guard that went with it. Named, measured, not attempted.
