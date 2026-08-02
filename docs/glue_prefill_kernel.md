# Batched Glue Prefill — kernel design decision

> **Status — implemented.** Collapses the per-boundary structural ("glue")
> prefill into a single batched GPU pass. Follows the removal of the projection
> cache (`unbounded_agents.md` §projection): structural K/V is re-derived every
> projection, so the dominant reproject cost is the glue forwards. The chosen
> path — a `template<bool GAP_FILL>` specialization of the existing paged-prefill
> kernel, plus the scheduler restructure below — has landed.
>
> One deviation from §7.2: the implementation rotates **Q** at the packed
> position (`prefix_len + q_pos`) rather than `col_actual_pos`. That matches the
> glue row's true position while glue is the contiguous writer tail — the only
> case the scheduler emits — so K and the causal mask use the true
> `col_actual_pos` and Q stays consistent. Generalizing Q-RoPE to
> `s_query_actual_pos` (for non-tail glue) is a noted follow-up.

---

## 1. Problem

After a projection or reprojection, a slot's prefix is assembled by interleaving
**sealed** content (Arc-cloned substrate K/V, free) with small **glue** runs —
the inter-turn boundary markers (`user_start`, `assistant_end`) and structural
template tokens that must be live-prefilled because their K/V depends on the
runtime causal prefix. The assembled slot looks like:

```
[ glue₀ ] [ sealed turn₀ ] [ glue₁ ] [ sealed turn₁ ] [ glue₂ ] [ sealed §₀ ] ...
  writer     Arc-clone         writer     Arc-clone        writer    Arc-clone
  ~5 tok     (free)            ~5 tok     (free)           ~5 tok    (free)
```

Each glue **island** is a single-digit-token writer run. Today
`apply_segments` (`scheduler/projection_assembler.rs:133`) prefills each island
as its **own** `forward_batched(&[parent_id], &[input])` call — one
single-sequence launch per boundary. A deep projection selects ~10–15 turns →
~12–17 glue launches, each a full-context attention forward. Measured: this is
~94% of the reproject's `apply_ms` (profile: `prefill:forward 979ms ×17`).

Two independent inefficiencies:

1. **Per-island, within a conversation** — N sequential launches for N boundaries.
2. **Per-conversation, across the fleet** — every conversation's projection runs
   its glue serially on the scheduler thread; nothing batches across the dozens
   of sessions that may be (re)projecting in the same wave.

**Goal:** one batched GPU pass — a *glue wave* — that fills every glue island of
every conversation currently waiting on a (re)projection, F16/F16, attending the
correct causal prefix, writing into the pre-allocated writer chunks.

---

## 2. What the existing paged-prefill kernel already provides

The existing kernel (`candle-kernels/src/paged-prefill/paged_prefill_kernel.cuh`,
`__global__ paged_prefill_attn_fwd_chunks_kernel` :2494, launcher :4141) is far
closer to what we need than "a normal prefill kernel" implies. It already:

- **Is true varlen / ragged.** One launch processes N sequences of different
  lengths; `blockIdx.z = batch_idx` (kernel :2534), indexed through
  `cu_seqlens_q` / `q_lens` / `kv_lens`
  (`prefill_utils.rs:819`, meta `batched_layer.rs:137`). Heterogeneous glue-run
  lengths in one call are already supported.
- **Gives every sequence a fully independent, disjoint block table.** Each slot
  is a `SlotHeader{n_slices, write_slice, slices_ptr, position_map_ptr}` →
  `TokenSlice[]` of **raw arena pointers** + per-head palettes/formats
  (`paged-decode/slot_types.cuh`; packed in `slot_state.rs:225,395`). There is
  **no shared block-table range** across sequences — sibling sequences may point
  at arbitrary, disjoint arena sets. So "batch conversations whose prefixes live
  in different arenas" is already a property of the kernel, not a new feature.
- **Fuses the KV writeback.** New tokens' K/V are written into the cache *inside*
  the attention pass (kernel :2901–2952 / :3800–3847) via
  `store_kv_chunk_arena` (:504), at **per-row** destinations
  `s_write_slice_idx[r]` / `s_write_in_blk[r]` resolved host-side. The write
  target is already per-row — scattered writeback needs no new mechanism.
- **Attends a pre-existing in-cache prefix.** The `HAS_PREFIX` path
  (`prefix_len = kv_len − q_len`, :2626) reads prefix columns `[0, prefix_len)`
  from the paged arena with full C0–C9 + R16 + palette dequant
  (`load_tile_prefix` :3251), and new-token columns `[prefix_len, kv_len)` from
  the input `k_packed`/`v_packed`. A query block attending sealed context that is
  *not* part of this call's Q is the explicit purpose of this path.
- **Position-based causal masking, per query row.** `row_max_k(r) =
  min(prefix_len + q_pos[r] + 1, kv_len)` (:2608); each query row already carries
  its own absolute position (`s_q_pos`). RoPE is applied in-kernel from the
  slot's per-slice `rope` base (:4158).
- **Supports F16/F16.** V is already F16; K can be written plain F16 (the
  `k_fmt == T_FORMAT` branch, store :549) — R16 is only the *provenance-capture*
  K variant (128-byte block bundling 32×F16 K + 32×u16 Q). Glue is never
  retrieved, so it needs **no** Q-capture → plain F16 K, smaller and simpler.

In short: the varlen batching, the disjoint per-slot block tables, the
quant-aware prefix attention, the fused per-row writeback, the masking, and the
F16 path **already exist**. The cross-conversation wave is, at the kernel layer,
already supported.

## 3. The one structural gap: actual-position causal masking

The existing kernel bakes in **one** assumption: the causal limit is the packed
index — `row_max_k(r) = prefix_len + q_pos[r] + 1`, with columns `< prefix_len`
read from cache and `≥ prefix_len` from the input `k_packed`. That index-based
limit is correct only because, in a normal prefill, packed index == actual
position. Glue breaks that identity, but **not** the source split.

The clean construction: present each conversation's glue to the kernel as a
**normal prefix-attention call**, with a twist on which tokens are "prefix" vs
"new":

- **Prefix region `[0, S)` = all the conversation's sealed content**, packed
  contiguous (glue gaps removed), read from cache exactly as today via a
  sealed-only `position_map`. **The dequant load path is unchanged.**
- **New region `[S, S+G)` = *all* the glue tokens (every island)**, packed in
  position order, read from `k_packed`. The existing kernel already makes new
  tokens attend each other through `k_packed` ("intra-call tokens see each other
  via the input K/V, not the cache"). So **earlier-island glue is attended for
  free** — no per-column source switch, no RAW hazard.

The *only* thing that breaks is the mask: because sealed and glue **interleave in
real position**, packed column index no longer equals actual position, so
`col_actual_pos[]` is non-monotonic and the scalar `row_max_k` limit is wrong.
Replace it with a per-column **actual-position** comparison:

```
glue row r attends column k_pos  ⟺  col_actual_pos[k_pos] < pos_of_glue[r]
```

where `col_actual_pos[kv_len]` carries every column's true sequence position
(sealed columns' real positions; glue columns' real positions) and
`pos_of_glue[r] = col_actual_pos[prefix_len + r]`. This single change makes glue
row *r* attend exactly the sealed-before-pᵣ **and** earlier-glue it should.

Everything else is unchanged or already per-row: the prefix dequant load, the
smem ring, the MMA, the online softmax. Glue RoPE uses the actual position
`pos_of_glue[r]` (sealed RoPE comes from the slice `rope` base, set to actual
positions host-side). The writeback already targets per-row
`s_write_slice_idx[r]` / `s_write_in_blk[r]` — populated from explicit glue
writer-chunk targets (appended to the slice array) rather than `resolve_pos`,
since the read `position_map` is sealed-only.

## 4. Two batching dimensions — and why they're separable

| Dimension | What it batches | Where it lives | Needs kernel change? |
|---|---|---|---|
| **Across conversations** (the *wave*) | island *r* of every pending conversation, one launch | scheduler restructure | **No** |
| **Across islands** (the *one-shot*) | all islands of a conversation, one launch | kernel: scattered source | **Yes** |

These are independent. The cross-conversation wave is purely a **scheduler**
change (§6) and works with the **existing** kernel, because if we batch island
*r* across conversations *in declaration order*, each conversation's island *r*
is still a contiguous suffix at its slot's current tail (the sealed content after
it hasn't been injected yet). That is the cheaper, lower-risk win and it requires
no kernel work.

The within-conversation one-shot (all of a conversation's islands in one launch)
is what needs the scattered-source kernel. Its incremental value over the wave is
**launch-count**: it turns *R* sequential rounds (R = the deepest conversation's
island count, the rounds are serially dependent) into **1** launch. On WDDM,
where per-launch overhead is the ~74 ms/token floor we measured
([[wddm-forward-floor]]), R≈14 rounds ≈ ~1 s of pure launch overhead per wave —
so the one-shot is a large win *today*. On the production TCC workstation, launch
overhead is small and the wave alone captures most of the benefit (the attention
*compute* is identical either way).

## 5. Decision

> **Build a new kernel *entry* (a new `__global__` + launcher + FFI symbol) that
> reuses the existing device-helper library, rather than (a) bending the existing
> contiguous-prefix kernel in place or (b) writing one from scratch.**

Reasoning:

- **Not from-scratch.** ~90% of the work is the device helpers that are
  format-, layout-, and math-correct and battle-tested:
  `load_tile_prefix` (quant/R16/palette dequant from the arena),
  `store_kv_chunk_arena` (the F16 writeback), the m16n8k16 MMA QK,
  the online-softmax core, in-kernel RoPE, and the `SlotHeader` / `TokenSlice` /
  `position_map` plumbing. A new entry `#include`s these unchanged.
- **Not a runtime branch in the existing kernel.** The contiguous-prefix kernel
  is the hot path for *normal* user/section prefill. Threading a per-column
  source decision through its inner KV-load and masking loops risks regressing
  that path and entangles two control flows in one already-4000-line file.
- **A separate specialization is clean and safe.** The gap-fill control flow —
  scattered `s_q_pos`, per-column source map, the "earlier-glue from input"
  branch — lives in its own kernel that shares the helpers. The common path is
  untouched; the two are independently optimizable. (Mechanically this can be a
  sibling `.cuh` `#include`d by new `_fp16`/`_bf16` TUs, mirroring
  `paged_prefill_api_*.cu`; whether it ends up a distinct `__global__` or a
  `template<bool GAP_FILL>` specialization of the existing one is an
  implementation detail — the binding contract is "compile-time specialized,
  shared helpers, common path untouched.")

The cross-conversation wave (§4, §6) is a **scheduler** change and is largely
independent of this decision; it can land first against the existing kernel.

## 6. Scheduler restructure (the wave)

The kernel layer already supports the wave; the blockers are control-flow:

1. **`apply_segments` is single-sequence and synchronous.** It snapshots /
   truncates one slot and issues `forward_batched(&[parent_id], …)` per island
   (`projection_assembler.rs:437`). It must become a **phase-synchronized
   multi-slot walk**: across all pending conversations, *inject-all-sealed for
   round r* (host-side Arc-clone, cheap), then **one batched `forward_batched`**
   over every conversation's round-*r* glue run, repeat until all exhaust. The
   existing `run_one_prefill_chunk` (`prefill.rs:213`) and
   `run_one_section_ingest_chunk` (`prefill.rs:90`) are the precedent: collect a
   ready set, build `seq_ids`/`inputs`, issue one ragged `forward_batched`.
2. **The two glue entry points must feed one queue.** Initial projection runs
   inline in the `SubmitTurn` arm (`mod.rs:1468`); reprojection runs in the
   serial `drain_pending_reprojections` loop (`decode.rs:559`). Convert both into
   a single **pending-glue set** drained as one batched pass — the existing
   `pending_reprojections: Vec<SequenceId>` (`mod.rs:844`) +
   `pending_section_quantize` queue-then-drain idiom is the model to copy.

With (1)+(2), the existing kernel already gives the cross-conversation wave (R
batched rounds shared across the fleet). Dropping in the §5 gap-fill kernel later
collapses those R rounds to 1.

## 7. The gap-fill kernel — inputs & semantics

Per conversation (one `blockIdx.z` batch element), the host provides:

- `q_packed / k_packed / v_packed`: the conversation's glue tokens — **all
  islands**, packed in position order (F16). These are the **new region**
  `[S, S+G)`; the existing kernel's intra-new attention makes them attend each
  other (so earlier-island glue is attended without a source switch).
- **Sealed-only `SlotHeader` / `position_map`**: maps the packed prefix
  `[0, S)` to the conversation's sealed chunks (glue gaps removed). Each slice's
  `rope` base is set to its first token's **actual** sequence position, so the
  unchanged prefix dequant + RoPE produce correct sealed K. `q_lens[b] = G`,
  `kv_lens[b] = S + G`, `prefix_len = S`.
- `col_actual_pos[kv_len]`: every column's **true** sequence position (sealed
  columns' real positions, then glue columns' real positions). Drives the mask.
- **Glue writer targets**: the pre-allocated glue writer chunks (F16, no R16
  Q-capture), appended to the slice array; per glue row the host supplies the
  write slice index + in-block offset, populating the existing
  `s_write_slice_idx[r]` / `s_write_in_blk[r]` directly (the read `position_map`
  is sealed-only, so the writeback can't go through `resolve_pos`).

Kernel changes, all localized:
1. **Mask** — replace the scalar `row_max_k(r) = prefix_len + q_pos + 1` with the
   per-column `col_actual_pos[k_pos] < col_actual_pos[prefix_len + r]`.
2. **Glue RoPE** — position `col_actual_pos[prefix_len + r]` instead of
   `prefix_len + q_pos`.
3. **Write target** — take `s_write_slice_idx`/`s_write_in_blk` from the supplied
   glue targets instead of `resolve_pos`.

The prefix dequant load, smem ring, MMA, and online softmax are **unchanged**.
Delivered as a `GAP_FILL` compile-time specialization of the kernel template
(common contiguous-prefix path untouched).

### Chunk allocation note
The "continuous set of chunks big enough to hold all the glue" is the glue
writer chunks, allocated as one F16 arena run and exposed as the appended write
slices. Their physical contiguity is independent of their interleaved logical
positions (carried by `col_actual_pos`), so one F16 allocation serves the whole
conversation's glue.

## 8. Correctness — bit-identical to the sequential reference

The gap-fill output is provably identical to today's sequential per-island
prefill, by the standard equivalence *batched causal forward ≡ sequential
forwards*:

- All glue is the new region in `k_packed`; per layer, glue token *g* attends
  (causally, by `col_actual_pos`) the sealed-before-pɢ and earlier glue. Its
  layer-L K/V is projected from its layer-L hidden state, which is the result of
  its layer-(L−1) attention over **the same** left-context the sequential
  reference saw. By induction over layers, every glue token's K/V matches.
- There is **no** intra-launch read-after-write on the arena: earlier glue is
  read from `k_packed` (the input), not the cache it is concurrently being
  written to. Sealed columns are immutable in-cache reads. So no ordering
  constraint between query tiles — exactly the invariant the existing kernel
  already relies on for its own contiguous new tokens.

This is what makes "exact fidelity" tractable: we are not approximating the
attention, only changing *how the same causal set is addressed* (sealed packed
as prefix, glue as new, mask by actual position). The TDD gate (§9) asserts the
glue K/V matches the sequential reference value-for-value.

## 9. Plan & phasing

1. **Wave on the existing kernel (no kernel change).** Restructure
   `apply_segments` into a phase-synchronized multi-slot walk; queue both glue
   entry points into one pending-glue drain. Measure `apply_ms` collapse on the
   control (expect per-conversation glue to fall from N single-seq forwards to N
   shared batched rounds; the reproject's `n_prefill` runs now ride one wave).
2. **Gap-fill kernel (the one-shot).** Add the §5/§7 specialization. Collapse the
   R rounds to 1 launch. Biggest win on WDDM (launch-overhead-bound today).
3. **F16 glue writer format.** Ensure glue writer chunks stamp plain F16 K (no
   R16 Q-capture) — smaller, and correct since glue is never a retrieval target.

## 10. Risks / open questions

- **Per-column source cost.** The gap-fill load must branch sealed-vs-glue per
  column. Glue columns are sparse (single-digit per island) and clustered; a
  run-list keeps the common (sealed) load on the existing fast cp.async path with
  a small glue fixup. Validate it doesn't regress the dominant sealed-load cost.
- **Wave ragged shape.** Conversations have different island counts; the wave
  loop must re-collect the ready set each round (mirroring `prefill.rs:215`).
- **Tail interaction.** Mid-decode reproject snapshots/restores the writer tail
  (`projection_assembler.rs:142,230`). The batched walk must preserve per-slot
  tail snapshot/restore around the shared rounds.
- **F16 K tag.** The writeback dispatches on the K format tag; either stamp glue
  chunks R16-but-skip-Q, or teach the writeback to accept a plain-F16 K tag for
  glue. Prefer the latter (true F16/F16, no wasted Q region).

---

**Bottom line.** The existing kernel already does the varlen cross-conversation
batching, the disjoint per-slot block tables, the quant-aware prefix attention,
the fused per-row F16 writeback, and the masking. The cross-conversation **wave**
is a scheduler restructure that needs **no** kernel change and should land first.
The within-conversation **one-shot** needs exactly one new capability — a
per-column "sealed-from-cache vs earlier-glue-from-input" source — best delivered
as a **new gap-fill kernel entry that reuses the existing device-helper library**,
not a from-scratch kernel and not a runtime branch in the hot contiguous-prefix
path.

---

## 11. Split-KV (deep-slot wall-clock)

The glue kernel's grid is `(slot, kv_head, glue_tile)`, and every block streams
the slot's **entire** column range `[0, kv_len)` (dequant-once *per block*).
With ~26 glue tiles all resident on the SMs at once, the wave's wall-clock per
layer is one full-slot stream — ~4-7 ms/layer at 5-6k columns, ~200 ms of the
reproject glue wave across 48 layers.

Split-KV partitions the column range across blocks instead:

- **Grid** becomes `(slot, kv_head, glue_tile x split)`. Each block streams one
  column window `[split * GLUE_SPLIT_COLS, +GLUE_SPLIT_COLS)` (quantum: 1024
  columns) and emits the un-normalized flash partial `(SwV, m, l)` for each of
  its rows into a per-`(glue row, query head, split)` scratch pool — the same
  grow-on-demand pool and combine kernel the paged-decode split-KV path uses
  (`fused_attn_partial_pool`, `int8_decode_combine_kernel`). The combine merges
  partials by natural-base log-sum-exp (the flash kernels accumulate with
  `fast_exp` e^x) and writes the normalized output.
- **`num_splits = ceil(max_kv / 1024)`**, from the batch's deepest slot;
  `num_splits == 1` (slots up to 1024 columns) keeps the exact single-pass
  direct-write path.
- **The quantum is fixed, not derived from the batch.** A slot's column
  partition depends only on its own `kv_len`: a shallow slot in a deep batch
  runs its real window(s) plus **null** windows (initial flash state, m=-inf,
  l=0) whose combine contribution is exactly zero. So every slot produces
  byte-for-byte what it produces alone — the batched-isolation contract of §8
  extends to split mode, verified by `paged_glue_split_kv_batched_isolation_f16`
  (a solo direct-write shallow slot vs the same slot padded through
  partials + combine). Only past `GLUE_MAX_SPLITS = 16` quanta (>16k-column
  glue slots) does the quantum grow, and the guarantee then holds only among
  batches sharing the grown quantum.
- The glue K/V **scatter phase still runs in every block** (all glue tokens,
  not just the block's tile/window): each block self-produces any glue column
  it may stream, so no cross-block ordering is needed — same-value concurrent
  writes are benign.

Wall-clock effect: the per-layer stream shortens from `kv_len` columns to
`min(kv_len, 1024)` per block, bounded by SM capacity — a 5-6k-column slot's
attention drops ~5x. Correctness gates: `paged_glue_split_kv_matches_reference_f16`
(multi-window vs the position-aware golden, including glue gaps straddling a
window edge) plus the isolation test above.
