# Paged-Glue Kernel

**Status — implemented + measured.** A dedicated attention kernel for the
reprojection "glue" forward, in `candle-kernels/src/paged-glue/`, wired into the
cross-conversation reproject wave and validated on `reproject_control`. Built as
a derivative of the **paged-decode** kernel, reusing its device helpers.

---

## 1. What it is

The glue forward attends `G` glue tokens (`G ≈ 70–150`) — each over `hpg` query
heads — against a long **quantized** prefix (`S ≈ thousands of tokens`), for
every layer, and writes the glue's own K/V into the writer chunks so later glue
(and later decode) can attend it. That is a **decode** shape (few queries, long
quantized prefix), so the kernel `#include`s and reuses the paged-decode device
helpers: `ArenaAccessor`/`PalIter` (C0–C9 palette dequant), `SlotHeader` /
`position_map` block tables, `resolve_pos`, `write_regs_to_arena` / `_r16`
(K/V writeback), `apply_rope_*`, and `fast_exp::exp2` (online softmax).

## 2. What we expected vs. what we measured — the correction that matters

The original premise for a separate kernel was: the paged-prefill kernel
re-dequantizes the prefix once **per Q-tile** (≈6× for a few-query glue shape),
so a *dequant-once* kernel should be ~6–10× faster. **Measurement refuted this.**

- **Glue-size sweep** (fixed depth, vary `G`): runtime is **flat** in `G` —
  1 glue-tile ≈ 16 glue-tiles (`5.73 → 6.00 ms`). The redundant prefix streams
  across `gridDim.z` glue-tiles **overlap on the SMs**, so redundant dequant is
  nearly free. The prefill's 6× redundancy is therefore **not its bottleneck**.
- **Zero-fill experiment** (replace the HBM dequant with a smem zero-fill,
  keeping the rest of the per-column path): only `1.45 → 1.17 µs/col`. **The
  dequant load is ~20% of the runtime.**

So the "dequant-once" advantage is worth almost nothing, and (corollary) the
prefill is *not* wasting meaningful time on its 6× re-dequant either. The real
cost — and the real lever — is elsewhere.

## 3. The real bottleneck: occupancy

The kernel keeps its flash-state — RoPE'd Q **and** the online-softmax `O`
accumulator for `G_TILE × hpg` rows — in shared memory. At `G_TILE = 8` that is
~64 KB, which on this sm_120 device (dynamic-smem cap **99 KB**, *not* the 227 KB
some Blackwell parts expose) pins the kernel to **1 block/SM**. With no
occupancy, nothing hides the per-column dependency chain (dequant → un-permute →
RoPE → score). The **depth sweep is perfectly linear** in `kv_len` at a fixed
cost/column — the signature of a latency-bound kernel (~250× off the compute
roofline), not a compute- or bandwidth-bound one.

Two fixes, both **bit-exact** (validated against the pure-tensor golden, F16 and
genuine C0 palette-quantized arenas):

1. **Tile-stream the prefix.** All `WARPS` warps each dequant one column of a
   tile in parallel (cp.async), with **one `__syncthreads` per `WARPS`-column
   tile** — versus v0's per-column dequant on warps 0–1 only, with *two* block
   syncs *per column* (≈8900 syncs/layer). → `2.77 → 1.45 µs/col` (1.9×).
2. **Shrink the flash-state for occupancy.** `GLUE_G_TILE = 2` drops the smem
   to ~24 KB → **3–4 blocks/SM**. Lower `G_TILE` trades more `gridDim.z` blocks
   (which overlap cheaply, per §2) for higher occupancy. → `1.45 → 0.65 µs/col`
   (another 2.2×).

**Result: `2.77 → 0.65 µs/col`, ~4.3× kernel.** End-to-end on `reproject_control`
the glue forward is **~642 ms, ~30% faster than the prefill (~900 ms)**; v0 was
*slower* than the prefill (1050–1257 ms). Decode stays coherent.

## 4. Kernel structure (as built)

Grid = `(slot, kv_head, glue_tile)`. Glue tokens are tiled across `gridDim.z`
(`GLUE_G_TILE` tokens per z-tile); the z-tiles run concurrently and their prefix
streams overlap. `block = WARPS_PER_BLOCK × 32` threads. Per block:

1. **Glue K/V writeback.** One warp per glue token scatters its K/V **un-rotated**
   into the writer chunks (re-RoPE'd at read), mirroring the decode scatter; the
   provenance R16 Q region is zeroed (glue is never a retrieval target).
2. **Stage the tile's query rows.** RoPE'd Q for `G_TILE × hpg` rows (each glue
   token rotated at its **true** position via `col_actual_pos`), plus the zeroed
   `m`/`l`/`O` flash-state, in smem.
3. **Stream `[0, kv_len)` in `WARPS`-column tiles.** Warp `w` dequants column
   `c0+w` (`ArenaAccessor::load_head_scaled`, cp.async), un-permutes palette →
   logical order (`PalIter`), and RoPEs K at the column's true position into the
   shared `k_col`/`v_col` tile buffers. One sync per tile.
4. **Score.** One warp per row; lanes own `VEC` head dims; a manual dot +
   warp-shuffle reduce per `(row, column)` (the dot is negligible vs. the
   streaming — §2 — so no MMA), then an online-softmax update of the smem `O`
   accumulator. The causal mask is by **actual** position (`col_actual_pos[col]
   > col_actual_pos[row]` → skip), so it generalizes to scattered glue islands;
   today glue is the contiguous tail, so this reduces to ordinary causal.
5. **Normalize + write** `O / l` to the output.

There is no split-KV/combine: the single per-block prefix stream plus the
`gridDim.z` glue-tiles already saturate the SMs for the reproject's `kv_head`
count.

## 5. Host wiring

`fire_gap_fill_batch` (the reproject wave) stages each slot's flat
`col_actual_pos` (sealed prefix ++ glue, true positions) on the session via
`set_pending_glue`, then runs the gap-fill `forward_batched`. That forward takes
the pending columns, validates lengths, and builds a `GlueMeta` attached to the
`BatchedPrefillMeta`. The layer's `forward_attn_batched_multi` sees the glue meta
and, for **HD128 + chunked caches**, routes to `paged_glue_attn` (which builds
the per-slot `SlotHeader`s via the shared `build_slot_headers`, the glue write
targets, and `cu_kvlens`, then launches the kernel through the `PagedGlueChunks`
op). Other head dims fall through to plain `paged_prefill_batched`.

## 6. Forward B-head window

By default glue rows attend **backward only** — the sealed prefix (A) plus
earlier glue, masked causally by true position. The optional **forward window**
additionally opens the first `fwd_b` columns of the *next* section (B), so glue
is generated as a bridge **into** B rather than a blind continuation of A.

- **Kernel** (`fwd_window` / `b_avail`): `fwd_b = min(fwd_window, b_avail[slot])`
  extends the column stream to `[0, kv_len + fwd_b)`. Columns `[kv_len,
  kv_len+fwd_b)` are B; the causal test is skipped for them (always visible),
  while glue↔glue causality is unchanged. B is **read-only** — the scatter still
  writes only `t < g_total`. `fwd_b == 0` is bit-identical to the backward-only
  kernel (same column set, same mask).
- **Asymmetric by design.** Backward is unbounded (A is true, fixed context —
  free and consistent to read); forward is capped at B's leading keys, which are
  stale (written against B's original, now-deleted neighbourhood) and only
  meaningful as far as B's heads reach back (~16–32 tokens).
- **Layout assumption.** B's leading columns must already be resident in the
  slot's `position_map` at `[kv_len, kv_len+fwd_b)`, with `col_actual_pos`
  carrying their assembled (downstream-of-glue) positions — then the existing
  dequant-once + per-column RoPE path handles them with no special case. If B
  lives in a separate slot/allocation it is **not** in this position_map and the
  window must stay closed (`b_avail == 0`); staging it there is a separate path.
- **Status.** Threaded end-to-end (kernel → FFI → `PagedGlueChunks` →
  `paged_glue_attn`). The production reproject path passes `fwd_window == 0` (B is
  not yet staged downstream of the glue in the slot's position_map), so the window
  is **landed dark**; the kernel gate `paged_glue_forward_window_f16` proves
  correctness by staging B into the slot's position_map and matching a glue→B-head
  cross-attention golden, while requiring the output to differ from the
  backward-only run.

## 7. Reuse from paged-decode

| Reused from paged-decode | Role in glue |
|---|---|
| `ArenaAccessor::load_head_scaled`, `PalIter` | one-time C0–C9 / palette dequant + un-permute |
| `SlotHeader` / `get_slice` / `get_head` / `resolve_pos` / `position_map` | per-slot block tables |
| `write_regs_to_arena` / `write_regs_to_r16` | glue K/V writeback |
| `apply_rope_rotary_f32` / `apply_rope_interleaved_f32` | glue Q + prefix K rotation at true position |
| `fast_exp::exp2<Softmax>` | online-softmax flash-state update |

New code is the tile-streamed dequant loop, the smem flash-state, the
`G_TILE × hpg` row scoring, the per-glue-row writeback targets, and the
launcher/FFI/host wiring (`PagedGlueChunks`, `paged_glue_attn`).

## 8. Remaining levers (diminishing)

- **Register-resident `O`.** The prefill already keeps its output accumulator in
  per-thread MMA-fragment registers (`smem table: "O accumulators | registers |
  ~0KB"`); the glue still parks `O` in smem. Moving it to registers would free
  smem (more occupancy) and drop the per-column `O` read-modify-write — an
  estimated further ~1.3–1.5×. This is the glue *catching up to the prefill's
  design*, so the optimization is glue-specific and does not port back to the
  prefill.
- **Diminishing end-to-end.** The reproject is now dominated by **tier
  migration** — `swap_ms + elevate_ms ≈ 6.5 s` — versus the glue forward's
  ~0.64 s. Further glue-kernel work buys little reproject wall-clock; the
  warm/cold-tier path is where the seconds are.
