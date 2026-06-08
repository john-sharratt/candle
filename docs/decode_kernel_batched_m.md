# Batched-M decode on tensor cores (1C final) — INT8 MMA + read-through V

Status: **design, building.** Combines INT8 (M3b) and batched-M into one kernel —
the tensor-core MMA *is* the INT8 QK, so there's no separate manual-dot step.
Replaces the per-token FP warp-stripe's QK with a per-tile INT8 `m16n8k32` MMA
that packs the GQA group's query heads into M.

## Why one kernel, not two
`batched-M` = INT8 QK on tensor cores. The committed FP stripe and a hypothetical
"M3b manual-INT8 stripe" both quantize K and do an INT8 QK; batched-M does the
same INT8 QK but on the tensor cores at M=hpg instead of `hpg` manual lane-dots.
So the shared work (INT8 K-quant, read-through V) carries over; only the QK
method + the per-token→per-tile structure change. Doing M3b first would rewrite
the QK twice — so we go straight to batched-M.

## Structure
- **Grid unchanged**: `(slot, kv_head, split)`. **warp = tile-stripe** (every warp
  owns its own KV tiles — keeps the all-8-warps occupancy win of the stripe).
- Per tile, the warp runs the **INT8 MMA over its 8 tokens** (N=8) for all `hpg`
  query heads at once (M=hpg), 4 MMAs (one per 32-wide palette) → 4 QK
  instructions/tile vs the FP stripe's ~24 manual dots.
- **PV stays manual** (read-through INT8 V on the CUDA cores): the PV contracts
  over tokens (K=8 < 32), so a tensor-core PV would under-fill K to 1/4.

## Fragment layouts (from mma_wrappers.cuh — pinned)
`mma.sync.m16n8k32.row.col.s32.s8.s8.s32`, A 16×32, B 8×32, C 16×8, all per-thread.

**A (Q), via `load_a_frag_m16k32(a, smem, 32, lane)`** — smem is 16×32 k-major:
```
a[0]=row(lane>>2)   cols (lane&3)*4..+3      a[2]=row(lane>>2)   cols (lane&3)*4+16..+19
a[1]=row(lane>>2)+8 cols (lane&3)*4..+3      a[3]=row(lane>>2)+8 cols (lane&3)*4+16..+19
```
→ stage `shared_qa[N_PALETTE][16][32]` INT8: **row m = query head m** (m<hpg),
rows hpg..15 zero. Loaded once per block (warp 0), RoPE'd + quantized.

**B (K), via `load_b_frag_n8k32(b, smem, 32, lane)`** — smem is 8×32 k-major:
row = token (lane&7), so `shared_kb[warp][N_PALETTE][8][32]` INT8 — the warp's
tile, 8 tokens, RoPE'd + quantized per-token.

**C (scores 16×8 s32), per-thread d[4]**:
```
d[0]=C[lane>>2,       (lane&3)*2]    d[1]=C[lane>>2,       (lane&3)*2+1]
d[2]=C[(lane>>2)+8,   (lane&3)*2]    d[3]=C[(lane>>2)+8,   (lane&3)*2+1]
```
→ **lane L holds head m=L>>2, tokens (L&3)*2 and +1**. Heads 0..hpg-1 are in lane
groups 0..hpg-1 (each group = 4 lanes covering the 8 tokens); d[2]/d[3] are
heads 8..15 (unused for hpg≤8).

## Scale composition (per-palette, can't defer)
`score[m][t] = Σ_p c_p[m][t] · scaleQ[m][p] · scaleK[t][p]`. The per-palette
scales differ, so accumulate the **scaled** contribution each palette (float
`acc_lo`/`acc_hi` per lane for its two tokens), exactly like the M=1 kernel but
with `m = lane>>2` instead of 0. `scaleQ[hpg][N_PALETTE]` staged with Q,
`scaleK[8][N_PALETTE]` staged with each tile's K.

## Score handoff: MMA lanes → all lanes
The MMA leaves head m's 8 scores in only 4 lanes (group m), but the softmax +
PV need them on all 32 lanes (PV spans head_dim across the warp). So after the
MMA each lane writes its `acc_lo/acc_hi` to **`scores_smem[warp][hpg][8]`**
(float), `__syncwarp`, then all lanes read. Tiny (`8·hpg·8` floats).

## Per-tile flow (warp w)
1. stage K tile → `shared_kb[w]` + `scaleK[8][N_PAL]` (load per-palette, RoPE,
   quant — reuse the existing K path).
2. 4 MMAs → per-lane `acc_lo/acc_hi`; write `scores_smem[w]`; `__syncwarp`.
3. **flash softmax** per head over the 8 tokens (read `scores_smem`), update
   per-head `m_i[HPG]`, `l_i[HPG]`, rescale `out_reg[HPG][VEC]`.
4. **PV** per head: read-through INT8 V gathered via `vi`, `out_reg[h]+=β·V`.
5. emit per-(head) partials; combine folds `split×warp` as today.

## State / occupancy
- smem: `shared_qa` 2 KB + `shared_kb` (8 warps × 4 × 8 × 32 = 8 KB) +
  `shared_kv int8 V` + `scores_smem` ≈ **~16–20 KB** (INT8, half the FP fear).
- per-head flash-state `out_reg[HPG][VEC]`, `m/l[HPG]` in registers (HPG
  compile-time, as the stripe). Q in smem.
- Target ≥4 blocks/SM (REG ≤ 64); re-profile after.

## Caveats
- **M=hpg=3 under-fills the tensor core (3/16).** 3× the old M=1 and ~6× fewer QK
  instructions, but the win over the FP stripe at M=3 is measure-dependent (MMA
  latency + smem tile staging vs 24 cheap dots). The decisive win needs
  **speculative/lookahead decode** to fill M→16 — a later lever.
- Per-tile smem staging re-introduces the traffic the per-token stripe avoided.

## Validate
Golden (`compare --golden`) on the GQA stripe scenarios + single-decode 4K–32K;
bench batch-8 all depths + single-decode. Must match the FP stripe's golden and
ideally beat its perf at ctx512+.
