# 1C — Warp-stripe batched decode (use all 8 warps)

Status: **design / in progress.** Builds on the committed split-KV + register work
(`0d194313`, `78f62d53`). Goal: activate the idle warps that split-KV exposed.

## The problem (measured)

Post-split-KV + register cut, `ncu` on `perf_b8_ctx2048` (GQA 24:8, hd128):

| metric | value |
|---|---|
| Achieved occupancy | 57% (theoretical 67%, reg+smem-bound at 4 blocks/SM) |
| Compute throughput | 51% |
| Memory throughput | 51% |
| DRAM throughput | 10% (huge bandwidth headroom) |
| Active warps / scheduler | 6.87 |
| **Eligible warps / scheduler** | **1.04** |
| No-eligible cycles | 47% |

The kernel maps **warp = query head**, and `heads_per_group = 24/8 = 3`, so only
**3 of 8 warps compute** (`warp_active = warp < hpg`). The other 5 are *resident*
(they sit at the `__syncthreads` barriers and help the cooperative load) but never
*eligible* — they pad occupancy without hiding latency. So the 57% "occupancy" is
misleading: **effective compute-occupancy is ~25%**. With the kernel latency-bound
(1.04 eligible warps, 47% empty issue slots) and DRAM 90% idle, the lever is to
make all 8 warps compute.

Expected payoff: activating the warps roughly **doubles the compute-eligible
pool** (~25% → ~50% effective), which on a latency-bound kernel with bandwidth to
spare should be **~1.5–2×** on top of today's 2.29× — even after any occupancy
give-back.

## The walls (why this is a rewrite, not a flag)

The current kernel is tightly co-designed; the obvious shortcuts all break:

1. **The QK MMA is hardwired to N=8** — it reads exactly the 8 tokens the 8 warps
   cooperatively load into `shared_k_int8[stage][warp]`. So dropping
   `WARPS_PER_BLOCK` to 4 (to fix the active ratio) would feed the MMA only 4
   valid tokens. No cheap probe.
2. **`hpg=3` divides neither `WARPS=8` nor `HEAD_DIM=128`** — there is no clean way
   to split (head × token × dim) work across 8 warps; every factoring leaves
   awkward remainders.
3. **Occupancy is smem+register-tight at 4 blocks/SM.** A warp-stripe that gives
   each warp its own tiles needs *either* those tiles staged in smem (8× the
   `shared_k/v` footprint → fewer blocks) *or* per-head accumulators in registers
   (→ fewer blocks). Both give back the 64-reg/4-block occupancy we just won.

## Chosen design

**Per-token warp-stripe, manual INT8 dot, per-head accumulators in smem, warp
dimension folded into the existing combine.**

- **warp = tile-stripe.** Each warp walks its own slice of the block's
  `[tile_lo, tile_hi)` KV range, **one token at a time** (not an 8-token tile).
  One token at a time avoids the 8× smem blowup — the warp holds the current
  token's K/V in registers, not a whole tile in smem.
- **Manual lane-collective INT8 dot for QK** (the path that already exists for
  `hd64`, `USE_MMA_QK=false`), looped over the `hpg` heads. We lose the M=1 MMA —
  but it was 1/16-efficient anyway, and we are latency-bound, not MMA-bound.
- **Per-head flash-state in smem** — `out_reg[warp][hpg][VEC]`, `m[warp][hpg]`,
  `l[warp][hpg]` (≈1.7 KB for WARPS=8, hpg=3). Keeps registers near 64 so we hold
  4 blocks/SM; the smem-accumulate latency is trivially hidden once 8 warps are
  eligible and DRAM is 90% idle.
- **Reuse the combine for the warp dimension.** Each warp writes its own partial
  to the global pool, indexed `(slot, head, split, warp)`. The combine already
  merges arbitrary fan-in with base-2 log-sum-exp, so it absorbs the warp axis for
  free — **no new in-kernel cross-warp reduction**. Partial pool grows ×WARPS
  (still tiny, L2-resident); combine reduces over `num_splits * WARPS_PER_BLOCK`.
- **Q for all heads in smem**, loaded + RoPE'd + quantized once, read by every warp.
- The new-token scatter stays idempotent (unchanged from split-KV).

This is effectively a second compute core living beside the existing MMA kernel.

## Implementation plan (milestones; branch stays green + golden at each)

1. **Combine generalization.** Extend the partial pool + `int8_decode_combine_kernel`
   to reduce over `(split × warp)` instead of `split` alone. With the kernel still
   writing only `warp 0`'s partial this is behavior-identical → golden unchanged.
2. **Warp-stripe compute.** Replace the warp=head tile loop with the per-token
   warp-stripe: each warp accumulates per-head flash-state (in smem) over its
   token-stripe, manual INT8 dot for QK. Each warp emits its partial. Golden must
   hold (the merge is exact).
3. **Tune.** Stripe assignment (contiguous vs strided), smem layout / bank
   conflicts, register budget vs `int8_decode_min_blocks`, and the
   split-vs-warp-stripe balance (we may lower `MAX_SPLITS` now that warps absorb
   parallelism). Re-profile `ncu`: target eligible-warps ↑, no-eligible ↓.

## Risks / open questions

- Manual dot may be slower per-op than the MMA; net depends on latency hiding
  winning. Measure at milestone 2.
- Per-token load is less coalesced than the cooperative 8-warp tile load; watch
  L1/memory throughput.
- `hpg > WARPS` (MQA/wide path, `WARPS=16`) keeps the existing kernel — the
  warp-stripe path is for `hpg < WARPS` (the GQA case that wastes warps).
- Validate against the single-decode 4K–32K set too (batch-1 is the most warp-
  starved and the project's core regime).
