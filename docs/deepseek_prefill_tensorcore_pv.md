# The latent prefill kernel (HEAD_DIM=512): tensor-core PV, 2-block occupancy, pre-quantized V corpus

Design of `candle-kernels/src/paged-latent/latent_prefill_kernel.cuh` — the paged
latent-attention prefill for the DeepSeek single-latent geometry (HEAD_DIM=512, K≡V,
MQA `n_kv_head=1`, 64 query heads, nope‖rope 448+64). Measured on RTX PRO 5000
Blackwell (sm_120, 100 KB smem/SM, 65536 regs/SM): **~65 ms / 4096 queries @ 200K
depth, topk 512** (harness `latent_prefill_bench`, gate `max_rel_err = 0.0088`)
— 2.7× the scalar-PV flash baseline this design replaced (178 ms). StoryRewrite
n=1/4/8 all 100%.

## Block architecture: 512 threads, two head-passes, 2 blocks/SM

One block owns one query (`grid = total_q × splits`): 16 warps (512 threads),
`__launch_bounds__(512, 2)` → **two blocks resident per SM**, which halves the
barrier stall (measured 7.2 → 3.5 cyc/inst) — the dominant stall of the
one-block layout.

The 64 heads × 512 dims of output are 32,768 accumulator floats; at 512 threads
that is the whole 64-register budget, so the block covers the heads in **two
sequential head-passes of 32**: the PV accumulator (`o_acc[8][4]` = 32 f32/lane,
warp = 2 row-tiles × 8 dim-groups) is emitted to the split-KV partials between
passes and reused. The tile loop runs once per pass; `sK`/`sVt` are rebuilt per
pass from L2-resident int8 (cheap — the kernel is smem/barrier-bound, not
global-bound).

Register economy that makes 2 blocks fit:
- **No `sQ` smem.** Q is invariant across key tiles, so each pass builds its QK
  A-fragment (`qa_frag[4][4]`, the m16n8k32 layout hand-packed byte-for-byte)
  **once, straight from L2**, and reuses it for every tile. `scaleQ` is
  recomputed per pass (32 heads wide, not 64).
- smem ≈ 50 KB/block: sK[32][528] + sVt[512][48] + s_p8[32][32] +
  scores[32][33] + scales/alpha/valid/vscale.

## Per-tile pipeline (32-key tiles)

1. **Stage** — warp w loads keys {w, w+16} (single-buffer register pipeline:
   the second key's global load overlaps the first's RoPE+quant); per-band int8
   → `sK[key][dim]` (one 16-byte store per lane) + `scaleK[key][band]`.
   Window keys read the arena **format-dispatched per band** (see below);
   fresh keys FP8-round-trip; comp keys read `comp_i8` (pre-roped, identity).
2. **sVt (PV B-operand)** — comp tiles **gather pre-quantized bytes from
   `comp_v8`** (no per-tile max/requant, rows written as two 16-byte stores,
   one fewer barrier); window/fresh tiles requantize from `sK` with a
   per-dim-per-tile scale.
3. **QK** — 8 warps (head-group × band), m16n8k32 with the pass's `qa_frag`,
   band partials atomicAdd-collapsed into `scores[32][33]` — the row stride is
   **33 words** so rows stagger across all 32 smem banks (at 32 the rows alias
   bank 0 and the atomics serialize on 8 banks: that aliasing was the top mio
   stall of every earlier variant; the +1 pad was worth −3.7 ms).
4. **Softmax** — warp owns 2 pass-heads; online m/l; P×127 → `s_p8`; alpha to
   smem.
5. **PV** — warp = (row_tile, dim_group); ldmatrix A from `s_p8`, B from `sVt`
   (software-pipelined); `o_acc = o_acc·alpha + D·vs`.

## The V corpus is pre-quantized per prefill (per-dim-global scale)

The corpus pre-pass (`latent_rope_quant_corpus_kernel`, grid-stride) ropes and
per-band int8-quantizes the attended entries once (`comp_i8`/`comp_scale`) and
folds **global per-dim |v| maxima** (`comp_vmax[512]`, register-accumulated,
one atomicMax per thread). A second pass (`latent_quant_v_corpus_kernel`)
emits `comp_v8[G,512]`: the PV operand quantized against `comp_vmax` — the
same value the attention kernel would compute per tile, with the scale pooled
over keys at full per-dim granularity. Measured **precision-neutral** (gate
exactly 0.0088): dims are the sensitive axis, keys are not (pooling dims —
per-band scales — fails the gate at 0.10). The PV epilogue scale for comp
tiles is therefore a kernel constant, loaded into `s_vscale` at the first comp
tile of each split. Canonical `comp` stays f32/pre-RoPE/position-free (§C of
the batched-attention plan); all of this is throwaway per-prefill scratch.

## Adaptive per-band window formats

The window walk dispatches on the KvHead's per-band format tag
(`load_band_elem` in `latent_common.cuh`, shared with the decode kernel):
F8E4M3 rows (the writer chunk's format — a hoisted fast loop), other float
dtypes, and all arena quant formats via `dequant_element_inline`. Quant bands
are token-oriented GGML blocks — with CHUNK=32, block `d` holds dim `d`'s 32
tokens (`base + d·BLOCK_BYTES`, element `within & 31`) — composed with the
per-band outer scale. Each lane's 16 dims sit in one band, so {ptr, fmt,
outer} resolve once per lane. The write paths (fused/glue scatter) stay FP8:
sealed chunks gain quant tags only through the compression policy. Gated
bit-exact by `mirror_bit_exact_mixed_band_formats` (decode) and within the
int8-PV envelope by `prefill_mixed_rows_equal_decode_steps` (prefill).

## Measured design boundaries (do not relitigate without new hardware)

- **1-block/1024-thread vs 2-block/512-thread:** at 1024 threads the 64-reg cap
  forces ~300 B of spill; eliminating the spill entirely (512 threads, 128
  regs, 1 block) is *slower* (69 ms) — the spill was not the bottleneck. The
  2-block layout wins only once its smem actually fits twice (a ~1 KB
  driver-static overhead silently demotes to 1 block — verify occupancy with
  ncu, not launch_bounds).
- **Remaining stalls at ~65 ms:** same-address atomic serialization of the QK
  band collapse (~90 M conflict phases; non-atomic alternatives need +4–13 KB
  smem against ~200 B headroom) and the 64-reg spill (~340 B) that
  `launch_bounds(512,2)` reimposes. Both trace to the accumulator density —
  32,768 floats over 512 threads.
- `ldmatrix.m16n16.trans.b8` (byte-transposing loads) does **not** exist on
  sm_120 (ptxas rejects; sm_100a only).

## Correctness contract

int8 P·V is a different PV than the decode kernel's scalar path, so
prefill↔decode gates are tolerance-based (`d < 0.03·scale`); the harness
float-reference gate (0.05, measured 0.0088), the mirror bit-exact suite
(including mixed band formats), `prefill_chunked`, and wave_paris /
StoryRewrite argmax are the guards.
