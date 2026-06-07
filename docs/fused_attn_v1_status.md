# Fused QKV + Decode-Attention Kernel — Implementation Status

**Branch:** `fused-attn-v1-wip`
**Last update:** 2026-05-03
**Source design:** `docs/new_decode_kernel.md`

## Changelog

- **2026-05-03** — Track A INT8 attention is functionally green. Three root
  causes fixed: per-warp `tile_logits` race, MMA B-fragment lane mapping,
  early return on `rope_interleaved`. A/B harness via
  `CANDLE_FUSED_ATTN_INT8` bitmask added. All variants (sm/shape/WPB/RoPE)
  re-enabled. See "Bug history" and "Iteration ladder" below.
- **2026-05-02** — initial scaffold; iter 0–1 validated, iter 2/3 written
  but unbuilt.

---

## TL;DR

- **Two parallel kernel tracks** are in-tree:
  1. **`int8_decode_kernel.cuh`** — v2-API-compatible (takes pre-projected Q/K/V). **Track A is functionally green**: iter 0–3 written, debugged, and validated against the gated test in all 6 useful flag combinations. Default mode (`CANDLE_FUSED_ATTN_INT8` unset) is full INT8: MMA QK^T (`m16n8k32`) + manual INT8 PV. Iter-1 FP-fmaf baseline kept reachable as a regression-check via mode 0.
  2. **`attn_fused_v1.cuh`** + role files — the design's full fused QKV+attention kernel. Compiles, but `fused_attn_v1_dispatch` still returns `cudaErrorNotSupported` for every shape. Phase 4 attention loop in `consumer_role.cuh` is still a stub (returns zeros), and several other Phase-B TODOs are open. Track B has not regressed since the last entry — it just hasn't been advanced.

- **Rust integration:**
  - `PagedDecode::cuda_fwd` has an env-var hook (`CANDLE_FUSED_ATTN_V1=1`) routing through `fused_attn_v1_v2_compat_dispatch` → `launch_int8_decode_attn`. ✓ Track A passes 17/17 modes here.
  - `candle-transformers/src/models/fused_qkv_attn.rs` is a stub Rust API for the full fused QKV path. **No model has been wired** to use it yet.

---

## Iteration ladder

| Iter | What it does                                             | Status        | Last test result      |
|------|----------------------------------------------------------|---------------|------------------------|
| 0    | Plumbing only — passthrough to v2                        | ✓ done        | 17/17 PASS              |
| 1    | Owned v2 clone — same FP fmaf logic, separate kernel sym | ✓ done        | 17/17 PASS (mode 0)    |
| 2a   | INT8 dot product for QK^T (manual lane-collective)       | ✓ done        | 17/17 PASS (mode 5/7)  |
| 2b   | INT8 MMA m16n8k32 for QK^T                               | ✓ done        | 17/17 PASS (mode 1/3)  |
| 3    | INT8 manual dot for PV                                   | ✓ done        | 17/17 PASS (mode 2/3)  |
| 4a   | Full fused kernel — Phase 2 W_qkv MMA + Phase 3 routing  | partial       | dispatch returns NotSupported |
| 4b   | Model integration — replace `q/k/v_proj × paged_decode`  | not started   | —                      |

`int8_decode_attn_impl` is templated on `<INT8_QK, INT8_PV, INT8_QK_USE_MMA>`.
The dispatcher reads `CANDLE_FUSED_ATTN_INT8` (bitmask, default 3 = full INT8)
once per process and selects one of 6 instantiations:

| Mode | bits | INT8 QK^T | INT8 PV | QK kernel | Use case                         |
|------|------|-----------|---------|-----------|----------------------------------|
| 0    | `000` | no        | no      | FP fmaf   | iter-1 baseline / regression test |
| 1    | `001` | yes       | no      | MMA       | isolate INT8 QK^T                 |
| 2    | `010` | no        | yes     | FP fmaf   | isolate INT8 PV                   |
| 3    | `011` | yes       | yes     | MMA       | **default** — full INT8           |
| 5    | `101` | yes       | no      | manual dot| MMA bug bisect                    |
| 7    | `111` | yes       | yes     | manual dot| MMA bug bisect (full)             |

All 6 modes pass 17/17.

Build cost note: rebuilding the full kernel archive after a header touch is
back to ~5 min on this workstation now that all sm/shape/WPB/RoPE variants are
re-enabled. While iterating earlier I trimmed launchers to a single shape and
the cycle was ~80 s; if a future iteration cycle needs that latency back,
gate the variants in `launch.cu` and `launch_int8_decode_attn`'s dispatch
behind `#if 0` blocks again.

---

## File map (what got changed)

### New files
- `candle-kernels/src/fused-attn-v1/`
  - `arch_traits.cuh` — m16n8k16/k32 MMA shape constants per SM_VERSION
  - `model_descriptor.cuh` — compile-time shape parameterization
  - `smem_arena.cuh` — Phase12/Phase4 union view (W staging tiled to 128 N-dims to fit 48 KB)
  - `cp_async.cuh` — `LoadDescriptor`, named-barrier wrappers, `cp_async_cg_16_raw`
  - `mma_wrappers.cuh` — INT8 `mma.sync` wrappers (m16n8k32, m16n8k16) + fragment loaders. **Register counts corrected** vs design doc (A=4 b32 / B=2 b32 for k32; A=2 / B=1 for k16).
  - `rope.cuh` — dispatch onto v2's `apply_rope_*_f32`
  - `softmax_state.cuh` — online softmax struct + `warp_reduce_max` (not in `simple/warp_reduce.cuh`)
  - `dequant_store.cuh` — per-tile K/V dequant + RoPE + per-palette/per-token re-quant
  - `loader_role.cuh` — descriptor-queue cp.async dispatcher
  - `dequant_role.cuh` — W2/W3 control flow with **W_qkv Q4→INT8 dequant body filled in**
  - `consumer_role.cuh` — W4–W7 control flow with **activation FP→INT8 quant + Phase 2 INT8 MMA loop filled in**; Phase 3 routing sketch present; **Phase 4 attention loop is a stub** (returns zero output)
  - `attn_fused_v1.cuh` — top-level kernel + role dispatch + `build_load_queue` (emits per-K-chunk W_qkv descriptors)
  - `int8_decode_kernel.cuh` — v2-API stepping-stone kernel (the actually-testable track)
  - `v2_compat_dispatch.cuh` — wrapper that calls `launch_int8_decode_attn`
  - `launch.cu` — 9 shape launchers + `extern "C"` dispatch entry points
  - `api.rs` — Rust FFI declarations
- `candle-transformers/src/models/fused_qkv_attn.rs` — Rust API stub for the fused QKV path (currently bails — kernel dispatch returns `NotSupported`)

### Modified files
- `candle-kernels/src/convert/convert_all.cuh` — added `ArenaAccessor::load_head_int8_unscaled<HEAD_DIM, USE_TC>` method (Q4_0/Q8_0/R16 → INT8 with separately-exposed scale, plus FP-source fallback)
- `candle-kernels/src/lib.rs` — exported `fused_attn_v1` module
- `candle-kernels/build_utils.rs` — added `fused_attn_v1` archive group
- `candle-transformers/src/models/prefill_utils.rs` — added `cuda_fwd_via_fused_v1` opt-in path on `PagedDecode` (env-var gated)
- `candle-transformers/src/models/mod.rs` — exposed `fused_qkv_attn` (cuda-only)

---

## What's known to work

- **Foundation builds cleanly** under `cargo build --release -p candle-transformers --features cuda,verbose`.
- **Gated test passes 17/17 modes** with:
  - env var unset (v2 path, baseline)
  - `CANDLE_FUSED_ATTN_V1=1` (default mode 3 — full INT8 MMA QK^T + INT8 PV)
  - all of modes 0/1/2/3/5/7 via `CANDLE_FUSED_ATTN_INT8`
- All sm versions (sm_86, sm_89, sm_120) and shape launchers in `launch.cu` are restored and compile cleanly.

Test command:
```powershell
$env:CANDLE_FUSED_ATTN_V1="1"; cargo test --release --features cuda,verbose --lib --package candle-transformers quantized_llama::tests::test_parallel_batched_forwarding_llama3 -- --ignored --nocapture
```

Performance status: t/s in INT8 mode is currently in the same range as v2
(~50–65 single-context F16 on RTX 4090 Mobile). The kernel is functionally
correct but the design's promised speedup has not been realised yet — see
"Next steps".

---

## How Track A's INT8 attention works (final form)

- **Per-palette Q INT8 quantization** in the kernel preamble. 8-lane xor-1/2/4
  reduction inside each palette group of 8 lanes; broadcast palette p's
  max-abs from lane `p*8` to all lanes; quantize using lane's own palette
  scale; pack 4 INT8 into `q_packed: uint32_t` for MMA shuffles.
- **Per-palette K INT8 quantization** in the `apply_rope_to_tile` epilogue,
  after RoPE + palette-reorder. Lane `p*8` writes the per-palette scale to
  `shared_k_scale[stage][warp_token][p]`. INT8 values written to
  `shared_k_int8[stage][warp_token][lane*VEC..lane*VEC+VEC-1]`.
- **Single per-token V INT8 quantization** (max-abs across full HEAD_DIM,
  full-warp xor-1/2/4/8/16 reduction).
- **QK^T MMA**: per palette `p`, one `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
  - A frag (16×32 INT8 row-major): `a[0] = __shfl_sync(q_packed, p*8 + (lane&3))`
    (palette p, K-cols 0..15); `a[2] = __shfl_sync(q_packed, p*8 + 4 + (lane&3))`
    (palette p, K-cols 16..31). `a[1]/a[3] = 0` (rows 8..15 unused for decode).
  - B frag (8×32 INT8 col-major): `b[0]` = K[token=lane/4][palette*32 + (lane%4)*4..+3];
    `b[1]` same row, +16 K-col offset.
  - C/D output: lane 0..3 hold row 0 cols 0..7. `c_p[0]/c_p[1]` for lane t
    = D[0][t*2], D[0][t*2+1] = palette p contributions to tokens (t*2, t*2+1).
  - Per-palette FP-scale composition: `acc += c_p[0/1] * scale_Q[p] * shared_k_scale[t][p]`.
  - Logits materialised to `tile_logits[stage][warp][token]` (per-warp buffer
    because each warp owns its own Q head — see "Bug history" below).
- **PV INT8 manual dot**: quantize `beta` (the FP softmax weight) per-lane,
  multiply by lane's V INT8 dims, scale to FP via `beta_scale * shared_v_scale`,
  accumulate into `out_reg`.
- **Safety**: invalid tokens get zero INT8 K + V and scale=1.0 in
  `load_tile`'s invalid path so dot products produce safe zeros.

## Bug history (root causes that fell out of this iteration)

1. **`tile_logits` was warp-shared, causing inter-warp races.** Each warp
   computes its own Q head's logits, but `tile_logits[stage][token]` was a
   single buffer across all warps. With 3+ active warps, last-writer wins
   and every warp ended up using the wrong head's scores. Fix: change to
   `tile_logits[stage][warp][token]` so each warp owns its own slot.

2. **MMA B-fragment lane mapping was wrong** for `m16n8k32 .s8.s8` col-major B.
   The kernel was using `lane%8 = N-row, (lane/8)*4 = K-col base`, but the
   actual PTX layout is `lane/4 = N-row, (lane%4)*4 = K-col base`. With the
   wrong mapping the MMA contracted Q dims with K data from unrelated token
   rows. Fix: swap the indices in the B-frag load. Confirmed by per-palette
   `c_p` matching a hand-rolled INT32 reference dot exactly after the fix.

3. **`launch_int8_decode_attn` returned early for `rope_interleaved=true`.**
   The Phase-1 build-time reduction had only the non-interleaved variant
   compiled, but Llama uses interleaved RoPE. The v2-compat dispatch
   silently no-op'd and the output buffer kept whatever uninitialised memory
   contained — every mode failed. Fix: compile both variants.

How the third bug surfaced and what to learn from it: the kernel ran ~2× faster
than v2 in single-context F16 mode despite producing garbage output. That speed
was the diagnostic — when the kernel actually runs, t/s should be in the same
ballpark as v2 (~60 t/s on this workstation for F16). Significantly higher t/s
+ wrong output ≈ the kernel is returning before launching anything.

## `attn_fused_v1.cuh` Iter 4a (full design — still partial)
- **Activation FP→INT8 quant** in consumer_role (per 32-element block, full-warp xor reduction).
- **Phase 2 INT8 MMA loop**: per K-chunk × per N-tile, `mma_int8_m16n8k32` against W_qkv staging.
- **Phase 3 routing** by N-axis: dim < Q_OUTPUT_DIM → Q (per-palette max-abs reduction across warp), Q_OUTPUT_DIM ≤ dim < K_END → write FP32 to `k_new_fp32`, else → `v_new_fp32`.
- **W_qkv Q4_0 → INT8 dequant** in dequant_role: 64 threads (2 warps) cover 128 N-dims per K-chunk; each thread handles 2 N-dims, reads the 18-byte Q4_0 block, unpacks 32 nibbles into INT8 with the FP16 scale.
- **build_load_queue** emits one descriptor per K-chunk pointing at the W_qkv source bytes.

**Phase B TODOs that I left explicit in the code:**
1. **Q-side RoPE** — the C-fragment lane→dim mapping doesn't match v2's RoPE helper's expected layout; requires either a smem round-trip or a new RoPE helper. Currently skipped.
2. **Phase 4 attention loop in consumer_role** — body is stubbed (returns zeros). Should lift the iter-2b/3 logic from int8_decode_kernel into a shared helper and reuse.
3. **Output writeback addressing** — placeholder; needs the lane→(q_head, dim) mapping to match the post-MMA C-fragment.
4. **K_new INT8 re-quant for tile-0 fast path** — design specifies that tile-0 reuses the freshly-projected K_new without going through arena dequant. Not implemented.
5. **Phase 4 KV tile cp.async via descriptor queue** — current dequant_role uses `ArenaAccessor` directly; design's pipelined cp.async path would speed this up.
6. **W_qkv layout contract** with the host — `build_load_queue` assumes K-chunk-major Q4_0 blocks but the actual host quantization layout is undefined.

Because of (1)–(6), the fused dispatch returns `cudaErrorNotSupported` for every shape — `fused_qkv_attn` Rust API will bail rather than invoke the kernel.

### Iter 4b (model integration)
- **Not started.** Hooking the fused kernel into a model (e.g. `quantized_llama` or `llama`) requires:
  1. Concatenating `q_proj`, `k_proj`, `v_proj` weights into a single `w_qkv` tensor at model load time.
  2. Quantizing that tensor to Q4_0 with the kernel's expected layout.
  3. Replacing the `q = q_proj(x); k = k_proj(x); v = v_proj(x); paged_decode_attn(...)` chain with `fused_qkv_attn(x, w_qkv, ...)`.

---

## Next steps

**A. Performance.** The INT8 path is functionally green but t/s is in the
same range as v2. The path forward, in rough order of expected wins:

1. **INT8 PV via real MMA.** The current PV is a manual lane dot per token,
   producing one accumulator scalar at a time. m16n8k32 PV would batch 32
   tokens worth of `beta × V` per palette. Need a different lane mapping
   from QK^T (B is now V[token][dim] not K[token][dim]) and a way to
   batch across 32 contraction tokens — likely by accumulating multiple
   tiles before the MMA call.
2. **Multi-token decode batching.** m16n8k32 has M=16; for single-token
   decode we waste 15/16 of the M dim. Decode for `m_active` queries in
   parallel (where `m_active` ≤ 16) by promoting the GQA group to fill rows
   of A, or batching `m_active` independent slots.
3. **Smarter K-tile pipeline.** Current load_tile uses ArenaAccessor →
   shared_k FP16 → INT8 quant in apply_rope_to_tile. Skipping the FP16
   round-trip when the source is already INT8 (Q4_0/Q8_0) would cut
   bandwidth and smem traffic. Requires per-format paths in load_tile.

**B. Track B (full fused QKV+attn).** Path forward:

1. **Lift iter-2b/3 Phase-4 attention loop into a shared helper** so
   `consumer_role.cuh` can call it instead of returning zeros. The same
   helper validates against the iter-2b/3 baseline trivially.
2. **Q-side RoPE via smem round-trip.** The C-fragment lane→dim layout
   doesn't match v2's RoPE helper; quickest is a one-shot smem scatter
   followed by gather in v2 layout, RoPE, then re-pack for INT8 PV.
3. **Output writeback addressing** — fill in the lane→(q_head, dim) map
   for the post-MMA C-fragment.
4. **K_new INT8 re-quant for tile-0 fast path** — the design specifies
   tile-0 reuses freshly-projected K_new without going through arena dequant.
5. **Phase-4 KV tile cp.async via descriptor queue** — replace the
   ArenaAccessor direct loads in dequant_role with the queue-based
   pipeline in `loader_role.cuh`.
6. **W_qkv host layout contract** — `build_load_queue` assumes
   K-chunk-major Q4_0 blocks; `fused_qkv_attn.rs` needs to enforce this
   when packing q/k/v_proj weights, OR the kernel needs to accept the
   default ggml layout.

After (1)–(6), `fused_attn_v1_dispatch` can return success for at least
the Llama-3.2-3B shape (`h128_q24_kv8_d3072_sm89`). Then iter 4b: model
integration in a `quantized_llama` variant — concatenate q/k/v_proj into
a single `w_qkv` tensor at load time, quantize to Q4_0 in the kernel's
expected layout, replace the per-projection chain with `fused_qkv_attn(...)`.

**C. Hardening.**

- The A/B `CANDLE_FUSED_ATTN_INT8` mode flag is currently a process-wide
  static. Useful for testing but should ultimately be removed once the
  INT8 path is the only one and proven on more models.
- `int8_decode_kernel.cuh` still has the `INT8_QK_USE_MMA=false` path
  compiled (the lane-collective manual dot). It's slower than MMA but
  was crucial for bisecting bug #2 above. Keep it for now as a
  regression bisector; remove once Track B is also passing tests.

---

## Cache invalidation note

Touching any `.cuh` in `candle-kernels/src/fused-attn-v1/` triggers a recompile
of `launch.cu`, which links the entire `fused_attn_v1.a` archive. Touching
`paged-decode/paged_decode_kernel.cuh` (which fused-attn-v1 includes) recompiles
**both** `paged_decode.a` and `fused_attn_v1.a` — about 5 min for the full
archive set on this workstation with all variants enabled, ~80 s if launchers
are trimmed to a single shape. To keep iteration fast:

- Make multiple kernel changes in one batch before rebuilding.
- Add `printf` debugging by changing only `int8_decode_kernel.cuh` (only one
  archive recompiles).
- Avoid editing `paged_decode_kernel.cuh` mid-iteration; if a v2 helper needs
  changing, copy it locally first.
- For tight bug-bisect cycles: gate launchers in `launch.cu` and the
  `launch_int8_decode_attn` dispatch behind a single shape (sm_89 / h128 /
  Llama-3.2-3B / WPB=8 / non-interleaved RoPE + interleaved RoPE for the
  test) and the rebuild collapses to ~80 s. Restore before merging.

---

## Test environment

- RTX 4090 Mobile 16 GB GDDR6 (sm_89 — `CUDA_COMPUTE_CAP=89`)
- Llama-3.2-3B (Nidum / VibeStudio uncensored fine-tune) for `test_parallel_batched_forwarding_llama3`
- `cargo --version` 1.86.x, NVCC 12.x
