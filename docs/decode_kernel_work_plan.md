# Decode / Fused-Attention Kernel — Work Plan

Status as of this branch (`decode-kernel-rewrite`), CUDA 12.9, sm_89 + sm_120.
This is the authoritative TODO for finishing the fused-attn-v1 kernel. It covers
**Track A** (the INT8 decode-attention kernel — correct, needs to be *fast*) and
**Track B** (the fully-fused QKV + attention kernel — the actual end goal, still
mostly scaffolding). Read alongside `docs/fused_attn_v1_status.md` (the original
design/status) and `docs/new_decode_kernel.md` (the design).

---

## 0. Current state (what is true right now)

### Works and is validated
- **Track A INT8 decode kernel** (`candle-kernels/src/fused-attn-v1/int8_decode_kernel.cuh`)
  is **functionally correct** across head dims and arena formats, golden-gated.
  - hd128: INT8 QK^T MMA path, validated vs FP32 ground truth (cosine ≈ 1.0 modulo quant).
  - hd64 / hd96 (`SUB_HEAD_DIM != 32`): **manual INT8 dot** path (see fix below).
  - hd256: **gated out** of the fused dispatch (falls back to V2).
- Reachable two ways:
  - Env: `CANDLE_FUSED_ATTN_V1=1`.
  - Programmatic: `DecodeBackend::FusedV1` via `paged_decode_attn_with_backend`
    (`candle-transformers/src/models/prefill_utils.rs`).
- Builds for sm_89 + sm_120 under **nvcc 12.9** (12.4 fails on `compute_120`; see `INSTALL.md`).

### The hd64 fix that landed this session
- **Root cause:** the INT8 QK^T `mma.m16n8k32` and its fragment assembly are hardwired
  for `SUB_HEAD_DIM == 32` (HEAD_DIM==128, VEC==4): `b_frag[1]` reads `+16` (cols 16–31 of
  a palette) and `q_packed` reads `q_int8[0..3]`. At hd64 (`SUB_HEAD_DIM=16`, VEC=2) the
  MMA straddled two palettes and read past `q_int8[VEC]` → structurally wrong logits
  (golden cosine 0.706).
- **Fix:** `constexpr bool USE_MMA_QK = INT8_QK_USE_MMA && (SUB_HEAD_DIM == 32);` — use the
  MMA only when a palette is exactly 32 dims, else the **manual per-lane INT8 dot**
  (already in-tree, correct for any VEC). Guarded `q_packed` so the OOB read can't happen.
- **Implication for hardening:** the manual-dot path is now a **production** path for
  `SUB_HEAD_DIM != 32`, *not* just a bisect tool. The status doc's "remove the
  `INT8_QK_USE_MMA=false` path" hardening item is **void** — keep it.

### Performance baseline (RTX 4090 Mobile, batch-8, CUDA-event GPU time)
Fused vs V2, **geomean ≈ 1.22–1.25×**, eroding to ~1.0× at deep context:

| ctx | f16 | real Q8_0 | real Q4_0 | real Q2_0 |
|---|---|---|---|---|
| 512 | 1.27× | 1.29× | 1.28× | 1.22× |
| 2048 | 1.17× | 1.08× | 1.45×¹ | 1.11× |

¹ Q4_0 ctx2048 "1.45×" is V2 getting *slower* reading Q4, not fused getting faster.

**The smoking gun:** the fused kernel **dequant→requants every arena to FP then back to
INT8**, so an int8 arena (Q8_0/Q4_0) buys it nothing today (306 µs at ctx512 for f16 ==
Q8_0 == Q4_0). Target is **≥ 2×**.

### Tooling built this session — `candle-examples/examples/decode_ab/`
A standalone A/B + correctness + perf harness for V2 vs fused. **Use it to validate every
step below.** Requires CUDA; run with the v12.9 toolchain on PATH.
- `compare` — A/B parity (V2 vs fused) across a scenario × format matrix.
- `compare --golden` — **ground-truth gate**: both kernels vs an FP32 reference (identity
  RoPE), pass/FAIL on **cosine ≥ 0.93** (precision-robust; catches structural bugs, passes
  quant loss down to Q2). This is the correctness gate for the rework.
- `bench` — **CUDA-event GPU kernel time** + tokens/s + speedup, **build-once** (fast),
  defaults to the **batch-8** perf scenarios (`perf_b8_ctx{128,512,1024,2048}`).
- `xcheck` — separates INT8-compute delta from quantized-read delta (V2 vs fused localizer).
- Real quant: `rq-uni-q8_0-L0` / `rq-uni-q4_0-L0` / `rq-uni-q2_0-L0` (uniform int8 arenas,
  unity palette) and `rq-adaptive-L{0..7}` (non-unity palette). Plain `q8_0` etc. are
  actually R16 capture format, **not** real int8 — use the `rq-*` formats for int8 work.

Standard loop while reworking the kernel:
```bash
export CUDA_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
export PATH="$CUDA_PATH/bin:$PATH"; export CUDARC_CUDA_VERSION=12090
cargo build --release --features cuda --example decode_ab
# correctness:
./target/release/examples/decode_ab.exe compare --golden \
  --formats f16,rq-uni-q8_0-L0,rq-uni-q4_0-L0,rq-adaptive-L5
# perf:
./target/release/examples/decode_ab.exe bench \
  --formats f16,rq-uni-q8_0-L0,rq-uni-q4_0-L0,rq-uni-q2_0-L0 --iters 200 --warmup 50
```

---

## 1. Track A — make the INT8 decode kernel fast (to ≥ 2×)

These three are **one coupled rework** of `int8_decode_kernel.cuh` (+ the shared convert
header), not three independent edits. The per-dim scale plumbing, the tile-batched
softmax, and the batched-M MMA all interlock through how the per-(dim,block) int8 scale is
applied to the contraction output.

### 1A. Per-dim-scale skip-dequant (zero-cost int8 read-through)
**Goal:** stop dequant→requanting. When the arena is native int8 (Q8_0/Q4_0/Q2_0), read
the int8 straight into the MMA operand and carry the scale, instead of
`load_head_scaled` (→FP) followed by re-quantization in `apply_rope_to_tile`.

**The blocker (precise):** the existing `ArenaAccessor::load_head_int8_unscaled`
(`candle-kernels/src/convert/convert_all.cuh`) **collapses the per-dim block scales to a
single `out_scale`** (it only writes `out_scale` from `lane==0 && i==0`). But Q8_0/Q4_0
store a scale **per (dim, 32-token block)**; the int8 values are meaningless without their
per-dim scale. That single-scale return is the "coarse fallback" the comment warns about —
it is **not** correct for non-uniform dims and must not be used as-is.

**Tasks:**
- [ ] Extend `load_head_int8_unscaled` to emit a **per-dim scale vector** (one scale per
      `dim` this lane owns, into a register array / smem), not one scalar. Keep the
      existing single-scalar overload only if something else relies on it (check call sites).
- [ ] Apply the int8 scale **on the contraction output**, per output dim: for V,
      `out[dim] = scale[dim][chunk] · Σ_token (beta_q[token] · v_int8[token][dim])`. The
      per-dim scale factors out of the int8 contraction cleanly — this is why it aligns
      with the tile-batched structure (1B): one Q8_0 token-oriented block == one 32-token
      chunk == one tile, so `scale[dim]` is constant within a tile.
- [ ] **K caveat:** K still needs RoPE in FP, so K **cannot** fully skip the FP round-trip.
      Skip-dequant primarily benefits **V** (no RoPE) and any K path that doesn't RoPE.
      Decide whether K reads int8 for the *non-RoPE* component only, or stays FP→int8.
- [ ] Add per-format paths in `load_tile` so int8-native formats take the direct route and
      FP formats (f16/bf16/R16) keep the dequant path. Make it a **compile-time** dispatch
      on the format tag where possible (zero runtime branch in the hot loop) — this is the
      "zero-cost abstraction" requirement.
- [ ] Validate: `compare --golden` must stay green on `rq-uni-q8_0/q4_0/q2_0` **and**
      `rq-adaptive-L{0,5,7}` (non-unity palette, per-dim scales actually vary).

**Expected win:** memory-bound deep-context cases (ctx ≥ 1024) — Q4_0 reads 4× less KV
traffic; the round-trip elimination is what converts that into real speedup.

### 1B. Tile-batched (FlashAttention-style) softmax
**Goal:** enable a token-batched PV. Today `process_tile` does a **per-token** online
softmax rescale (`out_reg = fmaf(out_reg, alpha, …)` per token, line ~606). An m16n8k32 PV
contracts 32 tokens at once, which is impossible with a per-token rescale.

**Tasks:**
- [ ] Restructure the tile loop to: (i) first pass computes all tile scores + the tile
      max, updates the running max `m_i` → `m_new`; (ii) **one** `alpha = exp(m_i - m_new)`
      rescale of `out_reg`/`l_i` per tile; (iii) compute `beta_token = exp(score - m_new)`
      for the tile; (iv) batched PV over the tile; (v) `l_i += Σ beta`.
- [ ] Tile size should align to the 32-token chunk (the Q8_0 block / one CHUNK_SIZE) so the
      per-dim scale (1A) is constant within a tile. Note `WARPS_PER_BLOCK` is currently 8 —
      decide whether to process 32-token tiles (multiple warps' tokens) or keep 8 and do
      4-token DP4A sub-batches.
- [ ] Validate with `compare --golden` after this step *before* touching the MMA — this
      change alone must be numerically identical (it's just reassociating the online softmax).

### 1C. Batched-M MMA for QK^T **and** PV (the actual throughput win)
**Goal:** fill the MMA `M=16` rows. Today one warp = one Q head ⇒ **M=1**, 15/16 wasted —
so even the existing QK MMA is underutilized, and a PV MMA added naively is break-even.

**Tasks:**
- [ ] Change the warp/CTA mapping so the **M dimension is filled** by multiple query rows:
      pack the **GQA-group heads** (`n_q_head / n_kv_head`, e.g. 3 for Llama-3.2-3B) and, if
      needed to reach 16, **batched slots** (`m_active ≤ 16`) into the A-fragment rows. This
      is the grid/CTA/warp-mapping restructure — the biggest single change.
- [ ] **QK^T MMA:** A = `m_active` query rows × 32 K-cols (per palette), B = K[token][dim].
      Reuse the working hd128 fragment assembly, but A now holds `m_active` rows.
- [ ] **PV MMA:** A = `m_active` query rows of `beta` × 32 contraction tokens, B =
      V[token][dim] (col-major, 8 N-dims/MMA, 16 MMAs to cover head_dim=128). Note the lane
      mapping differs from QK^T (B is V[token][dim], not K[token][dim]). Apply the per-dim
      int8 V scale (1A) on the C-fragment output.
- [ ] Keep the `SUB_HEAD_DIM == 32` guard (`USE_MMA_QK`) — hd64/96 stay on the manual dot
      regardless; only hd128 gets the batched-M MMA.
- [ ] Output writeback: the post-MMA C-fragment lane→(query, dim) mapping changes; fix the
      store accordingly.
- [ ] Validate: `compare --golden` green at hd128 for all formats; `bench` shows the move
      toward 2× (watch the batch-8 perf scenarios, both shallow and deep ctx).

**Expected win:** compute-bound shallow-context cases (ctx ≤ 512), plus the multiplier from
1A on deep context.

### 1D. Smaller perf follow-ups (after 1A–1C)
- [ ] Decode-batch `m_active` independent slots when the GQA group alone doesn't fill M=16.
- [ ] Re-bench `q2_0` specifically (2-bit was 1.0–1.2× — check it isn't a regression).
- [ ] Consider DP4A as a fallback contraction for the non-MMA (hd64/96) path if those ever
      need to be fast (currently they're correctness-only, unused by target models).

---

## 2. Track B — the fully-fused QKV + attention kernel (the end goal)

`candle-kernels/src/fused-attn-v1/attn_fused_v1.cuh` + role files
(`loader_role.cuh`, `dequant_role.cuh`, `consumer_role.cuh`, `smem_arena.cuh`,
`cp_async.cuh`, `mma_wrappers.cuh`, `model_descriptor.cuh`, `rope.cuh`,
`softmax_state.cuh`, `arch_traits.cuh`). The goal: **one kernel** that does the Q/K/V
projection (W_qkv · activations) *and* the attention, eliminating the separate
projection GEMMs and the K/V write+reread.

### Current state (Iter 4a — partial)
What's implemented in `consumer_role` / `dequant_role`:
- Activation FP→INT8 quant (per 32-element block, full-warp xor reduction).
- Phase 2 INT8 MMA loop: per K-chunk × per N-tile, `mma_int8_m16n8k32` against W_qkv staging.
- Phase 3 routing by N-axis: `dim < Q_OUTPUT_DIM` → Q (per-palette max-abs reduction);
  `Q_OUTPUT_DIM ≤ dim < K_END` → write FP32 to `k_new_fp32`; else → `v_new_fp32`.
- W_qkv Q4_0 → INT8 dequant in `dequant_role` (64 threads cover 128 N-dims/K-chunk;
  each thread handles 2 N-dims, reads the 18-byte Q4_0 block, unpacks 32 nibbles + FP16 scale).
- `build_load_queue` emits one descriptor per K-chunk pointing at the W_qkv source bytes.

**Blocking gaps** — `fused_attn_v1_dispatch` returns `cudaErrorNotSupported` for **every**
shape, and `fused_qkv_attn.rs` bails, because of the six items below.

### Phase-B TODOs (all required before the dispatch can return success)
1. [ ] **Phase-4 attention loop in `consumer_role.cuh`** — currently a **stub that emits
       zeros** ("Phase B placeholder: use OnlineSoftmaxState scaffold and emit zeros").
       Lift the now-correct iter-2b/3 attention (INT8 MMA QK^T + softmax + INT8 PV) from
       `int8_decode_kernel.cuh` into a **shared helper** and call it here. (This depends on
       Track A 1B/1C — do Track A first so the helper is the fast, correct version.)
2. [ ] **Q-side RoPE** — the C-fragment lane→dim layout (post W_qkv MMA) doesn't match v2's
       RoPE helper's expected layout. Quickest: one-shot **smem scatter → gather in v2
       layout → RoPE → re-pack** for INT8 PV. Or write a C-fragment-native RoPE helper.
       Currently **skipped**.
3. [ ] **Output writeback addressing** — placeholder; fill in the lane→(q_head, dim) map
       that matches the post-MMA C-fragment.
4. [ ] **K_new INT8 re-quant for the tile-0 fast path** — design: tile-0 reuses the
       freshly-projected K_new without going through arena dequant. Not implemented.
5. [ ] **Phase-4 KV tile `cp.async` via the descriptor queue** — `dequant_role` currently
       uses `ArenaAccessor` directly; switch to the queue-based pipeline in
       `loader_role.cuh` (the `LoadDescriptor` / named-barrier machinery already exists).
6. [ ] **W_qkv host layout contract** — `build_load_queue` assumes **K-chunk-major Q4_0
       blocks**, but the host quantization layout is currently undefined. Either
       `fused_qkv_attn.rs` enforces this layout when packing `q/k/v_proj`, **or** the kernel
       accepts the default ggml layout. Pin this down on both sides.

After (1)–(6), `fused_attn_v1_dispatch` should return success for at least the
**Llama-3.2-3B shape** (`h128_q24_kv8_d3072_sm89`). `model_descriptor.cuh` `static_assert`s
`HEAD_DIM==128`, `N_PALETTE==4` — Track B is hd128-only by design (fine; all target models).

### Iter 4b — model integration (not started)
- [ ] At model load, **concatenate** `q_proj` + `k_proj` + `v_proj` into a single `w_qkv`
      tensor.
- [ ] **Quantize** `w_qkv` to Q4_0 in the kernel's expected (K-chunk-major) layout — must
      match item (6).
- [ ] Replace `q = q_proj(x); k = k_proj(x); v = v_proj(x); paged_decode_attn(...)` with
      `fused_qkv_attn(x, w_qkv, …)` in a `quantized_llama` variant.
- [ ] `candle-transformers/src/models/fused_qkv_attn.rs` is the Rust entry (currently bails
      with `NotSupported`); wire it to the real dispatch once the kernel works.

### Track B validation
- The harness today only drives the **decode-attention** path (`paged_decode_attn`). To
  validate Track B (fused QKV+attn) it needs an extra mode that feeds activations + a
  quantized `w_qkv` and compares the fused projection+attention against
  `q_proj/k_proj/v_proj` + `paged_decode_attn` (and/or the FP32 golden extended to include
  the projection). **Add this when Track B is ready to test.**

---

## 3. Loose ends / known issues (carry forward)

- [ ] **hd256 fused decode** — gated out (`int8_decode_kernel` static smem 78 KB > 48 KB
      ptxas cap at hd256). Falls back to V2. Unused head dim; fix only if hd256 is ever needed
      (would require dynamic smem like the prefill fix).
- [ ] **hd256 V2 prefill residual** — the dynamic-smem prefill fix (your push) still
      OOB-writes `smem_q` at `paged-prefill/paged_prefill_kernel.cuh:2763` for hd256 (the
      triple-buffer variant under-sizes the smem_q dynamic region). Your code; unused head dim.
- [ ] **V2 hd64 R16 read** — small imprecision (golden cosine 0.986 vs 1.0 for f16). Passes
      the 0.93 gate; V2-side, hd64, not production. Investigate only if hd64 matters.
- [ ] **Hardening (revised):** the `CANDLE_FUSED_ATTN_INT8` process-wide mode flag is a
      test knob — remove once the INT8 path is the only one and proven on more models.
      **Do NOT** remove the `INT8_QK_USE_MMA=false` manual-dot path — it's now the
      production path for `SUB_HEAD_DIM != 32` (hd64/96).

---

## 4. Build / iteration notes

- **Toolchain:** nvcc **12.9** (sm_89 + sm_120). 12.4 fails `compute_120`; 13.x breaks
  cudarc/CCCL. See `INSTALL.md` (updated this branch). This session's shells inline the env
  (`CUDA_PATH`/`PATH`/`CUDARC_CUDA_VERSION=12090`) because the running process captured the
  old environment; new sessions pick up the Machine PATH automatically.
- **Rebuild cost:** touching any `.cuh` in `fused-attn-v1/` recompiles `launch.cu` and links
  the whole `fused_attn_v1.a` (~5 min). Touching `paged-decode/paged_decode_kernel.cuh`
  (included by fused-attn-v1) recompiles **both** archives. Batch kernel edits before
  rebuilding; the Rust-only example recompiles in ~1 min.
- **Validate every kernel step:** `compare --golden` (correctness) then `bench` (perf).
  Never advance a perf change that drops a golden cell below 0.93.

---

## 5. Suggested sequencing

1. **Commit the current branch** (hd64 fix, harness+golden+bench, INSTALL.md, fused graft).
2. **Track A 1B** (tile-batched softmax) — numerically identical, low risk, unblocks PV MMA.
3. **Track A 1A** (per-dim-scale skip-dequant) — memory-bound win, validate on `rq-*`.
4. **Track A 1C** (batched-M MMA QK+PV) — the throughput win; drive with `bench` to ≥ 2×.
5. **Track B 1** (lift the now-fast attention helper into `consumer_role`) + 2–6.
6. **Track B iter 4b** (model integration) + extend the harness to validate fused QKV+attn.
