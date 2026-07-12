# Q8 Quantized Matmul & Activation Pipeline

## Abstract

The inference inner loop is a chain of matmuls separated by floating-point "islands"
(RMSNorm, SwiGLU, softmax, the residual stream). This document specifies a
**tensor-core INT8 matmul path** in which the activations crossing every matmul are a
first-class 8-bit interchange format and the weights are a paired per-128 affine
format, so the hot path runs on the `m16n8k32` INT8 MMA (≈2× the FP16 rate on Ada /
Blackwell) instead of FP16-with-dequant.

**Theory.** Numerical error stays O(1) with depth because the FP islands —
specifically the FP residual stream — break error compounding; each quantized matmul
contributes a bounded, per-128 error that does not accumulate across layers. INT8 is
therefore a *correctness-neutral optimization layered on the FP path*, never a
replacement: non-tensor-core hardware falls back to the untouched FP kernels.

**Design.** Two paired formats — `q8a128` (per-128 symmetric 8-bit activations) and
`KO` (per-128 affine weights, `Q4_KO`/`Q5_KO`/`Q6_KO`/`Q8_KO`) — feed a single INT8
matmul with two compile-time modes (decode `Bm=16`, prefill `Bm=32` weight-reuse)
selected from the token count at a measured crossover. A single boolean, `int8mode`,
threads through both an activation converter (`to_dynamic`) and a weight repack
(`QMatMul::repack_for_optimization`); a `DynamicTensor` operand carries the mode so
one matmul call hides whether it runs in INT8 or FP. The KO weight format is chosen
one notch up a precision ladder (`Q4_K → Q5_KO`) so re-quantizing an already-quantized
weight is near-lossless.

**Results.** On an RTX 4090 Mobile (sm_89, ~576 GB/s GDDR6) the codec kernels run at
**87–96 % of memory bandwidth** (weight quantize 540–550 GB/s, weight dequant
500–518 GB/s, activation quantize/dequant 510–540 GB/s). The INT8 matmul is
**~1.2–1.4× faster than FP16 on single-session decode** (`Q5_KO` / `Q4_KO`) and
**~1.5× on large prefills and high decode concurrency** — where each weight is read
from VRAM once and *massively reused* across the many tokens it serves, so the matmul
is compute-bound and the 2× INT8 MMA dominates. (The grouped MoE expert path reaches
this once its mode-2 weight-reuse kernel lands; the dense projections already do.) The
INT8 path reproduces the FP baseline to **rel-L2 0.5 % (Q8) → 3.1 % (Q4→Q5_KO)**, and
the GPU codecs are **byte-identical to their CPU references** and **bit-identical
between the dense and grouped matmul paths**.

---

## 1. Theory & goal

Make **8-bit the default activation interchange format on the tensor-core path**:
producers emit it, the matmul consumes it, and FP exists only where the math demands
it. The `m16n8k32` INT8 MMA is the engine; the quantized activations are the
connective tissue.

This is sound because of the model's structure. Every matmul output flows into an FP
island before the next matmul; the islands — and above all the **FP residual stream**
(the depth accumulator the O(1)-error theorem protects) — stop per-layer quantization
error from compounding. Each quantized site contributes a bounded per-128 error; the
chain stays O(1) as long as every individual site holds. INT8 is thus **additive and
TC-only**: a missing INT8 variant is a performance gap, never a correctness bug,
because non-TC hardware (and a handful of deliberately-FP outlier-sensitive sites)
route to the existing FP16/BF16/F32 kernels.

---

## 2. The `q8a128` activation format

`q8a128` — **q8 (8-bit) · a (activation) · 128 (K-tile width)** — is the per-128
activation block. It is **per-128, not per-32**: one `{scale, sum}` describes a whole
128-element K-tile.

```cpp
struct __align__(16) block_q8a128 {   // 144 bytes, per (token, K/128 tile)
    half2  ds[4];        // ds[0] = the tile's {scale, sum}; ds[1..3] = 16-byte align pad
    int8_t qs[128];      // contiguous, 16-byte aligned
};
static_assert(sizeof(block_q8a128) == 144, "");
```

- **`scale` = amax/127** (the symmetric activation scale `s_a`); **`sum` = Σx** (the
  raw float sum over the 128 elements, feeding the affine-weight min correction
  `m_w·Σx`). Both live in `ds[0]`; `ds[1..3]` are alignment pad and are never written.
- **Contiguous `qs`, 16-aligned** → one wide `cp.async` and the standard A-fragment
  load, no de-interleave on the activation side.

**The `q8a1024` super-block.** On-device, activations are stored flat-grouped: eight
consecutive 128-tiles share one self-contained **1152-byte** super-block
(`[1024 B: 8×qs[128]][128 B: 8×ds slots]`), keyed only by `flat_tile = token *
tiles_per_row + k_tile`. De-interleaving the int8 quants from the scales makes every
`qs` tile a clean 128-byte run → 100 % `cp.async` sector efficiency. The packing is
position-independent, so the unified `n`-only dispatch is byte-identical to the typed
path.

`q8a128` is a **first-class QType, never a stored weight** — it is only ever an
activation intermediate, so it lives in the matmul/quantize machinery and is correctly
absent from the `GgmlDType` storage enum and the KV-cache format set.

---

## 3. The `KO` weight format

KO ("**K**-quant **O**rdered") is the per-128 **affine** weight format the INT8 matmul
reads: one `(scale, min)` per 128 K per output row, 4–8-bit quants, laid out
**lane-major** so each warp lane pulls its four sub-`uint32`s in a single wide LDS.

- **Not a permutation of a K-quant.** Unlike the original K-quant compact blocks, KO
  carries its own per-128 affine `(scale, min)` and is **re-quantized straight from
  F32** (offline, one-shot weight prep). It is therefore lossy relative to the source
  weight, not bit-identical — the loss is the subject of §8.3.
- **Four formats:** `Q4_KO` (4-bit), `Q5_KO` (5-bit), `Q6_KO` (6-bit), `Q8_KO` (8-bit
  symmetric, `min = 0`). The extra Q5/Q6 streams (`hi` bit, `crumb` 2-bit) sit in
  contiguous tail regions so the dequant adds at most one scalar read per sub-block.

### The precision ladder (`to_ko(mode)`)

A model weight stored as a K-quant already lives on a per-32 / per-16 sub-scale grid;
re-quantizing it to the **coarser per-128** KO grid loses precision. `to_ko` picks the
KO twin by [`Int8Mode`](#5-the-control-surface-int8mode), trading that granularity loss
against weight bytes and bandwidth:

| source | Performance (same-width) | Precision (step-up) |
|---|---|---|
| Q2_K, Q3_K | `Q4_KO` | `Q4_KO` |
| Q4_0/1, Q4_K | `Q4_KO` | `Q5_KO` |
| Q5_0/1, Q5_K | `Q5_KO` | `Q6_KO` |
| Q6_K | `Q6_KO` | `Q6_KO` |
| Q8_0/1, Q8_K | `Q8_KO` | `Q8_KO` |

- **Performance** takes the **same-width** twin and eats the per-32→per-128 granularity
  hit on the weight. Smallest and fastest (e.g. `Q4_K → Q4_KO` keeps 4-bit weights).
- **Precision** steps **one notch up the ladder** so the extra bit absorbs the
  granularity loss and the re-quant is near-lossless (e.g. `Q4_K → Q5_KO`).

At the top of the ladder there is no finer twin (no `Q7_KO`), so `Q6_K → Q6_KO` and
`Q8_* → Q8_KO` are same-width in **both** modes — the per-128 hit is already small at
those bit depths (§8.3, §8.6). The activation (`q8a128`) is identical for both modes;
only the weight twin differs.

---

## 4. The INT8 matmul

The dense and grouped (MoE) matmuls share one INT8 core: `q8a128 × KO → FP32`.

**Inner math.** KO weights dequant register-direct into the `n8k32` B-fragment;
`mma_int8_m16n8k32` accumulates `C = Σ q_x·q_w` in int32 over the 128-K tile; the
per-128 fold collapses the four `k32` sub-accumulations into one int32 then applies a
single scale:

```
out += d_w · s_a · C + m_w · Σx
```

where `d_w = scale_w`, `m_w = min_w` come from the KO weight and `s_a = scale`,
`Σx = sum` from `q8a128`. For symmetric `Q8_KO` (`m_w = 0`) the min term vanishes.
Because the activation `sum` is the raw-float `Σx`, the affine min correction needs no
extra activation-scale factor — it is exact.

### Two modes (compile-time, runtime-dispatched)

| | mode-1 (decode) | mode-2 (prefill) |
|---|---|---|
| ytype | `Q8A128V` | `Q8A128X` |
| `Bm` (tokens/block) | 16 | 32 |
| `N_SUB` (m16 sub-tiles) | 1 | 2 |
| weight dequant | per token-tile | **reused across 2 token sub-tiles** |
| selected when | `M < 64` | `M ≥ 64` |

Both share an `ACT_BUFS=2` double-buffered `cp.async` pipeline (the next tile's weight
chunk + activation tile prefetch in one group during the current tile's MMA; a single
reused weight slot drained to registers, so no weight ring is needed). The crossover
at **M ≈ 64** is measured: below it the per-token weight re-read of mode-2's larger
tile costs more than it saves; at/above it weight reuse wins. The partial final M-tile
is masked at three layers (global reads gated by `b_cnt`, zero-padded smem, bounded
stores), so any M is safe.

### Exclusive pairing

KO weights and `q8a128` activations are **exclusively paired**: the INT8 kernels read
only KO weights, and KO weights are only consumed through the INT8 path. The matmul
guards this — `Int8` must pair with a KO weight and `Float` with a non-KO weight; any
cross combination has no kernel and is rejected with a clear error rather than
producing garbage.

---

## 5. The control surface (`Int8Mode`)

One enum, **`Int8Mode`**, drives the whole numeric mode of a matmul:

- **`Off`** — FP16 GEMM. The numeric reference; no int8 anywhere.
- **`Performance`** — int8 with the same-width KO weight twin (§3). Fastest, lossier.
- **`Precision`** — int8 with the stepped-up KO weight twin (§3). Near-lossless vs source.

Because KO ⇔ INT8 is guard-enforced, the one mode safely sets both sides:

```
Int8Mode ──┬──▶ QMatMul::repack_for_optimization(mode)   weight: KO twin ↔ FP GEMX
           └──▶ to_dynamic(xs, mode)                      acts:   q8a128   ↔ float
                              │
                              ▼
                  dense_qmatmul / grouped_qmatmul(DynamicTensor, weight, …)
```

- **`DynamicTensor`** is the matmul operand: `Float(&Tensor)` or `Int8(&Q8a128Operand)`.
  It borrows; the owned `DynamicActs` (returned by `to_dynamic`) holds the chosen
  representation and hands out a borrow.
- **`Q8a128Operand`** carries the packed blocks, the mode (`ytype`, chosen from M at
  the crossover), and the activation's leading dims — so the INT8 matmul rebuilds the
  output rank (`[B,M,K]→[B,M,N]`) exactly like the FP path.
- **`repack_for_optimization(mode)`** for an int8 mode dequantizes the weight on-device
  and re-quantizes to the mode's KO twin (`run_quantize_ko`, no host round-trip); for
  `Off` it does the FP GEMX repack. **`to_dynamic(xs, mode)`** quantizes the activation
  to `q8a128` for any int8 mode (identical bytes for Performance/Precision) and keeps the
  float tensor for `Off`.

Selecting one mode switches an entire matmul — and, threaded through inference, the whole
model — between FP16, Performance-int8, and Precision-int8, with the pairing guard
guaranteeing the weight and activation sides never diverge.

### Wiring into inference

`QMatMul` (the `candle-transformers` wrapper) carries its `Int8Mode`: `from_*_with_mode`
bakes the KO twin in at load, and `forward` dispatches to `forward_via_int8` (q8a128 ×
KO → F32, cast back to the compute dtype) when the mode is an int8 mode. For
Qwen3-30B-A3B, `ModelWeights::from_gguf_by_path_with_int8(…, mode)` threads the mode onto
every **dense** projection — attention q/k/v/o, the MoE router gate, dense-MLP gate/up/
down, and `lm_head` (the §6 producers/consumers). The default `from_gguf_by_path` entry
**auto-selects** the mode via `Int8Mode::auto(device)`: `Precision` when the GPU can run
the int8 `m16n8k32` MMA (compute capability ≥ 8.0 / Ampere+), else `Off`; the chosen mode
is logged at `info` on load. **Expert** weights flow through the
`ExpertCache` repack-to-host / DMA staging pipeline, which is gemx-only, so experts stay
FP16 in every mode; an int8 grouped-expert path needs a KO variant of that staging
pipeline (and the mode-2 grouped kernel, §10) and is future work.

---

## 6. The decode pipeline (FP islands + q8a128 producers)

`q8a128` is the only activation input to the high-performance matmul; FP exists only at
the irreducible islands — the **residual stream** (depth accumulator), **SwiGLU** (the
nonlinearity), and **softmax** (routing + attention scores). RoPE is inside attention,
so it is not an island.

Per-token matmul flow (Qwen3-30B-A3B). Each `q8a128` input buffer (B1–B5) is produced
**once** by an FP-island epilogue and reused across its fan-out; every weight GEMM is
`q8a128 × KO → FP`, the weight repacked to its KO twin at load:

| #   | matmul                | act in        | weight (stored → KO) | out →             | produced by          |
|-----|-----------------------|---------------|----------------------|-------------------|----------------------|
| —   | embedding lookup      | —             | —                    | FP `h` (residual) | stays FP             |
| 1–3 | q/k/v_proj            | **q8a128** B1 | Q4_K → Q5_KO         | FP → q/k-norm, KV | `ln1` RMSNorm ⇒ B1   |
| —   | attn QK · PV          | int8 (in-kern)| int8 KV              | FP ctx            | Q in FP; RoPE in-kern|
| 4   | o_proj                | **q8a128** B2 | Q4_K → Q5_KO         | FP → residual     | attention out ⇒ B2   |
| 5   | router                | **q8a128** B3 | Q4_K → Q5_KO         | FP → softmax      | `ln2` RMSNorm ⇒ B3   |
| 6–7 | gate/up_proj          | **q8a128** B3 | Q4_K → Q5_KO         | FP → SwiGLU       | `ln2` RMSNorm ⇒ B3   |
| 8   | down_proj             | **q8a128** B4 | Q4_K → Q5_KO         | FP → residual     | SwiGLU ⇒ B4          |
| 9   | lm_head *(post-loop)* | **q8a128** B5 | Q6_K → Q6_KO         | FP logits         | final RMSNorm ⇒ B5   |

B1 feeds 3 matmuls, B3 feeds 3 — **quantize once per fan-out**. Every output is FP
because it feeds an island, which is why there is no `q8a128`-out matmul variant: no
back-to-back GEMM exists to consume it. (The KO column shows the **Precision** twin;
**Performance** uses the same-width twin — `Q4_K → Q4_KO`, `Q6_K → Q6_KO` — per §3. The
expert gate/up/down GEMMs are **not** in this table: they run through the gemx-only
`ExpertCache` staging path and stay FP16 regardless of mode, per §5.)

### Producers (fused epilogues)

To keep the hot path free of standalone quant launches (for M=1 decode the launch + FP
round-trip dominate, so fusion is the win, not raw bandwidth), the quantize is fused
into the producer epilogue, each **byte-identical to `run_quantize_q8a128`** (the
raw-byte test is the oracle):

1. **RMSNorm** (`ln1`/`ln2`/final) — row-RMS reduce → scale → per-128 amax/sum →
   `q8a128`. Produces **B1, B3, B5**.
2. **SwiGLU** (`silu(gate)·up`) → per-128 quantize → `q8a128`. Produces **B4**.
3. **Attention decode epilogue** — emit `q8a128` ctx (**B2**); until fused, the
   standalone `run_quantize_q8a128` covers this boundary.

### Stays FP (deliberately)

The residual stream `h`, every matmul output, the attention **Q** input (RoPE is
in-kernel, so Q arrives FP and is int8-quantized *inside* the attention kernel), and
the **KV cache** (its own adaptive 32-element C0–C9 format, not `q8a128`).

---

## 7. Invariants

1. **INT8 is a TC-only optimization; FP is the correctness floor.** Every INT8 kernel
   is additive; a missing variant is a perf gap, not a bug.
2. **On TC, the matmul input is ~always `q8a128`.** A q8-emitting producer feeds every
   matmul; FP-in is the exception (non-TC fallback + deliberately-FP outlier sites).
3. **KO ⇔ q8a128 are exclusively paired** (§4), so a single flag controls both sides
   without the two ever diverging.
4. **The same data in two byte-orders** (the q8a128 activation vs the KO weight) stays
   semantically locked — the byte-exact round-trip tests pin both.

---

## 8. Results (measured — RTX 4090 Mobile, sm_89, ~576 GB/s GDDR6)

### 8.1 Codec throughput

The quantize/dequant kernels are bandwidth-bound. Vectorization (`float4` loads,
`uint32`/`uchar4`/`float4` stores) plus format-appropriate cache hints (`__stcs` on the
write-once dequant output; `__ldcs` on the write-heavy `Q8_KO` quantize input only)
bring every path to **87–96 % of GDDR6 peak**.

**KO weights** (`ko_quant_throughput_bench`, 8192×8192, GB/s on the f32 side):

| format | quantize | dequant | vs f32 size |
|--------|---------:|--------:|------------:|
| Q4_KO  | 540 | 518 | 7.53× |
| Q5_KO  | 544 | 504 | 6.10× |
| Q6_KO  | 545 | 510 | 5.12× |
| Q8_KO  | 550 | 515 | 3.88× |

**q8a128 activations** (`q8a128_throughput_bench`, 8192×4096, GB/s):

| input | quantize | dequant |
|-------|---------:|--------:|
| f32   | 539 | 533 |
| f16   | 514 | 531 |
| bf16  | 510 | 528 |

### 8.2 What the matmul benchmarks measure

Three benchmarks at the gate/up dimensions (`[768×2048]`, weights pre-prepared so
quantize is off the timed path; column = FP16-time / INT8-time) probe different
weight-reuse regimes. **The regime, not the kernel, sets the INT8 win:**

- **Dense** (`ko_vs_k_int8_bench`) — one weight reused across all M tokens. After the
  first read the weight is L2-resident, so the kernel is **compute-bound** and INT8's
  faster MMA shows cleanly.
- **Grouped, single expert** (`grouped_int8_vs_legacy_bench`) — the grouped (MoE)
  kernel path on one L2-cached expert; confirms it tracks the dense kernel.
- **MoE simulator** (`grouped_moe_ko_vs_k_bench`) — the **production** path: 128
  distinct experts, top-8 routing (Qwen3-30B-A3B config), one launch over all
  (expert→token-slice) tiles. Each weight is read once but reused across its expert's
  token-slices, so the read amortizes — though the grouped kernel is currently mode-1
  only, so it does not yet amortize as fully as the dense mode-2 path (§8.5).

### 8.3 Dense (single weight, compute-bound)

INT8-KO beats FP16 across all M; the mode-1/mode-2 crossover sits at M ≈ 64:

| FP16 src → KO | M=1 | M=32 | M=128 | M=1024 | M=4096 |
|---------------|----:|-----:|------:|-------:|-------:|
| Q4_K → Q4_KO  | 1.28 | 1.76 | 1.35 | 1.42 | 1.10 |
| Q4_K → Q5_KO  | 1.22 | 1.49 | 1.28 | 1.11 | 0.92 |
| Q5_K → Q6_KO  | 1.28 | 1.80 | 1.58 | 1.24 | 1.10 |
| Q8_K → Q8_KO  | 1.62 | 1.61 | 1.60 | 1.52 | 1.22 |

INT8 wins **1.2–1.8× through decode and into large prefill** — the abstract's ~1.5× is
here at M ≈ 512–1024 (`Q8` 1.52×, `Q4_KO` 1.42×), the realistic high-concurrency regime,
where the weight read is amortized across the tile's tokens by mode-2. Only at the
extreme M = 4096 does even a cached weight saturate the read ports and the win soften
toward parity; the `Q4_K → Q5_KO` dip there (0.92×) is the +1-bit twin reading *more*
bytes than the FP16 Q4_K source.

### 8.4 Grouped, single expert (compute-bound)

The grouped kernel on one L2-cached expert tracks the dense kernel — INT8-KO **1.0–1.5×**
across M (`Q4→Q5_KO` 1.0–1.5×, `Q5→Q6_KO` 1.0–1.4×, `Q8→Q8_KO` 1.05–1.5×), confirming
the grouped path carries no penalty over dense when the weight is reused.

### 8.5 MoE simulator (128 experts — the production regime)

128 distinct experts, top-8 routing, FP16 K-quant vs the INT8 KO twin, one grouped
launch. At full load (batch ≥ 128) ~all 128 experts are active; the **sweet spot is
16–32 token-slices per expert** (batch 256–512), where decode throughput peaks
(~13k–22k tok/s). Speedups (FP16-K / INT8-KO), stable batch≥16 rows:

| FP16 src → KO | b=8 (~1/exp) | b=32 (~2/exp) | b=128 (8/exp) | b=256 (16/exp) | b=512 (32/exp) |
|---------------|---:|---:|---:|---:|---:|
| Q4_K → Q5_KO  | 1.30 | 0.95 | 0.96 | 0.92 | 1.03 |
| Q5_K → Q6_KO  | 1.38 | 1.11 | 1.12 | 1.09 | 1.09 |
| Q8_K → Q8_KO  | 1.48 | 1.20 | 1.19 | 1.18 | 1.16 |

Two things hold this measurement below the dense §8.3 ceiling, and **neither is a
fundamental bandwidth wall** — each weight is read once but *reused across its expert's
token-slices*, so the read amortizes the way the dense path's does:

1. **The grouped path is mode-1 only.** It re-reads each expert's weight once per
   16-token tile, so at 32 slices/expert it reads the weight twice and does not yet get
   the dense **mode-2 weight-reuse** (`Bm=32`). A mode-2 grouped kernel (future work,
   §10) recovers that reuse and lifts these rows to the dense numbers.
2. **The +1-bit step-up costs bytes.** `Q4_K → Q5_KO` reads more than the FP16 Q4_K
   source, so it shows worst here; same-width `Q8_K → Q8_KO` already wins **~1.2×** at
   mode-1, and `Q4_K → Q4_KO` (same bytes, where accuracy permits) recovers the win.

The real production picture is the **amortized** one. A weight read once from VRAM and
then massively reused across its expert's tokens makes the matmul compute-bound, where
INT8's 2× MMA dominates: **~1.2× single-session decode (`Q5_KO`; ~1.4× `Q4_KO`)** and
**~1.5× on large prefills and high decode concurrency** — the §8.3 dense numbers at
M ≈ 512–1024, which the dense projections (q/k/v/o, lm_head) already reach via mode-2
and which the grouped expert path reaches once mode-2 lands. With 128 experts and
saturated sessions the per-expert token count sits squarely in that reuse regime.

### 8.6 Precision

**KO codec round-trip vs F32** (`ko_quantize_dequant_roundtrip`) — the per-128 affine
quantization floor (≈ 1/maxq/√3):

| Q4_KO | Q5_KO | Q6_KO | Q8_KO |
|------:|------:|------:|------:|
| 0.065 | 0.033 | 0.017 | 0.004 |

**INT8 path vs the FP baseline** (`qmatmul_int8mode_baseline_bit_check`) — the same
weight run through `Off` (dequant-K-quant × float acts, the baseline) and through each
int8 mode (KO twin × `q8a128`), rel-L2 of the two matmul outputs. The residual is the
KO weight re-quant plus the 8-bit activation; Precision's step-up makes the weight term
near-lossless, Performance keeps the same width and so reads lighter but loses more:

| source | Performance twin → rel-L2 | Precision twin → rel-L2 |
|--------|--------------------------:|------------------------:|
| Q4_K | `Q4_KO` → **0.054** | `Q5_KO` → **0.032** |
| Q5_K | `Q5_KO` → 0.030 | `Q6_KO` → 0.017 |
| Q6_K | `Q6_KO` → 0.017 | `Q6_KO` → 0.017 |
| Q8_0 | `Q8_KO` → **0.005** | `Q8_KO` → **0.005** |

The delta shrinks with the twin's bit width (Q4 → Q8) and is stable across runs. The
mode is the precision/footprint knob: at Q4_K, Performance's same-width `Q4_KO` measures
**0.054** while Precision's +1-bit `Q5_KO` halves that to **0.032**; at Q6/Q8 the modes
coincide (top of the ladder). For reference, against an F32 ground truth
(`qmatmul_int8mode_flag_end_to_end`, Q4_K weight) the three modes measure Off **0.058**,
Performance **0.081**, Precision **0.068**.

### 8.7 Correctness

- **Byte-exact codecs.** GPU `quantize_ko` is **byte-identical** to the CPU reference
  for all four formats, and GPU `dequant_ko` is bit-exact (`ko_gpu_quantize_dequant_matches_cpu`).
  The `q8a128` quantize is raw-byte-asserted and its dequant is bit-exact.
- **Path equivalence.** The dense and grouped matmuls produce **bit-identical** output
  on the INT8 KO path (rel-L2 **0.000000**, all four formats —
  `dense_int8_matches_grouped`) and on the FP path (rel-L2 **0.000000** —
  `dense_qmatmul_float_matches_grouped`).
- **Flag machinery.** `to_dynamic` + `repack_for_optimization` + `dense_qmatmul`
  reconstruct the F32 ground-truth matmul at both flag settings within budget, and the
  INT8 dense arm preserves 3D activation rank exactly like the FP arm.
- **Pairing guard.** The two supported combinations run and the two unsupported crosses
  (Int8 × non-KO, Float × KO) are rejected, on both the dense and grouped entries.

### 8.8 End-to-end model integration (dense path)

The INT8-KO path is now **wired into the Qwen3-30B-A3B forward** for every **dense**
projection (attention q/k/v/o, MoE router gate, dense-MLP gate/up/down, `lm_head`), driven
by `Int8Mode` (§5). **Expert** GEMMs stay FP16 in every mode (the gemx-only `ExpertCache`
staging path; §5). These `test_parallel_batched_forwarding` runs compare the three modes
on the **same** built binary (selected by the `INT8MODE` env var), RTX 4090 Mobile, generate
throughput; `bulk` = aggregate over concurrent sessions, `single` = per-session. The `Off`
column reproduces the prior FP16 baseline exactly (all MoE machinery — ~67–70 % expert hit
rate, KV-tier DMA, prediction — is byte-for-byte unchanged):

| KV mode | sessions | bulk: Off → Perf / Prec | single: Off → Perf / Prec |
|---------|---------:|-------------------------|---------------------------|
| F16  | 1  | 423 → 435 / 420 | 10.2 → 9.4 / 10.3 |
| BF16 | 10 | 1394 → 1418 / 1423 | 138 → 125 / 126 |
| Q8_0 | 20 | 1875 → 1846 / 1837 | 196 → 183 / 185 |
| Q4_0 | 20 | 1959 → 1896 / 1897 | 225 → 204 / 213 |
| BF16 (warm) | 1 | 589 → 643 / 605 | 34.7 → 19.9 / 20.6 |
| C10 (max KV) | 5 | 1769 → 1766 / 1719 | 93 → 69 / 74 |

**Reading the result.** With experts held at FP16, the dense projections are a minority of
the model's FLOPs, so the dense-only effect is small and **direction-split**:

- **Bulk / prefill** (high effective M, compute-bound) is **neutral-to-slightly-positive** —
  e.g. BF16-warm bulk +9 %, BF16×10 +2 %; deltas under ~5 % are within the run-to-run noise
  visible in the F16×1 row.
- **Single-token decode** (M = 1) **regresses ~6–40 %**. This is the predicted unfused-quant
  penalty (§6): each dense matmul currently does a **standalone** `to_dynamic` quantize launch
  + an F32→F16 cast, ~9 dense GEMMs × 48 layers of extra launches per token, which at M = 1
  has no compute to amortise against. It is *not* a property of the int8 matmul — it is the
  missing producer-epilogue fusion.

**Performance vs Precision are within noise of each other** on throughput: both pay the same
quantize-launch overhead, and the only difference (Q4_KO vs Q5_KO dense weight bytes) is a
small fraction of total traffic. So **Precision is effectively free** here — take its tighter
accuracy (§8.6) at no measured decode cost.

The two levers that turn this into a win are exactly the documented future work (§10): (1)
**fuse the q8a128 producers** into the RMSNorm/SwiGLU epilogues to delete the standalone
decode launches (flips the M = 1 regression to a gain), and (2) the **KO expert path +
mode-2 grouped kernel** to put the bulk of the FLOPs (the experts) onto int8.

**Qwen3 (dense)** — higher absolute t/s (no expert DMA); the BF16 no-batch baseline is
run-to-run variant (~1.6k–2.3k t/s), so compare relatively:

| KV mode | sessions | t/s (bulk) | t/s (single) | weight compress |
|---------|---------:|-----------:|-------------:|----------------:|
| BF16 (no batch) | 1  | ~1.6k–2.3k | 34–38 | — (baseline) |
| Q8_0            | 4  | ~2.5k | 146 | 1.88× |
| C8 (adaptive KV)| 10 | 2453  | 249 | 4.83× |
| C10 (max KV)    | 5  | 2472  | 154 | 7.46× |

The INT8-KO matmul (§4) targets exactly the projection + FFN GEMMs these runs spend their
decode/prefill time in. Wiring `repack_for_optimization` + `to_dynamic` into the model
forward is what carries the §8.3–8.5 kernel wins (≈1.2–1.4× decode, ≈1.5× concurrent)
into these end-to-end numbers — the next integration step (§6, §10).

---

## 9. Conclusion — what was achieved

The INT8 tensor-core matmul and its two paired codecs are **built, byte-exact, and
benchmarked end-to-end**:

- A **per-128 `q8a128` activation format** and a **per-128 affine `KO` weight format**
  (four bit-widths on a precision ladder), each with vectorized GPU quantize/dequant
  kernels running at **87–96 % of memory bandwidth** and byte-identical to their CPU
  references.
- A **dense + grouped INT8 matmul** with decode (`Bm=16`) and prefill (`Bm=32`
  weight-reuse) modes auto-selected at the measured **M ≈ 64** crossover, **~1.2–1.4×
  faster than FP16 on single-session decode** and **~1.5× on large prefills and high
  decode concurrency** (where the weight read amortizes across the many tokens reusing
  it), reproducing the FP baseline to **0.5–3.1 % rel-L2** and bit-identical between the
  dense and grouped paths.
- A **single-`int8mode`-flag control surface** (`DynamicTensor` / `to_dynamic` /
  `repack_for_optimization`) that switches an entire matmul — and, threaded through
  inference, an entire model — between INT8 and FP, with a guard that makes the
  KO ⇔ `q8a128` pairing impossible to get wrong.

This realizes the design's matmul and quantization machinery. The remaining work is
**§6's fused producer epilogues** — emitting `q8a128` directly from the RMSNorm,
SwiGLU, and attention-decode epilogues so the hot path carries no standalone quant
launch — and the **end-to-end per-model accuracy validation** (perplexity) that the
O(1) argument reduces to "every site holds."

---

## 10. Risks & future work

- **Mode-2 grouped kernel** (the headline lever) — the grouped MoE expert path is
  currently mode-1 only, so it re-reads each expert's weight per 16-token tile and
  leaves the §8.5 win at ~1.2×. A `Bm=32` weight-reuse grouped kernel (the dense
  mode-2, applied per expert) amortizes the weight read across the per-expert tokens
  and lifts the MoE prefill to the dense ~1.5×.
- **Accuracy compounding** — the per-site outlier risk (§6) is bounded by the FP
  residual but must be measured per model; the mitigation ladder is finer blocks →
  SmoothQuant-style weight-side scale migration → FP8 for offending sites.
- **`Q4_K → Q5_KO` byte cost** — the +1-bit step-up reads more than the Q4_K source,
  the worst case at the bandwidth margin; `Q4_K → Q4_KO` (same bytes) recovers the win
  where the per-128 4-bit re-quant holds accuracy.
- **Fused epilogue byte-exactness** — each producer must match `run_quantize_q8a128`
  byte-for-byte; the raw-byte tests are the gate.
- **KV-cache boundary** — the QKV output format must align with the adaptive C0–C9 KV
  format for the attention-side stretch win.
