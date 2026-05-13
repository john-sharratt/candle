# FP8 KV Cache Quantization: F8_0 and F8_1

## Overview

F8_0 and F8_1 are floating-point 8-bit quantization types for the paged KV cache, extending the existing Q4_0, Q4_1, Q8_0, and Q8_1 integer quantization family. They share the same 32-token block granularity and per-block metadata structure as the integer variants, but store values in IEEE FP8 E4M3 format rather than as uniform integer levels.

The motivation is twofold: improved model quality through non-uniform value distribution better matched to KV activation statistics, and improved compute throughput by enabling native FP8 tensor core operations on Ada Lovelace hardware (sm_89), eliminating the pre-MMA dequantization step required by all integer quant types.

---

## FP8 Format: E4M3

Both F8_0 and F8_1 use the **E4M3** variant of FP8 (4 exponent bits, 3 mantissa bits), consistent with the inference and forward-pass FP8 convention established by NVIDIA and the ML community.

Key properties of E4M3:

- Representable range: approximately ±448.0
- Mantissa precision: 3 bits (~12.5% relative spacing between adjacent values at any given exponent)
- Non-uniform value spacing: density is highest near zero, increasing sparsity toward ±448
- No Inf representation; two NaN encodings exist: `0x7F` and `0xFF` (exp=15, mant=7, both signs). All other 254 bit patterns are finite values.

The non-uniform spacing is the defining characteristic that distinguishes F8_x from Q8_x. KV cache activations are typically concentrated near zero with sparse tails — a distribution that FP8's higher near-zero density matches naturally, whereas Q8_0's 256 uniform integer levels allocate equal representational capacity across the full range regardless of occupancy.

---

## Block Structure

Both types operate at **32-token block granularity**, identical to the existing Q8_x and Q4_x types. Metadata is stored as fp16, providing approximately 125× more precision than the fp8 values themselves — sufficient to sit well below the FP8 quantization noise floor.

### F8_0

Symmetric quantization. One scale factor per block.

```
[ 32 × fp8 values ] [ fp16 scale ]
  32 bytes            2 bytes
  ─────────────────────────────────
  Total: 34 bytes per block
```

This is byte-for-byte identical in footprint to Q8_0 (34 bytes).

**Quantization:**
```
scale    = max(|x|) / 448.0
x_fp8    = f32_to_fp8(x / scale)
```

**Dequantization:**
```
x_f32    = fp8_to_f32(x_fp8) * scale
```

### F8_1

Centre-point quantization. One scale factor and one centre value per block.

```
[ 32 × fp8 values ] [ fp16 scale ] [ fp16 center ]
  32 bytes            2 bytes        2 bytes
  ──────────────────────────────────────────────────
  Total: 36 bytes per block
```

This is byte-for-byte identical in footprint to Q8_1 (36 bytes).

**Quantization:**
```
actual_min  = min(x)
actual_max  = max(x)
center      = (actual_max + actual_min) / 2.0
scale       = (actual_max - actual_min) / (2.0 × 448.0)
x_fp8       = f32_to_fp8((x - center) / scale)
```

**Dequantization:**
```
x_f32       = fp8_to_f32(x_fp8) * scale + center
```

---

## Why F8_1 Over F8_0

The centre term provides a qualitatively more significant benefit for FP8 than the equivalent Q8_0 → Q8_1 transition does for integer quantization. The reason is structural and exploits a property unique to FP8.

**The sign bit problem with F8_0**

F8_0 is symmetric around zero. For a block whose actual value range is [3.2, 9.7], F8_0 must set its scale to cover the full ±9.7 range — but the negative half of the FP8 range [−9.7, 0] is entirely unoccupied. The sign bit is present in E4M3 but wasted; you are representing only positive values with a signed format, discarding half the effective dynamic range.

**What the centre term does**

F8_1 stores `center = (actual_max + actual_min) / 2` and shifts the block so its distribution is centred at zero before quantizing. The FP8 values now span **±448 symmetrically** around the actual distribution centre — the full E4M3 range, including the sign bit, is active and carrying information.

For the [3.2, 9.7] example: center = 6.45, the shifted block covers [−3.25, +3.25]. The scale is derived from the half-range (3.25 / 448.0) rather than the full range (9.7 / 448.0) — a 3× smaller scale, meaning representations occupy a much lower and finer exponent band in E4M3.

**The two compounding benefits**

First, range compression. The half-range scale is smaller than the full-range scale, pulling FP8 representations into lower exponent territory where mantissa precision is finer. Values that would have sat in the coarse high-exponent region of [0, 9.7] now sit in the fine low-exponent region of [0, 3.25].

Second, full sign bit utilisation. E4M3's symmetric ±448 range is fully exploited. A min-term approach covering [0, 6.5] uses only positive E4M3 values — approximately half the representable dynamic range. The centre term approach covering [−3.25, +3.25] uses both halves, with E4M3's near-zero density applied to the actual centre of the data distribution where it is most valuable.

These two effects compound: smaller scale and symmetric coverage together mean the active data maps into the densest, most precise region of the E4M3 number line.

**Q8_1 comparison**

Q8_1's min term helps with asymmetric distributions by redistributing uniform integer levels more efficiently. It does not fix any fundamental precision pathology — Q8_0's uniform spacing has no top-end degradation to correct. For FP8, the centre term addresses a genuine structural issue (wasted sign bit, high-exponent coarseness) rather than merely optimising level distribution.

This effect is most pronounced for V cache activations, which tend to have non-zero-centred distributions more often than K cache, making the block offset a common rather than exceptional condition.

---

## Compute Path

### Integer quantization path (Q8_0 baseline)

```
int8 values
  → dequant multiply (pre-MMA, every element)
  → BF16
  → MMA on BF16 tensor cores
  → FP32 accumulator
```

The dequantization step is synchronous with and blocking to the MMA pipeline.

### F8_0 / F8_1 compute path (Ada Lovelace, sm_89)

```
fp8 values
  → native FP8 MMA on FP8 tensor cores
  → FP32 accumulator
  → scale multiply + center add (post-accumulation, once per block boundary)
```

Two structural differences from the integer path:

**1. No pre-MMA dequantization.** FP8 values feed directly into the tensor core MMA operation. The conversion cost is eliminated entirely, not reduced.

**2. Post-accumulation metadata application.** Scale (and centre for F8_1) are applied once per block to the FP32 accumulator after the 32-token MMA completes. This is amortised across 32 token positions and is computationally negligible relative to the MMA work itself.

**3. Native FP8 tensor core throughput.** Ada Lovelace FP8 tensor core FLOPS is nominally 2× BF16 tensor core FLOPS. The MMA operation itself runs faster.

The combined effect is elimination of pre-MMA element-wise overhead plus a doubling of tensor core throughput on the MMA path itself.

---

## Comparison Table

| Property | Q4_0 | Q8_0 | F8_0 | F8_1 |
|---|---|---|---|---|
| Bits per value | 4 | 8 | 8 | 8 |
| Value format | int4, uniform | int8, uniform | fp8 E4M3, non-uniform | fp8 E4M3, non-uniform |
| Block metadata | 1× fp16 scale | 1× fp16 scale | 1× fp16 scale | 1× fp16 scale + 1× fp16 center |
| Bytes per block | 18 | 34 | 34 | 36 |
| Levels / precision | 16 uniform | 256 uniform | Non-uniform, near-zero dense | Non-uniform, centre-shifted, full ±448 |
| Pre-MMA dequant | Unpack + multiply | Multiply | None | None |
| MMA type | BF16 | BF16 | FP8 native | FP8 native |
| Asymmetric correction | No | No | No | Yes (centre term, full sign bit) |
| Top-end precision loss | — | — | Present | Mitigated by centre compression |
| Model quality | Lossy | Good | Better | Best |
| Relative storage vs Q8_0 | 0.53× | 1.0× | 1.0× | 1.06× |

---

## Quality Expectations

F8_0 and F8_1 do not represent a dramatic quality step change over Q8_0 — Q8_0 is already a high-fidelity format. The improvement is real but concentrated in specific conditions:

**Where the gain is most visible:**

- Long context generation (3,200+ tokens), where KV quantization error accumulates across sequence depth. FP8's better precision at actual activation values reduces error accumulation relative to uniform integer spacing.
- Attention heads in deeper layers with sparse, peaked activation distributions — where Q8_0's uniform levels waste representational capacity on unoccupied ranges.
- V cache specifically for F8_1, where non-zero-centred distributions make the centre term's range compression and full sign bit utilisation most effective.

**Where the gain is less visible:**

- Short context benchmarks where accumulated error has little depth to compound.
- Perplexity on standard evaluation sets, where the delta is likely sub-0.3 points.

The more significant gain relative to Q8_0 is on the compute path, not model quality. F8_1 provides both improvements simultaneously at a storage overhead of only 6% over Q8_0 (36 vs 34 bytes per block).

---

## Block Alignment

The 36-byte F8_1 block size is non-power-of-two, the same constraint as Q8_1. Alignment handling in the paged block allocator follows the same approach as Q8_1.

F8_0 at 34 bytes is identical to Q8_0 and inherits its alignment handling without modification.

---

## Implementation Notes

**Metadata precision.** Scale and centre are stored as fp16. This provides approximately 125× more precision than the fp8 values they describe, well below the FP8 quantization noise floor. Upgrading to fp32 metadata would increase F8_1 block size from 36 to 40 bytes (an 11% increase) for precision that cannot be observed under fp8 noise — not justified.

**Post-accumulation scale application.** The scale (and centre) correction must be applied to the FP32 accumulator after the block MMA completes, not inline per element. The block boundary in the paged KV attention kernel is the natural application point.

**E4M3 saturation.** Values exceeding ±448 after centre-shifting and scaling saturate to the E4M3 max. For well-calibrated block scales this should not occur; the scale is derived from the actual block half-range.

**Kernel path for F8_1 centre application.**
```
x_dequant = fp8_to_f32(x_fp8) * scale + center
```
In the native FP8 MMA path, this becomes a post-accumulation fused multiply-add per block, not a per-element operation.