//! One routed expert on the CPU: gate and up, SwiGLU, down, for every token
//! routed to it.
//!
//! The expert is read where it lies — a warm or pad slot image, the three
//! projections at the slot's fixed offsets (`slot_image::slot_offsets`) — and
//! the output is one F32 row of `hidden` per token, unweighted; the pool
//! applies the routing weights when it sums a token's experts.

use std::ops::Range;

use super::act::ActTiles;
use super::ko::{Ko, KoMatrix, Permuted};
use super::simd::{matvec, Isa};

/// An expert's three projections, borrowed from one slot image.
#[derive(Debug, Clone, Copy)]
pub struct Expert<'a> {
    pub gate: KoMatrix<'a>,
    pub up: KoMatrix<'a>,
    pub down: KoMatrix<'a>,
}

impl<'a> Expert<'a> {
    /// The expert in `slot`, its projections at `(gate, up, down)` byte
    /// offsets, for `hidden` → `ff` → `hidden`.
    pub fn from_slot(
        slot: &'a [u8],
        offsets: (usize, usize, usize),
        hidden: usize,
        ff: usize,
        ko: Ko,
    ) -> Self {
        let (g, u, d) = offsets;
        Self {
            gate: KoMatrix::new(&slot[g..], ff, hidden, ko),
            up: KoMatrix::new(&slot[u..], ff, hidden, ko),
            down: KoMatrix::new(&slot[d..], hidden, ff, ko),
        }
    }

    pub fn hidden(&self) -> usize {
        self.gate.ncols
    }

    pub fn ff(&self) -> usize {
        self.gate.nrows
    }
}

/// `e^x` for `x ≤ 0` as the device's SwiGLU computes it
/// (`fast_exp.cuh`, `exp_f32_high_softmax_ptx`): the input clamped at −88,
/// scaled by log2(e), its fractional part through the cubic minimax for 2^x
/// in three round-to-nearest FMAs, times 2^floor built from the exponent bits.
pub(crate) fn device_exp(x: f32) -> f32 {
    let xc = x.max(-88.0);
    let x2 = xc * f32::from_bits(0x3FB8_AA3B);
    let xi = x2.floor();
    let xf = x2 - xi;
    let poly = xf
        .mul_add(f32::from_bits(0x3D9D_9653), f32::from_bits(0x3E69_1E05))
        .mul_add(xf, f32::from_bits(0x3F31_F70E))
        .mul_add(xf, 1.0);
    poly * f32::from_bits(((xi as i32 + 127) << 23) as u32)
}

/// The device's `silu(x)`: `x · sigmoid(x)`, the exponent's argument kept
/// non-positive. Everything but the division is the device's bit for bit. The
/// device divides approximately (fast math: `rcp.approx`, and `e · rcp(1 + e)`
/// for the negative branch); no CPU instruction reproduces that reciprocal, so
/// here the division is exact. Measured over 4.2M gate values, 7% of results
/// sit an ulp from the device's and every one of them is that division
/// (`gpu_parity`). After quantization that very rarely moves an int8 code of
/// the down projection's input, which is all that separates the pool's
/// experts from the GPU's.
pub(crate) fn device_silu(x: f32) -> f32 {
    let s = if x >= 0.0 {
        1.0 / (1.0 + device_exp(-x))
    } else {
        let e = device_exp(x);
        e / (1.0 + e)
    };
    x * s
}

/// `silu(g) · u`, elementwise, as the device's fused SwiGLU-quantize computes
/// it (`fused_silu_mul.cu`): the down projection's input then quantizes to the
/// GPU's codes wherever the two do not straddle a rounding boundary.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = device_silu(g) * u;
    }
}

/// The gate and up projections for rows `rows` of the expert's FF width:
/// `gate_out` and `up_out` are `tokens × rows.len()`.
pub fn gate_up(
    isa: Isa,
    e: &Expert,
    act: &ActTiles,
    pa: &Permuted,
    rows: Range<usize>,
    gate_out: &mut [f32],
    up_out: &mut [f32],
) {
    matvec(isa, &e.gate, act, pa, rows.clone(), gate_out);
    matvec(isa, &e.up, act, pa, rows, up_out);
}

/// The whole expert for every token in `act` (`tokens × hidden` codes), into
/// `out` (`tokens × hidden`), on one thread — the reference the pool's split
/// schedule is tested against.
pub fn forward(isa: Isa, e: &Expert, act: &ActTiles, out: &mut [f32]) {
    let (tokens, hidden, ff) = (act.tokens, e.hidden(), e.ff());
    assert_eq!(out.len(), tokens * hidden);
    let pa = Permuted::new(act, e.gate.ko);
    let mut g = vec![0f32; tokens * ff];
    let mut u = vec![0f32; tokens * ff];
    gate_up(isa, e, act, &pa, 0..ff, &mut g, &mut u);
    let mut h = vec![0f32; tokens * ff];
    swiglu(&g, &u, &mut h);
    let ha = ActTiles::quantize(&h, tokens, ff);
    let hp = Permuted::new(&ha, e.down.ko);
    matvec(isa, &e.down, &ha, &hp, 0..hidden, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::kernel::ko::matvec_scalar;
    use crate::models::expert_lre::cpu_pool::kernel::simd::tiers_under_test;
    use candle::quantized::ko_quant::{dequant_ko, quantize_ko};
    use candle::quantized::GgmlDType;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// The device's exponential, raw: exact at 0, the polynomial's own value
    /// at the bucket edges, 0 past the clamp's floor bucket, and within its
    /// 0.0088% of `exp` across the range, plus the f32 roundings of its last
    /// two steps.
    #[test]
    fn the_device_exp_is_the_polynomial() {
        assert_eq!(device_exp(0.0), 1.0);
        // x = −ln 2 → x2 = −1 (to rounding), so xi = −1 or −2 by a hair.
        let half = device_exp(-std::f32::consts::LN_2);
        assert!((half - 0.5).abs() < 1e-4, "{half}");
        assert_eq!(device_exp(-200.0), device_exp(-88.0));
        for i in 0..1000 {
            let x = -(i as f32) * 0.05;
            let rel = (device_exp(x) as f64 / (x as f64).exp() - 1.0).abs();
            assert!(rel < 1.1e-4, "x {x}: {rel}");
        }
        assert_eq!(device_silu(0.0), 0.0);
        assert!((device_silu(2.0) - 2.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-4);
    }

    /// A small expert through `forward`:
    /// - byte-equal to the chain composed by hand from the scalar matvec,
    ///   SwiGLU and the down input's quantization — the wiring;
    /// - byte-equal on every available tier;
    /// - within 5% relative L2 of the f32 chain over the dequantized weights
    ///   and unquantized activations. That bound is a sanity check, not the
    ///   accuracy gate: two int8 activation roundings separate the two, the
    ///   second over a heavy-tailed SwiGLU output. The gate is the CPU's error
    ///   against the GPU's on the same reference (`docs/cpu_expert_compute.md` §6).
    #[test]
    fn forward_is_the_composed_chain() {
        let (hidden, ff, tokens) = (512, 256, 3);
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let mut seed = 21;
            let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(&mut seed) * 0.1).collect() };
            let (wg, wu, wd) = (w(ff * hidden), w(ff * hidden), w(hidden * ff));
            let (bg, bu, bd) = (
                quantize_ko(&wg, ff, hidden, dt),
                quantize_ko(&wu, ff, hidden, dt),
                quantize_ko(&wd, hidden, ff, dt),
            );
            let mut slot = Vec::new();
            let offsets = (0, bg.len(), bg.len() + bu.len());
            slot.extend_from_slice(&bg);
            slot.extend_from_slice(&bu);
            slot.extend_from_slice(&bd);
            let e = Expert::from_slot(&slot, offsets, hidden, ff, ko);

            let mut seed = 99;
            let x: Vec<f32> = (0..tokens * hidden).map(|_| lcg(&mut seed)).collect();
            let act = ActTiles::quantize(&x, tokens, hidden);
            let mut out = vec![0f32; tokens * hidden];
            forward(Isa::Scalar, &e, &act, &mut out);

            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            let mut g = vec![0f32; tokens * ff];
            let mut u = vec![0f32; tokens * ff];
            matvec_scalar(&e.gate, &act, 0..ff, &mut g);
            matvec_scalar(&e.up, &act, 0..ff, &mut u);
            let mut h = vec![0f32; tokens * ff];
            swiglu(&g, &u, &mut h);
            let mut composed = vec![0f32; tokens * hidden];
            matvec_scalar(
                &e.down,
                &ActTiles::quantize(&h, tokens, ff),
                0..hidden,
                &mut composed,
            );
            assert_eq!(
                bits(&out),
                bits(&composed),
                "{ko:?}: forward ≠ composed chain"
            );

            let (dg, du, dd) = (
                dequant_ko(&bg, ff, hidden, dt),
                dequant_ko(&bu, ff, hidden, dt),
                dequant_ko(&bd, hidden, ff, dt),
            );
            let (mut err, mut norm) = (0f64, 0f64);
            for t in 0..tokens {
                let xs = &x[t * hidden..(t + 1) * hidden];
                let mv = |w: &[f32], rows: usize, cols: usize, v: &[f32]| -> Vec<f64> {
                    (0..rows)
                        .map(|r| {
                            (0..cols)
                                .map(|k| w[r * cols + k] as f64 * v[k] as f64)
                                .sum()
                        })
                        .collect()
                };
                let (g, u) = (mv(&dg, ff, hidden, xs), mv(&du, ff, hidden, xs));
                let h: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(&g, &u)| (g / (1.0 + (-g).exp()) * u) as f32)
                    .collect();
                let y = mv(&dd, hidden, ff, &h);
                for (r, &yv) in y.iter().enumerate() {
                    err += (out[t * hidden + r] as f64 - yv).powi(2);
                    norm += yv.powi(2);
                }
            }
            let rel = (err / norm).sqrt();
            assert!(rel < 5e-2, "{ko:?}: rel L2 {rel}");

            for isa in tiers_under_test("forward_is_the_composed_chain") {
                let mut got = vec![0f32; tokens * hidden];
                forward(isa, &e, &act, &mut got);
                assert_eq!(bits(&got), bits(&out), "{ko:?} {isa:?}");
            }
        }
    }
}
