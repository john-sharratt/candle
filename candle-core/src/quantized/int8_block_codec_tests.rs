//! The host codecs of the INT8-parameter KV blocks — Q1_S, Q2_S, Q2_A — against
//! the bytes the KV kernels write (`quantize_block_q1_s` / `_q2_s` / `_q2_a`
//! under `candle-kernels/src/quantize/`). A host decode reads the arena's own
//! bytes, so its encoding has to be the kernel's to the bit.
//!
//! Inputs are chosen to land on `.5` before rounding, where the kernels'
//! `__float2int_rn` (half to even) and a naive `round` (half away) disagree.

use super::k_quants::{BlockQ1S, BlockQ2A, BlockQ2S, GgmlType};

/// The kernels' `1.0f / 127.0f`: a byte decodes as `byte · UNIT`.
const UNIT: f32 = 1.0 / 127.0;

/// A 32-element block from a repeating pattern.
fn block(pattern: &[f32]) -> Vec<f32> {
    pattern.iter().copied().cycle().take(32).collect()
}

/// Q1_S stores mean |x| as `round(mean · 127)`: 0.5 → 63.5 → 64 (even).
#[test]
fn q1_s_encodes_the_mean_magnitude_as_an_int8_byte() {
    let xs = block(&[0.5, -0.5, 0.5, 0.5]);
    let mut ys = vec![BlockQ1S::zeros(); 1];
    BlockQ1S::from_float(&xs, &mut ys);
    assert_eq!(ys[0].scale, 64);
    // Signs, low bit first: + - + + per nibble.
    assert_eq!(ys[0].qs, [0b1101_1101; 4]);

    let mut back = vec![0f32; 32];
    BlockQ1S::to_float(&ys, &mut back);
    assert_eq!(back[0], 64.0 * UNIT);
    assert_eq!(back[1], -64.0 * UNIT);
}

/// Q2_S stores amax/1.5 as `round(d · 127)` and quants against the rounded d.
#[test]
fn q2_s_encodes_its_scale_as_an_int8_byte() {
    let xs = block(&[1.5, -1.5, 0.5, -0.5]);
    let mut ys = vec![BlockQ2S::zeros(); 1];
    BlockQ2S::from_float(&xs, &mut ys);
    assert_eq!(ys[0].scale, 127);
    // q = round(x / d + 1.5) with d = 127/127 = 1: 3, 0, 2, 1.
    assert_eq!(ys[0].qs, [0b01_10_00_11; 8]);

    let mut back = vec![0f32; 32];
    BlockQ2S::to_float(&ys, &mut back);
    assert_eq!(&back[..4], &[1.5, -1.5, 0.5, -0.5]);
}

/// Q2_A stores range/3 and the minimum as INT8 bytes: 1.5/3 · 127 = 63.5 → 64,
/// and −0.5 · 127 = −63.5 → −64, both half to even.
#[test]
fn q2_a_encodes_scale_and_bias_as_int8_bytes() {
    let xs = block(&[-0.5, 1.0, 0.0, 0.5]);
    let mut ys = vec![BlockQ2A::zeros(); 1];
    BlockQ2A::from_float(&xs, &mut ys);
    assert_eq!(ys[0].scale, 64);
    assert_eq!(ys[0].bias, -64);
    // q = round((x − m) / d) with d = 64/127, m = −64/127:
    // −0.5 → 0, 1.0 → round(2.98) = 3, 0.0 → round(0.99) = 1, 0.5 → round(1.98) = 2.
    assert_eq!(ys[0].qs, [0b10_01_11_00; 8]);

    let mut back = vec![0f32; 32];
    BlockQ2A::to_float(&ys, &mut back);
    let (d, m) = (64.0 * UNIT, -64.0 * UNIT);
    // `block_q2_a.cuh`'s `d * q + m`.
    assert_eq!(back[0], d * 0.0 + m);
    assert_eq!(back[1], d * 3.0 + m);
}

/// An all-zero block encodes to zero scale and decodes to zeros, not NaN.
#[test]
fn a_zero_block_round_trips_to_zero() {
    let xs = vec![0f32; 32];
    let mut q2s = vec![BlockQ2S::zeros(); 1];
    BlockQ2S::from_float(&xs, &mut q2s);
    assert_eq!(q2s[0].scale, 0);
    let mut back = vec![1f32; 32];
    BlockQ2S::to_float(&q2s, &mut back);
    assert!(back.iter().all(|&x| x == 0.0), "{back:?}");
}
