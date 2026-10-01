//! Q0_V's curve codebooks are generated, not free-form.
//!
//! Each side's 128 curves are 8 buckets of 16 phases. Bucket `b`'s phase `p` is
//! the bucket's base curve rotated left by `2p` elements, and buckets 4–7 are
//! the exact negations of buckets 0–3:
//!
//! ```text
//! curve[b·16 + p][e] = ±base[b & 3][(e + 2p) mod 32]      (− for b ≥ 4)
//! ```
//!
//! The GPU decoder relies on this: it stores only the four base curves and
//! derives every other curve from the code's bucket and phase bits. These tests
//! pin the structure on the reference tables, so a recalibrated codebook that
//! breaks it fails here rather than decoding wrong numbers on the device.

use candle_core::quantized::k_quants::q0_v_tables::{CURVE_TABLE_K, CURVE_TABLE_V};

const ELEMS: usize = 32;
const PHASES: usize = 16;
const SLOTS: usize = 128;

fn assert_rotated_signed_bases(side: &str, table: &[[i8; ELEMS]; SLOTS]) {
    for slot in 0..SLOTS {
        let bucket = slot / PHASES;
        let phase = slot % PHASES;
        let base = &table[(bucket & 3) * PHASES];
        let sign: i16 = if bucket >= 4 { -1 } else { 1 };
        for e in 0..ELEMS {
            let want = sign * base[(e + 2 * phase) % ELEMS] as i16;
            assert_eq!(
                table[slot][e] as i16, want,
                "{side} slot {slot} (bucket {bucket}, phase {phase}) element {e}"
            );
        }
    }
}

/// Negating a base value must stay in i8 range, which it does only if no base
/// holds −128.
fn assert_negatable(side: &str, table: &[[i8; ELEMS]; SLOTS]) {
    for b in 0..4 {
        for (e, &v) in table[b * PHASES].iter().enumerate() {
            assert_ne!(v, i8::MIN, "{side} base {b} element {e} is -128");
        }
    }
}

#[test]
fn k_curves_are_rotated_signed_bases() {
    assert_negatable("K", &CURVE_TABLE_K);
    assert_rotated_signed_bases("K", &CURVE_TABLE_K);
}

#[test]
fn v_curves_are_rotated_signed_bases() {
    assert_negatable("V", &CURVE_TABLE_V);
    assert_rotated_signed_bases("V", &CURVE_TABLE_V);
}
