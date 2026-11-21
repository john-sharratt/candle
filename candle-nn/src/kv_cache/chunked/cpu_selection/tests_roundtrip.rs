//! Tests that each quantizing format round-trips a ramp within a loose error bound,
//! and that the formats the search may pick are all round-trippable.
//!
//! The per-format bounds are ceilings chosen to catch wiring errors in the dispatch
//! and do not assert the codec's quality. R16 is the only format without a round
//! trip, because it carries query values in the field a round trip would lose.
//! The production candidates for the C5 level are tested to round-trip as well,
//! since the search can choose any of them.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    // These fixtures fill and compare fixed-size K/V/assignment arrays by index
    // on both sides, mirroring the kernel's own indexing so a divergence reads
    // as an index. Iterator rewrites would hide exactly what is being asserted.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use crate::kv_cache::chunked::sampled_selection::params::production_adaptive_candidates;

    use crate::kv_cache::KvFormat;

    pub(in super::super) fn quant_from_kv(kv: KvFormat) -> QuantFormat {
        match kv {
            KvFormat::Quantized(q) => q,
            KvFormat::Float(_) => panic!("expected quantized format"),
        }
    }

    #[test]
    fn roundtrip_quantize_dispatch_q4_0() {
        let mut src = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            src[i] = (i as f32 - 16.0) * 0.1;
        }
        let out = roundtrip_block(QuantFormat::Q4_0, &src, 1.0, false).unwrap();
        for i in 0..CHUNK_SIZE {
            let e = (src[i] - out[i]).abs();
            assert!(e < 0.25, "Q4_0 element {i} error {e} too large");
        }
    }

    #[test]
    fn roundtrip_quantize_dispatch_q8_0() {
        let mut src = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            src[i] = (i as f32 - 16.0) * 0.1;
        }
        let out = roundtrip_block(QuantFormat::Q8_0, &src, 1.0, false).unwrap();
        for i in 0..CHUNK_SIZE {
            let e = (src[i] - out[i]).abs();
            assert!(e < 0.02, "Q8_0 element {i} error {e} too large");
        }
    }

    pub(in super::super) fn ramp_src() -> [f32; CHUNK_SIZE] {
        let mut src = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            src[i] = (i as f32 - 16.0) * 0.05;
        }
        src
    }

    /// Per-format reconstruction-error tolerance. These are loose ceilings
    /// chosen to detect dispatch wiring bugs, not to assert codec quality.
    pub(in super::super) fn fmt_error_bound(fmt: QuantFormat) -> f32 {
        match fmt {
            QuantFormat::Q8_0 | QuantFormat::Q8_1 | QuantFormat::Q8_KS => 0.05,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 => 0.1,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::Q4_KS => 0.25,
            QuantFormat::Q3_0 | QuantFormat::Q3_1 => 0.4,
            QuantFormat::Q2_0 | QuantFormat::Q2_1 | QuantFormat::Q2_S | QuantFormat::Q2_A => 0.8,
            QuantFormat::Q1_S | QuantFormat::Q1_A => 1.6,
            QuantFormat::Q0
            | QuantFormat::Q0_V
            | QuantFormat::Q0_X
            | QuantFormat::Q0_M2
            | QuantFormat::Q0_M4 => 2.0,
            QuantFormat::R16 => 0.01,
        }
    }

    pub(in super::super) fn run_roundtrip(fmt: QuantFormat) {
        let src = ramp_src();
        let out = roundtrip_block(fmt, &src, 1.0, true)
            .unwrap_or_else(|| panic!("{:?} dispatch returned None", fmt));
        let bound = fmt_error_bound(fmt);
        let mut max_err = 0.0f32;
        for i in 0..CHUNK_SIZE {
            let e = (src[i] - out[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < bound,
            "{:?} max error {} exceeds bound {}",
            fmt,
            max_err,
            bound
        );
    }

    #[test]
    fn roundtrip_q4_1() {
        run_roundtrip(QuantFormat::Q4_1);
    }

    #[test]
    fn roundtrip_q5_0() {
        run_roundtrip(QuantFormat::Q5_0);
    }

    #[test]
    fn roundtrip_q5_1() {
        run_roundtrip(QuantFormat::Q5_1);
    }

    #[test]
    fn roundtrip_q4_ks() {
        run_roundtrip(QuantFormat::Q4_KS);
    }

    #[test]
    fn roundtrip_q8_ks() {
        run_roundtrip(QuantFormat::Q8_KS);
    }

    #[test]
    fn roundtrip_q2_0() {
        run_roundtrip(QuantFormat::Q2_0);
    }

    #[test]
    fn roundtrip_q3_0() {
        run_roundtrip(QuantFormat::Q3_0);
    }

    #[test]
    fn roundtrip_q2_1() {
        run_roundtrip(QuantFormat::Q2_1);
    }

    #[test]
    fn roundtrip_q3_1() {
        run_roundtrip(QuantFormat::Q3_1);
    }

    #[test]
    fn roundtrip_q0() {
        run_roundtrip(QuantFormat::Q0);
    }

    #[test]
    fn roundtrip_q1_s() {
        run_roundtrip(QuantFormat::Q1_S);
    }

    #[test]
    fn roundtrip_q2_s() {
        run_roundtrip(QuantFormat::Q2_S);
    }

    #[test]
    fn roundtrip_q2_a() {
        run_roundtrip(QuantFormat::Q2_A);
    }

    #[test]
    fn roundtrip_q0_v() {
        run_roundtrip(QuantFormat::Q0_V);
    }

    #[test]
    fn roundtrip_q1_a() {
        run_roundtrip(QuantFormat::Q1_A);
    }

    #[test]
    fn roundtrip_q0_x() {
        run_roundtrip(QuantFormat::Q0_X);
    }

    #[test]
    fn roundtrip_q0_m2() {
        run_roundtrip(QuantFormat::Q0_M2);
    }

    #[test]
    fn roundtrip_q0_m4() {
        run_roundtrip(QuantFormat::Q0_M4);
    }

    #[test]
    fn roundtrip_q8_1() {
        run_roundtrip(QuantFormat::Q8_1);
    }

    #[test]
    fn roundtrip_unsupported_format_returns_none() {
        let src = ramp_src();
        // R16 carries Q activations in the `q[]` field; round-tripping
        // would lose them.
        assert!(roundtrip_block(QuantFormat::R16, &src, 1.0, true).is_none());
    }

    #[test]
    fn production_c5_candidates_all_round_trip() {
        let (k_cands, v_cands) = production_adaptive_candidates(5);
        let sides = k_cands
            .iter()
            .map(|kv| (kv, true))
            .chain(v_cands.iter().map(|kv| (kv, false)));
        for (kv, is_k) in sides {
            let q = quant_from_kv(*kv);
            let src = ramp_src();
            let out = roundtrip_block(q, &src, 1.0, is_k)
                .unwrap_or_else(|| panic!("C5 candidate {:?} round-trip None", q));
            let mut max_err = 0.0f32;
            for i in 0..CHUNK_SIZE {
                let e = (src[i] - out[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err < fmt_error_bound(q),
                "C5 {:?}: max_err {} >= bound {}",
                q,
                max_err,
                fmt_error_bound(q)
            );
        }
    }
}
