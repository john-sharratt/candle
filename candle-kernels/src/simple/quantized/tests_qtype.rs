//! Tests that the activation and weight quant type values stay fixed, since the kernels and the Rust side
//! read them as the same numbers.
//!
//! The QType values are locked because the CUDA side and the Rust side share them. The q8a128 activation and
//! the lane-major KO weight twins are GPU-only layouts, and their numbers must match the kernel's.

use super::*;

#[cfg(test)]
pub(super) mod kv_qtype_lock_tests {
    //! Pin the integer value of every variant in this KV-side `QType`.
    //! See the C++ `static_assert` block in `block_compact.cuh` and the
    //! Rust test `ggml_dtype_values_are_stable` in
    //! `candle-core/src/quantized/mod.rs` — together these form a three-
    //! way lock between C++ / Rust-KV / Rust-GgmlDType.
    use super::QType;

    #[test]
    fn kv_qtype_values_are_stable() {
        assert_eq!(QType::R16 as i32, 3);
        assert_eq!(QType::P2 as i32, 4);
        assert_eq!(QType::QAWQ as i32, 5);
        assert_eq!(QType::QAWQ_G64 as i32, 6);
        assert_eq!(QType::Q8_0 as i32, 7);
        assert_eq!(QType::Q8_1 as i32, 8);
        assert_eq!(QType::Q8_K as i32, 9);
        assert_eq!(QType::Q8_KS as i32, 10);
        assert_eq!(QType::Q6_K as i32, 11);
        assert_eq!(QType::Q5_0 as i32, 12);
        assert_eq!(QType::Q5_1 as i32, 13);
        assert_eq!(QType::Q5_K as i32, 14);
        assert_eq!(QType::Q4_0 as i32, 15);
        assert_eq!(QType::Q4_1 as i32, 16);
        assert_eq!(QType::Q4_K as i32, 17);
        assert_eq!(QType::Q4_KS as i32, 18);
        assert_eq!(QType::Q3_0 as i32, 19);
        assert_eq!(QType::Q3_1 as i32, 20);
        assert_eq!(QType::Q3_K as i32, 21);
        assert_eq!(QType::Q2_0 as i32, 22);
        assert_eq!(QType::Q2_1 as i32, 23);
        assert_eq!(QType::Q2_K as i32, 24);
        assert_eq!(QType::Q2_S as i32, 25);
        assert_eq!(QType::Q2_A as i32, 26);
        assert_eq!(QType::Q1_S as i32, 27);
        assert_eq!(QType::Q0_V as i32, 28);
        assert_eq!(QType::Q1_A as i32, 29);
        assert_eq!(QType::Q0_X as i32, 30);
        assert_eq!(QType::Q0_M2 as i32, 31);
        assert_eq!(QType::Q0_M4 as i32, 32);
        assert_eq!(QType::Q0 as i32, 33);
        assert_eq!(QType::F8E4M3 as i32, 34);
        assert_eq!(QType::F8E5M2 as i32, 35);
        // Kernel-only activation QType (no GgmlDType counterpart).
        assert_eq!(QType::Q8A128V as i32, 36);
        assert_eq!(QType::Q8A128X as i32, 37);
        // KO byte-permuted twins — mirror QTYPE_Q*_KO / GgmlDType::Q*_KO.
        assert_eq!(QType::Q4_KO as i32, 45);
        assert_eq!(QType::Q5_KO as i32, 46);
        assert_eq!(QType::Q6_KO as i32, 47);
        assert_eq!(QType::Q8_KO as i32, 48);
        assert_eq!(QType::MXFP4_KO as i32, 50);
        assert_eq!(QType::Q2_KO as i32, 51);
        assert_eq!(QType::Q3_KO as i32, 52);
    }
}
