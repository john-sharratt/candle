//! Tests that the dtype enumeration stays complete, and that its codes and KO
//! twins stay in agreement with the code that routes them.
//!
//! A sweep over a list of every dtype is held exhaustive by a match the compiler
//! checks, so adding a variant forces the list to grow with it. Every file code
//! must round-trip, and the KO list must match the predicate the routing uses.
//! Each KO type size is checked against the bytes the quantizer emits, because a
//! disagreement shifts every tensor after it in a GGUF file. The int8 twin table
//! pins each source to its twin in both modes and rejects any twin that is wider.

use super::*;

#[cfg(test)]
pub(super) mod ggml_dtype_lock_tests {
    //! These tests pin the exact integer value for every `GgmlDType` variant.
    //! They must stay in lockstep with:
    //!   - The C++ `QType` enum in candle-kernels/src/quantized/block_compact.cuh
    //!     (locked in via `static_assert` in that file)
    //!   - The Rust `QType` enum in candle-kernels/src/quantized/api.rs (which
    //!     has its own matching `qtype_values_are_stable` test)
    //!   - `ArenaFormat::*` in candle-kernels/src/arena_table.cuh
    //!   - `SELECT_FMT_*` in candle-kernels/src/quantize/select_kv_format.cuh
    //!
    //! F32/F16/BF16 (0/1/2) are specific to GgmlDType — they do not appear in
    //! QType, which starts at R16=3.
    use super::{ko_quant, GgmlDType, Int8Mode};

    #[test]
    fn ggml_dtype_values_are_stable() {
        assert_eq!(GgmlDType::F32 as u32, 0);
        assert_eq!(GgmlDType::F16 as u32, 1);
        assert_eq!(GgmlDType::BF16 as u32, 2);
        assert_eq!(GgmlDType::R16 as u32, 3);
        assert_eq!(GgmlDType::P2 as u32, 4);
        assert_eq!(GgmlDType::QAWQ as u32, 5);
        assert_eq!(GgmlDType::QAWQ_G64 as u32, 6);
        assert_eq!(GgmlDType::Q8_0 as u32, 7);
        assert_eq!(GgmlDType::Q8_1 as u32, 8);
        assert_eq!(GgmlDType::Q8_K as u32, 9);
        assert_eq!(GgmlDType::Q8_KS as u32, 10);
        assert_eq!(GgmlDType::Q6_K as u32, 11);
        assert_eq!(GgmlDType::Q5_0 as u32, 12);
        assert_eq!(GgmlDType::Q5_1 as u32, 13);
        assert_eq!(GgmlDType::Q5_K as u32, 14);
        assert_eq!(GgmlDType::Q4_0 as u32, 15);
        assert_eq!(GgmlDType::Q4_1 as u32, 16);
        assert_eq!(GgmlDType::Q4_K as u32, 17);
        assert_eq!(GgmlDType::Q4_KS as u32, 18);
        assert_eq!(GgmlDType::Q3_0 as u32, 19);
        assert_eq!(GgmlDType::Q3_1 as u32, 20);
        assert_eq!(GgmlDType::Q3_K as u32, 21);
        assert_eq!(GgmlDType::Q2_0 as u32, 22);
        assert_eq!(GgmlDType::Q2_1 as u32, 23);
        assert_eq!(GgmlDType::Q2_K as u32, 24);
        assert_eq!(GgmlDType::Q2_S as u32, 25);
        assert_eq!(GgmlDType::Q2_A as u32, 26);
        assert_eq!(GgmlDType::Q1_S as u32, 27);
        assert_eq!(GgmlDType::Q0_V as u32, 28);
        assert_eq!(GgmlDType::Q1_A as u32, 29);
        assert_eq!(GgmlDType::Q0_X as u32, 30);
        assert_eq!(GgmlDType::Q0_M2 as u32, 31);
        assert_eq!(GgmlDType::Q0_M4 as u32, 32);
        assert_eq!(GgmlDType::Q0 as u32, 33);
        assert_eq!(GgmlDType::F8E4M3 as u32, 34);
        assert_eq!(GgmlDType::F8E5M2 as u32, 35);
        // KO byte-permuted twins — must match QTYPE_Q*_KO / QType::Q*_KO (45-52).
        assert_eq!(GgmlDType::Q4_KO as u32, 45);
        assert_eq!(GgmlDType::Q5_KO as u32, 46);
        assert_eq!(GgmlDType::Q6_KO as u32, 47);
        assert_eq!(GgmlDType::Q8_KO as u32, 48);
        assert_eq!(GgmlDType::MXFP4 as u32, 49);
        assert_eq!(GgmlDType::MXFP4_KO as u32, 50);
        assert_eq!(GgmlDType::Q2_KO as u32, 51);
        assert_eq!(GgmlDType::Q3_KO as u32, 52);
    }

    /// Every `GgmlDType`, for the sweeps below.
    ///
    /// Held exhaustive by [`all_dtypes_covers_the_enum`], whose match the compiler breaks when
    /// a variant is added — which is the signal to add it here. A sweep over a list that has
    /// quietly stopped covering the enum is worse than no sweep, because it still passes.
    pub(in super::super) const ALL_DTYPES: &[GgmlDType] = &[
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::R16,
        GgmlDType::P2,
        GgmlDType::QAWQ,
        GgmlDType::QAWQ_G64,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q8_K,
        GgmlDType::Q8_KS,
        GgmlDType::Q6_K,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q5_K,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q4_K,
        GgmlDType::Q4_KS,
        GgmlDType::Q3_0,
        GgmlDType::Q3_1,
        GgmlDType::Q3_K,
        GgmlDType::Q2_0,
        GgmlDType::Q2_1,
        GgmlDType::Q2_K,
        GgmlDType::Q2_S,
        GgmlDType::Q2_A,
        GgmlDType::Q1_S,
        GgmlDType::Q0_V,
        GgmlDType::Q1_A,
        GgmlDType::Q0_X,
        GgmlDType::Q0_M2,
        GgmlDType::Q0_M4,
        GgmlDType::Q0,
        GgmlDType::F8E4M3,
        GgmlDType::F8E5M2,
        GgmlDType::U8,
        GgmlDType::I8,
        GgmlDType::U16,
        GgmlDType::I16,
        GgmlDType::U32,
        GgmlDType::I32,
        GgmlDType::U64,
        GgmlDType::I64,
        GgmlDType::F64,
        GgmlDType::Q4_KO,
        GgmlDType::Q5_KO,
        GgmlDType::Q6_KO,
        GgmlDType::Q8_KO,
        GgmlDType::MXFP4,
        GgmlDType::MXFP4_KO,
        GgmlDType::Q2_KO,
        GgmlDType::Q3_KO,
    ];

    /// [`ALL_DTYPES`] lists every variant, once.
    ///
    /// The `match` below exists only to be exhaustive: adding a variant to `GgmlDType` fails to
    /// compile *here*, which is the prompt to add it to the list. The duplicate and count
    /// checks catch the other direction — a paste that repeated an entry instead of adding one.
    #[test]
    fn all_dtypes_covers_the_enum() {
        #[allow(clippy::match_like_matches_macro)]
        fn is_a_variant(d: GgmlDType) -> bool {
            match d {
                GgmlDType::F32
                | GgmlDType::F16
                | GgmlDType::BF16
                | GgmlDType::R16
                | GgmlDType::P2
                | GgmlDType::QAWQ
                | GgmlDType::QAWQ_G64
                | GgmlDType::Q8_0
                | GgmlDType::Q8_1
                | GgmlDType::Q8_K
                | GgmlDType::Q8_KS
                | GgmlDType::Q6_K
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q5_K
                | GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q4_K
                | GgmlDType::Q4_KS
                | GgmlDType::Q3_0
                | GgmlDType::Q3_1
                | GgmlDType::Q3_K
                | GgmlDType::Q2_0
                | GgmlDType::Q2_1
                | GgmlDType::Q2_K
                | GgmlDType::Q2_S
                | GgmlDType::Q2_A
                | GgmlDType::Q1_S
                | GgmlDType::Q0_V
                | GgmlDType::Q1_A
                | GgmlDType::Q0_X
                | GgmlDType::Q0_M2
                | GgmlDType::Q0_M4
                | GgmlDType::Q0
                | GgmlDType::F8E4M3
                | GgmlDType::F8E5M2
                | GgmlDType::U8
                | GgmlDType::I8
                | GgmlDType::U16
                | GgmlDType::I16
                | GgmlDType::U32
                | GgmlDType::I32
                | GgmlDType::U64
                | GgmlDType::I64
                | GgmlDType::F64
                | GgmlDType::Q4_KO
                | GgmlDType::Q5_KO
                | GgmlDType::Q6_KO
                | GgmlDType::Q8_KO
                | GgmlDType::MXFP4
                | GgmlDType::MXFP4_KO
                | GgmlDType::Q2_KO
                | GgmlDType::Q3_KO => true,
            }
        }
        let mut seen: Vec<u32> = ALL_DTYPES
            .iter()
            .map(|d| {
                assert!(is_a_variant(*d));
                *d as u32
            })
            .collect();
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n, "ALL_DTYPES has a duplicate");
        assert_eq!(n, 53, "ALL_DTYPES must list every variant exactly once");
    }

    /// Every dtype survives a GGUF file-code round trip.
    ///
    /// `to_gguf_file_code` and `from_gguf_file_code` are hand-written inverses over ~40 arms,
    /// and nothing forced them to stay inverses: the compiler checks each is exhaustive, not
    /// that they agree. A code added to one side only is a tensor that writes as one dtype and
    /// reads back as another — or, for the KO twins whose codes are reserved rather than used,
    /// a `from_gguf_file_code` that silently rejects a format the writer can emit.
    #[test]
    fn every_gguf_file_code_round_trips() {
        for &d in ALL_DTYPES {
            let code = d.to_gguf_file_code();
            let back = GgmlDType::from_gguf_file_code(code).unwrap_or_else(|e| {
                panic!("{d:?} writes code {code}, which does not read back: {e}")
            });
            assert_eq!(back, d, "{d:?} → {code} → {back:?}");
        }
    }

    /// [`ko_quant::ALL_KO_DTYPES`] is exactly the set [`GgmlDType::is_ko`] accepts.
    ///
    /// The sweeps that cover KO formats iterate the list; the code that routes them tests the
    /// predicate. If those two ever disagree, every sweep silently stops covering a format
    /// while still passing — which is the failure this whole family of tests exists to end.
    #[test]
    fn ko_dtype_list_matches_is_ko() {
        let from_predicate: Vec<GgmlDType> =
            ALL_DTYPES.iter().copied().filter(|d| d.is_ko()).collect();
        let mut listed = ko_quant::ALL_KO_DTYPES.to_vec();
        let mut derived = from_predicate.clone();
        listed.sort_by_key(|d| *d as u32);
        derived.sort_by_key(|d| *d as u32);
        assert_eq!(
            listed, derived,
            "ALL_KO_DTYPES and is_ko() disagree — a KO format is missing from one of them"
        );
    }

    /// `type_size` × blocks must equal the bytes `quantize_ko` actually emits.
    ///
    /// **This is an offset invariant, not a size preference.** `type_size` is what the GGUF
    /// header uses to lay out and slice tensor data — offset accounting on write, length on
    /// read — so a KO twin whose `type_size` disagrees with its chunk width does not produce a
    /// wrong tensor, it shifts every tensor after it and reads past the end of the file. The
    /// two numbers are written down in different files (`mod.rs` and `ko_quant.rs`), which is
    /// exactly the shape of thing that drifts.
    #[test]
    fn ko_type_size_matches_the_emitted_chunk() {
        for &d in ALL_DTYPES {
            if !d.is_ko() {
                continue;
            }
            assert_eq!(d.block_size(), 128, "{d:?}: KO twins are per-128");
            let chunk = if d == GgmlDType::MXFP4_KO {
                ko_quant::MXFP4_KO_GPU_CHUNK_BYTES
            } else {
                ko_quant::ko_chunk_bytes(d)
            };
            assert_ne!(chunk, 0, "{d:?} has no chunk width");
            // One chunk is 8 rows × 128 K = 8 blocks, so type_size is the chunk over 8.
            assert_eq!(
                d.type_size() * 8,
                chunk,
                "{d:?}: type_size {} × 8 must equal the {chunk}-byte chunk quantize_ko emits",
                d.type_size(),
            );
        }
    }

    /// The `Int8Mode` twin table, pinned end to end.
    ///
    /// **The mapping this pins is the one that went wrong.** `Q2_K` and `Q3_K` mapped to
    /// `Q4_KO` for as long as no narrower twin existed — a 71% and a 31% weight inflation that
    /// read as a deliberate choice and was really a missing format. Nothing failed; the model
    /// simply carried wider weights, which costs VRAM once when weights are resident and PCIe
    /// bytes on *every forward* once they stream. A table with no test is how that survives.
    #[test]
    fn the_int8_twin_table_is_same_width_then_one_step_up() {
        use Int8Mode::{Off, Performance, Precision};
        // (source, Performance twin, Precision twin)
        let table = [
            (GgmlDType::Q2_K, GgmlDType::Q2_KO, GgmlDType::Q3_KO),
            (GgmlDType::Q3_K, GgmlDType::Q3_KO, GgmlDType::Q4_KO),
            (GgmlDType::Q4_0, GgmlDType::Q4_KO, GgmlDType::Q5_KO),
            (GgmlDType::Q4_1, GgmlDType::Q4_KO, GgmlDType::Q5_KO),
            (GgmlDType::Q4_K, GgmlDType::Q4_KO, GgmlDType::Q5_KO),
            (GgmlDType::Q5_0, GgmlDType::Q5_KO, GgmlDType::Q6_KO),
            (GgmlDType::Q5_1, GgmlDType::Q5_KO, GgmlDType::Q6_KO),
            (GgmlDType::Q5_K, GgmlDType::Q5_KO, GgmlDType::Q6_KO),
            // At the top of the ladder there is no finer twin, so both modes agree.
            (GgmlDType::Q6_K, GgmlDType::Q6_KO, GgmlDType::Q6_KO),
            (GgmlDType::Q8_0, GgmlDType::Q8_KO, GgmlDType::Q8_KO),
            (GgmlDType::Q8_1, GgmlDType::Q8_KO, GgmlDType::Q8_KO),
            (GgmlDType::Q8_K, GgmlDType::Q8_KO, GgmlDType::Q8_KO),
            // MXFP4 keeps its native nibbles in both modes — an exact permutation, no requant.
            (GgmlDType::MXFP4, GgmlDType::MXFP4_KO, GgmlDType::MXFP4_KO),
            // A float source is widened band-by-band by `dequantize_f32_into` and quantized
            // straight into its twin, so it needs no intermediate and both modes agree on the
            // widest grid. A checkpoint that carves one tensor out at higher precision —
            // `…NEO-IMATRIX-MAX…` keeps `output.weight` at F16 — takes this row.
            (GgmlDType::F32, GgmlDType::Q8_KO, GgmlDType::Q8_KO),
            (GgmlDType::F16, GgmlDType::Q8_KO, GgmlDType::Q8_KO),
            (GgmlDType::BF16, GgmlDType::Q8_KO, GgmlDType::Q8_KO),
        ];
        for (src, perf, prec) in table {
            assert_eq!(src.to_ko(Performance).unwrap(), perf, "{src:?} Performance");
            assert_eq!(src.to_ko(Precision).unwrap(), prec, "{src:?} Precision");
            assert!(src.to_ko(Off).is_err(), "{src:?}: Off has no twin");
        }

        // Performance is same-width for every affine source — no exceptions left. Stated one
        // direction only: **the twin is never WIDER**.
        //
        // The other direction is not assertable from bits-per-weight, and trying cost a false
        // failure worth recording. `Q4_1` is 5.0 bpw and its `Q4_KO` twin 4.25, which looks
        // like three quarters of a bit thrown away — but `Q4_1` carries a scale *and* a min per
        // 32 values where the twin carries one pair per 128. Both store 4-bit values; the whole
        // gap is scale overhead, i.e. the per-32→per-128 collapse this mode exists to make.
        // bpw cannot separate value precision from scale overhead, so a lower bound on it
        // measures the wrong quantity.
        //
        // The upper bound has no such problem: scale overhead only ever *falls* across the
        // repack, so a twin that comes out wider can only mean its value width grew — which is
        // exactly a missing same-width twin. `Q3_K → Q4_KO` was +0.81 and `Q2_K → Q4_KO`
        // +1.63; both are caught, and an `abs() < 1.0` band would have admitted the first.
        let bpw = GgmlDType::bits_per_weight;
        for (src, perf, _) in table {
            if src == GgmlDType::MXFP4 {
                continue; // 4-bit either way, but its chunk carries per-sub scales
            }
            let (s, p) = (bpw(src), bpw(perf));
            assert!(
                p <= s + 0.01,
                "{src:?} ({s:.4} bpw) → {perf:?} ({p:.4} bpw): Performance widened the weight, \
                 which means no same-width twin exists for this source and `to_ko` rounded up. \
                 Under layer streaming that is PCIe bytes on every forward."
            );
        }

        // A KO twin is its own twin — repacking an already-repacked weight is a no-op, which is
        // what lets a prepared GGUF load without a second pass.
        for &d in ALL_DTYPES {
            if d.is_ko() {
                assert_eq!(d.to_ko(Performance).unwrap(), d, "{d:?} is already KO");
                assert_eq!(d.to_ko(Precision).unwrap(), d, "{d:?} is already KO");
            }
        }
    }
}
