//! Tests for the recurrent-gate guard and the donor rules that repair it.
//!
//! The guard accepts every gate format a working checkpoint has been seen to use,
//! including Q8_0 as well as F32, and refuses the Q6_K gates that shipped broken.
//! A donor is admitted per tensor, on evidence: the same name, a width that clears
//! the rule, and the same shape. A healthy checkpoint asks nothing of a donor, and
//! a repair covers the output projection as well as the gates.

use super::*;

#[cfg(test)]
pub(super) mod recurrent_precision_tests {
    use super::{check_recurrent_precision, RECURRENT_GATES};
    use candle::quantized::gguf_file::{Content, TensorInfo};
    use candle::quantized::GgmlDType;
    use candle::Shape;
    use std::collections::HashMap;

    /// A header carrying one gate at `dtype` and an ordinary weight beside it.
    fn header(gate: GgmlDType) -> Content {
        let mut tensor_infos = HashMap::new();
        let mut put = |name: &str, ggml_dtype| {
            tensor_infos.insert(
                name.to_string(),
                TensorInfo {
                    ggml_dtype,
                    shape: Shape::from((32usize, 1024usize)),
                    offset: 0,
                },
            );
        };
        put("blk.0.ssm_alpha.weight", gate);
        put("blk.0.ssm_beta.weight", gate);
        // Quantized on purpose: the guard must object to the gates and to nothing else.
        put("blk.0.attn_q.weight", GgmlDType::Q6_K);
        put("blk.0.ssm_out.weight", GgmlDType::Q8_0);
        Content {
            magic: candle::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos,
            tensor_data_offset: 0,
        }
    }

    /// **Every format a checkpoint that works has been seen to use.**
    ///
    /// `F32` is `QWEN35_9B`'s choice and `Q8_0` is `QWEN35_0_8B`'s — and the 0.8B is why this
    /// test exists in this shape. The rule was written as "must be F32" from the 9B alone,
    /// which refused the 0.8B at load for storing gates the way it always has;
    /// `test_parallel_batched_forwarding_0_8b` failed on a model that had never stopped
    /// working. A guard that refuses those is a worse bug than the one it catches.
    #[test]
    fn gates_a_working_checkpoint_uses_are_accepted() {
        for dtype in [GgmlDType::F32, GgmlDType::Q8_0] {
            assert!(
                check_recurrent_precision(&header(dtype)).is_ok(),
                "{dtype:?} gates are used by a checkpoint known to generate correctly"
            );
        }
    }

    /// **The format that actually shipped broken.** `Q6_K` is what HauhauCS and DavidAU store
    /// their gates as, and both loaded silently and then generated incoherent text from the
    /// first token — HauhauCS generates correctly once the gates are read from a donor.
    #[test]
    fn undersized_gates_are_refused_with_the_tensor_named() {
        for dtype in [GgmlDType::Q6_K, GgmlDType::Q4_K, GgmlDType::Q5_K] {
            let e = check_recurrent_precision(&header(dtype))
                .expect_err(&format!("{dtype:?} gates must be refused"));
            let msg = e.to_string();
            assert!(
                msg.contains("blk.0.ssm_alpha.weight"),
                "{dtype:?}: the message must name the offending tensor, got: {msg}"
            );
            assert!(
                msg.contains("gate donor") && msg.contains("Q8_0"),
                "{dtype:?}: the message must name both remedies, got: {msg}"
            );
        }
    }

    /// The guard reads whole tensor names, so a weight that merely *contains* a gate's name
    /// is not one. `ends_with` is the rule; this is what would break if it became `contains`.
    #[test]
    fn only_the_gates_are_guarded() {
        let mut c = header(GgmlDType::F32);
        let info = c.tensor_infos["blk.0.attn_q.weight"].clone();
        c.tensor_infos
            .insert("blk.0.ssm_alpha.weight.scales".into(), info);
        assert!(check_recurrent_precision(&c).is_ok());
    }

    #[test]
    fn the_guarded_set_is_the_two_gates() {
        assert_eq!(RECURRENT_GATES, ["ssm_alpha.weight", "ssm_beta.weight"]);
    }

    /// The message points at the repair, not only at the defect.
    #[test]
    fn the_refusal_names_the_donor_as_a_remedy() {
        let e = check_recurrent_precision(&header(GgmlDType::Q6_K)).unwrap_err();
        assert!(
            e.to_string().contains("gate donor"),
            "an operator who has the base checkpoint should be told it is usable: {e}"
        );
    }

    /// **A donor is admitted per tensor, on evidence.** Substituting a weight from another
    /// file is only sound when it is the same weight, so each of these is a refusal rather
    /// than a silent mismatch — and each would otherwise load and be wrong in a way that
    /// reads as bad prose rather than as a fault.
    #[test]
    fn a_donor_is_refused_when_it_cannot_supply_the_gates() {
        let primary = header(GgmlDType::Q6_K);

        // A donor carrying the same defect at a different width: admitted by the format
        // filter, then refused by the gate rule. (At the *identical* width it is filtered out
        // earlier and supplies nothing — the same outcome by a shorter route.)
        let e = super::donor_gates(&primary, &header(GgmlDType::Q4_K)).unwrap_err();
        assert!(e.to_string().contains("same defect"), "{e}");

        // Missing the tensor entirely: a different architecture.
        let mut bare = header(GgmlDType::F32);
        bare.tensor_infos.remove("blk.0.ssm_alpha.weight");
        let e = super::donor_gates(&primary, &bare).unwrap_err();
        assert!(e.to_string().contains("has no"), "{e}");

        // Right name, wrong shape: a different model size.
        let mut wrong = header(GgmlDType::F32);
        let info = wrong
            .tensor_infos
            .get_mut("blk.0.ssm_alpha.weight")
            .unwrap();
        info.shape = candle::Shape::from((64usize, 1024usize));
        let e = super::donor_gates(&primary, &wrong).unwrap_err();
        assert!(e.to_string().contains("different models"), "{e}");
    }

    /// A checkpoint that needs no repair asks nothing of a donor, so a donor that could not
    /// have supplied anything is never consulted.
    #[test]
    fn a_healthy_checkpoint_reads_no_gates_from_a_donor() {
        let got = super::donor_gates(&header(GgmlDType::F32), &header(GgmlDType::Q6_K))
            .expect("nothing is undersized, so nothing is asked of the donor");
        assert!(got.is_empty());
    }

    /// **The whole recurrent path is repaired, not only the gates that failed the rule.**
    ///
    /// `ssm_out` breaks no precision rule — the stock conversion quantizes it too — but a
    /// recurrence assembled half from the base and half from a fine-tune that stored its half
    /// differently is a combination nobody has measured. Half-repairing it is exactly what
    /// this path did while `ssm_out` was still read by the layer store, and the model stayed
    /// incoherent until the other 24 tensors came from the donor too.
    #[test]
    fn a_repair_covers_the_output_projection_as_well_as_the_gates() {
        let mut primary = header(GgmlDType::Q6_K);
        // The reference's own choice for this tensor, which is not F32 and not the primary's.
        primary
            .tensor_infos
            .get_mut("blk.0.ssm_out.weight")
            .unwrap()
            .ggml_dtype = GgmlDType::Q6_K;
        let donor = header(GgmlDType::F32);
        let got = super::donor_gates(&primary, &donor).unwrap();
        assert!(
            got.iter().any(|n| n.ends_with("ssm_out.weight")),
            "the output projection must be repaired alongside the gates, got {got:?}"
        );
    }
}
