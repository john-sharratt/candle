//! Tests pinning the narrowing schedule to the table it was derived from.
//!
//! The schedule is the quality decision, so the whole table is written out and
//! any drift in an answer fails here. Roles the schedule does not name are
//! untouched, tensors outside the trunk's block namespace are never matched by
//! prefix, and the draft head keeps its own resident width. Every width the
//! schedule names must shrink its source, since one that does not would be
//! silently inert.

use super::*;

#[cfg(test)]
pub(super) mod narrowing_tests {
    use super::{streaming_twin, NARROW_STREAM_ENDS, NARROW_TWIN};
    use candle::quantized::GgmlDType;

    const N: usize = 64;

    fn twin(li: usize, role: &str) -> Option<GgmlDType> {
        streaming_twin(&format!("blk.{li}.{role}"), N)
    }

    /// The whole schedule, at the four positions where it changes answer.
    ///
    /// Written out rather than derived, because this table *is* the quality decision — a value
    /// that drifts here is a checkpoint quantized differently than it was measured, and nothing
    /// downstream would notice.
    #[test]
    fn the_schedule_is_the_table_it_was_derived_from() {
        // `ffn_down`: the checkpoint's Q5_K carve-out on blocks 0-3 stands; everything after it
        // goes to Q3_KO.
        for li in 0..4 {
            assert_eq!(twin(li, "ffn_down.weight"), None, "block {li}");
        }
        for li in [4, 5, 32, N - 1] {
            assert_eq!(
                twin(li, "ffn_down.weight"),
                Some(GgmlDType::Q3_KO),
                "block {li}"
            );
        }

        // `attn_qkv`: interior only — the first and last `NARROW_STREAM_ENDS` blocks keep theirs.
        for li in [0, NARROW_STREAM_ENDS - 1, N - NARROW_STREAM_ENDS, N - 1] {
            assert_eq!(twin(li, "attn_qkv.weight"), None, "block {li}");
        }
        for li in [NARROW_STREAM_ENDS, 32, N - NARROW_STREAM_ENDS - 1] {
            assert_eq!(
                twin(li, "attn_qkv.weight"),
                Some(GgmlDType::Q4_KO),
                "block {li}"
            );
        }

        // `attn_q`: every block, no carve-out.
        for li in [0, 1, 32, N - 1] {
            assert_eq!(
                twin(li, "attn_q.weight"),
                Some(GgmlDType::Q5_KO),
                "block {li}"
            );
        }
    }

    /// Roles the schedule does not name are left alone — including the ones sitting next to a
    /// role it does name, which is where a match arm typo lands.
    #[test]
    fn an_unnamed_role_is_untouched() {
        for role in [
            "ffn_gate.weight",
            "ffn_up.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ssm_out.weight",
            "attn_gate.weight",
            "attn_q_norm.weight",
        ] {
            assert_eq!(twin(32, role), None, "{role}");
        }
    }

    /// The draft head sits at `blk.{num_layers}` and is resident, not streamed. It is governed by
    /// `narrow_resident_twin`, so a schedule meant for the trunk must not reach it — the two
    /// disagree (the head takes `NARROW_TWIN` on every tensor, the trunk takes it on one role).
    #[test]
    fn the_draft_head_is_not_a_trunk_layer() {
        for role in ["ffn_down.weight", "attn_q.weight", "attn_qkv.weight"] {
            assert_eq!(
                streaming_twin(&format!("blk.{N}.{role}"), N),
                None,
                "{role}"
            );
            assert_eq!(
                streaming_twin(&format!("blk.{}.{role}", N + 3), N),
                None,
                "{role}"
            );
        }
        // The head's own answer is the resident one, and it is a different width.
        assert_eq!(NARROW_TWIN, GgmlDType::Q3_KO);
    }

    /// Tensors outside the `blk.{n}.` namespace — `output.weight`, `token_embd.weight`, the
    /// final norm — are not layers and must not be matched by prefix accident.
    #[test]
    fn a_non_layer_tensor_is_not_scheduled() {
        for name in [
            "output.weight",
            "token_embd.weight",
            "output_norm.weight",
            "blk.attn_q.weight",
            "blkX.4.ffn_down.weight",
        ] {
            assert_eq!(streaming_twin(name, N), None, "{name}");
        }
    }

    /// Every width the schedule names is a KO twin, and narrower than the source it replaces.
    ///
    /// `Loader::proj` and `slot_form` both clamp a request that is not narrower — so a schedule
    /// entry that never shrinks anything would be silently inert rather than an error.
    #[test]
    fn every_scheduled_width_can_actually_shrink_its_source() {
        use candle::quantized::Int8Mode;
        // (role, the source dtype this checkpoint carries there)
        for (role, src) in [
            ("ffn_down.weight", GgmlDType::Q4_K),
            ("attn_qkv.weight", GgmlDType::Q5_K),
            ("attn_q.weight", GgmlDType::Q6_K),
        ] {
            let n = twin(32, role).unwrap_or_else(|| panic!("{role} is scheduled"));
            assert!(n.is_ko(), "{role}: {n:?} is not a KO twin");
            let picked = src.to_ko(Int8Mode::Precision).unwrap();
            assert!(
                n.bits_per_weight() < picked.bits_per_weight(),
                "{role}: {n:?} does not shrink {picked:?}"
            );
        }
    }
}
