//! Tests that the seal guard refuses a real layer skew and accepts the benign
//! trailing skew that reconciliation removes.
//!
//! Aligned layers seal normally. A mid-sweep skew is refused with the chunk index
//! and both sides named, and a layer missing the chunk is refused rather than
//! panicking. Dropping trailing empty chunks aligns a benign creep skew, while a
//! filled chunk on the head's layer survives the drop and is refused. Each test
//! builds chunks with only the window geometry set, since that is all the check
//! reads.

use super::*;

#[cfg(test)]
pub(super) mod seal_alignment_tests {
    use super::assert_sealed_layers_aligned;
    use crate::models::sealed_injection::reconcile_sealed_prefix;
    use candle_nn::kv_cache::{ArenaLocation, HeadGids, SealedChunk, SealedSequence};
    use std::sync::Arc;

    /// One sealed chunk with the given window geometry; every other field is
    /// legitimately per-layer and plays no part in the check.
    fn chunk(offset: u16, token_count: u16) -> SealedChunk {
        SealedChunk {
            gids: HeadGids::from_vec(Vec::new()),
            offset,
            token_count,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(Vec::new()),
            v_fmt: Arc::new(Vec::new()),
            byte_size: 0,
            meta: None,
            q_sign: None,
        }
    }

    fn layer(windows: &[(u16, u16)]) -> SealedSequence {
        SealedSequence {
            chunks: windows.iter().map(|&(o, n)| chunk(o, n)).collect(),
            token_count: windows.iter().map(|&(_, n)| n as usize).sum(),
            chunk_size: 32,
            location: ArenaLocation::Cpu,
        }
    }

    /// Layers that agree seal normally — the overwhelmingly common case, and
    /// the one this must not start refusing.
    #[test]
    fn aligned_layers_seal() {
        let uniform = layer(&[(0, 32), (0, 32), (0, 17)]);
        let per_layer: Vec<SealedSequence> = (0..13).map(|_| uniform.clone()).collect();
        assert!(assert_sealed_layers_aligned(&per_layer, 4, "test").is_ok());
    }

    /// The production signature: a wave died mid-sweep, so a contiguous prefix
    /// of layers holds a full chunk the rest never got. Sealing this is what
    /// makes it permanent, so the seal must refuse.
    #[test]
    fn a_mid_sweep_skew_is_refused_and_named() {
        let ahead = layer(&[(0, 32), (0, 32), (0, 32)]);
        let behind = layer(&[(0, 32), (0, 32), (0, 0)]);
        let per_layer: Vec<SealedSequence> = (0..13)
            .map(|li| {
                if li < 3 {
                    ahead.clone()
                } else {
                    behind.clone()
                }
            })
            .collect();

        let err = assert_sealed_layers_aligned(&per_layer, 18, "test")
            .expect_err("a per-layer skew must not seal");
        let msg = err.to_string();
        // The chunk index and BOTH sides, so the report places the fault
        // instead of merely asserting one exists.
        assert!(msg.contains("chunk 2"), "{msg}");
        assert!(msg.contains("32 tokens"), "{msg}");
        assert!(msg.contains("0 tokens"), "{msg}");
        // The layer split is the diagnosis — a contiguous prefix means a
        // per-layer loop stopped early.
        assert!(msg.contains("[0, 1, 2]"), "{msg}");
    }

    /// A layer that is missing the chunk entirely (a shorter list, not just a
    /// different window) is caught too, and says so rather than panicking on
    /// the index.
    #[test]
    fn a_short_layer_is_refused() {
        let full = layer(&[(0, 32), (0, 32)]);
        let short = layer(&[(0, 32)]);
        let per_layer = vec![full.clone(), full, short];
        let err = assert_sealed_layers_aligned(&per_layer, 1, "test")
            .expect_err("a short layer must not seal");
        assert!(err.to_string().contains("no such chunk"), "{err}");
    }

    /// No layers at all is not a skew — an empty snapshot seals trivially.
    #[test]
    fn an_empty_snapshot_is_fine() {
        assert!(assert_sealed_layers_aligned(&[], 0, "test").is_ok());
    }

    /// A reconciled skew passes the seal guard: the two are each other's twin.
    #[test]
    fn reconciled_layers_pass_the_seal_guard() {
        let long = layer(&[(0, 32), (0, 32), (0, 9)]);
        let short = layer(&[(0, 32), (0, 32)]);
        let per_layer = vec![long.clone(), short.clone(), short];
        assert!(assert_sealed_layers_aligned(&per_layer, 0, "test").is_err());
        let fixed = reconcile_sealed_prefix(&per_layer).expect("length skew must reconcile");
        assert!(assert_sealed_layers_aligned(&fixed, 0, "test").is_ok());
    }

    /// **Dropping trailing empties aligns a benign creep skew.** A windowed creep
    /// leaves one layer with a trailing empty writer chunk the others do not yet
    /// have — the skew is only in the tail, and no token moves when it goes.
    /// After `drop_empty_tail` on every layer the layers describe the same token
    /// windows and the alignment check passes, rather than refusing a sequence
    /// that is actually fine.
    #[test]
    fn drop_empty_tail_reconciles_a_benign_trailing_writer_skew() {
        let ahead = layer(&[(0, 32), (0, 32), (0, 0)]);
        let behind = layer(&[(0, 32), (0, 32)]);
        let mut per_layer = vec![ahead, behind.clone(), behind];
        // Before reconciliation the trailing empty makes them diverge.
        assert!(
            assert_sealed_layers_aligned(&per_layer, 7, "test").is_err(),
            "an un-reconciled trailing empty reads as a skew"
        );
        for s in &mut per_layer {
            s.drop_empty_tail();
        }
        assert!(
            assert_sealed_layers_aligned(&per_layer, 7, "test").is_ok(),
            "dropping the trailing empty aligns the layers"
        );
    }

    /// **A real head skew survives `drop_empty_tail` and the drain refuses it.**
    /// When the MTP head's layer is a *filled* chunk off the trunk (a wave that
    /// died mid-sweep, not a trailing writer), the drop cannot remove it — the
    /// chunk holds tokens — so the alignment check fires and the drain refuses to
    /// seal the skew to the warm tier. This is the corruption that used to reach
    /// the substrate unchecked on this path.
    #[test]
    fn a_real_head_skew_survives_drop_empty_tail_and_is_refused() {
        let mut head = layer(&[(0, 32), (0, 32), (0, 32)]);
        let mut trunk = layer(&[(0, 32), (0, 32)]);
        head.drop_empty_tail();
        trunk.drop_empty_tail();
        let mut per_layer: Vec<SealedSequence> = (0..10).map(|_| trunk.clone()).collect();
        per_layer.push(head);
        let err = assert_sealed_layers_aligned(&per_layer, 9, "test")
            .expect_err("a filled-chunk head skew must not drain");
        assert!(err.to_string().contains("chunk 2"), "{err}");
        assert!(err.to_string().contains("[10]"), "{err}");
    }
}
