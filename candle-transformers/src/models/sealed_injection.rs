//! What injecting a sealed piece onto a slot's tail actually placed.
//!
//! A sealed piece whose layers disagree on their chunk windows is cut to the
//! common prefix before it enters the slot ([`reconcile_sealed_prefix`]), so
//! the slot can receive fewer tokens than the stored piece holds. The session
//! reports that ([`SealedInjection`]) rather than logging it, because only the
//! caller knows which stored unit it injected — and so which record to repair,
//! and how many positions the slot really advanced.

use candle_nn::kv_cache::SealedSequence;

/// The outcome of `BatchedInferenceSession::inject_sealed_at_tail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedInjection {
    /// `(block_start, block_end)` of the appended chunks, read from layer 0.
    pub range: (usize, usize),
    /// Tokens the slot advanced by.
    pub tokens: usize,
    /// Chunks cut from the longest layer to reach the common prefix; `0` when
    /// the piece's layers already agreed.
    pub cut_chunks: usize,
    /// Tokens the longest layer stored beyond [`Self::tokens`]. Zero when the
    /// chunks cut held none — a layer carrying a trailing empty chunk the
    /// others lack, which no position depends on.
    pub cut_tokens: usize,
}

impl SealedInjection {
    /// Whether the cut lost tokens: the slot holds fewer positions than the
    /// stored piece describes, so anything describing the whole piece — its
    /// index pages — no longer lines up with it. A cut of empty chunks only
    /// is not one.
    pub fn was_cut(&self) -> bool {
        self.cut_tokens > 0
    }
}

/// Cut every layer's sealed chunks back to the longest prefix on which all
/// layers agree `(offset, token_count)`, returning the reconciled per-layer
/// sequences — or `None` when the layers already agree chunk-for-chunk.
///
/// **The recovery twin of `batched_inference::assert_sealed_layers_aligned`.**
/// That guard refuses to *seal* a per-layer skew; this repairs one that is
/// being *injected*. A windowed-resume prefill commits a filled chunk on the
/// layers it has resumed while the rest still hold an empty one (see
/// `BatchedInferenceSession::reserve_glue_gap`); if such a section is sealed
/// and later re-injected verbatim, the skew desyncs the slot per layer, and
/// `unify_decode_layout` cannot heal it once appended turns push it into
/// mid-history (deep, Arc-shared) — every decode then diverges and the
/// sequence wedges. The chunks dropped here are exactly that undelivered creep
/// surplus, so cutting to the common prefix is the same safe truncation
/// `heal_tail_divergence` performs, moved to the inject boundary where the skew
/// would otherwise enter a live slot.
///
/// Also applied per piece before pieces are concatenated into one injection:
/// a skewed piece joined to the pieces after it misaligns all of them, and a
/// cut of the joined whole would then take every later piece with it.
pub fn reconcile_sealed_prefix(per_layer: &[SealedSequence]) -> Option<Vec<SealedSequence>> {
    let first = per_layer.first()?;
    let min_len = per_layer.iter().map(|s| s.chunks.len()).min().unwrap_or(0);
    let window = |s: &SealedSequence, i: usize| (s.chunks[i].offset, s.chunks[i].token_count);
    let mut common = min_len;
    for i in 0..min_len {
        let w = window(first, i);
        if per_layer.iter().any(|s| window(s, i) != w) {
            common = i;
            break;
        }
    }
    // Aligned: every layer already ends exactly at the common prefix, so there
    // is no skew to cut and the source can be injected as-is.
    if per_layer.iter().all(|s| s.chunks.len() == common) {
        return None;
    }
    Some(
        per_layer
            .iter()
            .map(|s| {
                let mut t = s.clone();
                t.chunks.truncate(common);
                t.token_count = t.chunks.iter().map(|c| c.token_count as usize).sum();
                t
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::kv_cache::{ArenaLocation, HeadGids, SealedChunk};
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

    fn windows(s: &SealedSequence) -> Vec<(u16, u16)> {
        s.chunks.iter().map(|c| (c.offset, c.token_count)).collect()
    }

    /// Aligned layers need no reconciliation — the source injects unchanged.
    #[test]
    fn aligned_layers_are_left_untouched() {
        let uniform = layer(&[(0, 32), (0, 32), (0, 17)]);
        let per_layer: Vec<SealedSequence> = (0..13).map(|_| uniform.clone()).collect();
        assert!(reconcile_sealed_prefix(&per_layer).is_none());
    }

    /// **The recovery case.** A contiguous prefix of layers a filled 32-token
    /// chunk ahead of the rest is cut back to the common prefix, so every
    /// layer describes the same windows. The dropped chunk is the undelivered
    /// creep surplus.
    #[test]
    fn a_mid_sweep_filled_skew_is_cut_to_the_common_prefix() {
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

        let fixed = reconcile_sealed_prefix(&per_layer).expect("a real skew must reconcile");
        for s in &fixed {
            assert_eq!(windows(s), vec![(0, 32), (0, 32)]);
            assert_eq!(s.token_count, 64);
        }
    }

    /// A layer carrying one extra trailing chunk the others lack is a skew too,
    /// and is trimmed to the common length rather than injected as-is.
    #[test]
    fn a_longer_layer_is_trimmed() {
        let long = layer(&[(0, 32), (0, 32), (0, 9)]);
        let short = layer(&[(0, 32), (0, 32)]);
        let per_layer = vec![long, short.clone(), short];
        let fixed = reconcile_sealed_prefix(&per_layer).expect("length skew must reconcile");
        for s in &fixed {
            assert_eq!(windows(s), vec![(0, 32), (0, 32)]);
            assert_eq!(s.token_count, 64);
        }
    }

    /// No layers is no skew.
    #[test]
    fn no_layers_is_aligned() {
        assert!(reconcile_sealed_prefix(&[]).is_none());
    }

    /// Only lost tokens make a cut: trimming a trailing empty chunk one layer
    /// carried moves no position. Measured live: a turn reported `cut_chunks=1`
    /// with 153 tokens stored and 153 injected, and lost its index for it.
    #[test]
    fn only_a_cut_that_loses_tokens_counts_as_cut() {
        let whole = SealedInjection {
            range: (0, 3),
            tokens: 81,
            cut_chunks: 0,
            cut_tokens: 0,
        };
        assert!(!whole.was_cut());
        let empty_trimmed = SealedInjection {
            cut_chunks: 1,
            ..whole
        };
        assert!(!empty_trimmed.was_cut());
        assert!(SealedInjection {
            cut_chunks: 1,
            cut_tokens: 32,
            ..whole
        }
        .was_cut());
    }
}
