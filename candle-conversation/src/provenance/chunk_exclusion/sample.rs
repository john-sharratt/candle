//! The calibration questions an incremental exclusion learns from: the
//! [`SAMPLE`] whose stream's hash is smallest (bottom-k sampling,
//! `docs/incremental_chunk_exclusion.md` §4.1).
//!
//! Uniform over the questions, with no random state: every member computes the
//! same sample from the same substrate. It changes one question at a time — a
//! new question enters exactly when its hash is below the largest kept, which
//! then leaves.

use crate::persistence::streams::StreamId;

/// Questions sampled per group.
pub const SAMPLE: usize = 4_000;

/// A question's place in the sampling order: SplitMix64 of its stream id, a
/// fixed mix, so the order is the same on every member and every build.
pub fn sample_hash(sid: StreamId) -> u64 {
    let mut z = sid.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `k` of `questions` with the smallest [`sample_hash`], in that order
/// (ties by stream id).
pub fn bottom_k(questions: impl IntoIterator<Item = StreamId>, k: usize) -> Vec<StreamId> {
    let mut all: Vec<(u64, StreamId)> = questions
        .into_iter()
        .map(|sid| (sample_hash(sid), sid))
        .collect();
    all.sort_unstable_by_key(|&(h, sid)| (h, sid.0));
    all.dedup_by_key(|&mut (_, sid)| sid);
    all.into_iter().take(k).map(|(_, sid)| sid).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The mix is fixed**: these values are the sampling order every member
    /// and every build agrees on.
    #[test]
    fn the_mix_is_fixed() {
        assert_eq!(sample_hash(StreamId(0)), 0xE220_A839_7B1D_CDAF);
        assert_eq!(sample_hash(StreamId(1)), 0x910A_2DEC_8902_5CC1);
    }

    /// **The sample is the same from any order**, duplicates counted once.
    #[test]
    fn the_sample_is_the_same_from_any_order() {
        let ids: Vec<StreamId> = (0..50).map(StreamId).collect();
        let forward = bottom_k(ids.iter().copied(), 7);
        let backward = bottom_k(ids.iter().rev().copied().chain([StreamId(3)]), 7);
        assert_eq!(forward.len(), 7);
        assert_eq!(forward, backward);
        let mut by_hash: Vec<StreamId> = ids.clone();
        by_hash.sort_by_key(|&s| sample_hash(s));
        assert_eq!(forward, by_hash[..7]);
    }

    /// **One question in, the largest out**: a question hashing below the
    /// sample's largest enters it and that one leaves; one hashing above it
    /// changes nothing.
    #[test]
    fn one_question_in_the_largest_out() {
        let ids: Vec<StreamId> = (0..40).map(StreamId).collect();
        let before = bottom_k(ids.iter().copied(), 5);
        let largest = sample_hash(*before.last().unwrap());
        let below = (100..10_000)
            .map(StreamId)
            .find(|&s| sample_hash(s) < largest)
            .unwrap();
        let above = (100..10_000)
            .map(StreamId)
            .find(|&s| sample_hash(s) > largest)
            .unwrap();
        let after = bottom_k(ids.iter().copied().chain([below]), 5);
        assert!(after.contains(&below));
        assert!(!after.contains(before.last().unwrap()));
        assert_eq!(after.len(), 5);
        assert_eq!(bottom_k(ids.iter().copied().chain([above]), 5), before);
    }
}
