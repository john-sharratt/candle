//! Which layers the pinned host tier holds.
//!
//! The warm tier is filled once at startup and never changes — same rule as the
//! expert cache's, and it licenses the same simplification: because no other
//! layer can ever claim a warm slot, a promoted layer's slot is not worth
//! reclaiming, so an eviction from VRAM lands it back on RAM by doing nothing.
//!
//! # Why the membership rule is not the expert cache's
//!
//! The expert warm tier takes a **stratified random** subset, and the reasoning
//! is explicitly about a popularity distribution: VRAM has already filtered the
//! head, what reaches RAM is the long tail, and a random sample of the tail
//! yields its own size as a hit rate.
//!
//! None of that transfers. Every streamed layer is read exactly once per
//! forward, so there is no popularity to sample and a random subset would be a
//! coin flip over layers that are all equally hot. The question is not *which
//! layers are hot* but **which layers are still streaming**.
//!
//! # The set is the head of the eviction order, and that is what makes it stable
//!
//! How many layers stream is not a constant. The zone opens small —
//! `INITIAL_KV_RESERVE` hands most of the span to the KV side at load — and grows
//! into spare KV ground over the first forwards; KV pressure later takes some
//! back. So the tier is drawn against a number that is about to change, and it
//! cannot be redrawn: it is gigabytes of pinned host memory filled by a pass over
//! the pack, which is not something to repeat in the gap between two forwards.
//!
//! It never needs to be, because the streamed set is a **prefix of
//! [`super::eviction_order`] at every capacity** — that is the nesting property
//! the order is built to have. The first `W` entries of that order are therefore
//! streaming under every capacity that streams at least `W` layers, and the
//! membership is correct before growth, during it, after it, and after a
//! concession, with no redraw and no policy.
//!
//! Measured on the 27B, which is what surfaced the original bug: drawn upward
//! from the load-time prefix the tier covered layers 5–55, and once the prefix
//! settled the streamed set was 16–63 — so **layers 55–63 fell to the cold tier
//! on every forward for the life of the process**, 9 synchronous NVMe reads a
//! forward, while the tier held 50 slots for 48 streamed layers. It was large
//! enough and aimed at a prefix that no longer existed. The rule that replaced it
//! drew from the top of the model, which was correct exactly while the resident
//! set was a *prefix of the layer order*; it stops being correct the moment the
//! resident set is spread, and this is its successor.
//!
//! # It is a set, not a run
//!
//! Spread residency means the layers that stream are spread too, so the fill is a
//! scatter over the pack rather than one sequential run. The members are still
//! handed over in ascending layer order so the reads walk the file forwards, but
//! they no longer walk it contiguously. That is a one-time startup cost paid to
//! put the right layers in the tier, which is the trade the measurement above
//! makes for itself.

use super::order::eviction_order;

/// Which layers the pinned host tier should hold, given room for `slots`.
///
/// The first `slots` layers of the eviction order — the ones given up soonest and
/// so streaming under every capacity — returned in ascending layer order so the
/// pack is read forwards.
///
/// The pinned head is absent from the eviction order and so from this: it has no
/// record in any tier, being uploaded straight from the checkpoint and never
/// loaded again, so a warm slot for one could never be read.
pub fn warm_membership(num_layers: usize, pinned: usize, slots: usize) -> Vec<usize> {
    let mut m: Vec<usize> = eviction_order(num_layers, pinned)
        .into_iter()
        .take(slots)
        .collect();
    m.sort_unstable();
    m
}

/// Warm slots worth allocating for a model of `num_layers`.
///
/// Never more than the streamable set: a tier larger than the layers it could
/// hold is pinned host RAM that can never be read, and pinned pages are the
/// scarcest thing on the host. `budget_slots` is what the host can afford — the
/// caller's measurement, not this module's guess.
pub fn warm_slots_for(num_layers: usize, pinned: usize, budget_slots: usize) -> usize {
    budget_slots.min(num_layers.saturating_sub(pinned))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **One membership is correct at every capacity the tier can reach.**
    ///
    /// The streamed set is a prefix of the eviction order, so a tier holding that
    /// order's first `W` entries covers the streamed set completely whenever at
    /// most `W` layers stream — whatever the capacity is, however it got there,
    /// and without a redraw. Below that it cannot cover the set by arithmetic:
    /// there are more streaming layers than slots, and no draw could.
    #[test]
    fn one_membership_covers_every_capacity_the_tier_can_reach() {
        const N: usize = 64;
        const W: usize = 50;
        let members = warm_membership(N, 2, W);
        let evict = eviction_order(N, 2);
        for streaming in 0..=W {
            for layer in &evict[..streaming] {
                assert!(
                    members.contains(layer),
                    "L{layer} streams at {streaming} and has no warm slot"
                );
            }
        }
    }

    /// A run drawn from the top of the *layer* order was correct only while the
    /// resident set was a prefix of that order. Under a spread residency it
    /// strands the layers that stream soonest — the ones the tier most needs.
    #[test]
    fn a_top_of_model_run_would_strand_the_first_layers_given_up() {
        const N: usize = 64;
        const W: usize = 20;
        let top: Vec<usize> = (N - W..N).collect();
        let evict = eviction_order(N, 2);
        // The first W layers given up are spread across the model, so a run at
        // the top misses most of them.
        let stranded: Vec<usize> = evict[..W]
            .iter()
            .copied()
            .filter(|l| !top.contains(l))
            .collect();
        assert!(
            stranded.len() > W / 2,
            "a top run should miss most of the spread set, missed {}",
            stranded.len()
        );
        // The eviction-order draw strands none of them.
        let m = warm_membership(N, 2, W);
        assert!(evict[..W].iter().all(|l| m.contains(l)));
    }

    /// A tier smaller than the streamed set cannot cover it, and the layers it
    /// gives up are the ones given up **last** — which are also the last to start
    /// streaming as capacity falls. The shortfall shrinks itself.
    #[test]
    fn a_short_tier_gives_up_the_layers_that_stream_last() {
        const N: usize = 64;
        let evict = eviction_order(N, 2);
        let members = warm_membership(N, 2, 20);
        for l in &evict[..20] {
            assert!(members.contains(l));
        }
        for l in &evict[20..] {
            assert!(!members.contains(l), "L{l} streams last but holds a slot");
        }
    }

    #[test]
    fn a_tier_larger_than_the_model_is_clipped() {
        assert_eq!(warm_membership(6, 2, 100), vec![2, 3, 4, 5]);
        assert_eq!(warm_slots_for(6, 2, 100), 4);
    }

    #[test]
    fn a_model_that_streams_nothing_warms_nothing() {
        // Every layer pinned — the case of a checkpoint that fits.
        assert!(warm_membership(2, 2, 8).is_empty());
        assert_eq!(warm_slots_for(2, 2, 8), 0);
        assert_eq!(warm_slots_for(64, 64, 8), 0);
    }

    #[test]
    fn a_zero_budget_warms_nothing() {
        assert!(warm_membership(64, 2, 0).is_empty());
        assert_eq!(warm_slots_for(64, 2, 0), 0);
    }

    /// Ascending, so the pack is read forwards — but no longer contiguous, which
    /// is the cost of holding the right set rather than a convenient one.
    #[test]
    fn membership_is_ascending_and_the_right_size() {
        let m = warm_membership(64, 2, 30);
        assert!(m.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(m.len(), 30);
        assert!(m.iter().all(|&l| l >= 2), "the pinned head has no record");
    }
}
