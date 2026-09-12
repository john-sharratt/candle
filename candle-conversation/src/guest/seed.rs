//! The seed a job samples with.
//!
//! One definition, because two would drift: the prose guest fell back to a
//! *constant* while the image guest drew from the clock, so unseeded prose
//! repeated itself and unseeded images did not. Both now come through here.
//!
//! A seed is also the only thing that makes a generation repeatable, so every
//! guest reports the one it used back to its caller. A draw nobody can recover
//! makes every good result a one-off.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// The seed a request will actually sample with: the caller's, or a fresh draw.
///
/// The counter is not decoration. Windows reads the system clock at ~100 ns
/// granularity and a drain serves its whole backlog back-to-back, so two jobs
/// can land inside one tick and read the same nanosecond — the same repetition
/// bug again, rarer and therefore harder to find. Mixing in a monotonic count
/// makes distinctness a property of this function rather than of the clock.
pub fn resolve_seed(asked: Option<u64>) -> u64 {
    static DRAWS: AtomicU64 = AtomicU64::new(0);
    asked.unwrap_or_else(|| {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5EED);
        now ^ DRAWS.fetch_add(1, Ordering::Relaxed).wrapping_mul(GOLDEN)
    })
}

/// The 64-bit golden-ratio odd constant, as used by SplitMix64.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// A seed split into independent streams.
///
/// A caller that wants several unrelated choices from one seed — which of N
/// names, which of M stations — must not take them from the raw value: `seed %
/// 24` and `seed % 12` are correlated, so the two choices move together and the
/// variety they were added for is not there. Each [`Seeded::pick`] advances the
/// state through SplitMix64 first, so consecutive draws are independent.
///
/// Deterministic in the seed, which is the whole point: the same seed must
/// rebuild the same prompt as well as replay the same sampling, or a
/// "reproduce this one" is only reproducing half of what made it.
pub struct Seeded(u64);

impl Seeded {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next value in the stream. SplitMix64's mixing function, which is
    /// what makes low bits of a poorly-distributed seed usable as a choice.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// One element of `items`, or `None` when there are none to pick from.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            return None;
        }
        items.get(self.next() as usize % items.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// **The bug this module exists for.** Two unseeded requests must not sample
    /// from the same state: the sampler is deterministic in its seed, so a
    /// constant fallback is a "Regenerate" button that cannot regenerate.
    #[test]
    fn an_unseeded_request_draws_a_fresh_seed_each_time() {
        let a = resolve_seed(None);
        let b = resolve_seed(None);
        assert_ne!(a, b, "two unseeded requests would produce identical output");
    }

    /// Distinct even when the clock does not move — which it does not, between
    /// two jobs of one drain on a 100 ns-granularity clock.
    #[test]
    fn a_thousand_draws_in_a_burst_are_all_distinct() {
        let seen: HashSet<u64> = (0..1000).map(|_| resolve_seed(None)).collect();
        assert_eq!(seen.len(), 1000, "the clock repeated and nothing caught it");
    }

    /// The other half: a caller that pins a seed gets exactly it, which is what
    /// makes a result somebody liked reproducible.
    #[test]
    fn a_pinned_seed_is_used_verbatim() {
        assert_eq!(resolve_seed(Some(7)), 7);
        assert_eq!(resolve_seed(Some(0)), 0, "zero is a seed, not an absence");
    }

    /// The same seed must rebuild the same choices, or "reproduce this one"
    /// reproduces the sampling and not the prompt it sampled against.
    #[test]
    fn one_seed_replays_one_sequence() {
        let items: Vec<u32> = (0..50).collect();
        let draw = |seed| {
            let mut s = Seeded::new(seed);
            (0..8).map(|_| *s.pick(&items).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(draw(12345), draw(12345));
        assert_ne!(draw(12345), draw(12346));
    }

    /// **Consecutive picks must be independent.** Taken straight off the seed,
    /// `seed % 24` and `seed % 12` move together — so two axes added for variety
    /// would vary as one, and the second would be doing nothing.
    #[test]
    fn consecutive_picks_do_not_move_together() {
        let a: Vec<u32> = (0..24).collect();
        let b: Vec<u32> = (0..12).collect();
        // Over many seeds, every (first, second) combination should appear —
        // correlated draws would cover only a diagonal slice of the 288.
        let mut pairs = HashSet::new();
        for seed in 0..20_000u64 {
            let mut s = Seeded::new(seed);
            pairs.insert((*s.pick(&a).unwrap(), *s.pick(&b).unwrap()));
        }
        assert_eq!(pairs.len(), 24 * 12, "the two axes are not independent");
    }

    #[test]
    fn picking_from_nothing_is_none() {
        let empty: [u8; 0] = [];
        assert!(Seeded::new(1).pick(&empty).is_none());
    }
}
