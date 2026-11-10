//! How many pending residences one hot→warm migrate group takes.
//!
//! A group's quantize needs destination slots in proportion to its size. When
//! a destination class has too few free slots, the quantize is refused rather
//! than allocating a new arena mid-wave, and the arenas owed for it are made at
//! the next gap between forwards. A class that is also a native hot class —
//! R16, the K fallback, which every prefill writes into as well — loses those
//! fresh arenas to the ingest's own allocations before the refused group
//! returns. A pass snapshots the same oldest residences each time, so a group
//! too big for what the class keeps free is refused at the head of the queue
//! forever, and nothing behind it moves.
//!
//! So a refused group is retried at half its size before it waits for a gap:
//! one residence needs a few hundred to a few thousand slots, which a class
//! almost always has free, and the backlog moves at whatever size fits. The cap
//! doubles back towards [`MAX_GROUP_RESIDENCES`] after each group that moves
//! (`docs/kv_drain_under_pressure.md` §2c).

/// The most pending residences one group takes: the bound that keeps the
/// migrate flight short enough for compaction to get in between groups.
pub const MAX_GROUP_RESIDENCES: usize = 8;

/// The group size the next snapshot takes, halved by each refusal and doubled
/// back by each group that moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupCap(usize);

impl GroupCap {
    /// A cap at [`MAX_GROUP_RESIDENCES`].
    pub fn new() -> Self {
        Self(MAX_GROUP_RESIDENCES)
    }

    /// The residences the next group takes.
    pub fn get(self) -> usize {
        self.0
    }

    /// A group was refused for want of destination slots. Halves the cap and
    /// answers whether it shrank — `false` once a single residence is refused,
    /// which only a gap between forwards can answer.
    pub fn refused(&mut self) -> bool {
        if self.0 <= 1 {
            return false;
        }
        self.0 /= 2;
        true
    }

    /// A group moved: double the cap back towards the maximum.
    pub fn moved(&mut self) {
        self.0 = (self.0 * 2).min(MAX_GROUP_RESIDENCES);
    }
}

impl Default for GroupCap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cap_starts_at_the_maximum() {
        assert_eq!(GroupCap::new().get(), 8);
    }

    #[test]
    fn each_refusal_halves_the_cap_down_to_one() {
        let mut cap = GroupCap::new();
        assert!(cap.refused());
        assert_eq!(cap.get(), 4);
        assert!(cap.refused());
        assert_eq!(cap.get(), 2);
        assert!(cap.refused());
        assert_eq!(cap.get(), 1);
    }

    #[test]
    fn a_single_residence_refused_does_not_shrink() {
        let mut cap = GroupCap::new();
        while cap.refused() {}
        assert_eq!(cap.get(), 1);
        assert!(!cap.refused(), "one residence is left to the gap");
        assert_eq!(cap.get(), 1);
    }

    #[test]
    fn a_group_that_moves_doubles_back_to_the_maximum() {
        let mut cap = GroupCap::new();
        while cap.refused() {}
        cap.moved();
        assert_eq!(cap.get(), 2);
        cap.moved();
        assert_eq!(cap.get(), 4);
        cap.moved();
        assert_eq!(cap.get(), 8);
        cap.moved();
        assert_eq!(cap.get(), 8, "never past the maximum");
    }
}
