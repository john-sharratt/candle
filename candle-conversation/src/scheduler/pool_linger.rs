//! Pieces that left a slot's pool, kept on the model for a few refreshes.
//!
//! The working set chooses its pieces by strength within the card's tokens at
//! every observation (`docs/hit_driven_projection.md` §7.2 step 1), and the
//! pieces at the edge of that budget leave and come back from one reprojection
//! to the next. Every comeback was a whole new admission: its pages read
//! again, decoded and placed on the device, about 1.1 ms a megabyte. On
//! BlackWorld (2026-11-18) a refresh admitted 5–21 pieces and dropped 2–22 of
//! about 130, and placing the comebacks was most of the pool's 84 ms.
//!
//! A piece that leaves therefore lingers: its layout and the pages it holds on
//! the model stay for [`LINGER_REFRESHES`] refreshes, at most
//! [`MAX_LINGERING`] pieces, oldest out first. One that comes back within them
//! is taken up as if it had stayed, reading only the turns it gained. A
//! lingering piece is in no pool: nothing scans, declares or places it; only
//! its pages stay where they are. Which pieces the pool holds is the working
//! set's choice alone.

use std::collections::HashMap;

use candle_transformers::models::piece_key::PieceKey;

use crate::projection::{TimelineId, TurnIndex};

/// How many refreshes a piece that left the pool keeps its pages.
pub(super) const LINGER_REFRESHES: u32 = 8;

/// How many pieces linger at most.
pub(super) const MAX_LINGERING: usize = 48;

/// A pooled conversation's pages per turn, as the pool holds them.
pub(super) type Layout = HashMap<TurnIndex, Vec<(usize, PieceKey)>>;

/// One piece that left: its layout, how many of its turns were walked, and how
/// many refreshes it has lingered.
#[derive(Debug, Clone)]
struct Lingering {
    layout: Layout,
    walked: u32,
    age: u32,
}

/// See the module documentation.
#[derive(Debug, Clone, Default)]
pub(super) struct PoolLinger {
    held: HashMap<TimelineId, Lingering>,
}

impl PoolLinger {
    /// A refresh begins: every lingering piece is a refresh older, and one
    /// past [`LINGER_REFRESHES`] lets go.
    pub fn age(&mut self) {
        self.held.retain(|_, l| {
            l.age += 1;
            l.age <= LINGER_REFRESHES
        });
    }

    /// `timeline` left the pool holding `layout`, `walked` turns walked.
    /// Past [`MAX_LINGERING`] the piece that has lingered longest lets go.
    pub fn linger(&mut self, timeline: TimelineId, layout: Layout, walked: u32) {
        self.held.insert(
            timeline,
            Lingering {
                layout,
                walked,
                age: 0,
            },
        );
        while self.held.len() > MAX_LINGERING {
            let oldest = self
                .held
                .iter()
                .max_by_key(|(tl, l)| (l.age, std::cmp::Reverse(tl.raw())))
                .map(|(tl, _)| *tl);
            match oldest {
                Some(tl) => {
                    self.held.remove(&tl);
                }
                None => break,
            }
        }
    }

    /// `timeline` is pooled again: its layout and walked count, if it was
    /// lingering.
    pub fn take(&mut self, timeline: TimelineId) -> Option<(Layout, u32)> {
        self.held.remove(&timeline).map(|l| (l.layout, l.walked))
    }

    /// The page keys the lingering pieces hold on the model.
    pub fn keys(&self) -> impl Iterator<Item = PieceKey> + '_ {
        self.held
            .values()
            .flat_map(|l| l.layout.values().flatten().map(|(_, key)| *key))
    }

    /// How many pieces linger.
    pub fn len(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn layout(seed: &[u8]) -> Layout {
        HashMap::from([(TurnIndex(0), vec![(32, PieceKey::of(seed))])])
    }

    /// **A piece that left comes back as it was within its refreshes**, and
    /// not after: past [`LINGER_REFRESHES`] its pages are let go.
    #[test]
    fn a_piece_comes_back_whole_within_its_refreshes() {
        let mut l = PoolLinger::default();
        l.linger(tl(1), layout(b"a"), 5);
        for _ in 0..LINGER_REFRESHES {
            l.age();
        }
        let (back, walked) = l.take(tl(1)).expect("still lingering");
        assert_eq!(walked, 5);
        assert_eq!(back, layout(b"a"));
        l.linger(tl(2), layout(b"b"), 1);
        for _ in 0..=LINGER_REFRESHES {
            l.age();
        }
        assert!(l.take(tl(2)).is_none(), "past its refreshes");
        assert_eq!(l.len(), 0);
    }

    /// The lingering pieces' keys are what the model keeps for them; past
    /// [`MAX_LINGERING`] the one that has lingered longest goes first.
    #[test]
    fn the_lingering_are_bounded_oldest_out() {
        let mut l = PoolLinger::default();
        l.linger(tl(1), layout(b"first"), 1);
        l.age();
        for n in 2..=(MAX_LINGERING as u64 + 1) {
            l.linger(tl(n), layout(&n.to_le_bytes()), 1);
        }
        assert_eq!(l.len(), MAX_LINGERING);
        assert!(l.take(tl(1)).is_none(), "the oldest went");
        let keys: Vec<PieceKey> = l.keys().collect();
        assert_eq!(keys.len(), MAX_LINGERING, "pieces 2 to 49, a key each");
        assert!(!keys.contains(&PieceKey::of(b"first")));
    }
}
