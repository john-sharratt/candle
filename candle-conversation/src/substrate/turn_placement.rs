//! Where a restored turn goes in its timeline, and the placeholders that hold
//! a position until its turn arrives (`docs/replicated_substrate.md` §6.5,
//! §6.6).
//!
//! A timeline's turns are dense: the turn at index *N* is the *N*-th entry, and
//! every caller derives the turn's stream id from that index. A position that
//! no turn here fills yet — a declaration lost in a crash, or another member's
//! turn whose records are still arriving — holds an empty placeholder, a
//! **gap**, so the turns above it stay at the indices their streams resolve
//! by. A turn that completes later takes its gap in place.
//!
//! A turn that arrives ranked below a turn this member already holds moves
//! that turn up a position. The indices of a live timeline must not move under
//! the hot path, so the member reloads (§6.6): [`Substrate::take_reload_due`]
//! says when.

use super::{StoredSequence, Substrate, TimelineId, TokenBuffer, TurnIndex, TurnLayout};

/// A turn's content as its records hold it — what a reload or an adoption
/// restores.
#[derive(Debug, Clone)]
pub struct RestoredTurn {
    pub layout: TurnLayout,
    pub token_ids: TokenBuffer,
    pub token_count: usize,
    /// The turn's K/V as cold references to its `Chunk` records, or `None`
    /// for a turn that holds none.
    pub cold: Option<Vec<StoredSequence>>,
    pub block_start: u64,
    pub block_end: u64,
}

impl Substrate {
    /// Restore an empty placeholder at `timeline`'s next index and record it as
    /// a gap a turn can take later. The timeline must be registered.
    pub fn restore_gap_turn(&mut self, timeline: TimelineId) -> TurnIndex {
        let index = self.restore_turn(
            timeline,
            TurnLayout::new(Vec::new()),
            TokenBuffer::default(),
            0,
            None,
            0,
            0,
        );
        self.timelines
            .get_mut(&timeline)
            .expect("restore_turn registered the timeline")
            .gaps
            .insert(index);
        index
    }

    /// Whether `index` of `timeline` is a gap.
    pub fn is_gap_turn(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.timelines
            .get(&timeline)
            .is_some_and(|tl| tl.gaps.contains(&index))
    }

    /// Whether a turn — anything but a gap — holds `index` of `timeline`.
    pub fn holds_turn(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.timelines
            .get(&timeline)
            .is_some_and(|tl| tl.turns.contains_key(&index) && !tl.gaps.contains(&index))
    }

    /// Put `turn` at `index` of `timeline`, which the caller has registered:
    /// into the gap there, or at the end with a gap at every position below it
    /// that holds nothing. `false`, and nothing changed, when a turn already
    /// holds `index`.
    pub fn place_turn(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        turn: RestoredTurn,
    ) -> bool {
        let count = self.turn_count(timeline);
        if index.0 < count {
            if !self.is_gap_turn(timeline, index) {
                return false;
            }
            self.fill_gap(timeline, index, turn);
            return true;
        }
        for _ in count..index.0 {
            self.restore_gap_turn(timeline);
        }
        self.restore_turn(
            timeline,
            turn.layout,
            turn.token_ids,
            turn.token_count,
            turn.cold,
            turn.block_start,
            turn.block_end,
        );
        self.gallery_epoch += 1;
        true
    }

    /// Install `turn` in the gap at `index` — the placeholder's entry and
    /// residence become the turn's.
    fn fill_gap(&mut self, timeline: TimelineId, index: TurnIndex, turn: RestoredTurn) {
        let tl = self
            .timelines
            .get_mut(&timeline)
            .expect("a gap is in a registered timeline");
        tl.gaps.remove(&index);
        let entry = tl
            .turns
            .get_mut(&index)
            .expect("a gap holds a placeholder entry");
        entry.block_range = (turn.block_start, turn.block_end);
        entry.content.layout = turn.layout;
        entry.content.token_ids = turn.token_ids;
        entry.content.token_count = turn.token_count;
        let residence = entry.content.residence;
        *self.timeline_token_totals.entry(timeline).or_default() += turn.token_count;
        if let Some(cold) = turn.cold.filter(|c| !c.is_empty()) {
            self.residence[residence.0].byte_size = cold
                .iter()
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.record_len)
                .sum();
            self.install_cold(residence, cold);
        }
        self.gallery_epoch += 1;
    }

    /// Whether a declaration moved a turn this member holds since the last
    /// call — the timeline's indices changed under it and the member must
    /// reload (§6.6). Clears the flag.
    pub fn take_reload_due(&mut self) -> bool {
        std::mem::take(&mut self.reload_due)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId};

    fn timeline() -> (Substrate, TimelineId) {
        let mut sub = Substrate::new();
        let tl = TimelineId::for_test(7);
        sub.register_timeline(
            tl,
            LayerId::from_raw(1).unwrap(),
            GroupId::from_raw(1).unwrap(),
        );
        (sub, tl)
    }

    fn turn(tokens: &[u32]) -> RestoredTurn {
        RestoredTurn {
            layout: TurnLayout::new(Vec::new()),
            token_ids: TokenBuffer::from(tokens.to_vec()),
            token_count: tokens.len(),
            cold: None,
            block_start: 0,
            block_end: 0,
        }
    }

    fn tokens(sub: &Substrate, tl: TimelineId) -> Vec<Vec<u32>> {
        (0..sub.turn_count(tl))
            .map(|i| sub.token_ids_of(tl, TurnIndex(i)))
            .collect()
    }

    /// A turn placed above the end leaves a gap at every position below it,
    /// and a turn placed later fills its gap without moving anything.
    #[test]
    fn a_turn_fills_its_gap_in_place() {
        let (mut sub, tl) = timeline();
        assert!(sub.place_turn(tl, TurnIndex(2), turn(&[5, 6])));
        assert_eq!(tokens(&sub, tl), vec![vec![], vec![], vec![5, 6]]);
        assert!(sub.is_gap_turn(tl, TurnIndex(0)) && sub.is_gap_turn(tl, TurnIndex(1)));
        assert!(!sub.holds_turn(tl, TurnIndex(1)) && sub.holds_turn(tl, TurnIndex(2)));

        assert!(sub.place_turn(tl, TurnIndex(1), turn(&[3])));
        assert_eq!(tokens(&sub, tl), vec![vec![], vec![3], vec![5, 6]]);
        assert!(sub.holds_turn(tl, TurnIndex(1)) && !sub.is_gap_turn(tl, TurnIndex(1)));
        assert_eq!(sub.timeline_token_total(tl), 3);
    }

    /// A position a turn already holds is refused, and nothing changes.
    #[test]
    fn a_held_position_is_refused() {
        let (mut sub, tl) = timeline();
        assert!(sub.place_turn(tl, TurnIndex(0), turn(&[1])));
        assert!(!sub.place_turn(tl, TurnIndex(0), turn(&[9])));
        assert_eq!(tokens(&sub, tl), vec![vec![1]]);
    }
}
