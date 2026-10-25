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

use super::{
    turn_stream_id, ResidenceIndex, StoredSequence, Substrate, TimelineId, TokenBuffer, TurnIndex,
    TurnLayout,
};

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

    /// Whether every turn of `timeline` holds its content: no gap — a turn
    /// still to arrive — and no other empty turn but a summary anchor, which
    /// holds none by design. A retired turn is emptied in place, as a reload
    /// leaves a turn whose records it could not use, and the projection skips
    /// every empty turn — so a conversation that is not whole cannot be placed
    /// as the whole it was read as. `false` for a timeline the substrate does
    /// not hold.
    pub fn holds_whole(&self, timeline: TimelineId) -> bool {
        self.timelines.get(&timeline).is_some_and(|tl| {
            tl.gaps.is_empty()
                && tl.turns.iter().all(|(idx, turn)| {
                    turn.content.token_count > 0
                        || tl.tree_meta.get(idx).is_some_and(|m| m.kind.is_summary())
                })
        })
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

    /// Put `turn` at `index` of `timeline` as a reload reads it from the log,
    /// whatever the timeline holds already: a reload runs on a live substrate
    /// as well as on an empty one.
    ///
    /// - A gap there, or no turn yet, is [`Self::place_turn`].
    /// - **The same turn there stays where it is**, its residence and the hot
    ///   and warm K/V in it untouched, and takes the cold references the log
    ///   now holds for it if it has none: a turn sealed before the reload and
    ///   written since is the case that needs them.
    /// - **A different turn there replaces it** — a turn ranked below one this
    ///   member held moved it up (§6.6). It takes a fresh residence: the old
    ///   one holds the displaced turn's K/V, and a live slot may still attend
    ///   it, so it is never rewritten under one. Its hot and warm copies are
    ///   dropped now when no working set pins it, and otherwise left to the
    ///   tiers' own eviction once the slot lets it go.
    /// - A log that holds no tokens for the turn is behind the live turn, not
    ///   different from it — a turn sealed moments before the reload whose
    ///   records are still queued — and the live turn stays.
    ///
    /// Restored by appending instead, every turn a timeline held came back a
    /// second time above the first, at indices no chunk is keyed by (`.5`,
    /// 2026-10-20: a rejoin's reload, then 70 minutes of `cold-load found no
    /// chunks` on every projection that selected them).
    pub fn reload_turn(&mut self, timeline: TimelineId, index: TurnIndex, turn: RestoredTurn) {
        if !self.holds_turn(timeline, index) {
            self.place_turn(timeline, index, turn);
            return;
        }
        let entry = self
            .timelines
            .get(&timeline)
            .and_then(|tl| tl.turns.get(&index))
            .expect("a held turn has an entry");
        let same = turn.token_ids.is_empty()
            || (entry.content.token_ids[..] == turn.token_ids[..]
                && entry.block_range == (turn.block_start, turn.block_end));
        let residence = if same {
            entry.content.residence
        } else {
            self.replace_turn_content(timeline, index, &turn)
        };
        if self.residence[residence.0].cold.is_none() {
            if let Some(cold) = turn.cold.filter(|c| !c.is_empty()) {
                if self.residence[residence.0].byte_size == 0 {
                    self.residence[residence.0].byte_size = cold
                        .iter()
                        .flat_map(|s| s.chunks.iter())
                        .map(|c| c.record_len)
                        .sum();
                }
                self.install_cold(residence, cold);
            }
        }
        self.gallery_epoch += 1;
    }

    /// Give the turn at `index` of `timeline` `turn`'s content in a fresh
    /// residence, and release the displaced one's hot and warm copies unless a
    /// working set pins it ([`Self::reload_turn`]). Returns the new residence.
    fn replace_turn_content(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        turn: &RestoredTurn,
    ) -> ResidenceIndex {
        let compression = self.timeline_compression.get(&timeline).copied();
        let fresh = self.alloc_residence(turn_stream_id(timeline.raw(), index.0), compression);
        let entry = self
            .timelines
            .get_mut(&timeline)
            .and_then(|tl| tl.turns.get_mut(&index))
            .expect("a held turn has an entry");
        let displaced = std::mem::replace(&mut entry.content.residence, fresh);
        let previous = entry.content.token_count;
        entry.block_range = (turn.block_start, turn.block_end);
        entry.content.layout = turn.layout.clone();
        entry.content.token_ids = turn.token_ids.clone();
        entry.content.token_count = turn.token_count;
        let total = self.timeline_token_totals.entry(timeline).or_default();
        *total = total.saturating_sub(previous) + turn.token_count;
        if !self.working_set_pins.contains(&displaced) {
            let slot = &mut self.residence[displaced.0];
            slot.hot = None;
            slot.warm = None;
            Self::remove_from_lru(&mut self.hot_lru, displaced);
            Self::remove_from_lru(&mut self.warm_lru, displaced);
        }
        fresh
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

    /// Mark `tl`'s turn `i` as resident hot and warm, as a slot borrowing it
    /// finds it; return its residence.
    fn resident(sub: &mut Substrate, tl: TimelineId, i: u32) -> ResidenceIndex {
        let r = sub.turn_residence(tl, TurnIndex(i)).unwrap();
        sub.residence[r.0].hot = Some(Vec::new());
        sub.residence[r.0].warm = Some(Vec::new());
        sub.hot_lru.push_front(r);
        sub.warm_lru.push_front(r);
        r
    }

    /// **The same turn, reloaded, stays where a live slot reads it**: the
    /// timeline keeps its count, and the turn its residence and the hot and
    /// warm K/V in it.
    #[test]
    fn a_reloaded_turn_keeps_its_residence_and_its_kv() {
        let (mut sub, tl) = timeline();
        sub.place_turn(tl, TurnIndex(0), turn(&[1, 2]));
        sub.place_turn(tl, TurnIndex(1), turn(&[3]));
        let r = resident(&mut sub, tl, 1);

        sub.reload_turn(tl, TurnIndex(0), turn(&[1, 2]));
        sub.reload_turn(tl, TurnIndex(1), turn(&[3]));
        assert_eq!(tokens(&sub, tl), vec![vec![1, 2], vec![3]]);
        assert_eq!(sub.turn_residence(tl, TurnIndex(1)), Some(r));
        assert!(sub.residence[r.0].hot.is_some() && sub.residence[r.0].warm.is_some());
        assert_eq!(sub.timeline_token_total(tl), 3);
    }

    /// A log that holds no tokens for a live turn is behind it: the live turn
    /// stays as it is.
    #[test]
    fn a_log_behind_the_live_turn_leaves_it() {
        let (mut sub, tl) = timeline();
        sub.place_turn(tl, TurnIndex(0), turn(&[4, 5]));
        let r = resident(&mut sub, tl, 0);
        sub.reload_turn(tl, TurnIndex(0), turn(&[]));
        assert_eq!(tokens(&sub, tl), vec![vec![4, 5]]);
        assert_eq!(sub.turn_residence(tl, TurnIndex(0)), Some(r));
        assert!(sub.residence[r.0].hot.is_some());
    }

    /// **A turn moved by a rerank takes a fresh residence**; the displaced
    /// turn's K/V is released when no working set holds it, and left for the
    /// tiers' eviction while one does.
    #[test]
    fn a_reranked_turn_takes_a_fresh_residence() {
        let (mut sub, tl) = timeline();
        sub.place_turn(tl, TurnIndex(0), turn(&[1]));
        sub.place_turn(tl, TurnIndex(1), turn(&[2, 2]));
        let free = resident(&mut sub, tl, 0);
        let held = resident(&mut sub, tl, 1);
        sub.working_set_pins.insert(held);

        sub.reload_turn(tl, TurnIndex(0), turn(&[9]));
        sub.reload_turn(tl, TurnIndex(1), turn(&[8, 8, 8]));
        assert_eq!(tokens(&sub, tl), vec![vec![9], vec![8, 8, 8]]);
        assert_eq!(sub.timeline_token_total(tl), 4);
        let (now_0, now_1) = (
            sub.turn_residence(tl, TurnIndex(0)).unwrap(),
            sub.turn_residence(tl, TurnIndex(1)).unwrap(),
        );
        assert!(now_0 != free && now_1 != held);
        assert!(sub.residence[now_0.0].hot.is_none() && sub.residence[now_1.0].hot.is_none());
        assert!(
            sub.residence[free.0].hot.is_none() && sub.residence[free.0].warm.is_none(),
            "an unpinned displaced turn's K/V is released"
        );
        assert!(
            sub.residence[held.0].hot.is_some(),
            "a pinned one stays for the slot attending it"
        );
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
