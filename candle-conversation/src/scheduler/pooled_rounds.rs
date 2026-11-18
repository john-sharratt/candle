//! The pooled rounds a hit declaration names after the placed ones
//! (`docs/hit_driven_projection.md` §6.3, §7.2).
//!
//! The pool holds each pooled conversation's turns' index pages at the
//! sampled layers. Every round of a pooled conversation the slot did not place
//! is unloaded, and the shadow scan scores the stride's kept queries against
//! its rows. A round is one whole turn, so it is declared as one entry per page
//! of that turn, each under the round's key — the hit density sums a key's
//! entries.

use std::collections::{HashMap, HashSet};

use candle_transformers::models::hit_layers::PooledRound;
use candle_transformers::models::piece_key::PieceKey;

use crate::projection::rounds::RoundKey;
use crate::projection::{TimelineId, TurnIndex};

/// What a slot's pool holds: per pooled conversation, per turn, its pages in
/// token order — each page's token width and key.
pub(super) type PoolPages = HashMap<TimelineId, HashMap<TurnIndex, Vec<(usize, PieceKey)>>>;

/// Every round of the conversations `pool` holds that `placed` does not name,
/// as entries for the model's declaration and the keys they count under, in
/// one deterministic order. `exchanges` names a conversation's exchanges.
pub(super) fn pooled_rounds(
    pool: &PoolPages,
    placed: &HashSet<RoundKey>,
    exchanges: impl Fn(TimelineId) -> Vec<Vec<TurnIndex>>,
) -> (Vec<RoundKey>, Vec<PooledRound>) {
    let mut timelines: Vec<&TimelineId> = pool.keys().collect();
    timelines.sort();
    let mut keys = Vec::new();
    let mut rounds = Vec::new();
    for &tl in timelines {
        let turns_pages = &pool[&tl];
        for turns in exchanges(tl) {
            let Some(&head) = turns.first() else {
                continue;
            };
            for (r, turn) in turns.iter().enumerate() {
                let key = RoundKey {
                    timeline: tl,
                    head,
                    index: r as u32,
                };
                if placed.contains(&key) {
                    continue;
                }
                let Some(pages) = turns_pages.get(turn) else {
                    continue;
                };
                for &(width, piece) in pages {
                    keys.push(key);
                    rounds.push(PooledRound {
                        piece,
                        positions: 0..width,
                    });
                }
            }
        }
    }
    (keys, rounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// A pooled conversation of one exchange of two turns, turn 0 held as two
    /// pages and turn 1 as one: each turn is a round, declared once per page of
    /// it, whole. A placed round is left out.
    #[test]
    fn a_pooled_conversations_unplaced_turns_are_declared_per_page() {
        let a = PieceKey::of(b"turn 0 page 0");
        let b = PieceKey::of(b"turn 0 page 1");
        let c = PieceKey::of(b"turn 1");
        let mut pool: PoolPages = HashMap::new();
        pool.insert(
            tl(5),
            HashMap::from([
                (TurnIndex(0), vec![(10, a), (30, b)]),
                (TurnIndex(1), vec![(50, c)]),
            ]),
        );
        let exchanges = |_: TimelineId| vec![vec![TurnIndex(0), TurnIndex(1)]];
        let key = |index| RoundKey {
            timeline: tl(5),
            head: TurnIndex(0),
            index,
        };
        let round = |piece, positions| PooledRound { piece, positions };

        let (keys, rounds) = pooled_rounds(&pool, &HashSet::new(), exchanges);
        assert_eq!(keys, vec![key(0), key(0), key(1)]);
        assert_eq!(
            rounds,
            vec![round(a, 0..10), round(b, 0..30), round(c, 0..50)]
        );

        let placed = HashSet::from([key(0)]);
        let (keys, rounds) = pooled_rounds(&pool, &placed, exchanges);
        assert_eq!(keys, vec![key(1)]);
        assert_eq!(rounds, vec![round(c, 0..50)]);
    }
}
