//! The pooled rounds a hit declaration names after the placed ones
//! (`docs/hit_driven_projection.md` §6.3, §7.2).
//!
//! The pool holds each pooled conversation's turns' index pages at the
//! sampled layers. Every round of a pooled conversation the slot did not place
//! is unloaded, and the shadow scan scores the stride's kept queries against
//! its rows. A round is one or two records; a record's rows sit in whichever of
//! its turn's pages cover its positions, so a round is declared as one entry
//! per page it touches, each under the round's key — the hit density sums a
//! key's entries.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use candle_transformers::models::hit_layers::PooledRound;
use candle_transformers::models::piece_key::PieceKey;

use crate::projection::rounds::{records_of, round_count, RoundKey};
use crate::projection::{TimelineId, TurnIndex};
use crate::turn_layout::TurnRecord;

/// What a slot's pool holds: per pooled conversation, per turn, its pages in
/// token order — each page's token width and key.
pub(super) type PoolPages = HashMap<TimelineId, HashMap<TurnIndex, Vec<(usize, PieceKey)>>>;

/// Every round of the conversations `pool` holds that `placed` does not name,
/// as entries for the model's declaration and the keys they count under, in
/// one deterministic order. `exchanges` names a conversation's exchanges;
/// `record_span` a record's positions within its turn.
pub(super) fn pooled_rounds(
    pool: &PoolPages,
    placed: &HashSet<RoundKey>,
    exchanges: impl Fn(TimelineId) -> Vec<Vec<TurnIndex>>,
    record_span: impl Fn(TimelineId, TurnIndex, TurnRecord) -> Range<usize>,
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
            for r in 0..round_count(turns.len()) {
                let key = RoundKey {
                    timeline: tl,
                    head,
                    index: r as u32,
                };
                if placed.contains(&key) {
                    continue;
                }
                for (turn, record) in records_of(&turns, r) {
                    let Some(pages) = turns_pages.get(&turn) else {
                        continue;
                    };
                    let span = record_span(tl, turn, record);
                    let mut at = 0usize;
                    for &(width, piece) in pages {
                        let (lo, hi) = (span.start.max(at), span.end.min(at + width));
                        if lo < hi {
                            keys.push(key);
                            rounds.push(PooledRound {
                                piece,
                                positions: lo - at..hi - at,
                            });
                        }
                        at += width;
                    }
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

    /// A pooled conversation of one exchange of two turns, each split at its
    /// record edge into two pages but turn 1, one page: its rounds are the
    /// request (turn 0's user record), the tool round (turn 0's call, turn 1's
    /// result) and the answer (turn 1's assistant record). A placed round is
    /// left out, and a record spanning two pages is declared once per page,
    /// under its round's key, at positions local to each page.
    #[test]
    fn a_pooled_conversations_unplaced_rounds_are_declared_per_page() {
        let a = PieceKey::of(b"turn 0 user");
        let b = PieceKey::of(b"turn 0 assistant");
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
        // Turn 0: user 0..10, assistant 10..40. Turn 1: user 0..12, assistant 12..50.
        let span = |_: TimelineId, t: TurnIndex, r: TurnRecord| match (t.0, r) {
            (0, TurnRecord::User) => 0..10,
            (0, TurnRecord::Assistant) => 10..40,
            (1, TurnRecord::User) => 0..12,
            _ => 12..50,
        };
        let key = |index| RoundKey {
            timeline: tl(5),
            head: TurnIndex(0),
            index,
        };
        let (keys, rounds) = pooled_rounds(&pool, &HashSet::new(), exchanges, span);
        let round = |piece, positions| PooledRound { piece, positions };
        assert_eq!(keys, vec![key(0), key(1), key(1), key(2)]);
        assert_eq!(
            rounds,
            vec![
                round(a, 0..10),
                round(b, 0..30),
                round(c, 0..12),
                round(c, 12..50)
            ]
        );

        let placed = HashSet::from([key(1)]);
        let (keys, rounds) = pooled_rounds(&pool, &placed, exchanges, span);
        assert_eq!(keys, vec![key(0), key(2)]);
        assert_eq!(rounds, vec![round(a, 0..10), round(c, 12..50)]);
    }
}
