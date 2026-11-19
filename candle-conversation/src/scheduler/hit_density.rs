//! What the model's own indexer read of each placed round over a stride
//! (`docs/hit_driven_projection.md` §6.4, §7.1).
//!
//! The model counts hits per declared piece — every placed round. A read
//! arrives in parts: the reprojection's own, and whatever an assembly in
//! between read back before it declared new pieces. A working set judges
//! whole conversations, so each conversation's density is its rounds' hits
//! over their blocks per counted query, across every part; a round's own
//! density is the same over its own blocks.

use std::collections::HashMap;
use std::hash::Hash;

use candle_transformers::models::hit_layers::PieceHits;

use crate::projection::rounds::RoundKey;
use crate::projection::{TimelineId, TurnIndex, TurnKey};

/// One part of a stride's read: the keys the pieces were declared under, and
/// what the model counted for them.
pub(super) type HitPart = (Vec<RoundKey>, PieceHits);

/// What a stride's read held, for the reprojection's observe split: whether
/// the model's own indexer is reading what the projection placed, and whether
/// the shadow scan finds anything in the pool.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct HitSummary {
    /// Parts read: the reprojection's own and any an assembly read before it.
    pub parts: usize,
    /// Query rows counted against loaded rounds, over every part.
    pub queries: u64,
    /// Kept queries the shadow scan scored against pooled rounds.
    pub shadow_queries: u64,
    /// Loaded rounds declared, and how many of them any query hit.
    pub loaded: usize,
    pub loaded_hit: usize,
    /// Pooled rounds declared, and how many of them any shadow query hit.
    pub pooled: usize,
    pub pooled_hit: usize,
    /// Hits counted on loaded and on pooled rounds.
    pub loaded_hits: u64,
    pub pooled_hits: u64,
}

/// [`HitSummary`] over `parts`.
pub(super) fn hit_summary(parts: &[HitPart]) -> HitSummary {
    let mut s = HitSummary {
        parts: parts.len(),
        ..HitSummary::default()
    };
    for (keys, hits) in parts {
        s.queries += hits.queries;
        s.shadow_queries += hits.shadow_queries;
        for i in 0..keys.len() {
            let h = hits.hits.get(i).copied().unwrap_or(0) as u64;
            if i < hits.loaded {
                s.loaded += 1;
                s.loaded_hit += usize::from(h > 0);
                s.loaded_hits += h;
            } else {
                s.pooled += 1;
                s.pooled_hit += usize::from(h > 0);
                s.pooled_hits += h;
            }
        }
    }
    s
}

/// Each conversation's hit density across `parts`.
pub(super) fn hit_density(parts: &[HitPart]) -> HashMap<TimelineId, f32> {
    density_by(parts, |k| k.timeline)
}

/// Each placed round's hit density across `parts`.
pub(super) fn round_density(parts: &[HitPart]) -> HashMap<RoundKey, f32> {
    density_by(parts, |k| *k)
}

/// Each turn's lift from its round's `lifts`: a round is one whole turn, so
/// round `index` of an exchange lifts that exchange's `index`-th turn.
/// `exchange` names the turns of the exchange a round's head opens.
pub(super) fn turn_lifts(
    lifts: impl IntoIterator<Item = (RoundKey, f32)>,
    mut exchange: impl FnMut(TimelineId, TurnIndex) -> Option<Vec<TurnIndex>>,
) -> HashMap<TurnKey, f32> {
    let mut out: HashMap<TurnKey, f32> = HashMap::new();
    for (key, lift) in lifts {
        let Some(turns) = exchange(key.timeline, key.head) else {
            continue;
        };
        if let Some(&turn) = turns.get(key.index as usize) {
            let at = out.entry(TurnKey::new(key.timeline, turn)).or_insert(lift);
            *at = at.max(lift);
        }
    }
    out
}

/// Hits over blocks × queries per `group(key)`, summed across parts — each
/// declared entry over the queries that could have hit it, the selection's
/// for a loaded round and the shadow scan's for a pooled one (§6.4). A key
/// declared more than once — a pooled round over several pages — sums its
/// entries. A group with no blocks counted is left out.
fn density_by<G: Eq + Hash>(parts: &[HitPart], group: impl Fn(&RoundKey) -> G) -> HashMap<G, f32> {
    let mut sums: HashMap<G, (f64, f64)> = HashMap::new();
    for (keys, hits) in parts {
        for (i, key) in keys.iter().enumerate() {
            let h = hits.hits.get(i).copied().unwrap_or(0) as f64;
            let b = hits.blocks.get(i).copied().unwrap_or(0) as f64;
            let e = sums.entry(group(key)).or_insert((0.0, 0.0));
            e.0 += h;
            e.1 += b * hits.queries_of(i) as f64;
        }
    }
    sums.into_iter()
        .filter(|&(_, (_, denom))| denom > 0.0)
        .map(|(g, (h, denom))| (g, (h / denom) as f32))
        .collect()
}

/// Each piece's strongest turn lift and how many of its turns are lifted, the
/// `n` most lifted pieces first, as one line — what the hits added to each
/// piece's scores this boundary, beside the scores the fill ranked
/// (`projection fill pieces`).
pub(super) fn piece_lifts_line(lifts: &HashMap<TurnKey, f32>, n: usize) -> String {
    let mut by_piece: HashMap<TimelineId, (f32, usize)> = HashMap::new();
    for (key, &lift) in lifts {
        let at = by_piece.entry(key.timeline).or_insert((0.0, 0));
        at.0 = at.0.max(lift);
        at.1 += 1;
    }
    let mut rows: Vec<(TimelineId, f32, usize)> = by_piece
        .into_iter()
        .map(|(tl, (top, count))| (tl, top, count))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
    rows.iter()
        .take(n)
        .map(|(tl, top, count)| format!("{}:top={top:.3} turns={count}", tl.raw()))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn key(timeline: u64, head: u32, index: u32) -> RoundKey {
        RoundKey {
            timeline: tl(timeline),
            head: TurnIndex(head),
            index,
        }
    }

    /// **A piece's line is its strongest turn lift and its lifted turns**, the
    /// most lifted pieces first, ties by timeline.
    #[test]
    fn piece_lifts_are_reported_strongest_first() {
        let k = |t: u64, i: u32| TurnKey::new(tl(t), TurnIndex(i));
        let lifts = HashMap::from([
            (k(7, 0), 1.25),
            (k(7, 3), 1.5),
            (k(9, 1), 2.0),
            (k(4, 0), 1.5),
        ]);
        assert_eq!(
            piece_lifts_line(&lifts, 2),
            "9:top=2.000 turns=1 | 4:top=1.500 turns=1"
        );
        assert_eq!(
            piece_lifts_line(&lifts, 3),
            "9:top=2.000 turns=1 | 4:top=1.500 turns=1 | 7:top=1.500 turns=2"
        );
    }

    /// A conversation's rounds pool their hits and blocks: two rounds of one
    /// file, 6 + 2 hits over 10 + 30 blocks across 4 queries, is 8 / 160.
    #[test]
    fn a_conversations_rounds_pool_their_hits() {
        let part = (
            vec![key(7, 0, 0), key(7, 0, 1), key(8, 0, 0)],
            PieceHits {
                hits: vec![6, 2, 0],
                blocks: vec![10, 30, 5],
                loaded: 3,
                queries: 4,
                shadow_queries: 0,
            },
        );
        let d = hit_density(std::slice::from_ref(&part));
        assert!((d[&tl(7)] - 0.05).abs() < 1e-7);
        assert_eq!(d[&tl(8)], 0.0);
        let r = round_density(std::slice::from_ref(&part));
        assert!((r[&key(7, 0, 0)] - 0.15).abs() < 1e-7);
        let none = (
            part.0.clone(),
            PieceHits {
                queries: 0,
                ..part.1.clone()
            },
        );
        assert!(hit_density(&[none]).is_empty());
    }

    /// **Parts read at different times pool by their own queries**: a round
    /// read 4 times over 10 blocks across 2 queries, then once more over the
    /// same blocks across 3, is 5 hits over 50 block-queries.
    #[test]
    fn parts_pool_by_their_own_queries() {
        let first = (
            vec![key(7, 0, 0)],
            PieceHits {
                hits: vec![4],
                blocks: vec![10],
                loaded: 1,
                queries: 2,
                shadow_queries: 0,
            },
        );
        let second = (
            vec![key(7, 0, 0)],
            PieceHits {
                hits: vec![1],
                blocks: vec![10],
                loaded: 1,
                queries: 3,
                shadow_queries: 0,
            },
        );
        let r = round_density(&[first, second]);
        assert!((r[&key(7, 0, 0)] - 0.1).abs() < 1e-7);
    }

    /// **A loaded and an unloaded round hit by every query that could have
    /// hit them score the same density, whatever the stride's row counts**
    /// (`docs/hit_driven_projection.md` §13.2): a loaded round of 8 blocks hit
    /// at every block by each of 300 prefill rows, and a pooled round of 12
    /// blocks — declared over two pages — hit at every block by each of the 64
    /// rows the shadow scan kept, both read 1.0.
    #[test]
    fn a_loaded_and_a_pooled_round_read_alike() {
        let part = (
            vec![key(7, 0, 0), key(9, 0, 1), key(9, 0, 1)],
            PieceHits {
                hits: vec![8 * 300, 5 * 64, 7 * 64],
                blocks: vec![8, 5, 7],
                loaded: 1,
                queries: 300,
                shadow_queries: 64,
            },
        );
        let r = round_density(std::slice::from_ref(&part));
        assert_eq!(r[&key(7, 0, 0)], 1.0);
        assert_eq!(r[&key(9, 0, 1)], 1.0);
        let c = hit_density(std::slice::from_ref(&part));
        assert_eq!((c[&tl(7)], c[&tl(9)]), (1.0, 1.0));
    }

    /// A round lifts its own turn: in an exchange of turns 4, 5, round 0 lifts
    /// turn 4 and round 1 lifts turn 5; a round past the exchange, or of no
    /// known exchange, lifts nothing.
    #[test]
    fn a_round_lifts_its_own_turn() {
        let exchange = |_: TimelineId, head: TurnIndex| {
            (head == TurnIndex(4)).then(|| vec![TurnIndex(4), TurnIndex(5)])
        };
        let lifts = turn_lifts(
            [
                (key(7, 4, 0), 1.5),
                (key(7, 4, 1), 2.0),
                (key(7, 4, 2), 9.0),
                (key(7, 9, 0), 3.0),
            ],
            exchange,
        );
        assert_eq!(lifts[&TurnKey::new(tl(7), TurnIndex(4))], 1.5);
        assert_eq!(lifts[&TurnKey::new(tl(7), TurnIndex(5))], 2.0);
        assert_eq!(lifts.len(), 2);
    }

    /// The summary counts each part's queries, and splits the declared rounds
    /// at each part's `loaded`: hits on the first are the selection's, on the
    /// rest the shadow scan's.
    #[test]
    fn the_summary_splits_loaded_and_pooled_rounds() {
        let first = (
            vec![key(7, 0, 0), key(7, 0, 1), key(8, 0, 0), key(9, 0, 0)],
            PieceHits {
                hits: vec![6, 0, 3, 0],
                blocks: vec![10, 30, 5, 5],
                loaded: 2,
                queries: 4,
                shadow_queries: 64,
            },
        );
        let second = (
            vec![key(7, 0, 0)],
            PieceHits {
                hits: vec![1],
                blocks: vec![10],
                loaded: 1,
                queries: 3,
                shadow_queries: 0,
            },
        );
        assert_eq!(
            hit_summary(&[first, second]),
            HitSummary {
                parts: 2,
                queries: 7,
                shadow_queries: 64,
                loaded: 3,
                loaded_hit: 2,
                pooled: 2,
                pooled_hit: 1,
                loaded_hits: 7,
                pooled_hits: 3,
            }
        );
        assert_eq!(hit_summary(&[]), HitSummary::default());
    }
}
