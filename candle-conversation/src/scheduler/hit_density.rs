//! What the model's own indexer read of each placed piece over a stride
//! (`docs/hit_driven_projection.md` §6.4, §7.1).
//!
//! The model counts hits per declared piece — every placed turn. A read
//! arrives in parts: the reprojection's own, and whatever an assembly in
//! between read back before it declared new pieces. A working set judges
//! whole conversations, so each conversation's density is its turns' hits
//! over their blocks per counted query, across every part; a turn's own
//! density is the same over its own blocks.

use std::collections::HashMap;
use std::hash::Hash;

use candle_transformers::models::hit_layers::PieceHits;

use crate::projection::{TimelineId, TurnKey};

/// One part of a stride's read: the keys the pieces were declared under, and
/// what the model counted for them.
pub(super) type HitPart = (Vec<TurnKey>, PieceHits);

/// Each conversation's hit density across `parts`.
pub(super) fn hit_density(parts: &[HitPart]) -> HashMap<TimelineId, f32> {
    density_by(parts, |k| k.timeline)
}

/// Each placed turn's hit density across `parts`.
pub(super) fn turn_density(parts: &[HitPart]) -> HashMap<TurnKey, f32> {
    density_by(parts, |k| *k)
}

/// Hits over blocks × queries per `group(key)`, summed across parts; a group
/// with no blocks counted is left out.
fn density_by<G: Eq + Hash>(parts: &[HitPart], group: impl Fn(&TurnKey) -> G) -> HashMap<G, f32> {
    let mut sums: HashMap<G, (f64, f64)> = HashMap::new();
    for (keys, hits) in parts {
        for (i, key) in keys.iter().enumerate() {
            let h = hits.hits.get(i).copied().unwrap_or(0) as f64;
            let b = hits.blocks.get(i).copied().unwrap_or(0) as f64;
            let e = sums.entry(group(key)).or_insert((0.0, 0.0));
            e.0 += h;
            e.1 += b * hits.queries as f64;
        }
    }
    sums.into_iter()
        .filter(|&(_, (_, denom))| denom > 0.0)
        .map(|(g, (h, denom))| (g, (h / denom) as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::TurnIndex;

    fn key(timeline: u64, index: u32) -> TurnKey {
        TurnKey::new(TimelineId::from_raw(timeline).unwrap(), TurnIndex(index))
    }

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// A conversation's turns pool their hits and blocks: two turns of one
    /// file, 6 + 2 hits over 10 + 30 blocks across 4 queries, is 8 / 160.
    #[test]
    fn a_conversations_turns_pool_their_hits() {
        let part = (
            vec![key(7, 0), key(7, 1), key(8, 0)],
            PieceHits {
                hits: vec![6, 2, 0],
                blocks: vec![10, 30, 5],
                queries: 4,
            },
        );
        let d = hit_density(std::slice::from_ref(&part));
        assert!((d[&tl(7)] - 0.05).abs() < 1e-7);
        assert_eq!(d[&tl(8)], 0.0);
        let t = turn_density(std::slice::from_ref(&part));
        assert!((t[&key(7, 0)] - 0.15).abs() < 1e-7);
        let none = (
            part.0.clone(),
            PieceHits {
                queries: 0,
                ..part.1.clone()
            },
        );
        assert!(hit_density(&[none]).is_empty());
    }

    /// **Parts read at different times pool by their own queries**: a turn
    /// read 4 times over 10 blocks across 2 queries, then once more over the
    /// same blocks across 3, is 5 hits over 50 block-queries.
    #[test]
    fn parts_pool_by_their_own_queries() {
        let first = (
            vec![key(7, 0)],
            PieceHits {
                hits: vec![4],
                blocks: vec![10],
                queries: 2,
            },
        );
        let second = (
            vec![key(7, 0)],
            PieceHits {
                hits: vec![1],
                blocks: vec![10],
                queries: 3,
            },
        );
        let t = turn_density(&[first, second]);
        assert!((t[&key(7, 0)] - 0.1).abs() < 1e-7);
    }
}
